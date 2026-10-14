import sys
import json
import time
from pathlib import Path
import traceback

import pandas as pd
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.utils.paths import find_metaqa_dir, find_project_root
from backend.services.knowledge_graph import load_graph
from backend.services.hop_classifier import load_artifacts, predict_hop
from backend.services.adaptive_pipeline import run_adaptive_pipeline
from backend.services.retrieval import retrieve_for_question
from backend.services.answer_engine import answer_question, parse_hop_plan, _TEMPLATE_MATCHER, _masked_question
from backend.services.entity_detection import detect_entities
from backend.utils.eval_metrics import score_answer_sets
from backend.scripts.inspect_dataset import parse_qa_line

_kg = None
_vectorizer = None
_model = None

def worker_init():
    global _kg, _vectorizer, _model
    _kg = load_graph()
    _vectorizer, _model = load_artifacts()

def main():
    print("Initializing Phase 10 Evaluation...")
    worker_init()
    
    results_dir = find_project_root() / "results" / "large_scale"
    
    # Load gold questions and selected questions log from Phase 8/9
    questions_log_path = results_dir / "evaluation_questions_3000.json"
    if not questions_log_path.exists():
        print(f"Error: Questions log not found at {questions_log_path}")
        sys.exit(1)
        
    with open(questions_log_path, "r", encoding="utf-8") as f:
        selected_log = json.load(f)
        
    metaqa = find_metaqa_dir()
    gold_map = {}
    for hop_val, folder in [(1, "1-hop"), (2, "2-hop"), (3, "3-hop")]:
        path = metaqa / folder / "vanilla" / "qa_test.txt"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    question, gold = parse_qa_line(line.rstrip("\r\n"))
                    # Normalize gold answers during evaluation
                    gold_map[question] = [g.strip() for g in gold if g.strip()]

    # Reconstruct 3000 tasks list
    tasks = []
    for hop_str, q_list in selected_log.items():
        hop_val = int(hop_str)
        for q in q_list:
            if q in gold_map:
                tasks.append((q, gold_map[q], hop_val))
                
    print(f"Loaded {len(tasks)} questions for Phase 10 evaluation.")
    
    records = []
    hop_metrics = {h: {"exact_match": 0, "hit": 0, "triples": 0, "time": 0, "count": 0} for h in [1, 2, 3]}
    combined_metrics = {"exact_match": 0, "hit": 0, "triples": 0, "time": 0, "count": 0}
    
    failures = []
    template_matcher_info_log = []
    
    start_time = time.time()
    
    for i, (question, gold_answers, true_hop) in enumerate(tasks):
        try:
            # Run Phase 10 AdaptKG-RAG (with TF-IDF template-aware relation mapping)
            res = run_adaptive_pipeline(
                question,
                kg=_kg,
                vectorizer=_vectorizer,
                model=_model,
                max_candidates_per_expansion=100,
                max_total_candidates=5000,
                pruning_strategy="relevance_bounded"
            )
            
            # Score against gold answers
            scores = score_answer_sets(res["answer"], gold_answers)
            em = scores["exact_match"]
            hit = scores["hit"]
            triples = res["candidate_triple_count"]
            latency = res["retrieval_time_ms"]
            
            # Log metrics
            combined_metrics["exact_match"] += 1 if em else 0
            combined_metrics["hit"] += 1 if hit else 0
            combined_metrics["triples"] += triples
            combined_metrics["time"] += latency
            combined_metrics["count"] += 1
            
            hop_metrics[true_hop]["exact_match"] += 1 if em else 0
            hop_metrics[true_hop]["hit"] += 1 if hit else 0
            hop_metrics[true_hop]["triples"] += triples
            hop_metrics[true_hop]["time"] += latency
            hop_metrics[true_hop]["count"] += 1
            
            rec = {
                "question": question,
                "true_hop": true_hop,
                "predicted_hop": res["predicted_hop"],
                "confidence": res["confidence"],
                "selected_depth": res["selected_retrieval_depth"],
                "exact_match": em,
                "hit": hit,
                "candidate_triples": triples,
                "retrieval_time_ms": latency,
                "predicted_answers": res["answer"],
                "gold_answers": gold_answers,
            }
            records.append(rec)
            
            # Diagnostic Error Analysis if EM is False
            if not em:
                # Determine precise error category
                det_res = detect_entities(question, _kg)
                seeds = [item["name"] for item in det_res["entities"] if item.get("in_kg")]
                
                if not seeds:
                    category = "Retrieval failure"
                else:
                    # 1. Check unpruned retrieval candidates for gold answer presence
                    unpruned_ret = retrieve_for_question(
                        question, max_hops=true_hop, kg=_kg,
                        max_candidates_per_expansion=None,
                        max_total_candidates=None,
                        pruning_strategy="none"
                    )
                    unpruned_nodes = set()
                    for t in unpruned_ret["triples"]:
                        unpruned_nodes.add(t["subject"])
                        unpruned_nodes.add(t["object"])
                        
                    gold_in_unpruned = any(g in unpruned_nodes for g in gold_answers)
                    
                    if not gold_in_unpruned:
                        category = "Missing gold answer from retrieved candidates"
                    else:
                        # 2. Check pruned candidates for gold answer presence
                        pruned_nodes = set()
                        for t in res.get("ranked_triples", []):
                            pruned_nodes.add(t["subject"])
                            pruned_nodes.add(t["object"])
                        # Or check full retrieved triples
                        pruned_ret = retrieve_for_question(
                            question, max_hops=true_hop, kg=_kg,
                            max_candidates_per_expansion=100,
                            max_total_candidates=5000,
                            pruning_strategy="relevance_bounded"
                        )
                        for t in pruned_ret["triples"]:
                            pruned_nodes.add(t["subject"])
                            pruned_nodes.add(t["object"])
                            
                        gold_in_pruned = any(g in pruned_nodes for g in gold_answers)
                        
                        if not gold_in_pruned:
                            category = "Pruning failure"
                        else:
                            # 3. Check template matching / parsing
                            q_masked = _masked_question(question)
                            plan, match_info = _TEMPLATE_MATCHER.match(q_masked, max_hops=true_hop)
                            if plan is None:
                                category = "Unsupported question template"
                            else:
                                if not res["answer"]:
                                    category = "Answer extraction failure"
                                else:
                                    category = "Wrong relation traversal"
                                    
                failures.append({
                    "question": question,
                    "gold_answers": gold_answers,
                    "predicted_answers": res["answer"],
                    "predicted_hop": res["predicted_hop"],
                    "selected_depth": res["selected_retrieval_depth"],
                    "category": category,
                    "true_hop": true_hop
                })
        except Exception as e:
            print(f"Error evaluating Q '{question}': {e}")
            traceback.print_exc()

        if (i + 1) % 500 == 0:
            print(f"Phase 10 Evaluation processed {i + 1}/3000 questions...")

    elapsed_time = time.time() - start_time
    print(f"Phase 10 Evaluation completed in {elapsed_time:.2f} seconds.")
    
    # Save CSVs
    df_rec = pd.DataFrame(records)
    df_rec.to_csv(results_dir / "phase10_evaluation_results_3000.csv", index=False)
    
    summary_rows = []
    for h in [1, 2, 3]:
        stats = hop_metrics[h]
        cnt = stats["count"]
        summary_rows.append({
            "Hop": f"{h}-hop",
            "Exact Match": stats["exact_match"] / cnt if cnt > 0 else 0,
            "Hit@1": stats["hit"] / cnt if cnt > 0 else 0,
            "Avg Triples": stats["triples"] / cnt if cnt > 0 else 0,
            "Avg Time (ms)": stats["time"] / cnt if cnt > 0 else 0,
            "Count": cnt
        })
    stats_c = combined_metrics
    cnt_c = stats_c["count"]
    summary_rows.append({
        "Hop": "Combined",
        "Exact Match": stats_c["exact_match"] / cnt_c if cnt_c > 0 else 0,
        "Hit@1": stats_c["hit"] / cnt_c if cnt_c > 0 else 0,
        "Avg Triples": stats_c["triples"] / cnt_c if cnt_c > 0 else 0,
        "Avg Time (ms)": stats_c["time"] / cnt_c if cnt_c > 0 else 0,
        "Count": cnt_c
    })
    
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(results_dir / "phase10_evaluation_summary_3000.csv", index=False)
    
    # Process Failure Taxonomy
    cat_counts = {
        "Wrong relation traversal": 0,
        "Unsupported question template": 0,
        "Missing gold answer from retrieved candidates": 0,
        "Answer extraction failure": 0,
        "Retrieval failure": 0,
        "Pruning failure": 0
    }
    
    for f in failures:
        c = f["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
        
    error_report = {
        "total_failures": len(failures),
        "category_counts": cat_counts,
        "category_percentages": {c: (cnt / len(failures) * 100 if len(failures) > 0 else 0.0) for c, cnt in cat_counts.items()},
        "failures_detail": failures[:20]
    }
    
    with open(results_dir / "phase10_error_analysis_3000.json", "w", encoding="utf-8") as f:
        json.dump(error_report, f, indent=2)
        
    # Read Phase 9 Error Analysis for Comparison
    ph9_error_path = results_dir / "error_analysis_3000.json"
    ph9_summary_path = results_dir / "evaluation_summary_3000.csv"
    
    ph9_em = "78.1%"
    ph9_hit = "83.3%"
    ph9_latency = "26.08 ms"
    ph9_errors = {}
    
    if ph9_error_path.exists():
        with open(ph9_error_path, "r", encoding="utf-8") as f:
            ph9_err_data = json.load(f)
            for item in ph9_err_data.get("categories", []):
                ph9_errors[item["category"]] = item["count"]

    print("\n==================================================")
    print("PHASE 10 BEFORE VS AFTER COMPARISON REPORT (3000 Qs)")
    print("==================================================")
    
    p10_em_val = (combined_metrics["exact_match"] / cnt_c) * 100
    p10_hit_val = (combined_metrics["hit"] / cnt_c) * 100
    p10_latency_val = combined_metrics["time"] / cnt_c
    
    print(f"Exact Match:     78.1% (Phase 9) vs {p10_em_val:.2f}% (Phase 10)")
    print(f"Hit@1:           83.3% (Phase 9) vs {p10_hit_val:.2f}% (Phase 10)")
    print(f"Avg Latency:     26.08 ms (Phase 9) vs {p10_latency_val:.2f} ms (Phase 10)")
    print("\nError Category Breakdown:")
    
    all_cats = [
        "Wrong relation traversal",
        "Unsupported question template",
        "Missing gold answer from retrieved candidates",
        "Answer extraction failure",
        "Retrieval failure",
        "Pruning failure"
    ]
    
    for cat in all_cats:
        p9_cnt = ph9_errors.get(cat, 0)
        p10_cnt = cat_counts.get(cat, 0)
        diff = p10_cnt - p9_cnt
        print(f"  - {cat}: {p9_cnt} (Phase 9) -> {p10_cnt} (Phase 10) [{diff:+d}]")
        
    print("==================================================")

if __name__ == "__main__":
    main()
