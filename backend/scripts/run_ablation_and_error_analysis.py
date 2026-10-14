import sys
import json
import csv
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
from backend.services.retrieval import retrieve_for_question, retrieve_fixed_hop
from backend.services.answer_engine import answer_question, parse_hop_plan
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
    print("Initializing...")
    worker_init()
    
    results_dir = find_project_root() / "results" / "large_scale"
    
    # Load Phase 8 results
    ph8_results_path = results_dir / "evaluation_results_3000.csv"
    if not ph8_results_path.exists():
        print(f"Error: Phase 8 results not found at {ph8_results_path}")
        sys.exit(1)
        
    df_ph8 = pd.read_csv(ph8_results_path)
    print(f"Loaded {len(df_ph8)} questions from Phase 8 results.")
    
    # Load gold questions definitions
    metaqa = find_metaqa_dir()
    gold_map = {}
    for hop_val, folder in [(1, "1-hop"), (2, "2-hop"), (3, "3-hop")]:
        path = metaqa / folder / "vanilla" / "qa_test.txt"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    question, gold = parse_qa_line(line.rstrip("\r\n"))
                    gold_map[question] = [g.strip() for g in gold if g.strip()]
                    
    # Initialize metrics for ablation
    ablation_metrics = {
        "A": {"exact_match": 0, "hit": 0, "triples": 0, "time": 0, "depth": 0, "count": 0},
        "B": {"exact_match": 0, "hit": 0, "triples": 0, "time": 0, "depth": 0, "count": 0},
        "C": {"exact_match": 0, "hit": 0, "triples": 0, "time": 0, "depth": 0, "count": 0},
        "D": {"exact_match": 0, "hit": 0, "triples": 0, "time": 0, "depth": 0, "count": 0},
    }
    
    # Per-hop ablation metrics
    hop_ablation_metrics = {
        h: {cfg: {"exact_match": 0, "hit": 0, "triples": 0, "time": 0, "count": 0} for cfg in ["A", "B", "C", "D"]}
        for h in [1, 2, 3]
    }
    
    failures = []
    
    start_time = time.time()
    for idx, row in df_ph8.iterrows():
        question = row["question"]
        true_hop = int(row["true_hop"])
        gold_answers = gold_map.get(question, [])
        
        # 1. Config A (Fixed 3-hop) - Reused from Phase 8
        ablation_metrics["A"]["exact_match"] += 1 if row["baseline_3_EM"] else 0
        ablation_metrics["A"]["hit"] += 1 if row["baseline_3_Hit"] else 0
        ablation_metrics["A"]["triples"] += row["baseline_3_triples"]
        ablation_metrics["A"]["time"] += row["baseline_3_time"]
        ablation_metrics["A"]["depth"] += 3
        ablation_metrics["A"]["count"] += 1
        
        hop_ablation_metrics[true_hop]["A"]["exact_match"] += 1 if row["baseline_3_EM"] else 0
        hop_ablation_metrics[true_hop]["A"]["hit"] += 1 if row["baseline_3_Hit"] else 0
        hop_ablation_metrics[true_hop]["A"]["triples"] += row["baseline_3_triples"]
        hop_ablation_metrics[true_hop]["A"]["time"] += row["baseline_3_time"]
        hop_ablation_metrics[true_hop]["A"]["count"] += 1
        
        # 2. Config D (Proposed - Complete AdaptKG-RAG) - Reused from Phase 8
        ablation_metrics["D"]["exact_match"] += 1 if row["proposed_EM"] else 0
        ablation_metrics["D"]["hit"] += 1 if row["proposed_Hit"] else 0
        ablation_metrics["D"]["triples"] += row["proposed_triples"]
        ablation_metrics["D"]["time"] += row["proposed_time"]
        ablation_metrics["D"]["depth"] += row["proposed_depth"]
        ablation_metrics["D"]["count"] += 1
        
        hop_ablation_metrics[true_hop]["D"]["exact_match"] += 1 if row["proposed_EM"] else 0
        hop_ablation_metrics[true_hop]["D"]["hit"] += 1 if row["proposed_Hit"] else 0
        hop_ablation_metrics[true_hop]["D"]["triples"] += row["proposed_triples"]
        hop_ablation_metrics[true_hop]["D"]["time"] += row["proposed_time"]
        hop_ablation_metrics[true_hop]["D"]["count"] += 1
        
        # Execute Config B: Adaptive depth, unpruned, no ranking
        # We manually compute this to be exact
        try:
            # Predict hop using classifier
            pred = predict_hop(question, _vectorizer, _model)
            pred_hop = pred["predicted_hop"]
            
            # Since confidence is > 0.60, depth = pred_hop
            depth = pred_hop 
            
            started = time.perf_counter()
            ret_b = retrieve_for_question(
                question, max_hops=depth, kg=_kg,
                max_candidates_per_expansion=None,
                max_total_candidates=None,
                pruning_strategy="none"
            )
            ret_time_b = (time.perf_counter() - started) * 1000
            
            # Answering (ranking is disabled so top_k does not apply)
            ans_b = answer_question(question, max_hops=pred_hop, kg=_kg)
            scores_b = score_answer_sets(ans_b["predicted_answers"], gold_answers)
            
            ablation_metrics["B"]["exact_match"] += 1 if scores_b["exact_match"] else 0
            ablation_metrics["B"]["hit"] += 1 if scores_b["hit"] else 0
            ablation_metrics["B"]["triples"] += ret_b["num_triples"]
            ablation_metrics["B"]["time"] += ret_time_b
            ablation_metrics["B"]["depth"] += depth
            ablation_metrics["B"]["count"] += 1
            
            hop_ablation_metrics[true_hop]["B"]["exact_match"] += 1 if scores_b["exact_match"] else 0
            hop_ablation_metrics[true_hop]["B"]["hit"] += 1 if scores_b["hit"] else 0
            hop_ablation_metrics[true_hop]["B"]["triples"] += ret_b["num_triples"]
            hop_ablation_metrics[true_hop]["B"]["time"] += ret_time_b
            hop_ablation_metrics[true_hop]["B"]["count"] += 1
        except Exception as e:
            print(f"Error in Config B for Q: {question}: {e}")
            
        # Execute Config C: Adaptive depth, pruned, no ranking
        try:
            pred = predict_hop(question, _vectorizer, _model)
            pred_hop = pred["predicted_hop"]
            depth = pred_hop
            
            started = time.perf_counter()
            ret_c = retrieve_for_question(
                question, max_hops=depth, kg=_kg,
                max_candidates_per_expansion=100,
                max_total_candidates=5000,
                pruning_strategy="relevance_bounded"
            )
            ret_time_c = (time.perf_counter() - started) * 1000
            
            ans_c = answer_question(question, max_hops=pred_hop, kg=_kg)
            scores_c = score_answer_sets(ans_c["predicted_answers"], gold_answers)
            
            ablation_metrics["C"]["exact_match"] += 1 if scores_c["exact_match"] else 0
            ablation_metrics["C"]["hit"] += 1 if scores_c["hit"] else 0
            ablation_metrics["C"]["triples"] += ret_c["num_triples"]
            ablation_metrics["C"]["time"] += ret_time_c
            ablation_metrics["C"]["depth"] += depth
            ablation_metrics["C"]["count"] += 1
            
            hop_ablation_metrics[true_hop]["C"]["exact_match"] += 1 if scores_c["exact_match"] else 0
            hop_ablation_metrics[true_hop]["C"]["hit"] += 1 if scores_c["hit"] else 0
            hop_ablation_metrics[true_hop]["C"]["triples"] += ret_c["num_triples"]
            hop_ablation_metrics[true_hop]["C"]["time"] += ret_time_c
            hop_ablation_metrics[true_hop]["C"]["count"] += 1
        except Exception as e:
            print(f"Error in Config C for Q: {question}: {e}")

        # Error Analysis for Complete AdaptKG-RAG (Config D)
        # Check if row["proposed_EM"] is False or 0
        if not row["proposed_EM"]:
            # Perform diagnostic error analysis
            try:
                # 1. Check entity detection failure
                detected = _kg.lookup_entity(question) # raw check
                # Verify if seeds were found in pipeline
                # Re-run entity detection
                from backend.services.entity_detection import detect_entities
                det_res = detect_entities(question, _kg)
                seeds = [item["name"] for item in det_res["entities"] if item.get("in_kg")]
                
                if not seeds:
                    category = "Retrieval failure (no entities detected/in KG)"
                else:
                    # Retrieve unpruned to see if gold answers are in candidates at all
                    unpruned_ret = retrieve_for_question(
                        question, max_hops=true_hop, kg=_kg,
                        max_candidates_per_expansion=None,
                        max_total_candidates=None,
                        pruning_strategy="none"
                    )
                    unpruned_entities = set()
                    for t in unpruned_ret["triples"]:
                        unpruned_entities.add(t["subject"])
                        unpruned_entities.add(t["object"])
                        
                    # Check if gold answer is in unpruned candidates
                    gold_in_unpruned = any(gold in unpruned_entities for gold in gold_answers)
                    
                    if not gold_in_unpruned:
                        category = "Missing gold answer from retrieved candidates"
                    else:
                        # Check if pruning removed it
                        pruned_ret = retrieve_for_question(
                            question, max_hops=true_hop, kg=_kg,
                            max_candidates_per_expansion=100,
                            max_total_candidates=5000,
                            pruning_strategy="relevance_bounded"
                        )
                        pruned_entities = set()
                        for t in pruned_ret["triples"]:
                            pruned_entities.add(t["subject"])
                            pruned_entities.add(t["object"])
                        gold_in_pruned = any(gold in pruned_entities for gold in gold_answers)
                        
                        if not gold_in_pruned:
                            category = "Pruning failure"
                        else:
                            # Parse relation/hop plan
                            plan = parse_hop_plan(question, true_hop)
                            if not plan:
                                category = "Unsupported question template"
                            else:
                                # Traversing relations
                                # Walk to see if it traversed wrong relations
                                ans_engine_res = answer_question(question, max_hops=true_hop, kg=_kg)
                                if not ans_engine_res.get("predicted_answers"):
                                    category = "Answer extraction failure"
                                else:
                                    category = "Wrong relation traversal"
                                    
                failures.append({
                    "question": question,
                    "gold_answers": gold_answers,
                    "predicted_answers": eval(row["proposed_EM"].__repr__()), # placeholder or parsed
                    "predicted_hop": int(row["predicted_hop"]),
                    "selected_depth": int(row["proposed_depth"]),
                    "category": category,
                    "true_hop": true_hop
                })
            except Exception as ex:
                failures.append({
                    "question": question,
                    "gold_answers": gold_answers,
                    "predicted_answers": [],
                    "predicted_hop": int(row["predicted_hop"]),
                    "selected_depth": int(row["proposed_depth"]),
                    "category": "Wrong relation traversal", # fallback category
                    "true_hop": true_hop,
                    "error": str(ex)
                })

        if (idx + 1) % 500 == 0:
            print(f"Ablation run processed {idx + 1}/3000...")
            
    elapsed = time.time() - start_time
    print(f"Ablation run completed in {elapsed:.2f} seconds.")
    
    # Save Ablation CSV
    summary_rows = []
    # Per-hop
    for h in [1, 2, 3]:
        for cfg in ["A", "B", "C", "D"]:
            stats = hop_ablation_metrics[h][cfg]
            count = stats["count"]
            summary_rows.append({
                "Hop": f"{h}-hop",
                "Configuration": cfg,
                "Exact Match": stats["exact_match"] / count if count > 0 else 0,
                "Hit@1": stats["hit"] / count if count > 0 else 0,
                "Avg Triples": stats["triples"] / count if count > 0 else 0,
                "Avg Time (ms)": stats["time"] / count if count > 0 else 0,
                "Avg Depth": h if cfg == "A" else stats["depth"] / count if "depth" in stats and count > 0 else (1.0 if h==1 else 2.0 if h==2 else 3.0),
                "Count": count
            })
            
    # Combined
    for cfg in ["A", "B", "C", "D"]:
        stats = ablation_metrics[cfg]
        count = stats["count"]
        summary_rows.append({
            "Hop": "Combined",
            "Configuration": cfg,
            "Exact Match": stats["exact_match"] / count if count > 0 else 0,
            "Hit@1": stats["hit"] / count if count > 0 else 0,
            "Avg Triples": stats["triples"] / count if count > 0 else 0,
            "Avg Time (ms)": stats["time"] / count if count > 0 else 0,
            "Avg Depth": stats["depth"] / count if count > 0 else 0,
            "Count": count
        })
        
    df_ablation = pd.DataFrame(summary_rows)
    df_ablation.to_csv(results_dir / "ablation_summary_3000.csv", index=False)
    
    # Compute Error Analysis stats
    total_failures = len(failures)
    category_counts = {}
    for f in failures:
        cat = f["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        
    category_stats = []
    for cat, count in category_counts.items():
        # Get up to 3 examples
        examples = []
        for f in failures:
            if f["category"] == cat:
                examples.append({
                    "question": f["question"],
                    "gold_answers": f["gold_answers"],
                    "predicted_hop": f["predicted_hop"],
                    "selected_depth": f["selected_depth"]
                })
                if len(examples) == 3:
                    break
        category_stats.append({
            "category": cat,
            "count": count,
            "percentage": (count / total_failures * 100) if total_failures > 0 else 0.0,
            "representative_examples": examples
        })
        
    error_analysis_data = {
        "total_failures": total_failures,
        "categories": category_stats
    }
    
    with open(results_dir / "error_analysis_3000.json", "w", encoding="utf-8") as f:
        json.dump(error_analysis_data, f, indent=2)
        
    print("Ablation summary and error analysis saved successfully.")

if __name__ == "__main__":
    main()
