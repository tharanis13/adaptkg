import sys
import json
import time
import csv
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
from backend.services.answer_engine import (
    answer_question, parse_hop_plan, _TEMPLATE_MATCHER, _masked_question, _walk
)
from backend.services.entity_detection import detect_entities
from backend.utils.eval_metrics import score_answer_sets
from backend.scripts.inspect_dataset import parse_qa_line

_kg = None
_vectorizer = None
_model = None

def init_globals():
    global _kg, _vectorizer, _model
    _kg = load_graph()
    _vectorizer, _model = load_artifacts()

def main():
    print("Initializing Phase 11 Three-Hop Failure Analysis...")
    init_globals()
    
    results_dir = find_project_root() / "results" / "large_scale"
    
    # Load gold questions and selected 3000 questions from Phase 8 log
    questions_log_path = results_dir / "evaluation_questions_3000.json"
    if not questions_log_path.exists():
        print(f"Error: Questions log not found at {questions_log_path}")
        sys.exit(1)
        
    with open(questions_log_path, "r", encoding="utf-8") as f:
        selected_log = json.load(f)
        
    metaqa = find_metaqa_dir()
    gold_map = {}
    path = metaqa / "3-hop" / "vanilla" / "qa_test.txt"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                question, gold = parse_qa_line(line.rstrip("\r\n"))
                gold_map[question] = [g.strip() for g in gold if g.strip()]

    # Filter strictly to the 1,000 3-hop questions from Phase 8/9/10
    three_hop_q_list = selected_log.get("3", [])
    tasks = []
    for q in three_hop_q_list:
        if q in gold_map:
            tasks.append((q, gold_map[q], 3))
            
    print(f"Loaded {len(tasks)} 3-hop questions for Phase 11 analysis.")
    if len(tasks) != 1000:
        print(f"Warning: Expected 1000 questions, got {len(tasks)}")

    records = []
    all_question_details = []
    failed_question_details = []
    
    start_time = time.time()
    
    # Metrics aggregators
    total_q = len(tasks)
    successful_em = 0
    successful_hit = 0
    
    unpruned_gold_present_count = 0
    pruned_gold_present_count = 0
    pruning_loss_count = 0
    
    fallback_count = 0
    matcher_count = 0
    
    successful_sim_scores = []
    failed_sim_scores = []
    
    # Failure category taxonomy tracking
    category_counts = {
        "A — Hop classification failure": 0,
        "B — Entity detection failure": 0,
        "C — Template/relation-plan failure": 0,
        "D — First-hop traversal failure": 0,
        "E — Second-hop traversal failure": 0,
        "F — Third-hop traversal failure": 0,
        "G — Evidence missing from unpruned retrieval": 0,
        "H — Pruning failure": 0,
        "I — Answer extraction failure": 0,
        "J — Other": 0
    }

    category_examples = {cat: [] for cat in category_counts}

    for idx, (question, gold_answers, true_hop) in enumerate(tasks):
        try:
            # ==================================================
            # 1. STRICT INFERENCE RULE: Run pipeline WITHOUT gold labels
            # ==================================================
            res = run_adaptive_pipeline(
                question,
                kg=_kg,
                vectorizer=_vectorizer,
                model=_model,
                max_candidates_per_expansion=100,
                max_total_candidates=5000,
                pruning_strategy="relevance_bounded"
            )
            
            # Score predicted answer against normalized gold answers
            scores = score_answer_sets(res["answer"], gold_answers)
            em = scores["exact_match"]
            hit = scores["hit"]
            
            if em:
                successful_em += 1
            if hit:
                successful_hit += 1
                
            # Log template match info
            q_masked = _masked_question(question)
            plan, info = _TEMPLATE_MATCHER.match(q_masked, max_hops=res["predicted_hop"])
            used_fallback = info.get("used_fallback", True)
            sim_score = info.get("similarity_score", 0.0)
            matched_template = info.get("matched_template", None)
            
            if used_fallback:
                fallback_count += 1
            else:
                matcher_count += 1
                
            if em:
                successful_sim_scores.append(sim_score)
            else:
                failed_sim_scores.append(sim_score)
                
            q_detail = {
                "question": question,
                "gold_answers": gold_answers,
                "predicted_answers": res["answer"],
                "predicted_hop": res["predicted_hop"],
                "classifier_confidence": res["confidence"],
                "selected_retrieval_depth": res["selected_retrieval_depth"],
                "detected_entities": res["detected_entities"],
                "seed_entities": res["seed_entities"],
                "matched_template": matched_template,
                "template_similarity_score": sim_score,
                "used_fallback": used_fallback,
                "predicted_relation_plan": res["hop_plan"],
                "retrieved_candidate_count": res["candidate_triple_count"],
                "ranked_candidate_count": len(res.get("ranked_triples", [])),
                "exact_match": em,
                "hit": hit
            }

            # ==================================================
            # 2. POST-HOC DIAGNOSTICS FOR FAILED QUESTIONS
            # ==================================================
            if not em:
                # 4. Unpruned vs Pruned Coverage Diagnostic
                unpruned_ret = retrieve_for_question(
                    question, max_hops=3, kg=_kg,
                    max_candidates_per_expansion=None,
                    max_total_candidates=None,
                    pruning_strategy="none"
                )
                unpruned_nodes = set()
                for t in unpruned_ret["triples"]:
                    unpruned_nodes.add(t["subject"])
                    unpruned_nodes.add(t["object"])
                gold_in_unpruned = any(g in unpruned_nodes for g in gold_answers)
                
                pruned_ret = retrieve_for_question(
                    question, max_hops=3, kg=_kg,
                    max_candidates_per_expansion=100,
                    max_total_candidates=5000,
                    pruning_strategy="relevance_bounded"
                )
                pruned_nodes = set()
                for t in pruned_ret["triples"]:
                    pruned_nodes.add(t["subject"])
                    pruned_nodes.add(t["object"])
                gold_in_pruned = any(g in pruned_nodes for g in gold_answers)
                
                if gold_in_unpruned:
                    unpruned_gold_present_count += 1
                if gold_in_pruned:
                    pruned_gold_present_count += 1
                if gold_in_unpruned and not gold_in_pruned:
                    pruning_loss_count += 1

                q_detail["gold_in_unpruned"] = gold_in_unpruned
                q_detail["gold_in_pruned"] = gold_in_pruned

                # Trace evidence paths in unpruned graph
                seeds = res["seed_entities"]
                evidence_paths = []
                if seeds and gold_in_unpruned:
                    # Find NetworkX paths from seeds to gold answers up to 3 hops
                    graph = _kg.graph
                    for seed in seeds:
                        for g in gold_answers:
                            if graph.has_node(seed) and graph.has_node(g):
                                try:
                                    import networkx as nx
                                    paths = list(nx.all_simple_paths(graph, source=seed, target=g, cutoff=3))
                                    for p in paths[:3]:
                                        # reconstruct relation sequence
                                        path_str = []
                                        for step_i in range(len(p) - 1):
                                            u, v = p[step_i], p[step_i + 1]
                                            edge_data = graph.get_edge_data(u, v)
                                            rel = edge_data[0]["relation"] if edge_data else "rel"
                                            path_str.append(f"{u} -[{rel}]-> {v}")
                                        evidence_paths.append(" -> ".join(path_str))
                                except Exception:
                                    pass
                q_detail["evidence_paths"] = evidence_paths[:3]

                # ==================================================
                # 3. DETERMINISTIC FAILURE TAXONOMY ASSIGNMENT
                # Priority: A -> B -> C -> D -> E -> F -> G -> H -> I -> J
                # ==================================================
                primary_category = None
                
                # Category A: Hop classification failure
                if res["predicted_hop"] != 3:
                    primary_category = "A — Hop classification failure"
                # Category B: Entity detection failure
                elif not seeds:
                    primary_category = "B — Entity detection failure"
                else:
                    # Parse plan matching
                    true_3hop_plan, _ = _TEMPLATE_MATCHER.match(q_masked, max_hops=3)
                    
                    # Check first-hop, second-hop, third-hop traversal step by step
                    # Category C: Template / relation-plan failure
                    if not true_3hop_plan and not res["hop_plan"]:
                        primary_category = "C — Template/relation-plan failure"
                    elif len(res["hop_plan"]) < 3:
                        primary_category = "C — Template/relation-plan failure"
                    else:
                        # Step-by-step traversal check
                        hop1_plan = [res["hop_plan"][0]]
                        ans1, _ = _walk(_kg, seeds, hop1_plan)
                        if not ans1:
                            primary_category = "D — First-hop traversal failure"
                        else:
                            hop2_plan = res["hop_plan"][:2]
                            ans2, _ = _walk(_kg, seeds, hop2_plan)
                            if not ans2:
                                primary_category = "E — Second-hop traversal failure"
                            else:
                                hop3_plan = res["hop_plan"][:3]
                                ans3, _ = _walk(_kg, seeds, hop3_plan)
                                gold_in_ans3 = any(g in ans3 for g in gold_answers)
                                
                                if not ans3 or not gold_in_ans3:
                                    if not gold_in_unpruned:
                                        primary_category = "G — Evidence missing from unpruned retrieval"
                                    elif gold_in_unpruned and not gold_in_pruned:
                                        primary_category = "H — Pruning failure"
                                    else:
                                        primary_category = "F — Third-hop traversal failure"
                                else:
                                    # ans3 contained gold answer, but scores["exact_match"] failed
                                    primary_category = "I — Answer extraction failure"

                if primary_category is None:
                    primary_category = "J — Other"

                q_detail["primary_failure_category"] = primary_category
                category_counts[primary_category] += 1
                
                example_record = {
                    "question": question,
                    "gold_answers": gold_answers,
                    "predicted_answers": res["answer"],
                    "predicted_hop": res["predicted_hop"],
                    "classifier_confidence": res["confidence"],
                    "selected_retrieval_depth": res["selected_retrieval_depth"],
                    "template_similarity_score": sim_score,
                    "used_fallback": used_fallback,
                    "evidence_paths": q_detail.get("evidence_paths", []),
                    "primary_failure_category": primary_category
                }
                
                if len(category_examples[primary_category]) < 5:
                    category_examples[primary_category].append(example_record)
                    
                failed_question_details.append(q_detail)
            
            all_question_details.append(q_detail)

        except Exception as e:
            print(f"Error evaluating 3-hop Q '{question}': {e}")
            traceback.print_exc()

        if (idx + 1) % 250 == 0:
            print(f"Phase 11 Analysis processed {idx + 1}/1000 3-hop questions...")

    elapsed_time = time.time() - start_time
    print(f"Phase 11 3-Hop Analysis completed in {elapsed_time:.2f} seconds.")

    # Save detailed JSONs
    with open(results_dir / "phase11_3hop_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_questions": total_q,
                "successful_em": successful_em,
                "em_rate": successful_em / total_q,
                "successful_hit": successful_hit,
                "hit_rate": successful_hit / total_q,
                "unpruned_gold_present_count": unpruned_gold_present_count,
                "pruned_gold_present_count": pruned_gold_present_count,
                "pruning_loss_count": pruning_loss_count,
                "fallback_count": fallback_count,
                "matcher_count": matcher_count,
                "avg_sim_score_successful": float(np.mean(successful_sim_scores)) if successful_sim_scores else 0.0,
                "avg_sim_score_failed": float(np.mean(failed_sim_scores)) if failed_sim_scores else 0.0,
                "category_counts": category_counts,
                "category_percentages": {c: (cnt / (total_q - successful_em) * 100) if (total_q - successful_em) > 0 else 0.0 for c, cnt in category_counts.items()},
                "questions": all_question_details
            },
            f, indent=2
        )

    with open(results_dir / "phase11_3hop_examples.json", "w", encoding="utf-8") as f:
        json.dump(category_examples, f, indent=2)

    # Generate Summary CSV
    failed_count = total_q - successful_em
    csv_rows = []
    csv_rows.append(["Category", "Count", "Percentage_of_Failures", "Percentage_of_Total_3Hop"])
    for cat, cnt in category_counts.items():
        pct_fail = (cnt / failed_count * 100) if failed_count > 0 else 0.0
        pct_total = (cnt / total_q * 100)
        csv_rows.append([cat, cnt, f"{pct_fail:.2f}%", f"{pct_total:.2f}%"])

    summary_csv_path = results_dir / "phase11_3hop_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print("\n==================================================")
    print("PHASE 11 DIAGNOSTIC SUMMARY (1000 3-Hop Questions)")
    print("==================================================")
    print(f"3-Hop Exact Match:  {successful_em}/{total_q} ({(successful_em/total_q)*100:.2f}%)")
    print(f"3-Hop Hit@1:        {successful_hit}/{total_q} ({(successful_hit/total_q)*100:.2f}%)")
    print(f"Total Failures:      {failed_count}")
    print("\nFailure Taxonomy Breakdown:")
    for cat, cnt in category_counts.items():
        pct = (cnt / failed_count * 100) if failed_count > 0 else 0.0
        print(f"  {cat:<45}: {cnt:>4} ({pct:>6.2f}%)")
        
    print("\nDiagnostic Research Questions:")
    print(f"1. Gold answer present in unpruned graph: {unpruned_gold_present_count}/{failed_count} failures ({(unpruned_gold_present_count/failed_count)*100:.2f}%)")
    print(f"2. Failures due to Hop Classification (pred!=3): {category_counts['A — Hop classification failure']}")
    print(f"3. Failures due to Relation Plan / Template: {category_counts['C — Template/relation-plan failure']}")
    print(f"4. Failures due to Pruning (present in unpruned, absent in pruned): {pruning_loss_count}")
    print(f"5. Failures during Answer Extraction (correct evidence reached, but wrong answer extracted): {category_counts['I — Answer extraction failure']}")
    print("==================================================")

if __name__ == "__main__":
    main()
