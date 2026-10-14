import sys
import json
import csv
import time
import random
from pathlib import Path
import traceback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.utils.paths import find_metaqa_dir, find_project_root
from backend.services.knowledge_graph import load_graph
from backend.services.hop_classifier import load_artifacts, predict_hop
from backend.services.adaptive_pipeline import run_adaptive_pipeline
from backend.services.retrieval import retrieve_for_question
from backend.services.answer_engine import answer_question
from backend.utils.eval_metrics import score_answer_sets
from backend.scripts.inspect_dataset import parse_qa_line

_kg = None
_vectorizer = None
_model = None

def worker_init():
    global _kg, _vectorizer, _model
    _kg = load_graph()
    _vectorizer, _model = load_artifacts()

def process_question(args):
    question, gold, true_hop = args
    global _kg, _vectorizer, _model
    
    results = {}
    pruning_kwargs = {
        "pruning_strategy": "relevance_bounded",
        "max_candidates_per_expansion": 100,
        "max_total_candidates": 5000,
    }
    
    # Baselines 1, 2, 3
    for h in (1, 2, 3):
        try:
            retrieval = retrieve_for_question(
                question, max_hops=h, kg=_kg,
                max_candidates_per_expansion=pruning_kwargs["max_candidates_per_expansion"],
                max_total_candidates=pruning_kwargs["max_total_candidates"],
                pruning_strategy=pruning_kwargs["pruning_strategy"]
            )
            ans = answer_question(question, max_hops=h, kg=_kg)
            scores = score_answer_sets(ans["predicted_answers"], gold)
            results[f"baseline_{h}"] = {
                "exact_match": scores["exact_match"],
                "hit": scores["hit"],
                "candidate_triples": retrieval["num_triples"],
                "retrieval_time_ms": retrieval["elapsed_ms"],
                "depth": h
            }
        except Exception as e:
            results[f"baseline_{h}"] = {"error": str(e), "trace": traceback.format_exc()}

    # Proposed (AdaptKG-RAG)
    try:
        adapt = run_adaptive_pipeline(
            question, kg=_kg, vectorizer=_vectorizer, model=_model,
            max_candidates_per_expansion=pruning_kwargs["max_candidates_per_expansion"],
            max_total_candidates=pruning_kwargs["max_total_candidates"],
            pruning_strategy=pruning_kwargs["pruning_strategy"]
        )
        scores = score_answer_sets(adapt["answer"], gold)
        results["proposed"] = {
            "exact_match": scores["exact_match"],
            "hit": scores["hit"],
            "candidate_triples": adapt["candidate_triple_count"],
            "retrieval_time_ms": adapt["retrieval_time_ms"],
            "depth": adapt["selected_retrieval_depth"],
            "predicted_hop": adapt["predicted_hop"],
            "true_hop": true_hop,
            "confidence": adapt["confidence"],
            "used_fallback": adapt["used_fallback"]
        }
    except Exception as e:
        results["proposed"] = {"error": str(e), "trace": traceback.format_exc()}

    return question, true_hop, results

def main():
    print("Loading test data...")
    metaqa = find_metaqa_dir()
    
    # Selection of 1000 questions per hop deterministically
    tasks = []
    selected_log = {}
    for hop_val, folder in [(1, "1-hop"), (2, "2-hop"), (3, "3-hop")]:
        path = metaqa / folder / "vanilla" / "qa_test.txt"
        hop_questions = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    question, gold = parse_qa_line(line)
                    hop_questions.append((question, gold, hop_val))
                    
        # Remove duplicates based on question text, then sort for absolute determinism
        unique_tasks = {}
        for q, g, h in hop_questions:
            unique_tasks[q] = g
        sorted_keys = sorted(unique_tasks.keys())
        
        # Seeded deterministic random sample
        rng = random.Random(42)
        sampled_keys = rng.sample(sorted_keys, min(1000, len(sorted_keys)))
        
        selected_hop_tasks = [(q, unique_tasks[q], hop_val) for q in sampled_keys]
        tasks.extend(selected_hop_tasks)
        selected_log[str(hop_val)] = sampled_keys
        
    print(f"Total test questions sampled: {len(tasks)} (1000 per hop complexity)")
    
    results_dir = find_project_root() / "results" / "large_scale"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the selected questions log
    with open(results_dir / "evaluation_questions_3000.json", "w", encoding="utf-8") as f:
        json.dump(selected_log, f, indent=2)
        
    worker_init()
    
    print("Running large-scale evaluation sequentially...")
    all_results = []
    
    # Checkpoint setup
    checkpoint_path = results_dir / "checkpoint_3000.json"
    start_idx = 0
    if checkpoint_path.is_file():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                saved_state = json.load(f)
                all_results = saved_state["results"]
                start_idx = saved_state["next_idx"]
                print(f"Resuming from checkpoint. Loaded {len(all_results)} previous results.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            all_results = []
            start_idx = 0

    start_time = time.time()
    
    for i in range(start_idx, len(tasks)):
        task = tasks[i]
        q, gold, true_hop = task
        _, _, res = process_question(task)
        all_results.append((q, true_hop, res))
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(tasks)} questions...")
            # Save checkpoint
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"results": all_results, "next_idx": i + 1}, f)
                
    elapsed_time = time.time() - start_time
    print(f"Evaluation complete in {elapsed_time:.2f} seconds.")
    
    # Remove checkpoint after successful run
    if checkpoint_path.is_file():
        checkpoint_path.unlink()
        
    # Metrics computation
    # Metrics dictionaries grouped by true hop
    hop_metrics = {
        1: {m: {"exact_match": 0, "hit": 0, "triples": 0, "time": 0, "count": 0} for m in ["baseline_1", "baseline_2", "baseline_3", "proposed"]},
        2: {m: {"exact_match": 0, "hit": 0, "triples": 0, "time": 0, "count": 0} for m in ["baseline_1", "baseline_2", "baseline_3", "proposed"]},
        3: {m: {"exact_match": 0, "hit": 0, "triples": 0, "time": 0, "count": 0} for m in ["baseline_1", "baseline_2", "baseline_3", "proposed"]},
    }
    
    combined_metrics = {m: {"exact_match": 0, "hit": 0, "triples": 0, "time": 0, "depth": 0, "count": 0} for m in ["baseline_1", "baseline_2", "baseline_3", "proposed"]}
    
    true_hops = []
    pred_hops = []
    confidences = []
    fallbacks = 0
    depths = []
    failed_questions = []
    
    records = []
    
    for q, true_hop, res in all_results:
        rec = {"question": q, "true_hop": true_hop}
        has_error = False
        
        for method in ["baseline_1", "baseline_2", "baseline_3", "proposed"]:
            if method in res and "error" not in res[method]:
                data = res[method]
                # Combined
                combined_metrics[method]["exact_match"] += 1 if data["exact_match"] else 0
                combined_metrics[method]["hit"] += 1 if data["hit"] else 0
                combined_metrics[method]["triples"] += data["candidate_triples"]
                combined_metrics[method]["time"] += data["retrieval_time_ms"]
                combined_metrics[method]["depth"] += data["depth"]
                combined_metrics[method]["count"] += 1
                
                # Per Hop
                hop_metrics[true_hop][method]["exact_match"] += 1 if data["exact_match"] else 0
                hop_metrics[true_hop][method]["hit"] += 1 if data["hit"] else 0
                hop_metrics[true_hop][method]["triples"] += data["candidate_triples"]
                hop_metrics[true_hop][method]["time"] += data["retrieval_time_ms"]
                hop_metrics[true_hop][method]["count"] += 1
                
                rec[f"{method}_EM"] = data["exact_match"]
                rec[f"{method}_Hit"] = data["hit"]
                rec[f"{method}_triples"] = data["candidate_triples"]
                rec[f"{method}_time"] = data["retrieval_time_ms"]
                rec[f"{method}_depth"] = data["depth"]
                
                if method == "proposed":
                    true_hops.append(true_hop)
                    pred_hops.append(data["predicted_hop"])
                    confidences.append(data["confidence"])
                    depths.append(data["depth"])
                    if data["used_fallback"]:
                        fallbacks += 1
                    rec["confidence"] = data["confidence"]
                    rec["used_fallback"] = data["used_fallback"]
                    rec["predicted_hop"] = data["predicted_hop"]
            else:
                has_error = True
                err_msg = res.get(method, {}).get("error", "Unknown error")
                rec[f"{method}_error"] = err_msg
                
        if has_error:
            failed_questions.append((q, true_hop, res))
            
        records.append(rec)
        
    # Write raw results
    df_raw = pd.DataFrame(records)
    df_raw.to_csv(results_dir / "evaluation_results_3000.csv", index=False)
    
    # Compile Summary
    summary_rows = []
    
    # 1. Report individual hops
    for hop_val in [1, 2, 3]:
        for method in ["baseline_1", "baseline_2", "baseline_3", "proposed"]:
            stats = hop_metrics[hop_val][method]
            count = stats["count"]
            if count > 0:
                summary_rows.append({
                    "Hop": f"{hop_val}-hop",
                    "Method": method,
                    "Exact Match": stats["exact_match"] / count,
                    "Hit@1": stats["hit"] / count,
                    "Avg Triples": stats["triples"] / count,
                    "Avg Time (ms)": stats["time"] / count,
                    "Count": count
                })
                
    # 2. Report Combined
    for method in ["baseline_1", "baseline_2", "baseline_3", "proposed"]:
        stats = combined_metrics[method]
        count = stats["count"]
        if count > 0:
            summary_rows.append({
                "Hop": "Combined",
                "Method": method,
                "Exact Match": stats["exact_match"] / count,
                "Hit@1": stats["hit"] / count,
                "Avg Triples": stats["triples"] / count,
                "Avg Time (ms)": stats["time"] / count,
                "Count": count
            })
            
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(results_dir / "evaluation_summary_3000.csv", index=False)
    
    # Classification metrics
    cm = confusion_matrix(true_hops, pred_hops, labels=[1, 2, 3])
    clf_report = classification_report(true_hops, pred_hops, labels=[1, 2, 3], output_dict=True, zero_division=0)
    
    # Hop prediction stats
    correct_predictions = sum(1 for t, p in zip(true_hops, pred_hops) if t == p)
    incorrect_predictions = len(true_hops) - correct_predictions
    
    depth_distribution = {str(d): depths.count(d) for d in [1, 2, 3]}
    pred_hop_distribution = {str(p): pred_hops.count(p) for p in [1, 2, 3]}
    
    clf_metrics = {
        "classification_report": clf_report,
        "confusion_matrix": cm.tolist(),
        "total_evaluated": len(true_hops),
        "correct_predictions": correct_predictions,
        "incorrect_predictions": incorrect_predictions,
        "accuracy": correct_predictions / len(true_hops) if len(true_hops) > 0 else 0.0,
        "fallback_decisions": fallbacks,
        "selected_retrieval_depth_distribution": depth_distribution,
        "predicted_hop_distribution": pred_hop_distribution,
        "average_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "failed_questions_count": len(failed_questions),
        "elapsed_evaluation_time_sec": elapsed_time
    }
    
    with open(results_dir / "classifier_metrics_3000.json", "w", encoding="utf-8") as f:
        json.dump(clf_metrics, f, indent=2)
        
    # Generate Charts
    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
    plt.xlabel('Predicted Hop')
    plt.ylabel('True Hop')
    plt.title('Hop Classifier Confusion Matrix (3,000 Questions)')
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix_3000.png")
    plt.close()
    
    # 2. Accuracy Comparison (Combined)
    plt.figure(figsize=(8, 5))
    combined_df = df_summary[df_summary["Hop"] == "Combined"]
    sns.barplot(data=combined_df, x="Method", y="Exact Match", palette="viridis")
    plt.title('Combined Exact Match Accuracy Comparison')
    plt.ylim(0, 1.1)
    for i, v in enumerate(combined_df["Exact Match"]):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_comparison_3000.png")
    plt.close()
    
    # 3. Efficiency Comparison (Avg Candidate Triples - Combined)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=combined_df, x="Method", y="Avg Triples", palette="magma")
    plt.title('Retrieval Efficiency: Avg Candidate Triples (Combined)')
    for i, v in enumerate(combined_df["Avg Triples"]):
        plt.text(i, v + (v * 0.02), f"{v:.0f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(results_dir / "efficiency_comparison_3000.png")
    plt.close()
    
    # 4. Retrieval Depth Distribution for AdaptKG-RAG
    plt.figure(figsize=(6, 5))
    depth_labels = list(depth_distribution.keys())
    depth_counts = list(depth_distribution.values())
    sns.barplot(x=depth_labels, y=depth_counts, palette="crest")
    plt.title('AdaptKG-RAG Selected Retrieval Depth Distribution')
    plt.xlabel('Selected Depth')
    plt.ylabel('Number of Questions')
    for i, v in enumerate(depth_counts):
        plt.text(i, v + (v * 0.02), str(v), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(results_dir / "retrieval_depth_distribution_3000.png")
    plt.close()
    
    print("Done! Evaluation results generated and stored successfully.")

if __name__ == "__main__":
    main()
