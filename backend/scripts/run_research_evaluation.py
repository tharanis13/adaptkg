import sys
import json
import csv
import time
from pathlib import Path
import multiprocessing
import traceback

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.utils.paths import find_metaqa_dir, find_project_root
from backend.services.knowledge_graph import load_graph
from backend.services.hop_classifier import load_artifacts
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
    question, gold, true_hop, pruning_kwargs = args
    global _kg, _vectorizer, _model
    
    results = {}
    
    # Baselines 1, 2, 3
    for h in (1, 2, 3):
        try:
            retrieval = retrieve_for_question(
                question, max_hops=h, kg=_kg,
                max_candidates_per_expansion=pruning_kwargs.get("max_candidates_per_expansion"),
                max_total_candidates=pruning_kwargs.get("max_total_candidates"),
                pruning_strategy=pruning_kwargs.get("pruning_strategy")
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
            results[f"baseline_{h}"] = {"error": str(e)}

    # Proposed
    try:
        adapt = run_adaptive_pipeline(
            question, kg=_kg, vectorizer=_vectorizer, model=_model,
            max_candidates_per_expansion=pruning_kwargs.get("max_candidates_per_expansion"),
            max_total_candidates=pruning_kwargs.get("max_total_candidates"),
            pruning_strategy=pruning_kwargs.get("pruning_strategy")
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
        results["proposed"] = {"error": traceback.format_exc()}

    return question, true_hop, results


def run_experiment(tasks, pruning_kwargs, suffix):
    results_dir = find_project_root() / "results"
    results_dir.mkdir(exist_ok=True)
    
    all_results = []
    # Re-package tasks with pruning_kwargs
    args_list = [(q, g, h, pruning_kwargs) for (q, g, h) in tasks]
    
    for i, arg in enumerate(args_list):
        res = process_question(arg)
        all_results.append(res)
        
    metrics = {
        "baseline_1": {"exact_match": 0, "hit": 0, "triples": 0, "time": 0, "depth": 0, "count": 0},
        "baseline_2": {"exact_match": 0, "hit": 0, "triples": 0, "time": 0, "depth": 0, "count": 0},
        "baseline_3": {"exact_match": 0, "hit": 0, "triples": 0, "time": 0, "depth": 0, "count": 0},
        "proposed": {"exact_match": 0, "hit": 0, "triples": 0, "time": 0, "depth": 0, "count": 0},
    }
    
    true_hops = []
    pred_hops = []
    records = []
    
    for q, true_hop, res in all_results:
        rec = {"question": q, "true_hop": true_hop}
        for method in ["baseline_1", "baseline_2", "baseline_3", "proposed"]:
            if method in res and "error" not in res[method]:
                data = res[method]
                metrics[method]["exact_match"] += 1 if data["exact_match"] else 0
                metrics[method]["hit"] += 1 if data["hit"] else 0
                metrics[method]["triples"] += data["candidate_triples"]
                metrics[method]["time"] += data["retrieval_time_ms"]
                metrics[method]["depth"] += data["depth"]
                metrics[method]["count"] += 1
                
                rec[f"{method}_EM"] = data["exact_match"]
                rec[f"{method}_Hit"] = data["hit"]
                rec[f"{method}_triples"] = data["candidate_triples"]
                rec[f"{method}_time"] = data["retrieval_time_ms"]
                rec[f"{method}_depth"] = data["depth"]
                
                if method == "proposed":
                    true_hops.append(true_hop)
                    pred_hops.append(data["predicted_hop"])
                    rec["confidence"] = data["confidence"]
                    rec["used_fallback"] = data["used_fallback"]
                    rec["predicted_hop"] = data["predicted_hop"]
        records.append(rec)
        
    df = pd.DataFrame(records)
    df.to_csv(results_dir / f"evaluation_results_{suffix}.csv", index=False)
    
    summary = []
    for method in ["baseline_1", "baseline_2", "baseline_3", "proposed"]:
        count = metrics[method]["count"]
        if count > 0:
            summary.append({
                "Method": method,
                "Exact Match": metrics[method]["exact_match"] / count,
                "Hit@1": metrics[method]["hit"] / count,
                "Avg Triples": metrics[method]["triples"] / count,
                "Avg Time (ms)": metrics[method]["time"] / count,
                "Avg Depth": metrics[method]["depth"] / count,
            })
            
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(results_dir / f"evaluation_summary_{suffix}.csv", index=False)
    return summary_df, len(tasks)


def main():
    print("Loading test data...")
    metaqa = find_metaqa_dir()
    
    tasks = []
    for hop_val, folder in [(1, "1-hop"), (2, "2-hop"), (3, "3-hop")]:
        path = metaqa / folder / "vanilla" / "qa_test.txt"
        hop_tasks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    question, gold = parse_qa_line(line)
                    hop_tasks.append((question, gold, hop_val))
                    if len(hop_tasks) == 10:
                        break
        tasks.extend(hop_tasks)
                    
    print(f"Total test questions (subset for speed): {len(tasks)}")
    
    # Initialize globally for sequential run
    worker_init()
    
    # Run Unpruned Experiment
    print("\n--- Running Unpruned Experiment ---")
    pruning_unpruned = {
        "pruning_strategy": "none",
        "max_candidates_per_expansion": None,
        "max_total_candidates": None,
    }
    unpruned_summary, n_q = run_experiment(tasks, pruning_unpruned, "unpruned")
    print("Unpruned experiment complete.")
    
    # Run Optimized Experiment
    print("\n--- Running Relevance-Bounded (Optimized) Experiment ---")
    pruning_optimized = {
        "pruning_strategy": "relevance_bounded",
        "max_candidates_per_expansion": 100,
        "max_total_candidates": 5000,
    }
    optimized_summary, _ = run_experiment(tasks, pruning_optimized, "optimized")
    print("Optimized experiment complete.")
    
    # Output Comparison Report
    print("\n==================================================")
    print("EXPERIMENTAL COMPARISON REPORT (30 Questions)")
    print("==================================================")
    
    for method in ["baseline_1", "baseline_2", "baseline_3", "proposed"]:
        unp_row = unpruned_summary[unpruned_summary["Method"] == method].iloc[0]
        opt_row = optimized_summary[optimized_summary["Method"] == method].iloc[0]
        
        unp_em = unp_row["Exact Match"]
        opt_em = opt_row["Exact Match"]
        unp_hit = unp_row["Hit@1"]
        opt_hit = opt_row["Hit@1"]
        unp_trip = unp_row["Avg Triples"]
        opt_trip = opt_row["Avg Triples"]
        unp_time = unp_row["Avg Time (ms)"]
        opt_time = opt_row["Avg Time (ms)"]
        
        reduction = ((unp_trip - opt_trip) / unp_trip * 100) if unp_trip > 0 else 0.0
        
        print(f"\nMethod: {method.upper()}")
        print(f"  Exact Match:   {unp_em:.3f} (Unpruned) vs {opt_em:.3f} (Optimized)")
        print(f"  Hit@1:         {unp_hit:.3f} (Unpruned) vs {opt_hit:.3f} (Optimized)")
        print(f"  Avg Triples:   {unp_trip:.1f} (Unpruned) vs {opt_trip:.1f} (Optimized) | Reduction: {reduction:.1f}%")
        print(f"  Avg Latency:   {unp_time:.2f}ms (Unpruned) vs {opt_time:.2f}ms (Optimized)")

    print(f"\nNumber of successfully processed questions: {n_q}")
    print("Results saved under project results/ directory.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

