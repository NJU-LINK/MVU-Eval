import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_results(folder_path):
    """Load all model result jsonlines in the given folder."""
    records = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def compute_accuracy_per_category(records):
    """Compute accuracy by task and overall accuracy for each model."""
    stats = {}
    for r in records:
        qa_idx = r.get("QA_index")
        task = r.get("task", "Unknown")
        gt = r.get("ground_truth")
        model_results = r.get("model_results", {})
        for model_name, result in model_results.items():
            pred = str(result.get("model_output", "")).strip().upper()[:1]
            if model_name not in stats:
                stats[model_name] = {"correct": 0, "total": 0, "by_task": {}}
            stats[model_name]["total"] += 1
            stats[model_name]["by_task"].setdefault(task, {"correct": 0, "total": 0})
            stats[model_name]["by_task"][task]["total"] += 1
            if pred == gt:
                stats[model_name]["correct"] += 1
                stats[model_name]["by_task"][task]["correct"] += 1
    # compute rates
    summary = []
    for model, info in stats.items():
        overall_acc = 100 * info["correct"] / info["total"]
        row = {"Model": model, "Overall": overall_acc}
        for task, subinfo in info["by_task"].items():
            row[task] = 100 * subinfo["correct"] / subinfo["total"]
        summary.append(row)
    df = pd.DataFrame(summary).fillna(0).sort_values("Overall", ascending=False)
    return df


if __name__ == "__main__":
    folder = "Model_output/max_pixel_720_nframes_32/QA_json_file/main"  # modify path if needed
    records = load_results(folder)
    df = compute_accuracy_per_category(records)
    print(df.to_markdown(index=False))
