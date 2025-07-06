"""
This computes average test_accuracy_score and final_cv_accuracy_score per method per dataset across replicates
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np


dataset_idx = 2 # [0, 1, 2]
# methods = ["bo_5", "bo_10", "bo_20", "bo_50", "random", "default"]
json_dir = Path(f"bo_experiments/{dataset_idx}")


# Initialize dictionary to collect scores per method
scores = defaultdict(lambda: {"cv": [], "test": []})

# print(json_dir.glob("*.json"))
# Iterate through all .json files
for json_file in json_dir.glob("*.json"):
   #  print(json_file)
    with open(json_file, "r") as f:
        result = json.load(f)
        method = result["method"]
        scores[method]["cv"].append(result["final_cv_accuracy_score"])
        scores[method]["test"].append(result["test_accuracy_score"])
    

# for method in scores:
#     print("Method: ", method)
#     cv_list = scores[method]["cv"]
#     print(len(cv_list))
#     for val in cv_list:
#         print(val, ", ")


# Compute average scores
averages = {}
for method, vals in scores.items():
    averages[method] = {
        "avg_cv_accuracy": float(np.mean(vals["cv"])),
        "avg_test_accuracy": float(np.mean(vals["test"]))
    }

for method in averages:
    print("Method: ", method)
    avg_cv_list = averages[method]["avg_cv_accuracy"]
    avg_test_list = averages[method]["avg_test_accuracy"]
    print("Average CV Accuracy: ", avg_cv_list)
    print("Average Test Accuracy: ", avg_test_list)


with open(json_dir/f"avg_accuracies.json", "w") as f:
    json.dump(averages, f, indent=2)