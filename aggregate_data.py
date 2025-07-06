"""
This computes average test_accuracy_score and final_cv_accuracy_score per method per dataset across replicates
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


dataset_idx = 0 # [0, 1, 2]
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


# Boxplot of cv and test accuracy scores, grouped by each method
methods = list(scores.keys())
cv_data = [scores[m]["cv"] for m in methods]
test_data = [scores[m]["test"] for m in methods]

print(cv_data)
print(test_data)

x = np.arange(len(methods)) # [0, 1, 2, 3, 4, 5, 6]
width = 0.35  # width of each boxplot

fig, ax = plt.subplots(figsize=(12, 6))

# Plot CV accuracy
bp1 = ax.boxplot(cv_data,
                 positions=x - width/2, # left of center
                 widths=width,
                 patch_artist=True,
                 boxprops=dict(facecolor="lightblue"),
                 medianprops=dict(color="black"))

# Plot Test accuracy
bp2 = ax.boxplot(test_data,
                 positions=x + width/2, # right of center
                 widths=width,
                 patch_artist=True,
                 boxprops=dict(facecolor="salmon"),
                 medianprops=dict(color="black"))

# Set x-axis
ax.set_xticks(x) # set center position
ax.set_xticklabels(methods, rotation=45) # label each "tick", rotate label by 45

# Add legend 
ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
          ["CV Accuracy", "Test Accuracy"],
          loc="upper right")

ax.set_ylabel("Accuracy")
ax.set_title(f"CV vs Test Accuracy per Method (Dataset {dataset_idx})")
ax.grid(True)
plt.tight_layout()
plt.savefig(json_dir/f"boxplot_{dataset_idx}.png", dpi=300)
plt.close()