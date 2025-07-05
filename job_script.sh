#!/bin/bash
#SBATCH --job-name=bo_exp
#SBATCH --output=logs/bo_%A_%a.out
#SBATCH --error=logs/bo_%A_%a.err
#SBATCH --array=0-359
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

#SBATCH --mail-type=END,FAIL                # Mail if jobs end, or fail
#SBATCH --mail-user=suzuekar@msu.edu

module load anaconda3/2022.05
conda activate bo_exp

# Define grid (3 datasets × 6 methods × 20 replicates = 360 total)
DATASETS=(0 1 2)
METHODS=(bo_5 bo_10 bo_20 bo_50 random default)

# Decode the SLURM_ARRAY_TASK_ID
# ID = DATASET_IDX * 120 + METHOD_IDX * 20 + REPLICATE
ID=$SLURM_ARRAY_TASK_ID
DATASET_IDX=$((ID / 120)) # 360 / 3 = 120 chunks per dataset
METHOD_IDX=$(((ID % 120) / 20)) # 120 / 6 = 20 chunks per method
REPLICATE=$((ID % 20)) # 20 replicates per method per dataset

DATASET=${DATASETS[$DATASET_IDX]}
METHOD=${METHODS[$METHOD_IDX]}

echo "Running: dataset=${DATASET}, method=${METHOD}, replicate=${REPLICATE}"

# Run your experiment
python init_bo_experiments.py \
    --dataset $DATASET \
    --method $METHOD \
    --replicate $REPLICATE \
    --evaluations 500
