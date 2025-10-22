#!/bin/bash

#SBATCH --partition=Nebula_GPU
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=1       # cpus per task
#SBATCH --mem=96GB              # total memory per nodei
#SBATCH --gres=gpu:L40S:1         # number of allocated gpus per node
#SBATCH --time=17:00:00          # total run time limit (HH:MM:SS)

echo "======================================================"
echo "Start Time  : $(date)"
echo "Submit Dir  : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Node List   : $SLURM_JOB_NODELIST"
echo "Num Tasks   : $SLURM_NTASKS total [$SLURM_NNODES nodes @ $SLURM_CPUS_ON_NODE CPUs/node]"
echo "======================================================"
echo ""
START_TIME_SECONDS=$(date +'%s')


# Set up working environment
module load anaconda3/2023.09
source /apps/pkg/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate DNABERT_S

NUM_WORKERS=${SLURM_CPUS_PER_TASK}

echo "CPUs per task: ${NUM_WORKERS}"

TNG_FILE_PATH="../data/training_dataset_seed_1151.tsv"
VAL_FILE_PATH="../data/validation_dataset_seed_3302.tsv"
NUM_OF_EPOCHS=15
LEARNING_RATE=0.01
BATCH_SIZE=36
EMBED_DIM=32
FEED_FWD_DIM=16
OUTPUT_DIM=4
KMER_LENGTH=8

PYTHON_FILE="../python/main_train_neural_network_with_saved_val_output_single_gpu_V2.py"

# Execute workflow
python ${PYTHON_FILE} ${TNG_FILE_PATH} ${VAL_FILE_PATH} ${NUM_OF_EPOCHS} ${LEARNING_RATE} ${BATCH_SIZE} ${EMBED_DIM} ${FEED_FWD_DIM} ${OUTPUT_DIM} ${KMER_LENGTH} ${NUM_WORKERS}

conda deactivate
module unload anaconda3/2023.09
END_TIME_SECONDS=$(date +'%s')

echo ""
echo "======================================================"
echo "End Time: $(date)"
echo "It took $((END_TIME_SECONDS - START_TIME_SECONDS)) seconds to train shallow ANN model using data from ${TNG_FILE_PATH}."
echo "======================================================"
