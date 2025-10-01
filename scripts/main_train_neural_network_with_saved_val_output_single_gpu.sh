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

DATA_FILE_PATH="../data/randomSeq.txt"
TRAIN_PROP=0.90
NUM_OF_EPOCHS=25
LEARNING_RATE=0.001
BATCH_SIZE=36
EMBED_DIM=32
FEED_FWD_DIM=16
KMER_LENGTH=8

PYTHON_FILE="../python/main_train_neural_network_with_saved_val_output_single_gpu.py"

# Execute workflow
python ${PYTHON_FILE} ${DATA_FILE_PATH} ${TRAIN_PROP} ${NUM_OF_EPOCHS} ${LEARNING_RATE} ${BATCH_SIZE} ${EMBED_DIM} ${FEED_FWD_DIM} ${KMER_LENGTH}

conda deactivate
module unload anaconda3/2023.09
END_TIME_SECONDS=$(date +'%s')

echo ""
echo "======================================================"
echo "End Time: $(date)"
echo "It took $((END_TIME_SECONDS - START_TIME_SECONDS)) seconds to train shallow ANN model using data from ${DATA_FILE_PATH}."
echo "======================================================"
