#! /bin/bash

#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --time=6:00:00

echo "======================================================"
echo "Start Time  : $(date)"
echo "Submit Dir  : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Node List   : $SLURM_JOB_NODELIST"
echo "Num Tasks   : $SLURM_NTASKS total [$SLURM_NNODES nodes @ $SLURM_CPUS_ON_NODE CPUs/node]"
echo "======================================================"
echo ""

# Example submission of SLURM job
# sbatch main_perform_permanova_on_numpy_file.sh 0 4000

START_TIME_SECONDS=$(date +'%s')
source /apps/pkg/anaconda3/2023.09/etc/profile.d/conda.sh
module load anaconda3/2023.09
conda activate kmeranalysisenv

# Set classification parameters
LOGITS_FILE_PATH=../data/training_dataset_val_output_logits.npy
LABELS_FILE_PATH=../data/training_dataset_val_output_labels.npy
START_INDEX=${1}
END_INDEX=${2}
NUM_OF_PCA_COMP=2
PERMUTATIONS=999
OUTPUT_DIR_PATH=../data
PYTHON_FILE=../python/main_perform_permanova_on_numpy_file.py

# Parse files within the target directory
python ${PYTHON_FILE} ${LOGITS_FILE_PATH} ${LABELS_FILE_PATH} ${START_INDEX} ${END_INDEX} ${NUM_OF_PCA_COMP} ${PERMUTATIONS} ${OUTPUT_DIR_PATH}

conda deactivate
module unload anaconda3/2023.09
END_TIME_SECONDS=$(date +'%s')

echo ""
echo "======================================================"
echo "End Time: $(date)"
echo "It took $((END_TIME_SECONDS - START_TIME_SECONDS)) seconds to perform PCA and PERMANOVA for input logits numpy file ${LOGITS_FILE_PATH} and class labels file ${LABELS_FILE_PATH}."
echo "======================================================"
