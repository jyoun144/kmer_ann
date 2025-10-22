#! /bin/bash

#SBATCH --partition=Nebula
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
module load anaconda3/2023.09

# Retrieve input parameters
SAMPLE_SIZE=${1}
SEED_VALUE=${2}
OUTPUT_FILE_PATH=${3}
PYTHON_FILE=../python/main_generate_tng_and_val_datasets.py

# Parse files within the target directory
python ${PYTHON_FILE} ${SAMPLE_SIZE} ${SEED_VALUE} ${OUTPUT_FILE_PATH}

module unload anaconda3/2023.09
END_TIME_SECONDS=$(date +'%s')

echo ""
echo "======================================================"
echo "End Time: $(date)"
echo "It took $((END_TIME_SECONDS - START_TIME_SECONDS)) seconds to generate random dataset and save file at ${OUTPUT_FILE_PATH}."
echo "======================================================"
