#!/bin/bash -l
#SBATCH --job-name=af3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=2:00:00
#SBATCH --account=pawsey0012-gpu  

module load singularity/4.1.0-nohost

input=5K_token_data.json
#input=2pv7_data_8mer.json
#input=2pv7_data.json
export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export XLA_PYTHON_CLIENT_ALLOCATOR=platform
unset XLA_PYTHON_CLIENT_ALLOCATOR
export HIP_VISIBLE_DEVICES=0,1
export SINGULARITYENV_XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter --xla_gpu_autotune_level=0"
#export SINGULARITYENV_XLA_FLAGS="--xla_gpu_autotune_level=0"
export "HSA_XNACK=1"
#export SINGULARITYENV_XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter,loop-pad-fusion,hlo_rematerialization --xla_gpu_autotune_level=0" 
echo "Running AlphaFold3 with the following environment variables:"
echo "XLA_PYTHON_CLIENT_PREALLOCATE: $XLA_PYTHON_CLIENT_PREALLOCATE"
echo "XLA_PYTHON_CLIENT_ALLOCATOR: $XLA_PYTHON_CLIENT_ALLOCATOR"
echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
echo "SINGULARITYENV_XLA_FLAGS: $SINGULARITYENV_XLA_FLAGS"
echo "HSA_XNACK: $HSA_XNACK"
echo "Running on "$(hostname)
echo "Using input file: $input"
#echo "Using additional flags: --num_diffusion_samples=1 --num_recycles=3"

srun -N 1 -n 1 -c 16 --gres=gpu:2 --gpus-per-task=2 \
singularity exec \
-B /scratch/pawsey0001/sbeecroft/george_af2/run_alphafold.py:/app/alphafold/run_alphafold.py \
-B /scratch/pawsey0001/sbeecroft/george_af2/src/alphafold3/model/model_config.py:/alphafold3_venv/lib/python3.12/site-packages/alphafold3/model/model_config.py \
-B /scratch/pawsey0001/sbeecroft/george_af2/src/alphafold3/model/components/utils.py:/alphafold3_venv/lib/python3.12/site-packages/alphafold3/model/components/utils.py \
af3_751a4b8.sif \
python run_alphafold.py \
--json_path=${MYSCRATCH}/george_af2/${input} \
--model_dir=/scratch/pawsey0001/sbeecroft/alphafold3/models/ \
--norun_data_pipeline \
--output_dir=${MYSCRATCH}/george_af2/af_output_large_${SLURM_JOB_ID} \
--flash_attention_implementation xla 
#--num_diffusion_samples=1 --num_recycles=3
