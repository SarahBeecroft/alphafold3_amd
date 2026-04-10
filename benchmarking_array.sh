#!/bin/bash -l
#SBATCH --job-name=af3
#SBATCH --account=
#SBATCH --partition=gpu
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=1-13
# #SBATCH --mail-user=
# #SBATCH --mail-type=END,FAIL

module load singularity/4.1.0-nohost

#export XLA_PYTHON_CLIENT_PREALLOCATE=true
#export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98
export SINGULARITYENV_XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=1"
#export HSA_XNACK=0

DATADIR="${MYSCRATCH}/af3_inputs"
#OUTDIR="${MYSCRATCH}/af3_benchmarks"
INPUT_FILE="${DATADIR}/2pv7_data_${SLURM_ARRAY_TASK_ID}mer.json"
VRAM_LOG="vram_logs/vram_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"


#mkdir -p ${OUTDIR}
mkdir -p vram_logs

echo "=== Test ${SLURM_ARRAY_TASK_ID}: ${INPUT_FILE} ==="
echo "Running AlphaFold3 with the following environment variables:"
echo "XLA_PYTHON_CLIENT_PREALLOCATE: $XLA_PYTHON_CLIENT_PREALLOCATE"
echo "XLA_PYTHON_CLIENT_ALLOCATOR: $XLA_PYTHON_CLIENT_ALLOCATOR"
echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
echo "SINGULARITYENV_XLA_FLAGS: $SINGULARITYENV_XLA_FLAGS"
echo "Also set --num_diffusion_samples=1 --num_recycles=3 for this run"
echo "HSA_XNACK: $HSA_XNACK"
echo "Running on $(hostname)"
echo "Using input file: ${INPUT_FILE}"

# Start VRAM monitoring in background (poll every 60 seconds)
(
  while true; do
    rocm-smi --showmemuse --csv | tail -n +2 >> "${VRAM_LOG}"
    sleep 60
  done
) &
VRAM_MONITOR_PID=$!

START=$(date +%s)
echo "Start: $(date)"

srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 \
singularity exec \
-B ${MYSCRATCH}/george_af2/run_alphafold.py:/app/alphafold/run_alphafold.py \
-B ${MYSCRATCH}/george_af2/src/alphafold3/model/model_config.py:/alphafold3_venv/lib/python3.12/site-packages/alphafold3/model/model_config.py \
-B ${MYSCRATCH}/george_af2/src/alphafold3/model/components/utils.py:/alphafold3_venv/lib/python3.12/site-packages/alphafold3/model/components/utils.py \
${MYSCRATCH}/af3_751a4b8.sif \
python ${MYSCRATCH}/george_af2/run_alphafold.py \
--json_path=${INPUT_FILE} \
--model_dir=${MYSCRATCH}/george_af2/models/ \
--norun_data_pipeline \
--output_dir=2pv7_data_${SLURM_ARRAY_TASK_ID}mer/ \
--flash_attention_implementation xla --num_diffusion_samples=1 --num_recycles=3

END=$(date +%s)
ELAPSED=$((END - START))
MINS=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))

# Stop VRAM monitoring
kill ${VRAM_MONITOR_PID} 2>/dev/null
wait ${VRAM_MONITOR_PID} 2>/dev/null

# Extract peak VRAM % usage from log
PEAK_VRAM_PERCENT=$(awk -F',' '{print $2}' "${VRAM_LOG}" | sort -n | tail -1)
echo "Peak VRAM usage: ${PEAK_VRAM_PERCENT}%"

echo "End: $(date)"
echo "Walltime: ${MINS}m ${SECS}s (${ELAPSED}s total)"

SUMMARY="summary.csv"
[ ! -f "${SUMMARY}" ] && echo "INPUT_FILE,TOKEN_SIZE,ELAPSED_TIME,PEAK_VRAM_PERCENT" > "${SUMMARY}"
echo "${INPUT_FILE},${ELAPSED},${PEAK_VRAM_PERCENT}" >> "${SUMMARY}"
