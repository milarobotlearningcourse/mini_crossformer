#/bin/bash
## Code to run all the jobs for this codebase 

## Atari RL envs
strings=(
    # "MontezumaRevengeNoFrameskip-v4"
    # "AsterixNoFrameskip-v4"
    # "batch_size=128"
    # "batch_size=256"
    "policy.obs_stacking=1"
    "policy.obs_stacking=2"
    "policy.obs_stacking=3"
    # "batch_size=512"
    # "PitfallNoFrameskip-v4"
    # "PhoenixNoFrameskip-v4"
)
echo "Running jobs for: $1"
for env in "${strings[@]}"; do
    echo "$env"
    # sbatch --array=1-3 --export=ALL,ENV_ID=$env,ARGSS='max_iters=50000' launchGPU.sh
    sbatch --array=1-2 --export=ALL,ENV_ID=$env,ARGSS="max_iters=50000 experiment.name=$1 $env" launchGPU.sh
done
