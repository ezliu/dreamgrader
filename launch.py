import time
import subprocess
import argparse

script = """\
#!/bin/bash
#SBATCH --partition=iris --qos=normal
#SBATCH --time=300:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1

#SBATCH --job-name="sample"
#SBATCH --output=sbatch/logs/sample-%j.out

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# Load environment
source /iris/u/moritzst/miniwob_venv/bin/activate

free -h
nvidia-smi
which python3
python3 --version

# Print command
echo "Running the following command:"
echo "python3 main.py {exp_name} -b instruction_agent.policy.type=\\"classifier\\" -c configs/default.json -c configs/fake_scroll_miniwob.json -s {seed}"

# Run command
python3 main.py {exp_name} -b instruction_agent.policy.type=\\"classifier\\" -c configs/default.json -c configs/fake_scroll_miniwob.json -s {seed}

# Done
echo "Done"
"""

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
        '-n', '--name', required=True, type=str)
arg_parser.add_argument(
        "-s", "--seed", default=0, help="random seed to use.", type=int)
args = arg_parser.parse_args()

print("Using seed {seed}".format(seed=args.seed))

TMP = "tmp.sh"
partition = "iris-hi"

with open(TMP, "w") as f:
    f.write(script.format(
        exp_name=f"{args.name}",
        seed=args.seed))

cmd = "sbatch --account=iris -p {partition} --time 300:00:00 --job-name=dream-miniwob-{exp_name} --nodelist=iris1,iris2,iris3 --output=sbatch/{exp_name}.txt {tmp}".format(partition=partition, exp_name=args.name, tmp=TMP)
subprocess.run(cmd, check=True, shell=True)
time.sleep(2)
