#!/bin/bash
#SBATCH --account=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=35
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --mail-user=shashank.s@research.iiit.ac.in
#SBATCH --mail-type=ALL

eval "$(conda shell.bash hook)"
conda activate torchenv
cd /home/talsperre/repos/rl-algo-implementations/test
echo Ran the Bash script
python test_trainer.py --save_dir ../cache --warm_start 50000 --env_id "BreakoutDeterministic-v4" --debug True --lr 0.0001 --replay_size 1000000 --epsilon_decay 50000 --update_every 1000 --optimizer Adam
