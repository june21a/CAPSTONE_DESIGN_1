#!/bin/bash
#SBATCH --job-name=tf_wp_010_0
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=/mnt/lustre/work/geiger/bjaeger25/garage_2_cleanup/results/logs/tfpp_wp_010_0_%a_%A.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/lustre/work/geiger/bjaeger25/garage_2_cleanup/results/logs/tfpp_wp_010_0_%a_%A.err   # File to which STDERR will be written
#SBATCH --partition=a100-galvani

# IMPORTANT: Start this script from within team_code folder, otherwise it will not work

# print info about current job
scontrol show job $SLURM_JOB_ID

pwd
export CARLA_ROOT=/home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/carla
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/envs/garage_2/bin/python

export OMP_NUM_THREADS=32  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=1 --max_restarts=0 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    train.py --id train_no_target_speed --use_disk_cache 0 --crop_image 1 --seed 0 --epochs 31 --batch_size 16 --lr 3e-4 --setting all \
    --root_dir /home/ec2-user/AD_challenge/CAPSTONE_DESIGN_1/carla_garage/training_data \
    --logdir /home/ec2-user/AD_challenge/experiments \
    --use_controller_input_prediction 0 --use_wp_gru 1 --continue_epoch 0 --cpu_cores 32 --num_repetitions 1 --use_cosine_schedule 1 --cosine_t0 1 \
    --image_architecture regnety_032 --lidar_architecture regnety_032
