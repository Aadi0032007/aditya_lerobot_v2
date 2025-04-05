#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 01:34:41 2024

@author: aadi
"""
"""
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --control.type=record \
  --control.fps=30 \
  --control.num_episodes=10 \
  --control.policy.path=home/revolabs/aditya/aditya_lerobot/outputs/train/act_koch_reach_the_object/checkpoints/last/pretrained_model
  
  
  python lerobot/scripts/control_robot.py \
     --robot.type=koch \
     --control.type=record \
     --control.fps=30 \
     --control.root=data \
     --control.repo_id=${HF_USER}/eval_koch_reach_the_object \
     --control.tags='["tutorial", "eval"]' \
     --control.warmup_time_s=5 \
     --control.episode_time_s=10 \
     --control.reset_time_s=30 \
     --control.num_episodes=1 \
     --control.policy.path=home/revolabs/aditya/aditya_lerobot/outputs/train/act_koch_reach_the_object/checkpoints/last/pretrained_model
 
 
 python lerobot/scripts/control_robot.py \
  --robot.type=revobot \
  --control.type=record_with_marker \
  --control.fps=30 \
  --control.root=data \
  --control.repo_id=${HF_USER}/koch_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=10 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.push_to_hub=false
  
  
  python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_reach_the_object \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_reach_the_object \
  --job_name=act_koch_test \
  --policy.device=cuda
  
  
  python lerobot/scripts/control_robot.py teleoperate \
  --robot-path lerobot/configs/robot/koch.yaml

python lerobot/scripts/control_robot.py --control.type=replay --robot.type=revobot --control.fps=30 --control.root=data --control.repo_id=koch_big_robot --control.episode=1 

"""
