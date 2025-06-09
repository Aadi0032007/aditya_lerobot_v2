# -*- coding: utf-8 -*-
"""
Created on Tue May 13 01:01:31 2025

@author: aadi
"""
"""


python lerobot/scripts/control_robot.py \
  --robot.type=revobots \
  --control.type=teleoperate \
  --control.display_data=true
  
  
  
python lerobot/scripts/control_robot.py \
    --robot.type=revobot  \
    --control.type=record \
    --control.single_task="Test the system."  \
    --control.fps=30 \
    --control.repo_id=test \
    --control.tags='["tutorial"]' \
    --control.warmup_time_s=5 \  
    --control.episode_time_s=30 \
    --control.reset_time_s=0 \
    --control.num_episodes=2  \
    --control.push_to_hub=false
    
  
  
python lerobot/scripts/train.py \
    --dataset.repo_id=test  \
    --policy.type=act \
    --output_dir=outputs/train/test_policy \
    --job_name=test_policy  \
    --policy.device=cuda



  
  
  """