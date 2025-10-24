# -*- coding: utf-8 -*-
"""
Created on Tue May 13 01:01:31 2025

@author: aadi
"""
"""

---------teleoperate local---------------
 
python lerobot/scripts/control_robot.py \
  --robot.type=revobot \
  --control.type=teleoperate \
  --control.display_data=true


------teleoperate local w/o cameras------

python lerobot/scripts/control_robot.py \
  --robot.type=revobot \
  --robot.cameras='{}' \
  --control.type=teleoperate


-----------------record------------------

python lerobot/scripts/control_robot.py \
    --robot.type=revobot  \
    --control.type=record \
    --control.single_task="Test the system."  \
    --control.fps=15 \
    --control.repo_id=debug_1 \
    --control.tags='["tutorial"]' \
    --control.warmup_time_s=5 \
    --control.episode_time_s=30 \
    --control.reset_time_s=0 \
    --control.num_episodes=2  \
    --control.push_to_hub=false
    
  
-------------train ---------------------

python lerobot/scripts/train.py \
    --dataset.repo_id=bartender_1_3cams  \
    --policy.type=act \
    --output_dir=outputs/train/bartender_1_3cams_act_100kl \
    --job_name=bartender_1_3cams_100kl  \
    --policy.device=cuda \
    --wandb.enable=true
    
  --policy.type=act \
  --policy.path=lerobot/pi0 \
     
      
------------restart train--------------
      
python lerobot/scripts/train.py \
    --config_path=outputs/train/bartender_bb/checkpoints/last/pretrained_model/ \
    --resume=true


------------mobile revobot---------------

python lerobot/scripts/control_robot.py \
    --robot.type=mobile_revobot \
    --control.type=remote_revobot  \
    --control.viewer_ip=192.168.0.96  \
    --control.viewer_port=1234

----------remote teleoperation-----------
python lerobot/scripts/control_robot.py \
    --robot.type=mobile_revobot  \
    --control.type=teleoperate \
    --control.fps=15 \
    --control.display_data=true

------------record with remote setup----------

python lerobot/scripts/control_robot.py \
    --robot.type=mobile_revobot  \
    --control.type=record \
    --control.single_task="Pick up bottle."  \
    --control.fps=15 \
    --control.repo_id=bartender_df_1 \
    --control.tags='["tutorial"]' \
    --control.warmup_time_s=10 \
    --control.episode_time_s=3600 \
    --control.reset_time_s=0 \
    --control.num_episodes=30  \
    --control.push_to_hub=false \
    --control.resume=true


------------calibration-----------------

python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_leader"]'


-------------replay-----------------------

python lerobot/scripts/control_robot.py \
  --robot.type=revobot \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=debug_2 \
  --control.episode=0
  
  
  """