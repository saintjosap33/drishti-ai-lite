# **Drishti-AI**

##### 

##### Autonomous Drone Surveillance using Reinforcement Learning and YOLOv8



Overview



Drishti-AI is an autonomous UAV system that integrates Proximal Policy Optimization (PPO) for navigation with YOLOv8 for real-time human detection.



The system operates through a closed perception–action loop, enabling intelligent and adaptive decision-making in dynamic environments. This work is based on an IEEE-published research study and focuses on building a simulation-driven autonomous surveillance framework.



Key Features

PPO-based autonomous navigation

Real-time human detection using YOLOv8

Closed-loop perception and decision-making system

Performance evaluation using structured training metrics

Pipeline



AirSim Simulation → State Observation → YOLOv8 Detection → PPO Policy → Action Execution → Reward Feedback



Project Structure

airsim\_env/   → UAV simulation setup  

training/     → Reinforcement learning training scripts  

YoloV8/       → Human detection module  

evaluation/   → Evaluation scripts and outputs  

graphs/       → RL and YOLO performance metrics  

logs/         → Lightweight training logs  

assets/       → Architecture diagram  

Results



The system demonstrates stable reinforcement learning convergence along with accurate human detection in a simulated environment.



Metrics

RL Reward: 347.0

Precision: 95.6%

Recall: 87.8%

mAP@0.5: 93.8%

Tech Stack

Python

PyTorch

OpenCV

AirSim

YOLOv8

Stable-Baselines3

Setup

git clone https://github.com/saintjosap33/drishti-ai-lite.git



cd drishti-ai-lite



pip install -r requirements.txt

Research



IEEE CINS 2025

Drishti-AI: Drone-based RL System for Intelligent Human Tracking and Identification



Applications

Defense surveillance

Autonomous UAV systems

Intelligent monitoring and tracking

Author



Adithya J

B.Tech Computer Science and Engineering

VIT Chennai

