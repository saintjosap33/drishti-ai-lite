🚁 Drishti-AI 

Autonomous Drone Surveillance using Reinforcement Learning and YOLOv8

📌 Overview

Drishti-AI is an autonomous UAV system integrating PPO-based reinforcement learning for navigation with YOLOv8 for real-time human detection.

It enables intelligent decision-making through a closed perception–action loop.

Based on an IEEE-published research work.

🧠 Key Features
PPO-based autonomous navigation
Real-time human detection (YOLOv8)
Closed-loop perception → decision system
Performance evaluation using training metrics
🏗️ Pipeline

AirSim simulation → State observation → YOLOv8 detection → PPO policy → Action execution → Reward feedback

📁 Project Structure

airsim_env/   → UAV simulation setup  

training/     → Reinforcement learning training scripts  

YoloV8/       → Human detection module  

evaluation/   → Evaluation scripts and outputs  

graphs/       → RL and YOLO performance metrics  

logs/         → Lightweight training logs  

assets/       → Architecture diagram  
📊 Results
Stable PPO convergence
Accurate human detection in simulation
📈 Metrics
RL Reward: 347.0
Precision: 95.6%
Recall: 87.8%
mAP@0.5: 93.8%
⚙️ Tech Stack

Python • PyTorch • OpenCV • AirSim • YOLOv8 • Stable-Baselines3

🚀 Setup
git clone https://github.com/saintjosap33/drishti-ai-lite.git
cd drishti-ai-lite
pip install -r requirements.txt
📄 Research

IEEE CINS 2025
Drishti-AI: Drone-based RL System for Intelligent Human Tracking and Identification

🎯 Applications
Defense surveillance
Autonomous UAV systems
Intelligent monitoring
👨‍💻 Author

Adithya J
B.Tech CSE, VIT Chennai