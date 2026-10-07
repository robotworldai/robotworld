# Towards Building AI-CPS with NVIDIA Isaac Sim: An Industrial Benchmark and Case Study for Robotics Manipulation
This folder contains all revelant code for the paper "Towards Building AI-CPS with NVIDIA Isaac Sim: An Industrial Benchmark and Case Study for Robotics Manipulation".

## Benchmark of Robotics Manipulation 

### Requirements:
1. Install Omniverse Isaac Sim: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_basic.html
2. Add Isaac Sim to PYTHON_PATH (with default installation location of ISAAC SIM)
   ```
   alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
   ```
    
2. Install Omniverse Isaac GYM Envs: https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs
3. Install SKRL, RTAMT, and Scipy in the Isaac Sim Python environment (the latter two are used for falsification): go to the Isaac folder, and run
   ```
   ./python.sh -m pip install skrl rtamt scipy
   ```
   
### Run the learning process:

To run SKRL with provided task environments (example):
```
cd Gym_Envs/
PYTHON_PATH skrl_train_PPO.py task=FrankaBallBalancing num_envs=16 headless=False
```

To launch Tensorboard: 
```
PYTHON_PATH -m tensorboard.main --logdir runs/FrankaBallBalancing/summaries/
```

## Falsification Tool
To run the falsification test for pre-trained agent, run:
```
cd Falsification_Tool/
PYTHON_PATH manipulator_testing.py headless=False
```

## Performance Evaluation
The performance evaluation uses the same framework as the falsification tool, but with the optimizer set to "random":
```
cd Evaluation/
PYTHON_PATH manipulator_eval.py headless=False
```
