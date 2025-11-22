[![License](https://img.shields.io/github/license/giansimone/ppo-gymnasium-lunarlander)](https://github.com/giansimone/ppo-gymnasium-lunarlander/blob/main/LICENSE)

# Proximal Policy Optimization (PPO) for Gymnasium's LunarLander Environment

A PyTorch implementation of the Proximal Policy Optimization (PPO) algorithm applied to the LunarLander environment from Gymnasium. This repository provides a simple and effective way to train an agent to land a lunar module safely.

## Installation

You can clone the repository and install the required dependencies using Poetry or pip. This project requires **Python 3.13**.

### Using Poetry (Recommended)

```bash
# 1. Clone the repository
git clone [https://github.com/giansimone/ppo-gymnasium-lunarlander.git](https://github.com/giansimone/ppo-gymnasium-lunarlander.git)
cd ppo-gymnasium-lunarlander

# 2. Initialize environment and install dependencies
poetry env use python3.13
poetry install

# 3. Activate the shell
eval $(poetry env activate)
```

### Using Pip

```bash
# 1. Clone the repository
git clone [https://github.com/giansimone/ppo-gymnasium-lunarlander.git](https://github.com/giansimone/ppo-gymnasium-lunarlander.git)
cd ppo-gymnasium-lunarlander

# 2. Create and activate virtual environment
python3.13 -m venv venv
source venv/bin/activate

# 3. Install package in editable mode
pip install -e .
```

## Project Structure

```bash
ppo-gymnasium-lunarlander/
├── ppo_gymnasium_lunarlander/
│   ├── __init__.py
│   ├── agent.py       # PPO Agent implementation
│   ├── buffer.py      # Rollout Buffer
│   ├── config.yaml    # Training hyperparameters
│   ├── enjoy.py       # Script to enjoy a trained agent
│   ├── environment.py # Gym environment wrappers
│   ├── export.py      # Hugging Face export script
│   ├── model.py       # Actor/Critic Networks
│   ├── train.py       # Main training loop
│   └── utils.py
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

## Usage

Ensure you are in the ```ppo_gymnasium_lunarlander``` source directory where ```config.yaml``` is located before running these commands.

```bash
cd ppo_gymnasium_lunarlander
```

### Training

Train a PPO agent with the default configuration.

```bash
python -m train
```

### Configuration

Edit ```config.yaml``` to customise training parameters.

```yaml
#Environment
env_id: LunarLander-v3
num_envs: 16
normalise_obs: false

#Network Architecture
hidden_dim: 128

#Training
total_timesteps: 10_000_000
n_steps: 1024
batch_size: 64

#PPO Agent
learning_rate: 0.0003
gamma: 0.999
gae_lambda: 0.95
clip_epsilon: 0.2
value_coef: 0.5
entropy_coef: 0.01
ppo_epochs: 10
max_grad_norm: 0.5

#Logging
log_dir: runs/

#System
seed: 42
```

### Enjoying a Trained Agent

Watch a trained agent by running the enjoy script. Point the artifact argument to your saved model file.

```bash
python -m enjoy \
    --artifact runs/ppo_LunarLander-v3_YYYY-MM-DD_HHhMMmSSs/final_model.pt \
    --num-episodes 5
```
### Exporting to Hugging Face Hub

Share your trained model, config, and a replay video to the Hugging Face Hub.

```bash
python -m export \
    --username YOUR_HF_USERNAME \
    --repo-name ppo-gymnasium-lunarlander-v3 \
    --artifact-path runs/ppo_LunarLander-v3_YYYY-MM-DD_HHhMMmSSs/final_model.pt \
    --movie-fps 30 \
    --n-eval 10
```

This will automatically:

- Upload the model weights and config.

- Generate a model card with evaluation metrics (Mean Reward +/- Std).

- Record and upload a video of the agent.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
