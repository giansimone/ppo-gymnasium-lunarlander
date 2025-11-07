[![License](https://img.shields.io/github/license/giansimone/ppo-gymnasium-lunarlander)](https://github.com/giansimone/ppo-gymnasium-lunarlander/blob/main/LICENSE)

# Proximal Policy Optimization (PPO) for Gymnasium's LunarLander Environment

A PyTorch implementation of the Proximal Policy Optimization (PPO) algorithm applied to the LunarLander environment from Gymnasium. This repository provides a simple and effective way to train an agent to land a lunar module safely.

## Installation

You can clone the repository and install the dependencies using Poetry:

```bash
git clone https://github.com/giansimone/ppo-gymnasium-lunarlander.git
cd ppo-gymnasium-lunarlander
poetry install
```

Alternatively, you can clone the repository and install the dependencies locally using pip:

```bash
git clone https://github.com/giansimone/ppo-gymnasium-lunarlander.git
cd ppo-gymnasium-lunarlander
pip install -e .
```

## Project Structure

```bash
ppo-gymnasium-lunarlander/
├── ppo_gymnasium_lunarlander/
│   ├── __init__.py
│   ├── agent.py
│   ├── buffer.py
│   ├── config.yaml
│   ├── enjoy.yaml
│   ├── environment.py
│   ├── export.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

## Quick Start

### Training

Train a PPO agent with the default configuration:

```bash
python -m ppo_gymnasium_lunarlander.train
```

### Configuration

Edit ```config.yaml``` to customise training parameters such as learning rate, number of episodes, and more.

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

Watch a trained agent by running the enjoy script:

```bash
python -m ppo_gymnasium_lunarlander.enjoy \
    --artifact-path runs/ppo_YYYY-MM-DD_HHhMMmSSs/final_model.pt \
    --num-episodes 5
```
### Exporting to Hugging Face Hub

Share your trained model on the Hugging Face Hub:

```bash
python -m ppo_gymnasium_lunarlander.export \
    --username YOUR_HF_USERNAME \
    --repo-name ppo-gymnasium-lunarlander-v3 \
    --artifact-path runs/ppo_YYYY-MM-DD_HHhMMmSSs/final_model.pt \
    --movie-fps 30 
```

This will:

- Create a repository on Hugging Face Hub.
- Upload the model weights, configuration, and evaluation results.
- Generate and upload a replay movie.
- Create a model card with usage instructions.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
