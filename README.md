# Gymnasium_RL: Reinforcement Learning in Gymnasium Environments

This repository contains implementations of reinforcement learning (RL) algorithms applied to classic environments from the [Gymnasium](https://gymnasium.farama.org/) framework. It includes solutions for two popular problems: **CartPole** and **FrozenLake**, showcasing Q-Learning.

---

## 📝 Project Overview

The project provides:

1. A **CartPole-v1** implementation using a discretized state space for Q-learning.
2. A **FrozenLake-v1** implementation with Q-learning for solving the slippery grid problem.
3. Modular, reusable code for training, evaluation, and visualization.
4. Saved Q-tables and training results for further analysis.

---

## 🔧 Features

- **CartPole-v1**: Balance a pole on a cart using Q-learning with discretized state spaces.
- **FrozenLake-v1**: Navigate a slippery grid world to reach the goal while avoiding holes.
- **Training and Evaluation**: Adjustable training episodes, exploration parameters, and learning rates.
- **Visualization**: Plot cumulative rewards over episodes.
- **Q-Table Persistence**: Save and load Q-tables using `pickle` for reuse.

---

## 📂 Repository Structure

```plaintext
Gymnasium_RL/
├── cartpole.py         # Q-learning implementation for CartPole-v1
├── frozen_lake.py      # Q-learning implementation for FrozenLake-v1
├── frozen_lake8x8.pkl  # Saved Q-table for FrozenLake
├── cartpole.pkl        # Saved Q-table for CartPole
├── cartpole.png        # Training reward plot for CartPole
├── frozen_lake8x8.png  # Training reward plot for FrozenLake
└── README.md           # Project documentation
