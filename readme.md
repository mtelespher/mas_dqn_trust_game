# Exploring the Emergence of Trust between Agents using Deep Q-learning in a Multi-Agent System 


![MAS Network](network_image.jpg)

## Objectives

This codebase implements a progressive series of experiments extending the classical game theoretical framework called the 'Trust Game' to multi-agent environments. Each artificial agent in this environment learns through interactions via the reinforcement learning technique Deep Q-learning.

The objective of the research behind this codebase, was to investigate the emergence of cooperative behaviour or 'trust' norms between agents in this multi-agent system. Existing literature [1], indicated fixed pairs of agents that learn through the Deep Q-learning technique constructed in TensorFlow generate cooperative norms - when trained for ~300k steps. The rates of cooperation are similar to those of human participants [3].

A major drawback of the original paper was that when agents were paired randomly during training, cooperation collapsed to defection. This research aimed to expand upon the original paper by implementing random pairing of agents at the start of each round and increasing the number of agents interacting to a multi-agent system, including a reputation mechanism which was adapted by another paper which successfully indicated cooperation in a multi-agent reinforcement learning game theory applciation.

## The Trust Game

The Trust Game is a behavioral economics experiment designed to measure trust and reciprocity between participants. Introduced by Berg, Dickhaut, and McCabe (1995) [3], it involves two players: a **Trustor** and a **Trustee**.

**Game Mechanics:**
1. The Trustor receives an initial endowment (typically 10 units)
2. The Trustor decides how much to send to the Trustee (0-10 units)
3. Any amount sent is tripled by the experimenter
4. The Trustee decides how much of the tripled amount to return to the Trustor
5. Both players keep their final amounts

**Strategic Tension:**
- **Nash Equilibrium:** Trustor sends 0 (anticipating no return)
- **Observed Behavior:** Humans typically send ~5 units and return ~15-20% of received amount
- **Cooperation Dilemma:** Mutual benefit requires trust (sending) and reciprocity (returning)

**Research Significance:**
The Trust Game quantifies trust through monetary transfers, making it ideal for studying cooperation emergence in artificial agents. Unlike other game-theoretic frameworks, it captures the asymmetric nature of trust relationships and the role of vulnerability in cooperation.

This framework allows researchers to investigate how AI agents develop trust norms through reinforcement learning, providing insights into cooperation in decentralised systems.

## Deep Q-Learning (DQN)

Deep Q-Learning combines Q-learning reinforcement learning with deep neural networks to enable agents to learn optimal strategies in complex environments.

**Core Components:**

**Q-Learning Foundation:**
- Agents learn Q-values: Q(state, action) = expected future reward
- Bellman equation: Q(s,a) = reward + γ × max Q(next_state, all_actions)
- Epsilon-greedy exploration: balance between trying new actions vs. exploiting known good actions

**Deep Neural Network Enhancement:**
- Replaces lookup tables with neural networks for large state spaces
- Architecture: Input (state) → Hidden layers → Output (Q-values for each action)
- Enables learning in high-dimensional environments impossible with tabular methods

**Stabilization Techniques:**
- **Experience Replay:** Store past experiences and sample randomly for training (prevents correlation)
- **Target Networks:** Separate network for calculating target Q-values (prevents moving target problem)
- **Epsilon Decay:** Gradually reduce exploration as agent learns optimal policies

**State Representations:**
- Trustors: [previous_sent, previous_received] → Neural network → Q-values for send amounts (0-10)
- Trustees: [amount_received] → Neural network → Q-values for return amounts (0-30)

**Learning Process:**
1. Agent observes current state
2. Neural network outputs Q-values for all possible actions
3. Agent selects action (epsilon-greedy)
4. Environment provides reward and next state
5. Store experience in replay buffer
6. Sample random batch and update network weights using Bellman equation

**Multi-Agent Innovation:**
This research implements **partner-specific DQNs** where each agent maintains separate neural networks for every potential partner, enabling relationship-specific strategy development while preserving the benefits of deep learning.


## Methodology

A three phased experiment which focused on innovating and iterating proven frameworks for agent based cooperation [1] [2].

- Experiment 1: PyTorch Baseline Replication

Replication of Wu et al. (2023) framework for fixed pairs of agents playing the Trust Game to understand whether cooperation emergence is framework independent.

- Experiment 2: Random Pairing with Partner-Specific Networks

Creation of a solution to the lack of cooperation between random agents when randomly paired at the start of each round. This was achieved by implementing partner specific memory, neural networks for learning and epsilon decay to ensure that each partnership decayed at the same rate as the original paper.

- Experiment 3: Global Reputation System

After Experiment 2 indicated that cooperation with random agent pairs can generate cooperation, a multi-agent system was constructed. This was done by implementing a reputation mechanism, with agents being scored higher for cooperative actions, this reputation mechanism was applied to partner selection at the start of each round and as an input to the neural network behind the Deep Q-learning process.


## Experiment 1: PyTorch Baseline Replication
File: trust_game_mas_experiment_1.py
Replicates Wu et al. (2023) methodology using PyTorch instead of TensorFlow to validate framework independence.

### Output Files:

- baseline_training_analysis.png - Training curves and final results
Individual agent memory files

### Configuration:

Agents: 10 fixed trustor-trustee pairs
Training: 350k steps per pair with sequential training
State Spaces: Trustor [last_sent, last_received], Trustee [amount_received]
Objective: Demonstrate cooperation emergence matches literature benchmarks

## Experiment 2: Random Pairing with Partner-Specific Networks
File: trust_game_mas_experiment_2.py
Extends to multi-agent environment with dynamic partner selection while maintaining relationship-specific learning.

### Output Files:

- partnership_smooth_curves.png - Individual partnership learning
- population_smooth_curves.png - System-wide trends
- experiment_summary.pkl - Complete results data

### Configuration:

Agents: 10 trustors (IDs 0-9), 10 trustees (IDs 12-21)
Training: 30 - 43M steps with random pairing each round, to replicate the 300k training steps required for cooperation in Wu et al. Trust Game.
Architecture: Each agent maintains separate DQN for every potential partner (200 total networks)
Innovation: Partner-specific epsilon decay and experience replay memories

Partnership-specific learning enables relationship-dependent strategies
Random pairing tests cooperation emergence without fixed relationships
Comprehensive data collection tracks partnership evolution

## Experiment 3: Global Reputation System
File: trust_game_mas_experiment_3.py

Configuration:

Agents: 20 trustors (IDs 0-19), 20 trustees (IDs 20-39)
Training: 10M steps with reputation-weighted partner selection
Reputation: Modified Ren-Zeng system with 0-10 scale, cooperation-only scoring
Selection: Weighted probability based on reputation 

Output Files:

- agent_data.csv - Final agent states (wealth, reputation)
- partnership_interactions.csv - All transactions with timestamps
- partnership_summary.csv - Aggregated partnership statistics
- cooperation_timeline.csv - System evolution metrics

## Key Innovations:

Reputation-Enhanced States: [base_observation, partner_reputation]
Wealth Tracking: Persistent accumulation from zero initial endowment
Cooperation Scoring: Gaussian bell curve rewarding optimal return rates (~50%)
Social Stratification: High-reputation agents preferentially selected

git clone [repository-url]
cd multi-agent-trust-game

## Install dependencies
pip install -r requirements.txt

## Verify CUDA (optional but recommended)
python -c "import torch; print(torch.cuda.is_available())"

## Learning Dynamics

Partner-specific networks enable relationship-dependent strategies
Random pairing with memory preserves cooperation incentives
Reputation-based selection amplifies cooperation differences
Technical Requirements

# System Requirements:

RAM: 16GB minimum, 32GB recommended for Experiment 3
GPU: CUDA-capable GPU with 8GB+ VRAM (optional but reduces training time 10x)
Storage: 5GB free space for results and model checkpoints

### Reproduction Notes
 
- All agents initialize with random weights for each run
- GPU based training significantly reduces training time (hours vs days)


## Key Findings

Experiment 1: Replicates Wu et al. [1] cooperation levels in PyTorch
Experiment 2: Cooperation emerges despite random pairing through partner-specific learning
Experiment 3: Reputation systems sustain cooperation, reputation is positive correlated with higher rates of cooperation and trustee agent selection. 


## Related Work

- 1 - Wu et al. (2023): "Building Socially Intelligent AI Systems: Evidence from the Trust Game Using Artificial Agents with Deep Learning" - Reference paper conducting Trust Game experiments with Deep Q-learning fixed agent pairs.
- 2 - Ren & Zeng (2024): "Reputation-Based Interaction Promotes Cooperation With Reinforcement Learning" - Reference paper containing a reputation system which was adapted for this work.
- 3 - Berg et al. (1995): Original Trust Game experimental framework - Original Trust Game formulation.
