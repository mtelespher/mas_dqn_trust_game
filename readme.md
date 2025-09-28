# Multi-Agent Trust Game with Global Reputation Systems 

## Objectives

This codebase implements a progressive series of experiments extending the classical game theoretical framework called the 'Trust Game' to multi-agent environments. Each artificial agent in this environment learns through interactions
via the reinforcement learning technique Deep Q-learning.

The objective of the research behind this codebase, was to investigate the emergence of cooperative behaviour or 'trust' norms between agents in this multi-agent system. Existing literature [1], indicated fixed pairs of agents that learn through the Deep Q-learning technique constructed in TensorFlow generate cooperative norms - when trained for ~300k steps. The rates of cooperation are similar to those of human participants [3].

A major drawback of the original paper was that when agents were paired randomly during training, cooperation collapsed to defection, therefore this research aimed to expand upon the original paper by implementing random pairing of agents at the start of each round and increasing the number of agents interacting to a multi-agent system, including a reputation mechanism which was adapted by another paper which successfully indicated cooperation in a multi-agent reinforcement learning game theory applciation.

## The Trust Game

The Trust Game is a behavioral economics experiment designed to measure trust and reciprocity between participants. Introduced by Berg, Dickhaut, and McCabe (1995), it involves two players: a **Trustor** and a **Trustee**.

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

This framework allows researchers to investigate how AI agents develop trust norms through reinforcement learning, providing insights into cooperation in decentralized systems.

### Methodology

A three phased experiment which focused on innovating and iterating proven frameworks for agent based cooperation [1] [2].

- Experiment 1: PyTorch Baseline Replication

Replication of Wu et al. (2023) framework for fixed pairs of agents playing the Trust Game to understand whether cooperation emergence is framework independent.
  
- Experiment 2: Random Pairing with Partner-Specific Networks

Creation of a solution to the lack of cooperation between random agents when randomly paired at the start of each round. This was achieved by implementing partner specific memory, neural networks for learning and epsilon decay to ensure that each partnership decayed at the same rate as the original paper.

- Experiment 3: Global Reputation System

After Experiment 2 indicated that cooperation with random agent pairs can generate cooperation, a multi-agent system was constructed. This was done by implementing a reputation mechanism, with agents being scored higher for cooperative actions, this reputation mechanism was applied to partner selection at the start of each round and as an input to the neural network behind the Deep Q-learning process.

## Experimental Progression

Experiment 1: PyTorch Baseline Replication
File: trust_game_mas_experiment_1.py
Replicates Wu et al. (2023) methodology using PyTorch instead of TensorFlow to validate framework independence.

## Configuration:

Agents: 10 fixed trustor-trustee pairs
Training: 350k steps per pair with sequential training
State Spaces: Trustor [last_sent, last_received], Trustee [amount_received]
Objective: Demonstrate cooperation emergence matches literature benchmarks

## Key Findings:

Trustor final average: ~5.45 units (Wu et al. target)
Trustee final average: ~6.20 units (Wu et al. target)
Validates PyTorch framework produces equivalent results to TensorFlow

Experiment 2: Random Pairing with Partner-Specific Networks
File: trust_game_mas_experiment_2.py
Extends to multi-agent environment with dynamic partner selection while maintaining relationship-specific learning.
Configuration:

Agents: 10 trustors (IDs 0-9), 10 trustees (IDs 12-21)
Training: 30M steps with random pairing each round
Architecture: Each agent maintains separate DQN for every potential partner (200 total networks)
Innovation: Partner-specific epsilon decay and experience replay memories

Key Features:

Partnership-specific learning enables relationship-dependent strategies
Random pairing tests cooperation emergence without fixed relationships
Comprehensive data collection tracks partnership evolution

Experiment 3: Global Reputation System
File: trust_game_mas_experiment_3.py

Configuration:

Agents: 20 trustors (IDs 0-19), 20 trustees (IDs 20-39)
Training: 10M steps with reputation-weighted partner selection
Reputation: Modified Ren-Zeng system with 0-10 scale, cooperation-only scoring
Selection: Weighted probability based on reputation (selection_bias=3.0)

Key Innovations:

Reputation-Enhanced States: [base_observation, partner_reputation]
Wealth Tracking: Persistent accumulation from zero initial endowment
Cooperation Scoring: Gaussian bell curve rewarding optimal return rates (~50%)
Social Stratification: High-reputation agents preferentially selected

Installation
bashgit clone [repository-url]
cd multi-agent-trust-game
pip install -r requirements.txt
Usage
Run Individual Experiments
python# Experiment 1: Baseline validation
python trust_game_mas_experiment_1.py

# Experiment 2: Partner-specific random pairing  
python trust_game_mas_experiment_2.py

# Experiment 3: Reputation system
python trust_game_mas_experiment_3.py
Key Parameters
Neural Networks:

Architecture: 800 → 1000 neurons, ReLU activation
Learning rate: 0.0016
Discount factor: 0.75
Experience replay: 50k-150k per partnership


## Key Findings
Cooperation Emergence

Experiment 1: Replicates Wu et al. [1] cooperation levels in PyTorch
Experiment 2: Stable cooperation emerges despite random pairing through partner-specific learning
Experiment 3: Reputation systems sustain cooperation while creating economic stratification

## Learning Dynamics

Partner-specific networks enable relationship-dependent strategies
Random pairing with memory preserves cooperation incentives
Reputation-based selection amplifies cooperation differences
Technical Requirements

Python 3.8+
PyTorch 1.9+
CUDA support recommended for large-scale experiments
16GB+ RAM for full 40-agent simulations

### Reproduction Notes
 
- All agents initialize with random weights for each run
- GPU acceleration significantly reduces training time (hours vs days)

## Related Work
This implementation extends:

1 - Wu et al. (2023): "Building Socially Intelligent AI Systems: Evidence from the Trust Game Using Artificial Agents with Deep Learning" - Reference paper conducting Trust Game experiments with Deep Q-learning fixed agent pairs.
2 - Ren & Zeng (2024): "Reputation-Based Interaction Promotes Cooperation With Reinforcement Learning" - Reference paper containing a reputation system which was adapted for this work.
3 - Berg et al. (1995): Original Trust Game experimental framework - Original Trust Game formulation.
