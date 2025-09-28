# %%
# ============================================================================
# MULTI-AGENT TRUST GAME WITH DEEP Q-NETWORKS
# ============================================================================
# Implementation of DQN agents playing the Trust Game in fixed pairs 
# Replicates Wu et al. (2023) TensorFlow methodology using PyTorch framework
# Understanding whether DQN agent pairs learning cooperation is Pythonframework independent.

# %%
# -----------------------------------------------------------------------------
# Import Required Libraries
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import math
import time
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# %%
# -----------------------------------------------------------------------------
# Deep Q-Network Architecture Definition
# -----------------------------------------------------------------------------
class DQNetwork(nn.Module):
    """
    Neural network for Q-learning with Trust Game-specific architecture.
    
    Architecture:
    - Input layer: variable size based on agent type (trustor=2, trustee=1)
    - Hidden layer 1: 800 neurons with ReLU activation
    - Hidden layer 2: 1000 neurons with ReLU activation  
    - Output layer: variable size based on action space
    """

    def __init__(self, n_features, n_actions):
        super(DQNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 800),
            nn.ReLU(),
            nn.Linear(800, 1000),
            nn.ReLU(),
            nn.Linear(1000, n_actions)
        )
        
        # Initialize weights to match original TensorFlow implementation
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with normal distribution matching TensorFlow defaults."""
        for module in self.model:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.3)
                nn.init.constant_(module.bias, 0.1)

    def forward(self, x):
        return self.model(x)

# -----------------------------------------------------------------------------
# DQN Agent Class Definition
# -----------------------------------------------------------------------------
class DQNAgent:
    """
    Deep Q-Network agent for Trust Game interactions.
    
    Implements experience replay, target networks, and epsilon-greedy exploration.
    Maintains separate memory for action sequences and full state transitions.
    """
    
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9,
                 epsilon_start=1.0, epsilon_min=0.1, epsilon_increment=0.001,
                 replace_target_iter=300, memory_size=500, batch_size=32, i=None):

        # Agent configuration
        self.n_actions = n_actions          # Number of possible actions
        self.n_features = n_features        # State space dimensionality
        self.lr = learning_rate            # Learning rate for optimizer
        self.gamma = reward_decay          # Discount factor for future rewards
        self.replace_target_iter = replace_target_iter  # Target network update frequency
        self.memory_size = memory_size     # Experience replay buffer size
        self.batch_size = batch_size       # Mini-batch size for training

        # Epsilon-greedy exploration parameters
        self.epsilon = epsilon_start       # Current exploration rate
        self.epsilon_start = epsilon_start # Initial exploration rate (1.0 = fully random)
        self.epsilon_min = epsilon_min     # Minimum exploration rate
        self.epsilon_increment = epsilon_increment  # Decay rate for epsilon

        # Agent identification and counters
        self.name = str(i)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory_action_counter = 0

        # Memory buffers for experience replay
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))  # Full transitions
        self.memory_action = np.arange(3).reshape(1, -1)  # Action sequences [own_action, other_action, reward]
        self.action_value = []  # Track Q-values over time
        self.cost_his = []      # Track training loss

        # GPU acceleration if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize evaluation and target networks
        self.eval_net = DQNetwork(n_features, n_actions).to(self.device)
        self.target_net = DQNetwork(n_features, n_actions).to(self.device)
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_):
        """
        Store complete state transition for experience replay.
        
        Args:
            s: Current state
            a: Action taken
            r: Reward received
            s_: Next state
        """
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def store_action(self, a, a_other, reward):
        """
        Store simplified action interaction for state construction.
        
        Maintains sequence of [own_action, partner_action, reward] for 
        building state representations in subsequent rounds.
        
        Args:
            a: Agent's own action
            a_other: Partner's action
            reward: Reward received this round
        """
        transition_action = np.hstack(([a, a_other, reward])).reshape(1, -1)

        if self.memory_action_counter == 0:
            self.memory_action = transition_action
        else:
            self.memory_action = np.vstack((self.memory_action, transition_action))

        self.memory_action_counter += 1

    def choose_action(self, observation):
        """
        Select action using epsilon-greedy strategy.
        
        With probability epsilon: choose random action (exploration)
        With probability 1-epsilon: choose action with highest Q-value (exploitation)
        
        Args:
            observation: Current state observation
            
        Returns:
            int: Selected action
        """
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)

        if np.random.rand() < self.epsilon:
            # EXPLORATION: Random action
            action = np.random.randint(0, self.n_actions)
            actions_value = self.eval_net(observation)
            self.action_value.append(torch.max(actions_value).item())
        else:
            # EXPLOITATION: Best action according to current Q-function
            actions_value = self.eval_net(observation)
            action = torch.argmax(actions_value).item()
            self.action_value.append(torch.max(actions_value).item())

        return action

    def store_value(self, observation):
        """Store Q-value for current observation (for analysis purposes)."""
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        actions_value = self.eval_net(observation)
        self.action_value.append(torch.max(actions_value).item())

    def learn(self):
        """
        Update neural network weights using experience replay and target networks.
        
        Implements Double DQN with target network for stable learning.
        Updates epsilon exploration rate after each learning step.
        """
        
        # Update target network periodically
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        # Wait until sufficient experience accumulated
        if self.memory_counter < self.batch_size:
            return

        # Sample random mini-batch from experience replay buffer
        sample_index = np.random.choice(min(self.memory_counter, self.memory_size), size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # Extract batch components
        b_s = torch.tensor(batch_memory[:, :self.n_features], dtype=torch.float32).to(self.device)      # Current states
        b_a = torch.tensor(batch_memory[:, self.n_features], dtype=torch.long).to(self.device)         # Actions taken
        b_r = torch.tensor(batch_memory[:, self.n_features + 1], dtype=torch.float32).to(self.device)  # Rewards received
        b_s_ = torch.tensor(batch_memory[:, -self.n_features:], dtype=torch.float32).to(self.device)   # Next states

        # Q-learning update using Bellman equation
        q_eval = self.eval_net(b_s).gather(1, b_a.unsqueeze(1)).squeeze()  # Current Q-values
        q_next = self.target_net(b_s_).detach()                            # Future Q-values (frozen target)
        q_target = b_r + self.gamma * q_next.max(1)[0]                     # Target Q-values

        # Compute loss and update network
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss.item())

        # Decay epsilon exploration rate
        if self.epsilon > self.epsilon_min:
            decay_factor = (1 - math.exp(-self.epsilon_increment * self.learn_step_counter))
            self.epsilon = max(self.epsilon_min, self.epsilon_start - decay_factor)

        self.learn_step_counter += 1

    def plot_cost(self):
        """Visualize training loss over time."""
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('Training steps')
        plt.show()

# %%
# -----------------------------------------------------------------------------
# Experimental Configuration
# -----------------------------------------------------------------------------
N = 10                    # Number of trustor/trustee agent pairs
TRAINING_STEPS = 350000   # Total training steps per pair
print(f'Training steps: {TRAINING_STEPS}')

# %%
# -----------------------------------------------------------------------------
# Initialize Trustee Agents
# -----------------------------------------------------------------------------
# Trustee action space: 0-30 (can return 0 to 3x amount received, max 30)
# State space: 1D (amount received from trustor)
RL_trustee = [DQNAgent(
    n_actions=len(np.arange(0, 31, 1)),  # 31 possible return amounts
    n_features=1,                        # Observes: amount received
    learning_rate=0.0016,
    reward_decay=0.75,
    epsilon_start=1.0,                   # Start fully exploratory
    epsilon_min=0.00001,                 # Nearly deterministic final policy
    epsilon_increment=0.0001,            # Exponential decay rate
    replace_target_iter=3000,            # Update target network every 3000 steps
    memory_size=100000,                  # Large experience replay buffer
    batch_size=64,
    i=i                                  # Agent identifier
) for i in range(N)]

# %%
# -----------------------------------------------------------------------------
# Initialize Trustor Agents  
# -----------------------------------------------------------------------------
# Trustor action space: 0-10 (amount to send from 10-unit endowment)
# State space: 2D (previous amount sent, previous amount received)
RL_trustor = [DQNAgent(
    n_actions=len(np.arange(0, 11, 1)),  # 11 possible send amounts
    n_features=2,                        # Observes: [last_sent, last_received]
    learning_rate=0.0016,
    reward_decay=0.75,
    epsilon_start=1.0,
    epsilon_min=0.00001,
    epsilon_increment=0.0001,
    replace_target_iter=3000,
    memory_size=100000,
    batch_size=64,
    i=i
) for i in range(N)]

# %%
# -----------------------------------------------------------------------------
# Agent Pairing Configuration
# -----------------------------------------------------------------------------
x = list(np.arange(N))  # Trustor indices [0, 1, 2, ...]
y = list(np.arange(N))  # Trustee indices [0, 1, 2, ...]
print(f"Agent pairs: {list(zip(x, y))}")

# %%
# ============================================================================
# MAIN TRAINING LOOP - FIXED PAIR METHODOLOGY
# ============================================================================
# Each trustor-trustee pair trains together for the complete training duration
# This replicates Wu et al. (2023) methodology exactly

print(f"Starting training for {len(x)} pairs...")
print(f"Training steps per pair: {TRAINING_STEPS}")
print(f"Total training steps: {len(x) * TRAINING_STEPS}")

# Global timing variables
total_start_time = time.time()

# Train each pair sequentially through complete training cycle
for pair_idx, (i, j) in enumerate(zip(x, y)):
    print(f"\n{'='*50}")
    print(f"Starting pair {pair_idx+1}/{len(x)}: Trustor {i} - Trustee {j}")
    print(f"{'='*50}")
    
    step = 0
    pair_start_time = time.time()

    # Each pair completes full training cycle
    for t in range(TRAINING_STEPS):
        
        # Progress reporting every 10k steps
        if step % 10000 == 0:
            current_time = time.time()
            elapsed_pair = current_time - pair_start_time
            elapsed_total = current_time - total_start_time
            
            if step > 0:
                # Calculate time estimates
                progress = step / TRAINING_STEPS
                estimated_pair_total = elapsed_pair / progress
                remaining_pair = estimated_pair_total - elapsed_pair
                
                pairs_completed = pair_idx
                pairs_remaining = len(x) - pair_idx - 1
                avg_time_per_pair = elapsed_total / max(1, pairs_completed + progress)
                total_remaining = pairs_remaining * avg_time_per_pair + remaining_pair
                
                print(f"  Step {step:,}/{TRAINING_STEPS:,} ({progress*100:.1f}%) | "
                      f"Pair time: {elapsed_pair/60:.1f}min | "
                      f"Est. remaining: {total_remaining/3600:.1f}h")
            else:
                print(f"  Step {step:,}/{TRAINING_STEPS:,} - Starting pair training...")

        # STEP 0: Random initialization to bootstrap interaction history
        if step == 0:
            print(f"    Initializing random actions...")
            # Generate random initial actions
            action_trustor = np.random.randint(0, 11)
            action_trustee = np.random.randint(0, 31)

            # Calculate Trust Game rewards
            # Trustor: keeps (10 - sent) + receives back
            # Trustee: receives (3 * sent) - gives back
            reward_trustor = 10 - action_trustor + action_trustee
            reward_trustee = 3 * action_trustor - action_trustee

            # Create initial state observations
            observation_trustor = np.array([action_trustor, action_trustee], dtype=np.float32)
            observation_trustee = np.array([action_trustor], dtype=np.float32)

            # Store initial values and actions
            RL_trustor[i].store_value(observation_trustor)
            RL_trustee[j].store_value(observation_trustee)
            RL_trustor[i].store_action(action_trustor, action_trustee, reward_trustor)
            RL_trustee[j].store_action(action_trustee, action_trustor, reward_trustee)
            
            print(f"    Initial actions - Trustor: {action_trustor}, Trustee: {action_trustee}")

        else:
            # NORMAL STEPS: Use learned behavior based on interaction history
            
            # Trustor constructs state from previous round interaction
            observation_trustor = np.array([
                RL_trustor[i].memory_action[step - 1, 0],  # Amount sent last round
                RL_trustor[i].memory_action[step - 1, 1]   # Amount received last round
            ], dtype=np.float32)

            # Trustor selects action based on previous interaction outcome
            action_trustor = RL_trustor[i].choose_action(observation_trustor)
            RL_trustor[i].store_value(observation_trustor)

            # Trustee observes trustor's current action and responds
            observation_trustee = np.array([action_trustor], dtype=np.float32)
            action_trustee = RL_trustee[j].choose_action(observation_trustee)
            RL_trustee[j].store_value(observation_trustee)

            # Calculate Trust Game rewards for this round
            reward_trustor = 10 - action_trustor + action_trustee
            reward_trustee = 3 * action_trustor - action_trustee

            # Store action sequence for next round's state construction
            RL_trustor[i].store_action(action_trustor, action_trustee, reward_trustor)
            RL_trustee[j].store_action(action_trustee, action_trustor, reward_trustee)

            # Construct next state observations for experience replay
            observation_trustor_ = np.array([
                RL_trustor[i].memory_action[step, 0],     # Current round sent
                RL_trustor[i].memory_action[step, 1]      # Current round received
            ], dtype=np.float32)

            # Predict trustor's next action for trustee's next state
            observation_trustee_ = np.array([
                RL_trustor[i].choose_action(observation_trustor_)
            ], dtype=np.float32)

            # LEARNING PHASE: Update neural networks after warmup period
            if (step > 200) and (step % 2 == 0):
                if step == 202:  # Log first learning step
                    print(f"    Starting learning at step {step}...")
                    learn_start = time.time()
                
                # Update both agents' Q-networks
                RL_trustor[i].learn()
                RL_trustee[j].learn()
                
                if step == 202:  # Estimate total learning time
                    learn_time = time.time() - learn_start
                    total_learn_steps = (TRAINING_STEPS - 200) // 2
                    estimated_learn_time = learn_time * total_learn_steps
                    print(f"    First learn() took {learn_time:.3f}s, "
                          f"estimated total learning time: {estimated_learn_time/60:.1f}min")

            # Store complete transitions for experience replay
            RL_trustor[i].store_transition(
                observation_trustor, action_trustor, reward_trustor, observation_trustor_
            )
            RL_trustee[j].store_transition(
                observation_trustee, action_trustee, reward_trustee, observation_trustee_
            )

        step += 1

    # Pair completion summary
    pair_end_time = time.time()
    pair_duration = pair_end_time - pair_start_time
    total_elapsed = pair_end_time - total_start_time
    
    print(f"\n✓ Pair {i}-{j} completed!")
    print(f"  Pair training time: {pair_duration/3600:.2f} hours")
    print(f"  Total elapsed time: {total_elapsed/3600:.2f} hours")
    
    # Estimate remaining time for remaining pairs
    pairs_completed = pair_idx + 1
    pairs_remaining = len(x) - pairs_completed
    if pairs_remaining > 0:
        avg_time_per_pair = total_elapsed / pairs_completed
        estimated_remaining = avg_time_per_pair * pairs_remaining
        print(f"  Estimated remaining time: {estimated_remaining/3600:.2f} hours")

# Final training summary
total_end_time = time.time()
total_duration = total_end_time - total_start_time

print(f"\n{'='*60}")
print(f"ALL TRAINING COMPLETED!")
print(f"{'='*60}")
print(f"Total training time: {total_duration/3600:.2f} hours")
print(f"Average time per pair: {total_duration/len(x)/3600:.2f} hours")

# %%
# ============================================================================
# POPULATION ANALYSIS AND VISUALIZATION
# ============================================================================
# Calculate population-level statistics and create Wu et al. style graphs

print("\nGenerating population analysis graphs...")

# Initialize lists for population-level metrics
pop_trustor_actions = []
pop_trustee_actions = []
pop_trustor_rewards = []
pop_trustee_rewards = []

# Calculate population averages at each time step
for step in range(TRAINING_STEPS):
    # Collect actions and rewards from all agents at this time step
    trustor_actions_t = []
    trustee_actions_t = []
    trustor_rewards_t = []
    trustee_rewards_t = []
    
    # Aggregate data across all agent pairs
    for agent_idx in range(N):
        # Check if agent has data for this step and handle array dimensions
        if (hasattr(RL_trustor[agent_idx], 'memory_action') and 
            RL_trustor[agent_idx].memory_action_counter > step):
            if RL_trustor[agent_idx].memory_action.shape[0] > step:
                trustor_actions_t.append(RL_trustor[agent_idx].memory_action[step, 0])  # Amount sent
                trustor_rewards_t.append(RL_trustor[agent_idx].memory_action[step, 2])  # Trustor reward
            
        if (hasattr(RL_trustee[agent_idx], 'memory_action') and 
            RL_trustee[agent_idx].memory_action_counter > step):
            if RL_trustee[agent_idx].memory_action.shape[0] > step:
                trustee_actions_t.append(RL_trustee[agent_idx].memory_action[step, 0])  # Amount returned
                trustee_rewards_t.append(RL_trustee[agent_idx].memory_action[step, 2])  # Trustee reward
    
    # Calculate population averages or use last available value
    if trustor_actions_t:
        pop_trustor_actions.append(np.mean(trustor_actions_t))
        pop_trustor_rewards.append(np.mean(trustor_rewards_t))
    else:
        if step > 0:
            pop_trustor_actions.append(pop_trustor_actions[-1])
            pop_trustor_rewards.append(pop_trustor_rewards[-1])
        else:
            pop_trustor_actions.append(0)
            pop_trustor_rewards.append(0)
            
    if trustee_actions_t:
        pop_trustee_actions.append(np.mean(trustee_actions_t))
        pop_trustee_rewards.append(np.mean(trustee_rewards_t))
    else:
        if step > 0:
            pop_trustee_actions.append(pop_trustee_actions[-1])
            pop_trustee_rewards.append(pop_trustee_rewards[-1])
        else:
            pop_trustee_actions.append(0)
            pop_trustee_rewards.append(0)

# Create time axis for plotting
steps = np.arange(len(pop_trustor_actions))

# Calculate smoothed moving averages for cleaner visualization
window = min(1000, len(steps) // 10)
if window < 10:
    window = min(10, len(steps))

if len(steps) > window and window > 0:
    # Apply moving average smoothing
    pop_trustor_actions_smooth = np.convolve(pop_trustor_actions, np.ones(window)/window, mode='valid')
    pop_trustee_actions_smooth = np.convolve(pop_trustee_actions, np.ones(window)/window, mode='valid')
    pop_trustor_rewards_smooth = np.convolve(pop_trustor_rewards, np.ones(window)/window, mode='valid')
    pop_trustee_rewards_smooth = np.convolve(pop_trustee_rewards, np.ones(window)/window, mode='valid')
    smooth_steps = steps[window-1:]
else:
    # Use raw data if insufficient points for smoothing
    pop_trustor_actions_smooth = pop_trustor_actions
    pop_trustee_actions_smooth = pop_trustee_actions
    pop_trustor_rewards_smooth = pop_trustor_rewards
    pop_trustee_rewards_smooth = pop_trustee_rewards
    smooth_steps = steps

# ============================================================================
# WU ET AL. STYLE FINAL PERIOD EVALUATION
# ============================================================================
# Calculate final period statistics using Wu et al.'s exact methodology

# Define evaluation period (last 10k steps or all steps if fewer)
final_period_length = min(10000, TRAINING_STEPS)
if TRAINING_STEPS >= 10000:
    eval_period_start = TRAINING_STEPS - 10000
else:
    eval_period_start = 0

print(f"\nEvaluating using Wu et al. methodology:")
print(f"Training steps: {TRAINING_STEPS}")
print(f"Evaluation period: steps {eval_period_start} to {TRAINING_STEPS-1}")
print(f"Evaluation period length: {final_period_length}")

# Calculate individual agent averages over final period
final_trustor_means = []
final_trustee_means = []
final_trustor_rewards = []
final_trustee_rewards = []

# Aggregate individual agent statistics
for i in range(N):
    # Process trustor agents
    if (hasattr(RL_trustor[i], 'memory_action') and 
        RL_trustor[i].memory_action_counter >= final_period_length):
        
        # Extract final period data
        agent_final_actions = RL_trustor[i].memory_action[-final_period_length:, 0]
        agent_final_rewards = RL_trustor[i].memory_action[-final_period_length:, 2]
        
        # Calculate individual averages
        final_trustor_means.append(np.mean(agent_final_actions))
        final_trustor_rewards.append(np.mean(agent_final_rewards))
    else:
        print(f"Warning: Trustor {i} has insufficient data for evaluation")

for j in range(N):
    # Process trustee agents
    if (hasattr(RL_trustee[j], 'memory_action') and 
        RL_trustee[j].memory_action_counter >= final_period_length):
        
        # Extract final period data
        agent_final_actions = RL_trustee[j].memory_action[-final_period_length:, 0]
        agent_final_rewards = RL_trustee[j].memory_action[-final_period_length:, 2]
        
        # Calculate individual averages
        final_trustee_means.append(np.mean(agent_final_actions))
        final_trustee_rewards.append(np.mean(agent_final_rewards))
    else:
        print(f"Warning: Trustee {j} has insufficient data for evaluation")

# Calculate population statistics from individual agent means
if final_trustor_means and final_trustee_means:
    # Population-level final averages
    final_trustor_actions = np.mean(final_trustor_means)
    final_trustee_actions = np.mean(final_trustee_means)
    final_trustor_rewards_mean = np.mean(final_trustor_rewards)
    final_trustee_rewards_mean = np.mean(final_trustee_rewards)
    
    # Population standard deviations
    trustor_std = np.std(final_trustor_means, ddof=1) if len(final_trustor_means) > 1 else 0
    trustee_std = np.std(final_trustee_means, ddof=1) if len(final_trustee_means) > 1 else 0
    
    print(f"\nIndividual agent final averages:")
    print(f"Trustor actions by agent: {[f'{x:.2f}' for x in final_trustor_means]}")
    print(f"Trustee actions by agent: {[f'{x:.2f}' for x in final_trustee_means]}")
else:
    print("Error: No valid agent data for evaluation")
    final_trustor_actions = 0
    final_trustee_actions = 0
    final_trustor_rewards_mean = 0
    final_trustee_rewards_mean = 0
    trustor_std = 0
    trustee_std = 0

# Create output directory
import os
os.makedirs("./final_baseline_run", exist_ok=True)

# ============================================================================
# VISUALIZATION: CREATE WU ET AL. STYLE ANALYSIS GRAPHS
# ============================================================================

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

# Main title
fig.suptitle(f'PyTorch Population Training Analysis (N={N} pairs, {TRAINING_STEPS:,} steps)', 
             fontsize=16, fontweight='bold')

# 1. POPULATION ACTIONS EVOLUTION (Top, spanning full width)
ax1 = fig.add_subplot(gs[0, :])
if len(steps) > 0:
    # Raw data with transparency
    ax1.plot(steps, pop_trustor_actions, alpha=0.3, color='blue', linewidth=0.5, label='Trustor Actions (raw)')
    ax1.plot(steps, pop_trustee_actions, alpha=0.3, color='red', linewidth=0.5, label='Trustee Actions (raw)')

    # Smoothed trends
    if len(smooth_steps) > 0:
        ax1.plot(smooth_steps, pop_trustor_actions_smooth, color='darkblue', linewidth=2, 
                 label=f'Trustor Actions ({window}-step avg)')
        ax1.plot(smooth_steps, pop_trustee_actions_smooth, color='darkred', linewidth=2, 
                 label=f'Trustee Actions ({window}-step avg)')
    
    # Wu et al. benchmark lines
    ax1.axhline(y=5.45, color='blue', linestyle='--', alpha=0.7, label='Wu et al. Trustor Target (5.45)')
    ax1.axhline(y=6.20, color='red', linestyle='--', alpha=0.7, label='Wu et al. Trustee Target (6.20)')

ax1.set_title('Population Action Evolution vs Literature Benchmarks')
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Average Amount')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.grid(True, alpha=0.3)

# 2. POPULATION REWARDS EVOLUTION (Bottom left)
ax2 = fig.add_subplot(gs[1, 0])
if len(steps) > 0:
    # Raw and smoothed reward trajectories
    ax2.plot(steps, pop_trustor_rewards, alpha=0.3, color='blue', linewidth=0.5, label='Trustor Rewards (raw)')
    ax2.plot(steps, pop_trustee_rewards, alpha=0.3, color='red', linewidth=0.5, label='Trustee Rewards (raw)')

    if len(smooth_steps) > 0:
        ax2.plot(smooth_steps, pop_trustor_rewards_smooth, color='darkblue', linewidth=2, 
                 label=f'Trustor Rewards ({window}-step avg)')
        ax2.plot(smooth_steps, pop_trustee_rewards_smooth, color='darkred', linewidth=2, 
                 label=f'Trustee Rewards ({window}-step avg)')

# Reference line for initial endowment
ax2.axhline(y=10, color='gray', linestyle='--', alpha=0.7, label='Initial Endowment')
ax2.set_title('Population Reward Evolution')
ax2.set_xlabel('Training Step')
ax2.set_ylabel('Average Reward')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. FINAL RESULTS COMPARISON WITH WU ET AL. (Bottom right)
ax3 = fig.add_subplot(gs[1, 1])

# Wu et al. target values from literature
wu_trustor_actions = 5.45
wu_trustee_actions = 6.20

categories = ['Trustor Actions', 'Trustee Actions']
pytorch_values = [final_trustor_actions, final_trustee_actions]
wu_values = [wu_trustor_actions, wu_trustee_actions]

# Create comparative bar chart
x_pos = np.arange(len(categories))
width = 0.35

bars1 = ax3.bar(x_pos - width/2, pytorch_values, width, label='PyTorch Results', color='blue', alpha=0.7)
bars2 = ax3.bar(x_pos + width/2, wu_values, width, label='Wu et al. Target', color='red', alpha=0.7)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.2f}', 
             ha='center', va='bottom', fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.2f}', 
             ha='center', va='bottom', fontweight='bold')

ax3.set_title('PyTorch Final Results vs Wu et al. (TensorFlow)')
ax3.set_ylabel('Average Amount')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(categories)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Adjust layout and save visualization
plt.tight_layout()
plt.savefig(r'D:\UCL\AI4SD\Multi agent systems\baseline_finalrun\baseline_training_analysis.png', 
            dpi=300, bbox_inches='tight')
print(r"Graph saved to: D:\UCL\AI4SD\Multi agent systems\baseline_finalrun\baseline_training_analysis.png")
plt.show()

# ============================================================================
# FINAL EVALUATION RESULTS AND STATISTICS
# ============================================================================

print(f"\n{'='*60}")
print("WU ET AL. STYLE EVALUATION RESULTS")
print(f"{'='*60}")

if final_trustor_means and final_trustee_means:
    print(f"Population averages (Wu et al. methodology):")
    print(f"  Trustor actions: {final_trustor_actions:.2f} ± {trustor_std:.2f} (Target: 5.45)")
    print(f"  Trustee actions: {final_trustee_actions:.2f} ± {trustee_std:.2f} (Target: 6.20)")
    print(f"  Trustor rewards: {final_trustor_rewards_mean:.2f}")
    print(f"  Trustee rewards: {final_trustee_rewards_mean:.2f}")
    print(f"  Total welfare: {final_trustor_rewards_mean + final_trustee_rewards_mean:.2f}")

    # Calculate cooperation metrics
    transfer_rate = final_trustor_actions / 10.0 * 100 if final_trustor_actions > 0 else 0
    if final_trustor_actions > 0:
        return_rate = final_trustee_actions / (3 * final_trustor_actions) * 100
    else:
        return_rate = 0

    print(f"\nCooperation efficiency:")
    print(f"  Transfer rate: {transfer_rate:.1f}% (Target: 54.5%)")
    print(f"  Return rate: {return_rate:.1f}% (Target: ~40%)")
    
    # Calculate deviation from literature benchmarks
    trustor_deviation = abs(final_trustor_actions - 5.45)
    trustee_deviation = abs(final_trustee_actions - 6.20)
    
    print(f"\nDeviation from Wu et al. targets:")
    print(f"  Trustor deviation: {trustor_deviation:.2f} units")
    print(f"  Trustee deviation: {trustee_deviation:.2f} units")
    print(f"  Total deviation: {trustor_deviation + trustee_deviation:.2f} units")
    
    print(f"\nAgent count used in evaluation:")
    print(f"  Trustor agents: {len(final_trustor_means)}/{N}")
    print(f"  Trustee agents: {len(final_trustee_means)}/{N}")
else:
    print("No evaluation data available - agents may not have sufficient training data")

print(f"\nEvaluation complete - ready for next phase!")

