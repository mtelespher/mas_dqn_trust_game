# %%
# ============================================================================
# MULTI-AGENT TRUST GAME WITH DEEP Q-NETWORKS
# ============================================================================
# Implementation of DQN agents playing the Trust Game with random pairing of agents in each round
# This implementation extends the classical Trust Game to a multi-agent environment
# Where agents maintain partner-specific Deep Q-Networks for studying trust emergence and cooperation dynamics in decentralised AI systems.
# Partner specific epsilon greedy decay and partner specific memory.


# %%
# -----------------------------------------------------------------------------
# Import Required Libraries
# -----------------------------------------------------------------------------

import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from collections import defaultdict, deque
from datetime import datetime


class DQNetwork(nn.Module):
    """
    Deep Q-Network implementation for trust game agents.
    
    This neural network approximates Q-values for state-action pairs in the trust game,
    enabling agents to learn optimal strategies through reinforcement learning.
    
    Args:
        n_features (int): Dimensionality of the input state space
        n_actions (int): Number of possible actions (sending/returning amounts)
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
        
        # Initialize weights to match TensorFlow baseline for consistency
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using normal distribution with specific parameters."""
        for module in self.model:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.3)
                nn.init.constant_(module.bias, 0.1)

    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)


class DQNAgent:
    """
    Deep Q-Network Agent with partner-specific learning capabilities.
    
    This agent maintains separate neural networks for each potential partner,
    enabling relationship-specific strategy development while supporting
    dynamic partner selection in multi-agent trust environments.
    
    Key Features:
    - Partner-specific DQN networks for individualized learning
    - Independent experience replay memories per partnership
    - Partnership-specific epsilon decay schedules
    - Interaction history tracking for reputation systems
    
    Args:
        n_actions (int): Number of possible actions
        n_features (int): Dimensionality of state space
        learning_rate (float): Learning rate for neural network optimization
        reward_decay (float): Discount factor for future rewards (gamma)
        epsilon_start (float): Initial exploration probability
        epsilon_min (float): Minimum exploration probability
        epsilon_increment (float): Rate of epsilon decay
        replace_target_iter (int): Frequency of target network updates
        memory_size (int): Size of experience replay buffer per partnership
        batch_size (int): Batch size for neural network training
        agent_id (int): Unique identifier for this agent
        partner_ids (list): List of potential partner agent IDs
    """
    
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9,
                 epsilon_start=1.0, epsilon_min=0.1, epsilon_increment=0.001,
                 replace_target_iter=300, memory_size=500, batch_size=32, i=None, 
                 agent_id=None, group_id=None, partner_ids=None):
        
        # Core learning parameters
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        # Epsilon parameters for exploration-exploitation balance
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_increment = epsilon_increment

        # Agent identification
        self.agent_id = agent_id if agent_id is not None else i
        self.name = str(i)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine partner list based on agent role
        if partner_ids is None:
            if self.agent_id < 12:  # trustor agents
                partner_ids = list(range(12, 24))
            else:  # trustee agents
                partner_ids = list(range(0, 10))
        
        # Partner-specific neural network components
        self.eval_nets = {}          # Evaluation networks for each partner
        self.target_nets = {}        # Target networks for stable learning
        self.optimizers = {}         # Optimizers for each network
        self.memories = {}           # Experience replay buffers per partnership
        self.learn_step_counters = {} # Learning step tracking per partnership
        self.memory_counters = {}    # Memory usage tracking per partnership
        self.epsilons = {}           # Individual epsilon for each partnership
        
        # Initialize components for each potential partner
        for partner_id in partner_ids:
            # Create neural networks
            self.eval_nets[partner_id] = DQNetwork(n_features, n_actions).to(self.device)
            self.target_nets[partner_id] = DQNetwork(n_features, n_actions).to(self.device)
            
            # Initialize optimizer
            self.optimizers[partner_id] = optim.RMSprop(
                self.eval_nets[partner_id].parameters(), lr=self.lr
            )
            
            # Initialize experience replay memory
            self.memories[partner_id] = np.zeros((memory_size, n_features * 2 + 2))
            
            # Initialize counters
            self.learn_step_counters[partner_id] = 0
            self.memory_counters[partner_id] = 0
            
            # Initialize epsilon for this partnership
            self.epsilons[partner_id] = epsilon_start
        
        # Shared training components
        self.action_value = []  # Q-value tracking for analysis
        self.cost_his = []      # Loss history for monitoring
        self.loss_func = nn.MSELoss()
        
        # Partner interaction tracking for reputation systems
        self.partner_memory = defaultdict(lambda: {
            'last_sent': None,
            'last_received': None,
            'interaction_count': 0,
            'transitions': []
        })
    
    def choose_action(self, observation, partner_id=None):
        """
        Select action using partner-specific network and epsilon-greedy strategy.
        
        Args:
            observation: Current state observation
            partner_id: ID of the partner for this interaction
            
        Returns:
            int: Selected action (amount to send/return)
        """
        if partner_id is None:
            raise ValueError("partner_id must be provided for partner-specific networks")
            
        if partner_id not in self.eval_nets:
            raise ValueError(f"No network found for partner {partner_id}")
        
        # Convert observation to tensor
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get current epsilon for this partnership
        current_epsilon = self.epsilons[partner_id]
        
        # Epsilon-greedy action selection
        if np.random.rand() < current_epsilon:
            # Explore: choose random action
            action = np.random.randint(0, self.n_actions)
            # Store Q-value for tracking
            with torch.no_grad():
                actions_value = self.eval_nets[partner_id](observation)
                self.action_value.append(torch.max(actions_value).item())
        else:
            # Exploit: choose action with highest Q-value
            with torch.no_grad():
                actions_value = self.eval_nets[partner_id](observation)
                action = torch.argmax(actions_value).item()
                self.action_value.append(torch.max(actions_value).item())
        
        return action
    
    def store_transition(self, s, a, r, s_, partner_id):
        """
        Store experience in partner-specific memory buffer.
        
        Args:
            s: Current state
            a: Action taken
            r: Reward received
            s_: Next state
            partner_id: Partner involved in this transition
        """
        if partner_id not in self.memories:
            raise ValueError(f"No memory found for partner {partner_id}")
        
        # Create transition tuple
        transition = np.hstack((s, [a, r], s_))
        
        # Store in circular buffer
        index = self.memory_counters[partner_id] % self.memory_size
        self.memories[partner_id][index, :] = transition
        self.memory_counters[partner_id] += 1
    
    def learn(self, current_partner_id):
        """
        Perform learning update using partner-specific network and memory.
        
        This method implements the Deep Q-Learning algorithm with experience replay
        and target networks for stable training.
        
        Args:
            current_partner_id: Partner for which to perform learning update
        """
        if current_partner_id not in self.eval_nets:
            return
        
        # Update target network periodically
        if self.learn_step_counters[current_partner_id] % self.replace_target_iter == 0:
            self.target_nets[current_partner_id].load_state_dict(
                self.eval_nets[current_partner_id].state_dict()
            )
        
        # Ensure sufficient experiences for this partnership
        if self.memory_counters[current_partner_id] < self.batch_size:
            return
        
        # Sample batch from partner-specific memory
        memory_size = min(self.memory_counters[current_partner_id], self.memory_size)
        sample_indices = np.random.choice(memory_size, size=self.batch_size, replace=True)
        batch_memory = self.memories[current_partner_id][sample_indices, :]
        
        # Extract batch components
        b_s = torch.tensor(batch_memory[:, :self.n_features], dtype=torch.float32).to(self.device)
        b_a = torch.tensor(batch_memory[:, self.n_features], dtype=torch.long).to(self.device)
        b_r = torch.tensor(batch_memory[:, self.n_features + 1], dtype=torch.float32).to(self.device)
        b_s_ = torch.tensor(batch_memory[:, self.n_features + 2:], dtype=torch.float32).to(self.device)
        
        # Compute Q-learning targets
        q_eval = self.eval_nets[current_partner_id](b_s).gather(1, b_a.unsqueeze(1)).squeeze()
        q_next = self.target_nets[current_partner_id](b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0]
        
        # Compute loss and perform gradient update
        loss = self.loss_func(q_eval, q_target)
        self.optimizers[current_partner_id].zero_grad()
        loss.backward()
        self.optimizers[current_partner_id].step()
        
        # Track training progress
        self.cost_his.append(loss.item())
        
        # Update epsilon for this specific partnership
        if self.epsilons[current_partner_id] > self.epsilon_min:
            import math
            decay_factor = (1 - math.exp(-self.epsilon_increment * self.learn_step_counters[current_partner_id]))
            self.epsilons[current_partner_id] = max(
                self.epsilon_min, 
                self.epsilon_start - decay_factor
            )
        
        self.learn_step_counters[current_partner_id] += 1
    
    def get_epsilon_stats(self):
        """Get epsilon statistics across all partnerships."""
        epsilons_list = list(self.epsilons.values())
        return {
            'min_epsilon': min(epsilons_list),
            'max_epsilon': max(epsilons_list),
            'avg_epsilon': np.mean(epsilons_list),
            'partner_epsilons': dict(self.epsilons)
        }
    
    def store_partner_interaction(self, partner_id, sent, received):
        """Update interaction history with specific partner."""
        self.partner_memory[partner_id]['last_sent'] = sent
        self.partner_memory[partner_id]['last_received'] = received
        self.partner_memory[partner_id]['interaction_count'] += 1
    
    def get_partner_history(self, partner_id):
        """Get interaction history with specific partner."""
        return self.partner_memory[partner_id]
    
    def get_partnership_stats(self):
        """Get statistics for all partnerships this agent is involved in."""
        stats = {}
        for partner_id in self.eval_nets.keys():
            partner_id = [pid for pid in partner_id if pid != self.agent_id][0]
            stats[partner_id] = {
                'experiences': self.memory_counters[partner_id],
                'learning_steps': self.learn_counters[partner_id],
                'avg_loss': np.mean(self.cost_his[partner_id][-100:]) if len(self.cost_his[partner_id]) > 0 else 0,
                'interactions': self.partner_memory[partner_id]['interaction_count']
            }
        return stats


# Agent Initialization
# ====================
# 
# This section creates the multi-agent system with two types of agents:
# 1. Trustor agents (IDs 0-9): Initiate trust interactions by sending amounts
# 2. Trustee agents (IDs 12-21): Respond to trust by returning amounts
#
# Key hyperparameter changes from baseline:
# - gamma reduced from 0.9 to 0.75 for shorter-term focus
# - epsilon_min increased to 0.0001 for continued exploration
# - memory_size set to 150k per partnership for extensive experience storage

# Trustor agents with simplified state space: [last_sent, last_received]
all_trustors = [
    DQNAgent( 
        n_actions=11,           # Can send 0-10 units
        n_features=2,           # [last_sent, last_received]
        learning_rate=0.0016,
        reward_decay=0.75,      # Reduced gamma for shorter-term learning
        epsilon_start=1.0,
        epsilon_min=0.0001,     # Maintain minimal exploration
        epsilon_increment=0.0001,
        replace_target_iter=3000,
        memory_size=150000,     # Large memory per partnership
        batch_size=64,
        agent_id=i,
        i=i,
        partner_ids=list(range(12, 22))  # Connect to trustees 12-21
    ) for i in range(10)
]

# Trustee agents with minimal state space: [amount_sent]
all_trustees = [
    DQNAgent( 
        n_actions=31,           # Can return 0-30 units (max 3x received)
        n_features=1,           # [amount_sent]
        learning_rate=0.0016,
        reward_decay=0.75,
        epsilon_start=1.0,
        epsilon_min=0.0001,
        epsilon_increment=0.0001,
        replace_target_iter=3000,
        memory_size=150000,
        batch_size=64,
        agent_id=i+12,
        i=i,
        partner_ids=list(range(0, 10))  # Connect to trustors 0-9
    ) for i in range(10)
]

# Combined agent list for convenience
all_agents = all_trustors + all_trustees

# Backward compatibility aliases
RL_trustor = all_trustors
RL_trustee = all_trustees

print(f"Created {len(all_trustors)} trustors (IDs 0-9)")
print(f"Created {len(all_trustees)} trustees (IDs 12-21)")
print(f"Total agents: {len(all_agents)}")

print(f"\n=== PURE REPUTATION SYSTEM ===")
print(f"• No group structure - all agents are individuals")
print(f"• Partner selection driven purely by random pairing scores")
print(f"• State space simplified (no group membership flags)")
print(f"• Focus on understanding whether cooperation emerges with random pairing")

print(f"\n=== AGENT SPECIFICATIONS ===")
print(f"Trustors:")
print(f"  • Agent IDs: 0-9")
print(f"  • State space: [last_sent, last_received]")
print(f"  • Action space: 0-10 (amount to send)")
print(f"  • Role: Initiate trust interactions")

print(f"Trustees:")
print(f"  • Agent IDs: 12-21")
print(f"  • State space: [amount_sent]")
print(f"  • Action space: 0-30 (amount to return)")
print(f"  • Role: Respond to trust with reciprocation")

print("="*60)


# Utility Functions for Data Management
# ====================================

def save_all_agent_networks(all_agents, save_dir, experiment_name="random_pairing_test"):
    """
    Save all partner-specific networks for all agents.
    
    This function creates a comprehensive backup of the entire multi-agent system,
    including neural network states, training progress, and interaction histories.
    
    Args:
        all_agents: List of all DQN agents in the system
        save_dir: Directory to save the networks
        experiment_name: Name prefix for the experiment files
        
    Returns:
        str: Path to the created save directory
    """
    # Create timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_save_dir = os.path.join(save_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(full_save_dir, exist_ok=True)
    
    print(f"Saving networks to: {full_save_dir}")
    
    # Save each agent's networks and data
    for agent in all_agents:
        agent_dir = os.path.join(full_save_dir, f"agent_{agent.agent_id}")
        os.makedirs(agent_dir, exist_ok=True)
        
        # Save each partner-specific network
        for partner_id in agent.eval_nets.keys():
            filename = f"partner_{partner_id}.pth"
            filepath = os.path.join(agent_dir, filename)
            
            # Save comprehensive checkpoint
            torch.save({
                'eval_net_state_dict': agent.eval_nets[partner_id].state_dict(),
                'target_net_state_dict': agent.target_nets[partner_id].state_dict(),
                'optimizer_state_dict': agent.optimizers[partner_id].state_dict(),
                'learn_counter': agent.learn_step_counters[partner_id],
                'memory_counter': agent.memory_counters[partner_id],
                'epsilon': agent.epsilons[partner_id],
                'agent_id': agent.agent_id,
                'partner_id': partner_id,
                'n_actions': agent.n_actions,
                'n_features': agent.n_features
            }, filepath)
        
        # Save agent metadata and interaction history
        metadata = {
            'agent_id': agent.agent_id,
            'partner_memory': dict(agent.partner_memory),
            'epsilon_stats': agent.get_epsilon_stats(),
            'cost_history': agent.cost_his,
            'action_values': agent.action_value[-1000:] if agent.action_value else []
        }
        
        metadata_path = os.path.join(agent_dir, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    print(f"Saved {len(all_agents)} agents with partner-specific networks")
    return full_save_dir


def save_experiment_summary(save_dir, partnership_sequences, training_steps, all_agents):
    """
    Save high-level experiment summary and results.
    
    Args:
        save_dir: Directory to save the summary
        partnership_sequences: Dictionary of partnership interaction data
        training_steps: Total number of training steps completed
        all_agents: List of all agents in the system
        
    Returns:
        dict: Summary statistics dictionary
    """
    # Calculate summary statistics
    all_interactions = []
    for partnership_data in partnership_sequences.values():
        all_interactions.extend(partnership_data)
    
    if all_interactions:
        final_period_start = int(0.9 * len(all_interactions))
        final_trustor_avg = np.mean([i['sent'] for i in all_interactions[final_period_start:]])
        final_trustee_avg = np.mean([i['returned'] for i in all_interactions[final_period_start:]])
    else:
        final_trustor_avg = 0
        final_trustee_avg = 0
    
    # Collect epsilon statistics
    epsilon_stats = {}
    for agent in all_agents:
        epsilon_stats[agent.agent_id] = agent.get_epsilon_stats()
    
    summary = {
        'experiment_info': {
            'training_steps': training_steps,
            'total_partnerships': len(partnership_sequences),
            'total_interactions': len(all_interactions),
            'agents_count': len(all_agents)
        },
        'final_results': {
            'trustor_avg_sent': final_trustor_avg,
            'trustee_avg_returned': final_trustee_avg,
            'cooperation_rate': final_trustee_avg / (final_trustor_avg * 3) if final_trustor_avg > 0 else 0
        },
        'partnership_data': partnership_sequences,
        'epsilon_evolution': epsilon_stats,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(save_dir, "experiment_summary.pkl")
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"Saved experiment summary to: {summary_path}")
    return summary


# Main Training Loop
# ==================

# Training parameters
TRAINING_STEPS = 30_000_000  # Total number of interaction steps
LOG_INTERVAL = 50_000        # Progress logging frequency
LEARNING_WARMUP = 200        # Minimum interactions before learning starts
LEARNING_FREQUENCY = 2       # Learn every N interactions per partnership

total_start_time = time.time()

# Interaction tracking
pair_counts = defaultdict(int)
agent_counts = defaultdict(int)
reputation_logs = []

print("="*60)
print(f"Starting PARTNER-SPECIFIC NETWORKS TEST")
print(f"• Fixed training steps: {TRAINING_STEPS:,}")
print(f"• Partner selection: Random")
print(f"• One-hot encoding: NO (removed)")
print(f"• Partner-specific networks: YES")
print("="*60)

# Initialize data collection structures
partnership_sequences = defaultdict(list)
cooperation_timeline = []

print("="*60)

# Main training loop
for step in range(TRAINING_STEPS):
    
    # === AGENT SELECTION ===
    # Randomly select a trustor to initiate interaction
    trustor = random.choice(all_trustors)
    
    # Randomly select a trustee from available partners
    available_trustees = [t for t in all_trustees if t.agent_id != trustor.agent_id]
    trustee = random.choice(available_trustees)
    
    # === INTERACTION TRACKING ===
    # Track partnership frequencies for analysis
    pair_key = tuple(sorted((trustor.agent_id, trustee.agent_id)))
    pair_counts[pair_key] += 1
    agent_counts[trustor.agent_id] += 1
    agent_counts[trustee.agent_id] += 1
    
    # === GET INTERACTION HISTORY ===
    # Retrieve previous interactions between these specific agents
    if hasattr(trustor, 'partner_memory'):
        trustor_history = trustor.partner_memory[trustee.agent_id]
        trustee_history = trustee.partner_memory[trustor.agent_id]
        first_interaction = trustor_history['last_sent'] is None
    else:
        first_interaction = trustor.memory_counter == 0
        trustor_history = {'last_sent': None, 'last_received': None}
        trustee_history = {'last_sent': None, 'last_received': None}
    
    # === ACTION SELECTION ===
    if first_interaction:
        # Random actions for first interaction
        action_trustor = np.random.randint(0, 11)
        action_trustee = np.random.randint(0, 31)
    else:
        # Use learned policies based on interaction history
        
        # Trustor state: [last_sent, last_received]
        obs_trustor = np.array([
            trustor_history['last_sent'],
            trustor_history['last_received']
        ], dtype=np.float32)
        
        action_trustor = trustor.choose_action(obs_trustor, partner_id=trustee.agent_id)
        
        # Trustee state: [amount_sent]
        obs_trustee = np.array([action_trustor], dtype=np.float32)
        action_trustee = trustee.choose_action(obs_trustee, trustor.agent_id)
    
    # === CALCULATE REWARDS ===
    # Trust game reward structure
    reward_trustor = 10 - action_trustor + action_trustee  # Keep what not sent + receive back
    reward_trustee = 3 * action_trustor - action_trustee   # Get 3x sent - give back
    
    # === UPDATE PARTNER MEMORY ===
    # Store interaction outcomes for future reference
    if hasattr(trustor, 'store_partner_interaction'):
        trustor.store_partner_interaction(trustee.agent_id, action_trustor, action_trustee)
        trustee.store_partner_interaction(trustor.agent_id, action_trustee, action_trustor)
    else:
        trustor.store_action(action_trustor, action_trustee, reward_trustor)
        trustee.store_action(action_trustee, action_trustor, reward_trustee)
    
    # === PARTNERSHIP DATA COLLECTION ===
    # Record detailed interaction data for analysis
    partnership_key = (trustor.agent_id, trustee.agent_id)
    partnership_sequences[partnership_key].append({
        'interaction_number': len(partnership_sequences[partnership_key]) + 1,
        'step': step,
        'sent': action_trustor,
        'returned': action_trustee,
        'trustor_reward': reward_trustor,
        'trustee_reward': reward_trustee
    })
    
    # === COOPERATION TIMELINE COLLECTION ===
    # Calculate population-level cooperation metrics periodically
    if step % 50000 == 0 and step > 0:
        recent_interactions = []
        
        # Analyze recent interactions across all partnerships
        for partnership, sequence in partnership_sequences.items():
            if len(sequence) >= 5:  # Only partnerships with sufficient history
                recent_5 = sequence[-5:]  # Last 5 interactions
                for interaction in recent_5:
                    if interaction['sent'] > 0:
                        cooperation_rate = interaction['returned'] / (interaction['sent'] * 3)
                        recent_interactions.append(min(1.0, cooperation_rate))
        
        if recent_interactions:
            cooperation_timeline.append({
                'step': step,
                'avg_cooperation': np.mean(recent_interactions),
                'cooperation_std': np.std(recent_interactions)
            })
    
    # === PERIODIC MONITORING ===
    # Monitor learning progress every 1M steps
    if step % 1_000_000 == 0:
        sample_agent = all_trustors[0]
        epsilon_stats = sample_agent.get_epsilon_stats()
        print(f"Step {step:,} - Epsilon range: {epsilon_stats['min_epsilon']:.3f} to {epsilon_stats['max_epsilon']:.3f}")
        
        # Check partnership activity distribution
        partnership_data = [(k, len(v)) for k, v in partnership_sequences.items()]
        partnership_data.sort(key=lambda x: -x[1])
        print(f"Most active partnership: {partnership_data[0]} interactions")
        print(f"Least active partnership: {partnership_data[-1]} interactions")
    
    # === PREPARE STATES FOR LEARNING ===
    # Create state representations for experience storage
    if not first_interaction and hasattr(trustor, 'partner_memory'):
        # Current states
        obs_trustor_curr = np.array([
            trustor_history['last_sent'] if trustor_history['last_sent'] is not None else 0,
            trustor_history['last_received'] if trustor_history['last_received'] is not None else 0,
        ], dtype=np.float32)
        
        obs_trustee_curr = np.array([action_trustor], dtype=np.float32)
        
        # Next states after interaction
        trustor_next_hist = trustor.partner_memory[trustee.agent_id]
        next_obs_trustor = np.array([
            trustor_next_hist['last_sent'],
            trustor_next_hist['last_received'],
        ], dtype=np.float32)
        
        next_obs_trustee = np.array([trustor.choose_action(next_obs_trustor, partner_id=trustee.agent_id)], dtype=np.float32)

        # Store transitions in experience replay buffers
        trustor.store_transition(obs_trustor_curr, action_trustor, reward_trustor, next_obs_trustor, trustee.agent_id)
        trustee.store_transition(obs_trustee_curr, action_trustee, reward_trustee, next_obs_trustee, trustor.agent_id)
    
    # === LEARNING UPDATES ===
    # Perform partnership-specific learning when sufficient data available
    partnership_interaction_count = trustor.partner_memory[trustee.agent_id]['interaction_count']
    
    if (partnership_interaction_count > LEARNING_WARMUP and 
        partnership_interaction_count % LEARNING_FREQUENCY == 0):
        trustor.learn(trustee.agent_id)
        trustee.learn(trustor.agent_id)
    
    # === PROGRESS LOGGING ===
    if step % LOG_INTERVAL == 0:
        elapsed = (time.time() - total_start_time) / 60
        progress = step / TRAINING_STEPS * 100
        
        print(f"Step {step:,}/{TRAINING_STEPS:,} ({progress:.1f}%) | "
              f"Elapsed: {elapsed:.1f} min")
        
        if step % 100000 == 0 and step > 0:
            print(f"  Recent interaction: Trustor {trustor.agent_id} -> Trustee {trustee.agent_id}")
            print(f"  Actions: Trustor sent {action_trustor}, Trustee returned {action_trustee}")
            
            # Show data collection progress
            print(f"  Partnerships formed: {len(partnership_sequences)}")
            if cooperation_timeline:
                print(f"  Current cooperation rate: {cooperation_timeline[-1]['avg_cooperation']:.3f}")

# === TRAINING COMPLETION AND ANALYSIS ===
duration = time.time() - total_start_time
hours = duration / 3600

print("\nRandom Pairing Test Complete")
print("="*60)
print(f"Total Training Time: {hours:.2f} hours")
print(f"Total Training Steps: {TRAINING_STEPS:,}")
print(f"Unique Pairs Interacted: {len(pair_counts)}")

# Data collection summary
print(f"\n=== DATA COLLECTION SUMMARY ===")
print(f"• Partnerships tracked: {len(partnership_sequences)}")
print(f"• Cooperation timeline points: {len(cooperation_timeline)}")
if partnership_sequences:
    interaction_counts = [len(seq) for seq in partnership_sequences.values()]
    print(f"• Average interactions per partnership: {np.mean(interaction_counts):.1f}")
    print(f"• Most active partnership: {max(interaction_counts)} interactions")

print(f"\nPartner Selection Patterns (Should be roughly uniform):")
print("Top 5 most frequent pairs:") 
for (aid1, aid2), count in sorted(pair_counts.items(), key=lambda x: -x[1])[:5]:
    print(f"  Agent {aid1} - Agent {aid2}: {count:,} interactions")

print("Bottom 5 least frequent pairs:")
for (aid1, aid2), count in sorted(pair_counts.items(), key=lambda x: x[1])[:5]:
    print(f"  Agent {aid1} - Agent {aid2}: {count:,} interactions")

# Test partner-specific learning
print(f"\n=== PARTNER-SPECIFIC LEARNING TEST ===")
print("Testing if agents learned different strategies for different partners...")

# Sample trustor behavior with different trustees
sample_trustor = all_trustors[0]
test_trustees = all_trustees[0:3]  # Test with first 3 trustees

print(f"\nTrustor {sample_trustor.agent_id} behavior with different trustees:")
for trustee in test_trustees:
    test_history = sample_trustor.partner_memory[trustee.agent_id]
    
    if test_history['last_sent'] is not None:
        test_obs = np.array([
            test_history['last_sent'],
            test_history['last_received']
        ], dtype=np.float32)
        
        # Get action without randomness
        old_epsilon = sample_trustor.epsilons[trustee.agent_id]
        sample_trustor.epsilons[trustee.agent_id] = 0
        test_action = sample_trustor.choose_action(test_obs, partner_id=trustee.agent_id)
        sample_trustor.epsilons[trustee.agent_id] = old_epsilon
        
        print(f"  -> Trustee {trustee.agent_id}: Would send {test_action}")

print("\n" + "="*60)
print("ANALYSIS:")
print("• Random pairing ensures all trustees get equal opportunity")
print("• Data collected for partnership analysis")
print("="*60)


# Population Analysis and Visualization
# ====================================

print("\n" + "="*60)
print("CALCULATING POPULATION STATISTICS FROM PARTNERSHIP DATA")
print("="*60)

# Process partnership data for analysis
try:
    print(f"Processing {len(partnership_sequences)} partnerships...")
    
    if len(partnership_sequences) == 0:
        print("ERROR: partnership_sequences is empty. Check data collection in training loop.")
        all_interactions = []
    else:
        print("Found partnership data!")
        
        # Collect all interactions in chronological order
        all_interactions = []
        for partnership_key, interactions in partnership_sequences.items():
            for interaction in interactions:
                all_interactions.append(interaction)

        # Sort by step number to get chronological order
        all_interactions.sort(key=lambda x: x['step'])
        print(f"Total interactions collected: {len(all_interactions)}")

except NameError:
    print("ERROR: partnership_sequences variable doesn't exist.")
    all_interactions = []
except Exception as e:
    print(f"ERROR: {e}")
    all_interactions = []

if len(all_interactions) > 0:
    # Extract arrays from sorted interactions
    steps_actual = [interaction['step'] for interaction in all_interactions]
    trustor_actions = [interaction['sent'] for interaction in all_interactions]
    trustee_actions = [interaction['returned'] for interaction in all_interactions]
    trustor_rewards = [interaction['trustor_reward'] for interaction in all_interactions]
    trustee_rewards = [interaction['trustee_reward'] for interaction in all_interactions]
    
    # Calculate rolling averages for smoother visualization
    window_size = min(1000, len(all_interactions) // 10)
    print(f"Using rolling window size: {window_size}")
    
    def rolling_average(data, window):
        """Calculate rolling average with given window size."""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Calculate rolling averages
    if len(all_interactions) >= window_size:
        trustor_actions_smooth = rolling_average(trustor_actions, window_size)
        trustee_actions_smooth = rolling_average(trustee_actions, window_size)
        trustor_rewards_smooth = rolling_average(trustor_rewards, window_size)
        trustee_rewards_smooth = rolling_average(trustee_rewards, window_size)
        steps_smooth = steps_actual[window_size-1:]
    else:
        trustor_actions_smooth = trustor_actions
        trustee_actions_smooth = trustee_actions
        trustor_rewards_smooth = trustor_rewards
        trustee_rewards_smooth = trustee_rewards
        steps_smooth = steps_actual

    # Calculate final period statistics (last 10% of interactions)
    final_period_start = int(0.9 * len(all_interactions))
    final_trustor_actions = np.mean(trustor_actions[final_period_start:])
    final_trustee_actions = np.mean(trustee_actions[final_period_start:])
    final_trustor_rewards = np.mean(trustor_rewards[final_period_start:])
    final_trustee_rewards = np.mean(trustee_rewards[final_period_start:])
    
    print(f"Final period analysis based on last {len(all_interactions) - final_period_start} interactions")


# Visualization Functions
# =======================

def create_partnership_smooth_curves():
    """Create smoothed curves for top 6 most active partnerships."""
    
    # Create output directory
    os.makedirs("./training_graphs_clean", exist_ok=True)
    print(f"Created/verified directory: {os.path.abspath('./training_graphs_clean')}")
    
    # Get top 6 partnerships by interaction count
    partnership_counts = {k: len(v) for k, v in partnership_sequences.items()}
    top_6_partnerships = sorted(partnership_counts.items(), key=lambda x: -x[1])[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Smoothed Action Curves - Top 6 Most Active Partnerships', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (partnership_key, interaction_count) in enumerate(top_6_partnerships):
        ax = axes[idx]
        
        # Get partnership data
        partnership_data = partnership_sequences[partnership_key]
        
        if len(partnership_data) > 0:
            # Extract data
            interaction_numbers = [i['interaction_number'] for i in partnership_data]
            sent_amounts = [i['sent'] for i in partnership_data]
            returned_amounts = [i['returned'] for i in partnership_data]
            
            # Apply smoothing (rolling average)
            window_size = min(1000, len(partnership_data) // 10)  # Adaptive window
            
            def smooth_data(data, window):
                if len(data) < window:
                    return data
                return np.convolve(data, np.ones(window)/window, mode='valid')
            
            if len(partnership_data) >= window_size:
                sent_smooth = smooth_data(sent_amounts, window_size)
                returned_smooth = smooth_data(returned_amounts, window_size)
                interaction_smooth = interaction_numbers[window_size-1:]
            else:
                sent_smooth = sent_amounts
                returned_smooth = returned_amounts
                interaction_smooth = interaction_numbers
            
            # Plot smoothed curves
            ax.plot(interaction_smooth, sent_smooth, color='blue', linewidth=2, 
                   label='Sent (smoothed)', alpha=0.8)
            ax.plot(interaction_smooth, returned_smooth, color='red', linewidth=2, 
                   label='Returned (smoothed)', alpha=0.8)
            
            # Add trend lines
            if len(partnership_data) > 10:
                # Linear trend for sent amounts
                sent_trend = np.polyfit(interaction_numbers, sent_amounts, 1)
                sent_line = np.poly1d(sent_trend)
                ax.plot(interaction_numbers, sent_line(interaction_numbers), 
                       '--', color='lightblue', alpha=0.7, label='Sent trend')
                
                # Linear trend for returned amounts
                returned_trend = np.polyfit(interaction_numbers, returned_amounts, 1)
                returned_line = np.poly1d(returned_trend)
                ax.plot(interaction_numbers, returned_line(interaction_numbers), 
                       '--', color='lightcoral', alpha=0.7, label='Returned trend')
        
        # Format subplot
        ax.set_title(f'Partnership {partnership_key[0]}->{partnership_key[1]}\n'
                    f'({interaction_count} interactions)', fontsize=12)
        ax.set_xlabel('Interaction Number')
        ax.set_ylabel('Amount')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 35)
    
    plt.tight_layout()
    
    # Save plot
    save_path = "./training_graphs_clean/partnership_smooth_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def create_population_smooth_curves():
    """Create population-wide smoothed curves matching partnership graph format."""
    
    # Collect all interactions chronologically
    all_interactions = []
    for partnership_data in partnership_sequences.values():
        all_interactions.extend(partnership_data)
    
    # Sort by step number
    all_interactions.sort(key=lambda x: x['step'])
    
    if len(all_interactions) == 0:
        print("No interaction data available")
        return
    
    # Extract population data
    steps = [i['step'] for i in all_interactions]
    trustor_actions = [i['sent'] for i in all_interactions]
    trustee_actions = [i['returned'] for i in all_interactions]
    
    # Create time-based bins for population averages
    bin_size = max(1000, len(all_interactions) // 200)  # ~200 points
    
    binned_steps = []
    binned_trustor_avg = []
    binned_trustee_avg = []
    
    for i in range(0, len(all_interactions), bin_size):
        bin_end = min(i + bin_size, len(all_interactions))
        bin_data = all_interactions[i:bin_end]
        
        if bin_data:
            binned_steps.append(np.mean([x['step'] for x in bin_data]))
            binned_trustor_avg.append(np.mean([x['sent'] for x in bin_data]))
            binned_trustee_avg.append(np.mean([x['returned'] for x in bin_data]))
    
    # Apply additional smoothing
    smooth_window = min(20, len(binned_steps) // 10)
    
    def smooth_curve(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    if len(binned_steps) >= smooth_window:
        trustor_smooth = smooth_curve(binned_trustor_avg, smooth_window)
        trustee_smooth = smooth_curve(binned_trustee_avg, smooth_window)
        steps_smooth = binned_steps[smooth_window-1:]
    else:
        trustor_smooth = binned_trustor_avg
        trustee_smooth = binned_trustee_avg
        steps_smooth = binned_steps
    
    # Create the plot with same format as partnership graphs
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Plot smoothed population curves
    ax.plot(steps_smooth, trustor_smooth, color='blue', linewidth=2, 
           label='Sent (smoothed)', alpha=0.8)
    ax.plot(steps_smooth, trustee_smooth, color='red', linewidth=2, 
           label='Returned (smoothed)', alpha=0.8)
    
    # Add trend lines
    if len(binned_steps) > 10:
        # Linear trend for sent amounts
        sent_trend = np.polyfit(binned_steps, binned_trustor_avg, 1)
        sent_line = np.poly1d(sent_trend)
        ax.plot(binned_steps, sent_line(binned_steps), 
               '--', color='lightblue', alpha=0.7, label='Sent trend')
        
        # Linear trend for returned amounts
        returned_trend = np.polyfit(binned_steps, binned_trustee_avg, 1)
        returned_line = np.poly1d(returned_trend)
        ax.plot(binned_steps, returned_line(binned_steps), 
               '--', color='lightcoral', alpha=0.7, label='Returned trend')
    
    # Format plot
    total_steps = f'{TRAINING_STEPS:,}' if 'TRAINING_STEPS' in globals() else 'Unknown'
    ax.set_title(f'Population-Wide Smoothed Action Curves\n'
                f'{total_steps} training steps, {len(all_interactions):,} total interactions', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Amount', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 35)
    
    plt.tight_layout()
    
    # Save plot
    save_path = "./training_graphs_clean/population_smooth_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


# Generate visualizations if data is available
if len(partnership_sequences) > 0:
    print("Creating partnership-specific smoothed curves...")
    create_partnership_smooth_curves()
    
    print("\nCreating population-wide smoothed curves...")
    create_population_smooth_curves()
    
    print("\nBoth plot types created and saved!")
else:
    print("No partnership data available for plotting")

# Explanation of visualization differences
print("\n" + "="*60)
print("PLOT EXPLANATION")
print("="*60)
print("PLOT 1: Partnership-Specific Smoothed Curves")
print("  • Shows learning curves for the 6 most active partnerships")
print("  • Each subplot shows one partnership's evolution over time")
print("  • Smoothed to reduce noise and show trends clearly")
print("  • X-axis: Interaction number within that partnership")
print("  • Y-axis: Amount sent/returned in that specific partnership")

print("\nPLOT 2: Population-Wide Smoothed Curves") 
print("  • Shows overall population behavior across ALL partnerships")
print("  • Combines data from all trustors and all trustees")
print("  • Heavily smoothed to show macro-level trends")
print("  • X-axis: Training step (chronological time)")
print("  • Y-axis: Population average amount sent/returned")

print("\nKEY DIFFERENCES:")
print("  • Partnership plots: Micro-level (individual relationships)")
print("  • Population plots: Macro-level (system-wide behavior)")
print("  • Partnership plots: Show partner-specific learning")
print("  • Population plots: Show overall cooperation emergence")
print("="*60)

# Save networks and experiment data
print("\n" + "="*60)
print("SAVING NETWORKS AND EXPERIMENT DATA")
print("="*60)

save_dir = "./training_graphs_clean"

# Save experiment summary with all collected data
save_experiment_summary(
    save_dir=save_dir,
    partnership_sequences=partnership_sequences,
    training_steps=TRAINING_STEPS,
    all_agents=all_trustors + all_trustees
)

print("All networks and data saved successfully!")
print(f"Saved to: {save_dir}")
print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)