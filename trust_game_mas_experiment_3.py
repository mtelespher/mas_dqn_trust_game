# %%
# ============================================================================
# MULTI-AGENT TRUST GAME WITH DEEP Q-NETWORKS
# ============================================================================

# Multi-Agent System with artificial agents using Deep Q-learning to find optimal strategies whilst playing the Trust Game

# Extension of the classical Trust Game to multiple agents and  reputation-based partner selection
# Partner specific memory and epsilon decay
# Partner selection at the start of each round based on reputation
# Wealth accumulation by storing rewards over each round 

#%% 
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim

from collections import defaultdict, deque
from datetime import datetime


class DQNetwork(nn.Module):
    """
    Deep Q-Network for trust game agents with reputation-enhanced state spaces.
    
    Neural network architecture processes state observations including interaction
    history and partner reputation information to output Q-values for all possible actions.
    """
    
    def __init__(self, n_features, n_actions):
        super(DQNetwork, self).__init__()
        self.model = nn.Sequential(  # model: sequential neural network layers
            nn.Linear(n_features, 800),
            nn.ReLU(),
            nn.Linear(800, 1000),
            nn.ReLU(),
            nn.Linear(1000, n_actions)
        )
        
        # Initialize weights to match TensorFlow baseline
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using normal distribution for consistency."""
        for module in self.model:  # module: individual layer in the neural network
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.3)
                nn.init.constant_(module.bias, 0.1)

    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)  # x: input tensor containing state and reputation information


class DQNAgent:
    """
    Base DQN Agent with partner-specific neural networks and epsilon decay.
    
    Each agent maintains separate Deep Q-Networks for every potential partner,
    enabling relationship-specific strategy development while supporting
    dynamic partner selection in multi-agent trust environments.
    """
    
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9,
                 epsilon_start=1.0, epsilon_min=0.1, epsilon_increment=0.001,
                 replace_target_iter=300, memory_size=500, batch_size=32, i=None, 
                 agent_id=None, group_id=None, partner_ids=None):
        
        # Core neural network parameters
        self.n_actions = n_actions  # n_actions: number of possible actions (send/return amounts)
        self.n_features = n_features  # n_features: dimensionality of state observation space
        self.lr = learning_rate  # lr: learning rate for neural network optimization
        self.gamma = reward_decay  # gamma: discount factor for future rewards in Q-learning
        self.replace_target_iter = replace_target_iter  # replace_target_iter: frequency of target network updates
        self.memory_size = memory_size  # memory_size: capacity of experience replay buffer per partnership
        self.batch_size = batch_size  # batch_size: number of experiences sampled for each learning update

        # Epsilon-greedy exploration parameters
        self.epsilon_start = epsilon_start  # epsilon_start: initial exploration probability
        self.epsilon_min = epsilon_min  # epsilon_min: minimum exploration probability floor
        self.epsilon_increment = epsilon_increment  # epsilon_increment: rate of epsilon decay per learning step

        # Agent identification and device setup
        self.agent_id = agent_id if agent_id is not None else i  # agent_id: unique identifier for this agent
        self.name = str(i)  # name: string representation of agent identifier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device: computation device (GPU/CPU)
        
        # Determine partner list based on agent role
        if partner_ids is None:
            if self.agent_id < 20:  # trustor agents (IDs 0-19)
                partner_ids = list(range(20, 40))  # partner_ids: trustee agents (IDs 20-39)
            else:  # trustee agents (IDs 20-39)
                partner_ids = list(range(0, 20))  # partner_ids: trustor agents (IDs 0-19)
        
        # Partner-specific neural network components
        self.eval_nets = {}  # eval_nets: evaluation networks for each partner relationship
        self.target_nets = {}  # target_nets: target networks for stable Q-learning updates
        self.optimizers = {}  # optimizers: RMSprop optimizers for each partner-specific network
        self.memories = {}  # memories: experience replay buffers per partnership
        self.learn_step_counters = {}  # learn_step_counters: learning steps completed per partnership
        self.memory_counters = {}  # memory_counters: memory usage tracking per partnership
        self.epsilons = {}  # epsilons: individual exploration rate for each partnership
        
        # Initialize neural network components for each potential partner
        for partner_id in partner_ids:  # partner_id: ID of specific partner agent
            # Create partner-specific neural networks
            self.eval_nets[partner_id] = DQNetwork(n_features, n_actions).to(self.device)
            self.target_nets[partner_id] = DQNetwork(n_features, n_actions).to(self.device)
            
            # Initialize optimizer for this partnership
            self.optimizers[partner_id] = optim.RMSprop(
                self.eval_nets[partner_id].parameters(), lr=self.lr
            )
            
            # Initialize experience replay memory (pre-allocated numpy array)
            self.memories[partner_id] = np.zeros((memory_size, n_features * 2 + 2))
            
            # Initialize learning progress counters
            self.learn_step_counters[partner_id] = 0
            self.memory_counters[partner_id] = 0
            
            # Initialize exploration rate for this specific partnership
            self.epsilons[partner_id] = epsilon_start
        
        # Shared training components
        self.action_value = []  # action_value: Q-value tracking for analysis and monitoring
        self.cost_his = []  # cost_his: loss history for monitoring training progress
        self.loss_func = nn.MSELoss()  # loss_func: mean squared error loss function for Q-learning
        
        # Partner interaction tracking for reputation systems
        self.partner_memory = defaultdict(lambda: {  # partner_memory: detailed interaction history per partner
            'last_sent': None,  # last_sent: amount sent in previous interaction with this partner
            'last_received': None,  # last_received: amount received in previous interaction with this partner
            'interaction_count': 0,  # interaction_count: total interactions with this specific partner
            'transitions': []  # transitions: list of state-action-reward transitions for analysis
        })
    
    def choose_action(self, observation, partner_id=None):
        """
        Select action using partner-specific network and epsilon-greedy strategy.
        
        Args:
            observation: Current state observation tensor
            partner_id: ID of the partner for this interaction
            
        Returns:
            int: Selected action (amount to send/return)
        """
        if partner_id is None:
            raise ValueError("partner_id must be provided for partner-specific networks")
            
        if partner_id not in self.eval_nets:
            raise ValueError(f"No network found for partner {partner_id}")
        
        # Convert observation to PyTorch tensor
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)  # observation: state tensor for neural network input
        
        # Get current exploration rate for this specific partnership
        current_epsilon = self.epsilons[partner_id]  # current_epsilon: exploration probability for this partner
        
        # Epsilon-greedy action selection strategy
        if np.random.rand() < current_epsilon:
            # Explore: choose random action for learning
            action = np.random.randint(0, self.n_actions)  # action: randomly selected action for exploration
            # Store Q-value for tracking (even during exploration)
            with torch.no_grad():
                actions_value = self.eval_nets[partner_id](observation)  # actions_value: Q-values for all possible actions
                self.action_value.append(torch.max(actions_value).item())
        else:
            # Exploit: choose action with highest Q-value
            with torch.no_grad():
                actions_value = self.eval_nets[partner_id](observation)  # actions_value: Q-values for all possible actions
                action = torch.argmax(actions_value).item()  # action: greedy action with highest Q-value
                self.action_value.append(torch.max(actions_value).item())
        
        return action
    
    def store_transition(self, s, a, r, s_, partner_id):
        """
        Store experience in partner-specific memory buffer for replay learning.
        
        Args:
            s: Current state observation
            a: Action taken in current state
            r: Reward received for action
            s_: Next state after action
            partner_id: Partner involved in this transition
        """
        if partner_id not in self.memories:
            raise ValueError(f"No memory found for partner {partner_id}")
        
        # Create transition tuple for experience replay
        transition = np.hstack((s, [a, r], s_))  # transition: experience tuple (state, action, reward, next_state)
        
        # Store in circular buffer (overwrites oldest experiences when full)
        index = self.memory_counters[partner_id] % self.memory_size  # index: circular buffer position for storage
        self.memories[partner_id][index, :] = transition
        self.memory_counters[partner_id] += 1
    
    def learn(self, current_partner_id):
        """
        Perform Deep Q-Learning update using partner-specific network and memory.
        
        Implements experience replay and target networks for stable training.
        Updates epsilon decay for this specific partnership.
        
        Args:
            current_partner_id: Partner for which to perform learning update
        """
        if current_partner_id not in self.eval_nets:
            return
        
        # Update target network periodically for training stability
        if self.learn_step_counters[current_partner_id] % self.replace_target_iter == 0:
            self.target_nets[current_partner_id].load_state_dict(
                self.eval_nets[current_partner_id].state_dict()
            )
        
        # Ensure sufficient experiences for meaningful batch learning
        if self.memory_counters[current_partner_id] < self.batch_size:
            return
        
        # Sample random batch from partner-specific experience replay buffer
        memory_size = min(self.memory_counters[current_partner_id], self.memory_size)  # memory_size: actual usable memory size
        sample_indices = np.random.choice(memory_size, size=self.batch_size, replace=True)  # sample_indices: random indices for experience sampling
        batch_memory = self.memories[current_partner_id][sample_indices, :]  # batch_memory: sampled experiences for training
        
        # Extract batch components for neural network training
        b_s = torch.tensor(batch_memory[:, :self.n_features], dtype=torch.float32).to(self.device)  # b_s: batch of current states
        b_a = torch.tensor(batch_memory[:, self.n_features], dtype=torch.long).to(self.device)  # b_a: batch of actions taken
        b_r = torch.tensor(batch_memory[:, self.n_features + 1], dtype=torch.float32).to(self.device)  # b_r: batch of rewards received
        b_s_ = torch.tensor(batch_memory[:, self.n_features + 2:], dtype=torch.float32).to(self.device)  # b_s_: batch of next states
        
        # Compute Q-learning targets using Bellman equation
        q_eval = self.eval_nets[current_partner_id](b_s).gather(1, b_a.unsqueeze(1)).squeeze()  # q_eval: Q-values for taken actions
        q_next = self.target_nets[current_partner_id](b_s_).detach()  # q_next: Q-values for next states (detached from gradient)
        q_target = b_r + self.gamma * q_next.max(1)[0]  # q_target: target Q-values using Bellman equation
        
        # Compute loss and perform gradient update
        loss = self.loss_func(q_eval, q_target)  # loss: mean squared error between predicted and target Q-values
        self.optimizers[current_partner_id].zero_grad()
        loss.backward()
        self.optimizers[current_partner_id].step()
        
        # Track training progress
        self.cost_his.append(loss.item())
        
        # Update epsilon for this specific partnership using exponential decay
        if self.epsilons[current_partner_id] > self.epsilon_min:
            import math
            decay_factor = (1 - math.exp(-self.epsilon_increment * self.learn_step_counters[current_partner_id]))  # decay_factor: exponential decay rate for epsilon
            self.epsilons[current_partner_id] = max(
                self.epsilon_min, 
                self.epsilon_start - decay_factor
            )
        
        self.learn_step_counters[current_partner_id] += 1
    
    def get_epsilon_stats(self):
        """Get epsilon statistics across all partnerships for monitoring."""
        epsilons_list = list(self.epsilons.values())  # epsilons_list: all epsilon values for statistical analysis
        return {
            'min_epsilon': min(epsilons_list),  # min_epsilon: lowest exploration rate across partnerships
            'max_epsilon': max(epsilons_list),  # max_epsilon: highest exploration rate across partnerships
            'avg_epsilon': np.mean(epsilons_list),  # avg_epsilon: average exploration rate across partnerships
            'partner_epsilons': dict(self.epsilons)  # partner_epsilons: exploration rate for each specific partner
        }
    
    def store_partner_interaction(self, partner_id, sent, received):
        """Update interaction history with specific partner for reputation tracking."""
        self.partner_memory[partner_id]['last_sent'] = sent  # sent: amount sent to this partner in current interaction
        self.partner_memory[partner_id]['last_received'] = received  # received: amount received from this partner
        self.partner_memory[partner_id]['interaction_count'] += 1
    
    def get_partner_history(self, partner_id):
        """Get complete interaction history with specific partner."""
        return self.partner_memory[partner_id]  # returns complete interaction history for specified partner
    
    def get_partnership_stats(self):
        """Get comprehensive statistics for all partnerships this agent maintains."""
        stats = {}  # stats: dictionary containing partnership performance metrics
        for partner_id in self.eval_nets.keys():
            partner_id = [pid for pid in partner_id if pid != self.agent_id][0]  # partner_id: extracted partner identifier
            stats[partner_id] = {
                'experiences': self.memory_counters[partner_id],  # experiences: number of stored transitions
                'learning_steps': self.learn_step_counters[partner_id],  # learning_steps: number of learning updates performed
                'avg_loss': np.mean(self.cost_his[-100:]) if len(self.cost_his) > 0 else 0,  # avg_loss: recent average training loss
                'interactions': self.partner_memory[partner_id]['interaction_count']  # interactions: total number of interactions with partner
            }
        return stats


class GlobalReputationDQNAgent(DQNAgent):
    """
    DQN Agent enhanced with global reputation system integration.
    
    State space enhanced with partner reputation information to enable
    reputation-aware decision making and partner selection strategies.
    Removes agent's own reputation from state to prevent feedback loops.
    """
    
    def __init__(self, n_actions, n_features, reputation_manager, 
                 selection_bias=3, **kwargs):
        
        # Enhanced state space: original features + partner reputation (excludes own reputation)
        enhanced_n_features = n_features + 1  # enhanced_n_features: original state size plus partner reputation
        super().__init__(n_actions=n_actions, n_features=enhanced_n_features, **kwargs)
        
        # Store original feature count and reputation system components
        self.original_n_features = n_features  # original_n_features: base state space size before reputation enhancement
        self.reputation_manager = reputation_manager  # reputation_manager: global reputation tracking system
        self.selection_bias = selection_bias  # selection_bias: weighting factor for reputation in partner selection
    
    def get_enhanced_observation(self, base_observation, partner_id):
        """
        Enhance base observation with partner reputation information only.
        
        Creates reputation-aware state: [original_state, partner_reputation]
        Excludes own reputation to prevent learning feedback loops.
        
        Args:
            base_observation: Original state without reputation
            partner_id: ID of partner for reputation lookup
            
        Returns:
            Enhanced observation including partner reputation
        """
        partner_reputation = self.reputation_manager.get_reputation(partner_id)  # partner_reputation: reputation score of interaction partner
        
        enhanced_obs = np.concatenate([  # enhanced_obs: state enhanced with reputation information
            base_observation,
            [partner_reputation]  # only partner reputation (own reputation excluded)
        ]).astype(np.float32)
        
        return enhanced_obs
    
    def choose_action_with_reputation(self, base_observation, partner_id):
        """Choose action using reputation-enhanced state (partner reputation only)."""
        enhanced_obs = self.get_enhanced_observation(base_observation, partner_id)  # enhanced_obs: state with partner reputation
        return self.choose_action(enhanced_obs, partner_id)
    
    def store_transition_with_reputation(self, base_s, a, r, base_s_, partner_id):
        """Store transition with reputation-enhanced states for both current and next states."""
        enhanced_s = self.get_enhanced_observation(base_s, partner_id)  # enhanced_s: current state with reputation
        enhanced_s_ = self.get_enhanced_observation(base_s_, partner_id)  # enhanced_s_: next state with reputation
        self.store_transition(enhanced_s, a, r, enhanced_s_, partner_id)
    
    def select_partner_by_reputation(self, available_partners):
        """
        Select partner based on reputation scores using weighted random selection.
        
        Higher reputation partners have higher probability of selection,
        creating incentives for reputation building and cooperation.
        
        Args:
            available_partners: List of potential partner agents
            
        Returns:
            Selected partner agent based on reputation weighting
        """
        if len(available_partners) <= 1:
            return available_partners[0] if available_partners else None
        
        partner_ids = [partner.agent_id for partner in available_partners]  # partner_ids: list of available partner IDs
        weights = self.reputation_manager.get_partner_selection_weights(  # weights: reputation-based selection probabilities
            partner_ids, self.selection_bias
        )
        
        # Weighted random selection favoring high-reputation partners
        selected_idx = np.random.choice(len(available_partners), p=np.array(weights)/sum(weights))  # selected_idx: index of chosen partner
        selected = available_partners[selected_idx]  # selected: chosen partner agent
        
        return selected


class WealthTrackingGlobalReputationDQNAgent(GlobalReputationDQNAgent):
    """
    Enhanced agent with persistent wealth accumulation from zero initial endowment.
    
    Extends reputation-aware agents with wealth tracking to study economic
    outcomes and inequality emergence in multi-agent trust systems.
    """
    
    def __init__(self, n_actions, n_features, reputation_manager, 
                 selection_bias=3, **kwargs):
        
        # Initialize parent class with reputation system
        super().__init__(
            n_actions=n_actions,
            n_features=n_features,
            reputation_manager=reputation_manager,
            selection_bias=selection_bias,
            **kwargs
        )
        
        # Wealth tracking starting from zero endowment
        self.current_wealth = 0.0  # current_wealth: accumulated wealth across all interactions
        self.wealth_history = [(0, 0.0)]  # wealth_history: list of (step, wealth) tuples for analysis
    
    def update_wealth(self, reward, step):
        """Accumulate wealth from trust game rewards over time."""
        self.current_wealth += reward  # reward: payoff from current trust game interaction
        self.wealth_history.append((step, self.current_wealth))  # step: global training step number
    
    def get_wealth_stats(self):
        """Get comprehensive wealth statistics for analysis."""
        return {
            'current_wealth': self.current_wealth,  # current_wealth: total accumulated wealth
            'total_interactions': len(self.wealth_history) - 1,  # total_interactions: number of wealth-affecting interactions
            'avg_reward_per_interaction': self.current_wealth / max(1, len(self.wealth_history) - 1),  # avg_reward_per_interaction: mean payoff per interaction
            'wealth_history': self.wealth_history  # wealth_history: complete wealth evolution timeline
        }


class RenZengGlobalReputationManager:
    """
    Modified global reputation system based on cooperation behavior only.
    
    Implements simplified reputation updates using 0-10 scale for better
    neural network learning. Removes partner selection component to focus
    purely on cooperation scoring.
    """
    
    def __init__(self, num_agents=40, theta=0.2, initial_reputation=5.0):
        self.theta = theta  # theta: exponential moving average update rate
        self.initial_reputation = initial_reputation  # initial_reputation: starting reputation for all agents
        
        # Global reputation scores using 0-10 scale
        self.agent_reputations = {i: initial_reputation for i in range(num_agents)}  # agent_reputations: reputation score for each agent
        
        # Track reputation evolution for analysis
        self.reputation_history = defaultdict(list)  # reputation_history: detailed reputation change history per agent
        
        print(f"Modified Reputation Manager initialized")
        print(f"   Cooperation-only scoring (removed partner selection)")
        print(f"   Reputation range: 0-10 scale")
        print(f"   Theta: {theta}")
    
    def calculate_cooperation_score(self, sent_amount, returned_amount, role='trustee'):
        """
        Calculate cooperation score using Gaussian bell curve for trustees.
        
        Rewards optimal cooperation rate (~50% return) with highest scores.
        Provides more generous scoring to encourage cooperative behavior.
        
        Args:
            sent_amount: Amount sent by trustor
            returned_amount: Amount returned by trustee
            role: Agent role ('trustor' or 'trustee')
            
        Returns:
            Cooperation score on 0-10 scale
        """
        if sent_amount == 0:
            return 5.0
            
        if role == 'trustee':
            return_rate = returned_amount / (sent_amount * 3)  # return_rate: fraction of possible return amount
            
            # Handle extreme cooperation cases
            if return_rate <= 0.02:  # nearly zero return (defection)
                return 0.5
            if return_rate >= 0.98:  # nearly complete return (maximum cooperation)
                return 1.0
                
            # Gaussian bell curve centered on optimal cooperation rate
            mu = 0.5  # mu: peak cooperation rate (50% return)
            sigma = 0.35  # sigma: curve width (increased for more forgiving scoring)
            
            # Gaussian formula: e^(-(x-μ)²/(2σ²))
            exponent = -((return_rate - mu) ** 2) / (2 * (sigma ** 2))  # exponent: Gaussian calculation component
            bell_value = np.exp(exponent)  # bell_value: Gaussian curve value (0-1)
            
            # Scale to reputation range 1.0-10.0 for generous scoring
            reputation = 1.0 + (bell_value * 9.0)  # reputation: scaled cooperation score
            
            return min(10.0, max(0.0, reputation))
            
        elif role == 'trustor':
            # Generous trustor scoring to reward risk-taking
            risk_score = sent_amount / 10.0  # risk_score: normalized amount sent (0-1)
            # Scale to 1-10 range to reward trust behavior
            return 1.0 + (risk_score * 9.0)
        
        return 5.0
    
    def update_reputation_complete(self, agent_id, sent_amount, returned_amount, 
                                 partner_id, role, step=None):
        """
        Update agent reputation using simplified cooperation-only system.
        
        Uses exponential moving average to incorporate cooperation behavior
        into reputation scores. Removes partner selection component for simplicity.
        
        Args:
            agent_id: ID of agent whose reputation to update
            sent_amount: Amount sent in interaction
            returned_amount: Amount returned in interaction
            partner_id: ID of interaction partner
            role: Agent role ('trustor' or 'trustee')
            step: Training step number for history tracking
            
        Returns:
            Updated reputation score
        """
        old_reputation = self.agent_reputations[agent_id]  # old_reputation: previous reputation score
        
        # Calculate cooperation score only (no partner selection component)
        cooperation_score = self.calculate_cooperation_score(sent_amount, returned_amount, role)  # cooperation_score: behavior-based reputation update
        
        # Exponential moving average update
        new_reputation = (1 - self.theta) * old_reputation + self.theta * cooperation_score  # new_reputation: updated reputation using EMA
        
        # Clamp to valid reputation range [0, 10]
        new_reputation = max(0.0, min(10.0, new_reputation))
        
        self.agent_reputations[agent_id] = new_reputation
        
        # Track detailed reputation evolution history
        if step is not None:
            self.reputation_history[agent_id].append({
                'step': step,  # step: global training step
                'old_reputation': old_reputation,  # old_reputation: reputation before update
                'new_reputation': new_reputation,  # new_reputation: reputation after update
                'cooperation_score': cooperation_score,  # cooperation_score: behavior-based score
                'sent': sent_amount,  # sent: amount sent in interaction
                'returned': returned_amount,  # returned: amount returned in interaction
                'role': role  # role: agent role in interaction
            })
        
        return new_reputation
    
    def get_reputation(self, agent_id):
        """Get current reputation score for specified agent."""
        return self.agent_reputations[agent_id]  # returns current reputation on 0-10 scale
    
    def get_reputation_stats(self):
        """Get population-wide reputation statistics for analysis."""
        reputations = list(self.agent_reputations.values())  # reputations: all current reputation scores
        return {
            'mean': np.mean(reputations),  # mean: average reputation across population
            'std': np.std(reputations),  # std: reputation standard deviation
            'min': min(reputations),  # min: lowest reputation in population
            'max': max(reputations),  # max: highest reputation in population
            'all_reputations': dict(self.agent_reputations)  # all_reputations: complete reputation mapping
        }
    
    def get_partner_selection_weights(self, available_partner_ids, selection_bias=2):
        """
        Calculate partner selection weights based on reputation scores.
        
        Transforms 0-10 reputation scale to selection probabilities,
        creating preference for high-reputation partners.
        
        Args:
            available_partner_ids: List of potential partner IDs
            selection_bias: Exponential weighting factor for reputation preference
            
        Returns:
            List of selection weights for probabilistic partner choice
        """
        # Get reputation scores for available partners
        reputations = [self.agent_reputations[pid] for pid in available_partner_ids]  # reputations: reputation scores for available partners
        min_rep = 1e-3  # min_rep: minimum weight to prevent zero probabilities

        # Apply exponential weighting based on reputation
        weights = [rep + min_rep ** selection_bias for rep in reputations]  # weights: reputation-based selection probabilities
        return weights


def update_both_agents_reputation_simplified(reputation_manager, trustor, trustee, 
                                         action_trustor, action_trustee, step):
    """
    Update reputation for both agents using simplified cooperation-only system.
    
    Updates both trustor and trustee reputations based solely on cooperation
    behavior, removing partner selection component for clearer incentives.
    
    Args:
        reputation_manager: Global reputation tracking system
        trustor: Trustor agent in interaction
        trustee: Trustee agent in interaction
        action_trustor: Amount sent by trustor
        action_trustee: Amount returned by trustee
        step: Global training step number
    """
    # Update trustee reputation based on cooperation behavior
    reputation_manager.update_reputation_complete(
        agent_id=trustee.agent_id,
        sent_amount=action_trustor,
        returned_amount=action_trustee,
        partner_id=trustor.agent_id,
        role='trustee',
        step=step
    )
    
    # Update trustor reputation based on risk-taking behavior
    reputation_manager.update_reputation_complete(
        agent_id=trustor.agent_id,
        sent_amount=action_trustor,
        returned_amount=action_trustee,
        partner_id=trustee.agent_id,
        role='trustor',
        step=step
    )


# Initialize global reputation manager for experiment
global_reputation_manager = RenZengGlobalReputationManager(
    num_agents=40,  # num_agents: total number of agents in the system
    theta=0.2,  # theta: reputation update rate (reduced for slower convergence)
    initial_reputation=5.0  # initial_reputation: starting reputation (middle of 0-10 range)
)


# Utility Functions for Data Management
def save_all_agent_networks(all_agents, save_dir, experiment_name="random_pairing_test"):
    """
    Save all partner-specific networks for comprehensive agent state preservation.
    
    Creates timestamped directory containing all neural networks, training states,
    and interaction histories for complete experiment reproducibility.
    
    Args:
        all_agents: List of all agents to save
        save_dir: Base directory for saving
        experiment_name: Descriptive name for experiment
        
    Returns:
        Path to created save directory
    """
    # Create timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # timestamp: unique identifier for this save
    full_save_dir = os.path.join(save_dir, f"{experiment_name}_{timestamp}")  # full_save_dir: complete path for saving
    os.makedirs(full_save_dir, exist_ok=True)
    
    print(f"Saving networks to: {full_save_dir}")
    
    # Save each agent's complete state
    for agent in all_agents:  # agent: individual agent to save
        agent_dir = os.path.join(full_save_dir, f"agent_{agent.agent_id}")  # agent_dir: directory for this agent's data
        os.makedirs(agent_dir, exist_ok=True)
        
        # Save each partner-specific network state
        for partner_id in agent.eval_nets.keys():  # partner_id: ID of partner for this network
            filename = f"partner_{partner_id}.pth"  # filename: network save file name
            filepath = os.path.join(agent_dir, filename)  # filepath: complete path for network file
            
            # Save comprehensive neural network checkpoint
            torch.save({
                'eval_net_state_dict': agent.eval_nets[partner_id].state_dict(),  # eval_net_state_dict: evaluation network weights
                'target_net_state_dict': agent.target_nets[partner_id].state_dict(),  # target_net_state_dict: target network weights
                'optimizer_state_dict': agent.optimizers[partner_id].state_dict(),  # optimizer_state_dict: optimizer state
                'learn_counter': agent.learn_step_counters[partner_id],  # learn_counter: learning steps completed
                'memory_counter': agent.memory_counters[partner_id],  # memory_counter: experiences stored
                'epsilon': agent.epsilons[partner_id],  # epsilon: current exploration rate
                'agent_id': agent.agent_id,  # agent_id: agent identifier
                'partner_id': partner_id,  # partner_id: partner identifier
                'n_actions': agent.n_actions,  # n_actions: action space size
                'n_features': agent.n_features  # n_features: state space size
            }, filepath)
        
        # Save agent metadata and interaction history
        metadata = {  # metadata: comprehensive agent state information
            'agent_id': agent.agent_id,  # agent_id: unique agent identifier
            'partner_memory': dict(agent.partner_memory),  # partner_memory: interaction history per partner
            'epsilon_stats': agent.get_epsilon_stats(),  # epsilon_stats: exploration statistics
            'cost_history': agent.cost_his,  # cost_history: training loss evolution
            'action_values': agent.action_value[-1000:] if agent.action_value else []  # action_values: recent Q-value history
        }
        
        metadata_path = os.path.join(agent_dir, "metadata.pkl")  # metadata_path: path for metadata file
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    print(f"Saved {len(all_agents)} agents with partner-specific networks")
    return full_save_dir


def save_experiment_summary(save_dir, partnership_sequences, training_steps, all_agents):
    """
    Save high-level experiment summary and comprehensive results analysis.
    
    Creates summary statistics, final outcomes, and complete partnership data
    for post-training analysis and visualization.
    
    Args:
        save_dir: Directory to save summary
        partnership_sequences: Dictionary of partnership interaction data
        training_steps: Total training steps completed
        all_agents: List of all agents in experiment
        
    Returns:
        Summary statistics dictionary
    """
    # Calculate summary statistics from all interactions
    all_interactions = []  # all_interactions: complete list of interactions across partnerships
    for partnership_data in partnership_sequences.values():  # partnership_data: interaction list for specific partnership
        all_interactions.extend(partnership_data)
    
    if all_interactions:
        final_period_start = int(0.9 * len(all_interactions))  # final_period_start: start of final 10% of interactions
        final_trustor_avg = np.mean([i['sent'] for i in all_interactions[final_period_start:]])  # final_trustor_avg: average sending in final period
        final_trustee_avg = np.mean([i['returned'] for i in all_interactions[final_period_start:]])  # final_trustee_avg: average returning in final period
    else:
        final_trustor_avg = 0
        final_trustee_avg = 0
    
    # Collect epsilon statistics from all agents
    epsilon_stats = {}  # epsilon_stats: exploration statistics per agent
    for agent in all_agents:  # agent: individual agent for statistics
        epsilon_stats[agent.agent_id] = agent.get_epsilon_stats()
    
    # Create comprehensive summary
    summary = {  # summary: complete experiment summary data
        'experiment_info': {
            'training_steps': training_steps,  # training_steps: total training iterations
            'total_partnerships': len(partnership_sequences),  # total_partnerships: number of unique partnerships
            'total_interactions': len(all_interactions),  # total_interactions: total interaction count
            'agents_count': len(all_agents)  # agents_count: total number of agents
        },
        'final_results': {
            'trustor_avg_sent': final_trustor_avg,  # trustor_avg_sent: final average sending amount
            'trustee_avg_returned': final_trustee_avg,  # trustee_avg_returned: final average returning amount
            'cooperation_rate': final_trustee_avg / (final_trustor_avg * 3) if final_trustor_avg > 0 else 0  # cooperation_rate: final cooperation level
        },
        'partnership_data': partnership_sequences,  # partnership_data: complete interaction sequences
        'epsilon_evolution': epsilon_stats,  # epsilon_evolution: exploration rate evolution
        'timestamp': datetime.now().isoformat()  # timestamp: experiment completion time
    }
    
    summary_path = os.path.join(save_dir, "experiment_summary.pkl")  # summary_path: path for summary file
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"Saved experiment summary to: {summary_path}")
    return summary


def run_global_reputation_experiment(training_steps=1_500_000):
    """
    Execute complete training experiment with simplified cooperation-only reputation system.
    
    Implements multi-agent trust game with reputation-based partner selection,
    wealth accumulation, and comprehensive data collection for analysis.
    
    Args:
        training_steps: Number of training iterations to execute
        
    Returns:
        Dictionary containing all experimental results and data
    """
    
    print(f"\n Starting Simplified Cooperation-Only Reputation Experiment")
    print(f"Training steps: {training_steps:,}")
    print(f"Cooperation-only reputation (0-10 scale)")
    print("="*60)
    
    # Training hyperparameters
    LOG_INTERVAL = 10_000  # LOG_INTERVAL: frequency of progress logging
    LEARNING_WARMUP = 200  # LEARNING_WARMUP: minimum interactions before learning starts
    LEARNING_FREQUENCY = 2  # LEARNING_FREQUENCY: learn every N interactions per partnership
    
    # System-wide cooperation tracking parameters
    COOPERATION_SAMPLE_INTERVAL = 5_000  # COOPERATION_SAMPLE_INTERVAL: frequency of cooperation sampling
    COOPERATION_WINDOW = 2_500  # COOPERATION_WINDOW: sliding window size for cooperation analysis
    
    total_start_time = time.time()  # total_start_time: experiment start timestamp
    
    # Data collection structures
    partnership_sequences = defaultdict(list)  # partnership_sequences: interaction history per partnership
    cooperation_timeline = []  # cooperation_timeline: population cooperation evolution
    reputation_timeline = []  # reputation_timeline: reputation distribution evolution
    system_cooperation_timeline = []  # system_cooperation_timeline: system-wide cooperation metrics
    pair_counts = defaultdict(int)  # pair_counts: interaction frequency per partnership
    agent_counts = defaultdict(int)  # agent_counts: interaction frequency per agent
    
    # Partner selection tracking
    reputation_selections = defaultdict(int)  # reputation_selections: selection frequency by reputation level
    
    # Main training loop
    for step in range(training_steps):  # step: current training iteration
        
        # Trustor selection (random from all trustors)
        trustor = random.choice(all_trustors)  # trustor: randomly selected agent to initiate interaction
        
        # Reputation-based trustee selection
        available_trustees = all_trustees  # available_trustees: all potential trustee partners
        
        # Select trustee based on reputation-weighted probability
        trustee = trustor.select_partner_by_reputation(available_trustees)  # trustee: reputation-selected partner agent
        
        # Track selection patterns for analysis
        trustee_reputation = global_reputation_manager.get_reputation(trustee.agent_id)  # trustee_reputation: selected trustee's reputation score
        reputation_selections[f"rep_{trustee_reputation:.1f}"] += 1
        
        # Track interaction frequency patterns
        pair_key = tuple(sorted((trustor.agent_id, trustee.agent_id)))  # pair_key: unique partnership identifier
        pair_counts[pair_key] += 1
        agent_counts[trustor.agent_id] += 1
        agent_counts[trustee.agent_id] += 1
        
        # Retrieve interaction history between these specific agents
        trustor_history = trustor.partner_memory[trustee.agent_id]  # trustor_history: trustor's memory of this trustee
        trustee_history = trustee.partner_memory[trustor.agent_id]  # trustee_history: trustee's memory of this trustor
        first_interaction = trustor_history['last_sent'] is None  # first_interaction: whether agents have interacted before
        
        # Action selection with reputation awareness
        if first_interaction:
            # Random actions for first interaction
            action_trustor = np.random.randint(0, 11)  # action_trustor: amount trustor sends (0-10)
            action_trustee = np.random.randint(0, 31)  # action_trustee: amount trustee returns (0-30)
    
            # Apply constraint: cannot return more than 3x received
            max_possible_return = action_trustor * 3  # max_possible_return: maximum returnable amount
            action_trustee = min(action_trustee, max_possible_return)

        else:
            # Use learned policies with reputation context
            
            # Trustor observation: previous interaction history
            obs_trustor = np.array([  # obs_trustor: trustor's state observation
                trustor_history['last_sent'],  # amount sent in previous interaction
                trustor_history['last_received']  # amount received in previous interaction
            ], dtype=np.float32)
            
            # Get trustor action with reputation enhancement
            action_trustor = trustor.choose_action_with_reputation(obs_trustor, trustee.agent_id)
            
            # Trustee observation: current amount received
            obs_trustee = np.array([action_trustor], dtype=np.float32)  # obs_trustee: trustee's state observation
            
            # Get trustee action with reputation enhancement
            action_trustee = trustee.choose_action_with_reputation(obs_trustee, trustor.agent_id)

            # Apply return constraint
            max_possible_return = action_trustor * 3
            action_trustee = min(action_trustee, max_possible_return)
                    
        # Calculate trust game rewards
        reward_trustor = 10 - action_trustor + action_trustee  # reward_trustor: trustor payoff (endowment - sent + returned)
        reward_trustee = 3 * action_trustor - action_trustee  # reward_trustee: trustee payoff (3x received - returned)
        
        # Update agent wealth accumulation
        trustor.update_wealth(reward_trustor, step)
        trustee.update_wealth(reward_trustee, step)

        # Update reputation scores using simplified cooperation-only system
        
        # Update trustee reputation based on cooperation behavior
        global_reputation_manager.update_reputation_complete(
            agent_id=trustee.agent_id,
            sent_amount=action_trustor,
            returned_amount=action_trustee,
            partner_id=trustor.agent_id,
            role='trustee',
            step=step
        )
        
        # Update trustor reputation based on risk-taking behavior
        global_reputation_manager.update_reputation_complete(
            agent_id=trustor.agent_id,
            sent_amount=action_trustor,
            returned_amount=action_trustee,
            partner_id=trustee.agent_id,
            role='trustor',
            step=step
        )
        
        # Update partner interaction memory
        trustor.store_partner_interaction(trustee.agent_id, action_trustor, action_trustee)
        trustee.store_partner_interaction(trustor.agent_id, action_trustee, action_trustor)
        
        # Enhanced data collection for comprehensive analysis
        partnership_key = (trustor.agent_id, trustee.agent_id)  # partnership_key: unique partnership identifier
        partnership_sequences[partnership_key].append({
            'interaction_number': len(partnership_sequences[partnership_key]) + 1,  # interaction_number: sequence within partnership
            'step': step,  # step: global training step
            'sent': action_trustor,  # sent: amount sent by trustor
            'returned': action_trustee,  # returned: amount returned by trustee
            'trustor_reward': reward_trustor,  # trustor_reward: trustor's payoff
            'trustee_reward': reward_trustee,  # trustee_reward: trustee's payoff
            'trustor_reputation': global_reputation_manager.get_reputation(trustor.agent_id),  # trustor_reputation: current trustor reputation
            'trustee_reputation': global_reputation_manager.get_reputation(trustee.agent_id),  # trustee_reputation: current trustee reputation
            'reputation_difference': global_reputation_manager.get_reputation(trustee.agent_id) - global_reputation_manager.get_reputation(trustor.agent_id),  # reputation_difference: reputation gap between agents
            'trustor_wealth': trustor.current_wealth,  # trustor_wealth: trustor's accumulated wealth
            'trustee_wealth': trustee.current_wealth,  # trustee_wealth: trustee's accumulated wealth
            'wealth_gap': abs(trustor.current_wealth - trustee.current_wealth),  # wealth_gap: absolute wealth difference
            'cooperation_rate': action_trustee / (3 * action_trustor) if action_trustor > 0 else 0  # cooperation_rate: fraction of possible return
        })
        
        # Track reputation evolution over time
        if step % 10000 == 0 and step > 0:
            reputation_stats = global_reputation_manager.get_reputation_stats()  # reputation_stats: population reputation metrics
            reputation_timeline.append({
                'step': step,  # step: current training step
                'reputation_stats': reputation_stats,  # reputation_stats: population statistics
                'selection_patterns': dict(reputation_selections)  # selection_patterns: partner selection frequency by reputation
            })
        
        # System-wide cooperation analysis using sliding window
        if step % COOPERATION_SAMPLE_INTERVAL == 0 and step > 0:
            # Collect recent interactions across all partnerships
            recent_interactions = []  # recent_interactions: cooperation rates in recent window
            
            # Calculate cutoff for sliding window analysis
            cutoff_step = step - COOPERATION_WINDOW  # cutoff_step: earliest step to include in analysis
            
            # Collect cooperation rates from recent interactions
            for partnership_data in partnership_sequences.values():  # partnership_data: interaction history for one partnership
                for interaction in partnership_data:  # interaction: individual interaction record
                    # Only consider interactions within the sliding window
                    if interaction['step'] >= cutoff_step and interaction['sent'] > 0:
                        coop_rate = interaction['returned'] / (3 * interaction['sent'])  # coop_rate: cooperation rate for this interaction
                        recent_interactions.append(min(1.0, coop_rate))  # cap cooperation rate at 1.0
            
            # Calculate comprehensive system-wide metrics
            if recent_interactions:
                mean_coop = np.mean(recent_interactions)  # mean_coop: average cooperation rate
                std_coop = np.std(recent_interactions)  # std_coop: cooperation rate standard deviation
                high_coop_fraction = sum(c > 0.7 for c in recent_interactions) / len(recent_interactions)  # high_coop_fraction: fraction with high cooperation
                medium_coop_fraction = sum(0.4 <= c <= 0.7 for c in recent_interactions) / len(recent_interactions)  # medium_coop_fraction: fraction with medium cooperation
                low_coop_fraction = sum(c < 0.4 for c in recent_interactions) / len(recent_interactions)  # low_coop_fraction: fraction with low cooperation
                
                # Calculate reputation-cooperation correlation
                recent_reps = []  # recent_reps: trustee reputations in recent interactions
                recent_coops = []  # recent_coops: cooperation rates in recent interactions
                for partnership_data in partnership_sequences.values():
                    for interaction in partnership_data:
                        if interaction['step'] >= cutoff_step and interaction['sent'] > 0:
                            recent_reps.append(interaction['trustee_reputation'])
                            recent_coops.append(min(1.0, interaction['returned'] / (3 * interaction['sent'])))
                
                rep_coop_corr = np.corrcoef(recent_reps, recent_coops)[0,1] if len(recent_reps) > 1 else 0  # rep_coop_corr: correlation between reputation and cooperation
                
                system_cooperation_timeline.append({
                    'step': step,  # step: current training step
                    'mean_cooperation': mean_coop,  # mean_cooperation: average cooperation rate
                    'cooperation_std': std_coop,  # cooperation_std: cooperation standard deviation
                    'cooperation_variance': std_coop**2,  # cooperation_variance: cooperation variance
                    'high_cooperation_fraction': high_coop_fraction,  # high_cooperation_fraction: high cooperation percentage
                    'medium_cooperation_fraction': medium_coop_fraction,  # medium_cooperation_fraction: medium cooperation percentage
                    'low_cooperation_fraction': low_coop_fraction,  # low_cooperation_fraction: low cooperation percentage
                    'total_interactions': len(recent_interactions),  # total_interactions: number of interactions analyzed
                    'reputation_cooperation_correlation': rep_coop_corr  # reputation_cooperation_correlation: rep-coop correlation
                })
        
        # Original cooperation timeline for compatibility
        if step % 5000 == 0 and step > 0:
            recent_interactions = []  # recent_interactions: recent cooperation rates for legacy tracking
            for partnership, sequence in partnership_sequences.items():  # partnership: partnership identifier, sequence: interaction sequence
                if len(sequence) >= 5:
                    recent_5 = sequence[-5:]  # recent_5: last 5 interactions for this partnership
                    for interaction in recent_5:
                        if interaction['sent'] > 0:
                            cooperation_rate = interaction['returned'] / (interaction['sent'] * 3)  # cooperation_rate: cooperation rate for interaction
                            recent_interactions.append(min(1.0, cooperation_rate))
            
            if recent_interactions:
                cooperation_timeline.append({
                    'step': step,  # step: current training step
                    'avg_cooperation': np.mean(recent_interactions),  # avg_cooperation: average cooperation rate
                    'cooperation_std': np.std(recent_interactions)  # cooperation_std: cooperation standard deviation
                })
        
        # Learning with reputation-enhanced states
        if not first_interaction:
            # Base observations for learning (without reputation)
            obs_trustor_curr = np.array([  # obs_trustor_curr: trustor's current state
                trustor_history['last_sent'] if trustor_history['last_sent'] is not None else 0,
                trustor_history['last_received'] if trustor_history['last_received'] is not None else 0,
            ], dtype=np.float32)
            
            obs_trustee_curr = np.array([action_trustor], dtype=np.float32)  # obs_trustee_curr: trustee's current state
            
            # Next states after interaction
            trustor_next_hist = trustor.partner_memory[trustee.agent_id]  # trustor_next_hist: updated trustor interaction history
            next_obs_trustor = np.array([  # next_obs_trustor: trustor's next state
                trustor_next_hist['last_sent'],
                trustor_next_hist['last_received'],
            ], dtype=np.float32)
            
            next_obs_trustee = np.array([trustor.choose_action_with_reputation(next_obs_trustor, trustee.agent_id)], dtype=np.float32)  # next_obs_trustee: trustee's next state
            
            # Store transitions with reputation enhancement
            trustor.store_transition_with_reputation(
                obs_trustor_curr, action_trustor, reward_trustor, next_obs_trustor, trustee.agent_id
            )
            trustee.store_transition_with_reputation(
                obs_trustee_curr, action_trustee, reward_trustee, next_obs_trustee, trustor.agent_id
            )
        
        # Partnership-specific learning updates
        partnership_interaction_count = trustor.partner_memory[trustee.agent_id]['interaction_count']  # partnership_interaction_count: interactions between this pair
        
        if (partnership_interaction_count > LEARNING_WARMUP and 
            partnership_interaction_count % LEARNING_FREQUENCY == 0):
            trustor.learn(trustee.agent_id)
            trustee.learn(trustor.agent_id)
        
        # Progress logging and monitoring
        if step % LOG_INTERVAL == 0:
            elapsed = (time.time() - total_start_time) / 60  # elapsed: training time in minutes
            progress = step / training_steps * 100  # progress: completion percentage
            
            print(f"Step {step:,}/{training_steps:,} ({progress:.1f}%) | Elapsed: {elapsed:.1f} min")
            
            # Show system cooperation metrics when available
            if system_cooperation_timeline:
                latest_coop = system_cooperation_timeline[-1]  # latest_coop: most recent cooperation metrics
                print(f"  System cooperation: {latest_coop['mean_cooperation']:.3f} ± {latest_coop['cooperation_std']:.3f}")
                print(f"  High cooperation: {latest_coop['high_cooperation_fraction']*100:.1f}%")
            
            if step % 20000 == 0 and step > 0:
                print(f"  Recent: Trustor {trustor.agent_id} -> Trustee {trustee.agent_id}")
                print(f"  Actions: Sent {action_trustor}, Returned {action_trustee}")
                
                # Show reputation information
                trustor_rep = global_reputation_manager.get_reputation(trustor.agent_id)  # trustor_rep: trustor's current reputation
                trustee_rep = global_reputation_manager.get_reputation(trustee.agent_id)  # trustee_rep: trustee's current reputation
                print(f"  Reputations: Trustor {trustor_rep:.1f}, Trustee {trustee_rep:.1f}")
                
                # Show population reputation statistics
                rep_stats = global_reputation_manager.get_reputation_stats()  # rep_stats: population reputation metrics
                print(f"  Population reputation: μ={rep_stats['mean']:.1f}, σ={rep_stats['std']:.1f}")
    
    # Final experiment statistics and analysis
    duration = time.time() - total_start_time  # duration: total experiment time in seconds
    hours = duration / 3600  # hours: total experiment time in hours
    
    print(f"\nSimplified Cooperation-Only Reputation Experiment Complete!")
    print("="*60)
    print(f"Training time: {hours:.2f} hours")
    print(f"Training steps: {training_steps:,}")
    print(f"Partnerships tracked: {len(partnership_sequences)}")
    
    # Final reputation analysis
    final_rep_stats = global_reputation_manager.get_reputation_stats()  # final_rep_stats: final reputation distribution
    print(f"\nFinal Reputation Distribution (0-10 scale):")
    print(f"  Mean: {final_rep_stats['mean']:.1f}")
    print(f"  Std: {final_rep_stats['std']:.1f}")
    print(f"  Range: [{final_rep_stats['min']:.1f}, {final_rep_stats['max']:.1f}]")
    
    # Reputation stratification analysis
    all_reps = list(final_rep_stats['all_reputations'].values())  # all_reps: all final reputation scores
    high_rep_agents = [aid for aid, rep in final_rep_stats['all_reputations'].items() if rep > 7.0]  # high_rep_agents: agents with high reputation
    low_rep_agents = [aid for aid, rep in final_rep_stats['all_reputations'].items() if rep < 3.0]  # low_rep_agents: agents with low reputation
    
    print(f"\nReputation Stratification:")
    print(f"  High reputation agents (>7.0): {len(high_rep_agents)} agents")
    print(f"  Low reputation agents (<3.0): {len(low_rep_agents)} agents")
    print(f"  Expected: Clear reputation diversity based on cooperation behavior")
    
    # Final cooperation summary
    if system_cooperation_timeline:
        final_coop = system_cooperation_timeline[-1]  # final_coop: final cooperation metrics
        print(f"\nFinal Cooperation Analysis:")
        print(f"  Mean cooperation rate: {final_coop['mean_cooperation']:.3f}")
        print(f"  High cooperation agents: {final_coop['high_cooperation_fraction']*100:.1f}%")
        print(f"  Reputation-cooperation correlation: {final_coop['reputation_cooperation_correlation']:.3f}")
    
    return {
        'partnership_sequences': partnership_sequences,  # partnership_sequences: complete interaction data per partnership
        'cooperation_timeline': cooperation_timeline,  # cooperation_timeline: legacy cooperation tracking
        'system_cooperation_timeline': system_cooperation_timeline,  # system_cooperation_timeline: comprehensive cooperation evolution
        'reputation_timeline': reputation_timeline,  # reputation_timeline: reputation distribution evolution
        'reputation_manager': global_reputation_manager,  # reputation_manager: final reputation system state
        'agents': all_agents,  # agents: all trained agents
        'training_stats': {
            'duration_hours': hours,  # duration_hours: total training time
            'training_steps': training_steps,  # training_steps: completed training iterations
            'partnerships': len(partnership_sequences),  # partnerships: number of unique partnerships
            'final_system_cooperation': system_cooperation_timeline[-1] if system_cooperation_timeline else None  # final_system_cooperation: final cooperation state
        }
    }


def save_complete_experiment_data(all_run_results, save_folder="./7_sep_10x2"):
    """
    Save comprehensive experiment data for complete analysis across multiple runs.
    
    Creates CSV files and safe pickle data for notebook analysis, handling
    potential serialization issues with DQN agents by extracting key data.
    
    Args:
        all_run_results: List of experiment results from multiple runs
        save_folder: Directory to save all analysis files
        
    Returns:
        Tuple of DataFrames for immediate analysis
    """
    print("SAVING COMPLETE EXPERIMENT DATA")
    print("="*50)
    
    os.makedirs(save_folder, exist_ok=True)
    
    # Agent data across all runs
    agent_data = []  # agent_data: final agent states across all runs
    for run_idx, result in enumerate(all_run_results):  # run_idx: run number, result: experiment result dictionary
        agents = result['agents']  # agents: list of trained agents
        reputation_manager = result['reputation_manager']  # reputation_manager: reputation system
        
        for agent in agents:  # agent: individual trained agent
            agent_data.append({
                'run': run_idx + 1,  # run: experiment run number
                'agent_id': agent.agent_id,  # agent_id: unique agent identifier
                'role': 'trustor' if agent.agent_id < 20 else 'trustee',  # role: agent type based on ID
                'wealth': agent.current_wealth,  # wealth: final accumulated wealth
                'reputation': reputation_manager.get_reputation(agent.agent_id)  # reputation: final reputation score
            })
    
    agent_df = pd.DataFrame(agent_data)  # agent_df: DataFrame of agent final states
    agent_df.to_csv(os.path.join(save_folder, "agent_data.csv"), index=False)
    
    # Partnership interaction data across all runs
    partnership_data = []  # partnership_data: all interactions across all runs
    for run_idx, result in enumerate(all_run_results):
        for (trustor_id, trustee_id), interactions in result['partnership_sequences'].items():  # trustor_id, trustee_id: partnership identifiers, interactions: interaction list
            for interaction in interactions:  # interaction: individual interaction record
                partnership_data.append({
                    'run': run_idx + 1,  # run: experiment run number
                    'trustor_id': trustor_id,  # trustor_id: trustor agent identifier
                    'trustee_id': trustee_id,  # trustee_id: trustee agent identifier
                    'interaction_number': interaction['interaction_number'],  # interaction_number: sequence within partnership
                    'step': interaction['step'],  # step: global training step
                    'sent': interaction['sent'],  # sent: amount sent by trustor
                    'returned': interaction['returned'],  # returned: amount returned by trustee
                    'trustor_reputation': interaction['trustor_reputation'],  # trustor_reputation: trustor's reputation at time
                    'trustee_reputation': interaction['trustee_reputation'],  # trustee_reputation: trustee's reputation at time
                    'cooperation_rate': interaction['cooperation_rate'],  # cooperation_rate: fraction of possible return
                    'trustor_wealth': interaction['trustor_wealth'],  # trustor_wealth: trustor's wealth at time
                    'trustee_wealth': interaction['trustee_wealth']  # trustee_wealth: trustee's wealth at time
                })
    
    partnership_df = pd.DataFrame(partnership_data)  # partnership_df: DataFrame of all interactions
    partnership_df.to_csv(os.path.join(save_folder, "partnership_interactions.csv"), index=False)
    
    # Partnership summary statistics
    partnership_summary = []  # partnership_summary: aggregated partnership statistics
    for run_idx, result in enumerate(all_run_results):
        for (trustor_id, trustee_id), interactions in result['partnership_sequences'].items():
            if len(interactions) >= 5:  # only partnerships with meaningful interaction history
                cooperation_rates = [i['cooperation_rate'] for i in interactions if i['sent'] > 0]  # cooperation_rates: cooperation rates for this partnership
                sent_amounts = [i['sent'] for i in interactions]  # sent_amounts: amounts sent in this partnership
                returned_amounts = [i['returned'] for i in interactions]  # returned_amounts: amounts returned in this partnership
                
                partnership_summary.append({
                    'run': run_idx + 1,  # run: experiment run number
                    'trustor_id': trustor_id,  # trustor_id: trustor agent identifier
                    'trustee_id': trustee_id,  # trustee_id: trustee agent identifier
                    'total_interactions': len(interactions),  # total_interactions: number of interactions in partnership
                    'avg_cooperation': np.mean(cooperation_rates) if cooperation_rates else 0,  # avg_cooperation: average cooperation rate
                    'avg_sent': np.mean(sent_amounts),  # avg_sent: average amount sent
                    'avg_returned': np.mean(returned_amounts),  # avg_returned: average amount returned
                    'final_trustor_rep': interactions[-1]['trustor_reputation'],  # final_trustor_rep: trustor's final reputation
                    'final_trustee_rep': interactions[-1]['trustee_reputation']  # final_trustee_rep: trustee's final reputation
                })
    
    partnership_summary_df = pd.DataFrame(partnership_summary)  # partnership_summary_df: DataFrame of partnership statistics
    partnership_summary_df.to_csv(os.path.join(save_folder, "partnership_summary.csv"), index=False)
    
    # System timeline data for cooperation evolution analysis
    timeline_data = []  # timeline_data: system-wide cooperation evolution across runs
    for run_idx, result in enumerate(all_run_results):
        if 'system_cooperation_timeline' in result and result['system_cooperation_timeline']:
            for point in result['system_cooperation_timeline']:  # point: cooperation measurement at specific time
                timeline_data.append({
                    'run': run_idx + 1,  # run: experiment run number
                    **point  # point: all cooperation metrics for this time point
                })
    
    timeline_df = pd.DataFrame(timeline_data)  # timeline_df: DataFrame of cooperation evolution
    timeline_df.to_csv(os.path.join(save_folder, "cooperation_timeline.csv"), index=False)
    
    # Safe results for pickle storage (excludes problematic agent objects)
    safe_results = []  # safe_results: experiment data without agent objects
    for run_idx, result in enumerate(all_run_results):
        safe_result = {  # safe_result: experiment result with serializable data only
            'run': run_idx + 1,  # run: experiment run number
            'partnership_sequences': result['partnership_sequences'],  # partnership_sequences: interaction data
            'cooperation_timeline': result['cooperation_timeline'],  # cooperation_timeline: legacy cooperation tracking
            'system_cooperation_timeline': result['system_cooperation_timeline'],  # system_cooperation_timeline: comprehensive cooperation data
            'reputation_timeline': result['reputation_timeline'],  # reputation_timeline: reputation evolution
            'training_stats': result['training_stats'],  # training_stats: training metadata
            'final_reputation_stats': result['reputation_manager'].get_reputation_stats()  # final_reputation_stats: final reputation distribution
        }
        safe_results.append(safe_result)
    
    print(f"COMPLETE DATA SAVED to {save_folder}:")
    print(f"  • agent_data.csv - Final agent states across all runs")
    print(f"  • partnership_interactions.csv - ALL transactions across all runs")
    print(f"  • partnership_summary.csv - Partnership-level statistics")
    print(f"  • cooperation_timeline.csv - System evolution across runs")
    
    return agent_df, partnership_df, partnership_summary_df, timeline_df


# Main Multi-Run Experiment Execution
print("STARTING MULTI-RUN EXPERIMENT SEQUENCE")
print("="*60)

# Execute multiple experimental runs
all_run_results = []  # all_run_results: results from all experimental runs
no_runs = 1  # no_runs: number of experimental runs to execute

for i in range(no_runs):  # i: current run index
    print(f"\nStarting Run {i+1}/{no_runs}")
    print("-" * 40)
    
    # Initialize fresh components for this run
    print("Creating fresh agents and reputation manager...")
    
    # Create new reputation manager for this run
    global_reputation_manager = RenZengGlobalReputationManager(
        num_agents=40,  # num_agents: total agents in system
        theta=0.2,  # theta: reputation update rate
        initial_reputation=5.0  # initial_reputation: starting reputation (middle of 0-10 scale)
    )
    
    # Create fresh trustor agents with random initialization
    all_trustors = [
        WealthTrackingGlobalReputationDQNAgent( 
            n_actions=11,  # n_actions: trustor action space (send 0-10)
            n_features=2,  # n_features: trustor state space size
            learning_rate=0.0016,  # learning_rate: neural network learning rate
            reward_decay=0.75,  # reward_decay: Q-learning discount factor
            epsilon_start=1.0,  # epsilon_start: initial exploration rate
            epsilon_min=0.05,  # epsilon_min: minimum exploration rate
            epsilon_increment=0.0001,  # epsilon_increment: exploration decay rate
            replace_target_iter=3000,  # replace_target_iter: target network update frequency
            memory_size=50000,  # memory_size: experience replay buffer size
            batch_size=64,  # batch_size: neural network training batch size
            agent_id=j,  # agent_id: unique trustor identifier (0-19)
            i=j,  # i: alternative agent identifier
            partner_ids=list(range(20, 40)),  # partner_ids: potential trustee partners (20-39)
            reputation_manager=global_reputation_manager,  # reputation_manager: global reputation system
            selection_bias=3  # selection_bias: reputation weighting in partner selection
        ) for j in range(20)  # j: trustor creation index
    ]
    
    # Create fresh trustee agents with random initialization
    all_trustees = [
        WealthTrackingGlobalReputationDQNAgent( 
            n_actions=31,  # n_actions: trustee action space (return 0-30)
            n_features=1,  # n_features: trustee state space size
            learning_rate=0.0016,  # learning_rate: neural network learning rate
            reward_decay=0.75,  # reward_decay: Q-learning discount factor
            epsilon_start=1.0,  # epsilon_start: initial exploration rate
            epsilon_min=0.05,  # epsilon_min: minimum exploration rate
            epsilon_increment=0.0001,  # epsilon_increment: exploration decay rate
            replace_target_iter=3000,  # replace_target_iter: target network update frequency
            memory_size=50000,  # memory_size: experience replay buffer size
            batch_size=64,  # batch_size: neural network training batch size
            agent_id=j+20,  # agent_id: unique trustee identifier (20-39)
            i=j,  # i: alternative agent identifier
            partner_ids=list(range(0, 20)),  # partner_ids: potential trustor partners (0-19)
            reputation_manager=global_reputation_manager,  # reputation_manager: global reputation system
            selection_bias=3  # selection_bias: reputation weighting in partner selection
        ) for j in range(20)  # j: trustee creation index
    ]
    
    all_agents = all_trustors + all_trustees  # all_agents: combined list of all agents
    print(f"Created {len(all_agents)} fresh agents with random weights")
    
    # Execute experiment with fresh agents
    result = run_global_reputation_experiment(training_steps=10_000_000)  # result: complete experimental results
    all_run_results.append(result)
    
    print(f"Run {i+1} completed!")

# Save comprehensive experimental data for analysis
print("="*60)
save_complete_experiment_data(all_run_results, save_folder="./7_sep_10x1")  # save all experimental data to specified folder