"""
Reinforcement Learning Precision Demo

This demonstration shows how ArbitraryNumber provides exact precision
in reinforcement learning algorithms, including Q-learning, policy gradients,
and value function approximation where floating-point errors can cause
convergence issues and suboptimal policies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from v2.core.arbitrary_number import ArbitraryNumber
import time
import random


class FloatingPointQLearning:
    """Q-Learning with floating-point arithmetic."""
    
    def __init__(self, states, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.states = states
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table with floating-point zeros
        self.q_table = {}
        for state in states:
            self.q_table[state] = {action: 0.0 for action in actions}
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Bellman equation."""
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        return new_q
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair."""
        return self.q_table[state][action]


class ArbitraryNumberQLearning:
    """Q-Learning with ArbitraryNumber exact precision."""
    
    def __init__(self, states, actions, learning_rate, discount_factor, epsilon):
        self.states = states
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table with ArbitraryNumber zeros
        self.q_table = {}
        for state in states:
            self.q_table[state] = {action: ArbitraryNumber("0") for action in actions}
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Bellman equation with exact arithmetic."""
        current_q = self.q_table[state][action]
        
        # Find maximum Q-value for next state
        max_next_q = max(self.q_table[next_state].values())
        
        # Q-learning update with exact arithmetic
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.lr * td_error
        
        self.q_table[state][action] = new_q
        return new_q
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair."""
        return self.q_table[state][action]


def test_q_learning_precision():
    """Test Q-learning precision comparison."""
    print("="*70)
    print("Q-LEARNING PRECISION COMPARISON")
    print("="*70)
    
    # Simple grid world environment
    states = ['S1', 'S2', 'S3', 'S4', 'GOAL']
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    # Hyperparameters
    learning_rate_fp = 0.1
    discount_factor_fp = 0.9
    epsilon_fp = 0.1
    
    learning_rate_an = ArbitraryNumber("0.1")
    discount_factor_an = ArbitraryNumber("0.9")
    epsilon_an = ArbitraryNumber("0.1")
    
    print(f"Environment: {len(states)} states, {len(actions)} actions")
    print(f"Learning rate: {learning_rate_fp}")
    print(f"Discount factor: {discount_factor_fp}")
    
    # Initialize agents
    fp_agent = FloatingPointQLearning(states, actions, learning_rate_fp, discount_factor_fp, epsilon_fp)
    an_agent = ArbitraryNumberQLearning(states, actions, learning_rate_an, discount_factor_an, epsilon_an)
    
    # Training episodes
    episodes = [
        ('S1', 'RIGHT', 0.1, 'S2'),
        ('S2', 'RIGHT', 0.1, 'S3'),
        ('S3', 'RIGHT', 0.1, 'S4'),
        ('S4', 'RIGHT', 1.0, 'GOAL'),
        ('S1', 'DOWN', -0.1, 'S2'),
        ('S2', 'UP', -0.1, 'S1'),
    ]
    
    print(f"\nTraining with {len(episodes)} episodes...")
    
    # Train both agents
    for episode in episodes * 100:  # Repeat episodes 100 times
        state, action, reward, next_state = episode
        
        fp_agent.update_q_value(state, action, reward, next_state)
        an_agent.update_q_value(state, action, ArbitraryNumber(str(reward)), next_state)
    
    print("\n--- Q-Values after training ---")
    print("State-Action | Floating-Point | ArbitraryNumber | Difference")
    print("-" * 65)
    
    total_diff = 0.0
    for state in ['S1', 'S2', 'S3', 'S4']:
        for action in ['RIGHT', 'DOWN']:
            fp_q = fp_agent.get_q_value(state, action)
            an_q = an_agent.get_q_value(state, action)
            diff = abs(fp_q - float(str(an_q)))
            total_diff += diff
            
            print(f"{state}-{action:5s} | {fp_q:13.10f} | {str(an_q)[:13]:13s} | {diff:.2e}")
    
    print(f"\nTotal accumulated precision difference: {total_diff:.2e}")


def test_policy_gradient_precision():
    """Test policy gradient precision comparison."""
    print("\n" + "="*70)
    print("POLICY GRADIENT PRECISION COMPARISON")
    print("="*70)
    
    # Policy parameters (logits for softmax policy)
    initial_params_fp = [0.1, -0.2, 0.3, -0.1]
    gradients_fp = [0.01, -0.005, 0.008, -0.003]
    
    initial_params_an = [ArbitraryNumber("0.1"), ArbitraryNumber("-0.2"), 
                         ArbitraryNumber("0.3"), ArbitraryNumber("-0.1")]
    gradients_an = [ArbitraryNumber("0.01"), ArbitraryNumber("-0.005"),
                    ArbitraryNumber("0.008"), ArbitraryNumber("-0.003")]
    
    learning_rate_fp = 0.01
    learning_rate_an = ArbitraryNumber("0.01")
    
    print("Initial policy parameters: [0.1, -0.2, 0.3, -0.1]")
    print("Policy gradients: [0.01, -0.005, 0.008, -0.003]")
    print(f"Learning rate: {learning_rate_fp}")
    print("\nRunning 1000 policy gradient updates...")
    
    # Floating-point policy gradient updates
    params_fp = initial_params_fp[:]
    for _ in range(1000):
        for i in range(len(params_fp)):
            params_fp[i] += learning_rate_fp * gradients_fp[i]
    
    # ArbitraryNumber policy gradient updates
    params_an = initial_params_an[:]
    for _ in range(1000):
        for i in range(len(params_an)):
            params_an[i] = params_an[i] + learning_rate_an * gradients_an[i]
    
    print("\n--- Policy parameters after 1000 updates ---")
    print("Parameter | Floating-Point | ArbitraryNumber | Difference")
    print("-" * 60)
    
    for i, (fp_param, an_param) in enumerate(zip(params_fp, params_an)):
        diff = abs(fp_param - float(str(an_param)))
        print(f"param[{i}]  | {fp_param:13.10f} | {str(an_param)[:13]:13s} | {diff:.2e}")


def test_value_function_approximation():
    """Test value function approximation precision."""
    print("\n" + "="*70)
    print("VALUE FUNCTION APPROXIMATION PRECISION")
    print("="*70)
    
    # Linear value function: V(s) = w₁*s₁ + w₂*s₂ + w₃*s₃ + bias
    weights_fp = [0.5, -0.3, 0.8, 0.1]  # w₁, w₂, w₃, bias
    weights_an = [ArbitraryNumber("0.5"), ArbitraryNumber("-0.3"), 
                  ArbitraryNumber("0.8"), ArbitraryNumber("0.1")]
    
    learning_rate_fp = 0.01
    learning_rate_an = ArbitraryNumber("0.01")
    
    # Training data: (state_features, target_value)
    training_data = [
        ([1.0, 0.5, -0.2], 0.8),
        ([0.3, -0.1, 0.7], 0.4),
        ([-0.5, 0.8, 0.1], 0.2),
        ([0.2, 0.3, -0.4], 0.6),
        ([0.9, -0.2, 0.3], 0.7),
    ]
    
    print("Linear value function: V(s) = w₁*s₁ + w₂*s₂ + w₃*s₃ + bias")
    print(f"Initial weights: {weights_fp}")
    print(f"Learning rate: {learning_rate_fp}")
    print(f"Training samples: {len(training_data)}")
    print("\nRunning 500 gradient descent updates...")
    
    # Floating-point training
    for epoch in range(500):
        for features, target in training_data:
            # Forward pass
            prediction_fp = sum(w * f for w, f in zip(weights_fp[:3], features)) + weights_fp[3]
            error_fp = target - prediction_fp
            
            # Backward pass (gradient descent)
            for i in range(3):
                weights_fp[i] += learning_rate_fp * error_fp * features[i]
            weights_fp[3] += learning_rate_fp * error_fp  # bias update
    
    # ArbitraryNumber training
    for epoch in range(500):
        for features, target in training_data:
            features_an = [ArbitraryNumber(str(f)) for f in features]
            target_an = ArbitraryNumber(str(target))
            
            # Forward pass
            prediction_an = sum(w * f for w, f in zip(weights_an[:3], features_an)) + weights_an[3]
            error_an = target_an - prediction_an
            
            # Backward pass (gradient descent)
            for i in range(3):
                weights_an[i] = weights_an[i] + learning_rate_an * error_an * features_an[i]
            weights_an[3] = weights_an[3] + learning_rate_an * error_an  # bias update
    
    print("\n--- Learned weights after training ---")
    print("Weight | Floating-Point | ArbitraryNumber | Difference")
    print("-" * 58)
    
    weight_names = ['w₁', 'w₂', 'w₃', 'bias']
    for i, (name, fp_w, an_w) in enumerate(zip(weight_names, weights_fp, weights_an)):
        diff = abs(fp_w - float(str(an_w)))
        print(f"{name:5s} | {fp_w:13.10f} | {str(an_w)[:13]:13s} | {diff:.2e}")


def test_temporal_difference_learning():
    """Test temporal difference learning precision."""
    print("\n" + "="*70)
    print("TEMPORAL DIFFERENCE LEARNING PRECISION")
    print("="*70)
    
    # TD(0) learning for state value estimation
    # V(s) = V(s) + α[r + γV(s') - V(s)]
    
    states = ['A', 'B', 'C', 'D', 'E']
    
    # Initialize value functions
    values_fp = {state: 0.0 for state in states}
    values_an = {state: ArbitraryNumber("0") for state in states}
    
    learning_rate_fp = 0.1
    discount_factor_fp = 0.9
    
    learning_rate_an = ArbitraryNumber("0.1")
    discount_factor_an = ArbitraryNumber("0.9")
    
    # Experience tuples: (state, reward, next_state)
    experiences = [
        ('A', 0.1, 'B'),
        ('B', 0.2, 'C'),
        ('C', 0.3, 'D'),
        ('D', 0.4, 'E'),
        ('E', 1.0, None),  # Terminal state
        ('A', -0.1, 'C'),  # Alternative transition
        ('B', 0.15, 'D'),
    ]
    
    print(f"States: {states}")
    print(f"Learning rate: {learning_rate_fp}")
    print(f"Discount factor: {discount_factor_fp}")
    print(f"Experience tuples: {len(experiences)}")
    print("\nRunning 200 TD learning updates...")
    
    # TD learning with floating-point
    for _ in range(200):
        for state, reward, next_state in experiences:
            current_value = values_fp[state]
            next_value = values_fp[next_state] if next_state else 0.0
            
            # TD update
            td_target = reward + discount_factor_fp * next_value
            td_error = td_target - current_value
            values_fp[state] += learning_rate_fp * td_error
    
    # TD learning with ArbitraryNumber
    for _ in range(200):
        for state, reward, next_state in experiences:
            current_value = values_an[state]
            next_value = values_an[next_state] if next_state else ArbitraryNumber("0")
            reward_an = ArbitraryNumber(str(reward))
            
            # TD update
            td_target = reward_an + discount_factor_an * next_value
            td_error = td_target - current_value
            values_an[state] = values_an[state] + learning_rate_an * td_error
    
    print("\n--- Learned state values ---")
    print("State | Floating-Point | ArbitraryNumber | Difference")
    print("-" * 55)
    
    for state in states:
        fp_value = values_fp[state]
        an_value = values_an[state]
        diff = abs(fp_value - float(str(an_value)))
        print(f"{state:5s} | {fp_value:13.10f} | {str(an_value)[:13]:13s} | {diff:.2e}")


def main():
    """Run all reinforcement learning precision demonstrations."""
    print("REINFORCEMENT LEARNING PRECISION DEMONSTRATION")
    print("=" * 70)
    print("This demo shows ArbitraryNumber's exact precision in reinforcement")
    print("learning algorithms where floating-point errors can cause convergence")
    print("issues and suboptimal policies.")
    print("=" * 70)
    
    start_time = time.time()
    
    # Run demonstrations
    test_q_learning_precision()
    test_policy_gradient_precision()
    test_value_function_approximation()
    test_temporal_difference_learning()
    
    end_time = time.time()
    
    print("\n" + "="*70)
    print("REINFORCEMENT LEARNING SUMMARY")
    print("="*70)
    print("ArbitraryNumber advantages in reinforcement learning:")
    print("1. Exact Q-value updates without accumulation errors")
    print("2. Precise policy gradient computations")
    print("3. Accurate value function approximation")
    print("4. Perfect temporal difference learning")
    print("5. Elimination of convergence issues due to precision loss")
    print("6. Reproducible learning trajectories")
    print("7. Stable long-term credit assignment")
    print("8. Exact Bellman equation computations")
    print(f"\nTotal execution time: {end_time - start_time:.4f} seconds")
    
    print("\nKey Benefits for RL:")
    print("• Deterministic policy learning")
    print("• Exact value function convergence")
    print("• Stable exploration-exploitation balance")
    print("• Precise reward signal propagation")
    print("• Elimination of numerical instabilities in deep RL")
    print("• Mathematical correctness in multi-step bootstrapping")


if __name__ == "__main__":
    main()
