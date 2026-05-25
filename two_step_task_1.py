"""
============================================================
The Two-Step Task: Model-Free vs. Model-Based RL
"The Shelby Trap" — Daw Two-Step Paradigm Implementation
============================================================
"""

import numpy as np
import matplotlib
#matplotlib.use("Agg")  # Not needed in Colab/Jupyter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

np.random.seed(42)

# ============================================================
# STAGE 1: THE TWO-STEP MDP ENVIRONMENT
# ============================================================

class TwoStepEnvironment:
    """
    A lightweight MDP with 3 states and 2 steps per episode.

    State 0 (Hub): Choose A_1 or A_2.
        A_1 -> State 1 with prob 0.8 (Common), State 2 with prob 0.2 (Rare)
        A_2 -> State 2 with prob 0.8 (Common), State 1 with prob 0.2 (Rare)

    States 1 & 2 (Terminals): Any action yields reward from hidden probability.
        State 1 reward prob: 0.2
        State 2 reward prob: 0.8
    """

    def __init__(self):
        self.n_states = 3          # 0=Hub, 1=Terminal-1, 2=Terminal-2
        self.n_actions = 2         # A_1=0, A_2=1

        # Transition probabilities from State 0
        # P[action][next_state] = probability
        self.transition_probs = {
            0: {1: 0.8, 2: 0.2},   # A_1: 80% -> State1, 20% -> State2
            1: {2: 0.8, 1: 0.2},   # A_2: 80% -> State2, 20% -> State1
        }

        # Hidden reward probabilities for terminal states
        self.reward_probs = {
            1: 0.2,
            2: 0.8,
        }

        self.current_state = 0
        self.step_count = 0

    def reset(self):
        self.current_state = 0
        self.step_count = 0
        return self.current_state

    def step(self, action):
        """Take an action and return (next_state, reward, done)."""
        done = False
        reward = 0.0

        if self.current_state == 0:
            # Step 1: Hub -> Terminal transition
            probs = self.transition_probs[action]
            next_state = np.random.choice(
                list(probs.keys()),
                p=list(probs.values())
            )
            self.current_state = next_state
            self.step_count += 1
            return next_state, 0.0, False

        else:
            # Step 2: Terminal -> Reward
            reward = float(np.random.random() < self.reward_probs[self.current_state])
            done = True
            return self.current_state, reward, done

    def forced_step(self, action, force_next_state=None):
        """
        Forced trajectory for Stage 3.
        force_next_state overrides the stochastic transition.
        """
        if self.current_state == 0:
            if force_next_state is not None:
                next_state = force_next_state
            else:
                probs = self.transition_probs[action]
                next_state = np.random.choice(
                    list(probs.keys()), p=list(probs.values())
                )
            self.current_state = next_state
            return next_state, 0.0, False
        else:
            reward = float(np.random.random() < self.reward_probs[self.current_state])
            return self.current_state, reward, True


# ============================================================
# STAGE 2: TABULAR Q-LEARNING AGENT
# ============================================================

class QLearningAgent:
    """
    Standard tabular Q-Learning agent.
    Bellman Update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    """

    def __init__(self, n_states=3, n_actions=2, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Initialize Q-table to zeros
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        """Standard Q-Learning (off-policy TD) update."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        # Bellman update
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def get_q_values(self):
        return self.Q.copy()


def train_agent(env, agent, n_episodes=2000, verbose=False):
    """Train the Q-Learning agent on the two-step environment."""
    rewards_history = []
    q_history = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        rewards_history.append(total_reward)
        q_history.append(agent.get_q_values()[0].copy())  # Track State 0 Q-values

        if verbose and (episode + 1) % 500 == 0:
            print(f"Episode {episode+1:4d} | "
                  f"Q(S0,A1)={agent.Q[0,0]:.4f} | "
                  f"Q(S0,A2)={agent.Q[0,1]:.4f}")

    return rewards_history, q_history


# ============================================================
# STAGE 3: THE BEHAVIORAL PROBE — FORCED TRAJECTORY
# ============================================================

def run_forced_trajectory():
    """
    Forced sequence:
      1. State 0 -> Action A_1
      2. RARE transition -> State 2  (even though A_1 usually goes to State 1)
      3. Take action in State 2 -> Reward R=1

    Then print Q-values for State 0.
    """
    print("\n" + "="*60)
    print("STAGE 3: THE BEHAVIORAL PROBE — FORCED TRAJECTORY")
    print("="*60)

    env = TwoStepEnvironment()
    # Fresh agent with higher alpha to see clear update
    agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.0)

    print("\nInitial Q-values (State 0):")
    print(f"  Q(S0, A_1) = {agent.Q[0,0]:.4f}")
    print(f"  Q(S0, A_2) = {agent.Q[0,1]:.4f}")

    # ---- STEP 1: State 0, take A_1 ----
    s0 = env.reset()
    action_step1 = 0  # A_1

    # ---- STEP 2: RARE transition -> State 2 ----
    next_state, r1, done = env.forced_step(action_step1, force_next_state=2)
    print(f"\nStep 1: State {s0} --[A_1]--> State {next_state} "
          f"(RARE transition! A_1 usually goes to State 1)")

    # Q-update for Step 1: reward=0, next_state=2
    agent.update(s0, action_step1, r1, next_state, done)

    # ---- STEP 3: Action in State 2 -> Reward=1 ----
    s2 = next_state
    action_step2 = 0  # any action
    # Force reward = 1
    reward_step2 = 1.0
    agent.update(s2, action_step2, reward_step2, s2, done=True)
    print(f"Step 2: State {s2} --[any action]--> Reward = {reward_step2}")

    # Now: State 2's Q-values are updated. Re-run Step 1 update with the
    # improved Q(State2) to propagate back to State 0.
    # (In a real episode, we'd do this in sequence; here we show the effect.)
    agent.update(s0, action_step1, 0.0, next_state, done=False)

    print("\nQ-values for State 0 AFTER forced trajectory:")
    print(f"  Q(S0, A_1) = {agent.Q[0,0]:.4f}  <-- INCREASED (Q-learning credits A_1)")
    print(f"  Q(S0, A_2) = {agent.Q[0,1]:.4f}")
    print(f"\n  Agent PREFERENCE: {'A_1' if agent.Q[0,0] > agent.Q[0,1] else 'A_2'}")
    print("\n  *** MODEL-FREE BLINDSPOT: The agent increased Q(S0, A_1) even though")
    print("      A_1 RARELY leads to State 2. A rational agent would prefer A_2! ***")

    return agent.Q.copy()


# ============================================================
# STAGE 4: MODEL-BASED AGENT (for comparison)
# ============================================================

class ModelBasedAgent:
    """
    Model-Based agent that explicitly stores and uses transition probabilities.
    Computes V(State_0) using the full Bellman expectation equation.
    """

    def __init__(self, n_states=3, n_actions=2, gamma=0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma

        # Learned transition counts: T[s][a][s'] = count
        self.T_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # Learned reward estimates: R[s][a] = mean reward
        self.R_sum = defaultdict(lambda: defaultdict(float))
        self.R_count = defaultdict(lambda: defaultdict(int))

        # True model (for demo; in practice these would be learned)
        self.true_transitions = {
            0: {0: {1: 0.8, 2: 0.2}, 1: {1: 0.2, 2: 0.8}},
        }
        self.true_rewards = {1: 0.2, 2: 0.8}

    def compute_V_model_based(self):
        """
        V(s1) = R(s1) = 0.2
        V(s2) = R(s2) = 0.8

        V(s0) = max_a  sum_{s'} P(s'|s0, a) * V(s')
        """
        V = {
            1: self.true_rewards[1],
            2: self.true_rewards[2],
        }

        # For each action at State 0
        Q_s0 = {}
        for action, trans in self.true_transitions[0].items():
            Q_s0[action] = sum(
                prob * V[s_prime]
                for s_prime, prob in trans.items()
            )
            action_name = f"A_{action+1}"
            print(f"  Q_MB(S0, {action_name}) = "
                  + " + ".join([f"{prob:.1f}*V(S{s}) " for s, prob in trans.items()])
                  + f"= {Q_s0[action]:.4f}")

        V[0] = max(Q_s0.values())
        return V, Q_s0


# ============================================================
# VISUALIZATION
# ============================================================

def plot_results(q_history, rewards_history):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Q-Learning on Two-Step MDP (Daw Task)", fontsize=14, fontweight='bold')

    # Plot 1: Q-values at State 0 over training
    q_arr = np.array(q_history)
    axes[0].plot(q_arr[:, 0], label='Q(S0, A_1)', color='royalblue', alpha=0.8)
    axes[0].plot(q_arr[:, 1], label='Q(S0, A_2)', color='tomato', alpha=0.8)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Q-value")
    axes[0].set_title("Q-values at State 0 During Training")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Expected optimal: Q(A_2) > Q(A_1) since A_2 reliably reaches high-reward State 2
    axes[0].axhline(y=0.8*0.8 + 0.2*0.2, color='tomato', linestyle='--', alpha=0.5,
                    label='Expected Q(A_2)')
    axes[0].axhline(y=0.8*0.2 + 0.2*0.8, color='royalblue', linestyle='--', alpha=0.5,
                    label='Expected Q(A_1)')
    axes[0].legend(fontsize=8)

    # Plot 2: Smoothed rewards
    window = 50
    smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    axes[1].plot(smoothed, color='mediumseagreen')
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward (smoothed)")
    axes[1].set_title(f"Average Reward (rolling window={window})")
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=0.8*0.8 + 0.2*0.2, color='gray', linestyle='--', alpha=0.7,
                    label=f'Optimal ≈ {0.8*0.8+0.2*0.2:.2f}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()  # Shows inline in Jupyter/Colab
    plt.close()
    print("\nTraining curves saved to training_curves.png")


def plot_mdp_diagram():
    """Draw the MDP structure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title("Two-Step MDP Structure (Daw Task)", fontsize=14, fontweight='bold')

    # Nodes
    circle_hub = plt.Circle((2, 3), 0.7, color='steelblue', zorder=5)
    circle_s1  = plt.Circle((7, 5), 0.7, color='mediumseagreen', zorder=5)
    circle_s2  = plt.Circle((7, 1), 0.7, color='tomato', zorder=5)

    ax.add_patch(circle_hub)
    ax.add_patch(circle_s1)
    ax.add_patch(circle_s2)

    ax.text(2, 3, 'S0\nHub', ha='center', va='center', fontsize=10,
            fontweight='bold', color='white', zorder=6)
    ax.text(7, 5, 'S1\nR=0.2', ha='center', va='center', fontsize=10,
            fontweight='bold', color='white', zorder=6)
    ax.text(7, 1, 'S2\nR=0.8', ha='center', va='center', fontsize=10,
            fontweight='bold', color='white', zorder=6)

    # Arrows A_1
    ax.annotate('', xy=(6.3, 4.6), xytext=(2.7, 3.4),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=2))
    ax.text(4.2, 4.3, 'A₁: 0.8 (Common)', fontsize=9, color='steelblue')

    ax.annotate('', xy=(6.3, 1.4), xytext=(2.7, 2.6),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5, linestyle='dashed'))
    ax.text(3.8, 1.7, 'A₁: 0.2 (Rare)', fontsize=9, color='steelblue', style='italic')

    # Arrows A_2
    ax.annotate('', xy=(6.3, 1.35), xytext=(2.7, 2.55),
                arrowprops=dict(arrowstyle='->', color='tomato', lw=2))
    ax.text(3.8, 2.2, 'A₂: 0.8 (Common)', fontsize=9, color='tomato')

    ax.annotate('', xy=(6.3, 4.55), xytext=(2.7, 3.35),
                arrowprops=dict(arrowstyle='->', color='tomato', lw=1.5, linestyle='dashed'))
    ax.text(4.0, 3.8, 'A₂: 0.2 (Rare)', fontsize=9, color='tomato', style='italic')

    blue_patch  = mpatches.Patch(color='steelblue', label='Action A₁')
    red_patch   = mpatches.Patch(color='tomato',    label='Action A₂')
    ax.legend(handles=[blue_patch, red_patch], loc='lower right')

    plt.tight_layout()
    plt.savefig('mdp_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()  # Shows inline in Jupyter/Colab
    plt.close()
    print("MDP diagram saved to mdp_diagram.png")


# ============================================================
# HYBRID AGENT DEMO
# ============================================================

def hybrid_agent_demo():
    """
    Demonstrate the hybrid Model-Free + Model-Based decision variable.
    Q_hybrid(s,a) = (1-w)*Q_MF(s,a) + w*Q_MB(s,a)
    """
    print("\n" + "="*60)
    print("HYBRID AGENT DEMONSTRATION")
    print("="*60)

    # After forced trajectory, MF agent prefers A_1 (wrong)
    Q_MF = np.array([0.16, 0.0])   # approx after forced trajectory
    # MB agent correctly computes
    Q_MB = np.array([0.8*0.2 + 0.2*0.8, 0.8*0.8 + 0.2*0.2])  # [0.32, 0.68]

    print(f"\nModel-Free  Q(S0): A_1={Q_MF[0]:.4f}, A_2={Q_MF[1]:.4f}")
    print(f"Model-Based Q(S0): A_1={Q_MB[0]:.4f}, A_2={Q_MB[1]:.4f}")

    weights = [0.0, 0.2, 0.5, 0.8, 1.0]
    print("\n  w (MB weight) | Q_hybrid(A_1) | Q_hybrid(A_2) | Preference")
    print("  " + "-"*55)
    for w in weights:
        Q_h = (1 - w) * Q_MF + w * Q_MB
        pref = "A_1 (WRONG)" if Q_h[0] > Q_h[1] else "A_2 (CORRECT)"
        print(f"  w={w:.1f}          |  {Q_h[0]:.4f}       |  {Q_h[1]:.4f}       | {pref}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("THE TWO-STEP TASK — DAW PARADIGM IMPLEMENTATION")
    print("Demonstrating the Shelby Trap in Tabular Q-Learning")
    print("="*60)

    # Draw MDP
    plot_mdp_diagram()

    # Stage 1+2: Train
    print("\n--- STAGE 2: TRAINING Q-LEARNING AGENT ---")
    env   = TwoStepEnvironment()
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.15)

    print("Training for 2000 episodes...")
    rewards_history, q_history = train_agent(env, agent, n_episodes=2000, verbose=True)

    print("\nFinal Q-values after training:")
    print(f"  Q(S0, A_1) = {agent.Q[0,0]:.4f}")
    print(f"  Q(S0, A_2) = {agent.Q[0,1]:.4f}")
    print(f"  Q(S1, A_1) = {agent.Q[1,0]:.4f}")
    print(f"  Q(S2, A_1) = {agent.Q[2,0]:.4f}")

    expected_q_a1 = 0.8*0.2 + 0.2*0.8
    expected_q_a2 = 0.8*0.8 + 0.2*0.2
    print(f"\n  Expected Q(S0, A_1) = {expected_q_a1:.4f}")
    print(f"  Expected Q(S0, A_2) = {expected_q_a2:.4f}")
    print(f"  Correct preference: A_2 (leads reliably to high-reward State 2)")

    plot_results(q_history, rewards_history)

    # Stage 3: Forced trajectory
    Q_after_forced = run_forced_trajectory()

    # Stage 4: Model-Based comparison
    print("\n" + "="*60)
    print("MODEL-BASED AGENT — Correct Computation")
    print("="*60)
    mb_agent = ModelBasedAgent()
    V, Q_mb = mb_agent.compute_V_model_based()
    print(f"\n  Model-Based correctly prefers: {'A_1' if Q_mb[0] > Q_mb[1] else 'A_2'}")
    print(f"  V(S0) = max(Q_MB) = {max(Q_mb.values()):.4f}")

    # Hybrid
    hybrid_agent_demo()

    print("\n" + "="*60)
    print("All stages complete. See training_curves.png and mdp_diagram.png")
    print("="*60)
