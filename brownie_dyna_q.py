"""
============================================================
BROWNIE POINTS: Dyna-Q (Model-Based World Model) on FrozenLake-v1
============================================================
Dyna-Q integrates a learned world model into Q-Learning.
After each real experience, the agent also "replays" simulated
experiences from its internal model to speed up learning.

Architecture:
  1. Real interaction: update Q and the world model (T, R)
  2. Planning: sample n random (s,a) from model, do Q updates
  3. Decision: epsilon-greedy on Q

This directly demonstrates the Model-Based advantage:
  - Same real samples as Q-Learning
  - Much faster learning due to planning steps
============================================================
"""

import numpy as np
import matplotlib
#matplotlib.use("Agg")  # Not needed in Colab/Jupyter
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import defaultdict

np.random.seed(42)


class DynaQAgent:
    """
    Dyna-Q: Model-Free Q-Learning + Model-Based Planning.

    After each real (s, a, r, s') experience:
      1. Update Q(s,a) via standard TD
      2. Store (s,a) -> (r, s') in model
      3. Randomly sample n_planning (s,a) pairs from model
         and do Q updates on simulated experience

    Reference: Sutton & Barto, Chapter 8
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=0.1, n_planning=10):
        self.n_states    = n_states
        self.n_actions   = n_actions
        self.alpha       = alpha
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.n_planning  = n_planning   # Number of planning steps per real step

        # Q-table
        self.Q = np.zeros((n_states, n_actions))

        # World model: model[s][a] = (reward, next_state)
        self.model = defaultdict(dict)

        # Track which (s,a) pairs have been experienced
        self.experienced_sa = []

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        """Real experience: update Q + model, then plan."""

        # 1. Q-Learning update (real experience)
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

        # 2. Update world model
        self.model[state][action] = (reward, next_state, done)
        if (state, action) not in [(s, a) for s, a in self.experienced_sa]:
            self.experienced_sa.append((state, action))

        # 3. Planning (simulated rollouts from model)
        for _ in range(self.n_planning):
            if not self.experienced_sa:
                break
            # Sample a random (s, a) that the agent has actually visited
            idx = np.random.randint(len(self.experienced_sa))
            s_sim, a_sim = self.experienced_sa[idx]

            if a_sim not in self.model[s_sim]:
                continue

            r_sim, s_next_sim, done_sim = self.model[s_sim][a_sim]

            # Q update from simulated experience
            if done_sim:
                target_sim = r_sim
            else:
                target_sim = r_sim + self.gamma * np.max(self.Q[s_next_sim])
            self.Q[s_sim, a_sim] += self.alpha * (target_sim - self.Q[s_sim, a_sim])


class PureQLearningAgent:
    """Standard Q-Learning (no model, no planning) for comparison."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.Q         = np.zeros((n_states, n_actions))

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])


def train_on_frozenlake(agent_class, agent_kwargs, n_episodes=1000, render=False):
    """Train an agent on FrozenLake-v1 and return success rates."""
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)
    agent = agent_class(**agent_kwargs)

    successes = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_success = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # FrozenLake: reward=1 only on reaching goal
            agent.update(state, action, reward, next_state, done)
            state = next_state

            if reward == 1.0:
                ep_success = 1

        successes.append(ep_success)

    env.close()
    return successes


def compare_agents():
    """Compare Dyna-Q vs Pure Q-Learning on FrozenLake."""
    print("\n" + "="*60)
    print("BROWNIE POINTS: Dyna-Q vs Q-Learning on FrozenLake-v1")
    print("="*60)

    n_episodes = 1000
    n_runs     = 5       # Average over multiple runs for stability
    window     = 50

    dyna_kwargs = dict(
        n_states=16, n_actions=4,
        alpha=0.1, gamma=0.99, epsilon=0.1, n_planning=10
    )
    qlearn_kwargs = dict(
        n_states=16, n_actions=4,
        alpha=0.1, gamma=0.99, epsilon=0.1
    )

    print(f"Running {n_runs} independent runs of {n_episodes} episodes each...")

    dyna_results   = []
    qlearn_results = []

    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...")
        dyna_results.append(train_on_frozenlake(DynaQAgent, dyna_kwargs, n_episodes))
        qlearn_results.append(train_on_frozenlake(PureQLearningAgent, qlearn_kwargs, n_episodes))

    dyna_mean   = np.mean(dyna_results,   axis=0)
    qlearn_mean = np.mean(qlearn_results, axis=0)

    # Smooth
    def smooth(x, w):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    dyna_smooth   = smooth(dyna_mean,   window)
    qlearn_smooth = smooth(qlearn_mean, window)

    final_dyna   = np.mean(dyna_mean[-200:])
    final_qlearn = np.mean(qlearn_mean[-200:])
    print(f"\nFinal success rate (last 200 episodes):")
    print(f"  Dyna-Q (n_planning=10): {final_dyna:.3f}")
    print(f"  Q-Learning:             {final_qlearn:.3f}")

    # Episodes to reach 50% success rate
    def episodes_to_threshold(arr, threshold=0.5, w=50):
        smoothed = smooth(arr, w)
        hits = np.where(smoothed >= threshold)[0]
        return hits[0] + w if len(hits) > 0 else n_episodes

    dyna_thresh   = episodes_to_threshold(dyna_mean)
    qlearn_thresh = episodes_to_threshold(qlearn_mean)
    print(f"\nEpisodes to reach 50% success rate (smoothed):")
    print(f"  Dyna-Q:      {dyna_thresh}")
    print(f"  Q-Learning:  {qlearn_thresh}")
    if qlearn_thresh < n_episodes and dyna_thresh < n_episodes:
        print(f"  Dyna-Q is {qlearn_thresh/dyna_thresh:.1f}x faster!")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    episodes_plot = np.arange(len(dyna_smooth)) + window
    ax.plot(episodes_plot, dyna_smooth,   label='Dyna-Q (n_planning=10)',
            color='royalblue', lw=2)
    ax.plot(episodes_plot, qlearn_smooth, label='Pure Q-Learning',
            color='tomato', lw=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='50% threshold')
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(f"Success Rate (rolling {window}-ep window)", fontsize=12)
    ax.set_title("Dyna-Q vs Q-Learning on FrozenLake-v1 (is_slippery=True)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig('dyna_q_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("\nDyna-Q comparison plot saved to dyna_q_comparison.png")

    return dyna_smooth, qlearn_smooth


if __name__ == "__main__":
    compare_agents()

    print("\n" + "="*60)
    print("DYNA-Q ALGORITHM SUMMARY")
    print("="*60)
    print("""
  Dyna-Q bridges Model-Free and Model-Based RL:

  For each real step (s, a, r, s'):
    1. Q(s,a) <- Q(s,a) + alpha * [r + gamma*max_a' Q(s',a') - Q(s,a)]
    2. Model(s,a) <- (r, s')       [store in world model]
    3. For n_planning steps:
         (s_sim, a_sim) <- random sample from visited (s,a)
         (r_sim, s'_sim) <- Model(s_sim, a_sim)
         Q(s_sim, a_sim) <- TD update using simulated experience

  Key insight: Planning steps are FREE (no env interaction).
  With n_planning=10, agent gets 10x more Q updates per real step.
  The world model acts as a "mental simulation" of the environment.
    """)
