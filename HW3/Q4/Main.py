

import argparse
import numpy as np

from frozenlake_env import FrozenLakeEnv
from q_learning_agent import QLearningAgent
from dp_agents import value_iteration, policy_iteration

def evaluate_policy_rollout(env, policy, episodes=100):
    successes = 0
    for _ in range(episodes):
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result
        done = False
        while not done:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward == 1:
                successes += 1
                break
    return successes / episodes

#def run_dp(env, mode, episodes=100, gamma=0.99, theta=1e-8, max_iters=1000):
    #if mode == "vi":
        #print("\n Running Value Iteration...")
        #policy, V = value_iteration(env, gamma=gamma, theta=theta,
                                    max_iters=max_iters, render_env=True)
        #print("[VI] Converged.")
        #success_rate = evaluate_policy_rollout(env, policy, episodes=episodes)
        #print(f"[DP-VI] Success rate over {episodes} episodes: {success_rate:.2f}")
    #elif mode == "pi":
        #print("\n Running Policy Iteration...")
        #policy, V = policy_iteration(env, gamma=gamma, theta=theta,
                                     max_iters=max_iters, render_env=True)
        #print("[PI] Converged.")
        #success_rate = evaluate_policy_rollout(env, policy, episodes=episodes)
        #print(f"[DP-PI] Success rate over {episodes} episodes: {success_rate:.2f}")#

def run_q_learning(env, episodes=5000, render=False):
    print("\n  Running Q-learning...")
    agent = QLearningAgent(env)
    q_table, _, _ = agent.train(num_episodes=episodes, render_env=render)
    success_rate = evaluate_q_policy(env, q_table, episodes=100)
    print(f"Success rate (last 100 greedy eval): {success_rate:.2f}")
    print(" Congratulations! You've successfully solved FrozenLake!")

def evaluate_q_policy(env, q_table, episodes=100):
    successes = 0
    for _ in range(episodes):
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result
        done = False
        while not done:
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward == 1:
                successes += 1
                break
    return successes / episodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["q", "vi", "pi"], required=True,
                        help="Choose algorithm: q (Q-learning), vi (Value Iteration), pi (Policy Iteration)")
    parser.add_argument("--map", choices=["4x4", "8x8"], default="4x4")
    parser.add_argument("--slippery", action="store_true",
                        help="Use stochastic (slippery) environment")
    parser.add_argument("--episodes", type=int, default=5000,
                        help="Number of training episodes (Q-learning only)")
    parser.add_argument("--render", action="store_true",
                        help="Enable graphical rendering")
    args = parser.parse_args()

    env = FrozenLakeEnv(render_mode="human" if args.render else None,
                        map_name=args.map,
                        is_slippery=args.slippery)

    print(f"\n Starting FrozenLake {args.map} — {'Stochastic (slippery)' if args.slippery else 'Deterministic (non-slippery)'} Environment")

    if args.mode in ["vi", "pi"]:
        run_dp(env, args.mode, episodes=100)
    elif args.mode == "q":
        run_q_learning(env, episodes=args.episodes, render=args.render)

if __name__ == "__main__":
    main()
