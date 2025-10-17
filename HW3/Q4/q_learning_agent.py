import numpy as np


class QLearningAgent:
    def __init__(
        self,
        env,
        episodes=3000,
        alpha=0.9,
        gamma=0.9,
        epsilon=1.0,
        eps_min=0.0,
        eps_decay=0.0001,
        max_steps=200,
        render=False,
        print_every=100
    ):
        self.env = env
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.max_steps = max_steps
        self.render = render
        self.print_every = print_every
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def train(self, num_episodes=None, render_env=None):
        self.eps_decay = 1 / (num_episodes - 1000)
        if num_episodes is None:
            num_episodes = self.episodes
        if render_env is None:
            render_env = self.render

        rewards = []
        steps_per_episode = []

        for ep in range(num_episodes):
            result = self.env.reset()
            state = result[0] if isinstance(result, tuple) else result
            done, total_reward, steps = False, 0, 0

            #  initial Q-table and episode for rendering
            if render_env and hasattr(self.env, 'render_mode') and self.env.render_mode == "human":
                self.env.set_episode(ep + 1)
                self.env.set_q(self.q_table)
                self.env.render()

            while not done and steps < self.max_steps:
                '''
                ACTIONS:
                0 = LEFT
                1 = DOWN
                2 = RIGHT
                3 = UP
                '''
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state,:])

                new_state, reward, terminated, truncated, _ = self.env.step(action)
                # Compute whether the episode has ended
                done = terminated or truncated

                # Calculate the TD target
                if not terminated:
                   best_next_action = np.argmax(self.q_table[new_state, :])
                   td_target = reward + self.gamma * self.q_table[new_state, best_next_action]
                else:
                   td_target = reward

                # Update the Q-table
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.alpha * td_error

                #  Q-table
                if render_env and hasattr(self.env, 'render_mode') and self.env.render_mode == "human":
                    self.env.set_q(self.q_table)

                    self.env.render()

                state = new_state
                total_reward += reward
                steps += 1

            if self.epsilon > self.eps_min:
                self.epsilon = max(self.epsilon - self.eps_decay, 0)

            rewards.append(total_reward)
            steps_per_episode.append(steps)

            if (ep + 1) % self.print_every == 0:
                avg_reward = np.mean(rewards[-self.print_every:])
                avg_steps = np.mean(steps_per_episode[-self.print_every:])
                success_rate = np.mean(
                    [1 if r > 0 else 0 for r in rewards[-self.print_every:]])

                print(f"[Q] Episode {ep+1}/{num_episodes} | "
                      f"avg_reward({self.print_every})={avg_reward:.2f} | "
                      f"success_rate={success_rate:.2f} | "
                      f"avg_steps={avg_steps:.1f} | "
                      f"eps={self.epsilon:.3f}")

        return self.q_table, rewards, steps_per_episode

    def get_policy(self):
        return np.argmax(self.q_table, axis=1)
