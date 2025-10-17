# frozenlake_env.py
# Enhanced Frozen Lake environment with Q-values overlay,
# episodes/iterations display, custom images, and keyboard shortcuts.

from contextlib import closing
from io import StringIO
from typing import List, Optional
import numpy as np
import pygame
import os
import pickle

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.utils import seeding

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [(0, 0)], set()
    while frontier:
        r, c = frontier.pop()
        if (r, c) not in discovered:
            discovered.add((r, c))
            for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= max_size or cc < 0 or cc >= max_size:
                    continue
                if board[rr][cc] == "G":
                    return True
                if board[rr][cc] != "H":
                    frontier.append((rr, cc))
    return False

def generate_random_map(size: int = 8, p: float = 0.8, seed: Optional[int] = None) -> List[str]:
    np_random, _ = seeding.np_random(seed)
    valid, board = False, None
    while not valid:
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        board[0][0], board[-1][-1] = "S", "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board]

class FrozenLakeEnv(Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, desc=None, map_name="4x4", is_slippery=True):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = self.desc.shape
        self.reward_range = (0, 1)

        nS, nA = self.nrow * self.ncol, 4
        self.initial_state_distrib = (self.desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(r, c): return r * self.ncol + c
        def inc(r, c, a):
            if a == LEFT: c = max(c - 1, 0)
            elif a == DOWN: r = min(r + 1, self.nrow - 1)
            elif a == RIGHT: c = min(c + 1, self.ncol - 1)
            elif a == UP: r = max(r - 1, 0)
            return r, c
        def update(r, c, a):
            nr, nc = inc(r, c, a)
            ns, letter = to_s(nr, nc), self.desc[nr, nc]
            done, reward = bytes(letter) in b"GH", float(letter == b"G")
            return ns, reward, done

        for r in range(self.nrow):
            for c in range(self.ncol):
                s = to_s(r, c)
                for a in range(nA):
                    li, letter = self.P[s][a], self.desc[r, c]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append((1 / 3, *update(r, c, b)))
                        else:
                            li.append((1.0, *update(r, c, a)))

        self.observation_space, self.action_space = spaces.Discrete(nS), spaces.Discrete(nA)
        self.render_mode = render_mode

        self.window_surface, self.clock = None, None
        self.window_size = (1000, 600)
        grid_width = int(self.window_size[0] * 0.67)
        grid_height = self.window_size[1]
        self.grid_size = (grid_width, grid_height)
        self.cell_size = (grid_width // self.ncol, grid_height // self.nrow)

        try:
            self.ice_img = pygame.transform.scale(pygame.image.load("tile.png"), self.cell_size)
            self.hole_img = pygame.transform.scale(pygame.image.load("frzngm1.png"), self.cell_size)
            self.goal_img = pygame.transform.scale(pygame.image.load("treasure.png"), self.cell_size)
            self.start_img = self.ice_img
            self.agent_img = pygame.transform.scale(pygame.image.load("teddy.png"), self.cell_size)
            self.has_images = True
        except:
            self.has_images = False
            print(" Warning: Image files not found. Using colored squares instead.")

        self.q_table, self.episode, self.iteration_text = None, "---", "---"
        self.pygame_initialized, self.text_padding = False, 5
        self.q_values_visible = True
        self.s, self.lastaction = 0, None

        self.paused = False
        self.show_grid = True
        self.highlight_best = True

    def _render_gui(self):
        if self.window_surface is None:
            pygame.init()
            self.pygame_initialized = True
            pygame.display.init()
            pygame.display.set_caption("Frozen Lake")
            self.window_surface = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
            self.ui_font = pygame.font.SysFont("Courier", 18)
            self.q_font = pygame.font.SysFont("Courier", 14)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                elif event.key == pygame.K_9:
                    self.q_values_visible = not self.q_values_visible
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                elif event.key == pygame.K_h:
                    self.highlight_best = not self.highlight_best
                elif event.key == pygame.K_s:
                    if self.q_table is not None:
                        with open("q_table_saved.pkl", "wb") as f:
                            pickle.dump(self.q_table, f)
                elif event.key == pygame.K_l:
                    try:
                        with open("q_table_saved.pkl", "rb") as f:
                            self.q_table = pickle.load(f)
                    except:
                        print(" Failed to load saved Q-table.")
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_EQUALS:
                    self.metadata["render_fps"] += 10
                elif event.key == pygame.K_MINUS:
                    self.metadata["render_fps"] = max(1, self.metadata["render_fps"] - 10)
                elif event.key == pygame.K_0:
                    self.metadata["render_fps"] = 0
                elif event.key == pygame.K_1:
                    self.metadata["render_fps"] = 4

        self.window_surface.fill((255, 255, 255))

        for row in range(self.nrow):
            for col in range(self.ncol):
                rect = pygame.Rect(col * self.cell_size[0], row * self.cell_size[1], *self.cell_size)
                s = row * self.ncol + col
                tile = self.desc[row, col].decode("utf-8")

                if self.has_images:
                    img = {
                        "S": self.start_img,
                        "F": self.ice_img,
                        "H": self.hole_img,
                        "G": self.goal_img
                    }.get(tile, self.ice_img)
                    self.window_surface.blit(img, rect)
                else:
                    color = {
                        "S": (173, 216, 230),
                        "F": (200, 200, 255),
                        "H": (50, 50, 50),
                        "G": (255, 215, 0)
                    }.get(tile, (200, 200, 255))
                    pygame.draw.rect(self.window_surface, color, rect)

                if self.show_grid:
                    pygame.draw.rect(self.window_surface, (180, 180, 180), rect, 1)

                if self.q_values_visible and self.q_table is not None and tile not in ['H', 'G']:
                    qvals = self.q_table[s]
                    for action, q in enumerate(qvals):
                        pos = list(rect.center)
                        x_offset = self.cell_size[0] // 3
                        y_offset = self.cell_size[1] // 3
                        offset = {
                            UP: (0, -y_offset),
                            DOWN: (0, y_offset),
                            LEFT: (-x_offset, 0),
                            RIGHT: (x_offset, 0)
                        }[action]
                        pos[0] += offset[0]
                        pos[1] += offset[1]

                        q_text = f"{q:.2f}" if abs(q) < 10 else f"{q:.1f}"
                        text = self.q_font.render(q_text, True, (0, 0, 0))
                        text_rect = text.get_rect(center=pos)
                        self.window_surface.blit(text, text_rect)

                    if self.highlight_best:
                        best_a = np.argmax(qvals)
                        best_pos = list(rect.center)
                        best_offset = {
                            UP: (0, -y_offset//2),
                            DOWN: (0, y_offset//2),
                            LEFT: (-x_offset//2, 0),
                            RIGHT: (x_offset//2, 0)
                        }[best_a]
                        best_pos[0] += best_offset[0]
                        best_pos[1] += best_offset[1]
                        pygame.draw.circle(self.window_surface, (255, 0, 0), best_pos, 5)

        agent_row, agent_col = self.s // self.ncol, self.s % self.ncol
        agent_rect = pygame.Rect(agent_col * self.cell_size[0], agent_row * self.cell_size[1], *self.cell_size)
        if self.has_images:
            self.window_surface.blit(self.agent_img, agent_rect)
        else:
            pygame.draw.circle(self.window_surface, (255, 0, 0), agent_rect.center, min(agent_rect.width, agent_rect.height) // 3)

        panel_x = self.grid_size[0] + 20
        lines = [
            "Shortcuts (Number Row):",
            "1 : Reset FPS",
            "0 : Unlimited FPS",
            "- : Decrease FPS",
            "= : Increase FPS",
            "9 : Toggle Q-values",
            "g : Toggle Grid",
            "h : Toggle Best Action",
            "p : Pause/Resume",
            "s : Save Q-table",
            "l : Load Q-table",
            "r : Reset Env",
            "ESC : Quit",
            "----------------------",
            f"Episode: {self.episode}",
            f"Iteration: {self.iteration_text}",
            f"State: {self.s}"
        ]

        for i, line in enumerate(lines):
            label = self.ui_font.render(line, True, (0, 0, 0))
            self.window_surface.blit(label, (panel_x, 30 + 25 * i))

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.s, self.lastaction = categorical_sample(self.initial_state_distrib, self.np_random), None
        if self.render_mode == "human" and hasattr(self, "_render_gui"):
            self._render_gui()
        return int(self.s), {"prob": 1}

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        elif self.render_mode in {"human", "rgb_array"}:
            return self._render_gui()

    def step(self, a):
        if self.paused:
            return int(self.s), 0, False, False, {}

        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s, self.lastaction = s, a
        if self.render_mode == "human":
            self._render_gui()
        return int(s), r, t, False, {"prob": p}

    def _render_text(self):
        desc = [[c.decode("utf-8") for c in line] for line in self.desc.tolist()]
        row, col = self.s // self.ncol, self.s % self.ncol
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        outfile = StringIO()
        outfile.write("\n".join("".join(line) for line in desc) + "\n")
        with closing(outfile): return outfile.getvalue()

    def set_q(self, q_table): 
        self.q_table = q_table

    def set_episode(self, episode):
        self.episode = episode

    def set_iteration(self, iteration_text): 
        self.iteration_text = iteration_text
