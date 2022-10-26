import re
from copy import deepcopy

import gym
import numpy as np
from pogema.envs import Pogema
from pogema.integrations.sample_factory import IsMultiAgentWrapper, AutoResetWrapper
from pogema.wrappers.metrics import MetricsWrapper
from pogema.wrappers.multi_time_limit import MultiTimeLimit

from stochastic_grid.animation import StochasticAnimationMonitor
from stochastic_grid.config import StochasticGridConfig
from stochastic_grid.custom_maps import MAPS_REGISTRY
from stochastic_grid.grid import StochasticGrid


class StochasticPogema(Pogema):
    def __init__(self, grid_config=StochasticGridConfig(num_agents=2)):
        super().__init__(grid_config)

    def reset(self, **kwargs):
        self.grid: StochasticGrid = StochasticGrid(grid_config=self.config)
        self.active = {agent_idx: True for agent_idx in range(self.config.num_agents)}
        return self._obs()

    def step(self, action: list):
        assert len(action) == self.config.num_agents
        rewards = []

        infos = [dict() for _ in range(self.config.num_agents)]

        dones = []
        for agent_idx in range(self.config.num_agents):
            if self.active[agent_idx]:
                self.grid.move(agent_idx, action[agent_idx])

            on_goal = self.grid.on_goal(agent_idx)
            if on_goal and self.active[agent_idx]:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
            dones.append(on_goal)

        for agent_idx in range(self.config.num_agents):
            if self.grid.on_goal(agent_idx):
                self.grid.hide_agent(agent_idx)
                self.active[agent_idx] = False

            infos[agent_idx]['is_active'] = self.active[agent_idx]

        self.grid.update_stochastic_obstacles()
        obs = self._obs()
        return obs, rewards, dones, infos

    def get_active_obstacles(self):
        result = self.grid.stochastic_obstacles.copy().astype(float)
        result[result >= 1] = 1.0
        result -= self.grid.positions
        result -= self.grid.targets
        result -= self.grid.obstacles

        height, width = result.shape
        r = self.config.obs_radius

        result[:r - 1, r - 1:width - r + 1] -= 1
        result[r - 1:height - r + 1, :r - 1] -= 1
        result[height - r:, r - 1:width - r + 1] -= 1
        result[r - 1:height - r + 1, width - r:] -= 1

        result[result <= 0] = 0.0
        return result

    def _obs(self):
        return [self._get_agents_obs(index) for index in range(self.config.num_agents)]


class MultiMapWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._configs = []
        self._rnd = np.random.default_rng(self.env.config.seed)
        self._curriculum_scores = None
        self._curriculum_tau = None
        self._curriculum_num_obstacles = 0
        self._steps = 0
        pattern = self.env.config.map_name
        if pattern:
            for map_name in MAPS_REGISTRY:
                if re.match(pattern, map_name):
                    cfg = deepcopy(self.env.config)
                    cfg.map = MAPS_REGISTRY[map_name]
                    cfg = StochasticGridConfig(**cfg.dict())
                    self._configs.append(cfg)

    def step(self, action):
        self._steps += 1
        obs, reward, done, infos = self.env.step(action)
        cfg: StochasticGridConfig = self.env.unwrapped.config

        if not cfg.use_curriculum:
            return obs, reward, done, infos

        info = infos[0]['episode_extra_stats']
        if all(done):
            infos[0]['episode_extra_stats'].update(stochastic_obstacles=self._curriculum_num_obstacles)
        if cfg.curriculum_target in infos[0]['episode_extra_stats']:
            # info[cfg.curriculum_target] = 1.0
            if self._curriculum_scores is None:
                self.init_curriculum()
            self._curriculum_scores[self._curriculum_tau % len(self._curriculum_scores)] = info[cfg.curriculum_target]
            self._curriculum_tau += 1

            # print(np.mean(self._curriculum_scores), f"num obstacles {cfg.num_obstacles}", f"steps: {self._steps}")
            if np.mean(self._curriculum_scores) > cfg.curriculum_threshold:
                # exit(0)
                self._curriculum_num_obstacles += 1
                self.init_curriculum()

        return obs, reward, done, infos

    def init_curriculum(self):
        self._curriculum_scores = [0 for _ in range(self.env.unwrapped.config.curriculum_score_horizon)]
        self._curriculum_tau = 0

    def reset(self, **kwargs):
        if self._configs is not None and len(self._configs) > 1:
            cfg = deepcopy(self._configs[self._rnd.integers(0, len(self._configs))])
            cfg.num_obstacles = self._curriculum_num_obstacles
            self.env.unwrapped.config = cfg

        cfg: StochasticGridConfig = self.env.unwrapped.config
        if cfg.use_curriculum:
            if self._curriculum_scores is None:
                self.init_curriculum()

        return self.env.reset(**kwargs)


def make_stochastic_pogema(grid_config, with_animations=False, auto_reset=True, egocentric_idx=None):
    env = StochasticPogema(grid_config=grid_config)
    env = MultiTimeLimit(env, grid_config.max_episode_steps)
    if with_animations:
        env = StochasticAnimationMonitor(env, egocentric_idx=egocentric_idx)

    env = MetricsWrapper(env)

    env.update_group_name(group_name='episode_extra_stats')
    env = IsMultiAgentWrapper(env)
    env = MultiMapWrapper(env)
    if auto_reset:
        env = AutoResetWrapper(env)

    return env
