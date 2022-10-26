from pogema import GridConfig
import random

from evaluation.eval_settings import AlgoDecentralized
from planning.astar_no_grid import AStar
from planning.astar_schochastic import AStarStochastic

INF = 1000000007


class DecentralizedAgent:
    def __init__(self, env, use_best_move: bool = True, max_steps: int = INF, inner_algo=AStar):
        self.env = env
        self.use_best_move = use_best_move
        gc: GridConfig = env.config
        self.previous_positions = [[] for _ in range(self.env.config.num_agents)]

        random.seed(env.config.seed)
        self.actions = {tuple(gc.MOVES[i]): i for i in range(len(gc.MOVES))}
        self.steps = 0
        self.planner = [inner_algo(self.env.grid.positions_xy[i], self.env.grid.finishes_xy[i], max_steps) for i in
                        range(gc.num_agents)]

    def act(self, obs, skip_agents=None):

        env = self.env
        cfg = self.env.config
        obs_radius = cfg.obs_radius
        action = []

        for k in range(cfg.num_agents):
            self.previous_positions[k].append(env.grid.positions_xy[k])
            if env.grid.positions_xy[k][0] == env.grid.finishes_xy[k][0] and env.grid.positions_xy[k][1] == \
                    env.grid.finishes_xy[k][1]:
                action.append(None)
                continue
            new_obs = obs[k][0]
            new_agents = obs[k][1]
            if len(new_obs) != obs_radius:
                new_obs = obs[k][0].copy()
                c = len(new_agents) // 2
                new_obs = new_obs[c - obs_radius: c + obs_radius + 1, c - obs_radius: c + obs_radius + 1]
            if len(new_agents) != obs_radius:
                new_agents = obs[k][1].copy()
                c = len(new_agents) // 2
                new_agents = new_agents[c - obs_radius: c + obs_radius + 1, c - obs_radius: c + obs_radius + 1]

            self.planner[k].update_obstacles(new_obs, new_agents, (
                self.env.grid.positions_xy[k][0] - obs_radius, self.env.grid.positions_xy[k][1] - obs_radius))

            if skip_agents and skip_agents[k]:
                action.append(None)
                continue

            self.planner[k].update_path(env.grid.positions_xy[k][0], env.grid.positions_xy[k][1])
            path = self.planner[k].get_next_node(self.use_best_move)
            if path:
                action.append(self.actions[(path[1][0] - path[0][0], path[1][1] - path[0][1])])
            else:
                action.append(None)
        self.steps += 1
        return action


class FixNonesWrapper:

    def __init__(self, agent):
        self.agent = agent
        self.env = agent.env

    def act(self, obs, skip_agents=None):
        actions = self.agent.act(obs, skip_agents=skip_agents)
        for idx in range(len(actions)):
            if actions[idx] is None:
                actions[idx] = 0
        return actions


class AStarHolder:
    def __init__(self, cfg: AlgoDecentralized):
        self.cfg = cfg
        self.agent = None
        self.fix_loops = cfg.fix_loops
        self.fix_nones = cfg.fix_nones
        self.stay_if_loop_prob = cfg.stay_if_loop_prob
        self.no_path_random = cfg.no_path_random
        self.use_best_move = cfg.use_best_move
        self.add_none_if_loop = cfg.add_none_if_loop

        self.env = None

    @staticmethod
    def get_name():
        return 'decentralized'

    def act(self, observations, rewards=None, dones=None, info=None, skip_agents=None):
        return self.agent.act(observations, skip_agents)

    def after_step(self, dones):
        if all(dones):
            self.agent = None

    def after_reset(self, env, ):
        self.env = env
        self.agent = DecentralizedAgent(env, use_best_move=self.use_best_move, max_steps=self.cfg.max_planning_steps)

        if self.fix_nones:
            self.agent = FixNonesWrapper(self.agent)

    def get_additional_info(self):
        return {"rl_used": 0.0}


class StochasticAStarHolder(AStarHolder):
    def after_reset(self, env, ):
        self.env = env
        self.agent = DecentralizedAgent(env, use_best_move=self.use_best_move, inner_algo=AStarStochastic,
                                        max_steps=self.cfg.max_planning_steps)

        if self.fix_nones:
            self.agent = FixNonesWrapper(self.agent)
