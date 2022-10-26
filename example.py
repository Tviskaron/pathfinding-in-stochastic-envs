from planning.decentralized import AStarHolder, AlgoDecentralized, StochasticAStarHolder
from evaluation.eval_settings import AlgoAPPO
from evaluation.policies import APPOHolder
from stochastic_grid.config import StochasticGridConfig
import gym

import os


def run(use='PPO'):
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['MKL_NUM_THREADS'] = "1"
    if use == 'PPO':
        algo = APPOHolder(algo_cfg=AlgoAPPO(path_to_weights='results/so', ), )
    elif use == 'A*':
        algo = AStarHolder(cfg=AlgoDecentralized(device='cpu'))
    elif use == 'SA*':
        algo = StochasticAStarHolder(cfg=AlgoDecentralized(device='cpu'))
    else:
        raise KeyError(f'No algo with name: {use}')

    for map_name in ['wc3-tranquilpaths']:
        for seed in [0, 1, 2]:
            sgc = StochasticGridConfig(seed=seed, obs_radius=5, max_episode_steps=512,
                                       num_obstacles=16, map_name=map_name,
                                       size_range=[5, 10], shake_r=5,
                                       show_range=[8, 16], hide_range=[8, 16], so_density=0.7, num_agents=1)

            env = gym.make('StochasticGrid-v0', grid_config=sgc, with_animations=True, auto_reset=False, egocentric_idx=0)

            dones = [False, ...]
            obs = env.reset()
            algo.after_reset(env)

            while not all(dones):
                action = algo.act(obs)
                obs, _, dones, info = env.step(action)
                algo.after_step(dones)

            os.makedirs(f'animations/', exist_ok=True)
            env.save_animation(f'animations/{map_name}-{str(seed).zfill(3)}-{use}.svg')


def main():
    run(use='PPO')
    run(use='A*')
    run(use='SA*')


if __name__ == '__main__':
    main()
