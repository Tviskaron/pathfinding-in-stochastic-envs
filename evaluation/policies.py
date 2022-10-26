import json
from os.path import join
from pathlib import Path

import torch

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import AttrDict, log

from evaluation.eval_utils import ResultsHolder
from planning.decentralized import AStarHolder, StochasticAStarHolder
from training_run import validate_config, register_custom_components, make_pogema
from utils.config_validation import Environment


class APPOHolder:
    def __init__(self, algo_cfg):
        self.cfg = algo_cfg

        path = algo_cfg.path_to_weights
        device = algo_cfg.device
        register_custom_components()

        self.path = path
        self.env = None
        config_path = join(path, 'cfg.json')
        with open(config_path, "r") as f:
            config = json.load(f)
        exp, flat_config = validate_config(config['full_config'])
        algo_cfg = flat_config

        env = create_env(algo_cfg.env, cfg=algo_cfg, env_config={})

        actor_critic = create_actor_critic(algo_cfg, env.observation_space, env.action_space)
        env.close()

        if device == 'cpu' or not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
        self.device = device

        # actor_critic.share_memory()
        actor_critic.model_to_device(device)
        policy_id = algo_cfg.policy_index
        checkpoints = join(path, f'checkpoint_p{policy_id}')
        checkpoints = LearnerWorker.get_checkpoints(checkpoints)
        checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])

        self.ppo = actor_critic
        self.device = device
        self.cfg = algo_cfg

        self.rnn_states = None

    def after_reset(self, env):
        self.env = env

    @staticmethod
    def get_additional_info():
        return {"rl_used": 1.0}

    def get_name(self):
        return Path(self.path).name

    def act(self, observations, rewards=None, dones=None, infos=None):
        if self.rnn_states is None or len(self.rnn_states) != len(observations):
            self.rnn_states = torch.zeros([len(observations), get_hidden_size(self.cfg)], dtype=torch.float32,
                                          device=self.device)

        with torch.no_grad():
            obs_torch = AttrDict(transform_dict_observations(observations))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(self.device).float()
            policy_outputs = self.ppo(obs_torch, self.rnn_states, with_action_distribution=True)
            self.rnn_states = policy_outputs.rnn_states
            actions = policy_outputs.actions

        return actions.cpu().numpy()

    def after_step(self, dones):
        for agent_i, done_flag in enumerate(dones):
            if done_flag:
                self.rnn_states[agent_i] = torch.zeros([get_hidden_size(self.cfg)], dtype=torch.float32,
                                                       device=self.device)
        if all(dones):
            self.rnn_states = None


def run_algo(eval_configs):
    algo_cfg = eval_configs[0].algo
    results = []

    if algo_cfg.name == 'APPO':
        algo = APPOHolder(algo_cfg)
    elif algo_cfg.name == 'A*':
        algo = AStarHolder(algo_cfg)
    elif algo_cfg.name == 'SA*':
        algo = StochasticAStarHolder(algo_cfg)
    else:
        raise KeyError(f"No algo with type {algo_cfg.name}")

    for cfg in eval_configs:
        env = make_pogema(Environment(**cfg.environment.dict()))

        obs = env.reset()
        algo.after_reset(env)
        results_holder = ResultsHolder()

        dones = [False for _ in range(len(obs))]
        infos = [{'is_active': True} for _ in range(len(obs))]
        rew = [0 for _ in range(len(obs))]
        with torch.no_grad():
            while True:
                obs, rew, dones, infos = env.step(algo.act(obs, rew, dones, infos))
                infos[0] = {**infos[0], **algo.get_additional_info()}
                results_holder.after_step(infos)
                algo.after_step(dones)

                if all(dones):
                    break

        results.append(results_holder.get_final())
        env.close()

    return results
