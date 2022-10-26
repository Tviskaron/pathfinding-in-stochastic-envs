import time

import numpy as np


class ResultsHolder:
    def __init__(self):
        self.results = dict()

        self.times = {}
        self.isr = []
        self.rl_used = []
        self.step = 0
        self.start_time = time.monotonic()

    def after_step(self, infos):
        self.step += 1

        for agent_idx in range(len(infos)):
            if 'ISR' in infos[agent_idx]['episode_extra_stats']:
                self.isr.append(infos[agent_idx]['episode_extra_stats']['ISR'])

            if 'Done' in infos[agent_idx]['episode_extra_stats']:
                if agent_idx not in self.times:
                    self.times[agent_idx] = self.step

            if 'CSR' in infos[agent_idx]['episode_extra_stats']:
                self.results['CSR'] = infos[agent_idx]['episode_extra_stats']['CSR']

            if 'rl_used' in infos[agent_idx]:
                self.rl_used.append(infos[agent_idx]['rl_used'])

    def get_final(self):
        self.results['FPS'] = round(self.step / (time.monotonic() - self.start_time), 5)
        self.results['flowtime'] = sum(self.times.values())
        self.results['makespan'] = self.step
        self.results['ISR'] = float(np.mean(self.isr))
        if self.rl_used:
            self.results['rl_used'] = float(np.mean(self.rl_used))

        return self.results
