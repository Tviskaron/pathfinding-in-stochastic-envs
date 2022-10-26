import os
from copy import deepcopy

import drawSvg
from pogema.animation import AnimationMonitor, GridHolder, AnimationSettings


class StochasticAnimationMonitor(AnimationMonitor):
    def __init__(self, env, animation_settings=AnimationSettings(obstacle_color='#bccacf', time_scale=0.14), egocentric_idx=None):
        super().__init__(env, animation_settings, egocentric_idx)
        self.so_history = None

    def step(self, action):
        obs, reward, dones, info = self.env.step(action)

        self.dones_history.append(dones)
        self.agents_xy_history.append(deepcopy(self.env.grid.positions_xy))
        self.so_history.append(self.env.get_active_obstacles())
        if all(dones):
            if not os.path.exists(self.cfg.directory):
                os.makedirs(self.cfg.directory, exist_ok=True)
            self.save_animation(name=self.cfg.directory + self.pick_name(self.grid_cfg, self._episode_idx))

        return obs, reward, dones, info

    def create_animation(self, egocentric_idx):
        render = super().create_animation(egocentric_idx)

        if egocentric_idx is None:
            egocentric_idx = self.egocentric_idx

        obstacles = self.env.get_obstacles()

        episode_length = len(self.dones_history)
        if egocentric_idx is not None:
            for step_idx, dones in enumerate(self.dones_history):
                if dones[egocentric_idx]:
                    episode_length = min(len(self.dones_history), step_idx + 1)
                    break
        gh = GridHolder(agents_xy=self.env.grid.get_agents_xy(), width=len(obstacles), height=len(obstacles[0]),
                        episode_length=episode_length)
        so = self.create_so(gh, self.so_history)

        self.animate_so(so, gh, self.so_history)
        for obj in so:
            render.append(obj)

        return render

    def clear_animation_info(self):
        self.dones_history = [[False for _ in range(self.env.config.num_agents)]]
        self.agents_xy_history = [deepcopy(self.env.grid.positions_xy)]
        self.so_history = [self.env.get_active_obstacles()]

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.clear_animation_info()
        return obs

    def create_so(self, gh, so_history):

        cfg = self.cfg

        result = []
        for i in range(gh.height):
            for j in range(gh.width):
                x, y = self.fix_point(i, j, gh.width)

                obs_settings = {}
                r = cfg.r + 0
                obs_settings.update(x=cfg.draw_start + i * cfg.scale_size - r,
                                    y=cfg.draw_start + j * cfg.scale_size - r,
                                    width=r * 2,
                                    height=r * 2,
                                    rx=cfg.rx,
                                    fill='#ee9873',
                                    stroke=cfg.obstacle_color,
                                    stroke_width=0,
                                    )

                obs_settings.update(opacity='1.0' if so_history[0][x][y] else "0.0")

                result.append(drawSvg.Rectangle(**obs_settings))

        return result

    def animate_so(self, so, gh, so_history):
        for i in range(gh.height):
            for j in range(gh.width):
                x, y = self.fix_point(i, j, gh.width)
                obstacle = so[i * gh.width + j]

                # obstacle.appendAnim(self.compressed_anim('visibility', ['visible'], self.cfg.time_scale))

                opacity = []
                for idx in range(len(so_history[:gh.episode_length])):
                    opacity.append('1.0' if so_history[idx][x][y] else "0.0")
                obstacle.appendAnim(self.compressed_anim('opacity', opacity, self.cfg.time_scale))

    def save_animation(self, name='render.svg', egocentric_idx=None):
        animation = self.create_animation(egocentric_idx)
        animation.saveSvg(fname=name)
