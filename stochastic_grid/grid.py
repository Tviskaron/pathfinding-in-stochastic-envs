from queue import PriorityQueue

import numpy as np
from pogema.grid import Grid

from stochastic_grid.config import StochasticGridConfig


class StochasticObstacle:
    def __init__(self, x, y, size, config, rnd=None, ):
        if rnd is None:
            rnd = np.random.default_rng()
        self.rnd = rnd

        self.size = size
        self.config: StochasticGridConfig = config
        self.obstacle = self.rnd.binomial(1, self.config.so_density, (self.size, self.size))

        self.x, self.y = x, y
        self.dx, self.dy = None, None
        self.shake_xy()

    def __lt__(self, other):
        return self.x < other.x and self.y < other.x

    def randint(self, low, high=None):
        if low == high:
            return low
        return self.rnd.integers(low=low, high=high)

    def get_xy(self):
        return self.x + self.dx, self.y + self.dy

    def shake_xy(self):
        r = self.config.shake_r
        self.dx = self.randint(-r, r + 1)
        self.dy = self.randint(-r, r + 1)


class StochasticGrid(Grid):
    def __init__(self, grid_config: StochasticGridConfig):
        super().__init__(grid_config)
        self.config: StochasticGridConfig = self.config
        self.stochastic_obstacles = np.zeros(shape=self.obstacles.shape, dtype=int)
        self.targets = np.zeros(shape=self.obstacles.shape, dtype=int)

        for x, y in self.get_targets_xy():
            self.targets[x, y] = 1
        self.hide_queue: PriorityQueue = PriorityQueue()
        self.show_queue: PriorityQueue = PriorityQueue()
        self._t = 0

        def select_random_xy(obstacle_size_, ):
            offset = self.config.shake_r + obstacle_size_
            x_left, x_right = offset, len(self.obstacles) - offset
            y_left, y_right = offset, len(self.obstacles[0]) - offset
            if x_left > x_right or y_left > y_right:
                return None, None
            return self.randint(x_left, x_right), self.randint(y_left, y_right)

        for _ in range(self.config.num_obstacles):
            obstacle_size = self.randint(*self.config.size_range)

            x, y = select_random_xy(obstacle_size)
            obstacle = StochasticObstacle(x, y, obstacle_size, config=self.config, rnd=self.rnd)

            r = obstacle.size
            # lets make 100 attempts to place stochastic obstacle properly
            for _ in range(100):
                x, y = select_random_xy(obstacle_size)
                if x is None or y is None:
                    continue

                overlap = self.obstacles[x:x + r, y:y + r].astype(int)

                overlap[overlap > 0] += 1
                intersection = overlap + obstacle.obstacle
                if np.count_nonzero(intersection == 1):
                    obstacle.x, obstacle.y = x, y
                    break
            if x is None:
                # Can't place stochastic obstacles with that size, so skipping it
                continue
                # raise ValueError("Can't place obstacle! Please check configuration.")
            self.show_queue.put((self.randint(*self.config.show_range), obstacle))

        for _ in range(self.config.init_steps):
            self.update_stochastic_obstacles()

    @staticmethod
    def transfuse_queues(from_q, to_q, t, t_delta_func, process_func):

        while from_q.qsize() > 0:
            timer, obstacle = from_q.get()

            obstacle: StochasticObstacle = obstacle
            if timer <= t:
                process_func(obstacle)
                new_time = t + max(1, t_delta_func())
                to_q.put((new_time, obstacle))
            else:
                from_q.put((timer, obstacle))
                return

    def randint(self, low, high=None):
        if low == high:
            return low
        return self.rnd.integers(low=low, high=high)

    def show_obstacle(self, o: StochasticObstacle):
        o.shake_xy()
        x, y = o.get_xy()
        r = o.size
        self.stochastic_obstacles[x:x + r, y:y + r] += o.obstacle

    def hide_obstacle(self, o: StochasticObstacle):
        x, y = o.get_xy()
        r = o.size
        self.stochastic_obstacles[x:x + r, y:y + r] -= o.obstacle

    def move(self, agent_id, action):
        x, y = self.positions_xy[agent_id]

        self.positions[x, y] = self.config.FREE

        dx, dy = self.config.MOVES[action]
        if self.obstacles[x + dx, y + dy] == self.config.FREE and self.positions[x + dx, y + dy] == self.config.FREE:
            if self.stochastic_obstacles[x + dx, y + dy] == self.config.FREE \
                    or self.targets[x + dx, y + dy] != self.config.FREE:
                x += dx
                y += dy

        self.positions_xy[agent_id] = (x, y)
        self.positions[x, y] = self.config.OBSTACLE

    def update_stochastic_obstacles(self):
        self._t += 1
        # show obstacles and schedule hiding
        self.transfuse_queues(from_q=self.show_queue, to_q=self.hide_queue, t=self._t,
                              t_delta_func=lambda: self.randint(*self.config.show_range),
                              process_func=self.show_obstacle)

        # hide obstacles and schedule showing
        self.transfuse_queues(from_q=self.hide_queue, to_q=self.show_queue, t=self._t,
                              t_delta_func=lambda: self.randint(*self.config.hide_range),
                              process_func=self.hide_obstacle)

    def get_obstacles_for_agent(self, agent_id):

        x, y = self.positions_xy[agent_id]
        r = self.config.obs_radius
        combined = self.obstacles[x - r:x + r + 1, y - r:y + r + 1] + self.stochastic_obstacles[x - r:x + r + 1,
                                                                      y - r:y + r + 1]
        combined[combined >= 1.1] = 1.0
        combined -= self.positions[x - r:x + r + 1, y - r:y + r + 1]
        combined -= self.targets[x - r:x + r + 1, y - r:y + r + 1]
        combined[combined < 0.1] = 0.0
        return combined
