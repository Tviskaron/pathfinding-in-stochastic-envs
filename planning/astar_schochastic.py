from planning.astar_no_grid import AStar, INF
import numpy as np


class AStarStochastic(AStar):
    def __init__(self, start: [int, int], goal: [int, int], max_steps: int = INF):
        super().__init__(start, goal, max_steps)
        self.stochastic_obstacles = set()
        self.traversable = set()

    def update_obstacles(self, obs, other_agents, n):
        for i in range(len(obs)):
            for j in range(len(obs)):
                c = (n[0] + i, n[1] + j)
                if obs[i][j] == 0:
                    self.traversable.add(c)
                    if c in self.obstacles:
                        self.obstacles.remove(c)
                else:
                    self.obstacles.add(c)
                    if c in self.traversable:
                        self.traversable.remove(c)
                        self.stochastic_obstacles.add(c)
        if n[0] > self.start[0]:
            range_i = [-1]
            range_j = [_ for _ in range(len(obs))]
        elif n[0] < self.start[0]:
            range_i = [len(obs)]
            range_j = [_ for _ in range(len(obs))]
        elif n[1] > self.start[1]:
            range_i = [_ for _ in range(len(obs))]
            range_j = [-1]
        else:
            range_i = [_ for _ in range(len(obs))]
            range_j = [len(obs)]
        for i in range_i:
            for j in range_j:
                c = (n[0] + i, n[1] + j)
                if c in self.obstacles and c in self.stochastic_obstacles:
                    self.obstacles.remove(c)

        self.other_agents.clear()
        agents = np.nonzero(other_agents)
        for k in range(len(agents[0])):
            self.other_agents.add((n[0] + agents[0][k], n[1] + agents[1][k]))
