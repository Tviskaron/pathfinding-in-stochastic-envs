from heapq import heappop, heappush

INF = 1000000007


class Node:
    def __init__(self, i, j, g=0, h=0, f=None, parent=None, k=0):
        self.i = i
        self.j = j
        self.g = g
        self.h = h
        self.k = k
        if f is None:
            self.F = g + h
        else:
            self.F = f
        self.parent = parent

    def __str__(self):
        return f'({self.i}, {self.j})'

    def __repr__(self):
        return f'({self.i}, {self.j})'

    def __eq__(self, other):
        return (self.i == other.i) and (self.j == other.j)

    def __lt__(self, other):  # self < other (self has higher priority)
        return self.F < other.F or ((self.F == other.F) and (self.h < other.h)) \
               or ((self.F == other.F) and (self.h == other.h) and (self.k > other.k))


class Open:

    def __init__(self):
        self.priority_queue = []  # this is to maintain prioritized Queue for the OPEN nodes
        self.ij_to_node = {}  # this is an auxiliary dictionary that will help us to quickly identify
        # whether the node is in OPEN

    def __iter__(self):
        return iter(self.ij_to_node.values())  # just in case someone wants to iterate through nodes in OPEN

    def __len__(self):
        return len(self.ij_to_node)

    def is_empty(self):
        return len(self.ij_to_node) == 0

    def add_node(self, item: Node):
        ij = item.i, item.j
        old_node = self.ij_to_node.get(ij)
        if old_node is None or item.g < old_node.g:
            self.ij_to_node[ij] = item  # here we add or UPDATE the lookuptable. So it never contains duplicates.
            heappush(self.priority_queue, item)  # here we ONLY add the node and never check whether the node with the
            # same (i,j) resides in PQ already. This leads to occasional duplicates.

    def get_best_node(self):
        best_node = heappop(self.priority_queue)
        ij = best_node.i, best_node.j

        while self.ij_to_node.pop(ij, None) is None:
            # this line checks whether we have retrieved a duplicate. If yes -
            # move on.
            best_node = heappop(self.priority_queue)
            ij = best_node.i, best_node.j
        return best_node


class Closed:

    def __init__(self):
        self.elements = {}

    def __iter__(self):
        return iter(self.elements.values())

    def __len__(self):
        return len(self.elements)

    def add_node(self, item: Node):
        ij = item.i, item.j
        self.elements[ij] = item

    def was_expanded(self, item: Node):
        ij = item.i, item.j
        return ij in self.elements.keys()


def make_path(goal):
    """
    Creates a path by tracing parent pointers from the goal node to the start node
    It also returns path's length.
    """

    current = goal
    path = []
    while current.parent:
        path.append(current)
        current = current.parent
    path.append(current)
    return path[::-1]


def manhattan_distance(i1, j1, i2, j2):
    return abs(int(i1) - int(i2)) + abs(int(j1) - int(j2))


def update_obs(grid, obs, cur_i, cur_j, step, limit):
    for i in range(len(obs)):
        for j in range(len(obs[0])):
            if obs[i][j] == 1:
                grid[cur_i + i][cur_j + j] = step + 1
            else:
                grid[cur_i + i][cur_j + j] = 0
    if limit is not None:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] <= step - limit:
                    grid[i][j] = 0
