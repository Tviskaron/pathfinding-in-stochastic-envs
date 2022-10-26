from pogema import GridConfig
from pydantic import validator, root_validator, Extra

from stochastic_grid.custom_maps import MAPS_REGISTRY


class StochasticGridConfig(GridConfig, extra=Extra.forbid):
    so_density: float = 1.0
    shake_r: int = 0
    num_obstacles: int = 10
    show_range: list = [1, 1]
    hide_range: list = [1, 1]
    size_range: list = [2, 5]
    init_steps: int = 117
    max_episode_steps: int = 512
    map_name: str = None

    use_curriculum: bool = False
    curriculum_score_horizon: int = 256
    curriculum_target: str = 'CSR'
    curriculum_threshold: float = 0.9

    @root_validator
    def create_map(cls, values):
        if values['map'] is None and values['map_name'] in MAPS_REGISTRY:
            values['map'], _, _ = cls.str_map_to_list(MAPS_REGISTRY[values['map_name']], values['FREE'],
                                                      values['OBSTACLE'], )
            values['size'] = max(len(values['map']), len(values['map'][0]))
        return values


def main():
    sgc = StochasticGridConfig(map_name='wc3-heart2heart')
    for line in sgc.map:
        print(line)


if __name__ == '__main__':
    main()
