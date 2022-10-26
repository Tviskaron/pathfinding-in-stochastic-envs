from typing import Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pydantic import BaseModel, Extra

from utils.config_validation import Environment


class AlgoBase(BaseModel):
    name: str = None
    num_process: int = 1
    device: str = 'cuda'


class AlgoDecentralized(AlgoBase, extra=Extra.forbid):
    name: Literal['A*', 'SA*'] = 'A*'
    num_process: int = 5
    fix_loops: bool = False
    no_path_random: bool = False
    fix_nones: bool = True
    add_none_if_loop: bool = False
    use_best_move: bool = True
    stay_if_loop_prob: float = None
    max_planning_steps: int = 10000
    device: str = 'cpu'


class AlgoAPPO(AlgoBase, extra=Extra.forbid):
    name: Literal['APPO'] = 'APPO'
    num_process: int = 3

    path_to_weights: str = "weights/multimap-512-v0"


class TabulateConfig(BaseModel):
    drop_keys: list = ['seed', 'flowtime']
    metrics: list = ['ISR', 'CSR', 'makespan', 'FPS']
    round_digits: int = 2


class View(BaseModel):
    # verbose: bool = False
    drop_keys: list = ['seed', 'flowtime']
    round_digits: int = 2
    rename_fields: dict = {"num_obstacles": "Stochastic obstacles",
                           "ISR": "Success rate",
                           "algo": "Agent",
                           "makespan": 'Episode length',
                           }
    sort_by: Union[str, List[str]] = None


class TabularView(View):
    print: bool = False
    type: Literal['tabular'] = 'tabular'
    table_format: str = 'psql'


class PlotView(View):
    type: Literal['plot'] = 'plot'
    x: str = None
    y: str = None
    by: str = 'algo'

    confidence_intervals: Union[int, Literal['sd']] = None

    plt_style: str = 'bmh'
    figure_dpi: int = 300
    font_size: int = 12
    legend_font_size: int = 14
    figure_face_color: str = '#FFFFFF'
    markers: bool = True
    line_types: bool = True


class MultiPlotView(PlotView):
    type: Literal['plot'] = 'multi-plot'
    over: str = None
    num_cols: int = 4


class EvaluationSettings(BaseModel, extra=Extra.forbid):
    name: str = None
    use_wandb: str = False
    environment: Union[Environment, dict] = Environment()
    algo: Union[AlgoDecentralized, AlgoAPPO] = None
    resolved_vars: dict = None
    results: dict = None
    id: int = None
    tabulate_config: TabulateConfig = TabulateConfig()
    eval_dir: str = 'results/eval_dir'
    results_views: List[Union[TabularView, PlotView]] = [TabularView]
