# Pathfinding in Stochastic Environments: Reinforcement Learning vs Planning

Thie repo provides official code implementation for [Pathfinding in Stochastic Environments: Learning vs Planning paper](https://peerj.com/articles/cs-1056/). 
We study and evaluate two orthogonal approaches to tackle the problem of reaching a goal under such conditions: planning and learning. Within planning, an agent constantly replans and updates the path based on the history of the observations using a search-based planner. Within learning, an agent asynchronously learns to optimize a policy function using recurrent neural networks using APPO algorithm.

| A*                                                                                                                                                                                                           | SA*                                                                                                                                                                                                           | APPO                                                                                                                                                                                                      |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [![Pogema logo](https://raw.githubusercontent.com/Tviskaron/pogema-svg/212675dbc8a11edef4b4b5cbf60f46617986d6ff/wc3-tranquilpaths-002-a_star.svg)](https://github.com/Tviskaron/pogema-stochastic-obstacles) | [![Pogema logo](https://raw.githubusercontent.com/Tviskaron/pogema-svg/212675dbc8a11edef4b4b5cbf60f46617986d6ff/wc3-tranquilpaths-002-sa_star.svg)](https://github.com/Tviskaron/pogema-stochastic-obstacles) | [![Pogema logo](https://raw.githubusercontent.com/Tviskaron/pogema-svg/212675dbc8a11edef4b4b5cbf60f46617986d6ff/wc3-tranquilpaths-002-ppo.svg)](https://github.com/Tviskaron/pogema-stochastic-obstacles) |

## Installation
Install all dependencies using:
```bash
pip install -r docker/requirements.txt
```

## Examples

Example of running pre-trained algorihms are provided in ```example.py```:
```bash
python example.py
```
Animations will be stored in ``animations`` folder.

## Training RL Policy
Run ```main.py``` with default training config ```train.yaml```:
```bash
python main.py
```
Full training takes ~16-24 hours with a single GPU.

## Evaluation

Run ```eval.py``` with default evaluation config ```eval.yaml```:
```bash
python eval.py
```
Full evaluation takes around 4-8 hours on a server with single GPU and ~16 CPUs. You can significantly speed up evaluation reducing number of seeds. 

## Metrics and Results
All statistics (for both training and evaluation) are logged using [Weights and Biases](https://wandb.ai/). 
The results of APPO training are available [here](https://wandb.ai/tviskaron/Pathfinding-in-stochastic-environments-APPO-training?workspace=user-tviskaron).
The evaluation plots, tables and raw data are available [here](https://wandb.ai/tviskaron/Pathfinding-in-stochastic-environments-evaluation/runs/2eycut8x/files/results/eval_dir/0001).

## Citing
If you use this work in your research, please use the following bibtex:
```bibtex
@article{skrynnik2022pathfinding,
  title={Pathfinding in stochastic environments: learning vs planning},
  author={Skrynnik, Alexey and Andreychuk, Anton and Yakovlev, Konstantin and Panov, Aleksandr},
  journal={PeerJ Computer Science},
  volume={8},
  pages={e1056},
  year={2022},
  publisher={PeerJ Inc.}
}
```

