import json
import os
import subprocess
import time
import argparse
from pathlib import Path

from sample_factory.utils.utils import log

from utils.config_validation import Experiment
from utils.files import select_free_dir_name
from utils.gs2dict import generate_variants
import yaml


def start_training_runs(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results = []
    for resolved_vars, spec in generate_variants(config):
        exp = Experiment(**spec)
        if exp.global_settings.experiments_root is None:
            exp.global_settings.experiments_root = select_free_dir_name(exp.global_settings.train_dir)

        cmd = f"python3 training_run.py --wandb_thread_mode=True --raw_config='{json.dumps(exp.dict())}'"

        env_vars = os.environ.copy()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, env=env_vars)
        output, err = process.communicate()

        exit_code = process.wait()

        if exit_code != 0:
            break

        time.sleep(5)

    return results


def main():
    parser = argparse.ArgumentParser(description='Process training configs.')
    parser.add_argument('--config_path', type=str, action="store", default='configs/train.yaml',
                        help='path to yaml file with single run configuration', required=False)
    params = parser.parse_args()
    start_training_runs(params.config_path)


if __name__ == '__main__':
    main()
