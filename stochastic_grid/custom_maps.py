from pathlib import Path

import yaml

with open(Path(__file__).parent / "maps.yaml", "r") as f:
    maps = yaml.safe_load(f)

MAPS_REGISTRY = maps


def main():
    print([key for key in MAPS_REGISTRY if 'da2' in key])


if __name__ == '__main__':
    main()
