#!/usr/bin/env python3
import argparse
import os
import shutil
from omegaconf import OmegaConf
import hydra
from omegaconf import DictConfig
import inspect

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(args: DictConfig):
    # Also upload the exact YAML for provenance
    yaml_str = OmegaConf.to_yaml(args)
    print(f"[{inspect.stack()[0][3]}] configuration:\n{yaml_str}")

if __name__ == "__main__":
    main()
