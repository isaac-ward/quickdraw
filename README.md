# quickdraw

A world-models shoot-out: long-horizon consistency as staying on a torus data-manifold, with an MPPI control eval. Needs an NVIDIA GPU + the NVIDIA Container Toolkit.

## Usage

Bring the container up and shell in:

```bash
# copy the keys template and set WANDB_API_KEY
cp .env.template .env
# build the image and start the container (idle)
docker compose up -d --build
# confirm the GPU is visible inside the container
docker compose exec app python -c "import torch; print(torch.cuda.is_available())"
# shell in
docker compose exec app bash
```

Then, inside the container, run the steps in order (`RUN` is your experiment name). Everything for a
run lands in `logs/<step>_<timestamp>_<experiment>/` — dataset, plots, checkpoints, all in one place.

```bash
# 1. generate the dataset + sample plots/summary (all under one logs/ run dir; it prints the path)
uv run python -m quickdraw.data_generation experiment=$RUN
#    -> set DATA=logs/data_generation_<ts>_$RUN   and   CKPT=logs/train_<ts>_$RUN  (after step 2)
# 2. train: train/val + the subscribed in-loop evals every N epochs
uv run python -m quickdraw.train             experiment=$RUN data.root=$DATA
# 3. post-hoc evals at the best checkpoint
uv run python -m quickdraw.eval_long_horizon experiment=$RUN data.root=$DATA checkpoint=$CKPT
uv run python -m quickdraw.eval_ood          experiment=$RUN data.root=$DATA checkpoint=$CKPT
uv run python -m quickdraw.eval_control      experiment=$RUN data.root=$DATA checkpoint=$CKPT
```

Override any Hydra field on the CLI.

## Links

- [Hugging Face Dataset](https://huggingface.co/datasets/isaac-ronald-ward/quickdraw-torus)
- [GitHub](https://github.com/isaac-ward/quickdraw)
