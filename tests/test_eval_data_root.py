"""Eval routines must reach the dataset through resolve_data_root.

A run configured with data.hf_repo leaves data.root null, so any routine reading cfg.data.root directly
gets None and dies on os.path.join/loader. Every test here composes exactly that config (hf_repo set,
root null) with snapshot_download stubbed to a local dir, and asserts the routine reads from that dir.
"""

from __future__ import annotations

import ast
import json
import os
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

import quickdraw.evaluation.routines as routines
import quickdraw.training.setup as setup
from quickdraw.controller import run as controller_run

CONF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf")
HF_REPO = "some-org/some-dataset"
COLORING = "circles"          # written into the stub dataset card; distinguishes it from the "rainbow" fallback


class _Stop(Exception):
    """Ends a routine as soon as it has named its data root — nothing past that point is under test."""


class _StubModel:
    """The routines only touch layout/modalities/training before they load data."""

    def __init__(self, heads=("proprio",), action_head_enabled=False):
        self.layout = [(h, None) for h in heads]
        self.modalities = {h: object() for h in heads}   # no .ae -> img_size falls back to 128
        self.training = False
        self.action_head_enabled = action_head_enabled

    def eval(self):
        pass

    def train(self):
        pass


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    """A stand-in for the downloaded HF snapshot, holding the run-dir files the routines read."""
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "dataset_card.json").write_text(json.dumps(
        {"split_env": {"eval_ood_visual": {"R": 2.5, "r": 0.25, "init_speed": 0.75}},
         "coloring": {"train": COLORING, "eval_ood_visual": COLORING}}))
    (root / "summary.json").write_text(json.dumps({"action_sampler": "bimodal"}))
    monkeypatch.setattr(setup, "_hf_root_cache", {})
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo_id, repo_type: str(root))
    return str(root)


@pytest.fixture
def cfg():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(config_name="config", overrides=[f"data.hf_repo={HF_REPO}"])


@pytest.fixture
def ecfg():
    return SimpleNamespace(R=1.0, r=0.3, dt=1 / 60, init_speed=0.5, a_max=1.0, gamma=0.1, mass=1.0)


@pytest.fixture
def writer(tmp_path):
    return SimpleNamespace(dir=str(tmp_path / "run" / "logs"))


def test_resolve_data_root_returns_snapshot_when_hf_repo_set(cfg, snapshot):
    assert cfg.data.root is None                       # the config under test: no local root at all
    assert setup.resolve_data_root(cfg) == snapshot


@pytest.mark.parametrize("routine, heads, action_head", [
    ("ood_horizon", ("proprio",), False),
    ("manifold", ("proprio",), False),
    ("interpret", ("proprio", "fpv"), False),          # vision probe: needs an image head or it self-skips
    ("action_distribution", ("proprio",), True),       # self-skips without an action head
])
def test_episode_loading_routines_use_resolved_root(routine, heads, action_head, cfg, ecfg, writer,
                                                    snapshot, monkeypatch):
    """Routines that load val episodes must pass the resolved snapshot to the loader, not cfg.data.root."""
    seen = []

    def stub_loader(root, split, **kw):
        seen.append(root)
        raise _Stop

    # the routines import the loader inside the function body, so patch it at its source module
    monkeypatch.setattr("quickdraw.data.dataset.load_split_episodes_mm", stub_loader)
    model = _StubModel(heads=heads, action_head_enabled=action_head)
    with pytest.raises(_Stop):
        routines.REGISTRY[routine](cfg, model, None, ecfg, writer, "cpu", step=0)
    assert seen == [snapshot]


def test_ood_axis_reads_dataset_card_from_resolved_root(cfg, ecfg, writer, snapshot, monkeypatch):
    """The OOD axes read their per-split geometry and coloring from the dataset card in the run dir."""
    seen = {}

    def stub_openloop(cfg_, model, norm, writer_, device, split, R, r, v_scale, prefix, step, coloring="rainbow", fps=60):
        seen.update(R=R, r=r, v_scale=v_scale, coloring=coloring)
        return {}

    monkeypatch.setattr(routines, "_openloop_split", stub_openloop)
    routines.REGISTRY["ood_visual"](cfg, _StubModel(), None, ecfg, writer, "cpu", step=0)
    assert seen == {"R": 2.5, "r": 0.25, "v_scale": 0.75, "coloring": COLORING}


def test_control_reads_dataset_card_from_resolved_root(cfg, ecfg, writer, snapshot, monkeypatch, capsys):
    """eval_control renders an FPV context coloured by the dataset card, so it reads the run dir too."""
    def stub_make_env(*a, **kw):
        raise _Stop

    monkeypatch.setattr(controller_run, "make_env", stub_make_env)
    with pytest.raises(_Stop):
        routines.REGISTRY["control"](cfg, _StubModel(heads=("proprio", "fpv")), None, ecfg, writer, "cpu", step=0)
    assert f"coloring={COLORING}" in capsys.readouterr().out


def _cfg_data_root_lines(path):
    """Line numbers of real `cfg.data.root` attribute reads (AST, so comments and strings don't count)."""
    tree = ast.parse(open(path).read())
    return [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr == "root"
            and isinstance(n.value, ast.Attribute) and n.value.attr == "data"
            and isinstance(n.value.value, ast.Name) and n.value.value.id == "cfg"]


def test_no_direct_cfg_data_root_reads_in_eval_path():
    """Guard for the reads that are hard to observe (the dataset-card/summary lookups swallow their errors)."""
    for path in (routines.__file__, controller_run.__file__):
        assert _cfg_data_root_lines(path) == [], f"{path} must read data through resolve_data_root"
