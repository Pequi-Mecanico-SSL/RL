from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_RAY_RESULTS_DIR = os.environ.get("RL_RAY_RESULTS_DIR", "/root/ray_results/PPO_selfplay_rec")
DEFAULT_VIDEO_DIR = os.environ.get("RL_VIDEO_DIR", os.path.abspath("volumes/videos"))


def load_yaml_config(config_path: str | os.PathLike[str] = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def collect_video_files(video_dir: str | os.PathLike[str], limit: int = 5) -> list[str]:
    video_root = Path(video_dir)
    try:
        if not video_root.exists():
            return []
        videos = sorted(video_root.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    except PermissionError:
        return []
    return [str(path) for path in videos[:limit]]


def list_checkpoint_paths(parent_dir: str | os.PathLike[str] = DEFAULT_RAY_RESULTS_DIR, limit: int = 50) -> list[str]:
    parent_path = Path(parent_dir)
    try:
        if not parent_path.exists():
            return []
        checkpoints = [path for path in parent_path.rglob("checkpoint_*") if path.is_dir()]
        checkpoints.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    except PermissionError:
        return []
    return [str(path) for path in checkpoints[:limit]]
