from __future__ import annotations

import contextlib
import io
from pathlib import Path
import threading
import time
import traceback
from typing import Any

import gradio as gr
import yaml

from src.gui_utils import DEFAULT_VIDEO_DIR, collect_video_files, list_checkpoint_paths

CONFIG_PATH = Path("config.yaml")
DEFAULT_CONFIG_TEXT = CONFIG_PATH.read_text(encoding="utf-8")
DEFAULT_CHECKPOINTS = list_checkpoint_paths()
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINTS[0] if DEFAULT_CHECKPOINTS else ""
DEFAULT_VIDEO = collect_video_files(DEFAULT_VIDEO_DIR, limit=1)
DEFAULT_VIDEO_PATH = DEFAULT_VIDEO[0] if DEFAULT_VIDEO else None


def parse_config_text(config_text: str) -> dict[str, Any]:
    parsed = yaml.safe_load(config_text)
    if not isinstance(parsed, dict):
        raise ValueError("config.yaml content must decode to a mapping")
    return parsed


class _ThreadSafeBuffer(io.StringIO):
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        with self._lock:
            return super().write(s)

    def snapshot(self) -> str:
        with self._lock:
            return self.getvalue()


def _tail_text(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def train_from_gui(config_text: str, evaluation: bool, stop_timesteps: int | None):
    file_configs = parse_config_text(config_text)
    original_timesteps = file_configs.get("timesteps_total")
    if stop_timesteps is not None and stop_timesteps > 0:
        file_configs["timesteps_total"] = int(stop_timesteps)

    temp_config_path = Path(".gradio_training_config.yaml")
    temp_config_path.write_text(yaml.safe_dump(file_configs, sort_keys=False), encoding="utf-8")

    from src.gui_backend import run_training

    stream_buffer = _ThreadSafeBuffer()
    train_result: dict[str, Any] = {}
    train_error: list[BaseException] = []

    def _runner() -> None:
        try:
            with contextlib.redirect_stdout(stream_buffer), contextlib.redirect_stderr(stream_buffer):
                train_result["result"] = run_training(
                    config_path=temp_config_path,
                    evaluation=evaluation,
                    stop_timesteps=int(file_configs["timesteps_total"]),
                )
        except BaseException as exc:
            train_error.append(exc)
            stream_buffer.write("\n" + traceback.format_exc() + "\n")

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()

    previous_text = ""
    while worker.is_alive():
        current_text = _tail_text(stream_buffer.snapshot())
        if current_text != previous_text:
            previous_text = current_text
            yield "Training running...", gr.update(), current_text
        time.sleep(1)

    if temp_config_path.exists():
        temp_config_path.unlink()

    logs = _tail_text(stream_buffer.snapshot())
    if train_error:
        message = f"Training failed: {train_error[0]}"
        yield message, gr.update(), logs
        return

    result = train_result.get("result", {})

    videos = collect_video_files(DEFAULT_VIDEO_DIR, limit=1)
    latest_video = videos[0] if videos else None
    summary = [
        f"Training finished.",
        f"Best checkpoint: {result.get('best_checkpoint')}",
        f"Latest experiment: {result.get('latest_experiment')}",
    ]
    if original_timesteps is not None:
        summary.append(f"Original timesteps_total: {original_timesteps}")
    summary.append(f"Recorded videos found: {len(videos)}")
    yield "\n".join(summary), gr.update(value=latest_video), logs


def eval_from_gui(config_text: str, checkpoint_blue: str, checkpoint_yellow: str, episodes: int):
    file_configs = parse_config_text(config_text)
    if not checkpoint_blue:
        raise ValueError("Select a checkpoint before running evaluation")

    temp_config_path = Path(".gradio_eval_config.yaml")
    temp_config_path.write_text(yaml.safe_dump(file_configs, sort_keys=False), encoding="utf-8")

    from src.gui_backend import run_eval_episode

    try:
        result = run_eval_episode(
            config_path=temp_config_path,
            checkpoint_path_blue=checkpoint_blue,
            checkpoint_path_yellow=checkpoint_yellow or checkpoint_blue,
            episodes=max(1, int(episodes)),
            render_mode="rgb_array",
        )
    finally:
        if temp_config_path.exists():
            temp_config_path.unlink()

    frames = result.get("frames", [])
    frame_gallery = frames[:200]
    latest_frame = frame_gallery[-1] if frame_gallery else None
    stats_lines = [
        f"Blue checkpoint: {result.get('checkpoint_blue')}",
        f"Yellow checkpoint: {result.get('checkpoint_yellow')}",
    ]
    for episode in result.get("episode_scores", []):
        stats_lines.append(
            f"Episode {episode['episode']}: frames={episode['frames']} score={episode['score']} done={episode['done']} truncated={episode['truncated']}"
        )

    return "\n".join(stats_lines), latest_frame, frame_gallery


with gr.Blocks(title="Pequi RL GUI") as demo:
    gr.Markdown("# Pequi RL GUI")
    gr.Markdown("Use the shared `config.yaml` content below to configure both training and evaluation.")

    config_text = gr.Textbox(
        label="config.yaml",
        value=DEFAULT_CONFIG_TEXT,
        lines=28,
        max_lines=40,
        interactive=True,
    )

    with gr.Tabs():
        with gr.Tab("Train"):
            train_evaluation = gr.Checkbox(label="Enable evaluation videos during training", value=True)
            train_timesteps = gr.Number(label="timesteps_total override", value=0, precision=0)
            train_button = gr.Button("Start training")
            train_status = gr.Textbox(label="Training status", lines=8)
            train_video = gr.Video(label="Latest recorded training video", value=DEFAULT_VIDEO_PATH)
            train_logs = gr.Textbox(label="RLlib terminal output (live)", lines=18, max_lines=28)

        with gr.Tab("Evaluate"):
            checkpoint_blue = gr.Dropdown(
                label="Blue checkpoint",
                choices=DEFAULT_CHECKPOINTS,
                value=DEFAULT_CHECKPOINT,
                allow_custom_value=True,
            )
            checkpoint_yellow = gr.Dropdown(
                label="Yellow checkpoint",
                choices=DEFAULT_CHECKPOINTS,
                value=DEFAULT_CHECKPOINT,
                allow_custom_value=True,
            )
            eval_episodes = gr.Number(label="Episodes to render", value=1, precision=0)
            eval_button = gr.Button("Run evaluation")
            eval_status = gr.Textbox(label="Evaluation status", lines=8)
            eval_frame = gr.Image(label="Latest rgb_array frame", type="numpy")
            eval_frames = gr.Gallery(label="Rendered frames", columns=4, height=320)

    train_button.click(
        fn=train_from_gui,
        inputs=[config_text, train_evaluation, train_timesteps],
        outputs=[train_status, train_video, train_logs],
    )

    eval_button.click(
        fn=eval_from_gui,
        inputs=[config_text, checkpoint_blue, checkpoint_yellow, eval_episodes],
        outputs=[eval_status, eval_frame, eval_frames],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0")
