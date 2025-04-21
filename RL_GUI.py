import gradio as gr
from RL_train import *
from functools import partial

def sparse_reward_struct(names, *weights):
    return {k:v for k, v in zip(names, weights)}

def dense_reward_struct(functions, args, *weights):
    return [(w, f, a) for w, f, a in zip(weights, functions, args)]

def train(sparse_rewards, dense_rewards):
    print(sparse_rewards)
    print(dense_rewards)

with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("## RL Training GUI")
    with gr.Row():
        checkpoint_dirs = list_dir(
            "/root/ray_results/PPO_selfplay_rec", 
            r"PPO_Soccer_\w+_\d+_\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
        )
        checkpoint_dirs.sort(key=lambda x: datetime.strptime("_".join(x.split('_')[-2:]), "%Y-%m-%d_%H-%M-%S"), reverse=True)
        gr.Dropdown(label="Checkpoint", choices=[None] + checkpoint_dirs)
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### Dense Rewards")
            dense_weights = []
            functions = []
            args = []
            for rw_score, rw_func, rw_args in DENSE_REWARDS:
                dense_weights.append(gr.Slider(minimum=0, maximum=1, label=rw_func.__name__))
                functions.append(rw_func)
                args.append(rw_args)

            dense_rewards = gr.State()
            dense_btn = gr.Button("Save Dense")
            dense_btn.click(
                fn=partial(dense_reward_struct, functions, args),
                inputs=dense_weights,
                outputs=dense_rewards
            )

            gr.Markdown("### Sparse Rewards")
            sparse_weights = []
            for rw_name, rw_score in SPARSE_REWARDS.items():
                sparse_weights.append(gr.Number(label=rw_name, precision=0))

            sparse_rewards = gr.State()
            sparse_btn = gr.Button("Save Sparse")
            sparse_btn.click(
                fn=partial(sparse_reward_struct, SPARSE_REWARDS.keys()), 
                inputs=sparse_weights,
                outputs=sparse_rewards
            )

        with gr.Column(scale=2):
            gr.Video(label="Video", value="videos/rl-video-episode-170.mp4")

    train_button = gr.Button("Train")
    train_button.click(
        fn=train, 
        inputs=[sparse_rewards, dense_rewards],
    )



    gr.Markdown("### Initial Positions")
    init_pos = gr.State()
    with gr.Row():
        with gr.Column(scale=1):
            blue_positions = []
            for i in range(1, 4):
                with gr.Row():
                    gr.Markdown(f"Blue robot {i}")
                    blue_positions.append([
                        gr.Number(label=f"X", value=-1.5 if i == 1 else -2.0),
                        gr.Number(label=f"Y", value=0.0 if i == 1 else (1.0 if i == 2 else -1.0)),
                        gr.Number(label=f"Angle", value=0.0)
                    ])
            with gr.Row():
                gr.Markdown(f"Ball")
                ball_position = [
                    gr.Number(label="X", value=0),
                    gr.Number(label="Y", value=0)
                ]

        with gr.Column(scale=1):
            yellow_positions = []
            for i in range(1, 4):
                with gr.Row():
                    gr.Markdown(f"Yellow robot {i}")
                    yellow_positions.append([
                        gr.Number(label=f"X", value=1.5 if i == 1 else 2.0),
                        gr.Number(label=f"Y", value=0.0 if i == 1 else (1.0 if i == 2 else -1.0)),
                        gr.Number(label=f"Angle", value=180.0)
            ])
            with gr.Row():
                gr.Markdown(f"Extra Config")
                field_type = gr.Number(label="Field Type", value=1, precision=0)
                fps = gr.Number(label="FPS", value=30, precision=0)
                match_time = gr.Number(label="Match Time (s)", value=40, precision=0)
                render_mode = gr.Dropdown(label="Render Mode", choices=["human", "rgb_array"], value="human")

    config_btn = gr.Button("Save Configuration")
    # config_btn.click(
    #     fn=lambda blue, yellow, ball, field, fps, time, render: {
    #         "init_pos": {
    #             "blue": {i + 1: pos for i, pos in enumerate(blue)},
    #             "yellow": {i + 1: pos for i, pos in enumerate(yellow)},
    #             "ball": ball
    #         },
    #         "field_type": field,
    #         "fps": fps,
    #         "match_time": time,
    #         "render_mode": render
    #     },
    #     inputs=[
    #         [pos for sublist in blue_positions for pos in sublist],
    #         [pos for sublist in yellow_positions for pos in sublist],
    #         ball_position,
    #         field_type,
    #         fps,
    #         match_time,
    #         render_mode
    #     ],
    #     outputs=init_pos
    # )



demo.launch(server_name="0.0.0.0")