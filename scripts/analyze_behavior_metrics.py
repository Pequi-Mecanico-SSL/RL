#!/usr/bin/env python3
"""Metricas comportamentais + decomposicao de reward a partir dos .npz coletados.

Uso: python scripts/analyze_behavior_metrics.py exp_results/behavior/*.npz
Gera por arquivo: <stem>_metrics.json e <stem>_heatmaps.png; imprime tabela
comparativa no stdout.
"""

import json
import sys
from pathlib import Path

import numpy as np

STOP_SPEED = 0.05          # m/s: abaixo disso o robo conta como parado
EVENT_GAP = 5              # frames: flags mais proximos que isso viram um so evento
TOUCH_DIST = 0.13          # m: raio de contato robo-bola
TOUCH_DVEL = 0.30          # m/s: variacao de velocidade da bola que conta toque
KICK_ACT_THRESHOLD = 0.0   # env legado: kick_v_x=3.0 se action[3] > 0 (binario)
GRID = (45, 30)            # celulas do heatmap (0,2 m por celula em Division B)


def load_npz(path):
    data = np.load(path)
    meta = json.loads(bytes(data["meta_json"]).decode())
    episodes = []
    for i in range(meta["episodes"]):
        episodes.append({k: data[f"ep{i:03d}_{k}"]
                         for k in ("ball", "robots", "actions", "rewards", "components")})
    return meta, episodes


def event_groups(flags):
    """Agrupa steps flagados em eventos; retorna lista de (inicio, fim)."""
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > EVENT_GAP)
    starts = np.concatenate([[0], splits + 1])
    ends = np.concatenate([splits, [len(idx) - 1]])
    return [(idx[s], idx[e]) for s, e in zip(starts, ends)]


def heatmap(xy, half_len, half_wid):
    grid, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=GRID,
        range=[[-half_len, half_len], [-half_wid, half_wid]])
    return grid / max(grid.sum(), 1)


def episode_metrics(ep, meta):
    fps = meta["fps"]
    length, width, goal_width = meta["field_length_width_goalwidth"]
    half_len, half_wid, half_goal = length / 2, width / 2, goal_width / 2
    ball = ep["ball"]
    robots = ep["robots"].reshape(len(ball), 6, 5)  # blue 0-2, yellow 3-5
    blue, yellow = robots[:, :3], robots[:, 3:]
    steps = len(ball)

    out = {"steps": steps, "sim_seconds": steps / fps}

    # movimento (blue)
    speed = np.hypot(blue[..., 3], blue[..., 4])                 # (T,3)
    out["blue_dist_per_robot_m"] = (speed.mean(0) * steps / fps).round(2).tolist()
    out["blue_speed_mean"] = float(speed.mean())
    out["blue_speed_p95"] = float(np.percentile(speed, 95))
    out["blue_pct_stopped"] = float((speed < STOP_SPEED).mean())
    yspeed = np.hypot(yellow[..., 3], yellow[..., 4])
    out["yellow_pct_stopped"] = float((yspeed < STOP_SPEED).mean())
    pair = [np.hypot(*(blue[:, i, :2] - blue[:, j, :2]).T)
            for i, j in ((0, 1), (0, 2), (1, 2))]
    out["blue_spread_mean_m"] = float(np.mean(pair))

    # bola por tercos (blue ataca +x)
    third = length / 6
    out["ball_pct_def"] = float((ball[:, 0] < -third).mean())
    out["ball_pct_mid"] = float((np.abs(ball[:, 0]) <= third).mean())
    out["ball_pct_att"] = float((ball[:, 0] > third).mean())

    # posse aproximada: time do robo mais proximo da bola
    dist_all = np.linalg.norm(robots[..., :2] - ball[:, None, :2], axis=2)  # (T,6)
    nearest = dist_all.argmin(1)
    out["possession_blue_pct"] = float((nearest < 3).mean())

    # toques: variacao de velocidade da bola com robo em contato
    dvel = np.zeros(steps)
    dvel[1:] = np.hypot(*np.diff(ball[:, 2:4], axis=0).T)
    contact = dist_all.min(1) < TOUCH_DIST
    touch = (dvel > TOUCH_DVEL) & contact
    out["touches_blue"] = len(event_groups(touch & (nearest < 3)))
    out["touches_yellow"] = len(event_groups(touch & (nearest >= 3)))

    # chutes (blue): eventos de acao kick>0 com bola em contato; no alvo se a
    # projecao linear da velocidade (no fim do evento) cruza o gol adversario
    blue_dist = dist_all[:, :3].min(1)
    kick_act = ep["actions"].reshape(steps, 6, 4)[:, :3, 3].max(1)
    groups = event_groups((kick_act > KICK_ACT_THRESHOLD) & (blue_dist < TOUCH_DIST))
    shots_on_target = 0
    for _, t_end in groups:
        vx, vy = ball[t_end, 2:4]
        if vx > 0.1:
            y_at_goal = ball[t_end, 1] + vy * (half_len - ball[t_end, 0]) / vx
            if abs(y_at_goal) < half_goal:
                shots_on_target += 1
    out["blue_kicks"] = len(groups)
    out["blue_shots_on_target"] = shots_on_target

    # progresso da bola e bola fora
    out["ball_speed_to_goal_mean"] = float(ball[:, 2].mean())  # v_x medio (gol blue = +x)
    last = ball[-1]
    out["ball_out"] = bool(abs(last[0]) >= half_len or abs(last[1]) >= half_wid) \
        and not (abs(last[0]) >= half_len and abs(last[1]) < half_goal)

    # decomposicao de reward (media dos 3 agentes blue, por componente)
    comp = ep["components"][:, :, :3].mean(2)  # (T, n_comp)
    weights = np.asarray(meta["component_weights"])
    out["reward_components_mean"] = dict(zip(meta["component_names"],
                                             comp.mean(0).round(4).tolist()))
    out["reward_weighted_share"] = dict(zip(
        meta["component_names"],
        (np.abs(weights * comp.mean(0)) /
         max(np.abs(weights * comp.mean(0)).sum(), 1e-9)).round(3).tolist()))
    return out


def summarize(path):
    meta, episodes = load_npz(path)
    length, width, _ = meta["field_length_width_goalwidth"]
    per_ep = [episode_metrics(ep, meta) for ep in episodes]
    terminals = meta["terminals"]

    summary = {
        "matchup": f"{Path(meta['checkpoint']).parent.name}/{Path(meta['checkpoint']).name} "
                   f"vs {Path(meta['yellow_checkpoint']).parent.name}/{Path(meta['yellow_checkpoint']).name}",
        "modes": f"blue={meta['blue_mode']} yellow={meta['yellow_mode']}",
        "episodes": meta["episodes"],
        "seeds": f"{meta['seed_start']}..{meta['seed_start'] + meta['episodes'] - 1}",
        "W/L/T": f"{terminals.count('blue_goal')}/{terminals.count('yellow_goal')}/"
                 f"{terminals.count('timeout')}",
        "ball_out_episodes": int(sum(m["ball_out"] for m in per_ep)),
        "time_to_goal_s": sorted(round(m["sim_seconds"], 1) for m, t in zip(per_ep, terminals)
                                 if t == "blue_goal"),
    }
    for key in ("blue_speed_mean", "blue_speed_p95", "blue_pct_stopped",
                "yellow_pct_stopped", "blue_spread_mean_m", "ball_pct_def",
                "ball_pct_mid", "ball_pct_att", "possession_blue_pct",
                "ball_speed_to_goal_mean"):
        summary[key] = round(float(np.mean([m[key] for m in per_ep])), 4)
    for key in ("touches_blue", "touches_yellow", "blue_kicks", "blue_shots_on_target"):
        summary[key + "_per_ep"] = round(float(np.mean([m[key] for m in per_ep])), 2)
    for comp in meta["component_names"]:
        summary[f"rw_{comp}_mean"] = round(float(np.mean(
            [m["reward_components_mean"][comp] for m in per_ep])), 4)
        summary[f"rw_{comp}_share"] = round(float(np.mean(
            [m["reward_weighted_share"][comp] for m in per_ep])), 3)

    # correlacao ponto-biserial: media do componente no episodio vs gol
    goal = np.array([t == "blue_goal" for t in terminals], dtype=float)
    if 0 < goal.sum() < len(goal):
        for i, comp in enumerate(meta["component_names"]):
            values = np.array([m["reward_components_mean"][comp] for m in per_ep])
            summary[f"corr_goal_{comp}"] = round(float(np.corrcoef(values, goal)[0, 1]), 3)

    # heatmaps agregados
    half_len, half_wid = length / 2, width / 2
    ball_all = np.concatenate([ep["ball"][:, :2] for ep in episodes])
    maps = {"ball": heatmap(ball_all, half_len, half_wid)}
    robots_all = np.concatenate(
        [ep["robots"].reshape(-1, 6, 5) for ep in episodes])
    for i in range(3):
        maps[f"blue_{i}"] = heatmap(robots_all[:, i, :2], half_len, half_wid)
    maps["yellow_all"] = heatmap(robots_all[:, 3:, :2].reshape(-1, 2), half_len, half_wid)
    return meta, summary, per_ep, maps


def render(maps, summary, out_png, field):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    length, width, goal_width = field
    names = ["ball", "blue_0", "blue_1", "blue_2", "yellow_all"]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6))
    for ax, name in zip(axes, names):
        ax.imshow(maps[name].T, origin="lower", cmap="hot",
                  extent=[-length / 2, length / 2, -width / 2, width / 2],
                  aspect="equal")
        ax.plot([length / 2] * 2, [-goal_width / 2, goal_width / 2], "c-", lw=3)
        ax.plot([-length / 2] * 2, [-goal_width / 2, goal_width / 2], "w-", lw=3)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_title(name, fontsize=10)
    fig.suptitle(f"{summary['matchup']} — {summary['W/L/T']} (W/L/T), "
                 f"posse={summary['possession_blue_pct']:.0%}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("uso: analyze_behavior_metrics.py <npz...>")
    summaries = []
    for path in paths:
        meta, summary, per_ep, maps = summarize(path)
        stem = Path(path).with_suffix("")
        with open(f"{stem}_metrics.json", "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "per_episode": per_ep}, fh, indent=1)
        render(maps, summary, f"{stem}_heatmaps.png",
               meta["field_length_width_goalwidth"])
        summaries.append(summary)
        print(f"ok: {stem}_metrics.json / _heatmaps.png")

    keys = [k for k in summaries[0] if k not in ("matchup", "modes", "seeds", "time_to_goal_s")]
    header = ["metric"] + [s["matchup"].split(" vs ")[0] for s in summaries]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for key in keys:
        row = [str(s.get(key, "-")) for s in summaries]
        print(f"| {key} | " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
