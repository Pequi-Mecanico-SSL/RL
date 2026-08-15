"""Gate hierarquico pre-registrado do endpoint H-sync (D260, campanha 3).

IC t pareado bilateral 95% (t79=1.9905) sobre 80 score differences
(win=+1, timeout=0, loss=-1), seeds 0..79, fail-closed: seed ausente,
duplicado ou metadata divergente aborta.

Ordem decisoria (verifier 2026-08-15):
 1. causal: D260-C260 vs ckpt3, LB>0  (se falha, para aqui)
 2. promocao: D260-iter235 vs ckpt3 LB>0 E ponto D260 vs ckpt0 - iter235
    vs ckpt0 >= -0.05 E losses(D260 vs ckpt0)-losses(iter235 vs ckpt0) <= 1
"""
import json
import math
import sys

T79_95 = 1.9905
SEEDS = list(range(80))
SCORE = {"blue_goal": 1, "yellow_goal": -1}


def load(path, expect_yellow=None):
    out = {}
    for ln in open(path):
        r = json.loads(ln)
        if r["mode"] != "deterministic":
            continue
        if r.get("yellow_mode") not in (None, "stochastic"):
            raise SystemExit(f"{path}: yellow_mode invalido {r.get('yellow_mode')}")
        if expect_yellow and expect_yellow not in r["yellow_checkpoint"]:
            raise SystemExit(f"{path}: yellow_checkpoint inesperado {r['yellow_checkpoint']}")
        s = r["seed"]
        if s in out:
            raise SystemExit(f"{path}: seed duplicado {s}")
        out[s] = r["terminal"]
    missing = [s for s in SEEDS if s not in out]
    if missing:
        raise SystemExit(f"{path}: seeds ausentes {missing[:5]}...")
    return out


def paired_ci(a, b):
    ds = [SCORE.get(a[s], 0) - SCORE.get(b[s], 0) for s in SEEDS]
    n = len(ds)
    m = sum(ds) / n
    var = sum((d - m) ** 2 for d in ds) / (n - 1)
    se = math.sqrt(var / n)
    return m, m - T79_95 * se, m + T79_95 * se


def wlt(d):
    w = sum(1 for s in SEEDS if d[s] == "blue_goal")
    l = sum(1 for s in SEEDS if d[s] == "yellow_goal")
    return w, l, 80 - w - l


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "experiment_results"
    d3 = load(f"{base}/h3sync_D260_vs_ckpt3_s0.jsonl", "checkpoint_000003")
    d0 = load(f"{base}/h3sync_D260_vs_ckpt0_s0.jsonl", "checkpoint_000000")
    c3 = load(f"{base}/crossplay_iter260_vs_ckpt3_bdet_ysample.jsonl", "checkpoint_000003")
    i3 = load(f"{base}/crossplay_iter235_vs_ckpt3_bdet_ysample.jsonl", "checkpoint_000003")
    i0 = load(f"{base}/crossplay_iter235_vs_ckpt0_bdet_ysample.jsonl", "checkpoint_000000")

    for nm, d in (("D260 vs ckpt3", d3), ("D260 vs ckpt0", d0), ("C260 vs ckpt3", c3),
                  ("iter235 vs ckpt3", i3), ("iter235 vs ckpt0", i0)):
        w, l, t = wlt(d)
        print(f"{nm}: {w}W/{l}L/{t}T")

    m1, lb1, ub1 = paired_ci(d3, c3)
    print(f"GATE 1 causal D260-C260 vs ckpt3: {m1:+.4f} IC95% [{lb1:+.4f},{ub1:+.4f}] LB>0: {lb1 > 0}")
    if lb1 <= 0:
        m_r, lb_r, ub_r = paired_ci(d3, i3)
        print(f"(info) D260-iter235 vs ckpt3: {m_r:+.4f} [{lb_r:+.4f},{ub_r:+.4f}]")
        verd = "REJEITADO" if m1 <= 0 else "INCONCLUSIVO"
        print(f"VEREDITO: {verd} (gate causal nao passou; hierarquia para aqui)")
        return
    m2, lb2, ub2 = paired_ci(d3, i3)
    m0 = sum(SCORE.get(d0[s], 0) - SCORE.get(i0[s], 0) for s in SEEDS) / 80
    dl = wlt(d0)[1] - wlt(i0)[1]
    print(f"GATE 2a D260-iter235 vs ckpt3: {m2:+.4f} IC95% [{lb2:+.4f},{ub2:+.4f}] LB>0: {lb2 > 0}")
    print(f"GATE 2b ponto vs ckpt0 (D260-iter235): {m0:+.4f} >= -0.05: {m0 >= -0.05}")
    print(f"GATE 2c derrotas adicionais vs ckpt0: {dl} <= 1: {dl <= 1}")
    if lb2 > 0 and m0 >= -0.05 and dl <= 1:
        print("VEREDITO: PROMOVER D260 (todas as condicoes cumulativas passaram)")
    elif m2 <= 0:
        print("VEREDITO: REJEITADO (ponto vs iter235 <= 0)")
    else:
        print("VEREDITO: INCONCLUSIVO (causal passou; promocao incompleta)")


if __name__ == "__main__":
    main()
