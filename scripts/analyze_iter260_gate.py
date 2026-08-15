import json, math

def load(path):
    out = {}
    for ln in open(path):
        r = json.loads(ln)
        if r['mode'] == 'deterministic':
            out[r['seed']] = r['terminal']
    return out

score = lambda t: {'blue_goal': 1, 'yellow_goal': -1}.get(t, 0)
Z = 2.2414  # IC97,5% bilateral (Bonferroni, 2 olhares)

def paired(a, b):
    seeds = sorted(set(a) & set(b))
    diffs = [score(a[s]) - score(b[s]) for s in seeds]
    n = len(diffs)
    m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    return m, m - Z * se, m + Z * se, n

c260_3 = load('experiment_results/crossplay_iter260_vs_ckpt3_bdet_ysample.jsonl')
c260_0 = load('experiment_results/crossplay_iter260_vs_ckpt0_bdet_ysample.jsonl')
c235_3 = load('experiment_results/crossplay_iter235_vs_ckpt3_bdet_ysample.jsonl')
c235_0 = load('experiment_results/crossplay_iter235_vs_ckpt0_bdet_ysample.jsonl')

m, lo, hi, n = paired(c260_3, c235_3)
print(f"pareado vs ckpt3 (260-235): {m:+.3f} IC97.5% [{lo:+.3f},{hi:+.3f}] n={n} -> LB>0: {lo > 0}")
m0, lo0, hi0, n0 = paired(c260_0, c235_0)
print(f"pareado vs ckpt0 (260-235): {m0:+.3f} IC97.5% [{lo0:+.3f},{hi0:+.3f}] n={n0} -> ponto>=0: {m0 >= 0} | UB<0: {hi0 < 0}")
imp = sum(1 for s in c260_3 if score(c260_3[s]) > score(c235_3[s]))
reg = sum(1 for s in c260_3 if score(c260_3[s]) < score(c235_3[s]))
print(f"vs ckpt3 por seed: {imp} melhorias, {reg} regressoes, {80 - imp - reg} inalterados")
imp0 = sum(1 for s in c260_0 if score(c260_0[s]) > score(c235_0[s]))
reg0 = sum(1 for s in c260_0 if score(c260_0[s]) < score(c235_0[s]))
print(f"vs ckpt0 por seed: {imp0} melhorias, {reg0} regressoes, {80 - imp0 - reg0} inalterados")
