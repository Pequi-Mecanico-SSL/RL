"""Cirurgia de checkpoint para o braco H-sync (campanha 3).

Copia um checkpoint RLlib 2.10 e substitui SOMENTE os pesos de policy_yellow
pelos pesos de policy_blue (mesma mecanica do sync "Updating Opponent"),
preservando spec/config da yellow (action dist espelhada fica na dist, nao
nos pesos). Valida igualdade tensor a tensor apos a escrita.

Uso (dentro do container rl-policy-training:c684c2b):
  python scripts/make_synced_checkpoint.py <checkpoint_src> <checkpoint_dst>
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from ray import cloudpickle as pickle


def load_state(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if dst.exists():
        raise SystemExit(f"destino ja existe: {dst}")
    shutil.copytree(src, dst)

    blue = load_state(dst / "policies" / "policy_blue" / "policy_state.pkl")
    ypath = dst / "policies" / "policy_yellow" / "policy_state.pkl"
    yellow = load_state(ypath)

    if set(yellow["weights"]) != set(blue["weights"]):
        raise SystemExit("estruturas de pesos divergem entre blue e yellow")

    yellow["weights"] = {k: np.array(v, copy=True) for k, v in blue["weights"].items()}
    with open(ypath, "wb") as f:
        pickle.dump(yellow, f)

    # Revalidacao pos-escrita
    check = load_state(ypath)
    n_equal = n_params = 0
    for k, v in blue["weights"].items():
        a, b = np.asarray(check["weights"][k]), np.asarray(v)
        if not np.array_equal(a, b):
            raise SystemExit(f"divergencia pos-escrita em {k}")
        n_equal += 1
        n_params += a.size
        if not np.isfinite(a).all():
            raise SystemExit(f"NaN/Inf em {k}")
    # Spec da yellow preservada (dist espelhada)
    dist = str(check.get("policy_spec", {})).count("beta_dist_yellow")
    l2 = float(np.sqrt(sum((np.asarray(v) ** 2).sum() for v in check["weights"].values())))
    print(f"OK: {n_equal} tensores, {n_params} params, yellow==blue bit-exato, "
          f"l2={l2:.4f}, spec_yellow_dist_refs={dist}")


if __name__ == "__main__":
    main()
