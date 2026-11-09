"""Erzeugt bue_mc_tabelle.json: Monte-Carlo-Vorteil V je Verteilung, Servicegrad, n und Zufalls-Seed (SplitMix64, 400.000 Ziehungen je Zelle).

Aufruf (aus dem Demo-Ordner): <python> tools/make_mc_table.py  (rund 2 bis 4 Minuten). Die App und tests/test_claims.py lesen die Datei; ein Test rechnet eine Zelle nach."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bue_algorithm as A  # noqa: E402
import bue_constants as C  # noqa: E402
import bue_evaluation as ev  # noqa: E402

QS = (0.95, 0.99)


def main() -> None:
    cells: dict[str, list[float]] = {}
    start = time.perf_counter()
    for d in ev.default_dists():
        for seed in C.SWEEP_SEEDS:
            chain = A.mc_sum_chain(d, C.N_MC_TABLE, C.MC_DRAWS, seed)
            for n, sample in chain.items():
                for q in QS:
                    cells.setdefault(ev.mc_key(d, q, n), []).append(A.advantage(d, n, q, A.mc_quantile(sample, q)))
        print(f"{ev.dist_label(d)} fertig ({time.perf_counter() - start:.0f} s)", flush=True)
    table = {"draws": C.MC_DRAWS, "seeds": list(C.SWEEP_SEEDS), "cells": {k: [round(v, 6) for v in vs] for k, vs in sorted(cells.items())}}
    with open(ev.MC_TABLE_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(table, f, indent=0, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
