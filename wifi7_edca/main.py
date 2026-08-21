#!/usr/bin/env python3
"""
main.py -- Chay toan bo pipeline tai hien paper:

    gen data  ->  ham utility chung  ->  training (GA)  ->  ve ket qua

Cach dung:
    python main.py                # chay tat ca cac hinh (Fig. 2-6)
    python main.py 3 5            # chi chay mot so hinh
    python main.py --quick        # GA nho hon, chay nhanh de kiem tra pipeline

Che do --quick ghi ra `results/figN_quick.json` va `figures/*_quick.png` de KHONG
de len ket qua cua lan chay day du (mot lan chay day du ton ~1 gio).
"""
from __future__ import annotations

import sys
import time

from plotting import figures as fg
from plotting import style as st
from training import experiments as ex

ALL = [2, 3, 4, 5, 6]


def run(fig_ids, quick: bool = False):
    paths = []
    t_start = time.time()
    fig45 = None
    tag = "_quick" if quick else ""
    st.NAME_SUFFIX = tag                 # hau to cho ten file hinh

    if 2 in fig_ids:
        print("[Fig. 2] Mo hinh vung AIFS ...")
        d = ex.exp_fig2(); ex.save("fig2" + tag, d); paths.append(fg.plot_fig2(d))

    if 3 in fig_ids:
        print("[Fig. 3] Do nhay tham so AIFS / TXOP ...")
        d = ex.exp_fig3(); ex.save("fig3" + tag, d); paths.append(fg.plot_fig3(d))

    if 4 in fig_ids or 5 in fig_ids:
        print("[Fig. 4+5] Toi uu bang GA (EDCA don link va MLO EDCA) ...")
        fig45 = ex.exp_fig4_fig5(n_pop=40 if quick else 200,
                                 n_gen=25 if quick else 300)
        ex.save("fig45" + tag, fig45)
        if 4 in fig_ids:
            paths.append(fg.plot_fig4(fig45))
        if 5 in fig_ids:
            paths.append(fg.plot_fig5(fig45))
        for name, cfg in fig45["configs"].items():
            print(f"          {name:15s} muc tieu = {cfg['objective']:7.3f} | "
                  f"kha thi = {cfg['feasible']}")

    if 6 in fig_ids:
        print("[Fig. 6] Anh huong cua nguong eps_1 ...")
        d = ex.exp_fig6(eps1_values=(1e-7, 1e-5) if quick else
                        (1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
                        n_pop=30 if quick else 120,
                        n_gen=15 if quick else 120)
        ex.save("fig6" + tag, d); paths.append(fg.plot_fig6(d))

    print(f"\nXong sau {time.time() - t_start:.1f}s. Cac hinh da luu:")
    for p in paths:
        print("  -", p)
    return paths


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    run([int(a) for a in args] if args else ALL, quick="--quick" in sys.argv)
