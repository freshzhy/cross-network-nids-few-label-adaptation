#!/usr/bin/env python3
"""
rq4_variant_holdout.py — 变体留出压力测试（variant-held-out stress test）
====================================================================================
问题：实验 A 显示"目标网络 5–10 条攻击标注 + 本地重训练"即可恢复大部分性能，且不受精确去重影响。
      剩下的疑点是：基准中的攻击流由少数工具生成、高度同质，少量标注是否只是"见过同一工具"？
      本测试回答：当目标域的少量标注【没有覆盖】测试时出现的攻击变体时，少量标注的收益是否仍然存在？

设计（运行前锁定）：
  A. 主测试 —— CIC-IDS2018 DoS 留一变体（4 折）：
       标注只来自 3 个 DoS 变体（GoldenEye / Hulk / SlowHTTPTest / Slowloris 中的 3 个），
       第 4 个变体完全不出现在训练中（既不作正例，也不作负例），只出现在 unseen 测试集。
  B. 次级测试 —— 同一网络内攻击族互换（CIC、ToN、BoT 三个网络 × DoS→DDoS / DDoS→DoS，共 6 个方向）：
       标注只来自一个攻击族，测试另一个攻击族（未见过的相关攻击类型，难度高于 A）。
  * 训练/测试划分：正例 = 已见变体（上限 30,000，与主实验相同），负例 = 不属于 DoS/DDoS 族的全部其他流
    （上限 10×正例），80/20 分层划分；seed 42–51。
  * 预算 = 已见变体的训练正例数 1, 2, 5, 10, 25, 50, 100, All（负例 1:10，同一 seed 内嵌套）。
    分类器 = Full-40 XGBoost（与主实验超参数相同；本测试只关心重训练本身，不涉及特征选择）。
  * 两个测试集（负例相同 = 测试划分中的负例）：
       seen   ：测试划分中的已见变体正例（参照：同分布）
       unseen ：留出变体/另一攻击族的正例，按 1:10 抽取（这些行从未参与训练）
  * 零样本基线（同一 unseen 测试集）：其他网络中【测试攻击族】的源模型，源端用与主实验相同的 seed 划分训练，
       Full-40-Direct（全部特征）与 SHAP-10-Direct（主实验冻结的源 SHAP Top-10，仅 DoS 有）各自对源网络取平均；
       比较时取两者中【较高者】作为零样本基线（对少量标注更不利、更保守）。
  * 指标：PR-AUC（主指标，阈值无关）、F1（0.5 阈值）、pAUC@1%FPR、TPR@1%FPR（1% 误报下的检出率）。
  * 预先设定的判读标准（主指标 PR-AUC，预算 10）：
       持续有效（persists）          ：unseen ≥ 0.8 × seen，且 unseen > 零样本基线（≥ 9/10 个 seed）
       下降但仍有用（degraded-useful）：unseen − 零样本 ≥ 0.05（seed 均值），但 unseen < 0.8 × seen
       失效（fails）                  ：其余情况
     同时报告预算 5 与 25 的同样判读，作为敏感性。折数少（A 为 4 折），因此只做描述性判读，不做配置级显著性检验。

运行（项目根目录）：
  ./run.sh rq4            # 与标注预算实验一起运行；后台：./run.sh bg rq4
  ./run.sh status
输出：results/rq4_variant/
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_T = os.environ.get("A7_THREADS")
if _T:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = _T

import numpy as np
import pandas as pd

import rq3_transfer as base

ROOT = Path(__file__).resolve().parent.parent   # repository root (scripts live in code/)
OUT = ROOT / "results" / "rq4_variant"
PARTS, LOGS, PROGRESS = OUT / "parts", OUT / "logs", OUT / "progress.log"
BUDGETS = [1, 2, 5, 10, 25, 50, 100, "All"]
NEG_RATIO = 10
FAMILY = {
    "DoS": {"UNSW": ["DoS"], "ToN": ["dos"], "BoT": ["DoS"],
            "CIC": ["DoS attacks-GoldenEye", "DoS attacks-Hulk", "DoS attacks-SlowHTTPTest", "DoS attacks-Slowloris"]},
    "DDoS": {"ToN": ["ddos"], "BoT": ["DDoS"], "CIC": ["DDoS attacks-LOIC-HTTP", "DDOS attack-HOIC"]},
}
CIC_DOS = FAMILY["DoS"]["CIC"]


def tests():
    """所有测试：(test_id, kind, target, seen_labels, unseen_labels, unseen_family)"""
    out = []
    for v in CIC_DOS:
        seen = [x for x in CIC_DOS if x != v]
        out.append((f"LOVO:{v.replace('DoS attacks-', '')}", "A_variant", "CIC", seen, [v], "DoS"))
    for tgt in ("CIC", "ToN", "BoT"):
        for f1, f2 in (("DoS", "DDoS"), ("DDoS", "DoS")):
            out.append((f"{tgt}:{f1}→{f2}", "B_family", tgt, FAMILY[f1][tgt], FAMILY[f2][tgt], f2))
    return out


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def plog(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def metrics(y, p):
    from sklearn.metrics import roc_curve
    m = base.metrics(y, p)
    try:
        fpr, tpr, _ = roc_curve(y, p)
        m["tpr_at_1fpr"] = round(float(tpr[fpr <= 0.01].max()) if np.any(fpr <= 0.01) else 0.0, 6)
    except ValueError:
        m["tpr_at_1fpr"] = np.nan
    return m


# ══════════════════════════════════════════════════════════════════════════
# worker：一个 seed 跑完全部测试
# ══════════════════════════════════════════════════════════════════════════
def build_target(tgt, seen, unseen, seed, idx_of):
    X, codes, meta = base.load_cache(tgt)
    labels = meta["labels"]
    fam_all = set(FAMILY["DoS"].get(tgt, [])) | set(FAMILY["DDoS"].get(tgt, []))
    c_seen = [labels.index(l) for l in seen if l in labels]
    c_unseen = [labels.index(l) for l in unseen if l in labels]
    c_neg = [i for i, l in enumerate(labels) if l not in fam_all]
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(np.isin(codes, c_seen))
    neg = np.flatnonzero(np.isin(codes, c_neg))
    if len(pos) > base.MAX_TRAIN_POS:
        pos = rng.choice(pos, base.MAX_TRAIN_POS, replace=False)
    if len(neg) > NEG_RATIO * len(pos):
        neg = rng.choice(neg, NEG_RATIO * len(pos), replace=False)
    from sklearn.model_selection import train_test_split
    idx = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos), np.int32), np.zeros(len(neg), np.int32)])
    tr, te, ytr, yte = train_test_split(idx, y, test_size=base.TEST_SIZE, random_state=seed, stratify=y)
    n_neg_te = int((yte == 0).sum())
    un_pool = np.flatnonzero(np.isin(codes, c_unseen))
    n_un = min(len(un_pool), max(1, n_neg_te // NEG_RATIO))
    un = np.random.default_rng(seed + 80000).choice(un_pool, n_un, replace=False)
    Xall = base.clean_subset(X[np.concatenate([tr, te, un])])
    Xtr, Xte, Xun = Xall[:len(tr)], Xall[len(tr):len(tr) + len(te)], Xall[len(tr) + len(te):]
    Xu = np.vstack([Xte[yte == 0], Xun])
    yu = np.concatenate([np.zeros(n_neg_te, np.int32), np.ones(len(un), np.int32)])
    r2 = np.random.default_rng(seed + 70000 + idx_of)
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xu=Xu, yu=yu,
                pos_all=r2.permutation(np.flatnonzero(ytr == 1)),
                neg_all=r2.permutation(np.flatnonzero(ytr == 0)), fc=meta["feature_cols"])


def cmd_worker(a):
    seed, threads = a.seed, a.threads
    tag = f"seed{seed}"
    hb = base.Heartbeat(LOGS / f"{tag}.log")
    hb(f"开始 {tag} threads={threads}")
    t0 = time.time()
    fsets = json.loads((ROOT / "results" / "rq3_main" / "feature_sets.json").read_text())["DoS"]
    rows, src_models = [], {}

    def source_models(fam, src):
        """其他网络中该攻击族的零样本源模型（与主实验相同的 seed 划分）。"""
        key = (fam, src)
        if key in src_models:
            return src_models[key]
        X, codes, meta = base.load_cache(src)
        pc = [meta["labels"].index(l) for l in FAMILY[fam][src] if l in meta["labels"]]
        tr, te, ytr, yte = base.split_indices(codes, pc, seed)
        Xtr = base.clean_subset(X[np.concatenate([tr, te])])[:len(tr)]
        ms = {"Full-40-Direct": (base.fit_xgb(Xtr, ytr, seed, threads), list(range(Xtr.shape[1])))}
        if fam == "DoS":
            fc = meta["feature_cols"]
            cols = [fc.index(f) for f in fsets[src]["SHAP-10"]]
            ms["SHAP-10-Direct"] = (base.fit_xgb(Xtr[:, cols], ytr, seed, threads), cols)
        src_models[key] = ms
        hb(f"源模型 {fam}/{src} 训练完成（{time.time()-t0:.0f}s）")
        return ms

    for ti, (tid, kind, tgt, seen, unseen, ufam) in enumerate(tests()):
        T = build_target(tgt, seen, unseen, seed, ti)
        base_row = dict(test_id=tid, kind=kind, target=tgt, seed=seed, seen=";".join(seen), unseen=";".join(unseen))
        # 零样本
        for src in [s for s in FAMILY[ufam] if s != tgt]:
            for cond, (m, cols) in source_models(ufam, src).items():
                for ts, X_, y_ in (("unseen", T["Xu"], T["yu"]), ("seen", T["Xte"], T["yte"])):
                    rows.append(dict(base_row, condition=cond, source=src, budget="—", testset=ts,
                                     **metrics(y_, m.predict_proba(X_[:, cols])[:, 1])))
        # 少量标注重训练（Full-40）
        for b in BUDGETS:
            if b == "All":
                idx = np.arange(len(T["ytr"]))
            else:
                if b > len(T["pos_all"]):
                    continue
                idx = np.sort(np.concatenate([T["pos_all"][:b], T["neg_all"][:min(len(T["neg_all"]), NEG_RATIO * b)]]))
            m = base.fit_xgb(T["Xtr"][idx], T["ytr"][idx], seed, threads)
            for ts, X_, y_ in (("unseen", T["Xu"], T["yu"]), ("seen", T["Xte"], T["yte"])):
                rows.append(dict(base_row, condition="Full-40-retrained", source="—", budget=str(b), testset=ts,
                                 n_pos_train=int((T["ytr"][idx] == 1).sum()), **metrics(y_, m.predict_proba(X_)[:, 1])))
        hb(f"{tid} 完成（unseen 正例 {int(T['yu'].sum()):,}；累计 {time.time()-t0:.0f}s）")
    out = pd.DataFrame(rows)
    tmp = PARTS / f"{tag}.csv.tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, PARTS / f"{tag}.csv")
    hb(f"完成 {tag}，{len(out)} 行，用时 {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════════
# run / status
# ══════════════════════════════════════════════════════════════════════════
def cmd_run(a):
    for p in (ROOT / "results" / "rq3_main" / "feature_sets.json",):
        if not p.exists():
            sys.exit(f"✗ 缺少 {p.relative_to(ROOT)}")
    miss = [d for d in base.DS_ORDER if not (base.CACHE / f"{d}_X.npy").exists()]
    if miss:
        sys.exit(f"✗ 主实验缓存缺失 {miss}")
    for d in (PARTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    seeds = list(range(base.SEED_BASE, base.SEED_BASE + a.n_seeds))
    todo = [s for s in seeds if not (PARTS / f"seed{s}.csv").exists()]
    (OUT / "run_config.json").write_text(json.dumps(vars(a), indent=1, default=str))
    plog(f"run 开始：总任务 {len(seeds)}，待完成 {len(todo)}，并行 {a.workers} × 线程 {a.threads}")
    env = dict(os.environ, A7_THREADS=str(a.threads), PYTHONUNBUFFERED="1")
    running, attempts, failed, durations, queue, last = {}, {}, [], [], list(todo), 0

    def launch(s):
        errf = open(LOGS / f"seed{s}.stderr", "a")
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--seed", str(s),
                              "--threads", str(a.threads)], env=env, stdout=errf, stderr=errf, start_new_session=True)
        running[s] = (p, time.time(), errf)
        attempts[s] = attempts.get(s, 0) + 1

    def kill(p):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            p.kill()

    try:
        while queue or running:
            while queue and len(running) < a.workers:
                launch(queue.pop(0))
            time.sleep(5)
            for s, (p, st, errf) in list(running.items()):
                lg = LOGS / f"seed{s}.log"
                rc = p.poll()
                beat = max(st, lg.stat().st_mtime) if lg.exists() else st
                stalled = (time.time() - beat) > a.stall_min * 60
                if rc is None and not stalled:
                    continue
                if rc is None:
                    kill(p)
                errf.close()
                del running[s]
                if (PARTS / f"seed{s}.csv").exists():
                    durations.append(time.time() - st)
                    plog(f"✓ seed{s} 完成（{(time.time()-st)/60:.1f} 分钟）")
                elif attempts[s] < 2:
                    plog(f"⚠ seed{s} 失败，重试一次；详情见 logs/seed{s}.stderr")
                    queue.append(s)
                else:
                    failed.append(s)
                    plog(f"✗ seed{s} 两次失败，跳过；详情见 logs/seed{s}.stderr")
            if time.time() - last > 60:
                last = time.time()
                done = len([s for s in seeds if (PARTS / f"seed{s}.csv").exists()])
                eta = f"，预计剩余 {np.mean(durations) * (len(seeds) - done) / a.workers / 60:.0f} 分钟" if durations else ""
                plog(f"进度 {done}/{len(seeds)}，运行中 {len(running)}，失败 {len(failed)}{eta}")
    except KeyboardInterrupt:
        for p, _, _ in running.values():
            kill(p)
        sys.exit(1)
    plog(f"run 结束：失败 {len(failed)} 个 {failed if failed else ''}")


def cmd_status(a):
    done = len(list(PARTS.glob("seed*.csv"))) if PARTS.exists() else 0
    print(f"已完成 {done}/10")
    if PROGRESS.exists():
        print("\n".join(PROGRESS.read_text(encoding="utf-8").strip().splitlines()[-5:]))
    if LOGS.exists():
        for lg in sorted(LOGS.glob("*.log")):
            if not (PARTS / f"{lg.stem}.csv").exists():
                print(f"  {lg.stem}: {(lg.read_text(encoding='utf-8').strip().splitlines() or [''])[-1]}")


# ══════════════════════════════════════════════════════════════════════════
# aggregate
# ══════════════════════════════════════════════════════════════════════════
MET = ["pr_auc", "f1", "pauc01", "tpr_at_1fpr"]


def cmd_aggregate(a):
    files = sorted(PARTS.glob("seed*.csv"))
    if not files:
        sys.exit("✗ 没有结果文件")
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw["budget"] = raw["budget"].astype(str)
    raw.to_csv(OUT / "raw.csv", index=False)
    seeds = sorted(raw.seed.unique())
    order = [str(b) for b in BUDGETS]
    # 零样本：每个 seed 先对源网络取平均，再在两个零样本条件中取较高者
    zs = (raw[raw.condition.str.endswith("Direct")]
          .groupby(["test_id", "kind", "seed", "testset", "condition"])[MET].mean().reset_index())
    zbest = zs.groupby(["test_id", "kind", "seed", "testset"])[MET].max().reset_index()
    rt = raw[raw.condition == "Full-40-retrained"]
    rows = []
    for (tid, kind), g in rt.groupby(["test_id", "kind"], sort=False):
        for b in order:
            gb = g[g.budget == b]
            if gb.empty:
                continue
            for met in MET:
                un = gb[gb.testset == "unseen"].set_index("seed")[met]
                se = gb[gb.testset == "seen"].set_index("seed")[met]
                z = zbest[(zbest.test_id == tid) & (zbest.testset == "unseen")].set_index("seed")[met]
                j = un.index.intersection(z.index).intersection(se.index)
                rows.append(dict(test_id=tid, kind=kind, budget=b, metric=met, n_seeds=len(j),
                                 unseen=un.loc[j].mean(), unseen_sd=un.loc[j].std(ddof=1), seen=se.loc[j].mean(),
                                 zero_shot_best=z.loc[j].mean(), retention=un.loc[j].mean() / se.loc[j].mean(),
                                 gain_vs_zero_shot=(un.loc[j] - z.loc[j]).mean(),
                                 seeds_above_zero_shot=int((un.loc[j] > z.loc[j]).sum())))
    s = pd.DataFrame(rows)
    s.round(4).to_csv(OUT / "summary_by_test.csv", index=False)

    def verdict(r):
        if r.retention >= 0.8 and r.seeds_above_zero_shot >= 0.9 * r.n_seeds:
            return "persists"
        if r.gain_vs_zero_shot >= 0.05:
            return "degraded-useful"
        return "fails"

    L = ["变体留出压力测试（variant-held-out stress test）", "=" * 72,
         f"seed {seeds[0]}–{seeds[-1]}；分类器 Full-40；零样本基线 = max(Full-40-Direct, SHAP-10-Direct)，各自先对源网络平均",
         "判读（预先设定，主指标 PR-AUC，预算 10）：persists = unseen ≥ 0.8×seen 且 ≥9/10 seed 高于零样本；"
         "degraded-useful = 比零样本高 ≥0.05；否则 fails。", ""]
    vt = []
    for b in ("5", "10", "25"):
        sb = s[(s.metric == "pr_auc") & (s.budget == b)].copy()
        sb["verdict"] = sb.apply(verdict, axis=1)
        vt.append(sb.assign(budget=b))
    vt = pd.concat(vt)
    vt.round(4).to_csv(OUT / "verdicts.csv", index=False)
    for kind, lab in (("A_variant", "A. CIC DoS 留一变体（主测试）"), ("B_family", "B. 同网络攻击族互换（次级测试）")):
        L.append(lab)
        for tid, g in vt[vt.kind == kind].groupby("test_id", sort=False):
            cells = "  ".join(f"[{r.budget}] unseen {r.unseen:.3f} / seen {r.seen:.3f} / 零样本 {r.zero_shot_best:.3f} "
                              f"→ {r.verdict}" for _, r in g.iterrows())
            L.append(f"  {tid:<26} {cells}")
        L.append("")
    for met, lab in (("pr_auc", "PR-AUC"), ("f1", "F1"), ("tpr_at_1fpr", "TPR@1%FPR"), ("pauc01", "pAUC@1%FPR")):
        t = s[s.metric == met].pivot_table(index="test_id", columns="budget", values="unseen", sort=False)
        t = t[[b for b in order if b in t.columns]]
        z = s[(s.metric == met)].groupby("test_id", sort=False)["zero_shot_best"].first()
        se10 = s[(s.metric == met) & (s.budget == "10")].set_index("test_id")["seen"]
        t["零样本"] = z
        t["seen@10"] = se10
        L += [f"unseen 测试集 {lab}（seed 均值）", t.round(3).to_string(), ""]
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    plot(s, order)
    plog(f"aggregate 完成 → {(OUT / 'summary.txt').relative_to(ROOT)}")


def plot(s, order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
    for met, lab in (("pr_auc", "PR-AUC (1:10)"), ("tpr_at_1fpr", "TPR at 1% FPR")):
        ids = list(dict.fromkeys(s.test_id))
        nc = 5
        nr = int(np.ceil(len(ids) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.0 * nr), squeeze=False)
        x = np.arange(len(order))
        for ax, tid in zip(axes.ravel(), ids):
            g = s[(s.test_id == tid) & (s.metric == met)].set_index("budget").reindex(order)
            ax.plot(x, g["seen"].values, color="0.5", marker="o", ms=2, lw=1, label="seen variant(s)")
            ax.plot(x, g["unseen"].values, color="#1f77b4", marker="o", ms=2.5, lw=1.3, label="unseen variant/family")
            ax.axhline(g["zero_shot_best"].dropna().iloc[0], color="purple", ls="-.", lw=0.9, label="best zero-shot")
            ax.set_title(tid, fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(order, fontsize=5.5)
            ax.set_ylim(0, 1.02)
        for ax in axes.ravel()[len(ids):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel(lab)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=6, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
        fig.supxlabel("Labelled positive flows from the seen variant(s)", fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_variant_{met}.png")
        fig.savefig(OUT / f"fig_variant_{met}.pdf")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="variant-held-out stress test")
    sp = ap.add_subparsers(dest="cmd", required=True)
    cpu = os.cpu_count() or 8
    p = sp.add_parser("run")
    p.add_argument("--workers", type=int, default=max(1, (cpu - 2) // 4))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--stall-min", type=float, default=40)
    p = sp.add_parser("worker")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--threads", type=int, default=4)
    sp.add_parser("status")
    sp.add_parser("aggregate")
    a = ap.parse_args()
    {"run": cmd_run, "worker": cmd_worker, "status": cmd_status, "aggregate": cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    main()
