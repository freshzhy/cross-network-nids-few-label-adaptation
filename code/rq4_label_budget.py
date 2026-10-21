#!/usr/bin/env python3
"""
code/rq4_label_budget.py — 实验 A：目标域标注预算（target-label budget / adaptation curve）
====================================================================================
问题：源网络迁移来的特征集要在目标网络上重训练才有用——那么目标网络需要多少有标注的流？

设计（预先锁定，跑完后不再改口径）：
  * 复用 RQ3 主实验的数组缓存、seed 42–51 的 80/20 划分、清洗方式和 1:10 / prevalence-matched 测试集
    （测试集与主实验逐行相同，且不参与任何特征选择或训练）。
  * 预算 = 目标训练集中的【正例数】：10, 25, 50, 100, 250, 500, 1000, All；负例按 1:10 抽取。
    同一 seed 内预算是嵌套的（100 个正例包含前 50/25/10 个；负例同理）。
    某档正例不够时记为 NA（不重复采样凑数）。All = 主实验的完整训练集。
  * 条件（全部只用当档预算数据训练分类器）：
      Source-SHAP-10 / Source-MI-10 / Source-Gain-10：源网络冻结特征集（results/rq3_main/feature_sets.json），
          目标端只重训练、不重新排序
      Target-SHAP-10 / Target-MI-10 / Target-Gain-10：只用当档预算数据完成排序与训练
          （SHAP、Gain 来自同档 Full-40 模型；SHAP 在同档训练正例上计算；MI 在同档数据上计算）
      Full-40：同档预算全部特征
      Random-10：R 组随机 10 特征子集的平均（默认 R=10，子集与主实验前 R 组相同）；第 0 组另存为 Random-10-single
    零样本（源端训练、目标端不用标签）不随预算变化，直接取主实验 rq3_main/raw.csv 中 seed 42–51 的结果作水平基线，
    不进入标注效率的统计检验。
  * 预先设定的命题（H2/H3 不成立也照实报告）：
      H1  随预算增加，各重训练条件性能总体提高（F1(All) > F1(10)）
      H2  低预算（≤100 个正例）时，源特征集相对 Random-10 保持正优势
      H3  源特征集与目标自选特征集之间的差异 |Source − Target| 随预算增加而缩小（低预算均值 > All）
  * 交叉点（预先定义）：对每个配置与方法 m ∈ {SHAP, MI, Gain}，
      b* = 最小预算 b，使得从 b 起所有更大预算都有 F1(Target-m) − F1(Source-m) ≥ −δ（δ = 0.01，1:10 测试集，
      seed 均值）。b* = 10 表示从最低预算起目标自选就不差；若到 All 仍不满足记为 ">All"。
  * 推断：配置级（n = 18，seed 均值）Wilcoxon + 9 簇（互逆配对合并）精确符号翻转检验，与主实验一致。

第二轮（实验 A 跑完后追加）：
  * --dedup：在 rq3_dedup 的精确去重缓存上重跑同一设计（标签冲突行全删、同标签重复只留一条），
    输出 results/rq4_budget_dedup/；零样本基线取 results/rq3_dedup/rq3_main/raw.csv（同为 seed 42–51），
    源特征集仍用主实验冻结的 feature_sets.json（与 rq3_transfer --dedup 一致）。
  * --posthoc：【事后探索性分析，非预注册】追加 1、2、5 个正例三档（与主预算同一嵌套顺序），
    单独存为 parts/*__posthoc.csv；不进入 H1–H3 与交叉点，只在 summary 的 "POST HOC EXPLORATORY" 一节报告。
  * dupdiag：统计每档预算下，1:10 测试正例中有多少与该档训练正例 / 训练负例【40 特征完全相同】，
    用来判断"少量标注就够"是否由重复流放大。
  * compare：原始数据 vs 去重数据并排对照（results/rq4_budget_dedup/compare_orig_vs_dedup.txt）。

运行（项目根目录）：
  ./run.sh rq4                 # 全部步骤（原始 + 去重 + 事后档 + 重复诊断 + 汇总 + 变体留出）
  ./run.sh bg rq4              # 后台运行 + 防休眠
  ./run.sh status              # 查看进度
  python code/rq4_label_budget.py run [--workers 4 --threads 4 --n-seeds 10 --n-random 10]
  python code/rq4_label_budget.py aggregate
输出：results/rq4_budget/
"""
import argparse
import itertools
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

import rq3_transfer as base          # 复用主实验的缓存、划分、清洗、训练与指标

ROOT = Path(__file__).resolve().parent.parent   # repository root (scripts live in code/)
OUT = ROOT / "results" / "rq4_budget"
PARTS, LOGS, PROGRESS = OUT / "parts", OUT / "logs", OUT / "progress.log"
MAIN = ROOT / "results" / "rq3_main"

BUDGETS = [10, 25, 50, 100, 250, 500, 1000, "All"]
LOW_BUDGETS = [10, 25, 50, 100]
DELTA = 0.01                       # 交叉点的实际等效阈值（F1）
METHODS = ["SHAP", "MI", "Gain"]
SRC_KEY = {"SHAP": "SHAP-10", "MI": "MI-10", "Gain": "Gain-10"}
NEG_RATIO = 10
METRICS = base.METRICS
POSTHOC_BUDGETS = [1, 2, 5]        # 事后探索性分析（非预注册）
DEDUP = False
ZS_RAW = MAIN / "raw.csv"


def set_mode(dedup):
    """--dedup：读 results/rq3_dedup/cache，输出到 results/rq4_budget_dedup/。"""
    global OUT, PARTS, LOGS, PROGRESS, DEDUP, ZS_RAW
    DEDUP = bool(dedup)
    base.set_mode(DEDUP)
    OUT = ROOT / "results" / ("rq4_budget_dedup" if DEDUP else "rq4_budget")
    PARTS, LOGS, PROGRESS = OUT / "parts", OUT / "logs", OUT / "progress.log"
    ZS_RAW = ROOT / "results" / ("rq3_dedup" if DEDUP else "rq3_main") / "raw.csv"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def plog(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def budget_label(b):
    return "All" if b == "All" else str(b)


# ══════════════════════════════════════════════════════════════════════════
# worker：一个 (类别, seed) 任务
# ══════════════════════════════════════════════════════════════════════════
def target_rankings(Xb, yb, seed, threads, fc):
    """只用当档预算数据：Full-40 模型 → Gain、训练正例 TreeSHAP；MI 单独计算。"""
    import xgboost as xgb
    from sklearn.feature_selection import mutual_info_classif
    m40 = base.fit_xgb(Xb, yb, seed, threads)
    score = m40.get_booster().get_score(importance_type="gain")
    gain = np.array([score.get(f"f{i}", 0.0) for i in range(len(fc))])
    pos = np.flatnonzero(yb == 1)
    if len(pos) > 5_000:
        pos = np.sort(np.random.default_rng(seed).choice(pos, 5_000, replace=False))
    contrib = m40.get_booster().predict(xgb.DMatrix(Xb[pos]), pred_contribs=True)[:, :-1]
    shap_v = np.abs(contrib).mean(axis=0)
    sel = np.arange(len(yb))
    if len(yb) > base.MI_MAX_ROWS:                          # 与主实验相同的分层子抽样
        rng = np.random.default_rng(seed)
        p_, n_ = np.flatnonzero(yb == 1), np.flatnonzero(yb == 0)
        k = min(len(p_), int(base.MI_MAX_ROWS * len(p_) / len(yb)) + 1)
        sel = np.concatenate([rng.choice(p_, k, replace=False),
                              rng.choice(n_, base.MI_MAX_ROWS - k, replace=False)])
    try:
        mi = mutual_info_classif(Xb[sel], yb[sel], random_state=seed, n_jobs=threads)
    except ValueError:                       # 极小预算下 kNN 估计可能失败 → Target-MI-10 记为 NA
        mi = None
    top = lambda v: [int(i) for i in np.argsort(-np.asarray(v), kind="stable")[:base.TOP_K]]
    return m40, {"SHAP": top(shap_v), "MI": None if mi is None else top(mi), "Gain": top(gain)}


def target_data(sem, tgt, seed, raw_counts=None):
    """与主实验完全相同的抽样、80/20 划分与清洗；返回目标端数据与嵌套预算顺序。"""
    mp = base.SEMANTIC_MAPPING[sem]
    X, codes, meta = base.load_cache(tgt)
    pos_codes = [meta["labels"].index(l) for l in mp[tgt] if l in meta["labels"]]
    tr, te, ytr, yte = base.split_indices(codes, pos_codes, seed)
    Xall = base.clean_subset(X[np.concatenate([tr, te])])
    Xtr, Xte = Xall[:len(tr)], Xall[len(tr):]
    del Xall
    Xn = yn = None
    if raw_counts is not None:
        nat = base.natural_test(codes, meta, raw_counts.get(tgt, {}), pos_codes, tr, te[yte == 1], seed)
        if nat is not None:
            Xn, yn = base.clean_subset(X[nat[0]]), nat[1]
    rng = np.random.default_rng(seed + 60000 + base.DS_ORDER.index(tgt))   # 嵌套预算：只排一次顺序
    pos_all = rng.permutation(np.flatnonzero(ytr == 1))
    neg_all = rng.permutation(np.flatnonzero(ytr == 0))
    return dict(fc=meta["feature_cols"], Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xn=Xn, yn=yn,
                pos_all=pos_all, neg_all=neg_all)


def budget_index(b, T):
    if b == "All":
        return np.arange(len(T["ytr"]))
    if b > len(T["pos_all"]):
        return None
    return np.sort(np.concatenate([T["pos_all"][:b], T["neg_all"][:min(len(T["neg_all"]), NEG_RATIO * b)]]))


def cmd_worker(a):
    sem, seed, threads = a.cls, a.seed, a.threads
    tag = f"{sem}__seed{seed}" + ("__posthoc" if a.posthoc else "")
    hb = base.Heartbeat(LOGS / f"{tag}.log")
    hb(f"开始 {tag} threads={threads} R={a.n_random} dedup={DEDUP}")
    t0 = time.time()
    fsets = json.loads((MAIN / "feature_sets.json").read_text())[sem]
    raw_counts = json.loads((ROOT / "results" / "rq3_diagnostics" / "raw_class_counts.json").read_text())
    mp = base.SEMANTIC_MAPPING[sem]
    dss = [d for d in base.DS_ORDER if d in mp]
    rows = []

    budgets = POSTHOC_BUDGETS if a.posthoc else BUDGETS
    for tgt in dss:
        T = target_data(sem, tgt, seed, raw_counts)
        fc = T["fc"]
        col = {f: i for i, f in enumerate(fc)}
        Xtr, ytr, Xte, yte, Xn, yn = T["Xtr"], T["ytr"], T["Xte"], T["yte"], T["Xn"], T["yn"]

        def evaluate(model, cols):
            out = {"std": base.metrics(yte, model.predict_proba(Xte[:, cols])[:, 1])}
            if Xn is not None:
                out["nat"] = base.metrics(yn, model.predict_proba(Xn[:, cols])[:, 1])
            return out

        subsets = base.random_subsets(fc, seed, a.n_random)
        hb(f"{tgt}: 训练正例 {len(T['pos_all']):,} / 负例 {len(T['neg_all']):,}")

        for b in budgets:
            idx = budget_index(b, T)
            if idx is None:
                hb(f"{tgt}: 预算 {b} 超过可用正例 {len(T['pos_all'])} → NA")
                continue
            Xb, yb = Xtr[idx], ytr[idx]
            npos, nneg = int((yb == 1).sum()), int((yb == 0).sum())

            def add(source, cond, res, feats=None, extra=None):
                for ts, met in res.items():
                    r = dict(semantic_class=sem, target=tgt, source=source, condition=cond,
                             budget=budget_label(b), n_pos_train=npos, n_neg_train=nneg,
                             seed=seed, testset=ts, posthoc=bool(a.posthoc), **met)
                    if feats is not None:
                        r["features"] = ";".join(fc[i] for i in feats)
                    if extra:
                        r.update(extra)
                    rows.append(r)

            all_cols = list(range(len(fc)))
            m40, ranks = target_rankings(Xb, yb, seed, threads, fc)
            add("—", "Full-40", evaluate(m40, all_cols))
            cache = {}
            for meth in METHODS:
                cols = ranks[meth]
                if cols is None:
                    continue
                key = tuple(sorted(cols))
                if key not in cache:
                    cache[key] = evaluate(base.fit_xgb(Xb[:, cols], yb, seed, threads), cols)
                add(tgt, f"Target-{meth}-10", cache[key], cols)
            for src in dss:
                if src == tgt:
                    continue
                for meth in METHODS:
                    feats = fsets[src][SRC_KEY[meth]]
                    cols = [col[f] for f in feats]
                    key = tuple(sorted(cols))
                    if key not in cache:
                        cache[key] = evaluate(base.fit_xgb(Xb[:, cols], yb, seed, threads), cols)
                    add(src, f"Source-{meth}-10", cache[key])
            acc = {}
            for ri, sub in enumerate(subsets):
                cols = [col[f] for f in sub]
                res = evaluate(base.fit_xgb(Xb[:, cols], yb, seed, threads), cols)
                if ri == 0:
                    add("Random", "Random-10-single", res)
                for ts, met in res.items():
                    acc.setdefault(ts, []).append(met)
            add("Random", "Random-10", {ts: pd.DataFrame(l).mean().round(6).to_dict() for ts, l in acc.items()},
                extra={"rand_n": len(subsets)})
            hb(f"{tgt}: 预算 {budget_label(b)}（正例 {npos:,}）完成  累计 {time.time()-t0:.0f}s")

    out = pd.DataFrame(rows)
    tmp = PARTS / f"{tag}.csv.tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, PARTS / f"{tag}.csv")
    hb(f"完成 {tag}，{len(out)} 行，用时 {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════════
# run：调度器（子进程 + 超时 + 心跳 + 断点续跑）
# ══════════════════════════════════════════════════════════════════════════
def cmd_run(a):
    for p in (MAIN / "feature_sets.json", ROOT / "results" / "rq3_diagnostics" / "raw_class_counts.json"):
        if not p.exists():
            sys.exit(f"✗ 缺少 {p.relative_to(ROOT)}：需要主实验（rq3_transfer prep）与 diag 的结果")
    miss = [d for d in base.DS_ORDER if not (base.CACHE / f"{d}_X.npy").exists()]
    if miss:
        sys.exit(f"✗ 缓存缺失 {miss}（{base.CACHE.relative_to(ROOT)}）")
    for d in (PARTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    seeds = list(range(base.SEED_BASE, base.SEED_BASE + a.n_seeds))
    sfx = "__posthoc" if a.posthoc else ""
    alltasks = [(c, s) for s in seeds for c in a.classes]
    tasks = [t for t in alltasks if not (PARTS / f"{t[0]}__seed{t[1]}{sfx}.csv").exists()]
    (OUT / ("run_config_posthoc.json" if a.posthoc else "run_config.json")).write_text(
        json.dumps(vars(a), indent=1, default=str))
    plog(f"run 开始（{'去重数据' if DEDUP else '原始数据'}{'，事后档 1/2/5' if a.posthoc else ''}）：总任务 {len(alltasks)}，待完成 {len(tasks)}，并行 {a.workers} × 线程 {a.threads}，R={a.n_random}")
    env = dict(os.environ, A7_THREADS=str(a.threads), PYTHONUNBUFFERED="1")
    running, attempts, failed, durations, queue, last = {}, {}, [], [], list(tasks), 0

    def launch(t):
        tag = f"{t[0]}__seed{t[1]}{sfx}"
        errf = open(LOGS / f"{tag}.stderr", "a")
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--cls", t[0],
                              "--seed", str(t[1]), "--threads", str(a.threads), "--n-random", str(a.n_random)]
                             + (["--dedup"] if DEDUP else []) + (["--posthoc"] if a.posthoc else []),
                             env=env, stdout=errf, stderr=errf, start_new_session=True)
        running[t] = (p, time.time(), errf)
        attempts[t] = attempts.get(t, 0) + 1

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
            for t, (p, st, errf) in list(running.items()):
                tag = f"{t[0]}__seed{t[1]}{sfx}"
                lg = LOGS / f"{tag}.log"
                rc = p.poll()
                beat = max(st, lg.stat().st_mtime) if lg.exists() else st
                stalled = (time.time() - beat) > a.stall_min * 60
                timeout = (time.time() - st) > a.timeout_min * 60
                if rc is None and not (stalled or timeout):
                    continue
                if rc is None:
                    kill(p)
                reason = ("超时" if timeout else "心跳停滞") if rc is None else f"退出码 {rc}"
                errf.close()
                del running[t]
                if (PARTS / f"{tag}.csv").exists():
                    durations.append(time.time() - st)
                    plog(f"✓ {tag} 完成（{(time.time()-st)/60:.1f} 分钟）")
                elif attempts[t] < 2:
                    plog(f"⚠ {tag} 失败（{reason}），重试一次；详情见 logs/{tag}.stderr")
                    queue.append(t)
                else:
                    failed.append(tag)
                    plog(f"✗ {tag} 两次失败（{reason}），跳过；详情见 logs/{tag}.stderr")
            if time.time() - last > 60:
                last = time.time()
                done = len([t for t in alltasks if (PARTS / f"{t[0]}__seed{t[1]}{sfx}.csv").exists()])
                eta = (f"，预计剩余 {np.mean(durations) * (len(alltasks) - done) / a.workers / 60:.0f} 分钟"
                       if durations else "")
                plog(f"进度 {done}/{len(alltasks)}，运行中 {len(running)}，失败 {len(failed)}{eta}")
    except KeyboardInterrupt:
        plog("收到中断，终止子进程（已完成的任务保留，重跑 run 自动续跑）")
        for p, _, _ in running.values():
            kill(p)
        sys.exit(1)
    plog(f"run 结束：失败 {len(failed)} 个 {failed if failed else ''}")


def cmd_status(a):
    cfg = OUT / "run_config.json"
    c = json.loads(cfg.read_text()) if cfg.exists() else {"n_seeds": 10, "classes": list(base.SEMANTIC_MAPPING)}
    total = c["n_seeds"] * len(c["classes"])
    done = len(list(PARTS.glob("*__seed*[0-9].csv"))) if PARTS.exists() else 0
    post = len(list(PARTS.glob("*__posthoc.csv"))) if PARTS.exists() else 0
    print(f"[{'去重' if DEDUP else '原始'}] 主预算已完成 {done}/{total}；事后档已完成 {post}/{total}")
    if LOGS.exists():
        for lg in sorted(LOGS.glob("*.log")):
            if (PARTS / f"{lg.stem}.csv").exists():
                continue
            last = (lg.read_text(encoding="utf-8").strip().splitlines() or [""])[-1]
            age = (time.time() - lg.stat().st_mtime) / 60
            print(f"  {lg.stem:<26} {age:5.1f} 分钟前：{last}{'  ⚠ 超过10分钟无心跳' if age > 10 else ''}")
    if PROGRESS.exists():
        print("\n最近进度：\n" + "\n".join(PROGRESS.read_text(encoding="utf-8").strip().splitlines()[-5:]))


# ══════════════════════════════════════════════════════════════════════════
# aggregate：曲线、H1–H3、交叉点、图
# ══════════════════════════════════════════════════════════════════════════
def level_test(cfg_delta, alternative):
    """cfg_delta: DataFrame[semantic_class, pair, delta] → 配置级 Wilcoxon + 9 簇符号翻转。"""
    from scipy.stats import wilcoxon
    D = cfg_delta["delta"].to_numpy()
    r = dict(n_configs=len(D), n_positive=int((D > 0).sum()), mean_delta=float(np.mean(D)) if len(D) else np.nan)
    r["p_config_wilcoxon"] = wilcoxon(D, alternative=alternative).pvalue if len(D) >= 3 and np.any(D != 0) else np.nan
    cl = cfg_delta.semantic_class + "|" + cfg_delta.pair.map(lambda s: "-".join(sorted(s.split("→"))))
    cm = cfg_delta.groupby(cl.values)["delta"].mean().to_numpy()
    if len(cm) >= 2:
        obs, cnt, N = cm.mean(), 0, 0
        for sg in itertools.product([1, -1], repeat=len(cm)):
            N += 1
            v = np.mean(np.array(sg) * cm)
            cnt += (v >= obs - 1e-12) if alternative == "greater" else (abs(v) >= abs(obs) - 1e-12)
        r["n_clusters"], r["p_cluster_signflip"] = len(cm), cnt / N
    return r


def cmd_aggregate(a):
    files = sorted(PARTS.glob("*__seed*[0-9].csv"))
    if not files:
        sys.exit("✗ 没有结果文件")
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw["posthoc"] = False
    pfiles = sorted(PARTS.glob("*__posthoc.csv"))
    if pfiles:
        raw = pd.concat([raw] + [pd.read_csv(f) for f in pfiles], ignore_index=True)
        raw["posthoc"] = raw["posthoc"].astype(bool)
    raw["budget"] = raw["budget"].astype(str)
    raw.to_csv(OUT / "raw.csv", index=False)
    seeds = sorted(raw.seed.unique())
    plog(f"aggregate（{'去重' if DEDUP else '原始'}数据）：{len(files)} 个主任务 + {len(pfiles)} 个事后任务，"
         f"{len(raw):,} 行，seed {seeds[0]}–{seeds[-1]}")
    order = [budget_label(b) for b in BUDGETS]                      # 预注册的预算（检验只用这些）
    ph = [str(b) for b in POSTHOC_BUDGETS if str(b) in set(raw.budget)]
    order_all = ph + order                                          # 曲线与图（事后档带 * 标记）

    # seed 均值曲线（目标端条件没有 source → 展开到每个 source→target 配置）
    cfgs = raw[raw.condition == "Source-SHAP-10"][["semantic_class", "target", "source"]].drop_duplicates()
    rows = []
    for _, c in cfgs.iterrows():
        sub = raw[(raw.semantic_class == c.semantic_class) & (raw.target == c.target)]
        sub = sub[(sub.source == c.source) | ~sub.condition.str.startswith("Source-")]
        g = sub.groupby(["condition", "budget", "testset"])[METRICS + ["n_pos_train"]].mean().reset_index()
        g.insert(0, "pair", f"{c.source}→{c.target}")
        g.insert(0, "semantic_class", c.semantic_class)
        rows.append(g)
    curve = pd.concat(rows, ignore_index=True)
    curve["budget"] = pd.Categorical(curve.budget, order_all, ordered=True)
    curve = curve.sort_values(["semantic_class", "pair", "condition", "testset", "budget"])
    curve.round(6).to_csv(OUT / "curve_by_config.csv", index=False)

    # 零样本基线（主实验 rq3_main/raw.csv，相同 seed）
    zs = None
    a7 = ZS_RAW
    if a7.exists():
        r7 = pd.read_csv(a7)
        r7 = r7[r7.seed.isin(seeds) & r7.condition.isin(["SHAP-10-Direct", "Random-10-Direct"])]
        zs = r7.groupby(["semantic_class", "target", "source", "condition", "testset"])[METRICS].mean().reset_index()
        zs["pair"] = zs.source + "→" + zs.target
        zs.round(6).to_csv(OUT / "zero_shot_baseline.csv", index=False)

    def cfg_delta(A, B, budget, ts="std", met="f1"):
        sub = curve[(curve.budget == budget) & (curve.testset == ts)]
        pa = sub[sub.condition == A].set_index(["semantic_class", "pair"])[met]
        pb = sub[sub.condition == B].set_index(["semantic_class", "pair"])[met]
        j = pa.index.intersection(pb.index)
        return pd.DataFrame({"semantic_class": [k[0] for k in j], "pair": [k[1] for k in j],
                             "delta": (pa.loc[j] - pb.loc[j]).to_numpy()})

    # 每档预算的配置级比较
    comps = [(f"Source-{m}-10", "Random-10", "greater") for m in METHODS] + \
            [(f"Target-{m}-10", "Random-10", "greater") for m in METHODS] + \
            [(f"Source-{m}-10", f"Target-{m}-10", "two-sided") for m in METHODS] + \
            [("Source-SHAP-10", "Full-40", "two-sided"), ("Source-SHAP-10", "Source-MI-10", "two-sided"),
             ("Source-SHAP-10", "Source-Gain-10", "two-sided")]
    lev = []
    for A, B, alt in comps:
        for b in order:
            for ts in ("std", "nat"):
                for met in ("f1", "pr_auc", "mcc", "pauc01"):
                    d = cfg_delta(A, B, b, ts, met)
                    if len(d) >= 3:
                        lev.append(dict(comparison=f"{A} vs {B}", budget=b, testset=ts, metric=met,
                                        **level_test(d, alt)))
    lv = pd.DataFrame(lev)
    lv.round(6).to_csv(OUT / "config_level_tests.csv", index=False)

    # 预设命题
    hyp = []
    conds = ["Full-40", "Random-10"] + [f"{s}-{m}-10" for s in ("Source", "Target") for m in METHODS]
    for c in conds:                                                     # H1：F1(All) − F1(10) > 0
        hi = curve[(curve.condition == c) & (curve.budget == "All") & (curve.testset == "std")].set_index(["semantic_class", "pair"])["f1"]
        lo = curve[(curve.condition == c) & (curve.budget == "10") & (curve.testset == "std")].set_index(["semantic_class", "pair"])["f1"]
        j = hi.index.intersection(lo.index)
        d = pd.DataFrame({"semantic_class": [k[0] for k in j], "pair": [k[1] for k in j], "delta": (hi.loc[j] - lo.loc[j]).values})
        hyp.append(dict(hypothesis="H1", test=f"{c}: F1(All) − F1(10)", **level_test(d, "greater")))
    for m in METHODS:                                                   # H2：低预算均值 Source − Random > 0
        ds_ = [cfg_delta(f"Source-{m}-10", "Random-10", str(b)) for b in LOW_BUDGETS]
        d = pd.concat(ds_).groupby(["semantic_class", "pair"], as_index=False)["delta"].mean()
        hyp.append(dict(hypothesis="H2", test=f"Source-{m}-10 − Random-10, mean over budgets {LOW_BUDGETS}", **level_test(d, "greater")))
    for m in METHODS:                                                   # H3：|S − T| 低预算均值 − |S − T|(All) > 0
        low = pd.concat([cfg_delta(f"Source-{m}-10", f"Target-{m}-10", str(b)) for b in LOW_BUDGETS])
        low["delta"] = low["delta"].abs()
        low = low.groupby(["semantic_class", "pair"])["delta"].mean()
        hi = cfg_delta(f"Source-{m}-10", f"Target-{m}-10", "All").set_index(["semantic_class", "pair"])["delta"].abs()
        j = low.index.intersection(hi.index)
        d = pd.DataFrame({"semantic_class": [k[0] for k in j], "pair": [k[1] for k in j], "delta": (low.loc[j] - hi.loc[j]).values})
        hyp.append(dict(hypothesis="H3", test=f"|Source-{m} − Target-{m}|: mean over {LOW_BUDGETS} − All", **level_test(d, "greater")))
    hy = pd.DataFrame(hyp)
    hy.round(6).to_csv(OUT / "hypotheses.csv", index=False)

    # 交叉点
    cross = []
    for (sem, pair), g in curve[curve.testset == "std"].groupby(["semantic_class", "pair"], observed=True):
        for m in METHODS:
            s = g[g.condition == f"Source-{m}-10"].set_index("budget")["f1"]
            t = g[g.condition == f"Target-{m}-10"].set_index("budget")["f1"]
            avail = [b for b in order if b in s.index and b in t.index and not np.isnan(s.get(b, np.nan))]
            diff = {b: float(t[b] - s[b]) for b in avail}
            bstar = ">All"
            for i, b in enumerate(avail):
                if all(diff[x] >= -DELTA for x in avail[i:]):
                    bstar = b
                    break
            cross.append(dict(semantic_class=sem, pair=pair, method=m, crossover_budget=bstar,
                              **{f"T_minus_S@{b}": round(diff[b], 4) for b in avail}))
    cr = pd.DataFrame(cross)
    cr.to_csv(OUT / "crossover.csv", index=False)

    # 与主实验的一致性核对（All 档 = 主实验完整训练集）
    chk = []
    if a7.exists():
        r7 = pd.read_csv(a7)
        r7 = r7[r7.seed.isin(seeds) & (r7.testset == "std")]
        allb = raw[(raw.budget == "All") & (raw.testset == "std")]
        for new, old in [("Full-40", "Full-40"), ("Source-SHAP-10", "SHAP-10"), ("Source-MI-10", "MI-10"),
                         ("Source-Gain-10", "Gain-10"), ("Random-10-single", "Random-10-single")]:
            x = allb[allb.condition == new][["semantic_class", "target", "source", "seed", "f1"]]
            y = r7[r7.condition == old][["semantic_class", "target", "source", "seed", "f1"]]
            on = ["semantic_class", "target", "seed"] + ([] if new in ("Full-40", "Random-10-single") else ["source"])
            mg = x.merge(y.drop(columns=[] if "source" in on else ["source"]), on=on, suffixes=("_new", "_main"))
            if len(mg):
                dd = (mg.f1_new - mg.f1_main).abs()
                chk.append(f"  {new:<18} 对 {old:<17} 匹配 {len(mg):>3} 行，F1 最大绝对差 {dd.max():.4f}，平均 {dd.mean():.4f}")

    # 文字摘要
    L = [f"实验 A：目标域标注预算（{'精确去重数据' if DEDUP else '原始数据'}）", "=" * 72, f"seed {seeds[0]}–{seeds[-1]}，预算（正例数）{order}，负例 1:{NEG_RATIO}，δ = {DELTA}", ""]
    if chk:
        L += ["All 档与主实验 rq3_main/raw.csv 的一致性核对（std 测试集；差异应接近 0，非 0 通常来自线程数）："] + chk + [""]
    for sem in sorted(curve.semantic_class.unique()):
        t = (curve[(curve.semantic_class == sem) & (curve.testset == "std")]
             .groupby(["condition", "budget"], observed=True)["f1"].mean().unstack("budget").round(3))
        t = t[[b for b in order_all if b in t.columns]]
        t.columns = [f"{b}*" if b in ph else b for b in t.columns]
        L += [f"{sem}：跨配置平均 F1（1:10 测试集；带 * 的列为事后探索性预算）", t.to_string()]
        if zs is not None:
            z = zs[(zs.semantic_class == sem) & (zs.testset == "std")].groupby("condition")["f1"].mean()
            L.append("  零样本（不随预算变化）：" + "，".join(f"{k} = {v:.3f}" for k, v in z.items()))
        L.append("")
    L += ["预设命题（配置级 Wilcoxon / 9 簇符号翻转，F1，1:10 测试集）："]
    for _, r in hy.iterrows():
        L.append(f"  {r.hypothesis}  {r.test:<62} Δ>0: {r.n_positive}/{r.n_configs}  均值 {r.mean_delta:+.4f}  "
                 f"p_cfg = {r.p_config_wilcoxon:.4f}  p_cluster = {r.get('p_cluster_signflip', np.nan):.4f}")
    L += ["", "源特征集 vs Random-10 / vs 目标自选（F1，1:10 测试集），按预算："]
    key = lv[(lv.testset == "std") & (lv.metric == "f1") &
             lv.comparison.isin([f"Source-{m}-10 vs Random-10" for m in METHODS] +
                                [f"Source-{m}-10 vs Target-{m}-10" for m in METHODS])]
    for comp, g in key.groupby("comparison", sort=False):
        L.append(f"  {comp}")
        for _, r in g.iterrows():
            L.append(f"    预算 {r.budget:>4}: {r.n_positive:>2}/{r.n_configs}  均值Δ {r.mean_delta:+.4f}  "
                     f"p_cfg {r.p_config_wilcoxon:.4f}  p_cluster {r.p_cluster_signflip:.4f}")
    L += ["", f"交叉点 b*（目标自选 Top-10 从该预算起始终不差于源特征集超过 {DELTA}）分布："]
    for m in METHODS:
        vc = cr[cr.method == m].crossover_budget.value_counts()
        L.append(f"  {m:<5} " + "，".join(f"{b}: {vc.get(b, 0)}" for b in order + [">All"] if vc.get(b, 0)))
    L += posthoc_section(curve, zs, order_all, ph)
    L += dup_section()
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    plot(curve, zs, order_all, ph)
    plog(f"aggregate 完成 → {OUT.relative_to(ROOT)}/summary.txt")


def plot(curve, zs, order, ph=()):
    lab_ = [f"{b}*" if b in ph else b for b in order]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
    sty = {"Source-SHAP-10": ("#1f77b4", "-", "o"), "Source-MI-10": ("#2ca02c", "-", "s"),
           "Source-Gain-10": ("#d62728", "-", "^"), "Target-SHAP-10": ("#1f77b4", "--", "o"),
           "Target-MI-10": ("#2ca02c", "--", "s"), "Target-Gain-10": ("#d62728", "--", "^"),
           "Full-40": ("black", "-", "D"), "Random-10": ("grey", ":", "x")}
    x = np.arange(len(order))
    for met, lab in (("f1", "F1"), ("pr_auc", "PR-AUC")):
        for ts in ("std", "nat"):
            sub = curve[curve.testset == ts]
            classes = sorted(sub.semantic_class.unique())
            if not classes:
                continue
            fig, axes = plt.subplots(1, len(classes), figsize=(3.6 * len(classes), 3.0), squeeze=False)
            for ax, sem in zip(axes[0], classes):
                s = sub[sub.semantic_class == sem]
                for cond, (c, ls, mk) in sty.items():
                    y = s[s.condition == cond].groupby("budget", observed=True)[met].mean().reindex(order)
                    ax.plot(x, y.values, color=c, ls=ls, marker=mk, ms=3, lw=1.1, label=cond)
                if zs is not None:
                    z = zs[(zs.semantic_class == sem) & (zs.testset == ts) & (zs.condition == "SHAP-10-Direct")][met].mean()
                    ax.axhline(z, color="purple", lw=0.9, ls="-.", label="SHAP-10-Direct (zero-shot)")
                ax.set_xticks(x)
                ax.set_xticklabels(lab_)
                if ph:
                    ax.axvspan(-0.5, len(ph) - 0.5, color="0.92", zorder=0)
                ax.set_xlabel("Target positive labels" + ("  (* = post hoc exploratory)" if ph else ""))
                ax.set_ylabel(f"{lab} ({'1:10' if ts == 'std' else 'prevalence-matched'})")
                ax.set_title(sem)
            h, l = axes[0][-1].get_legend_handles_labels()
            fig.legend(h, l, fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
            fig.savefig(OUT / f"fig_budget_{met}_{ts}.png")
            fig.savefig(OUT / f"fig_budget_{met}_{ts}.pdf")
            plt.close(fig)
    # 18 个配置的小图（F1，1:10）
    sub = curve[curve.testset == "std"]
    keys = sub[["semantic_class", "pair"]].drop_duplicates().values.tolist()
    n = len(keys)
    nc = 6
    fig, axes = plt.subplots(int(np.ceil(n / nc)), nc, figsize=(2.2 * nc, 1.9 * np.ceil(n / nc)), squeeze=False)
    for ax, (sem, pair) in zip(axes.ravel(), keys):
        s = sub[(sub.semantic_class == sem) & (sub.pair == pair)]
        for cond in ("Source-SHAP-10", "Target-SHAP-10", "Full-40", "Random-10"):
            c, ls, mk = sty[cond]
            y = s[s.condition == cond].set_index("budget")["f1"].reindex(order)
            ax.plot(x, y.values, color=c, ls=ls, marker=mk, ms=2, lw=0.9, label=cond)
        ax.set_title(f"{sem[:5]} {pair}", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(lab_, fontsize=5, rotation=45)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0][0].legend(fontsize=5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_budget_per_config_f1.png")
    fig.savefig(OUT / "fig_budget_per_config_f1.pdf")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# 事后探索性分析（非预注册）：各档预算 vs 零样本
# ══════════════════════════════════════════════════════════════════════════
def posthoc_section(curve, zs, order_all, ph):
    L = ["", "=" * 72, "POST HOC EXPLORATORY ANALYSIS（事后探索性分析，非预注册；不参与 H1–H3 与交叉点）",
         "  ① 1/2/5 个正例三档是在看到实验 A 结果后追加的；② 与零样本基线的比较也属于事后比较。", "=" * 72]
    if zs is None:
        return L + ["  （缺少零样本基线 rq3_main/raw.csv，跳过）"]
    out = []
    for cond in ("Full-40", "Target-SHAP-10", "Source-SHAP-10", "Random-10"):
        for ts, met in (("std", "f1"), ("nat", "pr_auc"), ("std", "pauc01")):
            z = zs[(zs.condition == "SHAP-10-Direct") & (zs.testset == ts)].set_index(["semantic_class", "pair"])[met]
            for b in order_all:
                c = curve[(curve.condition == cond) & (curve.budget == b) & (curve.testset == ts)].set_index(["semantic_class", "pair"])[met]
                j = c.index.intersection(z.index)
                if len(j) < 3:
                    continue
                d = pd.DataFrame({"semantic_class": [k[0] for k in j], "pair": [k[1] for k in j],
                                  "delta": (c.loc[j] - z.loc[j]).values})
                r = level_test(d, "greater")
                out.append(dict(condition=cond, testset=ts, metric=met, budget=b, posthoc_budget=b in ph,
                                mean_retrained=float(c.loc[j].mean()), mean_zero_shot=float(z.loc[j].mean()), **r))
    po = pd.DataFrame(out)
    po.round(6).to_csv(OUT / "posthoc_vs_zero_shot.csv", index=False)
    L.append("重训练（当档预算）vs 零样本 SHAP-10-Direct：配置级 Wilcoxon（单侧）/ 9 簇符号翻转")
    for (cond, ts, met), g in po.groupby(["condition", "testset", "metric"], sort=False):
        L.append(f"  {cond}，{met}（{'1:10' if ts == 'std' else 'prevalence-matched'}）")
        for _, r in g.iterrows():
            L.append(f"    预算 {r.budget + ('*' if r.posthoc_budget else ''):>5}: 重训练 {r.mean_retrained:.3f} vs 零样本 "
                     f"{r.mean_zero_shot:.3f}  Δ>0: {r.n_positive}/{r.n_configs}  p_cfg {r.p_config_wilcoxon:.4f}  "
                     f"p_cluster {r.get('p_cluster_signflip', np.nan):.4f}")
    return L


def dup_section():
    f = OUT / "dup_diag.csv"
    if not f.exists():
        return ["", "（未找到 dup_diag.csv：先运行 dupdiag）"]
    d = pd.read_csv(f)
    d["budget"] = d["budget"].astype(str)
    order = [str(b) for b in POSTHOC_BUDGETS] + [budget_label(b) for b in BUDGETS]
    t = (d.groupby(["semantic_class", "target", "budget"])[["test_pos_dup_in_train_pos", "test_pos_dup_in_train_neg"]]
         .mean().unstack("budget"))
    L = ["", "重复流诊断：1:10 测试正例中，与该档训练正例（/训练负例）40 特征完全相同的比例（seed 均值）"]
    for key, lab in (("test_pos_dup_in_train_pos", "与训练正例相同"), ("test_pos_dup_in_train_neg", "与训练负例相同（标签冲突）")):
        tt = t[key][[b for b in order if b in t[key].columns]].round(3)
        L += [f"  {lab}：", tt.to_string()]
    return L


def row_keys(A):
    A = np.ascontiguousarray(A)
    return [r.tobytes() for r in A]


def cmd_dupdiag(a):
    rows = []
    seeds = list(range(base.SEED_BASE, base.SEED_BASE + a.n_seeds))
    for sem in a.classes:
        dss = [d for d in base.DS_ORDER if d in base.SEMANTIC_MAPPING[sem]]
        for seed in seeds:
            for tgt in dss:
                T = target_data(sem, tgt, seed)
                tp = row_keys(T["Xte"][T["yte"] == 1])
                for b in POSTHOC_BUDGETS + BUDGETS:
                    idx = budget_index(b, T)
                    if idx is None:
                        continue
                    yb = T["ytr"][idx]
                    sp = set(row_keys(T["Xtr"][idx][yb == 1]))
                    sn = set(row_keys(T["Xtr"][idx][yb == 0]))
                    rows.append(dict(semantic_class=sem, target=tgt, seed=seed, budget=budget_label(b),
                                     n_test_pos=len(tp),
                                     test_pos_dup_in_train_pos=float(np.mean([k in sp for k in tp])),
                                     test_pos_dup_in_train_neg=float(np.mean([k in sn for k in tp]))))
            plog(f"dupdiag {'去重' if DEDUP else '原始'} {sem} seed {seed} 完成")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "dup_diag.csv", index=False)
    plog(f"dupdiag 完成 → {(OUT / 'dup_diag.csv').relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════
# compare：原始数据 vs 去重数据
# ══════════════════════════════════════════════════════════════════════════
def cmd_compare(a):
    O, D = ROOT / "results" / "rq4_budget", ROOT / "results" / "rq4_budget_dedup"
    for p in (O / "curve_by_config.csv", D / "curve_by_config.csv"):
        if not p.exists():
            sys.exit(f"✗ 缺少 {p.relative_to(ROOT)}，先分别运行 aggregate 与 aggregate --dedup")
    co, cd = pd.read_csv(O / "curve_by_config.csv"), pd.read_csv(D / "curve_by_config.csv")
    zo = pd.read_csv(O / "zero_shot_baseline.csv") if (O / "zero_shot_baseline.csv").exists() else None
    zd = pd.read_csv(D / "zero_shot_baseline.csv") if (D / "zero_shot_baseline.csv").exists() else None
    for c in (co, cd):
        c["budget"] = c["budget"].astype(str)
    order = [str(b) for b in POSTHOC_BUDGETS] + [budget_label(b) for b in BUDGETS]
    ph = [str(b) for b in POSTHOC_BUDGETS]
    L = ["原始数据 vs 精确去重数据：标注预算曲线对照（跨配置平均；带 * 的预算为事后探索性分析）", "=" * 72,
         "注意：两者的测试集不同（去重实验的测试集来自去重后的数据），差值是配置层面的描述性对照，不是同一测试样本上的配对比较。", ""]
    rows = []
    for ts, met in (("std", "f1"), ("nat", "pr_auc"), ("std", "pauc01"), ("nat", "f1")):
        for sem in sorted(co.semantic_class.unique()):
            for cond in ("Full-40", "Target-SHAP-10", "Source-SHAP-10", "Source-MI-10", "Random-10"):
                go = co[(co.semantic_class == sem) & (co.condition == cond) & (co.testset == ts)].groupby("budget")[met].mean()
                gd = cd[(cd.semantic_class == sem) & (cd.condition == cond) & (cd.testset == ts)].groupby("budget")[met].mean()
                for b in order:
                    if b in go.index or b in gd.index:
                        rows.append(dict(testset=ts, metric=met, semantic_class=sem, condition=cond, budget=b,
                                         original=go.get(b, np.nan), dedup=gd.get(b, np.nan)))
            for lab, z in (("original", zo), ("dedup", zd)):
                if z is not None:
                    v = z[(z.semantic_class == sem) & (z.testset == ts) & (z.condition == "SHAP-10-Direct")][met].mean()
                    rows.append(dict(testset=ts, metric=met, semantic_class=sem, condition="SHAP-10-Direct (zero-shot)",
                                     budget="—", **{lab: v}))
    cmp_ = pd.DataFrame(rows)
    cmp_ = cmp_.groupby(["testset", "metric", "semantic_class", "condition", "budget"], sort=False, as_index=False).first()
    cmp_["delta_dedup_minus_orig"] = cmp_["dedup"] - cmp_["original"]
    cmp_.round(4).to_csv(D / "compare_orig_vs_dedup.csv", index=False)
    for (ts, met, sem), g in cmp_.groupby(["testset", "metric", "semantic_class"], sort=False):
        L.append(f"{sem} — {met}（{'1:10' if ts == 'std' else 'prevalence-matched'}）")
        for cond, h in g.groupby("condition", sort=False):
            if cond.startswith("SHAP-10-Direct"):
                r = h.iloc[0]
                L.append(f"  {cond:<28} 原始 {r.original:.3f}  去重 {r.dedup:.3f}")
                continue
            cells = "  ".join(f"{b + ('*' if b in ph else '')}: {r.original:.3f}→{r.dedup:.3f}" for b, r in
                              h.set_index("budget").reindex([b for b in order if b in set(h.budget)]).iterrows())
            L.append(f"  {cond:<28} {cells}")
        L.append("")
    for lab, d_ in (("原始", O), ("去重", D)):
        f = d_ / "dup_diag.csv"
        if f.exists():
            dd = pd.read_csv(f)
            dd["budget"] = dd["budget"].astype(str)
            t = dd.groupby(["semantic_class", "budget"])["test_pos_dup_in_train_pos"].mean().unstack("budget")
            L += [f"{lab}数据：测试正例与该档训练正例完全相同的比例（跨目标、seed 平均）",
                  t[[b for b in order if b in t.columns]].round(3).to_string(), ""]
    (D / "compare_orig_vs_dedup.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    # 图：Full-40 与 Target-SHAP-10 的曲线，原始（实线）vs 去重（虚线），零样本水平线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
    classes = sorted(co.semantic_class.unique())
    fig, axes = plt.subplots(2, len(classes), figsize=(3.6 * len(classes), 5.6), squeeze=False)
    x = np.arange(len(order))
    for col_i, sem in enumerate(classes):
        for row_i, (ts, met, lab) in enumerate((("std", "f1", "F1 (1:10)"), ("nat", "pr_auc", "PR-AUC (prevalence-matched)"))):
            ax = axes[row_i][col_i]
            for cond, color in (("Full-40", "black"), ("Target-SHAP-10", "#1f77b4"), ("Source-SHAP-10", "#ff7f0e"), ("Random-10", "grey")):
                for data, ls, tag in ((co, "-", "orig"), (cd, "--", "dedup")):
                    y = data[(data.semantic_class == sem) & (data.condition == cond) & (data.testset == ts)].groupby("budget")[met].mean().reindex(order)
                    ax.plot(x, y.values, color=color, ls=ls, marker="o", ms=2.5, lw=1, label=f"{cond} ({tag})")
            for z, ls, tag in ((zo, "-.", "orig"), (zd, ":", "dedup")):
                if z is not None:
                    v = z[(z.semantic_class == sem) & (z.testset == ts) & (z.condition == "SHAP-10-Direct")][met].mean()
                    ax.axhline(v, color="purple", ls=ls, lw=0.9, label=f"zero-shot ({tag})")
            ax.axvspan(-0.5, len(ph) - 0.5, color="0.92", zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{b}*" if b in ph else b for b in order], fontsize=7)
            ax.set_title(f"{sem}", fontsize=8)
            ax.set_ylabel(lab)
            if row_i == 1:
                ax.set_xlabel("Target positive labels (* = post hoc)")
    h, l = axes[0][-1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(D / "fig_orig_vs_dedup.png")
    fig.savefig(D / "fig_orig_vs_dedup.pdf")
    plt.close(fig)
    print(f"→ {(D / 'compare_orig_vs_dedup.txt').relative_to(ROOT)}, fig_orig_vs_dedup.png")


def main():
    ap = argparse.ArgumentParser(description="Experiment A: target-label budget")
    sp = ap.add_subparsers(dest="cmd", required=True)
    cpu = os.cpu_count() or 8
    p = sp.add_parser("run")
    p.add_argument("--workers", type=int, default=max(1, (cpu - 2) // 4))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--classes", nargs="+", default=list(base.SEMANTIC_MAPPING))
    p.add_argument("--timeout-min", type=float, default=180)
    p.add_argument("--stall-min", type=float, default=30)
    p.add_argument("--posthoc", action="store_true", help="事后探索性分析：只跑 1/2/5 个正例三档")
    p = sp.add_parser("worker")
    p.add_argument("--cls", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--posthoc", action="store_true")
    sp.add_parser("status")
    sp.add_parser("aggregate")
    p = sp.add_parser("dupdiag")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--classes", nargs="+", default=list(base.SEMANTIC_MAPPING))
    sp.add_parser("compare")
    for sub in sp.choices.values():
        sub.add_argument("--dedup", action="store_true", help="在精确去重数据（rq3_dedup 缓存）上运行")
    a = ap.parse_args()
    set_mode(a.dedup)
    {"run": cmd_run, "worker": cmd_worker, "status": cmd_status, "aggregate": cmd_aggregate,
     "dupdiag": cmd_dupdiag, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
