#!/usr/bin/env python3
"""
code/deployment_cost.py — 实验 B（轻量）：部署成本与 NetFlow 字段需求
==========================================================================
目的（EAAI 稿 5.4 / 补充材料）：比较 Full-40 与目标自选 Top-10（Target-SHAP-10，嵌套选择，与 RQ2 相同口径）
在部署侧的计算成本，并给出 Top-10 所需的 NetFlow 字段采集能力等级。

测量内容（每个目标配置 × 每个 seed；配置 = DoS: UNSW/CIC/ToN/BoT，Reconnaissance: UNSW/ToN/BoT）：
  1. 训练时间（单线程，墙钟秒）；Target-10 另记一次性的特征选择时间（Full-40 训练 + TreeSHAP）
  2. 模型大小（booster UBJSON 字节数）、树节点总数
  3. 单流推理延迟（inplace_predict 1 行，单线程，重复 N 次取中位数与 p99，微秒）
  4. 批量吞吐（1:10 测试集整体 inplace_predict，单线程，重复 5 次取中位数，flows/s）
  5. 参考指标：1:10 测试集 F1 / PR-AUC / pAUC@1%FPR（核对与 RQ2 一致）
  6. 字段需求：每个特征按采集能力分 4 级（见 FIELD_LEVEL），统计 Top-10 需要的最高等级、各等级特征数、
     是否需要 DPI（应用层解析）；源特征集（feature_sets.json 中的 SHAP/MI/Gain-10）一并统计。

计时说明：所有计时均固定单线程（nthread=1），避免线程数差异；并行 worker 会相互争用内存带宽，
默认 --workers 1（约 30–60 分钟）。若只想快速出结果可设 WORKERS=4，计时绝对值会偏大但 Full-40/Top-10 比值基本不变。

用法（项目根目录，需要 RQ3 主实验的缓存 results/rq3_main/cache 与 feature_sets.json）：
  python code/deployment_cost.py run [--workers 1] [--n-seeds 5]
  python code/deployment_cost.py status
  python code/deployment_cost.py aggregate
  推荐：./run.sh cost（后台 + 防休眠，跑完自动汇总）
输出：results/deployment_cost/{parts/, raw.csv, summary_by_config.csv, fields_by_config.csv,
      field_levels.csv, summary.txt}
"""
import argparse, json, os, sys, time, traceback
os.environ.setdefault("A7_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # repository root (scripts live in code/)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq3_transfer as base

OUT = ROOT / "results" / "deployment_cost"
PARTS = OUT / "parts"
PROGRESS = OUT / "progress.log"
CONFIGS = [("DoS", t) for t in ["UNSW", "CIC", "ToN", "BoT"]] + [("Reconnaissance", t) for t in ["UNSW", "ToN", "BoT"]]
N_SINGLE = 2000      # 单流延迟重复次数
N_BATCH = 5          # 批量吞吐重复次数

# ── NetFlow 字段采集能力等级（依据 nProbe / NetFlow v9 / IPFIX 字段含义，由作者归类；可按需修改）──
#   L1 = 基本流计数器：任何 NetFlow/IPFIX 导出器（含路由器自带）均可提供（双向流需 biflow 聚合）
#   L1d = 由 L1 计数器与持续时间可推导的速率特征
#   L2 = 逐包头部统计：需要软件探针（nProbe/YAF 等）维护逐包状态（TTL、包长极值与直方图、窗口、重传、分向时长/标志）
#   L3 = 应用层解析（DPI）：需要载荷检查（nDPI 协议识别、DNS/FTP 字段）
FIELD_LEVEL = {
    "L4_SRC_PORT": "L1", "L4_DST_PORT": "L1", "PROTOCOL": "L1", "IN_BYTES": "L1", "IN_PKTS": "L1",
    "OUT_BYTES": "L1", "OUT_PKTS": "L1", "TCP_FLAGS": "L1", "FLOW_DURATION_MILLISECONDS": "L1",
    "SRC_TO_DST_SECOND_BYTES": "L1d", "DST_TO_SRC_SECOND_BYTES": "L1d",
    "SRC_TO_DST_AVG_THROUGHPUT": "L1d", "DST_TO_SRC_AVG_THROUGHPUT": "L1d",
    "CLIENT_TCP_FLAGS": "L2", "SERVER_TCP_FLAGS": "L2", "DURATION_IN": "L2", "DURATION_OUT": "L2",
    "MIN_TTL": "L2", "MAX_TTL": "L2", "LONGEST_FLOW_PKT": "L2", "SHORTEST_FLOW_PKT": "L2", "MIN_IP_PKT_LEN": "L2",
    "RETRANSMITTED_IN_BYTES": "L2", "RETRANSMITTED_IN_PKTS": "L2", "RETRANSMITTED_OUT_BYTES": "L2", "RETRANSMITTED_OUT_PKTS": "L2",
    "NUM_PKTS_UP_TO_128_BYTES": "L2", "NUM_PKTS_128_TO_256_BYTES": "L2", "NUM_PKTS_256_TO_512_BYTES": "L2",
    "NUM_PKTS_512_TO_1024_BYTES": "L2", "NUM_PKTS_1024_TO_1514_BYTES": "L2",
    "TCP_WIN_MAX_IN": "L2", "TCP_WIN_MAX_OUT": "L2", "ICMP_TYPE": "L2", "ICMP_IPV4_TYPE": "L2",
    "L7_PROTO": "L3", "DNS_QUERY_ID": "L3", "DNS_QUERY_TYPE": "L3", "DNS_TTL_ANSWER": "L3", "FTP_COMMAND_RET_CODE": "L3",
}
LEVELS = ["L1", "L1d", "L2", "L3"]
LEVEL_RANK = {"L1": 0, "L1d": 0, "L2": 1, "L3": 2}


def plog(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def model_stats(model):
    bst = model.get_booster()
    raw = bst.save_raw(raw_format="ubj")
    nodes = sum(d.count("\n") for d in bst.get_dump())  # 每行一个节点
    return len(raw), nodes


def bench(model, Xte):
    bst = model.get_booster()
    bst.set_param({"nthread": 1})
    Xte = np.ascontiguousarray(Xte, dtype=np.float32)
    one = Xte[:1]
    for _ in range(50):                              # 预热
        bst.inplace_predict(one)
    lat = np.empty(N_SINGLE)
    for i in range(N_SINGLE):
        row = Xte[i % len(Xte): i % len(Xte) + 1]
        t0 = time.perf_counter(); bst.inplace_predict(row); lat[i] = time.perf_counter() - t0
    bt = []
    for _ in range(N_BATCH):
        t0 = time.perf_counter(); bst.inplace_predict(Xte); bt.append(time.perf_counter() - t0)
    return dict(lat_med_us=float(np.median(lat) * 1e6), lat_p99_us=float(np.percentile(lat, 99) * 1e6),
                batch_n=len(Xte), batch_s=float(np.median(bt)), throughput_fps=float(len(Xte) / np.median(bt)))


def worker(sem, tgt, seed):
    import xgboost as xgb
    tag = f"{sem}__{tgt}__seed{seed}"
    X, codes, meta = base.load_cache(tgt)
    fc = meta["feature_cols"]
    pos_codes = [meta["labels"].index(l) for l in base.SEMANTIC_MAPPING[sem][tgt] if l in meta["labels"]]
    tr, te, ytr, yte = base.split_indices(codes, pos_codes, seed)
    Xall = base.clean_subset(X[np.concatenate([tr, te])])
    Xtr, Xte = Xall[:len(tr)], Xall[len(tr):]
    rows = []

    def record(cond, cols, model, fit_s, extra=None):
        size, nodes = model_stats(model)
        proba = model.predict_proba(Xte[:, cols])[:, 1]
        met = base.metrics(yte, proba)
        r = dict(semantic_class=sem, target=tgt, seed=seed, condition=cond, n_features=len(cols),
                 n_train=len(ytr), fit_s=fit_s, model_bytes=size, n_nodes=nodes,
                 f1=met["f1"], pr_auc=met["pr_auc"], pauc01=met["pauc01"],
                 features=";".join(fc[i] for i in cols), **bench(model, Xte[:, cols]))
        if extra:
            r.update(extra)
        rows.append(r)

    allc = list(range(len(fc)))
    t0 = time.perf_counter(); m40 = base.fit_xgb(Xtr, ytr, seed, 1); fit40 = time.perf_counter() - t0
    record("Full-40", allc, m40, fit40)
    # 嵌套 Target-10（与 rq3_transfer 完全相同）：Full-40 模型在训练正例上的 TreeSHAP
    t0 = time.perf_counter()
    contrib = m40.get_booster().predict(xgb.DMatrix(Xtr[ytr == 1]), pred_contribs=True)[:, :-1]
    t10 = [int(i) for i in np.argsort(np.abs(contrib).mean(axis=0))[::-1][:base.TOP_K]]
    shap_s = time.perf_counter() - t0
    t0 = time.perf_counter(); m10 = base.fit_xgb(Xtr[:, t10], ytr, seed, 1); fit10 = time.perf_counter() - t0
    record("Target-SHAP-10", t10, m10, fit10, dict(select_s=fit40 + shap_s, shap_s=shap_s))
    PARTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PARTS / f"{tag}.csv", index=False)
    return tag


def _run_one(t):
    try:
        t0 = time.time(); tag = worker(*t); return tag, time.time() - t0, None
    except Exception:
        return f"{t[0]}__{t[1]}__seed{t[2]}", 0, traceback.format_exc()


def cmd_run(a):
    miss = [d for d in base.DS_ORDER if not (base.CACHE / f"{d}_X.npy").exists()]
    if miss:
        sys.exit(f"✗ 缓存缺失 {miss}（需先跑 rq3_transfer prep）")
    seeds = list(range(base.SEED_BASE, base.SEED_BASE + a.n_seeds))
    tasks = [(s, t, sd) for sd in seeds for (s, t) in CONFIGS if not (PARTS / f"{s}__{t}__seed{sd}.csv").exists()]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_config.json").write_text(json.dumps(dict(vars(a), n_single=N_SINGLE, n_batch=N_BATCH,
                                                         xgboost=__import__("xgboost").__version__), indent=1))
    plog(f"run 开始：待完成 {len(tasks)}/{len(CONFIGS) * len(seeds)}，workers={a.workers}（计时均为单线程）")
    if a.workers <= 1:
        for t in tasks:
            tag, dt, err = _run_one(t)
            plog(f"{'✓' if not err else '✗'} {tag}（{dt / 60:.1f} 分钟）" + (f"\n{err}" if err else ""))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(a.workers) as ex:
            for fu in as_completed([ex.submit(_run_one, t) for t in tasks]):
                tag, dt, err = fu.result()
                plog(f"{'✓' if not err else '✗'} {tag}（{dt / 60:.1f} 分钟）" + (f"\n{err}" if err else ""))
    plog("run 结束")


def cmd_status(a):
    n = len(list(PARTS.glob("*.csv"))) if PARTS.exists() else 0
    print(f"已完成 {n}/{len(CONFIGS) * a.n_seeds}")
    if PROGRESS.exists():
        print("".join(PROGRESS.read_text(encoding="utf-8").splitlines(True)[-5:]))


def level_profile(feats):
    lv = [FIELD_LEVEL[f] for f in feats]
    d = {f"n_{l}": lv.count(l) for l in LEVELS}
    top = max(lv, key=lambda l: LEVEL_RANK[l])
    d["max_level"] = {"L1": "L1", "L1d": "L1", "L2": "L2", "L3": "L3"}[top]
    d["needs_dpi"] = int("L3" in lv)
    d["needs_probe"] = int(any(l in ("L2", "L3") for l in lv))
    return d


def cmd_aggregate(a):
    files = sorted(PARTS.glob("*.csv"))
    if not files:
        sys.exit("✗ 没有结果，先 run")
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw.to_csv(OUT / "raw.csv", index=False)
    fc = base.load_cache("UNSW")[2]["feature_cols"]
    assert set(fc) == set(FIELD_LEVEL), set(fc) ^ set(FIELD_LEVEL)
    pd.DataFrame([dict(feature=f, level=FIELD_LEVEL[f]) for f in fc]).to_csv(OUT / "field_levels.csv", index=False)

    num = ["fit_s", "model_bytes", "n_nodes", "lat_med_us", "lat_p99_us", "throughput_fps", "f1", "pr_auc", "pauc01"]
    g = raw.groupby(["semantic_class", "target", "condition"])[num].mean().reset_index()
    sel = raw[raw.condition == "Target-SHAP-10"].groupby(["semantic_class", "target"])[["select_s"]].mean()
    p = g.pivot_table(index=["semantic_class", "target"], columns="condition", values=num)
    s = pd.DataFrame(index=p.index)
    for m in num:
        s[f"{m}_full40"] = p[(m, "Full-40")]; s[f"{m}_top10"] = p[(m, "Target-SHAP-10")]
    s["fit_speedup"] = s.fit_s_full40 / s.fit_s_top10
    s["size_ratio"] = s.model_bytes_top10 / s.model_bytes_full40
    s["lat_ratio"] = s.lat_med_us_top10 / s.lat_med_us_full40
    s["thr_ratio"] = s.throughput_fps_top10 / s.throughput_fps_full40
    s["select_s_once"] = sel["select_s"]
    s["n_seeds"] = raw[raw.condition == "Full-40"].groupby(["semantic_class", "target"]).size()
    s = s.reset_index()
    s.round(4).to_csv(OUT / "summary_by_config.csv", index=False)

    # 字段需求：Target-SHAP-10（逐 seed）+ 源特征集（feature_sets.json）+ Full-40
    frows = []
    for (sem, tgt), d in raw[raw.condition == "Target-SHAP-10"].groupby(["semantic_class", "target"]):
        profs = pd.DataFrame([level_profile(f.split(";")) for f in d.features])
        frows.append(dict(semantic_class=sem, dataset=tgt, feature_set="Target-SHAP-10 (per seed, mean)",
                          **profs.drop(columns="max_level").mean().round(2).to_dict(),
                          max_level=profs.max_level.mode().iloc[0]))
    fs_path = base.OUT / "feature_sets.json"
    if fs_path.exists():
        fsets = json.loads(fs_path.read_text())
        for sem, dd in fsets.items():
            for ds, sets in dd.items():
                for name in ("SHAP-10", "MI-10", "Gain-10", "SHAP-10-NoPort"):
                    if name in sets:
                        frows.append(dict(semantic_class=sem, dataset=ds, feature_set=f"Source {name}", **level_profile(sets[name])))
    frows.append(dict(semantic_class="—", dataset="—", feature_set="Full-40", **level_profile(fc)))
    F = pd.DataFrame(frows)
    F.to_csv(OUT / "fields_by_config.csv", index=False)

    # 文本摘要
    L = []
    L.append("实验 B：部署成本（单线程计时；均值 over seeds）\n")
    L.append(f"seeds：{sorted(int(v) for v in raw.seed.unique())}；xgboost {json.loads((OUT / 'run_config.json').read_text()).get('xgboost') if (OUT / 'run_config.json').exists() else ''}\n")
    cols = ["semantic_class", "target", "n_seeds", "fit_s_full40", "fit_s_top10", "fit_speedup", "select_s_once",
            "model_bytes_full40", "model_bytes_top10", "lat_med_us_full40", "lat_med_us_top10",
            "throughput_fps_full40", "throughput_fps_top10", "thr_ratio", "f1_full40", "f1_top10"]
    L.append(s[cols].round(3).to_string(index=False) + "\n")
    L.append("\n跨 7 个配置的中位数：")
    for c in ["fit_speedup", "size_ratio", "lat_ratio", "thr_ratio"]:
        L.append(f"  {c}: 中位数 {s[c].median():.3f}（范围 {s[c].min():.3f}–{s[c].max():.3f}）")
    L.append(f"  F1 差（Full-40 − Top-10）：最大 {(s.f1_full40 - s.f1_top10).max():.4f}")
    L.append("\n字段需求（L1 基本计数器 / L1d 可由计数器推导的速率 / L2 逐包头部统计 / L3 应用层解析）：")
    L.append(F.to_string(index=False))
    tt = F[F.feature_set.str.startswith("Target")]
    L.append(f"\nTarget-SHAP-10 需要 DPI（L3）的 seed 比例：{tt[['semantic_class', 'dataset', 'needs_dpi']].to_string(index=False)}")
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--workers", type=int, default=1); r.add_argument("--n-seeds", type=int, default=5)
    st = sp.add_parser("status"); st.add_argument("--n-seeds", type=int, default=5)
    sp.add_parser("aggregate")
    a = ap.parse_args()
    {"run": cmd_run, "status": cmd_status, "aggregate": cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    main()
