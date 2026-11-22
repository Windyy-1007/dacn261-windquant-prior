"""Analysis and publication figures from recorded measurements only."""
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .io import atomic_json, csv_write, Results
from .quantization import PROJECTIONS


def frame(out, experiment):
    return pd.DataFrame(Results(out, experiment).export())


def rho(x, y):
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def cluster_interval(data, proxy, cluster, samples, seed):
    """Resample whole layers (matrices) or whole modules (chunks)."""
    groups = [g for _, g in data.groupby(cluster, sort=True)]
    if len(groups) < 3:
        return None, 0
    arrays = [g[[proxy, "delta_nll"]].to_numpy() for g in groups]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        sample = np.concatenate([arrays[i] for i in rng.integers(len(groups), size=len(groups))])
        value = rho(sample[:, 0], sample[:, 1])
        if value is not None:
            values.append(value)
    return np.quantile(values, [.025, .975]).tolist() if values else None, len(values)


def correlations(out, config):
    out = Path(out)
    rows, rankings = [], []
    for experiment, granularity in [("e2", "matrix"), ("e3", "chunk")]:
        df = frame(out, experiment)
        if df.empty:
            continue
        if df["protocol_id"].nunique() != 1:
            raise ValueError("Cannot pool different experimental protocols")
        cluster = "layer_idx" if granularity == "matrix" else "module_name"
        for (model, bits), by_bits in df.groupby(["model", "bit_width"]):
            for group in ("all", "attention", "mlp"):
                is_attention = by_bits["projection_type"].isin(PROJECTIONS[:4])
                subset = by_bits if group == "all" else by_bits[is_attention if group == "attention" else ~is_attention]
                if subset.empty:
                    continue
                cis, valid, coeff = {}, {}, {}
                for name, proxy in [("weight", "normalized_weight_mse"), ("output", "output_mse")]:
                    if not np.isfinite(subset[proxy]).all() or not np.isfinite(subset["delta_nll"]).all():
                        raise ValueError("Missing or non-finite reconstruction/sensitivity measurements")
                    coeff[name] = rho(subset[proxy], subset["delta_nll"])
                    cis[name], valid[name] = cluster_interval(subset, proxy, cluster,
                        config["experiments"]["e4"]["bootstrap_samples"], config["seed"])
                rows.append({"model": model, "granularity": granularity, "module_group": group,
                             "bit_width": int(bits), "n_units": len(subset),
                             "spearman_weight_vs_nll": coeff["weight"], "spearman_output_vs_nll": coeff["output"],
                             "confidence_interval": cis, "bootstrap_valid_samples": valid,
                             "cluster_unit": cluster, "n_clusters": subset[cluster].nunique(),
                             "bootstrap_samples": config["experiments"]["e4"]["bootstrap_samples"],
                             "seed": config["seed"]})
            ranked = by_bits.copy()
            for name, column in [("weight", "normalized_weight_mse"), ("output", "output_mse"), ("nll", "delta_nll")]:
                ranked[f"rank_{name}"] = ranked[column].rank(method="average", ascending=False)
            ranked["weight_rank_disagreement"] = (ranked["rank_weight"] - ranked["rank_nll"]).abs()
            ranked["output_rank_disagreement"] = (ranked["rank_output"] - ranked["rank_nll"]).abs()
            columns = ["model", "module_name", "layer_idx", "bit_width", "col_start", "col_end",
                       "delta_nll", "normalized_weight_mse", "output_mse", "rank_weight", "rank_output", "rank_nll",
                       "weight_rank_disagreement", "output_rank_disagreement"]
            ranked["granularity"] = granularity
            rankings.extend(ranked[columns + ["granularity"]].to_dict("records"))
    csv_write(out / "e4.csv", rows)
    atomic_json(out / "e4.json", rows)
    csv_write(out / "e4_rank_disagreements.csv", rankings)
    return pd.DataFrame(rows)


def style():
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
                         "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.dpi": 240, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, folder, name, data):
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{extension}", bbox_inches="tight")
    plt.close(fig)
    data.to_csv(folder / f"{name}.csv", index=False)


def heatmap(data, ax, title, xlabel="Projection", ylabel="Transformer layer"):
    array = data.to_numpy(dtype=float)
    finite = np.abs(array[np.isfinite(array)])
    maximum = float(finite.max()) if len(finite) else 1.
    maximum = maximum or 1.
    im = ax.imshow(array, aspect="auto", cmap="RdBu_r", vmin=-maximum, vmax=maximum)
    ax.set_xticks(range(len(data.columns)), data.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(data.index)), data.index)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return im


def figures(out):
    out = Path(out)
    folder = out / "figures"
    style()
    e1, e2, e3 = (frame(out, e) for e in ("e1", "e2", "e3"))
    if not e1.empty:
        q = e1[e1.precision != "FP"].sort_values("bit_width")
        fp = e1[e1.precision == "FP"]
        for column, ylabel, name in [("validation_ppl", "Validation perplexity", "E1-A_precision_quality"),
                                      ("normalized_weight_mse", "Normalized weight reconstruction MSE", "E1-B_precision_error")]:
            fig, ax = plt.subplots(figsize=(5.6, 3.8))
            ax.plot(q.bit_width, q[column], "o-", label="Group-wise affine RTN")
            if len(fp):
                ax.axhline(fp.iloc[0][column], color="gray", linestyle="--", label="Original precision reference")
            ax.set(xlabel="Weight precision (bits)", ylabel=ylabel, xticks=[2, 3, 4, 8])
            if column == "validation_ppl":
                ax.set_yscale("log")
                ax.set_ylabel("Validation perplexity (log scale)")
            else:
                ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
            ax.legend()
            save(fig, folder, name, e1.drop(columns=[c for c in e1 if c.endswith("evaluation")]))
    if not e2.empty:
        extremes = []
        e2.groupby(["bit_width", "projection_type"]).delta_nll.agg(
            ["count", "mean", "std", "median", "min", "max"]).reset_index().to_csv(out/"e2_group_summary.csv", index=False)
        for bits, df in e2.groupby("bit_width"):
            pivot = df.pivot(index="layer_idx", columns="projection_type", values="delta_nll").reindex(columns=PROJECTIONS)
            fig, ax = plt.subplots(figsize=(7, 5.5))
            im = heatmap(pivot, ax, f"Single-matrix sensitivity · W{bits}")
            fig.colorbar(im, ax=ax, label="Δ validation NLL (nats/token)")
            save(fig, folder, f"E2-A_heatmap_W{bits}", pivot.reset_index())
            fig, ax = plt.subplots(figsize=(7, 4))
            groups = [df[df.projection_type == p].delta_nll.to_numpy() for p in PROJECTIONS]
            ax.boxplot(groups, tick_labels=PROJECTIONS)
            ax.tick_params(axis="x", rotation=30)
            ax.set(xlabel="Projection type", ylabel="Δ validation NLL (nats/token)", title=f"Matrix sensitivity distribution · W{bits}")
            ax.axhline(0, color="gray", lw=.7)
            save(fig, folder, f"E2-B_distribution_W{bits}", df[["layer_idx", "projection_type", "delta_nll"]])
            fig, ax = plt.subplots(figsize=(7, 4))
            for p, group in df.groupby("projection_type"):
                group = group.sort_values("layer_idx")
                ax.plot(group.layer_idx, group.delta_nll, ".-", label=p)
            ax.set(xlabel="Transformer layer", ylabel="Δ validation NLL (nats/token)", title=f"Sensitivity by depth · W{bits}")
            ax.axhline(0, color="gray", lw=.7)
            ax.legend(ncol=3)
            save(fig, folder, f"E2-C_depth_W{bits}", df[["layer_idx", "projection_type", "delta_nll"]])
            for label, subset in [("most_sensitive", df.nlargest(10, "delta_nll")), ("least_sensitive", df.nsmallest(10, "delta_nll"))]:
                for row in subset[["module_name", "bit_width", "delta_nll", "normalized_weight_mse", "output_mse"]].to_dict("records"):
                    extremes.append({"ranking": label, **row})
        csv_write(out / "e2_extremes.csv", extremes)
    if not e3.empty:
        summaries = []
        for (bits, module), df in e3.groupby(["bit_width", "module_name"]):
            x = df.delta_nll
            summaries.append({"module_name": module, "bit_width": int(bits), "n_chunks": len(x), "mean": x.mean(),
                              "std": x.std(ddof=1), "median": x.median(), "minimum": x.min(), "maximum": x.max(),
                              "iqr": x.quantile(.75)-x.quantile(.25), "range": x.max()-x.min()})
        csv_write(out / "e3_summary.csv", summaries)
        for bits, df in e3.groupby("bit_width"):
            module_layers = df.drop_duplicates("module_name").set_index("module_name").layer_idx.to_dict()
            projection_order = {"q_proj": 0, "gate_proj": 1, "down_proj": 2}
            modules = sorted(df.module_name.unique(), key=lambda name:
                             (module_layers[name], projection_order.get(name.split(".")[-1], 3), name))
            chunk_width = int((df.col_end-df.col_start).max())
            fig, axes = plt.subplots(int(np.ceil(len(modules)/3)), 3,
                                     figsize=(12, 2.8*int(np.ceil(len(modules)/3))), squeeze=False)
            for ax, module in zip(axes.flat, modules):
                subset = df[df.module_name == module].sort_values("chunk_idx")
                ax.plot(subset.chunk_idx, subset.delta_nll, ".-")
                ax.axhline(0, color="gray", lw=.6)
                label = module.replace("model.layers.", "Layer ").replace("self_attn.", "").replace("mlp.", "")
                ax.set(title=label, xlabel=f"{chunk_width}-column chunk index", ylabel="Δ NLL")
            for ax in list(axes.flat)[len(modules):]:
                ax.set_visible(False)
            save(fig, folder, f"E3-A_chunks_W{bits}", df[["module_name", "chunk_idx", "delta_nll"]])
            labels = [m.replace("model.layers.", "L").replace("self_attn.", "").replace("mlp.", "") for m in modules]
            fig, ax = plt.subplots(figsize=(9, 4.6))
            ax.boxplot([df[df.module_name == m].delta_nll for m in modules], tick_labels=labels)
            ax.tick_params(axis="x", rotation=60)
            ax.set(ylabel="Δ validation NLL (nats/token)", title=f"Within-matrix sensitivity · W{bits}")
            ax.axhline(0, color="gray", lw=.7)
            save(fig, folder, f"E3-B_distribution_W{bits}", pd.DataFrame(summaries))
            pivot = df.pivot(index="module_name", columns="chunk_idx", values="delta_nll").reindex(index=modules)
            med = pivot.median(axis=1)
            iqr = pivot.quantile(.75, axis=1) - pivot.quantile(.25, axis=1)
            normalized = pivot.sub(med, axis=0).div(iqr.where(iqr > 1e-12), axis=0)
            normalized.index = labels
            fig, ax = plt.subplots(figsize=(10, 5))
            im = heatmap(normalized, ax, f"Within-matrix (ΔNLL − median) / IQR · W{bits}", "Chunk index", "Matrix")
            fig.colorbar(im, ax=ax, label="Robust normalized sensitivity; missing/zero IQR masked")
            save(fig, folder, f"E3-C_normalized_heatmap_W{bits}", normalized.reset_index())
    for proxy, title, name in [("normalized_weight_mse", "Normalized weight MSE", "E4-A_weight_vs_nll"),
                                ("output_mse", "Normalized output reconstruction error", "E4-B_output_vs_nll")]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        data = []
        for ax, (label, df) in zip(axes, [("Matrix", e2), ("Chunk", e3)]):
            if df.empty:
                continue
            for bits, group in df.groupby("bit_width"):
                ax.scatter(group[proxy], group.delta_nll, s=14, alpha=.7, label=f"W{bits}")
            ax.axhline(0, color="gray", lw=.6)
            ax.set(xlabel=title, ylabel="Δ validation NLL (nats/token)", title=f"{label}-level measurements")
            ax.set_xlim(left=0)
            ax.legend()
            data.append(df[["model", "module_name", "bit_width", "col_start", "col_end", proxy, "delta_nll"]].assign(granularity=label))
        if data:
            save(fig, folder, name, pd.concat(data))
        else:
            plt.close(fig)
    path = out / "e4_rank_disagreements.csv"
    if path.exists():
        ranks = pd.read_csv(path)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, granularity in zip(axes, ["matrix", "chunk"]):
            subset = ranks[ranks.granularity == granularity]
            for bits, group in subset.groupby("bit_width"):
                for proxy, marker in [("weight", "o"), ("output", "x")]:
                    ax.scatter(group[f"rank_{proxy}"], group.rank_nll, s=14, marker=marker, alpha=.6, label=f"{proxy} W{bits}")
            if len(subset):
                maximum = subset.rank_nll.max()
                ax.plot([1, maximum], [1, maximum], "--", color="gray", lw=.8)
            ax.set(xlabel="Local error rank (1 = highest)", ylabel="Measured ΔNLL rank", title=granularity.capitalize())
            ax.legend()
        save(fig, folder, "E4-C_ranking_agreement", ranks)
    extension_figures(out, folder)
    return sorted(folder.glob("*.png"))


def extension_figures(out, folder):
    e5, e6 = frame(out, "e5"), frame(out, "e6")
    if not e5.empty:
        fig, ax = plt.subplots(figsize=(5.8, 4.5))
        for category, df in e5.groupby("pair_type"):
            ax.scatter(df.predicted_additive_delta, df.observed_joint_delta, label=category, s=25)
        lo = min(e5.predicted_additive_delta.min(), e5.observed_joint_delta.min(), 0)
        hi = max(e5.predicted_additive_delta.max(), e5.observed_joint_delta.max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray", label="Additive prediction")
        ax.set(xlabel="Predicted additive ΔNLL", ylabel="Observed joint ΔNLL", title="Sampled W2 chunk pairs")
        ax.legend()
        save(fig, folder, "E5-A_predicted_vs_observed", e5[["region_i", "region_j", "pair_type", "predicted_additive_delta", "observed_joint_delta"]])
        fig, ax = plt.subplots(figsize=(6.5, 4))
        labels = sorted(e5.pair_type.unique())
        ax.boxplot([e5[e5.pair_type == p].interaction_residual for p in labels], tick_labels=labels)
        ax.axhline(0, color="gray", lw=.8)
        ax.set(ylabel="Interaction residual (nats/token)", xlabel="Pair category")
        save(fig, folder, "E5-B_residuals_by_category", e5[["pair_type", "interaction_residual", "interaction_ci"]])
        fig, ax = plt.subplots(figsize=(5.8, 4))
        ax.hist(e5.interaction_residual, bins=min(12, len(e5)))
        ax.axvline(0, color="black", linestyle="--", lw=.8)
        ax.set(xlabel="Interaction residual (nats/token)", ylabel="Number of sampled pairs")
        save(fig, folder, "E5-C_interaction_distribution", e5[["region_i", "region_j", "interaction_residual"]])
    if not e6.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        for strategy, df in e6.groupby("strategy"):
            ax.scatter(df.realized_bpw, df.validation_nll, label=strategy)
        ax.set(xlabel="Realized payload precision (bits/quantizable weight)", ylabel="Validation NLL (nats/token)")
        ax.set_ylim(bottom=0)
        ax.legend()
        save(fig, folder, "E6-A_quality_precision", e6[["strategy", "target_bpw", "realized_bpw", "validation_nll", "test_nll"]])
        for _, row in e6[e6.strategy == "sensitivity_greedy"].iterrows():
            entries = []
            manifest = json.loads((Path(out)/"manifest.json").read_text())
            for target in manifest["targets"]:
                entries.append({"layer_idx": target["layer"], "projection": target["projection"], "bits": row.allocation[target["name"]]})
            pivot = pd.DataFrame(entries).pivot(index="layer_idx", columns="projection", values="bits").reindex(columns=PROJECTIONS)
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(pivot, aspect="auto", vmin=2, vmax=4, cmap="viridis")
            ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)), pivot.index)
            ax.set(xlabel="Projection", ylabel="Transformer layer", title=f"Greedy allocation; realized {row.realized_bpw:.4f} bpw")
            fig.colorbar(im, ax=ax, ticks=[2, 3, 4], label="Weight bits")
            save(fig, folder, f"E6-B_allocation_{row.target_bpw}", pivot.reset_index())
        paired = e6[e6.strategy.isin(["sensitivity_greedy", "random_matched"])]
        fig, ax = plt.subplots(figsize=(6, 4))
        for strategy, df in paired.groupby("strategy"):
            ax.scatter(df.realized_bpw, df.test_nll, label=strategy)
        ax.set(xlabel="Matched realized payload precision (bits/weight)", ylabel="Held-out test NLL (nats/token)")
        ax.set_ylim(bottom=0)
        ax.legend()
        save(fig, folder, "E6-C_matched_costs", paired[["strategy", "target_bpw", "realized_bpw", "test_nll", "random_seed"]])


def markdown_table(df, columns):
    if df.empty:
        return "Chưa có phép đo.\n"
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value)
    return "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"]*len(columns)) + " |\n" + "\n".join(
        "| " + " | ".join(fmt(row[c]) for c in columns) + " |" for _, row in df.iterrows()) + "\n"


def acceptance(out, config):
    out = Path(out)
    path = out / "manifest.json"
    if not path.exists():
        return {"gate_b": False, "reason": "No model manifest"}
    manifest = json.loads(path.read_text())
    e1, e2, e3 = (frame(out, e) for e in ("e1", "e2", "e3"))
    e1_ok = not e1.empty and set(e1.precision) >= {"FP", "W8", "W4", "W3", "W2"}
    expected = {t["name"] for t in manifest["targets"]}
    e2_ok = not e2.empty and set(e2[e2.bit_width == 2].module_name) == expected
    selection_path = out / "e3_selection.json"
    selected = json.loads(selection_path.read_text())["modules"] if selection_path.exists() else []
    chunk_size = config["experiments"]["e3"]["chunk_size"]
    expected_chunks = {(t["name"], start, min(start+chunk_size, t["shape"][1]))
                       for t in manifest["targets"] if t["name"] in selected
                       for start in range(0, t["shape"][1], chunk_size)}
    actual_chunks = set() if e3.empty else set(e3[e3.bit_width == 2][["module_name", "col_start", "col_end"]].itertuples(index=False, name=None))
    e3_ok = len(selected) >= 6 and expected_chunks == actual_chunks
    e4_path = out / "e4.json"
    e4 = json.loads(e4_path.read_text()) if e4_path.exists() else []
    e4_ok = {(r["granularity"], r["module_group"]) for r in e4} >= {(g, m) for g in ("matrix", "chunk") for m in ("all", "attention", "mlp")}
    reconstruction_ok = all(not df.empty and np.isfinite(df[["normalized_weight_mse", "output_mse", "delta_nll"]]).all().all()
                            for df in (e2, e3))
    required_figures = ["E1-A_precision_quality", "E1-B_precision_error", "E2-A_heatmap_W2", "E2-B_distribution_W2",
                        "E2-C_depth_W2", "E3-A_chunks_W2", "E3-B_distribution_W2", "E3-C_normalized_heatmap_W2",
                        "E4-A_weight_vs_nll", "E4-B_output_vs_nll", "E4-C_ranking_agreement"]
    figures_ok = all((out/"figures"/f"{name}.{suffix}").exists() for name in required_figures for suffix in ("pdf", "png", "csv"))
    gate_a = json.loads((out/"gate_a.json").read_text()).get("passed", False) if (out/"gate_a.json").exists() else False
    result = {"gate_a": gate_a, "e1_complete": bool(e1_ok), "e2_complete": bool(e2_ok),
              "e3_complete": bool(e3_ok), "e4_complete": bool(e4_ok),
              "e2_expected_w2_matrices": len(expected), "e2_measured_w2_matrices": 0 if e2.empty else int((e2.bit_width==2).sum()),
              "e3_expected_w2_chunks": len(expected_chunks), "e3_measured_w2_chunks": len(actual_chunks),
              "reconstruction_metrics_complete": bool(reconstruction_ok), "required_figures_complete": figures_ok,
              "gate_b": bool(gate_a and e1_ok and e2_ok and e3_ok and e4_ok and reconstruction_ok and figures_ok)}
    pairs = frame(out, "e5")
    result["e5_measured_pairs"] = len(pairs)
    result["gate_c"] = len(pairs) >= 20 or len(frame(out, "e6")) > 0
    atomic_json(out / "acceptance.json", result)
    return result


def artifact_index(out):
    """Record genuine persistence timestamps, not invented experiment start times."""
    logs = []
    for path in sorted((out/"trials").glob("*/*.json"), key=lambda p: p.stat().st_mtime_ns):
        row = json.loads(path.read_text())
        logs.append({"persisted_at_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                     "timestamp_source": "trial file modification time", "experiment_id": row["experiment_id"],
                     "trial_key": row["trial_key"], "runtime": row.get("runtime"), "delta_nll": row.get("delta_nll"),
                     "artifact": str(path.relative_to(out))})
    (out/"measurement_log.jsonl").write_text("".join(json.dumps(row)+"\n" for row in logs))
    hashes = {}
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.suffix in (".pt", ".tmp", ".lock") or path.name == "artifact_index.json":
            continue
        hashes[str(path.relative_to(out))] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(out/"artifact_index.json", {"sha256": hashes})


def report(out, config):
    out = Path(out)
    e4 = correlations(out, config)
    plots = figures(out)
    status = acceptance(out, config)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    evaluated_counts = {name: data["evaluated_tokens"] for name, data in
                        manifest.get("datasets", {}).get("evaluation", {}).items()}
    quantizable_count = sum(int(np.prod(t["shape"])) for t in manifest.get("targets", []))
    e1, e2, e3 = (frame(out, e) for e in ("e1", "e2", "e3"))
    text = ["# Báo cáo thực nghiệm DACN — lượng tử hóa trọng số RTN\n",
            "## 1. Mục tiêu\nĐo ảnh hưởng của độ chính xác, độ nhạy ma trận/chunk và khả năng dự báo suy giảm NLL từ lỗi tái tạo cục bộ.\n",
            f"## 2. Phương pháp\nRTN affine theo nhóm {config['group_size']} cột đầu vào mỗi hàng; W2/W3/W4/W8. "
            "Giữ embedding, output head, normalization và activation ở độ chính xác nguồn. "
            f"Không huấn luyện, không RL/PPO. Chunk phân tích gồm {config['experiments']['e3']['chunk_size']} cột, giữ nguyên biên nhóm và giá trị RTN. "
            "Mỗi phép đo E2/E3 chỉ thay một vùng, khôi phục chính xác rồi so sánh logits tham chiếu.\n",
            "NLL là tổng cross-entropy chia số token đích thực tế; PPL = exp(NLL). "
            f"Context {config['evaluation']['context']}, stride {config['evaluation']['stride']}; overlap chỉ làm ngữ cảnh, "
            "không tính lặp loss. ΔNLL dùng cùng token validation. Khoảng tin cậy ΔNLL dùng bootstrap khối 8 cửa sổ; "
            "tương quan dùng bootstrap theo layer cho E2 và theo ma trận cho E3.\n",
            "## 3. Cấu hình mô hình/dữ liệu\n",
            f"- Mô hình: `{config['model']['name']}`; revision `{config['model']['revision']}`.\n"
            f"- Precision nguồn: {config['model']['dtype']}; seed: {config['seed']}.\n"
            f"- Tổng tham số: {manifest.get('parameter_count', 'không có metadata')}; tham số projection trong scope: {quantizable_count}.\n"
            f"- Calibration: C4 train, {config['data']['calibration']['samples']} mẫu × {config['data']['calibration']['length']} token.\n"
            f"- Validation: prefix cố định WikiText-2, {config['data']['validation_tokens']} token đích. Test: toàn bộ split nếu `test_tokens=null`.\n"
            f"- Số token đích thực tế: `{json.dumps(evaluated_counts)}`.\n"
            "- Chi tiết revision, token hash, ID calibration, GPU, thư viện, số tham số và cấu hình: [manifest.json](manifest.json).\n",
            "## 4. Trạng thái và bảng kết quả\n", f"```json\n{json.dumps(status, indent=2)}\n```\n",
            "Các bảng dưới đây chỉ sử dụng phép đo thực tế đã lưu. Một bảng chưa đủ hàng không được xem là hoàn thành thí nghiệm.\n",
            "### E1 — độ chính xác và chất lượng\n",
            markdown_table(e1.sort_values("bit_width", ascending=False) if not e1.empty else e1,
                           ["precision", "validation_nll", "validation_ppl", "test_nll", "test_ppl", "delta_nll", "normalized_weight_mse", "payload_bpw"]),
            "### E2/E3 — độ nhạy\n",
            f"E2 hiện có {len(e2)} phép đo; E3 có {len(e3)} phép đo. Dữ liệu đầy đủ: [E2](e2.csv), [E3](e3.csv), "
            "[ma trận nhạy nhất/ít nhạy nhất](e2_extremes.csv), [thống kê trong ma trận](e3_summary.csv).\n",
            "### E4 — tương quan Spearman\n",
            markdown_table(e4, ["granularity", "module_group", "bit_width", "n_units", "spearman_weight_vs_nll", "spearman_output_vs_nll", "confidence_interval"]),
            "## 5. Hình vẽ\n"]
    for path in plots:
        relative = path.relative_to(out)
        text.append(f"![{path.stem}]({relative})\n[PDF]({relative.with_suffix('.pdf')}) · [CSV]({relative.with_suffix('.csv')})\n")
    text.append("## 6. Quan sát từ phép đo\n")
    if not e1.empty and len(e1[e1.precision != "FP"]) >= 2:
        q = e1[e1.precision != "FP"].sort_values("bit_width", ascending=False)
        monotonic = bool((np.diff(q.validation_nll) >= 0).all())
        text.append(f"NLL {'tăng đơn điệu' if monotonic else 'không tăng đơn điệu'} khi giảm bit trong các cấu hình đã đo.\n")
        changes = np.diff(q.validation_nll)
        i = int(np.argmax(changes))
        text.append(f"Bước suy giảm NLL lớn nhất trong sweep lượng tử: W{q.iloc[i].bit_width} → W{q.iloc[i+1].bit_width}, Δ = {changes[i]:.6g} nats/token.\n")
    if not e2.empty:
        w2 = e2[e2.bit_width == 2]
        if len(w2):
            top, low = w2.loc[w2.delta_nll.idxmax()], w2.loc[w2.delta_nll.idxmin()]
            text.append(f"Trong các ma trận W2 đã đo, ΔNLL lớn nhất: `{top.module_name}` ({top.delta_nll:.6g}); "
                        f"nhỏ nhất: `{low.module_name}` ({low.delta_nll:.6g}). Có {(w2.delta_nll < 0).sum()} giá trị âm; không lọc bỏ.\n")
    if not e3.empty:
        spread = e3.groupby(["bit_width", "module_name"]).delta_nll.agg(lambda x: x.max()-x.min())
        text.append(f"Khoảng biến thiên ΔNLL trong từng ma trận ở các phép đo chunk: {spread.min():.6g} đến {spread.max():.6g} nats/token. "
                    "Không dùng tỷ số max/min khi độ nhạy gần 0.\n")
    if not e4.empty:
        for _, row in e4[e4.module_group == "all"].iterrows():
            weight_rho = "không xác định" if pd.isna(row.spearman_weight_vs_nll) else f"{row.spearman_weight_vs_nll:.4g}"
            output_rho = "không xác định" if pd.isna(row.spearman_output_vs_nll) else f"{row.spearman_output_vs_nll:.4g}"
            text.append(f"E4 ({row.granularity}, W{row.bit_width}, n={row.n_units}): Spearman weight-error/ΔNLL = "
                        f"{weight_rho}; output-error/ΔNLL = {output_rho}. "
                        "Đây là hệ số trên tập đã đo; cần đọc cùng CI trong bảng, không thay thế NLL bằng lỗi cục bộ.\n")
    e5, e6 = frame(out, "e5"), frame(out, "e6")
    if not e5.empty:
        excludes_zero = sum(ci is not None and (ci[0] > 0 or ci[1] < 0) for ci in e5.interaction_ci)
        text.append(f"E5: {len(e5)} cặp chunk; {(e5.interaction_residual>0).sum()} residual dương, "
                    f"{(e5.interaction_residual<0).sum()} residual âm. "
                    f"Trung bình |residual| = {e5.interaction_residual.abs().mean():.6g} nats/token. "
                    f"Có {excludes_zero} CI bootstrap không chứa 0 (mô tả, chưa hiệu chỉnh nhiều so sánh). "
                    "CI từng cặp nằm trong [e5.csv](e5.csv); đây là finite difference trên các cặp đã chọn, không phải kết luận nhân quả.\n")
    else:
        text.append("E5 chưa có phép đo; không kết luận về tương tác.\n")
    if e6.empty:
        text.append("E6 đã có mã nhưng chưa có phép đo; không kết luận về lợi ích mixed precision.\n")
    else:
        text.append(markdown_table(e6, ["strategy", "target_bpw", "realized_bpw", "validation_nll", "test_nll", "test_ppl"]))
    text.extend(["## 7. Giới hạn\n",
                 "- Fake quantization dùng tensor BF16 sau giải lượng tử; không đo hay khẳng định tiết kiệm VRAM hoặc tăng tốc. Payload là chi phí mã lý thuyết; metadata 32-bit scale + 8-bit zero-point được ghi riêng.\n"
                 "- Chỉ một kiến trúc/kích thước nếu chưa có lượt 3B; không khái quát cho mọi LLM.\n"
                 "- Context 512 và prefix validation có thể ảnh hưởng độ nhạy; kết quả không so trực tiếp với PPL dùng protocol khác.\n"
                 "- Calibration lấy từ 10.000 tài liệu đầu của một shard C4 train rồi chọn theo seed; không đại diện toàn bộ C4. Cache activation chỉ giữ các vị trí phân bố đều trên từng mẫu.\n"
                 "- Kiểm tra trùng lặp dùng hash văn bản chính xác sau chuẩn hóa khoảng trắng; không phát hiện trùng ngữ nghĩa. "
                 "Overlap giữa các hàng WikiText validation/test được ghi trong manifest, không dùng test để chọn cấu hình.\n"
                 "- CI là ước lượng có điều kiện trên tập mẫu; bootstrap khối không loại bỏ hoàn toàn phụ thuộc dài hạn. "
                 "E3 chỉ có 9 cụm ma trận (attention chỉ 3); CI tương quan có thể rộng và không ổn định.\n"
                 "- Chưa suy ra ý nghĩa thống kê hay quan hệ nhân quả từ tương quan. Không gộp độ nhạy ma trận với độ nhạy chunk.\n",
                 "## 8. Diễn giải cho các câu hỏi nghiên cứu\n",
                 "RQ1 dùng E1 để đánh giá trade-off precision–quality. RQ2 dùng bản đồ E2 và phân bố E3 để mô tả tính không đồng đều. "
                 "RQ3 dựa vào hệ số và CI của E4 để đánh giá mức dự báo của lỗi cục bộ; lỗi tái tạo không thay thế phép đo NLL toàn mô hình. "
                 "RQ4 chỉ được đánh giá trên các cặp E5 hoặc kế hoạch E6 đã thực sự đo; không suy diễn từ E1–E4.\n"])
    (out / "REPORT.md").write_text("\n".join(text))
    artifact_index(out)
    return status
