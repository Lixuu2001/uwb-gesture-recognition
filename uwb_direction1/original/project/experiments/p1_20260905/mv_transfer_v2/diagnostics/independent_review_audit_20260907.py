"""Read-only audit of saved P1 runs; writes only a separate review JSON.

No fitting, checkpoint inference, target-cache creation, or original-report edits.
The shuffle statistic is an upper bound, not a replay of CUDA random permutations.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
BASE = V2.parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_split(y, subjects, held):
    # Independently reproduce pipeline.py's documented quality/split protocol.
    good = ~(((subjects == 2) & np.isin(y, [10, 11])) |
             ((subjects == 6) & (y == 3)))
    rng = np.random.default_rng(2026)
    train, val = [], []
    for user in sorted(set(subjects) - {held}):
        for label in sorted(set(y)):
            ids = np.flatnonzero((subjects == user) & (y == label) & good)
            rng.shuffle(ids)
            if not len(ids):
                continue
            n = max(1, int(round(len(ids) * .1)))
            val.extend(ids[:n])
            train.extend(ids[n:])
    return np.sort(train), np.sort(val), np.flatnonzero((subjects == held) & good)


def paired(table, candidate, control, seeds):
    delta = np.array([np.mean([table[candidate, s, h] - table[control, s, h]
                              for s in seeds]) * 100 for h in range(1, 9)])
    rng = np.random.default_rng(20260906)
    boot = delta[rng.integers(0, 8, size=(10000, 8))].mean(1)
    return dict(mean_pp=float(delta.mean()), ci95_pp=np.percentile(boot, [2.5, 97.5]).tolist(),
                positive_users=int((delta > 0).sum()), per_user_pp=delta.tolist(), seeds=seeds)


def main():
    y = np.load(BASE / "prepared/y.npy")
    subjects = np.load(BASE / "prepared/subj.npy")
    splits = {h: canonical_split(y, subjects, h) for h in range(1, 9)}
    table, checks, shuffle, source_changes = {}, [], [], {}
    init_hashes, sampler_hashes = {}, {}
    for path in sorted((V2 / "runs").glob("*/status.json")):
        status = read(path)
        assert status["status"] == "complete", str(path)
        folder = path.parent
        h, seed, arm = status["held_subject"], status["model_seed"], status["method_tag"]
        tr, va, te = splits[h]
        with np.load(folder / "split.npz") as split:
            for key, ids in zip(["train_ids", "val_ids", "test_ids"], [tr, va, te]):
                np.testing.assert_array_equal(split[key], ids)
        with np.load(folder / "predictions.npz") as pred:
            np.testing.assert_array_equal(pred["ids"], te)
            np.testing.assert_array_equal(pred["y"], y[te])
            np.testing.assert_array_equal(pred["pred"], pred["logits"].argmax(-1))
            assert np.isfinite(pred["logits"]).all()
            acc = float((pred["pred"] == y[te]).mean())
            f1 = float(f1_score(y[te], pred["pred"], labels=np.unique(y[te]),
                                average="macro", zero_division=0))
        assert abs(acc - status["accuracy"]) < 1e-12
        assert abs(f1 - status["macro_f1"]) < 1e-12
        with np.load(folder / "validation_predictions.npz") as pred:
            np.testing.assert_array_equal(pred["ids"], va)
            np.testing.assert_array_equal(pred["y"], y[va])
            logits = pred["logits"].astype(np.float64)
            maximum = logits.max(1)
            ce = float((maximum + np.log(np.exp(logits - maximum[:, None]).sum(1))
                        - logits[np.arange(len(va)), y[va]]).mean())
            assert abs(ce - status["best_val_ce"]) < 1e-5
        history = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        best = min(history, key=lambda r: r["val_ce"])
        assert best["epoch"] == status["best_epoch"]
        config, prov = read(folder / "config.json"), read(folder / "provenance.json")
        assert prov["student_input_views"] == ["Right"]
        key = (h, seed)
        fingerprint = json.dumps(prov["initial_classifier_hash"], sort_keys=True)
        assert init_hashes.setdefault(key, fingerprint) == fingerprint
        with np.load(folder / "sampling.npz") as sample:
            groups = sample["groups"]
        digest = hashlib.sha256(np.ascontiguousarray(groups).tobytes()).hexdigest()[:32]
        assert digest == prov["sampler_manifest_hash"]
        assert sampler_hashes.setdefault((h, config["sampler"]), digest) == digest
        assert np.isin(groups[groups >= 0], tr).all()
        if arm == "B-SHUF":
            total, eligible = 0, 0
            for group in groups[:status["stopped_epoch"]].reshape(-1, groups.shape[-1]):
                ids = group[group >= 0]
                _, counts = np.unique(np.column_stack([y[ids], subjects[ids]]), axis=0,
                                      return_counts=True)
                total += len(ids)
                eligible += int(counts[counts > 1].sum())
            shuffle.append(dict(fold=h, exposures=total, max_changeable=eligible,
                                max_changed_fraction=eligible / total,
                                min_unchanged_fraction=1 - eligible / total))
        changed = []
        for name, recorded in prov["code_snapshot_sha256"].items():
            source = BASE / name
            if source.exists() and hashlib.sha256(source.read_bytes()).hexdigest() != recorded:
                changed.append(name)
        source_changes[folder.name] = changed
        table[arm, seed, h] = acc
        checks.append(folder.name)
    contrasts = {}
    for a, b in [("A-SC", "A-GS"), ("A-MV", "A-US"), ("B-MV", "B-SELF"),
                 ("A-US", "A-SC"), ("A-MV", "A-SC")]:
        for seeds in [[2026], [2027], [2026, 2027]]:
            contrasts[f"{a} minus {b} seeds {seeds}"] = paired(table, a, b, seeds)
    contrasts["B-SHUF minus B-MV seeds [2026]"] = paired(table, "B-SHUF", "B-MV", [2026])
    means = {f"{a} seed {s}": float(np.mean([table[a, s, h] for h in range(1, 9)]) * 100)
             for a, s in sorted({(a, s) for a, s, h in table})}
    c_summary = {}
    selected_sources = {}
    for name in ["predictability_residual", "predictability"]:
        source_folds = read(HERE / f"{name}.json")["per_fold"]
        folds = list(source_folds.values())
        selected_sources[name] = {}
        for fold, row in source_folds.items():
            coordinates = np.asarray(row["selected_coordinates"])
            selected_sources[name][fold] = dict(
                view_counts_L_T_R=np.bincount(coordinates // 224, minlength=3).tolist(),
                spatial_count=int(((coordinates % 224) >= 192).sum()),
                unique_temporal_channels=int(len(np.unique(coordinates[(coordinates % 224) < 192] % 32))))
        c_summary[name] = dict(
            eligible_mean=float(np.mean([f["selection"]["eligible"] for f in folds])),
            selected_skill_mean=float(np.mean([f["selection"]["mean_skill_selected"] for f in folds])),
            selected_probe_mean=float(np.mean([f["class_probe"]["selected"]["mean_inner_accuracy"] for f in folds])),
            random_probe_mean=float(np.mean([f["class_probe"]["random"]["mean_inner_accuracy"] for f in folds])),
            degenerate_denominators=sum(u["degenerate_denominator"] for f in folds for u in f["inner_users"]))
    historical = {}
    for seed in [2026, 2027]:
        accs = {}
        for arm in ["sv_R_concat", "sv_R_concat_kd"]:
            scores = []
            for h in range(1, 9):
                with np.load(BASE / "results_clean" / f"{arm}_s{seed}_h{h}_predictions.npz") as p:
                    np.testing.assert_array_equal(p["y"], y[p["ids"]])
                    np.testing.assert_array_equal(p["ids"], splits[h][2])
                    np.testing.assert_array_equal(p["pred"], p["logits"].argmax(-1))
                    scores.append(float((p["pred"] == p["y"]).mean()) * 100)
            accs[arm] = scores
        historical[str(seed)] = dict(
            concat_accuracy_percent=float(np.mean(accs["sv_R_concat"])),
            kd_accuracy_percent=float(np.mean(accs["sv_R_concat_kd"])),
            kd_minus_concat_pp=float(np.mean(np.array(accs["sv_R_concat_kd"]) - accs["sv_R_concat"])))
    out = dict(run_count=len(checks), checks_passed=checks, per_arm_mean_accuracy_percent=means,
               paired_contrasts=contrasts, shuffle_upper_bound_per_fold=shuffle,
               shuffle_min_unchanged_fraction_pooled=1 - sum(r["max_changeable"] for r in shuffle) / sum(r["exposures"] for r in shuffle),
               predictability_saved_json_reaggregation=c_summary,
               selected_coordinate_sources=selected_sources,
               historical_kd_saved_predictions_reaggregation=historical,
               recorded_sources_differing_from_current=source_changes,
               scope="Saved predictions, canonical IDs/labels/splits, validation choice, sampling manifests and recorded init hashes. No checkpoint forward pass or retraining. C values reaggregated from saved JSON, not independently refitted.")
    (HERE / "INDEPENDENT_AUDIT_20260907.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ["checks_passed", "recorded_sources_differing_from_current"]}, indent=2))


if __name__ == "__main__":
    main()
