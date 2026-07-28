#!/usr/bin/env python3
"""Export the layerbylayer result files into the atlas data plane.

Stage 1: downstream per-layer scores + model/task registries.
Design rules (worklog 4x/4y): folder-per-model, append-only, self-describing
records, raw per-layer scores only, open vocabularies via registries, per-
protocol canonical flags. Published-protocol numbers pin to the April canon;
this export never overwrites an existing record file (append-only).
"""

import json
import sys
from datetime import date
from pathlib import Path

SCHEMA_VERSION = "1.0.0"
SRC = Path("/home/akanatas/projects/layerbylayer/layerbylayer/output")
OUT = Path(__file__).resolve().parent.parent / "data"

# canonical model id -> downstream filename stem. Variants reference their base
# via variant_of; paper-replication files use the source paper's task keys.
MODELS = {
    "mert-v1-95M":     {"file": "mert-v1-95M",     "family": "masked"},
    "mert-v1-330M":    {"file": "mert-v1-330M",    "family": "masked"},
    "musicfm":         {"file": "musicfm",         "family": "masked"},
    "muq":             {"file": "muq",             "family": "masked"},
    "omar-rq-base":    {"file": "omar-rq-base",    "family": "masked"},
    "omar-rq-multifeature-25hz": {"file": "omar-rq-multifeature-25hz", "family": "masked"},
    "musicgen-small":  {"file": "musicgen-small",  "family": "autoregressive"},
    "musicgen-medium": {"file": "musicgen-medium", "family": "autoregressive"},
    "musicgen-large":  {"file": "musicgen-large",  "family": "autoregressive"},
    "yue-0.5b":        {"file": "yue-0.5b",        "family": "autoregressive"},
    "yue-7b":          {"file": "yue-7b",          "family": "autoregressive"},
    "clap":            {"file": "clap",            "family": "contrastive"},
    "myna":            {"file": "myna",            "family": "contrastive"},
    "maest":           {"file": "maest",           "family": "supervised", "in_paper": False},
    "mt2":             {"file": "mt2",             "family": "contrastive", "in_paper": False},
    "mt2-cls-avg":     {"file": "mt2-cls-avg",     "family": "contrastive", "in_paper": False, "variant_of": "mt2"},
    "mt2-cls-contrastive": {"file": "mt2-cls-contrastive", "family": "contrastive", "in_paper": False, "variant_of": "mt2"},
    "mt2-cls-equiv":   {"file": "mt2-cls-equiv",   "family": "contrastive", "in_paper": False, "variant_of": "mt2"},
    "muq-paper":       {"file": "muq_paper",       "family": "masked", "in_paper": False, "variant_of": "muq",
                        "note": "replication of the MuQ paper's own eval suite (different task vocabulary)"},
    "musicfm-paper":   {"file": "musicfm_paper",   "family": "masked", "in_paper": False, "variant_of": "musicfm",
                        "note": "replication of the MusicFM paper's own eval suite"},
    "qwen2audio-instruct": {"file": "qwen2audio-instruct", "family": "audio-llm", "in_paper": False},
    "musicflamingo":   {"file": "musicflamingo",   "family": "audio-llm", "in_paper": False},
}

# alias -> canonical task key (drift found in the source files)
TASK_ALIASES = {"emotion_r2a": "emo_r2a", "emotion_r2v": "emo_r2v"}

# Metrics that exist in probe outputs but must NEVER be exported or rendered.
# Evidence: GTZANBeatTracking probes train with loss_weights [1.0, 1.0, 0.0] --
# the tempo head is untrained (train/loss_tempo == 0.0) yet tempo metrics are
# still logged to metrics.csv. Parsing probe dirs without this guard would
# publish garbage numbers.
EXCLUDED_METRICS = {
    "tempo_acc1": "untrained head (beat-probe tempo loss weight = 0.0)",
    "tempo_acc2": "untrained head (beat-probe tempo loss weight = 0.0)",
    "tempo_mae":  "untrained head (beat-probe tempo loss weight = 0.0)",
}

# Known aggregation caveats carried on the record, not in prose (worklog #43):
METRIC_CAVEATS = {
    "beat_f1": "MARBLE corpus-pooled F-measure; per-track mir_eval variant differs",
    "downbeat_f1": "MARBLE corpus-pooled F-measure; per-track mir_eval variant differs",
}

# task family per the paper's Sec 3.3 grouping (unknown keys stay unassigned;
# the site renders them generically -- open-vocabulary rule)
TASK_FAMILY = {
    "key": "tonal", "nsynth_pitch": "tonal",
    "chords_ace_root": "tonal", "chords_ace_thirds": "tonal", "chords_ace_mirex": "tonal",
    "beat_f1": "rhythm", "downbeat_f1": "rhythm",
    "nsynth_instrument": "timbre",
    "genre": "semantic", "hxmsa": "semantic",
    "emo_r2": "semantic", "emo_r2a": "semantic", "emo_r2v": "semantic",
    "mtt_ap": "semantic", "mtt_auroc": "semantic",
    "mtg_genre_ap": "semantic", "mtg_genre_auroc": "semantic",
    "mtg_instrument_ap": "semantic", "mtg_instrument_auroc": "semantic",
    "mtg_mood_ap": "semantic", "mtg_mood_auroc": "semantic",
    "mtg_top50_ap": "semantic", "mtg_top50_auroc": "semantic",
    "dimsim_acc_centered": "similarity", "dimsim_acc_cosine": "similarity", "dimsim_acc_l2": "similarity",
}

# The probe every downstream number came from (MARBLE constrained track).
# seed: the probe runs behind the current numbers are single-seed (Lightning
# seed_everything: 1234, verified in the probe configs). The schema supports
# multiple records per (task, readout) differing by seed; the build renders
# mean +/- sd once n_seeds > 1. Backfill policy: new cells run 3 seeds.
READOUT = {
    "id": "mlp512-marble",
    "probe": "MLP, one 512-unit hidden layer",
    "protocol": "MARBLE constrained track, frozen encoder, per-layer",
    "seed": 1234,
    "n_seeds": 1,
}

# One primary metric per dataset for family aggregates (companion metrics stay
# in the expanded view; aggregating AP+AUROC together would double-count).
PRIMARY_METRICS = {
    "key", "nsynth_pitch", "chords_ace_mirex",
    "beat_f1", "downbeat_f1",
    "nsynth_instrument",
    "genre", "hxmsa", "emo_r2", "mtt_ap",
    "mtg_genre_ap", "mtg_instrument_ap", "mtg_mood_ap", "mtg_top50_ap",
    "dimsim_acc_centered",
}

# Family identity colors — validated categorical palette (dataviz skill,
# ALL CHECKS PASS on white; identity is never color-alone: names are printed).
FAMILY_COLOR = {
    "masked": "#2a78d6", "autoregressive": "#eb6834", "contrastive": "#1baf7a",
    "audio-llm": "#eda100", "supervised": "#e87ba4",
}
FAMILY_ORDER = ["masked", "autoregressive", "contrastive", "audio-llm", "supervised"]
TASK_FAMILY_ORDER = ["tonal", "rhythm", "timbre", "semantic", "similarity"]


def export_downstream():
    written, skipped, problems = [], [], []
    task_registry = {}
    for model_id, spec in MODELS.items():
        src = SRC / "downstream" / "data" / f"{spec['file']}.json"
        if not src.exists():
            problems.append(f"MISSING source: {src.name}")
            continue
        raw = json.loads(src.read_text())
        meta, tasks = raw.get("metadata", {}), raw.get("downstream", {})
        n_layers = meta.get("n_layers")
        out_dir = OUT / "results" / model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "downstream.json"
        if out_file.exists():                      # append-only: never overwrite
            skipped.append(model_id)
            continue
        records = []
        for key, scores in tasks.items():
            task_id = TASK_ALIASES.get(key, key)
            if task_id in EXCLUDED_METRICS:        # hard guard, never soft
                problems.append(f"BLOCKED metric in source: {model_id}/{task_id} "
                                f"({EXCLUDED_METRICS[task_id]})")
                continue
            if n_layers and len(scores) != n_layers:
                problems.append(f"{model_id}/{task_id}: {len(scores)} scores vs n_layers={n_layers}")
            rec = {
                "task": task_id,
                "task_family": TASK_FAMILY.get(task_id),
                "readout": READOUT["id"],
                "layers": scores,                  # raw per-layer, index 0..L
                "n_layers": len(scores),
            }
            if task_id in METRIC_CAVEATS:
                rec["caveat"] = METRIC_CAVEATS[task_id]
            records.append(rec)
            desc = meta.get("tasks", {}).get(key)
            if desc and task_id not in task_registry:
                task_registry[task_id] = desc
        out_file.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "model": model_id,
            "family": spec["family"],
            "in_paper": spec.get("in_paper", True),
            "variant_of": spec.get("variant_of"),
            "source": str(src.relative_to(SRC.parent.parent)),
            "readout_spec": READOUT,
            "exported": str(date.today()),
            "records": records,
        }, indent=1))
        written.append(model_id)
    return written, skipped, problems, task_registry


def write_registries(task_registry):
    reg = OUT / "registry"
    reg.mkdir(parents=True, exist_ok=True)
    models_yaml = ["# canonical model registry (open vocabulary; edit by hand, append-only)"]
    for mid, spec in MODELS.items():
        models_yaml.append(f"{mid}:")
        models_yaml.append(f"  family: {spec['family']}")
        models_yaml.append(f"  in_paper: {str(spec.get('in_paper', True)).lower()}")
        if spec.get("variant_of"):
            models_yaml.append(f"  variant_of: {spec['variant_of']}")
        if spec.get("note"):
            models_yaml.append(f"  note: \"{spec['note']}\"")
    (reg / "models.yaml").write_text("\n".join(models_yaml) + "\n")

    tasks_yaml = ["# canonical task registry; aliases map source-file drift onto one vocabulary"]
    for tid in sorted(task_registry):
        tasks_yaml.append(f"{tid}:")
        tasks_yaml.append(f"  description: \"{task_registry[tid]}\"")
        fam = TASK_FAMILY.get(tid)
        if fam:
            tasks_yaml.append(f"  family: {fam}")
    tasks_yaml.append("aliases:")
    for a, t in TASK_ALIASES.items():
        tasks_yaml.append(f"  {a}: {t}")
    tasks_yaml.append("excluded_metrics:   # never export/render; reasons are evidence-based")
    for m, why in EXCLUDED_METRICS.items():
        tasks_yaml.append(f"  {m}: \"{why}\"")
    (reg / "tasks.yaml").write_text("\n".join(tasks_yaml) + "\n")


def render_atlas_table(task_registry):
    """Build-time render of the results table into atlas.html (real HTML for
    crawler/LLM legibility; JS adds only sort + expand). Derived numbers
    (best layer, family means) are computed HERE, never persisted."""
    models = []
    for model_id, spec in MODELS.items():
        f = OUT / "results" / model_id / "downstream.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        per_task, fam_scores = {}, {}
        for r in d["records"]:
            t, scores = r["task"], r["layers"]
            best = max(range(len(scores)), key=lambda i: scores[i])
            per_task[t] = {"score": scores[best], "layer": best,
                           "caveat": r.get("caveat")}
            if t in PRIMARY_METRICS and r.get("task_family"):
                fam_scores.setdefault(r["task_family"], []).append(scores[best])
        models.append({
            "id": model_id, "family": d["family"], "in_paper": d["in_paper"],
            "n_layers": max((r["n_layers"] for r in d["records"]), default=0),
            "per_task": per_task,
            "fam_mean": {k: sum(v) / len(v) for k, v in fam_scores.items()},
            "fam_n": {k: len(v) for k, v in fam_scores.items()},
        })
    models.sort(key=lambda m: (FAMILY_ORDER.index(m["family"])
                               if m["family"] in FAMILY_ORDER else 9, m["id"]))
    fam_tasks = {tf: sorted(t for t, f in TASK_FAMILY.items() if f == tf)
                 for tf in TASK_FAMILY_ORDER}

    head_top = ["<tr><th rowspan='2' data-sort='str'>Model</th>"
                "<th rowspan='2' data-sort='str'>Paradigm</th>"
                "<th rowspan='2' data-sort='num'>Layers</th>"]
    head_sub = ["<tr>"]
    for tf in TASK_FAMILY_ORDER:
        n = len(fam_tasks[tf])
        head_top.append(f"<th colspan='1' class='fam-agg expandable' data-fam='{tf}'>"
                        f"{tf.capitalize()} <span class='chev'>&#9656;</span></th>")
        head_sub.append(f"<th class='fam-agg' data-sort='num' data-fam='{tf}'>mean</th>")
        for t in fam_tasks[tf]:
            head_top[-1] = head_top[-1]  # keep aggregate col; detail cols follow, hidden
            head_top.append(f"<th class='detail d-{tf}' hidden>{t}</th>")
            head_sub.append(f"<th class='detail d-{tf}' data-sort='num' hidden>&nbsp;</th>")
    head_top.append("</tr>")
    head_sub.append("</tr>")

    rows = []
    for m in models:
        c = FAMILY_COLOR.get(m["family"], "#94a3b8")
        extra = "" if m["in_paper"] else " <span class='xtra' title='not part of the paper&#39;s 12-model set'>+</span>"
        cells = [f"<tr style='--fam:{c}'>",
                 f"<td class='mname'>{m['id']}{extra}</td>",
                 f"<td><span class='dot'></span>{m['family']}</td>",
                 f"<td class='num'>{m['n_layers']}</td>"]
        for tf in TASK_FAMILY_ORDER:
            v = m["fam_mean"].get(tf)
            n_have = m["fam_n"].get(tf, 0)
            n_full = len([t for t in fam_tasks[tf] if t in PRIMARY_METRICS])
            if v is None:
                cells.append(f"<td class='num fam-agg na' data-fam='{tf}' "
                             f"title='not evaluated'>&mdash;</td>")
            elif n_have < n_full:   # partial mean: NOT comparable down the column
                cells.append(f"<td class='num fam-agg' data-fam='{tf}' "
                             f"title='mean over {n_have} of {n_full} primary tasks "
                             f"— not comparable to complete rows'>{v:.1f}"
                             f"<sup class='cav'>p</sup></td>")
            else:
                cells.append(f"<td class='num fam-agg' data-fam='{tf}'>{v:.1f}</td>")
            for t in fam_tasks[tf]:
                pt = m["per_task"].get(t)
                if pt is None:
                    cells.append(f"<td class='num detail d-{tf} na' hidden "
                                 f"title='not evaluated'>&mdash;</td>")
                else:
                    cv = (f"<sup class='cav' title='{pt['caveat']}'>&dagger;</sup>"
                          if pt.get("caveat") else "")
                    cells.append(f"<td class='num detail d-{tf}' hidden>"
                                 f"{pt['score']:.1f}{cv}"
                                 f"<span class='lyr'>L{pt['layer']}</span></td>")
        cells.append("</tr>")
        rows.append("".join(cells))

    tpl = (Path(__file__).resolve().parent / "atlas_template.html").read_text()
    page = (tpl.replace("<!--THEAD-->", "".join(head_top) + "".join(head_sub))
               .replace("<!--TBODY-->", "\n".join(rows))
               .replace("{{DATE}}", str(date.today()))
               .replace("{{NMODELS}}", str(len(models))))
    (OUT.parent / "atlas.html").write_text(page)
    return len(models)


def coverage_report():
    """The fill-as-we-go TODO: which (model, primary task) cells are missing.
    Written to data/coverage.json so the eval queue is data, not memory."""
    missing = {}
    for model_id in MODELS:
        f = OUT / "results" / model_id / "downstream.json"
        if not f.exists():
            continue
        have = {r["task"] for r in json.loads(f.read_text())["records"]}
        gaps = sorted(PRIMARY_METRICS - have)
        if gaps:
            missing[model_id] = gaps
    (OUT / "coverage.json").write_text(json.dumps({
        "generated": str(date.today()),
        "policy": "fill-as-we-go, append-only; new cells run 3 seeds",
        "missing_primary_cells": missing,
        "n_missing": sum(len(v) for v in missing.values()),
    }, indent=1))
    return missing


def main():
    written, skipped, problems, task_registry = export_downstream()
    write_registries(task_registry)
    n = render_atlas_table(task_registry)
    print(f"atlas.html rendered: {n} models")
    missing = coverage_report()
    print(f"coverage: {sum(len(v) for v in missing.values())} primary cells missing "
          f"across {len(missing)} models -> data/coverage.json")
    print(f"written : {len(written)} models -> data/results/<model>/downstream.json")
    print(f"skipped (append-only, already present): {len(skipped)}")
    print(f"tasks in registry: {len(task_registry)}")
    if problems:
        print("\nVALIDATION PROBLEMS:")
        for p in problems:
            print(f"  ! {p}")
        sys.exit(1)
    print("validation: clean")


if __name__ == "__main__":
    main()
