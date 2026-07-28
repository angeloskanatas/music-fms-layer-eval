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
# display = the paper's official model name (tex Table 1 / source papers).
# readout_of groups architecture-specific readouts of ONE model (MT2's
# multi-class-token heads, cf. seq-JEPA agg/glance) — views merge these rows.
# hidden = exported to data/ but excluded from views (different eval protocol,
# not comparable: the MuQ paper's own suite replication rows).
MODELS = {
    "mert-v1-95M":     {"file": "mert-v1-95M",  "display": "MERT-v1-95M",  "family": "masked"},
    "mert-v1-330M":    {"file": "mert-v1-330M", "display": "MERT-v1-330M", "family": "masked"},
    "musicfm":         {"file": "musicfm",      "display": "MusicFM (MSD)", "family": "masked"},
    "muq":             {"file": "muq",          "display": "MuQ (iter)",    "family": "masked"},
    "omar-rq-multifeature-25hz": {"file": "omar-rq-multifeature-25hz",
                        "display": "OMAR-RQ (multifeature)", "family": "masked"},
    "omar-rq-base":    {"file": "omar-rq-base", "display": "OMAR-RQ (base)",
                        "family": "masked", "in_paper": False},
    "musicgen-small":  {"file": "musicgen-small",  "display": "MusicGen-S", "family": "autoregressive"},
    "musicgen-medium": {"file": "musicgen-medium", "display": "MusicGen-M", "family": "autoregressive"},
    "musicgen-large":  {"file": "musicgen-large",  "display": "MusicGen-L", "family": "autoregressive"},
    "yue-0.5b":        {"file": "yue-0.5b", "display": "YuE-s1-0.5B", "family": "autoregressive"},
    "yue-7b":          {"file": "yue-7b",   "display": "YuE-s1-7B",   "family": "autoregressive"},
    "clap":            {"file": "clap",     "display": "LAION-CLAP",  "family": "contrastive"},
    "myna":            {"file": "myna",     "display": "Myna-Base",   "family": "contrastive"},
    "maest":           {"file": "maest",    "display": "MAEST", "family": "supervised", "in_paper": False},
    "mt2":             {"file": "mt2", "display": "MT2", "family": "contrastive",
                        "in_paper": False, "readout_of": "mt2", "readout_label": "token-avg"},
    "mt2-cls-avg":     {"file": "mt2-cls-avg", "display": "MT2", "family": "contrastive",
                        "in_paper": False, "readout_of": "mt2", "readout_label": "CLS-avg"},
    "mt2-cls-contrastive": {"file": "mt2-cls-contrastive", "display": "MT2", "family": "contrastive",
                        "in_paper": False, "readout_of": "mt2", "readout_label": "contrastive CLS"},
    "mt2-cls-equiv":   {"file": "mt2-cls-equiv", "display": "MT2", "family": "contrastive",
                        "in_paper": False, "readout_of": "mt2", "readout_label": "equivariant CLS"},
    "muq-paper":       {"file": "muq_paper", "display": "MuQ (paper suite)", "family": "masked",
                        "in_paper": False, "variant_of": "muq", "hidden": True,
                        "note": "replication of the MuQ paper's own eval suite (different protocol, not comparable)"},
    "musicfm-paper":   {"file": "musicfm_paper", "display": "MusicFM (paper suite)", "family": "masked",
                        "in_paper": False, "variant_of": "musicfm", "hidden": True,
                        "note": "replication of the MuQ paper's own eval suite (different protocol, not comparable)"},
    "qwen2audio-instruct": {"file": "qwen2audio-instruct", "display": "Qwen2-Audio-Instruct",
                        "family": "audio-llm", "in_paper": False},
    "musicflamingo":   {"file": "musicflamingo", "display": "Music-Flamingo",
                        "family": "audio-llm", "in_paper": False},
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

# Short display labels for table headers / selectors (full description from
# the registry rides on the title attribute). Machine keys never face users.
TASK_LABEL = {
    "key": "Key", "nsynth_pitch": "Pitch", "chords_ace_mirex": "Chords",
    "chords_ace_root": "Chords root", "chords_ace_thirds": "Chords thirds",
    "beat_f1": "Beat", "downbeat_f1": "Downbeat",
    "nsynth_instrument": "Instrument",
    "genre": "Genre", "hxmsa": "Structure",
    "emo_r2": "Emotion", "emo_r2a": "Arousal", "emo_r2v": "Valence",
    "mtt_ap": "MTT", "mtt_auroc": "MTT AUROC",
    "mtg_genre_ap": "MTG-Genre", "mtg_genre_auroc": "MTG-Genre AUROC",
    "mtg_instrument_ap": "MTG-Instr", "mtg_instrument_auroc": "MTG-Instr AUROC",
    "mtg_mood_ap": "MTG-Mood", "mtg_mood_auroc": "MTG-Mood AUROC",
    "mtg_top50_ap": "MTG-Top50", "mtg_top50_auroc": "MTG-Top50 AUROC",
    "dimsim_acc_centered": "DimSim", "dimsim_acc_cosine": "DimSim cos",
    "dimsim_acc_l2": "DimSim L2",
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


def sparkline(scores, best, color="#94a3b8", w=52, h=16):
    """Inline SVG sparkline of a layer profile with the best layer dotted.
    Build-time rendered: no JS, crawler-safe, ~200 bytes."""
    lo, hi = min(scores), max(scores)
    rng = (hi - lo) or 1.0
    n = len(scores)
    xs = [2 + i * (w - 4) / max(n - 1, 1) for i in range(n)]
    ys = [h - 2 - (s - lo) / rng * (h - 5) for s in scores]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (f"<svg class='spark' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
            f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='1.3'/>"
            f"<circle cx='{xs[best]:.1f}' cy='{ys[best]:.1f}' r='2.4' fill='#2563eb'/></svg>")


def render_atlas_table(task_registry):
    """Build-time render of the results table into atlas.html (real HTML for
    crawler/LLM legibility; JS adds only sort + expand). Derived numbers
    (best layer, family means) are computed HERE, never persisted."""
    models = []
    for model_id, spec in MODELS.items():
        if spec.get("hidden"):          # different protocol — data only, no views
            continue
        f = OUT / "results" / model_id / "downstream.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        per_task = {}
        for r in d["records"]:
            t, scores = r["task"], r["layers"]
            best = max(range(len(scores)), key=lambda i: scores[i])
            per_task[t] = {"score": scores[best], "layer": best, "scores": scores,
                           "caveat": r.get("caveat"), "model": model_id, "t": t,
                           "readout_label": spec.get("readout_label")}
        models.append({
            "id": model_id, "display": spec["display"], "family": d["family"],
            "in_paper": d["in_paper"], "readout_of": spec.get("readout_of"),
            "n_layers": max((r["n_layers"] for r in d["records"]), default=0),
            "per_task": per_task,
        })
    # merge architecture-specific readouts of one model (MT2 heads) into a
    # single row: per cell keep the best readout and tag it.
    merged, groups = [], {}
    for m in models:
        g = m["readout_of"]
        if not g:
            merged.append(m)
            continue
        if g not in groups:
            groups[g] = {**m, "id": g, "is_group": True}
            merged.append(groups[g])
        else:
            base = groups[g]
            for t, pt in m["per_task"].items():
                if t not in base["per_task"] or pt["score"] > base["per_task"][t]["score"]:
                    base["per_task"][t] = pt
    models = merged
    # mean RANK over primary tasks (scale-free; value-means across mixed
    # metrics are meaningless). Rank within each task over models that have it.
    n_prim = len(PRIMARY_METRICS)
    ranks = {m["id"]: [] for m in models}
    for t in PRIMARY_METRICS:
        have = sorted((m for m in models if t in m["per_task"]),
                      key=lambda m: -m["per_task"][t]["score"])
        for i, m in enumerate(have):
            ranks[m["id"]].append(i + 1)
    for m in models:
        r = ranks[m["id"]]
        m["mean_rank"] = sum(r) / len(r) if r else None
        m["rank_n"] = len(r)
    # sort: near-full-coverage models first by mean rank; partial-coverage
    # models (few-task mean ranks are flattering, not comparable) sink below.
    full_cov = max(m["rank_n"] for m in models)
    models.sort(key=lambda m: (0 if m["rank_n"] >= 0.75 * full_cov else 1,
                               m["mean_rank"] if m["mean_rank"] else 99))
    prim_tasks = {tf: sorted(t for t, f in TASK_FAMILY.items()
                             if f == tf and t in PRIMARY_METRICS)
                  for tf in TASK_FAMILY_ORDER}
    comp_tasks = {tf: sorted(t for t, f in TASK_FAMILY.items()
                             if f == tf and t not in PRIMARY_METRICS)
                  for tf in TASK_FAMILY_ORDER}

    def th_task(t, tf, hidden):
        h = " hidden" if hidden else ""
        cls = "detail d-" + tf if hidden else "ptask"
        label = TASK_LABEL.get(t, t)
        desc = task_registry.get(t, t)
        return f"<th class='{cls}' data-sort='num'{h} title='{desc}'>{label}</th>"

    head_top = ["<tr><th rowspan='2' data-sort='str'>Model</th>"
                "<th rowspan='2' data-sort='str'>Paradigm</th>"
                "<th rowspan='2' data-sort='num' title='mean rank across primary "
                "tasks the model was evaluated on; lower is better'>Rank</th>"]
    head_sub = ["<tr>"]
    for tf in TASK_FAMILY_ORDER:
        n_p, n_c = len(prim_tasks[tf]), len(comp_tasks[tf])
        exp = (f" <span class='chev' title='{n_c} companion metrics'>&#9656;</span>"
               if n_c else "")
        head_top.append(f"<th colspan='{n_p}' class='famgrp{' expandable' if n_c else ''}' "
                        f"data-fam='{tf}'>{tf.capitalize()}{exp}</th>")
        for t in prim_tasks[tf]:
            head_sub.append(th_task(t, tf, hidden=False))
        for t in comp_tasks[tf]:
            head_top[-1] = head_top[-1]
            head_sub.append(th_task(t, tf, hidden=True))
    head_top.append("</tr>")
    head_sub.append("</tr>")

    def cell(pt, tf, hidden):
        h = " hidden" if hidden else ""
        cls = ("detail d-" + tf) if hidden else "ptask"
        if pt is None:
            return f"<td class='num {cls} na'{h} title='not evaluated'>&mdash;</td>"
        cv = (f"<sup class='cav' title='{pt['caveat']}'>&dagger;</sup>"
              if pt.get("caveat") else "")
        tip = ("open the full layer curve" if not pt.get("readout_label") else
               f"best readout: {pt['readout_label']} — open the curves for all readouts")
        return (f"<td class='num {cls}'{h}>"
                f"<a class='cellink' href='explorer.html?model={pt['model']}&amp;task={pt['t']}' "
                f"title='{tip}'>"
                f"<span class='sc'>{pt['score']:.1f}{cv}"
                f"<span class='lyr'>L{pt['layer']}</span></span>"
                f"{sparkline(pt['scores'], pt['layer'])}</a></td>")

    rows = []
    for m in models:
        c = FAMILY_COLOR.get(m["family"], "#94a3b8")
        extra = "" if m["in_paper"] else " <span class='xtra' title='not part of the paper&#39;s 12-model set'>+</span>"
        rk = (f"{m['mean_rank']:.1f}" if m["mean_rank"] else "&mdash;")
        part = ("" if m["rank_n"] == n_prim else
                f"<sup class='cav' title='ranked on {m['rank_n']} of {n_prim} "
                f"primary tasks'>p</sup>")
        multi = ("<sup class='cav' title='multi-readout model: each cell shows its "
                 "best readout; click a cell to compare all readouts'>&#9702;</sup>"
                 if m.get("is_group") else "")
        cells = [f"<tr style='--fam:{c}'>",
                 f"<td class='mname' title='{m['id']}'>{m['display']}{multi}{extra}</td>",
                 f"<td><span class='dot'></span>{m['family']}</td>",
                 f"<td class='num'>{rk}{part}</td>"]
        for tf in TASK_FAMILY_ORDER:
            for t in prim_tasks[tf]:
                cells.append(cell(m["per_task"].get(t), tf, hidden=False))
            for t in comp_tasks[tf]:
                cells.append(cell(m["per_task"].get(t), tf, hidden=True))
        cells.append("</tr>")
        rows.append("".join(cells))

    tpl = (Path(__file__).resolve().parent / "atlas_template.html").read_text()
    page = (tpl.replace("<!--THEAD-->", "".join(head_top) + "".join(head_sub))
               .replace("<!--TBODY-->", "\n".join(rows))
               .replace("{{DATE}}", str(date.today()))
               .replace("{{NMODELS}}", str(len(models))))
    (OUT.parent / "atlas.html").write_text(page)

    # Layer Explorer: inject the model/task manifests into the template.
    # Readout-group members appear as their own selectable entries, properly
    # labeled ("MT2 — equivariant CLS"), since each has its own curve file.
    # One dropdown entry per MODEL; readout-group members become curves inside
    # the chart (readout names are expert jargon — the chart legend + a note
    # explain them in place instead of polluting the model list).
    explorer_models, seen_groups = [], {}
    for mid, spec in MODELS.items():
        if spec.get("hidden") or not (OUT / "results" / mid / "downstream.json").exists():
            continue
        g = spec.get("readout_of")
        if g:
            if g not in seen_groups:
                seen_groups[g] = {"id": g, "label": spec["display"],
                                  "family": spec["family"],
                                  "in_paper": spec.get("in_paper", True),
                                  "color": FAMILY_COLOR.get(spec["family"], "#94a3b8"),
                                  "members": []}
                explorer_models.append(seen_groups[g])
            seen_groups[g]["members"].append(
                {"id": mid, "readout": spec["readout_label"]})
        else:
            explorer_models.append(
                {"id": mid, "label": spec["display"], "family": spec["family"],
                 "in_paper": spec.get("in_paper", True),
                 "color": FAMILY_COLOR.get(spec["family"], "#94a3b8")})
    models_json = json.dumps(explorer_models)
    tasks_seen = sorted({t for m in models for t in m["per_task"]},
                        key=lambda t: (TASK_FAMILY_ORDER.index(TASK_FAMILY[t])
                                       if TASK_FAMILY.get(t) in TASK_FAMILY_ORDER
                                       else 9, t not in PRIMARY_METRICS, t))
    tasks_json = json.dumps([
        {"id": t, "label": TASK_LABEL.get(t, t), "desc": task_registry.get(t, t)}
        for t in tasks_seen])
    etpl = (Path(__file__).resolve().parent / "explorer_template.html").read_text()
    (OUT.parent / "explorer.html").write_text(
        etpl.replace("/*MODELS_JSON*/", models_json)
            .replace("/*TASKS_JSON*/", tasks_json)
            .replace("{{DATE}}", str(date.today())))
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
