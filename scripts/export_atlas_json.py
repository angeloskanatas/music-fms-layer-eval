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

PARAMS = {"mert-v1-95M": ("95M", 95), "mert-v1-330M": ("330M", 330),
     "musicfm": ("330M", 330), "muq": ("310M", 310),
     "omar-rq-multifeature-25hz": ("580M", 580),
     "musicgen-small": ("300M", 300), "musicgen-medium": ("1.5B", 1500),
     "musicgen-large": ("3.3B", 3300), "yue-0.5b": ("0.5B", 500),
     "yue-7b": ("7B", 7000), "clap": ("73M", 73), "myna": ("22M", 22),
     # extras, verified from primary sources 2026-07-28 (HF cards / papers /
     # exact config arithmetic; see worklog 4ab):
     "maest": ("87M", 87, "official checkpoint 86.9M incl. classification head"),
     "mt2": ("5.3M", 5.3, "paper-stated (arXiv 2507.12996)"),
     "omar-rq-base": ("90M", 90, "paper-stated (arXiv 2507.03482)"),
     "qwen2audio-instruct": ("637M", 637,
        "the probed audio tower (Whisper-derived, 32 layers — matches our 33 "
        "extracted states); full system 8.2B"),
     "musicflamingo": ("637M", 637,
        "the probed AF-Whisper tower (32 layers — matches our extracted states); "
        "full system 8.3B")}

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
        # registry collection must not depend on the append-only skip below,
        # else a re-run rebuilds tasks.yaml empty
        for key in tasks:
            task_id = TASK_ALIASES.get(key, key)
            desc = meta.get("tasks", {}).get(key)
            if desc and task_id not in task_registry:
                task_registry[task_id] = desc
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
        # all comparable models; only protocol-different rows stay data-only
        if spec.get("hidden"):
            continue
        f = OUT / "results" / model_id / "downstream.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        ff = OUT / "results" / model_id / "fusion.json"
        fus = {}
        if ff.exists():
            for r in json.loads(ff.read_text())["records"]:
                fus.setdefault(r["task"], {})[r["variant"]] = r["value"]
        per_task = {}
        for r in d["records"]:
            t, scores = r["task"], r["layers"]
            best = max(range(len(scores)), key=lambda i: scores[i])
            per_task[t] = {"score": scores[best], "layer": best, "scores": scores,
                           "caveat": r.get("caveat"), "model": model_id, "t": t,
                           "readout_label": spec.get("readout_label"),
                           "fusion": fus.get(t)}
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
                "<th rowspan='2' data-sort='num' title='parameters of the evaluated "
                "encoder, as reported in each paper (audio tower for two-tower models, "
                "decoder for autoregressive)'>Params</th>"
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
        fus = pt.get("fusion")
        if fus:
            proxy = {k: v for k, v in fus.items()
                     if k in ("topk_avg", "topk_stack", "top5_avg", "top5_stack")}
            train = {k: v for k, v in fus.items()
                     if k in ("learned_avg", "hconv", "attentive_ciernik")}
            parts = []
            for name, grp in (("proxy fusion", proxy), ("trainable fusion", train)):
                if grp:
                    bk = max(grp, key=grp.get)
                    parts.append(f"best {name}: {VARIANT_LABEL[bk]} "
                                 f"{grp[bk]:.1f} ({grp[bk] - pt['score']:+.1f})")
            if parts:
                tip += " | " + "; ".join(parts) + " — details on the Selection & Fusion page"
        return (f"<td class='num {cls}'{h}>"
                f"<a class='cellink' href='explorer.html?model={pt['model']}&amp;task={pt['t']}' "
                f"title='{tip}'>"
                f"<span class='sc'>{pt['score']:.1f}{cv}"
                f"<span class='lyr'>L{pt['layer']}</span></span>"
                f"{sparkline(pt['scores'], pt['layer'])}</a></td>")

    rows = []
    for m in models:
        c = FAMILY_COLOR.get(m["family"], "#94a3b8")
        rk = (f"{m['mean_rank']:.1f}" if m["mean_rank"] else "&mdash;")
        part = ("" if m["rank_n"] == n_prim else
                f"<sup class='cav' title='ranked on {m['rank_n']} of {n_prim} "
                f"primary tasks'>p</sup>")
        extra = ("" if m["in_paper"] else
                 "<sup class='cav' title='beyond the paper&#39;s 12-model study'>+</sup>")
        ncls = "mname" if m["in_paper"] else "mname mmuted"
        cells = [f"<tr style='--fam:{c}'>",
                 f"<td class='{ncls}' title='{m['id']}'>{m['display']}{extra}</td>",
                 f"<td><span class='dot'></span>{m['family']}</td>",
                 (lambda p: (f"<td class='num' data-val='{p[1]}'"
                             + (f" title='{p[2]}'" if len(p) > 2 else "")
                             + f">{p[0]}{'<sup class=cav>*</sup>' if len(p) > 2 else ''}</td>")
                  if p else
                  "<td class='num na' data-val='-1' title='not verified'>&mdash;</td>")(PARAMS.get(m['id'])),
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




DATA_README = """# Atlas data

Raw records behind every number on the site. One folder per model.

## Protocol (applies to every downstream number)
- Frozen encoder; representations from every layer (0 = first hidden state).
- Probe: MLP with one 512-unit hidden layer, MARBLE constrained-track settings,
  identical splits across models; single seed (1234) for all current numbers.
- New cells are filled append-only as they are evaluated (see `coverage.json`);
  newly added cells run three seeds, and multi-seed cells will report mean +/- sd.
- Scores can differ from the numbers in each model's own paper: papers may use
  different probes, model variants (e.g. the Myna paper's key result is for
  Myna-Vertical, not the Myna-Base evaluated here), or splits.

## Files
- `results/<model>/downstream.json` -- per-layer scores per task (the raw curves).
- `results/<model>/fusion.json` -- one score per (task, multi-layer method);
  tagging tasks scored as mean of AP and AUROC, matching the published protocol.
- `registry/models.yaml`, `registry/tasks.yaml` -- open vocabularies; aliases and
  excluded metrics (with reasons) live here.
- `coverage.json` -- which (model, task) cells are not yet evaluated.

Records are self-describing (schema_version, readout spec, provenance) and
append-only: existing records are never rewritten.
"""


def write_data_readme():
    (OUT / "README.md").write_text(DATA_README)

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
    write_data_readme()
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


# ---------------- stage 1c: fusion / multi-layer probe results ----------------
import csv

PROBE_DIR = Path("/home/akanatas/projects/layerbylayer/MARBLE/output")

# canonical task -> test column map, verbatim from
# MARBLE/scripts/analyze_multilayer_vs_proxy.py:534 (the published protocol;
# tagging = avg(AP, AUROC) there). Tempo columns are never in this map.
FUSION_TASKS = {
    "GS":               ("key", "test/weighted_score"),
    "NSynthP":          ("nsynth_pitch", "test/file_acc"),
    "NSynthI":          ("nsynth_instrument", "test/file_acc"),
    "GTZANGenre":       ("genre", "test/file_acc"),
    "GTZANBeatTracking": ("beat_f1", "test/beat_f1"),
    "EMO":              ("emo_r2", "test/r2"),
    "HXMSA":            ("hxmsa", "test/acc"),
    "MTT":              ("mtt", ("test/ap", "test/auroc")),
    "MTGGenre":         ("mtg_genre", ("test/ap", "test/auroc")),
    "MTGInstrument":    ("mtg_instrument", ("test/ap", "test/auroc")),
    "MTGMood":          ("mtg_mood", ("test/ap", "test/auroc")),
    "MTGTop50":         ("mtg_top50", ("test/ap", "test/auroc")),
    "ChordsACE":        ("chords_ace_mirex", "test/mirex"),
}
FUSION_EXTRA_COLS = {"GTZANBeatTracking": [("downbeat_f1", "test/downbeat_f1")],
                     "ChordsACE": [("chords_ace_root", "test/root"),
                                   ("chords_ace_thirds", "test/thirds")]}
FUSION_VARIANTS = {"uniform_avg", "softmax_avg", "learned_avg", "stack", "mlpreduce",
                   "hconv", "attentive", "attentive_ciernik",
                   "topk_avg", "topk_stack", "top5_avg", "top5_stack"}
PROBE_MODEL_MAP = {"mert-95M": "mert-v1-95M", "mert-330M": "mert-v1-330M",
                   "maest-30s-discogs-pw": "maest", "music-flamingo": "musicflamingo"}
# extraction-protocol variants: never merged into the canonical rows (4aa)
SKIP_MODEL_TOKENS = ("-centered", "-lastn", "maest-10s")


def _last_test_row(probe_dir):
    """Latest lightning version dir whose metrics.csv carries test/ columns."""
    versions = sorted(probe_dir.glob("lightning_logs/version_*"),
                      key=lambda p: int(p.name.split("_")[1]), reverse=True)
    for v in versions:
        f = v / "metrics.csv"
        if not f.exists():
            continue
        with f.open() as fh:
            r = list(csv.DictReader(fh))
        if r and any(k.startswith("test/") for k in r[0]):
            vals = {}
            for row in r:                       # last non-empty value per test col
                for k, x in row.items():
                    if k.startswith("test/") and x not in ("", None):
                        vals[k] = float(x)
            if vals:
                return vals, v.name
    return None, None


def export_fusion():
    parsed, skipped_models, missing_col, written = 0, set(), [], set()
    per_model = {}
    for d in sorted(PROBE_DIR.glob("probe.*")):
        parts = d.name.split(".")
        task_probe = parts[1]
        rest = ".".join(parts[2:])              # model token may contain dots (yue-0.5b)
        variant = rest.rsplit(".", 1)[-1]
        token = rest[: -(len(variant) + 1)]
        if task_probe not in FUSION_TASKS or variant not in FUSION_VARIANTS:
            continue
        if any(s in token for s in SKIP_MODEL_TOKENS):
            skipped_models.add(token)
            continue
        model = PROBE_MODEL_MAP.get(token, token)
        if model not in MODELS:
            skipped_models.add(token)
            continue
        vals, vdir = _last_test_row(d)
        if not vals:
            continue
        base_task, col = FUSION_TASKS[task_probe]
        recs = []
        if isinstance(col, tuple):              # published tagging protocol
            if all(c in vals for c in col):
                recs.append({"task": base_task, "metric_spec": "avg(ap,auroc)x100",
                             "value": round(100 * (vals[col[0]] + vals[col[1]]) / 2, 2)})
                recs.append({"task": base_task + "_ap", "metric_spec": "ap x100",
                             "value": round(100 * vals[col[0]], 2)})
                recs.append({"task": base_task + "_auroc", "metric_spec": "auroc x100",
                             "value": round(100 * vals[col[1]], 2)})
            else:
                missing_col.append(d.name)
        elif col in vals:
            recs.append({"task": base_task, "metric_spec": col + " x100",
                         "value": round(100 * vals[col], 2)})
        else:
            missing_col.append(d.name)
        for xt in FUSION_EXTRA_COLS.get(task_probe, []):
            if xt[1] in vals:
                recs.append({"task": xt[0], "metric_spec": xt[1] + " x100",
                             "value": round(100 * vals[xt[1]], 2)})
        for rec in recs:
            rec.update({"variant": variant, "source": d.name, "version": vdir,
                        "protocol": "published-map"})
            per_model.setdefault(model, []).append(rec)
            parsed += 1
    for model, recs in per_model.items():
        out = OUT / "results" / model / "fusion.json"
        if out.exists():
            continue                            # append-only
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "model": model,
            "readout_family": "multi-layer fusion probes",
            "note": "one score per (task, variant); fusion collapses the layer axis",
            "exported": str(date.today()), "records": recs}, indent=1))
        written.add(model)
    print(f"fusion: {parsed} records -> {len(written)} models "
          f"| skipped tokens: {sorted(skipped_models)} | missing cols: {len(missing_col)}")
    return per_model


if __name__ == "__main__" and "--fusion" in sys.argv:
    fm = export_fusion()
    # RECONCILIATION GATE (4aa anchor): GS x mert-v1-330M x topk_avg == 63.64
    anchor = [r for r in fm.get("mert-v1-330M", [])
              if r["task"] == "key" and r["variant"] == "topk_avg"]
    print("ANCHOR GS/mert-330M/topk_avg:", anchor[0]["value"] if anchor else "MISSING",
          "(expect 63.64)")


# ------------- stage 1c views: Selection & Fusion panel + table hover -------------

FUSION_GROUPS = [
    ("Reference", ["oracle", "middle", "last"]),
    ("Proxy-guided fusion (label-free layer choice)",
     ["topk_avg", "topk_stack", "top5_avg", "top5_stack"]),
    ("All-layer fusion (non-trainable)", ["uniform_avg", "stack"]),
    ("Trainable fusion", ["learned_avg", "hconv", "attentive_ciernik"]),
]
VARIANT_LABEL = {
    "oracle": "Oracle (best single layer)", "middle": "Middle layer",
    "last": "Last layer",
    "topk_avg": "Top-3 avg", "topk_stack": "Top-3 concat",
    "top5_avg": "Top-5 avg", "top5_stack": "Top-5 concat",
    "uniform_avg": "All-layer avg", "stack": "All-layer concat",
    "learned_avg": "Weighted sum", "hconv": "HConv",
    "attentive_ciernik": "Attentive",
}
# fusion-task id -> how to derive the SAME-METRIC single-layer curve (4aa rule)
FUSION_BASE = {  # task -> list of downstream record tasks to average per layer
    "mtt": ["mtt_ap", "mtt_auroc"], "mtg_genre": ["mtg_genre_ap", "mtg_genre_auroc"],
    "mtg_instrument": ["mtg_instrument_ap", "mtg_instrument_auroc"],
    "mtg_mood": ["mtg_mood_ap", "mtg_mood_auroc"],
    "mtg_top50": ["mtg_top50_ap", "mtg_top50_auroc"],
}
FUSION_TASK_LABEL = {
    "key": "Key (GiantSteps)", "nsynth_pitch": "Pitch (NSynth)",
    "nsynth_instrument": "Instrument (NSynth)", "genre": "Genre (GTZAN)",
    "beat_f1": "Beat (GTZAN)", "downbeat_f1": "Downbeat (GTZAN)",
    "emo_r2": "Emotion (EMO)", "hxmsa": "Structure (Harmonix)",
    "chords_ace_mirex": "Chords MIREX (ChordsACE)",
    "chords_ace_root": "Chords root (ChordsACE)",
    "chords_ace_thirds": "Chords thirds (ChordsACE)",
    "mtt": "Tagging (MTT, avg AP/AUROC)", "mtg_genre": "MTG-Genre (avg AP/AUROC)",
    "mtg_instrument": "MTG-Instr (avg AP/AUROC)", "mtg_mood": "MTG-Mood (avg AP/AUROC)",
    "mtg_top50": "MTG-Top50 (avg AP/AUROC)",
}


def _single_layer_curve(down_recs, task):
    """Same-metric per-layer curve for a fusion task (avg of components for tagging)."""
    comp = FUSION_BASE.get(task, [task])
    curves = [r["layers"] for r in down_recs if r["task"] in comp]
    if len(curves) != len(comp):
        return None
    return [sum(v) / len(curves) for v in zip(*curves)]


def render_fusion_panel():
    data = {}                                   # task -> model -> variant -> value
    models_here = []
    for model_id in MODELS:
        f = OUT / "results" / model_id / "fusion.json"
        if not f.exists():
            continue
        models_here.append(model_id)
        fus = json.loads(f.read_text())["records"]
        down = json.loads((OUT / "results" / model_id / "downstream.json")
                          .read_text())["records"]
        for r in fus:
            if r["task"].endswith("_ap") or r["task"].endswith("_auroc"):
                continue                        # components; panel shows the avg
            data.setdefault(r["task"], {}).setdefault(model_id, {})[r["variant"]] = r["value"]
        for task in list(data.keys()):
            if model_id in data[task] and "oracle" not in data[task][model_id]:
                curve = _single_layer_curve(down, task)
                if curve:
                    data[task][model_id]["oracle"] = round(max(curve), 2)
                    data[task][model_id]["middle"] = round(curve[len(curve) // 2], 2)
                    data[task][model_id]["last"] = round(curve[-1], 2)

    order = [t for t in FUSION_TASK_LABEL if t in data]
    sections = []
    for ti, task in enumerate(order):
        cols = [m for m in models_here if m in data[task]]
        head = ("<tr><th>Method</th>" +
                "".join(f"<th class='num'>{MODELS[m]['display']}</th>" for m in cols) +
                "</tr>")
        body = []
        for gname, variants in FUSION_GROUPS:
            body.append(f"<tr class='ghead'><td colspan='{len(cols)+1}'>{gname}</td></tr>")
            for v in variants:
                if not any(v in data[task].get(m, {}) for m in cols):
                    continue
                tds = []
                for m in cols:
                    val = data[task].get(m, {}).get(v)
                    orc = data[task].get(m, {}).get("oracle")
                    if val is None:
                        tds.append("<td class='num na'><span class='val'>&mdash;</span>"
                                   "<span class='dlt'></span></td>")
                    elif v in ("oracle", "middle", "last") or orc is None:
                        tds.append(f"<td class='num'><span class='val'>{val:.1f}</span>"
                                   f"<span class='dlt'></span></td>")
                    else:
                        d = val - orc
                        cls = " pos" if d >= 0 else ""
                        tds.append(f"<td class='num'><span class='val'>{val:.1f}</span>"
                                   f"<span class='dlt{cls}'>{d:+.1f}</span></td>")
                rcls = " class='oracle'" if v == "oracle" else ""
                body.append(f"<tr{rcls}><td>{VARIANT_LABEL[v]}</td>{''.join(tds)}</tr>")
        sections.append(
            f"<section class='ftask' id='ft-{task}'{'' if ti == 0 else ' hidden'}>"
            f"<table>{head}{''.join(body)}</table></section>")
    # per-MODEL pivot: methods x tasks for one model
    msections = []
    for model_id in models_here:
        tcols = [t for t in order if model_id in data[t]]
        head = ("<tr><th>Method</th>" +
                "".join(f"<th class='num' title='{FUSION_TASK_LABEL[t]}'>"
                        f"{TASK_LABEL.get(t, t)}</th>" for t in tcols) + "</tr>")
        body = []
        for gname, variants in FUSION_GROUPS:
            body.append(f"<tr class='ghead'><td colspan='{len(tcols)+1}'>{gname}</td></tr>")
            for v in variants:
                if not any(v in data[t].get(model_id, {}) for t in tcols):
                    continue
                tds = []
                for t in tcols:
                    val = data[t].get(model_id, {}).get(v)
                    orc = data[t].get(model_id, {}).get("oracle")
                    if val is None:
                        tds.append("<td class='num na'><span class='val'>&mdash;</span>"
                                   "<span class='dlt'></span></td>")
                    elif v in ("oracle", "middle", "last") or orc is None:
                        tds.append(f"<td class='num'><span class='val'>{val:.1f}</span>"
                                   f"<span class='dlt'></span></td>")
                    else:
                        d = val - orc
                        cls = " pos" if d >= 0 else ""
                        tds.append(f"<td class='num'><span class='val'>{val:.1f}</span>"
                                   f"<span class='dlt{cls}'>{d:+.1f}</span></td>")
                rcls = " class='oracle'" if v == "oracle" else ""
                body.append(f"<tr{rcls}><td>{VARIANT_LABEL[v]}</td>{''.join(tds)}</tr>")
        msections.append(
            f"<section class='ftask' id='fm-{model_id}' hidden>"
            f"<table>{head}{''.join(body)}</table></section>")
    mopts = "".join(f"<option value='fm-{m}'>{MODELS[m]['display']}</option>"
                    for m in models_here)
    opts = "".join(f"<option value='ft-{t}'>{FUSION_TASK_LABEL[t]}</option>"
                   for t in order)
    tpl = (Path(__file__).resolve().parent / "fusion_template.html").read_text()
    (OUT.parent / "fusion.html").write_text(
        tpl.replace("<!--OPTIONS-->", opts)
           .replace("<!--MOPTIONS-->", mopts)
           .replace("<!--SECTIONS-->", "\n".join(sections + msections))
           .replace("{{DATE}}", str(date.today())))
    print(f"fusion.html: {len(order)} tasks x {len(models_here)} models")


if __name__ == "__main__" and "--fusion" in sys.argv:
    render_fusion_panel()
