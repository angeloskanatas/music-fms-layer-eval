#!/usr/bin/env python3
"""Export per-layer results into data/ and render the site's pages.

One folder per model under data/results/, holding raw per-layer scores (downstream.json), label-free metrics
(metrics.json) and multi-layer fusion scores (fusion.json), with open model and task vocabularies under
data/registry/. Records are added after existing ones and replaced only for the models and metrics listed in
REPLACE_DOWNSTREAM and REPLACE_METRICS. `--render` rebuilds every page from data/ alone.
"""

import json
import os
import re
import sys
from datetime import date
from html import escape
from pathlib import Path

SCHEMA_VERSION = "1.0.0"
SRC = Path(os.environ.get("ATLAS_SRC", "/home/akanatas/projects/layerbylayer/layerbylayer/output"))
OUT = Path(__file__).resolve().parent.parent / "data"
BUILD_DATE = str(date.today())  # "results as of" on every page; --render sets it from the records

# canonical model id -> downstream filename stem. Variants reference their base
# via variant_of; paper-replication files use the source paper's task keys.
# display = the paper's official model name (tex Table 1 / source papers).
# readout_of groups architecture-specific readouts of ONE model (MT2's
# multi-class-token heads, cf. seq-JEPA agg/glance) — views merge these rows.
# hidden = kept out of the views.
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
    "myna-paper":      {"file": "myna_paper", "display": "Myna-Base (paper values)", "family": "contrastive",
                        "in_paper": False, "variant_of": "myna", "hidden": True, "pending": True,
                        },
    "mert-v2-30s":     {"file": "mert-v2-30s", "display": "MERT-v2-30s", "family": "masked", "in_paper": False},
    "mert-v2-fullsong": {"file": "mert-v2-fullsong", "display": "MERT-v2-FullSong", "family": "masked",
                        "in_paper": False},
    # audio-language models: kept out of the views until their own analysis is published
    "qwen2audio-instruct": {"file": "qwen2audio-instruct", "display": "Qwen2-Audio-Instruct",
                        "family": "audio-llm", "in_paper": False, "hidden": True},
    "musicflamingo":   {"file": "musicflamingo", "display": "Music-Flamingo",
                        "family": "audio-llm", "in_paper": False, "hidden": True},
}
# Models added after the paper (scripts/generate_model_pipeline.py). pending = exported once its
# downstream file exists, skipped silently before. 2D encoders: the mean over all time-frequency
# patches and the frequency-preserving readout (frequency patches concatenated per frame, then the
# time mean) are two readouts of one model, merged in the views as MT2's heads are.
_NEW = [("matpac-plus-music", "MATPAC++ (music)"), ("matpac-plus-as", "MATPAC++ (AudioSet)"),
        ("pupum2d-tiny", "PupuM2D-Tiny"), ("pupum2d-small", "PupuM2D-Small"),
        ("pupum2d-base", "PupuM2D-Base"), ("pupum2d-large", "PupuM2D-Large"),
        ("pupum2d-huge", "PupuM2D-Huge")]
# held back from the views until their task set is complete (MATPAC++: MTG and DimSim still running)
HOLD = {"matpac-plus-music", "matpac-plus-as"}
# not written to data/ at all (the folder is public): the audio-language models and the held-back models
UNPUBLISHED = ({"qwen2audio-instruct", "musicflamingo", "muq-paper", "musicfm-paper", "myna-paper"} | HOLD
               | {f"{m}-linpre" for m in HOLD})
for _mid, _display in _NEW:
    MODELS[_mid] = {"file": _mid, "display": _display, "family": "masked", "in_paper": False,
                    "pending": True, "readout_of": _mid, "readout_label": "patch mean",
                    **({"hidden": True} if _mid in HOLD else {})}
    MODELS[f"{_mid}-linpre"] = {"file": f"{_mid}-linpre", "display": _display, "family": "masked",
                                "in_paper": False, "pending": True, "readout_of": _mid,
                                "readout_label": "frequency-preserving",
                                **({"hidden": True} if _mid in HOLD else {})}
# Checkpoint variants of a site model (scripts/generate_variant_configs.py): the base model's family and
# protocol, other checkpoints; separate rows (not readouts), pending until their downstream file exists.
_VARIANTS = [("discogs-maest-5s-pw-129e", "MAEST (5 s)", "maest"),
             ("discogs-maest-10s-pw-129e", "MAEST (10 s)", "maest"),
             ("discogs-maest-20s-pw-129e", "MAEST (20 s)", "maest"),
             ("discogs-maest-10s-fs-129e", "MAEST (10 s, from scratch)", "maest"),
             ("discogs-maest-10s-dw-75e", "MAEST (10 s, DeiT init)", "maest"),
             ("discogs-maest-30s-pw-73e-ts", "MAEST (30 s, teacher-student)", "maest"),
             ("discogs-maest-30s-pw-129e-519l", "MAEST (30 s, 519 labels)", "maest")]
for _mid, _display, _base in _VARIANTS:
    MODELS[_mid] = {"file": _mid, "display": _display, "family": MODELS[_base]["family"], "in_paper": False,
                    "pending": True}

# alias -> canonical task key (drift found in the source files)
TASK_ALIASES = {"emotion_r2a": "emo_r2a", "emotion_r2v": "emo_r2v"}

# Metrics that exist in probe outputs but must NEVER be exported or rendered.
# The GTZAN beat probes train with loss weights [1.0, 1.0, 0.0]: the tempo head is not trained, although its
# metrics are logged, so they are never exported.
EXCLUDED_METRICS = {
    "tempo_acc1": "untrained head (beat-probe tempo loss weight = 0.0)",
    "tempo_acc2": "untrained head (beat-probe tempo loss weight = 0.0)",
    "tempo_mae":  "untrained head (beat-probe tempo loss weight = 0.0)",
}

# Notes on how a task's metric is aggregated, carried on its records and shown as a tooltip:
METRIC_CAVEATS: dict = {}

# task family per the paper's Sec 3.3 grouping (unknown keys stay unassigned;
# the site renders them generically -- open-vocabulary rule)
TASK_FAMILY = {
    "key": "tonal", "nsynth_pitch": "tonal",
    "chords_ace_root": "tonal", "chords_ace_thirds": "tonal",
    "chords_ace_triads": "tonal", "chords_ace_mirex": "tonal",
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
EXPORT_TASKS = set(TASK_FAMILY)
# what each task and metric is, as the pages and data/registry/tasks.yaml state it
TASK_DESC = {
    "key": "Key estimation, GiantSteps (mir_eval weighted score)",
    "nsynth_pitch": "Pitch classification, NSynth (accuracy)",
    "chords_ace_mirex": "Chord recognition, Chords1217 (mir_eval MIREX score)",
    "chords_ace_root": "Chord recognition, Chords1217 (mir_eval root score)",
    "chords_ace_thirds": "Chord recognition, Chords1217 (mir_eval thirds score)",
    "chords_ace_triads": "Chord recognition, Chords1217 (mir_eval triads score)",
    "beat_f1": "Beat tracking, GTZAN (F-measure)",
    "downbeat_f1": "Downbeat tracking, GTZAN (F-measure)",
    "nsynth_instrument": "Instrument family classification, NSynth (accuracy)",
    "genre": "Genre classification, GTZAN (accuracy)",
    "hxmsa": "Structural function recognition, Harmonix (clip-level accuracy)",
    "emo_r2": "Emotion regression, Emomusic (R2 of arousal and valence)",
    "emo_r2a": "Emotion regression, Emomusic (R2 of arousal)",
    "emo_r2v": "Emotion regression, Emomusic (R2 of valence)",
    "mtt_ap": "Tagging, MagnaTagATune (macro-averaged AP)",
    "mtt_auroc": "Tagging, MagnaTagATune (macro-averaged AUROC)",
    "mtg_genre_ap": "Genre tagging, MTG-Jamendo (macro-averaged AP)",
    "mtg_genre_auroc": "Genre tagging, MTG-Jamendo (macro-averaged AUROC)",
    "mtg_instrument_ap": "Instrument tagging, MTG-Jamendo (macro-averaged AP)",
    "mtg_instrument_auroc": "Instrument tagging, MTG-Jamendo (macro-averaged AUROC)",
    "mtg_mood_ap": "Mood and theme tagging, MTG-Jamendo (macro-averaged AP)",
    "mtg_mood_auroc": "Mood and theme tagging, MTG-Jamendo (macro-averaged AUROC)",
    "mtg_top50_ap": "Top-50 tagging, MTG-Jamendo (macro-averaged AP)",
    "mtg_top50_auroc": "Top-50 tagging, MTG-Jamendo (macro-averaged AUROC)",
    "dimsim_acc_centered": "Perceptual similarity, DimSim (triplet accuracy, centered cosine)",
    "dimsim_acc_cosine": "Perceptual similarity, DimSim (triplet accuracy, cosine)",
    "dimsim_acc_l2": "Perceptual similarity, DimSim (triplet accuracy, Euclidean distance)",
}                 # the tasks the pages show; no other record is written to data/

# The probe every downstream number came from (MARBLE constrained track).
# seed: the probe runs behind the current numbers are single-seed (Lightning
# seed_everything: 1234, verified in the probe configs). The schema supports
# multiple records per (task, readout) differing by seed; the build renders
# mean +/- sd once n_seeds > 1.
READOUT = {
    "id": "mlp512-marble",
    "probe": "MLP, one 512-unit hidden layer",
    "protocol": "MARBLE constrained track, frozen encoder, per-layer",
    "seed": 1234,
    "n_seeds": 1,
}
DEFAULT_READOUT = READOUT["id"]
# Further readouts of the same tasks, one downstream file each (scripts/aggregate_downstream.py
# --suffix): same protocol, data and seed, another probe. The views show DEFAULT_READOUT;
# the results table and the Layer Explorer offer the others as a choice of probe once they have records.
READOUTS = {
    DEFAULT_READOUT: READOUT,
    "linear-marble": {**READOUT, "id": "linear-marble", "probe": "linear head"},
    "maeb-probe-free": {"id": "maeb-probe-free",
                        "probe": "none: k-means, cosine ranking or 8-shot logistic regression (MAEB protocol)",
                        "protocol": "frozen encoder, per-layer, time-mean embeddings"},
    "compositionality-probe-free": {"id": "compositionality-probe-free",
                                    "probe": "none: closed-form audio compositionality scores (A-COAT, A-TRE)",
                                    "protocol": "frozen encoder, per-layer, time-mean embeddings"},
}
# downstream file suffix -> readout of every record in that file. Only these files are read: the
# feature-normalisation variants of the probes (suffixes ending in -bn or -standardize) are not exported.
SOURCE_READOUT = {"": DEFAULT_READOUT, "_linear": "linear-marble"}
# linear-probe files named after the metric prefix rather than the model id
LINEAR_FILE = {"mert-v1-95M": "mert-95M_linear", "mert-v1-330M": "mert-330M_linear", "clap": "clap_linear-laion"}
# the Jacobian effective rank stays off the site until it covers every model
SHOW_JER = False
# tasks with their own protocol in the main file (records exported before this rule keep theirs)
TASK_READOUT = {"maeb_": "maeb-probe-free", "audiocomp_": "compositionality-probe-free"}
# The probe-free page (probefree.html): the header of each protocol's column group, and the decimals of a
# protocol's scores (compositionality scores are cosines in [-1, 1]; every other score is in percent).
READOUT_LABEL = {"maeb-probe-free": "MAEB", "compositionality-probe-free": "Compositionality",
                 DEFAULT_READOUT: "MLP", "linear-marble": "Linear"}
SCORE_DIGITS = {"compositionality-probe-free": 3}


def task_readout(task_id, default=DEFAULT_READOUT):
    return next((r for prefix, r in TASK_READOUT.items() if task_id.startswith(prefix)), default)


def primary_records(records):
    """The records the views show: per task its primary readout (DEFAULT_READOUT, or the task's own
    protocol), in file order."""
    return [r for r in records if r.get("readout", DEFAULT_READOUT) in (DEFAULT_READOUT, task_readout(r["task"]))]


def probe_records(records, readout):
    """The records of one further probe: that readout, on the tasks without a protocol of their own."""
    return [r for r in records if r.get("readout", DEFAULT_READOUT) == readout
            and task_readout(r["task"]) == DEFAULT_READOUT]


def offered_probes(docs):
    """The registered probes a page can offer next to DEFAULT_READOUT, in registry order: those with at
    least one record in the given downstream files. Empty when every record is a primary one."""
    return [r for r in READOUTS if r != DEFAULT_READOUT and r not in TASK_READOUT.values()
            and any(probe_records(d["records"], r) for d in docs)]


def view_docs():
    """The downstream files behind the views, by model id (hidden models stay data-only)."""
    docs = {}
    for model_id, spec in MODELS.items():
        f = OUT / "results" / model_id / "downstream.json"
        if not spec.get("hidden") and f.exists():
            docs[model_id] = json.loads(f.read_text())
    return docs


def has_probe_free():
    """Whether the probe-free page exists (render_atlas_table writes it): a model of the views has a record
    of a task with a protocol of its own."""
    return any(task_readout(r["task"]) != DEFAULT_READOUT
               for d in view_docs().values() for r in primary_records(d["records"]))


def readout_label(readout):
    """What a page calls a readout: its short name, else the registry's probe text."""
    label = READOUT_LABEL.get(readout) or READOUTS[readout]["probe"]
    return label[:1].upper() + label[1:]

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

VARIANT_LABEL = {
    "oracle": "Best single layer", "middle": "Middle layer",
    "last": "Last layer",
    "topk_avg": "Top 3, averaged", "topk_stack": "Top 3, concatenated",
    "top5_avg": "Top 5, averaged", "top5_stack": "Top 5, concatenated",
    "uniform_avg": "Average", "stack": "Concatenation",
    "learned_avg": "Weighted sum", "hconv": "Hierarchical convolution",
    "attentive_ciernik": "Attention over layers",
}
# Short display labels for table headers / selectors (full description from
# the registry rides on the title attribute). Machine keys never face users.
TASK_LABEL = {
    "key": "Key", "nsynth_pitch": "Pitch", "chords_ace_mirex": "Chords",
    "chords_ace_root": "Chords root", "chords_ace_thirds": "Chords thirds",
    "chords_ace_triads": "Chords triads",
    "beat_f1": "Beat", "downbeat_f1": "Downbeat",
    "nsynth_instrument": "Instrument",
    "genre": "Genre", "hxmsa": "Structure",
    "emo_r2": "Emotion", "emo_r2a": "Arousal", "emo_r2v": "Valence",
    "mtt_ap": "MTT", "mtt_auroc": "MTT AUROC",
    "mtg_genre_ap": "MTG genre", "mtg_genre_auroc": "MTG genre AUROC",
    "mtg_instrument_ap": "MTG instrument", "mtg_instrument_auroc": "MTG instrument AUROC",
    "mtg_mood_ap": "MTG mood", "mtg_mood_auroc": "MTG mood AUROC",
    "mtg_top50_ap": "MTG top 50", "mtg_top50_auroc": "MTG top 50 AUROC",
    "dimsim_acc_centered": "Similarity", "dimsim_acc_cosine": "Similarity cosine",
    "dimsim_acc_l2": "Similarity L2",
    # fusion-only averaged ids (mean of AP and AUROC)
    "mtt": "MTT", "mtg_genre": "MTG genre", "mtg_instrument": "MTG instrument",
    "mtg_mood": "MTG mood", "mtg_top50": "MTG top 50",
    # probe-free tasks (MAEB; audio compositionality); no TASK_FAMILY yet, so the results table
    # leaves them out: the probe-free page and the Layer Explorer list them
    "maeb_GTZANGenreClustering_vmeasure": "GTZAN genre clustering",
    "maeb_MusicGenreClustering_vmeasure": "Music genre clustering",
    "maeb_GTZANAudioReranking_map": "GTZAN reranking",
    "maeb_JamAltArtistA2ARetrieval_ndcg10": "Artist retrieval",
    "maeb_BeijingOpera_acc8shot": "Beijing opera (8-shot)",
    "maeb_MridinghamTonic_acc8shot": "Mridangam tonic (8-shot)",
    "maeb_MridinghamStroke_acc8shot": "Mridangam stroke (8-shot)",
    "audiocomp_acoat": "A-COAT", "audiocomp_atre": "A-TRE", "audiocomp_atre_centered": "A-TRE centred",
}

PARAMS = {"mert-v1-95M": ("95M", 95), "mert-v1-330M": ("330M", 330),
     "musicfm": ("330M", 330), "muq": ("310M", 310),
     "omar-rq-multifeature-25hz": ("580M", 580),
     "musicgen-small": ("300M", 300), "musicgen-medium": ("1.5B", 1500),
     "musicgen-large": ("3.3B", 3300), "yue-0.5b": ("0.5B", 500),
     "yue-7b": ("7B", 7000), "clap": ("73M", 73), "myna": ("22M", 22),
     # further models, from their model cards, papers or checkpoints:
     "maest": ("87M", 87, "released checkpoint, including its classification head"),
     "mt2": ("5.3M", 5.3, "backbone of the released checkpoint"),
     "omar-rq-base": ("580M", 580, "Conformer of the released checkpoint, 24 layers, width 1024"),
     "mert-v2-30s": ("632M", 632, "all weights of the released checkpoint"),
     "mert-v2-fullsong": ("632M", 632, "all weights of the released checkpoint"),
     "qwen2audio-instruct": ("637M", 637,
        "the probed audio tower (Whisper-derived, 32 layers — matches our 33 "
        "extracted states); full system 8.2B"),
     "musicflamingo": ("637M", 637,
        "the probed AF-Whisper tower (32 layers — matches our extracted states); "
        "full system 8.3B"),
     # counted from the released checkpoints as the MARBLE wrappers load them, 2026-10-10
     "matpac-plus-music": ("85M", 85.4, "patch embedding and the 12-block encoder, counted from the checkpoint"),
     "matpac-plus-as": ("85M", 85.4, "patch embedding and the 12-block encoder, counted from the checkpoint"),
     "pupum2d-tiny": ("7.1M", 7.1, "patch embedding and 12 EVA blocks (SwiGLU), counted from the checkpoint"),
     "pupum2d-small": ("28M", 28.4, "patch embedding and 12 EVA blocks (SwiGLU), counted from the checkpoint"),
     "pupum2d-base": ("113M", 113.5, "patch embedding and 12 EVA blocks (SwiGLU), counted from the checkpoint"),
     # MAEST variants: the 30-s architecture with a shorter time-position table (31, 62, 125 columns of 768
     # instead of 187), so 0.12M, 0.10M and 0.05M fewer than the 30-s model (counted 2026-10-10)
     "discogs-maest-5s-pw-129e": ("87M", 86.8, "the 30-s checkpoint's architecture with a 5-s position table"),
     "discogs-maest-10s-pw-129e": ("87M", 86.8, "the 30-s checkpoint's architecture with a 10-s position table"),
     "discogs-maest-20s-pw-129e": ("87M", 86.9, "the 30-s checkpoint's architecture with a 20-s position table"),
     # the other two initialisations of the 10-s model: the same architecture as discogs-maest-10s-pw-129e
     "discogs-maest-10s-fs-129e": ("87M", 86.8, "the 10-s architecture, trained from scratch"),
     "discogs-maest-10s-dw-75e": ("87M", 86.8, "the 10-s architecture, initialised from DeiT (75 epochs)"),
     # the other two 30-s checkpoints: the site model's encoder (its count is used), other training
     "discogs-maest-30s-pw-73e-ts": ("87M", 87, "the 30-s architecture, teacher-student training (73 epochs)"),
     "discogs-maest-30s-pw-129e-519l": ("87M", 87, "the 30-s architecture, trained on 519 labels (larger output layer, unused here)")}

# Family identity colors — validated categorical palette (dataviz skill,
# ALL CHECKS PASS on white; identity is never color-alone: names are printed).
FAMILY_COLOR = {
    "masked": "#2a78d6", "autoregressive": "#eb6834", "contrastive": "#1baf7a",
    "audio-llm": "#eda100", "supervised": "#e87ba4",
}
FAMILY_ORDER = ["masked", "autoregressive", "contrastive", "audio-llm", "supervised"]
# user-facing paradigm names (paper's terms); machine tokens never face users
FAMILY_LABEL = {"masked": "Masked", "autoregressive": "Autoregressive",
                "contrastive": "Contrastive", "audio-llm": "Audio-Language",
                "supervised": "Supervised"}
TASK_FAMILY_ORDER = ["tonal", "rhythm", "timbre", "semantic", "similarity"]


# Models whose records follow their source file (append-only otherwise): a record is replaced when the source
# file's scores for its (task, readout) differ from it.
REPLACE_DOWNSTREAM = {"myna", "clap"}
# Chord scores: mir_eval scores against the annotated segments (keys *_annotated) of the MLP and linear probes.
CHORD_FILES = {
    "mert-v1-95M": ("mert-95M_layerwise-onthefly", "mert-95M_linear"),
    "mert-v1-330M": ("mert-330M_layerwise-onthefly-full", "mert-330M_linear-full"),
    "muq": ("muq_layerwise-onthefly", "muq_linear"),
    "musicfm": ("musicfm_layerwise-onthefly", "musicfm_linear"),
    "omar-rq-multifeature-25hz": ("omar-rq-multifeature-25hz_layerwise-onthefly-full",
                                  "omar-rq-multifeature-25hz_linear-full"),
    "omar-rq-base": ("omar-rq-base_layerwise-onthefly", "omar-rq-base_linear"),
    "mt2": ("mt2_layerwise-onthefly", "mt2_linear"),
}
CHORD_TASKS = ("chords_ace_root", "chords_ace_thirds", "chords_ace_triads", "chords_ace_mirex")


def apply_chord_scores():
    """Write each model's chord records from the files of CHORD_FILES."""
    done = []
    for model_id, (mlp_file, lin_file) in CHORD_FILES.items():
        out = OUT / "results" / model_id / "downstream.json"
        if not out.exists():
            continue
        doc = json.loads(out.read_text())
        changed = 0
        for stem, readout in ((mlp_file, DEFAULT_READOUT), (lin_file, "linear-marble")):
            f = SRC / "downstream" / "data" / f"{stem}.json"
            if not f.exists():
                continue
            src = json.loads(f.read_text()).get("downstream", {})
            for task in CHORD_TASKS:
                scores = src.get(f"{task}_annotated")
                if not scores:
                    continue
                rec = {"task": task, "task_family": "tonal", "readout": readout, "layers": scores,
                       "n_layers": len(scores)}
                idx = next((i for i, r in enumerate(doc["records"])
                            if r["task"] == task and r.get("readout", DEFAULT_READOUT) == readout), None)
                if idx is None:
                    doc["records"].append(rec)
                    changed += 1
                elif doc["records"][idx]["layers"] != scores:
                    doc["records"][idx] = rec
                    changed += 1
        if changed:
            doc.setdefault("readout_specs", {DEFAULT_READOUT: READOUT}).setdefault("linear-marble",
                                                                                   READOUTS["linear-marble"])
            srcs = doc["source"] if isinstance(doc["source"], list) else [doc["source"]]
            for stem in (mlp_file, lin_file):
                rel = f"layerbylayer/output/downstream/data/{stem}.json"
                if rel not in srcs and (SRC / "downstream" / "data" / f"{stem}.json").exists():
                    srcs.append(rel)
            doc["source"] = srcs
            doc["updated"] = str(date.today())
            out.write_text(json.dumps(doc, indent=1))
            done.append(f"{model_id} ({changed})")
    print(f"chord records written: {', '.join(done) or 'none changed'}")


def export_downstream():
    """Append-only at the record level: a (task, readout) record is written once and never changed, except
    for the models of REPLACE_DOWNSTREAM; later runs add records for new tasks or readouts after the
    existing ones."""
    written, skipped, problems = [], [], []
    task_registry = {}
    for model_id, spec in MODELS.items():
        if model_id in UNPUBLISHED:
            continue
        src = SRC / "downstream" / "data" / f"{spec['file']}.json"
        if not src.exists():
            if not spec.get("pending"):
                problems.append(f"MISSING source: {src.name}")
            continue
        records, sources = [], []
        for suffix, readout in SOURCE_READOUT.items():
            stem = LINEAR_FILE.get(model_id, spec["file"] + suffix) if suffix == "_linear" else spec["file"] + suffix
            f = SRC / "downstream" / "data" / f"{stem}.json"
            if not f.exists():
                continue
            raw = json.loads(f.read_text())
            meta, tasks = raw.get("metadata", {}), raw.get("downstream", {})
            n_layers = meta.get("n_layers")
            sources.append(str(f.relative_to(SRC.parent.parent)))
            for key, scores in tasks.items():
                task_id = TASK_ALIASES.get(key, key)
                if task_id not in EXPORT_TASKS or (task_id in CHORD_TASKS and model_id in CHORD_FILES):
                    continue                    # only the tasks the pages show; chords come from CHORD_FILES
                # registry collection must not depend on the append-only skips below,
                # else a re-run rebuilds tasks.yaml empty
                desc = meta.get("tasks", {}).get(key)
                if (desc or task_id in TASK_DESC) and task_id not in task_registry:
                    task_registry[task_id] = TASK_DESC.get(task_id, desc)
                if task_id in EXCLUDED_METRICS:    # hard guard, never soft
                    problems.append(f"BLOCKED metric in source: {model_id}/{task_id} "
                                    f"({EXCLUDED_METRICS[task_id]})")
                    continue
                if n_layers and len(scores) != n_layers:
                    problems.append(f"{model_id}/{task_id}: {len(scores)} scores vs n_layers={n_layers}")
                rec = {
                    "task": task_id,
                    "task_family": TASK_FAMILY.get(task_id),
                    "readout": task_readout(task_id, readout) if suffix == "" else readout,
                    "layers": scores,              # raw per-layer, index 0..L
                    "n_layers": len(scores),
                }
                if task_id in METRIC_CAVEATS:
                    rec["caveat"] = METRIC_CAVEATS[task_id]
                records.append(rec)
        out_dir = OUT / "results" / model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "downstream.json"
        if out_file.exists():
            doc = json.loads(out_file.read_text())
            fresh = {(r["task"], r["readout"]): r for r in records} if model_id in REPLACE_DOWNSTREAM else {}
            replaced = 0
            for i, r in enumerate(doc["records"]):
                now = fresh.get((r["task"], r.get("readout", DEFAULT_READOUT)))
                if now and now["layers"] != r["layers"]:
                    doc["records"][i] = now
                    replaced += 1
            have = {(r["task"], r.get("readout", DEFAULT_READOUT)) for r in doc["records"]}
            new = [r for r in records if (r["task"], r["readout"]) not in have]
            if not (new or replaced):
                skipped.append(model_id)
                continue
            doc["records"] += new
            if new:
                specs = doc.setdefault("readout_specs", {DEFAULT_READOUT: READOUT})
                for r in new:
                    specs.setdefault(r["readout"], READOUTS[r["readout"]])
            doc["updated"] = str(date.today())
            out_file.write_text(json.dumps(doc, indent=1))
            written.append(f"{model_id} (+{len(new)} records" + (f", {replaced} replaced)" if replaced else ")"))
            if replaced:
                print(f"replaced (REPLACE_DOWNSTREAM): {model_id}, {replaced} records")
            continue
        # the views read the first record of a task, so the primary readout comes first
        records.sort(key=lambda r: r["readout"] not in (DEFAULT_READOUT, task_readout(r["task"])))
        used = list(dict.fromkeys(r["readout"] for r in records))
        out_file.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "model": model_id,
            "family": spec["family"],
            "in_paper": spec.get("in_paper", True),
            "variant_of": spec.get("variant_of"),
            "source": sources[0] if len(sources) == 1 else sources,
            "readout_spec": READOUT,
            **({"readout_specs": {r: READOUTS[r] for r in used}} if used != [DEFAULT_READOUT] else {}),
            "exported": str(date.today()),
            "records": records,
        }, indent=1))
        written.append(model_id)
    return written, skipped, problems, task_registry


def write_registries(task_registry):
    reg = OUT / "registry"
    reg.mkdir(parents=True, exist_ok=True)
    models_yaml = ["# model registry"]
    for mid, spec in MODELS.items():
        if mid in UNPUBLISHED or (spec.get("pending") and not (OUT / "results" / mid / "downstream.json").exists()):
            continue
        models_yaml.append(f"{mid}:")
        models_yaml.append(f"  family: {spec['family']}")
        models_yaml.append(f"  in_paper: {str(spec.get('in_paper', True)).lower()}")
        if spec.get("variant_of"):
            models_yaml.append(f"  variant_of: {spec['variant_of']}")
        if spec.get("note"):
            models_yaml.append(f"  note: \"{spec['note']}\"")
    (reg / "models.yaml").write_text("\n".join(models_yaml) + "\n")

    task_registry.setdefault("chords_ace_triads", "Chord triads score (ChordsACE)")
    tasks_yaml = ["# task registry; aliases map other spellings of a task to its key"]
    for tid in sorted(task_registry):
        tasks_yaml.append(f"{tid}:")
        tasks_yaml.append(f"  description: \"{task_registry[tid]}\"")
        fam = TASK_FAMILY.get(tid)
        if fam:
            tasks_yaml.append(f"  family: {fam}")
    tasks_yaml.append("aliases:")
    for a, t in TASK_ALIASES.items():
        tasks_yaml.append(f"  {a}: {t}")
    tasks_yaml.append("excluded_metrics:   # not reported")
    for m, why in EXCLUDED_METRICS.items():
        tasks_yaml.append(f"  {m}: \"{why}\"")
    (reg / "tasks.yaml").write_text("\n".join(tasks_yaml) + "\n")


def template_sections(tpl, **on):
    """Optional sections of a page template. A section sits between two marker lines, `IF NAME` and
    `END NAME`, each an HTML or /* */ comment alone on its line. It is kept (without the marker lines) when
    on[NAME] is true and dropped whole otherwise, so a page without the feature has no trace of it."""
    for name, keep in on.items():
        first, last = (rf"^[ \t]*(?:<!--|/\*){word} {name}(?:-->|\*/)[ \t]*\n" for word in ("IF", "END"))
        if not (re.search(first, tpl, re.M) and re.search(last, tpl, re.M)):
            raise ValueError(f"template has no section {name}")
        tpl = (re.sub(f"{first}|{last}", "", tpl, flags=re.M) if keep
               else re.sub(f"{first}.*?{last}", "", tpl, flags=re.M | re.S))
    return tpl


SITE_PAGES = [("index", "index.html", "Results"), ("probefree", "probefree.html", "Probe-free tasks"),
              ("explorer", "explorer.html", "Layer profiles"), ("corr", "corr.html", "Metrics"),
              ("fusion", "fusion.html", "Layer selection")]
SITE_LINKS = [("https://arxiv.org/abs/2608.14819", "Paper"), ("https://github.com/angeloskanatas/req-metrics", "Code"),
              ("https://github.com/angeloskanatas/music-fms-layer-eval/tree/master/data", "Data")]
SITE_FOOT = """<div class="site-foot">
    <p class="update">We continue to update these results with more models and downstream tasks.</p>
  </div>"""


def site_nav(page):
    """The navigation bar of every page, with the current page marked."""
    free = has_probe_free()
    here = ' class="here" aria-current="page"'
    links = "".join(f'<a href="{href}"{here if key == page else ""}>{label}</a>'
                    for key, href, label in SITE_PAGES if key != "probefree" or free)
    ext = "".join(f'<a href="{href}">{label}</a>' for href, label in SITE_LINKS)
    return (f'<nav class="site-nav"><div class="inner"><a class="brand" href="index.html">What Makes a Good Layer?</a>'
            f'{links}<span class="ext">{ext}</span></div></nav>')


def page_template(name, **on):
    """A page's template with its optional sections settled, the navigation bar and the closing line."""
    tpl = (Path(__file__).resolve().parent / f"{name}_template.html").read_text()
    tpl = template_sections(tpl, **on) if on else tpl
    return tpl.replace("<!--NAV-->", site_nav(name)).replace("<!--FOOT-->", SITE_FOOT)


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
    (best layer, family means) are computed HERE, never persisted.
    The same rows serve the further probes (atlas.html's probe choice) and the
    probe-free page; both exist only once such records do."""
    docs = view_docs()                          # all comparable models

    def table_models(probe=None):
        """The table's rows, in its order, from the primary records or from one further probe's."""
        models = []
        for model_id, d in docs.items():
            spec = MODELS[model_id]
            ff = OUT / "results" / model_id / "fusion.json"
            fus = {}
            # fusion scores are compared with the primary records of the same probe runs
            if ff.exists() and not probe and model_id not in FUSION_EXCLUDE:
                for r in json.loads(ff.read_text())["records"]:
                    if r["task"] not in FUSION_SKIP_TASKS:
                        fus.setdefault(r["task"], {})[r["variant"]] = r["value"]
            per_task = {}
            for r in (probe_records(d["records"], probe) if probe else primary_records(d["records"])):
                t, scores = r["task"], r["layers"]
                best = max(range(len(scores)), key=lambda i: scores[i])
                per_task[t] = {"score": scores[best], "layer": best, "scores": scores,
                               "caveat": r.get("caveat"), "model": model_id, "t": t,
                               "readout_label": spec.get("readout_label"),
                               "fusion": fus.get(t), "probe": probe}
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
        # rows by paradigm, then by encoder size; a further probe lists only the models it has results for
        models = [m for m in models if m["per_task"]]
        models.sort(key=lambda m: (FAMILY_ORDER.index(m["family"]) if m["family"] in FAMILY_ORDER else 9,
                                   PARAMS.get(m["id"], ("", 1e9))[1]))
        return models

    models = table_models()
    prim_tasks = {tf: sorted(t for t, f in TASK_FAMILY.items()
                             if f == tf and t in PRIMARY_METRICS)
                  for tf in TASK_FAMILY_ORDER}
    comp_tasks = {tf: sorted(t for t, f in TASK_FAMILY.items()
                             if f == tf and t not in PRIMARY_METRICS)
                  for tf in TASK_FAMILY_ORDER}
    # column groups of a table: (group id, shown tasks, tasks behind the group's expander)
    task_groups = [(tf, prim_tasks[tf], comp_tasks[tf]) for tf in TASK_FAMILY_ORDER]

    def th_task(t, tf, hidden, first=False):
        h = " hidden" if hidden else ""
        cls = ("detail d-" + tf if hidden else "ptask") + (" gstart" if first else "")
        label = TASK_LABEL.get(t, t)
        desc = escape(task_registry.get(t, t))
        return f"<th class='{cls}' data-sort='num'{h} title='{desc}'>{label}</th>"

    def th_group(g, label, n_p, n_c, title=""):
        # the header widens over the extra columns when they are expanded (data-np, data-nc)
        exp = (f" <span class='chev' title='{n_c} more'>&#9656;</span>"
               if n_c else "")
        return (f"<th colspan='{n_p}' class='famgrp gstart{' expandable' if n_c else ''}' "
                f"data-fam='{g}' data-np='{n_p}' data-nc='{n_c}'{title}>{label}{exp}</th>")

    head_lead = ("<tr><th rowspan='2' data-sort='str'>Model</th>"
                 "<th rowspan='2' data-sort='str'>Paradigm</th>"
                 "<th rowspan='2' data-sort='num' title='Parameters of the evaluated "
                 "encoder (audio tower of two-tower models, decoder of autoregressive "
                 "models)'>Params</th>")
    head_top = [head_lead]
    head_sub = ["<tr>"]
    for tf, shown, companions in task_groups:
        head_top.append(th_group(tf, tf.capitalize(), len(shown), len(companions)))
        for i, t in enumerate(shown):
            head_sub.append(th_task(t, tf, hidden=False, first=i == 0))
        for t in companions:
            head_sub.append(th_task(t, tf, hidden=True))
    head_top.append("</tr>")
    head_sub.append("</tr>")

    def cell(pt, tf, hidden, digits=1, first=False):
        h = " hidden" if hidden else ""
        cls = (("detail d-" + tf) if hidden else "ptask") + (" gstart" if first else "")
        if pt is None:
            return f"<td class='num {cls} na'{h} title='not evaluated'>&mdash;</td>"
        cv = (f"<sup class='cav' title='{pt['caveat']}'>&dagger;</sup>"
              if pt.get("caveat") else "")
        tip = ("Open the layer curve" if not pt.get("readout_label") else
               f"Best output: {pt['readout_label']}. Open the curves of all outputs")
        probe = f"&amp;probe={pt['probe']}" if pt.get("probe") else ""
        return (f"<td class='num {cls}'{h}>"
                f"<a class='cellink' href='explorer.html?model={pt['model']}&amp;task={pt['t']}{probe}' "
                f"title='{tip}'>"
                f"<span class='sc'>{pt['score']:.{digits}f}{cv}"
                f"<span class='lyr'>L{pt['layer']}</span></span>"
                f"{sparkline(pt['scores'], pt['layer'])}</a></td>")

    def table_rows(models, groups):
        rows = []
        for m in models:
            c = FAMILY_COLOR.get(m["family"], "#94a3b8")
            cells = [f"<tr style='--fam:{c}'>",
                     f"<td class='mname' title='data/results/{m['id']}'>{m['display']}</td>",
                     f"<td><span class='dot'></span>{FAMILY_LABEL.get(m['family'], m['family'])}</td>",
                     (lambda p: (f"<td class='num' data-val='{p[1]}'"
                                 + (f" title='{p[2]}'" if len(p) > 2 else "")
                                 + f">{p[0]}</td>")
                      if p else
                      "<td class='num na' data-val='-1' title='not verified'>&mdash;</td>")(PARAMS.get(m['id']))]
            for g, shown, companions in groups:
                digits = SCORE_DIGITS.get(g, 1)
                for i, t in enumerate(shown):
                    cells.append(cell(m["per_task"].get(t), g, hidden=False, digits=digits, first=i == 0))
                for t in companions:
                    cells.append(cell(m["per_task"].get(t), g, hidden=True, digits=digits))
            cells.append("</tr>")
            rows.append("".join(cells))
        return rows

    # Probe choice: the rows of every further probe with records wait in a hidden table of the page.
    probes = offered_probes(docs.values())
    # Probe-free page: the tasks with a protocol of their own, one column group per protocol. A task with
    # a display label is a column; the other scores of a protocol sit behind its group's expander.
    seen = {t for m in models for t in m["per_task"]}
    free_groups = []
    for g in dict.fromkeys(TASK_READOUT.values()):
        own = [t for t in seen if task_readout(t) == g]
        shown = [t for t in TASK_LABEL if t in own] or sorted(own)
        if own:
            free_groups.append((g, shown, sorted(set(own) - set(shown))))

    # PROBES: the probe menu and its sentence, when a further probe has results
    tpl = page_template("index", PROBES=bool(probes))
    bands_head, bands_rows = cheatsheet_parts()
    page = (tpl.replace("<!--THEAD-->", "".join(head_top) + "".join(head_sub))
               .replace("<!--BANDS_HEAD-->", bands_head)
               .replace("<!--BANDS_ROWS-->", "\n".join(bands_rows))
               .replace("{{NTASKS}}", str(len(PRIMARY_METRICS)))
               .replace("<!--TBODY-->", "\n".join(table_rows(models, task_groups)))
               .replace("<!--PROBE_OPTIONS-->", "".join(
                   f"<option value='{p}'>{escape(readout_label(p))}</option>" for p in [DEFAULT_READOUT] + probes))
               .replace("<!--PROBE_TBODIES-->", "\n".join(
                   f"<tbody data-probe='{p}'>" + "\n".join(table_rows(table_models(p), task_groups)) + "</tbody>"
                   for p in probes))
               .replace("{{DATE}}", BUILD_DATE)
               .replace("{{NMODELS}}", str(len(models))))
    (OUT.parent / "index.html").write_text(page)

    if free_groups:
        # the registry's protocol text, without the "none: " that the page title already says
        about = {g: escape(re.sub("^none: ", "", READOUTS[g]["probe"]) + "; " + READOUTS[g]["protocol"])
                 for g, _, _ in free_groups}
        free_top, free_sub = [head_lead], ["<tr>"]
        for g, shown, companions in free_groups:
            free_top.append(th_group(g, readout_label(g), len(shown), len(companions), f" title='{about[g]}'"))
            for i, t in enumerate(shown):
                free_sub.append(th_task(t, g, hidden=False, first=i == 0))
            for t in companions:
                free_sub.append(th_task(t, g, hidden=True))
        ftpl = page_template("probefree", COMPANIONS=any(companions for _, _, companions in free_groups))
        (OUT.parent / "probefree.html").write_text(
            ftpl.replace("<!--THEAD-->", "".join(free_top) + "</tr>" + "".join(free_sub) + "</tr>")
                .replace("<!--TBODY-->", "\n".join(table_rows(models, free_groups)))
                .replace("<!--PROTOCOLS-->", "\n    &emsp; ".join(
                    f"<strong>{readout_label(g)}</strong> = {about[g]}" for g, _, _ in free_groups))
                .replace("{{DATE}}", BUILD_DATE))

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
    etpl = page_template("explorer", PROBES=bool(probes))
    (OUT.parent / "explorer.html").write_text(
        etpl.replace("/*MODELS_JSON*/", models_json)
            .replace("/*TASKS_JSON*/", tasks_json)
            .replace("/*PROBES_JSON*/", json.dumps([{"id": p, "label": readout_label(p)}
                                                    for p in [DEFAULT_READOUT] + probes]))
            .replace("/*OWN_PROTOCOL_JSON*/", json.dumps(list(TASK_READOUT)))
            .replace("{{DATE}}", BUILD_DATE))
    return len(models)




REDIRECT = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>What Makes a Good Layer?</title>
<meta http-equiv="refresh" content="0; url={target}"><link rel="canonical" href="{target}"></head>
<body><p>This page is now part of the <a href="{target}">results page</a>.</p></body></html>
"""


def render_cheatsheet(task_registry=None):
    """The old results and cheat-sheet addresses forward to the home page, which holds both tables."""
    (OUT.parent / "atlas.html").write_text(REDIRECT.format(target="index.html"))
    (OUT.parent / "layers.html").write_text(REDIRECT.format(target="index.html?view=bands"))
    return len(cheatsheet_parts()[1])


def cheatsheet_parts():
    """The three-layer bands of the home page: per (model, task family) the top-3 layers of the family-mean
    normalized curve. All numbers derived at build, never persisted. Returns (header cells, rows)."""
    models = []
    for model_id, spec in MODELS.items():
        if spec.get("hidden"):
            continue
        f = OUT / "results" / model_id / "downstream.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        per_task = {}
        for r in primary_records(d["records"]):
            if r["task"] not in PRIMARY_METRICS:
                continue
            s = r["layers"]
            per_task[r["task"]] = {"scores": s,
                                   "best": max(range(len(s)), key=lambda i: s[i])}
        models.append({"id": model_id, "display": spec["display"],
                       "family": d["family"], "in_paper": d["in_paper"],
                       "readout_of": spec.get("readout_of"),
                       "n_layers": max((r["n_layers"] for r in d["records"]), default=0),
                       "per_task": per_task})
    merged, groups = [], {}
    for m in models:                            # MT2 readout heads -> one row
        g = m["readout_of"]
        if not g:
            merged.append(m)
            continue
        if g not in groups:
            groups[g] = {**m, "id": g}
            merged.append(groups[g])
        else:
            base = groups[g]["per_task"]
            for t, pt in m["per_task"].items():
                if t not in base or max(pt["scores"]) > max(base[t]["scores"]):
                    base[t] = pt
    models = merged
    models.sort(key=lambda m: (FAMILY_ORDER.index(m["family"]), PARAMS.get(m["id"], ("", 1e9))[1]))

    def norm(s):
        lo, hi = min(s), max(s)
        rng = (hi - lo) or 1.0
        return [(x - lo) / rng for x in s]

    def depth(pt):
        return 100 * pt["best"] / max(len(pt["scores"]) - 1, 1)

    fam_tasks = {tf: [t for t in TASK_FAMILY
                      if TASK_FAMILY[t] == tf and t in PRIMARY_METRICS]
                 for tf in TASK_FAMILY_ORDER}
    famhead = "".join(
        f"<th class='num gstart' title='{', '.join(TASK_LABEL.get(t, t) for t in fam_tasks[tf])}'>"
        f"{tf.capitalize()}</th>" for tf in TASK_FAMILY_ORDER)

    def cell(m, tf):
        tasks = [t for t in fam_tasks[tf] if t in m["per_task"]]
        if not tasks:
            return "<td class='num na gstart' title='not evaluated'>&mdash;</td>"
        curves = [norm(m["per_task"][t]["scores"]) for t in tasks]
        n = len(curves[0])
        mean = [sum(c[i] for c in curves) / len(curves) for i in range(n)]
        top3 = sorted(sorted(range(n), key=lambda i: -mean[i])[:3])
        best = max(range(n), key=lambda i: mean[i])
        band = (f"L{top3[0]}&ndash;{top3[-1]}" if top3[-1] - top3[0] == 2
                else ", ".join(f"L{i}" for i in top3))
        dp = round(100 * best / max(n - 1, 1))
        det = "; ".join(f"{TASK_LABEL.get(t, t)} best L{m['per_task'][t]['best']}"
                        for t in tasks)
        return (f"<td class='num band gstart' title='top-3 layers of the family-mean curve "
                f"({len(tasks)} task{'s' if len(tasks) > 1 else ''}): {det}'>"
                f"<span class='sc'>{band}"
                f"<span class='lyr'>best L{best}, {dp}% depth</span></span>"
                f"{sparkline(mean, best)}</td>")

    rows = []
    for m in models:
        c = FAMILY_COLOR.get(m["family"], "#94a3b8")
        rows.append(
            f"<tr style='--fam:{c}'>"
            f"<td class='mname' title='data/results/{m['id']}'>{m['display']}</td>"
            f"<td><span class='dot'></span>{FAMILY_LABEL.get(m['family'], m['family'])}</td>"
            f"<td class='num'>{m['n_layers']}</td>"
            + "".join(cell(m, tf) for tf in TASK_FAMILY_ORDER) + "</tr>")

    return famhead, rows


DATA_README = """# Data

The records behind every number on the site, one folder per model.

## Protocol

- Each layer of the frozen encoder is evaluated on its own. Layer 0 is the earliest representation the model returns.
- The default probe is an MLP with one hidden layer of 512 units, trained with the settings of MARBLE's
  constrained track, on the same splits for every model and with seed 1234.
<!--IF OTHER_READOUTS-->
- Records with `readout: linear-marble` use a linear probe instead (see `readout_specs` in each file).
<!--END OTHER_READOUTS-->
- Scores can differ from those in each model's paper, which may use other probes, model variants or splits.
  For example, the key result in the Myna paper is for Myna-Vertical, not the Myna-Base evaluated here.

## Files

- `results/<model>/downstream.json`: per-layer scores for each task.
- `results/<model>/metrics.json`: per-layer label-free metrics for each (metric, variant).
- `results/<model>/fusion.json`: one score per (task, multi-layer method), and the single-layer rows they are
  compared with. Tagging tasks are scored as the mean of macro-averaged AP and AUROC.
- `registry/models.yaml` and `registry/tasks.yaml`: the model and task vocabularies.
- `coverage.json`: the (model, task) pairs not evaluated yet.

Each file names its probe and its source files.

## Contributing results

New models and tasks are welcome, by pull request or issue:

1. Evaluate every layer with the protocol above.
2. Add `results/<model>/downstream.json` in the format of the existing files, and an entry for the model in
   `registry/models.yaml` (`family`, `display`, `in_paper: false`).
3. Optionally add `results/<model>/metrics.json`, the per-layer metrics written by
   [req-metrics](https://github.com/angeloskanatas/req-metrics) (`Records.to_atlas_json`).
4. Give the checkpoint, the clip length and the pooling used for extraction.

`python scripts/export_atlas_json.py --render` rebuilds the pages from this folder alone.
"""


def write_data_readme():
    """The probe line gains its OTHER_READOUTS sentence once a linear-probe record is in data/."""
    other = any(r.get("readout", DEFAULT_READOUT) != DEFAULT_READOUT
                for f in (OUT / "results").glob("*/downstream.json") for r in json.loads(f.read_text())["records"])
    (OUT / "README.md").write_text(template_sections(DATA_README, OTHER_READOUTS=other))


def coverage_report():
    """The (model, primary task) pairs without results, written to data/coverage.json."""
    missing = {}
    for model_id in MODELS:
        f = OUT / "results" / model_id / "downstream.json"
        if not f.exists():
            continue
        have = {r["task"] for r in primary_records(json.loads(f.read_text())["records"])}
        gaps = sorted(PRIMARY_METRICS - have)
        if gaps:
            missing[model_id] = gaps
    (OUT / "coverage.json").write_text(json.dumps({
        "generated": BUILD_DATE,
        "policy": "(model, primary task) pairs not evaluated yet",
        "missing_primary_cells": missing,
        "n_missing": sum(len(v) for v in missing.values()),
    }, indent=1))
    return missing


def main():
    written, skipped, problems, task_registry = export_downstream()
    apply_chord_scores()
    append_fusion_reference()
    write_registries(task_registry)
    n = render_atlas_table(task_registry)
    print(f"index.html rendered: {n} models")
    render_cheatsheet(task_registry)
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


# ------------- label-free metric profiles -------------
# One record per (model, metric, variant); the variants used in the paper carry the tag paper-canonical.

METRIC_PREFIX = {
    "mert-v1-95M": "mert-95M", "mert-v1-330M": "mert-330M", "musicfm": "musicfm",
    "muq": "muq", "omar-rq-multifeature-25hz": "omar-rq-multifeature-25hz",
    "omar-rq-base": "omar-rq-base",
    "musicgen-small": "musicgen-small", "musicgen-medium": "musicgen-medium",
    "musicgen-large": "musicgen-large", "yue-0.5b": "yue-0.5b", "yue-7b": "yue-7b",
    "clap": "clap-laion", "myna": "myna", "maest": "maest-30s-discogs-pw",
    "mert-v2-30s": "mert-v2-30s", "mert-v2-fullsong": "mert-v2-fullsong",
    "mt2": "mt2", "mt2-cls-avg": "mt2-cls-avg",
    "mt2-cls-contrastive": "mt2-cls-contrastive", "mt2-cls-equiv": "mt2-cls-equiv",
    "qwen2audio-instruct": "qwen2-audio-7B", "musicflamingo": "music-flamingo",
    **{m: m for m, _ in _NEW},
    **{m: m for m, _, _ in _VARIANTS},
}
# (metric, dir, json key, value key, [(variant, filename token)])
METRIC_SPECS = [
    ("id", "sequence_level_intrinsic_dimension", "intrinsic_dimension", "id",
     [("gride_k8", "gride_k8"),
      ("twonn", "twonn")]),
    ("alpha", "alpha_req", "alpha_req", "alpha",
     [("default", None)]),
    ("effective_rank", "sequence_level_effective_rank", "effective_rank",
     "effective_rank", [("default", None)]),
    ("anisotropy", "sequence_level_anisotropy", "anisotropy", "anisotropy",
     [("spectral", "spectral"),
      ("cosine", "cosine")]),
    ("infonce", "infonce", "infonce", "raw",
     [("default", None)]),
    ("lidar", "lidar", "lidar", "effective_rank",
     [("default", ("effective_rank", None))]),
    # the signed angle where a model has it; otherwise the angle folded to [0, pi/2], negated in the views
    ("curvature", "curvature", "curvature", "raw",
     [("default", ("signed", None))]),
]
# A token is a filename token or a tuple of them, tried in order (None: no token).
# Records replaced when the file the exporter picks differs from the stored source, dropped when no file
# qualifies any more (append-only otherwise).
REPLACE_RECORDS = {("curvature", "default")}
# models whose metrics.json is rebuilt from its source files on every export
REPLACE_METRICS = {"clap"}
# the metrics of the released library; points files may hold records of metrics it no longer has
LIBRARY_METRICS = {
    "alignment", "alpha_req", "anisotropy/cosine", "anisotropy/spectral", "cka", "cls_patch_cosine",
    "cluster_quality", "cycle_knn", "dense_structure", "dime", "effective_rank", "eigenvalue_early_enrichment",
    "embedding_norm", "gaussianity", "infonce", "information_imbalance", "intrinsic_dimension/gride",
    "intrinsic_dimension/mle", "intrinsic_dimension/mlid", "intrinsic_dimension/mst", "intrinsic_dimension/twonn",
    "jacobian_effective_rank", "lidar", "matrix_entropy", "neighborhood_overlap", "normalized_std",
    "participation_ratio", "procrustes", "pte", "ridge_predictivity", "rsa", "self_clustering", "sparsity", "svcca",
    "token_cosine", "token_gram_drift", "token_norm_outliers", "trajectory_curvature", "uniformity",
}


METRICS_NOTE = ("pte: RMS phase (phase) or cross-power (cpsd) error of a linear (lin) or MLP probe, lower is better; "
                "curvature: mean angle in radians between successive frame displacements, in [0, pi] for files named "
                "signed and folded to [0, pi/2] otherwise")


def negated_in_views(rec):
    """PTE is an error, and a folded curvature angle runs opposite to the signed one, so both are negated in
    the views; a signed curvature angle is shown as it is (lower is straighter)."""
    return rec["metric"] == "pte" or (rec["metric"] == "curvature" and "signed" not in rec.get("source", ""))


def _pick_metric_file(data_dir, prefix, token):
    """Replicates the paper loader's preference chain: filename-token match,
    then discotube corpus, then n5000, then shortest name."""
    if not data_dir.exists():
        return None
    cands = [f for f in data_dir.iterdir()
             if f.name.startswith(prefix + "_") and f.suffix == ".json"]
    if token:
        cands = [f for f in cands if token in f.name]
    disco = [f for f in cands if "discotube" in f.name]
    cands = disco or cands
    n5000 = [f for f in cands if "n5000" in f.name]
    cands = n5000 or cands
    return min(cands, key=lambda f: len(f.name)) if cands else None


def _gride_k8_from_scales(data_dir, prefix):
    """GRIDE at k = 8 from a multi-scale file (one list of IDs per layer, scales k = 2, 4, 8, ...), for models
    without a separate gride_k8 file. Returns (file, per-layer values) or (None, None)."""
    f = _pick_metric_file(data_dir, prefix, "gride")
    if f is None:
        return None, None
    try:
        per_layer = json.loads(f.read_text())["intrinsic_dimension"]
        layers = sorted((k for k in per_layer if k.isdigit()), key=int)
        if [int(k) for k in layers] != list(range(len(layers))) or per_layer[layers[0]]["k_ranks"][2] != 8:
            return None, None
        vals = [per_layer[k]["id"][2] for k in layers]
    except (KeyError, TypeError, ValueError, IndexError):
        return None, None
    return f, vals


def export_metrics():
    written, problems, n_rec = [], [], 0
    for model_id, spec in MODELS.items():
        if spec.get("hidden"):
            continue
        prefix = METRIC_PREFIX.get(model_id)
        if not prefix:
            continue
        down = OUT / "results" / model_id / "downstream.json"
        n_layers = None
        if down.exists():
            recs = json.loads(down.read_text())["records"]
            n_layers = max((r["n_layers"] for r in recs), default=None)
        records = []
        for metric, mdir, mkey, vkey, variants in METRIC_SPECS:
            for variant, token in variants:
                for tok in (token if isinstance(token, tuple) else (token,)):
                    f = _pick_metric_file(SRC / mdir / "data", prefix, tok)
                    if f is not None:
                        break
                if f is None and variant == "gride_k8":
                    f, vals = _gride_k8_from_scales(SRC / mdir / "data", prefix)
                    if f is None:
                        continue
                elif f is None:
                    continue
                else:
                    try:
                        vals = json.loads(f.read_text())[mkey][vkey]
                    except (KeyError, TypeError):
                        continue
                if not isinstance(vals, list) or not vals:
                    continue
                if n_layers and len(vals) != n_layers:
                    problems.append(f"{model_id}/{metric}.{variant}: "
                                    f"{len(vals)} layers vs downstream {n_layers}")
                records.append({
                    "metric": metric, "variant": variant, "layers": vals,
                    "n_layers": len(vals),
                    "corpus": ("discotube" if "discotube" in f.name else
                               "mtg-jamendo" if "mtg-jamendo" in f.name else None),
                    "source": f.name,
                })
        # PTE / CPSD equivariance: {linear,mlp} x {phase,cpsd} RMSE configs
        for fsuf, probe in [("", "linear"), ("_mlp", "mlp")]:
            f = SRC / "cpsd_equivariance" / "data" / f"{prefix}{fsuf}.json"
            if not f.exists():
                continue
            shared = json.loads(f.read_text()).get("cpsd_equivariance_shared", {})
            for rkey, rname in [("phase_rmse", "phase"), ("cpsd_rmse", "cpsd")]:
                vals = shared.get(rkey)
                if not vals:
                    continue
                variant = f"{'lin' if probe == 'linear' else 'mlp'}_{rname}"
                rec = {"metric": "pte", "variant": variant, "layers": vals,
                       "n_layers": len(vals),
                       "lower_is_better": True, "source": f.name}
                records.append(rec)
        # Jacobian effective rank (scripts/run_frozen_jer.py): natural inputs, noise and random-init
        # controls, and delta = natural - random init
        f = SRC / "jer" / "data" / f"{prefix}.json"
        if f.exists() and SHOW_JER:
            d = json.loads(f.read_text())
            for cond, c in d["conditions"].items():
                rec = {"metric": "jer", "variant": cond, "layers": c["jer"], "n_layers": len(c["jer"]),
                       "source": f"jer/data/{f.name}",
                       "settings": d.get("settings", {}), "input_space": d.get("input_space")}
                if d["layers"] != list(range(len(c["jer"]))):
                    rec["layer_index"] = d["layers"]
                records.append(rec)
        # the rest of the req-metrics library on the cached metric set (scripts/run_library_metrics.py)
        f = SRC / "req_metrics" / "points" / f"{prefix}.json"
        if f.exists():
            prof = {}
            for r in json.loads(f.read_text()):
                if r.get("layer_b") is None and r["metric"] in LIBRARY_METRICS:
                    prof.setdefault(r["metric"], {})[r["layer"]] = r["value"]
            for m, v in sorted(prof.items()):
                vals = [v[k] for k in sorted(v)]
                records.append({"metric": m, "variant": "req-metrics", "layers": vals, "n_layers": len(vals),
                                "source": f"req_metrics/points/{f.name}"})
        if not records:
            continue
        out = OUT / "results" / model_id / "metrics.json"
        if out.exists() and model_id not in REPLACE_METRICS:   # append-only at the record level
            doc = json.loads(out.read_text())
            fresh = {(r["metric"], r["variant"]): r for r in records}
            kept, replaced = [], 0
            for r in doc["records"]:
                k = (r["metric"], r["variant"])
                if k in REPLACE_RECORDS and k not in fresh:
                    replaced += 1               # no qualifying file any more: dropped
                elif k in REPLACE_RECORDS and fresh[k]["source"] != r.get("source"):
                    kept.append(fresh[k])
                    replaced += 1
                else:
                    kept.append(r)
            doc["records"] = kept
            have = {(r["metric"], r["variant"]) for r in doc["records"]}
            new = [r for r in records if (r["metric"], r["variant"]) not in have]
            if new or replaced:
                doc["records"] += new
                doc["updated"] = str(date.today())
                out.write_text(json.dumps(doc, indent=1))
                written.append(f"{model_id} (+{len(new)} records" + (f", {replaced} replaced)" if replaced else ")"))
                n_rec += len(new)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "model": model_id,
            "space": "sequence-level embeddings, same extraction as downstream",
            "note": METRICS_NOTE,
            "exported": str(date.today()), "records": records}, indent=1))
        written.append(model_id)
        n_rec += len(records)
    print(f"metrics: {n_rec} records -> {len(written)} models")
    for p in problems:
        print(f"  layer-count note: {p}")


# ------------- metrics page: metric x task correlations -------------

METRIC_ROW_LABEL = {
    ("id", "twonn"): "ID (TwoNN)", ("id", "gride_k8"): "ID (GRIDE k=8)",
    ("alpha", "default"): "Alpha-ReQ",
    ("effective_rank", "default"): "Effective rank",
    ("anisotropy", "spectral"): "Anisotropy",
    ("infonce", "default"): "InfoNCE", ("lidar", "default"): "LiDAR",
    ("curvature", "default"): "Curvature",
    ("pte", "best"): "PTE",                     # the paper's PTE rule, over the stored PTE variants
}
# chord recognition is shown per mir_eval metric, never aggregated (the metrics differ in what they credit)
CORR_EXTRA_TASKS = {"chords_ace_root", "chords_ace_thirds", "chords_ace_triads"}
CORR_LABEL = {"chords_ace_mirex": "Chords MIREX"}


def _spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            for k in range(i, j + 1):
                r[s[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def render_correlations(task_registry):
    """corr.html: mean within-model Spearman between each canonical metric
    profile and each primary task's layer curve. Recomputed from data/;
    conventions stated on the page."""
    rows_data = {}                              # model -> {(metric,variant): layers}
    tasks_data = {}                             # model -> {task: layers}
    fam_of = {}
    for model_id, spec in MODELS.items():
        # readout groups: keep the group leader (its own metric profiles),
        # skip the other heads so one model contributes once
        if spec.get("hidden") or (spec.get("readout_of")
                                  and spec["readout_of"] != model_id):
            continue
        mf = OUT / "results" / model_id / "metrics.json"
        df = OUT / "results" / model_id / "downstream.json"
        if not (mf.exists() and df.exists()):
            continue
        mrecs = json.loads(mf.read_text())["records"]
        rows_data[model_id] = {(r["metric"], r["variant"]): r for r in mrecs
                               if (r["metric"], r["variant"]) in METRIC_ROW_LABEL}
        rows_data[model_id][("pte", "best")] = [r["layers"] for r in mrecs if r["metric"] == "pte"]
        tasks_data[model_id] = {
            r["task"]: r["layers"]
            for r in primary_records(json.loads(df.read_text())["records"])
            if r["task"] in PRIMARY_METRICS or r["task"] in CORR_EXTRA_TASKS}
        fam_of[model_id] = spec["family"]

    tcols = [t for tf in TASK_FAMILY_ORDER
             for t in sorted(t2 for t2, f in TASK_FAMILY.items()
                             if f == tf and (t2 in PRIMARY_METRICS
                                             or t2 in CORR_EXTRA_TASKS))]

    def table_for(models_subset):
        head = ("<tr><th>Metric</th>" +
                "".join(f"<th class='num' title='{task_registry.get(t, t)}'>"
                        f"{CORR_LABEL.get(t, TASK_LABEL.get(t, t))}</th>"
                        for t in tcols) + "</tr>")
        body = []
        for key, label in METRIC_ROW_LABEL.items():
            tds = []
            for t in tcols:
                vals = []
                for m in models_subset:
                    curve = tasks_data.get(m, {}).get(t)
                    if not curve:
                        continue
                    if key == ("pte", "best"):  # the paper's PTE rule
                        best = None
                        for layers in rows_data.get(m, {}).get(key, []):
                            if len(layers) != len(curve):
                                continue
                            r = _spearman(layers, curve)
                            if best is None or abs(r) > abs(best):
                                best = r
                        if best is None:
                            continue
                        vals.append((m, -best))  # PTE convention: negate RMSE-rho
                        continue
                    rec = rows_data.get(m, {}).get(key)
                    if not rec or len(rec["layers"]) != len(curve):
                        continue
                    mvals = [-v for v in rec["layers"]] if negated_in_views(rec) else rec["layers"]
                    vals.append((m, _spearman(mvals, curve)))
                if not vals:
                    tds.append("<td class='num na'>&mdash;</td>")
                    continue
                mean = sum(v for _, v in vals) / len(vals)
                a = max(abs(mean), 0.06)
                # paper polarity (RdBu_r): warm = positive, blue = negative
                bg = (f"rgba(235,104,52,{a:.2f})" if mean >= 0
                      else f"rgba(42,120,214,{a:.2f})")
                fg = "#fff" if abs(mean) > 0.55 else "var(--ink)"
                tip = f"n={len(vals)}: " + ", ".join(
                    f"{MODELS[m]['display']} {v:+.2f}" for m, v in vals)
                tds.append(f"<td class='num' style='background:{bg};color:{fg}' "
                           f"title='{tip}'>{mean:+.2f}</td>")
            if any("title" in td for td in tds):
                body.append(f"<tr><td class='mlabel'>{label}</td>{''.join(tds)}</tr>")
        return f"<table>{head}{''.join(body)}</table>"

    all_models = list(rows_data)
    paper = [m for m in all_models if MODELS[m].get("in_paper", True)]
    sections = [f"<section class='cview' id='cv-paper'>{table_for(paper)}</section>",
                f"<section class='cview' id='cv-all' hidden>{table_for(all_models)}</section>"]
    opts = [f"<option value='cv-paper'>The paper's models ({len(paper)})</option>",
            f"<option value='cv-all'>All models ({len(all_models)})</option>"]
    for fam in FAMILY_ORDER:
        sub = [m for m in all_models if fam_of[m] == fam]
        if len(sub) < 2:
            continue
        sections.append(f"<section class='cview' id='cv-{fam}' hidden>"
                        f"{table_for(sub)}</section>")
        opts.append(f"<option value='cv-{fam}'>{FAMILY_LABEL.get(fam, fam)} "
                    f"({len(sub)} models)</option>")

    tpl = page_template("corr")
    (OUT.parent / "corr.html").write_text(
        tpl.replace("<!--SECTIONS-->", "\n".join(sections))
           .replace("<!--OPTIONS-->", "".join(opts))
           .replace("{{DATE}}", BUILD_DATE))
    print(f"corr.html: {len(METRIC_ROW_LABEL)} metrics x {len(tcols)} tasks, "
          f"{len(all_models)} models")


# ---------------- multi-layer fusion results ----------------
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
# the fusion folders named omar-rq-base hold runs on OMAR-RQ multifeature-25hz embeddings (their configs'
# embedding_root; MARBLE/scripts/analyze_multilayer_vs_proxy.py reads them the same way)
PROBE_MODEL_MAP = {"omar-rq-base": "omar-rq-multifeature-25hz", "mert-95M": "mert-v1-95M", "mert-330M": "mert-v1-330M",
                   "maest-30s-discogs-pw": "maest", "music-flamingo": "musicflamingo"}
# runs with another extraction protocol, never merged into the rows of a model
SKIP_MODEL_TOKENS = ("-centered", "-lastn", "maest-10s-discogs-pw")  # the February MAEST-10s runs only, not
# the windowed 10-s variant (discogs-maest-10s-pw-129e)


def _last_test_row(probe_dir):
    """Latest lightning version dir whose metrics.csv carries test/ columns."""
    versions = sorted(probe_dir.glob("lightning_logs/version_*"),
                      key=lambda p: int(p.name.split("_")[1]), reverse=True)
    for v in versions:
        f = v / "metrics.csv"
        if not f.exists():
            continue
        try:
            with f.open() as fh:
                r = list(csv.DictReader(fh))
        except csv.Error as e:                  # a write cut short (NUL bytes): never guess values
            print(f"  unreadable, skipped: {f} ({e})")
            continue
        if r and any(k.startswith("test/") for k in r[0]):
            vals = {}
            for row in r:                       # last non-empty value per test col
                for k, x in row.items():
                    if k.startswith("test/") and x not in ("", None):
                        vals[k] = float(x)
            if vals:
                return vals, v.name
    # runs written by the layer-wise runner: one entry, keyed by the last layer
    f = probe_dir / "layerwise_results.json"
    if f.exists():
        res = json.loads(f.read_text())
        if len(res) == 1:
            vals = {k: float(x) for k, x in next(iter(res.values())).items() if k.startswith("test/")}
            if vals:
                return vals, f.name
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
        if (task_probe not in FUSION_TASKS or variant not in FUSION_VARIANTS
                or FUSION_TASKS[task_probe][0] in FUSION_SKIP_TASKS):
            continue
        if any(s in token for s in SKIP_MODEL_TOKENS):
            skipped_models.add(token)
            continue
        model = PROBE_MODEL_MAP.get(token, token)
        if (model not in MODELS or model in UNPUBLISHED or model in FUSION_EXCLUDE
                or not (OUT / "results" / model / "downstream.json").exists()):
            skipped_models.add(token)
            continue
        vals, _ = _last_test_row(d)
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
            rec.update({"variant": variant, "protocol": "multi-layer fusion"})
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


# ------------- layer selection page -------------

FUSION_GROUPS = [
    ("Single layers", ["oracle", "middle", "last"]),
    ("Layers ranked highest by a label-free metric", ["topk_avg", "topk_stack", "top5_avg", "top5_stack"]),
    ("All layers, no training", ["uniform_avg", "stack"]),
    ("All layers, trained", ["learned_avg", "hconv", "attentive_ciernik"]),
]
# fusion-task id -> the single-layer records averaged into the same metric as the fusion score
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
    "emo_r2": "Emotion (Emomusic)", "hxmsa": "Structure (Harmonix)",
    "chords_ace_mirex": "Chords MIREX (ChordsACE)",
    "chords_ace_root": "Chords root (ChordsACE)",
    "chords_ace_thirds": "Chords thirds (ChordsACE)",
    "mtt": "Tagging (MagnaTagATune)", "mtg_genre": "Genre tagging (MTG-Jamendo)",
    "mtg_instrument": "Instrument tagging (MTG-Jamendo)", "mtg_mood": "Mood tagging (MTG-Jamendo)",
    "mtg_top50": "Top-50 tagging (MTG-Jamendo)",
}


def _single_layer_curve(down_recs, task):
    """Same-metric per-layer curve for a fusion task (avg of components for tagging)."""
    comp = FUSION_BASE.get(task, [task])
    curves = [r["layers"] for r in down_recs if r["task"] in comp]
    if len(curves) != len(comp):
        return None
    return [sum(v) / len(curves) for v in zip(*curves)]


# The single-layer rows of the fusion tables come from the same probe runs as the fusion results.
FUSION_EXCLUDE = {"myna"}
FUSION_SKIP_TASKS = set(CHORD_TASKS)


def _paper_run_records(model_id):
    """Per-layer records of the probe runs the fusion results are compared with."""
    raw = json.loads((SRC / "downstream" / "data" / f"{MODELS[model_id]['file']}.json").read_text())["downstream"]
    return [{"task": TASK_ALIASES.get(k, k), "layers": v} for k, v in raw.items()]


def append_fusion_reference():
    """Store each fusion file's single-layer reference rows (best, middle and last layer) from the runs the
    fusion results were compared with, so the page can be rebuilt from data/ alone. Append-only."""
    done = []
    for model_id in MODELS:
        f = OUT / "results" / model_id / "fusion.json"
        if not f.exists() or model_id in UNPUBLISHED or model_id in FUSION_EXCLUDE:
            continue
        doc = json.loads(f.read_text())
        have = {(r["task"], r["variant"]) for r in doc["records"]}
        name = MODELS[model_id]["file"]
        down = _paper_run_records(model_id)
        new = []
        for task in sorted({r["task"] for r in doc["records"]} - FUSION_SKIP_TASKS):
            curve = _single_layer_curve(down, task)
            if not curve:
                continue
            for variant, value in (("oracle", max(curve)), ("middle", curve[len(curve) // 2]),
                                   ("last", curve[-1])):
                if (task, variant) not in have:
                    new.append({"task": task, "variant": variant, "value": round(value, 2),
                                "source": f"downstream/data/{name}.json", "protocol": "single-layer reference"})
        if new:
            doc["records"] += new
            doc["updated"] = str(date.today())
            f.write_text(json.dumps(doc, indent=1))
            done.append(model_id)
    print(f"fusion reference rows added: {', '.join(done) or 'none'}")


def render_fusion_panel():
    data = {}                                   # task -> model -> variant -> value
    models_here = []
    for model_id in MODELS:
        f = OUT / "results" / model_id / "fusion.json"
        if not f.exists() or MODELS[model_id].get("hidden") or model_id in FUSION_EXCLUDE:
            continue
        models_here.append(model_id)
        fus = json.loads(f.read_text())["records"]
        down = primary_records(json.loads((OUT / "results" / model_id / "downstream.json")
                                          .read_text())["records"])   # used only without stored reference rows
        for r in fus:
            if r["task"].endswith("_ap") or r["task"].endswith("_auroc") or r["task"] in FUSION_SKIP_TASKS:
                continue                        # components (the panel shows the avg); decoder-affected chords
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
    tpl = page_template("fusion")
    (OUT.parent / "fusion.html").write_text(
        tpl.replace("<!--OPTIONS-->", opts)
           .replace("<!--MOPTIONS-->", mopts)
           .replace("<!--SECTIONS-->", "\n".join(sections + msections))
           .replace("{{DATE}}", BUILD_DATE))
    print(f"fusion.html: {len(order)} tasks x {len(models_here)} models")


# ---------------- render-only mode: every page from data/ alone ----------------
# Reads the model and task registries and the records under data/, never the analysis
# outputs, so anyone with a clone can rebuild the site, and a model added to
# data/registry/models.yaml with its records under data/results/<id>/ appears on the pages.

def _read_registry(path):
    """The two-level YAML of data/registry: {key: {field: value}}, values as strings or booleans."""
    out, cur = {}, None
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            cur = line[:-1]
            out[cur] = {}
        elif cur is not None and ":" in line:
            k, v = line.strip().split(":", 1)
            v = v.strip().strip('"')
            out[cur][k] = {"true": True, "false": False}.get(v, v)
    return out


def _records_date():
    """The latest export date among the records, which is when the shown results last changed."""
    docs = [json.loads(f.read_text()) for f in (OUT / "results").glob("*/*.json")]
    dates = [d.get(k, "") for d in docs for k in ("exported", "updated")]
    cov = OUT / "coverage.json"
    if cov.exists():
        dates.append(json.loads(cov.read_text()).get("generated", ""))
    return max(d for d in dates if d)


def render_all():
    """Rebuild every page and the data README from data/ only."""
    global BUILD_DATE
    BUILD_DATE = _records_date()
    for mid, spec in _read_registry(OUT / "registry" / "models.yaml").items():
        if mid not in MODELS and (OUT / "results" / mid / "downstream.json").exists():
            MODELS[mid] = {"file": mid, "display": spec.get("display", mid), "family": spec.get("family", "masked"),
                           "in_paper": bool(spec.get("in_paper", False))}
    tasks = {t: spec.get("description", t) for t, spec in _read_registry(OUT / "registry" / "tasks.yaml").items()}
    print(f"index.html rendered: {render_atlas_table(tasks)} models")
    render_cheatsheet(tasks)
    render_correlations(tasks)
    render_fusion_panel()
    write_data_readme()
    coverage_report()
    print(f"rendered from data/ (results as of {BUILD_DATE})")


def _registry_tasks():
    """Task id -> description, from data/registry/tasks.yaml."""
    return {t: spec.get("description", t) for t, spec in _read_registry(OUT / "registry" / "tasks.yaml").items()}


if __name__ == "__main__":
    # Default: export the downstream records and render the pages. --render rebuilds the pages from data/ alone;
    # --metrics and --fusion add their steps after the default export, in this order.
    if "--render" in sys.argv:
        render_all()
    else:
        main()
        if "--metrics" in sys.argv:
            export_metrics()
            render_correlations(_registry_tasks())
        if "--fusion" in sys.argv:
            fm = export_fusion()
            # consistency check against a published value: GS x mert-v1-330M x topk_avg == 63.64
            anchor = [r for r in fm.get("mert-v1-330M", []) if r["task"] == "key" and r["variant"] == "topk_avg"]
            print("ANCHOR GS/mert-330M/topk_avg:", anchor[0]["value"] if anchor else "MISSING", "(expect 63.64)")
            render_fusion_panel()
