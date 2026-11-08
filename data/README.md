# Data

The records behind every number on the site, one folder per model.

## Protocol

- Each layer of the frozen encoder is evaluated on its own. Layer 0 is the earliest representation the model returns.
- The default probe is an MLP with one hidden layer of 512 units, trained with the settings of MARBLE's
  constrained track, on the same splits for every model and with seed 1234.
- Records with `readout: linear-marble` use a linear probe instead (see `readout_specs` in each file).
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
