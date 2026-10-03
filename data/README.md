# Atlas data

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
- `results/<model>/metrics.json` -- per-layer intrinsic metrics per (metric, variant).
- `results/<model>/fusion.json` -- one score per (task, multi-layer method);
  tagging tasks scored as mean of AP and AUROC, matching the published protocol.
- `registry/models.yaml`, `registry/tasks.yaml` -- open vocabularies; aliases and
  excluded metrics (with reasons) live here.
- `coverage.json` -- which (model, task) cells are not yet evaluated.

Records are self-describing (schema_version, readout spec, provenance) and
append-only: existing records are never rewritten.

## Contributing results
New models and tasks are added append-only, by pull request or issue:
1. Evaluate every layer under the protocol above, with three seeds per new cell.
2. Provide `results/<model>/downstream.json` in the schema of the existing files and an entry
   for the model in `registry/models.yaml` (`family`, `display`, `in_paper: false`).
3. Optionally provide `results/<model>/metrics.json`, the per-layer intrinsic metrics written by
   [req-metrics](https://github.com/angeloskanatas/req-metrics) (`Records.to_atlas_json`).
4. State the checkpoint, the clip length and the pooling used for extraction.

The pages are rebuilt from this folder alone with `python scripts/export_atlas_json.py --render`.
