# What Makes a Good Layer?

Per-layer results for music foundation models, from the ISMIR 2026 paper *What Makes a Good Layer? Assessing the
Layer-Wise Intrinsic Properties of Music Foundation Models* ([arXiv:2608.14819](https://arxiv.org/abs/2608.14819)).

Angelos-Nikolaos Kanatas<sup>1</sup>, Yuexuan Kong<sup>2,3</sup>, Pablo Alonso-Jiménez<sup>1</sup>,
Xavier Serra<sup>1</sup>, Dmitry Bogdanov<sup>1</sup>

<sup>1</sup> Music Technology Group, Universitat Pompeu Fabra; <sup>2</sup> Deezer Research;
<sup>3</sup> Nantes Université, École Centrale Nantes, CNRS, LS2N

The site is at **https://angeloskanatas.github.io/music-fms-layer-eval/** and has four pages:

- [Results](https://angeloskanatas.github.io/music-fms-layer-eval/): the downstream score of every layer of each
  model on each task, shown as the best layer per task or as the three layers to probe first, with a choice of
  probe where we have more than one.
- [Layer profiles](https://angeloskanatas.github.io/music-fms-layer-eval/explorer.html): the score at every layer
  for one model and task, with a label-free metric laid over it if you pick one.
- [Metrics](https://angeloskanatas.github.io/music-fms-layer-eval/corr.html): how closely each label-free metric
  follows the per-layer scores, per task and per paradigm.
- [Layer selection](https://angeloskanatas.github.io/music-fms-layer-eval/fusion.html): the best single layer
  against the layers ranked highest by a metric and against multi-layer fusion.

We keep adding models, tasks and probes.

## Data

[`data/`](data/) holds the records behind every number, one folder per model: `downstream.json` for the
per-layer scores, `metrics.json` for the per-layer metrics and `fusion.json` for the multi-layer methods.
[`data/README.md`](data/README.md) describes the protocol and the file format. The metrics are computed with
[req-metrics](https://github.com/angeloskanatas/req-metrics).

## Rebuilding the pages

```bash
python scripts/export_atlas_json.py --render
```

This rebuilds every page from `data/` alone and needs only the Python standard library. Without `--render`,
the script first exports new records from our analysis outputs, which are not in this repository.

## Citation

```bibtex
@inproceedings{kanatas2026goodlayer,
  title     = {What Makes a Good Layer? Assessing the Layer-Wise Intrinsic Properties of Music Foundation Models},
  author    = {Kanatas, Angelos-Nikolaos and Kong, Yuexuan and Alonso-Jim{\'e}nez, Pablo and Serra, Xavier and Bogdanov, Dmitry},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2026},
  note      = {arXiv:2608.14819}
}
```

## License

The website content is licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/). The first
version of the page was built from the
[Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template), adapted from
[Nerfies](https://nerfies.github.io/).
