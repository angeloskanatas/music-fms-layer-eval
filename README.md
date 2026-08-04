# What Makes a Good Layer?

### Assessing the Layer-Wise Intrinsic Properties of Music Foundation Models

**ISMIR 2026** &nbsp;|&nbsp; [Project page](https://angeloskanatas.github.io/music-fms-layer-eval/)

Angelos-Nikolaos Kanatas<sup>1</sup>, Yuexuan Kong<sup>2,3</sup>, Pablo Alonso-Jiménez<sup>1</sup>, Xavier Serra<sup>1</sup>, Dmitry Bogdanov<sup>1</sup>

<sup>1</sup> Music Technology Group, Universitat Pompeu Fabra &nbsp;·&nbsp; <sup>2</sup> Deezer Research &nbsp;·&nbsp; <sup>3</sup> Nantes Université, École Centrale Nantes, CNRS, LS2N

---

Music foundation models are used as frozen feature extractors, but *which layer* you extract from is
usually a guess. We analyze 12 music foundation models layer by layer, across masked, autoregressive
and contrastive pre-training, and ask whether label-free properties of a representation predict how
well that layer transfers.

**No single property is a universal proxy.** Intrinsic dimension and curvature track layer quality for
genre, emotion, tagging and beat — but every standard metric fails on tonal tasks. We introduce
**Pitch-Transposition Equivariance (PTE)**, a self-supervised diagnostic that provides a consistent
indicator of tonal quality across model families. Used together, these metrics reduce an exhaustive
layer search to a three-layer shortlist that lands within 0.4 pp of the per-task oracle and
outperforms trainable multi-layer fusion.

## Contents

- **[Project page](https://angeloskanatas.github.io/music-fms-layer-eval/)** — paper, abstract, findings.
- **[Layer Atlas](https://angeloskanatas.github.io/music-fms-layer-eval/atlas.html)** — layer-wise
  downstream results for 17 models across 15 tasks: best layer per cell, full layer-profile
  sparklines, sortable, CSV/LaTeX export.
- **[Layer Cheat Sheet](https://angeloskanatas.github.io/music-fms-layer-eval/layers.html)** — the
  three-layer band to probe first, per model and task family, with depth rules of thumb.
- **[Layer Explorer](https://angeloskanatas.github.io/music-fms-layer-eval/explorer.html)** —
  interactive per-layer curves for any model and task.
- **[Selection & Fusion](https://angeloskanatas.github.io/music-fms-layer-eval/fusion.html)** —
  proxy-guided and trainable multi-layer fusion against the best single layer.
- **[`data/`](data/)** — the raw per-layer records behind every number: one JSON folder per model,
  self-describing and append-only, with open model/task registries. See [`data/README.md`](data/README.md).
- **Code** — the analysis toolkit (intrinsic metrics, PTE, layer selection and fusion) will be
  released separately after the camera-ready.

## Citation

```bibtex
@inproceedings{kanatas2026goodlayer,
  title     = {What Makes a Good Layer? Assessing the Layer-Wise Intrinsic Properties of Music Foundation Models},
  author    = {Kanatas, Angelos-Nikolaos and Kong, Yuexuan and Alonso-Jim{\'e}nez, Pablo and Serra, Xavier and Bogdanov, Dmitry},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2026}
}
```

## Acknowledgments

The project page uses the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template),
adopted from the [Nerfies](https://nerfies.github.io/) page. Website content is licensed under
[CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).
