# Residual correction for action-robust cooperative MARL: research notes

Docs and results only. Code lives in [action-robust-marl](https://github.com/Ayesha049/action-robust-marl)
(`phase1/`, branch `phase-1`) and is cited by commit SHA.

This repo is about one proposed method: a residual correction that lets each agent infer action
corruption from its own history and push against it. The benchmark of existing methods, under
the same definition, lives in a separate repo (sihonglab/ayesha-benchmark-au).

## Contents

- `shared/`: snapshots of the benchmark repo's `formal_definition.md`,
  `existing_robust_methods.md` and `environments_and_budgets.md`. The method is stated and
  tested in their terms. Edit the originals, not these copies.
- `history/phase-1/`: the first study of the method (closed 2026-11-19). `PHASE-1.md` has the
  summary and verdict at the top; the framework doc defines the method; `residual-hypotheses.md`
  and `novelty-check.md` give the hypotheses and the literature position; `results/` holds the
  stage summaries.

References in `history/phase-1/` to `phase-0/PHASE-0.md` point to the benchmark repo's
`archive/phase-0/`. Lab servers are named "lab server A/B" here.
