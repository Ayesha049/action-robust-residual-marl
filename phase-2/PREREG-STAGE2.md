# Phase 2 Stage 2: confirmation pre-registration (MADDPG arm: spread and tag)

**Status: FINAL, approved by the user 2026-11-22.** Nothing below may change after the
confirmation runs start. Any later deviation is logged in the last section with its reason.
No confirmation-seed data had been looked at when it was approved: seeds 5-9 had not been
evaluated under any fault, and the tag bases on seeds 5-9 were still training. Server A's code
copy was checked file by file against the frozen commit (md5) before the launch was armed.
Stage 2 starts automatically when the tag bases finish.

## 1. Frozen code

- Code repo `action-robust-marl`, branch `phase-2`, commit `01d0f21` (full SHA
  01d0f21ff67c4f8e4152b78411d8abb1c0bb7036).
- Runner: `phase2/stage2.sh`. Tests: `phase2/stage2_tests.py`. Trainer: `phase2/train_fault.py`.
  The fault model is `phase2/faults.py`.

## 2. Bases (held out, never used in development)

| Task | Seeds | Origin |
| --- | --- | --- |
| spread | 5-9 | MADDPG bases from Phase 1 (server B `arm-phase1/runs`). Trained with the Phase 0 settings (1.5M steps, γ 0.95, lr 1e-3, noise 0.1, batch 1024), the same as development seeds 0-4. |
| tag | 5-9 | MADDPG trained 2026-11-22 with the Phase 0 tag settings (`train.py --task tag --method maddpg`, 1.5M steps, frozen Phase 0 prey `assets/tag_prey.pt`). |

If a tag base fails to train (clean return below the worst development base), it is still used
and reported. No base is replaced.

## 3. Methods (Stage 1 run 3 setup, frozen)

All four kinds are trained on every base with `phase2/train_fault.py --track 1` and the RC seed
equal to the base seed.
- **Everything else is at the trainer defaults:** action-space residual with bound 2; GRU 64;
  critic 128; 20k episodes; p_clean 0.25.
- **Training fault mix:** rotation 1/2; gain, bias and drift 1/6 each. Strength ranges: θ
  U[0, 180]°, g U[0.3, 1], ‖b‖ U[0, 0.9], η U[0, 0.04]. Victims: one or all, 50/50. Onset
  U{0..12}.
- **Intent-tracking weight** λ_track 1. Clean penalty c_r 1.

| Kind | Role |
| --- | --- |
| RC (`residual`) | the method |
| REC (`rec`) | the same network, base trained too: tests the frozen-base split (F3) |
| RC-blind (`blind`) | RC without executed-action evidence: tests the mechanism (F5b) |
| FT (`ft`) | the base fine-tuned, no memory: mechanism comparator (F5a) |
| MADDPG | the frozen base, no correction |

## 4. Evaluation

- **Cells:** every Stage 0 cell, with no oracle.
  - clean
  - gain g ∈ {0.7, 0.5, 0.3}, one agent and all agents
  - bias ‖b‖ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, one agent and all agents
  - drift η ∈ {0.01, 0.02, 0.04}, one agent
  - stuck, one agent
  - worst direction (bias 0.3, 8 directions × each agent)
  - rotation θ ∈ {30, 60, 90, 180}°, one agent and all agents
- **Episodes per cell:** 500 on tag, 100 on spread. Phase 0 evaluation seeds 100,000 + e, and the
  Stage 0 RNG keys, so every method faces identical onsets and fault draws.
- **Scale:** ρ = (J − J_rand) / (J0_base − J_rand), with J_rand = 1.05 on tag and −26.08 on
  spread. J0_base is the MADDPG base's own clean return.
- **Single-agent cells:** the victim (and, for the worst-direction grid, the direction) is the
  base's worst case under no correction. It is fixed per seed for every method.
- **Mechanism fit:** `phase2/diag_correction.py`, rot 90°, all agents, 100 episodes, for RC and
  FT. k_fit is the least-squares slope of the applied correction on the exact oracle's
  correction, per step k after onset.

## 5. Tests (per task, 5 seeds, paired; 95% percentile bootstrap, 10,000 resamples, seed 0)

δ = 0.05 ρ, which is 5% of the clean-to-random gap.

| Test | Claim | Pass rule |
| --- | --- | --- |
| F1 headline | RC beats MADDPG on rotation 90°, all agents | RC − MADDPG ≥ δ and CI lower > 0 |
| F2 breadth | RC beats MADDPG across rotation | over the 8 cells θ ∈ {30, 60, 90, 180} × {all, worst single}: CI lower > 0 in at least 5, CI upper < 0 in none |
| F3 split | RC beats REC | per seed, mean RC − REC over the 8 F2 cells: CI lower > 0 |
| F4 dose-response | RC's advantage grows with fault size | per seed, OLS slope of (ρ_RC − ρ_MADDPG) on θ ∈ {30, 60, 90, 180}, all agents: CI lower > 0 |
| F5 mechanism | RC learns the oracle correction from executed-action evidence | (a) per seed, the mean over k = 1..12 of k_fit(RC) − k_fit(FT): CI lower > 0; **and** (b) ρ(RC) − ρ(RC-blind) on the F1 cell: CI lower > 0 |
| F6 no harm | RC does not hurt under the authority-limited faults | every gain, bias, drift, stuck and worst-direction cell: CI lower of RC − MADDPG > −δ |
| F7 clean | no clean cost | clean *return* RC − MADDPG: CI lower > −0.5 |

**Outcome, per task:**
- **Full:** F1-F7 all pass.
- **Without split:** everything passes except F3. The claim then becomes "recurrent
  evidence-driven fault correction", not "residual on a frozen base".
- **Mechanism only:** F5, F6 and F7 pass.
- **No:** anything else.

**Headline claim:** made at the weaker of the two task outcomes. Both tasks are reported in full
either way.

## 6. Expectations from development (seeds 0-4, Stage 1 run 3, the same code)

Not part of the test. These are recorded so the confirmation can be read against them.

| Test | tag (dev) | spread (dev) |
| --- | --- | --- |
| F1 RC − MADDPG, rot 90° all | +0.727 [0.709, 0.741] | +0.748 [0.717, 0.781] |
| F2 cells better / worse | 8 / 0 | 8 / 0 |
| F3 RC − REC | +0.002 [−0.011, 0.017], fail | −0.002 [−0.004, 0.000], fail |
| F4 slope per 100° | +0.229 [0.195, 0.262] | +2.075 [2.030, 2.124] |
| F5a k_fit RC − FT | +0.967 [0.963, 0.971] | +0.860 [0.821, 0.893] |
| F5b RC − RC-blind | +0.383 [0.361, 0.408] | +0.119 [0.101, 0.138] |
| F6 lowest CI bound | −0.040 (gain 0.7, worst single) | −0.017 (bias 0.9, all) |
| F7 clean return | −0.17 [−0.44, +0.09] | +0.03 [−0.00, +0.06] |
| Outcome | Without split | Without split |

- **Expected outcome on both tasks: Without split.** F3 is expected to fail, as it did in Phase 1
  (RC − REC +0.004).
- **Margins to watch:**
  - F7 on tag: the development CI lower bound is −0.44 against the −0.5 bar.
  - F6 on tag: −0.040 against −0.05.
  - F5b on spread depends on RC-blind being clearly worse.

## 7. Reporting

- Every cell is reported for every method, with the full paired tables next to the test lines.
- Failures are reported with the same prominence as passes.
- A per-seed table is reported for F1, F3 and F7.

## 8. Deviations log

(none yet)
