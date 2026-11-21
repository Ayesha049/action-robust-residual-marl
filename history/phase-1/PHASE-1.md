# Phase 1: residual-corrected action-robust MARL, staged evaluation

Status 2026-11-19. **Phase 1 is closed.** Stages 0-2 and a confirmation run on fresh seeds are
complete and written up (section 5). The full evaluation (`tag`, ablations, NR-MADDPG-base
variant, M3DDPG / ERNIE-A-c / MIR3) was not run.

### Phase 1 in brief

**Verdict.** The residual correction (RC) gives a real, replicated, mechanistically explained
gain, but a narrow one. Under the pre-registered tests it does not beat Phase 0's best method
(PR-MADDPG) across the NR threat model: the outcome is **No** in Stage 2 (seeds 0-4) and again
in the confirmation run (seeds 5-9).

| | Stage 2 (seeds 0-4) | Confirmation (seeds 5-9) |
| --- | --- | --- |
| T1 headline: L-NR all, alpha 0.3, RC − PR-MADDPG | +0.042, pass | +0.036 [+0.022, +0.048], pass |
| T2 NR cells better (needs > 11 of 22) | 11, fail | 10, fail |
| T3 phi trend / T3' detect-then-compensate | fail | pass |
| T4 no harm on 23 PR cells | pass | 22 of 23, fail (takeover, by 0.006) |
| T5 clean return | pass | pass |
| Outcome | No | No |

**What Phase 1 established.**

1. **Room for any correction is small** (Stage 0). Even a correction that knows the attack
   exactly recovers at most about 0.05 competence, because the agents lack the actuator
   authority to undo a ±1 replacement under NR; predicting the attack is not the bottleneck.
2. **RC captures most of that room on the strongest attacks.** On learned NR at alpha 0.3, RC
   gains +0.036 to +0.053 over its base, in Stage 1 (MADDPG base) and on both seed sets of
   Stage 2 (PR-MADDPG base). FT, the same training without the residual architecture, gains
   nothing.
3. **The mechanism is detect, then compensate.** After about 10 steps of evidence RC pushes
   against the previous step's corruption with coefficient 0.41-0.43, close to the oracle's
   alpha/(1−alpha) = 0.43, and stays near the base policy on clean episodes. Removing the
   executed-action input removes about 86% of the gain (confirmation T3', pre-registered).
4. **The gain is narrow.** It is clear on learned NR at alpha 0.3 and on strong (alpha 0.5) or
   persistent (phi 0.99) random NR noise, and below seed noise elsewhere, where the base is
   already at 0.97-0.99. It does not grow with the predictability of the noise, which falsified
   the original mechanism hypothesis (T3).
5. **Memory is the useful ingredient.** Recurrent variants help, memoryless ones (FT, RES-MLP)
   do not; the frozen-base split adds little (RC − REC +0.004). Faster detection (RC-fast)
   does not help.
6. **No harm is nearly but not fully met.** RC passes 45 of 46 PR-cell checks over the two runs;
   the one miss is the full-replacement takeover cell in the confirmation run.

**Implications for what comes next.**
- On `spread` with these operators the ceiling for action correction is about 0.05 competence.
  A follow-up that wants a broad NR win needs a setting with more headroom (larger alpha, a task
  where actuator authority is not saturated, or attacks below the action limits), not a better
  corrector.
- The detect-then-compensate result is the strongest finding: a pre-registered, replicated
  mechanism with a causal ablation, plus an honest negative on the broad claim.
- If the full evaluation is ever run, pre-register T3' (not T3) and decide in advance whether
  T2's "majority of NR cells" bar is the right one given finding 4.

Run data stays on lab server A (`~/action-robust-marl/runs_phase1`, Stages 0-1 and the
diagnostic logs) and lab server B (`~/arm-phase1`: `runs_phase1s2`, `runs_phase1c`).

Method and full evaluation design:
`Residual-Corrected MAPPO for Action-Robust Cooperative MARL Framework Definition.md`
("framework doc"; section 7 is the evaluation framework). Hypotheses:
`residual-hypotheses.md`. Literature position: `novelty-check.md`. Phase 0 results:
`phase-0/PHASE-0.md`.

Code lives in [action-robust-marl](https://github.com/Ayesha049/action-robust-marl) (`phase1/`),
cited here by commit SHA. This repo holds docs and results only.

---

## 1. Goal

Check whether the residual framework improves on the Phase 0 results under the NR
(noisy-mixing) operator, its declared threat model. The claim is NR robustness with no harm under PR.
The method is built in steps so that each step is tested before the next:

1. **Stage 0:** do Phase 0's policies and attacks leave room for any history-based correction?
   (No training.)
2. **Stage 1:** MADDPG + residual. Does the residual mechanism work at all, on a base that is
   not robust?
3. **Stage 2:** PR-MADDPG + residual. Does it beat Phase 0's best method?
4. **Full evaluation:** framework doc section 7 (both tasks, ablations, NR-MADDPG-base variant).

Setup throughout: Phase 0's MADDPG code on MPE (`mpe2` 1.1.1, `spread` first), Phase 0's attack
operators and learned-adversary protocol, Phase 0 checkpoints as frozen bases.

## 2. Stages

### Stage 0: checks on Phase 0 checkpoints (no training)

| Check | Measures | Stop rule |
| --- | --- | --- |
| Saturation | share of intended actions with $\lvert a\rvert \le 1-\alpha$ and $\le 1-2\alpha$ per dimension, clean and under NR-R at alpha 0.3, for MADDPG, PR-MADDPG, NR-MADDPG | informs the oracle result; no stop rule on its own |
| Adversary speed | mean $\lVert r_t - r_{t-1}\rVert$ and lag-1 autocorrelation of Phase 0's learned NR adversaries, against i.i.d. uniform and $\mathrm{R}_\phi$ at $\phi$ 0.9 and 0.99 | informs where L sits on the predictability axis |
| Oracle compensation | frozen team plus the analytic NR correction $a = \mathrm{clip}\big((a^\ast - \alpha\hat r)/(1-\alpha)\big)$ with four predictors $\hat r$: none (no correction), zero (gain correction only), last ($r_{t-1}$, what an agent can infer from its motion), exact ($r_t$, clairvoyant ceiling) | **stop** if the "last" oracle gains less than 0.02 competence retained over "none" in every NR cell for MADDPG (mean over 5 seeds) |

Sources: R (i.i.d.), $\mathrm{R}_\phi$ ($\phi$ 0.9, 0.99), L (Phase 0's NR adversaries, all-agent
and agent-0 scope, alpha 0.1 and 0.3). Alpha 0.1, 0.3, 0.5 for R and $\mathrm{R}_\phi$. 100
episodes per cell with Phase 0's evaluation seeds. The L adversaries were trained against the
uncorrected team, not against the oracle, so oracle gains under L are optimistic.

The 0.02 threshold is about the size of Phase 0's smallest significant effects (M3DDPG's gains on `spread` were at
most +0.017) and is fixed before the run.

### Stage 1: MADDPG + residual

| Item | Choice |
| --- | --- |
| Task, seeds | `spread`, 3 seeds; Phase 0 MADDPG checkpoints, frozen |
| Residual training | NR only, alpha in {0.1, 0.3, 0.5}; sources R and $\mathrm{R}_\phi$ ($\phi$ 0.9, 0.99); no learned adversary in training |
| Compared against | MADDPG (the base); **FT** (MADDPG fine-tuned under the same NR mixture and steps, memoryless); NR-MADDPG (Phase 0) |
| Test cells | clean; NR-R alpha 0.3; NR-$\mathrm{R}_\phi$ $\phi$ 0.9 and 0.99 at alpha 0.3; NR-$\mathrm{R}_\phi$ 0.99 at alpha 0.1 and 0.5; L-NR all alpha 0.3 (adversary trained against the frozen residual policy); PR-R alpha 0.3 and takeover (no-harm checks) |
| Diagnostics | residual norm and gate, clean vs attacked steps |
| Go rule | residual better than FT at $\phi$ 0.99 (paired CI excludes 0); gain grows with $\phi$; clean return within 0.5 of MADDPG |

Stage 1 shows whether the residual adds NR robustness to a base that is not robust, which is a lower bar than
beating Phase 0. PR gains are not expected (P5); PR cells only check for harm.

**Fixed before the Stage 1 runs (2026-11-18).**

- Go rule, made precise. (1) RC − FT in competence retained, averaged over alpha 0.1, 0.3 and
  0.5 at $\phi$ 0.99, paired by seed: 95% bootstrap CI excludes 0. (2) The RC − FT point estimate at
  alpha 0.3 increases from R ($\phi$ 0) to $\phi$ 0.9 to $\phi$ 0.99. (3) Mean clean return of RC within
  0.5 of MADDPG on the same evaluation seeds. Each alpha is also reported separately. Stage 0 puts the correction ceiling at
  about 0.01 at alpha 0.3 and 0.03 at alpha 0.5, so with 3 seeds a fail is plausible even if the
  mechanism works.
- Implementation (`phase1/residual.py`, `phase1/train_residual.py`): per-agent residual on the
  frozen Phase 0 actor; GRU 64 over $[o_t, a_{t-1}, \tilde a_{t-1}, \mu_b]$; zero-initialised
  correction head, gate bias 0; $\beta = 1$ after a ramp. One centralised critic (128 units) on
  observations, executed joint action and privileged context $[\alpha, r_t]$. 20,000 episodes (500k
  steps); 2,000 critic-only episodes, then $\beta$ ramps over 8,000; batch 64 episodes, one update
  per episode; actor lr 3e-4, critic lr 1e-3, $\gamma$ 0.95, $\tau$ 0.01, exploration noise 0.1;
  clean share 0.25, clean penalty $c_r = 1$. FT uses the same everything with the base MLP copied
  and trained. No tuning: these are the first values, and they are used for both RC and FT.
- Learned cells: L-NR all alpha 0.3 and takeover (L-PR alpha 1, agent 0), 2 adversary seeds per
  team, Phase 0 protocol (`phase1/train_adversary_any.py`). MADDPG and NR-MADDPG reuse their
  Phase 0 adversaries.
- Checks passed before launch: a residual with $\beta = 0$ reproduces the base team's returns
  exactly (55 episodes, maximum difference 0.0).

### Stage 2: PR-MADDPG + residual

The Stage 1 pipeline with the frozen base swapped for PR-MADDPG, plus: PR-MADDPG as reference; the
GRAD-style recurrent control (REC, rate-bounded learned adversary in training); the learned NR
adversary in the residual's training mix; 5 seeds; the full Phase 0 core grid plus the
$\mathrm{R}_\phi$ cells. Tests T1-T6 of framework doc section 7.6.

**Fixed before the Stage 2 runs (2026-11-18).** The design choices below were agreed with the
user; the rest carries over from Stage 1, untuned.

- **Methods.** Five kinds, each on the frozen Phase 0 PR-MADDPG team of each seed (seeds 0-4):
  - **RC:** the Stage 1 residual.
  - **RC-fast:** RC plus explicit evidence. It adds $e = (\tilde a_{t-1} - a_{t-1})/0.3$ and the
    running maximum of $\lVert e\rVert$ to the GRU input, and the running maximum also feeds the
    gate (zero-initialised weight). The aim is the slow detection found in the Stage 1
    diagnostic.
  - **FT:** as in Stage 1.
  - **REC:** RC's network with the base weights trained too. It has memory and evidence but no
    frozen-base split. It keeps the same clean penalty against the untouched base that FT has.
  - **RES-MLP:** residual and gate from an MLP on $[o_t, \mu_b]$, with no memory and no evidence.

  At initialisation every kind reproduces the base's returns exactly (checked: 10 episodes,
  maximum difference 0.0).
- **Training mix.** Each episode is one of:
  - clean, with probability 0.25;
  - random NR, with probability 0.375: R, $\mathrm{R}_\phi$ 0.9 or $\mathrm{R}_\phi$ 0.99;
  - learned NR, with probability 0.375.

  Alpha is drawn from {0.1, 0.3, 0.5}, and all agents are victims.
- **Learned training adversary.** A DDPG adversary co-trained in the same run, the same scheme
  for every kind:
  - **Input and objective:** it sees [global observation, $r_{t-1}$, alpha] and maximises the
    negated team reward.
  - **Shared rate bound, as in GRAD:** $\lVert r_t - r_{t-1}\rVert \le 0.5$ per victim, with
    $r_{-1} = 0$.
    - The Stage 1 test adversaries moved at a mean rate of 0.10-0.12, with a 95th percentile of
      0.96-1.47.
    - They sit at the corners ($\lVert r\rVert \approx 1.41$) and jump between them. The bound
      therefore removes those jumps and leaves slow motion alone.
  - **Updates:** after each learned episode, 25 critic steps with an actor step every second
    one, batch 256, starting at 10k transitions. This is the Phase 0 adversary ratio.
  - **Pool:** a snapshot every 1,000 episodes, the latest 5 kept. Each learned episode uses the
    current adversary (with noise 0.1) with probability 0.5, otherwise a pool member.
  - **Test adversaries stay unbounded** (Phase 0 protocol), so training against a bounded
    attacker and testing against an unbounded one is part of the result.
- **Evaluation.**
  - **Random cells:** the Phase 0 core cells (R, PR and NR, scopes all and worst victim, alpha
    0.1, 0.3, 0.5) with Phase 0's exact seeding. Checked: PR-MADDPG reproduces its Phase 0
    R-cell and clean returns exactly. Plus the $\mathrm{R}_\phi$ cells ($\phi$ 0.9 and 0.99, PR
    and NR, the same scopes and alphas) and clean.
  - **Learned cells:** the 9 Phase 0 core learned cells, 2 adversary seeds each, for every new
    team. That is 450 adversaries. MADDPG, PR-MADDPG and NR-MADDPG are re-evaluated on the
    random cells and reuse their Phase 0 adversaries.
  - **Deferred to the full evaluation:** M3DDPG, ERNIE-A-c and MIR3.
- **Tests.** T1-T6 of framework doc section 7.6, with $\delta$ = 0.03 competence and a clean
  margin of 0.5 return, fixed now.
  - **NR cells (T2, 22):** R and $\mathrm{R}_\phi$ NR × 3 alphas × 2 scopes, plus L-NR × 2 alphas
    × 2 scopes.
  - **PR cells (T4, 23):** R and $\mathrm{R}_\phi$ PR × 3 alphas × 2 scopes, plus L-PR × 2 alphas ×
    2 scopes, plus takeover.
  - **T3:** the per-seed least-squares slope of RC − FT against $\phi \in \{0, 0.9, 0.99\}$, at
    alpha 0.3, scope all; its 95% bootstrap CI must lie above 0.
  - **RC-fast:** reported against RC on the same tests. It is a variant, not a replacement: the
    headline method stays RC unless RC-fast passes the same tests and beats RC on T1.

### Full evaluation

`tag`, the 7 ablations (framework doc section 5), and the frozen NR-MADDPG-base variant
(framework doc section 7.8).

## 3. Time frame

One person with Claude Code. Compute is cheap: in Phase 0 a 1.5M-step MADDPG run on `spread`
took about 25 min on one core (about 1,050 steps/s), and a team's 18 learned adversaries took
about 45 min.

| Block | Effort | Compute (two dedicated servers) |
| --- | --- | --- |
| Stage 0 checks | 2-3 days | < 1 h |
| Corruption channel, $\mathrm{R}_\phi$ source, executed-action evidence | 1-2 days | - |
| Recurrent residual in MADDPG (sequence replay, frozen base, gate, penalty) | 4-6 days; the main risk | - |
| FT control and sanity tests | 1-2 days | - |
| Stage 1 runs, adversaries, analysis | 2 days | a few hours, even at 3-5x Phase 0 cost per step |
| **Stage 1 total** | **about 1.5-2 weeks** | |
| Stage 2: REC control, learned adversary in training, 5 seeds, full grid (about 400-500 adversaries) | about 1 week | under a day |
| **Through Stage 2** | **about 3.5-4.5 weeks** | |
| Full evaluation | +1-1.5 weeks | under a day |

## 4. Compute

| Server | Hardware | Use |
| --- | --- | --- |
| lab server A | 64 cores, 2x RTX 5090, 503 GB | Phase 0 checkpoints and adversaries live here; Stage 0 and Stage 1 |
| lab server B | 128 cores, 8x RTX 6000 Ada, 755 GB | Stage 2 and full-grid adversaries |
| gpuhpc (UTA, SLURM, shared) | 8 nodes, 56 H100 + 8 B200; busy (26 running, 93 pending on 2026-11-18) | CPU overflow only; later MAPPO / VMAS / MaMuJoCo extension |

---

## 5. Results

### Stage 0 (done 2026-11-18)

Run: `phase1/stage0_checks.py` and `phase1/stage0_summary.py` (action-robust-marl 3553a9b,
branch phase-1), on lab server A, Phase 0 `spread` checkpoints of MADDPG, PR-MADDPG and NR-MADDPG,
5 seeds each, 100 episodes per cell. Data: `results/stage0/` (`summary.csv` is the per-cell table,
`summary_per_team.csv` per seed, `saturation.csv`, `speed.csv`, one CSV of episode returns per team).

**Sanity check.** The uncorrected L cells reproduce all 120 Phase 0 adversary evaluations exactly
(maximum difference in mean return 0.0000).

**Oracle gain over no correction**, competence retained, mean over 5 seeds (95% bootstrap CI for
"last"). "last" uses $r_{t-1}$, which an agent can recover from its own motion; "exact" is the
clairvoyant ceiling.

| Cell (all agents unless noted) | MADDPG: none | + last | + exact | PR-MADDPG: none | + last | + exact |
| --- | --- | --- | --- | --- | --- | --- |
| L-NR, alpha 0.3 | 0.683 | **+0.049** [0.043, 0.057] | +0.051 | 0.759 | **+0.039** [0.032, 0.045] | +0.042 |
| L-NR, alpha 0.3, agent 0 | 0.869 | +0.015 [0.013, 0.017] | +0.017 | 0.918 | +0.017 | +0.018 |
| L-NR, alpha 0.1 | 0.939 | +0.009 | +0.010 | 0.965 | +0.008 | +0.008 |
| R-NR, alpha 0.3 | 0.898 | +0.004 [0.000, 0.008] | +0.007 | 0.928 | +0.005 | +0.004 |
| R-NR, alpha 0.5 | 0.775 | -0.002 | +0.009 | 0.823 | +0.006 | +0.012 |
| $\mathrm{R}_\phi$ 0.9, alpha 0.3 | 0.885 | +0.011 [0.005, 0.018] | +0.013 | 0.916 | +0.005 | +0.006 |
| $\mathrm{R}_\phi$ 0.99, alpha 0.3 | 0.874 | +0.008 [0.006, 0.010] | +0.006 | 0.905 | +0.007 | +0.009 |
| $\mathrm{R}_\phi$ 0.99, alpha 0.5 | 0.634 | +0.025 [0.017, 0.033] | +0.026 | 0.697 | +0.032 | +0.032 |

Gain correction alone ("zero", $\hat r = 0$) adds at most +0.02 in any cell. NR-MADDPG behaves
like PR-MADDPG (`summary.csv`).

**Replacement-action speed** (all agents, averaged over methods; i.i.d. uniform for comparison):

| Source | Mean $\lVert r_t - r_{t-1}\rVert$ | Lag-1 autocorrelation | Mean $\lvert r\rvert$ |
| --- | --- | --- | --- |
| R (i.i.d.) | 1.04 | 0.00 | 0.50 |
| $\mathrm{R}_\phi$ 0.9 | 0.33 | 0.88 | 0.43 |
| $\mathrm{R}_\phi$ 0.99 | 0.11 | 0.99 | 0.47 |
| L, Phase 0 adversaries (NR, alpha 0.1 and 0.3) | 0.12-0.26 | 0.88-0.95 | 0.99 |

**Saturation** (share of intended action components with $\lvert a\rvert \le$ threshold, clean):

| Method | $\le 0.9$ | $\le 0.7$ ($1-\alpha$ at 0.3) | $\le 0.4$ ($1-2\alpha$ at 0.3) |
| --- | --- | --- | --- |
| MADDPG | 0.80 | 0.67 | 0.47 |
| PR-MADDPG | 0.82 | 0.71 | 0.53 |
| NR-MADDPG | 0.71 | 0.38 | 0.16 |

**Verdict: GO by the pre-registered rule, but the ceiling is low.**

1. The stop rule passes: MADDPG's "last" oracle gains +0.049 [0.043, 0.057] under L-NR all at
   alpha 0.3, above the 0.02 threshold. The other cells that clear 0.02 are L-NR all at 0.3 for
   PR-MADDPG and NR-MADDPG, and $\mathrm{R}_\phi$ at alpha 0.5.
2. **Prediction is not the bottleneck; actuator authority is.** "last" is within 0.01 of
   "exact" in every cell. Phase 0's learned adversaries change slowly (lag-1 autocorrelation
   0.88-0.95, close to $\mathrm{R}_\phi$ 0.9) and push to the action limits ($\lvert r\rvert$ about 0.99). Even
   perfect knowledge of $r$ recovers at most 0.05, because restoring $a^\ast$ under NR needs
   $\lvert a^\ast\rvert \le 1-2\alpha$ against a $\pm 1$ replacement (P4), and only 47-53% of the
   action components of MADDPG and PR-MADDPG are that small at alpha 0.3.
3. **i.i.d. random NR is not correctable** (gain at most 0.01 even for "exact"), as P5 predicts.
4. **On MADDPG, pure correction cannot reach PR-MADDPG** on the headline cell: 0.683 + 0.051 =
   0.73 < 0.759. On PR-MADDPG the correction ceiling is about +0.04 (0.759 to 0.80), which is the
   most a pure-correction residual can add to T1 in Stage 2.
5. **NR-MADDPG over-commands.** Only 16% of its clean action components are within 0.4,
   against 47% for MADDPG. This supports the gain-fault reading of 0a's NR result
   (`residual-hypotheses.md` 3.1).

The oracle bounds within-step correction only. A learned residual can also change the base's
behaviour, for example keep actions away from the limits so they stay correctable, so the oracle
is not a strict upper bound for Stage 1. Decision (2026-11-18): proceed with Stage 1 as planned.

### Stage 1 (done 2026-11-18)

Run: `phase1/stage1.sh` (train, evaluate, learned adversaries), `phase1/stage1_summary.py`
(action-robust-marl 3553a9b, branch phase-1), on lab server A. Three seeds; residual (RC) and FT
trained on the frozen Phase 0 MADDPG team of each seed, with the settings fixed above. MADDPG and
NR-MADDPG are the Phase 0 teams of the same seeds. Data: `results/stage1/` (`stage1_summary.csv`,
`stage1_per_team.csv`, per-episode evaluations in `eval/`, adversary results and training logs
per run).

**Competence retained**, mean over seeds 0-2; differences paired by seed, 95% bootstrap CI.

| Cell | MADDPG | NR-MADDPG | FT | RC | RC − FT | RC − NR-MADDPG |
| --- | --- | --- | --- | --- | --- | --- |
| Clean return | -11.02 | -10.94 | -10.89 | -10.99 | | |
| NR-R, alpha 0.1 | 0.972 | 0.979 | 0.977 | 0.989 | +0.013 [0.000, 0.020] | +0.010 [-0.007, 0.025] |
| NR-R, alpha 0.3 | 0.905 | 0.923 | 0.905 | 0.922 | +0.017 [0.004, 0.036] | -0.001 [-0.019, 0.025] |
| NR-R, alpha 0.5 | 0.783 | 0.812 | 0.784 | 0.810 | +0.026 [0.025, 0.028] | -0.002 [-0.025, 0.031] |
| NR-$\mathrm{R}_\phi$ 0.9, alpha 0.3 | 0.893 | 0.910 | 0.896 | 0.916 | +0.021 [0.009, 0.039] | +0.006 [-0.007, 0.029] |
| NR-$\mathrm{R}_\phi$ 0.9, alpha 0.5 | 0.721 | 0.744 | 0.725 | 0.755 | +0.030 [0.009, 0.043] | +0.011 [-0.002, 0.027] |
| NR-$\mathrm{R}_\phi$ 0.99, alpha 0.1 | 0.972 | 0.976 | 0.974 | 0.987 | +0.013 [0.010, 0.019] | +0.011 [0.002, 0.020] |
| NR-$\mathrm{R}_\phi$ 0.99, alpha 0.3 | 0.880 | 0.901 | 0.878 | 0.902 | +0.024 [0.015, 0.033] | +0.001 [-0.007, 0.009] |
| NR-$\mathrm{R}_\phi$ 0.99, alpha 0.5 | 0.692 | 0.714 | 0.691 | 0.722 | +0.032 [0.022, 0.043] | +0.008 [-0.013, 0.030] |
| **L-NR all, alpha 0.3** (adversary trained against each frozen team) | 0.693 | 0.713 | 0.699 | **0.746** | **+0.047** [0.025, 0.066] | **+0.032** [0.023, 0.049] |
| PR-R, alpha 0.3 (no-harm check) | 0.871 | 0.892 | 0.874 | 0.887 | +0.012 [0.010, 0.014] | -0.006 [-0.023, 0.012] |
| Takeover, L-PR alpha 1, agent 0 (no-harm check) | 0.320 | 0.397 | 0.344 | 0.330 | -0.014 [-0.029, 0.003] | **-0.067** [-0.083, -0.056] |

RC − MADDPG: +0.053 [0.034, 0.069] on L-NR all at 0.3, +0.010 [-0.026, 0.028] on takeover
(`stage1_summary.csv`).

**Go rule: all three parts pass, so Stage 1 is GO.**

1. RC − FT averaged over alpha at $\phi$ 0.99: +0.023 [0.016, 0.029].
2. RC − FT at alpha 0.3 rises with $\phi$: +0.017 (R), +0.021 (0.9), +0.024 (0.99).
3. Clean return: RC −10.99 vs MADDPG −11.02 (+0.03).

**Findings.**

1. **The residual beats its controls on the learned NR attack.** On L-NR all at alpha 0.3, with
   an adversary trained against each frozen team, RC retains 0.746 against 0.699 for FT and 0.713
   for NR-MADDPG. That cell was never part of RC's training mix, which used random sources only. The
   gain over MADDPG (+0.053) matches the Stage 0 oracle ceiling (+0.049 against non-adaptive
   adversaries), so the adaptive attacker did not erase it.
2. **The same training, without the residual, does almost nothing.** FT stays within 0.01 of
   MADDPG in every cell. So the gain comes from the residual architecture, not from extra
   NR training.
3. **RC matches NR-MADDPG under random NR and beats it under learned NR**, without NR-MADDPG's
   1.5M steps of training from scratch.
4. **Mechanism: see the diagnostic below.** RC − FT barely depends on $\phi$ (+0.017 to
   +0.024), and RC gains under i.i.d. R (+0.026 at alpha 0.5) and PR-R (+0.012), where the
   Stage 0 exact oracle gained at most +0.009 and 0. The residual is near zero on clean episodes
   and 15-25x larger on attacked ones (final training logs: mean norm 0.016-0.026 vs 0.37-0.40).
   From these numbers alone I first read the mechanism as attack-triggered behaviour change
   rather than correction. The diagnostic contradicts that reading: under NR the residual does
   compensate for the previous corruption, once it has detected the attack.
5. **No harm on PR**, as P5 predicts no gain there. Takeover is unchanged against MADDPG
   (+0.010, CI includes 0), and NR-MADDPG is better there (0.397), as in Phase 0.

**Caveats.** Three seeds. Settings untuned and shared by RC and FT. FT has no $\beta$ ramp
(it has no $\beta$); its actor updates start at the same episode as RC's. Learned-adversary
training is not bit-reproducible: Phase 0's `train_adversary.py` samples minibatches from NumPy's
unseeded global generator, which `train_adversary_any.py` keeps. A rerun of one Phase 0 adversary
gave -16.14 against the original -15.92, inside that team's spread over adversary seeds
(-15.92, -16.24). In the first pipeline attempt, all evaluation and adversary jobs failed because
a fresh `bash -c` resolved `python` to conda base. They were rerun with the `marl0` interpreter
given by absolute path; training was unaffected (its configs record torch 2.15.0+cpu).

### Stage 1 diagnostic: what the residual changes under attack (done 2026-11-18)

Run: `phase1/diag_stage1.py`, `phase1/diag_summary.py` (action-robust-marl 3553a9b, branch phase-1),
on lab server A. Evaluation only, no training. The same 100 evaluation episodes and random draws as
Stage 1: the logged returns reproduce the Stage 1 numbers exactly. Seeds 0-2, all four teams,
plus an ablation of RC. Every step logs the command, the frozen base command $\mu_b$ at the
same observation, the executed action, the corruption $r_t$, and the state. Summaries:
`results/stage1/diag/` (the per-step logs, 137 MB, stay on the server in `runs_phase1/diag/`).

**Measures.**

- $\delta = \text{cmd} - \mu_b$: how far a team's command departs from the MADDPG base at the
  same observation. For NR-MADDPG, which is a different policy, $\delta$ is not a residual.
- **Fit:** $\delta \approx k_\text{amp}\,\mu_b + k_\text{comp}\,(-r_{t-1})$, least squares pooled
  over steps $t \ge 1$. The Stage 0 "last" oracle has $k_\text{amp} = k_\text{comp} =
  \alpha/(1-\alpha) = 0.43$ at alpha 0.3, before clipping. $k_\text{comp} > 0$ means the
  command leans against the previous corruption.
- **Blind ablation:** RC run with $\tilde a_{t-1}$ replaced by $a_{t-1}$. This removes the
  residual's direct evidence input. Own velocity in the observation still carries the executed
  action, so the evidence is not removed entirely.

**Results**, mean over seeds. L-NR is all agents at alpha 0.3, against each team's own learned
adversaries (seeds 0, 1):

| Cell | Team | $\lVert\delta\rVert$ | $k_\text{comp}$ | $k_\text{amp}$ | $R^2$ | $\rho$ |
| --- | --- | --- | --- | --- | --- | --- |
| Clean | RC | 0.017 | 0.000 | -0.007 | 0.01 | 1 |
| L-NR | MADDPG | 0 | | | | 0.693 |
| L-NR | FT | 0.07 | 0.04 | -0.05 | 0.09 | 0.699 |
| L-NR | NR-MADDPG | 0.96 | 0.49 | -0.86 | 0.73 | 0.714 |
| L-NR | **RC** | 0.49 | **0.39** | -0.51 | 0.65 | **0.746** |
| L-NR | RC blind | 0.08 | 0.02 | -0.07 | 0.08 | 0.699 |
| NR-R 0.3 (i.i.d.) | RC | 0.45 | 0.24 | -0.33 | 0.32 | 0.922 |
| NR-$\mathrm{R}_\phi$ 0.99, 0.3 | RC | 0.49 | 0.28 | -0.40 | 0.40 | 0.902 |

$\lVert\delta\rVert$ for RC by time step under L-NR: 0.02 (t = 0), 0.07 (1), 0.17 (2),
0.26 (3), 0.38 (5), 0.52 (10), 0.66-0.70 (24). On clean episodes it stays at 0.016-0.020.

RC minus RC blind, in $\rho$: +0.041 to +0.053 on L-NR (every seed +0.027 or more), +0.017 to
+0.019 on R, $\mathrm{R}_\phi$ 0.99 and PR-R, 0 on clean.

**Findings.** These replace the reading in Stage 1 finding 4.

1. **Under NR the residual does compensate for the corruption.** On L-NR,
   $k_\text{comp} = 0.39$ against the oracle's 0.43, and the two-term fit explains 65% of
   $\delta$'s variance. FT's command barely moves ($k_\text{comp}$ 0.04), and FT does not gain.
2. **It detects first, then compensates.** At t = 1, $\delta$ is 0.07, where immediate
   compensation would give about 0.4. It grows over about 10 steps as evidence accumulates, and
   stays at the clean-episode level when there is no attack. That gradual onset explains part of
   the gap between RC and the oracle. A residual that is not shut on clean episodes could detect
   faster, but would pay on clean return.
3. **The executed-action input carries all of the gain.** Blinding it collapses $\delta$ to FT's
   level, and $\rho$ to MADDPG's (0.699 vs 0.693). Own velocity in the observation, which still
   reflects the executed action, is not enough for this network.
4. **$k_\text{amp} < 0$, where the oracle has $+0.43$: the residual compensates without
   amplifying.** Command norm stays at the base's (0.94 vs 0.92 under L-NR). A plausible cause
   is saturation: 30% of base action components are already at $|a| > 0.95$ under L-NR. There,
   compensation can only point inward, so $\delta$ correlates negatively with $\mu_b$. This is
   consistent with P4 (only $|a^*| \le 1 - 2\alpha$ is restorable), not a test of it.
5. **Under i.i.d. R the residual still leans against $r_{t-1}$** ($k_\text{comp}$ 0.24, while
   the fit on the current $r_t$ gives 0.004). That gains +0.017 over the blind ablation, although
   $r_{t-1}$ says nothing about $r_t$. The Stage 0 oracle, which applies full compensation for
   $r_{t-1}$, gained ≤ 0.01 there.
   - **Hypothesis tested and rejected (momentum check below):** MPE velocity carries over
     between steps, so a partial push against $r_{t-1}$ might cancel last step's drift. An
     oracle that does exactly this gains at most +0.003 under i.i.d. R.
   - The i.i.d.-noise gain is therefore still unexplained. The behaviour measures point to a
     more conservative policy under attack: lower speed (0.552 vs 0.564), fewer collisions
     (2.46 vs 2.56 per episode) and tighter coverage. This is untested.
6. **NR-MADDPG, a memoryless policy, has the same compensation signature** ($k_\text{comp}$
   0.49, from velocity feedback), and it applies it on clean episodes too. Its $\delta$ is 1.0
   when clean, so it over-commands at all times. RC applies compensation only under attack.
   That fits the clean-cost and takeover trade-off in Phase 0.
7. **The behaviour effect follows.** Under L-NR, RC ends episodes with landmark coverage 0.58
   (sum of nearest-agent distances; MADDPG 0.70) and 2.2 collisions per episode (MADDPG 2.4).

**What this means for the claim.** The P3/P4 correction route is supported:
- the residual learns compensation of close to the oracle's size;
- it learns it from the executed-action evidence;
- it applies it only under attack;
- it does so against an adversary that never appeared in training.

Two things are still open. First, the i.i.d.-noise gain (finding 5) is not explained by
correction, within a step or across steps (momentum check below). P5 can stand as a boundary on
correction, but the residual also gains by other means there. Second, the slow detection
(finding 2) is the obvious lever for closing the gap to the oracle. Stage 2 tests it as RC-fast.

### Stage 1 momentum check (done 2026-11-18)

Run: `phase1/momentum_oracle.py` (action-robust-marl 3553a9b, branch phase-1), frozen Phase 0 MADDPG
teams, seeds 0-2, the Stage 1 evaluation draws. Data: `results/stage1/momentum/`.

MPE integrates $v \leftarrow 0.75\,v + 0.5\,u$ (damping 0.25, sensitivity 5, dt 0.1, mass 1).
The execution error of step $t-1$, $e_{t-1} = \tilde a_{t-1} - a^*_{t-1}$, is therefore still in
the velocity, and 0.75 of it carries into the next step. The oracle command is

$$\text{cmd}_t = \operatorname{clip}\big((a^*_t - \alpha \hat r - c \cdot 0.75\, e_{t-1})/(1-\alpha)\big),$$

with $\hat r \in \{0, r_{t-1}\}$ and $c \in \{0, 0.5, 1, 1.5\}$. At $c = 1$ and alpha 0.3 it
predicts a coefficient on $-r_{t-1}$ of about 0.22 under i.i.d. R, close to RC's 0.24.

Gain in $\rho$ over the same team with NR rescaling only ($\hat r = 0$, $c = 0$), mean over seeds:

| Cell | $\hat r = 0$, best $c$ | $\hat r = r_{t-1}$, $c = 0$ | $\hat r = r_{t-1}$, best $c$ | RC − MADDPG (Stage 1) |
| --- | --- | --- | --- | --- |
| NR-R 0.1 | +0.003 | +0.000 | +0.001 | +0.017 |
| NR-R 0.3 | +0.002 | +0.001 | +0.001 | +0.017 |
| NR-R 0.5 | -0.001 | -0.005 | -0.005 | +0.027 |
| NR-$\mathrm{R}_\phi$ 0.99, 0.3 | +0.003 | +0.002 | +0.004 | +0.022 |
| PR-R 0.3 | -0.001 | | | +0.016 |
| L-NR 0.3 (Phase 0 adversaries, not retrained) | +0.014 | +0.028 | +0.030 | +0.053 (retrained) |

**Result.**

- Cancelling the carried-over momentum does not explain RC's gain under i.i.d. noise: it adds
  at most +0.003, and it hurts at alpha 0.5 and under PR.
- On the learned adversaries, compensating the previous corruption gives +0.028, and adding
  momentum cancellation gives at most +0.002 more.
- Against raw MADDPG the oracle's L-NR gain is larger: the rescaling by $1/(1-\alpha)$ alone
  lifts $\rho$ from 0.693 to 0.711, so the oracle total is about +0.046. That is close to RC's
  +0.053.
- The two numbers are measured differently. Here the adversaries are fixed; in Stage 1 they
  were retrained against RC, which makes RC's number conservative.

### Stage 2 (done 2026-11-19)

Run: `phase1/stage2.sh`, `phase1/stage2_summary.py` (action-robust-marl 3553a9b, branch phase-1), on
lab server B in `~/arm-phase1`, a fresh copy, because `~/action-robust-marl` there holds
unrelated uncommitted edits.
- **Timing:** launched 04:03, training done 04:42, all 40 evaluations and 450 adversaries done
  06:47 (server time). No failed jobs.
- **Design:** as fixed above. Every new kind is on the frozen Phase 0 PR-MADDPG team of its seed;
  seeds 0-4.
- **Data:** `results/stage2/`: `stage2_summary.csv` (all cells, all methods, paired CIs),
  `stage2_per_team.csv`, `stage2_tests.txt`.

**Pre-registered tests** (against PR-MADDPG; outcome rule of framework doc 7.6):

| Test | RC | RC-fast | REC | RES-MLP | FT |
| --- | --- | --- | --- | --- | --- |
| T1 L-NR all 0.3, better than PR-MADDPG and NR-MADDPG | **pass** (+0.042 [0.024, 0.061]; +0.073) | pass (+0.043; +0.074) | pass (+0.033; +0.064) | fail (+0.002) | fail (-0.001) |
| T2 NR cells better in a majority (> 11 of 22), worse in none | **fail: 11 better, 0 worse** | fail: 6, 0 | fail: 5, 0 | fail: 0, 10 worse | fail: 0, 7 worse |
| T3 slope of RC − FT over $\phi$ > 0 | fail: -0.0001 [-0.006, 0.005] | fail | fail | fail | n/a |
| T4 PR no-harm (all 23 cells, lower CI > -0.03) | pass | fail (3 cells) | pass | pass | pass |
| T5 clean no-harm (lower CI > -0.5 return) | pass (+0.02) | pass | pass | pass | pass |
| **Outcome** | **No** (T3) | No | No | No | No |

**Headline cell, L-NR all agents at alpha 0.3** (adversary trained against each final team):

| Method | MADDPG | NR-MADDPG | PR-MADDPG | FT | RES-MLP | REC | RC | RC-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Competence retained | 0.683 | 0.728 | 0.759 | 0.758 | 0.761 | 0.792 | **0.801** | 0.802 |

**T6 attribution**, mean over the 22 NR cells, paired by seed:

| Difference | Mean [95% CI] |
| --- | --- |
| RC − FT (what the residual adds over NR training) | +0.012 [0.005, 0.021] |
| REC − FT (what memory adds) | +0.008 [0.004, 0.013] |
| RC − REC (what the frozen-base split adds) | +0.004 [0.000, 0.008] |
| RC − RES-MLP (what history adds inside the residual) | +0.012 [0.006, 0.020] |
| RC-fast − RC | -0.003 [-0.008, 0.002] |

**Findings.**

1. **RC beats Phase 0's best method on the headline attack.** On L-NR all at alpha 0.3, RC
   retains 0.801 against 0.759 for PR-MADDPG and 0.728 for NR-MADDPG. That closes 17% of
   PR-MADDPG's remaining gap (0.24). The same pattern holds on L-NR one agent 0.3 (+0.021
   [0.014, 0.028]).
   - The test adversaries are unbounded and trained against the final team. Training used only
     a rate-bounded adversary, so the gain transfers from the bounded to the unbounded attacker.
2. **No harm anywhere.** RC passes T4 in all 23 PR cells, including takeover (0.420 vs 0.424).
   Its clean return is slightly better than PR-MADDPG's (+0.02).
3. **T2 misses by one cell.** RC is better than PR-MADDPG in 11 of 22 NR cells and worse in
   none. The 11 "unclear" cells are almost all low-alpha or worst-victim cells. There, PR-MADDPG
   is already at 0.97-0.99, so the room left is below the seed noise: point estimates are
   -0.001 to +0.010.
   - The rule (> half) was fixed before the run and is not relaxed here. The paper reports T2 as
     failed, together with the 11 / 0 count.
4. **T3 fails, as the Stage 1 diagnostic predicted.** RC − FT does not grow with $\phi$ (+0.014,
   +0.010, +0.017). The test assumed the gain scales with how predictable the corruption is. The
   diagnostic showed instead that the residual compensates $r_{t-1}$ after detecting the attack,
   and also gains under i.i.d. noise. So T3 tested the wrong mechanism. Under the pre-registered
   rule, though, its failure makes the outcome **No**, not **Partial**. A revised mechanism test
   would have to be pre-registered for a new run; it cannot be swapped in after the fact.
5. **The architecture is what helps.**
   - FT and RES-MLP, the controls without memory, equal PR-MADDPG or are worse (RES-MLP worse
     in 10 NR cells).
   - Memory alone helps (REC − FT +0.008), and the frozen-base split adds a small further gain
     (RC − REC +0.004, lower CI at 0.000).
   - RC − REC is the comparison that reviewers expect, because RFA and CAPABLE report the same
     direction. It is positive but small.
6. **RC-fast does not help.** It equals RC on the headline cell and is no better on average
   (-0.003). It is less stable: it is better in only 6 NR cells and fails T4 in three PR cells.
   Faster detection is not the bottleneck at this budget.

**Caveats.**
- Five seeds, and all settings untuned (the Stage 1 values).
- The adversaries are not bit-reproducible (Stage 1 caveat).
- The FT, REC and RES-MLP controls share RC's clean penalty against the untouched base. That may
  help their clean return (+0.11 to +0.15 over PR-MADDPG) and limit their robustness.
- M3DDPG, ERNIE-A-c and MIR3 are not in Stage 2 (deferred to the full evaluation). Phase 0
  ranked PR-MADDPG as its best method, which is why it is the bar here.

### Confirmation run on fresh seeds (pre-registered 2026-11-19, before launch)

The user chose this follow-up after Stage 2. Stage 2's pre-registered outcome stays **No**,
whatever this run shows. The confirmation run is a separate, new test. Its design, tests and
outcome rule are fixed here, before any of its data exist.

- **Fresh seeds 5-9.** New Phase 0 bases (MADDPG, PR-MADDPG, NR-MADDPG; `spread`; Phase 0
  `train.py` defaults, 1.5M steps), with their Phase 0 learned adversaries (9 cells × 2 seeds).
  The training code paths of these three methods are unchanged since Phase 0 (`ad57c12`); the
  later edits only add MIR3 branches.
  - **Reproduction check:** PR-MADDPG seed 0 is retrained into `runs_check/` and compared with
    the Phase 0 checkpoint. The check is reported; it does not gate the run.
- **Methods.** RC and FT on the new frozen PR-MADDPG bases, with the Stage 2 settings unchanged
  (`train_stage2.py` defaults). REC, RES-MLP and RC-fast are not rerun.
- **Evaluation.** As in Stage 2: `eval_stage2.py` cells, the 9 core learned cells × 2 adversary
  seeds per team, and the same evaluation episodes.
- **Tests.** T1, T2, T4 and T5 exactly as in Stage 2. **T3 is replaced by T3'** (detect, then
  compensate; `phase1/t3prime.py`). It uses cell L-NR all at alpha 0.3, each team's own two
  adversaries, 100 episodes each:
  - **(a) Compensation:** fit $\delta = \text{cmd} - \mu_b \approx k_\text{amp}\,\mu_b +
    k_\text{comp}\,(-r_{t-1})$ per seed, over steps $t \ge 1$, with $\mu_b$ the frozen base at
    the same observation. Passes if $k_\text{comp}$(RC) − $k_\text{comp}$(FT), paired by seed,
    has a 95% bootstrap CI above 0.
  - **(b) Evidence is causal:** RC-blind is RC with the executed-action input replaced by the
    command, run against RC's own adversaries. Passes if $\rho$(RC) − $\rho$(RC-blind), paired by
    seed, has a 95% bootstrap CI above 0.
  - **T3' passes** if (a) and (b) both pass.
- **Outcome rule.**
  - **Full:** T1, T2, T3', T4 and T5 all pass.
  - **Partial:** T2, T3', T4 and T5 pass, and T1 fails.
  - **Mechanism only:** T3', T4 and T5 pass, and T2 fails.
  - **No:** anything else.

  The rules of T1, T2, T4 and T5, including T2's "more than 11 of 22 NR cells", are unchanged.
- **Exploratory, not part of the test.** A smoke test of `t3prime.py` on Stage 2's seed 0, with
  10 episodes, gave $k_\text{comp}$ 0.42 (RC) vs 0.03 (FT) and $\rho$ 0.794 (RC) vs 0.744
  (RC-blind). Those data were already seen, which is why the confirmation uses seeds 5-9.
- **Where it runs:** lab server B, `~/arm-phase1`, `phase1/confirm.sh`; output in
  `runs_phase1c/`.

### Confirmation run: results (finished 2026-11-19 18:22 server time)

**Pre-registered outcome: No.** T1, T3' and T5 pass. T2 fails (10 of 22 NR cells better; it
needs more than 11). T4 fails narrowly, in one cell (L-PR one agent, alpha 1.0). The mechanism
test T3' passes clearly. Raw outputs are in `results/confirm/`; code is action-robust-marl
`phase1/confirm.sh` and `phase1/t3prime.py`, run on `fefad9c` plus those two files (committed as action-robust-marl `823d524`).

| Test | Result | Verdict |
| --- | --- | --- |
| T1 (L-NR all 0.3) | RC 0.816 vs PR 0.780: +0.036 [+0.022, +0.048]; vs NR 0.729: +0.087 [+0.051, +0.123] | pass |
| T2 (NR cells) | 10 better, 0 worse, of 22 | fail |
| T3' (a) compensation | k_comp RC 0.424 vs FT 0.038: +0.387 [+0.381, +0.393] | pass |
| T3' (b) evidence causal | rho RC 0.816 vs RC-blind 0.785: +0.031 [+0.020, +0.039] | pass |
| T4 (PR cells) | 22 of 23; L-pr-one-a1: RC 0.442 vs PR 0.455, −0.013 [−0.036, +0.009] (bar −0.03) | fail |
| T5 (clean return) | RC −10.589 vs PR −10.563: −0.025 [−0.049, −0.002] (margin −0.5) | pass |

Outcome rule: "Mechanism only" needs T3', T4 and T5; T4 fails, so the label is **No**.

**Findings**

1. **The headline result replicates.** On fresh seeds RC beats PR-MADDPG on L-NR all at
   alpha 0.3 by +0.036 (Stage 2: +0.042), and every one of the 5 seeds is positive
   (+0.009 to +0.050). FT does not (+0.003 [−0.004, +0.011]), so the gain is not from extra
   training on the corruption mix.
2. **The mechanism replicates and is causal on this cell.** RC's compensation coefficient is
   0.41–0.43 on every seed, close to the oracle's alpha/(1−alpha) = 0.43; FT's is 0.03–0.04.
   Blinding the executed-action input costs 0.031 of rho, which is about 86% of RC's gain over
   PR-MADDPG on this cell.
3. **The gain is narrow.** It is clear on the learned NR cells at alpha 0.3 and on the stronger
   or persistent random NR cells (alpha 0.5; phi 0.99), and unclear (point estimates
   +0.000 to +0.010) on alpha 0.1 and on i.i.d. alpha 0.3. The same 10-11 of 22 pattern as
   Stage 2; T2 as written does not pass. T3's phi trend is again not significant (slope
   +0.004 [−0.002, +0.010]; reported, not part of this outcome).
4. **The T4 miss is in the full-replacement PR cell** (an adversary that replaces one agent's
   action outright). Two seeds lose 0.04–0.05 there (seeds 7 and 8), and the lower CI bound
   misses the −0.03 bar by 0.006. Stage 2 passed this cell. The miss is narrow but counts as a
   fail under the pre-registered rule.
5. **Clean cost is small but nonzero.** RC is 0.025 below PR-MADDPG on clean return, a
   statistically clear but tiny loss (the margin is 0.5); in Stage 2 it was not detectable.

**Checks**

- **Base reproduction:** PR-MADDPG seed 0, retrained into `runs_check/`, has actor weights
  identical to the Phase 0 checkpoint (max absolute difference 0.0 on all three agents). The
  fresh bases are therefore Phase 0 bases.
- **No adversary training errors** (all `adv_errors.log` files empty); 270 + 180 adversaries
  and 25 evaluations completed.
- The new PR-MADDPG bases are slightly stronger on L-NR all 0.3 than seeds 0-4 (0.780 vs
  0.759); RC's margin over them is similar.
