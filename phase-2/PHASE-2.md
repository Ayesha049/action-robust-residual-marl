# Phase 2: residual correction under persistent actuator faults (sketch)

Status 2026-11-21. **Draft plan, not pre-registered.** Nothing here has been run. Every number
marked "proposed" is open until the pre-registration block of the stage concerned is written.

### Decisions so far (agreed with the user, 2026-11-21)

| Item | Choice |
| --- | --- |
| Headline threat model | Persistent faults on an agent's own actuator: gain, bias, slow drift (stuck actuator as a negative control) |
| Tasks | MPE `spread` (comparable with Phase 0/1) + VMAS Transport (room for teammate compensation, P6) |
| Base algorithm | MADDPG, frozen Phase 0 checkpoints on `spread`; new MADDPG bases on Transport |
| Success bar | RC on frozen MADDPG must beat **MADDPG** and **REC** (same network, base trained too). Every other method is reported, not required |

### Why this phase

Phase 1 found a real, causal detect-then-compensate mechanism, but on NR mixing the room for any
correction was about 0.05 competence (`history/phase-1/PHASE-1.md`, Stage 0). That ceiling is set
by NR's worst-case replacement: only actions with |a\*| ≤ 1 − 2α can be restored (P4). A
persistent fault is the framework's strongest case (P3): its rate η is 0, so after
identification the loss should not depend on the fault size ε, while the actions still fit the
range. None of the Phase 0 methods has a mechanism to infer a fault within an episode.

---

## 1. Fault conditions

All are L0 or L1 (benchmark `formal_definition.md` section 4). The onset step is drawn uniformly
from {0, …, 12} in a 25-step `spread` episode, so the fault is not present from step 0.

| Fault | Executed action | Strengths (proposed) | Scope | Fully restorable when | Role |
| --- | --- | --- | --- | --- | --- |
| Gain | clip(g · a) | g ∈ {0.7, 0.5, 0.3} | 1 agent (worst), all | \|a\*\| ≤ g | headline candidate |
| Bias | clip(a + b), b a fixed random direction | ‖b‖ ∈ {0.1, 0.3, 0.5} | 1 agent (worst), all | \|a\*\| + \|b\| ≤ 1 per dimension | headline candidate |
| Slow drift | clip(a + b_t), ‖b_t‖ grows by η per step from onset | η ∈ {0.01, 0.02, 0.04} | 1 agent (worst) | as bias, while ‖b_t‖ is small | transfer (P3, η > 0) |
| Stuck actuator | b held from onset | one strength | 1 agent (worst) | never by the agent itself | negative control for the agent's own residual (P5); teammate compensation only |
| Worst-direction bias | bias with b picked from 8 directions × victim, worst return counts | ‖b‖ 0.3 | 1 agent | as bias | cheap worst case, in place of a learned adversary |

"Worst agent" follows the benchmark: each agent is faulted in turn and the lowest return counts.

**Reused unchanged:** the Phase 0 core grid (R/L × NR/PR, takeover). Here it only checks for
harm.

---

## 2. Methods

All sit on the same frozen Phase 0 MADDPG team for each seed on `spread`.

| Method | What it is | Why |
| --- | --- | --- |
| **MADDPG** | Phase 0 base, unchanged | bar 1 |
| **RC** | Phase 1 residual: GRU 64 over [o, a_{t-1}, ã_{t-1}, μ_b], zero-initialised head, gate, frozen base | the method |
| **REC** | RC's network with the base weights trained as well; same clean penalty against the untouched base | bar 2: does the frozen-base split itself help? |
| **FT** | MADDPG fine-tuned on the same fault mix, no memory | shows the gain is not just extra training |
| RC-blind | RC with ã_{t-1} replaced by a_{t-1} (evaluation only) | mechanism test |
| Oracle | analytic Δ\* = (G⁻¹ − I)μ_b − G⁻¹b with known G, b, clipped | ceiling (Stage 0) |
| PR-MADDPG, NR-MADDPG, M3DDPG, ERNIE-A-c | Phase 0 checkpoints, evaluation only | reported for context; not part of the success bar |

**Risk to flag now:** in Phase 1, RC − REC was only +0.004 [0.000, 0.008]. With REC in the
success bar, the outcome depends on the split mattering more under persistent faults than under
NR. The argument for it: REC's base drifts as it trains on fault episodes, while RC keeps the
clean policy exactly and only has to learn Δ\*. That argument is the hypothesis being tested
(H4 in `history/phase-1/residual-hypotheses.md` predicts the opposite for NR).

---

## 3. Stages

### Stage 0: headroom check on Phase 0 checkpoints (no training, about 2 days)

- Evaluate every Phase 0 `spread` method (5 seeds, 100 episodes per cell, Phase 0 evaluation
  seeds) on every fault cell. This gives the first row of the Phase 2 results for free.
- **Oracle correction** on frozen MADDPG, with three predictors: none, exact (G and b known from
  onset), and delayed (G and b known k steps after onset, k ∈ {1, 3, 10}). The k = 10 case
  matches Phase 1's observed detection time.
- **Restorable share:** the fraction of MADDPG's clean action components with |a\*| ≤ g and with
  |a\*| + |b| ≤ 1.
- **Stop rule (proposed):** stop the `spread` arm if the exact oracle gains less than 0.05
  competence over MADDPG in every gain and bias cell. That would mean `spread` has no more room
  for this fault than it had for NR. Transport would then carry the phase alone.

**Fixed before the Stage 0 run (2026-11-21, launched 20:49 server time).**

- **Code:** action-robust-marl branch `phase-2`: `phase2/faults.py`, `phase2/stage0_faults.py`,
  `phase2/stage0_summary.py`, `phase2/stage0.sh`. Deployed to a fresh `~/arm-phase2` on both lab
  servers. Checkpoints are linked read-only.
- **Teams:** seeds 0-4 only, so seeds 5-9 stay unseen for Stage 2.
  - Lab server A (`~/action-robust-marl/runs`): MADDPG, PR-MADDPG, NR-MADDPG, MARTA.
  - Lab server B: M3DDPG (`runs`), ERNIE-A-c λ 1, ε 0.05 (`runs_ernie`), MIR3 λ 0.0005
    (`runs_mir3/main`).
- **Predictors:** the oracle (none, exact, d1, d3, d10) runs on MADDPG and PR-MADDPG. The other
  methods run with no correction only.
- **Random draws:** one onset t0 ~ U{0..12} and one fault direction per (fault kind, episode).
  These are shared across methods, predictors, strengths and scopes, so every comparison is
  paired.
- **Worst agent:** per team, the victim with the lowest ρ under "none" (the Phase 0 rule). The
  oracle is scored on that same victim.
- **Stop rule:** as above, threshold 0.05 exact-oracle gain for MADDPG on gain or bias cells.
- **Smoke test:** on a team with random weights, the exact oracle restores ρ to 1.000 in every
  cell with no clipping. This confirms the channel and the correction agree.

### Stage 0 results (done 2026-11-21, 20:49-20:52 server time)

Run: 85 jobs (70 on A, 15 on B), no errors. Data: `results/stage0/` (per-episode CSVs, plus
`rho_none.csv`, `oracle_gains.csv`, `per_team.csv`, `summary_printout.txt`). Summarised locally
with `phase2/stage0_summary.py`.

**Sanity check.** The clean returns of MADDPG, PR-MADDPG and NR-MADDPG reproduce Phase 1
Stage 0 exactly (15 teams, maximum difference 0.0000).

**Verdict: STOP for the `spread` arm, by the pre-set rule.** The largest exact-oracle gain for
MADDPG on any gain or bias cell is **+0.016** [0.009, 0.020] (bias 0.5, all agents), against a
threshold of 0.05.

**Competence retained with no correction** (mean over seeds 0-4; worst1 = worst single victim):

| Cell | MADDPG | PR-MADDPG | NR-MADDPG | M3DDPG | ERNIE-A-c | MIR3 |
| --- | --- | --- | --- | --- | --- | --- |
| bias 0.3, all | 0.983 | 0.994 | 0.986 | 0.981 | 0.987 | 0.981 |
| bias 0.5, all | 0.965 | 0.978 | 0.971 | 0.961 | 0.963 | 0.961 |
| worst-direction bias 0.3 | 0.985 | 0.987 | 0.986 | 0.986 | 0.986 | 0.986 |
| drift 0.04, worst1 | 0.990 | 0.996 | 0.994 | 0.992 | 0.992 | 0.991 |
| gain 0.5, all | 0.936 | 0.952 | 0.940 | 0.931 | 0.920 | 0.927 |
| gain 0.3, all | 0.873 | 0.901 | 0.884 | 0.876 | 0.855 | 0.861 |
| stuck, worst1 | 0.676 | 0.717 | 0.685 | 0.678 | 0.667 | 0.648 |

**Exact-oracle gain on MADDPG** (95% CI) and the share of corrected components it had to clip:

| Cell | Gain | Clipped |
| --- | --- | --- |
| bias 0.5, all | +0.016 [0.009, 0.020] | 14% |
| bias 0.3, all | +0.006 [0.003, 0.010] | 9% |
| worst-direction bias 0.3 | +0.008 [0.006, 0.010] | 9% |
| drift 0.04, worst1 | +0.007 [0.006, 0.008] | 12% |
| gain 0.3, all | +0.003 [-0.000, 0.006] | 45% |
| gain 0.5, all | +0.001 [-0.000, 0.002] | 35% |

The delayed oracles keep most of the exact gain (d1 ≈ exact; d10 roughly half or more). So
identification time is not what limits it.

**Findings.**

1. **Bias and drift barely hurt on `spread`.** Every method keeps ρ ≥ 0.96 even at ‖b‖ 0.5 on
   all agents. The base is a closed-loop policy on relative positions, so it steers back
   against a constant push without knowing about it. There is almost nothing left to correct:
   this is P5's state-feedback case, but for faults.
2. **Gain faults hurt, but they are an authority limit.** At g 0.3 on all agents ρ falls to 0.87,
   yet the exact oracle recovers +0.003. Up to 45% of components would need |a| > 1. The loss is
   speed the agent no longer has, and no command can give it back.
3. **Stuck actuator is the only big loss** (ρ 0.65-0.72, headroom about 0.3). The agent's own
   residual cannot help (P5). Any gain has to come from teammates covering the stuck agent's
   landmark, which is a behaviour change, not P6-style force compensation.
4. **PR-MADDPG is the most fault-robust Phase 0 method** in every bias, drift and gain cell, as
   it was on the Phase 0 grid.
5. **MARTA is excluded.** Its mean clean return is -43.15, below the random team's (-26.08).
   This is the known failed adaptation on `spread` (PHASE-0.md, open item "MARTA adaptation
   failed on `spread`"), so its ρ values are not meaningful.

**Implication.** On `spread` the room for a persistent-fault corrector (≤ 0.016) is smaller than
the room was for NR in Phase 1 (0.05). Following the plan, Transport carries the phase. Raising
the fault strengths now would be a decision made after seeing the data. If it is done, it has
to be labelled exploratory.

### Stage 0, exploratory: stronger bias (done 2026-11-21, after the Stage 0 verdict)

**Exploratory.** These strengths were chosen after the Stage 0 data was seen, so this run does
not change the STOP verdict.

- **Cells:** bias ‖b‖ 0.7 and 0.9, all agents and worst single victim. Same onsets and directions
  as the Stage 0 bias cells.
- **Teams:** the same as Stage 0, with MARTA excluded.
- **Code:** `phase2/stage0_faults.py --strengths bias=0.7,0.9`.
- **Data:** `results/stage0x/`. Clean returns come from Stage 0.

| ‖b‖ (all agents) | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
| --- | --- | --- | --- | --- | --- |
| MADDPG ρ, no correction | 0.996 | 0.983 | 0.965 | 0.936 | 0.875 |
| PR-MADDPG ρ, no correction | 0.999 | 0.994 | 0.978 | 0.955 | 0.908 |
| MADDPG exact-oracle gain | +0.001 | +0.006 | +0.016 | +0.029 [0.024, 0.033] | +0.056 [0.050, 0.063] |
| MADDPG d10-oracle gain | +0.001 | +0.005 | +0.010 | +0.015 | +0.022 |
| share clipped (exact) | 5% | 9% | 14% | 18% | 25% |

**Reading.**

- **The room grows faster than linearly with fault size**, but it only reaches the 0.05 bar at
  ‖b‖ 0.9. A fault that large takes up almost the whole action range in its direction.
- **Even then it barely beats the best existing method.** MADDPG plus a perfect corrector reaches
  about 0.93, against 0.908 for PR-MADDPG with no correction. A detector that takes 10 steps gets
  0.897, below PR-MADDPG.
- **This is the same ceiling as Phase 1's NR result** (about 0.05). On `spread`, the closed-loop
  base absorbs small faults by itself, and large faults run into the action limits.
- **Decision stands:** move to Transport. This sweep also suggests what Transport's Stage 0
  should include: bias up to about 0.9, so the dose-response is visible before strengths are
  fixed.

### Stage 1: `spread`, development (seeds 0-4, about 1.5 weeks)

- **Training mix (proposed):**
  - clean 0.25;
  - gain 0.25, bias 0.25, drift 0.25, with strengths drawn from continuous ranges that cover the
    test values;
  - victim scope 1 agent or all, chosen at random.
  - Stuck actuator is **held out** of training.
- **Settings:** the Phase 1 hyperparameters, as the starting point. Limited tuning is allowed
  here (β, clean penalty, GRU size) on seeds 0-4 only. Each setting tried is logged, so the
  search can be reported.
- **Output:** a frozen configuration and the final test list, written into the Stage 2
  pre-registration.

### Stage 2: `spread`, confirmation (seeds 5-9, pre-registered, about 3 days)

The Phase 0-style MADDPG bases for seeds 5-9 already exist from the Phase 1 confirmation run. No
fault data has been run on them. Settings are frozen from Stage 1, and the tests are those in
section 4.

### Stage 3 progress: Transport port and gate pilot (started 2026-11-21)

- **Environment:** `marl/envs.py` `VmasTeamEnv`, using vmas 1.5.2 with the BenchMARL `transport`
  settings: 4 agents, 1 package, mass 50, 100 steps.
  - It runs a fixed horizon and ignores VMAS's done. That matches Phase 0's replay, which stores
    no done flag, and the reward is 0 while the package stays on the goal.
  - Action f ∈ [-1, 1]² is the VMAS force (u_range 1, u_multiplier 0.6). Each agent observes 11
    values: position, velocity, package minus goal, package minus agent, package velocity,
    on-goal flag.
- **Conda env:** `marl2` on both servers, a clone of `marl0` plus vmas. Torch is unchanged
  (2.15.0+cpu), and `marl0` is untouched.
- **Reference policies:**
  - A random or zero team scores 0.00, so J_rand is about 0.
  - The VMAS heuristic and a scripted "get behind and push" policy score about 5 in 100 steps
    and deliver nothing. The agents slide off the box's edges, so the task needs learned
    closed-loop pushing.
  - The VMAS paper solved Transport with PPO variants (IPPO, MAPPO) at about 24M environment
    steps and did not test MADDPG. **MADDPG learning it is the main risk.**
- **Gate pilot:** tuning seeds 100-102 only, so seeds 0-9 stay unused.
  - Lab server B: γ 0.95 and 0.99, 3M steps (`phase2/transport_pilot.sh`), launched 21:01.
  - Lab server A, as a hedge: γ 0.99, 10M steps.
  - Gate as planned: package delivered in at least 80% of clean evaluation episodes
    (`phase2/eval_clean.py`).

### Stage 3: VMAS Transport (about 2 weeks)

- **Port:** move the Phase 0 MADDPG code to VMAS Transport (roughly 3-5 days). Check the port on
  clean training before doing anything else.
- **Gate:** clean MADDPG must reach a working policy (proposed: package delivered in at least
  80% of episodes). Otherwise stop and report, rather than tune the base for weeks.
- **Bases and runs:** 10 MADDPG bases (seeds 0-4 for development, 5-9 for confirmation), then
  RC, REC and FT, with the same fault rows.
- **Teammate test (P6):** with one agent stuck or biased, do the healthy agents' residuals move
  against the fault? Fit δ_healthy against the faulty agent's error, as Phase 1 did with
  k_comp.

---

## 4. Tests (proposed; fixed in each confirmation's pre-registration)

δ follows the benchmark rule (`shared/environments_and_budgets.md` section 6): the paired 95% CI
excludes 0, **and** the difference is at least 5% of the gap between clean and random-team
return. Competence retained is reported as in Phase 0/1.

| Test | Claim | Pass rule |
| --- | --- | --- |
| F1 headline | RC beats MADDPG on bias, 1 agent, ‖b‖ 0.3 (or gain 0.5, all agents; choose one after Stage 0) | RC − MADDPG exceeds δ |
| F2 breadth | RC beats MADDPG across gain, bias and drift cells | better in a majority, worse in none |
| F3 split | RC beats REC | mean RC − REC over the F2 cells: CI > 0 |
| F4 dose-response (P3) | RC's loss does not grow with fault size the way MADDPG's does | per-seed slope of (MADDPG loss − RC loss) against ‖b‖: CI > 0 |
| F5 mechanism | RC learns the oracle correction from executed-action evidence | (a) the fit δ ≈ k·Δ\* after onset gives k(RC) − k(FT) with CI > 0; (b) ρ(RC) − ρ(RC-blind): CI > 0 |
| F6 no harm | no loss on the Phase 0 core grid or stuck actuator | lower CI of RC − MADDPG > −δ in every cell |
| F7 clean | clean return | lower CI of RC − MADDPG > −0.5 return |

**Outcome rule (proposed):**
- **Full:** F1-F7 all pass.
- **Without split:** everything passes except F3. The claim then becomes "recurrent fault
  correction", not "residual on a frozen base".
- **Mechanism only:** F5, F6 and F7 pass.
- **No:** anything else.

Identification time is reported, not tested: the steps after onset until δ reaches half of Δ\*.

---

## 5. Time and compute

| Block | Effort | Compute |
| --- | --- | --- |
| Fault channel + oracle + Stage 0 | 2-3 days | < 1 h (lab server A) |
| Stage 1 (dev, with limited tuning) | 1-1.5 weeks | a few hours per sweep |
| Stage 2 confirmation | 2-3 days | < 1 day (lab server B) |
| VMAS port + gate | 3-5 days | - |
| Stage 3 dev + confirmation | about 1 week | about 1 day |
| **Total** | **about 4.5-5.5 weeks** | |

---

## 6. Still open (to settle before Stage 0)

1. Fault strengths and onset range (section 1 values are first guesses).
2. Whether NR stays in the training mix (it would keep the Phase 1 result alive, but would blur
   the persistent-fault claim).
3. Whether tuning in Stage 1 is acceptable, given Phase 1 was untuned. Tuning helps RC but must
   also be applied to REC and FT.
4. Whether F3 should be required for the headline, given Phase 1's +0.004, or reported as the
   "without split" outcome (as drafted).
5. Whether to add an adapted (learned) fault adversary, for example one that picks the bias
   direction per episode from the state. The worst-direction grid is the cheaper stand-in.
