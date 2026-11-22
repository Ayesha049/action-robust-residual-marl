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

### Decision, 2026-11-21 ~22:00: Transport moves to a MAPPO base; MADDPG gets a second task

- **Why:** at about 2.1M of 3M steps, all 6 runs of the gate pilot on server B were flat at a
  return of about 0 from 0.5M steps on (the last 100 episodes averaged −0.01 to +0.06, against a
  random team's 0.00). The 10M hedge on server A is left to finish as a data point only.
- **Transport base:** MAPPO, our own minimal implementation in the repo (same env wrapper,
  checkpoints and evaluation as MADDPG). It must reproduce the VMAS paper's Transport learning
  before it is trusted. RC and REC wrap the MAPPO actor's mean action. The gate stays: package
  delivered in at least 80% of clean episodes.
- **MADDPG's second task:** screen MPE `tag` first, using the existing Phase 0 checkpoints. If
  tag stops like spread did, move to VMAS `balance`.

### Stage 0 on `tag` (setup fixed before the run, 2026-11-21)

- **Teams:** the Phase 0 tag checkpoints, seeds 0-4. The team is 3 predators against the frozen
  Phase 0 prey (`assets/tag_prey.pt`, the same md5 on server A and locally).
  - Methods: MADDPG and PR-MADDPG with every predictor. NR and MARTA with no correction only.
    MARTA is reported but excluded from conclusions, as on spread.
- **Return scale:** J_rand = 1.05 (the Phase 0 value). A check on 100 episodes gave a random team
  1.00 and MADDPG s0 J0 = 18.0, with sd 22.
- **Episodes:** 500 per cell, not 100. Tag returns come in steps of 10 (catches), so 100 episodes
  would give a ρ standard error of about 0.1 per team. Pairing on shared episode seeds reduces
  this further.
- **Cells:** as on spread, except bias also gets ‖b‖ 0.7 and 0.9, so the full response curve is
  in the pre-set grid: gain {0.7, 0.5, 0.3}, bias {0.1, 0.3, 0.5, 0.7, 0.9}, drift {0.01, 0.02,
  0.04}, stuck, wdir 0.3. Onset is U{0..12} (25-step episodes, as on spread).
- **Stop rule (the same as spread):** stop the tag arm if the exact oracle on MADDPG gains less
  than 0.05 ρ in every gain and bias cell. The paired bootstrap CI is reported next to each gain.
- **Run:** lab server A, `TASK=tag STRENGTHS="bias=0.1,0.3,0.5,0.7,0.9" phase2/stage0.sh`, output
  in `runs_phase2/stage0_tag/`.

### Stage 0 on `tag` results (done 2026-11-21, 22:11-22:26 on server A): STOP

Files are in `results/stage0_tag/`: the per-episode CSVs (gzipped), the summaries and
`summary_printout.txt`. The 70 jobs ran with no errors. Clean J0: MADDPG 18.2, PR 18.1, NR 16.1,
MARTA 10.8.

- **Unlike spread, faults hurt a lot.** MADDPG ρ with no correction:

  | Bias ‖b‖ (all / worst single agent) | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
  | --- | --- | --- | --- | --- | --- |
  | all agents | 0.94 | 0.82 | 0.66 | 0.51 | 0.38 |
  | worst single agent | 0.95 | 0.86 | 0.73 | 0.62 | 0.54 |

  Gain g 0.7 / 0.5 / 0.3 on all agents gives ρ 0.51 / 0.21 / 0.10. A stuck single agent gives
  0.27. Drift at η 0.04 gives 0.83.
- **The oracle still recovers almost nothing.** The best MADDPG exact-oracle gain on a gain or
  bias cell is +0.036 (bias 0.9, worst single agent; CI 0.019-0.052). The worst-direction
  grid gives +0.041. **STOP for the tag arm by the pre-set rule.**
- **Why: the predators run at full force.** The exact oracle has to clip 42% of corrected
  components even at ‖b‖ 0.1, and 92-98% under gain faults. On spread these shares were 5% and
  24-47%.
  - Directly measured on clean episodes (50 per team): 78-86% of the tag predators' action
    components have |a| > 0.95, and mean |a| is 0.92-0.95. On spread the shares are 15-18%, with
    mean |a| 0.37-0.57.
  - A chasing policy is bang-bang, so any loss of authority cannot be made up from within
    [-1, 1]. This is P4's restorable-set limit, now binding in nearly every step.
- **Summary of the two MPE tasks:** on spread, faults barely hurt (the agents re-steer from
  positions). On tag, faults hurt, but the base already uses the whole actuator range. Neither
  leaves room for a correction. Any task where a DDPG-style policy saturates (likely also
  Balance and MuJoCo) will hit the tag problem.
- **PR-MADDPG** is again no less fault-robust than MADDPG: ρ is within ±0.03 in most cells, and
  its oracle gains are similar (≤ +0.046).

### Decision, 2026-11-21 ~22:40: reserved actuator authority on `tag`

Chosen by the user after the tag STOP above. The MADDPG Transport pilot is also final (3M steps,
all 6 runs delivered in 0% of clean episodes), which confirms the MAPPO switch.

- **Mechanism:** the base is trained with its actions scaled by c before the actuator
  (`--act-cap c`, `marl/envs.py`). The actuator still accepts [-1, 1], so the base uses at most c
  of the range and leaves 1 − c for the correction. This is the standard control-allocation
  reserve of fault-tolerant control.
  - `load_team` returns c·actor(o) in actuator units, so faults, the oracle and the clip share
    are all measured against the full ±1 actuator.
  - By P4: a bias ‖b‖ is restorable while c + |b_i| ≤ 1, and a gain g while c ≤ g.
- **Setup fixed before the run:**
  - c ∈ {0.5, 0.7}. MADDPG with the Phase 0 tag settings (1.5M steps, frozen Phase 0 prey).
    Development seeds 0-4 only.
  - Then the tag Stage 0 above, unchanged (the same grid, 500 episodes, the same stop rule), on
    each cap.
  - Also reported: the clean cost of the reserve, J0(cap) against the uncapped Phase 0 J0 of
    18.2.
  - The confirmation baseline must be the capped MADDPG base itself. That RC with a reserve
    beats an uncapped MADDPG is a separate question, and it will be reported.
- **Run:** lab server A, `phase2/tag_cap.sh`, output in `runs_phase2/tag_cap<c>/`.

### Reserved authority on `tag`: results (2026-11-21, 22:40-23:14 on server A)

Files are in `results/stage0_tagcap0.5/` and `results/stage0_tagcap0.7/` (the Stage 0 outputs and
training logs), plus `results/tag_cap_vs_uncapped_returns.csv`.

- **The cap costs most of the clean performance.** J0 is 2.12 at c 0.5 (close to random, 1.05)
  and 8.22 at c 0.7, against 18.24 uncapped.
  - The predators chase a full-authority prey, so in pursuit, actuator authority *is*
    performance. ρ at c 0.5 is degenerate (J0 − J_rand ≈ 1), so c 0.5 is unusable.
- **The headroom does open.** At c 0.7, the stop rule says GO. The exact-oracle gain is
  +0.62 at gain 0.7 (all agents), +0.36 at bias 0.5 (all), and +0.39 on the worst-direction grid.
  The d10 predictor recovers about a third of that (≤ +0.20).
- **But the reserve is dominated.** In absolute return, the uncapped base with *no* correction
  beats the c 0.7 base with the *exact* oracle in every cell:

  | Cell | uncapped, no correction | c 0.7, exact oracle | c 0.7, d10 |
  | --- | --- | --- | --- |
  | bias 0.3, all | 15.1 | 8.2 | 7.0 |
  | bias 0.9, all | 7.5 | 4.6 | 3.3 |
  | gain 0.7, all | 9.8 | 8.2 | 5.2 |
  | gain 0.5, all | 4.6 | 4.0 | 2.8 |
  | worst direction 0.3 | 15.3 | 8.2 | 6.5 |

- **Verdict:** a GO on paper, but not a usable arm. RC on a capped base could beat capped MADDPG
  and REC, yet it would lose to plain uncapped MADDPG under every fault tested. That fails the
  obvious reviewer comparison.

### Decision, 2026-11-21 ~23:25: rotation faults on uncapped `tag` (setup fixed before the run)

Chosen by the user after the reserve result above. This widens the threat model from gain, bias
and drift to faults that turn the force rather than shrink it: a misaligned or mis-wired
actuator.

- **Fault:** `rot`, exe_j = clip(R(σ_j θ) cmd_j), with σ_j = ±1 drawn per episode and θ ∈ {30,
  60, 90, 180}°. Victims are each single agent, and all agents. Onset is U{0..12}, as before.
  The oracle command is clip(R(−σ_j θ) a*).
  - 90° and 180° map the action box onto itself, so they are always exactly restorable
    (unit-tested: zero restoration error and zero clipping).
  - 30° and 60° lose only the part of a corner action that leaves the box. Uniform test
    actions clip 7-8% of components.
- **Teams:** the uncapped Phase 0 tag bases, seeds 0-4. MADDPG and PR-MADDPG with every
  predictor, and NR with no correction. 500 episodes per cell.
- **Stop rule:** GO for the tag arm if the exact oracle on MADDPG gains at least 0.05 ρ in some
  `rot` cell. A cell where the d10 oracle also gains at least 0.05 marks it as reachable by a
  causal corrector.
- **Also run, exploratory:** the same rotation grid on spread (seeds 0-4, 100 episodes, MADDPG
  and PR), for the self-correction story.
- **Run:** lab server A. `KINDS="clean rot" RULE=rot phase2/stage0.sh` writes to
  `runs_phase2/stage0_tagrot/` and `runs_phase2/stage0_spreadrot/`.

### Rotation faults: results (2026-11-22, 00:07-00:30 on server A): GO on tag and on spread

Files are in `results/stage0_tagrot/` and `results/stage0_spreadrot/`. All runs completed with
no errors. Clean J0 reproduces exactly (MADDPG tag 18.24, spread −11.12).

- **Rotation faults hurt, and no existing method resists them.** ρ with no correction:

  | θ (all agents / worst single agent) | 30° | 60° | 90° | 180° |
  | --- | --- | --- | --- | --- |
  | tag, MADDPG | 0.45 / 0.60 | 0.16 / 0.38 | 0.09 / 0.29 | 0.05 / 0.23 |
  | tag, PR-MADDPG | 0.48 / 0.58 | 0.17 / 0.34 | 0.10 / 0.27 | 0.05 / 0.19 |
  | tag, NR-MADDPG | 0.51 / 0.62 | 0.17 / 0.36 | 0.10 / 0.28 | 0.05 / 0.20 |
  | spread, MADDPG | 0.97 / 0.99 | 0.83 / 0.93 | 0.22 / 0.75 | −2.04 / 0.51 |

  PR and NR training buy nothing against rotation (within ±0.06 of MADDPG).
- **The oracle restores almost everything, at full authority.** Exact-oracle gains on MADDPG:

  | Cell | tag exact | tag d1 / d3 / d10 | spread exact | spread d1 / d3 / d10 |
  | --- | --- | --- | --- | --- |
  | 30°, all | +0.18 | 0.19 / 0.16 / 0.06 | +0.01 | 0.01 / 0.01 / 0.00 |
  | 60°, all | +0.48 | 0.43 / 0.33 / 0.07 | +0.15 | 0.14 / 0.12 / 0.05 |
  | 90°, all | +0.91 | 0.79 / 0.49 / 0.06 | +0.78 | 0.76 / 0.69 / 0.31 |
  | 180°, all | +0.95 | 0.71 / 0.29 / 0.00 | +3.04 | 2.99 / 2.82 / 1.41 |
  | 90°, worst single | +0.71 | 0.66 / 0.44 / 0.08 | +0.25 | 0.24 / 0.22 / 0.09 |

  - At 90° and 180° the clip share is 0. At 30° and 60° on tag it is about 0.46 (the
    saturated corners), and the gains are still large.
- **Detection speed is what matters on tag.** A pursuit is lost within a few steps, so d10 keeps
  only ≤ 0.08 of the gain, while d1 keeps most of it. A corrector has to identify the
  rotation within about 1-3 steps of onset, which is what executed-action evidence (ã_{t-1})
  gives in principle. On spread, the slower task, d10 still keeps +0.31 at 90° and +1.41 at
  180°.
- **Verdict:** GO for both MADDPG arms by the pre-set rule. This is the first fault class with
  large, restorable headroom on an uncapped base. Next is Stage 1 (development on seeds 0-4)
  with rotation as the headline fault.

### Transport gate check (2026-11-22): the 80%-delivery gate is infeasible at mass 50

- **Idealised pusher:** agents start right behind the package and steer it straight at the goal.
  At mass 50 it delivers in 0-1% of 100-step episodes, and in 7-12% even within 300-500 steps.
  The package reaches about 0.007 units/s against a median start distance of about 1.0.
  - Mass 10: 6% delivered within 100 steps, 38% within 300. Mass 5: 13% and 57%.
- **Return ceiling at mass 50 and 100 steps:** a scripted walk-up pusher scores 2.6 (sd 3.5),
  and 10.3 when it starts behind the package.
- **MAPPO pilot:**
  - Fast settings at 24M: clean return 0.24-0.59 and 0% delivered. The policy std collapsed to
    about 0.05, so it has converged to a poor solution.
  - Paper settings at about 9M of 24M: 0.23-0.32 and rising.
  - Both are below the crude script, so the Transport arm needs a decision on the task setup
    *and* on whether our MAPPO is good enough.

### Decisions, 2026-11-22 (user)

- **Rotation is the headline fault** on spread and tag. Gain, bias and drift are reported as the
  authority-limited contrast, explained by P4. Stage 1 development (seeds 0-4) trains RC, REC
  and FT on rotation faults.
- **Transport:** a lighter package and a longer horizon (proposed: mass 10, 300 steps). Re-pilot
  MAPPO there, with a gate relative to the scripted pusher's return.

### Transport m10 re-pilot (setup fixed before the run, 2026-11-22)

- **Task:** `transport_m10`, which is VMAS transport with package mass 10 and a fixed horizon of
  300 steps. Otherwise as before (4 agents, u_multiplier 0.6).
- **References** (100 clean episodes on the evaluation seeds, `eval_clean.py --policy`):
  - Scripted walk-up pusher: return 46.9 (sd 34.1), delivered at some point in 43% of episodes,
    on the goal at the end in 35%.
  - Random team: 0.00.
- **Gate:** a MAPPO configuration passes if its mean over the 3 tuning seeds has clean return
  ≥ 46.9 **and** ever-delivered ≥ 0.43, which means at least matching the scripted pusher.
  The configuration that passes with the highest return becomes the base for seeds 0-9.
- **Pilot:** seeds 100-102 × {paper, fast, fast_ent (fast with entropy bonus 0.01)}, 24M frames
  each, on lab server B (`TASK=transport_m10 CONFIGS="paper fast fast_ent"
  phase2/mappo_pilot.sh`, output in `runs_phase2/mappo_m10/`).
- The mass-50 paper-settings runs are left to finish, as a check of the MAPPO implementation
  against the scripted 2.8 on the standard task.

### Stage 1 development run 1 (launched 2026-11-22 00:28, lab server A)

- **Trainer:** `phase2/train_fault.py`, which is Phase 1 Stage 1 with the NR channel replaced by
  the fault channel. Every fault is affine per step, exe = clip(M_t cmd + b_t), so the channel
  is differentiable and (M_t, b_t) is the critic's privileged context.
- **Training mix (development default, untuned):**
  - Clean with probability 0.25. Otherwise a rotation fault with weight 1/2 (θ ~ U[0, 180]°,
    random sign), or gain, bias or drift with weight 1/6 each (g ~ U[0.3, 1], ‖b‖ ~ U[0, 0.9],
    η ~ U[0, 0.04]).
  - Victims: one random agent or all agents, 50/50. Onset U{0..12}.
  - The stuck actuator stays out of training as the negative control.
- **Hyperparameters:** Phase 1 values, untuned: 20k episodes, β 1, c_r 1, GRU 64.
- **Runs:** tag and spread × MADDPG bases seeds 0-4 × {RC, RC-blind, REC, FT}, 40 runs.
- **Evaluation:** every Stage 0 cell, including rot and bias up to 0.9, with no oracle. It uses
  the Stage 0 seeds and RNG keys, so each cell pairs with the base's Stage 0 rows. 500 episodes
  on tag, 100 on spread.
- **Read-out (development; the confirmation tests are fixed later):** RC − MADDPG, RC − REC and
  RC − RC-blind per cell, in absolute return, plus the clean cost.

### Stage 1 development run 1: results (2026-11-22, 00:28-01:13 on server A)

All 40 runs and 280 evaluation jobs completed with no errors. Summaries are in
`runs_phase2/stage1/summary_*.txt` (`phase2/stage1_summary.py`, ρ on the base's scale).

- **Tag: no method learned anything.** RC, REC, RC-blind and FT are all within about ±0.02 of
  MADDPG on every rotation cell. For example, at 90° on all agents: MADDPG 0.089, RC 0.087, REC
  0.089. Clean cost is nil.
- **Spread: partial gains, but REC is ahead, and the evidence input is unused.**

  | Cell | MADDPG | RC | REC | RC-blind | FT |
  | --- | --- | --- | --- | --- | --- |
  | rot 90°, all | 0.22 | 0.45 | 0.60 | 0.43 | 0.34 |
  | rot 180°, all | −2.04 | −1.70 | −0.40 | −1.79 | −1.43 |
  | rot 90°, worst single | 0.75 | 0.81 | 0.86 | 0.81 | 0.78 |
  | bias 0.9, all | 0.875 | 0.80 | 0.68 | 0.81 | 0.84 |

  - RC ≈ RC-blind, so the gain comes from the residual's state dependence, not from the
    executed-action evidence.
  - All trained methods lose 0.02-0.2 on large bias, a fault their training never made
    worthwhile to fix.
- **Diagnosis: the Phase 1 residual cannot express the correction.** The residual acts before the
  squash, mu = tanh(u_b + g β tanh(D)) with β = 1, so per component |Δu| ≤ 1.
  - Measured on clean states (5 seeds × 40 episodes): the tag base's pre-tanh outputs have a
    median |u_b| of 5.9 (81% exceed 2).
  - The share of oracle rotation commands reachable within the bound:

    | θ | tag β 1 | tag β 3 | tag β 5 | spread β 1 | spread β 3 | spread β 5 |
    | --- | --- | --- | --- | --- | --- | --- |
    | 30° | 0.04 | 0.17 | 0.31 | 0.67 | 0.87 | 0.91 |
    | 90° | 0.01 | 0.05 | 0.11 | 0.48 | 0.77 | 0.85 |
    | 180° | 0.00 | 0.02 | 0.06 | 0.39 | 0.72 | 0.79 |

  - On a saturating base, a pre-tanh residual is pinned by the saturation itself: it would need
    a huge Δu just to come off the flat part of tanh. This is a property of the
    parameterisation, not of training. It probably also capped Phase 1 on spread.

### Transport m10 gate pilot: results (finished 2026-11-22 05:35): PASS with `fast_ent`

Files are in `results/transport_pilot/` (clean evaluations and the m10 training logs; MAPPO and
MADDPG at mass 50 too).

| Config (mean over tuning seeds 100-102) | Clean return | Ever delivered | On goal at end |
| --- | --- | --- | --- |
| scripted pusher (reference) | 46.9 | 0.43 | 0.35 |
| random | 0.00 | 0.00 | 0.00 |
| paper (lr 5e-5, 45 epochs) | 10.0 | 0.09 | 0.09 |
| fast (lr 3e-4, 10 epochs) | 6.5 | 0.05 | 0.04 |
| **fast_ent** (fast + entropy 0.01) | **80.8** | **0.85** | **0.85** |

- **fast_ent passes on every seed** (80.7 / 80.8 / 80.9; 0.85-0.86 delivered), and it is the
  only configuration that does. Without the entropy bonus, the policy std collapses to about
  0.2 and the team settles on a weak solution.
- **Mass 50, for the record:** MAPPO paper and fast reach 0.24-0.59 at 24M (scripted 2.8).
  MADDPG at 3M reaches 0.00-0.14. Nobody delivers.
- **Next, by the pre-set rule:** fast_ent is the base configuration. The 10 MAPPO bases (seeds
  0-9, 24M frames) were launched on lab server B at 18:51 (`phase2/mappo_bases.sh`, output in
  `runs_phase2/mappo_bases/`).

### Stage 1 development run 2: action-space residual (decided by the user 2026-11-22)

- **Change (the only one from run 1):** the RC and REC head moves to action space,
  mu = clip(mu_b + g β · 2 · tanh(D), −1, 1), where β is the warm-up ramp (0 → 1) and 2 is the
  bound (`--space action --res-scale 2`, now the default in `phase2/train_fault.py`).
  - Zero-initialised D, the gate, the frozen base and the inputs are unchanged. RC-blind uses
    the same head, and FT is unaffected.
  - Unit check: on a base saturated at +1, the pre-tanh head cannot move the action at all,
    while the action-space head reaches −1. Both equal the base exactly at initialisation.
- **This departs from the Phase 1 parameterisation** and is reported as such. Phase 1 results
  used the pre-tanh head.
- **Runs and evaluation:** as in run 1 (tag and spread, seeds 0-4, 4 kinds, every Stage 0
  cell), output in `runs_phase2/stage1_run2/`.

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
