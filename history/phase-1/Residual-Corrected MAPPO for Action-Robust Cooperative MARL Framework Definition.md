# Residual-Corrected MAPPO for Action-Robust Cooperative MARL: Framework Definition

Nov 11, 2026 · @Ayesha49

## 1. Problem formulation

We model cooperative MARL as a Dec-POMDP with a corruption channel between each agent's intended action and the action the environment executes. The goal is the best worst-case team return under that channel, with a bounded loss on clean episodes.

**Base model.** A Dec-POMDP with N agents, state s, local observations o\_i, continuous bounded actions a\_i in \[-1, 1\]^d, shared reward r(s, ã) and discount γ. Each agent i keeps a history τ\_i of its observations and its own past intended actions.

**Corruption channel.** Agent i samples an intended action a\_i \~ π\_i(· | τ\_i). The environment receives the executed action ã\_i = C\_i(a\_i, ā\_i), where ā\_i comes from an adversary or fault process ν. Transitions and rewards depend on the executed joint action ã, never on a.

| Threat model | Executed action ã\_i | Can agent i's own residual fix it? | Who can compensate |
| --- | --- | --- | --- |
| Noisy action (NR-MDP, Tessler 2019) | (1 − α) a\_i + α ā\_i | Partly, if ā\_i is predictable from history | Agent i, within the step |
| Probabilistic override (PR-MDP) | ā\_i with probability α, else a\_i | No, the override discards a\_i that step | Agent i on later steps; teammates now |
| Persistent fault (bias, gain, drift) | G\_i a\_i + b\_i, with G\_i and b\_i changing slowly | Yes, by pre-compensating once G\_i and b\_i are inferred | Agent i, within the step |
| Teammate corruption (Byzantine) | Any of the above on a subset K of agents, \|K\| ≤ k | Not for the corrupted agent | Healthy teammates only |

**Adversary budget.** ν is limited three ways: at most k corrupted agents, a per-step size ‖ā\_i − a\_i‖ ≤ ε (or override probability α), and a rate bound ‖δ\_t − δ\_{t−1}‖ ≤ η on the perturbation δ for the temporally coupled case. The rate bound is the temporal-coupling constraint ε̄ of GRAD (Liang et al., ICLR 2024, arXiv 2307.12062), applied there to additive state and action perturbations in single-agent RL; ARA-RL (Yang et al., IEEE TAI 2025) uses a similar bounded-rate action adversary. It is what makes a perturbation learnable from history; with η unbounded, only recovery is possible.

**Objective.** Maximize worst-case return while keeping clean return within κ of a policy trained without attacks:

```latex
\max_{\pi}\ \min_{\nu \in \mathcal{V}_{k,\varepsilon,\eta}} J(\pi, \nu) \quad \text{s.t.} \quad J(\pi, \varnothing) \ge J(\pi^{\ast}_{\text{clean}}, \varnothing) - \kappa
```

In practice the constraint is relaxed into a mixed objective that matches the rollout mix in section 4: clean episodes (weight λ), random faults (weight ρ, an average) and the learned adversary (the remaining weight, a worst case). A penalty c on the residual applies in clean episodes:

```latex
\mathcal{L}(\pi) = \lambda\, J(\pi, \varnothing) + \rho\, \mathbb{E}_{f \sim \mathcal{F}}\big[J(\pi, f)\big] + (1-\lambda-\rho) \min_{\nu \in \mathcal{V}_{k,\varepsilon,\eta}} J(\pi, \nu) - c\, \mathbb{E}_{\text{clean}}\big[\lVert \Delta_i \rVert^2\big]
```

This is a Lagrangian-style relaxation of the constrained max-min above: λ acts as the multiplier on the clean-performance constraint, and only the adversarial term is a true worst case. Training approximates the inner min by alternating policy and adversary updates (section 4). That is an approximate saddle-point search, with no convergence guarantee for neural policies.

**What the residual can correct.** Each agent's action mean is decomposed as μ\_i = μ\_b(o\_i) + Δ\_i, with ‖Δ\_i‖ ≤ β. Under NR, the residual that restores the nominal executed action is:

```latex
\Delta_i^{\ast} = \frac{\alpha}{1-\alpha}\left(\mu_b - \hat{\bar a}_i\right)
```

Three limits follow. The residual needs a prediction of ā\_i, so it only helps against perturbations correlated in time. The bound β caps what it can cancel, about β(1 − α)/α in action units. Near the action limits there is no headroom left, so saturation blocks the correction. For a persistent fault, the matching residual is Δ\* = (G⁻¹ − I) μ\_b − G⁻¹ b, which needs G to be invertible. Under PR overrides, no correction within the step exists, and the residual's role is recovery on the next steps.

## 2. Network architecture

Each agent's actor is a base MLP plus a recurrent residual head that only acts on perturbation evidence; the two combine into one Gaussian, so MAPPO trains it unchanged.

&#91;embedded content: per-agent actor · base, residual head, corruption channel, training-only modules\]

Read left to right: observation and evidence feed the base and the residual, their outputs combine and are sampled, and the corruption channel turns the intended action into the executed one. Only the top part runs at deployment.

**Prior art for this architecture.** A frozen base plus a zero-initialised, bounded, recurrent residual driven by command-response evidence is established in single-agent fault adaptation: RFA (arXiv 2609.17404; frozen teacher, GRU residual, commanded-minus-measured joint features, persistent joint faults) and CAPABLE (arXiv 2610.11971; bounded residual on a frozen VLA, conditioned on a capability latent from commanded vs executed actions). ASGARD (arXiv 2609.20982) corrects attacked action commands from state history on a single UAV. The architecture is therefore not claimed as new; what is new is its use in cooperative CTDE MARL, on a PR-robust base, against Tessler's NR operator (`novelty-check.md`, section 3).

**Starting sizes** (to tune):

- **Base actor:** MLP, 2 × 64, tanh activations, linear mean head; one learned log σ per action dimension. Shared across agents with a one-hot agent ID, as in standard MAPPO.
- **Residual head:** a 64-unit GRU over \[o\_i, z\_i, stop-grad(μ\_b)\], then two linear heads: Δ̃ (d outputs, final layer initialized to zero) and gate g (sigmoid, bias initialized so g starts near 0).
- **Bound:** β ≤ 1, warmed up from 0; the residual is added before the tanh squash, so it stays active when the base saturates.
- **Critic:** MLP, 2 × 128, on the global state, plus privileged corruption features during training.
- **Perturbation estimator:** linear head on the GRU state, predicting δ (regression) and which teammates are corrupted (binary per teammate).

## 3. Inputs and outputs: training vs execution

At execution each agent runs only its base actor and residual head on local signals. The critic, the adversary and every privileged label exist only in training.

| Module | Inputs | Output | Training | Execution |
| --- | --- | --- | --- | --- |
| Base actor π\_b (shared weights + agent ID) | o\_i^t, agent ID | Mean μ\_b ∈ R^d; log σ (state-independent, learned) | Yes | Yes |
| Evidence encoder | Own intended a\_i^{t−1}; executed ã\_i^{t−1} if sensed, else the motion gap e\_i^t; teammates' observed motion | Evidence vector z\_i^t | Yes | Yes |
| Residual head (GRU) | o\_i^t, z\_i^t, stop-grad(μ\_b), hidden state h\_i^{t−1} | Raw correction Δ̃\_i, gate g\_i ∈ \[0, 1\], next h\_i^t | Yes | Yes |
| Forward model f (optional) | o\_i^{t−1}, a\_i^{t−1} | Predicted motion; gap e\_i^t = observed − predicted | Trained on clean rollouts, then frozen | Yes |
| Perturbation estimator (auxiliary head) | GRU hidden state h\_i^t | Estimated δ̂\_i (own) and corrupted-teammate probabilities | Yes, supervised by privileged labels | Optional (diagnostic only) |
| Centralized critic V | Global state s^t; optionally the corruption mask and δ (privileged) | Value V(s^t) | Yes | No |
| Adversary ν | s^t, intended joint action a^t, victim set K, remaining budget | Perturbation ā or override | Yes | No (separate attackers at evaluation) |

**Combined action.** The policy stays one Gaussian, so PPO's likelihood ratio is exact:

```latex
\mu_i = \tanh\!\big(u_b(o_i) + g_i \cdot \beta \cdot \tanh(\tilde\Delta_i)\big), \qquad \mu_b = \tanh(u_b), \qquad a_i \sim \mathcal{N}(\mu_i, \sigma^2 I)
```

The residual is added before the squash, to the base's pre-squash output u\_b. The outer tanh keeps the mean inside \[−1, 1\], and the correction stays active when the base saturates. An earlier draft scaled the residual by the headroom (1 − |μ\_b|); that switches it off near ±1, where MuJoCo policies often operate. RFA (Deng et al., arXiv 2609.17404) uses that headroom-scaled form, so it is kept as an ablation (section 5). The effective correction Δ\_i = μ\_i − μ\_b is what the clean-episode penalty and the bounds in section 1 refer to. The sampled action is clipped by the environment, and the log-probability uses the unclipped sample, as in standard MAPPO.

**Decentralised execution.** The residual head is part of each agent's actor. It uses only that agent's observation, its own past actions and its own GRU state, with no communication.

**Evidence at deployment.** The residual is only as good as its evidence, so the observation model must be stated per benchmark:

- **Own executed action sensed** (motor encoders, joint torque): z\_i holds ã\_i^{t−1} − a\_i^{t−1} directly. This is the strongest case.
- **Not sensed:** z\_i holds the motion gap e\_i^t from the frozen forward model. The perturbation must then be inferred from state changes, which mixes in dynamics noise.
- **Teammates:** their executed motion is visible when it appears in o\_i (neighbour velocities in MPE, shared body joints in MaMuJoCo). Their intended actions are never visible at execution.

**Identifiability.** The motion gap mixes three causes: the agent's own fault, a teammate's fault and contact dynamics. In VMAS the split is mild, because actions are forces and velocities are observed. In MaMuJoCo it is hard, because joint motion depends on every agent's torques. The auxiliary estimator therefore predicts own and teammate perturbations as separate outputs, and its accuracy is reported per environment.

**Trusted sensing.** The design assumes actuation can be corrupted while the agent's own sensing stays honest. That holds for faults. For attacks it is an explicit assumption: an attacker who also spoofs feedback is out of scope.

**Privileged signals in training.** The corruption mask, the true δ and the victim identity feed the critic and the auxiliary estimator only. The auxiliary loss forces the GRU state to encode the perturbation, which separates the residual from "just a wider actor".

## 4. Training procedure

Training is standard MAPPO on the combined Gaussian, plus three additions: a residual penalty on clean episodes, an auxiliary perturbation loss, and an adversary trained in alternation.

**Per-agent actor loss.**

```latex
\mathcal{L}_{\text{actor}} = \mathcal{L}^{\text{clip}}_{\text{PPO}} - c_H\, \mathcal{H}(\pi_i) + c_r\, \mathbb{1}[\text{clean}]\, \lVert \Delta_i \rVert^2 + c_a\, \mathcal{L}_{\text{aux}}(\hat\delta_i, \delta_i)
```

The critic uses the usual value loss on the centralized value V(s). Advantages come from GAE on the team reward of the executed joint action.

**Gradient routing (default).** The base actor gets PPO gradients from clean episodes only. The residual head gets gradients from all episodes. This makes the split identifiable: the base approximates the clean optimum, and the residual carries all robustness. A simpler variant lets both heads see all episodes and relies on the penalty alone; it is an ablation, not the default.

**Rollout mix.** Each episode draws one condition:

| Condition | Share of episodes | What is perturbed |
| --- | --- | --- |
| Clean | λ (start 0.5, anneal to 0.3) | Nothing |
| Random faults | (1 − λ) / 2 | Bias, gain or drift on up to k agents, rate-bounded by η, random onset |
| Learned adversary | (1 − λ) / 2 | NR or PR perturbation from ν on up to k victims chosen by ν |

The shares are starting points to tune, not results.

**Adversary.** ν is trained with PPO on the negated team return, as a zero-sum player. It picks victims within budget k and outputs ā under the per-step and rate bounds. Updates alternate: several policy updates per adversary update, the scheme Tessler et al. use for action-robust DDPG. Keeping a small population of past adversaries, as ROMANCE does, reduces overfitting to a single attacker.

**Curriculum.**

1. Train the base actor and critic on clean episodes until return plateaus, with the residual gate fixed at 0.
2. Freeze the base. Train the residual on random faults, warming the residual bound β up from 0 to its target.
3. Unfreeze the base under clean-only routing. Add the learned adversary and the full rollout mix.
4. Evaluate on held-out attackers and fault types never seen in training.

**Loop (per iteration).**

1. Sample conditions for each parallel environment; reset GRU hidden states.
2. Roll out: base mean, evidence, residual, combined Gaussian, sample, corruption channel, environment step. Store intended actions, executed actions, the clean flag and the privileged labels.
3. Compute GAE advantages from V(s) on the team reward.
4. Update the base on clean samples and the residual on all samples (PPO clip loss, residual penalty, auxiliary loss), using recurrent minibatches over chunks of the episode.
5. Update the critic on all samples.
6. Every m iterations, update the adversary on the attacked episodes.

## 5. Practicality

The framework is cheap to deploy: one small GRU per agent, no communication, and no privileged signals at execution. Its real constraints are what each agent can sense, and the lack of continuous-action robust MARL baselines.

**Cost.** A 64-unit GRU plus two small output layers adds on the order of tens of thousands of parameters per shared actor and one recurrent step per agent per tick. Execution stays decentralized; each agent only needs a short buffer of its own past actions.

**Implementation pitfalls.**

- **Intended vs executed action.** The PPO ratio must use the intended action a\_i that the policy sampled; rewards and transitions come from the executed ã\_i. Putting ã\_i in the ratio biases every update.
- **Residual collapse.** A penalty c\_r set too high keeps Δ at zero. Log the residual norm separately for clean and attacked episodes; robustness requires a clear gap.
- **Residual absorbs the task.** Without clean-only routing, the residual can learn task behaviour. Check by disabling it on clean episodes: return should barely move.
- **Over-pessimism.** An unbudgeted adversary makes the team overly cautious, which MARTA reports as well. Keep k, ε and η explicit and sweep them.
- **Policy hijack vs actuator fault.** If an agent's policy is hijacked (Byzantine), its own residual is irrelevant; if its actuator is faulty, its residual still runs and can help. Report the two cases separately.
- **Adaptive attackers.** Evaluate against an adversary trained against the final, frozen policy, residual included, not only the training attackers.

**Benchmarks (continuous actions).** Multi-Agent MuJoCo (for example HalfCheetah 2x3, Ant 4x2, Walker2d 2x3), where agents share one body, so teammate compensation is physically meaningful. Continuous MPE (simple\_spread, simple\_tag) tests coordination without that coupling.

**Baselines.** The Phase 1 comparison is fixed in section 7.3: the Phase 0 references and robust baselines plus three attribution controls, on the MADDPG backbone. The table below is the wider candidate list for the MAPPO extension (7.9).

| Baseline | Why it is included | Continuous actions? |
| --- | --- | --- |
| MAPPO | Clean reference | Yes |
| MAPPO, matched parameters (wider actor + same GRU, no residual split) | Tests whether the decomposition matters, or just capacity | Yes |
| MAPPO + action adversary (no residual) | Isolates the residual from adversarial training | Yes |
| M3DDPG | Minimax robust MARL | Yes |
| MIR3 | Robustness via mutual-information regularization, no adversary | Yes (swarm tasks in the paper) |
| EIR-MAPPO | Belief over Byzantine teammates | Needs porting (discrete in the paper) |
| ROMANCE | Population of action attackers | Needs porting (QMIX, discrete) |
| MARTA | Budgeted fault switching | Needs porting (QMIX or VDN, discrete) |

**Metrics** (fixed for Phase 1 in section 7.5). Clean return, worst-case return against the adaptive attacker, return across sweeps of α, ε and k, recovery time after fault onset, and the residual norm in clean versus attacked episodes.

**Ablations.** Remove one piece at a time: the evidence inputs, the GRU (use an MLP instead), the gate, clean-only routing, the residual penalty, and the auxiliary loss. Swap the pre-squash residual for RFA's headroom-scaled residual (section 3). Then swap the training threat model: train on NR, test on PR, and the reverse.

## 6. Open design decisions

Six choices shape the paper; the first two decide which claim it can make.

- [x] **Headline threat model.** Decided 2026-11-18: NR is claimed (matched evaluation, Tessler-style); PR is reported as transfer with a no-harm test (section 7.2).
- [ ] **Own-action vs teammate corruption.** Teammate compensation is the more novel claim; decide whether the residual gets explicit teammate evidence, such as attention over neighbours' observed motion.
- [ ] **Executed-action sensing.** Fix per benchmark whether ã\_i^{t−1} is observed, or inferred through the forward model.
- [ ] **Gradient routing.** Clean-only base updates (default) vs a penalty alone; the theory section depends on this choice.
- [ ] **Parameter sharing.** One residual shared across agents with an agent ID (default) vs one per agent for heterogeneous teams.
- [ ] **Theory to prove.** (a) Clean-return loss bounded in terms of β and the Lipschitz constant of Q; (b) the NR correctability bound ‖μ\_b − ā‖ ≤ β(1 − α)/α from section 1.

## 7. Evaluation framework

This section fixes what the framework claims, how the claim is tested, and against what. It
builds on the Phase 0 protocol (`phase-0/PHASE-0.md`, sections 2.1 and 6) so that Phase 1
numbers compare directly with Phase 0's.

**Backbone for this evaluation.** Phase 0 used MADDPG on MPE. For direct comparison, the
framework is evaluated first as **RC-MADDPG**: the same residual head and corruption channel on
a deterministic MADDPG actor (adaptation notes in `residual-hypotheses.md`, section 7). The
MAPPO / VMAS / MaMuJoCo version in sections 2-4 is a later extension (7.9).

### 7.1 From the Phase 0 gap to this framework

The Phase 0 gap is about **evaluation**: no published work compares robust cooperative-MARL
methods under one protocol covering both fault types (PR replacement and NR mixing), a strength
sweep, random and learned attackers, one-agent to all-agent scope, and a reference trained
against the corruption. The novelty check of 2026-11-18 (`novelty-check.md`) confirms this gap
and narrows what the method may claim: its single-agent ingredients (frozen-base residual,
command-response evidence, rate-bounded adversary, history-based correction of attacked
commands) are published (RFA, CAPABLE, ASGARD, GRAD, ARA-RL). Phase 1 therefore claims, in
order: (1) the PR/NR benchmark with matched and transfer cells and rank stability; (2) the
boundary of when history-based correction can help (P3-P5) and its test as a dose-response in
predictability; (3) own-action correction in cooperative CTDE MARL on a PR-robust base, under
NR. Phase 1 has two separate deliverables:

1. **The benchmark** closes the evaluation gap. It is method-agnostic: every method is evaluated
   on every cell of the grid in 7.4.
2. **The residual framework** is one method evaluated on it. It claims robustness only under
   its declared threat model (7.2). It is allowed to solve the gap partially.

| Element of the Phase 0 gap | How the benchmark covers it | What the residual framework claims |
| --- | --- | --- |
| Both fault types (PR and NR) | full PR x NR grid for every method | **NR only** (matched claim); PR reported as transfer with a no-harm test |
| Strength sweep | alpha in {0.1, 0.3, 0.5}, plus takeover for PR | NR gain at every alpha where the action is restorable (P4) |
| Random to adversarial attackers | random i.i.d., random temporally correlated (new), learned adaptive | gain grows with the predictability of the perturbation (P3, P5) |
| One-agent to all-agent scope | all agents, worst single victim, learned agent 0 | own-actuator correction at any scope; teammate compensation not claimed |
| Damage spreads through the team | team return under one-agent attacks | partial: each agent repairs its own executed action, which limits the spread at its source |
| Agents cannot see which action was corrupted | the corruption is never signalled to the agent | partial: an agent infers its **own** perturbation from its motion (exact in MPE except collisions); a teammate's is not inferred |
| Reference trained against the corruption | PR-MADDPG and NR-MADDPG (Tessler-style) | compared against both |

### 7.2 Threat models and the claim

**Channel** (as in Phase 0, section 2.1). Victim set $V$; replacement action
$r_i \in [-1,1]^2$; budget $\alpha$:

- **NR:** $\tilde a_i = \mathrm{clip}\big((1-\alpha)\,a_i + \alpha\, r_i\big)$ at every step.
- **PR:** $\tilde a_i = r_i$ with probability $\alpha$, else $a_i$; $\alpha = 1$ is takeover.

**Sources of $r$.**

| Source | Definition | Predictable from history? |
| --- | --- | --- |
| R, random (Phase 0) | $r_t \sim \mathcal U([-1,1]^2)$, i.i.d. | only its mean (zero) |
| $\mathrm{R}_\phi$, correlated random (new) | $r_t = \mathrm{clip}\big(\phi\, r_{t-1} + \sqrt{1-\phi^2}\,u_t\big)$, $u_t \sim \mathcal U([-1,1]^2)$, $\phi \in \{0.9, 0.99\}$; $\phi = 0$ is R | increasingly with $\phi$; $\phi \to 1$ is a persistent bias |
| L, learned (Phase 0) | DDPG adversary $r = \nu(s)$, trained against the frozen final policy, residual included | partly, through the slow change of $s$ |

$\mathrm{R}_\phi$ is added because the framework's mechanism depends on predictability. It turns
"does the residual help?" into a dose-response in $\phi$, which Phase 0's sources cannot show.
It is the random-fault counterpart of GRAD's coupling sweep: GRAD varies the adversary's
coupling bound $\bar\epsilon$; here the per-step change of $r$ shrinks as $\phi \to 1$, with
$\mathbb E\lVert r_t - r_{t-1}\rVert$ reported for each $\phi$ so the two axes can be compared.

**Claim (pre-registered).** Under NR, at every tested alpha, source and scope, RC-MADDPG
retains more competence than its base (PR-MADDPG) and than NR-MADDPG, with the gain
increasing in $\phi$. Under PR it is no worse than its base beyond a margin $\delta$.

**Matched vs transfer cells.** Following Tessler et al., a method's **matched** cells are those
whose operator equals its training operator; all others are **transfer** cells. Headlines use
matched cells; every cell is still reported, because the benchmark's purpose is to show
whether rankings survive a change of operator (Phase 0 Q3).

| Method | Training operator | Matched cells | Transfer cells |
| --- | --- | --- | --- |
| RC-MADDPG | NR (residual stage) on a PR-trained base | NR | PR |
| PR-MADDPG | PR | PR | NR |
| NR-MADDPG | NR | NR | PR |
| Controls (7.3) | NR (same stage as RC) | NR | PR |
| MADDPG, M3DDPG, ERNIE-A-c, MIR3 | none (no action operator) | none: all cells are reported as untargeted | — |

### 7.3 Methods

**Proposed: RC-MADDPG.** Base = a frozen PR-MADDPG actor (the Phase 0 checkpoint of the same
seed, so every comparison with PR-MADDPG is paired by seed). Residual = GRU head and gate
(section 2), trained under NR with alpha drawn from {0.1, 0.3, 0.5} per episode and $r$ drawn
from R, $\mathrm{R}_\phi$ and a pool of learned adversaries. Freezing the base makes gradient
routing trivial: the base receives no gradient, and any change from PR-MADDPG is the residual's.
Variant: the same residual on a frozen NR-MADDPG base.

**References (Phase 0 checkpoints, no retraining).** MADDPG (the normalisation reference),
PR-MADDPG (Phase 0's best method; the base) and NR-MADDPG (the operator-matched reference for
an NR claim).

**Phase 0 robust baselines (Phase 0 checkpoints, no retraining).**

| Baseline | Setting | Note |
| --- | --- | --- |
| M3DDPG | eps 0.1 (0b) | did not reproduce its own paper (Phase 0, 4.4-4.5) |
| ERNIE-A-c | lambda 1, eps 0.5 on `spread` (strongest working dose, Phase 0 4.3); lambda 1, eps 0.05 on `tag` (only setting trained there) | our continuous adaptation |
| MIR3 | lambda 0.0005 (selected) | did not reproduce its own paper (Phase 0, 4.6) |

**Attribution controls (new runs).** All start from the same frozen PR-MADDPG base seed, train
under the same NR mixture with the same steps and the same tuning budget as RC-MADDPG, and
differ in one thing:

| Control | Memory | Base/residual split | Isolates |
| --- | --- | --- | --- |
| FT: PR-MADDPG fine-tuned under the NR mixture (memoryless MLP, all weights trained) | no | no | extra NR training alone |
| REC: **GRAD-style recurrent defender**: recurrent actor (same GRU and evidence inputs) initialised from the base, all weights trained adversarially under the same NR mixture, including a rate-bounded learned adversary as in GRAD | yes | no | memory without the decomposition; also the strongest single-agent method (GRAD) ported to this setting |
| RES-MLP: residual head as in RC but an MLP on the current observation only, no evidence inputs | no | yes | the decomposition without memory |

Attribution, per cell, as paired differences: RC − FT = what the residual architecture adds
beyond NR training; REC − FT = what memory adds; RC − REC = what the split adds;
RC − RES-MLP = what history adds inside the residual.

RFA (single agent) already found the split ahead of a monolithic recurrent policy by about 2.6
points of success rate, and CAPABLE found a structured residual ahead of a parameter-matched
history baseline. RC − REC is therefore expected by reviewers rather than a contribution; a
null result here would be reported as such.

### 7.4 Evaluation grid

Phase 0's core grid (21 cells per task) is kept unchanged, so every Phase 0 number stays valid,
and one source is added.

| Operator | Source | alpha | Scope | Role for RC-MADDPG |
| --- | --- | --- | --- | --- |
| NR | R | 0.1, 0.3, 0.5 | all, worst victim | matched |
| NR | $\mathrm{R}_\phi$, $\phi$ 0.9, 0.99 (new) | 0.1, 0.3, 0.5 | all, worst victim | matched; mechanism test |
| NR | L | 0.1, 0.3 | all, agent 0 | matched; **headline cell: L-NR all, alpha 0.3** |
| PR | R | 0.1, 0.3, 0.5 | all, worst victim | transfer |
| PR | $\mathrm{R}_\phi$ (new) | 0.1, 0.3, 0.5 | all | transfer |
| PR | L | 0.1, 0.3 | all, agent 0 | transfer |
| PR | L, takeover | 1 | agent 0 | transfer |
| none | clean | 0 | — | clean cost |

$\mathrm{R}_\phi$ adds 18 cells per task (12 NR, 6 PR); evaluation only, for every method.
Learned adversaries follow Phase 0 exactly (300k steps, 2 adversary seeds, trained against
each frozen final team through the operator and alpha they are tested at), so adversaries
already trained in Phase 0 are reused for Phase 0 methods.

**Environments.** `simple_spread` (primary) and `simple_tag` (secondary; Phase 0 found most
`tag` effects within seed noise), with the Phase 0 wrapper (`mpe2` 1.1.1, 2-D force in
[-1, 1]^2). **Evidence:** the agent's previous executed action, $\tilde a_i^{t-1}$, which in MPE
is recoverable exactly from two consecutive own-velocity observations except on collision
steps. It is given to RC-MADDPG and REC; RES-MLP gets none, which tests what the evidence
adds.

### 7.5 Metrics

- **Clean return** $J_0$, as in Phase 0.
- **Competence retained** $\rho(\alpha) = (J(\alpha) - J_{\text{rand}})/(J_0 - J_{\text{rand}})$, as in
  Phase 0 ($J_{\text{rand}}$ = $-26.08$ on `spread`, $1.05$ on `tag`), with each method's own $J_0$.
  Raw attacked return $J(\alpha)$ is reported alongside, because a lower clean return can
  inflate $\rho$ (Phase 0, 4.3, MIR3 lambda 0.1).
- **Learned cells:** mean over the 2 adversary seeds (Phase 0 rule) and the worse of the two.
- **Residual diagnostics:** $\lVert\Delta\rVert$ and gate $g$ on clean vs attacked steps; error of
  the perturbation estimate $\hat\delta$; share of steps with $|a^\ast| \le 1-\alpha$ (restorable
  under R-NR) and $\le 1-2\alpha$ (worst case, P4).
- **Adversary rate:** mean $\lVert r_t - r_{t-1} \rVert$ of each learned adversary, which places
  the L source on the predictability axis of $\mathrm{R}_\phi$.

### 7.6 Statistics and decision rules

Phase 0's statistics are kept: per-seed paired differences, 95% bootstrap CI over 5 seeds;
**better** / **unclear** / **worse** as in Phase 0 (2.1); Kendall $\tau$ between method rankings
for each pair of cells. Pairing is exact for RC-MADDPG, its controls and PR-MADDPG, which share
base seeds.

Pre-registered tests, all against PR-MADDPG unless stated:

| ID | Test | Passes if |
| --- | --- | --- |
| T1 headline | L-NR all, alpha 0.3 | RC better than PR-MADDPG **and** NR-MADDPG |
| T2 NR coverage | all NR cells (R, $\mathrm{R}_\phi$, L) | RC better in a majority, worse in none |
| T3 mechanism | RC − FT under NR-$\mathrm{R}_\phi$ at alpha 0.3 | increases from $\phi = 0$ to 0.9 to 0.99 (positive slope, CI over seeds) |
| T4 PR no-harm | every PR cell, incl. takeover | lower CI bound of RC − PR-MADDPG > $-\delta$, $\delta = 0.03$ competence |
| T5 clean no-harm | clean return | lower CI bound of RC − PR-MADDPG > $-0.5$ return (about 5% on `spread`) |
| T6 attribution | RC − FT, REC − FT, RC − REC on NR cells | reported, no pass rule; RC − FT > 0 is required to credit the architecture |

$\delta$ and the clean margin are proposals, to be fixed before any Phase 1 run.

**Outcome labels.**
- **Full:** T1-T5 pass.
- **Partial:** T2-T5 pass and T1 fails, i.e. the residual corrects predictable NR (R, high
  $\phi$) but not an adaptive learned attacker. This is an acceptable result for the paper.
- **Mechanism only:** T3 passes but T2 fails: the gain exists only for near-persistent faults.
- **No:** T3 fails, or T4/T5 fail (the residual harms).

### 7.7 Hypotheses per cell (from `residual-hypotheses.md`)

| Cell family | Room above PR-MADDPG on `spread` | Expected |
| --- | --- | --- |
| NR-$\mathrm{R}_\phi$, $\phi$ 0.99 | unknown (new) | largest gain; near-persistent fault |
| NR-L all | 0.24 at alpha 0.3 | gain, limited by the adaptive attacker's rate |
| NR-R | 0.03-0.08 | small, growing with alpha (gain-fault correction) |
| PR, all sources and takeover | 0.04-0.58 | tie (P5); passes T4 |
| Clean | — | tie; passes T5 |

### 7.8 Budget

| Item | Count |
| --- | --- |
| Reused Phase 0 teams (6 methods x 5 seeds x 2 tasks; ERNIE strong dose `spread` only) | 60 |
| New residual-stage runs: RC, RC on NR base, FT, REC, RES-MLP; 5 seeds x 2 tasks | 50 |
| New learned adversaries: 50 new teams x 9 learned cells x 2 adversary seeds | 900 |
| Ablations (section 5, incl. RFA's headroom-scaled residual): 7 variants x 3 seeds, `spread` only; NR cells only | 21 runs |
| Random-source evaluation: every team, Phase 0 cells plus 18 $\mathrm{R}_\phi$ cells, 100 episodes each | evaluation only |

**Staged route to the full grid.** The grid above is reached in stages, each with its own go
rule: Stage 0 (checks on Phase 0 checkpoints), Stage 1 (MADDPG + residual), Stage 2
(PR-MADDPG + residual). Stages, go rules and time frame: `PHASE-1.md`.

### 7.9 Later extension: MAPPO, VMAS and MaMuJoCo

After the MADDPG evaluation, the same grid, claim and decision rules carry over to the MAPPO
version of sections 2-4 on VMAS (Navigation, Transport, Balance) and Multi-Agent MuJoCo
(HalfCheetah 2x3, Ant 4x2, Walker2d 2x3), where actions saturate more often and teammates
share one body. The baselines must then be rebuilt on MAPPO (AR-MAPPO for PR-/NR-MADDPG; MIR3
has a released MAPPO backbone). EIR-MAPPO and ROMANCE (discrete in their papers) are not part
of the Phase 1 comparison.

## 8. Theoretical analysis: is the framework action-robust?

Yes, conditionally. The framework is never much worse than its base under any attack, and it is provably better against perturbations that are hidden, persistent and within the actuator's authority. Against i.i.d. noise, random overrides and a state-feedback adversary it adds nothing beyond adversarial training alone.

The statements below are proof sketches under stated assumptions, not finished proofs.

**What is claimed as new.** P1 and P2 are standard consequences (tanh Lipschitz bound,
performance difference lemma); comparable bounds exist for constrained residual RL and for
disturbance-observer add-ons to RL policies. The theory contribution is P3-P5: which parts of an
action perturbation history can correct, and which no policy can. No paper found in the novelty
check (`novelty-check.md`) states or tests this boundary for action corruption.

| Perturbation | Does the residual add robustness? | Result |
| --- | --- | --- |
| Any attack, any size | No harm beyond a β-sized loss | P2 |
| Persistent or rate-bounded, own actuator | Yes: loss scales with rate η, not size ε | P3 |
| NR mixing with predictable ā | Yes, for nominal actions with \|a★\| ≤ 1 − 2α | P4 |
| i.i.d. noise, PR override, state-feedback adversary | No | P5 |
| Teammate fault, coupled task | Yes, if the fault lies in the healthy agents' control range | P6 |
| Hijacked (Byzantine) teammate | No guarantee | P6 |

**Setup and assumptions.**

- **A1 (Lipschitz value).** For a fixed perturbation process, Q(h, a) changes by at most L\_Q ‖a − a'‖ when the joint executed action changes. Histories h play the role of states.
- **A2 (channel).** The corruption channel does not expand differences in the intended action. Additive and NR channels satisfy this.
- **Notation.** N agents, d action dimensions each, k corrupted agents, effective correction Δ\_i = μ\_i − μ\_b,i.

### 8.1 Guarantees that always hold

**P1. The residual moves the action mean by at most β.** Because tanh is 1-Lipschitz, each component of the mean shifts by at most gβ:

```latex
\lvert \tanh(u_b + g\beta\tanh\tilde\Delta) - \tanh(u_b) \rvert \le g\beta \quad\Rightarrow\quad \lVert \Delta \rVert_2 \le \beta\sqrt{Nd}
```

**P2. Bounded cost, clean and attacked.** Couple the base and combined policies with the same Gaussian noise, so their actions differ by exactly Δ. The performance difference lemma and A1 then give, for any fixed perturbation process ν (including none):

```latex
\lvert J(\pi,\nu) - J(\pi_b,\nu) \rvert \;\le\; \frac{L_Q}{1-\gamma}\, \mathbb{E}_{d^{\pi}}\big[\lVert\Delta\rVert_2\big] \;\le\; \frac{L_Q}{1-\gamma}\sqrt{\mathbb{E}\lVert\Delta\rVert_2^2}
```

Two consequences follow.

- **Clean cost.** On clean episodes the right-hand side is the square root of the quantity the residual penalty minimizes. Driving that penalty to ρ bounds the clean loss by L\_Q √ρ / (1 − γ). This is the κ in the constraint of section 1.
- **Bounded downside.** Under any attack, including one designed against the residual, the framework loses at most L\_Q β √(Nd) / (1 − γ) relative to its base. An adversary cannot turn the residual into an unbounded liability.

The second point sets a design trade-off: β must be large enough to cancel the target perturbation (8.2), and every unit of β raises the worst-case downside.

### 8.2 Where the residual adds robustness

**P3. Rate-bounded perturbation on an agent's own actuator: the loss scales with the rate η, not the size ε.** Take the additive channel ã = a + δ, with ‖δ‖ ≤ ε and a per-step change in δ of at most η. If the agent senses its last executed action, it knows the previous δ exactly. Write a★ for the nominal action. Commanding the nominal action minus the previous δ gives:

```latex
\tilde a_t = a^{\ast}_t + (\delta_t - \delta_{t-1}), \qquad \lVert \tilde a_t - a^{\ast}_t \rVert \le \eta
```

By A1, the return loss relative to the clean nominal policy is then bounded as follows, where ε\_est is the estimation error when δ is inferred from motion and not sensed:

```latex
J(\pi^{\ast},\varnothing) - J(\pi,\nu) \;\le\; \frac{L_Q\sqrt{k}}{1-\gamma}\,(\eta + \varepsilon_{\text{est}})
```

The same bound for a policy that does not compensate has ε in place of η. A persistent fault has η = 0, so the residual removes it entirely after one step. Two conditions apply: the correction must fit inside the actuator range (|a★| + ε ≤ 1 per dimension), and β must be large enough to shift the action by ε, which takes a larger pre-squash β near saturation.

**P4. NR mixing: only nominal actions of magnitude at most 1 − 2α can be fully restored.** Under ã = (1 − α)a + αā, restoring a★ needs a = (a★ − αā)/(1 − α), which must lie in \[−1, 1\]. In the worst case ā = −sign(a★), so:

```latex
\lvert a^{\ast} \rvert + \alpha \le 1-\alpha \quad\Longleftrightarrow\quad \lvert a^{\ast} \rvert \le 1 - 2\alpha
```

So α must be below 0.5, and at α = 0.3 only nominal actions within ±0.4 are fully restorable. The agent must also be able to predict ā, which P3's rate bound provides.

**Why history helps here (value of information).** In P3 and P4 the perturbation has a hidden parameter θ (which agent, what bias, what gain). A policy that cannot adapt must pick one behaviour for all θ. An adaptive policy infers θ and responds to it. The gap is:

```latex
G \;=\; \mathbb{E}_{\theta}\Big[\max_{\pi} J(\pi,\theta)\Big] \;-\; \max_{\pi\ \text{non-adaptive}} \mathbb{E}_{\theta}\big[J(\pi,\theta)\big] \;\ge\; 0
```

The framework recovers G minus the return lost during identification, which lasts one or two steps when the executed action is sensed.

### 8.3 Where it cannot help

**P5. No policy can cancel the unpredictable part of a perturbation.** Split δ\_t into its prediction from history and a remainder ξ\_t. The command a\_t depends only on history, so the executed error always contains ξ\_t:

```latex
\delta_t = \mathbb{E}[\delta_t \mid h_t] + \xi_t, \qquad \mathbb{E}\lVert \tilde a_t - a^{\ast}_t \rVert^2 \;\ge\; \mathbb{E}\lVert \xi_t \rVert^2
```

Three cases follow.

- **i.i.d. noise:** the predictable part is the mean, so the gain is zero.
- **PR override:** on an override step the executed action does not depend on the command, so nothing can be corrected within the step.
- **State-feedback adversary (the NR-MDP equilibrium of Tessler et al.):** the perturbation is a function of the current state. A policy trained against it already accounts for it, so the residual adds nothing beyond AR-MAPPO.

In all three, P2 still holds: the framework is no worse than its base by more than the β bound.

### 8.4 Teammate compensation

**P6. Healthy agents can cancel a teammate's fault only if it lies in their control range.** Linearize the dynamics around the current state, with B\_i the effect of agent i's action. A fault δ\_j on agent j adds B\_j δ\_j. The healthy set H cancels it with corrections Δ\_H when:

```latex
B_H\,\Delta_H = -B_j\,\delta_j \ \text{ is solvable}, \qquad \text{otherwise the leftover is } \big\lVert (I - B_H B_H^{+})\,B_j\,\delta_j \big\rVert
```

- **VMAS Transport:** all agents push the same package, so every B\_i is the same. Each healthy agent adds −δ\_j/(N − 1). Compensation gets easier with more agents, and averaging the agents' separate estimates reduces estimation noise.
- **MaMuJoCo:** agents drive different joints, so B\_j is generally outside the range of B\_H. Only partial compensation is possible, and no exact guarantee holds.
- **Hijacked teammate:** the deviation is as large as the action range and chosen adversarially. A β-bounded correction cannot cover it, so nothing here applies.

Without communication, each healthy agent must infer δ\_j from its own observations. That makes teammate observability (agent\_obsk in MaMuJoCo) a precondition, not a detail.

### 8.5 Modifications that follow

- **State the claim as P2 + P3 + P6:** bounded downside always, η-scaled loss for persistent own-actuator faults, and teammate compensation under the matched condition. Do not claim gains for i.i.d. NR noise, PR overrides or hijacked teammates (for Phase 1, section 7.2 claims NR only).
- **Set β from the threat budget.** Choose the smallest β that produces an action shift of ε, and report the P2 downside it implies.
- **Soften clean-only routing.** With the base fixed at the clean optimum, the robust optimum must lie within β of it. Let the base take attacked-episode gradients with a small weight w, and report w = 0 as the ablation.
- **Sense the executed action where the platform allows.** It removes ε\_est from P3 and makes identification take one step.
- **Report the restorable set.** For each NR experiment, report the share of nominal actions with |a★| ≤ 1 − 2α; it predicts where the residual can and cannot work.
- **Still to prove rigorously:** A1 for the benchmarks used (or measure L\_Q empirically), and P3 with clipping at the action limits.
