# Correcting Persistent Action Corruption in Cooperative Multi-Agent Teams from Executed-Action Evidence

**Draft, 2026-11-22.** The results in Section 5 come from development runs (seeds 0–4).
Sections marked **[PENDING]** will be filled in from the pre-registered confirmation (held-out
seeds 5–9, `phase-2/PREREG-STAGE2.md`) and the Transport experiments, once those runs finish.

---

## Abstract

Trained multi-agent teams assume that every action they choose is carried out. In real systems,
actuators can be misaligned, mis-wired or degraded, so the executed action differs from the
chosen one, and this persists for the rest of the episode. We study how a team can correct such
*persistent* corruption without being retrained. We add a small recurrent corrector on top of
each agent's frozen policy. The corrector compares the action the agent chose with the action
that was actually executed, infers the corruption, and adjusts the command so that the executed
action matches what the original policy intended. It is trained with a centralised critic and a
privileged teacher that is used only during training. At execution time, each agent needs only
its own observations and a sense of its own executed action.

On two cooperative tasks, MPE cooperative navigation and predator-prey, a 90° actuator
rotation on every agent cuts the frozen teams to 0.22 and 0.09 of their clean competence. With
the corrector, they recover to 0.97 and 0.82, close to an oracle that knows the fault one step
after it starts. Without the
executed-action input, much of the gain disappears. We also explain where correction cannot
help: when corruption is unpredictable, or when it removes the actuator range needed to
compensate.

---

## 1. Introduction

Cooperative multi-agent reinforcement learning (MARL) produces teams of policies that
coordinate well in simulation. Deployment breaks one silent assumption: that the action an agent
chooses is the action its body executes. An actuator can be mounted at an angle, wired with an
axis flipped, or lose strength over time. These faults do not come and go. Once they happen,
they stay.

Most robust MARL methods train a single policy to tolerate a range of disturbances. That
policy is memoryless with respect to the fault: it reacts to the current state the same way
whether or not its actuator is broken. Against a persistent fault this wastes information,
because each step reveals the gap between the chosen and executed action. An agent that
watches this gap can work out *how* its actuator is wrong and invert the fault.

We make three points.

1. **When correction is possible.** Correcting from past evidence can only help when the
   corruption is *persistent* (predictable from the past) and *restorable* (the action needed to
   cancel it still fits inside the actuator's range). Rotation and sign-flip faults meet both
   conditions. Unpredictable noise fails the first. Faults that remove actuator authority fail
   the second, for example strength loss on an agent already pushing at full force.
2. **A corrector that does it.** A recurrent residual on each agent's frozen policy, driven by
   executed-action evidence and trained with a centralised critic and a privileged
   intent-tracking teacher, restores teams under rotation faults. It does so without clean-task
   cost and without harming the faults it cannot fix.
3. **The residual's form matters.** A correction added *before* the policy's output squashing
   cannot move a policy whose actions are saturated. It must be added in action space.

---

## 2. Related Work

**Robust MARL against action perturbations.** M3DDPG and ROMAX train critics against
worst-case teammate actions. ERNIE regularises the value against small action pushes. MARTA
trains against learned action overrides of one agent. Other work trains with random action
noise or faults (curricula over uncertainty, DSDF). These methods produce one memoryless policy.
They do not infer or undo a specific fault. Tessler et al. (2019) define the probabilistic (PR)
and noisy (NR) action-robust operators that we use as baselines.

**Faulty teammates.** RMADDPG, EIR-MAPPO and AACFT let healthy agents detect, down-weight or
work around faulty teammates. Relational Q-Functionals relearn after a malfunction. These
methods handle the *teammate's* fault. Our agents repair their *own* actuation.

**Single-agent correction from command–response evidence.** Several recent single-agent works
add a bounded, zero-initialised residual to a frozen policy, driven by the difference between
commanded and executed actions:
- RFA uses a recurrent residual under joint faults;
- CAPABLE uses a residual on a vision-language-action model;
- ASGARD corrects attacked commands on a UAV.

Disturbance-observer and L1-adaptive add-ons do the same with model-based controllers. We
build on this architecture and do not claim it as new. What we add is the cooperative
multi-agent setting with a centralised critic, the analysis of which corruptions can be
corrected at all, and the finding that the residual must act in action space on saturating
policies.

**Temporally coupled perturbations.** GRAD and ARA-RL study action perturbations that change
slowly over time, with recurrent defenders. They are single-agent and treat the perturbation as
adversarial noise, not as a persistent fault to be identified and inverted.

---

## 3. Method

### 3.1 Setting

Each agent $i$ chooses an action $a_i \in [-1,1]^2$, and the environment executes
$\tilde a_i = \mathrm{clip}(C(a_i))$, where $C$ is an unknown corruption that begins at some
step and then persists. We consider actuator faults that are affine per step,
$C(a) = M a + b$. This covers:
- rotation (M a rotation matrix);
- gain loss (M = gI);
- bias and slow drift (b constant or growing);
- a stuck actuator (M = 0).

Each agent can sense its own executed action, for example from encoders or odometry. In
simple physics this is recoverable from two consecutive velocity readings.

### 3.2 Corrector

Each agent keeps its pretrained policy $\mu_b$ (the *base*) frozen. A small recurrent
corrector reads the agent's observation, its previous chosen and executed actions, and the
base's proposal. It outputs a bounded correction and a gate:

$$
a_t = \mathrm{clip}\big(\mu_b(o_t) + 2\, g_t\, \delta_t\big), \qquad
\delta_t, g_t = f_\theta\big(o_t, a_{t-1}, \tilde a_{t-1}, \mu_b(o_t), h_{t-1}\big).
$$

Here $f_\theta$ is a GRU with a $\tanh$ residual head and a sigmoid gate. The residual head is
zero-initialised, so the corrector starts as an exact copy of the base. The correction is added
in action space with a bound of 2, which makes every action reachable however saturated the
base is. Agents do not communicate. Architecture details are in `ARCHITECTURE.md`.

### 3.3 Training

The correctors of all agents are trained jointly, with a centralised critic $Q(s, \tilde a, c)$
that sees the executed joint action and a privileged description $c$ of the current fault (its
$M$ and $b$). Each episode is clean (25%) or has one fault drawn from a broad mix:
- rotation (half of faulty episodes);
- gain, bias or drift (the rest);
- a random strength, a random onset, and one or all agents as victims.

The actor objective has three terms:
- **Value:** maximise $Q$, with the agent's own executed action recomputed through the
  differentiable fault channel.
- **Clean penalty:** keep the action equal to the base's on clean episodes.
- **Intent tracking:** make the executed action match the base's intended action,
  $\lVert \mathrm{clip}(M a + b) - \mu_b \rVert^2$.

The tracking term uses the same privileged fault parameters as the critic, so it acts as a
training-time teacher. Execution never sees $M$ or $b$; the corrector must infer the fault from
$(a_{t-1}, \tilde a_{t-1})$. Without this term, the critic's gradient alone did not tell the
corrector *which way* to correct on the sparse-reward task (Section 5.4).

### 3.4 What can be corrected

Two conditions bound any corrector that acts on past evidence:
- **Persistence.** A fault can be inverted only after it has been observed. The best possible
  causal corrector lags by one step. Corruption that is fresh each step, such as i.i.d. noise,
  cannot be cancelled by anyone.
- **Restorability.** The command that cancels the fault must lie inside $[-1,1]^2$.
  - Rotations by 90° or 180° map the action box onto itself, so they are always restorable.
  - Strength loss is not restorable on an agent that already pushes at full force, and
    trained pursuit policies do this most of the time.

These two conditions explain why the gains concentrate on rotation faults (Section 5).

---

## 4. Experimental Setup

- **Tasks:**
  - MPE cooperative navigation (*spread*): 3 agents, 25 steps.
  - MPE predator-prey (*tag*): 3 predators against a fixed prey policy, 25 steps.
  - **[PENDING]** VMAS Transport: 4 agents, package mass 10, 300 steps, MAPPO base.
- **Bases:** MADDPG, trained once and frozen. Development uses seeds 0–4. Confirmation uses
  held-out seeds 5–9 under a pre-registered protocol.
- **Faults at test time:**
  - rotation θ ∈ {30, 60, 90, 180}°;
  - gain {0.7, 0.5, 0.3};
  - bias ‖b‖ ∈ {0.1, …, 0.9};
  - drift;
  - a stuck actuator;
  - a worst-direction bias grid.

  Each is applied to one agent or all agents, from a random onset in the first half of the
  episode. Every method faces identical fault draws.
- **Methods:**
  - MADDPG (no correction);
  - PR- and NR-MADDPG (robust training);
  - RC (ours);
  - RC-blind (no executed-action input);
  - REC (the same network, base trained too);
  - FT (the base fine-tuned, no memory).

  All trained variants get the same objective.
- **Metric:** competence retained, ρ = (J − J_random) / (J_clean − J_random), on the base's
  scale (1 = clean base, 0 = random team). Differences are paired per seed, with 95% bootstrap
  confidence intervals. One-agent cells use the base's worst victim.

---

## 5. Experimental Results

### 5.1 Rotation faults break trained teams, and robust training does not help

Competence of the frozen bases under rotation (all agents faulted):

| θ | spread MADDPG | tag MADDPG | tag PR-MADDPG | tag NR-MADDPG |
| --- | --- | --- | --- | --- |
| 30° | 0.97 | 0.45 | 0.48 | 0.51 |
| 90° | 0.22 | 0.09 | 0.10 | 0.10 |
| 180° | −2.04 | 0.05 | 0.05 | 0.05 |

Training against action noise (PR, NR) gives no protection. An oracle that knows the fault and
inverts it restores almost everything: +0.78 (spread) and +0.91 (tag) at 90°. The room for
correction is real.

### 5.2 RC restores the team (development, seeds 0–4)

| Cell | spread MADDPG → RC | tag MADDPG → RC | RC − MADDPG (spread / tag) |
| --- | --- | --- | --- |
| rot 60°, all | 0.83 → 0.97 | 0.16 → 0.61 | +0.14 / +0.45 |
| rot 90°, all | 0.22 → 0.97 | 0.09 → 0.82 | +0.75 / +0.73 |
| rot 180°, all | −2.04 → 0.92 | 0.05 → 0.66 | +2.96 / +0.61 |
| rot 90°, one agent | 0.75 → 0.99 | 0.29 → 0.89 | +0.24 / +0.60 |
| clean | 1.00 → 1.00 | 1.00 → 0.99 | +0.00 / −0.01 |

- All rotation gains have confidence intervals well above zero.
- On tag at 90°, RC (0.82) approaches the one-step-delayed oracle (0.88), which is the best a
  causal corrector can do.
- On faults RC cannot fix (gain, large bias, stuck), it costs at most 0.02. Clean performance
  is unchanged.

**[PENDING]** Pre-registered confirmation on held-out seeds 5–9 (tests F1–F7).

### 5.3 The gain comes from executed-action evidence

| Comparison, rot 90° all agents | spread | tag |
| --- | --- | --- |
| RC − RC-blind (no executed-action input) | +0.12 | +0.38 |
| RC − FT (fine-tuned, no memory) | +0.62 | +0.73 |
| RC − REC (base also trained) | −0.00 | +0.00 |

- Removing the executed-action input costs much of the gain, and a memoryless fine-tune
  recovers almost nothing even with the same teacher.
- Step by step after the fault starts, RC's correction matches the oracle's direction from the
  first step on: cosine similarity 0.99 on tag and 0.89–0.95 on spread. Its size is 86–99% of
  the oracle's. RC makes no corrections before the onset.
- Freezing the base does not itself add accuracy: REC matches RC. The benefit lies in the
  evidence-driven recurrent correction, not in keeping the base frozen.

### 5.4 The residual must act in action space

Our first version added the correction before the base's output squashing ($\tanh$), as is
common. Trained tag predators push at nearly full force: the median pre-squash output is 5.9,
deep in saturation. A pre-squash correction bounded by 1 can reach only 1% of the oracle's
corrections at 90° on tag, and 48% on spread.

| Residual form, rot 90° all agents | spread RC | tag RC |
| --- | --- | --- |
| pre-squash, bound 1 | 0.45 | 0.09 |
| action space, bound 2 | 0.80 | 0.10 |
| action space + intent tracking (final) | 0.97 | 0.82 |

Moving the residual to action space unlocks spread. On tag the corrector then *detects* the
fault but corrects in the wrong direction (cosine 0.15–0.23), because the sparse catch reward
gives the critic a weak action gradient. The intent-tracking teacher supplies the direction.

### 5.5 Cooperative transport **[PENDING]**

The MAPPO bases on VMAS Transport deliver the package in 81–88% of clean episodes, against 43%
for a scripted pusher. The fault-headroom check and the corrector results are pending.

---

## 6. Conclusion

A persistent actuator fault leaves a trace: every step, the executed action differs from the
chosen one in a consistent way. A small recurrent corrector on each agent's frozen policy can
read this trace, identify the fault within a step, and invert it. On two cooperative tasks this
restores teams that rotation faults had broken, at no cost when nothing is wrong. Training
against action noise does not achieve this.

The approach has clear limits:
- It cannot cancel unpredictable corruption.
- It cannot restore authority a fault has taken away.
- In our experiments the frozen base itself adds nothing over training the same network
  end to end.
- On sparse-reward tasks it needs a privileged training-time teacher.

Next steps are confirming the results on held-out seeds, extending them to cooperative
transport, and studying how teammates can compensate when an agent's own fault is beyond
repair.
