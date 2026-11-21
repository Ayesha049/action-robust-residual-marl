# Residual-corrected policy vs Phase 0: desk analysis and hypotheses

Status 2026-11-18. Desk analysis only; nothing here has been run.

Question: can the residual framework
(`Residual-Corrected MAPPO for Action-Robust Cooperative MARL Framework Definition.md`, "the
framework doc") outperform the Phase 0 results, and where?

Reference point: Phase 0's best method, **PR-MADDPG** on `spread` (`phase-0/PHASE-0.md` 4.2). It
is better than MADDPG in 18 of 21 cells at no clean cost, and first in all 21 cells. Setup fixed
for any later test: Phase 0's MADDPG code on MPE `spread`, so numbers compare directly.

---

## 1. Summary

1. **The framework's own theory limits it on Phase 0's grid.** Proposition P5 says the residual
   adds nothing beyond adversarial training against i.i.d. noise, PR overrides or a
   state-feedback adversary. Phase 0's attacks are exactly these. So most Phase 0 cells
   predict a tie with an adversarially trained baseline.
2. **Two Phase 0 cells are exceptions, because P5 is stated too broadly** (section 3):
   - **R-NR:** the mean of random NR noise is a predictable gain fault, $(1-\alpha)a$.
   - **L-NR:** a learned adversary $r = \nu(s)$ changes slowly when the state does, so it is
     partly predictable from history.

   These are the only Phase 0 cells where the framework has a mechanism to beat PR-MADDPG.
3. **Takeover has the most headroom (0.58), but the framework is weakest there.** A hijacked
   agent cannot be fixed by its own residual. Teammate compensation (P6) would need behaviour
   changes larger than a small $\beta$ allows.
4. **The framework's real claim lies outside Phase 0's grid:** persistent or rate-bounded faults
   (bias, gain, drift). Phase 0 never tested them, so "outperform Phase 0" there means adding a
   new attack row and re-running the Phase 0 methods on it.
5. **The decomposition and the recurrence are separate claims.** Every gain in P3, P4 and P6
   comes from history (adaptivity), not from splitting base and residual. The split's own
   benefit, a bounded clean cost (P2), solves a problem Phase 0 did not have on `spread`:
   PR-MADDPG already improves clean return. A recurrent PR-MADDPG with the same evidence inputs
   is the control that separates the two.

---

## 2. What Phase 0's attacks are, in the framework's terms

| Phase 0 attack | Perturbation process | Predictable from history? | Framework's prediction |
| --- | --- | --- | --- |
| R-PR | i.i.d. uniform replacement, i.i.d. coin | no | P5: no within-step gain; recovery only |
| R-NR | $\tilde a = (1-\alpha)a + \alpha r$, $r$ i.i.d. uniform | **the mean is**: $\mathbb E[\tilde a \mid a] = (1-\alpha)a$, a gain fault | P5 says none; the gain part is correctable (3.1) |
| L-PR | $r = \nu(s)$, i.i.d. coin | no, for the override steps | P5: no within-step gain |
| L-NR | $r = \nu(s)$ at every step | **partly**: $\nu(s_t)$ moves slowly when $s_t$ does | P5 says none; partly correctable (3.2) |
| Takeover | victim always executes $\nu(s)$ | yes, but the victim's command is discarded | own residual irrelevant; teammates only (P6) |

Evidence available in MPE `spread` (checked in `mpe2` 1.1.1 `core.py`): the velocity update is
$v' = 0.75\,v + 5\,\tilde a\,\Delta t / m$, plus collision forces, and each agent observes its
own velocity. **An agent can recover its own executed action exactly from two consecutive
observations, except on collision steps.** This is the framework's "strong evidence" case, so
evidence quality is not the bottleneck on `spread`. Teammates' velocities are not observed,
only their relative positions, so teammate evidence needs two steps of history.

---

## 3. Where P5 is too broad, and what that predicts

### 3.1 R-NR hides a gain fault

With $r$ uniform on $[-1,1]^2$, the executed action is
$\tilde a = (1-\alpha)a + \alpha r$. Its mean $(1-\alpha)a$ is a deterministic gain
$G = (1-\alpha)I$. The doc's persistent-fault residual
$\Delta^\ast = (G^{-1}-I)\mu_b$ pre-compensates it by commanding $a^\ast/(1-\alpha)$. That is
possible when $|a^\ast| \le 1-\alpha$, a weaker condition than P4's $1-2\alpha$, which assumed
worst-case $r$. The remaining variance, $\alpha^2/3$ per dimension, cannot be removed.

A memoryless NR-MADDPG learns this scaling only for its training $\alpha$ (0.1). A residual
that infers $\alpha$ from history can adapt across the test sweep (0.1, 0.3, 0.5). This also
is consistent with a Phase 0 observation: NR-trained single agents scored *higher* under mild
NR than clean (0a finding 4). One explanation is that they learned to over-command, which suits
$G < I$ and over-shoots when $G = I$; this is untested.

### 3.2 A state-feedback adversary is temporally correlated

P5 argues that a policy trained against $\nu(s)$ already accounts for it. That holds for the
adversary seen in training. Phase 0's evaluation adversaries are new: they are trained
against each frozen team. Two consequences:

- $r_t = \nu(s_t)$ with smooth $\nu$ and slow dynamics gives an effective rate
  $\eta \approx L_\nu \lVert s_{t+1} - s_t \rVert$, so P3's $\eta$-scaled bound applies in part.
  A history-based policy can identify an unseen adversary within an episode; a memoryless one
  cannot.
- **Risk: an adaptive attacker can raise its own rate.** It is trained against the frozen
  residual, has no rate bound in Phase 0's protocol, and can learn a steep, switching $\nu$
  that defeats prediction ("bait and switch"). P2 bounds what this costs. Whether any gain
  survives depends on how high $\eta$ the attacker can reach while staying effective.

---

## 4. Hypotheses on the Phase 0 grid (`spread`, $\alpha = 0.3$)

Headroom = 1 − PR-MADDPG's competence retained, the most any method can gain over Phase 0's
best. "Recurrent AR" = PR-/NR-MADDPG with a GRU actor and the same evidence inputs, no
residual split.

| Cell | PR-MADDPG | Headroom | Mechanism | vs PR-MADDPG | vs recurrent AR |
| --- | --- | --- | --- | --- | --- |
| R-PR all | 0.90 | 0.10 | none (P5) | tie | tie |
| R-NR all | 0.92 | 0.08 | gain inference (3.1) | small gain, growing with $\alpha$ | tie |
| R-PR worst | 0.96 | 0.04 | none | tie | tie |
| R-NR worst | 0.97 | 0.03 | gain inference | at most 0.03 | tie |
| L-PR all | 0.63 | 0.37 | none for coins; adaptivity on other steps | tie | tie |
| **L-NR all** | **0.76** | **0.24** | **temporal correlation (3.2)** | **gain; the main hypothesis** | tie or small gain |
| L-PR one | 0.88 | 0.12 | teammate compensation (P6) | tie | tie |
| L-NR one | 0.91 | 0.09 | own residual (3.2) | small gain | tie |
| Takeover | 0.42 | 0.58 | P6 only; needs large $\beta$ | tie or loss | tie |

**H1 (main, on the Phase 0 grid).** Under L-NR all-agent at $\alpha$ 0.3, a residual policy
trained with Tessler-style NR adversaries retains more competence than PR-MADDPG (0.76) and
NR-MADDPG (0.72), against attackers trained on the frozen final policy. Predicted size: a
fraction of the 0.24 headroom, limited by the attacker's achievable rate (3.2).

**H2.** Under R-NR, the gain over NR-MADDPG grows with test $\alpha$ (0.1 < 0.3 < 0.5) and is
near zero at the training $\alpha$. Capped at about 0.08.

**H3 (null, a negative control).** On R-PR, L-PR and takeover, the framework ties PR-MADDPG
within Phase 0's CIs. A clear gain on these cells would contradict P5 and suggest a
capacity or tuning artefact, to be checked against the matched-parameter control.

**H4 (decomposition).** Wherever H1 or H2 holds, recurrent AR gets most of the gain. The
residual split adds a lower clean cost, which on `spread` is not measurable, since PR-MADDPG
has none.

**H5 (bounded downside, P2).** Against an attacker trained on the residual, the framework is
never worse than its own base by more than the $\beta$ bound, and on no cell worse than
PR-MADDPG beyond CI.

---

## 5. Hypotheses outside the Phase 0 grid

The framework's design target. These need a new attack row: a persistent bias $b$, gain $G$
or drift on one or all agents, random onset. Every Phase 0 method is evaluated on it from its
existing checkpoints, so this needs no retraining for the baselines.

**H6.** For persistent bias $\lVert b \rVert = \varepsilon$ on one agent, MADDPG and
PR-MADDPG lose competence that grows with $\varepsilon$. The residual's loss is set by
identification time (1-2 steps, given exact executed-action evidence, section 2), not by
$\varepsilon$, while $|a^\ast| + \varepsilon \le 1$.

**H7.** The gap in H6 is where the framework beats Phase 0 by the largest margin, and recurrent
AR without the split gets most of it too (as in H4).

---

## 6. What would falsify the framework's promise before coding

These checks use only Phase 0 checkpoints and a few hundred evaluation episodes; no training.

| Check | Measures | Kills or weakens |
| --- | --- | --- |
| Action saturation of Phase 0 policies | share of steps with $\lvert a^\ast \rvert \le 1-\alpha$ and $\le 1-2\alpha$ | H1, H2, H6 if most actions sit at the limits: no room to correct |
| Rate of Phase 0's learned adversaries | lag-1 autocorrelation and $\lVert r_t - r_{t-1} \rVert$ of $\nu(s_t)$ | H1 if $r_t$ already changes as fast as i.i.d. noise |
| Oracle compensation | frozen MADDPG plus the analytic correction from the true previous perturbation (P3/P4 formula, no learning) under L-NR, R-NR and persistent bias | an upper bound: if the oracle gains little, a learned residual gains less |
| Persistent-bias row on Phase 0 checkpoints | competence of all Phase 0 methods under bias $\varepsilon \in \{0.1, 0.3, 0.5\}$ | H6 if PR-MADDPG already handles bias well (little headroom) |

**The oracle check is the most informative.** It removes learning from the question. If the
best possible history-based correction cannot beat PR-MADDPG on L-NR, H1 fails before any
residual is built.

---

## 7. Adapting the framework to the Phase 0 setup (MADDPG, not MAPPO)

The framework doc is written for MAPPO. On Phase 0's MADDPG code:

- **Deterministic actor:** $\mu_i = \tanh(u_b + g\beta\tanh\tilde\Delta)$ with exploration noise,
  as in DDPG. The doc's "exact PPO ratio" argument no longer applies.
- **Critic action:** the TD target and Q input must use the **executed** joint action
  $\tilde a$, since that is what produced the transition. The actor gradient flows through
  $\partial\tilde a/\partial a$, which is $(1-\alpha)$ under NR and zero on PR override steps,
  as in PR-/NR-MADDPG.
- **Recurrent off-policy training:** the GRU needs sequence replay with burn-in (R2D2-style),
  not single-transition sampling. This is the largest code change and a source of instability
  that a MAPPO pilot would not have.
- **Gradient routing:** "base on clean episodes only" carries over as two actor losses on
  separately flagged replay samples.

---

## 8. Open points to decide before any pilot

1. Headline: H1 (on the Phase 0 grid, L-NR) or H6 (persistent fault, new row). H1 compares
   directly with Phase 0; H6 is where the theory is strongest.
2. Whether recurrent AR is a required control. H4 predicts it takes most of the gain; without
   it a positive result cannot be credited to the residual split.
3. Whether to run the four checks in section 6 first. They are evaluation-only on Phase 0
   checkpoints.
