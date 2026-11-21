> Snapshot of `formal_definition.md` from the benchmark repo (sihonglab/ayesha-benchmark-au, commit 961fdc5, 2026-11-21). Edit the original there, not this copy.

# Robust cooperative MARL under action uncertainty: formal definition

Status: draft v0.2, 2026-11-19. Continuous, bounded actions. One item is still open (section 6).

## 1. The problem in one paragraph

A team of agents shares one reward. Each agent chooses an action, but a **corrupter** may
change it before it is executed. The corrupter can be a random fault or an attacker; both are
drawn from a stated **allowed set**. A team policy is robust if it does well across a stated
list of such sets. Robustness is reported as a **profile** over that list, not as one number.

## 2. The team

A cooperative task with $n$ agents:

| Symbol | Meaning |
| --- | --- |
| $s_t$ | state of the world at step $t$ |
| $o_{i,t}$ | agent $i$'s observation |
| $a_{i,t} \in [-1,1]^d$ | action agent $i$ **chooses** |
| $\tilde a_{i,t}$ | action actually **executed** |
| $P(s_{t+1} \mid s_t, \tilde a_t)$, $r(s_t, \tilde a_t)$ | dynamics and shared reward, both driven by the executed actions |
| $\gamma \in [0,1)$ | discount |
| $h_{i,t}$ | agent $i$'s history: its past observations and its own chosen actions |

- Each agent acts on its own history: $\pi_i(a_{i,t} \mid h_{i,t})$. The team policy is
  $\pi = (\pi_1, \dots, \pi_n)$.
- Execution is decentralised. How a method is trained is not part of the problem.
- **What agents know about corruption** is whatever their observations contain. If an agent can
  sense its own executed action, that sensor is part of $o_{i,t}$. The definition does not
  treat it separately.

## 3. The corruption channel

Each agent's chosen action passes through a channel:

$$
\tilde a_{i,t} = \mathrm{clip}\big(C(a_{i,t}, u_{i,t})\big), \qquad C(a, u_\varnothing) = a .
$$

Here $u_{i,t}$ is the corruption chosen by the corrupter, $u_\varnothing$ means none, and
$\mathrm{clip}$ keeps the executed action within $[-1,1]^d$.

| Channel | $\tilde a$ |
| --- | --- |
| Additive push | $a + \delta$ |
| Noisy mixing (NR; Tessler et al. 2019) | $(1-\alpha)\,a + \alpha\, b$ |
| Override (PR; Tessler et al. 2019) | $b$ with probability $\alpha$, else $a$ |
| Actuator fault | $G\,a + c$ |
| Stuck actuator | $b$, held over several steps |

## 4. The corrupter

A **corrupter** $\nu$ is a rule, possibly random, that picks $u_t = (u_{1,t}, \dots, u_{n,t})$
at every step. An **allowed set** $\mathcal{V}$ of corrupters is fixed by a channel, a
budget and a level:

- **Random fault:** $\mathcal{V}$ contains one known random process.
- **Attacker:** $\mathcal{V}$ contains every corrupter within the budget; the most harmful one
  counts.

**Budget.** It has four parts. Any part may be "no limit".

| Part | Limits |
| --- | --- |
| $k$ | how many agents are corrupted |
| $\varepsilon$ | how large the corruption is at each step ($\lVert\delta\rVert$, or $\alpha$) |
| $\eta$ | how fast the corruption changes between steps |
| $K$ | how many steps of an episode are corrupted |

**Level.** This is what the corrupter can use when it picks $u_t$:

| Level | Depends on | Example |
| --- | --- | --- |
| L0 | the step number and its own randomness | i.i.d. noise; a motor that fails at step 50 |
| L1 | plus past states and executed actions | a fault that worsens with use |
| L2 | plus the current state $s_t$ | an attacker watching the world |

**Rules.**
- The corrupter and the team choose at the same time. The corrupter never sees the team's
  chosen action for the current step.
- An L2 corrupter may be **adapted**, meaning built against this team's policy. It sees the
  state but not the agents' internal memory. If the policy is deterministic, an adapted L2
  corrupter can work out the chosen actions from the state; this is allowed.
- The team knows the channel and the budget, but not which corrupter in $\mathcal{V}$ it
  faces.

## 5. Robustness

**Value** of team policy $\pi$ against corrupter $\nu$:

$$
J(\pi, \nu) = \mathbb{E}\Big[\textstyle\sum_{t \ge 0} \gamma^t\, r(s_t, \tilde a_t)\Big].
$$

A **condition** $c$ is one allowed set $\mathcal{V}_c$. Its value is the worst case over that set:

$$
J(\pi, c) = \inf_{\nu \in \mathcal{V}_c} J(\pi, \nu).
$$

For a random-fault condition this is just the average under that fault. Clean is the condition
$c_0$, with no corruption.

**Robustness profile.** Given a conditions list $\mathcal{K} = (c_0, c_1, \dots, c_m)$ from clean
to severe:

$$
P_{\mathcal{K}}(\pi) = \big(J(\pi,c_0), J(\pi,c_1), \dots, J(\pi,c_m)\big).
$$

**Comparison.** Two profiles are compared with a tolerance $\delta$:
- $\pi$ **dominates** $\pi'$ if it is at least as good in every condition (within $\delta$) and
  better by more than $\delta$ in at least one.
- Otherwise the two **trade off**, and each has conditions where it is the better one.

**The problem.** Robust cooperative MARL under action uncertainty means finding team policies
that no other policy dominates on a stated $\mathcal{K}$. In general there is no single best one.
Single numbers, such as the worst case or the area under the curve, may summarise a profile,
but they do not define robustness.

## 6. Open

- **The standard conditions list $\mathcal{K}$ and the tolerance $\delta$.** Which channels,
  budgets and levels make up the list, and how small a difference counts as "about equal". A
  proposal is in `environments_and_budgets.md`.

## 7. Challenges of the definition

1. **Order matters.** An adapted corrupter is chosen after the team policy, so it can target
   that policy's weak spots. "Best policy against the worst corrupter" and "worst corrupter
   against the best policy" are generally different numbers.
2. **A robust solution may not exist, and is hard to compute.** Finding the best decentralised
   policy is already very hard without corruption. Adding a worst case makes it a game, and in
   such games a stable solution can fail to exist. Han et al. (2022) show this for corrupted
   observations; for actions it is open.
3. **The allowed set is easy to get wrong.** Too large, and every policy is hopeless. Too
   small, and real faults are not covered. The four budget parts have different units, so sets
   are hard to compare.
4. **Physics caps what is achievable.** Actions are bounded. Under mixing with weight $\alpha$,
   a chosen action can only be fully restored if it lies within $1-2\alpha$ of zero. A low
   value may mean the task is impossible, not that the policy is poor.
5. **The worst case is only ever estimated.** Attackers found by training give *a* bad case,
   not *the* worst case. A policy can also look robust on one set and fail just outside it.
6. **Multi-agent effects.** Corrupting one agent changes what its teammates see, so harm
   spreads. Healthy agents may partly cover for it. A clever corrupter picks the weakest
   agents. A single team reward can hide which part of the task failed.
7. **Memory changes what is possible.** Corruption that is correlated in time can be inferred
   from an agent's own chosen actions and their observed effects. So policies must be allowed
   to use history. The corrupter's hidden internal state also makes the problem partially
   observed, even if the original task was not.
