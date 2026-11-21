> Snapshot of `environments_and_budgets.md` from the benchmark repo (sihonglab/ayesha-benchmark-au, commit 961fdc5, 2026-11-21). Edit the original there, not this copy.

# Experimental environments and attack budgets

Status: draft v0.1, 2026-11-21. This file proposes where to test and how strong the corruption
should be. It fills the open item in `formal_definition.md` section 6: the standard conditions
list $\mathcal{K}$ and the tolerance $\delta$. Terms (channel, budget $k, \varepsilon, \eta, K$,
level L0–L2, condition, profile) are as defined there.

## 1. What existing methods used

Most methods are tested on small MPE tasks. Each paper picks its own budget, usually one or two
values, and almost none sweeps the strength. Only MARTA (difficulty levels), the noise
curriculum (noise sweep) and Li et al.'s study (two scopes) report more than one setting.

| Method | Environment (continuous) | Channel | Budget used |
| --- | --- | --- | --- |
| M3DDPG | MPE, 4 mixed tasks | push, inside the critic | step size chosen from 0.001, 0.01, 0.1 |
| ROMAX | MPE, 2 mixed tasks | push, inside the critic | a small ball around the chosen actions |
| MARTA | MPE SimpleTag | override, one agent | fault probability $p$ 0.4, 0.6, 0.8; three difficulty levels |
| ERNIE-A-c | MPE spread (our runs) | push, inside the critic | $\varepsilon$ 0.05 |
| Curriculum | MPE, 3 tasks | Gaussian push, all agents | noise raised until training fails |
| DSDF | Water World | gain fault plus noise | degradation $\beta$, different per agent |
| RMADDPG | MPE navigation, 4 agents | random override, whole episode | $k$ 0, 1, 2 |
| AACFT | MPE, 4 tasks, 3 agents | stuck from a random step | $k$ 1 |
| CA-MQF / RQF | MaMuJoCo Ant, 4 agents | one leg stuck | $k$ 1, permanent |
| Li et al. 2025 (study) | DexHand, Quadrotor swarm, Voltage | mixing with random, greedy or learned $b$ | all agents $\varepsilon$ 0.1; one agent $\varepsilon$ 0.2 |
| Tessler et al. 2019 (single agent) | MuJoCo | PR, NR | $\alpha$ 0.1 in training |

Two points follow. First, a value such as "$\alpha$ 0.3" is not comparable across papers or
tasks: the same number can be mild in one task and severe in another. Second, no paper tests
slow drift, and only CA-MQF tests a fault on a real actuator.

## 2. Candidate environments

| Family | What it is | Strengths for us | Weaknesses |
| --- | --- | --- | --- |
| MPE (Lowe et al. 2017; PettingZoo) | 2D point agents, e.g. spread, tag | cheap; used by most methods above, so results can be compared | simple physics; short episodes (25 steps); little room for persistent faults to matter |
| MaMuJoCo (Peng et al. 2021) | MuJoCo robots split into agents, e.g. Ant, HalfCheetah | real torques in $[-1,1]$, so stuck and gain faults mean something; long episodes | locomotion only; coordination is inside one body |
| VMAS (Bettini et al. 2022) | vectorised 2D multi-robot tasks, e.g. transport, balance, navigation | fast on GPU; real cooperation between separate robots; bounded forces | less used by robust-MARL papers so far |
| Multi-drone simulators (e.g. gym-pybullet-drones, Panerati et al. 2021; quadrotor swarm used by Li et al. 2025) | quadrotors with motor commands | the most realistic actuator faults | slow; few baselines; set-up effort |

**Proposal.** Use three tiers. MPE `spread` and `tag` are the comparison tier: they link to the
existing literature. VMAS `transport` and `balance` are the main tier: separate robots that must
cooperate, fast enough for many seeds and conditions. MaMuJoCo Ant (4×2) and HalfCheetah (6×1)
are the actuator tier, where stuck and gain faults are physical. A drone task is optional, as a
final check, if time allows.

## 3. Budget values

Every condition uses the same three strengths, mild, medium and severe, so that profiles have the
same shape in every environment. The numbers start from values common in the literature and are
then calibrated, as described in section 4.

| Budget part | Values | Source |
| --- | --- | --- |
| $k$ (agents corrupted) | 1, or all | Li et al. 2025; RMADDPG; AACFT |
| $\varepsilon$ for additive push | 0.1, 0.2, 0.3 (actions in $[-1,1]$) | Li et al. 2025 |
| $\alpha$ for mixing (NR) | 0.1, 0.3, 0.5 | Tessler et al. 2019; Li et al. 2025 |
| $\alpha$ for override (PR) | 0.1, 0.3, 1.0 (1.0 = full takeover) | Tessler et al. 2019; MARTA |
| $G$ for gain fault | 0.75, 0.5, 0.25 | DSDF |
| $\eta$ for slow drift | drift reaches 0.1, 0.3, 0.5 over one episode | new; no paper uses it |
| $K$ (steps corrupted) | no limit, except stuck: from a random step onward | AACFT |

## 4. Calibration

Fixed numbers do not mean the same in every task. So, for each environment, the base (MADDPG)
is run once at every value in section 3, and each value gets a label from how much reward the
base loses. Loss is measured as the share of the gap between the base's clean reward and a random
team's reward.

| Label | Base loses |
| --- | --- |
| mild | under 10% |
| medium | 10–40% |
| severe | over 40% |

If a value lands in the wrong band, it is moved up or down the same scale until each band has
one value. The raw budget numbers and the base's raw rewards are reported next to the labels,
so readers can still compare with papers that use fixed numbers.

## 5. Proposed conditions list $\mathcal{K}$

Each row is one condition, run at mild, medium and severe strength. Adapted attackers (L2) are trained against each team policy
separately.

| # | Condition | Channel | $k$ | Level |
| --- | --- | --- | --- | --- |
| 0 | Clean | none | none | none |
| 1 | Random noise | additive Gaussian push, size $\varepsilon$ | all | L0 |
| 2 | Random mixing | NR with random $b$ | all | L0 |
| 3 | Mixing attack, all | NR with learned $b$ | all | adapted L2 |
| 4 | Mixing attack, one | NR with learned $b$ | 1 (worst agent) | adapted L2 |
| 5 | Random override | PR with random $b$ | 1 (worst agent) | L0 |
| 6 | Override attack | PR with learned $b$ | 1 | adapted L2 |
| 7 | Gain fault | $G\,a$ | 1 | L0 |
| 8 | Stuck actuator | stuck from a random step | 1 | L0 |
| 9 | Slow drift | push that grows at rate $\eta$ | 1 | L1 |

"Worst agent" means each agent is corrupted in turn and the lowest reward counts. Stuck actuator
has one strength only (stuck at a random value). That gives 8 × 3 + 1 + clean = 26 conditions
per environment. (If that is too many, conditions 2 and 4 are the first
to drop, since they are closest to 1 and 3.)

## 6. Proposed tolerance $\delta$

Two policies are "about equal" in a condition unless both of these hold:

- **The difference is real:** the paired 95% confidence interval over seeds (at least 5, paired
  by seed) excludes 0.
- **The difference is large enough to matter:** it is at least 5% of the gap between the base's
  clean reward and a random team's reward in that environment.

The first rule stops seed noise from counting as a difference. The second stops tiny but
significant differences, like the 0.1–0.3 reward gaps seen on `spread`, from counting as
robustness.

## 7. Open choices

- The band edges (10% and 40%) and the 5% tolerance are first guesses. They should be checked
  once the base has been run in each environment.
- Whether to add a second base besides MADDPG, such as MAPPO, since some methods (DSDF) are
  built on other algorithms.
- Which drone task, if any.
