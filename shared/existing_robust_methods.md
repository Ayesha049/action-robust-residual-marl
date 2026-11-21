> Snapshot of `existing_robust_methods.md` from the benchmark repo (sihonglab/ayesha-benchmark-au, commit 961fdc5, 2026-11-21). Edit the original there, not this copy.

# Existing action-robust methods in cooperative MARL

Status: draft v0.4, 2026-11-21. Each method is placed in the terms of `formal_definition.md`.
Every card is based on the paper's full text.

**Scope.** This file covers cooperative MARL methods shown on continuous actions, where the
executed action can differ from the chosen one. ERNIE-A is tested only on discrete actions, so it
is included as ERNIE-A-c, the continuous adaptation built for the archived Phase 0 runs. Section 4
lists what is out of scope, with reasons.

**Coverage.** The list was checked with two Undermind deep searches, "Action robust cooperative
MARL methods census 2026" (55 papers) and "Fault tolerant cooperative MARL actuator faults census
2026" (106 papers), plus a citation search from M3DDPG, ERNIE and MARTA. Every paper they returned
has a card in section 3 or a reason in section 4.

**Conditions.** The predictions use these seven example conditions, since the
standard list is still open in the definition:

| Condition | Channel | Corrupted agents | Level |
| --- | --- | --- | --- |
| Clean | none | none | none |
| Random noise | small additive push | all | L0 |
| Mixing attack | noisy mixing (NR) | all | adapted L2 |
| Random override | override (PR), probability $\alpha$ per step | one | L0 |
| Override attack | override (PR), probability $\alpha$ per step | one | adapted L2 |
| Stuck actuator | stuck from a random step onward | one | L0 |
| Slow drift | rate-limited additive push | one | L1 |

## 1. Summary

| Category | Method | In one line | Fit |
| --- | --- | --- | --- |
| adversarial | M3DDPG (Li et al., AAAI 2019) | critic assumes the others act slightly worse than chosen | partly |
| adversarial | ROMAX (Sun, Kim & How, ICRA 2022) | M3DDPG with a certified worst case | partly |
| adversarial | MARTA (Mguni et al., arXiv 2508.08800) | trains against learned overrides of one agent at harmful moments | inside, for its tests |
| regularisation | ERNIE-A-c (adapted from Bukharin et al., NeurIPS 2023) | penalises value changes under small action pushes | partly |
| random faults or noise | Curriculum over uncertainty (Agrawal et al., 2023) | action noise that grows over a curriculum | inside, for its action-noise tests |
| random faults or noise | DSDF (Perepu, Dey & Das, CODS-COMAD 2024) | noisy agents get a shorter planning horizon | inside, for a fixed degradation |
| random faults or noise | RMADDPG (Wang, Huang & Hou, 2022) | healthy agents spot and ignore randomly acting teammates | edge |
| random faults or noise | AACFT (Shi et al., IEEE T-ASE) | attention lets the team carry on when a flagged agent stops | inside, for 2 of 4 tasks |
| adaptation after a fault | CA-MQF / RQF (Findik et al., CDC 2024, UR 2024) | after a known fault, retrains with rewards reweighted | edge |

## 2. Expected performance

Each cell **predicts** the method against its base (usually MADDPG): ↑ better, ↓ worse, = about
the same, ? unclear. Symbols show direction only, not size; most expected gains are small.

| Method | Clean | Random noise | Mixing attack | Random override | Override attack | Stuck actuator | Slow drift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M3DDPG | = | ↑ | ↑ ¹ | = | = | = | = |
| ROMAX | ↓ | ↑ | ↑ ¹ | = | = | = | = |
| MARTA | = | = | ? | ↑ | ↑ | ↑ | = |
| ERNIE-A-c | = | ↑ | ↑ ² | = | = | = | = |
| Curriculum over uncertainty | = | ↑ | = | ? | = | = | = |
| DSDF | = | ↑ | = | ? | = | ? | = ³ |
| RMADDPG | = | = | = | ↑ ⁴ | ? | ? | = |
| AACFT | = | = | = | = | = | ↑ ⁵ | = |
| CA-MQF / RQF | = | = | = | = | = | = ³ | = |

¹ only at small $\alpha$. ² only if $\varepsilon$ is large. ³ ↑ only after retraining. ⁴ only if
$\alpha$ is high. ⁵ only if the fault is flagged; ? otherwise.

## 3. Methods

Methods are grouped by what trains the team. An "Also uses" line names any second mechanism.

### 3.1 Adversarial training

The team trains against a worst case: a small push inside the critic (M3DDPG, ROMAX) or a learned
corrupter of executed actions (MARTA). This targets adapted corrupters, but only as far as the
trained attacker reaches: a small push does not prepare for large overrides.

#### M3DDPG (Li et al., AAAI 2019)

**Method.** MADDPG with a pessimistic critic. Before scoring a joint action, each critic nudges
the other agents' actions, with one small gradient step, in the direction that lowers the score
most. Tested on four mixed MPE tasks against opponents trained by other methods.

**In our definition.** An additive push of size $\varepsilon$ on all other agents, every step,
adapted and computed from the chosen actions (the excluded level). It exists only inside the
critic, and the tests change opponents rather than corrupt actions. Fit: partly.

#### ROMAX (Sun, Kim & How, ICRA 2022)

**Method.** M3DDPG's idea with a better worst case: a convex relaxation of the critic gives a
certified lower bound on the value in a ball around the chosen actions. The actor optimises a mix
of this and the normal value. Tested in cross-play on two mixed MPE tasks.

**In our definition.** As M3DDPG: a push within $\varepsilon$ inside the critic only, at the
excluded level; no action corruption in the tests. Fit: partly.

#### MARTA (Mguni et al., arXiv 2508.08800)

**Method.** The team trains against two learned opponents. A Switcher decides when, and on which
agent, a malfunction happens, under a cost or a fixed number of faults; an Adversary then replaces
that agent's action with a random or worst-case one. Tested with random and worst-case faults at
three difficulty levels. The only continuous task is MPE SimpleTag on MADDPG.

**In our definition.** Override of one agent, unlimited size, step budget $K$, adapted; the
Switcher sees proposed actions (the excluded level). Its tests are random override (L0) and
worst-case override (adapted L2), and its difficulty levels are the closest existing work to a
robustness profile. Fit: inside, for its tests.

### 3.2 Regularisation

A penalty makes the value change little when actions change slightly. This gives smoothness
against small pushes from any source, but not against larger changes such as overrides.

#### ERNIE-A-c (our continuous adaptation of ERNIE-A; Bukharin et al., NeurIPS 2023)

**Method.** The critic is penalised by how much the team value changes under the worst small
action change. The paper tests this only on discrete traffic-light control. ERNIE-A-c, built on
MADDPG for the archived runs, uses $\lambda \max_{\lVert d\rVert_2 \le \varepsilon} (Q(s,a) -
Q(s,a+d))^2$, with $d$ found by three steps of projected gradient ascent ($\lambda$ 1,
$\varepsilon$ 0.05 after tuning).

**Also uses:** the worst-case push is a small adversarial step, as in M3DDPG.

**In our definition.** A push within $\varepsilon$ on the joint action, inside the critic only,
at the excluded level. Fit: partly.

### 3.3 Training with random faults or noise

Random faults or noise are injected in training, sometimes with a fault-aware part (detector,
attention, per-agent discount). This targets L0 random faults and gives no special protection
against an adapted attacker.

#### Curriculum over uncertainty (Agrawal, Aralikatti, Sun & Huang, arXiv 2310.08746, 2023)

**Method.** Training adds noise to rewards, observations or actions, and a curriculum slowly
raises it. Executed actions are drawn from a truncated Gaussian around the chosen ones. Tested on
three MPE tasks against training without the curriculum, reporting the highest noise level that
still reaches 90% success.

**In our definition.** For actions, a bounded random push on all team agents, L0. Its noise sweep
is a one-axis robustness profile. Fit: inside, for its action-noise tests.

#### DSDF (Perepu, Dey & Das, CODS-COMAD 2024; arXiv 2021 version)

**Method.** For teams in which some agents have degraded. A hypernetwork learns each agent's
discount factor from observations, so noisy agents plan short and take nearby tasks. A degraded
agent executes $a \pm \beta a \pm \epsilon$, with $\beta$ differing per agent. The policy is
retrained when degradation passes a threshold. The continuous test is Water World on FACMAC.

**Also uses:** retraining when the degradation changes.

**In our definition.** An actuator gain fault plus noise on some agents, persistent, L0, inferred
from observations. Fit: inside, for a fixed degradation level.

#### RMADDPG (Wang, Huang & Hou, J. Phys. Conf. Ser. 2258, 2022)

**Method.** Healthy agents train while some teammates act at random for the whole episode. Each
learns a classifier that flags faulty teammates from five recent observations. Tested on
four-agent cooperative navigation with up to two faulty agents.

**Also uses:** a learned detector of faulty teammates.

**In our definition.** Random override every step on up to $k$ agents, L0. But only the healthy
agents must cover the landmarks, so the task changes. Fit: edge.

#### AACFT (Shi et al., IEEE T-ASE; arXiv 2412.00534)

**Method.** MADDPG with attention in actor and critic. A failed agent stops moving and observing,
a fault flag replaces its inputs, and attention shifts away from it. Tested on four MPE tasks with
three agents, one failing at a random step; in two tasks the failure adds a new sub-task.

**Also uses:** attention steered by a fault flag.

**In our definition.** A stuck actuator on one of three agents, L0, with the flag as part of the
observation. Fit: inside for two tasks; edge for the two where the task changes.

### 3.4 Adaptation after a fault

The team is not made robust in advance; it retrains after a known fault. Our definition asks for a
fixed policy, so these methods are at its edge.

#### CA-MQF / RQF (Findik, Hasenfus & Azadeh, IEEE CDC 2024; UR 2024)

**Method.** Once the team knows which agent failed, a relational network reweights rewards so
healthy agents value helping it, and training continues. Tested on MaMuJoCo Ant with one leg per
agent, one leg immobilised after the team has learned to walk.

**Also uses:** a relational network over the agents' rewards.

**In our definition.** A permanent stuck actuator on one agent, L0. At deployment, before
retraining, it is just its base algorithm. Fit: edge.

## 4. Out of scope

Groups 1 and 2 are the closest to being in scope.

1. **Continuous modular robots tested under actuator faults, not trained for them.** Each joint or
   leg is an agent, and decentralised control tolerates a failed module. MAGCLA (Tao et al., ICRA
   2023); multi-agent space manipulator (Zhang et al., 2024); local-observation MARL (Zhao et
   al., 2023); soft arms (Hou et al., 2026; SoftGM, 2026); SpaceOctopus (Zhao et al., 2024);
   DEMOS (Guo et al., RA-L 2023); COMPOSER (Zhang et al., ICRA 2024).
2. **An agent leaves or is replaced, rather than corrupted.** MAGEC (Goeckner et al., IROS 2024);
   patrolling (Tong et al., 2022); agent dropout (Fiscko et al., TCNS 2023); C3F (Jia et al.,
   RA-L 2024); self-healing swarms (Dhanraj et al., 2026); swarm coordination (Ahmed et al.,
   2026; abstract only); intralogistics swarm (Alboraei et al., 2026; no text found); switched
   control (Shen et al., T-ASE 2026); BOSA (Wu et al., 2026).
3. **Discrete actions only.** ERNIE-A as published; ROMANCE (Yuan et al., AAAI 2023); WALL /
   Wolfpack (Lee et al., ICML 2025); IBAL (Lee et al., 2026); coach-assisted crashed agents
   (Zhao et al., 2022); Collaborative Adaptation (Findik et al., 2023); malfunctioning swarms
   (Pina et al., 2024, 2025); certified policy smoothing (Mu et al., 2022); Klíma et al. 2018.
4. **Hostile or compromised teammate.** MIR3, MIR2 (Li et al.); EIR-MAPPO / BARDec (Li et al.,
   ICLR 2024); BATPAL (Kazari & Dán, ICLR 2026); RADAR, ARTS (Phan et al.); ARNEQ / ART (Nika et
   al., TMLR 2024); resilient consensus (Hou et al., AAAI 2021); Byzantine MARL (Medhi et al.
   2023; Kandar & Baranwal 2025); Nisioti et al.; Sun & Wang 2026 (tabular; the attacker sees the
   planned action).
5. **Opponents changing policy in mixed games.** M3DDPG and ROMAX have cards only because later
   action-robust work uses them as baselines. FD2PG (Jiang et al., 2020); Friend-or-Foe MADDPG
   (Sun et al., 2022); MADDPGX (Paczolay & Harmati, 2024); Zhao et al. 2022; STeaM (Yan);
   rational policy gradient (Lauffer et al., 2025); adversarial style sampling (Osa & Harada,
   2024).
6. **Single agent.** PR/NR-MDP (Tessler et al., ICML 2019); ARRLC (Liu et al., 2023); EAR-DDPG
   (Kim & Shibuya, 2026); Tan et al. 2020; GRAD (Liang et al., ICLR 2024); ARA-RL (Yang et al.,
   2025); single-robot joint-fault adaptation (Deng, Pham, Okamoto, Castro, Li).
7. **Observation or sensor attacks only.** ATSA (Zhou et al., ICML 2025); RoQMIX (Guo et al.,
   2025); Zan & Zhu 2025; FT-Attn (Geng et al., 2019); ADSDS (Niu & Li, 2025); state-adversarial
   MARL (Han et al. 2022; He et al. 2023); robust mean-field AC (Zhou et al., 2022); R-CCMARL
   (Wang et al., 2025).
8. **Model uncertainty in rewards or transitions.** Zhang et al. (NeurIPS 2020); distributionally
   robust MARL (Qu et al. 2026; Farhadi et al. 2025; Shi et al. 2024).
9. **RL-tuned formation or consensus controllers with actuator faults.** Control-theory designs,
   not MARL policies; about 35 papers, e.g. Bai et al. 2023, Zhu et al. 2024, Mei et al. 2025–26,
   Elferik et al. 2023.
10. **Not methods** (studies, detectors, surveys, attacks, benchmarks). Li et al. (NeurIPS 2025);
    detection (Kazari et al. 2023, 2025; Lischke et al. 2022); failure analysis (Shefin et al.,
    2026); Utimula et al. 2024; Zhao et al. 2020; surveys (Standen et al. 2023, 2024); attacks
    (Hu & Zhang 2022; SAJA; CuDA2; AMCA; CAMA; Liu et al. 2022); Robust Gymnasium; Jing et al.
    2024; GNAs (Zhang et al. 2023).

## 5. What the tables suggest

- **Mixing attack and slow drift:** no method is expected to beat its base, and none keeps a
  memory of past corruption.
- **Random noise:** DSDF and the noise curriculum, which train for it.
- **Overrides:** MARTA.
- **Stuck actuators:** AACFT, but only when the fault is flagged.
- **Clean:** all close to their base.
