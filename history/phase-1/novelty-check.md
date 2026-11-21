# Novelty check: residual-corrected action-robust cooperative MARL (Phase 1)

Status 2026-11-18. Checks the framework doc
(`Residual-Corrected MAPPO for Action-Robust Cooperative MARL Framework Definition.md`, with
its section 7 evaluation framework) and the Phase 1 goal ("verify the framework can outperform
the Phase 0 results") against the literature.

## Verdict

**The building blocks are not new, but the combination is.** Every single-agent ingredient was
published in 2023-2026, most of it in the last three months:

- a bounded, zero-initialised residual on a frozen policy;
- a recurrent residual fed by the gap between commanded and executed action;
- a learned module that corrects attacked action commands;
- rate-bounded ("temporally coupled") action adversaries with a recurrent defender.

What we did not find anywhere:

1. Tessler's PR/NR operators applied to cooperative continuous MARL, as an evaluation or as a
   defence. This is the Phase 0 gap, and it still stands.
2. A cooperative MARL agent that infers its **own** executed-action corruption from its motion
   and corrects for it. Existing MARL work detects or down-weights **faulty teammates**; it does
   not repair the agent's own actuation.
3. A formal boundary for when history-based correction can beat a memoryless robust policy
   (framework P3-P5), tested as a dose-response in perturbation predictability.
4. A PR-robust frozen base composed with an NR-trained residual, with a matched-vs-transfer
   evaluation and rank stability across operators.

The paper must therefore be positioned on items 1-4. It must not claim the residual
architecture, command-response evidence, or the rate-bounded adversary as contributions, and it
must cite and compare against the closest work below.

---

## 1. Search coverage

| Source | What |
| --- | --- |
| Undermind deep searches, new (folder `/phase1-novelty/`) | history-based adaptation to temporally correlated action perturbations (156 papers); cooperative MARL inferring or compensating own or teammate faults (169); residual correction on a frozen robust base (156); evaluation protocols across perturbation types, matched vs transfer (85) |
| Undermind deep searches, earlier | residual correction for action robustness in RL and MARL (158, 2026-11-11); MAPPO with an embedded residual head (101, 2026-11-11); PR/NR action-robust MARL, action perturbation in MARL, robust-MARL benchmarks (2026-11-14, Phase 0 section 6) |
| Web and arXiv searches | residual or action correction in MARL; temporally coupled perturbations; fault-tolerant MARL with compensation; "action robust" multi-agent |
| Full texts read | RFA (2609.17404), CAPABLE (2610.11971), ASGARD (2609.20982), GRAD (2307.12062), IBAL (2605.18024v2), Li et al. 2025 empirical study (2510.11824), AACFT (2412.00534). Parts of some appendices came only as summaries; these are marked in the notes. |
| Abstract only | ARA-RL (IEEE TAI 2025; paywalled), Joint-Health-Aware RL, self-compensating VLA, RMADDPG, Kazari 2025, Relational Q-Functionals, stability-aligned residual adaptation, recurrent TD3 under hidden disturbances, L1-adaptive and DOB add-ons, Robust-Gymnasium, RobustRLlib, RoAd-RL |

Limits: the Undermind free quota ran out mid-check (resets 2026-11-18 21:30 UTC), so no
citation-chasing from GRAD, RFA or ASGARD was done. Semantic Scholar refused automated access.
The field is moving fast: four of the closest papers appeared between Sep and Oct 2026.

---

## 2. Closest prior work

### 2.1 Single-agent: residual or correction from command-response evidence

| Paper | What it does | Overlap with the framework | Difference |
| --- | --- | --- | --- |
| **RFA**, Deng et al., arXiv 2609.17404 (Sep 2026) | frozen healthy teacher plus a **recurrent residual (GRU)**, **zero-initialised**, bounded by the teacher's remaining **headroom**; inputs include command-response features ($e_t = \hat q_t - q_t$, $\Delta e_t$, $\Delta q_t$); recurrent PPO; persistent onset faults (lock, range limit, dropout, gain, delay, bias) | nearly the whole single-agent architecture. It also runs the attribution test: the residual split beats a monolithic recurrent policy on the same faults (90.6 vs 88.0% success) | single agent; no adversary; no theory; no clean-episode penalty; no auxiliary fault-estimation loss. Its headroom scaling is the variant the framework doc rejected (section 3) |
| **CAPABLE**, Khoshnazar et al., arXiv 2610.11971 (Oct 2026) | frozen VLA plus a bounded residual (±0.1, zero-init), conditioned on a capability latent inferred from **commanded vs executed** arm action history; self-supervised motion prediction | the evidence encoder and the auxiliary prediction loss | single arm; physical faults only; beats a parameter-matched global-history baseline by 17 points; no theory |
| **ASGARD**, Salehi & Pattabiraman, arXiv 2609.20982 (Sep 2026) | **action-space attacks** (additive bias $\tilde a = a + b_t$, including slow-drift "stealthy" bias) on a UAV; a teacher-student encoder over 80 steps of history plus a **monitor that outputs corrected commands** | history-based correction of attacked action commands, the framework's core idea | single UAV; hand-designed bias profiles, no learned attacker, no i.i.d. noise; the monitor replaces commands, it is not a residual; compared with ARMOR, not with Tessler-style training |
| Joint-Health-Aware RL, arXiv 2610.41348 (Oct 2026) | models degradation as **multiplicative attenuation of executed actions** (a gain fault); a HealthNet infers per-joint health from recent observations and actions | the gain-fault correction argued for R-NR (`residual-hypotheses.md` 3.1) | single robot, physical faults |
| Self-compensating VLA, arXiv 2609.37334 (Sep 2026) | online pre-compensation from the **commanded vs executed** residual | pre-compensation from command-response evidence | test-time update, single robot |
| Stability-aligned residual adaptation, arXiv 2603.07775 (Mar 2026) | bounded, gated online residual on a frozen RL policy; recovers from mid-episode actuation shifts | gate and bound on a frozen base | empirical "near-nominal" claims, no return bound |
| L1-adaptive add-on (Cheng et al., RA-L 2022); disturbance observer + RL (Kim et al., CDC 2019) | an adaptive controller added to a pre-trained RL policy | frozen-base compensation. The DOB paper proves a performance bound | model-based compensation, not learned; single agent |
| Recurrent TD3 under evolving hidden disturbances, arXiv 2307.15931 | action history matters when the response depends on past actions | the evidence-input argument | single agent; no adversary |

### 2.2 Single-agent: temporally coupled action perturbations

| Paper | What it does | Overlap | Difference |
| --- | --- | --- | --- |
| **GRAD**, Liang et al., ICLR 2024 (arXiv 2307.12062) | temporally coupled perturbations: $\lVert a_t - \tilde a_t\rVert \le \epsilon$ and $\lVert p_t - p_{t+1}\rVert \le \bar\epsilon$, on states and actions (**additive**); a partially observable zero-sum game; **LSTM defender**; baselines include PR-MDP and NR-MDP; robust baselines degrade under coupled attacks; $\bar\epsilon$ sweep | the framework's rate bound $\eta$ is exactly GRAD's $\bar\epsilon$; history-dependent defence; Tessler as baselines | single agent; additive perturbation, not NR mixing; no residual split; no theory of when history helps |
| **ARA-RL**, Yang et al., IEEE TAI 2025 | an adversary that generates temporally coupled **action** perturbations at a bounded rate; a **diagnostic module** that detects them online | action perturbation plus online detection | single agent (abstract only; full text not read) |

### 2.3 Multi-agent: faulty or adversarial teammates

| Paper | What it does | Difference from the framework |
| --- | --- | --- |
| EIR-MAPPO, Li et al. 2023 (arXiv 2305.12872) | Bayesian belief over Byzantine teammates | teammate belief only; discrete actions in the paper |
| RMADDPG, Wang et al. 2022 | discriminator that identifies faulty agents on MPE cooperative navigation, MADDPG | detects teammates; no own-action correction |
| AACFT, Shi et al., IEEE T-ASE (arXiv 2412.00534) | attention in MADDPG actor and critic to handle **incapacitated** agents | the fault is **signalled** (a sentinel input), not inferred; agents stop, actions are not corrupted; no PR/NR |
| Kazari et al., ECAI 2025 | decentralised CUSUM detector of attacks in continuous-action MARL | detection only, no correction |
| Relational Q-Functionals, Findik et al. 2024 | teammates adapt to an unforeseen malfunction (continuous actions) | adaptation by relearning, not online inference of executed-action corruption |
| MARTA, MIR3, ROMANCE, IBAL, WALL, M3DDPG, ERNIE | robust training against action replacement or attacks (Phase 0, section 6) | no correction module; IBAL is discrete with PR-like replacement |
| DSDF, Perepu et al. 2024 | known noisy agents; per-agent discount factor | non-adversarial |

### 2.4 Evaluation protocols

| Paper | What it covers | Gap left |
| --- | --- | --- |
| Li et al. 2025, arXiv 2510.11824 | 18 tasks, MADDPG, MAPPO and HAPPO; action uncertainty as a distribution mixture $\epsilon\hat\pi + (1-\epsilon)\pi$; finds **robustness does not transfer across modalities or scopes** | clean-trained policies only (no matched/transfer split); no PR/NR operators on executed actions; no strength sweep; no robust-method comparison |
| IBAL, arXiv 2605.18024 | each robust baseline evaluated off its own attack | discrete SMAC; PR-like replacement only |
| Honhaga & Szabo 2024 | training on one perturbation overfits and weakens against others | c-MARL, discrete; not PR/NR |
| Robust-Gymnasium (2025), RobustRLlib (2026), RoAd-RL (2026) | multi-disruption benchmarks, including MARL tasks in Robust-Gymnasium | no PR×NR matched/transfer matrix or rank stability for robust-MARL methods |

---

## 3. Claim by claim

| # | Claim in the framework | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Base plus bounded, zero-initialised, gated residual on a frozen policy | **taken** | RFA, CAPABLE, stability-aligned residual adaptation; residual RL since 2018 |
| 2 | Recurrent residual fed by commanded vs executed action evidence | **taken (single agent)** | RFA, CAPABLE, self-compensating VLA, Joint-Health-Aware RL |
| 3 | Residual split beats a monolithic recurrent policy | **already reported** | RFA +2.6 points; CAPABLE +17.4 vs parameter-matched history. Our attribution controls (section 7.3) will be expected by reviewers, not a contribution |
| 4 | Rate-bounded action adversary ($\eta$) | **taken** | GRAD $\bar\epsilon$; ARA-RL |
| 5 | History-based correction of attacked action commands | **taken (single agent)** | ASGARD (bias attacks), ARA-RL (diagnostic module) |
| 6 | Auxiliary perturbation-estimation loss with privileged labels | **taken** | ASGARD (teacher-student with attack info), RAFT (privileged critic, arXiv 2608.22976), CAPABLE (self-supervised) |
| 7 | Bounded downside P2 (residual costs at most a β-sized loss) | **weakly new** | follows from the performance difference lemma; the constrained residual has Lyapunov results, the DOB add-on a performance bound. A small theory contribution at most |
| 8 | **Boundary P3-P5: history helps only for the predictable part of the perturbation; i.i.d. noise, PR overrides and state-feedback adversaries are not correctable** | **open** | no paper found states or tests this for action corruption. GRAD's $\bar\epsilon$ sweep is the nearest empirical analogue. The deep-search summary: "a formal value-of-history comparison specifically for action corruption remains a clear gap" |
| 9 | **Own-action correction in cooperative CTDE MARL** | **open** | MARL work detects or down-weights teammates (EIR, RMADDPG, AACFT, Kazari) or adapts by relearning (RQF); none repairs the agent's own executed action |
| 10 | Teammate compensation (P6) | **partly open** | teammate belief and detection exist; physical compensation of a teammate's corrupted action by a learned decentralised policy was not found. Classical distributed fault-tolerant control covers it with observers and communication |
| 11 | **Tessler PR/NR operators in cooperative continuous MARL** | **open** | unchanged since Phase 0 section 6. Li et al. 2025 mixes action distributions on clean-trained policies; IBAL is discrete |
| 12 | **Matched vs transfer split across PR and NR, with rank stability** | **open at protocol level** | cross-modality non-transfer is known (Li et al. 2025, Honhaga 2024), so the motivation is established, but this matrix for robust-MARL methods is not |
| 13 | Correlated random source $\mathrm{R}_\phi$ as a predictability dose-response | **partly open** | GRAD sweeps coupling for an adversary in single-agent RL; as a random-fault axis in MARL, not found |
| 14 | **PR-robust frozen base composed with an NR-trained residual** | **open** | the deep search found no stacking of robustness to two action operators |

---

## 4. What this means for Phase 1

**Positioning.** Lead with the evaluation and the boundary, not the architecture:

1. First PR/NR action-robustness benchmark for cooperative continuous MARL, with a
   matched/transfer split and rank stability (claims 11-12; Phase 0 provides the base).
2. When history-based correction can help: P3-P5 plus the $\mathrm{R}_\phi$ dose-response
   (claims 8 and 13).
3. A MARL instantiation: own-action correction under CTDE on a PR-robust base, claimed for NR
   only (claims 9 and 14), with RFA, CAPABLE, ASGARD and GRAD cited as the single-agent
   precedents.

**Changes to the framework doc.**

- Section 1 "adversary budget": cite GRAD for the rate bound $\eta$; do not present it as new.
- Section 2 architecture: cite RFA and CAPABLE. Keep the pre-squash residual but add RFA's
  headroom-scaled residual as an ablation, because a reviewer will ask why ours differs.
- Section 7.3 controls: make the REC control explicitly a **GRAD-style recurrent defender**
  (LSTM, adversarially trained under the same NR mixture, no split). That makes it the strongest
  single-agent method ported, not an ad-hoc control.
- Section 7.4: keep $\mathrm{R}_\phi$; relate $\phi$ to GRAD's $\bar\epsilon$ in the text.
- Section 8: state P2 as a standard consequence; put the novelty claim on P3-P5.

**Risk.** Four of the closest single-agent papers appeared in Sep-Oct 2026, and a MARL version
of RFA, ASGARD or CAPABLE is a short step. Re-run this check before submission, and run the
citation searches skipped here (citers of GRAD, RFA, ASGARD, CAPABLE) once the Undermind quota
resets.

## 5. References (by arXiv id or DOI)

- RFA: arXiv 2609.17404. CAPABLE: arXiv 2610.11971. ASGARD: arXiv 2609.20982.
- Joint-Health-Aware RL: arXiv 2610.41348. Self-compensating VLA: arXiv 2609.37334.
- Stability-aligned residual adaptation: arXiv 2603.07775. RAFT: arXiv 2608.22976.
- Recurrent TD3 under hidden disturbances: arXiv 2307.15931.
- L1-adaptive add-on: doi 10.1109/LRA.2022.3169309. DOB + RL: doi 10.1109/CDC40024.2019.9028930.
- GRAD: arXiv 2307.12062 (ICLR 2024). ARA-RL: doi 10.1109/TAI.2024.3499938.
- Modified-action MDP (intended vs executed actions): arXiv 2102.07716.
- EIR-MAPPO: arXiv 2305.12872. RMADDPG: doi 10.1088/1742-6596/2258/1/012064.
- AACFT: arXiv 2412.00534 (doi 10.1109/TASE.2025.3592721). Kazari et al.: doi 10.3233/FAIA251233.
- Relational Q-Functionals: doi 10.1109/UR61395.2024.10597441.
- IBAL: arXiv 2605.18024. Li et al. empirical study: arXiv 2510.11824.
- Honhaga & Szabo: doi 10.1177/00375497241232432. Robust-Gymnasium: arXiv 2502.19652.
- RobustRLlib: arXiv 2610.32971. RoAd-RL: arXiv 2606.29867.
- Tessler et al. 2019: arXiv 1901.09184.
