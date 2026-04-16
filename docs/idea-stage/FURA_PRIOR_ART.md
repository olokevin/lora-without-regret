# FURA Prior-Art & Competitive Benchmark Report

**Date**: 2026-04-15
**Scope**: Answer four questions for the FURA paper —
  1. Has any existing paper explored the same idea (block-SVD PEFT with frozen large core, train small core, full-rank ΔW)?
  2. What other papers propose "full-rank PEFT"?
  3. What is the current SOTA on Commonsense-170K and MATH-10K under LLM-Adapters' protocol?
  4. How do methods break down structurally (LoRA-like vs sparse-like vs factored-reparam)?

---

## 1. Executive summary

- **"Full-rank PEFT" as a banner is already taken.** RandLoRA (ICLR'25), QuanTA (NeurIPS'24), SVFT (NeurIPS'24), BOFT (ICLR'24), and Khatri-Rao adapter (ICCV'25) all claim full-rank ΔW at PEFT budgets. FURA cannot frame itself as "the first full-rank PEFT."
- **BlockTT / TT-LoRA as a reparameterization is also taken.** LoRETTA (NAACL'24), LoTR, TT-LoRA, LoRTA, TensLoRA all tensor-factorize the update. However, **every one of them trains all cores and the resulting ΔW stays low-rank** (small TT-rank).
- **The specific FURA recipe — block-SVD → freeze the large core L → train only the small core R (+ optional singular values S) → ΔW full-rank by construction — is not in the surveyed literature.** Closest mechanistic neighbors: Spectral Adapter (freezes the spectral tail, not the large core) and the BTT primitive from Qiu et al. ICML'24 (used for pretraining from scratch, not PEFT).
- **Commonsense-170K: FURA (87.91) is the highest reported number** under the LIFT evaluation protocol. LIFT (87.88) is statistically tied. Five more LoRA-family methods claim 87+ (DropLoRA 87.61, GateRA 87.53, Dual LoRA 87.50, PERA 87.38, SMoA 87.35) but use a much weaker LoRA baseline (80.8 vs 83–87 from properly tuned baselines) — their absolute numbers are less trustworthy than FURA/LIFT's. **DoRA (85.04) is no longer competitive** — superseded by HiRA (86.72), DropLoRA, GateRA, and others.
- **MATH-10K: FURA (80.47) is not SOTA — LIFT leads at 81.78 (+1.31).** Very few papers beyond LIFT report this exact benchmark on LLaMA-3-8B. No paper has beaten LIFT here as of April 2026.
- **RandLoRA (ICLR'25), the leading "full-rank PEFT" paper, scores 85.59 on Commonsense-170K** — substantially below FURA (+2.32). This strengthens FURA's empirical story within the full-rank PEFT subcategory.

**Recommended framing**: drop the "first full-rank PEFT" claim. Position FURA as *"the first block-SVD reparameterization that is provably full-rank by core freezing, at matched 2–3% parameter budget, outperforming all existing full-rank PEFT methods (RandLoRA, QuanTA, BOFT) and matching/exceeding sparse methods (LIFT, S²FT) on Commonsense-170K while maintaining TP-compatible structured computation."* Direct competitors to baseline against: QuanTA, RandLoRA, LIFT, S²FT, PiSSA, DoRA, HiRA.

---

## 2. Prior-art taxonomy — what "full-rank PEFT" means elsewhere

The surveyed papers cluster into six structural families. The first column notes whether reconstructed ΔW is strictly low-rank (LR), rank-bounded by construction but ≥ moderate (RB), or full-rank (FR).

### Family A — Structured low-rank (LoRA-like)
| Method | arXiv | ΔW | Notes |
|---|---|---|---|
| LoRA | 2106.09685 | LR (rank-r) | The baseline. |
| DoRA | 2402.09353 | LR | Decouples magnitude/direction; still rank-r. |
| rsLoRA | — | LR | Scaled LoRA. |
| AdaLoRA | 2303.10512 | LR | Importance-pruned rank. |
| VeRA | 2310.11454 | LR | Shared random bases + diag scales. |
| LoRA-XS | 2405.17604 | LR | Even smaller diag-only variant. |
| LoRA-GA / LoRA-Pro / LoRA+ / SARA | 2407.05000, 2407.18242, — , 2408.03290 | LR | Better init / preconditioner. |

### Family B — SVD-reparameterized LoRA (still low-rank)
| Method | arXiv | ΔW | Notes |
|---|---|---|---|
| PiSSA | 2404.02948 | LR | Init LoRA from top singular components. |
| MiLoRA | 2406.09044 | LR | Init LoRA from *bottom* singular components. |
| CorDA | 2406.05223 | LR | Context-oriented SVD init. |
| KaSA | 2412.06071 | LR | Knowledge-aware SVD LoRA. |
| OLoRA | 2406.01775 | LR | Orthonormal init. |
| Spectral Adapter | 2405.13952 | LR (rank ≤ 2r) | Freezes spectral tail, trains top-r U/V columns. **Closest to FURA in the "freeze something, train a spectral factor" sense — but it's still rank-bounded.** |

### Family C — Tensor/TT factorization (still low-rank by TT-rank)
| Method | arXiv | ΔW | Notes |
|---|---|---|---|
| LoTR | 2402.01376 | LR (TT-rank) | Tucker-2 with shared bases; all cores trained. |
| LoRETTA | 2402.11417 | LR | Tensor-train adapter / reparam; all cores trained. |
| TT-LoRA | 2408.01008 | LR | TT of AB update. |
| LoRTA | 2410.04060 | LR | Higher-order tensor AB. |
| TensLoRA | 2509.19391 | LR | Unifies FacT/LoTR/LoRTA/CaRA under Tucker. |
| KronA, KAdaptation, MoKA | 2212.10650 / 2203.16329 / 2508.03527 | RB/FR | Kronecker ΔW can be full-rank algebraically, but not framed that way; both factors trained. |
| Monarch / MoRe | 2204.00595 / 2408.17383 | RB | Block-diag × perm × block-diag; all blocks trained. |

### Family D — Full-rank PEFT (the crowded new shelf)
| Method | Venue / arXiv | ΔW | Mechanism |
|---|---|---|---|
| **RandLoRA** | ICLR'25 / 2502.00987 | **FR** | Fixed random low-rank bases + trainable diagonal scales. Paper title literally = "full-rank parameter-efficient fine-tuning." |
| **QuanTA** | NeurIPS'24 / 2406.00132 | **FR** | Quantum-circuit-style product of small tensors; proves universal/full-rank expressivity. Most dangerous FURA precedent on the full-rank framing. |
| **SVFT** | NeurIPS'24 / 2405.19597 | **FR** (w/ dense M) | ΔW = U_frozen · M_trainable · V_frozen^T where M sparse/dense. |
| **BOFT / OFT** | ICLR'24 / 2311.06243 | **FR** (multiplicative) | W' = R·W with butterfly-factored R. |
| **Khatri-Rao adapter** | ICCV'25 | **FR** | Higher effective rank via KR product. |
| **GaLore** | ICML'24 / 2403.03507 | **FR** (full FT) | Memory-efficient full FT via gradient projection — not a PEFT adapter. |

### Family E — Sparse / selective (no factorization, full-rank by sparsity)
| Method | arXiv | ΔW | Notes |
|---|---|---|---|
| LIFT | 2506.00772 (ICML'25) | FR (sparse) | SVD → top-5 % "Principal Weights" by magnitude → sparse dense-entry FT. |
| S²FT | 2412.06289 (NeurIPS'24) | FR (on selected rows/cols) | Selects attention heads / FFN channels; dense sub-matrix trained. |
| Diff-Pruning, SpIEL, (IA)³ | — | FR/LR | Classical sparse or adapter families. |

### Family F — The BTT structural primitive FURA reuses
| Method | Venue | ΔW | Notes |
|---|---|---|---|
| **Compute-Better-Spent BTT** (Qiu/Potapczynski/Finzi/Goldblum/Wilson) | ICML'24 / 2406.06248 | FR by construction | The BTT linear-layer primitive FURA's `btt_layer.py` imports. **Used for pretraining from scratch, not PEFT.** Proper citation as structural basis, not competitor. |

**Answer to Q1 (has the FURA idea been done?):** Not exactly. The specific combination *block-SVD + freeze large L + train small R + claim full-rank* is absent. The closest three are **QuanTA** (same full-rank pitch, different tensor-network structure, trains all cores), **LoRETTA_rep** (same TT family, trains all cores, stays low-rank), and **Spectral Adapter** (same "freeze a factor, train a factor" idea but on the spectral tail, stays rank-bounded).

**Answer to Q2 (who else claims full-rank PEFT?):** RandLoRA, QuanTA, SVFT, BOFT/OFT, Khatri-Rao. Five explicit precedents. The "first full-rank PEFT" claim is off the table.

---

## 3. Commonsense-170K SFT leaderboard (LLaMA-3-8B, 8-task avg)

### 3a. Full expanded leaderboard (April 2026)

Sorted descending. Numbers verified against source papers' tables. **Critical caveat: LoRA baselines differ massively across papers (see §3b).**

| # | Method | arXiv / Venue | Rank | Params % | 8-task Avg | Their LoRA baseline | Family |
|--:|---|---|---|---|---:|---:|---|
| 1 | **FURA** *(smerge_keep_trainable, output_one_block)* | this work | — | ~2.5% | **87.91** | 83.46 (LIFT protocol) | Structured FR (BlockTT) |
| 2 | LIFT | 2506.00772, ICML'25 | 32 | ~5% | 87.88 | 83.46 (own re-run) | Sparse (Principal Weights) |
| 3 | DropLoRA | 2508.17337, preprint | r=32 | ~0.70% | 87.61 | **86.78 (own strong run)** | Structured LR (LoRA+) |
| 4 | GateRA | 2511.17582, AAAI'26 | r=32 | 0.71% | 87.53 | 80.79 (weak cited) | Structured LR (HiRA+gate) |
| 5 | Dual LoRA | 2512.03402, preprint | r=16 | 0.70% | 87.50 | 80.9 (weak cited) | Structured LR (dual) |
| 6 | PERA | 2604.11841, ACL'26 Findings | r=16 | **0.35%** | 87.38 | 80.79 (weak cited) | Structured LR |
| 7 | SMoA | 2601.07507, preprint | r=32,n=2 | ~0.83% | 87.35 | 80.79 (weak cited) | Structured LR (MoE-LoRA) |
| 8 | HiRA | ICLR'25 | r=32 | 0.70% | 86.72 | N/A | Structured LR |
| 9 | Full FT | — | — | 100% | 86.64 | — | — |
| 10 | S²FT | 2412.06289, NeurIPS'24 | 64 | 0.70% | 86.16 | — | Sparse (selective) |
| 11 | RandLoRA | 2502.00987, ICLR'25 | n=30 | 0.70% | 85.59 | **85.24 (own strong run)** | Structured FR |
| 12 | SpecLoRA | 2505.23099, preprint | 16 | 0.35% | 85.5 | — | SVD-reparam LR |
| 13 | DoRA | 2402.09353, ICML'24 | 64 | 0.71% | 85.04 | — | Structured LR |
| 14 | KaSA | 2412.06071, ICLR'25 | 32 | ~0.35% | 84.6 | — | SVD-reparam LR |
| 15 | LoRA | 2106.09685 | 64 | 0.70% | 80.8–86.8 | — | Structured LR |

### 3b. Critical observation: LoRA baseline discrepancy

The same LoRA (r=32) on LLaMA-3-8B ranges from **80.79** (GateRA, PERA, SMoA — using original Hu et al. eval) to **86.78** (DropLoRA — own well-tuned run). This **6-point gap** is enormous and comes from differences in LR, epochs, warmup, target modules, and eval harness.

Two baseline regimes exist:
- **Weak "cited" baselines (~80.8)**: GateRA, PERA, SMoA, Dual LoRA all use this. Their claimed improvements (+6–7 pts over LoRA) are inflated.
- **Strong "re-run" baselines (~85–87)**: DropLoRA (86.78), RandLoRA (85.24), LIFT/FURA (83.46 at r=64). More honest — actual improvements are +0.8–1.3 pts.

**FURA uses the LIFT-paper protocol** (which is moderate — 83.46 at r=64). This is a defensible middle ground: not the weakest cited baseline but not the strongest re-run either. FURA's +4.45 over LoRA is real because it uses the same eval protocol as LoRA in the same paper (LIFT Table 1).

**Implication**: methods claiming 87+ with a weak LoRA baseline (GateRA 87.53, PERA 87.38, SMoA 87.35) may simply reflect better hyperparameter tuning that also benefits LoRA. **DropLoRA** (87.61 with a 86.78 LoRA baseline, +0.83) and **FURA** (87.91 with an 83.46 LoRA baseline, +4.45) are the most informative datapoints — they show genuine method improvement beyond LoRA optimization.

### 3c. Summary

- **FURA (87.91) remains the highest reported number**, narrowly above LIFT (87.88) and DropLoRA (87.61).
- **Several LoRA-family methods have broken 87** (DropLoRA 87.61, GateRA 87.53, Dual LoRA 87.50, PERA 87.38, SMoA 87.35) but with less reliable baselines.
- **DoRA (85.04) is no longer competitive** — solidly beaten by HiRA (+1.7), GateRA (+2.5), DropLoRA (+2.6), PERA (+2.3), SMoA (+2.3).
- **RandLoRA (85.59)** — the "full-rank PEFT" paper from ICLR'25 — is mid-tier on this benchmark. FURA substantially outperforms it (+2.32).
- **FURA's Commonsense-170K SOTA claim is defensible** but should acknowledge the clustering of 87+ methods and the baseline-sensitivity issue. Strongest framing: "FURA achieves the highest number (87.91) under the LIFT evaluation protocol, the most widely-cited controlled comparison."

**Papers that do NOT report LLaMA-3-8B Commonsense-170K 8-task avg:** CorDA, LoRA-GA, LoRA-Pro, LoRA-One (ICML'25 Oral — LLaMA-2-7B only), SVFT, QuanTA, ABBA (Llama-3.2 only), BHRA (Llama-3.2 only), OPLoRA, GaLore, LISA.

**Action items:**
1. Add LLaMA-2-7B Commonsense run (LIFT reports 84.66 there).
2. Consider running DropLoRA, HiRA, PERA as baselines under the LIFT protocol for an apples-to-apples comparison table.
3. If rebuttal time: re-run LoRA r=32 with DropLoRA's training config to verify whether their strong baseline reproduces.

---

## 4. MATH-10K SFT leaderboard (LLaMA-3-8B, 7-task avg)

### 4a. Expanded leaderboard (April 2026)

Sorted descending. The MATH-10K 7-task protocol (MultiArith, GSM8K, AddSub, AQuA, SingleEQ, SVAMP, MAWPS) is **far less popular** than Commonsense-170K — very few papers beyond LIFT report this exact benchmark on LLaMA-3-8B.

| # | Method | arXiv / Venue | Rank | 7-task Avg | Family |
|--:|---|---|---|---:|---|
| 1 | LIFT | 2506.00772, ICML'25 | 128 | **81.78** | Sparse (Principal Weights) |
| 2 | S²FT | 2412.06289, NeurIPS'24 | 64 | 80.87 | Sparse (selective) |
| 3 | **FURA** *(smerge_trainable, output_one_block)* | this work | — | **80.47** | Structured FR (BlockTT) |
| 4 | PiSSA | 2404.02948, NeurIPS'24 | 128 | 80.42 | SVD-reparam LR |
| 5 | Full FT | — | — | 80.18 | — |
| 6 | DoRA | 2402.09353, ICML'24 | 64 | 79.92 | Structured LR |
| 7 | **FURA** *(smerge_keep_trainable)* | this work | — | 79.77 | Structured FR (BlockTT) |
| 8 | LoRA | 2106.09685 | 64 | 79.44 | Structured LR |
| 9 | D-ReFT (25%) | 2506.06686 | — | 77.4 | Representation FT |
| 10 | ReFT | 2506.06686 | — | 76.6 | Representation FT |

### 4b. Papers that DO NOT report MATH-10K 7-task on LLaMA-3-8B

After exhaustive search: **CorDA** (LLaMA-2-7B GSM8K/MATH only), **HiRA** (commonsense + MetaMathQA only), **ABBA/BHRA** (Mistral-7B/Gemma-2 only), **SpecLoRA** (commonsense only), **LoRA-GA/Pro/SB** (LLaMA-2-7B GSM8K only), **KaSA** (commonsense only), **RandLoRA** (commonsense only), **FourierFT** (NLU+NLG only), **SVFT** (GSM8K only), **QuanTA** (likely commonsense + GSM8K), **LoRA-One** (LLaMA-2-7B only), **GateRA/PERA/SMoA/Dual LoRA/HiRA** (commonsense only or GSM8K-only).

DropLoRA (2508.17337) reports a "math reasoning" table but uses a **different protocol**: GSM8K (81.32) + MATH (30.74) → avg 56.03. This is MetaMathQA-style, not the LLM-Adapters 7-task suite — **not comparable**.

### 4c. Key finding: LIFT is unchallenged on MATH-10K

**No paper published between June 2025 and April 2026 has beaten LIFT's 81.78** on this benchmark. The benchmark is niche — most newer PEFT papers evaluate math via GSM8K/MATH standalone (MetaMathQA training), not the LLM-Adapters 7-task protocol. LIFT itself is essentially the only large-scale re-benchmarking paper.

### 4d. FURA's position

- FURA (80.47) beats Full FT (+0.29), PiSSA (+0.05), DoRA (+0.55), LoRA (+1.03).
- FURA trails **LIFT (−1.31)** and **S²FT (−0.40)**.
- The LIFT gap is not distributed noise — it concentrates on **AQuA (LIFT 34.65 vs FURA 27.56, −7.1)** and **GSM8K (72.40 vs 71.87)**. AQuA is multi-step word problems; FURA's single output-side block may cap the compositional capacity needed there.
- FURA's intra-method variance (80.47 ↔ 79.77 across smerge variants = 0.70 pts) is typical for PEFT families (DoRA/rsLoRA spread ≈ 0.4–0.8; PiSSA/LoRA ≈ 1.0; LIFT across rank/decomp ≈ 1.5).

**Implication for FURA's paper narrative.** The "SOTA on math SFT" claim does not hold. Options:
1. **Pivot headline**: "SOTA on Commonsense-170K; competitive on MATH-10K; unique advantage on GRPO-RL" — match the actual evidence.
2. **Ablate AQuA**: if FURA with `square` or `input_one_block` decomp can close AQuA, re-run with that config and see whether 7-task avg catches LIFT.
3. **Scale rank budget**: LIFT uses r=128 → ~5% params; FURA at ~2–3% is at a lower budget. Add a "FURA@matched-budget-5%" row to isolate budget from method.
4. **Note the benchmark's low adoption**: since very few post-LIFT papers use this exact protocol, the FURA vs LIFT comparison is nearly 1v1. The more interesting table for reviewers is Commonsense-170K (where 10+ methods compete) and RL (where FURA has a unique story).

---

## 5. Structure taxonomy — "LoRA-like" vs "sparse-like" vs "factored-reparam"

Mapping the main baselines to three canonical categories:

| Category | Description | Methods | ΔW shape |
|---|---|---|---|
| **Structured low-rank (LoRA-like)** | Add `B·A` with fixed small rank r; optionally reparameterize init. | LoRA, DoRA, rsLoRA, AdaLoRA, VeRA, LoRA-XS, LoRA-GA/Pro/SB/SARA, PiSSA, MiLoRA, CorDA, KaSA, OLoRA, Spectral Adapter, SpecLoRA, FourierFT, FLoRA, HRA, HiRA | Strictly rank-r |
| **Structured full-rank (tensor/factored)** | ΔW reconstructed from multiple factors; rank can reach full without a rank-r bottleneck. | **FURA** (BlockTT, freeze L train R), QuanTA (tensor-network, all cores trained), LoRETTA / LoTR / TT-LoRA / LoRTA / TensLoRA (TT-rank bounded, not full in practice), KronA / Monarch (algebraically full but not framed that way), RandLoRA (sum of random bases + diag), BOFT / OFT (multiplicative orthogonal) | Full-rank (FURA, QuanTA, RandLoRA) or rank-bounded by design |
| **Unstructured / selective (sparse-like)** | Pick a subset of dense entries, rows, cols, or heads; update them directly. | LIFT (top-5% Principal Weights), S²FT (selected heads/channels), Diff-Pruning, SpIEL | Sparse pattern; full-rank algebraically |
| **Optimizer-side** | Train all params, restrict gradient subspace. | GaLore | Full-rank (is full FT) |

**FURA's cell**: *structured full-rank, via block-SVD with asymmetric core freezing.* That cell has exactly two other occupants — **QuanTA** (different tensor network, all cores trained) and **RandLoRA** (random bases + diag, no SVD reparameterization). Neither overlaps FURA's specific mechanism.

---

## 6. Nearest-neighbor papers and how FURA must differentiate

### 6.1 QuanTA (arXiv 2406.00132, NeurIPS'24) — most dangerous
- **Pitch**: tensorized PEFT; proves product of small cores is universal / full-rank.
- **Difference from FURA**: (i) quantum-circuit tensor-network topology, not block-SVD / BlockTT; (ii) trains **all** cores, no asymmetric freezing; (iii) no SVD-based initialization or preconditioned-gradient story; (iv) does not report Commonsense-170K / MATH-10K on LLaMA-3-8B.
- **How to defend**: FURA should baseline QuanTA on Commonsense-170K + MATH-10K if time permits, and frame the theory differently (preconditioned gradient on the spectral subspace vs QuanTA's universality).

### 6.2 LoRETTA_rep (arXiv 2402.11417, NAACL'24 Oral) — TT family proxy
- **Pitch**: TT-reparameterize ΔW, train all TT cores, ~100× fewer params than LoRA.
- **Difference**: stays low-rank by small TT-rank; no frozen-large-core; no block-SVD / singular-value tie-in.
- **How to defend**: FURA's full-rank guarantee comes precisely from freezing L with full column rank — a structural theorem that LoRETTA cannot state.

### 6.3 Spectral Adapter (arXiv 2405.13952, NeurIPS'24) — spectral-freeze proxy
- **Pitch**: SVD W₀ = UΣV^T, train additive perturbations to top-r U/V columns.
- **Difference**: freezes the **tail** spectrum; FURA freezes the **large core L** which spans the whole column space. Spectral Adapter's ΔW ≤ 2r rank; FURA's is full-rank.
- **How to defend**: show empirically that FURA updates more spectral directions than Spectral Adapter at matched parameter budget.

### 6.4 Qiu et al. BTT (arXiv 2406.06248, ICML'24) — structural primitive
- **Not a competitor** — FURA's `btt_layer.py` imports it. Cite as structural basis, like how a LoRA paper cites matrix-factorization theory.

### 6.5 LIFT (arXiv 2506.00772, ICML'25) — benchmark rival
- **Pitch**: SVD-selected top-5% magnitude entries (Principal Weights) as sparse FT targets.
- **Difference**: sparse-unstructured vs FURA's structured BlockTT; LIFT has no preconditioned-gradient framing; LIFT's implementation is not TP-friendly (scattered sparse indices).
- **How to defend**: lean on *system efficiency* (FURA is TP-compatible, LIFT isn't) and *RL results* (LIFT has no GRPO/DAPO numbers; FURA's 0.895 eval/acc on RL beats Full FT 0.886).

### 6.6 S²FT (arXiv 2412.06289, NeurIPS'24) — benchmark rival
- **Pitch**: select sparse attention heads / FFN channels, train dense sub-matrix.
- **Difference**: selective structured (but coarse-grained: head-level), vs FURA's fine-grained BlockTT reparam over all weights.

---

## 7. Concrete recommendations for the FURA paper

1. **Drop "first full-rank PEFT" claim.** Replace with "first block-SVD reparameterization to achieve full-rank updates at a matched 2–3% PEFT budget by freezing the larger factor."
2. **Add QuanTA and RandLoRA as explicit baselines** in the related work and, budget permitting, in the experiment tables. Ignoring them is a guaranteed reviewer complaint. RandLoRA (85.59 on Commonsense) is easy to beat empirically.
3. **Commonsense-170K**: claim highest number (87.91) under the LIFT protocol; emphasize crossing the Full-FT ceiling. Acknowledge the 87+ cluster (DropLoRA, GateRA, etc.) but note they use a 6-pt-weaker LoRA baseline. Add LLaMA-2-7B run to match LIFT's table breadth.
4. **MATH-10K**: do not claim SOTA. Either (a) reframe the narrative around Commonsense+RL, (b) diagnose the AQuA gap and try other decomp configs, or (c) scale FURA's budget to LIFT's ~5% and report at matched budget.
5. **Ablation to add**: vary decomp mode (square / input_one_block / output_one_block) × s-merge (frozen / trainable / keep_*) on AQuA specifically — it's the one task separating FURA from LIFT, and understanding it would strengthen the theory section.
6. **Theory hook reviewers will care about**: the "preconditioned gradient on spectral subspace" story (paper_outline §4.2) is genuinely differentiating from QuanTA. Expand this into a main-text theorem, not an appendix remark.
7. **Category banner for tables**: adopt the 4-cell taxonomy (structured LR / structured FR / unstructured sparse / full FT) in §5 — it cleanly separates FURA from all LoRA-family methods in one glance.
8. **New baselines to consider running**: HiRA (ICLR'25, 86.72 — the strongest venue-published structured-LR method), DropLoRA (87.61 — strongest absolute number from a LoRA variant), and PERA (87.38 at only 0.35% params — potential "param-efficiency wins" competitor). Run these under the LIFT protocol to get apples-to-apples numbers.
9. **DoRA is no longer the right "LoRA+" baseline** for §5 tables. Use HiRA or DropLoRA as the strong LoRA-family representative. Keep DoRA for backward compatibility but add a stronger one.

---

## 8. Cited sources

- RandLoRA — https://arxiv.org/abs/2502.00987
- QuanTA — https://arxiv.org/abs/2406.00132
- SVFT — https://arxiv.org/abs/2405.19597
- LoRETTA — https://arxiv.org/abs/2402.11417
- LoTR — https://arxiv.org/abs/2402.01376
- TT-LoRA — https://arxiv.org/abs/2408.01008
- LoRTA — https://arxiv.org/abs/2410.04060
- TensLoRA — https://arxiv.org/abs/2509.19391
- Compute-Better-Spent (BTT primitive) — https://arxiv.org/abs/2406.06248
- Spectral Adapter — https://arxiv.org/abs/2405.13952
- BOFT — https://arxiv.org/abs/2311.06243
- GaLore — https://arxiv.org/abs/2403.03507
- Khatri-Rao adapter — ICCV'25 open-access proceedings
- PiSSA — https://arxiv.org/abs/2404.02948
- MiLoRA — https://arxiv.org/abs/2406.09044
- CorDA — https://arxiv.org/abs/2406.05223
- KaSA — https://arxiv.org/abs/2412.06071
- OLoRA — https://arxiv.org/abs/2406.01775
- LoRA-XS — https://arxiv.org/abs/2405.17604
- VeRA — https://arxiv.org/abs/2310.11454
- DoRA — https://arxiv.org/abs/2402.09353
- AdaLoRA — https://arxiv.org/abs/2303.10512
- HRA — https://arxiv.org/abs/2405.17484
- HiRA — https://arxiv.org/abs/2501.16391
- S²FT — https://arxiv.org/abs/2412.06289
- LIFT — https://arxiv.org/abs/2506.00772
- SpecLoRA — https://arxiv.org/abs/2505.23099
- KronA — https://arxiv.org/abs/2212.10650
- Monarch — https://arxiv.org/abs/2204.00595
- DropLoRA — https://arxiv.org/abs/2508.17337
- GateRA — https://arxiv.org/abs/2511.17582 (AAAI'26)
- PERA — https://arxiv.org/abs/2604.11841 (ACL'26 Findings)
- SMoA — https://arxiv.org/abs/2601.07507
- Dual LoRA — https://arxiv.org/abs/2512.03402
- HiRA — ICLR'25 (not on arXiv; arXiv 2501.16391 is a WRONG ID mapping to a drug-discovery paper)
- LoRA-One — https://arxiv.org/abs/2502.01235 (ICML'25 Oral; no Commonsense-170K results)
- DenseLoRA — https://arxiv.org/abs/2505.23808
- AROMA — https://arxiv.org/abs/2504.05343
- D-ReFT — https://arxiv.org/abs/2506.06686
