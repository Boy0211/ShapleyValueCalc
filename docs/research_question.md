# Research question

## Main question

> **Under which properties of a cooperative game, and under which cost model, do sampling-design choices (stratification, variance-based allocation, complementary pairing and exact enumeration of border strata) improve Shapley value estimation beyond what a fitted surrogate model already achieves?**

In one sentence: *when do the classic sampling techniques still matter now that surrogate-based estimators exist?*

## Why this question

Shapley value estimation is split between two families that rarely compare against each other:

| Family | Representative methods | Typically tested on |
|---|---|---|
| **A. Pure sampling.** Reduces variance through stratification and allocation. | Castro (2009, 2017), Maleki (2013), Burgess & Chapman (2021), Zhang et al. (2023), Stratified SVARM (2024), van Campen et al. (2018) | Classic OR games (airport, voting, shoes), network games |
| **B. Sampling plus a surrogate.** Fits a model `g` of the game, computes Shapley(`g`) exactly and samples only `v − g`. | KernelSHAP (2017), Leverage SHAP (ICLR 2025), regression-adjusted MC (NeurIPS 2025), PolySHAP (ICLR 2026), Adalina (ICML 2026), EASE (preprint 2026) | ML feature attribution, data valuation, synthetic SOU games |

The existing reviews and benchmarks each stay inside one community:
- Chen et al., *Nat. Mach. Intell.* 2023
- SVBench, *VLDB* 2025
- Gupte & Paparrizos, *SIGMOD* 2025
- shapiq, NeurIPS 2024 D&B

None of them puts OR, network and ML games side by side, includes the strongest methods of both families, and counts the budget the same way for every method.

The recent unifying frameworks (Chen et al. NeurIPS 2025, Adalina, EASE) treat estimators as combinations of design choices. So the useful question is not *which named algorithm is best* but *which design choices matter for which kind of game*.

## Sub-questions

**SQ1. Describing the games.** Which measurable properties of a game predict how accurate each estimator will be?
- Additivity: the R² of a linear model fitted to `v`. More generally, the fraction of variance that an exactly solvable surrogate class explains (see [`deep_dive_partly_solvable.md`](deep_dive_partly_solvable.md)).
- Variance profile across coalition sizes.
- Heterogeneity between strata.
- Shape: whether the game is supermodular or submodular, which makes the stratum means monotone.
- Noise: how much `v(S)` varies when the same `S` is evaluated twice.

**SQ2. Isolating the design choices.** Within one framework, what is the separate effect of each choice on error per unit of cost?
- surrogate: none / linear / size×player / structured and exactly solvable
- stratification: none / by size / by player×size / by structure
- allocation: proportional / Neyman / Bernstein
- pairing: off / complementary
- border strata: none / fixed / capped at stratum size
- sampling: with / without replacement

**SQ3. Cost model.** How does the ranking of estimators change when cost is measured in evaluations instead of wall-clock time, as the cost of one evaluation of `v` goes from microseconds (OR games) to seconds (model retraining)?

**SQ4. Method contribution.** Is an EASE-type estimator whose optimal allocation is capped at stratum size, with take-all strata and sampling without replacement, guaranteed to be at least as accurate as the uncapped version? How large is the gain in practice?

**SQ5. Solvable fraction (from the deep dive).** How much of a game's Shapley value can be computed exactly through a structured surrogate (low-treewidth graph, weighted voting, typed players), leaving only the residual to estimate? Does that fraction predict which family wins?

## Hypotheses

- **H1:** The advantage of surrogate-based estimators grows with the solvable fraction; for linear surrogates that is additivity. Below some threshold, stratified estimators (Castro Neyman, Zhang Neyman) are competitive or better.
- **H2:** Stratification and allocation help most when the variance is concentrated in a few coalition sizes (airport, voting). Near-additive ML games benefit little.
- **H3:** When cost is wall-clock time and `v` is cheap, the overhead of fitting a surrogate reverses the ranking for small `n`.
- **H4:** The capped allocation is never worse than the uncapped one. Its gain is largest at small `n` and at large budgets per player.
- **H5:** Structured, exactly solvable surrogates outperform linear surrogates on OR and network games.

## Scope

- **Target:** the Shapley value only. Semivalues are future work.
- **Players:** `n` from about 10 to 200.
- **Games, one group per ground-truth source:**
  - OR games with closed-form or dynamic-programming values (airport, weighted voting). They are framed explicitly as test beds with controlled structure, not as problems that need estimation.
  - Synthetic SOU games, with analytic values.
  - ML games from shapiq's precomputed set, and tree models with exact TreeSHAP values.
  - Network connectivity games (Krebs, Zerkani), with exact values via treewidth dynamic programming where feasible and reference values from long runs otherwise. The imprecision of those references is reported.
- **Estimators:** permutation, antithetic, structured sampling (van Campen et al.), Castro Neyman, Zhang (plain, Neyman, Bernstein), Stratified SVARM, KernelSHAP, Leverage SHAP, regression-adjusted MC, EASE, and the variants from SQ4 and SQ5.
- **Metrics:** MSE and relative error at matched evaluation counts *and* at matched wall-clock time; Spearman correlation and top-k agreement as secondary metrics.

## Expected contributions

1. The first benchmark covering both communities, with budgets counted the same way for every method.
2. A breakdown of which design choices actually matter, instead of a ranking of named algorithms.
3. A practical guide: which estimator to use, given the structure of the game and how expensive `v` is to evaluate.
4. Optionally, method contributions: the capped-allocation estimator (SQ4) and structured control variates (SQ5).

## What this project is not

- Not "method X is best". The expected answer is conditional on the game and the cost model.
- Not a claim that exact border strata are new. They are credited to Stratified SVARM and the border trick in KernelSHAP/PolySHAP; EASE has a `boundary_policy` for them too.

## Before committing

1. **Check SQ1 early.** Compute the additivity and solvable-fraction measures for every game. If they all come out close to 1, or all close to 0, the conditional story fails.
2. **Check SQ4 and SQ5 for novelty** against the final versions of EASE, Adalina and regression-adjusted MC.
3. **Fix this repository first.**
   - Budget counting differs between methods: `Stratified` counts one marginal contribution (2 calls to `v`) as one evaluation, while `Simple` counts each call.
   - `StratifiedNeymanExact.run` uses `and` where it should use `or`.
   - The Bernstein denominator is `(3 * n-1)`.
   - `np.var(..., ddof=1)` returns NaN for small pilots.
