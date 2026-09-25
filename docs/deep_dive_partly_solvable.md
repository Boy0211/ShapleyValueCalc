# Is the Shapley value partly solvable?

A deep dive into which parts of the Shapley value can be computed exactly and which parts must be estimated, and one angle that neither research community appears to have combined yet.

All numerical claims are checked by [`verify_deep_dive.py`](verify_deep_dive.py), which computes Shapley values exactly on small games.

## Summary

1. **In general, no.** For an arbitrary game given only as a black box that returns `v(S)`, an exact Shapley value requires the value of *every* coalition. This is provable in a few lines (Section 1).
2. **"Partly solvable" has a precise meaning.** Shapley values are linear in the game: `φ(v) = φ(g) + φ(v − g)` for any game `g`. If `g` is from a class whose Shapley values can be computed exactly, only the residual `v − g` needs estimating. The border trick, KernelSHAP and EASE all use special cases of this (Section 2).
3. **The angle nobody has combined:** operations research already has exact polynomial algorithms for structured games: low-treewidth connectivity games, weighted voting, airport. ML has control-variate estimators that allow any exactly solvable surrogate. Nobody has used the *game-theoretic* exactly solvable classes as surrogates for OR and network games. In a small test this cut sampling variance by 13–55% (Section 3).
4. **Two other ideas failed when tested:**
   - Shape constraints (isotonic projection, certified bounds): the projection can't change the Shapley estimate, and the bounds are valid but too wide.
   - Per-size identities: they only shift every player by the same amount.
   Both are reported in Section 4 so nobody spends time on them.

## Notation

- Players `N = {1, …, n}`; game `v : 2^N → ℝ`.
- Shapley weight for a coalition `S` of size `s` without player `i`: `w(s) = s!(n−s−1)!/n! = 1 / (n · C(n−1, s))`.
- Shapley value: `φ_i(v) = Σ_{S ⊆ N∖{i}} w(|S|) · [v(S ∪ {i}) − v(S)]`.
- Stratum mean: `μ_i(s) = E[v(S ∪ {i}) − v(S)]` over a uniformly random `S ⊆ N∖{i}` with `|S| = s`. Then `φ_i = (1/n) Σ_{s=0}^{n−1} μ_i(s)`.

## 1. Exact computation needs every coalition

**Proposition 1 (sensitivity).** For a coalition `T` with `|T| = t`:
- `∂φ_i / ∂v(T) = w(t−1) = 1 / (n · C(n−1, t−1))` if `i ∈ T`,
- `∂φ_j / ∂v(T) = −w(t) = −1 / (n · C(n−1, t))` if `j ∉ T`.

*Proof.* `v(T)` appears in `φ_i` once as `v(S ∪ {i})` with `S = T∖{i}` when `i ∈ T`, and once as `−v(S)` with `S = T` when `i ∉ T`. ∎

**Theorem 2 (no exact shortcut).** Any algorithm that returns the exact Shapley value of every game using only calls to `v` must query every nonempty coalition. If `v(∅) = 0` is not given, it must query `∅` too.

*Proof.* Take any `T` the algorithm does not query. Define `v'` equal to `v` everywhere except `v'(T) = v(T) + 1`. The algorithm sees the same answers for both games, so it returns the same output. By Proposition 1 their Shapley values differ for every player in `T`. So it is wrong on at least one of the two games. ∎

This holds for randomised algorithms that must be exact with probability 1, by the same argument applied to each run.

**What Proposition 1 explains.** Each coalition size carries the same total weight `1/n` in `φ_i`, spread over `C(n−1, t−1)` coalitions:
- **Border sizes:** few coalitions, each with a large weight. Enumerating them is cheap and removes a lot of uncertainty.
- **Middle sizes:** exponentially many coalitions, each with a tiny weight. They can't be enumerated, but a random sample of them concentrates quickly.

So the Shapley value is *exactly solvable at the borders and only statistically solvable in the middle*. Enumerating sizes `0…k` and `n−k…n` makes `2k` of the `n` strata exact at a cost of about `2 · Σ_{t≤k} C(n, t)` evaluations. This is the mathematical reason the border trick (SVARM, KernelSHAP, PolySHAP, EASE) works.

Verified numerically: check 1 in the script matches Proposition 1 to 6 decimals.

## 2. What "partly solvable" can mean

Because `φ` is linear, for any game `g` and any coefficient `β`:

```
φ(v) = β · φ(g) + φ(v − β g)
```

If `g` belongs to a class `𝒢` whose Shapley values can be computed exactly and cheaply, then `β φ(g)` is the **solved part** and `φ(v − βg)` is the **residual part** that still needs sampling. With `β` chosen per player to minimise variance, sampling the residual is never worse than sampling `v`, apart from the small cost of estimating `β`.

This gives a measurable quantity for the research question:

> **Solvable fraction of a game with respect to a class 𝒢:** the share of the sampling variance of `φ(v)` that the best `g ∈ 𝒢` removes, i.e. the best squared correlation between the marginal contributions of `v` and of `g`.

For `𝒢` = additive games, this is essentially the additivity R² from SQ1 in [`research_question.md`](research_question.md). The existing methods correspond to particular classes:

| Method | Solvable class `𝒢` |
|---|---|
| Border trick (SVARM, KernelSHAP, EASE) | Games determined by the border coalitions |
| KernelSHAP, EASE-FO, Leverage SHAP | Additive games (plus size terms) |
| EASE-SP, Castro-style stratification | Games that depend on `(player, size)` |
| PolySHAP, sparse Möbius / SPEX | Low-order or sparse interaction games |
| Regression-adjusted MC (Witter et al., NeurIPS 2025) | Any family with exactly computable values, e.g. tree models via TreeSHAP |

## 3. The missing angle: structured, exactly solvable surrogates

### Exactly solvable classes that operations research already has

| Class | Exact Shapley | Reference | Game in this repo |
|---|---|---|---|
| Airport games (maximum of weights) | Closed form, O(n log n) | Littlechild & Owen (1973) | `airportgame` |
| Weighted voting games `[q; w]` | Dynamic programming, pseudo-polynomial in the total weight | Generating functions (Mann & Shapley 1962); complexity by Matsui & Matsui (2000) | `votinggame` |
| Connectivity games on graphs of bounded treewidth | Fixed-parameter tractable in the treewidth | van der Zanden, Bodlaender & Hamers (*Operational Research*, 2023) | `networkedgame` |
| Players with few types, where `v` depends only on counts per type | Polynomial in `Π(n_t + 1)` for a fixed number of types | Combinatorial; relation-stratified sampling uses the same idea | airport/voting with repeated weights, shoes |
| `k`-additive games (interactions of order ≤ `k`) | From coalitions of size ≤ `k` | Grabisch (1997) | — |
| Tree ensembles | TreeSHAP, polynomial | Lundberg et al. (2020) | `featureevaluationgame` with a tree model |

### The gap

- **OR side.** Exact algorithms and sampling methods are used as *alternatives*: exact when the structure allows it, sampling otherwise. Hamers and co-authors have both a treewidth-exact algorithm (van der Zanden et al. 2023) and a sampling method (van Campen, Hamers, Husslage & Lindelauf 2018), but I found no work combining them.
- **ML side.** Regression-adjusted MC allows any surrogate with exactly computable Shapley values, but uses generic ML models (linear, XGBoost).
- **Nobody**, as far as the searches show, uses the *game-theoretic* exact classes above as control variates for games that are *close to* but not *inside* those classes.

### Concrete proposal for the repo's network game

The game is `v_G(S) = |S|² − |S|` if `S` is connected in `G`, else 0.

1. Choose a subgraph `H ⊆ G` with low treewidth, keeping as many edges as possible.
2. Compute `φ(v_H)` exactly with treewidth dynamic programming.
3. Estimate `φ(v_G − β v_H)` by sampling. `v_G − v_H` is nonzero only for coalitions connected in `G` but not in `H`.

Check the details before relying on this: does the van der Zanden et al. dynamic program support size-dependent payoffs like `|S|² − |S|`? Tracking `|S|` in the dynamic-programming state should be a straightforward extension, but I haven't verified it.

**Toy evidence** (check 4 in the script): an 8-node ring with 2 chords as `G`, and three low-treewidth choices of `H`. The table shows the variance of permutation sampling on the residual, relative to sampling `v_G` directly:

| Surrogate `H` | `β = 1` (plain subtraction) | Optimal `β` per player |
|---|---|---|
| Spanning path (treewidth 1) | **3.38** (worse) | 0.87 |
| Ring only (treewidth 2) | 1.49 (worse) | 0.67 |
| Ring + 1 chord (treewidth 2) | 0.73 | **0.45** |

Two lessons:
1. **Plain subtraction can triple the variance.** Estimating `β` is essential.
2. **The closer `H` is to `G`, the larger the gain.** That turns the method into a graph-sparsification question: *which edges should be deleted to reach treewidth ≤ t while keeping the surrogate as correlated as possible?* That's a clean, new research problem.

The same pattern applies to other games:
- **Voting systems** that aren't weighted voting games, such as double majorities or multiple chambers: fit the closest weighted voting game, solve it exactly by dynamic programming, and sample the residual.
- **Games with nearly identical players:** cluster players into types, solve the typed game exactly, and sample the residual.

### Why this could matter

- It gives the OR and network games a method that uses their structure, which linear surrogates can't. This supports H5 in the research question.
- It produces a measurable "solvable fraction" per game, directly usable for SQ1 and SQ5.
- It sits between the two communities: exact algorithms from one, control variates from the other.

## 4. Ideas that failed when tested

**Shape constraints.** For supermodular (convex) games, `μ_i(s)` is nondecreasing in `s`; for submodular games it is nonincreasing.
*Proof by coupling:* draw a uniform `S'` of size `s+1`, remove a uniform element to get a uniform `S ⊂ S'`, and apply supermodularity.
The airport game is submodular; check 3 confirms its stratum means are monotone.

- **Isotonic projection does not help.** Projecting estimated stratum means onto the monotone cone (pool-adjacent-violators) preserves the mean when all strata have equal weight. Since `φ_i` is exactly that mean, the estimate is unchanged (check 5a). It can only matter with unequal allocations, and then the effect is an uncertain reweighting.
- **Certified bounds are valid but loose.** With border sizes enumerated, monotonicity gives a guaranteed interval for each `φ_i`. On a 30-player airport game (check 5b) the truth always falls inside. But the median interval width is 5× the true value for `k = 2` (932 evaluations) and 2× for `k = 3` (9,052 evaluations). The only players pinned down exactly are the smallest ones. At best this is a screening tool, not an estimator.

**Per-size identities.** Let `a_i(s)` be the mean of `v(S)` over coalitions of size `s` containing `i`, and `b_i(s)` the mean over those not containing `i`. Then `Σ_i a_i(s) = Σ_i b_i(s) = n · A(s)`, where `A(s)` is the mean of `v` over all coalitions of size `s` (check 2). These are extra linear constraints beyond efficiency. But projecting the estimates onto them with equal weights shifts every player's estimate by the same amount. So they can't change rankings and add little beyond enforcing efficiency.

## 5. Known results (don't claim these)

- Exact border strata: Stratified SVARM (AAAI 2024), KernelSHAP/PolySHAP's border trick, EASE's `boundary_policy`.
- Sparse or low-order interactions recovered with few queries: the sparse Möbius transform (Kang et al., arXiv:2402.02631) and SPEX / ProxySPEX (2025).
- FPRAS for monotone supermodular games: Liben-Nowell et al. (COCOON 2012).
- Exact connectivity-game algorithms: Michalak et al. (IJCAI 2013), which enumerate connected coalitions; van der Zanden et al. (2023), treewidth.

## 6. Related note on this repository

The `Structured` class in `mycode/oldmethods.py` appears to implement the structured random permutation sampling of **van Campen, Hamers, Husslage & Lindelauf (2018)**. That paper refined Castro's sampling and reported about 30% lower error on the WTC 9/11 network. Confirm this against the paper; if it holds, the method should be cited under that name.

## 7. Next experiments, cheapest first

1. **Measure the solvable fraction** of each repo game for three surrogate classes: additive, `(player, size)`, and structured (low-treewidth or weighted-voting). No new estimator is needed; this directly tests whether SQ1/SQ5 has signal.
2. **Build the treewidth control variate** for `networkedgame`. Validate it against brute force at small `n`, then run it on the Krebs network and compare with the `Structured` method, Castro Neyman and EASE.
3. **Fit weighted voting surrogates** to non-weighted voting rules and measure the variance reduction.

## Risks

- The treewidth of real terrorist networks may stay high even after deleting a few edges; I haven't measured it for Krebs or Zerkani.
- The control-variate gain depends on how close the surrogate is to the game. The toy example shows that a poor surrogate gives little benefit, and a poor `β` makes things worse.
- The general framework of Witter et al. can be argued to cover structured surrogates already. The contribution must be framed as the OR-specific surrogate classes and the sparsification problem, not the control-variate principle itself.
- Most sources were read through abstracts and search summaries, not full papers.
