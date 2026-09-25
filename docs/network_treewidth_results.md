# Network games: exact, partly exact, or sampled?

A follow-up to [`deep_dive_partly_solvable.md`](deep_dive_partly_solvable.md), Section 3. It tests the structured control-variate idea on the network types that `NetworkedGame` uses: Krebs, Zerkani, random, small-world and scale-free.

Produced by [`network_treewidth_analysis.py`](network_treewidth_analysis.py).

## Summary

1. **Krebs is fully solvable, and has already been solved.** The UCINET version of the 9/11 network (60 people, 126 ties) has treewidth **exactly 5**: the upper and lower bounds agree. van der Zanden, Bodlaender & Hamers (*Operational Research*, 2023) computed exact Shapley values for the 69-member 9/11 network with a treewidth algorithm. Before that it could only be approximated, including by the structured sampling method in this repo. **For Krebs, sampling is no longer needed. Its exact values should be used as ground truth.**
2. **The repo's game reduces to a counting problem.** For `v(S) = f(|S|)` when `G[S]` is connected (and 0 otherwise), the Shapley value depends only on the number of connected coalitions of each size, with and without player `i` (lemma below). Any method that counts connected induced subgraphs by size, such as treewidth dynamic programming, gives the Shapley value exactly.
3. **The control-variate idea only matters where treewidth is too high for exact computation.** In the ~60-node stand-ins, a surrogate of treewidth 6 cuts sampling variance **up to 11×** for the scale-free graph and 2× for small-world. But these stand-ins still have treewidth ≤ 7, so they may be exactly solvable as well. The random stand-in has treewidth between 7 and 12, the only case here where exact computation may be out of reach.
4. **Where "partly solvable" really applies:** large or random-like networks. Their treewidth grows with size, so exact computation fails, and a surrogate is only useful if it keeps most of the edges.

## The counting lemma

Let `c(k)` be the number of connected coalitions of size `k`, and `c_i(k)` the number of those that contain player `i`. For the game `v(S) = f(|S|) · 1[G[S] connected]`:

```
φ_i = Σ_{k=1..n} f(k) · [ w(k−1) · c_i(k)  −  w(k) · (c(k) − c_i(k)) ]      with w(s) = s!(n−s−1)!/n!
```

(the second term is 0 for `k = n`).

*Why:* in `φ_i = Σ_{S∌i} w(|S|)[v(S∪{i}) − v(S)]`, the first term is nonzero only when `S ∪ {i}` is connected, and the second only when `S` is. Grouping the connected coalitions by size gives the formula.

Checked against brute force on a 10-node random graph: maximum difference 5·10⁻¹⁵.

This holds for **any** payoff `f` of coalition size, so `|S|² − |S|` is covered. The Myerson variant (sum of `f` over components) needs more. Since `Σ_C |C|(|C|−1)` counts the ordered pairs of players connected within `S`, the Myerson game is a sum of pairwise "u and v are connected within S" games, one for each pair. That is `n²` smaller counting problems. It's a possible route, not something I've verified.

## Results

Setup:
- Each graph: treewidth bounds, then a greedy deletion of the most redundant edges until the heuristic width reaches `t`, giving the surrogate `H`.
- **Variance ratio:** permutation-sampling variance of the residual `v_G − β v_H`, summed over players, divided by that of `v_G`. Estimated from 2,000 random permutations. Values below 1 are gains; 0.10 means a 10× reduction.
- **DP states per bag:** `Bell(t+2)`, my rough count of the states a connectivity dynamic program tracks per bag (subset of the bag in `S` × partition into components), before polynomial factors. Use it to compare widths, not as a runtime prediction.

### Krebs 9/11 network (UCINET matrix, 60 nodes, 126 edges): treewidth exactly 5

| Target width `t` | Edges kept | States per bag | Connected game, β=1 | Connected game, best β | Myerson, β=1 | Myerson, best β |
|---|---|---|---|---|---|---|
| 1 | 43% | 5 | 1.00 | 1.00 | 1.40 | 0.84 |
| 2 | 45% | 15 | 1.00 | 1.00 | 1.27 | 0.83 |
| 3 | 80% | 52 | 0.25 | 0.17 | 0.70 | 0.37 |
| 4 | 91% | 203 | 0.02 | 0.02 | 0.10 | 0.08 |
| **5 = G** | 100% | 877 | exact | exact | exact | exact |

The mechanism works: at `t = 4`, deleting 11 edges leaves a surrogate that removes 98% of the variance. But at `t = 5` the whole graph is solvable with 877 states per bag, so there's nothing left to estimate.

### Random stand-in (Erdős–Rényi, 60 nodes, mean degree 3.7, 3 components): treewidth between 7 and 12

| `t` | Edges kept | States per bag | Connected, best β | Myerson, β=1 | Myerson, best β |
|---|---|---|---|---|---|
| 4 | 60% | 203 | 0.67 | 4.25 | 0.80 |
| 6 | 68% | 4,140 | 0.39 | 2.28 | 0.71 |
| 8 | 74% | 115,975 | 0.24 | 1.37 | 0.59 |
| 12 = G | 100% | 1.9·10⁸ | exact | exact | exact |

### Small-world stand-in (Watts–Strogatz, 60 nodes, k = 4, p = 0.1): treewidth between 4 and 7

| `t` | Edges kept | States per bag | Connected, β=1 | Connected, best β | Myerson, best β |
|---|---|---|---|---|---|
| 4 | 59% | 203 | 3.01 | 0.99 | 0.92 |
| 6 | 92% | 4,140 | 0.99 | 0.59 | 0.49 |
| 7 = G | 100% | 21,147 | exact | exact | exact |

### Scale-free stand-in (Barabási–Albert, 60 nodes, m = 2): treewidth between 6 and 7

| `t` | Edges kept | States per bag | Connected, β=1 | Connected, best β | Myerson, best β |
|---|---|---|---|---|---|
| 4 | 63% | 203 | 2.00 | 0.89 | 0.78 |
| 6 | 91% | 4,140 | 0.11 | **0.09** | 0.20 |
| 7 = G | 100% | 21,147 | exact | exact | exact |

### Zerkani (47 members): not analysed

Its edge list isn't publicly reachable from here. Run the script on the repo's own file, `data/networks/zerkani/zerkani.csv`; see "How to run".

## What the numbers say

1. **The surrogate has to be close to the graph.** Deleting half the edges to get a tree (`t = 1`) gives almost nothing (0.84–1.00). The large gains (Krebs `t = 4`: 0.02; scale-free `t = 6`: 0.09) come from surrogates that keep 90% or more of the edges.
2. **The connected game is harsher than the Myerson game.** Its payoff requires the entire coalition to be connected, so a surrogate missing a few bridges disagrees on almost all large coalitions. Myerson sums over components, so it degrades more gracefully (0.8 instead of 1.0 for tree surrogates).
3. **Plain subtraction is dangerous.** With `β = 1`, variance went up by as much as **10×** (random, Myerson, `t = 1`). An estimated `β` is essential.
4. **At this size, exact computation beats estimation.** Every graph except the random one has treewidth ≤ 7. If the dynamic program is implemented efficiently, the answer for the repo's networks at ~60 nodes is *solve exactly*, not *estimate*.

## Where the idea still has room

- **Larger or random-like networks.** For sparse random graphs above the connectivity threshold, treewidth grows linearly with `n` (known results for Erdős–Rényi graphs; check the literature for scale-free models). Exact computation then fails as `n` grows, and a close low-width surrogate is the only structured option. Whether one exists with high correlation at `n` in the hundreds is an open, measurable question.
- **Core–periphery networks.** Many real networks have a tree-like periphery around a small dense core. A decomposition that solves the periphery exactly and samples only coalitions that involve the core fits "partly solvable" exactly. Experimental work on the treewidth of real graphs (e.g. Maniu, Senellart & Jog, ICDT 2019) reports this structure; check it before relying on it.
- **Benchmark value.** Exact values for Krebs, and probably Zerkani and the ~60-node stand-ins, give the network games **exact ground truth**. That turns them into a proper benchmark for the research question: realistic, not closed-form, and exactly solvable.

## Consequences for this repository

1. **Implement the exact counting dynamic program**, or reuse van der Zanden et al.'s algorithm if code is available. Use it to generate ground truth for `networkedgame`, replacing long reference runs.
2. **Reframe the network part of the thesis.** Sampling for Krebs is solved; the network games become test beds with exact truth. The open question moves to networks whose treewidth is out of reach.
3. **Run the script on the repo's actual network files.** The stand-ins match Krebs's size and mean degree, but the real `random`, `smallworld` and `scalefree` files may differ.

## How to run

```bash
pip install networkx
# With the repo's data/networks/ folder present, all five networks are read from there:
python docs/network_treewidth_analysis.py
# Without it, pass the UCINET 9/11 matrix for Krebs; the other three are generated stand-ins:
python docs/network_treewidth_analysis.py --krebs-ucinet path/to/9_11_HIJACKERS_ASSOCIATES.csv
```

The UCINET matrix used here came from a public course repository mirroring the UCINET covert-networks collection. It lists 61 names with one duplicate (Tarek Maaroufi), merged here into 60 nodes. The repo's `krebs/edges.csv` and van der Zanden et al.'s 69-member version may differ.

## Caveats

- The treewidth upper bounds come from elimination heuristics and may overestimate. Krebs is exact because the lower bound matches.
- The DP cost column is a rough state count, not a benchmark. The practical limit depends on the implementation.
- Variance ratios are Monte Carlo estimates from 2,000 permutations; treat the second decimal as noise.
- The greedy edge deletion is simple. Better deletion rules could give better surrogates at the same width, which is itself part of the open problem.
