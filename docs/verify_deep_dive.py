"""
Numerical checks for the claims in docs/deep_dive_partly_solvable.md.

Every check computes Shapley values exactly by enumerating all 2^n
coalitions of a small game (n = 8), so no sampling noise is involved.
Run with: python docs/verify_deep_dive.py
"""

import itertools
import math
import random

random.seed(1)
n = 8
N = range(n)
SUBSETS = [frozenset(c) for s in range(n + 1) for c in itertools.combinations(N, s)]


def weight(s):
    """Shapley weight of a coalition of size s that excludes player i."""
    return math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)


def shapley(v):
    phi = [0.0] * n
    for i in N:
        for S in SUBSETS:
            if i not in S:
                phi[i] += weight(len(S)) * (v[S | {i}] - v[S])
    return phi


def marginal_moments(u, g, i):
    """Shapley-weighted variance of u's marginals and covariance with g's marginals."""
    xs, ys, ws = [], [], []
    for S in SUBSETS:
        if i in S:
            continue
        ws.append(weight(len(S)))
        xs.append(u[S | {i}] - u[S])
        ys.append(g[S | {i}] - g[S])
    mx = sum(w * x for w, x in zip(ws, xs))
    my = sum(w * y for w, y in zip(ws, ys))
    var_x = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    var_y = sum(w * (y - my) ** 2 for w, y in zip(ws, ys))
    cov = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    return var_x, var_y, cov


# 1. Every coalition affects the Shapley value (no exact shortcut)
v = {S: random.random() for S in SUBSETS}
T = frozenset({0, 3, 5})
t = len(T)
before = shapley(v)
v_perturbed = dict(v)
v_perturbed[T] += 1.0
after = shapley(v_perturbed)
print("1. Sensitivity of phi to a single coalition value")
print(f"   i in T:     {after[0] - before[0]:.6f}  predicted {1 / (n * math.comb(n - 1, t - 1)):.6f}")
print(f"   j not in T: {after[1] - before[1]:.6f}  predicted {-1 / (n * math.comb(n - 1, t)):.6f}")

# 2. Per-size identities: sum_i a_i(s) = sum_i b_i(s) = n * A(s)
print("2. Per-size identities")
for s in (2, 5):
    A = sum(v[S] for S in SUBSETS if len(S) == s) / math.comb(n, s)
    sum_a = sum(sum(v[S] for S in SUBSETS if len(S) == s and i in S) / math.comb(n - 1, s - 1) for i in N)
    sum_b = sum(sum(v[S] for S in SUBSETS if len(S) == s and i not in S) / math.comb(n - 1, s) for i in N)
    print(f"   size {s}: sum a = {sum_a:.6f}, sum b = {sum_b:.6f}, n*A = {n * A:.6f}")

# 3. Submodular airport game: stratum means are monotone in coalition size
W = [1, 1, 2, 3, 3, 5, 7, 10]
airport = {S: (max(W[j] for j in S) if S else 0) for S in SUBSETS}
print("3. Airport game stratum means mu_i(s)")
for i in (4, 7):
    mu = [sum(airport[S | {i}] - airport[S] for S in SUBSETS if i not in S and len(S) == s) / math.comb(n - 1, s)
          for s in range(n)]
    monotone = all(mu[k] >= mu[k + 1] - 1e-12 for k in range(n - 1))
    print(f"   player {i}: {[round(x, 3) for x in mu]}  nonincreasing: {monotone}")


# 4. Low-treewidth graph as a control variate for a connectivity game
def connected(S, edges):
    S = set(S)
    if not S:
        return False
    stack = [next(iter(S))]
    seen = set(stack)
    while stack:
        u = stack.pop()
        for a, b in edges:
            for x, y in ((a, b), (b, a)):
                if x == u and y in S and y not in seen:
                    seen.add(y)
                    stack.append(y)
    return seen == S


def connectivity_game(edges):
    return {S: (len(S) ** 2 - len(S) if connected(S, edges) else 0) for S in SUBSETS}


cycle = {(j, (j + 1) % n) for j in N}
G = cycle | {(0, 4), (2, 6)}
vG = connectivity_game(G)
surrogates = {
    "spanning path (tw 1)": {(j, j + 1) for j in range(n - 1)},
    "cycle only (tw 2)": cycle,
    "cycle + 1 chord (tw 2)": cycle | {(0, 4)},
}
print("4. Variance of permutation sampling on the residual, relative to the original game")
for name, H in surrogates.items():
    vH = connectivity_game(H)
    phi_G, phi_H = shapley(vG), shapley(vH)
    phi_R = shapley({S: vG[S] - vH[S] for S in SUBSETS})
    assert all(abs(phi_G[i] - phi_H[i] - phi_R[i]) < 1e-9 for i in N)  # linearity
    base = naive = optimal = 0.0
    for i in N:
        var_v, var_g, cov = marginal_moments(vG, vH, i)
        base += var_v
        naive += var_v - 2 * cov + var_g           # beta = 1
        optimal += var_v - (cov ** 2 / var_g if var_g > 0 else 0)  # best beta per player
    print(f"   {name:24s} beta=1: {naive / base:.3f}   optimal beta: {optimal / base:.3f}")


# 5. Shape constraints on the airport game (n = 30, exact values via Littlechild-Owen)
def pava_nonincreasing(y, w):
    """Weighted pool-adjacent-violators for a nonincreasing sequence."""
    blocks = []  # [mean, weight, length]
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            m2, w2, l2 = blocks.pop()
            m1, w1, l1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, l1 + l2])
    out = []
    for m, _, length in blocks:
        out.extend([m] * length)
    return out


rng = random.Random(7)
n5 = 30
W5 = sorted(rng.choice(range(1, 11)) for _ in range(n5))
truth, acc = [], 0.0
for j in range(n5):
    acc += (W5[j] - (W5[j - 1] if j > 0 else 0)) / (n5 - j)
    truth.append(acc)

# 5a. Isotonic projection with equal weights preserves the mean, so it cannot change phi
noisy = [rng.random() for _ in range(n5)]
projected = pava_nonincreasing(noisy, [1] * n5)
print("5a. Isotonic projection is mean-preserving:",
      f"mean before = {sum(noisy) / n5:.6f}, after = {sum(projected) / n5:.6f}")


# 5b. Certified bounds from exactly enumerated border strata
def exact_stratum_mean(i, s):
    """E[v(S+i) - v(S)] over uniform S of size s from the other players (airport game)."""
    others = sorted(W5[p] for p in range(n5) if p != i)
    if s == 0:
        return W5[i]
    total = math.comb(n5 - 1, s)
    mean, prev_cdf = 0.0, 0.0
    for x in sorted(set(others)):
        cdf = math.comb(sum(1 for o in others if o <= x), s) / total
        mean += (cdf - prev_cdf) * max(W5[i] - x, 0)
        prev_cdf = cdf
    return mean


print("5b. Certified intervals (monotone stratum means, border sizes enumerated)")
for k in (1, 2, 3):
    cost = 2 * sum(math.comb(n5, t) for t in range(k + 1))
    widths, inside = [], True
    for i in range(n5):
        mu = [exact_stratum_mean(i, s) for s in range(n5)]
        known = sum(mu[:k]) + sum(mu[n5 - k:])
        middle = n5 - 2 * k
        lo = (known + middle * mu[n5 - k]) / n5   # nonincreasing: middle strata >= mu(n-k)
        hi = (known + middle * mu[k - 1]) / n5    # and <= mu(k-1)
        inside &= lo - 1e-9 <= truth[i] <= hi + 1e-9
        widths.append((hi - lo) / truth[i])
    print(f"   k={k}: cost {cost:5d} evaluations, truth inside all intervals: {inside}, "
          f"relative width median {sorted(widths)[n5 // 2]:.2f}, min {min(widths):.2f}, max {max(widths):.2f}")
