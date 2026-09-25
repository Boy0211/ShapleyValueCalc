"""
Treewidth analysis for the structured control-variate idea on network games.

The idea (docs/deep_dive_partly_solvable.md, Section 3): for a connectivity
game on a graph G, pick a subgraph H of G with low treewidth, compute the
Shapley value of the same game on H exactly, and sample only the residual.
This script measures, for each network used by `NetworkedGame`:

  1. bounds on the treewidth of G (heuristic upper bound, minor-min-width lower bound);
  2. how many edges must be deleted to reach treewidth <= t;
  3. how much permutation-sampling variance the surrogate on H removes, for
     both game variants in mycode/utils/game.py:
       - "shapley": v(S) = |S|^2 - |S| if G[S] is connected, else 0
       - "myerson": v(S) = sum over components C of G[S] of |C|^2 - |C|
  4. the number of DP states per bag, Bell(t + 2), as a rough cost of the exact step.

Graph sources, in order of preference:
  - the repository's own files (the paths used by NetworkedGame, under data/networks/);
  - for Krebs, the UCINET 9/11 adjacency matrix passed with --krebs-ucinet;
  - for random / small-world / scale-free, generated stand-ins with the same
    number of nodes and mean degree as Krebs (clearly labelled "stand-in").

Usage:
    python docs/network_treewidth_analysis.py [--krebs-ucinet PATH] [--perms 2000]
"""

import argparse
import csv
import math
import os
import random

import networkx as nx
from networkx.algorithms.approximation import treewidth_min_degree, treewidth_min_fill_in

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Loading graphs
# ---------------------------------------------------------------------------

def load_repo_graph(name):
    """Load a network from the paths NetworkedGame uses; None if the file is missing."""
    base = os.path.join(REPO, "data", "networks")
    if name == "Krebs":
        path = os.path.join(base, "krebs", "krebs", "edges.csv")
        cols = ("source", "target")
    elif name == "Zerkani":
        path = os.path.join(base, "zerkani", "zerkani.csv")
        cols = ("Source", "Target")
    else:
        folder = {"Random": "random", "Small-world": "smallworld", "Scale-free": "scalefree"}[name]
        path = os.path.join(base, folder, f"{folder}.edgelist")
        if not os.path.exists(path):
            return None
        return nx.read_edgelist(path, nodetype=int)
    if not os.path.exists(path):
        return None
    G = nx.Graph()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            G.add_edge(row[cols[0]], row[cols[1]])
    return G


def load_krebs_ucinet(path):
    """UCINET 9/11 associates matrix (tab-separated). Duplicate names are merged."""
    rows = list(csv.reader(open(path, newline=""), delimiter="\t"))
    names = rows[0][1:]
    G = nx.Graph()
    G.add_nodes_from(set(names))
    for r in rows[1:]:
        a = r[0]
        for b, x in zip(names, r[1:]):
            if x.strip() not in ("", "0") and a != b:
                G.add_edge(a, b)
    return G


def stand_in(kind, n, mean_degree, seed=42):
    if kind == "Random":
        return nx.gnp_random_graph(n, mean_degree / (n - 1), seed=seed)
    if kind == "Small-world":
        k = max(2, int(round(mean_degree / 2)) * 2)
        return nx.watts_strogatz_graph(n, k, 0.1, seed=seed)
    if kind == "Scale-free":
        return nx.barabasi_albert_graph(n, max(1, int(round(mean_degree / 2))), seed=seed)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Treewidth
# ---------------------------------------------------------------------------

def tw_upper(G):
    """Best of two elimination heuristics; returns (width, decomposition)."""
    a = treewidth_min_fill_in(G)
    b = treewidth_min_degree(G)
    return a if a[0] <= b[0] else b


def tw_lower_mmd(G):
    """Minor-min-width (MMD+ with min-d contraction): a valid treewidth lower bound."""
    H = nx.Graph(G)
    best = 0
    while H.number_of_nodes() > 1:
        v = min(H.nodes, key=H.degree)
        d = H.degree(v)
        best = max(best, d)
        if d == 0:
            H.remove_node(v)
            continue
        u = min(H.neighbors(v), key=H.degree)
        H = nx.contracted_nodes(H, u, v, self_loops=False)
    return best


def reduce_to_width(G, t, max_steps=10_000):
    """Greedily delete edges until the heuristic decomposition has width <= t.

    Candidates are edges inside a largest bag; the most redundant one (most
    common neighbours, so least likely to disconnect anything) is removed first.
    """
    H = nx.Graph(G)
    for _ in range(max_steps):
        width, dec = tw_upper(H)
        if width <= t:
            return H, width
        big = [set(b) for b in dec.nodes if len(b) == width + 1]
        cands = [(u, v) for u, v in H.edges if any(u in b and v in b for b in big)]
        if not cands:
            cands = list(H.edges)
        u, v = max(cands, key=lambda e: (len(set(H[e[0]]) & set(H[e[1]])), H.degree(e[0]) + H.degree(e[1])))
        H.remove_edge(u, v)
    raise RuntimeError("did not converge")


def bell(k):
    row = [1]
    for _ in range(k):
        nxt = [row[-1]]
        for x in row:
            nxt.append(nxt[-1] + x)
        row = nxt
    return row[0]


# ---------------------------------------------------------------------------
# Permutation-sampling variance with and without the surrogate
# ---------------------------------------------------------------------------

def f(k):
    return k * k - k


def marginals(order, adj, mode):
    """Marginal contribution of each player along a permutation (union-find)."""
    parent, size = {}, {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    out, comps, total, prev = {}, 0, 0, 0
    for p in order:
        parent[p], size[p] = p, 1
        comps += 1
        for q in adj[p]:
            if q in parent:
                a, b = find(p), find(q)
                if a != b:
                    total += f(size[a] + size[b]) - f(size[a]) - f(size[b])
                    if size[a] < size[b]:
                        a, b = b, a
                    parent[b] = a
                    size[a] += size[b]
                    comps -= 1
        if mode == "myerson":
            value = total
        else:
            value = f(len(parent)) if comps == 1 else 0
        out[p] = value - prev
        prev = value
    return out


def variance_ratios(G, H, mode, perms, seed=0):
    """Summed per-player sampling variance of the residual relative to v_G.

    Returns (ratio with beta = 1, ratio with the optimal beta per player).
    """
    rng = random.Random(seed)
    nodes = list(G.nodes)
    adjG = {v: list(G[v]) for v in nodes}
    adjH = {v: list(H[v]) if v in H else [] for v in nodes}
    acc = {v: [0.0] * 5 for v in nodes}  # sum x, sum y, sum x^2, sum y^2, sum xy
    for _ in range(perms):
        order = nodes[:]
        rng.shuffle(order)
        mg, mh = marginals(order, adjG, mode), marginals(order, adjH, mode)
        for v in nodes:
            x, y = mg[v], mh[v]
            a = acc[v]
            a[0] += x; a[1] += y; a[2] += x * x; a[3] += y * y; a[4] += x * y
    base = naive = best = 0.0
    for v in nodes:
        sx, sy, sxx, syy, sxy = acc[v]
        vx = sxx / perms - (sx / perms) ** 2
        vy = syy / perms - (sy / perms) ** 2
        c = sxy / perms - (sx / perms) * (sy / perms)
        base += vx
        naive += vx - 2 * c + vy
        best += vx - (c * c / vy if vy > 1e-12 else 0.0)
    return naive / base, best / base


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--krebs-ucinet", help="path to 9_11_HIJACKERS_ASSOCIATES.csv (UCINET)")
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--widths", default="1,2,3,4,6,8")
    args = ap.parse_args()
    widths = [int(w) for w in args.widths.split(",")]

    graphs = {}
    krebs = load_repo_graph("Krebs")
    if krebs is not None:
        graphs["Krebs (repo data)"] = krebs
    elif args.krebs_ucinet:
        graphs["Krebs (UCINET)"] = load_krebs_ucinet(args.krebs_ucinet)
    zerkani = load_repo_graph("Zerkani")
    if zerkani is not None:
        graphs["Zerkani (repo data)"] = zerkani
    ref = next(iter(graphs.values()), None)
    n_ref = ref.number_of_nodes() if ref is not None else 60
    deg_ref = 2 * ref.number_of_edges() / n_ref if ref is not None else 5.0
    for kind in ("Random", "Small-world", "Scale-free"):
        g = load_repo_graph(kind)
        graphs[f"{kind} (repo data)" if g is not None else f"{kind} (stand-in)"] = (
            g if g is not None else stand_in(kind, n_ref, deg_ref))

    for name, G in graphs.items():
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        n, m = G.number_of_nodes(), G.number_of_edges()
        ub, _ = tw_upper(G)
        lb = tw_lower_mmd(G)
        print(f"\n## {name}: n = {n}, m = {m}, mean degree = {2 * m / n:.2f}, "
              f"components = {nx.number_connected_components(G)}, treewidth in [{lb}, {ub}]")
        print("| target width t | edges kept | DP states per bag, Bell(t+2) | "
              "shapley: beta=1 | shapley: best beta | myerson: beta=1 | myerson: best beta |")
        print("|---|---|---|---|---|---|---|")
        for t in [w for w in widths if w < ub] + [ub]:
            H, w = reduce_to_width(G, t)
            row = [f"{t}{' (= G)' if t == ub else ''}", f"{H.number_of_edges()}/{m} ({H.number_of_edges() / m:.0%})",
                   f"{bell(w + 2):,}"]
            for mode in ("shapley", "myerson"):
                naive, best = variance_ratios(G, H, mode, args.perms)
                row += [f"{naive:.2f}", f"{best:.2f}"]
            print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
