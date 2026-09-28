"""Quantum Tanner codes (Leverrier-Zemor) with vertex-ordered check matrices.

Qubits live on the faces ``f(g, a, b) = {(g,00), (ag,10), (gb,01), (agb,11)}`` of
the left-right Cayley complex of a group ``G`` with generating sets ``A, B``, so
``n = |G| * |A| * |B|``. Every vertex sees ``|A| x |B|`` faces, indexed by ``(a, b)``:

- X checks live on ``V0 = G x {00, 11}`` and are a basis of ``C_A (x) C_B``,
- Z checks live on ``V1 = G x {01, 10}`` and are a basis of ``C_A^perp (x) C_B^perp``.

The rows of ``hx`` and ``hz`` are ordered vertex by vertex, so each vertex's local
tensor code occupies ``rows_per_vertex`` consecutive rows. That is the natural
generalized-check grouping for :class:`ldpc.LrbmsDecoder`.
"""

import itertools
from dataclasses import dataclass, field
from typing import List

import numpy as np
import scipy.sparse

import ldpc.mod2


def _gf2_generator(h: np.ndarray) -> np.ndarray:
    """Generator matrix (basis of the kernel) of the code with parity-check ``h``."""
    k = ldpc.mod2.nullspace(scipy.sparse.csr_matrix(h))
    return np.asarray(k.todense() if scipy.sparse.issparse(k) else k, dtype=np.uint8) % 2


def _tensor_basis(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Rows ``u (x) v`` for all generator rows ``u`` of ``g1`` and ``v`` of ``g2``."""
    return np.array([np.outer(u, v).ravel() for u in g1 for v in g2], dtype=np.uint8)


@dataclass
class QuantumTannerCode:
    name: str
    hx: scipy.sparse.csr_matrix
    hz: scipy.sparse.csr_matrix
    rows_per_vertex_x: int
    rows_per_vertex_z: int
    group_order: int
    delta_a: int
    delta_b: int
    lx: np.ndarray = field(default=None, repr=False)  # X logicals (kernel of hz mod rowspace hx)
    lz: np.ndarray = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return self.hx.shape[1]

    @property
    def k(self) -> int:
        return self.n - ldpc.mod2.rank(self.hx) - ldpc.mod2.rank(self.hz)

    def vertex_groups(self, basis: str) -> List[List[int]]:
        """Row groups of ``hx`` (basis='x') or ``hz`` (basis='z'), one per vertex."""
        h, r = (self.hx, self.rows_per_vertex_x) if basis == "x" else (self.hz, self.rows_per_vertex_z)
        return [list(range(i, i + r)) for i in range(0, h.shape[0], r)]


def group_table(group):
    """Multiplication table, inverse table and identity index of a qldpc group."""
    els = list(group.generate())
    index = {e: i for i, e in enumerate(els)}
    mul = np.array([[index[x * y] for y in els] for x in els], dtype=np.int64)
    ident = next(i for i, e in enumerate(els) if all(mul[i, j] == j for j in range(len(els))))
    inv = np.array([int(np.where(mul[i] == ident)[0][0]) for i in range(len(els))])
    return mul, inv, ident


def random_symmetric_subset(mul, inv, ident, size, rng):
    """Random inverse-closed subset of the given size without the identity."""
    orbits = sorted({tuple(sorted({g, int(inv[g])})) for g in range(len(inv)) if g != ident})
    for _ in range(10_000):
        chosen, total = [], 0
        for o in rng.permutation(len(orbits)):
            if total + len(orbits[o]) <= size:
                chosen.extend(orbits[o])
                total += len(orbits[o])
            if total == size:
                return sorted(chosen)
    raise RuntimeError("could not build a symmetric subset of that size")


def conjugacy_units(mul, inv, ident):
    """Non-trivial conjugacy classes merged with their inverse classes (sorted lists)."""
    order = len(mul)
    seen, units = {ident}, []
    for x in range(order):
        if x in seen:
            continue
        cls = {int(mul[mul[g, x], inv[g]]) for g in range(order)}
        cls |= {int(inv[y]) for y in cls}
        seen |= cls
        units.append(sorted(cls))
    return units


def random_tnc_subsets(mul, inv, ident, size, rng):
    """Symmetric A, B of the given size drawn from disjoint conjugacy-class unions.

    No element of A is conjugate to an element of B, so the total no-conjugacy
    condition holds by construction.
    """
    units = conjugacy_units(mul, inv, ident)
    for _ in range(1000):
        side = rng.integers(0, 2, size=len(units))
        pools = [[x for u, s in zip(units, side) if s == k for x in u] for k in (0, 1)]
        if min(len(p) for p in pools) < size:
            continue
        subsets = []
        for pool in pools:
            orbits = sorted({tuple(sorted({g, int(inv[g])})) for g in pool})
            chosen = []
            for o in rng.permutation(len(orbits)):
                if len(chosen) + len(orbits[o]) <= size:
                    chosen.extend(orbits[o])
            if len(chosen) != size:
                break
            subsets.append(sorted(chosen))
        if len(subsets) == 2:
            return subsets
    raise RuntimeError("no TNC subsets found")


def satisfies_tnc(mul, A, B) -> bool:
    """Total no-conjugacy: a g != g b for all g, a, b (faces are non-degenerate)."""
    return all(mul[a, g] != mul[g, b] for g in range(len(mul)) for a in A for b in B)


def generates(mul, S, ident) -> bool:
    seen, frontier = {ident}, [ident]
    while frontier:
        g = frontier.pop()
        for s in S:
            h = int(mul[g, s])
            if h not in seen:
                seen.add(h)
                frontier.append(h)
    return len(seen) == len(mul)


def build_qtanner(mul, A, B, h_a, h_b, name="qtanner") -> QuantumTannerCode:
    """Build the quantum Tanner code from a group table, subsets and local parity checks."""
    order = len(mul)
    da, db = len(A), len(B)
    g_a, g_b = _gf2_generator(h_a), _gf2_generator(h_b)
    loc_x = _tensor_basis(g_a, g_b)                              # C_A (x) C_B
    loc_z = _tensor_basis(np.asarray(h_a) % 2, np.asarray(h_b) % 2)  # C_A^perp (x) C_B^perp

    def face(g, ia, ib):
        return (g * da + ia) * db + ib

    # local views: vertex -> face indices at positions (ia, ib), row-major
    views_x, views_z = [], []
    grid = list(itertools.product(range(da), range(db)))
    ident = next(i for i in range(order) if all(mul[i, j] == j for j in range(order)))
    inv = np.array([int(np.where(mul[i] == ident)[0][0]) for i in range(order)])
    for h in range(order):
        # (h,00): faces (h, a, b)
        views_x.append([face(h, ia, ib) for ia, ib in grid])
        # (h,11): faces (a^-1 h b^-1, a, b)
        views_x.append([face(mul[mul[inv[A[ia]], h], inv[B[ib]]], ia, ib) for ia, ib in grid])
    for h in range(order):
        # (h,10): faces (a^-1 h, a, b)
        views_z.append([face(mul[inv[A[ia]], h], ia, ib) for ia, ib in grid])
        # (h,01): faces (h b^-1, a, b)
        views_z.append([face(mul[h, inv[B[ib]]], ia, ib) for ia, ib in grid])

    def assemble(views, loc):
        rows, cols = [], []
        r = 0
        for view in views:
            view = np.asarray(view)
            for vec in loc:
                support = view[np.flatnonzero(vec)]
                rows.extend([r] * len(support))
                cols.extend(support.tolist())
                r += 1
        mat = scipy.sparse.csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                                      shape=(r, order * da * db))
        mat.data %= 2
        mat.eliminate_zeros()
        return mat

    hx, hz = assemble(views_x, loc_x), assemble(views_z, loc_z)
    assert not ((hx @ hz.T).toarray() % 2).any(), "X and Z checks do not commute"
    code = QuantumTannerCode(name, hx, hz, len(loc_x), len(loc_z), order, da, db)
    code.lx = css_logicals(hz, hx)
    code.lz = css_logicals(hx, hz)
    return code


def css_logicals(h_commute: scipy.sparse.spmatrix, h_stab: scipy.sparse.spmatrix) -> np.ndarray:
    """Basis of ker(h_commute) modulo rowspace(h_stab), as a dense uint8 array."""
    ker = ldpc.mod2.kernel(h_commute)
    ker = scipy.sparse.csr_matrix(ker)
    stab_rank = ldpc.mod2.rank(h_stab)
    stacked = scipy.sparse.vstack([h_stab, ker]).tocsr()
    pivots = np.asarray(ldpc.mod2.pivot_rows(stacked))
    logical_rows = [p - h_stab.shape[0] for p in pivots if p >= h_stab.shape[0]]
    out = ker[logical_rows].toarray().astype(np.uint8) % 2
    assert len(logical_rows) + stab_rank == ldpc.mod2.rank(stacked)
    return out
