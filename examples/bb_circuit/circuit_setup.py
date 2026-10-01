"""Circuit-level memory experiment for BB codes and detector groupings for LRB-MS.

Uses ldpc.ckt_noise.make_css_code_memory_circuit (Z-basis memory, Z detectors only) with uniform
circuit noise p on every location, and converts the stim detector error model (DEM) into a
check matrix H (detectors x error mechanisms), observables matrix and priors.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse
import stim

from ldpc.ckt_noise.css_code_memory_circuit import make_css_code_memory_circuit
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices
from ldpc.lrbms_decoder import overlap_check_groups

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bb"))
from bb_codes import BBCode  # noqa: E402


class CircuitProblem:
    def __init__(self, code_name: str, p: float, rounds: int):
        code = BBCode(code_name)
        self.code, self.p, self.rounds = code, p, rounds
        self.circuit = make_css_code_memory_circuit(
            x_stabilizers=code.hx, z_stabilizers=code.hz,
            x_logicals=scipy.sparse.csr_matrix(code.lx), z_logicals=scipy.sparse.csr_matrix(code.lz),
            num_rounds=rounds, basis="Z",
            after_clifford_depolarization=p, before_round_data_depolarization=p,
            before_measure_flip_probability=p, after_reset_flip_probability=p,
            idle_during_clifford_depolarization=p,
            include_opposite_basis_detectors=False,
            detector_coord_func=lambda q: [q, 0],
            shift_coords_per_round=[0, 1],
        )
        dem = self.circuit.detector_error_model(decompose_errors=False)
        mats = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
        self.H = scipy.sparse.csr_matrix(mats.check_matrix).astype(np.uint8)
        self.O = scipy.sparse.csr_matrix(mats.observables_matrix).astype(np.uint8)
        self.priors = np.asarray(mats.priors, dtype=float)
        # detector -> (round, stabilizer) from coordinates
        # detector -> (Z stabilizer row, time layer). The coordinate's first entry is the measure
        # qubit; detectors come in time order, so a stabilizer's k-th detector is time layer k
        # (layer `rounds` is the final data-qubit measurement).
        coords = self.circuit.get_detector_coordinates()
        q = np.array([int(coords[d][0]) for d in range(self.H.shape[0])])
        self.det_stab = q - q.min()
        seen = {}
        self.det_round = np.empty_like(q)
        for d, st in enumerate(self.det_stab):
            self.det_round[d] = seen.get(st, 0)
            seen[st] = self.det_round[d] + 1

    # ------------------------------------------------------------- groupings
    def groups_time(self, span: int):
        """Each stabilizer's detectors in blocks of `span` consecutive rounds."""
        groups = {}
        for d, (s, r) in enumerate(zip(self.det_stab, self.det_round)):
            groups.setdefault((s, r // span), []).append(d)
        return list(groups.values())

    def groups_space(self, ell: int, span: int = 1):
        """Greedy overlap grouping of the code's Z checks (size ell), repeated over blocks of
        `span` rounds: one group = those stabilizers' detectors in the block."""
        spatial = overlap_check_groups(self.code.hz, ell)
        by_stab = {}
        for gi, g in enumerate(spatial):
            for s in g:
                by_stab[s] = gi
        groups = {}
        for d, (s, r) in enumerate(zip(self.det_stab, self.det_round)):
            groups.setdefault((by_stab[s], r // span), []).append(d)
        return list(groups.values())

    def groups_greedy(self, ell: int):
        return overlap_check_groups(self.H, ell)

    def resolve(self, spec):
        if spec is None:
            return None
        kind = spec[0]
        if kind == "time":
            return self.groups_time(spec[1])
        if kind == "space":
            return self.groups_space(spec[1], spec[2] if len(spec) > 2 else 1)
        if kind == "greedy":
            return self.groups_greedy(spec[1])
        if kind == "greedyperm":  # greedy grouping grown from a random row order
            from ldpc.lrbms_decoder import permuted_overlap_groupings
            return permuted_overlap_groupings(self.H, spec[1], spec[2] + 1, seed=spec[3] if len(spec) > 3 else 0)[-1]
        raise ValueError(spec)
