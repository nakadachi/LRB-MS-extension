"""Circuit-level memory experiment for BB codes and detector groupings for LRB-MS.

Uses ldpc.ckt_noise.make_css_code_memory_circuit (Z-basis memory, Z detectors only) with uniform
circuit noise p on every location, and converts the stim detector error model (DEM) into a
check matrix H (detectors x error mechanisms), observables matrix and priors.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse

from ldpc.ckt_noise.css_code_memory_circuit import make_css_code_memory_circuit
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices
from ldpc.lrbms_decoder import overlap_check_groups, permuted_overlap_groupings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bb"))
from bb_codes import BBCode  # noqa: E402


class CircuitProblem:
    """Memory circuit, DEM matrices and detector coordinates of one (code, p, rounds)."""

    def __init__(self, code_name: str, p: float, rounds: int):
        code = BBCode(code_name)
        self.code, self.p, self.rounds = code, p, rounds
        self.circuit = make_css_code_memory_circuit(
            x_stabilizers=code.hx,
            z_stabilizers=code.hz,
            x_logicals=scipy.sparse.csr_matrix(code.lx),
            z_logicals=scipy.sparse.csr_matrix(code.lz),
            num_rounds=rounds,
            basis="Z",
            after_clifford_depolarization=p,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p,
            idle_during_clifford_depolarization=p,
            include_opposite_basis_detectors=False,
            detector_coord_func=lambda qubit: [qubit, 0],
            shift_coords_per_round=[0, 1],
        )
        dem = self.circuit.detector_error_model(decompose_errors=False)
        matrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
        self.check_matrix = scipy.sparse.csr_matrix(matrices.check_matrix).astype(np.uint8)
        self.observables = scipy.sparse.csr_matrix(matrices.observables_matrix).astype(np.uint8)
        self.priors = np.asarray(matrices.priors, dtype=float)
        # detector -> (Z stabilizer row, time layer). The coordinate's first entry is the measure
        # qubit; detectors come in time order, so a stabilizer's k-th detector is time layer k
        # (layer `rounds` is the final data-qubit measurement).
        coords = self.circuit.get_detector_coordinates()
        detector_count = self.check_matrix.shape[0]
        measure_qubit = np.array([int(coords[d][0]) for d in range(detector_count)])
        self.detector_stabilizer = measure_qubit - measure_qubit.min()
        layers_seen = {}
        self.detector_layer = np.empty_like(measure_qubit)
        for detector, stabilizer in enumerate(self.detector_stabilizer):
            self.detector_layer[detector] = layers_seen.get(stabilizer, 0)
            layers_seen[stabilizer] = self.detector_layer[detector] + 1

    def groups_time(self, span: int):
        """Each stabilizer's detectors in blocks of `span` consecutive rounds."""
        groups = {}
        for detector, (stabilizer, layer) in enumerate(
            zip(self.detector_stabilizer, self.detector_layer)
        ):
            groups.setdefault((stabilizer, layer // span), []).append(detector)
        return list(groups.values())

    def groups_space(self, ell: int, span: int = 1):
        """Greedy overlap grouping of the code's Z checks (size ell), repeated over blocks of
        `span` rounds: one group = those stabilizers' detectors in the block."""
        spatial_groups = overlap_check_groups(self.code.hz, ell)
        group_of_stabilizer = {}
        for group_index, group in enumerate(spatial_groups):
            for stabilizer in group:
                group_of_stabilizer[stabilizer] = group_index
        groups = {}
        for detector, (stabilizer, layer) in enumerate(
            zip(self.detector_stabilizer, self.detector_layer)
        ):
            key = (group_of_stabilizer[stabilizer], layer // span)
            groups.setdefault(key, []).append(detector)
        return list(groups.values())

    def groups_greedy(self, ell: int):
        return overlap_check_groups(self.check_matrix, ell)

    def resolve(self, spec):
        """Detector grouping for a spec: None, ("time", span), ("space", ell[, span]),
        ("greedy", ell) or ("greedyperm", ell, index[, seed])."""
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
            ell, index = spec[1], spec[2]
            seed = spec[3] if len(spec) > 3 else 0
            return permuted_overlap_groupings(self.check_matrix, ell, index + 1, seed=seed)[-1]
        raise ValueError(spec)
