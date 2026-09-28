import pathlib

import numpy as np
import sinter
import stim

from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices
from ldpc.lrbms_decoder import LrbmsDecoder


class SinterLrbmsDecoder(sinter.Decoder):
    """
    Sinter wrapper for the LRB-MS generalized-check decoder.

    Parameters
    ----------
    check_groups : Union[None, int, list]
        ``None`` (one detector per generalized check), an integer group size
        ``ell`` (greedy overlap grouping of detector rows), or explicit row groups.
    max_iter : int
        Maximum number of iterations (0 -> number of error mechanisms).
    ms_scaling_factor : float
        Scaling factor for GC-to-variable messages.
    lrbms_order : int
        LRB-MS order ``t``.
    gc_method : str
        ``'lrbms'`` or ``'trellis'``.
    schedule : str
        ``'parallel'`` or ``'serial'``.
    osd_method : str
        OSD fallback when LRB-MS does not converge: ``'off'``, ``'osd_0'``, ``'osd_cs'``, ``'osd_e'``.
    osd_order : int
        OSD order.

    Example
    -------
    >>> sinter.collect(..., custom_decoders={"lrbms": SinterLrbmsDecoder(check_groups=4, lrbms_order=8)})
    """

    def __init__(
        self,
        check_groups=None,
        max_iter=0,
        ms_scaling_factor=0.625,
        lrbms_order=0,
        gc_method="lrbms",
        schedule="parallel",
        osd_method="off",
        osd_order=0,
    ):
        self.check_groups = check_groups
        self.max_iter = max_iter
        self.ms_scaling_factor = ms_scaling_factor
        self.lrbms_order = lrbms_order
        self.gc_method = gc_method
        self.schedule = schedule
        self.osd_method = osd_method
        self.osd_order = osd_order

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,
    ) -> None:
        self.dem = stim.DetectorErrorModel.from_file(dem_path)
        self.matrices = detector_error_model_to_check_matrices(
            self.dem, allow_undecomposed_hyperedges=True
        )
        self.lrbms = LrbmsDecoder(
            self.matrices.check_matrix,
            error_channel=list(self.matrices.priors),
            check_groups=self.check_groups,
            max_iter=self.max_iter,
            ms_scaling_factor=self.ms_scaling_factor,
            lrbms_order=self.lrbms_order,
            gc_method=self.gc_method,
            schedule=self.schedule,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )

        shots = stim.read_shot_data_file(
            path=dets_b8_in_path, format="b8", num_detectors=num_dets
        )
        predictions = np.zeros((num_shots, num_obs), dtype=bool)
        for i in range(num_shots):
            predictions[i, :] = self.decode(shots[i, :])

        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=num_obs,
        )

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        corr = self.lrbms.decode(syndrome)
        return (self.matrices.observables_matrix @ corr) % 2
