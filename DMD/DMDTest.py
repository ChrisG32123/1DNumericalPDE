"""Basic Dynamic Mode Decomposition implementation.

This module provides a small helper class :class:`DMDTest` used in a few of the
experiments throughout this repository.  The original file was truncated and
contained unfinished code; it has been replaced with a minimal, self contained
implementation so that it can be imported without errors.
"""

from __future__ import annotations

import numpy as np
import numpy.linalg as la


class DMDTest:
    """Simple Dynamic Mode Decomposition (DMD).

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        Snapshot matrix where each column corresponds to a state at a
        particular time.
    r : int
        Truncation rank used for the singular value decomposition.
    dt : float
        Time step separating the snapshots in ``data``.
    """

    def __init__(self, data: np.ndarray, r: int, dt: float) -> None:
        self._data1 = np.asarray(data[:, :-1])
        self._data2 = np.asarray(data[:, 1:])
        self.r = r
        self.dt = dt

        self.modes: np.ndarray | None = None
        self.eigs: np.ndarray | None = None

    # ------------------------------------------------------------------
    def DMD(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the DMD modes and eigenvalues.

        Returns
        -------
        modes : :class:`numpy.ndarray`
            Array whose columns are the dynamic modes.
        eigs : :class:`numpy.ndarray`
            Array of eigenvalues associated with ``modes``.
        """

        # Singular value decomposition of the first snapshot matrix
        U, s, Vh = la.svd(self._data1, full_matrices=False)

        # Truncate to the requested rank
        r = min(self.r, len(s))
        Ur = U[:, :r]
        Sr = np.diag(s[:r])
        Vr = Vh.conj().T[:, :r]

        # Low‑rank linear operator
        A_tilde = Ur.conj().T @ self._data2 @ Vr @ la.inv(Sr)

        # Eigendecomposition of the reduced operator
        eigvals, W = la.eig(A_tilde)

        # Construct the DMD modes
        Phi = self._data2 @ Vr @ la.inv(Sr) @ W

        self.modes = Phi
        self.eigs = eigvals

        return self.modes, self.eigs


__all__ = ["DMDTest"]

