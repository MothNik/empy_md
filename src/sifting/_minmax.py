"""
Module :mod:`sifting._minmax`

This module implements functions to find the relative minima and maxima of a series of
data.

"""

# === Imports ===

from typing import Literal, Tuple

import numpy as np
from scipy.signal import argrelextrema

# === Functions ===


def _include_edge_indices(
    indices: np.ndarray,
    y_size: int,
) -> np.ndarray:
    """
    Includes the edge indices in the array of indices if they are not already included.

    """

    if indices[0] != 0:
        indices = np.concatenate(
            (
                np.array([0], dtype=indices.dtype),
                indices,
            )
        )

    max_possible_index = y_size - 1
    if indices[-1] != max_possible_index:
        indices = np.concatenate(
            (
                indices,
                np.array(
                    [max_possible_index],
                    dtype=indices.dtype,
                ),
            )
        )

    return indices


def find_relative_extrema_indices(
    y: np.ndarray,
    include_edges: bool,
    which: Literal["min", "max"],
) -> np.ndarray:
    """
    Finds the indices of the relative minima OR maxima of a series of data.

    Parameters
    ----------
    y : :class:`numpy.ndarray` of shape (n,)
        The data for which to find the relative extrema.
    include_edges : :class:`bool`
        Whether to specifically include the first and last points of the data as
        relative extrema even if they are not local extrema (``True``) or not
        (``False``).
    which : ``{"min", "max"}``
        Whether to find the minima (``"min"``) or the maxima (``"max"``).

    Returns
    -------
    rel_extrema_indices : :class:`numpy.ndarray` of shape (a,)
        The indices of the relative extrema of ``y``.

    """

    # the extrema are found
    rel_extrema_indices = argrelextrema(
        data=y,
        comparator=np.less_equal if which == "min" else np.greater_equal,
        order=1,
        mode="clip",
    )[0]

    # if the edges are not to be specifically included, the extrema are returned as they
    # are
    if not include_edges:
        return rel_extrema_indices

    # if the edges are to be included, they are unless they are already included
    rel_extrema_indices = _include_edge_indices(
        indices=rel_extrema_indices,
        y_size=len(y),
    )

    return rel_extrema_indices


def find_relative_minima_and_maxima_indices(
    y: np.ndarray,
    include_edges: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the indices of the relative minima AND maxima of a series of data.

    Parameters
    ----------
    y : :class:`numpy.ndarray` of shape (n,)
        The data for which to find the relative extrema.
    include_edges : :class:`bool`
        Whether to specifically include the first and last points of the data as
        relative extrema even if they are not local extrema (``True``) or not
        (``False``).

    Returns
    -------
    rel_maxima_indices : :class:`numpy.ndarray` of shape (a,)
        The indices of the relative maxima of ``y``.
    rel_minima_indices : :class:`numpy.ndarray` of shape (b,)
        The indices of the relative minima of ``y``.

    """

    return (
        find_relative_extrema_indices(
            y=y,
            include_edges=include_edges,
            which="max",
        ),
        find_relative_extrema_indices(
            y=y,
            include_edges=include_edges,
            which="min",
        ),
    )
