"""
Module :mod:`sifting._empirical_optimal_envelope`

This module implements the empirical optimal envelope extraction algorithm. It is an
algorithm that aims for obtaining an envelope that does not show the over- and
undershoots that are present in the upper and lower envelopes obtained by the cubic
spline interpolation of the maxima and minima of the signal.

"""

# === Imports ===

from typing import Any, Callable, Dict, Literal, Optional

import numpy as np
from _interpolate import interp_cubic_spline
from _minmax import find_relative_extrema_indices

# === Functions ===


def empirical_optimal_envelope(
    x: np.ndarray,
    y: np.ndarray,
    which: Literal["lower", "upper"],
    num_iters: int = 2,
    interpolator: Callable[..., np.ndarray] = interp_cubic_spline,
    interpolator_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Estimates the empirical optimal envelope of a series of data.

    Parameters
    ----------
    x, y : :class:`numpy.ndarray` of shape (n,)
        The x and y values of the series for which the empirical optimal envelope is to
        be estimated.
    which : ``{"lower", "upper"}``
        Whether to estimate the lower (``"lower"``) or upper (``"upper"``) empirical
        optimal envelope.
    num_iters : :class:`int`, default=2
        The number of refinement iterations to perform when estimating the empirical
        optimal envelope. Usually, less than 10 iterations are sufficient.
        The initial estimation is not included in this count.
        A value of ``0`` is equivalent to the simple interpolation of the Empirical Mode
        Decomposition (EMD) envelopes.
    interpolator : callable, default=interp_cubic_spline
        The interpolator to use when estimating the empirical optimal envelope.
        It has to be a callable with the signature

        .. code-block:: python

            interpolator(x_fit, y_fit, x_eval, **interpolator_kwargs) -> y_eval

        where ``x_fit`` and ``y_fit`` are the x and y values to fit the interpolation
        while ``y_eval`` is the interpolated values when the fit is evaluated at the
        points ``x_eval``.

    interpolator_kwargs : :class:`dict` or ``None``, default=``None``
        The optional keyword arguments to pass to the interpolator.

    Returns
    -------
    empirical_optimal_envelope : :class:`numpy.ndarray` of shape (n,)
        The estimated empirical optimal envelope of the series.

    """

    # --- Initial Iteration ---

    # first, the respective relative extrema are found as an initial estimate of the
    # empirical optimal envelope
    extremum_kind = "min" if which == "lower" else "max"
    envelope_indices = find_relative_extrema_indices(
        y=y,
        include_edges=True,
        which=extremum_kind,
    )

    # then, they are interpolated at all x-values
    interpolator_kwargs = (
        interpolator_kwargs if interpolator_kwargs is not None else dict()
    )
    empirical_optimal_envelope = interpolator(
        x_fit=x[envelope_indices],
        y_fit=y[envelope_indices],
        x_eval=x,
        **interpolator_kwargs,
    )

    # --- Refinement Iterations ---

    # the empirical optimal envelope is refined iteratively by moving the envelope base
    # points to make them tangent to the signal
    for _ in range(0, num_iters):

        # the distances to the envelope are calculated
        envelope_distance = y - empirical_optimal_envelope

        # the tangent points are found
        envelope_indices = find_relative_extrema_indices(
            y=envelope_distance,
            include_edges=True,
            which=extremum_kind,
        )

        # the envelope is updated
        empirical_optimal_envelope = interpolator(
            x_fit=x[envelope_indices],
            y_fit=y[envelope_indices],
            x_eval=x,
            **interpolator_kwargs,
        )

    return empirical_optimal_envelope


def empirical_optimal_envelope_imf(
    x: np.ndarray,
    y: np.ndarray,
    num_iters: int = 2,
    interpolator: Callable[..., np.ndarray] = interp_cubic_spline,
    interpolator_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Estimates an Intrinsic Mode Function (IMF) of a signal as the mean of its lower and
    upper empirical optimal envelope.

    Parameters
    ----------
    x, y : :class:`numpy.ndarray` of shape (n,)
        The x and y values of the series for which the empirical optimal envelope is to
        be estimated.
    which : ``{"lower", "upper"}``
        Whether to estimate the lower (``"lower"``) or upper (``"upper"``) empirical
        optimal envelope.
    num_iters : :class:`int`, default=2
        The number of refinement iterations to perform when estimating the empirical
        optimal envelope. Usually, less than 10 iterations are sufficient.
        The initial estimation is not included in this count.
        A value of ``0`` is equivalent to the simple interpolation of the Empirical Mode
        Decomposition (EMD) envelopes.
    interpolator : callable, default=interp_cubic_spline
        The interpolator to use when estimating the empirical optimal envelope.
        It has to be a callable with the signature

        .. code-block:: python

            interpolator(x_fit, y_fit, x_eval, **interpolator_kwargs) -> y_eval

        where ``x_fit`` and ``y_fit`` are the x and y values to fit the interpolation
        while ``y_eval`` is the interpolated values when the fit is evaluated at the
        points ``x_eval``.

    interpolator_kwargs : :class:`dict` or ``None``, default=``None``
        The optional keyword arguments to pass to the interpolator.


    Notes
    -----
    The Intrinsic Mode Function (IMF) is a signal that is defined as the mean of its
    lower and upper empirical optimal envelope.
    In contrary to the simple sifting algorithm where the envelopes are obtained by
    interpolation of the maxima and minima of the signal, the empirical optimal envelope
    algorithm iteratively refines the interpolation by moving the interpolation base
    points to make them tangent to the signal. This way, the over- and undershoots that
    the interpolation might introduce are minimised.

    References
    ----------
    .. [1] Jia L., et al., The empirical optimal envelope and its application to local
       mean decomposition, Digital Signal Processing (2019), Issue 87, pp. 166–177,
       DOI: doi:10.1016/j.dsp.2019.01.024

    """

    kwargs = dict(
        num_iters=num_iters,
        interpolator=interpolator,
        interpolator_kwargs=interpolator_kwargs,
    )
    lower_envelope = empirical_optimal_envelope(
        x=x,
        y=y,
        which="lower",
        **kwargs,  # type: ignore
    )
    upper_envelope = empirical_optimal_envelope(
        x=x,
        y=y,
        which="upper",
        **kwargs,  # type: ignore
    )

    return 0.5 * (lower_envelope + upper_envelope)
