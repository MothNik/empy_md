"""
Module :mod:`sifting._interpolate`

This module implements functions to interpolate a series of data.

"""

# === Imports ===

from typing import Any, Dict, Type, Union

import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator

# === Functions ===


def _interp_curve(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_eval: np.ndarray,
    interpolator: Union[
        Type[Akima1DInterpolator], Type[PchipInterpolator], Type[CubicSpline]
    ],
    kwargs: Dict[str, Any],
) -> np.ndarray:
    """
    Interpolates a curve at new points using the specified interpolator.

    Parameters
    ----------
    x_fit, y_fit : :class:`numpy.ndarray` of shape (n,)
        The points of the curve to interpolate.
    x_eval : :class:`numpy.ndarray` of shape (m,)
        The points at which to interpolate the curve.
    interpolator : :class:`type`
        The interpolator to use.
    kwargs : :class:`dict`
        The keyword arguments to pass to the interpolator when it is created.

    Returns
    -------
    y_new : :class:`numpy.ndarray` of shape (m,)
        The interpolated values of the curve at the new points.

    """

    # the interpolator is created and evaluated
    return interpolator(x=x_fit, y=y_fit, **kwargs)(x=x_eval)


def interp_cubic_spline(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """
    Interpolates a curve at new points using cubic spline interpolation with natural
    boundary conditions.
    For details on the input parameters, please refer to the function
    :func:`_interp_curve`.

    """

    # the cubic spline is created and evaluated
    return _interp_curve(
        x_fit=x_fit,
        y_fit=y_fit,
        x_eval=x_eval,
        interpolator=CubicSpline,
        kwargs=dict(bc_type="natural"),
    )


def interp_akima(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """
    Interpolates a curve at new points using Akima interpolation with modified weights
    to avoid over- and undershooting.
    For details on the input parameters, please refer to the function
    :func:`_interp_curve`.

    """

    # the Akima interpolator is created and evaluated
    return _interp_curve(
        x_fit=x_fit,
        y_fit=y_fit,
        x_eval=x_eval,
        interpolator=Akima1DInterpolator,
        kwargs=dict(method="makima"),
    )


def interp_pchip(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """
    Interpolates a curve at new points using PCHIP interpolation.
    For details on the input parameters, please refer to the function
    :func:`_interp_curve`.

    """

    # the PCHIP interpolator is created and evaluated
    return _interp_curve(
        x_fit=x_fit,
        y_fit=y_fit,
        x_eval=x_eval,
        interpolator=PchipInterpolator,
        kwargs=dict(),
    )
