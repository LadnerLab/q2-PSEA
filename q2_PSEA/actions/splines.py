# Provides a place to collect spline operations.

import numpy as np

from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from scipy import interpolate


SPLINE_TYPES = ["r-smooth", "py-smooth", "cubic"]


def smooth_spline(x, y, knots=3, s=0.788458):
    """Returns predicted values of `y` based on the given `x` values

    Parameters
    ----------
    x : list(float)

    y : list(float)

    knots : int

    s : float

    Returns
    -------
    list(float)
        Predicted y value for every given x value
    """
    x_new = np.linspace(0, 1, knots+2)[1:-1]
    q_knots = np.quantile(x, x_new)
    t, c, k = interpolate.splrep(x, y, t=q_knots, s=s)
    return interpolate.BSpline(t, c, k)(x)


r_splines = """
smooth_spline <- function(x, y)
{
    smooth_spline <- smooth.spline(x, y)
    yfit <- predict(smooth_spline, x)$y

    return(yfit)
}


cubic_spline <- function(x, y, degree, df)
{
    library(splines2)

    knots <- summary(x)[c(2, 3, 5)]
    sorted.x = sort(x)

    bsMat <- bSpline(
        x,
        knots = knots,
        degree = degree,
        df = df,
        intercept = TRUE
    )

    cubic_spline_obj <- lm(y ~ bsMat)
    cubic_spline_preds <- predict(
        cubic_spline_obj,
        newdata = list(sorted.x),
        se=TRUE
    )
    cubic_spline_se_bands <- cbind(
        cubic_spline_preds$fit + 2 * cubic_spline_preds$se.fit,
        cubic_spline_preds$fit - 2 * cubic_spline_preds$se.fit
    )

    return(cubic_spline_preds$fit)
}
"""

R_SPLINES = SignatureTranslatedAnonymousPackage(r_splines, "internal")
