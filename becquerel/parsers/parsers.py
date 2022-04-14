"""Define errors and warnings for I/O."""

import warnings

warnings.simplefilter("always", DeprecationWarning)


class BecquerelParserWarning(UserWarning):
    """Warnings encountered during parsing."""

    pass


class BecquerelParserError(Exception):
    """Failure encountered while parsing."""

    pass


def override_calibration(cal, **kwargs):
    """Override the settings of the calibration.

    Convenience function for altering calibrations while reading from file.

    Parameters
    ----------
    cal : Calibration
        Energy calibration to alter.
    kwargs : dict
        Kwargs to override the Calibration parameters read from file.

    Returns
    -------
    cal2 : Calibration
        Updated calibration.
    """
    cal2 = cal.copy()
    for key, value in kwargs.items():
        setattr(cal2, key, value)
    return cal2
