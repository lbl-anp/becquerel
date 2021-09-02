"""Define errors and warnings for I/O."""

import warnings

warnings.simplefilter("always", DeprecationWarning)


class BecquerelParserWarning(UserWarning):
    """Warnings encountered during parsing."""

    pass


class BecquerelParserError(Exception):
    """Failure encountered while parsing."""

    pass
