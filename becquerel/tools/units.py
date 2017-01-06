"""Units for becquerel using pint."""

from __future__ import print_function
import pint
from pint.unit import ScaleConverter
from pint.unit import UnitDefinition

units = pint.UnitRegistry()

# define dimensionless percent unit
# this is not trivial, the solution as of pint v0.7.2 is:
# http://stackoverflow.com/questions/39153885/how-to-define-and-use-percentage-in-pint

units.define(
    UnitDefinition('%', 'percent', (), ScaleConverter(1 / 100.0)))

# make some unit abbreviations the default
units.define('keV = kiloelectron_volt')
units.define('MeV = megaelectron_volt')
