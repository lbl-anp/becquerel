"""Test isotope.py classes."""

from __future__ import print_function
import datetime
from dateutil.parser import parse as dateutil_parse
from six import string_types
import numpy as np
from becquerel.tools import element
from becquerel.tools import isotope
import pytest


TEST_ISOTOPES = [
    ('H-3', 'H', 3, ''),
    ('He-4', 'He', 4, ''),
    ('K-40', 'K', 40, ''),
    ('Co-60', 'Co', 60, ''),
    ('U-238', 'U', 238, ''),
    ('Pu-244', 'Pu', 244, ''),
    ('Tc-99m', 'Tc', 99, 'm'),
    ('Tc-99m', 'Tc', '99', 'm'),
    ('Tc-99m', 'Tc', 99, 1),
    ('Pa-234m', 'Pa', 234, 'm'),
    ('Hf-178m1', 'Hf', 178, 'm1'),
    ('Hf-178m2', 'Hf', 178, 'm2'),
    ('Hf-178m3', 'Hf', 178, 'm3'),
    ('Hf-178m3', 'Hf', 178, 3),
]


# ----------------------------------------------------
#                   Isotope class
# ----------------------------------------------------

@pytest.mark.parametrize('iso_str, sym, A, m', TEST_ISOTOPES)
def test_isotope_init_args(iso_str, sym, A, m):
    """Test Isotope init with 2 or 3 args, depending on whether m is None."""
    sym = str(sym)
    name = element.element_name(sym)
    Z = element.element_z(sym)
    A = int(A)
    elems = [
        sym, sym.lower(), sym.upper(),
        name, name.lower(), name.upper(),
        Z, str(Z)]
    mass_numbers = [A, str(A)]
    if isinstance(m, int):
        if m == 0:
            m = ''
        else:
            m = 'm{}'.format(m)
    if m == '':
        isomer_levels = [None, '', 0]
    elif m.lower() == 'm':
        isomer_levels = ['m', 'M', 1]
    elif m.lower() == 'm1':
        isomer_levels = ['m1', 'M1']
    else:
        isomer_levels = [m.lower(), m.upper(), int(m[1:])]
    for elem in elems:
        for mass in mass_numbers:
            for isomer in isomer_levels:
                args_list = [(elem, mass, isomer)]
                if isomer == '' or isomer is None:
                    args_list.append((elem, mass))
                for args in args_list:
                    print('')
                    print(args)
                    i = isotope.Isotope(*args)
                    print(i)
                    assert i.symbol == sym
                    assert i.A == A
                    assert i.m == m


@pytest.mark.parametrize('elem, A, m', [
    ('Xx', 45, ''),     # invalid element symbol
    (119, 250, ''),     # invalid Z (not in range 1..118)
    ('H', -1, ''),      # invalid A (negative)
    ('H', 'AA', ''),    # invalid A (string that cannot be converted to int)
    ('Ge', 30, ''),     # invalid N (negative)
    ('Tc', 99, 'n'),    # invalid m (does not start with 'm')
    ('Tc', 99, 'm-1'),  # invalid m (negative, str)
    ('Tc', 99, -1),     # invalid m (negative, int)
    ('Tc', 99, 1.0),    # invalid m (floating point)
])
def test_isotope_init_args_exceptions(elem, A, m):
    """Test Isotope init raises exceptions in some cases."""
    with pytest.raises(isotope.IsotopeError):
        isotope.Isotope(elem, A, m)


def test_isotope_init_args_exception_noargs():
    """Test Isotope init raises exception if no arguments given."""
    with pytest.raises(isotope.IsotopeError):
        isotope.Isotope()


def test_isotope_init_args_exception_1arg_nonstring():
    """Test Isotope init raises exception if one non-string argument given."""
    with pytest.raises(isotope.IsotopeError):
        isotope.Isotope(32)


def test_isotope_init_args_exception_4args():
    """Test Isotope init raises exception if four arguments given."""
    with pytest.raises(isotope.IsotopeError):
        isotope.Isotope('Tc', 99, 'm', 0)


@pytest.mark.parametrize('sym1, A1, m1, sym2, A2, m2, equal', [
    ('Tc', 99, '', 'Tc', 99, '', True),       # symbol and A only
    ('Tc', 99, 'm', 'Tc', 99, 'm', True),     # symbol, A, and m
    ('Hf', 178, 'm', 'Hf', 178, 'm1', True),  # m and m1 are equivalent
    ('Ge', 68, '', 'Ga', 68, '', False),      # symbols differ
    ('Ge', 68, '', 'Ge', 69, '', False),      # masses differ
    ('Ge', 68, '', 'Ge', 69, 'm', False),     # metastable states differ
    ('Ge', 68, 'm1', 'Ge', 69, 'm2', False),  # metastable states differ
])
def test_isotope_equality(sym1, A1, m1, sym2, A2, m2, equal):
    """Test Isotope equality and inequality."""
    i1 = isotope.Isotope(sym1, A1, m1)
    i2 = isotope.Isotope(sym2, A2, m2)
    if equal:
        assert i1 == i2
    else:
        assert i1 != i2


def test_isotope_equality_exception():
    """Test TypeError is raised when comparing an Isotope to a non-Isotope."""
    i1 = isotope.Isotope('Tc', 99, 'm')
    with pytest.raises(TypeError):
        assert i1 == 'Tc-99m'


@pytest.mark.parametrize('iso_str, sym, A, m', TEST_ISOTOPES)
def test_isotope_init_str(iso_str, sym, A, m):
    """Test Isotope init with one (string) argument.

    Isotope is identified by element symbol, A, and isomer number.

    Run tests for element symbol and name, in mixed case, upper case,
    and lower case.
    """
    sym = str(sym)
    mass_number = str(A)
    if m is not None:
        if isinstance(m, int):
            mass_number += 'm{}'.format(m)
        else:
            mass_number += m
    expected = isotope.Isotope(sym, A, m)
    for name in [sym, element.element_name(sym)]:
        iso_tests = [
            name + '-' + mass_number,
            name + mass_number,
            mass_number + '-' + name,
            mass_number + name,
        ]
        for iso in iso_tests:
            for iso2 in [iso, iso.upper(), iso.lower()]:
                print('')
                print('{}-{}: {}'.format(sym, mass_number, iso2))
                i = isotope.Isotope(iso2)
                print(i)
                assert i == expected


@pytest.mark.parametrize('iso_str, raises', [
    ('He_4', True),       # underscore
    ('He--4', True),      # hyphens
    ('4399', True),       # all numbers
    ('abdc', True),       # all letters
    ('Tc-99m3m', True),   # too many ms
    ('55mN', False),      # ambiguous but valid (returns Mn-55, not N-55m)
    ('55m2N', False),     # like previous but unambiguous (returns N-55m2)
    ('24Mg', False),      # unambiguous because G is not an element
    ('24m2G', True),      # like previous but invalid
    ('2He4', True),       # invalid mass number
    ('2He-4', True),      # invalid mass number
    ('He2-4', True),      # invalid mass number
    ('Xz-90', True),      # invalid element
    ('H-AA', True),       # invalid A (string that cannot be converted to int)
    ('H-20Xm1', True),    # invalid A (string that cannot be converted to int)
    ('Tc-99n', True),     # invalid m (does not start with 'm')
    ('Hf-178n3', True),   # invalid m (does not start with 'm')
    ('Tc-99m1.0', True),  # invalid m (floating point)
])
def test_isotope_init_str_exceptions(iso_str, raises):
    """Test Isotope init exceptions with one (string) argument."""
    if raises:
        with pytest.raises(isotope.IsotopeError):
            isotope.Isotope(iso_str)
    else:
        isotope.Isotope(iso_str)


@pytest.mark.parametrize('iso_str, sym, A, m', TEST_ISOTOPES)
def test_isotope_str(iso_str, sym, A, m):
    """Test Isotope.__str__()."""
    i = isotope.Isotope(sym, A, m)
    print(str(i), iso_str)
    assert str(i) == iso_str


# ----------------------------------------------------
#               IsotopeQuantity class
# ----------------------------------------------------

@pytest.mark.parametrize('kwargs', [
    {'bq': 10.047 * 3.7e4},
    {'uci': 10.047}
])
@pytest.mark.parametrize('date', [
    datetime.datetime.now(),
    '2015-01-08 00:00:00',
    None
])
@pytest.mark.parametrize('iso', [
    isotope.Isotope('Cs-137'),
    isotope.Isotope('178M2HF'),
    isotope.Isotope('Strontium', 90)
])
def test_isotopequantity_init(iso, date, kwargs):
    """Test Isotope.__init__()."""

    iq = isotope.IsotopeQuantity(iso, date=date, **kwargs)
    assert iq.isotope is iso
    if 'bq' in kwargs:
        assert iq.ref_activity == kwargs['bq']
    else:
        assert iq.ref_activity == kwargs['uci'] * 3.7e4
    if isinstance(date, datetime.datetime):
        assert iq.ref_date == date
    elif isinstance(date, string_types):
        assert iq.ref_date == dateutil_parse(date)
    else:
        assert (datetime.datetime.now() - iq.ref_date).total_seconds() < 5


@pytest.mark.parametrize('iso, date, kwargs, error', [
    ('Cs-137', datetime.datetime.now(), {'uci': 10.047}, TypeError),
    (isotope.Isotope('Cs-137'), 123, {'bq': 456}, TypeError),
    (isotope.Isotope('Cs-137'), datetime.datetime.now(), {'asdf': 3},
     isotope.IsotopeError)
])
def test_isotopequantity_bad_init(iso, date, kwargs, error):
    """Test errors from Isotope.__init__()"""

    with pytest.raises(error):
        isotope.IsotopeQuantity(iso, date=date, **kwargs)


@pytest.fixture(params=[
    (isotope.Isotope('Cs-137'), datetime.datetime.now(), {'uci': 10.047})
])
def iq(request):
    """An IsotopeQuantity object"""

    iso = request.param[0]
    date = request.param[1]
    kwargs = request.param[2]
    return isotope.IsotopeQuantity(iso, date=date, **kwargs)


@pytest.mark.parametrize('halflife', (3.156e7, 3600, 0.11))
def test_isotopequantity_bq_at(iq, halflife):
    """Test IsotopeQuantity.bq_at()"""

    # since halflife is not built in to Isotope yet...
    iq.isotope.halflife = halflife

    assert iq.bq_at(iq.ref_date) == iq.ref_activity

    dt = datetime.timedelta(seconds=halflife)
    assert iq.bq_at(iq.ref_date + dt) == iq.ref_activity / 2
    assert iq.bq_at(iq.ref_date - dt) == iq.ref_activity * 2

    dt = datetime.timedelta(seconds=halflife * 50)
    assert iq.bq_at(iq.ref_date + dt) < 1

    dt = datetime.timedelta(seconds=halflife / 100)
    assert np.isclose(iq.bq_at(iq.ref_date + dt), iq.ref_activity, rtol=1e-2)


@pytest.mark.parametrize('halflife', (3.156e7, 3600, 0.11))
def test_isotopequantity_uci_at(iq, halflife):
    """Test IsotopeQuantity.uci_at()"""

    # since halflife is not built in to Isotope yet...
    iq.isotope.halflife = halflife

    assert iq.uci_at(iq.ref_date) == iq.ref_activity / isotope.BQ_TO_UCI


@pytest.mark.parametrize('halflife', (3.156e7, 3600, 0.11))
def test_isotopequantity_time_when(iq, halflife):
    """Test IsotopeQuantity.time_when()"""

    # since halflife is not built in to Isotope yet...
    iq.isotope.halflife = halflife

    assert iq.time_when(bq=iq.ref_activity) == iq.ref_date

    d = iq.ref_date - datetime.timedelta(seconds=halflife)
    assert iq.time_when(bq=iq.ref_activity * 2) == d

    d = iq.ref_date + datetime.timedelta(seconds=halflife)
    assert iq.time_when(bq=iq.ref_activity / 2) == d
