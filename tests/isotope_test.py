"""Test isotope.py classes."""

from __future__ import print_function
import datetime
from dateutil.parser import parse as dateutil_parse
from six import string_types
import numpy as np
from becquerel.tools import element
from becquerel.tools import isotope
from becquerel import Spectrum
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


ISOTOPE_PROPERTIES = [
    ('H-3', (3.888E8, False, None, '1/2+', 0., None, [['B-'], [100.]],)),
    ('He-4', (np.inf, True, 99.999866, '0+', 0., 2.4249, [[], []])),
    ('K-40', (3.938E16, False, 0.0117, '4-', 0., -33.53540, [['B-', 'EC'], [89.28, 10.72]])),
    ('Co-60', (1.663E8, False, None, '5+', 0., -61.6503, [['B-'], [100.]])),
    ('U-238', (1.41E17, False, 99.2742, '0+', 0., 47.3077, [['A', 'SF'], [100.00, 5.4E-5]])),
    ('Pu-244', (2.525E15, False, None, '0+', 0., 59.8060, [['A', 'SF'], [99.88, 0.12]])),
    ('Tc-99m', (21624.12, False, None, '1/2-', 0.1427, -87.1851, [['IT', 'B-'], [100., 3.7E-3]])),
    ('Pa-234m', (69.54, False, None, '(0-)', 0.0739, 40.413, [['IT', 'B-'], [0.16, 99.84]])),
    ('Hf-178', (np.inf, True, 27.28, '0+', 0., -52.4352, [[], []])),
    ('Hf-178m1', (4., False, None, '8-', 1.1474, -51.2878, [['IT'], [100.]])),
    ('Hf-178m2', (9.783E8, False, None, '16+', 2.4461, -49.9891, [['IT'], [100.]])),
]


@pytest.mark.webtest
@pytest.mark.parametrize('iso_str, props', ISOTOPE_PROPERTIES)
def test_isotope_properties(iso_str, props):
    """Test that isotope properties are correct."""
    i = isotope.Isotope(iso_str)
    half_life, is_stable, abundance, j_pi, energy_level, mass_excess, modes = \
        props
    if not np.isinf(half_life):
        assert np.isclose(i.half_life, half_life)
    else:
        assert np.isinf(i.half_life)
    assert i.is_stable == is_stable
    if abundance is None:
        assert i.abundance is None
    else:
        assert np.isclose(i.abundance.nominal_value, abundance)
    assert i.j_pi == j_pi
    assert np.isclose(i.energy_level, energy_level)
    print('mass excess:', i.mass_excess, type(i.mass_excess))
    if mass_excess is None:
        assert i.mass_excess is None
    else:
        print('mass excess:', i.mass_excess.nominal_value)
        print('mass excess:', mass_excess)
        print(np.isclose(mass_excess, mass_excess))
        print(np.isclose(mass_excess, i.mass_excess.nominal_value))
        print(np.isclose(mass_excess, (i.mass_excess).nominal_value))
        assert np.isclose(i.mass_excess.nominal_value, mass_excess)
    assert i.decay_modes[0] == modes[0]
    assert i.decay_modes[1] == modes[1]


# ----------------------------------------------------
#               IsotopeQuantity class
# ----------------------------------------------------

@pytest.fixture(params=(
    'Cs-137',
    'K-40',
    'Cs-134',
    'Tc-99m',
    'Tl-208'
))
def radioisotope(request):
    return isotope.Isotope(request.param)


@pytest.fixture(params=(
    'H-2',
    'Cs-133',
    'Pb-208'
))
def stable_isotope(request):
    return isotope.Isotope(request.param)


@pytest.fixture(params=[
    {'bq': 10.047 * isotope.UCI_TO_BQ},
    {'uci': 10.047},
    {'atoms': 1e24},
    {'g': 1e-5}
])
def iq_kwargs(request):
    return request.param


@pytest.fixture(params=[
    datetime.datetime.now(),
    '2015-01-08 00:00:00',
    None
])
def iq_date(request):
    return request.param


def test_isotopequantity_init_rad(radioisotope, iq_date, iq_kwargs):
    """Test IsotopeQuantity.__init__() for radioactive isotopes"""

    iq = isotope.IsotopeQuantity(radioisotope, date=iq_date, **iq_kwargs)
    assert iq.isotope is radioisotope
    assert iq.half_life == radioisotope.half_life
    assert iq.decay_const == radioisotope.decay_const

    # check string input
    iq = isotope.IsotopeQuantity(str(radioisotope), date=iq_date, **iq_kwargs)
    assert iq.isotope == radioisotope
    assert iq.half_life == radioisotope.half_life
    assert iq.decay_const == radioisotope.decay_const


def test_isotopequantity_init_stable(stable_isotope, iq_date, iq_kwargs):
    """Test IsotopeQuantity.__init__() for radioactive isotopes"""

    if 'bq' in iq_kwargs or 'uci' in iq_kwargs:
        with pytest.raises(isotope.IsotopeError):
            iq = isotope.IsotopeQuantity(
                stable_isotope, date=iq_date, **iq_kwargs)
        return None
    iq = isotope.IsotopeQuantity(stable_isotope, date=iq_date, **iq_kwargs)
    assert iq.isotope is stable_isotope
    assert iq.half_life == stable_isotope.half_life
    assert iq.decay_const == stable_isotope.decay_const


def test_isotopequantity_ref_atoms_rad(radioisotope, iq_kwargs):
    """Test IsotopeQuantity.ref_atoms for a radioactive isotope"""

    iq = isotope.IsotopeQuantity(radioisotope, **iq_kwargs)
    if 'atoms' in iq_kwargs:
        assert iq.ref_atoms == iq_kwargs['atoms']
    elif 'g' in iq_kwargs:
        assert iq.ref_atoms == (
            iq_kwargs['g'] / radioisotope.A * isotope.N_AV)
    elif 'bq' in iq_kwargs:
        assert iq.ref_atoms == iq_kwargs['bq'] / radioisotope.decay_const
    else:
        assert iq.ref_atoms == (
            iq_kwargs['uci'] * isotope.UCI_TO_BQ / radioisotope.decay_const)


def test_isotopequantity_ref_atoms_stable(stable_isotope):
    """Test IsotopeQuantity.ref_atoms for a stable isotope"""

    atoms = 1e24
    iq = isotope.IsotopeQuantity(stable_isotope, atoms=atoms)
    assert iq.ref_atoms == atoms

    g = 1e-5
    iq = isotope.IsotopeQuantity(stable_isotope, g=g)
    assert iq.ref_atoms == g / stable_isotope.A * isotope.N_AV


def test_isotopequantity_ref_date_rad(radioisotope, iq_date):
    """Test IsotopeQuantity.ref_date for a radioisotope"""

    iq = isotope.IsotopeQuantity(radioisotope, date=iq_date, atoms=1e24)
    if isinstance(iq_date, datetime.datetime):
        assert iq.ref_date == iq_date
    elif isinstance(iq_date, string_types):
        assert iq.ref_date == dateutil_parse(iq_date)
    else:
        assert (datetime.datetime.now() - iq.ref_date).total_seconds() < 5


def test_isotopequantity_ref_date_stable(stable_isotope, iq_date):
    """Test IsotopeQuantity.ref_date for a stable isotope"""

    iq = isotope.IsotopeQuantity(stable_isotope, date=iq_date, atoms=1e24)
    if isinstance(iq_date, datetime.datetime):
        assert iq.ref_date == iq_date
    elif isinstance(iq_date, string_types):
        assert iq.ref_date == dateutil_parse(iq_date)
    else:
        assert (datetime.datetime.now() - iq.ref_date).total_seconds() < 5


@pytest.mark.parametrize('iso, date, kwargs, error', [
    (['Cs-137'], None, {'atoms': 1e24}, TypeError),
    ('Cs-137', 123, {'bq': 456}, TypeError),
    ('Cs-137', datetime.datetime.now(), {'asdf': 3}, isotope.IsotopeError),
    ('Cs-137', None, {'bq': -13.3}, ValueError)
])
def test_isotopequantity_bad_init(iso, date, kwargs, error):
    """Test errors from Isotope.__init__()"""

    with pytest.raises(error):
        isotope.IsotopeQuantity(iso, date=date, **kwargs)


@pytest.fixture
def iq(radioisotope):
    """An IsotopeQuantity object"""

    date = datetime.datetime.now()
    kwargs = {'uci': 10.047}
    return isotope.IsotopeQuantity(radioisotope, date=date, **kwargs)


def test_isotopequantity_at_methods(iq):
    """Test IsotopeQuantity.*_at()"""

    half_life = iq.half_life
    iq.creation_date = False    # allow pre-refdate queries

    assert iq.atoms_at(iq.ref_date) == iq.ref_atoms
    assert np.isclose(
        iq.g_at(iq.ref_date), iq.ref_atoms * iq.isotope.A / isotope.N_AV)
    assert np.isclose(
        iq.bq_at(iq.ref_date), iq.ref_atoms * iq.decay_const)
    assert np.isclose(
        iq.uci_at(iq.ref_date),
        iq.ref_atoms * iq.decay_const / isotope.UCI_TO_BQ)

    if iq.half_life < 3.156e7 * 1000:   # OverflowError or year out of range
        dt1 = iq.ref_date + datetime.timedelta(seconds=half_life)
        dt2 = iq.ref_date - datetime.timedelta(seconds=half_life)
        dt3 = iq.ref_date + datetime.timedelta(seconds=half_life * 50)
        dt4 = iq.ref_date + datetime.timedelta(seconds=half_life / 100)

        assert np.isclose(iq.atoms_at(dt1), iq.ref_atoms / 2)
        assert np.isclose(iq.atoms_at(dt2), iq.ref_atoms * 2)
        assert np.isclose(iq.bq_at(dt3) / iq.bq_at(iq.ref_date), 0, atol=1e-12)
        assert np.isclose(iq.bq_at(dt4), iq.bq_at(iq.ref_date), rtol=1e-2)


@pytest.mark.parametrize('kw', ('atoms', 'g', 'bq', 'uci'))
def test_isotopequantity_time_when(iq, kw):
    """Test IsotopeQuantity.time_when()"""

    iq.creation_date = False

    ref_qty = getattr(iq, kw + '_at')(iq.ref_date)

    kwarg = {kw: ref_qty}
    assert iq.time_when(**kwarg) == iq.ref_date

    if iq.half_life > 1000 * 3.156e7:
        # avoid overflow errors in test calculations
        # just make sure the method doesn't error
        kwarg = {kw: ref_qty * 0.999999}
        iq.time_when(**kwarg)

        kwarg = {kw: ref_qty * 1.0000001}
        iq.time_when(**kwarg)
    else:
        d = iq.ref_date - datetime.timedelta(seconds=iq.half_life)
        kwarg = {kw: ref_qty * 2}
        assert iq.time_when(**kwarg) == d

        d = iq.ref_date + datetime.timedelta(seconds=iq.half_life)
        kwarg = {kw: ref_qty / 2}
        assert iq.time_when(**kwarg) == d


def test_isotopequantity_time_when_error(stable_isotope):
    """Test error for time_when on a stable isotope"""

    iq = isotope.IsotopeQuantity(stable_isotope, g=10)
    with pytest.raises(isotope.IsotopeError):
        iq.time_when(g=5)


def test_isotopequantity_creation_date(radioisotope):
    """Test IsotopeQuantity created or non-created at ref date"""

    iq = isotope.IsotopeQuantity(
        radioisotope, date='2017-01-01 00:00:00', bq=1, creation_date=True)
    with pytest.raises(isotope.IsotopeError):
        iq.atoms_at('2016-01-01 00:00:00')
    assert iq.time_when(atoms=iq.ref_atoms * 1.0001) is None
    iq.atoms_at('2017-01-02 00:00:00')
    if iq.half_life < 1000 * 3.156e7:
        iq.time_when(atoms=iq.ref_atoms * 0.9999)

    iq = isotope.IsotopeQuantity(
        radioisotope, date='2017-01-01 00:00:00', bq=1, creation_date=False)
    iq.atoms_at('2016-01-01 00:00:00')
    iq.atoms_at('2017-01-02 00:00:00')
    if iq.half_life < 1000 * 3.156e7:
        assert iq.time_when(atoms=iq.ref_atoms * 1.0001) is not None
        iq.time_when(atoms=iq.ref_atoms * 0.9999)

    # default True
    iq = isotope.IsotopeQuantity(
        radioisotope, date='2017-01-01 00:00:00', bq=1)
    with pytest.raises(isotope.IsotopeError):
        iq.atoms_at('2016-01-01 00:00:00')
    assert iq.time_when(atoms=iq.ref_atoms * 1.0001) is None
    iq.atoms_at('2017-01-02 00:00:00')
    if iq.half_life < 1000 * 3.156e7:
        iq.time_when(atoms=iq.ref_atoms * 0.9999)


def test_isotopequantity_activity_now(iq):
    """Test IsotopeQuantity.*_now()"""

    assert np.isclose(iq.bq_now(), iq.bq_at(datetime.datetime.now()))
    assert np.isclose(iq.uci_now(), iq.uci_at(datetime.datetime.now()))
    assert np.isclose(iq.atoms_now(), iq.atoms_at(datetime.datetime.now()))
    assert np.isclose(iq.g_now(), iq.g_at(datetime.datetime.now()))


def test_isotopequantity_decays_from(iq):
    """Test IsotopeQuantity.*_from()"""

    if iq.half_life > 1000 * 3.156e7:
        # avoid overflow errors in test calculations
        # just make sure the method doesn't error
        now = datetime.datetime.now()
        iq.decays_from(now, now + datetime.timedelta(seconds=3600))
    else:
        t0 = datetime.datetime.now()
        dt = datetime.timedelta(seconds=iq.half_life)
        t1 = t0 + dt
        t2 = t1 + dt

        assert np.isclose(iq.decays_from(t0, t1), iq.atoms_at(t1))
        assert np.isclose(iq.decays_from(t1, t2), iq.atoms_at(t2))
        assert np.isclose(iq.decays_from(t0, t2), 3 * iq.atoms_at(t2))

        assert np.isclose(iq.bq_from(t0, t1), iq.atoms_at(t1) / iq.half_life)

        assert np.isclose(
            iq.uci_from(t0, t1),
            iq.atoms_at(t1) / iq.half_life / isotope.UCI_TO_BQ)


def test_isotopequantity_decays_during(iq):
    """Test IsotopeQuantity.*_during()"""

    if iq.half_life > 1000 * 3.156e7:
        # avoid overflow errors in test calculations
        pass
    else:
        dt_s = iq.half_life
        t0 = datetime.datetime.now()
        t1 = t0 + datetime.timedelta(seconds=dt_s)
        spec = Spectrum(np.zeros(256), start_time=t0, stop_time=t1)

        assert np.isclose(iq.decays_during(spec), iq.atoms_now() / 2)
        assert np.isclose(iq.bq_during(spec), iq.decays_during(spec) / dt_s)
        assert np.isclose(iq.uci_during(spec),
                          iq.bq_during(spec) / isotope.UCI_TO_BQ)


# ----------------------------------------------------
#               NeutronIrradiation class
# ----------------------------------------------------

@pytest.mark.parametrize('start, stop, n_cm2, n_cm2_s', [
    ('2017-01-01 00:00:00', '2017-01-01 12:00:00', None, 1e12),
    ('2017-01-01 00:00:00', '2017-01-01 12:00:00', 1e15, None),
    ('2017-01-01 00:00:00', '2017-01-01 00:00:00', 1e15, None)
])
def test_irradiation_init(start, stop, n_cm2, n_cm2_s):
    """Test valid inits for NeutronIrradiation"""

    ni = isotope.NeutronIrradiation(start, stop, n_cm2=n_cm2, n_cm2_s=n_cm2_s)
    assert hasattr(ni, 'n_cm2')


@pytest.mark.parametrize('start, stop, n_cm2, n_cm2_s, error', [
    ('2017-01-01 00:00:00', '2017-01-01 12:00:00', 1e15, 1e12, ValueError),
    ('2017-01-01 12:00:00', '2017-01-01 00:00:00', None, 1e12, ValueError)
])
def test_irradiation_bad_init(start, stop, n_cm2, n_cm2_s, error):
    """Test invalid inits for NeutronIrradiation"""

    with pytest.raises(error):
        isotope.NeutronIrradiation(start, stop, n_cm2=n_cm2, n_cm2_s=n_cm2_s)


@pytest.fixture(params=[
    ('Cs-133', 'Cs-134'),
    ('Hg-202', 'Hg-203'),
    ('Na-23', 'Na-24')
])
def activation_pair(request):
    return isotope.Isotope(request.param[0]), isotope.Isotope(request.param[1])


def test_irradiation_activate_pulse(activation_pair):
    """Test NeutronIrradiation.activate() for duration = 0"""

    start = datetime.datetime.now()
    stop = start
    n_cm2 = 1e15

    iso0, iso1 = activation_pair
    barns = 1

    iq0 = isotope.IsotopeQuantity(iso0, date=start, atoms=1e24)
    expected_atoms = n_cm2 * barns * 1e-24 * iq0.ref_atoms

    ni = isotope.NeutronIrradiation(start, stop, n_cm2=n_cm2)

    # forward calculation
    iq1 = ni.activate(barns, initial_iso_q=iq0, activated_iso=iso1)
    assert iq1.ref_date == stop
    assert np.isclose(iq1.ref_atoms, expected_atoms)

    # backward calculation
    iq0a = ni.activate(barns, activated_iso_q=iq1, initial_iso=iso0)
    assert iq0a.ref_date == start
    assert np.isclose(iq0a.ref_atoms, iq0.ref_atoms)


def test_irradiation_activate_extended(activation_pair):
    """Test NeutronIrradiation.activate() with decay"""

    start = datetime.datetime.now()
    t_irr = 3600 * 12
    stop = start + datetime.timedelta(seconds=t_irr)
    n_cm2_s = 1e11

    iso0, iso1 = activation_pair
    barns = 1

    iq0 = isotope.IsotopeQuantity(iso0, date=start, atoms=1e24)
    expected_atoms = n_cm2_s * barns * 1e-24 * iq0.ref_atoms * (
        1 - np.exp(-iso1.decay_const * t_irr))

    ni = isotope.NeutronIrradiation(start, stop, n_cm2_s=n_cm2_s)

    # forward calculation
    iq1 = ni.activate(barns, initial_iso_q=iq0, activated_iso=iso1)
    assert iq1.ref_date == stop
    assert np.isclose(iq1.bq_at(iq1.ref_date), expected_atoms)

    # backward calculation
    iq0a = ni.activate(barns, initial_iso=iso0, activated_iso_q=iq1)
    assert iq0a.ref_date == start
    assert np.isclose(iq0a.ref_atoms, iq0.ref_atoms)


def test_irradiation_activate_errors():
    """Test NeutronIrradiation.activate() bad args"""

    iso0 = isotope.Isotope('Na-23')
    iso1 = isotope.Isotope('Na-24')
    iq0 = isotope.IsotopeQuantity(iso0, atoms=1e24)
    iq1 = isotope.IsotopeQuantity(iso1, atoms=1e24)

    start = datetime.datetime.now()
    stop = start
    n_cm2 = 1e15
    ni = isotope.NeutronIrradiation(start, stop, n_cm2=n_cm2)

    barns = 10

    with pytest.raises(isotope.IsotopeError):
        ni.activate(
            barns, initial_iso_q=iq0, activated_iso_q=iq1, initial_iso=iso0)
    with pytest.raises(isotope.IsotopeError):
        ni.activate(barns, initial_iso_q=iq0)
    with pytest.raises(isotope.IsotopeError):
        ni.activate(barns, activated_iso_q=iq0)

    iso2 = isotope.Isotope('Na-25')
    with pytest.raises(NotImplementedError):
        ni.activate(barns, initial_iso_q=iq1, activated_iso=iso2)
