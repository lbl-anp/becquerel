
from __future__ import print_function
import datetime
from dateutil.parser import parse as dateutil_parse
from six import string_types
import copy
import numpy as np
from uncertainties import ufloat
from becquerel.tools.isotope import Isotope
from becquerel.tools.isotope_qty import IsotopeQuantity, NeutronIrradiation
from becquerel.tools.isotope_qty import IsotopeQuantityError, UCI_TO_BQ, N_AV
from becquerel.tools.isotope_qty import NeutronIrradiationError
from becquerel.tools.isotope_qty import decay_normalize
from becquerel.tools.isotope_qty import decay_normalize_spectra
from becquerel import Spectrum
import pytest


# ----------------------------------------------------
#               IsotopeQuantity class
# ----------------------------------------------------

@pytest.fixture(params=(
    'Bi-212',
    'Cs-137',
    'K-40',
    'Cs-134',
    'N-13',
    'Rn-220',
    'Tc-99m',
    'Tl-208',
    'Th-232',
))
def radioisotope(request):
    return Isotope(request.param)


@pytest.fixture(params=(
    'H-2',
    'Cs-133',
    'Pb-208',
    'La-138',
    'Eu-151',
    'Ge-76',
))
def stable_isotope(request):
    return Isotope(request.param)


@pytest.fixture(params=[
    {'bq': 10.047 * UCI_TO_BQ},
    {'uci': 10.047},
    {'uci': ufloat(10.047, 0.025)},
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

    iq = IsotopeQuantity(radioisotope, date=iq_date, **iq_kwargs)
    assert iq.isotope is radioisotope
    assert iq.half_life == radioisotope.half_life
    assert iq.decay_const == radioisotope.decay_const

    # check string input
    iq = IsotopeQuantity(str(radioisotope), date=iq_date, **iq_kwargs)
    assert iq.isotope == radioisotope
    assert iq.half_life == radioisotope.half_life
    assert iq.decay_const == radioisotope.decay_const


def test_isotopequantity_init_stable(stable_isotope, iq_date, iq_kwargs):
    """Test IsotopeQuantity.__init__() for radioactive isotopes"""

    if 'bq' in iq_kwargs or 'uci' in iq_kwargs:
        with pytest.raises(IsotopeQuantityError):
            iq = IsotopeQuantity(stable_isotope, date=iq_date, **iq_kwargs)
        return None
    iq = IsotopeQuantity(stable_isotope, date=iq_date, **iq_kwargs)
    assert iq.isotope is stable_isotope
    assert iq.half_life == stable_isotope.half_life
    assert iq.decay_const == stable_isotope.decay_const


def test_isotopequantity_ref_atoms_rad(radioisotope, iq_kwargs):
    """Test IsotopeQuantity.ref_atoms for a radioactive isotope"""

    iq = IsotopeQuantity(radioisotope, **iq_kwargs)
    if 'atoms' in iq_kwargs:
        assert iq.ref_atoms == iq_kwargs['atoms']
    elif 'g' in iq_kwargs:
        assert iq.ref_atoms == (iq_kwargs['g'] / radioisotope.A * N_AV)
    elif 'bq' in iq_kwargs:
        assert iq.ref_atoms == iq_kwargs['bq'] / radioisotope.decay_const
    else:
        assert iq.ref_atoms == (
            iq_kwargs['uci'] * UCI_TO_BQ / radioisotope.decay_const)


def test_isotopequantity_ref_atoms_stable(stable_isotope):
    """Test IsotopeQuantity.ref_atoms for a stable isotope"""

    atoms = 1e24
    iq = IsotopeQuantity(stable_isotope, atoms=atoms)
    assert iq.ref_atoms == atoms

    g = 1e-5
    iq = IsotopeQuantity(stable_isotope, g=g)
    assert iq.ref_atoms == g / stable_isotope.A * N_AV


def test_isotopequantity_ref_date_rad(radioisotope, iq_date):
    """Test IsotopeQuantity.ref_date for a radioisotope"""

    iq = IsotopeQuantity(radioisotope, date=iq_date, atoms=1e24)
    if isinstance(iq_date, datetime.datetime):
        assert iq.ref_date == iq_date
    elif isinstance(iq_date, string_types):
        assert iq.ref_date == dateutil_parse(iq_date)
    else:
        assert (datetime.datetime.now() - iq.ref_date).total_seconds() < 5


def test_isotopequantity_ref_date_stable(stable_isotope, iq_date):
    """Test IsotopeQuantity.ref_date for a stable isotope"""

    iq = IsotopeQuantity(stable_isotope, date=iq_date, atoms=1e24)
    if isinstance(iq_date, datetime.datetime):
        assert iq.ref_date == iq_date
    elif isinstance(iq_date, string_types):
        assert iq.ref_date == dateutil_parse(iq_date)
    else:
        assert (datetime.datetime.now() - iq.ref_date).total_seconds() < 5


@pytest.mark.parametrize('iso, date, kwargs, error', [
    (['Cs-137'], None, {'atoms': 1e24}, TypeError),
    ('Cs-137', 123, {'bq': 456}, TypeError),
    ('Cs-137', datetime.datetime.now(), {'asdf': 3}, IsotopeQuantityError),
    ('Cs-137', None, {'bq': -13.3}, ValueError)
])
def test_isotopequantity_bad_init(iso, date, kwargs, error):
    """Test errors from Isotope.__init__()"""

    with pytest.raises(error):
        IsotopeQuantity(iso, date=date, **kwargs)


@pytest.fixture
def iq(radioisotope):
    """An IsotopeQuantity object"""

    date = datetime.datetime.now()
    kwargs = {'uci': 10.047}
    return IsotopeQuantity(radioisotope, date=date, **kwargs)


def test_isotopequantity_at_methods(iq):
    """Test IsotopeQuantity.*_at()"""

    half_life = iq.half_life

    assert iq.atoms_at(iq.ref_date) == iq.ref_atoms
    assert np.isclose(iq.g_at(iq.ref_date), iq.ref_atoms * iq.isotope.A / N_AV)
    assert np.isclose(iq.bq_at(iq.ref_date), iq.ref_atoms * iq.decay_const)
    assert np.isclose(
        iq.uci_at(iq.ref_date), iq.ref_atoms * iq.decay_const / UCI_TO_BQ)

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

    ref_qty = getattr(iq, kw + '_at')(iq.ref_date)

    kwarg = {kw: ref_qty}
    assert iq.time_when(**kwarg) == iq.ref_date

    if iq.half_life > 1000 * 3.156e7:
        # avoid overflow errors in test calculations
        # just make sure the method doesn't error
        kwarg = {kw: ref_qty * (1 - 1e-9)}
        iq.time_when(**kwarg)

        kwarg = {kw: ref_qty * (1 + 1e-9)}
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

    iq = IsotopeQuantity(stable_isotope, g=10)
    with pytest.raises(IsotopeQuantityError):
        iq.time_when(g=5)


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

        assert np.isclose(iq.uci_from(t0, t1),
                          iq.atoms_at(t1) / iq.half_life / UCI_TO_BQ)


def test_isotopequantity_decays_during(iq):
    """Test IsotopeQuantity.*_during()"""

    if iq.half_life > 1000 * 3.156e7:
        # avoid overflow errors in test calculations
        pass
    else:
        dt_s = iq.half_life
        t0 = iq.ref_date
        t1 = t0 + datetime.timedelta(seconds=dt_s)
        spec = Spectrum(np.zeros(256), start_time=t0, stop_time=t1)

        assert np.isclose(iq.decays_during(spec), iq.atoms_at(t0) / 2)
        assert np.isclose(iq.bq_during(spec), iq.decays_during(spec) / dt_s)
        assert np.isclose(iq.uci_during(spec), iq.bq_during(spec) / UCI_TO_BQ)


def test_isotopequantity_eq(iq):
    """Test IsotopeQuantity equality magic method"""

    iq2 = copy.deepcopy(iq)
    assert iq == iq2

    dt_s = min(iq.half_life, 10 * 86400)
    date = iq.ref_date + datetime.timedelta(seconds=dt_s)
    iq3 = IsotopeQuantity(iq.isotope, date=date, atoms=iq.atoms_at(date))
    assert iq == iq3


@pytest.mark.parametrize('f', [2, 0.5, 3.14])
def test_isotopequantity_mul_div(iq, f):
    """Test IsotopeQuantity multiplication and division magic methods"""

    iq2 = iq * f
    assert iq2.isotope == iq.isotope
    assert iq2.ref_date == iq.ref_date
    assert iq2.ref_atoms == iq.ref_atoms * f

    iq3 = iq / f
    assert iq3.isotope == iq.isotope
    assert iq3.ref_date == iq.ref_date
    assert iq3.ref_atoms == iq.ref_atoms / f


def test_isotopequantity_from_decays(iq):
    """Test IsotopeQuantity.from_decays instantiation"""

    start = iq.ref_date
    stop = start + datetime.timedelta(seconds=3600)

    # qualitative
    IsotopeQuantity.from_decays(
        iq.isotope, n_decays=1000, start_time=start, stop_time=stop)

    # quantitative
    if iq.half_life < 1000 * 3.156e7:  # floating point precision #65
        n = iq.decays_from(start, stop)
        iq2 = IsotopeQuantity.from_decays(
            iq.isotope, n_decays=n, start_time=start, stop_time=stop)
        assert np.isclose(iq.atoms_at(start), iq2.atoms_at(start))


def test_isotopequantity_from_comparison(iq):
    """Test IsotopeQuantity.from_comparison calculation"""

    counts1 = 1e3
    start = iq.ref_date
    stop = start + datetime.timedelta(seconds=3600)
    interval1 = (start, stop)
    iq2 = IsotopeQuantity.from_comparison(
        iq, counts1, interval1,
        counts1, interval1)
    assert iq2 == iq

    f = 3.1
    counts2 = counts1 * f
    iq3 = IsotopeQuantity.from_comparison(
        iq, counts1, interval1,
        counts2, interval1)
    assert iq3 == iq * f

    if iq.half_life < 1000 * 3.156e7:
        dt = datetime.timedelta(seconds=iq.half_life)
        interval2 = (start + dt, stop + dt)
        iq4 = IsotopeQuantity.from_comparison(
            iq, counts1, interval1,
            counts1, interval2)
        assert iq4 == iq * 2

        iq5 = IsotopeQuantity.from_comparison(
            iq, counts1, interval1,
            counts2, interval2)
        assert iq5 == iq * 2 * f

        interval3 = (start - dt, stop - dt)
        iq6 = IsotopeQuantity.from_comparison(
            iq, counts1, interval1,
            counts1, interval3)
        assert iq6 == iq / 2


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

    ni = NeutronIrradiation(start, stop, n_cm2=n_cm2, n_cm2_s=n_cm2_s)
    assert hasattr(ni, 'n_cm2')


@pytest.mark.parametrize('start, stop, n_cm2, n_cm2_s, error', [
    ('2017-01-01 00:00:00', '2017-01-01 12:00:00', 1e15, 1e12, ValueError),
    ('2017-01-01 12:00:00', '2017-01-01 00:00:00', None, 1e12, ValueError)
])
def test_irradiation_bad_init(start, stop, n_cm2, n_cm2_s, error):
    """Test invalid inits for NeutronIrradiation"""

    with pytest.raises(error):
        NeutronIrradiation(start, stop, n_cm2=n_cm2, n_cm2_s=n_cm2_s)


@pytest.fixture(params=[
    ('Cs-133', 'Cs-134'),
    ('Hg-202', 'Hg-203'),
    ('Na-23', 'Na-24')
])
def activation_pair(request):
    return Isotope(request.param[0]), Isotope(request.param[1])


def test_irradiation_activate_pulse(activation_pair):
    """Test NeutronIrradiation.activate() for duration = 0"""

    start = datetime.datetime.now()
    stop = start
    n_cm2 = 1e15

    iso0, iso1 = activation_pair
    barns = 1

    iq0 = IsotopeQuantity(iso0, date=start, atoms=1e24)
    expected_atoms = n_cm2 * barns * 1e-24 * iq0.ref_atoms

    ni = NeutronIrradiation(start, stop, n_cm2=n_cm2)

    # forward calculation
    iq1 = ni.activate(barns, initial=iq0, activated=iso1)
    assert iq1.ref_date == stop
    assert np.isclose(iq1.ref_atoms, expected_atoms)

    # backward calculation
    iq0a = ni.activate(barns, activated=iq1, initial=iso0)
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

    iq0 = IsotopeQuantity(iso0, date=start, atoms=1e24)
    expected_atoms = n_cm2_s * barns * 1e-24 * iq0.ref_atoms * (
        1 - np.exp(-iso1.decay_const * t_irr))

    ni = NeutronIrradiation(start, stop, n_cm2_s=n_cm2_s)

    # forward calculation
    iq1 = ni.activate(barns, initial=iq0, activated=iso1)
    assert iq1.ref_date == stop
    assert np.isclose(iq1.bq_at(iq1.ref_date), expected_atoms)

    # backward calculation
    iq0a = ni.activate(barns, initial=iso0, activated=iq1)
    assert iq0a.ref_date == start
    assert np.isclose(iq0a.ref_atoms, iq0.ref_atoms)


def test_irradiation_activate_errors():
    """Test NeutronIrradiation.activate() bad args"""

    iso0 = Isotope('Na-23')
    iso1 = Isotope('Na-24')
    iq0 = IsotopeQuantity(iso0, atoms=1e24)
    iq1 = IsotopeQuantity(iso1, atoms=1e24)

    start = datetime.datetime.now()
    stop = start
    n_cm2 = 1e15
    ni = NeutronIrradiation(start, stop, n_cm2=n_cm2)

    barns = 10

    with pytest.raises(NeutronIrradiationError):
        ni.activate(barns, initial=iq0, activated=iq1)
    with pytest.raises(NeutronIrradiationError):
        ni.activate(barns, initial=iso0, activated=iso1)
    with pytest.raises(TypeError):
        ni.activate(barns, initial=iq0, activated='asdf')

    iso2 = Isotope('Na-25')
    with pytest.raises(NotImplementedError):
        ni.activate(barns, initial=iq1, activated=iso2)


@pytest.mark.parametrize('start, stop, n_cm2, n_cm2_s', [
    ('2017-01-01 00:00:00', '2017-01-01 12:00:00', None, 1e12),
    ('2017-01-01 00:00:00', '2017-01-01 12:00:00', 1e15, None),
    ('2017-01-01 00:00:00', '2017-01-01 00:00:00', 1e15, None)
])
def test_irradiation_str(start, stop, n_cm2, n_cm2_s):
    """Test NeutronIrradiation string representation"""

    ni = NeutronIrradiation(start, stop, n_cm2=n_cm2, n_cm2_s=n_cm2_s)
    print(str(ni))


# ----------------------------------------------------
#               decay_normalize
# ----------------------------------------------------

def test_decay_normalize(radioisotope):
    """Test decay_normalize()"""

    now = datetime.datetime.now()
    interval1 = (now, now + datetime.timedelta(hours=1))
    assert np.isclose(decay_normalize(radioisotope, interval1, interval1), 1)

    # avoid numerical issues with large half-lives (#65)
    # and underflow with small half-lives
    # note: 2^(-(1 day) / (85 seconds)) ~ sys.float_info.min
    if 90 < radioisotope.half_life < 1000 * 3.156e7:
        interval2 = (now + datetime.timedelta(days=1),
                     now + datetime.timedelta(days=1, hours=1))
        assert decay_normalize(radioisotope, interval1, interval2) > 1
        assert decay_normalize(radioisotope, interval2, interval1) < 1


def test_decay_normalize_errors(radioisotope):
    """Test errors in decay_normalize()"""

    now = datetime.datetime.now()
    interval1 = (now, now + datetime.timedelta(hours=1))
    interval2 = (now + datetime.timedelta(days=1),
                 now + datetime.timedelta(days=1, hours=1))
    with pytest.raises(IsotopeQuantityError):
        decay_normalize(radioisotope, [now], interval2)
    with pytest.raises(IsotopeQuantityError):
        decay_normalize(radioisotope, interval1, [now, now, now])
    with pytest.raises(ValueError):
        decay_normalize(radioisotope, (interval1[1], interval1[0]), interval2)
    with pytest.raises(ValueError):
        decay_normalize(radioisotope, interval1, (interval2[1], interval2[0]))


def test_decay_normalize_spectra(radioisotope):
    """Test decay_normalize_spectra()"""

    t0 = datetime.datetime.now()
    t1 = t0 + datetime.timedelta(hours=1)
    spec1 = Spectrum(np.zeros(256), start_time=t0, stop_time=t1)
    assert np.isclose(decay_normalize_spectra(radioisotope, spec1, spec1), 1)

    # avoid numerical issues with large half-lives (#65)
    # and underflow with small half-lives
    # note: 2^(-(1 day) / (85 seconds)) ~ sys.float_info.min
    if radioisotope.half_life < 90 or radioisotope.half_life > 1000 * 3.156e7:
        pass
    else:
        t2 = t0 + datetime.timedelta(days=1)
        t3 = t1 + datetime.timedelta(days=1)
        spec2 = Spectrum(np.zeros(256), start_time=t2, stop_time=t3)
        assert decay_normalize_spectra(radioisotope, spec1, spec2) > 1
        assert decay_normalize_spectra(radioisotope, spec2, spec1) < 1
