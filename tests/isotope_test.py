"""Test isotope.py classes."""

import numpy as np
from becquerel.tools import element
from becquerel.tools import isotope
import pytest

TEST_ISOTOPES = [
    ("H-3", "H", 3, ""),
    ("He-4", "He", 4, ""),
    ("K-40", "K", 40, ""),
    ("Co-60", "Co", 60, ""),
    ("U-238", "U", 238, ""),
    ("Pu-244", "Pu", 244, ""),
    ("Tc-99m", "Tc", 99, "m"),
    ("Tc-99m", "Tc", "99", "m"),
    ("Tc-99m", "Tc", 99, 1),
    ("Pa-234m", "Pa", 234, "m"),
    ("Hf-178m1", "Hf", 178, "m1"),
    ("Hf-178m2", "Hf", 178, "m2"),
    ("Hf-178m3", "Hf", 178, "m3"),
    ("Hf-178m3", "Hf", 178, 3),
]


# ----------------------------------------------------
#                   Isotope class
# ----------------------------------------------------


@pytest.mark.parametrize("iso_str, sym, A, m", TEST_ISOTOPES)
def test_isotope_init_args(iso_str, sym, A, m):
    """Test Isotope init with 2 or 3 args, depending on whether m is None."""
    sym = str(sym)
    name = element.element_name(sym)
    Z = element.element_z(sym)
    A = int(A)
    elems = [sym, sym.lower(), sym.upper(), name, name.lower(), name.upper(), Z, str(Z)]
    mass_numbers = [A, str(A)]
    if isinstance(m, int):
        if m == 0:
            m = ""
        else:
            m = f"m{m}"
    if m == "":
        isomer_levels = [None, "", 0]
    elif m.lower() == "m":
        isomer_levels = ["m", "M", 1]
    elif m.lower() == "m1":
        isomer_levels = ["m1", "M1"]
    else:
        isomer_levels = [m.lower(), m.upper(), int(m[1:])]
    for elem in elems:
        for mass in mass_numbers:
            for isomer in isomer_levels:
                args_list = [(elem, mass, isomer)]
                if isomer == "" or isomer is None:
                    args_list.append((elem, mass))
                for args in args_list:
                    print("")
                    print(args)
                    i = isotope.Isotope(*args)
                    print(i)
                    assert i.symbol == sym
                    assert i.A == A
                    assert i.m == m


@pytest.mark.parametrize(
    "elem, A, m",
    [
        ("Xx", 45, ""),  # invalid element symbol
        (119, 250, ""),  # invalid Z (not in range 1..118)
        ("H", -1, ""),  # invalid A (negative)
        ("H", "AA", ""),  # invalid A (string that cannot be converted to int)
        ("Ge", 30, ""),  # invalid N (negative)
        ("Tc", 99, "n"),  # invalid m (does not start with 'm')
        ("Tc", 99, "m-1"),  # invalid m (negative, str)
        ("Tc", 99, -1),  # invalid m (negative, int)
        ("Tc", 99, 1.0),  # invalid m (floating point)
    ],
)
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
        isotope.Isotope("Tc", 99, "m", 0)


@pytest.mark.parametrize(
    "sym1, A1, m1, sym2, A2, m2, equal",
    [
        ("Tc", 99, "", "Tc", 99, "", True),  # symbol and A only
        ("Tc", 99, "m", "Tc", 99, "m", True),  # symbol, A, and m
        ("Hf", 178, "m", "Hf", 178, "m1", True),  # m and m1 are equivalent
        ("Ge", 68, "", "Ga", 68, "", False),  # symbols differ
        ("Ge", 68, "", "Ge", 69, "", False),  # masses differ
        ("Ge", 68, "", "Ge", 69, "m", False),  # metastable states differ
        ("Ge", 68, "m1", "Ge", 69, "m2", False),  # metastable states differ
    ],
)
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
    i1 = isotope.Isotope("Tc", 99, "m")
    with pytest.raises(TypeError):
        assert i1 == "Tc-99m"


@pytest.mark.parametrize("iso_str, sym, A, m", TEST_ISOTOPES)
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
            mass_number += f"m{m}"
        else:
            mass_number += m
    expected = isotope.Isotope(sym, A, m)
    for name in [sym, element.element_name(sym)]:
        iso_tests = [
            name + "-" + mass_number,
            name + mass_number,
            mass_number + "-" + name,
            mass_number + name,
        ]
        for iso in iso_tests:
            for iso2 in [iso, iso.upper(), iso.lower()]:
                print("")
                print(f"{sym}-{mass_number}: {iso2}")
                i = isotope.Isotope(iso2)
                print(i)
                assert i == expected


@pytest.mark.parametrize(
    "iso_str, raises",
    [
        ("He_4", True),  # underscore
        ("He--4", True),  # hyphens
        ("4399", True),  # all numbers
        ("abdc", True),  # all letters
        ("Tc-99m3m", True),  # too many ms
        ("55mN", False),  # ambiguous but valid (returns Mn-55, not N-55m)
        ("55m2N", False),  # like previous but unambiguous (returns N-55m2)
        ("24Mg", False),  # unambiguous because G is not an element
        ("24m2G", True),  # like previous but invalid
        ("2He4", True),  # invalid mass number
        ("2He-4", True),  # invalid mass number
        ("He2-4", True),  # invalid mass number
        ("Xz-90", True),  # invalid element
        ("H-AA", True),  # invalid A (string that cannot be converted to int)
        ("H-20Xm1", True),  # invalid A (string that cannot be converted to int)
        ("Tc-99n", True),  # invalid m (does not start with 'm')
        ("Hf-178n3", True),  # invalid m (does not start with 'm')
        ("Tc-99m1.0", True),  # invalid m (floating point)
    ],
)
def test_isotope_init_str_exceptions(iso_str, raises):
    """Test Isotope init exceptions with one (string) argument."""
    if raises:
        with pytest.raises(isotope.IsotopeError):
            isotope.Isotope(iso_str)
    else:
        isotope.Isotope(iso_str)


@pytest.mark.parametrize("iso_str, sym, A, m", TEST_ISOTOPES)
def test_isotope_str(iso_str, sym, A, m):
    """Test Isotope.__str__()."""
    i = isotope.Isotope(sym, A, m)
    print(str(i), iso_str)
    assert str(i) == iso_str


ISOTOPE_PROPERTIES = [
    (
        "H-3",
        (
            3.888e8,
            False,
            None,
            "1/2+",
            0.0,
            14.94980957,
            [["B-"], [100.0]],
        ),
    ),
    ("He-4", (np.inf, True, 99.999866, "0+", 0.0, 2.4249, [[], []])),
    (
        "K-40",
        (3.938e16, False, 0.0117, "4-", 0.0, -33.53540, [["B-", "EC"], [89.28, 10.72]]),
    ),
    ("Co-60", (1.663e8, False, None, "5+", 0.0, -61.6503, [["B-"], [100.0]])),
    (
        "U-238",
        (1.41e17, False, 99.2742, "0+", 0.0, 47.3077, [["A", "SF"], [100.00, 5.4e-5]]),
    ),
    (
        "Pu-244",
        (2.525e15, False, None, "0+", 0.0, 59.8060, [["A", "SF"], [99.88, 0.12]]),
    ),
    (
        "Tc-99m",
        (
            21624.12,
            False,
            None,
            "1/2-",
            0.1427,
            -87.1851,
            [["IT", "B-"], [100.0, 3.7e-3]],
        ),
    ),
    (
        "Pa-234m",
        (69.54, False, None, "(0-)", 0.0739, 40.413, [["IT", "B-"], [0.16, 99.84]]),
    ),
    ("Hf-178", (np.inf, True, 27.28, "0+", 0.0, -52.4352, [[], []])),
    ("Hf-178m1", (4.0, False, None, "8-", 1.1474, -51.2878, [["IT"], [100.0]])),
    ("Hf-178m2", (9.783e8, False, None, "16+", 2.4461, -49.9891, [["IT"], [100.0]])),
]


@pytest.mark.webtest
@pytest.mark.parametrize("iso_str, props", ISOTOPE_PROPERTIES)
def test_isotope_properties(iso_str, props):
    """Test that isotope properties are correct."""
    i = isotope.Isotope(iso_str)
    half_life, is_stable, abundance, j_pi, energy_level, mass_excess, modes = props
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
    print("mass excess:", i.mass_excess, type(i.mass_excess))
    if mass_excess is None:
        assert i.mass_excess is None
    else:
        print("mass excess:", i.mass_excess.nominal_value)
        print("mass excess:", mass_excess)
        print(np.isclose(mass_excess, mass_excess))
        print(np.isclose(mass_excess, i.mass_excess.nominal_value))
        print(np.isclose(mass_excess, (i.mass_excess).nominal_value))
        assert np.isclose(i.mass_excess.nominal_value, mass_excess)
    assert set(i.decay_modes[0]) == set(modes[0])
    assert set(i.decay_modes[1]) == set(modes[1])
