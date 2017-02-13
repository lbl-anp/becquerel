"""Make chart of nuclides using NNDC data."""

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from becquerel.tools import nndc, units


def colorscale_half_life(half_life):
    """Color scale to mimic NNDC's nuclear chart color scale."""
    if half_life is None:
        return '#E1E1E1'
    elif half_life < 1e-15 * units.s:
        return '#FF9473'
    elif half_life < 1e-7 * units.s:
        return '#F7BDDD'
    elif half_life < 1e-6 * units.s:
        return '#FFC6A5'
    elif half_life < 1e-5 * units.s:
        return '#FFE7C6'
    elif half_life < 1e-4 * units.s:
        return '#FFFF9B'
    elif half_life < 1e-3 * units.s:
        return '#FFFF0C'
    elif half_life < 1e-2 * units.s:
        return '#E7F684'
    elif half_life < 1e-1 * units.s:
        return '#D6EF38'
    elif half_life < 1e0 * units.s:
        return '#ADDE63'
    elif half_life < 1e1 * units.s:
        return '#53B552'
    elif half_life < 1e2 * units.s:
        return '#64BDB5'
    elif half_life < 1e3 * units.s:
        return '#63C6DE'
    elif half_life < 1e4 * units.s:
        return '#03A5C6'
    elif half_life < 1e5 * units.s:
        return '#0A9A94'
    elif half_life < 1e7 * units.s:
        return '#0284A5'
    elif half_life < 1e10 * units.s:
        return '#3152A5'
    elif half_life < 1e15 * units.s:
        return '#29016B'
    elif half_life > 1e15 * units.s:
        return 'black'


Z_RANGE = (1, 25)
N_RANGE = (0, 42)

data = nndc.NuclearWalletCardQuery(z_range=Z_RANGE, n_range=N_RANGE)
print(data)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
for z in range(Z_RANGE[0], Z_RANGE[1] + 1):
    for n in range(N_RANGE[0], N_RANGE[1] + 1):
        print('')
        print('-' * 70)
        print(z, n)
        isotope = (data['Z'] == z) & (data['N'] == n) & (data['M'] == 0)
        half_life = data[isotope]['T1/2']
        if len(half_life) == 0:
            continue
        print('half_life:', half_life)
        half_life = max(half_life)
        print('half_life:', half_life)
        facecolor = 'white'
        facecolor = colorscale_half_life(half_life)
        print('facecolor:', half_life, facecolor)
        ax1.add_patch(
            patches.Rectangle(
                (n - 0.5, z - 0.5), 1, 1,
                edgecolor='white', facecolor=facecolor
            )
        )
plt.xlim(N_RANGE[0] - 0.5, N_RANGE[1] + 0.5)
plt.ylim(Z_RANGE[0] - 0.5, Z_RANGE[1] + 0.5)
plt.xlabel('N')
plt.ylabel('Z')
plt.show()
