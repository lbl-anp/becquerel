"""Make chart of nuclides using NNDC data."""

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from becquerel.tools import nndc


def colorscale_half_life(half_life):
    """Color scale to mimic NNDC's nuclear chart color scale."""
    if half_life is None:
        return '#E1E1E1'
    elif half_life < 1e-15:
        return '#FF9473'
    elif half_life < 1e-7:
        return '#F7BDDD'
    elif half_life < 1e-6:
        return '#FFC6A5'
    elif half_life < 1e-5:
        return '#FFE7C6'
    elif half_life < 1e-4:
        return '#FFFF9B'
    elif half_life < 1e-3:
        return '#FFFF0C'
    elif half_life < 1e-2:
        return '#E7F684'
    elif half_life < 1e-1:
        return '#D6EF38'
    elif half_life < 1e0:
        return '#ADDE63'
    elif half_life < 1e1:
        return '#53B552'
    elif half_life < 1e2:
        return '#64BDB5'
    elif half_life < 1e3:
        return '#63C6DE'
    elif half_life < 1e4:
        return '#03A5C6'
    elif half_life < 1e5:
        return '#0A9A94'
    elif half_life < 1e7:
        return '#0284A5'
    elif half_life < 1e10:
        return '#3152A5'
    elif half_life < 1e15:
        return '#29016B'
    elif half_life > 1e15:
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
        half_life = data[isotope]['T1/2 (s)']
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
