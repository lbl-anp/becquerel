"""Plot cross sections from XCOM."""

from __future__ import print_function
import os
import matplotlib.pyplot as plt
from becquerel.tools import xcom


ENERGY_UNITS = 'MeV'

PLOT_KWARGS = {
    'T+C': {
        'label': xcom.COLUMNS_LONG['T+C'],
        'color': 'black',
        'lw': 3,
        'ls': '-',
    },
    'T-C': {
        'label': xcom.COLUMNS_LONG['T-C'],
        'color': 'green',
        'lw': 3,
        'ls': '--',
    },
    'C': {
        'label': xcom.COLUMNS_LONG['C'],
        'color': 'orange',
        'lw': 1,
        'ls': '--',
    },
    'I': {
        'label': xcom.COLUMNS_LONG['I'],
        'color': 'blue',
        'lw': 1,
        'ls': ':',
    },
    'PA': {
        'label': xcom.COLUMNS_LONG['PA'],
        'color': 'magenta',
        'lw': 1,
        'ls': '-',
    },
    'PPN': {
        'label': xcom.COLUMNS_LONG['PPN'],
        'color': 'cyan',
        'lw': 1,
        'ls': '-',
    },
    'PPE': {
        'label': xcom.COLUMNS_LONG['PPE'],
        'color': 'cyan',
        'lw': 1,
        'ls': '--',
    },
}


def plot_xcom(xcom_data, title):
    """Plot the XCOM data in the same fashion as the website."""
    fig = plt.figure(figsize=(7.4, 8.8))
    axis = fig.gca()
    axis.set_position([0.13, 0.25, 0.62, 0.64])
    erg = [x.to(ENERGY_UNITS).magnitude for x in xcom_data['energy']]
    for field in ['T+C', 'T-C', 'C', 'I', 'PA', 'PPN', 'PPE']:
        xs = [x.magnitude for x in xcom_data[field]]
        plt.loglog(erg, xs, **PLOT_KWARGS[field])
    plt.legend(
        prop={'size': 8}, loc='upper left',
        bbox_to_anchor=(0., -0.35, 1., 0.25))
    plt.title(title)
    plt.xlabel('Photon Energy ({})'.format(ENERGY_UNITS))
    plt.ylabel(u'Cross Section ({:~P})'.format(xcom_data['C'][0].units))


if __name__ == '__main__':
    SYMBOL = 'Pb'
    xd = xcom.XCOMQuery(SYMBOL, e_range=[1., 100000.])
    print(xd)
    plot_xcom(xd, SYMBOL)

    plt.figure(figsize=(8, 9.2))
    ax = plt.subplot(111)
    ax.axis('off')
    ax.set_position([0., 0., 1., 1.])
    img = plt.imread(
        os.path.join(os.path.dirname(__file__), 'xcom_element.png'))
    ax.imshow(img)

    plt.show()
