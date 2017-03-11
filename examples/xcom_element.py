"""Plot cross sections from XCOM."""

from __future__ import print_function
import os
import matplotlib.pyplot as plt
from becquerel.tools import xcom
import pandas as pd
pd.set_option('display.width', 120)


PLOT_KWARGS = {
    'total_w_coh': {
        'label': xcom.COLUMNS_LONG['total_w_coh'],
        'color': 'black',
        'lw': 3,
        'ls': '-',
    },
    'total_wo_coh': {
        'label': xcom.COLUMNS_LONG['total_wo_coh'],
        'color': 'green',
        'lw': 3,
        'ls': '--',
    },
    'coherent': {
        'label': xcom.COLUMNS_LONG['coherent'],
        'color': 'orange',
        'lw': 1,
        'ls': '--',
    },
    'incoherent': {
        'label': xcom.COLUMNS_LONG['incoherent'],
        'color': 'blue',
        'lw': 1,
        'ls': ':',
    },
    'photoelec': {
        'label': xcom.COLUMNS_LONG['photoelec'],
        'color': 'magenta',
        'lw': 1,
        'ls': '-',
    },
    'pair_nuc': {
        'label': xcom.COLUMNS_LONG['pair_nuc'],
        'color': 'cyan',
        'lw': 1,
        'ls': '-',
    },
    'pair_elec': {
        'label': xcom.COLUMNS_LONG['pair_elec'],
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
    erg = xcom_data['energy'] / 1000.  # MeV
    for field in ['total_w_coh', 'total_wo_coh', 'coherent', 'incoherent',
                  'photoelec', 'pair_nuc', 'pair_elec']:
        plt.loglog(erg, xcom_data[field], **PLOT_KWARGS[field])
    plt.legend(
        prop={'size': 8}, loc='upper left',
        bbox_to_anchor=(0., -0.35, 1., 0.25))
    plt.title('XCOMQuery result:\n' + title)
    plt.xlabel('Photon Energy (MeV)')
    plt.ylabel(r'Cross Section (cm$^2$/g)')


if __name__ == '__main__':
    SYMBOL = 'Pb'
    xd = xcom.xcom_data(SYMBOL, e_range_kev=[1., 100000.])
    print(xd)
    plot_xcom(xd, SYMBOL)

    plt.figure(figsize=(8, 9.2))
    plt.title('XCOM website screenshot:')
    ax = plt.subplot(111)
    ax.axis('off')
    ax.set_position([0., 0., 1., 0.9])
    img = plt.imread(
        os.path.join(os.path.dirname(__file__), 'xcom_element.png'))
    ax.imshow(img)

    plt.show()
