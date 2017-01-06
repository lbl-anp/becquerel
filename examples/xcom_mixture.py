"""Plot cross sections from XCOM."""

from __future__ import print_function
import os
import matplotlib.pyplot as plt
from becquerel.tools import xcom
from xcom_element import plot_xcom


if __name__ == '__main__':
    MIXTURE = ['H2O 0.9', 'NaCl 0.1']
    xd = xcom.XCOMQuery(e_range=[1., 100000.], mixture=MIXTURE)
    print(xd)
    plot_xcom(xd, ' '.join(MIXTURE))

    plt.figure(figsize=(8, 9.2))
    ax = plt.subplot(111)
    ax.axis('off')
    ax.set_position([0., 0., 1., 1.])
    img = plt.imread(
        os.path.join(os.path.dirname(__file__), 'xcom_mixture.png'))
    ax.imshow(img)

    plt.show()
