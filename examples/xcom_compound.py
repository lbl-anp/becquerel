"""Plot cross sections from XCOM."""

from __future__ import print_function
import os
import matplotlib.pyplot as plt
from becquerel.tools import xcom
from xcom_element import plot_xcom


if __name__ == '__main__':
    COMPOUND = 'H2O'
    xd = xcom.XCOMQuery(e_range=[1., 100000.], compound=COMPOUND)
    print(xd)
    plot_xcom(xd, COMPOUND)

    plt.figure(figsize=(8, 9.2))
    ax = plt.subplot(111)
    ax.axis('off')
    ax.set_position([0., 0., 1., 1.])
    img = plt.imread(
        os.path.join(os.path.dirname(__file__), 'xcom_compound.png'))
    ax.imshow(img)

    plt.show()
