"""Simple tools to perform HDF5 I/O."""

import pathlib
from typing import Union, Tuple
import h5py


def ensure_string(data):
    """Ensure the data are decoded to a string if they are bytes.

    Parameters
    ----------
    data : str or bytes
    """
    if isinstance(data, str):
        return data
    elif isinstance(data, bytes):
        return data.decode("ascii", "replace")
    else:
        raise TypeError(f"Data are neither bytes nor string: {type(data)}")


def is_h5_filename(name: str):
    """Return True if the lowercase filename ends with h5 or hdf5.

    Parameters
    ----------
    name : str
        The filename to examine.

    Returns
    -------
    is_h5 : bool
    """
    try:
        ext = pathlib.Path(name).suffix
        return ext.lower().endswith((".h5", ".hdf5"))
    except TypeError:
        return False


class open_h5:
    """Context manager to allow I/O given HDF5 filename, File, or Group."""

    def __init__(self, name: Union[str, h5py.File, h5py.Group], mode=None, **kwargs):
        """Initialize the context manager.

        Parameters
        ----------
        name : str, h5py.File, h5py.Group
            The filename or an open h5py File or Group.
        mode : str
            The I/O mode. Default is None.
        kwargs : dict
            Additional keyword arguments sent to h5py.File if initializing.
        """
        self._already_h5_obj = isinstance(name, (h5py.File, h5py.Group))
        if not self._already_h5_obj:
            assert is_h5_filename(name), "Filename must end in h5 or hdf5"
        self.name = name
        self.mode = mode
        self.kwargs = kwargs
        self.file = None

    def __enter__(self):
        """Open the file for I/O.

        Returns
        -------
        file : h5py.File, h5py.Group
            An open h5py File or Group.
        """
        if self._already_h5_obj:
            self.file = self.name
        else:
            self.file = h5py.File(self.name, mode=self.mode, **self.kwargs)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the file if it had to open it in __enter__."""
        if not self._already_h5_obj:
            self.file.close()


def write_h5(name: Union[str, h5py.File, h5py.Group], dsets: dict, attrs: dict) -> None:
    """Write the datasets and attributes to an HDF5 file or group.

    Parameters
    ----------
    name : str, h5py.File, h5py.Group
        The filename or an open h5py File or Group.
    dsets : dict
        Dictionary of data to be written as datasets.
    attrs : dict
        Dictionary of data to be written as attributes.
    """
    with open_h5(name, "w") as file:
        # write the datasets
        for key in dsets.keys():
            try:
                file.create_dataset(
                    key,
                    data=dsets[key],
                    compression="gzip",
                    compression_opts=9,
                )
            except TypeError:
                file.create_dataset(key, data=dsets[key])
        # write the attributes
        file.attrs.update(attrs)


def read_h5(name: Union[str, h5py.File, h5py.Group]) -> Tuple[dict, dict, list]:
    """Read the datasets and attributes from an HDF5 file or group.

    Parameters
    ----------
    name : str, h5py.File, h5py.Group
        The filename or an open h5py File or Group.

    Returns
    -------
    dsets : dict
        Dictionary of data to be written as datasets.
    attrs : dict
        Dictionary of data to be written as attributes.
    skipped : list
        List of keys of skipped items.
    """
    dsets = {}
    attrs = {}
    skipped = []
    with open_h5(name, "r") as file:
        # read the datasets
        for key in file.keys():
            # skip any non-datasets
            if not isinstance(file[key], h5py.Dataset):
                skipped.append(str(key))
            else:
                dsets[key] = file[key][()]
        # read the attributes
        attrs = dict(file.attrs)
    return dsets, attrs, skipped
