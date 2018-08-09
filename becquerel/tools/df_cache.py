"""A simple class for caching a pandas DataFrame."""

from __future__ import print_function
import os
import pandas as pd


class CacheError(Exception):
    """Problem fetching, saving, or retrieving cached data."""

    pass


class DataFrameCache(object):
    """Abstract base class for downloading, saving, and retrieving a DataFrame.

    Abstract methods:
      fetch: retrieve the DataFrame from a source.

    Properties (read-only):
      name: the name of the cache
      path: the path where the cache will be stored
      filename: the filename of the cache
      df: the DataFrame itself
      loaded: boolean telling whether the cache has been loaded

    Methods:
      check_path: check for a valid path
      check_file: check that the file exists
      write_file: write the DataFrame to file
      read_file: read the DataFrame from file
      delete_file: delete the cache file
      load: load the DataFrame, from file if available or from fetch()
    """

    name = 'base'
    path = None

    def __init__(self):
        """Initialize the cache.

        Raises:
          CacheError: if the path is invalid
        """

        if self.path is None:
            self.path = os.path.split(__file__)[0]
        self.check_path()
        self.filename = os.path.join(
            self.path, '__df_cache__' + self.name + '.csv')
        self.df = None
        self.loaded = False

    def check_path(self):
        """Test that the path exists.

        Raises:
          CacheError: if the path does not exist.
        """

        if not os.path.exists(self.path):
            raise CacheError('Cache path does not exist: {}'.format(self.path))
        if not os.path.isdir(self.path):
            raise CacheError(
                'Cache path is not a directory: {}'.format(self.path))

    def check_file(self):
        """Test that the file exists.

        Raises:
          CacheError: if the file does not exist.
        """

        if not os.path.exists(self.filename):
            raise CacheError(
                'Cache filename does not exist: {}'.format(self.filename))
        if not os.path.isfile(self.filename):
            raise CacheError(
                'Cache filename is not a file: {}'.format(self.filename))

    def write_file(self):
        """Write the DataFrame to the cache file.

        Raises:
          CacheError: if there was a problem writing the cache to file.
        """

        self.check_path()
        if not self.loaded:
            raise CacheError('Cache has not been fetched or loaded')
        try:
            self.df.to_csv(self.filename, float_format='%.12f')
        except:
            raise CacheError(
                'Problem writing cache to file {}'.format(self.filename))
        self.check_file()

    def read_file(self):
        """Read the cached DataFrame from file.

        Raises:
          CacheError: if there was a problem reading the cache from file.
        """

        self.check_file()
        try:
            self.df = pd.read_csv(self.filename)
        except:
            raise CacheError(
                'Problem reading cache from file {}'.format(self.filename))
        self.loaded = True

    def delete_file(self):
        """Delete the cache file.

        Raises:
          CacheError: if there was a problem deleting the file.
        """

        self.check_file()
        try:
            os.remove(self.filename)
        except:
            raise CacheError(
                'Problem deleting cache file {}'.format(self.filename))
        try:
            self.check_file()
        except CacheError:
            pass  # this should be raised
        else:
            raise CacheError(
                'Cache file was not deleted: {}'.format(self.filename))

    def fetch(self):
        """Fetch the DataFrame to be cached.

        This method must be implemented in child classes.
        """

        raise NotImplementedError('Must implement fetch method')

    def load(self):
        """Read or download the cached DataFrame.

        Raises:
          CacheError: if there was a problem fetching the data or reading
          the cache from file, or if there was a problem writing the cache
          to file.
        """

        try:
            self.read_file()
        except CacheError:
            try:
                self.fetch()
            except CacheError:
                raise CacheError('Cannot read or download DataFrame')
            self.write_file()
            self.read_file()
        self.loaded = True
