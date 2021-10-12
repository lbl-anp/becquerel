"""Test DataFrameCache class."""

import pandas as pd
from becquerel.tools.df_cache import DataFrameCache, CacheError
import pytest


class ExampleCache(DataFrameCache):

    name = "example"

    def fetch(self):
        self.df = pd.DataFrame()
        self.df["letters"] = ["a", "b", "c", "g", "h"]
        self.df["numbers"] = [1, 2, 6, 8, 3]
        self.loaded = True


class TestCacheFunctionality:
    def test_fetch(self):
        """Test ExampleCache.fetch()."""
        d = ExampleCache()
        d.fetch()

    def test_write_file(self):
        """Test ExampleCache.write_file()."""
        d = ExampleCache()
        d.fetch()
        d.write_file()

    def test_read_file(self):
        """Test ExampleCache.read_file()."""
        d = ExampleCache()
        d.fetch()
        d.write_file()
        d.read_file()

    def test_delete_file(self):
        """Test ExampleCache.delete_file()."""
        d = ExampleCache()
        d.fetch()
        d.write_file()
        d.delete_file()

    def test_load(self):
        """Test ExampleCache.load()."""
        d = ExampleCache()
        d.load()

    def test_load_multiple(self):
        """Test ExampleCache.load() and delete_file()."""
        d = ExampleCache()
        d.load()
        d.load()
        d.load()
        d.delete_file()


class TestCacheExceptions:
    def test_bad_path(self):
        """Test ExampleCache.check_path() exception for a bad path."""
        d = ExampleCache()
        d.path = "/bad/path"
        with pytest.raises(CacheError):
            d.check_path()
        with pytest.raises(CacheError):
            d.check_file()

    def test_df_none_write(self):
        """Test ExampleCache.write() exception if DataFrame not loaded."""
        d = ExampleCache()
        with pytest.raises(CacheError):
            d.write_file()

    def test_read_deleted_file(self):
        """Test ExampleCache.read_file() exception after delete_file."""
        d = ExampleCache()
        d.load()
        d.delete_file()
        with pytest.raises(CacheError):
            d.read_file()

    def test_delete_deleted_file(self):
        """Test ExampleCache.read_file() exception after delete_file."""
        d = ExampleCache()
        d.load()
        d.delete_file()
        with pytest.raises(CacheError):
            d.delete_file()
