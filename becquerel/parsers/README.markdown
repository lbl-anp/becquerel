Spectrum File Parsers
=====================

This directory contains Python classes for parsing SPE, SPC, and CNF
spectrum files.

The API is the same for all classes:

```python
spec = SpeFile(filename)
spec.read()
spec.apply_calibration()
```

The main spectral data are available in `spec.energies` and `spec.data`.
Other data available include:

* `spec.livetime`
* `spec.realtime`
* `spec.bin_edges_kev`
* `spec.energy_bin_widths`
* `spec.energy_bin_edges` (depricated)

Some parsers extract more information than others either because of their
stage of development or because the file format may not carry certain
information. Undefined data fields are defined to be '', 0, or None
depending on the data.

There is also a `write()` method defined but not implemented for any
parsers other than SpeFile.
