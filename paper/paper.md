---
title: 'Becquerel: a Python package for analyzing nuclear spectroscopic measurements'
tags:
  - Python
  - nuclear engineering
  - spectroscopy
  - calibration
  - peak fitting
  - nuclear data
authors:
  - name: Jayson R. Vavrek
    affiliation: 1
  - name: Mark S. Bandstra
    affiliation: 1
  - name: Brian Plimley
    affiliation: 1
  - name: Joseph C. Curtis
    affiliation: 1
  - name: Marco Salathe
    affiliation: 1
  - name: Chun Ho Chow
    affiliation: 1
  - name: Tenzing H.Y. Joshi
    affiliation: 1
affiliations:
  - name: Applied Nuclear Physics group, Lawrence Berkeley National Laboratory
    index: 1
date: 23 Feb 2021
bibliography: paper.bib
---

# Summary

Nuclear spectroscopic analysis follows a typical workflow: collect binmode or listmode spectra of radiation emissions, apply a calibration from detector observables to energy deposition, fit spectral peaks to determine the number of counts detected above background, and then relate changes in net counts to measurement parameters. While there are several open-source general scientific computing packages such as ``ROOT`` and ``numpy`` suitable for handling spectroscopic data, no dedicated solution exists for both managing and analyzing spectroscopic measurements, forcing students and researchers to develop their own codes independently.

``becquerel`` is a Python package for analyzing nuclear spectroscopic measurements that seeks to prevent this wide duplication of efforts. It provides open-source standard analysis tools, including peak finding and fitting, automated energy calibrations, and file I/O across several widely-used formats in nuclear spectroscopy. Built atop the Python scientific stack of ``numpy``, ``scipy``, ``uncertainties``, and ``lmfit``, ``becquerel`` is fast, flexible, and easy to use given even an introductory knowledge of Python. In addition, ``becquerel`` provides a comprehensive test suite, coverage metrics, and several example notebooks for quickly getting started with various analyses. Finally, it also provides a convenient Python interface to the NIST XCOM and NNDC nuclear databases, eliminating the need for ad-hoc downloads or manual data entry.

``becquerel`` was developed to be useable by a wide range of nuclear scientists and engineers, from undergraduates in laboratory courses to academic and national laboratory researchers. It has already been used to facilitate analyses in several papers and proceedings, including [@vavrek2020reconstructing], [@bandstra2020modeling], and [@salathe2019using], and will facilitate the rapid development of future spectroscopic analyses and workflows.

# Acknowledgements

# References
