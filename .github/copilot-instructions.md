# Copilot Instructions for pinax

## Overview
This is the repository for pinax, a Python library that provides useful tools to extract tabular data from different files and locations into a common format, and perform structured analysis on such tables. The library is designed to be modular and extensible, allowing users to easily add new data sources and analysis methods.

## Structure instructions
- This package will use a src layout
- This package should be installable via pip, but should also provide compatibility for future usage of Poetry or uv. Therefore, the pyproject.toml file should be standard PEP 621 compliant.
- If possible, there should also be an environment.yml file for conda users.
- Assume that if installing from this package, it will be done for development purposes
