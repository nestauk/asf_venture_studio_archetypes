# ASF Venture-Studio

This repository contains the analysis for the project in collaboration with Furbnow and Carno of the Venture Studio. The aim of the project is to identify property archetypes based on Trustmark data to inform retrofitting assessment.

## EPC data prototype

We use EPC data for the prototype phase. Trustmark data, or alternative sources, will be consider at later stages.

#### Instructions

To reproduce any part of the analysis:

- Download EPC data following the instructions of [asf_core_data](https://github.com/nestauk/asf_core_data)
- Setup `conda env` for the project (instructions below)
- Change the location of your local data directory in `config/base_epc.py`
- Run any selected part of the analysis form command line. For example:

`python asf_venture_studio_archetypes/pipeline/dimensionality_reduction.py`

## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `direnv` and `conda`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure `pre-commit`

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
