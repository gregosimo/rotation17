# rotletter18
These were files used to create the submission for Simonian et al (2019). This repository is mainly for evaluating the code
style for a project which has already been completed.

# Files

## paperexport.py

This file is the one that is directly called to generate the figures and tables in the paper.
It depends on many additional libraries which are not included in this repository.

## data_cache.py

Functions that return datasets. Because research is iterative where I often make small adjustments to the analysis,
reprocessing the data takes up a substantial amount of unnecessary time. As a result, all functions dealing with building the
samples are contained in data_cache.py, so that a reload(paperexport) does not require the data to be reprocessed again.

## main.tex

The LaTeX file containing the text of the paper.

## references.bib

The bibliography file for the references.

## makefile

A makefile to automatically run the LaTeX compiler, or to generate outputs for different journals.
