import os
import sys
import argparse

import numpy as np
import numpy.core.defchararray as npstr
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii

sys.path.append(os.path.join(os.environ["THESIS"], "scripts"))
import observations as obs
import path_config as paths
import read_catalog as catin
import hrplots as hr
import astropy_util as au
import catalog
import sed
import APOGEE_spectroscopy as spec

PAPER_PATH = paths.HOME_DIR / "papers" / "rotation17"
TABLE_PATH = PAPER_PATH / "tables"
FIGURE_PATH = PAPER_PATH / "fig"

def build_filepath(toplevel, filename, suffix="png"):
    '''Generate a full path to save a filename.'''

    fullpath = toplevel / ".".join((filename, suffix))
    return str(fullpath)

def get_asteroseismic_dwarfs():
    '''Get the final sample of the observing targets.
    
    Note that these include those without McQuillan detections.'''
    astero = spec.APOKASCSplitter()
    spec.initialize_asteroseismic_sample(astero)
    return astero

def get_cool_sample():
    '''Get the final sample of cool dwarfs.
    
    Note that these include those without McQuillan detections.'''
    apogee = spec.APOGEESplitter()
    spec.initialize_Jen_Sample_Splitter(aposplit)

def targeting_figure(dest=build_filepath(FIGURE_PATH, "targeting", "pdf")):
    '''Create figure showing where the two samples lie in the HR diagram.

    Asteroseismic targets should be blue while cool dwarfs ought to be red.'''
    asteroseismic = get_asteroseismic_dwarfs()
    cooldwarfs = catin.read

if __name__ == "__main__":

    desc = """Generate figures and tables.

Script to automatically generate the figures and tables needed for the paper on
looking for tidally-synchronized binaries in the Kepler field."""

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("figs", nargs="*")
    parser.add_argument("--list-figs", action="store_true")
    
    args = parser.parse_args()

    figlist = {
        "hrsample": create_sample_HR_diagram,
        "pspacesample": create_sample_Prot_diagram,
        "obssampletable": create_observing_sample_table,
        "apotable": create_APOGEE_table}

    genfigs = args.figs
    if not genfigs:
        print(figlist.keys())
    for figname in genfigs:
        figlist[figname]()
