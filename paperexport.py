import os
import sys
import argparse

import numpy as np
import numpy.core.defchararray as npstr
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.io import ascii

sys.path.append(os.path.join(os.environ["THESIS"], "scripts"))
import observations as obs
import path_config as paths
import read_catalog as catin
import hrplots as hr
import astropy_util as au
import catalog
import sed
import data_splitting as split
import biovis_colors as bc
import aspcap_corrections as aspcor

PAPER_PATH = paths.HOME_DIR / "papers" / "rotation17"
TABLE_PATH = PAPER_PATH / "tables"
FIGURE_PATH = PAPER_PATH / "fig"

def build_filepath(toplevel, filename, suffix="png"):
    '''Generate a full path to save a filename.'''

    fullpath = toplevel / ".".join((filename, suffix))
    return str(fullpath)

@au.memoized
def asteroseismic_data_splitter():
    '''Create a persistent datasplitter for the asteroseismic sample.'''
    astero = split.APOKASCSplitter()
    split.initialize_asteroseismic_sample(astero)
    return astero

@au.memoized
def dwarf_data_splitter():
    '''Create a persistent DataSplitter for the cool dwarf sample.'''
    apogee = split.APOGEESplitter()
    apogee = split.initialize_cool_dwarfs(apogee)
    return apogee

def get_asteroseismic_dwarfs():
    '''Get the final sample of the observing targets.
    
    Note that these include those without McQuillan detections.'''
    astero = asteroseismic_data_splitter()
    asteroseismic_dwarfs = astero.subsample(
        ["Asteroseismic Dwarfs", "~Bad"])

    # Fix the bad one.
    bad_dwarfs = astero.subsample(["Asteroseismic Dwarfs", "Bad"])
    # Make exception for 2M19580559+4422509.
    save_table = au.extract_subtable_from_column(
        bad_dwarfs, "2MASS_ID", ["2M19580559+4422509"])
    # Teff corrections
    save_table["TEFF_COR"] = (
        save_table["TEFF_FIT"] + aspcor.aspcap_dwarf_teff_correction( 
            save_table["FE_H"], save_table["LOGG_FIT"],
            save_table["LOGG_COR"]))

    data = vstack([asteroseismic_dwarfs, save_table])
    return data

def get_cool_sample():
    '''Get the final sample of cool dwarfs.
    
    Note that these include those without McQuillan detections.'''
    splitter = dwarf_data_splitter()

    # These are all the targets which were observed with APOGEE1
    apogee1_samp = get_cool_apogee1_sample(splitter)
    # Now get the targets which were observed in APOGEE2 that meet Jen's
    # selection criteria: KIC log(g) > 4.0, Pinsonneault SDSS Teff < 5500 K,
    # and 7 < H < 11.
    apogee2_samp = splitter.subsample([
        "APOGEE2_APOKASC_DWARF", "Jen Dwarf", "H Jen", "Jen Cool"])
    # There were a bunch of targets which did not have SDSS Teffs because they
    # were outside of the calibration region. Include the very cool ones.
    nosdss_samp = splitter.subsample([
        "APOGEE2_APOKASC_DWARF", "Jen Dwarf", "H Jen", "No SDSS Teff", 
        "KIC Jen Cool"])

    fullcool = vstack([apogee1_samp, apogee2_samp, nosdss_samp])

    return fullcool

def get_cool_apogee1_sample(splitter):
    '''Get the subset of Jen's sample that were observed in APOGEE1.

    This essentially selects based on the targeting flags in apogeesplitter to
    check for APOGEE_KEPLER_COOLDWARF.'''
    apogee1_samp = splitter.subsample(["APOGEE_KEPLER_COOLDWARF"])
    return apogee1_samp

def get_cool_apogee2_sample(apogeesplitter):
    '''Get the subset of Jen's sample observed in APOGEE2.'''
    pass

def targeting_figure(dest=build_filepath(FIGURE_PATH, "targeting", "pdf")):
    '''Create figure showing where the two samples lie in the HR diagram.

    Asteroseismic targets should be blue while cool dwarfs ought to be red.'''
    asteroseismic = get_asteroseismic_dwarfs()
    cooldwarfs = get_cool_sample()

    hr.logg_teff_plot(
        asteroseismic["TEFF_COR"], asteroseismic["LOGG_FIT"], color=bc.blue,
        marker="*", markersize=9, linestyle="")
    hr.logg_teff_plot(
        cooldwarfs["TEFF"], cooldwarfs["FPARAM"][:,1], color=bc.red,
        marker="o", linestyle="")
    plt.xlim(7000, 3500)
    plt.ylim(5.0, 0.0)

    plt.savefig(str(dest))


if __name__ == "__main__":

    desc = """Generate figures and tables.

Script to automatically generate the figures and tables needed for the paper on
looking for tidally-synchronized binaries in the Kepler field."""

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("figs", nargs="*")
    parser.add_argument("--list-figs", action="store_true")
    
    args = parser.parse_args()

    figlist = {
        "targeting": targeting_figure
    }

    genfigs = args.figs
    if not genfigs:
        print(figlist.keys())
    for figname in genfigs:
        figlist[figname]()
