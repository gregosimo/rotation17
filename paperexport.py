import os
import sys
import argparse

import numpy as np
import numpy.core.defchararray as npstr
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    cools = split.jen_cool_splitter()
    return cools

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
    apogeesplitter = dwarf_data_splitter()

    # I want the targeting dwarfs to be just those that have APOGEE targeting
    # flags. I also just want the targets with 5450 K > Teff > 4250 K.
    apotargs = apogeesplitter.subsample(["Good Teff", "Cold", "~Bad"])
    return apogeesplitter

def targeting_figure(dest=build_filepath(FIGURE_PATH, "targeting", "pdf")):
    '''Create figure showing where the two samples lie in the HR diagram.

    Asteroseismic targets should be blue while cool dwarfs ought to be red.'''
    asteroseismic = get_asteroseismic_dwarfs()
    cooldwarfs = get_cool_sample()

def DLSB_HR_Diagram(
        cool_dwarfs, dest=build_filepath(FIGURE_PATH, "cool_dlsb", "pdf")):
    '''Compare DLSB locations in HR diagram to non-DLSBs.'''
    non_dlsbs = cool_dwarfs.subsample(["~Bad", "No DLSB"])
    dlsbs = cool_dwarfs.subsample(["~Bad", "DLSB"])
    assert cool_dwarfs.subsample_len(["~Bad", "Unknown DLSB", "Vsini det"]) == 0
    
    hr.logg_teff_plot(non_dlsbs["TEFF"], non_dlsbs["LOGG_FIT"], 'k.',
                      label="Non-DLSB")
    hr.logg_teff_plot(dlsbs["TEFF"], dlsbs["LOGG_FIT"], 'ro', label="DLSB")

    plt.xlabel("APOGEE Teff")
    plt.ylabel("APOGEE Log(g)")
    plt.title("DLSBs on HR Diagram")

    plt.legend(loc="upper left")

def HR_Diagram_vsini_detections(
        cool_dwarfs, dest=build_filepath(FIGURE_PATH, "vsini_det", "pdf")):
    '''Plot targets with vsini detections on HR diagram.'''
    nondets = cool_dwarfs.subsample(["~Bad", "No DLSB", "Vsini nondet"])
    marginal = cool_dwarfs.subsample(["~Bad", "No DLSB", "Vsini marginal"])
    dets = cool_dwarfs.subsample(["~Bad", "No DLSB", "Vsini det"])

    hr.logg_teff_plot(
        nondets["TEFF"], nondets["LOGG_FIT"], color=bc.black, 
        linestyle="", marker=".", label="Vsini nondetection", style="")
    hr.logg_teff_plot(
        marginal["TEFF"], marginal["LOGG_FIT"], color=bc.green, 
        linestyle="", marker="v", label="Vsini marginal", style="")
    hr.logg_teff_plot(
        dets["TEFF"], dets["LOGG_FIT"], color="blue", linestyle="", 
        marker="o", label="Vsini detection", style="")

    plt.xlabel("APOGEE Teff")
    plt.ylabel("APOGEE Log(g)")
    plt.title("Detections on HR Diagram")

    plt.legend(loc="upper right")

def sample_with_McQuillan_detections(
        cool_dwarfs, dest=build_filepath(FIGURE_PATH, "mcq", "pdf")):
    '''Plot the cool darfs with McQuillan detections.'''
    perioddet = cool_dwarfs.subsample(["~Bad", "Mcq"])
    periodnondet = cool_dwarfs.subsample(["~Bad", "No Mcq"])
    nomcq = cool_dwarfs.subsample(["~Bad", "Unknown Mcq"])

    hr.logg_teff_plot(
        periodnondet["TEFF"], periodnondet["LOGG_FIT"], color=bc.black, 
        linestyle="", marker=".", label="No period", style="")
    hr.logg_teff_plot(
        nomcq["TEFF"], nomcq["LOGG_FIT"], color=bc.red, 
        linestyle="", marker="x", label="Out of McQuillan", style="")
    hr.logg_teff_plot(
        perioddet["TEFF"], perioddet["LOGG_FIT"], color="blue", linestyle="", 
        marker="o", label="McQuillan detection", style="")

    plt.xlabel("APOGEE Teff")
    plt.ylabel("APOGEE Log(g)")
    plt.title("McQuillan Detections on HR Diagram")

#    plt.xlim(6500, 3500)
#    plt.ylim(4.8, 3.5)

    plt.legend(loc="upper right")

def metallicity_on_hr_diagram(
        cool_dwarfs, dest=build_filepath(FIGURE_PATH, "hr_metallicity", "pdf")):
    '''Plot the metallicities of targets on the hr diagram.'''

    alltargets = cool_dwarfs.subsample(["~Bad"])

    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap("cool")
    metrange = np.max(np.abs(alltargets["FE_H"]))
    norm = mpl.colors.Normalize(vmin=-metrange, vmax=metrange)

    cax = ax.scatter(
        alltargets["TEFF"], alltargets["LOGG_FIT"],
        c=alltargets["FE_H"], cmap=cmap, marker="o")

    cbar = fig.colorbar(cax)
    cbar.ax.set_xlabel("[Fe/H]")
    plt.xlabel("APOGEE Teff")
    plt.ylabel("APOGEE Log(g)")
    plt.title("Metallicity of Cool Dwarf sample")

#    plt.xlim(6500, 3500)
#    plt.ylim(4.8, 3.5)
    hr.invert_x_axis()
    hr.invert_y_axis()

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
