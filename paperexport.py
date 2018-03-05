import os
import sys
import argparse

import numpy as np
import numpy.core.defchararray as npstr
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.table import Table, vstack
from astropy.io import ascii
import statop as stat
import functools

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
import rotation_consistency as rot
import sample_characterization as samp

PAPER_PATH = paths.HOME_DIR / "papers" / "rotletter18"
TABLE_PATH = PAPER_PATH / "tables"
FIGURE_PATH = PAPER_PATH / "fig"
PLOT_SUFFIX = "png"

def build_filepath(toplevel, filename, suffix=PLOT_SUFFIX):
    '''Generate a full path to save a filename.'''

    fullpath = toplevel / ".".join((filename, suffix))
    return str(fullpath)

def write_plot(filename, suffix=PLOT_SUFFIX, toplevel=FIGURE_PATH):
    '''Create a decorator that will write plots to a given file.'''
    def decorator(f): 
        @functools.wraps(f)
        def wrapper():
            plt.close("all")
            f()
            filepath = build_filepath(toplevel, filename, suffix=suffix)
            plt.savefig(filepath)
        return wrapper
    return decorator

@au.memoized
def asteroseismic_data_splitter():
    '''A persistent datasplitter for the asteroseismic sample.'''
    astero = split.APOKASCSplitter()
    split.initialize_asteroseismic_sample(astero)
    return astero

@au.memoized
def dwarf_data_splitter():
    '''A persistent DataSplitter for the cool dwarf sample.'''
    apogee = split.APOGEESplitter()
    split.initialize_general_APOGEE(apogee)
    return apogee

@au.memoized
def cool_data_splitter():
    '''A persistent Datasplitter for Jen's cool sample.'''
    dwarfs = dwarf_data_splitter()
    cool_dwarfs = split.general_to_cool_sample(dwarfs)
    split.initialize_cool_KICs(cool_dwarfs)
    return cool_dwarfs

@write_plot("Bruntt_comp")
def Bruntt_vsini_comparison():
    '''Create figure comparison vsini for asteroseismic targets.

    The APOGEE vsinis are compared against the Bruntt et al (2012)
    spectroscopic vsinis for a set of asteroseismic targets.'''

    astero_dwarfs = catin.bruntt_dr14_overlap()
    bad_indices = astero_dwarfs["VSINI"] < 0
    vsini_diff = ((astero_dwarfs["vsini"][~bad_indices] -
                   astero_dwarfs["VSINI"][~bad_indices]) /
                  astero_dwarfs["vsini"][~bad_indices])
    vsini_baddiff = np.ones(np.count_nonzero(bad_indices))
    plt.plot(astero_dwarfs["vsini"][~bad_indices], vsini_diff, 'ko')
    plt.plot(astero_dwarfs["vsini"][bad_indices], vsini_baddiff, 'rv')
    plt.plot([0, 35], [0, 0], 'k--')
    plt.ylim([-0.2, 0.5])
#   plt.plot(astero_dwarfs["vsini"][~bad_indices],
#            astero_dwarfs["VSINI"][~bad_indices], 'ko')
#   plt.plot(astero_dwarfs["vsini"][bad_indices],
#            np.zeros(np.count_nonzero(bad_indices)), 'rx')
#   plt.plot([0, 35], [0, 35], 'k--')
    outlier = astero_dwarfs[~bad_indices][np.abs(vsini_diff) > 1.0]
    assert len(outlier) == 1
    print("Ignoring KIC{0:d}: Bruntt vsini = {1:.1f}, ASPCAP vsini = {2:.1f}".format(
        outlier["kepid"][0], outlier["vsini"][0], outlier["VSINI"][0]))
    print("Bad objects:")
    print(astero_dwarfs[["KIC", "vsini"]][bad_indices])
    plt.xlabel("Bruntt vsini (km/s)")
    plt.ylabel("(Bruntt vsini - ASPCAP vsini) / Bruntt vsini")

@write_plot("Pleiades_comp")
def Pleiades_vsini_comparison():
    '''Create figure comparing vsini for Pleiades targets.'''
    targets = catin.Stauffer_APOGEE_overlap()
    non_dlsbs = targets[npstr.find(targets["Notes"], "5") < 0]
    good_targets = catalog.apogee_filter_quality(non_dlsbs, quality="bad")
    nondetections = good_targets["vsini lim"] == stat.UPPER
    detected_targets = good_targets[~nondetections]
    nondet_targets = good_targets[nondetections]
    detected_errors = (detected_targets["vsini"] / 2 / (
        1 + detected_targets["R"])).filled(0)
    det_frac = ((detected_targets["vsini"] - detected_targets["VSINI"]) /
                detected_targets["vsini"])
    frac_errors = detected_errors * (
        detected_targets["VSINI"] / detected_targets["vsini"]**2)
    nondet_frac = ((nondet_targets["vsini"] - nondet_targets["VSINI"]) /
                   nondet_targets["vsini"])
                   
#   plt.errorbar(detected_targets["vsini"], detected_targets["VSINI"], 
#                xerr=detected_errors, fmt='ko')
#   plt.plot(nondet_targets["vsini"], nondet_targets["VSINI"], 'r<')
#   plt.plot([0, 80], [0, 80], 'k-')
    plt.errorbar(detected_targets["vsini"], det_frac, yerr=frac_errors, 
                 xerr=detected_errors, fmt='ko')
#   plt.plot(nondet_targets["vsini"], nondet_frac, 'r<')
    plt.plot([15, 15], [-0.3, 0.5], 'r--')
    plt.plot([0, 75], [0, 0], 'k--')
    plt.ylabel("(SH vsini - APOGEE vsini) / (SH vsini)")
    plt.xlabel("SH vsini")

@write_plot("astero")
def asteroseismic_sample_loggs():
    '''Write an HR diagram consisting of the asteroseismic sample.
    
    Plot the asteroseismic and spectroscopic log(g) values separately. Should
    also plot the underlying APOKASC sample.'''
    apokasc = asteroseismic_data_splitter()
    apokasc.split_targeting("APOGEE2_APOKASC_DWARF")
    fulldata = apokasc.subsample(["~Bad", "APOGEE2_APOKASC_DWARF"])
    astero = apokasc.subsample(["~Bad", "Asteroseismic Dwarfs"])

    hr.logg_teff_plot(
        fulldata["TEFF_COR"], fulldata["LOGG_FIT"], color=bc.black, marker=".",
        label="APOGEE", ls="")
    hr.logg_teff_plot(astero["TEFF_COR"], astero["LOGG_DW"], color=bc.green,
                      marker="*", ms=8, label="Asteroseismic log(g)", ls="")
    hr.logg_teff_plot(astero["TEFF_COR"], astero["LOGG_FIT"], color=bc.sky_blue,
                      marker="o", ms=6, label="Spectroscopic log(g)", ls="")

    plt.xlim([6750, 4750])
    plt.ylim([5.0, 3.0])
    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("Log(g) [cm/s/s]")
    plt.legend(loc="upper left")

@write_plot("cool_sample")
def cool_dwarf_hr():
    '''Plot the cool dwarf sample on an HR diagram.

    Illustrate the subgiant/dwarf division, and overplot rapid rotators.'''
    cool = cool_data_splitter()
    cool_full = cool.subsample(["~Bad"])
    cool_subgiants = cool.subsample(["~Bad", "Subgiant"])
    cool_dwarfs = cool.subsample(["~Bad", "Dwarf"])
    cool_rapid_dwarfs = cool.subsample([
        "~Bad", "Vsini det", "~DLSB", "Dwarf", "Mcq"])
    hr.logg_teff_plot(cool_subgiants["TEFF"], cool_subgiants["LOGG_FIT"], 
                      color=bc.green, marker=".", label="Subgiants", ls="")
    hr.logg_teff_plot(cool_dwarfs["TEFF"], cool_dwarfs["LOGG_FIT"], 
                      color=bc.red, marker=".", label="Dwarfs", ls="")
    hr.logg_teff_plot(cool_rapid_dwarfs["TEFF"], cool_rapid_dwarfs["LOGG_FIT"], 
                      color=bc.sky_blue, marker="*", label="Rapid Rotators",
                      ls="", ms=7)
    plt.plot([4640, 5690], [3.72, 4.43], 'k-')
    plt.xlim(5750, 3500)
    plt.ylim(4.7, 3.5)
    plt.xlabel("APOGEE Teff")
    plt.ylabel("APOGEE Logg")
    plt.legend()

@write_plot("astero_rot")
def asteroseismic_rotation_analysis():
    '''Plot rotation comparison of asteroseismic sample.'''
    astero = asteroseismic_data_splitter()
    datapoints = astero.subsample([
        "~Bad", "Asteroseismic Dwarfs", "Vsini det", "~DLSB", "Mcq"])

    mcq = catin.read_McQuillan_catalog()
    periodpoints = au.join_by_id(datapoints, mcq, "kepid", "KIC")

    rot.compare_rotation_velocity_radius(
        periodpoints["VSINI"], periodpoints["Prot"], periodpoints["radius"],
        raderr_below=periodpoints["radius_err1"],
        raderr_above=periodpoints["radius_err2"])

@write_plot("cool_rot")
def cool_dwarf_rotation_analysis():
    '''Plot rotation comparison of cool dwarf sample.'''
    cool_dwarf = cool_data_splitter()
    datapoints = cool_dwarf.subsample([
        "~Bad", "Vsini det", "~DLSB", "Mcq", "Dwarf"])

    mcq = catin.read_McQuillan_catalog()
    periodpoints = au.join_by_id(datapoints, mcq, "kepid", "KIC")

    samp.generate_DSEP_radius_column_with_errors( periodpoints)

    rot.compare_rotation_velocity_radius(
        periodpoints["VSINI"], periodpoints["Prot"], periodpoints["DSEP radius"],
        raderr_below=periodpoints["DSEP radius lower"],
        raderr_above=periodpoints["DSEP radius upper"], vsini_lim=15)
        
    

def targeting_figure(dest=build_filepath(FIGURE_PATH, "targeting", "pdf")):
    '''Create figure showing where the two samples lie in the HR diagram.

    Asteroseismic targets should be blue while cool dwarfs ought to be red.'''
    asteroseismic = asteroseismic_data_splitter()
    hot_kic = hot_kic_data_splitter()
    hot_nonkic = hot_nonkic_data_splitter()
    cooldwarfs = cool_data_splitter()

    ast_mcq = asteroseismic.subsample(["Asteroseismic Dwarfs", "~Bad", "Mcq"])
    hot_kic_mcq = hot_kic.subsample(["~Bad", "Mcq"])
    hot_nonkic_mcq = hot_nonkic.subsample(["~Bad", "Mcq"])
    cool_dwarf_mcq = cooldwarfs.subsample(["~Bad", "Mcq"])

    ast_nomcq = asteroseismic.subsample([
        "Asteroseismic Dwarfs", "~Bad", "Unknown Mcq"])
    hot_kic_nomcq = hot_kic.subsample(["~Bad", "Unknown Mcq"])
    hot_nonkic_nomcq = hot_nonkic.subsample(["~Bad", "Unknown Mcq"])
    cool_dwarf_nomcq = cooldwarfs.subsample(["~Bad", "Unknown Mcq"])

    ast_data = asteroseismic.subsample(["Asteroseismic Dwarfs", "~Bad"])
    hot_kic_data = hot_kic.subsample(["~Bad"])
    hot_nonkic_data = hot_nonkic.subsample(["~Bad"])
    cool_dwarf_data = cooldwarfs.subsample(["~Bad"])

    fig, axarr = plt.subplots(2, 2, sharex="all", sharey="all", figsize=(7, 5))
    bigmark = 4
    smallmark=2
    hr.logg_teff_plot(
        ast_mcq["TEFF_COR"], ast_mcq["LOGG_FIT"], color=bc.yellow,
        marker="o", markersize=bigmark, linestyle="", style="",
        label="McQuillan", axis=axarr[0][0])
    hr.logg_teff_plot(
        ast_nomcq["TEFF_COR"], ast_nomcq["LOGG_FIT"], color=bc.orange,
        marker="o", markersize=bigmark, linestyle="", style="",
        label="Unanalyzed", axis=axarr[0][0])
    hr.logg_teff_plot(
        ast_data["TEFF_COR"], ast_data["LOGG_FIT"], color=bc.black,
        marker=".", markersize=smallmark, linestyle="", style="",
        label="Sample", axis=axarr[0][0])
    axarr[0][0].legend()
    axarr[0][0].set_ylabel("APOGEE log(g)")
    axarr[0][0].set_title("Asteroseismic")
    hr.logg_teff_plot(
        hot_kic_mcq["TEFF"], hot_kic_mcq["FPARAM"][:,1], color=bc.yellow,
        marker="o", markersize=bigmark, linestyle="", style="", axis=axarr[0][1])
    hr.logg_teff_plot(
        hot_kic_nomcq["TEFF"], hot_kic_nomcq["FPARAM"][:,1], color=bc.orange,
        marker="o", markersize=bigmark, linestyle="", style="", axis=axarr[0][1])
    hr.logg_teff_plot(
        hot_kic_data["TEFF"], hot_kic_data["FPARAM"][:,1], color=bc.black,
        marker=".", markersize=smallmark, linestyle="", style="", axis=axarr[0][1])
    axarr[0][1].set_title("Hot KIC")
    hr.logg_teff_plot(
        hot_nonkic_mcq["TEFF"], hot_nonkic_mcq["FPARAM"][:,1], color=bc.yellow,
        marker="o", markersize=bigmark, linestyle="", style="", axis=axarr[1][0])
    hr.logg_teff_plot(
        hot_nonkic_nomcq["TEFF"], hot_nonkic_nomcq["FPARAM"][:,1], color=bc.orange,
        marker="o", markersize=bigmark, linestyle="", style="", axis=axarr[1][0])
    hr.logg_teff_plot(
        hot_nonkic_data["TEFF"], hot_nonkic_data["FPARAM"][:,1], color=bc.black,
        marker=".", markersize=smallmark, linestyle="", style="", axis=axarr[1][0])
    axarr[1][0].set_xlabel("APOGEE Teff")
    axarr[1][0].set_ylabel("APOGEE log(g)")
    axarr[1][0].set_title("Hot Non-KIC")
    hr.logg_teff_plot(
        cool_dwarf_mcq["TEFF"], cool_dwarf_mcq["FPARAM"][:,1],
        color=bc.yellow, marker="o", markersize=bigmark, linestyle="", style="", 
        axis=axarr[1][1])
    hr.logg_teff_plot(
        cool_dwarf_nomcq["TEFF"], cool_dwarf_nomcq["FPARAM"][:,1],
        color=bc.orange, marker="o", markersize=bigmark, linestyle="", style="", 
        axis=axarr[1][1])
    hr.logg_teff_plot(
        cool_dwarf_data["TEFF"], cool_dwarf_data["FPARAM"][:,1],
        color=bc.black, marker=".", markersize=smallmark, linestyle="", style="", 
        axis=axarr[1][1])
    axarr[1][1].set_xlabel("APOGEE Teff")
    axarr[1][1].set_ylabel("")
    axarr[1][1].set_xlim(7000, 3500)
    axarr[1][1].set_ylim(5.0, 0.0)
    axarr[1][1].set_title("Cool Dwarf")

    plt.savefig(str(dest))

def DLSB_HR_Diagram(
        cool_dwarfs, dest=build_filepath(FIGURE_PATH, "cool_dlsb", "pdf"),
    teff_col="TEFF", logg_col="LOGG_FIT"):
    '''Compare DLSB locations in HR diagram to non-DLSBs.'''
    non_dlsbs = cool_dwarfs.subsample(["~Bad", "~DLSB"])
    dlsbs = cool_dwarfs.subsample(["~Bad", "DLSB"])
    assert cool_dwarfs.subsample_len(["~Bad", "Unknown DLSB", "Vsini det"]) == 0

    
    hr.logg_teff_plot(fullsample[teff_col], fullsample[logg_col], 'k.',
                      label="Full sample")
    hr.logg_teff_plot(dlsbs[teff_col], dlsbs[logg_col], 'ro', label="DLSB")

    plt.xlabel("APOGEE Teff")
    plt.ylabel("APOGEE Log(g)")
    plt.title("DLSBs on HR Diagram")

    plt.legend(loc="upper left")

def HR_Diagram_vsini_detections(
        cool_dwarfs, dest=build_filepath(FIGURE_PATH, "vsini_det", "pdf")):
    '''Plot targets with vsini detections on HR diagram.'''
    nondets = cool_dwarfs.subsample(["~Bad", "Vsini nondet"])
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
    plt.plot([6500, 3500], [4.2, 4.2], 'k--')

    plt.xlabel("APOGEE Teff")
    plt.ylabel("APOGEE Log(g)")
    plt.title("Detections on HR Diagram")
    plt.xlim(6500, 3500)

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

    # I want to do this with slices. 
    nrows, ncols = 2, 4
    bins = np.linspace(-1.1, 0.5, nrows*ncols+1)
    binindices = np.digitize(alltargets["FE_H"], bins)
    fig, axarr = plt.subplots(nrows, ncols, sharex="all", sharey="all")
    for r in np.arange(2):
        for c in np.arange(4):
            arrindex = r * ncols + c
            if arrindex != 8:
                curax = axarr[r, c]
                subtable = alltargets[binindices == arrindex+1]
                hr.logg_teff_plot(
                    subtable["TEFF"], subtable["LOGG_FIT"], marker="o",
                    axis=curax)
                curax.set_title("{0:.1f} <= [Fe/H] <= {1:.1f}".format(
                    bins[arrindex], bins[arrindex+1]))
                curax.set_xlim(6500, 3500)
                curax.set_ylim(4.8, 3.6)

    fig.suptitle("Metallicity Trend")

def plot_hot_kic_vs_nonkic():
    '''Plot the location of targets with and without original KIC targets.'''
    hot_kic = hot_kic_data_splitter()
    hot_kic_data = hot_kic.subsample(["~Bad"])
    hot_nonkic = hot_nonkic_data_splitter()
    hot_nonkic_data = hot_nonkic.subsample(["~Bad"])

    hr.logg_teff_plot(
        hot_kic_data["TEFF"], hot_kic_data["LOGG_FIT"], "k.", label="KIC")
    hr.logg_teff_plot(
        hot_nonkic_data["TEFF"], hot_nonkic_data["LOGG_FIT"], "ro", label="No KIC")

    plt.xlabel("APOGEE Teff")
    plt.ylabel("APOGEE log(g)")
    plt.legend()


def display_asteroseismic_census():
    '''Display relevant numbers in the asteroseismic sample.'''
    astero = asteroseismic_data_splitter()
    astero_dwarfs = astero.split_subsample(["Asteroseismic Dwarfs"])

    print("Initial number of asteroseismic targets: {0:d}".format(
        len(astero_dwarfs.data)))
    print("Bad targets: {0:d}/{1:d}".format(
        astero_dwarfs.subsample_len(["Bad"]), astero_dwarfs.subsample_len([])))
    print("Vsini detections that are DLSBs: {0:d}/{1:d}".format(
        astero_dwarfs.subsample_len(["~Bad", "DLSB"]),
        astero_dwarfs.subsample_len(["~Bad"])))
    print("Non-DLSB stars with McQuillan periods: {0:d}/{1:d}".format(
        astero_dwarfs.subsample_len(["~Bad", "~DLSB", "Mcq"]),
        astero_dwarfs.subsample_len(["~Bad", "~DLSB"])))

def display_hot_star_census():
    '''Display relevant numbers in the hot star sample.'''
    hot_nonkic = hot_nonkic_data_splitter()
    hot_kic = hot_kic_data_splitter()

    totalsample = hot_nonkic.subsample_len([]) + hot_kic.subsample_len([])
    print("Total number of targeted objects: {0:d}".format(totalsample))
    print("Targets with KIC parameters: {0:d}/{1:d}".format(
        hot_kic.subsample_len([]), totalsample))
    print("Targets without KIC parameters: {0:d}/{1:d}".format(
        hot_nonkic.subsample_len([]), totalsample))

    print("KIC parameter targets with bad fits: {0:d}/{1:d}".format(
        hot_kic.subsample_len(["Bad"]), hot_kic.subsample_len([])))
    print("Non-KIC parameter targets with bad fits: {0:d}/{1:d}".format(
        hot_nonkic.subsample_len(["Bad"]), hot_nonkic.subsample_len([])))

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
