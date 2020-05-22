import os
import sys
import argparse
import functools
import itertools
import string
import bisect
import tempfile

import numpy as np
import numpy.core.defchararray as npstr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.ticker as ticker
from astropy.table import Table, vstack, unique
from astropy.io import ascii, fits
import astropy.units as u
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import erf, erfc
from scipy import stats
from astroquery.gaia import Gaia
import emcee
import corner

# Until I figure out how to use Matplotlib styles. Use this instead.
mpl.rcParams["axes.titlesize"] = 24
mpl.rcParams["axes.labelsize"] = 26
mpl.rcParams["lines.linewidth"] = 5
mpl.rcParams["lines.markersize"] = 10
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16
mpl.rcParams["legend.fontsize"] = 14

sys.path.append(os.path.join(os.environ["RESEARCH"], "Binaries", "scripts"))
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
import data_cache as cache
import rotation_consistency as rot
import sample_characterization as samp
import mist
import dsep
import baraffe
import models
import yrec
import extinction
import jenboundary as jen
import binarycalcs as bincalc

PAPER_PATH = paths.HOME_DIR / "Documents" / "Papers" / "rotation17"
TABLE_PATH = PAPER_PATH / "tables"
FIGURE_PATH = PAPER_PATH / "fig"
PLOT_SUFFIX = "pdf"
PLOT_PATH = paths.HEAD_DIR / "plots"

figsize=(9, 9)

Protstr = r"$P_{\mathrm{rot}}$"
vsinistr = r"$v \sin i$"
kmsstr = r"km s$^{-1}$"
Teffstr = r"$T_{\mathrm{eff}}$"
MKstr = r"$M_{Ks}$"
fehstr = r"$[Fe/H]$"

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
            a = f()
            filepath = build_filepath(toplevel, filename, suffix=suffix)
            # Have a hidden file that is modified every time the figure is
            # written. This will work better with make.
#            touch(build_filepath(toplevel.parent, "."+filename, suffix="txt"))
            plt.savefig(filepath)
            return a
        return wrapper
    return decorator

def build_filepath(toplevel, filename, suffix="png"):
    '''Generate a full path to save a filename.'''

    fullpath = toplevel / ".".join((filename, suffix))
    return str(fullpath)

def check_SB2_vsini_period():
    '''Check if the vsini and period are concordant with SB2s.

    This figure will plot the vsini and period for the SB2 sample.'''
    pass

def apogee_cool_dwarf_hr_diagram():
    '''Plot the Cool Dwarfs in an HR diagram.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_apo = aposplit.subsample(["Dwarfs", "APOGEE Evolution Cool", "Mcq", "~DLSB"])

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    hr.absmag_teff_plot(cool_apo["TEFF"], cool_apo["M_K"], color='k', ls="",
                        marker=".", axis=ax)
    iso = mist.MISTIsochrone.isochrone_from_file(0.0)
    iso_table = iso.iso_table(1e9)
    hr.absmag_teff_plot(
        10**iso_table[iso.logteff_col], iso_table[mist.band_translation["K"]], 
        color='k', ls="-", marker="", axis=ax)
    ax.set_xlabel("APOGEE Teff")
    ax.set_ylabel("M_K")

    

def compare_visual_to_cc_sb2_identification():
    '''Compare my visual to the Kounkel SB2 identification.'''
    apo = cache.apogee_splitter_with_DSEP()
    apo.split_Kounkel_SB2()
    apodwarfs = apo.split_subsample(["Cool Dwarfs"])
    nosb2 = vstack([apodwarfs.subsample(["Not Kounkel SB2", "No DLSB"]),
                    apodwarfs.subsample(["Not Kounkel SB2", "~Vsini det"])])
    ksb2 = vstack([apodwarfs.subsample(["Kounkel SB2", "No DLSB"]),
                   apodwarfs.subsample(["Kounkel SB2", "~Vsini det"])])
    msb2 = apodwarfs.subsample(["Not Kounkel SB2", "DLSB"])
    bothsb2 = apodwarfs.subsample(["Kounkel SB2", "DLSB"])

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    ax.plot(nosb2["VSINI"], nosb2["K Excess"], marker=".", color=bc.black,
             ls="", label="Not an SB2", alpha=0.2)
    ax.plot(ksb2["VSINI"], ksb2["K Excess"], marker="s", color=bc.purple,
            ls="", label="Kounkel SB2")
    ax.plot(msb2["VSINI"], msb2["K Excess"], marker="^", color=bc.purple,
            ls="", label="Visual SB2")
    ax.plot(bothsb2["VSINI"], bothsb2["K Excess"], marker="o", color="red",
            ls="", label="Both SB2")
    ax.set_xscale("log")
    ax.set_xlabel("V sini (km/s)")
    ax.set_ylabel("K Excess")
    ax.legend(loc="lower right")
    hr.invert_y_axis(ax)

def check_high_vsini_chi2():
    '''Make a plot showing the chi2 fits of high vsini objects vs the full
    sample.'''
    apo = cache.apogee_splitter_with_DSEP()
    full = apo.subsample(["Dwarfs", "APOGEE Evolution Teff"])
    rapid = apo.subsample([
        "Dwarfs", "APOGEE Evolution Teff", "Vsini det", "No DLSB"])
    sb2s = apo.subsample([
        "Dwarfs", "APOGEE Evolution Teff", "Vsini det", "DLSB"])

    sorted_full = np.sort(full["ASPCAP_CHI2"])
    sorted_rapid = np.sort(rapid["ASPCAP_CHI2"])
    sorted_sb2 = np.sort(sb2s["ASPCAP_CHI2"])
    full_indices = np.arange(len(sorted_full))/(len(sorted_full)-1)
    rapid_indices = np.arange(len(sorted_rapid))/(len(sorted_rapid)-1)
    sb2_indices = np.arange(len(sorted_sb2))/(len(sorted_sb2)-1)
    plt.step(sorted_full, full_indices, where="pre", color=bc.black,
             label="Full")
    plt.step(sorted_rapid, rapid_indices, where="pre", color="blue",
             label="Rapid")
    plt.step(sorted_sb2, sb2_indices, where="pre", color="red",
             label="SB2")
    plt.xlabel("ASPCAP Chi-Squared")
    plt.title("Chi squared match for rapid rotators")
    plt.legend(loc="lower right")

def check_high_vsini_logg():
    '''Make a plot in logg/teff space of rapid rotators.'''
    apo = cache.apogee_splitter_with_DSEP()
    full = apo.subsample(["Dwarfs", "APOGEE Evolution Teff"])
    rapid = apo.subsample([
        "Dwarfs", "APOGEE Evolution Teff", "Vsini det", "No DLSB"])
    sb2 = apo.subsample([
        "Dwarfs", "APOGEE Evolution Teff", "Vsini det", "DLSB"])

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    hr.logg_teff_plot(
        full["TEFF"], full["LOGG_FIT"], color="k", marker=".", ls="",
        label="Full", axis=ax)
    hr.logg_teff_plot(
        rapid["TEFF"], rapid["LOGG_FIT"], color="blue", marker="o", ls="",
        label="vsini > 7", axis=ax)
    hr.logg_teff_plot(
        sb2["TEFF"], sb2["LOGG_FIT"], color="red", marker="*", ls="",
        label="SB2", axis=ax)
    ax.set_xlabel("APOGEE Teff (K)")
    ax.set_ylabel("ASPCAP log(g)")
    ax.legend(loc="upper right")

def SB2_Excess():
    '''Plot the SB2s in the cool dwarf domain compared to their K Excess.'''
    apo = cache.apogee_splitter_with_DSEP()
    full = apo.subsample([])
    sb2s = apo.subsample(["DLSB"])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        full["TEFF"], full["K Excess"], color="grey", alpha=0.3, ls="",
        marker=".")
    hr.absmag_teff_plot(
        sb2s["TEFF"], sb2s["K Excess"], color="red", ls="", marker="*")

@write_plot("f2")
def DLSB_Examples():
    '''Make a 4-panel plots showing examples of SB2s.

    The top two rows are showing two SB2s with a clear mismatch between the
    model and spectrum in two different locations. The bottom two rows show the
    appearance and disappearance of lines at two separate epochs.
    
    Both examples draw spectra from paths.HEAD_DIR. The model mismatch will
    need an aspcapStar fits file, while the epoch change will need an apStar
    fits file.'''
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 9))
    PIPELINE_VERSION = "r8-l31c.2"
    # Make the top two panels
    model_mismatch = "2M19082022+5010496"
    aspcap_format = "aspcapStar-{0}-{1}.fits"
    model_mismatch_file = paths.HEAD_DIR / aspcap_format.format(
        PIPELINE_VERSION, model_mismatch)
    aspcaphdu = fits.open(str(model_mismatch_file))
    dataspec = aspcaphdu[1].data
    datawavelengths = 10**(
        aspcaphdu[1].header["CRVAL1"] + aspcaphdu[1].header["CDELT1"] *
        np.arange(len(dataspec)))
    assert(aspcaphdu[1].header["CTYPE1"] == "LOG-LINEAR")
    modelspec = aspcaphdu[3].data
    modelwavelengths = 10**(
        aspcaphdu[3].header["CRVAL1"] + aspcaphdu[3].header["CDELT1"] *
        np.arange(len(modelspec)))
    assert(aspcaphdu[1].header["CTYPE1"] == "LOG-LINEAR")

    ax1.plot(datawavelengths, dataspec, color='k', marker='', ls="--")
    ax1.plot(modelwavelengths, modelspec, color='r', marker='', ls="-")
    ax1.annotate(
        "Companion\nFeature", xy=(15295.3, 0.89), xytext=(15295.1, 0.63), 
        arrowprops=dict(facecolor="black"), color="black",
        horizontalalignment="center")
    ax1.annotate(
        "Primary\nFeature", xy=(15298.5, 0.72), xytext=(15297.3, 0.63), 
        arrowprops=dict(facecolor="blue"), color="blue",
        horizontalalignment="center")
    ax2.plot(datawavelengths, dataspec, color='k', marker='', ls="--")
    ax2.plot(modelwavelengths, modelspec, color='r', marker='', ls="-")
    ax2.annotate(
        "Companion\nFeature", xy=(16719.3, 0.82), xytext=(16719.1, 0.63), 
        arrowprops=dict(facecolor="black"), color="black",
        horizontalalignment="center")
    ax2.annotate(
        "Primary\nFeature", xy=(16722.5, 0.72), xytext=(16721.3, 0.63), 
        arrowprops=dict(facecolor="blue"), color="blue",
        horizontalalignment="center")

    ax1.set_xlim(15291, 15302)
    ax1.set_ylim(0.55, 1.05)
    ax1.set_xlabel("")
    ax1.set_ylabel("Normalized Flux", fontsize=24)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax1.ticklabel_format(axis="x", useOffset=False)
    ax2.set_xlim(16717, 16726)
    ax2.set_ylim(0.55, 1.05)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2.ticklabel_format(axis="x", useOffset=False)
    ax1.set_xlabel("")
    ax2.set_ylabel("")

    # Make the bottom two panels
    epoch_change = "2M18534433+4323497"
    apstar_format = "apStar-{0}-{1}.fits"
    epoch_change_file = paths.HEAD_DIR / apstar_format.format(
        PIPELINE_VERSION[0:2], epoch_change)
    apstarhdu = fits.open(str(epoch_change_file))
    spec1 = apstarhdu[1].data[2,:]
    spec2 = apstarhdu[1].data[3,:]
    wavelengths = 10**(
        apstarhdu[1].header["CRVAL1"] + apstarhdu[1].header["CDELT1"] *
        np.arange(len(spec1)))
    assert(apstarhdu[1].header["CTYPE1"] == "LOG-LINEAR")

    ax3.plot(wavelengths, spec1, color="k", marker="", ls="-")
    ax4.plot(wavelengths, spec2, color="k", marker="", ls="-")
    ax4.annotate(
        "Companion\nFeatures", xy=(15743.3, 174.9), xytext=(15744.5, 118.0), 
        arrowprops=dict(facecolor="black"), color="black",
        horizontalalignment="center")
    ax4.annotate(
        "Companion\nFeatures", xy=(15751.1, 179.3), xytext=(15744.5, 118.0), 
        arrowprops=dict(facecolor="black"), color="black",
        horizontalalignment="center")
    ax4.annotate(
        "Companion\nFeatures", xy=(15767.8, 170.1), xytext=(15765.5, 118.0), 
        arrowprops=dict(facecolor="black"), color="black",
        horizontalalignment="center")
    ax4.annotate(
        "Primary\nFeatures", xy=(15746.0, 159.0), xytext=(15755.4, 118.0), 
        arrowprops=dict(facecolor="blue"), color="blue",
        horizontalalignment="center")
    ax4.annotate(
        "Primary\nFeatures", xy=(15753.2, 148.0), xytext=(15755.4, 118.0), 
        arrowprops=dict(facecolor="blue"), color="blue",
        horizontalalignment="center")
    ax4.annotate(
        "Primary\nFeatures", xy=(15769.5, 142.5), xytext=(15755.4, 118.0), 
        arrowprops=dict(facecolor="blue"), color="blue",
        horizontalalignment="center")

    ax3.set_xlim(15735, 15775)
    ax3.set_ylim(110, 240)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax3.ticklabel_format(axis="x", useOffset=False)
    ax3.set_xlabel("Wavelengths (Angstroms)", fontsize=24)
    ax3.set_ylabel(r"Flux ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)", fontsize=24)
    ax4.set_xlim(15735, 15775)
    ax4.set_ylim(110, 240)
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax4.ticklabel_format(axis="x", useOffset=False)
    ax4.set_xlabel("Wavelengths (Angstroms)", fontsize=24)
    ax4.set_ylabel("")

    plt.tight_layout()

@write_plot("f1")
def targeting_figure():
    '''Create figure showing where the two samples lie in the HR diagram.

    Asteroseismic targets should be blue while cool dwarfs ought to be red.'''
    clean_apogee = cache.clean_apogee_splitter()

#    f, (ax1, ax2) = plt.subplots(1,2, figsize=(24, 12))
    f, ax1 = plt.subplots(1, 1, figsize=figsize)
    cool_dwarfs = clean_apogee.subsample(["APOGEE_KEPLER_COOLDWARF"])
    apokasc_dwarf = clean_apogee.subsample(["APOGEE2_APOKASC_DWARF"])
    apokasc_giant = clean_apogee.subsample(["APOGEE2_APOKASC_GIANT"])
    apogee_EB = clean_apogee.subsample(["APOGEE_KEPLER_EB"])
    apogee2_EB = clean_apogee.subsample(["APOGEE2_EB"])
    apogee2_koi = clean_apogee.subsample(["APOGEE2_KOI"])
    apogee2_koi_control = clean_apogee.subsample(["APOGEE2_KOI_CONTROL"])
    apogee_seismic = clean_apogee.subsample(["APOGEE_KEPLER_SEISMO"])
    apogee2_monitor = clean_apogee.subsample(["APOGEE_RV_MONITOR_KEPLER"])
    apogee_hosts = clean_apogee.subsample(["APOGEE_KEPLER_HOST"])
    astero_dwarf = clean_apogee.subsample(["Asteroseismic"])
    fullsample = clean_apogee.subsample([])

    hr.absmag_teff_plot(
        apokasc_giant["TEFF"], apokasc_giant["M_K"], color=bc.black,
        marker=".", ls="", label="", axis=ax1, zorder=1)
    hr.absmag_teff_plot(
        apogee_seismic["TEFF"], apogee_seismic["M_K"], color=bc.black, 
        marker=".", ls="", label="Giants", axis=ax1)
    hr.absmag_teff_plot(
        apokasc_dwarf["TEFF"], apokasc_dwarf["M_K"], color=bc.brown, marker=".", 
        ls="", label="", axis=ax1, zorder=2)
    hr.absmag_teff_plot(
        cool_dwarfs["TEFF"], cool_dwarfs["M_K"], color=bc.brown, marker=".", 
        ls="", label="Dwarfs", axis=ax1, zorder=2)
#   hr.absmag_teff_plot(
#       apogee_EB["TEFF"], apogee_EB["M_K"], color=bc.sky_blue, marker="8", 
#       ls="", label="Eclipsing Binary", axis=ax1, zorder=4)
#   hr.absmag_teff_plot(
#       apogee2_EB["TEFF"], apogee2_EB["M_K"], color=bc.sky_blue, 
#       marker="8", ls="", label="", axis=ax1, zorder=4)
    hr.absmag_teff_plot(
        apogee2_koi["TEFF"], apogee2_koi["M_K"], color=bc.purple, 
        marker="d", ls="", label="KOI", axis=ax1, zorder=3)
    hr.absmag_teff_plot(
        apogee2_koi_control["TEFF"], apogee2_koi_control["M_K"],
        color=bc.purple, marker="d", ls="", label="", axis=ax1, zorder=3)
    hr.absmag_teff_plot(
        apogee2_monitor["TEFF"], apogee2_monitor["M_K"], color=bc.purple, 
        marker="d", ls="", label="", axis=ax1, zorder=3)
    hr.absmag_teff_plot(
        apogee_hosts["TEFF"], apogee_hosts["M_K"], color=bc.purple, 
        marker="d", ls="", label="", axis=ax1, zorder=3)
    hr.absmag_teff_plot(
        astero_dwarf["TEFF"], astero_dwarf["M_K"], color=bc.green, 
        marker="s", ls="", label="Asteroseismic Dwarfs", axis=ax1, zorder=5,
        ms=5)

    # Add separation between dwarfs and giants.
    hr.absmag_teff_plot(
        [5250, 5250, 3500], [-2.0, -0.45, -0.45], color=bc.sky_blue,
    ls="--", marker="", lw=7)

#   teff_bin_edges = np.arange(3500, 7000, 100)
#   mk_bin_edges = np.arange(-8, 8, 0.02)
#   count_cmap = plt.get_cmap("viridis")
#   count_cmap.set_under("white")
#   apogee_hist, xedges, yedges = np.histogram2d(
#       fullsample["TEFF"], fullsample["M_K"], 
#       bins=(teff_bin_edges, mk_bin_edges))
#   extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#   asp = (extent[1]-extent[0])/(extent[3]-extent[2])
#   im = ax2.imshow(apogee_hist.T, origin="lower", extent=extent,
#              aspect="auto", cmap=count_cmap, norm=Normalize(vmin=1, vmax=10))
#   f.colorbar(im, ax=ax2)

    # Show a 1 Gyr MIST Isochrone
#   test_teffs = np.linspace(3500, 7000, 100)
#   iso_ks = samp.calc_model_mag_fixed_age_feh_alpha(
#       test_teffs, 0.0, "Ks", age=1e9, model="MIST v1.1")
#   iso_ks_highmet = samp.calc_model_mag_fixed_age_feh_alpha(
#       test_teffs, 0.5, "Ks", age=1e9, model="MIST v1.1")
#   iso_ks_lowmet = samp.calc_model_mag_fixed_age_feh_alpha(
#       test_teffs, -0.5, "Ks", age=1e9, model="MIST v1.1")
#   ax2.plot(test_teffs, iso_ks_highmet, color=bc.pink, marker="", ls="--",
#            lw=2, label="[Fe/H] = 0.5")
#   ax2.plot(test_teffs, iso_ks, color=bc.pink, marker="", ls="-", lw=2,
#            label="[Fe/H] = 0.0")
#   ax2.plot(test_teffs, iso_ks_lowmet, color=bc.pink, marker="", ls=":", lw=2,
#            label="[Fe/H] = -0.5")

    # Add a representative error bar.
    dwarfs = fullsample["M_K"] > -2
    teff_error=np.median(fullsample[dwarfs]["TEFF_ERR"])
    median_k_errup = np.median(fullsample[dwarfs]["M_K_err1"]) 
    median_k_errdown = np.median(fullsample[dwarfs]["M_K_err2"])
    ax1.errorbar(
        [3700], [0.5], yerr=[[median_k_errdown], [median_k_errup]], 
        xerr=teff_error, elinewidth=3)

    # Now add isochrones
    lowmetiso = mist.MISTIsochrone.isochrone_from_file(-0.5)
    lowmet_table = lowmetiso.iso_table(1e9)
    hr.absmag_teff_plot(
        10**lowmet_table[lowmetiso.logteff_col], 
        lowmet_table[mist.band_translation["Ks"]], color=bc.pink,
        marker="", ls="--", label="", axis=ax1, lw=5, zorder=5)

    solmetiso = mist.MISTIsochrone.isochrone_from_file(0.0)
    solmet_table = solmetiso.iso_table(1e9)
    hr.absmag_teff_plot(
        10**solmet_table[solmetiso.logteff_col], 
        solmet_table[mist.band_translation["Ks"]], color=bc.pink,
        marker="", ls="-", label="MIST (1 Gyr)", axis=ax1, lw=5, zorder=5)

    highmetiso = mist.MISTIsochrone.isochrone_from_file(0.5)
    highmet_table = highmetiso.iso_table(1e9)
    hr.absmag_teff_plot(
        10**highmet_table[highmetiso.logteff_col], 
        highmet_table[mist.band_translation["Ks"]], color=bc.pink,
        marker="", ls=":", label="", axis=ax1, lw=5, zorder=5)

#    lowT, highT = 3500, 6600
#    full_T = np.linspace(lowT, highT, 200)
#    sol_ks = samp.calc_model_mag_fixed_age_feh_alpha(
#        full_T, 0.0, "Ks", age=1e9, model="MIST v1.2")
#    neg_ks = samp.calc_model_mag_fixed_age_feh_alpha(
#        full_T, -0.5, "Ks", age=1e9, model="MIST v1.2")
#    pos_ks = samp.calc_model_mag_fixed_age_feh_alpha(
#        full_T, 0.5, "Ks", age=1e9, model="MIST v1.2")
#    hr.absmag_teff_plot(
#        full_T, pos_ks, ls=":", marker="", color=bc.pink, lw=5,
#        label="", zorder=5)
#    hr.absmag_teff_plot(
#        full_T, sol_ks, ls="-", marker="", color=bc.pink, lw=5,
#        label="MIST (1 Gyr)", zorder=5)
#    hr.absmag_teff_plot(
#        full_T, neg_ks, ls="--", marker="", color=bc.pink, lw=5,
#        label="", zorder=5)

    ax1.set_xlim(6600, 3500)
    ax1.set_ylim(7.2, -2)
    ax1.set_xlabel("{0} (K)".format(Teffstr))
    ax1.set_ylabel(MKstr)
    ax1.legend(loc="lower left")
#   ax2.set_ylabel(MKstr)
#   ax2.set_xlim(7000, 3500)
#   ax2.set_ylim(7.2, -8)
#   ax2.set_xlabel("{0} (K)".format(Teffstr))
#   ax2.legend(loc="upper left")

    # Print out the number of objects in each category.
    print("Number of asteroseismic targets: {0:d}".format(
        len(apokasc_giant) + len(apogee_seismic)))
    print("Number of dwarfs: {0:d}".format(
        len(apokasc_dwarf) + len(cool_dwarfs)))
#   print("Number of EBs: {0:d}".format(len(apogee_EB)+len(apogee2_EB)))
    print("Number of Hosts: {0:d}".format(
        len(apogee2_koi) + len(apogee2_koi_control) + len(apogee2_monitor) +
        len(apogee_hosts)))

def logg_dwarf_subgiant_contrast():
    '''Show where the dwarfs and subgiants separate in the Teff-MK diagram.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    giants = aposplit.subsample(["Logg Giant"])
    dwarfs = aposplit.subsample(["Logg Dwarf"])

    hr.absmag_teff_plot(
        dwarfs["TEFF"], dwarfs["M_K"], color=bc.black, ls="", marker=".")
    hr.absmag_teff_plot(
        giants["TEFF"], giants["M_K"], color=bc.green, ls="", marker=".")

def pleiades_insync_calibration_overlap():
    '''Check overlap between IN-SYNC and the calibration clusters.'''
    pleiades = cache.pleiades_APOGEE_Literature_vsini()

    combined_targets = np.logical_and(
        pleiades["APOGEE_TARGET2"] & 2**10 > 0,
        pleiades["APOGEE2_TARGET3"] & 2**5 > 0)

    return pleiades[combined_targets]

def vsini_rapid_rotators():
    '''Plot the rapid rotators in the HR Diagram.'''
    apogee = cache.apogee_splitter_with_DSEP()

    apogee_tbl = apogee.subsample([])
    apogee_rapid = apogee.subsample(["Vsini det"])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        apogee_tbl["TEFF"], apogee_tbl["K Excess"], marker=".", color=bc.black,
        ls="", label="Full")
    hr.absmag_teff_plot(
        apogee_rapid["TEFF"], apogee_rapid["K Excess"], marker="o", color='red',
        ls="", label="vsini > 10 km/s")

    # Include evolutionary tracks.
    masses = [1.1, 1.3, 1.5]
    colors = ["blue", "green", "magenta"]
    for (m, c) in zip(masses, colors):
        track = mist.MISTEvolutionaryTrack.track_from_file(m, 0.0)
        track.restrict_phase([0, 2])
        trackteff = 10**track.tracktable[track.logteff_col]
        ms_kvals = samp.calc_model_mag_fixed_age_feh_alpha(
            trackteff, 0.0, "Ks", age=1e9, model="MIST v1.2")
        kdiff = track.tracktable[mist.band_translation["Ks"]] - ms_kvals

        hr.absmag_teff_plot(
            trackteff, kdiff, marker="", color=c, ls="-", 
            label="{0:.1f} Msun".format(m), lw=3)
    

    ax.set_xlabel("APOGEE Teff")
    ax.set_ylabel("K Excess")
    ax.legend(loc="upper right")

def vsini_upper_limits():
    '''Plot the vsini lower limits in the HR Diagram.'''
    apogee = cache.clean_apogee_splitter()

    apogee_tbl = apogee.subsample([])
    apogee_rapid = apogee.subsample(["Vsini lower"])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        apogee_tbl["TEFF"], apogee_tbl["M_K"], marker=".", color=bc.black,
        ls="", label="Full", axis=ax)
    hr.absmag_teff_plot(
        apogee_rapid["TEFF"], apogee_rapid["M_K"], marker="o", color='red',
        ls="", label="vsini > {0:.1f} km/s".format(10**(1.982-0.301/8)),
        axis=ax)

    ax.set_xlabel("APOGEE Teff")
    ax.set_ylabel("M_K")
    ax.legend(loc="lower left")

def plot_El_Badry_APOGEE():
    '''Plot the single and composite targets analyzed by El-Badry.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    singles = aposplit.subsample(["El-Badry Single"])
    sb1s = aposplit.subsample(["El-Badry SB1"])
    sb2s = aposplit.subsample(["El-Badry SB2"])
    hidden_triple = aposplit.subsample(["El-Badry Hidden Triple"])
    sb3s = aposplit.subsample(["El-Badry SB3"])
    noelb = aposplit.subsample(["No El-Badry Binarity"])

    # Read in the full El-Badry tables
    elb_sb2 = catin.read_El_Badry_SB2()
    kepler_elb_sb2 = catalog.join_by_2MASS_key(
        sb2s, elb_sb2, "APOGEE_ID", "APOGEE_ID", join_type="left")
    return kepler_elb_sb2

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        singles["TEFF"], singles["K Excess"], marker=".", color=bc.black,
        ls="", label="Single", axis=ax)
    hr.absmag_teff_plot(
        sb1s["TEFF"], sb1s["K Excess"], marker="s", color=bc.algae,
        ls="", label="SB1", axis=ax)
    hr.absmag_teff_plot(
        sb2s["TEFF"], sb2s["K Excess"], marker="*", color=bc.pink,
        ls="", label="SB2", axis=ax)
    hr.absmag_teff_plot(
        hidden_triple["TEFF"], hidden_triple["K Excess"], marker="d", 
        color=bc.sky_blue, ls="", label="Hidden Triple", axis=ax)
    hr.absmag_teff_plot(
        sb3s["TEFF"], sb3s["K Excess"], marker="p", color=bc.orange,
        ls="", label="SB3", axis=ax)
    hr.absmag_teff_plot(
        noelb["TEFF"], noelb["K Excess"], marker="x", color="grey",
        ls="", label="Not Analyzed", axis=ax, alpha=0.3)

    ax.set_xlabel("Teff (K)")
    ax.set_ylabel("K Excess")
    ax.legend(loc="upper right")

def plot_El_Badry_logg():
    '''Plot the single and composite targets analyzed by El-Badry.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    singles = aposplit.subsample(["El-Badry Single"])
    sb1s = aposplit.subsample(["El-Badry SB1"])
    sb2s = aposplit.subsample(["El-Badry SB2"])
    hidden_triple = aposplit.subsample(["El-Badry Hidden Triple"])
    sb3s = aposplit.subsample(["El-Badry SB3"])
    noelb = aposplit.subsample(["No El-Badry Binarity"])


    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.logg_teff_plot(
        singles["TEFF"], singles["LOGG_FIT"], marker=".", color=bc.black,
        ls="", label="Single", axis=ax)
    hr.logg_teff_plot(
        sb1s["TEFF"], sb1s["log g [dex]"], marker="s", color=bc.algae,
        ls="", label="SB1", axis=ax)
    hr.logg_teff_plot(
        sb2s["TEFF"], sb2s["log g [dex]"], marker="*", color=bc.pink,
        ls="", label="SB2", axis=ax)
    hr.logg_teff_plot(
        hidden_triple["TEFF"], hidden_triple["log g [dex]"], marker="d", 
        color=bc.sky_blue, ls="", label="Hidden Triple", axis=ax)
    hr.logg_teff_plot(
        sb3s["TEFF"], sb3s["log g [dex]"], marker="p", color=bc.orange,
        ls="", label="SB3", axis=ax)
    hr.logg_teff_plot(
        noelb["TEFF"], noelb["LOGG_FIT"], marker="x", color="grey",
        ls="", label="Not Analyzed", axis=ax, alpha=0.3)

    ax.set_xlabel("Teff (K)")
    ax.set_ylabel("APOGEE logg (cm/s/s)")
    ax.legend(loc="upper right")

def plot_El_Badry_APOGEE_logg_luminous_subgiants():
    '''Plot the single and composite targets for Luminous Subgiants.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    lum_single = aposplit.subsample(["El-Badry Single", "Luminous Subgiants"])
    lum_sb1 = aposplit.subsample(["El-Badry SB1", "Luminous Subgiants"])
    lum_sb2 = aposplit.subsample(["El-Badry SB2", "Luminous Subgiants"])
    lum_hidden = aposplit.subsample([
        "El-Badry Hidden Triple", "Luminous Subgiants"])
    lum_sb3 = aposplit.subsample(["El-Badry SB3", "Luminous Subgiants"])
    lum_noelb = aposplit.subsample([
        "No El-Badry Binarity", "Luminous Subgiants"])

    sub_single = aposplit.subsample(["El-Badry Single", "Subgiants"])
    sub_sb1 = aposplit.subsample(["El-Badry SB1", "Subgiants"])
    sub_sb2 = aposplit.subsample(["El-Badry SB2", "Subgiants"])
    sub_hidden = aposplit.subsample([
        "El-Badry Hidden Triple", "Subgiants"])
    sub_sb3 = aposplit.subsample(["El-Badry SB3", "Subgiants"])
    sub_noelb = aposplit.subsample([
        "No El-Badry Binarity", "Subgiants"])

    
    hotdwarfs = aposplit.subsample([
        "Hot Dwarfs"])
    cooldwarfs = aposplit.subsample([
        "Cool Dwarfs"])
    giants = aposplit.subsample([
        "Giants"])

    lum_singles = vstack([lum_single, lum_sb1])
    lum_binaries = vstack([lum_sb2, lum_hidden, lum_sb3])
    sub_singles = vstack([sub_single, sub_sb1])
    sub_binaries = vstack([sub_sb2, sub_hidden, sub_sb3])
    others = vstack([lum_noelb, sub_noelb, hotdwarfs, cooldwarfs, giants])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.logg_teff_plot(
        lum_singles["TEFF"], lum_singles["LOGG_FIT"], marker=".", 
        color=bc.sky_blue, ls="", label="El-Badry Single Luminous Subgiant", 
        axis=ax)
    hr.logg_teff_plot(
        lum_binaries["T_eff [K]"], lum_binaries["log g [dex]"], marker="*", 
        color=bc.sky_blue, ls="", label="El-Badry Binary Luminous Subgiant", 
        axis=ax, ms=12)
    hr.logg_teff_plot(
        lum_binaries["TEFF"], lum_binaries["LOGG_FIT"], marker="o", 
        color=bc.sky_blue, ls="", label="APOGEE Param",
        axis=ax, ms=12)
    if False:
        hr.logg_teff_plot(
            sub_singles["TEFF"], sub_singles["LOGG_FIT"], marker=".", 
            color=bc.algae, ls="", label="El-Badry Single Subgiant", 
            axis=ax)
        hr.logg_teff_plot(
            sub_binaries["T_eff [K]"], sub_binaries["log g [dex]"], marker="*", 
            color=bc.algae, ls="", label="El-Badry Binary Subgiant", 
            axis=ax, ms=12)
        hr.logg_teff_plot(
            sub_binaries["TEFF"], sub_binaries["LOGG_FIT"], marker="o", 
            color=bc.algae, ls="", label="",
            axis=ax, ms=12)
        for r in sub_binaries:
            hr.logg_teff_plot(
                [r["TEFF"], r["T_eff [K]"]], [r["LOGG_FIT"], r["log g [dex]"]],
                marker="", ls="-", color="k")
    hr.logg_teff_plot(
        others["TEFF"], others["LOGG_FIT"], marker="x", color="grey",
        ls="", label="Other", alpha=0.3, axis=ax)

    for r in lum_binaries:
        hr.logg_teff_plot(
            [r["TEFF"], r["T_eff [K]"]], [r["LOGG_FIT"], r["log g [dex]"]],
            marker="", ls="-", color="k")


    ax.set_xlabel("Teff (K)")
    ax.set_ylabel("APOGEE logg (cm/s/s)")
    ax.set_xlim(7000, 4800)
    ax.set_ylim(5.6, 2.8)
    ax.legend(loc="lower right")

def luminous_subgiant_vsini_distribution():
    aposplit = cache.apogee_splitter_with_DSEP()

    luminous_subgiants = aposplit.subsample(["Luminous Subgiants", "Low Alpha"])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    n, bins, patches = ax.hist(
        luminous_subgiants["VSINI"], bins=40, range=(0, 100), normed=False, 
        cumulative=False, color=bc.sky_blue, histtype="step", label="Full",
        linewidth=3)

    ax.text(40, 20, "Rapid Rotator Fraction: {0:.1f}%".format(
        aposplit.subsample_len([
            "Luminous Subgiants", "Low Alpha", "Vsini det"]) / 
        aposplit.subsample_len([
            "Luminous Subgiants", "Low Alpha", "~No Vsini"])*100))
    ax.plot([10, 10], [0, 45], 'r-')
    ax.plot([7, 7], [0, 45], 'r--')
    ax.set_xlabel("APOGEE Vsini")
    ax.set_ylabel("N")
    ax.set_title("Luminous Subgiants")

def subgiant_vsini_distribution():
    aposplit = cache.apogee_splitter_with_DSEP()

    subgiants = aposplit.subsample(["Subgiants", "Low Alpha"])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    n, bins, patches = ax.hist(
        subgiants["VSINI"], bins=40, range=(0, 100), normed=False, 
        cumulative=False, color=bc.sky_blue, histtype="step", label="Full",
        linewidth=3)

    ax.text(40, 40, "Rapid Rotator Fraction: {0:.1f}%".format(
        aposplit.subsample_len([
            "Subgiants", "Low Alpha", "Vsini det"]) / 
        aposplit.subsample_len([
            "Subgiants", "Low Alpha", "~No Vsini"])*100))
    ax.plot([7, 7], [0, 165], 'r--')
    ax.plot([10, 10], [0, 165], 'r-')
    ax.set_xlabel("APOGEE Vsini")
    ax.set_ylabel("N")
    ax.set_title("Subgiants")

def plot_vsini_against_luminosity():
    '''Correlate vsini against K-band luminosity.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    luminous_subgiants = aposplit.subsample(["Luminous Subgiants", "Low Alpha"])
    luminous_subgiants_rapid = luminous_subgiants["VSINI"] > 10
    subgiants = aposplit.subsample(["Subgiants", "Low Alpha"])
    subgiants_rapid = subgiants["VSINI"] > 10

    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(
        subgiants["M_K"][subgiants_rapid], subgiants["VSINI"][subgiants_rapid], 
        color=bc.algae, marker=".", ls="")
    ax.plot(
        luminous_subgiants["M_K"][luminous_subgiants_rapid], 
        luminous_subgiants["VSINI"][luminous_subgiants_rapid], color=bc.sky_blue, 
        marker=".", ls="")

    ax.set_xlabel("M_K")
    ax.set_ylabel("VSINI")


def plot_luminous_subgiant_alpha_poor():
    '''Plot the alpha poor rapid and slow rotators.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    luminous_subgiants = aposplit.subsample(["Luminous Subgiants"])
    alpha_rich_rapid = np.logical_and(
        luminous_subgiants["ALPHA_FE"] > 0.2, luminous_subgiants["VSINI"] > 10)
    alpha_rich_slow = np.logical_and(
        luminous_subgiants["ALPHA_FE"] > 0.2, luminous_subgiants["VSINI"] <= 10)
    others = aposplit.subsample(["~Luminous Subgiants"])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        luminous_subgiants["TEFF"], luminous_subgiants["K Excess"], 
        color=bc.sky_blue, marker=".", ls="")
    hr.absmag_teff_plot(
        luminous_subgiants["TEFF"][alpha_rich_rapid], 
        luminous_subgiants["K Excess"][alpha_rich_rapid], color='r',
        marker="*", ls="", ms=8)
    hr.absmag_teff_plot(
        luminous_subgiants["TEFF"][alpha_rich_slow], 
        luminous_subgiants["K Excess"][alpha_rich_slow], color='r',
        marker="o", ls="", ms=8)
    hr.absmag_teff_plot(
        others["TEFF"], others["K Excess"], color="grey", marker=".", ls="",
        ms=8, alpha=0.3)

    ax.set_xlabel("Teff (K)") 
    ax.set_ylabel("K Excess")

def plot_APOGEE_bins():
    '''Plot the different bins of evolutionary state.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    cool_dwarfs = aposplit.subsample(["Cool Dwarfs"])
    hot_dwarfs = aposplit.subsample(["Hot Dwarfs"])
    hot_subgiants = aposplit.subsample(["Subgiants"])
    luminous_subgiants = aposplit.subsample(["Luminous Subgiants"])
    giants = aposplit.subsample(["Giants"])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        cool_dwarfs["TEFF"], cool_dwarfs["K Excess"], marker=".", 
        color=bc.violet, ls="", label="Cool Dwarfs", axis=ax)
    hr.absmag_teff_plot(
        hot_dwarfs["TEFF"], hot_dwarfs["K Excess"], marker=".", 
        color=bc.orange, ls="", label="Hot Dwarfs", axis=ax)
    hr.absmag_teff_plot(
        hot_subgiants["TEFF"], hot_subgiants["K Excess"], marker=".", 
        color=bc.algae, ls="", label="Hot Subgiants", axis=ax)
    hr.absmag_teff_plot(
        luminous_subgiants["TEFF"], luminous_subgiants["K Excess"], marker=".", 
        color=bc.sky_blue, ls="", label="Luminous Subgiants", axis=ax)
    hr.absmag_teff_plot(
        giants["TEFF"], giants["K Excess"], marker=".", 
        color=bc.red, ls="", label="Giants", axis=ax)

    ax.set_xlabel("APOGEE Teff")
    ax.set_ylabel("K Excess")
    ax.set_xlim(6500, 3500)
    ax.set_ylim(1.0, -5.0) 
    ax.legend(loc="upper right")

def fractions_slow_stars():
    '''Describe the rapid rotator fraction for the slow star subsets.

    Calculate the rapid rotator fraction fo slow cool, luminous stars, and slow
    hot, less luminous stars.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    hot_rapid = (
        aposplit.subsample_len(["Vsini det", "~DLSB", "Slow Subgiants"]) +
        aposplit.subsample_len(["Vsini det", "~DLSB", "Red Stragglers"]))
    total_hot = (
        aposplit.subsample_len(["~DLSB", "Slow Subgiants"]) +
        aposplit.subsample_len(["~DLSB", "Red Stragglers"]))
    hot_rate = hot_rapid / total_hot*100
    hot_upper = au.binomial_upper(hot_rapid, total_hot)*100 - hot_rate
    hot_lower = hot_rate - au.binomial_lower(hot_rapid, total_hot)*100
    
    print("Fraction of hot rapid rotators: {0:.2f}+{1:.1f}-{2:.1f}%".format(
        hot_rate, hot_upper, hot_lower))

    cool_rapid = aposplit.subsample_len(["Vsini det", "~DLSB", "Slow Dwarfs"])
    total_cool = aposplit.subsample_len(["~DLSB", "Slow Dwarfs"])
    cool_rate = cool_rapid / total_cool*100
    cool_upper = au.binomial_upper(cool_rapid, total_cool)*100 - cool_rate
    cool_lower = cool_rate - au.binomial_lower(cool_rapid, total_cool)*100
    
    print("Fraction of cool rapid rotators: {0:.2f}+{1:.1f}-{2:.1f}%".format(
        cool_rate, cool_upper, cool_lower))


@write_plot("regimes")
def plot_APOGEE_bins_MK():
    '''Plot the different bins of evolutionary state.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    cool_dwarfs = aposplit.subsample(["Cool Singles"])
    photometric_binaries = aposplit.subsample(["Photometric Binaries"])
    subsubgiants = aposplit.subsample(["Subsubgiants"])
    slow_dwarfs = aposplit.subsample(["Slow Dwarfs"])
    slow_subgiants = aposplit.subsample(["Slow Subgiants"])
    fast_subgiants = aposplit.subsample(["Fast Subgiants"])
    red_stragglers = aposplit.subsample(["Red Stragglers"])
    blue_stragglers = aposplit.subsample(["Blue Stragglers"])
    giants = aposplit.subsample(["Giants"])

    f, ax = plt.subplots(1, 1, figsize=(15,15))
    hr.absmag_teff_plot(
        cool_dwarfs["TEFF"], cool_dwarfs["M_K"], marker=".", 
        color=bc.brown, ls="", label="Cool Dwarfs", axis=ax)
    hr.absmag_teff_plot(
        photometric_binaries["TEFF"], photometric_binaries["M_K"], marker=".", 
        color=bc.green, ls="", label="Photometric Binaries", axis=ax)
    hr.absmag_teff_plot(
        slow_dwarfs["TEFF"], slow_dwarfs["M_K"], marker=".", 
        color=bc.orange, ls="", label="Slow Dwarfs", axis=ax)
    hr.absmag_teff_plot(
        slow_subgiants["TEFF"], slow_subgiants["M_K"], marker=".", 
        color=bc.algae, ls="", label="Slow Subgiants", axis=ax)
    hr.absmag_teff_plot(
        fast_subgiants["TEFF"], fast_subgiants["M_K"], marker=".", 
        color=bc.light_pink, ls="", label="Fast Subgiants", axis=ax)
    hr.absmag_teff_plot(
        blue_stragglers["TEFF"], blue_stragglers["M_K"], marker=".",
        color=bc.blue, ls="", label="Blue Stragglers", axis=ax)
    hr.absmag_teff_plot(
        red_stragglers["TEFF"], red_stragglers["M_K"], marker=".",
        color=bc.red, ls="", label="Red Stragglers", axis=ax)
    hr.absmag_teff_plot(
        subsubgiants["TEFF"], subsubgiants["M_K"], marker=".",
        color=bc.sky_blue, ls="", label="Subsubgiants", axis=ax)
    hr.absmag_teff_plot(
        giants["TEFF"], giants["M_K"], marker=".", 
        color=bc.violet, ls="", label="Giants", axis=ax)

    lowT, highT = 3500, 6600
    ax.set_xlabel(Teffstr)
    ax.set_ylabel(MKstr)
    ax.set_xlim(highT, lowT)
    ax.set_ylim(7, -2) 
#   chartBox = ax.get_position()
#   legend_width = chartBox.width*0.3
#   ax.set_position([
#       chartBox.x0 + legend_width, chartBox.y0, chartBox.width - legend_width,
#       chartBox.height])
#   ax.legend(loc="lower left")
#    ax.legend(loc="upper center", bbox_to_anchor=(-0.4, 1.0))

def plot_regime_boundaries(ax, lowT, highT):
    '''Plot the boundaries of a regime on ax.'''
    dwarf_tempsep = 5250
    subgiant_tempsep = 5250
    luminous_subgiant_tempsep = 5250
    rgb_base_tempsep = 5250
    rgb_base_lowtemp = lowT
    dT = 3
    dM = 0.02
    w=3
    # Now plot the boundaries.
    cool_Ts = np.linspace(lowT, dwarf_tempsep, 100, endpoint=True)
    cool_dwarf_boundary = samp.calc_model_mag_fixed_age_feh_alpha(
        cool_Ts, 0.0, "Ks", age=1e9, model="MIST v1.2")
    hot_Ts = np.linspace(dwarf_tempsep, highT, 100, endpoint=True)
    hot_dwarf_boundary = samp.calc_model_mag_fixed_age_feh_alpha(
        hot_Ts, 0.0, "Ks", age=1e9, model="MIST v1.2")
    subgiant_Ts = np.linspace(subgiant_tempsep, highT, 100, endpoint=True)
    subgiant_boundary = samp.calc_model_mag_fixed_age_feh_alpha(
        subgiant_Ts, 0.0, "Ks", age=1e9, model="MIST v1.2")
    # Cool dwarfs
    hr.absmag_teff_plot(
        cool_Ts-dT, cool_dwarf_boundary+dM-0.3, marker="", ls="-", axis=ax,
        color=bc.violet, lw=w)
    hr.absmag_teff_plot(
        np.ones(2)*dwarf_tempsep-dT, np.array([cool_dwarf_boundary[-1]+dM-1.3, 7]), 
        marker="", ls="-", axis=ax, color=bc.violet, lw=w)
    # Photometric Binaries
    hr.absmag_teff_plot(
        cool_Ts-dT, cool_dwarf_boundary-dM-0.3, marker="", ls="-", axis=ax,
        color=bc.green, lw=w)
    hr.absmag_teff_plot(
        np.ones(2)*dwarf_tempsep-dT, np.array([
            cool_dwarf_boundary[-1]+dM-0.3, cool_dwarf_boundary[-1]+dM-1.3]), 
        marker="", ls="-", axis=ax, color=bc.green, lw=w)
    hr.absmag_teff_plot(
        cool_Ts-dT, cool_dwarf_boundary-dM-1.3, marker="", ls="-", axis=ax,
        color=bc.green, lw=w)
    # RGB Base
    hr.absmag_teff_plot(
        cool_Ts-dT, cool_dwarf_boundary-dM-1.3, marker="", ls="-", axis=ax,
        color=bc.light_pink, lw=w)
    hr.absmag_teff_plot(
        np.ones(2)*dwarf_tempsep-dT, np.array([
            cool_dwarf_boundary[-1]+dM-4.75, cool_dwarf_boundary[-1]+dM-1.3]), 
        marker="", ls="-", axis=ax, color=bc.light_pink, lw=w)
    hr.absmag_teff_plot(
        cool_Ts-dT, cool_dwarf_boundary+dM-4.75, marker="", ls="-", axis=ax,
        color=bc.light_pink, lw=w)
    # Hot dwarfs
    hr.absmag_teff_plot(
        hot_Ts+dT, hot_dwarf_boundary+dM-1.3, marker="", ls="-", axis=ax,
        color=bc.orange, lw=w)
    hr.absmag_teff_plot(
        np.ones(2)*dwarf_tempsep+dT, np.array([hot_dwarf_boundary[0]+dM-1.2, 7]),
        marker="", ls="-", axis=ax, color=bc.orange, lw=w)
    # Subgiants
    hr.absmag_teff_plot(
        subgiant_Ts+dT, subgiant_boundary-dM-1.3, marker="", ls="-", axis=ax,
        color=bc.algae, lw=w)
    hr.absmag_teff_plot(
        subgiant_Ts+dT, subgiant_boundary+dM-4.75, marker="", ls="-", axis=ax,
        color=bc.algae, lw=w)
    hr.absmag_teff_plot(
        np.ones(2)*subgiant_tempsep+dT, np.ones(2) * subgiant_boundary[0] + 
        np.array([-1.3-dM, -4.75+dM]), marker="", ls="-", axis=ax, 
        color=bc.algae, lw=w)

    # Giants
    hr.absmag_teff_plot(
        cool_Ts-dT, cool_dwarf_boundary-dM-4.75, marker="", ls="-", axis=ax,
        color=bc.red, lw=w)
    hr.absmag_teff_plot(
        np.ones(2)*dwarf_tempsep-dT, np.array([
            cool_dwarf_boundary[-1]-dM-4.75, cool_dwarf_boundary[-1]-dM-4.75]),
        marker="", ls="-", axis=ax, color=bc.red, lw=w)
    hr.absmag_teff_plot(
        hot_Ts-dT, subgiant_boundary-dM-4.75, marker="", ls="-", axis=ax,
        color=bc.red, lw=w)

def plot_APOGEE_bins_Lbol():
    '''Plot the different bins of evolutionary state.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    cool_dwarfs = aposplit.subsample(["Cool Dwarfs"])
    hot_dwarfs = aposplit.subsample(["Hot Dwarfs"])
    hot_subgiants = aposplit.subsample(["Subgiants"])
    luminous_subgiants = aposplit.subsample(["Luminous Subgiants"])
    giants = aposplit.subsample(["Giants"])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(
        cool_dwarfs["TEFF"], cool_dwarfs["Gaia L"], marker=".", 
        color=bc.violet, ls="", label="Cool Dwarfs")
    ax.plot(
        hot_dwarfs["TEFF"], hot_dwarfs["Gaia L"], marker=".", 
        color=bc.orange, ls="", label="Hot Dwarfs")
    ax.plot(
        hot_subgiants["TEFF"], hot_subgiants["Gaia L"], marker=".", 
        color=bc.algae, ls="", label="Hot Subgiants")
    ax.plot(
        luminous_subgiants["TEFF"], luminous_subgiants["Gaia L"], marker=".", 
        color=bc.sky_blue, ls="", label="Luminous Subgiants")
    ax.plot(
        giants["TEFF"], giants["Gaia L"], marker=".", 
        color=bc.red, ls="", label="Giants")
    ax.set_yscale("log")

    ax.set_xlabel("APOGEE Teff")
    ax.set_ylabel("Lbol")
    ax.set_xlim(6500, 3500)
    ax.set_ylim(10**-2, 10**4) 
    ax.legend(loc="upper left")

def plot_APOGEE_bin_rapid_rotators():
    '''Plot the rapid rotators on top of bins of evolutionary state.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    cool_dwarfs = aposplit.subsample(["Cool Dwarfs"])
    hot_dwarfs = aposplit.subsample(["Hot Dwarfs"])
    hot_subgiants = aposplit.subsample(["Subgiants"])
    luminous_subgiants = aposplit.subsample(["Luminous Subgiants"])
    giants = aposplit.subsample(["Giants"])

    lower_lim = aposplit.subsample(["Vsini lower"])
    rapid_rot = aposplit.subsample(["Vsini det"])
    marginal_rot = aposplit.subsample(["Vsini marginal"])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        cool_dwarfs["TEFF"], cool_dwarfs["M_K"], marker=".", 
        color=bc.violet, ls="", label="Cool Dwarfs", axis=ax)
    hr.absmag_teff_plot(
        hot_dwarfs["TEFF"], hot_dwarfs["M_K"], marker=".", 
        color=bc.orange, ls="", label="Hot Dwarfs", axis=ax)
    hr.absmag_teff_plot(
        hot_subgiants["TEFF"], hot_subgiants["M_K"], marker=".", 
        color=bc.algae, ls="", label="Subgiants", axis=ax)
    hr.absmag_teff_plot(
        luminous_subgiants["TEFF"], luminous_subgiants["M_K"], marker=".", 
        color=bc.sky_blue, ls="", label="Luminous Subgiants", axis=ax)
    hr.absmag_teff_plot(
        giants["TEFF"], giants["M_K"], marker=".", 
        color=bc.red, ls="", label="Giants", axis=ax)

    hr.absmag_teff_plot(
        rapid_rot["TEFF"], rapid_rot["M_K"], marker="o", 
        color='k', ls="", label="VSINI > 7 km/s", axis=ax, ms=3)
    hr.absmag_teff_plot(
        marginal_rot["TEFF"], marginal_rot["M_K"], marker="o", 
        color='k', ls="", label="", axis=ax, ms=3)
    hr.absmag_teff_plot(
        lower_lim["TEFF"], lower_lim["M_K"], marker="o", 
        color='k', ls="", label="", axis=ax, ms=3)

    ax.set_xlabel("APOGEE Teff")
    ax.set_ylabel("M_K")
    ax.set_xlim(6700, 3500)
    ax.set_ylim(6.5, -7.5) 
    ax.legend(loc="upper right")
    ax.set_title("Rapid Rotators in APOGEE")

def background_sample():
    '''Highlight the background sample where single-star rotators aren't.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    fullsamp = aposplit.subsample(["~Giants"])
    fullsamp = vstack([
        aposplit.subsample(["Hot Dwarfs", "~DLSB"]), 
        aposplit.subsample(["Subgiants", "~DLSB"]),
        aposplit.subsample(["RGB Base", "~DLSB"])])
    cooldwarfs = vstack([
        aposplit.subsample(["Cool Dwarfs", "~DLSB"]), 
        aposplit.subsample(["Photometric Binaries", "~DLSB"])])
    # First the hot stars.

    # Add on Jen's boundaries.
    jen_fast = jen.jen_fast_boundary()
    jen_fast_teff = jen_fast[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_fast_M_K = jen_fast[jen.format_Jen_column(0.0, 10, "M_K")]
    jen_fast_interp = interp1d(
        jen_fast_M_K[~jen_fast_M_K.mask],
        jen_fast_teff[~jen_fast_teff.mask], bounds_error=False)
    jen_min_M_K = np.ma.max(jen_fast_M_K)
    jen_max_M_K = np.ma.min(jen_fast_M_K)
    jen_max_teff = np.ma.max(jen_fast_teff)
    bottom_point = (6150, 3.15)
    print(jen_min_M_K)
    print(jen_max_M_K)

    boundary = np.zeros(len(fullsamp))

    MK_out_of_range = fullsamp["M_K"] < jen_max_M_K
    MK_boundary_jen = np.logical_and(
        fullsamp["M_K"] < jen_min_M_K, fullsamp["M_K"] >= jen_max_M_K)
    MK_boundary_linear = fullsamp["M_K"] >= jen_min_M_K
    
    boundary[MK_boundary_jen] = (
        jen_fast_interp(np.ma.filled(fullsamp["M_K"][MK_boundary_jen])))
    slope =  (jen_max_teff - bottom_point[0]) / (jen_min_M_K - bottom_point[1])
    boundary[MK_boundary_linear] = (
        bottom_point[0] + slope * (
            fullsamp["M_K"][MK_boundary_linear] - bottom_point[1]))


    singles = fullsamp["TEFF"] > boundary - 150
    not_singles = fullsamp["TEFF"] <= boundary - 150

    giant_corner = np.logical_and(fullsamp["TEFF"] < 5500, fullsamp["M_K"] < 2)
    giant_not_singles = np.logical_and(not_singles, giant_corner)
    dwarf_not_singles = np.logical_and(not_singles, ~giant_corner)


    rapid_rotators = fullsamp["VSINI"] > 10
    giant_not_singles_rapid_rotators = np.logical_and(
        giant_not_singles, rapid_rotators)
    dwarf_not_singles_rapid_rotators = np.logical_and(
        dwarf_not_singles, rapid_rotators)
    not_single_rapid_rotators = np.logical_and(not_singles, rapid_rotators)

    
    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        fullsamp["TEFF"][singles], fullsamp["M_K"][singles], color=bc.black,
        ls="", marker="x", axis=ax, alpha=0.3)
    hr.absmag_teff_plot(
        cooldwarfs["TEFF"], cooldwarfs["M_K"], color=bc.black, ls="",
        marker="x", axis=ax, alpha=0.3)
    hr.absmag_teff_plot(
        fullsamp["TEFF"][dwarf_not_singles], fullsamp["M_K"][dwarf_not_singles], 
        color=bc.black, ls="", marker=".", axis=ax)
    hr.absmag_teff_plot(
        fullsamp["TEFF"][giant_not_singles], fullsamp["M_K"][giant_not_singles], 
        color=bc.sky_blue, ls="", marker=".", axis=ax)
    hr.absmag_teff_plot(
        fullsamp["TEFF"][not_single_rapid_rotators], 
        fullsamp["M_K"][not_single_rapid_rotators], color=bc.pink,
        ls="", marker="o", axis=ax)


    giant_rapid_rots = np.count_nonzero(giant_not_singles_rapid_rotators)
    dwarf_rapid_rots = np.count_nonzero(dwarf_not_singles_rapid_rotators)
    giantcount = np.count_nonzero(giant_not_singles)
    dwarfcount = np.count_nonzero(dwarf_not_singles)
    print("Number of Giant Rapid Rotators: {0:d}".format(giant_rapid_rots))
    print("Total Giants: {0:d}".format(giantcount))
    print("Number of Dwarf Rapid Rotators: {0:d}".format(dwarf_rapid_rots))
    print("Total Dwarfs: {0:d}".format(dwarfcount))
    
    ax.set_xlim(6700, 3500)
    ax.set_ylim(6.5, -2) 
    ax.set_xlabel(Teffstr)
    ax.set_ylabel(MKstr)

#@write_plot("f4")
def plot_APOGEE_bins_vsini_sizes():
    '''Plot the APOGEE bins where size correlates with vsini.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    init_size = 1
    f, ax = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]+2))
    fullsamp = aposplit.subsample(["~Vsini det", "~DLSB", "~Giants"])
    hr.absmag_teff_plot(
        fullsamp["TEFF"], fullsamp["M_K"], marker="x", color="grey", ls="", 
        label="", axis=ax, ms=5, alpha=0.15)

    rapidrots = aposplit.subsample(["Vsini det", "~DLSB", "~Giants"])

    vsini_bins = np.array([10, 20, 30])
    indices = np.digitize(rapidrots["VSINI"], vsini_bins)

    for i in range(1, len(vsini_bins)+1):
        vsini_indices = indices == i
        size = init_size + 2*i
        if i == len(vsini_bins):
            label = "{0} > {1} {2}".format(vsinistr, vsini_bins[i-1], kmsstr)
        else:
            label = "{0} {1} <= {2} < {3} {1}".format(
                vsini_bins[i-1], kmsstr, vsinistr, vsini_bins[i])
        hr.absmag_teff_plot(
            rapidrots["TEFF"][vsini_indices],
            rapidrots["M_K"][vsini_indices], marker="o", color=bc.pink,
            ls="", ms=size, axis=ax, label=label)

    # Now plot the DLSBs
    dlsbs = aposplit.subsample(["DLSB", "~Giants"])

    hr.absmag_teff_plot(
        dlsbs["TEFF"], dlsbs["M_K"], marker="*", color=bc.sky_blue, ls="",
        ms=10, axis=ax, label="SB2")

    # Add on Jen's boundaries.
    jen_fast = jen.jen_fast_boundary()
    jen_slow = jen.jen_slow_boundary()
    jen_fast_teff = jen_fast[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_fast_M_K = jen_fast[jen.format_Jen_column(0.0, 10, "M_K")]
    jen_slow_teff = jen_slow[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_slow_M_K = jen_slow[jen.format_Jen_column(0.0, 10, "M_K")]

    hr.absmag_teff_plot(
        jen_fast_teff, jen_fast_M_K, color=bc.brown, marker="", ls="-",
        label="Fast launch", axis=ax, lw=5)
    hr.absmag_teff_plot(
        jen_slow_teff, jen_slow_M_K, color=bc.brown, marker="", ls="--",
        label="Slow launch", axis=ax, lw=5)

    add_boundaries(ax)


    ax.set_xlabel("{0} (K)".format(Teffstr))
    ax.set_ylabel(MKstr)
    ax.set_xlim(6700, 3500)
    ax.set_ylim(6.5, -2) 
    ax.legend(loc="lower left", fontsize=16)

def add_boundaries(ax, alpha=1.0):
    '''Add the boundaries for interesting subgroups.'''
    subgiant_rect = patches.Rectangle(
        (4800, 0.0), 1890, 1.5, linewidth=5, edgecolor=bc.light_pink,
        facecolor="None", alpha=alpha)
    ax.add_patch(subgiant_rect)

    blue_straggler_rect = patches.Rectangle(
        (5250, -2.1), 1600, 1.65, linewidth=5, edgecolor=bc.purple,
        facecolor="None", ls="--", alpha=alpha) 
    ax.add_patch(blue_straggler_rect)

    red_straggler_rect = patches.Rectangle(
        (4150, -0.45), 425, 2.45, linewidth=5, edgecolor=bc.violet, 
        facecolor="None", ls="--", alpha=alpha) 
    ax.add_patch(red_straggler_rect)

    Path = mpath.Path
    subsubgiant_temps = np.linspace(3500, 5250, 25)
    subsubgiant_ks = -2.5*np.log10(3) + samp.calc_model_mag_fixed_age_feh_alpha(
        subsubgiant_temps, 0.00, "Ks", age=1e9)
    verts = list(zip(subsubgiant_temps, subsubgiant_ks))
    verts.append((subsubgiant_temps[-1], 2))
    verts.append((subsubgiant_temps[0], 2))
    verts.append((subsubgiant_temps[0], subsubgiant_ks[0]))
    codes = [Path.LINETO]*len(verts)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    subsubgiant_path = mpath.Path(verts, codes)
    subsubgiant_patch = patches.PathPatch(
        subsubgiant_path, edgecolor=bc.violet, facecolor="None", linewidth=5,
        ls="--", alpha=alpha)
    ax.add_patch(subsubgiant_patch)

    # Split slow subgiants from slow dwarfs.
    plt.plot([5250, 5500, 5500], [2, 2, 1.0], color=bc.algae, ls=":", lw=5,
             label="", alpha=alpha)

@write_plot("f11a")
def subgiant_zoomin():
    '''Plot the rapid and slow rotators in the subgiant regime.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    rapids = aposplit.subsample(["Vsini det"])
    slows = aposplit.subsample(["~Vsini det"])

    f, ax = plt.subplots(1, 1, figsize=figsize)

    hr.absmag_teff_plot(
        slows["TEFF"], slows["M_K"], marker=".", color=bc.black, ls="", axis=ax, 
        label="Slow")
    hr.absmag_teff_plot(
        rapids["TEFF"], rapids["M_K"], marker="o", color=bc.pink, 
        ls="", axis=ax, label="Rapid", ms=10)
    ax.plot(
        [6700, 4800], [0.5, 0.5], marker="", ls="-", color=bc.light_pink, lw=5)
    ax.plot(
        [6700, 4800], [1.0, 1.0], marker="", ls="-", color=bc.light_pink, lw=5)
    ax.plot(
        [6700, 4800], [1.25, 1.25], marker="", ls="-", color=bc.light_pink, lw=5)

    # Read in the boundary once.
    jen_fast = jen.jen_fast_boundary()
    jen_slow = jen.jen_slow_boundary()
    jen_fast_teff = jen_fast[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_slow_teff = jen_slow[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_fast_kmag = jen_fast[jen.format_Jen_column(0.0, 10, "M_K")]
    jen_slow_kmag = jen_slow[jen.format_Jen_column(0.0, 10, "M_K")]

    hr.absmag_teff_plot(
        jen_fast_teff, jen_fast_kmag, color=bc.brown, marker="", ls="-",
        label="Fast launch", axis=ax, lw=4)
    hr.absmag_teff_plot(
        jen_slow_teff, jen_slow_kmag, color=bc.brown, marker="", ls="--",
        label="Slow launch", axis=ax, lw=4)

    ax.set_xlabel("{0} (K)".format(Teffstr))
    ax.set_ylabel(MKstr)
    ax.set_xlim(6700, 4800)
    ax.set_ylim(1.5, 0) 
    ax.legend(loc="upper left")

def rapid_rotator_fractions():
    '''Write out the number of rapid rotators in each sector of the HR diagram.'''

    aposplit = cache.apogee_splitter_with_DSEP()

    categories = [
        "Cool Dwarfs", "Hot Dwarfs", "Subgiants", "Luminous Subgiants"]
    template_str = "{0}: {1:d}/{2:d} = {3:.2f}%"
    for cat in categories:
        rapid_num = aposplit.subsample_len([cat, "Vsini det"])
        total = aposplit.subsample_len([cat, "~No Vsini"])

        print(template_str.format(
            cat, rapid_num, total, rapid_num / total * 100))

def photometric_rapid_rotator_fractions():
    '''Write out the number of photometrix rapid rotators in each sector of the 
    HR diagram.'''

    aposplit = cache.apogee_splitter_with_DSEP()
    mcq = catin.read_McQuillan_catalog()

    categories = [
        "Cool Dwarfs", "Hot Dwarfs", "Subgiants", "Luminous Subgiants"]
    template_str = "{0}: {1:d}/{2:d} = {3:.2f}%"
    for cat in categories:
        mcq_samp = aposplit.subsample([cat, "Mcq"])
        mcq_comb = au.join_by_id(mcq_samp, mcq, "kepid", "KIC",
                                 join_type="inner")
        maxps = rot.vsini_to_max_period(10, mcq_comb["Gaia R"])
        rapid_num = np.count_nonzero(mcq_comb["Prot"] < maxps)
        total = aposplit.subsample_len([cat, "~Unknown Mcq"])

        print(template_str.format(
            cat, rapid_num, total, rapid_num / total * 100))

def period_vsini_matrix(othercats=[]):
    '''Compare the number of periods and vsini in a given subsection.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    mcq = catin.read_McQuillan_catalog()
    # I will put in asteroseismic at the front.
    categories = [
        "Cool Singles", "Photometric Binaries", "Asteroseismic"]
    vsini_cats = ["Vsini det", "Vsini marginal", "Vsini nondet", "No Vsini"]
    dlsb_cats = ["DLSB", "~DLSB"]


    countmatrix=np.zeros(
        (len(vsini_cats), len(vsini_cats)+4, len(categories), len(dlsb_cats)), 
        dtype=np.int)

    for i, c in enumerate(categories):
        for j, v in enumerate(vsini_cats):
            for k, d in enumerate(dlsb_cats):
                if c is "Asteroseismic":
                    noperiod_cat = "No Garcia"
                    period_cat = "Garcia"
                    unknown_cat = "Not Garcia Asteroseismic"
                    periods = catin.read_Garcia_periods()
                else:
                    noperiod_cat = "No Mcq"
                    period_cat = "Mcq"
                    unknown_cat = "Unknown Mcq"
                    periods = catin.read_McQuillan_catalog()
                countmatrix[j, 3, i, k] = aposplit.subsample_len(
                    [c, v, noperiod_cat, d] + othercats)
                countmatrix[j, 4, i, k] = aposplit.subsample_len(
                    [c, v, unknown_cat, d] + othercats)
                countmatrix[j, 5, i, k] = aposplit.subsample_len(
                    [c, v, period_cat, d] + othercats)
                period_sample = aposplit.subsample(
                    [c, v, period_cat, d] + othercats)

                try:
                    full_periods = au.join_by_id(
                        period_sample, periods, "kepid", "KIC", join_type="inner")
                except ValueError:
                    continue
                assert len(full_periods) == len(period_sample)

                if c is "Cool Dwarfs":
                    radcol = "MIST R (APOGEE)"
                elif c is "Asteroseismic":
                    # Add in the radius column.
                    apokasc = catin.read_APOKASC_catalog()[[
                        "KEPLER_INT", "RADIUS_DW"]]
                    full_periods = au.join_by_id(
                        full_periods, apokasc, "kepid", "KEPLER_INT",
                        join_type="left")
                    radcol="RADIUS_DW"
                else:
                    radcol = "Gaia R"
                max_periods_lower = rot.vsini_to_max_period(
                    10, full_periods[radcol])
                max_periods_upper = rot.vsini_to_max_period(
                    7, full_periods[radcol])

                countmatrix[j, 0, i, k] = np.count_nonzero(
                    full_periods["Prot"] < max_periods_lower)
                countmatrix[j, 1, i, k] = np.count_nonzero(np.logical_and(
                    full_periods["Prot"] > max_periods_lower, 
                    full_periods["Prot"] < max_periods_upper))
                countmatrix[j, 2, i, k] = np.count_nonzero(
                    full_periods["Prot"] > max_periods_upper)

                velocities = rot.period_to_velocities(
                    full_periods["Prot"], full_periods[radcol])
                new_rapid_count = rot.calc_spec_rapid_num(np.log10(velocities))
                countmatrix[j, 6, i, k] = new_rapid_count


    return countmatrix

def format_fraction(num, denom):
    template = "{0:d}/{1:d} = {2:.2f}\%"
    try:
        percentage = num / denom * 100
    except ZeroDivisionError:
        percentage = np.nan
    
    return template.format(num, denom, percentage)

def write_detection_fractions():
    '''Write the detection fractions to a table.'''
    countmatrix = period_vsini_matrix()

    categories = [
        "Sample", "Cool Singles", "Photometric Binaries", "Asteroseismic"]
    rowcats = [
        r'\(\vsini > 10 \kms\) (Total APOGEE)', 
        # I want an \hline here.
        r'\(P < 2 \pi R / (10 \kms)\)',
        r'Predicted Rapid \vsini{} Fraction',
        # Another hline here.
        r'\(\vsini > 10 \kms\) (P Detections)', 
        r'\(\vsini > 10 \kms\) (P Nondetections)', 
        r'SB2 Fraction']
    num_matrix = []
    denom_matrix = []
    string_matrix = []
    for i in range(len(categories)-1):
        high_vsini_mcquillan = countmatrix[0, 5, i, 1]
        total_mcquillan = np.sum(countmatrix[:, 5, i, 1])
        high_vsini_nondetections = countmatrix[0, 3, i, 1]
        total_nondetections = np.sum(countmatrix[:, 3, i, 1])
        high_vsini_total = np.sum(countmatrix[0, 3:6, i, 1])
        total_nondlsb = np.sum(countmatrix[:, 3:6, i, 1])
        sb2_total = np.sum(countmatrix[:, 3:6, i, 0])
        total_apogee = np.sum(countmatrix[:, 3:6, i, :])

        mcquillan_rapid = np.sum(countmatrix[:, 0, i, :])
        total_mcquillan_2 = np.sum(countmatrix[:, 0:3, i, :])
        convolved_rapid = np.sum(countmatrix[:, 6, i, :])

        num_array = [
            high_vsini_total, mcquillan_rapid, convolved_rapid,
            high_vsini_mcquillan, high_vsini_nondetections, sb2_total]
        denom_array = [
            total_nondlsb, total_mcquillan_2, total_mcquillan_2,
            total_mcquillan, total_nondetections, total_apogee]

        num_matrix.append(num_array)
        denom_matrix.append(denom_array)
            

        strings = [
            format_fraction(n, d) for n, d in zip(num_array, denom_array)]
        string_matrix.append(strings)

    string_matrix.insert(0, rowcats)

    outputtab = Table(string_matrix, names=categories)

    # I want to split this long table up into two small tables.
    # Maybe change to splitdeluxetable for the journal article if desired.
    outputdwarfs = outputtab[[
        "Sample", "Cool Singles", "Photometric Binaries", "Asteroseismic"]]
    
    dwarftitle = r"Dwarf Rapid Rotator Fractions\label{tab:rapidfracdwarf}"
    alignment = "l " + " ".join([" c"]*(len(outputdwarfs.colnames)-1))
    dwarffootercomment = (
        r"\tablecomments{",
        r"The spectroscopic rapid rotator fraction of the category is the ",
        r"first row. The rotation period sample which is being compared to ",
        r"is that of \citet{McQuillan14}. To evaluate the agreement between ",
        r"the spectrscopic and photometric rapid rotator rates, compare the ",
        r"third and fourth lines. The spectroscopic rapid rotator fractions ",
        r"do not include SB2s.}")

    latexdict = {
        "col_align": alignment, "caption": dwarftitle, 
        "tablefoot": dwarffootercomment, "tabletype": "deluxetable*"}

    outputdwarfs.write(
        str(TABLE_PATH / "rapidfrac_dwarfs.tex"), format="ascii.aastex", 
        latexdict=latexdict, overwrite=True)

#   outputevolved = outputtab[[
#       "Sample", "Asteroseismic", "Subgiants", "RGB Base"]]

#   evolvedtitle = r"Evolved Rapid Rotator Fractions\label{tab:rapidfracevolved}"
#   alignment = "l " + " ".join([" c"]*(len(outputevolved.colnames)-1))
#   evolvedfootercomment = (
#       r"\tablecomments{",
#       r"Similar to \cref{tab:rapidfracdwarf}. Asteroseismic}",
#       r"statistics are based on \citet{Garcia14} periods and ",
#       r"asteroseismic radii.}")

#   latexdict = {
#       "col_align": alignment, "caption": evolvedtitle, 
#       "tablefoot": evolvedfootercomment, "tabletype": "deluxetable*"}

#   outputevolved.write(
#       str(TABLE_PATH / "rapidfrac_evolved.tex"), format="ascii.aastex", 
#       latexdict=latexdict, overwrite=True)

def write_high_alpha_detection_fractions():
    '''Write the detection fractions to a table for high-alpha stars.'''
    countmatrix = period_vsini_matrix(othercats=["High Alpha"])

    categories = [
        "Sample", "Cool Dwarfs", "Hot Dwarfs", "Subgiants"]
    rowcats = [
        r'SB2 Fraction', r'\(\vsini > 10 \kms\) (Mcq Detections)', 
        r'\(\vsini > 10 \kms\) (Mcq Nondetections)', 
        r'\(\vsini > 10 \kms\) (Total APOGEE)', r'\(P < 2 \pi R / (10 \kms)\)',
        r'Predicted Rapid \vsini{} Fraction']
    num_matrix = []
    denom_matrix = []
    string_matrix = []
    for i in range(len(categories)-1):
        high_vsini_mcquillan = countmatrix[0, 5, i, 1]
        total_mcquillan = np.sum(countmatrix[:, 5, i, 1])
        high_vsini_nondetections = countmatrix[0, 3, i, 1]
        total_nondetections = np.sum(countmatrix[:, 3, i, 1])
        high_vsini_total = np.sum(countmatrix[0, 3:6, i, 1])
        total_nondlsb = np.sum(countmatrix[:, 3:6, i, 1])
        sb2_total = np.sum(countmatrix[:, 3:6, i, 0])
        total_apogee = np.sum(countmatrix[:, 3:6, i, :])

        mcquillan_rapid = np.sum(countmatrix[:, 0, i, :])
        total_mcquillan_2 = np.sum(countmatrix[:, 0:3, i, :])
        convolved_rapid = np.sum(countmatrix[:, 6, i, :])

        num_array = [
            sb2_total, high_vsini_mcquillan, high_vsini_nondetections, 
            high_vsini_total, mcquillan_rapid, convolved_rapid]
        denom_array = [
            total_apogee, total_mcquillan, total_nondetections, total_nondlsb, 
            total_mcquillan_2, total_mcquillan_2]

        num_matrix.append(num_array)
        denom_matrix.append(denom_array)
            
        strings = [
            format_fraction(n, d) for n, d in zip(num_array, denom_array)]
        string_matrix.append(strings)

    string_matrix.insert(0, rowcats)

    outputtab = Table(string_matrix, names=categories)
    tabletitle = r"High Alpha Rapid Rotator Fractions\label{tab:highalphafrac}"
    alignment = " l c c c c "
    footercomment = (
        r"\tablecomments{Spectroscopic rapid rotator fraction is calculated "
        r"for three different samples: (1) Just those with significant periods " 
        r"detected in \citet{McQuillan14} (2) Just those which "
        r"\citet{McQuillan14} did not detect a significant period and (3) the "
        r"full sample, regardless of \citet{McQuillan14}}")

    latexdict = {
        "col_align": alignment, "caption": tabletitle, 
        "tablefoot": footercomment, "tabletype": "table*"}

    outputtab.write(
        str(TABLE_PATH / "highalphafrac.tex"), format="ascii.latex", 
        latexdict=latexdict, overwrite=True)


def elbadry_binary_fractions():
    '''Write out the El-badry binary fractions in each sector.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    categories = [
        "Cool Dwarfs", "Hot Dwarfs", "Subgiants", "Luminous Subgiants"]
    template_str = "{0}: {1:d}/{2:d} = {3:.2f}%"
    for cat in categories:
        sb2_num = aposplit.subsample_len([cat, "El-Badry SB2"])
        sb3_num = aposplit.subsample_len([cat, "El-Badry SB3"])
        trip_num = aposplit.subsample_len([cat, "El-Badry Hidden Triple"])
        total = aposplit.subsample_len([cat, "~No El-Badry Binarity"])

        sumnum = sb2_num + sb3_num + trip_num
        print(template_str.format(
            cat, sumnum, total, sumnum/total*100))

def elbadry_rotation_cross_table():
    '''Write out the number of targets in binarity and rotation bins.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    binarities = [
        "El-Badry Single", "El-Badry SB1", "El-Badry SB2", 
        "El-Badry Hidden Triple", "El-Badry SB3", "No El-Badry Binarity"]
    vsinis = ["Vsini det", "Vsini marginal", "Vsini nondet"]
    photbins = ["Photometric Binaries", "Photometric Singles"]

    countmatrix = np.zeros(
        (len(binarities)*len(photbins), len(vsinis)), dtype=np.int)
    # This is useful for checking if the countmatrix is working well.
    commmatrix = np.zeros(
        (len(binarities)*len(photbins), len(vsinis)), dtype='<U100')


    for i, b in enumerate(binarities):
        for j, v in enumerate(vsinis):
            for k, p in enumerate(photbins):
                countmatrix[k*len(binarities)+i, j] = aposplit.subsample_len(
                    ["Cool Dwarfs", b, v, p])
                commmatrix[k*len(binarities)+i, j] = " ".join(
                    ["Cool Dwarfs", p, b, v])

    return countmatrix

def elbadry_photbin_cross_table():
    '''Write out the number of targets in spectroscopic and photometric binaries.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    spectroscopic = [
        "El-Badry Single", "El-Badry SB1", "El-Badry SB2", 
        "El-Badry Hidden Triple", "El-Badry SB3"]
    photometric = ["Photometric Singles", "Photometric Binaries"]

    countmatrix = np.zeros((len(spectroscopic), len(photometric)), dtype=np.int)

    for i, b in enumerate(spectroscopic):
        for j, v in enumerate(photometric):
            countmatrix[i, j] = aposplit.subsample_len(["Cool Dwarfs", b, v])

    return countmatrix

def elbadry_binary_photometric_single_plot():
    '''Illustrate the HR diagram position of PS SB2s.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    photometric_singles = aposplit.subsample([
        "Photometric Singles", "~No El-Badry Binarity"])
    binary_sb2s = aposplit.subsample([
        "Photometric Binaries", "El-Badry SB2"])
    single_sb2s = aposplit.subsample([
        "Photometric Singles", "El-Badry SB2"])
    full = aposplit.subsample([
        "~Photometric Giants", "~No El-Badry Binarity"])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        full["TEFF"], full["ElBadry K Excess"], color="grey", marker=".", ls="",
        alpha=0.3, axis=ax, label="Photometric Binaries")
    hr.absmag_teff_plot(
        photometric_singles["TEFF"], photometric_singles["ElBadry K Excess"], 
        color=bc.black, marker=".", ls="", axis=ax, label="Photometric Singles")
    hr.absmag_teff_plot(
        binary_sb2s["TEFF"], binary_sb2s["ElBadry K Excess"], color=bc.sky_blue, 
        marker=".", ls="", axis=ax, label="SB2s")
    hr.absmag_teff_plot(
        single_sb2s["TEFF"], single_sb2s["ElBadry K Excess"], color="red", marker="*", 
        ls="", axis=ax, label="Photometric Single SB2s")

    ax.set_xlabel("Teff")
    ax.set_ylabel("K Excess")
    ax.set_title("Photometric Single SB2s")
    ax.legend(loc="lower left")

    # Add on the SB2 parameters to see what this is about.
    elbadry_sb2s = catin.read_El_Badry_SB2()
    combined_sb2s = catalog.join_by_2MASS_key(
        single_sb2s, elbadry_sb2s, "APOGEE_ID", "APOGEE_ID",
        conflict_suffixes=("_DUP", ""))
    return combined_sb2s

def elbadry_cool_dwarf_photometric_binary_comparison():
    '''Calculate the effective rate of El-Badry binaries in the cool dwarfs.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    pb_single = aposplit.subsample([
        "Cool Dwarfs", "Photometric Binaries", "El-Badry Single"])
    pb_sb1 = aposplit.subsample([
        "Cool Dwarfs", "Photometric Binaries", "El-Badry SB1"])
    pb_sb2 = aposplit.subsample([
        "Cool Dwarfs", "Photometric Binaries", "El-Badry SB2"])
    pb_ht = aposplit.subsample([
        "Cool Dwarfs", "Photometric Binaries", "El-Badry Hidden Triple"])
    pb_sb3 = aposplit.subsample([
        "Cool Dwarfs", "Photometric Binaries", "El-Badry SB3"])
    pb_singles = vstack([pb_single, pb_sb1])
    pb_multiples = vstack([pb_sb2, pb_sb3, pb_ht])
    ps_single = aposplit.subsample([
        "Cool Dwarfs", "Photometric Singles", "El-Badry Single"])
    ps_sb1 = aposplit.subsample([
        "Cool Dwarfs", "Photometric Singles", "El-Badry SB1"])
    ps_sb2 = aposplit.subsample([
        "Cool Dwarfs", "Photometric Singles", "El-Badry SB2"])
    ps_ht = aposplit.subsample([
        "Cool Dwarfs", "Photometric Singles", "El-Badry Hidden Triple"])
    ps_sb3 = aposplit.subsample([
        "Cool Dwarfs", "Photometric Singles", "El-Badry SB3"])
    ps_singles = vstack([ps_single, ps_sb1])
    ps_multiples = vstack([ps_sb2, ps_sb3, ps_ht])

    not_analyzed = aposplit.subsample([
        "Cool Dwarfs", "No El-Badry Binarity"])

    f, ax = plt.subplots(1, 1, figsize=figsize)

    hr.absmag_teff_plot(
        not_analyzed["TEFF"], not_analyzed["M_K"], color="grey", alpha=0.4,
        marker=".", ls="", label="Not Analyzed", axis=ax)
    hr.absmag_teff_plot(
        ps_singles["TEFF"], ps_singles["M_K"], color="black", marker=".",
        ls="", label="Photometric Singles", axis=ax)
    hr.absmag_teff_plot(
        pb_singles["TEFF"], pb_singles["M_K"], color="red", marker=".",
        ls="", label="Photometric Binaries", axis=ax)
    hr.absmag_teff_plot(
        ps_multiples["TEFF"], ps_multiples["M_K"], color="black", marker="o",
        ls="", label="El-Badry Binary", mec="blue", mew=2, axis=ax)
    hr.absmag_teff_plot(
        pb_multiples["TEFF"], pb_multiples["M_K"], color="red", marker="o",
        ls="", label="", mec="blue", mew=2, axis=ax)

    ax.set_xlabel("Teff")
    ax.set_ylabel("M_K")

    n_pb_single = (aposplit.subsample_len([
        "Cool Dwarfs", "Photometric Binaries", "El-Badry Single"]) + 
                   aposplit.subsample_len([
        "Cool Dwarfs", "Photometric Binaries", "El-Badry SB1"]))
    n_pb_multiple = (aposplit.subsample_len([
        "Cool Dwarfs", "Photometric Binaries", "El-Badry SB2"]) +
                     aposplit.subsample_len([
        "Cool Dwarfs", "Photometric Binaries", "El-Badry Hidden Triple"]) +
                     aposplit.subsample_len([
        "Cool Dwarfs", "Photometric Binaries", "El-Badry SB3"]))

    n_ps_single = (aposplit.subsample_len([
        "Cool Dwarfs", "Photometric Singles", "El-Badry Single"]) +
                   aposplit.subsample_len([
        "Cool Dwarfs", "Photometric Singles", "El-Badry SB1"]))
    n_ps_multiple = (aposplit.subsample_len([
        "Cool Dwarfs", "Photometric Singles", "El-Badry SB2"]) +
                     aposplit.subsample_len([
        "Cool Dwarfs", "Photometric Singles", "El-Badry Hidden Triple"]) +
                     aposplit.subsample_len([
        "Cool Dwarfs", "Photometric Singles", "El-Badry SB3"]))

    print("Photometric Singles classified as Singles: {0:d}".format(
        n_ps_single))
    print("Photometric Singles classified as Binaries: {0:d}".format(
        n_ps_multiple))
    print("Photometric Binaries classified as Singles: {0:d}".format(
        n_pb_single))
    print("Photometric Binaries classified as Binaries: {0:d}".format(
        n_pb_multiple))


def teff_comparison_single_sb2():
    '''Show how the El-Badry Teff differs between photometric and
    non-photometric binaries.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    binary_sb2s = aposplit.subsample([
        "Photometric Binaries", "El-Badry SB2"])
    single_sb2s = aposplit.subsample([
        "Photometric Singles", "El-Badry SB2"])

    # Add on the SB2 parameters to see what this is about.
    elbadry_sb2s = catin.read_El_Badry_SB2()
    combined_single_sb2s = catalog.join_by_2MASS_key(
        single_sb2s, elbadry_sb2s, "APOGEE_ID", "APOGEE_ID",
        conflict_suffixes=("_DUP", ""))
    combined_binary_sb2s = catalog.join_by_2MASS_key(
        binary_sb2s, elbadry_sb2s, "APOGEE_ID", "APOGEE_ID",
        conflict_suffixes=("_DUP", ""))

    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(
        combined_single_sb2s["TEFF"]/combined_single_sb2s["SDSS-Teff"], 
        combined_single_sb2s["T_eff [K]"]/combined_single_sb2s["SDSS-Teff"], 
        'r.')
    ax.plot(
        combined_binary_sb2s["TEFF"]/combined_binary_sb2s["SDSS-Teff"], 
        combined_binary_sb2s["T_eff [K]"]/combined_binary_sb2s["SDSS-Teff"], 
        'c.')

    ax.set_xlabel("APOGEE Teff / Pinsonneault Teff")
    ax.set_ylabel("El-Badry Teff / Pinsonneault Teff")

def elbadry_sb2_identification():
    '''Determine how often El-Badry determines visually classified SB2s.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    correct_nonidentification = aposplit.subsample_len([
        "El-Badry Single", "No DLSB"])
    correct_identification = aposplit.subsample_len([
        "El-Badry SB2", "DLSB"])
    el_badry_det_sb2 = aposplit.subsample_len([
        "El-Badry SB2", "No DLSB"])
    el_badry_missed_sb2 = aposplit.subsample_len([
        "El-Badry Single", "DLSB"])

    confusion_matrix = Table([
        ["No Visual SB2", "Visual SB2"], 
        [correct_nonidentification, el_badry_missed_sb2],
        [el_badry_det_sb2, correct_identification]], names=(
            " ", "El-Badry Single", "El-Badry SB2"))

    
    tabletitle = r"El-Badry/Visual classification\label{tab:elbadryconfusion}"
    alignment = " l c c "
    footercomment = (
        r"\tablecomments{Total number is much smaller than the sum of surveys "
        r"because of large overlap between surveys. Resolution is the lowest "
        r"resolution of the sample.}")

    latexdict = {
        "col_align": alignment, "caption": tabletitle}

    confusion_matrix.write(
        str(TABLE_PATH / "confmatrix.tex"), format="ascii.latex", 
        latexdict=latexdict, overwrite=True, fill_values=[("0", "", "R")])

def hot_elbadry_q_comparison():
    '''Compare the spectroscopic mass-ratio distributions for hot stars.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    elbadry_sb2s = catin.read_El_Badry_SB2()

    subgiants = ["Hot Dwarfs", "Subgiants", "Luminous Subgiants"]
    f, ax = plt.subplots(1, 1, figsize=figsize)
    for s in subgiants:
        sb2s = aposplit.subsample([s, "El-Badry SB2"])
        sb2_combo = catalog.join_by_2MASS_key(
            sb2s, elbadry_sb2s, "APOGEE_ID", "APOGEE_ID",
            conflict_suffixes=("_DUP", ""))
        ax.hist(sb2_combo["q_spec"], bins=np.linspace(0.5, 1.1, 5+1),
                normed=False, histtype="step", label=s)

    ax.set_xlabel("El-Badry Mass Ratio")
    ax.set_ylabel("N")
    ax.legend()

@write_plot("triple_vsini", toplevel=PLOT_PATH)
def plot_cool_rotation():
    '''Plot the rapid rotator agreement for cool dwarfs.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    apokasc_split = cache.astero_splitter()
    cool_apo = aposplit.subsample(["Dwarfs", "APOGEE Evolution Cool", "Mcq", "~DLSB"])
    apo = aposplit.subsample(["Dwarfs", "Mcq", "~DLSB"])
    apokasc = apokasc_split.subsample([
        "Asteroseismic Dwarfs", "~Bad", "~No vsini", "~DLSB", "Mcq"])
    mcq = catin.read_McQuillan_catalog()
    garcia = catin.read_Garcia_periods()
    apo_mcq = au.join_by_id(apo, mcq, "kepid", "KIC")
    cool_apo_mcq = au.join_by_id(cool_apo, mcq, "kepid", "KIC")
    astero_mcq = au.join_by_id(apokasc, garcia, "KEPLER_INT", "KIC")
    apo_mcq_rapid = apo_mcq[apo_mcq["Prot"] < 1.5]

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 12))
    rot.plot_vsini_velocity(
        cool_apo_mcq["VSINI"], cool_apo_mcq["Prot"], cool_apo_mcq["e_Prot"],
        cool_apo_mcq["MIST R (APOGEE)"], cool_apo_mcq["MIST R Err (APOGEE)"], ax=ax1)
    astero_radius_err = (
        (astero_mcq["RADIUS_DW_PERR"] + astero_mcq["RADIUS_DW_MERR"])/2)
    rot.plot_vsini_velocity(
        astero_mcq["VSINI"], astero_mcq["Prot"], astero_mcq["e_Prot"], 
        astero_mcq["RADIUS_DW"], astero_radius_err, ax=ax2)
    rot.plot_vsini_velocity(
        apo_mcq["VSINI"], apo_mcq["Prot"], apo_mcq["e_Prot"], 
        apo_mcq["Gaia R"], apo_mcq["Gaia R err"], ax=ax3)
    rot.plot_vsini_velocity(
        apo_mcq_rapid["VSINI"], apo_mcq_rapid["Prot"], apo_mcq_rapid["e_Prot"], 
        apo_mcq_rapid["Gaia R"], apo_mcq_rapid["Gaia R err"],
        color=bc.sky_blue, ax=ax3)
    ax1.set_title("Cool Dwarfs")
    ax2.set_title("Asteroseismic")
    ax3.set_title("Full Gaia")

########################
# Asteroseismic Sample #
########################

def asteroseismic_targets():
    '''Show location of asteroseismic targets on the HR Diagram.'''
    astero = cache.astero_splitter()
    apokasc = astero.subsample([
        "Asteroseismic Dwarfs", "~Bad", "~No vsini", "~DLSB"])
    full_apo = astero.subsample(["~Bad"])
    garcia = catin.read_Garcia_periods()
    astero_garcia = au.join_by_id(apokasc, garcia, "KEPLER_INT", "KIC")

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    hr.absmag_teff_plot(
        full_apo["TEFF_COR"], full_apo["M_K"], color="k", marker=".",
        alpha=0.5, ls="", label="")
    hr.absmag_teff_plot(
        astero_garcia["TEFF_COR"], astero_garcia["M_K"], color="r", marker="o",
        ls="", label="Asteroseismic")
    ax.set_xlim(6750, 3600)
    ax.set_ylim(6.5, 0.15)
    ax.set_xlabel("APOGEE Teff (K)")
    ax.set_ylabel("MK")
    ax.legend(loc="lower left")

def Bruntt_comparison():
    '''Compare the APOGEE vsini to the Bruntt vsini.'''
    bruntt = catin.bruntt_dr14_overlap()

    f, ax = plt.subplots(1, 1, figsize=figsize)
    vdiff = np.log10(bruntt["VSINI"]) - np.log10(bruntt["vsini"])
    ax.plot(bruntt["vsini"], vdiff, 'ko')
    ax.plot([1, 100], [0, 0], 'k-')
    ax.set_xscale("log")
    ax.set_xlabel("Bruntt vsini")
    ax.set_ylabel("Log(APOGEE vsini / Bruntt vsini)")

@write_plot("f6")
def asteroseismic_vsini():
    '''Plot the vsini agreement for the asteroseismic sample.'''
    astero = cache.astero_splitter()
    apokasc = astero.subsample([
        "Asteroseismic Dwarfs", "~Bad", "~No vsini", "~DLSB"])
    dlsbs = astero.subsample(["Asteroseismic Dwarfs", "~Bad", "~No vsini", "DLSB"])
    apo_dwarfs = astero.subsample(["~Bad", "Asteroseismic Dwarfs"])
    garcia = catin.read_Garcia_periods()
    astero_garcia = au.join_by_id(apokasc, garcia, "KEPLER_INT", "KIC")
    dlsb_garcia = au.join_by_id(dlsbs, garcia, "KEPLER_INT", "KIC")
    dwarf_garcia = au.join_by_id(apo_dwarfs, garcia, "KEPLER_INT", "KIC")

    f, ax = plt.subplots(1, 1, figsize=figsize)
    astero_radius_err = (
        (astero_garcia["RADIUS_DW_PERR"] + astero_garcia["RADIUS_DW_MERR"])/2)
    dlsb_radius_err = (
        (dlsb_garcia["RADIUS_DW_PERR"] + dlsb_garcia["RADIUS_DW_MERR"])/2)
    rot.plot_vsini_velocity(
        astero_garcia["VSINI"], astero_garcia["Prot"], astero_garcia["e_Prot"], 
        astero_garcia["RADIUS_DW"], astero_radius_err, ax=ax,
        label="")
    rot.plot_vsini_velocity(
        dlsb_garcia["VSINI"], dlsb_garcia["Prot"], dlsb_garcia["e_Prot"], 
        dlsb_garcia["RADIUS_DW"], dlsb_radius_err, ax=ax, color=bc.sky_blue,
        label="SB2", marker="*", ms=10)

#   ax.set_xlabel("Veq")
#   ax.set_ylabel("Vsini")
    ax.legend(loc="upper left")

    # I want to make sure upper limits are actually detected as lower limits.
    dwarf_velocities = rot.period_to_velocities(
        dwarf_garcia["Prot"], dwarf_garcia["RADIUS_DW"])
    sini_cutoff = 0.5

    print(
        "Asteroseismic sample with Garcia periods: {0:d}".format(
            len(dwarf_garcia)))
    print("Number with nontrivial rotation: {0:d}".format(
        np.count_nonzero(np.logical_or(
            dwarf_velocities > 10, dwarf_garcia["VSINI"].filled(0.0) > 10))))
    phot_rapid_rotators = np.count_nonzero(dwarf_velocities > 10)
    phot_rapid_frac = phot_rapid_rotators / len(dwarf_velocities)*100
    print(
        "Fraction of photometric rapid rotators: " + 
        "{0:d}/{1:d}={2:.1f}+{3:.1f}-{4:.1f}\%".format(
            phot_rapid_rotators, len(dwarf_velocities), phot_rapid_frac, 
            au.binomial_upper(phot_rapid_rotators, len(dwarf_velocities))*100 - 
            phot_rapid_frac, phot_rapid_frac -
            au.binomial_lower(phot_rapid_rotators, len(dwarf_velocities))*100))
    pred_rapid_rotators = rot.calc_spec_rapid_num(
        np.log10(dwarf_velocities), sini_cutoff=sini_cutoff)
    pred_rapid_frac = pred_rapid_rotators / len(dwarf_velocities)*100
    print(
        "Predicted fraction of spectroscopic rapid rotators: " + 
        "{0:.2f}/{1:d}={2:.1f}+{3:.1f}-{4:.1f}\%".format(
            pred_rapid_rotators, len(dwarf_velocities), pred_rapid_frac, 
            au.binomial_upper(pred_rapid_rotators, len(dwarf_velocities))*100 - 
            pred_rapid_frac, pred_rapid_frac -
            au.binomial_lower(pred_rapid_rotators, len(dwarf_velocities))*100))
    # Filling so that the evolved stars without vsini calculated by APOGEE are
    # counted as slow rotators.
    spec_rapid_rotators = np.count_nonzero(
        dwarf_garcia["VSINI"].filled(0.0) > 10)
    spec_rapid_frac = spec_rapid_rotators / len(dwarf_velocities)*100
    print(
        "Fraction of spectroscopic rapid rotators: " + 
        "{0:d}/{1:d}={2:.1f}+{3:.1f}-{4:.1f}\%".format(
            spec_rapid_rotators, len(dwarf_velocities), spec_rapid_frac, 
            au.binomial_upper(spec_rapid_rotators, len(dwarf_velocities))*100 - 
            spec_rapid_frac, spec_rapid_frac -
            au.binomial_lower(spec_rapid_rotators, len(dwarf_velocities))*100))


def asteroseismic_vsini_with_Gaia():
    '''Plot the vsini agreement for the asteroseismic sample.'''
    astero = cache.astero_splitter()
    apokasc = astero.subsample([
        "Asteroseismic Dwarfs", "~Bad", "~No vsini", "~DLSB"])
    full = cache.apogee_splitter_with_DSEP()
    fulltable = full.subsample([])
    fullastero = au.join_by_id(
        apokasc, fulltable, "KEPLER_INT", "kepid", join_type="left",
        conflict_suffixes=("_APOKASC", "_APOGEE"))
    garcia = catin.read_Garcia_periods()
    astero_garcia = au.join_by_id(fullastero, garcia, "KEPLER_INT", "KIC")

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    rot.plot_vsini_velocity(
        astero_garcia["VSINI_APOKASC"], astero_garcia["Prot"], 
        astero_garcia["e_Prot"], astero_garcia["Gaia R"], 
        astero_garcia["Gaia R err"], ax=ax)

#   ax.set_xlabel("Veq")
#   ax.set_ylabel("Vsini")
    ax.set_title("Asteroseismic vsini agreement")

def asteroseismic_gaia_radius_comparison():
    astero = cache.astero_splitter()
    apokasc = astero.subsample(["Asteroseismic Dwarfs"])
    full = cache.apogee_splitter_with_DSEP()
    fulltable = full.subsample([])
    apokasc_full = au.join_by_id(
        apokasc, fulltable, "KEPLER_INT", "kepid", 
        conflict_suffixes=("_APOGEE", "_APOKASC"))

    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.errorbar(
        apokasc_full["RADIUS_DW"], apokasc_full["Gaia R"],
        yerr=apokasc_full["Gaia R err"], 
        xerr=[-apokasc_full["RADIUS_DW_MERR"], apokasc_full["RADIUS_DW_PERR"]],
    marker=".", color="k", ls="")
    # I also want to plot a linear fit
    dw_rad_err = (
        apokasc_full["RADIUS_DW_PERR"] - apokasc_full["RADIUS_DW_MERR"]) / 2
    combined_err = np.sqrt(apokasc_full["Gaia R err"]**2 + (
        apokasc_full["RADIUS_DW_MERR"] - apokasc_full["RADIUS_DW_PERR"])**2)
    indices = ~np.ma.getmaskarray(combined_err)
    coeff, cov = np.polyfit(
        apokasc_full["RADIUS_DW"][indices], apokasc_full["Gaia R"][indices], 
        1, w=1/combined_err[indices], cov="unscaled")
    print(combined_err[indices])
    asterorads = np.linspace(0.8, 5.0, 2, endpoint=True)
    gaiarads = coeff[1] + coeff[0] * asterorads
    ax.plot(asterorads, gaiarads, 'r-')
    print("Fit equation is y = {0:.2f} x + {1:.2f}".format(coeff[0], coeff[1]))
    print("Slope error: {0:.3f} Intercept error: {1:.3f}".format(
        np.sqrt(cov[1,1]), np.sqrt(cov[0,0])))

    ax.plot(asterorads, asterorads, 'k-')
    ax.set_xlabel("Asteroseismic R")
    ax.set_ylabel("Gaia R")
    ax.set_xlim(asterorads[0], asterorads[1])
    ax.set_ylim(asterorads[0], asterorads[1])


    ratio = (apokasc_full["RADIUS_DW"] - apokasc_full["Gaia R"])**2 / (
        dw_rad_err**2 + apokasc_full["Gaia R err"]**2)
    num_pairs = np.count_nonzero(indices)
    chisq = np.sum(ratio)
    chisq_dof = chisq / num_pairs
    print("Overlap sample size is {0:d}".format(num_pairs))
    print("Reduced chi-squared is {0:.2f}".format(chisq_dof))

    fracunc = np.sqrt(np.mean(
        (apokasc_full["RADIUS_DW"] - apokasc_full["Gaia R"])**2 /
         apokasc_full["RADIUS_DW"]**2))
    print("RMS fractional uncertainty is {0:.1f}%".format(fracunc*100))

@write_plot("astero_gaia_radcomp")
def asteroseismic_gaia_logradius_comparison():
    astero = cache.astero_splitter()
    apokasc = astero.subsample(["Asteroseismic Dwarfs"])
    full = cache.apogee_splitter_with_DSEP()
    fulltable = full.subsample([])
    apokasc_full = au.join_by_id(
        apokasc, fulltable, "KEPLER_INT", "kepid", 
        conflict_suffixes=("_APOGEE", "_APOKASC"))

    f, ax = plt.subplots(1, 1, figsize=figsize)
    astero_lograd = np.log10(apokasc_full["RADIUS_DW"])
    gaia_lograd = np.log10(apokasc_full["Gaia R"])
    dw_rad_err = (
        apokasc_full["RADIUS_DW_PERR"] - apokasc_full["RADIUS_DW_MERR"]) / 2
    astero_lograd_err = dw_rad_err / apokasc_full["RADIUS_DW"] / np.log(10)
    gaia_lograd_err = (
        apokasc_full["Gaia R err"] / apokasc_full["Gaia R"] / np.log(10))
    ax.errorbar(
        astero_lograd, gaia_lograd, yerr=gaia_lograd_err,
        xerr=astero_lograd_err, marker=".", color="k", ls="")
    # I also want to plot a linear fit
    combined_err = np.sqrt(gaia_lograd_err**2 + astero_lograd_err**2)
    notmasked = ~np.ma.getmaskarray(combined_err)
    coeff, cov = np.polyfit(
        astero_lograd[notmasked], gaia_lograd[notmasked], 1, 
        w=1/combined_err[notmasked], cov="unscaled")
    xrads = np.linspace(-0.1, 0.7, 2, endpoint=True)
    yrads = coeff[1] + coeff[0] * xrads
    ax.plot(xrads, yrads, 'r-')
    print("Fit equation is y = {0:.2f} x + {1:.2f}".format(coeff[0], coeff[1]))
    print("Slope error: {0:.3f} Intercept error: {1:.3f}".format(
        np.sqrt(cov[1,1]), np.sqrt(cov[0,0])))

    ax.plot(xrads, xrads, 'k-')
    ax.set_xlabel("Asteroseismic R")
    ax.set_ylabel("Gaia R")
    ax.set_xlim(xrads[0], xrads[1])
    ax.set_ylim(xrads[0], xrads[1])


    ratio = (apokasc_full["RADIUS_DW"] - apokasc_full["Gaia R"])**2 / (
        dw_rad_err**2 + apokasc_full["Gaia R err"]**2)
    num_pairs = np.count_nonzero(notmasked)
    chisq = np.sum(ratio)
    chisq_dof = chisq / num_pairs
    print("Overlap sample size is {0:d}".format(num_pairs))
    print("Reduced chi-squared is {0:.2f}".format(chisq_dof))

@write_plot("Berger_comp")
def berger_MIST_radius_comparison():
    '''Compare the Berger radii to those in our study.'''
    full = cache.apogee_splitter_with_DSEP()
    cool_dwarfs = full.subsample(["Cool Dwarfs"])
    hot_dwarfs = full.subsample(["Hot Dwarfs"])
    subgiants = full.subsample(["Subgiants"])
    luminous_subgiants = full.subsample(["Luminous Subgiants"])
    nongiants = full.subsample([])

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    ax1.errorbar(
        cool_dwarfs["rad"], cool_dwarfs["Gaia R"], 
        yerr=cool_dwarfs["Gaia R err"], 
        xerr=[cool_dwarfs["rad_down"], cool_dwarfs["rad_up"]], marker=".",
        color=bc.black, ls="", label="No PB Correction")
    ax1.errorbar(
        hot_dwarfs["rad"], hot_dwarfs["Gaia R"], 
        yerr=hot_dwarfs["Gaia R err"], 
        xerr=[ hot_dwarfs["rad_down"], hot_dwarfs["rad_up"]], marker=".",
        color=bc.black, ls="", label="")
    ax1.errorbar(
        cool_dwarfs["rad"], cool_dwarfs["MIST R (APOGEE)"], 
        yerr=cool_dwarfs["MIST R Err (APOGEE)"], 
        xerr=[cool_dwarfs["rad_down"], cool_dwarfs["rad_up"]], marker=".",
        color='r', ls="", label="Deprojected")
    ax1.errorbar(
        hot_dwarfs["rad"], hot_dwarfs["MIST R (APOGEE)"], 
        yerr=hot_dwarfs["MIST R Err (APOGEE)"], 
        xerr=[ hot_dwarfs["rad_down"], hot_dwarfs["rad_up"]], marker=".",
        color='r', ls="", label="")
    ax1.plot([0.35, 2.5], [0.35, 2.5], 'c-')
    ax1.legend(loc="upper left")
    ax1.set_xlabel("Berger Radius (Rsun)")
    ax1.set_ylabel("This Work Radius (Rsun)")
    ax1.set_title("Main Sequence")

    ax2.errorbar(
        subgiants["rad"], subgiants["Gaia R"], 
        yerr=subgiants["Gaia R err"], 
        xerr=[subgiants["rad_down"], subgiants["rad_up"]], marker=".",
        color=bc.black, ls="", label="")
    ax2.errorbar(
        luminous_subgiants["rad"], luminous_subgiants["Gaia R"], 
        yerr=luminous_subgiants["Gaia R err"], 
        xerr=[luminous_subgiants["rad_down"], luminous_subgiants["rad_up"]], 
        marker=".", color=bc.black, ls="", label="")
    ax2.plot([1.3, 9], [1.3, 9], 'c-')
    ax2.set_xlabel("Berger Radius (Rsun)")
    ax2.set_ylabel("")
    ax2.set_title("Subgiants")

    # Calculate the lienar least-squares fit.
    p = np.polyfit(nongiants["rad"], nongiants["Gaia R"], 1)
    print("y = {0:.3f} x + {1:.2f}".format(p[0], p[1]))
    modelpoly = np.poly1d(p)
    modeledRs = modelpoly(nongiants["rad"])
    print("RMS Scatter: {0:.3f}".format(
        np.sqrt(np.mean((np.log10(nongiants["Gaia R"]) - np.log10(modeledRs))**2))))

@write_plot("apokasc_vdists")
def asteroseismic_velocity_comparison():
    '''Plot the agreement between predicted and actual vsini.'''
    astero = cache.astero_splitter()
    apokasc = astero.subsample([
        "Asteroseismic Dwarfs", "~Bad"])
    garcia = catin.read_Garcia_periods()
    astero_garcia = au.join_by_id(apokasc, garcia, "KEPLER_INT", "KIC")
    # of the No vsini targets, only 2M19261575+4935339 (KIC 11558593) has a v_eq 
    # which should turn up. It has a veq of 146 km/s. When I checked Rafa's new 
    # calculations, this star now has a much longer period, but still has a
    # peak at 1 day. 1 day is also the edge of their search grid. Light curve
    # indicates two periods.

    # So in this case, I think it would be fair to treat the No vsini targets
    # as slow rotators.


    # I want to make sure upper limits are actually detected as lower limits.
    astero_velocities = rot.period_to_velocities(
        astero_garcia["Prot"], astero_garcia["RADIUS_DW"])
    sini_cutoff = 0.0

    f, ax = plt.subplots(1, 1, figsize=figsize)
    rot.plot_vsini_dist(
        np.log10(astero_velocities), ax=ax, sini_cutoff=sini_cutoff)

    numspec = rot.calc_spec_rapid_num(
        np.log10(astero_velocities), sini_cutoff=sini_cutoff)
    print("Fraction of photometric rapid rotators: {0:d}/{1:d}".format(
        np.count_nonzero(astero_velocities > 10), len(astero_velocities)))
    print("Predicted spectroscopic rapid rotators: {0:.1f}/{1:d}".format(
        numspec, len(astero_velocities)))
    # Filling so that the evolved stars without vsini calculated by APOGEE are
    # counted as slow rotators.
    print("Actual spectroscopic rapid rotators: {0:d}/{1:d}".format(
        np.count_nonzero(astero_garcia["VSINI"].filled(0.0) > 10), 
        len(astero_garcia)))

#   rot.compare_vsini_distribution_full(
#       astero_garcia["Prot"], 
#       astero_garcia["e_Prot"] / astero_garcia["Prot"] / np.log(10),
#       astero_garcia["RADIUS_DW"], (
#           astero_garcia["RADIUS_DW_PERR"] - astero_garcia["RADIUS_DW_MERR"])
#       / 2 / astero_garcia["RADIUS_DW"] / np.log(10), astero_garcia["VSINI"], 
#       vsini_percent=10, vsini_cutoff=10)

def asteroseismic_target_sectors():
    '''Plot where in the Teff-MK diagram the asteroseismic targets fall.'''
    astero = cache.astero_splitter()
    apokasc = astero.subsample(["Asteroseismic Dwarfs"])
    full = cache.apogee_splitter_with_DSEP()
    fulltable = full.subsample([])
    apokasc_full = au.join_by_id(
        apokasc, fulltable, "KEPLER_INT", "kepid", 
        conflict_suffixes=("_APOGEE", "_APOKASC"))

    cool_dwarfs = full.subsample(["Cool Dwarfs"])
    hot_dwarfs = full.subsample(["Hot Dwarfs"])
    hot_subgiants = full.subsample(["Subgiants"])
    luminous_subgiants = full.subsample(["Luminous Subgiants"])
    giants = full.subsample(["Giants"])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        cool_dwarfs["TEFF"], cool_dwarfs["K Excess"], marker=".", 
        color=bc.violet, ls="", label="Cool Dwarfs", axis=ax)
    hr.absmag_teff_plot(
        hot_dwarfs["TEFF"], hot_dwarfs["K Excess"], marker=".", 
        color=bc.orange, ls="", label="Hot Dwarfs", axis=ax)
    hr.absmag_teff_plot(
        hot_subgiants["TEFF"], hot_subgiants["K Excess"], marker=".", 
        color=bc.algae, ls="", label="Hot Subgiants", axis=ax)
    hr.absmag_teff_plot(
        luminous_subgiants["TEFF"], luminous_subgiants["K Excess"], marker=".", 
        color=bc.sky_blue, ls="", label="Luminous Subgiants", axis=ax)
    hr.absmag_teff_plot(
        giants["TEFF"], giants["K Excess"], marker=".", 
        color=bc.red, ls="", label="Giants", axis=ax)

    hr.absmag_teff_plot(
        apokasc_full["TEFF"], apokasc_full["K Excess"], marker="*",
        color=bc.black, ls="", label="Asteroseismic", axis=ax, ms=8)


    ax.set_xlabel("APOGEE Teff")
    ax.set_ylabel("K Excess")
    ax.set_xlim(6500, 3500)
    ax.set_ylim(1.0, -5.0) 
    ax.legend(loc="upper right")

###############
# Cool Dwarfs #
###############

def cool_dwarf_targets():
    '''Highlight the Cool Dwarfs on an HR Diagram.'''
    full = cache.full_apogee_splitter()
    split.initialize_clean_APOGEE(full)
    full.split_teff(
        "TEFF", [5250], (
            "APOGEE Evolution Cool", "APOGEE Evolution Hot", "No APOGEE Evolution Teff"), 
        null_value=np.ma.masked, teff_crit="APOGEE Evolution Region")
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_apo = aposplit.subsample(["Dwarfs", "APOGEE Evolution Cool", "Mcq"])
    fullsamp = full.subsample([
        "~No APOGEE Evolution Teff", "K Detection", "In Gaia"])

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    hr.absmag_teff_plot(
        fullsamp["TEFF"], fullsamp["M_K"], color="k", marker=".", alpha=0.5,
        axis=ax, label="", ls="")
    hr.absmag_teff_plot(
        cool_apo["TEFF"], cool_apo["M_K"], color="r", marker="o", axis=ax,
        label="Cool dwarfs", ls="")
    ax.set_xlim(6750, 3600)
    ax.set_ylim(6.5, 0.15)
    ax.set_xlabel("APOGEE Teff (K)")
    ax.set_ylabel("MK")
    ax.legend(loc="lower left")

def cool_dwarf_photometric_binaries():
    '''Count the number of photometric binaries in cool dwarf regime.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_singles = aposplit.subsample(["Cool Dwarfs", "Photometric Singles"])
    cool_binaries = aposplit.subsample(["Cool Dwarfs", "Photometric Binaries"])

    f, ax = plt.subplots(1, 1, figsize=figsize)

    hr.absmag_teff_plot(
        cool_singles["TEFF"], cool_singles["M_K"], color=bc.black, marker=".",
        ls="", axis=ax)
    hr.absmag_teff_plot(
        cool_binaries["TEFF"], cool_binaries["M_K"], color="red", marker=".",
        ls="", axis=ax)
    ax.set_xlabel("TEFF")
    ax.set_ylabel("M_K")

    num_singles = aposplit.subsample_len(["Cool Dwarfs", "Photometric Singles"])
    num_binaries = aposplit.subsample_len(["Cool Dwarfs", "Photometric Binaries"])
    frac_binaries = num_singles / (num_singles + num_binaries)

    print("APOGEE Photometric Binary Fraction: {0}".format(
        format_fraction(num_binaries, num_singles+num_binaries)))

def cool_vsini_noperiod():
    '''Count number of cool dwarfs without rotation periods with high vsini.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_apo = aposplit.subsample(["Dwarfs", "APOGEE Evolution Cool", "No Mcq"])

def cool_vsini_veq_agreement_MIST():
    '''Plot the vsini and veq in a single plot with direct MIST R.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_apo = aposplit.subsample(["Dwarfs", "APOGEE Evolution Cool", "Mcq", "~DLSB"])
    dlsb = aposplit.subsample(["Dwarfs", "APOGEE Evolution Cool", "Mcq", "DLSB"])
    mcq = catin.read_McQuillan_catalog()
    cool_apo_mcq = au.join_by_id(cool_apo, mcq, "kepid", "KIC")
    dlsb_mcq = au.join_by_id(dlsb, mcq, "kepid", "KIC")
    f, ax = plt.subplots(1, 1, figsize=(12,12))
    rot.plot_vsini_velocity(
        cool_apo_mcq["VSINI"], cool_apo_mcq["Prot"], cool_apo_mcq["e_Prot"],
        cool_apo_mcq["MIST R (APOGEE)"], cool_apo_mcq["MIST R Err (APOGEE)"],
        ax=ax, label="Rapid Rotators")
    rot.plot_vsini_velocity(
        dlsb_mcq["VSINI"], dlsb_mcq["Prot"], dlsb_mcq["e_Prot"],
        dlsb_mcq["MIST R (APOGEE)"], dlsb_mcq["MIST R Err (APOGEE)"], ax=ax,
        color="r", label="SB2")
    ax.plot([1, 100], [1.15, 115], color='k', ls="-.", marker="")
    ax.set_title("Direct MIST Radius")
    ax.legend(loc="lower right")

def cool_vsini_veq_agreement_Baraffe():
    '''Plot the vsini and veq in a single plot with direct Baraffe R.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_apo = aposplit.subsample(["Dwarfs", "APOGEE Evolution Cool", "Mcq", "~DLSB"])
    dlsb = aposplit.subsample(["Dwarfs", "APOGEE Evolution Cool", "Mcq", "DLSB"])
    mcq = catin.read_McQuillan_catalog()
    cool_apo_mcq = au.join_by_id(cool_apo, mcq, "kepid", "KIC")
    dlsb_mcq = au.join_by_id(dlsb, mcq, "kepid", "KIC")
    f, ax = plt.subplots(1, 1, figsize=(12,12))

    iso = baraffe.BaraffeIsochrone.isochrone_from_file()
    baraffe_rad = iso.interpolate_isochrone_cols(
        1.0, cool_apo_mcq["TEFF"], iso.teff_col, iso.radius_col)
    baraffe_rad_err = cool_apo_mcq["TEFF_ERR"] * iso.isochrone_derivative(
        1.0, cool_apo_mcq["TEFF"], iso.teff_col, iso.radius_col)
    dlsb_rad = iso.interpolate_isochrone_cols(
        1.0, dlsb_mcq["TEFF"], iso.teff_col, iso.radius_col)
    dlsb_rad_err = dlsb_mcq["TEFF_ERR"] * iso.isochrone_derivative(
        1.0, dlsb_mcq["TEFF"], iso.teff_col, iso.radius_col)
    rot.plot_vsini_velocity(
        cool_apo_mcq["VSINI"], cool_apo_mcq["Prot"], cool_apo_mcq["e_Prot"],
        baraffe_rad,baraffe_rad_err, ax=ax, label="Rapid Rotator")
    rot.plot_vsini_velocity(
        dlsb_mcq["VSINI"], dlsb_mcq["Prot"], dlsb_mcq["e_Prot"],
        dlsb_rad,dlsb_rad_err, ax=ax, label="SB2", color="r")
    ax.plot([1, 100], [1.15, 115], color='k', ls="-.", marker="")
    ax.set_title("Direct Baraffe Radius")
    ax.legend(loc="lower right")

def cool_vsini_veq_agreement_Baraffe_Lbol():
    '''Plot the vsini and veq in a single plot using Baraffe BC.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_apo = aposplit.subsample(["Dwarfs", "APOGEE Evolution Cool", "Mcq", "~DLSB"])
    dlsb = aposplit.subsample(["Dwarfs", "APOGEE Evolution Cool", "Mcq", "DLSB"])
    mcq = catin.read_McQuillan_catalog()
    cool_apo_mcq = au.join_by_id(cool_apo, mcq, "kepid", "KIC")
    dlsb_mcq = au.join_by_id(dlsb, mcq, "kepid", "KIC")

    # STILL UNDER CONSTRUCTION!
    f, ax = plt.subplots(1, 1, figsize=(12,12))
    iso = baraffe.BaraffeIsochrone.isochrone_from_file()
    log_baraffe_lum = iso.interpolate_isochrone_cols(
        0.12, masses, iso.mass_col, iso.logL_col)
    apogee_rad = 10**(
        0.5*(log_baraffe_lum - 4*(np.log10(pleiades_vsini["TEFF"]) - np.log10(5777))))
    apogee_lograd_err = 2*pleiades_vsini["TEFF_ERR"] / pleiades_vsini["TEFF"] / np.log(10)
    apogee_rad_err = apogee_lograd_err * apogee_rad * np.log(10)

@write_plot("f8a")
def cool_vsini_veq_agreement():
    '''Plot the vsini and veq in a single plot with bolometric R.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_apo = aposplit.subsample(["Cool Singles", "Mcq", "~DLSB"])
    phot_bins = aposplit.subsample(["Photometric Binaries", "Mcq", "~DLSB"])
    # I know cool dwarfs have no spectroscopic binaries
    dlsb = aposplit.subsample(["Photometric Binaries", "Mcq", "DLSB"])
    mcq = catin.read_McQuillan_catalog()
    cool_apo_mcq = au.join_by_id(cool_apo, mcq, "kepid", "KIC")
    phot_bin_mcq = au.join_by_id(phot_bins, mcq, "kepid", "KIC")
    dlsb_mcq = au.join_by_id(dlsb, mcq, "kepid", "KIC")

    phot_bins = cool_apo_mcq["K Excess"] < -0.3

    f, ax = plt.subplots(1, 1, figsize=figsize)
    rot.plot_vsini_velocity(
        cool_apo_mcq["VSINI"], cool_apo_mcq["Prot"], 
        cool_apo_mcq["e_Prot"], cool_apo_mcq["MIST R (APOGEE)"], 
        cool_apo_mcq["MIST R Err (APOGEE)"], ax=ax, 
        label="Photometric Single", color=bc.black, marker="o")
    rot.plot_vsini_velocity(
        phot_bin_mcq["VSINI"], phot_bin_mcq["Prot"], 
        phot_bin_mcq["e_Prot"], phot_bin_mcq["MIST R (APOGEE)"], 
        phot_bin_mcq["MIST R Err (APOGEE)"], ax=ax, color=bc.green,
        label="Photometric Binary", marker="o")
    rot.plot_vsini_velocity(
        dlsb_mcq["VSINI"], dlsb_mcq["Prot"], dlsb_mcq["e_Prot"],
        dlsb_mcq["MIST R (APOGEE)"], dlsb_mcq["MIST R Err (APOGEE)"], ax=ax,
        color=bc.sky_blue, marker="*", label="SB2", ms=10)
    ax.plot([1, 100], [1.13, 113], color='k', ls="-.", marker="")
    ax.set_title("APOGEE Temperature")
    ax.legend(loc="lower right")

@write_plot("f8b")
def cool_vsini_veq_agreement_elBadry():
    '''Plot the vsini and veq in a single plot with R from El-Badry.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    elbadry_binary = vstack([
        aposplit.subsample([
            "Cool Singles", "Mcq", "El-Badry SB2", "~DLSB"]),
        aposplit.subsample([
            "Cool Singles", "Mcq", "El-Badry Hidden Triple", "~DLSB"]),
        aposplit.subsample([
            "Cool Singles", "Mcq", "El-Badry SB3", "~DLSB"]),
        aposplit.subsample([
            "Photometric Binaries", "Mcq", "El-Badry SB2", "~DLSB"]),
        aposplit.subsample([
            "Photometric Binaries", "Mcq", "El-Badry Hidden Triple", "~DLSB"]),
        aposplit.subsample([
            "Photometric Binaries", "Mcq", "El-Badry SB3", "~DLSB"])])
    elbadry_single = vstack([
        aposplit.subsample([
            "Cool Singles", "Mcq", "El-Badry Single", "~DLSB"]),
        aposplit.subsample([
            "Cool Singles", "Mcq", "El-Badry SB1", "~DLSB"]),
        aposplit.subsample([
            "Photometric Binaries", "Mcq", "El-Badry Single", "~DLSB"]),
        aposplit.subsample([
            "Photometric Binaries", "Mcq", "El-Badry SB1", "~DLSB"])])
    sb2s = vstack([
        aposplit.subsample([
            "Cool Singles", "Mcq", "~No El-Badry Binarity", "DLSB"]),
        aposplit.subsample([
            "Photometric Binaries", "Mcq", "~No El-Badry Binarity", "DLSB"])])

    mcq = catin.read_McQuillan_catalog()
    binary_mcq = au.join_by_id(elbadry_binary, mcq, "kepid", "KIC")
    single_mcq = au.join_by_id(elbadry_single, mcq, "kepid", "KIC")
    sb2_mcq = au.join_by_id(sb2s, mcq, "kepid", "KIC")

    f, ax = plt.subplots(1, 1, figsize=figsize)
    rot.plot_vsini_velocity(
        single_mcq["VSINI"], single_mcq["Prot"], 
        single_mcq["e_Prot"], single_mcq["MIST R (El-Badry)"], 
        single_mcq["MIST R Err (APOGEE)"], ax=ax, 
        label="El Badry Single", color=bc.black, marker="o")
    rot.plot_vsini_velocity(
        binary_mcq["VSINI"], binary_mcq["Prot"], 
        binary_mcq["e_Prot"], binary_mcq["MIST R (El-Badry)"], 
        binary_mcq["MIST R Err (APOGEE)"], ax=ax, color=bc.green,
        label="El-Badry Binary", marker="o")
    rot.plot_vsini_velocity(
        sb2_mcq["VSINI"], sb2_mcq["Prot"], 
        sb2_mcq["e_Prot"], sb2_mcq["MIST R (El-Badry)"], 
        sb2_mcq["MIST R Err (APOGEE)"], ax=ax, 
        label="SB2", color=bc.sky_blue, marker="*")
    ax.plot([1, 100], [1.15, 115], color='k', ls="-.", marker="")
    ax.legend(loc="lower right")
    ax.set_title("El-Badry et al (2018b) Temperature")

@write_plot("f9")
def elBadry_radius_bias_cooldwarf():
    '''Calculate the radius bias from the El-Badry corrected teffs.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    elbadry_binary = vstack([
        aposplit.subsample([
            "Cool Singles", "Mcq", "El-Badry SB2"]),
        aposplit.subsample([
            "Cool Singles", "Mcq", "El-Badry Hidden Triple"]),
        aposplit.subsample([
            "Cool Singles", "Mcq", "El-Badry SB3"]),
        aposplit.subsample([
            "Photometric Binaries", "Mcq", "El-Badry SB2"]),
        aposplit.subsample([
            "Photometric Binaries", "Mcq", "El-Badry Hidden Triple"]),
        aposplit.subsample([
            "Photometric Binaries", "Mcq", "El-Badry SB3"])])

    raw_radius = elbadry_binary["MIST R (APOGEE)"]
    corrected_radius = elbadry_binary["MIST R (El-Badry)"]

    radius_bias = (raw_radius - corrected_radius) / corrected_radius * 100

    f, ax = plt.subplots(1, 1, figsize=figsize)
    n, bins, patches = ax.hist(
        radius_bias, bins=18, range=(-17, 1), color=bc.blue)
    max_num = max(n)

    quartiles = np.percentile(radius_bias,  [33, 50, 67])
    ax.plot([quartiles[0], quartiles[0]], [0, max_num], color=bc.black, lw=3,
            ls="--", marker="")
    ax.plot([quartiles[1], quartiles[1]], [0, max_num], color=bc.black, lw=5,
            ls="-", marker="")
    ax.plot([quartiles[2], quartiles[2]], [0, max_num], color=bc.black, lw=3,
            ls="--", marker="")
    print(quartiles)

    ax.set_ylim(0, max_num)
    ax.set_xlabel("Radius Bias (%)")
    ax.set_ylabel("N")

def cool_dwarf_outliers():
    '''Print the number of high chi-squared outliers for cool dwarfs.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_apo = aposplit.subsample([
        "Cool Singles", "Mcq", "~DLSB", "Vsini det"])
    phot_bins = aposplit.subsample([
        "Photometric Binaries", "Mcq", "~DLSB", "Vsini det"])
    cool_apo_elb = aposplit.subsample([
        "Cool Singles", "Mcq", "~DLSB", "Vsini det", "~No El-Badry Binarity"])
    phot_bins_elb = aposplit.subsample([
        "Photometric Binaries", "Mcq", "~DLSB", "Vsini det", 
        "~No El-Badry Binarity"])
    samp = vstack([cool_apo, phot_bins])
    samp_elb = vstack([cool_apo_elb, phot_bins_elb])

    mcq = catin.read_McQuillan_catalog()
    samp_mcq = au.join_by_id(samp, mcq, "kepid", "KIC")
    samp_mcq_elb = au.join_by_id(samp_elb, mcq, "kepid", "KIC")

    lowvel, vel, highvel = rot.period_to_velocities_uncertainties(
        samp_mcq["Prot"], samp_mcq["MIST R (APOGEE)"], 
        samp_mcq["MIST R Err (APOGEE)"], samp_mcq["MIST R Err (APOGEE)"])

    lowvel_elb, vel_elb, highvel_elb = rot.period_to_velocities_uncertainties(
        samp_mcq_elb["Prot"], samp_mcq_elb["MIST R (El-Badry)"], 
        samp_mcq_elb["MIST R Err (APOGEE)"], samp_mcq_elb["MIST R Err (APOGEE)"])

    chi_sq = (
        np.log10(samp_mcq["VSINI"]) - np.where(
            vel > 10, np.log10(vel), np.log10(vel)))**2 / (
                (0.12 / np.log(10))**2 + (lowvel / vel / np.log(10))**2)
    chi_sq_inf = (
        np.log10(samp_mcq["VSINI"]) - np.where(
            vel > 10, np.log10(vel), np.log10(vel)) - np.log(1.13))**2 / (
                (0.12 / np.log(10))**2 + (lowvel / vel / np.log(10))**2)
    chi_sq_elb = (
        np.log10(samp_mcq_elb["VSINI"]) - np.where(
            vel_elb > 10, np.log10(vel_elb), np.log10(vel_elb)) - np.log(1.1))**2 / ( 
                (0.12 / np.log(10))**2 + (lowvel_elb / vel_elb / np.log(10))**2)
    ok_points = samp_mcq["VSINI"] < vel
    ok_points_inf = samp_mcq["VSINI"] < 1.1*vel
    ok_points_elb = samp_mcq_elb["VSINI"] < vel_elb
    chi_sq[ok_points] = 0
    chi_sq_inf[ok_points_inf] = 0
    chi_sq_elb[ok_points_elb] = 0

    total = (
        aposplit.subsample_len([
            "Cool Singles", "~Unknown Mcq"]) +
        aposplit.subsample_len([
            "Photometric Binaries", "~Unknown Mcq"]))
    total_elb = (
        aposplit.subsample_len([
            "Cool Singles", "~Unknown Mcq", "~No El-Badry Binarity"]) +
        aposplit.subsample_len([
            "Photometric Binaries", "~Unknown Mcq", "~No El-Badry Binarity"]))
    outliers = chi_sq > 3**2
    offsets = np.count_nonzero(outliers)
    outliers_elb = chi_sq_elb > 3**2
    offsets_elb = np.count_nonzero(outliers_elb)
    outliers_inf = chi_sq_inf > 3**2
    offsets_inf = np.count_nonzero(outliers_inf)

    print("{0:d}/{1:d} are outliers".format(
        offsets, total))
    print("{0:d}/{1:d} are outliers with inflation".format(
        offsets_inf, total))
    print("{0:d}/{1:d} are outliers after El-Badry".format(
        offsets_elb, total_elb))

    f, ax1, = plt.subplots(1, 1, figsize=(9, 9))
#   f, ax2, = plt.subplots(1, 1, figsize=(9, 9))
    f, ax3, = plt.subplots(1, 1, figsize=(9, 9))
    ax1.errorbar(
        np.log10(vel), 
        np.log10(samp_mcq["VSINI"]), yerr=0.12/np.log(10), 
        xerr=lowvel/vel/np.log(10), marker=".", 
        color= bc.black, ls="")
    ax1.errorbar(
        np.log10(vel[outliers]), 
        np.log10(samp_mcq["VSINI"][outliers]), yerr=0.12/ np.log(10), 
        xerr=lowvel[outliers]/vel[outliers]/np.log(10), marker=".", 
        color=bc.pink, ls="", ms=10)
#   ax2.errorbar(
#       np.log10(vel_elb), 
#       np.log10(samp_mcq_elb["VSINI"]), yerr=0.12/np.log(10), 
#       xerr=lowvel_elb/vel_elb/np.log(10), marker=".", 
#       color= bc.black, ls="")
#   ax2.errorbar(
#       np.log10(vel_elb[outliers_elb]),
#       np.log10(samp_mcq_elb["VSINI"][outliers_elb]), yerr=0.12/ np.log(10), 
#       xerr=lowvel_elb[outliers_elb]/vel_elb[outliers_elb]/np.log(10), marker=".", 
#       color=bc.pink, ls="", ms=10)
    ax3.errorbar(
        np.log10(vel),
        np.log10(samp_mcq["VSINI"]), yerr=0.12/np.log(10),
        xerr=lowvel/vel/np.log(10), marker=".", color=bc.black, ls="")
    ax3.errorbar(
        np.log10(vel[outliers_inf]),
        np.log10(samp_mcq["VSINI"][outliers_inf]), yerr=0.12/np.log(10),
        xerr=lowvel[outliers_inf]/vel[outliers_inf]/np.log(10), marker=".", 
        color=bc.pink, ls="")
    ax1.plot([0, 2], [0, 2], 'k--', lw=4)
#   ax2.plot([0, 2], [0, 2], 'k--', lw=4)
    ax3.plot([0, 2], [0, 2], 'k--', lw=4)
    ax3.plot([0, 2], [0.13, 2.13], 'k--', lw=2)
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 2)
#   ax2.set_xlim(0, 2)
#   ax2.set_ylim(0, 2)
    ax3.set_xlim(0, 2)
    ax3.set_ylim(0, 2)


@write_plot("cool_vdists")
def cool_dwarf_velocity_comparison():
    '''Plot the agreement between predicted and actual vsini.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_apo = aposplit.subsample(["Cool Dwarfs", "Mcq", "~DLSB"])
    mcq = catin.read_McQuillan_catalog()
    cool_apo_mcq = au.join_by_id(cool_apo, mcq, "kepid", "KIC")

    # I want to make sure upper limits are actually detected as lower limits.
    cool_velocities = rot.period_to_velocities(
        cool_apo_mcq["Prot"], cool_apo_mcq["MIST R (APOGEE)"])
    sini_cutoff = 0.5

    f, ax = plt.subplots(1, 1, figsize=figsize)
    rot.plot_vsini_dist(
        np.log10(cool_velocities), ax=ax, sini_cutoff=sini_cutoff)

    numspec = rot.calc_spec_rapid_num(
        np.log10(cool_velocities), sini_cutoff=sini_cutoff)
    print("Fraction of photometric rapid rotators: {0:d}/{1:d}".format(
        np.count_nonzero(cool_velocities > 10), len(cool_velocities)))
    print("Predicted spectroscopic rapid rotators: {0:.1f}/{1:d}".format(
        numspec, len(cool_velocities)))
    print("Actual spectroscopic rapid rotators: {0:d}/{1:d}".format(
        np.count_nonzero(cool_apo_mcq["VSINI"] > 10), len(cool_apo_mcq)))

def teff_bias_radius_impact():
    '''Calculate the impact of teff on the radius of a star.'''
    orig_teff = 4000
    teff_offset = np.array([400])

    orig_k = samp.calc_model_mag_fixed_age_feh_alpha(
        [orig_teff], 0.0, "Ks", age=1e9)
    offset_k = samp.calc_model_mag_fixed_age_feh_alpha(
        orig_teff - teff_offset, 0.0, "Ks", age=1e9)

    orig_bc = samp.calc_model_fixed_age_feh_alpha(
        np.log10([orig_teff]), mist.MISTIsochrone.logteff_col,
        "BC K", 0.0, 1e9)
    offset_bc = samp.calc_model_fixed_age_feh_alpha(
        np.log10(orig_teff - teff_offset), mist.MISTIsochrone.logteff_col,
        "BC K", 0.0, 1e9)

    k_term = -0.4 * (offset_k - orig_k)
    bc_term = -0.4 * (offset_bc - orig_bc)
    teff_term = -4 * (np.log10(1-teff_offset / orig_teff))
    r_term = k_term + bc_term + teff_term
    print("K Term: {0:.2f}".format(k_term[0]))
    print("BC Term: {0:.2f}".format(bc_term[0]))
    print("Teff Term: {0:.2f}".format(teff_term[0]))
    print("R Term: {0:.2f}".format(r_term[0]))

def cool_vsini_binarity():
    '''Plot the rapid rotators on an HR diagram.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_apo_rapid_mcq = aposplit.subsample([
        "Dwarfs", "APOGEE Evolution Cool", "~DLSB", "Vsini det", "Mcq"])
    cool_apo_rapid_nomcq = aposplit.subsample([
        "Dwarfs", "APOGEE Evolution Cool", "~DLSB", "Vsini det", "No Mcq"])
    cool_apo_slow = aposplit.subsample([
        "Dwarfs", "APOGEE Evolution Cool", "Vsini nondet"])

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    hr.absmag_teff_plot(cool_apo_slow["TEFF"], cool_apo_slow["K Excess"],
                        color="k", marker=".", ls="", label="Slow")
    hr.absmag_teff_plot(cool_apo_rapid_mcq["TEFF"], cool_apo_rapid_mcq["K Excess"],
                        color="r", marker="o", ls="", label="High Vsini (Mcq)")
    hr.absmag_teff_plot(cool_apo_rapid_nomcq["TEFF"], cool_apo_rapid_nomcq["K Excess"],
                        color=bc.algae, marker="o", ls="", 
                        label="High Vsini (No Mcq)")
    ax.set_xlabel("APOGEE Teff (K)")
    ax.set_ylabel("K Excess (mag)")
    ax.legend(loc="upper right")

def vsini_logg_degeneracy():
    '''Plot a quiver diagram showing how changing the logg will affect
    vsini.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_apo = aposplit.subsample(["Dwarfs", "APOGEE Evolution Cool", "Mcq", "~DLSB"])

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    covs = np.zeros(len(cool_apo))
    loggvars = np.zeros(len(cool_apo))
    vsinivars = np.zeros(len(cool_apo))
    for i in range(len(cool_apo)):
        covs[i] = cool_apo["FPARAM_COV"][i][1, 7]
        loggvars[i] = cool_apo["FPARAM_COV"][i][1, 1]
        vsinivars[i] = cool_apo["FPARAM_COV"][i][7, 7]

    varx = loggvars
    vary = vsinivars
    print("X Variance: {0:.2f}".format(np.median(varx)))
    print("Y Variance: {0:.2f}".format(np.median(vary)))
    majorax = np.sqrt((varx + vary) / 2 + np.sqrt((varx - vary)**2/4 + covs**2))
    minorax = np.sqrt((varx + vary) / 2 - np.sqrt((varx - vary)**2/4 + covs**2))
    focus = np.sqrt(majorax**2 - minorax**2)
    angle = np.arctan(2 * covs / (varx - vary))/2 + np.pi

    quiverx = cool_apo["LOGG_FIT"]
    quivery = cool_apo["VSINI"]
    quiverscale = 1
    correctedangle = np.where(varx > vary, angle, angle-np.pi/2)
    quiveru = focus * np.cos(correctedangle)
    quiverv = focus * np.sin(correctedangle)

    ax.quiver(quiverx, quivery, quiveru, quiverv)
    ax.set_xlabel("log(g)")
    ax.set_ylabel("VSINI")

@write_plot("photometric_vsini", toplevel=PLOT_PATH)
def vsini_agreement_photometric_temperatures():
    '''Compare the relation using both spectroscopic and photometric temps.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_apo = aposplit.subsample(["Dwarfs", "APOGEE Evolution Cool", "Mcq", "~DLSB"])
    cool_kspc = aposplit.subsample(["Dwarfs", "KSPC Evolution Cool", "Mcq", "~DLSB"])
    cool_pin = aposplit.subsample(["Dwarfs", "Pinsonneault Evolution Cool", "Mcq", "~DLSB"])
    cool_elb = aposplit.subsample(["Dwarfs", "El-Badry Evolution Cool", "Mcq", "~DLSB"])
    mcq = catin.read_McQuillan_catalog()
    cool_apo_mcq = au.join_by_id(cool_apo, mcq, "kepid", "KIC")
    cool_kspc_mcq = au.join_by_id(cool_kspc, mcq, "kepid", "KIC")
    cool_pin_mcq = au.join_by_id(cool_pin, mcq, "kepid", "KIC")
    cool_elb_mcq = au.join_by_id(cool_elb, mcq, "kepid", "KIC")

#   f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(12*5, 12))
    f, ax1 = plt.subplots(1, 1, figsize=(12, 12))
    f, ax2 = plt.subplots(1, 1, figsize=(12, 12))
    f, ax3 = plt.subplots(1, 1, figsize=(12, 12))
    f, ax4 = plt.subplots(1, 1, figsize=(12, 12))
    f, ax5 = plt.subplots(1, 1, figsize=(12, 12))
    rot.plot_vsini_velocity(
        cool_apo_mcq["VSINI"], cool_apo_mcq["Prot"], 
        cool_apo_mcq["MIST R (APOGEE)"], 0, 0, ax=ax1)
    rot.plot_vsini_velocity(
        cool_pin_mcq["VSINI"], cool_pin_mcq["Prot"], 
        cool_pin_mcq["MIST R (Pinsonneault)"], 0, 0, 
        ax=ax2)
    rot.plot_vsini_velocity(
        cool_kspc_mcq["VSINI"], cool_kspc_mcq["Prot"], 
        cool_kspc_mcq["MIST R (KSPC)"], 0, 0, ax=ax3)
    rot.plot_vsini_velocity(
        cool_elb_mcq["VSINI"], cool_elb_mcq["Prot"], 
        cool_elb_mcq["MIST R (El-Badry)"], 0, 0, ax=ax4)
    rot.plot_vsini_velocity(
        cool_apo_mcq["VSINI"], cool_apo_mcq["Prot"], 
        cool_apo_mcq["Gaia R"], 
        -cool_apo_mcq["Gaia R"] * (1-2**-0.5), 0, ax=ax5)
    ax1.set_title("APOGEE Temperature")
    ax2.set_title("Pinsonneault Temperature")
    ax3.set_title("Huber Temperature")
    ax4.set_title("El-Badry Temperature")
    ax5.set_title("Gaia Radius")
    ax1.set_xlim(0, 70)
    ax2.set_xlim(0, 70)
    ax3.set_xlim(0, 70)
    ax4.set_xlim(0, 70)
    ax5.set_xlim(0, 70)

def ElBadry_SB1_Vmacro_check():
    '''Compare vmacro from El Badry to the vmacro for SB1s.

    The two should have fairly agreeable values.''' 
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_dwarfs = aposplit.subsample(
        ["Dwarfs", "APOGEE Evolution Cool", "~DLSB"])
    sb1s = catin.read_El_Badry_SB1()
    joined = catalog.join_by_2MASS_key(
        cool_dwarfs, sb1s, "APOGEE_ID", "APOGEE_ID")

    plt.plot(joined["VSINI"], joined["v_macro [km/s]"], 'k.')

def Kepler_RMIST_Rbol_comparison():
    '''Compare the radii from MIST and from bolometric luminosities.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_dwarfs = aposplit.subsample(["Cool Dwarfs", "Photometric Singles"])
    photbins = aposplit.subsample(["Cool Dwarfs", "Photometric Binaries"])

    f, ax = plt.subplots(1, 1, figsize=figsize)

    mist_err_med = np.median(cool_dwarfs["MIST R Err (APOGEE)"])
    gaia_err_med = np.median(cool_dwarfs["Gaia R err"])

    ax.errorbar(
        cool_dwarfs["MIST R (APOGEE)"], cool_dwarfs["Gaia R"],
        color='k', marker='o', ls="", label="Cool Dwarfs")
    ax.errorbar(
        photbins["MIST R (APOGEE)"], photbins["Gaia R"], color='r', marker='.', 
        ls="", label="Photometric Binaries")
    ax.errorbar(
        [0.9], [0.4], yerr=[gaia_err_med], xerr=[mist_err_med], color='k',
        marker='.', ls="", label="")
    ax.plot([0.3, 1.0], [0.3, 1.0], 'k-')
    ax.set_xlabel("MIST R")
    ax.set_ylabel("Gaia R")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="upper left")

##############
# Hot Dwarfs #
##############

@write_plot("Hot_Dwarf_vsini_veq_comparison")
def hot_vsini_veq_agreement_Lbol():
    '''Plot the vsini and veq in a single plot with bolometric R.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    dlsbs = aposplit.subsample(["Hot Dwarfs", "Mcq", "DLSB"])
    phot_singles = aposplit.subsample([
        "Hot Dwarfs", "Mcq", "~DLSB", "Photometric Singles"])
    # Both Singles and SB1s count as singles.
    elBadry_singles = aposplit.subsample([
        "Hot Dwarfs", "Mcq", "~DLSB", "~Photometric Singles", 
        "El-Badry Single"])
    elBadry_SB1s = aposplit.subsample([
        "Hot Dwarfs", "Mcq", "~DLSB", "~Photometric Singles", 
        "El-Badry SB1"])
    elBadry_single_group = vstack([elBadry_singles, elBadry_SB1s])
    # El-Badry multiples are in SB2s, Hidden Triples, and SB3s.
    elBadry_SB2s = aposplit.subsample([
        "Hot Dwarfs", "Mcq", "~DLSB", "~Photometric Singles", 
        "El-Badry SB2"])
    elBadry_hts = aposplit.subsample([
        "Hot Dwarfs", "Mcq", "~DLSB", "~Photometric Singles", 
        "El-Badry Hidden Triple"])
    elBadry_SB3s = aposplit.subsample([
        "Hot Dwarfs", "Mcq", "~DLSB", "~Photometric Singles", 
        "El-Badry SB3"])
    elBadry_multiples = vstack([elBadry_SB2s, elBadry_hts, elBadry_SB3s])
    no_elBadry = aposplit.subsample([
        "Hot Dwarfs", "Mcq", "~DLSB", "~Photometric Singles", 
        "No El-Badry Binarity"])
    mcq = catin.read_McQuillan_catalog()
    dlsb_mcq = au.join_by_id(dlsbs, mcq, "kepid", "KIC")
    phot_single_mcq = au.join_by_id(phot_singles, mcq, "kepid", "KIC")
    elbadry_single_mcq = au.join_by_id(
        elBadry_single_group, mcq, "kepid", "KIC")
    elbadry_multiple_mcq = au.join_by_id(elBadry_multiples, mcq, "kepid", "KIC")
    no_elbadry_mcq = au.join_by_id(no_elBadry, mcq, "kepid", "KIC")

    f, ax = plt.subplots(1, 1, figsize=figsize)
    rot.plot_vsini_velocity(
        dlsb_mcq["VSINI"], dlsb_mcq["Prot"], dlsb_mcq["e_Prot"],
        dlsb_mcq["Gaia R"], dlsb_mcq["Gaia R err"], ax=ax, color=bc.sky_blue,
        marker="*", label="SB2", ms=10)
    rot.plot_vsini_velocity(
        phot_single_mcq["VSINI"], phot_single_mcq["Prot"], 
        phot_single_mcq["e_Prot"], phot_single_mcq["Gaia R"], 
        phot_single_mcq["Gaia R err"], ax=ax, color=bc.black, 
        label="Photometric Single", marker="o")
    rot.plot_vsini_velocity(
        elbadry_single_mcq["VSINI"], elbadry_single_mcq["Prot"], 
        elbadry_single_mcq["e_Prot"], elbadry_single_mcq["Gaia R"], 
        elbadry_single_mcq["Gaia R err"], ax=ax, color=bc.black, 
        label="El-Badry Single", marker="s")
    rot.plot_vsini_velocity(
        elbadry_multiple_mcq["VSINI"], elbadry_multiple_mcq["Prot"], 
        elbadry_multiple_mcq["e_Prot"], elbadry_multiple_mcq["Gaia R"], 
        elbadry_multiple_mcq["Gaia R err"], ax=ax, color=bc.sky_blue, 
        label="El-Badry Multiple", marker="s")
    rot.plot_vsini_velocity(
        no_elbadry_mcq["VSINI"], no_elbadry_mcq["Prot"], 
        no_elbadry_mcq["e_Prot"], no_elbadry_mcq["Gaia R"], 
        no_elbadry_mcq["Gaia R err"], ax=ax, label="No El-Badry Information",
        color=bc.pink, marker="s")
    ax.plot([1, 100], [1.15, 115], color=bc.black, ls="-.")
    ax.set_title("Hot Dwarfs")
    ax.legend(loc="lower right")

@write_plot("hot_vdists")
def hot_dwarf_velocity_comparison():
    '''Plot the agreement between predicted and actual vsini.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    hot_apo = aposplit.subsample(["Hot Dwarfs", "Mcq"])
    mcq = catin.read_McQuillan_catalog()
    hot_apo_mcq = au.join_by_id(hot_apo, mcq, "kepid", "KIC")

    # I want to make sure upper limits are actually detected as lower limits.
    hot_velocities = rot.period_to_velocities(
        hot_apo_mcq["Prot"], hot_apo_mcq["Gaia R"])
    sini_cutoff = 0.5

    f, ax = plt.subplots(1, 1, figsize=figsize)
    rot.plot_vsini_dist(
        np.log10(hot_velocities), ax=ax, sini_cutoff=sini_cutoff)
    
    phot_singles = hot_apo_mcq["K Excess"] >= -0.3

    print("Photometric Singles:")
    numspec = rot.calc_spec_rapid_num(
        np.log10(hot_velocities[phot_singles]), sini_cutoff=sini_cutoff)
    totalnum = aposplit.subsample_len(
        ["Hot Dwarfs", "Mcq", "Photometric Singles"])
    print("Fraction of photometric rapid rotators: {0:d}/{1:d}".format(
        np.count_nonzero(hot_velocities[phot_singles] > 10), totalnum))
    print("Predicted spectroscopic rapid rotators: {0:.1f}/{1:d}".format(
        numspec, totalnum))
    spec_rapid = aposplit.subsample_len(
        ["Hot Dwarfs", "Mcq", "Photometric Singles", "~DLSB", "Vsini det"])
    print("Actual spectroscopic rapid rotators: {0:d}/{1:d}".format(
        spec_rapid, totalnum))

    print("Photometric Binaries:")
    numspec = rot.calc_spec_rapid_num(
        np.log10(hot_velocities[~phot_singles]), sini_cutoff=sini_cutoff)
    totalnum = aposplit.subsample_len(
        ["Hot Dwarfs", "Mcq", "Photometric Binaries"])
    print("Fraction of photometric rapid rotators: {0:d}/{1:d}".format(
        np.count_nonzero(hot_velocities[~phot_singles] > 10), totalnum))
    print("Predicted spectroscopic rapid rotators: {0:.1f}/{1:d}".format(
        numspec, totalnum))
    spec_rapid = aposplit.subsample_len(
        ["Hot Dwarfs", "Mcq", "Photometric Binaries", "~DLSB", "Vsini det"])
    print("Actual spectroscopic rapid rotators: {0:d}/{1:d}".format(
        spec_rapid, totalnum))


############
# Pleiades #
############

def write_Pleiades_count_overlap():
    '''Report the number of overlapping targets with APOGEE.'''
    full_pleiades = cache.pleiades_APOGEE_Literature_vsini()
    pleiades_group = full_pleiades.group_by("APOGEE_ID")
    apogee_avg = pleiades_group[[
        "APOGEE_ID", "VSINI"]].groups.aggregate(np.mean)
    unique_pleiades = unique(full_pleiades, keys="APOGEE_ID")
    unique_pleiades.remove_column("VSINI")
    pleiades = pleiades_group[[
        "APOGEE_ID", "VSINI", "(V-K)0", "Delmag", "vsini_QuelozE", 
        "vsini_err_QuelozE", "vsini_QuelozC", "vsini_lim_QuelozC", 
        "vsini_err_QuelozC", "vsini_Terndrup", "vsini_err_Terndrup",
        "vsini_lim_Terndrup", "vsini_Soderblom", "vsini_lim_Soderblom",
        "vsini_SH", "vsini_err_SH", "vsini_lim_SH", "vsini_S84",
        "vsini_err_S84", "vsini_lim_S84", "vsini_Jackson", "vsini_err_Jackson",
    "vsini_lim_Jackson"]].groups.aggregate(np.mean)
    pleiades = au.join_by_id(
        unique_pleiades, apogee_avg, "APOGEE_ID", "APOGEE_ID")
    phot_bin = np.logical_or(
        np.logical_and(pleiades["(V-K)0"] <= 5, pleiades["Delmag"] >= 0.3),
        np.logical_and(pleiades["(V-K)0"] > 5, pleiades["Delmag"] >= 0.3))
    apogee_dets = pleiades["VSINI"] > 10

    # I want to do a census of each survey.
    queloz_elodie = ~pleiades["vsini_QuelozE"].mask
    queloz_coravel = ~pleiades["vsini_QuelozC"].mask
    full_queloz = np.logical_or(queloz_elodie, queloz_coravel)
    queloz_coravel_upper = pleiades["vsini_lim_QuelozC"] == "<"
    queloz_coravel_lower = pleiades["vsini_lim_QuelozC"] == ">"
    queloz_coravel_det = np.logical_not(
        np.logical_or(queloz_coravel_upper,queloz_coravel_lower))
    queloz_count = np.count_nonzero(full_queloz)
    print("Total overlap with Queloz: {0:d}".format(queloz_count))

    terndrup = ~pleiades["vsini_Terndrup"].mask
    terndrup_detections = pleiades["vsini_lim_Terndrup"] == "d"
    terndrup_limits = pleiades["vsini_lim_Terndrup"] == "u"
    terndrup_count = np.count_nonzero(terndrup)
    print("Total overlap with Terndrup: {0:d}".format(terndrup_count))

    soderblom = ~pleiades["vsini_Soderblom"].mask
    soderblom_detections = pleiades["vsini_lim_Soderblom"] == "d"
    soderblom_limits = pleiades["vsini_lim_Soderblom"] == "u"
    soderblom_count = np.count_nonzero(soderblom)
    print("Total overlap with Soderblom: {0:d}".format(soderblom_count))

    sh = ~pleiades["vsini_SH"].mask
    sh_detections = pleiades["vsini_lim_SH"] == "d"
    sh_upper = pleiades["vsini_lim_SH"] == "u"
    sh_count = np.count_nonzero(sh)
    print("Total overlap with Stauffer and Hartmann: {0:d}".format(sh_count))

    s84 = ~pleiades["vsini_S84"].mask
    s84_detections = pleiades["vsini_lim_S84"] == "d"
    s84_upper = pleiades["vsini_lim_S84"] == "u"
    s84_count = np.count_nonzero(s84)
    print("Total overlap with Stauffer: {0:d}".format(s84_count))

    jackson = ~pleiades["vsini_Jackson"].mask
    jackson_detections = pleiades["vsini_lim_Jackson"] == "d"
    jackson_upper = pleiades["vsini_lim_Jackson"] == "u"
    jackson_count = np.count_nonzero(jackson)
    print("Total overlap with Jackson: {0:d}".format(jackson_count))

    fullindices = au.multi_logical_or(
        full_queloz, sh, jackson)
    fullcount = np.count_nonzero(fullindices)
    print("Total number of targets in overlap: {0:d}".format(fullcount))

    citations = [
        r'\citet{Queloz98}', r'\citet{Stauffer87}', r'\citet{Jackson18}',
        'TOTAL']
    counts = [
        queloz_count, sh_count, jackson_count, fullcount]

    resolutions = [30000, 40000, 17000, 0]

    counttable = Table([citations, resolutions, counts], 
                       names=("Survey", "R", "Overlap"))

    tabletitle = r"APOGEE Overlap with Previous Literature\label{tab:pleiadescount}"
    alignment = " l c c "
    footercomment = (
        r"\tablecomments{Total number is much smaller than the sum of surveys "
        r"because of large overlap between surveys. Resolution is the lowest "
        r"resolution of the sample.}")

    latexdict = {
        "col_align": alignment, "caption": tabletitle, 
        "tablefoot": footercomment}

    counttable.write(
        str(TABLE_PATH / "counttable.tex"), format="ascii.aastex", 
        latexdict=latexdict, overwrite=True, fill_values=[("0", "", "R")])

def write_Pleiades_Supplemental_Table():
    '''Write the cross-matched Pleiades table.'''
    full_pleiades = cache.pleiades_APOGEE_Literature_vsini()
    pleiades_group = full_pleiades.group_by("APOGEE_ID")
    apogee_avg = pleiades_group[[
        "APOGEE_ID", "VSINI"]].groups.aggregate(np.mean)
    unique_pleiades = unique(full_pleiades, keys="APOGEE_ID")
    unique_pleiades.remove_column("VSINI")
    pleiades = pleiades_group[[
        "APOGEE_ID", "VSINI", "HII", "vsini_QuelozE", 
        "vsini_err_QuelozE", "vsini_QuelozC", "vsini_lim_QuelozC", 
        "vsini_err_QuelozC", "vsini_Terndrup", "vsini_err_Terndrup",
        "vsini_lim_Terndrup", "vsini_Soderblom", "vsini_lim_Soderblom",
        "vsini_SH", "vsini_err_SH", "vsini_lim_SH", "vsini_S84",
        "vsini_err_S84", "vsini_lim_S84", "vsini_Jackson", "vsini_err_Jackson",
    "vsini_lim_Jackson"]].groups.aggregate(np.mean)
    pleiades = au.join_by_id(
        unique_pleiades, apogee_avg, "APOGEE_ID", "APOGEE_ID")
    pleiades = pleiades[
        au.multi_logical_or(
            ~pleiades["vsini_QuelozE"].mask, 
            ~pleiades["vsini_QuelozC"].mask, 
            ~pleiades["vsini_SH"].mask, 
            ~pleiades["vsini_Jackson"].mask)]

    # Do some preprocessing.
    # Add a space between HII and the number.
    pleiades["HII label"] = npstr.replace(pleiades["HII"], "HII", "HII ")
    pleiades["HII label"].mask = pleiades["HII"].mask
    # Remove the uncertainties for limits when publishing table.
    pleiades["vsini_err_SH"].mask = au.multi_logical_or(
        pleiades["vsini_err_SH"].mask, pleiades["vsini_lim_SH"] != "d")
    # Replace l, u, and d with >, <, and " ". Unfortunately, npstr ignores
    # masks, so the mask has to be preserved.
    sh_mask = pleiades["vsini_lim_SH"].mask.copy()
    pleiades["vsini_lim_SH"] = npstr.replace(npstr.replace(npstr.replace(
        pleiades["vsini_lim_SH"], "l", ">"), "u", "<"), "d", ""),
    pleiades["vsini_lim_SH"].mask = sh_mask
    jackson_mask = pleiades["vsini_lim_Jackson"].mask.copy()
    pleiades["vsini_lim_Jackson"] = npstr.replace(npstr.replace(npstr.replace(
        pleiades["vsini_lim_Jackson"], "l", ">"), "u", "<"), "d", ""),
    pleiades["vsini_lim_Jackson"].mask = jackson_mask
    # For some reason, the lim for QuelozC is a byte array instead of a string
    # array, so it needs to be typecast.
    quelozc_mask = pleiades["vsini_lim_QuelozC"].mask.copy()
    string_quelozc = npstr.replace(npstr.replace(npstr.replace(np.asarray(
        pleiades["vsini_lim_QuelozC"], np.str), "l", ">"), "u", "<"), "d", "")
    del(pleiades["vsini_lim_QuelozC"])
    pleiades["vsini_lim_QuelozC"] = string_quelozc
    pleiades["vsini_lim_QuelozC"].mask = quelozc_mask
    # Remove nans from the Queloz columns.
    pleiades["vsini_QuelozE"] = np.ma.masked_invalid(pleiades["vsini_QuelozE"])
    pleiades["vsini_err_QuelozE"] = np.ma.masked_invalid(
        pleiades["vsini_err_QuelozE"])
    pleiades["vsini_QuelozC"] = np.ma.masked_invalid(pleiades["vsini_QuelozC"])
    pleiades["vsini_err_QuelozC"] = np.ma.masked_invalid(
        pleiades["vsini_err_QuelozC"])
    pleiades["vsini_lim_QuelozC"].mask = pleiades["vsini_QuelozC"].mask.copy()

    include_names = (
        "APOGEE_ID", "HII label", "vsini_lim_SH", "vsini_SH", "vsini_err_SH", 
        "vsini_QuelozE", "vsini_err_QuelozE", "vsini_lim_QuelozC", 
        "vsini_QuelozC", "vsini_err_QuelozC", "vsini_lim_Jackson",
        "vsini_Jackson", "vsini_err_Jackson")

    tabletitle = r"Pleiades Overlap Sample\label{tab:pleiadessample}"
    alignment = " l l c c c c c c c c c c c c"
    footercomment = (
        r"\tablecomments{Columns: (1) The 2MASS designation for the target. "
        r"(2) Other designations which are used in the original \vsini{} "
        r"papers. (3), (4), and (5) represent the limit designation, "
        r"\vsini{} and uncertainty reported in \citet{Stauffer87} "
        r"(6) and (7) represent the \vsini{} and uncertainty reported in "
        r"\citet{Queloz98} with the ELODIE instrument. (8), (9), and (10) "
        r"represent the limit designation, \vsini{}, and uncertainty reported "
        r"in \citet{Queloz98} with the CORAVEL instrument. Finally, (11), "
        r"(12), and (13) represent the limit designation, \vsini{} and "
        r"uncertainty reported in \citet{Jackson18}. "
        r"\Cref{tab:pleiadessample} is published in its entirely in the "
        r"machine-readable format. A portion is shown here for guidance "
        r"regarding its form and content.}")

    latexdict = {
        "col_align": alignment, "caption": tabletitle, 
        "tablefoot": footercomment}

    latex_names=(
        "APOGEE ID", "Other ID", 
        r"\citet{Stauffer87} " + vsinistr + " Limit", 
        r"\citet{Stauffer87} " + vsinistr, 
        r"\citet{Stauffer87} " + vsinistr + " Error", 
        r"\citet{Queloz98} ELODIE " + vsinistr, 
        r"\citet{Queloz98} ELODIE " + vsinistr + " Error", 
        r"\citet{Queloz98} CORAVEL " + vsinistr + " Limit", 
        r"\citet{Queloz98} CORAVEL " + vsinistr, 
        r"\citet{Queloz98} CORAVEL " + vsinistr + " Error", 
        r"\citet{Jackson18} " + vsinistr + " Limit", 
        r"\citet{Jackson18} " + vsinistr , 
        r"\citet{Jackson18} " + vsinistr + " Error")


    pleiades[include_names][0:5].write(
        str(TABLE_PATH / "total_pleiades.tex"), format="ascii.aastex", 
        latexdict=latexdict, overwrite=True, fill_values=[(ascii.masked, "")],
        names=latex_names)

    basic_names = (
        "APOGEE_ID", "OTHER_ID", "SH87_LIM", "SH87_VSINI", "SH87_ERR",
        "Q98E_VSINI", "Q98E_ERR", "Q98C_LIM", "Q98C_VSINI", "Q98C_ERR",
        "J18_LIM", "J18_VSINI", "J18_ERR")

    pleiades.meta["comments"] = [
        "APOGEE_ID: 2MASS ID for the target",
     "OTHER_ID: Additional ID used in the original work",
     "SH87_LIM: Stauffer & Hartmann (1987) flag whether vsini is a limit",
     "SH87_VSINI: Vsini as measured by Stauffer & Hartmann (1987)",
     "SH87_ERR: Vsini uncertainty reported by Stauffer & Hartmann (1987)",
     "Q98E_VSINI: Vsini as measured by Queloz et al (1998) with ELODIE",
     "Q98E_ERR: ELODIE Vsini uncertainty reported by Queloz et al (1998)",
     "Q98C_LIM: Queloz et al (1998) flag whether CORAVEL vsini is a limit",
     "Q98C_VSINI: Vsini as measured by Queloz et al (1998) with CORAVEL",
     "Q98C_ERR: CORAVEL Vsini uncertainty reported by Queloz et al (1998)",
     "J18_LIM: Jackson et al (2018) flag whether vsini is a limit",
     "J18_VSINI: Vsini as measured by Jackson et al (2018)",
     "J18_ERR: Vsini uncertainty reported by Jackson et al (2018)"]

    pleiades[include_names].write(
        str(TABLE_PATH / "total_pleiades.dat"), format="ascii.fixed_width",
        overwrite=True, fill_values=[(ascii.masked, "")], names=basic_names,
        comment="# ", formats={"SH87_ERR": "%.1f"})

def Pleiades_literature_vsini_agreement():
    '''Plot the vsini in the literature vs the ASPCAP vsini.'''
    pleiades = cache.pleiades_APOGEE_Literature_vsini()

    good_indices = catalog.aspcap_quality_bitmask_indices(
        pleiades["ASPCAPFLAG"], "good")
    warn_indices = np.logical_or(
        catalog.aspcap_quality_bitmask_indices(pleiades["ASPCAPFLAG"], "warn"),
        catalog.aspcap_quality_bitmask_indices(pleiades["ASPCAPFLAG"], "vsini"))

    cool_indices = pleiades["TEFF"] < 5500

    queloz_indices = np.logical_or(
        np.isfinite(pleiades["vsini_QuelozE"]).filled(0.0),
        np.isfinite(pleiades["vsini_QuelozC"]).filled(0.0))
    terndrup_indices = au.multi_logical_and(
        ~queloz_indices, ~pleiades["vsini_Terndrup"].mask)
    soderblom_indices = au.multi_logical_and(
        ~queloz_indices, ~terndrup_indices, ~pleiades["vsini_Soderblom"].mask)
    sh_indices = au.multi_logical_and(
        ~queloz_indices, ~terndrup_indices, ~soderblom_indices, 
        ~pleiades["vsini_SH"].mask)
    s84_indices = au.multi_logical_and(
        ~queloz_indices, ~terndrup_indices, ~soderblom_indices, ~sh_indices, 
        ~pleiades["vsini_S84"].mask)
    unknown_indices = au.multi_logical_and(
        ~queloz_indices, ~terndrup_indices, ~soderblom_indices, ~sh_indices,
        ~s84_indices)

    good_queloz_indices = au.multi_logical_and(
        good_indices, cool_indices, queloz_indices)
    good_terndrup_indices = au.multi_logical_and(
        good_indices, cool_indices, terndrup_indices)
    good_soderblom_indices = au.multi_logical_and(
        good_indices, cool_indices, soderblom_indices)
    good_sh_indices = au.multi_logical_and(
        good_indices, cool_indices, sh_indices)
    good_s84_indices = au.multi_logical_and(
        good_indices, cool_indices, s84_indices)
    warn_queloz_indices = au.multi_logical_and(
        warn_indices, cool_indices, queloz_indices)
    warn_terndrup_indices = au.multi_logical_and(
        warn_indices, cool_indices, terndrup_indices)
    warn_soderblom_indices = au.multi_logical_and(
        warn_indices, cool_indices, soderblom_indices)
    warn_sh_indices = au.multi_logical_and(
        warn_indices, cool_indices, sh_indices)
    warn_s84_indices = au.multi_logical_and(
        warn_indices, cool_indices, s84_indices)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    ax1.errorbar(
        pleiades["vsini"][good_queloz_indices], 
        pleiades["VSINI"][good_queloz_indices], 
        yerr=0.15*pleiades["VSINI"][good_queloz_indices], 
        xerr=0.15*pleiades["vsini"][good_queloz_indices], marker="", 
        color=bc.red, ls="", label="Queloz")
    ax1.errorbar(
        pleiades["vsini"][good_terndrup_indices], 
        pleiades["VSINI"][good_terndrup_indices],
        yerr=0.15*pleiades["VSINI"][good_terndrup_indices],
        xerr=0.15*pleiades["vsini"][good_terndrup_indices], marker="", 
        color="blue", ls="", label="Terndrup")
    ax1.errorbar(
        pleiades["vsini"][good_soderblom_indices], 
        pleiades["VSINI"][good_soderblom_indices],
        yerr=0.15*pleiades["VSINI"][good_soderblom_indices],
        xerr=0.15*pleiades["vsini"][good_soderblom_indices], marker="",
        color="magenta", ls="", label="Soderblom")
    ax1.errorbar(
        pleiades["vsini"][good_sh_indices], pleiades["VSINI"][good_sh_indices],
        yerr=0.15*pleiades["VSINI"][good_sh_indices],
        xerr=0.15*pleiades["vsini"][good_sh_indices], marker="", color=bc.sky_blue,
        ls="", label="Stauffer & Hartman")
    ax1.errorbar(
        pleiades["vsini"][good_s84_indices], pleiades["VSINI"][good_s84_indices],
        yerr=0.15*pleiades["VSINI"][good_s84_indices],
        xerr=0.15*pleiades["vsini"][good_s84_indices], marker="", color=bc.algae,
        ls="", label="Stauffer84")
    ax2.errorbar(
        pleiades["vsini"][warn_queloz_indices], 
        pleiades["VSINI"][warn_queloz_indices], 
        yerr=0.15*pleiades["VSINI"][warn_queloz_indices], 
        xerr=0.15*pleiades["vsini"][warn_queloz_indices], marker="", 
        color=bc.red, ls="", label="Queloz")
    ax2.errorbar(
        pleiades["vsini"][warn_terndrup_indices], 
        pleiades["VSINI"][warn_terndrup_indices],
        yerr=0.15*pleiades["VSINI"][warn_terndrup_indices],
        xerr=0.15*pleiades["vsini"][warn_terndrup_indices], marker="", 
        color="blue", ls="", label="Terndrup")
    ax2.errorbar(
        pleiades["vsini"][warn_soderblom_indices], 
        pleiades["VSINI"][warn_soderblom_indices],
        yerr=0.15*pleiades["VSINI"][warn_soderblom_indices],
        xerr=0.15*pleiades["vsini"][warn_soderblom_indices], marker="",
        color="magenta", ls="", label="Soderblom")
    ax2.errorbar(
        pleiades["vsini"][warn_sh_indices], pleiades["VSINI"][warn_sh_indices],
        yerr=0.15*pleiades["VSINI"][warn_sh_indices],
        xerr=0.15*pleiades["vsini"][warn_sh_indices], marker="", color=bc.sky_blue,
        ls="", label="Stauffer & Hartman")
    ax2.errorbar(
        pleiades["vsini"][warn_s84_indices], pleiades["VSINI"][warn_s84_indices],
        yerr=0.15*pleiades["VSINI"][warn_s84_indices],
        xerr=0.15*pleiades["vsini"][warn_s84_indices], marker="", color=bc.algae,
        ls="", label="Stauffer84")
    ax1.plot([1, 100], [1, 100], 'k--')
    ax2.plot([1, 100], [1, 100], 'k--')

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax1.set_xlabel("Literature Vsini")
    ax2.set_xlabel("Literature Vsini")
    ax1.set_xlabel("APOGEE Vsini")
    ax1.set_xlim(1, 100)
    ax1.set_ylim(1, 100)
    ax2.set_xlim(1, 100)
    ax2.set_ylim(1, 100)
    ax2.legend(loc="upper left")
    ax1.set_title("Good targets")
    ax2.set_title("Warn targets")

def Pleiades_vsini_comparison():
    '''Compare the APOGEE to literature vsini for the Pleiades.'''
    pleiades = cache.pleiades_APOGEE_Literature_vsini()
    vsini_lim = 10

    cool_indices = pleiades["TEFF"] < 5500

    queloz_indices = np.logical_or(
        np.isfinite(pleiades["vsini_QuelozE"]).filled(0.0),
        np.isfinite(pleiades["vsini_QuelozC"]).filled(0.0))
    terndrup_indices = au.multi_logical_and(
        ~queloz_indices, ~pleiades["vsini_Terndrup"].mask)
    soderblom_indices = au.multi_logical_and(
        ~queloz_indices, ~terndrup_indices, ~pleiades["vsini_Soderblom"].mask)
    sh_indices = au.multi_logical_and(
        ~queloz_indices, ~terndrup_indices, ~soderblom_indices, 
        ~pleiades["vsini_SH"].mask)
    s84_indices = au.multi_logical_and(
        ~queloz_indices, ~terndrup_indices, ~soderblom_indices, ~sh_indices, 
        ~pleiades["vsini_S84"].mask)
    unknown_indices = au.multi_logical_and(
        ~queloz_indices, ~terndrup_indices, ~soderblom_indices, ~sh_indices,
        ~s84_indices)

    good_queloz = au.multi_logical_and(cool_indices, queloz_indices)
    good_terndrup = au.multi_logical_and(cool_indices, terndrup_indices)
    good_soderblom = au.multi_logical_and(cool_indices, soderblom_indices)
    good_sh = au.multi_logical_and(cool_indices, sh_indices)
    good_s84 = au.multi_logical_and(cool_indices, s84_indices)

    vdiff = np.log10(pleiades["VSINI"]) - np.log10(pleiades["vsini"])
    dets = np.logical_and(pleiades["VSINI"] > vsini_lim, ~queloz_indices)

    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.errorbar(
        pleiades["vsini"][good_queloz], vdiff[good_queloz],
        yerr=0.15*np.sqrt(2)/np.log(10),
        xerr=0.15*pleiades["vsini"][good_queloz], color=bc.red, marker=".",
        ls="", label="Queloz")
    ax.errorbar(
        pleiades["vsini"][good_terndrup], vdiff[good_terndrup],
        yerr=0.15*np.sqrt(2)/np.log(10),
        xerr=0.15*pleiades["vsini"][good_terndrup], color="blue", marker=".",
        ls="", label="Terndrup")
    ax.errorbar(
        pleiades["vsini"][good_soderblom], vdiff[good_soderblom],
        yerr=0.15*np.sqrt(2)/np.log(10),
        xerr=0.15*pleiades["vsini"][good_soderblom], color="magenta", marker=".",
        ls="", label="Soderblom")
    ax.errorbar(
        pleiades["vsini"][good_sh], vdiff[good_sh],
        yerr=0.15*np.sqrt(2)/np.log(10),
        xerr=0.15*pleiades["vsini"][good_sh], color=bc.sky_blue, marker=".",
        ls="", label="Stauffer & Hartmann")
    ax.errorbar(
        pleiades["vsini"][good_s84], vdiff[good_s84],
        yerr=0.15*np.sqrt(2)/np.log(10),
        xerr=0.15*pleiades["vsini"][good_s84], color=bc.algae, marker=".",
        ls="", label="Stauffer 84")
    ax.plot([1, 100], [0, 0], 'k-')
    ax.plot([vsini_lim, vsini_lim], [-0.5, 0.5], 'r--')

    offset = np.mean(vdiff[dets])
    disp = np.std(vdiff[dets])
    print("Offset: {0:.2f} +/- {1:.3f}".format(
        offset, disp/np.sqrt(np.count_nonzero(dets))))
    ax.plot([1, 100], [offset, offset], 'm:')

    ax.set_xlim(1, 100)
    ax.set_xscale("log")
    ax.set_xlabel("Literature vsini")
    ax.set_ylabel("Log(APOGEE vsini / Lit vsini)")
    ax.legend(loc="upper right")

def Pleiades_vsini_agreement():
    '''Plot the vsini vs veq diagram for the Pleiades.'''
    pleiades = cache.pleiades()
    pleiades_vsini = pleiades[
        np.logical_and(~pleiades["VSINI"].mask, pleiades["TEFF"] < 5500)]

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    rot.plot_vsini_velocity(
        pleiades_vsini["VSINI"], pleiades_vsini["Per1"],
        pleiades_vsini["Per1"]*0.07, pleiades_vsini["MIST R"],
        pleiades_vsini["MIST R Err"], ax=ax)
    ax.plot([0, 70], [0, 100], ls='-.', c='k', marker="")
    ax.set_xlabel(r"$v_{eq}$ (km/s) from MIST $R$ and $P_{rot}$")
    ax.title("Radius from MIST")

def Stauffer_Hartmann_vsini_agreement():
    '''Plot the vsini vs veq diagram for the Pleiades.'''
    pleiades = cache.stauffer_hartmann_pleiades()
    sh_upperlimits = pleiades["vsini lim"] == "u"

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    rot.plot_vsini_velocity(
        pleiades["vsini_SH"][~sh_upperlimits],
        pleiades["Per1"][~sh_upperlimits],
        pleiades["Per1"][~sh_upperlimits]*0.07, 
        pleiades["Baraffe Radius"][~sh_upperlimits],
        pleiades["Baraffe Radius"][~sh_upperlimits]*0.05, ax=ax, vsini_lim=10)
    rot.plot_vsini_velocity(
        pleiades["vsini_SH"][sh_upperlimits], pleiades["Per1"][sh_upperlimits],
        pleiades["Per1"][sh_upperlimits]*0.07, 
        pleiades["Baraffe Radius"][sh_upperlimits], 
        pleiades["Baraffe Radius"][sh_upperlimits]*0.05, ax=ax, marker="v",
        color="r", vsini_fracerr=0.0, vsini_lim=10)
    ax.plot([0, 85], [0, 100], ls='-.', c='k', marker="")
    ax.set_xlabel(r"$v_{eq}$ (km/s) from MIST $R$ and $P_{rot}$")
    ax.set_ylabel("Stauffer & Hartmann vsini")

def Pleiades_vsini_agreement_Lbol():
    '''Plot the vsini vs veq diagram for the Pleiades.
    
    In this figure, the radius is not derived from MIST isochrones, but rather
    from the K-band absolute magnitude.'''
    pleiades = cache.pleiades()
    pleiades_vsini = pleiades[
        np.logical_and(~pleiades["VSINI"].mask, pleiades["TEFF"] < 5500)]

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    rot.plot_vsini_velocity(
        pleiades_vsini["VSINI"], pleiades_vsini["Per1"], 
        pleiades_vsini["K-band R"], 0, 0, ax=ax)
    ax.plot([0, 70], [0, 100], ls='-.', c='k', marker="")

def Pleiades_vsini_agreement_RBaraffe():
    '''Plot the vsini vs. veq diagram for Pleiades using direct Baraffe R.
    
    This function uses the relationship between R and Teff for Baraffe
    isochrones to derive the radius. This is analogous to what is done with
    MIST.'''
    pleiades = cache.pleiades()
    pleiades_vsini = pleiades[
        np.logical_and(~pleiades["VSINI"].mask, pleiades["TEFF"] < 5500)]
    phot_binaries = pleiades_vsini["Delmag"] > 0.2

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    iso = baraffe.BaraffeIsochrone.isochrone_from_file()
    baraffe_rad = iso.interpolate_isochrone_cols(
        0.12, pleiades_vsini["TEFF"], iso.teff_col, iso.radius_col)
    baraffe_rad_err = pleiades_vsini["TEFF_ERR"] * iso.isochrone_derivative(
        0.12, pleiades_vsini["TEFF"], iso.teff_col, iso.radius_col)
    rot.plot_vsini_velocity(
        pleiades_vsini["VSINI"], pleiades_vsini["Per1"],
        pleiades_vsini["Per1"]*0.07, baraffe_rad, baraffe_rad_err, ax=ax,
        label="Phot. Single")
    rot.plot_vsini_velocity(
        pleiades_vsini["VSINI"][phot_binaries],
        pleiades_vsini["Per1"][phot_binaries],
        pleiades_vsini["Per1"][phot_binaries]*0.07, baraffe_rad[phot_binaries],
        baraffe_rad_err[phot_binaries], ax=ax, color="r", label="Phot Binary")
    ax.plot([0, 70], [0, 100], ls='-.', c='k', marker="")
    ax.set_title("Baraffe Direct Radius")
    ax.legend(loc="lower left")

def Pleiades_vsini_agreement_RStauffer():
    '''Plot the vsini vs veq diagram for the Pleiades.
    
    In this figure, the radius is not derived from MIST isochrones, but rather
    from the K-band absolute magnitude.'''
    pleiades = cache.pleiades()
    pleiades_vsini = pleiades[
        np.logical_and(~pleiades["VSINI"].mask, pleiades["TEFF"] < 5500)]
    phot_binaries = pleiades_vsini["Delmag"] > 0.2

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    masses = pleiades_vsini["Mass"]
    iso = baraffe.BaraffeIsochrone.isochrone_from_file()
    baraffe_rad = iso.interpolate_isochrone_cols(
        0.12, masses, iso.mass_col, iso.radius_col)
    log_baraffe_lum = iso.interpolate_isochrone_cols(
        0.12, masses, iso.mass_col, iso.logL_col)
    apogee_rad = 10**(
        0.5*(log_baraffe_lum - 4*(np.log10(pleiades_vsini["TEFF"]) - np.log10(5777))))
    apogee_lograd_err = 2*pleiades_vsini["TEFF_ERR"] / pleiades_vsini["TEFF"] / np.log(10)
    apogee_rad_err = apogee_lograd_err * apogee_rad * np.log(10)
    rot.plot_vsini_velocity(
        pleiades_vsini["VSINI"], pleiades_vsini["Per1"],
        pleiades_vsini["Per1"]*0.07, pleiades_vsini["K-band R (Baraffe)"],
        pleiades_vsini["K-band R Err (Baraffe)"], ax=ax, label="Phot. Singles")
    rot.plot_vsini_velocity(
        pleiades_vsini["VSINI"][phot_binaries],
        pleiades_vsini["Per1"][phot_binaries],
        pleiades_vsini["Per1"][phot_binaries]*0.07, 
        pleiades_vsini["K-band R (Baraffe)"][phot_binaries],
        pleiades_vsini["K-band R Err (Baraffe)"][phot_binaries], ax=ax, 
        label="Phot. Binaries", color="r")
    ax.plot([0.85, 85], [1, 100], ls='-.', c='k', marker="")
    ax.set_title("Radius from Deprojected mass")
    ax.legend(loc="lower left")

def Pleiades_vsini_discrepant_points():
    '''Highlight vsini discrepancies in literature vs APOGEE.
    
    In this figure, the radius is not derived from MIST isochrones, but rather
    from the K-band absolute magnitude.'''
    pleiades = cache.pleiades()
    pleiades_vsini = pleiades[
        np.logical_and(~pleiades["VSINI"].mask, pleiades["TEFF"] < 5500)]
    discrepant_targets = np.logical_and(
        ~pleiades_vsini["vsini"].mask, 
        pleiades_vsini["VSINI"] > 1.5 * np.maximum(pleiades_vsini["vsini"], 7))

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    masses = pleiades_vsini["Mass"]
    iso = baraffe.BaraffeIsochrone.isochrone_from_file()
    baraffe_rad = iso.interpolate_isochrone_cols(
        0.12, masses, iso.mass_col, iso.radius_col)
    log_baraffe_lum = iso.interpolate_isochrone_cols(
        0.12, masses, iso.mass_col, iso.logL_col)
    apogee_rad = 10**(
        0.5*(log_baraffe_lum - 4*(np.log10(pleiades_vsini["TEFF"]) - np.log10(5777))))
    apogee_lograd_err = 2*pleiades_vsini["TEFF_ERR"] / pleiades_vsini["TEFF"] / np.log(10)
    apogee_rad_err = apogee_lograd_err * apogee_rad * np.log(10)
    rot.plot_vsini_velocity(
        pleiades_vsini["VSINI"], pleiades_vsini["Per1"],
        pleiades_vsini["Per1"]*0.07, pleiades_vsini["K-band R (Baraffe)"],
        pleiades_vsini["K-band R Err (Baraffe)"], ax=ax, label="Phot. Singles")
    rot.plot_vsini_velocity(
        pleiades_vsini["vsini"][discrepant_targets],
        pleiades_vsini["Per1"][discrepant_targets],
        pleiades_vsini["Per1"][discrepant_targets]*0.07, 
        pleiades_vsini["K-band R (Baraffe)"][discrepant_targets],
        pleiades_vsini["K-band R Err (Baraffe)"][discrepant_targets], ax=ax, 
        label="Discrepant APOGEE vsini", color="r")
    rot.plot_vsini_velocity(
        pleiades_vsini["VSINI"][discrepant_targets],
        pleiades_vsini["Per1"][discrepant_targets],
        pleiades_vsini["Per1"][discrepant_targets]*0.07, 
        pleiades_vsini["K-band R (Baraffe)"][discrepant_targets],
        pleiades_vsini["K-band R Err (Baraffe)"][discrepant_targets], ax=ax, 
        label="Discrepant Literature vsini", color="cyan")
    ax.plot([0.85, 85], [1, 100], ls='-.', c='k', marker="")
    ax.set_title("Radius from Deprojected mass")
    ax.legend(loc="lower left")

    multip_cols = [
        "APOGEE_ID", "Per_MU", "dd", "ddm", "shch", "beat", "cpeak", "resc", 
        "resd", "dscu", "cloud", "PPer", "SPer", "TPer", "QPer"]
    print(pleiades_vsini[multip_cols][discrepant_targets])


def Pleiades_vsini_radius_inflation_teff():
    '''Plot the vsini vs veq diagram for the Pleiades.
    
    In this figure, the radius is not derived from MIST isochrones, but rather
    from the K-band absolute magnitude.'''
    pleiades = cache.pleiades()
    pleiades_vsini = pleiades[
        np.logical_and(~pleiades["VSINI"].mask, pleiades["TEFF"] < 5500)]
    cools = pleiades_vsini["TEFF"] < 5000

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    masses = pleiades_vsini["Mass"]
    iso = baraffe.BaraffeIsochrone.isochrone_from_file()
    baraffe_rad = iso.interpolate_isochrone_cols(
        0.12, masses, iso.mass_col, iso.radius_col)
    log_baraffe_lum = iso.interpolate_isochrone_cols(
        0.12, masses, iso.mass_col, iso.logL_col)
    apogee_rad = 10**(
        0.5*(log_baraffe_lum - 4*(np.log10(pleiades_vsini["TEFF"]) - np.log10(5777))))
    apogee_lograd_err = 2*pleiades_vsini["TEFF_ERR"] / pleiades_vsini["TEFF"] / np.log(10)
    apogee_rad_err = apogee_lograd_err * apogee_rad * np.log(10)
    rot.plot_vsini_velocity(
        pleiades_vsini["VSINI"], pleiades_vsini["Per1"],
        pleiades_vsini["Per1"]*0.07, apogee_rad, apogee_rad_err, ax=ax,
        label="Teff > 5000 K")
    rot.plot_vsini_velocity(
        pleiades_vsini["VSINI"][cools],
        pleiades_vsini["Per1"][cools],
        pleiades_vsini["Per1"][cools]*0.07, apogee_rad[cools],
        apogee_rad_err[cools], ax=ax, label="Teff < 5000 K", color="r")
    ax.plot([0.85, 85], [1, 100], ls='-.', c='k', marker="")
    ax.set_title("Radius from Deprojected mass")
    ax.legend(loc="lower left")

def Pleiades_RMIST_Rbol_comparison():
    '''Compare the radii from MIST and from bolometric luminosities.'''
    pleiades = cache.pleiades()
    ok = pleiades["memb"] == "ok"

    f, ax = plt.subplots(1, 1, figsize=(12, 12))

    validRs = ~np.ma.getmaskarray(pleiades["Stauffer R (Baraffe)"])
    coeffs = np.polyfit(
        pleiades["Stauffer R (MIST)"][validRs], 
        pleiades["Stauffer R (Baraffe)"][validRs], 1)
    print("The relationship is y = {0:.2f} x + {1:.2f}".format(
        coeffs[0], coeffs[1]))
    xvals = np.linspace(0.3, 1.5, 2, endpoint=True)
    yvals = coeffs[0] * xvals + coeffs[1]

    ax.plot(
        pleiades["Stauffer R (MIST)"], pleiades["Stauffer R (Baraffe)"], 
        color='k', marker='o', ls="", label="K")
    ax.plot(xvals, yvals, 'r-')
    ax.plot([xvals[0], xvals[1]], [xvals[0], xvals[1]], 'k-')
    ax.set_xlabel("MIST R")
    ax.set_ylabel("Baraffe R")
    ax.legend(loc="lower right")

@write_plot("Pleiades_MIST_Baraffe_Comp")
def Pleiades_log_RMIST_Rbol_comparison():
    '''Compare the radii from MIST and from bolometric luminosities.'''
    pleiades = cache.pleiades()
    ok = pleiades["memb"] == "ok"

    f, ax = plt.subplots(1, 1, figsize=(12, 12))

    validRs = ~np.ma.getmaskarray(pleiades["Stauffer R (Baraffe)"])
    logMISTrads = np.log10(pleiades["Stauffer R (MIST)"][validRs])
    logBarafferads = np.log10(pleiades["Stauffer R (Baraffe)"][validRs])
    coeffs = np.polyfit(logMISTrads, logBarafferads, 1)
        
    print("The relationship is y = {0:.3f} x + {1:.3f}".format(
        coeffs[0], coeffs[1]))
    xvals = np.linspace(-0.6, 0.2, 2, endpoint=True)
    yvals = coeffs[0] * xvals + coeffs[1]

    ax.plot(
        logMISTrads, logBarafferads,
        color='k', marker='o', ls="", label="K")
    ax.plot(xvals, yvals, 'r-')
    ax.plot([xvals[0], xvals[1]], [xvals[0], xvals[1]], 'k-')
    ax.set_xlabel("MIST log(R/Rsun)")
    ax.set_ylabel("Baraffe log(R/Rsun)")
    ax.legend(loc="lower right")

def Pleiades_log_MIST_APOGEE_radius_comparison():
    '''Compare radii directly from MIST to that from SB and APOGEE.'''
    pleiades = cache.pleiades()
    ok = pleiades["memb"] == "ok"

    f, ax = plt.subplots(1, 1, figsize=(12, 12))

    validRs = ~np.ma.getmaskarray(pleiades["Stauffer R (MIST)"])
    logMISTrads = np.log10(pleiades["Stauffer R (MIST)"][validRs])
    logAPOGEErads = np.log10(pleiades["Stauffer R (APOGEE)"][validRs])
    coeffs = np.polyfit(logMISTrads, logAPOGEErads, 1)
        
    print("The relationship is y = {0:.3f} x + {1:.3f}".format(
        coeffs[0], coeffs[1]))
    xvals = np.linspace(-0.6, 0.2, 2, endpoint=True)
    yvals = coeffs[0] * xvals + coeffs[1]

    ax.plot(
        logMISTrads, logBarafferads,
        color='k', marker='o', ls="", label="K")
    ax.plot(xvals, yvals, 'r-')
    ax.plot([xvals[0], xvals[1]], [xvals[0], xvals[1]], 'k-')
    ax.set_xlabel("MIST log(R/Rsun)")
    ax.set_ylabel("APOGEE log(R/Rsun)")
    ax.legend(loc="lower right")

def reproduce_Pleiades_Stauffer_Deprojection():
    '''Remake Fig 4 from Stauffer et al (2016).'''
    pleiades = cache.pleiades()
    ok = pleiades["memb"] == "ok"
    best = pleiades["memb"] == "best"

    f, ax = plt.subplots(1, 1, figsize=figsize)

    stauffer_mv = (
        pleiades["(V-K)0"] + pleiades["Stauffer MK"] + 5 * np.log10(136.2/10) +
        0.12)
    MV = pleiades["Vmag_RE"] 

    ax.plot(pleiades["(V-K)0"][best], stauffer_mv[best], color='k', 
            marker="o", ls="", label="Deprojected")
    ax.plot(pleiades["(V-K)0"][best], MV[best], color='m', 
            marker=".", ls="", label="Raw")

    ax.legend(loc="lower left")
    ax.set_xlabel("(V-K)0")
    ax.set_ylabel("M_K0")
    hr.invert_y_axis(ax)

def Pleiades_Stauffer_Deprojection():
    '''Plot the deprojected points from Stauffer.'''
    pleiades = cache.pleiades()
    ok = pleiades["memb"] == "ok"
    best = pleiades["memb"] == "best"

    f, ax = plt.subplots(1, 1, figsize=figsize)
    MK = pleiades["Ksmag_RE"] - 5 * np.log10(136.2/10) - 0.01

    ax.plot(pleiades["(V-K)0"][best], pleiades["Stauffer MK"][best], color='k', 
            marker="o", ls="", label="Deprojected")
    ax.plot(pleiades["(V-K)0"][best], MK[best], color='m', 
            marker=".", ls="", label="Dereddened")

    mist_iso = mist.MISTIsochrone.isochrone_from_file(0.0)
    mist_pleiades = mist_iso.iso_table(1.2e8)
    mist_k = mist_pleiades[mist.band_translation["Ks"]]
    mist_v_k = (
        mist_pleiades[mist.band_translation["V"]] -
        mist_pleiades[mist.band_translation["Ks"]])
    
    ax.plot(mist_v_k, mist_k, color="r", ls="-", marker="", label="MIST")

    baraffe_iso = baraffe.BaraffeIsochrone.isochrone_from_file()
    baraffe_pleiades = baraffe_iso.iso_table(0.12)
    baraffe_k = baraffe_pleiades[baraffe.band_translation["Ks"]]
    baraffe_v_k = (
        baraffe_pleiades[baraffe.band_translation["V"]] -
        baraffe_pleiades[baraffe.band_translation["Ks"]])

    ax.plot(baraffe_v_k, baraffe_k, color="c", ls="--", marker="",
            label="Baraffe")

    yrec_iso = yrec.YRECIsochrone.isochrone_from_file()
    yrec_pleiades = yrec_iso.iso_table(0.12)
    yrec_k = (
        yrec_pleiades[yrec.band_translation["V"]] -
        yrec_pleiades[yrec.band_translation["V-K"]])
    yrec_v_k = yrec_pleiades[yrec.band_translation["V-K"]]

    ax.plot(yrec_v_k, yrec_k, color="b", ls=":", marker="", label="YREC")


    ax.legend(loc="lower left")
    ax.set_xlabel("(V-K)0")
    ax.set_ylabel("M_K0")
    hr.invert_y_axis(ax)

def Pleiades_Stauffer_Deprojection_V_V_K():
    '''Plot the deprojected points from Stauffer.'''
    pleiades = cache.pleiades()
    ok = pleiades["memb"] == "ok"
    best = pleiades["memb"] == "best"

    f, ax = plt.subplots(1, 1, figsize=figsize)

    stauffer_mv = pleiades["(V-K)0"] + pleiades["Stauffer MK"]
    MV = pleiades["Vmag_RE"] - 5 * np.log10(136.2/10) - 0.12

    ax.plot(pleiades["(V-K)0"][best], stauffer_mv[best], color='k', 
            marker="o", ls="", label="Deprojected")
    ax.plot(pleiades["(V-K)0"][best], MV[best], color='m', 
            marker=".", ls="", label="Raw")

    mist_iso = mist.MISTIsochrone.isochrone_from_file(0.0)
    mist_pleiades = mist_iso.iso_table(1.2e8)
    mist_v = mist_pleiades[mist.band_translation["V"]]
    mist_v_k = (
        mist_pleiades[mist.band_translation["V"]] -
        mist_pleiades[mist.band_translation["Ks"]])
    
    ax.plot(mist_v_k, mist_v, color="r", ls="-", marker="", label="MIST")

    baraffe_iso = baraffe.BaraffeIsochrone.isochrone_from_file()
    baraffe_pleiades = baraffe_iso.iso_table(0.12)
    baraffe_v = baraffe_pleiades[baraffe.band_translation["V"]]
    baraffe_v_k = (
        baraffe_pleiades[baraffe.band_translation["V"]] -
        baraffe_pleiades[baraffe.band_translation["Ks"]])

    ax.plot(baraffe_v_k, baraffe_v, color="c", ls="--", marker="",
            label="Baraffe")

    yrec_iso = yrec.YRECIsochrone.isochrone_from_file()
    yrec_pleiades = yrec_iso.iso_table(0.12)
    yrec_v = yrec_pleiades[yrec.band_translation["V"]]
    yrec_v_k = yrec_pleiades[yrec.band_translation["V-K"]]

    ax.plot(yrec_v_k, yrec_v, color="b", ls=":", marker="", label="YREC")

    ax.set_xlabel("(V-K)0")
    ax.set_ylabel("M_V0")
    ax.legend(loc="lower left")
    hr.invert_y_axis(ax)

def Pleiades_Color_vs_Teff_MK():
    '''Compare the Single-star MK derived from color vs APOGEE Teff.'''
    pleiades = cache.pleiades()
    ok = pleiades["memb"] == "ok"

def Pleiades_Color_vs_Teff_MK():
    '''Compare the Single-star MK derived from color vs APOGEE Teff.'''
    pleiades = cache.pleiades()
    ok = pleiades["memb"] == "ok"

    f, ax = plt.subplots(1, 1, figsize=figsize)

    hr.absmag_teff_plot(
        pleiades["TEFF"], pleiades["Stauffer MK"], color="k", marker="o",
        ls="", label="Deprojected")

    iso = mist.MISTIsochrone.isochrone_from_file(0.0)
    pleiad_iso = iso.iso_table(1.2e8)
    hr.absmag_teff_plot(
        10**pleiad_iso[iso.logteff_col], 
        pleiad_iso[mist.band_translation["Ks"]], color="r", marker="", ls="-", 
        label="MIST")

    biso = baraffe.BaraffeIsochrone.isochrone_from_file()
    pleiad_biso = biso.iso_table(0.12)
    hr.absmag_teff_plot(
        pleiad_biso[biso.teff_col],
        pleiad_biso[baraffe.band_translation["Ks"]], color='c', marker="",
        ls="--", label="Baraffe") 

    ax.set_xlabel("TEFF")
    ax.set_ylabel("M_K")
    ax.legend(loc="upper right")

def Pleiades_Color_Teff_Comparison():
    '''Show the color-Teff relation of the deprojected sample vs models.'''
    pleiades = cache.pleiades()
    ok = pleiades["memb"] == "ok"

    f, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(
        pleiades["(V-K)0"], pleiades["TEFF"], color="k", marker="o",
        ls="", label="Pleiades")

    iso = mist.MISTIsochrone.isochrone_from_file(0.0)
    pleiad_iso = iso.iso_table(1.2e8)
    pleiad_iso = models.interpolation_table_increasing_stretch(
        pleiad_iso, mono_col=iso.logteff_col)
    mist_V_K = (
        pleiad_iso[mist.band_translation["V"]] - 
        pleiad_iso[mist.band_translation["Ks"]])
    ax.plot(
        mist_V_K, 10**pleiad_iso[iso.logteff_col], color="r", marker="", ls="-", 
        label="MIST")

    biso = baraffe.BaraffeIsochrone.isochrone_from_file()
    pleiad_biso = biso.iso_table(0.12)
    pleiad_biso = models.interpolation_table_increasing_stretch(
        pleiad_biso, mono_col=biso.teff_col)
    baraffe_V_K = (
        pleiad_biso[baraffe.band_translation["V"]] -
        pleiad_biso[baraffe.band_translation["Ks"]])
    ax.plot(
        baraffe_V_K, pleiad_biso[biso.teff_col], color='c', marker="", ls="--", 
        label="Baraffe") 

    ax.set_xlabel("V-K")
    ax.set_ylabel("Teff")
    ax.legend(loc="upper right")


def Pleiades_hr_diagram():
    '''An HR diagram of the Pleiades targets observed by APOGEE with periods.'''
    pleiades = cache.pleiades()
    high_vsini = pleiades["VSINI"] > 10

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    
    yerr = np.sqrt(pleiades["K_ERR"]**2 + (5*1.2/136.2/np.log(10))**2)
    hr.absmag_teff_plot(
        pleiades["TEFF"], pleiades["MK"], yerr=yerr, xerr=pleiades["TEFF_ERR"], color=bc.black, ls="", marker=".")
    hr.absmag_teff_plot(
        pleiades[high_vsini]["TEFF"], pleiades[high_vsini]["MK"],
        yerr=yerr[high_vsini], xerr=pleiades[high_vsini]["TEFF_ERR"], color=bc.red, ls="", marker="o")

    mist_iso = mist.MISTIsochrone.isochrone_from_file(0.0)
    tab = mist_iso.iso_table(1.2e8)
    hr.absmag_teff_plot(
        10**tab[mist_iso.logteff_col], tab[mist.band_translation["K"]], 
        color="k", ls="-", marker="")
                        
def Pleiades_K_Excess():
    '''An HR diagram of the Pleiades targets observed by APOGEE with periods.'''
    pleiades = cache.pleiades()
    high_vsini = pleiades["VSINI"] > 10

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    
    hr.absmag_teff_plot(
        pleiades["TEFF"], pleiades["K Excess"], color=bc.black, ls="", marker=".")
    hr.absmag_teff_plot(
        pleiades[high_vsini]["TEFF"], pleiades[high_vsini]["K Excess"], color=bc.red, ls="", marker="o")

def APOGEE_Pleiades_targeting():
    '''Read in all stars targeted by APOGEE in the Pleiades.'''
    pleiades = catin.read_dr14_allStar(opt="Pleiades")

    calib_indices = catalog.target_indices(pleiades, "APOGEE_CALIB_CLUSTER")
    young_indices = catalog.target_indices(pleiades, "APOGEE2_YOUNG_CLUSTER")
    apokasc_indices = catalog.target_indices(pleiades, "APOGEE2_APOKASC")

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        pleiades["TEFF"][calib_indices], pleiades["K"][calib_indices],
        yerr=pleiades["K_ERR"][calib_indices], 
        xerr=pleiades["TEFF_ERR"][calib_indices], marker=".", color=bc.red,
        ls="", axis=ax, label="Calibration")
    hr.absmag_teff_plot(
        pleiades["TEFF"][young_indices], pleiades["K"][young_indices],
        yerr=pleiades["K_ERR"][young_indices], 
        xerr=pleiades["TEFF_ERR"][young_indices], marker=".", color=bc.algae,
        ls="", axis=ax, label="Young Clusters")

    pleiades_mist = mist.MISTIsochrone.isochrone_from_file(0.0)
    pleiades_iso = pleiades_mist.iso_table(1.2e8)
    ms_iso = pleiades_iso[pleiades_iso["phase"] >= 0]

    hr.absmag_teff_plot(
        10**ms_iso[pleiades_mist.logteff_col],
        ms_iso[mist.band_translation["Ks"]] + 5 * np.log10(136/10) + 0.01, 
        ls="-", color=bc.black, marker="", label="MIST 120 Gyr")

    ax.legend(loc="lower left")
    ax.set_xlim(7500, 3500)
    ax.set_ylim(12, 7)
    ax.set_ylabel("K")

def Pleiades_APOGEE_targeting():
    '''Show which stars were targeted by APOGEE in different programs.'''
    pleiades = cache.pleiades_APOGEE_Literature_vsini()

    calib_indices = catalog.target_indices(pleiades, "APOGEE_CALIB_CLUSTER")
    young_indices = catalog.target_indices(pleiades, "APOGEE2_YOUNG_CLUSTER")
    apokasc_indices = catalog.target_indices(pleiades, "APOGEE2_APOKASC")
    telluric = np.logical_or(
        catalog.target_indices(pleiades, "APOGEE_TELLURIC"),
        catalog.target_indices(pleiades, "APOGEE2_TELLURIC"))
    others = ~au.multi_logical_or(
        calib_indices, young_indices, apokasc_indices, telluric)

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        pleiades["TEFF"][calib_indices], pleiades["K"][calib_indices],
        yerr=pleiades["K_ERR"][calib_indices], 
        xerr=pleiades["TEFF_ERR"][calib_indices], marker=".", color=bc.red,
        ls="", axis=ax, label="Calibration")
    hr.absmag_teff_plot(
        pleiades["TEFF"][young_indices], pleiades["K"][young_indices],
        yerr=pleiades["K_ERR"][young_indices], 
        xerr=pleiades["TEFF_ERR"][young_indices], marker=".", color=bc.algae,
        ls="", axis=ax, label="Young Clusters")

    ax.legend(loc="lower left")
    ax.set_xlim(7500, 3500)
    ax.set_ylim(12, 7)
    ax.set_ylabel("K")

def Pleiades_Rebull_vsini_comparison():
    '''Compare the APOGEE vsini to the vsini collected from Rebull.'''
    pleiades = cache.pleiades()
    discrepant_targets = np.logical_and(
        ~pleiades["vsini"].mask, 
        pleiades["VSINI"] > 1.5 * np.maximum(pleiades["vsini"], 7))
    
    f, (ax1) = plt.subplots(1, 1, figsize=(12, 12)) 
    ax1.plot(pleiades["vsini"], pleiades["VSINI"], 'k.')
    ax1.plot(pleiades["vsini"][discrepant_targets],
            pleiades["VSINI"][discrepant_targets], 'ro')
    ax1.plot([1, 90], [1, 90], 'k-')
    ax1.plot([1, 90], [1.1, 99], 'k--')
    ax1.plot([1, 90], [0.9, 81], 'k--')
    ax1.plot([1, 90], [7, 7], 'r:')
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Literature Vsini")
    ax1.set_ylabel("APOGEE vsini")

    hr.absmag_teff_plot(
        pleiades["TEFF"], pleiades["MK"], color=bc.black, ls="", marker=".",
        axis=ax2)
    hr.absmag_teff_plot(
        pleiades[discrepant_targets]["TEFF"],
        pleiades[discrepant_targets]["MK"], color="b",
        ls="", marker="o", axis=ax2)

    mist_iso = mist.MISTIsochrone.isochrone_from_file(0.0)
    tab = mist_iso.iso_table(1.2e8)
    hr.absmag_teff_plot(
        10**tab[mist_iso.logteff_col], tab[mist.band_translation["K"]], 
        color="k", ls="-", marker="", axis=ax2)
    ax2.set_xlabel("APOGEE Teff")
    ax2.set_ylabel("MK")
    print(pleiades[["APOGEE_ID", "LOCATION_ID", "VSINI", "vsini"]][discrepant_targets])

@write_plot("quelozcomp")
def compare_Queloz_to_others():
    '''Compare Queloz vsini to other surveys.'''
    def strip_index(col, prefix, dtype):
        try:
            strippedcol = npstr.strip(col, prefix)
        except TypeError:
            strippedcol = col
        return strippedcol.astype(dtype)

    queloz_pleiades = catin.read_Queloz_Pleiades()[
        ["Star", "m_Star", "l_vsiniC", "vsiniC", "e_vsiniC", "vsiniE",
         "e_vsiniE"]]
    queloz_pleiades.rename_column("m_Star", "SBflag_Queloz")
    queloz_pleiades.rename_column("l_vsiniC", "vsini_lim_QuelozC")
    queloz_pleiades.rename_column("vsiniC", "vsini_QuelozC")
    queloz_pleiades.rename_column("vsiniE", "vsini_QuelozE")
    queloz_pleiades.rename_column("e_vsiniC", "vsini_err_QuelozC")
    queloz_pleiades.rename_column("e_vsiniE", "vsini_err_QuelozE")
    pleiades_queloz = queloz_pleiades
    redo_col = np.ma.masked_where(
        pleiades_queloz["Star"] == 0, 
        npstr.add("HII", pleiades_queloz["Star"].astype("<U4")))
    pleiades_queloz["HII"] = redo_col
    del(pleiades_queloz["Star"])


    sh_pleiades = catin.read_Stauffer_Pleiades()[[
        "Star", "vsini", "vsini lim", "R"]]
    sh_pleiades = sh_pleiades[npstr.startswith(sh_pleiades["Star"], "HII")]
    sh_pleiades.rename_column("vsini", "vsini_SH")
    sh_pleiades.rename_column("vsini lim", "vsini_lim_SH")
    sh_pleiades["vsini_err_SH"] = sh_pleiades["vsini_SH"] / 2 / (
        1 + sh_pleiades["R"])
    lastcol = np.array([s[-1] for s in sh_pleiades["Star"]], dtype="<U1")
    sh_pleiades["SBflag_SH"] = np.ma.masked_where(
        npstr.isalpha(lastcol), lastcol)
    pleiades_sh = au.join_by_id(
        sh_pleiades, pleiades_queloz, "Star", "HII", idproc=lambda x:
        strip_index(x, string.ascii_letters, dtype=np.int_), join_type="outer")
    del(pleiades_sh["Star"])
    redo_col = np.ma.masked_where(
        pleiades_sh["HII"] == 0, 
        npstr.add("HII", pleiades_sh["HII"].astype("<U9")))
    del(pleiades_sh["HII"])
    pleiades_sh["HII"] = redo_col

    soderblom_pleiades = catin.read_Soderblom_1993b_vsini()[
        ["HII", "vsini", "vsini lim"]]
    soderblom_pleiades.rename_column("vsini", "vsini_Soderblom1")
    soderblom_pleiades.rename_column("vsini lim", "vsini_lim_Soderblom1")
    pleiades_soderblom = au.join_by_id(
        soderblom_pleiades, pleiades_sh, "HII", "HII", idproc=lambda x:
        strip_index(x, string.ascii_letters, dtype=np.int_), join_type="outer")
    redo_col = np.ma.masked_where(
        pleiades_soderblom["HII"] == 0, 
        npstr.add("HII", pleiades_soderblom["HII"].astype("<U9")))
    del(pleiades_soderblom["HII"])
    pleiades_soderblom["HII"] = redo_col


    more_soderblom = catin.read_Soderblom_1993b_additional_vsini()[
        ["HII", "vsini", "vsini lim"]]
    more_soderblom.rename_column("vsini", "vsini_Soderblom6")
    more_soderblom.rename_column("vsini lim", "vsini_lim_Soderblom6")
    more_soderblom_hii = more_soderblom[
        ~npstr.startswith(more_soderblom['HII'], "P")]
    more_soderblom_pels = more_soderblom[
        npstr.startswith(more_soderblom['HII'], "P")]
    more_soderblom_pels.rename_column("HII", "PELS")
    more_soderblom_hii_pleiades = au.join_by_id(
        more_soderblom_hii, pleiades_soderblom, "HII", "HII", 
        idproc=lambda x: strip_index(x, string.ascii_letters, dtype=np.int_), 
        join_type="outer")
    redo_col = np.ma.masked_where(
        more_soderblom_hii_pleiades["HII"] == 0, 
        npstr.add("HII", more_soderblom_hii_pleiades["HII"].astype("<U9")))
    del(more_soderblom_hii_pleiades["HII"])
    more_soderblom_hii_pleiades["HII"] = redo_col
    # Make a separate table for Pels objects.

    more_soderblom_hii_pleiades["vsini_Soderblom"] = np.ma.where(
        more_soderblom_hii_pleiades["vsini_Soderblom1"].mask,
        more_soderblom_hii_pleiades["vsini_Soderblom6"],
        more_soderblom_hii_pleiades["vsini_Soderblom1"])
    del(more_soderblom_hii_pleiades["vsini_Soderblom1"])
    del(more_soderblom_hii_pleiades["vsini_Soderblom6"])
    more_soderblom_hii_pleiades["vsini_lim_Soderblom"] = np.ma.where(
        more_soderblom_hii_pleiades["vsini_lim_Soderblom1"].mask,
        more_soderblom_hii_pleiades["vsini_lim_Soderblom6"],
        more_soderblom_hii_pleiades["vsini_lim_Soderblom1"])
    del(more_soderblom_hii_pleiades["vsini_lim_Soderblom1"])
    del(more_soderblom_hii_pleiades["vsini_lim_Soderblom6"])

    stauffer = catin.read_Stauffer_84()[["Star", "vsini", "vsini lim"]]
    stauffer = stauffer[npstr.isdigit(stauffer["Star"])]
    stauffer.rename_column("Star", "HII")
    stauffer.rename_column("vsini", "vsini_S84")
    stauffer.rename_column("vsini lim", "vsini_lim_S84")
    stauffer["vsini_err_S84"] = np.where(
        stauffer["vsini_S84"] < 50, 0.1, 0.2) * stauffer["vsini_S84"]
    pleiades_s84 = au.join_by_id(
        stauffer, more_soderblom_hii_pleiades, "HII", "HII", idproc=lambda x:
        strip_index(x, string.ascii_letters, dtype=np.int_), join_type="outer")
    redo_col = np.ma.masked_where(
        pleiades_s84["HII"] == 0, 
        npstr.add("HII", pleiades_s84["HII"].astype("<U9")))
    del(pleiades_s84["HII"])
    pleiades_s84["HII"] = redo_col

    f, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(36, 12), sharey=True)
    pleiades = pleiades_s84
    queloz_coravel_detections = npstr.isspace(pleiades["vsini_lim_QuelozC"])
    queloz_coravel_uppers = pleiades["vsini_lim_QuelozC"] == "<"
    queloz_coravel_lowers = pleiades["vsini_lim_QuelozC"] == ">"

    sh_detections = (pleiades["vsini_lim_SH"] == "d")
    sh_uppers = (pleiades["vsini_lim_SH"] == "u")

    sh_det_coravel_det = np.logical_and(
        queloz_coravel_detections, sh_detections)
    sh_det_coravel_upper = np.logical_and(
        queloz_coravel_uppers, sh_detections)
    sh_det_coravel_lower = np.logical_and(
        queloz_coravel_lowers, sh_detections)

    sh_upper_coravel_det = np.logical_and(
        queloz_coravel_detections, sh_uppers)
    sh_upper_coravel_upper = np.logical_and(
        queloz_coravel_uppers, sh_uppers)
    sh_upper_coravel_lower = np.logical_and(
        queloz_coravel_lowers, sh_uppers)

    ax2.errorbar(
        pleiades["vsini_SH"][sh_det_coravel_det], 
        pleiades["vsini_QuelozC"][sh_det_coravel_det],
        yerr=pleiades["vsini_err_QuelozC"][sh_det_coravel_det],
        xerr=pleiades["vsini_err_SH"][sh_det_coravel_det], color=bc.black,
        marker="o", ls="")
    ax2.errorbar(
        pleiades["vsini_SH"][sh_det_coravel_upper], 
        pleiades["vsini_QuelozC"][sh_det_coravel_upper], yerr=0,
        xerr=pleiades["vsini_err_SH"][sh_det_coravel_upper], color='red',
        marker="v", ls="")
    ax2.errorbar(
        pleiades["vsini_SH"][sh_det_coravel_lower], 
        pleiades["vsini_QuelozC"][sh_det_coravel_lower], yerr=0,
        xerr=pleiades["vsini_err_SH"][sh_det_coravel_lower], color='red',
        marker="^", ls="")

    ax2.errorbar(
        pleiades["vsini_SH"][sh_upper_coravel_det], 
        pleiades["vsini_QuelozC"][sh_upper_coravel_det],
        yerr=pleiades["vsini_err_QuelozC"][sh_upper_coravel_det],
        xerr=0, color='red', marker="<", ls="")
    ax2.errorbar(
        pleiades["vsini_SH"][sh_upper_coravel_upper], 
        pleiades["vsini_QuelozC"][sh_upper_coravel_upper],
        yerr=0, xerr=0, color='m', marker="x", ls="")
    ax2.errorbar(
        pleiades["vsini_SH"][sh_upper_coravel_upper], 
        pleiades["vsini_QuelozC"][sh_upper_coravel_upper],
        yerr=0, xerr=0, color='m', marker="x", ls="")
    ax2.plot([1, 100], [1, 100], 'k-')
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Stauffer & Hartmann vsini")
    ax2.set_ylabel("Queloz vsini")

    soderblom_detections = (pleiades["vsini_lim_Soderblom"] == "d")
    soderblom_uppers = (pleiades["vsini_lim_Soderblom"] == "u")

    soderblom_det_coravel_det = np.logical_and(
        queloz_coravel_detections, soderblom_detections)
    soderblom_det_coravel_upper = np.logical_and(
        queloz_coravel_uppers, soderblom_detections)
    soderblom_det_coravel_lower = np.logical_and(
        queloz_coravel_lowers, soderblom_detections)

    soderblom_upper_coravel_det = np.logical_and(
        queloz_coravel_detections, soderblom_uppers)
    soderblom_upper_coravel_upper = np.logical_and(
        queloz_coravel_uppers, soderblom_uppers)
    soderblom_upper_coravel_lower = np.logical_and(
        queloz_coravel_lowers, soderblom_uppers)

    ax3.errorbar(
        pleiades["vsini_Soderblom"][soderblom_det_coravel_det], 
        pleiades["vsini_QuelozC"][soderblom_det_coravel_det],
        yerr=pleiades["vsini_err_QuelozC"][soderblom_det_coravel_det],
        xerr=0, color=bc.black, marker="o", ls="")
    ax3.errorbar(
        pleiades["vsini_Soderblom"][soderblom_det_coravel_upper], 
        pleiades["vsini_QuelozC"][soderblom_det_coravel_upper], yerr=0,
        xerr=0, color='red', marker="v", ls="")
    ax3.errorbar(
        pleiades["vsini_Soderblom"][soderblom_det_coravel_lower], 
        pleiades["vsini_QuelozC"][soderblom_det_coravel_lower], yerr=0,
        xerr=0, color='red', marker="^", ls="")

    ax3.errorbar(
        pleiades["vsini_Soderblom"][soderblom_upper_coravel_det], 
        pleiades["vsini_QuelozC"][soderblom_upper_coravel_det],
        yerr=pleiades["vsini_err_QuelozC"][soderblom_upper_coravel_det],
        xerr=0, color='red', marker="<", ls="")
    ax3.errorbar(
        pleiades["vsini_Soderblom"][soderblom_upper_coravel_upper], 
        pleiades["vsini_QuelozC"][soderblom_upper_coravel_upper],
        yerr=0, xerr=0, color='m', marker="x", ls="")
    ax3.errorbar(
        pleiades["vsini_Soderblom"][soderblom_upper_coravel_upper], 
        pleiades["vsini_QuelozC"][soderblom_upper_coravel_upper],
        yerr=0, xerr=0, color='m', marker="x", ls="")
    ax3.plot([1, 100], [1, 100], 'k-')
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Soderblom vsini")

    s84_detections = (pleiades["vsini_lim_SH"] == "d")
    s84_uppers = (pleiades["vsini_lim_SH"] == "u")

    s84_det_coravel_det = np.logical_and(
        queloz_coravel_detections, s84_detections)
    s84_det_coravel_upper = np.logical_and(
        queloz_coravel_uppers, s84_detections)
    s84_det_coravel_lower = np.logical_and(
        queloz_coravel_lowers, s84_detections)

    s84_upper_coravel_det = np.logical_and(
        queloz_coravel_detections, s84_uppers)
    s84_upper_coravel_upper = np.logical_and(
        queloz_coravel_uppers, s84_uppers)
    s84_upper_coravel_lower = np.logical_and(
        queloz_coravel_lowers, s84_uppers)

    ax4.errorbar(
        pleiades["vsini_S84"][s84_det_coravel_det], 
        pleiades["vsini_QuelozC"][s84_det_coravel_det],
        yerr=pleiades["vsini_err_QuelozC"][s84_det_coravel_det],
        xerr=pleiades["vsini_err_S84"][s84_det_coravel_det], color=bc.black,
        marker="o", ls="")
    ax4.errorbar(
        pleiades["vsini_S84"][s84_det_coravel_upper], 
        pleiades["vsini_QuelozC"][s84_det_coravel_upper], yerr=0,
        xerr=pleiades["vsini_err_S84"][s84_det_coravel_upper], color='red',
        marker="v", ls="")
    ax4.errorbar(
        pleiades["vsini_S84"][s84_det_coravel_lower], 
        pleiades["vsini_QuelozC"][s84_det_coravel_lower], yerr=0,
        xerr=pleiades["vsini_err_S84"][s84_det_coravel_lower], color='red',
        marker="^", ls="")

    ax4.errorbar(
        pleiades["vsini_S84"][s84_upper_coravel_det], 
        pleiades["vsini_QuelozC"][s84_upper_coravel_det],
        yerr=pleiades["vsini_err_QuelozC"][s84_upper_coravel_det],
        xerr=0, color='red', marker="<", ls="")
    ax4.errorbar(
        pleiades["vsini_S84"][s84_upper_coravel_upper], 
        pleiades["vsini_QuelozC"][s84_upper_coravel_upper],
        yerr=0, xerr=0, color='m', marker="x", ls="")
    ax4.errorbar(
        pleiades["vsini_S84"][s84_upper_coravel_upper], 
        pleiades["vsini_QuelozC"][s84_upper_coravel_upper],
        yerr=0, xerr=0, color='m', marker="x", ls="")
    ax4.plot([1, 100], [1, 100], 'k-')
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("Stauffer (1984) vsini")

def vsini_comparison_chisq():
    '''Compare the ASPCAP vsini to literature and correlate with chi-square.'''
    full_pleiades = cache.pleiades_APOGEE_Literature_vsini()
    pleiades_group = full_pleiades.group_by("APOGEE_ID")
    apogee_avg = pleiades_group[[
        "APOGEE_ID", "VSINI"]].groups.aggregate(np.mean)
    unique_pleiades = unique(full_pleiades, keys="APOGEE_ID")
    unique_pleiades.remove_column("VSINI")
    pleiades = pleiades_group[[
        "APOGEE_ID", "VSINI", "(V-K)0", "Delmag", "vsini_QuelozE", 
        "vsini_err_QuelozE", "vsini_QuelozC", "vsini_lim_QuelozC", 
        "vsini_err_QuelozC", "vsini_Terndrup", "vsini_err_Terndrup",
        "vsini_lim_Terndrup", "vsini_Soderblom", "vsini_lim_Soderblom",
        "vsini_SH", "vsini_err_SH", "vsini_lim_SH", "vsini_S84",
        "vsini_err_S84", "vsini_lim_S84"]].groups.aggregate(np.mean)
    pleiades = au.join_by_id(
        unique_pleiades, apogee_avg, "APOGEE_ID", "APOGEE_ID")
    phot_bin = np.logical_or(
        np.logical_and(pleiades["(V-K)0"] <= 5, pleiades["Delmag"] >= 0.3),
        np.logical_and(pleiades["(V-K)0"] > 5, pleiades["Delmag"] >= 0.3))
    apogee_dets = pleiades["VSINI"] > 10

    # I only have the energy to do one figure...
    f, ax1 = plt.subplots(
        1, 1, figsize=figsize, sharey=True)
    
    # Plot the Queloz targets in the first panel.
    queloz_elodie = ~pleiades["vsini_QuelozE"].mask
    queloz_coravel = ~pleiades["vsini_QuelozC"].mask
    full_queloz = np.logical_or(queloz_elodie, queloz_coravel)
    queloz_coravel_upper = pleiades["vsini_lim_QuelozC"] == "<"
    queloz_coravel_lower = pleiades["vsini_lim_QuelozC"] == ">"
    queloz_coravel_det = np.logical_not(
        np.logical_or(queloz_coravel_upper,queloz_coravel_lower))

    queloz_elodie_photbin = np.logical_and(queloz_elodie, phot_bin)
    queloz_elodie_single = np.logical_and(queloz_elodie, ~phot_bin)
    queloz_coravel_photbin = np.logical_and(queloz_coravel_det, phot_bin)
    queloz_coravel_single = np.logical_and(queloz_coravel_det, ~phot_bin)
    queloz_coravel_upper_photbin = np.logical_and(
        queloz_coravel_upper, phot_bin)
    queloz_coravel_upper_single = np.logical_and(
        queloz_coravel_upper, ~phot_bin)
    queloz_coravel_lower_photbin = np.logical_and(
        queloz_coravel_lower, phot_bin)
    queloz_coravel_lower_single = np.logical_and(
        queloz_coravel_lower, ~phot_bin)

    # Split the Queloz detections into APOGEE detections and nondetections.
    queloz_elodie_dets_apogee = np.logical_and(
        queloz_elodie_single, apogee_dets)
    queloz_elodie_nondets_apogee = np.logical_and(
        queloz_elodie_single, ~apogee_dets)
    queloz_coravel_dets_apogee = np.logical_and(
        queloz_coravel_single, apogee_dets)
    queloz_coravel_nondets_apogee = np.logical_and(
        queloz_coravel_single, ~apogee_dets)
    # Split the queloz photometric binaries into APOGEE detections and nondetections.
    queloz_elodie_dets_photbins = np.logical_and(
        queloz_elodie_photbin, apogee_dets)
    queloz_elodie_nondets_photbins = np.logical_and(
        queloz_elodie_photbin, ~apogee_dets)
    queloz_coravel_dets_photbins = np.logical_and(
        queloz_coravel_photbin, apogee_dets)
    queloz_coravel_nondets_photbins = np.logical_and(
        queloz_coravel_photbin, ~apogee_dets)
    # Split the queloz upper limits into APOGEE detections and nondetections.
    queloz_coravel_dets_upper_single = np.logical_and(
        queloz_coravel_upper_single, apogee_dets)
    queloz_coravel_nondets_upper_single = np.logical_and(
        queloz_coravel_upper_single, ~apogee_dets)
    queloz_coravel_dets_lower_single = np.logical_and(
        queloz_coravel_lower_single, apogee_dets)
    queloz_coravel_nondets_lower_single = np.logical_and(
        queloz_coravel_lower_single, ~apogee_dets)
    queloz_coravel_dets_upper_photbin = np.logical_and(
        queloz_coravel_upper_photbin, apogee_dets)
    queloz_coravel_nondets_upper_photbin = np.logical_and(
        queloz_coravel_upper_photbin, ~apogee_dets)
    queloz_coravel_dets_lower_photbin = np.logical_and(
        queloz_coravel_lower_photbin, apogee_dets)
    queloz_coravel_nondets_lower_photbin = np.logical_and(
        queloz_coravel_lower_photbin, ~apogee_dets)

    chi2colors = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=max(pleiades["ASPCAP_CHI2"]))
    colors = norm(chi2colors(pleiades["ASPCAP_CHI2"]))

    sc = ax1.scatter(
        pleiades["vsini_QuelozE"][queloz_elodie_dets_apogee],
        pleiades["VSINI"][queloz_elodie_dets_apogee], 
        c=pleiades["ASPCAP_CHI2"][queloz_elodie_dets_apogee], marker="o", 
        cmap=chi2colors, norm=norm)
    ax1.errorbar(
        pleiades["vsini_QuelozE"][queloz_elodie_dets_apogee],
        pleiades["VSINI"][queloz_elodie_dets_apogee], 
        yerr=pleiades["VSINI_ERR"][queloz_elodie_dets_apogee], 
        xerr=pleiades["vsini_err_QuelozE"][queloz_elodie_dets_apogee], 
        marker="", color=bc.red, ls="")
    sc = ax1.scatter(
        pleiades["vsini_QuelozE"][queloz_elodie_nondets_apogee],
        pleiades["VSINI"][queloz_elodie_nondets_apogee], 
        c=pleiades["ASPCAP_CHI2"][queloz_elodie_nondets_apogee], marker="o", 
        cmap=chi2colors, norm=norm)
    ax1.errorbar(
        pleiades["vsini_QuelozE"][queloz_elodie_nondets_apogee],
        pleiades["VSINI"][queloz_elodie_nondets_apogee], 
        yerr=pleiades["VSINI_ERR"][queloz_elodie_nondets_apogee],
        xerr=pleiades["vsini_err_QuelozE"][queloz_elodie_nondets_apogee], 
        marker="", color="grey", ls="", alpha=0.3)
    sc = ax1.scatter(
        pleiades["vsini_QuelozE"][queloz_elodie_dets_photbins],
        pleiades["VSINI"][queloz_elodie_dets_photbins], 
        c=pleiades["ASPCAP_CHI2"][queloz_elodie_dets_photbins], marker="8", 
        cmap=chi2colors, norm=norm)
    ax1.errorbar(
        pleiades["vsini_QuelozE"][queloz_elodie_dets_photbins],
        pleiades["VSINI"][queloz_elodie_dets_photbins], 
        yerr=pleiades["VSINI_ERR"][queloz_elodie_dets_photbins], 
        xerr=pleiades["vsini_err_QuelozE"][queloz_elodie_dets_photbins], 
        marker="", color="red", ls="")
    sc = ax1.scatter(
        pleiades["vsini_QuelozE"][queloz_elodie_nondets_photbins],
        pleiades["VSINI"][queloz_elodie_nondets_photbins], 
        c=pleiades["ASPCAP_CHI2"][queloz_elodie_nondets_photbins], marker="8", 
        cmap=chi2colors, norm=norm)
    ax1.errorbar(
        pleiades["vsini_QuelozE"][queloz_elodie_nondets_photbins],
        pleiades["VSINI"][queloz_elodie_nondets_photbins], 
        yerr=pleiades["VSINI_ERR"][queloz_elodie_nondets_photbins], 
        xerr=pleiades["vsini_err_QuelozE"][queloz_elodie_nondets_photbins], 
        marker="8", color="grey", ls="", alpha=0.3)
    sc = ax1.scatter(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_apogee],
        pleiades["VSINI"][queloz_coravel_dets_apogee], 
        c=pleiades["ASPCAP_CHI2"][queloz_coravel_dets_apogee], marker="o", 
        cmap=chi2colors, norm=norm)
    ax1.errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_apogee], 
        pleiades["VSINI"][queloz_coravel_dets_apogee], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_dets_apogee], 
        xerr=pleiades["vsini_err_QuelozC"][queloz_coravel_dets_apogee], 
        marker="", color=bc.red, ls="")
    sc = ax1.scatter(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_photbins],
        pleiades["VSINI"][queloz_coravel_dets_photbins], 
        c=pleiades["ASPCAP_CHI2"][queloz_coravel_dets_photbins], marker="8", 
        cmap=chi2colors, norm=norm)
    ax1.errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_photbins],
        pleiades["VSINI"][queloz_coravel_dets_photbins], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_dets_photbins], 
        xerr=pleiades["vsini_err_QuelozC"][queloz_coravel_dets_photbins], 
        marker="", color="red", ls="")
    sc = ax1.scatter(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_photbins],
        pleiades["VSINI"][queloz_coravel_nondets_photbins], 
        c=pleiades["ASPCAP_CHI2"][queloz_coravel_nondets_photbins], marker="8", 
        cmap=chi2colors, norm=norm)
    ax1.errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_photbins],
        pleiades["VSINI"][queloz_coravel_nondets_photbins], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_nondets_photbins], 
        xerr=pleiades["vsini_err_QuelozC"][queloz_coravel_nondets_photbins], 
        marker="", color="red", ls="", alpha=0.3)
    sc = ax1.scatter(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_upper_single],
        pleiades["VSINI"][queloz_coravel_dets_upper_single], 
        c=pleiades["ASPCAP_CHI2"][queloz_coravel_dets_upper_single], marker="<", 
        cmap=chi2colors, norm=norm)
    ax1.errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_upper_single], 
        pleiades["VSINI"][queloz_coravel_dets_upper_single], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_dets_upper_single], xerr=0,
        marker="", color=bc.red, ls="")
    sc = ax1.scatter(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_upper_single],
        pleiades["VSINI"][queloz_coravel_nondets_upper_single], 
        c=pleiades["ASPCAP_CHI2"][queloz_coravel_nondets_upper_single], 
        marker="<", cmap=chi2colors, norm=norm)
    ax1.errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_upper_single], 
        pleiades["VSINI"][queloz_coravel_nondets_upper_single], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_nondets_upper_single], xerr=0,
        marker="", color="grey", ls="", alpha=0.3)
    sc = ax1.scatter(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_lower_single],
        pleiades["VSINI"][queloz_coravel_dets_lower_single], 
        c=pleiades["ASPCAP_CHI2"][queloz_coravel_dets_lower_single], marker=">", 
        cmap=chi2colors, norm=norm)
    ax1.errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_lower_single], 
        pleiades["VSINI"][queloz_coravel_dets_lower_single], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_dets_lower_single], 
        xerr=pleiades["vsini_err_QuelozC"][queloz_coravel_dets_lower_single], 
        marker="", color=bc.red, ls="")
    sc = ax1.scatter(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_lower_single],
        pleiades["VSINI"][queloz_coravel_nondets_lower_single], 
        c=pleiades["ASPCAP_CHI2"][queloz_coravel_nondets_lower_single], 
        marker=">", cmap=chi2colors, norm=norm)
    ax1.errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_lower_single], 
        pleiades["VSINI"][queloz_coravel_nondets_lower_single], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_nondets_lower_single], 
        xerr=pleiades["vsini_err_QuelozC"][queloz_coravel_nondets_lower_single], 
        marker="", color="grey", ls="", alpha=0.3)
    sc = ax1.scatter(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_upper_photbin],
        pleiades["VSINI"][queloz_coravel_dets_upper_photbin], 
        c=pleiades["ASPCAP_CHI2"][queloz_coravel_dets_upper_photbin], marker="<", 
        cmap=chi2colors, norm=norm)
    ax1.errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_upper_photbin], 
        pleiades["VSINI"][queloz_coravel_dets_upper_photbin], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_dets_upper_photbin],  xerr=0, 
        marker="", color='red', ls="")
    sc = ax1.scatter(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_upper_photbin],
        pleiades["VSINI"][queloz_coravel_nondets_upper_photbin], 
        c=pleiades["ASPCAP_CHI2"][queloz_coravel_nondets_upper_photbin], 
        marker="<", cmap=chi2colors, norm=norm)
    ax1.errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_upper_photbin], 
        pleiades["VSINI"][queloz_coravel_nondets_upper_photbin], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_nondets_upper_photbin], xerr=0,
        marker="", color="red", ls="", alpha=0.3)
    sc = ax1.scatter(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_lower_photbin],
        pleiades["VSINI"][queloz_coravel_dets_lower_photbin], 
        c=pleiades["ASPCAP_CHI2"][queloz_coravel_dets_lower_photbin], marker=">", 
        cmap=chi2colors, norm=norm)
    ax1.errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_lower_photbin], 
        pleiades["VSINI"][queloz_coravel_dets_lower_photbin], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_dets_lower_photbin],  xerr=0,
        marker="", color='red', ls="")
    sc = ax1.scatter(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_lower_photbin],
        pleiades["VSINI"][queloz_coravel_nondets_lower_photbin], 
        c=pleiades["ASPCAP_CHI2"][queloz_coravel_nondets_lower_photbin], 
        marker=">", cmap=chi2colors, norm=norm)
    ax1.errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_lower_photbin], 
        pleiades["VSINI"][queloz_coravel_nondets_lower_photbin], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_nondets_lower_photbin],  xerr=0,
        marker="", color="red", ls="", alpha=0.3)
    ax1.plot([1, 100], [1, 100], 'k-')

    f.colorbar(sc, ax=ax1)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(1, 100)
    ax1.set_ylim(1, 100)
    ax1.set_xlabel("Queloz Vsini")
    ax1.set_ylabel("APOGEE Vsini")
    ax1.set_title("Queloz measurements")

@write_plot("Delmag_Stauffer")
def Pleiades_delmag_comparison():
    '''Compare the photometric binary test used in Stauffer et al (2017).'''
    full_pleiades = cache.pleiades_APOGEE_Literature_vsini()
    pleiades_group = full_pleiades.group_by("APOGEE_ID")
    apogee_avg = pleiades_group[[
        "APOGEE_ID", "VSINI"]].groups.aggregate(np.mean)
    unique_pleiades = unique(full_pleiades, keys="APOGEE_ID")
    unique_pleiades.remove_column("VSINI")
    pleiades = pleiades_group[[
        "APOGEE_ID", "VSINI", "(V-K)0", "Delmag", "vsini_QuelozE", 
        "vsini_err_QuelozE", "vsini_QuelozC", "vsini_lim_QuelozC", 
        "vsini_err_QuelozC", "vsini_Terndrup", "vsini_err_Terndrup",
        "vsini_lim_Terndrup", "vsini_Soderblom", "vsini_lim_Soderblom",
        "vsini_SH", "vsini_err_SH", "vsini_lim_SH", "vsini_S84",
        "vsini_err_S84", "vsini_lim_S84", "vsini_Jackson", "vsini_err_Jackson",
    "vsini_lim_Jackson"]].groups.aggregate(np.mean)
    pleiades = au.join_by_id(
        unique_pleiades, apogee_avg, "APOGEE_ID", "APOGEE_ID")
    photbin = np.logical_or(
        np.logical_and(pleiades["(V-K)0"] <= 5, pleiades["Delmag"] >= 0.3),
        np.logical_and(pleiades["(V-K)0"] > 5, pleiades["Delmag"] >= 0.3))

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    ax1.plot(
        pleiades["(V-K)_ST"][~photbin], pleiades["Delmag"][~photbin], 
        color=bc.red, ls="", marker="o")
    ax2.plot(
        pleiades["(V-K)_ST"][~photbin], pleiades["Dereddened V"][~photbin], 
        color=bc.red, ls="", marker="o")
    ax3.plot(
        pleiades["(V-K)_ST"][~photbin], pleiades["Dereddened K"][~photbin], 
        color=bc.red, ls="", marker="o")
    ax1.plot(
        pleiades["(V-K)_ST"][photbin], pleiades["Delmag"][photbin], 
        color='r', ls="", marker="o")
    ax2.plot(
        pleiades["(V-K)_ST"][photbin], pleiades["Dereddened V"][photbin], 
        color='r', ls="", marker="o")
    ax3.plot(
        pleiades["(V-K)_ST"][photbin], pleiades["Dereddened K"][photbin], 
        color='r', ls="", marker="o")

    hr.invert_y_axis(ax2)
    hr.invert_y_axis(ax3)
    ax1.set_xlim(1, 5)
    ax2.set_xlim(1, 5)
    ax3.set_xlim(1, 5)
    ax2.set_xlabel("(V-K)0")
    ax1.set_ylabel("Delta V")
    ax2.set_ylabel("Dereddened V")
    ax3.set_ylabel("Dereddened K")

def teff_color_comparison():
    '''Compare the APOGEE Teff to (V-K) color for Pleiades targets.'''
    full_pleiades = cache.pleiades_APOGEE_Literature_vsini()
    pleiades_group = full_pleiades.group_by("APOGEE_ID")
    apogee_avg = pleiades_group[[
        "APOGEE_ID", "VSINI"]].groups.aggregate(np.mean)
    unique_pleiades = unique(full_pleiades, keys="APOGEE_ID")
    unique_pleiades.remove_column("VSINI")
    pleiades = pleiades_group[[
        "APOGEE_ID", "VSINI", "(V-K)0", "Delmag", "vsini_QuelozE", 
        "vsini_err_QuelozE", "vsini_QuelozC", "vsini_lim_QuelozC", 
        "vsini_err_QuelozC", "vsini_Terndrup", "vsini_err_Terndrup",
        "vsini_lim_Terndrup", "vsini_Soderblom", "vsini_lim_Soderblom",
        "vsini_SH", "vsini_err_SH", "vsini_lim_SH", "vsini_S84",
        "vsini_err_S84", "vsini_lim_S84", "vsini_Jackson", "vsini_err_Jackson",
    "vsini_lim_Jackson"]].groups.aggregate(np.mean)
    pleiades = au.join_by_id(
        unique_pleiades, apogee_avg, "APOGEE_ID", "APOGEE_ID")
    photbin = np.logical_or(
        np.logical_and(pleiades["(V-K)0"] <= 5, pleiades["Delmag"] >= 0.3),
        np.logical_and(pleiades["(V-K)0"] > 5, pleiades["Delmag"] >= 0.3))

    f, ax1 = plt.subplots(1, 1, figsize=figsize)

    ax1.plot(
        pleiades["TEFF"][~photbin], pleiades["(V-K)_ST"][~photbin], 
        color=bc.red, ls="", marker="o")
    ax1.plot(
        pleiades["TEFF"][photbin], pleiades["(V-K)_ST"][photbin], 
        color='r', ls="", marker="o")

    ax1.set_ylim(1, 5)
    ax1.set_xlim(6750, 3500)
    ax1.set_ylabel("Dereddened V-K", fontsize=40)
    ax1.set_xlabel("APOGEE Teff", fontsize=40)

@write_plot("Pleiades_HR")
def Pleiades_HR_Diagram():
    '''Plot the Pleiades targets on an HR Diagram.'''
    full_pleiades = cache.pleiades_APOGEE_Literature_vsini()
    pleiades_group = full_pleiades.group_by("APOGEE_ID")
    apogee_avg = pleiades_group[[
        "APOGEE_ID", "VSINI"]].groups.aggregate(np.mean)
    unique_pleiades = unique(full_pleiades, keys="APOGEE_ID")
    unique_pleiades.remove_column("VSINI")
    pleiades = pleiades_group[[
        "APOGEE_ID", "VSINI", "(V-K)0", "Delmag", "vsini_QuelozE", 
        "vsini_err_QuelozE", "vsini_QuelozC", "vsini_lim_QuelozC", 
        "vsini_err_QuelozC", "vsini_Terndrup", "vsini_err_Terndrup",
        "vsini_lim_Terndrup", "vsini_Soderblom", "vsini_lim_Soderblom",
        "vsini_SH", "vsini_err_SH", "vsini_lim_SH", "vsini_S84",
        "vsini_err_S84", "vsini_lim_S84", "vsini_Jackson", "vsini_err_Jackson",
    "vsini_lim_Jackson"]].groups.aggregate(np.mean)
    pleiades = au.join_by_id(
        unique_pleiades, apogee_avg, "APOGEE_ID", "APOGEE_ID")
    phot_bin = np.logical_or(
        np.logical_and(pleiades["(V-K)0"] <= 5, pleiades["Delmag"] >= 0.3),
        np.logical_and(pleiades["(V-K)0"] > 5, pleiades["Delmag"] >= 0.3))

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Plot the Queloz targets in the first panel.
    queloz_elodie = ~pleiades["vsini_QuelozE"].mask
    queloz_coravel = ~pleiades["vsini_QuelozC"].mask
    full_queloz = np.logical_or(queloz_elodie, queloz_coravel)
    queloz_coravel_upper = pleiades["vsini_lim_QuelozC"] == "<"
    queloz_coravel_lower = pleiades["vsini_lim_QuelozC"] == ">"
    queloz_coravel_det = np.logical_not(
        np.logical_or(queloz_coravel_upper,queloz_coravel_lower))

    queloz_elodie_photbin = np.logical_and(queloz_elodie, phot_bin)
    queloz_elodie_single = np.logical_and(queloz_elodie, ~phot_bin)
    queloz_coravel_photbin = np.logical_and(queloz_coravel_det, phot_bin)
    queloz_coravel_single = np.logical_and(queloz_coravel_det, ~phot_bin)
    queloz_coravel_upper_photbin = np.logical_and(
        queloz_coravel_upper, phot_bin)
    queloz_coravel_upper_single = np.logical_and(
        queloz_coravel_upper, ~phot_bin)
    queloz_coravel_lower_photbin = np.logical_and(
        queloz_coravel_lower, phot_bin)
    queloz_coravel_lower_single = np.logical_and(
        queloz_coravel_lower, ~phot_bin)
    queloz_coravel_all_photbin = au.multi_logical_or(
        queloz_coravel_photbin, queloz_coravel_lower_photbin,
        queloz_coravel_upper_photbin)
    queloz_coravel_all_single = au.multi_logical_or(
        queloz_coravel_single, queloz_coravel_lower_single,
        queloz_coravel_upper_single)

    hr.absmag_teff_plot(
        pleiades["TEFF"][queloz_elodie_photbin],
        pleiades["Dereddened K"][queloz_elodie_photbin],
#       yerr=pleiades["K_ERR"][queloz_elodie_photbin],
#       xerr=pleiades["TEFF_ERR"][queloz_elodie_photbin], 
        marker="o", color=bc.green, ls="", axis=ax, label="Photometric Binaries")
    hr.absmag_teff_plot(
        pleiades["TEFF"][queloz_elodie_single],
        pleiades["Dereddened K"][queloz_elodie_single],
#       yerr=pleiades["K_ERR"][queloz_elodie_single],
#       xerr=pleiades["TEFF_ERR"][queloz_elodie_single], 
        marker="o", color=bc.violet, ls="", axis=ax, label="Queloz et al (1998)")
    hr.absmag_teff_plot(
        pleiades["TEFF"][queloz_coravel_all_single],
        pleiades["Dereddened K"][queloz_coravel_all_single],
#       yerr=pleiades["K_ERR"][queloz_coravel_all_single],
#       xerr=pleiades["TEFF_ERR"][queloz_coravel_all_single], 
        marker="o", color=bc.violet, ls="", axis=ax, label="")
    hr.absmag_teff_plot(
        pleiades["TEFF"][queloz_coravel_all_photbin],
        pleiades["Dereddened K"][queloz_coravel_all_photbin],
#       yerr=pleiades["K_ERR"][queloz_coravel_all_photbin],
#       xerr=pleiades["TEFF_ERR"][queloz_coravel_all_photbin], 
        marker="o", color=bc.green, ls="", axis=ax, label="")


    # Plot Terndrup targets in the second panel
    terndrup = ~pleiades["vsini_Terndrup"].mask
    terndrup_detections = pleiades["vsini_lim_Terndrup"] == "d"
    terndrup_limits = pleiades["vsini_lim_Terndrup"] == "u"

    terndrup_photbin = np.logical_and(terndrup_detections, phot_bin)
    terndrup_singles = np.logical_and(terndrup_detections, ~phot_bin)
    terndrup_lim_photbin = np.logical_and(terndrup_limits, phot_bin)
    terndrup_lim_singles = np.logical_and(terndrup_limits, ~phot_bin)

    terndrup_all_photbin = au.multi_logical_or(
        terndrup_photbin, terndrup_lim_photbin)
    terndrup_all_singles = au.multi_logical_or(
        terndrup_singles, terndrup_lim_singles)

    # Now plot the Terndrup Points
    hr.absmag_teff_plot(
        pleiades["TEFF"][terndrup_all_singles],
        pleiades["Dereddened K"][terndrup_all_singles],
#       yerr=pleiades["K_ERR"][terndrup_all_singles],
#       xerr=pleiades["TEFF_ERR"][terndrup_all_singles], 
        marker="s", color='blue', ls="", axis=ax, label="Terndrup et al (2000)")
    hr.absmag_teff_plot(
        pleiades["TEFF"][terndrup_all_photbin],
        pleiades["Dereddened K"][terndrup_all_photbin],
#       yerr=pleiades["K_ERR"][terndrup_all_photbin],
#       xerr=pleiades["TEFF_ERR"][terndrup_all_photbin], 
        marker="s", color=bc.green, ls="", axis=ax, label="")

    # Now plot Soderblom points
    soderblom = ~pleiades["vsini_Soderblom"].mask
    soderblom_detections = pleiades["vsini_lim_Soderblom"] == "d"
    soderblom_limits = pleiades["vsini_lim_Soderblom"] == "u"

    soderblom_det_photbins = np.logical_and(soderblom_detections, phot_bin)
    soderblom_det_singles = np.logical_and(soderblom_detections, ~phot_bin)
    soderblom_lim_photbins = np.logical_and(soderblom_limits, phot_bin)
    soderblom_lim_singles = np.logical_and(soderblom_limits, ~phot_bin)

    soderblom_all_photbins = au.multi_logical_or(
        soderblom_det_photbins, soderblom_lim_photbins)
    soderblom_all_singles = au.multi_logical_or(
        soderblom_det_singles, soderblom_lim_singles)

    # Now plot the Soderblom Points
    hr.absmag_teff_plot(
        pleiades["TEFF"][soderblom_all_singles],
        pleiades["Dereddened K"][soderblom_all_singles],
#       yerr=pleiades["K_ERR"][soderblom_all_singles],
#       xerr=pleiades["TEFF_ERR"][soderblom_all_singles], 
        marker="p", color=bc.red, ls="", axis=ax, label="Soderblom et al (1993)")
    hr.absmag_teff_plot(
        pleiades["TEFF"][soderblom_all_photbins],
        pleiades["Dereddened K"][soderblom_all_photbins],
#       yerr=pleiades["K_ERR"][soderblom_all_photbins],
#       xerr=pleiades["TEFF_ERR"][soderblom_all_photbins], 
        marker="p", color=bc.green, ls="", axis=ax, label="")



    # Now plot Stauffer & Hartmann points.
    sh = ~pleiades["vsini_SH"].mask
    sh_detections = pleiades["vsini_lim_SH"] == "d"
    sh_upper = pleiades["vsini_lim_SH"] == "u"

    sh_det_singles = np.logical_and(sh_detections, ~phot_bin)
    sh_det_photbin = np.logical_and(sh_detections, phot_bin)
    sh_lim_singles = np.logical_and(sh_upper, ~phot_bin)
    sh_lim_photbin = np.logical_and(sh_upper, phot_bin)
    
    sh_all_photbin = au.multi_logical_or(
        sh_det_photbin, sh_lim_photbin)
    sh_all_singles = au.multi_logical_or(
        sh_det_singles, sh_lim_singles)

    # Now plot the Stauffer & Hartmann Points
    hr.absmag_teff_plot(
        pleiades["TEFF"][sh_all_singles],
        pleiades["Dereddened K"][sh_all_singles],
#       yerr=pleiades["K_ERR"][sh_all_singles],
#       xerr=pleiades["TEFF_ERR"][sh_all_singles], 
        marker="*", color=bc.light_pink, ls="", axis=ax, 
        label="Stauffer & Hartmann (1987)", ms=10)
    hr.absmag_teff_plot(
        pleiades["TEFF"][sh_all_photbin],
        pleiades["Dereddened K"][sh_all_photbin],
#       yerr=pleiades["K_ERR"][sh_all_photbin],
#       xerr=pleiades["TEFF_ERR"][sh_all_photbin], 
        marker="*", color=bc.green, ls="", axis=ax, label="", ms=10)


    # Lastly Stauffer points.
    s84 = ~pleiades["vsini_S84"].mask
    s84_detections = pleiades["vsini_lim_S84"] == "d"
    s84_upper = pleiades["vsini_lim_S84"] == "u"

    s84_det_singles = np.logical_and(s84_detections, ~phot_bin)
    s84_det_photbin = np.logical_and(s84_detections, phot_bin)
    s84_lim_singles = np.logical_and(s84_upper, ~phot_bin)
    s84_lim_photbin = np.logical_and(s84_upper, phot_bin)

    s84_all_photbin = au.multi_logical_or(
        s84_det_photbin, s84_lim_photbin)
    s84_all_singles = au.multi_logical_or(
        s84_det_singles, s84_lim_singles)

    # Now plot the Stauffer & Hartmann Points
    hr.absmag_teff_plot(
        pleiades["TEFF"][s84_all_singles],
        pleiades["Dereddened K"][s84_all_singles],
#       yerr=pleiades["K_ERR"][s84_all_singles],
#       xerr=pleiades["TEFF_ERR"][s84_all_singles], 
        marker="X", color=bc.algae, ls="", axis=ax, label="Stauffer et al (1984)")
    hr.absmag_teff_plot(
        pleiades["TEFF"][s84_all_photbin],
        pleiades["Dereddened K"][s84_all_photbin],
#       yerr=pleiades["K_ERR"][s84_all_photbin],
#       xerr=pleiades["TEFF_ERR"][s84_all_photbin], 
        marker="X", color=bc.green, ls="", axis=ax, label="")

    # Another round of points from Jackson and Jeffries
    jackson = ~pleiades["vsini_Jackson"].mask
    jackson_detections = pleiades["vsini_lim_Jackson"] == "d"
    jackson_upper = pleiades["vsini_lim_Jackson"] == "u"

    jackson_det_singles = np.logical_and(jackson_detections, ~phot_bin)
    jackson_det_photbin = np.logical_and(jackson_detections, phot_bin)
    jackson_lim_singles = np.logical_and(jackson_upper, ~phot_bin)
    jackson_lim_photbin = np.logical_and(jackson_upper, phot_bin)

    jackson_all_photbin = au.multi_logical_or(
        jackson_det_photbin, jackson_lim_photbin)
    jackson_all_singles = au.multi_logical_or(
        jackson_det_singles, jackson_lim_singles)

    # Now plot the Jackson and Jeffries Points
    hr.absmag_teff_plot(
        pleiades["TEFF"][jackson_all_singles],
        pleiades["Dereddened K"][jackson_all_singles],
#       yerr=pleiades["K_ERR"][jackson_all_singles],
#       xerr=pleiades["TEFF_ERR"][jackson_all_singles], 
        marker="d", color=bc.purple, ls="", axis=ax, label="Jackson et al (2018)")
    hr.absmag_teff_plot(
        pleiades["TEFF"][jackson_all_photbin],
        pleiades["Dereddened K"][jackson_all_photbin],
#       yerr=pleiades["K_ERR"][jackson_all_photbin],
#       xerr=pleiades["TEFF_ERR"][jackson_all_photbin], 
        marker="d", color=bc.green, ls="", axis=ax, label="")


    fulls = au.multi_logical_or(
        queloz_elodie_single, queloz_elodie_photbin, queloz_coravel_all_single,
        queloz_coravel_all_photbin, terndrup_all_singles, terndrup_all_photbin,
        soderblom_all_singles, soderblom_all_photbins, sh_all_singles,
        sh_all_photbin, s84_all_singles, s84_all_photbin, jackson_all_singles,
        jackson_all_photbin)


    median_teff_err = np.median(pleiades["TEFF_ERR"][fulls])
    median_k_err = np.median(pleiades["K_ERR"][fulls])
    
    hr.absmag_teff_plot(
        [4000], [7.5], yerr=[median_k_err], xerr=[median_teff_err], marker="", 
        color=bc.black, elinewidth=3)
    ax.set_xlim(6750, 3500)
    ax.set_ylim(12, 7)
    ax.set_ylabel("Dereddened $K_S$")
    ax.set_xlabel(Teffstr + " (K)")
    ax.legend(loc="lower left", fontsize=17)

def Pleiades_vsini_comparisons():
    '''Compare APOGEE and literature vsini for different surveys.'''
    full_pleiades = cache.pleiades_APOGEE_Literature_vsini()
    pleiades_group = full_pleiades.group_by("APOGEE_ID")
    apogee_avg = pleiades_group[[
        "APOGEE_ID", "VSINI"]].groups.aggregate(np.mean)
    unique_pleiades = unique(full_pleiades, keys="APOGEE_ID")
    unique_pleiades.remove_column("VSINI")
    pleiades = pleiades_group[[
        "APOGEE_ID", "VSINI", "(V-K)0", "Delmag", "vsini_QuelozE", 
        "vsini_err_QuelozE", "vsini_QuelozC", "vsini_lim_QuelozC", 
        "vsini_err_QuelozC", "vsini_Terndrup", "vsini_err_Terndrup",
        "vsini_lim_Terndrup", "vsini_Soderblom", "vsini_lim_Soderblom",
        "vsini_SH", "vsini_err_SH", "vsini_lim_SH", "vsini_S84",
        "vsini_err_S84", "vsini_lim_S84", "vsini_Jackson", "vsini_err_Jackson",
    "vsini_lim_Jackson"]].groups.aggregate(np.mean)
    pleiades = au.join_by_id(
        unique_pleiades, apogee_avg, "APOGEE_ID", "APOGEE_ID")
    phot_bin = np.logical_or(
        np.logical_and(pleiades["(V-K)0"] <= 5, pleiades["Delmag"] >= 0.3),
        np.logical_and(pleiades["(V-K)0"] > 5, pleiades["Delmag"] >= 0.3))
    apogee_dets = pleiades["VSINI"] > 10

    apogee_logvsini_err = 0.12 / np.log(10)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the Queloz targets in the first panel.
    queloz_elodie = ~pleiades["vsini_QuelozE"].mask
    queloz_coravel = ~pleiades["vsini_QuelozC"].mask
    full_queloz = np.logical_or(queloz_elodie, queloz_coravel)
    queloz_coravel_upper = pleiades["vsini_lim_QuelozC"] == "<"
    queloz_coravel_lower = pleiades["vsini_lim_QuelozC"] == ">"
    queloz_coravel_det = np.logical_not(
        np.logical_or(queloz_coravel_upper,queloz_coravel_lower))

    queloz_elodie_photbin = np.logical_and(queloz_elodie, phot_bin)
    queloz_elodie_single = np.logical_and(queloz_elodie, ~phot_bin)
    queloz_coravel_photbin = np.logical_and(queloz_coravel_det, phot_bin)
    queloz_coravel_single = np.logical_and(queloz_coravel_det, ~phot_bin)
    queloz_coravel_upper_photbin = np.logical_and(
        queloz_coravel_upper, phot_bin)
    queloz_coravel_upper_single = np.logical_and(
        queloz_coravel_upper, ~phot_bin)
    queloz_coravel_lower_photbin = np.logical_and(
        queloz_coravel_lower, phot_bin)
    queloz_coravel_lower_single = np.logical_and(
        queloz_coravel_lower, ~phot_bin)
    queloz_coravel_all_photbin = au.multi_logical_or(
        queloz_coravel_photbin, queloz_coravel_lower_photbin,
        queloz_coravel_upper_photbin)
    queloz_coravel_all_single = au.multi_logical_or(
        queloz_coravel_single, queloz_coravel_lower_single,
        queloz_coravel_upper_single)

    # Split the Queloz detections into APOGEE detections and nondetections.
    queloz_elodie_dets_apogee = np.logical_and(
        queloz_elodie_single, apogee_dets)
    queloz_elodie_nondets_apogee = np.logical_and(
        queloz_elodie_single, ~apogee_dets)
    queloz_coravel_dets_apogee = np.logical_and(
        queloz_coravel_single, apogee_dets)
    queloz_coravel_nondets_apogee = np.logical_and(
        queloz_coravel_single, ~apogee_dets)
    # Split the queloz photometric binaries into APOGEE detections and nondetections.
    queloz_elodie_dets_photbins = np.logical_and(
        queloz_elodie_photbin, apogee_dets)
    queloz_elodie_nondets_photbins = np.logical_and(
        queloz_elodie_photbin, ~apogee_dets)
    queloz_coravel_dets_photbins = np.logical_and(
        queloz_coravel_photbin, apogee_dets)
    queloz_coravel_nondets_photbins = np.logical_and(
        queloz_coravel_photbin, ~apogee_dets)
    # Split the queloz upper limits into APOGEE detections and nondetections.
    queloz_coravel_dets_upper_single = np.logical_and(
        queloz_coravel_upper_single, apogee_dets)
    queloz_coravel_nondets_upper_single = np.logical_and(
        queloz_coravel_upper_single, ~apogee_dets)
    queloz_coravel_dets_lower_single = np.logical_and(
        queloz_coravel_lower_single, apogee_dets)
    queloz_coravel_nondets_lower_single = np.logical_and(
        queloz_coravel_lower_single, ~apogee_dets)
    queloz_coravel_dets_upper_photbin = np.logical_and(
        queloz_coravel_upper_photbin, apogee_dets)
    queloz_coravel_nondets_upper_photbin = np.logical_and(
        queloz_coravel_upper_photbin, ~apogee_dets)
    queloz_coravel_dets_lower_photbin = np.logical_and(
        queloz_coravel_lower_photbin, apogee_dets)
    queloz_coravel_nondets_lower_photbin = np.logical_and(
        queloz_coravel_lower_photbin, ~apogee_dets)


    queloze_logvsini_err = (
        pleiades["vsini_err_QuelozE"] / pleiades["vsini_QuelozE"] / np.log(10))
    quelozc_logvsini_err = (
        pleiades["vsini_err_QuelozC"] / pleiades["vsini_QuelozC"] / np.log(10))

    ax1.errorbar(
        np.log10(pleiades["vsini_QuelozE"][queloz_elodie_dets_apogee]),
        np.log10(pleiades["VSINI"][queloz_elodie_dets_apogee]), 
        yerr=apogee_logvsini_err, 
        xerr=queloze_logvsini_err[queloz_elodie_dets_apogee], marker="o", 
        color=bc.violet, ls="")
    ax1.errorbar(
        np.log10(pleiades["vsini_QuelozE"][queloz_elodie_nondets_apogee]),
        np.log10(pleiades["VSINI"][queloz_elodie_nondets_apogee]), 
        yerr=apogee_logvsini_err,
        xerr=queloze_logvsini_err[queloz_elodie_nondets_apogee], marker="o", 
        color="grey", ls="", alpha=0.3)
    ax1.errorbar(
        np.log10(pleiades["vsini_QuelozC"][queloz_coravel_dets_apogee]), 
        np.log10(pleiades["VSINI"][queloz_coravel_dets_apogee]), 
        yerr=apogee_logvsini_err, 
        xerr=quelozc_logvsini_err[queloz_coravel_dets_apogee], marker="o", 
        color=bc.violet, ls="")
    ax1.errorbar(
        np.log10(pleiades["vsini_QuelozC"][queloz_coravel_dets_upper_single]), 
        np.log10(pleiades["VSINI"][queloz_coravel_dets_upper_single]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color=bc.violet, ls="")
    ax1.errorbar(
        np.log10(pleiades["vsini_QuelozC"][queloz_coravel_nondets_upper_single]), 
        np.log10(pleiades["VSINI"][queloz_coravel_nondets_upper_single]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color="grey", ls="", 
        alpha=0.3)
    ax1.errorbar(
        np.log10(pleiades["vsini_QuelozC"][queloz_coravel_dets_lower_single]), 
        np.log10(pleiades["VSINI"][queloz_coravel_dets_lower_single]), 
        yerr=apogee_logvsini_err, 
        xerr=quelozc_logvsini_err[queloz_coravel_dets_lower_single], marker=">", 
        color=bc.violet, ls="")
    ax1.errorbar(
        np.log10(pleiades["vsini_QuelozC"][queloz_coravel_nondets_lower_single]), 
        np.log10(pleiades["VSINI"][queloz_coravel_nondets_lower_single]), 
        yerr=apogee_logvsini_err,
        xerr=quelozc_logvsini_err[queloz_coravel_nondets_lower_single], 
        marker=">", color="grey", ls="", alpha=0.3)
    ax1.plot([0, 2], [0, 2], 'k-')

    ax2.errorbar(
        np.log10(pleiades["vsini_QuelozE"][queloz_elodie_dets_photbins]),
        np.log10(pleiades["VSINI"][queloz_elodie_dets_photbins]), 
        yerr=apogee_logvsini_err,
        xerr=queloze_logvsini_err[queloz_elodie_dets_photbins], marker="o", 
        color=bc.green, ls="")
    ax2.errorbar(
        np.log10(pleiades["vsini_QuelozE"][queloz_elodie_nondets_photbins]),
        np.log10(pleiades["VSINI"][queloz_elodie_nondets_photbins]), 
        yerr=apogee_logvsini_err,
        xerr=queloze_logvsini_err[queloz_elodie_nondets_photbins], 
        marker="o", color="grey", ls="", alpha=0.3)
    ax2.errorbar(
        np.log10(pleiades["vsini_QuelozC"][queloz_coravel_dets_photbins]),
        np.log10(pleiades["VSINI"][queloz_coravel_dets_photbins]), 
        yerr=apogee_logvsini_err,
        xerr=quelozc_logvsini_err[queloz_coravel_dets_photbins], 
        marker="o", color=bc.green, ls="")
    ax2.errorbar(
        np.log10(pleiades["vsini_QuelozC"][queloz_coravel_nondets_photbins]),
        np.log10(pleiades["VSINI"][queloz_coravel_nondets_photbins]), 
        yerr=apogee_logvsini_err,
        xerr=quelozc_logvsini_err[queloz_coravel_nondets_photbins], 
        marker="o", color="grey", ls="", alpha=0.3)
    ax2.errorbar(
        np.log10(pleiades["vsini_QuelozC"][queloz_coravel_dets_upper_photbin]), 
        np.log10(pleiades["VSINI"][queloz_coravel_dets_upper_photbin]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color=bc.green, ls="")
    ax2.errorbar(
        np.log10(pleiades["vsini_QuelozC"][queloz_coravel_nondets_upper_photbin]), 
        np.log10(pleiades["VSINI"][queloz_coravel_nondets_upper_photbin]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color="grey", ls="", 
        alpha=0.3)
    ax2.errorbar(
        np.log10(pleiades["vsini_QuelozC"][queloz_coravel_dets_lower_photbin]), 
        np.log10(pleiades["VSINI"][queloz_coravel_dets_lower_photbin]), 
        yerr=apogee_logvsini_err, xerr=0, marker=">", color=bc.green, ls="")
    ax2.errorbar(
        np.log10(pleiades["vsini_QuelozC"][queloz_coravel_nondets_lower_photbin]), 
        np.log10(pleiades["VSINI"][queloz_coravel_nondets_lower_photbin]), 
        yerr=apogee_logvsini_err,  xerr=0, marker=">", color="grey", ls="", 
        alpha=0.3)
    ax2.plot([0, 2], [0, 2], 'k-')

    # Plot Terndrup targets in the second panel
    terndrup = ~pleiades["vsini_Terndrup"].mask
    terndrup_detections = pleiades["vsini_lim_Terndrup"] == "d"
    terndrup_limits = pleiades["vsini_lim_Terndrup"] == "u"

    terndrup_vdiff = (
        np.log10(pleiades["VSINI"]) - np.log10(pleiades["vsini_Terndrup"]))

    terndrup_photbin = np.logical_and(terndrup_detections, phot_bin)
    terndrup_singles = np.logical_and(terndrup_detections, ~phot_bin)
    terndrup_lim_photbin = np.logical_and(terndrup_limits, phot_bin)
    terndrup_lim_singles = np.logical_and(terndrup_limits, ~phot_bin)

    terndrup_all_photbin = au.multi_logical_or(
        terndrup_photbin, terndrup_lim_photbin)
    terndrup_all_singles = au.multi_logical_or(
        terndrup_singles, terndrup_lim_singles)

    # Mark Terndrup detections as APOGEE detections
    terndrup_dets_apogee = np.logical_and(terndrup_singles, apogee_dets)
    terndrup_nondets_apogee = np.logical_and(terndrup_singles, ~apogee_dets)
    # Mark Terndrup photometric binaries as APOGEE detections
    terndrup_dets_photbin = np.logical_and(terndrup_photbin, apogee_dets)
    terndrup_nondets_photbin = np.logical_and(terndrup_photbin, ~apogee_dets)
    # Mark Terndrup upper limits as APOGEE detections
    terndrup_dets_limits = np.logical_and(terndrup_lim_singles, apogee_dets)
    terndrup_nondets_limits = np.logical_and(terndrup_lim_singles, ~apogee_dets)
    # Mark terndrup upper limit photometric binaries as APOGEE detections
    terndrup_dets_lim_photbin = np.logical_and(terndrup_lim_photbin, apogee_dets)
    terndrup_nondets_lim_photbin = np.logical_and(terndrup_lim_photbin, ~apogee_dets)

    terndrup_logvsini_err = (
        pleiades["vsini_err_Terndrup"] / pleiades["vsini_Terndrup"] / np.log(10))

    ax1.errorbar(
        np.log10(pleiades["vsini_Terndrup"][terndrup_dets_apogee]), 
        np.log10(pleiades["VSINI"][terndrup_dets_apogee]), 
        yerr=apogee_logvsini_err,
        xerr=terndrup_logvsini_err[terndrup_dets_apogee], marker="s", color="blue", ls="")
    ax1.errorbar(
        np.log10(pleiades["vsini_Terndrup"][terndrup_nondets_apogee]), 
        np.log10(pleiades["VSINI"][terndrup_nondets_apogee]), 
        yerr=apogee_logvsini_err,
        xerr=terndrup_logvsini_err[terndrup_nondets_apogee], marker="s", 
        color="grey", ls="", alpha=0.3)
    ax1.errorbar(
        np.log10(pleiades["vsini_Terndrup"][terndrup_dets_limits]), 
        np.log10(pleiades["VSINI"][terndrup_dets_limits]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color="blue", ls="")
    ax1.errorbar(
        np.log10(pleiades["vsini_Terndrup"][terndrup_nondets_limits]), 
        np.log10(pleiades["VSINI"][terndrup_nondets_limits]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color="grey", ls="", 
        alpha=0.3)

    ax2.errorbar(
        np.log10(pleiades["vsini_Terndrup"][terndrup_dets_photbin]),
        np.log10(pleiades["VSINI"][terndrup_dets_photbin]), 
        yerr=apogee_logvsini_err,
        xerr=terndrup_logvsini_err[terndrup_dets_photbin], marker="s", 
        color=bc.green, ls="")
    ax2.errorbar(
        np.log10(pleiades["vsini_Terndrup"][terndrup_nondets_photbin]),
        np.log10(pleiades["VSINI"][terndrup_nondets_photbin]), 
        yerr=apogee_logvsini_err,
        xerr=terndrup_logvsini_err[terndrup_nondets_photbin], marker="s", 
        color="grey", ls="", alpha=0.3)
    ax2.errorbar(
        np.log10(pleiades["vsini_Terndrup"][terndrup_dets_lim_photbin]),
        np.log10(pleiades["VSINI"][terndrup_dets_lim_photbin]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color=bc.green, ls="")
    ax2.errorbar(
        np.log10(pleiades["vsini_Terndrup"][terndrup_nondets_lim_photbin]),
        np.log10(pleiades["VSINI"][terndrup_nondets_lim_photbin]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color="grey", ls="", 
        alpha=0.3)

    # Now plot Soderblom points
    soderblom = ~pleiades["vsini_Soderblom"].mask
    soderblom_detections = pleiades["vsini_lim_Soderblom"] == "d"
    soderblom_limits = pleiades["vsini_lim_Soderblom"] == "u"

    soderblom_det_photbins = np.logical_and(soderblom_detections, phot_bin)
    soderblom_det_singles = np.logical_and(soderblom_detections, ~phot_bin)
    soderblom_lim_photbins = np.logical_and(soderblom_limits, phot_bin)
    soderblom_lim_singles = np.logical_and(soderblom_limits, ~phot_bin)

    soderblom_all_photbins = au.multi_logical_or(
        soderblom_det_photbins, soderblom_lim_photbins)
    soderblom_all_singles = au.multi_logical_or(
        soderblom_det_singles, soderblom_lim_singles)

    # Mark Soderblom detections as APOGEE detections
    soderblom_dets_apogee = np.logical_and(soderblom_det_singles, apogee_dets)
    soderblom_nondets_apogee = np.logical_and(soderblom_det_singles, ~apogee_dets)
    # Mark Soderblom photometric binaries as APOGEE detections
    soderblom_dets_photbin = np.logical_and(soderblom_det_photbins, apogee_dets)
    soderblom_nondets_photbin = np.logical_and(soderblom_det_photbins, ~apogee_dets)
    # Mark Soderblom limits as APOGEE detections
    soderblom_dets_limits = np.logical_and(soderblom_lim_singles, apogee_dets)
    soderblom_nondets_limits = np.logical_and(soderblom_lim_singles, ~apogee_dets)
    # Mark Soderblom photometric binary limits as APOGEE detections
    soderblom_dets_lim_photbin = np.logical_and(soderblom_lim_photbins, apogee_dets)
    soderblom_nondets_lim_photbin = np.logical_and(soderblom_lim_photbins, ~apogee_dets)

    ax1.errorbar(
        np.log10(pleiades["vsini_Soderblom"][soderblom_dets_apogee]), 
        np.log10(pleiades["VSINI"][soderblom_dets_apogee]), 
        yerr=apogee_logvsini_err, xerr=0, marker="p", color=bc.red, ls="")
    ax1.errorbar(
        np.log10(pleiades["vsini_Soderblom"][soderblom_nondets_apogee]), 
        np.log10(pleiades["VSINI"][soderblom_nondets_apogee]), 
        yerr=apogee_logvsini_err, xerr=0, marker="p", color="grey", ls="", 
        alpha=0.3)
    ax1.errorbar(
        np.log10(pleiades["vsini_Soderblom"][soderblom_dets_limits]), 
        np.log10(pleiades["VSINI"][soderblom_dets_limits]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color=bc.red, ls="")
    ax1.errorbar(
        np.log10(pleiades["vsini_Soderblom"][soderblom_nondets_limits]), 
        np.log10(pleiades["VSINI"][soderblom_nondets_limits]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color="grey", ls="", 
        alpha=0.3)

    ax2.errorbar(
        np.log10(pleiades["vsini_Soderblom"][soderblom_dets_photbin]),
        np.log10(pleiades["VSINI"][soderblom_dets_photbin]), 
        yerr=apogee_logvsini_err, xerr=0, marker="p", color=bc.green, ls="")
    ax2.errorbar(
        np.log10(pleiades["vsini_Soderblom"][soderblom_nondets_photbin]),
        np.log10(pleiades["VSINI"][soderblom_nondets_photbin]), 
        yerr=apogee_logvsini_err, xerr=0, marker="p", color="grey", ls="", 
        alpha=0.3)
    ax2.errorbar(
        np.log10(pleiades["vsini_Soderblom"][soderblom_dets_lim_photbin]),
        np.log10(pleiades["VSINI"][soderblom_dets_lim_photbin]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color=bc.green, ls="")
    ax2.errorbar(
        np.log10(pleiades["vsini_Soderblom"][soderblom_nondets_lim_photbin]),
        np.log10(pleiades["VSINI"][soderblom_nondets_lim_photbin]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color="grey", ls="", 
        alpha=0.3)

    # Now plot Stauffer & Hartmann points.
    sh = ~pleiades["vsini_SH"].mask
    sh_detections = pleiades["vsini_lim_SH"] == "d"
    sh_upper = pleiades["vsini_lim_SH"] == "u"

    sh_det_singles = np.logical_and(sh_detections, ~phot_bin)
    sh_det_photbin = np.logical_and(sh_detections, phot_bin)
    sh_lim_singles = np.logical_and(sh_upper, ~phot_bin)
    sh_lim_photbin = np.logical_and(sh_upper, phot_bin)
    
    sh_all_photbin = au.multi_logical_or(
        sh_det_photbin, sh_lim_photbin)
    sh_all_singles = au.multi_logical_or(
        sh_det_singles, sh_lim_singles)

    # Mark Stauffer Hartmann detections as APOGEE detections
    sh_dets_apogee = np.logical_and(sh_det_singles, apogee_dets)
    sh_nondets_apogee = np.logical_and(sh_det_singles, ~apogee_dets)
    # Mark Stauffer Hartmann photometric binaries as APOGEE detections
    sh_dets_photbin = np.logical_and(sh_det_photbin, apogee_dets)
    sh_nondets_photbin = np.logical_and(sh_det_photbin, ~apogee_dets)
    # Mark Stauffer Hartmann upper limits as APOGEE detections
    sh_dets_limits = np.logical_and(sh_lim_singles, apogee_dets)
    sh_nondets_limits = np.logical_and(sh_lim_singles, ~apogee_dets)
    # Mark Stauffer Hartmann photometric binary upper limits as APOGEE detections
    sh_dets_lim_photbin = np.logical_and(sh_lim_photbin, apogee_dets)
    sh_nondets_lim_photbin = np.logical_and(sh_lim_photbin, ~apogee_dets)

    sh_logvsini_err = (
        pleiades["vsini_err_SH"] / pleiades["vsini_SH"] / np.log(10))

    ax1.errorbar(
        np.log10(pleiades["vsini_SH"][sh_dets_apogee]), 
        np.log10(pleiades["VSINI"][sh_dets_apogee]), 
        yerr=apogee_logvsini_err,
        xerr=sh_logvsini_err[sh_dets_apogee], marker="*", color=bc.light_pink, 
        ls="")
    ax1.errorbar(
        np.log10(pleiades["vsini_SH"][sh_nondets_apogee]), 
        np.log10(pleiades["VSINI"][sh_nondets_apogee]), 
        yerr=apogee_logvsini_err,
        xerr=sh_logvsini_err[sh_nondets_apogee], 
        marker="*", color="grey", ls="", alpha=0.3)
    ax1.errorbar(
        np.log10(pleiades["vsini_SH"][sh_dets_limits]), 
        np.log10(pleiades["VSINI"][sh_dets_limits]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color=bc.light_pink, ls="")
    ax1.errorbar(
        np.log10(pleiades["vsini_SH"][sh_nondets_limits]), 
        np.log10(pleiades["VSINI"][sh_nondets_limits]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color="grey", ls="", 
        alpha=0.3)


    ax2.errorbar(
        np.log10(pleiades["vsini_SH"][sh_dets_photbin]),
        np.log10(pleiades["VSINI"][sh_dets_photbin]), 
        yerr=apogee_logvsini_err,
        xerr=sh_logvsini_err[sh_dets_photbin], marker="*", color=bc.green, ls="")
    ax2.errorbar(
        np.log10(pleiades["vsini_SH"][sh_nondets_photbin]),
        np.log10(pleiades["VSINI"][sh_nondets_photbin]), 
        yerr=apogee_logvsini_err,
        xerr=sh_logvsini_err[sh_nondets_photbin], marker="*", color="grey", 
        ls="", alpha=0.3)
    ax2.errorbar(
        np.log10(pleiades["vsini_SH"][sh_dets_lim_photbin]),
        np.log10(pleiades["VSINI"][sh_dets_lim_photbin]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color=bc.green, ls="")
    ax2.errorbar(
        np.log10(pleiades["vsini_SH"][sh_nondets_lim_photbin]),
        np.log10(pleiades["VSINI"][sh_nondets_lim_photbin]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color="grey", ls="", 
        alpha=0.3)

    # Lastly Stauffer points.
    s84 = ~pleiades["vsini_S84"].mask
    s84_detections = pleiades["vsini_lim_S84"] == "d"
    s84_upper = pleiades["vsini_lim_S84"] == "u"

    s84_det_singles = np.logical_and(s84_detections, ~phot_bin)
    s84_det_photbin = np.logical_and(s84_detections, phot_bin)
    s84_lim_singles = np.logical_and(s84_upper, ~phot_bin)
    s84_lim_photbin = np.logical_and(s84_upper, phot_bin)

    s84_all_photbin = au.multi_logical_or(
        s84_det_photbin, s84_lim_photbin)
    s84_all_singles = au.multi_logical_or(
        s84_det_singles, s84_lim_singles)

    # Mark Stauffer detections as APOGEE detections
    s84_dets_apogee = np.logical_and(s84_det_singles, apogee_dets)
    s84_nondets_apogee = np.logical_and(s84_det_singles, ~apogee_dets)
    # Mark Stauffer photometric binaries as APOGEE detections
    s84_dets_photbin = np.logical_and(s84_det_photbin, apogee_dets)
    s84_nondets_photbin = np.logical_and(s84_det_photbin, ~apogee_dets)
    # Mark Stauffer upper limits as APOGEE detections
    s84_dets_limits = np.logical_and(s84_lim_singles, apogee_dets)
    s84_nondets_limits = np.logical_and(s84_lim_singles, ~apogee_dets)
    # Mark Stauffer photometric binary upper limits as APOGEE detections
    s84_dets_lim_photbin = np.logical_and(s84_lim_photbin, apogee_dets)
    s84_nondets_lim_photbin = np.logical_and(s84_lim_photbin, ~apogee_dets)

    s84_logvsini_err = (
        pleiades["vsini_err_S84"] / pleiades["vsini_S84"] / np.log(10))

    ax1.errorbar(
        np.log10(pleiades["vsini_S84"][s84_dets_apogee]), 
        np.log10(pleiades["VSINI"][s84_dets_apogee]), 
        yerr=apogee_logvsini_err, 
        xerr=s84_logvsini_err[s84_dets_apogee], 
        marker="X", color=bc.algae, ls="")
    ax1.errorbar(
        np.log10(pleiades["vsini_S84"][s84_nondets_apogee]), 
        np.log10(pleiades["VSINI"][s84_nondets_apogee]), 
        yerr=apogee_logvsini_err, 
        xerr=sh_logvsini_err[s84_nondets_apogee], 
        marker="X", color="grey", ls="", alpha=0.3)
    ax1.errorbar(
        np.log10(pleiades["vsini_S84"][s84_dets_limits]), 
        np.log10(pleiades["VSINI"][s84_dets_limits]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color=bc.algae, ls="")
    ax1.errorbar(
        np.log10(pleiades["vsini_S84"][s84_nondets_limits]), 
        np.log10(pleiades["VSINI"][s84_nondets_limits]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color="grey", ls="", 
        alpha=0.3)

    ax2.errorbar(
        np.log10(pleiades["vsini_S84"][s84_dets_photbin]),
        np.log10(pleiades["VSINI"][s84_dets_photbin]), 
        yerr=apogee_logvsini_err, 
        xerr=s84_logvsini_err[s84_dets_photbin], 
        marker="X", color=bc.green, ls="")
    ax2.errorbar(
        np.log10(pleiades["vsini_S84"][s84_nondets_photbin]),
        np.log10(pleiades["VSINI"][s84_nondets_photbin]), 
        yerr=apogee_logvsini_err, 
        xerr=sh_logvsini_err[s84_nondets_photbin], 
        marker="X", color="grey", ls="", alpha=0.3)
    ax2.errorbar(
        np.log10(pleiades["vsini_S84"][s84_dets_lim_photbin]),
        np.log10(pleiades["VSINI"][s84_dets_lim_photbin]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color=bc.green, ls="")
    ax2.errorbar(
        np.log10(pleiades["vsini_S84"][s84_nondets_lim_photbin]),
        np.log10(pleiades["VSINI"][s84_nondets_lim_photbin]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color="grey", ls="", 
        alpha=0.3)

    # Another round of points from Jackson and Jeffries
    jackson = ~pleiades["vsini_Jackson"].mask
    jackson_detections = pleiades["vsini_lim_Jackson"] == "d"
    jackson_upper = pleiades["vsini_lim_Jackson"] == "u"

    jackson_det_singles = np.logical_and(jackson_detections, ~phot_bin)
    jackson_det_photbin = np.logical_and(jackson_detections, phot_bin)
    jackson_lim_singles = np.logical_and(jackson_upper, ~phot_bin)
    jackson_lim_photbin = np.logical_and(jackson_upper, phot_bin)

    jackson_all_photbin = au.multi_logical_or(
        jackson_det_photbin, jackson_lim_photbin)
    jackson_all_singles = au.multi_logical_or(
        jackson_det_singles, jackson_lim_singles)

    # Mark Stauffer detections as APOGEE detections
    jackson_dets_apogee = np.logical_and(jackson_det_singles, apogee_dets)
    jackson_nondets_apogee = np.logical_and(jackson_det_singles, ~apogee_dets)
    # Mark Stauffer photometric binaries as APOGEE detections
    jackson_dets_photbin = np.logical_and(jackson_det_photbin, apogee_dets)
    jackson_nondets_photbin = np.logical_and(jackson_det_photbin, ~apogee_dets)
    # Mark Stauffer upper limits as APOGEE detections
    jackson_dets_limits = np.logical_and(jackson_lim_singles, apogee_dets)
    jackson_nondets_limits = np.logical_and(jackson_lim_singles, ~apogee_dets)
    # Mark Stauffer photometric binary upper limits as APOGEE detections
    jackson_dets_lim_photbin = np.logical_and(jackson_lim_photbin, apogee_dets)
    jackson_nondets_lim_photbin = np.logical_and(jackson_lim_photbin, ~apogee_dets)

    jackson_logvsini_err = (
        pleiades["vsini_err_Jackson"] / pleiades["vsini_Jackson"] / np.log(10))

    ax1.errorbar(
        np.log10(pleiades["vsini_Jackson"][jackson_dets_apogee]), 
        np.log10(pleiades["VSINI"][jackson_dets_apogee]), 
        yerr=apogee_logvsini_err, 
        xerr=jackson_logvsini_err[jackson_dets_apogee], marker="o", 
        color=bc.purple, ls="")
    ax1.errorbar(
        np.log10(pleiades["vsini_Jackson"][jackson_nondets_apogee]), 
        np.log10(pleiades["VSINI"][jackson_nondets_apogee]), 
        yerr=apogee_logvsini_err, 
        xerr=jackson_logvsini_err[jackson_nondets_apogee], marker="o", 
        color="grey", ls="", alpha=0.3)
    ax1.errorbar(
        np.log10(pleiades["vsini_Jackson"][jackson_dets_limits]), 
        np.log10(pleiades["VSINI"][jackson_dets_limits]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color=bc.purple, ls="")
    ax1.errorbar(
        np.log10(pleiades["vsini_Jackson"][jackson_nondets_limits]), 
        np.log10(pleiades["VSINI"][jackson_nondets_limits]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color="grey", ls="", 
        alpha=0.3)

    ax2.errorbar(
        np.log10(pleiades["vsini_Jackson"][jackson_dets_photbin]),
        np.log10(pleiades["VSINI"][jackson_dets_photbin]), 
        yerr=apogee_logvsini_err, 
        xerr=jackson_logvsini_err[jackson_dets_photbin], marker="d", 
        color=bc.green, ls="")
    ax2.errorbar(
        np.log10(pleiades["vsini_Jackson"][jackson_nondets_photbin]),
        np.log10(pleiades["VSINI"][jackson_nondets_photbin]), 
        yerr=apogee_logvsini_err, 
        xerr=jackson_logvsini_err[jackson_nondets_photbin], marker="d", 
        color="grey", ls="", alpha=0.3)
    ax2.errorbar(
        np.log10(pleiades["vsini_Jackson"][jackson_dets_lim_photbin]),
        np.log10(pleiades["VSINI"][jackson_dets_lim_photbin]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color=bc.green, ls="")
    ax2.errorbar(
        np.log10(pleiades["vsini_Jackson"][jackson_nondets_lim_photbin]),
        np.log10(pleiades["VSINI"][jackson_nondets_lim_photbin]), 
        yerr=apogee_logvsini_err, xerr=0, marker="<", color="grey", ls="", 
        alpha=0.3)


    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 2)
    ax1.set_title("Photometric Singles")
    ax1.set_xlabel(r"Literature $\log_{10}(v \sin i)$")
    ax1.set_ylabel(r"APOGEE $\log_{10}(v \sin i)$")

    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 2)
    ax2.set_title("Photometric Binaries")
    ax2.set_xlabel(r"Literature $\log_{10}(v \sin i)$")
    ax2.set_ylabel(r"APOGEE $\log_{10}(v \sin i)$")

    plt.tight_layout()

@write_plot("f7")
def Pleiades_vsini_outliers():
    '''Compare APOGEE and literature vsini for different surveys.'''
    full_pleiades = cache.pleiades_APOGEE_Literature_vsini()
    pleiades_group = full_pleiades.group_by("APOGEE_ID")
    apogee_avg = pleiades_group[[
        "APOGEE_ID", "VSINI"]].groups.aggregate(np.mean)
    unique_pleiades = unique(full_pleiades, keys="APOGEE_ID")
    unique_pleiades.remove_column("VSINI")
    pleiades = pleiades_group[[
        "APOGEE_ID", "VSINI", "(V-K)0", "Delmag", "vsini_QuelozE", 
        "vsini_err_QuelozE", "vsini_QuelozC", "vsini_lim_QuelozC", 
        "vsini_err_QuelozC", "vsini_SH", "vsini_err_SH", "vsini_lim_SH", 
        "vsini_Jackson", "vsini_err_Jackson", 
        "vsini_lim_Jackson"]].groups.aggregate(np.mean)
    pleiades = au.join_by_id(
        unique_pleiades, apogee_avg, "APOGEE_ID", "APOGEE_ID")
    apogee_dets = pleiades["VSINI"] > 10
    # I want to mask the nondetections.
    pleiades["vsini_err_QuelozC"][pleiades["vsini_lim_QuelozC"] != " "] = np.ma.masked
    pleiades["vsini_err_SH"][pleiades["vsini_lim_SH"] != "d"] = np.ma.masked
    pleiades["vsini_err_Jackson"][pleiades["vsini_lim_Jackson"] != "d"] = np.ma.masked
    # Also mask the nans
    pleiades["vsini_QuelozE"] = np.ma.masked_invalid(
        pleiades["vsini_QuelozE"])
    pleiades["vsini_QuelozC"] = np.ma.masked_invalid(
        pleiades["vsini_QuelozC"])
    pleiades["vsini_err_QuelozE"] = np.ma.masked_invalid(
        pleiades["vsini_err_QuelozE"])
    pleiades["vsini_err_QuelozC"] = np.ma.masked_invalid(
        pleiades["vsini_err_QuelozC"])
    # Stauffer and Hartmann don't have errors for vsini estimated from H-alpha.
    # Exclude these targets.
    pleiades["vsini_SH"][np.where(npstr.count(pleiades["Notes_SH"], "2"))] = \
        np.ma.masked
    with_errors = au.multi_logical_or(
        ~pleiades["vsini_err_QuelozE"].mask, ~pleiades["vsini_err_QuelozC"].mask,
        ~pleiades["vsini_err_SH"].mask, ~pleiades["vsini_err_Jackson"].mask)
    valid_pleiades = au.multi_logical_and(apogee_dets, with_errors)

    apogee_logvsini_err = 0.12 / np.log(10)
    apogee_uperr = pleiades["VSINI"] * (10**apogee_logvsini_err-1)
    apogee_downerr = pleiades["VSINI"] * (1-10**-apogee_logvsini_err)
    queloze_logvsini_err = (
        pleiades["vsini_err_QuelozE"] / pleiades["vsini_QuelozE"] / np.log(10))
    queloze_uperr = pleiades["vsini_QuelozE"] * (10**queloze_logvsini_err-1)
    queloze_downerr = pleiades["vsini_QuelozE"] * (1-10**-queloze_logvsini_err)
    quelozc_logvsini_err = (
        pleiades["vsini_err_QuelozC"] / pleiades["vsini_QuelozC"] / np.log(10))
    quelozc_uperr = pleiades["vsini_QuelozC"] * (10**quelozc_logvsini_err-1)
    quelozc_downerr = pleiades["vsini_QuelozC"] * (1-10**-quelozc_logvsini_err)
    sh_logvsini_err = (
        pleiades["vsini_err_SH"] / pleiades["vsini_SH"] / np.log(10))
    sh_uperr = pleiades["vsini_SH"] * (10**sh_logvsini_err-1)
    sh_downerr = pleiades["vsini_SH"] * (1-10**-sh_logvsini_err)
    jackson_logvsini_err = (
        pleiades["vsini_err_Jackson"] / pleiades["vsini_Jackson"] / np.log(10))
    jackson_uperr = pleiades["vsini_Jackson"] * (10**jackson_logvsini_err-1)
    jackson_downerr = pleiades["vsini_Jackson"] * (1-10**-jackson_logvsini_err)


    # Split the sample up into upper/lower/detections.
    queloz_elodie = ~pleiades["vsini_QuelozE"].mask
    queloz_coravel = ~pleiades["vsini_QuelozC"].mask
    full_queloz = np.logical_or(queloz_elodie, queloz_coravel)
    queloz_coravel_upper = pleiades["vsini_lim_QuelozC"] == "<"
    queloz_coravel_lower = pleiades["vsini_lim_QuelozC"] == ">"
    queloz_coravel_det = np.logical_and(
        queloz_coravel, np.logical_not(np.logical_or(
            queloz_coravel_upper, queloz_coravel_lower)))
    # Check that there are no targets without errors.
    assert not any(pleiades["vsini_err_QuelozC"][queloz_coravel_det].mask)

    sh = ~pleiades["vsini_SH"].mask
    sh_detections = np.logical_and(sh, pleiades["vsini_lim_SH"] == "d")
    sh_upper = pleiades["vsini_lim_SH"] == "u"
    # Check that there are no targets without errors.
    assert not any(pleiades["vsini_err_SH"][sh_detections].mask)

    jackson = ~pleiades["vsini_Jackson"].mask
    jackson_detections = np.logical_and(
        jackson, pleiades["vsini_lim_Jackson"] == "d")
    jackson_upper = pleiades["vsini_lim_Jackson"] == "u"
    assert not any(pleiades["vsini_err_Jackson"][jackson_detections].mask)

    # Calculate the outliers.
    queloze_chisq = (
        (np.log10(pleiades["vsini_QuelozE"]) - np.log10(pleiades["VSINI"]))**2 /
        (queloze_logvsini_err**2 + apogee_logvsini_err**2))
    quelozc_chisq = (
        (np.log10(pleiades["vsini_QuelozC"]) - np.log10(pleiades["VSINI"]))**2 /
        (quelozc_logvsini_err**2 + apogee_logvsini_err**2))
    sh_chisq = (
        (np.log10(pleiades["vsini_SH"]) - np.log10(pleiades["VSINI"]))**2 /
        (sh_logvsini_err**2 + apogee_logvsini_err**2))
    jackson_chisq = (
        (np.log10(pleiades["vsini_Jackson"]) - np.log10(pleiades["VSINI"]))**2 /
        (jackson_logvsini_err**2 + apogee_logvsini_err**2))

    full_matrix = np.ma.array([
        queloze_chisq, quelozc_chisq, sh_chisq, jackson_chisq])

    outliers = np.ma.logical_and(full_matrix > 3**2, apogee_dets)

    def_outliers = np.ma.all(outliers, axis=0)
    maybe = np.ma.logical_and(
        np.ma.any(outliers, axis=0), np.ma.logical_not(def_outliers))
    def_consistent = np.ma.all(np.ma.logical_not(outliers), axis=0)
    assert all(def_outliers.mask == maybe.mask)
    assert all(def_outliers.mask == def_consistent.mask)
    unknown = def_outliers.mask
    def_outliers = def_outliers.filled(0)
    maybe = maybe.filled(0)
    def_consistent = def_consistent.filled(0)
    assert all(np.count_nonzero(
        [unknown, def_outliers, maybe, def_consistent], axis=0) == 1)

    outlier_inds = np.logical_and(
        np.logical_or(def_outliers, maybe), apogee_dets)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 9))

    # Plot Stauffer & Hartmann points.
    # Mark Stauffer Hartmann detections as APOGEE detections
    sh_dets_apogee = np.logical_and(sh_detections, apogee_dets)
    sh_nondets_apogee = np.logical_and(sh_detections, ~apogee_dets)
    # Mark Stauffer Hartmann upper limits as APOGEE detections
    sh_dets_limits = np.logical_and(sh_upper, apogee_dets)
    sh_nondets_limits = np.logical_and(sh_upper, ~apogee_dets)

    sh_outliers = outliers[2,:]

    ax1.errorbar(
        pleiades["vsini_SH"][sh_dets_apogee], 
        pleiades["VSINI"][sh_dets_apogee], 
        yerr=[
            apogee_downerr[sh_dets_apogee],
            apogee_uperr[sh_dets_apogee]], 
        xerr=[
            sh_downerr[sh_dets_apogee],
            sh_uperr[sh_dets_apogee]], 
        marker="*", color=bc.light_pink, 
        ls="", label="Stauffer & Hartmann (1987)")
    ax1.errorbar(
        pleiades["vsini_SH"][sh_outliers], 
        pleiades["VSINI"][sh_outliers], 
        yerr=[
            apogee_downerr[sh_outliers],
            apogee_uperr[sh_outliers]], 
        xerr=[
            sh_downerr[sh_outliers],
            sh_uperr[sh_outliers]], 
        marker="*", color=bc.light_pink, 
        ls="", mec=bc.green, mew=2, label="")
    ax1.errorbar(
        pleiades["vsini_SH"][sh_nondets_apogee], 
        pleiades["VSINI"][sh_nondets_apogee], 
        yerr=[
            apogee_downerr[sh_nondets_apogee],
            apogee_uperr[sh_nondets_apogee]], 
        xerr=[
            sh_downerr[sh_nondets_apogee],
            sh_uperr[sh_nondets_apogee]], 
        marker="*", color="grey", ls="", alpha=0.3, label="")
    ax1.errorbar(
        pleiades["vsini_SH"][sh_dets_limits], 
        pleiades["VSINI"][sh_dets_limits], 
        yerr=[
            apogee_downerr[sh_dets_limits],
            apogee_uperr[sh_dets_limits]], 
        xerr=0, marker="<", color=bc.light_pink,
        ls="", label="")
    ax1.errorbar(
        pleiades["vsini_SH"][sh_nondets_limits], 
        pleiades["VSINI"][sh_nondets_limits], 
        yerr=[
            apogee_downerr[sh_nondets_limits],
            apogee_uperr[sh_nondets_limits]], 
        xerr=0, marker="<", color="grey", ls="", 
        alpha=0.3, label="")

    # Split the Queloz detections into APOGEE detections and nondetections.
    queloz_elodie_dets_apogee = np.logical_and(
        queloz_elodie, apogee_dets)
    queloz_elodie_nondets_apogee = np.logical_and(
        queloz_elodie, ~apogee_dets)
    queloz_coravel_dets_apogee = np.logical_and(
        queloz_coravel_det, apogee_dets)
    queloz_coravel_nondets_apogee = np.logical_and(
        queloz_coravel_det, ~apogee_dets)
    # Split the queloz upper limits into APOGEE detections and nondetections.
    queloz_coravel_dets_upper = np.logical_and(
        queloz_coravel_upper, apogee_dets)
    queloz_coravel_nondets_upper = np.logical_and(
        queloz_coravel_upper, ~apogee_dets)
    queloz_coravel_dets_lower = np.logical_and(
        queloz_coravel_lower, apogee_dets)
    queloz_coravel_nondets_lower = np.logical_and(
        queloz_coravel_lower, ~apogee_dets)

    queloze_outliers = outliers[0,:]
    quelozc_outliers = outliers[1,:]

    ax1.errorbar(
        pleiades["vsini_QuelozE"][queloz_elodie_dets_apogee],
        pleiades["VSINI"][queloz_elodie_dets_apogee], 
        yerr=[
            apogee_downerr[queloz_elodie_dets_apogee],
            apogee_uperr[queloz_elodie_dets_apogee]], 
        xerr=[
            queloze_downerr[queloz_elodie_dets_apogee],
            queloze_uperr[queloz_elodie_dets_apogee]], marker="o", 
        color=bc.violet, ls="", label="Queloz et al (1998)")
    ax1.errorbar(
        pleiades["vsini_QuelozE"][queloze_outliers],
        pleiades["VSINI"][queloze_outliers], 
        yerr=[
            apogee_downerr[queloze_outliers],
            apogee_uperr[queloze_outliers]], 
        xerr=[
            queloze_downerr[queloze_outliers],
            queloze_uperr[queloze_outliers]], marker="o", 
        color=bc.violet, ls="", mec=bc.green, mew=2, label="")
    ax1.errorbar(
        pleiades["vsini_QuelozE"][queloz_elodie_nondets_apogee],
        pleiades["VSINI"][queloz_elodie_nondets_apogee], 
        yerr=[
            apogee_downerr[queloz_elodie_nondets_apogee],
            apogee_uperr[queloz_elodie_nondets_apogee]], 
        xerr=[
            queloze_downerr[queloz_elodie_nondets_apogee],
            queloze_uperr[queloz_elodie_nondets_apogee]], marker="o", 
        color="grey", ls="", alpha=0.3, label="")
    ax1.errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_apogee], 
        pleiades["VSINI"][queloz_coravel_dets_apogee], 
        yerr=[
            apogee_downerr[queloz_coravel_dets_apogee],
            apogee_uperr[queloz_coravel_dets_apogee]], 
        xerr=[
            quelozc_downerr[queloz_coravel_dets_apogee],
            quelozc_uperr[queloz_coravel_dets_apogee]], marker="o", 
        color=bc.violet, ls="", label="")
    ax1.errorbar(
        pleiades["vsini_QuelozC"][quelozc_outliers],
        pleiades["VSINI"][quelozc_outliers], 
        yerr=[
            apogee_downerr[quelozc_outliers],
            apogee_uperr[quelozc_outliers]], 
        xerr=[
            quelozc_downerr[quelozc_outliers],
            quelozc_uperr[quelozc_outliers]], marker="o", 
        color=bc.violet, ls="", mec=bc.green, mew=2, label="")
    ax1.errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_upper], 
        pleiades["VSINI"][queloz_coravel_dets_upper], 
        yerr=[
            apogee_downerr[queloz_coravel_dets_upper],
            apogee_uperr[queloz_coravel_dets_upper]], 
        xerr=0, marker="<", color=bc.violet, ls="",
        label="")
    ax1.errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_upper], 
        pleiades["VSINI"][queloz_coravel_nondets_upper], 
        yerr=[
            apogee_downerr[queloz_coravel_nondets_upper],
            apogee_uperr[queloz_coravel_nondets_upper]], 
        xerr=0, marker="<", color="grey", ls="", 
        alpha=0.3, label="")
    ax1.errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_lower], 
        pleiades["VSINI"][queloz_coravel_dets_lower], 
        yerr=[
            apogee_downerr[queloz_coravel_dets_lower],
            apogee_uperr[queloz_coravel_dets_lower]], 
        xerr=[
            quelozc_downerr[queloz_coravel_dets_lower],
            quelozc_uperr[queloz_coravel_dets_lower]], marker=">", 
        color=bc.violet, ls="", label="")
    ax1.errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_lower], 
        pleiades["VSINI"][queloz_coravel_nondets_lower], 
        yerr=[
            apogee_downerr[queloz_coravel_nondets_lower],
            apogee_uperr[queloz_coravel_nondets_lower]], 
        xerr=[
            quelozc_downerr[queloz_coravel_nondets_lower],
            quelozc_uperr[queloz_coravel_nondets_lower]], 
        marker=">", color="grey", ls="", alpha=0.3, label="")
    ax1.plot([0, 100], [0, 100], 'k-')



    # Another round of points from Jackson and Jeffries
    # Mark Jackson detections as APOGEE detections
    jackson_dets_apogee = np.logical_and(jackson_detections, apogee_dets)
    jackson_nondets_apogee = np.logical_and(jackson_detections, ~apogee_dets)
    # Mark Jackson upper limits as APOGEE detections
    jackson_dets_limits = np.logical_and(jackson_upper, apogee_dets)
    jackson_nondets_limits = np.logical_and(jackson_upper, ~apogee_dets)

    jackson_outliers = outliers[3,:]

    ax1.errorbar(
        pleiades["vsini_Jackson"][jackson_dets_apogee], 
        pleiades["VSINI"][jackson_dets_apogee], 
        yerr=[
            apogee_downerr[jackson_dets_apogee],
            apogee_uperr[jackson_dets_apogee]], 
        xerr=[
            jackson_downerr[jackson_dets_apogee],
            jackson_uperr[jackson_dets_apogee]], 
        marker="s", color=bc.purple, ls="", label="Jackson et al (2018)")
    ax1.errorbar(
        pleiades["vsini_Jackson"][jackson_outliers], 
        pleiades["VSINI"][jackson_outliers], 
        yerr=[
            apogee_downerr[jackson_outliers],
            apogee_uperr[jackson_outliers]], 
        xerr=[
            jackson_downerr[jackson_outliers],
            jackson_uperr[jackson_outliers]], 
        marker="s", color=bc.purple, ls="", mec=bc.green, mew=2, label="")
    ax1.errorbar(
        pleiades["vsini_Jackson"][jackson_nondets_apogee], 
        pleiades["VSINI"][jackson_nondets_apogee], 
        yerr=[
            apogee_downerr[jackson_nondets_apogee],
            apogee_uperr[jackson_nondets_apogee]], 
        xerr=[
            jackson_downerr[jackson_nondets_apogee],
            jackson_uperr[jackson_nondets_apogee]], 
        marker="s", color="grey", ls="", alpha=0.3, label="")
    ax1.errorbar(
        pleiades["vsini_Jackson"][jackson_dets_limits], 
        pleiades["VSINI"][jackson_dets_limits], 
        yerr=[
            apogee_downerr[jackson_dets_limits],
            apogee_uperr[jackson_dets_limits]], 
        xerr=0, marker="<", color=bc.purple, ls="",
        label="")
    ax1.errorbar(
        pleiades["vsini_Jackson"][jackson_nondets_limits], 
        pleiades["VSINI"][jackson_nondets_limits], 
        yerr=[
            apogee_downerr[jackson_nondets_limits],
            apogee_uperr[jackson_nondets_limits]], 
        xerr=0, marker="<", color="grey", ls="", 
        alpha=0.3, label="")

    hr.color_mag_plot(
        pleiades["(V-K)_ST"][def_consistent], pleiades["MK"][def_consistent], 
        marker=".", color=bc.black, ls="", axis=ax2, label="")
    hr.color_mag_plot(
        pleiades["(V-K)_ST"][maybe], pleiades["MK"][maybe], marker="o", 
        color=bc.black, ls="", mec=bc.green, mew=2, ms=6, axis=ax2, label="")
    hr.color_mag_plot(
        pleiades["(V-K)_ST"][def_outliers], pleiades["MK"][def_outliers], 
        marker="o", color=bc.black, ls="", mec=bc.green, mew=2, ms=6, axis=ax2, 
        label="")

    ax1.set_xlim(1, 100)
    ax1.set_ylim(1, 100)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("")
    ax1.set_xlabel(r"Literature {0} ({1})".format(vsinistr, kmsstr))
    ax1.set_ylabel(r"APOGEE {0} ({1})".format(vsinistr, kmsstr))
    ax1.legend(loc="lower right")

    ax2.set_xlim(1, 5)
    ax2.set_ylim(6.5, 1)
    ax2.set_title("")
    ax2.set_xlabel(r"V-K")
    ax2.set_ylabel(MKstr)

    plt.tight_layout()


def Pleiades_direct_vsini_comparisons():
    '''Compare APOGEE and literature vsini for different surveys.'''
    full_pleiades = cache.pleiades_APOGEE_Literature_vsini()
    pleiades_group = full_pleiades.group_by("APOGEE_ID")
    apogee_avg = pleiades_group[[
        "APOGEE_ID", "VSINI"]].groups.aggregate(np.mean)
    unique_pleiades = unique(full_pleiades, keys="APOGEE_ID")
    unique_pleiades.remove_column("VSINI")
    pleiades = pleiades_group[[
        "APOGEE_ID", "VSINI", "(V-K)0", "Delmag", "vsini_QuelozE", 
        "vsini_err_QuelozE", "vsini_QuelozC", "vsini_lim_QuelozC", 
        "vsini_err_QuelozC", "vsini_Terndrup", "vsini_err_Terndrup",
        "vsini_lim_Terndrup", "vsini_Soderblom", "vsini_lim_Soderblom",
        "vsini_SH", "vsini_err_SH", "vsini_lim_SH", "vsini_S84",
        "vsini_err_S84", "vsini_lim_S84", "vsini_Jackson", "vsini_err_Jackson",
    "vsini_lim_Jackson"]].groups.aggregate(np.mean)
    pleiades = au.join_by_id(
        unique_pleiades, apogee_avg, "APOGEE_ID", "APOGEE_ID")
    phot_bin = np.logical_or(
        np.logical_and(pleiades["(V-K)0"] <= 5, pleiades["Delmag"] >= 0.3),
        np.logical_and(pleiades["(V-K)0"] > 5, pleiades["Delmag"] >= 0.3))
    apogee_dets = pleiades["VSINI"] > 10

    f, axes = plt.subplots(
        6, 2, figsize=(20, 20*1.1), sharey=False, sharex=False)

    axes = np.transpose(axes)
    
    # Plot the Queloz targets in the first panel.
    queloz_elodie = ~pleiades["vsini_QuelozE"].mask
    queloz_coravel = ~pleiades["vsini_QuelozC"].mask
    full_queloz = np.logical_or(queloz_elodie, queloz_coravel)
    queloz_coravel_upper = pleiades["vsini_lim_QuelozC"] == "<"
    queloz_coravel_lower = pleiades["vsini_lim_QuelozC"] == ">"
    queloz_coravel_det = np.logical_not(
        np.logical_or(queloz_coravel_upper,queloz_coravel_lower))

    queloz_elodie_photbin = np.logical_and(queloz_elodie, phot_bin)
    queloz_elodie_single = np.logical_and(queloz_elodie, ~phot_bin)
    queloz_coravel_photbin = np.logical_and(queloz_coravel_det, phot_bin)
    queloz_coravel_single = np.logical_and(queloz_coravel_det, ~phot_bin)
    queloz_coravel_upper_photbin = np.logical_and(
        queloz_coravel_upper, phot_bin)
    queloz_coravel_upper_single = np.logical_and(
        queloz_coravel_upper, ~phot_bin)
    queloz_coravel_lower_photbin = np.logical_and(
        queloz_coravel_lower, phot_bin)
    queloz_coravel_lower_single = np.logical_and(
        queloz_coravel_lower, ~phot_bin)
    queloz_coravel_all_photbin = au.multi_logical_or(
        queloz_coravel_photbin, queloz_coravel_lower_photbin,
        queloz_coravel_upper_photbin)
    queloz_coravel_all_single = au.multi_logical_or(
        queloz_coravel_single, queloz_coravel_lower_single,
        queloz_coravel_upper_single)

    # Split the Queloz detections into APOGEE detections and nondetections.
    queloz_elodie_dets_apogee = np.logical_and(
        queloz_elodie_single, apogee_dets)
    queloz_elodie_nondets_apogee = np.logical_and(
        queloz_elodie_single, ~apogee_dets)
    queloz_coravel_dets_apogee = np.logical_and(
        queloz_coravel_single, apogee_dets)
    queloz_coravel_nondets_apogee = np.logical_and(
        queloz_coravel_single, ~apogee_dets)
    # Split the queloz photometric binaries into APOGEE detections and nondetections.
    queloz_elodie_dets_photbins = np.logical_and(
        queloz_elodie_photbin, apogee_dets)
    queloz_elodie_nondets_photbins = np.logical_and(
        queloz_elodie_photbin, ~apogee_dets)
    queloz_coravel_dets_photbins = np.logical_and(
        queloz_coravel_photbin, apogee_dets)
    queloz_coravel_nondets_photbins = np.logical_and(
        queloz_coravel_photbin, ~apogee_dets)
    # Split the queloz upper limits into APOGEE detections and nondetections.
    queloz_coravel_dets_upper_single = np.logical_and(
        queloz_coravel_upper_single, apogee_dets)
    queloz_coravel_nondets_upper_single = np.logical_and(
        queloz_coravel_upper_single, ~apogee_dets)
    queloz_coravel_dets_lower_single = np.logical_and(
        queloz_coravel_lower_single, apogee_dets)
    queloz_coravel_nondets_lower_single = np.logical_and(
        queloz_coravel_lower_single, ~apogee_dets)
    queloz_coravel_dets_upper_photbin = np.logical_and(
        queloz_coravel_upper_photbin, apogee_dets)
    queloz_coravel_nondets_upper_photbin = np.logical_and(
        queloz_coravel_upper_photbin, ~apogee_dets)
    queloz_coravel_dets_lower_photbin = np.logical_and(
        queloz_coravel_lower_photbin, apogee_dets)
    queloz_coravel_nondets_lower_photbin = np.logical_and(
        queloz_coravel_lower_photbin, ~apogee_dets)

    hr.absmag_teff_plot(
        pleiades["TEFF"][queloz_elodie_single],
        pleiades["Dereddened K"][queloz_elodie_single],
        yerr=pleiades["K_ERR"][queloz_elodie_single],
        xerr=pleiades["TEFF_ERR"][queloz_elodie_single], marker="o",
        color=bc.red, ls="", axis=axes[0, 0])
    hr.absmag_teff_plot(
        pleiades["TEFF"][queloz_elodie_photbin],
        pleiades["Dereddened K"][queloz_elodie_photbin],
        yerr=pleiades["K_ERR"][queloz_elodie_photbin],
        xerr=pleiades["TEFF_ERR"][queloz_elodie_photbin], marker="8",
        color='r', ls="", axis=axes[0, 0])
    hr.absmag_teff_plot(
        pleiades["TEFF"][queloz_coravel_all_single],
        pleiades["Dereddened K"][queloz_coravel_all_single],
        yerr=pleiades["K_ERR"][queloz_coravel_all_single],
        xerr=pleiades["TEFF_ERR"][queloz_coravel_all_single], marker="o",
        color=bc.red, ls="", axis=axes[0, 0])
    hr.absmag_teff_plot(
        pleiades["TEFF"][queloz_coravel_all_photbin],
        pleiades["Dereddened K"][queloz_coravel_all_photbin],
        yerr=pleiades["K_ERR"][queloz_coravel_all_photbin],
        xerr=pleiades["TEFF_ERR"][queloz_coravel_all_photbin], marker="8",
        color='r', ls="", axis=axes[0, 0])

    axes[0, 0].set_xlim(6750, 3500)
    axes[0, 0].set_ylim(11.7, 7)
    axes[0, 0].set_title("Queloz et al (1998)")
    axes[0, 0].set_ylabel("Dereddened $K_S$")
    axes[0, 0].set_xlabel(Teffstr)

    axes[1, 0].errorbar(
        pleiades["vsini_QuelozE"][queloz_elodie_dets_apogee],
        pleiades["VSINI"][queloz_elodie_dets_apogee], 
        yerr=pleiades["VSINI_ERR"][queloz_elodie_dets_apogee], 
        xerr=pleiades["vsini_err_QuelozE"][queloz_elodie_dets_apogee], 
        marker="o", color=bc.red, ls="")
    axes[1, 0].errorbar(
        pleiades["vsini_QuelozE"][queloz_elodie_nondets_apogee],
        pleiades["VSINI"][queloz_elodie_nondets_apogee], 
        yerr=pleiades["VSINI_ERR"][queloz_elodie_nondets_apogee],
        xerr=pleiades["vsini_err_QuelozE"][queloz_elodie_nondets_apogee], 
        marker="o", color="grey", ls="", alpha=0.3)
    axes[1, 0].errorbar(
        pleiades["vsini_QuelozE"][queloz_elodie_dets_photbins],
        pleiades["VSINI"][queloz_elodie_dets_photbins], 
        yerr=pleiades["VSINI_ERR"][queloz_elodie_dets_photbins], 
        xerr=pleiades["vsini_err_QuelozE"][queloz_elodie_dets_photbins], 
        marker="8", color="red", ls="")
    axes[1, 0].errorbar(
        pleiades["vsini_QuelozE"][queloz_elodie_nondets_photbins],
        pleiades["VSINI"][queloz_elodie_nondets_photbins], 
        yerr=pleiades["VSINI_ERR"][queloz_elodie_nondets_photbins], 
        xerr=pleiades["vsini_err_QuelozE"][queloz_elodie_nondets_photbins], 
        marker="8", color="grey", ls="", alpha=0.3)
    axes[1, 0].errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_apogee], 
        pleiades["VSINI"][queloz_coravel_dets_apogee], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_dets_apogee], 
        xerr=pleiades["vsini_err_QuelozC"][queloz_coravel_dets_apogee], 
        marker="o", color=bc.red, ls="")
    axes[1, 0].errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_photbins],
        pleiades["VSINI"][queloz_coravel_dets_photbins], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_dets_photbins], 
        xerr=pleiades["vsini_err_QuelozC"][queloz_coravel_dets_photbins], 
        marker="8", color="red", ls="")
    axes[1, 0].errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_photbins],
        pleiades["VSINI"][queloz_coravel_nondets_photbins], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_nondets_photbins], 
        xerr=pleiades["vsini_err_QuelozC"][queloz_coravel_nondets_photbins], 
        marker="8", color="red", ls="", alpha=0.3)
    axes[1, 0].errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_upper_single], 
        pleiades["VSINI"][queloz_coravel_dets_upper_single], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_dets_upper_single], xerr=0,
        marker="<", color=bc.red, ls="")
    axes[1, 0].errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_upper_single], 
        pleiades["VSINI"][queloz_coravel_nondets_upper_single], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_nondets_upper_single], xerr=0,
        marker="<", color="grey", ls="", alpha=0.3)
    axes[1, 0].errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_lower_single], 
        pleiades["VSINI"][queloz_coravel_dets_lower_single], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_dets_lower_single], 
        xerr=pleiades["vsini_err_QuelozC"][queloz_coravel_dets_lower_single], 
        marker=">", color=bc.red, ls="")
    axes[1, 0].errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_lower_single], 
        pleiades["VSINI"][queloz_coravel_nondets_lower_single], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_nondets_lower_single], 
        xerr=pleiades["vsini_err_QuelozC"][queloz_coravel_nondets_lower_single], 
        marker=">", color="grey", ls="", alpha=0.3)
    axes[1, 0].errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_upper_photbin], 
        pleiades["VSINI"][queloz_coravel_dets_upper_photbin], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_dets_upper_photbin],  xerr=0, 
        marker="<", color='red', ls="")
    axes[1, 0].errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_upper_photbin], 
        pleiades["VSINI"][queloz_coravel_nondets_upper_photbin], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_nondets_upper_photbin], xerr=0,
        marker="<", color="red", ls="", alpha=0.3)
    axes[1, 0].errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_dets_lower_photbin], 
        pleiades["VSINI"][queloz_coravel_dets_lower_photbin], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_dets_lower_photbin],  xerr=0,
        marker=">", color='red', ls="")
    axes[1, 0].errorbar(
        pleiades["vsini_QuelozC"][queloz_coravel_nondets_lower_photbin], 
        pleiades["VSINI"][queloz_coravel_nondets_lower_photbin], 
        yerr=pleiades["VSINI_ERR"][queloz_coravel_nondets_lower_photbin],  xerr=0,
        marker=">", color="red", ls="", alpha=0.3)
    axes[1, 0].plot([1, 100], [1, 100], 'k-')

    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlim(1, 100)
    axes[1, 0].set_ylim(1, 100)
    axes[1, 0].set_xlabel("Queloz Vsini")
    axes[1, 0].set_ylabel("APOGEE Vsini")


    # Plot Terndrup targets in the second panel
    terndrup = ~pleiades["vsini_Terndrup"].mask
    terndrup_detections = pleiades["vsini_lim_Terndrup"] == "d"
    terndrup_limits = pleiades["vsini_lim_Terndrup"] == "u"

    terndrup_vdiff = (
        np.log10(pleiades["VSINI"]) - np.log10(pleiades["vsini_Terndrup"]))

    terndrup_photbin = np.logical_and(terndrup_detections, phot_bin)
    terndrup_singles = np.logical_and(terndrup_detections, ~phot_bin)
    terndrup_lim_photbin = np.logical_and(terndrup_limits, phot_bin)
    terndrup_lim_singles = np.logical_and(terndrup_limits, ~phot_bin)

    terndrup_all_photbin = au.multi_logical_or(
        terndrup_photbin, terndrup_lim_photbin)
    terndrup_all_singles = au.multi_logical_or(
        terndrup_singles, terndrup_lim_singles)

    # Mark Terndrup detections as APOGEE detections
    terndrup_dets_apogee = np.logical_and(terndrup_singles, apogee_dets)
    terndrup_nondets_apogee = np.logical_and(terndrup_singles, ~apogee_dets)
    # Mark Terndrup photometric binaries as APOGEE detections
    terndrup_dets_photbin = np.logical_and(terndrup_photbin, apogee_dets)
    terndrup_nondets_photbin = np.logical_and(terndrup_photbin, ~apogee_dets)
    # Mark Terndrup upper limits as APOGEE detections
    terndrup_dets_limits = np.logical_and(terndrup_lim_singles, apogee_dets)
    terndrup_nondets_limits = np.logical_and(terndrup_lim_singles, ~apogee_dets)
    # Mark terndrup upper limit photometric binaries as APOGEE detections
    terndrup_dets_lim_photbin = np.logical_and(terndrup_lim_photbin, apogee_dets)
    terndrup_nondets_lim_photbin = np.logical_and(terndrup_lim_photbin, ~apogee_dets)

    # Now plot the Terndrup Points
    hr.absmag_teff_plot(
        pleiades["TEFF"][terndrup_all_singles],
        pleiades["Dereddened K"][terndrup_all_singles],
        yerr=pleiades["K_ERR"][terndrup_all_singles],
        xerr=pleiades["TEFF_ERR"][terndrup_all_singles], marker="o",
        color='blue', ls="", axis=axes[0, 1])
    hr.absmag_teff_plot(
        pleiades["TEFF"][terndrup_all_photbin],
        pleiades["Dereddened K"][terndrup_all_photbin],
        yerr=pleiades["K_ERR"][terndrup_all_photbin],
        xerr=pleiades["TEFF_ERR"][terndrup_all_photbin], marker="8",
        color='r', ls="", axis=axes[0, 1])

    axes[0, 1].set_xlim(6750, 3500)
    axes[0, 1].set_ylim(11.7, 7)
    axes[0, 1].set_title("Terndrup et al (2001)")
    axes[0, 1].set_xlabel(Teffstr)
    axes[0, 1].set_ylabel("Dereddened $K_S$")

    axes[1, 1].errorbar(
        pleiades["vsini_Terndrup"][terndrup_dets_apogee], 
        pleiades["VSINI"][terndrup_dets_apogee], 
        yerr=pleiades["VSINI_ERR"][terndrup_dets_apogee], 
        xerr=pleiades["vsini_err_Terndrup"][terndrup_dets_apogee], 
        marker="o", color="blue", ls="")
    axes[1, 1].errorbar(
        pleiades["vsini_Terndrup"][terndrup_dets_photbin],
        pleiades["VSINI"][terndrup_dets_photbin], 
        yerr=pleiades["VSINI_ERR"][terndrup_dets_photbin], 
        xerr=pleiades["vsini_err_Terndrup"][terndrup_dets_photbin], 
        marker="8", color="red", ls="")
    axes[1, 1].errorbar(
        pleiades["vsini_Terndrup"][terndrup_nondets_apogee], 
        pleiades["VSINI"][terndrup_nondets_apogee], 
        yerr=pleiades["VSINI_ERR"][terndrup_nondets_apogee], 
        xerr=pleiades["vsini_err_Terndrup"][terndrup_nondets_apogee], 
        marker="o", color="grey", ls="", alpha=0.3)
    axes[1, 1].errorbar(
        pleiades["vsini_Terndrup"][terndrup_nondets_photbin],
        pleiades["VSINI"][terndrup_nondets_photbin], 
        yerr=pleiades["VSINI_ERR"][terndrup_nondets_photbin], 
        xerr=pleiades["vsini_err_Terndrup"][terndrup_nondets_photbin], 
        marker="8", color="red", ls="", alpha=0.3)
    axes[1, 1].errorbar(
        pleiades["vsini_Terndrup"][terndrup_dets_limits], 
        pleiades["VSINI"][terndrup_dets_limits], 
        yerr=pleiades["VSINI_ERR"][terndrup_dets_limits],  xerr=0,
        marker="<", color="blue", ls="")
    axes[1, 1].errorbar(
        pleiades["vsini_Terndrup"][terndrup_dets_lim_photbin],
        pleiades["VSINI"][terndrup_dets_lim_photbin], 
        yerr=pleiades["VSINI_ERR"][terndrup_dets_lim_photbin],  xerr=0,
        marker="<", color="red", ls="")
    axes[1, 1].errorbar(
        pleiades["vsini_Terndrup"][terndrup_nondets_limits], 
        pleiades["VSINI"][terndrup_nondets_limits], 
        yerr=pleiades["VSINI_ERR"][terndrup_nondets_limits],  xerr=0,
        marker="<", color="grey", ls="", alpha=0.3)
    axes[1, 1].errorbar(
        pleiades["vsini_Terndrup"][terndrup_nondets_lim_photbin],
        pleiades["VSINI"][terndrup_nondets_lim_photbin], 
        yerr=pleiades["VSINI_ERR"][terndrup_nondets_lim_photbin],  xerr=0,
        marker="<", color="red", ls="", alpha=0.3)
    axes[1, 1].plot([1, 100], [1, 100], 'k-')

    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlim(1, 100)
    axes[1, 1].set_ylim(1, 100)
    axes[1, 1].set_xlabel("Terndrup Vsini")


    # Now plot Soderblom points
    soderblom = ~pleiades["vsini_Soderblom"].mask
    soderblom_detections = pleiades["vsini_lim_Soderblom"] == "d"
    soderblom_limits = pleiades["vsini_lim_Soderblom"] == "u"

    soderblom_det_photbins = np.logical_and(soderblom_detections, phot_bin)
    soderblom_det_singles = np.logical_and(soderblom_detections, ~phot_bin)
    soderblom_lim_photbins = np.logical_and(soderblom_limits, phot_bin)
    soderblom_lim_singles = np.logical_and(soderblom_limits, ~phot_bin)

    soderblom_all_photbins = au.multi_logical_or(
        soderblom_det_photbins, soderblom_lim_photbins)
    soderblom_all_singles = au.multi_logical_or(
        soderblom_det_singles, soderblom_lim_singles)

    # Mark Soderblom detections as APOGEE detections
    soderblom_dets_apogee = np.logical_and(soderblom_det_singles, apogee_dets)
    soderblom_nondets_apogee = np.logical_and(soderblom_det_singles, ~apogee_dets)
    # Mark Soderblom photometric binaries as APOGEE detections
    soderblom_dets_photbin = np.logical_and(soderblom_det_photbins, apogee_dets)
    soderblom_nondets_photbin = np.logical_and(soderblom_det_photbins, ~apogee_dets)
    # Mark Soderblom limits as APOGEE detections
    soderblom_dets_limits = np.logical_and(soderblom_lim_singles, apogee_dets)
    soderblom_nondets_limits = np.logical_and(soderblom_lim_singles, ~apogee_dets)
    # Mark Soderblom photometric binary limits as APOGEE detections
    soderblom_dets_lim_photbin = np.logical_and(soderblom_lim_photbins, apogee_dets)
    soderblom_nondets_lim_photbin = np.logical_and(soderblom_lim_photbins, ~apogee_dets)

    # Now plot the Soderblom Points
    hr.absmag_teff_plot(
        pleiades["TEFF"][soderblom_all_singles],
        pleiades["Dereddened K"][soderblom_all_singles],
        yerr=pleiades["K_ERR"][soderblom_all_singles],
        xerr=pleiades["TEFF_ERR"][soderblom_all_singles], marker="o",
        color=bc.violet, ls="", axis=axes[0, 2])
    hr.absmag_teff_plot(
        pleiades["TEFF"][soderblom_all_photbins],
        pleiades["Dereddened K"][soderblom_all_photbins],
        yerr=pleiades["K_ERR"][soderblom_all_photbins],
        xerr=pleiades["TEFF_ERR"][soderblom_all_photbins], marker="8",
        color='r', ls="", axis=axes[0, 2])

    axes[0, 2].set_xlim(6750, 3500)
    axes[0, 2].set_ylim(11.7, 7)
    axes[0, 2].set_title("Soderblom et al (1993)")
    axes[0, 2].set_xlabel(Teffstr)
    axes[0, 2].set_ylabel("Dereddened $K_S$")

    axes[1, 2].errorbar(
        pleiades["vsini_Soderblom"][soderblom_dets_apogee], 
        pleiades["VSINI"][soderblom_dets_apogee], 
        yerr=pleiades["VSINI_ERR"][soderblom_dets_apogee],  xerr=0, marker="o", 
        color=bc.violet, ls="")
    axes[1, 2].errorbar(
        pleiades["vsini_Soderblom"][soderblom_dets_photbin],
        pleiades["VSINI"][soderblom_dets_photbin], 
        yerr=pleiades["VSINI_ERR"][soderblom_dets_photbin],  xerr=0, marker="8", 
        color="red", ls="")
    axes[1, 2].errorbar(
        pleiades["vsini_Soderblom"][soderblom_nondets_apogee], 
        pleiades["VSINI"][soderblom_nondets_apogee], 
        yerr=pleiades["VSINI_ERR"][soderblom_nondets_apogee],  xerr=0, marker="o", 
        color="grey", ls="", alpha=0.3)
    axes[1, 2].errorbar(
        pleiades["vsini_Soderblom"][soderblom_nondets_photbin],
        pleiades["VSINI"][soderblom_nondets_photbin], 
        yerr=pleiades["VSINI_ERR"][soderblom_nondets_photbin],  xerr=0, marker="8", 
        color="red", ls="", alpha=0.3)
    axes[1, 2].errorbar(
        pleiades["vsini_Soderblom"][soderblom_dets_limits], 
        pleiades["VSINI"][soderblom_dets_limits], 
        yerr=pleiades["VSINI_ERR"][soderblom_dets_limits],  xerr=0, marker="<", 
        color=bc.violet, ls="")
    axes[1, 2].errorbar(
        pleiades["vsini_Soderblom"][soderblom_dets_lim_photbin],
        pleiades["VSINI"][soderblom_dets_lim_photbin], 
        yerr=pleiades["VSINI_ERR"][soderblom_dets_lim_photbin],  xerr=0, 
        marker="<", color="red", ls="")
    axes[1, 2].errorbar(
        pleiades["vsini_Soderblom"][soderblom_nondets_limits], 
        pleiades["VSINI"][soderblom_nondets_limits], 
        yerr=pleiades["VSINI_ERR"][soderblom_nondets_limits],  xerr=0, marker="<", 
        color="grey", ls="", alpha=0.3)
    axes[1, 2].errorbar(
        pleiades["vsini_Soderblom"][soderblom_nondets_lim_photbin],
        pleiades["VSINI"][soderblom_nondets_lim_photbin], 
        yerr=pleiades["VSINI_ERR"][soderblom_nondets_lim_photbin],  xerr=0, 
        marker="<", color="red", ls="", alpha=0.3)
    axes[1, 2].plot([1, 100], [1, 100], 'k-')

    axes[1, 2].set_xscale("log")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xlim(1, 100)
    axes[1, 2].set_ylim(1, 100)
    axes[1, 2].set_xlabel("Soderblom Vsini")
    axes[1, 2].set_ylabel("APOGEE Vsini")

    # Now plot Stauffer & Hartmann points.
    sh = ~pleiades["vsini_SH"].mask
    sh_detections = pleiades["vsini_lim_SH"] == "d"
    sh_upper = pleiades["vsini_lim_SH"] == "u"

    sh_det_singles = np.logical_and(sh_detections, ~phot_bin)
    sh_det_photbin = np.logical_and(sh_detections, phot_bin)
    sh_lim_singles = np.logical_and(sh_upper, ~phot_bin)
    sh_lim_photbin = np.logical_and(sh_upper, phot_bin)
    
    sh_all_photbin = au.multi_logical_or(
        sh_det_photbin, sh_lim_photbin)
    sh_all_singles = au.multi_logical_or(
        sh_det_singles, sh_lim_singles)

    # Mark Stauffer Hartmann detections as APOGEE detections
    sh_dets_apogee = np.logical_and(sh_det_singles, apogee_dets)
    sh_nondets_apogee = np.logical_and(sh_det_singles, ~apogee_dets)
    # Mark Stauffer Hartmann photometric binaries as APOGEE detections
    sh_dets_photbin = np.logical_and(sh_det_photbin, apogee_dets)
    sh_nondets_photbin = np.logical_and(sh_det_photbin, ~apogee_dets)
    # Mark Stauffer Hartmann upper limits as APOGEE detections
    sh_dets_limits = np.logical_and(sh_lim_singles, apogee_dets)
    sh_nondets_limits = np.logical_and(sh_lim_singles, ~apogee_dets)
    # Mark Stauffer Hartmann photometric binary upper limits as APOGEE detections
    sh_dets_lim_photbin = np.logical_and(sh_lim_photbin, apogee_dets)
    sh_nondets_lim_photbin = np.logical_and(sh_lim_photbin, ~apogee_dets)

    # Now plot the Stauffer & Hartmann Points
    hr.absmag_teff_plot(
        pleiades["TEFF"][sh_all_singles],
        pleiades["Dereddened K"][sh_all_singles],
        yerr=pleiades["K_ERR"][sh_all_singles],
        xerr=pleiades["TEFF_ERR"][sh_all_singles], marker="o",
        color=bc.sky_blue, ls="", axis=axes[0, 3])
    hr.absmag_teff_plot(
        pleiades["TEFF"][sh_all_photbin],
        pleiades["Dereddened K"][sh_all_photbin],
        yerr=pleiades["K_ERR"][sh_all_photbin],
        xerr=pleiades["TEFF_ERR"][sh_all_photbin], marker="8",
        color='r', ls="", axis=axes[0, 3])

    axes[0, 3].set_xlim(6750, 3500)
    axes[0, 3].set_ylim(11.7, 7)
    axes[0, 3].set_title("Stauffer & Hartmann (1987)")
    axes[0, 3].set_xlabel(Teffstr)
    axes[0, 3].set_ylabel("Dereddened $K_S$")

    axes[1, 3].errorbar(
        pleiades["vsini_SH"][sh_dets_apogee], 
        pleiades["VSINI"][sh_dets_apogee], 
        yerr=pleiades["VSINI_ERR"][sh_dets_apogee], 
        xerr=pleiades["vsini_err_SH"][sh_dets_apogee], 
        marker="o", color=bc.sky_blue, ls="")
    axes[1, 3].errorbar(
        pleiades["vsini_SH"][sh_dets_photbin],
        pleiades["VSINI"][sh_dets_photbin], 
        yerr=pleiades["VSINI_ERR"][sh_dets_photbin], 
        xerr=pleiades["vsini_err_SH"][sh_dets_photbin], 
        marker="8", color="red", ls="")
    axes[1, 3].errorbar(
        pleiades["vsini_SH"][sh_nondets_apogee], 
        pleiades["VSINI"][sh_nondets_apogee], 
        yerr=pleiades["VSINI_ERR"][sh_nondets_apogee], 
        xerr=pleiades["vsini_err_SH"][sh_nondets_apogee], 
        marker="o", color="grey", ls="", alpha=0.3)
    axes[1, 3].errorbar(
        pleiades["vsini_SH"][sh_nondets_photbin],
        pleiades["VSINI"][sh_nondets_photbin], 
        yerr=pleiades["VSINI_ERR"][sh_nondets_photbin], 
        xerr=pleiades["vsini_err_SH"][sh_nondets_photbin], 
        marker="8", color="red", ls="", alpha=0.3)
    axes[1, 3].errorbar(
        pleiades["vsini_SH"][sh_dets_limits], 
        pleiades["VSINI"][sh_dets_limits], 
        yerr=pleiades["VSINI_ERR"][sh_dets_limits],  xerr=0,
        marker="<", color=bc.sky_blue, ls="")
    axes[1, 3].errorbar(
        pleiades["vsini_SH"][sh_dets_lim_photbin],
        pleiades["VSINI"][sh_dets_lim_photbin], 
        yerr=pleiades["VSINI_ERR"][sh_dets_lim_photbin],  xerr=0,
        marker="<", color="red", ls="")
    axes[1, 3].errorbar(
        pleiades["vsini_SH"][sh_nondets_limits], 
        pleiades["VSINI"][sh_nondets_limits], 
        yerr=pleiades["VSINI_ERR"][sh_nondets_limits],  xerr=0,
        marker="<", color="grey", ls="", alpha=0.3)
    axes[1, 3].errorbar(
        pleiades["vsini_SH"][sh_nondets_lim_photbin],
        pleiades["VSINI"][sh_nondets_lim_photbin], 
        yerr=pleiades["VSINI_ERR"][sh_nondets_lim_photbin],  xerr=0,
        marker="<", color="red", ls="", alpha=0.3)
    axes[1, 3].plot([1, 100], [1, 100], 'k-')

    axes[1, 3].set_xscale("log")
    axes[1, 3].set_yscale("log")
    axes[1, 3].set_xlim(1, 100)
    axes[1, 3].set_ylim(1, 100)
    axes[1, 3].set_xlabel("Stauffer & Hartmann Vsini")
    axes[1, 3].set_ylabel("APOGEE Vsini")

    # Lastly Stauffer points.
    s84 = ~pleiades["vsini_S84"].mask
    s84_detections = pleiades["vsini_lim_S84"] == "d"
    s84_upper = pleiades["vsini_lim_S84"] == "u"

    s84_det_singles = np.logical_and(s84_detections, ~phot_bin)
    s84_det_photbin = np.logical_and(s84_detections, phot_bin)
    s84_lim_singles = np.logical_and(s84_upper, ~phot_bin)
    s84_lim_photbin = np.logical_and(s84_upper, phot_bin)

    s84_all_photbin = au.multi_logical_or(
        s84_det_photbin, s84_lim_photbin)
    s84_all_singles = au.multi_logical_or(
        s84_det_singles, s84_lim_singles)

    # Mark Stauffer detections as APOGEE detections
    s84_dets_apogee = np.logical_and(s84_det_singles, apogee_dets)
    s84_nondets_apogee = np.logical_and(s84_det_singles, ~apogee_dets)
    # Mark Stauffer photometric binaries as APOGEE detections
    s84_dets_photbin = np.logical_and(s84_det_photbin, apogee_dets)
    s84_nondets_photbin = np.logical_and(s84_det_photbin, ~apogee_dets)
    # Mark Stauffer upper limits as APOGEE detections
    s84_dets_limits = np.logical_and(s84_lim_singles, apogee_dets)
    s84_nondets_limits = np.logical_and(s84_lim_singles, ~apogee_dets)
    # Mark Stauffer photometric binary upper limits as APOGEE detections
    s84_dets_lim_photbin = np.logical_and(s84_lim_photbin, apogee_dets)
    s84_nondets_lim_photbin = np.logical_and(s84_lim_photbin, ~apogee_dets)

    # Now plot the Stauffer & Hartmann Points
    hr.absmag_teff_plot(
        pleiades["TEFF"][s84_all_singles],
        pleiades["Dereddened K"][s84_all_singles],
        yerr=pleiades["K_ERR"][s84_all_singles],
        xerr=pleiades["TEFF_ERR"][s84_all_singles], marker="o",
        color=bc.algae, ls="", axis=axes[0, 4])
    hr.absmag_teff_plot(
        pleiades["TEFF"][s84_all_photbin],
        pleiades["Dereddened K"][s84_all_photbin],
        yerr=pleiades["K_ERR"][s84_all_photbin],
        xerr=pleiades["TEFF_ERR"][s84_all_photbin], marker="8",
        color='r', ls="", axis=axes[0, 4])

    axes[0, 4].set_xlim(6750, 3500)
    axes[0, 4].set_ylim(11.7, 7)
    axes[0, 4].set_title("Stauffer (1984)")
    axes[0, 4].set_xlabel(Teffstr)
    axes[0, 4].set_ylabel("Dereddened $K_S$")

    axes[1, 4].errorbar(
        pleiades["vsini_S84"][s84_dets_apogee], 
        pleiades["VSINI"][s84_dets_apogee], 
        yerr=pleiades["VSINI_ERR"][s84_dets_apogee],  
        xerr=pleiades["vsini_err_S84"][s84_dets_apogee], 
        marker="o", color=bc.algae, ls="")
    axes[1, 4].errorbar(
        pleiades["vsini_S84"][s84_dets_photbin],
        pleiades["VSINI"][s84_dets_photbin], 
        yerr=pleiades["VSINI_ERR"][s84_dets_photbin],  
        xerr=pleiades["vsini_err_S84"][s84_dets_photbin], 
        marker="8", color="red", ls="")
    axes[1, 4].errorbar(
        pleiades["vsini_S84"][s84_nondets_apogee], 
        pleiades["VSINI"][s84_nondets_apogee], 
        yerr=pleiades["VSINI_ERR"][s84_nondets_apogee],  
        xerr=pleiades["vsini_err_S84"][s84_nondets_apogee], 
        marker="o", color="grey", ls="", alpha=0.3)
    axes[1, 4].errorbar(
        pleiades["vsini_S84"][s84_nondets_photbin],
        pleiades["VSINI"][s84_nondets_photbin], 
        yerr=pleiades["VSINI_ERR"][s84_nondets_photbin],  
        xerr=pleiades["vsini_err_S84"][s84_nondets_photbin], 
        marker="8", color="red", ls="", alpha=0.3)
    axes[1, 4].errorbar(
        pleiades["vsini_S84"][s84_dets_limits], 
        pleiades["VSINI"][s84_dets_limits], 
        yerr=pleiades["VSINI_ERR"][s84_dets_limits],  xerr=0,
        marker="<", color=bc.algae, ls="")
    axes[1, 4].errorbar(
        pleiades["vsini_S84"][s84_dets_lim_photbin],
        pleiades["VSINI"][s84_dets_lim_photbin], 
        yerr=pleiades["VSINI_ERR"][s84_dets_lim_photbin],  xerr=0,
        marker="<", color="red", ls="")
    axes[1, 4].errorbar(
        pleiades["vsini_S84"][s84_nondets_limits], 
        pleiades["VSINI"][s84_nondets_limits], 
        yerr=pleiades["VSINI_ERR"][s84_nondets_limits],  xerr=0,
        marker="<", color="grey", ls="", alpha=0.3)
    axes[1, 4].errorbar(
        pleiades["vsini_S84"][s84_nondets_lim_photbin],
        pleiades["VSINI"][s84_nondets_lim_photbin], 
        yerr=pleiades["VSINI_ERR"][s84_nondets_lim_photbin],  xerr=0, 
        marker="<", color="red", ls="", alpha=0.3)
    axes[1, 4].plot([1, 100], [1, 100], 'k-')

    axes[1, 4].set_xscale("log")
    axes[1, 4].set_yscale("log")
    axes[1, 4].set_xlim(1, 100)
    axes[1, 4].set_ylim(1, 100)
    axes[1, 4].set_xlabel("Stauffer Vsini")
    axes[1, 4].set_ylabel("APOGEE Vsini")

    # Another round of points from Jackson and Jeffries
    jackson = ~pleiades["vsini_Jackson"].mask
    jackson_detections = pleiades["vsini_lim_Jackson"] == "d"
    jackson_upper = pleiades["vsini_lim_Jackson"] == "u"

    jackson_det_singles = np.logical_and(jackson_detections, ~phot_bin)
    jackson_det_photbin = np.logical_and(jackson_detections, phot_bin)
    jackson_lim_singles = np.logical_and(jackson_upper, ~phot_bin)
    jackson_lim_photbin = np.logical_and(jackson_upper, phot_bin)

    jackson_all_photbin = au.multi_logical_or(
        jackson_det_photbin, jackson_lim_photbin)
    jackson_all_singles = au.multi_logical_or(
        jackson_det_singles, jackson_lim_singles)

    # Mark Stauffer detections as APOGEE detections
    jackson_dets_apogee = np.logical_and(jackson_det_singles, apogee_dets)
    jackson_nondets_apogee = np.logical_and(jackson_det_singles, ~apogee_dets)
    # Mark Stauffer photometric binaries as APOGEE detections
    jackson_dets_photbin = np.logical_and(jackson_det_photbin, apogee_dets)
    jackson_nondets_photbin = np.logical_and(jackson_det_photbin, ~apogee_dets)
    # Mark Stauffer upper limits as APOGEE detections
    jackson_dets_limits = np.logical_and(jackson_lim_singles, apogee_dets)
    jackson_nondets_limits = np.logical_and(jackson_lim_singles, ~apogee_dets)
    # Mark Stauffer photometric binary upper limits as APOGEE detections
    jackson_dets_lim_photbin = np.logical_and(jackson_lim_photbin, apogee_dets)
    jackson_nondets_lim_photbin = np.logical_and(jackson_lim_photbin, ~apogee_dets)

    # Now plot the Stauffer & Hartmann Points
    hr.absmag_teff_plot(
        pleiades["TEFF"][jackson_all_singles],
        pleiades["Dereddened K"][jackson_all_singles],
        yerr=pleiades["K_ERR"][jackson_all_singles],
        xerr=pleiades["TEFF_ERR"][jackson_all_singles], marker="o",
        color=bc.green, ls="", axis=axes[0, 5])
    hr.absmag_teff_plot(
        pleiades["TEFF"][jackson_all_photbin],
        pleiades["Dereddened K"][jackson_all_photbin],
        yerr=pleiades["K_ERR"][jackson_all_photbin],
        xerr=pleiades["TEFF_ERR"][jackson_all_photbin], marker="8",
        color='r', ls="", axis=axes[0, 5])

    axes[0, 5].set_xlim(6750, 3500)
    axes[0, 5].set_ylim(11.7, 7)
    axes[0, 5].set_title("Jackson et al. (2018)")
    axes[0, 5].set_xlabel(Teffstr)
    axes[0, 5].set_ylabel("Dereddened $K_S$")

    axes[1, 5].errorbar(
        pleiades["vsini_Jackson"][jackson_dets_apogee], 
        pleiades["VSINI"][jackson_dets_apogee], 
        yerr=pleiades["VSINI_ERR"][jackson_dets_apogee],  
        xerr=pleiades["vsini_err_Jackson"][jackson_dets_apogee], 
        marker="o", color=bc.green, ls="")
    axes[1, 5].errorbar(
        pleiades["vsini_Jackson"][jackson_dets_photbin],
        pleiades["VSINI"][jackson_dets_photbin], 
        yerr=pleiades["VSINI_ERR"][jackson_dets_photbin],  
        xerr=pleiades["vsini_err_Jackson"][jackson_dets_photbin], 
        marker="8", color="red", ls="")
    axes[1, 5].errorbar(
        pleiades["vsini_Jackson"][jackson_nondets_apogee], 
        pleiades["VSINI"][jackson_nondets_apogee], 
        yerr=pleiades["VSINI_ERR"][jackson_nondets_apogee],  
        xerr=pleiades["vsini_err_Jackson"][jackson_nondets_apogee], 
        marker="o", color="grey", ls="", alpha=0.3)
    axes[1, 5].errorbar(
        pleiades["vsini_Jackson"][jackson_nondets_photbin],
        pleiades["VSINI"][jackson_nondets_photbin], 
        yerr=pleiades["VSINI_ERR"][jackson_nondets_photbin],  
        xerr=pleiades["vsini_err_Jackson"][jackson_nondets_photbin], 
        marker="8", color="red", ls="", alpha=0.3)
    axes[1, 5].errorbar(
        pleiades["vsini_Jackson"][jackson_dets_limits], 
        pleiades["VSINI"][jackson_dets_limits], 
        yerr=pleiades["VSINI_ERR"][jackson_dets_limits],  xerr=0,
        marker="<", color=bc.green, ls="")
    axes[1, 5].errorbar(
        pleiades["vsini_Jackson"][jackson_dets_lim_photbin],
        pleiades["VSINI"][jackson_dets_lim_photbin], 
        yerr=pleiades["VSINI_ERR"][jackson_dets_lim_photbin],  xerr=0,
        marker="<", color="red", ls="")
    axes[1, 5].errorbar(
        pleiades["vsini_Jackson"][jackson_nondets_limits], 
        pleiades["VSINI"][jackson_nondets_limits], 
        yerr=pleiades["VSINI_ERR"][jackson_nondets_limits],  xerr=0,
        marker="<", color="grey", ls="", alpha=0.3)
    axes[1, 5].errorbar(
        pleiades["vsini_Jackson"][jackson_nondets_lim_photbin],
        pleiades["VSINI"][jackson_nondets_lim_photbin], 
        yerr=pleiades["VSINI_ERR"][jackson_nondets_lim_photbin],  xerr=0, 
        marker="<", color="red", ls="", alpha=0.3)
    axes[1, 5].plot([1, 100], [1, 100], 'k-')

    axes[1, 5].set_xscale("log")
    axes[1, 5].set_yscale("log")
    axes[1, 5].set_xlim(1, 100)
    axes[1, 5].set_ylim(1, 100)
    axes[1, 5].set_xlabel("Jackson Vsini")
    axes[1, 5].set_ylabel("APOGEE Vsini")

    plt.tight_layout()

@write_plot("Zeropoints")
def Pleiades_zero_point_comparison():
    '''Compare APOGEE and literature vsini for different surveys.'''
    pleiades = cache.pleiades_APOGEE_Literature_vsini()
    phot_bin = np.logical_or(
        np.logical_and(pleiades["(V-K)0"] <= 5, pleiades["Delmag"] >= 0.3),
        np.logical_and(pleiades["(V-K)0"] > 5, pleiades["Delmag"] >= 0.3))
    apogee_dets = pleiades["VSINI"] > 10

    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        1, 5, figsize=(80, 12), sharey=True)
    
    # Plot the Queloz targets in the first panel.
    queloz_elodie = ~pleiades["vsini_QuelozE"].mask
    queloz_coravel = ~pleiades["vsini_QuelozC"].mask
    full_queloz = np.logical_or(queloz_elodie, queloz_coravel)
    queloz_coravel_upper = pleiades["vsini_lim_QuelozC"] == "<"
    queloz_coravel_lower = pleiades["vsini_lim_QuelozC"] == ">"
    queloz_coravel_det = np.logical_not(
        np.logical_or(queloz_coravel_upper,queloz_coravel_lower))

    elodie_vdiff = (
        np.log10(pleiades["VSINI"]) - np.log10(pleiades["vsini_QuelozE"])) 
    coravel_vdiff = (
        np.log10(pleiades["VSINI"]) - np.log10(pleiades["vsini_QuelozC"])) 

    queloz_elodie_photbin = np.logical_and(queloz_elodie, phot_bin)
    queloz_elodie_single = np.logical_and(queloz_elodie, ~phot_bin)
    queloz_coravel_photbin = np.logical_and(queloz_coravel, phot_bin)
    queloz_coravel_single = np.logical_and(queloz_coravel, ~phot_bin)

    # Split the Queloz detections into APOGEE detections and nondetections.
    queloz_elodie_dets_apogee = np.logical_and(
        queloz_elodie_single, apogee_dets)
    queloz_elodie_nondets_apogee = np.logical_and(
        queloz_elodie_single, ~apogee_dets)
    queloz_coravel_dets_apogee = np.logical_and(
        queloz_coravel_single, apogee_dets)
    queloz_coravel_nondets_apogee = np.logical_and(
        queloz_coravel_single, ~apogee_dets)
    # Split the queloz photometric binaries into APOGEE detections and nondetections.
    queloz_elodie_dets_photbins = np.logical_and(
        queloz_elodie_photbin, apogee_dets)
    queloz_elodie_nondets_photbins = np.logical_and(
        queloz_elodie_photbin, ~apogee_dets)
    queloz_coravel_dets_photbins = np.logical_and(
        queloz_coravel_photbin, apogee_dets)
    queloz_coravel_nondets_photbins = np.logical_and(
        queloz_coravel_photbin, ~apogee_dets)
    # Split the queloz upper limits into APOGEE detections and nondetections.
    queloz_coravel_dets_uppers = np.logical_and(
        queloz_coravel_upper, apogee_dets)
    queloz_coravel_nondets_uppers = np.logical_and(
        queloz_coravel_upper, ~apogee_dets)

    ax1.errorbar(
        pleiades["VSINI"][queloz_elodie_dets_apogee],
        elodie_vdiff[queloz_elodie_dets_apogee],
        yerr=0.20/np.log(10), 
        xerr=0.15*pleiades["VSINI"][queloz_elodie_dets_apogee], 
        marker="o", color=bc.red, ls="")
    ax1.errorbar(
        pleiades["VSINI"][queloz_elodie_nondets_apogee],
        elodie_vdiff[queloz_elodie_nondets_apogee],
        yerr=0.20/np.log(10), 
        xerr=0.15*pleiades["VSINI"][queloz_elodie_nondets_apogee], 
        marker="o", color="grey", ls="", alpha=0.3)
    ax1.errorbar(
        pleiades["VSINI"][queloz_elodie_dets_photbins],
        elodie_vdiff[queloz_elodie_dets_photbins],
        yerr=0.20/np.log(10), 
        xerr=0.15*pleiades["VSINI"][queloz_elodie_dets_photbins], 
        marker="8", color="red", ls="")
    ax1.errorbar(
        pleiades["VSINI"][queloz_elodie_nondets_photbins],
        elodie_vdiff[queloz_elodie_nondets_photbins],
        yerr=0.20/np.log(10), 
        xerr=0.15*pleiades["VSINI"][queloz_elodie_nondets_photbins], 
        marker="8", color="grey", ls="")
    ax1.errorbar(
        pleiades["VSINI"][queloz_coravel_dets_apogee], 
        coravel_vdiff[queloz_coravel_dets_apogee],
        yerr=0.20/np.log(10), 
        xerr=0.15*pleiades["VSINI"][queloz_coravel_dets_apogee], 
        marker="o", color=bc.red, ls="")
    ax1.errorbar(
        pleiades["VSINI"][queloz_coravel_dets_photbins],
        coravel_vdiff[queloz_coravel_dets_photbins],
        yerr=0.20/np.log(10), 
        xerr=0.15*pleiades["VSINI"][queloz_coravel_dets_photbins], 
        marker="8", color="red", ls="")
    ax1.errorbar(
        pleiades["VSINI"][queloz_coravel_nondets_photbins],
        coravel_vdiff[queloz_coravel_nondets_photbins],
        yerr=0.20/np.log(10), 
        xerr=0.15*pleiades["VSINI"][queloz_coravel_nondets_photbins], 
        marker="8", color="grey", ls="")
    ax1.errorbar(
        pleiades["VSINI"][queloz_coravel_dets_uppers], 
        coravel_vdiff[queloz_coravel_dets_uppers],
        yerr=0, xerr=0.15*pleiades["VSINI"][queloz_coravel_dets_uppers], 
        marker="^", color=bc.red, ls="")
    ax1.errorbar(
        pleiades["VSINI"][queloz_coravel_nondets_uppers], 
        coravel_vdiff[queloz_coravel_nondets_uppers],
        yerr=0, xerr=0.15*pleiades["VSINI"][queloz_coravel_nondets_uppers], 
        marker="^", color="grey", ls="")
    ax1.errorbar(
        pleiades["VSINI"][queloz_coravel_lower], 
        coravel_vdiff[queloz_coravel_lower],
        yerr=0, xerr=0.15*pleiades["VSINI"][queloz_coravel_lower], marker="v",
        color=bc.red, ls="")
    ax1.plot([1, 100], [0, 0], 'k-')

    ax1.set_xscale("log")
    ax1.set_yscale("linear")
    ax1.set_xlim(1, 100)
    ax1.set_xlabel("APOGEE vsini")
    ax1.set_ylabel("log(APOGEE / Queloz)")
    ax1.set_title("Queloz measurements")

    # Plot Terndrup targets in the second panel
    terndrup = ~pleiades["vsini_Terndrup"].mask
    terndrup_detections = pleiades["vsini_lim_Terndrup"] == "d"
    terndrup_limits = pleiades["vsini_lim_Terndrup"] == "u"

    terndrup_vdiff = (
        np.log10(pleiades["VSINI"]) - np.log10(pleiades["vsini_Terndrup"]))

    terndrup_photbin = np.logical_and(terndrup_detections, phot_bin)
    terndrup_singles = np.logical_and(terndrup_detections, ~phot_bin)
    print(pleiades["vsini_lim_Terndrup"][terndrup_detections])

    # Mark Terndrup detections as APOGEE detections
    terndrup_dets_apogee = np.logical_and(terndrup_singles, apogee_dets)
    terndrup_nondets_apogee = np.logical_and(terndrup_singles, ~apogee_dets)
    # Mark Terndrup photometric binaries as APOGEE detections
    terndrup_dets_photbin = np.logical_and(terndrup_photbin, apogee_dets)
    terndrup_nondets_photbin = np.logical_and(terndrup_photbin, ~apogee_dets)
    # Mark Terndrup upper limits as APOGEE detections
    terndrup_dets_limits = np.logical_and(terndrup_limits, apogee_dets)
    terndrup_nondets_limits = np.logical_and(terndrup_limits, ~apogee_dets)

    # Now plot the terndrup points.
    ax2.errorbar(
        pleiades["VSINI"][terndrup_dets_apogee], 
        terndrup_vdiff[terndrup_dets_apogee], yerr=0.20/np.log(10), 
        xerr=0.15*pleiades["VSINI"][terndrup_dets_apogee], marker="o", 
        color="blue", ls="")
    ax2.errorbar(
        pleiades["VSINI"][terndrup_nondets_apogee], 
        terndrup_vdiff[terndrup_nondets_apogee], yerr=0.20/np.log(10), 
        xerr=0.15*pleiades["VSINI"][terndrup_nondets_apogee], marker="o", 
        color="grey", ls="", alpha=0.3)
    ax2.errorbar(
        pleiades["VSINI"][terndrup_dets_photbin], 
        terndrup_vdiff[terndrup_dets_photbin], yerr=0.20/np.log(10), 
        xerr=0.15*pleiades["VSINI"][terndrup_dets_photbin], marker="8", 
        color="red", ls="")
    ax2.errorbar(
        pleiades["VSINI"][terndrup_nondets_photbin], 
        terndrup_vdiff[terndrup_nondets_photbin], yerr=0.20/np.log(10), 
        xerr=0.15*pleiades["VSINI"][terndrup_nondets_photbin], marker="8", 
        color="grey", ls="", alpha=0.3)
    ax2.errorbar(
        pleiades["VSINI"][terndrup_dets_limits], 
        terndrup_vdiff[terndrup_dets_limits], yerr=0,
        xerr=0.15*pleiades["VSINI"][terndrup_dets_limits], marker="^", 
        color="blue", ls="")
    ax2.errorbar(
        pleiades["VSINI"][terndrup_nondets_limits], 
        terndrup_vdiff[terndrup_nondets_limits], yerr=0,
        xerr=0.15*pleiades["VSINI"][terndrup_nondets_limits], marker="^", 
        color="grey", ls="", alpha=0.3)
    ax2.plot([1, 100], [0, 0], 'k-')

    ax2.set_xscale("log")
    ax2.set_xlim(1, 100)
    ax2.set_xlabel("APOGEE vsini")
    ax2.set_ylabel("log(APOGEE / Terndrup)")
    ax2.set_title("Terndrup measurements")

    # Now plot Soderblom points
    soderblom = ~pleiades["vsini_Soderblom"].mask

    soderblom_vdiff = (
        np.log10(pleiades["VSINI"]) - np.log10(pleiades["vsini_Soderblom"]))

    soderblom_photbins = np.logical_and(soderblom, phot_bin)
    soderblom_singles = np.logical_and(soderblom, ~phot_bin)

    # Mark Soderblom detections as APOGEE detections
    soderblom_dets_apogee = np.logical_and(soderblom_singles, apogee_dets)
    soderblom_nondets_apogee = np.logical_and(soderblom_singles, ~apogee_dets)
    # Mark Soderblom photometric binaries as APOGEE detections
    soderblom_dets_photbins = np.logical_and(soderblom_photbins, apogee_dets)
    soderblom_nondets_photbins = np.logical_and(soderblom_photbins, ~apogee_dets)

    ax3.errorbar(
        pleiades["VSINI"][soderblom_dets_apogee],
        soderblom_vdiff[soderblom_dets_apogee], yerr=0.20/np.log(10),
        xerr=0.15*pleiades["VSINI"][soderblom_dets_apogee], marker="o",
        color=bc.violet, ls=""), 
    ax3.errorbar(
        pleiades["VSINI"][soderblom_nondets_apogee],
        soderblom_vdiff[soderblom_nondets_apogee], yerr=0.20/np.log(10),
        xerr=0.15*pleiades["VSINI"][soderblom_nondets_apogee], marker="o",
        color="grey", ls="", alpha=0.3), 
    ax3.errorbar(
        pleiades["VSINI"][soderblom_dets_photbins],
        soderblom_vdiff[soderblom_dets_photbins], yerr=0.20/np.log(10),
        xerr=0.15*pleiades["VSINI"][soderblom_dets_photbins], marker="8",
        color="red", ls=""), 
    ax3.errorbar(
        pleiades["VSINI"][soderblom_nondets_photbins],
        soderblom_vdiff[soderblom_nondets_photbins], yerr=0.20/np.log(10),
        xerr=0.15*pleiades["VSINI"][soderblom_nondets_photbins], marker="8",
        color="grey", ls="", alpha=0.3), 
    ax3.plot([1, 100], [0, 0], 'k-')

    ax3.set_xscale("log")
    ax3.set_xlim(1, 100)
    ax3.set_xlabel("APOGEE vsini")
    ax3.set_ylabel("log(APOGEE / Soderblom)")
    ax3.set_title("Soderblom measurements")

    # Now plot Stauffer & Hartmann points.
    sh = ~pleiades["vsini_SH"].mask
    sh_detections = pleiades["vsini_lim_SH"] == "d"
    sh_upper = pleiades["vsini_lim_SH"] == "u"

    sh_vdiff = (
        np.log10(pleiades["VSINI"]) - np.log10(pleiades["vsini_SH"]))

    sh_singles = np.logical_and(sh_detections, ~phot_bin)
    sh_photbins = np.logical_and(sh_detections, phot_bin)

    # Mark Stauffer Hartmann detections as APOGEE detections
    sh_dets_apogee = np.logical_and(sh_singles, apogee_dets)
    sh_nondets_apogee = np.logical_and(sh_singles, ~apogee_dets)
    # Mark Stauffer Hartmann photometric binaries as APOGEE detections
    sh_dets_photbins = np.logical_and(sh_photbins, apogee_dets)
    sh_nondets_photbins = np.logical_and(sh_photbins, ~apogee_dets)
    # Mark Stauffer Hartmann upper upper as APOGEE detections
    sh_dets_upper = np.logical_and(sh_upper, apogee_dets)
    sh_nondets_upper = np.logical_and(sh_upper, ~apogee_dets)

    ax4.errorbar(
        pleiades["VSINI"][sh_dets_apogee],
        sh_vdiff[sh_dets_apogee], yerr=0.20/np.log(10),
        xerr=0.15*pleiades["VSINI"][sh_dets_apogee], marker="o",
        color=bc.sky_blue, ls=""), 
    ax4.errorbar(
        pleiades["VSINI"][sh_nondets_apogee],
        sh_vdiff[sh_nondets_apogee], yerr=0.20/np.log(10),
        xerr=0.15*pleiades["VSINI"][sh_nondets_apogee], marker="o",
        color="grey", ls="", alpha=0.3), 
    ax4.errorbar(
        pleiades["VSINI"][sh_dets_photbins],
        sh_vdiff[sh_dets_photbins], yerr=0.20/np.log(10),
        xerr=0.15*pleiades["VSINI"][sh_dets_photbins], marker="8",
        color="red", ls=""), 
    ax4.errorbar(
        pleiades["VSINI"][sh_nondets_photbins],
        sh_vdiff[sh_nondets_photbins], yerr=0.20/np.log(10),
        xerr=0.15*pleiades["VSINI"][sh_nondets_photbins], marker="8",
        color="grey", ls="", alpha=0.3), 
    ax4.errorbar(
        pleiades["VSINI"][sh_dets_upper],
        sh_vdiff[sh_dets_upper], yerr=0,
        xerr=0.15*pleiades["VSINI"][sh_dets_upper], marker="^",
        color=bc.sky_blue, ls=""), 
    ax4.errorbar(
        pleiades["VSINI"][sh_nondets_upper],
        sh_vdiff[sh_nondets_upper], yerr=0,
        xerr=0.15*pleiades["VSINI"][sh_nondets_upper], marker="^",
        color="grey", ls="", alpha=0.3), 
    ax4.plot([1, 100], [0, 0], 'k-')

    ax4.set_xscale("log")
    ax4.set_xlim(1, 100)
    ax4.set_xlabel("APOGEE vsini")
    ax4.set_ylabel("log(APOGEE / Stauffer & Hartman)")
    ax4.set_title("Stauffer & Hartman measurements")

    # Lastly Stauffer points.
    s84 = ~pleiades["vsini_S84"].mask
    s84_detections = pleiades["vsini_lim_S84"] == "d"
    s84_upper = pleiades["vsini_lim_S84"] == "u"

    s84_vdiff = (
        np.log10(pleiades["VSINI"]) - np.log10(pleiades["vsini_S84"]))

    s84_singles = np.logical_and(s84_detections, ~phot_bin)
    s84_photbins = np.logical_and(s84_detections, phot_bin)
    # Mark Stauffer Hartmann detections as APOGEE detections
    s84_dets_apogee = np.logical_and(s84_singles, apogee_dets)
    s84_nondets_apogee = np.logical_and(s84_singles, ~apogee_dets)
    # Mark Stauffer Hartmann photometric binaries as APOGEE detections
    s84_dets_photbins = np.logical_and(s84_photbins, apogee_dets)
    s84_nondets_photbins = np.logical_and(s84_photbins, ~apogee_dets)
    # Mark Stauffer Hartmann upper upper as APOGEE detections
    s84_dets_upper = np.logical_and(s84_upper, apogee_dets)
    s84_nondets_upper = np.logical_and(s84_upper, ~apogee_dets)

    ax5.errorbar(
        pleiades["VSINI"][s84_dets_apogee],
        s84_vdiff[s84_dets_apogee], yerr=0.20/np.log(10),
        xerr=0.15*pleiades["VSINI"][s84_dets_apogee], marker="o",
        color=bc.algae, ls=""), 
    ax5.errorbar(
        pleiades["VSINI"][s84_nondets_apogee],
        s84_vdiff[s84_nondets_apogee], yerr=0.20/np.log(10),
        xerr=0.15*pleiades["VSINI"][s84_nondets_apogee], marker="o",
        color="grey", ls="", alpha=0.3), 
    ax5.errorbar(
        pleiades["VSINI"][s84_dets_photbins],
        s84_vdiff[s84_dets_photbins], yerr=0.20/np.log(10),
        xerr=0.15*pleiades["VSINI"][s84_dets_photbins], marker="8",
        color="red", ls=""), 
    ax5.errorbar(
        pleiades["VSINI"][s84_nondets_photbins],
        s84_vdiff[s84_nondets_photbins], yerr=0.20/np.log(10),
        xerr=0.15*pleiades["VSINI"][s84_nondets_photbins], marker="8",
        color="grey", ls="", alpha=0.3), 
    ax5.errorbar(
        pleiades["VSINI"][s84_dets_upper],
        s84_vdiff[s84_dets_upper], yerr=0,
        xerr=0.15*pleiades["VSINI"][s84_dets_upper], marker="^",
        color=bc.algae, ls=""), 
    ax5.errorbar(
        pleiades["VSINI"][s84_nondets_upper],
        s84_vdiff[s84_nondets_upper], yerr=0,
        xerr=0.15*pleiades["VSINI"][s84_nondets_upper], marker="^",
        color="grey", ls="", alpha=0.3), 
    ax5.plot([1, 100], [0, 0], 'k-')

    ax5.set_xscale("log")
    ax5.set_xlim(1, 100)
    ax5.set_ylim(-1, 1)
    ax5.set_xlabel("APOGEE vsini")
    ax5.set_ylabel("log(APOGEE / Stauffer)")
    ax5.set_title("Stauffer measurements")

def Pleiades_Baraffe_radius_comparison_APOGEE():
    '''Compare the deprojected radius from direct Baraffe vs SB-Law with APOGEE Teff.'''
    pleiades = cache.pleiades()

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    masses = pleiades["Mass"]
    iso = baraffe.BaraffeIsochrone.isochrone_from_file()
    baraffe_rad = iso.interpolate_isochrone_cols(
        0.12, masses, iso.mass_col, iso.radius_col)
    log_baraffe_lum = iso.interpolate_isochrone_cols(
        0.12, masses, iso.mass_col, iso.logL_col)
    apogee_rad = 10**(
        0.5*(log_baraffe_lum - 4*(np.log10(pleiades["TEFF"]) - np.log10(5777))))

    ax.plot(baraffe_rad, apogee_rad, 'k.', label="APOGEE Teff")
    ax.plot([0, 2], [0, 2], 'k-')
    ax.set_xlabel("Baraffe Radius")
    ax.set_ylabel("APOGEE Radius")

def Pleiades_Baraffe_radius_comparison_PM():
    '''Compare the deprojected radius from direct Baraffe vs SB-Law with PM-Teff.'''
    pleiades = cache.pleiades()

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    masses = pleiades["Mass"]
    iso = baraffe.BaraffeIsochrone.isochrone_from_file()
    baraffe_rad = iso.interpolate_isochrone_cols(
        0.12, masses, iso.mass_col, iso.radius_col)
    log_baraffe_lum = iso.interpolate_isochrone_cols(
        0.12, masses, iso.mass_col, iso.logL_col)
    pm_rad = pleiades["Radius"]

    ax.plot(baraffe_rad, pm_rad, 'k.', label="PM Teff")
    ax.plot([0, 2], [0, 2], 'k-')
    ax.set_xlabel("Baraffe Radius")
    ax.set_ylabel("PM Radius")

def Pleiades_Baraffe_radius_comparison_MIST():
    '''Compare the deprojected radius from direct Baraffe vs direct MIST.'''
    pleiades = cache.pleiades()

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    masses = pleiades["Mass"]
    iso = baraffe.BaraffeIsochrone.isochrone_from_file()
    baraffe_rad = iso.interpolate_isochrone_cols(
        0.12, masses, iso.mass_col, iso.radius_col)
    mist_rad = pleiades["MIST R"]

    ax.plot(baraffe_rad, mist_rad, 'k.', label="MIST Teff")
    ax.plot([0, 2], [0, 2], 'k-')
    ax.set_xlabel("Baraffe Radius")
    ax.set_ylabel("MIST Radius")

def compare_MIST_Baraffe_mass_lum():
    '''Compare the mass-luminosity relationship between MIST and Baraffe.
    
    Unlike many other relationships between isochrones, this should be
    well-calibrated.'''
    mist_iso = mist.MISTIsochrone.isochrone_from_file(0.0)
    baraffe_iso = baraffe.BaraffeIsochrone.isochrone_from_file()

    mist_table = mist_iso.iso_table(1.2e8)
    baraffe_table = baraffe_iso.iso_table(0.12)

    mist_lowmass = mist_table[mist_iso.mass_col] < 2
    baraffe_lowmass = baraffe_table[baraffe_iso.mass_col] < 2

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    ax.plot(
        mist_table[mist_iso.mass_col][mist_lowmass],
        mist_table[mist_iso.logL_col][mist_lowmass], ls="-", marker="o",
        color="r", label="MIST")
    ax.plot(
        baraffe_table[baraffe_iso.mass_col][baraffe_lowmass],
        baraffe_table[baraffe_iso.logL_col][baraffe_lowmass], ls="-",
        marker="o", color="k", label="Baraffe")
    ax.set_xlabel("Mass (Msun)")
    ax.set_ylabel("Log L (Lsun)")
    ax.legend(loc="lower right")

    f, ax2 = plt.subplots(1, 1, figsize=(12,12))
    testmasses = np.linspace(0.4, 1.5, 100)
    mist_lums = mist_iso.interpolate_isochrone_cols(
        1.2e8, testmasses, mist_iso.mass_col, mist_iso.logL_col)
    baraffe_lums = baraffe_iso.interpolate_isochrone_cols(
        0.12, testmasses, baraffe_iso.mass_col, baraffe_iso.logL_col)
    lumdiff = mist_lums - baraffe_lums 
    ax2.plot(testmasses, lumdiff, ls="-", marker="", color="k")
    ax2.plot(testmasses, np.zeros(len(testmasses)), ls="--", marker="",
             color="k")
    ax2.set_xlabel("Mass (Msun)")
    ax2.set_ylabel("Log L (MIST) - Log L (Baraffe)")

def Pleiades_binarity():
    '''Plot the photometric binaries in the Pleiades.'''
    pleiades = cache.pleiades()
    phot_binaries = pleiades["Delmag"] > 0.2

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    hr.absmag_teff_plot(
        pleiades["TEFF"], -pleiades["Delmag"], color="k", marker=".", ls="",
        axis=ax)
    hr.absmag_teff_plot(
        pleiades["TEFF"][phot_binaries], -pleiades["Delmag"][phot_binaries], 
        color="r", marker=".", ls="", axis=ax)
#   ax.plot(
#       pleiades["(V-K)0"], -pleiades["Delmag"], color="k", marker=".", ls="")
#   hr.invert_y_axis(ax)
    ax.set_xlabel("APOGEE TEFF")
    ax.set_ylabel("V Excess")

def Pleiades_vsini_vrot_agreement():
    '''Check to see if the Pleaides velocities and vsini are consistent.'''
    pleiades = cache.stauffer_hartmann_pleiades()

    # I want to make sure upper limits are actually detected as lower limits.
    pleiades_vsini = np.where(
        pleiades["vsini lim"] == "u", 1, pleiades["vsini_SH"])
    pleiades_velocities = rot.period_to_velocities(
        pleiades["Per1"], pleiades["Baraffe Radius"])

    rot.compare_vsini_distribution(
        pleiades_velocities, pleiades_vsini, vsini_cutoff=10, maxv=100, 
        nbins=100)

def Pleiades_vsini_vrot_agreement_Rinflate():
    '''Check to see if the Pleaides velocities and vsini are consistent.'''
    pleiades = cache.pleiades()

    pleiades_vsini = pleiades["VSINI"]
    pleiades_velocities = rot.period_to_velocities(
        pleiades["Per1"], 1.00*pleiades["K-band R (Baraffe)"])

    rot.compare_vsini_distribution(
        pleiades_velocities, pleiades_vsini, vsini_cutoff=10, maxv=100, 
        nbins=100)

def Pleiades_compare_velocities_samples():
    '''Compare literature to APOGEE velocity for different subsamples.'''
    pleiades = cache.pleiades_APOGEE_Literature_vsini()

    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(80, 12))
    
    # Plot the Queloz targets in the first panel.
    queloz_elodie = ~pleiades["vsini_QuelozE"].mask
    queloz_coravel = ~pleiades["vsini_QuelozC"].mask
    full_queloz = np.logical_or(queloz_elodie, queloz_coravel)
    queloz_coravel_upper = pleiades["vsini_lim_QuelozC"] == "<"
    queloz_coravel_lower = pleiades["vsini_lim_QuelozC"] == ">"
    queloz_coravel_det = np.logical_not(
        np.logical_or(queloz_coravel_upper,queloz_coravel_lower))

    rot.plot_vsini_velocity(
        pleiades["VSINI"][full_queloz], pleiades["Per1"][full_queloz],
        pleiades["Per1"][full_queloz]*0.07, 
        pleiades["K-band R (Baraffe)"][full_queloz], 
        pleiades["K-band R (Baraffe)"][full_queloz]*0.1, color="k",
        marker="o", vsini_lim=1, vsini_fracerr=0.15, ax=ax1, sini_label=False)
    rot.plot_vsini_velocity(
        pleiades["vsini_QuelozE"][full_queloz], 
        pleiades["Per1"][queloz_elodie],
        pleiades["Per1"][queloz_elodie]*0.07, 
        pleiades["K-band R (Baraffe)"][queloz_elodie], 
        pleiades["K-band R (Baraffe)"][queloz_elodie]*0.1, color=bc.red,
        marker="*", vsini_lim=1, vsini_fracerr=0.15, ax=ax1, sini_label=False)
    rot.plot_vsini_velocity(
        pleiades["vsini_QuelozC"][queloz_coravel_det], 
        pleiades["Per1"][queloz_coravel_det],
        pleiades["Per1"][queloz_coravel_det]*0.07, 
        pleiades["K-band R (Baraffe)"][queloz_coravel_det], 
        pleiades["K-band R (Baraffe)"][queloz_coravel_det]*0.1, color=bc.red,
        marker="*", vsini_lim=1, vsini_fracerr=0.15, ax=ax1, sini_label=False)
    rot.plot_vsini_velocity(
        pleiades["vsini_QuelozC"][queloz_coravel_upper], 
        pleiades["Per1"][queloz_coravel_upper],
        pleiades["Per1"][queloz_coravel_upper]*0.07, 
        pleiades["K-band R (Baraffe)"][queloz_coravel_upper], 
        pleiades["K-band R (Baraffe)"][queloz_coravel_upper]*0.1, color=bc.red,
        marker="v", vsini_lim=1, vsini_fracerr=0.0, ax=ax1, sini_label=False)
    rot.plot_vsini_velocity(
        pleiades["vsini_QuelozC"][queloz_coravel_lower], 
        pleiades["Per1"][queloz_coravel_lower],
        pleiades["Per1"][queloz_coravel_lower]*0.07, 
        pleiades["K-band R (Baraffe)"][queloz_coravel_lower], 
        pleiades["K-band R (Baraffe)"][queloz_coravel_lower]*0.1, color=bc.red,
        marker="^", vsini_lim=1, vsini_fracerr=0.0, ax=ax1, sini_label=False)

    # Plot Terndrup targets in the second panel
    terndrup = ~pleiades["vsini_Terndrup"].mask
    terndrup_detections = pleiades["vsini_lim_Terndrup"] == "d"
    terndrup_limits = pleiades["vsini_lim_Terndrup"] == "u"

    # Now plot the terndrup points.
    rot.plot_vsini_velocity(
        pleiades["VSINI"][terndrup], pleiades["Per1"][terndrup],
        pleiades["Per1"][terndrup]*0.07, 
        pleiades["K-band R (Baraffe)"][terndrup], 
        pleiades["K-band R (Baraffe)"][terndrup]*0.1, color="k",
        marker="o", vsini_lim=1, vsini_fracerr=0.15, ax=ax2, sini_label=False)
    rot.plot_vsini_velocity(
        pleiades["vsini_Terndrup"][terndrup], pleiades["Per1"][terndrup],
        pleiades["Per1"][terndrup]*0.07, 
        pleiades["K-band R (Baraffe)"][terndrup], 
        pleiades["K-band R (Baraffe)"][terndrup]*0.1, color="blue",
        marker="*", vsini_lim=1, vsini_fracerr=0.15, ax=ax2, sini_label=False)
    rot.plot_vsini_velocity(
        pleiades["vsini_Terndrup"][terndrup_limits], 
        pleiades["Per1"][terndrup_limits],
        pleiades["Per1"][terndrup_limits]*0.07, 
        pleiades["K-band R (Baraffe)"][terndrup_limits], 
        pleiades["K-band R (Baraffe)"][terndrup_limits]*0.1, color="blue",
        marker="v", vsini_lim=1, vsini_fracerr=0.15, ax=ax2, sini_label=False)

    # Now plot the Soderblom points.
    soderblom = ~pleiades["vsini_Soderblom"].mask
    rot.plot_vsini_velocity(
        pleiades["VSINI"][soderblom], pleiades["Per1"][soderblom],
        pleiades["Per1"][soderblom]*0.07, 
        pleiades["K-band R (Baraffe)"][soderblom], 
        pleiades["K-band R (Baraffe)"][soderblom]*0.1, color="k",
        marker="o", vsini_lim=1, vsini_fracerr=0.15, ax=ax3, sini_label=False)
    rot.plot_vsini_velocity(
        pleiades["vsini_Soderblom"][soderblom], pleiades["Per1"][soderblom],
        pleiades["Per1"][soderblom]*0.07, 
        pleiades["K-band R (Baraffe)"][soderblom], 
        pleiades["K-band R (Baraffe)"][soderblom]*0.1, color=bc.violet,
        marker="*", vsini_lim=1, vsini_fracerr=0.15, ax=ax3, sini_label=False)

    # Now plot the Stauffer & Hartmann points (1987)
    sh = ~pleiades["vsini_SH"].mask
    sh_detections = pleiades["vsini_lim_SH"] == "d"
    sh_upper = pleiades["vsini_lim_SH"] == "u"
    rot.plot_vsini_velocity(
        pleiades["VSINI"][sh], pleiades["Per1"][sh],
        pleiades["Per1"][sh]*0.07, 
        pleiades["K-band R (Baraffe)"][sh], 
        pleiades["K-band R (Baraffe)"][sh]*0.1, color="k",
        marker="o", vsini_lim=1, vsini_fracerr=0.15, ax=ax4, sini_label=False)
    rot.plot_vsini_velocity(
        pleiades["vsini_SH"][sh_detections], pleiades["Per1"][sh_detections],
        pleiades["Per1"][sh_detections]*0.07, 
        pleiades["K-band R (Baraffe)"][sh_detections], 
        pleiades["K-band R (Baraffe)"][sh_detections]*0.1, color=bc.sky_blue,
        marker="*", vsini_lim=1, vsini_fracerr=0.15, ax=ax4, sini_label=False)
    rot.plot_vsini_velocity(
        pleiades["vsini_SH"][sh_upper], pleiades["Per1"][sh_upper],
        pleiades["Per1"][sh_upper]*0.07, 
        pleiades["K-band R (Baraffe)"][sh_upper], 
        pleiades["K-band R (Baraffe)"][sh_upper]*0.1, color=bc.sky_blue,
        marker="v", vsini_lim=1, vsini_fracerr=0.0, ax=ax4, sini_label=False)

    # Now plot the Stauffer et al (1984) points.
    s84 = ~pleiades["vsini_S84"].mask
    s84_detections = pleiades["vsini_lim_S84"] == "d"
    s84_upper = pleiades["vsini_lim_S84"] == "u"
    rot.plot_vsini_velocity(
        pleiades["VSINI"][s84], pleiades["Per1"][s84],
        pleiades["Per1"][s84]*0.07, 
        pleiades["K-band R (Baraffe)"][s84], 
        pleiades["K-band R (Baraffe)"][s84]*0.1, color="k",
        marker="o", vsini_lim=1, vsini_fracerr=0.15, ax=ax5, sini_label=False)
    rot.plot_vsini_velocity(
        pleiades["vsini_S84"][s84_detections], pleiades["Per1"][s84_detections],
        pleiades["Per1"][s84_detections]*0.07, 
        pleiades["K-band R (Baraffe)"][s84_detections], 
        pleiades["K-band R (Baraffe)"][s84_detections]*0.1, color=bc.algae,
        marker="*", vsini_lim=1, vsini_fracerr=0.15, ax=ax5, sini_label=False)
    rot.plot_vsini_velocity(
        pleiades["vsini_S84"][s84_upper], pleiades["Per1"][s84_upper],
        pleiades["Per1"][s84_upper]*0.07, 
        pleiades["K-band R (Baraffe)"][s84_upper], 
        pleiades["K-band R (Baraffe)"][s84_upper]*0.1, color=bc.algae,
        marker="v", vsini_lim=1, vsini_fracerr=0.0, ax=ax5, sini_label=False)

    ax2.set_ylabel("")
    ax3.set_ylabel("")
    ax4.set_ylabel("")
    ax5.set_ylabel("")
    ax1.set_title("Queloz et al (1998)")
    ax2.set_title("Terndrup et al (2000)")
    ax3.set_title("Soderblom et al (1993)")
    ax4.set_title("Stauffer & Hartmann (1987)")
    ax5.set_title("Stauffer et al (1984)")

def Pleiades_calibrator_sample_sizes():
    '''List the number of sample sizes for Pleiades calibrators.'''
    vsini_dets = cache.pleiades_APOGEE_Literature_vsini()
    study_list = [
        "Jackson", "S84", "SH", "QuelozC", "QuelozE", "Terndrup", "Soderblom"]

    counts = []
    for s in study_list:
        studycol = "{0}_{1}".format("vsini", s)
        col = np.ma.masked_invalid(vsini_dets[studycol])
        count = np.ma.count(col)
        counts.append(count)

    tab = Table([study_list, counts], names=("Study", "Number"))
    print(tab)

@write_plot("Pleiades_vsini_breakdown")
def Pleiades_vsini_sources():
    '''Plot where vsini from different sources lie.'''
    # NOW that every star (except HII906) is accounted for, plot them.
    # The priorities I'm taking are:
    # Queloz > Terndrup > Soderblom > SH > S84.
    vsini_dets = cache.pleiades_APOGEE_Literature_vsini()
    queloz_indices = np.isfinite(vsini_dets["vsini_QuelozE"]).filled(0.0)
    terndrup_indices = au.multi_logical_and(
        ~vsini_dets["vsini_Terndrup"].mask)
    soderblom_indices = au.multi_logical_and(
        ~vsini_dets["vsini_Soderblom"].mask)
    sh_indices = au.multi_logical_and(
        ~vsini_dets["vsini_SH"].mask)
    s84_indices = au.multi_logical_and(
        ~vsini_dets["vsini_S84"].mask)
    jackson_indices = au.multi_logical_and(
        ~vsini_dets["vsini_Jackson"].mask)
    unknown_indices = au.multi_logical_and(
        ~queloz_indices, ~terndrup_indices, ~soderblom_indices, ~sh_indices,
        ~s84_indices)

    f, axes = plt.subplots(3, 6, figsize=(40, 30))
    hr.absmag_teff_plot(
        vsini_dets["TEFF"][queloz_indices], vsini_dets["MK"][queloz_indices],
        color=bc.red, marker="o", ls="", label="Queloz", axis=axes[0][0])
    hr.absmag_teff_plot(
        vsini_dets["TEFF"][terndrup_indices], vsini_dets["MK"][terndrup_indices],
        color=bc.blue, marker="d", ls="", label="Terndrup", axis=axes[0][1])
    hr.absmag_teff_plot(
        vsini_dets["TEFF"][soderblom_indices], vsini_dets["MK"][soderblom_indices],
        color=bc.violet, marker="*", ls="", label="Soderblom", axis=axes[0][2])
    hr.absmag_teff_plot(
        vsini_dets["TEFF"][sh_indices], vsini_dets["MK"][sh_indices],
        color=bc.sky_blue, marker="s", ls="", label="Stauffer & Hartmann",
        axis=axes[0][3])
    hr.absmag_teff_plot(
        vsini_dets["TEFF"][s84_indices], vsini_dets["MK"][s84_indices],
        color=bc.algae, marker="+", ls="", label="Stauffer 1984",
        axis=axes[0][4])
    hr.absmag_teff_plot(
        vsini_dets["TEFF"][jackson_indices], vsini_dets["MK"][jackson_indices],
        color=bc.green, marker="^", ls="", label="Jackson",
        axis=axes[0][5])

    axes[1][0].plot(
        vsini_dets["TEFF"][queloz_indices],
        vsini_dets["VSINI"][queloz_indices], color=bc.red, marker="o", ls="")
    axes[1][1].plot(
        vsini_dets["TEFF"][terndrup_indices],
        vsini_dets["VSINI"][terndrup_indices], color=bc.blue, marker="d", ls="")
    axes[1][2].plot(
        vsini_dets["TEFF"][soderblom_indices],
        vsini_dets["VSINI"][soderblom_indices], color=bc.violet, marker="*", ls="")
    axes[1][3].plot(
        vsini_dets["TEFF"][sh_indices],
        vsini_dets["VSINI"][sh_indices], color=bc.sky_blue, marker="s", ls="")
    axes[1][4].plot(
        vsini_dets["TEFF"][s84_indices],
        vsini_dets["VSINI"][s84_indices], color=bc.algae, marker="+", ls="")
    axes[1][5].plot(
        vsini_dets["TEFF"][jackson_indices],
        vsini_dets["VSINI"][jackson_indices], color=bc.green, marker="^", ls="")

    
    # Now plot vsini vs veq
    rot.plot_vsini_velocity(
        vsini_dets["vsini"][queloz_indices],
        vsini_dets["Per1"][queloz_indices],
        vsini_dets["Per1"][queloz_indices]*0.07,
        vsini_dets["K-band R (Baraffe)"][queloz_indices], 
        vsini_dets["MIST R Err"][queloz_indices], color=bc.red,
        marker="o", ax=axes[2][0])
    rot.plot_vsini_velocity(
        vsini_dets["vsini"][terndrup_indices],
        vsini_dets["Per1"][terndrup_indices],
        vsini_dets["Per1"][terndrup_indices]*0.07,
        vsini_dets["K-band R (Baraffe)"][terndrup_indices], 
        vsini_dets["MIST R Err"][terndrup_indices], color=bc.blue,
        marker="d", ax=axes[2][1])
    rot.plot_vsini_velocity(
        vsini_dets["vsini"][soderblom_indices],
        vsini_dets["Per1"][soderblom_indices],
        vsini_dets["Per1"][soderblom_indices]*0.07,
        vsini_dets["K-band R (Baraffe)"][soderblom_indices], 
        vsini_dets["MIST R Err"][soderblom_indices], color=bc.violet,
        marker="*", ax=axes[2][2])
    rot.plot_vsini_velocity(
        vsini_dets["vsini"][sh_indices],
        vsini_dets["Per1"][sh_indices],
        vsini_dets["Per1"][sh_indices]*0.07,
        vsini_dets["K-band R (Baraffe)"][sh_indices], 
        vsini_dets["MIST R Err"][sh_indices], color=bc.sky_blue,
        marker="s", ax=axes[2][3])
    rot.plot_vsini_velocity(
        vsini_dets["vsini"][s84_indices],
        vsini_dets["Per1"][s84_indices],
        vsini_dets["Per1"][s84_indices]*0.07,
        vsini_dets["K-band R (Baraffe)"][s84_indices], 
        vsini_dets["MIST R Err"][s84_indices], color=bc.algae,
        marker="+", ax=axes[2][4])
    rot.plot_vsini_velocity(
        vsini_dets["vsini"][jackson_indices],
        vsini_dets["Per1"][jackson_indices],
        vsini_dets["Per1"][jackson_indices]*0.07,
        vsini_dets["K-band R (Baraffe)"][jackson_indices], 
        vsini_dets["MIST R Err"][jackson_indices], color=bc.green,
        marker="^", ax=axes[2][5])
    for i in range(6):
        axes[0][i].set_xlim(6700, 3500)
        axes[0][i].set_ylim(6.0, 2.2)
        axes[0][i].set_xlabel("")
        axes[0][i].set_ylabel("")
        axes[1][i].set_xlim(6700, 3500)
        axes[1][i].set_yscale("log")
        axes[1][i].set_ylim(1, 100)
        axes[1][i].set_xlabel("Teff (K)")
        axes[2][i].set_ylabel("")
    axes[0][0].set_ylabel("MK")
    axes[1][0].set_ylabel("Rebull vsini")
    axes[2][0].set_ylabel("Rebull vsini")
    axes[0][0].set_title("Queloz et al (1998)")
    axes[0][1].set_title("Terndrup et al (2000)")
    axes[0][2].set_title("Soderblom et al (1993))")
    axes[0][3].set_title("Stauffer & Hartmann (1987)")
    axes[0][4].set_title("Stauffer et al (1984)")
    axes[0][5].set_title("Jackson et al (2017)")

def queloz_vs_literature_vsini():
    '''Plot how well Queloz et al (1998) measurements compare to literature.'''
    pleiades = cache.pleiades_APOGEE_Literature_vsini()
    queloz_elodie_vsini = np.ma.masked_invalid(pleiades["vsini_QuelozE"])
    queloz_elodie_detections = ~queloz_elodie_vsini.mask
    queloz_coravel_vsini = pleiades["vsini_QuelozC"]
    queloz_coravel_vsini_upper = pleiades["vsini_lim_QuelozC"] == "<"
    queloz_coravel_vsini_lower = pleiades["vsini_lim_QuelozC"] == ">"
    queloz_coravel_detections = np.logical_and(
        np.logical_not(queloz_coravel_vsini_upper),
        np.logical_not(queloz_coravel_vsini_lower))

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    # Stauffer & Hartmann points.
    sh_vsini = pleiades["vsini_SH"]
    sh_vsini_upper = pleiades["vsini_lim_SH"] == "u"
    sh_vsini_detections = pleiades["vsini_lim_SH"] == "d"
    # Indices for upper and lower limits.
    sh_elodie_double_detections = np.logical_and(
        queloz_elodie_detections, sh_vsini_detections)
    sh_upper_elodie_detection = np.logical_and(
        queloz_elodie_detections, sh_vsini_upper)
    sh_coravel_double_detections = np.logical_and(
        queloz_coravel_detections, sh_vsini_detections)
    sh_detection_coravel_upper = np.logical_and(
        queloz_coravel_vsini_upper, sh_vsini_detections)
    sh_detection_coravel_lower = np.logical_and(
        queloz_coravel_vsini_lower, sh_vsini_detections)
    sh_upper_coravel_detection = np.logical_and(
        queloz_coravel_detections, sh_vsini_upper)
    sh_upper_coravel_upper = np.logical_and(
        queloz_coravel_vsini_upper, sh_vsini_upper)
    sh_upper_coravel_lower = np.logical_and(
        queloz_coravel_vsini_lower, sh_vsini_upper)

    ax.plot(
        queloz_elodie_vsini[sh_elodie_double_detections], 
        sh_vsini[sh_elodie_double_detections], color=bc.sky_blue, marker="o",
        ls="", label="Stauffer & Hartmann (1987)")
    ax.plot(
        queloz_elodie_vsini[sh_upper_elodie_detection],
        sh_vsini[sh_upper_elodie_detection], color=bc.sky_blue, marker='v',
        ls="", label="")
    ax.plot(
        queloz_coravel_vsini[sh_coravel_double_detections], 
        sh_vsini[sh_coravel_double_detections], color=bc.sky_blue, marker="o",
        ls="", label="")
    ax.plot(
        queloz_coravel_vsini[sh_upper_coravel_detection],
        sh_vsini[sh_upper_coravel_detection], color=bc.sky_blue, marker='v',
        ls="", label="")
    ax.plot(
        queloz_coravel_vsini[sh_detection_coravel_upper], 
        sh_vsini[sh_detection_coravel_upper], color=bc.sky_blue, marker="<", 
        ls="", label="")
    ax.plot(
        queloz_coravel_vsini[sh_upper_coravel_upper],
        sh_vsini[sh_upper_coravel_upper], color=bc.sky_blue, marker='x',
        ls="", label="")
    ax.plot(
        queloz_coravel_vsini[sh_detection_coravel_lower],
        sh_vsini[sh_detection_coravel_lower], color=bc.sky_blue, marker="^",
        ls="", label="")
    ax.plot(
        queloz_coravel_vsini[sh_upper_coravel_lower],
        sh_vsini[sh_upper_coravel_lower], color=bc.sky_blue, marker='x',
        ls="", label="")

    # Now do the Soderblom targets.
    sod_vsini = pleiades["vsini_Soderblom"]
    sod_vsini_detections = ~sod_vsini.mask

    ax.plot(
        queloz_elodie_vsini[queloz_elodie_detections], 
        sod_vsini[queloz_elodie_detections], color=bc.violet, marker="o",
        ls="", label="")

    
    ax.plot([1, 20], [1, 20], 'k-')
    ax.legend(loc="lower right")
    ax.set_xlabel("Queloz et al (1998) vsini")
    ax.set_ylabel("Comparison vsini")

def pleiades_photbin_vsini_comparison():
    '''Exclude the photometric binaries from vsini comparisons.'''
    pleiades = cache.pleiades_APOGEE_Literature_vsini()
    phot_bin = np.logical_or(
        np.logical_and(pleiades["(V-K)0"] <= 5, pleiades["Delmag"] >= 0.45),
        np.logical_and(pleiades["(V-K)0"] > 5, pleiades["Delmag"] >= 0.55))

    f, ax = plt.subplots(1, 1, figsize=figsize)
    vdiff = np.log10(pleiades["VSINI"]) - np.log10(pleiades["vsini"])
    ax.plot(pleiades["VSINI"][~phot_bin], vdiff[~phot_bin], 'ko')
    ax.plot(pleiades["VSINI"][phot_bin], vdiff[phot_bin], 'ro',
            alpha=0.3)
    ax.plot([1, 100], [0, 0], 'k-')
    ax.set_xscale("log")

@write_plot("Pleiades_uncertainty")
def pleiades_uncertainty_calibration():
    '''Calculate the uncertainty of the APOGEE data from the Pleiades.'''
    pleiades = cache.pleiades_APOGEE_Literature_vsini()
    apogee_det = pleiades["VSINI"] > 10
    phot_sing = pleiades["Delmag"] < 0.3
    sh_indices = au.multi_logical_and(
        apogee_det, phot_sing, pleiades["vsini_lim_SH"] == "d", 
        ~pleiades["vsini_err_SH"].mask, pleiades["HII"] != "HII1653")
    elodie_indices = au.multi_logical_and(
        apogee_det, phot_sing, ~pleiades["vsini_QuelozE"].mask, 
        np.isfinite(pleiades["vsini_QuelozE"]))
    coravel_indices = au.multi_logical_and(
        apogee_det, phot_sing, pleiades["vsini_lim_QuelozC"] == " ", 
        ~pleiades["vsini_QuelozC"].mask, 
        np.isfinite(pleiades["vsini_QuelozC"]))
    jackson_indices = au.multi_logical_and(
        apogee_det, phot_sing, pleiades["vsini_lim_Jackson"] == "d",
        ~pleiades["vsini_err_Jackson"].mask)
        
            
    vmax=200
    data_indices = [sh_indices, coravel_indices, jackson_indices]
    data_cols = ["vsini_SH", "vsini_QuelozC", "vsini_Jackson"]
    err_cols = ["vsini_err_SH", "vsini_err_QuelozC", "vsini_err_Jackson"]
    labels = ["SH87", "Q98", "J17"]
    apogee_vsini_col = "VSINI"
    colors = [bc.algae, bc.red, "g"]
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    combined = []
    for idx, col, errcol, c, l in zip(
            data_indices, data_cols, err_cols, colors, labels):
        print(l)
        v1 = pleiades[col][idx] 
        v2 = pleiades[apogee_vsini_col][idx]
        sig1 = pleiades[errcol][idx]

#       datax.errorbar(
#           np.log10(v1), np.log10(v2), xerr=sig1/v1/np.log(10), color=c, 
#           marker=".", ls="")

        ndim, nwalkers = 1, len(v1)*2 
        sig2_ml = samp.calc_ml(
            np.log10(v1), np.log10(v2), sig1/v1/np.log(10), np.log10(vmax))
        print("Maximum Likelihood Error: {0:.1f}%".format(
            sig2_ml*np.log(10)*100))
        pos = [np.log(sig2_ml) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, samp.lnprob, args=(
                np.log10(v1), np.log10(v2), sig1/v1/np.log(10), np.log10(vmax)))

        sampler.run_mcmc(pos, 500)

        samples = sampler.chain[:,50:,:].reshape((-1, ndim))
        percentages = np.exp(samples[:,0:1])*np.log(10)*100
        combined.append(percentages)

#       fig = corner.corner(
#           np.exp(samples[:,0:1])*np.log(10)*100, labels=["$sig2$"],
#           color=c, fig=fig)

        ax.hist(percentages, bins=40, range=(0, 25),
                color=c, histtype="step", normed=True, label=l)

        qs = np.percentile(percentages, [16, 50, 84])
        print("Confidence Intervals: {0:.1f} + {1:.1f} - {2:.1f}".format(
            qs[1], qs[2], qs[0]))

#       walkerax.plot(sampler.chain[0, :, 0])

#       datax.errorbar(
#           np.log10(v1), np.log10(v2), yerr=qs[1]/100/np.log(10), color="r", 
#           marker=".", ls="", alpha=0.5)

    full_sample = np.concatenate(combined)
    ax.hist(full_sample, bins=40, range=(0, 25),
            color='k', histtype="step", normed=True, label="Combined", lw=3)
    qs = np.percentile(full_sample, [16, 50, 84])
    print("Confidence Intervals (full): {0:.1f} + {1:.1f} - {2:.1f}".format(
        qs[1], qs[2], qs[0]))

    ax.set_xlabel("Fractional Uncertainty (%)")
    ax.set_ylabel("Probability")
    ax.legend(loc="upper right")

def Pleiades_photometric_binaries():
    '''Count and plot the photometric binaries in the Pleiades.'''
    full_pleiades = cache.pleiades_APOGEE_Literature_vsini()
    pleiades_group = full_pleiades.group_by("APOGEE_ID")
    apogee_avg = pleiades_group[[
        "APOGEE_ID", "VSINI"]].groups.aggregate(np.mean)
    unique_pleiades = unique(full_pleiades, keys="APOGEE_ID")
    unique_pleiades.remove_column("VSINI")
    pleiades = pleiades_group[[
        "APOGEE_ID", "VSINI", "(V-K)0", "Delmag", "vsini_QuelozE", 
        "vsini_err_QuelozE", "vsini_QuelozC", "vsini_lim_QuelozC", 
        "vsini_err_QuelozC", "vsini_Terndrup", "vsini_err_Terndrup",
        "vsini_lim_Terndrup", "vsini_Soderblom", "vsini_lim_Soderblom",
        "vsini_SH", "vsini_err_SH", "vsini_lim_SH", "vsini_S84",
        "vsini_err_S84", "vsini_lim_S84", "vsini_Jackson", "vsini_err_Jackson",
    "vsini_lim_Jackson"]].groups.aggregate(np.mean)
    pleiades = au.join_by_id(
        unique_pleiades, apogee_avg, "APOGEE_ID", "APOGEE_ID")
    pleiades_useful = au.multi_logical_or(
        ~pleiades["vsini_QuelozE"].mask, ~pleiades["vsini_QuelozC"].mask,
        ~pleiades["vsini_SH"].mask, ~pleiades["vsini_Jackson"].mask)
    pleiades = pleiades[pleiades_useful]
    phot_bin = np.logical_or(
        np.logical_and(pleiades["(V-K)0"] <= 5, pleiades["Delmag"] >= 0.3),
        np.logical_and(pleiades["(V-K)0"] > 5, pleiades["Delmag"] >= 0.3))
    
    f, ax = plt.subplots(1, 1, figsize=figsize)

    ax.errorbar(
        pleiades["(V-K)0"][~phot_bin], pleiades["MK"][~phot_bin], 
        yerr=pleiades["K_ERR"][~phot_bin], marker=".", color=bc.black, ls="")
    ax.errorbar(
        pleiades["(V-K)0"][phot_bin], pleiades["MK"][phot_bin], 
        yerr=pleiades["K_ERR"][phot_bin], marker="o", color="r", ls="")
    
    hr.invert_y_axis(ax)

    print("Photometric Binaries: {0}".format(format_fraction(
        np.count_nonzero(phot_bin), len(pleiades))))

def Pleiades_photometric_binaries_inconsistent():
    '''Count and plot the photometric binaries in the Pleiades.'''
    full_pleiades = cache.pleiades_APOGEE_Literature_vsini()
    pleiades_group = full_pleiades.group_by("APOGEE_ID")
    apogee_avg = pleiades_group[[
        "APOGEE_ID", "VSINI"]].groups.aggregate(np.mean)
    unique_pleiades = unique(full_pleiades, keys="APOGEE_ID")
    unique_pleiades.remove_column("VSINI")
    pleiades = pleiades_group[[
        "APOGEE_ID", "VSINI", "(V-K)0", "Delmag", "vsini_QuelozE", 
        "vsini_err_QuelozE", "vsini_QuelozC", "vsini_lim_QuelozC", 
        "vsini_err_QuelozC", "vsini_Terndrup", "vsini_err_Terndrup",
        "vsini_lim_Terndrup", "vsini_Soderblom", "vsini_lim_Soderblom",
        "vsini_SH", "vsini_err_SH", "vsini_lim_SH", "vsini_S84",
        "vsini_err_S84", "vsini_lim_S84", "vsini_Jackson", "vsini_err_Jackson",
    "vsini_lim_Jackson"]].groups.aggregate(np.mean)
    pleiades = au.join_by_id(
        unique_pleiades, apogee_avg, "APOGEE_ID", "APOGEE_ID")
    phot_bin = np.logical_or(
        np.logical_and(pleiades["(V-K)0"] <= 5, pleiades["Delmag"] >= 0.3),
        np.logical_and(pleiades["(V-K)0"] > 5, pleiades["Delmag"] >= 0.3))
    vsini_detection = pleiades["VSINI"] > 10
    # I want to mask the nondetections.
    pleiades["vsini_err_QuelozC"][pleiades["vsini_lim_QuelozC"] != " "] = np.ma.masked
    pleiades["vsini_err_Terndrup"][pleiades["vsini_lim_Terndrup"] != "d"] = np.ma.masked
    pleiades["vsini_err_SH"][pleiades["vsini_lim_SH"] != "d"] = np.ma.masked
    pleiades["vsini_err_S84"][pleiades["vsini_lim_S84"] != "d"] = np.ma.masked
    pleiades["vsini_err_Jackson"][pleiades["vsini_lim_Jackson"] != "d"] = np.ma.masked
    # Also mask the nans
    pleiades["vsini_err_QuelozE"] = np.ma.masked_invalid(
        pleiades["vsini_err_QuelozE"])
    pleiades["vsini_err_QuelozC"] = np.ma.masked_invalid(
        pleiades["vsini_err_QuelozC"])
    with_errors = au.multi_logical_or(
        ~pleiades["vsini_err_QuelozE"].mask, ~pleiades["vsini_err_QuelozC"].mask,
        ~pleiades["vsini_err_Terndrup"].mask, ~pleiades["vsini_err_SH"].mask,
        ~pleiades["vsini_err_S84"].mask, ~pleiades["vsini_err_Jackson"].mask)
    valid_pleiades = au.multi_logical_and(
        vsini_detection, phot_bin, with_errors)

    f, ax = plt.subplots(1, 1, figsize=figsize)

    phot_bin_pleiades = pleiades[valid_pleiades]

    queloze_chisq = (
        (np.log10(phot_bin_pleiades["vsini_QuelozE"]) -
         np.log10(phot_bin_pleiades["VSINI"]))**2 /
        ((phot_bin_pleiades["vsini_err_QuelozE"] /
          phot_bin_pleiades["vsini_QuelozE"] / np.log(10))**2 + 
         (0.1/np.log(10))**2))
    quelozc_chisq = (
        (np.log10(phot_bin_pleiades["vsini_QuelozC"]) -
         np.log10(phot_bin_pleiades["VSINI"]))**2 /
        ((phot_bin_pleiades["vsini_err_QuelozC"] /
          phot_bin_pleiades["vsini_QuelozC"] / np.log(10))**2 + 
         (0.1/np.log(10))**2))
    terndrup_chisq = (
        (np.log10(phot_bin_pleiades["vsini_Terndrup"]) -
         np.log10(phot_bin_pleiades["VSINI"]))**2 /
        ((phot_bin_pleiades["vsini_err_Terndrup"] /
          phot_bin_pleiades["vsini_Terndrup"] / np.log(10))**2 + 
         (0.1/np.log(10))**2))
    sh_chisq = (
        (np.log10(phot_bin_pleiades["vsini_SH"]) -
         np.log10(phot_bin_pleiades["VSINI"]))**2 /
        ((phot_bin_pleiades["vsini_err_SH"] /
          phot_bin_pleiades["vsini_SH"] / np.log(10))**2 + 
         (0.1/np.log(10))**2))
    s84_chisq = (
        (np.log10(phot_bin_pleiades["vsini_S84"]) -
         np.log10(phot_bin_pleiades["VSINI"]))**2 /
        ((phot_bin_pleiades["vsini_err_S84"] /
          phot_bin_pleiades["vsini_S84"] / np.log(10))**2 + 
         (0.1/np.log(10))**2))
    jackson_chisq = (
        (np.log10(phot_bin_pleiades["vsini_Jackson"]) -
         np.log10(phot_bin_pleiades["VSINI"]))**2 /
        ((phot_bin_pleiades["vsini_err_Jackson"] /
          phot_bin_pleiades["vsini_Jackson"] / np.log(10))**2 + 
         (0.1/np.log(10))**2))

    full_matrix = np.ma.array([
        queloze_chisq, quelozc_chisq, terndrup_chisq, sh_chisq, s84_chisq,
        jackson_chisq])

    outliers = full_matrix > 3**2
    outliers[1,:] = np.logical_and(
        outliers[1,:], phot_bin_pleiades["vsini_lim_QuelozC"] == " ")
    outliers[2,:] = np.logical_and(
        outliers[2,:], phot_bin_pleiades["vsini_lim_Terndrup"] == "d")
    outliers[3,:] = np.logical_and(
        outliers[3,:], phot_bin_pleiades["vsini_lim_SH"] == "d")
    outliers[4,:] = np.logical_and(
        outliers[4,:], phot_bin_pleiades["vsini_lim_S84"] == "d")
    outliers[5,:] = np.logical_and(
        outliers[5,:], phot_bin_pleiades["vsini_lim_Jackson"] == "d")

    def_outliers = np.ma.all(outliers, axis=0)
    maybe = np.logical_and(np.ma.any(outliers, axis=0), ~def_outliers)
    def_consistent = np.ma.all(~outliers, axis=0)

    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_QuelozE"][def_outliers]), 
        np.log10(phot_bin_pleiades["VSINI"][def_outliers]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_QuelozE"][def_outliers] / 
              phot_bin_pleiades["vsini_QuelozE"][def_outliers] / np.log(10)), 
        color="red", marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_QuelozE"][maybe]), 
        np.log10(phot_bin_pleiades["VSINI"][maybe]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_QuelozE"][maybe] / 
              phot_bin_pleiades["vsini_QuelozE"][maybe] / np.log(10)), 
        color=bc.orange, marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_QuelozE"][def_consistent]), 
        np.log10(phot_bin_pleiades["VSINI"][def_consistent]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_QuelozE"][def_consistent] / 
              phot_bin_pleiades["vsini_QuelozE"][def_consistent] / np.log(10)), 
        color=bc.green, marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_QuelozC"][def_outliers]), 
        np.log10(phot_bin_pleiades["VSINI"][def_outliers]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_QuelozC"][def_outliers] / 
              phot_bin_pleiades["vsini_QuelozC"][def_outliers] / np.log(10)), 
        color="red", marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_QuelozC"][maybe]), 
        np.log10(phot_bin_pleiades["VSINI"][maybe]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_QuelozC"][maybe] / 
              phot_bin_pleiades["vsini_QuelozC"][maybe] / np.log(10)), 
        color=bc.orange, marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_QuelozC"][def_consistent]), 
        np.log10(phot_bin_pleiades["VSINI"][def_consistent]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_QuelozC"][def_consistent] / 
              phot_bin_pleiades["vsini_QuelozC"][def_consistent] / np.log(10)), 
        color=bc.green, marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_Terndrup"][def_outliers]), 
        np.log10(phot_bin_pleiades["VSINI"][def_outliers]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_Terndrup"][def_outliers] / 
              phot_bin_pleiades["vsini_Terndrup"][def_outliers] / np.log(10)), 
        color="red", marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_Terndrup"][maybe]), 
        np.log10(phot_bin_pleiades["VSINI"][maybe]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_Terndrup"][maybe] / 
              phot_bin_pleiades["vsini_Terndrup"][maybe] / np.log(10)), 
        color=bc.orange, marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_Terndrup"][def_consistent]), 
        np.log10(phot_bin_pleiades["VSINI"][def_consistent]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_Terndrup"][def_consistent] / 
              phot_bin_pleiades["vsini_Terndrup"][def_consistent] / np.log(10)), 
        color=bc.green, marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_S84"][def_outliers]), 
        np.log10(phot_bin_pleiades["VSINI"][def_outliers]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_S84"][def_outliers] / 
              phot_bin_pleiades["vsini_S84"][def_outliers] / np.log(10)), 
        color="red", marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_S84"][maybe]), 
        np.log10(phot_bin_pleiades["VSINI"][maybe]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_S84"][maybe] / 
              phot_bin_pleiades["vsini_S84"][maybe] / np.log(10)), 
        color=bc.orange, marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_S84"][def_consistent]), 
        np.log10(phot_bin_pleiades["VSINI"][def_consistent]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_S84"][def_consistent] / 
              phot_bin_pleiades["vsini_S84"][def_consistent] / np.log(10)), 
        color=bc.green, marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_SH"][def_outliers]), 
        np.log10(phot_bin_pleiades["VSINI"][def_outliers]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_SH"][def_outliers] / 
              phot_bin_pleiades["vsini_SH"][def_outliers] / np.log(10)), 
        color="red", marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_SH"][maybe]), 
        np.log10(phot_bin_pleiades["VSINI"][maybe]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_SH"][maybe] / 
              phot_bin_pleiades["vsini_SH"][maybe] / np.log(10)), 
        color=bc.orange, marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_SH"][def_consistent]), 
        np.log10(phot_bin_pleiades["VSINI"][def_consistent]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_SH"][def_consistent] / 
              phot_bin_pleiades["vsini_SH"][def_consistent] / np.log(10)), 
        color=bc.green, marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_Jackson"][def_outliers]), 
        np.log10(phot_bin_pleiades["VSINI"][def_outliers]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_Jackson"][def_outliers] / 
              phot_bin_pleiades["vsini_Jackson"][def_outliers] / np.log(10)), 
        color="red", marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_Jackson"][maybe]), 
        np.log10(phot_bin_pleiades["VSINI"][maybe]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_Jackson"][maybe] / 
              phot_bin_pleiades["vsini_Jackson"][maybe] / np.log(10)), 
        color=bc.orange, marker=".", ls="")
    ax.errorbar(
        np.log10(phot_bin_pleiades["vsini_Jackson"][def_consistent]), 
        np.log10(phot_bin_pleiades["VSINI"][def_consistent]), yerr=0.12/np.log(10), 
        xerr=(phot_bin_pleiades["vsini_err_Jackson"][def_consistent] / 
              phot_bin_pleiades["vsini_Jackson"][def_consistent] / np.log(10)), 
        color=bc.green, marker=".", ls="")
    ax.plot([1, 2], [1, 2], 'k-')
    ax.set_xlabel("Literature Vsini")
    ax.set_ylabel("APOGEE Vsini")

    print("Total: {0:d}".format(len(phot_bin_pleiades)))
    print("Total Consistent: {0:d}".format(np.count_nonzero(def_consistent)))
    print("Total Inconsistent: {0:d}".format(np.count_nonzero(def_outliers)))
    print("Total Ambiguous: {0:d}".format(np.count_nonzero(maybe)))
    return phot_bin_pleiades, full_matrix

def ElBadry_Pleiades_Comparison():
    '''See how well the labels in El-Badry agree with photometric binaries.'''
    apo_pleiades = cache.pleiades()

    f, ax = plt.subplots(1, 1, figsize=figsize)

    pleiad_groups = apo_pleiades.group_by("Binarity")
    for grp in pleiad_groups.groups:
        grpcls = grp["Binarity"][0]
        if grpcls in ["Single", "SB1"]:
            plt_kws = {"color": bc.black, "marker": "o", "ls":""}
        elif grpcls in ["SB2", "Hidden Triple", "SB3"]:
            plt_kws = {"color": "red", "marker": "o", "ls":""}
        else:
            plt_kws = {"color": "grey", "marker": ".", "ls": "", "alpha": 0.3}
        ax.plot(
            grp["(V-K)0"], grp["Dereddened K"], **plt_kws)

    hr.invert_y_axis(ax)
    ax.set_xlabel("V-K")
    ax.set_ylabel("M_K")

            


#########
# Rates #
#########

def list_problematic_vsini_rates_cool_dwarfs():
    '''Calculate the rate of objects with vsini too high for their periods.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    # Rate of DLSBs
    # Rate of problematic vsinis
    # Do this for both regular stars and PBs.
    dlsbs = aposplit.subsample([
        "Dwarfs", "APOGEE Evolution Cool", "DLSB", "~Unknown Mcq"])
    cool_apo_nomcq = aposplit.subsample([
        "Dwarfs", "APOGEE Evolution Cool", "~DLSB", "No Mcq"])
    cool_apo = aposplit.subsample(["Dwarfs", "APOGEE Evolution Cool", "Mcq", "~DLSB"])
    mcq = catin.read_McQuillan_catalog()
    cool_apo_mcq = au.join_by_id(cool_apo, mcq, "kepid", "KIC")

    # Problematic vsini are those with vsini - veq > 2 * err.
    vsini_lim = 10
    veq = rot.period_to_velocities(
        cool_apo_mcq["Prot"], cool_apo_mcq["Gaia R"])
    logverr = rot.logperiod_and_logradius_to_logvelocity_err(
        cool_apo_mcq["Gaia R err"] / cool_apo_mcq["Gaia R"] / np.log(10),
        cool_apo_mcq["e_Prot"] / cool_apo_mcq["Prot"] / np.log(10))
    too_rapid = np.count_nonzero(
        np.log10(cool_apo_mcq["VSINI"]) - 
        np.log10(np.where(cool_apo_mcq["VSINI"] > vsini_lim, veq, vsini_lim)) > 
        2 * np.sqrt((0.15/np.log(10))**2 + logverr**2))
    # Problematic vsinis are also rapid rotators without period detections.
    high_vsini_nomcq = np.count_nonzero(
        np.log10(cool_apo_nomcq["VSINI"]) - np.log10(vsini_lim) > 
        0.15/np.log(10))

    totalsamp = len(dlsbs) + len(cool_apo) + len(cool_apo_nomcq)
    print("Number of SB2: {0}/{1} = {2:.1f}%".format(
        len(dlsbs), totalsamp, len(dlsbs) / totalsamp * 100))
    print("Number of problematic vsini: {0}/{1} = {2:.1f}%".format(
        too_rapid + high_vsini_nomcq, totalsamp, 
        (too_rapid + high_vsini_nomcq) / totalsamp * 100))

    # Now normalize to the number of photometric binaries.
    dlsb_photindices = dlsbs["K Excess"] < -0.2
    mcq_photindices = cool_apo_mcq["K Excess"] < -0.2
    nomcq_photindices = cool_apo_nomcq["K Excess"] < -0.2

    # Counts for photometric binaries
    num_dlsbs_pb = np.count_nonzero(dlsb_photindices)
    too_rapid_pb = np.count_nonzero(
        np.log10(cool_apo_mcq["VSINI"][mcq_photindices]) - 
        np.log10(np.where(
            cool_apo_mcq["VSINI"][mcq_photindices] > vsini_lim, veq[mcq_photindices], 
            vsini_lim)) > 2 * np.sqrt(
                (0.15/np.log(10))**2 + logverr[mcq_photindices]**2))
    high_vsini_nomcq_pb = np.count_nonzero(
        np.log10(cool_apo_nomcq["VSINI"][nomcq_photindices]) - np.log10(vsini_lim) > 
        0.15/np.log(10))

    totalphotbins = (
        num_dlsbs_pb + np.count_nonzero(mcq_photindices) + 
        np.count_nonzero(nomcq_photindices))
    print("Now for photometric binaries")
    print("Number of SB2: {0}/{1} = {2:.1f}%".format(
        num_dlsbs_pb, totalphotbins, num_dlsbs_pb / totalphotbins * 100))
    print("Number of problematic vsini: {0}/{1} = {2:.1f}%".format(
        too_rapid_pb + high_vsini_nomcq_pb, totalphotbins, 
        (too_rapid_pb + high_vsini_nomcq_pb) / totalphotbins * 100))

##########
# Errors #
##########

def compare_error_derivation(temp, period, period_err):
    '''Compare full to quick error derivation.

    This function will calculate a distribution of v_eq in two ways. First, by
    actually doing the full numerical integral, and second by approximating it
    as a Gaussian. '''
    # I want to make this plot for both a hot and cool star.
    lowteff = 4000
    highteff = 6000
    teff_err = 100

    # I also want a rapid and slow rotator.
    low_period = 1.0
    high_period = 30
    # And a precise and imprecise error
    high_frac_period_err = 0.1
    low_frac_period_err = 0.005

    if temp == "low":
        teff = lowteff
    else:
        teff = highteff

    if period == "low":
        p = low_period
    else:
        p = high_period

    if period_err == "low":
        p_err = low_frac_period_err
    else:
        p_err = high_frac_period_err


    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    v_vals = np.linspace(1, 100+1, 100, endpoint=True)
    def joint_dist(T, T0, P0, Terr, Perr, v):
        P = (2 * np.pi * samp.calc_model_over_feh_fixed_age_alpha(
            np.log10(T), mist.MISTIsochrone.logteff_col,
            mist.MISTIsochrone.radius_col, 0.0, 1e9) / v)
        (1/2/np.pi/Terr/Perr*(
            np.exp(-(T-T0)**2/2/Terr**2)*np.exp(-(P-P0)**2/2/Perr**2)))
    pdf = np.array([quad(joint_dist, -np.inf, np.inf, args=(
        teff, p, teff_err, p_err, v)) for v in v_vals])
    plt.plot(v_vals, pdf, 'k-')

def compare_Gaia_to_asteroseismic_radius_errors():
    '''Plot the Gaia and Asteroseismic radius measurements with errors.'''
    astero = cache.astero_splitter()

    astero_dwarfs = astero.subsample([
        "~Bad", "Asteroseismic Dwarfs", "~No APOGEE Teff"])

    # Derive R using Gaia parallaxes
    apogee_logteff_err = (
        astero_dwarfs["TEFF_COR_ERR"] / astero_dwarfs["TEFF_COR"] / np.log(10))
    astero_dwarfs["MIST BC (sol)"] = samp.calc_model_over_feh_fixed_age_alpha(
        np.log10(astero_dwarfs["TEFF_COR"]), mist.MISTIsochrone.logteff_col,
        "BC K", 0.0, 1e9)
    astero_dwarfs["MIST BC err"] = samp.calc_model_err_fixed_age_feh_alpha(
        np.log10(astero_dwarfs["TEFF_COR"]), mist.MISTIsochrone.logteff_col, 
        "BC K", apogee_logteff_err, 0.0, age=1e9)
    # Add the zero-point offset.
    astero_dwarfs["Gaia L"] = 10**(
        -0.4 * (astero_dwarfs["M_K"] + astero_dwarfs["MIST BC (sol)"] - 4.74))
    astero_dwarfs["Gaia L err"] = (
        0.4 * np.log(10) * astero_dwarfs["Gaia L"] *np.sqrt(
            astero_dwarfs["K_MAG_ERR"]**2 + (
                5 * astero_dwarfs["parallax_error"] / astero_dwarfs["parallax"] /
                np.log(10))**2 + astero_dwarfs["MIST BC err"]**2))
    astero_dwarfs["Gaia R"] = 10**(
        0.5*(np.log10(astero_dwarfs["Gaia L"]) - 4*(
            np.log10(astero_dwarfs["TEFF_COR"]) - np.log10(5777))))
    astero_dwarfs["Gaia R err"] = (
        astero_dwarfs["Gaia R"] * np.log(10) * np.sqrt(
            (0.2*astero_dwarfs["K_MAG_ERR"])**2 + 
            (astero_dwarfs["parallax_error"] / astero_dwarfs["parallax"] /
             np.log(10))**2 +
            (2 * apogee_logteff_err)**2 + 
            (0.2 * samp.calc_model_err_fixed_age_feh_alpha(
                np.log10(astero_dwarfs["TEFF_COR"]), 
                mist.MISTIsochrone.logteff_col, mist.MISTIsochrone.radius_col, 
                apogee_logteff_err, 0.0)**2)))

    symmetric_astero_uncertainty = ((
        astero_dwarfs["RADIUS_DW_PERR"] - astero_dwarfs["RADIUS_DW_MERR"])/2)
    symmetric_gaia_uncertainty = astero_dwarfs["Gaia R err"]

    chi_squared = ((astero_dwarfs["Gaia R"] - astero_dwarfs["RADIUS_DW"])**2 /
                   (symmetric_astero_uncertainty**2 +
                    symmetric_gaia_uncertainty**2))

    # Flag objects more than 3-sigma away.
    outliers = chi_squared > 9
    print(np.count_nonzero(~outliers.mask))

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.errorbar(
        astero_dwarfs["RADIUS_DW"], astero_dwarfs["Gaia R"],
        yerr=astero_dwarfs["Gaia R err"],
        xerr=[astero_dwarfs["RADIUS_DW_PERR"], -astero_dwarfs["RADIUS_DW_MERR"]], 
        color="k", linestyle="", marker=".")
    ax1.errorbar(
        astero_dwarfs["RADIUS_DW"][outliers], astero_dwarfs["Gaia R"][outliers],
        yerr=astero_dwarfs["Gaia R err"][outliers],
        xerr=[astero_dwarfs["RADIUS_DW_PERR"][outliers], 
              -astero_dwarfs["RADIUS_DW_MERR"][outliers]], 
        color="r", linestyle="", marker=".")
    ax1.plot([0, 5], [0, 5], 'r-')
    ax1.set_xlabel("Asteroseismic Radius")
    ax1.set_ylabel("Gaia Radius")
    ax1.set_xlim(0.8, 5.0)
    ax1.set_ylim(0.8, 5.0)
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    astero_fractional_error = (
        (astero_dwarfs["RADIUS_DW_PERR"] - astero_dwarfs["RADIUS_DW_MERR"])/2 
        / astero_dwarfs["RADIUS_DW"])
    gaia_fractional_error = (
        astero_dwarfs["Gaia R err"] / astero_dwarfs["Gaia R"])


    ax2.plot(astero_fractional_error, gaia_fractional_error, 'ko', label="")
    ax2.plot([np.median(astero_fractional_error)],
             [np.median(gaia_fractional_error)], 'ro', label="Median")
    ax2.set_xlabel("Asteroseismic Fractional Radius Error")
    ax2.set_ylabel("Gaia Fractional Radius Error")
    ax2.set_xlim(0.0, 0.12)
    ax2.set_ylim(0.0, 0.12)
    ax2.legend(loc="lower left")

                


                        

###########
# Periods #
###########

@write_plot("periodcomp")
def compare_McQuillan_Garcia_periods():
    '''Plot McQuillan vs Garcia period.'''
    mcq = catin.read_McQuillan_catalog()
    garcia = catin.read_Garcia_periods()
    overlap = au.join_by_id(
        mcq, garcia, "KIC", "KIC", conflict_suffixes=("_Mcq", "_Gar"))

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.errorbar(
        overlap["Prot_Gar"], overlap["Prot_Mcq"], yerr=overlap["e_Prot_Mcq"],
        xerr=overlap["e_Prot_Gar"], color="k", marker=".", ls="")
    ax.plot([0, 55], [0, 55], 'k-')
    ax.set_xlabel("Garcia Period (day)")
    ax.set_ylabel("McQuillan Period (day)")
    print("Size of overlap sample is {0:d}.".format(len(overlap)))

    chisq = np.sum((overlap["Prot_Gar"] - overlap["Prot_Mcq"])**2 / (
        overlap["e_Prot_Gar"]**2 + overlap["e_Prot_Gar"]**2))
    chisq_dof = chisq / len(overlap)
    print("Chi-squared per degree of freedom: {0:.2f}".format(chisq_dof))


#############
# Subgiants #
#############

def subgiant_mass_hr_diagram():
    '''Plot mass on HR diagram for asteroseismic dwarfs.'''
    astero = cache.astero_splitter()
    apokasc = astero.subsample([
        "Asteroseismic Dwarfs", "~Bad"])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    masscolors = plt.get_cmap("viridis")
    norm = Normalize()
    norm_colors = norm(apokasc["MASS_DW"])
    ax.scatter(
        apokasc["TEFF_COR"], apokasc["M_K"],
        color=masscolors(norm_colors), marker=".")
    hr.invert_x_axis(ax)
    hr.invert_y_axis(ax)

def subgiant_period_hr_diagram():
    '''Plot period on HR diagram for asteroseismic dwarfs.'''
    astero = cache.astero_splitter()
    apokasc = astero.subsample([
        "Asteroseismic Dwarfs", "~Bad"])
    garcia = catin.read_Garcia_periods()
    apokasc_period = au.join_by_id(apokasc, garcia, "KEPLER_INT", "KIC")

    # Add MIST Evolutionary Tracks
    mass10 = mist.MISTEvolutionaryTrack.track_from_file(1.0, 0.0)
    mass12 = mist.MISTEvolutionaryTrack.track_from_file(1.2, 0.0)
    mass14 = mist.MISTEvolutionaryTrack.track_from_file(1.4, 0.0)
    mass16 = mist.MISTEvolutionaryTrack.track_from_file(1.6, 0.0)
    mass18 = mist.MISTEvolutionaryTrack.track_from_file(1.8, 0.0)
    mass20 = mist.MISTEvolutionaryTrack.track_from_file(2.0, 0.0)
    masses = [mass10, mass12, mass14, mass16, mass18, mass20]

    trunc_tracks = []
    for m in masses:
        ms_start = min(np.where(m.tracktable["phase"] == 0)[0])
        ms_end = max(np.where(m.tracktable["phase"] == 0)[0])+1
        ms_table = m.tracktable[ms_start:ms_end]
        subgiant_index = max(np.where(ms_table[m.logg_col] > 3.5)[0])+1

        trunc_tracks.append(ms_table[:subgiant_index])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    rotcolors = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=np.log10(max(apokasc_period["Prot"])))
    norm_colors = norm(apokasc_period["Prot"])
    print(norm_colors)
    sc = ax.scatter(
        apokasc_period["TEFF_COR"], apokasc_period["M_K"],
        c=np.log10(apokasc_period["Prot"]), marker="o", cmap=rotcolors, norm=norm)
    f.colorbar(sc, ax=ax)
    ax.set_xlim(7000, 5000)
    hr.invert_y_axis(ax)

    for t in trunc_tracks:
        hr.absmag_teff_plot(
            10**t[mass10.logteff_col], t[mist.band_translation["Ks"]], color='k',
            marker="", linestyle="-")



def subgiant_mass_age_diagram():
    '''Plot correlation between mass and age.'''
    astero = cache.astero_splitter()
    apokasc = astero.subsample([
        "Asteroseismic Dwarfs", "~Bad"])
    garcia = catin.read_Garcia_periods()
    apokasc_period = au.join_by_id(apokasc, garcia, "KEPLER_INT", "KIC")

    # This is a rough method to weed out rapid rotators.
    rapid_subgiants = au.multi_logical_or(
        np.logical_and(
            apokasc_period["MASS_DW"] > 1.5, apokasc_period["Prot"] < 3),
        np.logical_and(
            np.logical_and(
                apokasc_period["MASS_DW"] > 1.1, 
                apokasc_period["MASS_DW"] < 1.5), apokasc_period["Prot"] < 10),
        np.logical_and(
            apokasc_period["MASS_DW"] < 1.1, apokasc_period["Prot"] < 40))


    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(apokasc_period["MASS_DW"][rapid_subgiants],
            apokasc_period["Prot"][rapid_subgiants], 'ro')
    ax.plot(apokasc_period["MASS_DW"][~rapid_subgiants],
            apokasc_period["Prot"][~rapid_subgiants], 'k.')

def astero_period_age_binned_masses():
    '''Show gyrochronology for three mass bins.'''
    astero = cache.astero_splitter()
    apokasc = astero.subsample([
        "Asteroseismic Dwarfs", "~Bad"])
    garcia = catin.read_Garcia_periods()
    apokasc_period = au.join_by_id(apokasc, garcia, "KEPLER_INT", "KIC")

    mass_bin_edges = np.array([1.1, 1.5])
    mass_bins = np.digitize(apokasc_period["MASS_DW"], mass_bin_edges)

    formats = ['r.', 'b.', 'g.']
    f, ax = plt.subplots(1, 1, figsize=figsize)
    for i in range(3):
        mass_indices = (mass_bins == i)
        ax.plot(apokasc_period["Prot"][mass_indices],
                apokasc_period["AGE_DW"][mass_indices], formats[i])


def subgiant_period_temperature_ages():
    '''Verify correlation between asteroseismic age with period and temperature.'''
    astero = cache.astero_splitter()
    apokasc = astero.subsample([
        "Asteroseismic Dwarfs", "~Bad"])
    garcia = catin.read_Garcia_periods()
    apokasc_period = au.join_by_id(apokasc, garcia, "KEPLER_INT", "KIC")

    f, ax = plt.subplots(1, 1, figsize=figsize)
    agecolors = plt.get_cmap("viridis")
    norm = Normalize()
    norm_colors = norm(apokasc_period["AGE_DW"])
    ax.scatter(
        apokasc_period["TEFF_COR"], apokasc_period["Prot"],
        color=agecolors(norm_colors), marker=".")
    hr.invert_x_axis(ax)

def luminous_subgiant_vsini_period_comparison():
    '''Compare vsini to rotation period using Gaia radii.'''
    full = cache.apogee_splitter_with_DSEP()
    fulltable = full.subsample(["Luminous Subgiants", "~DLSB", "~No Vsini"])
    mcquillan = catin.read_Garcia_periods()
    mcq_combo = au.join_by_id(fulltable, mcquillan, "kepid", "KIC")

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    rot.plot_vsini_velocity(
        mcq_combo["VSINI"], mcq_combo["Prot"], 
        mcq_combo["e_Prot"], mcq_combo["Gaia R"], 
        mcq_combo["Gaia R err"], ax=ax)

#   ax.set_xlabel("Veq")
#   ax.set_ylabel("Vsini")
    ax.set_title("Luminous Subgiant vsini agreement")

def subgiant_vsini_period_comparison():
    '''Compare vsini to rotation period using Gaia radii.'''
    full = cache.apogee_splitter_with_DSEP()
    fulltable = full.subsample(["Subgiants", "~DLSB", "~No Vsini"])
    garcia = catin.read_Garcia_periods()
    mcq_combo = au.join_by_id(fulltable, garcia, "kepid", "KIC")

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    rot.plot_vsini_velocity(
        mcq_combo["VSINI"], mcq_combo["Prot"], 
        mcq_combo["e_Prot"], mcq_combo["Gaia R"], 
        mcq_combo["Gaia R err"], ax=ax)

#   ax.set_xlabel("Veq")
#   ax.set_ylabel("Vsini")
    ax.set_title("Subgiant vsini agreement")

@write_plot("Subgiant_vsini_veq_comparison")
def subgiant_veq_agreement_Lbol():
    '''Plot the vsini and veq in a single plot with bolometric R.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    dlsbs = aposplit.subsample(["Subgiants", "Mcq", "DLSB"])
    # Both Singles and SB1s count as singles.
    subgiant_singles = aposplit.subsample([
        "Subgiants", "Mcq", "~DLSB", "El-Badry Single"])
    subgiant_SB1s = aposplit.subsample([
        "Subgiants", "Mcq", "~DLSB", "El-Badry SB1"])
    subgiant_single_group = vstack([subgiant_singles, subgiant_SB1s])
    # El-Badry multiples are in SB2s, Hidden Triples, and SB3s.
    subgiant_SB2s = aposplit.subsample([
        "Subgiants", "Mcq", "~DLSB", "El-Badry SB2"])
    subgiant_hts = aposplit.subsample([
        "Subgiants", "Mcq", "~DLSB", "El-Badry Hidden Triple"])
    subgiant_SB3s = aposplit.subsample([
        "Subgiants", "Mcq", "~DLSB", "El-Badry SB3"])
    evolved_multiples = vstack([
        subgiant_SB2s, subgiant_hts, subgiant_SB3s])
    no_subgiant = aposplit.subsample([
        "Subgiants", "Mcq", "~DLSB", "No El-Badry Binarity"])
    mcq = catin.read_McQuillan_catalog()
    dlsb_mcq = au.join_by_id(dlsbs, mcq, "kepid", "KIC")
    subgiant_single_mcq = au.join_by_id(
        subgiant_single_group, mcq, "kepid", "KIC")
    evolved_multiple_mcq = au.join_by_id(evolved_multiples, mcq, "kepid", "KIC")
    no_subgiant_mcq = au.join_by_id(no_subgiant, mcq, "kepid", "KIC")


    f, ax = plt.subplots(1, 1, figsize=figsize)
    rot.plot_vsini_velocity(
        dlsb_mcq["VSINI"], dlsb_mcq["Prot"], dlsb_mcq["e_Prot"],
        dlsb_mcq["Gaia R"], dlsb_mcq["Gaia R err"], ax=ax, color=bc.sky_blue,
        marker="*", label="SB2", ms=10)
    rot.plot_vsini_velocity(
        subgiant_single_mcq["VSINI"], subgiant_single_mcq["Prot"], 
        subgiant_single_mcq["e_Prot"], subgiant_single_mcq["Gaia R"], 
        subgiant_single_mcq["Gaia R err"], ax=ax, color=bc.black, marker="s", 
        label="Subgiant Single")
    rot.plot_vsini_velocity(
        evolved_multiple_mcq["VSINI"], evolved_multiple_mcq["Prot"], 
        evolved_multiple_mcq["e_Prot"], evolved_multiple_mcq["Gaia R"], 
        evolved_multiple_mcq["Gaia R err"], ax=ax, color=bc.sky_blue,
        marker="s", label="El-Badry Multiple")
    rot.plot_vsini_velocity(
        no_subgiant_mcq["VSINI"], no_subgiant_mcq["Prot"], 
        no_subgiant_mcq["e_Prot"], no_subgiant_mcq["Gaia R"], 
        no_subgiant_mcq["Gaia R err"], ax=ax, label="No El-Badry Subgiants",
        color=bc.pink, marker="s")
    ax.plot([1, 100], [1.15, 115], color='k', ls="-.", marker="")
    ax.set_title("Subgiants")
    ax.legend(loc="lower right")

def subgiant_vsini_outliers():
    '''Select off the objects which are outliers in subgiant vsini.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    subgiants = aposplit.subsample(["Subgiants", "Mcq", "~DLSB"])
    subgiant_dlsbs = aposplit.subsample(["Subgiants", "Mcq", "DLSB"])
    luminous_subgiants = aposplit.subsample(
        ["Luminous Subgiants", "Mcq", "~DLSB"])
    luminous_dlsbs = aposplit.subsample(["Luminous Subgiants", "Mcq", "DLSB"])
    total = vstack([subgiants, luminous_subgiants])
    dlsbs = vstack([subgiant_dlsbs, luminous_dlsbs])
    mcq = catin.read_McQuillan_catalog()
    total_mcq = au.join_by_id(total, mcq, "kepid", "KIC")
    dlsbs_mcq = au.join_by_id(dlsbs, mcq, "kepid", "KIC")

    vels = rot.period_to_velocities(total_mcq["Prot"], total_mcq["Gaia R"])
    outliers = (total_mcq["VSINI"] / np.maximum(vels, 10) > 1.5)

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    hr.absmag_teff_plot(
        total_mcq["TEFF"][~outliers], total_mcq["M_K"][~outliers], marker=".",
        color="k", ls="")
    hr.absmag_teff_plot(
        total_mcq["TEFF"][outliers], total_mcq["M_K"][outliers], marker=".",
        color="r", ls="")

    return total_mcq[outliers]

@write_plot("subgiant_vdists")
def subgiant_velocity_comparison():
    '''Plot the agreement between predicted and actual vsini.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    subgiants = aposplit.subsample(["Subgiants", "Mcq"])
    luminous_subgiants = aposplit.subsample(["Luminous Subgiants", "Mcq"])
    total = vstack([subgiants, luminous_subgiants])
    mcq = catin.read_McQuillan_catalog()
    total_mcq = au.join_by_id(total, mcq, "kepid", "KIC")

    # I want to make sure upper limits are actually detected as lower limits.
    total_velocities = rot.period_to_velocities(
        total_mcq["Prot"], total_mcq["Gaia R"])
    sini_cutoff = 0.5

    f, ax = plt.subplots(1, 1, figsize=figsize)
    rot.plot_vsini_dist(
        np.log10(total_velocities), ax=ax, sini_cutoff=sini_cutoff)
    
def photometric_binary_fraction_met():
    '''Calculate how the photometric binary fraction changes with metallicity.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    met_bins = ["Low Met", "Sol Met", "High Met"]

    for met in met_bins:
        binaries = aposplit.subsample_len([met, "Photometric Binaries"])
        singles = aposplit.subsample_len([met, "Cool Dwarfs"])

        binary_frac = binaries / (binaries + singles)
        fractemp = "{0:d}/{1:d} = {2:.1f}%"
        print(fractemp.format(binaries, binaries+singles, binary_frac*100))

def plot_high_met_photbins():
    '''Plot the photometric binaries at high metallicity.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    binaries = aposplit.subsample(["High Met", "Photometric Binaries"])
    singles = aposplit.subsample(["High Met", "Cool Dwarfs"])


    f, ax = plt.subplots(1, 1, figsize=figsize)

    hr.absmag_teff_plot(
        binaries["TEFF"], binaries["K Excess"], color=bc.green, marker=".", ls="")
    hr.absmag_teff_plot(
        singles["TEFF"], singles["K Excess"], color=bc.black, marker=".", ls="")
        


@write_plot("f13")
def jen_subgiant_boundary_mets():
    '''Plot the objects against Jen's boundary.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    # Solar Metallicity 
    solmet_dets = aposplit.subsample(
        ["~Giants", "Vsini det", "Sol Met"])
    solmet_nondets = aposplit.subsample(
        ["~Giants", "~Vsini det", "Sol Met"])
    solmet_lowers = aposplit.subsample(
        ["~Giants", "Vsini lower", "Sol Met"])
    solmet_marginal = aposplit.subsample(
        ["~Giants", "Vsini marginal", "Sol Met"])
    solmet_novsini = aposplit.subsample(
        ["~Giants", "No Vsini", "Sol Met"])
    # Low Metallicity
    lowmet_lowers = aposplit.subsample(
        ["~Giants", "Vsini lower", "Low Met"])
    lowmet_dets = aposplit.subsample(
        ["~Giants", "Vsini det", "Low Met"])
    lowmet_marginal = aposplit.subsample(
        ["~Giants", "Vsini marginal", "Low Met"])
    lowmet_nondets = aposplit.subsample(
        ["~Giants", "Vsini nondet", "Low Met"])
    lowmet_novsini = aposplit.subsample(
        ["~Giants", "No Vsini", "Low Met"])
    # High Metallicity
    highmet_lowers = aposplit.subsample(
        ["~Giants", "Vsini lower", "High Met"])
    highmet_dets = aposplit.subsample(
        ["~Giants", "Vsini det", "High Met"])
    highmet_marginal = aposplit.subsample(
        ["~Giants", "Vsini marginal", "High Met"])
    highmet_nondets = aposplit.subsample(
        ["~Giants", "Vsini nondet", "High Met"])
    highmet_novsini = aposplit.subsample(
        ["~Giants", "No Vsini", "High Met"])

    jen_fast = jen.read_Jen_SGB_fast_launch()
    jen_slow = jen.read_Jen_SGB_slow_launch()
    f, axes = plt.subplots(1, 3, figsize=(27,9))
    for (met, lowers, dets, marginal, nondets, novsini, ax) in zip(
             [-0.2, 0.0, 0.2], [lowmet_lowers, solmet_lowers, highmet_lowers], 
             [lowmet_dets, solmet_dets, highmet_dets],
             [lowmet_marginal, solmet_marginal, highmet_marginal],
             [lowmet_nondets, solmet_nondets, highmet_nondets], 
             [lowmet_novsini, solmet_novsini, highmet_novsini], axes):
#       jen.make_additional_columns(met, 10, jen_fast)
#       jen.make_additional_columns(met, 10, jen_slow)
#       jen_fast_teff = jen_fast[jen.format_Jen_column(met, 10, "Teff")]
#       jen_fast_M_K = jen_fast[jen.format_Jen_column(met, 10, "M_K")]
#       jen_slow_teff = jen_slow[jen.format_Jen_column(met, 10, "Teff")]
#       jen_slow_M_K = jen_slow[jen.format_Jen_column(met, 10, "M_K")]


        hr.absmag_teff_plot(
            novsini["TEFF"], novsini["M_K"], color=bc.black, marker=".", ls="",
            alpha=0.3, axis=ax, label="")
        hr.absmag_teff_plot(
            nondets["TEFF"], nondets["M_K"], color=bc.black, marker=".", ls="",
            alpha=0.3, axis=ax, label="")
        hr.absmag_teff_plot(
            marginal["TEFF"], marginal["M_K"], color=bc.black, marker=".", ls="",
            alpha=0.3, axis=ax, label="")
        hr.absmag_teff_plot(
            dets["TEFF"], dets["M_K"], color=bc.pink, marker="o", ls="", axis=ax,
            label="{0} > 10 km/s".format(vsinistr))
        hr.absmag_teff_plot(
            lowers["TEFF"], lowers["M_K"], color=bc.pink, marker="o", ls="",
            axis=ax, label="")

#       hr.absmag_teff_plot(
#           jen_fast_teff, jen_fast_M_K, color=bc.brown, marker="", ls="-",
#           label="Fast launch", axis=ax, lw=3)
#       hr.absmag_teff_plot(
#           jen_slow_teff, jen_slow_M_K, color=bc.brown, marker="", ls="--",
#           label="Slow launch", axis=ax, lw=3)

        if met == -0.2:
            title = r"[Fe/H] $\leq -0.2$"
        elif met == 0.0:
            title = r"$-0.2 < $[Fe/H]$ \leq 0.2$"
        elif met == 0.2:
            title = r"[Fe/H] $> 0.2$"

    
        ax.set_xlabel("{0} (K)".format(Teffstr))
        ax.set_ylabel("")
        ax.set_title(title)
        ax.set_ylim(6.3, -1.3)
        ax.set_xlim(6600, 3600)
    axes[0].set_ylabel(MKstr)
    axes[0].legend(loc="lower left")
    plt.tight_layout()
    
def metallicity_samples_postdoc():
    '''Plot the objects against Jen's boundary.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    # Low Metallicity
    lowmet_singles = aposplit.subsample(
        ["Dwarfs", "APOGEE Evolution Cool", "Low Met", "Photometric Singles"])
    lowmet_binaries = aposplit.subsample(
        ["Dwarfs", "APOGEE Evolution Cool", "Low Met", "Photometric Bins"])
    # High Metallicity
    highmet_singles = aposplit.subsample(
        ["Dwarfs", "APOGEE Evolution Cool", "High Met", "Photometric Singles"])
    highmet_binaries = aposplit.subsample(
        ["Dwarfs", "APOGEE Evolution Cool", "High Met", "Photometric Bins"])

    f, axes = plt.subplots(1, 2, figsize=(18,9))

    hr.absmag_teff_plot(
        lowmet_singles["TEFF"], lowmet_singles["M_K"], ls="", marker=".",
        color="k", axis=axes[0], label="Single Stars")
    hr.absmag_teff_plot(
        highmet_singles["TEFF"], highmet_singles["M_K"], ls="", marker=".",
        color="k", axis=axes[1])
    hr.absmag_teff_plot(
        lowmet_binaries["TEFF"], lowmet_binaries["M_K"], ls="", marker="o",
        color="r", axis=axes[0], label="Photometric Binaries")
    hr.absmag_teff_plot(
        highmet_binaries["TEFF"], highmet_binaries["M_K"], ls="", marker="o",
        color="r", axis=axes[1])

    hr.absmag_teff_plot(
        [4100], [3.0], yerr=np.median(lowmet_singles["M_K_err1"]),
        xerr=np.median(lowmet_singles["TEFF_ERR"]), axis=axes[0], color="k") 
    hr.absmag_teff_plot(
        [4100], [3.0], yerr=np.median(highmet_singles["M_K_err1"]),
        xerr=np.median(highmet_singles["TEFF_ERR"]), axis=axes[1], color="k") 

    print(np.count_nonzero(highmet_singles[highmet_singles["TEFF"] < 5000]))
    print(np.count_nonzero(highmet_binaries[highmet_binaries["TEFF"] < 5000]))
    print(np.count_nonzero(lowmet_singles[lowmet_singles["TEFF"] < 5000]))
    print(np.count_nonzero(lowmet_binaries[lowmet_binaries["TEFF"] < 5000]))

    # Plot the isochrones
    lowmet_sortargs = np.argsort(lowmet_singles["TEFF"])
    highmet_sortargs = np.argsort(highmet_singles["TEFF"])
    hr.absmag_teff_plot(
        lowmet_singles["TEFF"][lowmet_sortargs], 
        lowmet_singles["MIST K (sol)"][lowmet_sortargs], ls="-", 
        marker="", color=bc.pink, lw=5, zorder=5, label="MIST (1 Gyr)", 
        axis=axes[0])
    hr.absmag_teff_plot(
        highmet_singles["TEFF"][highmet_sortargs], 
        highmet_singles["MIST K (sol)"][highmet_sortargs], ls="-", 
        marker="", color=bc.pink, lw=5, zorder=5, label="", 
        axis=axes[1])

    axes[0].set_xlabel("{0} (K)".format(Teffstr))
    axes[1].set_xlabel("{0} (K)".format(Teffstr))
    axes[0].set_ylabel(MKstr)
    axes[1].set_ylabel("")
    axes[0].set_title("$[Fe/H] < -0.2$".format(fehstr))
    axes[1].set_title("$[Fe/H] > +0.2$".format(fehstr))
    axes[0].set_ylim(5.3, 2.5)
    axes[0].set_xlim(5000, 3900)
    axes[1].set_ylim(5.3, 2.5)
    axes[1].set_xlim(5000, 3900)
    axes[0].set_ylabel(MKstr)
    axes[0].legend(loc="lower left")
    plt.tight_layout()
    
def metallicity_samples():
    '''Plot the objects against Jen's boundary.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    # Low Metallicity
    lowmet_singles = aposplit.subsample(
        ["Dwarfs", "APOGEE Evolution Cool", "Low Met", "Photometric Singles"])
    lowmet_binaries = aposplit.subsample(
        ["Dwarfs", "APOGEE Evolution Cool", "Low Met", "Photometric Bins"])
    # High Metallicity
    highmet_singles = aposplit.subsample(
        ["Dwarfs", "APOGEE Evolution Cool", "High Met", "Photometric Singles"])
    highmet_binaries = aposplit.subsample(
        ["Dwarfs", "APOGEE Evolution Cool", "High Met", "Photometric Bins"])

    

@write_plot("subgiants")
def jen_subgiants():
    '''Plot the subgiants and boundary in radius space.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    # First do the cool dwarf vsini detections.
    cool_lowers = aposplit.subsample(
        ["Cool Dwarfs", "Vsini lower", "Low Alpha"])
    cool_dets = aposplit.subsample(
        ["Cool Dwarfs", "Vsini det", "Low Alpha"])
    # Now hot dwarf vsini detections.
    hot_lowers = aposplit.subsample(
        ["Hot Dwarfs", "Vsini lower", "Low Alpha"])
    hot_dets = aposplit.subsample(
        ["Hot Dwarfs", "Vsini det", "Low Alpha"])
    # Now the subgiant vsini detections.
    sub_lowers = aposplit.subsample(
        ["Subgiants", "Vsini lower", "Low Alpha"])


    

@write_plot("subgiants")
def jen_subgiants():
    '''Plot the subgiants and boundary in radius space.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    # First do the cool dwarf vsini detections.
    cool_lowers = aposplit.subsample(
        ["Cool Dwarfs", "Vsini lower", "Low Alpha"])
    cool_dets = aposplit.subsample(
        ["Cool Dwarfs", "Vsini det", "Low Alpha"])
    # Now hot dwarf vsini detections.
    hot_lowers = aposplit.subsample(
        ["Hot Dwarfs", "Vsini lower", "Low Alpha"])
    hot_dets = aposplit.subsample(
        ["Hot Dwarfs", "Vsini det", "Low Alpha"])
    # Now the subgiant vsini detections.
    sub_lowers = aposplit.subsample(
        ["Subgiants", "Vsini lower", "Low Alpha"])
    sub_dets = aposplit.subsample(
        ["Subgiants", "Vsini det", "Low Alpha"])
    # Now the luminous subgiant vsini detections.
    lum_sub_lowers = aposplit.subsample(
        ["Luminous Subgiants", "Vsini lower", "Low Alpha"])
    lum_sub_dets = aposplit.subsample(
        ["Luminous Subgiants", "Vsini det", "Low Alpha"])
    # Now the nonrotators
    marginal = aposplit.subsample(
        ["~Giants", "Vsini marginal", "Low Alpha"])
    nondets = aposplit.subsample(
        ["~Giants", "Vsini nondet", "Low Alpha"])
    novsini = aposplit.subsample(
        ["~Giants", "No Vsini", "Low Alpha"])
    # Now separate the high alphas
    alphalowers = aposplit.subsample([
        "~Giants", "Vsini lower", "High Alpha"])
    alphadets = aposplit.subsample([
        "~Giants", "Vsini det", "High Alpha"])
    alphamarginal = aposplit.subsample([
        "~Giants", "Vsini marginal", "High Alpha"])
    alphanondets = aposplit.subsample([
        "~Giants", "Vsini nondet", "High Alpha"])
    alphanovsini = aposplit.subsample([
        "~Giants", "No Vsini", "High Alpha"])

@write_plot("subgiants")
def jen_subgiants():
    '''Plot the subgiants and boundary in radius space.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    # First do the cool dwarf vsini detections.
    cool_lowers = aposplit.subsample(
        ["Cool Dwarfs", "Vsini lower", "Low Alpha"])
    cool_dets = aposplit.subsample(
        ["Cool Dwarfs", "Vsini det", "Low Alpha"])
    # Now hot dwarf vsini detections.
    hot_lowers = aposplit.subsample(
        ["Hot Dwarfs", "Vsini lower", "Low Alpha"])
    hot_dets = aposplit.subsample(
        ["Hot Dwarfs", "Vsini det", "Low Alpha"])
    # Now the subgiant vsini detections.
    sub_lowers = aposplit.subsample(
        ["Subgiants", "Vsini lower", "Low Alpha"])
    sub_dets = aposplit.subsample(
        ["Subgiants", "Vsini det", "Low Alpha"])
    # Now the luminous subgiant vsini detections.
    lum_sub_lowers = aposplit.subsample(
        ["Luminous Subgiants", "Vsini lower", "Low Alpha"])
    lum_sub_dets = aposplit.subsample(
        ["Luminous Subgiants", "Vsini det", "Low Alpha"])
    # Now the nonrotators
    marginal = aposplit.subsample(
        ["~Giants", "Vsini marginal", "Low Alpha"])
    nondets = aposplit.subsample(
        ["~Giants", "Vsini nondet", "Low Alpha"])
    novsini = aposplit.subsample(
        ["~Giants", "No Vsini", "Low Alpha"])
    # Now separate the high alphas
    alphalowers = aposplit.subsample([
        "~Giants", "Vsini lower", "High Alpha"])
    alphadets = aposplit.subsample([
        "~Giants", "Vsini det", "High Alpha"])
    alphamarginal = aposplit.subsample([
        "~Giants", "Vsini marginal", "High Alpha"])
    alphanondets = aposplit.subsample([
        "~Giants", "Vsini nondet", "High Alpha"])
    alphanovsini = aposplit.subsample([
        "~Giants", "No Vsini", "High Alpha"])

    jen_fast = jen.read_Jen_SGB_fast_launch()
    jen_slow = jen.read_Jen_SGB_slow_launch()
    jen.make_additional_columns(0.0, 10, jen_fast)
    jen.make_additional_columns(0.0, 10, jen_slow)
    jen_fast_teff = jen_fast[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_fast_M_K = jen_fast[jen.format_Jen_column(0.0, 10, "M_K")]
    jen_slow_teff = jen_slow[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_slow_M_K = jen_slow[jen.format_Jen_column(0.0, 10, "M_K")]

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        novsini["TEFF"], novsini["M_K"], color=bc.black, marker=".", ls="",
        alpha=0.3, axis=ax, label="")
    hr.absmag_teff_plot(
        nondets["TEFF"], nondets["M_K"], color=bc.black, marker=".", ls="",
        alpha=0.3, axis=ax, label="vsini <= 10 km/s")
    hr.absmag_teff_plot(
        marginal["TEFF"], marginal["M_K"], color=bc.black, marker=".", ls="",
        alpha=0.3, axis=ax, label="")
    # Now color-code the detections
    hr.absmag_teff_plot(
        cool_dets["TEFF"], cool_dets["M_K"], color=bc.violet, marker="o", ls="", 
        axis=ax, label="vsini > 10 km/s")
    hr.absmag_teff_plot(
        cool_lowers["TEFF"], cool_lowers["M_K"], color=bc.violet, marker="o", 
        ls="", axis=ax, label="")
    hr.absmag_teff_plot(
        hot_dets["TEFF"], hot_dets["M_K"], color=bc.orange, marker="o", ls="", 
        axis=ax, label="")
    hr.absmag_teff_plot(
        hot_lowers["TEFF"], hot_lowers["M_K"], color=bc.orange, marker="o", 
        ls="", axis=ax, label="")
    hr.absmag_teff_plot(
        sub_dets["TEFF"], sub_dets["M_K"], color=bc.algae, marker="o", ls="", 
        axis=ax, label="")
    hr.absmag_teff_plot(
        sub_lowers["TEFF"], sub_lowers["M_K"], color=bc.algae, marker="o", 
        ls="", axis=ax, label="")
    hr.absmag_teff_plot(
        lum_sub_dets["TEFF"], lum_sub_dets["M_K"], color=bc.sky_blue, 
        marker="o", ls="", axis=ax, label="")
    hr.absmag_teff_plot(
        lum_sub_lowers["TEFF"], lum_sub_lowers["M_K"], color=bc.sky_blue, 
        marker="o", ls="", axis=ax, label="")
    hr.absmag_teff_plot(
        alphanovsini["TEFF"], alphanovsini["M_K"], color=bc.light_pink, 
        marker="*", ls="", axis=ax, label="")
    hr.absmag_teff_plot(
        alphanondets["TEFF"], alphanondets["M_K"], color=bc.light_pink, 
        marker="*", ls="", axis=ax, label="")
    hr.absmag_teff_plot(
        alphamarginal["TEFF"], alphamarginal["M_K"], color=bc.light_pink, 
        marker="*", ls="", axis=ax, label="")
    hr.absmag_teff_plot(
        alphadets["TEFF"], alphadets["M_K"], color=bc.light_pink, marker="*", 
        ls="", axis=ax, ms=12, label="[a/Fe] > 0.2")
    hr.absmag_teff_plot(
        alphalowers["TEFF"], alphalowers["M_K"], color=bc.light_pink, 
        marker="*", ls="", axis=ax, ms=12, label="")

    hr.absmag_teff_plot(
        jen_fast_teff, jen_fast_M_K, color=bc.black, marker="", ls="-",
        label="Fast launch", axis=ax)
    hr.absmag_teff_plot(
        jen_slow_teff, jen_slow_M_K, color=bc.black, marker="", ls="--",
        label="Slow launch", axis=ax)

    ax.set_xlabel("Teff (K)")
    ax.set_xlim(6600, 3600)
    ax.set_ylabel("M_K")
    ax.set_ylim(6.3, -1.3)
    ax.legend()

def jen_subgiants_radius():
    '''Plot the subgiants and boundary in radius space.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    # Solar Metallicity 
    lowers = aposplit.subsample(
        ["~Giants", "Vsini lower", "Low Alpha"])
    dets = aposplit.subsample(
        ["~Giants", "Vsini det", "Low Alpha"])
    marginal = aposplit.subsample(
        ["~Giants", "Vsini marginal", "Low Alpha"])
    nondets = aposplit.subsample(
        ["~Giants", "Vsini nondet", "Low Alpha"])
    novsini = aposplit.subsample(
        ["~Giants", "No Vsini", "Low Alpha"])
    alphalowers = aposplit.subsample(
        ["~Giants", "Vsini lower", "High Alpha"])
    alphadets = aposplit.subsample(
        ["~Giants", "Vsini det", "High Alpha"])
    alphamarginal = aposplit.subsample(
        ["~Giants", "Vsini marginal", "High Alpha"])
    alphanondets = aposplit.subsample(
        ["~Giants", "Vsini nondet", "High Alpha"])
    alphanovsini = aposplit.subsample(
        ["~Giants", "No Vsini", "High Alpha"])

    jen_fast = jen.read_Jen_SGB_fast_launch()
    jen_slow = jen.read_Jen_SGB_slow_launch()
    jen_fast_teff = jen_fast[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_fast_rad = jen_fast[jen.format_Jen_column(0.0, 10, "Rad")]
    jen_slow_teff = jen_slow[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_slow_rad = jen_slow[jen.format_Jen_column(0.0, 10, "Rad")]

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    hr.radius_teff_plot(
        novsini["TEFF"], np.log10(novsini["Gaia R"]), color=bc.black, marker=".", ls="",
        alpha=0.3, axis=ax, label="")
    hr.radius_teff_plot(
        nondets["TEFF"], np.log10(nondets["Gaia R"]), color=bc.black, marker=".", ls="",
        alpha=0.3, axis=ax, label="vsini <= 10 km/s")
    hr.radius_teff_plot(
        marginal["TEFF"], np.log10(marginal["Gaia R"]), color=bc.black, marker=".", ls="",
        alpha=0.3, axis=ax, label="")
    hr.radius_teff_plot(
        dets["TEFF"], np.log10(dets["Gaia R"]), color=bc.sky_blue, marker="o", ls="", axis=ax,
        label="vsini > 10 km/s")
    hr.radius_teff_plot(
        lowers["TEFF"], np.log10(lowers["Gaia R"]), color=bc.sky_blue, marker="o", ls="",
        axis=ax, label="")
    hr.radius_teff_plot(
        alphanovsini["TEFF"], np.log10(alphanovsini["Gaia R"]), color=bc.purple, 
        marker="*", ls="", alpha=0.3, axis=ax, label="")
    hr.radius_teff_plot(
        alphanondets["TEFF"], np.log10(alphanondets["Gaia R"]), color=bc.purple, 
        marker="*", ls="", alpha=0.3, axis=ax, label="")
    hr.radius_teff_plot(
        alphamarginal["TEFF"], np.log10(alphamarginal["Gaia R"]), color=bc.orange, 
        marker="*", ls="", alpha=0.3, axis=ax, label="")
    hr.radius_teff_plot(
        alphadets["TEFF"], np.log10(alphadets["Gaia R"]), color=bc.orange, marker="*", 
        ls="", axis=ax, ms=12, label="[a/Fe] > 0.2")
    hr.radius_teff_plot(
        alphalowers["TEFF"], np.log10(alphalowers["Gaia R"]), color=bc.orange, 
        marker="*", ls="", axis=ax, ms=12, label="")

    hr.radius_teff_plot(
        jen_fast_teff, np.log10(jen_fast_rad), color=bc.black, marker="", ls="-",
        label="Fast launch", axis=ax)
    hr.radius_teff_plot(
        jen_slow_teff, np.log10(jen_slow_rad), color=bc.black, marker="", ls="--",
        label="Slow launch", axis=ax)

    ax.set_xlabel("Teff (K)")
    ax.set_xlim(6600, 3600)
    ax.set_ylabel("log(Radius)")
    ax.legend()

@write_plot("dwarf_giant_boundary")
def dwarf_giant_grid_boundary():
    '''Display the boundary between the dwarf and subgiant grid.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    dwarfs = aposplit.subsample(["~No Vsini"])
    giants = aposplit.subsample(["No Vsini"])

    teff_bin_edges = np.linspace(3500, 6600, 35//1+1, endpoint=True)
    mk_bin_edges = np.linspace(-2, 7, 32/1+1, endpoint=True)
    count_cmap = plt.get_cmap("plasma")
    count_cmap.set_bad("white")
    dwarf_hist, xedges, yedges = np.histogram2d(
        dwarfs["TEFF"], dwarfs["M_K"], 
        bins=(teff_bin_edges, mk_bin_edges))
    giant_hist, xedges, yedges = np.histogram2d(
        giants["TEFF"], giants["M_K"], 
        bins=(teff_bin_edges, mk_bin_edges))
    dwarf_frac = dwarf_hist / (dwarf_hist + giant_hist)

    # I don't want to mark bins with less than 50 points as invalid.
    unpop_bins = (dwarf_hist + giant_hist) < 0
    dwarf_frac[unpop_bins] = np.nan
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    asp = (extent[1]-extent[0])/(extent[3]-extent[2])
    f, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        dwarf_frac.T, origin="lower", extent=extent, aspect="auto", 
        cmap=count_cmap, norm=Normalize(vmin=0, vmax=1))
    cbar = f.colorbar(im, ax=ax)
    cbar.set_label("Dwarf Fraction")

    ax.set_ylim(7, -2)
    ax.set_xlim(6600, 3500)
    ax.set_ylabel(MKstr)
    ax.set_xlabel("{0} (K)".format(Teffstr))
    ax.legend()





def subgiant_rapid_rotation():
    '''Calculate the fraction of rapid rotators in the subgiant regime.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    subgiants_det = aposplit.subsample([
        "Subgiants", "Vsini det", "Low Alpha"])
    total_subgiants = aposplit.subsample([
        "Subgiants", "Low Alpha"])
    lum_subgiants_det = aposplit.subsample([
        "Luminous Subgiants", "Vsini det", "Low Alpha"])
    total_lum_subgiants = aposplit.subsample([
        "Luminous Subgiants", "Low Alpha"])

    jen_fast = jen.read_Jen_SGB_fast_launch()
    jen_slow = jen.read_Jen_SGB_slow_launch()
    jen_fast_teff = jen_fast[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_slow_teff = jen_slow[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_fast_rad = jen_fast[jen.format_Jen_column(0.0, 10, "Rad")]
    jen_slow_rad = jen_slow[jen.format_Jen_column(0.0, 10, "Rad")]

    min_rad_fast = np.ma.min(jen_fast_rad)
    max_rad_fast = np.ma.max(jen_fast_rad)
    min_rad_slow = np.ma.min(jen_slow_rad)
    max_rad_slow = np.ma.max(jen_slow_rad)
    min_rad = max(min_rad_fast, min_rad_slow)
    max_rad = min(max_rad_fast, max_rad_slow)

    subgiants_det = subgiants_det[np.logical_and(
        subgiants_det["Gaia R"] > min_rad, 
        subgiants_det["Gaia R"] < max_rad_fast)]
    total_subgiants = total_subgiants[np.logical_and(
        total_subgiants["Gaia R"] > min_rad, 
        total_subgiants["Gaia R"] < max_rad_fast)]
    lum_subgiants_det = lum_subgiants_det[np.logical_and(
        lum_subgiants_det["Gaia R"] > min_rad, 
        lum_subgiants_det["Gaia R"] < max_rad_fast)]
    total_lum_subgiants = total_lum_subgiants[np.logical_and(
        total_lum_subgiants["Gaia R"] > min_rad, 
        total_lum_subgiants["Gaia R"] < max_rad_fast)]

    jen_fast_interp = interp1d(
        jen_fast_rad[~jen_fast_rad.mask],
        jen_fast_teff[~jen_fast_teff.mask], bounds_error=False)
    jen_slow_interp = interp1d(
        jen_slow_rad[~jen_slow_rad.mask],
        jen_slow_teff[~jen_slow_teff.mask], bounds_error=False)

    fast_subgiants_det = subgiants_det[
        subgiants_det["TEFF"] > jen_slow_interp(
            subgiants_det["Gaia R"].filled())]
    fast_total_subgiants = total_subgiants[
        total_subgiants["TEFF"] > jen_slow_interp(
            total_subgiants["Gaia R"].filled())]
    fast_lum_subgiants_det_toobig = (
        lum_subgiants_det["Gaia R"] > max_rad_slow)
    fast_lum_subgiants_det = lum_subgiants_det[
        np.logical_and(
            ~fast_lum_subgiants_det_toobig,
            lum_subgiants_det["TEFF"] > jen_slow_interp(
                lum_subgiants_det["Gaia R"].filled()))]
    fast_total_lum_subgiants_toobig = (
        total_lum_subgiants["Gaia R"] > max_rad_slow)
    fast_total_lum_subgiants = total_lum_subgiants[
        np.logical_and(
            ~fast_total_lum_subgiants_toobig,
            total_lum_subgiants["TEFF"] > jen_slow_interp(
                total_lum_subgiants["Gaia R"].filled()))]
    trans_subgiants_det = subgiants_det[np.logical_and(
        subgiants_det["TEFF"] < jen_slow_interp(
            subgiants_det["Gaia R"].filled()),
        subgiants_det["TEFF"] > jen_fast_interp(
            subgiants_det["Gaia R"].filled()))]
    trans_total_subgiants = total_subgiants[np.logical_and(
        total_subgiants["TEFF"] < jen_slow_interp(
            total_subgiants["Gaia R"].filled()),
        total_subgiants["TEFF"] > jen_fast_interp(
            total_subgiants["Gaia R"].filled()))]
    trans_lum_subgiants_toobig = lum_subgiants_det["Gaia R"] < max_rad_slow
    trans_lum_subgiants_det = lum_subgiants_det[
        np.logical_or(
            np.logical_and(
                ~trans_lum_subgiants_toobig,
                np.logical_and(
                    lum_subgiants_det["TEFF"] < jen_slow_interp(
                        lum_subgiants_det["Gaia R"].filled()),
                    lum_subgiants_det["TEFF"] > jen_fast_interp(
                        lum_subgiants_det["Gaia R"].filled()))),
            np.logical_and(
                lum_subgiants_det["Gaia R"] > max_rad_slow,
                lum_subgiants_det["TEFF"] > jen_fast_interp(
                    lum_subgiants_det["Gaia R"].filled())))]
    trans_total_lum_subgiants_toobig = (
        total_lum_subgiants["Gaia R"] < max_rad_slow)
    trans_total_lum_subgiants = total_lum_subgiants[
        np.logical_or(
            np.logical_and(
                ~trans_total_lum_subgiants_toobig,
                np.logical_and(
                    total_lum_subgiants["TEFF"] < jen_slow_interp(
                        total_lum_subgiants["Gaia R"].filled()),
                    total_lum_subgiants["TEFF"] > jen_fast_interp(
                        total_lum_subgiants["Gaia R"].filled()))),
            np.logical_and(
                total_lum_subgiants["Gaia R"] > max_rad_slow,
                total_lum_subgiants["TEFF"] > jen_fast_interp(
                    total_lum_subgiants["Gaia R"].filled())))]
    slow_subgiants_det = subgiants_det[
        subgiants_det["TEFF"] < jen_fast_interp(
            subgiants_det["Gaia R"].filled())]
    slow_total_subgiants = total_subgiants[
        total_subgiants["TEFF"] < jen_fast_interp(
            total_subgiants["Gaia R"].filled())]
    slow_lum_subgiants_det = lum_subgiants_det[
        lum_subgiants_det["TEFF"] < jen_fast_interp(
            lum_subgiants_det["Gaia R"].filled())]
    slow_total_lum_subgiants = total_lum_subgiants[
        total_lum_subgiants["TEFF"] < jen_fast_interp(
            total_lum_subgiants["Gaia R"].filled())]

    fast_subgiants_frac = len(fast_subgiants_det) / len(fast_total_subgiants)
    fast_lum_subgiants_frac = (
        len(fast_lum_subgiants_det) / len(fast_total_lum_subgiants))
    trans_subgiants_frac = len(trans_subgiants_det) / len(trans_total_subgiants)
    trans_lum_subgiants_frac = (
        len(trans_lum_subgiants_det) / len(trans_total_lum_subgiants))
    slow_subgiants_frac = len(slow_subgiants_det) / len(slow_total_subgiants)
    slow_lum_subgiants_frac = (
        len(slow_lum_subgiants_det) / len(slow_total_lum_subgiants))

    fractemp = "{0:d}/{1:d} = {2:.1f}%"

    fracstrs = np.array([
        fractemp.format(len(d), len(f), p*100) for d, f, p in zip(
        [fast_subgiants_det, trans_subgiants_det, slow_subgiants_det,
         fast_lum_subgiants_det, trans_lum_subgiants_det,
         slow_lum_subgiants_det],
        [fast_total_subgiants, trans_total_subgiants, slow_total_subgiants,
         fast_total_lum_subgiants, trans_total_lum_subgiants,
         slow_total_lum_subgiants],
        [fast_subgiants_frac, trans_subgiants_frac, slow_subgiants_frac,
         fast_lum_subgiants_frac, trans_lum_subgiants_frac,
         slow_lum_subgiants_frac])])

    fracstrs = fracstrs.reshape(2, 3).transpose()

    fractable = Table(fracstrs, names=("Subgiants", "Luminous Subgiants"))
    fractable["Prediction"] = ["Detectable", "In Transition", "Nondetectable"]

    print(fractable)

@write_plot("f12")
def high_alpha_HR_diagram():
    '''Plot the high alpha stars here.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    high_alpha = aposplit.subsample(["High Mg", "~Vsini det"])
    rapidrots = aposplit.subsample(["High Mg", "Vsini det"])
    fullsamp = aposplit.subsample(["High Mg"])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        high_alpha["TEFF"], high_alpha["M_K"], color=bc.black, marker=".",
        ls="", axis=ax, label=r"[Mg/Fe] > 0.2")
    hr.absmag_teff_plot(
        rapidrots["TEFF"], rapidrots["M_K"], color=bc.pink, marker="o",
        ls="", axis=ax, label=r"$v \sin i > 10$ km s$^{-1}$")

    # Plot an old, metal-poor isochrone now.
    iso = dsep.DSEPIsochrone.isochrone_from_file(-0.5, afe=3)
    highmet_iso = dsep.DSEPIsochrone.isochrone_from_file(-0.0, afe=3)
    isotab = iso.iso_table(10)
    highmettab = highmet_iso.iso_table(10)
    tablesubset = isotab[isotab[iso.mass_col] < 1]
    highmetsubset = highmettab[highmettab[highmet_iso.mass_col] < 1.5]
    hr.absmag_teff_plot(
        10**tablesubset[iso.logteff_col],
        tablesubset[dsep.band_translation["Ks"]], color="r", marker="", ls="-",
        axis=ax, label=r"$[Fe/H] = -0.5$ isochrone", lw=2)
    hr.absmag_teff_plot(
        10**highmetsubset[highmet_iso.logteff_col],
        highmetsubset[dsep.band_translation["Ks"]], color="r", marker="",
        ls="--", axis=ax, label=r"$[Fe/H] = 0.0$ isochrone", lw=2)

    # Plot meaningful errors.
    lowlum = fullsamp[fullsamp["M_K"] > -2]
    median_teff_err = np.median(lowlum["TEFF_ERR"])
    median_k1_err = np.median(lowlum["M_K_err1"])
    median_k2_err = np.median(lowlum["M_K_err2"])

    hr.absmag_teff_plot(
        [3900], [0], yerr=[[median_k2_err], [median_k1_err]], 
        xerr=[median_teff_err], color=bc.black, marker="", ls="")

    track = dsep.DSEPEvolutionaryTrack.track_from_file(1.3, -0.5, afe=3)
    highmet_track = dsep.DSEPEvolutionaryTrack.track_from_file(1.3, 0.0, afe=3)
    track.restrict_pms()
    highmet_track.restrict_pms()
    hr.absmag_teff_plot(
        10**track.tracktable[track.logteff_col],
        track.tracktable[dsep.band_translation["Ks"]], marker="", ls="-", 
        label=r"$[Fe/H] = -0.5$ track", color=bc.algae, lw=3)
    hr.absmag_teff_plot(
        10**highmet_track.tracktable[highmet_track.logteff_col],
        highmet_track.tracktable[dsep.band_translation["Ks"]], marker="", ls="--", 
        label=r"$[Fe/H] = -0.0$ track", color=bc.algae, lw=3)

    ax.set_xlabel("{0} (K)".format(Teffstr))
    ax.set_ylabel(MKstr)
    ax.set_xlim(6600, 3500)
    ax.set_ylim(7, -2)
    ax.legend(loc="lower left", fontsize=15)

def low_alpha_HR_diagram():
    '''Plot the high alpha stars here.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    high_alpha = aposplit.subsample(["Low Alpha", "~Vsini det"])
    rapidrots = aposplit.subsample(["Low Alpha", "Vsini det"])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        high_alpha["TEFF"], high_alpha["M_K"], color=bc.black, marker=".",
        ls="", axis=ax)
    hr.absmag_teff_plot(
        rapidrots["TEFF"], rapidrots["M_K"], color="r", marker="o",
        ls="", axis=ax)

    # I want to calculate the rapid rotator fraction in a box.
    recpatch = patches.Rectangle(
        (5950, -1.95), 600, 5.2, linewidth=3, edgecolor=bc.brown,
         facecolor="none")
    ax.add_patch(recpatch)

def alpha_metallicity_comparison():
    '''Compare the metallicity distributions for high and low-alpha samples.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    high_alpha = aposplit.subsample(["High Alpha"])
    low_alpha = aposplit.subsample(["Low Alpha"])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    n, bins, patches = ax.hist(
        [high_alpha["FE_H"], low_alpha["FE_H"]], bins=10, range=(-1, 0.5), 
        normed=True, histtype="step", color=["blue", "red"], stacked=False,
        label=["High Alpha", "Low Alpha"])
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")

def high_alpha_rapid_rotation():
    '''Calculate the fraction of rapid rotators in the subgiant regime.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    subgiants_det = aposplit.subsample([
        "Subgiants", "Vsini det", "High Alpha"])
    total_subgiants = aposplit.subsample([
        "Subgiants", "High Alpha"])
    lum_subgiants_det = aposplit.subsample([
        "Luminous Subgiants", "Vsini det", "High Alpha"])
    total_lum_subgiants = aposplit.subsample([
        "Luminous Subgiants", "High Alpha"])

    jen_fast = jen.read_Jen_SGB_fast_launch()
    jen_slow = jen.read_Jen_SGB_slow_launch()
    jen_fast_teff = jen_fast[jen.format_Jen_column(-0.4, 10, "Teff")]
    jen_slow_teff = jen_slow[jen.format_Jen_column(-0.4, 10, "Teff")]
    jen_fast_rad = jen_fast[jen.format_Jen_column(-0.4, 10, "Rad")]
    jen_slow_rad = jen_slow[jen.format_Jen_column(-0.4, 10, "Rad")]

    min_rad_fast = np.ma.min(jen_fast_rad)
    max_rad_fast = np.ma.max(jen_fast_rad)
    min_rad_slow = np.ma.min(jen_slow_rad)
    max_rad_slow = np.ma.max(jen_slow_rad)
    min_rad = max(min_rad_fast, min_rad_slow)
    max_rad = min(max_rad_fast, max_rad_slow)

    subgiants_det = subgiants_det[np.logical_and(
        subgiants_det["Gaia R"] > min_rad, 
        subgiants_det["Gaia R"] < max_rad_fast)]
    total_subgiants = total_subgiants[np.logical_and(
        total_subgiants["Gaia R"] > min_rad, 
        total_subgiants["Gaia R"] < max_rad_fast)]
    lum_subgiants_det = lum_subgiants_det[np.logical_and(
        lum_subgiants_det["Gaia R"] > min_rad, 
        lum_subgiants_det["Gaia R"] < max_rad_fast)]
    total_lum_subgiants = total_lum_subgiants[np.logical_and(
        total_lum_subgiants["Gaia R"] > min_rad, 
        total_lum_subgiants["Gaia R"] < max_rad_fast)]

    jen_fast_interp = interp1d(
        jen_fast_rad[~jen_fast_rad.mask],
        jen_fast_teff[~jen_fast_teff.mask], bounds_error=False)
    jen_slow_interp = interp1d(
        jen_slow_rad[~jen_slow_rad.mask],
        jen_slow_teff[~jen_slow_teff.mask], bounds_error=False)

    fast_subgiants_det = subgiants_det[
        subgiants_det["TEFF"] > jen_slow_interp(
            subgiants_det["Gaia R"].filled())]
    fast_total_subgiants = total_subgiants[
        total_subgiants["TEFF"] > jen_slow_interp(
            total_subgiants["Gaia R"].filled())]
    fast_lum_subgiants_det_toobig = (
        lum_subgiants_det["Gaia R"] > max_rad_slow)
    fast_lum_subgiants_det = lum_subgiants_det[
        np.logical_and(
            ~fast_lum_subgiants_det_toobig,
            lum_subgiants_det["TEFF"] > jen_slow_interp(
                lum_subgiants_det["Gaia R"].filled()))]
    fast_total_lum_subgiants_toobig = (
        total_lum_subgiants["Gaia R"] > max_rad_slow)
    fast_total_lum_subgiants = total_lum_subgiants[
        np.logical_and(
            ~fast_total_lum_subgiants_toobig,
            total_lum_subgiants["TEFF"] > jen_slow_interp(
                total_lum_subgiants["Gaia R"].filled()))]
    trans_subgiants_det = subgiants_det[np.logical_and(
        subgiants_det["TEFF"] < jen_slow_interp(
            subgiants_det["Gaia R"].filled()),
        subgiants_det["TEFF"] > jen_fast_interp(
            subgiants_det["Gaia R"].filled()))]
    trans_total_subgiants = total_subgiants[np.logical_and(
        total_subgiants["TEFF"] < jen_slow_interp(
            total_subgiants["Gaia R"].filled()),
        total_subgiants["TEFF"] > jen_fast_interp(
            total_subgiants["Gaia R"].filled()))]
    trans_lum_subgiants_toobig = lum_subgiants_det["Gaia R"] < max_rad_slow
    trans_lum_subgiants_det = lum_subgiants_det[
        np.logical_or(
            np.logical_and(
                ~trans_lum_subgiants_toobig,
                np.logical_and(
                    lum_subgiants_det["TEFF"] < jen_slow_interp(
                        lum_subgiants_det["Gaia R"].filled()),
                    lum_subgiants_det["TEFF"] > jen_fast_interp(
                        lum_subgiants_det["Gaia R"].filled()))),
            np.logical_and(
                lum_subgiants_det["Gaia R"] > max_rad_slow,
                lum_subgiants_det["TEFF"] > jen_fast_interp(
                    lum_subgiants_det["Gaia R"].filled())))]
    trans_total_lum_subgiants_toobig = (
        total_lum_subgiants["Gaia R"] < max_rad_slow)
    trans_total_lum_subgiants = total_lum_subgiants[
        np.logical_or(
            np.logical_and(
                ~trans_total_lum_subgiants_toobig,
                np.logical_and(
                    total_lum_subgiants["TEFF"] < jen_slow_interp(
                        total_lum_subgiants["Gaia R"].filled()),
                    total_lum_subgiants["TEFF"] > jen_fast_interp(
                        total_lum_subgiants["Gaia R"].filled()))),
            np.logical_and(
                total_lum_subgiants["Gaia R"] > max_rad_slow,
                total_lum_subgiants["TEFF"] > jen_fast_interp(
                    total_lum_subgiants["Gaia R"].filled())))]
    slow_subgiants_det = subgiants_det[
        subgiants_det["TEFF"] < jen_fast_interp(
            subgiants_det["Gaia R"].filled())]
    slow_total_subgiants = total_subgiants[
        total_subgiants["TEFF"] < jen_fast_interp(
            total_subgiants["Gaia R"].filled())]
    slow_lum_subgiants_det = lum_subgiants_det[
        lum_subgiants_det["TEFF"] < jen_fast_interp(
            lum_subgiants_det["Gaia R"].filled())]
    slow_total_lum_subgiants = total_lum_subgiants[
        total_lum_subgiants["TEFF"] < jen_fast_interp(
            total_lum_subgiants["Gaia R"].filled())]

    fast_subgiants_frac = len(fast_subgiants_det) / len(fast_total_subgiants)
    try:
        fast_lum_subgiants_frac = (
            len(fast_lum_subgiants_det) / len(fast_total_lum_subgiants))
    except ZeroDivisionError:
        fast_lum_subgiants_frac = np.nan
    try:
        trans_subgiants_frac = len(trans_subgiants_det) / len(trans_total_subgiants)
    except ZeroDivisionError:
        trans_subgiants_frac = np.nan
    try:
        trans_lum_subgiants_frac = (
            len(trans_lum_subgiants_det) / len(trans_total_lum_subgiants))
    except ZeroDivisionError:
        trans_lum_subgiants_frac = np.nan
    slow_subgiants_frac = len(slow_subgiants_det) / len(slow_total_subgiants)
    slow_lum_subgiants_frac = (
        len(slow_lum_subgiants_det) / len(slow_total_lum_subgiants))

    fractemp = "{0:d}/{1:d} = {2:.1f}%"

    fracstrs = np.array([
        fractemp.format(len(d), len(f), p*100) for d, f, p in zip(
        [fast_subgiants_det, trans_subgiants_det, slow_subgiants_det,
         fast_lum_subgiants_det, trans_lum_subgiants_det,
         slow_lum_subgiants_det],
        [fast_total_subgiants, trans_total_subgiants, slow_total_subgiants,
         fast_total_lum_subgiants, trans_total_lum_subgiants,
         slow_total_lum_subgiants],
        [fast_subgiants_frac, trans_subgiants_frac, slow_subgiants_frac,
         fast_lum_subgiants_frac, trans_lum_subgiants_frac,
         slow_lum_subgiants_frac])])

    fracstrs = fracstrs.reshape(2, 3).transpose()

    fractable = Table(fracstrs, names=("Subgiants", "Luminous Subgiants"))
    fractable["Prediction"] = ["Detectable", "In Transition", "Nondetectable"]

    print(fractable)
    
@write_plot("f5a")
def subgiant_rapid_fraction_heatmap():
    '''Show a heatmap of the rapid rotator fraction.'''
    aposplit = cache.apogee_splitter_with_DSEP()
#   subgiants = aposplit.subsample(["Subgiants"])
#   luminous_subgiants = aposplit.subsample(["Luminous Subgiants"])
#   full = vstack([subgiants, luminous_subgiants])
    full = aposplit.subsample([])
#   subgiants_rapid = aposplit.subsample(["Subgiants", "Vsini det"])
#   luminous_subgiants_rapid = aposplit.subsample([
#       "Luminous Subgiants", "Vsini det"])
#   rapid = vstack([subgiants_rapid, luminous_subgiants_rapid])
    rapid = vstack([
        aposplit.subsample(["Vsini det"]), 
        aposplit.subsample(["Vsini lower"])])

    teff_bin_edges = np.linspace(3500, 6600, 35//1+1, endpoint=True)
    mk_bin_edges = np.linspace(-2, 7, 16/1+1, endpoint=True)
    count_cmap = plt.get_cmap("viridis_r")
    count_cmap.set_bad("white")
    apogee_hist, xedges, yedges = np.histogram2d(
        full["TEFF"], full["M_K"], 
        bins=(teff_bin_edges, mk_bin_edges))
    rapid_hist, xedges, yedges = np.histogram2d(
        rapid["TEFF"], rapid["M_K"], 
        bins=(teff_bin_edges, mk_bin_edges))
    rapid_frac = rapid_hist / apogee_hist

    # I don't want to mark bins with less than 50 points as invalid.
    unpop_bins = apogee_hist < 0
    rapid_frac[unpop_bins] = np.nan
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    asp = (extent[1]-extent[0])/(extent[3]-extent[2])
    f, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        rapid_frac.T, origin="lower", extent=extent, aspect="auto", 
        cmap=count_cmap, norm=Normalize(vmin=0, vmax=1))
    cbar = f.colorbar(im, ax=ax)
    cbar.set_label("Rapid Rotator Fraction")

    # Add on Jen's boundaries.
    jen_fast = jen.jen_fast_boundary()
    jen_slow = jen.jen_slow_boundary()
    jen_fast_teff = jen_fast[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_fast_M_K = jen_fast[jen.format_Jen_column(0.0, 10, "M_K")]
    jen_slow_teff = jen_slow[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_slow_M_K = jen_slow[jen.format_Jen_column(0.0, 10, "M_K")]

    hr.absmag_teff_plot(
        jen_fast_teff, jen_fast_M_K, color=bc.brown, marker="", ls="-", axis=ax, 
        lw=5, alpha=0.5)
    hr.absmag_teff_plot(
        jen_slow_teff, jen_slow_M_K, color=bc.brown, marker="", ls="--", 
        axis=ax, lw=5, alpha=0.5)

    add_boundaries(ax, alpha=0.5)

    samplemasses = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    periods = np.array([1, 3, 10])
    massratio = 0.9
    teff_array = np.zeros((len(samplemasses), len(periods)))
    k_array = np.zeros((len(samplemasses), len(periods)))
    for i, mass in enumerate(samplemasses):
        solartrack = mist.MISTEvolutionaryTrack.track_from_file(mass, 0.0)
        solartrack.restrict_phase([0, 2])
        totalmass = mass * (1 + massratio)

        semimajors = (periods * u.day).to(
            u.solRad, equivalencies=bincalc.keplerian_binary(
                totalmass * u.solMass)).value
        roche_ratio = 0.49 * massratio**(2/3) / (
            0.6 * massratio**(2/3) + np.log(1 + massratio**(1/3)))
        cutoff_radius = semimajors * roche_ratio
        for j, rad in enumerate(cutoff_radius):
            age = solartrack.age_at_radius(rad)
            row = solartrack.interpolate_at_age(age)
            teff_array[i,j] = 10**row[solartrack.logteff_col]
            k_array[i,j] = row[mist.band_translation["Ks"]]

    styles = ["-", "--", ":"]
    for i, (p, s) in enumerate(zip(periods, styles)):
        label = "{0:d} day merged".format(p)
        hr.absmag_teff_plot(
            teff_array[:,i], k_array[:,i], color=bc.black, ls=s, axis=ax, 
            marker="", label=label, lw=5)
    

    ax.set_ylabel(MKstr)
    ax.set_xlabel("{0} (K)".format(Teffstr))
    ax.set_title("{0} Heatmap".format(vsinistr))
    ax.legend(loc="lower left")

def subsubgiant_info():
    '''Print various info about subsubgiants.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    red_stragglers = aposplit.subsample(["Red Stragglers"])
    subsubgiants = aposplit.subsample(["Subsubgiants", "~DLSB"])
    mcq = catin.read_McQuillan_catalog()

    red_stragglers_mcq = au.join_by_id(
        red_stragglers, mcq, "kepid", "KIC", join_type="inner")
    subsubgiants_mcq = au.join_by_id(
        subsubgiants, mcq, "kepid", "KIC", join_type="inner")

    print("Number of Red Stragglers: {0:d}".format(len(red_stragglers)))
    print("Number of Subsubgiants: {0:d}".format(len(subsubgiants)))

    print("Number of rapidly rotating red stragglers: {0:d}".format(
        np.count_nonzero(red_stragglers["VSINI"] > 10)))
    print("Number of rapidly rotating subsubgiants: {0:d}".format(
        np.count_nonzero(subsubgiants["VSINI"] > 10)))

    print("Number of red stragglers with multiple RV observations: {0:d}".format(
        np.count_nonzero(red_stragglers["VSCATTER"] > 0)))
    print("Number of subsubgiants with multiple RV observations: {0:d}".format(
        np.count_nonzero(subsubgiants["VSCATTER"] > 0)))
    print("Number of red stragglers with RV variability: {0:d}".format(
        np.count_nonzero(red_stragglers["VSCATTER"] > 0.5)))
    print("Number of subsubgiants with RV variability: {0:d}".format(
        np.count_nonzero(subsubgiants["VSCATTER"] > 0.5)))

    print("Number of red stragglers with McQuillan Periods: {0:d}".format(
        len(red_stragglers_mcq)))
    print("Number of subsubgiants with McQuillan Periods: {0:d}".format(
        len(subsubgiants_mcq)))

    print("Rotation period of red straggler: {0:.2f}".format(
        red_stragglers_mcq["Prot"][0]))
    print("Rotation period of rapidly-rotating subsubgiant: {0:.2f}".format(
        subsubgiants_mcq["VSCATTER"][subsubgiants_mcq["VSINI"] > 70][0]))

    f, ax = plt.subplots(1, 1, figsize=figsize)
    rot.plot_vsini_velocity(
        subsubgiants_mcq["VSINI"], subsubgiants_mcq["Prot"], 
        subsubgiants_mcq["e_Prot"], subsubgiants_mcq["Gaia R"], 
        subsubgiants_mcq["Gaia R err"], ax=ax, 
        label="Cool Dwarfs", color=bc.black, marker="o")
    ax.plot([1, 100], [1.13, 113], color='k', ls="-.", marker="")
    ax.set_title("Subsubgiants")
    ax.legend(loc="lower right")

@write_plot("f11b")
def binned_rapid_fraction():
    '''Plot the rapid fraction as a function of teff at fixed M_K.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    full = aposplit.subsample([])
    rapid = aposplit.subsample(["Vsini det"])

    MAX_TEFF = 6700
    MIN_TEFF = 4800
    full_within_teff = np.logical_and(
        full["TEFF"] > MIN_TEFF, full["TEFF"] < MAX_TEFF)
    rapid_within_teff = np.logical_and(
        rapid["TEFF"] > MIN_TEFF, rapid["TEFF"] < MAX_TEFF)
    m_k_edges = np.array([0, 0.5, 1, 1.25, 1.5])
    m_k_avg = (m_k_edges[:-1] + m_k_edges[1:]) / 2
    colors = [bc.red, bc.blue, bc.green, bc.orange]
    assert len(colors) == len(m_k_avg)
    num_in_bin = 25

    # Read in the boundary once.
    jen_fast = jen.jen_fast_boundary()
    jen_slow = jen.jen_slow_boundary()
    jen_fast_teff = jen_fast[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_slow_teff = jen_slow[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_fast_kmag = jen_fast[jen.format_Jen_column(0.0, 10, "M_K")]
    jen_slow_kmag = jen_slow[jen.format_Jen_column(0.0, 10, "M_K")]

    jen_fast_interp = interp1d(
        jen_fast_kmag[~jen_fast_kmag.mask],
        jen_fast_teff[~jen_fast_teff.mask], bounds_error=False)
    jen_slow_interp = interp1d(
        jen_slow_kmag[~jen_slow_kmag.mask],
        jen_slow_teff[~jen_slow_teff.mask], bounds_error=False)
    jen_slow_boundaries = jen_slow_interp(m_k_avg)
    jen_fast_boundaries = jen_fast_interp(m_k_avg)
    jen_slow_edges = jen_slow_interp(m_k_edges)
    jen_fast_edges = jen_fast_interp(m_k_edges)
    f, axes = plt.subplots(
        len(m_k_edges)-1, 1, figsize=(12, 12), sharex=True, sharey=True)
    max_teff = max(full["TEFF"])
    min_teff = min(full["TEFF"])
    # Need to split the analysis into multiple iterations to accommodate
    # different bin strategies.
    for i in range(len(m_k_avg)):
        full_within_MK = np.logical_and(
            full["M_K"] > m_k_edges[i], full["M_K"] <= m_k_edges[i+1])
        rapid_within_MK = np.logical_and(
            rapid["M_K"] > m_k_edges[i], rapid["M_K"] <= m_k_edges[i+1])
        bintable = full[np.logical_and(full_within_teff, full_within_MK)]
        rapidtable = rapid[np.logical_and(rapid_within_teff, rapid_within_MK)]
        teff_edges = stats.mstats.mquantiles(bintable["TEFF"], np.linspace(
            0, 1, len(bintable) // num_in_bin))

        full_hist, edges = np.histogram(
            bintable["TEFF"], bins=teff_edges)
        rapid_hist, edges = np.histogram(
            rapidtable["TEFF"], bins=teff_edges)

        rapid_fracs = rapid_hist / full_hist
        upper_errs = au.binomial_upper(rapid_hist, full_hist) - rapid_fracs
        lower_errs = rapid_fracs - au.binomial_lower(rapid_hist, full_hist)

        max_lim = max(rapid_fracs)
        min_lim = min(rapid_fracs)

        teff_avg = (teff_edges[:-1] + teff_edges[1:]) / 2
        teff_width = -(teff_edges[:-1] - teff_edges[1:]) / 2

        label = "{0} < {1} <= {2}".format(m_k_edges[i], MKstr, m_k_edges[i+1])

        axes[i].errorbar(
            teff_avg, rapid_fracs, yerr=[lower_errs, upper_errs],
            xerr=teff_width, ls="", marker="o", color=colors[i])
        axes[i].plot(
            [jen_slow_boundaries[i]]*2, [0, 1], color=bc.brown, marker="",
            ls="--", lw=3)
        axes[i].plot(
            [jen_fast_boundaries[i]]*2, [0, 1], color=bc.brown, marker="",
            ls="-", lw=3)
        # Shade Jen's variation.
        axes[i].fill_between(
            [jen_slow_edges[i], jen_slow_edges[i+1]], [1, 1], [0, 0],
            color=bc.light_pink, hatch="/", label="Slow Launch")
        axes[i].fill_between(
            [jen_fast_edges[i], jen_fast_edges[i+1]], [1, 1], [0, 0],
            color=bc.light_pink, alpha=0.3, label="Fast Launch")
        axes[i].plot(
            [max_teff, min_teff], [max_lim]*2, color=bc.black,
             marker="", ls=":", lw=1)
        axes[i].plot(
            [max_teff, min_teff], [min_lim]*2, color=bc.black,
             marker="", ls=":", lw=1)


        axes[i].set_xlabel("")
        axes[i].set_ylabel("RR Frac")
        axes[i].set_title(label)
    axes[-1].set_xlabel("{0} (K)".format(Teffstr))
    axes[0].set_xlim(teff_edges[-1], teff_edges[0])

@write_plot("mcq_binned_rapid_rotators")
def mcq_binned_rapid_fraction():
    '''Plot the rapid fraction as a function of teff at fixed M_K.'''
    mcqsplit = cache.mcquillan_splitter_with_DSEP()
    nomcqsplit = cache.mcquillan_nondetections_splitter_with_DSEP()
    full_mcq = mcqsplit.subsample([])
    full_nomcq = nomcqsplit.subsample([])
    full = vstack([full_mcq, full_nomcq])
    rapid = mcqsplit.subsample(["McQuillan Rapid Vel"])

    m_k_edges = np.linspace(0, 2, 4+1)
    teff_edges = np.arange(4700, 6500, 200)

    full_hist, edges = np.histogramdd(
        np.array([full["M_K"], full["teff"]]).T, 
        bins=[m_k_edges, teff_edges])
    rapid_hist, edges = np.histogramdd(
        np.array([rapid["M_K"], rapid["teff"]]).T, 
        bins=[m_k_edges, teff_edges])

    rapid_fracs = rapid_hist / full_hist
    upper_errs = au.binomial_upper(rapid_hist, full_hist) - rapid_fracs
    lower_errs = rapid_fracs - au.binomial_lower(rapid_hist, full_hist)

    medians = (np.amax(rapid_fracs, axis=1) + np.amin(rapid_fracs, axis=1)) / 2

    teff_avg = (teff_edges[:-1] + teff_edges[1:]) / 2
    m_k_avg = (m_k_edges[:-1] + m_k_edges[1:]) / 2

    labels = [
        "{0} < {1} <= {2}".format(m_k_edges[i], MKstr, m_k_edges[i+1]) for i in
        range(len(m_k_avg))]


    jen_fast = jen.read_Jen_SGB_fast_launch()
    jen_slow = jen.read_Jen_SGB_slow_launch()
    jen.make_additional_columns(0.0, 10, jen_fast)
    jen.make_additional_columns(0.0, 10, jen_slow)
    jen_fast_teff = jen_fast[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_slow_teff = jen_slow[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_fast_kmag = jen_fast[jen.format_Jen_column(0.0, 10, "M_K")]
    jen_slow_kmag = jen_slow[jen.format_Jen_column(0.0, 10, "M_K")]

    jen_fast_interp = interp1d(
        jen_fast_kmag[~jen_fast_kmag.mask],
        jen_fast_teff[~jen_fast_teff.mask], bounds_error=False)
    jen_slow_interp = interp1d(
        jen_slow_kmag[~jen_slow_kmag.mask],
        jen_slow_teff[~jen_slow_teff.mask], bounds_error=False)
    jen_slow_boundaries = jen_slow_interp(m_k_avg)
    jen_fast_boundaries = jen_fast_interp(m_k_avg)

    f, axes = plt.subplots(
        full_hist.shape[0], 1, figsize=(12, 12), sharex=True, sharey=True)
    colors = [bc.red, bc.blue, bc.green, bc.orange]
    for i in range(full_hist.shape[0]):
        axes[i].errorbar(
            teff_avg, rapid_fracs[i,:], yerr=[lower_errs[i,:], upper_errs[i,:]], 
            ls="", marker="o", color=colors[i])
        axes[i].plot(
            [jen_slow_boundaries[i]]*2, [0, 1], color=bc.brown, marker="",
            ls="--", lw=3)
        axes[i].plot(
            [jen_fast_boundaries[i]]*2, [0, 1], color=bc.brown, marker="",
            ls="-", lw=3)
        axes[i].plot(
            [teff_edges[0], teff_edges[-1]], [medians[i]]*2, color=bc.black,
             marker="", ls=":", lw=1)


        axes[i].set_xlabel("")
        axes[i].set_ylabel("RR Frac")
        axes[i].set_title(labels[i])
    axes[-1].set_xlabel("{0} K".format(Teffstr))
    axes[0].set_xlim(teff_edges[-1], teff_edges[0])
    axes[0].set_ylim(0, 0.4)

@write_plot("ElBadry_heatmap")
def elbadry_rapid_fraction_heatmap():
    '''Show a heatmap of the rapid rotator fraction.'''
    aposplit = cache.apogee_splitter_with_DSEP()
#   subgiants = aposplit.subsample(["Subgiants"])
#   luminous_subgiants = aposplit.subsample(["Luminous Subgiants"])
#   full = vstack([subgiants, luminous_subgiants])
    full = vstack([
        aposplit.subsample(["El-Badry Single"]), 
        aposplit.subsample(["El-Badry SB1"])])
#   subgiants_rapid = aposplit.subsample(["Subgiants", "Vsini det"])
#   luminous_subgiants_rapid = aposplit.subsample([
#       "Luminous Subgiants", "Vsini det"])
#   rapid = vstack([subgiants_rapid, luminous_subgiants_rapid])
    rapid = vstack([
        aposplit.subsample(["Vsini det", "El-Badry Single"]),
        aposplit.subsample(["Vsini det", "El-Badry SB1"])])

    teff_bin_edges = np.linspace(3500, 6600, 35//1+1, endpoint=True)
    mk_bin_edges = np.linspace(-7, 7, 32/1+1, endpoint=True)
    count_cmap = plt.get_cmap("viridis_r")
    count_cmap.set_bad("white")
    apogee_hist, xedges, yedges = np.histogram2d(
        full["TEFF"], full["M_K"], 
        bins=(teff_bin_edges, mk_bin_edges))
    rapid_hist, xedges, yedges = np.histogram2d(
        rapid["TEFF"], rapid["M_K"], 
        bins=(teff_bin_edges, mk_bin_edges))
    rapid_frac = rapid_hist / apogee_hist

    # I don't want to mark bins with less than 50 points as invalid.
    unpop_bins = apogee_hist < 0
    rapid_frac[unpop_bins] = np.nan
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    asp = (extent[1]-extent[0])/(extent[3]-extent[2])
    f, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        rapid_frac.T, origin="lower", extent=extent, aspect="auto", 
        cmap=count_cmap, norm=Normalize(vmin=0, vmax=1))
    f.colorbar(im, ax=ax)

    # Add on Jen's boundaries.
    jen_fast = jen.read_Jen_SGB_fast_launch()
    jen_slow = jen.read_Jen_SGB_slow_launch()
    jen.make_additional_columns(0.0, 10, jen_fast)
    jen.make_additional_columns(0.0, 10, jen_slow)
    jen_fast_teff = jen_fast[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_fast_M_K = jen_fast[jen.format_Jen_column(0.0, 10, "M_K")]
    jen_slow_teff = jen_slow[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_slow_M_K = jen_slow[jen.format_Jen_column(0.0, 10, "M_K")]

    hr.absmag_teff_plot(
        jen_fast_teff, jen_fast_M_K, color=bc.pink, marker="", ls="-",
        label="Fast launch", axis=ax, lw=4)
    hr.absmag_teff_plot(
        jen_slow_teff, jen_slow_M_K, color=bc.pink, marker="", ls="--",
        label="Slow launch", axis=ax, lw=4)
    
    ax.set_ylabel(MKstr)
    ax.set_xlabel("{0} (K)".format(Teffstr))
    ax.legend(loc="upper left")

@write_plot("f5b")
def mcquillan_rapid_rotation_heatmap():
    '''Make a heatmap for the full McQuillan sample.'''
    mcqsplit = cache.mcquillan_splitter_with_DSEP()
    nomcqsplit = cache.mcquillan_nondetections_splitter_with_DSEP()
#   subgiants = aposplit.subsample(["Subgiants"])
#   luminous_subgiants = aposplit.subsample(["Luminous Subgiants"])
#   full = vstack([subgiants, luminous_subgiants])
    full_mcq = mcqsplit.subsample([])
    full_nomcq = nomcqsplit.subsample([])
#   subgiants_rapid = aposplit.subsample(["Subgiants", "Vsini det"])
#   luminous_subgiants_rapid = aposplit.subsample([
#       "Luminous Subgiants", "Vsini det"])
#   rapid = vstack([subgiants_rapid, luminous_subgiants_rapid])
    rapid_mcq = mcqsplit.subsample(["McQuillan Rapid Vel"])

    teff_bin_edges = np.linspace(3500, 6600, 35//1+1, endpoint=True)
    mk_bin_edges = np.linspace(-2, 7, 16/1+1, endpoint=True)
    count_cmap = plt.get_cmap("viridis_r")
    count_cmap.set_bad("white")
    mcq_hist, xedges, yedges = np.histogram2d(
        full_mcq["teff"], full_mcq["M_K"], 
        bins=(teff_bin_edges, mk_bin_edges))
    nomcq_hist, xedges, yedges = np.histogram2d(
        full_nomcq["teff"], full_nomcq["M_K"], 
        bins=(teff_bin_edges, mk_bin_edges))
    rapid_hist, xedges, yedges = np.histogram2d(
        rapid_mcq["teff"], rapid_mcq["M_K"], 
        bins=(teff_bin_edges, mk_bin_edges))
    rapid_frac = rapid_hist / (mcq_hist + nomcq_hist)

    # I don't want to mark bins with less than 50 points as invalid.
    unpop_bins = nomcq_hist < 10
    rapid_frac[unpop_bins] = np.nan
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    asp = (extent[1]-extent[0])/(extent[3]-extent[2])
    f, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        rapid_frac.T, origin="lower", extent=extent, aspect="auto", 
        cmap=count_cmap, norm=Normalize(vmin=0, vmax=1))
    cbar = f.colorbar(im, ax=ax)
    cbar.set_label("Rapid Rotator Fraction")

    # Add on Jen's boundaries.
    jen_fast = jen.jen_fast_boundary()
    jen_slow = jen.jen_slow_boundary()
    jen_fast_teff = jen_fast[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_fast_M_K = jen_fast[jen.format_Jen_column(0.0, 10, "M_K")]
    jen_slow_teff = jen_slow[jen.format_Jen_column(0.0, 10, "Teff")]
    jen_slow_M_K = jen_slow[jen.format_Jen_column(0.0, 10, "M_K")]

    hr.absmag_teff_plot(
        jen_fast_teff, jen_fast_M_K, color=bc.brown, marker="", ls="-",
        axis=ax, lw=4, alpha=0.5)
    hr.absmag_teff_plot(
        jen_slow_teff, jen_slow_M_K, color=bc.brown, marker="", ls="--",
        axis=ax, lw=4, alpha=0.5)

    add_boundaries(ax, alpha=0.5)
    
    ax.set_ylabel(MKstr)
    ax.set_xlabel("{0} (K)".format(Teffstr))
    ax.set_title(r"$v_{eq}$ Heatmap")
    ax.legend(loc="lower left")


def plot_evolutionary_track():
    '''Plot evolutionary tracks before mergers.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    full = aposplit.subsample([])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        full["TEFF"], full["M_K"], color="grey", alpha=0.1, ls="", marker=".",
        axis=ax)

    samplemasses = np.array([1.1, 1.2, 1.3])
    massratio = 0.9
    for mass in samplemasses:
        solartrack = mist.MISTEvolutionaryTrack.track_from_file(mass, 0.0)
        solartrack.restrict_phase([0, 2])
        totalmass = mass * (1 + massratio)

        periods = np.linspace(1, 10, 10, endpoint=True)
        semimajors = (periods * u.day).to(
            u.solRad, equivalencies=bincalc.keplerian_binary(
                totalmass * u.solMass)).value
        roche_ratio = 0.49 * massratio**(2/3) / (
            0.6 * massratio**(2/3) + np.log(1 + massratio**(1/3)))
        cutoff_radius = semimajors * roche_ratio
        zeroage_row = solartrack.interpolate_at_age(1e9)
        teffs = [10**zeroage_row[solartrack.logteff_col]]
        ks = [zeroage_row[solartrack.logteff_col]]
        for rad in cutoff_radius:
            age = solartrack.age_at_radius(rad)
            if age < 1e10:
                row = solartrack.interpolate_at_age(age)
                teffs.append(10**row[solartrack.logteff_col])
                ks.append(row[mist.band_translation["Ks"]])

        hr.absmag_teff_plot(
            teffs, ks, color=bc.black, ls="--", axis=ax, marker="o")
        hr.absmag_teff_plot(
            teffs[0:1], ks[0:1], color=bc.yellow, marker="*", ls="", axis=ax,
            ms=5)

def APOGEE_Gaia_missing():
    '''Get the targets in APOGEE that are missing in Gaia.'''
    aposplit = cache.categorized_apogee_splitter()

    notinberger = aposplit.subsample(["Not in Gaia"])
    assert(all(notinberger["source_id"].mask))

    # I checked, and all targets that were not masked were also flagged by 
    # aspcor.flag_aspcap_giants. So I can automatically discard stars with
    # calibrated LOGG values.
    giants = ~notinberger["LOGG"].mask
    dwarfs_notinberger = notinberger[~giants]

    Gaia.login_gui()

    # Place the table as a VOTable on the filesystem.
    with open("/tmp/noberger.vo", "w") as fp:
        dwarfs_notinberger[["APOGEE_ID", "ra", "dec"]].write(fp, format="votable")

    job = Gaia.launch_job_async(
        """SELECT * 
        FROM gaiadr2.gaia_source as g 
        RIGHT OUTER JOIN user_gsimonia.noberger as nob
        ON 1=CONTAINS(
        POINT('ICRS', nob.ra, nob.dec), 
        CIRCLE('ICRS', g.ra, g.dec, 1.5/3600.))""")

    r = job.get_results()
    count_not_in_gaia = np.count_nonzero(r["source_id"].mask)
    print("{0:d} not in Gaia".format(count_not_in_gaia))

    return job

########
# MIST #
########

def compare_Bolometric_Correction_metallicity():
    '''Plot the K-band bolometric correction at different metallicities.'''
    solmet = mist.MISTIsochrone.isochrone_from_file(0.0)
    lowmet = mist.MISTIsochrone.isochrone_from_file(-0.5)
    himet = mist.MISTIsochrone.isochrone_from_file(0.5)

    f, ax = plt.subplots(1, 1, figsize=figsize)
    teffs = np.linspace(4000, 7000, 100)
    sol_bcs = solmet.interpolate_isochrone_cols(
        1e9, np.log10(teffs), solmet.logteff_col, "BC K")
    low_bcs = lowmet.interpolate_isochrone_cols(
        1e9, np.log10(teffs), lowmet.logteff_col, "BC K")
    hi_bcs = himet.interpolate_isochrone_cols(
        1e9, np.log10(teffs), himet.logteff_col, "BC K")

    ax.plot(teffs, low_bcs, 'b-', label="[Fe/H] = -0.5")
    ax.plot(teffs, sol_bcs, 'k-', label="[Fe/H] = 0.0")
    ax.plot(teffs, hi_bcs, 'r-', label="[Fe/H] = 0.5")
    hr.invert_x_axis(ax)
    ax.set_xlabel("Teff")
    ax.set_ylabel("K Bolometric Correction")
    ax.legend(loc="upper left")
    
    meddiff = np.abs(np.median(hi_bcs - low_bcs)/2)
    print("Typical difference is: {0:.3f} mag".format(meddiff))

@write_plot("bc_fig")
def compare_MIST_BC_to_Casagrande():
    '''Compare the MIST BC to the empirical one by Casagrande for solar met.'''
    colorvals = np.linspace(0.93, 3.03, 1000)
    teffvals = sed.Casagrande_Teff("V-KS", colorvals, 0.0)
    casagrande_bol_table = sed.read_Casagrande_10_Table_5()

    casagrande_row = casagrande_bol_table[np.logical_and(
        casagrande_bol_table["Band"] == "KS",
        casagrande_bol_table["Color"] == "V-KS")]

    cr = casagrande_row
    mets = [-0.5, 0.0, 0.5]
    BCDict = {}
    for m in mets:
        polysum = (
            cr["b0"] + colorvals * cr["b1"] + colorvals**2 * cr["b2"] + 
            colorvals**3 * cr["b3"] + colorvals * m * cr["b4"] + m * cr["b5"] +
            m**2 * cr["b6"])*1e-5
        BCs = 4.74 - 2.5 * np.log10(polysum*4*np.pi*3.086e19**2/3.839e33)
        BCDict[m] = BCs

    MISTDict = {}
    for m in mets:
        iso = mist.MISTIsochrone.isochrone_from_file(m) 
        mist_BCs = iso.interpolate_isochrone_cols(
            1e9, np.log10(teffvals), iso.logteff_col, "BC K")
        MISTDict[m] = mist_BCs

    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(
        teffvals, MISTDict[0.5]-MISTDict[0.0], 'k:', label="MIST [Fe/H]=0.5", 
        lw=1)
    ax.plot(teffvals, MISTDict[0.0]-MISTDict[0.0], 'k-', label="", lw=2)
    ax.plot(
        teffvals, MISTDict[-0.5]-MISTDict[0.0], 'k--', label="MIST [Fe/H]=-0.5", 
        lw=1)
    ax.plot(
        teffvals, BCDict[0.5]-MISTDict[0.0], "r:", 
        label="Casagrande [Fe/H]=0.5", lw=1)
    ax.plot(
        teffvals, BCDict[0.0]-MISTDict[0.0], "r-", 
        label="Casagrande [Fe/H]=0.0")
    ax.plot(
        teffvals, BCDict[-0.5]-MISTDict[0.0], "r--", 
        label="Casagrande [Fe/H]=-0.5", lw=1)
#   ax.plot(colorvals, polysum/10**-5, 'b-')
    ax.set_xlabel("Teff")
    ax.set_ylabel("K-band BC (reference to MIST [Fe/H]=0.0)")
    ax.set_ylim(-0.06, 0.06)
    ax.set_xlim(6900, 4300)
    ax.legend(loc="upper left")
    hr.invert_x_axis(ax)

    print("The RMS between the relations is: {0:.3f}".format(
        np.sqrt(np.mean(BCDict[0.0]-MISTDict[0.0])**2)))

def radius_variation_with_metallicity():
    '''Plot the change in radius with metallicity.'''
    solmet = mist.MISTIsochrone.isochrone_from_file(0.0)
    lowmet = mist.MISTIsochrone.isochrone_from_file(-0.5)
    himet = mist.MISTIsochrone.isochrone_from_file(0.5)

    f, ax = plt.subplots(1, 1, figsize=figsize)
    teffs = np.linspace(4000, 7000, 100)
    sol_rads = solmet.interpolate_isochrone_cols(
        1e9, np.log10(teffs), solmet.logteff_col, solmet.radius_col)
    low_rads = lowmet.interpolate_isochrone_cols(
        1e9, np.log10(teffs), lowmet.logteff_col, lowmet.radius_col)
    hi_rads = himet.interpolate_isochrone_cols(
        1e9, np.log10(teffs), himet.logteff_col, himet.radius_col)

    ax.plot(teffs, hi_rads, 'r-', label="[Fe/H] = 0.5")
    ax.plot(teffs, sol_rads, 'k-', label="[Fe/H] = 0.0")
    ax.plot(teffs, low_rads, 'b-', label="[Fe/H] = -0.5")
    hr.invert_x_axis(ax)
    ax.set_xlabel("Teff")
    ax.set_ylabel("Radius")
    ax.legend(loc="upper radius")
    
    meddiff = np.abs(np.median((hi_rads - low_rads) / sol_rads)/2)*100
    print("Typical difference is: {0:.3f}%".format(meddiff))

##########################
# Spectral Contamination #
##########################

@write_plot("f10")
def plot_contamination_probability_function():
    '''Plot the probability of a given RV offset for a full binary population.'''
    # These are the Raghavan et al parameters.
    mu = 5.03
    sig = 2.28
    # Montecarlo parameters
    samplesize = 1000000
    thetas = np.random.rand(samplesize) * 2 * np.pi
    sinis = np.sqrt(1-np.random.rand(samplesize)**2)
    logPs = np.random.randn(samplesize)*sig+mu
    vmaxes = 15 * 10**(-(logPs - np.log10(365))/3)
    raddiffs_mc = 2 * np.abs(vmaxes * np.sin(thetas) * sinis)

    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.hist(raddiffs_mc, bins=500, range=(0, 20), normed=True, cumulative=False,
            color="grey", alpha=0.5, align="mid")
    vdiffs = np.linspace(0, 20, 101, endpoint=True)
    C = 10**((mu-np.log10(365))/3) * np.exp(sig**2*np.log(10)**2/18) / 60

    probs = C * erfc(
        (9 * np.log10(vdiffs/30) + 3 * mu - 3 * np.log10(365) + sig**2 *
         np.log(10)) / (3 * np.sqrt(2) * sig))

    target_inds = np.logical_and(vdiffs > 7.0, vdiffs < 14.0)
    ax.fill_between(
        vdiffs[target_inds], probs[target_inds], 
        np.zeros(np.count_nonzero(target_inds)), color="red", alpha=0.5)
    print("Integral is: {0:.1f}%".format(
        np.count_nonzero(np.logical_and(
            raddiffs_mc > 7.0, raddiffs_mc < 14.0)) / 
        len(raddiffs_mc) * 100))

    ax.plot(vdiffs, probs, color=bc.black, ls="-", lw=3)
    ax.set_xlabel("RV Offset ({0})".format(kmsstr))
    ax.set_ylabel("Probability")


####################
# Rotation Mapping #
####################

@write_plot("f3")
def plot_vsini_cut_in_period_space():
    '''Plot a cut in vsini to period space.'''
    vsini_lim = 10
    full = cache.apogee_splitter_with_DSEP()
    cool_dwarfs = full.subsample(["Cool Singles"])
    photometric_binaries = full.subsample(["Photometric Binaries"])
    cool_dwarfs = vstack([cool_dwarfs, photometric_binaries])
    asteroseismic_raw = full.subsample(["Asteroseismic"])
    apokasc = catin.read_APOKASC_catalog()[["KEPLER_INT", "RADIUS_DW"]]
    asteroseismic = au.join_by_id(
        asteroseismic_raw, apokasc, "kepid", "KEPLER_INT", join_type="left")
    hotstars = vstack([
        full.subsample(["Slow Dwarfs"]), full.subsample(["Fast Subgiants"]),
        full.subsample(["Slow Subgiants"]), full.subsample(["Red Stragglers"]),
        full.subsample(["Blue Stragglers"]), full.subsample(["Subsubgiants"])])

    all_tabs = [cool_dwarfs, asteroseismic]
    cool_dwarfs["MaxPers"] = rot.vsini_to_max_period(
        vsini_lim, cool_dwarfs["MIST R (APOGEE)"])
    hotstars["MaxPers"] = rot.vsini_to_max_period(
        vsini_lim, hotstars["Gaia R"])
    asteroseismic["MaxPers"] = rot.vsini_to_max_period(
        vsini_lim, asteroseismic["RADIUS_DW"])
    fullsamp = vstack([cool_dwarfs, hotstars, asteroseismic])

    med_inds = [np.argsort(tab["MaxPers"])[len(tab)//2] for tab in all_tabs]
    med_teffs = [tab["TEFF"][i] for tab, i in zip(all_tabs, med_inds)]
    med_Ks = [tab["M_K"][i] for tab, i in zip(all_tabs, med_inds)]
    med_pers = [tab["MaxPers"][i] for tab, i in zip(all_tabs, med_inds)]

    min_P = min(cool_dwarfs["MaxPers"])
    max_P = max(hotstars["MaxPers"])
    print(max_P)

    f, ax1 = plt.subplots(1, 1, figsize=figsize)

    radcolors = plt.get_cmap("viridis_r")
    norm = Normalize(vmin=min_P, vmax=max_P)
    
    sc = ax1.scatter(
        fullsamp["TEFF"], fullsamp["M_K"], c=fullsamp["MaxPers"], marker=".", 
        cmap=radcolors, norm=norm)
    cbar = f.colorbar(sc, ax=ax1)
    cbar.set_label(r"Period at $v_{eq} = 10$ km s$^{-1}$")
    ax1.plot(med_teffs, med_Ks, 'r*', ms=20)
    ax1.set_xlabel("{0} (K)".format(Teffstr))
    ax1.set_ylabel(MKstr)
    ax1.set_xlim(6800, 3500)
    ax1.set_ylim(7, -2)




def map_mcquillan_detections_nondetections():
    full = cache.apogee_splitter_with_DSEP()
    mcq_detections = full.subsample(["Mcq"])
    mcq_nondetections = full.subsample(["No Mcq"])
    mcq_nonanalyzed = full.subsample(["Unknown Mcq"])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        mcq_detections["TEFF"], mcq_detections["K Excess"], marker="o",
        color="r", ls="", axis=ax, label="Period Detection")
    hr.absmag_teff_plot(
        mcq_nondetections["TEFF"], mcq_nondetections["K Excess"], marker="o",
        color="b", ls="", axis=ax, label="Period Nondetection")
    hr.absmag_teff_plot(
        mcq_nonanalyzed["TEFF"], mcq_nonanalyzed["K Excess"], marker="x",
        color="grey", ls="", axis=ax, alpha=0.3, label="Not analyzed")

    ax.set_xlabel("APOGEE Teff")
    ax.set_ylabel("K Excess")
    ax.legend()

####################
# Huber Provenance #
####################

@write_plot("f14")
def huber_class_comparison():
    '''Plot Teff comparisons for Cool and Hot dwarfs and giants separately.'''
    full = cache.apogee_splitter_with_DSEP()
    print("Number of Huber stars now with APOGEE Teffs: {0:d}".format(
        full.subsample_len(["~Bad", "Huber Photometry"])))

    full_sample = full.subsample(["~Bad", "Huber Photometry"])

    hot_stars = full_sample[full_sample["TEFF"] > 5250]
    cool_dwarfs = full_sample[
        np.logical_and(full_sample["TEFF"] <= 5250, full_sample["M_K"] > 2)]
    giants = full_sample[
        np.logical_and(full_sample["TEFF"] <= 5250, full_sample["M_K"] <= 2)]

    cool_dwarf_teff_err = (cool_dwarfs["e_Teff"] + cool_dwarfs["E_Teff"])/2
    hot_star_teff_err = (hot_stars["e_Teff"] + hot_stars["E_Teff"])/2
    giant_teff_err = (giants["e_Teff"] + giants["E_Teff"])/2
    cool_dwarf_logteff_err = cool_dwarf_teff_err / cool_dwarfs["Teff"] / np.log(10)
    hot_star_logteff_err = hot_star_teff_err / hot_stars["Teff"] / np.log(10)
    giant_logteff_err = giant_teff_err / giants["Teff"] / np.log(10)
    cool_dwarf_apogee_logteff_err = (
        cool_dwarfs["TEFF_ERR"] / cool_dwarfs["TEFF"] / np.log(10))
    hot_star_apogee_logteff_err = (
        hot_stars["TEFF_ERR"] / hot_stars["TEFF"] / np.log(10))
    giant_apogee_logteff_err = (
        giants["TEFF_ERR"] / giants["TEFF"] / np.log(10))

    # Flag Outliers
    # There are no outliers in the cool dwarfs. So these will be ignored.
    cool_dwarf_chisq = (
        np.log10(cool_dwarfs["TEFF"]) - np.log10(cool_dwarfs["Teff"]))**2 / (
            cool_dwarf_apogee_logteff_err**2 + cool_dwarf_logteff_err**2)
    cool_dwarf_outliers = cool_dwarf_chisq > 16 # 4-sigma-ish
    hot_star_chisq = (
        np.log10(hot_stars["TEFF"]) - np.log10(hot_stars["Teff"]))**2 / (
            hot_star_apogee_logteff_err**2 + hot_star_logteff_err**2)
    hot_star_outliers = hot_star_chisq > 16 # 4-sigma-ish
    # I want to split the giant outliers into ones that are too cool, and ones
    # that are too warm.
    giant_chisq = (
        np.log10(giants["TEFF"]) - np.log10(giants["Teff"]))**2 / (
            giant_apogee_logteff_err**2 + giant_logteff_err**2)
    giant_outliers = giant_chisq > 16
    giant_apogee_greater_outliers = np.logical_and(
        giant_outliers, giants["TEFF"] > giants["Teff"])
    giant_huber_greater_outliers = np.logical_and(
        giant_outliers, giants["TEFF"] < giants["Teff"])

    # For non-outliers, I'd like to have a representative error bar.
    f, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(29, 9), sharex=False, sharey=False)

    median_cool_dwarf_APOGEE_Terr = np.median(
        cool_dwarfs["TEFF_ERR"][~cool_dwarf_outliers])
    median_cool_dwarf_Huber_lowTerr = np.median(
        cool_dwarfs["e_Teff"][~cool_dwarf_outliers])
    median_cool_dwarf_Huber_highTerr = np.median(
        cool_dwarfs["E_Teff"][~cool_dwarf_outliers])
    median_hot_star_APOGEE_Terr = np.median(
        hot_stars["TEFF_ERR"][~hot_star_outliers])
    median_hot_star_Huber_lowTerr = np.median(
        hot_stars["e_Teff"][~hot_star_outliers])
    median_hot_star_Huber_highTerr = np.median(
        hot_stars["E_Teff"][~hot_star_outliers])
    median_giant_APOGEE_Terr = np.median(
        giants["TEFF_ERR"][~giant_outliers])
    median_giant_Huber_lowTerr = np.median(
        giants["e_Teff"][~giant_outliers])
    median_giant_Huber_highTerr = np.median(
        giants["E_Teff"][~giant_outliers])

    ax1.errorbar(
        cool_dwarfs["TEFF"][~cool_dwarf_outliers],
        cool_dwarfs["Teff"][~cool_dwarf_outliers], marker=".", color=bc.black, 
        ls="")
    ax1.errorbar(
        [4200], [5500], yerr=[
            [median_cool_dwarf_Huber_lowTerr], 
            [median_cool_dwarf_Huber_highTerr]], 
        xerr=[median_cool_dwarf_APOGEE_Terr], marker="", color=bc.black)
    ax1.plot([3550, 5800], [3350, 5800], 'r-', lw=4)
    ax2.errorbar(
        hot_stars["TEFF"][~hot_star_outliers],
        hot_stars["Teff"][~hot_star_outliers], marker=".", color=bc.black, 
        ls="")
    ax2.errorbar(
        [5300], [6600], yerr=[
            [median_hot_star_Huber_lowTerr], [median_hot_star_Huber_highTerr]], 
        xerr=[median_hot_star_APOGEE_Terr], marker="", color=bc.black)
    ax2.plot([4250, 7050], [4250, 7050], 'r-', lw=4)
    ax3.errorbar(
        giants["TEFF"][~giant_outliers],
        giants["Teff"][~giant_outliers], marker=".", color=bc.black, 
        ls="")
    ax3.errorbar(
        [3800], [5400], yerr=[
            [median_giant_Huber_lowTerr], [median_giant_Huber_highTerr]], 
        xerr=[median_giant_APOGEE_Terr], marker="", color=bc.black)
    ax3.plot([3200, 5700], [3200, 5700], 'r-', lw=4)

    # Plot the regression line.
    cool_dwarf_fit = np.polyfit(
        cool_dwarfs["TEFF"][~cool_dwarf_outliers],
        cool_dwarfs["Teff"][~cool_dwarf_outliers], 1)
    hot_star_fit = np.polyfit(
        hot_stars["TEFF"][~hot_star_outliers],
        hot_stars["Teff"][~hot_star_outliers], 1)
    giant_fit = np.polyfit(
        giants["TEFF"][~giant_outliers],
        giants["Teff"][~giant_outliers], 1)


    print("Cool Dwarf Slope: {0:.2f}".format(cool_dwarf_fit[0]))
    print("Cool Dwarf Intercept: {0:.3f}".format(cool_dwarf_fit[1]))
    print("Hot Star Slope: {0:.2f}".format(hot_star_fit[0]))
    print("Hot Star Intercept: {0:.3f}".format(hot_star_fit[1]))
    print("Evolved Star Slope: {0:.2f}".format(giant_fit[0]))
    print("Evolved Star Intercept: {0:.3f}".format(giant_fit[1]))

    cool_dwarf_apogee_teffvals = np.array([3550, 5800])
    hot_star_apogee_teffvals = np.array([4250, 7050])
    giant_apogee_teffvals = np.array([3200, 5700])

    cool_dwarf_huber_teffvals = np.poly1d(cool_dwarf_fit)(cool_dwarf_apogee_teffvals)
    hot_star_huber_teffvals = np.poly1d(hot_star_fit)(hot_star_apogee_teffvals)
    giant_huber_teffvals = np.poly1d(giant_fit)(giant_apogee_teffvals)

    ax1.plot(cool_dwarf_apogee_teffvals, cool_dwarf_huber_teffvals, marker="",
             ls="--", color=bc.pink, lw=4)
    ax2.plot(hot_star_apogee_teffvals, hot_star_huber_teffvals, marker="",
             ls="--", color=bc.pink, lw=4)
    ax3.plot(giant_apogee_teffvals, giant_huber_teffvals, marker="",
             ls="--", color=bc.pink, lw=4)

    # Calculate the scatter.
    cool_dwarf_scatter = np.sqrt(np.mean(
        (cool_dwarfs["Teff"][~cool_dwarf_outliers] - 
         np.poly1d(cool_dwarf_fit)(
             cool_dwarfs["TEFF"][~cool_dwarf_outliers]))**2))
    hot_star_scatter = np.sqrt(np.mean(
        (hot_stars["Teff"][~hot_star_outliers] - 
         np.poly1d(hot_star_fit)(
             hot_stars["TEFF"][~hot_star_outliers]))**2))
    giant_scatter = np.sqrt(np.mean(
        (giants["Teff"][~giant_outliers] - 
         np.poly1d(giant_fit)(
             giants["TEFF"][~giant_outliers]))**2))

    print()
    print("Cool Dwarf Scatter: {0:.1f}".format(cool_dwarf_scatter))
    print("Hot Star Scatter: {0:.1f}".format(hot_star_scatter))
    print("Evolved Star Scatter: {0:.1f}".format(giant_scatter))

    # Now plot outliers.
    ax1.errorbar(
        cool_dwarfs["TEFF"][cool_dwarf_outliers],
        cool_dwarfs["Teff"][cool_dwarf_outliers], yerr=[
            cool_dwarfs["e_Teff"][cool_dwarf_outliers],
            cool_dwarfs["E_Teff"][cool_dwarf_outliers]],
        xerr=cool_dwarfs["TEFF_ERR"][cool_dwarf_outliers], marker=".", 
        color="red", ls="")
    ax2.errorbar(
        hot_stars["TEFF"][hot_star_outliers],
        hot_stars["Teff"][hot_star_outliers], yerr=[
            hot_stars["e_Teff"][hot_star_outliers],
            hot_stars["E_Teff"][hot_star_outliers]],
        xerr=hot_stars["TEFF_ERR"][hot_star_outliers], marker=".", 
        color=bc.light_pink, ls="")
    ax3.errorbar(
        giants["TEFF"][giant_apogee_greater_outliers],
        giants["Teff"][giant_apogee_greater_outliers], yerr=[
            giants["e_Teff"][giant_apogee_greater_outliers],
            giants["E_Teff"][giant_apogee_greater_outliers]],
        xerr=giants["TEFF_ERR"][giant_apogee_greater_outliers], 
        marker=".", color=bc.green, ls="")
    ax3.errorbar(
        giants["TEFF"][giant_huber_greater_outliers],
        giants["Teff"][giant_huber_greater_outliers], yerr=[
            giants["e_Teff"][giant_huber_greater_outliers],
            giants["E_Teff"][giant_huber_greater_outliers]],
        xerr=giants["TEFF_ERR"][giant_huber_greater_outliers], 
        marker=".", color=bc.blue, ls="")

    ax1.set_xlabel("APOGEE {0} (K)".format(Teffstr))
    ax2.set_xlabel("APOGEE {0} (K)".format(Teffstr))
    ax3.set_xlabel("APOGEE {0} (K)".format(Teffstr))
    ax1.set_ylabel("Huber et al. (2014) {0} (K)".format(Teffstr))
    ax2.set_ylabel("")
    ax3.set_ylabel("")
    ax1.set_xlim(3550, 5800)
    ax2.set_xlim(4250, 7050)
    ax3.set_xlim(3200, 5700)
    ax1.set_ylim(3550, 5800)
    ax2.set_ylim(4250, 7050)
    ax3.set_ylim(3200, 5700)
    ax1.set_title("Cool Dwarfs")
    ax2.set_title("Hot Stars")
    ax3.set_title("Evolved Stars")

def apogee_dwarf_targeting():
    '''Write how many Kepler objects were targeted by APOKASC.'''
    parm = catin.stelparms_triple_KIC()
    apokasc_dwarfs = np.logical_and(
        np.logical_and(
            np.logical_and(parm["Teff"] >= 5000, parm["Teff"] <= 6500),
            parm["log(g)"] > 3.5),
        np.logical_and(parm["hmag"] > 7, parm["hmag"] < 11))
    cool_dwarfs = np.logical_and(
        np.logical_and(
            parm["SDSS-Teff"] < 5500, parm["KIC logg"] > 4.0),
        np.logical_and(parm["hmag"] > 7, parm["hmag"] < 11))
    total_targeted = np.count_nonzero(
        np.logical_or(apokasc_dwarfs, cool_dwarfs))
    print("Total Targeted: {0:d}".format(total_targeted))
    
def apogee_dwarf_observed():
    '''Write how many Kepler dwarfs were observed by APOKASC.'''
    apo = cache.categorized_apogee_splitter()
    observed = np.logical_or(apo.indices["APOGEE2_APOKASC_DWARF"],
                             apo.indices["APOGEE_KEPLER_COOLDWARF"])
    print("Total observed: {0:d}".format(np.count_nonzero(observed)))

def mcquillan_overlap():
    apo = cache.categorized_apogee_splitter()
    mcq_apokasc = apo.subsample_len(["Mcq", "APOGEE2_APOKASC_DWARF"])
    mcq_cool = apo.subsample_len(["Mcq", "APOGEE_KEPLER_COOLDWARF"])
    overlap = apo.subsample_len(["Mcq", "APOGEE_KEPLER_COOLDWARF",
                                 "APOGEE2_APOKASC_DWARF"])

    combined_num = mcq_apokasc + mcq_cool - overlap
    print("Mcquillan overlap: {0:d}".format(combined_num))

def write_overlap_samples():
    '''Write a table describing the overlap of the APOGEE sample with other
    datasets.'''

    apo = cache.apogee_splitter_with_DSEP()

    samples = [
        r"\citet{McQuillan14} Detections", r"\citet{McQuillan14} Nondetections", 
        r"\citet{Serenelli17}", r"\citet{Garcia14}", r"\citet{ElBadry18b}", 
        r"\citet{Berger18a}"]

    sizes = [34030, 99000, 426, 310, 20142, 177911]

    overlap = []

    overlap.append(apo.subsample_len(["Mcq", "~Giants"]))
    overlap.append(apo.subsample_len(["No Mcq", "~Giants"]))
    overlap.append(apo.subsample_len(["Asteroseismic", "~Giants"]))
    overlap.append(apo.subsample_len(["Garcia", "~Giants"]))
    overlap.append(apo.subsample_len(["~No El-Badry Binarity", "~Giants"]))
    overlap.append(apo.subsample_len(["~No KSPC Evolution Teff", "~Giants"]))

    title = r"Validation Sample Overlap\label{tab:validation}"
    alignment = "l c c"
    footercomment = (
        r"\tablecomments{", 
        r"Overlap between validation samples and our main APOGEE sample",
        r"described in Section~\ref{sec:sample}.}")

    latexdict = {
        "col_align": alignment, "caption": title, "tabletype": "deluxetable*",
        "tablefoot": footercomment}

    overlap_table= Table([samples, sizes, overlap], names=(
        "Sample", "Total Size", "Overlap with APOGEE"))

    overlapfile = TABLE_PATH / "overlap.tex"
    overlap_table.write(
        str(overlapfile), format="ascii.aastex", 
            latexdict=latexdict, overwrite=True)


def write_targeting_count():
    '''Write a table for stars of each Kepler targeting program.'''
    apo = cache.apogee_splitter_with_DSEP()
    categories = [
        "APOGEE_KEPLER_COOLDWARF", "APOGEE2_APOKASC_DWARF",
        "APOGEE2_APOKASC_GIANT", "APOGEE2_KOI",
        "APOGEE2_KOI_CONTROL", "APOGEE_KEPLER_SEISMO", 
        "APOGEE_RV_MONITOR_KEPLER", "APOGEE_KEPLER_HOST"]
    latex_categories = [
        r"\APOGEECOOLDWARF", r"\APOKASCDWARF", r"\APOKASCGIANT", 
        r"\APOGEEKOI", r"\APOGEEKOICONTROL", r"\APOGEESEISMO",
        r"\APOGEERVMONITOR", r"\APOGEEHOST"]

    fullcounts = []
    cutcounts = []
    for cat in categories:
        fullcount = apo.subsample_len([cat])
        cutcount = apo.subsample_len([cat, "~Giants", "~Bad"])
        fullcounts.append(fullcount)
        cutcounts.append(cutcount)
    # Add the total sample.
    fullcounts.append(apo.subsample_len([]))
    cutcounts.append(apo.subsample_len(["~Giants", "~Bad"]))
    latex_categories.append("Total")

    title = r"APOGEE Targeting Flags\label{tab:targeting}"
    alignment = "l c c"
    footercomment = (
        r"\tablecomments{", 
        r"Number of stars in each APOGEE targeting program as of DR14, and "
        r"those that pass the Dwarf/subgiant cut. Some stars were targeted in ",
        r"multiple programs, so the total sample will be less than the sum ",
        r"of all targeting flags.}")

    latexdict = {
        "col_align": alignment, "caption": title, "tabletype": "deluxetable",
        "tablefoot": footercomment}

    targettab = Table(
        [latex_categories, fullcounts, cutcounts], names=(
            "Targeting Flag", "Total", "Dwarfs/Subgiants"))
    targetfile = TABLE_PATH / "targets.tex"
    targettab.write(
        str(targetfile), format="ascii.aastex", 
            latexdict=latexdict, overwrite=True)

    # I think the easiest thing is to just read it in, put an hline, and then
    # rewrite.

    with targetfile.open("r") as inputfile:
        tablelines = inputfile.readlines()
    tablelines.insert(-3-len(footercomment), r"\hline" + "\n")
    tablelines.insert(-3-len(footercomment), r"\hline" + "\n")
    with targetfile.open("w") as outputfile:
        outputfile.write("".join(tablelines))



@write_plot("f15")
def huber_classification():
    '''Compare the classification of Huber vs APOGEE and Gaia.'''
    full = cache.apogee_splitter_with_DSEP()
    print("Number of Huber stars now with APOGEE Teffs: {0:d}".format(
        full.subsample_len(["~Bad", "Huber Photometry"])))
    full_sample = full.subsample(["~Bad", "Huber Photometry"])

    hot_stars = full_sample[full_sample["TEFF"] > 5250]
    cool_dwarfs = full_sample[
        np.logical_and(full_sample["TEFF"] <= 5250, full_sample["M_K"] > 2)]
    giants = full_sample[
        np.logical_and(full_sample["TEFF"] <= 5250, full_sample["M_K"] <= 2)]
    
    cool_dwarf_r_err = (cool_dwarfs["e_R"] + cool_dwarfs["E_R"])/2
    hot_star_r_err = (hot_stars["e_R"] + hot_stars["E_R"])/2
    giant_r_err = (giants["e_R"] + giants["E_R"])/2
    cool_dwarf_teff_err = (cool_dwarfs["e_Teff"] + cool_dwarfs["E_Teff"])/2
    hot_star_teff_err = (hot_stars["e_Teff"] + hot_stars["E_Teff"])/2
    giant_teff_err = (giants["e_Teff"] + giants["E_Teff"])/2

    cool_dwarf_logr_err = cool_dwarf_r_err / cool_dwarfs["R"] / np.log(10)
    hot_star_logr_err = hot_star_r_err / hot_stars["R"] / np.log(10)
    giant_logr_err = giant_r_err / giants["R"] / np.log(10)
    cool_dwarf_logteff_err = cool_dwarf_teff_err / cool_dwarfs["Teff"] / np.log(10)
    hot_star_logteff_err = hot_star_teff_err / hot_stars["Teff"] / np.log(10)
    giant_logteff_err = giant_teff_err / giants["Teff"] / np.log(10)

    cool_dwarf_apogee_logteff_err = (
        cool_dwarfs["TEFF_ERR"] / cool_dwarfs["TEFF"] / np.log(10))
    hot_star_apogee_logteff_err = (
        hot_stars["TEFF_ERR"] / hot_stars["TEFF"] / np.log(10))
    giant_apogee_logteff_err = (
        giants["TEFF_ERR"] / giants["TEFF"] / np.log(10))

    # Flag Outliers
    cool_dwarf_chisq = (
        np.log10(cool_dwarfs["TEFF"]) - np.log10(cool_dwarfs["Teff"]))**2 / (
            cool_dwarf_apogee_logteff_err**2 + cool_dwarf_logteff_err**2)
    cool_dwarf_outliers = cool_dwarf_chisq > 16 # 4-sigma-ish
    hot_star_chisq = (
        np.log10(hot_stars["TEFF"]) - np.log10(hot_stars["Teff"]))**2 / (
            hot_star_apogee_logteff_err**2 + hot_star_logteff_err**2)
    hot_star_outliers = hot_star_chisq > 16 # 4-sigma-ish
    giant_chisq = (
        np.log10(giants["TEFF"]) - np.log10(giants["Teff"]))**2 / (
            giant_apogee_logteff_err**2 + giant_logteff_err**2)
    giant_outliers = giant_chisq > 16 # 4-sigma-ish
    giant_apogee_greater_outliers = np.logical_and(
        giant_outliers, giants["TEFF"] > giants["Teff"])
    giant_huber_greater_outliers = np.logical_and(
        giant_outliers, giants["TEFF"] < giants["Teff"])

    cool_dwarfs["Huber BC"] = samp.calc_model_over_feh_fixed_age_alpha(
        np.log10(cool_dwarfs["Teff"]), mist.MISTIsochrone.logteff_col,
        "BC K", 0.0, 1e9)
    cool_dwarfs["Huber BC err"] = samp.calc_model_err_fixed_age_feh_alpha(
        np.log10(cool_dwarfs["Teff"]), mist.MISTIsochrone.logteff_col, "BC K",
        cool_dwarf_logteff_err, 0.0, age=1e9)
    hot_stars["Huber BC"] = samp.calc_model_over_feh_fixed_age_alpha(
        np.log10(hot_stars["Teff"]), mist.MISTIsochrone.logteff_col,
        "BC K", 0.0, 1e9)
    hot_stars["Huber BC err"] = samp.calc_model_err_fixed_age_feh_alpha(
        np.log10(hot_stars["Teff"]), mist.MISTIsochrone.logteff_col, "BC K",
        hot_star_logteff_err, 0.0, age=1e9)
    giants["Huber BC"] = samp.calc_model_over_feh_fixed_age_alpha(
        np.log10(giants["Teff"]), mist.MISTIsochrone.logteff_col,
        "BC K", 0.0, 1e9)
    giants["Huber BC err"] = samp.calc_model_err_fixed_age_feh_alpha(
        np.log10(giants["Teff"]), mist.MISTIsochrone.logteff_col, "BC K",
        giant_logteff_err, 0.0, age=1e9)


    cool_dwarfs["Huber MK"] = (
        4.74 - 5 * np.log10(cool_dwarfs["R"]) - 
        10 * np.log10(cool_dwarfs["Teff"]/5778) - cool_dwarfs["Huber BC"])
    cool_dwarfs["Huber MK err"] = np.sqrt(
        (5 * cool_dwarf_logr_err)**2 + (10 * cool_dwarf_logteff_err)**2 +
        cool_dwarfs["Huber BC err"]**2)
    hot_stars["Huber MK"] = (
        4.74 - 5 * np.log10(hot_stars["R"]) - 
        10 * np.log10(hot_stars["Teff"]/5778) - hot_stars["Huber BC"])
    hot_stars["Huber MK err"] = np.sqrt(
        (5 * hot_star_logr_err)**2 + (10 * hot_star_logteff_err)**2 +
        hot_stars["Huber BC err"]**2)
    giants["Huber MK"] = (
        4.74 - 5 * np.log10(giants["R"]) - 
        10 * np.log10(giants["Teff"]/5778) - giants["Huber BC"])
    giants["Huber MK err"] = np.sqrt(
        (5 * giant_logr_err)**2 + (10 * giant_logteff_err)**2 +
        giants["Huber BC err"]**2)
    
    f, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10*2, 10), sharex=False, sharey=False)

    hr.absmag_teff_plot(
        giants["TEFF"][~giant_outliers],
        giants["M_K"][~giant_outliers], color=bc.red, marker=".",
        ls="", axis=ax1, label="Evolved Stars")
    hr.absmag_teff_plot(
        hot_stars["TEFF"][~hot_star_outliers],
        hot_stars["M_K"][~hot_star_outliers], color=bc.algae, marker=".",
        ls="", axis=ax1, label="Hot Stars")
    hr.absmag_teff_plot(
        cool_dwarfs["TEFF"][~cool_dwarf_outliers],
        cool_dwarfs["M_K"][~cool_dwarf_outliers], color=bc.violet, marker=".",
        ls="", axis=ax1, label="Cool Dwarfs")

    hr.absmag_teff_plot(
        cool_dwarfs["TEFF"][cool_dwarf_outliers],
        cool_dwarfs["M_K"][cool_dwarf_outliers], color="red", marker="o",
        ls="", axis=ax1, label="", ms=10)
    hr.absmag_teff_plot(
        hot_stars["TEFF"][hot_star_outliers],
        hot_stars["M_K"][hot_star_outliers], color=bc.light_pink, marker="o",
        ls="", axis=ax1, label="", ms=10)
    hr.absmag_teff_plot(
        giants["TEFF"][giant_apogee_greater_outliers],
        giants["M_K"][giant_apogee_greater_outliers], color=bc.green, 
        marker="o", ls="", axis=ax1, label="", ms=10)
    hr.absmag_teff_plot(
        giants["TEFF"][giant_huber_greater_outliers],
        giants["M_K"][giant_huber_greater_outliers], color=bc.blue, 
        marker="o", ls="", axis=ax1, label="", ms=10)

    # Plot uncertainties
    hr.absmag_teff_plot(
        [3500], [1.0], yerr=[
            [np.median(cool_dwarfs["M_K_err1"])],
            [np.median(cool_dwarfs["M_K_err2"])]], 
        xerr=[np.median(cool_dwarfs["TEFF_ERR"])], color=bc.violet, axis=ax1)
    hr.absmag_teff_plot(
        [3750], [0.0], yerr=[
            [np.median(hot_stars["M_K_err1"])], 
            [np.median(hot_stars["M_K_err1"])]], 
        xerr=[np.median(hot_stars["TEFF_ERR"])], color=bc.algae, axis=ax1)
    hr.absmag_teff_plot(
        [4000], [-1.0], yerr= [
            [np.median(giants["M_K_err1"])], 
            [np.median(giants["M_K_err1"])]], 
        xerr=[np.median(giants["TEFF_ERR"])], color=bc.red, axis=ax1)

    hr.absmag_teff_plot(
        cool_dwarfs["Teff"][~cool_dwarf_outliers], 
        cool_dwarfs["Huber MK"][~cool_dwarf_outliers], color=bc.violet, 
        marker=".", ls="", axis=ax2)
    hr.absmag_teff_plot(
        hot_stars["Teff"][~hot_star_outliers], 
        hot_stars["Huber MK"][~hot_star_outliers], color=bc.algae, marker=".",
        ls="", axis=ax2)
    hr.absmag_teff_plot(
        giants["Teff"][~giant_outliers], 
        giants["Huber MK"][~giant_outliers], color=bc.red, marker=".",
        ls="", axis=ax2)

    hr.absmag_teff_plot(
        cool_dwarfs["Teff"][cool_dwarf_outliers], 
        cool_dwarfs["Huber MK"][cool_dwarf_outliers], color="red", 
        marker="o", ls="", axis=ax2, ms=10)
    hr.absmag_teff_plot(
        hot_stars["Teff"][hot_star_outliers], 
        hot_stars["Huber MK"][hot_star_outliers], color=bc.light_pink, marker="o",
        ls="", axis=ax2, ms=10)
    hr.absmag_teff_plot(
        giants["Teff"][giant_apogee_greater_outliers], 
        giants["Huber MK"][giant_apogee_greater_outliers], 
        color=bc.green, marker="o", ls="", axis=ax2, ms=10)
    hr.absmag_teff_plot(
        giants["Teff"][giant_huber_greater_outliers], 
        giants["Huber MK"][giant_huber_greater_outliers], 
        color=bc.blue, marker="o", ls="", axis=ax2, ms=10)
    
    # Plot uncertainties
    hr.absmag_teff_plot(
        [3500], [1.0], yerr=[np.median(cool_dwarfs["Huber MK err"])], xerr=[
            [np.median(cool_dwarfs["e_Teff"])],
            [np.median(cool_dwarfs["E_Teff"])]], 
        color=bc.violet,
        axis=ax2)
    hr.absmag_teff_plot(
        [3750], [0.0], yerr=[np.median(hot_stars["Huber MK err"])], xerr=[
            [np.median(hot_stars["e_Teff"])], [np.median(hot_stars["E_Teff"])]], 
        color=bc.algae, axis=ax2)
    hr.absmag_teff_plot(
        [4000], [-1.0], yerr=[np.median(giants["Huber MK err"])], xerr=[
            [np.median(giants["e_Teff"])],
            [np.median(giants["E_Teff"])]],
        color=bc.red, axis=ax2)
    
    # Plot uncertainties
    hr.absmag_teff_plot(
        [3500], [1.0], yerr=[np.median(cool_dwarfs["Huber MK err"])], xerr=[
            [np.median(cool_dwarfs["e_Teff"])],
            [np.median(cool_dwarfs["E_Teff"])]], 
        color=bc.violet,
        axis=ax2)
    hr.absmag_teff_plot(
        [3750], [0.0], yerr=[np.median(hot_stars["Huber MK err"])], xerr=[
            [np.median(hot_stars["e_Teff"])], [np.median(hot_stars["E_Teff"])]], 
        color=bc.algae, axis=ax2)
    hr.absmag_teff_plot(
        [4000], [-1.0], yerr=[np.median(giants["Huber MK err"])], xerr=[
            [np.median(giants["e_Teff"])],
            [np.median(giants["E_Teff"])]],
        color=bc.red, axis=ax2)

    # Calculate number of Huber giants that are actually misclassified dwarfs.
    huber_giants_from_giants = giants[
        np.logical_and(
            giants["Teff"] < 5250, giants["Huber MK"] < 2)]
    huber_giants_from_hot_stars = hot_stars[
        np.logical_and(
            hot_stars["Teff"] < 5250, hot_stars["Huber MK"] < 2)]
    huber_giants_from_cool_dwarfs = cool_dwarfs[
        np.logical_and(
            cool_dwarfs["Teff"] < 5250, cool_dwarfs["Huber MK"] < 2)]

    print("Correctly classified giants: {0:d}".format(
        len(np.unique(huber_giants_from_giants["kepid"]))))
    print("Hot stars misclassified as giants: {0:d}".format(
        len(np.unique(huber_giants_from_hot_stars["kepid"]))))
    print("Cool dwarfs misclassified as giants: {0:d}".format(
        len(np.unique(huber_giants_from_cool_dwarfs["kepid"]))))

    # Calculate number of Huber dwarfs that are actually misclassified giants.
    huber_dwarfs_from_giants = giants[
        np.logical_and(
            giants["Teff"] < 5250, giants["Huber MK"] > 3)]
    huber_dwarfs_from_hot_stars = hot_stars[
        np.logical_and(
            hot_stars["Teff"] < 5250, hot_stars["Huber MK"] > 3)]
    huber_dwarfs_from_cool_dwarfs = cool_dwarfs[
        np.logical_and(
            cool_dwarfs["Teff"] < 5250, cool_dwarfs["Huber MK"] > 3)]

    print("Correctly classified dwarfs: {0:d}".format(
        len(np.unique(huber_dwarfs_from_giants["kepid"]))))
    print("Hot stars misclassified as dwarfs: {0:d}".format(
        len(np.unique(huber_dwarfs_from_hot_stars["kepid"]))))
    print("Giants  misclassified as dwarfs: {0:d}".format(
        len(np.unique(huber_dwarfs_from_cool_dwarfs["kepid"]))))


    ax1.set_xlim(6850, 3200)
    ax1.set_ylim(7.6, -7) 
    ax2.set_xlim(6850, 3200)
    ax2.set_ylim(7.6, -7) 
    ax1.set_xlabel("APOGEE {0} (K)".format(Teffstr))
    ax1.set_ylabel("Gaia " + MKstr)
    ax2.set_xlabel("Huber et al. (2014) {0} (K)".format(Teffstr))
    ax2.set_ylabel("Huber " + MKstr)
    ax1.legend(loc="upper left")

def huber_provenance_comparison():
    '''Plot the APOGEE Teff to Huber 2MASS Teff.'''
    full = cache.categorized_apogee_splitter()
    print("Number of Huber stars now with APOGEE Teffs: {0:d}".format(
        full.subsample_len(["~Bad", "Huber Photometry"])))
    nokic = full.subsample(["~Bad", "Huber Photometry"])

    # Try to highlight outliers.
    apogee_logteff_err = nokic["TEFF_ERR"] / nokic["TEFF"] / np.log(10)
    avg_huber_err = (nokic["e_Teff"] + nokic["E_Teff"]) / 2
    huber_logteff_err = avg_huber_err / nokic["Teff"] / np.log(10)

    chisq = (np.log10(nokic["TEFF"]) - np.log10(nokic["Teff"]))**2 / (
        apogee_logteff_err**2 + huber_logteff_err**2)
    outliers = chisq > 16 # 4-sigma-ish
    outliers_hot = np.logical_and(outliers, nokic["TEFF"] > 5500)
    outliers_cool_lowhuber = au.multi_logical_and(
        outliers, nokic["TEFF"] < 5000, nokic["Teff"] < nokic["TEFF"])
    outliers_cool_highhuber = au.multi_logical_and(
        outliers, nokic["TEFF"] < 5000, nokic["Teff"] > nokic["TEFF"])

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12*3, 12))
    ax1.errorbar(
        nokic["TEFF"][~outliers], nokic["Teff"][~outliers], 
        yerr=[nokic["e_Teff"][~outliers], nokic["E_Teff"][~outliers]],
        xerr=nokic["TEFF_ERR"][~outliers], marker=".", color="k", ls="")
    ax1.errorbar(
        nokic["TEFF"][outliers_hot], nokic["Teff"][outliers_hot], 
        yerr=[nokic["e_Teff"][outliers_hot], nokic["E_Teff"][outliers_hot]],
        xerr=nokic["TEFF_ERR"][outliers_hot], marker=".", color="m", ls="")
    ax1.errorbar(
        nokic["TEFF"][outliers_cool_lowhuber], nokic["Teff"][outliers_cool_lowhuber], 
        yerr=[nokic["e_Teff"][outliers_cool_lowhuber], nokic["E_Teff"][outliers_cool_lowhuber]],
        xerr=nokic["TEFF_ERR"][outliers_cool_lowhuber], marker=".", color="b", ls="")
    ax1.errorbar(
        nokic["TEFF"][outliers_cool_highhuber], nokic["Teff"][outliers_cool_highhuber], 
        yerr=[nokic["e_Teff"][outliers_cool_highhuber], nokic["E_Teff"][outliers_cool_highhuber]],
        xerr=nokic["TEFF_ERR"][outliers_cool_highhuber], marker=".",
        color=bc.green, ls="")
    ax1.plot([3000, 7500], [3000, 7500], 'c-')
    ax1.set_xlim(3000, 7500)
    ax1.set_ylim(3000, 7500)
    ax1.set_xlabel("APOGEE Teff (K)")
    ax1.set_ylabel("Huber Teff (K)")

    hr.absmag_teff_plot(
        nokic["Teff"][~outliers], nokic["log(g)"][~outliers], marker="o", 
        color="k", ls="", axis=ax2)
    hr.absmag_teff_plot(
        nokic["Teff"][outliers_hot], nokic["log(g)"][outliers_hot], marker="o", 
        color="m", ls="", axis=ax2)
    hr.absmag_teff_plot(
        nokic["Teff"][outliers_cool_lowhuber], nokic["log(g)"][outliers_cool_lowhuber], marker="o", 
        color="b", ls="", axis=ax2)
    hr.absmag_teff_plot(
        nokic["Teff"][outliers_cool_highhuber], nokic["log(g)"][outliers_cool_highhuber], marker="o", 
        color=bc.green, ls="", axis=ax2)
    ax2.set_xlim(7500, 3000)
    ax2.set_xlabel("Huber Teff (K)")
    ax2.set_ylabel("log(g)")

    hr.absmag_teff_plot(
        nokic["TEFF"][~outliers], nokic["M_K"][~outliers], marker="o", 
        color="k", ls="", axis=ax3)
    hr.absmag_teff_plot(
        nokic["TEFF"][outliers_hot], nokic["M_K"][outliers_hot], marker="o", 
        color="m", ls="", axis=ax3)
    hr.absmag_teff_plot(
        nokic["TEFF"][outliers_cool_lowhuber], nokic["M_K"][outliers_cool_lowhuber], marker="o", 
        color="b", ls="", axis=ax3)
    hr.absmag_teff_plot(
        nokic["TEFF"][outliers_cool_highhuber], nokic["M_K"][outliers_cool_highhuber], marker="o", 
        color=bc.green, ls="", axis=ax3)
    ax3.set_xlim(7500, 3000)
    ax3.set_xlabel("APOGEE TEFF (K)")
    ax3.set_ylabel("M_K")

    print("Hot outliers: {0:d}".format(np.count_nonzero(outliers_hot)))
    print("Low Huber outliers: {0:d}".format(
        np.count_nonzero(outliers_cool_lowhuber)))
    print("High Huber outliers: {0:d}".format(
        np.count_nonzero(outliers_cool_highhuber)))

    # Calculate the lienar least-squares fit.
    maskedvals = np.logical_or(nokic["TEFF"].mask, nokic["Teff"].mask)
    fittedvals = np.logical_not(np.logical_or(maskedvals, outliers))
    p = np.polyfit(nokic["TEFF"][fittedvals], nokic["Teff"][fittedvals], 1)
    print("y = {0:.3f} x + {1:.2f}".format(p[0], p[1]))
    modelpoly = np.poly1d(p)
    modeledTeffs = modelpoly(nokic["TEFF"][fittedvals])
    print("RMS Scatter: {0:.3f}".format(
        np.sqrt(np.mean((np.log10(nokic["Teff"][fittedvals]) - np.log10(modeledTeffs))**2))))

#######################
# Evolution of binary #
#######################

def HR_Binary_Luminosity_Evolution():
    '''Show the evolution of the combined light of a binary.'''
    apo = cache.apogee_splitter_with_DSEP()
    fullsamp = apo.subsample([])
    ages = np.linspace(1, 6, 21)*1e9
    primary_mass = mist.MISTEvolutionaryTrack.nearest_mass(1.2)
    secondary_mass = primary_mass - 0.1
    massratio = secondary_mass / primary_mass

    track1 = mist.MISTEvolutionaryTrack.track_from_file(primary_mass, 0.0)
    track2 = mist.MISTEvolutionaryTrack.track_from_file(secondary_mass, 0.0)
    teff1 = np.zeros(len(ages))
    teff2 = np.zeros(len(ages))
    k1 = np.zeros(len(ages))
    k2 = np.zeros(len(ages))
    for i, a in enumerate(ages):
        track1_age = track1.interpolate_at_age(a)
        track2_age = track2.interpolate_at_age(a)
        teff1[i] = (10**track1_age[track1.logteff_col])
        teff2[i] = (10**track2_age[track2.logteff_col])
        k1[i] = (track1_age[mist.band_translation["Ks"]])
        k2[i] = (track2_age[mist.band_translation["Ks"]])
    combined_teff = teff1 - 50
    combined_k = sed.sum_binary_mag(k1, k2)

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        teff1, k1, color=bc.blue, ls="-", marker=".", axis=ax, label="Primary")
    hr.absmag_teff_plot(
        teff2, k2, color=bc.red, ls="-", marker=".", axis=ax, label="Secondary")
    hr.absmag_teff_plot(
        combined_teff, combined_k, color=bc.black, ls="-", marker=".", lw=3,
        axis=ax, label="Combined")
    hr.absmag_teff_plot(
        fullsamp["TEFF"], fullsamp["M_K"], color="grey", alpha=0.4, marker=".",
        ls="", axis=ax)

    ax.set_xlabel("Teff")
    ax.set_ylabel("M_K")
    ax.set_title("Massratio: {0:.2f}".format(massratio))
    ax.legend()

def luminosity_contrast_evolution():
    '''Show the evolution of luminosity contrast with primary luminosity.'''
    baseage = 1e9
    age = 4.5e9
    met = 0.0
    massratio = 0.9
    iso = mist.MISTIsochrone.isochrone_from_file(met)

    Klums = np.linspace(0.4, 3, 20)
    primary_masses = iso.interpolate_isochrone_cols(
        age, Klums, mist.band_translation["Ks"], iso.mass_col, increase=False)
    secondary_masses = massratio * primary_masses
    secondary_Klums = iso.interpolate_isochrone_cols(
        age, secondary_masses, iso.mass_col, mist.band_translation["Ks"])

    log_primary_teffs = iso.interpolate_isochrone_cols(
        age, Klums, mist.band_translation["Ks"], iso.logteff_col,
        increase=False)
    log_combined_teffs = log_primary_teffs

    base_Klum = iso.interpolate_isochrone_cols(
        baseage, log_primary_teffs, iso.logteff_col, 
        mist.band_translation["Ks"])

    combined_base_Klum = iso.interpolate_isochrone_cols(
        baseage, log_combined_teffs, iso.logteff_col,
        mist.band_translation["Ks"])

    kdiff = -2.5 * np.log10(1 + 10**(-0.4*(secondary_Klums - Klums))) - (
        combined_base_Klum - base_Klum)

    plt.plot(Klums, kdiff, 'k-')

def luminosity_contrast_massratios():
    '''Show the evolution of luminosity contrast with primary luminosity.'''
    baseage = 1e9
    age = 4.5e9
    met = 0.0
    massratio = 0.9
    klums = [3, 2, 1]
    iso = mist.MISTIsochrone.isochrone_from_file(met)

    for Klum in klums:
        massratios = np.linspace(0.5, 1, 10)
        primary_mass= iso.interpolate_isochrone_cols(
            age, [Klum], mist.band_translation["Ks"], iso.mass_col, 
            increase=False)
        secondary_masses = massratios * primary_mass
        secondary_Klums = iso.interpolate_isochrone_cols(
            age, secondary_masses, iso.mass_col, mist.band_translation["Ks"])

        log_primary_teff = iso.interpolate_isochrone_cols(
            age, [Klum], mist.band_translation["Ks"], iso.logteff_col,
            increase=False)
        log_combined_teffs = log_primary_teff * np.ones(len(secondary_Klums))

        base_Klum = iso.interpolate_isochrone_cols(
            baseage, log_primary_teff, iso.logteff_col, 
            mist.band_translation["Ks"])

        combined_base_Klum = iso.interpolate_isochrone_cols(
            baseage, log_combined_teffs, iso.logteff_col,
            mist.band_translation["Ks"])

        kdiff = -2.5 * np.log10(1 + 10**(-0.4*(secondary_Klums - Klum))) - (
            combined_base_Klum - base_Klum)

        plt.plot(massratios, kdiff, 'k-')

    hr.invert_y_axis()

def luminosity_contrast_massratio_fixedteff():
    '''Find the difference between combined luminosity at fixed Teff.'''
    TEFF = 5500
    track = mist.MISTIsoTemp(TEFF)
    targetK = np.array([1.8])
    kmass_interp = interp1d(
        track.tracktab[mist.band_translation["Ks"]],
        track.tracktab[track.mass_col], kind="linear")
    kage_interp = interp1d(
        track.tracktab[mist.band_translation["Ks"]],
        track.tracktab[track.age_col], kind="linear")
    targetmass = kmass_interp(targetK)
    targetage = kage_interp(targetK)
    f, ax = plt.subplots(1, 1, figsize=figsize)
    for mass, K, t in zip(targetmass, targetK, targetage):
        lowmasses = mist.MISTEvolutionaryTrack.masses[
            np.logical_and(
                mist.MISTEvolutionaryTrack.masses < mass,
                # I put this because lower-mass stars are still on the PMS.
                mist.MISTEvolutionaryTrack.masses > 0.4)]
        secondary_lums = []
        for m in lowmasses:
            second_track = mist.MISTEvolutionaryTrack.track_from_file(m, 0.0)
            second_lum = second_track.interpolate_at_age(t)[
                mist.band_translation["Ks"]][0]
            secondary_lums.append(second_lum)
        massratios = lowmasses / mass
        massratios = np.insert(massratios, len(massratios), 1)
        secondary_lums.append(K)
        secondary_lums = np.array(secondary_lums)
        deltaKs = -2.5 * np.log10(1+10**(-0.4*(secondary_lums - K)))

        ax.plot(massratios, deltaKs, marker="o", ls="-", 
                label="MK={0:.1f}".format(K))
    hr.invert_y_axis(ax)

    iso = mist.MISTIsochrone.isochrone_from_file(0.0)
    targmass = iso.interpolate_isochrone_cols(
        1.2e8, np.log10([TEFF]), iso.logteff_col, iso.mass_col)
    targK = iso.interpolate_isochrone_cols(
        1.2e8, np.log10([TEFF]), iso.logteff_col, mist.band_translation["Ks"])

    lowmasses = mist.MISTEvolutionaryTrack.masses[
        np.logical_and(
            mist.MISTEvolutionaryTrack.masses < targmass,
            # I put this because lower-mass stars are still on the PMS.
            mist.MISTEvolutionaryTrack.masses > 0.4)]
    secondary_lums = iso.interpolate_isochrone_cols(
        1.2e8, lowmasses, iso.mass_col, mist.band_translation["Ks"])
    massratios = lowmasses / targmass
    massratios = np.insert(massratios, len(massratios), 1)
    secondary_lums = np.insert(secondary_lums, len(secondary_lums), targK)
    deltaKs = -2.5 * np.log10(1+10**(-0.4*(secondary_lums - targK)))
    
    ax.plot(massratios, deltaKs, marker="o", ls="-", 
            label="Base".format(K))
    ax.legend(loc="upper left")
    return


    Klums = [3, 2, 1]
    # Now I want to find evolutionary tracks that intersect at TEFF and Klum.
    searchmasses = np.linspace(2, 1, 21, endpoint=True)
    searchks = np.zeros(len(searchmasses))
    for i, m in enumerate(searchmasses):
        t = mist.MISTEvolutionaryTrack.track_from_file(m, 0.0)
        t.restrict_phase([0])
        age_at_teff = t.age_at_col(t.logteff_col, np.log10(TEFF))
        params = interpolate_at_age(age_at_teff)
        searchks[i] = params[mist.band_translation["Ks"]][0]


def apogee_temperature_calibration_classification():
    '''Plot the giant and dwarf temperature corrections.'''
    full = cache.apogee_splitter_with_DSEP()
    tab = full.subsample(["~Bad"])

    giants = tab["FPARAM"][:,1] < np.minimum(
        2 + 2 / 1300 * (tab["FPARAM"][:,0] - 3500), 4.0)
    dwarfs = tab["FPARAM"][:,1] >= np.minimum(
        2 + 2 / 1300 * (tab["FPARAM"][:,0] - 3500), 4.0)

    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(
        tab["FPARAM"][giants,3], 
        tab["FPARAM"][giants,0] - tab["TEFF"][giants], 'r.')
    ax.plot(
        tab["FPARAM"][dwarfs,3], 
        tab["FPARAM"][dwarfs,0] - tab["TEFF"][dwarfs], 'b.')
    ax.set_xlabel("[M/H]")
    ax.set_ylabel("ASPCAP - Calibrated Teff")

def split_apogee_temperature_calibrations():
    '''Plot the giants and dwarfs according to temperature calibration.'''
    full = cache.apogee_splitter_with_DSEP()
    tab = full.subsample(["~Bad"])

    giant_calibration = (np.abs(
        tab["FPARAM"][:,0] - tab["TEFF"] - (
            -51.5903 + 61.4774 * tab["FPARAM"][:,3] + 7.17561 * 
            tab["FPARAM"][:,3]**2)) < 0.5)
    dwarf_calibration = (np.abs(
        tab["FPARAM"][:,0] - tab["TEFF"] - (
            -36.3822 + 13.1614 * tab["FPARAM"][:,3] + -26.0953 * 
            tab["FPARAM"][:,3]**2)) < 0.5)
    giants = tab["FPARAM"][:,1] < np.minimum(
        2 + 2 / 1300 * (tab["FPARAM"][:,0] - 3500), 4.0)
    dwarfs = tab["FPARAM"][:,1] >= np.minimum(
        2 + 2 / 1300 * (tab["FPARAM"][:,0] - 3500), 4.0)
    double_giant = np.logical_and(giant_calibration, giants)
    double_dwarf = np.logical_and(dwarf_calibration, dwarfs)
    giant_calibration_dwarfs = np.logical_and(giant_calibration, dwarfs)
    dwarf_calibration_giants = np.logical_and(dwarf_calibration, giants)
    other = np.logical_not(np.logical_or(giant_calibration, dwarf_calibration))

    test_teffs = np.linspace(3532, 7500, 200)
    logg_boundary = np.minimum(2 / 1300 * (test_teffs - 3500) + 2, 4.0)

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.logg_teff_plot(
        tab["FPARAM"][double_giant,0], tab["FPARAM"][double_giant,1], 'r.')
    hr.logg_teff_plot(
        tab["FPARAM"][double_dwarf,0], tab["FPARAM"][double_dwarf,1], 'b.')
    hr.logg_teff_plot(
        tab["FPARAM"][dwarf_calibration_giants,0], 
        tab["FPARAM"][dwarf_calibration_giants,1], 'b.')
    hr.logg_teff_plot(
        tab["FPARAM"][giant_calibration_dwarfs,0], 
        tab["FPARAM"][giant_calibration_dwarfs,1], 'r.')
    hr.logg_teff_plot(
        test_teffs, logg_boundary, 'k-', lw=3)
    ax.set_xlabel("TEFF")
    ax.set_ylabel("logg")

def apogee_metallicity_calibration_classification():
    '''Plot the giant and dwarf metallicity corrections.'''
    full = cache.apogee_splitter_with_DSEP()
    tab = full.subsample(["~Bad"])

    giants = tab["FPARAM"][:,1] < np.minimum(
        2 + 2 / 1300 * (tab["FPARAM"][:,0] - 3500), 4.0)
    dwarfs = tab["FPARAM"][:,1] >= np.minimum(
        2 + 2 / 1300 * (tab["FPARAM"][:,0] - 3500), 4.0)

    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(
        tab["FPARAM"][giants,3], 
        tab["FPARAM"][giants,3] - tab["M_H"][giants], 'r.')
    ax.plot(
        tab["FPARAM"][dwarfs,3], 
        tab["FPARAM"][dwarfs,3] - tab["M_H"][dwarfs], 'b.')
    ax.set_xlabel("[M/H]")
    ax.set_ylabel("ASPCAP - Calibrated [M/H]")

def APOGEE_LOGG_Sample_Split():
    '''Plot the Dwarf/Giant separation according to logg.'''
    full= cache.apogee_splitter_with_DSEP()
    full.split_logg(
        "LOGG_FIT", 3.5, ("Logg Giants", "Logg Dwarfs"), 
        logg_crit="Spec Logg")
    giants = full.subsample(["Logg Giants"])
    dwarfs = full.subsample(["Logg Dwarfs"])
    
    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        giants["TEFF"], giants["M_K"], marker=".", color=bc.black, ls="",
        label="Giants")
    hr.absmag_teff_plot(
        dwarfs["TEFF"], dwarfs["M_K"], marker=".", color=bc.violet, ls="",
        label="Dwarfs")
    ax.plot([5250, 5250, 3500], [-2, -0.45, -0.45], color=bc.sky_blue,
            marker="", ls="--", lw=4, label="")

    ax.set_ylim(7, -2)
    ax.set_xlim(6650, 3500)
    ax.set_xlabel("{0} (K)".format(Teffstr))
    ax.set_ylabel(MKstr)
    ax.legend(loc="lower left")

def APOGEE_Full_Extinction():
    '''Calculate the extinction of the full APOGEE sample.'''
    full = cache.apogee_splitter_with_DSEP()
    allstars = full.subsample([])
    percentiles = np.array([50-67/2, 50, 50+67/2])
    ext_per = np.percentile(allstars["AV"], percentiles)
    print("Lowest percentile of extinction: {0:.2f}".format(ext_per[0]))
    print("Median percentile of extinction: {0:.2f}".format(ext_per[1]))
    print("Highest percentile of extinction: {0:.2f}".format(ext_per[2]))

def APOGEE_Nongiant_Extinction():
    '''Calculate the median extinction of APOGEE Dwarfs.'''
    full = cache.apogee_splitter_with_DSEP()
    nongiants = full.subsample(["~Giants"])
    percentiles = np.array([50-67/2, 50, 50+67/2])
    ext_per = np.percentile(nongiants["AV"], percentiles)
    k_ext = extinction.AV_to_Aband(ext_per, "Ks", system="green18")
    print("Lowest percentile of extinction: {0:.3f}".format(k_ext[0]))
    print("Median percentile of extinction: {0:.3f}".format(k_ext[1]))
    print("Highest percentile of extinction: {0:.3f}".format(k_ext[2]))
    k_err = np.median(nongiants["K_ERR"])
    print("Ratio of extinction to photometric error: "
        "{0:.3f}/{1:.3f} = {2:.3f}".format(k_ext[1], k_err, k_ext[1]/k_err))

def compare_dwarf_extinctions_APOGEE():
    full = cache.categorized_apogee_splitter()
    full.split_logg(
        "LOGG", 0, ("Spec Dwarfs", "Spec Giants", "Spec Masked"), 
        logg_crit="APOGEE Mask", null_value=np.ma.masked)
    dwarfs = full.subsample(["Spec Masked", "In Gaia"])
    berger_av_err = 0.1 * dwarfs["AV"]
    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.errorbar(dwarfs["av"], dwarfs["AV"], yerr=berger_av_err, xerr=[
        -dwarfs["av_err2"], dwarfs["av_err1"]], color="k", marker=".",
                linestyle="")
    ax.plot([0, 1.75], [0, 1.75], 'k-')
    ax.set_xlabel("Huber AV")
    ax.set_ylabel("Berger AV")

def compare_dwarf_extinction_distance_correlation():
    '''Compare A_Huber - A_Berger to 5 log(d/R).
    
    This will determine if the change in extinction is simply due to improved
    distance/radii determinations.'''
    full = cache.categorized_apogee_splitter()
    full.split_logg(
        "LOGG", 0, ("Spec Dwarfs", "Spec Giants", "Spec Masked"), 
        logg_crit="APOGEE Mask", null_value=np.ma.masked)
    dwarfs = full.subsample(["Spec Masked", "In Gaia"])
    huber_av_err = (dwarfs["av_err1"] - dwarfs["av_err2"])/2
    berger_av_err = 0.1 * dwarfs["AV"]
    huber_dist_err = (dwarfs["dist_err1"] - dwarfs["dist_err2"])/2
    huber_radius_err = (dwarfs["radius_err1"] - dwarfs["radius_err2"])/2
    berger_dist_err = (dwarfs["D_down"] + dwarfs["D_up"])/2
    berger_radius_err = (dwarfs["rad_down"] + dwarfs["rad_up"])/2
    ext_diff =  dwarfs["AV"] - dwarfs["av"]
    dist_diff = 5 * np.log10(dwarfs["dist"] / dwarfs["D"])
    dr_diff = 5 * (
        np.log10(dwarfs["dist"] / dwarfs["radius"]) - 
        np.log10(dwarfs["D"] / dwarfs["rad"]))
    ext_diff_err = np.sqrt(berger_av_err**2 + huber_av_err**2)
    dist_diff_err = 5 / np.log(10) * np.sqrt(
        (huber_dist_err / dwarfs["dist"])**2 +
        (berger_dist_err / dwarfs["D"])**2)
    dr_diff_err = 5 / np.log(10) * np.sqrt(
        (huber_dist_err / dwarfs["dist"])**2 + 
        (huber_radius_err / dwarfs["radius"])**2 +
        (berger_dist_err / dwarfs["D"])**2 +
        (berger_radius_err / dwarfs["rad"])**2)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.errorbar(
        ext_diff, dist_diff, color="k", marker=".", linestyle="")
    ax1.errorbar(
        [0.3], [1.5], yerr=[np.median(dist_diff_err)], 
        xerr=[np.median(ext_diff_err)], color='r', marker='.', linestyle="")
    ax2.errorbar(
        ext_diff, dr_diff, color="k", marker=".", linestyle="")
    ax2.errorbar(
        [0.3], [0.075], yerr=[np.median(dr_diff_err)],
        xerr=[np.median(ext_diff_err)], color="r", marker=".", linestyle="")
    ax1.set_xlim(-0.4, 0.4)
    ax1.set_ylim(-2, 2)
    ax2.set_xlim(-0.4, 0.4)
    ax2.set_ylim(-0.1, 0.1)
    ax1.set_xlabel("(Huber - Berger) AV")
    ax1.set_ylabel("(Huber - Berger) 5 log (d/10)")
    ax2.set_xlabel("(Huber - Berger) AV")
    ax2.set_ylabel("(Huber - Berger) 5 (log (d/10) - log (R/Rsun))")
    
def DLSB_rates():
    '''Calculate the SB2 rates in different parts of the HR diagram.'''
    apo = cache.apogee_splitter_with_DSEP()

    categories = [
        "Cool Dwarfs", "Hot Dwarfs", "Subgiants", "Luminous Subgiants", 
        "RGB Base", "Giants"]

    for c in categories:
        dlsbs = apo.subsample_len([c, "DLSB"])
        full_len = apo.subsample_len([c])

        print(c)
        print(format_fraction(dlsbs, full_len))

def DLSB_Gaia():
    '''Check that SB2s in Gaia is more well-behaved than in APOGEE.'''
    apo = cache.apogee_splitter_with_DSEP()

    sb2s = apo.subsample(["DLSB"])
    nonsb2s = apo.subsample(["~DLSB"])
    # There are some masked values for BP-RP. I want to ignore them for this
    # purpose.
    nonsb2s = nonsb2s[~nonsb2s["phot_bp_mean_mag"].mask]

    iso = mist.MISTIsochrone.isochrone_from_file(0.0, colors=[])
    iso.make_color_col(
        mist.band_translation["BP"], mist.band_translation["RP"], "BP-RP")
    modelteffs = np.linspace(4000, 6500, 200)
    modelKs = iso.interpolate_isochrone_cols(
        1e9, np.log10(modelteffs), iso.logteff_col, mist.band_translation["Ks"])
    sb2modelKs = iso.interpolate_isochrone_cols(
        1e9, np.log10(sb2s["TEFF"]), iso.logteff_col,
        mist.band_translation["Ks"])
    nonsb2modelKs = iso.interpolate_isochrone_cols(
        1e9, np.log10(nonsb2s["TEFF"]), iso.logteff_col,
        mist.band_translation["Ks"])

    sb2_kdiffs = sb2s["M_K"] - sb2modelKs
    nonsb2_kdiffs = nonsb2s["M_K"] - nonsb2modelKs
    sb2_kdwarfs = sb2_kdiffs > -1.3
    nonsb2_kdwarfs = nonsb2_kdiffs > -1.3
    sb2_ksubgiantlum = np.logical_and(sb2_kdiffs < -1.3, sb2_kdiffs > -2.3)
    nonsb2_ksubgiantlum = np.logical_and(
        nonsb2_kdiffs < -1.3, nonsb2_kdiffs > -2.3)
    sb2_klumsubgiantlum = np.logical_and(sb2_kdiffs < -2.3, sb2_kdiffs > -4.5)
    nonsb2_klumsubgiantlum = np.logical_and(
        nonsb2_kdiffs < -2.3, nonsb2_kdiffs > -4.5)

    cool_sb2s = sb2s["TEFF"] <= 5250
    cool_nonsb2s = nonsb2s["TEFF"] <= 5250
    hot_sb2s = sb2s["TEFF"] > 5250
    hot_nonsb2s = nonsb2s["TEFF"] > 5250

    sb2_cooldwarfs = np.logical_and(cool_sb2s, sb2_kdwarfs)
    nonsb2_cooldwarfs = np.logical_and(cool_nonsb2s, nonsb2_kdwarfs)
    sb2_hotdwarfs = np.logical_and(hot_sb2s, sb2_kdwarfs)
    nonsb2_hotdwarfs = np.logical_and(hot_nonsb2s, nonsb2_kdwarfs)
    sb2_subgiants = np.logical_and(hot_sb2s, sb2_ksubgiantlum)
    nonsb2_subgiants = np.logical_and(hot_nonsb2s, nonsb2_ksubgiantlum)
    sb2_lumsubgiants = np.logical_and(hot_sb2s, sb2_klumsubgiantlum)
    nonsb2_lumsubgiants = np.logical_and(hot_nonsb2s, nonsb2_klumsubgiantlum)
    
    catalog.generate_abs_mag_column(
        sb2s, "phot_g_mean_mag", "M_G", lambda x: x, parallaxcol="parallax")
    catalog.generate_abs_mag_column(
        nonsb2s, "phot_g_mean_mag", "M_G", lambda x: x, parallaxcol="parallax")
    sb2s["BP-RP"] = sb2s["phot_bp_mean_mag"] - sb2s["phot_rp_mean_mag"]
    nonsb2s["BP-RP"] = nonsb2s["phot_bp_mean_mag"] - nonsb2s["phot_rp_mean_mag"]

    # I see that the Gaia G isochrone is offset from the data. I attempt to
    # correct for this by subtracting gaiaG_offset from the isochrone.
    gaiaG_offset = -0.4
    modelBP_RPs = np.linspace(0.5, 2.0, 200)
    modelGs = gaiaG_offset + iso.interpolate_isochrone_cols(
        1e9, modelBP_RPs, "BP-RP", mist.band_translation["G"], increase=False)
    sb2modelGs = gaiaG_offset + iso.interpolate_isochrone_cols(
        1e9, sb2s["BP-RP"], "BP-RP", mist.band_translation["G"], increase=False)
    nonsb2modelGs = gaiaG_offset + iso.interpolate_isochrone_cols(
        1e9, nonsb2s["BP-RP"], "BP-RP", mist.band_translation["G"],
        increase=False)


    sb2_gdiffs = sb2s["M_G"] - sb2modelGs
    nonsb2_gdiffs = nonsb2s["M_G"] - nonsb2modelGs
    sb2_gdwarfs = sb2_gdiffs > -1.3
    nonsb2_gdwarfs = nonsb2_gdiffs > -1.3
    sb2_gsubgiantlum = np.logical_and(sb2_gdiffs < -1.3, sb2_gdiffs > -2.3)
    nonsb2_gsubgiantlum = np.logical_and(
        nonsb2_gdiffs < -1.3, nonsb2_gdiffs > -2.3)
    sb2_glumsubgiantlum = np.logical_and(sb2_gdiffs < -2.3, sb2_gdiffs > -4.5)
    nonsb2_glumsubgiantlum = np.logical_and(
        nonsb2_gdiffs < -2.3, nonsb2_gdiffs > -4.5)

    temp_boundary = iso.interpolate_isochrone_cols(
        1e9, np.log10([5250]), iso.logteff_col, "BP-RP")

    lowbprp_sb2s = sb2s["BP-RP"] <= temp_boundary
    lowbprp_nonsb2s = nonsb2s["BP-RP"] <= temp_boundary
    highbprp_sb2s = sb2s["BP-RP"] > temp_boundary
    highbprp_nonsb2s = nonsb2s["BP-RP"] > temp_boundary

    sb2_gcooldwarfs = np.logical_and(highbprp_sb2s, sb2_gdwarfs)
    nonsb2_gcooldwarfs = np.logical_and(highbprp_nonsb2s, nonsb2_gdwarfs)
    sb2_ghotdwarfs = np.logical_and(lowbprp_sb2s, sb2_gdwarfs)
    nonsb2_ghotdwarfs = np.logical_and(lowbprp_nonsb2s, nonsb2_gdwarfs)
    sb2_gsubgiants = np.logical_and(lowbprp_sb2s, sb2_gsubgiantlum)
    nonsb2_gsubgiants = np.logical_and(lowbprp_nonsb2s, nonsb2_gsubgiantlum)
    sb2_glumsubgiants = np.logical_and(lowbprp_sb2s, sb2_glumsubgiantlum)
    nonsb2_glumsubgiants = np.logical_and(lowbprp_nonsb2s, nonsb2_glumsubgiantlum)
    

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    hr.absmag_teff_plot(
        nonsb2s["TEFF"][nonsb2_cooldwarfs], nonsb2s["M_K"][nonsb2_cooldwarfs], 
        marker=".", color=bc.violet, ls="", axis=ax1)
    hr.absmag_teff_plot(
        nonsb2s["TEFF"][nonsb2_hotdwarfs], nonsb2s["M_K"][nonsb2_hotdwarfs], 
        marker=".", color=bc.orange, ls="", axis=ax1)
    hr.absmag_teff_plot(
        nonsb2s["TEFF"][nonsb2_subgiants], nonsb2s["M_K"][nonsb2_subgiants], 
        marker=".", color=bc.algae, ls="", axis=ax1)
    hr.absmag_teff_plot(
        nonsb2s["TEFF"][nonsb2_lumsubgiants], 
        nonsb2s["M_K"][nonsb2_lumsubgiants], marker=".", color=bc.sky_blue, 
        ls="", axis=ax1)
    hr.absmag_teff_plot(
        sb2s["TEFF"], sb2s["M_K"], marker="*", color="r", ls="", axis=ax1, ms=3)
    hr.absmag_teff_plot(
        modelteffs, modelKs, color=bc.pink, marker="", ls="-", axis=ax1)

    ax2.plot(
        nonsb2s["BP-RP"][nonsb2_gcooldwarfs],
        nonsb2s["M_G"][nonsb2_gcooldwarfs], marker=".", color=bc.violet, ls="")
    ax2.plot(
        nonsb2s["BP-RP"][nonsb2_ghotdwarfs],
        nonsb2s["M_G"][nonsb2_ghotdwarfs], marker=".", color=bc.orange, ls="")
    ax2.plot(
        nonsb2s["BP-RP"][nonsb2_gsubgiants],
        nonsb2s["M_G"][nonsb2_gsubgiants], marker=".", color=bc.algae, ls="")
    ax2.plot(
        nonsb2s["BP-RP"][nonsb2_glumsubgiants],
        nonsb2s["M_G"][nonsb2_glumsubgiants], marker=".", color=bc.sky_blue, 
        ls="")
    ax2.plot(
        sb2s["BP-RP"], sb2s["M_G"], marker="*", color="r", ls="", ms=3)
    ax2.plot(modelBP_RPs, modelGs, marker="", ls="-", color=bc.pink)
    hr.invert_y_axis(ax2)

    ax1.set_xlabel(Teffstr)
    ax1.set_ylabel(MKstr)
    ax2.set_xlabel("BP-RP")
    ax2.set_ylabel("M_G")
    
    print("Cool Dwarf SB2 Fraction: {0}".format(format_fraction(
        np.count_nonzero(sb2_gcooldwarfs), np.count_nonzero(sb2_gcooldwarfs) +
    np.count_nonzero(nonsb2_gcooldwarfs))))
    print("Hot Dwarf SB2 Fraction: {0}".format(format_fraction(
        np.count_nonzero(sb2_ghotdwarfs), np.count_nonzero(sb2_ghotdwarfs) +
    np.count_nonzero(nonsb2_ghotdwarfs))))
    print("Subgiant SB2 Fraction: {0}".format(format_fraction(
        np.count_nonzero(sb2_gsubgiants), np.count_nonzero(sb2_gsubgiants) +
    np.count_nonzero(nonsb2_gsubgiants))))
    print("Luminous Subgiant SB2 Fraction: {0}".format(format_fraction(
        np.count_nonzero(sb2_glumsubgiants), np.count_nonzero(sb2_glumsubgiants) +
    np.count_nonzero(nonsb2_glumsubgiants))))


def DLSB_HR_Diagram(
        cool_dwarfs, dest=build_filepath(FIGURE_PATH, "cool_dlsb", "pdf"),
    teff_col="TEFF", logg_col="LOGG_FIT"):
    '''Compare DLSB locations in HR diagram to non-DLSBs.'''
    non_dlsbs = cool_dwarfs.subsample(["~Bad", "~DLSB"])
    dlsbs = cool_dwarfs.subsample(["~Bad", "DLSB"])
    assert cool_dwarfs.subsample_len(["~Bad", "Unknown DLSB", "Vsini det"]) == 0
    
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
    
def DLSB_HR_Diagram(
        cool_dwarfs, dest=build_filepath(FIGURE_PATH, "cool_dlsb", "pdf"),
    teff_col="TEFF", logg_col="LOGG_FIT"):
    '''Compare DLSB locations in HR diagram to non-DLSBs.'''
    non_dlsbs = cool_dwarfs.subsample(["~Bad", "~DLSB"])
    dlsbs = cool_dwarfs.subsample(["~Bad", "DLSB"])
    assert cool_dwarfs.subsample_len(["~Bad", "Unknown DLSB", "Vsini det"]) == 0
    
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

def write_Don_Pleiades_file():
    '''Write a file with the cross-matched vsinis.'''
    pleiades = cache.pleiades_APOGEE_Literature_vsini()

    columns = [
        "APOGEE_ID", "TEFF", "(V-K)0", "Delmag", "MK", "VSINI", "VSINI_ERR",
        "vsini_QuelozC", "vsini_err_QuelozC", "vsini_lim_QuelozC", 
        "vsini_QuelozE", "vsini_err_QuelozE", "vsini_Terndrup", 
        "vsini_err_Terndrup", "vsini_lim_Terndrup", "vsini_Soderblom",
        "vsini_lim_Soderblom", "vsini_SH", "vsini_err_SH", "vsini_lim_SH",
        "vsini_S84", "vsini_err_S84", "vsini_lim_S84", "vsini_Jackson",
        "vsini_err_Jackson", "vsini_lim_Jackson", "Per1", "Per2", "Per3", "Per4", 
        "Per_MU"]

    dontab = pleiades[columns]
    dontab.rename_column("Delmag", "DeltaV")
    dontab.rename_column("VSINI", "vsini_APOGEE")
    dontab.rename_column("VSINI_ERR", "vsini_err_APOGEE")
    dontab.rename_column("Per_MU", "Multi")
    dontab["vsini_QuelozC"].mask = np.logical_or(
        dontab["vsini_QuelozC"].mask, np.isnan(dontab["vsini_QuelozC"]))
    dontab["vsini_err_QuelozC"].mask = np.logical_or(
        dontab["vsini_err_QuelozC"].mask, np.isnan(dontab["vsini_err_QuelozC"]))
    dontab["vsini_QuelozE"].mask = np.logical_or(
        dontab["vsini_QuelozE"].mask, np.isnan(dontab["vsini_QuelozE"]))
    dontab["vsini_err_QuelozE"].mask = np.logical_or(
        dontab["vsini_err_QuelozE"].mask, np.isnan(dontab["vsini_err_QuelozE"]))

    comments = [
        "File containing cross-matched vsinis for Pleiades targets observed by",
        "APOGEE. Columns are:",
        "APOGEE_ID: 2MASS ID which identifies the APOGEE target",
        "TEFF: APOGEE-determined effective temperature",
        "(V-K)0: De-reddened V-K color for the target",
        "DeltaV: V-band excess above the main sequence",
        "MK: Absolute K-band magnitude given Pleiades distance",
        "vsini_APOGEE: APOGEE-determined vsini",
        "vsini_QuelozaC: CORAVEL spectrograph in Queloz (with limits)",
        "vsini_QuelozE: Elodie spectrograph in Queloz (no limits reported)",
        "vsini_Terndrup: Terndrup (with limits and 10\% errors)",
        "vsini_Soderblom: Soderblom (no errors reported)",
        "vsini_SH: Stauffer & Hartmann (with limits and errors)",
        "vsini_S84: Stauffer 1984 (with limits and 10\%/20\% errors </> 50km/s",
        "vsini_Jackson: Jackson/Jeffries vsini for Pleiades targets",
        "Per1/2/3/4: Periods measured by Rebull",
        "Multi: Flag for multiperiodic systems ('single' or 'multi')"]

    dontab.meta["comments"] = comments

    dontab.write(str(paths.HEAD_DIR / "don_vsini_pleiades.tab"),
                 format="ascii.fixed_width", overwrite=True,
                 formats={
                     "vsini_APOGEE": "%.2f", "vsini_err_Terndrup": "%.1f",
                     "vsini_err_SH": "%.1f", "vsini_err_S84": "%.1f", 
                     "MK": ".2f"})

def write_Jen_APOGEE_file():
    '''Write a file for Jen to predict rotational velocities.

    These files should have the TEFF, Lbol, and [M/H]. I also want to include
    quantities that were used to calculate these final products just in case
    Jen can get better answers for these targets.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    cool_dwarfs = aposplit.subsample(["Cool Dwarfs"])
    cool_dwarfs["Regime"] = "Cool Dwarfs"
    hot_dwarfs = aposplit.subsample(["Hot Dwarfs"])
    hot_dwarfs["Regime"] = "Hot Dwarfs"
    hot_subgiants = aposplit.subsample(["Subgiants"])
    hot_subgiants["Regime"] = "Subgiants"
    luminous_subgiants = aposplit.subsample(["Luminous Subgiants"])
    luminous_subgiants["Regime"] = "Luminous Subgiants"
    giants = aposplit.subsample(["Giants"])
    giants["Regime"] = "Giants"

    targs = vstack([
        cool_dwarfs, hot_dwarfs, hot_subgiants, luminous_subgiants, giants])

    newtable = targs[[
        "Regime", "APOGEE_ID", "TEFF", "TEFF_ERR", "M_K", "MIST BC (sol)", 
        "Gaia L", "Gaia L err", "M_H", "M_H_ERR", "FE_H"]]

    newtable.rename_column("Gaia L", "L/Lbol")
    newtable.rename_column("Gaia L err", "L/Lbol err")
    newtable.rename_column("MIST BC (sol)", "BC K")

    comments = [
        "File containing APOKASC targets. This file should have necessary", 
        "information for predicting rotational velocities for all targets.", 
        "Columns are: ",
        "Regime: Denotes whether targets is classified as part of the ", 
        "'Cool Dwarfs', 'Hot Dwarfs', 'Subgiants', 'Luminous Subgiants', or ",
        "'Giants'.",
        "APOGEE_ID: The APOGEE ID of the target.",
        "TEFF: The effective temperature according to APOGEE.",
        "M_K: The absolute K-band magnitude of the target.",
        "BC K: The bolometric correction for dwarfs at the target's APOGEE ",
        "temperature calculated by MIST Isochrones. Gravity corrections to ",
        "the BC are ignored. Because MIST overpredicts the ",
        "K-band luminosity with metallicity, I apply a solar-metallicity BC.",
        "L/Lbol: Bolometric Luminosity calculated from the K-band absolute",
        "magnitude and the Bolometric Correction.",
        "M_H: APOGEE Bulk metallicity of the object.",
        "FE_H: APOGEE iron abundance for the object."]

    newtable.meta["comments"] = comments

    newtable.write(
        str(paths.HEAD_DIR / "jen_apogee_targets.tab"), 
        format="ascii.fixed_width", overwrite=True, formats={
            "TEFF": "%.1f", "TEFF_ERR": "%.2f", "M_K": "%.4f", "BC K": "%.4f", 
            "L/Lbol": "%.4f", "L/Lbol err": "%.5f", "M_H": "%.3f", 
            "M_H_ERR": "%.4f", "FE_H": "%.3f"})

def write_binning_Marc_Jen():
    '''Send the data to Marc and Jen for binning.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    data = aposplit.subsample([])

    desired_columns = [
        "APOGEE_ID", "kepid", "TEFF", "TEFF_ERR", "M_K", "M_K_err1", 
        "M_K_err2", "VSINI", "VSINI_ERR"]

    newtable = data[desired_columns]
    newtable.sort("APOGEE_ID")


    comments = [
        "File containing APOGEE targets which will be used for binning",
        "rapid rotators. Columns are: ",
        "APOGEE_ID: The APOGEE ID",
        "kepid: The KIC ID",
        "TEFF: The APOGEE effective temperature.",
        "TEFF_ERR: Error in the APOGEE effective temperature.",
        "M_K: Ks-band absolute magnitude.",
        "M_K_err1: Lower uncertainty of Ks-band absolute magnitude.",
        "M_K_err2: Upper uncertainty of Ks-band absolute magnitude.",
        "VSINI: APOGEE Vsini",
        "VSINI_ERR: ASPCAP error on Vsini (unreliable; use 12\%)"]

    newtable.meta["comments"] = comments

    newtable.write(
        str(paths.HEAD_DIR / "data_for_binning.tab"),
        format="ascii.fixed_width", overwrite=True, formats={
            "TEFF": "%.1f", "TEFF_ERR": "%.2f", "M_K": "%.4f", 
            "M_K_err1": "%.4f", "M_K_err2": "%.4f", "VSINI": "%.1f", 
            "VSINI_ERR": "%.1f" })

def write_Jen_vsini_lower_limit_file():
    '''Write the file containing just vsini lower limits to send to Jen.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    cool_dwarfs = aposplit.subsample(["Cool Dwarfs", "Vsini lower"])
    cool_dwarfs["Regime"] = "Cool Dwarfs"
    hot_dwarfs = aposplit.subsample(["Hot Dwarfs", "Vsini lower"])
    hot_dwarfs["Regime"] = "Hot Dwarfs"
    hot_subgiants = aposplit.subsample(["Subgiants", "Vsini lower"])
    hot_subgiants["Regime"] = "Subgiants"
    luminous_subgiants = aposplit.subsample([
        "Luminous Subgiants", "Vsini lower"])
    luminous_subgiants["Regime"] = "Luminous Subgiants"
    giants = aposplit.subsample(["Giants", "Vsini lower"])
    giants["Regime"] = "Giants"

    targs = vstack([
        cool_dwarfs, hot_dwarfs, hot_subgiants, luminous_subgiants, giants])

    newtable = targs[[
        "Regime", "APOGEE_ID", "TEFF", "TEFF_ERR", "M_K", "MIST BC (sol)", 
        "Gaia L", "Gaia L err", "M_H", "M_H_ERR", "FE_H"]]

    newtable.rename_column("Gaia L", "L/Lbol")
    newtable.rename_column("Gaia L err", "L/Lbol err")
    newtable.rename_column("MIST BC (sol)", "BC K")

    comments = [
        "File containing APOKASC targets flagged as bad due to rapid rotation.",
        "This file should have necessary", 
        "information for predicting rotational velocities for all targets.", 
        "Columns are: ",
        "Regime: Denotes whether targets is classified as part of the ", 
        "'Cool Dwarfs', 'Hot Dwarfs', 'Subgiants', 'Luminous Subgiants', or ",
        "'Giants'.",
        "APOGEE_ID: The APOGEE ID of the target.",
        "TEFF: The effective temperature according to APOGEE.",
        "M_K: The absolute K-band magnitude of the target.",
        "BC K: The bolometric correction for dwarfs at the target's APOGEE ",
        "temperature calculated by MIST Isochrones. Gravity corrections to ",
        "the BC are ignored. Because MIST overpredicts the ",
        "K-band luminosity with metallicity, I apply a solar-metallicity BC.",
        "L/Lbol: Bolometric Luminosity calculated from the K-band absolute",
        "magnitude and the Bolometric Correction.",
        "M_H: APOGEE Bulk metallicity of the object.",
        "FE_H: APOGEE iron abundance for the object."]

    newtable.meta["comments"] = comments

    newtable.write(
        str(paths.HEAD_DIR / "jen_vsini_lower_lims.tab"), 
        format="ascii.fixed_width", overwrite=True, formats={
            "TEFF": "%.1f", "TEFF_ERR": "%.2f", "M_K": "%.4f", "BC K": "%.4f", 
            "L/Lbol": "%.4f", "L/Lbol err": "%.5f", "M_H": "%.3f", 
            "M_H_ERR": "%.4f", "FE_H": "%.3f"})

def write_figs():
    '''Run to write all figures.'''
    figlist = [
        targeting_figure, DLSB_Examples, plot_vsini_cut_in_period_space, 
        plot_APOGEE_bins_vsini_sizes, subgiant_rapid_fraction_heatmap, 
        mcquillan_rapid_rotation_heatmap, asteroseismic_vsini, 
        Pleiades_vsini_outliers, cool_vsini_veq_agreement,
        elBadry_radius_bias_cooldwarf, plot_contamination_probability_function,
        subgiant_zoomin, binned_rapid_fraction, high_alpha_HR_diagram,
        jen_subgiant_boundary_mets, huber_class_comparison]

    for fig in figlist:
        fig()   

if __name__ == "__main__":

    desc = """Generate figures and tables.

Script to automatically generate the figures and tables needed for the paper on
looking for tidally-synchronized binaries in the Kepler field."""

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("figs", nargs="*")
    parser.add_argument("--list-figs", action="store_true")
    
    args = parser.parse_args()

    figlist = [
        targeting_figure, DLSB_Examples, plot_vsini_cut_in_period_space, 
        plot_APOGEE_bins_vsini_sizes, subgiant_rapid_fraction_heatmap, 
        mcquillan_rapid_rotation_heatmap, asteroseismic_vsini, 
        Pleiades_vsini_outliers, cool_vsini_veq_agreement,
        elBadry_radius_bias_cooldwarf, plot_contamination_probability_function,
        subgiant_zoomin, binned_rapid_fraction, high_alpha_HR_diagram,
        jen_subgiant_boundary_mets, huber_class_comparison]

    for fig in figlist:
        fig()
