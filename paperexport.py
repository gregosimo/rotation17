import os
import sys
import argparse
import functools
import string
import bisect
import tempfile

import numpy as np
import numpy.core.defchararray as npstr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from astropy.table import Table, vstack, unique
from astropy.io import ascii
from scipy.integrate import quad
from astroquery.gaia import Gaia

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
import data_cache as cache
import rotation_consistency as rot
import sample_characterization as samp
import mist
import baraffe
import models
import yrec

PAPER_PATH = paths.HOME_DIR / "papers" / "rotation17"
TABLE_PATH = PAPER_PATH / "tables"
FIGURE_PATH = PAPER_PATH / "fig"
PLOT_SUFFIX = "png"
PLOT_PATH = paths.HEAD_DIR / "plots"

figsize=(12, 12)

Protstr = r"$P_{\mathrm{rot}}$"
Teffstr = r"$T_{\mathrm{eff}}$"
MKstr = r"$M_{Ks}$"

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
    apodwarfs = apo.split_subsample(["Dwarfs", "APOGEE Evolution Teff"])
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


@write_plot("targeting")
def targeting_figure():
    '''Create figure showing where the two samples lie in the HR diagram.

    Asteroseismic targets should be blue while cool dwarfs ought to be red.'''
    clean_apogee = cache.clean_apogee_splitter()

    f, (ax1, ax2) = plt.subplots(1,2, figsize=(24, 12))
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
    fullsample = clean_apogee.subsample([])

    hr.absmag_teff_plot(
        apokasc_giant["TEFF"], apokasc_giant["M_K"], color=bc.black,
        marker=".", ls="", label="", axis=ax1, zorder=1)
    hr.absmag_teff_plot(
        apogee_seismic["TEFF"], apogee_seismic["M_K"], color=bc.black, 
        marker=".", ls="", label="Asteroseismic", axis=ax1)
    hr.absmag_teff_plot(
        apokasc_dwarf["TEFF"], apokasc_dwarf["M_K"], color=bc.brown, marker=".", 
        ls="", label="", axis=ax1, zorder=2)
    hr.absmag_teff_plot(
        cool_dwarfs["TEFF"], cool_dwarfs["M_K"], color=bc.brown, marker=".", 
        ls="", label="Dwarfs", axis=ax1, zorder=2)
    hr.absmag_teff_plot(
        apogee_EB["TEFF"], apogee_EB["M_K"], color=bc.sky_blue, marker="8", 
        ls="", label="Eclipsing Binary", axis=ax1, zorder=4)
    hr.absmag_teff_plot(
        apogee2_EB["TEFF"], apogee2_EB["M_K"], color=bc.sky_blue, 
        marker="8", ls="", label="", axis=ax1, zorder=4)
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

    teff_bin_edges = np.arange(3500, 7000, 100)
    mk_bin_edges = np.arange(-8, 8, 0.02)
    count_cmap = plt.get_cmap("viridis")
    count_cmap.set_under("white")
    apogee_hist, xedges, yedges = np.histogram2d(
        fullsample["TEFF"], fullsample["M_K"], 
        bins=(teff_bin_edges, mk_bin_edges))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    asp = (extent[1]-extent[0])/(extent[3]-extent[2])
    im = ax2.imshow(apogee_hist.T, origin="lower", extent=extent,
               aspect="auto", cmap=count_cmap, norm=Normalize(vmin=1, vmax=10))
    f.colorbar(im, ax=ax2)

    # Show a 1 Gyr MIST Isochrone
    test_teffs = np.linspace(3500, 7000, 100)
    iso_ks = samp.calc_model_mag_fixed_age_feh_alpha(
        test_teffs, 0.0, "Ks", age=1e9, model="MIST v1.1")
    iso_ks_highmet = samp.calc_model_mag_fixed_age_feh_alpha(
        test_teffs, 0.5, "Ks", age=1e9, model="MIST v1.1")
    iso_ks_lowmet = samp.calc_model_mag_fixed_age_feh_alpha(
        test_teffs, -0.5, "Ks", age=1e9, model="MIST v1.1")
    ax2.plot(test_teffs, iso_ks_highmet, color=bc.pink, marker="", ls="--",
             lw=2, label="[Fe/H] = 0.5")
    ax2.plot(test_teffs, iso_ks, color=bc.pink, marker="", ls="-", lw=2,
             label="[Fe/H] = 0.0")
    ax2.plot(test_teffs, iso_ks_lowmet, color=bc.pink, marker="", ls=":", lw=2,
             label="[Fe/H] = -0.5")

    # Add a representative error bar.
    dwarfs = np.logical_and(
        fullsample["TEFF"] < 5500, fullsample["M_K"] > 2.95)
    teff_error=np.median(fullsample[dwarfs]["TEFF_ERR"])
    print(teff_error)
    median_k_errup = np.median(fullsample[dwarfs]["M_K_err1"]) 
    median_k_errdown = np.median(fullsample[dwarfs]["M_K_err2"])
    ax1.errorbar(
        [6500], [6.0], yerr=[[median_k_errdown], [median_k_errup]], 
        xerr=teff_error, elinewidth=3)
    ax1.set_xlim(7000, 3500)
    ax1.set_ylim(7.2, -8)
    ax1.set_xlabel("{0} (K)".format(Teffstr))
    ax1.set_ylabel(MKstr)
    ax1.legend(loc="upper left")
    ax2.set_ylabel(MKstr)
    ax2.set_xlim(7000, 3500)
    ax2.set_ylim(7.2, -8)
    ax2.set_xlabel("{0} (K)".format(Teffstr))
    ax2.legend(loc="upper left")

    # Print out the number of objects in each category.
    print("Number of asteroseismic targets: {0:d}".format(
        len(apokasc_giant) + len(apogee_seismic)))
    print("Number of dwarfs: {0:d}".format(
        len(apokasc_dwarf) + len(cool_dwarfs)))
    print("Number of EBs: {0:d}".format(len(apogee_EB)+len(apogee2_EB)))
    print("Number of Hosts: {0:d}".format(
        len(apogee2_koi) + len(apogee2_koi_control) + len(apogee2_monitor) +
        len(apogee_hosts)))

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

@write_plot("regimes")
def plot_APOGEE_bins_MK():
    '''Plot the different bins of evolutionary state.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    cool_dwarfs = aposplit.subsample(["Cool Dwarfs"])
    hot_dwarfs = aposplit.subsample(["Hot Dwarfs"])
    hot_subgiants = aposplit.subsample(["Subgiants"])
    luminous_subgiants = aposplit.subsample(["Luminous Subgiants"])
    giants = aposplit.subsample(["Giants"])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        cool_dwarfs["TEFF"], cool_dwarfs["M_K"], marker=".", 
        color=bc.violet, ls="", label="Cool Dwarfs", axis=ax)
    hr.absmag_teff_plot(
        hot_dwarfs["TEFF"], hot_dwarfs["M_K"], marker=".", 
        color=bc.orange, ls="", label="Hot Dwarfs", axis=ax)
    hr.absmag_teff_plot(
        hot_subgiants["TEFF"], hot_subgiants["M_K"], marker=".", 
        color=bc.algae, ls="", label="Hot Subgiants", axis=ax)
    hr.absmag_teff_plot(
        luminous_subgiants["TEFF"], luminous_subgiants["M_K"], marker=".", 
        color=bc.sky_blue, ls="", label="Luminous Subgiants", axis=ax)
    hr.absmag_teff_plot(
        giants["TEFF"], giants["M_K"], marker=".", 
        color=bc.red, ls="", label="Giants", axis=ax)

    lowT, highT = 3500, 6600
    tempsep = 5250
    dT = 5
    dM = 0.04
    # Now plot the boundaries.
    cool_Ts = np.linspace(lowT, tempsep, 100, endpoint=True)
    cool_dwarf_boundary = samp.calc_model_mag_fixed_age_feh_alpha(
        cool_Ts, 0.0, "Ks", age=1e9, model="MIST v1.1")
    hot_Ts = np.linspace(tempsep, highT, 100, endpoint=True)
    hot_dwarf_boundary = samp.calc_model_mag_fixed_age_feh_alpha(
        hot_Ts, 0.0, "Ks", age=1e9, model="MIST v1.1")
    # Cool dwarfs
    hr.absmag_teff_plot(
        cool_Ts-dT, cool_dwarf_boundary+dM-1.2, marker="", ls="-", axis=ax,
        color=bc.violet, lw=3)
    hr.absmag_teff_plot(
        np.ones(2)*tempsep-dT, np.array([cool_dwarf_boundary[-1]+dM-1.2, 7]), 
        marker="", ls="-", axis=ax, color=bc.violet, lw=3)
    # Hot dwarfs
    hr.absmag_teff_plot(
        hot_Ts+dT, hot_dwarf_boundary+dM-1.2, marker="", ls="-", axis=ax,
        color=bc.orange, lw=3)
    hr.absmag_teff_plot(
        np.ones(2)*tempsep+dT, np.array([hot_dwarf_boundary[0]+dM-1.2, 7]),
        marker="", ls="-", axis=ax, color=bc.orange, lw=3)
    # Subgiants
    hr.absmag_teff_plot(
        hot_Ts+dT, hot_dwarf_boundary-dM-1.2, marker="", ls="-", axis=ax,
        color=bc.algae, lw=3)
    hr.absmag_teff_plot(
        hot_Ts+dT, hot_dwarf_boundary+dM-2.2, marker="", ls="-", axis=ax,
        color=bc.algae, lw=3)
    hr.absmag_teff_plot(
        np.ones(2)*tempsep+dT, np.ones(2) * hot_dwarf_boundary[0] + 
        np.array([-1.2-dM, -2.2+dM]), marker="", ls="-", axis=ax, 
        color=bc.algae, lw=3)
    # Luminous Subgiants
    hr.absmag_teff_plot(
        hot_Ts+dT, hot_dwarf_boundary-dM-2.2, marker="", ls="-", axis=ax,
        color=bc.sky_blue, lw=3)
    hr.absmag_teff_plot(
        hot_Ts+dT, hot_dwarf_boundary+dM-4.75, marker="", ls="-", axis=ax,
        color=bc.sky_blue, lw=3)
    hr.absmag_teff_plot(
        np.ones(2)*tempsep+dT, np.ones(2) * hot_dwarf_boundary[0] + 
        np.array([-2.2-dM, -4.75+dM]), marker="", ls="-", axis=ax, 
        color=bc.sky_blue, lw=3)
    # Giants
    hr.absmag_teff_plot(
        cool_Ts-dT, cool_dwarf_boundary-dM-1.2, marker="", ls="-", axis=ax,
        color=bc.red, lw=3)
    hr.absmag_teff_plot(
        np.ones(2)*tempsep-dT, np.ones(2) * hot_dwarf_boundary[0] + 
        np.array([-1.2-dM, -4.75+dM]), marker="", ls="-", axis=ax, 
        color=bc.red, lw=3)
    hr.absmag_teff_plot(
        hot_Ts-dT, hot_dwarf_boundary-dM-4.75, marker="", ls="-", axis=ax,
        color=bc.red, lw=3)


                        


    ax.set_xlabel("APOGEE Teff")
    ax.set_ylabel("M_K")
    ax.set_xlim(highT, lowT)
    ax.set_ylim(7, -7) 
    ax.legend(loc="upper left")

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

@write_plot("vsini_bins")
def plot_APOGEE_bins_vsini_sizes():
    '''Plot the APOGEE bins where size correlates with vsini.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    categories = [
        "Cool Dwarfs", "Hot Dwarfs", "Subgiants", "Luminous Subgiants",
        "Giants"]
    colors = [bc.violet, bc.orange, bc.algae, bc.sky_blue, bc.red]

    init_size = 1
    f, ax = plt.subplots(1, 1, figsize=figsize)
    fullsamp = aposplit.subsample([])
    hr.absmag_teff_plot(
        fullsamp["TEFF"], fullsamp["M_K"], marker="x", color="grey", ls="", 
        label="Full", axis=ax, ms=5, alpha=0.15)

    for cat, col in zip(categories, colors):
        rapidrots = aposplit.subsample([cat, "Vsini det"])
        marginals = aposplit.subsample([cat, "Vsini marginal"])

        if cat is categories[-1]:
            label = "7 <= vsini < 10"
        else:
            label=""
        
        hr.absmag_teff_plot(
            marginals["TEFF"], marginals["M_K"], marker="o", color=col, ls="", 
            label=label, axis=ax, ms=init_size)

        vsini_bins = np.array([10, 15, 20, 25, 30])
        indices = np.digitize(rapidrots["VSINI"], vsini_bins)

        for i in range(1, len(vsini_bins)+1):
            vsini_indices = indices == i
            size = init_size + 1*i
            if cat is categories[-1]:
                if i == len(vsini_bins):
                    label = "vsini > {0}".format(vsini_bins[i-1])
                else:
                    label = "{0} <= vsini < {1}".format(
                        vsini_bins[i-1], vsini_bins[i])
            else:
                label=""
            hr.absmag_teff_plot(
                rapidrots["TEFF"][vsini_indices],
                rapidrots["M_K"][vsini_indices], marker="o", color=col,
                ls="", ms=size, axis=ax, label=label)

    ax.set_xlabel("APOGEE Teff")
    ax.set_ylabel("M_K")
    ax.set_xlim(6700, 3500)
    ax.set_ylim(6.5, -7.5) 
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

def period_vsini_matrix():
    '''Compare the number of periods and vsini in a given subsection.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    mcq = catin.read_McQuillan_catalog()
    categories = [
        "Cool Dwarfs", "Hot Dwarfs", "Subgiants", "Luminous Subgiants"]
    categories = categories[-1:]
    vsini_cats = ["Vsini det", "Vsini marginal", "Vsini nondet", "No Vsini"]

    countmatrix=np.zeros(
        (len(vsini_cats), len(vsini_cats)+1, len(categories)), dtype=np.int)

    for i, c in enumerate(categories):
        for j, v in enumerate(vsini_cats):
            countmatrix[j, 3, i] = aposplit.subsample_len([c, v, "No Mcq"])
            countmatrix[j, 4, i] = aposplit.subsample_len(
                [c, v, "Unknown Mcq"])
            mcq_sample = aposplit.subsample([c, v, "Mcq"])
            full_mcq = au.join_by_id(mcq_sample, mcq, "kepid", "KIC")
            assert len(full_mcq) == len(mcq_sample)

            max_periods_lower = rot.vsini_to_max_period(10, full_mcq["Gaia R"])
            max_periods_upper = rot.vsini_to_max_period(7, full_mcq["Gaia R"])

            countmatrix[j, 0, i] = np.count_nonzero(
                full_mcq["Prot"] < max_periods_lower)
            countmatrix[j, 1, i] = np.count_nonzero(np.logical_and(
                full_mcq["Prot"] > max_periods_lower, 
                full_mcq["Prot"] < max_periods_upper))
            countmatrix[j, 2, i] = np.count_nonzero(
                full_mcq["Prot"] > max_periods_upper)

    return countmatrix




def elbadry_binary_fractions():
    '''Write out the El-badry binary fractions in each sector.'''
    aposplit = cache.apogee_splitter_with_DSEP()

    categories = [
        "Cool Dwarfs", "Hot Dwarfs", "Subgiants", "Luminous Subgiants"]
    template_str = "{0}: {1:d}/{2:d} = {3:.2f}%"
    for cat in categories:
        sb2_num = aposplit.subsample_len([cat, "El-Badry SB2"])
        sb3_num = aposplit.subsample_len([cat, "El-Badry SB3"])
        total = aposplit.subsample_len([cat, "~No El-Badry Binarity"])

        sumnum = sb2_num + sb3_num
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

@write_plot("astero_vsini_comparison")
def asteroseismic_vsini():
    '''Plot the vsini agreement for the asteroseismic sample.'''
    astero = cache.astero_splitter()
    apokasc = astero.subsample([
        "Asteroseismic Dwarfs", "~Bad", "~No vsini", "~DLSB"])
    full_apo = astero.subsample(["~Bad"])
    garcia = catin.read_Garcia_periods()
    astero_garcia = au.join_by_id(apokasc, garcia, "KEPLER_INT", "KIC")

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    astero_radius_err = (
        (astero_garcia["RADIUS_DW_PERR"] + astero_garcia["RADIUS_DW_MERR"])/2)
    rot.plot_vsini_velocity(
        astero_garcia["VSINI"], astero_garcia["Prot"], astero_garcia["e_Prot"], 
        astero_garcia["RADIUS_DW"], astero_radius_err, ax=ax)

#   ax.set_xlabel("Veq")
#   ax.set_ylabel("Vsini")
    ax.set_title("Asteroseismic vsini agreement")

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

@write_plot("apokasc_vdists")
def asteroseismic_velocity_comparison():
    '''Plot the agreement between predicted and actual vsini.'''
    astero = cache.astero_splitter()
    apokasc = astero.subsample([
        "Asteroseismic Dwarfs", "~Bad", "~No vsini", "~DLSB"])
    full_apo = astero.subsample(["~Bad"])
    garcia = catin.read_Garcia_periods()
    astero_garcia = au.join_by_id(apokasc, garcia, "KEPLER_INT", "KIC")


    # I want to make sure upper limits are actually detected as lower limits.
    astero_velocities = rot.period_to_velocities(
        astero_garcia["Prot"], astero_garcia["RADIUS_DW"])

    rot.compare_vsini_distribution(
        astero_velocities, astero_garcia["VSINI"], vsini_cutoff=10, maxv=100, 
        nbins=100)

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

@write_plot("Cool_Dwarf_vsini_veq_comparison")
def cool_vsini_veq_agreement_Lbol():
    '''Plot the vsini and veq in a single plot with bolometric R.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_apo = aposplit.subsample(["Dwarfs", "APOGEE Evolution Cool", "Mcq", "~DLSB"])
    dlsb = aposplit.subsample(["Dwarfs", "APOGEE Evolution Cool", "Mcq", "DLSB"])
    mcq = catin.read_McQuillan_catalog()
    cool_apo_mcq = au.join_by_id(cool_apo, mcq, "kepid", "KIC")
    dlsb_mcq = au.join_by_id(dlsb, mcq, "kepid", "KIC")

    phot_bins = cool_apo_mcq["K Excess"] < -0.3

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    rot.plot_vsini_velocity(
        cool_apo_mcq["VSINI"][~phot_bins], cool_apo_mcq["Prot"][~phot_bins], 
        cool_apo_mcq["e_Prot"][~phot_bins], cool_apo_mcq["Gaia R"][~phot_bins], 
        cool_apo_mcq["Gaia R err"][~phot_bins], ax=ax, label="Rapid Rotators")
    rot.plot_vsini_velocity(
        cool_apo_mcq["VSINI"][phot_bins], cool_apo_mcq["Prot"][phot_bins], 
        cool_apo_mcq["e_Prot"][phot_bins], cool_apo_mcq["Gaia R"][phot_bins], 
        cool_apo_mcq["Gaia R err"][phot_bins], ax=ax, color="r",
        label="Photometric Binaries")
    rot.plot_vsini_velocity(
        dlsb_mcq["VSINI"], dlsb_mcq["Prot"], dlsb_mcq["e_Prot"],
        dlsb_mcq["Gaia R"], dlsb_mcq["Gaia R err"], ax=ax, color="m",
        marker="v", label="SB2")
    ax.plot([1, 100], [1.15, 115], color='k', ls="-.", marker="")
    ax.set_title("MIST Bolometric Radius")
    ax.legend(loc="lower right")

@write_plot("cool_vdists")
def cool_dwarf_velocity_comparison():
    '''Plot the agreement between predicted and actual vsini.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    cool_apo = aposplit.subsample(["Dwarfs", "APOGEE Evolution Cool", "Mcq", "~DLSB"])
    mcq = catin.read_McQuillan_catalog()
    cool_apo_mcq = au.join_by_id(cool_apo, mcq, "kepid", "KIC")

    # I want to make sure upper limits are actually detected as lower limits.
    cool_velocities = rot.period_to_velocities(
        cool_apo_mcq["Prot"], cool_apo_mcq["MIST R (APOGEE)"])

    rot.compare_vsini_distribution(
        cool_velocities, cool_apo_mcq["VSINI"], vsini_cutoff=10, maxv=100, 
        nbins=100)

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
    hot_apo = aposplit.subsample(["Dwarfs", "APOGEE Evolution Hot", "Mcq", "~DLSB"])
    dlsb = aposplit.subsample(["Dwarfs", "APOGEE Evolution Hot", "Mcq", "DLSB"])
    mcq = catin.read_McQuillan_catalog()
    hot_apo_mcq = au.join_by_id(hot_apo, mcq, "kepid", "KIC")
    dlsb_mcq = au.join_by_id(dlsb, mcq, "kepid", "KIC")

    phot_bins = hot_apo_mcq["K Excess"] < -0.3

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    rot.plot_vsini_velocity(
        hot_apo_mcq["VSINI"][~phot_bins], hot_apo_mcq["Prot"][~phot_bins], 
        hot_apo_mcq["e_Prot"][~phot_bins], hot_apo_mcq["Gaia R"][~phot_bins], 
        hot_apo_mcq["Gaia R err"][~phot_bins], ax=ax, label="Rapid Rotators")
    rot.plot_vsini_velocity(
        hot_apo_mcq["VSINI"][phot_bins], hot_apo_mcq["Prot"][phot_bins], 
        hot_apo_mcq["e_Prot"][phot_bins], hot_apo_mcq["Gaia R"][phot_bins], 
        hot_apo_mcq["Gaia R err"][phot_bins], ax=ax, color="r",
        label="Photometric Binaries")
    rot.plot_vsini_velocity(
        dlsb_mcq["VSINI"], dlsb_mcq["Prot"], dlsb_mcq["e_Prot"],
        dlsb_mcq["Gaia R"], dlsb_mcq["Gaia R err"], ax=ax, color="m",
        marker="v", label="SB2")
    ax.plot([1, 100], [1.15, 115], color='k', ls="-.", marker="")
    ax.set_title("MIST Bolometric Radius")
    ax.legend(loc="lower right")

@write_plot("hot_vdists")
def hot_dwarf_velocity_comparison():
    '''Plot the agreement between predicted and actual vsini.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    hot_apo = aposplit.subsample(["Dwarfs", "APOGEE Evolution Hot", "Mcq", "~DLSB"])
    mcq = catin.read_McQuillan_catalog()
    hot_apo_mcq = au.join_by_id(hot_apo, mcq, "kepid", "KIC")

    # I want to make sure upper limits are actually detected as lower limits.
    hot_velocities = rot.period_to_velocities(
        hot_apo_mcq["Prot"], hot_apo_mcq["MIST R (APOGEE)"])

    rot.compare_vsini_distribution(
        hot_velocities, hot_apo_mcq["VSINI"], vsini_cutoff=10, maxv=100, 
        nbins=100)

############
# Pleiades #
############

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
    ax1.set_ylabel("Dereddened V-K")
    ax1.set_xlabel("APOGEE Teff")


@write_plot("vsinicomp")
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
        2, 6, figsize=(92, 24), sharey=False)
    
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
        pleiades["(V-K)_ST"][queloz_elodie_single],
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
    axes[0, 0].set_ylabel("Dereddened K")
    axes[0, 0].set_xlabel("TEFF")

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
    axes[0, 1].set_xlabel("TEFF")
    axes[0, 1].set_ylabel("")

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
    axes[0, 2].set_xlabel("TEFF")
    axes[0, 2].set_ylabel("")

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
    axes[0, 3].set_xlabel("TEFF")
    axes[0, 3].set_ylabel("")

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
    axes[0, 4].set_xlabel("TEFF")
    axes[0, 4].set_ylabel("")

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
    axes[0, 5].set_xlabel("TEFF")
    axes[0, 5].set_ylabel("")

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
    subgiants = aposplit.subsample(["Subgiants", "Mcq", "~No Vsini", "~DLSB"])
    luminous_subgiants = aposplit.subsample(
        ["Luminous Subgiants", "Mcq", "~No Vsini", "~DLSB"])
    mcq = catin.read_McQuillan_catalog()
    subgiants_mcq = au.join_by_id(subgiants, mcq, "kepid", "KIC")
    luminous_subgiants_mcq = au.join_by_id(
        luminous_subgiants, mcq, "kepid", "KIC")

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    rot.plot_vsini_velocity(
        subgiants_mcq["VSINI"], subgiants_mcq["Prot"], subgiants_mcq["e_Prot"], 
        subgiants_mcq["Gaia R"], subgiants_mcq["Gaia R err"], ax=ax, label="")
    rot.plot_vsini_velocity(
        luminous_subgiants_mcq["VSINI"], luminous_subgiants_mcq["Prot"], 
        luminous_subgiants_mcq["e_Prot"], luminous_subgiants_mcq["Gaia R"], 
        luminous_subgiants_mcq["Gaia R err"], ax=ax, label="", color="m")
    ax.plot([1, 100], [1.15, 115], color='k', ls="-.", marker="")
    ax.set_title("MIST Bolometric Radius")
    ax.legend(loc="lower right")

def subgiant_vsini_outliers():
    '''Select off the objects which are outliers in subgiant vsini.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    subgiants = aposplit.subsample(["Subgiants", "Mcq", "~DLSB"])
    luminous_subgiants = aposplit.subsample(
        ["Luminous Subgiants", "Mcq", "~DLSB"])
    total = vstack([subgiants, luminous_subgiants])
    mcq = catin.read_McQuillan_catalog()
    total_mcq = au.join_by_id(total, mcq, "kepid", "KIC")

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
def hot_dwarf_velocity_comparison():
    '''Plot the agreement between predicted and actual vsini.'''
    aposplit = cache.apogee_splitter_with_DSEP()
    subgiants = aposplit.subsample(["Subgiants", "Mcq"])
    luminous_subgiants = aposplit.subsample(["Luminous Subgiants", "Mcq"])
    total = vstack([subgiants, luminous_subgiants])
    mcq = catin.read_McQuillan_catalog()
    total_mcq = au.join_by_id(total, mcq, "kepid", "KIC")

    # I want to make sure upper limits are actually detected as lower limits.
    total_velocities = rot.period_to_velocities(
        total_mcq["Prot"], total_mcq["MIST R (APOGEE)"])

    rot.compare_vsini_distribution(
        total_velocities, total_mcq["VSINI"], vsini_cutoff=10, maxv=100, 
        nbins=100)

########
# Gaia #
########

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

####################
# Rotation Mapping #
####################

@write_plot("period_vel")
def plot_vsini_cut_in_period_space():
    '''Plot a cut in vsini to period space.'''
    vsini_lim = 10
    full = cache.apogee_splitter_with_DSEP()
    cool_dwarfs = full.subsample(["Cool Dwarfs"])
    hot_dwarfs = full.subsample(["Hot Dwarfs"])
    subgiants = full.subsample(["Subgiants"])
    luminous_subgiants = full.subsample(["Luminous Subgiants"])

    all_tabs = [cool_dwarfs, hot_dwarfs, subgiants, luminous_subgiants]

    for tab in all_tabs:
        tab.sort("Gaia R")

    med_inds = [len(tab)//2 for tab in all_tabs]
    med_teffs = [tab["TEFF"][i] for tab, i in zip(all_tabs, med_inds)]
    med_Ks = [tab["M_K"][i] for tab, i in zip(all_tabs, med_inds)]
    med_rads = [tab["Gaia R"][i] for tab, i in zip(all_tabs, med_inds)]
    med_pers = rot.vsini_to_max_period(vsini_lim, np.array(med_rads))

    min_R = min(cool_dwarfs["Gaia R"])
    max_R = max(luminous_subgiants["Gaia R"])

    radii = np.linspace(min_R, max_R, 1000)
    max_periods = rot.vsini_to_max_period(vsini_lim, radii)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    radcolors = plt.get_cmap("viridis_r")
    norm = Normalize(vmin=min_R, vmax=max_R)
    
    for targs, med_i in zip(all_tabs, med_inds):
        sc = ax1.scatter(
            targs["TEFF"], targs["M_K"], c=targs["Gaia R"], marker=".", 
            cmap=radcolors, norm=norm)
    f.colorbar(sc, ax=ax1)
    ax1.plot(med_teffs, med_Ks, 'r*', ms=5)
    ax1.set_xlabel("Teff")
    ax1.set_ylabel("M_K")
    ax1.set_xlim(6800, 3500)
    ax1.set_ylim(7, -2)

    ax2.scatter(radii, max_periods, c=radii, marker=".", cmap=radcolors,
                norm=norm)
    ax2.plot(med_rads, med_pers, 'r*', ms=5)
    ax2.set_xlabel("Radius (Rsun)")
    ax2.set_ylabel("Maximum period (day)")

    med_R_cd = np.median(cool_dwarfs["Gaia R"])
    med_R_hd = np.median(hot_dwarfs["Gaia R"])
    med_R_sg = np.median(subgiants["Gaia R"])
    med_R_ls = np.median(luminous_subgiants["Gaia R"])

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

def APOGEE_Dwarf_Extinction():
    '''Calculate the median extinction of APOGEE Dwarfs.'''
    full = cache.categorized_apogee_splitter()
    full.split_logg(
        "LOGG", 0, ("Spec Dwarfs", "Spec Giants", "Spec Masked"), 
        logg_crit="APOGEE Mask", null_value=np.ma.masked)
    dwarfs = full.subsample(["Spec Dwarfs", "In Gaia"])
    med_ext = np.median(dwarfs["AV"])
    print(med_ext)

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
