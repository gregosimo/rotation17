import os
import sys
import argparse
import itertools

import numpy as np
import numpy.core.defchararray as npstr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import Normalize
from astropy.table import Table, vstack
from astropy.io import ascii
from astropy.stats import median_absolute_deviation
from astropy.modeling.models import Gaussian1D,Const1D
from astropy.modeling import models, fitting
import statop as stat
import functools
import scipy

sys.path.append(os.path.join(os.environ["THESIS"], "scripts"))
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
import mist
import dsep
import data_cache as cache
import eclipsing_binaries as ebs

PAPER_PATH = paths.HOME_DIR / "papers" / "rotletter18"
TABLE_PATH = PAPER_PATH / "tables"
FIGURE_PATH = PAPER_PATH / "fig"
PLOT_SUFFIX = "pdf"

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

def touch(fname, mode=0o666, dir_fd=None, **kwargs):
    '''Function which acts as touch(1) for the given file.
    
    If the given file exists, it will change the modification time to the
    current time. If it doesn't exist, it will create one.'''
    flags = os.O_CREAT | os.O_APPEND
    # In Python 3.6, os.open will accept path-like objects.
    with os.fdopen(os.open(str(fname), flags=flags, mode=mode, dir_fd=dir_fd)) as f:
        os.utime(f.fileno() if os.utime in os.supports_fd else fname,
                 dir_fd=None if os.supports_fd else dir_fd, **kwargs)



def missing_gaia_targets():
    '''Plot properties of targets missing Gaia observations.'''
    full = cache.full_apogee_splitter()
    full_data = full.subsample(["~Bad"])
    missing_gaia = full_data[full_data["dis"].mask]

    hr.logg_teff_plot(full_data["teff"], full_data["logg"], marker=".",
                        color=bc.black, ls="")
    hr.logg_teff_plot(missing_gaia["teff"], missing_gaia["logg"], marker=".",
                        color=bc.green, ls="")


@write_plot("apogee_selection")
def apogee_selection_coordinates():
    '''Show the APOGEE and McQuillan samples in selection coordinates.'''
    clean_apogee = cache.clean_apogee_splitter()

    f, (ax1, ax2) = plt.subplots(1,2, figsize=(24, 12), sharex=True, sharey=True)
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
        apokasc_dwarf["TEFF"], apokasc_dwarf["M_K"], color=bc.black, marker=".", 
        ls="", label="", axis=ax1, zorder=1)
    hr.absmag_teff_plot(
        apogee_seismic["TEFF"], apogee_seismic["M_K"], color=bc.black, 
        marker=".", ls="", label="Asteroseismic", axis=ax1)
    hr.absmag_teff_plot(
        cool_dwarfs["TEFF"], cool_dwarfs["M_K"], color=bc.brown, marker=".", 
        ls="", label="Cool Dwarf", axis=ax1, zorder=2)
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
    im = ax2.imshow(apogee_hist.T, origin="lower", extent=extent,
               aspect=(extent[1]-extent[0])/(extent[3]-extent[2]),
               cmap=count_cmap, norm=Normalize(vmin=1, vmax=10))
    f.colorbar(im, ax=ax2)

    # Add a representative error bar.
    dwarfs = np.logical_and(
        fullsample["TEFF"] < 5500, fullsample["M_K"] > 2.95)
    teff_error=120
    median_k_errup = np.median(fullsample[dwarfs]["M_K_err1"]) 
    median_k_errdown = np.median(fullsample[dwarfs]["M_K_err2"])
    ax1.errorbar(
        [3500], [3.0], yerr=[[median_k_errdown], [median_k_errup]], 
        xerr=teff_error, elinewidth=3)
    ax1.set_xlim(7000, 3000)
    ax1.set_ylim(8, -8)
    ax1.set_xlabel("APOGEE Teff (K)")
    ax1.set_ylabel("Gaia $M_K$")
    ax1.legend(loc="upper left")
    ax2.set_xlabel("APOGEE Teff (K)")

    # Print out the number of objects in each category.
    print("Number of asteroseismic targets: {0:d}".format(
        len(apokasc_dwarf) + len(apokasc_giant) + len(apogee_seismic)))
    print("Number of cool dwarfs: {0:d}".format(len(cool_dwarfs)))
    print("Number of EBs: {0:d}".format(len(apogee_EB)+len(apogee2_EB)))
    print("Number of Hosts: {0:d}".format(
        len(apogee2_koi) + len(apogee2_koi_control) + len(apogee2_monitor) +
        len(apogee_hosts)))

@write_plot("mcquillan_selection")
def mcquillan_selection_coordinates():
    mcq = catin.mcquillan_with_stelparms()
    nomcq = catin.mcquillan_nondetections_with_stelparms()

    catalog.generate_abs_mag_column_with_errors(
        mcq, "kmag", "kmag_err", "KIC M_K", "KIC M_K_err1", "KIC M_K_err2", 
        samp.AV_to_AK, samp.AV_err_to_AK_err, distcol="dist", 
        dist_up_col="dist_err1", dist_down_col="dist_err2", avcol="av",
        avupcol="av_err1", avdowncol="av_err2")
    catalog.generate_abs_mag_column_with_errors(
        nomcq, "kmag", "kmag_err", "KIC M_K", "KIC M_K_err1", "KIC M_K_err2", 
        samp.AV_to_AK, samp.AV_err_to_AK_err, distcol="dist", 
        dist_up_col="dist_err1", dist_down_col="dist_err2", avcol="av",
        avupcol="av_err1", avdowncol="av_err2")

    f, (ax2, ax3) = plt.subplots(
        1,2, figsize=(24,12))
    teff_bin_edges = np.arange(4000, 7000, 50)
    mk_bin_edges = np.arange(-7, 8, 0.02)

    count_cmap = plt.get_cmap("viridis")
    count_cmap.set_under("white")
    mcq_gaia_hist, xedges, yedges = np.histogram2d(
        mcq["SDSS-Teff"], mcq["M_K"], bins=(teff_bin_edges, mk_bin_edges))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax2.imshow(mcq_gaia_hist.T, origin="lower", extent=extent,
               aspect=(extent[1]-extent[0])/(extent[3]-extent[2]),
               cmap=count_cmap, norm=Normalize(vmin=1))
    f.colorbar(im, ax=ax2)
    ratio_cmap = plt.get_cmap("viridis")
    ratio_cmap.set_bad(color="white")
    mcq_nondet_hist, xedges, yedges = np.histogram2d(
        nomcq["SDSS-Teff"], nomcq["M_K"], bins=(teff_bin_edges, mk_bin_edges))
    mcq_detfrac_hist = np.ma.masked_invalid(
        mcq_gaia_hist / (mcq_gaia_hist+mcq_nondet_hist))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax3.imshow(mcq_detfrac_hist.T, origin="lower", extent=extent,
               aspect=(extent[1]-extent[0])/(extent[3]-extent[2]),
               cmap=ratio_cmap)
    f.colorbar(im, ax=ax3)

    ax2.set_xlim(7000, 4000)
    ax2.set_ylim(8, -7)
    ax2.set_ylabel("Gaia $M_K$")
    ax2.set_xlabel("Pinsonneault et al (2012) Teff (K)")
    ax2.set_title("Post-Gaia period detections")
    ax3.set_xlim(7000, 4000)
    ax3.set_ylim(8, -7)
    ax3.set_xlabel("Pinsonneault et al (2012) Teff (K)")
    ax3.set_title("Post-Gaia detection fraction")

def isochrone_radius_teff_age():
    '''Plot how the radius changes with age as a function of Teff.'''
    dsep_solar = dsep.DSEPIsochrone.isochrone_from_file(0.0)

    test_teff = np.linspace(3500, 6500, 1000)
    young_radii = dsep_solar.interpolate_to_radius(
        1.0, np.log10(test_teff), dsep_solar.logteff_col, interp_kind="linear",
        mask_outside_bounds=True)
    old_radii = dsep_solar.interpolate_to_radius(
        10.0, np.log10(test_teff), dsep_solar.logteff_col, interp_kind="linear",
        mask_outside_bounds=True)

    plt.plot(test_teff, old_radii/young_radii, color=bc.blue, ls="-", marker="")
    plt.plot([3500, 6500], [1, 1], 'k-')
    plt.xlabel("Teff (K)")
    plt.ylabel("R (10 Gyr) / R(1 Gyr)")
    hr.invert_x_axis()

def isochrone_difference_ages():
    '''Plot the difference between DSEP and MIST isochrones at different ages.

    This will plot the differences in K between DSEP and MIST isochrones as a
    function of temperature for different ages.'''
    mist_solar = mist.MISTIsochrone.isochrone_from_file(0.0)
    dsep_solar = dsep.DSEPIsochrone.isochrone_from_file(0.0)

    test_teff = np.linspace(3500, 6500, 1000)
    mist_young_k = mist.interpolate_MIST_isochrone_cols(
        mist_solar, 1.0, np.log10(test_teff), outcol="2MASS_Ks")
    dsep_young_k = dsep.interpolate_DSEP_isochrone_cols(
        dsep_solar, 1.0, np.log10(test_teff), outcol="Ks")
    mist_medium_k = mist.interpolate_MIST_isochrone_cols(
        mist_solar, 5.5, np.log10(test_teff))
    dsep_medium_k = dsep.interpolate_DSEP_isochrone_cols(
        dsep_solar, 5.5, np.log10(test_teff))
    mist_old_k = mist.interpolate_MIST_isochrone_cols(
        mist_solar, 10.0, np.log10(test_teff))
    dsep_old_k = dsep.interpolate_DSEP_isochrone_cols(
        dsep_solar, 10.0, np.log10(test_teff))

    f, ax = plt.subplots(1,1, figsize=(12,12))

    hr.absmag_teff_plot(test_teff, mist_young_k-dsep_young_k, marker="",
                        linestyle="-", color=bc.black, label="1 Gyr", axis=ax)
    hr.absmag_teff_plot(test_teff, mist_medium_k-dsep_medium_k, marker="",
                        linestyle="--", color=bc.black, label="5.5 Gyr",
                        axis=ax)
    hr.absmag_teff_plot(test_teff, mist_old_k-dsep_old_k, marker="",
                        linestyle=":", color=bc.black, label="10 Gyr")
    ax.set_xlabel("Teff")
    ax.set_ylabel("MIST Ks - DSEP Ks (mag)")
    ax.set_xlim(6500, 3500)
    ax.set_ylim(0.4, -0.4)
    plt.legend(loc="upper right")

def isochrone_difference_metallicity():
    '''Plots the difference in isochrones over a large swath of metallicity.'''
    teffpoints = np.linspace(6000, 3500, 6, endpoint=True)
    fehpoints = np.linspace(-2.5, 0.5, 20)
    plotcolors = ["#253494", "#2c7fb8", "#41b6c4", "#7fcdbb", "#c7e9b4", "#ffffcc"]
    f, ax = plt.subplots(1,1, figsize=(12,12))
    for teff, color in zip(teffpoints, plotcolors):
        mist_ks = samp.calc_model_mag_fixed_age_alpha(
            [teff], fehpoints, "Ks", model="MIST")
        dsep_ks = samp.calc_model_mag_fixed_age_alpha(
            [teff], fehpoints, "Ks", model="DSEP")
        ax.plot(fehpoints, mist_ks-dsep_ks, color=color, linestyle="-",
                marker="o", label="{0:d} K".format(int(teff)),
                markerfacecolor=bc.black)
    hr.invert_y_axis(ax)
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("MIST Ks - DSEP Ks (mag)")
    ax.legend(loc="upper left")

@write_plot("mist_hr")
def mist_iso_hr_test():
    '''Plot the HR diagram with MIST isochrones overlayed.'''
    full = cache.clean_apogee_splitter()
    full_data = full.subsample([])

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    minorlocator = AutoMinorLocator()
    # Plot the data.
    hr.absmag_teff_plot(
        full_data["TEFF"], full_data["M_K"], marker=".", color=bc.black, ls="",
        axis=ax, label="")

    # Now put on the tracks
    teffs = np.linspace(6500, 3500, 200)
    youngks = samp.calc_model_mag_fixed_age_feh_alpha(teffs, 0.0, "Ks", age=1e9)
    midks = samp.calc_model_mag_fixed_age_feh_alpha(teffs, 0.0, "Ks", age=4.5e9)
    oldks = samp.calc_model_mag_fixed_age_feh_alpha(teffs, 0.0, "Ks", age=9e9)
    hr.absmag_teff_plot(
        teffs, youngks, marker="", ls="-", color=bc.blue, axis=ax, 
        label="1 Gyr", lw=3)
    hr.absmag_teff_plot(
        teffs, midks, marker="", ls="-", color=bc.purple, axis=ax, 
        label="4.5 Gyr", lw=3)
    hr.absmag_teff_plot(
        teffs, oldks, marker="", ls="-", color=bc.red, axis=ax, label="9 Gyr",
        lw=3)
    ax.set_xlabel("APOGEE Teff (K)")
    ax.set_ylabel("$M_K$")
    ax.set_ylim(6, 2.5)
    ax.legend(loc="lower left")

@write_plot("mist_uncertainties")
def mist_uncertainties():
    '''Plot the uncertainty in Ks due to [Fe/H] uncertainty.'''
    f, ax = plt.subplots(1, 1, figsize=(12,12))
    teffgrid, teffstep = np.linspace(
        3500, 7000, 200+1, endpoint=True, retstep=True)
    fehs = [-0.25, 0.0, 0.25]
    lowmet = samp.calc_model_mag_fixed_age_feh_alpha(
        teffgrid, fehs[0], "Ks", age=1e9)
    solmet = samp.calc_model_mag_fixed_age_feh_alpha(
        teffgrid, fehs[1], "Ks", age=1e9)
    highmet = samp.calc_model_mag_fixed_age_feh_alpha(
        teffgrid, fehs[2], "Ks", age=1e9)
    oldmet = samp.calc_model_mag_fixed_age_feh_alpha(
        teffgrid, fehs[1], "Ks", age=9e9)
    # Uncertainties due to metallicity
    meterr = 0.1*np.abs((highmet[1:-1] - lowmet[1:-1])/(fehs[2]-fehs[0]))
    nometerr = 0.15*np.abs((highmet[1:-1] - lowmet[1:-1])/(fehs[2]-fehs[0]))

    # Uncertainties due to temperature.
    teffdiff = np.diff(solmet)
    apogee_teffdiff = np.exp(4.58343 + 0.000289796*(teffgrid[1:-1]-4500))
    tefferr = apogee_teffdiff*np.abs((teffdiff[1:]+teffdiff[:-1])/(2*teffstep))

    # Uncertainties due to age.
    ageerr = np.abs((oldmet[1:-1] - solmet[1:-1])) / np.sqrt(12)

    fullmeterr = np.sqrt(meterr**2 + tefferr**2)
    fullnometerr = np.sqrt(nometerr**2 + tefferr**2)

    newteffgrid = teffgrid[1:-1]

    ax.plot(newteffgrid, tefferr, marker="", ls="-", color=bc.red, 
            label="APOGEE Teff Errors")
    ax.plot(newteffgrid, meterr, marker="", ls="-", color=bc.blue,
            label="0.1 dex [Fe/H] Errors")
    ax.plot(newteffgrid, nometerr, marker="", ls="--", color=bc.blue,
            label="0.15 dex [Fe/H] Errors")
    ax.plot(newteffgrid, fullmeterr, marker="", ls="-", color=bc.black,
            label="Teff + [Fe/H]", lw=3)
    ax.plot(newteffgrid, fullnometerr, marker="", ls="--", color=bc.black, lw=3)
    ax.plot(newteffgrid, ageerr, marker="", ls="-", color=bc.orange,
            label="Age Error")
    hr.invert_x_axis(ax)
    ax.set_xlabel("Teff (K)")
    ax.set_ylabel(r"$M_K$ error")
    ax.legend(loc="upper left")

@write_plot("mist_teff_uncertainty")
def mist_teff_uncertainty():
    '''Plot the uncertainty in Ks due to Teff uncertainty.'''
    f, ax = plt.subplots(1, 1, figsize=(12,12))
    teffgrid, teffstep = np.linspace(
        3500, 7000, 101, endpoint=True, retstep=True)
    kerr = samp.calc_model_mag_err_fixed_age_feh_alpha(
        teffgrid, 0.0, "Ks", teff_err=150, age=1e9)
    
    ax.plot(teffgrid, kerr, marker="", ls="-", color=bc.black)
    hr.invert_x_axis(ax)
    ax.set_xlabel("Teff (K)")
    ax.set_ylabel(r"$(120)\frac{dK}{dT_{eff}}$")

@write_plot("full_kdiff")
def subtracted_K_plot():
    '''Show the full sample in coordinates of subtracted K.'''
    full = cache.apogee_splitter_with_DSEP()
    full_data = full.subsample([])

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    minorLocator = AutoMinorLocator()
    hr.absmag_teff_plot(
        full_data["TEFF"], full_data["K Excess"], color=bc.black, marker=".",
        ls="", axis=ax)
    ax.plot([6500, 3500], [0, 0], 'k-')
    ax.plot([6500, 3500], [-1.3, -1.3], 'k--')
    ax.set_xlabel("APOGEE Teff (K)")
    ax.set_ylabel(r"$M_K$ - $M_K$ (DSEP; [Fe/H] adjusted)")

@write_plot("metallicity")
def dwarf_metallicity():
    '''Show the metallicity distribution of the cool dwarfs.'''
    full = cache.apogee_splitter_with_DSEP()
    full_data = full.subsample(["Dwarfs", "APOGEE Statistics Teff"])

    f, ax = plt.subplots(1,1, figsize=(12,12))
    minorLocator = AutoMinorLocator()
    xminorLocator = AutoMinorLocator()
    ax.hist(full_data["FE_H"], cumulative=True, normed=True, bins=200)
    median = np.median(full_data["FE_H"])
    ax.yaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.plot([-1.25, median], [0.5, 0.5], 'k-', lw=3)
    ax.plot([median, median], [0.5, 0], 'k-', lw=3)
    ax.plot([-1.25, -0.14], [0.165, 0.165], 'k-', lw=1)
    ax.plot([-0.14, -0.14], [0.165, 0], 'k--', lw=1)
    ax.plot([-1.25, 0.235], [0.835, 0.835], 'k-', lw=1)
    ax.plot([0.235, 0.235], [0.835, 0], 'k--', lw=1)
    ax.set_xlabel("APOGEE [Fe/H]")
    ax.set_ylabel("Cumulative distribution")
    ax.set_xlim(-1.25, 0.46)

def median_absolute_deviation_custom(vals, centerval=None):
    '''Calculate the median absolute deviation about a centerval.
    
    The usual definition of median absolute deviation uses the deviation about
    the median of the sample. This function allows the point about which the
    deviation is calculated to be specified.'''
    if centerval is None:
        mad =  median_absolute_deviation(vals)
    else:
        deviations = vals - centerval
        absolute_deviations = np.abs(deviations)
        mad = np.median(absolute_deviations)
        
    return mad


def metallicity_bins():
    '''Make a 2x2 plot of models in each metallicity bin.'''
    f, axes = plt.subplots(2,2, figsize=(12,12), sharex=True, sharey=True)
    axeslist = itertools.chain(*axes)
    targs = cache.apogee_splitter_with_DSEP()
    dwarfs = targs.subsample(["Dwarfs"])
#    metallicity_bin_edges = np.array([-2.0, -0.5, 0.0, 0.2, 0.5])
    metallicity_bin_edges = np.array([-0.5, -0.3, 0.0, 0.2, 0.5])
    mean_mets = (metallicity_bin_edges[1:] + metallicity_bin_edges[:-1])/2
    bin_indices = np.digitize(dwarfs["FE_H"], metallicity_bin_edges)
    inputteffs = np.linspace(3500, 6500, 500)
    # These should both be of the shape len(mean_mets) x len(teffs)
    oldKs = samp.calc_model_mag_fixed_age_alpha(
        inputteffs, mean_mets, "Ks", age=9e9, model="MIST")
    youngKs = samp.calc_model_mag_fixed_age_alpha(
        inputteffs, mean_mets, "Ks", age=1e9, model="MIST")
    for low_met_index, high_met_index, mid_met, ax, in zip(
            range(len(metallicity_bin_edges)-1), 
            range(1, len(metallicity_bin_edges)), mean_mets, axeslist):
        data_bin = dwarfs[bin_indices == high_met_index]
        # Plot the data.
        hr.absmag_teff_plot(
            data_bin["TEFF"], data_bin["K Excess"], color=bc.black, marker=".", 
            ls="", axis=ax, label="Spectroscopic Sample")
        ax.plot([6500, 3500], [0, 0], color=bc.black, lw=3, ls="-", marker="")
        # Now calculate the median and lower 1-sigma percentile.
        # Is this even necessary anymore? Or will it be shown on median
        # metallicity.
        teff_bins = np.linspace(5500, 4000, 4)
        teff_indices = np.digitize(data_bin["TEFF"], teff_bins)
        k_medians = np.zeros(len(teff_bins)-1)
        k_1sigs = np.zeros(len(teff_bins)-1)
        for i in range(1, len(teff_bins)):
            teff_data_bin = data_bin[teff_indices == i]
            if len(teff_data_bin) > 0:
                k_medians[i-1] = np.percentile(teff_data_bin["K Excess"], 100-25)
                teff_data_bin.sort("K Excess")
                single_data = teff_data_bin[len(teff_data_bin)//2:]
                k_1sigs[i-1] = k_medians[i-1] + median_absolute_deviation(
                    single_data["K Excess"])
            else:
                k_medians[i-1] = np.nan
                k_1sigs[i-1] = np.nan
        # Now plot the medians and lower 1-sigma percentiles
        teff_means = (teff_bins[:-1]+teff_bins[1:])/2
#       hr.absmag_teff_plot(teff_means, k_medians, color=bc.red, marker="x",
#                           ls="--", ms=4, label="Median", axis=ax)
#       hr.absmag_teff_plot(teff_means, k_1sigs, color=bc.red, marker=".",
#                           ls="--", ms=4, label="1-sigma", axis=ax)

        # Now plot a isochrone at the median metallicity.
        # While I want the full isochrone, it's been doing weird things over
        # mass, so I'm just going to stick to the teff space.
        # The DSEP isochrones will stop keeping track of masses at some point.
        hr.absmag_teff_plot(
            inputteffs, oldKs[low_met_index,:]-youngKs[low_met_index,:], 
            color=bc.blue, label="9 Gyr", ls="-", marker="", axis=ax)

        ax.set_title("{0:3.1f} < [Fe/H] <= {1:3.1f}".format(
            metallicity_bin_edges[low_met_index],
            metallicity_bin_edges[high_met_index]))
        ax.set_xlabel("")
        ax.set_ylabel("")
    ax.set_xlim(6500, 3500)
    ax.set_ylim(0.5, -1.2)
    axes[0][0].set_ylabel("M_K - MIST K")
    axes[1][0].set_ylabel("M_K - MSIT K")
    axes[1][0].set_xlabel("Teff (K)")
    axes[1][1].set_xlabel("Teff (K)")

def running_percentile(xvals, yvals, size, percentile=0.5):
    '''Calculate a running percentile for xvals and yvals.'''
    median_xs = np.zeros(len(xvals)-size)
    median_ys = np.zeros(len(yvals)-size)
    for start_index, end_index in zip(
            range(0, len(xvals)-size), range(size, len(xvals))):
        median_xs[start_index] = np.mean(xvals[start_index:end_index])
        median_ys[start_index] = np.percentile(
            xvals[start_index:end_index], percentile)
    return median_xs, median_ys

def median_over_metallicity():
    '''Plot the behavior of the 25th percentile over metallicity.'''
    targs = cache.apogee_splitter_with_DSEP()
    dwarfs = targs.subsample(["Dwarfs"])
    # I want evenly-populated bins.
    metallicity_bin_edges = np.percentile(
        dwarfs["FE_H"], np.linspace(0, 100, 5+1, endpoint=True))
    metallicity_bin_indices = np.digitize(dwarfs["FE_H"], metallicity_bin_edges)
    teff_bin_edges = np.linspace(4000, 5500, 2+1, endpoint=True)
    teff_bin_indices = np.digitize(dwarfs["TEFF"], teff_bin_edges)
    # This will hold the percentiles for the hot and cool bins.
    percentiles = np.zeros((len(metallicity_bin_edges)-1, len(teff_bin_edges)))
    mads = np.zeros((len(metallicity_bin_edges)-1, len(teff_bin_edges)))
    teff_colors = [bc.red, bc.blue]
    for low_met_index, high_met_index in zip(
            range(len(metallicity_bin_edges)-1), 
            range(1, len(metallicity_bin_edges))):
        for low_teff_index, high_teff_index in zip(
                range(0, len(teff_bin_edges)-1), 
                range(1, len(teff_bin_edges))):
            tablebin = dwarfs[
                np.logical_and(metallicity_bin_indices == high_met_index, 
                teff_bin_indices == high_teff_index)]
            print(len(tablebin))
            try:
                percentiles[low_met_index,low_teff_index] = np.percentile(
                    tablebin["K Excess"], 100-25)
            except IndexError:
                percentiles[low_met_index,low_teff_index] = np.nan
                mads[low_met_index,low_teff_index] = np.nan
            else:
                tablebin.sort("K Excess")
                dwarfbin = tablebin[len(tablebin)//2:]
                plt.plot(
                    dwarfbin["FE_H"], dwarfbin["K Excess"], marker=".",
                    color=teff_colors[low_teff_index], ls="", label="")
                mads[low_met_index,low_teff_index] = median_absolute_deviation(
                    dwarfbin["K Excess"])

    met_avg = (metallicity_bin_edges[:-1]+metallicity_bin_edges[1:])/2
    teff_avg = (teff_bin_edges[:-1]+teff_bin_edges[1:])/2
    plt.errorbar(met_avg, percentiles[:,0], marker="o", ls="-", color=bc.blue,
             label="{0:d} < Teff < {1:d}".format(
                 int(teff_bin_edges[0]), int(teff_bin_edges[1])),
             yerr=mads[:,0], capsize=5)
    plt.errorbar(met_avg, percentiles[:,1], marker="o", ls="-", color=bc.red,
             label="{0:d} < Teff < {1:d}".format(
                 int(teff_bin_edges[1]), int(teff_bin_edges[2])),
             yerr=mads[:,1], capsize=5)

    # Now calculate the age differences.
    oldtracks = samp.calc_model_mag_fixed_age_alpha(
        teff_avg, met_avg, "Ks", age=9e9, model="MIST")
    youngtracks = samp.calc_model_mag_fixed_age_alpha(
        teff_avg, met_avg, "Ks", age=1e9, model="MIST")
    oldexcess = oldtracks - youngtracks
    plt.fill_between(
        met_avg, oldexcess[:,0], np.zeros(oldexcess.shape[0]), color=bc.blue, 
        alpha=0.3, label="T={0:d} K; 1-9 Gyr range".format(int(teff_avg[0])))
    plt.fill_between(
        met_avg, oldexcess[:,1], np.zeros(oldexcess.shape[0]), color=bc.red, 
        alpha=0.3, label="T={0:d} K; 1-9 Gyr range".format( int(teff_avg[1])))
    plt.plot(met_avg, np.zeros(len(met_avg)), color=bc.black, marker="",
             ls="--")

    redchisq = np.sum(
        (percentiles/(1.4826*mads))**2, axis=0)/(len(met_avg)-1)
    for teff, chisq in zip(teff_avg, redchisq):
        print("Chi-squared for {0:d} K bin is {1:.1f}.".format(
            int(teff), chisq))

    plt.xlabel("[Fe/H]")
    plt.ylabel("M_K - DSEP K (1.0 Gyr) 25% excess percentile")
    hr.invert_y_axis()
    plt.legend(loc="upper right")

@write_plot("metallicity_correction")
def metallicity_corrected_excesses():
    '''Plot the indices corrected by metallicity.'''
    f, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [2, 1]}, sharex=True, 
        figsize=(12, 15))
    targs = cache.apogee_splitter_with_DSEP()
    coolsamp = targs.subsample(["Dwarfs", "APOGEE MetCor Teff"])
    # I want values in metallicity bins.
    metallicity_bin_edges = np.percentile(
        coolsamp["FE_H"], np.linspace(0, 100, 6+1, endpoint=True))
#   metallicity_bin_edges = np.array([-1.2, -0.5, -0.25, 0.0, 0.25, 0.5])
    print(len(coolsamp))
    metallicity_bin_indices = np.digitize(
        coolsamp["FE_H"], metallicity_bin_edges)
    percentiles = np.zeros(len(metallicity_bin_edges)-1)
    med_met = np.zeros(len(metallicity_bin_edges)-1)
    weights = np.zeros(len(metallicity_bin_edges)-1)
    # I want ind to start at 1 and end right before the binedges length.
    # metallicity_bin_edges[ind] denotes the high index.
    for ind in range(1, len(metallicity_bin_edges)):
        tablebin = coolsamp[metallicity_bin_indices == ind]
        percentiles[ind-1] = np.percentile(
            tablebin["K Excess"], 100-25)
        med_met[ind-1] = np.mean(tablebin["FE_H"])
        weights[ind-1] = len(tablebin)
    cor_coeff = np.polyfit(med_met, percentiles, 2)
    cor_poly = np.poly1d(cor_coeff)
    
    ax1.plot(coolsamp["FE_H"], coolsamp["K Excess"], marker=".", color=bc.black,
             ls="", label="Original")
    ax1.plot(med_met, percentiles, marker="o", color=bc.red, ls="",
             label="Binned")
    testx = np.linspace(-1.0, 0.5, 100, endpoint=True)
    ax1.plot(testx, cor_poly(testx), color=bc.red, linestyle="-", label="Fit")
    ax1.plot([testx[0], testx[-1]], [0,0], 'k--', label="")
    ax1.set_ylabel("$M_K$ - $M_K$ (MIST; 1 Gyr)")
    ax1.set_ylim(0.3, -1.2)
    ax1.set_xlim(-0.55, 0.45)
    ax1.legend(loc="upper left")
    residual = coolsamp["K Excess"] - cor_poly(coolsamp["FE_H"])
    singles = residual > np.percentile(residual, 50)
    binaries = ~singles
    mad = median_absolute_deviation_custom(residual[singles])
    ax2.plot(
        coolsamp["FE_H"][singles], residual[singles], color=bc.black, 
        marker="o", ls="", label="Single")
    ax2.plot(
        coolsamp["FE_H"][binaries], residual[binaries], color=bc.black, 
        marker="o", ls="", label="Binary")
    ax2.plot([testx[0], testx[-1]], [0,0], 'k--', label="")
    hr.invert_y_axis(ax2)
    ax2.set_xlabel("[Fe/H]")
    ax2.set_ylabel("Residual")

@write_plot("spec_teff_correction")
def spec_temperature_correction():
    f, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [2, 1]}, sharex=True, 
        figsize=(12, 15))
    targs = cache.apogee_splitter_with_DSEP()
    coolsamp = targs.subsample(["Dwarfs", "APOGEE MetCor Teff"])
    # I want values in metallicity bins.
    teff_bin_edges = np.percentile(
        coolsamp["TEFF"], np.linspace(0, 100, 5+1, endpoint=True))
    teff_bin_indices = np.digitize(
        coolsamp["TEFF"], teff_bin_edges)
    percentiles = np.zeros(len(teff_bin_edges)-1)
    med_teff = np.zeros(len(teff_bin_edges)-1)
    # I want ind to start at 1 and end right before the binedges length.
    # teff_bin_edges[ind] denotes the high index.
    for ind in range(1, len(teff_bin_edges)):
        tablebin = coolsamp[teff_bin_indices == ind]
        percentiles[ind-1] = np.percentile(
            tablebin["Partly Corrected K Excess"], 100-25)
        med_teff[ind-1] = np.mean(tablebin["TEFF"])
    cor_coeff = np.polyfit(med_teff, percentiles, 1)
    cor_poly = np.poly1d(cor_coeff)
    
    ax1.plot(coolsamp["TEFF"], coolsamp["Partly Corrected K Excess"], 
             marker=".", color=bc.black, ls="", label="Original")
    ax1.plot(med_teff, percentiles, marker="o", color=bc.red, ls="",
             label="Binned")
    testx = np.linspace(4000, 5250, endpoint=True)
    ax1.plot(testx, cor_poly(testx), color=bc.red, linestyle="-", label="Fit")
    ax1.plot([testx[0], testx[-1]], [0,0], 'k--', label="")
    ax1.set_ylabel("$M_K$ - $M_K$ (MIST; 1 Gyr)")
    ax1.set_ylim(0.3, -1.2)
    ax1.legend(loc="upper left")
    residual = coolsamp["Partly Corrected K Excess"] - cor_poly(coolsamp["TEFF"])
    mad = median_absolute_deviation(residual)
    ax2.plot(coolsamp["TEFF"], residual, color=bc.black, marker="o", ls="")
    ax2.plot([testx[0], testx[-1]], [0,0], 'k--', label="")
    hr.invert_y_axis(ax2)
    hr.invert_x_axis(ax2)
    ax2.set_xlabel("APOGEE Teff (K)")
    ax2.set_ylabel("Residual")
    ax2.set_xlim(5250, 4000)

@write_plot("phot_teff_correction")
def phot_temperature_correction():
    '''Plot the behavior of the sample over [Fe/H].'''
    f, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [2, 1]}, sharex=True, 
        figsize=(12, 15))
    targs = cache.apogee_splitter_with_DSEP()
    coolsamp = targs.subsample(["Dwarfs", "Pinsonneault MetCor Teff"])
    # I want values in metallicity bins.
    teff_bin_edges = np.percentile(
        coolsamp["SDSS-Teff"], np.linspace(0, 100, 5+1, endpoint=True))
    teff_bin_indices = np.digitize(
        coolsamp["SDSS-Teff"], teff_bin_edges)
    percentiles = np.zeros(len(teff_bin_edges)-1)
    med_teff = np.zeros(len(teff_bin_edges)-1)
    # I want ind to start at 1 and end right before the binedges length.
    # teff_bin_edges[ind] denotes the high index.
    for ind in range(1, len(teff_bin_edges)):
        tablebin = coolsamp[teff_bin_indices == ind]
        percentiles[ind-1] = np.percentile(
            tablebin["Solar K Excess"], 100-25)
        med_teff[ind-1] = np.mean(tablebin["SDSS-Teff"])
    cor_coeff = np.polyfit(med_teff, percentiles, 1)
    cor_poly = np.poly1d(cor_coeff)
    
    ax1.plot(coolsamp["SDSS-Teff"], coolsamp["Solar K Excess"], marker=".", 
             color=bc.black, ls="", label="Original")
    ax1.plot(med_teff, percentiles, marker="o", color=bc.red, ls="",
             label="Binned")
    testx = np.linspace(4000, 5250, endpoint=True)
    ax1.plot(testx, cor_poly(testx), color=bc.red, linestyle="-", label="Fit")
    ax1.plot([testx[0], testx[-1]], [0,0], 'k--', label="")
    ax1.set_ylabel("$M_K$ - $M_K$ (MIST; [Fe/H]=0.08; 1 Gyr)")
    ax1.set_ylim(0.3, -1.2)
    ax1.legend(loc="upper left")
    residual = coolsamp["Solar K Excess"] - cor_poly(coolsamp["SDSS-Teff"])
    mad = median_absolute_deviation(residual)
    ax2.plot(coolsamp["SDSS-Teff"], residual, color=bc.black, marker="o", ls="")
    ax2.plot([testx[0], testx[-1]], [0,0], 'k--', label="")
    hr.invert_y_axis(ax2)
    hr.invert_x_axis(ax2)
    ax2.set_xlabel("Pinsonneault Teff (K)")
    ax2.set_ylabel("Residual")
    ax2.set_xlim(5250, 4000)

def metallicity_corrected_excesses_temperature():
    '''Plot the corrected excesses against temperature.'''
    targs = cache.apogee_splitter_with_DSEP()
    dwarfs = targs.subsample(["Dwarfs", "APOGEE MetCor Teff"])

    hr.absmag_teff_plot(
        dwarfs["TEFF"], dwarfs["Corrected K Excess"], marker=".", 
        color=bc.black, ls="")
    plt.plot([5000, 4000], [0, 0], 'k--')

def metallicity_corrected_excesses_metallicity():
    '''Plot the corrected excesses against [Fe/H]'''
    targs = cache.apogee_splitter_with_DSEP()
    dwarfs = targs.subsample(["Dwarfs", "APOGEE MetCor Teff"])

    plt.plot(
        dwarfs["FE_H"], dwarfs["Corrected K Excess"], marker=".", 
        color=bc.black, ls="")
    plt.plot([-1.0, 0.5], [0, 0], 'k--')
    hr.invert_y_axis()

def plot_solar_corrected_comparisons():
    '''Plot the solar corrected values and the trend.'''
    targs = cache.apogee_splitter_with_DSEP()
    coolsamp = targs.subsample(["Dwarfs", "APOGEE MetCor Teff"])
    
    f, ax = plt.subplots(1, 1, figsize=(12,12))
    # Want a linear characterization of the temperature.
    teff_bin_edges = np.percentile(
        coolsamp["TEFF"], np.linspace(0, 100, 5+1, endpoint=True))
    teff_bin_indices = np.digitize(
        coolsamp["TEFF"], teff_bin_edges)
    percentiles = np.zeros(len(teff_bin_edges)-1)
    med_teff = np.zeros(len(teff_bin_edges)-1)
    for ind in range(1, len(teff_bin_edges)):
        tablebin = coolsamp[teff_bin_indices == ind]
        percentiles[ind-1] = np.percentile(
            tablebin["Corrected K Solar"], 100-25)
        med_teff[ind-1] = np.mean(tablebin["TEFF"])
    cor_coeff = np.polyfit(med_teff, percentiles, 1)
    cor_poly = np.poly1d(cor_coeff)

    hr.absmag_teff_plot(
        coolsamp["TEFF"], coolsamp["Corrected K Solar"], marker="o", 
        color=bc.black, ls="", label="Solar", axis=ax)
    print(percentiles)
    ax.plot(med_teff, percentiles, marker="o", color=bc.red, ls="",
             label="Binned")
    testx = np.linspace(4000, 5100, 100, endpoint=True)
    ax.plot(testx, cor_poly(testx), color=bc.red, linestyle="-", label="Fit")
    ax.plot([5100, 4000], [0, 0], 'k--')

def plot_McQuillan_corrected_comparison():
    '''Plot the trend in McQuillan.'''
    mcq = cache.mcquillan_splitter_with_DSEP()
    nondet = cache.mcquillan_nondetections_splitter_with_DSEP()
    mcq_dwarfs = mcq.subsample(["Dwarfs", "Right MetCor Teff"])
    nondet_dwarfs = nondet.subsample(["Dwarfs", "Right MetCor Teff"])

    # Combine the McQuillan detections and nondetections.
    correction_cols = ["teff", "K Excess"]
    combotable = vstack(
        [mcq_dwarfs[correction_cols], nondet_dwarfs[correction_cols]])
    
    f, ax = plt.subplots(1, 1, figsize=(12,12))
    # Want a linear characterization of the temperature.
    teff_bin_edges = np.percentile(
        combotable["teff"], np.linspace(0, 100, 5+1, endpoint=True))
    teff_bin_indices = np.digitize(
        combotable["teff"], teff_bin_edges)
    percentiles = np.zeros(len(teff_bin_edges)-1)
    med_teff = np.zeros(len(teff_bin_edges)-1)
    for ind in range(1, len(teff_bin_edges)):
        tablebin = combotable[teff_bin_indices == ind]
        percentiles[ind-1] = np.percentile(
            tablebin["K Excess"], 100-25)
        med_teff[ind-1] = np.mean(tablebin["teff"])
    cor_coeff = np.polyfit(med_teff, percentiles, 1)
    cor_poly = np.poly1d(cor_coeff)

    hr.absmag_teff_plot(
        combotable["teff"], combotable["K Excess"], marker="o", 
        color=bc.black, ls="", label="Raw Excess", axis=ax, zorder=1)
    print(percentiles)
    ax.plot(med_teff, percentiles, marker="o", color=bc.red, ls="",
             label="Binned", zorder=2)
    testx = np.linspace(4000, 5100, 100, endpoint=True)
    ax.plot(testx, cor_poly(testx), color=bc.red, linestyle="-", label="Fit",
            zorder=3)
    ax.plot([5100, 4000], [0, 0], 'k--')


def compare_MS_dispersion():
    '''Calculate the dispersion before and after metallicity correction.'''
    targs = cache.apogee_splitter_with_DSEP()
    dwarfs = targs.subsample(["Dwarfs", "APOGEE MetCor Teff"])

    single_ind = (dwarfs["Corrected K Excess"] > np.percentile(
        dwarfs["Corrected K Excess"], 35))

    print("Metallicity-corrected dispersion: {0:.3f}".format(
        median_absolute_deviation(dwarfs["Corrected K Excess"][single_ind])))

    print("Metallicity-ignorant dispersion: {0:.3f}".format(
        median_absolute_deviation(dwarfs["Corrected K Solar"][single_ind])))

    fig, ax = plt.subplots(1, 1)
    hr.absmag_teff_plot(
        dwarfs["TEFF"][single_ind], dwarfs["Corrected K Excess"][single_ind],
        marker=".", color=bc.black, ls="", label="Dwarfs", axes=ax)
    hr.absmag_teff_plot(
        dwarfs["TEFF"][~single_ind], dwarfs["Corrected K Excess"][~single_ind],
        marker=".", color=bc.red, ls="", label="Binaries", axes=ax)
    ax.set_ylim(1.2, -1.2)
    ax.set_title("Metallicity Corrected")

    fig, ax = plt.subplots(1, 1)
    plt.plot(
        dwarfs["FE_H"][single_ind], dwarfs["Corrected K Solar"][single_ind],
        marker=".", color=bc.black, ls="", label="Dwarfs", axes=ax)
    plt.plot(
        dwarfs["FE_H"][~single_ind], dwarfs["Corrected K Solar"][~single_ind],
        marker=".", color=bc.red, ls="", label="Binaries", axes=ax)
#   hr.absmag_teff_plot(
#       dwarfs["TEFF"][single_ind], dwarfs["Corrected K Solar"][single_ind],
#       marker=".", color=bc.black, ls="", label="Dwarfs", axes=ax)
#   hr.absmag_teff_plot(
#       dwarfs["TEFF"][~single_ind], dwarfs["Corrected K Solar"][~single_ind],
#       marker=".", color=bc.red, ls="", label="Binaries", axes=ax)
    ax.set_ylim(1.2, -1.2)
    ax.set_title("No Metallicity")

@write_plot("excess_hist")
def collapsed_met_histogram():
    '''Plot the distribution of K Excesses in the cool, unevolved sample.'''
    targs = cache.apogee_splitter_with_DSEP()
    cooldwarfs = targs.subsample(["Dwarfs", "Cool Noev"])

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    arraylist, bins, patches = ax.hist(
        [cooldwarfs["Corrected K Excess"], cooldwarfs["Corrected K Solar"]], 
         bins=80, color=[bc.blue, bc.red], alpha=0.5, range=(-1.6, 1.1),
        label=["[Fe/H] Corrected", "[Fe/H] = 0.08"],  histtype="bar")
    metarray = arraylist[0]
    nometarray = arraylist[1]
    singlemodel = Gaussian1D(50, 0, 0.1, bounds={"mean": (-1.6, 1.1)})
    binarymodel = Gaussian1D(10, -0.75, 0.1, bounds={"mean": (-1.6, 1.1)})
    floormodel = Const1D(3, bounds={"amplitude": (0, 100)})
    dualmodel = singlemodel+binarymodel
    fitter = fitting.SLSQPLSQFitter()
    fittedmet = fitter(dualmodel, (bins[1:]+bins[:-1])/2, metarray)
    inputexcesses = np.linspace(-1.6, 1.1, 200)
    metmodel = fittedmet(inputexcesses)
    fittednomet = fitter(
        dualmodel, (bins[1:]+bins[:-1])/2, nometarray)
    nometmodel = fittednomet(inputexcesses)
    ax.plot(inputexcesses, metmodel, color=bc.blue, ls="-", lw=3, marker="")
    ax.plot(inputexcesses, nometmodel, color=bc.red, ls="-", lw=3, marker="")
    print("Width w/ metallicity: {0:.03f}".format(fittedmet.stddev_0.value))
    print("Width w/o metallicity: {0:.03f}".format(fittednomet.stddev_0.value))
    ax.set_xlabel("K Excess Distribution")
    ax.set_ylabel("N")
    ax.legend(loc="upper left")

@write_plot("sample_dk")
def K_Excess_hr_diagram():
    '''Plot a full HR diagram in subtracted K space.'''
    targs = cache.apogee_splitter_with_DSEP()
    dwarfs = targs.subsample(["Dwarfs"])
    fullsamp = targs.subsample(["Not Dwarfs"])

    f, ax1 = plt.subplots(1, 1, figsize=(12, 12))
    hr.absmag_teff_plot(
        dwarfs["TEFF"], dwarfs["K Excess"], 
        marker=".", color=bc.red, ls="", 
        label="MS + Binaries", axis=ax1, zorder=1)
    hr.absmag_teff_plot(
        fullsamp["TEFF"], fullsamp["K Excess"], 
        marker=".", color=bc.black, ls="", label="Full Sample", axis=ax1, 
        zorder=2)
    # A representative error bar.
    hr.absmag_teff_plot(
        3700, 1.4, yerr=[
            [np.median(dwarfs["K Excess Error Down"])], 
            [np.median(dwarfs["K Excess Error Up"])]], 
        xerr=np.median(dwarfs["TEFF_ERR"]), color=bc.red, ls="", label="", 
        axis=ax1)
    minorLocator = AutoMinorLocator()
    ax1.yaxis.set_minor_locator(minorLocator)
    ax1.plot([6600, 3000], [0, 0], 'k-')
    ax1.plot([6600, 3000], [-1.3, -1.3], 'k--')
    ax1.set_xlim(6600, 3500)
    ax1.set_ylim(2, -4)
    ax1.legend(loc="upper left")
    ax1.set_xlabel("APOGEE Teff (K)")
    ax1.set_ylabel("K Excess")

@write_plot("ages")
def age_isochrones():
    '''Plot age isochrones on the APOGEE sample.'''
    targs = cache.apogee_splitter_with_DSEP()
    dwarfs = targs.subsample(["Dwarfs"])

    f, ax2 = plt.subplots(1, 1, figsize=(12, 12))
    hr.absmag_teff_plot(
        dwarfs["TEFF"], dwarfs["K Excess"], marker=".", color=bc.black, ls="", 
        label="APOGEE Dwarfs", axis=ax2, alpha=0.2)
    hr.absmag_teff_plot(
        [3700], [0.3], yerr=[
            [np.median(dwarfs["K Excess Error Down"])], 
            [np.median(dwarfs["K Excess Error Up"])]],
        xerr=[np.median(dwarfs["TEFF_ERR"])], marker="", color=bc.black, ls="",
        label="", axis=ax2, alpha=0.2)
    # Plot the bins.
    teff_bin_edges = np.linspace(6000, 4000, 15+1)
    teff_bin_indices = np.digitize(dwarfs["TEFF"], teff_bin_edges)
    percentiles = np.zeros(len(teff_bin_edges)-1)
    med_teff = np.zeros(len(teff_bin_edges)-1)
    for ind in range(1, len(teff_bin_edges)):
        tablebin = dwarfs[teff_bin_indices == ind]
        percentiles[ind-1] = np.percentile(
            tablebin["K Excess"], 100-25)
        med_teff[ind-1] = np.mean(tablebin["TEFF"])
    hr.absmag_teff_plot(
        med_teff, percentiles, marker="o", color=bc.algae, ls="-", 
        label="25th percentile")
    # Now include a MIST isochrone.
    teffvals = np.linspace(3500, 6500, 200)
    youngks = samp.calc_model_mag_fixed_age_feh_alpha(
        teffvals, 0.00, "Ks", age=1e9)
    oldks = samp.calc_model_mag_fixed_age_feh_alpha(
        teffvals, 0.00, "Ks", age=9e9)
    hr.absmag_teff_plot(
        teffvals, oldks-youngks, marker="", color=bc.purple, ls="-", 
        label="9 Gyr", axis=ax2)
    ax2.plot([7000, 3000], [0, 0], 'k-')
    ax2.plot([7000, 3000], [-0.75, -0.75], 'k--')
    ax2.plot([5200, 5200], [-1.5, 0.5], 'k:')
    ax2.set_xlim([6500, 3500])
    ax2.set_ylim(0.5, -1.3)
    ax2.set_xlabel("APOGEE Teff (K)")
    ax2.set_ylabel("K Excess")

@write_plot("Teff_relation")
def teff_comparison():
    targs = cache.apogee_splitter_with_DSEP()
    dwarfs = targs.subsample(["Dwarfs", "APOGEE MetCor Teff"])

    f, ax = plt.subplots(1,1, figsize=(12,12))
    comparable_dwarfs = dwarfs[~dwarfs["SDSS-Teff"].mask]
                                                      
    ax.plot(comparable_dwarfs["TEFF"], comparable_dwarfs["SDSS-Teff"], 'k.')
    ax.plot(flagged_dwarfs["TEFF"], flagged_dwarfs["SDSS-Teff"], 'rx')
    coeff = np.polyfit(comparable_dwarfs["TEFF"], comparable_dwarfs["SDSS-Teff"], 1)
    poly = np.poly1d(coeff)
    testteffs = np.linspace(4000, 5100, 200)
    testys = poly(testteffs)
    ax.plot(testteffs, testys, 'k-')
    ax.plot(testteffs, testteffs, 'k--')
    scatter = np.std(
        flagged_dwarfs["SDSS-Teff"] - poly(flagged_dwarfs["TEFF"]))
    print("The scatter in the relation is {0:4.0f} K".format(scatter))
    ax.set_xlabel("APOGEE Teff (K)")
    ax.set_ylabel("Pinsonneault Teff (K)")
    ax.set_title("Effective Temperature Comparison")

def teff_comparison_mcquillan():
    '''Compare Huber vs Pinsonneault Teffs for the full McQuillan sample.'''
    mcq = cache.mcquillan_splitter_with_DSEP()
    nomcq = cache.mcquillan_nondetections_splitter_with_DSEP()

    mcq_dwarf = mcq.subsample(["Dwarfs", "Right Statistics Teff"])
    nomcq_dwarf = nomcq.subsample(["Dwarfs", "Right Statistics Teff"])
    correction_cols = ["teff", "teff_prov", "SDSS-Teff"]
    combotable = vstack([mcq_dwarf[correction_cols],
                         nomcq_dwarf[correction_cols]])

    f, ax = plt.subplots(1,1, figsize=(12,12))
    comparable_dwarfs = combotable[~combotable["SDSS-Teff"].mask]

    provenances = ["KIC0", "PHO1"]
    labels = ["Brown et al (2010)", "Pinsonneault et al (2012)"]
    colors = [bc.green, bc.violet]
                                                      
    oppo_array = np.ones(len(comparable_dwarfs), dtype=np.int)
    for prov, lab, c in zip(provenances, labels, colors):
        prov_indicator = comparable_dwarfs["teff_prov"] == prov
        ax.plot(comparable_dwarfs["teff"][prov_indicator],
                comparable_dwarfs["SDSS-Teff"][prov_indicator], color=c,
                label=lab, marker=".", ls="")
        oppo_array[prov_indicator] = 0
        print("{0}: {1:d}".format(prov, np.count_nonzero(prov_indicator)))
    ax.plot(comparable_dwarfs["teff"][oppo_array],
            comparable_dwarfs["SDSS-Teff"][oppo_array], color=bc.black,
            label="Others", marker=".", ls="", ms=7)
    print("{0}: {1:d}".format("Others", np.count_nonzero(oppo_array)))
    print(comparable_dwarfs["SDSS-Teff"][oppo_array])

    coeff = np.polyfit(comparable_dwarfs["teff"], comparable_dwarfs["SDSS-Teff"], 1)
    poly = np.poly1d(coeff)
    testteffs = np.linspace(4000, 5100, 200)
    testys = poly(testteffs)
    ax.plot(testteffs, testys, 'k-')
    ax.plot(testteffs, testteffs, 'k--')
    scatter = np.std(
        comparable_dwarfs["teff"] - poly(comparable_dwarfs["SDSS-Teff"]))
    print("The relationship is y = {0:.1f} x + {1:.1f}".format(
        coeff[0], coeff[1]))
    print("The scatter in the relation is {0:4.0f} K".format(scatter))
    ax.legend()
    ax.set_xlabel("Huber Teff (K)")
    ax.set_ylabel("Pinsonneault Teff (K)")
    ax.set_title("Effective Temperature Comparison")

def compare_photometric_spectroscopic_temperatures():
    '''Plot two-panels to compare photometric to spectroscopic temps.'''
    targs = cache.apogee_splitter_with_DSEP()
    apocool = targs.subsample(["Dwarfs", "APOGEE MetCor Teff"])
    hubercool = targs.subsample(["Dwarfs", "Huber MetCor Teff"])
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    comparable_dwarfs = apocool[
        np.logical_or(apocool["teff_prov"] == "PHO54", 
        np.logical_or(apocool["teff_prov"] == "KIC0",
        np.logical_or(apocool["teff_prov"] == "PHO2", 
                      apocool["teff_prov"] == "PHO1")))]

    # Make the same plot as before with the spectroscopic temperatures.
    teff_bin_edges = np.percentile(
        comparable_dwarfs["TEFF"], np.linspace(0, 100, 5+1, endpoint=True))
    teff_bin_indices = np.digitize(
        comparable_dwarfs["TEFF"], teff_bin_edges)
    percentiles = np.zeros(len(teff_bin_edges)-1)
    med_teff = np.zeros(len(teff_bin_edges)-1)
    for ind in range(1, len(teff_bin_edges)):
        tablebin = comparable_dwarfs[teff_bin_indices == ind]
        percentiles[ind-1] = np.percentile(
            tablebin["Corrected K Solar"], 100-25)
        med_teff[ind-1] = np.mean(tablebin["TEFF"])
    cor_coeff = np.polyfit(med_teff, percentiles, 1)
    cor_poly = np.poly1d(cor_coeff)

    hr.absmag_teff_plot(
        comparable_dwarfs["TEFF"], comparable_dwarfs["Corrected K Solar"], 
        marker="o", color=bc.black, ls="", label="Solar", axis=ax1)
    ax1.plot(med_teff, percentiles, marker="o", color=bc.red, ls="",
             label="Binned")
    testx = np.linspace(4000, 5100, 100, endpoint=True)
    ax1.plot(testx, cor_poly(testx), color=bc.red, linestyle="-", label="Fit")
    ax1.plot([5100, 4000], [0, 0], 'k--')
    ax1.set_xlabel("APOGEE Teff (K)")
    ax1.set_ylabel("Corrected K Excess")
    ax1.set_ylim(0.5, -1.5)

    # Now make the previous plot with the photometric temperatures.
    comparable_dwarfs = hubercool[
        np.logical_or(hubercool["teff_prov"] == "PHO54", 
        np.logical_or(hubercool["teff_prov"] == "KIC0",
        np.logical_or(hubercool["teff_prov"] == "PHO2", 
                      hubercool["teff_prov"] == "PHO1")))]
    teff_bin_edges = np.percentile(
        comparable_dwarfs["teff"], np.linspace(0, 100, 5+1, endpoint=True))
    teff_bin_indices = np.digitize(
        comparable_dwarfs["teff"], teff_bin_edges)
    percentiles = np.zeros(len(teff_bin_edges)-1)
    med_teff = np.zeros(len(teff_bin_edges)-1)
    for ind in range(1, len(teff_bin_edges)):
        tablebin = comparable_dwarfs[teff_bin_indices == ind]
        percentiles[ind-1] = np.percentile(
            tablebin["Corrected Phot Teff K Solar"], 100-25)
        med_teff[ind-1] = np.mean(tablebin["teff"])
    cor_coeff = np.polyfit(med_teff, percentiles, 1)
    cor_poly = np.poly1d(cor_coeff)

    hr.absmag_teff_plot(
        comparable_dwarfs["teff"], 
        comparable_dwarfs["Corrected Phot Teff K Solar"], marker="o", 
        color=bc.black, ls="", label="Solar", axis=ax2)
    ax2.plot(med_teff, percentiles, marker="o", color=bc.red, ls="",
             label="Binned")
    testx = np.linspace(4000, 5100, 100, endpoint=True)
    ax2.plot(testx, cor_poly(testx), color=bc.red, linestyle="-", label="Fit")
    ax2.plot([5100, 4000], [0, 0], 'k--')
    ax2.set_xlabel("Huber Teff (K)")
    ax2.set_ylabel("")
    ax2.set_ylim(0.5, -1.5)

def compare_McQuillan_to_nondetections():
    '''Compare binarity of the McQuillan period detections to nondetections.'''
    periods = cache.mcquillan_corrected_splitter()
    nondets = cache.mcquillan_nondetections_corrected_splitter()

    periods_stats = periods.subsample(["Dwarfs", "Right Statistics Teff"])
    nondets_stats = nondets.subsample(["Dwarfs", "Right Statistics Teff"])

    arraylist, bins, patches = plt.hist(
        [periods_stats["Corrected K Excess"], 
         nondets_stats["Corrected K Excess"]], bins=80, color=[bc.blue, bc.red], 
        alpha=0.5, range=(-1.6, 1.1), normed=True,
        label=["Period Detections", "Nondetections"],  histtype="step", 
        cumulative=True)
    sig = scipy.stats.anderson_ksamp([
        periods_stats["Corrected K Excess"], 
        nondets_stats["Corrected K Excess"]])
    print(sig)
    plt.xlabel("Metallicity-corrected K Excess")
    plt.ylabel("N (< K) / N")
    plt.legend(loc="upper left")

@write_plot("ElBadry_Excess")
def el_badry_excess():
    '''Plot the distribution of K Excesses using the El-Badry temperatures.'''
    targs = cache.apogee_splitter_with_DSEP()
    cooldwarfs = targs.subsample(["Dwarfs", "ElBadry Statistics Teff"])

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    singleinds = cooldwarfs["Binarity"] == "Single"
    singles = cooldwarfs[singleinds]
    hr.absmag_teff_plot(
        singles["TEFF"], singles["Corrected K Excess"], 
        marker=".", color=bc.black, ls="", axis=ax, label="Single Stars", zorder=2)
    multiples = cooldwarfs[~singleinds]
    hr.absmag_teff_plot(
        multiples["T_eff [K]"], multiples["Corrected ElBadry K Excess"], 
        marker="*", color=bc.red, ls="", axis=ax, label="El-Badry", ms=7)
    hr.absmag_teff_plot(
        multiples["TEFF"], multiples["Corrected K Excess"], 
        marker="*", color=bc.black, ls="", axis=ax, label="ASPCAP", ms=7)
    for (apoteff, apoex, elbteff, elbex) in zip(
            multiples["TEFF"], multiples["Corrected K Excess"], 
            multiples["T_eff [K]"], multiples["Corrected ElBadry K Excess"]):
        ax.arrow(apoteff, apoex, (elbteff-apoteff), (elbex-apoex))
    ax.set_xlabel("Teff (K)")
    ax.set_ylabel("K Excess")
    ax.legend(loc="lower right")

@write_plot("apogee_rapid_excess")
def rapid_rotator_bins():
    '''Plot the distribution of K Excesses for different bins of rotation.'''
    targs = cache.apogee_splitter_with_DSEP()
    cooldwarfs = targs.subsample(["Dwarfs", "APOGEE Statistics Teff"])
    mcq = catin.read_McQuillan_catalog()
    ebs = catin.read_villanova_EBs()
    
    mcq_cooldwarfs = au.join_by_id(cooldwarfs, mcq, "kepid", "KIC")
    eb_cooldwarfs = au.join_by_id(cooldwarfs, ebs, "kepid", "KIC")
    periodbins = np.flipud(np.array([1, 5, 13]))
    f, axes = plt.subplots(
        1, len(periodbins), figsize=(12*len(periodbins), 12), sharex=True, 
        sharey=True)
    mcq_period_indices = np.digitize(mcq_cooldwarfs["Prot"], periodbins)
    eb_period_indices = np.digitize(eb_cooldwarfs["period"], periodbins)
    titles = ["{0:d} day < Prot <= {1:d} day".format(p2, p1) for (p1, p2) in
              zip(periodbins[:-1], periodbins[1:])]
    titles.insert(0, "Prot > {0:d} day".format(periodbins[0]))
    for i, (title, ax) in enumerate(zip(titles, axes)):
        mcq_periodbin = mcq_cooldwarfs[mcq_period_indices == i]
        eb_periodbin = eb_cooldwarfs[eb_period_indices == i]
        hr.absmag_teff_plot(
            mcq_periodbin["TEFF"], mcq_periodbin["Corrected K Excess"], 
            marker=".", color=bc.black, ls="", axis=ax, label="Period in Bin")
        hr.absmag_teff_plot(
            eb_periodbin["TEFF"], eb_periodbin["Corrected K Excess"],
            marker="*", color=bc.pink, ls="", ms=18, axis=ax, label="EB")

        ax.set_ylabel("")
        ax.set_xlabel("APOGEE Teff (K)")
        ax.set_title(title)
        ax.plot(
            [4000, 5250], [-0.3, -0.3], marker="", ls="--", color=bc.purple, 
            lw=3)
        ax.plot(
            [4000, 5250], [-0.2, -0.2], marker="", ls="--", color=bc.algae, 
            lw=3)
        ax.plot([4000, 5250], [-0.0, -0.0], 'k-', lw=2)
#       plt.setp(ax.get_yticklabels(), visible=False)
        axes[0].legend(loc="upper right")
    axes[0].set_ylabel("Metallicity-Corrected K Excess")
    axes[0].set_xlim(5250, 4000)
    axes[0].set_ylim(0.3, -1.25)

def plot_teff_prov():
    '''Plot the spectroscopic Teff vs photometric teffs.'''
    clean = cache.clean_apogee_splitter()
    # Continue this to show how the plots differ depending on the provenance
    # that's used.
    clean.data["Spec K"] = samp.calc_model_mag_fixed_age_alpha(
        clean.data["TEFF"], 0.03, "Ks", age=1e9, model="MIST")
    clean.data["Phot K"] = samp.calc_model_mag_fixed_age_alpha(
        clean.data["teff"], 0.03, "Ks", age=1e9, model="MIST")


@write_plot("full_mcquillan_rr_excess")
def mcquillan_rapid_rotator_bins():
    '''Plot the rapid rotator bins in the full McQuillan sample.'''
    mcq = cache.mcquillan_corrected_splitter()
    ebs = cache.eb_splitter_with_DSEP()
    dwarfs = mcq.subsample(["Dwarfs", "Right Statistics Teff"])
    eb_dwarfs = ebs.subsample(["Dwarfs", "Right Statistics Teff"])

    periodbins = np.flipud(np.array([1, 5, 13]))
    f, axes = plt.subplots(
        1, len(periodbins), figsize=(12*len(periodbins), 12), sharex=True, 
        sharey=True)
    mcq_period_indices = np.digitize(dwarfs["Prot"], periodbins)
    eb_period_indices = np.digitize(eb_dwarfs["period"], periodbins)
    titles = ["{0:d} day < Prot <= {1:d} day".format(p2, p1) for (p1, p2) in
              zip(periodbins[:-1], periodbins[1:])]
    titles.insert(0, "Prot > {0:d} day".format(periodbins[0]))
    for i, (title, ax) in enumerate(zip(titles, axes)):
        mcq_periodbin = dwarfs[mcq_period_indices == i]
        eb_periodbin = eb_dwarfs[eb_period_indices == i]
        hr.absmag_teff_plot(
            mcq_periodbin["SDSS-Teff"], mcq_periodbin["Corrected K Excess"], 
            marker=".", color=bc.black, ls="", axis=ax, zorder=1)
        hr.absmag_teff_plot(
            eb_periodbin["SDSS-Teff"], eb_periodbin["Corrected K Excess"], 
            marker="*", color=bc.pink, ls="", ms=18, axis=ax, zorder=2)
        ax.set_ylabel("")
        ax.set_xlabel("Pinsonneault et al (2012) Teff")
        ax.set_title(title)
        ax.plot(
            [4000, 5250], [-0.3, -0.3], marker="", ls="--", color=bc.purple, 
            lw=3, zorder=3)
        ax.plot(
            [4000, 5250], [-0.2, -0.2], marker="", ls="--", color=bc.algae, 
            lw=3, zorder=3)
        ax.plot([3500, 6500], [-0.0, -0.0], 'k-', lw=2, zorder=4)
    axes[0].set_ylabel("K Excess")
    axes[0].set_xlim(5250, 4000)
    axes[0].set_ylim(0.3, -1.25)

@write_plot("mcquillan_transition")
def rapid_rotator_transition():
    '''Make bins in the transition region of the McQuillan rapid rotators.'''
    mcq = cache.mcquillan_corrected_splitter()
    ebs = cache.eb_splitter_with_DSEP()
    dwarfs = mcq.subsample(["Dwarfs", "Right Statistics Teff"])
    eb_dwarfs = ebs.subsample(["Dwarfs", "Right Statistics Teff"])

    periodbins = np.array([5, 7, 9, 11, 13])
    f, axes = plt.subplots(
        2, 2, figsize=(24,24), sharex=True, 
        sharey=True)
    mcq_period_indices = np.digitize(dwarfs["Prot"], periodbins)
    eb_period_indices = np.digitize(eb_dwarfs["period"], periodbins)
    titles = ["{0:0d} day < Prot <= {1:0d} day".format(p1, p2) for (p1, p2) in
              zip(periodbins[:-1], periodbins[1:])]
    for i, (title, ax) in enumerate(zip(titles, np.ravel(axes))):
        mcq_periodbin = dwarfs[mcq_period_indices == i+1]
        eb_periodbin = eb_dwarfs[eb_period_indices == i+1]
        hr.absmag_teff_plot(
            mcq_periodbin["SDSS-Teff"], mcq_periodbin["Corrected K Excess"], 
            marker=".", color=bc.black, ls="", axis=ax, zorder=1)
        hr.absmag_teff_plot(
            eb_periodbin["SDSS-Teff"], eb_periodbin["Corrected K Excess"], 
            marker="*", color=bc.pink, ls="", ms=18, axis=ax, zorder=2)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title(title)
        ax.plot(
            [4000, 5250], [-0.3, -0.3], marker="", ls="--", color=bc.purple, 
            lw=3, zorder=3)
        ax.plot(
            [4000, 5250], [-0.2, -0.2], marker="", ls="--", color=bc.algae, 
            lw=3, zorder=3)
        ax.plot([3500, 6500], [-0.0, -0.0], 'k-', lw=2, zorder=4)
    axes[1][0].set_xlabel("Pinsonneault et al (2012) Teff (K)")
    axes[1][1].set_xlabel("Pinsonneault et al (2012) Teff (K)")
    axes[0][0].set_ylabel("Corrected K Excess")
    axes[1][0].set_ylabel("Corrected K Excess")
    ax.set_xlim(5250, 4000)
    ax.set_ylim(0.3, -1.25)

def calculate_APOGEE_binary_significance(min_P, max_P):
    '''Calculates the significance of binarity for APOGEE binaries.'''
    targs = cache.apogee_splitter_with_DSEP()
    dwarfs = targs.subsample(["Dwarfs", "APOGEE Statistics Teff"])
    mcq = catin.read_McQuillan_catalog()
    nomcq = catin.read_McQuillan_nondetections()
    combocols = ["Corrected K Excess"]
    dwarfmcq = au.join_by_id(dwarfs, mcq, "kepid", "KIC")
    dwarfnomcq = au.join_by_id(dwarfs, nomcq, "kepid", "KIC")
    fulldwarfs = vstack([dwarfmcq[combocols], dwarfnomcq[combocols]])
    ebs = catin.read_villanova_EBs()
    dwarfebs = au.join_by_id(dwarfs, ebs, "kepid", "KIC")
    # We only want detached systems
    dwarfebs = dwarfebs[dwarfebs["morph"] < 0.55]

    # The fraction of binaries in the full sample.
    full_binaryfrac = (
        (np.count_nonzero(fulldwarfs["Corrected K Excess"] < -0.3) +
         np.count_nonzero(dwarfebs["Corrected K Excess"] < -0.3)) / 
        (len(fulldwarfs) + len(dwarfebs)))
    # Index rapid rotators in the period bin.
    rapidrot = np.logical_and(
        dwarfmcq["Prot"] > min_P, dwarfmcq["Prot"] < max_P)
    # Index EBs in the period bin.
    rapidebs = np.logical_and(
        dwarfebs["period"] > min_P, dwarfebs["period"] < max_P)
    # Number of rapid rotators showing binary excesses
    rapidbins = np.count_nonzero(
        dwarfmcq["Corrected K Excess"][rapidrot] < -0.3)
    # Number of ebs in the period bin showing binary excesses
    rapidebbins = np.count_nonzero(
        dwarfebs["Corrected K Excess"][rapidebs] < -0.3)
    fullrapid = np.count_nonzero(rapidrot) + np.count_nonzero(rapidebs)
    rapidsig = scipy.stats.binom_test(
        rapidbins+rapidebbins, fullrapid, full_binaryfrac, alternative="greater")
    print("Number of rapid binaries: {0:d}".format(rapidbins+rapidebbins))
    print("Number of rapid rotators: {0:d}".format(fullrapid))
    print("Full Binary Fraction: {0:.2f}".format(full_binaryfrac))
    print(rapidsig)

def calculate_McQuillan_binary_significance(min_P, max_P):
    '''Calculates the significance of binary for McQuillan binaries.'''
    mcq = cache.mcquillan_corrected_splitter()
    nomcq = cache.mcquillan_nondetections_corrected_splitter()
    ebs = cache.eb_splitter_with_DSEP()
    mcq_dwarfs = mcq.subsample(["Dwarfs", "Right Statistics Teff"])
    nomcq_dwarfs = nomcq.subsample(["Dwarfs", "Right Statistics Teff"])
    generic_columns = ["Corrected K Excess"]
    dwarfs = vstack([
        mcq_dwarfs[generic_columns], nomcq_dwarfs[generic_columns]])
    dwarf_ebs = ebs.subsample(["Dwarfs", "Right Statistics Teff"])

    # This is the fraction of binaries in the whole sample.
    total_binaries = (
        np.count_nonzero(dwarfs["Corrected K Excess"] < -0.3) + 
        np.count_nonzero(dwarf_ebs["Corrected K Excess"] < -0.3))
    full_binaryfrac = total_binaries / (len(dwarfs) + len(dwarf_ebs))
    # Number of rapid rotators.
    rapidrot = np.logical_and(
        mcq_dwarfs["Prot"] > min_P, mcq_dwarfs["Prot"] < max_P)
    # Number of EBs that are tidally-synchronized.
    rapidebs = np.logical_and(
        dwarf_ebs["period"] > min_P, dwarf_ebs["period"] < max_P)
    # Total Number of rapidly rotating binaries.
    rapidbins = np.count_nonzero(
        mcq_dwarfs["Corrected K Excess"][rapidrot] < -0.3)
    # Total number of close EBs
    rapidebbins = np.count_nonzero(
        dwarf_ebs["Corrected K Excess"][rapidebs] < -0.3)
    # Total number of rapid rotators
    fullrapid = np.count_nonzero(rapidrot) + np.count_nonzero(rapidebs)
    rapidsig = scipy.stats.binom_test(
        rapidbins+rapidebbins, fullrapid, full_binaryfrac,
        alternative="greater")
    print("Number of rapid binaries: {0:d}".format(rapidbins+rapidebbins))
    print("Number of rapid rotators: {0:d}".format(fullrapid))
    print("Full binary fraction: {0:.2f}".format(full_binaryfrac))
    print("Total sample: {0:.2f}".format(len(dwarfs)+len(dwarf_ebs)))
    print(rapidsig)

@write_plot("eclipseprob")
def verify_eb_rapid_rotator_rate():
    '''Compare the rate of EBs to the rate of rapid rotators.'''
    mcq = cache.mcquillan_corrected_splitter()
    nomcq = cache.mcquillan_nondetections_corrected_splitter()
    eb_split = cache.eb_splitter_with_DSEP()
    mcq_dwarfs = mcq.subsample(["Dwarfs", "Right Statistics Teff"])
    nomcq_dwarfs = nomcq.subsample(["Dwarfs", "Right Statistics Teff"])
    generic_columns = ["kepid", "Corrected K Excess"]
    dwarfs = vstack([
        mcq_dwarfs[generic_columns], nomcq_dwarfs[generic_columns]])
    dwarf_ebs = eb_split.subsample(["Dwarfs", "Right Statistics Teff"])
    # We only want detached systems
    dwarf_ebs = dwarf_ebs[dwarf_ebs["period"] > 1]

    # Check the intersection between the two samples.
    dwarfs = au.filter_column_from_subtable(
        dwarfs, "kepid", dwarf_ebs["kepid"])

    f, ax = plt.subplots(1, 1, figsize=(12,12))
    # Now bin the EBs
    period_bins, dp = np.linspace(1, 12, 11+1, retstep=True)
    period_bin_centers = np.sqrt(period_bins[1:] * period_bins[:-1])
    eb_hist, _ = np.histogram(dwarf_ebs["period"], bins=period_bins)
    totalobjs = len(dwarfs) + len(dwarf_ebs)
    normalized_ebs = eb_hist / totalobjs
    eb_upperlim = (au.poisson_upper(eb_hist, 1) - eb_hist) / totalobjs
    eb_lowerlim = (eb_hist - au.poisson_lower(eb_hist, 1)) / totalobjs
    ax.step(period_bins, np.append(normalized_ebs, [0]), where="post", 
            color=bc.red, ls="-", label="EBs", alpha=0.5)
    ax.set_xscale("linear")

    # Bin the rapid rotators
    rapid_hist, _ = np.histogram(mcq_dwarfs["Prot"], bins=period_bins)
    normalized_rapid = rapid_hist / totalobjs 
    rapid_upperlim = (au.poisson_upper(rapid_hist, 1) - rapid_hist) / totalobjs
    rapid_lowerlim = (rapid_hist - au.poisson_lower(rapid_hist, 1)) / totalobjs
    ax.step(period_bins, np.append(normalized_rapid, [0]), where="post", color=bc.blue,
            ls="-", label="Rapid rotators")
    ax.errorbar(period_bin_centers, normalized_rapid, 
            yerr=[rapid_lowerlim, rapid_upperlim], marker="", ls="", 
            color=bc.blue, capsize=5)

    short_ebs = dwarf_ebs[dwarf_ebs["period"] < 100]
    geo_spline = ebs.read_Kirk_geometric_correction_spline()
    # To translate from eb fraction to rapid fraction.
    geo_factors = geo_spline(np.log10(short_ebs["period"].filled()))
    correction_factor = (np.maximum(0, np.sqrt(3)/2 -  geo_factors) / 
                         geo_factors)
    pred_hist, _ = np.histogram(
        short_ebs["period"], bins=period_bins, weights=correction_factor)
    normalized_pred = pred_hist / totalobjs
    scale_factor = np.where(
        normalized_ebs, normalized_pred / normalized_ebs, 0)
    pred_upperlim =  eb_upperlim * np.where(
        scale_factor, scale_factor, period_bin_centers)
    pred_lowerlim = eb_lowerlim * np.where(
        scale_factor, scale_factor, period_bin_centers)
    ax.step(period_bins, np.append(normalized_pred, [0]), where="post", 
            color=bc.red, linestyle=":", 
            label="Predicted Rapid Rotators from EBs")
    ax.errorbar(period_bin_centers, normalized_pred, 
            yerr=[pred_lowerlim, pred_upperlim], marker="", ls="", 
            color=bc.red, capsize=5)
    ax.set_xlabel("Period (day)")
    ax.set_ylabel("# in period bin / Full Teff Sample")
    ax.legend(loc="upper left")

@write_plot("vsini_check")
def vsini_check():
    '''Plot showing that photometric rapid rotators have high vsini.'''
    targs = cache.apogee_splitter_with_DSEP()
    cooldwarfs = targs.subsample(["Dwarfs", "APOGEE Statistics Teff"])
    mcq = catin.read_McQuillan_catalog()
    ebs = catin.read_villanova_EBs()
    
    mcq_cooldwarfs = au.join_by_id(cooldwarfs, mcq, "kepid", "KIC")
    eb_cooldwarfs = au.join_by_id(cooldwarfs, ebs, "kepid", "KIC")
    rapid = np.logical_and(
        mcq_cooldwarfs["Prot"] < 5, mcq_cooldwarfs["Prot"] > 1)
    veryrapid = mcq_cooldwarfs["Prot"] < 1
    slow = np.logical_and(~rapid, ~veryrapid)
    print(np.count_nonzero(veryrapid))

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.plot(mcq_cooldwarfs[slow]["VSINI"], 
            mcq_cooldwarfs[slow]["Corrected K Excess"], color=bc.black, 
            marker=".", ls="", label="P > 5 day")
    ax.plot(mcq_cooldwarfs[rapid]["VSINI"], 
            mcq_cooldwarfs[rapid]["Corrected K Excess"], color=bc.violet,
            marker="o", ls="", ms=8, label="1 day < P < 5 day")
    ax.plot(mcq_cooldwarfs[veryrapid]["VSINI"], 
            mcq_cooldwarfs[veryrapid]["Corrected K Excess"], color=bc.blue, 
            marker=".", ls="", label="P < 1 day")
    ax.set_xlabel(r"$v \sin i$")
    ax.set_ylabel("Corrected K Excess")
    ax.legend(loc="lower right")
    hr.invert_y_axis(ax)

@write_plot("binary_fraction")
def binary_fractions_with_period():
    '''Measure the binary fraction with period.'''
    mcq = cache.mcquillan_corrected_splitter()
    nomcq = cache.mcquillan_nondetections_corrected_splitter()
    ebs = cache.eb_splitter_with_DSEP()
    mcq_dwarfs = mcq.subsample(["Dwarfs", "Right Statistics Teff"])
    nomcq_dwarfs = nomcq.subsample(["Dwarfs", "Right Statistics Teff"])
    generic_columns = ["Corrected K Excess"]
    dwarfs = vstack([
        mcq_dwarfs[generic_columns], nomcq_dwarfs[generic_columns]])
    dwarf_ebs = ebs.subsample(["Dwarfs", "Right Statistics Teff"])

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    # Create the histograms
    period_bins = np.arange(1, 51, 2)
    # These are for the inclusive sample.
    singles02 =  mcq_dwarfs["Prot"][mcq_dwarfs["Corrected K Excess"] >= -0.2]
    binaries02 =  mcq_dwarfs["Prot"][mcq_dwarfs["Corrected K Excess"] < -0.2]
    single_ebs02 = dwarf_ebs["period"][dwarf_ebs["Corrected K Excess"] >= -0.2]
    binary_ebs02 = dwarf_ebs["period"][dwarf_ebs["Corrected K Excess"] < -0.2]
    binary_rot_hist02, _ = np.histogram(binaries02, bins=period_bins)
    binary_eb_hist02, _ = np.histogram(binary_ebs02, bins=period_bins)
    single_rot_hist02, _ = np.histogram(singles02, bins=period_bins)
    single_eb_hist02, _ = np.histogram(single_ebs02, bins=period_bins)
    # These are for the conservative sample.
    singles03 =  mcq_dwarfs["Prot"][mcq_dwarfs["Corrected K Excess"] >= -0.3]
    binaries03 =  mcq_dwarfs["Prot"][mcq_dwarfs["Corrected K Excess"] < -0.3]
    single_ebs03 = dwarf_ebs["period"][dwarf_ebs["Corrected K Excess"] >= -0.3]
    binary_ebs03 = dwarf_ebs["period"][dwarf_ebs["Corrected K Excess"] < -0.3]
    binary_rot_hist03, _ = np.histogram(binaries03, bins=period_bins)
    binary_eb_hist03, _ = np.histogram(binary_ebs03, bins=period_bins)
    single_rot_hist03, _ = np.histogram(singles03, bins=period_bins)
    single_eb_hist03, _ = np.histogram(single_ebs03, bins=period_bins)
    full_rot_hist, _ = np.histogram(mcq_dwarfs["Prot"], bins=period_bins)
    full_eb_hist, _ = np.histogram(dwarf_ebs["period"], bins=period_bins)

    # The binary fraction for the full sample
    fullsamp_frac02 = (
        (np.count_nonzero(mcq_dwarfs["Corrected K Excess"] < -0.2) + 
         np.count_nonzero(dwarf_ebs["Corrected K Excess"] < -0.2) + 
         np.count_nonzero(nomcq_dwarfs["Corrected K Excess"] < -0.2)) /
        (len(mcq_dwarfs) + len(dwarf_ebs) + len(nomcq_dwarfs)))
    fullsamp_frac03 = (
        (np.count_nonzero(mcq_dwarfs["Corrected K Excess"] < -0.3) + 
         np.count_nonzero(dwarf_ebs["Corrected K Excess"] < -0.3) + 
         np.count_nonzero(nomcq_dwarfs["Corrected K Excess"] < -0.3)) /
        (len(mcq_dwarfs) + len(dwarf_ebs) + len(nomcq_dwarfs)))


    # Sum up the binaries, single, and total histograms
    total_binaries02 = binary_rot_hist02 + binary_eb_hist02
    total_singles02 = single_rot_hist02 + single_eb_hist02
    total_binaries03 = binary_rot_hist03 + binary_eb_hist03
    total_singles03 = single_rot_hist03 + single_eb_hist03
    total = full_rot_hist + full_eb_hist

    # Measure the binary fraction
    frac02 = total_binaries02 / total
    frac_uppers02 = au.binomial_upper(total_binaries02, total) - frac02
    frac_lowers02 = frac02 - au.binomial_lower(total_binaries02, total)
    frac03 = total_binaries03 / total
    frac_uppers03 = au.binomial_upper(total_binaries03, total) - frac03
    frac_lowers03 = frac03 - au.binomial_lower(total_binaries03, total)

    period_mids = (period_bins[1:] + period_bins[:-1])/2
    ax.errorbar(
        period_mids, frac02, yerr=[frac_lowers02, frac_uppers02],
        color=bc.algae, linestyle="", marker="o", 
        label=r"$\Delta K < -0.2$ mag")
    ax.errorbar(
        period_mids, frac03, yerr=[frac_lowers03, frac_uppers03],
        color=bc.purple, linestyle="", marker="o", 
        label=r"$\Delta K < -0.3$ mag")
    ax.plot([1, 20], [fullsamp_frac02, fullsamp_frac02], ls="--", marker="",
            color=bc.algae)
    ax.plot([1, 20], [fullsamp_frac03, fullsamp_frac03], ls="--", marker="",
            color=bc.purple)
    ax.set_xlabel("Period (day)")
    ax.set_ylabel("Photometric Binary Fraction")
    ax.set_xlim(1, 20)
    ax.legend(loc="upper right")
    
@write_plot("tsb_analysis")
def tsb_distribution():
    '''Try to measure the tidally-synchronized binary distribution.
    
    This is done by assuming that full sample is made up of two populations
    with different binary fractions: a single-star evolution sequence with the
    binary fraction of the full sample, and a tidally-synchronized sequence
    with the binary fraction of '''
    mcq = cache.mcquillan_corrected_splitter()
    eb_split = cache.eb_splitter_with_DSEP()
    mcq_dwarfs = mcq.subsample(["Dwarfs", "Right Statistics Teff"])
    dwarf_ebs = eb_split.subsample(["Dwarfs", "Right Statistics Teff"])
    # We only want detached systems
    dwarf_ebs = dwarf_ebs[dwarf_ebs["period"] > 1]

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    # Define the photometric binaries and photometric singles.
    mcq_binaries = mcq_dwarfs["Corrected K Excess"] < -0.3
    eb_binaries = dwarf_ebs["Corrected K Excess"] < -0.3
    mcq_singles = mcq_dwarfs["Corrected K Excess"] >= -0.3
    eb_singles = dwarf_ebs["Corrected K Excess"] >= -0.3
    # Define the rapid and slow rotators
    mcq_rapid = np.logical_and(mcq_dwarfs["Prot"] < 5, mcq_dwarfs["Prot"] > 1)
    eb_rapid = np.logical_and(dwarf_ebs["period"] < 5, dwarf_ebs["period"] > 1)
    mcq_slow = mcq_dwarfs["Prot"] > 15
    eb_slow = dwarf_ebs["period"] > 15

    # Define the binary fractions
    num_rapid_binaries = (
        np.count_nonzero(np.logical_and(mcq_binaries, mcq_rapid)) + 
        np.count_nonzero(np.logical_and(eb_binaries, eb_rapid)))
    num_rapid_singles = (
        np.count_nonzero(np.logical_and(mcq_singles, mcq_rapid)) + 
        np.count_nonzero(np.logical_and(eb_singles, eb_rapid)))
    num_slow_binaries = (
        np.count_nonzero(np.logical_and(mcq_binaries, mcq_slow)) + 
        np.count_nonzero(np.logical_and(eb_binaries, eb_slow)))
    num_slow_singles = (
        np.count_nonzero(np.logical_and(mcq_singles, mcq_slow)) + 
        np.count_nonzero(np.logical_and(eb_singles, eb_slow)))
    f_ts = num_rapid_binaries / (num_rapid_binaries + num_rapid_singles)
    f_ss = num_slow_binaries / (num_slow_binaries + num_slow_singles)
    fts_fss = f_ts - f_ss

    periodbins, dp = np.linspace(1, 20, 19+1, endpoint=True, retstep=True)
    # The total number of observed binaries.
    mcq_binary_perioddist, _ = np.histogram(
        mcq_dwarfs["Prot"][mcq_binaries], bins=periodbins)
    eb_binary_perioddist, _ = np.histogram(
        dwarf_ebs["period"][eb_binaries], bins=periodbins)
    binary_perioddist = mcq_binary_perioddist + eb_binary_perioddist
    # The total number of observed singles
    mcq_single_perioddist, _ = np.histogram(
        mcq_dwarfs["Prot"][mcq_singles], bins=periodbins)
    eb_single_perioddist, _ = np.histogram(
        dwarf_ebs["period"][eb_singles], bins=periodbins)
    single_perioddist = mcq_single_perioddist + eb_single_perioddist

    tidsync_perioddist = -(
        (f_ss*single_perioddist - (1-f_ss)*binary_perioddist) / (fts_fss))
    singlestar_perioddist = (
        (f_ts*single_perioddist - (1-f_ts)*binary_perioddist) / (fts_fss))

    e_fts = np.sqrt(num_rapid_binaries) / (num_rapid_binaries + num_rapid_singles)
    e_fss = np.sqrt(num_slow_binaries) / (num_slow_binaries + num_slow_singles)
    tidsync_errors = np.sqrt(((1-f_ss)/fts_fss)**2*binary_perioddist + 
                      (f_ss/fts_fss)**2*single_perioddist +
                      (tidsync_perioddist/fts_fss)**2*e_fts**2 +
                      (singlestar_perioddist/fts_fss)**2*e_fss**2)
    singlestar_errors = np.sqrt(((1-f_ts)/fts_fss)**2*binary_perioddist +
                         (f_ts/fts_fss)**2*single_perioddist +
                         (tidsync_perioddist/fts_fss)**2*e_fts**2 +
                         (singlestar_perioddist/fts_fss)**2*e_fss**2)

    mcq_full_perioddist, _ = np.histogram(mcq_dwarfs["Prot"], bins=periodbins)
    eb_full_perioddist, _ = np.histogram(dwarf_ebs["period"], bins=periodbins)
    full_perioddist = mcq_full_perioddist + eb_full_perioddist

    ax.step(periodbins, np.append(tidsync_perioddist, [0]), where="post", 
            color=bc.red, ls="-", label="Synchronized")
    ax.step(periodbins, np.append(singlestar_perioddist, [0]), where="post", 
            color=bc.blue, ls="-", label="Unsynchronized")
    ax.step(periodbins, np.append(full_perioddist, [0]), where="post", 
            color=bc.black, ls="-", label="Total")
    ax.errorbar(periodbins[:-1]+dp/2, tidsync_perioddist, yerr=tidsync_errors,
                color=bc.red, ls="", marker="")
    ax.errorbar(periodbins[:-1]+dp/2, singlestar_perioddist, yerr=singlestar_errors,
                color=bc.blue, ls="", marker="")
    ax.plot([1, 20], [0, 0], 'k-')

    ax.set_xlabel("Period (day)")
    ax.set_ylabel("Population Number")
    ax.set_xlim(1, 20)
    ax.legend(loc="upper left")

@write_plot("Bruntt_comp")
def Bruntt_vsini_comparison():
    '''Create figure comparison vsini for asteroseismic targets.

    The APOGEE vsinis are compared against the Bruntt et al (2012)
    spectroscopic vsinis for a set of asteroseismic targets.'''

    f, ax = plt.subplots(1,1, figsize=(10,10))
    astero_dwarfs = catin.bruntt_dr14_overlap()
    bad_indices = astero_dwarfs["VSINI"] < 0
#   vsini_diff = ((astero_dwarfs["vsini"][~bad_indices] -
#                  astero_dwarfs["VSINI"][~bad_indices]) /
#                 astero_dwarfs["vsini"][~bad_indices])
#   vsini_baddiff = np.ones(np.count_nonzero(bad_indices))
#   plt.plot(astero_dwarfs["vsini"][~bad_indices], vsini_diff, 'ko')
#   plt.plot(astero_dwarfs["vsini"][bad_indices], vsini_baddiff, 'rv')
#   plt.plot([0, 35], [0, 0], 'k--')
#   plt.ylim([-0.2, 0.5])
    ax.loglog(astero_dwarfs["vsini"][~bad_indices], 
               astero_dwarfs["VSINI"][~bad_indices], 'ko')
    ax.loglog(astero_dwarfs["vsini"][bad_indices],
             np.zeros(np.count_nonzero(bad_indices)), 'rx')
    detection_lim = 7
    ax.loglog([1, 40], [detection_lim, detection_lim], 'k:')

    # Now to fit the data to a line.
    detected_table = astero_dwarfs[astero_dwarfs["VSINI"] >= detection_lim]
    meanx = np.mean(np.log10(detected_table["vsini"]))
    fitval, cov = np.polyfit(
        np.log10(detected_table["vsini"])-meanx, np.log10(detected_table["VSINI"]), 1, 
        cov=True)
    polyeval = np.poly1d(fitval)
    polyx = np.log10(np.linspace(1, 40, 10)) - meanx
    polyy = 10**polyeval(polyx)
    ax.loglog(10**(polyx+meanx), polyy, 'k--', linewidth=3)
    # Calculate the slope and error in the slope.
    slope = fitval[0]
    intercept = fitval[1]-meanx
    slope_error = np.sqrt(cov[0, 0])
    intercept_error = np.sqrt(cov[1, 1])
    print("The measured slope is {0:.2f} +/- {1:.2f}".format(
        slope, slope_error))
    print("The measured intercept is {0:.2f} +/- {1:.2f}".format(
        intercept, intercept_error))
    
    # Calculate the 1-sigma offset of the intercept.
    residuals = (
        np.log10(detected_table["VSINI"]) - 
        polyeval(np.log10(detected_table["vsini"])-meanx))
    var = np.sum(residuals**2) / (len(residuals) - 2 - 1)
    offset = np.sqrt(var)
#    offset = intercept_error/2
    print("Uncertainty is {0:.1f}%".format(offset*np.log(10)*100))
#   ax.fill_between(
#       10**(polyx+meanx), polyy*10**(offset/2), polyy*10**(-offset/2),
#       facecolor="gray")
#   outlier = astero_dwarfs[~bad_indices][np.abs(vsini_diff) > 1.0]
#   assert len(outlier) == 1
#   print("Ignoring KIC{0:d}: Bruntt vsini = {1:.1f}, ASPCAP vsini = {2:.1f}".format(
#       outlier["kepid"][0], outlier["vsini"][0], outlier["VSINI"][0]))
    print("Bad objects:")
    print(astero_dwarfs[["KIC", "vsini"]][bad_indices])
    ax.set_xlabel(r"Bruntt $v \sin i$ (km/s)")
    ax.set_ylabel(r"APOGEE $v \sin i$ (km/s)")
    return cov

def plot_lower_limit_Bruntt_test():
    '''Plot results of two-sample AD test for checking lower limit.'''
    astero_dwarfs = catin.bruntt_dr14_overlap()
    bad_indices = astero_dwarfs["VSINI"] < 0
    notbad_dwarfs = astero_dwarfs[~bad_indices]

    bruntt_vsinis = notbad_dwarfs["vsini"]
    apogee_vsinis = notbad_dwarfs["VSINI"]
    vsini_diffs = bruntt_vsinis - apogee_vsinis
    # Enable to test how often false positives occur.
    vsini_diffs = np.where(
        bruntt_vsinis > 10, scipy.stats.norm.rvs(size=len(bruntt_vsinis)), 
        scipy.stats.uniform.rvs(size=len(bruntt_vsinis), scale=10))

    ad_values = np.zeros(len(notbad_dwarfs))
    for i in range(len(ad_values)):
        below_vals = bruntt_vsinis <= bruntt_vsinis[i]
        above_vals = bruntt_vsinis > bruntt_vsinis[i]

        try:
            ad_values[i] = scipy.stats.anderson_ksamp([
                vsini_diffs[below_vals],
                vsini_diffs[above_vals]]).significance_level
        except ValueError:
            ad_values[i] = np.nan

    sort_indices = np.argsort(bruntt_vsinis)
    plt.plot(bruntt_vsinis[sort_indices], ad_values[sort_indices], '-')
    plt.xlabel("Bruntt vsini (km/s)")
    plt.ylabel("Probability that high and low distributions are the same")



@write_plot("Pleiades_comp")
def Pleiades_vsini_comparison():
    '''Create figure comparing vsini for Pleiades targets.'''
    f, ax = plt.subplots(1,1, figsize=(10,10))
    targets = catin.Stauffer_APOGEE_overlap()
    notestr = au.byte_to_unicode_cast(targets["Notes"])
    non_dlsbs = targets[npstr.find(notestr, "5") < 0]
    good_targets = catalog.apogee_filter_quality(non_dlsbs, quality="bad")
    nondetections = good_targets["vsini lim"] == stat.UPPER
    detected_targets = good_targets[~nondetections]
    nondet_targets = good_targets[nondetections]
    detected_errors = (detected_targets["vsini"] / 2 / (
        1 + detected_targets["R"])).filled(0)

#   det_frac = ((detected_targets["vsini"] - detected_targets["VSINI"]) /
#               detected_targets["vsini"])
#   frac_errors = detected_errors * (
#       detected_targets["VSINI"] / detected_targets["vsini"]**2)
#   nondet_frac = ((nondet_targets["vsini"] - nondet_targets["VSINI"]) /
#                  nondet_targets["vsini"])
                   
    ax.set_xscale("log")
    ax.set_yscale("log")
    weird_targets = np.logical_and(detected_targets["VSINI"] < 10,
                                   detected_targets["vsini"] > 10)
    ax.errorbar(detected_targets["vsini"], detected_targets["VSINI"], 
                 xerr=detected_errors, fmt='ko')
    ax.plot(nondet_targets["vsini"], nondet_targets["VSINI"], 'r<')
    one_to_one = np.array([1, 100])
    ax.plot(one_to_one, one_to_one, 'k-')
    ax.plot(one_to_one, [7, 7], 'k:')
    ax.plot(one_to_one, [10, 10], 'k:')
    
    ax.set_xlabel(r"Stauffer and Hartmann (1987) $v \sin i$ (km/s)")
    ax.set_ylabel(r"APOGEE $v \sin i$ (km/s)")
    ax.set_xlim(1, 100)
    ax.set_ylim(1, 100)


def Pleiades_teff_vsini_comparison():
    '''Show vsini uncertainties with Teff.'''
    targets = catin.Stauffer_APOGEE_overlap()
    non_dlsbs = targets[npstr.find(targets["Notes"], "5") < 0]
    good_targets = catalog.apogee_filter_quality(non_dlsbs, quality="bad")
    nondetections = good_targets["vsini lim"] == stat.UPPER
    detected_targets = good_targets[~nondetections]
    nondet_targets = good_targets[nondetections]
    detected_errors = (detected_targets["vsini"] / 2 / (
        1 + detected_targets["R"])).filled(0)
    vsini_diff = detected_targets["VSINI"] - detected_targets["vsini"]
    weird_targets = np.logical_and(detected_targets["VSINI"] < 10,
                                   detected_targets["vsini"] > 10)

    plt.errorbar(detected_targets["TEFF"], vsini_diff, yerr=detected_errors,
                 fmt="ko")
    plt.errorbar(detected_targets["TEFF"][weird_targets],
                 vsini_diff[weird_targets], yerr=detected_errors[weird_targets],
                 fmt="b*")
    plt.ylabel("APOGEE - SH vsini (km/s)")
    plt.xlabel("APOGEE Teff (K)")
    hr.invert_x_axis()
    print(np.std(vsini_diff[detected_targets["TEFF"] < 4000]))

def plot_lower_limit_Pleiades_test():
    '''Plot results of two-sample AD test for checking lower limit.'''
    targets = catin.Stauffer_APOGEE_overlap()
    non_dlsbs = targets[npstr.find(targets["Notes"], "5") < 0]
    good_targets = catalog.apogee_filter_quality(non_dlsbs, quality="bad")

    sh_vsinis = good_targets["vsini"]
    apogee_vsinis = good_targets["VSINI"]
    vsini_diffs = np.log10(sh_vsinis) - np.log10(apogee_vsinis)
    # Enable to test how often false positives occur.
#    vsini_diffs = scipy.stats.norm.rvs(size=len(sh_vsinis))

    ad_values = np.zeros(len(good_targets))
    for i in range(len(ad_values)):
        below_vals = sh_vsinis <= sh_vsinis[i]
        above_vals = sh_vsinis > sh_vsinis[i]

        try:
            ad_values[i] = scipy.stats.anderson_ksamp([
                vsini_diffs[below_vals],
                vsini_diffs[above_vals]]).significance_level
        except ValueError:
            ad_values[i] = np.nan

    sort_indices = np.argsort(sh_vsinis)
    plt.plot(sh_vsinis[sort_indices], ad_values[sort_indices], 'k-')
    plt.xlabel("Stauffer-Hartmann vsini (km/s)")
    plt.ylabel("Probability that high and low distributions are the same")

@write_plot("astero_rot")
def asteroseismic_rotation_analysis():
    '''Plot rotation comparison of asteroseismic sample.'''
    astero = asteroseismic_data_splitter()
    detections = astero.subsample([
        "~Bad", "Asteroseismic Dwarfs", "Vsini det", "~DLSB"])

    garcia = catin.read_Garcia_periods()
    detections_periodpoints = au.join_by_id(detections, garcia, "kepid", "KIC")

    f, ax = plt.subplots(1,1, figsize=(10,10))

    rot.plot_vsini_velocity(
        detections_periodpoints["VSINI"], detections_periodpoints["Prot"],
        detections_periodpoints["radius"], 
        raderr_below=detections_periodpoints["radius_err1"],
        raderr_above=detections_periodpoints["radius_err2"], color=bc.blue,
        ax=ax)

@write_plot("turnoff")
def old_turnoff():
    '''Plot the turnoff for 10 Gyr populations at three metallicities.'''
    f, ax = plt.subplots(1,1, figsize=(10,10))
    massrange = np.linspace(0.5, 1.5, 500)
    highmet = dsep.DSEPIsochrone.isochrone_from_file(0.5)
    highmet_teff = 10**dsep.interpolate_DSEP_isochrone_cols(
        highmet, 10, massrange, incol="M/Mo", outcol="LogTeff")
    highmet_K = dsep.interpolate_DSEP_isochrone_cols(
        highmet, 10, massrange, incol="M/Mo", outcol="Ks")
    solmet = dsep.DSEPIsochrone.isochrone_from_file(0.0)
    solmet_teff = 10**dsep.interpolate_DSEP_isochrone_cols(
        solmet, 10, massrange, incol="M/Mo", outcol="LogTeff")
    solmet_K = dsep.interpolate_DSEP_isochrone_cols(
        solmet, 10, massrange, incol="M/Mo", outcol="Ks")
    lowmet = dsep.DSEPIsochrone.isochrone_from_file(-0.5)
    lowmet_teff = 10**dsep.interpolate_DSEP_isochrone_cols(
        lowmet, 10, massrange, incol="M/Mo", outcol="LogTeff")
    lowmet_K = dsep.interpolate_DSEP_isochrone_cols(
        lowmet, 10, massrange, incol="M/Mo", outcol="Ks")

    hr.absmag_teff_plot(
        highmet_teff, highmet_K, color=bc.red, linestyle="-", marker="", ls=3,
        axis=ax, label="[Fe/H] = 0.5")
    hr.absmag_teff_plot(
        solmet_teff, solmet_K, color=bc.black, linestyle="-", marker="", ls=3,
        axis=ax, label="[Fe/H] = 0.0")
    hr.absmag_teff_plot(
        lowmet_teff, lowmet_K, color=bc.blue, linestyle="-", marker="", ls=3,
        axis=ax, label="[Fe/H] = -0.5")

    # Now add some data.
    full = cache.full_apogee_splitter()
    full_data = full.subsample([
        "~Bad", "H APOGEE", "In Gaia", "~Berger Giant"])

    hr.absmag_teff_plot(full_data["TEFF"], full_data["M_K"], color=bc.black,
                        marker=".", ls="", label="Berger Main Sequence")

    plt.plot([6250, 4500], [2.7, 2.7], 'k--')

    ax.set_xlim(6200, 4500)
    ax.set_ylim(5, 1)
    ax.legend(loc="lower left")
    ax.set_xlabel("APOGEE Teff (K)")
    ax.set_ylabel("Ks")

@write_plot("binarycut")
def plot_metallicity_excess():
    apo = cache.full_apogee_splitter()
    targets = apo.subsample(["~Bad", "H APOGEE", "In Gaia", "K Detection"])
    
    print("Excluding {0:d} not bad targets with bad Teffs".format(
        np.ma.count_masked(targets["TEFF"])))
    targets = targets[~np.ma.getmaskarray(targets["TEFF"])]

    print("Excluding {0:d} further bad targets with bad [Fe/H]".format(
        np.ma.count_masked(targets["FE_H"])))
    targets = targets[~np.ma.getmaskarray(targets["FE_H"])]

    targets["DSEP K"] = np.diag(samp.calc_model_mag_fixed_age_alpha(
        targets["TEFF"], targets["FE_H"], "Ks", model="MIST"))
    targets["K Excess"] = targets["M_K"] - targets["DSEP K"]

    dwarfs = targets[targets["M_K"] > 2.7]

    f, ax = plt.subplots(1,1, figsize=(10,10))
    hr.absmag_teff_plot(targets["TEFF"], targets["K Excess"], color=bc.black,
                        linestyle="", marker=".", label="Full", axis=ax)
    hr.absmag_teff_plot(dwarfs["TEFF"], dwarfs["K Excess"], color=bc.green,
                        linestyle="", marker=".", label="Dwarfs", axis=ax)
    ax.plot([8000, 3500], [0, 0], 'k-', lw=3)
    ax.plot([8000, 3500], [-0.75, -0.75], 'k:')
    ax.plot([5500, 5500], [-6, 6], 'k--', lw=2)

    ax.set_xlabel("APOGEE Teff (K)")
    ax.set_ylabel("M_K - DSEP K (Age: 5.5; [Fe/H] adjusted)")
    ax.legend(loc="upper left")
    ax.set_ylim(1.5, -3)
    ax.set_xlim(6100, 3500)

@write_plot("astero")
def asteroseismic_sample_MK():
    '''Write an HR diagram consisting of the asteroseismic sample.
    
    Plot the asteroseismic and spectroscopic log(g) values separately. Should
    also plot the underlying APOKASC sample.'''
    f, ax = plt.subplots(1,1, figsize=(12,12))
    apokasc = asteroseismic_data_splitter()
    apokasc.split_targeting("APOGEE2_APOKASC_DWARF")
    fulldata_k = apokasc.subsample(["~Bad", "APOGEE2_APOKASC_DWARF", "Good K"])
    fulldata_b = apokasc.subsample(["~Bad", "APOGEE2_APOKASC_DWARF", "Blend"])
    astero_k = apokasc.subsample(["~Bad", "Asteroseismic Dwarfs", "Good K"])
    astero_b = apokasc.subsample(["~Bad", "Asteroseismic Dwarfs", "Blend"])
#   spec_rapid = apokasc.subsample([
#       "~Bad", "APOGEE2_APOKASC_DWARF", "Vsini det", "~DLSB"])
#   spec_marginal = apokasc.subsample([
#       "~Bad", "APOGEE2_APOKASC_DWARF", "Vsini marginal", "~DLSB"])
#   dlsbs = apokasc.subsample(["~Bad", "APOGEE2_APOKASC_DWARF", "DLSB"])

    hr.absmag_teff_plot(
        fulldata_k["TEFF_COR"], fulldata_k["M_K"], color=bc.black, marker=".",
        label="APOGEE Hot Sample", ls="", axis=ax,
        yerr=[fulldata_k["M_K_err2"], fulldata_k["M_K_err1"]])
    hr.absmag_teff_plot(
        fulldata_b["TEFF_COR"], fulldata_b["M_K"], color=bc.black, marker="v",
        ls="", axis=ax, label="")
    hr.absmag_teff_plot(
        astero_k["TEFF_COR"], astero_k["M_K"], color=bc.green, marker="s", ms=8, 
        label="Asteroseismic sample", ls="", axis=ax, 
        yerr=[astero_k["M_K_err2"], astero_k["M_K_err1"]])
    hr.absmag_teff_plot(
        astero_b["TEFF_COR"], astero_b["M_K"], color=bc.green, marker="s", ms=8, 
        ls="", axis=ax, label="")
#   hr.absmag_teff_plot(
#       spec_rapid["TEFF_COR"], spec_rapid["M_K"], color=bc.blue, marker="o", 
#       label="Rapid Rotators", ls="", ms=7, axis=ax)
#   hr.absmag_teff_plot(
#       spec_marginal["TEFF_COR"], spec_marginal["M_K"], color=bc.sky_blue, 
#       marker="o", label="Marginal Rotators", ls="", ms=7, axis=ax)
#   hr.absmag_teff_plot(
#       dlsbs["TEFF_COR"], dlsbs["M_K"], color=bc.light_pink, marker="*", 
#       label="SB2", ls="", axis=ax, ms=7)

    plt.xlim([6750, 4750])
    plt.ylim([6, -3])
    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel(r"$M_K$")
    plt.legend(loc="upper left")

def asteroseismic_logg_Gaia_comparison():
    '''Write an HR diagram consisting of the asteroseismic sample.
    
    Plot the asteroseismic and spectroscopic log(g) values separately. Should
    also plot the underlying APOKASC sample.'''
    apokasc = asteroseismic_data_splitter()
    apokasc.split_targeting("APOGEE2_APOKASC_DWARF")
    fulldata = apokasc.subsample(["~Bad", "APOGEE2_APOKASC_DWARF"])
    astero_samp = apokasc.split_subsample(["~Bad", "Asteroseismic Dwarfs"])
    astero_samp.split_logg(
        "LOGG_DW", 4.2, ("Cool astero subgiant", "Cool astero dwarfs"), 
        logg_crit="Seismic logg")
    astero_hot = astero_samp.subsample(["Hot"])
    astero_subgiants = astero_samp.subsample(["Cool", "Cool astero subgiant"])
    astero_dwarfs = astero_samp.subsample(["Cool", "Cool astero dwarfs"])

    hr.absmag_teff_plot(
        fulldata["TEFF_COR"], fulldata["M_K"], color=bc.black, marker=".",
        label="APOGEE", ls="")
    hr.absmag_teff_plot(
        astero_hot["TEFF_COR"], astero_hot["M_K"], color=bc.green, marker="*", 
        ms=8, label="Hot stars", ls="")
    hr.absmag_teff_plot(
        astero_subgiants["TEFF_COR"], astero_subgiants["M_K"], 
        color=bc.light_pink, marker="o", ms=6, label="Cool subgiants", ls="")
    hr.absmag_teff_plot(
        astero_dwarfs["TEFF_COR"], astero_dwarfs["M_K"], 
        color=bc.violet, marker="o", ms=6, label="Cool dwarfs", ls="")

    plt.xlim([6750, 4750])
    plt.ylim([6, -8])
    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("M_K")
    plt.legend(loc="upper left")


@write_plot("cool_mk_sample")
def cool_dwarf_mk():
    '''Plot the cool dwarf sample on an HR diagram with M_K.

    Illustrate the dwarf/subgiant division in Gaia parallax space.'''
    f, ax = plt.subplots(1,1, figsize=(12,12))
    cool = cool_data_splitter()
    cool_full = cool.subsample(["~Bad"])
    cool_subgiants_k = cool.subsample(["~Bad", "Berger Subgiant", "Good K"])
    cool_subgiants_b = cool.subsample(["~Bad", "Berger Subgiant", "Blend"])
    cool_dwarfs_k = cool.subsample([
        "~Bad", "Modified Berger Main Sequence", "Good K"])
    cool_dwarfs_b = cool.subsample([
        "~Bad", "Modified Berger Main Sequence", "Blend"])
    cool_giants_k = cool.subsample(["~Bad", "Berger Giant", "Good K"])
    cool_giants_b = cool.subsample(["~Bad", "Berger Giant", "Blend"])
    cool_binaries_k = cool.subsample([
        "~Bad", "Modified Berger Cool Binary", "Good K"])
    cool_binaries_b = cool.subsample([
        "~Bad", "Modified Berger Cool Binary", "Blend"])
    cool_rapid = cool.subsample([
        "~Bad", "Vsini det", "~DLSB"])
    cool_marginal = cool.subsample([
        "~Bad", "Vsini marginal", "~DLSB"])
    dlsbs = cool.subsample([ "~Bad", "DLSB", "~Berger Giant"])

#   cool_mcq = cool.subsample(["~Bad", "Mcq"]) 
#   mcq = catin.read_McQuillan_catalog()
#   mcq_joined = au.join_by_id(cool_mcq, mcq, "kepid", "KIC")
#   rapid_rotators = mcq_joined[mcq_joined["Prot"] < 3]

    hr.absmag_teff_plot(
        cool_giants_k["TEFF"], cool_giants_k["M_K"], color=bc.orange, marker=".", 
        label="Giants", ls="", axis=ax, 
        yerr=[cool_giants_k["M_K_err2"], cool_giants_k["M_K_err1"]])
    hr.absmag_teff_plot(
        cool_giants_b["TEFF"], cool_giants_b["M_K"], color=bc.orange, marker="v", 
        ls="", axis=ax, label="")
    hr.absmag_teff_plot(
        cool_subgiants_k["TEFF"], cool_subgiants_k["M_K"], color=bc.purple, 
        marker=".", label="Subgiants", ls="", axis=ax,
        yerr=[cool_subgiants_k["M_K_err2"], cool_subgiants_k["M_K_err1"]])
    hr.absmag_teff_plot(
        cool_subgiants_b["TEFF"], cool_subgiants_b["M_K"], color=bc.purple, 
        marker="v", ls="", axis=ax, label="")
    hr.absmag_teff_plot(
        cool_dwarfs_k["TEFF"], cool_dwarfs_k["M_K"], color=bc.algae, marker=".", 
        label="Dwarfs", ls="", axis=ax,
        yerr=[cool_dwarfs_k["M_K_err2"], cool_dwarfs_k["M_K_err1"]])
    hr.absmag_teff_plot(
        cool_dwarfs_b["TEFF"], cool_dwarfs_b["M_K"], color=bc.algae, marker="v", 
        ls="", axis=ax, label="")
    hr.absmag_teff_plot(
        cool_binaries_k["TEFF"], cool_binaries_k["M_K"], color=bc.green, 
        marker=".", label="Binaries", ls="", axis=ax,
        yerr=[cool_binaries_k["M_K_err2"], cool_binaries_k["M_K_err1"]])
    hr.absmag_teff_plot(
        cool_binaries_b["TEFF"], cool_binaries_b["M_K"], color=bc.green, 
        marker="v", ls="", axis=ax, label="")
#   hr.absmag_teff_plot(
#       cool_rapid["TEFF"], cool_rapid["M_K"], color=bc.blue, marker="o", 
#       label="Rapid Rotators", ls="", ms=7, axis=ax)
#   hr.absmag_teff_plot(
#       cool_marginal["TEFF"], cool_marginal["M_K"], color=bc.sky_blue, 
#       marker="o", label="Marginal Rotators", ls="", ms=7, axis=ax)
#   hr.absmag_teff_plot(
#       rapid_rotators["TEFF"], rapid_rotators["M_K"], color=bc.violet, 
#       marker="d", label="P < 3 day", ls="", axis=ax)
#   hr.absmag_teff_plot(
#       dlsbs["TEFF"], dlsbs["M_K"], color=bc.light_pink, marker="*", 
#       label="SB2", ls="", axis=ax, ms=7)
#   hr.absmag_teff_plot(
#       apogee_misclassified_subgiants["TEFF"], 
#       apogee_misclassified_subgiants["M_K"], color=bc.red, marker="x", 
#       label="Mismatched Spec. Ev. State", ls="", ms=7, axis=ax)
#   hr.absmag_teff_plot(
#       apogee_misclassified_dwarfs["TEFF"], apogee_misclassified_dwarfs["M_K"], 
#       color=bc.red, marker="x", ls="", label="", ms=7)

    plt.plot([5450, 5450], [6, -8], 'k:')
    plt.xlim(5750, 3500)
    plt.ylim(6, -8)
    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel(r"$M_K$")
    plt.legend(loc="upper left")

#   print("APOGEE-classified dwarfs: {0:d}".format(cool.subsample_len(
#       ["~Bad", "Subgiant", "APOGEE Dwarf"])))
#   print("Misclassified Rapid Rotators: {0:d}".format(cool.subsample_len(
#       ["~Bad", "Subgiant", "APOGEE Dwarf", "Vsini det", "~DLSB"])))
#   print("APOGEE-classified subgiants: {0:d}".format(cool.subsample_len(
#       ["~Bad", "Dwarf", "APOGEE Subgiant", "~DLSB"])))
#   print("Misclassified Rapid Rotators: {0:d}".format(cool.subsample_len(
#       ["~Bad", "Dwarf", "APOGEE Subgiant", "Vsini det", "~DLSB"])))

def cool_dwarf_logg():
    '''Plot the cool dwarf sample on an HR diagram using log(g).

    Illustrate the subgiant/dwarf division, and overplot rapid rotators.'''
    cool = cool_data_splitter()
    cool_full = cool.subsample(["~Bad"])
    cool_subgiants = cool.subsample(["~Bad", "Subgiant"])
    cool_dwarfs = cool.subsample(["~Bad", "Dwarf"])
    cool_giants = cool.subsample(["~Bad", "Giant"])
    cool_rapid_dwarfs = cool.subsample([
        "~Bad", "Vsini det", "~DLSB", "Dwarf", "Mcq"])
    hr.logg_teff_plot(cool_giants["TEFF"], cool_giants["LOGG_FIT"], 
                      color=bc.orange, marker=".", label="Giants", ls="")
    hr.logg_teff_plot(cool_subgiants["TEFF"], cool_subgiants["LOGG_FIT"], 
                      color=bc.purple, marker=".", label="Subgiants", ls="")
    hr.logg_teff_plot(cool_dwarfs["TEFF"], cool_dwarfs["LOGG_FIT"], 
                      color=bc.red, marker=".", label="Dwarfs", ls="")
    hr.logg_teff_plot(cool_rapid_dwarfs["TEFF"], cool_rapid_dwarfs["LOGG_FIT"], 
                      color=bc.blue, marker="*", label="Rapid Rotators",
                      ls="", ms=7)
    plt.plot([4500, 5690], [3.62, 4.43], 'k-')
    plt.plot([5450, 5450], [3.5, 4.7], 'k:')
    plt.xlim(5750, 3500)
    plt.ylim(4.7, 0.5)
    plt.xlabel("APOGEE Teff")
    plt.ylabel("APOGEE Logg")
    plt.legend()

def huber_apogee_teff_comparison():
    '''Plot Huber vs APOGEE Teffs.'''

    cool = cool_data_splitter()
    cool_full = cool.subsample(["~Bad"])
    cool_subgiants = cool.subsample(["~Bad", "Berger Subgiant"])
    cool_dwarfs = cool.subsample(["~Bad", "Berger Main Sequence"])
    cool_giants = cool.subsample(["~Bad", "Berger Giant"])
    cool_binaries = cool.subsample(["~Bad", "Berger Cool Binary"])
    cool_rapid = cool.subsample([
        "~Bad", "Vsini det", "~DLSB", "~Giant"])
    cool_marginal = cool.subsample([
        "~Bad", "Vsini marginal", "~DLSB", "~Giant"])
    apogee_misclassified_subgiants = cool.subsample([
        "~Bad", "APOGEE Subgiant", "Dwarf"])
    apogee_misclassified_dwarfs = cool.subsample([
        "~Bad", "APOGEE Dwarf", "Subgiant"])
    dlsbs = cool.subsample([ "~Bad", "DLSB", "~Giant"])

    plt.errorbar(
        cool_giants["TEFF"], cool_giants["teff"]-cool_giants["TEFF"], 
        yerr=[cool_giants["teff_err1"], -cool_giants["teff_err2"]], 
        color=bc.orange, marker=".", label="Giants", ls="")
    plt.errorbar(
        cool_subgiants["TEFF"], cool_subgiants["teff"]-cool_subgiants["TEFF"], 
        yerr=[cool_subgiants["teff_err1"], -cool_subgiants["teff_err2"]], 
        color=bc.purple, marker=".", label="Subgiants", ls="")
    plt.errorbar(
        cool_dwarfs["TEFF"], cool_dwarfs["teff"]-cool_dwarfs["TEFF"], 
        yerr=[cool_dwarfs["teff_err1"], -cool_dwarfs["teff_err2"]], 
        color=bc.algae, marker=".", label="Dwarfs", ls="")
    plt.errorbar(
        cool_binaries["TEFF"], cool_binaries["teff"]-cool_binaries["TEFF"], 
        yerr=[cool_binaries["teff_err1"], -cool_binaries["teff_err2"]], 
        color=bc.green, marker=".", label="Binaries", ls="")
    plt.errorbar(
        cool_rapid["TEFF"], cool_rapid["teff"]-cool_rapid["TEFF"], 
        yerr=[cool_rapid["teff_err1"], -cool_rapid["teff_err2"]], 
        color=bc.blue, marker="o", label="Rapid Rotators", ls="", ms=7)
    plt.errorbar(
        cool_marginal["TEFF"], cool_marginal["teff"]-cool_marginal["TEFF"], 
        yerr=[cool_marginal["teff_err1"], -cool_marginal["teff_err2"]], 
        color=bc.sky_blue, marker="o", label="Rapid Rotators", ls="", ms=7)
    plt.errorbar(
        dlsbs["TEFF"], dlsbs["teff"]-dlsbs["TEFF"], 
        yerr=[dlsbs["teff_err1"], -dlsbs["teff_err2"]], 
        color=bc.light_pink, marker="*", label="SB2", ls="")
    plt.errorbar(
        apogee_misclassified_subgiants["TEFF"], 
        apogee_misclassified_subgiants["teff"]-
        apogee_misclassified_subgiants["TEFF"], 
        yerr=[apogee_misclassified_subgiants["teff_err1"], -apogee_misclassified_subgiants["teff_err2"]], 
        color=bc.red, marker="x", label="Mismatched Spec. Ev. State", ls="", 
        ms=7)
    plt.errorbar(
        apogee_misclassified_dwarfs["TEFF"],
        apogee_misclassified_dwarfs["teff"]-
        apogee_misclassified_dwarfs["TEFF"], 
        yerr=[apogee_misclassified_dwarfs["teff_err1"], -apogee_misclassified_dwarfs["teff_err2"]], 
        color=bc.red, marker="x", ls="", label="", ms=7)

    plt.plot([3500, 6500], [0.0, 0.0], 'k-')
    plt.legend(loc="lower right")
    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("Huber - APOGEE Teff (K)")

@write_plot("teff_vsini")
def vsini_teff_trend():
    '''Show vsini against Teff.'''
    cool = cool_data_splitter()
    cool_dwarfs_rapid = cool.subsample([
        "~Bad", "Dwarf", "~DLSB", "Mcq", "Vsini det", "~Too Hot"])
    cool_dwarfs_marginal = cool.subsample([
        "~Bad", "Dwarf", "~DLSB", "Mcq", "Vsini marginal", "~Too Hot"])
    cool_dwarfs_nondet = cool.subsample([
        "~Bad", "Dwarf", "~DLSB", "Mcq", "Vsini nondet", "~Too Hot"])

    # Black triangles for nondetections
    plt.plot(cool_dwarfs_nondet["TEFF"], 7*np.ones(len(cool_dwarfs_nondet)), 
             marker="v", color=bc.black, ls="", label="Nondetections")
    # Light blue star for marginal rotators
    plt.errorbar(cool_dwarfs_marginal["TEFF"], cool_dwarfs_marginal["VSINI"],
                 cool_dwarfs_marginal["VSINI"]*0.15, marker="*",
                 color=bc.sky_blue, ls="", label=r"$v \sin i$ marginal")
    # Dark blue star for rapid rotators
    plt.errorbar(cool_dwarfs_rapid["TEFF"], cool_dwarfs_rapid["VSINI"],
                 cool_dwarfs_rapid["VSINI"]*0.15, marker="*", color=bc.blue,
                 ls="", label=r"$v \sin i$ detection")

    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("APOGEE $v \sin i$ (km/s)")
    plt.legend(loc="upper left")
                 
    hr.invert_x_axis()

@write_plot("teff_radius")
def radius_teff_trend():
    '''Show radius against Teff.'''
    cool = cool_data_splitter()
    cool_dwarfs = cool.split_subsample([
        "~Bad", "Dwarf", "~DLSB", "Mcq", "~Too Hot"])
    samp.generate_DSEP_radius_column_with_errors(cool_dwarfs.data)

    cool_dwarfs_rapid = cool_dwarfs.subsample(["Vsini det"])
    cool_dwarfs_marginal = cool_dwarfs.subsample(["Vsini marginal"])
    cool_dwarfs_nondet = cool_dwarfs.subsample(["Vsini nondet"])

    # Black circles for nondetections
    plt.errorbar(
        cool_dwarfs_nondet["TEFF"], cool_dwarfs_nondet["DSEP radius"], 
        yerr=[-cool_dwarfs_nondet["DSEP radius lower"], 
              cool_dwarfs_nondet["DSEP radius upper"]], 
        marker=".", color=bc.black, ls="", ms=5)
    # Light blue star for marginal rotators
    plt.errorbar(
        cool_dwarfs_marginal["TEFF"], cool_dwarfs_marginal["DSEP radius"], 
        yerr=[-cool_dwarfs_marginal["DSEP radius lower"], 
              cool_dwarfs_marginal["DSEP radius upper"]], 
        marker="*", color=bc.sky_blue, ls="", ms=5)
    # Dark blue star for rapid rotators
    plt.errorbar(
        cool_dwarfs_rapid["TEFF"], cool_dwarfs_rapid["DSEP radius"], 
        yerr=[-cool_dwarfs_rapid["DSEP radius lower"], 
              cool_dwarfs_rapid["DSEP radius upper"]], 
        marker="*", color=bc.blue, ls="")

    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("DSEP Radius")
    hr.invert_x_axis()

@write_plot("teff_period")
def period_teff_trend():
    '''Show radius against Teff.'''
    cool = cool_data_splitter()
    cool_dwarfs = cool.split_subsample([
        "~Bad", "Dwarf", "~DLSB", "Mcq", "~Too Hot"])
    mcq = catin.read_McQuillan_catalog()
    cool_dwarfs.data = au.join_by_id(
        cool_dwarfs.data, mcq, "kepid", "KIC", join_type="left")

    cool_rapid = cool.subsample([
        "Vsini det", "~Bad", "Dwarf", "~DLSB", "Mcq", "~Too Hot"])
    cool_marginal = cool.subsample([
        "Vsini marginal", "~Bad", "Dwarf", "~DLSB", "Mcq", "~Too Hot"])
    cool_nondet = cool.subsample([
        "Vsini nondet", "~Bad", "Dwarf", "~DLSB", "Mcq", "~Too Hot"])

    cool_dwarfs_rapid = au.join_by_id(cool_rapid, mcq, "kepid", "KIC")
    cool_dwarfs_marginal = au.join_by_id(cool_marginal, mcq, "kepid", "KIC")
    cool_dwarfs_nondet = au.join_by_id(cool_nondet, mcq, "kepid", "KIC")

    # Black circles for nondetections
    plt.errorbar(
        cool_dwarfs_nondet["TEFF"], cool_dwarfs_nondet["Prot"], 
        yerr=cool_dwarfs_nondet["e_Prot"],
        marker=".", color=bc.black, ls="")
    # Light blue star for marginal rotators
    plt.errorbar(
        cool_dwarfs_marginal["TEFF"], cool_dwarfs_marginal["Prot"], 
        yerr=cool_dwarfs_marginal["e_Prot"],
        marker="*", color=bc.sky_blue, ls="")
    # Dark blue star for rapid rotators
    plt.errorbar(
        cool_dwarfs_rapid["TEFF"], cool_dwarfs_rapid["Prot"], 
        yerr=cool_dwarfs_rapid["e_Prot"],
        marker="*", color=bc.blue, ls="")

    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("McQuillan Period")
    hr.invert_x_axis()

@write_plot("detection_fraction")
def plot_rr_fractions():
    '''Plot spectroscopic and photometric rapid rotator fractions.'''
    cool_data = cool_data_splitter()
    cool_dwarfs_mcq = vstack([
        cool_data.subsample([
            "~Bad", "~DLSB", "Mcq", "Modified Berger Main Sequence", 
            "~Too Hot"]), 
        cool_data.subsample([
            "~Bad", "~DLSB", "Mcq", "Modified Berger Cool Binary", 
            "~Too Hot"])])
    cool_dwarfs_nomcq = vstack([
        cool_data.subsample([
            "~Bad", "~DLSB", "~Mcq", "Modified Berger Main Sequence", 
            "~Too Hot"]), 
        cool_data.subsample([
            "~Bad", "~DLSB", "~Mcq", "Modified Berger Cool Binary", 
            "~Too Hot"])])

    plt.figure(figsize=(10,10))

    mcq = catin.read_McQuillan_catalog()
    periods = au.join_by_id(cool_dwarfs_mcq, mcq, "kepid", "KIC")

    samp.generate_DSEP_radius_column_with_errors(periods)

    samp.spectroscopic_photometric_rotation_fraction_comparison_plot(
        periods["VSINI"], periods["Prot"], 
        periods["DSEP radius"], min_limit=6, max_limit=12,
        color=bc.black, label="")
    samp.plot_rapid_rotation_detection_limits(
        cool_dwarfs_nomcq["VSINI"], label="Mcquillan Nondetections",
        color=bc.black, ls=":", min_limit=6, max_limit=12) 
    plt.legend(loc="lower right")
    plt.ylim(0.0, 1.0)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.set_ylim(0.7, 1)
    ticks = ax1.get_yticks()
    print(ticks)
    fracticks = 1 - ticks
    print(fracticks)
    ax2.set_yticks(np.linspace(0, 1, len(fracticks), endpoint=True))
    ax2.set_yticklabels(["{0:.2f}".format(f) for f in fracticks])
    ax1.set_xlim(5.5, 12.5)
    ax2.set_xlim(5.5, 12.5)
    ax1.set_ylabel(r"$N (< v \sin i) / N$")
    ax1.set_xlabel(r"$v \sin i$")
    ax2.set_ylabel("Rapid Rotator Fraction")



@write_plot("cool_rot")
def cool_dwarf_rotation_analysis():
    '''Plot rotation comparison of cool dwarf sample.'''
    cool_dwarf = cool_data_splitter()
    marginal_dwarfs = cool_dwarf.subsample([
            "~Bad", "Vsini marginal", "~DLSB", "Mcq", 
            "Modified Berger Main Sequence"]) 
    marginal_binaries = cool_dwarf.subsample([
            "~Bad", "Vsini marginal", "~DLSB", "Mcq", 
            "Modified Berger Cool Binary"])
    dwarf_detections = cool_dwarf.subsample([
            "~Bad", "Vsini det", "~DLSB", "Mcq", 
            "Modified Berger Main Sequence"])
    binary_detections = cool_dwarf.subsample([
            "~Bad", "Vsini det", "~DLSB", "Mcq", 
            "Modified Berger Cool Binary"])

    mcq = catin.read_McQuillan_catalog()
    marginal_dwarf_periodpoints = au.join_by_id(
        marginal_dwarfs, mcq, "kepid", "KIC")
    marginal_binary_periodpoints = au.join_by_id(
        marginal_binaries, mcq, "kepid", "KIC")
    dwarf_detections_periodpoints = au.join_by_id(
        dwarf_detections, mcq, "kepid", "KIC")
    binary_detections_periodpoints = au.join_by_id(
        binary_detections, mcq, "kepid", "KIC")

    samp.generate_DSEP_radius_column_with_errors(marginal_dwarf_periodpoints)
    samp.generate_DSEP_radius_column_with_errors(marginal_binary_periodpoints)
    samp.generate_DSEP_radius_column_with_errors(dwarf_detections_periodpoints)
    samp.generate_DSEP_radius_column_with_errors(binary_detections_periodpoints)

    subplot_tup = rot.plot_rotation_velocity_radius(
        dwarf_detections_periodpoints["VSINI"], 
        dwarf_detections_periodpoints["Prot"], 
        dwarf_detections_periodpoints["DSEP radius"],
        raderr_below=dwarf_detections_periodpoints["DSEP radius lower"],
        raderr_above=dwarf_detections_periodpoints["DSEP radius upper"],
        color=bc.blue, label="Cool dwarfs") 

    rot.plot_rotation_velocity_radius(
        binary_detections_periodpoints["VSINI"], 
        binary_detections_periodpoints["Prot"], 
        binary_detections_periodpoints["DSEP radius"],
        raderr_below=binary_detections_periodpoints["DSEP radius lower"],
        raderr_above=binary_detections_periodpoints["DSEP radius upper"],
        subplot_tup=subplot_tup, color=bc.blue, marker="8")
        
    rot.plot_rotation_velocity_radius(
        marginal_dwarf_periodpoints["VSINI"], 
        marginal_dwarf_periodpoints["Prot"], 
        marginal_dwarf_periodpoints["DSEP radius"],
        raderr_below=marginal_dwarf_periodpoints["DSEP radius lower"],
        raderr_above=marginal_dwarf_periodpoints["DSEP radius upper"],
        subplot_tup=subplot_tup, color=bc.sky_blue)
        
    rot.plot_rotation_velocity_radius(
        marginal_binary_periodpoints["VSINI"], 
        marginal_binary_periodpoints["Prot"], 
        marginal_binary_periodpoints["DSEP radius"],
        raderr_below=marginal_binary_periodpoints["DSEP radius lower"],
        raderr_above=marginal_binary_periodpoints["DSEP radius upper"],
        subplot_tup=subplot_tup, color=bc.sky_blue, marker="8")

def full_sample_mk():
    '''Plot the full sample on an HR diagram with M_K'''
    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    full = cache.full_apogee_splitter()
    targets = full.subsample(["~Bad", "In Gaia", "K Detection"])

    print("Excluding {0:d} not bad targets with bad Teffs".format(
        np.ma.count_masked(targets["TEFF"])))
    targets = targets[~np.ma.getmaskarray(targets["TEFF"])]
    dwarfs = targets[targets["M_K"] > 2.95]

    hr.absmag_teff_plot(
        targets["TEFF"], targets["M_K"], 
        yerr=[targets["M_K_err2"], targets["M_K_err1"]], color=bc.black, 
        linestyle="", marker=".", label="Full")
    hr.absmag_teff_plot(
        dwarfs["TEFF"], dwarfs["M_K"], color=bc.yellow, linestyle="", 
        marker=".", label="Dwarf")
    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("M_K")
    plt.legend(loc="lower left")

def plot_solar_excess():
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw = {"height_ratios": [2, 1]},
                               sharex=True)
    apo = cache.full_apogee_splitter()
    targets = apo.subsample(["~Bad", "In Gaia", "K Detection"])

    print("Excluding {0:d} not bad targets with bad Teffs".format(
        np.ma.count_masked(targets["TEFF"])))

    targets = targets[~np.ma.getmaskarray(targets["TEFF"])]
    targets["DSEP K"] = samp.calc_solar_DSEP_model_mag(targets["TEFF"], "Ks")
    targets["K Excess"] = targets["M_K"] - targets["DSEP K"]
    print(np.ma.count_masked(targets["K Excess"]))

    # These are high and low metallicity isochrones.
    testteffs = np.linspace(6500, 3500, 100)
    highmet_k = samp.calc_DSEP_model_mag_fixed_age_alpha(
        testteffs, np.ones(len(testteffs))*0.1, "Ks")
    solmet_k = samp.calc_DSEP_model_mag_fixed_age_alpha(
        testteffs, np.ones(len(testteffs))*0.0, "Ks")
    lowmet_k = samp.calc_DSEP_model_mag_fixed_age_alpha(
        testteffs, np.ones(len(testteffs))*-0.1, "Ks")

    dwarfs = targets[targets["M_K"] > 2.95]

    hr.absmag_teff_plot(targets["TEFF"], targets["K Excess"], color=bc.black,
                        linestyle="", marker=".", label="Full", axis=a0)
    hr.absmag_teff_plot(dwarfs["TEFF"], dwarfs["K Excess"], color=bc.yellow,
                        linestyle="", marker=".", label="Dwarfs", axis=a0)
    hr.absmag_teff_plot(targets["TEFF"], targets["K Excess"], color=bc.black,
                        linestyle="", marker=".", axis=a1)
    hr.absmag_teff_plot(dwarfs["TEFF"], dwarfs["K Excess"], color=bc.yellow,
                        linestyle="", marker=".", axis=a1)
    a0.plot([8000, 3500], [0, 0], 'k-')
    a1.plot([8000, 3500], [0, 0], 'k-')
    a0.plot([8000, 3500], [-0.75, -0.75], 'k:', label="Equal mass")
    a1.plot([8000, 3500], [-0.75, -0.75], 'k:')
    a0.plot(testteffs, highmet_k-solmet_k, 'k--', label="[Fe/H] +- 0.1")
    a0.plot(testteffs, lowmet_k-solmet_k, 'k--')
    a1.plot(testteffs, highmet_k-solmet_k, 'k--')
    a1.plot(testteffs, lowmet_k-solmet_k, 'k--')

    a0.set_xlabel("")
    a1.set_xlabel("APOGEE Teff (K)")
    a0.set_ylabel("M_K - DSEP K (Age: 5.5; solar)")
    a1.set_ylabel("M_K - DSEP K")
    a1.set_ylim(1, -1.5)
    a0.legend(loc="upper left")

    targets = targets[~np.ma.getmaskarray(targets["TEFF"])]

@write_plot("binary_cut")
def plot_k_excess_rotation_apogee():
    '''Plot the K-band excess against rotation for McQuillan targets'''
    apo = cache.full_apogee_splitter()
    good_phot = apo.split_subsample(["~Bad", "In Gaia", "K Detection"])
    good_phot.split_mag(
        "M_K", 2.95, ("Nondwarfs", "Dwarfs", "No mag"), 
        mag_crit="K dwarf separation", null_value=np.ma.masked)
    good_phot.split_teff(
        "TEFF", 5200, ["Apo Cool", "Apo Hot",  "No APOGEE Teff"], 
        teff_crit="Exclude hot", null_value=np.ma.masked)
    targets = good_phot.subsample(["Dwarfs", "Apo Cool"])
    dlsbs = good_phot.subsample(["Dwarfs", "Apo Cool", "DLSB"])

    print("Excluding {0:d} not bad targets with bad Teffs".format(
        good_phot.subsample_len(["No APOGEE Teff"])))

    print("Excluding {0:d} further bad targets with bad [Fe/H]".format(
        np.ma.count_masked(targets["FE_H"])))
    targets = targets[~np.ma.getmaskarray(targets["FE_H"])]

    targets["DSEP K"] = samp.calc_model_mag_fixed_age_alpha(
        targets["TEFF"], targets["FE_H"], "Ks", age=9.65, model="MIST")
    dlsbs["DSEP K"] = samp.calc_model_mag_fixed_age_alpha(
        dlsbs["TEFF"], dlsbs["FE_H"], "Ks", age=9.65)
    targets["K Excess"] = targets["M_K"] - targets["DSEP K"]
    dlsbs["K Excess"] = dlsbs["M_K"] - dlsbs["DSEP K"]

    plt.plot(targets["VSINI"], targets["K Excess"], marker=".", ls="",
             color=bc.black, label="Dwarfs")
    plt.plot(dlsbs["VSINI"], dlsbs["K Excess"], marker="*", ls="",
             color=bc.light_pink, label="SB2 (unrel. vsini)")
    hr.invert_y_axis()

    plt.plot([0, 75], [0, 0], 'k-')
    plt.plot([0, 75], [-0.75, -0.75], 'k--')
    plt.plot([7, 7], [1.35, -1.65], 'k--')
    plt.ylabel("M_K - DSEP K (5.5 Gyr; [Fe/H] adjusted) ")
    plt.xlabel("vsini (km/s)")
    plt.legend(loc="lower right")
    plt.title("Teff < 5200 K")



def plot_binarity_diagram():
    cool_dwarf = cool_data_splitter()
    dwarfs_k = cool_dwarf.subsample([
            "~Bad", "Modified Berger Main Sequence", "Good K"]) 
    dwarfs_b = cool_dwarf.subsample([
            "~Bad", "Modified Berger Main Sequence", "Blend"]) 
    binaries_k = cool_dwarf.subsample([
            "~Bad", "Modified Berger Cool Binary", "Good K"]) 
    binaries_b = cool_dwarf.subsample([
            "~Bad", "Modified Berger Cool Binary", "Blend"]) 
    subgiants_k = cool_dwarf.subsample([
            "~Bad", "Berger Subgiant", "Good K"]) 
    subgiants_b = cool_dwarf.subsample([
            "~Bad", "Berger Subgiant", "Blend"]) 

    oldsoliso = sed.DSEPInterpolator(age=10.0, feh=0.0, minlogG=3.5, lowT=3700)
    oldsoldata = oldsoliso._get_isochrone_data("Ks")
    oldsolteff = 10**oldsoldata["LogTeff"]
    oldsolMK = oldsoldata["Ks"]
    oldsolmagdiff = samp.calc_photometric_excess(
        oldsolteff, np.zeros(len(oldsolteff)), "Ks", oldsolMK)

    oldlowiso = sed.DSEPInterpolator(age=10.0, feh=-0.5, minlogG=3.5, lowT=3700)
    oldlowdata = oldlowiso._get_isochrone_data("Ks")
    oldlowteff = 10**oldlowdata["LogTeff"]
    oldlowMK = oldlowdata["Ks"]
    oldlowmagdiff = samp.calc_photometric_excess(
        oldlowteff, np.zeros(len(oldlowteff))-0.5, "Ks", oldlowMK)

    oldhighiso = sed.DSEPInterpolator(age=10.0, feh=0.5, minlogG=3.5, lowT=3700)
    oldhighdata = oldhighiso._get_isochrone_data("Ks")
    oldhighteff = 10**oldhighdata["LogTeff"]
    oldhighMK = oldhighdata["Ks"]
    oldhighmagdiff = samp.calc_photometric_excess(
        oldhighteff, np.zeros(len(oldhighteff))+0.5, "Ks", oldhighMK)

    subgiantdiff = samp.calc_photometric_excess(
        subgiants_k["TEFF"], subgiants_k["FE_H"], "Ks", subgiants_k["M_K"])

    f = plt.figure(figsize=(10,10))
    fulltable = vstack([dwarfs_k, dwarfs_b, binaries_k, binaries_b])

    hr.absmag_teff_plot(
        oldsolteff, oldsolmagdiff, ls="-", color=bc.black, 
        label="[Fe/H] = 0.0")
    hr.absmag_teff_plot(
        oldlowteff, oldlowmagdiff, ls="-", color=bc.blue, 
        label="[Fe/H] = -0.5")
    hr.absmag_teff_plot(
        oldhighteff, oldhighmagdiff, ls="-", color=bc.red, 
        label="[Fe/H] = 0.5")
    hr.absmag_teff_plot(
        subgiants_k["TEFF"], subgiantdiff, ls="", color=bc.purple,
        label="Subgiants", marker=".")
    samp.plot_photometric_binary_excess(
        fulltable["TEFF"], fulltable["FE_H"], "Ks", fulltable["M_K"])
    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("Photometric excess")
    plt.xlim(5500, 3700)
    plt.ylim(0.3, -2.5)

@write_plot("binarity")
def rapid_rotator_binarity():
    '''Show the binarity of rapid rotators by marking their luminosity excess.'''
    cool_dwarf = cool_data_splitter()
    dwarfs = cool_dwarf.subsample([
            "~Bad", "Modified Berger Main Sequence"]) 
    binaries = cool_dwarf.subsample([
            "~Bad", "Modified Berger Cool Binary"]) 
    rapid = vstack([
        cool_dwarf.subsample([
            "~Bad", "Modified Berger Main Sequence", "Vsini det",
            "~DLSB"]), 
        cool_dwarf.subsample([
            "~Bad", "Modified Berger Cool Binary", "Vsini det",
            "~DLSB"])])
    marginal = vstack([
        cool_dwarf.subsample([
            "~Bad", "Modified Berger Main Sequence", 
            "Vsini marginal", "~DLSB"]), 
        cool_dwarf.subsample([
            "~Bad", "Modified Berger Cool Binary", 
            "Vsini marginal", "~DLSB"])])
    dlsb  = vstack([
        cool_dwarf.subsample([
            "~Bad", "Modified Berger Main Sequence", "DLSB"]), 
        cool_dwarf.subsample([
            "~Bad", "Modified Berger Cool Binary", "DLSB"])])

    mcq = catin.read_McQuillan_catalog()
    mcq_dwarfs = au.join_by_id(dwarfs, mcq, "kepid", "KIC", join_type="inner")
    mcq_binaries = au.join_by_id(binaries, mcq, "kepid", "KIC",
                                 join_type="inner")

    phot_rapid_dwarf = mcq_dwarfs[mcq_dwarfs["Prot"] < 3]
    phot_rapid_binary = mcq_binaries[mcq_binaries["Prot"] < 3]
    phot_rapid = vstack([phot_rapid_dwarf, phot_rapid_binary])

    oldsoliso = sed.DSEPInterpolator(age=14.0, feh=0.0, minlogG=3.5, lowT=3700)
    oldsoldata = oldsoliso._get_isochrone_data("Ks")
    oldsolteff = 10**oldsoldata["LogTeff"]
    oldsolMK = oldsoldata["Ks"]
    oldsolmagdiff = samp.calc_photometric_excess(
        oldsolteff, np.zeros(len(oldsolteff)), "Ks", oldsolMK)

    oldlowiso = sed.DSEPInterpolator(age=14.0, feh=-0.5, minlogG=3.5, lowT=3700)
    oldlowdata = oldlowiso._get_isochrone_data("Ks")
    oldlowteff = 10**oldlowdata["LogTeff"]
    oldlowMK = oldlowdata["Ks"]
    oldlowmagdiff = samp.calc_photometric_excess(
        oldlowteff, np.zeros(len(oldlowteff))-0.5, "Ks", oldlowMK)

    oldhighiso = sed.DSEPInterpolator(age=14.0, feh=0.5, minlogG=3.5, lowT=3700)
    oldhighdata = oldhighiso._get_isochrone_data("Ks")
    oldhighteff = 10**oldhighdata["LogTeff"]
    oldhighMK = oldhighdata["Ks"]
    oldhighmagdiff = samp.calc_photometric_excess(
        oldhighteff, np.zeros(len(oldhighteff))+0.5, "Ks", oldhighMK)

    dwarfdiff = samp.calc_photometric_excess(
        dwarfs["TEFF"], dwarfs["FE_H"], "Ks", dwarfs["M_K"])
    binarydiff = samp.calc_photometric_excess(
        binaries["TEFF"], binaries["FE_H"], "Ks", binaries["M_K"])
    rapiddiff = samp.calc_photometric_excess(
        rapid["TEFF"], rapid["FE_H"], "Ks", rapid["M_K"])
    marginaldiff = samp.calc_photometric_excess(
        marginal["TEFF"], marginal["FE_H"], "Ks", marginal["M_K"])
    dlsbdiff = samp.calc_photometric_excess(
        dlsb["TEFF"], dlsb["FE_H"], "Ks", dlsb["M_K"])
    photdiff = samp.calc_photometric_excess(
        phot_rapid["TEFF"], phot_rapid["FE_H"], "Ks", phot_rapid["M_K"])

    f = plt.figure(figsize=(10,10))
    fulltable = vstack([dwarfs, binaries])

    hr.absmag_teff_plot(
        oldsolteff, oldsolmagdiff, ls="-", color=bc.black, 
        label="[Fe/H] = 0.0")
    hr.absmag_teff_plot(
        oldlowteff, oldlowmagdiff, ls="-", color=bc.blue, 
        label="[Fe/H] = -0.5")
    hr.absmag_teff_plot(
        oldhighteff, oldhighmagdiff, ls="-", color=bc.red, 
        label="[Fe/H] = 0.5")
    hr.absmag_teff_plot(
        dwarfs["TEFF"], dwarfdiff, ls="", color=bc.algae,
        label="Dwarf", marker=".")
    hr.absmag_teff_plot(
        binaries["TEFF"], binarydiff, ls="", color=bc.green,
        label="Binary", marker=".")
    hr.absmag_teff_plot(
        rapid["TEFF"], rapiddiff, ls="", color=bc.blue,
        label="Rapid Rotator", marker="o")
    hr.absmag_teff_plot(
        marginal["TEFF"], marginaldiff, ls="", color=bc.sky_blue,
        label="Marginal Rotator", marker="o")
    hr.absmag_teff_plot(
        dlsb["TEFF"], dlsbdiff, ls="", color=bc.light_pink,
        label="SB2", marker="*")
    hr.absmag_teff_plot(
        phot_rapid["TEFF"], photdiff, ls="", color=bc.pink,
        label="Phot Rapid Rotator", marker="d")
    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("Photometric excess")
    plt.legend(loc="upper right")
    plt.xlim(5500, 3700)
    plt.ylim(0.3, -2.5)

@write_plot("mcq_binarity")
def plot_mcq_binarity_histogram():
    '''Compare magnitude excess of rapid rotators to regular targets.'''
    mcq = split.McQuillanSplitter()
    split.initialize_mcquillan_sample(mcq)
    mcq.split_mag(
        "M_K", 2.95, ("Nondwarfs", "Dwarfs", "No mag"), 
        mag_crit="K dwarf separation")
    mcq.split_teff("teff", 5200, ("Cool", "Subgiant regime"),
                   teff_crit="Subgiant Confusion split")

    fullsamp = mcq.subsample(["Dwarfs", "Cool"])
    rapid = mcq.subsample(["Dwarfs", "Cool", "~Slow"])

    fullsamp["DSEP K"] = samp.calc_solar_DSEP_model_mag(
        fullsamp["teff"], "Ks")
    rapid["DSEP K"] = samp.calc_solar_DSEP_model_mag(
        rapid["teff"], "Ks") 
    fullsamp["K Excess"] = fullsamp["M_K"] - fullsamp["DSEP K"]
    rapid["K Excess"] = rapid["M_K"] - rapid["DSEP K"]

    f = plt.figure(figsize=(10,10))
    bins = np.linspace(-4.5, 3.5, 76, endpoint=True)
    plt.hist(fullsamp["K Excess"], normed=True, bins=bins, histtype="step",
             cumulative=True, label="Full Mcquillan", color=bc.black)
    plt.hist(rapid["K Excess"], normed=True, bins=bins, histtype="step",
             cumulative=True, label="P < 3 day", color=bc.blue)
    plt.xlabel("Mag excess")
    plt.ylabel("N(< Mag excess)/N")
    plt.legend(loc="upper left")
    plt.xlim(-4.5, 3.5)

def plot_apogee_binarity_histogram():
    '''Compare magnitude excess of APOGEE rapid rotators to regular targets.'''
    cools = cool_data_splitter()
    cools_mk = cools.split_subsample(["In Gaia", "K Detection"])
    cools_mk.split_teff("TEFF", [3500, 5500], ("Low MS", "MS", "High MS"), 
                     teff_crit="MS Split")
    cools_mk.split_mag("M_K", 2.95, splitnames=("High", "Low"), 
                    mag_crit="M_K split")
    cool_data = cools_mk.subsample(["~Bad", "MS", "Low"])
    
    fullsamp = cools_mk.subsample(["~Bad", "MS", "Low"])
    rapid = cools_mk.subsample(["~Bad", "MS", "Low", "Vsini det"])

    full_magdiff = samp.calc_photometric_excess(
        fullsamp["TEFF"], fullsamp["FE_H"], fullsamp["ALPHA_FE"], "Ks",
        fullsamp["M_K"])
    rapid_magdiff = samp.calc_photometric_excess(
        rapid["TEFF"], rapid["FE_H"], rapid["ALPHA_FE"], "Ks", rapid["M_K"])
    print(len(full_magdiff))
    print(len(rapid_magdiff))

    f = plt.figure(figsize=(10,10))
    bins = np.linspace(-4.5, 3.5, 76, endpoint=True)
    plt.hist(full_magdiff, normed=True, bins=bins, histtype="step",
             cumulative=-1, label="Full APOGEE", color=bc.black)
    plt.hist(rapid_magdiff, normed=True, bins=bins, histtype="step",
             cumulative=-1, label="Rapid Rotators", color=bc.blue)
    plt.xlabel("Mag excess")
    plt.ylabel("N(< Mag excess)/N")
    plt.legend(loc="upper left")
    plt.xlim(-4.5, 3.5)

def plot_k_excess_rotation():
    '''Plot the K-band excess against rotation for McQuillan targets.'''
    mcq = split.McQuillanSplitter()
    split.initialize_mcquillan_sample(mcq)

    fullsamp = vstack([
        mcq.subsample(["Right Teff", "Berger Main Sequence"]), 
        mcq.subsample(["Right Teff", "Berger Cool Binary"])])

    full_magdiff = samp.calc_photometric_excess(
        fullsamp["teff"], 0.0, 0.0, "Ks", fullsamp["M_K"])

#    plt.plot(full_magdiff, fullsamp["Prot"], 'k.')
    plt.plot(fullsamp["Prot"], full_magdiff, 'k.')
    plt.plot([0.1, 69.9], [0, 0], 'k--')
    plt.ylabel("Mag Excess")
    plt.xlabel("Period (day)")
    plt.title("McQuillan Rapid Rotator Excess")
    hr.invert_y_axis()


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

def low_metallicity_selection():
    '''Determine whether low-metallicity targets should have been observed in
    APOGEE'''

    apo = cache.apogee_splitter_with_DSEP()
    targs = apo.subsample(["Dwarfs"])
    lowmet = targs[targs["FE_H"] < -0.5]

    targs["MIST H"] = np.diag(samp.calc_model_mag_fixed_age_alpha(
        targs["TEFF"], targs["FE_H"], "H", age=1e9, alpha=0.0, model="MIST"))
    targs["MIST H Apparent"] = (
        targs["MIST H"] + 5 * np.log10(targs["dis"]/10) + 0.190 *
        targs["av"])

    hr.absmag_teff_plot(targs["FE_H"], targs["H"], color=bc.black, ls="",
                        marker="o", label="Original")
    hr.absmag_teff_plot(targs["FE_H"], targs["MIST H Apparent"],
                        color=bc.red, ls="", marker="x", label="MIST")
    plt.plot([6500, 3500], [11, 11], 'k--')
    plt.xlabel("APOGEE [Fe/H]")
    plt.ylabel("Apparent H")


def APOGEE_metallicity_agreement():
    '''Compare predicted MK to DSEP isochrones.'''
    cools = cool_data_splitter()
    cools_mk = cools.split_subsample(["In Gaia", "K Detection"])
    cools_mk.split_teff("TEFF", [3500, 5500], ("Low MS", "MS", "High MS"), 
                     teff_crit="MS Split")
    cools_mk.split_mag("M_K", 2.95, splitnames=("High", "Low"), 
                    mag_crit="M_K split")
    cool_data = cools_mk.subsample(["~Bad", "MS", "Low"])

    teffbins = np.linspace(3500, 5500, 5, endpoint=True)
    bin_indices = np.digitize(cool_data["TEFF"], teffbins)
    colors = [bc.red, bc.orange, bc.green, bc.blue]

    dsepmag = samp.calc_DSEP_model_mags(
        cool_data["TEFF"], cool_data["FE_H"], cool_data["ALPHA_FE"], "Ks", 
        age=5.5)

    metrange = np.array([0.36, 0.21, 0.07, -0.5, -1.0, -1.5])
    for i in range(1, len(teffbins)):
        ind = bin_indices == i
        plt.errorbar(
            cool_data["FE_H"][ind], cool_data["M_K"][ind]-dsepmag[ind], 
            marker=".", color=colors[i-1], ls="", label="{0}<T<{1}".format(
            teffbins[i-1], teffbins[i]))
        youngs = samp.calc_DSEP_model_mags(
            np.ones(len(metrange))*np.mean(teffbins[i-1:i]), metrange,
            np.ones(len(metrange))*0.0, "Ks", age=3)
        olds = samp.calc_DSEP_model_mags(
            np.ones(len(metrange))*np.mean(teffbins[i-1:i]), metrange,
            np.ones(len(metrange))*0.0, "Ks", age=8)
        meds = samp.calc_DSEP_model_mags(
            np.ones(len(metrange))*np.mean(teffbins[i-1:i]), metrange,
            np.ones(len(metrange))*0.0, "Ks", age=5.5)
        plt.plot(metrange, youngs-meds, color=colors[i-1], ls="--", marker="")
        plt.plot(metrange, olds-meds, color=colors[i-1], ls="--", marker="")

    plt.plot([-1.4, 0.5], [0.0, 0.0], 'k-', lw=3)
    hr.invert_y_axis()
    plt.ylim(0.5, -1.0)
    plt.xlabel("APOGEE [Fe/H]")
    plt.ylabel("M_K - DSEP M_K (5.5 Gyr)")
    plt.legend(loc="lower left")

def APOGEE_metallicity_slice():
    '''Plot a slice of targets at a metallicity.'''
    cools = cool_data_splitter()
    cools_mk = cools.split_subsample(["In Gaia", "K Detection"])
    cools_mk.split_teff("TEFF", 5400, ("Lower MS", "Higher MS"), 
                     teff_crit="Lower MS Split")
    cools_mk.split_mag("M_K", 2.95, splitnames=("High", "Low"), 
                    mag_crit="M_K split")
    cools_mk.split_metallicity(
        [-0.1, 0.1], ["High met", "Sol met", "Low Met", "No met"], col="FE_H", 
        null_value=np.ma.masked)
    cool_data = cools_mk.subsample(["~Bad", "Lower MS", "Low", "Sol met"])

    teff_unc = np.exp(4.58343 + 0.000289796 * (cool_data["TEFF"] - 4500))


    tefflines = np.linspace(3600, 5400, 200)
    dsepmag = samp.calc_DSEP_model_mags(
        tefflines, 0.0, 0.0, "Ks", age=5.5)
    highmet = samp.calc_DSEP_model_mags(
        tefflines, 0.1, 0.0, "Ks", age=1)
    lowmet = samp.calc_DSEP_model_mags(
        tefflines, -0.1, 0.0, "Ks", age=1)
    solmet = samp.calc_DSEP_model_mags(
        tefflines, 0.0, 0.0, "Ks", age=1)
    older = samp.calc_DSEP_model_mags(
        tefflines, 0.0, 0.0, "Ks", age=8)
    younger = samp.calc_DSEP_model_mags(
        tefflines, 0.0, 0.0, "Ks", age=3)

    hr.absmag_teff_plot(
        cool_data["TEFF"], cool_data["M_K"], ls="", marker=".", 
        color=bc.black, yerr=np.array([
            cool_data["M_K_err2"], cool_data["M_K_err1"]]), xerr=teff_unc, label="APOGEE")
    hr.absmag_teff_plot(tefflines, dsepmag, ls="-", marker="",
                        color=bc.red, label="5 Gyr")
    hr.absmag_teff_plot(
        tefflines, highmet, ls=":", marker="",
        color=bc.red, label="[Fe/H] +/- 0.1")
    hr.absmag_teff_plot(
        tefflines, lowmet, ls=":", marker="",
        color=bc.red, label="")
    hr.absmag_teff_plot(
        tefflines, older, ls="--", marker="",
        color=bc.red, label="8 Gyr")
    hr.absmag_teff_plot(
        tefflines, younger, ls="--", marker="",
        color=bc.red, label="3 Gyr")

    minorLocator = AutoMinorLocator()
    ax = plt.gca()
    ax.yaxis.set_minor_locator(minorLocator)
    plt.legend(loc="upper right")
    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("M_K")
    plt.title("APOGEE -0.1 < [Fe/H] < 0.1")

def MIST_metallicity_between_interpolation():
    '''Plot MIST isochrones interpolating over metallicity.'''
    age = 9e9
    feh_gridpoints = np.linspace(-0.5, 0.5, 4+1, endpoint=True)

    feh_models = np.linspace(-0.5, 0.5, 8+1, endpoint=True)
    teffpoints = np.linspace(3500, 6500, 500)
    kvals = samp.calc_model_mag_fixed_age_alpha(
        teffpoints, feh_models, "Ks", age=age, model="MIST")
    colors = [bc.black, bc.red, bc.green, bc.algae, bc.brown, bc.sky_blue,
              bc.purple, bc.blue, bc.orange]
    for i, (feh, c) in enumerate(zip(feh_models, colors)):
        hr.absmag_teff_plot(
            teffpoints, kvals[i, :], color=c, marker="", ls="-", 
            label="[Fe/H] = {0:.2f}".format(feh))
        if feh in feh_gridpoints:
            model = mist.MISTIsochrone.isochrone_from_file(feh)
            met_table = model.iso_table(age)
            hr.absmag_teff_plot(
                10**met_table[model.logteff_col],
                met_table[mist.band_translation["Ks"]], color=c, marker="o",
                ls="", label="")
    plt.xlabel("Teff (K)") 
    plt.ylabel("Ks")
    plt.legend(loc="lower left")

def MIST_age_difference_with_metallicity():
    '''Plot the age evolution for different metallicity targets.

    Show the original points as well just to make sure that there isn't
    weirdness with plotting.'''
    youngage = 1e9
    oldage=9e9
    feh_gridpoints = np.linspace(-0.5, 0.5, 4+1, endpoint=True)

    feh_models = np.linspace(-0.5, 0.5, 8+1, endpoint=True)
    teffpoints = np.linspace(3500, 6500, 500)
    masspoints = np.linspace(0.1, 2.0, 500)
    oldteffs = 10**samp.calc_model_over_feh_fixed_age_alpha(
        masspoints, mist.MISTIsochrone.mass_col,
        mist.MISTIsochrone.logteff_col, feh_models, oldage, model="MIST")
    oldks = samp.calc_model_over_feh_fixed_age_alpha(
        masspoints, mist.MISTIsochrone.mass_col,
        mist.band_translation["Ks"], feh_models, oldage, model="MIST")
    assert np.all(oldteffs.mask == oldks.mask)
    colors = [bc.black, bc.red, bc.green, bc.algae, bc.brown, bc.sky_blue,
              bc.purple, bc.blue, bc.orange]
    for i, (feh, c) in enumerate(zip(feh_models, colors)):
        iso_badmass = oldteffs[i,:].mask
        oldteffs_feh = oldteffs[i,:][~iso_badmass]
        oldks_feh = oldks[i,:][~iso_badmass]
        youngks_feh = samp.calc_model_mag_fixed_age_alpha(
            oldteffs_feh, feh, "Ks", age=youngage, model="MIST")
        hr.absmag_teff_plot(
            oldteffs_feh, oldks_feh-youngks_feh, color=c, marker="", ls="-", 
            label="[Fe/H] = {0:.2f}".format(feh))
        if feh in feh_gridpoints:
            oldmodel = mist.MISTIsochrone.isochrone_from_file(feh)
            met_table = oldmodel.iso_table(oldage)
            young_ys = samp.calc_model_mag_fixed_age_alpha(
                10**met_table[oldmodel.logteff_col], feh, "Ks", age=youngage,
                model="MIST")
            hr.absmag_teff_plot(
                10**met_table[oldmodel.logteff_col],
                met_table[mist.band_translation["Ks"]]-young_ys, color=c, 
                marker="o", ls="", label="")
    plt.xlabel("Teff (K)") 
    plt.ylabel("Ks (9 Gyr)- Ks (1 Gyr)")
    plt.legend(loc="lower left")


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

def plot_targs_with_parallaxes():
    '''Display the rapid rotators in a CMD.'''
    cool_dwarf = cool_data_splitter()
    fullsamp = cool_dwarf.subsample(["~Bad", "Dwarf", "~Too Hot"])
    dlsbs = cool_dwarf.subsample(["~Bad", "Dwarf", "~Too Hot", "DLSB"])
    marginal = cool_dwarf.subsample([
        "~Bad", "Vsini marginal", "~DLSB", "Mcq", "Dwarf", "~Too Hot"])
    detections = cool_dwarf.subsample([
        "~Bad", "Vsini det", "~DLSB", "Mcq", "Dwarf", "~Too Hot"])

    gaia = catin.read_Gaia_DR2_Kepler()
    gaia_targets = gaia
    samp_gaia = au.join_by_id(gaia_targets, fullsamp, "kepid", "kepid")
    samp_gaia["K_ABSMAG"] = (
        samp_gaia["K"] + 5 * np.log10(samp_gaia["parallax"]/100))
    dlsbs_gaia = au.join_by_id(
        samp_gaia, dlsbs, "kepid", "kepid", conflict_suffixes=("", "_copy"))
    marginal_gaia = au.join_by_id(samp_gaia, marginal, "kepid", "kepid",
                                  conflict_suffixes=("", "_copy"))
    detections_gaia = au.join_by_id(samp_gaia, detections, "kepid", "kepid",
                                    conflict_suffixes=("", "_copy"))

    mcq = catin.read_McQuillan_catalog()
    mcq_short = mcq[mcq["Prot"] < 5]
    mcq_gaia = au.join_by_id(samp_gaia, mcq_short, "kepid", "KIC")

    iso_lowmet = sed.DSEPInterpolator(3, -0.25)
    iso_highmet = sed.DSEPInterpolator(3, 0.25)
    isomean = sed.DSEPInterpolator(3, 0.0)
    dsepteffs = np.linspace(min(fullsamp["TEFF"]), max(fullsamp["TEFF"]), 50)
    dsepmags = isomean.teff_to_abs_mag(dsepteffs, "Ks")
    lowmetmags = iso_lowmet.teff_to_abs_mag(dsepteffs, "Ks")
    highmetmags = iso_highmet.teff_to_abs_mag(dsepteffs, "Ks")

    plt.plot(dsepteffs, dsepmags, color=bc.black, ls="-", 
             label="DSEP Fiducial")
    plt.plot(dsepteffs, dsepmags-2.5*np.log10(2), color=bc.black, ls="--", 
             label="DSEP Binary")
    plt.plot(dsepteffs, lowmetmags, color=bc.blue, ls="-", 
             label="[Fe/H] = -0.25")
    plt.plot(dsepteffs, highmetmags, color=bc.red, ls="-", 
             label="[Fe/H] = +0.25")

    plt.plot(
        samp_gaia["TEFF"], samp_gaia["K_ABSMAG"],
        color=bc.black, ls="None", marker=".", label="APOGEE Dwarfs")
    plt.plot(
        dlsbs_gaia["TEFF"], dlsbs_gaia["K_ABSMAG"],
        color=bc.purple, ls="None", marker="*", label="Known DLSBs", ms=10)
    plt.plot(
        marginal_gaia["TEFF"], marginal_gaia["K_ABSMAG"],
        color=bc.sky_blue, ls="None", marker="o", label="Marginal vsini")
    plt.plot(
        detections_gaia["TEFF"], detections_gaia["K_ABSMAG"],
        color=bc.blue, ls="None", marker="o", label="Vsini detection")
    plt.plot(
        mcq_gaia["TEFF"], mcq_gaia["K_ABSMAG"],
        color=bc.orange, ls="None", marker="^", label="Photometric rapid")
    print(len(samp_gaia))
    print(len(dlsbs_gaia))
    print(len(marginal_gaia))
    print(len(detections_gaia))
    print(len(mcq_gaia))

    hr.invert_x_axis()
    hr.invert_y_axis()

    plt.xlabel("TEFF")
    plt.ylabel("M_Ks")
    plt.legend(loc="upper right")


def plot_distances_with_targs():
    '''Display the rapid rotators in a CMD.'''
    cool_dwarf = cool_data_splitter()
    fullsamp = cool_dwarf.subsample(["~Bad", "Dwarf", "~Too Hot"])
    dlsbs = cool_dwarf.subsample(["~Bad", "Dwarf", "~Too Hot", "DLSB"])
    marginal = cool_dwarf.subsample([
        "~Bad", "Vsini marginal", "~DLSB", "Mcq", "Dwarf", "~Too Hot"])
    detections = cool_dwarf.subsample([
        "~Bad", "Vsini det", "~DLSB", "Mcq", "Dwarf", "~Too Hot"])

    gaia = catin.read_Gaia_DR2_Kepler()
    gaia_targets = gaia
    samp_gaia = au.join_by_id(gaia_targets, fullsamp, "kepid", "kepid")
    samp_gaia["K_ABSMAG"] = (
        samp_gaia["K"] + 5 * np.log10(samp_gaia["parallax"]/100))
    dlsbs_gaia = au.join_by_id(
        samp_gaia, dlsbs, "kepid", "kepid", conflict_suffixes=("", "_copy"))
    marginal_gaia = au.join_by_id(samp_gaia, marginal, "kepid", "kepid",
                                  conflict_suffixes=("", "_copy"))
    detections_gaia = au.join_by_id(samp_gaia, detections, "kepid", "kepid",
                                    conflict_suffixes=("", "_copy"))

    mcq = catin.read_McQuillan_catalog()
    mcq_short = mcq[mcq["Prot"] < 5]
    mcq_gaia = au.join_by_id(samp_gaia, mcq_short, "kepid", "KIC")

    iso_highmet = sed.DSEPInterpolator(3, 0.4)
    isomean = sed.DSEPInterpolator(3, 0.0)
    binaries = (samp_gaia["K_ABSMAG"] < 
               iso_highmet.teff_to_abs_mag(samp_gaia["TEFF"]))
    singles  = np.logical_not(binaries)

    plt.plot(
        samp_gaia["TEFF"][singles], 100/samp_gaia["parallax"][singles],
        color=bc.black, ls="None", marker=".", label="APOGEE Singles")
    plt.plot(
        samp_gaia["TEFF"][binaries], 100/samp_gaia["parallax"][binaries],
        color=bc.pink, ls="None", marker="8", label="APOGEE Binaries")
    plt.plot(
        dlsbs_gaia["TEFF"], 100/dlsbs_gaia["parallax"],
        color=bc.purple, ls="None", marker="*", label="Known DLSBs", ms=10)
    plt.plot(
        marginal_gaia["TEFF"], 100/marginal_gaia["parallax"],
        color=bc.sky_blue, ls="None", marker="o", label="Marginal vsini")
    plt.plot(
        detections_gaia["TEFF"], 100/detections_gaia["parallax"],
        color=bc.blue, ls="None", marker="o", label="Vsini detection")
    plt.plot(
        mcq_gaia["TEFF"], 100/mcq_gaia["parallax"],
        color=bc.orange, ls="None", marker="^", label="Photometric rapid")

    hr.invert_x_axis()

    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("Distance (Mpc)")
    plt.legend(loc="upper right")

def plot_targs_rv_scatter():
    '''Display the rapid rotators in a CMD.'''
    cool_dwarf = cool_data_splitter()
    fullsamp = cool_dwarf.subsample(["~Bad", "Dwarf", "~Too Hot"])
    dlsbs = cool_dwarf.subsample(["~Bad", "Dwarf", "~Too Hot", "DLSB"])
    marginal = cool_dwarf.subsample([
        "~Bad", "Vsini marginal", "~DLSB", "Mcq", "Dwarf", "~Too Hot"])
    detections = cool_dwarf.subsample([
        "~Bad", "Vsini det", "~DLSB", "Mcq", "Dwarf", "~Too Hot"])

    gaia = catin.read_Gaia_DR2_Kepler()
    gaia_targets = gaia
    samp_gaia = au.join_by_id(gaia_targets, fullsamp, "kepid", "kepid")
    samp_gaia["K_ABSMAG"] = (
        samp_gaia["K"] + 5 * np.log10(samp_gaia["parallax"]/100))
    dlsbs_gaia = au.join_by_id(
        samp_gaia, dlsbs, "kepid", "kepid", conflict_suffixes=("", "_copy"))
    marginal_gaia = au.join_by_id(samp_gaia, marginal, "kepid", "kepid",
                                  conflict_suffixes=("", "_copy"))
    detections_gaia = au.join_by_id(samp_gaia, detections, "kepid", "kepid",
                                    conflict_suffixes=("", "_copy"))

    mcq = catin.read_McQuillan_catalog()
    mcq_short = mcq[mcq["Prot"] < 5]
    mcq_gaia = au.join_by_id(samp_gaia, mcq_short, "kepid", "KIC")

    isomean = sed.DSEPInterpolator(3, 0.0)

    plt.plot(
        samp_gaia["radial_velocity_error"], 
        samp_gaia["K_ABSMAG"] - isomean.teff_to_abs_mag(samp_gaia["TEFF"], "Ks"), 
        color=bc.black, ls="None", marker=".", label="APOGEE Dwarfs")
    plt.plot(
        dlsbs_gaia["radial_velocity_error"], 
        dlsbs_gaia["K_ABSMAG"] - isomean.teff_to_abs_mag(
            dlsbs_gaia["TEFF"], "Ks"),
        color=bc.purple, ls="None", marker="*", label="Known DLSBs", ms=10)
    plt.plot(
        marginal_gaia["radial_velocity_error"],
        marginal_gaia["K_ABSMAG"] - isomean.teff_to_abs_mag(
            marginal_gaia["TEFF"], "Ks"),
        color=bc.sky_blue, ls="None", marker="o", label="Marginal vsini")
    plt.plot(
        detections_gaia["radial_velocity_error"],
        detections_gaia["K_ABSMAG"] - isomean.teff_to_abs_mag(
            detections_gaia["TEFF"], "Ks"),
        color=bc.blue, ls="None", marker="o", label="Vsini detection")
    plt.plot(
        mcq_gaia["radial_velocity_error"],
        mcq_gaia["K_ABSMAG"] - isomean.teff_to_abs_mag(
            mcq_gaia["TEFF"], "Ks"),
        color=bc.orange, ls="None", marker="^", label="Photometric rapid")

    hr.invert_y_axis()

    plt.xlabel("RV uncertainty (km/s)")
    plt.ylabel("Photometric excess")
    plt.legend(loc="upper right")

def rapid_rotator_HR_diagram(ax):
    apo_splitter = cache.apogee_splitter_with_DSEP()
    apo_splitter.split_teff("TEFF", [5500], splitnames=("Cool", "Hot"),
                            null_value=None, teff_crit="Age evolution Teff")
    apo_splitter.split_vsini(
        [10], ("Vsini nondet", "Vsini det", "No Vsini"),
        null_value=np.ma.masked)
    apo_full = apo_splitter.subsample(["Dwarfs", "Cool"])
    apo_rapid = apo_splitter.subsample([
        "Dwarfs", "Cool", "Vsini det", "~DLSB"])
    apo_dlsb = apo_splitter.subsample(["Dwarfs", "Cool", "DLSB"])
    mcq = catin.read_McQuillan_catalog()
    apo_mcq = au.join_by_id(apo_full, mcq, "kepid", "KIC")
    apo_mcq_rapid = apo_mcq[apo_mcq["Prot"] < 3]
    print(len(apo_rapid)/len(apo_full))

    # Full sample
    hr.absmag_teff_plot(apo_full["TEFF"], apo_full["M_K"], color=bc.black,
                        marker=".", ls="", label="APOGEE")
    # Vsini Rapid rotator
    hr.absmag_teff_plot(apo_rapid["TEFF"], apo_rapid["M_K"], color=bc.blue,
                        marker="o", ls="", label="Vsini > 10 km/s")
    # DLSB
    hr.absmag_teff_plot(apo_dlsb["TEFF"], apo_dlsb["M_K"], color=bc.pink,
                        marker="*", ls="", label="SB2")
    # McQuillan rapid rotators
    hr.absmag_teff_plot(
        apo_mcq_rapid["TEFF"], apo_mcq_rapid["M_K"], color=bc.red, marker="x", 
        ls="", label="P < 3 day", ms=7)
    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("$M_K$")
    plt.legend(loc="lower left")



def aspcap_telecon_plot():
    '''Want two plots: rapid rotators on HR diagram. And the veq vs vsini
    plot.'''

if __name__ == "__main__":

    desc = """Generate figures and tables.

Script to automatically generate the figures and tables needed for the paper on
looking for tidally-synchronized binaries in the Kepler field."""

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("figs", nargs="*")
    parser.add_argument("--list-figs", action="store_true")
    
    args = parser.parse_args()

    figlist = {
        "Bruntt": Bruntt_vsini_comparison,
        "SH": Pleiades_vsini_comparison,
        "asterosamp": asteroseismic_sample_MK,
        "coolsamp": cool_dwarf_mk,
        "asterorot": asteroseismic_rotation_analysis,
        "coolrot": cool_dwarf_rotation_analysis,
        "rrfracs": plot_rr_fractions,
        "binarity": plot_binarity_diagram
    }

    genfigs = args.figs
    if not genfigs:
        print(figlist.keys())
    for figname in genfigs:
        if figname == "all":
            for fig in figlist.values():
                fig()
            break
        else:
            figlist[figname]()
