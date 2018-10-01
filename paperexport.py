import os
import sys
import argparse
import functools

import numpy as np
import numpy.core.defchararray as npstr
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.table import Table, vstack
from astropy.io import ascii
from scipy.integrate import quad

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

PAPER_PATH = paths.HOME_DIR / "papers" / "rotation17"
TABLE_PATH = PAPER_PATH / "tables"
FIGURE_PATH = PAPER_PATH / "fig"
PLOT_SUFFIX = "png"
PLOT_PATH = paths.HEAD_DIR / "plots"

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

############
# Pleiades #
############

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

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    iso = baraffe.BaraffeIsochrone.isochrone_from_file()
    baraffe_rad = iso.interpolate_isochrone_cols(
        0.12, pleiades_vsini["TEFF"], iso.teff_col, iso.radius_col)
    baraffe_rad_err = pleiades_vsini["TEFF_ERR"] * iso.isochrone_derivative(
        0.12, pleiades_vsini["TEFF"], iso.teff_col, iso.radius_col)
    rot.plot_vsini_velocity(
        pleiades_vsini["VSINI"], pleiades_vsini["Per1"],
        pleiades_vsini["Per1"]*0.07, baraffe_rad, baraffe_rad_err, ax=ax)
    ax.plot([0, 70], [0, 100], ls='-.', c='k', marker="")
    ax.set_title("Baraffe Direct Radius")

def Pleiades_vsini_agreement_RStauffer():
    '''Plot the vsini vs veq diagram for the Pleiades.
    
    In this figure, the radius is not derived from MIST isochrones, but rather
    from the K-band absolute magnitude.'''
    pleiades = cache.pleiades()
    pleiades_vsini = pleiades[
        np.logical_and(~pleiades["VSINI"].mask, pleiades["TEFF"] < 5500)]

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
        pleiades_vsini["Per1"]*0.07, apogee_rad, apogee_rad_err, ax=ax)
    ax.plot([0, 70], [0, 100], ls='-.', c='k', marker="")
    ax.set_title("Radius from Deprojected mass")

def Pleiades_RMIST_Rbol_comparison():
    '''Compare the radii from MIST and from bolometric luminosities.'''
    pleiades = cache.pleiades()
    ok = pleiades["memb"] == "ok"

def Pleiades_RMIST_Rbol_comparison():
    '''Compare the radii from MIST and from bolometric luminosities.'''
    pleiades = cache.pleiades()
    ok = pleiades["memb"] == "ok"

    f, ax = plt.subplots(1, 1, figsize=(12, 12))

    ax.plot(pleiades["MIST R"], pleiades["K-band R"], color='k', marker='o',
            ls="", label="K")
    ax.plot(pleiades["MIST R"], pleiades["V-band R"], color='k', marker='.', ls="", alpha=0.5,
            label="V")
    ax.plot([0.3, 1.5], [0.3, 1.5], 'k-')
    ax.set_xlabel("MIST R")
    ax.set_ylabel("Bol R")
    ax.legend(loc="lower right")

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

def Pleiades_Rebull_vsini_comparison():
    '''Compare the APOGEE vsini to the vsini collected from Rebull.'''
    pleiades = cache.pleiades()
    discrepant_targets = np.logical_and(
        ~pleiades["vsini"].mask, 
        pleiades["VSINI"] > 1.5 * np.maximum(pleiades["vsini"], 7))
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12)) 
    ax1.plot(pleiades["vsini"], pleiades["VSINI"], 'k.')
    ax1.plot(pleiades["vsini"][discrepant_targets],
            pleiades["VSINI"][discrepant_targets], 'bo')
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
    print(pleiades["VSCATTER"][discrepant_targets])

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



                        

###########
# Periods #
###########

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
    print(len(overlap))

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
