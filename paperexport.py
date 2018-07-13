import os
import sys
import argparse

import numpy as np
import numpy.core.defchararray as npstr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from astropy.table import Table, vstack
from astropy.io import ascii
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

@au.memoized
def asteroseismic_data_splitter():
    '''A persistent datasplitter for the asteroseismic sample.'''
    astero = split.APOKASCSplitter()
    split.initialize_asteroseismic_sample(astero)
    return astero

@au.memoized
def full_apogee_splitter():
    '''A persistent DataSplitter for the cool dwarf sample.'''
    apogee = split.APOGEESplitter()
    # No need to split off objects w/ and w/o Gaia detections
    split.initialize_full_APOGEE(apogee)
    targeted_splitted = apogee.split_subsample(["Targeted"])
    return targeted_splitted

@au.memoized
def cool_data_splitter():
    '''A persistent Datasplitter for the cool sample.'''
    full = full_apogee_splitter()
    cool = full.split_subsample(["Cool", "H Jen"])
    split.initialize_cool_KICs(cool)
    return cool

@au.memoized
def hot_dwarf_splitter():
    '''A persistent Datasplitter for the hot dwarf sample.'''
    full = full_apogee_splitter()
    hot = full.split_subsample(["Hot"])
    return hot

def missing_gaia_targets():
    '''Plot properties of targets missing Gaia observations.'''
    full = full_apogee_splitter()
    full_data = full.subsample(["~Bad"])
    missing_gaia = full_data[full_data["dis"].mask]

    hr.logg_teff_plot(full_data["teff"], full_data["logg"], marker=".",
                        color=bc.black, ls="")
    hr.logg_teff_plot(missing_gaia["teff"], missing_gaia["logg"], marker=".",
                        color=bc.green, ls="")




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

@write_plot("selection")
def selection_coordinates():
    '''Show the APOKASC and Ancillary samples in selection coordinates.'''
    fullsamp = catin.stelparms_with_original_KIC()
    aposamp = full_apogee_splitter()

    f, ax = plt.subplots(1,1, figsize=(12,12))
    apokasc_giant = aposamp.subsample([
        "APOGEE2_APOKASC", "~APOGEE2_APOKASC_DWARF", "~Cool Sample"])
    apokasc_dwarf = aposamp.subsample(["APOGEE2_APOKASC_DWARF", "~Cool Sample"])
    jen = aposamp.subsample(["Cool Sample"])

    hr.logg_teff_plot(
        fullsamp["teff"], fullsamp["logg"], color=bc.black, marker="o", ls="",
    label="Full Kepler")
    hr.logg_teff_plot(
        apokasc_giant["teff"], apokasc_giant["logg"], color=bc.green, 
        marker=".", ls="", label="APOKASC Giant")
    hr.logg_teff_plot(
        apokasc_dwarf["teff"], apokasc_dwarf["logg"], color=bc.blue, marker=".", 
        ls="", label="APOKASC Dwarf")
    hr.logg_teff_plot(
        jen["teff"], jen["logg"], color=bc.red, marker=".", ls="", 
        label="Cool Dwarf")
    plt.xlim(7000, 3000)
    plt.xlabel("Huber Teff (K)")
    plt.ylabel("Huber log(g) (cm/s/s)")
    plt.legend(loc="upper left")

@write_plot("metallicity")
def dwarf_metallicity():
    '''Show the metallicity distribution of the cool dwarfs.'''
    cool = cool_data_splitter()
    cool_data = cool.subsample(["~Bad", "Dwarf"])

    f, ax = plt.subplots(1,1, figsize=(12,12))
    ax.hist(cool_data["M_H"], cumulative=True, normed=True, bins=100)
    ax.set_xlabel("APOGEE [M/H]")
    ax.set_ylabel("Cumulative distribution")
    ax.set_xlim(-1.5, 0.46)


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

def full_sample_mk():
    '''Plot the full sample on an HR diagram with M_K'''
    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    full = full_apogee_splitter()
    full_data = full.subsample(["In Gaia", "K Detection"])

    hr.absmag_teff_plot(full_data["TEFF"], full_data["M_K"], color=bc.black,
                        marker=".", ls="")

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


@write_plot("astero_rot")
def asteroseismic_rotation_analysis():
    '''Plot rotation comparison of asteroseismic sample.'''
    astero = asteroseismic_data_splitter()
    marginal = astero.subsample([
        "~Bad", "Asteroseismic Dwarfs", "Vsini marginal", "~DLSB", "Mcq"])
    detections = astero.subsample([
        "~Bad", "Asteroseismic Dwarfs", "Vsini det", "~DLSB", "Mcq"])

    mcq = catin.read_McQuillan_catalog()
    marginal_periodpoints = au.join_by_id(marginal, mcq, "kepid", "KIC")
    detections_periodpoints = au.join_by_id(detections, mcq, "kepid", "KIC")

    f, ax = plt.subplots(1,1, figsize=(10,10))

    rot.plot_vsini_velocity(
        marginal_periodpoints["VSINI"], marginal_periodpoints["Prot"],
        marginal_periodpoints["radius"], 
        raderr_below=marginal_periodpoints["radius_err1"],
        raderr_above=marginal_periodpoints["radius_err2"], color=bc.sky_blue,
        ax=ax, sini_label=False)

    rot.plot_vsini_velocity(
        detections_periodpoints["VSINI"], detections_periodpoints["Prot"],
        detections_periodpoints["radius"], 
        raderr_below=detections_periodpoints["radius_err1"],
        raderr_above=detections_periodpoints["radius_err2"], color=bc.blue,
        ax=ax)

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

@write_plot("binary_cut")
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

    fullsamp = vstack([
        mcq.subsample(["Right Teff", "Berger Main Sequence"]), 
        mcq.subsample(["Right Teff", "Berger Cool Binary"])])
    rapid = vstack([
        mcq.subsample(["Right Teff", "Berger Main Sequence", "~Slow"]), 
        mcq.subsample(["Right Teff", "Berger Cool Binary", "~Slow"])])

    full_magdiff = samp.calc_photometric_excess(
        fullsamp["teff"], 0.0, 0.0, "Ks", fullsamp["M_K"])
    rapid_magdiff = samp.calc_photometric_excess(
        rapid["teff"], 0.0, 0.0, "Ks", rapid["M_K"])
    print(len(full_magdiff))
    print(len(rapid_magdiff))

    f = plt.figure(figsize=(10,10))
    bins = np.linspace(-4.5, 3.5, 76, endpoint=True)
    plt.hist(full_magdiff, normed=True, bins=bins, histtype="step",
             cumulative=-1, label="Full Mcquillan", color=bc.black)
    plt.hist(rapid_magdiff, normed=True, bins=bins, histtype="step",
             cumulative=-1, label="Rapid Rotators", color=bc.blue)
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

def plot_k_excess_rotation_apogee():
    '''Plot the K-band excess against rotation for McQuillan targets.'''
    cools = cool_data_splitter()
    cools_mk = cools.split_subsample(["In Gaia", "K Detection"])
    cools_mk.split_teff("TEFF", [3500, 5500], ("Low MS", "MS", "High MS"), 
                     teff_crit="MS Split")
    cools_mk.split_mag("M_K", 2.95, splitnames=("High", "Low"), 
                    mag_crit="M_K split")
    cool_data = cools_mk.subsample(["~Bad", "MS", "Low"])
    
    fullsamp = cools_mk.subsample(["~Bad", "MS", "Low"])

    full_magdiff = samp.calc_photometric_excess(
        fullsamp["TEFF"], fullsamp["FE_H"], fullsamp["ALPHA_FE"], "Ks",
        fullsamp["M_K"])

#    plt.plot(full_magdiff, fullsamp["Prot"], 'k.')
    plt.plot(fullsamp["VSINI"], full_magdiff, 'k.')
    plt.plot([0, 75], [0, 0], 'k--')
    plt.ylabel("Mag Excess")
    plt.xlabel("vsini (km/s)")
    plt.title("APOGEE Rapid Rotator Excess")
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
