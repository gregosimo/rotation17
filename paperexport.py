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

@au.memoized
def hot_dwarf_splitter():
    '''A persistent Datasplitter for the hot dwarf sample.'''
    dwarfs = dwarf_data_splitter()
    hot_dwarfs = dwarfs.split_subsample(["APOGEE2_APOKASC_DWARF"])
    return hot_dwarfs

@write_plot("Bruntt_comp")
def Bruntt_vsini_comparison():
    '''Create figure comparison vsini for asteroseismic targets.

    The APOGEE vsinis are compared against the Bruntt et al (2012)
    spectroscopic vsinis for a set of asteroseismic targets.'''

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
    plt.loglog(astero_dwarfs["vsini"][~bad_indices], 
               astero_dwarfs["VSINI"][~bad_indices], 'ko')
    plt.loglog(astero_dwarfs["vsini"][bad_indices],
             np.zeros(np.count_nonzero(bad_indices)), 'rx')
    detection_lim = 7
    plt.loglog([detection_lim, detection_lim], [1, 40], 'k:')

    # Now to fit the data to a line.
    detected_table = astero_dwarfs[astero_dwarfs["VSINI"] >= detection_lim]
    meanx = np.mean(np.log10(detected_table["vsini"]))
    fitval, cov = np.polyfit(
        np.log10(detected_table["vsini"])-meanx, np.log10(detected_table["VSINI"]), 1, 
        cov=True)
    polyeval = np.poly1d(fitval)
    polyx = np.log10(np.linspace(1, 40, 10)) - meanx
    polyy = 10**polyeval(polyx)
    plt.loglog(10**(polyx+meanx), polyy, 'k--', linewidth=3)
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
#   plt.fill_between(
#       10**(polyx+meanx), polyy*10**(offset/2), polyy*10**(-offset/2),
#       facecolor="gray")
#   outlier = astero_dwarfs[~bad_indices][np.abs(vsini_diff) > 1.0]
#   assert len(outlier) == 1
#   print("Ignoring KIC{0:d}: Bruntt vsini = {1:.1f}, ASPCAP vsini = {2:.1f}".format(
#       outlier["kepid"][0], outlier["vsini"][0], outlier["VSINI"][0]))
    print("Bad objects:")
    print(astero_dwarfs[["KIC", "vsini"]][bad_indices])
    plt.xlabel("Bruntt vsini (km/s)")
    plt.ylabel("APOGEE vsini (km/s)")
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
    targets = catin.Stauffer_APOGEE_overlap()
    non_dlsbs = targets[npstr.find(targets["Notes"], "5") < 0]
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
                   
    plt.xscale("log")
    plt.yscale("log")
    weird_targets = np.logical_and(detected_targets["VSINI"] < 10,
                                   detected_targets["vsini"] > 10)
    plt.errorbar(detected_targets["vsini"], detected_targets["VSINI"], 
                 xerr=detected_errors, fmt='ko')
    plt.plot(nondet_targets["vsini"], nondet_targets["VSINI"], 'r<')
    one_to_one = np.array([1, 100])
    plt.plot(one_to_one, one_to_one, 'k-')
    plt.plot([12, 12], one_to_one, 'k:')
    
    plt.xlabel("Stauffer and Hartmann (1987) vsini (km/s)")
    plt.ylabel("APOGEE vsini (km/s)")
    plt.xlim(1, 100)
    plt.ylim(1, 100)


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

@write_plot("astero")
def asteroseismic_sample_MK():
    '''Write an HR diagram consisting of the asteroseismic sample.
    
    Plot the asteroseismic and spectroscopic log(g) values separately. Should
    also plot the underlying APOKASC sample.'''
    apokasc = asteroseismic_data_splitter()
    apokasc.split_targeting("APOGEE2_APOKASC_DWARF")
    fulldata = apokasc.subsample(["~Bad", "APOGEE2_APOKASC_DWARF"])
    astero = apokasc.subsample(["~Bad", "Asteroseismic Dwarfs"])

    hr.absmag_teff_plot(
        fulldata["TEFF_COR"], fulldata["M_K"], color=bc.black, marker=".",
        label="APOGEE Hot Sample", ls="")
    hr.logg_teff_plot(astero["TEFF_COR"], astero["M_K"], color=bc.green,
                      marker="*", ms=8, label="Asteroseismic sample", ls="")

    # Let's throw some isochrones here.
    iso1GyrM00 = sed.DSEPInterpolator(
        1, 0.0, highT=7000, lowT=3000, minlogG=1.0)
    iso10GyrM00 = sed.DSEPInterpolator(
        1, 0.0, highT=7000, lowT=3000, minlogG=1.0)
    iso1GyrM05 = sed.DSEPInterpolator(
        1, 0.5, highT=7000, lowT=3000, minlogG=1.0)
    iso10GyrM05 = sed.DSEPInterpolator(
        1, 0.5, highT=7000, lowT=3000, minlogG=1.0)
    iso1GyrMm05 = sed.DSEPInterpolator(
        1, -0.5, highT=7000, lowT=3000, minlogG=1.0)
    iso10GyrMm05 = sed.DSEPInterpolator(
        1, -0.5, highT=7000, lowT=3000, minlogG=1.0)

    fullmasses = np.linspace(0.4, 1.4, 1000)
    Teff1GyrM00 = iso1GyrM00.mass_to_teff(fullmasses)
    MK1GyrM00 = iso1GyrM00.mass_to_abs_mag(fullmasses, "Ks")
    plt.plot(Teff1GyrM00, MK1GyrM00, 'r-')

    Teff10GyrM00 = iso10GyrM00.mass_to_teff(fullmasses[fullmasses < 1.043])
    MK10GyrM00 = iso10GyrM00.mass_to_abs_mag(fullmasses[fullmasses < 1.043], "Ks")
    plt.figure()
    plt.plot(fullmasses[fullmasses < 1.043], Teff10GyrM00, 'k-')
    plt.figure()
    plt.plot(fullmasses[fullmasses < 1.043], MK10GyrM00, 'k-')
    plt.figure()
    print(fullmasses[fullmasses < 1.043])
    print(Teff10GyrM00)
    print(MK10GyrM00)
    plt.plot(Teff10GyrM00, MK10GyrM00, 'c-')

    

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

    cool_mcq = cool.subsample(["~Bad", "Mcq"]) 
    mcq = catin.read_McQuillan_catalog()
    mcq_joined = au.join_by_id(cool_mcq, mcq, "kepid", "KIC")
    rapid_rotators = mcq_joined[mcq_joined["Prot"] < 3]

    hr.absmag_teff_plot(cool_giants["TEFF"], cool_giants["M_K"], 
                      color=bc.orange, marker=".", label="Giants", ls="")
    hr.absmag_teff_plot(cool_subgiants["TEFF"], cool_subgiants["M_K"], 
                      color=bc.purple, marker=".", label="Subgiants", ls="")
    hr.absmag_teff_plot(cool_dwarfs["TEFF"], cool_dwarfs["M_K"], 
                      color=bc.algae, marker=".", label="Dwarfs", ls="")
    hr.absmag_teff_plot(cool_binaries["TEFF"], cool_binaries["M_K"], 
                      color=bc.green, marker=".", label="Binaries", ls="")
    hr.absmag_teff_plot(cool_rapid["TEFF"], cool_rapid["M_K"], 
                      color=bc.blue, marker="o", label="Rapid Rotators",
                      ls="", ms=7)
    hr.absmag_teff_plot(cool_marginal["TEFF"], cool_marginal["M_K"], 
                      color=bc.sky_blue, marker="o", label="Rapid Rotators",
                      ls="", ms=7)
    hr.absmag_teff_plot(rapid_rotators["TEFF"], rapid_rotators["M_K"],
                        color=bc.violet, marker="d", label="P < 3 day", ls="")
    hr.absmag_teff_plot(dlsbs["TEFF"], dlsbs["M_K"], color=bc.light_pink, marker="*",
                        label="SB2", ls="")
    hr.absmag_teff_plot(apogee_misclassified_subgiants["TEFF"],
                        apogee_misclassified_subgiants["M_K"], color=bc.red,
                        marker="x", label="Mismatched Spec. Ev. State", ls="",
                        ms=7)
    hr.absmag_teff_plot(apogee_misclassified_dwarfs["TEFF"],
                        apogee_misclassified_dwarfs["M_K"], color=bc.red,
                        marker="x", ls="", label="", ms=7)

    plt.plot([5750, 3500], [2.4, 2.4], 'k-')
    plt.plot([5750, 3500], [0.7, 0.7], 'k-')
    plt.plot([5450, 5450], [6, -8], 'k:')
    plt.xlim(5750, 3500)
    plt.ylim(6, -8)
    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel(r"$M_K$")
    plt.legend(loc="upper left")

    print("APOGEE-classified dwarfs: {0:d}".format(cool.subsample_len(
        ["~Bad", "Subgiant", "APOGEE Dwarf"])))
    print("Misclassified Rapid Rotators: {0:d}".format(cool.subsample_len(
        ["~Bad", "Subgiant", "APOGEE Dwarf", "Vsini det", "~DLSB"])))
    print("APOGEE-classified subgiants: {0:d}".format(cool.subsample_len(
        ["~Bad", "Dwarf", "APOGEE Subgiant", "~DLSB"])))
    print("Misclassified Rapid Rotators: {0:d}".format(cool.subsample_len(
        ["~Bad", "Dwarf", "APOGEE Subgiant", "Vsini det", "~DLSB"])))

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
    cool_dwarfs_mcq = cool_data.subsample([
        "~Bad", "~DLSB", "Mcq", "Dwarf", "~Too Hot"])
    cool_dwarfs_nomcq = cool_data.subsample([
        "~Bad", "~DLSB", "~Mcq", "Dwarf", "~Too Hot"])

    mcq = catin.read_McQuillan_catalog()
    periodpoints = au.join_by_id(cool_dwarfs_mcq, mcq, "kepid", "KIC")

    samp.generate_DSEP_radius_column_with_errors(periodpoints)

    samp.spectroscopic_photometric_rotation_fraction_comparison_plot(
        periodpoints["VSINI"], periodpoints["Prot"], 
        periodpoints["DSEP radius"], min_limit=5, max_limit=15)
    samp.plot_rapid_rotation_detection_limits(
        cool_dwarfs_nomcq["VSINI"], label="Mcquillan Nondetections",
        color=bc.black, ls="--", min_limit=5, max_limit=15) 
    plt.ylim(0.7, 1.0)
    plt.legend(loc="lower right")


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

    f, ax = plt.subplots(1,1, figsize=(6,6))

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
    marginal = cool_dwarf.subsample([
        "~Bad", "Vsini marginal", "~DLSB", "Mcq", "Dwarf", "~Too Hot"])
    detections = cool_dwarf.subsample([
        "~Bad", "Vsini det", "~DLSB", "Mcq", "Dwarf", "~Too Hot"])

    mcq = catin.read_McQuillan_catalog()
    marginal_periodpoints = au.join_by_id(marginal, mcq, "kepid", "KIC")
    detections_periodpoints = au.join_by_id(detections, mcq, "kepid", "KIC")

    samp.generate_DSEP_radius_column_with_errors(marginal_periodpoints)
    samp.generate_DSEP_radius_column_with_errors(detections_periodpoints)

    subplot_tup = rot.plot_rotation_velocity_radius(
        detections_periodpoints["VSINI"], detections_periodpoints["Prot"], 
        detections_periodpoints["DSEP radius"],
        raderr_below=detections_periodpoints["DSEP radius lower"],
        raderr_above=detections_periodpoints["DSEP radius upper"],
        color=bc.blue, label="Cool dwarfs") 

    rot.plot_rotation_velocity_radius(
        marginal_periodpoints["VSINI"], marginal_periodpoints["Prot"], 
        marginal_periodpoints["DSEP radius"],
        raderr_below=marginal_periodpoints["DSEP radius lower"],
        raderr_above=marginal_periodpoints["DSEP radius upper"],
        subplot_tup=subplot_tup, color=bc.sky_blue)
        

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
        "asterosamp": asteroseismic_sample_loggs,
        "coolsamp": cool_dwarf_hr,
        "asterorot": asteroseismic_rotation_analysis,
        "coolrot": cool_dwarf_rotation_analysis,
        "rrfracs": plot_rr_fractions
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
