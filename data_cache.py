import numpy as np
import astropy_util as au
from astropy.table import vstack, Table

import data_splitting as split
import sample_characterization as samp
import read_catalog as catin

@au.memoized
def asteroseismic_data_splitter():
    '''A persistent datasplitter for the asteroseismic sample.'''
    astero = split.APOKASCSplitter()
    split.initialize_asteroseismic_sample(astero)
    return astero

def full_apogee_splitter():
    '''A persistent DataSplitter for the cool dwarf sample.'''
    apogee = split.APOGEESplitter()
    # No need to split off objects w/ and w/o Gaia detections
    split.initialize_full_APOGEE(apogee)
    targeted_splitted = apogee.split_subsample(["Targeted"])
    return targeted_splitted

def clean_apogee_splitter():
    '''A persistent DataSplitter that can be used for isochrones. Nohelp.'''
    full = full_apogee_splitter()
    split.initialize_clean_APOGEE(full)
    clean = full.split_subsample([
        "~Bad", "~Bad APOGEE Teff", "K Detection", "In Gaia", "~No Met"])
    # Add Temperature uncertainties from Holtzmann et al (2018).
    giants = clean.data["LOGG_FIT"] < 2 + 2 / 1500 * (clean.data["TEFF"] - 4500)
    giant_err = np.exp(
        4.3609 + 0.000604303*(clean.data["TEFF"]-4500) -0.00196400 *
        clean.data["M_H"] -0.0659445 * (clean.data["SNREV"]-100))
    dwarf_err = np.exp(
        4.58343+ 0.000289796*(clean.data["TEFF"]-4500) -0.00129746 *
        clean.data["M_H"] -0.2434860 * (clean.data["SNREV"]-100)*0)
    clean.data["TEFF_ERR"] = np.where(giants, giant_err, dwarf_err)
    return clean

@au.memoized
def apogee_splitter_with_DSEP():
    '''A datasplitter with DSEP isochrones included. Help!'''
    clean = clean_apogee_splitter()
    clean.data["MIST K"] = np.diag(samp.calc_model_mag_fixed_age_alpha(
        clean.data["TEFF"], clean.data["FE_H"], "Ks", age=1e9, model="MIST"))
    toohigh_met = clean.data["FE_H"] > 0.5
    clean.data["MIST K"][toohigh_met] = samp.calc_model_mag_fixed_age_alpha(
        clean.data["TEFF"][toohigh_met], 0.5, "Ks", age=1e9, model="MIST")
    # I can add another gridpoint to the MIST isochrones in lieu of this.
    toolow_met = clean.data["FE_H"] < -2.5
    clean.data["MIST K"][toolow_met] = samp.calc_model_mag_fixed_age_alpha(
        clean.data["TEFF"][toolow_met], -2.5, "Ks", age=1e9, model="MIST")
    # Add the uncertainty in the MIST K. I don't want to deal with arbitrary
    # metallicities right now, so solar metallicity is good enough.
    # Since I don't care about the giants, I'm just going to assume everything
    # is a dwarf.
    clean.data["MIST K Error"] = samp.calc_model_mag_err_fixed_age_feh_alpha(
        clean.data["TEFF"], 0.0, "Ks", teff_err=clean.data["TEFF_ERR"], age=1e9, 
        model="MIST")
    clean.data["K Excess"] = clean.data["M_K"] - clean.data["MIST K"] 
    clean.data["K Excess Error Down"] = np.sqrt(
        clean.data["M_K_err2"]**2 + clean.data["MIST K Error"]**2)
    clean.data["K Excess Error Up"] = np.sqrt(
        clean.data["M_K_err1"]**2 + clean.data["MIST K Error"]**2)
    clean.split_mag("K Excess", -1.3, splitnames=("Not Dwarfs", "Dwarfs"),
                    null_value=None)
    # Remove tilt in metallicity space.
    dwarfs = clean.subsample(["Dwarfs", "APOGEE MetCor Teff"])
    polycoeff = samp.flatten_MS_metallicity(
        dwarfs["K Excess"], dwarfs["FE_H"], deg=2)
    polycorrect = np.poly1d(polycoeff)
    clean.data["Partly Corrected K Excess"] = (
        clean.data["K Excess"] - polycorrect(clean.data["FE_H"]))
    dwarfs = clean.subsample(["Dwarfs", "APOGEE MetCor Teff"])
    tempcoeff = samp.flatten_MS_metallicity(
        dwarfs["Partly Corrected K Excess"], dwarfs["TEFF"])
    tempcorrect = np.poly1d(tempcoeff)
    clean.data["Corrected K Excess"] = (
        clean.data["Partly Corrected K Excess"] - tempcorrect(clean.data["TEFF"]))

    # I will add in solar K values.
    clean.data["Solar K"] = samp.calc_model_mag_fixed_age_alpha(
        clean.data["TEFF"], 0.08, "Ks", age=1e9, model="MIST")
    clean.data["Solar K Excess"] = (
        clean.data["M_K"] - clean.data["Solar K"] - polycorrect(0.08))
    dwarfs = clean.subsample(["Dwarfs", "APOGEE MetCor Teff"])
    solar_coeff = samp.flatten_MS_metallicity(
        dwarfs["Solar K Excess"], dwarfs["TEFF"], deg=1)
    solar_correct = np.poly1d(solar_coeff)
    clean.data["Corrected K Solar"] = (
        clean.data["Solar K Excess"] - solar_correct(clean.data["TEFF"]))

    # Now add in solar K values using photometric temperatures.
    # This is tricky because not all APOGEE objects have photometric
    # temperatures. I want to calculate K magnitudes for objects that do, then
    # remask them.
    # First make a splitter of objects with good photometric temperatures.
    good_sdss_teff = ~clean.data["SDSS-Teff"].mask
    clean.data["Phot Teff K"] = np.ma.ones(len(clean.data))*-9999.0
    clean.data["Phot Teff K"][good_sdss_teff] = (
        samp.calc_model_mag_fixed_age_alpha(
            clean.data["SDSS-Teff"][good_sdss_teff], 0.08, "Ks", age=1e9, 
            model="MIST"))
    clean.data["Phot Teff K"] = np.ma.masked_values(
        clean.data["Phot Teff K"], -9999.0)
    clean.data["Phot Teff K Excess"] = (
        clean.data["M_K"] - clean.data["Phot Teff K"] - polycorrect(0.08))
    # Polyfit doesn't play well with masked values, so leave those out for the
    # fit. Dwarfs should now officially have no bad Pinsonneault objects.
    dwarfs = clean.subsample([
        "Dwarfs", "APOGEE MetCor Teff", "~No Pinsonneault MetCor"])
    solar_coeff = samp.flatten_MS_metallicity(
        dwarfs["Phot Teff K Excess"], dwarfs["SDSS-Teff"], deg=1)
    solar_correct = np.poly1d(solar_coeff)
    # Since the poly1d doesn't deal with masks well, just go around them.
    clean.data["Corrected Phot Teff K Excess"] = np.ma.ones(
        len(clean.data))*-9999.0
    clean.data["Corrected Phot Teff K Excess"][good_sdss_teff] = (
        clean.data["Phot Teff K Excess"][good_sdss_teff] - 
        solar_correct(clean.data["SDSS-Teff"][good_sdss_teff]))
    clean.data["Corrected Phot Teff K Excess"] = np.ma.masked_values(
        clean.data["Corrected Phot Teff K Excess"], -9999.0)

    # Now do a split based solely on the El-Badry temperatures.
    # Since only a subset of these objects have El-Badry temperatues, I need to
    # be wary of masked  values.
    elbadry_teff_indices = ~clean.data["T_eff [K]"].mask
    clean.data["ElBadry K"] = np.ma.ones(len(clean.data))*-9999.0
    clean.data["ElBadry K"][elbadry_teff_indices] = np.diag(
        samp.calc_model_mag_fixed_age_alpha(
            clean.data["T_eff [K]"][elbadry_teff_indices], 
            clean.data["[Fe/H] [dex]"][elbadry_teff_indices], "Ks", age=1e9, 
            model="MIST"))
    clean.data["ElBadry K"] = np.ma.masked_values(
        clean.data["ElBadry K"], -9999.0)
    clean.data["ElBadry K Excess"] = (
        clean.data["M_K"] - clean.data["ElBadry K"] - 
        polycorrect(clean.data["[Fe/H] [dex]"]))
    # Since the El-Badry temperatures should be identical to the APOGEE ones,
    # use the APOGEE fit instead.
    clean.data["Corrected ElBadry K Excess"] = (
        clean.data["ElBadry K Excess"] - tempcorrect(
            clean.data["T_eff [K]"]))

    return clean

@au.memoized
def cool_data_splitter():
    '''A persistent Datasplitter for the cool sample.'''
    full = full_apogee_splitter()
    cool = full.split_subsample(["Cool", "H APOGEE"])
    split.initialize_cool_KICs(cool)
    return cool

@au.memoized
def hot_dwarf_splitter():
    '''A persistent Datasplitter for the hot dwarf sample.'''
    full = full_apogee_splitter()
    hot = full.split_subsample(["Hot"])
    return hot

def mcquillan_splitter():
    mcq = split.McQuillanSplitter()
    split.initialize_mcquillan_sample(mcq)
    return mcq

def clean_mcquillan_splitter():
    '''A persistent DataSplitter that can be used for isochrones.'''
    mcq = mcquillan_splitter()
    clean = mcq.split_subsample([
        "K Detection", "In Gaia", "Good Isochrone Teff"])
    return clean

def mcquillan_splitter_with_DSEP():
    clean = clean_mcquillan_splitter()
    # One of the McQuillan has a Teff of 7300, which is too hot even for the 1 
    # Gyr isochrone.
    clean.data["MIST K"] = samp.calc_model_mag_fixed_age_alpha(
        clean.data["SDSS-Teff"], 0.08, "Ks", age=1e9, model="MIST")
    clean.data["MIST K Error"] = samp.calc_model_mag_err_fixed_age_feh_alpha(
        clean.data["SDSS-Teff"], 0.0, "Ks", teff_err=150, age=1e9, model="MIST")
    clean.data["K Excess"] = clean.data["M_K"] - clean.data["MIST K"]
    clean.data["K Excess Error Down"] = np.sqrt(
        clean.data["M_K_err2"]**2 + clean.data["MIST K Error"]**2)
    clean.data["K Excess Error Up"] = np.sqrt(
        clean.data["M_K_err1"]**2 + clean.data["MIST K Error"]**2)
    clean.split_mag("K Excess", -1.2, splitnames=("Not Dwarfs", "Dwarfs"),
                    null_value=None)

    return clean

@au.memoized
def mcquillan_corrected_splitter():
    nondet = mcquillan_nondetections_splitter_with_DSEP()
    mcq = mcquillan_splitter_with_DSEP()

    mcq_dwarf = mcq.subsample(["Dwarfs", "Right MetCor Teff"])
    nondet_dwarf = nondet.subsample(["Dwarfs", "Right MetCor Teff"])
    # I want to use both the nondetections and detections.
    correction_cols = ["SDSS-Teff", "K Excess"]
    combotable = vstack(
        [mcq_dwarf[correction_cols], nondet_dwarf[correction_cols]])
    solar_coeff = samp.flatten_MS_metallicity(
        combotable["K Excess"], combotable["SDSS-Teff"], deg=1)
    solar_correct = np.poly1d(solar_coeff)
    mcq.data["Corrected K Excess"] = (
        mcq.data["K Excess"] - solar_correct(mcq.data["SDSS-Teff"]))
    return mcq
    
def mcquillan_nondetections_splitter_with_DSEP():
    '''Calculate DSEP mags for the McQuillan Nondetections'''
    nondet = catin.mcquillan_nondetections_with_stelparms()
    nondet_splitter = split.KeplerSplitter(data=nondet)

    nondet_splitter.split_teff(
        "SDSS-Teff", [4000, 5000], (
            "Too Cool MetCor", "Right MetCor Teff", "Too Hot MetCor", 
            "No MetCor Teff"), teff_crit="MetCor Teff",
        null_value=np.ma.masked)
    nondet_splitter.split_teff(
        "SDSS-Teff", [4000, 5000], (
            "Too Cool Statistics", "Right Statistics Teff", 
            "Too Hot Statistics", "No Statistics Teff"), 
        teff_crit="Statistics Teff", null_value=np.ma.masked)
    nondet_splitter.split_teff(
        "SDSS-Teff", 7000, splitnames=(
            "Good Isochrone Teff", "Too Hot for Isochrone", 
            "Bad Isochrone Teff"), teff_crit="Isochrone Temperature",
        null_value=np.ma.masked)
    nondet_splitter.split_photometric_quality(
        "kmag", "kmag_err", splitnames=("K Detection", "Blend", "Bad K"),
        crit="MK blend")
    nondet_splitter.split_Gaia()
    clean_nondets = nondet_splitter.split_subsample([
        "K Detection", "In Gaia", "Good Isochrone Teff"])
    clean_nondets.data["MIST K"] = samp.calc_model_mag_fixed_age_alpha(
        clean_nondets.data["SDSS-Teff"], 0.08, "Ks", age=1e9, model="MIST")
    clean_nondets.data["MIST K Error"] = samp.calc_model_mag_err_fixed_age_feh_alpha(
        clean_nondets.data["SDSS-Teff"], 0.0, "Ks", teff_err=150, age=1e9, 
        model="MIST")
    clean_nondets.data["K Excess"] = (
        clean_nondets.data["M_K"] - clean_nondets.data["MIST K"])
    clean_nondets.data["K Excess Error Down"] = np.sqrt(
        clean_nondets.data["M_K_err2"]**2 + 
        clean_nondets.data["MIST K Error"]**2)
    clean_nondets.data["K Excess Error Up"] = np.sqrt(
        clean_nondets.data["M_K_err1"]**2 + 
        clean_nondets.data["MIST K Error"]**2)
    clean_nondets.split_mag("K Excess", -1.2, splitnames=(
        "Not Dwarfs", "Dwarfs"), null_value=None)
    return clean_nondets

@au.memoized
def mcquillan_nondetections_corrected_splitter():
    '''Corrected magnitude excesses for the McQuillan Nondetections.'''
    nondet = mcquillan_nondetections_splitter_with_DSEP()
    mcq = mcquillan_splitter_with_DSEP()

    mcq_dwarf = mcq.subsample(["Dwarfs", "Right MetCor Teff"])
    nondet_dwarf = nondet.subsample(["Dwarfs", "Right MetCor Teff"])
    # I want to use both the nondetections and detections.
    correction_cols = ["SDSS-Teff", "K Excess"]
    combotable = vstack(
        [mcq_dwarf[correction_cols], nondet_dwarf[correction_cols]])
    solar_coeff = samp.flatten_MS_metallicity(
        combotable["K Excess"], combotable["SDSS-Teff"], deg=1)
    solar_correct = np.poly1d(solar_coeff)
    nondet.data["Corrected K Excess"] = (
        nondet.data["K Excess"] - solar_correct(nondet.data["SDSS-Teff"]))
    return nondet
    
@au.memoized
def eb_splitter_with_DSEP():
    ebs = catin.ebs_with_stelparms()
    eb_splitter = split.McQuillanSplitter(data=ebs)

    eb_splitter.split_teff(
        "SDSS-Teff", [4000, 5000], (
            "Too Cool MetCor", "Right MetCor Teff", "Too Hot MetCor", 
            "No Huber Teff"), teff_crit="MetCor Teff", null_value=np.ma.masked)
    eb_splitter.split_teff(
        "SDSS-Teff", [4000, 5250], (
            "Too Cool Statistics", "Right Statistics Teff", 
            "Too Hot Statistics", "No Statistics Teff"), 
        teff_crit="Statistics Teff", null_value=np.ma.masked)
    eb_splitter.split_teff(
        "SDSS-Teff", 7000, splitnames=(
            "Good Isochrone Teff", "Too Hot for Isochrone", 
            "Bad Isochrone Teff"), teff_crit="Isochrone Temperature", 
        null_value=np.ma.masked)
    eb_splitter.split_photometric_quality(
        "kmag", "kmag_err", splitnames=("K Detection", "Blend", "Bad K"),
        crit="MK blend")
    eb_splitter.split_Gaia()
    clean_ebs = eb_splitter.split_subsample([
        "K Detection", "In Gaia", "Good Isochrone Teff"])
    clean_ebs.data["MIST K"] = samp.calc_model_mag_fixed_age_alpha(
        clean_ebs.data["SDSS-Teff"], 0.08, "Ks", age=1e9, model="MIST")
    clean_ebs.data["K Excess"] = (clean_ebs.data["M_K"] - 
                                  clean_ebs.data["MIST K"])
    clean_ebs.data["MIST K Error"] = samp.calc_model_mag_err_fixed_age_feh_alpha(
        clean_ebs.data["SDSS-Teff"], 0.0, "Ks", teff_err=150, age=1e9, 
        model="MIST")
    clean_ebs.data["K Excess Error Down"] = np.sqrt(
        clean_ebs.data["M_K_err2"]**2 + clean_ebs.data["MIST K Error"]**2)
    clean_ebs.data["K Excess Error Up"] = np.sqrt(
        clean_ebs.data["M_K_err1"]**2 + clean_ebs.data["MIST K Error"]**2)
    clean_ebs.split_mag("K Excess", -1.2, splitnames=("Not Dwarfs", "Dwarfs"),
                        null_value=None)
    # Now apply the same correction as the McQuillan sample.
    nondet = mcquillan_nondetections_splitter_with_DSEP()
    mcq = mcquillan_splitter_with_DSEP()

    mcq_dwarf = mcq.subsample(["Dwarfs", "Right MetCor Teff"])
    nondet_dwarf = nondet.subsample(["Dwarfs", "Right MetCor Teff"])
    correction_cols = ["SDSS-Teff", "K Excess"]
    combotable = vstack(
        [mcq_dwarf[correction_cols], nondet_dwarf[correction_cols]])
    solar_coeff = samp.flatten_MS_metallicity(
        combotable["K Excess"], combotable["SDSS-Teff"], deg=1)
    solar_correct = np.poly1d(solar_coeff)
    clean_ebs.data["Corrected K Excess"] = (
        clean_ebs.data["K Excess"] - solar_correct(clean_ebs.data["SDSS-Teff"]))

    return clean_ebs

