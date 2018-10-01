import numpy as np
import astropy_util as au
from astropy.table import vstack, Table

import data_splitting as split
import sample_characterization as samp
import read_catalog as catin
import mist
import sed

def astero_splitter():
    '''A splitter with the asteroseismic targets.'''
    apokasc = split.APOKASCSplitter()
    split.initialize_asteroseismic_sample(apokasc)
    apokasc.split_teff(
        "TEFF_COR", 5500, ("APOGEE Cool", "APOGEE Hot", "No APOGEE Teff"),
        teff_crit="APOGEE Teff", null_value=np.ma.masked)

    return apokasc

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
    split.initialize_vsini(full)
    full.split_teff(
        "TEFF", [5250], (
            "APOGEE Evolution Cool", "APOGEE Evolution Hot", "No APOGEE Evolution Teff"), 
        null_value=np.ma.masked, teff_crit="APOGEE Evolution Region")
    full.split_teff(
        "teff", [5250], (
            "KSPC Evolution Cool", "KSPC Evolution Hot", "No KSPC Evolution Teff"), 
        null_value=np.ma.masked, teff_crit="KSPC Evolution Region")
    full.split_teff(
        "SDSS-Teff", [5250], (
            "Pinsonneault Evolution Cool", "Pinsonneault Evolution Hot", 
            "No Pinsonneault Evolution Teff"), 
        null_value=np.ma.masked, teff_crit="Pinsonneault Evolution Region")
    full.split_teff(
        "T_eff [K]", [5250], (
            "El-Badry Evolution Cool", "El-Badry Evolution Hot", "No El-Badry Evolution Teff"), 
        null_value=np.ma.masked, teff_crit="El-Badry Evolution Region")
    full.split_metallicity(
        -0.5, ("Low Met", "High Met", "No Met"), col="FE_H",
        null_value=np.ma.masked)

    clean = full.split_subsample([
        "~Bad", "~No APOGEE Evolution Teff", "K Detection", "In Gaia", "~No Met", 
        "~No Vsini"])
    # Add Temperature uncertainties from Holtzmann et al (2018).
    giants = clean.data["LOGG_FIT"] < 2 + 2 / 1500 * (clean.data["TEFF"] - 4500)
    giant_err = np.exp(
        4.3609 + 0.000604303*(clean.data["TEFF"]-4500) -0.0659445 *
        clean.data["M_H"] -0.00196400 * (clean.data["SNREV"]-100))
    dwarf_err = np.exp(
        4.58343+ 0.000289796*(clean.data["TEFF"]-4500) -0.2434860 *
        clean.data["M_H"] -0.00129746 * (clean.data["SNREV"]-100))
    return clean

@au.memoized
def apogee_splitter_with_DSEP():
    '''A datasplitter with DSEP isochrones included. Help!'''
    clean = clean_apogee_splitter()
    clean.data["MIST K"] = np.diag(samp.calc_model_mag_fixed_age_alpha(
        clean.data["TEFF"], clean.data["FE_H"], "Ks", age=1e9, model="MIST v1.2"))
    toohigh_met = clean.data["FE_H"] > 0.5
    clean.data["MIST K"][toohigh_met] = samp.calc_model_mag_fixed_age_alpha(
        clean.data["TEFF"][toohigh_met], 0.5, "Ks", age=1e9, model="MIST v1.2")
    # I can add another gridpoint to the MIST isochrones in lieu of this.
    toolow_met = clean.data["FE_H"] < -2.5
    clean.data["MIST K"][toolow_met] = samp.calc_model_mag_fixed_age_alpha(
        clean.data["TEFF"][toolow_met], -2.5, "Ks", age=1e9, model="MIST v1.2")
    # Instead of "Corrected" K Excess, I'll use a solar metallicity excess for
    # now.
    clean.data["MIST K (sol)"] = samp.calc_model_mag_fixed_age_feh_alpha(
        clean.data["TEFF"], 0.0, "Ks", age=1e9, model="MIST v1.2")
    # Add the uncertainty in the MIST K. I don't want to deal with arbitrary
    # metallicities right now, so solar metallicity is good enough.
    # Since I don't care about the giants, I'm just going to assume everything
    # is a dwarf.
    clean.data["MIST K Error"] = samp.calc_model_mag_err_fixed_age_feh_alpha(
        clean.data["TEFF"], 0.0, "Ks", teff_err=clean.data["TEFF_ERR"], age=1e9, 
        model="MIST v1.2")
    clean.data["K Excess"] = clean.data["M_K"] - clean.data["MIST K"] 
    clean.data["K Excess Error Down"] = np.sqrt(
        clean.data["M_K_err2"]**2 + clean.data["MIST K Error"]**2)
    clean.data["K Excess Error Up"] = np.sqrt(
        clean.data["M_K_err1"]**2 + clean.data["MIST K Error"]**2)
    clean.split_mag("K Excess", -2.4, splitnames=("Giants", "Dwarfs"),
                    null_value=None)
    # Now do a split based solely on the El-Badry temperatures.
    # Since only a subset of these objects have El-Badry temperatues, I need to
    # be wary of masked  values.
    elbadry_teff_indices = ~clean.data["T_eff [K]"].mask
    clean.data["ElBadry K"] = np.ma.ones(len(clean.data))*-9999.0
    clean.data["ElBadry K"][elbadry_teff_indices] = np.diag(
        samp.calc_model_mag_fixed_age_alpha(
            clean.data["T_eff [K]"][elbadry_teff_indices], 
            clean.data["[Fe/H] [dex]"][elbadry_teff_indices], "Ks", age=1e9, 
            model="MIST v1.2"))
    clean.data["ElBadry K"] = np.ma.masked_values(
        clean.data["ElBadry K"], -9999.0)
    clean.data["ElBadry K Excess"] = (
        clean.data["M_K"] - clean.data["ElBadry K"])
    # Now derive 1-Gyr radii for the sample.
    clean.data["MIST R (APOGEE)"] = np.diag(samp.calc_model_over_feh_fixed_age_alpha(
        np.log10(clean.data["TEFF"]), mist.MISTIsochrone.logteff_col,
        mist.MISTIsochrone.radius_col, clean.data["FE_H"], 1e9))
    apogee_logteff_err = (
        clean.data["TEFF_ERR"] / clean.data["TEFF"] / np.log(10))
    clean.data["MIST R Err (APOGEE)"] = samp.calc_model_err_fixed_age_feh_alpha(
        np.log10(clean.data["TEFF"]), mist.MISTIsochrone.logteff_col,
        mist.MISTIsochrone.radius_col, apogee_logteff_err, 0.0, age=1e9)
    # Derive 1-Gyr radii using photometric temperatures.
    clean.data["MIST R (KSPC)"] = np.diag(
        samp.calc_model_over_feh_fixed_age_alpha(
            np.log10(clean.data["teff"]), mist.MISTIsochrone.logteff_col,
            mist.MISTIsochrone.radius_col, clean.data["FE_H"], 1e9))
    mean_kspc_teff_err = (
        (clean.data["teff_err1"] + (-clean.data["teff_err2"]))/2)
    kspc_logteff_err = mean_kspc_teff_err / clean.data["teff"] / np.log(10)
    clean.data["MIST R Err (KSPC)"] = samp.calc_model_err_fixed_age_feh_alpha(
        np.log10(clean.data["teff"]), mist.MISTIsochrone.logteff_col,
        mist.MISTIsochrone.radius_col, mean_kspc_teff_err, 0.0, age=1e9)
    # Derive 1 Gyr radii using the Pinsonneault temperatures.
    clean.data["MIST R (Pinsonneault)"] = np.ma.zeros(len(clean.data))
    clean.data["MIST R Err (Pinsonneault)"] = np.ma.zeros(len(clean.data))
    pinsonneault_teff_masked = clean.data["SDSS-Teff"].mask
    clean.data["MIST R (Pinsonneault)"][~pinsonneault_teff_masked] = np.diag(
        samp.calc_model_over_feh_fixed_age_alpha(
            np.log10(clean.data["SDSS-Teff"][~pinsonneault_teff_masked]), 
            mist.MISTIsochrone.logteff_col, mist.MISTIsochrone.radius_col, 
            clean.data["FE_H"][~pinsonneault_teff_masked], 1e9))
    pinsonneault_logteff_err = (
        clean.data["e_SDSS-Teff"][~pinsonneault_teff_masked] /
        clean.data["SDSS-Teff"][~pinsonneault_teff_masked] / np.log(10))
    clean.data["MIST R Err (Pinsonneault)"][~pinsonneault_teff_masked] = samp.calc_model_err_fixed_age_feh_alpha(
        np.log10(clean.data["SDSS-Teff"][~pinsonneault_teff_masked]), mist.MISTIsochrone.logteff_col,
        mist.MISTIsochrone.radius_col, pinsonneault_logteff_err, 0.0, age=1e9)
    clean.data["MIST R (Pinsonneault)"][pinsonneault_teff_masked] = np.ma.masked
    clean.data["MIST R Err (Pinsonneault)"][pinsonneault_teff_masked] = np.ma.masked
    # Derive 1 Gyr radii using the El Badry Temperatures
    clean.data["MIST R (El-Badry)"] = np.ma.zeros(len(clean.data))
    clean.data["MIST R Err (El-Badry)"] = np.ma.zeros(len(clean.data))
    elbadry_teff_masked = clean.data["T_eff [K]"].mask
    clean.data["MIST R (El-Badry)"][~elbadry_teff_masked] = np.diag(
        samp.calc_model_over_feh_fixed_age_alpha(
            np.log10(clean.data["T_eff [K]"][~elbadry_teff_masked]), 
            mist.MISTIsochrone.logteff_col, mist.MISTIsochrone.radius_col, 
            clean.data["[Fe/H] [dex]"][~elbadry_teff_masked], 1e9))
    clean.data["MIST R Err (El-Badry)"][~elbadry_teff_masked]= samp.calc_model_err_fixed_age_feh_alpha(
        np.log10(clean.data["T_eff [K]"][~elbadry_teff_masked]), mist.MISTIsochrone.logteff_col,
        mist.MISTIsochrone.radius_col, apogee_logteff_err[~elbadry_teff_masked], 0.0, 
        age=1e9)
    clean.data["MIST R (El-Badry)"][elbadry_teff_masked] = np.ma.masked
    clean.data["MIST R Err (El-Badry)"][elbadry_teff_masked] = np.ma.masked

    # Derive R using Gaia magnitudes.
    bolometric_correction = samp.calc_model_over_feh_fixed_age_alpha(
        np.log10(clean.data["TEFF"]), mist.MISTIsochrone.logteff_col,
        "BC K", 0.0, 1e9)
    log_bol_lum = (0.4 * (
        clean.data["MIST K (sol)"] + bolometric_correction - 4.74))
    clean.data["Gaia R"] = 10**(
        0.5*(log_bol_lum - 4*(np.log10(clean.data["TEFF"]) - np.log10(5777))))
    clean.data["Gaia R err"] = (
        clean.data["Gaia R"] * np.log(10) * np.sqrt(
            (0.2*clean.data["K_ERR"])**2 + 
            (clean.data["parallax_error"] / clean.data["parallax"] /
             np.log(10))**2 +
            (2 * apogee_logteff_err)**2 + 
            (0.2 * samp.calc_model_err_fixed_age_feh_alpha(
                np.log10(clean.data["TEFF"]), mist.MISTIsochrone.logteff_col,
                mist.MISTIsochrone.radius_col, apogee_logteff_err, 0.0)**2)))
                

    return clean

@au.memoized
def pleiades():
    '''A Table that holds information about pleiades targets.'''
    pleiades = catin.Rebull_Pleiades_Periods()
    pleiades_good = pleiades[~pleiades["TEFF"].mask]

    pleiades_good["MIST R"] = samp.calc_model_over_feh_fixed_age_alpha(
        np.log10(pleiades_good["TEFF"]), mist.MISTIsochrone.logteff_col, 
        mist.MISTIsochrone.radius_col, 0.0, 1.2e8)
    apogee_logteff_err = (
        pleiades_good["TEFF_ERR"] / pleiades_good["TEFF"] / np.log(10))
    pleiades_good["MIST R Err"] = samp.calc_model_err_fixed_age_feh_alpha(
        np.log10(pleiades_good["TEFF"]), mist.MISTIsochrone.logteff_col,
        mist.MISTIsochrone.radius_col, apogee_logteff_err, 0.0, age=1e9)
    pleiades_good["MK"] = pleiades_good["K"] - 5 * np.log10(136/10) - 0.01
    pleiades_good["MIST MK"] = samp.calc_model_mag_fixed_age_alpha(
        pleiades_good["TEFF"], 0.0, "Ks", age=1.2e8, model="MIST v1.2")
    logLbol_MK = (-0.4*(
        pleiades_good["MK"] + samp.calc_model_over_feh_fixed_age_alpha(
            np.log10(pleiades_good["TEFF"]), mist.MISTIsochrone.logteff_col,
            "BC K", 0.0, 1.2e8) - 4.75))
    pleiades_good["K Excess"] = pleiades_good["MK"] - pleiades_good["MIST MK"]
    pleiades_good["K-band R"] = 10**(0.5*(logLbol_MK - 4 * (
        np.log10(pleiades_good["TEFF"]) - np.log10(5777))))
    pleiades_good["MV"] = (
        pleiades_good["(V-K)0"] + pleiades_good["K"] - 0.01 - 5 *
        np.log10(132/10))
    pleiades_good["MIST MV"] = samp.calc_model_mag_fixed_age_alpha(
        pleiades_good["TEFF"], 0.0, "V", age=1.2e8, model="MIST v1.2")
    logLbol_MV = (-0.4*(
        pleiades_good["MV"] + samp.calc_model_over_feh_fixed_age_alpha(
            np.log10(pleiades_good["TEFF"]), mist.MISTIsochrone.logteff_col,
            "BC V", 0.0, 1.2e8) - 4.75))
    pleiades_good["V-band R"] = 10**(0.5*(logLbol_MV - 4 * (
        np.log10(pleiades_good["TEFF"]) - np.log10(5777))))

    return pleiades_good
