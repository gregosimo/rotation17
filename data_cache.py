import string

import numpy as np
import numpy.core.defchararray as npstr
import astropy_util as au
from astropy.table import vstack, Table

import data_splitting as split
import sample_characterization as samp
import read_catalog as catin
import mist
import sed
import catalog
import baraffe

@au.memoized
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
    notbad = full.split_subsample([
        "~Bad"])
    # I want to populate the objects which should have vsini limits.
    giants = notbad.data["LOGG_FIT"] < np.minimum(
        2 + 2 / 1300 * (notbad.data["TEFF"] - 3500), 4.0)
    vsini_targets = np.logical_not(notbad.data["VSINI"].mask)
    vsini_upperlims = np.logical_and(
        vsini_targets, notbad.data["VSINI"] > 10**(1.982 - 0.301/8))
    vsini_lowerlims = np.logical_and(
        vsini_targets, notbad.data["VSINI"] < 10**(0.176 + 0.301/8))
    vsini_limits = np.logical_or(vsini_upperlims, vsini_lowerlims)
    assert(all(np.ma.getmask(notbad.data["TEFF"][vsini_limits])))
    giant_vsini_limits = np.logical_and(giants, vsini_limits)
    dwarf_vsini_limits = np.logical_and(~giant, vsini_limits)
    notbad.data["TEFF"][giant_vsini_limits] = (
        notbad.data["FPARAM"][giant_vsini_limits,0] - (
            -51.5903 + 61.4774 * notbad.data["FPARAM"][giant_vsini_limits,3] +
            7.17561 * notbad["FPARAM"][giant_vsini_limits,3]**2))
    notbad.data["TEFF"][dwarf_vsini_limits] = (
        notbad.data["FPARAM"][dwarf_vsini_limits,0] - (
            -36.3822 + 13.1614 * notbad.data["FPARAM"][dwarf_vsini_limits,3] +
            -26.0953 * notbad["FPARAM"][dwarf_vsini_limits,3]**2))
    # This calibration is too complicated. See
    # pe.apogee_metallicity_calibration_classification.
    notbad_data["M_H"][vsini_limits] = notbad_data["FPARAM"][vsini_limits,3]

    notbad.split_teff(
        "TEFF", [5250], (
            "APOGEE Evolution Cool", "APOGEE Evolution Hot", "No APOGEE Evolution Teff"), 
        null_value=np.ma.masked, teff_crit="APOGEE Evolution Region")
    notbad.split_teff(
        "teff", [5250], (
            "KSPC Evolution Cool", "KSPC Evolution Hot", "No KSPC Evolution Teff"), 
        null_value=np.ma.masked, teff_crit="KSPC Evolution Region")
    notbad.split_teff(
        "SDSS-Teff", [5250], (
            "Pinsonneault Evolution Cool", "Pinsonneault Evolution Hot", 
            "No Pinsonneault Evolution Teff"), 
        null_value=np.ma.masked, teff_crit="Pinsonneault Evolution Region")
    notbad.split_teff(
        "T_eff [K]", [5250], (
            "El-Badry Evolution Cool", "El-Badry Evolution Hot", "No El-Badry Evolution Teff"), 
        null_value=np.ma.masked, teff_crit="El-Badry Evolution Region")
    notbad.split_metallicity(
        -0.5, ("Low Met", "High Met", "No Met"), col="FE_H",
        null_value=np.ma.masked)
    notbad.split_alpha(
        0.2, ("Low Alpha", "High Alpha", "No Alpha"), col="ALPHA_FE",
        null_value=np.ma.masked)

    clean = notbad.split_subsample([
        "~Bad", "~No APOGEE Evolution Teff", "K Detection", "In Gaia", 
        "~No Met", "~No Alpha"])
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
    clean.data["K Excess"] = clean.data["M_K"] - clean.data["MIST K (sol)"] 
    clean.data["K Excess Error Down"] = np.sqrt(
        clean.data["M_K_err2"]**2 + clean.data["MIST K Error"]**2)
    clean.data["K Excess Error Up"] = np.sqrt(
        clean.data["M_K_err1"]**2 + clean.data["MIST K Error"]**2)
    clean.split_mag(
        "K Excess", -2.4, splitnames=("Giants", "Dwarfs"), null_value=None,
        mag_crit="Giants")
    clean.split_mag(
        "K Excess", [-1.3, -0.3], splitnames=(
            "Photometric Giants", "Photometric Binaries", "Photometric Singles"), null_value=None, 
        mag_crit="Photbins")
    # Split sample into bins in the HR diagram.
    clean.split_APOGEE_evstates(teff_col="TEFF", kcol="K Excess")
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
    clean.data["MIST BC (sol)"] = samp.calc_model_over_feh_fixed_age_alpha(
        np.log10(clean.data["TEFF"]), mist.MISTIsochrone.logteff_col,
        "BC K", 0.0, 1e9)
    clean.data["MIST BC err"] = samp.calc_model_err_fixed_age_feh_alpha(
        np.log10(clean.data["TEFF"]), mist.MISTIsochrone.logteff_col, "BC K",
        apogee_logteff_err, 0.0, age=1e9)
    # Add the zero-point offset.
    clean.data["Gaia L"] = 10**(
        -0.4 * (clean.data["M_K"] + clean.data["MIST BC (sol)"] - 4.75))
    clean.data["Gaia L (ms)"] = 10**(
        -0.4 * (clean.data["MIST K (sol)"] + clean.data["MIST BC (sol)"] - 4.75))
    clean.data["Gaia L err"] = (
        0.4 * np.log(10) * clean.data["Gaia L"] *np.sqrt(
            clean.data["K_ERR"]**2 + (
                5 * clean.data["parallax_error"] / clean.data["parallax"] /
                np.log(10))**2 + clean.data["MIST BC err"]**2))
    clean.data["Gaia R"] = 10**(
        0.5*(np.log10(clean.data["Gaia L"]) - 4*(
            np.log10(clean.data["TEFF"]) - np.log10(5777))))
    clean.data["Gaia MS R"] = 10**(
        0.5*(np.log10(clean.data["Gaia L (ms)"]) - 4*(
            np.log10(clean.data["TEFF"]) - np.log10(5777))))
    clean.data["Gaia R err"] = (
        clean.data["Gaia R"] * np.log(10) * np.sqrt(
            (0.2*clean.data["K_ERR"])**2 + 
            (clean.data["parallax_error"] / clean.data["parallax"] /
             np.log(10))**2 +
            (2 * apogee_logteff_err)**2 + 
            (0.2 * samp.calc_model_err_fixed_age_feh_alpha(
                np.log10(clean.data["TEFF"]), mist.MISTIsochrone.logteff_col,
                mist.MISTIsochrone.radius_col, apogee_logteff_err, 0.0)**2)))
    clean.data["Gaia MS R err"] = clean.data["Gaia R err"]
                

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
    pleiades_good["K-band R (MIST)"] = 10**(0.5*(logLbol_MK - 4 * (
        np.log10(pleiades_good["TEFF"]) - np.log10(5777))))
    # Add in Baraffe-derived radius
    iso = baraffe.BaraffeIsochrone.isochrone_from_file()
    Kband_radius = np.zeros(len(pleiades_good))
    masses = pleiades_good["Mass"]
    stauffer_mass_exists = ~np.ma.getmaskarray(masses)
    baraffe_rad = iso.interpolate_isochrone_cols(
        0.12, masses[stauffer_mass_exists], iso.mass_col, iso.radius_col)
    log_baraffe_lum = iso.interpolate_isochrone_cols(
        0.12, masses[stauffer_mass_exists], iso.mass_col, iso.logL_col)
    Kband_radius[stauffer_mass_exists] = 10**(
        0.5*(log_baraffe_lum - 4*(
            np.log10(pleiades_good["TEFF"][stauffer_mass_exists]) - 
            np.log10(5777))))
    pleiades_good["K-band R (Baraffe)"] = np.ma.masked_equal(Kband_radius, 0)
    apogee_lograd_err = (
        2*pleiades_good["TEFF_ERR"] / pleiades_good["TEFF"] / np.log(10))
    pleiades_good["K-band R Err (Baraffe)"] = (
        apogee_lograd_err * pleiades_good["K-band R (Baraffe)"] * np.log(10))
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

@au.memoized
def stauffer_hartmann_pleiades():
    '''Use Pleiades targets observed just by Stauffer and Hartmann.'''
    # Read in the Rebull catalog.
    pleiades = catin.read_Rebull_Pleiades_Periods()
    # Add in masses from Stauffer.
    stauffer_masses = catin.read_Stauffer_Pleiades_Properties()
    pleiades_combined = au.join_by_id(
        pleiades, stauffer_masses, "EPIC", "EPIC")

    # Now I just want to use the vsinis from Stauffer and Hartmann.
    sh = catin.read_Stauffer_Pleiades()
    simbad_translation = catin.read_SIMBAD_file(
        "simbad_Stauffer_2MASS_ids.txt")
    tmass_names = np.array([j[:23] for j in simbad_translation["identifier"]])
    stauffer_translation_table = Table(
        [simbad_translation["typed ident"], tmass_names], 
        names=("ident", "2mass"))
    stauffer_translate = au.join_by_id(
        sh, stauffer_translation_table, "Star", "ident", join_type="left")
    rebull_translate = catin.read_Rebull_cross_ids()
    stauffer_hartmann_pleiades_cross = catalog.join_by_2MASS_key(
        stauffer_translate, rebull_translate, "2mass", "2MASS")
    stauffer_hartmann_pleiades = au.join_by_id(
        pleiades_combined, stauffer_hartmann_pleiades_cross, "EPIC", "EPIC",
        conflict_suffixes=("_Rebull", "_SH"))
    
    masses = stauffer_hartmann_pleiades["Mass"]
    iso = baraffe.BaraffeIsochrone.isochrone_from_file()
    stauffer_hartmann_pleiades["Baraffe Radius"] = iso.interpolate_isochrone_cols(
        0.12, masses, iso.mass_col, iso.radius_col)
    stauffer_hartmann_pleiades["Baraffe Radius Err"] = (
        iso.isochrone_derivative(0.12, masses, iso.mass_col, iso.radius_col))
    return stauffer_hartmann_pleiades

def pleiades_APOGEE_Literature_vsini():
    '''Return a table containing literature vsini values for APOGEE targets.

    This table contains vsini values taken from the literature for targets
    observed with APOGEE. It is complete in terms of number of observations,
    but every catalog has not been scoured for every observation.'''
    apo_pleiades = pleiades()

    def strip_index(col, prefix, dtype):
        try:
            strippedcol = npstr.strip(col, prefix)
        except TypeError:
            strippedcol = col
        return strippedcol.astype(dtype)
    terndrup = catin.read_Terndrup_Pleiades_KPNO()[
        ["HCG", "vsini", "vsini lim", "Other"]]
    terndrup.rename_column("vsini", "vsini_Terndrup")
    terndrup.rename_column("vsini lim", "vsini_lim_Terndrup")
    terndrup["HII"] = np.ma.masked_values(np.where(
        terndrup["HCG"].mask, npstr.replace(terndrup["Other"], "HII", "", 1), 
        '0').astype(np.int_), 0)
    terndrup_hcg = terndrup[~terndrup["HCG"].mask]
    del(terndrup_hcg["HII"])
    terndrup_hii = terndrup[terndrup["HCG"].mask]
    del(terndrup_hii["HCG"])
    pleiades_terndrup_hcg = au.join_by_id(
        terndrup_hcg, apo_pleiades, "HCG", "HCG", 
        idproc=lambda x: strip_index(x, string.ascii_letters, dtype=np.int_), 
        join_type="right")
    redo_col = np.ma.masked_where(
        pleiades_terndrup_hcg["HCG"] == 0, 
        npstr.add("HCG", pleiades_terndrup_hcg["HCG"].astype("<U7")))
    del(pleiades_terndrup_hcg["HCG"])
    pleiades_terndrup_hcg["HCG"] = redo_col
    pleiades_terndrup_hii = au.join_by_id(
        terndrup_hii, pleiades_terndrup_hcg, "HII", "HII", 
        idproc=lambda x: strip_index(x, string.ascii_letters, dtype=np.int_), 
        conflict_suffixes=["HII", "HCG"], join_type="right")
    redo_col = np.ma.masked_where(
        pleiades_terndrup_hii["HII"] == 0, 
        npstr.add("HII", pleiades_terndrup_hii["HII"].astype("<U9")))
    del(pleiades_terndrup_hii["HII"])
    pleiades_terndrup_hii["HII"] = redo_col
    pleiades_terndrup_hii["vsini_Terndrup"] = np.ma.where(
        pleiades_terndrup_hii["vsini_TerndrupHCG"].mask,
        pleiades_terndrup_hii["vsini_TerndrupHII"],
        pleiades_terndrup_hii["vsini_TerndrupHCG"])
    del(pleiades_terndrup_hii["vsini_TerndrupHCG"])
    del(pleiades_terndrup_hii["vsini_TerndrupHII"])
    pleiades_terndrup_hii["vsini_lim_Terndrup"] = np.ma.where(
        pleiades_terndrup_hii["vsini_lim_TerndrupHCG"].mask,
        pleiades_terndrup_hii["vsini_lim_TerndrupHII"],
        pleiades_terndrup_hii["vsini_lim_TerndrupHCG"])
    del(pleiades_terndrup_hii["vsini_lim_TerndrupHCG"])
    del(pleiades_terndrup_hii["vsini_lim_TerndrupHII"])
    # Add in Keck observations from Table 2.
    terndrup_keck = catin.read_Terndrup_Pleiades_Keck()[
        ["HHJ", "vsini", "vsini lim", "Other"]]
    terndrup_keck.rename_column("vsini", "vsini_Terndrup")
    terndrup_keck.rename_column("vsini lim", "vsini_lim_Terndrup")
    pleiades_terndrup_keck = au.join_by_id(
        terndrup_keck, pleiades_terndrup_hii, "HHJ", "HHJ", 
        idproc=lambda x: strip_index(x, string.ascii_letters, dtype=np.int_), 
        join_type="right", conflict_suffixes=["_KPNO", "_KECK"])
    redo_col = np.ma.masked_where(
        pleiades_terndrup_keck["HHJ"] == 0, 
        npstr.add("HHJ", pleiades_terndrup_keck["HHJ"].astype("<U6")))
    del(pleiades_terndrup_keck["HHJ"])
    pleiades_terndrup_keck["HHJ"] = redo_col
    pleiades_terndrup_keck["vsini_Terndrup"] = np.ma.where(
        pleiades_terndrup_keck["vsini_Terndrup_KPNO"].mask,
        pleiades_terndrup_keck["vsini_Terndrup_KECK"],
        pleiades_terndrup_keck["vsini_Terndrup_KPNO"])
    del(pleiades_terndrup_keck["vsini_Terndrup_KPNO"])
    del(pleiades_terndrup_keck["vsini_Terndrup_KECK"])
    pleiades_terndrup_keck["vsini_lim_Terndrup"] = np.ma.where(
        pleiades_terndrup_keck["vsini_lim_Terndrup_KPNO"].mask,
        pleiades_terndrup_keck["vsini_lim_Terndrup_KECK"],
        pleiades_terndrup_keck["vsini_lim_Terndrup_KPNO"])
    del(pleiades_terndrup_keck["vsini_lim_Terndrup_KPNO"])
    del(pleiades_terndrup_keck["vsini_lim_Terndrup_KECK"])

    pleiades_terndrup_keck["vsini_err_Terndrup"] = (
        0.1 * pleiades_terndrup_keck["vsini_Terndrup"])

    pleiades_terndrup = pleiades_terndrup_keck

    queloz_pleiades = catin.read_Queloz_Pleiades()[
        ["Star", "m_Star", "l_vsiniC", "vsiniC", "e_vsiniC", "vsiniE",
         "e_vsiniE"]]
    queloz_pleiades.rename_column("l_vsiniC", "vsini_lim_QuelozC")
    queloz_pleiades.rename_column("vsiniC", "vsini_QuelozC")
    queloz_pleiades.rename_column("vsiniE", "vsini_QuelozE")
    queloz_pleiades.rename_column("e_vsiniC", "vsini_err_QuelozC")
    queloz_pleiades.rename_column("e_vsiniE", "vsini_err_QuelozE")
    pleiades_queloz = au.join_by_id(
        queloz_pleiades, pleiades_terndrup, "Star", "HII", idproc=lambda x:
        strip_index(x, string.ascii_letters, dtype=np.int_), join_type="right")
    del(pleiades_queloz["Star"])
    redo_col = np.ma.masked_where(
        pleiades_queloz["HII"] == 0, 
        npstr.add("HII", pleiades_queloz["HII"].astype("<U9")))
    del(pleiades_queloz["HII"])
    pleiades_queloz["HII"] = redo_col

    sh_pleiades = catin.read_Stauffer_Pleiades()[[
        "Star", "vsini", "vsini lim", "R"]]
    sh_pleiades = sh_pleiades[npstr.startswith(sh_pleiades["Star"], "HII")]
    sh_pleiades.rename_column("vsini", "vsini_SH")
    sh_pleiades.rename_column("vsini lim", "vsini_lim_SH")
    sh_pleiades["vsini_err_SH"] = sh_pleiades["vsini_SH"] / 2 / (
        1 + sh_pleiades["R"])
    pleiades_sh = au.join_by_id(
        sh_pleiades, pleiades_queloz, "Star", "HII", idproc=lambda x:
        strip_index(x, string.ascii_letters, dtype=np.int_), join_type="right")
    del(pleiades_sh["Star"])
    redo_col = np.ma.masked_where(
        pleiades_sh["HII"] == 0, 
        npstr.add("HII", pleiades_sh["HII"].astype("<U9")))
    del(pleiades_sh["HII"])
    pleiades_sh["HII"] = redo_col

    stauffer = catin.read_Stauffer_84()[["Star", "vsini", "vsini lim"]]
    stauffer = stauffer[npstr.isdigit(stauffer["Star"])]
    stauffer.rename_column("Star", "HII")
    stauffer.rename_column("vsini", "vsini_S84")
    stauffer.rename_column("vsini lim", "vsini_lim_S84")
    stauffer["vsini_err_S84"] = np.where(
        stauffer["vsini_S84"] < 50, 0.1, 0.2) * stauffer["vsini_S84"]
    pleiades_s84 = au.join_by_id(
        stauffer, pleiades_sh, "HII", "HII", idproc=lambda x:
        strip_index(x, string.ascii_letters, dtype=np.int_), join_type="right")
    redo_col = np.ma.masked_where(
        pleiades_s84["HII"] == 0, 
        npstr.add("HII", pleiades_s84["HII"].astype("<U9")))
    del(pleiades_s84["HII"])
    pleiades_s84["HII"] = redo_col

    soderblom_pleiades = catin.read_Soderblom_1993b_vsini()[
        ["HII", "vsini", "vsini lim"]]
    soderblom_pleiades.rename_column("vsini", "vsini_Soderblom1")
    soderblom_pleiades.rename_column("vsini lim", "vsini_lim_Soderblom1")
    pleiades_soderblom = au.join_by_id(
        soderblom_pleiades, pleiades_s84, "HII", "HII", idproc=lambda x:
        strip_index(x, string.ascii_letters, dtype=np.int_), join_type="right")
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
        join_type="right")
    redo_col = np.ma.masked_where(
        more_soderblom_hii_pleiades["HII"] == 0, 
        npstr.add("HII", more_soderblom_hii_pleiades["HII"].astype("<U9")))
    del(more_soderblom_hii_pleiades["HII"])
    more_soderblom_hii_pleiades["HII"] = redo_col
    # Now add in the Pels objects.
    more_soderblom_pels_pleiades = au.join_by_id(
        more_soderblom_pels, more_soderblom_hii_pleiades, "PELS", "PELS", 
        idproc=lambda x: strip_index(x, string.ascii_letters, dtype=np.int_), 
        join_type="right", conflict_suffixes=["_PELS", "_HII"])
    redo_col = np.ma.masked_where(
        more_soderblom_pels_pleiades["PELS"] == 0, 
        npstr.add("PELS", more_soderblom_pels_pleiades["PELS"].astype("<U7")))
    del(more_soderblom_pels_pleiades["PELS"])
    more_soderblom_pels_pleiades["PELS"] = redo_col
    more_soderblom_pels_pleiades["vsini_Soderblom6"] = np.ma.where(
        more_soderblom_pels_pleiades["vsini_Soderblom6_HII"].mask,
        more_soderblom_pels_pleiades["vsini_Soderblom6_PELS"],
        more_soderblom_pels_pleiades["vsini_Soderblom6_HII"])
    del(more_soderblom_pels_pleiades["vsini_Soderblom6_HII"])
    del(more_soderblom_pels_pleiades["vsini_Soderblom6_PELS"])
    more_soderblom_pels_pleiades["vsini_lim_Soderblom6"] = np.ma.where(
        more_soderblom_pels_pleiades["vsini_lim_Soderblom6_HII"].mask,
        more_soderblom_pels_pleiades["vsini_lim_Soderblom6_PELS"],
        more_soderblom_pels_pleiades["vsini_lim_Soderblom6_HII"])
    del(more_soderblom_pels_pleiades["vsini_lim_Soderblom6_HII"])
    del(more_soderblom_pels_pleiades["vsini_lim_Soderblom6_PELS"])

    more_soderblom_pels_pleiades["vsini_Soderblom"] = np.ma.where(
        more_soderblom_pels_pleiades["vsini_Soderblom1"].mask,
        more_soderblom_pels_pleiades["vsini_Soderblom6"],
        more_soderblom_pels_pleiades["vsini_Soderblom1"])
    del(more_soderblom_pels_pleiades["vsini_Soderblom1"])
    del(more_soderblom_pels_pleiades["vsini_Soderblom6"])
    more_soderblom_pels_pleiades["vsini_lim_Soderblom"] = np.ma.where(
        more_soderblom_pels_pleiades["vsini_lim_Soderblom1"].mask,
        more_soderblom_pels_pleiades["vsini_lim_Soderblom6"],
        more_soderblom_pels_pleiades["vsini_lim_Soderblom1"])
    del(more_soderblom_pels_pleiades["vsini_lim_Soderblom1"])
    del(more_soderblom_pels_pleiades["vsini_lim_Soderblom6"])

    del(more_soderblom_pels_pleiades["Name"])

    jackson = catin.read_Jackson_2017_vsini_Table()[[
        "Name", "VSINI", "EVSINI", "VSINI lim"]]
    jackson.rename_column("VSINI", "vsini_Jackson")
    jackson.rename_column("VSINI lim", "vsini_lim_Jackson")
    jackson.rename_column("EVSINI", "vsini_err_Jackson")
    jackson_pleiades = catalog.join_by_2MASS_key(
        jackson, more_soderblom_pels_pleiades, "Name", "APOGEE_ID", 
        join_type="right", conflict_suffixes=("_Jackson", "_APO"))
    del(jackson_pleiades["Name"])

    return jackson_pleiades

