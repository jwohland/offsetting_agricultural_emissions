# -*- coding: utf-8 -*-
"""
Package of functions to calculate CDR rates and associated costs to offset agricultural methane 

@author: Nicoletta Brazzola 
"""

import numpy as np
import pandas as pd

from scipy import stats as stats
from scipy.optimize import minimize_scalar

from fair.forward import fair_scm
from fair.RCPs import rcp26

# Get RCP modules & natural forcings
from fair.ancil import natural, cmip6_volcanic, cmip6_solar


def update_rcp26():
    """
    Function to update the built-in RCP2.6 CMIP6 emission scenarios of FaIR with the latest
    release from IIASA's RCP database and with observations prior to 2015
    URL: https://tntcat.iiasa.ac.at/RcpDb/dsd?Action=htmlpage&page=welcome
    # accessed: 24-11-2020
    Returns: class of RCP2.6 emissions

    """
    # Defining indexes
    index_2000, index_start, index_end = 2000 - 1765, 2015 - 1765, 2101 - 1765
    # Defining relevant columns from SSP CMIP6 emissions
    columns = [
        "CMIP6 Emissions|CO2",
        "CMIP6 Emissions|CO2|AFOLU",
        "CMIP6 Emissions|CH4",
        "CMIP6 Emissions|N2O",
        "CMIP6 Emissions|CO",
        "CMIP6 Emissions|VOC",
        "CMIP6 Emissions|NOx",
        "CMIP6 Emissions|BC",
        "CMIP6 Emissions|OC",
        "CMIP6 Emissions|NH3",
        "CMIP6 Emissions|CF4",
        "CMIP6 Emissions|C2F6",
        "CMIP6 Emissions|SF6",
    ]
    # load SSPs
    df_world = load_SSP_data(columns, "SSP1-26")
    # Interpolate data to annual values
    df_world = df_world.resample("AS").asfreq()
    df_world = df_world.astype(float).interpolate(method="polynomial", order = 2)
    # copy FaIR's RCP2.6
    class rcp26_new(rcp26.Emissions):
        pass

    # update emissions between 2015-2100 from SSP database
    CO2_nonAFOLU = (
        df_world.loc[:, "CMIP6 Emissions|CO2"]
        - df_world.loc[:, "CMIP6 Emissions|CO2|AFOLU"]
    )
    rcp26_new.co2_fossil[index_start:index_end] = CO2_nonAFOLU.values / (
        3.677 * 10 ** 3
    )
    rcp26_new.co2_land[index_start:index_end] = df_world.loc[
        :, "CMIP6 Emissions|CO2|AFOLU"
    ].values / (3.677 * 10 ** 3)
    rcp26_new.co2 = np.sum(rcp26_new.emissions[:, 1:3], axis=1)
    rcp26_new.ch4[index_start:index_end] = df_world["CMIP6 Emissions|CH4"].values
    rcp26_new.n2o[index_start:index_end] = df_world["CMIP6 Emissions|N2O"].values / (
        (44.01 / 28.01) * 10 ** 3 # from ktN2O to MtN2
    )
    rcp26_new.co[index_start:index_end] = df_world["CMIP6 Emissions|CO"].values
    rcp26_new.nmvoc[index_start:index_end] = df_world["CMIP6 Emissions|VOC"].values
    rcp26_new.nox[index_start:index_end] = df_world["CMIP6 Emissions|NOx"].values / (
        4.101
    )
    rcp26_new.bc[index_start:index_end] = df_world["CMIP6 Emissions|BC"].values
    rcp26_new.oc[index_start:index_end] = df_world["CMIP6 Emissions|OC"].values
    rcp26_new.nh3[index_start:index_end] = df_world["CMIP6 Emissions|NH3"].values
    rcp26_new.cf4[index_start:index_end] = df_world["CMIP6 Emissions|CF4"].values
    rcp26_new.c2f6[index_start:index_end] = df_world["CMIP6 Emissions|C2F6"].values
    rcp26_new.sf6[index_start:index_end] = df_world["CMIP6 Emissions|SF6"].values
    # update RCP2.6 with observations of relevant species up to 2015
    rcp26_new.co2[index_2000 + 10 : index_2000 + 15] = [
        9.82634675,
        10.0317621,
        10.23717745,
        10.44259281,
        10.64800816,
    ]
    rcp26_new.co2_fossil[index_2000 + 10 : index_2000 + 15] = (
        rcp26_new.co2[index_2000 + 10 : index_2000 + 15]
        - rcp26_new.co2_land[index_2000 + 10 : index_2000 + 15]
    )
    rcp26_new.ch4[index_2000 : index_2000 + 15] = [
        310.188,
        319.24,
        328.292,
        337.344,
        346.396,
        351.2962,
        356.1964,
        361.0966,
        365.9968,
        370.897,
        374.3322,
        377.7674,
        381.2026,
        384.6378,
        388.073,
    ]

    hist_N2O = [11.718, 12.075, 12.317, 11.571, 8.911] # from RCP database IIASA
    dates_hist = pd.to_datetime([2000, 2005, 2010, 2020, 2030], format="%Y")
    df_histN2O = pd.DataFrame(data=np.asarray([hist_N2O]).T, index=dates_hist, columns=["N2O"])
    df_histN2O = df_histN2O.resample("AS").asfreq()
    df_histN2O = df_histN2O.astype(float).interpolate(method="polynomial", order = 2)
    rcp26_new.n2o[index_2000: index_2000 + 31] = df_histN2O["N2O"].values/(44.01/28.01) #get MtN2/yr
    return rcp26_new


def calc_early_CH4_data():
    """
    Function to retrieve historical agricultural methane from IIASA's RCP database
    URL: https://tntcat.iiasa.ac.at/RcpDb/dsd?Action=htmlpage&page=welcome

    Returns: panda dataframe with historical agricultural and agricultural waste methane emissions
    """
    agriCH4_hist = [
        119.358,
        125.866,
        132.143,
        134.961,
        137.725,
    ]  # Historical agricultural + agricultural methane emissions from SSP database
    agriCH4_waste_hist = [
        1.315,
        1.583,
        1.299,
        1.144,
        1.369,
    ]
    dates_hist = pd.to_datetime(
        [2000, 2005, 2010, 2012, 2014,], format="%Y"
    )  # dates of historical observations
    columns = [
        "CMIP6 Emissions|CH4|Agriculture",
        "CMIP6 Emissions|CH4|Agricultural Waste Burning",
    ]
    return pd.DataFrame(
        data=np.asarray([agriCH4_hist, agriCH4_waste_hist]).T,
        index=dates_hist,
        columns=columns,
    )

def calc_early_N2O_data():
    """
    Function to retrieve historical agricultural nitrous oxide from FAOSTAT
    URL: http://www.fao.org/faostat/en/#data/GT

    Returns: panda dataframe with historical agricultural and agricultural waste methane emissions
    """
    agriN2O_hist = [
        6409.704,
        6560.078,
        6637.232,
        6631.073,
        6817.920,
    ] # Historical land use N2O emissions from FAO database

    dates_hist = pd.to_datetime(
        [2000, 2001, 2002, 2003, 2004,], format="%Y"
    )  # dates of historical observations
    columns = [
        "Emissions|N2O|Land Use",
    ]
    return pd.DataFrame(
        data=np.asarray([agriN2O_hist]).T,
        index=dates_hist,
        columns=columns,
    )



def calc_agri_emissions(scenario):
    """
    Function to calculate agricultural methane emissions for a given SSP scenario 
    
    Inputs: 
        scenario : SSP scenario; e.g 'SSP1-26', 'SSP3-70 (Baseline)'
    """
    # Import SSP scenarios and shape the table so to be able to interpolate between the data points
    columns = [
        "CMIP6 Emissions|CH4|Agriculture",
        "CMIP6 Emissions|CH4|Agricultural Waste Burning",
    ]
    df_world = load_SSP_data(columns, scenario)

    # Add data prior to 2015 from URL
    early_CH4_data = calc_early_CH4_data()
    df_world = early_CH4_data.append(df_world)

    # Interpolate data to annual values
    df_world = df_world.resample("AS").asfreq()
    df_world = df_world.astype(float).interpolate(method="polynomial", order = 2)
    return df_world.values.sum(axis=1)  # output sum of agriculture and wasteburning


def calc_agriN2O(scenario, model = 'IMAGE'):
    """
    Function to calculate agricultural methane emissions for a given SSP scenario

    Inputs:
        scenario : SSP scenario; e.g 'SSP1-26', 'SSP3-70 (Baseline)'
        model : model underlying the SSP scenario chose (in this analysis, IMAGE)
    """
    # Import SSP scenarios and shape the table so to be able to interpolate between the data points
    columns = [
        "Emissions|N2O|Land Use",
    ]
    df_world = load_IAM_data(columns, scenario, model)

    # Add data prior to 2005 from URL
    early_N2O_data = calc_early_N2O_data()
    df_world = early_N2O_data.append(df_world)/ (
        (44.01 / 28.01) * 10 ** 3
    )

    # Interpolate data to annual values
    df_world = df_world.resample("AS").asfreq()
    df_world = df_world.astype(float).interpolate(method="polynomial", order = 2)
    return df_world.values.sum(axis=1)  # output sum of agriculture and wasteburning


# Create arrays of agricultural methane emissions in the different scenarios.
def make_agri_scenarios(ghg=None,
    scenario1="SSP1-26", scenario2="const", scenario3="SSP3-70 (Baseline)", year=2015
):
    """
    Function to make the three agricultural emission scenarios used in this analysis

    Args:
        scenario1: scenario name from IIASA database, default: "SSP1-26"
        scenario2: scenario name from IIASA database, default: "const"
        scenario3: scenario name from IIASA database, default: "SSP3-70 (Baseline)"
        year: in constant scenario, methane is used in this year, default: 2015

    Returns:
        3 arrays of annual agricultural methane emissions correspoding to the chosen scenarios 1,2,3
    """
    E_CH4_agri_1 = calc_agri_emissions(scenario1)
    i_year = year - 2000  # Scenarios start in 2000

    if scenario2 == "const":
        E_CH4_agri_2 = E_CH4_agri_1.copy()
        E_CH4_agri_2[i_year:] = E_CH4_agri_1[i_year]
    else:
        E_CH4_agri_2 = calc_agri_emissions(scenario2)

    E_CH4_agri_3 = calc_agri_emissions(scenario3)

    if ghg is None:
        return E_CH4_agri_1, E_CH4_agri_2, E_CH4_agri_3
    else:
        E_N2O_agri_1 = calc_agriN2O(scenario1)
        E_N2O_agri_3 = calc_agriN2O(scenario3)
        if scenario2 == "const":
            E_N2O_agri_2 = E_N2O_agri_1.copy()
            for i in np.arange(0,86):
                if E_N2O_agri_1[i_year] > E_N2O_agri_1[i_year+i]:
                    E_N2O_agri_2[i_year+i] = E_N2O_agri_1[i_year]
                else:
                    E_N2O_agri_2[i_year+i] = E_N2O_agri_1[i_year+i]
        else:
            E_N2O_agri_2 = calc_agriN2O(scenario2)
        return [E_CH4_agri_1,
                E_CH4_agri_2,
                E_CH4_agri_3,
                E_N2O_agri_1,
                E_N2O_agri_2,
                E_N2O_agri_3,
                ]


# Integrate different scenarios of agricultural methane emissions in the RCP2.6 emissions
def make_scenarios(ghg=None):
    """
    Calculate all emissions under RCP2.6 and different methane emission assumptions

    Args:
        ghg: tells what scenarios of agricultural emissions are you considering (None = just methane,
        "N2O" = just nitrous oxide, "CH4&N2O" = both)

    Returns:
    Emissions following RCP2.6, RCP2.6 + constant agricultural CH4, RCP2.6 + BAU agricultural CH4 for 2000 - 2100
    """

    # make agricultural scenarios
    if ghg is None:
        E_CH4_agri_1, E_CH4_agri_2, E_CH4_agri_3 = make_agri_scenarios()
    else:
        E_CH4_agri_1, E_CH4_agri_2, E_CH4_agri_3, E_N2O_agri_1, E_N2O_agri_2, E_N2O_agri_3 \
            = make_agri_scenarios('CH4&N2O')

    rcp26_newE = update_rcp26()
    rcp26_emissions = rcp26_newE.emissions
    rcp26_emissions_2 = rcp26_emissions.copy()
    rcp26_emissions_3 = rcp26_emissions.copy()
    if ghg != 'N2O':
    # compute non-agricultural CH4 emissions
        rcp26_emissions_CH4nonagri = rcp26_emissions[:, 3].copy()
        rcp26_emissions_CH4nonagri[235:336] -= E_CH4_agri_1
        # compute total CH4 emissions for constant and BAU scenarios
        rcp26_emissions_2[235:336, 3] = E_CH4_agri_2 + rcp26_emissions_CH4nonagri[235:336]
        rcp26_emissions_3[235:336, 3] = E_CH4_agri_3 + rcp26_emissions_CH4nonagri[235:336]
        if ghg is not None:
            # compute non agricultural N2O emissions
            rcp26_emissions_N2Ononagri = rcp26_emissions[:,4].copy()
            rcp26_emissions_N2Ononagri[235:336] -= E_N2O_agri_1
            # compute total N2O emissions for constant and BAU scenarios
            rcp26_emissions_2[235:336, 4] = E_N2O_agri_2 + rcp26_emissions_N2Ononagri[235:336]
            rcp26_emissions_3[235:336, 4] = E_N2O_agri_3 + rcp26_emissions_N2Ononagri[235:336]
    elif ghg == 'N2O':
        # compute non agricultural N2O emissions
        rcp26_emissions_N2Ononagri = rcp26_emissions[:, 4].copy()
        rcp26_emissions_N2Ononagri[235:336] -= E_N2O_agri_1
        # compute total N2O emissions for constant and BAU scenarios
        rcp26_emissions_2[235:336, 4] = E_N2O_agri_2 + rcp26_emissions_N2Ononagri[235:336]
        rcp26_emissions_3[235:336, 4] = E_N2O_agri_3 + rcp26_emissions_N2Ononagri[235:336]

    return rcp26_emissions, rcp26_emissions_2, rcp26_emissions_3


def calc_early_ffCH4_data():
    """
    Calculate historical fossil fuel methane emissions from IIASA's SSP database
    URL: https://tntcat.iiasa.ac.at/RcpDb/dsd?Action=htmlpage&page=welcome

    Returns: array with historical methane emissions from fossil fuel sources

    """
    # From SSP database
    CH4tot_hist = [310.188, 346.396, 370.897, 380.811, 387.874]
    energyCH4_hist = [110.657, 131.446, 147.989, 153.882, 154.206]
    transpCH4_hist = [0.736, 0.683, 0.597, 0.585, 0.568]
    indCH4_hist = [0.626, 0.719, 0.827, 0.836, 0.890]
    residCH4_hist = [10.979, 11.116, 12.052, 12.059, 12.086]
    dates_hist = pd.to_datetime(
        [2000, 2005, 2010, 2012, 2014], format="%Y"
    )  # dates of historical observations
    columns = [
        "CMIP6 Emissions|CH4",
        "CMIP6 Emissions|CH4|Energy Sector",
        "CMIP6 Emissions|CH4|Transportation Sector",
        "CMIP6 Emissions|CH4|Industrial Sector",
        "CMIP6 Emissions|CH4|Residential Commercial Other",
    ]
    return pd.DataFrame(
        data=np.asarray(
            [CH4tot_hist, energyCH4_hist, transpCH4_hist, indCH4_hist, residCH4_hist]
        ).T,
        index=dates_hist,
        columns=columns,
    )


def load_SSP_data(columns, scenario):
    """

    Functions loading IIASA's SSP database and returning data frame with just selected columns
    for a given SSP scenario

    Args:
        columns: list a column names
        scenario: scenario_name, e.g. 'SSP1-26' or 'SSP3-70 (Baseline)'

    Returns: SSP data for selected columns for a given SSP scenario

    """

    data = pd.read_csv(
        "Data/SSPs/SSP_CMIP6_201811.csv",
        delimiter=",",
        usecols=[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    )
    df_world = (
        data.loc[data["REGION"] == "World"].loc[data["SCENARIO"] == scenario].copy()
    )
    df_world = df_world.T
    df_world.columns = df_world.iloc[2, :]  # set column names
    df_world = df_world.iloc[3:, :]  # get rid of headers
    df_world.index = pd.to_datetime(df_world.index)
    return df_world[columns]

def load_IAM_data(columns, scenario, model):
    """

    Functions loading IIASA's SSP database and returning data frame with just selected columns
    for a given SSP scenario

    Args:
        columns: list a column names
        scenario: scenario_name, e.g. 'SSP1-26' or 'SSP3-70 (Baseline)'

    Returns: SSP data for selected columns for a given SSP scenario

    """

    data = pd.read_csv(
        "Data/SSPs/SSP_IAM_V2_201811.csv",
        delimiter=","
    )

    if scenario == 'SSP3-70 (Baseline)':
        scenario =  'SSP3-Baseline'

    df_world = (
        data.loc[data["REGION"] == "World"]
            .loc[data["SCENARIO"] == scenario]
            .loc[data["MODEL"]==model]
            .copy()
    )
    df_world = df_world.T
    df_world.columns = df_world.iloc[3,:]  # set column names
    df_world = df_world.iloc[5:, :]  # get rid of headers
    df_world.index = pd.to_datetime(df_world.index)
    return df_world[columns]


# calculate percentage of CH4 from FF emissions
def calc_ffch4_emissions(scenario):
    """

    Function calculating the amount of methane emissions from fossil fuel sources under
    a giver SSP scenario

    Args:
        scenario: SSP scenario, e.g. 'SSP1-26'

    Returns: methane emissions from fossil fuels under a given SSP scenario

    """
    # Import SSP scenarios and shape the table so to be able to interpolate between the data points
    columns = [
        "CMIP6 Emissions|CH4",
        "CMIP6 Emissions|CH4|Energy Sector",
        "CMIP6 Emissions|CH4|Transportation Sector",
        "CMIP6 Emissions|CH4|Industrial Sector",
        "CMIP6 Emissions|CH4|Residential Commercial Other",
    ]
    df_world = load_SSP_data(columns, scenario)

    # Add data prior to 2015 from URL # todo add url
    early_ffCH4_data = calc_early_ffCH4_data()
    df_world = early_ffCH4_data.append(df_world)

    # Interpolate data to annual values
    df_world = df_world.resample("AS").asfreq()
    df_world = df_world.astype(float).interpolate(method="polynomial", order = 2)

    # Return methane emissions due to fossil fuels
    return df_world.iloc[:, 1:].values.sum(axis=1)


# Calculate percentage of methane from fossil fuels for each emission scenario
def make_percFFch4_scenarios(scenario):
    """

    Calculating the percentage of methane, in a given scenario, corresponding to fossil fuel sources

    Args:
        scenario: scenario name from IIASA database: "SSP1-26", "const.", or "SSP3-70 (Baseline)"

    Returns:
        1 array of fraction of fossil fuel methane emissions corresponding to a chosen scenario
    """
    # Calculate total emission scenarios
    E_tot_1, E_tot_2, E_tot_3 = make_scenarios()

    # Calculate amount of methane emissions from fossil fuels under the RCP2.6
    ffch4_RCP26 = calc_ffch4_emissions("SSP1-26")

    # Calculate fraction of fossil fuel methane emissions in a chosen scenario
    if scenario == "SSP1-26":
        return (
            ffch4_RCP26 / E_tot_1[235:336, 3]
        )  # select 2000-2100 (235:336) and methand (3)
    if scenario == "const":
        return ffch4_RCP26 / E_tot_2[235:336, 3]
    if scenario == "SSP3-70 (Baseline)":
        return ffch4_RCP26 / E_tot_3[235:336, 3]


def run_fair(E, scenario, n_runs, CDR=None,
             index_2000 = 2000 - 1765, index_end = 2101 - 1765):
    """
    Function to run FaIR from a given emission input potentially accounting for CDR.
    Produces both single simulation with best estimate of parameters and ensemble simulation.

    Args:
        E: array with all input emission species
        scenario: scenario of agricultural methane emissions: 'SSP1-26', 'const', or 'SSP3-70 (Baseline)'
        n_runs: number of ensemble members
        CDR: None or Array of CDR from 2000 to 2100
        index_2000: sets the start of the analysis (default, index of year 2000 in RCP2.6)
        index_end: sets the end of the analysis (default, index of year 2100 in RCP2.6)

    Returns: class with single and ensemble simulations of concentrations, ERF, and T anomalies as objects

    """

    class Fair_outputs:
        """
        Class to store Fair output nicely
        """

        def __init__(self, C_single, C_Co2_ens, F_single, F_tot_ens, T_single, T_ens):
            self.C_single = C_single
            self.C_Co2ens = C_Co2_ens
            self.F_single = F_single
            self.F_tot_ens = F_tot_ens
            self.T_single = T_single
            self.T_ens = T_ens


    # adding CDR rates to fossil fuel emissions
    E_wCDR = E.copy()
    if CDR is not None:
        E_wCDR[index_2000:index_end, 1] += CDR

    # Running single simulations with best estimates
    C_single, F_single, T_single_nodT = fair_scm(
        emissions=E_wCDR[:index_end, :],
        fossilCH4_frac=np.concatenate(
            (np.zeros(index_2000), make_percFFch4_scenarios(scenario))
        ),  # Fossil fuel methane emission become available in 2000
        natural=natural.Emissions.emissions[:index_end],
        F_volcanic=cmip6_volcanic.Forcing.volcanic[:index_end],
        F_solar=cmip6_solar.Forcing.solar[:index_end],
    )
    # Making temperature anomaly relative to 1850-1900 time period
    dT_1850_1900 = np.mean(T_single_nodT[1850 - 1765 : 1900 - 1765])
    T_single = T_single_nodT - dT_1850_1900

    # Calculating ensemble simulations
    C_CO2ens, F_tot_ens, T_ens_nodT, = calc_ensemble(
        E_wCDR[:index_end],
        np.concatenate((np.zeros(index_2000), make_percFFch4_scenarios(scenario))),
        n=n_runs,
        index_end=index_end
    )

    # Making temperature anomaly relative to 1850-1900 time period
    T_ens = T_ens_nodT - dT_1850_1900

    return Fair_outputs(C_single, C_CO2ens, F_single, F_tot_ens, T_single, T_ens,)


def calc_CDR_RCP_2100(scenario="SSP1-26", model="IMAGE"):
    """
    Calculate the total CDR in RCP2.6 scenario from the SSP database

    Args:
        scenario: SSP scenario considered; default: "SSP1-26"
        model: IAM model; default: "IMAGE" (model used to produce RCP2.6 in van Vuuren et al., 2011)

    Returns: Array with CDR rates between 2000-2100 in GtC (negative values for negative emissions)

    """
    columns = [
        "Emissions|CO2|Land Use",
        "Emissions|CO2|Carbon Capture and Storage",
    ]
    # Import and manipulate data from SSP's IAM
    data = pd.read_csv("Data/SSPs/SSP_IAM_V2_201811.csv", delimiter=",")
    df_world = (
        data.loc[data["REGION"] == "World"]
        .loc[data["SCENARIO"] == scenario]
        .loc[data["MODEL"] == model]
        .copy()
    )
    df_world = df_world.T
    df_world.columns = df_world.iloc[3, :]
    df_world = df_world.iloc[5:, :]
    df_world.index = pd.to_datetime(df_world.index)
    df_world = df_world[columns]
    # Interpolate between data points
    df_world = df_world.resample("AS").asfreq()
    df_world = df_world.astype(float).interpolate(method="polynomial", order = 2)
    # Remove all positive emissions from land use
    df_world["Emissions|CO2|Land Use"].loc[df_world["Emissions|CO2|Land Use"] > 0] = 0
    # Add CCS to negative land use emissions
    E_CDR = (
        df_world.loc[:, "Emissions|CO2|Carbon Capture and Storage"]
        - df_world["Emissions|CO2|Land Use"]
    )
    # data only starts in 2005, so we add zeros to make it start in 2000
    return np.concatenate(
        ([0.0, 0.0, 0.0, 0.0, 0.0], -E_CDR.values / (3.677 * 10 ** 3))
    )


def statistical_parameters(n, seed):
    """
    Generates ensemble parameters:
    tcres = Transient Climate Response and Equilibrium Climate Sensitivity,
    r0 = pre-industrial sensitivity of carbon sinks
    rc = sensitivity to cumulative CO2 emissions
    rt =  sensitivity to temperature change
    d = ocean temperature response parameters

    Args:
        n: ensemble size
        seed: seed for statistics

    Returns: ensemble parameters

    """
    from fair.tools.ensemble import tcrecs_generate

    # generate TCR and ECS pairs, using a lognormal distribution informed by CMIP5 models
    tcrecs = tcrecs_generate("cmip5", n=n, dist="lognorm", correlated=True, seed=seed)

    # generate ensemble for carbon cycle parameters
    r0 = stats.norm.rvs(size=n, loc=35, scale=3.5, random_state=41000)
    rc = stats.norm.rvs(size=n, loc=0.019, scale=0.0019, random_state=42000)
    rt = stats.norm.rvs(size=n, loc=4.165, scale=0.4165, random_state=45000)

    # generate ensemble for ocean temperature response parameters
    d1 = stats.norm.rvs(size=n, loc=239.0, scale=23.9, random_state=41000)
    d2 = stats.norm.rvs(size=n, loc=4.1, scale=0.41, random_state=41000)
    d = np.array([d1, d2])
    return tcrecs, r0, rc, rt, d


def calc_ensemble(
    E, ffCH4, n=100, C_pi_new=np.array([278.0, 722.0, 273.0, 0] + [0.0] * 25 + [0, 0]),
        index_end=336, index_start = 0
):
    """
    Calculate ensemble simulations of CO2 ERF and total temperature anomalies

    Args:
        E: input emissions (needs to start in start_year and end in end_year)
        ffCH4: array of share of methane emissions from fossil fuel sources
        n: number of ensemble members; default: 100
        C_pi_new: concentrations in the start year of the analysis; default: string of PI concentrations used in FaIR
        index_end =  sets index of end of analysis (default corresponds to year 2100)
        index_start = sets index of start of analysis (default corresponds to zero, year 2000)

    Returns: temperature anomaly ensemble, CO2 concentrations ensemble, CO2 ERF ensemble, and boolean of constrained ensemble members

    """
    from fair.tools.constrain import hist_temp

    # laod Cowtan & Way in-filled dataset of global temperatures (historical)
    CW = np.loadtxt("Data/tools/tempobs/had4_krig_annual_v2_0_0.csv")

    # set parameters
    seed = 38571
    tot_CO2 = E[:, 1] + E[:, 2]  # sum of fossil fuels and land use change

    # define variables
    nt = len(E[index_start:index_end, 0])  # number of timesteps
    T = np.zeros((nt, n))  # Temperature anomalies
    C_CO2 = np.zeros((nt, n))  # Co2 concentrations
    F = np.zeros((nt, n))  # CO2 forcings
    constrained = np.zeros(n, dtype=bool)  # mask for runs that are not plausible

    # generate ensemble parameters
    tcrecs, r0, rc, rt, d = statistical_parameters(n, seed)

    # loop over ensemble members
    for i in range(n):
        tmp_C, tmp_F, T[:, i] = fair_scm(
            emissions=E[index_start:index_end, :],
            r0=r0[i],
            rc=rc[i],
            rt=rt[i],
            tcrecs=tcrecs[i, :],
            d=d[:, i],
            C_pi=C_pi_new,
            fossilCH4_frac=ffCH4[index_start:],
            natural=natural.Emissions.emissions[index_start:index_end, :],
            F_volcanic=cmip6_volcanic.Forcing.volcanic[index_start:index_end],
            F_solar=cmip6_solar.Forcing.solar[index_start:index_end],
        )
        C_CO2[:, i] = tmp_C[:, 0]  # CO2 concentrations
        F[:, i] = tmp_F.sum(axis=1)  # cumulative forcing of all species
        constrained[i], _, _, _, _ = hist_temp(
            CW[30:, 1], T[1880 - 1765 : 2017 - 1765, i], CW[30:, 0], CI=0.9
        )  # confidence interval between 5-95%

    return (
        C_CO2[(index_start) : (index_end - index_start), constrained],
        F[(index_start) : (index_end - index_start), constrained],
        T[(index_start) : (index_end - index_start), constrained],
    )


class Fair_inversion:
    """
    Class to numerically invert FAIR accounting for all species.

    The implemented method in FAIR uses the Myhre equations which is overly simplified.

    Starting point: We know the forcing F and we are interested in the emissions E. FAIR operates the other way:
        FAIR(E) = C, F, T
    We therefore numerically invert FAIR to find an E* Fair(E*)[1]=F*

    Iterative process:
        E denotes all emissions but CO2 emissions from 2015 == 0
        while t_end <= 2100
            t_end = 2015
                find E'[t_end, 1] s.t.
                    sum_over_forcings(Fair(E'[:t_end])[1]) = sum_over_forcing(F_rcp2.6)
                add E' to E
            T_end += 1
    """

    def __init__(self, emissions, F_rcp26, scenario, end_year):
        """
        Args:
            emissions: Array of emissions (only used until end_year)
            F_rcp26: Array of forcings of the rcp2.6 scenario
            scenario: Current scenario
            end_year: current end year. allowed values 2015 - 2100
        """
        self.emissions = emissions
        self.F_rcp26 = F_rcp26  # rcp2.6 forcing
        self.scenario = scenario
        self.end_index = end_year - 1765

    def cost_function(self, CDR_rates):
        """
        Calculates the deviation of total forcing in the end_year between:
            a) emissions and scenario of an instance of this class
            b) RCP2.6
        Minimize this cost function to ensure that CDR_rates lead to same forcing as in RCP2.6
        Args:
            CDR_rates: CO2 removal emissions in end_year (convention: removal negative number)

        Returns:

        """
        index_2000 = 2000 - 1765
        guess_emissions = self.emissions.copy()
        guess_emissions[self.end_index - 1, 1] += CDR_rates
        F = fair_scm(
            emissions=guess_emissions[: self.end_index, :],
            fossilCH4_frac=np.concatenate(
                (np.zeros(index_2000), make_percFFch4_scenarios(self.scenario))
            )[
                : self.end_index
            ],  # Fossil fuel methane emission become available in 2000
            natural=natural.Emissions.emissions[: self.end_index],
            F_volcanic=cmip6_volcanic.Forcing.volcanic[: self.end_index],
            F_solar=cmip6_solar.Forcing.solar[: self.end_index],
        )[1]
        diff = np.sum(F[self.end_index - 1, :]) - np.sum(
            self.F_rcp26[self.end_index - 1, :]
        )
        return np.abs(diff)


def calc_CDR_ERF(RF_ref, E, scenario, policy_start_year=2020):  #
    """
    Calculate CDR rates under the ERF approach by numerically inverting the Etminan equation (2016)
    Args:
        RF_ref: RCP2.6 ERF
        E: Emissions under the scenario considered
        scenario: scenario of agricultural methane emissions, e.g. 'SSP1-26', 'const', or 'SSP3-70 (Baseline)'
        policy_start_year: start year of the offsetting policy

    Returns: CDR rates (negative sign = negative CO2 emissions)

    """
    E_CDR = [0] * 15  # for internal consistency, E_CDR has to start in 2000
    # copy emissions in order not to modify the original ones
    E_tmp = E.copy()
    # calculate in each year the CDR needed to force the ERF on the RCP2.6 pathway
    for end_year in range(2015, 2101):
        Fair_inv = Fair_inversion(E_tmp, RF_ref, scenario, end_year)
        res = minimize_scalar(Fair_inv.cost_function)
        CDR_tmp = res.x
        print("In year " + str(end_year) + " CDR is " + str(CDR_tmp))
        E_CDR.append(CDR_tmp)
        E_tmp[end_year - 1765, 1] += CDR_tmp
    # policy can not start in the past. Restribute small pre-2020 CDR evenly to post-2020
    start_index = policy_start_year - 2000
    E_CDR = np.asarray(E_CDR)
    E_CDR[start_index:] += np.sum(E_CDR[:start_index]) / (E_CDR.size - start_index)
    E_CDR[:start_index] = 0
    return np.asarray(E_CDR)


def calc_CDR_GWP100(E_agri, GWP100=28, year=2020):
    """
    Function to calculate CDR in GtC from emission pathway with the GWP100-based approach

    Args:
        E_agri: array of agricultural methane emissions under a given scenario
        GWP100: GWP100 conversion metric; default: 28 (IPCC, 2014)
        year: start of offsetting policy; default: 2020

    Returns: array of CDR rates (negative sign = negative CO2 emissions)

    """
    CDR = np.zeros(len(E_agri))
    CDR[(year - 2000) :] = (
        -E_agri[(year - 2000) :] * GWP100 / (1000 * 3.677)
    )  # from MtCH4 to MtCO2 (w GWP100) to GtC (divided by 1000*3.67)
    return CDR


def calc_econs(CDR, cost_min, cost_mean, cost_max, E, E_2 = None, unit = 'tCH4', GWP100 = 28, GWP100_2= 265):
    """
    Function to calculate cost associated with CDR use and to calculate price on agricultural emissions
    needed to internalize the cost of additional CDR
    Args:
        CDR: additional or total CDR rates
        cost_min: lowest estimate of CDR cost
        cost_mean: mid-range estimate of CDR cost
        cost_max: highest estimate of CDR cost
        E: emissions on which the cost of CDR is levied (should be agricultural only)

    Returns: lowest, mid-range, and highest CDR cost per year and price on methane emissions per year

    """

    class Econs_outputs:
        """
        Class to store Fair output nicely
        """

        def __init__(
            self,
            CDR_cost_min,
            CDR_cost_mean,
            CDR_cost_max,
            ptax_min,
            ptax_mean,
            ptax_max,
        ):
            self.CDR_cost_min = CDR_cost_min
            self.CDR_cost_mean = CDR_cost_mean
            self.CDR_cost_max = CDR_cost_max
            self.ptax_min = ptax_min
            self.ptax_mean = ptax_mean
            self.ptax_max = ptax_max

    CDR_tCO2 = CDR.copy() * 3.677 * 10 ** 9  # to get CDR rates in tCO2/yr
    CDR_tCO2[
        CDR_tCO2 > 0
    ] = 0.0  # set all CDR rates above 0 equal to 0 since there is no cost associated with positive CDR emissions
    CDR_cost_min = -CDR_tCO2 * cost_min
    CDR_cost_mean = -CDR_tCO2 * cost_mean
    CDR_cost_max = -CDR_tCO2 * cost_max

    if unit == 'tCH4':
        ptax_min = CDR_cost_min / (
            E * 10 ** 6
        )  # divide CDR costs by emissions (E given in Mt, so E * 10 **6 in t) to get price per ton
        ptax_mean = CDR_cost_mean / (E * 10 ** 6)
        ptax_max = CDR_cost_max / (E * 10 ** 6)
    elif unit == 'tN2':
        ptax_min = CDR_cost_min / (
            E * 10 ** 6
        )  # divide CDR costs by emissions (E given in Mt, so E * 10 **6 in t) to get price per ton
        ptax_mean = CDR_cost_mean / (E * 10 ** 6)
        ptax_max = CDR_cost_max / (E * 10 ** 6)
    elif unit == 'tCO2':
        E_tot = ((E * GWP100 * 10 ** 6)+(E_2*(44.01/28.01)*GWP100_2*10**6)) # tot emissions in tCO2
        ptax_min = CDR_cost_min / E_tot
        ptax_mean = CDR_cost_mean / E_tot
        ptax_max = CDR_cost_max / E_tot
    return Econs_outputs(
        CDR_cost_min, CDR_cost_mean, CDR_cost_max, ptax_min, ptax_mean, ptax_max
    )


def calculate_deltap(p_tax, GHGintensity, GWP100):
    """
    Function to calculate tax-induced increase in price of agricultural commodity.

    Args:
        p_tax: array of tax price per methane emission unit
        GHGintensity: intensity of methane emissions of agricultural commodity; default: beef methane intensity
        GWP100: GWP100 to convert the methane emissions in CO2 equivalent; default: methane GWP100

    Returns: increase in price in agricultural commodity due to agricultural methane tax introduction

    """
    deltap = (
        (p_tax / GWP100) * GHGintensity / 10 ** 3
    )  # so that is in dollars per kg and not per ton
    return deltap


def mean_deltap(
    econs_const, econs_BAU, GHGintensity, GWP100=28, start_year_policy=2020
):
    """
    Calculating the mean increase in agricultural commodities with the policy

    Args:
        econs_const: class with all economic aspects of the policy, constant scenario
        econs_BAU: class with all economic aspects of the policy, constant scenario
        GHGintensity: methane intensity in kg/kgCO2-equivalent of a given agricultural product
        GWP100: Global Warming Potential; default: methane = 28 (IPCC, 2014)
        start_year_policy: year in which offsetting policy starts

    Returns: three lists with mean increase in price for both const. and BAU scenarios under mean, min
    and max CDR cost assumptions

    """
    index_policy = start_year_policy - 2000
    return (
        [
            np.mean(
                calculate_deltap(econs_const.ptax_mean, GHGintensity, GWP100)[
                    index_policy:
                ]
            ),
            np.mean(
                calculate_deltap(econs_BAU.ptax_mean, GHGintensity, GWP100)[
                    index_policy:
                ]
            ),
        ],
        [
            np.mean(
                calculate_deltap(econs_const.ptax_max, GHGintensity, GWP100)[
                    index_policy:
                ]
            ),
            np.mean(
                calculate_deltap(econs_BAU.ptax_max, GHGintensity, GWP100)[
                    index_policy:
                ]
            ),
        ],
        [
            np.mean(
                calculate_deltap(econs_const.ptax_min, GHGintensity, GWP100)[
                    index_policy:
                ]
            ),
            np.mean(
                calculate_deltap(econs_BAU.ptax_min, GHGintensity, GWP100)[
                    index_policy:
                ]
            ),
        ],
    )


def set_barplot(
    list1, list1_max, list1_min,
):
    """
    calculating inputs for barplots (values + error bars)
    Args:
        list1: values of bar plot (list)
        list1_max: values of upper limit of error bars (list)
        list1_min: values of lower limit of error bars (list)

    Returns: values for barplots + values for error bars

    """
    bar1 = np.array(list1)
    erbar1 = np.array(
        [
            (list1[0] - list1_min[0]),
            (list1[1] - list1_min[1]),
            (list1_max[0] - list1[0]),
            (list1_max[1] - list1[1]),
        ]
    ).reshape(2, 2)
    return bar1, erbar1


def make_quants_df(np_ens):
    """
    Calculate quantiles of ensemble simulations

    Args:
        np_ens: array with all ensemble members of simulation

    Returns: dataframe with different quantiles

    """
    ens_df = pd.DataFrame(np_ens)
    df_quantiles = {
        "Median": ens_df.quantile(q=0.5, axis=1),
        "2.5": ens_df.quantile(q=0.025, axis=1),
        "97.5": ens_df.quantile(q=0.975, axis=1),
        "Q1": ens_df.quantile(q=0.25, axis=1),
        "Q3": ens_df.quantile(q=0.75, axis=1),
        "Max": ens_df.max(axis=1),
        "Min": ens_df.min(axis=1),
    }
    df_quant = pd.DataFrame(df_quantiles)
    return df_quant


def export_CDRrates(
        dates,
        CDR_GWP_const_1,
        CDR_GWP_BAU_1,
        CDR_ERF_const_1,
        CDR_ERF_BAU_1,
        CDR_GWP_const_2 = None,
        CDR_GWP_BAU_2 = None,
        CDR_ERF_const_2 = None,
        CDR_ERF_BAU_2 = None,
        CDR_GWP_const_3 = None,
        CDR_GWP_BAU_3 = None,
        CDR_ERF_const_3 = None,
        CDR_ERF_BAU_3 = None,
):
    if CDR_GWP_const_2 is None:
        data = np.asarray([CDR_GWP_const_1, CDR_GWP_BAU_1, CDR_ERF_const_1, CDR_ERF_BAU_1]).T
        columns = ["GWP100 const", "GWP100 BAU", "ERF const", "ERF BAU"]
    elif CDR_GWP_const_2 is not None and CDR_GWP_const_3 is None:
        data = np.asarray([CDR_GWP_const_1, CDR_GWP_BAU_1, CDR_ERF_const_1, CDR_ERF_BAU_1,
                          CDR_GWP_const_2, CDR_GWP_BAU_2, CDR_ERF_const_2, CDR_ERF_BAU_2]).T
        columns = ["GWP100 const", "GWP100 BAU", "ERF const", "ERF BAU",
                   "GWP100 const 2", "GWP100 BAU 2", "ERF const 2", "ERF BAU 2"]
    else:
        data = np.asarray([CDR_GWP_const_1, CDR_GWP_BAU_1, CDR_ERF_const_1, CDR_ERF_BAU_1,
                           CDR_GWP_const_2, CDR_GWP_BAU_2, CDR_ERF_const_2, CDR_ERF_BAU_2,
                           CDR_GWP_const_3, CDR_GWP_BAU_3, CDR_ERF_const_3, CDR_ERF_BAU_3]).T
        columns = ["GWP100 const", "GWP100 BAU", "ERF const", "ERF BAU",
                   "GWP100 const 2", "GWP100 BAU 2", "ERF const 2", "ERF BAU 2",
                   "GWP100 const 3", "GWP100 BAU 3", "ERF const 3", "ERF BAU 3"]

    df = pd.DataFrame(
        data=data,
        index=dates,
        columns=columns,
    )
    return df.to_csv(r'CDRrates.csv')

def retrieve_CDRrates(inputs):
    """
    Function to retrieve saved CDR rates so to avoid recalculating them.
    Args:
        inputs: INT indicating the number of scenarios involved (CH4, N20, and CH4+N2O)

    Returns: arrays with different CDR rates

    """
    df = pd.read_csv('Outputs/CDRrates.csv')
    CDR_GWP_const_1 = df.iloc[:, 1].values
    CDR_GWP_BAU_1 = df.iloc[:, 2].values
    CDR_ERF_const_1 = df.iloc[:, 3].values
    CDR_ERF_BAU_1 = df.iloc[:, 4].values
    if inputs == 1:
        return CDR_GWP_const_1, CDR_GWP_BAU_1, CDR_ERF_const_1, CDR_ERF_BAU_1
    elif inputs > 1:
        CDR_GWP_const_2 = df.iloc[:, 5].values
        CDR_GWP_BAU_2 = df.iloc[:, 6].values
        CDR_ERF_const_2 = df.iloc[:, 7].values
        CDR_ERF_BAU_2 = df.iloc[:, 8].values
        if inputs == 2:
            return CDR_GWP_const_1, CDR_GWP_BAU_1, CDR_ERF_const_1, CDR_ERF_BAU_1,\
                   CDR_GWP_const_2, CDR_GWP_BAU_2, CDR_ERF_const_2, CDR_ERF_BAU_2
        elif inputs == 3:
            CDR_GWP_const_3 = df.iloc[:, 9].values
            CDR_GWP_BAU_3 = df.iloc[:, 10].values
            CDR_ERF_const_3 = df.iloc[:, 11].values
            CDR_ERF_BAU_3 = df.iloc[:, 12].values
            return CDR_GWP_const_1, CDR_GWP_BAU_1, CDR_ERF_const_1, CDR_ERF_BAU_1, \
                   CDR_GWP_const_2, CDR_GWP_BAU_2, CDR_ERF_const_2, CDR_ERF_BAU_2, \
                   CDR_GWP_const_3, CDR_GWP_BAU_3, CDR_ERF_const_3, CDR_ERF_BAU_3



def calc_CDR_GWPstar(E_agri, GWPH, H, year = 2020
):
    """

    Function to calculate the GWP* of SLCP as described in Allen et al. (2018) "A solution to the
    misrepresentations of CO2-equivalent emissions of short-lived climate pollutants
    under ambitious mitigation"

    Args:
        E_agri: Agricultural emissions to offset
        GWPH: GWP over time frame of H years
        H: number of years in the time frame of H

    Returns: E_GWP*_CDR

    """
    E_GWPstar_CDR = []
    for i in np.arange(1, len(E_agri)):
        E_GWPstar_CDR.append((E_agri[i]-E_agri[i-1])*GWPH*H)
    E_GWPstar_CDR = -np.array(E_GWPstar_CDR)/ (1000 * 3.677)
    E_GWPstar_CDR[E_GWPstar_CDR>0] = 0.
    E_GWPstar_CDR[:(year - 2000)] = 0.
    return E_GWPstar_CDR
