# get_ipython().run_line_magic('matplotlib', 'inline')

# Import functions and updated RCP2.6 emissions (from IIASA RCP database)
import make_plots
import CDR_functions
from importlib import reload
reload(CDR_functions)
reload(make_plots)
from CDR_functions import *
from make_plots import *


# Defining the variables
DATES = np.arange(2000, 2101, step=1)
INDEX_2000 = 235  # index of year 2000 in RCPs scenarios
INDEX_2101 = 336  # index of year 2101 in RCPs scenarios
CH4GWP100 = 28  # IPCC AR5 GWP100 of methane
N2OGWP100 = 265  # IPCC AR5 GWP100 of N2O
CH4GWP100_AR4 = 25 # GWP100 of methane used in FAO report (to calculate GHG intensity)
N2OGWP100_AR4 = 298 # GWP100 of N2O used in FAO report (to calculate GHG intensity)
YEAR_POLICY = 2020  # set year in which offsetting policy starts
N_RUNS = 1000  # number of ensemble members
P_USA_BEEF = 12.5 # USDA all beef products average, average 2018-2020
P_USA_RICE = 1.58 # US Statistic Bureau
P_USA_MILK = 0.83*1.03 # USDA milk price average over 2018-2020
P_W_BEEF = 4.45 # World Bank commodity price, average 2017-2019, $/kg
rice = [398.9, 420.7, 418.0, 384.7, 408.1, 410.4, 379.9, 401.1, 393.5,  363.2, 406.1, 351.9]
P_W_RICE = np.mean(rice)/1000 # World Bank commodity price, average 2017-2019, $/kg
P_W_MILK =  1.56*1.03 # Average price between largest milk-consumers in 2017, Statista, $/kg
CH4INT_BEEF = (0.426+0.014)*46.2 # from FAO
# Using GWP100 as in IPCC AR4
N2OINT_BEEF = (0.181+0.074+0.036)*46.2 # from FAO
# Using GWP100 as in IPCC AR4
CH4INT_RICE = (6550/(6550+2940+92.4))*1.47 #Brodt et al. (2014) Table 6
# Using GWP100 as in IPCC AR4
N2OINT_RICE = (2940/(6550+2940+92.4))*1.47 #Brodt et al. (2014) Table 6
# Using GWP100 as in IPCC AR4
CH4INT_MILK = (0.465+0.038)*2.8 #from FAO
# Using GWP100 as in IPCC AR4
N2OINT_MILK = (0.170+0.074+0.054)*2.8 #from FAO
# Using GWP100 as in IPCC AR4
CDR_min = 35 # minimal CDR cost in Fuss et al. (2018)
CDR_mean = 150 # average CDR cost in Fuss et al. (2018)
CDR_max = 235 # maximal CDR cost in Fuss et al. (2018)
SCENARIO1 = "SSP1-26"
SCENARIO2 = "const"
SCENARIO3 = "SSP3-70 (Baseline)"
saved_GWP100 = True # switch to False to calculate CDR rates from scratch
saved_ERF = True # switch to False to calculate CDR rates from scratch
price = "Global"

# ============== RUNNING CALCULATIONS =====================================================

# Make different agricultural methane + N2O emission pathways
E_CH4_agri_26, E_CH4_agri_const, E_CH4_agri_BAU, E_N2O_agri_26, E_N2O_agri_const, E_N2O_agri_BAU \
    = make_agri_scenarios('CH4&N2O')

# Make different input emission pathways
E_RCP26, E_const, E_BAU = make_scenarios('CH4&N2O')

# Calculate tot CDR in RCP2.6 scenario - without offsetting scheme
E_CDR_RCP26_IMAGE = calc_CDR_RCP_2100()

# run simulations of FaIR for RCP2.6, constant and BAU scenario, CH4 & N2O
fair26 = run_fair(E_RCP26, SCENARIO1, N_RUNS)
fairconst = run_fair(E_const, SCENARIO2, N_RUNS)
fairBAU = run_fair(E_BAU, SCENARIO3, N_RUNS)

print("Finished running FaIR simulations of input scenarios")
#=================================== CALCULATE CDR SCENARIOS =============================
if saved_GWP100 == False:
    # GWP100-based approach
    # To offset methane: calculate CDR rates and scenarios with the GWP100-based offsetting approach
    CDR_GWP100_CH4_const = calc_CDR_GWP100(E_CH4_agri_const - E_CH4_agri_26)
    CDR_GWP100_CH4_BAU = calc_CDR_GWP100(E_CH4_agri_BAU - E_CH4_agri_26)
    # To offset N2O: calculate CDR rates and scenarios with the GWP100-based offsetting approach
    CDR_GWP100_N2O_const = calc_CDR_GWP100(E_N2O_agri_const - E_N2O_agri_26, N2OGWP100*(44.01 / 28.01)) # first convert to MtN2O
    CDR_GWP100_N2O_BAU = calc_CDR_GWP100(E_N2O_agri_BAU - E_N2O_agri_26, N2OGWP100*(44.01 / 28.01)) # first convert to MtN2O
    # Tot. CDR rates
    CDR_GWP100_const = CDR_GWP100_CH4_const + CDR_GWP100_N2O_const
    CDR_GWP100_BAU = CDR_GWP100_CH4_BAU + CDR_GWP100_N2O_BAU
if saved_ERF == False:
    # Calculate CDR rates and scenarios with the ERF-based offsetting approach #CH4+N2O
    CDR_ERF_const = calc_CDR_ERF(fair26.F_single, E_const, SCENARIO2, policy_start_year=2020)
    CDR_ERF_BAU = calc_CDR_ERF(
        fair26.F_single, E_BAU, SCENARIO3, policy_start_year=2020
    )
else:
    CDR_GWP100_const, CDR_GWP100_BAU, CDR_ERF_const, \
    CDR_ERF_BAU = retrieve_CDRrates(1)

#================================ CALCULATE ERF AND T =========================================

# GWP100-based approach
# Run fair for constant & BAU agricultural N2O+CH4 emissions + GWP100-based CDR offsetting
fairconst_GWP100 = run_fair(E_const, SCENARIO2, N_RUNS, CDR_GWP100_const)
fairBAU_GWP100 = run_fair(E_BAU, SCENARIO3, N_RUNS, CDR_GWP100_BAU)
print("Finished ensemble simulation GWP100")

# ERF-based approach
# Run fair for constant & BAU agricultural CH4+N2O emissions + ERF-based CDR offsetting
fairconst_ERF = run_fair(E_const, SCENARIO2, N_RUNS, CDR_ERF_const)
fairBAU_ERF = run_fair(E_BAU, SCENARIO3, N_RUNS, CDR_ERF_BAU)
print("Finished ensemble simulation ERF")

#============================ ECONOMIC CALCULATIONS ======================================
# Calculate the policy-related cost ($/yr) and tax ($/tCH4) in the two policy scenarios (ERF-based offsetting)
econs_const = calc_econs(CDR_ERF_const, CDR_min, CDR_mean, CDR_max, E_CH4_agri_const, E_N2O_agri_const, unit = "tCO2")
econs_BAU = calc_econs(CDR_ERF_BAU, CDR_min, CDR_mean, CDR_max, E_CH4_agri_BAU, E_N2O_agri_BAU, unit = "tCO2")

# Calculate the total CDR cost ($/yr) in the two policy scenarios (ERF-based offsetting)
econs_tot_const = calc_econs(CDR_ERF_const + E_CDR_RCP26_IMAGE, CDR_min, CDR_mean, CDR_max,
                             E_CH4_agri_const, E_N2O_agri_const, unit = "tCO2")
econs_tot_BAU = calc_econs(CDR_ERF_BAU + E_CDR_RCP26_IMAGE, CDR_min, CDR_mean, CDR_max,
                           E_CH4_agri_BAU, E_N2O_agri_BAU, unit = "tCO2")

# CH4&N2O: calculate mean change in price of agricultural commodities due to policy using GWP100
mean_deltap_beef = mean_deltap(econs_const, econs_BAU, GHGintensity=N2OINT_BEEF*N2OGWP100/(N2OGWP100_AR4)+
                                                                    CH4INT_BEEF*CH4GWP100/(CH4GWP100_AR4), GWP100=1)
mean_deltap_rice = mean_deltap(econs_const, econs_BAU, GHGintensity=N2OINT_RICE*N2OGWP100/(N2OGWP100_AR4)+
                                                                    CH4INT_RICE*CH4GWP100/(CH4GWP100_AR4), GWP100=1)
mean_deltap_milk = mean_deltap(econs_const, econs_BAU, GHGintensity=N2OINT_MILK*N2OGWP100/(N2OGWP100_AR4)
                                                                    +CH4INT_MILK*CH4GWP100/(CH4GWP100_AR4), GWP100=1)

# =============================== PRODUCE OUTPUTS ============================================================

# Start writing output
output_file = open("Outputs/offset_CH4N2O_output.txt", "w")
# Plot input CH4&N2O scenarios
plot_input("CH4N2O",E_RCP26, E_BAU, E_const, E_CH4_agri_BAU, E_CH4_agri_const, E_CH4_agri_26, E_N2O_agri_BAU, E_N2O_agri_const, E_N2O_agri_26)
# Plot agri CH4&N2O scenarios
plot_agriemissions("CH4N2O",E_RCP26,E_CH4_agri_BAU, E_CH4_agri_const, E_CH4_agri_26, E_N2O_agri_BAU, E_N2O_agri_const, E_N2O_agri_26)


# Plot share of agricultural CH4&N2O
plot_percentage_agri("CH4N2O",E_RCP26, E_BAU, E_const, E_CH4_agri_BAU, E_CH4_agri_const, E_CH4_agri_26, E_N2O_agri_BAU, E_N2O_agri_const, E_N2O_agri_26)

# Calculate decrease in agricultural emissions in the RCP2.6 by 2100 relative to 2015
output_file.write(
    "\n % decrease in agricultural CH4 emissions in RCP2.6: "
    + str((E_CH4_agri_26[100] - E_CH4_agri_26[15]) / E_CH4_agri_26[15] * 100)
    + "\n % decrease in agricultural N2O emissions in RCP2.6: "
    + str((E_N2O_agri_26[100] - E_N2O_agri_26[15]) / E_N2O_agri_26[15] * 100)
)

# Difference in ERF and dT between RCP2.6 and alternative emission scenarios
output_file.write(
    "\n By 2100 in const (CH4&N2O): increase in ERF = "
    + str(np.sum(fairconst.F_single - fair26.F_single, axis=1)[335])
    + "  & in dT = "
    + str(fairconst.T_single[-1] - fair26.T_single[-1])
    +"\n By 2100 in BAU (CH4&N2O): increase in ERF = "
    + str(np.sum(fairBAU.F_single - fair26.F_single, axis=1)[335])
    + "  & in dT = "
    + str(fairBAU.T_single[-1] - fair26.T_single[-1])
)


# Check when maximal additional CDR rates
output_file.write(
    "\n max agri CDR GWP100 cost (CH4&N2O): "
    + str(np.amin(CDR_GWP100_const))
    + " in year "
    + str(*np.argwhere(CDR_GWP100_const == np.amin(CDR_GWP100_const))[0])
    + "\n max agri CDR GWP100 BAU (CH4&N2O): "
    + str(np.amin(CDR_GWP100_BAU))
    + " in year "
    + str(*np.argwhere(CDR_GWP100_BAU == np.amin(CDR_GWP100_BAU))[0])
)


# Check capacity of CDR vs. policy
output_file.write(
    "\n RCP2.6 CDR when > -0.75 GtC/yr (all):"
    + str(*np.argwhere(E_CDR_RCP26_IMAGE < -0.75)[0])
    + "\n ERF CDR + RCP2.6: BAU "
    + str(*np.argwhere(CDR_ERF_BAU+E_CDR_RCP26_IMAGE < -0.75)[0])
    + "\n ERF CDR + RCP2.6: const "
    + str(*np.argwhere(CDR_ERF_const+E_CDR_RCP26_IMAGE < -0.75)[0])
)

output_file.write(
    "\n RCP2.6 CDR when > -1.5 GtC/yr (all):"
    + str(*np.argwhere(E_CDR_RCP26_IMAGE < -1.5)[0])
    + "\n ERF CDR + RCP2.6: BAU "
    + str(*np.argwhere(CDR_ERF_BAU+E_CDR_RCP26_IMAGE < -1.5)[0])
    + "\n ERF CDR + RCP2.6: const "
    + str(*np.argwhere(CDR_ERF_const+E_CDR_RCP26_IMAGE < -1.5)[0])
)

output_file.write(
    "\n Estimated max. CDR capacity by 2050: 2.3-5.75"
    + "\n 2050 tot CDR RCP2.6: "
    + str(E_CDR_RCP26_IMAGE[50])
    + "\n ERF CDR + RCP2.6: BAU "
    + str(E_CDR_RCP26_IMAGE[50]+CDR_ERF_BAU[50])
    + "\n ERF CDR + RCP2.6: const "
    + str(E_CDR_RCP26_IMAGE[50]+CDR_ERF_const[50])
)

output_file.write(
    "\n Estimated max. CDR capacity by 2100: 5-20"
    + "\n 2100 tot CDR RCP2.6: "
    + str(E_CDR_RCP26_IMAGE[-1])
    + "\n ERF CDR + RCP2.6: BAU "
    + str(E_CDR_RCP26_IMAGE[-1]+CDR_ERF_BAU[-1])
    + "\n ERF CDR + RCP2.6: const "
    + str(E_CDR_RCP26_IMAGE[-1]+CDR_ERF_const[-1])
)



# Check when maximal additional CDR rates
output_file.write(
    "\n max agri CDR ERF cost: "
    + str(np.amin(CDR_ERF_const))
    + " in year "
    + str(*np.argwhere(CDR_ERF_const == np.amin(CDR_ERF_const))[0])
    + "\n max agri CDR ERF BAU: "
    + str(np.amin(CDR_ERF_BAU))
    + " in year "
    + str(*np.argwhere(CDR_ERF_BAU == np.amin(CDR_ERF_BAU))[0])
)

# Plot policy CDR rates only under both GWP100-based and ERF-based offsetting approaches
plot_CDR_rates("all",CDR_GWP100_const, CDR_GWP100_BAU, CDR_ERF_const, CDR_ERF_BAU)

# Plot total CO2 emissions under both GWP100-based and ERF-based offsetting approaches
plot_totCO2(
    "CH4N2O",
    E_RCP26,
    E_const,
    E_BAU,
    E_CDR_RCP26_IMAGE,
    CDR_GWP100_const,
    CDR_GWP100_BAU,
    CDR_ERF_const,
    CDR_ERF_BAU,
    )

# Plot % CDR out of CO2 emissions under both GWP100-based and ERF-based offsetting approaches
plot_percentage_CDR(
    "CH4N2O",
    E_CDR_RCP26_IMAGE,
    CDR_GWP100_const,
    CDR_GWP100_BAU,
    CDR_ERF_const,
    CDR_ERF_BAU,
    )

# Check when CDR rates start being over > 0.1
output_file.write(
    "\n RCP2.6: start year of CDR > 0.1 in year: "
    + str(*np.argwhere(E_CDR_RCP26_IMAGE < -0.1)[0])
    + "\n GWP100: start year of CDR > 0.1, const: "
    + str(*np.argwhere(E_CDR_RCP26_IMAGE + CDR_GWP100_const < -0.1)[0])
    + " & BAU: "
    + str(*np.argwhere(E_CDR_RCP26_IMAGE + CDR_GWP100_BAU < -0.1)[0])
    + "\n ERF: start year of CDR > 0.1, const: "
    + str(*np.argwhere(E_CDR_RCP26_IMAGE + CDR_ERF_const < -0.1)[0])
    + " & BAU: "
    + str(*np.argwhere(E_CDR_RCP26_IMAGE + CDR_ERF_BAU < -0.1)[0])
)

# Producing simulations of ERF and T anomalies under the two offsetting approaches
# Plot ERF & T anomalies for GWP100-based offsetting
plot_fair_CDR("GWP100", fair26, fairconst, fairBAU, fairconst_GWP100, fairBAU_GWP100)
# Plot ERF & T anomalies for ERF-based offsetting
plot_fair_CDR("ERF", fair26, fairconst, fairBAU, fairconst_ERF, fairBAU_ERF)

# Check the divergence in ERF and T anomaly under the GWP100-based offsetting
output_file.write(
    "\n CH4 & N2O:"
    + "\n: Mean divergence between GWP100-based offsetting & RCP2.6; ERF (CH4&N2O): "
    + str(np.mean(np.sum(fair26.F_single[INDEX_2000:, :]
                - fairconst_GWP100.F_single[INDEX_2000:, :],axis=1)))
    + " & T anomaly: "
    + str(
        np.mean(fair26.T_single[INDEX_2000:] - fairconst_GWP100.T_single[INDEX_2000:])
    )
    + "\n BAU: Mean divergence between GWP100-based offsetting & RCP2.6; ERF: "
    + str(
        np.mean(
            np.sum(fair26.F_single[INDEX_2000:, :]
                   - fairBAU_GWP100.F_single[INDEX_2000:, :],axis=1)))
    + " & T anomaly: "
    + str(np.mean(fair26.T_single[INDEX_2000:] - fairBAU_GWP100.T_single[INDEX_2000:]))
)

output_file.write(
    "\n CH4&N2O:"
    + "\n const: 2100 divergence between GWP100-based offsetting & RCP2.6; ERF: "
    + str(np.sum(fair26.F_single[INDEX_2101 - 1, :]
            - fairconst_GWP100.F_single[INDEX_2101 - 1, :]))
    + " & T anomaly: "
    + str(fair26.T_single[INDEX_2101 - 1] - fairconst_GWP100.T_single[INDEX_2101 - 1])
    + "\n BAU: 2100 divergence between GWP100-based offsetting & RCP2.6; ERF: "
    + str(np.sum(fair26.F_single[INDEX_2101 - 1, :]
            - fairBAU_GWP100.F_single[INDEX_2101 - 1, :]))
    + " & T anomaly: "
    + str(fair26.T_single[INDEX_2101 - 1] - fairBAU_GWP100.T_single[INDEX_2101 - 1])
)

# Check the divergence in ERF and T anomaly under the ERF-based offsetting
output_file.write(
    "\n CH4 & N2O:"
    + "\n: Mean divergence between ERF-based offsetting & RCP2.6; ERF (CH4&N2O): "
    + str(np.mean(np.sum(fair26.F_single[INDEX_2000:, :]
                - fairconst_ERF.F_single[INDEX_2000:, :],axis=1)))
    + " & T anomaly: "
    + str(
        np.mean(fair26.T_single[INDEX_2000:] - fairconst_ERF.T_single[INDEX_2000:])
    )
    + "\n BAU: Mean divergence between ERF-based offsetting & RCP2.6; ERF: "
    + str(
        np.mean(
            np.sum(fair26.F_single[INDEX_2000:, :]
                   - fairBAU_ERF.F_single[INDEX_2000:, :],axis=1)))
    + " & T anomaly: "
    + str(np.mean(fair26.T_single[INDEX_2000:] - fairBAU_ERF.T_single[INDEX_2000:]))
)

output_file.write(
    "\n CH4&N2O:"
    + "\n const: 2100 divergence between ERF-based offsetting & RCP2.6; ERF: "
    + str(np.sum(fair26.F_single[INDEX_2101 - 1, :]
            - fairconst_ERF.F_single[INDEX_2101 - 1, :]))
    + " & T anomaly: "
    + str(fair26.T_single[INDEX_2101 - 1] - fairconst_ERF.T_single[INDEX_2101 - 1])
    + "\n BAU: 2100 divergence between ERF-based offsetting & RCP2.6; ERF: "
    + str(np.sum(fair26.F_single[INDEX_2101 - 1, :]
            - fairBAU_ERF.F_single[INDEX_2101 - 1, :]))
    + " & T anomaly: "
    + str(fair26.T_single[INDEX_2101 - 1] - fairBAU_ERF.T_single[INDEX_2101 - 1])
)

# Plot tax and % of total CDR financed
plot_econs("CH4&N2O", econs_const, econs_tot_const, econs_BAU, econs_tot_BAU, unit = 'tCO2eq', GWP100=1)

# Output mean tax
output_file.write(
    "\n CH4&N2O:"
    +"\n Mean tax price; const: "
    + str(np.mean(econs_const.ptax_mean))
    + " & BAU: "
    + str(np.mean(econs_BAU.ptax_mean))
)

# Output max tax
output_file.write(
    "\n CH4&N2O:"
    +"\n Max tax price; const: "
    + str(np.amax(econs_const.ptax_mean))
    + " in year: "
    + str(*np.argwhere(econs_const.ptax_mean == np.amax(econs_const.ptax_mean)))
    + "BAU: "
    + str(np.amax(econs_BAU.ptax_mean))
    + " in year: "
    + str(*np.argwhere(econs_BAU.ptax_mean == np.amax(econs_BAU.ptax_mean)))
)

output_file.write(
    "\n CH4&N2O: "
    +"\n Percentage CDR financed by tax by 2100; const: "
    + str(econs_const.CDR_cost_mean[-1] / econs_tot_const.CDR_cost_mean[-1] * 100)
    + " & BAU: "
    + str(econs_BAU.CDR_cost_mean[-1] / econs_tot_BAU.CDR_cost_mean[-1] * 100)
    + "\n CH4 only: "
    + "\n Percentage CDR financed by tax by 2100; const: "
)

if price == "Global":
    P_BEEF = P_W_BEEF
    P_MILK = P_W_MILK
    P_RICE = P_W_RICE
elif price == "USA":
    P_BEEF = P_USA_BEEF
    P_MILK = P_USA_MILK
    P_RICE = P_USA_RICE

# Plot change in price of agricultural commodities due to policy
plot_changeagriprice(
    "CH4&N2O_"+price,
    mean_deltap_beef,
    mean_deltap_rice,
    mean_deltap_milk,
    p_beef_US=P_USA_BEEF,
    p_rice_US=P_USA_RICE,
    p_milk_US=P_W_MILK,
    p_beef_W=P_W_BEEF,
    p_rice_W=P_W_RICE,
    p_milk_W=P_W_MILK,
)

# Output mean tax and increase in price relative to 2019 for agricultural commodities
# beef
output_file.write(
    "\n CH4&N2O: "
    + "\n Mean tax on beef; const: "
    + str(mean_deltap_beef[0][0])
    + " & BAU: "
    + str(mean_deltap_beef[0][1])
    + "%"
    + "\n Mean increase in beef price relative to "+price+"; const: "
    + str(mean_deltap_beef[0][0] / P_BEEF * 100)
    + "% & BAU: "
    + str(mean_deltap_beef[0][1] / P_BEEF * 100)
    + "%"
)


# rice
output_file.write(
    "\n CH4&N2O: "
    + "\n Mean tax on rice; const: "
    + str(mean_deltap_rice[0][0])
    + " & BAU: "
    + str(mean_deltap_rice[0][1])
    +  "\n Mean increase in rice price relative to"+price+"; const: "
    + str(mean_deltap_rice[0][0] / P_RICE * 100)
    + "% & BAU: "
    + str(mean_deltap_rice[0][1] / P_RICE * 100)
    + "%"
)

# milk
output_file.write(
    "\n CH4&N2O: "
    +"\n Mean tax on milk; const: "
    + str(mean_deltap_milk[0][0])
    + " & BAU: "
    + str(mean_deltap_milk[0][1])
    + "\n Mean increase in milk price relative to "+price+"; const: "
    + str(mean_deltap_milk[0][0] / P_MILK * 100)
    + "% & BAU: "
    + str(mean_deltap_milk[0][1] / P_MILK * 100)
    + "%"
)

# Output increase in tot CDR under our policy scenarios
output_file.write(
    "\n CH4&N20:"
    + "\n Total increase in CDR (%) ERF-approach; const: "
    + str(np.sum(CDR_ERF_const) / np.sum(E_CDR_RCP26_IMAGE) * 100)
    + "% & BAU: "
    + str(np.sum(CDR_ERF_BAU) / np.sum(E_CDR_RCP26_IMAGE) * 100)
    + "%"
)


# Output increase in tot CDR under our policy scenarios
output_file.write(
    "\n CH4&N20:"
    +"\n Total increase in CDR (%) GWP100-approach; const: "
    + str(np.sum(CDR_GWP100_const) / np.sum(E_CDR_RCP26_IMAGE) * 100)
    + "% & BAU: "
    + str(np.sum(CDR_GWP100_BAU) / np.sum(E_CDR_RCP26_IMAGE) * 100)
    + "%"
)

output_file.close()

if saved_GWP100 == False:
    export_CDRrates(
            DATES,
            CDR_GWP100_const,
            CDR_GWP100_BAU,
            CDR_ERF_const,
            CDR_ERF_BAU,
    )