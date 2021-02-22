# get_ipython().run_line_magic('matplotlib', 'inline')

# Import functions and updated RCP2.6 emissions (from IIASA RCP database)
import make_plots
import CDR_functions
# reload packages everytime that you let the code run (to update changes)
from importlib import reload
reload(CDR_functions)
reload(make_plots)
# import functions
from CDR_functions import *
from make_plots import *

# Defining the variables
DATES = np.arange(2000, 2101, step=1)
INDEX_2000 = 235  # index of year 2000 in RCPs scenarios
INDEX_2101 = 336  # index of year 2101 in RCPs scenarios
CH4GWP100 = 28  # IPCC AR5 GWP100 of methane
N2OGWP100 = 265  # IPCC AR5 GWP100 of N2O
CH4GWP20 = 84  # IPCC AR5 GWP20 of CH4
N2OGWP20 = 264  # IPCC AR5 GWP20 of N2O
CH4GTP100 = 4  # IPCC AR5 GWP100 of methane
N2OGTP100 = 234  # IPCC AR5 GWP100 of N2O
CH4GTP20 = 67  # IPCC AR5 GWP20 of CH4
N2OGTP20 = 277  # IPCC AR5 GWP20 of N2O
CH4GWP100_AR4 = 25  # GWP100 of methane used in FAO report (to calculate GHG intensity)
N2OGWP100_AR4 = 298  # GWP100 of N2O used in FAO report (to calculate GHG intensity)
YEAR_POLICY = 2020  # set year in which offsetting policy starts
N_RUNS = 1000  # number of ensemble members
# Using GWP100 as in IPCC AR4
SCENARIO1 = "SSP1-26"
SCENARIO2 = "const"
SCENARIO3 = "SSP3-70 (Baseline)"
saved_CM = False  # switch to False to calculate CDR rates from scratch
N2Oanalysis = False # switch to True to perform same analysis for N2O

# ============== RUNNING CALCULATIONS =====================================================

# Make different agricultural methane + N2O emission pathways
E_CH4_agri_26, E_CH4_agri_const, E_CH4_agri_BAU, E_N2O_agri_26, E_N2O_agri_const, E_N2O_agri_BAU \
    = make_agri_scenarios('CH4&N2O')

# Make different input emission pathways
E_RCP26, E_const, E_BAU = make_scenarios('CH4&N2O')
# N2O
E_RCP26_N2Oonly, E_const_N2Oonly, E_BAU_N2Oonly = make_scenarios('N2O')
# CH4
E_RCP26_CH4only, E_const_CH4only, E_BAU_CH4only = make_scenarios()

# Calculate tot CDR in RCP2.6 scenario - without offsetting scheme
E_CDR_RCP26_IMAGE = calc_CDR_RCP_2100()


if N2Oanalysis == True:
    # run simulations of FaIR for RCP2.6, constant and BAU scenario, N2O
    fair26_N2Oonly = run_fair(E_RCP26_N2Oonly, SCENARIO1, N_RUNS)
    fairconst_N2Oonly = run_fair(E_const_N2Oonly, SCENARIO2, N_RUNS)
    fairBAU_N2Oonly = run_fair(E_BAU_N2Oonly, SCENARIO3, N_RUNS)
    print("Finished running FaIR simulations of input scenarios")

    if saved_CM == False:
        metricsN2O = [N2OGWP100, N2OGWP20]
        CDR_N2O = ["CDR_GWP100_N2O_const", "CDR_GWP20_N2O_const", "CDR_GWP100_N2O_BAU", "CDR_GWP20_N2O_BAU"]
        fair_N2O = ["fair_GWP100_N2O_const", "fair_GWP20_N2O_const", "fair_GWP100_N2O_BAU", "fair_GWP20_N2O_BAU"]
        names_fair_N2O = ["GWP100", "GWP20"]

        for i in np.arange(0, len(metricsN2O)):
            N2OCM = metricsN2O[i]
            # GWP100-based approach
            # To offset N2O: calculate CDR rates and scenarios with the GWP100-based offsetting approach
            CDR_N2O[i] = calc_CDR_GWP100(E_N2O_agri_const - E_N2O_agri_26,
                                         GWP100=N2OCM * (44.01 / 28.01))  # first convert to MtN2O
            CDR_N2O[i + 2] = calc_CDR_GWP100(E_N2O_agri_BAU - E_N2O_agri_26,
                                             GWP100=N2OCM * (44.01 / 28.01))  # first convert to MtN2O
            fair_N2O[i] = run_fair(E_const_N2Oonly, SCENARIO2, N_RUNS, CDR_N2O[i])
            fair_N2O[i + 2] = run_fair(E_BAU_N2Oonly, SCENARIO3, N_RUNS, CDR_N2O[i + 2])
            print("Finished CDR_GWPt ensemble simulation")
            plot_fair_CDR(names_fair_N2O[i], fair26_N2Oonly, fairconst_N2Oonly, fairBAU_N2Oonly,
                      fair_N2O[i], fair_N2O[i + 2])

    # GWP*-based approach
    CDR_GWPstar_N2O_const = calc_CDR_GWPstar(E_N2O_agri_const - E_N2O_agri_26, N2OGWP100 * (44.01 / 28.01), 100)
    CDR_GWPstar_N2O_BAU = calc_CDR_GWPstar(E_N2O_agri_BAU - E_N2O_agri_26, N2OGWP100 * (44.01 / 28.01), 100)
    # Run fair for constant & BAU agricultural N2O emissions + GWP*-based CDR offsetting
    fairconst_N2O_GWPstar = run_fair(E_const_N2Oonly, SCENARIO2, N_RUNS, CDR_GWPstar_N2O_const, index_end=2100 - 1765)
    fairBAU_N2O_GWPstar = run_fair(E_BAU_N2Oonly, SCENARIO3, N_RUNS, CDR_GWPstar_N2O_BAU, index_end=2100 - 1765)
    print("Finished CDR_GWP* ensemble simulation")

    compare_metrics(fair26_N2Oonly,
                        fair_N2O[0],
                        fair_N2O[2],
                        fair_N2O[1],
                        fair_N2O[3],
                        fairconst_N2O_GWPstar,
                        fairBAU_N2O_GWPstar,
                        )


if N2Oanalysis == False:
    # run simulations of FaIR for RCP2.6, constant and BAU scenario, CH4
    fair26_CH4only = run_fair(E_RCP26_CH4only, SCENARIO1, N_RUNS)
    fairconst_CH4only = run_fair(E_const_CH4only, SCENARIO2, N_RUNS)
    fairBAU_CH4only = CDR_functions.run_fair(E_BAU_CH4only, SCENARIO3, N_RUNS)

    # run simulations of FaIR for RCP2.6, constant and BAU scenario, CH4 - shorter version
    fair26_CH4only_short = run_fair(E_RCP26_CH4only, SCENARIO1, N_RUNS, index_end=2100 - 1765)
    fairconst_CH4only_short = run_fair(E_const_CH4only, SCENARIO2, N_RUNS, index_end=2100 - 1765)
    fairBAU_CH4only_short = run_fair(E_BAU_CH4only, SCENARIO3, N_RUNS, index_end=2100 - 1765)
    print("Finished running FaIR simulations of input scenarios")

    if saved_CM == False:
        metricsCH4 = [CH4GWP100, CH4GWP20]
        metricsN2O = [N2OGWP100, N2OGWP20]
        CDR_CH4 = ["CDR_GWP100_CH4_const", "CDR_GWP20_CH4_const", "CDR_GWP100_CH4_BAU", "CDR_GWP20_CH4_BAU"]
        fair_CH4 = ["fair_GWP100_CH4_const", "fair_GWP20_CH4_const", "fair_GWP100_CH4_BAU", "fair_GWP20_CH4_BAU"]

        for i in np.arange(0, len(metricsCH4)):
            CH4CM = metricsCH4[i]
            CDR_CH4[i] = calc_CDR_GWP100(E_CH4_agri_const - E_CH4_agri_26, CH4CM)
            CDR_CH4[i + 2] = calc_CDR_GWP100(E_CH4_agri_BAU - E_CH4_agri_26, CH4CM)
            fair_CH4[i] = run_fair(E_const_CH4only, SCENARIO2, N_RUNS, CDR_CH4[i])
            fair_CH4[i + 2] = run_fair(E_BAU_CH4only, SCENARIO3, N_RUNS, CDR_CH4[i + 2])
            print("Finished CDR_GWPt ensemble simulation")

    # GWP*-based approach
    CDR_GWPstar_CH4_const = calc_CDR_GWPstar(E_CH4_agri_const - E_CH4_agri_26, CH4GWP100, 100)
    CDR_GWPstar_CH4_BAU = calc_CDR_GWPstar(E_CH4_agri_BAU - E_CH4_agri_26, CH4GWP100, 100)
    # Run fair for constant & BAU agricultural N2O emissions + GWP*-based CDR offsetting
    fairconst_CH4_GWPstar = run_fair(E_const_CH4only, SCENARIO2, N_RUNS, CDR_GWPstar_CH4_const, index_end=2100 - 1765)
    fairBAU_CH4_GWPstar = run_fair(E_BAU_CH4only, SCENARIO3, N_RUNS, CDR_GWPstar_CH4_BAU, index_end=2100 - 1765)
    print("Finished CDR_GWP* ensemble simulation")

# =============================== PRODUCE OUTPUTS ============================================================

    # Start writing output
    output_file = open("Outputs/offset_differentCM_output.txt", "w")

    output_file.write(
        "\n By 2100 in const (CH4): increase in ERF = "
        + str(np.sum(fairconst_CH4only.F_single - fair26_CH4only.F_single, axis=1)[335])
        + "  & in dT = "
        + str(fairconst_CH4only.T_single[-1] - fair26_CH4only.T_single[-1])
        + "\n By 2100 in BAU (CH4): increase in ERF = "
        + str(np.sum(fairBAU_CH4only.F_single - fair26_CH4only.F_single, axis=1)[335])
        + "  & in dT = "
        + str(fairBAU_CH4only.T_single[-1] - fair26_CH4only.T_single[-1])
    )

    names_CDR_CH4 = ["CDR_GWP100_CH4_const", "CDR_GWP20_CH4_const", "CDR_GWP100_CH4_BAU", "CDR_GWP20_CH4_BAU"]
    names_fair_CH4 = ["GWP100", "GWP20"]

    compare_metrics(fair26_CH4only,
                        fair_CH4[0],
                        fair_CH4[2],
                        fair_CH4[1],
                        fair_CH4[3],
                        fairconst_CH4_GWPstar,
                        fairBAU_CH4_GWPstar,
                        )

    for i in np.arange(0, len(metricsCH4)):

        # Check when maximal additional CDR rates
        output_file.write(
            "\n max " + names_CDR_CH4[i]
            + str(np.amin(CDR_CH4[i]))
            + " in year "
            + str(*np.where(CDR_CH4[i] == np.amin(CDR_CH4[i])))
            + "\n max : " +  names_CDR_CH4[i+2]
            + str(np.amin(CDR_CH4[i+2]))
            + " in year "
            + str(*np.where(CDR_CH4[i+2] == np.amin(CDR_CH4[i+2])))
        )

        # Producing simulations of ERF and T anomalies under the two offsetting approaches
        # Plot ERF & T anomalies for GWP100-based offsetting
        #plot_fair_CDR(names_fair_tot[i], fair26, fairconst, fairBAU, fair_tot[i], fair_tot[i+2])
        plot_fair_CDR(names_fair_CH4[i], fair26_CH4only, fairconst_CH4only, fairBAU_CH4only,
                      fair_CH4[i], fair_CH4[i+2])
        #plot_fair_CDR(names_fair_N2O[i], fair26_N2Oonly, fairconst_N2Oonly, fairBAU_N2Oonly,
         #             fair_N2O[i], fair_N2O[i+2])

        output_file.write(
            "\n CH4 only:"
            + "\nConst: Mean divergence between "+ names_fair_CH4[i] + " & RCP2.6; ERF (CH4&N2O): "
            + str(np.mean(np.sum(fair26_CH4only.F_single[INDEX_2000:, :]
                                 - fair_CH4[i].F_single[INDEX_2000:, :], axis=1)))
            + " & T anomaly: "
            + str(
                np.mean(fair26_CH4only.T_single[INDEX_2000:] - fair_CH4[i].T_single[INDEX_2000:])
            )
            + "\n BAU: Mean divergence between"+ names_fair_CH4[i] + " & RCP2.6; ERF: "
            + str(
                np.mean(
                    np.sum(fair26_CH4only.F_single[INDEX_2000:, :]
                           - fair_CH4[i+2].F_single[INDEX_2000:, :], axis=1)))
            + " & T anomaly: "
            + str(np.mean(fair26_CH4only.T_single[INDEX_2000:] - fair_CH4[i+2].T_single[INDEX_2000:]))
        )


        output_file.write(
            "\n CH4 only:"
            + "\n const: 2100 divergence between "+ names_fair_CH4[i] + " & RCP2.6; ERF: "
            + str(np.sum(fair26_CH4only.F_single[INDEX_2101 - 1, :]
                         - fair_CH4[i].F_single[INDEX_2101 - 1, :]))
            + " & T anomaly: "
            + str(fair26_CH4only.T_single[INDEX_2101 - 1] - fair_CH4[i].T_single[INDEX_2101 - 1])
            + "\n BAU: 2100 divergence between"+ names_fair_CH4[i] + " & RCP2.6; ERF: "
            + str(np.sum(fair26_CH4only.F_single[INDEX_2101 - 1, :]
                         - fair_CH4[i+2].F_single[INDEX_2101 - 1, :]))
            + " & T anomaly: "
            + str(fair26_CH4only.T_single[INDEX_2101 - 1] - fair_CH4[i+2].T_single[INDEX_2101 - 1])
        )

    # Check when maximal additional CDR rates
    output_file.write(
        "\n max CDR GWP* const:"
        + str(np.amin(CDR_GWPstar_CH4_const))
        + " in year "
        + str(*np.where(CDR_GWPstar_CH4_const == np.amin(CDR_GWPstar_CH4_const)))
        + "\n max CDR GWP* BAU: "
        + str(np.amin(CDR_GWPstar_CH4_BAU))
        + " in year "
        + str(*np.where(CDR_GWPstar_CH4_BAU == np.amin(CDR_GWPstar_CH4_BAU)))
    )
    output_file.write(
        "\n CH4 only:"
        + "\n const: 2100 divergence between GWP* & RCP2.6; ERF: "
        + str(np.sum(fair26_CH4only_short.F_single[INDEX_2101 -2, :]
                     - fairconst_CH4_GWPstar.F_single[INDEX_2101 - 2, :]))
        + " & T anomaly: "
        + str(fair26_CH4only.T_single[INDEX_2101 - 2]
              - fairconst_CH4_GWPstar.T_single[INDEX_2101 - 2])
        + "\n BAU: 2100 divergence between GWP* & RCP2.6; ERF: "
        + str(np.sum(fair26_CH4only.F_single[INDEX_2101 - 2, :]
                     - fairBAU_CH4_GWPstar.F_single[INDEX_2101 - 2, :]))
        + " & T anomaly: "
        + str(fair26_CH4only.T_single[INDEX_2101 - 2]
              - fairBAU_CH4_GWPstar.T_single[INDEX_2101 - 2])
    )

    output_file.write(
        "\n CH4 only:"
        + "\n const: max divergence between GWP* & RCP2.6; ERF: "
        + str(np.amax(np.sum(fair26_CH4only_short.F_single[INDEX_2000:, :]
                             - fairconst_CH4_GWPstar.F_single[INDEX_2000:, :], axis=1)))
        + " & T anomaly:"
        + str(np.amax(fair26_CH4only_short.T_single[INDEX_2000:]
                      - fairconst_CH4_GWPstar.T_single[INDEX_2000:]))
        + "\n BAU: max divergence between GWP* & RCP2.6; ERF: "
        + str(np.amax(np.sum(fair26_CH4only_short.F_single[INDEX_2000:, :]
                             - fairBAU_CH4_GWPstar.F_single[INDEX_2000:, :], axis=1)))
        + " & T anomaly:"
        + str(np.amax(fair26_CH4only_short.T_single[INDEX_2000:]
                      - fairBAU_CH4_GWPstar.T_single[INDEX_2000:]))
    )

    plot_fair_CDR("GWP*", fair26_CH4only_short, fairconst_CH4only_short, fairBAU_CH4only_short,
                  fairconst_CH4_GWPstar, fairBAU_CH4_GWPstar)


    output_file.close()
