# get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from CDR_functions import make_quants_df, set_barplot

# Define colour palette
RCP85 = "#4393c3"
RCP85_shade = "#77b1d4"
RCP60 = "#f46d43"
RCP60_shade = "#fd9a7c"

dgrey = "#737373"
grey = "#a6a6a6"
lgrey = "#d9d9d9"
ddgrey = "#424242"

# Defining the variables
DATES = np.arange(2000, 2101, step=1)
INDEX_2000 = 235  # index of year 2000 in RCPs scenarios
INDEX_2101 = 336  # index of year 2101 in RCPs scenarios
CH4GWP100 = 28  # IPCC AR5 GWP100 of methane
YEAR_POLICY = 2020  # set year in which offsetting policy starts
N_RUNS = 10  # number of ensemble members


def plot_input(name,
    E_RCP26, E_BAU, E_const, E_agri_BAU, E_agri_const, E_agri_26,
               E_agri_BAU_2=None, E_agri_const_2=None, E_agri_26_2=None
):
    """
    plot methane emissions under the three different emission scenarios

    Args:
        E_RCP26:
        E_BAU:
        E_const:
        E_CH4_agri_BAU:
        E_CH4_agri_const:
        E_CH4_agri_26:

    Returns:
    """
    if name == "CH4":
        plt.figure(figsize=(4, 2.5))
        plt.fill_between(
            DATES,
            E_BAU[INDEX_2000:INDEX_2101, 3],
            np.zeros(len(DATES)),
            color=RCP85_shade,
            alpha=0.3,
            label="Tot CH$_4$ (worst-case)",
        )
        plt.fill_between(
            DATES,
            E_const[INDEX_2000:INDEX_2101, 3],
            np.zeros(len(DATES)),
            color=RCP60_shade,
            alpha=0.6,
            label="Tot CH$_4$ (constant)",
        )
        plt.fill_between(
            DATES,
            E_RCP26[INDEX_2000:INDEX_2101, 3],
            np.zeros(len(DATES)),
            color=lgrey,
            alpha=1,
            label="Tot CH$_4$ (SSP1-2.6)",
        )
        plt.plot(DATES, E_agri_BAU, color=RCP85, label="Agricultural CH$_4$ (worst-case)")
        plt.plot(DATES, E_agri_const, color=RCP60, label="Agricultural CH$_4$ (constant)")
        plt.plot(DATES, E_agri_26, color=dgrey, label="Agricultural CH$_4$ (SSP1-2.6)")
        plt.ylabel("CH$_4$ emissions (MtCH$_4$/yr)")
        plt.rcParams["axes.labelsize"] = 10
        plt.legend(bbox_to_anchor=(1.05, 0.95))
    elif name == "N2O":
        plt.figure(figsize=(4, 2.5))
        plt.fill_between(
            DATES,
            E_BAU[INDEX_2000:INDEX_2101, 4],
            np.zeros(len(DATES)),
            color=RCP85_shade,
            alpha=0.3,
            label="Tot N$_2$O (worst-case)",
        )
        plt.fill_between(
            DATES,
            E_const[INDEX_2000:INDEX_2101, 4],
            np.zeros(len(DATES)),
            color=RCP60_shade,
            alpha=0.6,
            label="Tot N$_2$O (constant)",
        )
        plt.fill_between(
            DATES,
            E_RCP26[INDEX_2000:INDEX_2101, 4],
            np.zeros(len(DATES)),
            color=lgrey,
            alpha=1,
            label="Tot N$_2$O (SSP1-2.6)",
        )
        plt.plot(DATES, E_agri_BAU, color=RCP85, label="Agricultural N$_2$O (worst-case)")
        plt.plot(DATES, E_agri_const, color=RCP60, label="Agricultural N$_2$O (constant)")
        plt.plot(DATES, E_agri_26, color=dgrey, label="Agricultural N$_2$O (SSP1-2.6)")
        plt.ylabel("N$_2$O emissions (MtN$_2$/yr)")
        plt.rcParams["axes.labelsize"] = 10
        plt.legend(bbox_to_anchor=(1.05, 0.95))
    else:
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(7, 3))
        ax1, ax2= (
            ax.flatten()[0],
            ax.flatten()[1]
        )

        # Methane pathways
        ax1.fill_between(
            DATES,
            E_BAU[INDEX_2000:INDEX_2101, 3],
            np.zeros(len(DATES)),
            color=RCP85_shade,
            alpha=0.3,
            label="Tot (worst-case)",
        )
        ax1.fill_between(
            DATES,
            E_const[INDEX_2000:INDEX_2101, 3],
            np.zeros(len(DATES)),
            color=RCP60_shade,
            alpha=0.6,
            label="Tot (constant)",
        )
        ax1.fill_between(
            DATES,
            E_RCP26[INDEX_2000:INDEX_2101, 3],
            np.zeros(len(DATES)),
            color=lgrey,
            alpha=1,
            label="Tot (SSP1-2.6)",
        )
        ax1.plot(
            DATES, E_agri_BAU, color=RCP85, label="Agricultural (worst-case)"
        )
        ax1.plot(
            DATES, E_agri_const, color=RCP60, label="Agricultural (constant)"
        )
        ax1.plot(
            DATES, E_agri_26, color=dgrey, label="Agricultural (SSP1-2.6)"
        )

        # N2O emissions
        ax2.fill_between(
            DATES,
            E_BAU[INDEX_2000:INDEX_2101, 4],
            np.zeros(len(DATES)),
            color=RCP85_shade,
            alpha=0.3,
            label="Tot N$_2$O(worst-case)",
        )
        ax2.fill_between(
            DATES,
            E_const[INDEX_2000:INDEX_2101, 4],
            np.zeros(len(DATES)),
            color=RCP60_shade,
            alpha=0.6,
            label="Tot N$_2$O (constant)",
        )
        ax2.fill_between(
            DATES,
            E_RCP26[INDEX_2000:INDEX_2101, 4],
            np.zeros(len(DATES)),
            color=lgrey,
            alpha=1,
            label="Tot N$_2$O (SSP1-2.6)",
        )
        ax2.plot(
            DATES, E_agri_BAU_2, color=RCP85, label="Agricultural N$_2$O (worst-case)"
        )
        ax2.plot(
            DATES, E_agri_const_2, color=RCP60, label="Agricultural N$_2$O (constant)"
        )
        ax2.plot(
            DATES, E_agri_26_2, color=dgrey, label="Agricultural N$_2$O (SSP1-2.6)"
        )
        ax1.text(0.05, 0.9, "a", transform=ax1.transAxes, va="top", ha="left")
        ax2.text(0.05, 0.9, "b", transform=ax2.transAxes, va="top", ha="left")
        ax1.text(0.9, 0.05, "CH$_4$", transform=ax1.transAxes, va="bottom", ha="right")
        ax2.text(0.9, 0.05, "N$_2$O", transform=ax2.transAxes, va="bottom", ha="right")
        ax1.set_ylabel("CH$_4$ emissions (MtCH$_4$/yr)")
        ax1.legend(bbox_to_anchor=(0,1.02,2.2,1.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
        ax2.set_ylabel("N$_2$O emissions (MtN$_2$/yr)")
        plt.rcParams["axes.labelsize"] = 10
        plt.tight_layout(pad=0.9)

    plt.savefig("Figures/input_"+name+".png", dpi=850, bbox_inches="tight")

def plot_percentage_agri(name,
    E_RCP26, E_BAU, E_const, E_agri_BAU, E_agri_const, E_agri_26,
               E_agri_BAU_2=None, E_agri_const_2=None, E_agri_26_2=None
):
    """
    plot methane emissions under the three different emission scenarios

    Args:
        E_RCP26:
        E_BAU:
        E_const:
        E_CH4_agri_BAU:
        E_CH4_agri_const:
        E_CH4_agri_26:

    Returns:
    """
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(6, 2.5))
    ax1, ax2 = (
        ax.flatten()[0],
        ax.flatten()[1]
    )

    # Methane pathways
    ax1.plot(
        DATES,
        E_agri_BAU/E_BAU[INDEX_2000:INDEX_2101, 3]*100,
        color=RCP85_shade,
        label="CH$_4$ (worst-case)",
    )
    ax1.plot(
        DATES,
        E_agri_const/E_const[INDEX_2000:INDEX_2101, 3]*100,
        color=RCP60_shade,
        label="CH$_4$ (constant)",
    )
    ax1.plot(
        DATES,
        E_agri_26/E_RCP26[INDEX_2000:INDEX_2101, 3]*100,
        color=lgrey,
        label="CH$_4$ (SSP1-2.6)",
    )
    # N2O emissions
    ax2.plot(
        DATES,
        E_agri_BAU_2 / E_BAU[INDEX_2000:INDEX_2101, 4] * 100,
        color=RCP85_shade,
        label="N$_2$O (worst-case)",
    )
    ax2.plot(
        DATES,
        E_agri_const_2 / E_const[INDEX_2000:INDEX_2101, 4] * 100,
        color=RCP60_shade,
        label="N$_2$O (constant)",
    )
    ax2.plot(
        DATES,
        E_agri_26_2 / E_RCP26[INDEX_2000:INDEX_2101, 4] * 100,
        color=lgrey,
        alpha=1,
        label="N$_2$O (SSP1-2.6)",
    )
    ax1.text(0.05, 0.9, "a", transform=ax1.transAxes, va="top", ha="left")
    ax2.text(0.05, 0.9, "b", transform=ax2.transAxes, va="top", ha="left")
    ax1.set_ylabel("share agricultural CH$_4$ (%)")
    ax1.legend()
    ax2.legend()
    ax1.set_ylim(30, 100)
    ax2.set_ylim(30, 100)
    ax2.set_ylabel("share agricultural N$_2$O (%)")
    plt.rcParams["axes.labelsize"] = 11
    plt.tight_layout(pad=0.7)
    plt.savefig("Figures/share_" + name + ".png", dpi=850, bbox_inches="tight")



def plot_fair_simulations(
    E_BAU,
    E_const,
    E_RCP26,
    E_CH4_agri_BAU,
    E_CH4_agri_const,
    E_CH4_agri_26,
    E_CDR_RCP26_IMAGE,
    fair26,
    fairconst,
    fairBAU,
):
    """
    Plot concentrations, forcing, and temperatures
    Args:
        E_BAU:
        E_const:
        E_RCP26:
        E_CH4_agri_BAU:
        E_CH4_agri_const:
        E_CH4_agri_26:
        fair26:
        fairconst:
        fairBAU:

    Returns:

    """
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(9, 5))
    ax1, ax2, ax3, ax4 = (
        ax.flatten()[0],
        ax.flatten()[1],
        ax.flatten()[2],
        ax.flatten()[3],
    )

    # Methane pathways
    ax.flatten()[0].fill_between(
        DATES,
        E_BAU[INDEX_2000:INDEX_2101, 3],
        np.zeros(len(DATES)),
        color=RCP85_shade,
        alpha=0.3,
        label="Tot CH$_4$ (worst-case)",
    )
    ax.flatten()[0].fill_between(
        DATES,
        E_const[INDEX_2000:INDEX_2101, 3],
        np.zeros(len(DATES)),
        color=RCP60_shade,
        alpha=0.6,
        label="Tot CH$_4$ (constant)",
    )
    ax.flatten()[0].fill_between(
        DATES,
        E_RCP26[INDEX_2000:INDEX_2101, 3],
        np.zeros(len(DATES)),
        color=lgrey,
        alpha=1,
        label="Tot CH$_4$ (SSP1-2.6)",
    )
    ax.flatten()[0].plot(
        DATES, E_CH4_agri_BAU, color=RCP85, label="Agricultural CH$_4$ (worst-case)"
    )
    ax.flatten()[0].plot(
        DATES, E_CH4_agri_const, color=RCP60, label="Agricultural CH$_4$ (constant)"
    )
    ax.flatten()[0].plot(
        DATES, E_CH4_agri_26, color=dgrey, label="Agricultural CH$_4$ (SSP1-2.6)"
    )

    # CO2 emissions and CDR
    ax2.plot(
        DATES,
        (E_RCP26[INDEX_2000:INDEX_2101, 1] + E_RCP26[INDEX_2000:INDEX_2101, 2]),
        color=lgrey,
        alpha=1,
        label="Tot CO$_2$",
    )  # CDR emissions
    ax2.plot(DATES, E_CDR_RCP26_IMAGE, color=dgrey, label="CDR")
    ax2.hlines(0, 2000, 2100, color=ddgrey, linestyle="dashed", label="Net-zero CO$_2$")

    # Ensemble 2.6 - agri 8.5
    ax3.plot(
        DATES, fairBAU.F_tot_ens[INDEX_2000:INDEX_2101, :], color=RCP85_shade, alpha=0.1
    )
    ax4.plot(
        DATES, fairBAU.T_ens[INDEX_2000:INDEX_2101, :], color=RCP85_shade, alpha=0.1
    )

    # Ensemble 2.6
    ax3.plot(DATES, fair26.F_tot_ens[INDEX_2000:INDEX_2101, :], color=lgrey, alpha=0.1)
    ax4.plot(DATES, fair26.T_ens[INDEX_2000:INDEX_2101, :], color=lgrey, alpha=0.1)

    # Ensemble 2.6 - agri const
    ax3.plot(
        DATES,
        fairconst.F_tot_ens[INDEX_2000:INDEX_2101, :],
        color=RCP60_shade,
        alpha=0.1,
    )
    ax4.plot(
        DATES, fairconst.T_ens[INDEX_2000:INDEX_2101, :], color=RCP60_shade, alpha=0.1
    )

    # Single 2.6 - agri 8.5
    ax3.plot(
        DATES, np.sum(fairBAU.F_single[INDEX_2000:INDEX_2101, :], axis=1), color=RCP85
    )
    ax4.plot(DATES, fairBAU.T_single[INDEX_2000:INDEX_2101], color=RCP85, label="worst-case")

    # Single 2.6 - agri const
    ax3.plot(
        DATES, np.sum(fairconst.F_single[INDEX_2000:INDEX_2101, :], axis=1), color=RCP60
    )
    ax4.plot(
        DATES, fairconst.T_single[INDEX_2000:INDEX_2101], color=RCP60, label="constant"
    )

    # Single 2.6
    ax3.plot(
        DATES, np.sum(fair26.F_single[INDEX_2000:INDEX_2101, :], axis=1), color=dgrey
    )
    ax4.plot(DATES, fair26.T_single[INDEX_2000:INDEX_2101], color=dgrey, label="SSP1-2.6")
    ax3.hlines(2.6, 2000, 2100, color=ddgrey, linestyle="dashed", label="2,6 Wm$^{-2}$")
    ax4.hlines(1.5, 2000, 2100, color=ddgrey, linestyle="dashed", label="1.5°C target")

    ax4.set_ylim(0.8, 2)
    ax3.set_ylim(2, 3.8)
    ax1.set_ylim(0, 700)

    ax1.text(0.05, 0.9, "a", transform=ax1.transAxes, va="top", ha="left")
    ax2.text(0.05, 0.9, "b", transform=ax2.transAxes, va="top", ha="left")
    ax3.text(0.05, 0.9, "c", transform=ax3.transAxes, va="top", ha="left")
    ax4.text(0.05, 0.9, "d", transform=ax4.transAxes, va="top", ha="left")

    ax1.set_ylabel("CH$_4$ emissions (GtCH$_4$/yr)")
    ax1.legend(ncol=2)
    ax2.legend()
    ax4.legend(ncol=2)
    ax2.set_ylabel("CO$_2$ emissions (GtC/yr)")
    ax3.set_ylabel("Radiative forcing (Wm$^{-2}$)")
    ax4.set_ylabel("Temperature change (°C)")
    plt.rcParams["axes.labelsize"] = 11
    plt.tight_layout(pad=0.7)
    plt.savefig("Figures/newRCPs_ens_2100_ref.png", dpi=850, bbox_inches="tight")


def plot_totCO2(
    name,
    E_RCP26,
    E_const,
    E_BAU,
    E_CDR_RCP26_IMAGE,
    E_CDR_GWP100_const,
    E_CDR_GWP100_BAU,
    E_CDR_ERF_const,
    E_CDR_ERF_BAU,
):
    # Figure 1 - CDR rates
    fig, ax = plt.subplots(ncols=2, figsize=(6, 2.5))
    ax1, ax3 = ax.flatten()[0], ax.flatten()[1]

    dates = np.arange(2000, 2101)

    # GWP100-based appraoch
    ax1.plot(
        dates,
        E_RCP26[INDEX_2000:INDEX_2101, 1] + E_RCP26[INDEX_2000:INDEX_2101, 2],
        color=lgrey,
        alpha=1,
        label="Tot (SSP1-2.6)",
    )
    ax1.plot(
        dates,
        E_const[INDEX_2000:INDEX_2101, 1]
        + E_const[INDEX_2000:INDEX_2101, 2]
        + E_CDR_GWP100_const,
        color=RCP60_shade,
        alpha=0.6,
        label="Tot (const)",
    )
    ax1.plot(
        dates,
        E_BAU[INDEX_2000:INDEX_2101, 1]
        + E_BAU[INDEX_2000:INDEX_2101, 2]
        + E_CDR_GWP100_BAU,
        color=RCP85_shade,
        alpha=0.9,
        label="Tot (worst-case)",
    )
    ax1.plot(dates, E_CDR_RCP26_IMAGE, color=dgrey)  # , label = 'CDR (SSP1-2.6)'  )
    ax1.plot(
        dates[19:], E_CDR_RCP26_IMAGE[19:] + E_CDR_GWP100_BAU[19:], color=RCP85
    )  # , label='CDR (worst-case)')
    ax1.plot(
        dates[19:], E_CDR_RCP26_IMAGE[19:] + E_CDR_GWP100_const[19:], color=RCP60
    )  # , label='CDR (const)')
    ax1.hlines(0, 2000, 2100, color=ddgrey, linestyle="dashed", label="Net-zero CO$_2$")

    # ERF-based approach
    ax3.plot(
        dates,
        E_RCP26[INDEX_2000:INDEX_2101, 1] + E_RCP26[INDEX_2000:INDEX_2101, 2],
        color=lgrey,
        alpha=1,
        label="Tot CO$_2$ (SSP1-2.6)",
    )
    ax3.plot(
        dates,
        E_const[INDEX_2000:INDEX_2101, 1]
        + E_const[INDEX_2000:INDEX_2101, 2]
        + E_CDR_ERF_const,
        color=RCP60_shade,
        alpha=0.6,
        label="Tot CO$_2$ (constant)",
    )
    ax3.plot(
        dates,
        E_BAU[INDEX_2000:INDEX_2101, 1]
        + E_RCP26[INDEX_2000:INDEX_2101, 2]
        + E_CDR_ERF_BAU,
        color=RCP85_shade,
        alpha=0.9,
        label="Tot CO$_2$ (worst-case)",
    )
    ax3.plot(dates, E_CDR_RCP26_IMAGE, color=dgrey, label="CDR (SSP1-2.6)")
    ax3.plot(
        dates[19:],
        E_CDR_RCP26_IMAGE[19:] + E_CDR_ERF_BAU[19:],
        color=RCP85,
        label="CDR (worst-case)",
    )
    ax3.plot(
        dates[19:],
        E_CDR_RCP26_IMAGE[19:] + E_CDR_ERF_const[19:],
        color=RCP60,
        label="CDR (constant)",
    )
    ax3.hlines(0, 2000, 2100, color=ddgrey, linestyle="dashed", label="Net-zero CO$_2$")

    ax1.text(0.05, 0.05, "GWP100", transform=ax1.transAxes, va="bottom", ha="left")
    ax3.text(0.05, 0.05, "ERF", transform=ax3.transAxes, va="bottom", ha="left")
    ax1.text(0.95, 0.95, "a", transform=ax1.transAxes, va="top", ha="right")
    ax3.text(0.95, 0.95, "b", transform=ax3.transAxes, va="top", ha="right")

    ax1.set_ylabel("CO$_2$ emissions (GtC)")
    ax3.legend(bbox_to_anchor=(1.05, 0.95))
    plt.rcParams["axes.labelsize"] = 10
    plt.savefig("Figures/totCDRrates_"+name+".png", dpi=850, bbox_inches="tight")


def plot_CDR_rates(name, CDR_GWP100_const, CDR_GWP100_BAU, CDR_ERF_const, CDR_ERF_BAU):
    """
    Figure 1 - CDR rates
    Args:
        CDR_GWP100_const:
        CDR_GWP100_BAU:
        CDR_ERF_const:
        CDR_ERF_BAU:

    Returns:

    """
    fig, ax = plt.subplots(ncols=2, figsize=(6, 2.5))
    ax1, ax3 = ax[0], ax[1]
    # GWP100-based approach
    ax1.plot(
        DATES[YEAR_POLICY - 2000 :],
        -CDR_GWP100_const[YEAR_POLICY - 2000 :],
        color=RCP60,
        label="constant",
    )
    ax1.plot(
        DATES[YEAR_POLICY - 2000 :],
        -CDR_GWP100_BAU[YEAR_POLICY - 2000 :],
        color=RCP85,
        label="worst-case",
    )
    # ERF-based approach
    ax3.plot(
        DATES[YEAR_POLICY - 2000 :],
        -CDR_ERF_const[YEAR_POLICY - 2000 :],
        color=RCP60,
        label="constant",
    )
    ax3.plot(
        DATES[YEAR_POLICY - 2000 :],
        -CDR_ERF_BAU[YEAR_POLICY - 2000 :],
        color=RCP85,
        label="worst-case",
    )
    ax1.text(0.9, 0.9, "GWP100", transform=ax1.transAxes, va="top", ha="right")
    ax3.text(0.9, 0.9, "ERF", transform=ax3.transAxes, va="top", ha="right")
    ax1.set_ylim(-0.1, 4.5)
    ax3.set_ylim(-0.1, 4.5)
    ax1.set_ylabel("CDR rates (GtC removed)")
    ax1.legend(loc="upper left")
    ax3.legend(loc="upper left")
    plt.rcParams["axes.labelsize"] = 10
    plt.savefig("Figures/agriCDRrates_"+ name + ".png", dpi=850, bbox_inches="tight")


def plot_fair_CDR(name, fair26, fairconst, fairBAU, fairconst_policy, fairBAU_policy):
    # Plot concentrations, forcing, and temperatures under CDR policy and references
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 5))
    ax1, ax2, ax3, ax4 = (
        ax.flatten()[0],
        ax.flatten()[1],
        ax.flatten()[2],
        ax.flatten()[3],
    )

    if name == "GWP*":
        dates = np.arange(1765, 2100)
    else:
        dates = np.arange(1765, 2101)

    # Calculate ERF quantiles for ensemble simulation plotting
    fair26_Fens_quant = make_quants_df(fair26.F_tot_ens)
    fairconst_Fens_quant = make_quants_df(fairconst.F_tot_ens)
    fairBAU_Fens_quant = make_quants_df(fairBAU.F_tot_ens)
    fairconst_policy_Fens_quant = make_quants_df(fairconst_policy.F_tot_ens)
    fairBAU_policy_Fens_quant = make_quants_df(fairBAU_policy.F_tot_ens)

    # Calculate dT quantiles for ensemble simulation plotting
    fair26_Tens_quant = make_quants_df(fair26.T_ens)
    fairconst_Tens_quant = make_quants_df(fairconst.T_ens)
    fairBAU_Tens_quant = make_quants_df(fairBAU.T_ens)
    fairconst_policy_Tens_quant = make_quants_df(fairconst_policy.T_ens)
    fairBAU_policy_Tens_quant = make_quants_df(fairBAU_policy.T_ens)

    # Plot 1: RF const. scenario - RCP2.6, RCP2.6+const,RCP2.6+const+policy (GWP100)
    ax1.fill_between(
        dates,
        fair26_Fens_quant["2.5"],
        fair26_Fens_quant["97.5"],
        color=lgrey,
        alpha=0.2,
    )
    ax1.fill_between(
        dates,
        fairconst_Fens_quant["2.5"],
        fairconst_Fens_quant["97.5"],
        color=dgrey,
        alpha=0.2,
    )
    ax1.fill_between(
        dates,
        fairconst_policy_Fens_quant["2.5"],
        fairconst_policy_Fens_quant["97.5"],
        color=RCP60_shade,
        alpha=0.2,
    )
    ax1.plot(dates, np.sum(fair26.F_single, axis=1), color=dgrey, label="SSP1-2.6")
    ax1.plot(
        dates, np.sum(fairconst.F_single, axis=1), color=ddgrey, label="constant"
    )
    ax1.plot(
        dates,
        np.sum(fairconst_policy.F_single, axis=1),
        color=RCP60,
        label="constant + "+name+"-offset",
    )

    # Plot 2 - T const. scenario - RCP2.6, RCP2.6+const, RCP2.6+const+policy (GWP100)
    ax2.fill_between(
        dates,
        fair26_Tens_quant["2.5"],
        fair26_Tens_quant["97.5"],
        color=lgrey,
        alpha=0.3,
    )
    ax2.fill_between(
        dates,
        fairconst_Tens_quant["2.5"],
        fairconst_Tens_quant["97.5"],
        color=dgrey,
        alpha=0.3,
    )
    ax2.fill_between(
        dates,
        fairconst_policy_Tens_quant["2.5"],
        fairconst_policy_Tens_quant["97.5"],
        color=RCP60_shade,
        alpha=0.3,
    )
    ax2.plot(dates, fair26.T_single, color=dgrey, label="SSP1-2.6")
    ax2.plot(dates, fairconst.T_single, color=ddgrey, label="constant")
    ax2.plot(dates, fairconst_policy.T_single, color=RCP60, label="constant + "+name+"-offset")

    # Plot 3 - RF BAU. scenario - RCP2.6, RCP2.6+BAU, RCP2.6+BAU+policy (GWP100)
    ax3.fill_between(
        dates,
        fair26_Fens_quant["2.5"],
        fair26_Fens_quant["97.5"],
        color=lgrey,
        alpha=0.3,
    )
    ax3.fill_between(
        dates,
        fairBAU_Fens_quant["2.5"],
        fairBAU_Fens_quant["97.5"],
        color=dgrey,
        alpha=0.3,
    )
    ax3.fill_between(
        dates,
        fairBAU_policy_Fens_quant["2.5"],
        fairBAU_policy_Fens_quant["97.5"],
        color=RCP85_shade,
        alpha=0.3,
    )
    ax3.plot(dates, np.sum(fair26.F_single, axis=1), color=dgrey, label="SSP1-2.6")
    ax3.plot(dates, np.sum(fairBAU.F_single, axis=1), color=ddgrey, label="worst-case ")
    ax3.plot(
        dates,
        np.sum(fairBAU_policy.F_single, axis=1),
        color=RCP85,
        label="worst-case + "+name+"-offset",
    )

    # Plot 4 - T BAU scenario - RCP2.6, RCP2.6+BAU, RCP2.6+BAU+policy (GWP100)
    ax4.fill_between(
        dates,
        fair26_Tens_quant["2.5"],
        fair26_Tens_quant["97.5"],
        color=lgrey,
        alpha=0.3,
    )
    ax4.fill_between(
        dates,
        fairBAU_Tens_quant["2.5"],
        fairBAU_Tens_quant["97.5"],
        color=dgrey,
        alpha=0.3,
    )
    ax4.fill_between(
        dates,
        fairBAU_policy_Tens_quant["2.5"],
        fairBAU_policy_Tens_quant["97.5"],
        color=RCP85_shade,
        alpha=0.3,
    )
    ax4.plot(dates, fair26.T_single, color=dgrey, label="SSP1-2.6")
    ax4.plot(dates, fairBAU.T_single, color=ddgrey, label="worst-case ")
    ax4.plot(dates, fairBAU_policy.T_single, color=RCP85, label="worst-case + "+name+"-offset")

    ax1.text(0.05, 0.9, "a", transform=ax1.transAxes, va="top", ha="left")
    ax2.text(0.05, 0.9, "b", transform=ax2.transAxes, va="top", ha="left")
    ax3.text(0.05, 0.9, "c", transform=ax3.transAxes, va="top", ha="left")
    ax4.text(0.05, 0.9, "d", transform=ax4.transAxes, va="top", ha="left")

    ax1.set_ylabel("Radiative forcing (Wm$^{-2}$)")
    ax2.set_ylabel("Temperature change (°C)")
    ax3.set_ylabel("Radiative forcing (Wm$^{-2}$)")
    ax4.set_ylabel("Temperature change (°C)")

    ax1.set_xlim(2000, 2100)
    ax2.set_xlim(2000, 2100)
    ax3.set_xlim(2000, 2100)
    ax4.set_xlim(2000, 2100)

    ax1.set_ylim(2, 4)
    ax2.set_ylim(0.8, 2.2)
    ax3.set_ylim(2, 4)
    ax4.set_ylim(0.8, 2.2)

    ax1.legend(loc="lower center")
    ax2.legend(loc="lower right")
    ax3.legend(loc="lower center")
    ax4.legend(loc="lower right")

    plt.rcParams["axes.labelsize"] = 11
    plt.tight_layout(pad=0.7)
    if name == "GWP*":
        plt.savefig("Figures/RCPs_ens_GWPstar.png", dpi=850, bbox_inches="tight")
    else:
        plt.savefig("Figures/RCPs_ens_"+name+".png", dpi=850, bbox_inches="tight")


def plot_econs(name,
               econs_const,
               econs_tot_const,
               econs_BAU,
               econs_tot_BAU,
               unit = 'tCO2eq',
               GWP100=28):
    # Figure 2 - radiative forcing

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(6, 2.5))
    ax1, ax3 = ax.flatten()[0], ax.flatten()[1]

    # define variables
    dates = np.arange(2000, 2101)

    # Plot tax price
    if unit == 'tCO2eq':
        ax1.fill_between(
            dates[20:],
            econs_const.ptax_min[20:] / GWP100,
            econs_const.ptax_max[20:] / GWP100,
            color=RCP60_shade,
            alpha=0.1,
        )
        ax1.fill_between(
            dates[20:],
            econs_BAU.ptax_min[20:] / GWP100,
            econs_BAU.ptax_max[20:] / GWP100,
            color=RCP85_shade,
            alpha=0.1,
        )
        ax1.plot(
            dates[20:],
            econs_const.ptax_min[20:] / GWP100,
            "--",
            color=RCP60_shade,
            alpha=0.5,
        )
        ax1.plot(
            dates[20:],
            econs_const.ptax_max[20:] / GWP100,
            "--",
            color=RCP60_shade,
            alpha=0.5,
        )
        ax1.plot(
            dates[20:],
            econs_BAU.ptax_min[20:] / GWP100,
            "--",
            color=RCP85_shade,
            alpha=0.5,
        )
        ax1.plot(
            dates[20:],
            econs_BAU.ptax_max[20:] / GWP100,
            "--",
            color=RCP85_shade,
            alpha=0.5,
        )
        ax1.plot(dates[20:], econs_BAU.ptax_mean[20:] / GWP100, color=RCP85, label="worst-case")
        ax1.plot(
            dates[20:], econs_const.ptax_mean[20:] / GWP100, color=RCP60, label="constant")
        np.seterr(divide="ignore", invalid="ignore")
        ax1.set_ylabel("tax price (\$/tCO$_2$eq)")
    else:
        ax1.fill_between(
            dates[20:],
            econs_const.ptax_min[20:],
            econs_const.ptax_max[20:],
            color=RCP60_shade,
            alpha=0.1,
        )
        ax1.fill_between(
            dates[20:],
            econs_BAU.ptax_min[20:],
            econs_BAU.ptax_max[20:],
            color=RCP85_shade,
            alpha=0.1,
        )
        ax1.plot(
            dates[20:],
            econs_const.ptax_min[20:],
            "--",
            color=RCP60_shade,
            alpha=0.5,
        )
        ax1.plot(
            dates[20:],
            econs_const.ptax_max[20:],
            "--",
            color=RCP60_shade,
            alpha=0.5,
        )
        ax1.plot(
            dates[20:],
            econs_BAU.ptax_min[20:],
            "--",
            color=RCP85_shade,
            alpha=0.5,
        )
        ax1.plot(
            dates[20:],
            econs_BAU.ptax_max[20:],
            "--",
            color=RCP85_shade,
            alpha=0.5,
        )
        ax1.plot(dates[20:], econs_BAU.ptax_mean[20:], color=RCP85, label="worst-case")
        ax1.plot(
            dates[20:], econs_const.ptax_mean[20:], color=RCP60, label="constant"
        )
        np.seterr(divide="ignore", invalid="ignore")
        if unit == 'tCH4':
            ax1.set_ylabel("tax price (\$/tCH$_4$)")
        elif unit == 'tN2':
            ax1.set_ylabel("tax price (\$/tN$_2$)")
        elif unit == 'tCH4&tCO2eq':
            ax1.set_ylabel("tax price (\$/tCH$_4$)")
            ax1_sec = ax1.twinx()
            # set twin scale (convert tCH4 to tCO2eq)
            tCO2eq = lambda tCH4: tCH4/GWP100
            # get left axis limits
            ymin, ymax = ax1.get_ylim()
            # apply function and set transformed values to right axis limits
            ax1_sec.set_ylim((tCO2eq(ymin), tCO2eq(ymax)))
            # set an invisible artist to twin axes
            # to prevent falling back to initial values on rescale events
            ax1_sec.plot([], [])
            ax1_sec.set_ylabel("tax price (\$/tCO$_2$eq)")
        elif unit == 'tN2&tCO2eq':
            ax1.set_ylabel("tax price (\$/tN$_2$)")
            ax1_sec = ax1.twinx()
            # set twin scale (convert tCH4 to tCO2eq)
            tCO2eq = lambda tN2: tN2*(44.01/28.01)/GWP100
            # get left axis limits
            ymin, ymax = ax1.get_ylim()
            # apply function and set transformed values to right axis limits
            ax1_sec.set_ylim((tCO2eq(ymin), tCO2eq(ymax)))
            # set an invisible artist to twin axes
            # to prevent falling back to initial values on rescale events
            ax1_sec.plot([], [])
            ax1_sec.set_ylabel("tax price (\$/tCO$_2$eq)")
    # Percentage CDR due to policy
    percCDR_agriBAU = (
        econs_BAU.CDR_cost_mean[20:] / econs_tot_BAU.CDR_cost_mean[20:] * 100
    )
    percCDR_agriconst = (
        econs_const.CDR_cost_mean[20:] / econs_tot_const.CDR_cost_mean[20:] * 100
    )
    percCDR_agriBAU[0] = 100
    percCDR_agriconst[0] = 100
    ax3.plot(dates[20:], percCDR_agriBAU, color=RCP85, label="worst-case")
    ax3.plot(dates[20:], percCDR_agriconst, color=RCP60, label="constant")

    ax1.text(0.95, 0.9, "a", transform=ax1.transAxes, va="top", ha="right")
    ax3.text(0.95, 0.9, "b", transform=ax3.transAxes, va="top", ha="right")

    ax3.set_ylim(-5, 105)

    ax3.set_ylabel("CDR financed (%)")
    ax1.legend(loc="upper left")
    ax3.legend(loc="upper center")
    plt.rcParams["axes.labelsize"] = 11
    plt.tight_layout(pad=0.7)
    plt.savefig("Figures/taxandpercCDR_"+name+".png", dpi=850, bbox_inches="tight")
    # subplot_adjust --> giving margins


def plot_changeagriprice(
    name,
    mean_deltap_beef,
    mean_deltap_rice,
    mean_deltap_milk,
    p_beef_US,
    p_rice_US,
    p_milk_US,
    p_beef_W,
    p_rice_W,
    p_milk_W,
):

    # create new function / change calculate_deltap so that it just outputs the mean in the two scenarios as a list
    #

    barbeef, erbarbeef = set_barplot(
        mean_deltap_beef[0], mean_deltap_beef[1], mean_deltap_beef[2]
    )
    barrice, erbarrice = set_barplot(
        mean_deltap_rice[0], mean_deltap_rice[1], mean_deltap_rice[2]
    )
    barmilk, erbarmilk = set_barplot(
        mean_deltap_milk[0], mean_deltap_milk[1], mean_deltap_milk[2]
    )

    bar_dpbeef_US, erbar_dpbeef_US = set_barplot(
        np.array(mean_deltap_beef[0]) / p_beef_US * 100,
        np.array(mean_deltap_beef[1]) / p_beef_US * 100,
        np.array(mean_deltap_beef[2]) / p_beef_US * 100,
    )
    bar_dprice_US, erbar_dprice_US = set_barplot(
        np.array(mean_deltap_rice[0]) / p_rice_US * 100,
        np.array(mean_deltap_rice[1]) / p_rice_US * 100,
        np.array(mean_deltap_rice[2]) / p_rice_US * 100,
    )
    bar_dpmilk_US, erbar_dpmilk_US = set_barplot(
        np.array(mean_deltap_milk[0]) / p_milk_US * 100,
        np.array(mean_deltap_milk[1]) / p_milk_US * 100,
        np.array(mean_deltap_milk[2]) / p_milk_US * 100,
    )
    bar_dpbeef_W, erbar_dpbeef_W = set_barplot(
        np.array(mean_deltap_beef[0]) / p_beef_W * 100,
        np.array(mean_deltap_beef[1]) / p_beef_W * 100,
        np.array(mean_deltap_beef[2]) / p_beef_W * 100,
    )
    bar_dprice_W, erbar_dprice_W = set_barplot(
        np.array(mean_deltap_rice[0]) / p_rice_W * 100,
        np.array(mean_deltap_rice[1]) / p_rice_W * 100,
        np.array(mean_deltap_rice[2]) / p_rice_W * 100,
    )
    bar_dpmilk_W, erbar_dpmilk_W = set_barplot(
        np.array(mean_deltap_milk[0]) / p_milk_W * 100,
        np.array(mean_deltap_milk[1]) / p_milk_W * 100,
        np.array(mean_deltap_milk[2]) / p_milk_W * 100,
    )

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(8, 2.5))
    ax2, ax1, ax3 = ax.flatten()[0], ax.flatten()[1], ax.flatten()[2]

    # Barplot setup
    barWidth = 0.3
    x = ["constant", "worst-case"]
    y = [0.45, 1.45]
    r2 = [i + barWidth for i in y]
    r3 = [i + 2 * barWidth for i in y]

    ax2.bar(y, barbeef, yerr=erbarbeef, width=barWidth, label="Beef", color=RCP60)
    ax2.bar(r2, barmilk, yerr=erbarmilk, width=barWidth, label="Milk", color=grey)
    ax2.bar(r3, barrice, yerr=erbarrice, width=barWidth, label="Rice", color=RCP85)
    ax2.set_xticks([0.75, 1.75])
    ax2.set_xticklabels(x)
    ax2.set_yscale("log")

    ax3.bar(y, bar_dpbeef_US, yerr=erbar_dpbeef_US, width=barWidth, label="Beef", color=RCP60)
    ax3.bar(r2, bar_dpmilk_US, yerr=erbar_dpmilk_US, width=barWidth, label="Milk", color=grey)
    ax3.bar(
        r3, bar_dprice_US, yerr=erbar_dprice_US, width=barWidth, label="Rice", color=RCP85
    )
    ax3.set_xticks([0.75, 1.75])
    ax3.set_xticklabels(x)

    ax1.bar(y, bar_dpbeef_W, yerr=erbar_dpbeef_W, width=barWidth, label="Beef", color=RCP60)
    ax1.bar(r2, bar_dpmilk_W, yerr=erbar_dpmilk_W, width=barWidth, label="Milk", color=grey)
    ax1.bar(
        r3, bar_dprice_W, yerr=erbar_dprice_W, width=barWidth, label="Rice", color=RCP85
    )
    ax1.set_xticks([0.75, 1.75])
    ax1.set_xticklabels(x)

    ax2.text(0.95, 0.95, "a", transform=ax2.transAxes, va="top", ha="right")
    ax1.text(0.95, 0.95, "b", transform=ax1.transAxes, va="top", ha="right")
    ax1.text(0.05, 0.95, "Global", transform=ax1.transAxes, va="top", ha="left")
    ax3.text(0.95, 0.95, "c", transform=ax3.transAxes, va="top", ha="right")
    ax3.text(0.05, 0.95, "United States", transform=ax3.transAxes, va="top", ha="left")


    ax2.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    ax1.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    ax3.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    ax2.set_ylabel("tax price ($/kg)")
    ax1.set_ylabel("$\Delta$ price (%)")
    ax3.set_ylabel("$\Delta$ price (%)")
    plt.rcParams["axes.labelsize"] = 10
    plt.tight_layout()
    plt.savefig("Figures/summary_taxbeefrice_"+name+".png", dpi=850, bbox_inches="tight")

def plot_percentage_CDR(
        name,
        E_CDR_RCP26_IMAGE,
        E_CDR_GWP100_const,
        E_CDR_GWP100_BAU,
        E_CDR_ERF_const,
        E_CDR_ERF_BAU,
):
    """
    plot methane emissions under the three different emission scenarios

    Args:
        name: figure name
        E_RCP26: emissions RCP2.5
        E_const: emissions const. scenario
        E_BAU: emissions BAU scenario
        E_CDR_RCP26_IMAGE: CDR rates in RCP2.6
        E_CDR_GWP100_const: CDR rates GWP100-based const
        E_CDR_GWP100_BAU: CDR rates GWP100-based BAU
        E_CDR_ERF_const: CDR rates ERF-based const
        E_CDR_ERF_BAU: CDR rates ERF-based BAU

    Returns:
    """
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(6, 2.5))
    ax1, ax2 = (
        ax.flatten()[0],
        ax.flatten()[1]
    )

    start_year = 20

    #E_CDR_RCP26_IMAGE[E_CDR_RCP26_IMAGE > -0.001] = -0.01
    perc_CDR_GWP100_const = E_CDR_GWP100_const[start_year:]/(E_CDR_RCP26_IMAGE[start_year:]+E_CDR_GWP100_const[start_year:])
    perc_CDR_GWP100_BAU = E_CDR_GWP100_BAU[start_year:]/(E_CDR_RCP26_IMAGE[start_year:] + E_CDR_GWP100_BAU[start_year:])
    perc_CDR_ERF_const = E_CDR_ERF_const[start_year:]/(E_CDR_RCP26_IMAGE[start_year:]+E_CDR_ERF_const[start_year:])
    perc_CDR_ERF_BAU = E_CDR_ERF_BAU[start_year:]/(E_CDR_RCP26_IMAGE[start_year:] + E_CDR_ERF_BAU[start_year:])

    # Conversion metric
    ax1.plot(
        DATES[start_year:],
        perc_CDR_GWP100_BAU *100,
        color=RCP85_shade,
        label="worst-case",
    )
    ax1.plot(
        DATES[start_year:],
        perc_CDR_GWP100_const*100,
        color=RCP60_shade,
        label="constant",
    )
    # N2O emissions
    ax2.plot(
        DATES[start_year:],
        perc_CDR_ERF_BAU * 100,
        color=RCP85_shade,
        label="worst-case",
    )
    ax2.plot(
        DATES[start_year:],
        perc_CDR_ERF_const* 100,
        color=RCP60_shade,
        label="constant",
    )
    ax1.text(0.05, 0.9, "a", transform=ax1.transAxes, va="top", ha="left")
    ax2.text(0.05, 0.9, "b", transform=ax2.transAxes, va="top", ha="left")
    ax1.set_ylabel("$\Delta$CDR with policy (%)")
    ax1.legend()
    ax2.legend()
    #ax1.set_ylim(30, 100)
    #ax2.set_ylim(30, 100)
    ax2.set_ylabel("$\Delta$CDR with policy (%)")
    plt.rcParams["axes.labelsize"] = 11
    plt.tight_layout(pad=0.7)
    plt.savefig("Figures/shareCDR_" + name + ".png", dpi=850, bbox_inches="tight")


def plot_agriemissions(name,
    E_RCP26, E_agri_BAU, E_agri_const, E_agri_26,
               E_agri_BAU_2, E_agri_const_2, E_agri_26_2
):
    """
    plot methane emissions under the three different emission scenarios

    Args:
        E_RCP26:
        E_BAU:
        E_const:
        E_CH4_agri_BAU:
        E_CH4_agri_const:
        E_CH4_agri_26:

    Returns:
    """

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(6.5, 3))
    ax1, ax2= (
        ax.flatten()[0],
        ax.flatten()[1]
    )

    # Methane pathways
    ax1.plot(
        DATES, E_agri_BAU, color=RCP85, label="Agricultural (worst-case)"
    )
    ax1.plot(
        DATES, E_agri_const, color=RCP60, label="Agricultural (constant)"
    )
    ax1.plot(
        DATES, E_agri_26, color=dgrey, label="Agricultural (SSP1-2.6)"
    )
    ax1.plot(
        DATES, E_RCP26[INDEX_2000:INDEX_2101,3]-E_agri_26, color=ddgrey, label="Non-agricultural (all)"
    )
    # N2O emissions
    # Methane pathways
    ax2.plot(
        DATES, E_agri_BAU_2, color=RCP85, label="Agricultural (worst-case)"
    )
    ax2.plot(
        DATES, E_agri_const_2, color=RCP60, label="Agricultural (constant)"
    )
    ax2.plot(
        DATES, E_agri_26_2, color=dgrey, label="Agricultural (SSP1-2.6)"
    )
    ax2.plot(
        DATES, E_RCP26[INDEX_2000:INDEX_2101,4]-E_agri_26_2, color=ddgrey, label="Non-agricultural (all)"
    )
    ax1.text(0.05, 0.9, "a", transform=ax1.transAxes, va="top", ha="left")
    ax2.text(0.05, 0.9, "b", transform=ax2.transAxes, va="top", ha="left")
    ax1.text(0.9, 0.1, "CH$_4$", transform=ax1.transAxes, va="bottom", ha="right")
    ax2.text(0.9, 0.1, "N$_2$O", transform=ax2.transAxes, va="bottom", ha="right")
    ax1.set_ylabel("CH$_4$ emissions (MtCH$_4$/yr)")
    ax1.legend(bbox_to_anchor=(0,1.02,2,1.2), loc="lower left",
               mode="expand", borderaxespad=0, ncol=2)
    ax2.set_ylabel("N$_2$O emissions (MtN$_2$/yr)")
    plt.rcParams["axes.labelsize"] = 10
    plt.tight_layout(pad=0.9)
    plt.savefig("Figures/input_agri_"+name+".png", dpi=850, bbox_inches="tight")


def compare_metrics(fair26,
                    fairconst_GWP100,
                    fairBAU_GWP100,
                    fairconst_GWP20,
                    fairBAU_GWP20,
                    fairconst_GWPstar,
                    fairBAU_GWPstar,
                    ):
    # Plot concentrations, forcing, and temperatures under CDR policy and references
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 5))
    ax1, ax2, ax3, ax4 = (
        ax.flatten()[0],
        ax.flatten()[1],
        ax.flatten()[2],
        ax.flatten()[3],
    )
    
    dates_star = np.arange(1765, 2100)
    dates = np.arange(1765, 2101)

    # Calculate ERF quantiles for ensemble simulation plotting
    #fair26_Fens_quant = make_quants_df(fair26.F_tot_ens)
    fairconst_GWP100_Fens_quant = make_quants_df(fairconst_GWP100.F_tot_ens-fair26.F_tot_ens)
    fairBAU_GWP100_Fens_quant = make_quants_df(fairBAU_GWP100.F_tot_ens-fair26.F_tot_ens)
    fairconst_GWP20_Fens_quant = make_quants_df(fairconst_GWP20.F_tot_ens-fair26.F_tot_ens)
    fairBAU_GWP20_Fens_quant = make_quants_df(fairBAU_GWP20.F_tot_ens-fair26.F_tot_ens)
    fairconst_GWPstar_Fens_quant = make_quants_df(fairconst_GWPstar.F_tot_ens-fair26.F_tot_ens[0:335])
    fairBAU_GWPstar_Fens_quant = make_quants_df(fairBAU_GWPstar.F_tot_ens-fair26.F_tot_ens[0:335])

    # Calculate dT quantiles for ensemble simulation plotting
    #fair26_Tens_quant = make_quants_df(fair26.T_ens)
    fairconst_GWP100_Tens_quant = make_quants_df(fairconst_GWP100.T_ens-fair26.T_ens)
    fairBAU_GWP100_Tens_quant = make_quants_df(fairBAU_GWP100.T_ens-fair26.T_ens)
    fairconst_GWP20_Tens_quant = make_quants_df(fairconst_GWP20.T_ens-fair26.T_ens)
    fairBAU_GWP20_Tens_quant = make_quants_df(fairBAU_GWP20.T_ens-fair26.T_ens)
    fairconst_GWPstar_Tens_quant = make_quants_df(fairconst_GWPstar.T_ens-fair26.T_ens[0:335])
    fairBAU_GWPstar_Tens_quant = make_quants_df(fairBAU_GWPstar.T_ens-fair26.T_ens[0:335])

    # Plot 1: RF const. scenario - RCP2.6, RCP2.6+const,RCP2.6+const+policy (GWP100)
    ax1.hlines(0, 2000, 2100, linestyles="dashed")
    ax1.fill_between(
        dates,
        fairconst_GWP100_Fens_quant["2.5"],#-fair26_Fens_quant["2.5"],
        fairconst_GWP100_Fens_quant["97.5"],#-fair26_Fens_quant["97.5"],
        color=lgrey,
        alpha=0.2,
    )
    ax1.fill_between(
        dates,
        fairconst_GWP20_Fens_quant["2.5"],#-fair26_Fens_quant["2.5"],
        fairconst_GWP20_Fens_quant["97.5"],#-fair26_Fens_quant["97.5"],
        color=dgrey,
        alpha=0.2,
    )
    ax1.fill_between(
        dates_star,
        fairconst_GWPstar_Fens_quant["2.5"],#-fair26_Fens_quant["2.5"][0:335],
        fairconst_GWPstar_Fens_quant["97.5"],#-fair26_Fens_quant["97.5"][0:335],
        color=RCP60_shade,
        alpha=0.2,
    )
    ax1.plot(dates, np.sum(fairconst_GWP100.F_single, axis=1) - np.sum(fair26.F_single, axis=1), color=dgrey, label="GWP100")
    ax1.plot(
        dates, np.sum(fairconst_GWP20.F_single, axis=1)- np.sum(fair26.F_single, axis=1), color=ddgrey, label="GWP20"
    )
    ax1.plot(
        dates_star,
        np.sum(fairconst_GWPstar.F_single, axis=1)- np.sum(fair26.F_single[0:335], axis=1),
        color=RCP60,
        label="GWP*",
    )

    # Plot 2 - T const. scenario - RCP2.6, RCP2.6+const, RCP2.6+const+policy (GWP100)
    ax2.hlines(0, 2000, 2100, linestyles="dashed")
    ax2.fill_between(
        dates,
        fairconst_GWP100_Tens_quant["2.5"],#-fair26_Tens_quant["2.5"],
        fairconst_GWP100_Tens_quant["97.5"],#-fair26_Tens_quant["97.5"],
        color=lgrey,
        alpha=0.3,
    )
    ax2.fill_between(
        dates,
        fairconst_GWP20_Tens_quant["2.5"],#-fair26_Tens_quant["2.5"],
        fairconst_GWP20_Tens_quant["97.5"],#-fair26_Tens_quant["97.5"],
        color=dgrey,
        alpha=0.3,
    )
    ax2.fill_between(
        dates_star,
        fairconst_GWPstar_Tens_quant["2.5"],#-fair26_Tens_quant["2.5"][0:335],
        fairconst_GWPstar_Tens_quant["97.5"],#-fair26_Tens_quant["97.5"][0:335],
        color=RCP60_shade,
        alpha=0.3,
    )
    ax2.plot(dates, fairconst_GWP100.T_single - fair26.T_single, color=dgrey, label="GWP100")
    ax2.plot(dates, fairconst_GWP20.T_single - fair26.T_single, color=ddgrey, label="GWP20")
    ax2.plot(dates_star, fairconst_GWPstar.T_single - fair26.T_single[0:335], color=RCP60, label="GWP*")

    # Plot 3 - RF BAU. scenario - RCP2.6, RCP2.6+BAU, RCP2.6+BAU+policy (GWP100)
    ax3.hlines(0, 2000, 2100, linestyles="dashed")
    ax3.fill_between(
        dates,
        fairBAU_GWP100_Fens_quant["2.5"],#-fair26_Fens_quant["2.5"],
        fairBAU_GWP100_Fens_quant["97.5"],#-fair26_Fens_quant["97.5"],
        color=lgrey,
        alpha=0.3,
    )
    ax3.fill_between(
        dates,
        fairBAU_GWP20_Fens_quant["2.5"],#-fair26_Fens_quant["2.5"],
        fairBAU_GWP20_Fens_quant["97.5"],#-fair26_Fens_quant["97.5"],
        color=dgrey,
        alpha=0.3,
    )
    ax3.fill_between(
        dates_star,
        fairBAU_GWPstar_Fens_quant["2.5"],#-fair26_Fens_quant["2.5"][0:335],
        fairBAU_GWPstar_Fens_quant["97.5"],#-fair26_Fens_quant["97.5"][0:335],
        color=RCP85_shade,
        alpha=0.3,
    )
    ax3.plot(dates, np.sum(fairBAU_GWP100.F_single, axis=1) -np.sum(fair26.F_single, axis=1), color=dgrey, label="GWP100")
    ax3.plot(dates, np.sum(fairBAU_GWP20.F_single, axis=1) -np.sum(fair26.F_single, axis=1), color=ddgrey, label="GWP20")
    ax3.plot(
        dates_star,
        np.sum(fairBAU_GWPstar.F_single, axis=1) -np.sum(fair26.F_single[0:335], axis=1),
        color=RCP85,
        label="GWP*",
    )

    # Plot 4 - T BAU scenario - RCP2.6, RCP2.6+BAU, RCP2.6+BAU+policy (GWP100)
    ax4.hlines(0, 2000, 2100, linestyles="dashed")
    ax4.fill_between(
        dates,
        fairBAU_GWP100_Tens_quant["2.5"],#-fair26_Tens_quant["2.5"],
        fairBAU_GWP100_Tens_quant["97.5"],#-fair26_Tens_quant["97.5"],
        color=lgrey,
        alpha=0.3,
    )
    ax4.fill_between(
        dates,
        fairBAU_GWP20_Tens_quant["2.5"],#-fair26_Tens_quant["2.5"],
        fairBAU_GWP20_Tens_quant["97.5"],#-fair26_Tens_quant["97.5"],
        color=dgrey,
        alpha=0.3,
    )
    ax4.fill_between(
        dates_star,
        fairBAU_GWPstar_Tens_quant["2.5"],#-fair26_Tens_quant["2.5"][0:335],
        fairBAU_GWPstar_Tens_quant["97.5"],#-fair26_Tens_quant["97.5"][0:335],
        color=RCP85_shade,
        alpha=0.3,
    )
    ax4.plot(dates, fairBAU_GWP100.T_single - fair26.T_single, color=dgrey, label="GWP100")
    ax4.plot(dates, fairBAU_GWP20.T_single - fair26.T_single, color=ddgrey, label="GWP20")
    ax4.plot(dates_star, fairBAU_GWPstar.T_single - fair26.T_single[0:335], color=RCP85, label="GWP*")

    ax1.text(0.9, 0.9, "a", transform=ax1.transAxes, va="top", ha="left")
    ax2.text(0.9, 0.9, "b", transform=ax2.transAxes, va="top", ha="left")
    ax3.text(0.9, 0.9, "c", transform=ax3.transAxes, va="top", ha="left")
    ax4.text(0.9, 0.9, "d", transform=ax4.transAxes, va="top", ha="left")

    ax1.text(0.045, 0.95, "constant", transform=ax1.transAxes, va="top", ha="left")
    ax2.text(0.045, 0.95, "constant", transform=ax2.transAxes, va="top", ha="left")
    ax3.text(0.045, 0.95, "worst-case", transform=ax3.transAxes, va="top", ha="left")
    ax4.text(0.045, 0.95, "worst-case", transform=ax4.transAxes, va="top", ha="left")

    ax1.set_ylabel("$\Delta$ Radiative forcing (Wm$^{-2}$)")
    ax2.set_ylabel("$\Delta$ Temperature anomaly (°C)")
    ax3.set_ylabel("$\Delta$ Radiative forcing (Wm$^{-2}$)")
    ax4.set_ylabel("$\Delta$ Temperature anomaly (°C)")

    ax1.set_xlim(2000, 2100)
    ax2.set_xlim(2000, 2100)
    ax3.set_xlim(2000, 2100)
    ax4.set_xlim(2000, 2100)

    ax1.set_ylim(-0.65, 0.4)
    ax2.set_ylim(-0.25, 0.2)
    ax3.set_ylim(-0.65, 0.4)
    ax4.set_ylim(-0.25, 0.2)

    ax1.legend(loc="lower left")
    ax2.legend(loc="lower left")
    ax3.legend(loc="lower left")
    ax4.legend(loc="lower left")

    plt.rcParams["axes.labelsize"] = 11
    plt.tight_layout(pad=0.7)
    plt.savefig("Figures/RCPs_deviations.png", dpi=850, bbox_inches="tight")
