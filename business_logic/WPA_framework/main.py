import performance_analysis_library_IoV.main as perf_analysis
from tqdm import tqdm
from scipy.stats import bernoulli
from scipy import io as spio
from mpmath import *
import numpy as np
import matplotlib.pyplot as plt
import datetime
import logging
import os.path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')

# sys.path.append('./performance_analysis_library_IoV')

# sys.path.append('./performance_analysis_library_IoV')
# from performance_analysis_library_IoV import main as perf_analysis


# Constants that affect the dependent variables
fc_wifi = 5.2e9  # fc in Hz, taken from "Analysis and Optimization of
# Channel Bonding in Dense IEEE 802.11 WLANs"
fc_LTE = 2.5e9  # fc in Hz
fc_5GNR_1 = 3.5e9  # fc in Hz
fc_5GNR_2 = 700e6  # fc in Hz
c_wired = 10 ** (
    -8)  # propagation delay per bit per unit length, taken from "Minimizing
# frequencies from 0.5 to 100 GHz (3GPP TR 38.901 version 16.1.0 Release 16)"
d_b = 200  # Length of the b-th wired backhaul link
m_omega = d_b  # Backhaul delay Gamma shaping parameter
T = 0.5e-3  # Feedback period in s, i.e., feedback latency
Pt_dBm = 23  # Transmit power in dBm, tested 23 dBm
Pt = 10 ** (Pt_dBm / 10) * 1e-3  # Transmit power in Watts


# Constants end


# These variables are meant to be localised within each function
# K_size = 10e3 * 8  # Task size in bits, tested 10K Bytes
# BW = 10e6  # BW in Hz, tested 20KHz taken from "Study on channel model for
# v = 20 / 3.6  # Vehicle speed in m/s, input in Km/h. Tested 10km/h, 20 km/h
# N = 3  # Number of hops between cloud server and AMR
# ra = 200  # Communications radius in m
# numSamples = 10 ** 4
# Substitute variables end


# # These are the dependent variables
# eta = 20 ** ((-174 + 10 * log10(BW)) / 10) * 1e-3  # noise power
# Pt_norm = Pt / (eta)  # Normalised transmit power

# R_wifi = 2.32e6 / BW  # 802.11 for MCS 0, taken from "Physical Layer
# thres_tx_wifi = 2 ** R_wifi - 1  # 802.11ac SINR threshold


# # Evaluation of V2X Communications Technologies: 5G NR-V2X, LTE-V2X,
# # IEEE 802.11bd, and IEEE 802.11p"
# R_LTE = 4.08e6 / BW  # LTE, for MCS 6, taken from "Physical Layer Evaluation
# # of V2X Communications Technologies: 5G NR-V2X, LTE-V2X, IEEE 802.11bd,
# # and IEEE 802.11p"
# thres_tx_LTE = 2 ** R_LTE - 1  # LTE SINR threshold
# R_5GNR = 4.63e6 / BW  # 5G NR, for MCS 6, taken from "Physical Layer
# # Evaluation of V2X Communications Technologies: 5G NR-V2X, LTE-V2X,
# # IEEE 802.11bd, and IEEE 802.11p"
# thres_tx_5GNR = 2 ** R_5GNR - 1  # 5G NR SINR threshold
# fd_wifi = v * fc_wifi / (3 * 10 ** 8)
# epsilon_wifi = j0(2 * pi * fd_wifi * T)

# fd_LTE = v * fc_LTE / (3 * 10 ** 8)
# epsilon_LTE = j0(2 * pi * fd_LTE * T)

# fd_5GNR_1 = v * fc_5GNR_1 / (3 * 10 ** 8)
# epsilon_5GNR_1 = j0(2 * pi * fd_5GNR_1 * T)
# fd_5GNR_2 = v * fc_5GNR_2 / (3 * 10 ** 8)
# epsilon_5GNR_2 = j0(2 * pi * fd_5GNR_2 * T)
# theta_omega = c_wired * K_size  # Backhaul delay Gamma scale parameter
# tau_timeout_wired = m_omega * theta_omega  # Wired backhaul timeout


# # Dependent variables end

def process_raw_input(input_variables: Dict[str, List[float]]):
    """Convert the raw inputs to processed input according to business logic"""
    K_size: int = input_variables['task_size_kb'][
        0] * 8e3  # KB to bits conversion
    BW: int = input_variables['bandwidth_MHz'][0] * 1e6  # MHz to Hertz
    v: float = input_variables['speed_mps'][
        0] / 3.6  # kmph to metre per second
    numSamples: float = input_variables['number_of_samples_thousands'][
        0] * 1e3  # Frontend captures number fo
    # samples/1000
    return K_size, BW, v, input_variables['hops'][0], \
        input_variables['communication_rad_m'][0], int(numSamples)


M_SBS = 2  # Number of SBSs, tested 2
M_MBS = 1  # Number of BSs, tested 1
M_int = 2  # Number of interferers, tested 1, 2

l_UMa = 2.2 + 1e-6  # Pathloss exponent, taken from "Study on channel model
# for frequencies from 0.5 to 100 GHz (3GPP TR 38.901 version 16.1.0 Release
# 16)"
l_InF_wifi = 3.19 + 1e-6  # Pathloss exponent, taken from "The Industrial
# Indoor Channel: Large-Scale and Temporal Fading at 900, 2400, and 5200 MHz"
l_InF_LTE = 3.19 + 1e-6  # Pathloss exponent, taken from "The Industrial
# Indoor Channel: Large-Scale and Temporal Fading at 900, 2400, and 5200 MHz"
l_InF_5GNR1 = 3.19 + 1e-6  # Pathloss exponent, taken from "The Industrial
# Indoor Channel: Large-Scale and Temporal Fading at 900, 2400, and 5200 MHz"
l_InF_5GNR2 = 3.19 + 1e-6  # Pathloss exponent, taken from "The Industrial
# Indoor Channel: Large-Scale and Temporal Fading at 900, 2400, and 5200 MHz"
l_RMa = 2 + 1e-6  # Pathloss exponent, taken from "Study on channel model
# for frequencies from 0.5 to 100 GHz (3GPP TR 38.901 version 16.1.0 Release
# 16)"


K_i = 10 ** (
    5 / 10)  # Rician K factor of the i-th SBS, tested 0, 5, taken from
# "On Long-Term Statistical Dependences in Channel Gains for Fixed Wireless
# Links in Factories"
K_j = K_i  # Rician K factor of the j-th MBS
K_z = K_i  # Rician K factor of the z-th IAMR

# Latency to Support VR Social Interactions Over Wireless Cellular Systems
# via Bandwidth Allocation"


# threshold in seconds, tested 0.01
dclutter = 15  # tested 10, 15
LOS_r = 0.1  # tested 0.1, 0.15
L_LOS_UMa_wifi = 10 ** ((28 + 20 * log10(
    fc_wifi / 1e9)) / 10)  # taken from "Study on channel model for
# frequencies from 0.5 to 100 GHz (3GPP TR 38.901 version 16.1.0 Release 16)"
L_LOS_UMa_LTE = 10 ** ((28 + 20 * log10(
    fc_LTE / 1e9)) / 10)  # taken from "Study on channel model for
# frequencies from 0.5 to 100 GHz (3GPP TR 38.901 version 16.1.0 Release 16)"
L_LOS_UMa_5GNR_1 = 10 ** ((28 + 20 * log10(
    fc_5GNR_1 / 1e9)) / 10)  # taken from "Study on channel model for
# frequencies from 0.5 to 100 GHz (3GPP TR 38.901 version 16.1.0 Release 16)"
L_LOS_UMa_5GNR_2 = 10 ** ((28 + 20 * log10(
    fc_5GNR_2 / 1e9)) / 10)  # taken from "Study on channel model for
# frequencies from 0.5 to 100 GHz (3GPP TR 38.901 version 16.1.0 Release 16)"

L_LOS_InFSL_wifi = 10 ** ((31.84 + 20 * log10(
    fc_wifi / 1e9)) / 10)  # taken from "Study on channel model for
# frequencies from 0.5 to 100 GHz (3GPP TR 38.901 version 16.1.0 Release 16)"
L_LOS_InFSL_LTE = 10 ** ((31.84 + 20 * log10(
    fc_LTE / 1e9)) / 10)  # taken from "Study on channel model for
# frequencies from 0.5 to 100 GHz (3GPP TR 38.901 version 16.1.0 Release 16)"
L_LOS_InFSL_5GNR_1 = 10 ** ((31.84 + 20 * log10(
    fc_5GNR_1 / 1e9)) / 10)  # taken from "Study on channel model for
# frequencies from 0.5 to 100 GHz (3GPP TR 38.901 version 16.1.0 Release 16)"
L_LOS_InFSL_5GNR_2 = 10 ** ((31.84 + 20 * log10(
    fc_5GNR_2 / 1e9)) / 10)  # taken from "Study on channel model for
# frequencies from 0.5 to 100 GHz (3GPP TR 38.901 version 16.1.0 Release 16)"
d0 = 1  # InF reference distance, based on "Study on channel model for
# frequencies from 0.5 to 100 GHz (3GPP TR 38.901 version 16.1.0 Release 16)"

L_LOS_RMa_wifi = 10 ** ((20 * log10(40 * pi / 3 * fc_wifi / 1e9)) / 10)
L_LOS_RMa_LTE = 10 ** ((20 * log10(40 * pi / 3 * fc_LTE / 1e9)) / 10)
L_LOS_RMa_5GNR_1 = 10 ** ((20 * log10(40 * pi / 3 * fc_5GNR_1 / 1e9)) / 10)
L_LOS_RMa_5GNR_2 = 10 ** ((20 * log10(40 * pi / 3 * fc_5GNR_2 / 1e9)) / 10)

tr_UMa = 6  # tested 6
tr_InF = 5  # tested 5
tr_RMa = 5

# Date and time format
dateFormat = '%Y-%m-%d'
timeFormat = '%H:%M:%S'


##
# This function simulates the CDF of gamma_x
def simulate_CDF_gamma_x(
        K_x, L_LOS, thres_tx, p_LOS, epsilon, rint, Pt_norm,
        K_z, M_int, l, ra, d0, numSamples):
    # Simulate gamma_x
    d_x = perf_analysis.euclidean_dist(ra, 0, numSamples)
    d_z = [perf_analysis.euclidean_dist_annulus(min_rad=ra, ra=rint, H=0,
                                                numSamples=M_int) for _ in
           range(numSamples)] if M_int > 0 else []
    h_hat_x = perf_analysis.ricianFading_RV(K_factor=K_x,
                                            numSamples=numSamples)
    e_x = perf_analysis.nakFading_RV(1, numSamples)
    h_z = np.transpose(
        [perf_analysis.ricianFading_RV(K_factor=K_z, numSamples=numSamples) for
         _ in range(M_int)]) if M_int > 0 else []
    beta_x = [1 / L_LOS if bernoulli.rvs(float(p_LOS), size=1) else 0 for _ in
              range(numSamples)]

    gamma_x = [
        beta_x[i] * Pt_norm * epsilon ** 2 * h_hat_x[i] * (d_x[i] / d0) ** (
            -l)
        / (1 + Pt_norm / L_LOS * (1 - epsilon ** 2) * e_x[i] * (
            d_x[i] / d0) ** (-l) + sum(
            Pt_norm / L_LOS * h_z[i] * (d_z[i] / d0) ** (-l)))
        for i in range(numSamples)] if M_int > 0 \
        else [
        beta_x[i] * Pt_norm * epsilon ** 2 * h_hat_x[i] * (d_x[i] / d0) ** (
            -l) / (1 + Pt_norm / L_LOS * (1 - epsilon ** 2) * e_x[i] * (
                d_x[i] / d0) ** (-l))
        for i in range(numSamples)]

    simulated_CDF_gamma_x = sum(
        [1 for idx in range(len(gamma_x)) if gamma_x[idx] < thres_tx]) / len(
        gamma_x)

    if simulated_CDF_gamma_x < 1e2 / (numSamples):
        simulated_CDF_gamma_x = 0

    return simulated_CDF_gamma_x


##
# This function simulates the throughput for cloud-only offloading (IEEE
# 802.11ac or LTE)
def simulate_throughput_CL(
        K_i, K_j, K_z, L_LOS, thres_tx, p_LOS, epsilon, rint, R_wifi,
        R_LTE, Pt_norm, M_int, ra, M_MBS,
        M_SBS, N,
        tau_timeout_wired, m_omega,
        theta_omega, numSamples, lte, l,
        d0, BW):
    R = R_LTE if lte else R_wifi

    # Simulate CDF_i, CDF_j, CCDF_b
    CDF_i_sim = simulate_CDF_gamma_x(K_x=K_i, L_LOS=L_LOS, thres_tx=thres_tx,
                                     p_LOS=p_LOS, epsilon=epsilon, ra=ra,
                                     rint=rint, K_z=K_z, M_int=M_int,
                                     Pt_norm=Pt_norm, l=l, d0=d0,
                                     numSamples=numSamples)
    CDF_j_sim = simulate_CDF_gamma_x(K_x=K_j, L_LOS=L_LOS, thres_tx=thres_tx,
                                     p_LOS=p_LOS, epsilon=epsilon, ra=ra,
                                     rint=rint, K_z=K_z, M_int=M_int,
                                     Pt_norm=Pt_norm, l=l, d0=d0,
                                     numSamples=numSamples)
    CCDF_b_sim = simulate_CCDF_T_wired_b(tau_timeout_wired=tau_timeout_wired,
                                         m_omega=m_omega,
                                         theta_omega=theta_omega,
                                         numSamples=numSamples)

    # Compute throughput for cloud-only offloading
    simulated_throughput_CL = ((1 - CDF_i_sim ** M_SBS) + (
        CDF_i_sim ** M_SBS) * (1 - CDF_j_sim ** M_MBS)) * (
        1 - CCDF_b_sim) ** (N - 1) * (R * BW)

    return simulated_throughput_CL


##
# This function simulates the throughput for computation offloading (5GNR
# SBS with LTE MBS)
def simulate_throughput_CO(
        K_i, K_j, K_z, L_LOS, p_LOS, epsilon, rint,
        thres_tx_5GNR,
        thres_tx_LTE, R_5GNR, R_LTE, Pt_norm,
        l, M_int, ra, M_MBS, M_SBS, N,
        tau_timeout_wired, m_omega,
        theta_omega, numSamples, d0, BW):
    # Simulate CDF_i, CDF_j, CCDF_b
    CDF_i_sim = simulate_CDF_gamma_x(K_x=K_i, L_LOS=L_LOS,
                                     thres_tx=thres_tx_5GNR, p_LOS=p_LOS,
                                     epsilon=epsilon, ra=ra, rint=rint,
                                     K_z=K_z,
                                     M_int=M_int, Pt_norm=Pt_norm, l=l,
                                     d0=d0, numSamples=numSamples)
    CDF_j_sim = simulate_CDF_gamma_x(K_x=K_j, L_LOS=L_LOS,
                                     thres_tx=thres_tx_LTE, p_LOS=p_LOS,
                                     epsilon=epsilon, ra=ra, rint=rint,
                                     K_z=K_z,
                                     M_int=M_int, Pt_norm=Pt_norm, l=l,
                                     d0=d0, numSamples=numSamples)
    CCDF_b_sim = simulate_CCDF_T_wired_b(tau_timeout_wired=tau_timeout_wired,
                                         m_omega=m_omega,
                                         theta_omega=theta_omega,
                                         numSamples=numSamples)

    # Compute throughput for cloud-only offloading
    simulated_throughput_CL = (1 - CDF_i_sim ** M_SBS) * (R_5GNR * BW) \
        + ((CDF_i_sim ** M_SBS) * (
            1 - CDF_j_sim ** M_MBS) * (1 - CCDF_b_sim) ** (N - 1) * (
            R_LTE * BW))

    return simulated_throughput_CL


##
# This function simulates the complementary CDF of T_wired_b
def simulate_CCDF_T_wired_b(
        tau_timeout_wired, m_omega,
        theta_omega, numSamples):
    # Simulate T_wired_b
    T_wired_b = np.random.gamma(m_omega, theta_omega, numSamples)

    simulated_CDF_T_wired_b = sum([1 for idx in range(len(T_wired_b)) if
                                   T_wired_b[idx] < tau_timeout_wired]) / len(
        T_wired_b)

    if simulated_CDF_T_wired_b < 1e2 / (numSamples):
        simulated_CDF_T_wired_b = 0

    return 1 - simulated_CDF_T_wired_b


##
# This function computes the CDF of gamma_x
def compute_CDF_gamma_x(
        K_x, L_LOS, thres_tx, p_LOS, epsilon, tr, rint, Pt_norm,
        l,
        K_z, M_int, ra, d0=d0):
    # Initialise variables
    multinom_idx = perf_analysis.load_multinom_idx(M_int + 1, n=1, tr=tr)

    def alpha_x(p): return perf_analysis.alpha_Rician_func(
        Pt_norm / (L_LOS) * epsilon ** 2, K_x, sum(multinom_idx[int(p)]) - 1,
        thres_tx) \
        * factorial(sum(multinom_idx[int(p)])) / (nprod(
            lambda i: factorial(multinom_idx[int(p)][int(i)]),
            [0, np.shape(multinom_idx)[1] - 1])) \
        * 2 * (ra / d0) ** (l * (
            sum(multinom_idx[int(p)]) - int(
                multinom_idx[int(p)][int(M_int)]))) / (l * (
                    sum(multinom_idx[int(p)]) - int(
                        multinom_idx[int(p)][int(M_int)])) + 2) \
        * perf_analysis.fractionalMoments_nakFading(1,
                                                    Pt_norm
                                                    / L_LOS * (
                                                        1 - epsilon ** 2),
                                                    int(
                                                        multinom_idx[
                                                            int(p)][
                                                            int(M_int)])) \
        * nprod(lambda z:
                perf_analysis.fractionalMoments_ricianFading(
                    Pt_norm / L_LOS,
                    int(multinom_idx[int(p)][int(z)]), K_z)
                * 2 * ((rint) ** (2 - l * int(
                    multinom_idx[int(p)][int(z)])) - (
                    ra / d0) ** (2 - l * int(
                        multinom_idx[int(p)][int(z)]))) / (
                    (rint ** 2 - ra ** 2) * (
                        2 - l * int(
                            multinom_idx[int(p)][int(z)]))), [0, M_int - 1])

    # Compute CDF of gamma_x
    computed_gamma_x_CDF = (1 - p_LOS) + p_LOS * nsum(alpha_x, [0,
                                                                len(multinom_idx) - 1])

    return computed_gamma_x_CDF


##
# This function computes the throughput for cloud-only offloading (IEEE
# 802.11ac or LTE)
def compute_throughput_CL(
        K_i, K_j, K_z, L_LOS, thres_tx, p_LOS, epsilon, tr, rint,
        R_wifi,
        R_LTE, Pt_norm, l, M_int, ra, M_MBS,
        M_SBS, N,
        tau_timeout_wired, m_omega,
        theta_omega, lte, d0, BW
):
    R = R_LTE if lte else R_wifi

    # Compute CDF_i, CDF_j, CCDF_b
    CDF_i = compute_CDF_gamma_x(K_x=K_i, L_LOS=L_LOS, thres_tx=thres_tx,
                                p_LOS=p_LOS, epsilon=epsilon, tr=tr, ra=ra,
                                rint=rint, K_z=K_z, M_int=M_int,
                                Pt_norm=Pt_norm, l=l, d0=d0)
    CDF_j = compute_CDF_gamma_x(K_x=K_j, L_LOS=L_LOS, thres_tx=thres_tx,
                                p_LOS=p_LOS, epsilon=epsilon, tr=tr, ra=ra,
                                rint=rint, K_z=K_z, M_int=M_int,
                                Pt_norm=Pt_norm, l=l, d0=d0)
    CCDF_b = compute_CCDF_T_wired_b(tau_timeout_wired=tau_timeout_wired,
                                    m_omega=m_omega, theta_omega=theta_omega)

    # Compute throughput for cloud-only offloading
    computed_throughput_CL = ((1 - CDF_i ** M_SBS) + (CDF_i ** M_SBS) * (
        1 - CDF_j ** M_MBS)) * (R * BW) * (1 - CCDF_b) ** (N - 1)

    return computed_throughput_CL


##
# This function computed the throughput for computation offloading (5GNR SBS
# with LTE MBS)
def compute_throughput_CO(
        K_i, K_j, K_z, L_LOS, p_LOS, epsilon, tr, rint,
        thres_tx_5GNR, thres_tx_LTE, R_5GNR,
        R_LTE, Pt_norm, l, M_int, ra, M_MBS,
        M_SBS, N,
        tau_timeout_wired, m_omega,
        theta_omega, d0, BW):
    # Compute CDF_i, CDF_j, CCDF_b
    CDF_i = compute_CDF_gamma_x(K_x=K_i, L_LOS=L_LOS, thres_tx=thres_tx_5GNR,
                                p_LOS=p_LOS, epsilon=epsilon, tr=tr, ra=ra,
                                rint=rint, K_z=K_z, M_int=M_int,
                                Pt_norm=Pt_norm, l=l, d0=d0)
    CDF_j = compute_CDF_gamma_x(K_x=K_j, L_LOS=L_LOS, thres_tx=thres_tx_LTE,
                                p_LOS=p_LOS, epsilon=epsilon, tr=tr, ra=ra,
                                rint=rint, K_z=K_z, M_int=M_int,
                                Pt_norm=Pt_norm, l=l, d0=d0)
    CCDF_b = compute_CCDF_T_wired_b(tau_timeout_wired=tau_timeout_wired,
                                    m_omega=m_omega, theta_omega=theta_omega)

    # Compute throughput for cloud-only offloading
    computed_throughput_CO = (1 - CDF_i ** M_SBS) * (R_5GNR * BW) \
        + ((CDF_i ** M_SBS) * (1 - CDF_j ** M_MBS) * (
            1 - CCDF_b) ** (N - 1) * (R_LTE * BW))

    return computed_throughput_CO


##
# This function computes the complementary CDF of T_wired_b
def compute_CCDF_T_wired_b(
        tau_timeout_wired, m_omega,
        theta_omega
):
    # Compute CDF of T_wired_b
    computed_CCDF_T_wired_b = (1 - gammainc(m_omega, 0,
                                            tau_timeout_wired / theta_omega)
                               / gamma(
        m_omega))

    return computed_CCDF_T_wired_b


##
# This function generates results for gamma_x_CDF vs ra


##
# This function generates results for throughput vs ra in the UMa scenario
def throughput_vs_ra_UMa(file_name: str,
                         input_variables: Dict[str, List]) -> None:
    K_size, BW, v, N, _, numSamples = process_raw_input(input_variables)
    fd_5GNR_2 = v * fc_5GNR_2 / (3 * 10 ** 8)
    epsilon_5GNR_2 = j0(2 * pi * fd_5GNR_2 * T)
    fd_5GNR_1 = v * fc_5GNR_1 / (3 * 10 ** 8)
    epsilon_5GNR_1 = j0(2 * pi * fd_5GNR_1 * T)
    R_wifi = 2.32e6 / BW  # 802.11 for MCS 0, taken from "Physical Layer
    thres_tx_wifi = 2 ** R_wifi - 1  # 802.11ac SINR threshold
    fd_wifi = v * fc_wifi / (3 * 10 ** 8)
    epsilon_wifi = j0(2 * pi * fd_wifi * T)
    fd_LTE = v * fc_LTE / (3 * 10 ** 8)
    epsilon_LTE = j0(2 * pi * fd_LTE * T)

    R_LTE = 4.08e6 / BW  # LTE, for MCS 6, taken from "Physical Layer
    # Evaluation
    # of V2X Communications Technologies: 5G NR-V2X, LTE-V2X, IEEE 802.11bd,
    # and IEEE 802.11p"
    thres_tx_LTE = 2 ** R_LTE - 1  # LTE SINR threshold
    eta = 20 ** ((-174 + 10 * log10(BW)) / 10) * 1e-3  # noise power
    Pt_norm = Pt / (eta)  # Normalised transmit power
    theta_omega = c_wired * K_size  # Backhaul delay Gamma scale parameter
    tau_timeout_wired = m_omega * theta_omega  # Wired backhaul timeout

    R_5GNR = 4.63e6 / BW  # 5G NR, for MCS 6, taken from "Physical Layer
    # Evaluation of V2X Communications Technologies: 5G NR-V2X, LTE-V2X,
    # IEEE 802.11bd, and IEEE 802.11p"
    thres_tx_5GNR = 2 ** R_5GNR - 1  # 5G NR SINR threshold

    desc = datetime.datetime.now().strftime(
        dateFormat + "|" + timeFormat) + (": Simulating and computing "
                                          "throughput vs ra in the UMa "
                                          "scenario")

    # Initialise variables
    ra_min = 20
    ra_max = 400  # tested 400
    ra_step = 20
    ra = np.arange(start=ra_min, stop=ra_max + ra_step, step=ra_step)
    rint = 2 * ra  # tested 2
    p_LOS_UMa = [18 / ra[i] + (1 - 18 / ra[i]) * exp(-(ra[i]) / 63) for i in
                 range(len(ra))]

    computed_throughput = np.zeros(shape=(4, len(ra)))
    simulated_throughput = np.zeros(shape=(4, len(ra)))

    for k in tqdm(range(len(ra)), desc=desc):
        # Simulate throughput
        simulated_throughput[0][k] = simulate_throughput_CL(K_i, K_j, K_z,
                                                            L_LOS=L_LOS_UMa_wifi,
                                                            thres_tx=thres_tx_wifi,
                                                            p_LOS=p_LOS_UMa[k],
                                                            epsilon=epsilon_wifi,
                                                            M_int=M_int + 1,
                                                            ra=ra[k],
                                                            rint=rint[k],
                                                            lte=False,
                                                            l=l_UMa,
                                                            d0=1,
                                                            BW=BW,
                                                            M_MBS=M_MBS,
                                                            M_SBS=M_SBS,
                                                            N=N,
                                                            R_LTE=R_LTE,
                                                            R_wifi=R_wifi,
                                                            m_omega=m_omega,
                                                            numSamples=numSamples,
                                                            tau_timeout_wired=tau_timeout_wired,
                                                            theta_omega=theta_omega,
                                                            Pt_norm=Pt_norm)
        simulated_throughput[1][k] = simulate_throughput_CL(K_i, K_j, K_z,
                                                            L_LOS=L_LOS_UMa_LTE,
                                                            thres_tx=thres_tx_LTE,
                                                            p_LOS=p_LOS_UMa[k],
                                                            epsilon=epsilon_LTE,
                                                            M_int=M_int,
                                                            ra=ra[k],
                                                            rint=rint[k],
                                                            lte=True, l=l_UMa,
                                                            d0=1,
                                                            BW=BW,
                                                            M_MBS=M_MBS,
                                                            M_SBS=M_SBS,
                                                            N=N,
                                                            R_LTE=R_LTE,
                                                            R_wifi=R_wifi,
                                                            m_omega=m_omega,
                                                            numSamples=numSamples,
                                                            tau_timeout_wired=tau_timeout_wired,
                                                            theta_omega=theta_omega,
                                                            Pt_norm=Pt_norm)
        simulated_throughput[2][k] = simulate_throughput_CO(K_i, K_j, K_z,
                                                            L_LOS=L_LOS_UMa_5GNR_1,
                                                            p_LOS=p_LOS_UMa[k],
                                                            epsilon=epsilon_5GNR_1,
                                                            M_int=M_int,
                                                            ra=ra[k],
                                                            rint=rint[k],
                                                            l=l_UMa, d0=1,
                                                            BW=BW,
                                                            M_MBS=M_MBS,
                                                            M_SBS=M_SBS,
                                                            N=N,
                                                            R_LTE=R_LTE,
                                                            m_omega=m_omega,
                                                            numSamples=numSamples,
                                                            tau_timeout_wired=tau_timeout_wired,
                                                            theta_omega=theta_omega,
                                                            Pt_norm=Pt_norm,
                                                            R_5GNR=R_5GNR,
                                                            thres_tx_LTE=thres_tx_LTE,
                                                            thres_tx_5GNR=thres_tx_5GNR)
        simulated_throughput[3][k] = simulate_throughput_CO(K_i, K_j, K_z,
                                                            L_LOS=L_LOS_UMa_5GNR_2,
                                                            p_LOS=p_LOS_UMa[k],
                                                            epsilon=epsilon_5GNR_2,
                                                            M_int=M_int,
                                                            ra=ra[k],
                                                            rint=rint[k],
                                                            l=l_UMa, d0=1,
                                                            BW=BW,
                                                            M_MBS=M_MBS,
                                                            M_SBS=M_SBS,
                                                            N=N,
                                                            R_LTE=R_LTE,
                                                            m_omega=m_omega,
                                                            numSamples=numSamples,
                                                            tau_timeout_wired=tau_timeout_wired,
                                                            theta_omega=theta_omega,
                                                            Pt_norm=Pt_norm,
                                                            R_5GNR=R_5GNR,
                                                            thres_tx_LTE=thres_tx_LTE,
                                                            thres_tx_5GNR=thres_tx_5GNR)

        # Compute throughput
        computed_throughput[0][k] = compute_throughput_CL(K_i, K_j, K_z,
                                                          L_LOS=L_LOS_UMa_wifi,
                                                          thres_tx=thres_tx_wifi,
                                                          p_LOS=p_LOS_UMa[k],
                                                          epsilon=epsilon_wifi,
                                                          M_int=M_int + 1,
                                                          ra=ra[k],
                                                          rint=rint[k],
                                                          lte=False, l=l_UMa,
                                                          tr=tr_UMa, d0=1,
                                                          BW=BW,
                                                          M_MBS=M_MBS,
                                                          M_SBS=M_SBS,
                                                          N=N,
                                                          R_LTE=R_LTE,
                                                          m_omega=m_omega,
                                                          tau_timeout_wired=tau_timeout_wired,
                                                          theta_omega=theta_omega,
                                                          Pt_norm=Pt_norm,
                                                          R_wifi=R_wifi)
        computed_throughput[1][k] = compute_throughput_CL(K_i, K_j, K_z,
                                                          L_LOS=L_LOS_UMa_LTE,
                                                          thres_tx=thres_tx_LTE,
                                                          p_LOS=p_LOS_UMa[k],
                                                          epsilon=epsilon_LTE,
                                                          M_int=M_int,
                                                          ra=ra[k],
                                                          rint=rint[k],
                                                          lte=True, l=l_UMa,
                                                          tr=tr_UMa, d0=1,
                                                          BW=BW,
                                                          M_MBS=M_MBS,
                                                          M_SBS=M_SBS,
                                                          N=N,
                                                          R_LTE=R_LTE,
                                                          m_omega=m_omega,
                                                          tau_timeout_wired=tau_timeout_wired,
                                                          theta_omega=theta_omega,
                                                          Pt_norm=Pt_norm,
                                                          R_wifi=R_wifi)
        computed_throughput[2][k] = compute_throughput_CO(K_i, K_j, K_z,
                                                          L_LOS=L_LOS_UMa_5GNR_1,
                                                          p_LOS=p_LOS_UMa[k],
                                                          epsilon=epsilon_5GNR_1,
                                                          M_int=M_int,
                                                          ra=ra[k],
                                                          rint=rint[k],
                                                          l=l_UMa,
                                                          tr=tr_UMa, d0=1,
                                                          BW=BW,
                                                          M_MBS=M_MBS,
                                                          M_SBS=M_SBS,
                                                          N=N,
                                                          R_LTE=R_LTE,
                                                          m_omega=m_omega,
                                                          tau_timeout_wired=tau_timeout_wired,
                                                          theta_omega=theta_omega,
                                                          Pt_norm=Pt_norm,
                                                          R_5GNR=R_5GNR,
                                                          thres_tx_5GNR=thres_tx_5GNR,
                                                          thres_tx_LTE=thres_tx_LTE
                                                          )
        computed_throughput[3][k] = compute_throughput_CO(K_i, K_j, K_z,
                                                          L_LOS=L_LOS_UMa_5GNR_2,
                                                          p_LOS=p_LOS_UMa[k],
                                                          epsilon=epsilon_5GNR_2,
                                                          M_int=M_int,
                                                          ra=ra[k],
                                                          rint=rint[k],
                                                          l=l_UMa,
                                                          tr=tr_UMa, d0=1,
                                                          BW=BW,
                                                          M_MBS=M_MBS,
                                                          M_SBS=M_SBS,
                                                          N=N,
                                                          R_LTE=R_LTE,
                                                          m_omega=m_omega,
                                                          tau_timeout_wired=tau_timeout_wired,
                                                          theta_omega=theta_omega,
                                                          Pt_norm=Pt_norm,
                                                          R_5GNR=R_5GNR,
                                                          thres_tx_5GNR=thres_tx_5GNR,
                                                          thres_tx_LTE=thres_tx_LTE)

        print(datetime.datetime.now().strftime(
            dateFormat + "|" + timeFormat) + ": "
            + "\n\t ra (m): {:n} ".format(ra[k])
              + "\n\t Throughput (Mbps) - 802.11ac (Computed): \t{"
                ":.4n}".format(
            computed_throughput[0][k] / 1e6)
            + "\n\t Throughput (Mbps) - 802.11ac (Simulated): \t{"
            ":.4n}".format(
            simulated_throughput[0][k] / 1e6)
            + "\n\t Throughput (Mbps) - LTE (Computed): \t\t{:.4n}".format(
            computed_throughput[1][k] / 1e6)
            + "\n\t Throughput (Mbps) - LTE (Simulated): \t\t{:.4n}".format(
            simulated_throughput[1][k] / 1e6)
            + "\n\t Throughput (Mbps) - 5GNR-1 (Computed): \t{:.4n}".format(
            computed_throughput[2][k] / 1e6)
            + "\n\t Throughput (Mbps) - 5GNR-1 (Simulated): \t{:.4n}".format(
            simulated_throughput[2][k] / 1e6)
            + "\n\t Throughput (Mbps) - 5GNR-2 (Computed): \t{:.4n}".format(
            computed_throughput[3][k] / 1e6)
            + "\n\t Throughput (Mbps) - 5GNR-2 (Simulated): \t{:.4n}".format(
            simulated_throughput[3][k] / 1e6)
        )
    os.makedirs(name=os.path.dirname(p=file_name), exist_ok=True)
    spio.savemat(file_name,
                 {
                     'simulated_throughput': simulated_throughput,
                     'computed_throughput': computed_throughput,
                     'ra': ra
                 })


##
# This function generates results for throughput vs ra in the InF-SL scenario
def throughput_vs_ra_InFSL(file_name: str,
                           input_variables: Dict[str, List]) -> None:
    K_size, BW, v, N, _, numSamples = process_raw_input(input_variables)
    fd_5GNR_2 = v * fc_5GNR_2 / (3 * 10 ** 8)
    epsilon_5GNR_2 = j0(2 * pi * fd_5GNR_2 * T)
    fd_5GNR_1 = v * fc_5GNR_1 / (3 * 10 ** 8)
    epsilon_5GNR_1 = j0(2 * pi * fd_5GNR_1 * T)
    R_wifi = 2.32e6 / BW  # 802.11 for MCS 0, taken from "Physical Layer
    thres_tx_wifi = 2 ** R_wifi - 1  # 802.11ac SINR threshold
    fd_wifi = v * fc_wifi / (3 * 10 ** 8)
    epsilon_wifi = j0(2 * pi * fd_wifi * T)
    fd_LTE = v * fc_LTE / (3 * 10 ** 8)
    epsilon_LTE = j0(2 * pi * fd_LTE * T)

    R_LTE = 4.08e6 / BW  # LTE, for MCS 6, taken from "Physical Layer
    # Evaluation
    # of V2X Communications Technologies: 5G NR-V2X, LTE-V2X, IEEE 802.11bd,
    # and IEEE 802.11p"
    thres_tx_LTE = 2 ** R_LTE - 1  # LTE SINR threshold
    eta = 20 ** ((-174 + 10 * log10(BW)) / 10) * 1e-3  # noise power
    Pt_norm = Pt / (eta)  # Normalised transmit power
    theta_omega = c_wired * K_size  # Backhaul delay Gamma scale parameter
    tau_timeout_wired = m_omega * theta_omega  # Wired backhaul timeout

    R_5GNR = 4.63e6 / BW  # 5G NR, for MCS 6, taken from "Physical Layer
    # Evaluation of V2X Communications Technologies: 5G NR-V2X, LTE-V2X,
    # IEEE 802.11bd, and IEEE 802.11p"
    thres_tx_5GNR = 2 ** R_5GNR - 1  # 5G NR SINR threshold

    desc = datetime.datetime.now().strftime(
        dateFormat + "|" + timeFormat) + (": Simulating and computing "
                                          "throughput vs ra in the InF-SL "
                                          "scenario")

    # Initialise variables
    ra_min = 20
    ra_max = 200  # tested 200
    ra_step = 20
    ra = np.arange(start=ra_min, stop=ra_max + ra_step, step=ra_step)
    rint = 2 * ra  # tested 2
    p_LOS_InFSL = [exp(-(ra[i]) / (-dclutter / ln(1 - LOS_r))) for i in
                   range(len(ra))]

    computed_throughput = np.zeros(shape=(4, len(ra)))
    simulated_throughput = np.zeros(shape=(4, len(ra)))

    for k in tqdm(range(len(ra)), desc=desc):
        # Simulate throughput
        simulated_throughput[0][k] = simulate_throughput_CL(K_i, K_j, K_z,
                                                            L_LOS=L_LOS_InFSL_wifi,
                                                            thres_tx=thres_tx_wifi,
                                                            p_LOS=p_LOS_InFSL[
                                                                k],
                                                            epsilon=epsilon_wifi,
                                                            rint=rint[k],
                                                            M_int=M_int + 1,
                                                            ra=ra[k],
                                                            lte=False,
                                                            l=l_InF_wifi,
                                                            BW=BW,
                                                            M_MBS=M_MBS,
                                                            M_SBS=M_SBS,
                                                            N=N,
                                                            R_LTE=R_LTE,
                                                            R_wifi=R_wifi,
                                                            m_omega=m_omega,
                                                            numSamples=numSamples,
                                                            tau_timeout_wired=tau_timeout_wired,
                                                            theta_omega=theta_omega,
                                                            d0=d0,
                                                            Pt_norm=Pt_norm)
        simulated_throughput[1][k] = simulate_throughput_CL(K_i, K_j, K_z,
                                                            L_LOS=L_LOS_InFSL_LTE,
                                                            thres_tx=thres_tx_LTE,
                                                            p_LOS=p_LOS_InFSL[
                                                                k],
                                                            epsilon=epsilon_LTE,
                                                            rint=rint[k],
                                                            M_int=M_int,
                                                            ra=ra[k], lte=True,
                                                            l=l_InF_LTE,
                                                            BW=BW,
                                                            M_MBS=M_MBS,
                                                            M_SBS=M_SBS,
                                                            N=N,
                                                            R_LTE=R_LTE,
                                                            R_wifi=R_wifi,
                                                            m_omega=m_omega,
                                                            numSamples=numSamples,
                                                            tau_timeout_wired=tau_timeout_wired,
                                                            theta_omega=theta_omega,
                                                            d0=d0,
                                                            Pt_norm=Pt_norm)
        simulated_throughput[2][k] = simulate_throughput_CO(K_i, K_j, K_z,
                                                            L_LOS=L_LOS_InFSL_5GNR_1,
                                                            p_LOS=p_LOS_InFSL[
                                                                k],
                                                            epsilon=epsilon_5GNR_1,
                                                            rint=rint[k],
                                                            M_int=M_int,
                                                            ra=ra[k],
                                                            l=l_InF_5GNR1,
                                                            BW=BW,
                                                            M_MBS=M_MBS,
                                                            M_SBS=M_SBS,
                                                            N=N,
                                                            R_LTE=R_LTE,
                                                            m_omega=m_omega,
                                                            numSamples=numSamples,
                                                            tau_timeout_wired=tau_timeout_wired,
                                                            theta_omega=theta_omega,
                                                            d0=d0,
                                                            Pt_norm=Pt_norm,
                                                            R_5GNR=R_5GNR,
                                                            thres_tx_5GNR=thres_tx_5GNR,
                                                            thres_tx_LTE=thres_tx_LTE
                                                            )
        simulated_throughput[3][k] = simulate_throughput_CO(K_i, K_j, K_z,
                                                            L_LOS=L_LOS_InFSL_5GNR_2,
                                                            p_LOS=p_LOS_InFSL[
                                                                k],
                                                            epsilon=epsilon_5GNR_2,
                                                            rint=rint[k],
                                                            M_int=M_int,
                                                            ra=ra[k],
                                                            l=l_InF_5GNR2,
                                                            BW=BW,
                                                            M_MBS=M_MBS,
                                                            M_SBS=M_SBS,
                                                            N=N,
                                                            R_LTE=R_LTE,
                                                            m_omega=m_omega,
                                                            numSamples=numSamples,
                                                            tau_timeout_wired=tau_timeout_wired,
                                                            theta_omega=theta_omega,
                                                            d0=d0,
                                                            Pt_norm=Pt_norm,
                                                            R_5GNR=R_5GNR,
                                                            thres_tx_5GNR=thres_tx_5GNR,
                                                            thres_tx_LTE=thres_tx_LTE)

        # Compute throughput
        computed_throughput[0][k] = compute_throughput_CL(K_i, K_j, K_z,
                                                          L_LOS=L_LOS_InFSL_wifi,
                                                          thres_tx=thres_tx_wifi,
                                                          p_LOS=p_LOS_InFSL[k],
                                                          epsilon=epsilon_wifi,
                                                          rint=rint[k],
                                                          M_int=M_int + 1,
                                                          ra=ra[k], lte=False,
                                                          l=l_InF_wifi,
                                                          tr=tr_InF,
                                                          BW=BW,
                                                          M_MBS=M_MBS,
                                                          M_SBS=M_SBS,
                                                          N=N,
                                                          R_LTE=R_LTE,
                                                          m_omega=m_omega,
                                                          tau_timeout_wired=tau_timeout_wired,
                                                          theta_omega=theta_omega,
                                                          d0=d0,
                                                          Pt_norm=Pt_norm,
                                                          R_wifi=R_wifi)
        computed_throughput[1][k] = compute_throughput_CL(K_i, K_j, K_z,
                                                          L_LOS=L_LOS_InFSL_LTE,
                                                          thres_tx=thres_tx_LTE,
                                                          p_LOS=p_LOS_InFSL[k],
                                                          epsilon=epsilon_LTE,
                                                          rint=rint[k],
                                                          M_int=M_int,
                                                          ra=ra[k],
                                                          lte=True,
                                                          l=l_InF_LTE,
                                                          tr=tr_InF,
                                                          BW=BW,
                                                          M_MBS=M_MBS,
                                                          M_SBS=M_SBS,
                                                          N=N,
                                                          R_LTE=R_LTE,
                                                          m_omega=m_omega,
                                                          tau_timeout_wired=tau_timeout_wired,
                                                          theta_omega=theta_omega,
                                                          d0=d0,
                                                          Pt_norm=Pt_norm,
                                                          R_wifi=R_wifi)
        computed_throughput[2][k] = compute_throughput_CO(K_i, K_j, K_z,
                                                          L_LOS=L_LOS_InFSL_5GNR_1,
                                                          p_LOS=p_LOS_InFSL[k],
                                                          epsilon=epsilon_5GNR_1,
                                                          rint=rint[k],
                                                          M_int=M_int,
                                                          ra=ra[k],
                                                          l=l_InF_5GNR1,
                                                          tr=tr_InF,
                                                          BW=BW,
                                                          M_MBS=M_MBS,
                                                          M_SBS=M_SBS,
                                                          N=N,
                                                          R_LTE=R_LTE,
                                                          m_omega=m_omega,
                                                          tau_timeout_wired=tau_timeout_wired,
                                                          theta_omega=theta_omega,
                                                          d0=d0,
                                                          Pt_norm=Pt_norm,
                                                          R_5GNR=R_5GNR,
                                                          thres_tx_5GNR=thres_tx_5GNR,
                                                          thres_tx_LTE=thres_tx_LTE
                                                          )
        computed_throughput[3][k] = compute_throughput_CO(K_i, K_j, K_z,
                                                          L_LOS=L_LOS_InFSL_5GNR_2,
                                                          p_LOS=p_LOS_InFSL[k],
                                                          epsilon=epsilon_5GNR_2,
                                                          rint=rint[k],
                                                          M_int=M_int,
                                                          ra=ra[k],
                                                          l=l_InF_5GNR2,
                                                          tr=tr_InF,
                                                          BW=BW,
                                                          M_MBS=M_MBS,
                                                          M_SBS=M_SBS,
                                                          N=N,
                                                          R_LTE=R_LTE,
                                                          m_omega=m_omega,
                                                          tau_timeout_wired=tau_timeout_wired,
                                                          theta_omega=theta_omega,
                                                          d0=d0,
                                                          Pt_norm=Pt_norm,
                                                          R_5GNR=R_5GNR,
                                                          thres_tx_5GNR=thres_tx_5GNR,
                                                          thres_tx_LTE=thres_tx_LTE)

        print(datetime.datetime.now().strftime(
            dateFormat + "|" + timeFormat) + ": "
            + "\n\t ra (m): {:n} ".format(ra[k])
              + "\n\t Throughput (Mbps) - 802.11ac (Computed): \t{"
                ":.4n}".format(
            computed_throughput[0][k] / 1e6)
            + "\n\t Throughput (Mbps) - 802.11ac (Simulated): \t{"
            ":.4n}".format(
            simulated_throughput[0][k] / 1e6)
            + "\n\t Throughput (Mbps) - LTE (Computed): \t\t{:.4n}".format(
            computed_throughput[1][k] / 1e6)
            + "\n\t Throughput (Mbps) - LTE (Simulated): \t\t{:.4n}".format(
            simulated_throughput[1][k] / 1e6)
            + "\n\t Throughput (Mbps) - 5GNR-1 (Computed): \t{:.4n}".format(
            computed_throughput[2][k] / 1e6)
            + "\n\t Throughput (Mbps) - 5GNR-1 (Simulated): \t{:.4n}".format(
            simulated_throughput[2][k] / 1e6)
            + "\n\t Throughput (Mbps) - 5GNR-2 (Computed): \t{:.4n}".format(
            computed_throughput[3][k] / 1e6)
            + "\n\t Throughput (Mbps) - 5GNR-2 (Simulated): \t{:.4n}".format(
            simulated_throughput[3][k] / 1e6)
        )
    os.makedirs(name=os.path.dirname(p=file_name), exist_ok=True)
    spio.savemat(file_name,
                 {
                     'simulated_throughput': simulated_throughput,
                     'computed_throughput': computed_throughput,
                     'ra': ra
                 })


##
# This function generates results for throughput vs ra in the RMa scenario


##
# This function generates results for eta_ee vs ra in the UMa scenario
def eta_ee_vs_ra_UMa(file_name: str, input_variables: Dict[str, List]) -> None:
    K_size, BW, v, N, _, numSamples = process_raw_input(input_variables)
    fd_5GNR_2 = v * fc_5GNR_2 / (3 * 10 ** 8)
    epsilon_5GNR_2 = j0(2 * pi * fd_5GNR_2 * T)
    fd_5GNR_1 = v * fc_5GNR_1 / (3 * 10 ** 8)
    epsilon_5GNR_1 = j0(2 * pi * fd_5GNR_1 * T)
    R_wifi = 2.32e6 / BW  # 802.11 for MCS 0, taken from "Physical Layer
    thres_tx_wifi = 2 ** R_wifi - 1  # 802.11ac SINR threshold
    fd_wifi = v * fc_wifi / (3 * 10 ** 8)
    epsilon_wifi = j0(2 * pi * fd_wifi * T)
    fd_LTE = v * fc_LTE / (3 * 10 ** 8)
    epsilon_LTE = j0(2 * pi * fd_LTE * T)

    R_LTE = 4.08e6 / BW  # LTE, for MCS 6, taken from "Physical Layer
    # Evaluation
    # of V2X Communications Technologies: 5G NR-V2X, LTE-V2X, IEEE 802.11bd,
    # and IEEE 802.11p"
    thres_tx_LTE = 2 ** R_LTE - 1  # LTE SINR threshold
    eta = 20 ** ((-174 + 10 * log10(BW)) / 10) * 1e-3  # noise power
    Pt_norm = Pt / (eta)  # Normalised transmit power
    theta_omega = c_wired * K_size  # Backhaul delay Gamma scale parameter
    tau_timeout_wired = m_omega * theta_omega  # Wired backhaul timeout

    R_5GNR = 4.63e6 / BW  # 5G NR, for MCS 6, taken from "Physical Layer
    # Evaluation of V2X Communications Technologies: 5G NR-V2X, LTE-V2X,
    # IEEE 802.11bd, and IEEE 802.11p"
    thres_tx_5GNR = 2 ** R_5GNR - 1  # 5G NR SINR threshold

    desc = datetime.datetime.now().strftime(
        dateFormat + "|" + timeFormat) + (": Simulating and computing eta_ee "
                                          "vs ra in the UMa scenario")

    # Initialise variables
    ra_min = 20
    ra_max = 400  # tested 400
    ra_step = 20
    ra = np.arange(start=ra_min, stop=ra_max + ra_step, step=ra_step)
    rint = 2 * ra  # tested 2
    p_LOS_UMa = [18 / ra[i] + (1 - 18 / ra[i]) * exp(-(ra[i]) / 63) for i in
                 range(len(ra))]

    computed_eta_ee = np.zeros(shape=(4, len(ra)))
    simulated_eta_ee = np.zeros(shape=(4, len(ra)))

    for k in tqdm(range(len(ra)), desc=desc):
        # Simulate eta_ee
        simulated_eta_ee[0][k] = simulate_throughput_CL(K_i, K_j, K_z,
                                                        L_LOS=L_LOS_UMa_wifi,
                                                        thres_tx=thres_tx_wifi,
                                                        p_LOS=p_LOS_UMa[k],
                                                        epsilon=epsilon_wifi,
                                                        M_int=M_int + 1,
                                                        ra=ra[k], rint=rint[k],
                                                        lte=False, l=l_UMa,
                                                        d0=1, BW=BW,
                                                        M_MBS=M_MBS,
                                                        M_SBS=M_SBS,
                                                        N=N,
                                                        Pt_norm=Pt_norm,
                                                        R_LTE=R_LTE,
                                                        R_wifi=R_wifi,
                                                        m_omega=m_omega,
                                                        numSamples=numSamples,
                                                        tau_timeout_wired=tau_timeout_wired,
                                                        theta_omega=theta_omega
                                                        ) / Pt
        simulated_eta_ee[1][k] = simulate_throughput_CL(K_i, K_j, K_z,
                                                        L_LOS=L_LOS_UMa_LTE,
                                                        thres_tx=thres_tx_LTE,
                                                        p_LOS=p_LOS_UMa[k],
                                                        epsilon=epsilon_LTE,
                                                        M_int=M_int, ra=ra[k],
                                                        rint=rint[k], lte=True,
                                                        l=l_UMa, d0=1, BW=BW,
                                                        M_MBS=M_MBS,
                                                        M_SBS=M_SBS,
                                                        N=N,
                                                        Pt_norm=Pt_norm,
                                                        R_LTE=R_LTE,
                                                        R_wifi=R_wifi,
                                                        m_omega=m_omega,
                                                        numSamples=numSamples,
                                                        tau_timeout_wired=tau_timeout_wired,
                                                        theta_omega=theta_omega) / Pt
        simulated_eta_ee[2][k] = simulate_throughput_CO(K_i, K_j, K_z,
                                                        L_LOS=L_LOS_UMa_5GNR_1,
                                                        p_LOS=p_LOS_UMa[k],
                                                        epsilon=epsilon_5GNR_1,
                                                        M_int=M_int, ra=ra[k],
                                                        rint=rint[k], l=l_UMa,
                                                        d0=1, BW=BW,
                                                        M_MBS=M_MBS,
                                                        M_SBS=M_SBS,
                                                        N=N,
                                                        Pt_norm=Pt_norm,
                                                        R_LTE=R_LTE,
                                                        m_omega=m_omega,
                                                        numSamples=numSamples,
                                                        tau_timeout_wired=tau_timeout_wired,
                                                        theta_omega=theta_omega,
                                                        R_5GNR=R_5GNR,
                                                        thres_tx_LTE=thres_tx_LTE,
                                                        thres_tx_5GNR=thres_tx_5GNR) / Pt
        simulated_eta_ee[3][k] = simulate_throughput_CO(K_i, K_j, K_z,
                                                        L_LOS=L_LOS_UMa_5GNR_2,
                                                        p_LOS=p_LOS_UMa[k],
                                                        epsilon=epsilon_5GNR_2,
                                                        M_int=M_int, ra=ra[k],
                                                        rint=rint[k], l=l_UMa,
                                                        d0=1, BW=BW,
                                                        M_MBS=M_MBS,
                                                        M_SBS=M_SBS,
                                                        N=N,
                                                        Pt_norm=Pt_norm,
                                                        R_LTE=R_LTE,
                                                        m_omega=m_omega,
                                                        numSamples=numSamples,
                                                        tau_timeout_wired=tau_timeout_wired,
                                                        theta_omega=theta_omega,
                                                        R_5GNR=R_5GNR,
                                                        thres_tx_5GNR=thres_tx_5GNR,
                                                        thres_tx_LTE=thres_tx_LTE
                                                        ) / Pt

        # Compute eta_ee
        computed_eta_ee[0][k] = compute_throughput_CL(K_i, K_j, K_z,
                                                      L_LOS=L_LOS_UMa_wifi,
                                                      thres_tx=thres_tx_wifi,
                                                      p_LOS=p_LOS_UMa[k],
                                                      epsilon=epsilon_wifi,
                                                      M_int=M_int + 1,
                                                      ra=ra[k],
                                                      rint=rint[k], lte=False,
                                                      l=l_UMa, tr=tr_UMa,
                                                      d0=1, BW=BW,
                                                      M_MBS=M_MBS,
                                                      M_SBS=M_SBS,
                                                      N=N,
                                                      Pt_norm=Pt_norm,
                                                      R_LTE=R_LTE,
                                                      m_omega=m_omega,
                                                      tau_timeout_wired=tau_timeout_wired,
                                                      theta_omega=theta_omega,
                                                      R_wifi=R_wifi) / Pt
        computed_eta_ee[1][k] = compute_throughput_CL(K_i, K_j, K_z,
                                                      L_LOS=L_LOS_UMa_LTE,
                                                      thres_tx=thres_tx_LTE,
                                                      p_LOS=p_LOS_UMa[k],
                                                      epsilon=epsilon_LTE,
                                                      M_int=M_int, ra=ra[k],
                                                      rint=rint[k], lte=True,
                                                      l=l_UMa, tr=tr_UMa,
                                                      d0=1, BW=BW,
                                                      M_MBS=M_MBS,
                                                      M_SBS=M_SBS,
                                                      N=N,
                                                      Pt_norm=Pt_norm,
                                                      R_LTE=R_LTE,
                                                      m_omega=m_omega,
                                                      tau_timeout_wired=tau_timeout_wired,
                                                      theta_omega=theta_omega,
                                                      R_wifi=R_wifi) / Pt
        computed_eta_ee[2][k] = compute_throughput_CO(K_i, K_j, K_z,
                                                      L_LOS=L_LOS_UMa_5GNR_1,
                                                      p_LOS=p_LOS_UMa[k],
                                                      epsilon=epsilon_5GNR_1,
                                                      M_int=M_int, ra=ra[k],
                                                      rint=rint[k], l=l_UMa,
                                                      tr=tr_UMa, d0=1, BW=BW,
                                                      M_MBS=M_MBS,
                                                      M_SBS=M_SBS,
                                                      N=N,
                                                      Pt_norm=Pt_norm,
                                                      R_LTE=R_LTE,
                                                      m_omega=m_omega,
                                                      tau_timeout_wired=tau_timeout_wired,
                                                      theta_omega=theta_omega,
                                                      R_5GNR=R_5GNR,
                                                      thres_tx_5GNR=thres_tx_5GNR,
                                                      thres_tx_LTE=thres_tx_LTE) / Pt
        computed_eta_ee[3][k] = compute_throughput_CO(K_i, K_j, K_z,
                                                      L_LOS=L_LOS_UMa_5GNR_2,
                                                      p_LOS=p_LOS_UMa[k],
                                                      epsilon=epsilon_5GNR_2,
                                                      M_int=M_int, ra=ra[k],
                                                      rint=rint[k], l=l_UMa,
                                                      tr=tr_UMa, d0=1, BW=BW,
                                                      M_MBS=M_MBS,
                                                      M_SBS=M_SBS,
                                                      N=N,
                                                      Pt_norm=Pt_norm,
                                                      R_LTE=R_LTE,
                                                      m_omega=m_omega,
                                                      tau_timeout_wired=tau_timeout_wired,
                                                      theta_omega=theta_omega,
                                                      R_5GNR=R_5GNR,
                                                      thres_tx_5GNR=thres_tx_5GNR,
                                                      thres_tx_LTE=thres_tx_LTE) / Pt

        print(datetime.datetime.now().strftime(
            dateFormat + "|" + timeFormat) + ": "
            + "\n\t ra (m): {:n} ".format(ra[k])
              + "\n\t EE (Mbps/Watts) - 802.11ac (Computed): \t{:.4n}".format(
            computed_eta_ee[0][k] / 1e6)
            + "\n\t EE (Mbps/Watts) - 802.11ac (Simulated): \t{:.4n}".format(
            simulated_eta_ee[0][k] / 1e6)
            + "\n\t EE (Mbps/Watts) - LTE (Computed): \t\t{:.4n}".format(
            computed_eta_ee[1][k] / 1e6)
            + "\n\t EE (Mbps/Watts) - LTE (Simulated): \t\t{:.4n}".format(
            simulated_eta_ee[1][k] / 1e6)
            + "\n\t EE (Mbps/Watts) - 5GNR-1 (Computed): \t\t{:.4n}".format(
            computed_eta_ee[2][k] / 1e6)
            + "\n\t EE (Mbps/Watts) - 5GNR-1 (Simulated): \t\t{:.4n}".format(
            simulated_eta_ee[2][k] / 1e6)
            + "\n\t EE (Mbps/Watts) - 5GNR-2 (Computed): \t\t{:.4n}".format(
            computed_eta_ee[3][k] / 1e6)
            + "\n\t EE (Mbps/Watts) - 5GNR-2 (Simulated): \t\t{:.4n}".format(
            simulated_eta_ee[3][k] / 1e6)
        )
    os.makedirs(name=os.path.dirname(p=file_name), exist_ok=True)
    spio.savemat(file_name,
                 {
                     'simulated_eta_ee': simulated_eta_ee,
                     'computed_eta_ee': computed_eta_ee,
                     'ra': ra
                 })


##
# This function generates results for eta_ee vs ra in the InF-SL scenario
def eta_ee_vs_ra_InFSL(file_name: str,
                       input_variables: Dict[str, List]) -> None:
    K_size, BW, v, N, _, numSamples = process_raw_input(input_variables)
    fd_5GNR_2 = v * fc_5GNR_2 / (3 * 10 ** 8)
    epsilon_5GNR_2 = j0(2 * pi * fd_5GNR_2 * T)
    fd_5GNR_1 = v * fc_5GNR_1 / (3 * 10 ** 8)
    epsilon_5GNR_1 = j0(2 * pi * fd_5GNR_1 * T)
    R_wifi = 2.32e6 / BW  # 802.11 for MCS 0, taken from "Physical Layer
    thres_tx_wifi = 2 ** R_wifi - 1  # 802.11ac SINR threshold
    fd_wifi = v * fc_wifi / (3 * 10 ** 8)
    epsilon_wifi = j0(2 * pi * fd_wifi * T)
    fd_LTE = v * fc_LTE / (3 * 10 ** 8)
    epsilon_LTE = j0(2 * pi * fd_LTE * T)

    R_LTE = 4.08e6 / BW  # LTE, for MCS 6, taken from "Physical Layer
    # Evaluation
    # of V2X Communications Technologies: 5G NR-V2X, LTE-V2X, IEEE 802.11bd,
    # and IEEE 802.11p"
    thres_tx_LTE = 2 ** R_LTE - 1  # LTE SINR threshold
    eta = 20 ** ((-174 + 10 * log10(BW)) / 10) * 1e-3  # noise power
    Pt_norm = Pt / (eta)  # Normalised transmit power
    theta_omega = c_wired * K_size  # Backhaul delay Gamma scale parameter
    tau_timeout_wired = m_omega * theta_omega  # Wired backhaul timeout

    R_5GNR = 4.63e6 / BW  # 5G NR, for MCS 6, taken from "Physical Layer
    # Evaluation of V2X Communications Technologies: 5G NR-V2X, LTE-V2X,
    # IEEE 802.11bd, and IEEE 802.11p"
    thres_tx_5GNR = 2 ** R_5GNR - 1  # 5G NR SINR threshold

    desc = datetime.datetime.now().strftime(
        dateFormat + "|" + timeFormat) + (": Simulating and computing eta_ee "
                                          "vs ra in the InF-SL scenario")

    # Initialise variables
    ra_min = 20
    ra_max = 200  # tested 200
    ra_step = 20
    ra = np.arange(start=ra_min, stop=ra_max + ra_step, step=ra_step)
    rint = 2 * ra  # tested 2
    p_LOS_InFSL = [exp(-(ra[i]) / (-dclutter / ln(1 - LOS_r))) for i in
                   range(len(ra))]

    computed_eta_ee = np.zeros(shape=(4, len(ra)))
    simulated_eta_ee = np.zeros(shape=(4, len(ra)))

    for k in tqdm(range(len(ra)), desc=desc):
        # Simulate throughput
        simulated_eta_ee[0][k] = simulate_throughput_CL(K_i, K_j, K_z,
                                                        L_LOS=L_LOS_InFSL_wifi,
                                                        thres_tx=thres_tx_wifi,
                                                        p_LOS=p_LOS_InFSL[k],
                                                        epsilon=epsilon_wifi,
                                                        rint=rint[k],
                                                        M_int=M_int + 1,
                                                        ra=ra[k], lte=False,
                                                        l=l_InF_wifi,
                                                        R_wifi=R_wifi,
                                                        R_LTE=R_LTE,
                                                        Pt_norm=Pt_norm,
                                                        M_MBS=M_MBS,
                                                        M_SBS=M_SBS,
                                                        N=N,
                                                        tau_timeout_wired=tau_timeout_wired,
                                                        m_omega=m_omega,
                                                        theta_omega=theta_omega,
                                                        numSamples=numSamples,
                                                        d0=d0, BW=BW) / Pt

        simulated_eta_ee[1][k] = simulate_throughput_CL(K_i, K_j, K_z,
                                                        L_LOS=L_LOS_InFSL_LTE,
                                                        thres_tx=thres_tx_LTE,
                                                        p_LOS=p_LOS_InFSL[k],
                                                        epsilon=epsilon_LTE,
                                                        rint=rint[k],
                                                        M_int=M_int, ra=ra[k],
                                                        lte=True,
                                                        l=l_InF_LTE,
                                                        R_wifi=R_wifi,
                                                        R_LTE=R_LTE,
                                                        Pt_norm=Pt_norm,
                                                        M_MBS=M_MBS,
                                                        M_SBS=M_SBS,
                                                        N=N,
                                                        tau_timeout_wired=tau_timeout_wired,
                                                        m_omega=m_omega,
                                                        theta_omega=theta_omega,
                                                        numSamples=numSamples,
                                                        d0=d0, BW=BW) / Pt
        simulated_eta_ee[2][k] = simulate_throughput_CO(K_i, K_j, K_z,
                                                        L_LOS=L_LOS_InFSL_5GNR_1,
                                                        p_LOS=p_LOS_InFSL[k],
                                                        epsilon=epsilon_5GNR_1,
                                                        rint=rint[k],
                                                        M_int=M_int, ra=ra[k],
                                                        l=l_InF_5GNR1,
                                                        M_MBS=M_MBS,
                                                        M_SBS=M_SBS,
                                                        N=N,
                                                        Pt_norm=Pt_norm,
                                                        R_5GNR=R_5GNR,
                                                        R_LTE=R_LTE,
                                                        m_omega=m_omega,
                                                        numSamples=numSamples,
                                                        tau_timeout_wired=tau_timeout_wired,
                                                        theta_omega=theta_omega,
                                                        thres_tx_5GNR=thres_tx_5GNR,
                                                        thres_tx_LTE=thres_tx_LTE,
                                                        d0=d0,
                                                        BW=BW
                                                        ) / Pt
        simulated_eta_ee[3][k] = simulate_throughput_CO(K_i, K_j, K_z,
                                                        L_LOS=L_LOS_InFSL_5GNR_2,
                                                        p_LOS=p_LOS_InFSL[k],
                                                        epsilon=epsilon_5GNR_2,
                                                        rint=rint[k],
                                                        M_int=M_int, ra=ra[k],
                                                        l=l_InF_5GNR2,
                                                        M_MBS=M_MBS,
                                                        M_SBS=M_SBS,
                                                        N=N,
                                                        Pt_norm=Pt_norm,
                                                        R_5GNR=R_5GNR,
                                                        R_LTE=R_LTE,
                                                        m_omega=m_omega,
                                                        numSamples=numSamples,
                                                        tau_timeout_wired=tau_timeout_wired,
                                                        theta_omega=theta_omega,
                                                        thres_tx_5GNR=thres_tx_5GNR,
                                                        thres_tx_LTE=thres_tx_LTE,
                                                        d0=d0,
                                                        BW=BW
                                                        ) / Pt

        # Compute throughput
        computed_eta_ee[0][k] = compute_throughput_CL(K_i, K_j, K_z,
                                                      L_LOS=L_LOS_InFSL_wifi,
                                                      thres_tx=thres_tx_wifi,
                                                      p_LOS=p_LOS_InFSL[k],
                                                      epsilon=epsilon_wifi,
                                                      rint=rint[k],
                                                      M_int=M_int + 1,
                                                      ra=ra[k],
                                                      lte=False, l=l_InF_wifi,
                                                      tr=tr_InF,
                                                      M_MBS=M_MBS,
                                                      M_SBS=M_SBS,
                                                      N=N,
                                                      Pt_norm=Pt_norm,
                                                      R_LTE=R_LTE,
                                                      R_wifi=R_wifi,
                                                      d0=d0,
                                                      m_omega=m_omega,
                                                      tau_timeout_wired=tau_timeout_wired,
                                                      theta_omega=theta_omega,
                                                      BW=BW
                                                      ) / Pt
        computed_eta_ee[1][k] = compute_throughput_CL(K_i, K_j, K_z,
                                                      L_LOS=L_LOS_InFSL_LTE,
                                                      thres_tx=thres_tx_LTE,
                                                      p_LOS=p_LOS_InFSL[k],
                                                      epsilon=epsilon_LTE,
                                                      rint=rint[k],
                                                      M_int=M_int,
                                                      ra=ra[k], lte=True,
                                                      l=l_InF_LTE,
                                                      tr=tr_InF,
                                                      M_MBS=M_MBS,
                                                      M_SBS=M_SBS,
                                                      N=N,
                                                      Pt_norm=Pt_norm,
                                                      R_LTE=R_LTE,
                                                      R_wifi=R_wifi,
                                                      d0=d0,
                                                      m_omega=m_omega,
                                                      tau_timeout_wired=tau_timeout_wired,
                                                      theta_omega=theta_omega,
                                                      BW=BW) / Pt
        computed_eta_ee[2][k] = compute_throughput_CO(K_i, K_j, K_z,
                                                      L_LOS=L_LOS_InFSL_5GNR_1,
                                                      p_LOS=p_LOS_InFSL[k],
                                                      epsilon=epsilon_5GNR_1,
                                                      rint=rint[k],
                                                      M_int=M_int,
                                                      ra=ra[k], l=l_InF_5GNR1,
                                                      tr=tr_InF,
                                                      M_MBS=M_MBS,
                                                      M_SBS=M_SBS,
                                                      N=N,
                                                      Pt_norm=Pt_norm,
                                                      R_5GNR=R_5GNR,
                                                      R_LTE=R_LTE,
                                                      d0=d0,
                                                      m_omega=m_omega,
                                                      tau_timeout_wired=tau_timeout_wired,
                                                      theta_omega=theta_omega,
                                                      thres_tx_5GNR=thres_tx_5GNR,
                                                      thres_tx_LTE=thres_tx_LTE,
                                                      BW=BW
                                                      ) / Pt
        computed_eta_ee[3][k] = compute_throughput_CO(K_i, K_j, K_z,
                                                      L_LOS=L_LOS_InFSL_5GNR_2,
                                                      p_LOS=p_LOS_InFSL[k],
                                                      epsilon=epsilon_5GNR_2,
                                                      rint=rint[k],
                                                      M_int=M_int,
                                                      ra=ra[k], l=l_InF_5GNR2,
                                                      tr=tr_InF,
                                                      M_MBS=M_MBS,
                                                      M_SBS=M_SBS,
                                                      N=N,
                                                      Pt_norm=Pt_norm,
                                                      R_5GNR=R_5GNR,
                                                      R_LTE=R_LTE,
                                                      d0=d0,
                                                      m_omega=m_omega,
                                                      tau_timeout_wired=tau_timeout_wired,
                                                      theta_omega=theta_omega,
                                                      thres_tx_5GNR=thres_tx_5GNR,
                                                      thres_tx_LTE=thres_tx_LTE,
                                                      BW=BW) / Pt

        print(datetime.datetime.now().strftime(
            dateFormat + "|" + timeFormat) + ": "
            + "\n\t ra (m): {:n} ".format(ra[k])
              + "\n\t EE (Mbps) - 802.11ac (Computed): \t{:.4n}".format(
            computed_eta_ee[0][k] / 1e6)
            + "\n\t EE (Mbps) - 802.11ac (Simulated): \t{:.4n}".format(
            simulated_eta_ee[0][k] / 1e6)
            + "\n\t EE (Mbps) - LTE (Computed): \t\t{:.4n}".format(
            computed_eta_ee[1][k] / 1e6)
            + "\n\t EE (Mbps) - LTE (Simulated): \t\t{:.4n}".format(
            simulated_eta_ee[1][k] / 1e6)
            + "\n\t EE (Mbps) - 5GNR-1 (Computed): \t\t{:.4n}".format(
            computed_eta_ee[2][k] / 1e6)
            + "\n\t EE (Mbps) - 5GNR-1 (Simulated): \t\t{:.4n}".format(
            simulated_eta_ee[2][k] / 1e6)
            + "\n\t EE (Mbps) - 5GNR-2 (Computed): \t\t{:.4n}".format(
            computed_eta_ee[3][k] / 1e6)
            + "\n\t EE (Mbps) - 5GNR-2 (Simulated): \t\t{:.4n}".format(
            simulated_eta_ee[3][k] / 1e6)
        )
    os.makedirs(name=os.path.dirname(p=file_name), exist_ok=True)
    spio.savemat(file_name,
                 {
                     'simulated_eta_ee': simulated_eta_ee,
                     'computed_eta_ee': computed_eta_ee,
                     'ra': ra
                 })


##
# This function generates results for eta_ee vs Pt_dBm in the UMa scenario
def eta_ee_vs_Pt_dBm_UMa(file_name: str,
                         input_variables: Dict[str, List[float]]) -> None:
    K_size, BW, v, N, _, numSamples = process_raw_input(input_variables)
    fd_5GNR_2 = v * fc_5GNR_2 / (3 * 10 ** 8)
    epsilon_5GNR_2 = j0(2 * pi * fd_5GNR_2 * T)
    fd_5GNR_1 = v * fc_5GNR_1 / (3 * 10 ** 8)
    epsilon_5GNR_1 = j0(2 * pi * fd_5GNR_1 * T)
    R_wifi = 2.32e6 / BW  # 802.11 for MCS 0, taken from "Physical Layer
    thres_tx_wifi = 2 ** R_wifi - 1  # 802.11ac SINR threshold
    fd_wifi = v * fc_wifi / (3 * 10 ** 8)
    epsilon_wifi = j0(2 * pi * fd_wifi * T)
    fd_LTE = v * fc_LTE / (3 * 10 ** 8)
    epsilon_LTE = j0(2 * pi * fd_LTE * T)

    R_LTE = 4.08e6 / BW  # LTE, for MCS 6, taken from "Physical Layer
    # Evaluation
    # of V2X Communications Technologies: 5G NR-V2X, LTE-V2X, IEEE 802.11bd,
    # and IEEE 802.11p"
    thres_tx_LTE = 2 ** R_LTE - 1  # LTE SINR threshold
    eta = 20 ** ((-174 + 10 * log10(BW)) / 10) * 1e-3  # noise power
    # Pt_norm = Pt / (eta)  # Normalised transmit power
    theta_omega = c_wired * K_size  # Backhaul delay Gamma scale parameter
    tau_timeout_wired = m_omega * theta_omega  # Wired backhaul timeout

    R_5GNR = 4.63e6 / BW  # 5G NR, for MCS 6, taken from "Physical Layer
    # Evaluation of V2X Communications Technologies: 5G NR-V2X, LTE-V2X,
    # IEEE 802.11bd, and IEEE 802.11p"
    thres_tx_5GNR = 2 ** R_5GNR - 1  # 5G NR SINR threshold

    desc = datetime.datetime.now().strftime(
        dateFormat + "|" + timeFormat) + (": Simulating and computing eta_ee "
                                          "vs Pt_dBm in the UMa scenario")

    # Initialise variables
    Pt_dBm_min = 10
    Pt_dBm_max = 40  # tested
    Pt_dBm_step = 2
    Pt_dBm = np.arange(start=Pt_dBm_min, stop=Pt_dBm_max + Pt_dBm_step,
                       step=Pt_dBm_step)
    Pt = 10 ** (Pt_dBm / 10) * 1e-3
    Pt_norm = Pt / (eta)
    ra = 200
    rint = 2 * ra  # tested 2
    p_LOS_UMa = 18 / ra + (1 - 18 / ra) * exp(-(ra) / 63)

    computed_eta_ee = np.zeros(shape=(4, len(Pt_dBm)))
    simulated_eta_ee = np.zeros(shape=(4, len(Pt_dBm)))

    for k in tqdm(range(len(Pt_dBm)), desc=desc):
        # Simulate eta_ee
        simulated_eta_ee[0][k] = simulate_throughput_CL(K_i, K_j, K_z,
                                                        L_LOS=L_LOS_UMa_wifi,
                                                        thres_tx=thres_tx_wifi,
                                                        p_LOS=p_LOS_UMa,
                                                        epsilon=epsilon_wifi,
                                                        M_int=M_int + 1, ra=ra,
                                                        rint=rint, lte=False,
                                                        l=l_UMa, d0=1,
                                                        Pt_norm=Pt_norm[k],
                                                        BW=BW,
                                                        M_MBS=M_MBS,
                                                        M_SBS=M_SBS,
                                                        N=N,
                                                        R_LTE=R_LTE,
                                                        R_wifi=R_wifi,
                                                        m_omega=m_omega,
                                                        numSamples=numSamples,
                                                        tau_timeout_wired=tau_timeout_wired,
                                                        theta_omega=theta_omega
                                                        ) / \
            Pt[k]
        simulated_eta_ee[1][k] = simulate_throughput_CL(K_i, K_j, K_z,
                                                        L_LOS=L_LOS_UMa_LTE,
                                                        thres_tx=thres_tx_LTE,
                                                        p_LOS=p_LOS_UMa,
                                                        epsilon=epsilon_LTE,
                                                        M_int=M_int, ra=ra,
                                                        rint=rint, lte=True,
                                                        l=l_UMa, d0=1,
                                                        Pt_norm=Pt_norm[k],
                                                        BW=BW,
                                                        M_MBS=M_MBS,
                                                        M_SBS=M_SBS,
                                                        N=N,
                                                        R_LTE=R_LTE,
                                                        R_wifi=R_wifi,
                                                        m_omega=m_omega,
                                                        numSamples=numSamples,
                                                        tau_timeout_wired=tau_timeout_wired,
                                                        theta_omega=theta_omega) / \
            Pt[k]
        simulated_eta_ee[2][k] = simulate_throughput_CO(K_i, K_j, K_z,
                                                        L_LOS=L_LOS_UMa_5GNR_1,
                                                        p_LOS=p_LOS_UMa,
                                                        epsilon=epsilon_5GNR_1,
                                                        M_int=M_int, ra=ra,
                                                        rint=rint, l=l_UMa,
                                                        d0=1,
                                                        Pt_norm=Pt_norm[k],
                                                        BW=BW,
                                                        M_MBS=M_MBS,
                                                        M_SBS=M_SBS,
                                                        N=N,
                                                        R_LTE=R_LTE,
                                                        m_omega=m_omega,
                                                        numSamples=numSamples,
                                                        tau_timeout_wired=tau_timeout_wired,
                                                        theta_omega=theta_omega,
                                                        R_5GNR=R_5GNR,
                                                        thres_tx_LTE=thres_tx_LTE,
                                                        thres_tx_5GNR=thres_tx_5GNR) / \
            Pt[k]
        simulated_eta_ee[3][k] = simulate_throughput_CO(K_i, K_j, K_z,
                                                        L_LOS=L_LOS_UMa_5GNR_2,
                                                        p_LOS=p_LOS_UMa,
                                                        epsilon=epsilon_5GNR_2,
                                                        M_int=M_int, ra=ra,
                                                        rint=rint, l=l_UMa,
                                                        d0=1,
                                                        Pt_norm=Pt_norm[k],
                                                        BW=BW,
                                                        M_MBS=M_MBS,
                                                        M_SBS=M_SBS,
                                                        N=N,
                                                        R_LTE=R_LTE,
                                                        m_omega=m_omega,
                                                        numSamples=numSamples,
                                                        tau_timeout_wired=tau_timeout_wired,
                                                        theta_omega=theta_omega,
                                                        R_5GNR=R_5GNR,
                                                        thres_tx_LTE=thres_tx_LTE,
                                                        thres_tx_5GNR=thres_tx_5GNR) / \
            Pt[k]

        # Compute eta_ee
        computed_eta_ee[0][k] = compute_throughput_CL(K_i, K_j, K_z,
                                                      L_LOS=L_LOS_UMa_wifi,
                                                      thres_tx=thres_tx_wifi,
                                                      p_LOS=p_LOS_UMa,
                                                      epsilon=epsilon_wifi,
                                                      M_int=M_int + 1, ra=ra,
                                                      rint=rint, lte=False,
                                                      l=l_UMa, tr=tr_UMa, d0=1,
                                                      Pt_norm=Pt_norm[k],
                                                      BW=BW,
                                                      M_MBS=M_MBS,
                                                      M_SBS=M_SBS,
                                                      N=N,
                                                      R_LTE=R_LTE,
                                                      R_wifi=R_wifi,
                                                      m_omega=m_omega,
                                                      tau_timeout_wired=tau_timeout_wired,
                                                      theta_omega=theta_omega) / \
            Pt[
            k]
        computed_eta_ee[1][k] = compute_throughput_CL(K_i, K_j, K_z,
                                                      L_LOS=L_LOS_UMa_LTE,
                                                      thres_tx=thres_tx_LTE,
                                                      p_LOS=p_LOS_UMa,
                                                      epsilon=epsilon_LTE,
                                                      M_int=M_int, ra=ra,
                                                      rint=rint, lte=True,
                                                      l=l_UMa, tr=tr_UMa, d0=1,
                                                      Pt_norm=Pt_norm[k],
                                                      BW=BW,
                                                      M_MBS=M_MBS,
                                                      M_SBS=M_SBS,
                                                      N=N,
                                                      R_LTE=R_LTE,
                                                      m_omega=m_omega,
                                                      tau_timeout_wired=tau_timeout_wired,
                                                      theta_omega=theta_omega,
                                                      R_wifi=R_wifi) / \
            Pt[
            k]
        computed_eta_ee[2][k] = compute_throughput_CO(K_i, K_j, K_z,
                                                      L_LOS=L_LOS_UMa_5GNR_1,
                                                      p_LOS=p_LOS_UMa,
                                                      epsilon=epsilon_5GNR_1,
                                                      M_int=M_int, ra=ra,
                                                      rint=rint, l=l_UMa,
                                                      tr=tr_UMa, d0=1,
                                                      Pt_norm=Pt_norm[k],
                                                      BW=BW,
                                                      M_MBS=M_MBS,
                                                      M_SBS=M_SBS,
                                                      N=N,
                                                      R_LTE=R_LTE,
                                                      m_omega=m_omega,
                                                      tau_timeout_wired=tau_timeout_wired,
                                                      theta_omega=theta_omega,
                                                      R_5GNR=R_5GNR,
                                                      thres_tx_LTE=thres_tx_LTE,
                                                      thres_tx_5GNR=thres_tx_5GNR) / \
            Pt[
            k]
        computed_eta_ee[3][k] = compute_throughput_CO(K_i, K_j, K_z,
                                                      L_LOS=L_LOS_UMa_5GNR_2,
                                                      p_LOS=p_LOS_UMa,
                                                      epsilon=epsilon_5GNR_2,
                                                      M_int=M_int, ra=ra,
                                                      rint=rint, l=l_UMa,
                                                      tr=tr_UMa, d0=1,
                                                      Pt_norm=Pt_norm[k],
                                                      BW=BW,
                                                      M_MBS=M_MBS,
                                                      M_SBS=M_SBS,
                                                      N=N,
                                                      R_LTE=R_LTE,
                                                      m_omega=m_omega,
                                                      tau_timeout_wired=tau_timeout_wired,
                                                      theta_omega=theta_omega,
                                                      R_5GNR=R_5GNR,
                                                      thres_tx_5GNR=thres_tx_5GNR,
                                                      thres_tx_LTE=thres_tx_LTE
                                                      ) / Pt[k]

        print(datetime.datetime.now().strftime(
            dateFormat + "|" + timeFormat) + ": "
            + "\n\t Pt (dBm): {:n} ".format(Pt_dBm[k])
              + "\n\t EE (Mbps/Watts) - 802.11ac (Computed): \t{:.4n}".format(
            computed_eta_ee[0][k] / 1e6)
            + "\n\t EE (Mbps/Watts) - 802.11ac (Simulated): \t{:.4n}".format(
            simulated_eta_ee[0][k] / 1e6)
            + "\n\t EE (Mbps/Watts) - LTE (Computed): \t\t{:.4n}".format(
            computed_eta_ee[1][k] / 1e6)
            + "\n\t EE (Mbps/Watts) - LTE (Simulated): \t\t{:.4n}".format(
            simulated_eta_ee[1][k] / 1e6)
            + "\n\t EE (Mbps/Watts) - 5GNR-1 (Computed): \t\t{:.4n}".format(
            computed_eta_ee[2][k] / 1e6)
            + "\n\t EE (Mbps/Watts) - 5GNR-1 (Simulated): \t\t{:.4n}".format(
            simulated_eta_ee[2][k] / 1e6)
            + "\n\t EE (Mbps/Watts) - 5GNR-2 (Computed): \t\t{:.4n}".format(
            computed_eta_ee[3][k] / 1e6)
            + "\n\t EE (Mbps/Watts) - 5GNR-2 (Simulated): \t\t{:.4n}".format(
            simulated_eta_ee[3][k] / 1e6)
        )

    os.makedirs(name=os.path.dirname(p=file_name), exist_ok=True)
    logging.debug(msg=f'Dumping results on {file_name}.')
    spio.savemat(file_name,
                 {
                     'simulated_eta_ee': simulated_eta_ee,
                     'computed_eta_ee': computed_eta_ee,
                     'Pt_dBm': Pt_dBm
                 })


##
# This function generates results for eta_ee vs Pt_dBm in the InF-SL scenario
def eta_ee_vs_Pt_dBm_InFSL(file_name: str,
                           input_variables: Dict[str, List[float]]) -> None:
    """Save the results in the file_name"""
    K_size, BW, v, N, _, numSamples = process_raw_input(input_variables)
    fd_5GNR_2 = v * fc_5GNR_2 / (3 * 10 ** 8)
    epsilon_5GNR_2 = j0(2 * pi * fd_5GNR_2 * T)
    fd_5GNR_1 = v * fc_5GNR_1 / (3 * 10 ** 8)
    epsilon_5GNR_1 = j0(2 * pi * fd_5GNR_1 * T)
    R_wifi = 2.32e6 / BW  # 802.11 for MCS 0, taken from "Physical Layer
    thres_tx_wifi = 2 ** R_wifi - 1  # 802.11ac SINR threshold
    fd_wifi = v * fc_wifi / (3 * 10 ** 8)
    epsilon_wifi = j0(2 * pi * fd_wifi * T)
    fd_LTE = v * fc_LTE / (3 * 10 ** 8)
    epsilon_LTE = j0(2 * pi * fd_LTE * T)

    R_LTE = 4.08e6 / BW  # LTE, for MCS 6, taken from "Physical Layer
    # Evaluation
    # of V2X Communications Technologies: 5G NR-V2X, LTE-V2X, IEEE 802.11bd,
    # and IEEE 802.11p"
    thres_tx_LTE = 2 ** R_LTE - 1  # LTE SINR threshold
    eta = 20 ** ((-174 + 10 * log10(BW)) / 10) * 1e-3  # noise power
    # Pt_norm = Pt / (eta)  # Normalised transmit power
    theta_omega = c_wired * K_size  # Backhaul delay Gamma scale parameter
    tau_timeout_wired = m_omega * theta_omega  # Wired backhaul timeout

    R_5GNR = 4.63e6 / BW  # 5G NR, for MCS 6, taken from "Physical Layer
    # Evaluation of V2X Communications Technologies: 5G NR-V2X, LTE-V2X,
    # IEEE 802.11bd, and IEEE 802.11p"
    thres_tx_5GNR = 2 ** R_5GNR - 1  # 5G NR SINR threshold

    desc = datetime.datetime.now().strftime(
        dateFormat + "|" + timeFormat) + (": Simulating and computing eta_ee "
                                          "vs Pt_dBm in the InF-SL scenario")

    # Initialise variables
    Pt_dBm_min = 10
    Pt_dBm_max = 40  # tested
    Pt_dBm_step = 2
    Pt_dBm = np.arange(start=Pt_dBm_min, stop=Pt_dBm_max + Pt_dBm_step,
                       step=Pt_dBm_step)
    Pt = 10 ** (Pt_dBm / 10) * 1e-3
    Pt_norm = Pt / (eta)
    ra = 200
    rint = 2 * ra  # tested 2
    p_LOS_InFSL = exp(-(ra) / (-dclutter / ln(1 - LOS_r)))

    computed_eta_ee = np.zeros(shape=(4, len(Pt_norm)))
    simulated_eta_ee = np.zeros(shape=(4, len(Pt_norm)))

    for k in tqdm(range(len(Pt_norm)), desc=desc):
        # Simulate throughput
        simulated_eta_ee[0][k] = simulate_throughput_CL(K_i, K_j, K_z,
                                                        L_LOS=L_LOS_InFSL_wifi,
                                                        thres_tx=thres_tx_wifi,
                                                        p_LOS=p_LOS_InFSL,
                                                        epsilon=epsilon_wifi,
                                                        rint=rint,
                                                        M_int=M_int + 1, ra=ra,
                                                        lte=False,
                                                        l=l_InF_wifi,
                                                        Pt_norm=Pt_norm[k],
                                                        BW=BW,
                                                        M_MBS=M_MBS,
                                                        M_SBS=M_SBS,
                                                        N=N,
                                                        R_LTE=R_LTE,
                                                        R_wifi=R_wifi,
                                                        m_omega=m_omega,
                                                        numSamples=numSamples,
                                                        tau_timeout_wired=tau_timeout_wired,
                                                        theta_omega=theta_omega,
                                                        d0=d0) / \
            Pt[k]
        simulated_eta_ee[1][k] = simulate_throughput_CL(K_i, K_j, K_z,
                                                        L_LOS=L_LOS_InFSL_LTE,
                                                        thres_tx=thres_tx_LTE,
                                                        p_LOS=p_LOS_InFSL,
                                                        epsilon=epsilon_LTE,
                                                        rint=rint, M_int=M_int,
                                                        ra=ra, lte=True,
                                                        l=l_InF_LTE,
                                                        Pt_norm=Pt_norm[k],
                                                        BW=BW,
                                                        M_MBS=M_MBS,
                                                        M_SBS=M_SBS,
                                                        N=N,
                                                        R_LTE=R_LTE,
                                                        R_wifi=R_wifi,
                                                        m_omega=m_omega,
                                                        numSamples=numSamples,
                                                        tau_timeout_wired=tau_timeout_wired,
                                                        theta_omega=theta_omega,
                                                        d0=d0) / \
            Pt[k]
        simulated_eta_ee[2][k] = simulate_throughput_CO(K_i, K_j, K_z,
                                                        L_LOS=L_LOS_InFSL_5GNR_1,
                                                        p_LOS=p_LOS_InFSL,
                                                        epsilon=epsilon_5GNR_1,
                                                        rint=rint, M_int=M_int,
                                                        ra=ra, l=l_InF_5GNR1,
                                                        Pt_norm=Pt_norm[k],
                                                        BW=BW,
                                                        M_MBS=M_MBS,
                                                        M_SBS=M_SBS,
                                                        N=N,
                                                        R_LTE=R_LTE,
                                                        m_omega=m_omega,
                                                        numSamples=numSamples,
                                                        tau_timeout_wired=tau_timeout_wired,
                                                        theta_omega=theta_omega,
                                                        R_5GNR=R_5GNR,
                                                        thres_tx_LTE=thres_tx_LTE,
                                                        thres_tx_5GNR=thres_tx_5GNR,
                                                        d0=d0) / \
            Pt[k]
        simulated_eta_ee[3][k] = simulate_throughput_CO(K_i, K_j, K_z,
                                                        L_LOS=L_LOS_InFSL_5GNR_2,
                                                        p_LOS=p_LOS_InFSL,
                                                        epsilon=epsilon_5GNR_2,
                                                        rint=rint, M_int=M_int,
                                                        ra=ra, l=l_InF_5GNR2,
                                                        Pt_norm=Pt_norm[k],
                                                        BW=BW,
                                                        M_MBS=M_MBS,
                                                        M_SBS=M_SBS,
                                                        N=N,
                                                        R_LTE=R_LTE,
                                                        m_omega=m_omega,
                                                        numSamples=numSamples,
                                                        tau_timeout_wired=tau_timeout_wired,
                                                        theta_omega=theta_omega,
                                                        R_5GNR=R_5GNR,
                                                        thres_tx_LTE=thres_tx_LTE,
                                                        thres_tx_5GNR=thres_tx_5GNR,
                                                        d0=d0) / \
            Pt[k]

        # Compute throughput
        computed_eta_ee[0][k] = compute_throughput_CL(K_i, K_j, K_z,
                                                      L_LOS=L_LOS_InFSL_wifi,
                                                      thres_tx=thres_tx_wifi,
                                                      p_LOS=p_LOS_InFSL,
                                                      epsilon=epsilon_wifi,
                                                      rint=rint,
                                                      M_int=M_int + 1, ra=ra,
                                                      lte=False, l=l_InF_wifi,
                                                      tr=tr_InF,
                                                      Pt_norm=Pt_norm[k],
                                                      BW=BW,
                                                      M_MBS=M_MBS,
                                                      M_SBS=M_SBS,
                                                      N=N,
                                                      R_LTE=R_LTE,
                                                      m_omega=m_omega,
                                                      tau_timeout_wired=tau_timeout_wired,
                                                      theta_omega=theta_omega,
                                                      R_wifi=R_wifi, d0=d0) / \
            Pt[
            k]
        computed_eta_ee[1][k] = compute_throughput_CL(K_i, K_j, K_z,
                                                      L_LOS=L_LOS_InFSL_LTE,
                                                      thres_tx=thres_tx_LTE,
                                                      p_LOS=p_LOS_InFSL,
                                                      epsilon=epsilon_LTE,
                                                      rint=rint, M_int=M_int,
                                                      ra=ra, lte=True,
                                                      l=l_InF_LTE, tr=tr_InF,
                                                      Pt_norm=Pt_norm[k],
                                                      BW=BW,
                                                      M_MBS=M_MBS,
                                                      M_SBS=M_SBS,
                                                      N=N,
                                                      R_LTE=R_LTE,
                                                      m_omega=m_omega,
                                                      tau_timeout_wired=tau_timeout_wired,
                                                      theta_omega=theta_omega,
                                                      R_wifi=R_wifi, d0=d0) / \
            Pt[
            k]
        computed_eta_ee[2][k] = compute_throughput_CO(K_i, K_j, K_z,
                                                      L_LOS=L_LOS_InFSL_5GNR_1,
                                                      p_LOS=p_LOS_InFSL,
                                                      epsilon=epsilon_5GNR_1,
                                                      rint=rint, M_int=M_int,
                                                      ra=ra, l=l_InF_5GNR1,
                                                      tr=tr_InF,
                                                      Pt_norm=Pt_norm[k],
                                                      BW=BW,
                                                      M_MBS=M_MBS,
                                                      M_SBS=M_SBS,
                                                      N=N,
                                                      R_LTE=R_LTE,
                                                      m_omega=m_omega,
                                                      tau_timeout_wired=tau_timeout_wired,
                                                      theta_omega=theta_omega,
                                                      R_5GNR=R_5GNR,
                                                      thres_tx_5GNR=thres_tx_5GNR,
                                                      thres_tx_LTE=thres_tx_LTE,
                                                      d0=d0) / \
            Pt[
            k]
        computed_eta_ee[3][k] = compute_throughput_CO(K_i, K_j, K_z,
                                                      L_LOS=L_LOS_InFSL_5GNR_2,
                                                      p_LOS=p_LOS_InFSL,
                                                      epsilon=epsilon_5GNR_2,
                                                      rint=rint, M_int=M_int,
                                                      ra=ra, l=l_InF_5GNR2,
                                                      tr=tr_InF,
                                                      Pt_norm=Pt_norm[k],
                                                      BW=BW,
                                                      M_MBS=M_MBS,
                                                      M_SBS=M_SBS,
                                                      N=N,
                                                      R_LTE=R_LTE,
                                                      m_omega=m_omega,
                                                      tau_timeout_wired=tau_timeout_wired,
                                                      theta_omega=theta_omega,
                                                      R_5GNR=R_5GNR,
                                                      thres_tx_5GNR=thres_tx_5GNR,
                                                      thres_tx_LTE=thres_tx_LTE,
                                                      d0=d0) / \
            Pt[
            k]

        print(datetime.datetime.now().strftime(
            dateFormat + "|" + timeFormat) + ": "
            + "\n\t Pt (dBm): {:n} ".format(Pt_dBm[k])
              + "\n\t EE (Mbps) - 802.11ac (Computed): \t{:.4n}".format(
            computed_eta_ee[0][k] / 1e6)
            + "\n\t EE (Mbps) - 802.11ac (Simulated): \t{:.4n}".format(
            simulated_eta_ee[0][k] / 1e6)
            + "\n\t EE (Mbps) - LTE (Computed): \t\t{:.4n}".format(
            computed_eta_ee[1][k] / 1e6)
            + "\n\t EE (Mbps) - LTE (Simulated): \t\t{:.4n}".format(
            simulated_eta_ee[1][k] / 1e6)
            + "\n\t EE (Mbps) - 5GNR-1 (Computed): \t\t{:.4n}".format(
            computed_eta_ee[2][k] / 1e6)
            + "\n\t EE (Mbps) - 5GNR-1 (Simulated): \t\t{:.4n}".format(
            simulated_eta_ee[2][k] / 1e6)
            + "\n\t EE (Mbps) - 5GNR-2 (Computed): \t\t{:.4n}".format(
            computed_eta_ee[3][k] / 1e6)
            + "\n\t EE (Mbps) - 5GNR-2 (Simulated): \t\t{:.4n}".format(
            simulated_eta_ee[3][k] / 1e6)
        )

    os.makedirs(name=os.path.dirname(p=file_name), exist_ok=True)
    spio.savemat(file_name,
                 {
                     'simulated_eta_ee': simulated_eta_ee,
                     'computed_eta_ee': computed_eta_ee,
                     'Pt_dBm': Pt_dBm
                 })


##
# This function generates results for latency vs ra in the UMa scenario
def latency_vs_ra_UMa(file_name: str,
                      input_variables: Dict[str, List[float]]) -> None:
    K_size, BW, v, N, _, numSamples = process_raw_input(input_variables)
    fd_5GNR_2 = v * fc_5GNR_2 / (3 * 10 ** 8)
    epsilon_5GNR_2 = j0(2 * pi * fd_5GNR_2 * T)
    fd_5GNR_1 = v * fc_5GNR_1 / (3 * 10 ** 8)
    epsilon_5GNR_1 = j0(2 * pi * fd_5GNR_1 * T)
    R_wifi = 2.32e6 / BW  # 802.11 for MCS 0, taken from "Physical Layer
    thres_tx_wifi = 2 ** R_wifi - 1  # 802.11ac SINR threshold
    fd_wifi = v * fc_wifi / (3 * 10 ** 8)
    epsilon_wifi = j0(2 * pi * fd_wifi * T)
    fd_LTE = v * fc_LTE / (3 * 10 ** 8)
    epsilon_LTE = j0(2 * pi * fd_LTE * T)

    R_LTE = 4.08e6 / BW  # LTE, for MCS 6, taken from "Physical Layer
    # Evaluation
    # of V2X Communications Technologies: 5G NR-V2X, LTE-V2X, IEEE 802.11bd,
    # and IEEE 802.11p"
    thres_tx_LTE = 2 ** R_LTE - 1  # LTE SINR threshold
    eta = 20 ** ((-174 + 10 * log10(BW)) / 10) * 1e-3  # noise power
    Pt_norm = Pt / (eta)  # Normalised transmit power
    theta_omega = c_wired * K_size  # Backhaul delay Gamma scale parameter
    tau_timeout_wired = m_omega * theta_omega  # Wired backhaul timeout

    R_5GNR = 4.63e6 / BW  # 5G NR, for MCS 6, taken from "Physical Layer
    # Evaluation of V2X Communications Technologies: 5G NR-V2X, LTE-V2X,
    # IEEE 802.11bd, and IEEE 802.11p"
    thres_tx_5GNR = 2 ** R_5GNR - 1  # 5G NR SINR threshold

    desc = datetime.datetime.now().strftime(
        dateFormat + "|" + timeFormat) + (": Simulating and computing latency"
                                          " vs ra in the UMa scenario")

    # Initialise variables
    ra_min = 20
    ra_max = 400
    ra_step = 20
    ra = np.arange(start=ra_min, stop=ra_max + ra_step, step=ra_step)
    rint = 2 * ra  # tested 2
    p_LOS_UMa = [18 / ra[i] + (1 - 18 / ra[i]) * exp(-(ra[i]) / 63) for i in
                 range(len(ra))]

    computed_latency = np.zeros(shape=(4, len(ra)))
    simulated_latency = np.zeros(shape=(4, len(ra)))

    for k in tqdm(range(len(ra)), desc=desc):
        # Simulate throughput
        simulated_latency[0][k] = K_size / simulate_throughput_CL(K_i, K_j,
                                                                  K_z,
                                                                  L_LOS=L_LOS_UMa_wifi,
                                                                  thres_tx=thres_tx_wifi,
                                                                  p_LOS=p_LOS_UMa[k],
                                                                  epsilon=epsilon_wifi,
                                                                  M_int=M_int + 1,
                                                                  ra=ra[k],
                                                                  rint=rint[k],
                                                                  lte=False,
                                                                  l=l_UMa,
                                                                  d0=1,
                                                                  BW=BW,
                                                                  M_MBS=M_MBS,
                                                                  M_SBS=M_SBS,
                                                                  N=N,
                                                                  Pt_norm=Pt_norm,
                                                                  R_LTE=R_LTE,
                                                                  R_wifi=R_wifi,
                                                                  m_omega=m_omega,
                                                                  numSamples=numSamples,
                                                                  tau_timeout_wired=tau_timeout_wired,
                                                                  theta_omega=theta_omega)
        simulated_latency[1][k] = K_size / simulate_throughput_CL(K_i, K_j,
                                                                  K_z,
                                                                  L_LOS=L_LOS_UMa_LTE,
                                                                  thres_tx=thres_tx_LTE,
                                                                  p_LOS=p_LOS_UMa[k],
                                                                  epsilon=epsilon_LTE,
                                                                  M_int=M_int,
                                                                  ra=ra[k],
                                                                  rint=rint[k],
                                                                  lte=True,
                                                                  l=l_UMa,
                                                                  d0=1,
                                                                  BW=BW,
                                                                  M_MBS=M_MBS,
                                                                  M_SBS=M_SBS,
                                                                  N=N,
                                                                  Pt_norm=Pt_norm,
                                                                  R_LTE=R_LTE,
                                                                  R_wifi=R_wifi,
                                                                  m_omega=m_omega,
                                                                  numSamples=numSamples,
                                                                  tau_timeout_wired=tau_timeout_wired,
                                                                  theta_omega=theta_omega)
        simulated_latency[2][k] = K_size / simulate_throughput_CO(K_i, K_j,
                                                                  K_z,
                                                                  L_LOS=L_LOS_UMa_5GNR_1,
                                                                  p_LOS=p_LOS_UMa[k],
                                                                  epsilon=epsilon_5GNR_1,
                                                                  M_int=M_int,
                                                                  ra=ra[k],
                                                                  rint=rint[k],
                                                                  l=l_UMa,
                                                                  d0=1,
                                                                  BW=BW,
                                                                  M_MBS=M_MBS,
                                                                  M_SBS=M_SBS,
                                                                  N=N,
                                                                  Pt_norm=Pt_norm,
                                                                  R_LTE=R_LTE,
                                                                  m_omega=m_omega,
                                                                  numSamples=numSamples,
                                                                  tau_timeout_wired=tau_timeout_wired,
                                                                  theta_omega=theta_omega,
                                                                  R_5GNR=R_5GNR,
                                                                  thres_tx_5GNR=thres_tx_5GNR,
                                                                  thres_tx_LTE=thres_tx_LTE)
        simulated_latency[3][k] = K_size / simulate_throughput_CO(K_i, K_j,
                                                                  K_z,
                                                                  L_LOS=L_LOS_UMa_5GNR_2,
                                                                  p_LOS=p_LOS_UMa[k],
                                                                  epsilon=epsilon_5GNR_2,
                                                                  M_int=M_int,
                                                                  ra=ra[k],
                                                                  rint=rint[k],
                                                                  l=l_UMa,
                                                                  d0=1,
                                                                  BW=BW,
                                                                  M_MBS=M_MBS,
                                                                  M_SBS=M_SBS,
                                                                  N=N,
                                                                  Pt_norm=Pt_norm,
                                                                  R_LTE=R_LTE,
                                                                  m_omega=m_omega,
                                                                  numSamples=numSamples,
                                                                  tau_timeout_wired=tau_timeout_wired,
                                                                  theta_omega=theta_omega,
                                                                  R_5GNR=R_5GNR,
                                                                  thres_tx_5GNR=thres_tx_5GNR,
                                                                  thres_tx_LTE=thres_tx_LTE)

        # Compute throughput
        computed_latency[0][k] = K_size / compute_throughput_CL(K_i, K_j, K_z,
                                                                L_LOS=L_LOS_UMa_wifi,
                                                                thres_tx=thres_tx_wifi,
                                                                p_LOS=p_LOS_UMa[
                                                                    k],
                                                                epsilon=epsilon_wifi,
                                                                M_int=M_int
                                                                + 1,
                                                                ra=ra[k],
                                                                rint=rint[k],
                                                                lte=False,
                                                                l=l_UMa,
                                                                tr=tr_UMa,
                                                                d0=1,
                                                                BW=BW,
                                                                M_MBS=M_MBS,
                                                                M_SBS=M_SBS,
                                                                N=N,
                                                                Pt_norm=Pt_norm,
                                                                R_LTE=R_LTE,
                                                                m_omega=m_omega,
                                                                tau_timeout_wired=tau_timeout_wired,
                                                                theta_omega=theta_omega,
                                                                R_wifi=R_wifi)
        computed_latency[1][k] = K_size / compute_throughput_CL(K_i, K_j, K_z,
                                                                L_LOS=L_LOS_UMa_LTE,
                                                                thres_tx=thres_tx_LTE,
                                                                p_LOS=p_LOS_UMa[
                                                                    k],
                                                                epsilon=epsilon_LTE,
                                                                M_int=M_int,
                                                                ra=ra[k],
                                                                rint=rint[k],
                                                                lte=True,
                                                                l=l_UMa,
                                                                tr=tr_UMa,
                                                                d0=1,
                                                                BW=BW,
                                                                M_MBS=M_MBS,
                                                                M_SBS=M_SBS,
                                                                N=N,
                                                                Pt_norm=Pt_norm,
                                                                R_LTE=R_LTE,
                                                                m_omega=m_omega,
                                                                tau_timeout_wired=tau_timeout_wired,
                                                                theta_omega=theta_omega,
                                                                R_wifi=R_wifi)
        computed_latency[2][k] = K_size / compute_throughput_CO(K_i, K_j, K_z,
                                                                L_LOS=L_LOS_UMa_5GNR_1,
                                                                p_LOS=p_LOS_UMa[
                                                                    k],
                                                                epsilon=epsilon_5GNR_1,
                                                                M_int=M_int,
                                                                ra=ra[k],
                                                                rint=rint[k],
                                                                l=l_UMa,
                                                                tr=tr_UMa,
                                                                d0=1,
                                                                BW=BW,
                                                                M_MBS=M_MBS,
                                                                M_SBS=M_SBS,
                                                                N=N,
                                                                Pt_norm=Pt_norm,
                                                                R_LTE=R_LTE,
                                                                m_omega=m_omega,
                                                                tau_timeout_wired=tau_timeout_wired,
                                                                theta_omega=theta_omega,
                                                                R_5GNR=R_5GNR,
                                                                thres_tx_5GNR=thres_tx_5GNR,
                                                                thres_tx_LTE=thres_tx_LTE
                                                                )
        computed_latency[3][k] = K_size / compute_throughput_CO(K_i, K_j, K_z,
                                                                L_LOS=L_LOS_UMa_5GNR_2,
                                                                p_LOS=p_LOS_UMa[
                                                                    k],
                                                                epsilon=epsilon_5GNR_2,
                                                                M_int=M_int,
                                                                ra=ra[k],
                                                                rint=rint[k],
                                                                l=l_UMa,
                                                                tr=tr_UMa,
                                                                d0=1,
                                                                BW=BW,
                                                                M_MBS=M_MBS,
                                                                M_SBS=M_SBS,
                                                                N=N,
                                                                Pt_norm=Pt_norm,
                                                                R_LTE=R_LTE,
                                                                m_omega=m_omega,
                                                                tau_timeout_wired=tau_timeout_wired,
                                                                theta_omega=theta_omega,
                                                                R_5GNR=R_5GNR,
                                                                thres_tx_5GNR=thres_tx_5GNR,
                                                                thres_tx_LTE=thres_tx_LTE)

        print(datetime.datetime.now().strftime(
            dateFormat + "|" + timeFormat) + ": "
            + "\n\t ra (m): {:n} ".format(ra[k])
              + "\n\t Latency (ms) - 802.11ac (Computed): \t{:.4n}".format(
            computed_latency[0][k] * 1e3)
            + "\n\t Latency (ms) - 802.11ac (Simulated): \t{:.4n}".format(
            simulated_latency[0][k] * 1e3)
            + "\n\t Latency (ms) - LTE (Computed): \t\t{:.4n}".format(
            computed_latency[1][k] * 1e3)
            + "\n\t Latency (ms) - LTE (Simulated): \t\t{:.4n}".format(
            simulated_latency[1][k] * 1e3)
            + "\n\t Latency (ms) - 5GNR-1 (Computed): \t\t{:.4n}".format(
            computed_latency[2][k] * 1e3)
            + "\n\t Latency (ms) - 5GNR-1 (Simulated): \t{:.4n}".format(
            simulated_latency[2][k] * 1e3)
            + "\n\t Latency (ms) - 5GNR-2 (Computed): \t\t{:.4n}".format(
            computed_latency[3][k] * 1e3)
            + "\n\t Latency (ms) - 5GNR-2 (Simulated): \t{:.4n}".format(
            simulated_latency[3][k] * 1e3)
        )
    os.makedirs(name=os.path.dirname(file_name), exist_ok=True)
    spio.savemat(file_name,
                 {
                     'simulated_latency': simulated_latency,
                     'computed_latency': computed_latency,
                     'ra': ra
                 })


##
# This function generates results for latency vs ra in the InF-SL scenario
def latency_vs_ra_InFSL(file_name: str,
                        input_variables: Dict[str, List[float]]) -> None:
    K_size, BW, v, N, _, numSamples = process_raw_input(input_variables)
    fd_5GNR_2 = v * fc_5GNR_2 / (3 * 10 ** 8)
    epsilon_5GNR_2 = j0(2 * pi * fd_5GNR_2 * T)
    fd_5GNR_1 = v * fc_5GNR_1 / (3 * 10 ** 8)
    epsilon_5GNR_1 = j0(2 * pi * fd_5GNR_1 * T)
    R_wifi = 2.32e6 / BW  # 802.11 for MCS 0, taken from "Physical Layer
    thres_tx_wifi = 2 ** R_wifi - 1  # 802.11ac SINR threshold
    fd_wifi = v * fc_wifi / (3 * 10 ** 8)
    epsilon_wifi = j0(2 * pi * fd_wifi * T)
    fd_LTE = v * fc_LTE / (3 * 10 ** 8)
    epsilon_LTE = j0(2 * pi * fd_LTE * T)

    R_LTE = 4.08e6 / BW  # LTE, for MCS 6, taken from "Physical Layer
    # Evaluation
    # of V2X Communications Technologies: 5G NR-V2X, LTE-V2X, IEEE 802.11bd,
    # and IEEE 802.11p"
    thres_tx_LTE = 2 ** R_LTE - 1  # LTE SINR threshold
    eta = 20 ** ((-174 + 10 * log10(BW)) / 10) * 1e-3  # noise power
    Pt_norm = Pt / (eta)  # Normalised transmit power
    theta_omega = c_wired * K_size  # Backhaul delay Gamma scale parameter
    tau_timeout_wired = m_omega * theta_omega  # Wired backhaul timeout

    R_5GNR = 4.63e6 / BW  # 5G NR, for MCS 6, taken from "Physical Layer
    # Evaluation of V2X Communications Technologies: 5G NR-V2X, LTE-V2X,
    # IEEE 802.11bd, and IEEE 802.11p"
    thres_tx_5GNR = 2 ** R_5GNR - 1  # 5G NR SINR threshold

    desc = datetime.datetime.now().strftime(
        dateFormat + "|" + timeFormat) + (": Simulating and computing latency "
                                          "vs ra in the InF-SL scenario")

    # Initialise variables
    ra_min = 20
    ra_max = 200
    ra_step = 20
    ra = np.arange(start=ra_min, stop=ra_max + ra_step, step=ra_step)
    rint = 2 * ra  # tested 2
    p_LOS_InFSL = [exp(-(ra[i]) / (-dclutter / ln(1 - LOS_r))) for i in
                   range(len(ra))]

    computed_latency = np.zeros(shape=(4, len(ra)))
    simulated_latency = np.zeros(shape=(4, len(ra)))

    for k in tqdm(range(len(ra)), desc=desc):
        # Simulate throughput
        simulated_latency[0][k] = K_size / simulate_throughput_CL(K_i, K_j,
                                                                  K_z,
                                                                  L_LOS=L_LOS_InFSL_wifi,
                                                                  thres_tx=thres_tx_wifi,
                                                                  p_LOS=p_LOS_InFSL[
                                                                      k],
                                                                  epsilon=epsilon_wifi,
                                                                  rint=rint[k],
                                                                  M_int=M_int + 1,
                                                                  ra=ra[k],
                                                                  lte=False,
                                                                  l=l_InF_wifi,
                                                                  BW=BW,
                                                                  M_MBS=M_MBS,
                                                                  M_SBS=M_SBS,
                                                                  N=N,
                                                                  Pt_norm=Pt_norm,
                                                                  R_LTE=R_LTE,
                                                                  R_wifi=R_wifi,
                                                                  d0=d0,
                                                                  m_omega=m_omega,
                                                                  numSamples=numSamples,
                                                                  tau_timeout_wired=tau_timeout_wired,
                                                                  theta_omega=theta_omega
                                                                  )
        simulated_latency[1][k] = K_size / simulate_throughput_CL(K_i, K_j,
                                                                  K_z,
                                                                  L_LOS=L_LOS_InFSL_LTE,
                                                                  thres_tx=thres_tx_LTE,
                                                                  p_LOS=p_LOS_InFSL[
                                                                      k],
                                                                  epsilon=epsilon_LTE,
                                                                  rint=rint[k],
                                                                  M_int=M_int,
                                                                  ra=ra[k],
                                                                  lte=True,
                                                                  l=l_InF_LTE,
                                                                  BW=BW,
                                                                  M_MBS=M_MBS,
                                                                  M_SBS=M_SBS,
                                                                  N=N,
                                                                  Pt_norm=Pt_norm,
                                                                  R_LTE=R_LTE,
                                                                  R_wifi=R_wifi,
                                                                  d0=d0,
                                                                  m_omega=m_omega,
                                                                  numSamples=numSamples,
                                                                  tau_timeout_wired=tau_timeout_wired,
                                                                  theta_omega=theta_omega)
        simulated_latency[2][k] = K_size / simulate_throughput_CO(K_i, K_j,
                                                                  K_z,
                                                                  L_LOS=L_LOS_InFSL_5GNR_1,
                                                                  p_LOS=p_LOS_InFSL[
                                                                      k],
                                                                  epsilon=epsilon_5GNR_1,
                                                                  rint=rint[k],
                                                                  M_int=M_int,
                                                                  ra=ra[k],
                                                                  l=l_InF_5GNR1,
                                                                  BW=BW,
                                                                  M_MBS=M_MBS,
                                                                  M_SBS=M_SBS,
                                                                  Pt_norm=Pt_norm,
                                                                  N=N,
                                                                  R_5GNR=R_5GNR,
                                                                  R_LTE=R_LTE,
                                                                  d0=d0,
                                                                  m_omega=m_omega,
                                                                  numSamples=numSamples,
                                                                  tau_timeout_wired=tau_timeout_wired,
                                                                  theta_omega=theta_omega,
                                                                  thres_tx_5GNR=thres_tx_5GNR,
                                                                  thres_tx_LTE=thres_tx_LTE
                                                                  )
        simulated_latency[3][k] = K_size / simulate_throughput_CO(K_i, K_j,
                                                                  K_z,
                                                                  L_LOS=L_LOS_InFSL_5GNR_2,
                                                                  p_LOS=p_LOS_InFSL[
                                                                      k],
                                                                  epsilon=epsilon_5GNR_2,
                                                                  rint=rint[k],
                                                                  M_int=M_int,
                                                                  ra=ra[k],
                                                                  l=l_InF_5GNR2,
                                                                  BW=BW,
                                                                  M_MBS=M_MBS,
                                                                  M_SBS=M_SBS,
                                                                  Pt_norm=Pt_norm,
                                                                  N=N,
                                                                  R_5GNR=R_5GNR,
                                                                  R_LTE=R_LTE,
                                                                  d0=d0,
                                                                  m_omega=m_omega,
                                                                  numSamples=numSamples,
                                                                  tau_timeout_wired=tau_timeout_wired,
                                                                  theta_omega=theta_omega,
                                                                  thres_tx_5GNR=thres_tx_5GNR,
                                                                  thres_tx_LTE=thres_tx_LTE)

        # Compute throughput
        computed_latency[0][k] = K_size / compute_throughput_CL(K_i, K_j, K_z,
                                                                L_LOS=L_LOS_InFSL_wifi,
                                                                thres_tx=thres_tx_wifi,
                                                                p_LOS=p_LOS_InFSL[k],
                                                                epsilon=epsilon_wifi,
                                                                rint=rint[k],
                                                                M_int=M_int
                                                                + 1,
                                                                ra=ra[k],
                                                                lte=False,
                                                                l=l_InF_wifi,
                                                                tr=tr_InF,
                                                                BW=BW,
                                                                M_MBS=M_MBS,
                                                                M_SBS=M_SBS,
                                                                Pt_norm=Pt_norm,
                                                                N=N,
                                                                R_LTE=R_LTE,
                                                                d0=d0,
                                                                m_omega=m_omega,
                                                                tau_timeout_wired=tau_timeout_wired,
                                                                theta_omega=theta_omega,
                                                                R_wifi=R_wifi)
        computed_latency[1][k] = K_size / compute_throughput_CL(K_i, K_j, K_z,
                                                                L_LOS=L_LOS_InFSL_LTE,
                                                                thres_tx=thres_tx_LTE,
                                                                p_LOS=p_LOS_InFSL[k],
                                                                epsilon=epsilon_LTE,
                                                                rint=rint[k],
                                                                M_int=M_int,
                                                                ra=ra[k],
                                                                lte=True,
                                                                l=l_InF_LTE,
                                                                tr=tr_InF,
                                                                BW=BW,
                                                                M_MBS=M_MBS,
                                                                M_SBS=M_SBS,
                                                                Pt_norm=Pt_norm,
                                                                N=N,
                                                                R_LTE=R_LTE,
                                                                d0=d0,
                                                                m_omega=m_omega,
                                                                tau_timeout_wired=tau_timeout_wired,
                                                                theta_omega=theta_omega,
                                                                R_wifi=R_wifi)
        computed_latency[2][k] = K_size / compute_throughput_CO(K_i, K_j, K_z,
                                                                L_LOS=L_LOS_InFSL_5GNR_1,
                                                                p_LOS=p_LOS_InFSL[k],
                                                                epsilon=epsilon_5GNR_1,
                                                                rint=rint[k],
                                                                M_int=M_int,
                                                                ra=ra[k],
                                                                l=l_InF_5GNR1,
                                                                tr=tr_InF,
                                                                BW=BW,
                                                                M_MBS=M_MBS,
                                                                M_SBS=M_SBS,
                                                                Pt_norm=Pt_norm,
                                                                N=N,
                                                                R_LTE=R_LTE,
                                                                d0=d0,
                                                                m_omega=m_omega,
                                                                tau_timeout_wired=tau_timeout_wired,
                                                                theta_omega=theta_omega,
                                                                R_5GNR=R_5GNR,
                                                                thres_tx_5GNR=thres_tx_5GNR,
                                                                thres_tx_LTE=thres_tx_LTE)
        computed_latency[3][k] = K_size / compute_throughput_CO(K_i, K_j, K_z,
                                                                L_LOS=L_LOS_InFSL_5GNR_2,
                                                                p_LOS=p_LOS_InFSL[k],
                                                                epsilon=epsilon_5GNR_2,
                                                                rint=rint[k],
                                                                M_int=M_int,
                                                                ra=ra[k],
                                                                l=l_InF_5GNR2,
                                                                tr=tr_InF,
                                                                BW=BW,
                                                                M_MBS=M_MBS,
                                                                M_SBS=M_SBS,
                                                                Pt_norm=Pt_norm,
                                                                N=N,
                                                                R_LTE=R_LTE,
                                                                d0=d0,
                                                                m_omega=m_omega,
                                                                tau_timeout_wired=tau_timeout_wired,
                                                                theta_omega=theta_omega,
                                                                R_5GNR=R_5GNR,
                                                                thres_tx_5GNR=thres_tx_5GNR,
                                                                thres_tx_LTE=thres_tx_LTE)

        print(datetime.datetime.now().strftime(
            dateFormat + "|" + timeFormat) + ": "
            + "\n\t ra (m): {:n} ".format(ra[k])
              + "\n\t Latency (ms) - 802.11ac (Computed): \t{:.4n}".format(
            computed_latency[0][k] * 1e3)
            + "\n\t Latency (ms) - 802.11ac (Simulated): \t{:.4n}".format(
            simulated_latency[0][k] * 1e3)
            + "\n\t Latency (ms) - LTE (Computed): \t\t{:.4n}".format(
            computed_latency[1][k] * 1e3)
            + "\n\t Latency (ms) - LTE (Simulated): \t\t{:.4n}".format(
            simulated_latency[1][k] * 1e3)
            + "\n\t Latency (ms) - 5GNR-1 (Computed): \t\t{:.4n}".format(
            computed_latency[2][k] * 1e3)
            + "\n\t Latency (ms) - 5GNR-1 (Simulated): \t{:.4n}".format(
            simulated_latency[2][k] * 1e3)
            + "\n\t Latency (ms) - 5GNR-2 (Computed): \t\t{:.4n}".format(
            computed_latency[3][k] * 1e3)
            + "\n\t Latency (ms) - 5GNR-2 (Simulated): \t{:.4n}".format(
            simulated_latency[3][k] * 1e3)
        )
    os.makedirs(name=os.path.dirname(file_name), exist_ok=True)
    spio.savemat(file_name,
                 {
                     'simulated_latency': simulated_latency,
                     'computed_latency': computed_latency,
                     'ra': ra
                 })


##
# This function generates results for latency vs K_size in the UMa scenario
def latency_vs_K_size_UMa(file_name: str,
                          input_variables: Dict[str, List[float]]) -> None:
    K_size, BW, v, N, _, numSamples = process_raw_input(input_variables)
    fd_5GNR_2 = v * fc_5GNR_2 / (3 * 10 ** 8)
    epsilon_5GNR_2 = j0(2 * pi * fd_5GNR_2 * T)
    fd_5GNR_1 = v * fc_5GNR_1 / (3 * 10 ** 8)
    epsilon_5GNR_1 = j0(2 * pi * fd_5GNR_1 * T)
    R_wifi = 2.32e6 / BW  # 802.11 for MCS 0, taken from "Physical Layer
    thres_tx_wifi = 2 ** R_wifi - 1  # 802.11ac SINR threshold
    fd_wifi = v * fc_wifi / (3 * 10 ** 8)
    epsilon_wifi = j0(2 * pi * fd_wifi * T)
    fd_LTE = v * fc_LTE / (3 * 10 ** 8)
    epsilon_LTE = j0(2 * pi * fd_LTE * T)

    R_LTE = 4.08e6 / BW  # LTE, for MCS 6, taken from "Physical Layer
    # Evaluation
    # of V2X Communications Technologies: 5G NR-V2X, LTE-V2X, IEEE 802.11bd,
    # and IEEE 802.11p"
    thres_tx_LTE = 2 ** R_LTE - 1  # LTE SINR threshold
    eta = 20 ** ((-174 + 10 * log10(BW)) / 10) * 1e-3  # noise power
    Pt_norm = Pt / (eta)  # Normalised transmit power
    theta_omega = c_wired * K_size  # Backhaul delay Gamma scale parameter
    tau_timeout_wired = m_omega * theta_omega  # Wired backhaul timeout

    R_5GNR = 4.63e6 / BW  # 5G NR, for MCS 6, taken from "Physical Layer
    # Evaluation of V2X Communications Technologies: 5G NR-V2X, LTE-V2X,
    # IEEE 802.11bd, and IEEE 802.11p"
    thres_tx_5GNR = 2 ** R_5GNR - 1  # 5G NR SINR threshold

    desc = datetime.datetime.now().strftime(
        dateFormat + "|" + timeFormat) + (": Simulating and computing latency "
                                          "vs K_size in the UMa scenario")

    # Initialise variables
    K_size_min = 1e3 * 8
    K_size_max = 10e3 * 8
    K_size_step = 1e3 * 8
    K_size = np.arange(start=K_size_min, stop=K_size_max + K_size_step,
                       step=K_size_step)
    theta_omega = c_wired * K_size
    tau_timeout_wired = m_omega * theta_omega
    ra = 200
    rint = 2 * ra
    p_LOS_UMa = 18 / ra + (1 - 18 / ra) * exp(-(ra) / 63)

    computed_latency = np.zeros(shape=(4, len(K_size)))
    simulated_latency = np.zeros(shape=(4, len(K_size)))

    for k in tqdm(range(len(K_size)), desc=desc):
        # Simulate throughput
        simulated_latency[0][k] = K_size[k] / simulate_throughput_CL(K_i, K_j,
                                                                     K_z,
                                                                     L_LOS=L_LOS_UMa_wifi,
                                                                     thres_tx=thres_tx_wifi,
                                                                     p_LOS=p_LOS_UMa,
                                                                     epsilon=epsilon_wifi,
                                                                     M_int=M_int + 1,
                                                                     theta_omega=theta_omega[
                                                                         k],
                                                                     tau_timeout_wired=tau_timeout_wired[
                                                                         k],
                                                                     ra=ra,
                                                                     rint=rint,
                                                                     lte=False,
                                                                     l=l_UMa,
                                                                     d0=1,
                                                                     BW=BW,
                                                                     M_MBS=M_MBS,
                                                                     M_SBS=M_SBS,
                                                                     N=N,
                                                                     Pt_norm=Pt_norm,
                                                                     R_LTE=R_LTE,
                                                                     R_wifi=R_wifi,
                                                                     m_omega=m_omega,
                                                                     numSamples=numSamples)
        simulated_latency[1][k] = K_size[k] / simulate_throughput_CL(K_i, K_j,
                                                                     K_z,
                                                                     L_LOS=L_LOS_UMa_LTE,
                                                                     thres_tx=thres_tx_LTE,
                                                                     p_LOS=p_LOS_UMa,
                                                                     epsilon=epsilon_LTE,
                                                                     M_int=M_int,
                                                                     theta_omega=theta_omega[
                                                                         k],
                                                                     tau_timeout_wired=tau_timeout_wired[
                                                                         k],
                                                                     ra=ra,
                                                                     rint=rint,
                                                                     lte=True,
                                                                     l=l_UMa,
                                                                     d0=1,
                                                                     BW=BW,
                                                                     M_MBS=M_MBS,
                                                                     M_SBS=M_SBS,
                                                                     N=N,
                                                                     Pt_norm=Pt_norm,
                                                                     R_LTE=R_LTE,
                                                                     R_wifi=R_wifi,
                                                                     m_omega=m_omega,
                                                                     numSamples=numSamples)
        simulated_latency[2][k] = K_size[k] / simulate_throughput_CO(K_i, K_j,
                                                                     K_z,
                                                                     L_LOS=L_LOS_UMa_5GNR_1,
                                                                     p_LOS=p_LOS_UMa,
                                                                     epsilon=epsilon_5GNR_1,
                                                                     M_int=M_int,
                                                                     theta_omega=theta_omega[
                                                                         k],
                                                                     tau_timeout_wired=tau_timeout_wired[
                                                                         k],
                                                                     ra=ra,
                                                                     rint=rint,
                                                                     l=l_UMa,
                                                                     d0=1,
                                                                     BW=BW,
                                                                     R_5GNR=R_5GNR,
                                                                     R_LTE=R_LTE,
                                                                     M_MBS=M_MBS,
                                                                     M_SBS=M_SBS,
                                                                     N=N,
                                                                     Pt_norm=Pt_norm,
                                                                     m_omega=m_omega,
                                                                     numSamples=numSamples,
                                                                     thres_tx_5GNR=thres_tx_5GNR,
                                                                     thres_tx_LTE=thres_tx_LTE)
        simulated_latency[3][k] = K_size[k] / simulate_throughput_CO(K_i, K_j,
                                                                     K_z,
                                                                     L_LOS=L_LOS_UMa_5GNR_2,
                                                                     p_LOS=p_LOS_UMa,
                                                                     epsilon=epsilon_5GNR_2,
                                                                     M_int=M_int,
                                                                     theta_omega=theta_omega[
                                                                         k],
                                                                     tau_timeout_wired=tau_timeout_wired[
                                                                         k],
                                                                     ra=ra,
                                                                     rint=rint,
                                                                     l=l_UMa,
                                                                     d0=1,
                                                                     BW=BW,
                                                                     R_5GNR=R_5GNR,
                                                                     R_LTE=R_LTE,
                                                                     M_MBS=M_MBS,
                                                                     M_SBS=M_SBS,
                                                                     N=N,
                                                                     Pt_norm=Pt_norm,
                                                                     m_omega=m_omega,
                                                                     numSamples=numSamples,
                                                                     thres_tx_5GNR=thres_tx_5GNR,
                                                                     thres_tx_LTE=thres_tx_LTE)

        # Compute throughput
        computed_latency[0][k] = K_size[k] / compute_throughput_CL(K_i, K_j,
                                                                   K_z,
                                                                   L_LOS=L_LOS_UMa_wifi,
                                                                   thres_tx=thres_tx_wifi,
                                                                   p_LOS=p_LOS_UMa,
                                                                   epsilon=epsilon_wifi,
                                                                   M_int=M_int + 1,
                                                                   theta_omega=theta_omega[
                                                                       k],
                                                                   tau_timeout_wired=tau_timeout_wired[
                                                                       k],
                                                                   ra=ra,
                                                                   rint=rint,
                                                                   lte=False,
                                                                   l=l_UMa,
                                                                   tr=tr_UMa,
                                                                   d0=1,
                                                                   BW=BW,
                                                                   R_LTE=R_LTE,
                                                                   M_MBS=M_MBS,
                                                                   M_SBS=M_SBS,
                                                                   N=N,
                                                                   Pt_norm=Pt_norm,
                                                                   m_omega=m_omega,
                                                                   R_wifi=R_wifi)
        computed_latency[1][k] = K_size[k] / compute_throughput_CL(K_i, K_j,
                                                                   K_z,
                                                                   L_LOS=L_LOS_UMa_LTE,
                                                                   thres_tx=thres_tx_LTE,
                                                                   p_LOS=p_LOS_UMa,
                                                                   epsilon=epsilon_LTE,
                                                                   M_int=M_int,
                                                                   theta_omega=theta_omega[
                                                                       k],
                                                                   tau_timeout_wired=tau_timeout_wired[
                                                                       k],
                                                                   ra=ra,
                                                                   rint=rint,
                                                                   lte=True,
                                                                   l=l_UMa,
                                                                   tr=tr_UMa,
                                                                   d0=1,
                                                                   BW=BW,
                                                                   R_LTE=R_LTE,
                                                                   M_MBS=M_MBS,
                                                                   M_SBS=M_SBS,
                                                                   N=N,
                                                                   Pt_norm=Pt_norm,
                                                                   m_omega=m_omega,
                                                                   R_wifi=R_wifi)
        computed_latency[2][k] = K_size[k] / compute_throughput_CO(K_i, K_j,
                                                                   K_z,
                                                                   L_LOS=L_LOS_UMa_5GNR_1,
                                                                   p_LOS=p_LOS_UMa,
                                                                   epsilon=epsilon_5GNR_1,
                                                                   M_int=M_int,
                                                                   theta_omega=theta_omega[
                                                                       k],
                                                                   tau_timeout_wired=tau_timeout_wired[
                                                                       k],
                                                                   ra=ra,
                                                                   rint=rint,
                                                                   l=l_UMa,
                                                                   tr=tr_UMa,
                                                                   d0=1,
                                                                   BW=BW,
                                                                   R_LTE=R_LTE,
                                                                   M_MBS=M_MBS,
                                                                   M_SBS=M_SBS,
                                                                   N=N,
                                                                   Pt_norm=Pt_norm,
                                                                   m_omega=m_omega,
                                                                   R_5GNR=R_5GNR,
                                                                   thres_tx_5GNR=thres_tx_5GNR,
                                                                   thres_tx_LTE=thres_tx_LTE
                                                                   )
        computed_latency[3][k] = K_size[k] / compute_throughput_CO(K_i, K_j,
                                                                   K_z,
                                                                   L_LOS=L_LOS_UMa_5GNR_2,
                                                                   p_LOS=p_LOS_UMa,
                                                                   epsilon=epsilon_5GNR_2,
                                                                   M_int=M_int,
                                                                   theta_omega=theta_omega[
                                                                       k],
                                                                   tau_timeout_wired=tau_timeout_wired[
                                                                       k],
                                                                   ra=ra,
                                                                   rint=rint,
                                                                   l=l_UMa,
                                                                   tr=tr_UMa,
                                                                   d0=1,
                                                                   BW=BW,
                                                                   R_LTE=R_LTE,
                                                                   M_MBS=M_MBS,
                                                                   M_SBS=M_SBS,
                                                                   N=N,
                                                                   Pt_norm=Pt_norm,
                                                                   m_omega=m_omega,
                                                                   R_5GNR=R_5GNR,
                                                                   thres_tx_5GNR=thres_tx_5GNR,
                                                                   thres_tx_LTE=thres_tx_LTE)

        print(datetime.datetime.now().strftime(
            dateFormat + "|" + timeFormat) + ": "
            + "\n\t K_size (KB): {:n} ".format(K_size[k] / 8e3)
              + "\n\t Latency (ms) - 802.11ac (Computed): \t{:.4n}".format(
            computed_latency[0][k] * 1e3)
            + "\n\t Latency (ms) - 802.11ac (Simulated): \t{:.4n}".format(
            simulated_latency[0][k] * 1e3)
            + "\n\t Latency (ms) - LTE (Computed): \t\t{:.4n}".format(
            computed_latency[1][k] * 1e3)
            + "\n\t Latency (ms) - LTE (Simulated): \t\t{:.4n}".format(
            simulated_latency[1][k] * 1e3)
            + "\n\t Latency (ms) - 5GNR-1 (Computed): \t\t{:.4n}".format(
            computed_latency[2][k] * 1e3)
            + "\n\t Latency (ms) - 5GNR-1 (Simulated): \t{:.4n}".format(
            simulated_latency[2][k] * 1e3)
            + "\n\t Latency (ms) - 5GNR-2 (Computed): \t\t{:.4n}".format(
            computed_latency[3][k] * 1e3)
            + "\n\t Latency (ms) - 5GNR-2 (Simulated): \t{:.4n}".format(
            simulated_latency[3][k] * 1e3)
        )
    os.makedirs(name=os.path.dirname(file_name), exist_ok=True)
    spio.savemat(file_name,
                 {
                     'simulated_latency': simulated_latency,
                     'computed_latency': computed_latency,
                     'K_size': K_size
                 })


##
# This function generates results for latency vs K_size in the UMa scenario
def latency_vs_K_size_InFSL(file_name: str,
                            input_variables: Dict[str, List[float]]) -> None:
    K_size, BW, v, N, _, numSamples = process_raw_input(input_variables)
    fd_5GNR_2 = v * fc_5GNR_2 / (3 * 10 ** 8)
    epsilon_5GNR_2 = j0(2 * pi * fd_5GNR_2 * T)
    fd_5GNR_1 = v * fc_5GNR_1 / (3 * 10 ** 8)
    epsilon_5GNR_1 = j0(2 * pi * fd_5GNR_1 * T)
    R_wifi = 2.32e6 / BW  # 802.11 for MCS 0, taken from "Physical Layer
    thres_tx_wifi = 2 ** R_wifi - 1  # 802.11ac SINR threshold
    fd_wifi = v * fc_wifi / (3 * 10 ** 8)
    epsilon_wifi = j0(2 * pi * fd_wifi * T)
    fd_LTE = v * fc_LTE / (3 * 10 ** 8)
    epsilon_LTE = j0(2 * pi * fd_LTE * T)

    R_LTE = 4.08e6 / BW  # LTE, for MCS 6, taken from "Physical Layer
    # Evaluation
    # of V2X Communications Technologies: 5G NR-V2X, LTE-V2X, IEEE 802.11bd,
    # and IEEE 802.11p"
    thres_tx_LTE = 2 ** R_LTE - 1  # LTE SINR threshold
    eta = 20 ** ((-174 + 10 * log10(BW)) / 10) * 1e-3  # noise power
    Pt_norm = Pt / (eta)  # Normalised transmit power
    theta_omega = c_wired * K_size  # Backhaul delay Gamma scale parameter
    tau_timeout_wired = m_omega * theta_omega  # Wired backhaul timeout

    R_5GNR = 4.63e6 / BW  # 5G NR, for MCS 6, taken from "Physical Layer
    # Evaluation of V2X Communications Technologies: 5G NR-V2X, LTE-V2X,
    # IEEE 802.11bd, and IEEE 802.11p"
    thres_tx_5GNR = 2 ** R_5GNR - 1  # 5G NR SINR threshold

    desc = datetime.datetime.now().strftime(
        dateFormat + "|" + timeFormat) + (": Simulating and computing latency"
                                          " vs K_size in the InF-SL scenario")

    # Initialise variables
    K_size_min = 1e3 * 8
    K_size_max = 10e3 * 8
    K_size_step = 1e3 * 8
    K_size = np.arange(start=K_size_min, stop=K_size_max + K_size_step,
                       step=K_size_step)
    theta_omega = c_wired * K_size
    tau_timeout_wired = m_omega * theta_omega
    ra = 200
    rint = 2 * ra
    p_LOS_InFSL = exp(-(ra) / (-dclutter / ln(1 - LOS_r)))

    computed_latency = np.zeros(shape=(4, len(K_size)))
    simulated_latency = np.zeros(shape=(4, len(K_size)))

    for k in tqdm(range(len(K_size)), desc=desc):
        # Simulate throughput
        simulated_latency[0][k] = K_size[k] / simulate_throughput_CL(K_i, K_j,
                                                                     K_z,
                                                                     L_LOS=L_LOS_InFSL_wifi,
                                                                     thres_tx=thres_tx_wifi,
                                                                     p_LOS=p_LOS_InFSL,
                                                                     epsilon=epsilon_wifi,
                                                                     rint=rint,
                                                                     M_int=M_int + 1,
                                                                     theta_omega=theta_omega[
                                                                         k],
                                                                     tau_timeout_wired=tau_timeout_wired[
                                                                         k],
                                                                     ra=ra,
                                                                     lte=False,
                                                                     l=l_InF_wifi,
                                                                     BW=BW,
                                                                     M_MBS=M_MBS,
                                                                     M_SBS=M_SBS,
                                                                     N=N,
                                                                     Pt_norm=Pt_norm,
                                                                     R_LTE=R_LTE,
                                                                     R_wifi=R_wifi,
                                                                     d0=d0,
                                                                     m_omega=m_omega,
                                                                     numSamples=numSamples
                                                                     )
        simulated_latency[1][k] = K_size[k] / simulate_throughput_CL(K_i, K_j,
                                                                     K_z,
                                                                     L_LOS=L_LOS_InFSL_LTE,
                                                                     thres_tx=thres_tx_LTE,
                                                                     p_LOS=p_LOS_InFSL,
                                                                     epsilon=epsilon_LTE,
                                                                     rint=rint,
                                                                     M_int=M_int,
                                                                     theta_omega=theta_omega[
                                                                         k],
                                                                     tau_timeout_wired=tau_timeout_wired[
                                                                         k],
                                                                     ra=ra,
                                                                     lte=True,
                                                                     l=l_InF_LTE,
                                                                     BW=BW,
                                                                     M_MBS=M_MBS,
                                                                     M_SBS=M_SBS,
                                                                     N=N,
                                                                     Pt_norm=Pt_norm,
                                                                     R_LTE=R_LTE,
                                                                     R_wifi=R_wifi,
                                                                     d0=d0,
                                                                     m_omega=m_omega,
                                                                     numSamples=numSamples)
        simulated_latency[2][k] = K_size[k] / simulate_throughput_CO(K_i, K_j,
                                                                     K_z,
                                                                     L_LOS=L_LOS_InFSL_5GNR_1,
                                                                     p_LOS=p_LOS_InFSL,
                                                                     epsilon=epsilon_5GNR_1,
                                                                     rint=rint,
                                                                     M_int=M_int,
                                                                     theta_omega=theta_omega[
                                                                         k],
                                                                     tau_timeout_wired=tau_timeout_wired[
                                                                         k],
                                                                     ra=ra,
                                                                     l=l_InF_5GNR1,
                                                                     M_MBS=M_MBS,
                                                                     M_SBS=M_SBS,
                                                                     N=N,
                                                                     Pt_norm=Pt_norm,
                                                                     R_LTE=R_LTE,
                                                                     d0=d0,
                                                                     m_omega=m_omega,
                                                                     numSamples=numSamples,
                                                                     BW=BW,
                                                                     R_5GNR=R_5GNR,
                                                                     thres_tx_5GNR=thres_tx_5GNR,
                                                                     thres_tx_LTE=thres_tx_LTE
                                                                     )
        simulated_latency[3][k] = K_size[k] / simulate_throughput_CO(K_i, K_j,
                                                                     K_z,
                                                                     L_LOS=L_LOS_InFSL_5GNR_2,
                                                                     p_LOS=p_LOS_InFSL,
                                                                     epsilon=epsilon_5GNR_2,
                                                                     rint=rint,
                                                                     M_int=M_int,
                                                                     theta_omega=theta_omega[
                                                                         k],
                                                                     tau_timeout_wired=tau_timeout_wired[
                                                                         k],
                                                                     ra=ra,
                                                                     l=l_InF_5GNR2,
                                                                     M_MBS=M_MBS,
                                                                     M_SBS=M_SBS,
                                                                     N=N,
                                                                     Pt_norm=Pt_norm,
                                                                     R_LTE=R_LTE,
                                                                     d0=d0,
                                                                     m_omega=m_omega,
                                                                     numSamples=numSamples,
                                                                     BW=BW,
                                                                     R_5GNR=R_5GNR,
                                                                     thres_tx_5GNR=thres_tx_5GNR,
                                                                     thres_tx_LTE=thres_tx_LTE)

        # Compute throughput
        computed_latency[0][k] = K_size[k] / compute_throughput_CL(K_i, K_j,
                                                                   K_z,
                                                                   L_LOS=L_LOS_InFSL_wifi,
                                                                   thres_tx=thres_tx_wifi,
                                                                   p_LOS=p_LOS_InFSL,
                                                                   epsilon=epsilon_wifi,
                                                                   rint=rint,
                                                                   M_int=M_int + 1,
                                                                   theta_omega=theta_omega[
                                                                       k],
                                                                   tau_timeout_wired=tau_timeout_wired[
                                                                       k],
                                                                   ra=ra,
                                                                   lte=False,
                                                                   l=l_InF_wifi,
                                                                   tr=tr_InF,
                                                                   M_MBS=M_MBS,
                                                                   M_SBS=M_SBS,
                                                                   N=N,
                                                                   Pt_norm=Pt_norm,
                                                                   R_LTE=R_LTE,
                                                                   d0=d0,
                                                                   m_omega=m_omega,
                                                                   BW=BW,
                                                                   R_wifi=R_wifi
                                                                   )
        computed_latency[1][k] = K_size[k] / compute_throughput_CL(K_i, K_j,
                                                                   K_z,
                                                                   L_LOS=L_LOS_InFSL_LTE,
                                                                   thres_tx=thres_tx_LTE,
                                                                   p_LOS=p_LOS_InFSL,
                                                                   epsilon=epsilon_LTE,
                                                                   rint=rint,
                                                                   M_int=M_int,
                                                                   theta_omega=theta_omega[
                                                                       k],
                                                                   tau_timeout_wired=tau_timeout_wired[
                                                                       k],
                                                                   ra=ra,
                                                                   lte=True,
                                                                   l=l_InF_LTE,
                                                                   tr=tr_InF,
                                                                   M_MBS=M_MBS,
                                                                   M_SBS=M_SBS,
                                                                   N=N,
                                                                   Pt_norm=Pt_norm,
                                                                   R_LTE=R_LTE,
                                                                   d0=d0,
                                                                   m_omega=m_omega,
                                                                   BW=BW,
                                                                   R_wifi=R_wifi)
        computed_latency[2][k] = K_size[k] / compute_throughput_CO(K_i, K_j,
                                                                   K_z,
                                                                   L_LOS=L_LOS_InFSL_5GNR_1,
                                                                   p_LOS=p_LOS_InFSL,
                                                                   epsilon=epsilon_5GNR_1,
                                                                   rint=rint,
                                                                   M_int=M_int,
                                                                   theta_omega=theta_omega[
                                                                       k],
                                                                   tau_timeout_wired=tau_timeout_wired[
                                                                       k],
                                                                   ra=ra,
                                                                   l=l_InF_5GNR1,
                                                                   tr=tr_InF,
                                                                   M_MBS=M_MBS,
                                                                   M_SBS=M_SBS,
                                                                   N=N,
                                                                   Pt_norm=Pt_norm,
                                                                   R_LTE=R_LTE,
                                                                   d0=d0,
                                                                   m_omega=m_omega,
                                                                   BW=BW,
                                                                   R_5GNR=R_5GNR,
                                                                   thres_tx_5GNR=thres_tx_5GNR,
                                                                   thres_tx_LTE=thres_tx_LTE)
        computed_latency[3][k] = K_size[k] / compute_throughput_CO(K_i, K_j,
                                                                   K_z,
                                                                   L_LOS=L_LOS_InFSL_5GNR_2,
                                                                   p_LOS=p_LOS_InFSL,
                                                                   epsilon=epsilon_5GNR_2,
                                                                   rint=rint,
                                                                   M_int=M_int,
                                                                   theta_omega=theta_omega[
                                                                       k],
                                                                   tau_timeout_wired=tau_timeout_wired[
                                                                       k],
                                                                   ra=ra,
                                                                   l=l_InF_5GNR2,
                                                                   tr=tr_InF,
                                                                   M_MBS=M_MBS,
                                                                   M_SBS=M_SBS,
                                                                   N=N,
                                                                   Pt_norm=Pt_norm,
                                                                   R_LTE=R_LTE,
                                                                   d0=d0,
                                                                   m_omega=m_omega,
                                                                   BW=BW,
                                                                   R_5GNR=R_5GNR,
                                                                   thres_tx_5GNR=thres_tx_5GNR,
                                                                   thres_tx_LTE=thres_tx_LTE)

        print(datetime.datetime.now().strftime(
            dateFormat + "|" + timeFormat) + ": "
            + "\n\t K_size (KB): {:n} ".format(K_size[k] / 8e3)
              + "\n\t Latency (ms) - UMa-802.11ac (Computed): \t{:.4n}".format(
            computed_latency[0][k] * 1e3)
            + "\n\t Latency (ms) - UMa-802.11ac (Simulated): \t{"
            ":.4n}".format(
            simulated_latency[0][k] * 1e3)
            + "\n\t Latency (ms) - LTE (Computed): \t\t{:.4n}".format(
            computed_latency[1][k] * 1e3)
            + "\n\t Latency (ms) - LTE (Simulated): \t\t{:.4n}".format(
            simulated_latency[1][k] * 1e3)
            + "\n\t Latency (ms) - 5GNR-1 (Computed): \t\t{:.4n}".format(
            computed_latency[2][k] * 1e3)
            + "\n\t Latency (ms) - 5GNR-1 (Simulated): \t{:.4n}".format(
            simulated_latency[2][k] * 1e3)
            + "\n\t Latency (ms) - 5GNR-2 (Computed): \t\t{:.4n}".format(
            computed_latency[3][k] * 1e3)
            + "\n\t Latency (ms) - 5GNR-2 (Simulated): \t{:.4n}".format(
            simulated_latency[3][k] * 1e3)
        )
    os.makedirs(name=os.path.dirname(file_name), exist_ok=True)
    spio.savemat(file_name,
                 {
                     'simulated_latency': simulated_latency,
                     'computed_latency': computed_latency,
                     'K_size': K_size
                 })


##
# This function plots the results for gamma_x_CDF vs ra
def plot_gamma_x_CDF_vs_ra_UMa():
    # Load results
    numerical_results = spio.loadmat(
        r'numerical_results/gamma_x_CDF_vs_ra.mat')
    computed_CDF = numerical_results['computed_CDF']
    simulated_CDF = numerical_results['simulated_CDF']
    ra = numerical_results['ra'][0]

    # Plot results
    lines = plt.semilogy(
        ra, computed_CDF[0],
        ra, computed_CDF[1],
        ra, simulated_CDF[0], 'k.',
        ra, simulated_CDF[1], 'k.',
    )
    plt.setp(lines[0], 'marker', 's')
    plt.setp(lines[1], 'marker', '>')
    # plt.setp(lines[2], 'marker', 's', 'ls', '-.', 'c', 'r')
    # plt.setp(lines[3], 'marker', 'v', 'ls', '-.', 'c', lines[2]._color)
    plt.legend(
        (lines[0], lines[1], lines[2]),
        (
            'UMa',
            'InF',
            'Simulation'
        ), fontsize=8)

    plt.xlabel('$r_{a}$ (m)')
    plt.ylabel("CDF")
    # plt.ylim(1e-3,1e-1)
    # plt.ylim(0.7, 1.05)
    # plt.xlim(cache_size[0], cache_size[len(cache_size)-1])

    # ax = plt.gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, which="both")
    plt.savefig(fname=r"./figures/gamma_x_CDF_vs_ra.eps", format="eps")
    # plt.show()


##
# This function plots the results for throughput vs ra in the UMa scenario
def plot_throughput_vs_ra_UMa(in_file: str, out_file: str,
                              input_variables: Dict[str, List[float]]) -> None:
    # Load results
    if not os.path.isfile(path=in_file):
        throughput_vs_ra_UMa(file_name=in_file,
                             input_variables=input_variables)
    numerical_results = spio.loadmat(file_name=in_file)
    computed_throughput = numerical_results['computed_throughput'] / 1e6
    simulated_throughput = numerical_results['simulated_throughput'] / 1e6
    ra = numerical_results['ra'][0]

    # Plot results
    lines = plt.plot(
        ra, computed_throughput[0],
        ra, computed_throughput[1],
        ra, computed_throughput[2],
        ra, computed_throughput[3],
        ra, simulated_throughput[0], 'k.',
        ra, simulated_throughput[1], 'k.',
        ra, simulated_throughput[2], 'k.',
        ra, simulated_throughput[3], 'k.',
    )
    plt.setp(lines[0], 'marker', 's')
    plt.setp(lines[1], 'marker', '>')
    plt.setp(lines[2], 'marker', 'v')
    plt.setp(lines[3], 'marker', '*', 'c', lines[2]._color)
    # plt.setp(lines[2], 'marker', 's', 'ls', '-.', 'c', 'r')
    # plt.setp(lines[3], 'marker', 'v', 'ls', '-.', 'c', lines[2]._color)
    plt.legend(
        (lines[0], lines[1], lines[2], lines[3], lines[4]),
        (
            'IEEE 802.11ac (5.2 GHz)',
            'LTE (2.4 GHz)',
            '5G-NR with MEC (3.5GHz)',
            '5G-NR with MEC (700 MHz)',
            'Simulation'
        ), fontsize=8)

    plt.xlabel('$r_{a}$ (m)')
    plt.ylabel("Average Throughput (Mbps)")
    # plt.ylim(1e-3,1e-1)
    # plt.ylim(0.7, 1.05)
    # plt.xlim(cache_size[0], cache_size[len(cache_size)-1])

    # ax = plt.gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Urban Macro")
    plt.grid(True, which="both")
    plt.savefig(fname=out_file)
    # # plt.show()
    plt.close()


##
# This function plots the results for throughput vs ra in the InF-SL scenario
def plot_throughput_vs_ra_InFSL(in_file: str, out_file: str,
                                input_variables: Dict[
                                    str, List[float]]) -> None:
    if not os.path.isfile(path=in_file):
        throughput_vs_ra_InFSL(file_name=in_file,
                               input_variables=input_variables)
    numerical_results = spio.loadmat(file_name=in_file)
    computed_throughput = numerical_results['computed_throughput'] / 1e6
    simulated_throughput = numerical_results['simulated_throughput'] / 1e6
    ra = numerical_results['ra'][0]

    # Plot results
    lines = plt.plot(
        ra, computed_throughput[0],
        ra, computed_throughput[1],
        ra, computed_throughput[2],
        ra, computed_throughput[3],
        ra, simulated_throughput[0], 'k.',
        ra, simulated_throughput[1], 'k.',
        ra, simulated_throughput[2], 'k.',
        ra, simulated_throughput[3], 'k.',
    )
    plt.setp(lines[0], 'marker', 's')
    plt.setp(lines[1], 'marker', '>')
    plt.setp(lines[2], 'marker', 'v')
    plt.setp(lines[3], 'marker', '*', 'c', lines[2]._color)
    # plt.setp(lines[2], 'marker', 's', 'ls', '-.', 'c', 'r')
    # plt.setp(lines[3], 'marker', 'v', 'ls', '-.', 'c', lines[2]._color)
    plt.legend(
        (lines[0], lines[1], lines[2], lines[3], lines[4]),
        (
            'IEEE 802.11ac (5.2 GHz)',
            'LTE (2.4 GHz)',
            '5G-NR with MEC (3.5GHz)',
            '5G-NR with MEC (700 MHz)',
            'Simulation'
        ), fontsize=8)

    plt.xlabel('$r_{a}$ (m)')
    plt.ylabel(r"Average Throughput (Mbps)")
    # plt.ylim(1e-3,1e-1)
    # plt.ylim(0.7, 1.05)
    # plt.xlim(cache_size[0], cache_size[len(cache_size)-1])

    # ax = plt.gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Indoor Factory")
    plt.grid(True, which="both")
    plt.savefig(fname=out_file)
    # plt.show()
    plt.close()


##
# This function plots the results for eta_ee vs ra in the UMa scenario
def plot_eta_ee_vs_ra_UMa(in_file: str, out_file: str,
                          input_variables: Dict[str, List[float]]) -> None:
    if not os.path.isfile(path=in_file):
        eta_ee_vs_ra_UMa(file_name=in_file, input_variables=input_variables)
    numerical_results = spio.loadmat(file_name=in_file)
    computed_eta_ee = numerical_results['computed_eta_ee'] / 1e6
    simulated_eta_ee = numerical_results['simulated_eta_ee'] / 1e6
    ra = numerical_results['ra'][0]

    # Plot results
    lines = plt.plot(
        ra, computed_eta_ee[0],
        ra, computed_eta_ee[1],
        ra, computed_eta_ee[2],
        ra, computed_eta_ee[3],
        ra, simulated_eta_ee[0], 'k.',
        ra, simulated_eta_ee[1], 'k.',
        ra, simulated_eta_ee[2], 'k.',
        ra, simulated_eta_ee[3], 'k.',
    )
    plt.setp(lines[0], 'marker', 's')
    plt.setp(lines[1], 'marker', '>')
    plt.setp(lines[2], 'marker', 'v')
    plt.setp(lines[3], 'marker', '*', 'c', lines[2]._color)
    # plt.setp(lines[2], 'marker', 's', 'ls', '-.', 'c', 'r')
    # plt.setp(lines[3], 'marker', 'v', 'ls', '-.', 'c', lines[2]._color)
    plt.legend(
        (lines[0], lines[1], lines[2], lines[3], lines[4]),
        (
            'IEEE 802.11ac (5.2 GHz)',
            'LTE (2.4 GHz)',
            '5G-NR with MEC (3.5GHz)',
            '5G-NR with MEC (700 MHz)',
            'Simulation'
        ), fontsize=8)

    plt.xlabel('$r_{a}$ (m)')
    plt.ylabel("Average EE (Mbps/Watts)")
    # plt.ylim(1e-3,1e-1)
    # plt.ylim(0.7, 1.05)
    # plt.xlim(cache_size[0], cache_size[len(cache_size)-1])

    # ax = plt.gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Urban Macro")
    plt.grid(True, which="both")
    plt.savefig(fname=out_file)
    # plt.show()
    plt.close()


##
# This function plots the results for eta_ee vs ra in the InF-SL scenario
def plot_eta_ee_vs_ra_InFSL(in_file: str, out_file: str,
                            input_variables: Dict[str, List[float]]) -> None:
    if not os.path.isfile(path=in_file):
        eta_ee_vs_ra_InFSL(file_name=in_file, input_variables=input_variables)
    numerical_results = spio.loadmat(file_name=in_file)
    computed_eta_ee = numerical_results['computed_eta_ee'] / 1e6
    simulated_eta_ee = numerical_results['simulated_eta_ee'] / 1e6
    ra = numerical_results['ra'][0]

    # Plot results
    lines = plt.plot(
        ra, computed_eta_ee[0],
        ra, computed_eta_ee[1],
        ra, computed_eta_ee[2],
        ra, computed_eta_ee[3],
        ra, simulated_eta_ee[0], 'k.',
        ra, simulated_eta_ee[1], 'k.',
        ra, simulated_eta_ee[2], 'k.',
        ra, simulated_eta_ee[3], 'k.',
    )
    plt.setp(lines[0], 'marker', 's')
    plt.setp(lines[1], 'marker', '>')
    plt.setp(lines[2], 'marker', 'v')
    plt.setp(lines[3], 'marker', '*', 'c', lines[2]._color)
    # plt.setp(lines[2], 'marker', 's', 'ls', '-.', 'c', 'r')
    # plt.setp(lines[3], 'marker', 'v', 'ls', '-.', 'c', lines[2]._color)
    plt.legend(
        (lines[0], lines[1], lines[2], lines[3], lines[4]),
        (
            'IEEE 802.11ac (5.2 GHz)',
            'LTE (2.4 GHz)',
            '5G-NR with MEC (3.5GHz)',
            '5G-NR with MEC (700 MHz)',
            'Simulation'
        ), fontsize=8)

    plt.xlabel('$r_{a}$ (m)')
    plt.ylabel(r"Average EE (Mbps/Watts)")
    # plt.ylim(1e-3,1e-1)
    # plt.ylim(0.7, 1.05)
    # plt.xlim(cache_size[0], cache_size[len(cache_size)-1])

    # ax = plt.gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Indoor Factory")
    plt.grid(True, which="both")
    plt.savefig(fname=out_file)
    # plt.show()
    plt.close()


##
# This function plots the results for eta_ee vs Pt_dBm in the UMa scenario
def plot_eta_ee_vs_Pt_dBm_UMa(in_file: str, out_file: str,
                              input_variables: Dict[str, List[float]]) -> None:
    if not os.path.isfile(path=in_file):
        eta_ee_vs_Pt_dBm_UMa(file_name=in_file,
                             input_variables=input_variables)
    numerical_results = spio.loadmat(file_name=in_file)
    computed_eta_ee = numerical_results['computed_eta_ee'] / 1e6
    simulated_eta_ee = numerical_results['simulated_eta_ee'] / 1e6
    Pt_dBm = numerical_results['Pt_dBm'][0]

    # Plot results
    lines = plt.plot(
        Pt_dBm, computed_eta_ee[0],
        Pt_dBm, computed_eta_ee[1],
        Pt_dBm, computed_eta_ee[2],
        Pt_dBm, computed_eta_ee[3],
        Pt_dBm, simulated_eta_ee[0], 'k.',
        Pt_dBm, simulated_eta_ee[1], 'k.',
        Pt_dBm, simulated_eta_ee[2], 'k.',
        Pt_dBm, simulated_eta_ee[3], 'k.',
    )
    plt.setp(lines[0], 'marker', 's')
    plt.setp(lines[1], 'marker', '>')
    plt.setp(lines[2], 'marker', 'v')
    plt.setp(lines[3], 'marker', '*', 'c', lines[2]._color)
    # plt.setp(lines[2], 'marker', 's', 'ls', '-.', 'c', 'r')
    # plt.setp(lines[3], 'marker', 'v', 'ls', '-.', 'c', lines[2]._color)
    plt.legend(
        (lines[0], lines[1], lines[2], lines[3], lines[4]),
        (
            'IEEE 802.11ac (5.2 GHz)',
            'LTE (2.4 GHz)',
            '5G-NR with MEC (3.5GHz)',
            '5G-NR with MEC (700 MHz)',
            'Simulation'
        ), fontsize=8)

    plt.xlabel('$P_{t}$ (dBm)')
    plt.ylabel("Average EE (Mbps/Watts)")
    # plt.ylim(1e-3,1e-1)
    # plt.ylim(0.7, 1.05)
    # plt.xlim(cache_size[0], cache_size[len(cache_size)-1])

    # ax = plt.gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Urban Macro")
    plt.grid(True, which="both")
    plt.savefig(fname=out_file)
    plt.close()
    # plt.show()


##
# This function plots the results for eta_ee vs Pt_dBm in the InF-SL scenario
def plot_eta_ee_vs_Pt_dBm_InFSL(in_file: str, out_file: str,
                                input_variables: Dict[
                                    str, List[float]]) -> None:
    if not os.path.isfile(path=in_file):
        eta_ee_vs_Pt_dBm_InFSL(file_name=in_file,
                               input_variables=input_variables)
    numerical_results = spio.loadmat(file_name=in_file)
    computed_eta_ee = numerical_results['computed_eta_ee'] / 1e6
    simulated_eta_ee = numerical_results['simulated_eta_ee'] / 1e6
    Pt_dBm = numerical_results['Pt_dBm'][0]

    # Plot results
    lines = plt.plot(
        Pt_dBm, computed_eta_ee[0],
        Pt_dBm, computed_eta_ee[1],
        Pt_dBm, computed_eta_ee[2],
        Pt_dBm, computed_eta_ee[3],
        Pt_dBm, simulated_eta_ee[0], 'k.',
        Pt_dBm, simulated_eta_ee[1], 'k.',
        Pt_dBm, simulated_eta_ee[2], 'k.',
        Pt_dBm, simulated_eta_ee[3], 'k.',
    )
    plt.setp(lines[0], 'marker', 's')
    plt.setp(lines[1], 'marker', '>')
    plt.setp(lines[2], 'marker', 'v')
    plt.setp(lines[3], 'marker', '*', 'c', lines[2]._color)
    # plt.setp(lines[2], 'marker', 's', 'ls', '-.', 'c', 'r')
    # plt.setp(lines[3], 'marker', 'v', 'ls', '-.', 'c', lines[2]._color)
    plt.legend(
        (lines[0], lines[1], lines[2], lines[3], lines[4]),
        (
            'IEEE 802.11ac (5.2 GHz)',
            'LTE (2.4 GHz)',
            '5G-NR with MEC (3.5GHz)',
            '5G-NR with MEC (700 MHz)',
            'Simulation'
        ), fontsize=8)

    plt.xlabel('$P_{t}$ (dBm)')
    plt.ylabel(r"Average EE (Mbps/Watts)")
    # plt.ylim(1e-3,1e-1)
    # plt.ylim(0.7, 1.05)
    # plt.xlim(cache_size[0], cache_size[len(cache_size)-1])

    # ax = plt.gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Indoor Factory")
    plt.grid(True, which="both")
    plt.savefig(fname=out_file)
    plt.close()


##
# This function plots the results for throughput vs ra in the RMa scenario


##
# This function plots the results for latency vs ra in the UMa scenario
def plot_latency_vs_ra_UMa(in_file: str, out_file: str,
                           input_variables: Dict[str, List[float]]) -> None:
    if not os.path.isfile(path=in_file):
        latency_vs_ra_UMa(file_name=in_file, input_variables=input_variables)
    numerical_results = spio.loadmat(file_name=in_file)
    computed_latency = numerical_results['computed_latency'] * 1e3
    simulated_latency = numerical_results['simulated_latency'] * 1e3
    ra = numerical_results['ra'][0]

    # Plot results
    lines = plt.plot(
        ra, computed_latency[0],
        ra, computed_latency[1],
        ra, computed_latency[2],
        ra, computed_latency[3],
        ra, simulated_latency[0], 'k.',
        ra, simulated_latency[1], 'k.',
        ra, simulated_latency[2], 'k.',
        ra, simulated_latency[3], 'k.',
    )
    plt.setp(lines[0], 'marker', 's')
    plt.setp(lines[1], 'marker', '>')
    plt.setp(lines[2], 'marker', 'v')
    plt.setp(lines[3], 'marker', '*', 'c', lines[2]._color)
    # plt.setp(lines[2], 'marker', 's', 'ls', '-.', 'c', 'r')
    # plt.setp(lines[3], 'marker', 'v', 'ls', '-.', 'c', lines[2]._color)
    plt.legend(
        (lines[0], lines[1], lines[2], lines[3], lines[4]),
        (
            'IEEE 802.11ac (5.2 GHz)',
            'LTE (2.4 GHz)',
            '5G-NR with MEC (3.5GHz)',
            '5G-NR with MEC (700 MHz)',
            'Simulation'
        ), fontsize=8)

    plt.xlabel('$r_{a}$ (m)')
    plt.ylabel("Average Latency (ms)")
    # plt.ylim(1e-3,1e-1)
    # plt.ylim(0.7, 1.05)
    # plt.xlim(cache_size[0], cache_size[len(cache_size)-1])

    # ax = plt.gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Urban Macro")
    plt.grid(True, which="both")
    plt.savefig(fname=out_file)
    # plt.show()
    plt.close()


##
# This function plots the results for latency vs ra in the InF-SL scenario
def plot_latency_vs_ra_InFSL(in_file: str, out_file: str,
                             input_variables: Dict[str, List[float]]) -> None:
    if not os.path.isfile(path=in_file):
        latency_vs_ra_InFSL(file_name=in_file, input_variables=input_variables)
    numerical_results = spio.loadmat(file_name=in_file)
    computed_latency = numerical_results['computed_latency'] * 1e3
    simulated_latency = numerical_results['simulated_latency'] * 1e3
    ra = numerical_results['ra'][0]

    # Plot results
    lines = plt.plot(
        ra, computed_latency[0],
        ra, computed_latency[1],
        ra, computed_latency[2],
        ra, computed_latency[3],
        ra, simulated_latency[0], 'k.',
        ra, simulated_latency[1], 'k.',
        ra, simulated_latency[2], 'k.',
        ra, simulated_latency[3], 'k.',
    )
    plt.setp(lines[0], 'marker', 's')
    plt.setp(lines[1], 'marker', '>')
    plt.setp(lines[2], 'marker', 'v')
    plt.setp(lines[3], 'marker', '*', 'c', lines[2]._color)
    # plt.setp(lines[2], 'marker', 's', 'ls', '-.', 'c', 'r')
    # plt.setp(lines[3], 'marker', 'v', 'ls', '-.', 'c', lines[2]._color)
    plt.legend(
        (lines[0], lines[1], lines[2], lines[3], lines[4]),
        (
            'IEEE 802.11ac (5.2 GHz)',
            'LTE (2.4 GHz)',
            '5G-NR with MEC (3.5GHz)',
            '5G-NR with MEC (700 MHz)',
            'Simulation'
        ), fontsize=8)

    plt.xlabel('$r_{a}$ (m)')
    plt.ylabel(r"Average Latency (ms)")
    # plt.ylim(1e-3,1e-1)
    # plt.ylim(0.7, 1.05)
    # plt.xlim(cache_size[0], cache_size[len(cache_size)-1])

    # ax = plt.gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Indoor Factory")
    plt.grid(True, which="both")
    plt.savefig(fname=out_file)
    # plt.show()
    plt.close()


##
# This function plots the results for latency vs K_size in the UMa scenario
def plot_latency_vs_K_size_UMa(in_file: str, out_file: str,
                               input_variables: Dict[
                                   str, List[float]]) -> None:
    if not os.path.isfile(path=in_file):
        latency_vs_K_size_UMa(file_name=in_file,
                              input_variables=input_variables)
    numerical_results = spio.loadmat(file_name=in_file)
    computed_latency = numerical_results['computed_latency'] * 1e3
    simulated_latency = numerical_results['simulated_latency'] * 1e3
    K_size = numerical_results['K_size'][0] / 8e3

    # Plot results
    lines = plt.plot(
        K_size, computed_latency[0],
        K_size, computed_latency[1],
        K_size, computed_latency[2],
        K_size, computed_latency[3],
        K_size, simulated_latency[0], 'k.',
        K_size, simulated_latency[1], 'k.',
        K_size, simulated_latency[2], 'k.',
        K_size, simulated_latency[3], 'k.',
    )
    plt.setp(lines[0], 'marker', 's')
    plt.setp(lines[1], 'marker', '>')
    plt.setp(lines[2], 'marker', 'v')
    plt.setp(lines[3], 'marker', '*', 'c', lines[2]._color)
    # plt.setp(lines[2], 'marker', 's', 'ls', '-.', 'c', 'r')
    # plt.setp(lines[3], 'marker', 'v', 'ls', '-.', 'c', lines[2]._color)
    plt.legend(
        (lines[0], lines[1], lines[2], lines[3], lines[4]),
        (
            'IEEE 802.11ac (5.2 GHz)',
            'LTE (2.4 GHz)',
            '5G-NR with MEC (3.5GHz)',
            '5G-NR with MEC (700 MHz)',
            'Simulation'
        ), fontsize=8)

    plt.xlabel('$K_{size}$ (KB)')
    plt.ylabel("Average Latency (ms)")
    # plt.ylim(1e-3,1e-1)
    # plt.ylim(0.7, 1.05)
    # plt.xlim(cache_size[0], cache_size[len(cache_size)-1])

    # ax = plt.gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Urban Macro")
    plt.grid(True, which="both")
    plt.savefig(fname=out_file)
    # plt.show()
    plt.close()


##
# This function plots the results for latency vs K_size in the InF-SL scenario
def plot_latency_vs_K_size_InFSL(in_file: str, out_file: str,
                                 input_variables: Dict[
                                     str, List[float]]) -> None:
    if not os.path.isfile(path=in_file):
        latency_vs_K_size_InFSL(file_name=in_file,
                                input_variables=input_variables)
    numerical_results = spio.loadmat(file_name=in_file)
    computed_latency = numerical_results['computed_latency'] * 1e3
    simulated_latency = numerical_results['simulated_latency'] * 1e3
    K_size = numerical_results['K_size'][0] / 8e3

    # Plot results
    lines = plt.plot(
        K_size, computed_latency[0],
        K_size, computed_latency[1],
        K_size, computed_latency[2],
        K_size, computed_latency[3],
        K_size, simulated_latency[0], 'k.',
        K_size, simulated_latency[1], 'k.',
        K_size, simulated_latency[2], 'k.',
        K_size, simulated_latency[3], 'k.',
    )
    plt.setp(lines[0], 'marker', 's')
    plt.setp(lines[1], 'marker', '>')
    plt.setp(lines[2], 'marker', 'v')
    plt.setp(lines[3], 'marker', '*', 'c', lines[2]._color)
    # plt.setp(lines[2], 'marker', 's', 'ls', '-.', 'c', 'r')
    # plt.setp(lines[3], 'marker', 'v', 'ls', '-.', 'c', lines[2]._color)
    plt.legend(
        (lines[0], lines[1], lines[2], lines[3], lines[4]),
        (
            'IEEE 802.11ac (5.2 GHz)',
            'LTE (2.4 GHz)',
            '5G-NR with MEC (3.5GHz)',
            '5G-NR with MEC (700 MHz)',
            'Simulation'
        ), fontsize=8)

    plt.xlabel('$K_{size}$ (KB)')
    plt.ylabel(r"Average Latency (ms)")
    # plt.ylim(1e-3,1e-1)
    # plt.ylim(0.7, 1.05)
    # plt.xlim(cache_size[0], cache_size[len(cache_size)-1])

    # ax = plt.gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Indoor Factory")
    plt.grid(True, which="both")
    plt.savefig(fname=out_file)
    # plt.show()
    plt.close()


##
# This function runs the main function
def main():
    print(datetime.datetime.now().strftime(
        dateFormat + "|" + timeFormat) + ": Starting Python script")
    start_time = datetime.datetime.now()

    # Generate results
    throughput_vs_ra_UMa()
    throughput_vs_ra_InFSL()
    eta_ee_vs_ra_UMa()
    eta_ee_vs_ra_InFSL()
    eta_ee_vs_Pt_dBm_UMa()
    eta_ee_vs_Pt_dBm_InFSL()
    latency_vs_ra_UMa()
    latency_vs_ra_InFSL()
    latency_vs_K_size_UMa()
    latency_vs_K_size_InFSL()

    print(
        "Python script run time: " + str(datetime.datetime.now() - start_time))

    # Plot results
    plot_throughput_vs_ra_UMa()
    plot_throughput_vs_ra_InFSL()
    plot_eta_ee_vs_ra_UMa()
    plot_eta_ee_vs_ra_InFSL()
    plot_eta_ee_vs_Pt_dBm_UMa()
    plot_eta_ee_vs_Pt_dBm_InFSL()
    plot_latency_vs_ra_UMa()
    plot_latency_vs_ra_InFSL()
    plot_latency_vs_K_size_UMa()
    plot_latency_vs_K_size_InFSL()


if __name__ == "__main__":
    main()
