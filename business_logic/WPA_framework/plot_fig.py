from main import *  # imports main.py to call the plotting functions

plot_thres = 1e2


def main():
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


# Runs main()
if __name__ == "__main__":
    main()
