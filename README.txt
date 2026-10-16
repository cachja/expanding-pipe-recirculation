Run:
    python recirculation_onset_materials.py

Required packages:
    numpy
    scipy
    matplotlib

Inputs:
    peixinho2013.txt, if supplied; otherwise embedded onset data are used.

Outputs:
    01_semilogy_sinalpha_vs_Re.pdf
    02_loglog_sinalpha_vs_Re_with_endpoint_inset.pdf
    source_data_theoretical_curves.csv
    source_data_peixinho_besnard_2013.csv

The BVP returns k_raw. The reported radius-based parameter used in the
figures is K_R = pi * k_raw. Published diameter-based Reynolds numbers
are obtained using Re_R = Re_d/2.