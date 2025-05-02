# Parallel-ScientificComputing-Project

Main source files for the project are located in the `svd_algorithms/` directory, which contains the following files:
- [svd_algorithms/svd_QR_omp.cpp](svd_algorithms/svd_QR_omp.cpp): Implementation of the QR algorithm for SVD using OpenMP.
- [svd_algorithms/golub_kahan_omp.cpp](svd_algorithms/golub_kahan_omp.cpp): Implementation of the Golub-Kahan algorithm for SVD using OpenMP.

The `plots/` directory contains the plots generated for weak scaling and strong scaling experiments. The plots are generated using Python and Matplotlib.

The `generate_data/` directory contains the scripts used to generate the input data for the experiments.