
**[[250,10,15]]**, logical error rate

| decoder | p = 0.0392 | p = 0.0518 | p = 0.0685 | p = 0.0906 |
|---|---|---|---|---|
| BP+OSD-CS7 | 3.0e-03 | 1.9e-02 | 9.1e-02 | 3.3e-01 |
| MBP4+OSD-1 | 9.0e-04 | 2.6e-03 | 1.8e-02 | 1.2e-01 |
| GMBP4+OSD-1 [Mostad et al.] | 0 (/100000) | 0 (/100000) | 2.3e-04 | 1.0e-02 |
| SOGRAND [Rapp et al.] | 2.8e-03 | 1.0e-02 | 2.9e-02 | 1.4e-01 |
| SOGRAND+XZ [Rapp et al.] | 9.0e-05 | 1.1e-03 | 7.1e-03 | 3.4e-02 |
| LEAD [Xiao et al.] | 3.5e-02 | 1.0e-01 | 2.8e-01 | 6.4e-01 |
| MBP4+LRB-MS-8 (ours) | 3.0e-05 | 1.6e-04 | 9.5e-04 | 1.5e-02 |
| MBP4+MAP, 100 it. (exact GC) | 2.9e-04 | 3.7e-04 | 1.5e-03 | 1.8e-02 |
| LEAD α=0.01 [Xiao et al.] | 5.7e-03 | 2.7e-02 | 1.2e-01 | 4.2e-01 |
| MBP4+LRB-MS-8+ladder (ours) | 0 (/200000) | 2.0e-05 | 2.3e-04 | 5.7e-03 |
| SOGRAND+XZ+ladder [Rapp et al. + ours] | 0 (/100000) | 3.0e-05 | 5.3e-04 | 1.1e-02 |

**[[250,10,15]]**, mean decode time per shot

| decoder | p = 0.0392 | p = 0.0518 | p = 0.0685 | p = 0.0906 |
|---|---|---|---|---|
| BP+OSD-CS7 | 0.15 ms | 0.37 ms | 0.58 ms | 1.38 ms |
| MBP4+OSD-1 | 1.14 ms | 1.60 ms | 2.16 ms | 3.21 ms |
| GMBP4+OSD-1 [Mostad et al.] | 1.15 ms | 1.58 ms | 2.43 ms | 3.57 ms |
| SOGRAND [Rapp et al.] | 0.65 ms | 1.03 ms | 1.74 ms | 5.33 ms |
| SOGRAND+XZ [Rapp et al.] | 0.67 ms | 0.90 ms | 1.44 ms | 4.22 ms |
| LEAD [Xiao et al.] | 2.30 ms | 4.72 ms | 5.88 ms | 6.71 ms |
| MBP4+LRB-MS-8 (ours) | 0.32 ms | 0.37 ms | 0.51 ms | 1.11 ms |
| MBP4+MAP, 100 it. (exact GC) | 0.68 ms | 0.78 ms | 1.11 ms | 2.01 ms |
| LEAD α=0.01 [Xiao et al.] | 1.52 ms | 1.96 ms | 2.82 ms | 4.58 ms |
| MBP4+LRB-MS-8+ladder (ours) | 0.33 ms | 0.38 ms | 0.52 ms | 1.44 ms |
| SOGRAND+XZ+ladder [Rapp et al. + ours] | 0.66 ms | 0.90 ms | 1.62 ms | 5.49 ms |

**[[432,16,28]]**, logical error rate

| decoder | p = 0.06 | p = 0.07 | p = 0.08 | p = 0.09 | p = 0.1 |
|---|---|---|---|---|---|
| BP+OSD-CS7 | 7.2e-02 | 1.9e-01 | 4.0e-01 | 6.3e-01 | 8.1e-01 |
| MBP4+OSD-1 | 4.5e-02 | 1.5e-01 | 3.2e-01 | 5.2e-01 | 7.9e-01 |
| GMBP4+OSD-1 [Mostad et al.] | 0 (/12000) | 1.7e-04 | 4.2e-04 | 5.4e-03 | 2.9e-02 |
| SOGRAND+XZ [Rapp et al.] | 1.7e-03 | 4.1e-03 | 6.0e-03 | 2.0e-02 | 5.2e-02 |
| LEAD [Xiao et al.] | 3.0e-01 | 5.3e-01 | 7.6e-01 | 8.9e-01 | 9.6e-01 |
| LEAD α=0.01 [Xiao et al.] | 1.5e-01 | 3.0e-01 | 5.8e-01 | 7.7e-01 | 9.1e-01 |
| MBP4+LRB-MS-8 (ours) | 1.0e-05 | 3.0e-05 | 1.6e-04 | 1.5e-03 | 9.8e-03 |
| MBP4+LRB-MS-8+ladder (ours) | 0 (/200000) | 0 (/200000) | 4.0e-05 | 5.2e-04 | 4.6e-03 |
| SOGRAND+XZ+ladder [Rapp et al. + ours] | 8.3e-05 | 8.3e-05 | 1.7e-04 | 2.2e-03 | 1.2e-02 |

**[[432,16,28]]**, mean decode time per shot

| decoder | p = 0.06 | p = 0.07 | p = 0.08 | p = 0.09 | p = 0.1 |
|---|---|---|---|---|---|
| BP+OSD-CS7 | 3.21 ms | 7.61 ms | 14.48 ms | 22.55 ms | 33.25 ms |
| MBP4+OSD-1 | 18.59 ms | 24.32 ms | 32.66 ms | 35.37 ms | 44.96 ms |
| GMBP4+OSD-1 [Mostad et al.] | 41.86 ms | 57.61 ms | 70.80 ms | 87.43 ms | 102.27 ms |
| SOGRAND+XZ [Rapp et al.] | 17.42 ms | 22.06 ms | 27.39 ms | 42.95 ms | 71.05 ms |
| LEAD [Xiao et al.] | 21.78 ms | 32.96 ms | 47.19 ms | 56.53 ms | 57.34 ms |
| LEAD α=0.01 [Xiao et al.] | 11.33 ms | 21.84 ms | 32.08 ms | 46.74 ms | 56.53 ms |
| MBP4+LRB-MS-8 (ours) | 0.73 ms | 0.86 ms | 1.07 ms | 1.48 ms | 2.17 ms |
| MBP4+LRB-MS-8+ladder (ours) | 0.76 ms | 0.92 ms | 1.08 ms | 1.45 ms | 2.79 ms |
| SOGRAND+XZ+ladder [Rapp et al. + ours] | 17.94 ms | 22.64 ms | 28.29 ms | 48.31 ms | 108.41 ms |

**[[576,32,≤16]] C16**, logical error rate

| decoder | p = 0.07 | p = 0.08 | p = 0.09 | p = 0.1 | p = 0.11 |
|---|---|---|---|---|---|
| BP+OSD-CS7 | 4.3e-02 | 1.2e-01 | 2.4e-01 | 5.6e-01 | 7.7e-01 |
| MBP4+OSD-1 | 1.1e-02 | 4.2e-02 | 1.2e-01 | 3.4e-01 | 6.1e-01 |
| GMBP4+OSD-1 [Mostad et al.] | 5.0e-05 | 5.0e-05 | 9.5e-04 | 8.3e-03 | 5.5e-02 |
| SOGRAND+XZ [Rapp et al.] | 1.2e-02 | 1.8e-02 | 4.7e-02 | 6.2e-02 | 1.5e-01 |
| LEAD [Xiao et al.] | 2.6e-01 | 5.5e-01 | 7.4e-01 | 9.2e-01 | 9.6e-01 |
| LEAD α=0.01 [Xiao et al.] | 7.0e-02 | 2.8e-01 | 4.7e-01 | 7.8e-01 | 9.0e-01 |
| MBP4+LRB-MS-8 (ours) | 3.0e-05 | 9.0e-05 | 3.9e-04 | 1.6e-03 | 7.8e-03 |
| MBP4+LRB-MS-8+ladder (ours) | 1.5e-05 | 6.5e-05 | 1.7e-04 | 9.5e-04 | 3.7e-03 |
| SOGRAND+XZ+ladder [Rapp et al. + ours] | 1.0e-04 | 0 (/20000) | 1.1e-03 | 5.5e-03 | 2.4e-02 |

**[[576,32,≤16]] C16**, mean decode time per shot

| decoder | p = 0.07 | p = 0.08 | p = 0.09 | p = 0.1 | p = 0.11 |
|---|---|---|---|---|---|
| BP+OSD-CS7 | 1.50 ms | 3.15 ms | 6.79 ms | 14.87 ms | 21.50 ms |
| MBP4+OSD-1 | 9.92 ms | 15.25 ms | 18.97 ms | 25.37 ms | 30.41 ms |
| GMBP4+OSD-1 [Mostad et al.] | 10.88 ms | 14.97 ms | 20.02 ms | 25.76 ms | 35.08 ms |
| SOGRAND+XZ [Rapp et al.] | 5.52 ms | 6.76 ms | 10.09 ms | 14.87 ms | 28.57 ms |
| LEAD [Xiao et al.] | 17.73 ms | 27.45 ms | 36.94 ms | 53.91 ms | 57.58 ms |
| LEAD α=0.01 [Xiao et al.] | 8.62 ms | 18.01 ms | 26.10 ms | 43.16 ms | 54.74 ms |
| MBP4+LRB-MS-8 (ours) | 1.00 ms | 1.15 ms | 1.36 ms | 1.62 ms | 2.31 ms |
| MBP4+LRB-MS-8+ladder (ours) | 1.02 ms | 1.17 ms | 1.41 ms | 1.82 ms | 3.03 ms |
| SOGRAND+XZ+ladder [Rapp et al. + ours] | 5.58 ms | 7.35 ms | 10.64 ms | 17.98 ms | 40.58 ms |

**BB [[144,12,12]]**, logical error rate

| decoder | p = 0.04 | p = 0.05 | p = 0.06 |
|---|---|---|---|
| BP+OSD-CS7 | 3.1e-03 | 1.1e-02 | 2.0e-02 |
| MBP4+OSD-1 | 9.0e-05 | 6.0e-04 | 2.8e-03 |
| GMBP4+OSD-1 [Mostad et al.] | 1.1e-04 | 4.1e-04 | 2.3e-03 |
| MBP4+LRB-MS-6 (ours) | 1.0e-04 | 3.7e-04 | 1.5e-03 |

**BB [[144,12,12]]**, mean decode time per shot

| decoder | p = 0.04 | p = 0.05 | p = 0.06 |
|---|---|---|---|
| BP+OSD-CS7 | 0.04 ms | 0.06 ms | 0.08 ms |
| MBP4+OSD-1 | 0.30 ms | 0.33 ms | 0.41 ms |
| GMBP4+OSD-1 [Mostad et al.] | 0.30 ms | 0.41 ms | 0.55 ms |
| MBP4+LRB-MS-6 (ours) | 0.15 ms | 0.18 ms | 0.28 ms |
