max |Im lambda_+| = 2.4e-02

## Hierarchy ÷ non-hierarchy, by update rule

| structure | α | rule | N = 300 | 10,000 | 100,000 |
|---|---|---|---|---|---|
| grow_r0.8 | 1 | synchronous | 2.49 | 2.68 | 2.72 |
| grow_r0.8 | 1 | continuous | 2.33 | 2.58 | 2.64 |
| grow_r0.8 | 1 | half-weight | 2.40 | 2.63 | 2.68 |
| grow_r0.8 | 4 | synchronous | 1.46 | 0.83 | 0.64 |
| grow_r0.8 | 4 | continuous | 1.04 | 0.77 | 0.65 |
| grow_r0.8 | 4 | half-weight | 1.04 | 0.76 | 0.64 |
| grow_r0.8 | 6 | synchronous | 1.62 | 1.00 | 0.82 |
| grow_r0.8 | 6 | continuous | 0.81 | 0.61 | 0.52 |
| grow_r0.8 | 6 | half-weight | 0.80 | 0.59 | 0.51 |
| grow_r0.65 | 1 | synchronous | 3.32 | 4.42 | 4.64 |
| grow_r0.65 | 1 | continuous | 2.96 | 4.02 | 4.29 |
| grow_r0.65 | 1 | half-weight | 3.12 | 4.20 | 4.45 |
| grow_r0.65 | 4 | synchronous | 2.07 | 1.34 | 1.04 |
| grow_r0.65 | 4 | continuous | 1.73 | 1.31 | 1.04 |
| grow_r0.65 | 4 | half-weight | 1.78 | 1.33 | 1.04 |
| grow_r0.65 | 6 | synchronous | 2.09 | 1.05 | 0.73 |
| grow_r0.65 | 6 | continuous | 1.21 | 0.81 | 0.61 |
| grow_r0.65 | 6 | half-weight | 1.22 | 0.80 | 0.60 |
| fixed_r0.8 | 1 | synchronous | 2.46 | 2.46 | 2.46 |
| fixed_r0.8 | 1 | continuous | 2.31 | 2.31 | 2.31 |
| fixed_r0.8 | 1 | half-weight | 2.38 | 2.38 | 2.38 |
| fixed_r0.8 | 4 | synchronous | 1.47 | 1.40 | 1.40 |
| fixed_r0.8 | 4 | continuous | 1.02 | 1.02 | 1.03 |
| fixed_r0.8 | 4 | half-weight | 1.02 | 1.03 | 1.03 |
| fixed_r0.8 | 6 | synchronous | 1.61 | 1.56 | 1.55 |
| fixed_r0.8 | 6 | continuous | 0.80 | 0.85 | 0.87 |
| fixed_r0.8 | 6 | half-weight | 0.78 | 0.85 | 0.86 |

## Power law vs log law (growing structures; window N ≤ 10k → window N ≥ 10k)

| structure | α | q | rule | power-law exponent | log-law slope (steps per unit ln N) |
|---|---|---|---|---|---|
| grow_r0.8 | 1 | 0.05 | synchronous | +0.194 → +0.139 | 43.2 → 49.7 |
| grow_r0.8 | 1 | 0.05 | continuous | +0.188 → +0.136 | 43.2 → 49.7 |
| grow_r0.8 | 1 | 0.05 | half-weight | +0.191 → +0.137 | 86.3 → 99.4 |
| grow_r0.8 | 1 | 1.0 | synchronous | +0.174 → +0.132 | 14.9 → 17.6 |
| grow_r0.8 | 1 | 1.0 | continuous | +0.160 → +0.125 | 14.9 → 17.5 |
| grow_r0.8 | 1 | 1.0 | half-weight | +0.167 → +0.129 | 29.8 → 35.1 |
| grow_r0.8 | 4 | 0.05 | synchronous | +0.005 → +0.002 | 0.4 → 0.1 |
| grow_r0.8 | 4 | 0.05 | continuous | +0.064 → +0.030 | 5.0 → 2.7 |
| grow_r0.8 | 4 | 0.05 | half-weight | +0.067 → +0.031 | 9.9 → 5.3 |
| grow_r0.8 | 4 | 1.0 | synchronous | +0.160 → +0.115 | 12.8 → 13.7 |
| grow_r0.8 | 4 | 1.0 | continuous | +0.146 → +0.109 | 12.7 → 13.7 |
| grow_r0.8 | 4 | 1.0 | half-weight | +0.152 → +0.112 | 25.5 → 27.3 |
| grow_r0.8 | 6 | 0.05 | synchronous | +0.006 → +0.003 | 0.5 → 0.2 |
| grow_r0.8 | 6 | 0.05 | continuous | +0.043 → +0.017 | 2.3 → 1.0 |
| grow_r0.8 | 6 | 0.05 | half-weight | +0.046 → +0.019 | 4.6 → 2.0 |
| grow_r0.8 | 6 | 1.0 | synchronous | +0.137 → +0.090 | 9.4 → 8.7 |
| grow_r0.8 | 6 | 1.0 | continuous | +0.123 → +0.084 | 9.4 → 8.7 |
| grow_r0.8 | 6 | 1.0 | half-weight | +0.129 → +0.087 | 18.8 → 17.3 |
| grow_r0.65 | 1 | 0.05 | synchronous | +0.167 → +0.102 | 30.9 → 27.7 |
| grow_r0.65 | 1 | 0.05 | continuous | +0.161 → +0.099 | 30.9 → 27.7 |
| grow_r0.65 | 1 | 0.05 | half-weight | +0.164 → +0.100 | 61.7 → 55.5 |
| grow_r0.65 | 1 | 1.0 | synchronous | +0.082 → +0.084 | 3.9 → 5.1 |
| grow_r0.65 | 1 | 1.0 | continuous | +0.071 → +0.075 | 3.9 → 5.1 |
| grow_r0.65 | 1 | 1.0 | half-weight | +0.076 → +0.079 | 7.8 → 10.1 |
| grow_r0.65 | 4 | 0.05 | synchronous | +0.064 → +0.042 | 6.3 → 4.9 |
| grow_r0.65 | 4 | 0.05 | continuous | +0.085 → +0.040 | 8.6 → 4.9 |
| grow_r0.65 | 4 | 0.05 | half-weight | +0.088 → +0.041 | 17.2 → 9.8 |
| grow_r0.65 | 4 | 1.0 | synchronous | +0.191 → +0.150 | 11.5 → 14.7 |
| grow_r0.65 | 4 | 1.0 | continuous | +0.169 → +0.140 | 11.4 → 14.6 |
| grow_r0.65 | 4 | 1.0 | half-weight | +0.179 → +0.145 | 22.8 → 29.3 |
| grow_r0.65 | 6 | 0.05 | synchronous | -0.004 → -0.000 | -0.4 → -0.0 |
| grow_r0.65 | 6 | 0.05 | continuous | +0.054 → +0.022 | 3.6 → 1.6 |
| grow_r0.65 | 6 | 0.05 | half-weight | +0.057 → +0.023 | 7.2 → 3.2 |
| grow_r0.65 | 6 | 1.0 | synchronous | +0.199 → +0.151 | 12.1 → 15.1 |
| grow_r0.65 | 6 | 1.0 | continuous | +0.176 → +0.141 | 12.1 → 15.1 |
| grow_r0.65 | 6 | 1.0 | half-weight | +0.186 → +0.145 | 24.1 → 30.2 |
