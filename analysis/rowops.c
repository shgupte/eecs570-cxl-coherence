// Host-side embedding row gather and update, OpenMP (for host_gather_bench.py).
// gcc -O3 -march=native -fopenmp -shared -fPIC -o rowops.so rowops.c
#include <omp.h>
#include <stdint.h>
#include <string.h>

void fill(float *tab, long n, int nt) {            // parallel first touch
#pragma omp parallel for num_threads(nt) schedule(static)
    for (long i = 0; i < n; i++) tab[i] = 0.001f;
}

void gather(const float *tab, const int64_t *idx, float *out, long n, int d, int nt) {
#pragma omp parallel for num_threads(nt) schedule(static)
    for (long i = 0; i < n; i++) {
        if (i + 8 < n) __builtin_prefetch(tab + idx[i + 8] * d);
        memcpy(out + i * d, tab + idx[i] * d, (size_t)d * sizeof(float));
    }
}

void scatter_add(float *tab, const int64_t *idx, const float *g, long n, int d, float alpha, int nt) {
#pragma omp parallel for num_threads(nt) schedule(static)   // idx is unique: rows are disjoint
    for (long i = 0; i < n; i++) {
        if (i + 8 < n) __builtin_prefetch(tab + idx[i + 8] * d, 1);
        float *r = tab + idx[i] * d;
        const float *x = g + i * d;
        for (int j = 0; j < d; j++) r[j] += alpha * x[j];
    }
}
