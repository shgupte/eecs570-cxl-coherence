// Greedy sample placement, the inner loop of sample_placement.py (same
// scores, same tie-breaking). Build: gcc -O2 -shared -fPIC -o placer.so placer.c
#include <stdint.h>
#include <stddef.h>

// rid: B*T compact row ids (-1 = no lookup). w: per-row weight. cur: per-row
// mask of GPUs that already read the row this step (in/out, caller zeroes).
// prior, bonus: optional per-row GPU masks. owner: optional static owner GPU.
void assign(int B, int T, const int32_t *rid, int G, int cap, const float *w,
            double lam, const uint8_t *prior, const uint8_t *bonus, double beta,
            const int8_t *owner, double wown, uint8_t *cur, int8_t *out)
{
    long load[8] = {0};
    for (int s = 0; s < B; s++) {
        double sc[8] = {0};
        const int32_t *r = rid + (size_t)s * T;
        for (int t = 0; t < T; t++) {
            int x = r[t];
            if (x < 0) continue;
            uint8_t c = cur[x];
            float wx = w[x];
            if (wx != 0) {
                for (int g = 0; g < G; g++) sc[g] += wx * ((c >> g) & 1);
                if (prior) { uint8_t p = prior[x]; for (int g = 0; g < G; g++) sc[g] += 0.5 * wx * ((p >> g) & 1); }
            }
            if (bonus) { uint8_t b = bonus[x]; for (int g = 0; g < G; g++) sc[g] += beta * ((b >> g) & 1); }
            if (owner) sc[owner[x]] += wown;
        }
        int best = -1; double bs = 0;
        for (int g = 0; g < G; g++) {
            if (load[g] >= cap) continue;
            double v = sc[g] - lam * load[g] / cap;
            if (best < 0 || v > bs) { best = g; bs = v; }
        }
        out[s] = (int8_t)best; load[best]++;
        for (int t = 0; t < T; t++) if (r[t] >= 0) cur[r[t]] |= (uint8_t)(1 << best);
    }
}
