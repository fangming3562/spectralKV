
#include <cstdint>
#include <vector>
extern "C" int64_t prepare(int64_t n, int64_t m, const int64_t *nbr,
                           const float *kap, const float *wf, double *weight,
                           int64_t *offsets, int64_t *incoming,
                           double *strength, double *initial, double *lower) {
  for (int64_t i = 0; i <= n; ++i)
    offsets[i] = 0;
  for (int64_t i = 0; i < n; ++i) {
    weight[i] = double(wf[i]);
    initial[i] = weight[i];
    for (int64_t s = 0; s < m; ++s) {
      int64_t e = i * m + s;
      if (kap[e] > 0 && wf[i] > 0)
        ++offsets[nbr[e] + 1];
    }
  }
  for (int64_t i = 0; i < n; ++i)
    offsets[i + 1] += offsets[i];
  std::vector<int64_t> cursor(offsets, offsets + n);
  for (int64_t i = 0; i < n; ++i) {
    double maximum = 0.;
    for (int64_t s = 0; s < m; ++s) {
      int64_t e = i * m + s;
      double value = double(kap[e]);
      if (value > maximum)
        maximum = value;
      if (value <= 0 || wf[i] <= 0)
        continue;
      int64_t j = nbr[e], p = cursor[j]++;
      incoming[p] = i;
      strength[p] = value;
      initial[j] = initial[j] + weight[i] * value;
    }
    if (lower)
      lower[i] = weight[i] * (1. - maximum);
  }
  return offsets[n];
}
