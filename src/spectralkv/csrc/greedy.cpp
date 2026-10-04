
#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <utility>
#include <vector>
extern "C" void solve(int64_t n, int64_t steps, int64_t batch,
                      const double *weight, const int64_t *offsets,
                      const int64_t *incoming, const double *strength,
                      const double *initial, const uint8_t *eligible,
                      int64_t *selected, double *marginals, double *covered,
                      int64_t *stats, void *blas_dot) {
  using Item = std::pair<double, int64_t>;
  using Dot =
      double (*)(int64_t, const double *, int64_t, const double *, int64_t);
  Dot dot = reinterpret_cast<Dot>(blas_dot);
  int64_t maximum = 0;
  for (int64_t i = 0; i < n; ++i)
    maximum = std::max(maximum, offsets[i + 1] - offsets[i]);
  std::vector<double> left(maximum), right(maximum);
  std::vector<int64_t> seen(n, -1);
  std::vector<Item> data;
  data.reserve(n);
  for (int64_t i = 0; i < n; ++i)
    if (eligible[i])
      data.emplace_back(-initial[i], i);
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap(
      std::greater<Item>(), std::move(data));
  int64_t accepted = 0, evaluations = 0, rounds = 0, largest = 0, conflicts = 0;
  while (accepted < steps) {
    std::vector<Item> prefix;
    prefix.reserve(std::min(batch, steps - accepted));
    while ((int64_t)prefix.size() < batch &&
           accepted + (int64_t)prefix.size() < steps) {
      auto old = heap.top();
      heap.pop();
      int64_t j = old.second;
      int64_t count = offsets[j + 1] - offsets[j];
      for (int64_t p = offsets[j]; p < offsets[j + 1]; ++p) {
        int64_t i = incoming[p];
        left[p - offsets[j]] = weight[i];
        right[p - offsets[j]] = std::max(strength[p] - covered[i], 0.0);
      }
      double subtotal =
          count ? dot(count, left.data(), 1, right.data(), 1) : 0.0;
      double gain = weight[j] * (1 - covered[j]) + subtotal;
      ++evaluations;
      Item current(-gain, j);
      if (!heap.empty() && current > heap.top()) {
        heap.push(current);
        continue;
      }
      bool conflict = seen[j] == rounds;
      for (int64_t p = offsets[j]; p < offsets[j + 1] && !conflict; ++p)
        conflict = seen[incoming[p]] == rounds;
      if (conflict) {
        heap.push(current);
        ++conflicts;
        break;
      }
      prefix.push_back(current);
      seen[j] = rounds;
      for (int64_t p = offsets[j]; p < offsets[j + 1]; ++p)
        seen[incoming[p]] = rounds;
    }
    largest = std::max(largest, (int64_t)prefix.size());
    for (auto item : prefix) {
      int64_t j = item.second;
      selected[accepted] = j;
      marginals[accepted] = -item.first;
      ++accepted;
      for (int64_t p = offsets[j]; p < offsets[j + 1]; ++p) {
        int64_t i = incoming[p];
        covered[i] = std::max(covered[i], strength[p]);
      }
      covered[j] = 1.0;
    }
    ++rounds;
  }
  stats[0] = evaluations;
  stats[1] = rounds;
  stats[2] = largest;
  stats[3] = conflicts;
}
