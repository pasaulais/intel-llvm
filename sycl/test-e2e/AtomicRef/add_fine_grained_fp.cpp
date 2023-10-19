// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

using namespace sycl;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T, typename Difference = T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void add_plus_equal_test(queue q, size_t N) {
  T *sum = malloc_shared<T>(1, q);
  T *output = malloc_shared<T>(N, q);
  T *output_begin = &output[0], *output_end = &output[N];
  sum[0] = T(0);
  std::fill(output_begin, output_end, T(0));
  {
    q.submit([&](handler &cgh) {
       cgh.parallel_for(range<1>(N), [=](item<1> it) {
         int gid = it.get_id(0);
         auto atm = AtomicRef < T,
              (order == memory_order::acquire || order == memory_order::release)
                  ? memory_order::relaxed
                  : order,
              scope, space > (sum[0]);
         output[gid] = atm += Difference(1);
       });
     }).wait_and_throw();
  }

  // All work-items increment by 1, so final value should be equal to N
  std::cout << "Sum: " << sum[0] << std::endl;
  assert(sum[0] == T(N));

  // += returns updated value: will be in [1, N]
  auto min_e = std::min_element(output_begin, output_end);
  auto max_e = std::max_element(output_begin, output_end);
  assert(*min_e == T(1) && *max_e == T(N));

  // Intermediate values should be unique
  std::sort(output_begin, output_end);
  assert(std::unique(output_begin, output_end) == output_end);

  free(sum, q);
  free(output, q);
}

int main() {
  queue q;
  constexpr int N = 32;

  add_plus_equal_test<::sycl::atomic_ref, access::address_space::global_space,
                      float, float, memory_order::relaxed,
                      memory_scope::device>(q, N);
  std::cout << "Test passed." << std::endl;
  return 0;
}
