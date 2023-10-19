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
          access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void xor_global_fine_grained_test(queue q) {
  const size_t N = 32;
  T *cum = malloc_shared<T>(1, q);
  T *output = malloc_shared<T>(N, q);
  T *output_begin = &output[0], *output_end = &output[N];
  cum[0] = T(0);
  std::fill(output_begin, output_end, T(0));
  {
    q.submit([&](handler &cgh) {
       cgh.parallel_for(range<1>(N), [=](item<1> it) {
         size_t gid = it.get_id(0);
         auto atm = AtomicRef < T,
              (order == memory_order::acquire || order == memory_order::release)
                  ? memory_order::relaxed
                  : order,
              scope, space > (cum[0]);
         output[gid] = atm.fetch_xor(T(1ll << gid), order);
       });
     }).wait_and_throw();
  }

  // Final value should be equal to N ones
  std::cout << "Cumulative: " << cum[0] << std::endl;
  assert(cum[0] == T((1ll << N) - 1));

  // All other values should be unique; each wxork-item sets one bit to 1
  std::sort(output_begin, output_end);
  assert(std::unique(output_begin, output_end) == output_end);

  free(cum, q);
  free(output, q);
}

int main() {
  queue q;
  xor_global_fine_grained_test<
      ::sycl::atomic_ref, access::address_space::global_space, unsigned int,
      memory_order::relaxed, memory_scope::device>(q);
  std::cout << "Test passed." << std::endl;
  return 0;
}
