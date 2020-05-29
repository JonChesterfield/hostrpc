#ifndef HOSTRPC_COMMON_H_INCLUDED
#define HOSTRPC_COMMON_H_INCLUDED

#include <cassert>
#include <cstdint>

namespace hostrpc
{
struct cacheline_t
{
  alignas(64) uint64_t element[8];
};
static_assert(sizeof(cacheline_t) == 64, "");

struct page_t
{
  alignas(4096) cacheline_t cacheline[64];
};
static_assert(sizeof(page_t) == 4096, "");

// The bitmap in question is usually a runtime parameter, but as it's
// invariant during program execution I think it's worth tolerating this
// anyway. Going to lead to a switch somewhere.

// Need a bitmap which sets an element by CAS
// Also need to support find-some-bit-set

// Uses:
// client needs one to track which slots in the buffer are currently owned
// client uses another to publish which slots are available to read
// server reads from one a word at a time, but probably wants another as a cache
// server uses one to track which slots in the buffer are currently owned
// server uses another to publish which slots are available to read
// client probably wants a cache as well
// six instances
// different scoping rules needed for the various updates, e.g. the bitmap
// which is only used from one device should have device scope on the cas
// not sure if we can get away with relaxed atomic read/writes

namespace
{
namespace detail
{
inline bool multiple_of_64(uint64_t x) { return (x % 64) == 0; }

inline uint64_t round_up_to_multiple_of_64(uint64_t x)
{
  return 64u * ((x + 63u) / 64u);
}

inline bool nthbitset64(uint64_t x, uint64_t n)
{
  assert(n < 64);
  return x & (1ull << n);
}

inline uint64_t setnthbit64(uint64_t x, uint64_t n)
{
  assert(n < 64);
  return x | (1ull << n);
}

inline uint64_t clearnthbit64(uint64_t x, uint64_t n)
{
  assert(n < 64);
  return x & ~(1ull << n);
}

}  // namespace detail
}  // namespace

// probably need scope as a template parameter on this
// not a general purpose bitmap
template <size_t N, size_t scope = __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>
struct slot_bitmap
{
  static_assert(N != 0, "");
  static_assert(N % 64 == 0, "Size must be multiple of 64");

  constexpr slot_bitmap() = default;

  static constexpr size_t size() { return N; }

  bool operator[](size_t i) const
  {
    size_t w = index_to_element(i);
    uint64_t d = load_relaxed(w);
    return detail::nthbitset64(d, index_to_subindex(i));
  }

  bool claim_slot(size_t i)
  {
    assert(i < N);
    size_t w = index_to_element(i);
    uint64_t subindex = index_to_subindex(i);

    uint64_t d = load_relaxed(w);
    uint64_t* addr = &data[w];

    for (;;)
      {
        // if the bit was already set then we've lost the race

        // can either check the bit is zero, or unconditionally set it and check
        // if this changed the value
        uint64_t proposed = detail::setnthbit64(d, subindex);
        if (proposed == d)
          {
            return false;
          }

        // bool __atomic_compare_exchange(size_t size, void *ptr, void
        // *expected, void *desired, int success_order, int failure_order)
        // not sure about the memory orders
        // need to use an intrinsic which understand scope

        // opencl wants the array marked atomic, which might be fair enough
        //(void)__opencl_atomic_compare_exchange_strong(
        // (_Atomic volatile T *)address, &compare, val, __ATOMIC_SEQ_CST,
        // __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);

        uint64_t compare = d;
        bool r = __atomic_compare_exchange(addr, &compare, &proposed, false,
                                           __ATOMIC_SEQ_CST, __ATOMIC_RELAXED);

        if (r)
          {
            // success, it's all ours
            return true;
          }

        // cas failed. reasons:
        // we lost the slot
        // another slot in the same word changed
        // spurious
        // in any case, go around again

        d = proposed;  // docs not totally clear, but an updated copy of the
                       // word should be in one of the passed parameters. Might
                       // be in compare
      }
  }

  void release_slot(size_t i)
  {
    // programming error if the slot is not set on entry
    assert(i < N);
    size_t w = index_to_element(i);
    uint64_t subindex = index_to_subindex(i);

    uint64_t d = load_relaxed(w);
    uint64_t* addr = &data[w];

    for (;;)
      {
        uint64_t proposed = detail::clearnthbit64(d, subindex);
        // error if set on entry
        // cas should not return false if the write succeeded
        assert(proposed != d);

        uint64_t compare = d;
        bool r = __atomic_compare_exchange(addr, &compare, &proposed, false,
                                           __ATOMIC_SEQ_CST, __ATOMIC_RELAXED);

        if (r)
          {
            // success, it's all ours
            return;
          }

        // cas failed. reasons:
        // another slot in the same word changed
        // spurious

        d = proposed;
      }
  }

 private:
  static uint64_t index_to_element(uint64_t x)
  {
    assert(x < size());
    return x / 64u;
  }

  static uint64_t index_to_subindex(uint64_t x)
  {
    assert(x < size());
    return x % 64u;
  }

  uint64_t load_relaxed(size_t i)
  {
    return __atomic_load_n(&data[i], __ATOMIC_RELAXED);
  }

  alignas(64) uint64_t data[N / 64] = {0};
};

}  // namespace hostrpc

#endif
