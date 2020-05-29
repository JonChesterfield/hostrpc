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

inline uint64_t ctz64(uint64_t value)
{
  if (value == 0)
    {
      return 64;
    }
#if defined(__has_builtin) && __has_builtin(__builtin_ctzl)
  static_assert(
      sizeof(unsigned long) == sizeof(uint64_t),
      "Calling __builtin_ctzl on a uint64_t requires 64 bit unsigned long");
  return (uint64_t)__builtin_ctzl(value);
#else
  uint64_t pos = 0;
  while (!(value & 1))
    {
      value >>= 1;
      ++pos;
    }
  return pos;
#endif
}

inline uint64_t clz64(uint64_t value)
{
  if (value == 0)
    {
      return 64;
    }
#if defined(__has_builtin) && __has_builtin(__builtin_clzl)
  static_assert(
      sizeof(unsigned long) == sizeof(uint64_t),
      "Calling __builtin_clzl on a uint64_t requires 64 bit unsigned long");
  return (uint64_t)__builtin_clzl(value);
#else
#error "Unimplemented clz64"
#endif
}

}  // namespace detail
}  // namespace

// probably need scope as a template parameter on this
// not a general purpose bitmap
template <size_t N, size_t scope = __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>
struct slot_bitmap
{
  static_assert(N != 0, "");
  static_assert(N != SIZE_MAX, "Used as a sentinel");
  static_assert(N % 64 == 0, "Size must be multiple of 64");

  constexpr slot_bitmap() = default;

  static constexpr size_t size() { return N; }

  bool operator[](size_t i) const
  {
    size_t w = index_to_element(i);
    uint64_t d = load_relaxed(w);
    return detail::nthbitset64(d, index_to_subindex(i));
  }

  void dump() const
  {
    uint64_t w = N / 64;
    printf("Size %lu / words %lu\n", size(), w);
    for (uint64_t i = 0; i < w; i++)
      {
        printf("[%2lu]:", i);
        for (uint64_t j = 0; j < 64; j++)
          {
            if (j % 8 == 0)
              {
                printf(" ");
              }
            printf("%c", this->operator[](64 * i + j) ? '1' : '0');
          }
        printf("\n");
      }
  }

  // probably want a specialisation where the slot is required to be empty
  // initially
  bool claim_slot(size_t i);

  void release_slot(size_t i);

  size_t find_slot()  // SIZE_MAX if none available
  {
    // find a zero. May be worth inverting in order to find a set
    const size_t words = N / 64;
    for (size_t i = 0; i < words; i++)
      {
        uint64_t w = ~load_relaxed(i);
        if (w != 0)
          {
            static_assert(sizeof(unsigned long) == sizeof(uint64_t),
                          "Calling __builtin_ctzl on a uint64_t requires 64 "
                          "bit unsigned long");

            return 64 * i + (detail::ctz64(w));
          }
      }

    return SIZE_MAX;
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

  uint64_t load_relaxed(size_t i) const
  {
    // TODO: Can probably do this with very narrow scope
    return __opencl_atomic_load(&data[i], __ATOMIC_RELAXED,
                                __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
  }

  alignas(64) _Atomic uint64_t data[N / 64] = {};
};

template <size_t N, size_t scope>
bool slot_bitmap<N, scope>::claim_slot(size_t i)
{
  assert(i < N);
  size_t w = index_to_element(i);
  uint64_t subindex = index_to_subindex(i);

  uint64_t d = load_relaxed(w);
  _Atomic uint64_t* addr = &data[w];

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

      uint64_t compare = d;
      bool r = __opencl_atomic_compare_exchange_weak(
          addr, &compare, proposed, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED,
          __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);

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

template <size_t N, size_t scope>
void slot_bitmap<N, scope>::release_slot(size_t i)
{
  // programming error if the slot is not set on entry
  assert(i < N);
  size_t w = index_to_element(i);
  uint64_t subindex = index_to_subindex(i);

  uint64_t d = load_relaxed(w);
  _Atomic uint64_t* addr = &data[w];

  for (;;)
    {
      uint64_t proposed = detail::clearnthbit64(d, subindex);
      // error if set on entry
      // cas should not return false if the write succeeded
      assert(proposed != d);

      uint64_t compare = d;

      bool r = __opencl_atomic_compare_exchange_weak(
          addr, &compare, proposed, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED,
          __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);

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

}  // namespace hostrpc

#endif
