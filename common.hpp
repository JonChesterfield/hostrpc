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
struct slot_bitmap;

template <size_t N>
using mailbox_t = slot_bitmap<N, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>;
template <size_t N>
using cache_t = slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE>;

template <size_t N>
void update_cache(const mailbox_t<N>* mbox, cache_t<N>* cache);

template <size_t N, size_t scope>
struct slot_bitmap
{
  static_assert(N != 0, "");
  static_assert(N != SIZE_MAX, "Used as a sentinel");
  static_assert(N % 64 == 0, "Size must be multiple of 64");

  friend void update_cache(const mailbox_t<N>* mbox, cache_t<N>* cache);

  constexpr slot_bitmap() = default;

  static constexpr size_t size() { return N; }
  static constexpr size_t words() { return N/64;}
  
  bool operator[](size_t i) const
  {
    size_t w = index_to_element(i);
    uint64_t d = load_word(w);
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

  // cas, true on success
  bool try_claim_empty_slot(size_t i);




  size_t try_claim_any_empty_slot()
  {
    size_t slot = find_empty_slot();
    if (slot != SIZE_MAX)
      {
        if (try_claim_empty_slot(slot))
          {
            return slot;
          }
      }
    return SIZE_MAX;
  }


  // not yet implemented, may be able to achieve the same
  // effect by toggling 0/1

  bool try_claim_full_slot(size_t)
  {
    return false;
  }
  size_t try_claim_any_full_slot()
  {
    return SIZE_MAX;
  }
  size_t find_full_slot() { return SIZE_MAX; }

  
  // assumes slot available
  void claim_slot(size_t i) { set_slot_given_already_clear(i); }

  // assumes slot taken
  void release_slot(size_t i) { clear_slot_given_already_set(i); }


  size_t find_empty_slot()  // SIZE_MAX if none available
  {
    // find a zero. May be worth inverting in order to find a set
    const size_t words = N / 64;
    for (size_t i = 0; i < words; i++)
      {
        uint64_t w = ~load_word(i);
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

  uint64_t load_word(size_t i) const
  {
    assert(i < words());
    // TODO: Can probably do this with very narrow scope
    return __opencl_atomic_load(&data[i], __ATOMIC_SEQ_CST,
                                __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
  }


  bool cas(uint64_t element, uint64_t expect, uint64_t replace, uint64_t * loaded)
  {
    _Atomic uint64_t * addr = &data[element];
    bool r = __opencl_atomic_compare_exchange_weak(
                                                   addr, &expect, replace, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED,
                                                   __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);

    // if cas succeeded, the bits in memory matched what was expected
    // if it failed, the above call wrote the bits found in memoru into expect
    *loaded = expect;
      return r;
  }
  
private:
  void clear_slot_given_already_set(size_t i)
  {
    assert(i < N);
    size_t w = index_to_element(i);
    uint64_t subindex = index_to_subindex(i);
    assert(detail::nthbitset64(load_word(w), subindex));

    // and with everything other than the slot set
    uint64_t mask = ~detail::setnthbit64(0, subindex);

    uint64_t before =
        __opencl_atomic_fetch_and(&data[w], mask, __ATOMIC_SEQ_CST,
                                  __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
    (void)before;
    assert(detail::nthbitset64(before, subindex));
  }

  void set_slot_given_already_clear(size_t i)
  {
    assert(i < N);
    size_t w = index_to_element(i);
    uint64_t subindex = index_to_subindex(i);
    assert(!detail::nthbitset64(load_word(w), subindex));

    // or with only the slot set
    uint64_t mask = detail::setnthbit64(0, subindex);

    uint64_t before =
        __opencl_atomic_fetch_or(&data[w], mask, __ATOMIC_SEQ_CST,
                                 __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
    (void)before;
    assert(!detail::nthbitset64(before, subindex));
  }

  static uint64_t index_to_element(uint64_t x)
  {
    assert(x < size());
    uint64_t r = x / 64u;
    assert(r < words());
    return r;
  }

  static uint64_t index_to_subindex(uint64_t x)
  {
    assert(x < size());
    return x % 64u;
  }

  alignas(64) _Atomic uint64_t data[words()] = {};
};

template <size_t N, size_t scope>
bool slot_bitmap<N, scope>::try_claim_empty_slot(size_t i)
{
  assert(i < N);
  size_t w = index_to_element(i);
  uint64_t subindex = index_to_subindex(i);

  uint64_t d = load_word(w);

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

      // If the bit is known zero, can use fetch_or to set it

      uint64_t compare = d;
      bool r = cas(w, compare, proposed, &compare);

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

template <size_t N>
void update_cache(const mailbox_t<N>* mbox, cache_t<N>* cache)
{
  for (size_t i = 0; i < N / 64; i++)
    {
      uint64_t l = __opencl_atomic_load(&mbox->data[i], __ATOMIC_ACQUIRE,
                                        __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
      __opencl_atomic_store(&cache->data[i], l, __ATOMIC_RELEASE,
                            __OPENCL_MEMORY_SCOPE_DEVICE);
    }
}

struct nop_stepper
{
  void operator()(int) {}
};
  
  
}  // namespace hostrpc

#endif
