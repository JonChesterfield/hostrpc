#ifndef HOSTRPC_COMMON_H_INCLUDED
#define HOSTRPC_COMMON_H_INCLUDED

#include <stdint.h>

#include "../base_types.hpp"
#include "../platform.hpp"
#include "../platform/detect.hpp"
#include "cxx.hpp"

namespace hostrpc
{
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
template <typename Word>
HOSTRPC_ANNOTATE inline uint32_t index_to_element(uint32_t x)
{
  uint32_t wordBits = 8 * sizeof(Word);
  return x / wordBits;
}

template <typename Word>
HOSTRPC_ANNOTATE inline uint32_t index_to_element(port_t x)
{
  return index_to_element<Word>(static_cast<uint32_t>(x));
}

template <typename Word>
HOSTRPC_ANNOTATE inline uint32_t index_to_subindex(uint32_t x)
{
  uint32_t wordBits = 8 * sizeof(Word);
  return x % wordBits;
}

template <typename Word>
HOSTRPC_ANNOTATE inline uint32_t index_to_subindex(port_t x)
{
  return index_to_subindex<Word>(static_cast<uint32_t>(x));
}

namespace bits
{
HOSTRPC_ANNOTATE inline bool nthbitset(uint32_t x, uint32_t n)
{
  assert(n < 32);
  return x & (UINT32_C(1) << n);
}

HOSTRPC_ANNOTATE inline bool nthbitset(uint64_t x, uint32_t n)
{
  assert(n < 64);
  return x & (UINT64_C(1) << n);
}

HOSTRPC_ANNOTATE inline uint32_t setnthbit(uint32_t x, uint32_t n)
{
  assert(n < 32);
  return x | (UINT32_C(1) << n);
}

HOSTRPC_ANNOTATE inline uint64_t setnthbit(uint64_t x, uint32_t n)
{
  assert(n < 64);
  return x | (UINT64_C(1) << n);
}

HOSTRPC_ANNOTATE inline uint32_t clearnthbit(uint32_t x, uint32_t n)
{
  assert(n < 32);
  return x & ~(UINT32_C(1) << n);
}

HOSTRPC_ANNOTATE inline uint64_t clearnthbit(uint64_t x, uint32_t n)
{
  assert(n < 64);
  return x & ~(UINT64_C(1) << n);
}

HOSTRPC_ANNOTATE inline uint32_t ctz(uint32_t value)
{
  if (value == 0)
    {
      return 32;
    }
#if defined(__has_builtin)
#if __has_builtin(__builtin_ctz)
  static_assert(sizeof(unsigned) == sizeof(uint32_t),
                "Calling __builtin_ctz on a uint32_t requires 32 bit unsigned");
  return (uint32_t)__builtin_ctz(value);
#else
  uint32_t pos = 0;
  while (!(value & 1))
    {
      value >>= 1;
      ++pos;
    }
  return pos;
#endif
#endif
}

HOSTRPC_ANNOTATE inline uint32_t ctz(uint64_t value)
{
  if (value == 0)
    {
      return 64;
    }
#if defined(__has_builtin)
#if __has_builtin(__builtin_ctzl)
  static_assert(
      sizeof(unsigned long) == sizeof(uint64_t),
      "Calling __builtin_ctzl on a uint64_t requires 64 bit unsigned long");
  return (uint32_t)__builtin_ctzl(value);
#else
  uint32_t pos = 0;
  while (!(value & 1))
    {
      value >>= 1;
      ++pos;
    }
  return pos;
#endif
#endif
}

HOSTRPC_ANNOTATE inline uint32_t clz(uint32_t value)
{
  if (value == 0)
    {
      return 32;
    }
#if defined(__has_builtin)
#if __has_builtin(__builtin_clz)
  static_assert(
      sizeof(unsigned) == sizeof(uint32_t),
      "Calling __builtin_clzl on a uint32_t requires 32 bit unsigned");
  return (uint32_t)__builtin_clz(value);
#else
#error "Unimplemented clz(32)"
#endif
#endif
}

HOSTRPC_ANNOTATE inline uint32_t clz(uint64_t value)
{
  if (value == 0)
    {
      return 64;
    }
#if defined(__has_builtin)
#if __has_builtin(__builtin_clzl)
  static_assert(
      sizeof(unsigned long) == sizeof(uint64_t),
      "Calling __builtin_clzl on a uint64_t requires 64 bit unsigned long");
  return (uint32_t)__builtin_clzl(value);
#else
#error "Unimplemented clz(64)"
#endif
#endif
}

}  // namespace bits
}  // namespace
namespace detail
{
namespace
{
HOSTRPC_ANNOTATE inline bool multiple_of_64(uint64_t x)
{
  return (x % 64) == 0;
}

HOSTRPC_ANNOTATE inline uint64_t round_up_to_multiple_of_64(uint64_t x)
{
  return 64u * ((x + 63u) / 64u);
}

HOSTRPC_ANNOTATE inline uint32_t setbitsrange32(uint32_t l, uint32_t h)
{
  uint32_t base = UINT32_MAX;
  uint32_t width = (h - l) + 1;
  base >>= (UINT64_C(31) & (UINT32_C(32) - width));
  base <<= (UINT64_C(31) & l);
  return base;
}

HOSTRPC_ANNOTATE inline uint64_t setbitsrange64(uint32_t l, uint32_t h)
{
  uint64_t base = UINT64_MAX;
  uint32_t width = (h - l) + 1;
  // The &63 is eliminated by the backend for x86-64 as that's the
  // behaviour of the shift instruction.
  base >>= (UINT64_C(63) & (UINT64_C(64) - width));
  base <<= (UINT64_C(63) & l);
  return base;
}
}  // namespace
}  // namespace detail

namespace properties
{
// atomic operations on fine grained memory are limited to those that the
// pci-e bus supports. There is no cache involved to mask this - fetch_and on
// the gpu will silently do the wrong thing if the pci-e bus doesn't support
// it. That means using cas (or swap, or faa) to communicate or buffering. The
// fetch_and works fine on coarse grained memory, but multiple waves will
// clobber each other, leaving the flag flickering from the other device
// perspective. Can downgrade to swap fairly easily, which will be roughly as
// expensive as a load & store.

// x64 has cas, fetch op
// amdgcn has cas, fetch on gpu and cas on pcie
// nvptx has cas, fetch on gpu
template <bool HasFetchOpArg, bool HasCasOpArg>
struct base
{
  HOSTRPC_ANNOTATE static constexpr bool hasFetchOp() { return HasFetchOpArg; }
  HOSTRPC_ANNOTATE static constexpr bool hasCasOp() { return HasCasOpArg; }
};

template <typename Word>
struct fine_grain : public base<false, true>
{
  using Ty = __attribute__((aligned(64))) HOSTRPC_ATOMIC(Word);
};

template <typename Word>
struct device_local : base<true, true>
{
  using Ty = __attribute__((aligned(64)))
#if defined(__AMDGCN__) && !defined(__HIP__)
  // HIP errors on this, may want to mark the variable __device__
  __attribute__((address_space(1)))
#endif
  HOSTRPC_ATOMIC(Word);
};

}  // namespace properties

template <typename Word, size_t scope, typename Prop>
struct slot_bitmap;

template <typename Word>
struct slot_bytemap;

#if 1
// bytemap is working for persistent kernel but not for loader executable
// see: SLOT_BYTEMAP_ATOMIC
template <typename Word>
using message_bitmap = slot_bitmap<Word, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
                                   properties::fine_grain<Word>>;
#else
template <typename Word>
using message_bitmap = slot_bytemap<Word>;
#endif

template <typename Word>
using slot_bitmap_device_local = slot_bitmap<Word, __OPENCL_MEMORY_SCOPE_DEVICE,
                                             properties::device_local<Word>>;

template <typename Word, size_t scope, typename Prop>
struct slot_bitmap
{
  using Ty = typename Prop::Ty;

  // could check the types, expecting uint64_t or uint32_t
  static_assert((sizeof(Word) == 8) || (sizeof(Word) == 4), "");

  HOSTRPC_ANNOTATE constexpr size_t wordBits() const
  {
    return 8 * sizeof(Word);
  }

  static_assert(sizeof(Word) == sizeof(HOSTRPC_ATOMIC(Word)), "");
  static_assert(sizeof(Word *) == 8, "");
  static_assert(sizeof(HOSTRPC_ATOMIC(Word) *) == 8, "");

  Ty *a;
  HOSTRPC_ANNOTATE static constexpr uint32_t bits_per_slot() { return 1; }
  HOSTRPC_ANNOTATE slot_bitmap() : a(nullptr) {}
  HOSTRPC_ANNOTATE slot_bitmap(Ty *d) : a(d)
  {
    // can't necessarily write to a from this object. if the memory is on
    // the gpu, but this instance is being constructed on the gpu first,
    // then direct writes will fail. However, the data does need to be
    // zeroed for the bitmap to work.
  }

  HOSTRPC_ANNOTATE ~slot_bitmap() {}
  HOSTRPC_ANNOTATE Ty *data() { return a; }

  HOSTRPC_ANNOTATE bool read_bit(uint32_t size, port_t i, Word *loaded) const
  {
    uint32_t w = index_to_element<Word>(i);
    Word d = load_word(size, w);
    *loaded = d;
    return bits::nthbitset(d, index_to_subindex<Word>(i));
  }

  HOSTRPC_ANNOTATE bool read_bit(uint32_t size, port_t i) const
  {
    Word loaded;
    return read_bit(size, i, &loaded);
  }

  HOSTRPC_ANNOTATE void dump(uint32_t size) const
  {
    (void)size;
#if HOSTRPC_HAVE_STDIO
    uint32_t w = size / wordBits();
    // printf("Size %u / words %u\n", size, w);
    for (uint32_t i = 0; i < w; i++)
      {
        printf("[%2u]:", i);
        for (uint32_t j = 0; j < wordBits(); j++)
          {
            if (j % 8 == 0)
              {
                printf(" ");
              }
            printf("%c", read_bit(size, wordBits() * i + j) ? '1' : '0');
          }
        printf("\n");
      }
#endif
  }

  // assumes slot available
  HOSTRPC_ANNOTATE Word claim_slot_returning_updated_word(uint32_t size,
                                                          port_t i)
  {
    (void)size;
    assert(static_cast<uint32_t>(i) < size);
    uint32_t w = index_to_element<Word>(i);
    uint32_t subindex = index_to_subindex<Word>(i);
    assert(!bits::nthbitset(load_word(size, w), subindex));

    // or with only the slot set
    Word mask = bits::setnthbit((Word)0, subindex);

    Word before = fetch_or(w, mask);
    assert(!bits::nthbitset(before, subindex));
    return before | mask;
  }

  // assumes slot taken
  HOSTRPC_ANNOTATE void release_slot(uint32_t size, port_t i)
  {
    release_slot_returning_updated_word(size, i);
  }

  HOSTRPC_ANNOTATE Word release_slot_returning_updated_word(uint32_t size,
                                                            port_t i)
  {
    (void)size;
    assert(static_cast<uint32_t>(i) < size);
    uint32_t w = index_to_element<Word>(i);
    uint32_t subindex = index_to_subindex<Word>(i);
    assert(bits::nthbitset(load_word(size, w), subindex));

    // and with everything other than the slot set
    Word mask = ~bits::setnthbit((Word)0, subindex);

    Word before = fetch_and(w, mask);
    return before & mask;
  }

  HOSTRPC_ANNOTATE Word load_word(uint32_t size, uint32_t w) const
  {
    (void)size;
    assert(w < (size / (8 * sizeof(Word))));
    return platform::atomic_load<Word, __ATOMIC_RELAXED, scope>(&a[w]);
  }

  HOSTRPC_ANNOTATE bool cas(Word element, Word expect, Word replace,
                            Word *loaded)
  {
    Ty *addr = &a[element];

    // this cas function is not used across devices by this library
    return platform::atomic_compare_exchange_weak<Word, __ATOMIC_ACQ_REL,
                                                  scope>(addr, expect, replace,
                                                         loaded);
  }

  // returns value from before the and/or
  // these are used on memory visible fromi all svm devices

 private:
  HOSTRPC_ANNOTATE Word fetch_and(uint32_t element, Word mask)
  {
    Ty *addr = &a[element];
    if (Prop::hasFetchOp())
      {
        // This seems to work on amdgcn, but only with acquire. acq/rel fails
        return platform::atomic_fetch_and<Word, __ATOMIC_ACQ_REL, scope>(addr,
                                                                         mask);
      }
    else
      {
        // load and atomic cas have similar cost across pcie, may be faster to
        // use a (usually wrong) initial guess instead of a load
        Word current =
            platform::atomic_load<Word, __ATOMIC_RELAXED, scope>(addr);
        while (1)
          {
            Word replace = current & mask;

            Word loaded;
            bool r =
                platform::atomic_compare_exchange_weak<Word, __ATOMIC_ACQ_REL,
                                                       scope>(addr, current,
                                                              replace, &loaded);

            if (r)
              {
                return loaded;
              }
          }
      }
  }

  HOSTRPC_ANNOTATE Word fetch_or(uint32_t element, Word mask)
  {
    Ty *addr = &a[element];
    if (Prop::hasFetchOp())
      {
        return platform::atomic_fetch_or<Word, __ATOMIC_ACQ_REL, scope>(addr,
                                                                        mask);
      }
    else
      {
        Word current =
            platform::atomic_load<Word, __ATOMIC_RELAXED, scope>(addr);
        while (1)
          {
            Word replace = current | mask;

            Word loaded;
            bool r =
                platform::atomic_compare_exchange_weak<Word, __ATOMIC_ACQ_REL,
                                                       scope>(addr, current,
                                                              replace, &loaded);

            if (r)
              {
                return loaded;
              }
          }
      }
  }
};

template <typename Word>
struct lock_bitmap
{
  using Prop = typename properties::device_local<Word>;
  using Ty = typename Prop::Ty;
  static_assert(sizeof(Word) == sizeof(HOSTRPC_ATOMIC(Word)), "");
  static_assert(sizeof(HOSTRPC_ATOMIC(Word) *) == 8, "");
  static_assert(Prop::hasFetchOp(), "");
  static_assert(Prop::hasCasOp(), "");

 private:
  HOSTRPC_ANNOTATE constexpr size_t wordBits() const
  {
    return 8 * sizeof(Word);
  }

  HOSTRPC_ANNOTATE bool read_bit(uint32_t size, uint32_t i) const
  {
    uint32_t w = index_to_element<Word>(i);
    Word d = load_word(size, w);
    return bits::nthbitset(d, index_to_subindex<Word>(i));
  }

 public:
  Ty *a;
  HOSTRPC_ANNOTATE static constexpr size_t bits_per_slot() { return 1; }

  HOSTRPC_ANNOTATE lock_bitmap() : a(nullptr) {}
  HOSTRPC_ANNOTATE lock_bitmap(Ty *d) : a(d) {}

  HOSTRPC_ANNOTATE ~lock_bitmap() {}
  HOSTRPC_ANNOTATE Ty *data() { return a; }

  HOSTRPC_ANNOTATE void dump(uint32_t size) const
  {
    (void)size;
#if HOSTRPC_HAVE_STDIO
    uint32_t w = size / wordBits();
    // printf("Size %u / words %u\n", size, w);
    for (uint32_t i = 0; i < w; i++)
      {
        printf("[%2u]:", i);
        for (uint32_t j = 0; j < wordBits(); j++)
          {
            if (j % 8 == 0)
              {
                printf(" ");
              }
            printf("%c", read_bit(size, wordBits() * i + j) ? '1' : '0');
          }
        printf("\n");
      }
#endif
  }

  // cas, true on success
  // on return true, loaded contains active[w]
  template <typename T>
  HOSTRPC_ANNOTATE bool try_claim_empty_slot(T active_threads, uint32_t size,
                                             uint32_t slot,
                                             uint64_t *cas_fail_count)
  {
    assert(slot < size);
    uint32_t w = index_to_element<Word>(slot);
    uint32_t subindex = index_to_subindex<Word>(slot);

    Word d = load_word(size, w);

    // printf("Slot %lu, w %lu, subindex %lu, d %lu\n", i, w, subindex, d);
    uint64_t local_fail_count = 0;
    for (;;)
      {
        // if the bit was already set then we've lost the race

        // can either check the bit is zero, or unconditionally set it and check
        // if this changed the value
        Word proposed = bits::setnthbit(d, subindex);
        if (proposed == d)
          {
            *cas_fail_count = *cas_fail_count + local_fail_count;
            return false;
          }

        // If the bit is known zero, can use fetch_or to set it

        Word unexpected_contents = 0;
        uint32_t r = 0;
        if (platform::is_master_lane(active_threads))
          {
            r = cas(w, d, proposed, &unexpected_contents);
          }
        r = platform::broadcast_master(active_threads, r);
        unexpected_contents =
            platform::broadcast_master(active_threads, unexpected_contents);

        if (r)
          {
            // success, got the lock, and active word was set to proposed
            *cas_fail_count = *cas_fail_count + local_fail_count;
            return true;
          }

        local_fail_count++;
        // cas failed. reasons:
        // we lost the slot
        // another slot in the same word changed
        // spurious

        // try again if the slot is still empty
        // may want a give up count / sleep or similar
        d = unexpected_contents;
      }
  }

  // assumes slot taken
  HOSTRPC_ANNOTATE void release_slot(uint32_t size, port_t i)
  {
    (void)size;
    assert(static_cast<uint32_t>(i) < size);
    uint32_t w = index_to_element<Word>(i);
    uint32_t subindex = index_to_subindex<Word>(i);
    assert(bits::nthbitset(load_word(size, w), subindex));

    // and with everything other than the slot set
    Word mask = ~bits::setnthbit((Word)0, subindex);

    Ty *addr = &a[w];
    platform::atomic_fetch_and<Word, __ATOMIC_ACQ_REL,
                               __OPENCL_MEMORY_SCOPE_DEVICE>(addr, mask);
  }

  HOSTRPC_ANNOTATE Word load_word(uint32_t size, uint32_t w) const
  {
    (void)size;
    assert(w < (size / (8 * sizeof(Word))));

    return platform::atomic_load<Word, __ATOMIC_RELAXED,
                                 __OPENCL_MEMORY_SCOPE_DEVICE>(&a[w]);
  }

 private:
  HOSTRPC_ANNOTATE bool cas(uint32_t element, Word expect, Word replace,
                            Word *loaded)
  {
    Ty *addr = &a[element];
    // this cas function is not used across devices by this library
    return platform::atomic_compare_exchange_weak<Word, __ATOMIC_ACQ_REL,
                                                  __OPENCL_MEMORY_SCOPE_DEVICE>(
        addr, expect, replace, loaded);
  }
};

template <typename Word>
struct slot_bytemap
{
  static_assert(sizeof(Word) == 8, "Unimplemented for uint32_t");
  // gcn back end fails to select AtomicStore<(store syncscope("one-as")
#define SLOT_BYTEMAP_ATOMIC 0

  // assumes sizeof a is a multiple of 64, may be worth passing size to the
  // constructor and asserting
#if SLOT_BYTEMAP_ATOMIC
  using Ty = __attribute__((aligned(64))) HOSTRPC_ATOMIC(uint8_t);
  using AliasingWordTy = __attribute__((aligned(64))) __attribute__((may_alias))
  HOSTRPC_ATOMIC(Word);
  static_assert(sizeof(uint8_t) == sizeof(HOSTRPC_ATOMIC(uint8_t)), "");
  static_assert(sizeof(HOSTRPC_ATOMIC(uint8_t)) == 1, "");
#else
  using Ty = __attribute__((aligned(64))) uint8_t;
  using AliasingWordTy =
      __attribute__((aligned(64))) __attribute__((may_alias)) Word;
#endif

  HOSTRPC_ANNOTATE constexpr size_t wordBits() const
  {
    return 8 * sizeof(Word);
  }

  Ty *a;
  HOSTRPC_ANNOTATE static constexpr size_t bits_per_slot() { return 8; }
  HOSTRPC_ANNOTATE slot_bytemap() : a(nullptr) {}
  HOSTRPC_ANNOTATE slot_bytemap(Ty *d) : a(d)
  {
    // can't necessarily write to a from this object. if the memory is on
    // the gpu, but this instance is being constructed on the gpu first,
    // then direct writes will fail. However, the data does need to be
    // zeroed for the bytemap to work.
  }

  HOSTRPC_ANNOTATE ~slot_bytemap() {}
  HOSTRPC_ANNOTATE Ty *data() { return a; }

  // assumes slot available
  HOSTRPC_ANNOTATE void claim_slot(uint32_t size, port_t i)
  {
    write_byte<1>(size, i);
  }

  // assumes slot taken
  HOSTRPC_ANNOTATE void release_slot(uint32_t size, port_t i)
  {
    write_byte<0>(size, i);
  }

  HOSTRPC_ANNOTATE
  Word load_word(uint32_t size, uint32_t w) const
  {
    (void)size;
    (void)w;
    assert(w < (size / wordBits()));
    uint32_t i = w * wordBits();
    assert(i < size);
    return pack_words(&a[i]);
  }

  HOSTRPC_ANNOTATE bool operator()(uint32_t size, uint32_t i,
                                   Word *loaded) const
  {
    // TODO: Works iff load_word matches bitmap
    uint32_t w = index_to_element<Word>(i);
    Word d = load_word(size, w);
    *loaded = d;
    return bits::nthbitset(d, index_to_subindex<Word>(i));
  }

 private:
  template <uint8_t v>
  HOSTRPC_ANNOTATE void write_byte(uint32_t size, port_t port)
  {
    uint32_t i = static_cast<uint32_t>(port);
    (void)size;
    assert(i < size);
#if SLOT_BYTEMAP_ATOMIC
    platform::atomic_store<uint8_t, __ATOMIC_RELAXED,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(&a[i], v);
#else
    a[i] = v;
#endif
  }

  HOSTRPC_ANNOTATE uint8_t pack_word(uint64_t x) const
  {
    // x = 0000000h 0000000g 0000000f 0000000e 0000000d 0000000c 0000000b
    // 0000000a
    uint64_t m = x * UINT64_C(0x102040810204080);
    // m = hgfedcba -------- -------- -------- -------- -------- --------
    // --------
    uint64_t r = m >> 56u;
    // r = 00000000 00000000 00000000 00000000 00000000 00000000 00000000
    // hgfedcba
    return r;
  }

  HOSTRPC_ANNOTATE uint8_t pack_word(uint32_t x) const
  {
    // x = 0000000d 0000000c 0000000b 0000000a
    uint32_t m = x * UINT64_C(0x10204080);
    // m = dcba---- -------- -------- --------
    uint32_t r = m >> 28u;
    // r = 00000000 00000000 00000000 0000dcba
    return r;
  }

  HOSTRPC_ANNOTATE uint64_t pack_words(Ty *data) const
  {
    static_assert(sizeof(Word) == 8, "");
    AliasingWordTy *words = (AliasingWordTy *)data;
    uint64_t res = 0;
    for (unsigned i = 0; i < 8; i++)
      {
        AliasingWordTy w = words[i];  // should probably be a relaxed load
        res |= ((uint64_t)pack_word(w) & UINT8_C(0xff)) << 8 * i;
      }
    return res;
  }
};

template <bool InitialState, typename Word, size_t Sscope, typename SProp,
          size_t Vscope, typename VProp>
HOSTRPC_ANNOTATE void update_visible_from_staging(
    uint32_t size, port_t i, slot_bitmap<Word, Sscope, SProp> *staging,
    slot_bitmap<Word, Vscope, VProp> *visible, uint64_t *cas_fail_count,
    uint64_t *cas_help_count)
{
  // Write value ~InitialState to slot[i]

  assert((void *)visible != (void *)staging);
  assert(static_cast<uint32_t>(i) < size);
  const uint32_t w = index_to_element<Word>(i);
  const uint32_t subindex = index_to_subindex<Word>(i);

  // InitialState locked for staged_release, clear for staged_claim
  assert(InitialState ==
         bits::nthbitset(staging->load_word(size, w), subindex));
  assert(InitialState ==
         bits::nthbitset(visible->load_word(size, w), subindex));

  // (InitialState ? fetch_and : fetch_or) to update staging
  Word staged_result =
      InitialState ? staging->release_slot_returning_updated_word(size, i)
                   : staging->claim_slot_returning_updated_word(size, i);
  assert(!InitialState == bits::nthbitset(staged_result, subindex));

  // propose a value that could plausibly be in visible. can refactor to drop
  // the arithmetic
  Word guess = InitialState ? bits::setnthbit(staged_result, subindex)
                            : bits::clearnthbit(staged_result, subindex);

  // initialise the value with the latest view of staging that is already
  // available
  Word proposed = staged_result;

  uint64_t local_fail_count = 0;
  uint64_t local_help_count = 0;
  while (!visible->cas(w, guess, proposed, &guess))
    {
      local_fail_count++;
      if (!InitialState == bits::nthbitset(guess, subindex))
        {
          // Cas failed, but another thread has done our work
          local_help_count++;
          proposed = guess;
          break;
        }

      // Update our view of proposed and try again
      proposed = staging->load_word(size, w);
      assert(!InitialState == bits::nthbitset(proposed, subindex));
    }
  *cas_fail_count = *cas_fail_count + local_fail_count;
  *cas_help_count = *cas_help_count + local_help_count;

  assert(!InitialState ==
         bits::nthbitset(visible->load_word(size, w), subindex));
}

template <typename Word, size_t Sscope, typename SProp, size_t Vscope,
          typename VProp>
HOSTRPC_ANNOTATE void staged_claim_slot(
    uint32_t size, port_t i, slot_bitmap<Word, Sscope, SProp> *staging,
    slot_bitmap<Word, Vscope, VProp> *visible, uint64_t *cas_fail_count,
    uint64_t *cas_help_count)
{
  update_visible_from_staging<false>(size, i, staging, visible, cas_fail_count,
                                     cas_help_count);
}

template <typename Word, size_t Sscope, typename SProp, size_t Vscope,
          typename VProp>
HOSTRPC_ANNOTATE void staged_release_slot(
    uint32_t size, port_t i, slot_bitmap<Word, Sscope, SProp> *staging,
    slot_bitmap<Word, Vscope, VProp> *visible, uint64_t *cas_fail_count,
    uint64_t *cas_help_count)
{
  update_visible_from_staging<true>(size, i, staging, visible, cas_fail_count,
                                    cas_help_count);
}

template <bool InitialState, typename Word, size_t Sscope, typename SProp>
HOSTRPC_ANNOTATE void update_visible_from_staging(
    uint32_t size, port_t i, slot_bitmap<Word, Sscope, SProp> *staging,
    slot_bytemap<Word> *visible, uint64_t *, uint64_t *)
{
  // Write value ~InitialState to slot[i]

  assert((void *)visible != (void *)staging);
  assert(static_cast<uint32_t>(i) < size);

  // (InitialState ? fetch_and : fetch_or) to update staging
  if (InitialState)
    {
      staging->release_slot_returning_updated_word(size, i);
    }
  else
    {
      staging->claim_slot_returning_updated_word(size, i);
    }

  // Write single byte
  if (InitialState)
    {
      visible->release_slot(size, i);
    }
  else
    {
      visible->claim_slot(size, i);
    }
}

template <typename Word, size_t Sscope, typename SProp>
HOSTRPC_ANNOTATE void staged_claim_slot(
    uint32_t size, uint32_t i, slot_bitmap<Word, Sscope, SProp> *staging,
    slot_bytemap<Word> *visible, uint64_t *cas_fail_count,
    uint64_t *cas_help_count)
{
  update_visible_from_staging<false>(size, i, staging, visible, cas_fail_count,
                                     cas_help_count);
}

template <typename Word, size_t Sscope, typename SProp>
HOSTRPC_ANNOTATE void staged_release_slot(
    uint32_t size, uint32_t i, slot_bitmap<Word, Sscope, SProp> *staging,
    slot_bytemap<Word> *visible, uint64_t *cas_fail_count,
    uint64_t *cas_help_count)
{
  update_visible_from_staging<true>(size, i, staging, visible, cas_fail_count,
                                    cas_help_count);
}

// each platform defines platform::native_width(), presently either 1|32|64
// the application provides a function of type void (*)(port_t, page_t*) and
// is responsible for using get_lane_id or similar to iterate across the page

namespace detail
{
namespace apply
{
enum class apply_case
{
  same_width,
  page_wider,
  page_narrower,
  nonintegral_ratio,
};

constexpr bool divides_exactly(size_t x, size_t y)
{
  if ((x == 0) || (y == 0))
    {
      return false;
    }

  if (x == y)
    {
      return true;
    }

  if (x < y)
    {
      size_t ratio = y / x;
      bool exact = y == ratio * x;
      return exact;
    }
  else
    {
      return divides_exactly(y, x);
    }
}

// No zero
static_assert(!divides_exactly(1, 0), "");
static_assert(!divides_exactly(0, 1), "");
static_assert(!divides_exactly(0, 0), "");

// Multiples of one
static_assert(divides_exactly(1, 1), "");
static_assert(divides_exactly(2, 1), "");
static_assert(divides_exactly(1, 2), "");

// Multiples of not-one
static_assert(divides_exactly(2, 4), "");
static_assert(divides_exactly(4, 2), "");

// Not multiple
static_assert(!divides_exactly(5, 2), "");
static_assert(!divides_exactly(2, 5), "");

template <size_t page_width, size_t platform_width>
constexpr apply_case classify()
{
  if (page_width == platform_width)
    {
      return apply_case::same_width;
    }

  if (page_width < platform_width)
    {
      constexpr bool exact = divides_exactly(page_width, platform_width);
      if (exact)
        {
          return apply_case::page_narrower;
        }
    }

  if (page_width > platform_width)
    {
      constexpr bool exact = divides_exactly(page_width, platform_width);
      if (exact)
        {
          return apply_case::page_wider;
        }
    }

  return apply_case::nonintegral_ratio;
}

static_assert(platform::native_width() != 0, "");

static_assert(classify<page_t::width, platform::native_width()>() !=
                  apply_case::nonintegral_ratio,
              "");

template <typename Func, apply_case c>
struct apply;

template <typename Func>
struct apply<Func, apply_case::same_width>
{
  Func f;
  HOSTRPC_ANNOTATE apply(Func &&f_) : f(cxx::forward<Func>(f_)) {}

  HOSTRPC_ANNOTATE void operator()(port_t port, page_t *page)
  {
    auto id = platform::get_lane_id();
    hostrpc::cacheline_t *L = &page->cacheline[id];
    f(port, id, L->element);
  }
};

template <typename Func>
struct apply<Func, apply_case::page_wider>
{
  Func f;
  HOSTRPC_ANNOTATE apply(Func &&f_) : f(cxx::forward<Func>(f_)) {}

  HOSTRPC_ANNOTATE void operator()(port_t port, page_t *page)
  {
    constexpr size_t ratio = page_t::width / platform::native_width();
    static_assert(ratio != 1, "");
    static_assert(ratio * platform::native_width() == page_t::width, "");

    for (size_t step = 0; step < ratio; step++)
      {
        // todo: compile time id? requires unrolling loop
        uint32_t id =
            platform::get_lane_id().value() + step * platform::native_width();
        hostrpc::cacheline_t *L = &page->cacheline[id];
        f(port, id, L->element);
      }
  }
};

template <typename Func>
struct apply<Func, apply_case::page_narrower>
{
  Func f;
  HOSTRPC_ANNOTATE apply(Func &&f_) : f(cxx::forward<Func>(f_)) {}

  HOSTRPC_ANNOTATE void operator()(port_t port, page_t *page)
  {
    auto id = platform::get_lane_id();
    if (id < page_t::width)
      {
        hostrpc::cacheline_t *L = &page->cacheline[id];
        f(port, id, L->element);
      }
  }
};

}  // namespace apply
}  // namespace detail

template <typename Func>
HOSTRPC_ANNOTATE auto make_apply(Func &&f)
{
  using namespace detail::apply;
  // Takes an object defining:
  // void operator()(hostrpc::port_t, uint32_t call_number, uint64_t
  // (&element)[8]) and returns a callable object defining:
  // void operator()(port_t port, page_t *page)
  // which maps the element[8] function
  // across the rows in the page
  constexpr apply_case c = classify<page_t::width, platform::native_width()>();
  return apply<Func, c>{cxx::forward<Func>(f)};
}

}  // namespace hostrpc

#endif
