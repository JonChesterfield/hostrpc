#ifndef HOSTRPC_COMMON_H_INCLUDED
#define HOSTRPC_COMMON_H_INCLUDED

#include <stdint.h>

#include "../base_types.hpp"
#include "platform.hpp"

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
inline uint32_t index_to_element(uint32_t x)
{
  uint32_t wordBits = 8 * sizeof(Word);
  return x / wordBits;
}

template <typename Word>
inline uint32_t index_to_subindex(uint32_t x)
{
  uint32_t wordBits = 8 * sizeof(Word);
  return x % wordBits;
}

namespace bits
{
inline bool nthbitset(uint32_t x, uint32_t n)
{
  assert(n < 32);
  return x & (UINT32_C(1) << n);
}

inline bool nthbitset(uint64_t x, uint32_t n)
{
  assert(n < 64);
  return x & (UINT64_C(1) << n);
}

inline uint32_t setnthbit(uint32_t x, uint32_t n)
{
  assert(n < 32);
  return x | (UINT32_C(1) << n);
}

inline uint64_t setnthbit(uint64_t x, uint32_t n)
{
  assert(n < 64);
  return x | (UINT64_C(1) << n);
}

inline uint32_t clearnthbit(uint32_t x, uint32_t n)
{
  assert(n < 32);
  return x & ~(UINT32_C(1) << n);
}

inline uint64_t clearnthbit(uint64_t x, uint32_t n)
{
  assert(n < 64);
  return x & ~(UINT64_C(1) << n);
}

inline uint32_t ctz(uint32_t value)
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

inline uint32_t ctz(uint64_t value)
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

inline uint32_t clz(uint32_t value)
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

inline uint32_t clz(uint64_t value)
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

namespace detail
{
inline bool multiple_of_64(uint64_t x) { return (x % 64) == 0; }

inline uint64_t round_up_to_multiple_of_64(uint64_t x)
{
  return 64u * ((x + 63u) / 64u);
}

inline uint32_t setbitsrange32(uint32_t l, uint32_t h)
{
  uint32_t base = UINT32_MAX;
  uint32_t width = (h - l) + 1;
  base >>= (UINT64_C(31) & (UINT32_C(32) - width));
  base <<= (UINT64_C(31) & l);
  return base;
}

inline uint64_t setbitsrange64(uint32_t l, uint32_t h)
{
  uint64_t base = UINT64_MAX;
  uint32_t width = (h - l) + 1;
  // The &63 is eliminated by the backend for x86-64 as that's the
  // behaviour of the shift instruction.
  base >>= (UINT64_C(63) & (UINT64_C(64) - width));
  base <<= (UINT64_C(63) & l);
  return base;
}

}  // namespace detail
}  // namespace

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

template <bool HasFetchOpArg>
struct base
{
  static constexpr bool hasFetchOp() { return HasFetchOpArg; }
};

template <typename Word>
struct fine_grain : public base<false>
{
  using Ty = __attribute__((aligned(64))) HOSTRPC_ATOMIC(Word);
};

template <typename Word>
struct coarse_grain : public base<true>
{
#if defined(__AMDGCN__)
  using Ty = __attribute__((aligned(64))) __attribute__((address_space(1)))
  HOSTRPC_ATOMIC(Word);
#else
  using Ty = __attribute__((aligned(64))) HOSTRPC_ATOMIC(Word);
#endif
};

}  // namespace properties

template <typename Word, size_t scope, typename Prop>
struct slot_bitmap;

template <typename Word>
using message_bitmap = slot_bitmap<Word, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
                                   properties::fine_grain<Word>>;

template <typename Word>
using slot_bitmap_coarse = slot_bitmap<Word, __OPENCL_MEMORY_SCOPE_DEVICE,
                                       properties::coarse_grain<Word>>;

template <typename Word, size_t scope, typename Prop>
struct slot_bitmap
{
  using Ty = typename Prop::Ty;

  // could check the types, expecting uint64_t or uint32_t
  static_assert((sizeof(Word) == 8) || (sizeof(Word) == 4), "");

  constexpr size_t wordBits() { return 8 * sizeof(Word); }

  static_assert(sizeof(Word) == sizeof(HOSTRPC_ATOMIC(Word)), "");
  static_assert(sizeof(Word *) == 8, "");
  static_assert(sizeof(HOSTRPC_ATOMIC(Word) *) == 8, "");

  Ty *a;
  static constexpr uint32_t bits_per_slot() { return 1; }
  slot_bitmap() : a(nullptr) {}
  slot_bitmap(Ty *d) : a(d)
  {
    // can't necessarily write to a from this object. if the memory is on
    // the gpu, but this instance is being constructed on the gpu first,
    // then direct writes will fail. However, the data does need to be
    // zeroed for the bitmap to work.
  }

  ~slot_bitmap() {}
  Ty *data() { return a; }

  bool operator()(uint32_t size, uint32_t i, Word *loaded) const
  {
    uint32_t w = index_to_element<Word>(i);
    Word d = load_word(size, w);
    *loaded = d;
    return bits::nthbitset(d, index_to_subindex<Word>(i));
  }

  void dump(uint32_t size) const
  {
    Word loaded = 0;
    (void)loaded;
    uint32_t w = size / wordBits();
    printf("Size %lu / words %lu\n", size, w);
    for (uint32_t i = 0; i < w; i++)
      {
        printf("[%2lu]:", i);
        for (uint32_t j = 0; j < wordBits(); j++)
          {
            if (j % 8 == 0)
              {
                printf(" ");
              }
            printf("%c", this->operator()(size, wordBits() * i + j, &loaded)
                             ? '1'
                             : '0');
          }
        printf("\n");
      }
  }

  // assumes slot available
  Word claim_slot_returning_updated_word(uint32_t size, uint32_t i)
  {
    (void)size;
    assert(i < size);
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
  void release_slot(uint32_t size, uint32_t i)
  {
    release_slot_returning_updated_word(size, i);
  }

  Word release_slot_returning_updated_word(uint32_t size, uint32_t i)
  {
    (void)size;
    assert(i < size);
    uint32_t w = index_to_element<Word>(i);
    uint32_t subindex = index_to_subindex<Word>(i);
    assert(bits::nthbitset(load_word(size, w), subindex));

    // and with everything other than the slot set
    Word mask = ~bits::setnthbit((Word)0, subindex);

    Word before = fetch_and(w, mask);
    return before & mask;
  }

  Word load_word(uint32_t size, uint32_t w) const
  {
    (void)size;
    assert(w < (size / (8 * sizeof(Word))));
    return platform::atomic_load<Word, __ATOMIC_RELAXED, scope>(&a[w]);
  }

  bool cas(Word element, Word expect, Word replace, Word *loaded)
  {
    Ty *addr = &a[element];

    // this cas function is not used across devices by this library
    bool r = __opencl_atomic_compare_exchange_weak(
        addr, &expect, replace, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED, scope);

    // on success, bits in memory have been set to replace
    // on failure, value found is now in expect
    // if cas succeeded, the bits in memory matched what was expected and now
    // match replace if it failed, the above call wrote the bits found in memory
    // into expect
    *loaded = expect;
    return r;
  }

  // returns value from before the and/or
  // these are used on memory visible fromi all svm devices

 private:
  Word fetch_and(uint32_t element, Word mask)
  {
    Ty *addr = &a[element];
    if (Prop::hasFetchOp())
      {
        // This seems to work on amdgcn, but only with acquire. acq/rel fails
        return __opencl_atomic_fetch_and(addr, mask, __ATOMIC_ACQ_REL, scope);
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

            bool r = __opencl_atomic_compare_exchange_weak(
                addr, &current, replace, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
                scope);

            if (r)
              {
                return current;
              }
          }
      }
  }

  Word fetch_or(uint32_t element, Word mask)
  {
    Ty *addr = &a[element];
    if (Prop::hasFetchOp())
      {
        return __opencl_atomic_fetch_or(addr, mask, __ATOMIC_ACQ_REL, scope);
      }
    else
      {
        Word current =
            platform::atomic_load<Word, __ATOMIC_RELAXED, scope>(addr);
        while (1)
          {
            Word replace = current | mask;

            bool r = __opencl_atomic_compare_exchange_weak(
                addr, &current, replace, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
                scope);
            if (r)
              {
                return current;
              }
          }
      }
  }
};

template <typename Word>
struct lock_bitmap
{
  using Ty = typename properties::coarse_grain<Word>::Ty;
  static_assert(sizeof(Word) == sizeof(HOSTRPC_ATOMIC(Word)), "");
  static_assert(sizeof(HOSTRPC_ATOMIC(Word) *) == 8, "");
  Ty *a;
  static constexpr size_t bits_per_slot() { return 1; }

  lock_bitmap() : a(nullptr) {}
  lock_bitmap(Ty *d) : a(d) {}

  ~lock_bitmap() {}
  Ty *data() { return a; }

  // cas, true on success
  // on return true, loaded contains active[w]
  bool try_claim_empty_slot(uint32_t size, uint32_t slot,
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

        Word unexpected_contents;
        uint32_t r = platform::critical<uint32_t>(
            [&]() { return cas(w, d, proposed, &unexpected_contents); });

        unexpected_contents = platform::broadcast_master(unexpected_contents);

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
  void release_slot(uint32_t size, uint32_t i)
  {
    (void)size;
    assert(i < size);
    uint32_t w = index_to_element<Word>(i);
    uint32_t subindex = index_to_subindex<Word>(i);
    assert(bits::nthbitset(load_word(size, w), subindex));

    // and with everything other than the slot set
    Word mask = ~bits::setnthbit((Word)0, subindex);

    Ty *addr = &a[w];
    __opencl_atomic_fetch_and(addr, mask, __ATOMIC_ACQ_REL,
                              __OPENCL_MEMORY_SCOPE_DEVICE);
  }

  Word load_word(uint32_t size, uint32_t w) const
  {
    (void)size;
    assert(w < (size / (8 * sizeof(Word))));

    return platform::atomic_load<Word, __ATOMIC_RELAXED,
                                 __OPENCL_MEMORY_SCOPE_DEVICE>(&a[w]);
  }

 private:
  bool cas(uint32_t element, Word expect, Word replace, Word *loaded)
  {
    Ty *addr = &a[element];

    // this cas function is not used across devices by this library
    bool r = __opencl_atomic_compare_exchange_weak(
        addr, &expect, replace, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
        __OPENCL_MEMORY_SCOPE_DEVICE);

    // on success, bits in memory have been set to replace
    // on failure, value found is now in expect
    // if cas succeeded, the bits in memory matched what was expected and now
    // match replace if it failed, the above call wrote the bits found in memory
    // into expect
    *loaded = expect;
    return r;
  }
};

template <typename Word>
struct slot_bytemap
{
  static_assert(sizeof(Word) == 8, "Unimplemented for uint32_t");
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

  constexpr size_t wordBits() const { return 8 * sizeof(Word); }

  Ty *a;
  static constexpr size_t bits_per_slot() { return 8; }
  slot_bytemap() : a(nullptr) {}
  slot_bytemap(Ty *d) : a(d)
  {
    // can't necessarily write to a from this object. if the memory is on
    // the gpu, but this instance is being constructed on the gpu first,
    // then direct writes will fail. However, the data does need to be
    // zeroed for the bytemap to work.
  }

  ~slot_bytemap() {}
  Ty *data() { return a; }

  // assumes slot available
  void claim_slot(uint32_t size, uint32_t i) { write_byte<1>(size, i); }

  // assumes slot taken
  void release_slot(uint32_t size, uint32_t i) { write_byte<0>(size, i); }

  Word load_word(uint32_t size, uint32_t w) const
  {
    (void)size;
    (void)w;
    assert(w < (size / wordBits()));
    uint32_t i = w * wordBits();
    assert(i < size);
    return pack_words(&a[i]);
  }

 private:
  template <uint8_t v>
  void write_byte(uint32_t size, uint32_t i)
  {
    (void)size;
    assert(i < size);
#if SLOT_BYTEMAP_ATOMIC
    platform::atomic_store<uint8_t, __ATOMIC_RELAXED,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(&a[i], v);
#else
    a[i] = v;
#endif
  }

  uint8_t pack_word(uint64_t x) const
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

  uint64_t pack_words(Ty *data) const
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
void update_visible_from_staging(uint32_t size, uint32_t i,
                                 slot_bitmap<Word, Sscope, SProp> *staging,
                                 slot_bitmap<Word, Vscope, VProp> *visible,
                                 uint64_t *cas_fail_count,
                                 uint64_t *cas_help_count)
{
  // Write value ~InitialState to slot[i]

  assert((void *)visible != (void *)staging);
  assert(i < size);
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
void staged_claim_slot(uint32_t size, uint32_t i,
                       slot_bitmap<Word, Sscope, SProp> *staging,
                       slot_bitmap<Word, Vscope, VProp> *visible,
                       uint64_t *cas_fail_count, uint64_t *cas_help_count)
{
  update_visible_from_staging<false>(size, i, staging, visible, cas_fail_count,
                                     cas_help_count);
}

template <typename Word, size_t Sscope, typename SProp, size_t Vscope,
          typename VProp>
void staged_release_slot(uint32_t size, uint32_t i,
                         slot_bitmap<Word, Sscope, SProp> *staging,
                         slot_bitmap<Word, Vscope, VProp> *visible,
                         uint64_t *cas_fail_count, uint64_t *cas_help_count)
{
  update_visible_from_staging<true>(size, i, staging, visible, cas_fail_count,
                                    cas_help_count);
}

template <bool InitialState, typename Word, size_t Sscope, typename SProp>
void update_visible_from_staging(uint32_t size, uint32_t i,
                                 slot_bitmap<Word, Sscope, SProp> *staging,
                                 slot_bytemap<Word> *visible, uint64_t *,
                                 uint64_t *)
{
  // Write value ~InitialState to slot[i]

  assert((void *)visible != (void *)staging);
  assert(i < size);

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
void staged_claim_slot(uint32_t size, uint32_t i,
                       slot_bitmap<Word, Sscope, SProp> *staging,
                       slot_bytemap<Word> *visible, uint64_t *cas_fail_count,
                       uint64_t *cas_help_count)
{
  update_visible_from_staging<false>(size, i, staging, visible, cas_fail_count,
                                     cas_help_count);
}

template <typename Word, size_t Sscope, typename SProp>
void staged_release_slot(uint32_t size, uint32_t i,
                         slot_bitmap<Word, Sscope, SProp> *staging,
                         slot_bytemap<Word> *visible, uint64_t *cas_fail_count,
                         uint64_t *cas_help_count)
{
  update_visible_from_staging<true>(size, i, staging, visible, cas_fail_count,
                                    cas_help_count);
}

inline void step(HOSTRPC_ATOMIC(uint64_t) * steps_left)
{
  if (platform::atomic_load<uint64_t, __ATOMIC_ACQUIRE,
                            __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
          steps_left) == UINT64_MAX)
    {
      // Disable stepping
      return;
    }
  while (platform::atomic_load<uint64_t, __ATOMIC_ACQUIRE,
                               __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
             steps_left) == 0)
    {
      // Don't burn all the cpu waiting for a step
      platform::sleep_briefly();
    }

  steps_left--;
}

struct nop_stepper
{
  static void call(int, void *) {}
};

struct default_stepper_state
{
  default_stepper_state(HOSTRPC_ATOMIC(uint64_t) * val, bool show_step = false,
                        const char *name = "unknown")
      : val(val), show_step(show_step), name(name)
  {
  }

  HOSTRPC_ATOMIC(uint64_t) * val;
  bool show_step;
  const char *name;
};

struct default_stepper
{
  static void call(int line, void *v)
  {
    if (v)
      {
        default_stepper_state *state = static_cast<default_stepper_state *>(v);
        if (state->show_step)
          {
            printf("%s:%d: step\n", state->name, line);
          }
        (void)line;
        step(state->val);
      }
  }
};

// Depending on the host / client device and how they're connected together,
// copying data can be a no-op (shared memory, single buffer in use),
// pull and push from one of the two, routed through a third buffer

template <typename T>
struct copy_functor_interface
{
  // dst then src, memcpy style. Copies a single page
  static void push_from_client_to_server(page_t *dst, const page_t *src)
  {
    T::push_from_client_to_server_impl(dst, src);
  }
  static void pull_to_client_from_server(page_t *dst, const page_t *src)
  {
    T::pull_to_client_from_server_impl(dst, src);
  }

  static void push_from_server_to_client(page_t *dst, const page_t *src)
  {
    T::push_from_server_to_client_impl(dst, src);
  }
  static void pull_to_server_from_client(page_t *dst, const page_t *src)
  {
    T::pull_to_server_from_client_impl(dst, src);
  }

 private:
  friend T;
  copy_functor_interface() = default;

  // Default implementations are no-ops
  static void push_from_client_to_server_impl(page_t *, const page_t *) {}
  static void pull_to_client_from_server_impl(page_t *, const page_t *) {}
  static void push_from_server_to_client_impl(page_t *, const page_t *) {}
  static void pull_to_server_from_client_impl(page_t *, const page_t *) {}
};

struct copy_functor_memcpy_pull
    : public copy_functor_interface<copy_functor_memcpy_pull>
{
  friend struct copy_functor_interface<copy_functor_memcpy_pull>;

 private:
  static void pull_to_client_from_server_impl(page_t *dst, const page_t *src)
  {
    size_t N = sizeof(page_t);
    __builtin_memcpy(dst, src, N);
  }
  static void pull_to_server_from_client_impl(page_t *dst, const page_t *src)
  {
    size_t N = sizeof(page_t);
    __builtin_memcpy(dst, src, N);
  }
};

struct copy_functor_given_alias
    : public copy_functor_interface<copy_functor_given_alias>
{
  friend struct copy_functor_interface<copy_functor_given_alias>;

  static void push_from_client_to_server_impl(page_t *dst, const page_t *src)
  {
    assert(src == dst);
    (void)src;
    (void)dst;
  }
  static void pull_to_client_from_server_impl(page_t *dst, const page_t *src)
  {
    assert(src == dst);
    (void)src;
    (void)dst;
  }
  static void push_from_server_to_client_impl(page_t *dst, const page_t *src)
  {
    assert(src == dst);
    (void)src;
    (void)dst;
  }
  static void pull_to_server_from_client_impl(page_t *dst, const page_t *src)
  {
    assert(src == dst);
    (void)src;
    (void)dst;
  }
};

}  // namespace hostrpc

#endif
