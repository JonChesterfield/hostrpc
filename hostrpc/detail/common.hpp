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

// Need a bitmap which sets an element by CAS or fetch_or
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
HOSTRPC_ANNOTATE inline uint32_t index_to_subindex(uint32_t x)
{
  uint32_t wordBits = 8 * sizeof(Word);
  return x % wordBits;
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


template <typename Word, size_t scope>
struct slot_bitmap;

static_assert(__OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES !=
                  __OPENCL_MEMORY_SCOPE_DEVICE,
              "Expected these to be distinct");

template <typename WordT, size_t scope>
struct slot_bitmap
{
  // Would like this to be addrspace qualified but that's currently rejected by
  // most languages.
  using Ty = __attribute__((aligned(64))) HOSTRPC_ATOMIC(WordT);
  using Word = WordT;

  static constexpr bool system_scope()
  {
    return scope == __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES;
  }

  static constexpr bool device_scope()
  {
    return scope == __OPENCL_MEMORY_SCOPE_DEVICE;
  }

  static_assert(system_scope() || device_scope(), "");
  static_assert(system_scope() != device_scope(), "");

  // could check the types, expecting uint64_t or uint32_t
  static_assert((sizeof(Word) == 8) || (sizeof(Word) == 4), "");


  static_assert(sizeof(Word) == sizeof(HOSTRPC_ATOMIC(Word)), "");
  static_assert(sizeof(Word *) == 8, "");
  static_assert(sizeof(HOSTRPC_ATOMIC(Word) *) == 8, "");

 private:
  Ty *underlying;

 public:
  // allocate in coarse grain memory can be followed by placement new of
  // the default constructor, which means the default constructor can't write
  // anything
  HOSTRPC_ANNOTATE slot_bitmap() /*: underlying(nullptr)*/ {}
  HOSTRPC_ANNOTATE slot_bitmap(Ty *d) : underlying(d)
  {
    // can't necessarily write to a from this object. if the memory is on
    // a gpu, but this instance is being constructed on a cpu first,
    // then direct writes will fail. However, the data does need to be
    // zeroed for the bitmap to work.
  }
  HOSTRPC_ANNOTATE ~slot_bitmap() = default;

  // assumes slot available, i.e. it's 0 and we're writing a 1
  HOSTRPC_ANNOTATE void claim_slot(uint32_t size, uint32_t i)
  {
    (void)size;
    assert(i < size);
    assert(!bits::nthbitset(load_word(size, index_to_element<Word>(i)),
                            index_to_subindex<Word>(i)));

    uint32_t w = index_to_element<Word>(i);
    uint32_t subindex = index_to_subindex<Word>(i);

    Ty *addr = &underlying[w];
    Word before;
    if (system_scope())
      {
        Word addend = (Word)1 << subindex;
        before = platform::atomic_fetch_add<Word, __ATOMIC_ACQ_REL, scope>(
            addr, addend);
      }
    else
      {
        Word mask = bits::setnthbit((Word)0, subindex);
        before = platform::atomic_fetch_or<Word, __ATOMIC_ACQ_REL, scope>(addr,
                                                                          mask);
      }
    assert(!bits::nthbitset(before, subindex));
    (void)before;
  }

  // assumes slot taken before, i.e. it's 1 and we're writing a 0
  HOSTRPC_ANNOTATE void release_slot(uint32_t size, uint32_t i)
  {
    (void)size;
    assert(i < size);
    uint32_t w = index_to_element<Word>(i);
    uint32_t subindex = index_to_subindex<Word>(i);
    assert(bits::nthbitset(load_word(size, w), subindex));

    Ty *addr = &underlying[w];
    Word before;
    if (system_scope())
      {
        Word addend = 1 + ~((Word)1 << subindex);
        before = platform::atomic_fetch_add<Word, __ATOMIC_ACQ_REL, scope>(
            addr, addend);
      }
    else
      {
        // and with everything other than the slot set
        Word mask = ~bits::setnthbit((Word)0, subindex);

        before = platform::atomic_fetch_and<Word, __ATOMIC_ACQ_REL, scope>(
            addr, mask);
      }
    assert(bits::nthbitset(before, subindex));
    (void)before;
  }

  HOSTRPC_ANNOTATE Word load_word(uint32_t size, uint32_t w) const
  {
    (void)size;
    assert(w < (size / (8 * sizeof(Word))));
    Ty *addr = &underlying[w];
    return platform::atomic_load<Word, __ATOMIC_RELAXED, scope>(addr);
  }

  // Returns value of bit before writing true to it
  // Don't know the value before,
  HOSTRPC_ANNOTATE bool test_and_set_slot(uint32_t size, uint32_t i)
  {
    assert(i < size);
    uint32_t w = index_to_element<Word>(i);
    uint32_t subindex = index_to_subindex<Word>(i);
    (void)size;

    static_assert(device_scope(), "");

    Word mask = bits::setnthbit((Word)0, subindex);

    Ty *addr = &underlying[w];
    Word before =
        platform::atomic_fetch_or<Word, __ATOMIC_ACQ_REL, scope>(addr, mask);
    return bits::nthbitset(before, subindex);
  }
};

template <typename Word>
using mailbox_bitmap = slot_bitmap<Word, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>;

template <typename WordT, bool Inverted>
struct inbox_bitmap
{
  using Word = WordT;
  using mailbox_t = mailbox_bitmap<Word>;
  using Ty = typename mailbox_t::Ty;

 private:
  mailbox_t a;

 public:
  HOSTRPC_ANNOTATE inbox_bitmap() /*: a(nullptr)*/ {}
  HOSTRPC_ANNOTATE inbox_bitmap(mailbox_t d) : a(d) {}
  HOSTRPC_ANNOTATE ~inbox_bitmap() = default;

  HOSTRPC_ANNOTATE void dump(uint32_t size) const { a.dump(size); }

  HOSTRPC_ANNOTATE Word load_word(uint32_t size, uint32_t w) const
  {
    Word tmp = a.load_word(size, w);
    return Inverted ? ~tmp : tmp;
  }
};

template <typename WordT>
struct outbox_bitmap
{
  using Word = WordT;
  using mailbox_t = mailbox_bitmap<Word>;
  using Ty = typename mailbox_t::Ty;

 private:
  mailbox_t a;

 public:
  HOSTRPC_ANNOTATE outbox_bitmap() /*: a(nullptr)*/ {}
  HOSTRPC_ANNOTATE outbox_bitmap(mailbox_t d) : a(d) {}
  HOSTRPC_ANNOTATE ~outbox_bitmap() = default;

  HOSTRPC_ANNOTATE void dump(uint32_t size) const { a.dump(size); }

  HOSTRPC_ANNOTATE void claim_slot(uint32_t size, uint32_t i)
  {
    return a.claim_slot(size, i);
  }

  HOSTRPC_ANNOTATE void release_slot(uint32_t size, uint32_t i)
  {
    return a.release_slot(size, i);
  }

  HOSTRPC_ANNOTATE Word load_word(uint32_t size, uint32_t w) const
  {
    return a.load_word(size, w);
  }
};

template <typename Word>
struct lock_bitmap
{
  using bitmap_t = slot_bitmap<Word, __OPENCL_MEMORY_SCOPE_DEVICE>;

  using Ty = typename bitmap_t::Ty;

 private:
  bitmap_t a;

 public:
  HOSTRPC_ANNOTATE lock_bitmap() /*: a(nullptr)*/ {}
  HOSTRPC_ANNOTATE lock_bitmap(Ty *d) : a(d) {}
  HOSTRPC_ANNOTATE ~lock_bitmap() = default;

  HOSTRPC_ANNOTATE void dump(uint32_t size) const { a.dump(size); }

  template <typename T>
  HOSTRPC_ANNOTATE bool try_claim_empty_slot(T active_threads, uint32_t size,
                                             uint32_t slot)
  {
    // requires hasFetchOp for correctness, need to refactor that
    // specifically this needs to hit fetchOr, not fetchAdd
    assert(slot < size);

    uint32_t before = 0;
    if (platform::is_master_lane(active_threads))
      {
        // returns boolean true if the corresponding bit was set before this
        // call
        before = a.test_and_set_slot(size, slot) ? 1u : 0u;
      }
    before = platform::broadcast_master(active_threads, before);

    if (before)
      {
        // was already locked
        return false;
      }
    else
      {
        return true;
      }
  }

  HOSTRPC_ANNOTATE void release_slot(uint32_t size, uint32_t i)
  {
    a.release_slot(size, i);
  }

  HOSTRPC_ANNOTATE Word load_word(uint32_t size, uint32_t w) const
  {
    return a.load_word(size, w);
  }
};

}  // namespace hostrpc

#endif
