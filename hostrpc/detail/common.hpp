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

HOSTRPC_ANNOTATE inline uint32_t togglenthbit(uint32_t x, uint32_t n)
{
  assert(n < 32);
  return x ^ ~(UINT32_C(1) << n);
}

HOSTRPC_ANNOTATE inline uint64_t togglenthbit(uint64_t x, uint32_t n)
{
  assert(n < 64);
  return x ^ ~(UINT64_C(1) << n);
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
// Observation by Tom Scogland - if the value of the bit is known ahead of time,
// it can be set by fetch_add with the corresponding integer, and cleared with
// fetch_sub (thus fetch_add via complement). This is expected to be available
// over pcie.

// x64 has cas, fetch op
// amdgcn has cas, fetch on gpu and cas on pcie
// nvptx has cas, fetch on gpu
template <bool HasFetchOpArg, bool HasCasOpArg, bool HasAddOpArg>
struct base
{
  HOSTRPC_ANNOTATE static constexpr bool hasFetchOp() { return HasFetchOpArg; }
  HOSTRPC_ANNOTATE static constexpr bool hasCasOp() { return HasCasOpArg; }
  HOSTRPC_ANNOTATE static constexpr bool hasAddOp() { return HasAddOpArg; }
};

template <typename Word>
struct fine_grain : public base<false, true, false>
{
  using Ty = __attribute__((aligned(64))) HOSTRPC_ATOMIC(Word);
};

template <typename Word>
struct device_local : base<true, true, true>
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
using message_bitmap = slot_bitmap<Word, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
                                   properties::fine_grain<Word>>;

template <typename WordT, size_t scope, typename PropT>
struct slot_bitmap
{
  using Prop = PropT;
  using Ty = typename Prop::Ty;
  using Word = WordT;

  // could check the types, expecting uint64_t or uint32_t
  static_assert((sizeof(Word) == 8) || (sizeof(Word) == 4), "");

  HOSTRPC_ANNOTATE constexpr size_t wordBits() const
  {
    return 8 * sizeof(Word);
  }

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
    // the gpu, but this instance is being constructed on the gpu first,
    // then direct writes will fail. However, the data does need to be
    // zeroed for the bitmap to work.
  }
  HOSTRPC_ANNOTATE ~slot_bitmap() = default;

  HOSTRPC_ANNOTATE bool read_bit(uint32_t size, port_t i) const
  {
    uint32_t w = index_to_element<Word>(i);
    Word d = load_word(size, w);
    return bits::nthbitset(d, index_to_subindex<Word>(i));
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

  // claim / release / toggle assume this is the only possible writer to that
  // index

  // Returns value of bit before writing true to it
  template <bool KnownClearBefore>
  HOSTRPC_ANNOTATE bool set_slot(uint32_t size, port_t i)
  {
    assert(static_cast<uint32_t>(i) < size);
    uint32_t w = index_to_element<Word>(i);
    uint32_t subindex = index_to_subindex<Word>(i);

    (void)size;
    if (KnownClearBefore)
      {
        assert(!bits::nthbitset(load_word(size, w), subindex));
      }

    if (Prop::hasFetchOp())
      {
        Word mask = bits::setnthbit((Word)0, subindex);

        Ty *addr = &underlying[w];
        Word before = platform::atomic_fetch_or<Word, __ATOMIC_ACQ_REL, scope>(
            addr, mask);
        return bits::nthbitset(before, subindex);
      }
    else if (KnownClearBefore && Prop::hasAddOp())
      {
        Word addend = (Word)1 << subindex;
        Word before = fetch_add(w, addend);
        return bits::nthbitset(before, subindex);
      }
    else
      {
        // cas is currently hidden behind fetch_or but probably shouldn't be
        Word mask = bits::setnthbit((Word)0, subindex);
        Word before = fetch_or(w, mask);
        return bits::nthbitset(before, subindex);
      }
  }

  // assumes slot available
  HOSTRPC_ANNOTATE void claim_slot(uint32_t size, port_t i)
  {
    assert(static_cast<uint32_t>(i) < size);
    assert(!bits::nthbitset(load_word(size, index_to_element<Word>(i)),
                            index_to_subindex<Word>(i)));
    bool before = set_slot<true>(size, i);
    (void)before;
    assert(before == false);
  }

  // assumes slot taken
  HOSTRPC_ANNOTATE void release_slot(uint32_t size, port_t i)
  {
    (void)size;
    assert(static_cast<uint32_t>(i) < size);
    uint32_t w = index_to_element<Word>(i);
    uint32_t subindex = index_to_subindex<Word>(i);
    assert(bits::nthbitset(load_word(size, w), subindex));

    static_assert(Prop::hasCasOp(), "");

    if (Prop::hasFetchOp())
      {
        // and with everything other than the slot set
        Word mask = ~bits::setnthbit((Word)0, subindex);
        Word before = fetch_and(w, mask);
        assert(bits::nthbitset(before, subindex));
        (void)before;
      }
    else if (Prop::hasAddOp())
      {
        Word addend = 1 + ~((Word)1 << subindex);
        Word before = fetch_add(w, addend);
        assert(bits::nthbitset(before, subindex));
        (void)before;
      }
    else
      {
        // cas, indirectly
        Word mask = ~bits::setnthbit((Word)0, subindex);
        Word before = fetch_and(w, mask);
        assert(bits::nthbitset(before, subindex));
        (void)before;
      }
  }

  HOSTRPC_ANNOTATE void toggle_slot(uint32_t size, port_t i)
  {
    (void)size;
    assert(static_cast<uint32_t>(i) < size);
    uint32_t w = index_to_element<Word>(i);
    uint32_t subindex = index_to_subindex<Word>(i);
#ifndef NDEBUG
    bool bit_before = bits::nthbitset(load_word(size, w), subindex);
#endif
    // xor with only the slot set
    Word mask = bits::setnthbit((Word)0, subindex);

    Word before = fetch_xor(w, mask);
    assert(bit_before == bits::nthbitset(before, subindex));
    (void)before;
  }

  HOSTRPC_ANNOTATE Word load_word(uint32_t size, uint32_t w) const
  {
    (void)size;
    assert(w < (size / (8 * sizeof(Word))));
    Ty *addr = &underlying[w];
    return platform::atomic_load<Word, __ATOMIC_RELAXED, scope>(addr);
  }

  HOSTRPC_ANNOTATE bool cas(Word element, Word expect, Word replace,
                            Word *loaded)
  {
    Ty *addr = &underlying[element];
    return platform::atomic_compare_exchange_weak<Word, __ATOMIC_ACQ_REL,
                                                  scope>(addr, expect, replace,
                                                         loaded);
  }

  // returns value from before the and/or
  // these are used on memory visible from all svm devices

 private:
  template <typename Op>
  HOSTRPC_ANNOTATE Word fetch_op(uint32_t element, Word mask)
  {
    Ty *addr = &underlying[element];
    if (Prop::hasFetchOp())
      {
        // This seems to work on amdgcn, but only with acquire. acq/rel fails
        return Op::Atomic(addr, mask);
      }
    else
      {
        // load and atomic cas have similar cost across pcie, may be faster to
        // use a (usually wrong) initial guess instead of a load
        Word current =
            platform::atomic_load<Word, __ATOMIC_RELAXED, scope>(addr);
        while (1)
          {
            Word replace = Op::Simple(current, mask);
            Word loaded;

            bool r = cas(element, current, replace, &loaded);

            if (r)
              {
                return loaded;
              }

            // new best guess at what is in memory
            current = loaded;
          }
      }
  }

  HOSTRPC_ANNOTATE Word fetch_and(uint32_t element, Word mask)
  {
    struct And
    {
      HOSTRPC_ANNOTATE static Word Simple(Word lhs, Word rhs)
      {
        return lhs & rhs;
      }
      HOSTRPC_ANNOTATE static Word Atomic(Ty *addr, Word value)
      {
        return platform::atomic_fetch_and<Word, __ATOMIC_ACQ_REL, scope>(addr,
                                                                         value);
      }
    };
    return fetch_op<And>(element, mask);
  }

  HOSTRPC_ANNOTATE Word fetch_or(uint32_t element, Word mask)
  {
    struct Or
    {
      HOSTRPC_ANNOTATE static Word Simple(Word lhs, Word rhs)
      {
        return lhs | rhs;
      }
      HOSTRPC_ANNOTATE static Word Atomic(Ty *addr, Word value)
      {
        return platform::atomic_fetch_or<Word, __ATOMIC_ACQ_REL, scope>(addr,
                                                                        value);
      }
    };
    return fetch_op<Or>(element, mask);
  }

  HOSTRPC_ANNOTATE Word fetch_xor(uint32_t element, Word mask)
  {
    struct Xor
    {
      HOSTRPC_ANNOTATE static Word Simple(Word lhs, Word rhs)
      {
        return lhs ^ rhs;
      }
      HOSTRPC_ANNOTATE static Word Atomic(Ty *addr, Word value)
      {
        return platform::atomic_fetch_xor<Word, __ATOMIC_ACQ_REL, scope>(addr,
                                                                         value);
      }
    };
    return fetch_op<Xor>(element, mask);
  }

  HOSTRPC_ANNOTATE Word fetch_add(uint32_t element, Word value)
  {
    Ty *addr = &underlying[element];
    return platform::atomic_fetch_add<Word, __ATOMIC_ACQ_REL, scope>(addr,
                                                                     value);
  }
};

template <typename Word>
using mailbox_bitmap = slot_bitmap<Word, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
                                   typename properties::fine_grain<Word>>;

template <typename WordT, bool Inverted>
struct inbox_bitmap
{
  using Word = WordT;
  using mailbox_t = mailbox_bitmap<Word>;
  using Ty = typename mailbox_t::Prop::Ty;

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
  using Ty = typename mailbox_t::Prop::Ty;

 private:
  mailbox_t a;

 public:
  HOSTRPC_ANNOTATE outbox_bitmap() /*: a(nullptr)*/ {}
  HOSTRPC_ANNOTATE outbox_bitmap(mailbox_t d) : a(d) {}
  HOSTRPC_ANNOTATE ~outbox_bitmap() = default;

  HOSTRPC_ANNOTATE void dump(uint32_t size) const { a.dump(size); }

  HOSTRPC_ANNOTATE void claim_slot(uint32_t size, port_t i)
  {
    return a.claim_slot(size, i);
  }

  HOSTRPC_ANNOTATE void release_slot(uint32_t size, port_t i)
  {
    return a.release_slot(size, i);
  }

  HOSTRPC_ANNOTATE void toggle_slot(uint32_t size, port_t i)
  {
    return a.toggle_slot(size, i);
  }

  HOSTRPC_ANNOTATE Word load_word(uint32_t size, uint32_t w) const
  {
    return a.load_word(size, w);
  }
};

template <typename Word>
struct lock_bitmap
{
  using bitmap_t = slot_bitmap<Word, __OPENCL_MEMORY_SCOPE_DEVICE,
                               typename properties::device_local<Word>>;

  using Prop = typename bitmap_t::Prop;
  using Ty = typename Prop::Ty;
  static_assert(Prop::hasFetchOp(), "");
  static_assert(Prop::hasCasOp(), "");

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
    if (Prop::hasFetchOp())
      {
        return try_claim_empty_slot_nospin(active_threads, size, slot);
      }
    else
      {
        return try_claim_empty_slot_cas(active_threads, size, slot);
      }
  }

  HOSTRPC_ANNOTATE void release_slot(uint32_t size, port_t i)
  {
    a.release_slot(size, i);
  }

  HOSTRPC_ANNOTATE Word load_word(uint32_t size, uint32_t w) const
  {
    return a.load_word(size, w);
  }

 private:
  template <typename T>
  HOSTRPC_ANNOTATE bool try_claim_empty_slot_nospin(T active_threads,
                                                    uint32_t size,
                                                    uint32_t slot)
  {
    // requires hasFetchOp for correctness, need to refactor that
    // specifically this needs to hit fetchOr, not fetchAdd
    assert(slot < size);

    uint32_t before = 0;
    if (platform::is_master_lane(active_threads))
      {
        before = a.template set_slot<false>(size, static_cast<port_t>(slot))
                     ? 1u
                     : 0u;
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

  // cas, true on success
  template <typename T>
  HOSTRPC_ANNOTATE bool try_claim_empty_slot_cas(T active_threads,
                                                 uint32_t size, uint32_t slot)
  {
    assert(slot < size);
    uint32_t w = index_to_element<Word>(slot);
    uint32_t subindex = index_to_subindex<Word>(slot);

    Word d = load_word(size, w);

    // printf("Slot %lu, w %lu, subindex %lu, d %lu\n", i, w, subindex, d);
    for (;;)
      {
        // if the bit was already set then we've lost the race

        // can either check the bit is zero, or unconditionally set it and check
        // if this changed the value
        Word proposed = bits::setnthbit(d, subindex);
        if (proposed == d)
          {
            return false;
          }

        // If the bit is known zero, can use fetch_or to set it

        Word unexpected_contents = 0;
        uint32_t r = 0;
        if (platform::is_master_lane(active_threads))
          {
            r = a.cas(w, d, proposed, &unexpected_contents);
          }
        r = platform::broadcast_master(active_threads, r);
        unexpected_contents =
            platform::broadcast_master(active_threads, unexpected_contents);

        if (r)
          {
            // success, got the lock, and active word was set to proposed
            return true;
          }

        // cas failed. reasons:
        // we lost the slot
        // another slot in the same word changed
        // spurious

        // try again if the slot is still empty
        // may want a give up count / sleep or similar
        d = unexpected_contents;
      }
  }
};

// each platform defines platform::native_width(), presently either 1|32|64
// the application provides a function of type void (*)(port_t, page_t*) and
// is responsible for using get_lane_id or similar to iterate across the page
// the following defines an adapter by:
// (adapter holds references so must not outlive f)
template <typename Func>
HOSTRPC_ANNOTATE auto make_apply(Func &&f);
// Takes an object defining one of:
// void operator()(port_t port, page_t *page)
// void operator()
// (hostrpc::port_t, uint32_t call_number, uint64_t (&element)[8])
// void operator()
// (hostrpc::port_t, uint32_t call_number, uint64_t *element)
// and returns a callable object defining:
// void operator()(port_t port, page_t *page)
// which maps the passed function  across the rows in the page

namespace detail
{
enum class callback_type
{
  to_page,
  to_line,
  unknown,
};

template <typename T, size_t N>
class classify_callback_type
{
  template <typename U>
  static constexpr decltype(
      cxx::declval<U>().operator()(cxx::declval<hostrpc::port_t>(),
                                   cxx::declval<hostrpc::page_t *>()),
      callback_type())
  test(int)
  {
    return callback_type::to_page;
  }

  template <typename U>
  static constexpr decltype(
      cxx::declval<U>().operator()(cxx::declval<hostrpc::port_t>(),
                                   cxx::declval<uint32_t>(),
                                   cxx::declval<uint64_t *>()),
      callback_type())
  test(int)
  {
    return callback_type::to_line;
  }

  template <typename U>
  static constexpr decltype(
      cxx::declval<U>().operator()(cxx::declval<hostrpc::port_t>(),
                                   cxx::declval<uint32_t>(),
                                   cxx::declval<uint64_t (&)[N]>()),
      callback_type())
  test(int)
  {
    return callback_type::to_line;
  }

  // would use (...) as the worst match but opencl rejects it
  static constexpr callback_type test(long) { return callback_type::unknown; }

 public:
  static constexpr callback_type value() { return test<T>(0); }
};

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
constexpr apply_case classify_relative_width()
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

static_assert(
    classify_relative_width<page_t::width, platform::native_width()>() !=
        apply_case::nonintegral_ratio,
    "");

template <typename Func, apply_case c>
struct apply;

template <typename Func>
struct apply<Func, apply_case::same_width>
{
  Func &&f;
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
  Func &&f;
  HOSTRPC_ANNOTATE apply(Func &&f_) : f(cxx::forward<Func>(f_)) {}

  HOSTRPC_ANNOTATE void operator()(port_t port, page_t *page)
  {
    constexpr size_t ratio = page_t::width / platform::native_width();
    for (size_t step = 0; step < ratio; step++)
      {
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
  Func &&f;
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

template <typename Func, apply::apply_case, callback_type>
struct apply_function_iff_required;

template <typename Func, apply::apply_case C>
struct apply_function_iff_required<Func, C, callback_type::to_page>
{
  apply_function_iff_required(const apply_function_iff_required &) = delete;
  apply_function_iff_required(apply_function_iff_required &&) = default;

  Func &&f;
  HOSTRPC_ANNOTATE apply_function_iff_required(Func &&f_)
      : f(cxx::forward<Func>(f_))
  {
  }

  HOSTRPC_ANNOTATE void operator()(port_t port, page_t *page) { f(port, page); }
};

template <typename Func, apply::apply_case C>
struct apply_function_iff_required<Func, C, callback_type::to_line>
{
  apply_function_iff_required(const apply_function_iff_required &) = delete;
  apply_function_iff_required(apply_function_iff_required &&) = default;

  apply::apply<Func, C> f;
  HOSTRPC_ANNOTATE apply_function_iff_required(Func &&f_)
      : f(cxx::forward<Func>(f_))
  {
  }

  HOSTRPC_ANNOTATE void operator()(port_t port, page_t *page) { f(port, page); }
};

}  // namespace detail

// Example use:
// auto ApplyFill = hostrpc::make_apply<Fill>(cxx::forward<Fill>(fill));
// followed by passing ApplyFill around
// Unfortunately, opencl thwarts us here as well. The Func instance is sometimes
// on the stack, and that wins error: field may not be qualified with __private

template <typename Func>
HOSTRPC_ANNOTATE auto make_apply(Func &&f)
{
  // Work out some properties of Func
  constexpr size_t N = 8;
  constexpr auto CallbackType =
      detail::classify_callback_type<Func, N>::value();
  constexpr auto ApplyType =
      detail::apply::classify_relative_width<page_t::width,
                                             platform::native_width()>();

  // Need to be taking an operator() that acts on either pages or lines
  static_assert(CallbackType == detail::callback_type::to_page ||
                    CallbackType == detail::callback_type::to_line,
                "");

  // page & platform width need to be an integral ratio
  static_assert(ApplyType != detail::apply::apply_case::nonintegral_ratio, "");

  // build a class that does the mapping across lines if necessary
  auto r = detail::apply_function_iff_required<Func, ApplyType, CallbackType>{
      cxx::forward<Func>(f)};

  // check we've built something that takes a page at a time
  static_assert(detail::classify_callback_type<decltype(r), N>::value() ==
                    detail::callback_type::to_page,
                "");
  return cxx::move(r);
}

}  // namespace hostrpc

#endif
