#ifndef HOSTRPC_COMMON_H_INCLUDED
#define HOSTRPC_COMMON_H_INCLUDED

#include <stdatomic.h>
#include <stdint.h>

#include "memory.hpp"  // free, aligned alloc
#include "platform.hpp"

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
inline uint64_t index_to_element(uint64_t x) { return x / 64u; }

inline uint64_t index_to_subindex(uint64_t x) { return x % 64u; }

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

inline uint64_t setbitsrange64(uint64_t l, uint64_t h)
{
  uint64_t base = UINT64_MAX;
  uint64_t width = (h - l) + 1;
  // The &63 is eliminated by the backend for x86-64 as that's the
  // behaviour of the shift instruction.
  base >>= (UINT64_C(63) & (UINT64_C(64) - width));
  base <<= (UINT64_C(63) & l);
  return base;
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

template <size_t N>
struct cache
{
  cache() = default;

  void dump() { printf("[%lu] %lu/%lu/%lu\n", slot, i, o, a); }

  bool is(uint8_t s)
  {
    assert(s < 8);
    bool r = s == concat();
    if (!r) dump();
    return r;
  }

  void init(uint64_t s)
  {
    slot = s;
    word = index_to_element(s);
    subindex = index_to_subindex(s);
  }

  uint64_t i = 0;
  uint64_t o = 0;
  uint64_t a = 0;

  uint64_t slot = UINT64_MAX;
  uint64_t word = UINT64_MAX;
  uint64_t subindex = UINT64_MAX;

 private:
  uint8_t concat()
  {
    unsigned r = detail::nthbitset64(i, subindex) << 2 |
                 detail::nthbitset64(o, subindex) << 1 |
                 detail::nthbitset64(a, subindex) << 0;
    return static_cast<uint8_t>(r);
  }
};

// probably need scope as a template parameter on this
// not a general purpose bitmap

template <size_t N, size_t scope>
struct slot_bitmap;

template <size_t N>
using mailbox_t = slot_bitmap<N, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>;

template <size_t N>
using lockarray_t = slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE>;

template <size_t size>
struct slot_bitmap_data
{
  constexpr const static size_t align = 64;
  static_assert(size % 64 == 0, "Size must be multiple of 64");

  static slot_bitmap_data *alloc()
  {
    void *memory = ::aligned_alloc(align, size);
    assert(memory);
    return new (memory) slot_bitmap_data;
  }
  static void free(slot_bitmap_data *d)
  {
    d->~slot_bitmap_data();
    ::free(d);
  }
  alignas(align) _Atomic uint64_t data[size / 64];

  struct deleter
  {
    void operator()(slot_bitmap_data *d) { slot_bitmap_data::free(d); }
  };
};

template <size_t N, size_t scope>
struct slot_bitmap
{
  static_assert(N != 0, "");
  static_assert(N != SIZE_MAX, "Used as a sentinel");
  static_assert(N % 64 == 0, "Size must be multiple of 64");
  static constexpr size_t size() { return N; }
  static constexpr size_t words() { return N / 64; }

  static_assert(sizeof(uint64_t) == sizeof(_Atomic uint64_t), "");

  using slot_bitmap_data_t = slot_bitmap_data<size()>;

  static_assert(sizeof(slot_bitmap_data_t *) == 8, "");

  slot_bitmap_data_t *a;
  slot_bitmap(slot_bitmap_data_t *memory)
  {
    assert(memory);
    a = memory;
    for (size_t i = 0; i < words(); i++)
      {
        a->data[i] = 0;
      }
  }

  ~slot_bitmap() {}

  bool operator[](size_t i) const
  {
    size_t w = index_to_element(i);
    uint64_t d = load_word(w);
    return detail::nthbitset64(d, index_to_subindex(i));
  }

  bool operator()(size_t i, uint64_t *loaded) const
  {
    size_t w = index_to_element(i);
    uint64_t d = load_word(w);
    *loaded = d;
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
  bool try_claim_empty_slot(size_t i, uint64_t *);

  size_t try_claim_any_empty_slot()
  {
    uint64_t tmp;
    size_t slot = find_empty_slot();
    if (slot != SIZE_MAX)
      {
        if (try_claim_empty_slot(slot, &tmp))
          {
            return slot;
          }
      }
    return SIZE_MAX;
  }

  // not yet implemented, may be able to achieve the same
  // effect by toggling 0/1

  bool try_claim_full_slot(size_t) { return false; }
  size_t try_claim_any_full_slot() { return SIZE_MAX; }
  size_t find_full_slot() { return SIZE_MAX; }

  // assumes slot available
  void claim_slot(size_t i) { set_slot_given_already_clear(i); }
  uint64_t claim_slot_returning_updated_word(size_t i)
  {
    return set_slot_given_already_clear(i);
  }

  // assumes slot taken
  void release_slot(size_t i) { clear_slot_given_already_set(i); }
  uint64_t release_slot_returning_updated_word(size_t i)
  {
    return clear_slot_given_already_set(i);
  }

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
    return __opencl_atomic_load(&a->data[i], __ATOMIC_RELAXED, scope);
  }

  bool cas(uint64_t element, uint64_t expect, uint64_t replace)
  {
    uint64_t loaded;
    return cas(element, expect, replace, &loaded);
  }

  bool cas(uint64_t element, uint64_t expect, uint64_t replace,
           uint64_t *loaded)
  {
    _Atomic uint64_t *addr = &a->data[element];

    // cas is not used across devices by this library
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

  // returns value from before the and/or
  // these are used on memory visible fromi all svm devices
  uint64_t fetch_and(uint64_t element, uint64_t mask)
  {
    _Atomic uint64_t *addr = &a->data[element];
    return __opencl_atomic_fetch_and(addr, mask, __ATOMIC_ACQ_REL,
                                     __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
  }

  uint64_t fetch_or(uint64_t element, uint64_t mask)
  {
    _Atomic uint64_t *addr = &a->data[element];
    return __opencl_atomic_fetch_or(addr, mask, __ATOMIC_ACQ_REL,
                                    __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
  }

 private:
  uint64_t clear_slot_given_already_set(size_t i)
  {
    assert(i < N);
    size_t w = index_to_element(i);
    uint64_t subindex = index_to_subindex(i);
    assert(detail::nthbitset64(load_word(w), subindex));

    // and with everything other than the slot set
    uint64_t mask = ~detail::setnthbit64(0, subindex);

    uint64_t before = fetch_and(w, mask);
    assert(detail::nthbitset64(before, subindex));
    return before & mask;
  }

  uint64_t set_slot_given_already_clear(size_t i)
  {
    assert(i < N);
    size_t w = index_to_element(i);
    uint64_t subindex = index_to_subindex(i);
    assert(!detail::nthbitset64(load_word(w), subindex));

    // or with only the slot set
    uint64_t mask = detail::setnthbit64(0, subindex);

    uint64_t before = fetch_or(w, mask);
    assert(!detail::nthbitset64(before, subindex));
    return before | mask;
  }
};

// on return true, loaded contains active[w]
template <size_t N, size_t scope>
bool slot_bitmap<N, scope>::try_claim_empty_slot(size_t i, uint64_t *loaded)
{
  assert(i < N);
  size_t w = index_to_element(i);
  uint64_t subindex = index_to_subindex(i);

  uint64_t d = load_word(w);

  // printf("Slot %lu, w %lu, subindex %lu, d %lu\n", i, w, subindex, d);

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

      uint64_t unexpected_contents;
      bool r = cas(w, d, proposed, &unexpected_contents);
      if (r)
        {
          // success, got the lock, and active word was set to proposed
          *loaded = proposed;
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

struct slot_owner;
extern thread_local unsigned my_id;
extern slot_owner tracker;
struct slot_owner
{
  void dump()
  {
    for (unsigned i = 0; i < sizeof(slots) / sizeof(slots[0]); i++)
      {
        uint32_t v = __c11_atomic_load(&slots[i], __ATOMIC_SEQ_CST);
        printf("slot[%u] owned by %u\n", i, v);
      }
  }
  static const bool verbose = false;
  slot_owner()
  {
    for (unsigned i = 0; i < sizeof(slots) / sizeof(slots[0]); i++)
      {
        slots[i] = UINT32_MAX;
      }
  }
  _Atomic uint32_t slots[128];

  void claim(uint64_t slot) { claim(my_id, slot); }

  void release(uint64_t slot) { release(my_id, slot); }

  void claim(uint32_t id, uint64_t slot)
  {
    assert(slot < 128);
    uint32_t v = __c11_atomic_load(&slots[slot], __ATOMIC_SEQ_CST);
    if (v != UINT32_MAX)
      {
        printf("slot[%lu] <- %u failed, owned by %u\n", slot, id, slots[slot]);
      }
    assert(v == UINT32_MAX);
    if (verbose)
      {
        printf("slot[%lu] <- %u\n", slot, id);
      }
    __c11_atomic_store(&slots[slot], id, __ATOMIC_SEQ_CST);
  }

  void release(uint32_t id, uint64_t slot)
  {
    assert(slot < 128);
    uint32_t v = __c11_atomic_load(&slots[slot], __ATOMIC_SEQ_CST);
    if (v != id)
      {
        printf("slot[%lu] owned by %u, can't be freed by %u\n", slot,
               slots[slot], id);
      }
    assert(v == id);
    __c11_atomic_store(&slots[slot], UINT32_MAX, __ATOMIC_SEQ_CST);
    if (verbose)
      {
        printf("slot[%lu] -> free\n", slot);
      }
  }
};

template <bool enable>
struct malloc_lock
{
  malloc_lock() { init(); }

  void init()
  {
    if (!enable)
      {
        return;
      }
    held = 0;
    for (unsigned i = 0; i < 64; i++)
      {
        data[i] = nullptr;
      }
  }

  void lock(uint64_t x)
  {
    if (!enable)
      {
        return;
      }
    assert(held == 0);
    init();
    for (uint64_t i = 0; i < 64; i++)
      {
        data[i] = detail::nthbitset64(x, i) ? aligned_alloc(1, 1) : nullptr;
      }
    held = x;
  }

  void unlock(uint64_t x)
  {
    if (!enable)
      {
        return;
      }
    if (x != held)
      {
        printf("locked %lx but unlocking %lx\n", held, x);
      }
    assert(x == held);
    for (uint64_t i = 0; i < 64; i++)
      {
        if (detail::nthbitset64(x, i))
          {
            free(data[i]);
          }
      }
    init();
    assert(held == 0);
  }

  uint64_t held;
  void *data[64];
};

template <size_t N, typename G>
void try_garbage_collect_word(
    G garbage_bits, const mailbox_t<N> inbox, mailbox_t<N> outbox,
    slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE> active, uint64_t w)
{
  malloc_lock<true> lk;
  if (platform::is_master_lane())
    {
      uint64_t i = inbox.load_word(w);
      uint64_t o = outbox.load_word(w);
      uint64_t a = active.load_word(w);
      __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);

      uint64_t garbage_visible = garbage_bits(i, o);
      uint64_t garbage_available = garbage_visible & ~a;

#if 0
#if defined(__AMDGCN__)
  // Need to enable the other lanes, broadcast the result,
  // possibly return to the caller, then disable the other lanes
  // Leaving this until there are benchmarks for x86 to help choose
  // between early exit and cache line thrashing
#endif

  // if there's no garbage, this function will cas a with a, fetch-add ~0 twice
  // early exit means loading three cache lines then branch
  // continuing takes an exclusive lock on active[slot], outbox[slot]
  if (garbage_available == 0)
    {
      return;
    }
#endif

      // proposed set of locks is the current set and the ones we're claiming
      assert((garbage_available & a) == 0);  // disjoint
      uint64_t proposed = garbage_available | a;
      uint64_t result;
      bool won_cas = active.cas(w, a, proposed, &result);

#if 0
  // if (!won_cas) can return false immediately, or set locks_held to zero
  // if chosing to continue, fetch_and with ~0 is a potentially expensive no-op
  if (!won_cas)
    {
      return;
    }
#endif

      uint64_t locks_held = won_cas ? garbage_available : 0;

      // Some of the slots may have already been garbage collected
      // in which case some of the input may be work-available again
      i = inbox.load_word(w);
      o = outbox.load_word(w);
      __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);

      uint64_t garbage_and_locked = garbage_bits(i, o) & locks_held;

      // clear locked bits in outbox
      __c11_atomic_thread_fence(__ATOMIC_RELEASE);
      uint64_t before = outbox.fetch_and(w, ~garbage_and_locked);
      (void)before;

      // drop locks
      active.fetch_and(w, ~locks_held);
    }
}

inline void step(_Atomic(uint64_t) * steps_left)
{
  if (atomic_load(steps_left) == UINT64_MAX)
    {
      // Disable stepping
      return;
    }
  while (atomic_load(steps_left) == 0)
    {
      // Don't burn all the cpu waiting for a step
      platform::sleep_briefly();
    }

  steps_left--;
}

struct nop_stepper
{
  void operator()(int) {}
};

struct default_stepper
{
  default_stepper(_Atomic(uint64_t) * val, bool show_step = false,
                  const char *name = "unknown")
      : val(val), show_step(show_step), name(name)
  {
  }

  void operator()(int line)
  {
    if (show_step)
      {
        printf("%s:%d: step\n", name, line);
      }
    step(val);
  }
  _Atomic(uint64_t) * val;
  bool show_step;
  const char *name;
};

// Depending on the host / client device and how they're connected together,
// copying data can be a no-op (shared memory, single buffer in use),
// pull and push from one of the two, routed through a third buffer

template <typename T>
struct copy_functor_interface
{
  // Function type is that of memcpy, i.e. dst first, N in bytes

  void push_from_client_to_server(void *dst, const void *src, size_t N)
  {
    impl().push_from_client_to_server_impl(dst, src, N);
  }
  void pull_to_client_from_server(void *dst, const void *src, size_t N)
  {
    impl().pull_to_client_from_server_impl(dst, src, N);
  }

  void push_from_server_to_client(void *dst, const void *src, size_t N)
  {
    impl().push_from_server_to_client_impl(dst, src, N);
  }
  void pull_to_server_from_client(void *dst, const void *src, size_t N)
  {
    impl().pull_to_server_from_client_impl(dst, src, N);
  }

 private:
  friend T;
  copy_functor_interface() = default;
  T &impl() { return *static_cast<T *>(this); }

  // Default implementations are no-ops
  void push_from_client_to_server_impl(void *, const void *, size_t) {}
  void pull_to_client_from_server_impl(void *, const void *, size_t) {}
  void push_from_server_to_client_impl(void *, const void *, size_t) {}
  void pull_to_server_from_client_impl(void *, const void *, size_t) {}
};

struct copy_functor_memcpy_pull
    : public copy_functor_interface<copy_functor_memcpy_pull>
{
  friend struct copy_functor_interface<copy_functor_memcpy_pull>;

 private:
  void pull_to_client_from_server_impl(void *dst, const void *src, size_t N)
  {
    __builtin_memcpy(dst, src, N);
  }
  void pull_to_server_from_client_impl(void *dst, const void *src, size_t N)
  {
    __builtin_memcpy(dst, src, N);
  }
};

}  // namespace hostrpc

#endif
