#ifndef BASE_TYPES_HPP_INCLUDED
#define BASE_TYPES_HPP_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "detail/platform_detect.hpp"

#if HOSTRPC_HAVE_STDIO
#include <stdio.h>
#endif

namespace hostrpc
{
static constexpr size_t round8(size_t x) { return 8u * ((x + 7u) / 8u); }
_Static_assert(0 == round8(0), "");
_Static_assert(8 == round8(1), "");
_Static_assert(8 == round8(2), "");
_Static_assert(8 == round8(7), "");
_Static_assert(8 == round8(8), "");
_Static_assert(16 == round8(9), "");
_Static_assert(16 == round8(10), "");
_Static_assert(16 == round8(15), "");
_Static_assert(16 == round8(16), "");
_Static_assert(24 == round8(17), "");

static constexpr size_t round64(size_t x) { return 64u * ((x + 63u) / 64u); }
_Static_assert(0 == round64(0), "");
_Static_assert(64 == round64(1), "");
_Static_assert(64 == round64(2), "");
_Static_assert(64 == round64(63), "");
_Static_assert(64 == round64(64), "");
_Static_assert(128 == round64(65), "");
_Static_assert(128 == round64(127), "");
_Static_assert(128 == round64(128), "");
_Static_assert(192 == round64(129), "");
}  // namespace hostrpc

template <size_t lhs, size_t rhs>
HOSTRPC_ANNOTATE constexpr bool static_equal()
{
  static_assert(lhs == rhs, "");
  return lhs == rhs;
}

namespace hostrpc
{
struct cacheline_t
{
  alignas(64) uint64_t element[8];
};
static_assert(sizeof(cacheline_t) == 64, "");

struct page_t
{
  enum
  {
    width = 64
  };
  alignas(4096) cacheline_t cacheline[width];
};
static_assert(sizeof(page_t) == 4096, "");

namespace size_detail
{
template <uint64_t T>
struct bits
{
  enum : uint8_t
  {
    value = T <= UINT8_MAX    ? 8
            : T <= UINT16_MAX ? 16
            : T <= UINT32_MAX ? 32
                              : 64,
  };
};

template <uint8_t bits>
struct dispatch;

template <>
struct dispatch<8>
{
  using type = uint8_t;
};
template <>
struct dispatch<16>
{
  using type = uint16_t;
};

template <>
struct dispatch<32>
{
  using type = uint32_t;
};

template <>
struct dispatch<64>
{
  using type = uint64_t;
};

template <uint64_t V>
struct sufficientType
{
  using type = typename dispatch<bits<V>::value>::type;
};

}  // namespace size_detail

template <typename T>
struct fastint_runtime
{
 private:
  T SZ;

 public:
  using type = T;
  using selfType = fastint_runtime<type>;
  HOSTRPC_ANNOTATE constexpr fastint_runtime(type N) : SZ(hostrpc::round64(N))
  {
  }

  HOSTRPC_ANNOTATE constexpr fastint_runtime() : SZ(0) {}

  HOSTRPC_ANNOTATE constexpr type value() const { return SZ; }

  HOSTRPC_ANNOTATE constexpr operator type() const { return value(); }

  HOSTRPC_ANNOTATE constexpr selfType popcount() const
  {
    return __builtin_popcountl(value());
  }

  HOSTRPC_ANNOTATE constexpr selfType findFirstSet() const
  {
    return __builtin_ffsl(value());
  }

  // implicit type narrowing hazards here, for now assert on mismatch
  template <typename Y>
  HOSTRPC_ANNOTATE constexpr selfType bitwiseOr(Y y)
  {
    static_assert(sizeof(Y) == sizeof(type), "TODO");
    return value() | y;
  }

  template <typename Y>
  HOSTRPC_ANNOTATE constexpr selfType bitwiseAnd(Y y)
  {
    static_assert(sizeof(Y) == sizeof(type), "TODO");
    return value() & y;
  }
};

template <uint64_t V>
struct fastint_compiletime
{
  using type = typename size_detail::sufficientType<V>::type;

 private:
  HOSTRPC_ANNOTATE constexpr static fastint_runtime<type> rt()
  {
    return fastint_runtime<type>(value());
  }

  template <uint64_t Y>
  HOSTRPC_ANNOTATE constexpr static
      typename size_detail::sufficientType<Y>::type
      retype()
  {
    static_assert(
        sizeof(typename size_detail::sufficientType<Y>::type) == sizeof(type),
        "TODO");
    return Y;
  }

 public:
  HOSTRPC_ANNOTATE constexpr fastint_compiletime() {}

  HOSTRPC_ANNOTATE constexpr static type value() { return V; }

  HOSTRPC_ANNOTATE constexpr auto popcount() const
  {
    return fastint_compiletime<rt().popcount()>();
  }
  HOSTRPC_ANNOTATE constexpr auto findFirstSet() const
  {
    return fastint_compiletime<rt().findFirstSet()>();
  }

  template <uint64_t Y>
  HOSTRPC_ANNOTATE constexpr auto bitwiseOr() const
  {
    constexpr auto n = retype<Y>();
    static_assert(sizeof(n) == sizeof(type), "TODO");
    return fastint_compiletime<rt().bitwiseOr(n)>();
  }

  template <uint64_t Y>
  HOSTRPC_ANNOTATE constexpr auto bitwiseAnd() const
  {
    constexpr auto n = retype<Y>();
    static_assert(sizeof(n) == sizeof(type), "TODO");
    return fastint_compiletime<rt().bitwiseAnd(n)>();
  }
};

template <typename T>
using size_runtime = fastint_runtime<T>;

template <uint64_t S>
using size_compiletime = fastint_compiletime<hostrpc::round64(S)>;

template <size_t Size, size_t Align>
struct storage
{
  HOSTRPC_ANNOTATE static constexpr size_t size() { return Size; }
  HOSTRPC_ANNOTATE static constexpr size_t align() { return Align; }

  template <typename T>
  HOSTRPC_ANNOTATE T *open()
  {
    return __builtin_launder(reinterpret_cast<T *>(data));
  }

  // TODO: Allow move construct into storage
  template <typename T>
  HOSTRPC_ANNOTATE T *construct(T t)
  {
    return new (reinterpret_cast<T *>(data)) T(t);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void destroy()
  {
    open<T>()->~T();
  }

  alignas(Align) unsigned char data[Size];
};

struct client_counters
{
  enum : unsigned
  {
    cc_no_candidate_slot = 0,
    cc_missed_lock_on_candidate_slot = 1,
    cc_got_lock_after_work_done = 2,
    cc_waiting_for_result = 3,
    cc_cas_lock_fail = 4,
    cc_garbage_cas_fail = 5,
    cc_publish_cas_fail = 6,
    cc_finished_cas_fail = 7,

    cc_garbage_cas_help = 8,
    cc_publish_cas_help = 9,
    cc_finished_cas_help = 10,

    cc_total_count,

  };

  uint64_t state[cc_total_count];
  HOSTRPC_ANNOTATE client_counters()
  {
    for (unsigned i = 0; i < cc_total_count; i++)
      {
        state[i] = 0;
      }
  }

#if HOSTRPC_HAVE_STDIO
  HOSTRPC_ANNOTATE void dump() const
  {
    printf("CC: no_candidate_slot: %lu\n", state[cc_no_candidate_slot]);
    printf("CC: missed_lock_on_candidate_slot: %lu\n",
           state[cc_missed_lock_on_candidate_slot]);
    printf("CC: got_lock_after_work_done: %lu\n",
           state[cc_got_lock_after_work_done]);
    printf("CC: waiting_for_result: %lu\n", state[cc_waiting_for_result]);
    printf("CC: cas_lock_fail: %lu\n", state[cc_cas_lock_fail]);
    printf("CC: garbage_cas_fail: %lu\n", state[cc_garbage_cas_fail]);
    printf("CC: garbage_cas_help: %lu\n", state[cc_garbage_cas_help]);
    printf("CC: publish_fail: %lu\n", state[cc_publish_cas_fail]);
    printf("CC: publish_help: %lu\n", state[cc_publish_cas_help]);
    printf("CC: finished_fail: %lu\n", state[cc_finished_cas_fail]);
    printf("CC: finished_help: %lu\n", state[cc_finished_cas_help]);
  }
#endif
};

struct server_counters
{
  enum : unsigned
  {
    sc_no_candidate_bitmap = 0,
    sc_cas_lock_fail = 1,
    sc_missed_lock_on_candidate_bitmap = 2,
    sc_missed_lock_on_word = 3,

    sc_garbage_cas_fail = 4,
    sc_publish_cas_fail = 5,

    sc_garbage_cas_help = 6,
    sc_publish_cas_help = 7,
    sc_got_lock_after_work_done = 8,

    sc_total_count,
  };
  uint64_t state[sc_total_count];
  HOSTRPC_ANNOTATE server_counters()
  {
    for (unsigned i = 0; i < sc_total_count; i++)
      {
        state[i] = 0;
      }
  }

#if HOSTRPC_HAVE_STDIO
  HOSTRPC_ANNOTATE void dump() const
  {
    printf("SC: no_candidate_bitmap: %lu\n", state[sc_no_candidate_bitmap]);
    printf("SC: cas_lock_fail: %lu\n", state[sc_cas_lock_fail]);
    printf("SC: missed_lock_on_candidate_bitmap: %lu\n",
           state[sc_missed_lock_on_candidate_bitmap]);
    printf("SC: missed_lock_on_word: %lu\n", state[sc_missed_lock_on_word]);

    printf("SC: got_lock_after_work_done: %lu\n",
           state[sc_got_lock_after_work_done]);

    printf("SC: garbage_cas_fail: %lu\n", state[sc_garbage_cas_fail]);
    printf("SC: garbage_cas_help: %lu\n", state[sc_garbage_cas_help]);
    printf("SC: publish_fail: %lu\n", state[sc_publish_cas_fail]);
    printf("SC: publish_help: %lu\n", state[sc_publish_cas_help]);
  }
#endif
};
}  // namespace hostrpc

#endif
