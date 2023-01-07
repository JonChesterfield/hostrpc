#ifndef BASE_TYPES_HPP_INCLUDED
#define BASE_TYPES_HPP_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "detail/fastint.hpp"
#include "platform/detect.hpp"

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

template <typename T>
struct size_runtime : public fastint_runtime<T>
{
  using base = fastint_runtime<T>;
  size_runtime() : base() {}
  size_runtime(T x) : base(hostrpc::round64(x)) {}
};

template <uint64_t S>
struct size_compiletime : public fastint_compiletime<hostrpc::round64(S)>
{
  using base = fastint_compiletime<hostrpc::round64(S)>;
  using typename base::fastint_compiletime;
};

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


}  // namespace hostrpc

#endif
