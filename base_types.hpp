#ifndef BASE_TYPES_HPP_INCLUDED
#define BASE_TYPES_HPP_INCLUDED

#include <stddef.h>
#include <stdint.h>

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

struct size_runtime
{
  size_runtime(size_t N) : SZ(N) {}
  size_t N() const { return SZ; }

 private:
  size_t SZ;
};

template <size_t SZ>
struct size_compiletime
{
  size_compiletime() {}
  size_compiletime(size_t) {}
  constexpr size_t N() const { return SZ; }
};

}  // namespace hostrpc

#endif
