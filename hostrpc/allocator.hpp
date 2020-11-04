#ifndef ALLOCATOR_HPP_INCLUDED
#define ALLOCATOR_HPP_INCLUDED

#include "detail/platform_detect.h"

#include <stddef.h>
#if (HOSTRPC_HOST)
#include <tuple>
#endif

namespace hostrpc
{
namespace allocator
{
typedef enum
{
  success = 0,
  failure = 1,
} status;

template <size_t Align, typename Base>
struct interface
{
  _Static_assert(Align != 0, "");
  _Static_assert((Align & (Align - 1)) == 0, "");  // align is power two

  // some data is cache line aligned. hsa page aligns by default.
  _Static_assert((Align >= 64) && (Align <= 4096), "Current assumption");

  static const constexpr size_t align = Align;
  struct local_t
  {
    local_t(void *p) : ptr(p) {}
    void *ptr;
  };

  struct remote_t
  {
    remote_t(void *p) : ptr(p) {}
    void *ptr;
  };

  struct raw;
  raw allocate(size_t N) { return static_cast<Base *>(this)->allocate(N); }

  struct raw
  {
    raw(void *p) : ptr(p) {}
    int destroy() { return Base::destroy(*this); }
    local_t local() { return Base::local(*this); }
    remote_t remote() { return Base::remote(*this); }
    void *ptr;
  };

 private:
  // The raw/local/remote conversion is a no-op for most allocators
  static local_t local(raw x) { return x.ptr; }
  static remote_t remote(raw x) { return x.ptr; }
};

#if (HOSTRPC_HOST)
template <typename... Args>
struct raw_store_t
{
  raw_store_t(Args... args) : s(std::forward<Args>(args)...){};

  status destroy() { return destroy_impl(s); }

 private:
  std::tuple<Args...> s;

  template <size_t I = 0, typename... P>
  typename std::enable_if<I == sizeof...(P), status>::type destroy_impl(
      std::tuple<Args...> &)
  {
    return success;
  }

  template <size_t I = 0, typename... P>
      typename std::enable_if <
      I<sizeof...(P), status>::type destroy_impl(std::tuple<Args...> &args)
  {
    status car = std::get<I>(args).destroy();
    status cdr = destroy_impl<I + 1>(args);
    return (car == success && cdr == success) ? success : failure;
  }
};

template <typename... Args>
raw_store_t<Args...> raw_store(Args... args)
{
  return raw_store_t<Args...>(std::forward<Args>(args)...);
}
#endif
}  // namespace allocator
}  // namespace hostrpc

#endif
