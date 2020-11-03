#ifndef ALLOCATOR_HPP_INCLUDED
#define ALLOCATOR_HPP_INCLUDED

#include <stddef.h>

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
  raw allocate(size_t A, size_t N)
  {
    return static_cast<Base *>(this)->allocate(A, N);
  }

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

}  // namespace allocator
}  // namespace hostrpc

#endif
