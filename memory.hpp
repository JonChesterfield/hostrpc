#ifndef HOSTRPC_MEMORY_HPP_INCLUDED
#define HOSTRPC_MEMORY_HPP_INCLUDED

// malloc, free, memcpy, possibly the atomic ops
// allocating for:
// device-local locks bitmap
// device-local or shared virtual memory page_t[] buffer
// mailboxes

#include <stddef.h>

namespace hostrpc
{
template <typename T>
struct copy_functor_interface
{
  void push_from_client_to_server(void *dst, void *src, size_t N)
  {
    impl().push_from_client_to_server_impl(dst, src, N);
  }
  void pull_to_client_from_server(void *dst, void *src, size_t N)
  {
    impl().pull_to_client_from_server(dst, src, N);
  }

  void push_from_server_to_client(void *dst, void *src, size_t N)
  {
    impl().push_from_server_to_client(dst, src, N);
  }
  void pull_to_server_from_client(void *dst, void *src, size_t N)
  {
    impl().pull_to_server_from_client(dst, src, N);
  }

 private:
  friend T;
  copy_functor_interface() = default;
  T &impl() { return *static_cast<T *>(this); }

  // Default implementations are no-ops
  void push_from_client_to_server_impl(void *, void *, size_t) {}
  void pull_to_client_from_server_impl(void *, void *, size_t) {}
  void push_from_server_to_client_impl(void *, void *, size_t) {}
  void pull_to_server_from_client_impl(void *, void *, size_t) {}
};

struct copy_functor_x64_x64 : public copy_functor_interface<copy_functor_x64_x64>
{
  void pull_to_client_from_server_impl(void *dst, void *src, size_t N)
  {
    __builtin_memcpy(dst, src, N);
  }
  void pull_to_server_from_client_impl(void *dst, void *src, size_t N)
  {
    __builtin_memcpy(dst, src, N);
  }
};

}  // namespace hostrpc
#endif
