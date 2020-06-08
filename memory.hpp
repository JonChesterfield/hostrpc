#ifndef HOSTRPC_MEMORY_HPP_INCLUDED
#define HOSTRPC_MEMORY_HPP_INCLUDED

// malloc, free, memcpy, possibly the atomic ops
// allocating for:
// device-local locks bitmap
// device-local or shared virtual memory page_t[] buffer
// mailboxes

#include <stddef.h>

// Placement new is declared in #include <new>, which is not available
// Strictly it takes std::size_t, but cstddef isn't available either
void *operator new(size_t size, void *ptr) noexcept;

// hsa (and possibly catch) include cstdlib

// stdlib.h not necessarily available either
// todo: wrap these in an indirection, targeting libc or hsa
void free(void *);
void *aligned_alloc(size_t alignment, size_t size);

namespace hostrpc
{
template <typename T>
struct allocator_functor_interface
{
  // Functions take alignment, bytes for now
  void *allocate_buffer(size_t A, size_t N)
  {
    return impl().allocate_buffer_impl(A, N);
  }
  void *allocate_outbox(size_t A, size_t N)
  {
    return impl().allocate_outbox_impl(A, N);
  }
  void *allocate_locks(size_t A, size_t N)
  {
    return impl().allocate_locks_impl(A, N);
  }

  void free_buffer(void *d, size_t N) { impl().free_buffer_impl(d, N); }
  void free_outbox(void *d, size_t N) { impl().free_outbox_impl(d, N); }
  void free_locks(void *d, size_t N) { impl().free_locks_impl(d, N); }

 private:
  friend T;
  allocator_functor_interface() = default;
  T &impl() { return *static_cast<T *>(this); }

  void *allocate_buffer_impl(size_t, size_t) = delete;
  void *allocate_outbox_impl(size_t, size_t) = delete;
  void *allocate_locks_impl(size_t, size_t) = delete;

  void free_buffer_impl(void *, size_t) = delete;
  void free_outbox_impl(void *, size_t) = delete;
  void free_locks_impl(void *, size_t) = delete;
};

// TODO: Move to memory_x64 or similar, stdlib.h is probably using glibc
// x64 can use the same allocator as client or server
// amdgcn may have bootstrapping problems there

struct allocator_functor_x64_x64
    : public allocator_functor_interface<allocator_functor_x64_x64>
{
  friend struct allocator_functor_interface<allocator_functor_x64_x64>;

 private:
  void *allocate_buffer_impl(size_t A, size_t N) { return aligned_alloc(A, N); }
  void *allocate_outbox_impl(size_t A, size_t N) { return aligned_alloc(A, N); }
  void *allocate_locks_impl(size_t A, size_t N) { return aligned_alloc(A, N); }

  void free_buffer_impl(void *d, size_t) { free(d); }
  void free_outbox_impl(void *d, size_t) { free(d); }
  void free_locks_impl(void *d, size_t) { free(d); }
};

}  // namespace hostrpc
#endif
