#ifndef ALLOCATOR_HPP_INCLUDED
#define ALLOCATOR_HPP_INCLUDED

#include "detail/platform_detect.h"

#include <stddef.h>
#include <stdint.h>

#if HOSTRPC_HAVE_STDIO
#include <stdio.h>
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
    HOSTRPC_ANNOTATE local_t(void *p) : ptr(p) {}
    HOSTRPC_ANNOTATE operator void *() { return ptr; }
    void *ptr;
  };

  struct remote_t
  {
    HOSTRPC_ANNOTATE remote_t(void *p) : ptr(p) {}
    HOSTRPC_ANNOTATE operator void *() { return ptr; }
    void *ptr;
  };

  struct raw;
  HOSTRPC_ANNOTATE raw allocate(size_t N)
  {
    return static_cast<Base *>(this)->allocate(N);
  }

  struct raw
  {
    HOSTRPC_ANNOTATE raw() : ptr(nullptr) {}
    HOSTRPC_ANNOTATE raw(void *p) : ptr(p) {}

    // Move only makes it easier to avoid calling free twice on the same pointer
    HOSTRPC_ANNOTATE raw(const raw &) = delete;
    HOSTRPC_ANNOTATE raw(raw &&other) : ptr(other.ptr) { other.ptr = nullptr; }
    HOSTRPC_ANNOTATE bool valid() { return ptr != nullptr; }
    HOSTRPC_ANNOTATE status destroy()
    {
      // assumes that destroy(nullptr) is a nop
      status res = Base::destroy(this);
      ptr = nullptr;
      return res;
    }
    HOSTRPC_ANNOTATE local_t local_ptr() { return Base::local_ptr(*this); }
    HOSTRPC_ANNOTATE remote_t remote_ptr() { return Base::remote_ptr(*this); }
    void *ptr;
    HOSTRPC_ANNOTATE void dump(const char *str)
    {
#if HOSTRPC_HAVE_STDIO
      fprintf(stderr, "%s: %p (%p, %p)\n", str, ptr, local_ptr().ptr,
              remote_ptr().ptr);
#else
      (void)str;
#endif
    }
  };

 private:
  // The raw/local/remote conversion is a no-op for most allocators
  HOSTRPC_ANNOTATE static local_t local_ptr(const raw &x) { return x.ptr; }
  HOSTRPC_ANNOTATE static remote_t remote_ptr(const raw &x) { return x.ptr; }
};

HOSTRPC_ANNOTATE inline void *align_pointer_up(void *ptr, size_t align)
{
  uint64_t top_misaligned = (uint64_t)ptr + align - 1;
  uint64_t aligned = top_misaligned & ~(align - 1);
  return (void *)aligned;
}

namespace host_libc_impl
{
HOSTRPC_ANNOTATE void *allocate(size_t align, size_t bytes);
HOSTRPC_ANNOTATE void deallocate(void *);
}  // namespace host_libc_impl

template <size_t Align>
struct host_libc : public interface<Align, host_libc<Align>>
{
  using typename interface<Align, host_libc<Align>>::raw;
  HOSTRPC_ANNOTATE host_libc() {}
  HOSTRPC_ANNOTATE raw allocate(size_t N)
  {
    return {host_libc_impl::allocate(Align, N)};
  }

  HOSTRPC_ANNOTATE static status destroy(raw *x)
  {
    host_libc_impl::deallocate(x->ptr);
    return success;
  }
};

namespace hsa_impl
{
HOSTRPC_ANNOTATE void *allocate(uint64_t hsa_region_t_handle, size_t align,
                                size_t bytes);

HOSTRPC_ANNOTATE void *allocate_fine_grain(size_t bytes);

HOSTRPC_ANNOTATE int deallocate(void *);

HOSTRPC_ANNOTATE int memsetzero_gpu(void *, size_t size);
}  // namespace hsa_impl

template <size_t Align>
struct hsa : public interface<Align, hsa<Align>>
{
  using typename interface<Align, hsa<Align>>::raw;
  uint64_t hsa_region_t_handle;
  HOSTRPC_ANNOTATE hsa(uint64_t hsa_region_t_handle)
      : hsa_region_t_handle(hsa_region_t_handle)
  {
  }
  HOSTRPC_ANNOTATE raw allocate(size_t N)
  {
    return {hsa_impl::allocate(hsa_region_t_handle, Align, N)};
  }
  HOSTRPC_ANNOTATE static status destroy(raw *x)
  {
    return (hsa_impl::deallocate(x->ptr) == 0) ? success : failure;
  }
};

template <size_t Align>
struct hsa_shared : public interface<Align, hsa_shared<Align>>
{
  using typename interface<Align, hsa_shared<Align>>::raw;

  HOSTRPC_ANNOTATE hsa_shared() { static_assert(Align <= 4096, ""); }
  HOSTRPC_ANNOTATE raw allocate(size_t N)
  {
    return {hsa_impl::allocate_fine_grain(N)};
  }
  HOSTRPC_ANNOTATE static status destroy(raw *x)
  {
    return (hsa_impl::deallocate(x->ptr) == 0) ? success : failure;
  }
};

namespace cuda_impl
{
// caller gets to align the result
// docs claim 'free' can return errors from unrelated launches (??), so I guess
// that should be propagated up
HOSTRPC_ANNOTATE void *allocate_gpu(size_t size);
HOSTRPC_ANNOTATE int deallocate_gpu(void *);

HOSTRPC_ANNOTATE void *allocate_shared(size_t size);
HOSTRPC_ANNOTATE int deallocate_shared(void *);

HOSTRPC_ANNOTATE void *device_ptr_from_host_ptr(void *);

HOSTRPC_ANNOTATE int memsetzero_gpu(void *, size_t size);
}  // namespace cuda_impl

template <size_t Align>
struct cuda_gpu : public interface<Align, cuda_gpu<Align>>
{
  using Base = interface<Align, cuda_gpu<Align>>;
  using typename Base::local_t;
  using typename Base::raw;
  using typename Base::remote_t;

  HOSTRPC_ANNOTATE cuda_gpu() {}
  HOSTRPC_ANNOTATE raw allocate(size_t N)
  {
    size_t adj = N + Align - 1;
    return {cuda_impl::allocate_gpu(adj)};  // zeros
  }
  HOSTRPC_ANNOTATE static status destroy(raw *x)
  {
    return cuda_impl::deallocate_gpu(x->ptr) == 0 ? success : failure;
  }
  HOSTRPC_ANNOTATE static local_t local_ptr(const raw &)
  {
    // local is on the host (as only have a host cuda allocator at present)
    return {0};
  }
  HOSTRPC_ANNOTATE static remote_t remote_ptr(const raw &x)
  {
    return {align_pointer_up(x.ptr, Align)};
  }
};

template <size_t Align>
struct cuda_shared : public interface<Align, cuda_shared<Align>>
{
  // allocate a pointer on the host and derive a device one from it
  using Base = interface<Align, cuda_shared<Align>>;
  using typename Base::local_t;
  using typename Base::raw;
  using typename Base::remote_t;

  HOSTRPC_ANNOTATE cuda_shared() {}
  HOSTRPC_ANNOTATE raw allocate(size_t N)
  {
    size_t adj = N + Align - 1;
    return {cuda_impl::allocate_shared(adj)};  // zeros
  }
  HOSTRPC_ANNOTATE static status destroy(raw *x)
  {
    return cuda_impl::deallocate_shared(x->ptr) == 0 ? success : failure;
  }
  HOSTRPC_ANNOTATE static local_t local_ptr(const raw &x)
  {
    return {align_pointer_up(x.ptr, Align)};
  }
  HOSTRPC_ANNOTATE static remote_t remote_ptr(const raw &x)
  {
    if (!x.ptr)
      {
        return {0};
      }
    void *dev = cuda_impl::device_ptr_from_host_ptr(x.ptr);
    return {align_pointer_up(dev, Align)};
  }
};

namespace openmp_impl
{
HOSTRPC_ANNOTATE void *allocate_device(int device_num, size_t bytes);
HOSTRPC_ANNOTATE int deallocate_device(int device_num, void *);

HOSTRPC_ANNOTATE void *allocate_shared(size_t bytes);
HOSTRPC_ANNOTATE int deallocate_shared(void *);

}  // namespace openmp_impl

template <size_t Align, int device_num>
struct openmp_device : public interface<Align, openmp_device<Align, device_num>>
{
  using Base = interface<Align, openmp_device<Align, device_num>>;
  using typename Base::local_t;
  using typename Base::raw;
  using typename Base::remote_t;

  HOSTRPC_ANNOTATE
  openmp_device() {}
  HOSTRPC_ANNOTATE raw allocate(size_t N)
  {
    size_t adj = N + Align - 1;
    return {openmp_impl::allocate_device(device_num, adj)};
  }
  HOSTRPC_ANNOTATE static status destroy(raw *x)
  {
    return (openmp_impl::deallocate_device(device_num, x->ptr) == 0) ? success
                                                                     : failure;
  }

  HOSTRPC_ANNOTATE static local_t local_ptr(const raw &) { return {0}; }
  HOSTRPC_ANNOTATE static remote_t remote_ptr(const raw &x)
  {
    return {align_pointer_up(x.ptr, Align)};
  }
};

template <size_t Align>
struct openmp_shared : public interface<Align, openmp_shared<Align>>
{
  using Base = interface<Align, openmp_shared<Align>>;
  using typename Base::local_t;
  using typename Base::raw;
  using typename Base::remote_t;

  HOSTRPC_ANNOTATE
  openmp_shared() {}

  HOSTRPC_ANNOTATE raw allocate(size_t N)
  {
    size_t adj = N + Align - 1;
    return {openmp_impl::allocate_shared(adj)};
  }
  HOSTRPC_ANNOTATE static status destroy(raw *x)
  {
    return (openmp_impl::deallocate_shared(x->ptr) == 0) ? success : failure;
  }

  HOSTRPC_ANNOTATE static local_t local_ptr(const raw &x)
  {
    return {align_pointer_up(x.ptr, Align)};
  }
  HOSTRPC_ANNOTATE static remote_t remote_ptr(const raw &x)
  {
    return {align_pointer_up(x.ptr, Align)};
  }
};

namespace cxx
{
// std::move reimplemented
template <class T>
struct remove_reference
{
  typedef T type;
};
template <class T>
struct remove_reference<T &>
{
  typedef T type;
};
template <class T>
struct remove_reference<T &&>
{
  typedef T type;
};

template <class T>
inline constexpr typename remove_reference<T>::type &&move(T &&x)
{
  typedef typename remove_reference<T>::type U;
  return static_cast<U &&>(x);
}
}  // namespace cxx

template <typename AllocBuffer, typename AllocInboxOutbox, typename AllocLocal,
          typename AllocRemote>
struct store_impl
{
  typename AllocBuffer::raw buffer;
  typename AllocInboxOutbox::raw recv;
  typename AllocInboxOutbox::raw send;
  typename AllocLocal::raw local_lock;
  typename AllocLocal::raw local_staging;
  typename AllocRemote::raw remote_lock;
  typename AllocRemote::raw remote_staging;

  HOSTRPC_ANNOTATE store_impl() = default;
  HOSTRPC_ANNOTATE ~store_impl() = default;
  HOSTRPC_ANNOTATE store_impl(typename AllocBuffer::raw &&buffer,
                              typename AllocInboxOutbox::raw &&recv,
                              typename AllocInboxOutbox::raw &&send,
                              typename AllocLocal::raw &&local_lock,
                              typename AllocLocal::raw &&local_staging,
                              typename AllocRemote::raw &&remote_lock,
                              typename AllocRemote::raw &&remote_staging)
      : buffer(cxx::move(buffer)),
        recv(cxx::move(recv)),
        send(cxx::move(send)),
        local_lock(cxx::move(local_lock)),
        local_staging(cxx::move(local_staging)),
        remote_lock(cxx::move(remote_lock)),
        remote_staging(cxx::move(remote_staging))
  {
  }

  HOSTRPC_ANNOTATE store_impl(const store_impl &) = delete;
  HOSTRPC_ANNOTATE store_impl &operator=(const store_impl &) = delete;

  HOSTRPC_ANNOTATE store_impl(store_impl &&) = default;
  HOSTRPC_ANNOTATE store_impl &operator=(store_impl &&) = default;

  HOSTRPC_ANNOTATE bool valid()
  {
    return buffer.valid() && recv.valid() && send.valid() &&
           local_lock.valid() && local_staging.valid() && remote_lock.valid() &&
           remote_staging.valid();
  }

  HOSTRPC_ANNOTATE void dump()
  {
    buffer.dump("buffer");
    recv.dump("recv");
    send.dump("send");
    local_lock.dump("local_lock");
    local_staging.dump("local_staging");
    remote_lock.dump("remote_lock");
    remote_staging.dump("remote_staging");
  }

  HOSTRPC_ANNOTATE status destroy()
  {
    status rc = success;
    rc = destroy_help(buffer, rc);
    rc = destroy_help(recv, rc);
    rc = destroy_help(send, rc);
    rc = destroy_help(local_lock, rc);
    rc = destroy_help(local_staging, rc);
    rc = destroy_help(remote_lock, rc);
    rc = destroy_help(remote_staging, rc);
    return rc;
  }

 private:
  template <typename T>
  HOSTRPC_ANNOTATE status destroy_help(T &raw, status acc)
  {
    // destroy on nullptr considered success
    status rc = raw.valid() ? raw.destroy() : success;
    return (acc == success && rc == success) ? success : failure;
  }
};

}  // namespace allocator
}  // namespace hostrpc

#endif
