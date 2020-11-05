#ifndef ALLOCATOR_HPP_INCLUDED
#define ALLOCATOR_HPP_INCLUDED

#include "detail/platform_detect.h"

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

  static const constexpr size_t align = Align;
  struct local_t
  {
    local_t(void *p) : ptr(p) {}
    operator void *() { return ptr; }
    void *ptr;
  };

  struct remote_t
  {
    remote_t(void *p) : ptr(p) {}
    operator void *() { return ptr; }
    void *ptr;
  };

  struct raw;
  raw allocate(size_t N) { return static_cast<Base *>(this)->allocate(N); }

  struct raw
  {
    raw(void *p) : ptr(p) {}
    bool valid() { return ptr != nullptr; }
    status destroy() { return Base::destroy(*this); }
    local_t local() { return Base::local(*this); }
    remote_t remote() { return Base::remote(*this); }
    void *ptr;
  };

 private:
  // The raw/local/remote conversion is a no-op for most allocators
  static local_t local(raw x) { return x.ptr; }
  static remote_t remote(raw x) { return x.ptr; }
};

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

  store_impl(typename AllocBuffer::raw buffer,
             typename AllocInboxOutbox::raw recv,
             typename AllocInboxOutbox::raw send,
             typename AllocLocal::raw local_lock,
             typename AllocLocal::raw local_staging,
             typename AllocRemote::raw remote_lock,
             typename AllocRemote::raw remote_staging)
      : buffer(buffer),
        recv(recv),
        send(send),
        local_lock(local_lock),
        local_staging(local_staging),
        remote_lock(remote_lock),
        remote_staging(remote_staging)
  {
  }

  bool valid()
  {
    return buffer.valid() && recv.valid() && send.valid() &&
           local_lock.valid() && local_staging.valid() && remote_lock.valid() &&
           remote_staging.valid();
  }

  status destroy()
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
  status destroy_help(T raw, status acc)
  {
    // destroy on nullptr considered success
    status rc = raw.valid() ? raw.destroy() : success;
    return (acc == success && rc == success) ? success : failure;
  }
};

}  // namespace allocator
}  // namespace hostrpc

#endif
