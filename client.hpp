#ifndef CLIENT_HPP_INCLUDED
#define CLIENT_HPP_INCLUDED

// TODO: Rename file

#include <stdint.h>
#include <stddef.h>

namespace hostrpc
{
// Lifecycle management is tricky for objects which are allocated on one system
// and copied to another, where they contain pointers into each other.
// One owning object is created. If successful, that can construct instances of
// a client or server class. These can be copied by memcpy, which is necessary
// to set up the instance across pcie. The client/server objects don't own
// memory so can be copied at will. They can be used until the owning instance
// destructs.

namespace client
{
template <typename T>
struct interface
{
  bool invoke(void *x) noexcept { return derived().invoke_impl(x); }
  bool invoke_async(void *x) noexcept { return derived().invoke_async_impl(x); }

 protected:
  interface() {}

 private:
  friend T;
  T &derived() { return *static_cast<T *>(this); }
  bool invoke_impl(void *) { return false; }
  bool invoke_async_impl(void *) { return false; }
};
}  // namespace client
namespace server
{
template <typename T>
struct interface
{
  bool handle(void *x, uint64_t *location_arg) noexcept
  {
    return derived().handle_impl(x, location_arg);
  }
  bool handle(void *x) noexcept
  {
    uint64_t loc;
    return handle(x,&loc);
  }

 protected:
  interface() {}

 private:
  friend T;
  T &derived() { return *static_cast<T *>(this); }
  bool handle_impl(void *, uint64_t *) { return false; }
};
}  // namespace server

struct x64_x64_t
{
  // This probably can't be copied, but could be movable
  x64_x64_t(size_t);
  ~x64_x64_t();
  x64_x64_t(const x64_x64_t &) = delete;
  bool valid(); // true if construction succeeded
  
  struct client_t : public client::interface<client_t>
  {
    friend struct client::interface<client_t>;
    friend struct x64_x64_t;
    client_t() {} // would like this to be private
   private:
    bool invoke_impl(void *);
    bool invoke_async_impl(void *);

    // state needs to be an x64_x64_client<128> or similar from the perspective
    // of calling methods on it and an array of bytes from the perspective of
    // moving it around
    // the client_impl<> type is essentially a short list of pointers.
    // Leaning towards putting the values into void* [5] in the right order
    // and reinterpret_casting the start of the array as alternatives routing
    // through integers are hitting the inttoptr blocks
    __attribute__((__may_alias__)) uint64_t state[5];
  };

  struct server_t : public server::interface<server_t>
  {
    friend struct server::interface<server_t>;
    friend struct x64_x64_t;
    server_t(){}
   private:
    bool handle_impl(void *, uint64_t *);
    __attribute__((__may_alias__)) uint64_t state[5];
  };

  client_t client();
  server_t server();

 private:
  void *state;
  size_t N;
};

}  // namespace hostrpc

#endif
