#ifndef CLIENT_HPP_INCLUDED
#define CLIENT_HPP_INCLUDED

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

  // true if construction succeeded
  bool valid() noexcept { return derived().valid_impl(); }

  // RPC sometimes involves creating the client/server objects on one
  // system and then copying them to another.
  static constexpr size_t serialize_size() noexcept
  {
    return T::serialize_size_impl();
  }
  void serialize(uint64_t *to) noexcept { derived().serialize_impl(to); }
  void deserialize(uint64_t *from) noexcept
  {
    derived().deserialize_impl(from);
  }

 protected:
  interface() {}

 private:
  friend T;
  T &derived() { return *static_cast<T *>(this); }
  bool invoke_impl(void *) { return false; }
  bool invoke_async_impl(void *) { return false; }
  bool valid_impl() { return false; }

  static constexpr size_t serialize_size_impl() noexcept { return SIZE_MAX; }
  void serialize_impl(uint64_t *) noexcept {}
  void deserialize_impl(uint64_t *) noexcept {}
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

  // true if construction succeeded
  bool valid() noexcept { return derived().valid_impl(); }

  static constexpr size_t serialize_size() noexcept
  {
    return T::serialize_size_impl();
  }
  void serialize(uint64_t *to) noexcept { derived().serialize_impl(to); }
  void deserialize(uint64_t *from) noexcept
  {
    derived().deserialize_impl(from);
  }

 protected:
  interface() {}

 private:
  friend T;
  T &derived() { return *static_cast<T *>(this); }
  bool handle_impl(void *, uint64_t *) { return false; }
  bool valid_impl() { return false; }
  static constexpr size_t serialize_size_impl() noexcept { return SIZE_MAX; }
  void serialize_impl(uint64_t *) noexcept {}
  void deserialize_impl(uint64_t *) noexcept {}
};
}  // namespace server

struct x64_x64_t
{
  // This probably can't be copied, but could be movable
  x64_x64_t(size_t);
  ~x64_x64_t();
  x64_x64_t(const x64_x64_t &) = delete;

  struct client_t : public client::interface<client_t>
  {
    friend struct client::interface<client_t>;
    friend struct x64_x64_t;
    client_t(size_t);  // might be private
    ~client_t();       // probably doesn't do anything
   private:
    client_t();
    bool invoke_impl(void *);
    bool invoke_async_impl(void *);
    bool valid_impl();

    // state needs to be an x64_x64_client<128> or similar from the perspective
    // of calling methods on it and an array of bytes from the perspective of
    // moving it around
    // the client_impl<> type is essentially a short list of pointers.
    uint64_t state[5];
  };

  struct server_t : public server::interface<server_t>
  {
    friend struct server::interface<server_t>;
    friend struct x64_x64_t;
    server_t(size_t);  // might be private
    ~server_t();       // probably doesn't do anything
   private:
    server_t();
    bool handle_impl(void *, uint64_t *);
    bool valid_impl();
    uint64_t state[5];
  };

  client_t client();
  server_t server();

 private:
  void *state;
  size_t N;
};

}  // namespace hostrpc

#endif
