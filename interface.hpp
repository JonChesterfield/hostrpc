#ifndef INTERFACE_HPP_INCLUDED
#define INTERFACE_HPP_INCLUDED

#include "base_types.hpp"
#include <stddef.h>
#include <stdint.h>

namespace hostrpc
{
// Lifecycle management is tricky for objects which are allocated on one system
// and copied to another, where they contain pointers into each other.
// One owning object is created. If successful, that can construct instances of
// a client or server class. These can be copied by memcpy, which is necessary
// to set up the instance across pcie. The client/server objects don't own
// memory so can be copied at will. They can be used until the owning instance
// destructs.

// Notes on the legality of the char state[] handling and aliasing.
// Constructing an instance into state[] is done with placement new, which needs
// the header <new> that is unavailable for amdgcn at present. Following
// libunwind's solution discussed at D57455, operator new is added as a member
// function to client_impl, server_impl. Combined with a reinterpret cast to
// select the right operator new, that creates the object. Access is via
// std::launder'ed reinterpret cast, but as one can't assume C++17 and doesn't
// have <new> for amdgcn, this uses __builtin_launder.

struct x64_x64_t
{
  // This probably can't be copied, but could be movable
  x64_x64_t(size_t minimum_number_slots);  // != 0
  ~x64_x64_t();
  x64_x64_t(const x64_x64_t &) = delete;
  bool valid();  // true if construction succeeded

  struct client_t
  {
    friend struct x64_x64_t;
    client_t() {}  // would like this to be private
    using state_t = hostrpc::storage<48, 8>;

    bool invoke(closure_func_t fill, void *fill_state, closure_func_t use,
                void *use_state);
    bool invoke_async(closure_func_t fill, void *fill_state, closure_func_t use,
                      void *use_state);

    template <typename Fill, typename Use>
    bool invoke(Fill f, Use u)
    {
      auto cbf = [](hostrpc::page_t *page, void *vf) {
        Fill *f = static_cast<Fill *>(vf);
        (*f)(page);
      };
      auto cbu = [](hostrpc::page_t *page, void *vf) {
        Use *f = static_cast<Use *>(vf);
        (*f)(page);
      };
      return invoke(cbf, static_cast<void *>(&f), cbu, static_cast<void *>(&u));
    }

    template <typename Fill, typename Use>
    bool invoke_async(Fill f, Use u)
    {
      auto cbf = [](hostrpc::page_t *page, void *vf) {
        Fill *f = static_cast<Fill *>(vf);
        (*f)(page);
      };
      auto cbu = [](hostrpc::page_t *page, void *vf) {
        Use *f = static_cast<Use *>(vf);
        (*f)(page);
      };
      return invoke_async(cbf, static_cast<void *>(&f), cbu,
                          static_cast<void *>(&u));
    }

   private:
    state_t state;
  };

  struct server_t
  {
    friend struct x64_x64_t;
    server_t() {}
    using state_t = hostrpc::storage<48, 8>;

    bool handle(closure_func_t operate, void *state, uint64_t *loc);

    template <typename Func>
    bool handle(Func f, uint64_t *loc)
    {
      auto cb = [](hostrpc::page_t *page, void *vf) {
        Func *f = static_cast<Func *>(vf);
        (*f)(page);
      };
      return handle(cb, static_cast<void *>(&f), loc);
    }

   private:
    state_t state;
  };

  client_t client();
  server_t server();

 private:
  void *state;
};


}  // namespace hostrpc

#endif
