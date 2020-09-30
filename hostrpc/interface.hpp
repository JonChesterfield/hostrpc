#ifndef INTERFACE_HPP_INCLUDED
#define INTERFACE_HPP_INCLUDED

#include "base_types.hpp"
#include <stddef.h>
#include <stdint.h>

#if defined(__x86_64__)
#include "test_common.hpp"  // round
#include "x64_host_x64_client.hpp"
#endif

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

inline constexpr size_t client_counter_overhead()
{
  return client_counters::cc_total_count * sizeof(_Atomic(uint64_t));
}

inline constexpr size_t server_counter_overhead()
{
  return server_counters::sc_total_count * sizeof(_Atomic(uint64_t));
}

struct gcn_x64_t
{
  gcn_x64_t(size_t minimum_number_slots, uint64_t hsa_region_t_fine_handle,
            uint64_t hsa_region_t_coarse_handle);

  ~gcn_x64_t();
  gcn_x64_t(const gcn_x64_t &) = delete;
  bool valid();

  struct client_t
  {
    friend struct gcn_x64_t;
    client_t() {}

    using state_t = hostrpc::storage<56 + client_counter_overhead(), 8>;

    void invoke(hostrpc::page_t *);
    void invoke_async(hostrpc::page_t *);

   private:
    template <typename ClientType>
    client_t(ClientType ct)
    {
      static_assert(static_equal<state_t::size(), sizeof(ClientType)>(), "");
      static_assert(state_t::size() == sizeof(ClientType), "");
      static_assert(state_t::align() == alignof(ClientType), "");
      auto *cv = state.construct<ClientType>(ct);
      assert(cv == state.open<ClientType>());
      (void)cv;
    }
    state_t state;
  };

  struct server_t
  {
    friend struct gcn_x64_t;
    server_t() {}
    using state_t = hostrpc::storage<72 + server_counter_overhead(), 8>;

    bool handle(hostrpc::page_t *, uint32_t *loc);

   private:
    template <typename ServerType>
    server_t(ServerType st)
    {
      static_assert(static_equal<state_t::size(), sizeof(ServerType)>(), "");
      static_assert(state_t::size() == sizeof(ServerType), "");
      static_assert(state_t::align() == alignof(ServerType), "");
      auto *sv = state.construct<ServerType>(st);
      assert(sv == state.open<ServerType>());
      (void)sv;
    }
    state_t state;
  };

  client_t client();
  server_t server();

 private:
  void *state;
};

}  // namespace hostrpc

#endif
