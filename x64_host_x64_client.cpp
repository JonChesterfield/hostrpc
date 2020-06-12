#include "x64_host_x64_client.hpp"

#include "client.hpp"
#include "client_impl.hpp"
#include "memory.hpp"
#include "server_impl.hpp"

#include <string.h>

#include <new>

namespace hostrpc
{
// TODO: Handle N variable w/out loss efficiency
using ty = x64_x64_pair<128>;

x64_x64_t::x64_x64_t(size_t N) : state(nullptr), N(N)
{
  if (N <= 128)
    {
      // TODO: want the nothrow new but don't have <new>
      x64_x64_pair<128> *s = new x64_x64_pair<128>;
      state = static_cast<void *>(s);
    }
}

x64_x64_t::~x64_x64_t()
{
  ty *s = static_cast<ty *>(state);
  if (s)
    {
      delete s;
    }
}

bool x64_x64_t::valid() { return state != nullptr; }

static decltype(ty::client) *open_client(uint64_t *state)
{
  return reinterpret_cast<decltype(ty::client) *>(state);
}
static decltype(ty::server) *open_server(uint64_t *state)
{
  return reinterpret_cast<decltype(ty::server) *>(state);
}

x64_x64_t::client_t x64_x64_t::client()
{
  ty *s = static_cast<ty *>(state);
  assert(s);
  client_t res;
  auto *cl = reinterpret_cast<decltype(ty::client) *>(&res.state[0]);
  *cl = s->client;
  return res;
}

  __attribute__((used))
x64_x64_t::server_t x64_x64_t::server()
{
  asm ("# HERE");
  ty *s = static_cast<ty *>(state);
  assert(s);
  server_t res;
  auto *cl = reinterpret_cast<decltype(ty::server) *>(&res.state[0]);
  *cl = s->server;
  return res;
}

bool x64_x64_t::client_t::invoke_impl(void *application_state)
{
  auto *cl = open_client(&state[0]);
  return cl->rpc_invoke<true>(application_state);
}

bool x64_x64_t::client_t::invoke_async_impl(void *application_state)
{
  auto *cl = open_client(&state[0]);
  return cl->rpc_invoke<false>(application_state);
}

bool x64_x64_t::server_t::handle_impl(void *application_state, uint64_t *l)
{
  auto *se = open_server(&state[0]);
  return se->rpc_handle(application_state, l);
}

}  // namespace hostrpc

namespace hostrpc
{
slot_owner tracker;
}  // namespace hostrpc

