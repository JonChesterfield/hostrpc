#include "x64_host_x64_client.hpp"
#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "interface.hpp"
#include "memory.hpp"
#include "test_common.hpp"
#include <string.h>

#include <new>

namespace hostrpc
{
// This doesn't especially care about fill/use/operate/step
// It needs new, probably shouldn't try to compile it on non-x64
using ty = x64_x64_pair_T<hostrpc::size_runtime, indirect::fill, indirect::use,
                          indirect::operate, indirect::clear>;

x64_x64_t::x64_x64_t(size_t N) : state(nullptr)
{
  N = hostrpc::round(N);
  hostrpc::size_runtime sz(N);
  assert(sz.N() != 0);
  ty *s = new (std::nothrow) ty(sz);
  state = static_cast<void *>(s);
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

x64_x64_t::client_t x64_x64_t::client()
{
  ty *s = static_cast<ty *>(state);
  assert(s);
  ty::client_type &ct = s->client;
  return {ct};
}

x64_x64_t::server_t x64_x64_t::server()
{
  ty *s = static_cast<ty *>(state);
  assert(s);
  ty::server_type &st = s->server;
  return {st};
}

bool x64_x64_t::client_t::invoke(hostrpc::closure_func_t fill, void *fill_state,
                                 hostrpc::closure_func_t use, void *use_state)
{
  return invoke<ty::client_type>(fill, fill_state, use, use_state);
}

bool x64_x64_t::client_t::invoke_async(hostrpc::closure_func_t fill,
                                       void *fill_state, closure_func_t use,
                                       void *use_state)
{
  return invoke_async<ty::client_type>(fill, fill_state, use, use_state);
}

hostrpc::client_counters x64_x64_t::client_t::get_counters()
{
  return state.open<ty::client_type>()->get_counters();
}

bool x64_x64_t::server_t::handle(hostrpc::closure_func_t func,
                                 void *application_state, uint64_t *l)
{
  return handle<ty::server_type>(func, application_state, l);
}

}  // namespace hostrpc
