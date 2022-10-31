#include "hostrpc_printf_enable.hpp"

#include "hostrpc_printf_client.hpp"
#include "hostrpc_printf_server.hpp"

#include "x64_x64_type.hpp"

namespace
{
struct global
{
  enum
  {
    s = 64
  };
  hostrpc::size_compiletime<s> SZ;
  using type = hostrpc::x64_x64_type<hostrpc::size_compiletime<s>>;
  using wrap_state = wrap_server_state<type>;

  std::unique_ptr<wrap_state> state;

  global()
      : state(std::make_unique<wrap_state>(
            std::make_unique<type>(SZ, hostrpc::arch::x64{},
                                   hostrpc::arch::x64{}),
            SZ.value()))
  {
  }
  ~global() {}
} global_instance;

}  // namespace

HOSTRPC_PRINTF_INSTANTIATE_CLIENT(global::type::client_type, &(global_instance.state->p->client));
