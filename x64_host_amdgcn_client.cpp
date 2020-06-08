#include "x64_host_amdgcn_client.hpp"
#include "catch.hpp"
#include "hsa.hpp"

#include <string>

namespace hostrpc
{
x64_amdgcn_pair::x64_amdgcn_pair() {}
~x64_amdgcn_pair::x64_amdgcn_pair() {}
}  // namespace hostrpc

namespace
{
hsa::init global_state;
}

TEST_CASE("Is init")
{
  REQUIRE(global_state.status == HSA_STATUS_SUCCESS);

  SECTION("check executable_destroy handles nullptr ok")
  {
    hsa_executable_t ex = {reinterpret_cast<uint64_t>(nullptr)};
    CHECK(hsa_executable_destroy(ex) == HSA_STATUS_ERROR_INVALID_EXECUTABLE);
  }

  std::vector<hsa_agent_t> kernel_agents;
  std::vector<hsa_agent_t> other_agents;
  hsa::iterate_agents([&](hsa_agent_t agent) -> hsa_status_t {
    auto features = hsa::agent_get_info_feature(agent);
    std::vector<hsa_agent_t>* list =
        (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH) ? &kernel_agents
                                                       : &other_agents;
    list->push_back(agent);
    return HSA_STATUS_SUCCESS;
  });

  printf("Found %zu kernel agents\n", kernel_agents.size());
  printf("Found %zu other agents\n", other_agents.size());

  FILE* fh = fopen("device.o", "rb");
  assert(fh);
  int fn = fileno(fh);
  assert(fn >= 0);

  hsa::executable ex(kernel_agents[0], fn);
  CHECK(ex.valid());

  hsa_executable_symbol_t symbol = ex.get_symbol_by_name("device_entry.kd");

  hsa_symbol_kind_t kind = hsa::symbol_get_info_type(symbol);
  REQUIRE(kind == HSA_SYMBOL_KIND_KERNEL);
  uint64_t address = hsa::symbol_get_info_kernel_object(symbol);

  printf("Kernel %s at address 0x%lx\n",
         hsa::symbol_get_info_name(symbol).c_str(), address);

  printf("buffer_start at 0x%lx\n",
         hsa::symbol_get_info_variable_address(
             ex.get_symbol_by_name("mulbytwo_buffer_start")));

  // This claims 0x68, was hoping for 0x0
  // get_info functions probably need to report failure
  printf("absent at 0x%lx\n", hsa::symbol_get_info_variable_address(
                                  ex.get_symbol_by_name("notasymbol")));
}
