#include "catch.hpp"
#include "hsa.hpp"

#include <string>

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

  FILE *fh = fopen("device.o", "rb");
  assert(fh);
  int fn = fileno(fh);
  assert(fn >= 0);

  hsa::executable ex(kernel_agents[0], fn);
  CHECK(ex.valid());  
  
}
