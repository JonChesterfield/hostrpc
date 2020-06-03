#include "catch.hpp"
#include "hsa.hpp"

namespace
{
hsa::init global_state;
}

TEST_CASE("Is init")
{
  REQUIRE(global_state.status == HSA_STATUS_SUCCESS);

  hsa::iterate_agents([](hsa_agent_t agent) -> hsa_status_t {
    (void)agent;
    return HSA_STATUS_SUCCESS;
  });
}
