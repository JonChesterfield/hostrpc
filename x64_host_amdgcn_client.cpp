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

  hsa::iterate_agents([](hsa_agent_t agent) -> hsa_status_t {
    (void)agent;

    auto name = hsa::agent_get_info_name(agent);
    auto vname = hsa::agent_get_info_vendor_name(agent);

    std::string name_str(name.begin(), name.end());
    std::string vname_str(vname.begin(), vname.end());

    printf("Agent name %s / %s\n", name_str.c_str(), vname_str.c_str());

    return HSA_STATUS_SUCCESS;
  });
}
