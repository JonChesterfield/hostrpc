#include "hsa.hpp"
#include "catch.hpp"

TEST_CASE("Is init")
{
  hsa::init state;
  CHECK(state.status == HSA_STATUS_SUCCESS);
}
