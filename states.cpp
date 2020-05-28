#include "catch.hpp"

struct machine
{
  machine() {}
  bool G;
  bool W;
  bool H;
  bool T;
};

struct slot
{
  slot() {} 
  machine host;
  machine gpu;
};

TEST_CASE("base")
{
  CHECK(1);
}
