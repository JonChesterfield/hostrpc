#include "EvilUnit.h"

#include "../thirdparty/EvilUnit/EvilUnit_selftest.c"

static MODULE(another_module) { CHECK(1); }

MODULE(demo)
{
  int life = 42;
  DEPENDS(another_module);
  CHECK(life == 42);
  TEST("truth") { CHECK(life != 43); }
  TEST("lies")
  {
    int four = 2 + 2;
    CHECK(four == 5);
  }
}

MAIN_MODULE()
{
  DEPENDS(demo);
  DEPENDS(evilunit_selftest);
}
