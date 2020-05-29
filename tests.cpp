#include "catch.hpp"

#include "client.hpp"
#include "server.hpp"

TEST_CASE("Bitmap")
{
  SECTION("set and clear each element")
  {
    hostrpc::slot_bitmap<128> b;
    if (0)
      for (size_t i = 0; i < b.size(); i++)
        {
          CHECK(!b[i]);
          CHECK(b.claim_slot(i));
          CHECK(b[i]);
          b.release_slot(i);
          CHECK(!b[i]);
        }
  }

  SECTION("find and claim each element")
  {
    hostrpc::slot_bitmap<128> b;
    for (size_t i = 0; i < b.size(); i++)
      {
        size_t e = b.find_slot();
        CHECK(e != SIZE_MAX);
        CHECK(!b[e]);
        CHECK(b.claim_slot(e));
        CHECK(b[e]);
      }

    CHECK(b.find_slot() == SIZE_MAX);
  }

  SECTION("find elements in the middle of the bitmap")
  {
    hostrpc::slot_bitmap<128> b;
    for (size_t i = 0; i < b.size(); i++)
      {
        b.claim_slot(i);
      }

    for (unsigned L : {0, 3, 63, 64, 65, 126, 127})
      {
        CHECK(b[L]);
        b.release_slot(L);
        CHECK(!b[L]);
        CHECK(b.find_slot() == L);
        CHECK(b.claim_slot(L));
        CHECK(b[L]);
      }
  }
}

TEST_CASE("Instantiate bitmap")
{
  hostrpc::slot_bitmap<64> bm64;
  hostrpc::slot_bitmap<128> bm128;
  (void)bm64;
  (void)bm128;
}
