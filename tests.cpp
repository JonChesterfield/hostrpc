#include "catch.hpp"

#include "client.hpp"
#include "server.hpp"

#include <thread>

TEST_CASE("Bitmap")
{
  SECTION("set and clear each element")
  {
    hostrpc::slot_bitmap<128> b;
    if (0)
      for (size_t i = 0; i < b.size(); i++)
        {
          CHECK(!b[i]);
          CHECK(b.try_claim_empty_slot(i));
          CHECK(b[i]);
          b.release_slot(i);
          CHECK(!b[i]);
          b.claim_slot(i);
          CHECK(b[i]);
        }
  }

  SECTION("find and unconditionally claim each element")
  {
    hostrpc::slot_bitmap<128> b;
    for (size_t i = 0; i < b.size(); i++)
      {
        size_t e = b.find_empty_slot();
        CHECK(e != SIZE_MAX);
        CHECK(!b[e]);
        b.claim_slot(e);
        CHECK(b[e]);
      }

    CHECK(b.find_empty_slot() == SIZE_MAX);
  }

  SECTION("find and try claim each element")
  {
    hostrpc::slot_bitmap<128> b;
    for (size_t i = 0; i < b.size(); i++)
      {
        size_t e = b.find_empty_slot();
        CHECK(e != SIZE_MAX);
        CHECK(!b[e]);
        CHECK(b.try_claim_empty_slot(e));
        CHECK(b[e]);
      }

    CHECK(b.find_empty_slot() == SIZE_MAX);
  }

  SECTION("find elements in the middle of the bitmap")
  {
    hostrpc::slot_bitmap<128> b;
    for (size_t i = 0; i < b.size(); i++)
      {
        b.try_claim_empty_slot(i);
      }

    for (unsigned L : {0, 3, 63, 64, 65, 126, 127})
      {
        CHECK(b[L]);
        b.release_slot(L);
        CHECK(!b[L]);
        CHECK(b.find_empty_slot() == L);
        b.claim_slot(L);
        CHECK(b[L]);
        b.release_slot(L);
        CHECK(!b[L]);
        CHECK(b.try_claim_empty_slot(L));
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

struct safe_thread
{
  template <typename Function, typename... Args>
  explicit safe_thread(Function f, Args... args)
      : t(std::forward<Function>(f), std::forward<Args>(args)...)
  {
  }
  ~safe_thread() { t.join(); }

 private:
  std::thread t;
};

TEST_CASE("set up single word system")
{
  using namespace hostrpc;

  using cb_type = std::function<void(hostrpc::page_t*)>;

  cb_type fill = [](page_t* p) { p->cacheline[0].element[0] = 4; };
  cb_type operate = [](page_t* p) { p->cacheline[0].element[0] *= 2; };
  cb_type use = [](page_t* p) {
    printf("Got %lu\n", p->cacheline[0].element[0]);
  };

  mailbox_t<64> send;
  mailbox_t<64> recv;
  page_t buffer[64];

  bool run = true;

  {
    safe_thread cl([&]() {
      auto cl = client<64, nop_stepper>(&recv, &send, &buffer[0], nop_stepper{},
                                        fill, use);
      printf("Built a client\n");

      for (int x = 0; x < 3; x++)
        {
          cl.rpc_invoke();

          if (!run)
            {
              return;
            }
        }
    });

    safe_thread sv([&]() {
      auto sv = server<64, nop_stepper>(&send, &recv, &buffer[0], nop_stepper{},
                                        operate);

      printf("Built a server\n");

      for (int x = 0; x < 3; x++)
        {
          sv.rpc_handle();

          if (!run)
            {
              return;
            }
        }
    });

    printf("Threads spawned and running\n");
    run = false;
  }

  printf("But didn't wait\n");
}
