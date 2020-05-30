#include "catch.hpp"

#include "client.hpp"
#include "server.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <unistd.h>

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

  void step()
  {
    while (steps_left == 0)
      {
        // Don't burn all the cpu waiting
        {
          std::unique_lock<std::mutex> lk(m);
          cv.wait_for(lk, std::chrono::milliseconds(10));
          lk.unlock();
        }
      }

    steps_left--;
    return;
  }

  void run(uint64_t x) { steps_left += x; }

 private:
  std::atomic<std::uint64_t> steps_left{0};
  std::thread t;

  std::mutex m;
  std::condition_variable cv;
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
    safe_thread cl_thrd([&]() {
      auto st = [&](int line) {
        printf("client.hpp:%d: step\n", line);
        cl_thrd.step();
      };

      auto cl =
          client<64, decltype(st)>(&recv, &send, &buffer[0], st, fill, use);
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

    safe_thread sv_thrd([&]() {
      auto st = [&](int line) {
        printf("server.hpp:%d: step\n", line);
        sv_thrd.step();
      };

      auto sv = server<64, decltype(st)>(&send, &recv, &buffer[0], st, operate);

      printf("Built a server\n");

      for (int x = 0; x < 3; x++)
        {
          sv.rpc_handle();
          printf("Server handled req %d\n", x);
          if (!run)
            {
              return;
            }
        }

      printf("Server finished\n");
    });

    printf("Threads spawned and running\n");

    for (unsigned i = 0; i < 500; i++)
      {
        cl_thrd.run(1);
        sv_thrd.run(1);
        usleep(100);
      }

    run = false;
  }

  printf("But didn't wait\n");
}
