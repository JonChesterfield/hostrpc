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
  uint64_t tmp;
  SECTION("set and clear each element")
    {
    hostrpc::slot_bitmap<128> b;
    if (0)
      for (size_t i = 0; i < b.size(); i++)
        {
          CHECK(!b[i]);
          CHECK(b.try_claim_empty_slot(i,&tmp));
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
        CHECK(b.try_claim_empty_slot(e,&tmp));
        CHECK(b[e]);
      }

    CHECK(b.find_empty_slot() == SIZE_MAX);
  }

  SECTION("find elements in the middle of the bitmap")
  {
    hostrpc::slot_bitmap<128> b;
    for (size_t i = 0; i < b.size(); i++)
      {
        b.try_claim_empty_slot(i,&tmp);
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
        CHECK(b.try_claim_empty_slot(L,&tmp));
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
    if (steps_left == UINT64_MAX)
      {
        // Disable stepping
        return;
      }
    while (steps_left == 0)
      {
        // Don't burn all the cpu waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }

    steps_left--;
    return;
  }

  void run(uint64_t x) { steps_left += x; }

 private:
  std::atomic<std::uint64_t> steps_left{0};
  std::thread t;
};

TEST_CASE("set up single word system")
{
  using namespace hostrpc;

  using cb_type = std::function<void(hostrpc::page_t*)>;

  std::atomic<uint64_t> val(UINT64_MAX);

  cb_type fill = [&](page_t* p) {
    val++;
    printf("Passing %lu\n", static_cast<uint64_t>(val));
    p->cacheline[0].element[0] = val;
  };
  cb_type operate = [](page_t* p) {
    uint64_t r = p->cacheline[0].element[0];
    printf("Server received %lu, forwarding as %lu\n", r, 2 * r);
    p->cacheline[0].element[0] = 2 * r;
  };
  cb_type use = [](page_t* p) {
    printf("Returned %lu\n", p->cacheline[0].element[0]);
  };

  mailbox_t<64> send;
  mailbox_t<64> recv;
  page_t buffer[64];

  const uint64_t calls_planned = 1024;
  std::atomic<uint64_t> calls_launched(0);
  std::atomic<uint64_t> calls_handled(0);

  const bool show_step = false;
  {
    safe_thread cl_thrd([&]() {
      auto st = [&](int line) {
        if (show_step)
          {
            printf("client.hpp:%d: step\n", line);
          }
        cl_thrd.step();
      };

      auto cl =
          client<64, decltype(st)>(&recv, &send, &buffer[0], st, fill, use);

      while (calls_launched < calls_planned)
        {
          if (cl.rpc_invoke<false>())
            {
              calls_launched++;
            }
          if (cl.rpc_invoke<true>())
            {
              calls_launched++;
            }
        }
    });

    safe_thread sv_thrd([&]() {
      auto st = [&](int line) {
        if (show_step)
          {
            printf("server.hpp:%d: step\n", line);
          }
        sv_thrd.step();
      };

      auto sv = server<64, decltype(st)>(&send, &recv, &buffer[0], st, operate);
      for (;;)
        {
          if (sv.rpc_handle())
            {
              calls_handled++;
            }
          if (calls_handled >= calls_planned)
            {
              return;
            }
        }
    });

    printf("Threads spawned and running\n");

    if (1)
      {
        cl_thrd.run(UINT64_MAX);
        sv_thrd.run(UINT64_MAX);
      }
    else
      {
        for (unsigned i = 0; i < 20000; i++)
          {
            cl_thrd.run(1);
            sv_thrd.run(1);
            usleep(100);
          }
      }

    {
      uint64_t l = (uint64_t)calls_launched;
      uint64_t h = (uint64_t)calls_handled;
      do
        {
          uint64_t nl = (uint64_t)calls_launched;
          uint64_t nh = (uint64_t)calls_handled;

          if (nl != l || nh != h)
            {
              printf("%lu launched, %lu handled\n", nl, nh);
              l = nl;
              h = nh;
            }

          usleep(100000);
        }
      while ((calls_launched != calls_handled) ||
             (calls_launched != calls_planned));
    }
  }
}
