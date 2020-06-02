#include "catch.hpp"

#include "client.hpp"
#include "server.hpp"

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
          CHECK(b.try_claim_empty_slot(i, &tmp));
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
        CHECK(b.try_claim_empty_slot(e, &tmp));
        CHECK(b[e]);
      }

    CHECK(b.find_empty_slot() == SIZE_MAX);
  }

  SECTION("find elements in the middle of the bitmap")
  {
    hostrpc::slot_bitmap<128> b;
    for (size_t i = 0; i < b.size(); i++)
      {
        b.try_claim_empty_slot(i, &tmp);
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
        CHECK(b.try_claim_empty_slot(L, &tmp));
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

  _Atomic(uint64_t) val(UINT64_MAX);

  auto fill = [&](page_t* p) -> void {
    val++;
    printf("Passing %lu\n", static_cast<uint64_t>(val));
    p->cacheline[0].element[0] = val;
  };
  auto operate = [](page_t* p) -> void {
    uint64_t r = p->cacheline[0].element[0];
    printf("Server received %lu, forwarding as %lu\n", r, 2 * r);
    p->cacheline[0].element[0] = 2 * r;
  };
  auto use = [](page_t* p) -> void {
    printf("Returned %lu\n", p->cacheline[0].element[0]);
  };

  mailbox_t<64> send;
  mailbox_t<64> recv;
  page_t client_buffer[64];
  page_t server_buffer[64];

  const uint64_t calls_planned = 1024;
  _Atomic(uint64_t) calls_launched(0);
  _Atomic(uint64_t) calls_handled(0);

  _Atomic(uint64_t) client_steps(0);
  _Atomic(uint64_t) server_steps(0);

  const bool show_step = false;
  {
    safe_thread cl_thrd([&]() {
      auto stepper = hostrpc::default_stepper(&client_steps, show_step);
      slot_bitmap<64, __OPENCL_MEMORY_SCOPE_DEVICE> active;
      auto cl = client<64, decltype(fill), decltype(use), default_stepper>(
          &recv, &send, &active, &server_buffer[0], &client_buffer[0], stepper,
          fill, use);

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
      auto stepper = hostrpc::default_stepper(&server_steps, show_step);
      slot_bitmap<64, __OPENCL_MEMORY_SCOPE_DEVICE> active;
      auto sv = server<64, decltype(operate), default_stepper>(
          &send, &recv, &active, &client_buffer[0], &server_buffer[0], stepper,
          operate);
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
        client_steps = UINT64_MAX;
        server_steps = UINT64_MAX;
      }
    else
      {
        for (unsigned i = 0; i < 20000; i++)
          {
            client_steps++;
            server_steps++;
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
