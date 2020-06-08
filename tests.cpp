#include "catch.hpp"

#include "client.hpp"
#include "server.hpp"
#include "tests.hpp"

#include <thread>
#include <unistd.h>

TEST_CASE("Bitmap")
{
  using test_bitmap_t =
      hostrpc::slot_bitmap<128, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
                           hostrpc::x64_x64_slot_bitmap_data>;
  using bitmap_ptr_t =
      std::unique_ptr<test_bitmap_t::slot_bitmap_data_t,
                      test_bitmap_t::slot_bitmap_data_t::deleter>;

  bitmap_ptr_t ptr(test_bitmap_t::slot_bitmap_data_t::alloc());

  uint64_t tmp;
  SECTION("set and clear each element")
  {
    test_bitmap_t b(ptr.get());
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
    test_bitmap_t b(ptr.get());
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
    test_bitmap_t b(ptr.get());
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
    test_bitmap_t b(ptr.get());
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

TEST_CASE("set up single word system")
{
  using namespace hostrpc;
  constexpr size_t N = 64;
  page_t client_buffer[N];
  page_t server_buffer[N];

  _Atomic(uint64_t) val(UINT64_MAX);

  auto fill = [&](page_t* p, void*) -> void {
    val++;
    // printf("Passing %lu\n", static_cast<uint64_t>(val));
    p->cacheline[0].element[0] = val;
  };
  auto operate = [](page_t* p, void*) -> void {
    uint64_t r = p->cacheline[0].element[0];
    // printf("Server received %lu, forwarding as %lu\n", r, 2 * r);
    p->cacheline[0].element[0] = 2 * r;
  };
  auto use = [](page_t* p, void*) -> void {
    (void)p;
    // printf("Returned %lu\n", p->cacheline[0].element[0]);
  };

  using mailbox_ptr_t =
      std::unique_ptr<mailbox_t<N>::slot_bitmap_data_t,
                      mailbox_t<N>::slot_bitmap_data_t::deleter>;

  using lockarray_ptr_t =
      std::unique_ptr<lockarray_t<N>::slot_bitmap_data_t,
                      lockarray_t<N>::slot_bitmap_data_t::deleter>;

  mailbox_ptr_t send_data(mailbox_t<N>::slot_bitmap_data_t::alloc());
  mailbox_ptr_t recv_data(mailbox_t<N>::slot_bitmap_data_t::alloc());
  lockarray_ptr_t client_active_data(
      lockarray_t<N>::slot_bitmap_data_t::alloc());
  lockarray_ptr_t server_active_data(
      lockarray_t<N>::slot_bitmap_data_t::alloc());

  mailbox_t<N> send(send_data.get());
  mailbox_t<N> recv(recv_data.get());
  lockarray_t<N> client_active(client_active_data.get());
  lockarray_t<N> server_active(server_active_data.get());

  const uint64_t calls_planned = 1024;
  _Atomic(uint64_t) calls_launched(0);
  _Atomic(uint64_t) calls_handled(0);

  _Atomic(uint64_t) client_steps(0);
  _Atomic(uint64_t) server_steps(0);

  hostrpc::copy_functor_memcpy_pull cp;

  const bool show_step = false;
  {
    safe_thread cl_thrd([&]() {
      auto stepper_state =
          hostrpc::default_stepper_state(&client_steps, show_step);

      using client_type =
          client<N, hostrpc::x64_x64_bitmap_types, decltype(cp), decltype(fill),
                 decltype(use), hostrpc::default_stepper>;
      client_type cl = {cp,
                        recv,
                        send,
                        client_active,
                        &server_buffer[0],
                        &client_buffer[0],
                        hostrpc::default_stepper{},
                        fill,
                        use};

      void* application_state = static_cast<void*>(&stepper_state);

      while (calls_launched < calls_planned)
        {
          if (cl.rpc_invoke<false>(application_state))
            {
              calls_launched++;
            }
          if (cl.rpc_invoke<true>(application_state))
            {
              calls_launched++;
            }
        }
    });

    safe_thread sv_thrd([&]() {
      auto stepper_state =
          hostrpc::default_stepper_state(&server_steps, show_step);

      using server_type = server<N, hostrpc::x64_x64_bitmap_types, decltype(cp),
                                 decltype(operate), hostrpc::default_stepper>;

      server_type sv = {
          cp, send,   recv, server_active, &client_buffer[0], &server_buffer[0],
          {}, operate};

      void* application_state = static_cast<void*>(&stepper_state);

      for (;;)
        {
          if (sv.rpc_handle(application_state))
            {
              calls_handled++;
            }
          if (calls_handled >= calls_planned)
            {
              return;
            }
        }
    });

    // printf("Threads spawned and running\n");

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
              // printf("%lu launched, %lu handled\n", nl, nh);
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
