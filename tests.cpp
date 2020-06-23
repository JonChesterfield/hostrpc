#include "catch.hpp"

#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "interface.hpp"
#include "memory.hpp"
#include "tests.hpp"

#include <thread>
#include <unistd.h>

TEST_CASE("instantiate")
{
  hostrpc::x64_x64_t foo(64);
  CHECK(foo.valid());

  foo.client();
  foo.server();
}

#if 0
TEST_CASE("Bitmap")
{
  static constexpr const size_t N = 128;
  using ty = hostrpc::size_compiletime<N>;
  using test_bitmap_t = hostrpc::slot_bitmap_all_svm<ty>;
  using bitmap_ptr_t =
      std::unique_ptr<_Atomic uint64_t,
                      hostrpc::x64_allocate_slot_bitmap_data_deleter>;

  bitmap_ptr_t ptr(hostrpc::x64_allocate_slot_bitmap_data(N));

  uint64_t tmp;
  SECTION("set and clear each element")
  {
    test_bitmap_t b(ptr.get(), ty{N});
    if (0)
      for (size_t i = 0; i < b.size(); i++)
        {
          CHECK(!b(i));
          CHECK(b.try_claim_empty_slot(i, &tmp));
          CHECK(b(i));
          b.release_slot(i);
          CHECK(!b(i));
          b.claim_slot(i);
          CHECK(b(i));
        }
  }

  SECTION("find and unconditionally claim each element")
  {
    test_bitmap_t b(ptr.get(), ty{N});
    for (size_t i = 0; i < b.size(); i++)
      {
        size_t e = b.find_empty_slot();
        CHECK(e != SIZE_MAX);
        CHECK(!b(e));
        b.claim_slot(e);
        CHECK(b(e));
      }

    CHECK(b.find_empty_slot() == SIZE_MAX);
  }

  SECTION("find and try claim each element")
  {
    test_bitmap_t b(ptr.get(), ty{N});
    for (size_t i = 0; i < b.size(); i++)
      {
        size_t e = b.find_empty_slot();
        CHECK(e != SIZE_MAX);
        CHECK(!b(e));
        CHECK(b.try_claim_empty_slot(e, &tmp));
        CHECK(b(e));
      }

    CHECK(b.find_empty_slot() == SIZE_MAX);
  }

  SECTION("find elements in the middle of the bitmap")
  {
    test_bitmap_t b(ptr.get(), ty{N});
    for (size_t i = 0; i < b.size(); i++)
      {
        b.try_claim_empty_slot(i, &tmp);
      }

    for (unsigned L : {0, 3, 63, 64, 65, 126, 127})
      {
        CHECK(b(L));
        b.release_slot(L);
        CHECK(!b(L));
        CHECK(b.find_empty_slot() == L);
        b.claim_slot(L);
        CHECK(b(L));
        b.release_slot(L);
        CHECK(!b(L));
        CHECK(b.try_claim_empty_slot(L, &tmp));
        CHECK(b(L));
      }
  }
}
#endif

static _Atomic uint64_t* x64_allocate_slot_bitmap_data(size_t size)
{
  assert(size % 64 == 0 && "Size must be a multiple of 64");
  constexpr const static size_t align = 64;
  void* memory = hostrpc::x64_native::allocate(align, size);
  return hostrpc::careful_array_cast<_Atomic uint64_t>(memory, size);
}

struct x64_allocate_slot_bitmap_data_deleter
{
  void operator()(_Atomic uint64_t* d)
  {
    hostrpc::x64_native::deallocate(static_cast<void*>(d));
  }
};

TEST_CASE("set up single word system")
{
  using namespace hostrpc;
  constexpr size_t N = 64;
  page_t client_buffer[N];
  page_t server_buffer[N];

  const bool show_step = false;

  _Atomic(uint64_t) client_steps(0);
  _Atomic(uint64_t) server_steps(0);

  _Atomic(uint64_t) val(UINT64_MAX);

  struct application_state_t
  {
    application_state_t(_Atomic(uint64_t) * val,
                        _Atomic(uint64_t) * client_steps, bool show_step)
        : val(val), stepper(client_steps, show_step)
    {
    }
    _Atomic(uint64_t) * val;
    default_stepper_state stepper;
  };

  struct stepper
  {
    static void call(int line, void* v)
    {
      application_state_t* state = static_cast<application_state_t*>(v);
      if (state->stepper.show_step)
        {
          printf("%s:%d: step\n", state->stepper.name, line);
        }
      step(state->stepper.val);
    }
  };

  struct fill
  {
    static void call(page_t* p, void* v)
    {
      application_state_t* state = static_cast<application_state_t*>(v);
      state->val++;
      // printf("Passing %lu\n", static_cast<uint64_t>(val));
      p->cacheline[0].element[0] = *(state->val);
    }
  };

  struct operate
  {
    static void call(page_t* p, void*)
    {
      uint64_t r = p->cacheline[0].element[0];
      // printf("Server received %lu, forwarding as %lu\n", r, 2 * r);
      p->cacheline[0].element[0] = 2 * r;
    }
  };

  struct use
  {
    static void call(page_t* p, void*) { (void)p; }
  };

  using mailbox_ptr_t =
      std::unique_ptr<_Atomic uint64_t, x64_allocate_slot_bitmap_data_deleter>;

  using lockarray_ptr_t =
      std::unique_ptr<_Atomic uint64_t, x64_allocate_slot_bitmap_data_deleter>;

  mailbox_ptr_t send_data(x64_allocate_slot_bitmap_data(N));
  mailbox_ptr_t recv_data(x64_allocate_slot_bitmap_data(N));
  lockarray_ptr_t client_active_data(x64_allocate_slot_bitmap_data(N));
  lockarray_ptr_t server_active_data(x64_allocate_slot_bitmap_data(N));

  using SZ = hostrpc::size_compiletime<N>;
  slot_bitmap_all_svm send(N, send_data.get());
  slot_bitmap_all_svm recv(N, recv_data.get());
  slot_bitmap_device client_active(N, client_active_data.get());
  slot_bitmap_device server_active(N, server_active_data.get());

  const uint64_t calls_planned = 1024;
  _Atomic(uint64_t) calls_launched(0);
  _Atomic(uint64_t) calls_handled(0);

  hostrpc::copy_functor_memcpy_pull cp;

  {
    safe_thread cl_thrd([&]() {
      auto app_state = application_state_t(&val, &client_steps, show_step);

      using client_type = client_impl<SZ, hostrpc::copy_functor_memcpy_pull,
                                      fill, use, stepper>;
      client_type cl = {SZ{},
                        recv,
                        send,
                        client_active,
                        &server_buffer[0],
                        &client_buffer[0]};

      void* application_state_ptr = static_cast<void*>(&app_state);

      while (calls_launched < calls_planned)
        {
          if (cl.rpc_invoke<false>(application_state_ptr,
                                   application_state_ptr))
            {
              calls_launched++;
            }
          if (false &&
              cl.rpc_invoke<true>(application_state_ptr, application_state_ptr))
            {
              calls_launched++;
            }
        }

      // printf("client done, launched %lu / %lu\n", calls_launched,
      // calls_planned);
    });

    safe_thread sv_thrd([&]() {
      auto stepper_state =
          hostrpc::default_stepper_state(&server_steps, show_step);

      using server_type =
          server_impl<SZ, decltype(cp), operate, hostrpc::default_stepper>;

      server_type sv = {SZ{},
                        send,
                        recv,
                        server_active,
                        &client_buffer[0],
                        &server_buffer[0]};

      void* application_state = static_cast<void*>(&stepper_state);
      uint64_t loc_arg = 0;
      (void)loc_arg;
      for (;;)
        {
          if (sv.rpc_handle(application_state, &loc_arg))

            {
              calls_handled++;
            }
          if (calls_handled >= calls_planned)
            {
              // printf("server done, handled %lu / %lu\n", calls_handled,
              // calls_planned);
              return;
            }
        }
    });

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
          else
            {
              break;
            }

          usleep(10000);
        }
      while ((calls_launched != calls_handled) ||
             (calls_launched != calls_planned));
    }

    printf("Done, now just waiting on threads\n");
  }
}
