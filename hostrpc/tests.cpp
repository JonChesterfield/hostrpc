#include "catch.hpp"

#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "interface.hpp"
#include "memory.hpp"
#include "tests.hpp"

#include <list>
#include <thread>
#include <unistd.h>

struct x64_alloc_deleter
{
  void operator()(_Atomic(uint8_t) * d)
  {
    hostrpc::x64_native::deallocate(static_cast<void *>(d));
  }
  void operator()(_Atomic(uint64_t) * d)
  {
    hostrpc::x64_native::deallocate(static_cast<void *>(d));
  }
};

template <typename T>
static T x64_alloc(
    size_t size,
    std::list<std::unique_ptr<typename T::Ty, x64_alloc_deleter>> *store)
{
  using DelTy = std::unique_ptr<typename T::Ty, x64_alloc_deleter>;
  constexpr size_t bps = T::bits_per_slot();
  static_assert(bps == 1 || bps == 8, "");
  assert(size % 64 == 0 && "Size must be a multiple of 64");
  constexpr const static size_t align = 64;
  void *memory = hostrpc::x64_native::allocate(align, size * bps);
  typename T::Ty *m =
      hostrpc::careful_array_cast<typename T::Ty>(memory, size * bps);
  store->emplace_back(DelTy{m});
  return {m};
}

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
    static void call(int line, void *v)
    {
      application_state_t *state = static_cast<application_state_t *>(v);
      if (state->stepper.show_step)
        {
          printf("%s:%d: step\n", state->stepper.name, line);
        }
      step(state->stepper.val);
    }
  };

  struct fill
  {
    static void call(page_t *p, void *v)
    {
      application_state_t *state = static_cast<application_state_t *>(v);
      state->val++;
      // printf("Passing %lu\n", static_cast<uint64_t>(val));
      p->cacheline[0].element[0] = *(state->val);
    }
  };

  struct operate
  {
    static void call(page_t *p, void *)
    {
      uint64_t r = p->cacheline[0].element[0];
      // printf("Server received %lu, forwarding as %lu\n", r, 2 * r);
      p->cacheline[0].element[0] = 2 * r;
    }
  };

  struct clear
  {
    static void call(page_t *, void *) {}
  };

  struct use
  {
    static void call(page_t *p, void *) { (void)p; }
  };

  using SZ = hostrpc::size_compiletime<N>;

  std::list<std::unique_ptr<_Atomic(uint8_t), x64_alloc_deleter>> store8;
  std::list<std::unique_ptr<_Atomic(uint64_t), x64_alloc_deleter>> store64;

  hostrpc::copy_functor_memcpy_pull cp;

  using client_type = client_impl<SZ, decltype(cp), fill, use, stepper>;

  using server_type =
      server_impl<SZ, decltype(cp), operate, clear, hostrpc::default_stepper>;

  auto send = x64_alloc<client_type::outbox_t>(N, &store8);
  auto recv = x64_alloc<client_type::inbox_t>(N, &store64);
  auto client_active = x64_alloc<lock_bitmap>(N, &store64);
  auto client_staging = x64_alloc<client_type::staging_t>(N, &store64);
  auto server_active = x64_alloc<lock_bitmap>(N, &store64);
  auto server_staging = x64_alloc<server_type::staging_t>(N, &store64);

  const uint64_t calls_planned = 1024;
  _Atomic(uint64_t) calls_launched(0);
  _Atomic(uint64_t) calls_handled(0);

  {
    safe_thread cl_thrd([&]() {
      auto app_state = application_state_t(&val, &client_steps, show_step);

      client_type cl = {SZ{},
                        client_active,
                        recv,
                        send,
                        client_staging,
                        &server_buffer[0],
                        &client_buffer[0]};

      void *application_state_ptr = static_cast<void *>(&app_state);

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

      server_type sv = {SZ{},
                        server_active,
                        send,
                        recv,
                        server_staging,
                        &client_buffer[0],
                        &server_buffer[0]};

      void *application_state = static_cast<void *>(&stepper_state);
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
