#include "catch.hpp"

#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "memory.hpp"
#include "tests.hpp"

#include <list>
#include <thread>
#include <unistd.h>

#include "memory_host.hpp"

struct x64_alloc_deleter
{
  struct D
  {
    void operator()(void *d) { hostrpc::x64_native::deallocate(d); }
  };

  using UPtr = std::unique_ptr<void, D>;

  template <typename T>
  void operator()(T *d)
  {
    s.emplace_back(UPtr{static_cast<void *>(d)});
  }

  std::list<UPtr> s;
};

template <typename T>
static T x64_alloc(size_t size, x64_alloc_deleter *store)
{
  constexpr size_t bps = T::bits_per_slot();
  static_assert(bps == 1 || bps == 8, "");
  assert(size % 64 == 0 && "Size must be a multiple of 64");
  constexpr const static size_t align = 64;
  void *memory = hostrpc::x64_native::allocate(align, size * bps);
  typename T::Ty *m =
      hostrpc::careful_array_cast<typename T::Ty>(memory, size * bps);
  (*store)(m);
  return {m};
}

TEST_CASE("set up single word system")
{
  using namespace hostrpc;
  constexpr size_t N = 64;
  page_t client_buffer[N];
  page_t server_buffer[N];

  const bool show_step = false;

  HOSTRPC_ATOMIC(uint64_t) client_steps(0);
  HOSTRPC_ATOMIC(uint64_t) server_steps(0);

  HOSTRPC_ATOMIC(uint64_t) val(UINT64_MAX);

  struct application_state_t
  {
    application_state_t(HOSTRPC_ATOMIC(uint64_t) * val,
                        HOSTRPC_ATOMIC(uint64_t) * steps, bool show_step)
        : val(val), stepper(steps, show_step)
    {
    }
    HOSTRPC_ATOMIC(uint64_t) * val;
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

  x64_alloc_deleter store;

  hostrpc::copy_functor_memcpy_pull cp;

  using Word = uint64_t;
  using client_type = client_impl<Word, SZ, decltype(cp), stepper>;

  using server_type = server_impl<Word, SZ, decltype(cp), stepper>;

  auto send = x64_alloc<client_type::outbox_t>(N, &store);
  auto recv = x64_alloc<client_type::inbox_t>(N, &store);
  auto client_active = x64_alloc<client_type::lock_t>(N, &store);
  auto client_staging = x64_alloc<client_type::staging_t>(N, &store);
  auto server_active = x64_alloc<server_type::lock_t>(N, &store);
  auto server_staging = x64_alloc<server_type::staging_t>(N, &store);

  const uint64_t calls_planned = 1024;
  HOSTRPC_ATOMIC(uint64_t) calls_launched(0);
  HOSTRPC_ATOMIC(uint64_t) calls_handled(0);

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
          if (cl.rpc_invoke<fill, use, false>(application_state_ptr,
                                              application_state_ptr))
            {
              calls_launched++;
            }
          if (false && cl.rpc_invoke<fill, use, true>(application_state_ptr,
                                                      application_state_ptr))
            {
              calls_launched++;
            }
        }

      // printf("client done, launched %lu / %lu\n", calls_launched,
      // calls_planned);
    });

    safe_thread sv_thrd([&]() {
      auto app_state = application_state_t(&val, &server_steps, show_step);

      server_type sv = {SZ{},
                        server_active,
                        send,
                        recv,
                        server_staging,
                        &client_buffer[0],
                        &server_buffer[0]};

      void *application_state_ptr = static_cast<void *>(&app_state);
      uint32_t loc_arg = 0;
      for (;;)
        {
          if (sv.rpc_handle<operate, clear>(application_state_ptr,
                                            application_state_ptr, &loc_arg))

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
