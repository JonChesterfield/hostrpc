#include "catch.hpp"

#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "memory.hpp"
#include "tests.hpp"

#include <list>
#include <thread>

#include "allocator.hpp"

struct x64_alloc_deleter
{
  struct D
  {
    void operator()(void *d)
    {
      hostrpc::allocator::host_libc_impl::deallocate(d);
    }
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
  void *memory =
      hostrpc::allocator::host_libc_impl::allocate(align, size * bps);
  typename T::Ty *m =
      hostrpc::careful_array_cast<typename T::Ty>(memory, size * bps);
  (*store)(m);
  return {m};
}

TEST_CASE("set up single word system")
{
  CHECK(0);
  return;  // disable for now

  using namespace hostrpc;
  constexpr size_t N = 64;
  page_t shared_buffer[N];

  HOSTRPC_ATOMIC(uint64_t) val(UINT64_MAX);

  struct fill
  {
    fill(_Atomic(uint64_t) * s) : state(s) {}
    _Atomic(uint64_t) * state;
    void operator()(page_t *p) { p->cacheline[0].element[0] = *state; }
  };

  struct use
  {
    void operator()(page_t *) {}
  };

  struct operate
  {
    void operator()(page_t *p)
    {
      uint64_t r = p->cacheline[0].element[0];
      // printf("Server received %lu, forwarding as %lu\n", r, 2 * r);
      p->cacheline[0].element[0] = 2 * r;
    }
  };

  struct clear
  {
    void operator()(page_t *) {}
  };

  using SZ = hostrpc::size_compiletime<N>;

  x64_alloc_deleter store;

  using Word = uint64_t;
  using client_type = client_impl<Word, SZ>;

  using server_type = server_impl<Word, SZ>;

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
      client_type cl = {SZ{}, client_active,  recv,
                        send, client_staging, &shared_buffer[0]};

      fill f(&val);
      use u;
      while (calls_launched < calls_planned)
        {
          if (cl.rpc_invoke<fill>(f))
            {
              calls_launched++;
            }
          if (false && cl.rpc_invoke<fill, use>(f, u))
            {
              calls_launched++;
            }
        }

      // printf("client done, launched %lu / %lu\n", calls_launched,
      // calls_planned);
    });

    safe_thread sv_thrd([&]() {
      server_type sv = {SZ{}, server_active,  send,
                        recv, server_staging, &shared_buffer[0]};

      uint32_t loc_arg = 0;
      for (;;)
        {
          if (sv.rpc_handle<operate, clear>(operate{}, clear{}, &loc_arg))

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

          platform::sleep();
        }
      while ((calls_launched != calls_handled) ||
             (calls_launched != calls_planned));
    }

    printf("Done, now just waiting on threads\n");
  }
}
