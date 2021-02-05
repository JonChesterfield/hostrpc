#include "threads.hpp"

#include "enqueue_dispatch.hpp"

// namespace {
HOSTRPC_ATOMIC(uint32_t) implcount = 0;

uint32_t count()
{
  return platform::atomic_load<uint32_t, __ATOMIC_ACQUIRE,
                               __OPENCL_MEMORY_SCOPE_DEVICE>(&implcount);
}
void reset()
{
  platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                         __OPENCL_MEMORY_SCOPE_DEVICE>(&implcount, 0);
}

void toimpl()
{
  platform::atomic_fetch_add<uint32_t, __ATOMIC_ACQ_REL,
                             __OPENCL_MEMORY_SCOPE_DEVICE>(&implcount, 1);
}
//}  // namespace

hostrpc::threads::ty<16> instance;

#if HOSTRPC_HOST

#include "catch.hpp"
#include <pthread.h>

static int pthread_create_detached(void* (*start_routine)(void*), void* arg)
{
  int rc = 0;
  pthread_attr_t attr;

  rc = pthread_attr_init(&attr);
  if (rc != 0)
    {
      return 1;
    }

  pthread_t handle;
  rc = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  if (rc != 0)
    {
      (void)pthread_attr_destroy(&attr);
      return 1;
    }

  rc = pthread_create(&handle, NULL, start_routine, arg);
  (void)pthread_attr_destroy(&attr);

  return rc;
}

static void* pthread_start_routine(void* p);

static int pthread_spawn_with_uuid(uint32_t UUID)
{
  uint64_t s = UUID;
  void* arg;
  __builtin_memcpy(&arg, &s, 8);
  return !!pthread_create_detached(pthread_start_routine, arg);
}

static void* pthread_start_routine(void* p)
{
  uint64_t u64;
  __builtin_memcpy(&u64, &p, 8);
  uint32_t u32 = static_cast<uint32_t>(u64);

  while (instance.innerloop(toimpl, pthread_spawn_with_uuid, u32))
    ;
  return NULL;
}

namespace hostrpc
{
namespace threads
{
void bootstrap()
{
  if (instance.alive() == 0)
    {
      instance.spawn(pthread_spawn_with_uuid);
    }
}

}  // namespace threads
}  // namespace hostrpc

#endif

#if HOSTRPC_AMDGCN

// On the stack, this hits an infinite recursion in instcombine
// Probably due to the address spaces
// Ignore that for now, as the buffer can be elided later anyway
alignas(64) __attribute__((address_space(3)))
__attribute__((loader_uninitialized)) static unsigned char buf[64];

__attribute__((always_inline)) int hsa_spawn_with_uuid(uint32_t UUID)
{
  __attribute__((address_space(4))) void* p = __builtin_amdgcn_dispatch_ptr();
  __builtin_memcpy(&buf, (char*)p, 64);

  uint64_t addr = UUID;  // derive from UUID somehow
  __builtin_memcpy((char*)buf + 40, &addr, 8);

  enqueue_dispatch((unsigned char*)buf);
  return 0;  // succeeds
}

// extern C as called from opencl
// as __device_threads_bootstrap(__global char*)
__attribute__((always_inline)) extern "C" void hsa_start_routine()
{
  __attribute__((address_space(4))) void* p = __builtin_amdgcn_dispatch_ptr();
  // p is a hsa_kernel_dispatch_packet_t, which contains a kernarg address at
  // offset 40 required to be 16 byte aligned according to spec. Probably costs
  // ~ 16 pages, maybe choose a better encoding.
  uint64_t addr;
  __builtin_memcpy(&addr, (char*)p + 40, 8);

  uint32_t uuid = addr;  // derive from addr somehow

  const unsigned rep = 1;

  for (unsigned r = 0; r < rep; r++)
    {
      if (!instance.innerloop(toimpl, hsa_spawn_with_uuid, uuid))
        {
          return;
        }
    }

  enqueue_self();
  return;
}

#endif

#if HOSTRPC_HOST
extern "C" int usleep(unsigned);  // #include <unistd.h>
void sleep(unsigned x) { usleep(1000000 * x); }

TEST_CASE("Spawn thread with requested == 0")
{
  using namespace hostrpc;
  using namespace threads;
  reset();

  CHECK(instance.alive() == 0);
  CHECK(instance.requested() == 0);

  instance.spawn(pthread_spawn_with_uuid);

  CHECK(((instance.alive() == 0) || (instance.alive() == 1)));
  CHECK(instance.requested() == 0);

  while (instance.alive())
    {
    }

  CHECK(count() == 0);
}

TEST_CASE("Bootstrap does nothing with zero req")
{
  using namespace hostrpc;
  using namespace threads;
  reset();

  CHECK(instance.alive() == 0);
  CHECK(instance.requested() == 0);
  bootstrap();
  while (instance.alive())
    ;
  CHECK(instance.requested() == 0);
  CHECK(count() == 0);
}

TEST_CASE("Bootstrap creates a thread")
{
  using namespace hostrpc;
  using namespace threads;
  reset();

  CHECK(instance.alive() == 0);
  CHECK(instance.requested() == 0);
  instance.set_requested(1);
  CHECK(instance.alive() == 0);
  fprintf(stderr, "Bootstrap creates a thread\n");

  CHECK(count() == 0);
  bootstrap();
  while (count() == 0)
    ;
  CHECK(instance.alive() == 1);

  instance.set_requested(0);
  while (instance.alive())
    ;
}

TEST_CASE("Spawn thread with requested == 1")
{
  using namespace hostrpc;
  using namespace threads;
  reset();

  CHECK(instance.alive() == 0);
  CHECK(instance.requested() == 0);

  instance.set_requested(1);
  CHECK(instance.alive() == 0);
  CHECK(instance.requested() == 1);

  fprintf(stderr, "Go\n");
  instance.spawn(pthread_spawn_with_uuid);
  CHECK(instance.alive() == 1);
  CHECK(instance.requested() == 1);

  if (0)
    {
      sleep(3);
      instance.set_requested(4);
      sleep(3);
      instance.set_requested(1);
      sleep(3);
      instance.set_requested(8);
      sleep(3);
      instance.set_requested(3);
      sleep(3);
      instance.set_requested(16);
      sleep(3);
    }

  instance.set_requested(0);
  while (instance.alive())
    {
    }
}
#endif
