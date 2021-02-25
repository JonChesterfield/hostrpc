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

__attribute__((always_inline)) int hsa_spawn_with_uuid(uint32_t UUID)
{
  // Needs to be run from a kernel which runs hsa_start_routine as relaunches
  // self
  __attribute__((address_space(4))) void* p = __builtin_amdgcn_dispatch_ptr();
  auto func = [=](unsigned char* packet) {
    uint64_t addr = UUID;  // derive from UUID somehow
    __builtin_memcpy(packet + 40, &addr, 8);
  };

  // copies from p, then calls func
  enqueue_dispatch(func, (const unsigned char*)p);
  return 0;  // succeeds
}

// extern C as called from opencl
// as __device_threads_bootstrap(__global char*)

__attribute__((always_inline)) static char* get_reserved_addr()
{
  __attribute__((address_space(4))) void* p = __builtin_amdgcn_dispatch_ptr();
  return (char*)p + 48;
}

__attribute__((always_inline)) extern "C" void hsa_bootstrap_routine(void)
{
  __attribute__((address_space(4))) void* p = __builtin_amdgcn_dispatch_ptr();
  unsigned char* kernarg;
  __builtin_memcpy(&kernarg, (char*)p + 40, 8);

  __attribute__((address_space(4))) void* ks =
      __builtin_amdgcn_kernarg_segment_ptr();
  assert(kernarg == (unsigned char*)ks);

  uint32_t UUID = instance.allocate();
  assert(UUID == 0);  // not presently called while threads are already running,
                      // but could be
  if (UUID != 0)
    {
      instance.deallocate();
      return;
    }

  // Inline argument used to set initial number of threads
  uint64_t res2;
  __builtin_memcpy(&res2, get_reserved_addr(), 8);
  instance.set_requested((uint32_t)res2);

  // Might need a fence here for new kernel to see the alloc result

  // This reports the packet is malformed
  enqueue_dispatch([](unsigned char*) {}, kernarg);
}

__attribute__((always_inline)) extern "C" void hsa_set_requested()
{
  // Requested number of threads in argument field
  uint64_t res2;
  __builtin_memcpy(&res2, get_reserved_addr(), 8);
  instance.set_requested((uint32_t)res2);

  // Would be useful to launch the top level kernel here, so as to make
  // sure an instance is running. To make that work, would need to pass
  // the 64 byte packet to run in the kernarg
}

__attribute__((always_inline)) extern "C" void hsa_toplevel()
{
  // Read my uuid argument from reserved field
  uint64_t res2;
  __builtin_memcpy(&res2, get_reserved_addr(), 8);
  uint32_t uuid = res2;

  if (!instance.innerloop(toimpl, hsa_spawn_with_uuid, uuid))
    {
      // uuid exceeds requested, don't reschedule self
      return;
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
