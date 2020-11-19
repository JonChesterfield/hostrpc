#include "hostrpc_thread.hpp"

#include <new>

#include <pthread.h>

namespace hostrpc
{
namespace detail
{
struct thread_impl
{
  thread_impl(pthread_t id) : id(id) {}
  pthread_t id;
};

thread_impl *spawn(void *(*cb)(void *), void *state)
{
  pthread_t id;
  int rc = pthread_create(&id, NULL, cb, state);

  if (rc == 0)
    {
      return new (std::nothrow) thread_impl(id);
    }
  else
    {
      return nullptr;
    }
}

void join(thread_impl *impl)
{
  if (impl)
    {
      void *value_ptr;
      pthread_join(impl->id, &value_ptr);
    }
}

void dtor(thread_impl *impl)
{
  if (impl)
    {
      delete (impl);
    }
}
}  // namespace detail
}  // namespace hostrpc
