#ifndef HOSTRPC_THREAD_H_INCLUDED
#define HOSTRPC_THREAD_H_INCLUDED

namespace hostrpc
{
namespace detail
{
struct thread_impl;
thread_impl *spawn(void *(*cb)(void *), void *state);
void join(thread_impl *);
void dtor(thread_impl *impl);
}  // namespace detail

template <typename F>
struct thread
{
  thread(F *f) { impl = hostrpc::detail::spawn(cb, static_cast<void *>(f)); }

  thread(const thread &) = delete;
  thread(thread &&other)
  {
    impl = other.impl;
    other.impl = nullptr;
  };
  ~thread()
  {
    hostrpc::detail::dtor(impl);
  }
  bool valid() { return impl != nullptr; }

  void join() { hostrpc::detail::join(impl);}

 private:
  hostrpc::detail::thread_impl *impl;

  static void *cb(void *data)
  {
    F *cb = static_cast<F *>(data);
    (*cb)();
    return nullptr;
  }
};

template <typename F>
thread<F> make_thread(F *f)
{
  return thread<F>(f);
};

}  // namespace hostrpc

#endif
