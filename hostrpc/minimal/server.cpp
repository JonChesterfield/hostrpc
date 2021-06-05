template <typename W, typename C> bool server_t::run(W work, C clean) {
  bool in = inbox->load(std::memory_order_relaxed);
  bool out = outbox->load(std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_acquire);

  if (in & out) {
    // work done, wait for client
    while (in) {
      in = inbox->load(std::memory_order_relaxed);
    }
    std::atomic_thread_fence(std::memory_order_acquire);
  }

  if (!in & out) {
    // all done, clean up buffer
    clean(buffer);
    std::atomic_thread_fence(std::memory_order_release);
    outbox->store(0, std::memory_order_release);
    out = 0;
    return true;
  }

  if (!in & !out) {
    // nothing to do, wait for work
    while (!in) {
      in = inbox->load(std::memory_order_relaxed);
    }
    std::atomic_thread_fence(std::memory_order_acquire);
  }

  if (in & !out) {
    // do work then signal client
    work(buffer);
    std::atomic_thread_fence(std::memory_order_release);
    outbox->store(1, std::memory_order_release);
    out = 1;
    return true;
  }

  return false;
}
