template <typename F, typename U> void client_t::run(F fill, U use) {
  bool in = inbox->load(std::memory_order_relaxed);
  bool out = outbox->load(std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_acquire);

  if (!in & !out) {
    // ready! write to buffer then to outbox
    fill(buffer);
    std::atomic_thread_fence(std::memory_order_release);
    outbox->store(1, std::memory_order_release);
    out = 1;
  }

  if (!in & out) {
    // wait for result
    while (!in) {
      in = inbox->load(std::memory_order_relaxed);
    }
    std::atomic_thread_fence(std::memory_order_acquire);
  }

  if (in & out) {
    // read from buffer then write to outbox
    use(buffer);
    std::atomic_thread_fence(std::memory_order_release);
    outbox->store(0, std::memory_order_release);
    out = 0;
  }

  if (in & !out) {
    /// wait for server to garbage collect
    while (in) {
      in = inbox->load(std::memory_order_relaxed);
    }
    std::atomic_thread_fence(std::memory_order_acquire);
  }
}
