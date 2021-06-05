template <typename F, typename U> void client_t::run(F fill, U use) {
  bool in = inbox->load(memory_order_relaxed);
  bool out = outbox->load(memory_order_relaxed);
  atomic_thread_fence(memory_order_acquire);

  if (!in & !out) {
    // ready! write to buffer then to outbox
    fill(buffer);
    atomic_thread_fence(memory_order_release);
    outbox->store(1, memory_order_release);
    out = 1;
  }

  if (!in & out) {
    // wait for result
    while (!in) {
      in = inbox->load(memory_order_relaxed);
    }
    atomic_thread_fence(memory_order_acquire);
  }

  if (in & out) {
    // read from buffer then write to outbox
    use(buffer);
    atomic_thread_fence(memory_order_release);
    outbox->store(0, memory_order_release);
    out = 0;
  }

  if (in & !out) {
    /// wait for server to garbage collect
    while (in) {
      in = inbox->load(memory_order_relaxed);
    }
    atomic_thread_fence(memory_order_acquire);
  }
}
