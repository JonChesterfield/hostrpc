template <typename F, typename U>
void client_t::run(F fill, U use)
{
  bool in = inbox->load(NS::memory_order_acquire);
  bool out = outbox->load(NS::memory_order_relaxed);
  NS::atomic_thread_fence(NS::memory_order_acquire);

  if (!in & !out)
    {
      // ready! write to buffer then to outbox
      fill(buffer);
      NS::atomic_thread_fence(NS::memory_order_release);
      outbox->store(1, NS::memory_order_release);
      out = 1;
    }

  if (!in & out)
    {
      // wait for result
      while (!in)
        {
          in = inbox->load(NS::memory_order_relaxed);
          yield();
        }
      NS::atomic_thread_fence(NS::memory_order_acquire);
    }

  if (in & out)
    {
      // read from buffer then write to outbox
      use(buffer);
      NS::atomic_thread_fence(NS::memory_order_release);
      outbox->store(0, NS::memory_order_release);
      out = 0;
    }

  if (in & !out)
    {
      /// wait for server to garbage collect
      while (in)
        {
          in = inbox->load(NS::memory_order_relaxed);
          yield();
        }
      NS::atomic_thread_fence(NS::memory_order_acquire);
    }
}
