
struct process_t
{
  const NS::atomic_bool *inbox;
  NS::atomic_bool *outbox;
  uint32_t *buffer;
};

struct client_t : public process_t
{
  template <typename F, typename U>
  void run(F fill, U use);
};

struct server_t : public process_t
{
  // return true if a callback was invoked
  template <typename W, typename C>
  bool run(W work, C clean);
};

// Unimplemented here
void client_fill(uint32_t *);
void server_work(uint32_t *);
void client_use(uint32_t *);
void server_clean(uint32_t *);
