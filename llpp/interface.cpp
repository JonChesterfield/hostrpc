struct callback_t {
  void operator()(char *buffer);
};

struct client {
  client(void *);
  
  template <typename Fill, typename Use>
  bool rpc_invoke(Fill fill, Use use) noexcept;

  template <typename Fill> bool rpc_invoke(Fill fill) noexcept;

private:
  void *state;
};

struct server {
  server(void *);
  
  template <typename Operate, typename Clear>
  bool rpc_handle(Operate op, Clear cl) noexcept;

private:
  void *state;
};
