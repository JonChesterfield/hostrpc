#if HOSTRPC_AMDGCN
#pragma omp declare target

// overrides weak functions in target_impl.hip
extern "C" {
void *__kmpc_impl_malloc(size_t);
void __kmpc_impl_free(void *);
}

using client_type = hostrpc::x64_gcn_type<hostrpc::size_runtime>::client_type;
static client_type *get_client();

void *__kmpc_impl_malloc(size_t x) {
  uint64_t data[8] = {0};
  data[0] = opcodes_malloc;
  data[1] = x;
  fill f(&data[0]);
  use u(&data[0]);
  client_type *c = get_client();
  bool success = false;
  while (!success) {
    success = c->rpc_invoke(f, u);
  }
  void *res;
  __builtin_memcpy(&res, &data[0], 8);
  return res;
}

void __kmpc_impl_free(void *x) {
  uint64_t data[8] = {0};
  data[0] = opcodes_free;
  __builtin_memcpy(&data[1], &x, 8);
  fill f(&data[0]);
  client_type *c = get_client();
  bool success = false;
  while (!success) {
    success = c->rpc_invoke(f); // async
  }
}

#pragma omp end declare target
#endif
