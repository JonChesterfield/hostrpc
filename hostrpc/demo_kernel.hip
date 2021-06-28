__global__ extern "C" void on_gpu(
    hostrpc::x64_gcn_type<SZ>::client_type *client, int, char **, int *)
{
  auto inv = [=](uint64_t x[8]) -> bool {
    return invoke<hostrpc::x64_gcn_type<SZ>::client_type>(client, x);
  };

  const uint64_t buffer_size = 16;
  uint64_t tmp[8];
  tmp[0] = hostrpc::allocate_op_hsa;
  tmp[1] = buffer_size;
  inv(tmp);

  char *buf = (char *)tmp[0];

  buf[0] = 'h';
  buf[1] = 'i';
  buf[2] = '\n';
  buf[3] = '\0';

  tmp[0] = hostrpc::syscall_op;
  tmp[1] = __NR_write;
  tmp[2] = 2;
  tmp[3] = (uint64_t)buf;
  tmp[4] = 3;

  inv(tmp);

  tmp[0] = hostrpc::syscall_op;
  tmp[1] = __NR_fsync;
  tmp[2] = 2;

  inv(tmp);

  tmp[0] = hostrpc::free_op_hsa;
  tmp[1] = (uint64_t)buf;
  tmp[2] = buffer_size;
  inv(tmp);
}
