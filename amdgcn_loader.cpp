#include "hsa.hpp"

int main(int argc, char** argv)
{
  hsa::init hsa_state;

  if (argc < 2) {
    fprintf(stderr, "Require at least one argument\n");
  }
  
  hsa_agent_t kernel_agent;
  if (HSA_STATUS_INFO_BREAK !=
      hsa::iterate_agents([&](hsa_agent_t agent) -> hsa_status_t {
        auto features = hsa::agent_get_info_feature(agent);
        if (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH)
          {
            kernel_agent = agent;
            return HSA_STATUS_INFO_BREAK;
          }
        return HSA_STATUS_SUCCESS;
      }))
    {
      fprintf(stderr, "Failed to find a kernel agent\n");
      return 1;
    }


  FILE* fh = fopen(argv[1], "rb");
  int fn = fh ? fileno(fh) : -1;
  if (fn < 0)
    {
      fprintf(stderr, "Failed to open file %s\n", argv[1]);
      return 1;
    }

  hsa::executable ex(kernel_agent, fn);
  if (!ex.valid())
    {
      fprintf(stderr, "HSA failed to load contents of %s\n", argv[1]);
      return 1;
    }
  

  // need to be able to handle failure here
  hsa_executable_symbol_t symbol = ex.get_symbol_by_name("device_entry.kd");
  (void)symbol;
    
  
  for (int i = 0; i < argc; i++)
    {
      printf("argv[%d] = %s\n", i, argv[i]);
    }

  return 0;
  
}
