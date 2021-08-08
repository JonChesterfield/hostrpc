#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>


bool __printf_conversion_specifier_p(char c)
{
  // excluding % which is handled elsewhere

  char specs[] = {'c', 's', 'd', 'i',
                  'o', 'x', 'X', 'u',
                  'f', 'F', 'e', 'E',
                  'a', 'A', 'g', 'G',
                  'n', 'p'};
  size_t N = sizeof(specs);

  bool hit = false;
  for (size_t i = 0; i < N; i++)
    {
      hit = hit | (c == specs[i]);
    }
  return hit;
}

bool __printf_conversion_specifier_p_faster(char c)
{
  return c;
}

int main()
{
  for (unsigned i = 0; i < 256; i++)
    {
      unsigned char u = (unsigned char)i;
      char c;
      __builtin_memcpy(&c,&u,1);

      bool current = __printf_conversion_specifier_p(c);
      bool proposed = __printf_conversion_specifier_p_faster(c);
      if (current != proposed)
        {
          printf("Differ at %u, %u != %u\n", i, (unsigned)current, (unsigned)proposed);
        }

    }

  return 0;

}
