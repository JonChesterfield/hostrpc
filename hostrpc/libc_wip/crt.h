#ifndef LIBC_CRT_H_INCLUDED
#define LIBC_CRT_H_INCLUDED

// maybe saner than the individual offsets used at present
struct arg_type
{
  int argv;
  char ** argv; // pointers into strtab

  char strtab[];
};

#endif
