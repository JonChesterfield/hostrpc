# Hostrpc

## Archived

Core implementation shipped as plumbing in llvm libc. Future performance work
and documentation is likely to be found at https://libc.llvm.org/gpu/rpc.html

Credit to my colleague and paper coauthor Joseph Huber for getting this across
the valley from prototype to production. This project would not have contributed
to llvm libc were he not determined to make that so.

Thank you also to the external reviewers for their ongoing encouragement.

### Successes

Core state machine works. RPC can be implemented on shared memory. This repo
has the core structure in detail/state_machine.hpp, the translated class in llvm
is mostly in [libc/src/__support/RPC/rpc.h](https://github.com/llvm/llvm-project/blob/main/libc/src/__support/RPC/rpc.h)

Loaders for freestanding GPU code that define main() work. In this repo tools/loader,
in llvm [libc/utils/gpu/loader](https://github.com/llvm/llvm-project/tree/main/libc/utils/gpu/loader). This gives a way to write
C++ code that defines main() as usual and runs as ./a.out or similar. This is
mostly used to run test code.

Two papers (llpp and not yet presented) and a talk on linear types.

### Partial results

Correct by construction concurrent data structures can be done. The state
machine reified in the type system, lifetimes handled by linear types. This
gives a C++ program which fails to compile if it would deadlock at runtime.
However, the programmer ergonomics of this are so poor that the production
code does not have these guard rails, relyng on code review instead.

Running the same code on x64 or on amdgpu is indeed helpful for debugging.
A partial result because the thread sanitizers I wanted to use do not support
thread fences and the implementation uses thread fences for most things.

### Failures

Code can be written in the subset of freestanding C++, opencl, cuda. hip,
openmp. It shouldn't be. The payoff is poor relative to exposing a C header
and compiling freestanding C++ to LLVM IR. OpenCL in particular rules out
a lot of C++, e.g. you can't infer types from functions because it errors
on the mere idea of a function pointer.

Expanding printf in the C preprocessor can be done but you really don't want to.

I thought this could ship as headers included in the OpenMP language runtime. I
did not sell that idea adequately.

Writing this without compiler patches was reasonable but taken too far, in
particular some of the nvptx workarounds would have been easier fixed upstream.

This was intended to be ported to Xilinx but I have not (yet?) done so.

### Conclusions taken forward

Correct by construction concurrency is the right thing. It requires linear
types. In general it probably requires a more expressive type sysem than C++.

Abstracting over amdgpu, nvptx, x64 is a really useful thing for differential
debugging of compiler implementations.


## Readme entry from before archiving below.

Compiler enforced mutual exclusion between heterogenous processors.

Given an array of a trivially copyable type in shared memory, and a few
administrative bitmaps, this library will provide compiler-enforced mutual
exclusion over elements of that array between heterogenous processors.

This is sufficient for safe remote procedure calls. Examples:
- A gpu warp calling a function on the host machine to allocate memory
- An x64 machine calling a function on an amdgpu to do arithmetic
- A Linux process providing audited syscall access to a seccomp'ed one
- A gpu calling a function on another gpu on the same pcie bus

The target use case is low level heterogenous programming. The specific use
that started this project was performing memory allocation and I/O from a GPU.
Low level in the sense that one should be able to build parts of libc on this.
It is in the subset of freestanding c++ that is acceptable to opencl.

Correctness and performance are prioritised over programmer ergonomics. For
example, strictly correct code on Volta is required to pass a thread mask
representing the CFG to intrinsic functions. This library therefore accepts
a thread mask to each function. As this mask affects control flow and is
sometimes known at compile time, the active_threads parameter passed
everywhere can be a compile time value as well as a runtime value.


## Pseudocode interface

Where a port is a typed wrapper around an index into the array under management.

```
port open();                 // Waits for a port
void use(port, callback);    // Safe, we own it
port send(port);             // Also safe
// port use(port, callback); // Won't compile
port recv(port);             // Wait for the other side
port use(port, callback);    // OK again
close(port);                 // Done
```

## Pseudocode vs reality

### C++ linear types

The compile time enforced exclusion claim requires using callbacks for the
entire interface or using the -Werror=consumed Clang extension. For example,
open could take a callback which is only invoked if a port was available.
Consumed annotations make for linear control flow instead of CPS and are
preferred here.

This implementation represents everything which is known at compile time in
the type of the port object and tracks everything which is discovered at
runtime using linear types implemented using the consumed annotation. Thus
each `port` type in the pseudocode above is a move-only type template. The
representation looks more like:

```
template <unsigned I, unsigned O>
struct port_t;

template <unsigned S>
port_t<S,S> open(); // I == O from open

template <unsigned S, typename F>
void use(port_t<S,S> const& port, F cb); // Requires I == O

template <unsigned S>
port_t<S, !S> send(port_t<S, S> && port); // Changes O parameter

```

The template parameters are sufficient to record that the return value from
send cannot be passed to use. The two integers do not match, clang will error
that there is no use() function it can call.

They are not sufficient to prevent all the errors in the following:

```
// Open a new port and send it
port_t<0, 0> first open();
port_t<0, 1> sent send(move(first));

// Can't call send again on the returned type, doesn't have I == 0
// send(move(sent));

// But can call send again on the original variable
send(move(first));

// And, maybe worse, can still use the buffer we no longer own
void use(first, cb);

close(first); // Should also not compile
close(sent);  // This should compile

```

Linear types are sufficient to prevent the erroneous two calls above. First
was used exactly once in moving it to send so further uses raise errors.

The only use of sent above that passed the template type checking is to close,
which compiles successfully.

Further, if that trailing close was missing, the port would leak. The consumed
annotation rejects programs that miss this cleanup. It's a linear type, not an
affine type.


### Out of resource

The array has finite length and each element corresponds to a port. Literally
an index into that array with type metadata wrapped around it. Thus open can
fail when all element indices are currently opened as ports.

This can either be solved by waiting, as in the pseudocode or a open_blocking()
call, or by indicating failure. The C++ options there are:
- Throw an exception
- Return a sentinel value (-1 / ~0 probably)
- Return std::optional<>
- Return structure equivalent to optional

Returning a sentinel works but puts a branch (or a runtime assert) at the start
of each API call. Exceptions also introduce control flow and aren't always
available on hardware targets. This library doesn't assume libc++ exists yet, so
no std::optional.

Port open() therefore returns a hostrpc::maybe<port> which is essentially an
optional with clangs consumed annotation. The canonical usage pattern is:

```
auto maybe = s.open();
if (!maybe) {
  return;
}
auto port = maybe.value();
```

The `.value()` method will raise a error under clang -Werror=consumed if the
branch on `explicit operator() bool` is missing. Likewise if `.value()` is
not called, the destructor will raise an error under -Werror=consumed.

### Schedulers may not be fair

In particular, the OpenCL model makes no assurance about whether a descheduled
thread will run again. HSA is not much stronger. Waiting in a busy spin can
be a problem there.


| Operation | Scheduling properties                     |
|-----------|-------------------------------------------|
| open      | Bounded time, fails on out of resource.   |
| close     | Bounded time, cannot fail.                |
| use       | Branch free if callback is branch free.   |
| send      | Bounded time, cannot fail.                |
| recv      | Unbounded time, cannot fail. Blocking.    |
| query     | Bounded time, returns either<>.           |  

The difficult case for unfair schedulers is recv. It waits for the other agent,
potentially for unbounded lengths of time. This is unsound as a primitive.

It is possible to test whether recv will complete without waiting beforehand:
```
while (!recv_finite_time(port))
  ;
port = recv(port);
```

The query operation is a similar idea. It does a single load from the other
agent. If that indicates that the agent is finished with the buffer, it returns
the same type that recv would do. Otherwise, it returns its argument unchanged.
Because the element ownership is encoded in the type system, this would be a
function which returns different types based on a runtime value, that is:

```
either<typed_port<0, 1>, // unchanged
       typed_port<1, 1>  // recv succeeded
      > query(typed_port<0, 1> && port);
```

The either<> type here is implemented similarly to the maybe<> type used to
indicate failure to open a port. It is itself a linear type. The port argument
was consumed by query and the either<> returned owns that same element. Exactly
one of the two types contained within either<> corresponds to the result.

## Components

### port_t<>

### maybe

### either

### bitmap

## state machine



## What is this?

Define a trivially copyable type BufferElement. Put an array of N of them in
shared memory, and allocate a couple of bitmaps in the same shared memory.

Given those pointers, this code will coordinate mutually exclusive access to
that array of BufferElements. Each is always unambiguously owned by one of the
two systems, and that ownership is tracked by the C++ type system.

The current owner has read/write access to that element. It can transfer
ownership to the other system. It can also query for whether the other system
has given the buffer back yet. It has neither read nor write access until then.

This repo is therefore an overengineered type system over two bitmaps used to
track and change which of two systems currently owns a given array element.

Since shared memory does not fail without killing the attached processors,
this is sufficient to implement remote procedure calls which do not embed
additional network failure related noise in their interface. Thus
`void *malloc(size_t)` executing on the other system looks the same as local.

This particular repo doesn't have much to say about the syntax above. Syntax is
contentious, with people split on whether it should be conjured by compilers,
code generators, the C preprocessor or by typing. The expectation should be
that this repo will give you `void syscall(uint64_t data[8])` easily
and that serialising arbitrary C++ types across that is largely an exercise for
the layer above.

## Requirements

Shared memory between two processors that wish to communicate. The llpp paper
in this repo assumed only load and store operations on the shared memory, the
present implementation requires atomic fetch_add as well. This is available on
pcie and that assumption simplifies the implementation.

There is an application-specific setup phase which involves writing a few
pointers to shared memory to each processor. This is difficult to abstract over
as different environments have differing ideas about how best to do that.
Similarly at end of execution something should deallocate those pointers.

Compiled and tested on x64, amdgpu, nvptx, under freestanding C++, openmp, cuda,
hip. OpenCL cannot currently express the linear type system but otherwise builds.



## Implementation status

Alpha. Works for me. May work for you if held carefully. Core algorithm believed correct, test code and surrounding infra less solid.
Doesn't have build scripts. Need to rework some nvidia specific atomic hacks to get the entire implementation header-only.


## Assumptions
- host may be any of x64, amdgcn, nvptx64
- client may be any of x64, amdgcn, nvptx64
- implementation may be openmp, opencl, cuda, hip or freestanding c++
- client scheduler is not fair under contention
- host scheduler is fair under contention
- cas/faa over pci-e is expensive
- zero acceptable probability of deadlock


Opencl c++, powerpc, arch64 are intended to work but not yet tested.
Volta nvptx needs control flow masks passed around. This impacts the API for other targets.
Across network, as opposed to across pcie, should be implementable.

## Interface

Assume for the moment that an instance of a 'client type' exists on the client, and a matching instance of a 'server type' exists on the server. How those are constructed is described further down.

One method on the client instance is:
```
template <typename Fill, typename Use>
bool rpc_invoke(Fill f, Use u);
```
where Fill and Use are types that expose
```
struct cacheline_t {
  uint64_t element[8];
};
struct page_t {
  cacheline_t cacheline[64];
};
struct some_fill {
  void operator()(page_t *);
};
```

Similarly, a method on the server instance is:
```
template <typename Operate, typename Clear>
bool rpc_handle(Operate op, Clear cl); 
```

Given a server (thread or wavefront) that repeatedly calls rpc_handle, the behaviour exposed then follows:

```
// On the client
Fill fill;
Use use;
client.rpc_invoke(fill, use)
{
  page_t * page = get_page();
  fill(page); // write stuff to page

  // time passes

  use(page);
  drop_page(page);
}
```

```
// On the server
Operate operate;
Clear clear;

// given data from invoke(fill):
server.rpc_handle(operate, clear)
{
  page_t * page = get_page();
  operate(page);
}

// given data from invoke(use):
server.rpc_handle(operate, clear)
{
  page_t * page = get_page();
  clear(page);
}

```

The clear() hook is useful for writing no-op across the page before it gets reused by another fill/operate/use sequence.

# State machine description
All assume one server, one client. Muliple devices or queues can use multiple RPC structures.
'Inbox' and 'outbox' refer to single element queues.

## Single thread, single task per device

Each device has write access to an outbox and read access to an inbox.
A client/server pair uses two mailboxes in total. Writes to the outbox become visible as reads from the inbox after some delay.

| Mailbox | Client | Server |
|:-:      |:-:     |:-:     |
| A       | Inbox  | Outbox |
| B       | Outbox | Inbox  |

### Server
|Inbox  | Outbox | State               |
|:-:    |:-:     |:-                   |
| 0     | 0      | Idle / waiting      |
| 0     | 1      | Garbage to clean up |
| 1     | 0      | Work to do          |
| 1     | 1      | Waiting for client  |

1. Waits for work
2. Gets request to do work
3. Allocates only block
4. Does said work
5. Posts that the work is done
6. Waits for client to acknowledge
7. Deallocates only block
8. Goto 1

### Client
|Inbox  | Outbox | State                  |
|:-:    |:-:     |:-                      |
| 0     | 0      | Idle                   |
| 0     | 1      | Work posted for server |
| 1     | 0      | Finished with result   |
| 1     | 1      | Result available       |

#### Synchronous
1. Called by application
2. Runs garbage collection
3. Allocates only block
4. Writes arguments into only block
5. Posts that work is available
6. Waits for result from server
7. Calls result handler on said result
8. Posts that result is no longer needed
9. Deallocates only block
10. Return to application

#### Asynchronous
1. Called by application
2. Runs garbage collection
2. Allocates only block
3. Writes arguments into only block
4. Posts that work is available
5. Return to application

### Client / server, synchronous call

Each mailbox transitions from 0 to 1 and then from 1 to 0 exactly once per call.
At most one value changes on each event.
Inbox reads lag the corresponding outbox write.
Buffer is accessible by at most one of client and server at a time.

|Client Inbox | Client Outbox | Server Inbox | Server Outbox | Buffer accessible | Event                   |
|:-:          |:-:            |:-:           |:-:            |:-                 |:-                       |
|0            | 0             | 0            | 0             |                   | Idle                    | 
|0            | 0             | 0            | 0             | Client            | Application call        | 
|0            | **1**         | 0            | 0             |                   | Client post             | 
|0            | 1             | **1**        | 0             | Server            | Server updated inbox    | 
|0            | 1             | 1            | 0             | Server            | Server does work        | 
|0            | 1             | 1            | **1**         |                   | Server post             | 
|**1**        | 1             | 1            | 1             | Client            | Client updated inbox    | 
|1            | 1             | 1            | 1             | Client            | Client processes result |
|1            | **0**         | 1            | 1             |                   | Client post             |
|1            | 0             | **0**        | 1             |                   | Server updated inbox    |
|1            | 0             | 0            | 1             |                   | Server garbage collect  |
|1            | 0             | 0            | **0**         |                   | Server post             |
|**0**        | 0             | 0            | 0             |                   | Client updated inbox    |


## Multiple thread, multiple task per device
Replace the single bit mailbox with a bitmap representing N mailboxes for up to N simultaneous tasks.
The inbox remains read-only. Introduce an array of locks on each device to coordinate access to outbox. The invariant is that outbox may only be written to (set or cleared) while the corresponding lock is held.

|Inbox  | Outbox | Lock   | State                |
|:-:    |:-:     |:-:     |:-                    |
| 0     | 0      | 0      | Idle / waiting       |
| 0     | 0      | 1      | Idle thread          |
| 0     | 1      | 0      | Garbage to clean up  |
| 0     | 1      | 1      | Garbage with owner   |
| 1     | 0      | 0      | Work to do           |
| 1     | 0      | 1      | Work with owner      |
| 1     | 1      | 0      | Waiting for client   |
| 1     | 1      | 1      | Work done with owner |


Instantiation is per host-gpu pair, and per hsa queue on the gpu. May use one or more host threads per instantiation.

Terminology:
GPUs use 'thread' to refer to a lane of a warp or wavefront. This hostrpc is not per-thread in that sense. It is per wave - e.g. on gfx9 64 threads make a hostrpc call together. No assumptions that all waves within a workground make the call together, or at all. Also no assumption that all lanes are active.

Communication between cpu and gpu options:

Shared memory. Read/write to the common address space. Care needed over synchronisation scopes. There are atomic pcie operations available. Essentially polling.

GPU can raise an interrupt on the CPU, e.g. via the hsa signal mechanism. The signals are reliable, even when the underling handler can drop interrupts, as there's a fallback path and mailboxes involved. Each signal imposes some performance overhead on the CPU.

A CPU can launch a kernel on the GPU, e.g. to run code that modifies atomic state on the GPU. Efficient but unreliable - if the GPU is fully occupied, the new kernel will still run, but with potentially unfortunate interleaving with the existing kernels.


Proposal:
CPU reads data from the GPU to discover work to do. GPU reads data from the CPU to discover when the work has been done. Interrupts are raised at an as-yet-undetermined frequency to wake cpu threads(s) to to this work. Queue is kicked off the GPU at as-yet-undetermined frequency, possibly based on the number of hostcalls outstanding, to stop waiting kernels blocking other queues.

Let there be N independent slots. If N >= number-cu * number-smid * number-waves, e.g. N = 2048, then the queue can't starve itself. I.e. if one queue owns the whole GPU, and every wave makes a hostcall at the same time, every wave will have a slot available.

The slots are independent in that a wave reading/writing to one has no effect on other waves. In particular, acquiring a slot must not require cooperation from another wave, as said cooperation may never occur.

An asynchronous call occupies a slot until the host has read the corresponding data, so if there are some asynchronous calls outstanding then the wave may have to spin until the host unblocks a slot. The wave waiting on the host does not deadlock as the host scheduler is fair.

Bitmaps:
Distinguish between cpu-internal, cpu-external, gpu-internal, gpu-external. The external ones are periodically read by the other machine, the internal ones never are.

Call process:

Slot is a uint16_t representing an offset into an array of N 4kb pages in host memory.
Four booleans involved per slot. Trying to give them unambiguous names - thread is on the x86 host. All four zero indicates available, though thread is not directly readable. Process starts on a wave.

GPU, WAVE, HOST, THREAD
GPU is written by device, read by host. HOST is written by host, read by device.
WAVE is RW only by device, THREAD is RW only by HOST

Available
  G W H T
H 0   0 0
D 0 0 0

Wave acquires a slot by CAS into W. If the CAS fails, try another slot index
  G W H T
H 0   0 0
D 0 1 0


Wave writes 4k into the kth slot on the host. Sets G when the write is complete with CAS.
No risk of another wave writing to that slot, but it's a bitmap and can't lose other wave's update.
  G W H T
H 0   0 0
D 1 1 0

Host reads gpu bitmap, compares to host bitmap. G1 && H0 indicates work to do.
  G W H T
H 1   0 0
D 1 1 0


Host thread acquires a slot that has the bit set in G by CAS into T. If CAS fails, try another slot, or do something else.

  G W H T
H 1   0 1
D 1 1 0

Host thread now exclusively holds the slot. Reads the data, writes stuff back. Atomic cas into the known zero H to avoid losing other updates.

  G W H T
H 1   1 1
D 1 1 0

Host thread no longer needs to do anything, CAS 0 into T if H=1 or G=1
  G W H T
H 1   1 0
D 1 1 0

GPU is spinning on reads to H. Replaces its local cache of the value:

  G W H T
H 1   1 1
D 1 1 1

H set for current slot, G & W will still be set. Read the results out.

Clearing G will be noticed by the host and interpreted as slot available.
Clearing W will be noticed by the GPU and interpreted as slot available.
Thus clear G first, then W. 

  G W H T       G W H T
H 1   1 1  >  H 0   1 1
D 0 1 1       D 0 1 1  

    v

  G W H T
H 1   1 1
D 0 0 1  

GPU now considers the slot is available, but we haven't yet reached all-zero to start again

G=0 is noticed by host after some delay, indicates that gpu is no longer interested in that slot

  G W H T       G W H T       G W H T
H 0   1 1  >  H 0   0 1  >  H 0   0 0
D 0 1 1       D 0 1 1         D 0 1 1  

H=0 is noticed by gpu after some delay, indicates that host is no longer interested in that slot

  G W H T
H 0   0 0
D 0 1 0  

Once G & H both zero, can zero W and return.

Quite a lot of transitions. 64 states. Some can occur independently of processing a given call.

H=1 => T->0, because T publishes to H to indicate it's finished

Device G=1 -> host   G->1
Host   H=1 -> device H->1

