# hostrpc
Remote procedure calls within a shared address space. Expecting 'remote' to mean across pcie.

## Assumptions
- host may be any of x64, amdgcn, pre-volta nvptx64
- client may be any of x64, amdgcn, pre-volta nvptx64
- implementation may be cuda, hip, openmp or freestanding c++
- client scheduler is not fair under contention
- host scheduler is fair under contention
- cas/faa over pci-e is expensive
- zero acceptable probability of deadlock

Opencl c++, powerpc, arch64 are intended to work but not yet tested.
Volta nvptx needs refactoring to pass control flow masks around.
Across network, as opposed to across pcie, should be implementable

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








