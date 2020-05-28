# hostrpc
RPC within a shared address space

Primary intended use is for asking an x86 process to run code on behalf of an amdgpu attached over pcie.

Assumptions:
- gpu scheduler is not fair under contention
- host scheduler is fair under contention
- cas/faa over pci-e is expensive
- zero acceptable probability of deadlock

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








