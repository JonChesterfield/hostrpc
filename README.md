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