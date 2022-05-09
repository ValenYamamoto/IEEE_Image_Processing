## VPI

VPI (Vision Programming Interface) is computer vision library that makes the
use of various hardware accelerators very easy

### VPI Backends

A VPI backend is any device VPI can run on, which is one of the following:

#### CPU
Your central processing unit. Even though it does have any particular
graphics-minded accelerations, if all other accelerators are being used, the CPU
can also be used to calculations, just a bit slower.

#### CUDA
The name for any CUDA-enabled GPU. VPI will abstract away any actual CUDA code,
which makes using the GPU much easier.

#### PVA
PVA (Programmable Vision Accelerator) is a specialized processor for image
processing/computer vision. Not in the Jetson though :(

#### VIC
VIC (Video Image Compositor) is a device specialized for low-level image
processing functions (rescaling, color space conversion, etc.). Good for
offloading non-critical tasks from GPU to free GPU for other more calculation
intense operations

### General Use
In python, specifying backends is done mostly with context managers, like as
follows:

```python
with vpi.Backend.CUDA:
	inp = image.convert( vpi.Format.S16 )
	corners, scores = inp.harriscorners( sensitivity=0.01 )
```

Which will run the commands to convert the image format and run the harris
corners algorithm on the native CUDA-enabled GPU. Alternatively, you can specify
which backend to use for a specific command by doing:

```python
out = inp.convert( vpi.Format.BGR8, backend=vpi.Backend.CUDA )
```

VPI provides easy to use code for a small sample of vision algorithms, such as
LP filters like the box and gaussian filters, the FFT, morphological operations
such as erode and dilate, and a general convolution algorithm. Some more
involved algorithms provided are lens distortion correction, color equalization
histograms, and background subtractors. Each algorithm has been written to run
on a subset of available backends--a list of algorithms and the backends they
run on can be found [here](docs.nvidia.com/vpi/algorithms/html)

Python simplifies a lot of the synchronization and parallelism that is inherent
in using multiple backends. If you have done some parallel processing in
something like MPI or are generally interested in running many different streams
at once, consider looking into the C++ API for greater control of algorithm
scheduling. Unfortunately, the Python API is practically non-existent and the
examples are written in VPI 2.0 while the jetsons currently have version 1, so
some hacking may be required to get everything to work as intended. The C++ API
can be found [here](docs.nvidia.com/vpi/usergroup0.html)
