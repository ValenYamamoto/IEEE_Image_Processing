## Using the GPU
VPI only offers a limited amount of quick, GPU supported algorithms. What if you
want to do something that VPI does not offer? You can decide to write your own
CUDA code, which is terrible, or you could decide to build the GPU enabled
version of openCV for the jetson, which is annoying, or you can just manually
write all the code for your algorithms in a framework like pyTorch or
Tensorflow, which will convert your code into CUDA for you.

### GPUs in PyTorch
For all PyTorch tensors/layers/model, you can specify on which device to place
it on. In the case of using a CUDA-enabled GPU, you can set the device to "cuda"
in order to place a tensor/layer/model on the GPU.

```python
import torch

cuda = torch.device( 'cuda' )

x = torch.ones( [5, 2], device=cuda )
```

The above code creates the device *cuda* to hold the GPU we are going to be
using and then *cuda* can be passed in wherever there is a device parameter to
tell torch to use the GPU.

Additionally, you can make torch use the GPU by using a context manager.

```python
with torch.cuda.device(0):
	# code here
```

This context manager tells torch to use the 0th, or first, GPU available. The
jetson nano only has a single GPU, so always pass 0 here.

PyTorch provides functions for adding, subtracting, multiplying, convoluting,
and many other useful mathematical functions with GPU support. Torch tensors can
be transfered to VPI arrays and images and vice versa, so the two libraries can
be run in collaboration. Unlike VPI (or openCV), torch does not have pre-written
computer vision code--you will have to write these yourself. But torch will take
care of the GPU and CUDA for you
