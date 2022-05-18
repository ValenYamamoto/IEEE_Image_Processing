# IEEE Jetson Nano Workshop Pt. 2 - Image Processing and Computer Vision

## Overview
Courtesy of Nvidia, we have 15 Jetsons. We might as well do some image
processing on them.

The Jetsons boast a CUDA-enabled GPU to accelerate computation heavy image
processing workloads. Nvidia provides the VPI library for running specific
prewritten CV algorithms on the backends provided on the jetson. For more
general operations, GPU use is easily facillitated by a library like PyTorch or
Tensorflow, which will emit CUDA instructions for you.

!INCLUDE "image_processing.md"

!INCLUDE "VPI.md"

!INCLUDE "GPU.md"

!INCLUDE "camera.md"

!INCLUDE "resources.md"
