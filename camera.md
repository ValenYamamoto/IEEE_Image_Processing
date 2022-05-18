## Cameras

We will be using MIPI CSI cameras during the workshop. In general, the jetson
nanos can take input streams from many different places and output to many
different places. 

### Getting Camera Input
Since we have a single camera port and are using CSI cameras, cameras should be
located at *csi://0*. Frames can be accessed via the following python code:

```python
import sys
import jetson.utils

camera = jetson.utils.videoSource( "csi://0", argv=sys.argv )
image = camera.Capture()

# do something with image
```

To capture and save images without having to run code, you can run the
nvgstcapture-1.0 command and press j-[Enter] to capture a single image, then
q-[Enter] to exit the program. Though if running headless, it may be hard to run

### Streaming Output
Output can either be sent out as a stream, displayed either on the display
attached the jetson or on a remote screen through UDP, or saved in a video file.
This can be specified by passing a URI when creating a videoOutput object.

The following options are available:

![camera_output](images/camera_output.PNG)

To run code while output is streaming, use the following code:

```python
import sys
import jetson.utils

display = jetson.utils.videoOutput( "display://0", argv=sys.argv )

while display.IsStreaming():
	# some code here

	display.Render( image )
        display.SetStatus( "{:s} | {:d}x{:d} | {:.1f} FPS".format( "Harris Camera Viewer", image.width, image.height, display.GetFrameRate() ) )
```

#### For the Workshop ...
For this workshop, we are running headless through micro USB, which makes it
hard to launch a second window to stream output. Therefore, we are using the
python framework Flask to create a local website to stream the video to,
allowing us to view it on our laptops

```python3
from flask import Response, Flask
import numpy as np
import cv2
import jetson.utils
import sys

camera = jetson.utils.videoSource( "csi://0", argv=sys.argv )

app = Flask( __name__ )

def get_image():
    image = camera.Capture()
    image = jetson.utils.cudaToNumpy( image ) 
    return image

def encode_video():
    while True:
        image = get_image()
        return_key, encoded = cv2.imencode( ".jpg", image )
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +  encoded.tobytes()  + b'\r\n\r\n' )

@app.route( "/" )
def streamFrames():
    return Response( encode_video(), mimetype="multipart/x-mixed-replace; boundary=frame" )

if __name__ == "__main__":
    app.run( "0.0.0.0", port="8000" )
```    
