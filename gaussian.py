from flask import Response, Flask
import numpy as np
import cv2
import jetson.utils
import sys
import vpi

camera = jetson.utils.videoSource( "csi://0", argv=sys.argv )

app = Flask( __name__ )

def get_image():
    image = camera.Capture()
    inp = vpi.asimage( np.uint8( jetson.utils.cudaToNumpy( image ) ) )
    with vpi.Backend.CUDA:
        output = inp.convert( vpi.Format.U8 )
        blurred = output.gaussian_filter( 5, 1.2, border=vpi.Border.ZERO )
    blurred = blurred.cpu()
    return blurred

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
    
