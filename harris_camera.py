import numpy as np
import jetson.utils
import sys
display = jetson.utils.videoOutput( "display://0", argv=sys.argv )
import cv2
import vpi

#display = jetson.utils.glDisplay()

#camera = jetson.utils.gstCamera( 1920, 1280, '0' )
#camera.Open()

def harris( image ):
    with vpi.Backend.CUDA:
        inp = image.convert( vpi.Format.S16 )
        corners, scores = inp.harriscorners( sensitivity=0.01 )

    out = inp.convert( vpi.Format.BGR8, backend=vpi.Backend.CUDA )

    if corners.size > 0:
        with out.lock(), scores.lock(), corners.lock():
            out_data = out.cpu()
            scores_data = scores.cpu()
            corners_data = corners.cpu()
            cmap = cv2.applyColorMap( np.arange(0, 256, dtype=np.uint8 ), cv2.COLORMAP_HOT )

            maxscore = scores_data.max()
            for i in range( corners.size ):
                color = tuple( [int(x) for x in cmap[255*scores_data[i]//maxscore, 0]] )
                kpt = tuple( corners_data[i].astype( np.int16 ) )
                cv2.circle( out_data, kpt, 3, color, -1 )

    out = out.convert( vpi.Format.RGB8, backend=vpi.Backend.CUDA  )
    return out

camera = jetson.utils.videoSource( "csi://0", argv=sys.argv )
#display = jetson.utils.videoOutput( "display://0", argv=sys.argv )

while display.IsStreaming():
    image = camera.Capture()
    inp = vpi.asimage( np.uint8( jetson.utils.cudaToNumpy( image ) ), vpi.Format.BGR8 )
    output = harris( inp )
    vpi.clear_cache()

    display.Render( jetson.utils.cudaFromNumpy( output.cpu() ) )
    display.SetStatus( "{:s} | {:d}x{:d} | {:.1f} FPS".format( "Harris Camera Viewer", image.width, image.height, display.GetFrameRate() ) )
    #cv2.imwrite( 'harris_corners.png', output.cpu() )
    #break
