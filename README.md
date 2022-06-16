
<h1 align="center">rgb-camera-calibration</h1>

<p align="center">
    <img src="./calibcalib.gif" width="600">
</p>




Repository for calibrate simple monocular rgb cameras with opencv library.

</b>

## How it works

**[1]** Capture images with:
```bash
./capture_img.py -I 2
```

**[2]** For running the calibration use the tag `-H` for the horizontal intern corners of the chessboard's squares, `-V` the same for the vertical and `-S` for the size in mm of a single square:
```bash
./calibrate_camera.py -H 19 -V 13 -S 20
```

## A bit of theory behind camera calibration

A camera has two types of parameters which describes itss model: `intrinsics` and `extrinsics`. These parameters describes how the camera works.

The `extrinsics` describes where the camera is in the 3D space, and its describe by a **6D vector** with translation and rotation. If you want to compute *camera localization* you want to compure these parameters:

```
    [X]
    [Y]
v = [Z]
    [α]
    [β]
    [γ]
``` 

On the other hand, the `intrinsics` describes how the camera maps a 3D world point on to the 2D image plane using a *pinhole camera model*. These internal parameters are composed by 4 or 5 elements. If you want to remove *distortion* (e.g. sscaling, skewing, barrel, pincushion or tangential distortion) from your images, this what you need to compute.

```
    [  f   fβ   Δx   
A =    0   αf   Δy
       0    0    1  ]    
```

