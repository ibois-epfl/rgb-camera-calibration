# rgb-camera-calibration

Repository for calibrate simple monocular rgb cameras with opencv library.

</b>

# A bit of theory behind camera calibration

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

