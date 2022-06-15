#!/usr/bin/env python


import numpy as np
import cv2 as cv
import glob
import pickle


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

# number of internal corners in the chessboard
chessboardSize = (19,13)

# camera frame size
with open("frame_resolution.txt", "r") as f:
    frame_width, frame_height = f.read().split()
frameSize = (int(frame_width),int(frame_height))
print(f"[INFO] Camera frame width: {frame_width}")
print(f"[INFO] Camera frame height: {frame_height}")

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob('./images/*.png')

for image in images:
    print(image)

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    print(ret)
    print(corners)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(300)

cv.destroyAllWindows()

# Run calibration
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

print("[INFO] ret: \n", ret)
print("[INFO] cameraMatrix: \n", cameraMatrix)
print("[INFO] dist: \n", dist)
print("[INFO] tvecs: \n", tvecs)

cam_calib = {"cam_matrix": cameraMatrix,
             "dist_coeffs": dist}
with open("cam_calib.p", "wb") as f:
    pickle.dump(cam_calib, f)