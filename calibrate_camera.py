#!/usr/bin/env python


import numpy as np
import cv2 as cv
import glob
import json
import argparse

def main(chessboard_horizontal : int,
         chessboard_vertical : int,
         chessboard_size : int) -> None:

    chessboardSize = (chessboard_horizontal,chessboard_vertical)  # e.g. (19,13)

    # camera frame size
    with open("frame_resolution.txt", "r") as f:
        frame_width, frame_height = f.read().split()
    frameSize = (int(frame_width),int(frame_height))
    print(f"[INFO] Camera frame width: {frame_width}")
    print(f"[INFO] Camera frame height: {frame_height}")


    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    size_of_chessboard_squares_mm = chessboard_size  # e.g. 20mm
    objp = objp * size_of_chessboard_squares_mm

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


    images = glob.glob('./images/*.png')

    for image in images:

        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        if ret == True:

            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(300)

    cv.destroyAllWindows()


    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    # show cameras with translation vectors and rotation in a plot


    print("[INFO] ret: \n", ret)  # ?
    print("[INFO] cameraMatrix: \n", cameraMatrix)  # A = [fx 0 cx; 0 fy cy; 0 0 1] (linear distortion)
    print("[INFO] dist: \n", dist)  # distortion coefficients (additional non-linear distortion)
    print("[INFO] tvecs: \n", tvecs)  # translations for each camera pose
    print("[INFO] tvecs: \n", rvecs)  # rotations for each camera pose

    print("[INFO] Printing results json file ..")
    print("[INFO] Saving camera 3x3 matrix and vector of distortion coefficients")
    with open("camera_calibration.json", "w") as f:
        json.dump({"cameraMatrix": cameraMatrix.tolist(),
                "distortion": dist.tolist()}, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute calibration for camera')
    parser.add_argument('--chessBoardHorizontal', '-H',
                        type=int,
                        nargs='?',
                        required=True,
                        help='an integer corresponding to the internal corners of the chessboard horizontal side')
    parser.add_argument('--chessBoardVertical', '-V',
                        type=int,
                        nargs='?',
                        required=True,
                        help='an integer corresponding to the internal corners of the chessboard vertical side')
    parser.add_argument('--chessBoardSize', '-S',
                        type=int,
                        nargs='?',
                        required=True,
                        help='the size in mm of on chessboard square')

    args = parser.parse_args()
    if args.chessBoardHorizontal is not None:
        _chessboard_horizontal = args.chessBoardHorizontal
    if args.chessBoardVertical is not None:
        _chessboard_vertical = args.chessBoardVertical
    if args.chessBoardSize is not None:
        _chessboard_size = args.chessBoardSize
        
    main(chessboard_horizontal=_chessboard_horizontal,
         chessboard_vertical=_chessboard_vertical,
         chessboard_size=_chessboard_size)