#!/usr/bin/env python

import json
import argparse
import sys
import cv2 as cv
import numpy as np
import time
import os

def main(camera_index : int,
         json_path : str,
         mapping : int) -> None:

    print("[INFO] Parsing Json calib file ..")
    with open(json_path, 'r') as calib_file:
        calib_dict = json.load(calib_file)
    calib_w = calib_dict["cameraRes"][0]
    calib_h = calib_dict["cameraRes"][1]
    calib_mtx = np.asarray(calib_dict["cameraMatrix"])
    calib_dist = np.asarray(calib_dict["distortion"])

    print("[INFO] Opening camera stream ..")
    cap = cv.VideoCapture(camera_index)
    if int(cap.get(cv.CAP_PROP_FRAME_WIDTH))!=calib_w and int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))!=calib_h:
        sys.exit("[ERROR] camera width and height not corresponding with current camera res")

    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(calib_mtx,
                                                        calib_dist,
                                                        (calib_w,calib_h),
                                                        1,
                                                        (calib_w,calib_h))
    x, y, w, h = roi

    print(cap.get(3), cap.get(4))

    DIR = "./out"
    FPS = 25
    record = False
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    filename = os.path.join(DIR, f'{timestamp}_record_undistorted.avi')
    out = cv.VideoWriter(filename, fourcc, FPS, (calib_w,  calib_h))

    while True:
        ret, img = cap.read()

        if ret==True and mapping==0:
            dst = cv.undistort(img,
                            calib_mtx,
                            calib_dist,
                            None,
                            newCameraMatrix)
            dst = dst[y:y+h, x:x+w]
        else:
            mapx, mapy = cv.initUndistortRectifyMap(calib_mtx,
                                                    calib_dist,
                                                    None,
                                                    newCameraMatrix,
                                                    (calib_w,calib_h),
                                                    5)
            dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        
        show_frame = dst.copy()
        if record:
            out.write(dst)
            cv.putText(show_frame, "recording", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv.putText(show_frame, "NOT recording", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow('caliResult2', show_frame)
        
        key = cv.waitKey(1)
        if key%256 == 32:
            record = not record
        elif key == ord("q"):
            print("[INFO] exiting...")
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get frame feed from camera by undistorting')
    parser.add_argument('--cameraIndex', '-I',
                        type=int,
                        nargs='?',
                        required=True,
                        help='an integer corresponding to the camera port index')
    parser.add_argument('--jsonPath', '-J',
                        type=str,
                        required=True,
                        help='the path to the Json undistortion param for the camera')
    parser.add_argument('--mapping', '-M',
                        type=int,
                        required=False,
                        help='Set 1 to undistort with mapping, 0 to undistort without')

    args = parser.parse_args()
    if args.cameraIndex is not None:
        _camera_index = args.cameraIndex
    if args.jsonPath is not None:
        _json_path = args.jsonPath
    if args.mapping is not None:
        _mapping = args.jsonPath
    else:
        _mapping = 0
        
    main(camera_index =_camera_index,
         json_path =_json_path,
         mapping = _mapping)