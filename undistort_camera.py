#!/usr/bin/env python

import json
import argparse
import sys
import cv2 as cv
import numpy as np

def main(camera_index : int,
         json_path : str) -> None:

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
    

    # i = 0
    while True:
        ret, img = cap.read()


        # undistort
        dst_1 = cv.undistort(img,
                           calib_mtx,
                           calib_dist,
                           None,
                           newCameraMatrix)
        # crop image
        dst_1 = dst_1[y:y+h, x:x+w]
        # crop the image
        cv.imshow('caliResult1', dst_1)



        mapx, mapy = cv.initUndistortRectifyMap(calib_mtx,
                                                calib_dist,
                                                None,
                                                newCameraMatrix,
                                                (calib_w,calib_h),
                                                5)
        dst_2 = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        cv.imshow('caliResult2', dst_2)





        # sho_frame = frame.copy()
        # # cv.putText(sho_frame, "Press 'Backspace' to capture", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # cv.putText(sho_frame, "Press 'q' to exit", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # cv.imshow("preview", sho_frame)
        
        key = cv.waitKey(1)
        # if key == 32:  # backspace
        #     cv.imwrite("images/img" + str(i) + ".png", frame)
        #     i += 1
        #     print(f"[INFO ]Image {i} saved")
        if key == ord("q"):
            print("[INFO] exiting...")
            break

    cap.release()
    cv.destroyAllWindows()



    # img = cv.imread('cali5.png')
    # h,  w = img.shape[:2]
    
    # newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix,
    #                                                     dist,
    #                                                     (calib_w,calib_h),
    #                                                     1,
    #                                                     (calib_w,calib_h))



    # # Undistort
    # dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

    # # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    # cv.imwrite('caliResult1.png', dst)



    # # Undistort with Remapping
    # mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
    # dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    # cv.imwrite('caliResult2.png', dst)




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

    args = parser.parse_args()
    if args.cameraIndex is not None:
        _camera_index = args.cameraIndex
    if args.jsonPath is not None:
        _json_path = args.jsonPath
        
    main(camera_index=_camera_index,
         json_path=_json_path)