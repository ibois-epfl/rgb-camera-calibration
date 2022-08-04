#!/usr/bin/env python

import cv2 as cv
import numpy as np
import time
import os
import argparse

def main(camera_index : int) -> None:
    
    if not os.path.exists("images"):
        os.makedirs("images")
    else:
        print("[INFO] erasing images in the folder?")
        if ((input("[Y/N] ").capitalize()) == "Y"):
            for file in os.listdir("images"):
                os.remove("images/" + file)
        else:
            print("[INFO] exiting...")
            return
    
    cv.namedWindow("camera capture preview")
    cap = cv.VideoCapture(camera_index)

    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    with open("frame_resolution.txt", "w") as f:
        f.write(str(frame_width) + " " + str(frame_height))

    i = 0
    while True:

        ret, frame = cap.read()

        sho_frame = frame.copy()
        cv.putText(sho_frame, "Press 'space' to capture", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.putText(sho_frame, "Press 'q' to exit", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.imshow("preview", sho_frame)
        
        key = cv.waitKey(1)
        if key == 32:  # space
            cv.imwrite("images/img" + str(i) + ".png", frame)
            i += 1
            print(f"[INFO] Image {i} saved")
        elif key == ord("q"):
            print("[INFO] exiting...")
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capturing images from camera')
    parser.add_argument('--cameraIndex', '-I',
                        type=int,
                        nargs='?',
                        required=True,
                        help='an integer corresponding to the camera port index')

    args = parser.parse_args()
    if args.cameraIndex is not None:
        _camera_index = args.cameraIndex
        
    main(camera_index=_camera_index)