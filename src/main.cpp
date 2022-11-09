#include <iostream>
#include "Calibrator.h"

using namespace cv;
using namespace std;

const int ESC_KEY = 27;

int main()
{
    Calibrator calibrator;

    VideoCapture camera;
    camera.open(0);
    if (!camera.isOpened())
    {
        cerr << "ERROR: Could not open camera" << endl;
        return 1;
    }

    cv::Mat frame;

    //! [capture]
    int delay = 1500;
    clock_t prevTimestamp = 0;

    bool isCapturing = false;

    while(true){
        camera >> frame;
        bool blinkOutput = false;

        if (isCapturing && double(clock() - prevTimestamp) > delay * 1e-3 * CLOCKS_PER_SEC){
            Mat frameCopy;
            frame.copyTo(frameCopy);
            calibrator.AddImage(frameCopy);
            prevTimestamp = clock();
            blinkOutput = true;
        }

        if(blinkOutput){
            bitwise_not(frame, frame);
        }

        imshow("Image View", frame);
        char key = (char)waitKey(blinkOutput? 500 : 1);

        if( key  == ESC_KEY )
            break;

        if( camera.isOpened() && key == 'g' )
        {
            isCapturing = !isCapturing;
        }

        if (calibrator.imageList.size() >= 5) {
            break;
        }
        //! [await_input]
    }

    calibrator.RunCalibration();

    cout << "camera matrix: " << calibrator.cameraMatrix << endl;
    cout << "dist coeffs: " << calibrator.distCoeffs << endl;

    calibrator.Save("test_calib.yml");

    return 0;
}