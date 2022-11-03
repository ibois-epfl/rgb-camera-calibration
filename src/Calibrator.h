# pragma once
#include <iostream>

#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

class Calibrator {
public:
    Calibrator();
    ~Calibrator();

    void ValidateAndUpdateFlag();
    void AddImage(const cv::Mat& image);
    void SetGridWidth(float width);
    void Calibrate(cv::Mat *imgForDisplay = nullptr);

    static enum Pattern { NOT_EXISTING, CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

private:
    void DetectPattern(cv::Mat *imgForDisplay = nullptr);


public:
    cv::Size boardSize = cv::Size(19, 13);  // The size of the board -> Number of items by width and height
    Pattern calibrationPattern = Pattern::CHESSBOARD;   // One of the Chessboard, circles, or asymmetric circle pattern
    float squareSize = 20.0f;                           // The size of a square in your defined unit (point, millimeter,etc).
    float aspectRatio = 0.0f;                           // The aspect ratio, can be 0 if no need
    bool writeExtrinsics = true;                        // Write extrinsic parameters
    bool calibZeroTangentDist = true;                   // Assume zero tangential distortion
    bool calibFixPrincipalPoint = true;                 // Fix the principal point at the center
    bool flipVertical = false;                          // Flip the captured images around the horizontal axis
    std::string outputFileName = "camera";              // The name of the file where to write
    bool useFisheye = false;                            // use fisheye camera model for calibration
    bool fixDistortion = false;                         // fix K1 distortion coefficient

    int winSize = 11;                                   // The size of the window capturing
    float gridWidth = squareSize * (boardSize.width - 1);    // The width of the board -> number of items in width * squareSize
    bool isGridWidthSet = false;                        // Flag to check if the grid width is set

    std::vector<cv::Mat> imageList;
    bool goodInput;
    int flag;

private:
    std::vector<std::vector<cv::Point2f> > imagePoints;
};

