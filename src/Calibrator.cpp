//
// Created by ibois on 03/11/22.
//

#include "Calibrator.h"

void Calibrator::ValidateAndUpdateFlag()
{
    goodInput = true;
    if (boardSize.width <= 0 || boardSize.height <= 0)
    {
        std::cerr << "Invalid Board size: " << boardSize.width << " " << boardSize.height << std::endl;
        goodInput = false;
    }
    if (squareSize <= 10e-6)
    {
        std::cerr << "Invalid square size " << squareSize << std::endl;
        goodInput = false;
    }

    flag = 0;

    if(calibFixPrincipalPoint) flag |= cv::CALIB_FIX_PRINCIPAL_POINT;
    if(calibZeroTangentDist)   flag |= cv::CALIB_ZERO_TANGENT_DIST;
    if(aspectRatio)            flag |= cv::CALIB_FIX_ASPECT_RATIO;
    if(fixDistortion)          flag |= cv::CALIB_FIX_K1 | cv::CALIB_FIX_K2 | cv::CALIB_FIX_K3 | cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5;

    if (useFisheye) {
        // the fisheye model has its own enum, so overwrite the flags
        flag = cv::fisheye::CALIB_FIX_SKEW | cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
        if(fixDistortion)                   flag |= cv::fisheye::CALIB_FIX_K1;
        if(fixDistortion)                   flag |= cv::fisheye::CALIB_FIX_K2;
        if(fixDistortion)                   flag |= cv::fisheye::CALIB_FIX_K3;
        if(fixDistortion)                   flag |= cv::fisheye::CALIB_FIX_K4;
        if (calibFixPrincipalPoint) flag |= cv::fisheye::CALIB_FIX_PRINCIPAL_POINT;
    }

}

void Calibrator::AddImage(const cv::Mat &image) {
    imageList.push_back(cv::Mat());
    image.copyTo(imageList.back());
}

void Calibrator::SetGridWidth(float width) {
    gridWidth = width;
    isGridWidthSet = true;
}

void Calibrator::Calibrate() {
    //! [fixed_aspect]
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if( !s.useFisheye && s.flag & CALIB_FIX_ASPECT_RATIO )
        cameraMatrix.at<double>(0,0) = s.aspectRatio;
    //! [fixed_aspect]
    if (s.useFisheye) {
        distCoeffs = Mat::zeros(4, 1, CV_64F);
    } else {
        distCoeffs = Mat::zeros(8, 1, CV_64F);
    }

    vector<vector<Point3f> > objectPoints(1);
    calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);
    objectPoints[0][s.boardSize.width - 1].x = objectPoints[0][0].x + grid_width;
    newObjPoints = objectPoints[0];

    objectPoints.resize(imagePoints.size(),objectPoints[0]);

    //Find intrinsic and extrinsic camera parameters
    double rms;

    if (s.useFisheye) {
        Mat _rvecs, _tvecs;
        rms = fisheye::calibrate(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, _rvecs,
                                 _tvecs, s.flag);

        rvecs.reserve(_rvecs.rows);
        tvecs.reserve(_tvecs.rows);
        for(int i = 0; i < int(objectPoints.size()); i++){
            rvecs.push_back(_rvecs.row(i));
            tvecs.push_back(_tvecs.row(i));
        }
    } else {
        int iFixedPoint = -1;
        if (release_object)
            iFixedPoint = s.boardSize.width - 1;
        rms = calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint,
                                cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
                                s.flag | CALIB_USE_LU);

        cout << "New object points: " << newObjPoints << endl;
        cout << "camera matrix: " << cameraMatrix << endl;
        cout << "dist coeffs: " << distCoeffs << endl;
    }

    if (release_object) {
        cout << "New board corners: " << endl;
        cout << newObjPoints[0] << endl;
        cout << newObjPoints[s.boardSize.width - 1] << endl;
        cout << newObjPoints[s.boardSize.width * (s.boardSize.height - 1)] << endl;
        cout << newObjPoints.back() << endl;
    }

    cout << "Re-projection error reported by calibrateCamera: "<< rms << endl;

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    objectPoints.clear();
    objectPoints.resize(imagePoints.size(), newObjPoints);
    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix,
                                            distCoeffs, reprojErrs, s.useFisheye);


}

void Calibrator::DetectPattern(cv::Mat *imgForDisplay) {
    //------------------------- Camera Calibration ------------------------
    int chessBoardFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;

    if(!useFisheye) {
        // fast check erroneously fails with high distortions like fisheye
        chessBoardFlags |= cv::CALIB_CB_FAST_CHECK;
    }

    for(const auto& img: imageList){
        auto imageSize = img.size();  // Format input image.
        if( flipVertical )    flip( img, img, 0 );

        //! [find_pattern]
        std::vector<cv::Point2f> pointBuf;

        bool found;

        switch( calibrationPattern ) // Find feature points on the input format
        {
            case Pattern::CHESSBOARD:
                found = findChessboardCorners( img, boardSize, pointBuf, chessBoardFlags);
                break;
            case Pattern::CIRCLES_GRID:
                found = findCirclesGrid( img, boardSize, pointBuf );
                break;
            case Pattern::ASYMMETRIC_CIRCLES_GRID:
                found = findCirclesGrid( img, boardSize, pointBuf, cv::CALIB_CB_ASYMMETRIC_GRID );
                break;
            default:
                found = false;
                break;
        }

        //! [find_pattern]
        //! [pattern_found]
        if (found)                // If done with success,
        {
            // improve the found corners' coordinate accuracy for chessboard
            if( calibrationPattern == Pattern::CHESSBOARD)
            {
                cv::Mat viewGray;
                cvtColor(img, viewGray, cv::COLOR_BGR2GRAY);
                cornerSubPix( viewGray, pointBuf, cv::Size(winSize,winSize),
                              cv::Size(-1,-1), cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.0001 ));
            }

            imagePoints.push_back(pointBuf);

            // Draw the corners.
            drawChessboardCorners( img, boardSize, cv::Mat(pointBuf), found );
        }

        img.copyTo(*imgForDisplay);
    }

    if (imagePoints.size() < 2)
    {
        // cout << "No enough data to perform calibration." << endl;
    }

    // cout << "Calibrating..." << endl;
    if(runCalibrationAndSave(s, imageSize,  cameraMatrix, distCoeffs, imagePoints, grid_width, release_object)){
        // cout << "Calibration succeeded." << endl;
        mode = CALIBRATED;
    } else {
        // cout << "Calibration failed." << endl;
        mode = DETECTION;
    }
}
