/* --- --- --- --- --- --- --- --- ---

THREE FRAMES DIFFERENCE MOTION DETECTION

PROS:
- Objects are more likely to move consistently from frame to frame while
  noises are usually only appear between two frames. With `bitwise_or` of 
  two frame differences, objects' responses are enhanced while noises' 
  responses remain.

CONS:
- The background must remains static. It only works for static cameral.

- Easily disturbed by noise(like tree leaves moving) and background 
  changes(like sudden brightness changes).


--- --- --- --- --- --- --- --- --- */

#include "fd.hpp"
#include <opencv2/opencv.hpp>



/* --- --- --- --- --- --- --- --- ---

FUNC NAME: threeFramesDiff

# Description
Calculate the response of three frames differences.

# Arguments
@ cur_fra:  Current frame, frame `t`.
@ pre_fra:  Previous frame, frame `t-1`.
@ pp_fra:   Frame `t-2`.

# Returns
@ res:      Response of three frames differences.

--- --- --- --- --- --- --- --- --- */

Mat fd::threeFramesDiff(Mat cur_fra, Mat pre_fra, Mat pp_fra){

    Mat cur_g, pre_g, pp_g;
    cv::cvtColor(cur_fra, cur_g, cv::COLOR_BGR2GRAY);
    cv::cvtColor(pre_fra, pre_g, cv::COLOR_BGR2GRAY);
    cv::cvtColor(pp_fra, pp_g, cv::COLOR_BGR2GRAY);

    Mat cur_pre_d, pre_pp_d;
    cv::absdiff(cur_g, pre_g, cur_pre_d);
    cv::absdiff(pre_g, pp_g, pre_pp_d);

    Mat dd,res;
    cv::bitwise_or(cur_pre_d, pre_pp_d, dd);

    cv::threshold(dd, res,FD_THRESHOLD, 255, cv::THRESH_BINARY);

    return res;
}


/* --- --- --- --- --- --- --- --- ---

FUNC NAME: getRects

# Description
Process frames difference's response and return Rects of detected objects.

# Arguments
@ resp:     Response of frames difference.

# Returns
@ objects:  Vector of Rects(bounding boxes) of detected objects.

--- --- --- --- --- --- --- --- --- */

vector<Rect> fd::getRects(Mat resp) {

    vector<Rect> objects;
    vector<vector<cv::Point2i>> contours;
    
    cv:: findContours(resp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const vector<cv::Point2i>& contour : contours) {

        Rect bbox = cv::boundingRect(contour);
        
        if (bbox.area() > MIN_BBOX_SIZE) { 
            objects.push_back(bbox);
        }
    }

    return objects;
}
