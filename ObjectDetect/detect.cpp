

/**
 * @file detect.cpp
 * @brief Handles the obejct detection part of the MOT system.
 * @author wantSomeChips
 * @date 2025
 * 
 */

#include "detect.hpp"
#include <opencv2/opencv.hpp>
#include "funcs.hpp"

/**
 * @brief Get the bounding box of Detected object.
 * 
 * Encapsulation protects class data by using functions for access, 
 * preventing accidental changes.
 * 
 * @param void void.
 * 
 * @return the bounding box of the Detected object.
 * 
 */
Rect fdObject::resultRect(void) const{

    return _result;
}

/**
 * @brief Top-level abstract function for the object Detection. Handle the Detection logic.
 *
 * @param frame     A single frame image input.
 * 
 * @return Boolean value. Return `true` if the Tracking goes on properly. 
 * 
 */
bool objDetect::tick(const Mat& frame){

    /* Use frame.clone() or copyTo() to Deep Copy. Otherwise it would be Shallow Copy. */

    /* Handle Underflow. Do it explicitly. */
    Mat& pre_frame = _p_frms[((_clock - 1 + FRM_BUFFER_SIZE) % FRM_BUFFER_SIZE)];
    Mat& cur_frame = _p_frms[(_clock % FRM_BUFFER_SIZE)];
    cv::cvtColor(frame, cur_frame, cv::COLOR_BGR2GRAY);


    ++ _clock;
    /* Handle Overflow. Although compiler will handle it, do it explicitly. */
    _clock = (_clock % UINT_FAST32_MAX);

    /* Model background every frame. */
    if(false == _backgrnd_initialized){
        /* 2 Frames Difference. */
        /* Remove noise. */
        Mat cur_frm_blur, pre_frm_blur;
        cv::medianBlur(cur_frame,cur_frm_blur, 5);
        cv::medianBlur(pre_frame,pre_frm_blur, 5);

        Mat fd_diff;
        cv::absdiff(cur_frm_blur, pre_frm_blur, fd_diff);

        cv::threshold(fd_diff, _fd_resp, FD_THRESHOLD, 255, cv::THRESH_BINARY);
    

        /* Process response, get detected objects. */
        Rect image_rect(Point(0,0),Size(cur_frame.cols,cur_frame.rows));
    
        Mat bigger_kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(9,9));
        cv::morphologyEx(_fd_resp, _fd_resp,cv::MORPH_CLOSE, bigger_kernel);


        vector<Rect> obj_rects = getRects(_fd_resp);

        backgrndUpdate(cur_frame, obj_rects);
    
    }


    /* Only process once per period. */
    if((_clock % _period) > 0U){

        return false;
    }

    /* Background Frame Difference. */
    vector<Rect> obj_rects;

    if(_backgrnd_initialized){

        Mat backgrnd_diff, kernel;
        cv::absdiff(cur_frame, _backgrnd, backgrnd_diff);

        Mat final_resp;

        bool getResp = getBackgrndDiffResp(cur_frame, final_resp);

        obj_rects = getRects(final_resp);
        backgrndUpdate(cur_frame, obj_rects);

    }
    else{
        obj_rects = getRects(_fd_resp);
        
    }

    for(const Rect& obj_rect: obj_rects){

        _objs.push_back(fdObject(obj_rect));
    }

    /* Collect and return detected obejects. */
    if(_objs.size() != 0){
        /* Get _objs prepared for next detection. */
        _res = std::move(_objs);
        return true;
    }

    return false;
}

/**
 * @brief Get background frame difference response, and use a kernel to mitigate fragmentations.
 * 
 * Use 2 threshold, `low_threshold` and `high_threshold`, to generate the final response.
 * 
 * The final response is based on the low threshold response.
 * 
 * A pixel in low threshold response will be kept in the final response, only when it has a 
 * neighbor in high threshold response. Neighbor means it's in the range of the kernel.
 * 
 * Possible fragments within an object will be connected in this way. 
 * 
 * We are using a pretty "narrow" kernel, based on the observation that object fragmentation often 
 * occurs along the horizontal axis, while object adhesion mostly happens along the vertical axis.
 *
 * @param cur_frame     Current input frame.
 * @param final_resp    Background frame difference response. 
 *                      This is the result of this funciton.
 * 
 * @return Boolean value. Return `true` if the function goes on properly. 
 * 
 */
bool objDetect::getBackgrndDiffResp(const Mat& cur_frame, Mat& final_resp){
    /* Kernele height should be an odd number. */
    Mat backgrnd_diff;
    cv::absdiff(cur_frame, _backgrnd, backgrnd_diff);

    Mat high_thresh_resp, low_thresh_resp;

    const Size kernel(1,9);
    const int low_thresh = 15, high_thresh = 50, max_val = 255;

    cv::threshold(backgrnd_diff, low_thresh_resp, low_thresh, max_val, cv::THRESH_BINARY);
    cv::threshold(backgrnd_diff, high_thresh_resp, high_thresh, max_val, cv::THRESH_BINARY);

    high_thresh_resp.copyTo(final_resp);

    const int dx = (int)((kernel.width -1) / 2), dy = (int)((kernel.height -1) / 2);
    const int y_max = cur_frame.rows, x_max = cur_frame.cols;

    for(int y = 0; y < y_max; ++ y){
        for(int x = 0; x < x_max; ++ x){
            if((0 == high_thresh_resp.at<uchar>(y , x)) && (0 != low_thresh_resp.at<uchar>(y , x))){
                /* Boundary check. */
                const int x_from = MAX(x - dx, 0), x_to = MIN(x + dx, x_max);
                const int y_from = MAX(y - dy, 0), y_to = MIN(y + dy, y_max);

                bool finded = false;
                for(int ky = y_from; ky <= y_to; ++ ky){
                    if(finded) break;

                    for(int kx = x_from; kx <= x_to; ++ kx){
                        if(0 != high_thresh_resp.at<uchar>(ky , kx)){
                            final_resp.at<uchar>(y, x) = max_val;
                            finded = true;
                            break;
                        }
                    }
                }
                
            }
        }
    }

    // imshow("final", final_resp);
    // imshow("high", high_thresh_resp);
    // imshow("low", low_thresh_resp);

    return true;
}

/**
 * @brief Update the background model. 
 *
 * @param frame         A single frame image input. 
 * @param obj_rects     Bounding boxes of all objects detected or currently tracked. 
 *                      They will be masked out when updateing background model.
 * 
 * @return Boolean value. Return `true` if the update goes on properly. 
 * 
 */
bool objDetect::backgrndUpdate(const Mat& frame, const vector<Rect>& obj_rects){

    float alpha = _alpha;

    Rect image_rect(Point(0,0),Size(frame.cols,frame.rows));

    Mat mask(frame.size(), CV_8UC1, cv::Scalar(255));

    /* Expand target Rects when calculating background mask.*/
    float expand_ratio = 1.2f;

    for(const Rect& rec: obj_rects){

        Point center(rec.x + rec.width / 2.0f, rec.y + rec.height / 2.0f);
        
        Size new_size(rec.width * expand_ratio, rec.height * expand_ratio);
        Point new_tl (center.x - 0.5f * new_size.width, center.y - 0.5f * new_size.height);
        Rect expanded_rect(new_tl, new_size);
        expanded_rect = expanded_rect & image_rect;

        cv::rectangle(mask, expanded_rect, cv::Scalar(0), cv::FILLED);
    }

    for(const Rect& rec: _tracked_ROIs){

        Point center(rec.x + rec.width / 2.0f, rec.y + rec.height / 2.0f);
        
        Size new_size(rec.width * expand_ratio, rec.height * expand_ratio);
        Point new_tl (center.x - 0.5f * new_size.width, center.y - 0.5f * new_size.height);
        Rect expanded_rect(new_tl, new_size);
        expanded_rect = expanded_rect & image_rect;

        cv::rectangle(mask, expanded_rect, cv::Scalar(0), cv::FILLED);
    }
    
    /* Can be accelerated by CPU Branch Prediction. */
    if(_backgrnd_initialized == false){
        
        -- _backgrnd_init_counter;
        if(_backgrnd_init_counter == 0){

            _backgrnd_initialized = true;
        }

        alpha = _alpha_init;
    }

    Mat new_backgrnd = (1.0f - alpha) * _backgrnd + alpha * frame;
    new_backgrnd.copyTo(_backgrnd, mask);

    return true;

}

/**
 * @brief Process frames difference's response and return bounding boxes
 * of detected objects.
 *
 * @param resp      Response of frames difference.
 * 
 * @return Bounding boxes of detected objects.
 * 
 */
vector<Rect> objDetect::getRects(Mat resp) {

    vector<Rect> objects;
    vector<vector<cv::Point2i>> contours;
    
    cv:: findContours(resp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const vector<cv::Point2i>& contour : contours) {

        Rect bbox = cv::boundingRect(contour);
        
        if (bbox.height > MIN_BBOX_HEIGHT && bbox.width > MIN_BBOX_WIDTH) { 
            objects.push_back(bbox);
        }
    }

    return objects;
}

/**
 * @brief Get the Detection result.
 *
 * Encapsulation protects class data by using functions for access, 
 * preventing accidental changes.
 * 
 * @param void void.
 * 
 * @return Detection result. A set of detected objects.
 * 
 */
vector<fdObject> objDetect::getObjects(void) const{

    return _res;
}



