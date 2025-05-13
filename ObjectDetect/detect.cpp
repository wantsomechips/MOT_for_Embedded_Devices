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

#include "detect.hpp"
#include <opencv2/opencv.hpp>
#include "funcs.hpp"

Rect fdObject::resultRect(void) const{

    return _result;
}

bool fdObject::isSameObject(const Rect& bbox) const{

    float iou = func::IoU(_result, bbox);

    return (iou > _min_iou_req);
}

bool fdObject::getResult(){

    if(_rects.size() == 2){

        Rect pre = _rects[0];
        Rect cur = _rects[1];
        Rect final_rect = _result;

        /* Use centroid to determine direction. */
        Point pre_center(pre.x + pre.width / 2.0f, pre.y + pre.height / 2.0f);
        Point cur_center(cur.x + cur.width / 2.0f, cur.y + cur.height / 2.0f);
        Point dir =  cur_center - pre_center;

        /* Normalization and expand. */
        float expand_ratio = 0.1f;
        float norm = std::sqrt(dir.x * dir.x + dir.y * dir.y);

        float dx = dir.x / norm * expand_ratio * cur.width;
        float dy = dir.y / norm * expand_ratio * cur.height;

        /* Top left points. */
        float tl_x = final_rect.x;
        float tl_y = final_rect.y;

        /* Bottom right points. */
        float br_x = final_rect.x + final_rect.width;
        float br_y = final_rect.y + final_rect.height;

        if(dx > 0){
            br_x += dx;
        }
        else{
            tl_x += dx;
        }

        if(dy > 0){
            br_y += dy;
        }
        else{
            tl_y += dy;
        }

        /* Boundry check. Could be empty! */
        // _result = Rect(tl_x, tl_y, (br_x - tl_x), (br_y - tl_y)) & _image_rect;

        if(_result.empty()){
           
            _result = final_rect;
            return false;
        }

        /* Return `true` if the expansion succeeded.*/
        return true;
    }

    return false;
}



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
    if(_backgrnd_initialized == false){
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
    if(_backgrnd_initialized){

        Mat backgrnd_diff, kernel;
        cv::absdiff(cur_frame, _backgrnd, backgrnd_diff);

        Mat final_resp;

        bool getResp = getBackgrndDiffResp(cur_frame, final_resp);

        vector<Rect> obj_rects = getRects(final_resp);
        backgrndUpdate(cur_frame, obj_rects);

        for(const Rect& obj_rect: obj_rects){

            _objs.push_back(fdObject(obj_rect));
        }


    }
    else{
        vector<Rect> obj_rects = getRects(_fd_resp);
        
        for(const Rect& obj_rect: obj_rects){

            _objs.push_back(fdObject(obj_rect));
        }
    }

    /* Collect and return detected obejects. */
    if(_objs.size() != 0){
        /* Get _objs prepared for next detection. */
        _res = std::move(_objs);
        return true;
    }



    return false;
}

bool objDetect::getBackgrndDiffResp(const Mat& cur_frame, Mat& final_resp){
    /* Kernele height should be an odd number. */
    Mat backgrnd_diff;
    cv::absdiff(cur_frame, _backgrnd, backgrnd_diff);

    Mat high_thresh_resp, low_thresh_resp;

    const Size kernel(1,20);
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



Mat objDetect::getFinalResp(){

    Mat final_resp;


    if(_backgrnd_initialized){

        final_resp = _backgrnd_resp;
    }
    else{
        final_resp = _fd_resp;
    }


    return final_resp;

}

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



/* --- --- --- --- --- --- --- --- ---

FUNC NAME: FramesDiff

# Description
Calculate the response of 2 or 3 frames difference.

# Arguments
@ cur_fra:  Current frame, frame `t`.
@ pre_fra:  Previous frame, frame `t-1`.
@ pp_fra:   Frame `t-2`. It's empty for 2 frames difference.

# Returns
@ res:      Response of frames differences.

--- --- --- --- --- --- --- --- --- */

Mat objDetect::FramesDiff(const Mat& pre_fra, const Mat& cur_fra){

    Mat cur_b, pre_b, resp;

    cv::medianBlur(cur_fra,cur_b, 5);
    cv::medianBlur(pre_fra,pre_b, 5);

    cv::absdiff(cur_b, pre_b, resp);
    cv::threshold(resp, resp,FD_THRESHOLD, 255, cv::THRESH_BINARY);

    return resp;
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

vector<Rect> objDetect::getRects(Mat resp) {

    vector<Rect> objects;
    vector<vector<cv::Point2i>> contours;

    // Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    // cv::morphologyEx(resp, resp,cv::MORPH_CLOSE,kernel);

    // cv::imshow("CLOSE Resp", resp);
    
    cv:: findContours(resp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const vector<cv::Point2i>& contour : contours) {

        Rect bbox = cv::boundingRect(contour);
        
        if (bbox.height > MIN_BBOX_HEIGHT && bbox.width > MIN_BBOX_WIDTH) { 
            objects.push_back(bbox);
        }
    }

    return objects;
}

Mat objDetect::getBackgrndResp(void) const{
    return _backgrnd_resp;
}

bool fdObject::addRect(const Rect& bbox){

    _rects.push_back(bbox);
    return true;
}

vector<fdObject> objDetect::getObjects(void) const{

    return _res;
}

bool objDetect::addTrackedObjs(const vector<Rect>& rois){
    
    _tracked_ROIs = rois;

    return true;
}


