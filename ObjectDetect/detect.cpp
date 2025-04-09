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

    float iou = func::IoU(_result,bbox);

    // cout << "DEBUG: fdObject::isSameObject - IOU: "<< iou <<endl;

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
        _result = Rect(tl_x, tl_y, (br_x - tl_x), (br_y - tl_y)) & _image_rect;

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

    int round = _clock % _period;

    cv::cvtColor(frame, _p_frms[round], cv::COLOR_BGR2GRAY);

    Mat& pre_frame = _p_frms[(_clock -1) % _period];
    Mat& cur_frame = _p_frms[round];

    /* Use frame.clone() or copyTo() to Deep Copy. Otherwise it would be Shallow Copy. */
    if(_clock < _clock_bound){
        ++ _clock;
    }
    else {
        _clock = 0;
    }

    /* Only process the first 2 frames per period. */
    if(round > 1){

        return false;
    }

    /* Run 3 Frames Difference and process results at the second frame. */
    bool run_three_frm_diff = (round == 1);
    bool get_result = (round == 1);

    /* 2 Frames Difference. */
    /* Remove noise. */
    Mat cur_frm_blur, pre_frm_blur;
    cv::medianBlur(cur_frame,cur_frm_blur, 5);
    cv::medianBlur(pre_frame,pre_frm_blur, 5);

    Mat& two_fd_diff = _fd_diff[round];
    cv::absdiff(cur_frm_blur, pre_frm_blur, two_fd_diff);

    Mat& two_fd_resp = _fd_resp[round];
    cv::threshold(two_fd_diff, two_fd_resp, FD_THRESHOLD, 255, cv::THRESH_BINARY);
    
    /* Kernel for morphology operations. */
    Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    Mat big_kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(9,9));

    /* 3 Frames Difference after collecting two 2FD. */
    if(run_three_frm_diff){
        Mat three_fd_diff;
        cv::bitwise_or(_fd_diff[0], _fd_diff[1], three_fd_diff);
        cv::threshold(three_fd_diff, _three_fd_resp,FD_THRESHOLD, 255, cv::THRESH_BINARY);
    }
    

    /* Background Frame Difference. */
    if(_backgrnd_initialized){

        Mat backgrnd_diff;
        cv::absdiff(cur_frame, _backgrnd_i, backgrnd_diff);
        
        Mat& backgrnd_resp = _backgrnd_resp[round];
        cv::threshold(backgrnd_diff, backgrnd_resp, BAKCGRND_THRESHOLD, 255, cv::THRESH_BINARY);

        Mat _kernel = cv::getStructuringElement(cv::MORPH_CROSS,cv::Size(3,3));
        cv::morphologyEx(backgrnd_resp, backgrnd_resp,cv::MORPH_OPEN, _kernel);

        _kernel = cv::getStructuringElement(cv::MORPH_CROSS,cv::Size(6,6));
        cv::morphologyEx(backgrnd_resp, backgrnd_resp,cv::MORPH_CLOSE, _kernel);

        // cv::morphologyEx(_three_fd_resp, _three_fd_resp,cv::MORPH_OPEN, kernel);
        cv::morphologyEx(_three_fd_resp, _three_fd_resp,cv::MORPH_DILATE, big_kernel);

    }

    cv::morphologyEx(two_fd_resp, two_fd_resp,cv::MORPH_CLOSE,big_kernel);
 
    /* Get objects base on 2FD. This is used to supplement the final result. */
    _fd_obj_rects[round] = getRects(two_fd_resp);

    /* Collect and return detected obejects. */
    if(get_result){

        Rect image_rect(Point(0,0),Size(cur_frame.cols,cur_frame.rows));

        Mat final_resp = getFinalResp();

        cv::morphologyEx(final_resp, final_resp,cv::MORPH_CLOSE, kernel);


        vector<Rect> final_obj_rects = getRects(final_resp);

        for(const Rect& obj_rect: final_obj_rects){

            _objs.push_back(fdObject(obj_rect, image_rect));
        }


        this -> backgrndUpdate(cur_frame);

        if(_objs.size() != 0){
            /* Get _objs prepared for next detection. */
            _res = std::move(_objs);
            return true;
        }
    }

    return false;
}


Mat objDetect::getFinalResp(){

    Mat final_resp;


    if(_backgrnd_initialized){

        cv::bitwise_and(_backgrnd_resp[1], _three_fd_resp, final_resp);

    }
    else{

        final_resp =  _fd_resp[1];

    }


    // imshow("3FD_diff", _three_fd_resp);
    // if(_backgrnd_initialized){
    //     imshow("backgrnd_resp_2", _backgrnd_resp[1]);
    // }
    // imshow("final_resp", final_resp);
    
    // cv::waitKey();

    return final_resp;

}

bool objDetect::backgrndUpdate(const Mat& frame){

    float alpha = _alpha;

    Mat mask = Mat(frame.size(), CV_8UC1, cv::Scalar(255));

    Rect image_rect(Point(0,0),Size(frame.cols,frame.rows));

    /* Expand target Rects when calculating background mask.*/
    float expand_ratio = 1.2;

    for(fdObject& obj:_objs){

        Rect rec = obj.resultRect();
        Point center(rec.x + rec.width / 2.0, rec.y + rec.height / 2.0);

        Size new_size(rec.width * expand_ratio, rec.height * expand_ratio);
        Point new_tl (center.x - 0.5 * new_size.width, center.y - 0.5 * new_size.height);
        Rect expanded_rect(new_tl, new_size);
        expanded_rect = expanded_rect & image_rect;

        cv::rectangle(mask, expanded_rect, cv::Scalar(0), cv::FILLED);
    }

    for(const Rect& rec: _tracked_ROIs){

        Point center(rec.x + rec.width / 2.0, rec.y + rec.height / 2.0);
        
        Size new_size(rec.width * expand_ratio, rec.height * expand_ratio);
        Point new_tl (center.x - 0.5 * new_size.width, center.y - 0.5 * new_size.height);
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

    Mat frame_f;
    frame.convertTo(frame_f, CV_32FC1);

    Mat new_backgrnd = (1.0 - alpha) * _backgrnd + alpha * frame_f;
    new_backgrnd.copyTo(_backgrnd, mask);

    _backgrnd.convertTo(_backgrnd_i, CV_8UC1);
    // imshow("Bakcgrnd", _backgrnd_i);

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

Mat objDetect::FramesDiff(Mat cur_fra, Mat pre_fra, Mat pp_fra , bool three_frame_diff){

    Mat cur_b, pre_b;
    // cv::cvtColor(cur_fra, cur_g, cv::COLOR_BGR2GRAY);
    // cv::cvtColor(pre_fra, pre_g, cv::COLOR_BGR2GRAY);

    cv::medianBlur(cur_fra,cur_b, 5);
    cv::medianBlur(pre_fra,pre_b, 5);

    Mat cur_pre_d;
    cv::absdiff(cur_b, pre_b, cur_pre_d);

    Mat resp,res;

    /* 2 Frames Difference. */
    if(three_frame_diff == false){

        cv::threshold(cur_pre_d, resp,FD_THRESHOLD, 255, cv::THRESH_BINARY);
    }
    /* 3 Frames Difference. */
    else{

        Mat pre_pp_d;
        cv::absdiff(pre_fra, pp_fra, pre_pp_d);

        Mat dd;
        cv::bitwise_or(cur_pre_d, pre_pp_d, dd);

        cv::threshold(dd, resp,FD_THRESHOLD, 255, cv::THRESH_BINARY);

    }
 
    /* Morphology operations. */
    Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));

    if(_backgrnd_initialized){
        /* Remove noise. */
        kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
        cv::morphologyEx(resp, resp,cv::MORPH_OPEN,kernel);

        kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(9,9));
        cv::morphologyEx(resp, resp,cv::MORPH_DILATE,kernel);
    }
    else{
        kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(9,9));
        cv::morphologyEx(resp, resp,cv::MORPH_CLOSE,kernel);

        /* Remove noise. */
        // kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
        // cv::morphologyEx(resp, resp,cv::MORPH_OPEN,kernel);
    }

    if(_backgrnd_initialized){

        Mat backgrnd_diff;
        cv::absdiff(cur_fra, _backgrnd_i, backgrnd_diff);
        cv::threshold(backgrnd_diff, backgrnd_diff,BAKCGRND_THRESHOLD, 255, cv::THRESH_BINARY);

        cv::bitwise_and(resp, backgrnd_diff, res);

        imshow("Backgrnd Diff",backgrnd_diff);

    }
    else{

        res = resp;
    }
    
    imshow("Res",res);

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
    return _backgrnd_resp[1];
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


