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

THIS FILE USES TWO FRAMES DIFFERENCE.


--- --- --- --- --- --- --- --- --- */

#include "detect.hpp"
#include <opencv2/opencv.hpp>
#include "funcs.hpp"

Rect fdObject::resultRect(void){

    return _result;
}

bool fdObject::isSameObject(const Rect& bbox){

    Rect newest = _rects.back();
    float iou = func::IoU(newest,bbox);

    /* If one bbox is included in the other. */
    if((1.0 - iou) < 1e-3) {

        return false;
    }
    // cout << "DEBUG: fdObject::isSameObject - IOU: "<< iou <<endl;

    return (iou > _min_iou_req);
}

bool fdObject::getResult(void){

    /* Detection failed. */
    if (_rects.size() < _min_frm_req){
        return false;
    }


    if(_rects.size() <= 2){

        Rect pre = _rects[0];
        Rect cur = _rects[1];

        /* Use centroid to determine direction. */
        Point pre_center(pre.x + pre.width / 2.0, pre.y + pre.height / 2.0);
        Point cur_center(cur.x + cur.width / 2.0, cur.y + cur.height / 2.0);
        Point dir =  cur_center - pre_center;

        /* Normalization and expand. */
        float expand_ratio = 0.1;
        float norm = std::sqrt(dir.x * dir.x + dir.y * dir.y);

        float dx = dir.x / norm * expand_ratio * cur.width;
        float dy = dir.y / norm * expand_ratio * cur.height;

        /* Top left points. */
        float tl_x = cur.x;
        float tl_y = cur.y;

        /* Bottom right points. */
        float br_x = cur.x + cur.width;
        float br_y = cur.y + cur.height;

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
           
            _result = cur;
        }
    }
    else{
        /* Simply merge rects. */
        Rect merged = _rects[0];
        for (int i = 1; i < _rects.size(); ++i) {
            merged |= _rects[i];
        }

        _result = merged;
    }


    return true;

}



bool objDetect::tick(const Mat& frame){

    int round = _clock % _period;

    cv::cvtColor(frame, _p_frms[round], cv::COLOR_BGR2GRAY);

    Mat& pre_frame = _p_frms[(_clock -1) % _period];
    Mat& cur_frame = _p_frms[round];


    // cout << "DEBUG: objDetect::tick - round: "<< round <<endl;

    if(_clock < _clock_bound){
        ++ _clock;
    }
    else {
        _clock = 0;
    }


    /* It's turn to calculate 3 Frames Difference. */
    bool in_three_frm_turn = ((round % 2) == 1);

    /* Use frame.clone() to Deep Copy. Otherwise it would be Shallow Copy. */

    /* 2 Frames Difference. */
    /* Remove noise. */
    Mat cur_frm_blur, pre_frm_blur;
    cv::medianBlur(cur_frame,cur_frm_blur, 5);
    cv::medianBlur(pre_frame,pre_frm_blur, 5);

    Mat two_fd_diff;
    cv::absdiff(cur_frm_blur, pre_frm_blur, two_fd_diff);

    /* Use frame.clone() or copyTo() to Deep Copy. Otherwise it would be Shallow Copy. */
    two_fd_diff.copyTo(_last_fd[round % 2]);

    Mat two_fd_resp;
    cv::threshold(two_fd_diff, two_fd_resp, FD_THRESHOLD, 255, cv::THRESH_BINARY);


    /* 3 Frames Difference after collecting two 2FD. */
    Mat three_fd_diff, three_fd_resp;
    if(in_three_frm_turn){

        cv::bitwise_or(_last_fd[0], _last_fd[1], three_fd_diff);
        cv::threshold(three_fd_diff, three_fd_resp,FD_THRESHOLD, 255, cv::THRESH_BINARY);
    }
 
    /* Morphology operations. */
    Mat kernel;
    if(_backgrnd_initialized){
        /* Remove noise. */
        // kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
        // cv::morphologyEx(two_fd_resp, two_fd_resp,cv::MORPH_OPEN,kernel);

        kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(9,9));
        cv::morphologyEx(two_fd_resp, two_fd_resp,cv::MORPH_DILATE,kernel);

        if(round % 2 ==  1){
            kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
            cv::morphologyEx(three_fd_resp, three_fd_resp,cv::MORPH_OPEN,kernel);

            kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
            cv::morphologyEx(three_fd_resp, three_fd_resp,cv::MORPH_CLOSE,kernel);
        }
    }
    else{
        kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(9,9));
        cv::morphologyEx(two_fd_resp, two_fd_resp,cv::MORPH_CLOSE,kernel);

        /* Remove noise. */
        // kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
        // cv::morphologyEx(resp, resp,cv::MORPH_OPEN,kernel);
    }

    imshow("2FD_Resp", two_fd_resp);
    if(in_three_frm_turn){
        imshow("3FD_Resp", three_fd_resp);
    }

    /* Background Frame Difference.*/
    Mat final_resp;
    Mat backgrnd_diff, backgrnd_resp;
    if(_backgrnd_initialized){

        cv::absdiff(cur_frame, _backgrnd_i, backgrnd_diff);
        cv::threshold(backgrnd_diff, backgrnd_resp, BAKCGRND_THRESHOLD, 255, cv::THRESH_BINARY);

        cv::bitwise_and(two_fd_resp, backgrnd_resp, final_resp);

        imshow("Backgrnd Resp",backgrnd_resp);

    }
    else{
        final_resp = two_fd_resp;
    }

    vector<Rect> obj_rects = objDetect::getRects(final_resp);


    /* First round. Detect objects only in first round. */
    if(round == 0){

        Rect image_rect(Point(0,0),Size(cur_frame.cols,cur_frame.rows));

        // cout << "DEBUG: objDetect::tick - obj_rects.size(): "<< obj_rects.size() <<endl;
        
        for(const Rect& obj_rect: obj_rects){

            _objs.push_back(fdObject(obj_rect, image_rect));
        }
    }
    else{

        for(const Rect& obj_rect: obj_rects){

            for(fdObject& obj: _objs){

                if(obj.isSameObject(obj_rect)){
                    obj.addRect(obj_rect);
                    break;
                }
            }
        }
    }

    /* Collect and return qualified detected obejects. */
    if(round == _period -1 ){

        if(_objs.size() != 0){

            int i_write = 0;
            for(int i_read = 0; i_read < _objs.size(); ++ i_read){

                if(_objs[i_read].getResult()){
                    _objs[i_write] = _objs[i_read];
                    ++ i_write;
                }
            }
            _objs.resize(i_write);

            this -> backgrndUpdate(cur_frame);

            if(_objs.size() != 0){
                /* Get _objs prepared for next detection. */
                _res = std::move(_objs);
                return true;
            }
        }
    }

    return false;
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

    imshow("Resp", resp);

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
        
        if (bbox.area() > MIN_BBOX_SIZE) { 
            objects.push_back(bbox);
        }
    }

    return objects;
}


bool fdObject::addRect(const Rect& bbox){

    _rects.push_back(bbox);
    return true;
}

vector<fdObject> objDetect::getObjects(void){

    return _res;
}

bool objDetect::addTrackedObjs(const vector<Rect>& rois){
    
    _tracked_ROIs = rois;

    return true;
}


