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


bool fdObject::isSameObject(const Rect& bbox){

    Rect newest = _rects.back();
    double iou = func::IoU(newest,bbox);

    cout << "DEBUG: fdObject::isSameObject - IOU: "<< iou <<endl;

    return (iou > _min_iou_req);
}

bool fdObject::addFrame(const Rect& bbox){

    _rects.push_back(bbox);
    return true;
}

bool fdObject::getResult(void){

    /* Detection failed. */
    if (_rects.size() < MIN_DETEC_FRM_REQ){
        return false;
    }

    /* Merge rects. */
    Rect merged = _rects[0];
    for (int i = 1; i < _rects.size(); ++i) {
        merged |= _rects[i];
    }

    // _result = merged;
    _result = _rects.back();
    

    return true;

}

Rect fdObject::resultRect(void){

    return _result;
}



bool objDetect::tick(const Mat& frame){

    int round = _clock % _period;

    cout << "DEBUG: objDetect::tick - round: "<< round <<endl;


    if(_clock < (0xFFFFFFF5)){
        _clock ++ ;
    }
    else{
        _clock = 0;
    }

    /* Deep Copy. Otherwise it would be Shallow Copy. */
    _p_frms[round] = frame.clone();

    if(round == 2){
        
        Mat resp = objDetect::threeFramesDiff(_p_frms[round], _p_frms[round-1], _p_frms[round-2]);
        vector<Rect> obj_rects = objDetect::getRects(resp);

        cv::imshow("Resp", resp);

        // cout << "DEBUG: objDetect::tick - obj_rects.size(): "<< obj_rects.size() <<endl;
        
        for(const Rect& obj_rect: obj_rects){

            _objs.push_back(fdObject(obj_rect));
        }
    }
    else if(round > 2){

        Mat resp = objDetect::threeFramesDiff(_p_frms[round], _p_frms[round-1], _p_frms[round-2]);
        vector<Rect> obj_rects = objDetect::getRects(resp);

        cv::imshow("Resp", resp);

        for(const Rect& obj_rect: obj_rects){

            for(fdObject& obj: _objs){

                if(obj.isSameObject(obj_rect)){
                    obj.addFrame(obj_rect);
                    break;
                }
            }
        }
    }

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

            if(_objs.size() != 0){
                _res = std::move(_objs);
                return true;
            }
        }
    }

    return false;
}


vector<fdObject> objDetect::getObjects(void){

    return _res;
}


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

Mat objDetect::threeFramesDiff(Mat cur_fra, Mat pre_fra, Mat pp_fra){

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

vector<Rect> objDetect::getRects(Mat resp) {

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