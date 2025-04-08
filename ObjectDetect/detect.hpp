#pragma once

#ifndef _FRAMES_DIFFERENCE_H_
#define _FRAMES_DIFFERENCE_H_

#include "funcs.hpp"

#include <vector>
using std::vector;

/* Frame Difference threshold. */
#define FD_THRESHOLD (15)
#define BAKCGRND_THRESHOLD (35)
#define MIN_BBOX_SIZE (500)

/* Detect Objects every DETEC_INTV frames. */
#define DETEC_INTV (2)

/* Miminum frames number requirement for detection. */
#define MIN_DETEC_FRM_REQ ( DETEC_INTV / 2 + 1)


class fdObject;
class objectDetect;


class fdObject{

public:

    fdObject():_min_iou_req(-1),_min_frm_req(-1){}

    fdObject(const Rect& bbox, Rect image_rect,float min_iou_req = MIN_IOU_REQ, 
                int min_frm_req = MIN_DETEC_FRM_REQ):
                _image_rect(image_rect), _min_iou_req(min_iou_req), _min_frm_req(min_frm_req){
        
        /* Store the result. */
        _result = bbox;
    }

    bool addRect(const Rect& bbox);

    bool getResult(void);

    bool isSameObject(const Rect& bbox) const;

    Rect resultRect(void) const;

protected:

    Rect _result;

    Rect _image_rect;

    vector<Rect> _rects;

    float _min_iou_req;
    int _min_frm_req;

};


class objDetect{

public:

    objDetect():_period(0) {}

    objDetect(const Mat& frame, int period = DETEC_INTV):_period(period){
        
        if(_period < 2){

            throw std::runtime_error("ERR:Period must greater than 1");
        }

        _clock_bound =  0xFFFFFFF0 + _period;

        _clock = _period;

        _p_frms = new Mat[_period];

        cv::cvtColor(frame, _p_frms[_period - 1], cv::COLOR_BGR2GRAY);

        _backgrnd = Mat(frame.size(), CV_32FC1, cv::Scalar(0));
    }

    ~objDetect(){

        if(_p_frms != nullptr){
            delete[] _p_frms;
        }
    }

    bool tick(const Mat& frame);
    vector<fdObject> getObjects(void) const;

    Mat FramesDiff(Mat cur_fra, Mat pre_fra, Mat pp_fra = Mat(), bool three_frame_diff = false);
    vector<Rect> getRects(Mat resp);

    Mat getFinalResp(void);

    Mat getBackgrndResp(void) const;

    bool backgrndUpdate(const Mat& frame);

    bool addTrackedObjs(const vector<Rect>& rois);

protected:
    vector<fdObject> _objs;
    vector<fdObject> _res;
    vector<Rect> _tracked_ROIs;

    /* Pointers to binary images. */
    Mat * _p_frms = nullptr;

    /* CV_32FC1 background. */
    Mat _backgrnd = Mat();
    /* CV_8UC1 background. */
    Mat _backgrnd_i = Mat();

    /* Store last two 2 FD results. */
    Mat _fd_diff[2];
    Mat _fd_resp[2];

    vector<Rect> _fd_obj_rects[2];

    /* Store last the 3 FD Difference result. */
    Mat _three_fd_resp;

    /* Store last two Background Difference results. */
    Mat _backgrnd_resp[2];


    
    bool _backgrnd_initialized = false;
    int _backgrnd_init_counter = 7;
    float _alpha_init = 0.8;
    float _alpha = 0.1;    

    const int _period;
    unsigned int  _clock = 0;
    unsigned int _clock_bound = 0;

};


#endif
