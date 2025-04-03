#pragma once

#ifndef _FRAMES_DIFFERENCE_H_
#define _FRAMES_DIFFERENCE_H_

#include "funcs.hpp"

#include <vector>
using std::vector;

/* Frame Difference threshold. */
#define FD_THRESHOLD (15)
#define BAKCGRND_THRESHOLD (20)
#define MIN_BBOX_SIZE (500)

/* Detect Objects every DETEC_INTV frames. */
#define DETEC_INTV (3)

/* Miminum frames number requirement for detection. */
#define MIN_DETEC_FRM_REQ ( DETEC_INTV / 2 + 1)


class fdObject;
class objectDetect;


class fdObject{

public:

    fdObject():_min_iou_req(-1),_min_frm_req(-1){}

    fdObject(const Rect& bbox, float min_iou_req = MIN_IOU_REQ, int min_frm_req = MIN_DETEC_FRM_REQ):
                _min_iou_req(min_iou_req), _min_frm_req(min_frm_req){
        /* Deep Copy. */
        _rects.push_back(bbox);
    }

    bool isSameObject(const Rect& bbox);

    bool addFrame(const Rect& bbox);

    bool getResult(void);

    Rect resultRect(void);

protected:
    Rect _result;

    vector<Rect> _rects;

    float _min_iou_req;
    int _min_frm_req;

};


class objDetect{

public:

    objDetect():_period(0) {}

    objDetect(int period = DETEC_INTV):_period(period){

        _clock_bound =  0xFFFFFFF0 + _period;
        
        if(_period < 3){

            throw std::runtime_error("ERR:Period must greater than 3");
        }

        _p_frms = new Mat[period];
    }

    ~objDetect(){

        if(_p_frms != nullptr){
            delete[] _p_frms;
        }
    }

    bool tick(const Mat& frame);
    vector<fdObject> getObjects(void);

    Mat FramesDiff(Mat cur_fra, Mat pre_fra, Mat pp_fra = Mat());
    vector<Rect> getRects(Mat resp);
    bool backgrndUpdate(const Mat& frame);

    bool addTrackedObjs(const vector<Rect>& rois);

protected:
    vector<fdObject> _objs;
    vector<fdObject> _res;
    vector<Rect> _tracked_ROIs;

    /* Pointers to binary images. */
    Mat * _p_frms = nullptr;

    /* 2 Frames Difference - _frm_bound = 1;
       3 Frames Difference - _frm_bound = 2. */
    int _frm_bound = 1;

    /* CV_32FC1 background. */
    Mat _backgrnd = Mat();
    /* CV_8UC1 background. */
    Mat _backgrnd_i = Mat();
    bool _backgrnd_initialized = false;
    int _backgrnd_init_counter = 5;
    float _alpha_init = 0.8;
    float _alpha = 0.1;
    /* Expand target Rects when calculating background mask.*/
    float _expand_ratio = 1.2;
    

    const int _period;
    unsigned int  _clock = 0;
    unsigned int _clock_bound = 0;

};


#endif
