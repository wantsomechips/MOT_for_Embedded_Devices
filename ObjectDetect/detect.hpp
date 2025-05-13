#pragma once

#ifndef _FRAMES_DIFFERENCE_H_
#define _FRAMES_DIFFERENCE_H_

#include "funcs.hpp"

#include <vector>
using std::vector;

#include <queue>

#include <stdint.h>

/* Frame Difference threshold. */
#define FD_THRESHOLD (15)
#define BAKCGRND_THRESHOLD (25)
#define MIN_BBOX_HEIGHT (20)
#define MIN_BBOX_WIDTH (10)

/* Detect Objects every DETEC_INTV frames. */
#define DETEC_INTV (5)

/* Miminum frames number requirement for detection. */
#define MIN_DETEC_FRM_REQ ( DETEC_INTV / 2 + 1)

/* Size of frames buffer. Use 2 Frames Difference, so the size is 2. */
#define FRM_BUFFER_SIZE (2)

#if !defined(MAX) || !defined(MIN)
#define MAX(a,b) ((a) > (b) ? (a):(b))
#define MIN(a,b) ((a) < (b) ? (a):(b))
#endif



class fdObject{

public:

    fdObject():_min_iou_req(-1),_min_frm_req(-1){}

    fdObject(const Rect& bbox,float min_iou_req = MIN_IOU_REQ, 
                int min_frm_req = MIN_DETEC_FRM_REQ):
                _min_iou_req(min_iou_req), _min_frm_req(min_frm_req){
        
        /* Store the result. */
        _result = bbox;
    }

    bool addRect(const Rect& bbox);

    bool getResult(void);

    bool isSameObject(const Rect& bbox) const;

    Rect resultRect(void) const;

protected:

    Rect _result;

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

        _p_frms = new Mat[FRM_BUFFER_SIZE];

        cv::cvtColor(frame, _p_frms[0], cv::COLOR_BGR2GRAY);

        /* Pre-process done. */
        _clock = 1;

        _backgrnd = Mat(frame.size(), CV_8UC1, cv::Scalar(0));
    }

    ~objDetect(){

        if(_p_frms != nullptr){
            delete[] _p_frms;
        }
    }

    bool tick(const Mat& frame);
    vector<fdObject> getObjects(void) const;

    Mat FramesDiff(const Mat& pre_fra, const Mat& cur_fra);
    vector<Rect> getRects(Mat resp);

    Mat getFinalResp(void);

    Mat getBackgrndResp(void) const;

    bool backgrndUpdate(const Mat& frame, const vector<Rect>& obj_rects);

    bool addTrackedObjs(const vector<Rect>& rois);

    bool getBackgrndDiffResp(const Mat& cur_frame, Mat& final_resp);


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
    Mat _fd_resp;

    /* Store last two Background Difference results. */
    Mat _backgrnd_resp;


    
    bool _backgrnd_initialized = false;
    int _backgrnd_init_counter = 20;
    float _alpha_init = 0.8;
    float _alpha = 0.1;    

    const uint_fast32_t _period;

    /* 64 bits could be faster than 32 bits in 64 bits platform. 
       uint_fast32_t can handle it. */
    uint_fast32_t  _clock;

};


#endif
