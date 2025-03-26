#pragma once

#ifndef _FRAMES_DIFFERENCE_H_
#define _FRAMES_DIFFERENCE_H_

#include "funcs.hpp"

#include <vector>
using std::vector;

#include <tuple>
using std::tuple;

/* Frame Difference threshold. */
#define FD_THRESHOLD (100)
#define MIN_BBOX_SIZE (500)

/* Detect Objects every 5 frames. */
#define DETEC_INTV (5)

/* Miminum frames number requirement for detection. */
#define MIN_DETEC_FRM_REQ ( DETEC_INTV / 2 )

/* Expand ratio for detected result. */
#define DETEC_EXPD_RATIO (1.2)


class fdObject;
class objectDetect;


namespace fd{

}


class fdObject{

public:

    fdObject():_min_iou_req(-1),_min_frm_req(-1){}

    fdObject(const Rect& bbox, int min_iou_req = MIN_IOU_REQ, int min_frm_req = MIN_DETEC_FRM_REQ):
                _min_iou_req(min_iou_req), _min_frm_req(min_frm_req){
        /* Deep Copy. */
        _rects.push_back(bbox);
    }

    bool isSameObject(const Rect& bbox);

    bool addFrame(const Rect& bbox);

    bool getResult(void);

    Rect getRect(void);

protected:
    Rect _result;

    vector<Rect> _rects;

    int _min_iou_req;
    int _min_frm_req;

};


class objDetect{

public:

    objDetect():_period(0) {}

    objDetect(int period):_period(period){

        _p_frms = new Mat[period];
    }

    ~objDetect(){
        if(_p_frms != nullptr){
            delete[] _p_frms;
        }
    }

    bool tick(const Mat& frame);
    vector<fdObject> getObjects(void);

    static Mat threeFramesDiff(Mat cur_fra, Mat pre_fra, Mat pp_fra);
    static vector<Rect> getRects(Mat resp);

protected:
    vector<fdObject> _objs;
    Mat * _p_frms = nullptr;

    const int _period;
    unsigned int  _clock = 0;

};


#endif
