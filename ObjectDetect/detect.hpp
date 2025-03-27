#pragma once

#ifndef _FRAMES_DIFFERENCE_H_
#define _FRAMES_DIFFERENCE_H_

#include "funcs.hpp"

#include <vector>
using std::vector;

/* Frame Difference threshold. */
#define FD_THRESHOLD (45)
#define MIN_BBOX_SIZE (500)

/* Detect Objects every 5 frames. */
#define DETEC_INTV (3)

/* Miminum frames number requirement for detection. */
#define MIN_DETEC_FRM_REQ ( DETEC_INTV / 2 )


class fdObject;
class objectDetect;


namespace fd{

}


class fdObject{

public:

    fdObject():_min_iou_req(-1),_min_frm_req(-1){}

    fdObject(const Rect& bbox, double min_iou_req = MIN_IOU_REQ, int min_frm_req = MIN_DETEC_FRM_REQ):
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

    double _min_iou_req;
    int _min_frm_req;

};


class objDetect{

public:

    objDetect():_period(0) {}

    objDetect(int period = DETEC_INTV):_period(period){

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

    static Mat threeFramesDiff(Mat cur_fra, Mat pre_fra, Mat pp_fra);
    static vector<Rect> getRects(Mat resp);

protected:
    vector<fdObject> _objs;
    vector<fdObject> _res;
    Mat * _p_frms = nullptr;

    const int _period;
    unsigned int  _clock = 0;

};


#endif
