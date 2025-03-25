#pragma once

#ifndef _FRAMES_DIFFERENCE_H_
#define _FRAMES_DIFFERENCE_H_

#include "funcs.hpp"
#include <vector>

using std::vector;


/* Frame Difference threshold. */
#define FD_THRESHOLD (100)
#define MIN_BBOX_SIZE (500)

/* Detect Objects every 5 frames. */
#define DETEC_INTV (5)

/* Miminum frames number requirement for detection. */
#define MIN_DETEC_FRM_REQ ( DETEC_INTV / 2 + 1)

/* Expand ratio for detected result. */
#define DETEC_EXPD_RATIO (1.2)


#define IOU_THRESHOLD (0.6)

class fdObject;
class objectDetect;


namespace fd{

}


class fdObject{

public:
    vector<Rect> rects;
    Rect result;

    fdObject(){}
    fdObject(const Rect& bbox){
        /* Deep Copy. */
        rects.push_back(bbox);
    }

    bool isSameObject(const Rect& bbox);

    bool addFrame(const Rect& bbox);

    bool getResult(void);

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

    Mat threeFramesDiff(Mat cur_fra, Mat pre_fra, Mat pp_fra);
    vector<Rect> getRects(Mat resp);
    vector<tuple>

protected:
    vector<fdObject> _objs;
    Mat * _p_frms = nullptr;

    const int _period;
    unsigned int  _clock = 0;

};


#endif
