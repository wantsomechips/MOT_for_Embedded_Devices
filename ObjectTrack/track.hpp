#pragma once

#ifndef _TRACK_H_
#define _TRACK_H_


#include "funcs.hpp"
#include "detect.hpp"
#include "kcftracker.hpp"

class Tracking{

public:

    /* 8 bit. */
    char state;

    Tracking():_id(-1), _min_iou_req(-1) {}
    Tracking(int id, double min_iou_req = MIN_IOU_REQ)
                :_id(id), _min_iou_req(min_iou_req){
        state = TCR_INIT;
    }
    ~ Tracking(){
        if( _p_kcf != nullptr){
            delete _p_kcf;
        }
    }

    bool update(Mat& frame);
    bool init(Mat first_f, Rect roi,bool hog = true, bool fixed_window = true, 
                                    bool multiscale = true, bool lab = true);

    bool set_id(int new_id);
    int id(void);
    bool isSameObject(const Rect& bbox);

protected:
    int _id;
    KCFTracker* _p_kcf = nullptr;
    Rect _roi;
    double _min_iou_req;

};


class objTrack{

public:

    objTrack():max_tcr(0){}

    objTrack(int max_tcr = MAX_TCR):max_tcr(max_tcr){

        _p_tcrs = new Tracking[max_tcr];

        for(int i = 0; i < max_tcr; ++ i){
            _p_tcrs[i] = std::move(Tracking(i));
        }

    }
    ~objTrack(){
        if(_p_tcrs != nullptr){
            delete[] _p_tcrs;
        }
    }

    bool tick(Mat& frame, vector<fdObject> fd_objs = {});
    bool tcrFullHandler(void);

    const int max_tcr;

protected:

    Tracking* _p_tcrs = nullptr;
    int _tcr_count = 0;

};



#endif
