#pragma once

#ifndef _TRACK_H_
#define _TRACK_H_


#include "funcs.hpp"
#include "detect.hpp"
#include "kcftracker.hpp"

/* Tracker States. */
/* Inited but not used yet. */
#define TCR_INIT (0x01 << 1)
/* Should be removed after object lost for a long while. */
#define TCR_RMVD (0x01 << 2)
/* Object lost. */
#define TCR_LOST (0x01 << 3)
/* It's correctly running. */
#define TCR_RUNN (0x01 << 4)

/* Sub-state of TCR_RUNN, represents how many detections needed 
   to  confirm a newly detected object. */
#define TCR_RUNN_3 (TCR_RUNN + 3)


#define MAX_TCR 20

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

    int id(void);
    bool isSameObject(const Rect& bbox);
    bool restart(Mat first_f, Rect roi,bool hog = true, bool fixed_window = true, 
        bool multiscale = true, bool lab = true);

    Rect getROI(void);

protected:
    int _id;
    KCFTracker* _p_kcf = nullptr;
    Rect _roi;
    float _min_iou_req;

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
    vector<Rect> getROIs(void);

    const int max_tcr;

protected:

    Tracking* _p_tcrs = nullptr;
    int _tcr_count = 0;

};



#endif
