#pragma once

#ifndef _FUNCS_H_
#define _FUNCS_H_



#include <iostream>
#include <opencv2/opencv.hpp>
#include "kcftracker.hpp"

#include<vector>
using std::vector;

#include <array>
using std::array;

#include <string>
using std::string;

using std::cin, std::cout, std::endl;
using cv::Mat, cv::Rect, cv::Point, cv::Size;

/* seqinfo.ini of test set. */
#define NAME "PETS09-S2L1"
#define imDir "img1"
#define frameRate 7
#define seqLength 795
#define imWidth 768
#define imHeight 576
#define imExt ".jpg"

/* Minmum IoU requirement. */
#define MIN_IOU_REQ (0.7)

/* Tracker States. */
/* Inited but not used yet. */
#define TCR_INIT (0x01 << 1)
/* It's correctly running. */
#define TCR_RUNN (0x01 << 2)
/* Object lost. */
#define TCR_LOST (0x01 << 3)
/* Should be removed after object lost for a long while. */
#define TCR_RMVD (0x01 << 4)


#define MAX_TCR 20

class Tracking;

namespace func{

    double IoU(const Rect& bbox_a, const Rect& bbox_b);
    bool MOT(string input);
}


class Tracking{

public:

    /* 8 bit. */
    char state;

    Tracking():_id(-1), _min_iou_req(-1) {}
    Tracking(int id, int min_iou_req = MIN_IOU_REQ)
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
    int _min_iou_req;

};


class objTrack{

public:

    objTrack():max_tcr(0){}

    objTrack(int max_tcr):max_tcr(max_tcr){

        _p_tcrs = new Tracking[max_tcr];

        for(int i = 0; i < max_tcr; ++ i){
            _p_tcrs[i].set_id(i);
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
