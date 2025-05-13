#pragma once

#ifndef _TRACK_H_
#define _TRACK_H_


#include "funcs.hpp"
#include "detect.hpp"
#include "kcftracker.hpp"

/* Tracker States. */
/* It's correctly running. */
#define TCR_RUNN (0x01 << 2)
/* Sub-state of TCR_RUNN, represents how many detections needed 
   to  confirm a newly detected object. */
#define TCR_RUNN_3 (TCR_RUNN + 2)
/* Ready to use. */
#define TCR_READY (0x01 << 3)
/* Object lost. */
#define TCR_LOST (0x01 << 4)
#define TCR_LOST_3 (TCR_LOST + 3)

#define INVAILD_INDEX (-1)


#define MAX_TCR (20)

class Tracking{

public:

    Tracking():_id(-1), _min_iou_req(-1) {}
    Tracking(int id, double min_iou_req = MIN_IOU_REQ)
                :_id(id), _min_iou_req(min_iou_req){
        state = TCR_READY;
    }
    ~ Tracking(){
        if( _p_kcf != nullptr){
            delete _p_kcf;
        }
    }

    bool update(Mat& frame);

    bool restart(Mat first_f, Rect roi, char _state = TCR_RUNN, 
        bool hog = true, bool fixed_window = true, bool multiscale = true, 
        bool lab = true);
    
    bool isSameObject(const Rect& bbox) const;
    Rect getROI(void) const;
    float getScore(void) const;
    float getApce(void) const;
    float getPeak(void) const;
    Mat getAppearance(void) const;
    bool apceIsAccepted(void) const;

    bool getParas(Size& sz, float& scale, float& adjust) const;


    /* 8 bit. */
    char state;

protected:
    int _id;
    KCFTracker* _p_kcf = nullptr;
    Rect _roi;
    float _min_iou_req;

    float _score = 0;

    Mat _newest_appearance;

    /* APCE. */
    float _beta_1 = 0.5;
    float _beta_2 = 0.5;  
    float _alpha_apce = 0.1;  

    float _mean_apce_value = 0;
    float _current_apce_value = 0;

    float _peak_value = 0;
    float _mean_peak_value = 0;

    bool _apce_accepted = true;

};


class objTrack{

public:

    objTrack():max_tcr(0){}

    objTrack(int max_tcr = MAX_TCR, float min_iou_req = MIN_IOU_REQ):
                    max_tcr(max_tcr), _min_iou_req(min_iou_req){

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

    bool getCostMatrix(const Mat& frame, const vector<fdObject>& fd_objs, Mat& cost);
    bool hungarianMatch(const vector<fdObject>& fd_objs, const Mat& cost, vector<int>& matched_tcr_index);

    int tcrFullHandler(void);
    
    int getFreeTcrIndex(void);

    vector<Rect> getROIs(void) const;
    bool addBackgrndResp(Mat backgrnd_resp);

    Mat getFeature(const Rect roi, const Mat& frame, Size tmpl_sz, float scale = 1.0f, float adjust = 1.0f);


    const int max_tcr;

protected:



    Tracking* _p_tcrs = nullptr;
    float _min_iou_req;

    /* CV_8UC1 background. */
    Mat _backgrnd_resp = Mat();

    float _scale_step = 1.05f;

};



#endif
