#pragma once

#ifndef _TRACK_H_
#define _TRACK_H_


#include "funcs.hpp"
#include "detect.hpp"
#include "kcftracker.hpp"

/* Tracker States. */

/* It's correctly running. */
#define TCR_RUNN (0x01 << 2)

/* (ABANDONED)Sub-state of TCR_RUNN, represents how many detections needed 
   to  confirm a newly detected object. */
#define TCR_RUNN_3 (TCR_RUNN + 3)

/* Ready to use. */
#define TCR_READY (0x01 << 3)

/* Object lost. */
#define TCR_LOST (0x01 << 4)
#define TCR_LOST_3 (TCR_LOST + 3)

/* If the given states are the same state, including sub-state. */
#define IS_SAME_STATE(state_1, state_2) ((state_1 >> 2) == (state_2 >> 2) )

/* If the given states are state and sub-state. */
#define IS_SUB_STATE(state_1, state_2) (IS_SAME_STATE(state_1, state_2) &&\
                                        ((state_1) != (state_2)))

/* Reduce a sub-state. A sub-state will reach a final state by reducing. */
#define REDUCE_SUB_STATE(state) (-- state)

/* Meaning-less index. */
#define INVALID_INDEX (-1)

/* Maximum trackers running at the same time. */
#define MAX_TCR (20)



/**
 * @class Tracking
 * @brief Represent a sinlge tracked object.
 * 
 */
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

    /* `start` is included in `restart`. */
    bool restart(Mat first_f, Rect roi, char _state = TCR_RUNN, 
        bool hog = true, bool fixed_window = true, bool multiscale = true, 
        bool lab = true);
    
    Rect getROI(void) const;
    float getScore(void) const;
    float getApce(void) const;
    float getPeak(void) const;
    Mat getAppearance(void) const;

    /* 8 bit. */
    char state;

protected:
    int _id;
    KCFTracker* _p_kcf = nullptr;
    Rect _roi;
    float _min_iou_req;

    float _score = 0.0f;

    Mat _newest_appearance;

    /* APCE. */
    float _beta_1 = 0.5f;
    float _beta_2 = 0.5f;  
    float _alpha_apce = 0.1f;  

    float _mean_apce_value = 0.0f;
    float _current_apce_value = 0.0f;

    float _peak_value = 0.0f;
    float _mean_peak_value = 0.0f;

    bool _apce_accepted = true;

};


/**
 * @class objTrack
 * @brief Handle the whole Tracking process.
 * 
 */
class objTrack{

public:

    objTrack():max_tcr(0){}

    objTrack(int max_tcr = MAX_TCR):
                    max_tcr(max_tcr){

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
    
    int getFreeTcrIndex(void);

    vector<Rect> getROIs(void) const;

    Mat getFeature(const Rect roi, const Mat& frame);

    const int max_tcr;

protected:

    Tracking* _p_tcrs = nullptr;


};



#endif
