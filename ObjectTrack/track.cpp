#include "funcs.hpp"
#include "detect.hpp"
#include "track.hpp"

#include <cstdio>


bool objTrack::tick(Mat& frame, vector<fdObject> fd_objs){

    // cout << "DEBUG:objTrack-tick - fd_objs.size: " << fd_objs.size() << endl;

    /* Update trackers. */
    for(int i = 0; i < max_tcr; i ++){

        char state = _p_tcrs[i].state;

        if(state & TCR_RUNN){
            _p_tcrs[i].update(frame);
        }
        else if(state == TCR_LOST){
            state = TCR_READY;
        }
        else if(state & TCR_LOST){
            _p_tcrs[i].update(frame);
        }
    }

    /* Detecting new objects. */
    int i_write = 0;
    for(int i_read = 0; i_read < fd_objs.size(); i_read ++){

        Rect fd_rect = fd_objs[i_read].resultRect();

        bool existed = false;
        for(int i = 0; i < max_tcr; ++ i){

            Tracking& cur_tcr  = _p_tcrs[i];

            /* It's TRC_RUNN or sut-states. */
            if(cur_tcr.state & TCR_RUNN){



                existed = cur_tcr.isSameObject(fd_rect);

                if(existed) {
                    /* It's a sub-state of TCR_RUNN. */
                    if(cur_tcr.state != TCR_RUNN){

                        cur_tcr.state --;
                        /* Re-start KCF with new Rect if it's more reliable. */
                        Rect kcf_rect = cur_tcr.getROI();
                        float iou = func::IoU(fd_rect, kcf_rect);

                        if(iou > _min_iou_req && fd_rect.area() > 1.1 * kcf_rect.area() 
                            && fd_rect.area() < 1.3 * kcf_rect.area()){

                            cur_tcr.restart(frame, fd_rect, cur_tcr.state);
                        }

                    }
                    break;
                }
            }
        }

        if(existed == false){
            fd_objs[i_write] = fd_objs[i_read];
            ++ i_write;
        }
    }
    fd_objs.resize(i_write);

    for(fdObject& fd_obj: fd_objs){

        int index = getFreeTcrIndex();

        if(index == INVALID_INDEX){

            index = tcrFullHandler();
        }

        _p_tcrs[index].restart(frame, fd_obj.resultRect());

    }

    return true;

}

vector<Rect> objTrack::getROIs(void){

    vector<Rect> res;

    for(int i = 0; i < max_tcr; ++ i){

        if(_p_tcrs[i].state & TCR_RUNN){
            res.push_back( _p_tcrs[i].getROI());
        }
    }

    return res;
}

bool Tracking::update(Mat& frame){

    Rect bbox;
    bbox = _p_kcf -> update(frame, _beta_1, _beta_2, _alpha_apce, _peak_value, _mean_peak_value, 
                            _mean_apce_value, _current_apce_value, _apce_accepted);
    _roi = bbox;

    if(_apce_accepted){
        _score = 1000.0 + _current_apce_value + _peak_value;

    }
    else{
        _score = 0.0 + _current_apce_value + _peak_value;
        if(state & TCR_RUNN){
            state = TCR_LOST_3;
        }
        else{
            state --;
        }
    }

    char title[6];
    snprintf(title, sizeof(title), "id:%02d", _id);

    char apce_datas[23];
    snprintf(apce_datas, sizeof(apce_datas), "APCE: %04.1f / %04.1f %c",
                _current_apce_value, _mean_apce_value, _apce_accepted? 'T' : 'F');

    char peak_datas[20];
    snprintf(peak_datas, sizeof(peak_datas), "Peak: %04.1f / %04.1f",
                _peak_value * 100, _mean_peak_value * 100);

    cv::putText(frame, title, cv::Point(bbox.x, bbox.y - 1),cv::FONT_HERSHEY_SIMPLEX,
                         0.5, cv::Scalar(0,0,255), 1, cv::LINE_AA);
    cv::putText(frame, apce_datas, cv::Point(bbox.x,bbox.y + bbox.height + 13),cv::FONT_HERSHEY_SIMPLEX,
                         0.3, cv::Scalar(0,255,0), 1, cv::LINE_AA);
    cv::putText(frame, peak_datas, cv::Point(bbox.x,bbox.y + bbox.height + 13 * 2),cv::FONT_HERSHEY_SIMPLEX,
                         0.3, cv::Scalar(0,255,0), 1, cv::LINE_AA);
    cv::rectangle(frame,bbox, cv::Scalar(0,0,255));

    return true;
}

int objTrack::tcrFullHandler(void){

    /* Return the index of Tracker with minimum score. */
    int index = 0;
    float score_min = _p_tcrs[0].getScore();
    float score;
    for(int i = 1; i < max_tcr; ++ i){
        score = _p_tcrs[i].getScore();
        if(score < score_min){
            score_min = score;
            index = i;
        }
    }

    return index;
}


int objTrack::getFreeTcrIndex(void){

    for(int i = 0;i < max_tcr; i++){

        if(_p_tcrs[i].state == TCR_READY){
            
            return i;
        }
    }

    return INVALID_INDEX;
}

bool Tracking::restart(Mat first_f, Rect roi, char _state, bool hog, 
    bool fixed_window, bool multiscale, bool lab){

    _roi = roi;
    state = _state;
    if(_p_kcf != nullptr) delete _p_kcf;
    _p_kcf = new KCFTracker(hog, fixed_window, multiscale, lab);
    _p_kcf -> init(roi,first_f);

    return true;
}

float Tracking::getScore(void){
    return _score;
}

bool Tracking::isSameObject(const Rect& bbox){

    float iou = func::IoU(_roi, bbox);

    return (iou > _min_iou_req);
}

Rect Tracking::getROI(void){

    return _roi;
}







