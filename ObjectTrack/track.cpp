#include "funcs.hpp"
#include "detect.hpp"
#include "track.hpp"

#include <cstdio>


bool objTrack::tick(Mat& frame, vector<fdObject> fd_objs){

    // cout << "DEBUG:objTrack-tick - fd_objs.size: " << fd_objs.size() << endl;

    if(fd_objs.empty()) {
        for(int i = 0; i < max_tcr; ++ i){

            Tracking& cur_tcr = _p_tcrs[i];
    
            if(cur_tcr.state & TCR_RUNN){
                cur_tcr.update(frame);
            }
        }
        return true;
    }

    Mat cost = getCostMatrix(fd_objs);

    vector<int> matched_tcr_index;
    hungarianMatch(fd_objs, cost, matched_tcr_index);
    for(int i:matched_tcr_index)
    {
        cout<< i << " ";
    }
    cout<<endl;


    vector<bool> tcr_matched(max_tcr, false);
    int n = fd_objs.size();

    for(int i = 0; i < n; ++ i){
        if(matched_tcr_index[i] != INVAILD_INDEX){
            tcr_matched[ matched_tcr_index[i] ] = true;
            Tracking& cur_tcr = _p_tcrs[ matched_tcr_index[i] ];

            /* Re-start KCF with new Rect if it's more reliable. */
            if((cur_tcr.state & TCR_RUNN) && (cur_tcr.state != TCR_RUNN)){
                -- cur_tcr.state;
                Rect kcf_rect = cur_tcr.getROI();
                Rect fd_rect = fd_objs[i].resultRect();

                if(fd_rect.area() > 0.8f * kcf_rect.area() 
                    && fd_rect.area() < 1.3f * kcf_rect.area()){

                    cur_tcr.restart(frame, fd_rect, cur_tcr.state);
                }
            }
            else{
                cur_tcr.state = TCR_RUNN;
                cur_tcr.update(frame);
            }
        }
        else{
            int index = getFreeTcrIndex();

            if(index == INVAILD_INDEX){

                index = tcrFullHandler();
            }
            tcr_matched[ index ] = true;
            _p_tcrs[index].restart(frame, fd_objs[i].resultRect());
        }
    }

    /* If it's a static object. */
    for(int i = 0; i < max_tcr; ++ i){
        Tracking& cur_tcr = _p_tcrs[i];

        if((cur_tcr.state & TCR_RUNN) && (tcr_matched[i] == false)){
            Rect roi = cur_tcr.getROI() & Rect(0,0, _backgrnd_resp.cols, _backgrnd_resp.rows);
            Mat resp = _backgrnd_resp(roi);
            int count = cv::countNonZero(resp);

            cout << "DEBUG: count: " << count << " roi.area: " << roi.area() <<endl;

            if(count > 0.6f * roi.area()){
                tcr_matched[i] = true;
                cur_tcr.update(frame);
            }
        }
    }

    for(int i = 0; i < max_tcr; ++ i){

        char& state = _p_tcrs[i].state;

        if(state & TCR_RUNN){
            if(tcr_matched[i] == false){
                state = TCR_LOST_3;
            }
        }
        else if(state == TCR_LOST){
            state = TCR_READY;
        }
        else if(state & TCR_LOST){
            -- state;
        }
    }

    return true;

}

Mat objTrack::getCostMatrix(const vector<fdObject>& fd_objs){
    Mat cost(Size(fd_objs.size(), max_tcr), CV_32FC1, cv::Scalar(1.0f));

    int tcr_counter = 0;

    /* Find the biggest APCE value and Peak value. */
    float max_apce = 0.0f, max_peak = 0.0f;
    for(int i = 0; i < max_tcr; ++ i){
        const Tracking& cur_tcr = _p_tcrs[i];
        if((cur_tcr.state & TCR_RUNN) == 0x00){
            continue;
        }

        ++ tcr_counter;

        float apce = cur_tcr.getApce();
        float peak = cur_tcr.getPeak();

        if(apce > max_apce){
            max_apce = apce;
        }
        if(peak > max_peak){
            max_peak  = peak;
        }
    }

    if(tcr_counter == 0){
        return cost;
    }

    /* Calculate costs. */
    for(int y = 0; y < max_tcr; ++ y){

        const Tracking& cur_tcr = _p_tcrs[y];
        if((cur_tcr.state & (TCR_RUNN | TCR_LOST)) == 0x00){
            continue;
        }

        float kcf_score, iou;
        if(cur_tcr.apceIsAccepted()){
            kcf_score = 0.5f * cur_tcr.getApce() / max_apce + 0.5f * cur_tcr.getPeak() / max_peak;
        }
        else{
            kcf_score = 0.0f;
        }

        for(int x = 0; x < fd_objs.size(); ++ x){
            iou = func::IoU(cur_tcr.getROI(), fd_objs[x].resultRect());
            cost.at<float>(y,x) = 0.6f * (1.0f - iou) + 0.4f * (1.0f - kcf_score);
        }
    }

    return cost;
}

bool objTrack::hungarianMatch(const vector<fdObject>& fd_objs, const Mat& cost, vector<int>& matched_tcr_index){
    int n = fd_objs.size();

    /* Variable length array is not allowed, use `vector` instead. */
    vector<bool> tcr_used(max_tcr, false );
    vector<bool> obj_used(n, false);

    /* Point every fdObject to none tracker(index -1). */
    matched_tcr_index.resize(n);
    for(int i = 0; i < n; ++ i){
        matched_tcr_index[i] = INVAILD_INDEX;
    }

    for(int i = 0; i < max_tcr; ++ i){
        if((_p_tcrs[i].state & (TCR_RUNN | TCR_LOST)) == 0x00){
            tcr_used[i] = true;
        }
    }

    while(true){
        /* Bigger than maximum cost `1.0f`.*/
        float best_cost = 2.0f;
        int best_x = -1;
        int best_y = -1;

        for(int y = 0; y < max_tcr; ++ y){
            if(tcr_used[y]) continue;

            for(int x = 0; x < n; ++ x){
                if(obj_used[x]) continue;
                float _cost = cost.at<float>(y, x);

                if(_cost > 0.7f) continue;

                if(_cost < best_cost){
                    best_cost = _cost;
                    best_x = x;
                    best_y = y;
                }
            }
        }

        if(best_x == -1) break;

        matched_tcr_index[best_x] = best_y;

        tcr_used[best_y] = true;
        obj_used[best_x] = true;

    }
    

    return true;
}


vector<Rect> objTrack::getROIs(void) const{

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
        _score = 1000.0f + _current_apce_value + _peak_value;
    }
    else{
        _score = 0.0f + _current_apce_value + _peak_value;
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

    return INVAILD_INDEX;
}

bool objTrack::addBackgrndResp(Mat backgrnd_resp){
    backgrnd_resp.copyTo(_backgrnd_resp);

    return true;
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

float Tracking::getScore(void) const{
    return _score;
}

bool Tracking::isSameObject(const Rect& bbox) const{

    float iou = func::IoU(_roi, bbox);

    return (iou > _min_iou_req);
}

Rect Tracking::getROI(void) const{

    return _roi;
}

float Tracking::getApce(void) const{
    return _current_apce_value;
}

float Tracking::getPeak(void) const{
    return _peak_value;
}

bool Tracking::apceIsAccepted(void) const{
    return _apce_accepted;
}





