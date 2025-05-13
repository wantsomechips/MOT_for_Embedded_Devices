#include "funcs.hpp"
#include "detect.hpp"
#include "track.hpp"
#include "ffttools.hpp"

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

    Mat cost;
    getCostMatrix(frame, fd_objs, cost);

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
        int index = matched_tcr_index[i];
        if(index != INVAILD_INDEX){

            tcr_matched[ index ] = true;
            Tracking& cur_tcr = _p_tcrs[ index ];

            if(cur_tcr.state & TCR_RUNN){

                cur_tcr.update(frame);
            }
            else{
                cur_tcr.restart(frame, fd_objs[i].resultRect());
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

    /* Exempt specific objects. */
    for(int i = 0; i < max_tcr; ++ i){
        Tracking& cur_tcr = _p_tcrs[i];

        /* Objects unmatched within few detections. */
        if((cur_tcr.state & TCR_RUNN) && (tcr_matched[i] == false) 
            && (cur_tcr.state < TCR_RUNN_3)){
            ++ cur_tcr.state;
            tcr_matched[i] = true;
            cur_tcr.update(frame);
        }
    }

    for(int i = 0; i < max_tcr; ++ i){

        char& state = _p_tcrs[i].state;

        if((state & TCR_RUNN) && (tcr_matched[i] == false)){
                state = TCR_LOST;
        }
    }

    return true;

}

bool objTrack::getCostMatrix(const Mat& frame, const vector<fdObject>& fd_objs, Mat& cost){
    cost = std::move( Mat(Size(fd_objs.size(), max_tcr), CV_32FC1, cv::Scalar(1.0f)));

    /* Find the biggest APCE value and Peak value. */
    float max_apce = 1.0f, max_peak = 1.0f;
    for(int i = 0; i < max_tcr; ++ i){
        const Tracking& cur_tcr = _p_tcrs[i];
        if((cur_tcr.state & TCR_RUNN) == 0x00){
            continue;
        }

        float apce = cur_tcr.getApce();
        float peak = cur_tcr.getPeak();

        if(apce > max_apce){
            max_apce = apce;
        }
        if(peak > max_peak){
            max_peak  = peak;
        }
    }

    /* Calculate costs. */
    for(int y = 0; y < max_tcr; ++ y){

        const Tracking& cur_tcr = _p_tcrs[y];

        if((cur_tcr.state & (TCR_RUNN | TCR_LOST)) == 0x00){
            continue;
        }

        float iou, max_iou_needed = 0.7f;
        float appearance_score = 0.0f;

        for(int x = 0; x < fd_objs.size(); ++ x){
            Rect fd_roi = fd_objs[x].resultRect();
            Rect kcf_roi = cur_tcr.getROI();

            // Point new_tl((int)(fd_roi.x + fd_roi.width / 2.0f - kcf_roi.width / 2.0f), (int)(fd_roi.y + fd_roi.height / 2.0f - kcf_roi.height / 2.0f));
            // Rect new_roi(new_tl, kcf_roi.size());

            // new_roi = new_roi & Rect(0, 0, frame.cols, frame.rows);

            Size sz;

            float scale, adjust;
            
            bool get_paras_success = cur_tcr.getParas(sz, scale, adjust);

            Mat roi_appearance = getFeature(fd_roi, frame, sz, scale, adjust);

            Mat tcr_appearance = cur_tcr.getAppearance();


            // cout << roi_appearance.size() << " --- " << tcr_appearance.size() << endl;

            // cv::normalize(roi_appearance, roi_appearance);
            // cv::normalize(tcr_appearance, tcr_appearance);


            cv::reduce(roi_appearance , roi_appearance , 1, cv::REDUCE_AVG);
            cv::reduce(tcr_appearance, tcr_appearance, 1, cv::REDUCE_AVG);


            float sigma = 0.05f;
            Mat diff = roi_appearance - tcr_appearance;
            appearance_score = std::exp(- diff.dot(diff) / (2 * sigma * sigma));


            // std::cout << "norm² = " << diff.dot(diff)  << " score = " << appearance_score << std::endl;

            
            iou = func::IoU(kcf_roi, fd_roi);

            /* Truncate. */
            iou = (iou > max_iou_needed)? 1.0f : iou;

            if(cur_tcr.state & TCR_RUNN){ 
                cost.at<float>(y,x) = 0.7f * (1.0f - iou) + 0.3f * (1.0f - appearance_score);
            }
            else{
                cost.at<float>(y,x) = 1.0f * (1.0f - appearance_score);
            }

            cout << "Ap: " << appearance_score << endl << "IoU: " << iou << endl << "Ct: " << cost.at<float>(y,x) <<endl<<endl;
        }
        cout<< "--- --- ---" <<endl << endl;
    }

    return true;
}

bool Tracking::getParas(Size& sz, float& scale, float& adjust) const{
    sz = _p_kcf -> _tmpl_sz;

    scale = _p_kcf -> latest_scale;

    adjust = _p_kcf -> latest_adjust;

    return true;
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


    float biggest_cost_allowed = 0.5f;
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

                if(_cost > biggest_cost_allowed) continue;

                if(_cost < best_cost){
                    best_cost = _cost;
                    best_x = x;
                    best_y = y;
                }
            }
        }

        if(best_x == -1 || best_y == -1) break;

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

    for(int i = 0;i < max_tcr; ++ i){        
        if(_p_tcrs[i].state & TCR_READY){
            
            return i;
        }
    }

    for(int i = 0;i < max_tcr; ++ i){
        if(_p_tcrs[i].state & TCR_LOST){
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
    _p_kcf -> init(roi, first_f);

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




Mat Tracking::getAppearance(void) const{
    /* Return the features KCF tracker is using. */
    return _p_kcf -> _tmpl;
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





Mat objTrack::getFeature(const Rect roi, const Mat& frame, Size tmpl_sz, float scale, float adjust){
    bool hog = true, fixed_window = true;
    bool multiscale = true, lab = true;

    static KCFTracker tmp_kcf(hog, fixed_window, multiscale, lab);
    Mat appearance;

    tmp_kcf.getRoiFeature(roi, frame, appearance, tmpl_sz, scale, adjust);

    return appearance;
}