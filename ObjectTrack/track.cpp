

/**
 * @file track.cpp
 * @brief Handles the obejct tracking part of the MOT system.
 * @author wantSomeChips
 * @date 2025
 * 
 */

#include "funcs.hpp"
#include "detect.hpp"
#include "track.hpp"
#include "ffttools.hpp"

#include <cstdio>

/**
 * @brief Top-level abstract function for the object Tracking. Handle the Tracking logic.
 *
 * @param frame     A single frame image input.
 * @param fd_objs   Detected objects, which is the result of Detection.
 *                  It's empty when only trackers update and Detection is skipped 
 *                  due to Detection interval. 
 * 
 * @return Boolean value. Return `true` if the Tracking goes on properly. 
 * 
 */
bool objTrack::tick(Mat& frame, vector<fdObject> fd_objs){

    // cout << "DEBUG:objTrack-tick - fd_objs.size: " << fd_objs.size() << endl;

    if(fd_objs.empty()) {
        for(int i = 0; i < max_tcr; ++ i){

            Tracking& cur_tcr = _p_tcrs[i];
    
            if(IS_SAME_STATE(cur_tcr.state, TCR_RUNN)){
                cur_tcr.update(frame);
            }
        }
        return true;
    }

    Mat cost;
    getCostMatrix(frame, fd_objs, cost);

    vector<int> matched_tcr_index;
    hungarianMatch(fd_objs, cost, matched_tcr_index);

    vector<bool> tcr_matched(max_tcr, false);

    int n = fd_objs.size();

    /* Update only when cost less than `max_cost_allowed`. Otherwise, `restart` it. */
    float max_cost_allowed = 0.5f;
    for(int i = 0; i < n; ++ i){
        int index = matched_tcr_index[i];
        /* If sucessfully matched a tracker. */
        if(index != INVALID_INDEX){

            tcr_matched[ index ] = true;
            Tracking& cur_tcr = _p_tcrs[ index ];

            if(cost.at<float>(index,i) < max_cost_allowed){

                cur_tcr.update(frame);
            }
            else{
                cur_tcr.restart(frame, fd_objs[i].resultRect());
            }
        }
        else{

            int index = getFreeTcrIndex();
            
            tcr_matched[ index ] = true;
            _p_tcrs[index].restart(frame, fd_objs[i].resultRect());
        }
    }

    // /* Exempt specific objects. */
    // for(int i = 0; i < max_tcr; ++ i){
    //     Tracking& cur_tcr = _p_tcrs[i];

    //     /* Objects unmatched within few detections. */
    //     if((cur_tcr.state & TCR_RUNN) && (tcr_matched[i] == false) 
    //         && (cur_tcr.state < TCR_RUNN_3)){
    //         ++ cur_tcr.state;
    //         tcr_matched[i] = true;
    //         cur_tcr.update(frame);
    //     }
    // }

    for(int i = 0; i < max_tcr; ++ i){

        char& state = _p_tcrs[i].state;

        if(IS_SAME_STATE(state, TCR_RUNN) && (tcr_matched[i] == false)){
                state = TCR_LOST_3;
        }
    }

    return true;

}

/**
 * @brief Calculate the cost matrix for the appropriate Hungarian Algorithm to pair the detected objects and trackers. 
 * 
 * A higher cost indicates a lower confidence for matching this detected object to the tracker.
 *
 * @param frame     A single frame image input.
 * @param fd_objs   Detected objects.
 * @param cost      Cost matrix. This is the result of this function.
 * 
 * @return Boolean value. Return `true` if the calculation goes on properly. 
 * 
 */
bool objTrack::getCostMatrix(const Mat& frame, const vector<fdObject>& fd_objs, Mat& cost){

    const int n = fd_objs.size();
    cost = std::move( Mat(Size(n, max_tcr), CV_32FC1, cv::Scalar(1.0f)));

    /* Get features of all detected objects. */
    vector<Mat> fd_features(n);

    for (int i = 0; i < n; ++ i){

        Rect fd_roi = fd_objs[i].resultRect();

        fd_features[i] = getFeature(fd_roi, frame);

        /* `reduce` is faster and more accurate than `normalize`. */
        cv::reduce(fd_features[i] , fd_features[i] , 1, cv::REDUCE_AVG);

    }


    /* Get features of all trackers. */
    vector<Mat> tcr_features(max_tcr);

    for (int i = 0; i < max_tcr; ++ i){

        const char& state = _p_tcrs[i].state;

        if(IS_SAME_STATE(state, TCR_RUNN) || IS_SAME_STATE(state, TCR_LOST)){
            
            tcr_features[i] = _p_tcrs[i].getAppearance();

            /* `reduce` is faster and more accurate than `normalize`. */
            cv::reduce(tcr_features[i] , tcr_features[i] , 1, cv::REDUCE_AVG);
        }

    }


    /* Calculate costs. */
    for(int y = 0; y < max_tcr; ++ y){

        Tracking& cur_tcr = _p_tcrs[y];

        char& state = cur_tcr.state;

        if(!(IS_SAME_STATE(state, TCR_RUNN) || IS_SAME_STATE(state, TCR_LOST))){
            continue;
        }

        /* Exmpt newly lost tracker. */
        // if(IS_SUB_STATE(state, TCR_LOST)){

        //     REDUCE_SUB_STATE(state);
        //     continue;
        // }

        for(int x = 0; x < n; ++ x){

            /* Get IoU value. */
            float iou;

            /* IoU is meaning-less for a lost tracker. */
            if(IS_SAME_STATE(state, TCR_RUNN)){
                Rect kcf_roi = cur_tcr.getROI();
                Rect fd_roi = fd_objs[x].resultRect();

                iou = func::IoU(kcf_roi, fd_roi);
                
            }
            else{
                iou = 0.0f;
            }
            

            /* Get feature similarity using Gaussian Kernel Function. */
            float appearance_score = 0.0f;
            
            /* Gaussian Kernel Funciton. */
            float sigma = 0.05f;
            Mat diff = fd_features[x] - tcr_features[y];
            appearance_score = std::exp(- diff.dot(diff) / (2 * sigma * sigma));

            // appearance_score = 0.0f;

            // std::cout << "norm² = " << diff.dot(diff)  << " score = " << appearance_score << std::endl;

            cost.at<float>(y,x) = 0.5f * (1.0f - iou) + 0.5f * (1.0f - appearance_score);
 

            // cout << "Ap: " << appearance_score << endl << "IoU: " << iou << endl << endl; 
            // cout << "Ct: " << cost.at<float>(y,x) <<endl<<endl;
        }
        // cout<< "--- --- ---" <<endl << endl;
    }

    return true;
}

/**
 * @brief An greedy approximate Hungarian Match Algorithm implementation.
 *
 * @param fd_objs           Detected objects.
 * @param cost              Cost matrix we calculated.
 * @param matched_tcr_index Index of the successfully matched trackers. 
 *                          This is the result of this function.
 * 
 * @return Boolean value. Return `true` if the match goes on properly. 
 * 
 */
bool objTrack::hungarianMatch(const vector<fdObject>& fd_objs, const Mat& cost, vector<int>& matched_tcr_index){
    int n = fd_objs.size();

    /* Variable length array is not allowed, use `vector` instead. */
    vector<bool> tcr_used(max_tcr, false );
    vector<bool> obj_used(n, false);

    /* Point every detected objects to none tracker index (-1). */
    matched_tcr_index.resize(n);
    for(int i = 0; i < n; ++ i){
        matched_tcr_index[i] = INVALID_INDEX;
    }

    for(int i = 0; i < max_tcr; ++ i){
        const char& state = _p_tcrs[i].state;
        if(!(IS_SAME_STATE(state, TCR_RUNN) || IS_SAME_STATE(state, TCR_LOST))){
            tcr_used[i] = true;
        }
    }


    float biggest_cost_allowed = 1.0f;
    while(true){
        /* Bigger than maximum cost `1.0f`.*/
        float best_cost = 2.0f;
        int best_x = -1;
        int best_y = -1;

        for(int y = 0; y < max_tcr; ++ y){
            /* Skip paired or uninitiated trackers. */
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


/**
 * @brief Get the index of a free KCF tracker. 
 * 
 * A maximum number of trackers is enforced to control overall resource usage.
 * 
 * If all tracker are busy, schduling, based on tracking quality, will be performed.
 *
 * @param void void.
 * 
 * @return The index of a free tracker.
 * 
 */
int objTrack::getFreeTcrIndex(void){

    /* Return the index of Tracker with minimum score. */

    /* If there's free tracker. */
    for(int i = 0;i < max_tcr; ++ i){  

        if(IS_SAME_STATE(_p_tcrs[i].state, TCR_READY)){
            return i;
        }
    }

    /* No free tracker. Get a lost tracker. */
    for(int i = 0;i < max_tcr; ++ i){

        if(IS_SAME_STATE(_p_tcrs[i].state, TCR_LOST)){
            return i;
        }
    }

    /* All trackers are tracking. Find the tracker with lowest score. */
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

/**
 * @brief Extract the features of region of interest exactly in the same way as tracker.
 * 
 *
 * @param roi       Bounding box of the region of interest.
 * @param frame     A single frame image input.
 * 
 * @return Boolean value. Return `true` if the calculation goes on properly. 
 * 
 */
Mat objTrack::getFeature(const Rect roi, const Mat& frame){
    bool hog = true, fixed_window = true;
    bool multiscale = true, lab = true;

    static KCFTracker tmp_kcf(hog, fixed_window, multiscale, lab);
    Mat appearance;

    tmp_kcf.getRoiFeature(roi, frame, appearance);

    return appearance;
}

/**
 * @brief Get all the bounding boxes currently tracking.
 *
 * Encapsulation protects class data by using functions for access, 
 * preventing accidental changes.
 * 
 * @return Bounding boxes currently tracking.
 * 
 */
vector<Rect> objTrack::getROIs(void) const{

    vector<Rect> res;

    for(int i = 0; i < max_tcr; ++ i){

        if(IS_SAME_STATE(_p_tcrs[i].state, TCR_RUNN)){
            res.push_back( _p_tcrs[i].getROI());
        }
    }

    return res;
}

/**
 * @brief Update all the trackers with an new single frame of image.
 * 
 * And paint Detection and Tracking results on the image.
 *
 * @param frame     A single frame image input.
 * 
 * @return Boolean value. Return `true` if the updating goes on properly. 
 * 
 */
bool Tracking::update(Mat& frame){

    Rect bbox;
    bbox = _p_kcf -> update(frame, _beta_1, _beta_2, _alpha_apce, _peak_value, _mean_peak_value, 
                            _mean_apce_value, _current_apce_value, _apce_accepted);
    _roi = bbox;

    if(_apce_accepted){
        /* Bonus for accepted trackers. */
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

/**
 * @brief Start or restart a tracking process with an new tracker.
 *
 * Initializes a new tracker with the given parameters and sets the tracker's state.
 *
 * @param first_f       The initial frame used for tracker initialization.
 * @param roi           Bounding box of the region of interest to track.
 * @param _state        Initial state assigned to the tracker. Default value is `TCR_RUNN`.
 * @param hog           Whether to use HOG features. Default value is `true`.
 * @param fixed_window  Whether to use a fixed window size. Default value is `true`.
 * @param multiscale    Whether to enable multi-scale tracking. Default value is `true`.
 * @param lab           Whether to include Lab color features. Default value is `true`.
 * 
 * @return Boolean value. Return `true` if the initialization goes on properly. 
 * 
 */
bool Tracking::restart(Mat first_f, Rect roi, char _state, bool hog, 
    bool fixed_window, bool multiscale, bool lab){

    _roi = roi;
    state = _state;
    if(_p_kcf != nullptr) delete _p_kcf;
    _p_kcf = new KCFTracker(hog, fixed_window, multiscale, lab);
    _p_kcf -> init(roi, first_f);

    return true;
}

/**
 * @brief Get the tracking quality score of KCF tracker.
 *
 * Encapsulation protects class data by using functions for access, 
 * preventing accidental changes.
 * 
 * @param void void.
 * 
 * @return The tracking quality score of KCF tracker.
 * 
 */
float Tracking::getScore(void) const{
    return _score;
}

/**
 * @brief Get the bounding box of a tracked object.
 *
 * Encapsulation protects class data by using functions for access, 
 * preventing accidental changes.
 * 
 * @return The bounding box of a tracked object.
 * 
 */
Rect Tracking::getROI(void) const{

    return _roi;
}


/**
 * @brief Get the appearance features of KCF tracker.
 *
 * Encapsulation protects class data by using functions for access, 
 * preventing accidental changes.
 * 
 * @param void void.
 * 
 * @return The appearance features of KCF tracker.
 * 
 */
Mat Tracking::getAppearance(void) const{
    /* Return the features KCF tracker is using. */
    return _p_kcf -> getTmpl();
}

/**
 * @brief Get the Average Peak-to-Correlation Energy (APCE) of KCF tracker.
 *
 * Encapsulation protects class data by using functions for access, 
 * preventing accidental changes.
 * 
 * @param void void.
 * 
 * @return APCE value of KCF tracker.
 * 
 */
float Tracking::getApce(void) const{
    return _current_apce_value;
}

/**
 * @brief Get the peak value of KCF response map.
 *
 * Encapsulation protects class data by using functions for access, 
 * preventing accidental changes.
 * 
 * @param void void.
 * 
 * @return The peak value of KCF response map.
 * 
 */
float Tracking::getPeak(void) const{
    return _peak_value;
}
