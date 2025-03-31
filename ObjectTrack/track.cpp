#include "funcs.hpp"
#include "detect.hpp"
#include "track.hpp"

#include <cstdio>


bool objTrack::tick(Mat& frame, vector<fdObject> fd_objs){

    // cout << "DEBUG:objTrack-tick - fd_objs.size: " << fd_objs.size() << endl;
    // cout << "DEBUG:objTrack-tick - _tcr_count: " << _tcr_count << endl;

    for(int i = 0; i < _tcr_count; ++ i){

        _p_tcrs[i].update(frame);
    }

    int i_write = 0;
    for(int i_read = 0; i_read < fd_objs.size(); ++ i_read){

        Rect fd_rect = fd_objs[i_read].resultRect();

        bool existed = false;
        for(int i = 0; i < _tcr_count; ++ i){

            existed = _p_tcrs[i].isSameObject(fd_rect);

            if(existed) {
                int state_and = _p_tcrs[i].state & TCR_RUNN;
                /* It's a sub-state of TCR_RUNN. */
                if(state_and == TCR_RUNN && _p_tcrs[i].state > TCR_RUNN ){

                   _p_tcrs[i].state --;


                }

                break;
            }
        }

        if(existed == false){
            fd_objs[i_write] = fd_objs[i_read];
            ++ i_write;
        }
    }
    fd_objs.resize(i_write);

    for(fdObject& fd_obj: fd_objs){

        _p_tcrs[_tcr_count].init(frame, fd_obj.resultRect());
        ++ _tcr_count;

        if(_tcr_count >= max_tcr){

            this -> tcrFullHandler();
        }
    }

    return true;

}

vector<Rect> objTrack::getROIs(void){

    vector<Rect> res;

    for(int i = 0; i < _tcr_count; ++ i){

        res.push_back( _p_tcrs[i].getROI());
    }

    return res;
}

bool objTrack::tcrFullHandler(void){
    /* To be done. */
    bool tcr_full_handle = false;
    _tcr_count --;


    if(tcr_full_handle == false){
        std::cerr << "ERROR: Failed to handle tcrFullHandler" << endl;
        return false;
    }
    return true;
}

int Tracking::id(void){
    return _id;
}

bool Tracking::isSameObject(const Rect& bbox){

    double iou = func::IoU(_roi, bbox);

    return (iou > _min_iou_req);
}

bool Tracking::update(Mat& frame){
    Rect bbox;
    bbox = _p_kcf -> update(frame);
    _roi = bbox;

    char title[6];
    snprintf(title, sizeof(title), "id:%02d", _id);

    cv::putText(frame, title, cv::Point(bbox.x,bbox.y-1),cv::FONT_HERSHEY_SIMPLEX,
                         0.5, cv::Scalar(0,0,255), 1, cv::LINE_AA);
    cv::rectangle(frame,bbox, cv::Scalar(0,0,255));

    return true;
}


bool Tracking::init(Mat first_f, Rect roi, bool hog, bool fixed_window,
                         bool multiscale, bool lab){

    state = TCR_RUNN_3;
    _roi = roi;
    _p_kcf = new KCFTracker(hog, fixed_window, multiscale, lab);
    _p_kcf -> init(roi,first_f);

    return true;
}

bool Tracking::restart(Mat first_f, Rect roi, bool hog, bool fixed_window,
    bool multiscale, bool lab){

    _roi = roi;
    if(_p_kcf != nullptr) delete _p_kcf;
    _p_kcf = new KCFTracker(hog, fixed_window, multiscale, lab);
    _p_kcf -> init(roi,first_f);

    return true;
}

Rect Tracking::getROI(void){

    return _roi;
}







