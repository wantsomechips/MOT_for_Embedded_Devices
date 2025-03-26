#include "funcs.hpp"
#include "detect.hpp"


bool func::MOT(string input){

    cv::VideoCapture cap;

    /* Input is camera. */
    if (std::isdigit(input[0])) {

        int cam_index = std::stoi(input);
        cap.open(cam_index); 

    } 
    /* Input is video or image sequence. */
    else {

        cap.open(input);
    }

    if (!cap.isOpened()) {
        std::cerr << "ERRO: Failed to Open Input: " << input << std::endl;
        return;
    }

    objDetect* detect = new objDetect(DETEC_INTV);
    objTrack* track = new objTrack(MAX_TCR);

    Mat frame;

    while(cap.read(frame)){

        if(detect -> tick(frame)){
            vector<fdObject> fd_objs = detect -> getObjects();

            track -> tick(frame, fd_objs);

            /* Only for testing object detection. */
            for(fdObject& fd_obj: fd_objs){
                /* Blue(FD). */
                cv::rectangle(frame,fd_obj.getRect(),cv::Scalar(255,0,0));
            }
        }
        else{
            track -> tick(frame);
        }

        cv::imshow(string("Test Set: ") + NAME,frame);
        

        /* Press `ESC` to quit. */
        if (cv::waitKey(1000 / frameRate) == 27){
            break;
        }

    }

    delete detect;
    delete track;

    return true;

}

bool objTrack::tick(Mat& frame, vector<fdObject> fd_objs = {}){

    if(_tcr_count == 0 && fd_objs.size() == 0){
        return false;
    }

    if(_tcr_count == 0){

        for(fdObject& fd_obj: fd_objs){
            _p_tcrs[_tcr_count].init(frame, fd_obj.getRect());
            ++ _tcr_count;

            if(_tcr_count >= max_tcr){
                this -> tcrFullHandler();
            }
        }
    }
    else if(fd_objs.size() == 0 ){

        for(int i = 0; i < _tcr_count; ++ i){

            _p_tcrs[i].update(frame);
        }
    }
    else{


    }

}

bool objTrack::tcrFullHandler(void){
    /* To be done. */
    return false;
}

bool Tracking::set_id(int new_id){
    _id = new_id;
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

    state = TCR_RUNN;
    _roi = roi;
    _p_kcf = new KCFTracker(hog, fixed_window, multiscale, lab);
    _p_kcf -> init(roi,first_f);

    return true;
}

/* --- --- --- --- --- --- --- --- ---

FUNC NAME: IoU

# Description
Calculate IoU of two Rects.

# Arguments
@ bbox_a:   First bounding box;
@ bbox_b:   Second bounding box;

# Returns
@ iou:      0.0 ~ 1.0，intersection over union;

--- --- --- --- --- --- --- --- --- */
double func::IoU(const Rect& bbox_a, const Rect& bbox_b){

    int inter_area = ( bbox_a & bbox_b).area();

    double iou = 1.0 * inter_area / (bbox_a.area() + bbox_b.area() - inter_area);
    
    return iou;
}







