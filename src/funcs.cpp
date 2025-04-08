#include "funcs.hpp"
#include "detect.hpp"
#include "track.hpp"

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
        return false;
    }

    Mat frame;

    if(cap.read(frame) == false){
        throw std::runtime_error("Failed to read first frame.");
    }

    objDetect* detect = new objDetect(frame,DETEC_INTV);
    objTrack* track = new objTrack(MAX_TCR);

    vector<fdObject> fd_objs;
    while(cap.read(frame)){

        if(detect -> tick(frame)){
            fd_objs = detect -> getObjects();

            track -> addBackgrndResp( detect -> getBackgrndResp());

            track -> tick(frame, fd_objs);

            detect -> addTrackedObjs( track -> getROIs());
        

            /* Only for testing object detection. */
            for(fdObject& fd_obj: fd_objs){
                /* Blue(FD). */
                cv::rectangle(frame,fd_obj.resultRect(),cv::Scalar(255,0,0));
            }
        }
        else{
            track -> addBackgrndResp( detect -> getBackgrndResp());
            track -> tick(frame);
            detect -> addTrackedObjs( track -> getROIs());
        }

        cv::imshow(string("Test Set: ") + NAME,frame);

        cv::waitKey();
        

        /* Press `ESC` to quit. */
        if (cv::waitKey(1000 / frameRate) == 27){
            break;
        }

    }

    delete detect;
    delete track;

    return true;

}




/* --- --- --- --- --- --- --- --- ---

FUNC NAME: IoU

# Description
Calculate IoU of two Rects. The formula is modified in order to 
merge the small fragments caused by occlusion.

# Arguments
@ bbox_a:   First bounding box;
@ bbox_b:   Second bounding box;

# Returns
@ iou:      0.0 ~ 1.0，intersection over union;

--- --- --- --- --- --- --- --- --- */
float func::IoU(const Rect& bbox_a, const Rect& bbox_b){

    int inter_area = ( bbox_a & bbox_b).area();

    double inter_area_ratio = 1.0f * inter_area / std::min(bbox_a.area(), bbox_b.area());

    /* Believe that it's just a fragment caused by occlusion. */

    float iou = 1.0f * inter_area / (bbox_a.area() + bbox_b.area() - inter_area);

    /* Return 1.0 only when one bbox is included in the other. */
    if(inter_area_ratio > 0.8f && iou < MIN_IOU_REQ){

        return 1.0f;
    }
    
    return iou;
}







