

/**
 * @file funcs.cpp
 * @brief Additional functions and main function for the MOT system.
 * @author wantSomeChips
 * @date 2025
 * 
 */

#include "funcs.hpp"
#include "detect.hpp"
#include "track.hpp"

/**
 * @brief Top-level abstract function that describes the overall system logic.
 *
 * @param input     Input for the MOT system. It could be a path to iamge sequence, video 
 *                  or a camera index. 
 *                  Default input is the test set `PETS09-S2L1`, which is an image sequence.
 * 
 * 
 * @return Boolean value. Return `true` if the MOT system exited successfully.
 * 
 */
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

            // track -> addBackgrndResp( detect -> getBackgrndResp());

            track -> tick(frame, fd_objs);

            // detect -> addTrackedObjs( track -> getROIs());
        

            /* Only for testing object detection. */
            for(fdObject& fd_obj: fd_objs){
                /* Blue(FD). */
                cv::rectangle(frame,fd_obj.resultRect(),cv::Scalar(255,0,0));
            }
        }
        else{
            // track -> addBackgrndResp( detect -> getBackgrndResp());
            track -> tick(frame);
            // detect -> addTrackedObjs( track -> getROIs());
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



/**
 * @brief Calculate Intersection over Union (IoU) of two bounding boxes. 
 *
 * @param bbox_a    First bounding box;
 * @param bbox_b    Second bounding box;
 * 
 * @return IoU value, it's in range [0.0, 1.0].
 * 
 */
float func::IoU(const Rect& bbox_a, const Rect& bbox_b){

    int inter_area = ( bbox_a & bbox_b).area();

    /* Believe that it's just a fragment caused by occlusion. */
    float iou = 1.0f * inter_area / (bbox_a.area() + bbox_b.area() - inter_area);

    /* Return 1.0 only when one bbox is included in the other. */
    // if(inter_area_ratio > 0.8f && iou < MIN_IOU_REQ){

    //     return 1.0f;
    // }
    
    return iou;
}







