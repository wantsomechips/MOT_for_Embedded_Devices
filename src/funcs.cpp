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
                cv::rectangle(frame,fd_obj.resultRect(),cv::Scalar(255,0,0));
            }
        }
        else{
            track -> tick(frame);
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

    cout << "DEBUG::IoU: " << iou << endl;
    
    return iou;
}







