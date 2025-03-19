

#include "funcs.hpp"
#include <opencv2/opencv.hpp>

#include <string>
using std::string;

#include <cstdio>

#include "fd.hpp"


/* seqinfo.ini of test set. */
#define NAME "PETS09-S2L1"
#define imDir "img1"
#define frameRate 7
#define seqLength 795
#define imWidth 768
#define imHeight 576
#define imExt ".jpg"


void test_fd();

int main(void){
    
    test_fd();


    return 0;

}

void test_fd(void){
    string path = string("../") + NAME + "/" +  imDir + "/";

    for (int i = 1; i <= seqLength; ++i){

        /* Remember to spare a `char` for '\0'. */
        char file_name[10 + 1];

        snprintf(file_name,sizeof(file_name),(string("%06d") + imExt).c_str(),i);

        Mat f = cv::imread(path + file_name);

        if (f.empty()) {
            std::cerr << "ERROR:Load Frame Failed: " << file_name << endl;
            break;
        }

        Mat cur_f, pre_f, pp_f, resp;
        if (i > 3){
            cur_f = f;

            snprintf(file_name,sizeof(file_name),(string("%06d") + imExt).c_str(),i-1);
            pre_f = cv::imread(path + file_name);

            snprintf(file_name,sizeof(file_name),(string("%06d") + imExt).c_str(),i-2);
            pp_f = cv::imread(path + file_name);

            resp = fd::threeFramesDiff(cur_f, pre_f, pp_f);

            vector<Rect> objects = fd::getRects(resp);

            for(const Rect& object:objects){
                cv::rectangle(f,object,cv::Scalar(0,0,255));
            }
        }
        
        cv::imshow(string("Test Set: ") + NAME,f);

        /* Press `ESC` to quit. */
        if (cv::waitKey(1000 / frameRate) == 27){
            break;
        }
    }
}
