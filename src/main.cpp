

#include "funcs.hpp"
#include <opencv2/opencv.hpp>

#include <string>
using std::string;

#include <cstdio>

#include "detect.hpp"


void testing();

int main(void){
    
    testing();


    return 0;

}

void testing(void){
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
        if (i >= 3){
            cur_f = f;

            snprintf(file_name,sizeof(file_name),(string("%06d") + imExt).c_str(),i-1);
            pre_f = cv::imread(path + file_name);

            snprintf(file_name,sizeof(file_name),(string("%06d") + imExt).c_str(),i-2);
            pp_f = cv::imread(path + file_name);

            resp = fd::threeFramesDiff(cur_f, pre_f, pp_f);

            vector<Rect> objects = fd::getRects(resp);

            /* Init KCF */
            if(i == 3){
                
                func::tcrs_init(tcrs);

                for(const Rect& object:objects){
                    if(tcr_count >= MAX_TCR){
                        cout << "WARNING: MAX_TCR Reached !" << endl;
                        break;
                    }
                    tcrs[tcr_count].init(cur_f, object);
                    ++ tcr_count;

                }
            }
            else{
                for(int i = 0; i < tcr_count; ++i){
                    tcrs[i].update(f);
                }
            }

            for(const Rect& object:objects){
                /* Blue(FD). */
                cv::rectangle(f,object,cv::Scalar(255,0,0));
            }
        }
        
        cv::imshow(string("Test Set: ") + NAME,f);

        /* Press `ESC` to quit. */
        if (cv::waitKey(1000 / frameRate) == 27){
            break;
        }
    }
}
