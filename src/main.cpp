

/**
 * @file main.cpp
 * @brief Program entrance.
 * @author wantSomeChips
 * @date 2025
 * 
 */

#include "funcs.hpp"
#include <opencv2/opencv.hpp>

#include <string>
using std::string;


int main(int argc, char* argv[]){
    
    if(argc > 2){
        std::cerr << "ERROR: Only one argument allowed." << endl;
        return ERR_ARG_NUM;
    }

    string path;
    if(2 == argc){
        path = argv[1];
    }
    /* Default test set. */
    else if(1 == argc){
        path = string("../") + NAME + "/" +  imDir + "/%06d" + imExt;
    }
    cout<< path<<endl;
    func::MOT(path);

    return 0;
}
