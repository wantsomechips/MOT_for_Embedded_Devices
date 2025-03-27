

#include "funcs.hpp"
#include <opencv2/opencv.hpp>

#include <string>
using std::string;


int main(void){
    
    string path = string("../") + NAME + "/" +  imDir + "/%06d" + imExt;
    
    func::MOT(path);


    return 0;

}
