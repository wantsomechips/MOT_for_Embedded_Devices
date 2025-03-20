#pragma once

#ifndef _FUNCS_H_
#define _FUNCS_H_



#include <iostream>
#include <opencv2/opencv.hpp>
#include "kcftracker.hpp"

#include<vector>
using std::vector;

#include <array>
using std::array;

using std::cin, std::cout, std::endl;
using cv::Mat, cv::Rect;

/* Minmum IoU requirement. */
#define MIN_IOU_REQ (0.7)

/* Tracker States. */
/* Inited but not used yet. */
#define TCR_INIT (0x01 << 1)
/* It's correctly running. */
#define TCR_RUNN (0x01 << 2)
/* Object lost. */
#define TCR_LOST (0x01 << 3)
/* Should be removed after object lost for a long while. */
#define TCR_RMVD (0x01 << 4)

#define MAX_TCR 20

class Tracking;

namespace func{

    double IoU(const Rect& bbox_a, const Rect& bbox_b);
    bool tcrs_init(array<Tracking,MAX_TCR>& tcrs);
}



extern int tcr_count;

extern array<Tracking,MAX_TCR> tcrs;


class Tracking{

public:
    int id;
    std::unique_ptr<KCFTracker> kcf_p;

    /* 8 bit. */
    char state;

    Tracking() {}
    Tracking(int id){
        this -> id = id;
        state = TCR_INIT;
    }

    bool update(Mat& frame);
    bool init(Mat first_f, Rect roi,bool hog = true, bool fixed_window = true, 
                                    bool multiscale = true, bool lab = true);

};

#endif
