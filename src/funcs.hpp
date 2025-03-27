#pragma once

#ifndef _FUNCS_H_
#define _FUNCS_H_

#include <iostream>
#include <stdexcept>
#include <opencv2/opencv.hpp>

#include<vector>
using std::vector;

#include <array>
using std::array;

#include <string>
using std::string;

using std::cin, std::cout, std::endl;
using cv::Mat, cv::Rect, cv::Point, cv::Size;

/* seqinfo.ini of test set. */
#define NAME "PETS09-S2L1"
#define imDir "img1"
#define frameRate 1
#define seqLength 795
#define imWidth 768
#define imHeight 576
#define imExt ".jpg"

/* Minmum IoU requirement. */
#define MIN_IOU_REQ (0.6)

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


namespace func{

    double IoU(const Rect& bbox_a, const Rect& bbox_b);
    bool MOT(string input);
}

#endif