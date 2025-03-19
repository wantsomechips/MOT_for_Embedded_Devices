#pragma once

#ifndef _FRAMES_DIFFERENCE_H_
#define _FRAMES_DIFFERENCE_H_

#include "funcs.hpp"
#include <vector>

using std::vector;


/* Frame Difference threshold. */
#define FD_THRESHOLD 100
#define MIN_BBOX_SIZE 500

namespace fd{

    Mat threeFramesDiff(Mat cur_fra, Mat pre_fra, Mat pp_fra);
    vector<Rect> getRects(Mat resp);


}

#endif
