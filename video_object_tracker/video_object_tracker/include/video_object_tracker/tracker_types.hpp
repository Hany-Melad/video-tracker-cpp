#ifndef VIDEO_OBJECT_TRACKER__TRACKER_TYPES_HPP_
#define VIDEO_OBJECT_TRACKER__TRACKER_TYPES_HPP_

#include <deque>
#include <string>
#include <opencv2/opencv.hpp>

namespace tracker
{

struct Detection
{
    int class_id;
    std::string class_name;
    float confidence;
    cv::Rect2d bbox;
    cv::Point2d center;
};

struct TrackedObject
{
    int id;
    std::string class_name;
    cv::Rect2d bbox;
    cv::Point2d center;
    cv::Point2d velocity;
    std::deque<cv::Point2d> trajectory;
    int frames_missing;
    cv::Scalar color;
};

}

#endif
