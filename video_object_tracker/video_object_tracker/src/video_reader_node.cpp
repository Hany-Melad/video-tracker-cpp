#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
using namespace std::chrono_literals;

class VideoReaderNode : public rclcpp::Node
{
public:
    VideoReaderNode() : Node("video_reader_node")
    {
        this->declare_parameter<std::string>("video_path", "");
        this->declare_parameter<double>("fps", 30.0);
        this->declare_parameter<bool>("loop", true);
        
        video_path_ = this->get_parameter("video_path").as_string();
        fps_ = this->get_parameter("fps").as_double();
        loop_ = this->get_parameter("loop").as_bool();
        
        if (video_path_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No video path provided!");
            RCLCPP_ERROR(this->get_logger(), "Use: -p video_path:=/path/to/video.mp4");
            return;
        }
        
        cap_.open(video_path_);
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open video: %s", video_path_.c_str());
            return;
        }
        
        width_ = (int)cap_.get(cv::CAP_PROP_FRAME_WIDTH);
        height_ = (int)cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
        total_frames_ = (int)cap_.get(cv::CAP_PROP_FRAME_COUNT);
        
        RCLCPP_INFO(this->get_logger(), "================================");
        RCLCPP_INFO(this->get_logger(), "VIDEO READER NODE STARTED");
        RCLCPP_INFO(this->get_logger(), "Video: %s", video_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "Resolution: %dx%d", width_, height_);
        RCLCPP_INFO(this->get_logger(), "Total Frames: %d", total_frames_);
        RCLCPP_INFO(this->get_logger(), "FPS: %.1f", fps_);
        RCLCPP_INFO(this->get_logger(), "================================");
        
        pub_ = this->create_publisher<sensor_msgs::msg::Image>("video_frames", 10);
        
        double period_ms = 1000.0 / fps_;
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds((int)period_ms),
            std::bind(&VideoReaderNode::timer_callback, this));
    }

private:
    void timer_callback()
    {
        cv::Mat frame;
        if (!cap_.read(frame)) {
            if (loop_) {
                cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
                cap_.read(frame);
                RCLCPP_INFO(this->get_logger(), "Video looped");
            } else {
                RCLCPP_INFO(this->get_logger(), "Video ended");
                timer_->cancel();
                return;
            }
        }
        
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
        msg->header.stamp = this->now();
        msg->header.frame_id = "camera";
        pub_->publish(*msg);
        
        frame_count_++;
        if (frame_count_ % 100 == 0) {
            RCLCPP_INFO(this->get_logger(), "Published frame %d / %d", frame_count_, total_frames_);
        }
    }
    
    std::string video_path_;
    double fps_;
    bool loop_;
    cv::VideoCapture cap_;
    int width_ = 0, height_ = 0, total_frames_ = 0;
    int frame_count_ = 0;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VideoReaderNode>());
    rclcpp::shutdown();
    return 0;
}
