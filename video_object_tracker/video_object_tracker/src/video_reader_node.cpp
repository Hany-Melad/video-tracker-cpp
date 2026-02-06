
/**
 * @file video_reader_node.cpp
 * @brief ROS2 Node for reading and publishing video frames
 * @author Hany Melad Sadak
 * @date February 2025
 * 
 * This node reads frames from a video file and publishes them
 * as ROS2 Image messages at a configurable frame rate.
 */


#include <chrono>
#include <memory>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

/**
 * @class VideoPublisherNode
 * @brief ROS2 node that publishes video frames
 * 
 * Reads video file using OpenCV and publishes frames
 * to /video_frames topic at specified FPS.
 */
class VideoPublisherNode : public rclcpp::Node {
public:
    /**
     * @brief Constructor - initializes the video publisher
     * 
     * Opens video file and sets up timer for frame publishing.
     */
    VideoPublisherNode() : Node("video_publisher") {
        
        // ===========================================
        // DECLARE ROS2 PARAMETERS
        // ===========================================
        // Parameters can be set from launch file or command line
        
        this->declare_parameter<std::string>("video_path", "");
        this->declare_parameter<double>("fps", 30.0);
        
        // Get parameter values
        std::string video_path = this->get_parameter("video_path").as_string();
        double fps = this->get_parameter("fps").as_double();
        
        // ===========================================
        // VALIDATE VIDEO PATH
        // ===========================================
        
        if (video_path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No video path provided!");
            return;
        }
        
        // ===========================================
        // OPEN VIDEO FILE
        // ===========================================
        
        cap_.open(video_path);
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open video: %s", video_path.c_str());
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "=== VIDEO PUBLISHER STARTED ===");
        RCLCPP_INFO(this->get_logger(), "Video: %s", video_path.c_str());
        RCLCPP_INFO(this->get_logger(), "FPS: %.1f", fps);
        
        // ===========================================
        // CREATE PUBLISHER
        // ===========================================
        // Publishes to /video_frames topic
        // Queue size 10: buffers up to 10 messages
        
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/video_frames", 10);
        
        // ===========================================
        // CREATE TIMER
        // ===========================================
        // Timer triggers frame publishing at desired FPS
        // Period = 1000ms / FPS (e.g., 25 FPS = 40ms period)
        
        int period_ms = static_cast<int>(1000.0 / fps);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(period_ms),
            std::bind(&VideoPublisherNode::publishFrame, this));
    }

private:
    /**
     * @brief Timer callback - reads and publishes one frame
     * 
     * Called periodically based on FPS setting.
     * Loops video when reaching the end.
     */
    void publishFrame() {
        cv::Mat frame;
        
        // ===========================================
        // READ FRAME FROM VIDEO
        // ===========================================
        
        if (!cap_.read(frame)) {
            // End of video reached - loop back to beginning
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            
            if (!cap_.read(frame)) {
                RCLCPP_ERROR(this->get_logger(), "Cannot read frame!");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Video looped");
        }
        
        // ===========================================
        // CONVERT AND PUBLISH
        // ===========================================
        // Use cv_bridge to convert OpenCV Mat to ROS2 Image message
        
        auto msg = cv_bridge::CvImage(
            std_msgs::msg::Header(),  // Empty header (will fill below)
            "bgr8",                   // Encoding: Blue-Green-Red, 8-bit
            frame                     // OpenCV image data
        ).toImageMsg();
        
        // Set message header
        msg->header.stamp = this->now();      // Current timestamp
        msg->header.frame_id = "camera";      // Reference frame
        
        // Publish message
        image_pub_->publish(*msg);
    }
    
    // ===========================================
    // MEMBER VARIABLES
    // ===========================================
    
    cv::VideoCapture cap_;   // OpenCV video capture object
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;  // ROS2 publisher
    rclcpp::TimerBase::SharedPtr timer_;  // Timer for periodic publishing
};

/**
 * @brief Main function - ROS2 node entry point
 */
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VideoPublisherNode>());
    rclcpp::shutdown();
    return 0;
}
