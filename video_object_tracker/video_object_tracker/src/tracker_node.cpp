#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>

struct Track {
    int id;
    cv::Rect box;
    cv::Point2f center;
    cv::Scalar color;
    std::vector<cv::Point2f> trajectory;
    int lost_frames;
    int age;
};

class TrackerNode : public rclcpp::Node {
public:
    TrackerNode() : Node("tracker"), next_id_(0) {
        
        // Parameters - BIGGER values = LESS detections
        min_area_ = 500.0;      // Minimum object size
        max_area_ = 500000.0;     // Maximum object size
        
        // Background subtractor - less sensitive
        bg_subtractor_ = cv::createBackgroundSubtractorMOG2(1000, 100, false);
        bg_subtractor_->setNMixtures(3);
        
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/video_frames", 10,
            std::bind(&TrackerNode::imageCallback, this, std::placeholders::_1));
            
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/tracked_image", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/trajectory_markers", 10);
        
        RCLCPP_INFO(this->get_logger(), "=== TRACKER STARTED (IMPROVED) ===");
        RCLCPP_INFO(this->get_logger(), "Min Area: %.0f", min_area_);
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        cv::Mat display = frame.clone();
        
        // Get foreground mask
        cv::Mat fg_mask;
        bg_subtractor_->apply(frame, fg_mask, 0.001);  // Very slow learning rate
        
        // Strong noise removal
        cv::threshold(fg_mask, fg_mask, 200, 255, cv::THRESH_BINARY);
        
        // Morphological operations - remove noise
        cv::Mat kernel_small = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::Mat kernel_large = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
        
        cv::erode(fg_mask, fg_mask, kernel_small, cv::Point(-1,-1), 2);
        cv::dilate(fg_mask, fg_mask, kernel_large, cv::Point(-1,-1), 3);
        cv::erode(fg_mask, fg_mask, kernel_small, cv::Point(-1,-1), 1);
        
        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fg_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Filter detections
        std::vector<cv::Rect> detections;
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < min_area_ || area > max_area_) continue;
            
            cv::Rect box = cv::boundingRect(contour);
            
            // Filter by aspect ratio (not too thin)
            float aspect = (float)box.width / (float)box.height;
            if (aspect < 0.15 || aspect > 4.0) continue;
            
            // Filter by size
            if (box.width < 30 || box.height < 60) continue;
            
            detections.push_back(box);
        }
        
        // Update tracks
        updateTracks(detections);
        
        // Draw tracks
        for (auto& [id, track] : tracks_) {
            if (track.lost_frames > 0) continue;
            
            // Draw box
            cv::rectangle(display, track.box, track.color, 3);
            
            // Draw ID
            std::string label = "ID:" + std::to_string(track.id);
            cv::putText(display, label, 
                cv::Point(track.box.x, track.box.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, track.color, 2);
            
            // Draw trajectory
            if (track.trajectory.size() > 1) {
                for (size_t i = 1; i < track.trajectory.size(); i++) {
                    cv::line(display, track.trajectory[i-1], track.trajectory[i], track.color, 2);
                }
            }
        }
        
        // Show active track count
        int active = 0;
        for (auto& [id, track] : tracks_) {
            if (track.lost_frames == 0) active++;
        }
        
        cv::putText(display, "Objects: " + std::to_string(active), 
            cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3);
        
        // Publish
        auto out_msg = cv_bridge::CvImage(msg->header, "bgr8", display).toImageMsg();
        image_pub_->publish(*out_msg);
    }
    
    void updateTracks(const std::vector<cv::Rect>& detections) {
        // Mark all tracks as lost initially
        for (auto& [id, track] : tracks_) {
            track.lost_frames++;
        }
        
        // Match detections to existing tracks
        std::vector<bool> matched(detections.size(), false);
        
        for (auto& [id, track] : tracks_) {
            double best_iou = 0.3;  // Minimum IoU threshold
            int best_idx = -1;
            
            for (size_t i = 0; i < detections.size(); i++) {
                if (matched[i]) continue;
                
                double iou = computeIoU(track.box, detections[i]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_idx = i;
                }
            }
            
            if (best_idx >= 0) {
                matched[best_idx] = true;
                track.box = detections[best_idx];
                track.center = cv::Point2f(
                    track.box.x + track.box.width / 2.0f,
                    track.box.y + track.box.height / 2.0f);
                track.trajectory.push_back(track.center);
                if (track.trajectory.size() > 100) {
                    track.trajectory.erase(track.trajectory.begin());
                }
                track.lost_frames = 0;
                track.age++;
            }
        }
        
        // Create new tracks for unmatched detections
        for (size_t i = 0; i < detections.size(); i++) {
            if (!matched[i]) {
                Track new_track;
                new_track.id = next_id_++;
                new_track.box = detections[i];
                new_track.center = cv::Point2f(
                    new_track.box.x + new_track.box.width / 2.0f,
                    new_track.box.y + new_track.box.height / 2.0f);
                new_track.color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
                new_track.trajectory.push_back(new_track.center);
                new_track.lost_frames = 0;
                new_track.age = 0;
                tracks_[new_track.id] = new_track;
                
                RCLCPP_INFO(this->get_logger(), "NEW OBJECT ID: %d", new_track.id);
            }
        }
        
        // Remove old lost tracks
        std::vector<int> to_remove;
        for (auto& [id, track] : tracks_) {
            if (track.lost_frames > 15) {
                to_remove.push_back(id);
            }
        }
        for (int id : to_remove) {
            tracks_.erase(id);
        }
    }
    
    double computeIoU(const cv::Rect& a, const cv::Rect& b) {
        int x1 = std::max(a.x, b.x);
        int y1 = std::max(a.y, b.y);
        int x2 = std::min(a.x + a.width, b.x + b.width);
        int y2 = std::min(a.y + a.height, b.y + b.height);
        
        if (x2 <= x1 || y2 <= y1) return 0.0;
        
        double inter = (x2 - x1) * (y2 - y1);
        double area_a = a.width * a.height;
        double area_b = b.width * b.height;
        
        return inter / (area_a + area_b - inter);
    }
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    cv::Ptr<cv::BackgroundSubtractorMOG2> bg_subtractor_;
    std::map<int, Track> tracks_;
    int next_id_;
    double min_area_;
    double max_area_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrackerNode>());
    rclcpp::shutdown();
    return 0;
}
