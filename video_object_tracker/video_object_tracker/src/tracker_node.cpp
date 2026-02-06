/**
 * @file tracker_node.cpp
 * @brief ROS2 Node for real-time object detection and tracking
 * @author Hany Melad Sadak
 * @date February 2025
 * 
 * This node receives video frames, detects moving objects using
 * background subtraction, and tracks them with unique IDs.
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>

/**
 * @struct Track
 * @brief Data structure to store information about a tracked object
 */
struct Track {
    int id;                              // Unique identifier for this track
    cv::Rect box;                        // Current bounding box position
    cv::Point2f center;                  // Center point of bounding box
    cv::Scalar color;                    // Display color (randomly assigned)
    std::vector<cv::Point2f> trajectory; // History of center positions
    int lost_frames;                     // Counter: frames since last detection
    int age;                             // Counter: total frames tracked
};

/**
 * @class TrackerNode
 * @brief Main ROS2 node class for object tracking
 * 
 * Uses MOG2 background subtraction for motion detection
 * and IoU-based matching for multi-object tracking.
 */
class TrackerNode : public rclcpp::Node {
public:
    /**
     * @brief Constructor - initializes the tracker node
     * 
     * Sets up background subtractor, ROS2 publishers/subscribers,
     * and configures detection parameters.
     */
    TrackerNode() : Node("tracker"), next_id_(0) {
        
        // ===========================================
        // DETECTION PARAMETERS (TUNABLE)
        // ===========================================
        // These values were tuned through testing to detect
        // both cars (large) and humans (small)
        
        min_area_ = 5000.0;       // Minimum contour area in pixels²
                                  // Lower = detect smaller objects (humans)
                                  // Higher = filter more noise
                                  // Tested range: 500 - 50000
        
        max_area_ = 500000.0;     // Maximum contour area in pixels²
                                  // Filters out full-frame detections
        
        // ===========================================
        // BACKGROUND SUBTRACTOR SETUP
        // ===========================================
        // MOG2: Mixture of Gaussians algorithm
        // Learns background model and detects foreground (moving objects)
        
        bg_subtractor_ = cv::createBackgroundSubtractorMOG2(
            1000,   // History: number of frames for background model
                    // Higher = slower adaptation, more stable
            100,    // Variance threshold: sensitivity to changes
                    // Higher = less sensitive (reduces noise)
            false   // Shadow detection: disabled for simplicity
        );
        bg_subtractor_->setNMixtures(3);  // Number of Gaussian mixtures
        
        // ===========================================
        // ROS2 SUBSCRIBER
        // ===========================================
        // Receives video frames from video_publisher_node
        
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/video_frames",  // Topic name
            10,               // Queue size
            std::bind(&TrackerNode::imageCallback, this, std::placeholders::_1));
        
        // ===========================================
        // ROS2 PUBLISHERS
        // ===========================================
        
        // Publisher for processed frames with tracking visualization
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/tracked_image", 10);
        
        // Publisher for RViz2 trajectory markers
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/trajectory_markers", 10);
        
        RCLCPP_INFO(this->get_logger(), "=== TRACKER STARTED ===");
        RCLCPP_INFO(this->get_logger(), "Min Area: %.0f (detects humans + cars)", min_area_);
    }

private:
    /**
     * @brief Main callback function - processes each video frame
     * @param msg Incoming ROS2 Image message
     * 
     * Pipeline:
     * 1. Convert ROS message to OpenCV Mat
     * 2. Apply background subtraction
     * 3. Filter noise with morphological operations
     * 4. Find and filter contours
     * 5. Update tracks with new detections
     * 6. Draw visualization and publish results
     */
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        
        // Convert ROS2 Image message to OpenCV Mat
        cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        cv::Mat display = frame.clone();  // Copy for drawing
        
        // ===========================================
        // STEP 1: BACKGROUND SUBTRACTION
        // ===========================================
        // Creates binary mask: white = foreground, black = background
        
        cv::Mat fg_mask;
        bg_subtractor_->apply(
            frame,    // Input frame
            fg_mask,  // Output mask
            0.001     // Learning rate (very slow = stable background)
        );
        
        // ===========================================
        // STEP 2: NOISE REMOVAL
        // ===========================================
        
        // Convert to pure black/white (remove gray pixels)
        // Threshold 200: only keep high-confidence detections
        cv::threshold(fg_mask, fg_mask, 200, 255, cv::THRESH_BINARY);
        
        // Morphological kernels (ellipse shape for natural objects)
        cv::Mat kernel_small = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(5, 5));   // For fine operations
        cv::Mat kernel_large = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(15, 15)); // For filling gaps
        
        // Erosion: removes small noise pixels
        // Iterations=2: stronger noise removal
        cv::erode(fg_mask, fg_mask, kernel_small, cv::Point(-1,-1), 2);
        
        // Dilation: fills gaps in detected objects
        // Iterations=3: connects nearby regions
        cv::dilate(fg_mask, fg_mask, kernel_large, cv::Point(-1,-1), 3);
        
        // Final erosion: refines object boundaries
        cv::erode(fg_mask, fg_mask, kernel_small, cv::Point(-1,-1), 1);
        
        // ===========================================
        // STEP 3: FIND CONTOURS (OBJECT BOUNDARIES)
        // ===========================================
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(
            fg_mask,           // Input binary mask
            contours,          // Output contours
            cv::RETR_EXTERNAL, // Only outer contours (no holes)
            cv::CHAIN_APPROX_SIMPLE  // Compress contour points
        );
        
        // ===========================================
        // STEP 4: FILTER DETECTIONS
        // ===========================================
        // Apply multiple filters to remove false positives
        
        std::vector<cv::Rect> detections;
        for (const auto& contour : contours) {
            
            // FILTER 1: Area filter
            // Removes too small (noise) and too large (full frame) detections
            double area = cv::contourArea(contour);
            if (area < min_area_ || area > max_area_) continue;
            
            // Get bounding rectangle around contour
            cv::Rect box = cv::boundingRect(contour);
            
            // FILTER 2: Aspect ratio filter
            // Allows humans (tall/thin: ~0.3) and cars (wide: ~2.0)
            // Rejects very thin lines or very flat shapes
            float aspect = (float)box.width / (float)box.height;
            if (aspect < 0.15 || aspect > 4.0) continue;
            
            // FILTER 3: Minimum size filter
            // Ensures detected objects have reasonable dimensions
            // Width 30: allows thin humans
            // Height 60: allows tall humans
            if (box.width < 30 || box.height < 60) continue;
            
            // Detection passed all filters - add to list
            detections.push_back(box);
        }
        
        // ===========================================
        // STEP 5: UPDATE TRACKS
        // ===========================================
        // Match detections to existing tracks or create new ones
        
        updateTracks(detections);
        
        // ===========================================
        // STEP 6: DRAW VISUALIZATION
        // ===========================================
        
        for (auto& [id, track] : tracks_) {
            // Skip tracks that are currently lost
            if (track.lost_frames > 0) continue;
            
            // Draw bounding box
            cv::rectangle(display, track.box, track.color, 3);
            
            // Draw ID label above box
            std::string label = "ID:" + std::to_string(track.id);
            cv::putText(display, label, 
                cv::Point(track.box.x, track.box.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, track.color, 2);
            
            // Draw trajectory line (movement history)
            if (track.trajectory.size() > 1) {
                for (size_t i = 1; i < track.trajectory.size(); i++) {
                    cv::line(display, 
                        track.trajectory[i-1], 
                        track.trajectory[i], 
                        track.color, 2);
                }
            }
        }
        
        // Count and display active tracks
        int active = 0;
        for (auto& [id, track] : tracks_) {
            if (track.lost_frames == 0) active++;
        }
        
        cv::putText(display, "Objects: " + std::to_string(active), 
            cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1.2, 
            cv::Scalar(0, 255, 0), 3);
        
        // ===========================================
        // STEP 7: PUBLISH RESULTS
        // ===========================================
        
        // Publish processed image
        auto out_msg = cv_bridge::CvImage(msg->header, "bgr8", display).toImageMsg();
        image_pub_->publish(*out_msg);
        
        // Publish RViz2 markers
        publishMarkers();
    }
    
    /**
     * @brief Updates tracks with new detections using IoU matching
     * @param detections Vector of detected bounding boxes
     * 
     * Algorithm:
     * 1. Mark all existing tracks as potentially lost
     * 2. Match detections to tracks using IoU (Intersection over Union)
     * 3. Update matched tracks with new positions
     * 4. Create new tracks for unmatched detections
     * 5. Remove tracks that have been lost too long
     */
    void updateTracks(const std::vector<cv::Rect>& detections) {
        
        // Mark all tracks as lost initially
        // Will be reset if matched with a detection
        for (auto& [id, track] : tracks_) {
            track.lost_frames++;
        }
        
        // Track which detections have been matched
        std::vector<bool> matched(detections.size(), false);
        
        // ===========================================
        // MATCH DETECTIONS TO EXISTING TRACKS
        // ===========================================
        
        for (auto& [id, track] : tracks_) {
            double best_iou = 0.3;  // Minimum IoU threshold for matching
            int best_idx = -1;
            
            // Find detection with highest IoU overlap
            for (size_t i = 0; i < detections.size(); i++) {
                if (matched[i]) continue;  // Skip already matched detections
                
                double iou = computeIoU(track.box, detections[i]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_idx = i;
                }
            }
            
            // If good match found, update track
            if (best_idx >= 0) {
                matched[best_idx] = true;
                track.box = detections[best_idx];
                track.center = cv::Point2f(
                    track.box.x + track.box.width / 2.0f,
                    track.box.y + track.box.height / 2.0f);
                
                // Add to trajectory history
                track.trajectory.push_back(track.center);
                
                // Limit trajectory length (memory management)
                if (track.trajectory.size() > 100) {
                    track.trajectory.erase(track.trajectory.begin());
                }
                
                track.lost_frames = 0;  // Reset lost counter
                track.age++;
            }
        }
        
        // ===========================================
        // CREATE NEW TRACKS FOR UNMATCHED DETECTIONS
        // ===========================================
        
        for (size_t i = 0; i < detections.size(); i++) {
            if (!matched[i]) {
                Track new_track;
                new_track.id = next_id_++;  // Assign unique ID
                new_track.box = detections[i];
                new_track.center = cv::Point2f(
                    new_track.box.x + new_track.box.width / 2.0f,
                    new_track.box.y + new_track.box.height / 2.0f);
                
                // Random color for visualization
                new_track.color = cv::Scalar(
                    rand() % 255, rand() % 255, rand() % 255);
                
                new_track.trajectory.push_back(new_track.center);
                new_track.lost_frames = 0;
                new_track.age = 0;
                
                tracks_[new_track.id] = new_track;
                
                RCLCPP_INFO(this->get_logger(), "NEW OBJECT ID: %d", new_track.id);
            }
        }
        
        // ===========================================
        // REMOVE OLD LOST TRACKS
        // ===========================================
        // If track not detected for 15 frames, delete it
        
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
    
    /**
     * @brief Computes Intersection over Union between two rectangles
     * @param a First rectangle
     * @param b Second rectangle
     * @return IoU value between 0.0 and 1.0
     * 
     * IoU = Area of Intersection / Area of Union
     * Used to measure overlap between predicted and detected boxes
     */
    double computeIoU(const cv::Rect& a, const cv::Rect& b) {
        // Calculate intersection rectangle
        int x1 = std::max(a.x, b.x);
        int y1 = std::max(a.y, b.y);
        int x2 = std::min(a.x + a.width, b.x + b.width);
        int y2 = std::min(a.y + a.height, b.y + b.height);
        
        // No intersection
        if (x2 <= x1 || y2 <= y1) return 0.0;
        
        // Calculate areas
        double inter = (x2 - x1) * (y2 - y1);     // Intersection area
        double area_a = a.width * a.height;       // Area of rectangle a
        double area_b = b.width * b.height;       // Area of rectangle b
        
        // IoU formula
        return inter / (area_a + area_b - inter);
    }
    
    /**
     * @brief Publishes trajectory markers for RViz2 visualization
     * 
     * Creates LINE_STRIP markers showing movement paths
     * of all active tracked objects
     */
    void publishMarkers() {
        visualization_msgs::msg::MarkerArray marker_array;
        
        int marker_id = 0;
        for (auto& [id, track] : tracks_) {
            // Skip lost tracks or tracks with insufficient trajectory
            if (track.lost_frames > 0 || track.trajectory.size() < 2) continue;
            
            // Create line strip marker for trajectory
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = this->now();
            marker.ns = "trajectories";
            marker.id = marker_id++;
            marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.scale.x = 0.02;  // Line width
            
            // Use track's color
            marker.color.r = track.color[2] / 255.0;
            marker.color.g = track.color[1] / 255.0;
            marker.color.b = track.color[0] / 255.0;
            marker.color.a = 1.0;
            
            marker.lifetime = rclcpp::Duration::from_seconds(0.1);
            
            // Add trajectory points
            for (const auto& pt : track.trajectory) {
                geometry_msgs::msg::Point p;
                p.x = pt.x / 100.0;  // Scale to RViz coordinates
                p.y = pt.y / 100.0;
                p.z = 0;
                marker.points.push_back(p);
            }
            
            marker_array.markers.push_back(marker);
        }
        
        marker_pub_->publish(marker_array);
    }
    
    // ===========================================
    // MEMBER VARIABLES
    // ===========================================
    
    // ROS2 communication
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    
    // OpenCV background subtractor
    cv::Ptr<cv::BackgroundSubtractorMOG2> bg_subtractor_;
    
    // Tracking data
    std::map<int, Track> tracks_;  // Map of track ID to Track object
    int next_id_;                  // Counter for assigning new track IDs
    
    // Detection parameters
    double min_area_;  // Minimum contour area (filters small noise)
    double max_area_;  // Maximum contour area (filters large noise)
};

/**
 * @brief Main function - ROS2 node entry point
 */
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrackerNode>());
    rclcpp::shutdown();
    return 0;
}