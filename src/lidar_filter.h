#ifndef lidar_filter_LIDAR_FILTER_H
#define lidar_filter_LIDAR_FILTER_H

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <math.h>
#include <string>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <geometry_msgs/TransformStamped.h>

#include <cstdio>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
const float bad_point = std::numeric_limits<float>::quiet_NaN();

class Filter {
public:
    boost::shared_ptr<tf2_ros::Buffer> tfBuffer_p;
    boost::shared_ptr<tf2_ros::TransformListener> tfListener_p;

    ros::Publisher pcl_pub;
    ros::Subscriber sub;

    pcl::CropBox<pcl::PCLPointCloud2> bbox;
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;

    std::string base_frame;
    bool drop;

    Filter(ros::NodeHandle *nh);
    void callback(const sensor_msgs::PointCloud2ConstPtr& msg);
};

#endif
