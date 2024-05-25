#include "lidar_filter.h"
using namespace std;

Filter::Filter(ros::NodeHandle *nh) {
    std::string lidar_topic, filter_topic;
    nh->getParam("lidar_topic", lidar_topic);
    nh->getParam("filter_topic", filter_topic);
    nh->getParam("base_frame", base_frame);
    double range, z_limit, resolution;
    nh->getParam("range", range);
    nh->getParam("z_limit", z_limit);
    drop = false;

    pcl_pub = nh->advertise<sensor_msgs::PointCloud2>(filter_topic, 1);

    tfBuffer_p = boost::make_shared<tf2_ros::Buffer>();
    tfListener_p = boost::make_shared<tf2_ros::TransformListener>(*tfBuffer_p);
    ros::Duration(0.5).sleep();

    sor.setLeafSize(0.1, 0.1, 0.1);
    bbox.setMin(Eigen::Vector4f(-range, -range, -INFINITY, 1.0));
    bbox.setMax(Eigen::Vector4f(range, range, z_limit, 1.0));

    sub = nh->subscribe(lidar_topic, 1, &Filter::callback, this);
    ROS_INFO("PointCloud filtering started");
}

void Filter::callback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    if(!drop){
        // transform point cloud to base_frame
        sensor_msgs::PointCloud2Ptr cloud_tf = boost::make_shared<sensor_msgs::PointCloud2>();
        geometry_msgs::TransformStamped transform;
        try{
            transform = tfBuffer_p->lookupTransform(base_frame, msg->header.frame_id, msg->header.stamp);
        }
        catch (tf2::TransformException &ex) {
            ROS_WARN("%s", ex.what());
            return;
        }
        Eigen::Matrix4f mat = tf2::transformToEigen(transform).matrix().cast <float> ();
        pcl_ros::transformPointCloud(mat, *msg, *cloud_tf);
        cloud_tf->header.frame_id = base_frame;

        // transform to PCL data type
        pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;
        pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
        pcl_conversions::toPCL(*cloud_tf, *cloud);

        // apply bounding box
        pcl::PCLPointCloud2* cloud_cropped = new pcl::PCLPointCloud2;
        pcl::PCLPointCloud2ConstPtr cloudPtr2(cloud_cropped);
        bbox.setInputCloud(cloudPtr);
        bbox.filter(*cloud_cropped);

        // perform voxel grid filtering
        pcl::PCLPointCloud2 cloud_filtered;
        sor.setInputCloud(cloudPtr2);
        sor.filter(cloud_filtered);

        // convert to ROS data type
        sensor_msgs::PointCloud2 output;
        pcl_conversions::fromPCL(cloud_filtered, output);

        // publish the data
        pcl_pub.publish (output);
    }
    drop = !drop;  //reduce the frequency of the output point cloud
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar_filter", ros::init_options::AnonymousName);
    ros::NodeHandle nh;
    Filter pcl_filter = Filter(&nh);
    ros::spin();
}
