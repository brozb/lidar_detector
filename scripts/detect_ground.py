#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, Float32MultiArray, MultiArrayDimension
from grid_map_msgs.msg import GridMap, GridMapInfo

import tf2_ros
import tf2_py as tf2
import ros_numpy
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
import sensor_msgs.point_cloud2 as pc2
import time


class GroundDetector:
    def __init__(self):
        """removes ground from the LiDAR image, decreases range"""

        rospy.init_node("ground_detector", anonymous=True)
        rospy.sleep(0.2)

        self.base_frame = rospy.get_param("base_frame")
        self.filter_topic = rospy.get_param("filter_topic")
        self.ground_projection = rospy.get_param("ground_projection")
        self.z_offset = rospy.get_param("z_offset")
        self.tolerance = rospy.get_param("tolerance")
        self.range = rospy.get_param("range")
        self.z_limit = rospy.get_param("z_limit")
        self.resolution = rospy.get_param("resolution")
        self.publish_ground = rospy.get_param("publish_ground")
        self.output_topic = rospy.get_param("output_topic")
        self.offset = self.range / self.resolution
        self.n_cells = 2 * int(self.range / self.resolution)

        # self.pool = torch.nn.AvgPool2d(
        #     (3, 3), padding=1, stride=1, count_include_pad=False
        # )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(0.5)

        if self.publish_ground:
            self.gridmap_publisher = rospy.Publisher(
                "ground/gridmap", GridMap, queue_size=1
            )
        self.clean_publisher = rospy.Publisher(
            self.output_topic, PointCloud2, queue_size=1
        )
        self.lidar_listener = rospy.Subscriber(
            self.filter_topic, PointCloud2, self.process, queue_size=1
        )
        self.max = -np.inf
        self.time = 0
        self.n = 0
        self.body_height = np.nan

        self.tim = rospy.Timer(rospy.Duration(0.05), self.get_height)

        rospy.on_shutdown(self.shutdown_hook)
        rospy.loginfo("detecting ground")

    def shutdown_hook(self):
        if self.n != 0:
            rospy.loginfo(
                "avg time: %f in %d iterations | maximum: %f"
                % (self.time / self.n, self.n, self.max)
            )
        if self.lidar_listener is not None:
            self.lidar_listener.unregister()
        self.tim.shutdown()
        rospy.sleep(0.5)

    def get_height(self, _):
        """read height of the robot above ground"""
        if self.ground_projection == "":
            return
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, self.ground_projection, rospy.Time(0)
            )
        except (
            tf2.LookupException,
            tf2.ExtrapolationException,
            tf2.ConnectivityException,
        ) as ex:
            rospy.logwarn(ex)
            return
        self.body_height = trans.transform.translation.z

    def process(self, msg):
        """processes one LiDAR image

        :param msg: downsampled PointCloud2 (full scan can exceed 100000 points, that is unnecessary, downsampling allows speedup)
        """
        if self.ground_projection != "" and np.isnan(self.body_height):
            return

        t1 = time.time()

        ground = 1000 * np.ones((self.n_cells, self.n_cells))

        # try:
        #     trans = self.tf_buffer.lookup_transform(
        #         self.base_frame,
        #         msg.header.frame_id,
        #         msg.header.stamp,
        #         rospy.Duration(0.1),
        #     )
        # except (
        #     tf2.LookupException,
        #     tf2.ExtrapolationException,
        #     tf2.ConnectivityException,
        # ) as ex:
        #     rospy.logwarn(ex)
        #     return
        cloud_points = np.transpose(
            ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        )
        # cloud_points = np.concatenate(
        #     (cloud_points, np.ones((1, np.shape(cloud_points)[1]))), axis=0
        # )
        # t = ros_numpy.numpify(trans.transform)
        # points_in = np.matmul(t, cloud_points)
        points_in = cloud_points

        if cloud_points.shape != (0,):
            # cut out smaller area
            # points_in = points_in[
            #     :,
            #     np.where(
            #         np.logical_and(
            #             self.range >= np.abs(points_in[0]),
            #             np.logical_and(
            #                 self.range >= np.abs(points_in[1]),
            #                 points_in[2] <= self.z_limit,
            #             ),
            #         )
            #     )[0],
            # ]
            # get the lowest height in each cell
            xy = points_in[:2]
            xy = ((xy + self.resolution / 2) // self.resolution).astype(int)
            xy = np.clip(xy, -self.offset, self.offset - 1)
            xy = xy + np.array([[self.offset], [self.offset]])
            xy = xy.astype(int)
            np.minimum.at(ground, (xy[0], xy[1]), points_in[2])

            # set ground level in areas with no vision to "0"
            if self.ground_projection == "":
                ground[ground == 1000] = self.z_offset
            else:
                ground[ground == 1000] = self.body_height

            # smooth out the ground by averaging
            # x = torch.from_numpy(ground)
            # x = torch.unsqueeze(x, 0)
            # out = self.pool(x)
            # arr = out.numpy()
            # ground = np.squeeze(arr)

            # remove spikes by median filter, make the elevation map smooth by gaussian filter
            ground = median_filter(ground, 5)
            ground = gaussian_filter(ground, 1.0)

            # remove ground
            arr = points_in[
                :, np.where(points_in[2] - ground[xy[0], xy[1]] >= self.tolerance)[0]
            ]

            # send the results
            h = Header()
            h.stamp = msg.header.stamp
            h.frame_id = self.base_frame
            l6 = np.transpose(arr[0:3]).tolist()
            cloud = pc2.create_cloud(h, msg.fields[0:3], l6)
            self.clean_publisher.publish(cloud)

            if self.publish_ground:
                gi = GridMapInfo()
                gi.header.frame_id = self.base_frame
                gi.header.stamp = rospy.Time.now()
                gi.resolution = self.resolution
                gi.length_x = 2 * self.range
                gi.length_y = 2 * self.range
                gi.pose.orientation.w = 1.0

                dim0 = MultiArrayDimension(
                    "row_index",
                    self.n_cells,
                    self.n_cells**2,
                )
                dim1 = MultiArrayDimension(
                    "column_index",
                    self.n_cells,
                    self.n_cells,
                )
                d = Float32MultiArray()
                d.data = np.rot90(np.rot90(ground)).flatten("F").tolist()
                d.layout.dim = [dim1, dim0]

                g = GridMap()
                g.info = gi
                g.layers = ["elevation"]
                g.basic_layers = ["elevation"]
                g.data = [d]

                self.gridmap_publisher.publish(g)
        t2 = time.time()
        self.time += t2 - t1
        self.max = np.max([self.max, t2 - t1])
        self.n += 1


if __name__ == "__main__":
    det = GroundDetector()
    rospy.spin()
