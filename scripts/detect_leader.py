#!/usr/bin/env python

from __future__ import print_function
import rospy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, Header
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_py as tf2
import ros_numpy
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import DBSCAN
import genpy
import time


class Detector:
    def __init__(self):
        rospy.init_node("detector", anonymous=True)
        rospy.sleep(0.2)
        self.pos_frame = rospy.get_param("estimate_frame")
        self.base_frame = rospy.get_param("base_frame")
        self.baseline_estimate = rospy.get_param("baseline_estimate")
        self.ready_topic = rospy.get_param("ready_topic")
        self.clusters_topic = rospy.get_param("cluster_topic")
        self.clean_topic = rospy.get_param("output_topic")
        self.publish_clusters = rospy.get_param("publish_clusters")
        self.max_movement = rospy.get_param("human_max_movement")
        self.cluster_min = rospy.get_param("cluster_min_size")
        self.cluster_max = rospy.get_param("cluster_max_size")
        self.cluster_algo = DBSCAN(eps=0.17, min_samples=5)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_bro = tf2_ros.TransformBroadcaster()
        rospy.sleep(0.2)

        self.clean_listener = None
        self.ready_publisher = rospy.Publisher(self.ready_topic, Bool, queue_size=1)
        self.cluster_publisher = rospy.Publisher(
            self.clusters_topic, PointCloud2, queue_size=1
        )

        self.baseline = np.array([[np.nan], [np.nan], [np.nan]])
        self.estimate = np.array([[np.nan], [np.nan], [np.nan]])

        self.tim1 = rospy.Timer(rospy.Duration(0.1), self.get_baseline)

        self.clean_listener = rospy.Subscriber(
            self.clean_topic, PointCloud2, self.detect, queue_size=1
        )

        rospy.on_shutdown(self.shutdown_hook)

    def shutdown_hook(self):
        self.tim1.shutdown()
        if self.clean_listener is not None:
            self.clean_listener.unregister()
        rospy.sleep(0.2)

    def get_transform(self, tf_from, tf_to, out="matrix", time=None, dur=0.1):
        """returns the latest transformation between the given frames
        the result of multiplying point in frame tf_to by the output matrix is in the frame tf_from

        :param tf_from: find transform from this frame
        :param tf_to: find transform to this frame
        :param out: the return type
                    - 'matrix' - returns numpy array with the tf matrix
                    - 'tf' - returns TransformStamped
        :param time: the desired timestamp of the transform (ROS Time)
        :param dur: the timeout of the lookup (float)
        :return: as selected by out parameter or None in case of tf2 exception
                    - only ConnectivityException is logged
        """
        if time is None:
            tf_time = rospy.Time(0)
        else:
            if not isinstance(time, rospy.Time) and not isinstance(time, genpy.Time):
                raise TypeError("parameter time has to be ROS Time")
            tf_time = time

        try:
            t = self.tf_buffer.lookup_transform(
                tf_from, tf_to, tf_time, rospy.Duration(dur)
            )
        except (tf2.LookupException, tf2.ExtrapolationException):
            return None
        except tf2.ConnectivityException as ex:
            rospy.logerr(ex)
            return None

        # return the selected type
        if out == "matrix":
            return ros_numpy.numpify(t.transform)
        elif out == "tf":
            return t
        else:
            raise ValueError("argument out should be 'matrix' or 'tf'")

    def get_baseline(self, _):
        t = self.get_transform(self.base_frame, self.baseline_estimate)
        if t is not None:
            self.baseline = t[0:3, 3:4]

    def detect(self, msg):
        t0 = time.time()
        if np.any(np.isnan(self.baseline)):
            # no data yet
            return

        cloud_points = np.transpose(
            ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        )

        estimate = self.estimate
        if np.any(np.isnan(self.estimate)):
            estimate = self.baseline
        print(estimate.T)

        # cluster only points in the proximity of the last known position of the leader
        to_cluster = cloud_points[
            :,
            np.where(
                norm(cloud_points[0:2, :] - estimate[0:2, :], axis=0)
                <= 1.5 * self.max_movement
            )[0],
        ]
        if to_cluster.shape[1] == 0:
            rospy.logwarn("Visual contact with the preceding robot lost")
            # delete the last known position to reinitialize from baseline locator
            self.estimate = np.array([[np.nan], [np.nan], [np.nan]])
            return
        clustering = self.cluster_algo.fit(to_cluster.T)
        labels = clustering.labels_
        clusters = np.unique(labels).tolist()

        if self.publish_clusters:
            fields = [
                PointField("x", 0, PointField.FLOAT32, 1),
                PointField("y", 4, PointField.FLOAT32, 1),
                PointField("z", 8, PointField.FLOAT32, 1),
                PointField("intensity", 12, PointField.FLOAT32, 1),
            ]
            h = Header()
            h.stamp = rospy.Time.now()
            h.frame_id = self.base_frame

            clustered = np.concatenate((to_cluster, labels[None, :]))

            l = clustered.T.tolist()

            cloud = pc2.create_cloud(h, fields, l)
            self.cluster_publisher.publish(cloud)

        # find centroids of the clusters
        centroids = []
        for c in clusters:
            if c != -1:
                # do not include the noise
                points = []
                points = to_cluster[:, labels == c]
                if self.cluster_min <= points.shape[1] <= self.cluster_max:
                    # select only clusters with reasonable size
                    centroids.append(np.mean(points, axis=1, keepdims=True))

        # the preceding robot is the closest centroid to the previous position (too far if visual contact was lost)
        best = None
        d = None
        for c in centroids:
            d_tmp = norm(c[0:2, :] - estimate[0:2, :])
            if best is None:
                best = c
                d = d_tmp
            elif d_tmp < d:
                best = c
                d = d_tmp

        if best is not None:
            print(norm(best[0:2, :] - estimate[0:2, :]), d)
        if d is None or (d is not None and d >= self.max_movement):
            rospy.logwarn("Visual contact with the preceding robot lost")
            # delete the last known position to reinitialize from baseline locator
            self.estimate = np.array([[np.nan], [np.nan], [np.nan]])
        elif d is not None:
            self.estimate = best

            tf_stmp = TransformStamped()
            tf_stmp.header.frame_id = self.base_frame
            tf_stmp.header.stamp = msg.header.stamp
            tf_stmp.child_frame_id = self.pos_frame
            tf_stmp.transform.translation.x = self.estimate[0]
            tf_stmp.transform.translation.y = self.estimate[1]
            tf_stmp.transform.translation.z = self.estimate[2]
            tf_stmp.transform.rotation.x = 0.0
            tf_stmp.transform.rotation.y = 0.0
            tf_stmp.transform.rotation.z = 0.0
            tf_stmp.transform.rotation.w = 1.0

            self.tf_bro.sendTransform(tf_stmp)
        t1 = time.time()
        print(t1 - t0)


if __name__ == "__main__":
    det = Detector()
    rospy.spin()
