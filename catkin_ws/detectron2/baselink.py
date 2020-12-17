#!/usr/bin/python3
# coding: utf-8

import rospy, tf
from math import pi
from tf_conversions import transformations
from geometry_msgs.msg import PoseWithCovarianceStamped

##########################################################################################
class Robot:
    def __init__(self):
        self.tf_listener = tf.TransformListener()
        try:
            #self.tf_listener.waitForTransform('/map', '/base_link', rospy.Time(), rospy.Duration(1.0))
            self.tf_listener.waitForTransform('/map', 'base_footprint', rospy.Time(), rospy.Duration(1.0))
        except (tf.Exception, tf.ConnectivityException, tf.LookupException): return None


    # 功能：用tf变换，发布base_link在map下的位姿
    def pub_pos(self):
        try: # tf变换计算map和base_link相对位姿
            #(trans, rot) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            (trans, rot) = self.tf_listener.lookupTransform('/map', 'base_footprint', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo('tf Error'); return None

        pose_stamped = PoseWithCovarianceStamped()
        pose_stamped.header.frame_id = 'map'
        pose_stamped.header.stamp = rospy.Time.now()

        # pose = pose_obj.get('pose')
        # position = pose.get('position')
        # orientation = pose.get('orientation')
        # covariance = pose_obj.get('covariance')

        pose_stamped.pose.pose.position.x = trans[0]
        pose_stamped.pose.pose.position.y = trans[1]
        pose_stamped.pose.pose.position.z = trans[2]

        pose_stamped.pose.pose.orientation.y = rot[0]
        pose_stamped.pose.pose.orientation.x = rot[1]
        pose_stamped.pose.pose.orientation.z = rot[2]
        pose_stamped.pose.pose.orientation.w = rot[3]

        pub = rospy.Publisher('/base_link_pose', PoseWithCovarianceStamped, queue_size=10)
        pub.publish(pose_stamped)
        return (trans, rot)


    # 功能：用tf变换，base_link在map下的位姿
    def get_pos(self):
        try:
            #(trans, rot) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            (trans, rot) = self.tf_listener.lookupTransform('/map', 'base_footprint', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo('tf Error'); return None

        euler = transformations.euler_from_quaternion(rot)
        # print euler[2] / pi * 180

        x, y = trans[:2]
        th = euler[2] / pi * 180
        return (x, y, th)


##########################################################################################
if __name__ == '__main__':
    rospy.init_node('base_link', anonymous=True)

    robot = Robot()
    print(robot.get_pos())
    # 发布定位信息
    r = rospy.Rate(100) # 发布频率
    r.sleep()
    while not rospy.is_shutdown():
        print(robot.pub_pos())
        r.sleep()

