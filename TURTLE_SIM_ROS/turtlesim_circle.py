#!/usr/bin/env python
#importing all the requires libraries 
import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose


flagstop =0 #flag value for when to stop the while loop for revolution
#function for  stoping
def stop():
	global flagstop
	flagstop=1
	#change the values to 0 and publish to velocity publisher
	move_cmd.linear.x = 0
	move_cmd.angular.z = 0
	move_cmd.linear.y =0
	velocity_publisher.publish(move_cmd)
	print("target reached")

#callback function for subscriber
def pose_callback(msg):
	x=msg.theta
	# condition to check the theta value with a tolerence of +/-0.07 and is checked after 0.07 sec
	if(x<=0.007 and x>=-0.007 and rospy.Time.now() > now + rospy.Duration.from_sec(2)):
		stop()#stop function call
		
		
rospy.init_node('turtle_revolve', anonymous=True) #initialize turtle revolve in init mode
velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)#initiaze the cmd_vel publisher
move_cmd = Twist() 
#initialize the velocity messages with the value to ensure circular motion
move_cmd.linear.x = 1
move_cmd.angular.z = 1
#initialize pose subscriber node
pose_subs=rospy.Subscriber("/turtle1/pose", Pose, pose_callback)
#time while stariting of rotation is initialized to now variable
now = rospy.Time.now()
#rospy rate is fixed
rate = rospy.Rate(10)
#while loop for circular motion
while (flagstop==0):
	velocity_publisher.publish(move_cmd)#publisher for cmd_vel msg
	print("Moving in a circle")
        rate.sleep()
