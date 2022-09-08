#!/usr/bin/env python3
import rospy

from butia_quiz.srv import ButiaQuizComm

def callback(req):
    pass

if __name__ == "__main__":
    rospy.init_node("butia_quiz_node", anonymous=False)

    butia_quiz_service_param = rospy.get_param("servers/butia_quiz/service")
    rospy.Service(butia_quiz_service_param, ButiaQuizComm, handler=callback)

    rospy.spin()