#!/usr/bin/env python3
import rospy
from butia_quiz.srv import ButiaQuizComm, ButiaQuizCommResponse

def answer_question(req):
    question = req.question
    # Find answer in questions file
    answer = "Yes, I entered here"
    
    response = ButiaQuizCommResponse()
    response.answer = answer
    return response

if __name__ == "__main__":
    rospy.init_node("butia_quiz_node", anonymous=False)

    butia_quiz_service_param = rospy.get_param("servers/butia_quiz/service")
    rospy.Service(butia_quiz_service_param, ButiaQuizComm, answer_question)

    rospy.spin()