#!/usr/bin/env python3
#! coding: "utf-8"
import rospy

from butia_quiz.srv import ButiaQuizComm, ButiaQuizCommResponse
from butia_quiz.answer_question import AnswerQuestion
from std_msgs.msg import Bool

def getAnswer(req):
    question = req.question

    questions_file = rospy.get_param("quiz/file")
    
    answer_question = AnswerQuestion(filepath=questions_file)
    answer = answer_question.getAnswer(question=question)

    butia_quiz_comm_response = ButiaQuizCommResponse()
    if answer == "":
        butia_quiz_comm_response.answer = "I'm afraid I don't know the answer."
        butia_quiz_comm_response.success = Bool(False)

    else:
        butia_quiz_comm_response.answer = answer
        butia_quiz_comm_response.success = Bool(True)

    return butia_quiz_comm_response

if __name__ == "__main__":
    rospy.init_node("butia_quiz", anonymous=False)
    butia_quiz_service_param = rospy.get_param("servers/butia_quiz/service", "/butia_quiz/bq/question")

    rospy.Service(butia_quiz_service_param, ButiaQuizComm, getAnswer)

    rospy.spin()
