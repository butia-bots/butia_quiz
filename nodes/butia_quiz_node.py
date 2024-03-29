#!/usr/bin/env python3
import os
import rospy
import rospkg
from butia_quiz.srv import ButiaQuizComm, ButiaQuizCommResponse

import json
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Levenshtein import distance
    
PACKAGE_DIR = rospkg.RosPack().get_path("butia_quiz")
DORIS_PERSONAL_QUESTIONS_FILEPATH = os.path.join(PACKAGE_DIR, "resources/where_is_this.json")

def merge_question_array(question_array):
    question = ''
    for word in question_array:
        question = question + word
    return question

def pre_process_question(question):
    # removes punctuation
    question = question.replace('\'s', ' is')
    question = question.replace('n\'t', ' not')
    question = question.replace('\'ll', ' will')
    question = question.replace('\'ve', ' have')

    question = ' '.join(word.strip(string.punctuation)
                        for word in question.split())

    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(question)

    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    return merge_question_array(filtered_sentence), filtered_sentence

def find_question(question: str, questions):

    question, _ = pre_process_question(question)
    question = question.lower()
    l_distances = []
    for index, file_question in enumerate(questions):
        merged_file_question = merge_question_array(file_question['question_array']).lower()
        l_distances.append({'distance': distance(merged_file_question, question),'index': index})
    
    min_distance = l_distances[0]

    for i in range(1, len(l_distances)):
        if l_distances[i]['distance'] < min_distance['distance']:
            min_distance = l_distances[i]
    
    # if min_distance['distance'] > 5:
    #     return {'question': '', 'question_array': []}

    question_obj = questions[min_distance['index']]

    return question_obj

def answer_question(req):
    with open(DORIS_PERSONAL_QUESTIONS_FILEPATH) as json_file:
        all_questions = json.load(json_file)["questions"]

    question = req.question
    rospy.loginfo("---------------------")
    rospy.loginfo(f"Question: {question}")
    # Find answer in questions file
    question_obj = find_question(question, all_questions)
    rospy.loginfo(f"Processed question: {question_obj['question']}")
    if question_obj["question"] == "":
        answer = "I don't understand your question."
    else:
        answer = question_obj["answer"]
    rospy.loginfo(f"Answer: {answer}")
    rospy.loginfo("---------------------")

    response = ButiaQuizCommResponse()
    response.answer = answer
    return response

if __name__ == "__main__":
    rospy.init_node("butia_quiz_node", anonymous=False)

    butia_quiz_service_param = rospy.get_param("servers/butia_quiz/service")
    rospy.Service(butia_quiz_service_param, ButiaQuizComm, answer_question)

    rospy.spin()