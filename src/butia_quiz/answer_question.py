import rospy

from butia_quiz.nlp import NLP

class AnswerQuestion():

    def __init__(self, filepath: str):
        self.filepath = filepath

        self.nlp = NLP(self.filepath)

    def getAnswer(self, question):
        answer = ""

        question = question.lower()
        self.nlp.find_question(question)

        return answer
