import rospy

class AnswerQuestion():

    def __init__(self, filepath: str):
        self.filepath = filepath

        self.file = open(self.filepath, "r")

    def getAnswer(self, question):
        answer = ""

        """ Faz a busca no arquivo """

        return answer
