
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Levenshtein import distance

import re
import json
import string

REPLACES = {
    "\'s": " is",
    "n\'t": " not",
    "\'ll": " will",
    "\'ve": " have"
}

class NLP():

    def __init__(self, question: str, filepath: str):
        # self.questions_file = open(filepath)
        # self.questions = json.load(self.questions_file)["questions"]

        self.question = question

    @staticmethod
    def _merge_question_array(question_array):
        question = ""
        for word in question_array:
            question = question + word
        return question.lower()
    
    @staticmethod
    def generate_question_json(self, files, output_file):
        data = {}
        data["questions"] = []
        for file in files:
            handler = open(file, "r")
            lines = handler.readlines()
            for line in lines:
                self.question = line
                self.filtered_sentence = self._pre_process_question()
                data["questions"].append({
                    "question": self.question,
                    "question_array": self.filtered_sentence
                })

        with open(output_file, "w") as outfile:
            json.dump(data, outfile)
            
    def _pre_process_question(self):
        regex = re.compile("(%s)" % "|".join(map(re.escape, REPLACES.keys())))
        regex.sub(lambda mo: REPLACES[mo.string[mo.start():mo.end()]], self.question)

        self.question = " ".join(word.strip(string.punctuation) for word in self.question.split())
        
        sw = set(stopwords.words("english"))
        wt = word_tokenize(self.question)
        self.filtered_sentence = [word for word in wt if not word in sw]

        return self._merge_question_array(self.filtered_sentence)

    def find_question(self):
        self.question = self._pre_process_question()
        l_distances = []
        for idx, file_question in enumerate(self.question):
            merged_file_question = self._merge_question_array(file_question["question_array"]).lower()
            l_distances.append({
                "distance": distance(merged_file_question, self.question),
                "index": idx
            })
        
        min_distance = l_distances[0]

        for i in range(1, len(l_distances)):
            if l_distances[i]["distance"] < min_distance["distance"]:
                min_distance = l_distances[i]
        
        if min_distance['distance'] > 5:
            return {'question': '', 'question_array': []}
        
        return self.questions[min_distance["index"]]
