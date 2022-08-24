import json
import argparse
import requests
from html import unescape
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Levenshtein import distance

import re
import json
import string

def searchAnswer(question: str):
    question = question.replace(" ", "%20")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36'}
    r = requests.get(
        'https://www.google.com/search?q={:s}'.format(question),
        headers=headers
    )
    print(f"---Status code: {r.status_code}---")
    answer = ''
    index = r.text.find("=\"Z0LcW XcVN5d")
    if index != -1:
        answer = r.text[index:]
        answer = answer.replace('<a', '')
        answer = answer.replace('</a>', '')
        index = answer.find('>')
        answer = answer[index+1:]
        index = answer.find('<')
        answer = answer[:index]
    if answer == '':
        index = r.text.find('data-tts-answer=')
        if (index != -1):
            answer = r.text[index+1:]
            index = answer.find('"')
            answer = answer[:index]
    if answer == '':
        index = r.text.find('class="hgKElc"')
        if index != -1:
            answer = r.text[index:]
            index = answer.find('>')
            answer = answer[index+1:]
            index = answer.find('</span>')
            answer = answer[:index]
    if answer == '':
        index = r.text.find('class="MWXBS"')
        if index != -1:
            answer = r.text[index:]
            index = answer.find('>')
            answer = answer[index+1:]
            index = answer.find('</div>')
            answer = answer[:index]
    if answer != '':
        answer = answer.replace('<b>', '')
        answer = answer.replace('</b>', '')
        if '>' in answer:
            index = answer.find('>')
            answer = answer[index+1:]

    answer = unescape(answer)
    return answer

REPLACES = {
    "\'s": " is",
    "n\'t": " not",
    "\'ll": " will",
    "\'ve": " have"
}

def pre_process_question(question):
    q_a =[]
    for w in question.split():
        for t in REPLACES.keys():
            if t in w:
                w=w.replace(t, REPLACES[t])
        q_a.append(w)
            
    question = ""
    for w in q_a:
        question += w + " "
    
    return question

def createJsonFile(filepath: str):
    data = {}
    data["questions"] = []
    with open(filepath, "r") as f:
        for line in tqdm(f.readlines()):
            question = line
            question = pre_process_question(question)
            print(f"Question: {question}")

            answer = searchAnswer(question)
            print(f"Answer: {answer}")

            data["questions"].append({
                "question": question,
                "answer": answer
            })
    
    with open(filepath.replace(".txt", ".json"), "w") as outfile:
        json.dump(data, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a json file with answer by text file of questions.")
    parser.add_argument("-f", "--filepath", type=str, help="Filepath of text file with questions.", required=True)
    args = vars(parser.parse_args())

    createJsonFile(args["filepath"])