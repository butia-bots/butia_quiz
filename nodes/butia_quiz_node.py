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

from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import Clarifai
from langchain.embeddings import HuggingFaceHubEmbeddings, ClarifaiEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
    
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

    original_question = question
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
    
    if min_distance['distance'] > 5:
        return {'question': original_question, 'answer': question_answering_chain({'query': question})['result']}

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
    pkg_path = rospkg.RosPack().get_path('butia_quiz')
    pdf_loader = PyPDFLoader(file_path=os.path.join(pkg_path, "resources", "docs", "Rulebook.pdf"))
    documents = pdf_loader.load()
    #embeddings = HuggingFaceHubEmbeddings()
    #embeddings = ClarifaiEmbeddings(user_id="openai", app_id="embed", model_id="text-embedding-ada")
    embeddings = OpenAIEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    docsearch = Chroma.from_documents(texts, embeddings)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    #llm = Replicate(model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3", model_kwargs={"temperature": 0.5, "max_new_tokens": 1024, "top_p": 1})
    #llm = Clarifai(user_id="openai", app_id="chat-completion", model_id="GPT-4")
    #llm = Clarifai(user_id="openai", app_id="completion", model_id="gpt-3_5-turbo-instruct")
    #llm = Clarifai(user_id="meta", app_id="Llama-2", model_id="llama2-13b-chat")
    question_answering_chain = RetrievalQA.from_chain_type(llm=llm, retriever=docsearch.as_retriever(), chain_type="stuff")
    #question_answering_chain.combine_documents_chain.llm_chain.prompt.template = "Use the following pieces of context to answer the question at the end. Remember to always answer in at most 3 phrases. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}\n\nQuestion: {question}\nHelpful Answer:"
    #print(question_answering_chain)
    #exit()
    butia_quiz_service_param = rospy.get_param("servers/butia_quiz/service")
    rospy.Service(butia_quiz_service_param, ButiaQuizComm, answer_question)

    rospy.spin()