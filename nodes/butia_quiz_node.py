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

from langchain.document_loaders import PyPDFLoader, JSONLoader
from langchain.schema import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.llms import Clarifai
from langchain.embeddings import HuggingFaceHubEmbeddings, ClarifaiEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

PACKAGE_DIR = rospkg.RosPack().get_path("butia_quiz")
DORIS_PERSONAL_QUESTIONS_FILEPATH = os.path.join(PACKAGE_DIR, "resources/where_is_this.json")

#TODO find a better way to set threshold
MAX_COSINE_DISTANCE = 0.1

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use one sentence maximum and keep the answer as concise as possible.  
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

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

    #return {'question': question, 'answer': question_answering_chain({'query': question})['result']}

    result, cosine_distance = predefined_store.similarity_search_with_score(query=question, k=1)[0]
    if cosine_distance <= MAX_COSINE_DISTANCE:
        answer = result.metadata['answer']
        return {'question': result.page_content, 'answer': answer}
    else:
        return {'question': question, 'answer': question_answering_chain({'query': question})['result']}
    '''original_question = question
    question, _ = pre_process_question(question)
    question = question.lower()
    l_distances = []
    for index, file_question in enumerate(questions):
        merged_file_question = merge_question_array(file_question['question_array']).lower()
        #l_distances.append({'distance': distance(merged_file_question, question),'index': index})
        
    
    print([(q['question'],d['distance']) for q, d in zip(questions, l_distances)])
    min_distance = l_distances[0]

    for i in range(1, len(l_distances)):
        if l_distances[i]['distance'] < min_distance['distance']:
            min_distance = l_distances[i]
    
    if min_distance['distance'] > 5:
        return {'question': original_question, 'answer': question_answering_chain({'query': question})['result']}

    question_obj = questions[min_distance['index']]

    return question_obj'''

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
    pdf_loader = PyPDFLoader(file_path=os.path.join(pkg_path, "resources", "Questions.pdf"))
    #json_loader = JSONLoader(file_path=DORIS_PERSONAL_QUESTIONS_FILEPATH, jq_schema=".questions[]", text_content=False)
    documents = pdf_loader.load()# + json_loader.load()
    #embeddings = HuggingFaceHubEmbeddings()
    #embeddings = ClarifaiEmbeddings(user_id="openai", app_id="embed", model_id="text-embedding-ada")
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceEmbeddings()
    predefined_documents = []
    with open(DORIS_PERSONAL_QUESTIONS_FILEPATH) as json_file:
        all_questions = json.load(json_file)["questions"]
    for question in all_questions:
        predefined_documents.append(Document(page_content=question['question'], metadata=dict(answer=question['answer'])))
    predefined_store = Chroma.from_documents(predefined_documents, embeddings)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    docsearch = Chroma.from_documents(texts, embeddings)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True, callbacks=[StreamingStdOutCallbackHandler(),])
    #llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-small", task="text2text-generation", model_kwargs={"do_sample": False, "max_length": 1024})
    #llm = Replicate(model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3", model_kwargs={"temperature": 0.5, "max_new_tokens": 1024, "top_p": 1})
    #llm = Clarifai(user_id="openai", app_id="chat-completion", model_id="GPT-4")
    #llm = Clarifai(user_id="openai", app_id="completion", model_id="gpt-3_5-turbo-instruct")
    #llm = Clarifai(user_id="meta", app_id="Llama-2", model_id="llama2-13b-chat")
    question_answering_chain = RetrievalQA.from_chain_type(llm=llm, retriever=docsearch.as_retriever(), chain_type="stuff", chain_type_kwargs={'prompt': QA_CHAIN_PROMPT})
    #print(question_answering_chain)
    #exit()
    butia_quiz_service_param = rospy.get_param("servers/butia_quiz/service")
    rospy.Service(butia_quiz_service_param, ButiaQuizComm, answer_question)

    rospy.spin()