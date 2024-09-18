#!/usr/bin/env python3
import rospy
import rospkg
import os
from butia_quiz.srv import ButiaQuizComm, ButiaQuizCommResponse
from fbot_db.srv import RedisRagInjectSrv, RedisRagRetrieverSrv
from butia_quiz.plugins import RedisRAGRetriever

from langchain_community.llms import Ollama
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document

from std_msgs.msg import Char

TEMPLATE = """
            Use the following context and only the context to answer the query at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Use one sentence maximum and keep the answer as concise as possible, but try to include the question context on the answer. 
            Dont leave the sentence unfinished, always finish the sentence.
            {context}
            Question: {query}
            Answer: 
        """

PACKAGE_DIR = rospkg.RosPack().get_path("butia_quiz")
PDF_FILEPATH = os.path.join(PACKAGE_DIR, "resources", "2024")

class ButiaQuizLocalLLM(RedisRAGRetriever):
    def __init__(self, ollama_configs) -> None:
        super().__init__(k=12)
        self.llm = Ollama(**ollama_configs)
        self.prompt = ChatPromptTemplate.from_template(TEMPLATE)
        if not rospy.get_param("context/path"):
            rospy.loginfo("No context path provided. Using default context.")
            self.context_path = PDF_FILEPATH
        else:
            rospy.loginfo("Context path provided. Using context from the path.")
            self.context_path = rospy.get_param("context/path")
        self._initRosComm()
    
    def _initRosComm(self):
        self.question_publisher_param =  rospy.get_param("~publishers/butia_quiz_question/topic","/butia_quiz/bq/question")
        self.answer_publisher_param =  rospy.get_param("~publishers/butia_quiz_answer/topic","/butia_quiz/bq/answer")
        
        self.answer_publisher = rospy.Publisher(self.answer_publisher_param, Char, queue_size=1)
        self.question_publisher = rospy.Publisher(self.question_publisher_param, Char, queue_size=1)
    
    def run(self):
        rospy.loginfo("ButiaQuizLocalLLM node started")
        if self.on_load_pdf(self.context_path):
            print("PDF loaded successfully")
        '''if not self._injectContext():
            rospy.logerr("Error injecting context into Redis.")'''
        # Set up the ROS service
        butia_quiz_service_param = rospy.get_param("servers/butia_quiz/service", "/butia_quiz/bq/question")
        rospy.Service(butia_quiz_service_param, ButiaQuizComm, self._answerQuestion)
        
        rospy.spin()
    
    def _separatePdfContext(self, text):
        print(text)
        page_context = text[0].page_content
        
        # Split the text to get only the "Questions - context" part
        context_start = page_context.find("Questions - context")
        predefined_start = page_context.find("Questions Predefined")
        
        if context_start != -1 and predefined_start != -1:
            # Extract only the "Questions - context" section
            context_text = page_context[context_start:predefined_start]
            context_text = [Document(page_content=context_text)]
            
        else:
            print("Markers not found in the PDF file. Using full text as context.")
            context_text = text  # Fallback to full text if markers are not found
        
        return context_text
    
    def _getAllContext(self):
        loader = PyPDFDirectoryLoader(self.context_path)
        docs = loader.load()
        docs = self._separatePdfContext(docs)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        
        # Extract text content from each chunk
        texts = [chunk.page_content for chunk in chunks]
        return texts
    
    def _injectContext(self):
        rospy.wait_for_service('redis_rag_inject_srv')
        try:
            redis_inject_service = rospy.ServiceProxy('redis_rag_inject_srv', RedisRagInjectSrv)
            response = redis_inject_service(self.context_path)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            response = False
        return response
        
    def _retrieveContext(self, question):
        # Call RedisRAGRetriever service to get relevant context
        rospy.wait_for_service('redis_rag_retriever_srv')
        try:
            redis_retriever_service = rospy.ServiceProxy('redis_rag_retriever_srv', RedisRagRetrieverSrv)
            retrieved_context = redis_retriever_service(question, 12)  # '12' is the value for 'k'
            
            # Convert metadata into dictionaries if they are not already
            documents = []
            for page, meta in zip(retrieved_context.page_contents, retrieved_context.metadata):
                # Check if meta is already a dictionary; if it's a string, convert it
                if isinstance(meta, str):
                    try:
                        meta_dict = eval(meta)  # Convert string to dictionary (use json.loads for safer conversion)
                    except:
                        meta_dict = {}  # Fallback to empty dict if conversion fails
                else:
                    meta_dict = meta  # Already a dictionary

                # Construct Document object
                documents.append(Document(page_content=page, metadata=meta_dict))
                
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            documents = self._getAllContext()
        return documents
    
    def publishQuestionAnswer(self):
        while not rospy.is_shutdown():
            self.question_publisher.publish(Char(self.question))
            self.answer_publisher.publish(Char(self.answer))
    
    def _answerQuestion(self, req):
        self.question = req.question
        print("Question: ", self.question)
        
        rag_chain = (
            {"context": self.retriever,  "query": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        try:
            answer = rag_chain.invoke(self.uestion)
            self.answer = answer.strip()  
        except Exception as e:
            rospy.logerr(f"Error invoking the LLM: {e}")
            self.answer = "I don't know"
        
        print("Answer: ", self.answer)
        
        
        response = ButiaQuizCommResponse()
        response.answer = self.answer
        return response




if __name__ == "__main__":
    rospy.init_node("butia_quiz_local_llm_node", anonymous=False)
    
    ollama_configs = rospy.get_param("ollama")
    plugin = ButiaQuizLocalLLM(ollama_configs)
    plugin.run()
    plugin.publishQuestionAnswer()