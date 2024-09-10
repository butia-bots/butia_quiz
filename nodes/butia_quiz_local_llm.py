#!/usr/bin/env python3
import rospy
from butia_quiz.srv import ButiaQuizComm, ButiaQuizCommResponse
from fbot_db.srv import RedisRagRetrieverSrv, RedisRagInjectSrv

from langchain_community.llms import Ollama
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.docstore.document import Document

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

class ButiaQuizLocalLLM:
    def __init__(self, ollama_configs) -> None:
        self.llm = Ollama(**ollama_configs)
        self.prompt = ChatPromptTemplate.from_template(TEMPLATE)
        if not rospy.get_param("context/path"):
            self.context_path = PDF_FILEPATH
        else:
            self.context_path = rospy.get_param("context/path")
    
    def run(self):
        rospy.loginfo("ButiaQuizLocalLLM node started")
        
        if not self._injectContext():
            rospy.logerr("Error injecting context into Redis.")
        # Set up the ROS service
        butia_quiz_service_param = rospy.get_param("servers/butia_quiz/service")
        rospy.Service(butia_quiz_service_param, ButiaQuizComm, self._answerQuestion)
        
        rospy.spin()
    
    def _separatePdfContext(text):
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
            context = " ".join([doc.page_content for doc in retrieved_context.documents])
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            context = self._getAllContext()
        return context
    
    def _answerQuestion(self, req):
        question = req.question
        
        context = self._retrieveContext(question)
            
        # Create the RAG chain
        rag_chain = (
            {"context": context, "query": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        answer = rag_chain.invoke(question)
        
        response = ButiaQuizCommResponse()
        response.answer = answer
        return response




if __name__ == "__main__":
    rospy.init_node("butia_quiz_local_llm_node", anonymous=False)
    
    ollama_configs = rospy.get_param("ollama")
    plugin = ButiaQuizLocalLLM(ollama_configs)
    plugin.run()