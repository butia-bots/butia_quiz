#!/usr/bin/env python3
import rospy
import rospkg
import os
from termcolor import colored
from butia_quiz.srv import ButiaQuizComm, ButiaQuizCommResponse
#from fbot_db.srv import RedisRagInjectSrv, RedisRagRetrieverSrv
from butia_quiz.plugins import RedisRAGRetriever

from langchain_community.llms import Ollama
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
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

class ButiaQuizLocalLLM(RedisRAGRetriever):
    """Class to handle the Butia Quiz Local LLM node."""

    def __init__(self, ollama_configs) -> None:
        """Initialize the ButiaQuizLocalLLM node.

        Args:
            ollama_configs: Configuration parameters for the Ollama LLM.
        """
        super().__init__(k=12)
        self.llm = Ollama(**ollama_configs)
        self.prompt = ChatPromptTemplate.from_template(TEMPLATE)
        if not rospy.get_param("context/path"):
            rospy.loginfo("No context path provided. Using default context.")
            self.context_path = PDF_FILEPATH
        else:
            rospy.loginfo("Context path provided. Using context from the path.")
            self.context_path = rospy.get_param("context/path")
    
    def run(self):
        """Run the ButiaQuizLocalLLM node."""
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
        """Separate the context from the PDF text.

        Args:
            text: The text extracted from the PDF.

        Returns:
            The separated context text.
        """
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
        """Get all context from the PDF files.

        Returns:
            A list of text chunks extracted from the PDF files.
        """
        loader = PyPDFDirectoryLoader(self.context_path)
        docs = loader.load()
        docs = self._separatePdfContext(docs)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        
        # Extract text content from each chunk
        texts = [chunk.page_content for chunk in chunks]
        return texts
    
    def _injectContext(self):
        """Inject context into the Redis RAG service.

        Returns:
            A boolean indicating whether the context injection was successful.
        """
        rospy.wait_for_service('redis_rag_inject_srv')
        try:
            redis_inject_service = rospy.ServiceProxy('redis_rag_inject_srv', RedisRagInjectSrv)
            response = redis_inject_service(self.context_path)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            response = False
        return response
        
    def _retrieveContext(self, question):
        """Retrieve context relevant to the given question.

        Args:
            question: The question for which context is to be retrieved.

        Returns:
            A list of Document objects containing the relevant context.
        """
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
    
    def _answerQuestion(self, req):
        """Answer the given question using the LLM.

        Args:
            req: The request containing the question.

        Returns:
            The response containing the answer.
        """
        self.question = req.question
        print(colored(f"Question: {self.question}", "green"))
        
        rag_chain = (
            {"context": self.retriever,  "query": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        try:
            answer = rag_chain.invoke(self.question)
            self.answer = answer.strip()  
        except Exception as e:
            rospy.logerr(f"Error invoking the LLM: {e}")
            self.answer = "I don't know"
        
        print(colored(f"Question: {answer}", "blue"))
        
        response = ButiaQuizCommResponse()
        response.answer = self.answer
        return response


if __name__ == "__main__":
    # Initialize the ROS node
    rospy.init_node("butia_quiz_local_llm_node", anonymous=False)
    
    # Get the Ollama configurations from ROS parameters
    ollama_configs = rospy.get_param("ollama")
    
    # Create an instance of the ButiaQuizLocalLLM class and run it
    plugin = ButiaQuizLocalLLM(ollama_configs)
    plugin.run()