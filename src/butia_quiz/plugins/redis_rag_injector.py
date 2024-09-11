from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.redis import Redis
from langchain.docstore.document import Document

import rospy
import rospkg
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380")
class RedisRAGInjector():
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_store = Redis(
            embedding=self.embeddings, 
            index_name="quiz-context",
            redis_url= REDIS_URL
            )
    
    def _separate_pdf_context(self, text):
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
    
    def _injectToRedis(self, texts):
        documents = [Document(page_content=text) for text in texts]
        self.vector_store.add_documents(documents=documents)
        rospy.loginfo('Data pushed to Redis')
    
    def on_load_pdf(self, req):
        pdf_path = req
        pdf_path = pdf_path.split("\\")
        pkg_dir = rospkg.RosPack().get_path(pdf_path[0])
        pdf_path_new = os.path.join(pkg_dir, pdf_path[1], pdf_path[2])
        
        # Load and extract text from the PDF file
        loader = PyPDFDirectoryLoader(pdf_path_new)
        docs = loader.load()
        docs = self._separate_pdf_context(docs)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        
        # Extract text content from each chunk
        texts = [chunk.page_content for chunk in chunks]
        self._injectToRedis(texts)

        return True