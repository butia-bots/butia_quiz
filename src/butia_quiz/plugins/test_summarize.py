from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.redis import Redis
from langchain.docstore.document import Document

from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter

from langchain_community.llms import Ollama

from langchain.chains.combine_documents import create_stuff_documents_chain

#import rospy
#import rospkg
import os

REFINE_PROMPT_TEMPLATE = """
                      Organize the following context delimited by triple backquotes.
                      There might be more than one theme in the context.
                      Return your response in bullet points which covers the main points and any important details.
                      Do not forget the numbers.
                      ```{context}```
                      """
              
class RedisRAGInjector():
    def __init__(self):
        #rospy.loginfo('Initializing RedisRAGInjector')
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
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
        #rospy.loginfo('Data pushed to Redis')
    
    def on_load_pdf(self):
        #pdf_path = req
        #pdf_path = pdf_path.split("\\")
        #pkg_dir = rospkg.RosPack().get_path(pdf_path[0])
        pdf_path_new = "C:\\Users\\luisf\\Documents\\Fbot\\fbot_ws\\butia_quiz\\resources\\2024"
        
        # Load and extract text from the PDF file
        loader = PyPDFDirectoryLoader(pdf_path_new)
        docs = loader.load()
        docs = self._separate_pdf_context(docs)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2024, chunk_overlap=0)
        chunks = text_splitter.split_documents(docs)
        
        # Extract text content from each chunk
        texts = [chunk.page_content for chunk in chunks]
        #self._injectToRedis(texts)
        #print("Texts before: ",texts)
        print("-----------------")
        organizer = LLMContextManager(ollama_default_configs)
        #print(docs)
        #print(texts)
        org_texts = organizer(texts)
        return True

class LLMContextManager():
    def __init__(self, ollama_configs):
        self.llm = Ollama(**ollama_configs)
        self.refine_template = PromptTemplate(template=REFINE_PROMPT_TEMPLATE, input_variables=["text"])
        self.chain = create_stuff_documents_chain(self.llm, self.refine_template)
        
    def parse_summary(self, result):
        output_text = result['output_text']
        
        # Split the text into sections
        sections = output_text.split('\n\n')
        
        # Initialize a dictionary to hold the structured data
        structured_data = {}
        
        # Iterate through the sections and add them to the dictionary
        current_section = None
        for section in sections:
            if section.startswith('**'):
                current_section = section.strip('**').strip(':')
                structured_data[current_section] = []
            elif current_section:
                structured_data[current_section].append(section.strip())
        
        # Print the structured data for better visualization
        for section, content in structured_data.items():
            print(f"{section}:")
            for item in content:
                print(f"  - {item}")
            print()
            
    def split_text_into_documents(self, text):
        # Split the text into sections based on double newlines
        sections = text.split('\n\n')
        
        # Initialize a list to hold the Document objects
        documents = []
        
        # Iterate through the sections and create Document objects
        for section in sections:
            # Remove leading and trailing whitespace
            section = section.strip()
            
            # Skip empty sections
            if not section:
                continue
            
            # Create a Document object for each section
            documents.append(Document(page_content=section))
        
        return documents
    
    def __call__(self, texts):
        documents = [Document(page_content=text) for text in texts]
        #print(documents)
        result = self.chain.invoke({"context": documents})
        print(result)
        print("-----------------")
        return print(self.split_text_into_documents(result))
    
    
if __name__ == "__main__":
    ollama_default_configs = {
        'base_url': "http://localhost:11434",
        'model': "llama3.2:1b-instruct-q5_1",
        'temperature': 0,
        'keep_alive': 600,
    }
    
    redis_injector = RedisRAGInjector()
    redis_injector.on_load_pdf()
    