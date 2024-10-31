from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain

import rospy
import rospkg
import os

REFINE_PROMPT_TEMPLATE = """
                      Organize the following context delimited by triple backquotes.
                      There might be more than one theme in the context.
                      Return your response in bullet points which covers the main points and any important details.
                      Do not forget the numbers.
                      ```{context}```
                      """

class LLMContextManager():
    def __init__(self):
        ollama_configs = {
        'base_url': "http://localhost:11434",
        'model': "llama3.2:3b-instruct-q5_1",
        'temperature': 0,
        'keep_alive': 600,
        }
        self.llm = Ollama(**ollama_configs)
        self.refine_template = PromptTemplate(template=REFINE_PROMPT_TEMPLATE, input_variables=["text"])
        self.chain = create_stuff_documents_chain(self.llm, self.refine_template)
        
        rospy.loginfo('Initializing LLMContextManager')
    
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
        result = self.chain.invoke({"context": documents})
        return self.split_text_into_documents(result)
    