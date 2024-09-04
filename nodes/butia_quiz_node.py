#!/usr/bin/env python3
import os
import time
#import rospy
#import rospkg
import PyPDF2
#from butia_quiz.srv import ButiaQuizComm, ButiaQuizCommResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
import torch

#PACKAGE_DIR = rospkg.RosPack().get_path("butia_quiz")
PDF_FILEPATH = "./resources/Questions.pdf"

# Path to the locally saved model and tokenizer
LOCAL_MODEL_DIRECTORY = "./gemma_model"
# Load the model and tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIRECTORY)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_DIRECTORY, device_map="auto", quantization_config=quantization_config)

def load_pdf_context(pdf_path):
    # Load and extract text from the PDF file
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    
    # Split the text to get only the "Questions - context" part
    context_start = text.find("Questions - context")
    predefined_start = text.find("Questions Predefined")
    
    if context_start != -1 and predefined_start != -1:
        # Extract only the "Questions - context" section
        context_text = text[context_start:predefined_start]
    else:
        print("Markers not found in the PDF file. Using full text as context.")
        context_text = text  # Fallback to full text if markers are not found
    
    return context_text

def prompt_treatment(question, context, max_tokens):
    template = """
    <start_of_turn>user
    Use the following context to answer the question at the end. Use one sentence maximum and keep the answer as concise as possible, but try to include the question context on the answer. Use a maximum of {max_tokens} words for the answer. Dont leave the sentence unfinished, always finish the sentence.
    {context}
    Question: {question}
    Helpful Answer:<end_of_turn>
    <start_of_turn>model
    """
    
    template = template.format(context=context, question=question, max_tokens=(max_tokens-12))
    return template

def extract_first_qa_pair(generated_text):
    """
    Extract the first question-answer pair from the generated text.
    """
    # Split the generated text into lines
    lines = generated_text.splitlines()
    
    question = None
    answer = None
    
    for line in lines:
        # Identify the question
        if line.strip().lower().startswith("question:"):
            question = line.strip().split(":", 1)[1].strip()
        # Identify the answer
        elif line.strip().lower().startswith("helpful answer:"):
            answer = line.strip().split(":", 1)[1].strip()
            break  # Stop after finding the first QA pair
    
    if question and answer:
        return f"Question: {question}\nAnswer: {answer}"
    else:
        return "I couldn't find a clear question-answer pair in the response."

def answer_question(req):
    question = req
    max_new_tokens = 64
 
    context = load_pdf_context(PDF_FILEPATH)
    treated_prompt = prompt_treatment(question, context, max_new_tokens)
    
    start_time = time.time()
    
    input_ids = tokenizer(treated_prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids, max_new_tokens=max_new_tokens)
    end_time = time.time()
    inference_time = end_time - start_time
    print(f'Tempo de inferência: {inference_time} segundos')
    
    generated_text = tokenizer.decode(outputs[0])
    
    # Extract the first question-answer pair
    qa_pair = generated_text
    
    return qa_pair

    ''' #rospy.logwarn("----------------------------")
    #rospy.logwarn(f"Question: {question}")

    # Generate the input text for the model
    context = load_pdf_context(PDF_FILEPATH)
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    
    # Use the gemma model to generate a response
    input_ids = tokenizer(input_text, return_tensors="pt")
    start_time = time.time()
    outputs = model.generate(**input_ids, max_new_tokens=256)
    end_time = time.time()
    inference_time = end_time - start_time
    print(f'Tempo de inferência: {inference_time} segundos')
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    #rospy.logwarn(f"Answer: {answer}")
    #rospy.logwarn("---------------------")

   # response = ButiaQuizCommResponse()
    #response.answer = answer
    return answer'''

if __name__ == "__main__":
    prompt = input("Prompt: ")
    answer = answer_question(prompt)
    print(answer)
    '''rospy.init_node("butia_quiz_node", anonymous=False)

    # Set up the ROS service
    butia_quiz_service_param = rospy.get_param("servers/butia_quiz/service")
    rospy.Service(butia_quiz_service_param, ButiaQuizComm, answer_question)

    rospy.spin()'''
