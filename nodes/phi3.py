#!/usr/bin/env python3
import os
import time
import rospy
import rospkg
import PyPDF2
from butia_quiz.srv import ButiaQuizComm, ButiaQuizCommResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

PACKAGE_DIR = rospkg.RosPack().get_path("butia_quiz")
PDF_FILEPATH = os.path.join(PACKAGE_DIR, "resources", "Questions.pdf")

# Load the model and tokenizer 
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
#quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="auto", torch_dtype="auto",  
    trust_remote_code=True,)

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

def prompt_treatment(question, context):
    messages = [
        {
            "role": "system", "content": "Use the following context to answer the question at the end. Use one sentence maximum and keep the answer as concise as possible, but try to include the question context on the answer. Dont leave the sentence unfinished, always finish the sentence.",
            "role": "user", 
            "content": 
                """
                    {context}
                    Question: {question}
                    Helpful Answer:
                """,
        },
]
    
    template = messages[0]['content'].format(context=context, question=question)
    #prompt = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
    return template

def extract_first_qa_pair(generated_text):
    """
    Extract the first question-answer pair from the generated text,
    ensuring the answer is complete and succinct.
    """
    lines = generated_text.splitlines()
    
    question = None
    answer_lines = []
    
    for line in lines:
        if line.strip().lower().startswith("question:"):
            if question is None:  # Start of a new question
                question = line.strip().split(":", 1)[1].strip()
        elif line.strip().lower().startswith("helpful answer:"):
            if question is not None:  # Start of a new answer
                answer_lines.append(line.strip().split(":", 1)[1].strip())
            else:
                break  # Stop processing if no new question was found
        elif question is not None and answer_lines:  # Continue collecting answer lines
            answer_lines.append(line.strip())
            # Stop if we hit a complete sentence (end with period)
            if line.endswith("."):
                break
    
    if question and answer_lines:
        answer = " ".join(answer_lines)
        return question, answer
    else:
        return "I dont know."
    
def generate_complete_response(treated_prompt, max_length=128):
    """
    Generate a succinct response with a limit on the total length.
    """
    input_ids = tokenizer(treated_prompt, return_tensors="pt").to("cuda")
    
    # Generate the response with a moderate limit
    outputs = model.generate(**input_ids, max_new_tokens=max_length, do_sample=False)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

def answer_question(req):
    question = req.question
    context = load_pdf_context(PDF_FILEPATH)
    
    rospy.logwarn("----------------------------")
    rospy.logwarn(f"Asked Question: {question}")
    rospy.logwarn("----------------------------")
 
    treated_prompt = prompt_treatment(question, context)
    
    start_time = time.time()
    
    # Generate the response
    generated_text = generate_complete_response(treated_prompt, max_length=64)  # Limit set to 64 tokens
    
    end_time = time.time()
    inference_time = end_time - start_time
    rospy.logwarn(f'Tempo de inferência: {inference_time} segundos')
    rospy.logwarn("---------------------")
    
    # Extract the first question-answer pair and stop after a complete sentence
    p_question, answer = extract_first_qa_pair(generated_text)
    
    rospy.logwarn(f"Prompted Question: {p_question}")
    rospy.logwarn("---------------------")
    rospy.logwarn(f"Answer: {answer}")
    rospy.logwarn("---------------------")
    
    response = ButiaQuizCommResponse()
    response.answer = answer
    return response

if __name__ == "__main__":
    rospy.init_node("butia_quiz_node", anonymous=False)

    # Set up the ROS service
    butia_quiz_service_param = rospy.get_param("servers/butia_quiz/service")
    rospy.Service(butia_quiz_service_param, ButiaQuizComm, answer_question)

    rospy.spin()