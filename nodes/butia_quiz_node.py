#!/usr/bin/env python3
import os
import rospy
import rospkg
import PyPDF2
from butia_quiz.srv import ButiaQuizComm, ButiaQuizCommResponse
from transformers import AutoTokenizer, AutoModelForCausalLM

PACKAGE_DIR = rospkg.RosPack().get_path("butia_quiz")
PDF_FILEPATH = os.path.join(PACKAGE_DIR, "resources", "Questions.pdf")

# Path to the locally saved model and tokenizer
LOCAL_MODEL_DIRECTORY = os.path.join(PACKAGE_DIR, "resources/models", "gemma_model")

# Load the model and tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIRECTORY)
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_DIRECTORY)

def load_pdf_context(pdf_path):
    # Load and extract text from the PDF file
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def answer_question(req):
    question = req.question
    rospy.logwarn("----------------------------")
    rospy.logwarn(f"Question: {question}")

    # Generate the input text for the model
    context = load_pdf_context(PDF_FILEPATH)
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    
    # Use the gemma model to generate a response
    input_ids = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**input_ids)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
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
