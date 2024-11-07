import rospy
import rospkg
import os
from termcolor import colored
from butia_quiz.srv import ButiaQuizComm, ButiaQuizCommResponse

from langchain.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class ButiaFinalLocalLLM():
    """Class to handle the Butia Quiz Local LLM node."""

    def __init__(self, ollama_configs) -> None:
        """Initialize the ButiaQuizLocalLLM node.

        Args:
            ollama_configs: Configuration parameters for the Ollama LLM.
        """
        self.llm = ChatOllama(**ollama_configs)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Help the human with his request. If your answer has instructions, please guide the user through them.",
                ),
                ("human", "{query}"),
            ]
        )
    
    def run(self):
        """Run the ButiaQuizLocalLLM node."""
        rospy.loginfo("ButiaQuizLocalLLM node started")
        # Set up the ROS service
        butia_quiz_service_param = rospy.get_param("servers/butia_quiz/service", "/butia_quiz/bq/question")
        rospy.Service(butia_quiz_service_param, ButiaQuizComm, self._answerQuestion)
        
        rospy.spin()
    

    
    def _answerQuestion(self, req):
        """Answer the given question using the LLM.

        Args:
            req: The request containing the question.

        Returns:
            The response containing the answer.
        """
        self.question = req.question
        print(colored(f"Question: {self.question}", "green"))
        
        
        chain = self.prompt | self.llm

        try:
            answer = chain.invoke({"query": self.question})
            self.answer = answer.content  
        except Exception as e:
            rospy.logerr(f"Error invoking the LLM: {e}")
            self.answer = "I don't know"
        
        print(colored(f"Answer: {self.answer}", "blue"))
        
        response = ButiaQuizCommResponse()
        response.answer = self.answer
        return response

        
if __name__ == "__main__":
    # Initialize the ROS node
    rospy.init_node("butia_temp_final_node", anonymous=False)
    
    ollama_default_configs = {
        'base_url': "http://localhost:11434",
        'model': "llama3.2:3b-instruct-q5_1",
        'temperature': 0.6,
        'keep_alive': 600,
    }
    
    # Get the Ollama configurations from ROS parameters
    ollama_configs = rospy.get_param("~ollama/", ollama_default_configs)
    
    # Create an instance of the ButiaQuizLocalLLM class and run it
    plugin = ButiaFinalLocalLLM(ollama_configs)
    plugin.run()