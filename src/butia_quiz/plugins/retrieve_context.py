import rospy
from butia_quiz.plugins import RedisRAGInjector

class RedisRAGRetriever(RedisRAGInjector):
    def __init__(self, k):
        super().__init__()
        rospy.loginfo('Initializing RedisRAGRetriever')
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        
    def _retrieve_from_redis(self, question) :
        
        # Retrieve relevant documents
        context = self.retriever.get_relevant_documents(question)
        rospy.loginfo('Retrieved context from Redis')
        
        return context