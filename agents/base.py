class Agent:
    def __init__(self,id):
        self.id = id
        self.state = "idle"
        self.inbox=[]
        self.knowledge = {}
        self.history = {
            "offers_sent": [],
            "offers_received": [],
            "agreements": []
        }
    
    def execute(self):
        pass

    def evaluate(self, task_description: str = None):
        """
        Evaluate a task and return confidence score and other metrics using LLM.
        
        Args:
            task_description: Description of the task to evaluate
            
        Returns:
            Dict containing evaluation metrics including confidence score
        """
        if not task_description:
            return {
                'confidence': 0.5,
                'estimated_time': 'unknown',
                'requirements': [],
                'capabilities': [],
                'status': 'evaluated'
            }
        
        # Use LLM to evaluate the task
        from services.llm_service import LLMService
        
        evaluation_prompt = f"""
        You are an agent capability evaluator. Analyze the given task and determine how well this agent can handle it.
        
        Task Description: {task_description}
        
        Agent Type: {self.__class__.__name__}
        Agent ID: {self.id}
        
        Evaluate the task and provide:
        1. Confidence score (0.0 to 1.0) - how confident the agent can complete this task
        2. Estimated time to complete
        3. Required information/inputs
        4. Agent capabilities relevant to this task
        5. Any potential challenges or limitations
        
        Respond in this JSON format:
        {{
            "confidence": 0.85,
            "estimated_time": "3-5 minutes",
            "requirements": ["email_address", "subject", "content"],
            "capabilities": ["email_sending", "communication"],
            "challenges": ["requires recipient email"],
            "status": "evaluated"
        }}
        """
        
        try:
            llm = LLMService()
            model = llm.get_model()
            response = model.invoke(evaluation_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
                return evaluation
            else:
                # Fallback to default if JSON parsing fails
                return {
                    'confidence': 0.5,
                    'estimated_time': 'unknown',
                    'requirements': [],
                    'capabilities': [],
                    'status': 'evaluated'
                }
                
        except Exception as e:
            # Fallback to default if LLM evaluation fails
            return {
                'confidence': 0.5,
                'estimated_time': 'unknown',
                'requirements': [],
                'capabilities': [],
                'status': 'evaluated',
                'error': str(e)
            }
    
    def counter(self):
        pass

    def update_state(self):
        pass

    def generate_agent_card(self):
        pass

    def run(self,user_input:str,messages=[]):
        pass

    def receive(self, message: dict):
        """Receive a message from another agent or coordinator."""
        self.inbox.append(message)

    def send(self, recipient: 'Agent', message: dict):
        """Send message to another agent."""
        recipient.receive(message)
    
    def utility(self, offer: dict) -> float:
        """Return how good an offer is based on internal preferences."""
        pass