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

    def evaluate(self):
        pass
    
    def counter(self):
        pass

    def update_state(self):
        pass

    def generate_agent_card(self):
        pass

    def run(self):
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