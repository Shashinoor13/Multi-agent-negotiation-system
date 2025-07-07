from agents.calendar_agent import GoogleCalendarAgent
from agents.gmail_agent import GmailAgent
from agents.search_agent import SearchAgent
from environment.negotiation_environment import NegotiationEnvironment
from strategy.simple import SimpleNegotiationStrategy


if __name__ == "__main__":

    task = "email rista shrestha about the meeting at 10 on monday her email is ristashrestha10@gmail.com"

    # Avaliable agents
    gmail_agent = GmailAgent("gmail_agent_01")
    calendar_agent = GoogleCalendarAgent("calendar_agent_01")
    search_agent= SearchAgent("calendar_agent_01")

    #Avaliable Strageties
    simple =  SimpleNegotiationStrategy()

    negotiation_environment = NegotiationEnvironment(agents=[gmail_agent,calendar_agent,search_agent],strategy=simple)
    negotiation_environment.set_task(task=task)
    negotiation_environment.negotiate()

    


    # Set up environment variables for email configuration
    # os.environ['SENDER_EMAIL'] = 'your-email@gmail.com'
    # os.environ['SENDER_PASSWORD'] = 'your-app-password'
    
    # print("🚀 AI-MAS Agent Examples")
    # print("=" * 50)
    
    # Gmail Agent Examples
    # print("\n📧 Gmail Agent Examples:")
    # print("-" * 30)
    # gmail_agent = GmailAgent("gmail_agent_01")
    
    # # Example 1: Complete email request
    # result1 = gmail_agent.run("Send an email to john@example.com with subject 'Meeting Tomorrow' and body 'Hi John, just confirming our meeting tomorrow at 2 PM. Best regards!'")
    # print("Result 1:", result1)
    
    # # Example 2: Incomplete request (now auto-completes)
    # result2 = gmail_agent.run("Send an email to sarah@example.com about the project update")
    # print("Result 2:", result2)
    
    # # Calendar Agent Examples
    # print("\n📅 Calendar Agent Examples:")
    # print("-" * 30)
    # calendar_agent = GoogleCalendarAgent("calendar_agent_01")
    
    # # Example 1: Create event
    # result3 = calendar_agent.run("Create a meeting with John tomorrow at 2 PM for project discussion")
    # print("Result 3:", result3)
    
    # # Example 2: List events
    # result4 = calendar_agent.run("Show me my upcoming events for this and upcomming month")
    # print("Result 4:", result4)
    
    # # Example 3: Incomplete request
    # result5 = calendar_agent.run("Schedule a dentist appointment")
    # print("Result 5:", result5)
    
    # # Search Agent Examples
    # print("\n🔍 Search Agent Examples:")
    # print("-" * 30)
    # search_agent = SearchAgent("search_agent_01")
    
    # # Example 1: Simple search
    # result6 = search_agent.run("latest news about artificial intelligence")
    # print("Result 6:", result6)
    
    # # Example 2: Technical search
    # result7 = search_agent.run(user_input="Python LangGraph tutorial examples")
    # print("Result 7:", result7)
    
    # # Example 3: Ambiguous query
    # result8 = search_agent.run(user_input="search for it")
    # print("Result 8:", result8)