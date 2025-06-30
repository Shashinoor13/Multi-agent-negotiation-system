from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from agents.base import Agent
from services.llm_service import LLMService
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import os

# Define the state structure for LangGraph
class EmailState(TypedDict):
    messages: Annotated[List[Dict], "Messages in the conversation"]
    recipient: str
    subject: str
    body: str
    status: str  # input_required, error, completed
    error_message: str

# Email sending tool
@tool
def send_email_tool(recipient: str, subject: str, body: str) -> Dict[str, Any]:
    """
    Send an email using SMTP.
    
    Args:
        recipient: Email address of the recipient
        subject: Subject line of the email
        body: Body content of the email
    
    Returns:
        Dict with status and message
    """
    try:
        # Email configuration - these should be set as environment variables
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        sender_email = os.getenv('SENDER_EMAIL')
        sender_password = os.getenv('SENDER_PASSWORD')
        
        if not sender_email or not sender_password:
            return {
                "status": "error",
                "message": "Email credentials not configured. Please set SENDER_EMAIL and SENDER_PASSWORD environment variables."
            }
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, recipient):
            return {
                "status": "error",
                "message": f"Invalid email address: {recipient}"
            }
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject
        
        # Add body to email
        msg.attach(MIMEText(body, 'plain'))
        
        # Gmail SMTP configuration
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Enable security
        server.login(sender_email, sender_password)
        
        # Send email
        text = msg.as_string()
        server.sendmail(sender_email, recipient, text)
        server.quit()
        
        return {
            "status": "completed",
            "message": f"Email sent successfully to {recipient}"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to send email: {str(e)}"
        }

class GmailAgent(Agent):
    def __init__(self, id):
        self.id = id
        self.llm = LLMService.get_model()
        self.tools = [send_email_tool]
        self.tool_node = ToolNode(self.tools)
        self.graph = self._create_graph()
        
    def root_instruction(self):
        SYSTEM_INSTRUCTION = (
            'You are a specialized assistant for email management and negotiation support. '
            "Your purpose is to help users send emails by extracting recipient's email address, "
            "email subject, and email body from their requests. "
            "You can auto-generate missing subject lines and body content, but NEVER generate recipient email addresses. "
            "If recipient is missing, indicate that recipient information is required. "
            "For negotiation tasks, focus on professional communication, meeting coordination, and follow-up emails."
        )
        
        FORMAT_INSTRUCTION = (
            'Analyze the user input and extract:\n'
            '1. Recipient email address (REQUIRED - do not generate)\n'
            '2. Email subject (auto-generate if missing)\n'
            '3. Email body content (auto-generate if missing)\n\n'
            'If recipient is missing, set status to "input_required".\n'
            'If subject or body are missing, auto-generate appropriate content.\n'
            'If all information is present and valid, use the send_email_tool to send the email.\n'
            'Set status to "error" if there are any errors, "ready" if ready to send.'
        )
        
        return f"{SYSTEM_INSTRUCTION}\n\n{FORMAT_INSTRUCTION}"
    
    def _create_graph(self):
        """Create the LangGraph workflow"""
        
        def analyze_request(state: EmailState) -> EmailState:
            """Analyze the user request and extract email components"""
            messages = state.get("messages", [])
            if not messages:
                return {
                    **state,
                    "status": "input_required",
                    "error_message": "No input provided"
                }
            
            user_input = messages[-1].get("content", "")
            
            # Use LLM to analyze and extract email components with auto-completion
            prompt = f"""
            {self.root_instruction()}
            
            User request: {user_input}
            
            Extract the following information:
            - recipient: email address (REQUIRED - do not generate if missing)
            - subject: email subject line (auto-generate if missing)
            - body: email body content (auto-generate if missing)
            
            Respond in this format:
            RECIPIENT: [email or MISSING]
            SUBJECT: [subject or auto-generated subject]
            BODY: [body or auto-generated body]
            STATUS: [input_required/ready/error]
            """
            
            try:
                # The LLM service will automatically complete missing fields (except recipient)
                response = self.llm.invoke(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Parse LLM response
                recipient = self._extract_field(response_text, "RECIPIENT")
                subject = self._extract_field(response_text, "SUBJECT")
                body = self._extract_field(response_text, "BODY")
                status = self._extract_field(response_text, "STATUS")
                
                # Only fail if recipient is missing (we don't auto-generate email addresses)
                if recipient == "MISSING":
                    return {
                        **state,
                        "status": "input_required",
                        "error_message": "Missing information: recipient email address (required)"
                    }
                
                # If we have a recipient, we can proceed even if subject/body were auto-generated
                return {
                    **state,
                    "recipient": recipient,
                    "subject": subject if subject != "MISSING" else "Auto-generated subject",
                    "body": body if body != "MISSING" else "Auto-generated body content",
                    "status": "ready"
                }
                
            except Exception as e:
                return {
                    **state,
                    "status": "error",
                    "error_message": f"Error analyzing request: {str(e)}"
                }
        
        def send_email_node(state: EmailState) -> EmailState:
            """Send the email using the extracted information"""
            try:
                result = send_email_tool.invoke({
                    "recipient": state["recipient"],
                    "subject": state["subject"],
                    "body": state["body"]
                })
                
                return {
                    **state,
                    "status": result["status"],
                    "error_message": result["message"] if result["status"] == "error" else ""
                }
                
            except Exception as e:
                return {
                    **state,
                    "status": "error",
                    "error_message": f"Error sending email: {str(e)}"
                }
        
        def should_send_email(state: EmailState) -> str:
            """Determine the next step based on current state"""
            status = state.get("status", "")
            if status == "ready":
                return "send_email"
            else:
                return "end"
        
        # Create the graph
        workflow = StateGraph(EmailState)
        
        # Add nodes
        workflow.add_node("analyze", analyze_request)
        workflow.add_node("send_email", send_email_node)
        
        # Add edges
        workflow.set_entry_point("analyze")
        workflow.add_conditional_edges(
            "analyze",
            should_send_email,
            {
                "send_email": "send_email",
                "end": END
            }
        )
        workflow.add_edge("send_email", END)
        
        return workflow.compile()
    
    def _extract_field(self, text: str, field: str) -> str:
        """Extract a specific field from LLM response"""
        pattern = rf"{field}:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "MISSING"
    
    def send_email(self, recipient: str, subject: str, body: str) -> Dict[str, Any]:
        """Legacy method for backward compatibility"""
        return send_email_tool(
            recipient=recipient,
            subject=subject,
            body=body
        )
    
    def execute(self, user_input: str) -> Dict[str, Any]:
        """Execute the email sending workflow"""
        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
            "recipient": "",
            "subject": "",
            "body": "",
            "status": "",
            "error_message": ""
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Format response
        if result["status"] == "completed":
            return {
                "status": "completed",
                "message": "Email sent successfully!",
                "details": {
                    "recipient": result["recipient"],
                    "subject": result["subject"]
                }
            }
        elif result["status"] == "input_required":
            return {
                "status": "input_required",
                "message": result["error_message"],
                "required_fields": self._get_missing_fields(result)
            }
        else:
            return {
                "status": "error",
                "message": result["error_message"]
            }
    
    def _get_missing_fields(self, state: EmailState) -> List[str]:
        """Get list of missing required fields"""
        missing = []
        if not state.get("recipient"):
            missing.append("recipient")
        # Note: subject and body are auto-generated, so we don't consider them missing
        return missing
    
    def evaluate(self):
        return super().evaluate()
    
    def counter(self):
        return super().counter()
    
    def update_state(self):
        return super().update_state()
    
    def generate_agent_card(self):
        return {
            "id": self.id,
            "name": "Gmail Agent",
            "description": "Specialized agent for sending emails using LangGraph workflow",
            "capabilities": [
                "Extract email components from natural language",
                "Validate email addresses",
                "Send emails via SMTP",
                "Handle missing information gracefully"
            ],
            "tools": ["send_email_tool"],
            "status": "active"
        }
    
    def run(self, user_input: str = None):
        """Main entry point for the agent"""
        if user_input:
            return self.execute(user_input)
        else:
            return {
                "status": "input_required",
                "message": "Please provide email details (recipient, subject, and body)"
            }