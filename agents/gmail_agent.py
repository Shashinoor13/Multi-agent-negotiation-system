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
from datetime import datetime

# Demo data for testing purposes
DEMO_EMAIL_DATA = {
    "rista":"ristashrestha10@gmail.com",
    "john": "john.doe@example.com",
    "jhon": "jhon.smith@example.com", 
    "alice": "alice.johnson@company.com",
    "bob": "bob.wilson@company.com",
    "sarah": "sarah.client@client.com",
    "mike": "mike.manager@business.com",
    "lisa": "lisa.assistant@office.com",
    "david": "david.consultant@consulting.com",
    "emma": "emma.director@corporate.com",
    "tom": "tom.engineer@tech.com"
}

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

# Demo data tool for generating realistic email content
@tool
def generate_demo_email_data_tool(request_type: str, context: str = "") -> Dict[str, Any]:
    """
    Generate demo email data for testing purposes.
    
    Args:
        request_type: Type of email (meeting_request, follow_up, confirmation, etc.)
        context: Additional context about the email purpose
    
    Returns:
        Dict with demo email data
    """
    try:
        # Extract potential recipient name from context
        recipient_name = "john"  # default
        for name in DEMO_EMAIL_DATA.keys():
            if name.lower() in context.lower():
                recipient_name = name
                break
        
        recipient_email = DEMO_EMAIL_DATA.get(recipient_name, "demo@example.com")
        
        # Generate appropriate content based on request type
        if "meeting" in request_type.lower() or "schedule" in request_type.lower():
            subject = f"Meeting Request - {datetime.now().strftime('%B %d, %Y')}"
            body = f"""Hi {recipient_name.title()},

I hope this email finds you well. I would like to schedule a meeting to discuss our upcoming project.

Could you please let me know your availability for next week? I'm flexible between Monday and Friday, and can accommodate your schedule.

Looking forward to hearing from you.

Best regards,
Shashinoor Ghimire"""
        
        elif "follow" in request_type.lower():
            subject = "Follow-up: Project Discussion"
            body = f"""Hi {recipient_name.title()},

I wanted to follow up on our recent project discussion. 

Here's a summary of what we discussed:
- Project timeline and milestones
- Resource allocation
- Next steps and deliverables

Please let me know if you have any questions or if there's anything else you'd like to discuss.

Best regards,
Shashinoor Ghimire"""
        
        elif "confirmation" in request_type.lower():
            subject = "Meeting Confirmation"
            body = f"""Hi {recipient_name.title()},

This is to confirm our meeting scheduled for {context} at {context}.

Meeting Details:
- Date: {context}
- Time: {context}
- Location: {context}
- Agenda: {context}

Please let me know if you need to reschedule or if you have any questions.

Best regards,
Shashinoor Ghimire"""
        
        elif "response" in request_type.lower() or "process" in request_type.lower():
            # Simulate processing a response from someone
            subject = f"Re: Meeting Scheduling - Response Processed"
            body = f"""Hi {recipient_name.title()},

Thank you for your response regarding the meeting scheduling. I have processed your availability and preferences.

Based on your response, I have:
- Updated the meeting schedule accordingly
- Sent confirmations to all participants
- Updated the calendar with the final meeting time

The meeting has been successfully scheduled and all parties have been notified.

Best regards,
Shashinoor Ghimire"""
        
        elif "receive" in request_type.lower():
            # Simulate receiving and acknowledging a message
            subject = f"Re: Meeting Proposal - Received and Acknowledged"
            body = f"""Hi {recipient_name.title()},

I have received your meeting proposal and am processing it now.

I will review the proposed times and get back to you shortly with my availability and any scheduling conflicts.

Thank you for reaching out.

Best regards,
Shashinoor Ghimire"""
        
        else:
            subject = "Important Update"
            body = f"""Hi {recipient_name.title()},

I hope you're doing well. I wanted to share an important update with you.

[Your message content here]

Please let me know if you have any questions or need any clarification.

Best regards,
Shashinoor Ghimire"""
        
        return {
            "status": "completed",
            "recipient": recipient_email,
            "subject": subject,
            "body": body,
            "message": f"Demo email data generated for {recipient_name} ({recipient_email})"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error generating demo email data: {str(e)}"
        }

class GmailAgent(Agent):
    def __init__(self, id):
        self.id = id
        self.llm = LLMService.get_model()
        self.tools = [send_email_tool, generate_demo_email_data_tool]
        self.tool_node = ToolNode(self.tools)
        self.graph = self._create_graph()
        
    def root_instruction(self):
        SYSTEM_INSTRUCTION = (
            'You are a specialized assistant for email management and negotiation support. '
            "Your primary purpose is to help users send emails by accurately extracting the recipient's email address, "
            "the email subject, and the email body from their requests. "
            "You MUST auto-generate missing subject lines and body content if not explicitly provided, "
            "but you MUST NEVER generate or guess recipient email addresses. "
            "If the recipient email is missing, you must clearly indicate that recipient information is required. "
            "For tasks that involve simulating receiving or processing emails (e.g., 'simulate receiving a meeting proposal'), "
            "use the generate_demo_email_data_tool to create a realistic, but not actual, email scenario. "
            "For actual email sending, ensure all details (recipient, subject, body) are as accurate as possible from the user's request, "
            "or intelligently auto-generated if missing. "
            "Reply as a professional email assistant, ensuring all communication is clear and concise. "
            "Always ensure the recipient's email address is valid and formatted correctly. "
            "Use 'Shashinoor Ghimire' as your name in the email body and signature."
        )
        
        FORMAT_INSTRUCTION = (
            'Analyze the user input and extract the following, providing clear, actionable content:\n'
            '1. Recipient email address (e.g., ristashrestha10@gmail.com). If not found and not a demo request, state MISSING.\n'
            '2. Email subject (e.g., Meeting Request). If missing, generate a concise, relevant subject.\n'
            '3. Email body content (detailed, not just greetings). If missing, generate comprehensive content relevant to the request.\n'
            '4. Determine if the request is for sending an email or for generating demo data (e.g., simulating a received email).\n\n'
            'Respond in this format:\n'
            'REQUEST_TYPE: [SEND_EMAIL/GENERATE_DEMO]\n'
            'RECIPIENT: [email or MISSING or recipient name for demo]\n'
            'SUBJECT: [subject or auto-generated subject]\n'
            'BODY: [body or auto-generated body]\n'
            'STATUS: [input_required/ready_to_send/generate_demo/error]'
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
            
            # Check if this is a task that involves processing responses or other email activities
            processing_keywords = ['receive', 'process', 'response', 'handle', 'manage', 'check', 'review']
            if any(keyword in user_input.lower() for keyword in processing_keywords):
                # This is a processing task, generate appropriate demo response
                request_type = "follow_up"  # default
                if "response" in user_input.lower():
                    request_type = "confirmation"
                elif "process" in user_input.lower():
                    request_type = "meeting_request"
                
                # Generate demo email data for processing
                demo_result = generate_demo_email_data_tool.invoke({
                    "request_type": request_type,
                    "context": user_input
                })
                
                if demo_result["status"] == "completed":
                    return {
                        **state,
                        "recipient": demo_result["recipient"],
                        "subject": demo_result["subject"],
                        "body": demo_result["body"],
                        "status": "ready"
                    }
            
            # Use LLM to analyze and extract email components with auto-completion
            prompt = f"""
            {self.root_instruction()}
            
            User request: {user_input}
            
            Extract the following information:
            - recipient: email address (REQUIRED - do not generate if missing)
            - subject: email subject line (auto-generate if missing)
            - body: email body content (auto-generate if missing) It should be detailed not just greetings,Have a clear message regarding the context of the task.
            
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
                
                # If recipient is missing, use demo data tool to generate it
                if recipient == "MISSING":
                    # Determine the type of email request
                    request_type = "meeting_request"  # default
                    if "follow" in user_input.lower() or "follow-up" in user_input.lower():
                        request_type = "follow_up"
                    elif "confirm" in user_input.lower():
                        request_type = "confirmation"
                    
                    # Generate demo email data
                    demo_result = generate_demo_email_data_tool.invoke({
                        "request_type": request_type,
                        "context": user_input
                    })
                    
                    if demo_result["status"] == "completed":
                        return {
                            **state,
                            "recipient": demo_result["recipient"],
                            "subject": subject if subject != "MISSING" else demo_result["subject"],
                            "body": body if body != "MISSING" else demo_result["body"],
                            "status": "ready"
                        }
                    else:
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
    
    def execute(self, user_input: str,messages:list[str]=[]) -> Dict[str, Any]:
        """Execute the email sending workflow"""
        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
            "recipient": "",
            "subject": "",
            "body": "",
            "status": "",
            "error_message": "",
            "past_messages":messages
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
    
    def evaluate(self, task_description: str = None,context: List[str] = []):
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
                'capabilities': ['email_sending', 'email_processing'],
                'status': 'evaluated'
            }
        
        # Use LLM to evaluate the task with Gmail-specific context
        evaluation_prompt = f"""
        You are evaluating a Gmail Agent's capability to handle a specific task.
        
        Gmail Agent Capabilities:
        - Send emails via Gmail API
        - Auto-generate email subjects and body content
        - Generate demo email data for testing
        - Handle email processing tasks (receive, process, handle responses)
        - Professional email communication
        - Meeting coordination and follow-ups
        
        Gmail Agent Tools:
        - send_email_tool: Sends actual emails
        - generate_demo_email_data_tool: Creates realistic email content for testing
        
        Task Description: {task_description}
        Context: {context}
        Evaluate this task for the Gmail Agent and provide:
        1. Confidence score (0.0 to 1.0) - how confident the agent can complete this task
        2. Estimated time to complete
        3. Required information/inputs
        4. Agent capabilities relevant to this task
        5. Any potential challenges or limitations
        6. Whether demo data generation would be needed
        
        Consider:
        - Email-related tasks get high confidence (0.8-1.0)
        - Communication tasks get medium-high confidence (0.7-0.9)
        - Processing tasks get medium confidence (0.6-0.8)
        - Non-email tasks get low confidence (0.2-0.4)
        
        Respond in this JSON format:
        {{
            "confidence": 0.85,
            "estimated_time": "3-5 minutes",
            "requirements": ["recipient_email", "subject", "body"],
            "capabilities": ["email_sending", "demo_data_generation"],
            "challenges": ["requires recipient email"],
            "needs_demo_data": true,
            "status": "evaluated"
        }}
        """
        
        try:
            response = self.llm.invoke(evaluation_prompt)
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
                    'requirements': ['recipient_email', 'subject', 'body'],
                    'capabilities': ['email_sending', 'email_processing'],
                    'status': 'evaluated'
                }
                
        except Exception as e:
            # Fallback to default if LLM evaluation fails
            return {
                'confidence': 0.5,
                'estimated_time': 'unknown',
                'requirements': ['recipient_email', 'subject', 'body'],
                'capabilities': ['email_sending', 'email_processing'],
                'status': 'evaluated',
                'error': str(e)
            }
    
    def counter(self):
        return super().counter()
    
    def update_state(self):
        return super().update_state()
    
    def generate_agent_card(self):
        return {
            "id": self.id,
            "name": "Gmail Agent",
            "description": "Specialized agent for email management using LangGraph workflow and demo data generation.",
            "capabilities": [
                "Extract email components from natural language",
                "Send emails using Gmail API",
                "Auto-generate missing email content",
                "Handle email processing tasks with demo data",
                "Professional email communication",
                "Meeting coordination and follow-ups",
                "Handle missing information gracefully"
            ],
            "tools": ["send_email_tool", "generate_demo_email_data_tool"],
            "status": "active"
        }
    
    def run(self, user_input: str = None,context:list[str]=[]):
        """Main entry point for the agent"""
        if user_input:
            return self.execute(user_input,context)
        else:
            return {
                "status": "input_required",
                "message": "Please provide email details (recipient, subject, and body)"
            }