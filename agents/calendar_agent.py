from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from agents.base import Agent
from services.llm_service import LLMService
import re
import os
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import json

# Define the state structure for LangGraph
class CalendarState(TypedDict):
    messages: Annotated[List[Dict], "Messages in the conversation"]
    title: str
    start_datetime: str
    end_datetime: str
    description: str
    attendees: List[str]
    location: str
    status: str  # input_required, error, completed
    error_message: str

# Calendar event creation tool
@tool
def create_calendar_event_tool(
    title: str, 
    start_datetime: str, 
    end_datetime: str, 
    description: str = "", 
    attendees: List[str] = None, 
    location: str = ""
) -> Dict[str, Any]:
    """
    Create a Google Calendar event.
    
    Args:
        title: Event title/summary
        start_datetime: Start date and time in ISO format (e.g., "2024-01-15T10:00:00")
        end_datetime: End date and time in ISO format (e.g., "2024-01-15T11:00:00")
        description: Event description (optional)
        attendees: List of attendee email addresses (optional)
        location: Event location (optional)
    
    Returns:
        Dict with status and message
    """
    try:
        # Get credentials
        creds = _get_calendar_credentials()
        if not creds:
            return {
                "status": "error",
                "message": "Google Calendar credentials not configured. Please run authentication setup."
            }
        
        # Build the Calendar API service
        service = build('calendar', 'v3', credentials=creds)
        
        # Prepare event data
        event = {
            'summary': title,
            'location': location,
            'description': description,
            'start': {
                'dateTime': start_datetime,
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': end_datetime,
                'timeZone': 'UTC',
            },
        }
        
        # Add attendees if provided
        if attendees:
            event['attendees'] = [{'email': email} for email in attendees]
        
        # Create the event
        created_event = service.events().insert(calendarId='primary', body=event).execute()
        
        return {
            "status": "completed",
            "message": f"Calendar event '{title}' created successfully",
            "event_id": created_event.get('id'),
            "event_link": created_event.get('htmlLink')
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create calendar event: {str(e)}"
        }

# Calendar event listing tool
@tool
def list_calendar_events_tool(
    start_date: str = "", 
    end_date: str = "", 
    max_results: int = 10
) -> Dict[str, Any]:
    """
    List Google Calendar events.
    
    Args:
        start_date: Start date in ISO format (optional, defaults to now)
        end_date: End date in ISO format (optional, defaults to 1 week from now)
        max_results: Maximum number of events to return
    
    Returns:
        Dict with status, message, and events list
    """
    try:
        # Get credentials
        creds = _get_calendar_credentials()
        if not creds:
            return {
                "status": "error",
                "message": "Google Calendar credentials not configured. Please run authentication setup."
            }
        
        # Build the Calendar API service
        service = build('calendar', 'v3', credentials=creds)
        
        # Set default date range if not provided
        if not start_date:
            start_date = datetime.utcnow().isoformat() + 'Z'
        if not end_date:
            end_date = (datetime.utcnow() + timedelta(days=7)).isoformat() + 'Z'
        
        # Get events
        events_result = service.events().list(
            calendarId='primary',
            timeMin=start_date,
            timeMax=end_date,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        # Format events for response
        formatted_events = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            formatted_events.append({
                'id': event.get('id'),
                'title': event.get('summary', 'No Title'),
                'start': start,
                'location': event.get('location', ''),
                'description': event.get('description', ''),
                'attendees': [attendee.get('email') for attendee in event.get('attendees', [])]
            })
        
        return {
            "status": "completed",
            "message": f"Found {len(formatted_events)} events",
            "events": formatted_events
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to list calendar events: {str(e)}"
        }

@tool
def get_others_time():
    """
    List Google Calendar events for other users
    """
    return {
        "status": "completed",
        "message": "Events for other user",
        "events": [
            {
                "id": "123",
                "title": "Dummy Title",
                "start": "2022-01-01T00:00:00Z",
                "location": "Dummy Location",
                "description": "Dummy Description",
                "attendees": ["dummy@example.com"]
            }
        ]
    }

# Demo data tool for generating realistic calendar event data
@tool
def generate_demo_calendar_data_tool(event_type: str, context: str = "") -> Dict[str, Any]:
    """
    Generate demo calendar event data for testing purposes.
    
    Args:
        event_type: Type of event (meeting, lunch, call, etc.)
        context: Additional context about the event purpose
    
    Returns:
        Dict with demo calendar event data
    """
    try:
        from datetime import datetime, timedelta
        
        # Generate realistic dates and times
        now = datetime.now()
        
        # Default to next business day at 2 PM
        next_business_day = now + timedelta(days=1)
        while next_business_day.weekday() >= 5:  # Skip weekends
            next_business_day += timedelta(days=1)
        
        start_time = next_business_day.replace(hour=14, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(hours=1)
        
        # Generate appropriate content based on event type
        if "meeting" in event_type.lower():
            title = "Team Meeting"
            description = "Regular team meeting to discuss project updates and next steps."
            attendees = ["john.doe@example.com", "alice.johnson@company.com"]
            location = "Conference Room A"
            
        elif "lunch" in event_type.lower():
            title = "Team Lunch"
            description = "Team lunch to discuss project progress and team building."
            attendees = ["bob.wilson@company.com", "sarah.client@client.com"]
            location = "Office Cafeteria"
            start_time = start_time.replace(hour=12, minute=0)
            end_time = start_time + timedelta(hours=1)
            
        elif "call" in event_type.lower() or "phone" in event_type.lower():
            title = "Client Call"
            description = "Scheduled call with client to discuss project requirements."
            attendees = ["client@example.com"]
            location = "Virtual Meeting"
            
        elif "review" in event_type.lower():
            title = "Project Review"
            description = "Review meeting to discuss project milestones and deliverables."
            attendees = ["manager@company.com", "team@company.com"]
            location = "Board Room"
            
        else:
            title = "Scheduled Event"
            description = "General scheduled event."
            attendees = ["participant@example.com"]
            location = "TBD"
        
        return {
            "status": "completed",
            "title": title,
            "start_datetime": start_time.isoformat(),
            "end_datetime": end_time.isoformat(),
            "description": description,
            "attendees": attendees,
            "location": location,
            "message": f"Demo calendar event data generated: {title} on {start_time.strftime('%B %d, %Y at %I:%M %p')}"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error generating demo calendar data: {str(e)}"
        }

def _get_calendar_credentials():
    """Get Google Calendar API credentials"""
    try:
        # Check for stored credentials
        creds_path = os.getenv('GOOGLE_CALENDAR_CREDENTIALS_PATH', 'client_secret_287440476338-njd85ng1uu56ontjt46gllm6qgtitl5d.apps.googleusercontent.com.json')
        token_path = os.getenv('GOOGLE_CALENDAR_TOKEN_PATH', 'calendar_token.json')
        
        creds = None
        
        # Load existing token
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path)
        
        # If there are no valid credentials, run the OAuth flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(creds_path):
                    print("Google Calendar credentials file not found. Please set up OAuth2 credentials.")
                    print("Download from: https://console.cloud.google.com/apis/credentials")
                    return None
                
                # Read the credentials file to get the correct redirect URI
                with open(creds_path, 'r') as f:
                    creds_data = json.load(f)
                
                # Use the first redirect URI from the credentials file
                redirect_uri = creds_data.get('installed', {}).get('redirect_uris', ['urn:ietf:wg:oauth:2.0:oob'])[0]
                
                # Define required scopes for calendar access
                required_scopes = SCOPES = [
                    'https://www.googleapis.com/auth/calendar',
                    'https://mail.google.com/',            # From your GmailAgent
                    'https://www.googleapis.com/auth/userinfo.email',
                    'https://www.googleapis.com/auth/userinfo.profile',
                    'openid' # Often implicitly included with user info scopes
                ]
                
                flow = Flow.from_client_secrets_file(
                    creds_path,
                    scopes=required_scopes,
                    redirect_uri=redirect_uri
                )
                
                # Generate authorization URL
                auth_url, _ = flow.authorization_url(
                    access_type='offline',
                    include_granted_scopes='true',
                    prompt='consent'
                )
                
                print(f'🔗 Please visit this URL to authorize the application:')
                print(f'{auth_url}')
                print('\n📋 After authorization, you will get a code. Please enter it below:')
                
                try:
                    code = input('Enter the authorization code: ').strip()
                    if not code:
                        print("❌ No authorization code provided. Authentication cancelled.")
                        return None
                    
                    flow.fetch_token(code=code)
                    creds = flow.credentials
                    
                    # Save credentials for next run
                    with open(token_path, 'w') as token:
                        token.write(creds.to_json())
                    
                    print("✅ Authentication successful! Credentials saved.")
                    
                except KeyboardInterrupt:
                    print("\n❌ Authentication cancelled by user.")
                    return None
                except Exception as e:
                    print(f"❌ Error during authentication: {str(e)}")
                    # If there's a scope mismatch, try to use the token anyway
                    if "Scope has changed" in str(e):
                        print("⚠️  Scope mismatch detected, but attempting to use token...")
                        try:
                            # Try to refresh the token to get the correct scope
                            if creds and creds.refresh_token:
                                creds.refresh(Request())
                                with open(token_path, 'w') as token:
                                    token.write(creds.to_json())
                                print("✅ Token refreshed with correct scope.")
                                return creds
                        except:
                            pass
                    return None
        
        return creds
        
    except Exception as e:
        print(f"❌ Error getting calendar credentials: {str(e)}")
        return None

class GoogleCalendarAgent(Agent):
    def __init__(self, id):
        self.id = id
        self.llm = LLMService.get_model()
        self.tools = [create_calendar_event_tool, list_calendar_events_tool,get_others_time, generate_demo_calendar_data_tool]
        self.tool_node = ToolNode(self.tools)
        self.graph = self._create_graph()
        
    def root_instruction(self):
        SYSTEM_INSTRUCTION = (
            'You are a specialized assistant for Google Calendar management and negotiation support. '
            "Your purpose is to help users create calendar events, list events, check availability, and support negotiation processes. "
            "You can also get other users time to check for avaliability"
            "You can auto-generate missing event details like descriptions, but NEVER generate dates/times without user input. "
            "Always ensure event times are properly formatted and realistic. "
            "For negotiation tasks, focus on availability checking, meeting coordination, and scheduling optimization."
        )
        
        FORMAT_INSTRUCTION = (
            'Analyze the user input and extract:\n'
            '1. Event title (REQUIRED - auto-generate if missing)\n'
            '2. Start date and time (REQUIRED - ask if missing)\n'
            '3. End date and time (auto-generate if missing, default to 1 hour after start)\n'
            '4. Description (auto-generate if missing)\n'
            '5. Attendees (optional)\n'
            '6. Location (optional)\n\n'
            'If title, start time are missing, set status to "input_required".\n'
            'If end time is missing, auto-generate it as 1 hour after start time.\n'
            'Convert all times to ISO format (YYYY-MM-DDTHH:MM:SS).\n'
            'Set status to "ready" if ready to create event, "list" if user wants to list events or check availability.'
        )
        
        return f"{SYSTEM_INSTRUCTION}\n\n{FORMAT_INSTRUCTION}"
    
    def evaluation_instruction(self):
        SYSTEM_INSTRUCTION = (
            'You are a specialized calendar evaluation assistant for negotiation support. '
            "Your purpose is to evaluate meeting offers and check user availability for specified time frames. "
            "You can also get other users time to check for avaliability"
            "You analyze proposed meeting times and determine if the user is available during those periods. "
            "You provide availability assessments and suggest alternative times if conflicts exist. "
            "For negotiation tasks, focus on finding optimal meeting slots that work for all parties."
        )
        
        FORMAT_INSTRUCTION = (
            'Analyze the offers and evaluate user availability:\n'
            '1. Extract time frames from each offer (start_datetime, end_datetime)\n'
            '2. Check user calendar for conflicts during each time frame\n'
            '3. Assess availability status for each offer:\n'
            '   - AVAILABLE: User is free during the entire time frame\n'
            '   - PARTIALLY_AVAILABLE: User has some conflicts but can attend\n'
            '   - UNAVAILABLE: User has major conflicts or is busy\n'
            '   - NEEDS_ALTERNATIVE: User cannot attend, suggest alternative times\n'
            '4. For each offer, provide:\n'
            '   - Availability status\n'
            '   - Conflict details (if any)\n'
            '   - Suggested alternative times (if needed)\n'
            '   - Priority ranking (1-5, 5 being highest priority)\n\n'
            'Respond in this format:\n'
            'OFFER_1:\n'
            'TIME_FRAME: [start_datetime] to [end_datetime]\n'
            'AVAILABILITY: [status]\n'
            'CONFLICTS: [details or NONE]\n'
            'ALTERNATIVES: [suggested times or NONE]\n'
            'PRIORITY: [1-5]\n'
            'RECOMMENDATION: [accept/decline/negotiate]\n\n'
            'Repeat for each offer. Provide overall recommendation at the end.'
        )
        
        return f"{SYSTEM_INSTRUCTION}\n\n{FORMAT_INSTRUCTION}"

    def _create_graph(self):
        """Create the LangGraph workflow"""
        
        def analyze_request(state: CalendarState) -> CalendarState:
            """Analyze the user request and extract calendar event components"""
            messages = state.get("messages", [])
            if not messages:
                return {
                    **state,
                    "status": "input_required",
                    "error_message": "No input provided"
                }
            
            user_input = messages[-1].get("content", "")
            
            # First, use LLM to determine the intent
            intent_prompt = f"""
            Analyze this calendar request and determine the primary intent:
            
            Request: {user_input}
            
            Intent options:
            - LIST_EVENTS: User wants to see calendar events, check availability, view upcoming events
            - CREATE_EVENT: User wants to create/schedule a new event or meeting
            - CHECK_AVAILABILITY: User wants to check if they're available at specific times
            
            Respond with only: LIST_EVENTS, CREATE_EVENT, or CHECK_AVAILABILITY
            """
            
            try:
                intent_response = self.llm.invoke(intent_prompt)
                intent_text = intent_response.content if hasattr(intent_response, 'content') else str(intent_response)
                intent = intent_text.strip().upper()
                
                # Handle list events and availability checks
                if "LIST_EVENTS" in intent or "CHECK_AVAILABILITY" in intent:
                    return {
                        **state,
                        "status": "list_events",
                        "error_message": ""
                    }
                
                # Handle create event requests
                if "CREATE_EVENT" in intent:
                    # Use LLM to analyze and extract calendar event components
                    prompt = f"""
                    {self.root_instruction()}
                    
                    User request: {user_input}
                    Current date and time: {datetime.now().isoformat()}
                    
                    Extract the following information for creating a calendar event:
                    - title: event title (auto-generate if missing)
                    - start_datetime: start date and time in ISO format YYYY-MM-DDTHH:MM:SS (REQUIRED)
                    - end_datetime: end date and time in ISO format (auto-generate as 1 hour after start if missing)
                    - description: event description (auto-generate if missing)
                    - attendees: comma-separated email addresses (optional)
                    - location: event location (optional)
                    
                    For negotiation tasks, focus on:
                    - Meeting scheduling and coordination
                    - Availability checking and comparison
                    - Event creation with proper details
                    
                    Respond in this format:
                    TITLE: [title or auto-generated title]
                    START_DATETIME: [ISO datetime or MISSING]
                    END_DATETIME: [ISO datetime or auto-generated]
                    DESCRIPTION: [description or auto-generated]
                    ATTENDEES: [email1,email2 or NONE]
                    LOCATION: [location or NONE]
                    STATUS: [input_required/ready/error]
                    """
                    
                    response = self.llm.invoke(prompt)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    
                    # Parse LLM response
                    title = self._extract_field(response_text, "TITLE")
                    start_datetime = self._extract_field(response_text, "START_DATETIME")
                    end_datetime = self._extract_field(response_text, "END_DATETIME")
                    description = self._extract_field(response_text, "DESCRIPTION")
                    attendees_str = self._extract_field(response_text, "ATTENDEES")
                    location = self._extract_field(response_text, "LOCATION")
                    status = self._extract_field(response_text, "STATUS")
                    
                    # Check required fields
                    missing_fields = []
                    if title == "MISSING":
                        missing_fields.append("event title")
                    if start_datetime == "MISSING":
                        missing_fields.append("start date and time")
                    
                    # If we have missing fields, use demo data tool to generate them
                    if missing_fields:
                        # Determine the type of event
                        event_type = "meeting"  # default
                        if "lunch" in user_input.lower():
                            event_type = "lunch"
                        elif "call" in user_input.lower() or "phone" in user_input.lower():
                            event_type = "call"
                        elif "review" in user_input.lower():
                            event_type = "review"
                        
                        # Generate demo calendar data
                        demo_result = generate_demo_calendar_data_tool.invoke({
                            "event_type": event_type,
                            "context": user_input
                        })
                        
                        if demo_result["status"] == "completed":
                            # Parse attendees properly
                            demo_attendees = demo_result.get("attendees", [])
                            if isinstance(demo_attendees, str):
                                demo_attendees = [email.strip() for email in demo_attendees.split(',') if email.strip()]
                            
                            # Parse attendees from LLM response
                            llm_attendees = []
                            if attendees_str and attendees_str != "NONE":
                                llm_attendees = [email.strip() for email in attendees_str.split(',') if email.strip()]
                            
                            return {
                                **state,
                                "title": title if title != "MISSING" else demo_result["title"],
                                "start_datetime": start_datetime if start_datetime != "MISSING" else demo_result["start_datetime"],
                                "end_datetime": end_datetime if end_datetime != "MISSING" else demo_result["end_datetime"],
                                "description": description if description != "MISSING" else demo_result["description"],
                                "attendees": llm_attendees if llm_attendees else demo_attendees,
                                "location": location if location != "NONE" else demo_result["location"],
                                "status": "ready"
                            }
                        else:
                            return {
                                **state,
                                "status": "input_required",
                                "error_message": f"Missing information: {', '.join(missing_fields)}"
                            }
                    
                    # Parse attendees
                    attendees = []
                    if attendees_str and attendees_str != "NONE":
                        attendees = [email.strip() for email in attendees_str.split(',') if email.strip()]
                    
                    # Auto-generate end time if missing
                    if end_datetime == "MISSING" and start_datetime != "MISSING":
                        try:
                            start_dt = datetime.fromisoformat(start_datetime.replace('Z', ''))
                            end_dt = start_dt + timedelta(hours=1)
                            end_datetime = end_dt.isoformat()
                        except:
                            end_datetime = start_datetime  # Fallback
                    
                    return {
                        **state,
                        "title": title if title != "MISSING" else "Untitled Event",
                        "start_datetime": start_datetime,
                        "end_datetime": end_datetime,
                        "description": description if description != "MISSING" else "",
                        "attendees": attendees,
                        "location": location if location != "NONE" else "",
                        "status": "ready"
                    }
                
                # Default to list events if intent is unclear
                return {
                    **state,
                    "status": "list_events",
                    "error_message": ""
                }
                
            except Exception as e:
                return {
                    **state,
                    "status": "error",
                    "error_message": f"Error analyzing request: {str(e)}"
                }
        
        def create_event_node(state: CalendarState) -> CalendarState:
            """Create the calendar event using the extracted information"""
            try:
                # Ensure attendees is always a list
                attendees = state.get("attendees", [])
                if attendees is None:
                    attendees = []
                elif isinstance(attendees, str):
                    if attendees == "MISSING" or attendees == "NONE":
                        attendees = []
                    else:
                        attendees = [email.strip() for email in attendees.split(',') if email.strip()]
                
                result = create_calendar_event_tool.invoke({
                    "title": state["title"],
                    "start_datetime": state["start_datetime"],
                    "end_datetime": state["end_datetime"],
                    "description": state["description"],
                    "attendees": attendees,
                    "location": state["location"]
                })
                
                return {
                    **state,
                    "status": result["status"],
                    "error_message": result["message"] if result["status"] == "error" else result["message"]
                }
                
            except Exception as e:
                return {
                    **state,
                    "status": "error",
                    "error_message": f"Error creating calendar event: {str(e)}"
                }
        
        def list_events_node(state: CalendarState) -> CalendarState:
            """List calendar events"""
            try:
                result = list_calendar_events_tool.invoke({
                    "max_results": 10
                })
                
                return {
                    **state,
                    "status": result["status"],
                    "error_message": result["message"] if result["status"] == "error" else result["message"]
                }
                
            except Exception as e:
                return {
                    **state,
                    "status": "error",
                    "error_message": f"Error listing calendar events: {str(e)}"
                }
        
        def determine_action(state: CalendarState) -> str:
            """Determine the next step based on current state"""
            status = state.get("status", "")
            if status == "ready":
                return "create_event"
            elif status == "list_events":
                return "list_events"
            else:
                return "end"
        
        # Create the graph
        workflow = StateGraph(CalendarState)
        
        # Add nodes
        workflow.add_node("analyze", analyze_request)
        workflow.add_node("create_event", create_event_node)
        workflow.add_node("list_events", list_events_node)
        
        # Add edges
        workflow.set_entry_point("analyze")
        workflow.add_conditional_edges(
            "analyze",
            determine_action,
            {
                "create_event": "create_event",
                "list_events": "list_events",
                "end": END
            }
        )
        workflow.add_edge("create_event", END)
        workflow.add_edge("list_events", END)
        
        return workflow.compile()
    
    def _extract_field(self, text: str, field: str) -> str:
        """Extract a specific field from LLM response"""
        pattern = rf"{field}:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "MISSING"
    
    def create_event(self, title: str, start_datetime: str, end_datetime: str, 
                    description: str = "", attendees: List[str] = None, location: str = "") -> Dict[str, Any]:
        """Legacy method for backward compatibility"""
        return create_calendar_event_tool.invoke({
            "title": title,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "description": description,
            "attendees": attendees or [],
            "location": location
        })
    
    def list_events(self, start_date: str = "", end_date: str = "", max_results: int = 10) -> Dict[str, Any]:
        """List calendar events"""
        return list_calendar_events_tool.invoke({
            "start_date": start_date,
            "end_date": end_date,
            "max_results": max_results
        })
    
    def execute(self, user_input: str,messages:list[str]=[]) -> Dict[str, Any]:
        """Execute the calendar management workflow"""
        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
            "title": "",
            "start_datetime": "",
            "end_datetime": "",
            "description": "",
            "attendees": [],
            "location": "",
            "status": "",
            "error_message": "",
            "messages":messages
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Format response
        if result["status"] == "completed":
            # Check if this was a list events request by looking at the original user input
            user_input_lower = initial_state["messages"][0]["content"].lower()
            is_list_request = any(keyword in user_input_lower for keyword in ['list', 'show', 'view', 'upcoming', 'events'])
            
            if is_list_request:
                # For list events, we need to get the actual events from the tool result
                # The events are stored in the tool's response, not in the state
                # Let's call the list tool directly to get the events
                try:
                    list_result = list_calendar_events_tool.invoke({
                        "max_results": 10
                    })
                    
                    if list_result["status"] == "completed":
                        events = list_result.get("events", [])
                        events_summary = []
                        for event in events:
                            events_summary.append({
                                "title": event.get("title", "No Title"),
                                "start": event.get("start", "No start time"),
                                "location": event.get("location", ""),
                                "description": event.get("description", "")[:100] + "..." if len(event.get("description", "")) > 100 else event.get("description", "")
                            })
                        
                        return {
                            "status": "completed",
                            "message": f"Found {len(events)} events",
                            "action": "list_events",
                            "events": events_summary,
                            "total_count": len(events)
                        }
                    else:
                        return {
                            "status": "error",
                            "message": list_result.get("message", "Failed to retrieve events")
                        }
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error retrieving events: {str(e)}"
                    }
            else:
                return {
                    "status": "completed",
                    "message": result["error_message"],
                    "action": "create_event",
                    "details": {
                        "title": result["title"],
                        "start_datetime": result["start_datetime"],
                        "end_datetime": result["end_datetime"]
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
    

    def _get_missing_fields(self, state: CalendarState) -> List[str]:
        """Get list of missing required fields"""
        missing = []
        if not state.get("title"):
            missing.append("title")
        if not state.get("start_datetime"):
            missing.append("start_datetime")
        return missing
    
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
                'capabilities': ['calendar_management', 'event_creation', 'availability_checking'],
                'status': 'evaluated'
            }
        
        # Use LLM to evaluate the task with Calendar-specific context
        evaluation_prompt = f"""
        You are evaluating a Google Calendar Agent's capability to handle a specific task.
        
        Calendar Agent Capabilities:
        - Create calendar events with Google Calendar API
        - List and view calendar events
        - Check user availability and conflicts
        - Auto-generate missing event details (titles, descriptions, end times)
        - Generate demo calendar data for testing
        - Handle attendees and locations
        - Validate date and time formats
        
        Calendar Agent Tools:
        - create_calendar_event_tool: Creates actual calendar events
        - list_calendar_events_tool: Lists user's calendar events
        - generate_demo_calendar_data_tool: Creates realistic calendar data for testing
        
        Task Description: {task_description}
        
        Evaluate this task for the Calendar Agent and provide:
        1. Confidence score (0.0 to 1.0) - how confident the agent can complete this task
        2. Estimated time to complete
        3. Required information/inputs
        4. Agent capabilities relevant to this task
        5. Any potential challenges or limitations
        6. Whether demo data generation would be needed
        
        Consider:
        - Calendar event creation tasks get high confidence (0.8-1.0)
        - Availability checking tasks get very high confidence (0.9-1.0)
        - Event listing tasks get high confidence (0.8-1.0)
        - Scheduling tasks get medium-high confidence (0.7-0.9)
        - Non-calendar tasks get low confidence (0.2-0.4)
        
        Respond in this JSON format:
        {{
            "confidence": 0.85,
            "estimated_time": "2-5 minutes",
            "requirements": ["title", "start_datetime", "end_datetime"],
            "capabilities": ["event_creation", "demo_data_generation"],
            "challenges": ["requires date and time"],
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
                    'requirements': ['title', 'start_datetime', 'end_datetime'],
                    'capabilities': ['calendar_management', 'event_creation'],
                    'status': 'evaluated'
                }
                
        except Exception as e:
            # Fallback to default if LLM evaluation fails
            return {
                'confidence': 0.5,
                'estimated_time': 'unknown',
                'requirements': ['title', 'start_datetime', 'end_datetime'],
                'capabilities': ['calendar_management', 'event_creation'],
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
            "name": "Google Calendar Agent",
            "description": "Specialized agent for Google Calendar management using LangGraph workflow",
            "capabilities": [
                "Create calendar events from natural language",
                "List upcoming calendar events",
                "Auto-generate missing event details",
                "Handle attendees and locations",
                "Validate date and time formats",
                "Generate demo data for testing"
            ],
            "tools": ["create_calendar_event_tool", "list_calendar_events_tool", "generate_demo_calendar_data_tool"],
            "status": "active"
        }
    
    def run(self, user_input: str = None,messages:list[str]=[]):
        """Main entry point for the agent"""
        if user_input:
            return self.execute(user_input,messages)
        else:
            return {
                "status": "input_required",
                "message": "Please provide calendar event details or ask to list events"
            }