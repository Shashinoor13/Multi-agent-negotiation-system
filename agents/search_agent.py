from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from agents.base import Agent
from services.llm_service import LLMService
import re
import os
from pydantic import BaseModel, Field

# Import Tavily
from tavily import TavilyClient

# Define the state structure for LangGraph
class SearchState(TypedDict):
    messages: Annotated[List[Dict], "Messages in the conversation"]
    query: str
    search_results: str
    status: str  # input_required, error, completed
    error_message: str

# Define a Pydantic model for the search query to improve tool schema
class SearchToolInput(BaseModel):
    query: str = Field(description="The search query string to use for web search.")
    max_results: int = Field(default=5, description="Maximum number of search results to return.")
    # You could also add other Tavily parameters like 'include_raw_content', 'search_depth', 'topic'

# Web search tool using Tavily API
@tool(args_schema=SearchToolInput)
def tavily_search_tool(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Perform a web search using Tavily Search API.
    
    Args:
        query: The search query string.
        max_results: Maximum number of search results to return.
    
    Returns:
        Dict with status, message, and search_results
    """
    try:
        tavily_api_key = os.getenv('TAVILY_API_KEY')
        
        if not tavily_api_key:
            return {
                "status": "error",
                "message": "Tavily API key not configured. Please set TAVILY_API_KEY environment variable."
            }
        
        client = TavilyClient(api_key=tavily_api_key)
        
        # Use Tavily's search method
        # You can choose to get raw results or a summarized answer
        # For simplicity, we'll get basic results and let our LLM summarize
        response = client.search(query=query, search_depth="basic", max_results=max_results, include_answer=False) 
        
        # Format results for the agent
        formatted_results = []
        if response and response['results']:
            for result in response['results']:
                formatted_results.append(f"Title: {result['title']}\nURL: {result['url']}\nContent: {result['content'][:500]}...") # Truncate content
        
        results_string = "\n---\n".join(formatted_results) if formatted_results else "No relevant results found."

        return {
            "status": "completed",
            "message": f"Search completed for '{query}'.",
            "search_results": results_string
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to perform search with Tavily: {str(e)}"
        }

# Specialized search tool for checking holidays and events
@tool
def check_holidays_events_tool(date: str, location: str = "general") -> Dict[str, Any]:
    """
    Search for holidays, events, and other factors that might block scheduling on a specific date.
    
    Args:
        date: The date to check in YYYY-MM-DD format
        location: The location to check (city, country, or "general" for worldwide)
    
    Returns:
        Dict with status, message, and blocking factors found
    """
    try:
        tavily_api_key = os.getenv('TAVILY_API_KEY')
        
        if not tavily_api_key:
            return {
                "status": "error",
                "message": "Tavily API key not configured. Please set TAVILY_API_KEY environment variable."
            }
        
        client = TavilyClient(api_key=tavily_api_key)
        
        # Create specific search queries for different types of blocking factors
        search_queries = [
            f"public holidays {date} {location}",
            f"major events {date} {location}",
            f"conferences {date} {location}",
            f"business closures {date} {location}",
            f"transportation strikes {date} {location}",
            f"weather events {date} {location}"
        ]
        
        all_results = []
        
        for query in search_queries:
            try:
                response = client.search(query=query, search_depth="basic", max_results=3, include_answer=False)
                if response and response['results']:
                    for result in response['results']:
                        all_results.append({
                            'query': query,
                            'title': result['title'],
                            'content': result['content'][:300],
                            'url': result['url']
                        })
            except Exception as e:
                # Continue with other queries if one fails
                continue
        
        if all_results:
            # Format results
            formatted_results = []
            for result in all_results:
                formatted_results.append(f"Query: {result['query']}\nTitle: {result['title']}\nContent: {result['content']}\nURL: {result['url']}")
            
            results_string = "\n---\n".join(formatted_results)
            
            return {
                "status": "completed",
                "message": f"Found potential blocking factors for {date} in {location}",
                "blocking_factors": results_string,
                "date": date,
                "location": location
            }
        else:
            return {
                "status": "completed",
                "message": f"No significant blocking factors found for {date} in {location}",
                "blocking_factors": "No relevant blocking factors found",
                "date": date,
                "location": location
            }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to check holidays and events: {str(e)}"
        }

class SearchAgent(Agent):
    def __init__(self, id):
        self.id = id
        self.llm = LLMService.get_model()
        self.tools = [tavily_search_tool, check_holidays_events_tool]
        self.tool_node = ToolNode(self.tools)
        self.graph = self._create_graph()
        
    def root_instruction(self):
        SYSTEM_INSTRUCTION = (
            'You are a specialized assistant for web searching and negotiation support. '
            "Your purpose is to help users find information on the internet by formulating effective search queries "
            "and returning relevant results from Tavily Search. "
            "You should always ask for clarification if the user's request is ambiguous or lacks a clear search query. "
            "NEVER invent or guess search queries without sufficient user input. "
            "For negotiation tasks, focus on finding relevant information about scheduling, availability, and coordination."
        )
        
        FORMAT_INSTRUCTION = (
            'Analyze the user input and extract:\n'
            '1. Search query (REQUIRED - ask if missing/ambiguous)\n\n'
            'If the search query is missing or unclear, set status to "input_required".\n'
            'If the search query is identified, set status to "ready".\n'
            'After a search, reflect on the results and provide a concise summary or direct answer to the user.'
        )
        
        return f"{SYSTEM_INSTRUCTION}\n\n{FORMAT_INSTRUCTION}"
    
    def evaluation_instruction(self):
        SYSTEM_INSTRUCTION = (
            'You are a specialized search evaluation assistant for negotiation support. '
            "Your purpose is to evaluate meeting offers by searching for holidays, events, and other factors "
            "that might block or affect the requested time frames. "
            "You search for public holidays, major events, weather conditions, and other relevant information "
            "that could impact scheduling and availability. "
            "For negotiation tasks, focus on identifying potential scheduling conflicts and external factors."
        )
        
        FORMAT_INSTRUCTION = (
            'Analyze the offers and search for blocking factors:\n'
            '1. Extract time frames from each offer (start_datetime, end_datetime)\n'
            '2. Search for holidays, events, and other factors that might block each time frame:\n'
            '   - Public holidays in the relevant location\n'
            '   - Major events, conferences, or gatherings\n'
            '   - Weather conditions or natural events\n'
            '   - Transportation issues or strikes\n'
            '   - Business closures or special hours\n'
            '3. Assess blocking status for each offer:\n'
            '   - NO_BLOCKS: No significant blocking factors found\n'
            '   - MINOR_BLOCKS: Some minor factors that might affect attendance\n'
            '   - MAJOR_BLOCKS: Significant events that could prevent attendance\n'
            '   - HOLIDAY_BLOCK: Falls on a public holiday\n'
            '   - EVENT_BLOCK: Conflicts with major events\n'
            '4. For each offer, provide:\n'
            '   - Blocking status\n'
            '   - Specific blocking factors found\n'
            '   - Location-specific considerations\n'
            '   - Alternative suggestions (if needed)\n'
            '   - Risk assessment (LOW/MEDIUM/HIGH)\n\n'
            'Respond in this format:\n'
            'OFFER_1:\n'
            'TIME_FRAME: [start_datetime] to [end_datetime]\n'
            'LOCATION: [location or GENERAL]\n'
            'BLOCKING_STATUS: [status]\n'
            'BLOCKING_FACTORS: [specific factors found or NONE]\n'
            'LOCATION_FACTORS: [location-specific considerations or NONE]\n'
            'ALTERNATIVES: [suggested alternatives or NONE]\n'
            'RISK_LEVEL: [LOW/MEDIUM/HIGH]\n'
            'RECOMMENDATION: [proceed/caution/avoid]\n\n'
            'Repeat for each offer. Provide overall recommendation at the end.'
        )
        
        return f"{SYSTEM_INSTRUCTION}\n\n{FORMAT_INSTRUCTION}"
    
    def _create_graph(self):
        """Create the LangGraph workflow"""
        
        def analyze_request(state: SearchState) -> SearchState:
            """Analyze the user request and extract the search query"""
            messages = state.get("messages", [])
            if not messages:
                return {
                    **state,
                    "status": "input_required",
                    "error_message": "No input provided"
                }
            
            user_input = messages[-1].get("content", "")
            
            # Use LLM to analyze and extract the search query
            prompt = f"""
            {self.root_instruction()}
            
            User request: {user_input}
            
            Extract the search query:
            - query: The concise and effective search query string. (REQUIRED)
            
            Respond in this format:
            QUERY: [search query or MISSING]
            STATUS: [input_required/ready/error]
            """
            
            try:
                response = self.llm.invoke(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Parse LLM response
                query = self._extract_field(response_text, "QUERY")
                status = self._extract_field(response_text, "STATUS")
                
                if query == "MISSING" or status == "input_required":
                    return {
                        **state,
                        "status": "input_required",
                        "error_message": "Missing information: search query. Please tell me what you want to search for."
                    }
                
                return {
                    **state,
                    "query": query,
                    "status": "ready"
                }
                
            except Exception as e:
                return {
                    **state,
                    "status": "error",
                    "error_message": f"Error analyzing request: {str(e)}"
                }
        
        def perform_search_node(state: SearchState) -> SearchState:
            """Perform the web search using the extracted query"""
            try:
                result = tavily_search_tool.invoke({ # <--- Changed tool here
                    "query": state["query"]
                })
                
                return {
                    **state,
                    "status": result["status"],
                    "error_message": result["message"] if result["status"] == "error" else "",
                    "search_results": result.get("search_results", "No results found.")
                }
                
            except Exception as e:
                return {
                    **state,
                    "status": "error",
                    "error_message": f"Error performing search: {str(e)}"
                }
        
        def provide_summary_node(state: SearchState) -> SearchState:
            """Summarize the search results for the user."""
            if state["status"] == "completed" and state["search_results"] != "No relevant results found.":
                # Use LLM to summarize the results
                prompt = f"""
                You have just performed a search with the query: "{state['query']}".
                Here are the raw search results:
                
                {state['search_results']}
                
                Please provide a concise and helpful summary of these results to the user.
                If the results don't directly answer the user's original query, indicate that.
                """
                
                try:
                    response = self.llm.invoke(prompt)
                    summary = response.content if hasattr(response, 'content') else str(response)
                    return {
                        **state,
                        "status": "completed",
                        "error_message": f"Search successful. Here's a summary: {summary}"
                    }
                except Exception as e:
                    return {
                        **state,
                        "status": "error",
                        "error_message": f"Search successful, but failed to summarize results: {str(e)}. Raw results: {state['search_results']}"
                    }
            elif state["status"] == "completed" and state["search_results"] == "No relevant results found.":
                 return {
                    **state,
                    "status": "completed",
                    "error_message": "No relevant results found for your search query."
                }
            else:
                return state # Pass through any error state

        def should_perform_search(state: SearchState) -> str:
            """Determine the next step based on current state"""
            status = state.get("status", "")
            if status == "ready":
                return "perform_search"
            else:
                return "end" # If input_required or error, end

        def after_search_action(state: SearchState) -> str:
            """Determine next step after search is performed"""
            if state.get("status") == "completed":
                return "summarize_results"
            else:
                return "end" # End if search failed

        # Create the graph
        workflow = StateGraph(SearchState)
        
        # Add nodes
        workflow.add_node("analyze", analyze_request)
        workflow.add_node("perform_search", perform_search_node)
        workflow.add_node("summarize_results", provide_summary_node)
        
        # Add edges
        workflow.set_entry_point("analyze")
        workflow.add_conditional_edges(
            "analyze",
            should_perform_search,
            {
                "perform_search": "perform_search",
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "perform_search",
            after_search_action,
            {
                "summarize_results": "summarize_results",
                "end": END # If perform_search ended in error
            }
        )
        workflow.add_edge("summarize_results", END) # After summarizing, we end

        return workflow.compile()
    
    def _extract_field(self, text: str, field: str) -> str:
        """Extract a specific field from LLM response"""
        pattern = rf"{field}:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "MISSING"
    
    # You can keep this for backward compatibility if needed, but the graph is the primary execution path
    def search_web(self, query: str) -> Dict[str, Any]:
        """Perform a web search."""
        return tavily_search_tool.invoke({"query": query})

    def execute(self, user_input: str,messages:list[str]=[]) -> Dict[str, Any]:
        """Execute the web search workflow"""
        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
            "query": "",
            "search_results": "",
            "status": "",
            "error_message": "",
            "messages":messages
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Format response
        if result["status"] == "completed":
            return {
                "status": "completed",
                "message": result["error_message"], # This will contain the summary
                "details": {
                    "query": result["query"],
                    "results_summary": result["error_message"]
                }
            }
        elif result["status"] == "input_required":
            return {
                "status": "input_required",
                "message": result["error_message"],
                "required_fields": ["query"]
            }
        else: # Error state
            return {
                "status": "error",
                "message": result["error_message"]
            }
    
    def evaluate(self, task_description: str = None,context: List[str] = []) -> Dict[str, Any]:
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
                'capabilities': ['web_search', 'information_gathering', 'research'],
                'status': 'evaluated'
            }
        
        # Use LLM to evaluate the task with Search-specific context
        evaluation_prompt = f"""
        You are evaluating a Web Search Agent's capability to handle a specific task.
        
        Search Agent Capabilities:
        - Perform web searches using DuckDuckGo API
        - Gather information from search results
        - Research topics and provide summaries
        - Check for holidays, events, and scheduling conflicts
        - Evaluate meeting offers and availability
        - Find relevant information for decision making
        - Analyze search results for relevance and accuracy
        
        Search Agent Tools:
        - search_tool: Performs web searches and returns results
        - evaluate_meeting_offer_tool: Analyzes meeting offers for conflicts
        
        Task Description: {task_description}
        Context: {context}
        Evaluate this task for the Search Agent and provide:
        1. Confidence score (0.0 to 1.0) - how confident the agent can complete this task
        2. Estimated time to complete
        3. Required information/inputs
        4. Agent capabilities relevant to this task
        5. Any potential challenges or limitations
        6. Whether the task requires external information gathering
        
        Consider:
        - Research and information gathering tasks get high confidence (0.8-1.0)
        - Web search tasks get very high confidence (0.9-1.0)
        - Holiday/event checking tasks get high confidence (0.8-1.0)
        - Meeting evaluation tasks get medium-high confidence (0.7-0.9)
        - Non-search tasks get low confidence (0.2-0.4)
        
        Respond in this JSON format:
        {{
            "confidence": 0.85,
            "estimated_time": "1-3 minutes",
            "requirements": ["search_query", "topic"],
            "capabilities": ["web_search", "information_gathering"],
            "challenges": ["requires specific search terms"],
            "needs_external_data": true,
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
                    'requirements': ['search_query', 'topic'],
                    'capabilities': ['web_search', 'information_gathering'],
                    'status': 'evaluated'
                }
                
        except Exception as e:
            # Fallback to default if LLM evaluation fails
            return {
                'confidence': 0.5,
                'estimated_time': 'unknown',
                'requirements': ['search_query', 'topic'],
                'capabilities': ['web_search', 'information_gathering'],
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
            "name": "Search Agent",
            "description": "Specialized agent for web searching and offer evaluation using LangGraph workflow and Tavily Search.",
            "capabilities": [
                "Extract search queries from natural language",
                "Perform web searches using Tavily API",
                "Summarize search results",
                "Handle missing search queries",
                "Evaluate offers for holidays and events",
                "Check for blocking factors in scheduling",
                "Provide risk assessments for meeting times"
            ],
            "tools": ["tavily_search_tool", "check_holidays_events_tool"],
            "status": "active"
        }
    
    def run(self, user_input: str = None,context:list[str]=[]):
        """Main entry point for the agent"""
        if user_input:
            return self.execute(user_input,context)
        else:
            return {
                "status": "input_required",
                "message": "Please provide a search query."
            }