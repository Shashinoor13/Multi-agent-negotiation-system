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

class SearchAgent(Agent):
    def __init__(self, id):
        self.id = id
        self.llm = LLMService.get_model()
        self.tools = [tavily_search_tool] # <--- Changed tool here
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

    def execute(self, user_input: str) -> Dict[str, Any]:
        """Execute the web search workflow"""
        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
            "query": "",
            "search_results": "",
            "status": "",
            "error_message": ""
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
    
    def evaluate(self):
        return super().evaluate()
    
    def counter(self):
        return super().counter()
    
    def update_state(self):
        return super().update_state()
    
    def generate_agent_card(self):
        return {
            "id": self.id,
            "name": "Search Agent",
            "description": "Specialized agent for web searching using LangGraph workflow and Tavily Search.",
            "capabilities": [
                "Extract search queries from natural language",
                "Perform web searches using Tavily API",
                "Summarize search results",
                "Handle missing search queries"
            ],
            "tools": ["tavily_search_tool"],
            "status": "active"
        }
    
    def run(self, user_input: str = None):
        """Main entry point for the agent"""
        if user_input:
            return self.execute(user_input)
        else:
            return {
                "status": "input_required",
                "message": "Please provide a search query."
            }