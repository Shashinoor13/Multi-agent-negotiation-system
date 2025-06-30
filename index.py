# index.py
import os
from typing import Any, Callable
import mesop as me
import mesop.labs as mel
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

# Assuming these imports are correctly set up in your project structure
# Ensure these modules and their classes exist and are correctly implemented
from agents.calendar_agent import GoogleCalendarAgent
from agents.gmail_agent import GmailAgent
from agents.search_agent import SearchAgent
from environment.negotiation_environment import NegotiationEnvironment
from strategy.simple import SimpleNegotiationStrategy


app = FastAPI()

@me.stateclass
class State:
    # This is the only state variable we need now for the negotiation log
    output_messages: list[str]
    is_negotiating: bool = False
    count = 0

async def start_negotiation(event: me.ClickEvent):
    state = me.state(State)
    state.output_messages = [] # Clear previous messages for a new run
    state.is_negotiating = True
    yield
    # Mesop will automatically re-render here because state.is_negotiating changed.
    
    # Available agents (these would likely be initialized elsewhere in a real app)
    # Ensure these agents are properly configured and callable
    gmail_agent = GmailAgent("gmail_agent_01")
    calendar_agent = GoogleCalendarAgent("calendar_agent_01")
    search_agent = SearchAgent("search_agent_01")

    # Available Strategies
    simple = SimpleNegotiationStrategy()

    negotiation_environment = NegotiationEnvironment(agents=[gmail_agent, calendar_agent, search_agent], strategy=simple)
    
    task = "Plan a meeting with Rista next week, his email is ristashrestha10@gmail.com"

    # Iterate through the yielded messages from set_task
    for message in negotiation_environment.set_task(task=task):
        # print(message)
        state.output_messages.append(message)
        yield
        state.count =+1
        # Mesop will automatically re-render because state.output_messages changed.

    # Iterate through the yielded messages from negotiate
    for message in negotiation_environment.negotiate():
        if isinstance(message, dict): # Check if the last yielded item is the results dict
            state.output_messages.append(f"\nFinal Results: {message}")
            yield
        else:
            state.output_messages.append(message)
            yield
        # Mesop will automatically re-render because state.output_messages changed.
    
    state.is_negotiating = False
    yield
    # Mesop will automatically re-render here because state.is_negotiating changed.



@me.page(
    # Security policy is less critical if no external scripts, but harmless to keep
    security_policy=me.SecurityPolicy(
        allowed_script_srcs=[
            "https://cdn.jsdelivr.net",
        ]
    )
)
def negotiation_page():
    state = me.state(State)
    
    with me.box(style=me.Style(
        padding=me.Padding.all(20),
        # max_width=me.Length(800), # Keeping max_width for better readability on large screens
        margin=me.Margin.symmetric(horizontal="auto")
    )):
        me.text("# Agent Negotiation Environment", style=me.Style(font_weight="bold", font_size="2em")) # Increased font size
        me.text("Click the button to start the multi-agent negotiation process for task assignment.")
        
        me.button(
            "Start Negotiation", 
            on_click=start_negotiation, 
            type="flat",
            disabled=state.is_negotiating,
            style=me.Style(margin=me.Margin.symmetric(vertical=20), font_size="1.2em", padding=me.Padding.all(10)) # Styled button
        )
        
        if state.is_negotiating:
            me.text("Negotiation in progress...", style=me.Style(color="orange", font_weight="bold"))
        
        if state.output_messages:
            me.divider()
            me.text("## Negotiation Log", style=me.Style(font_weight="bold", margin=me.Margin(bottom=10), font_size="1.5em"))
            with me.box(style=me.Style(
                background="lightgray",
                padding=me.Padding.all(10),
                # border_radius=me.BorderRadius.all(5),
                # height=me.Length(600), # Increased fixed height for more log content
                overflow_y="auto" # Enable scrolling
            )):
                for message in state.output_messages:
                    # Using me.markdown for potential richer text formatting if needed in future
                    me.text(message, style=me.Style(font_family="monospace", white_space="pre-wrap")) # pre-wrap for preserving newlines
        
# Removed increment, Value, on_value, web_component functions and their imports
# as they are no longer needed.


app.mount(
    "/",
    WSGIMiddleware(
        me.create_wsgi_app(debug_mode=os.environ.get("DEBUG_MODE", "") == "true")
    ),
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "index:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_includes=["*.py", "*.js"],
        timeout_graceful_shutdown=0,
    )