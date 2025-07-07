import os
import mesop as me
import mesop.labs as mel
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from agents.calendar_agent import GoogleCalendarAgent
from agents.gmail_agent import GmailAgent
from agents.search_agent import SearchAgent
from environment.negotiation_environment import NegotiationEnvironment
from services.embedding_service import save_embeddings
from services.llm_service import LLMService
from strategy.simple import SimpleNegotiationStrategy

app = FastAPI()

@me.stateclass
class State:
    output_messages: list[str]
    is_negotiating: bool = False
    user_task:str
    input:str
    count = 0

async def start_negotiation(e:me.ClickEvent):
    state = me.state(State)
    state.output_messages = []  # Clear previous messages for a new run
    state.is_negotiating = True
    yield

    gmail_agent = GmailAgent("gmail_agent_01")
    calendar_agent = GoogleCalendarAgent("calendar_agent_01")
    search_agent = SearchAgent("search_agent_01")
    simple = SimpleNegotiationStrategy()
    negotiation_environment = NegotiationEnvironment(
        agents=[gmail_agent, calendar_agent, search_agent], strategy=simple
    )
    task = state.user_task

    for message in negotiation_environment.set_task(task=task):
        state.output_messages.append(message)
        yield
        state.count += 1

    for message in negotiation_environment.negotiate():
        if isinstance(message, dict):
            state.output_messages.append(f"[Final Results] {message}")
            yield
        else:
            state.output_messages.append(message)
            yield

    state.is_negotiating = False
    yield

def parse_step(message: str):
    """Parse step/phase and agent from message. Expects format: [Step][Agent] message"""
    step = ""
    agent = ""
    msg = message
    if message.startswith("["):
        end_step = message.find("]")
        if end_step != -1:
            step = message[1:end_step]
            rest = message[end_step+1:].lstrip()
            if rest.startswith("["):
                end_agent = rest.find("]")
                if end_agent != -1:
                    agent = rest[1:end_agent]
                    msg = rest[end_agent+1:].lstrip()
                else:
                    msg = rest
            else:
                msg = rest
    return step, agent, msg

@me.component
def negotiation_bubble(message: str, agent: str = "", step: str = "", is_final: bool = False):
    """Styled chat bubble for negotiation messages, with agent and step label."""
    with me.box(
        style=me.Style(
            display='flex',
            flex_direction='column',
            align_items='flex-start',
            width="100%",
        )
    ):
        label = ""
        if step:
            label += f"{step} "
        if agent:
            label += f"({agent})"
        if label:
            me.text(
                label,
                style=me.Style(
                    font_size="0.95em",
                    font_weight="bold",
                    color="#0ea5e9",  # Blue for step/agent label

                )
            )
        with me.box(
            style=me.Style(
                background="#f3f4f6" if not is_final else "#dbeafe",  # Light bubble, blue for final
                color="#18181b",
                border_radius=12,
                margin=me.Margin(top=4, bottom=4,left=4),
                padding=me.Padding(top=2, bottom=2, left=16, right=16),
                max_width='90%',
                width="fit-content",
            )
        ):
            me.markdown(message, style=me.Style(font_family='monospace', font_size='1em'))

def on_blur(e: me.InputBlurEvent):
    state = me.state(State)
    state.user_task = e.value

@me.page(
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
        margin=me.Margin.symmetric(horizontal="auto"),
        max_width="100%",
        background="#f8fafc",  # Lighter background for better contrast
        border_radius=12,
        box_shadow='0 2px 8px rgba(0,0,0,0.10)',
        min_height="100vh"
    )):
        me.text("Agent Negotiation Environment", style=me.Style(font_weight="bold", font_size="2em", color="#18181b"))
        me.text(
            "Enter a task and start the multi-agent negotiation process",
            style=me.Style(color="#334155", margin=me.Margin(bottom=2))
        )

        with me.box(style=me.Style(min_width="100vw",display="flex", flex_direction="row", gap=8, margin=me.Margin(bottom=18))):
            me.input(
                value=state.user_task,
                label="Task",
                on_blur=on_blur,
                style=me.Style(min_width="85vw", background="#fff", color="#18181b")
            )
            me.button(
                "Send",
                on_click=start_negotiation,
                disabled=state.is_negotiating,
                style=me.Style(
                    font_size="1.1em",
                    padding=me.Padding(top=8, bottom=8, left=18, right=18),
                    background="#0ea5e9",
                    color="#fff",
                    min_width="100px",
                    border_radius=8,
                    box_shadow='0 1px 3px rgba(0,0,0,0.10)'
                )
            )

        if state.is_negotiating:
            me.text("Negotiation in progress...", style=me.Style(color="#fbbf24", font_weight="bold", margin=me.Margin(bottom=2)))

        if state.output_messages:
            me.divider()
            me.text("Negotiation Log", style=me.Style(font_weight="bold", font_size="1.3em", color="#18181b"))
            with me.box(style=me.Style(
                background="#e5e7eb",
                border_radius=8,
                height="100%",
                overflow_y="auto",
                box_shadow='0 1px 3px rgba(0,0,0,0.08)'
            )):
                for idx, message in enumerate(state.output_messages):
                    step, agent, msg = parse_step(message)
                    negotiation_bubble(
                        msg,
                        agent=agent,
                        step=step,
                        is_final=step.lower() == "final results"
                    )

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