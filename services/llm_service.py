import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv
import re
import json

from agents.base import Agent
from environment.negotiation_environment import Task

class LLMService:
    _model = None
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def get_model():
        """Get the configured Gemini model for use in GmailAgent"""
        if LLMService._model is None:
            LLMService._initialize_gemini()
        return GeminiModelWrapper(LLMService._model)
    
    @staticmethod
    def _initialize_gemini():
        """Initialize Gemini API with configuration"""
        try:
            load_dotenv()
            # Get API key from environment variable
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Initialize the model
            model_name = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
            
            generation_config = GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048,
            )
            
            LLMService._model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config
            )
            
            print(f"✅ Gemini model '{model_name}' initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing Gemini: {str(e)}")
            raise
    
    def split_task(self, task):
        system_prompt = """
    You are a multi-agent task planner. Your job is to take a complex task and decompose it into smaller subtasks that can be assigned to different intelligent agents. Each agent has specific capabilities and responsibilities.
    Your output must be a structured breakdown of the task, assigning specific subtasks to the relevant agents. Think step-by-step and ensure subtasks are clear, actionable, and aligned with each agent's capabilities.
    DO NOT RETURN MORE THAN 4 SUBTASKS AT A TIME.
    DO NOT GENERATE EMAIL ADDRESSES OR RECIPIENT NAMES.
    """
        examples = [
            {
                "input": "Organize a team lunch next week at a time everyone is free.",
                "output": [
                    {
                        "subtask": "Fetch all participants' availability for next week between 10am and 5pm."
                    },
                    {
                        "subtask": "Check for holidays or non-working days next week."
                    },
                    {
                        "subtask": "Send available lunch time slots to participants and gather their preferences."
                    }
                ]
            }
        ]

        output = [
            {
                "subtask": "Fetch available time slots for the user and Person X from their calendars for next week between 10am and 5pm."
            },
            {
                "subtask": "Check if there are any holidays or restricted work hours next week that affect scheduling."
            },
            {
                "subtask": "Send proposed meeting times to Person X and handle responses to finalize a mutually agreed slot."
            }
        ]

        response = LLMService._model.generate_content(
            f"{system_prompt}\nExamples:{examples}\n\n output:{output}\n\nTask: {task}"
        )

        return response

        
    
    def select_agents_according_to_task(self, agents: list[Agent], tasks: list[Task]):
        try:
            # Step 1: Gather agent profiles
            agent_cards = []
            agent_id_map = {}
            for agent in agents:
                card = agent.generate_agent_card()
                agent_cards.append({
                    "id": agent.id,
                    "card": card
                })
                agent_id_map[agent.id] = agent  # for future use

            # Step 2: Prepare subtasks
            task_list = []
            for task in tasks:
                task_list.append({
                    "id": task.id,
                    "description": task.task
                })

            # Step 3: Prepare prompt for Gemini
            prompt = f"""
    You are a multi-agent coordinator. You will be given a list of intelligent agents with their capabilities and a list of subtasks that need to be performed.
    Your job is to assign the best-suited agent to each subtask. Match based on the agent's skills or domain mentioned in their profile.
    Return a JSON array in this format:
    [
    {{
        "task_id": "string",
        "assigned_to": "agent_id"
    }},
    ...
    ]

    Agents:
    {json.dumps(agent_cards, indent=2)}

    Tasks:
    {json.dumps(task_list, indent=2)}
            """

            # Step 4: Get Gemini model
            model = LLMService.get_model()

            # Step 5: Ask Gemini to perform the assignment
            response = model.invoke(prompt)
            # Step 6: Parse response
            return self.parse_assignment_output(response)

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to assign agents: {str(e)}"
            }

    
    def parse_output(self, response):
        """Parse the LLM response and extract structured task breakdown"""
        try:
            # Extract the text content from the response
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Find JSON content in the response (handle markdown code blocks)
            import json
            import re
            
            # Look for JSON in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no markdown, try to find JSON array directly
                json_match = re.search(r'(\[.*?\])', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    raise ValueError("No JSON content found in response")
            
            # Parse the JSON
            tasks = json.loads(json_str)
            
            # Format the output nicely
            formatted_tasks = []
            for i, task in enumerate(tasks, 1):
                formatted_task = {
                    "id": i,
                    "agent": task.get("agent", "Unknown Agent"),
                    "subtask": task.get("subtask", "No subtask specified"),
                    "status": "pending"
                }
                formatted_tasks.append(formatted_task)
            
            return {
                "status": "success",
                "total_tasks": len(formatted_tasks),
                "tasks": formatted_tasks,
                "raw_response": response_text
            }
            
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "message": f"Failed to parse JSON: {str(e)}",
                "raw_response": response_text if 'response_text' in locals() else str(response)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error parsing response: {str(e)}",
                "raw_response": str(response)
            }
        
    def parse_assignment_output(self, response):
        try:
            response_text = response.content if hasattr(response, 'content') else str(response)

            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\[.*?\])', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON array found in response")

            json_str = json_match.group(1)
            assignments = json.loads(json_str)

            return {
                "status": "success",
                "assignments": assignments,
                "raw_response": response_text
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error parsing agent assignment: {str(e)}",
                "raw_response": str(response)
            }


    def split_and_parse_task(self, task):
        """Split a task and parse the response in one step"""
        try:
            # Split the task
            response = self.split_task(task)
            
            # Parse the response
            parsed_result = self.parse_output(response)
            
            return parsed_result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error in split_and_parse_task: {str(e)}"
            }

class GeminiModelWrapper:
    """Wrapper class to make Gemini model compatible with LangChain-like interface"""
    
    def __init__(self, model):
        self.model = model
        self.system_prompt = """You are an intelligent email assistant that helps complete incomplete email prompts. Your role is to:

1. Analyze the given email prompt and identify missing fields
2. Auto-generate appropriate content for missing fields EXCEPT for recipient/email address
3. NEVER generate or guess email addresses or recipient names
4. For missing subjects, generate a clear, professional subject line based on the email content
5. For missing body content, expand on the given information appropriately
6. Maintain the professional tone and context of the original prompt

When completing prompts:
- If subject is missing: Generate a concise, descriptive subject line
- If body content is incomplete: Expand naturally while maintaining the original intent
- If recipient is missing: Leave it blank and note that recipient information is required
- Always preserve any existing content and formatting

Respond with the completed email prompt, clearly indicating any auto-generated sections."""

    def _is_incomplete_prompt(self, prompt: str) -> bool:
        """Check if the prompt appears to be incomplete"""
        # Check for common incomplete patterns
        incomplete_indicators = [
            r'subject:\s*$',  # Empty subject
            r'subject:\s*\n',  # Subject with just newline
            r'to:\s*$',  # Empty recipient
            r'to:\s*\n',  # Recipient with just newline
            r'body:\s*$',  # Empty body
            r'body:\s*\n',  # Body with just newline
            r'content:\s*$',  # Empty content
            r'content:\s*\n',  # Content with just newline
        ]
        
        prompt_lower = prompt.lower()
        for pattern in incomplete_indicators:
            if re.search(pattern, prompt_lower, re.MULTILINE):
                return True
        
        # Check if prompt is too short (likely incomplete)
        if len(prompt.strip()) < 50:
            return True
            
        return False

    def _complete_prompt(self, prompt: str) -> str:
        """Complete an incomplete prompt using the system prompt"""
        completion_prompt = f"""{self.system_prompt}

Please complete the following incomplete email prompt. Only generate content for missing fields (except recipient/email addresses):

{prompt}

Complete the prompt by filling in missing fields:"""

        try:
            response = self.model.generate_content(completion_prompt)
            return response.text
        except Exception as e:
            print(f"Error completing prompt: {str(e)}")
            return prompt  # Return original if completion fails

    def invoke(self, prompt: str):
        """Invoke the model with a prompt and return response"""
        try:
            # Check if prompt is incomplete and complete it if necessary
            if self._is_incomplete_prompt(prompt):
                print("🔧 Detected incomplete prompt, attempting to complete...")
                completed_prompt = self._complete_prompt(prompt)
                print(f"✅ Completed prompt: {completed_prompt[:100]}...")
                prompt = completed_prompt
            
            response = self.model.generate_content(prompt)
            return ModelResponse(response.text)
        except Exception as e:
            print(f"Error invoking Gemini model: {str(e)}")
            return ModelResponse(f"Error: {str(e)}")

class ModelResponse:
    """Response wrapper to provide consistent interface"""
    
    def __init__(self, content: str):
        self.content = content
    
    def __str__(self):
        return self.content