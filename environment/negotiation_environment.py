from agents.base import Agent

class Task:
    def __init__(self,id,task) -> None:
        self.id = id
        self.task = task

class TaskAssignment:
    def __init__(self,task:Task,agent:Agent) -> None:
        self.agent:Agent = agent
        self.task:Task = task


class NegotiationEnvironment:
    def __init__(self,agents:list[Agent],strategy,max_rounds=5):
        self.agents = agents
        self.strategy = strategy
        self.tasks:list[Task] = []
        self.taskAssignments:list[TaskAssignment]=[]
        self.max_rounds = max_rounds
    
    def set_task(self, task):
        from services.llm_service import LLMService

        llm = LLMService()
        raw_llm_response=llm.split_task(task=task)
        
        parsed_result = llm.parse_output(raw_llm_response)
        
        if parsed_result["status"] == "success":
            tasks = parsed_result["tasks"] 
            print("Tasks successfully split and parsed")
            for task_item in tasks:
                self.tasks.append(Task(task_item['id'],task_item['subtask']))
        else:
            print(f"Error splitting or parsing task: {parsed_result['message']}")
            print(f"Raw response: {parsed_result.get('raw_response', 'N/A')}")
    
    def distribute_task(self):
        from services.llm_service import LLMService
        llm = LLMService()
        parsed_result = llm.select_agents_according_to_task(agents=self.agents,tasks=self.tasks)
        if parsed_result["status"] == "success":
            assignments = parsed_result["assignments"] 
            print("Tasks successfully assigned")
            for task_assignments in assignments:
                print(f"{task_assignments['task_id']} assigned to {task_assignments['assigned_to']}")
                agent = next((a for a in self.agents if a.id == task_assignments["assigned_to"]), None)
                if agent:
                    # Find the corresponding task
                    task = next((t for t in self.tasks if t.id == task_assignments["task_id"]), None)
                    if task:
                        assignment = TaskAssignment(task, agent)
                        print(f"Assignment created for task {task.id} to {agent.id}")
                        self.taskAssignments.append(assignment)
                    else:
                        print(f"Task with id {task_assignments['task_id']} not found.")
                else:
                    print(f"Agent with id {task_assignments['assigned_to']} not found.")

    def negotiate(self):
        """Execute the negotiation process by running assigned tasks"""
        self.distribute_task()
        
        print(f"\n🚀 Starting negotiation with {len(self.taskAssignments)} task assignments")
        print("=" * 60)
        
        for i, task_assignment in enumerate(self.taskAssignments, 1):
            print(f"\n📋 Executing Task {i}: {task_assignment.task.task}")
            print(f"🤖 Agent: {task_assignment.agent.id}")
            print("-" * 40)
            
            try:
                # Execute the task using the agent's run method
                result = task_assignment.agent.run(task_assignment.task.task)
                
                print(f"✅ Task Result: {result['status']}")
                if result['status'] == 'completed':
                    print(f"📝 Message: {result.get('message', 'Task completed successfully')}")
                elif result['status'] == 'input_required':
                    print(f"❓ Required: {result.get('message', 'Additional input needed')}")
                else:
                    print(f"❌ Error: {result.get('message', 'Task failed')}")
                    
            except Exception as e:
                print(f"❌ Exception during task execution: {str(e)}")
            
            print("-" * 40)
        
        print(f"\n🎯 Negotiation completed with {len(self.taskAssignments)} tasks executed")
