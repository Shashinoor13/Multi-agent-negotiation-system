# negotiation_environment.py
from agents.base import Agent
import time
from threading import Thread

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

        yield f"\n🎯 Original Task: {task}"
        yield "=" * 80
        
        llm = LLMService()
        raw_llm_response=llm.split_task(task=task)
        
        parsed_result = llm.parse_output(raw_llm_response)
        
        if parsed_result["status"] == "success":
            tasks = parsed_result["tasks"] 
            yield "✅ Tasks successfully split and parsed"
            yield f"📊 Total subtasks created: {len(tasks)}"
            yield "-" * 60
            
            for task_item in tasks:
                yield f"📋 Subtask {task_item['id']}: {task_item['subtask']}"
                self.tasks.append(Task(task_item['id'],task_item['subtask']))
            yield "-" * 60
        else:
            yield f"❌ Error splitting or parsing task: {parsed_result['message']}"
            yield f"Raw response: {parsed_result.get('raw_response', 'N/A')}"
    
    def distribute_task(self):
        """Distribute tasks to agents with improved initial assignment logic"""
        yield "🔄 Phase 0: Initial Task Distribution with Cross-Evaluation"
        yield "-" * 60
        
        # First, let's evaluate all possible task-agent combinations
        all_evaluations = {}
        
        yield "🔍 Evaluating all possible task-agent combinations..."
        for task in self.tasks:
            all_evaluations[task.id] = {}
            for agent in self.agents:
                evaluation = agent.evaluate(task.task)
                all_evaluations[task.id][agent.id] = evaluation
                yield f"   Task {task.id} + Agent {agent.id}: confidence {evaluation.get('confidence', 'N/A')}"
        
        # Find optimal assignments using greedy approach
        yield "\n🎯 Finding optimal initial assignments..."
        optimal_assignments = []
        used_agents = set()
        
        # Sort tasks by complexity (lower confidence = more complex)
        task_complexity = []
        for task_id, evaluations in all_evaluations.items():
            avg_confidence = sum(eval.get('confidence', 0.5) for eval in evaluations.values()) / len(evaluations)
            task_complexity.append((task_id, avg_confidence))
        
        # Sort by complexity (most complex first)
        task_complexity.sort(key=lambda x: x[1])  # Lower confidence = more complex
        
        for task_id, _ in task_complexity:
            task = next(t for t in self.tasks if t.id == task_id)
            evaluations = all_evaluations[task_id]
            
            # Find best available agent for this task
            best_agent = None
            best_confidence = -1
            
            for agent_id, evaluation in evaluations.items():
                if agent_id not in used_agents:
                    confidence = evaluation.get('confidence', 0.0)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_agent = next(a for a in self.agents if a.id == agent_id)
            
            if best_agent:
                optimal_assignments.append({
                    'task_id': task_id,
                    'assigned_to': best_agent.id,
                    'confidence': best_confidence
                })
                used_agents.add(best_agent.id)
                yield f"   ✅ Task {task_id} → Agent {best_agent.id} (confidence: {best_confidence:.2f})"
            else:
                # If no optimal agent available, use LLM-based assignment
                yield f"   ⚠️  No optimal agent available for Task {task_id}, using LLM assignment"
                from services.llm_service import LLMService
                llm = LLMService()
                parsed_result = llm.select_agents_according_to_task(agents=self.agents, tasks=[task])
                if parsed_result["status"] == "success" and parsed_result["assignments"]:
                    assignment = parsed_result["assignments"][0]
                    optimal_assignments.append(assignment)
                    yield f"   ✅ Task {task_id} → Agent {assignment['assigned_to']} (LLM assigned)"
        
        # Create task assignments
        yield "\n📋 Creating task assignments..."
        for assignment_info in optimal_assignments:
            task_id = assignment_info['task_id']
            agent_id = assignment_info['assigned_to']
            
            # Find the task and agent
            task = next((t for t in self.tasks if t.id == task_id), None)
            agent = next((a for a in self.agents if a.id == agent_id), None)
            
            if task and agent:
                assignment = TaskAssignment(task, agent)
                self.taskAssignments.append(assignment)
                
                # Show assignment details
                task_content = task.task
                confidence = assignment_info.get('confidence', 'N/A')
                yield f"📋 Task {task_id} → {agent_id}"
                yield f"   Content: {task_content}"
                yield f"   Confidence: {confidence}"
                yield f"   ✅ Assignment created successfully"
            else:
                yield f"   ❌ Failed to create assignment for Task {task_id} → Agent {agent_id}"
        
        yield "-" * 60
        yield f"✅ Initial distribution complete: {len(self.taskAssignments)} assignments created"



    def negotiate(self):
        """Execute the negotiation process with agent-to-agent communication"""
        yield from self.distribute_task() # Use yield from for nested generator
        
        yield f"\n🚀 Starting negotiation with {len(self.taskAssignments)} task assignments"
        yield "=" * 80
        
        # Phase 1: Initial evaluation and capability assessment
        yield "\n🔍 Phase 1: Agent Capability Assessment"
        yield "-" * 50
        
        task_evaluations = {}
        agent_capabilities = {}
        
        # Each agent evaluates all tasks to understand capabilities
        for task_assignment in self.taskAssignments:
            agent = task_assignment.agent
            task = task_assignment.task
            
            # Agent evaluates its assigned task
            evaluation = agent.evaluate(task.task)
            task_evaluations[task.id] = {
                'assigned_agent': agent.id,
                'evaluation': evaluation,
                'task': task
            }
            
            # Track agent capabilities
            if agent.id not in agent_capabilities:
                agent_capabilities[agent.id] = []
            agent_capabilities[agent.id].append({
                'task_id': task.id,
                'confidence': evaluation.get('confidence', 0.5),
                'estimated_time': evaluation.get('estimated_time', 'unknown'),
                'requirements': evaluation.get('requirements', [])
            })
            
            yield f"🤖 Agent {agent.id} evaluated Task {task.id}: " \
                  f"Confidence: {evaluation.get('confidence', 'N/A')}"
        
        # Phase 2: Cross-agent evaluation for optimization
        yield f"\n🔄 Phase 2: Cross-Agent Evaluation & Negotiation"
        yield "-" * 50
        
        # Let each agent evaluate tasks assigned to other agents
        optimization_suggestions = []
        
        for task_id, task_info in task_evaluations.items():
            current_agent = task_info['assigned_agent']
            task_desc = task_info['task'].task
            current_confidence = task_info['evaluation'].get('confidence', 0.5)
            
            yield f"\n📋 Evaluating Task {task_id}: '{task_desc}'"
            yield f"   Current assignee: Agent {current_agent} (confidence: {current_confidence})"
            
            # Other agents evaluate this task
            better_alternatives = []
            for other_assignment in self.taskAssignments:
                other_agent = other_assignment.agent
                if other_agent.id != current_agent:
                    alt_evaluation = other_agent.evaluate(task_desc)
                    alt_confidence = alt_evaluation.get('confidence', 0.0)
                    
                    yield f"   🔍 Agent {other_agent.id} evaluation: confidence {alt_confidence}"
                    
                    # More flexible improvement criteria
                    improvement_threshold = 0.1  # Reduced from 0.2 to 0.1 (10% improvement)
                    
                    # Consider reassignment if:
                    # 1. Another agent has significantly higher confidence
                    # 2. Current agent has very low confidence (< 0.4) and another agent has medium+ confidence (> 0.6)
                    # 3. Another agent has at least 15% higher confidence
                    
                    should_reassign = False
                    reason = ""
                    
                    if alt_confidence > current_confidence + improvement_threshold:
                        should_reassign = True
                        reason = f"Significant confidence improvement (+{alt_confidence - current_confidence:.2f})"
                    elif current_confidence < 0.4 and alt_confidence > 0.6:
                        should_reassign = True
                        reason = f"Current agent struggling (confidence: {current_confidence:.2f}), alternative much better ({alt_confidence:.2f})"
                    elif alt_confidence > current_confidence * 1.15:  # 15% relative improvement
                        should_reassign = True
                        reason = f"Relative improvement of {((alt_confidence/current_confidence)-1)*100:.1f}%"
                    
                    if should_reassign:
                        better_alternatives.append({
                            'agent_id': other_agent.id,
                            'confidence': alt_confidence,
                            'evaluation': alt_evaluation,
                            'improvement': alt_confidence - current_confidence,
                            'relative_improvement': (alt_confidence / current_confidence) if current_confidence > 0 else float('inf'),
                            'reason': reason
                        })
            
            if better_alternatives:
                # Sort by multiple criteria: improvement, relative improvement, and confidence
                better_alternatives.sort(key=lambda x: (
                    x['improvement'],  # Absolute improvement
                    x['relative_improvement'],  # Relative improvement
                    x['confidence']  # Final confidence
                ), reverse=True)
                
                best_alternative = better_alternatives[0]
                
                optimization_suggestions.append({
                    'task_id': task_id,
                    'current_agent': current_agent,
                    'suggested_agent': best_alternative['agent_id'],
                    'improvement': best_alternative['improvement'],
                    'relative_improvement': best_alternative['relative_improvement'],
                    'reason': best_alternative['reason'],
                    'current_confidence': current_confidence,
                    'suggested_confidence': best_alternative['confidence']
                })
                
                yield f"   💡 Suggestion: Reassign to Agent {best_alternative['agent_id']}"
                yield f"      Confidence: {current_confidence:.2f} → {best_alternative['confidence']:.2f}"
                yield f"      Improvement: +{best_alternative['improvement']:.2f} ({best_alternative['reason']})"
                
                # Show other good alternatives too
                if len(better_alternatives) > 1:
                    yield f"      Other alternatives:"
                    for alt in better_alternatives[1:3]:  # Show top 3 alternatives
                        yield f"        - Agent {alt['agent_id']}: {alt['confidence']:.2f} (+{alt['improvement']:.2f})"
            else:
                # Even if no better alternatives, show all agent evaluations for transparency
                yield f"   ✅ Current assignment appears optimal"
                yield f"      All agent evaluations:"
                for other_assignment in self.taskAssignments:
                    other_agent = other_assignment.agent
                    if other_agent.id != current_agent:
                        alt_evaluation = other_agent.evaluate(task_desc)
                        alt_confidence = alt_evaluation.get('confidence', 0.0)
                        diff = alt_confidence - current_confidence
                        status = "✅ Better" if diff > 0 else "❌ Worse" if diff < 0 else "🟡 Equal"
                        yield f"        - Agent {other_agent.id}: {alt_confidence:.2f} ({diff:+.2f}) {status}"
                
                # Still suggest alternatives if they're close (within 5%)
                close_alternatives = []
                for other_assignment in self.taskAssignments:
                    other_agent = other_assignment.agent
                    if other_agent.id != current_agent:
                        alt_evaluation = other_agent.evaluate(task_desc)
                        alt_confidence = alt_evaluation.get('confidence', 0.0)
                        
                        # If alternative is within 5% of current, suggest it as an option
                        if abs(alt_confidence - current_confidence) <= 0.05 and alt_confidence > 0.6:
                            close_alternatives.append({
                                'agent_id': other_agent.id,
                                'confidence': alt_confidence,
                                'difference': alt_confidence - current_confidence
                            })
                
                if close_alternatives:
                    yield f"      💡 Close alternatives (within 5%):"
                    for alt in close_alternatives:
                        yield f"        - Agent {alt['agent_id']}: {alt['confidence']:.2f} ({alt['difference']:+.2f})"
                        yield f"          Consider this agent as an alternative option"
        
        # Phase 3: Negotiation and consensus
        yield f"\n🤝 Phase 3: Agent Negotiation & Consensus"
        yield "-" * 50
        
        if optimization_suggestions:
            yield f"Found {len(optimization_suggestions)} optimization opportunities:"
            
            # Simulate negotiation process with more sophisticated logic
            accepted_changes = []
            for suggestion in optimization_suggestions:
                yield f"\n🎯 Negotiating Task {suggestion['task_id']} reassignment:"
                yield f"   From: Agent {suggestion['current_agent']} (confidence: {suggestion['current_confidence']:.2f})"
                yield f"   To: Agent {suggestion['suggested_agent']} (confidence: {suggestion['suggested_confidence']:.2f})"
                yield f"   Reason: {suggestion['reason']}"
                
                # Find the agents involved
                current_agent = None
                suggested_agent = None
                
                for assignment in self.taskAssignments:
                    if assignment.agent.id == suggestion['current_agent']:
                        current_agent = assignment.agent
                    elif assignment.agent.id == suggestion['suggested_agent']:
                        suggested_agent = assignment.agent
                
                # More sophisticated negotiation logic
                if current_agent and suggested_agent:
                    # Current agent's response to giving up the task
                    current_response = current_agent.evaluate(f"Release task {suggestion['task_id']} to another agent")
                    current_willingness = current_response.get('confidence', 0.5)
                    
                    # Suggested agent's response to taking on the task
                    suggested_response = suggested_agent.evaluate(f"Take on additional task {suggestion['task_id']}")
                    suggested_willingness = suggested_response.get('confidence', 0.7)
                    
                    yield f"   🤖 Agent {suggestion['current_agent']} willingness to release: {current_willingness:.2f}"
                    yield f"   🤖 Agent {suggestion['suggested_agent']} willingness to accept: {suggested_willingness:.2f}"
                    
                    # More flexible acceptance criteria
                    # Accept if:
                    # 1. Both agents are willing (current >= 0.3, suggested >= 0.6)
                    # 2. OR if improvement is very significant (> 0.3) and suggested agent is willing
                    # 3. OR if current agent is struggling (< 0.4) and suggested agent is confident (> 0.7)
                    
                    improvement_significant = suggestion['improvement'] > 0.3
                    current_struggling = suggestion['current_confidence'] < 0.4
                    suggested_confident = suggestion['suggested_confidence'] > 0.7
                    
                    accept_change = False
                    accept_reason = ""
                    
                    if current_willingness >= 0.3 and suggested_willingness >= 0.6:
                        accept_change = True
                        accept_reason = "Both agents agree"
                    elif improvement_significant and suggested_willingness >= 0.5:
                        accept_change = True
                        accept_reason = f"Significant improvement ({suggestion['improvement']:.2f}) and willing agent"
                    elif current_struggling and suggested_confident and suggested_willingness >= 0.4:
                        accept_change = True
                        accept_reason = "Current agent struggling, suggested agent confident"
                    
                    if accept_change:
                        accepted_changes.append(suggestion)
                        yield f"   ✅ Agreement reached! {accept_reason}"
                        yield f"   🔄 Task will be reassigned."
                    else:
                        yield f"   ❌ No consensus reached. Task remains with current agent."
                        yield f"      Reason: Current willingness {current_willingness:.2f}, Suggested willingness {suggested_willingness:.2f}"
            
            # Apply accepted changes
            if accepted_changes:
                yield f"\n🔄 Applying {len(accepted_changes)} agreed changes:"
                for change in accepted_changes:
                    # Find and update task assignment
                    for assignment in self.taskAssignments:
                        if assignment.task.id == change['task_id']:
                            # Find the new agent
                            for other_assignment in self.taskAssignments:
                                if other_assignment.agent.id == change['suggested_agent']:
                                    assignment.agent = other_assignment.agent
                                    yield f"   ✅ Task {change['task_id']} reassigned to Agent {change['suggested_agent']}"
                                    yield f"      Confidence improved: {change['current_confidence']:.2f} → {change['suggested_confidence']:.2f}"
                                    break
                            break
        else:
            yield "No optimization opportunities found. Current assignments are optimal."
        
        # Phase 4: Collaborative execution
        yield f"\n🚀 Phase 4: Collaborative Task Execution"
        yield "=" * 80
        
        def execute_task_assignment(task_assignment, message_queue):
            try:
                message_queue.append(f"\n📋 Executing Task {task_assignment.task.id}:")
                message_queue.append(f"🤖 Agent: {task_assignment.agent.id}")
                message_queue.append(f"📝 Task: {task_assignment.task.task}")
                message_queue.append("-" * 60)
                
                # Pre-execution evaluation for final check
                pre_eval = task_assignment.agent.evaluate(task_assignment.task.task)
                message_queue.append(f"🔍 Pre-execution confidence: {pre_eval.get('confidence', 'N/A')}")
                
                time.sleep(0.5)  # simulate preparation delay
                result = task_assignment.agent.run(task_assignment.task.task,message_queue)
                
                message_queue.append(f"✅ Task {task_assignment.task.id} Result: {result['status']}")
                if result['status'] == 'completed':
                    message_queue.append(f"📝 Message: {result.get('message', 'Task completed successfully')}")
                elif result['status'] == 'input_required':
                    message_queue.append(f"❓ Required: {result.get('message', 'Additional input needed')}")
                else:
                    message_queue.append(f"❌ Error: {result.get('message', 'Task failed')}")
                
                message_queue.append("-" * 60)
                return result
                
            except Exception as e:
                message_queue.append(f"❌ Exception during task execution: {str(e)}")
                message_queue.append("-" * 60)
                return {'status': 'error', 'message': str(e)}
        
        # Execute tasks with improved assignments
        results = []
        threads = []
        
        # Use a list to collect messages from threads
        thread_messages = [] 
        
        for task_assignment in self.taskAssignments:
            thread = Thread(target=lambda ta=task_assignment: results.append(execute_task_assignment(ta, thread_messages)))
            threads.append(thread)
            thread.start()
            time.sleep(0.3)  # slight delay between task launches
        
        for thread in threads:
            thread.join()

        # Yield messages collected from threads
        for msg in thread_messages:
            yield msg

        # Phase 5: Post-execution analysis
        yield f"\n📊 Phase 5: Negotiation Results Analysis"
        yield "-" * 50
        
        successful_tasks = sum(1 for r in results if r and r.get('status') == 'completed')
        total_tasks = len(self.taskAssignments)
        
        yield f"📈 Success Rate: {successful_tasks}/{total_tasks} ({successful_tasks/total_tasks*100:.1f}%)"
        yield f"🔄 Optimizations Applied: {len(accepted_changes) if 'accepted_changes' in locals() else 0}"
        yield f"💡 Suggestions Generated: {len(optimization_suggestions) if 'optimization_suggestions' in locals() else 0}"
        
        yield f"\n🎯 Negotiation completed with agent collaboration!"
        yield "=" * 80
        
        # Return the final result dictionary (optional, but good for structured output)
        yield {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'optimizations_applied': len(accepted_changes) if 'accepted_changes' in locals() else 0,
            'results': results
        }