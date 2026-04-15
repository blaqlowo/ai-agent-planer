import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Agent:
    def __init__(self):
        # Read from environment
        api_key = os.getenv("OPENAI_API_KEY", "ollama")
        base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "llama3")

        # Initialize the OpenAI client with potentially custom base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.memory = []

    def get_completion(self, system_prompt: str, user_prompt: str, is_json=False) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        args = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
        }
        
        # We try to use JSON format if requested, but some local models might not fully support Native JSON mode.
        if is_json:
            args["response_format"] = {"type": "json_object"}
            
        try:
            response = self.client.chat.completions.create(**args)
            return response.choices[0].message.content
        except Exception as e:
            return f"Error connecting to local model: {e}"

    def generate_plan(self, goal: str) -> list:
        system_prompt = (
            "You are an expert AI orchestrator. "
            "Given a goal, break it down into a sequence of actionable steps. "
            "You MUST return a valid JSON object with a single key 'steps' which contains a list of strings. "
            "Example: {\"steps\": [\"Step 1 description\", \"Step 2 description\"]}"
        )
        content = self.get_completion(system_prompt, goal, is_json=True)
        
        # Fallback manual parsing if local model ignores JSON format
        try:
            data = json.loads(content)
            return data.get("steps", [])
        except json.JSONDecodeError:
            # Try to extract JSON manually
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    return data.get("steps", [])
                except:
                    pass
            # If all fails, return a single step string as list
            return [content]

    def execute_step(self, step: str, goal: str) -> str:
        # Build context from memory
        context = "\n".join([f"- {m['step']}: {m['result'][:100]}..." for m in self.memory])
        
        system_prompt = (
            "You are an AI execution agent. "
            f"Your overarching goal is: {goal}\n"
            "Here is what has been done so far:\n"
            f"{context}\n\n"
            "Now, execute the next step to the best of your ability, outputting a detailed result."
        )
        
        result = self.get_completion(system_prompt, step)
        
        # Store in memory
        self.memory.append({"step": step, "result": result})
        return result

    def run_generator(self, goal: str):
        yield f"data: {json.dumps({'status': 'planning', 'message': 'Thinking about the goal...'})}\n\n"
        
        steps = self.generate_plan(goal)
        if not isinstance(steps, list):
            steps = [str(steps)]
            
        yield f"data: {json.dumps({'status': 'plan_ready', 'steps': steps})}\n\n"
        
        for i, step in enumerate(steps):
            yield f"data: {json.dumps({'status': 'executing', 'step_index': i, 'step': step})}\n\n"
            
            result = self.execute_step(step, goal)
            
            yield f"data: {json.dumps({'status': 'step_done', 'step_index': i, 'result': result})}\n\n"
            
        yield f"data: {json.dumps({'status': 'complete', 'message': 'All tasks finished!'})}\n\n"
