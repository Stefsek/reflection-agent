"""
Simple example to test the ReflectionAgent
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from core.reflection_agent import ReflectionAgent

# Initialize the agent
agent = ReflectionAgent(
    model_name="gemini-3-flash-preview",
    temperature=0.2,
    max_iterations=3
)

# User input
user_input = """
Create a prompt for a chatbot that helps users troubleshoot Wi-Fi connection issues
"""

print(f"\nUser Input: {user_input}")
print("Running reflection agent...")

# Run the agent
result = agent.run(user_input)

print("Done")

# Print all generations
print("GENERATIONS:")
print("-" * 80)
for i, gen in enumerate(result['generations'], 1):
    print(f"\nGeneration {i}:")
    print(f"Prompt: {gen['generated_prompt']}")
    print(f"Reasoning: {gen['reasoning']}")
    print()

# Print all reflections
print("\nREFLECTIONS:")
print("-" * 80)
for i, ref in enumerate(result['reflections'], 1):
    print(f"\nReflection {i}:")
    print(f"Critique: {ref['critique']}")
    print(f"Suggestions: {ref['suggestions']}")
    print()

# Print token usage
print("\nTOKEN USAGE:")
print("-" * 80)
print(f"Generation tokens: {result['generation_tokens']}")
print(f"Reflection tokens: {result['reflection_tokens']}")
print(f"Total tokens: {result['total_tokens']}")
