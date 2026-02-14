"""
visualize.py
------------
Visualize the reflection agent graph structure.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import ReflectionAgent


# Initialize agent (no API key needed for visualization)
agent = ReflectionAgent()

# Generate visualization
agent.visualize_graph(save_path="reflection_agent_graph.png")
