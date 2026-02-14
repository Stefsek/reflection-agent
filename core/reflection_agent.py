"""LangGraph-based reflection agent for iterative prompt generation and refinement.

This module implements a multi-agent system that iteratively generates and refines
prompts through a reflection loop. The agent uses two nodes:
- Generation node: Creates prompts based on user input and previous feedback
- Reflection node: Analyzes generated prompts and provides constructive critique

Typical usage example:

    agent = ReflectionAgent(model_name="gemini-2.0-flash-exp", max_iterations=3)
    result = agent.run("Create a prompt for email validation")
    print(result['generations'][-1])
"""

from typing import Annotated, List, Dict, Any, Tuple
from typing_extensions import TypedDict
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from utils.token_utils import TokenUsageUtils
from prompts import GENERATION_SYSTEM_MESSAGE, REFLECTION_SYSTEM_MESSAGE, PromptFormatter
from schemas import GenerationOutput, ReflectionOutput

# Load environment variables from .env file
load_dotenv()


class AgentState(TypedDict):
    """State container for the reflection agent graph.

    Attributes:
        messages: List of conversation messages with automatic message merging.
        iterations: Current iteration count in the reflection loop.
        max_iterations: Maximum number of reflection iterations allowed.
        token_usage: Overall token usage across all nodes.
        generation_tokens: Token usage specific to generation node.
        reflection_tokens: Token usage specific to reflection node.
        generations: List of all generated prompt outputs.
        reflections: List of all reflection outputs.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    iterations: int
    max_iterations: int
    token_usage: Dict[str, Any]
    generation_tokens: Dict[str, Any]
    reflection_tokens: Dict[str, Any]
    generations: List[Dict[str, Any]]
    reflections: List[Dict[str, Any]]


class ReflectionAgent:
    """Reflection agent that iteratively generates and refines prompts.

    This agent uses a two-node LangGraph workflow to create high-quality prompts
    through iterative refinement. The generation node creates prompts, and the
    reflection node provides critique and suggestions for improvement.

    Attributes:
        model_name: Name of the Gemini model being used.
        max_iterations: Maximum number of reflection iterations.
        chat_model: Configured ChatGoogleGenerativeAI instance.
        graph: Compiled LangGraph workflow.
    """

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        temperature: float = 0.2,
        max_iterations: int = 3
    ):
        """Initializes the reflection agent.

        Args:
            model_name: Name of the Gemini model to use (e.g., "gemini-2.0-flash-exp").
            temperature: Temperature parameter for model generation (0.0-1.0).
                Lower values make output more deterministic.
            max_iterations: Maximum number of reflection iterations before terminating.
        """
        self.model_name = model_name
        self.max_iterations = max_iterations

        # Initialize the chat model (uses GOOGLE_API_KEY from .env)
        self.chat_model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        )
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Builds the LangGraph workflow.

        Creates a state graph with two nodes (generate_prompt and reflect_prompt)
        connected in a loop. The generation node produces prompts, which are then
        critiqued by the reflection node. The loop continues until max_iterations
        is reached.

        Returns:
            Compiled StateGraph ready for execution.
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("generate_prompt", self._generation_node)
        workflow.add_node("reflect_prompt", self._reflection_node)

        # Set entry point
        workflow.set_entry_point("generate_prompt")

        # Add edges
        workflow.add_conditional_edges(
            "generate_prompt",
            self.should_continue,
            {
                "reflect_prompt": "reflect_prompt",
                END: END
            }
        )
        workflow.add_edge("reflect_prompt", "generate_prompt")

        return workflow.compile()

    def _generation_node(self, state: AgentState) -> Dict[str, Any]:
        """Generates a prompt based on user input and previous reflections.

        This node takes the conversation history (including any reflection feedback)
        and generates a new or improved prompt. Token usage is tracked and aggregated.

        Args:
            state: Current agent state containing messages and token counters.

        Returns:
            Dictionary containing updated messages, token usage, and generation history.
        """

        conversation_messages = [SystemMessage(content=GENERATION_SYSTEM_MESSAGE)]
        conversation_messages.extend(state.get('messages'))

        # Call model with tracking
        generation_output, node_token_usage = self._call_model_with_tracking(
            conversation_messages, GenerationOutput
        )

        formatted_output = PromptFormatter.format_generation_output(
            generated_prompt=generation_output.generated_prompt,
            reasoning=generation_output.reasoning
        )

        # Create AI message with the generated prompt
        ai_message = AIMessage(content=formatted_output)

        # Update token usage
        updated_total_usage = self._update_token_usage(state.get("token_usage"), node_token_usage)
        updated_generation_tokens = self._update_token_usage(state.get("generation_tokens"), node_token_usage)

        # Append generation output to history
        current_generations = state.get("generations")
        updated_generations = current_generations + [generation_output.model_dump()]

        return {
            "messages": [ai_message],
            "token_usage": updated_total_usage,
            "generation_tokens": updated_generation_tokens,
            "generations": updated_generations
        }

    def _reflection_node(self, state: AgentState) -> Dict[str, Any]:
        """Reflects on the generated prompt and provides critique.

        This node analyzes the most recent prompt generation and provides constructive
        feedback including critique and specific suggestions for improvement. The
        feedback is formatted as a HumanMessage to guide the next generation iteration.

        Args:
            state: Current agent state containing messages and token counters.

        Returns:
            Dictionary containing updated messages, token usage, reflection history,
            and incremented iteration count.
        """

        conversation_messages = [SystemMessage(content=REFLECTION_SYSTEM_MESSAGE)]
        conversation_messages.extend(state.get('messages'))

        # Call model with tracking
        reflection_output, node_token_usage = self._call_model_with_tracking(
            conversation_messages, ReflectionOutput
        )

        # Create HumanMessage with reflection feedback using the template
        formatted_feedback = PromptFormatter.format_reflection_output(
            critique=reflection_output.critique,
            suggestions=reflection_output.suggestions
        )
        human_message = HumanMessage(content=formatted_feedback)

        # Update token usage
        updated_total_usage = self._update_token_usage(state.get("token_usage"), node_token_usage)
        updated_reflection_tokens = self._update_token_usage(state.get("reflection_tokens"), node_token_usage)

        # Increment iterations
        updated_iterations = state["iterations"] + 1

        # Append reflection output to history
        current_reflections = state.get("reflections", [])
        updated_reflections = current_reflections + [reflection_output.model_dump()]

        return {
            "messages": [human_message],
            "iterations": updated_iterations,
            "token_usage": updated_total_usage,
            "reflection_tokens": updated_reflection_tokens,
            "reflections": updated_reflections
        }


    def _call_model_with_tracking(
        self,
        messages: List[BaseMessage],
        output_schema: type[BaseModel]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Calls the model with structured output and tracks token usage.

        Wraps the model invocation with LangChain's usage metadata callback to
        automatically track input/output tokens and aggregate across multiple
        model calls.

        Args:
            messages: List of messages to send to the model.
            output_schema: Pydantic schema defining the expected structured output.

        Returns:
            A tuple containing:
                - Structured output matching the provided schema
                - Token usage dictionary with keys: input_tokens, output_tokens,
                  total_tokens, successful_requests, usage_metadata
        """
        with get_usage_metadata_callback() as usage_callback:
            structured_llm = self.chat_model.with_structured_output(output_schema)
            model_output = structured_llm.invoke(messages)

            # Extract usage metadata
            callback_metadata = usage_callback.usage_metadata

            # Aggregate tokens across all models/providers
            total_input_tokens = TokenUsageUtils.sum_tokens(callback_metadata, "input_tokens")
            total_output_tokens = TokenUsageUtils.sum_tokens(callback_metadata, "output_tokens")
            total_tokens = TokenUsageUtils.sum_tokens(callback_metadata, "total_tokens")

            token_usage_summary = {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "successful_requests": len(callback_metadata),
                "usage_metadata": callback_metadata,
            }

            return model_output, token_usage_summary

    def _update_token_usage(
        self,
        current_tokens: Dict[str, Any],
        new_tokens: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Updates token usage by adding new tokens to current totals.

        Args:
            current_tokens: Current token usage dictionary containing input_tokens,
                output_tokens, total_tokens, and successful_requests.
            new_tokens: New token usage to add with the same structure.

        Returns:
            Updated token usage dictionary with accumulated totals.
        """
        return {
            "input_tokens": current_tokens["input_tokens"] + new_tokens["input_tokens"],
            "output_tokens": current_tokens["output_tokens"] + new_tokens["output_tokens"],
            "total_tokens": current_tokens["total_tokens"] + new_tokens["total_tokens"],
            "successful_requests": current_tokens["successful_requests"] + new_tokens["successful_requests"],
        }


    def should_continue(self, state: AgentState) -> str:
        """Determines whether to continue reflection loop or terminate.

        Args:
            state: Current agent state containing iteration counters.

        Returns:
            Either "reflect_prompt" to continue the loop or END to terminate.
        """
        # Check if we've reached max iterations
        if state["iterations"] >= state["max_iterations"]:
            return END

        return "reflect_prompt"

    

    def run(self, user_input: str, max_iterations: int = None) -> Dict[str, Any]:
        """Runs the reflection agent on user input.

        Executes the full reflection loop, generating and refining prompts based
        on the user's request. The loop continues until max_iterations is reached.

        Args:
            user_input: The user's request describing what kind of prompt to create.
            max_iterations: Optional override for the default max_iterations set
                during initialization.

        Returns:
            Dictionary containing:
                - messages: Final conversation history
                - iterations: Number of iterations completed
                - generation_tokens: Token usage from generation nodes
                - reflection_tokens: Token usage from reflection nodes
                - total_tokens: Overall token usage
                - generations: List of all generated prompts with reasoning
                - reflections: List of all reflection critiques and suggestions
        """
        starting_state = {
            "messages": [HumanMessage(content=user_input)],
            "iterations": 0,
            "max_iterations": max_iterations or self.max_iterations,
            "token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "successful_requests": 0
            },
            "generation_tokens": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "successful_requests": 0
            },
            "reflection_tokens": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "successful_requests": 0
            },
            "generations": [],
            "reflections": []
        }

        # Run the graph
        completed_state = self.graph.invoke(starting_state)

        return {
            "messages": completed_state["messages"],
            "iterations": completed_state["iterations"],
            "generation_tokens": completed_state["generation_tokens"],
            "reflection_tokens": completed_state["reflection_tokens"],
            "total_tokens": completed_state["token_usage"],
            "generations": completed_state["generations"],
            "reflections": completed_state["reflections"]
        }

    def visualize_graph(self, save_path: str = "reflection_agent_graph.png") -> None:
        """Generates and saves a graph visualization.

        Creates a Mermaid diagram showing the workflow structure (nodes and edges)
        and saves it as a PNG image file.

        Args:
            save_path: Path where the PNG file will be saved. Defaults to
                "reflection_agent_graph.png" in the current directory.

        Raises:
            Exception: If graph visualization generation fails (e.g., missing
                dependencies or write permission issues).
        """
        try:
            graph_png_bytes = self.graph.get_graph().draw_mermaid_png()

            with open(save_path, "wb") as output_file:
                output_file.write(graph_png_bytes)

            print(f"Graph visualization saved to {save_path}")

        except Exception as error:
            print(f"Failed to generate graph visualization: {error}")
