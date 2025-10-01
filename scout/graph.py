from pydantic import BaseModel
from typing import Annotated, List, Generator
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessageChunk
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver
from scout.tools import query_db, generate_visualization
from scout.prompts import prompts
from scout.env import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, DATABASE_URI

# Global flag to track if Groq config has been logged
_groq_config_logged = False


def get_groq_chat_model() -> ChatGroq:
    """Get the main Groq chat model for complex tasks."""
    api_key = GROQ_API_KEY
    model = GROQ_MODEL
    temperature = GROQ_TEMPERATURE

    # Validate API key
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Please set the GROQ_API_KEY environment variable.")

    # Only print config once during startup
    global _groq_config_logged
    if not _groq_config_logged:
        masked_key = (api_key[:4] + "...") if api_key else "<missing>"
        print("🔧 Groq Model Config:")
        print(f"   Model: {model}")
        print(f"   API Key: {masked_key}")
        print(f"   Temperature: {temperature}")
        _groq_config_logged = True

    # Create model with explicit parameters to avoid any configurable context issues
    return ChatGroq(
        model=model,         # note: use `model=...` per most Groq bindings
        temperature=temperature,
        api_key=api_key,
        # Explicitly disable any configurable features that might cause issues
        streaming=False,  # Disable streaming at model level to avoid context issues
        timeout=30.0,     # Set explicit timeout
    )


class  ScoutState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = []
    chart_json: str = ""


class Agent:
    """
    Agent class for implementing Langgraph agents.

    Attributes:
        name: The name of the agent.
        tools: The tools available to the agent.
        model: The model to use for the agent.
        system_prompt: The system prompt for the agent.
        temperature: The temperature for the agent.
    """
    def __init__(
            self, 
            name: str, 
            tools: List = [query_db, generate_visualization],
            model: str = None, 
            system_prompt: str = "You are a helpful assistant.",
            temperature: float = None
            ):
        self.name = name
        self.tools = tools
        self.model = model or GROQ_MODEL
        self.system_prompt = system_prompt
        self.temperature = temperature if temperature is not None else GROQ_TEMPERATURE
        
        self.llm = get_groq_chat_model().bind_tools(self.tools)
        
        self.runnable = self.build_graph()


    def build_graph(self):
        """
        Build the LangGraph application.
        """
        def scout_node(state: ScoutState) -> ScoutState:
            response = self.llm.invoke(
                [SystemMessage(content=self.system_prompt)] +
                state.messages
                )
            state.messages = state.messages + [response]
            return state
        
        def router(state: ScoutState) -> str:
            last_message = state.messages[-1]
            if not last_message.tool_calls:
                return END
            else:
                return "tools"

        builder = StateGraph(ScoutState)

        builder.add_node("chatbot", scout_node)
        builder.add_node("tools", ToolNode(self.tools))

        builder.add_edge(START, "chatbot")
        builder.add_conditional_edges("chatbot", router, ["tools", END])
        builder.add_edge("tools", "chatbot")

        # Use PostgreSQL checkpointer for persistence
        checkpointer = None
        if DATABASE_URI:
            try:
                checkpointer = PostgresSaver.from_conn_string(DATABASE_URI)
                checkpointer.setup()  # Create tables if they don't exist
                print("✅ Using PostgreSQL checkpointer for persistence")
            except Exception as e:
                print(f"⚠️  PostgreSQL checkpointer failed: {e}")
                print("   Falling back to MemorySaver (no persistence)")
                checkpointer = MemorySaver()
        else:
            print("⚠️  DATABASE_URI not set, using MemorySaver (no persistence)")
            checkpointer = MemorySaver()
        
        return builder.compile(checkpointer=checkpointer)
    

    def inspect_graph(self):
        """
        Visualize the graph using the mermaid.ink API.
        """
        from IPython.display import display, Image

        graph = self.build_graph()
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))


    def invoke(self, message: str, **kwargs) -> str:
        """Synchronously invoke the graph.

        Args:
            message: The user message.

        Returns:
            str: The LLM response.
        """
        result = self.runnable.invoke(
            input = {
                "messages": [HumanMessage(content=message)]
            },
            **kwargs
        )

        return result["messages"][-1].content
    

    def stream(self, message: str, **kwargs) -> Generator[str, None, None]:
        """Synchronously stream the results of the graph run.

        Args:
            message: The user message.

        Returns:
            str: The final LLM response or tool call response
        """
        for message_chunk, metadata in self.runnable.stream(
            input = {
                "messages": [HumanMessage(content=message)]
            },
            stream_mode="messages",
            **kwargs
        ):
            if isinstance(message_chunk, AIMessageChunk):
                if message_chunk.response_metadata:
                    finish_reason = message_chunk.response_metadata.get("finish_reason", "")
                    if finish_reason == "tool_calls":
                        yield "\n\n"

                if message_chunk.tool_call_chunks:
                    tool_chunk = message_chunk.tool_call_chunks[0]

                    tool_name = tool_chunk.get("name", "")
                    args = tool_chunk.get("args", "")

                    
                    if tool_name:
                        tool_call_str = f"\n\n< TOOL CALL: {tool_name} >\n\n"

                    if args:
                        tool_call_str = args
                    yield tool_call_str
                else:
                    yield message_chunk.content
                continue


# Define and instantiate the agent 
agent = Agent(
        name="Scout",
        system_prompt=prompts.scout_system_prompt
        )
graph = agent.build_graph()
