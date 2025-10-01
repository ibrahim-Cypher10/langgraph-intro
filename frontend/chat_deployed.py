import os
import json
import uuid
import httpx
from dotenv import load_dotenv
load_dotenv()


LANGGRAPH_SERVER_URL = os.getenv("LANGGRAPH_SERVER_URL")
# LANGGRAPH_SERVER_URL = https://langgraph-intro-35xv.onrender.com


async def create_thread(user_id: str) -> dict:
    """Create a new thread for the given user."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=f"{LANGGRAPH_SERVER_URL}/threads",
                json={
                    "thread_id": str(uuid.uuid4()),
                    "metadata": {
                        "user_id": user_id
                    },
                    "if_exists": "do_nothing"
                },
                timeout=120.0 # Added timeout to wait for Render spin up
            )
            response.raise_for_status()

            return response.json()
    except Exception as e:
        print(f"Request failed: {e}")
        raise


async def get_thread_state(thread_id: str) -> dict:
    """Get the state of the thread."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url=f"{LANGGRAPH_SERVER_URL}/threads/{thread_id}/state"
            )
            response.raise_for_status()

            return response.json()
    except Exception as e:
        print(f"Request failed: {e}")
        raise


async def get_thread_messages(thread_id: str) -> dict:
    """Get the message history of the thread."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url=f"{LANGGRAPH_SERVER_URL}/threads/{thread_id}/messages"
            )
            response.raise_for_status()

            return response.json()
    except Exception as e:
        print(f"Request failed: {e}")
        raise


async def save_thread_state(thread_id: str, state_data: dict) -> dict:
    """Save the current state of the thread."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=f"{LANGGRAPH_SERVER_URL}/threads/{thread_id}/save",
                json=state_data
            )
            response.raise_for_status()

            return response.json()
    except Exception as e:
        print(f"Request failed: {e}")
        raise


def process_line(line: str, current_event: str) -> str:
    """Process a single data line from the streaming response."""
    try:
        # Process data lines
        if line.startswith("data: "):
            data_content = line[6:]

            if current_event == "messages":
                message_chunk, metadata = json.loads(data_content)
            
                if "type" in message_chunk and message_chunk["type"] == "AIMessageChunk":
                    if message_chunk["response_metadata"]:
                        finish_reason = message_chunk["response_metadata"].get("finish_reason", "")
                        if finish_reason == "tool_calls":
                            return "\n\n"
                        
                    if message_chunk["tool_call_chunks"]:
                        tool_chunk = message_chunk["tool_call_chunks"][0]

                        tool_name = tool_chunk.get("name", "")
                        args = tool_chunk.get("args", "")
                        
                        if tool_name:
                            tool_call_str = f"\n\n< TOOL CALL: {tool_name} >\n\n"

                        if args:
                            tool_call_str = args
                        return tool_call_str
                    else:
                        return message_chunk["content"]
                
                # You can handle other event types here
                
            elif current_event == "metadata":
                return

    except Exception as e:
        print(f"Error processing line: {type(e).__name__}: {str(e)}")
        raise


async def get_stream(thread_id: str, message: str):
    """Send a message to the thread and process the streaming response.

    Args:
        thread_id: The thread ID to send the message to
        message: The message content
        seen_tool_call_ids: A set of tool call IDs that have already been seen

    Returns:
        str: The complete response from the assistant
    """
    full_content = ""
    current_event = None

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url=f"{LANGGRAPH_SERVER_URL}/threads/{thread_id}/runs/stream",
                json={
                    "assistant_id": "scout",
                    "input": {
                        "messages": [
                            {"role": "human", "content": message}
                        ]
                    },
                    "stream_mode": "messages-tuple"
                },
                timeout=60.0
            ) as stream_response:
                async for line in stream_response.aiter_lines():
                    if line:
                        # Process event lines
                        if line.startswith("event: "):
                            current_event = line[7:].strip()

                        # Process data lines
                        else:
                            message_chunk = process_line(line, current_event)
                            if message_chunk:
                                full_content += message_chunk
                                print(message_chunk, end="", flush=True)

        return full_content
    except Exception as e:
        print(f"Error in get_stream: {type(e).__name__}: {str(e)}")
        raise


async def main():
    try:
        # Create a thread
        response = await create_thread(user_id="kenny")
        thread_id = response["thread_id"]

        current_chart = None

        # Stream responses
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            elif user_input.lower() in ["history", "h"]:
                # Show thread history
                print(f"\n---- Thread History ---- \n")
                try:
                    messages = await get_thread_messages(thread_id)
                    if "messages" in messages:
                        for msg in messages["messages"]:
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            print(f"{role.upper()}: {content}\n")
                    else:
                        print("No message history found.")
                except Exception as e:
                    print(f"Error retrieving history: {e}")
                print("")
                continue
            elif user_input.lower() in ["save", "s"]:
                # Save current thread state
                print(f"\n---- Saving Thread State ---- \n")
                try:
                    thread_state = await get_thread_state(thread_id)
                    save_result = await save_thread_state(thread_id, thread_state)
                    print(f"Thread state saved successfully: {save_result}")
                except Exception as e:
                    print(f"Error saving thread state: {e}")
                print("")
                continue

            print(f"\n---- User ---- \n\n{user_input}\n")

            print(f"---- Assistant ---- \n")
            # Get the response using our simplified get_stream function
            result = await get_stream(thread_id, user_input)

            # check state after run
            thread_state = await get_thread_state(thread_id)

            if "chart_json" in thread_state["values"]:
                chart_json = thread_state["values"]["chart_json"]
                # render any new charts generated
                if chart_json and chart_json != current_chart:
                    import plotly.io as pio
                    fig = pio.from_json(chart_json)
                    fig.show()

                    current_chart = chart_json
            print("")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        raise


if __name__ == "__main__":
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    print(f"\nGreetings!\n\nTry asking Scout to show you a preview of the data.\n\nCommands:\n- Type 'history' or 'h' to view thread history\n- Type 'save' or 's' to save current thread state\n- Type 'exit' or 'quit' to end the session\n\n{'=' * 40}\n\n")

    asyncio.run(main())
