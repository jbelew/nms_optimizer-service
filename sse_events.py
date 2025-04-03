# sse_events.py
import json

def sse_message(message: str, event=None) -> str:
    """Formats a message as an SSE message.

    Args:
        message (str): The message string to send.
        event (str, optional): The event type. Defaults to None.

    Returns:
        str: The formatted SSE message.
    """
    data = {"event": message}  # Wrap the message in the {"event": ...} structure
    msg = f"data: {json.dumps(data)}\n"
    if event is not None:
        msg = f"event: {event}\n{msg}"
    msg += "\n"  # Add an extra new line to separate messages
    return msg
