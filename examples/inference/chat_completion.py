#!/usr/bin/env python3
"""
Chat Completion API with TenSafe

This example demonstrates how to use TenSafe's chat completion API for
multi-turn conversations. The chat format maintains conversation context
and supports system prompts for customizing assistant behavior.

What this example demonstrates:
- Setting up chat-style inference
- Managing conversation history
- Using system prompts
- Handling multi-turn conversations
- Chat message roles and formatting

Key concepts:
- Messages: List of role-content pairs
- System prompt: Instructions for assistant behavior
- Conversation history: Context for coherent responses
- Chat templates: Model-specific formatting

Prerequisites:
- TenSafe server running
- Chat-tuned LoRA adapter

Expected Output:
    Chat session started

    System: You are a helpful coding assistant.

    User: How do I read a file in Python?
    Assistant: You can read a file in Python using...

    User: What about writing to a file?
    Assistant: To write to a file, you can use...

    Conversation tokens: 256
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""
    role: Literal["system", "user", "assistant"]
    content: str
    name: Optional[str] = None  # Optional name for multi-user chats


@dataclass
class ChatCompletionConfig:
    """Configuration for chat completions."""
    model: str = "meta-llama/Llama-3-8B"
    adapter: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stop_sequences: List[str] = field(default_factory=list)


class ChatSession:
    """Manage a chat conversation session."""

    def __init__(self, config: ChatCompletionConfig, system_prompt: Optional[str] = None):
        self.config = config
        self.messages: List[ChatMessage] = []
        self.total_tokens = 0

        if system_prompt:
            self.messages.append(ChatMessage(role="system", content=system_prompt))

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append(ChatMessage(role="assistant", content=content))

    def get_context_messages(self, max_messages: Optional[int] = None) -> List[ChatMessage]:
        """Get messages for context, optionally limiting history."""
        if max_messages is None:
            return self.messages.copy()

        # Always include system prompt if present
        if self.messages and self.messages[0].role == "system":
            system = [self.messages[0]]
            history = self.messages[1:]
        else:
            system = []
            history = self.messages

        # Get most recent messages
        recent = history[-(max_messages - len(system)):]
        return system + recent

    def clear_history(self, keep_system: bool = True) -> None:
        """Clear conversation history."""
        if keep_system and self.messages and self.messages[0].role == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []


def simulate_chat_response(messages: List[ChatMessage]) -> str:
    """Simulate a chat response based on conversation context."""
    last_message = messages[-1].content.lower()

    # Check for system prompt context
    system_context = ""
    if messages and messages[0].role == "system":
        system_context = messages[0].content.lower()

    # Simulated responses based on content
    if "read" in last_message and "file" in last_message:
        return """To read a file in Python, you can use the built-in `open()` function:

```python
# Read entire file
with open('filename.txt', 'r') as f:
    content = f.read()

# Read line by line
with open('filename.txt', 'r') as f:
    for line in f:
        print(line.strip())
```

The `with` statement ensures the file is properly closed after reading."""

    elif "write" in last_message and "file" in last_message:
        return """To write to a file in Python:

```python
# Write (overwrite)
with open('filename.txt', 'w') as f:
    f.write('Hello, World!')

# Append to file
with open('filename.txt', 'a') as f:
    f.write('New line')
```

Use 'w' mode to overwrite or 'a' mode to append."""

    elif "thank" in last_message:
        return "You're welcome! Feel free to ask if you have more questions."

    else:
        return "I'd be happy to help. Could you provide more details?"


def main():
    """Demonstrate chat completion API."""

    # =========================================================================
    # Step 1: Understanding chat completions
    # =========================================================================
    print("=" * 60)
    print("CHAT COMPLETION API")
    print("=" * 60)
    print("""
    Chat completions enable multi-turn conversations:

    Message roles:
    - system: Instructions for the assistant (usually first)
    - user: Human input
    - assistant: Model responses

    Example conversation:
    [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Python?"},
      {"role": "assistant", "content": "Python is a programming..."},
      {"role": "user", "content": "Show me an example."},
    ]

    The model sees the full conversation history and generates
    a contextually appropriate response.
    """)

    # =========================================================================
    # Step 2: Create a chat session
    # =========================================================================
    print("\nCreating chat session...")

    config = ChatCompletionConfig(
        model="meta-llama/Llama-3-8B",
        adapter="chat-assistant-v1",
        temperature=0.7,
        max_tokens=512,
    )

    system_prompt = """You are a helpful coding assistant. You provide clear,
concise explanations with code examples when appropriate."""

    session = ChatSession(config, system_prompt)

    print(f"  Model: {config.model}")
    print(f"  Adapter: {config.adapter}")
    print(f"  System prompt set: Yes")

    # =========================================================================
    # Step 3: Conduct a conversation
    # =========================================================================
    print("\n" + "=" * 60)
    print("CONVERSATION")
    print("=" * 60)

    print(f"\n[System]: {system_prompt[:60]}...")

    # Turn 1
    user_input1 = "How do I read a file in Python?"
    print(f"\n[User]: {user_input1}")

    session.add_user_message(user_input1)
    response1 = simulate_chat_response(session.messages)
    session.add_assistant_message(response1)

    print(f"\n[Assistant]: {response1}")

    # Turn 2
    user_input2 = "What about writing to a file?"
    print(f"\n[User]: {user_input2}")

    session.add_user_message(user_input2)
    response2 = simulate_chat_response(session.messages)
    session.add_assistant_message(response2)

    print(f"\n[Assistant]: {response2}")

    # Turn 3
    user_input3 = "Thanks for your help!"
    print(f"\n[User]: {user_input3}")

    session.add_user_message(user_input3)
    response3 = simulate_chat_response(session.messages)
    session.add_assistant_message(response3)

    print(f"\n[Assistant]: {response3}")

    # =========================================================================
    # Step 4: Show conversation state
    # =========================================================================
    print("\n" + "=" * 60)
    print("CONVERSATION STATE")
    print("=" * 60)

    print(f"\nTotal messages: {len(session.messages)}")
    print(f"  System: 1")
    print(f"  User: {sum(1 for m in session.messages if m.role == 'user')}")
    print(f"  Assistant: {sum(1 for m in session.messages if m.role == 'assistant')}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("BEST PRACTICES")
    print("=" * 60)
    print("""
    Tips for effective chat completions:

    1. System prompts
       - Be specific about desired behavior
       - Include constraints and guidelines
       - Provide examples for complex tasks

    2. Context management
       - Monitor token usage
       - Implement history truncation
       - Consider summarization for long chats

    3. Error handling
       - Handle rate limits gracefully
       - Implement retry logic
       - Validate message format
    """)


if __name__ == "__main__":
    main()
