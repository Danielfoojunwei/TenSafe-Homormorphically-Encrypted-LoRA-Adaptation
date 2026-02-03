"""
Chat Completion Example

Demonstrates chat-style completions with multi-turn conversations.

Requirements:
- TenSafe account and API key

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python chat_completion.py
"""

import os
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))

    tc = client.create_training_client(model_ref="meta-llama/Llama-3-8B-Chat")

    # Multi-turn conversation
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant focused on privacy."},
        {"role": "user", "content": "What is differential privacy?"},
    ]

    print("User: What is differential privacy?")
    response = tc.chat(messages=messages, max_tokens=200, temperature=0.7)
    print(f"Assistant: {response.content}\n")

    # Continue conversation
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": "How does it protect user data?"})

    print("User: How does it protect user data?")
    response = tc.chat(messages=messages, max_tokens=200)
    print(f"Assistant: {response.content}\n")

    # Add another turn
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": "Can you give a simple example?"})

    print("User: Can you give a simple example?")
    response = tc.chat(messages=messages, max_tokens=300)
    print(f"Assistant: {response.content}")


if __name__ == "__main__":
    main()
