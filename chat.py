"""
Text chat controller for the RAG Gmail Assistant.
"""

from rag import ask_question


def start_chat() -> None:
    """Interactive text-based chat loop."""
    print('(System): Starting a new conversation..')
    print('(System): type "exit" to quit the chat.')
    print('(System): type "new conversation" to start over')

    new_conversation = True
    messages = None

    while True:
        query = input("## You: ")

        if query.lower() == "new conversation":
            print("(System): Starting a new conversation..")
            new_conversation = True
            continue
        elif query.lower() == "exit":
            print("(System): Exiting... Thank you!")
            break
        else:
            if new_conversation:
                messages, assistant_reply = ask_question(query)
            else:
                messages, assistant_reply = ask_question(query, messages=messages)
            print("## Model:", assistant_reply)

        new_conversation = False
