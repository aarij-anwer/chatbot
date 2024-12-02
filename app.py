from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Read the NVIDIA API key from the environment
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY environment variable not set!")

# Initialize the NVIDIA model
model = ChatNVIDIA(model="meta/llama3-70b-instruct", api_key=NVIDIA_API_KEY)

# Store conversation history (in-memory, for now)
conversations = {}

@app.route('/chat', methods=['POST'])
def chat():
    user_id = request.json.get('user_id')  # Unique identifier for the user
    user_message = request.json.get('message')
    language = request.json.get('language', 'English')  # Default to English

    if not user_message or not user_id:
        return jsonify({"error": "Both 'user_id' and 'message' are required"}), 400

    # Initialize conversation history if user is new
    if user_id not in conversations:
        # Add system message with the language context
        conversations[user_id] = [
            SystemMessage(content=f"You are a helpful assistant. Answer all questions to the best of your ability in {language}.")
        ]

    try:
        # Add user's message to history
        conversations[user_id].append(HumanMessage(content=user_message))

        # Generate response
        response = model.invoke(conversations[user_id])

        # Add AI's response to history
        conversations[user_id].append(AIMessage(content=response.content))

        return jsonify({"response": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
