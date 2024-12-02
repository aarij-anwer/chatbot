from flask import Flask, request, Response
from flask_cors import CORS
from dotenv import load_dotenv
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Read the NVIDIA API key
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY environment variable not set!")

# Initialize the NVIDIA model
model = ChatNVIDIA(model="meta/llama3-70b-instruct", api_key=NVIDIA_API_KEY)

# Store conversation history
conversations = {}

@app.route('/chat-stream', methods=['GET'])
def chat_stream():
    user_id = request.args.get('user_id')
    user_message = request.args.get('message')
    language = request.args.get('language', 'English')  # Default to English

    if not user_message or not user_id:
        return Response("Invalid parameters", status=400)

    # Initialize conversation history if user is new
    if user_id not in conversations:
        conversations[user_id] = [
            SystemMessage(content=f"You are a helpful assistant. Answer all questions to the best of your ability in {language}.")
        ]

    # Add user's message to conversation history
    conversations[user_id].append(HumanMessage(content=user_message))

    # Stream the response from the model
    def generate_response():
        try:
            response = model.invoke(conversations[user_id])
            for word in response.content.split():  # Stream word-by-word
                yield f"data: {word}\n\n"  # SSE format
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

    return Response(generate_response(), content_type="text/event-stream")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
