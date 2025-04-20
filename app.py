import os
import base64
import json
import traceback # For detailed error logging

# Flask imports for web server and templating
from flask import Flask, request, jsonify, render_template

# Google Cloud / Vertex AI imports
# Use vertexai library for Gemini models
from vertexai.generative_models import GenerativeModel, Part, Content, GenerationConfig, SafetySetting, HarmCategory
import google.cloud.aiplatform as aiplatform # For initialization
import google.oauth2.service_account # <-- Import for explicit credential loading

# --- Configuration (Loaded from Environment Variables) ---
GCP_PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID")
GCP_LOCATION = os.environ.get("GOOGLE_LOCATION")
MODEL_ID = "gemini-2.0-flash-001"

# --- Authentication & Vertex AI Initialization Function ---
# NOTE: This function is called later during app startup

def setup_credentials_and_vertexai():
    """
    Sets up credentials from env var and initializes Vertex AI
    explicitly using direct JSON parsing, avoiding file operations.
    Returns the credentials object on success, None on failure.
    """
    print("Attempting to set up credentials and initialize Vertex AI...") # Changed log message slightly
    credentials_json_content = os.environ.get("GOOGLE_CREDENTIALS_JSON")

    # --- DEBUGGING START ---
    if not credentials_json_content:
        print("ERROR: GOOGLE_CREDENTIALS_JSON environment variable is not set or empty.")
        return None
    else:
        print(f"DEBUG: GOOGLE_CREDENTIALS_JSON (first 20 chars): {credentials_json_content[:20]}...")
        print(f"DEBUG: GOOGLE_CREDENTIALS_JSON (last 20 chars): ...{credentials_json_content[-20:]}")
    # --- DEBUGGING END ---

    try:
        # Parse the credentials JSON directly from string
        service_account_info = json.loads(credentials_json_content)
        credentials = google.oauth2.service_account.Credentials.from_service_account_info(
            service_account_info
        )
        print("DEBUG: Successfully loaded credentials object from JSON string.")
    except json.JSONDecodeError as je:
        print(f"ERROR: Invalid JSON in GOOGLE_CREDENTIALS_JSON: {je}")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"ERROR loading credentials object from service_account_info: {e}") # More specific error location
        traceback.print_exc()
        return None

    if not GCP_PROJECT_ID or not GCP_LOCATION:
        print("ERROR: GOOGLE_PROJECT_ID or GOOGLE_LOCATION environment variables not set.")
        return None

    try:
        # Initialize the Vertex AI client library, passing credentials explicitly
        print(f"Attempting to initialize Vertex AI SDK for project {GCP_PROJECT_ID}...") # Log before init
        aiplatform.init(
            project=GCP_PROJECT_ID,
            location=GCP_LOCATION,
            credentials=credentials
        )
        print(f"✅ Vertex AI SDK Initialized for project '{aiplatform.initializer.global_config.project}' in location '{aiplatform.initializer.global_config.location}' using provided credentials.")
        return credentials
    except Exception as e:
        print(f"❌ ERROR initializing Vertex AI SDK with explicit credentials: {e}") # Clearer error message
        traceback.print_exc()
        return None

# --- System Instruction / Prompt Definition ---
SI_TEXT1 = """... (Your system instruction remains here) ..."""

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Initialize Vertex AI on App Startup ---
print("Starting server and attempting to initialize Vertex AI...")
LOADED_CREDENTIALS = setup_credentials_and_vertexai()
INITIALIZATION_SUCCESSFUL = LOADED_CREDENTIALS is not None
print(f"INITIALIZATION_SUCCESSFUL flag set to: {INITIALIZATION_SUCCESSFUL}") # Log final flag status

# --- Gemini Model Interaction Logic ---
def generate_response(history_content):
    # ... (generate_response function remains the same) ...
    # Use the global INITIALIZATION_SUCCESSFUL flag (set after setup runs)
    if not INITIALIZATION_SUCCESSFUL:
        print("ERROR: Vertex AI not initialized successfully. Cannot generate response.")
        return "Error: Backend server configuration issue. Please check logs."

    try:
        # Initialize the GenerativeModel class. It should use the credentials
        # set globally via aiplatform.init() when we passed them explicitly.
        model = GenerativeModel(
            MODEL_ID,
            system_instruction=SI_TEXT1
        )
        # ... (rest of try block) ...
    except Exception as e:
        print(f"ERROR during Gemini API call: {e}")
        traceback.print_exc()
        return "Error: An exception occurred while communicating with the AI model."


# --- Flask Routes ---
@app.route('/health')
def health():
    # ... (health endpoint remains the same) ...
    status = {
        "status": "online",
        "google_creds_present": bool(os.environ.get("GOOGLE_CREDENTIALS_JSON")),
        "project_id_present": bool(GCP_PROJECT_ID),
        "location_present": bool(GCP_LOCATION),
        "initialization_successful": INITIALIZATION_SUCCESSFUL,
        "model_id": MODEL_ID
    }
    return jsonify(status)


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    # ... (chat endpoint remains the same) ...
    if not INITIALIZATION_SUCCESSFUL: # Add extra check here for safety
         return jsonify({"error": "Server not initialized correctly"}), 503

    # ... (rest of chat endpoint) ...
    response_text = generate_response(history_content)
    # ... (return logic) ...


@app.route('/')
def index():
    # ... (index route remains the same) ...
    if not INITIALIZATION_SUCCESSFUL:
         return "Error: Backend server failed to initialize. Please check logs and visit /health for diagnostic information.", 503
    # ... (rest of index route) ...


@app.route('/debug')
def debug():
    # ... (debug route remains the same) ...
    return """..."""


# --- Run the Flask App ---
if __name__ == '__main__':
    # ... (run block remains the same) ...
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)