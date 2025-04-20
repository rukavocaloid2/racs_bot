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

# --- Configuration (Loaded from Environment Variables) ---
GCP_PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID")
GCP_LOCATION = os.environ.get("GOOGLE_LOCATION")
MODEL_ID = "gemini-2.0-flash-001" # Or consider making this an env var too

# --- Authentication & Vertex AI Initialization ---
SERVICE_ACCOUNT_KEY_PATH = "/tmp/mindful-life-457009-t7-eb15a4d06fe7.json" # Writable path on Heroku

def setup_credentials_and_vertexai():
    """Sets up credentials from env var and initializes Vertex AI."""
    print("Setting up credentials and initializing Vertex AI...")
    credentials_json_content = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not credentials_json_content:
        print("ERROR: GOOGLE_CREDENTIALS_JSON environment variable not set.")
        # In a real app, you might raise an exception or exit
        return False # Indicate failure

    try:
        # Write the credentials JSON content to the temp file
        with open(SERVICE_ACCOUNT_KEY_PATH, "w") as f:
            f.write(credentials_json_content)
        # Set the environment variable for Application Default Credentials (ADC)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_PATH
        print(f"Credentials file created at: {SERVICE_ACCOUNT_KEY_PATH}")
    except Exception as e:
        print(f"ERROR writing credentials file: {e}")
        traceback.print_exc()
        return False # Indicate failure

    if not GCP_PROJECT_ID or not GCP_LOCATION:
        print("ERROR: GOOGLE_PROJECT_ID or GOOGLE_LOCATION environment variables not set.")
        return False # Indicate failure

    try:
        # Initialize the Vertex AI client library
        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        print(f"Vertex AI SDK Initialized for project '{GCP_PROJECT_ID}' in location '{GCP_LOCATION}'")
        return True # Indicate success
    except Exception as e:
        print(f"ERROR initializing Vertex AI SDK: {e}")
        traceback.print_exc()
        return False # Indicate failure

# Call the setup function when the script loads
INITIALIZATION_SUCCESSFUL = setup_credentials_and_vertexai()

# --- Flask App Initialization ---
# Flask automatically looks for 'templates' and 'static' folders
app = Flask(__name__)

# --- System Instruction / Prompt Definition ---
# Using the text provided previously
SI_TEXT1 = """Initial Greeting: Begin the conversation with: \"Hello, how are you! Welcome to the RACS interview, what's the specialty you are applying for?\" After the user responds with their specialty, confirm their application country (New Zealand or Australia) before proceeding with the first scenario.
Role: You are an experienced interviewer for the Royal Australasian College of Surgeons (RACS) specialty interview. Your goal is to help candidates practice and receive constructive feedback.
Persona: Maintain a professional yet approachable and empathetic demeanor, similar to a real RACS interviewer.
Task: Conduct a simulated RACS interview by presenting a series of scenario-based questions relevant to the candidate's chosen surgical specialty and their application country (New Zealand or Australia).
Interview Format:
Scenario Presentation: Present a realistic and emotionally engaging clinical scenario related to the candidate's chosen specialty.
Targeted Questions (3 per scenario, asked separately): Ask three distinct and focused follow-up questions, one at a time, directly related to the scenario. These questions should probe:
Clinical reasoning and decision-making.
Ethical considerations.
Teamwork and collaboration.
Cultural safety aspects specific to the candidate's application country (Māori perspectives for NZ, Aboriginal and Torres Strait Islander perspectives for Australia). At least one question should heavily involve Māori cultural aspects when the candidate is from NZ.
Turn-Taking: Wait for the candidate to answer each question before revealing the next one.
Feedback and Exemplary Answer: After the candidate has answered all three questions for a scenario:
Provide concise feedback (maximum 150 words) evaluating their performance against RACS competencies: Communication, Collaboration, Cultural Safety, Health Advocacy, Judgment, Professionalism, and Scholarship.
Highlight specific strengths and areas for improvement based on their responses.
Offer an overall indicative score (e.g., out of 100) and a general percentile ranking (e.g., \"in the top quartile\").
Crucially, always provide an exemplary answer that reflects RACS standards, region-specific practices (including healthcare systems, referral pathways, and cultural safety), and demonstrates a strong understanding of the scenario's nuances. For NZ candidates, exemplary answers should include local data, such as comparisons between rural and metro hospitals and relevant Māori cultural practices.
Continue or Conclude: After providing feedback and the exemplary answer, ask: \"Would you like to proceed to another scenario?\" If the candidate indicates they are finished, provide a comprehensive final evaluation summarizing their overall performance across all scenarios.
Country-Specific Considerations:
New Zealand Candidates:Integrate relevant Māori terms and tales where appropriate in scenarios, questions, and exemplary answers.
Focus scenarios and questions on the New Zealand healthcare system, cultural contexts (including Te Tiriti o Waitangi principles), and prevalent health issues.
Exemplary answers should demonstrate cultural safety, empathy, and a strong understanding of Māori health frameworks, including comparisons of healthcare delivery in rural vs. metropolitan settings.
Australian Candidates:Focus scenarios and questions on the Australian healthcare system and prevalent local conditions.
Exemplary answers should consider Aboriginal and Torres Strait Islander health perspectives and demonstrate cultural safety within the Australian context.
Specialty Focus: Adapt scenarios and questions to the candidate's declared surgical specialty (General Surgery, Cardiothoracic Surgery, Neurosurgery, Orthopedic Surgery, Otolaryngology (ENT), Pediatric Surgery, Plastic and Reconstructive Surgery, Urology, Vascular Surgery).
Key Instructions for Gemini Flash 2.0:
Generate unique and varied scenarios throughout the interaction. Do not repeat scenarios.
Maintain a natural and conversational flow while adhering to the structured format.
Remember the conversation history to avoid repetition and tailor future scenarios appropriately.
Always offer the option to view an exemplary answer after your feedback on each scenario.
Provide a comprehensive final evaluation when the candidate indicates they are finished."""

# --- Gemini Model Interaction Logic ---
def generate_response(history_content):
    """
    Calls the Gemini model with the provided conversation history.

    Args:
        history_content: A list of vertexai.generative_models.Content objects.

    Returns:
        The generated text response from the model, or an error string.
    """
    if not INITIALIZATION_SUCCESSFUL:
        print("ERROR: Vertex AI not initialized successfully. Cannot generate response.")
        return "Error: Backend server configuration issue. Please check logs."

    try:
        # Initialize the GenerativeModel class with the specific model ID
        # and pass the system instruction during initialization.
        model = GenerativeModel(
            MODEL_ID,
            system_instruction=SI_TEXT1 # Pass system instruction here
            )

        # Define generation configuration
        generation_config = GenerationConfig(
            temperature=1.0,
            top_p=0.95,
            max_output_tokens=8192,
        )

        # Define safety settings (blocking none as per original code)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        }

        # Define tools (set to None as per previous simplification, add back if needed)
        tools = None

        print(f"Sending {len(history_content)} history entries to model.") # Log history size

        # Generate content using the provided history
        response = model.generate_content(
            contents=history_content, # Pass the Content objects list
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            stream=False # Get the full response at once for the API
        )

        print("Raw API Response received.") # Log reception
        # print(f"DEBUG Response object: {response}") # Detailed debug log (optional)

        # Extract the text response safely
        # Check candidate and parts structure
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
             full_response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
             if full_response_text:
                 print("Successfully extracted text from response candidate parts.")
                 return full_response_text
             else:
                  print("Warning: Response candidate parts found but contained no text.")
                  return "Error: Model returned empty parts."
        # Fallback check for simple text attribute (less common for generative models)
        elif hasattr(response, 'text') and response.text:
            print("Successfully extracted text from response.text attribute.")
            return response.text
        # Handle cases where the response might be blocked or empty
        elif response.candidates and response.candidates[0].finish_reason:
             reason = response.candidates[0].finish_reason.name
             print(f"Warning: Model response finished with reason: {reason}")
             # You might want specific messages for SAFETY, RECITATION, etc.
             if reason == "SAFETY":
                 return "Error: The response was blocked due to safety settings."
             else:
                return f"Error: Model generation stopped unexpectedly (Reason: {reason})."
        else:
            print("Warning: Could not extract text from response. Unexpected structure.")
            print(f"DEBUG Response structure: {response}") # Log structure for debugging
            return "Error: Could not process the model's response format."


    except Exception as e:
        print(f"ERROR during Gemini API call: {e}")
        traceback.print_exc() # Log the full traceback for debugging
        return "Error: An exception occurred while communicating with the AI model."

# --- Flask Routes ---

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    API endpoint to interact with the chatbot via JSON.
    Expects JSON: { "history": [ {"role": "user/model", "parts": [{"text": "message"}] } ] }
    Returns JSON: { "response": "chatbot message" } or { "error": "..." }
    """
    if not request.is_json:
        return jsonify({"error": "Request content type must be application/json"}), 415 # Unsupported Media Type

    data = request.get_json()
    history_raw = data.get('history')

    if not history_raw:
        return jsonify({"error": "Missing 'history' field in JSON request body"}), 400 # Bad Request
    if not isinstance(history_raw, list):
         return jsonify({"error": "'history' must be a list of message objects"}), 400 # Bad Request

    # --- Convert raw history from JSON to Vertex AI Content objects ---
    history_content = []
    try:
        for item in history_raw:
            role = item.get("role")
            parts_raw = item.get("parts")

            # Basic validation of each history item
            if not role or role.lower() not in ["user", "model"] or \
               not parts_raw or not isinstance(parts_raw, list):
                print(f"Skipping invalid history item format: {item}")
                continue # Skip malformed items silently or return error

            # Extract text from parts (assuming format [{"text": "..."}, ...])
            text_parts = [Part.from_text(p.get("text","")) for p in parts_raw if isinstance(p, dict) and "text" in p]

            if not text_parts:
                 print(f"Skipping history item with no valid text parts: {item}")
                 continue # Skip items where no text could be extracted

            # Create Content object with validated role and parts
            history_content.append(Content(role=role.lower(), parts=text_parts))

        if not history_content and history_raw:
             # If input had items but none were valid
             return jsonify({"error": "No valid message content found in 'history'"}), 400

    except Exception as e:
        print(f"Error processing history JSON: {e}")
        traceback.print_exc()
        return jsonify({"error": "Server error processing history format"}), 500 # Internal Server Error
    # --------------------------------------------------------------------

    # Call the backend function to get the response from Gemini
    response_text = generate_response(history_content)

    # Check if the response indicates an internal error from generate_response
    if response_text and response_text.startswith("Error:"):
         # Return a server error status code if the backend had issues
         print(f"Returning error to client: {response_text}")
         return jsonify({"error": response_text}), 500 # Internal Server Error
    else:
        # Return the successful response from the bot
        return jsonify({"response": response_text})


# Serve the HTML web interface on the root URL
@app.route('/')
def index():
    """Renders the main chat interface HTML page."""
    if not INITIALIZATION_SUCCESSFUL:
         # Optionally show an error page if initialization failed
         return "Error: Backend server failed to initialize. Please check logs.", 503 # Service Unavailable
    return render_template('index.html')


# --- Run the Flask App ---
if __name__ == '__main__':
    # Get port from environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    # Run the app, listening on all interfaces (0.0.0.0) as required by Heroku
    # Set debug=False for production environments
    app.run(host='0.0.0.0', port=port, debug=False)