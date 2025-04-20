import os
import base64
import json
from flask import Flask, request, jsonify
from google.cloud import aiplatform # Use the aiplatform library directly for ADC
from google.protobuf import struct_pb2 # Needed for System Instructions

# --- Configuration ---
# Load from environment variables set on Heroku
# These will be set in Heroku Config Vars later
GCP_PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID")
GCP_LOCATION = os.environ.get("GOOGLE_LOCATION")
MODEL_ID = "gemini-2.0-flash-001" # Or load from env var if needed

# --- Authentication ---
# Heroku will use Application Default Credentials (ADC)
# We'll set GOOGLE_APPLICATION_CREDENTIALS in Heroku Config Vars
# pointing to the key file we'll create from another env var.

# Create the credentials file path in a writable location (Heroku's /tmp)
CREDENTIALS_FILE_PATH = "/Users/jennyhuang/racs_bot/mindful-life-457009-t7-eb15a4d06fe7.json"

try:
    # Get the JSON key content from the environment variable
    credentials_json_content = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not credentials_json_content:
        print("ERROR: GOOGLE_CREDENTIALS_JSON environment variable not set.")
        # Handle error appropriately in a real app (e.g., raise exception or exit)
    else:
        # Write the content to the temporary file
        with open(CREDENTIALS_FILE_PATH, "w") as f:
            f.write(credentials_json_content)
        # Set the environment variable for ADC to find the file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILE_PATH
        print(f"Credentials file created at: {CREDENTIALS_FILE_PATH}")

except Exception as e:
    print(f"ERROR writing credentials file: {e}")
    # Handle error appropriately

# Initialize Vertex AI Client (AFTER setting GOOGLE_APPLICATION_CREDENTIALS)
try:
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    # This uses ADC behind the scenes now
    print(f"Vertex AI Client Initialized for project '{GCP_PROJECT_ID}' in location '{GCP_LOCATION}'")
except Exception as e:
    print(f"ERROR initializing Vertex AI Client: {e}")
    # Handle error appropriately

# --- Flask App ---
app = Flask(__name__)

# --- System Instruction / Prompt Definition ---
# (Keep your system instruction definition)
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

system_instruction_struct = struct_pb2.Struct()
system_instruction_struct['text'] = SI_TEXT1
system_instruction_protobuf = [system_instruction_struct]


# --- Chatbot Logic ---
def generate_response(history):
    """
    Calls the Gemini model with the provided conversation history.

    Args:
        history: A list of content objects representing the conversation.

    Returns:
        The generated text response from the model, or None if an error occurs.
    """
    if not GCP_PROJECT_ID or not GCP_LOCATION:
       print("ERROR: GCP Project ID or Location not configured.")
       return "Error: Server configuration issue."

    try:
        # Load the model
        model = aiplatform.gapic.ModelServiceClient().get_model(name=f"projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/publishers/google/models/{MODEL_ID}")
        # Note: The original code used genai.Client which might map to a different underlying API call.
        # This uses the aiplatform library directly which is standard for Vertex on GCP.
        # If issues arise, you might need to adjust how the client/model is loaded or use
        # from google.cloud.aiplatform.preview.language_models import TextGenerationModel directly.

        # Simplified: Use GenerativeModel from aiplatform
        from vertexai.generative_models import GenerativeModel, Part, Tool, GenerationConfig, SafetySetting, HarmCategory

        model = GenerativeModel(
            MODEL_ID,
            system_instruction=SI_TEXT1 # Pass system instruction here
            )


        # Define generation config (adjust as needed)
        generation_config = GenerationConfig(
            temperature=1.0,
            top_p=0.95,
            max_output_tokens=8192,
            # response_modalities=["TEXT"], # Not a direct param for GenerativeModel GenerationConfig
        )

        # Define safety settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        }

        # Define tools (if needed, e.g., Google Search)
        # from vertexai.generative_models import FunctionDeclaration, Tool
        # search_tool = Tool(function_declarations=[...]) # Define search function if required
        # tools = [search_tool]
        tools = None # Keep it simple for now, add tools back if needed

        # Call the model
        response = model.generate_content(
            contents=history, # Pass the full history
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            stream=False # Get the full response at once for API
        )

        print(f"Raw API Response: {response}") # Log the raw response for debugging

        # Extract text - check the structure of 'response' carefully
        # The structure might be response.text or response.candidates[0].content.parts[0].text
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
             return response.candidates[0].content.parts[0].text
        elif hasattr(response, 'text'):
             return response.text
        else:
            print("Warning: Could not extract text from response. Structure might have changed.")
            print(f"Response structure: {dir(response)}")
            if response.candidates:
                print(f"Candidate structure: {dir(response.candidates[0])}")
                print(f"Candidate content: {response.candidates[0].content}")
            return "Error: Could not process model response."


    except Exception as e:
        print(f"Error generating response: {e}")
        # Log the full error traceback if possible in production logging
        import traceback
        traceback.print_exc()
        return "Error: Could not connect to the AI model."

# --- Flask Routes ---
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    API endpoint to interact with the chatbot.
    Expects JSON: { "history": [ {"role": "user/model", "parts": [{"text": "message"}] } ] }
    Returns JSON: { "response": "chatbot message" }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    history = data.get('history')

    if not history:
        return jsonify({"error": "Missing 'history' in request"}), 400

    # Basic validation of history format (can be more robust)
    if not isinstance(history, list):
         return jsonify({"error": "'history' must be a list"}), 400

    # Convert history from simple dicts to the required Part objects if needed
    # The `generate_content` expects specific types.
    # Let's assume the input format matches the required structure for now.
    # If input is simpler like [{"role": "user", "text": "hi"}], convert it here.

    # Example conversion if input is simpler:
    # from vertexai.generative_models import Part, Content
    # formatted_history = []
    # for item in history:
    #    role = item.get("role")
    #    text = item.get("text")
    #    if role and text:
    #        formatted_history.append(Content(role=role, parts=[Part.from_text(text)]))
    # history = formatted_history # Use the formatted list

    response_text = generate_response(history)

    return jsonify({"response": response_text})

@app.route('/')
def index():
    # Simple index route to check if the app is running
    return "Gemini Chatbot backend is running!"

# --- Run the App ---
if __name__ == '__main__':
    # Use the PORT environment variable provided by Heroku
    port = int(os.environ.get('PORT', 8080))
    # Run on 0.0.0.0 to be accessible externally (required by Heroku)
    app.run(host='0.0.0.0', port=port, debug=False) # Turn Debug off for production