from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys
import json
import base64
import tempfile
import traceback

app = Flask(__name__)
CORS(app)

# Create a route for testing
@app.route('/test')
def test():
    return "Test route is working!"

# Add debugging information
@app.route('/debug')
def debug():
    return jsonify({
        "python_version": sys.version,
        "environment": dict(os.environ),
        "current_directory": os.getcwd(),
        "files_in_directory": os.listdir()
    })

# Set up Google Cloud credentials from environment
credentials_temp_path = None
auth_status = "Not attempted"

try:
    if 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
        print("Found credentials in environment variables")
        # Get the encoded credentials
        encoded_credentials = os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON']
        
        # Decode the credentials
        decoded_credentials = base64.b64decode(encoded_credentials).decode('utf-8')
        
        # Write the credentials to a temporary file
        fd, credentials_temp_path = tempfile.mkstemp()
        with os.fdopen(fd, 'w') as tmp:
            tmp.write(decoded_credentials)
        
        # Set the environment variable to point to the temporary file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_temp_path
        print(f"Credentials saved to {credentials_temp_path}")
        
        auth_status = "Credentials file created"
    else:
        print("No credentials found in environment variables")
        auth_status = "No credentials in environment"
except Exception as e:
    print(f"Error setting up credentials: {str(e)}")
    traceback.print_exc()
    auth_status = f"Error setting up credentials: {str(e)}"

# Try importing Google libraries separately to catch specific import errors
try:
    print("Importing google.generativeai...")
    import google.generativeai as genai
    print("Successfully imported google.generativeai")
    
    print("Importing google.generativeai.types...")
    from google.generativeai import types
    print("Successfully imported google.generativeai.types")
    
    def get_client():
        print("Initializing genai client...")
        client = genai.Client(
            vertexai=True,
            project="mindful-life-457009-t7",
            location="us-central1",
        )
        print("Successfully initialized genai client")
        return client
    
    # Test authentication but don't let it crash the app
    try:
        print("Testing authentication...")
        test_client = get_client()
        print("Authentication successful!")
        auth_status = "Successfully authenticated with Google Cloud"
    except Exception as e:
        print(f"Authentication test failed: {str(e)}")
        traceback.print_exc()
        auth_status = f"Authentication test failed: {str(e)}"
        
except ImportError as e:
    print(f"Failed to import Google libraries: {str(e)}")
    traceback.print_exc()
    auth_status = f"Import error: {str(e)}"
except Exception as e:
    print(f"Unexpected error during setup: {str(e)}")
    traceback.print_exc()
    auth_status = f"Setup error: {str(e)}"

# System instruction - only define if imports succeeded
SI_TEXT = """Initial Greeting: Begin the conversation with: \"Hello, how are you! Welcome to the RACS interview, what's the specialty you are applying for?\" After the user responds with their specialty, confirm their application country (New Zealand or Australia) before proceeding with the first scenario.
Initial Greeting: Begin the conversation with: "Hello, how are you! Welcome to the RACS interview, what's the specialty you are applying for?" After the user responds with their specialty, confirm their application country (New Zealand or Australia) before proceeding with the first scenario.
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
Offer an overall indicative score (e.g., out of 100) and a general percentile ranking (e.g., "in the top quartile").
Crucially, always provide an exemplary answer that reflects RACS standards, region-specific practices (including healthcare systems, referral pathways, and cultural safety), and demonstrates a strong understanding of the scenario's nuances. For NZ candidates, exemplary answers should include local data, such as comparisons between rural and metro hospitals and relevant Māori cultural practices.
Continue or Conclude: After providing feedback and the exemplary answer, ask: "Would you like to proceed to another scenario?" If the candidate indicates they are finished, provide a comprehensive final evaluation summarizing their overall performance across all scenarios.
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
Provide a comprehensive final evaluation when the candidate indicates they are finished.
"""

@app.route('/')
def index():
    # Simplified response just to get the app working
    try:
        return render_template('index.html', auth_status=auth_status)
    except Exception as e:
        return f"""
        <html>
            <head><title>RACS Bot Status</title></head>
            <body>
                <h1>RACS Interview Bot</h1>
                <p>Auth Status: {auth_status}</p>
                <p>Error: {str(e)}</p>
                <p>Try the <a href="/test">test route</a> or <a href="/debug">debug information</a>.</p>
            </body>
        </html>
        """

# Simplified chat endpoint for testing
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        # For testing, just echo back the message if there are import issues
        if 'genai' not in sys.modules:
            return jsonify({
                "response": f"Echo: {user_message} (Note: Gemini API not available - auth status: {auth_status})"
            })
            
        # Initialize client
        client = get_client()
        
        # Basic response for testing
        try:
            model = "gemini-2.0-flash-001"
            
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_message)]
                )
            ]
            
            generate_content_config = types.GenerateContentConfig(
                temperature=1,
                top_p=0.95,
                max_output_tokens=100,  # Reduced for testing
                response_modalities=["TEXT"],
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
                ],
                system_instruction=[types.Part.from_text(text=SI_TEXT)],
            )

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            
            bot_response = response.text
            return jsonify({"response": bot_response})
            
        except Exception as e:
            print(f"Error in generate_content: {str(e)}")
            traceback.print_exc()
            return jsonify({
                "response": f"Sorry, there was an error generating a response: {str(e)}"
            })
            
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in chat endpoint: {error_details}")
        return jsonify({
            "response": f"Sorry, there was an error processing your request: {str(e)}",
            "error": str(e),
            "stack_trace": error_details
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)