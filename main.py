from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
import os
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    conversation_history = data.get('history', [])
    
    client = genai.Client(
        vertexai=True,
        project="mindful-life-457009-t7",
        location="us-central1",
    )

    si_text1 = """Initial Greeting: Begin the conversation with: \"Hello, how are you! Welcome to the RACS interview, what's the specialty you are applying for?\" After the user responds with their specialty, confirm their application country (New Zealand or Australia) before proceeding with the first scenario.
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

    model = "gemini-2.0-flash-001"
    
    # Convert conversation history to the format expected by the API
    contents = []
    
    # Add conversation history if available
    if conversation_history:
        for message in conversation_history:
            role = "user" if message["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=message["content"])]
                )
            )
    
    # Add the new user message
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )
    )
    
    tools = [
        types.Tool(google_search=types.GoogleSearch()),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ],
        tools=tools,
        system_instruction=[types.Part.from_text(text=si_text1)],
    )

    # Generate response
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    
    bot_response = response.text
    
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))