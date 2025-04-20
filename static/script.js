// Get references to the HTML elements
const chatbox = document.getElementById('chatbox');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');

// Store the conversation history (as expected by the backend)
let conversationHistory = [];

// --- Helper Function to Add Messages to the Chatbox ---
function addMessageToChatbox(sender, text) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');

    // Add specific class for styling and potentially identifying sender
    if (sender === 'user') {
        messageDiv.classList.add('user-message');
    } else if (sender === 'bot') {
        messageDiv.classList.add('bot-message');
    } else if (sender === 'error') {
        messageDiv.classList.add('error-message'); // Add error class
        text = `Error: ${text}`; // Prepend "Error:"
    }

    // Use <p> tag inside the div for better structure and styling
    const messageParagraph = document.createElement('p');
    messageParagraph.textContent = text; // Use textContent for security
    messageDiv.appendChild(messageParagraph);

    chatbox.appendChild(messageDiv);

    // Auto-scroll to the bottom of the chatbox
    chatbox.scrollTop = chatbox.scrollHeight;
}


// --- Helper Function to Show Loading Indicator ---
let loadingIndicator = null;
function showLoadingIndicator() {
    if (!loadingIndicator) {
        loadingIndicator = document.createElement('div');
        loadingIndicator.classList.add('message', 'bot-message', 'loading'); // Style like a bot message
        loadingIndicator.innerHTML = '<p><i>Typing...</i></p>';
        chatbox.appendChild(loadingIndicator);
        chatbox.scrollTop = chatbox.scrollHeight;
    }
}

// --- Helper Function to Remove Loading Indicator ---
function removeLoadingIndicator() {
    if (loadingIndicator) {
        chatbox.removeChild(loadingIndicator);
        loadingIndicator = null;
    }
}


// --- Function to Send Message to Backend ---
async function sendMessage() {
    const userText = userInput.value.trim();
    if (!userText) return; // Don't send empty messages

    // Clear the input field
    userInput.value = '';

    // 1. Add user message to chatbox and history
    addMessageToChatbox('user', userText);
    conversationHistory.push({ role: "user", parts: [{ text: userText }] });

    // 2. Show loading indicator
    showLoadingIndicator();

    try {
        // 3. Send history to the backend '/chat' endpoint
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ history: conversationHistory }) // Send the whole history
        });

        // 4. Remove loading indicator
        removeLoadingIndicator();

        if (!response.ok) {
            // Handle HTTP errors (like 4xx, 5xx)
            const errorData = await response.json();
            const errorMessage = errorData.error || `Server error: ${response.status}`;
             addMessageToChatbox('error', `Failed to get response. ${errorMessage}`);
            // Remove the last user message from history if the API call failed badly
            conversationHistory.pop();
            return;
        }

        // 5. Get response from backend
        const data = await response.json();
        const botText = data.response;

        if (botText) {
             // 6. Add bot message to chatbox and history
            addMessageToChatbox('bot', botText);
            conversationHistory.push({ role: "model", parts: [{ text: botText }] });
        } else {
             addMessageToChatbox('error', 'Received an empty response from the bot.');
        }

    } catch (error) {
        // Handle network errors or other issues during fetch
        removeLoadingIndicator();
        console.error("Error sending message:", error);
        addMessageToChatbox('error', 'Could not connect to the server or an error occurred.');
        // Remove the last user message from history if the API call failed badly
        conversationHistory.pop();
    }
}

// --- Event Listeners ---
// Send message when button is clicked
sendButton.addEventListener('click', sendMessage);

// Send message when Enter key is pressed in the input field
userInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevent default form submission (if it was in a form)
        sendMessage();
    }
});

// --- Initial Setup ---
// (Optional) You could potentially send an initial empty message here
// to get the bot's greeting, but the current setup expects the user
// to type "hi" or similar first, as indicated by the initial message
// placed directly in the HTML.