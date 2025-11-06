import os
import requests
from flask import Flask, jsonify, render_template, request
from datetime import datetime, timedelta
import joblib
import numpy as np
import webbrowser
import threading
from werkzeug.utils import secure_filename
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# GitHub API base URL and configuration
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = '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'  # Replace with your GitHub token
HEADERS = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'User-Agent': 'Flask-GitHub-App'
}

# Load personality prediction model
PERSONALITY_MODEL_PATH = "./models/personality_model_1"
personality_tokenizer = AutoTokenizer.from_pretrained(PERSONALITY_MODEL_PATH)
personality_model = AutoModelForSequenceClassification.from_pretrained(PERSONALITY_MODEL_PATH)

# Load GitHub popularity prediction model
github_popularity_model = joblib.load("github_popularity_model.pkl")  # Replace with your model path

# Configure upload folder
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_personality(text):
    """Predict personality traits from text using the pre-trained model."""
    tokens = personality_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    output = personality_model(**tokens)
    logits = output.logits
    probabilities = torch.softmax(logits, dim=-1).tolist()[0]
    labels = ["Analytical", "Creative", "Detail-Oriented", "Extroverted", "Leader"]
    return [{label: round(prob, 2)} for label, prob in zip(labels, probabilities)]


import random

# Define the question bank
questions_bank = {
 "Analytical": [
        "Describe a situation where you had to solve a complex problem.",
        "How do you approach decision-making in uncertain situations?",
        "Can you give an example of a time you used data to make a decision?",
        "How do you prioritize tasks when you have multiple deadlines?",
        "Tell me about a time you analyzed data and drew conclusions.",
        "What tools do you use for data analysis?",
        "How do you approach problem-solving in a team?",
        "What is your process for handling large sets of data?",
        "Can you describe a time you identified an error in data?",
        "How do you ensure accuracy when working with complex data?"
    ],
    "Creative": [
        "Tell me about a time when you had to think outside the box.",
        "What creative methods do you use when brainstorming ideas?",
        "Can you describe a project where your creativity was key to its success?",
        "How do you approach a new, unstructured problem?",
        "Describe an innovative solution you developed for a challenge.",
        "What is your process for generating new ideas?",
        "How do you stay inspired and creative in your work?",
        "Tell me about a time when you improved a process with creativity.",
        "How do you balance creativity with practicality?",
        "Can you share an example of when you used creativity to overcome an obstacle?"
    ],
    "Detail-Oriented": [
        "Give an example of a time when your attention to detail made a difference.",
        "How do you ensure that your work is always accurate and precise?",
        "Tell me about a time when you caught a mistake others missed.",
        "How do you manage tasks that require high attention to detail?",
        "What strategies do you use to avoid making errors?",
        "How do you stay organized when handling detailed tasks?",
        "Can you describe a time when your attention to detail improved a process?",
        "How do you double-check your work to ensure it's error-free?",
        "What tools do you use to maintain accuracy in your work?",
        "Can you tell me about a time you fixed an error in a project?"
    ],
    "Extroverted": [
        "How do you manage relationships with colleagues or team members?",
        "Tell me about a time when you motivated others to perform better.",
        "How do you feel about working in teams versus working alone?",
        "What techniques do you use to communicate effectively with a team?",
        "How do you build strong working relationships with people you don’t know well?",
        "Can you describe a situation where your social skills helped resolve a conflict?",
        "What role do you usually take on in team settings?",
        "How do you handle situations when team dynamics are challenging?",
        "Tell me about a time when you helped a team member overcome a difficulty.",
        "How do you adapt your communication style to different people?"
    ],
    "Leader": [
        "Describe a time when you led a team to success.",
        "How do you handle conflict within a team?",
        "What is your approach to mentoring junior colleagues?",
        "Can you share a time when you had to make a difficult leadership decision?",
        "How do you motivate your team to achieve its goals?",
        "Tell me about a time when you had to delegate tasks effectively.",
        "How do you ensure your team stays productive and focused?",
        "What is your leadership style?",
        "How do you approach giving constructive feedback to team members?",
        "Tell me about a time when you led by example."
    ]
}


def generate_question_sets(personality_scores):
    random.shuffle(personality_scores)
    sorted_traits = sorted(personality_scores, key=lambda x: list(x.values())[0], reverse=True)
    sorted_traits = [(trait, prob) for trait_dict in sorted_traits for trait, prob in trait_dict.items()]

    assigned_questions = set()
    question_sets = [[] for _ in range(3)]

    for trait, _ in sorted_traits:
        random.shuffle(questions_bank[trait])

        for i in range(len(question_sets)):
            if len(question_sets[i]) >= 10:
                continue
            for question in questions_bank[trait]:
                if question not in assigned_questions and len(question_sets[i]) < 10:
                    question_sets[i].append(question)
                    assigned_questions.add(question)

    return question_sets


@app.route('/')
def index():
    return render_template('index1.html')


@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['resume']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            resume_text = ''.join([page.extract_text() for page in pdf_reader.pages])

        personality_traits = predict_personality(resume_text)
        questions = generate_question_sets(personality_traits)

        return render_template(
            'personality_result.html',
            resume_text=resume_text,
            personality_traits=personality_traits,
            questions=questions
        )
    except Exception as e:
        return jsonify({'error': f'Error processing the file: {str(e)}'}), 500


# API route to get GitHub profile and project data, and predict popularity
@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    username = request.form['username']  # Get the username from the form input
    
    try:
        # Fetch user profile data from GitHub
        user_url = f'{GITHUB_API_URL}/users/{username}'
        user_response = requests.get(user_url, headers=HEADERS)

        if user_response.status_code == 200:
            user_data = user_response.json()

            # Fetch repositories of the user
            repos_url = f'{GITHUB_API_URL}/users/{username}/repos'
            repos_response = requests.get(repos_url, headers=HEADERS)

            if repos_response.status_code == 200:
                repos = repos_response.json()
                completed_projects = []

                # Collect relevant data for prediction
                one_year_ago = datetime.now() - timedelta(days=365)
                since_date = one_year_ago.isoformat()

                total_stars = 0
                num_repos = 0

                for repo in repos:
                    num_repos += 1
                    total_stars += repo['stargazers_count']

                    # Fetch the number of commits in the past year
                    commits_url = f'{GITHUB_API_URL}/repos/{username}/{repo["name"]}/commits'
                    commits_params = {'since': since_date}
                    commits_response = requests.get(commits_url, headers=HEADERS, params=commits_params)

                    num_commits = len(commits_response.json()) if commits_response.status_code == 200 else 0

                    # Add repository details to the completed_projects list
                    completed_projects.append({
                        'name': repo['name'],
                        'description': repo['description'] or 'No description available',
                        'stars': repo['stargazers_count'],
                        'num_commits_last_year': num_commits,
                        'url': repo['html_url']
                    })

                # Use the machine learning model to predict popularity based on followers, repos, and stars
                input_data = np.array([[user_data['followers'], num_repos, total_stars]])
                predicted_popularity = github_popularity_model.predict(input_data)

                # Prepare the response data to be displayed on the webpage
                result = {
                    'name': user_data['name'],
                    'bio': user_data['bio'],
                    'followers': user_data['followers'],
                    'location': user_data['location'],
                    'company': user_data.get('company', 'N/A'),
                    'completed_projects': completed_projects,  # List of project info
                    'predicted_popularity': predicted_popularity[0]  # The predicted popularity score
                }

                return render_template('result.html', result=result)  # Render the results in an HTML template
            else:
                return jsonify({'error': 'Could not fetch repositories for this user'}), repos_response.status_code
        else:
            return jsonify({'error': 'User not found'}), user_response.status_code

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})

# Auto open browser on startup
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    app.run(debug=True)




    