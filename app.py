import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load trained RandomForest Model
with open('randomforest_model.pkl', 'rb') as f:
    model = pickle.pickle.load(f) if hasattr(pickle, 'pickle') else pickle.load(f)

# HTML Template with Embedded Modern Styling, Interactive Effects, and Animations
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Prediction</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --card-bg: rgba(255, 255, 255, 0.95);
            --text-main: #0f172a;
            --text-muted: #64748b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 800px;
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: fadeIn 0.8s ease-out;
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            color: var(--text-main);
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }

        .header p {
            color: var(--text-muted);
            margin-top: 0.5rem;
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-main);
        }

        .input-group input, .input-group select {
            padding: 0.75rem 1rem;
            border: 1.5px solid #e2e8f0;
            border-radius: 10px;
            font-size: 0.95rem;
            transition: all 0.2s ease;
            outline: none;
            background: #f8fafc;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--primary);
            background: #ffffff;
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.1);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 1rem;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .btn-submit:hover {
            background: var(--primary-hover);
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.4);
        }

        /* Click Ripple Effect */
        .ripple {
            position: absolute;
            background: rgba(255, 255, 255, 0.5);
            border-radius: 50%;
            transform: scale(0);
            animation: ripple-animation 0.6s linear;
            pointer-events: none;
        }

        @keyframes ripple-animation {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }

        /* Result Display Card */
        .result-card {
            margin-top: 2rem;
            padding: 1.25rem;
            border-radius: 12px;
            text-align: center;
            font-weight: 600;
            font-size: 1.1rem;
            animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .result-pass {
            background: #dcfce7;
            color: #15803d;
            border: 1px solid #86efac;
        }

        .result-fail {
            background: #fee2e2;
            color: #b91c1c;
            border: 1px solid #fca5a5;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes popIn {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Student Performance Predictor</h1>
        <p>Enter academic details to predict student outcome</p>
    </div>

    <form method="POST" action="/predict" class="form-grid" id="predictForm">
        <div class="input-group">
            <label>Age</label>
            <input type="number" name="Age" min="15" max="100" required placeholder="e.g. 20">
        </div>

        <div class="input-group">
            <label>Gender</label>
            <select name="Gender" required>
                <option value="0">Male</option>
                <option value="1">Female</option>
            </select>
        </div>

        <div class="input-group">
            <label>Department (Numeric Code)</label>
            <input type="number" name="Department" required placeholder="e.g. 1">
        </div>

        <div class="input-group">
            <label>Study Hours / Day</label>
            <input type="number" step="0.1" name="Study_Hours_Per_Day" required placeholder="e.g. 4.5">
        </div>

        <div class="input-group">
            <label>Attendance (%)</label>
            <input type="number" step="0.1" name="Attendance_Percentage" min="0" max="100" required placeholder="e.g. 85.0">
        </div>

        <div class="input-group">
            <label>Assignments Completed</label>
            <input type="number" name="Assignments_Completed" min="0" required placeholder="e.g. 10">
        </div>

        <div class="input-group">
            <label>Midterm Score</label>
            <input type="number" step="0.1" name="Midterm_Score" min="0" max="100" required placeholder="e.g. 78.5">
        </div>

        <div class="input-group">
            <label>Final Score</label>
            <input type="number" step="0.1" name="Final_Score" min="0" max="100" required placeholder="e.g. 82.0">
        </div>

        <button type="submit" class="btn-submit" id="submitBtn">
            <span id="btnText">Run Prediction</span>
        </button>
    </form>

    {% if prediction %}
        <div class="result-card {% if prediction == 'Pass' %}result-pass{% else %}result-fail{% endif %}">
            Prediction Result: {{ prediction }}
        </div>
    {% endif %}
</div>

<script>
    // Button Click Effect (Ripple & Loading State)
    const button = document.getElementById('submitBtn');
    
    button.addEventListener('click', function (e) {
        // Create Ripple Effect
        const circle = document.createElement('span');
        const diameter = Math.max(button.clientWidth, button.clientHeight);
        const radius = diameter / 2;

        circle.style.width = circle.style.height = `${diameter}px`;
        circle.style.left = `${e.clientX - button.getBoundingClientRect().left - radius}px`;
        circle.style.top = `${e.clientY - button.getBoundingClientRect().top - radius}px`;
        circle.classList.add('ripple');

        const ripple = button.getElementsByClassName('ripple')[0];
        if (ripple) ripple.remove();

        button.appendChild(circle);

        // Form Submit Loading Feedback
        if(document.getElementById('predictForm').checkValidity()) {
            document.getElementById('btnText').innerText = 'Processing...';
        }
    });
</script>

</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_LAYOUT)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features in the expected categorical and numerical order
        features = [
            float(request.form['Age']),
            float(request.form['Gender']),
            float(request.form['Department']),
            float(request.form['Study_Hours_Per_Day']),
            float(request.form['Attendance_Percentage']),
            float(request.form['Assignments_Completed']),
            float(request.form['Midterm_Score']),
            float(request.form['Final_Score'])
        ]
        
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        
        return render_template_string(HTML_LAYOUT, prediction=prediction[0])
    except Exception as e:
        return render_template_string(HTML_LAYOUT, prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
