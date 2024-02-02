# deployment.py
import subprocess

# Train the model
subprocess.run(['python', 'model_training.py'])

# Deploy the Streamlit app
subprocess.run(['streamlit', 'run', 'app.py'])
