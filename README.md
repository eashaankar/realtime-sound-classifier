# Real time sound classifier

A full-stack environmental sound classification system that listens through the browser, records audio snippets, and uses a machine learning model to identify sounds such as **door knocks**, **water boiling**, **cat meows**, and **dog barks**.

This repository includes:

- **Frontend:** HTML/JS/CSS web interface + live microphone level meter  
- **Backend:** FastAPI server that performs ML inference  
- **Model Training Workflow:** Google Colab notebook for training and exporting your own sound classifier  


## Features

- Live microphone listening inside the browser  
- Real-time RMS level meter animation  
- Automatic recording of short audio chunks (1–5 seconds)  
- Uploads audio to backend for classification  
- Shows predicted label + confidence  
- Optional “ding” alert for detected events  
- Works with your own custom dataset

## How to Run the Project

Below are the exact instructions to run both backend and frontend on your machine.


### Step 1 — Install Python dependencies

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
brew install ffmpeg #macOS

```
### Step 2 — Run the backend server

```bash
python app.py
```

### Step 3 — Run the frontend

- Open frontend/ in VS Code
- Install Live Server extension
- Right-click index.html → Open with Live Server
- Visit: http://127.0.0.1:5500/



## Author
<a href="https://github.com/eashaankar">
  <img src="https://github.com/eashaankar.png" width="80" height="80" style="border-radius:50%" />
</a>

**Eashaankar Sahai**  
GitHub: https://github.com/eashaankar

<a href="https://github.com/lyndialin">
  <img src="https://github.com/lyndialin.png" width="80" height="80" style="border-radius:50%" />
</a>

**Lyndia Lin**  
GitHub: https://github.com/lyndialin

