# Sound Detector

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

