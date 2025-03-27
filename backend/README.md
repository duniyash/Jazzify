# MusicXML Chord Prediction API

A FastAPI application that processes MusicXML files to predict and add chord notations using a transformer-based model.

## Overview

This API accepts MusicXML files as input, processes them to extract musical notes and their properties, predicts appropriate chords using a pre-trained transformer model, and returns a new MusicXML file with the predicted chords added in the bass clef.

## Features

-   **FastAPI Backend**: Modern, high-performance web framework
-   **MusicXML Processing**: Parse and generate MusicXML files using music21
-   **Chord Prediction**: Uses a pre-trained transformer model to predict chords based on melody
-   **Error Handling**: Comprehensive error handling and logging

## Project Structure

```
musicxml-chord-api/
├── main.py             # FastAPI application and endpoints
├── processing.py       # MusicXML parsing and processing functions
├── model.py            # Transformer model definition and prediction logic
├── utils.py            # Helper functions
├── requirements.txt    # Project dependencies
├── transformersv2/     # Directory containing the model files
│   └── chord_predictor.safetensors  # Pre-trained model weights
```

## Installation

## **1. Install Python 3.10 or 3.11**

Since Python 3.13 might not be fully compatible with `music21`, install a stable version like **Python 3.10 or 3.11**.

### **Windows:**

1. Download **Python 3.11 or 3.10** from [python.org](https://www.python.org/downloads/).
2. Install it and check **"Add Python to PATH"** during installation.
3. Verify installation by running:
    ```sh
    python3.11 --version
    ```
    If that doesn’t work, try:
    ```sh
    py -3.11 --version
    ```

### **macOS/Linux:**

1. Install Python using Homebrew (macOS) or APT (Linux):
    ```sh
    brew install python@3.11  # macOS
    sudo apt install python3.11  # Ubuntu/Linux
    ```
2. Verify installation:
    ```sh
    python3.11 --version
    ```

---

## **2. Create a Virtual Environment**

After installing Python, create a virtual environment using Python 3.11.

### **Windows:**

```sh
py -3.11 -m venv myenv
```

### **macOS/Linux:**

```sh
python3.11 -m venv myenv
```

This creates a virtual environment named `myenv`.

---

## **3. Activate the Virtual Environment**

### **Windows (Command Prompt):**

```sh
myenv\Scripts\activate
```

### **macOS/Linux (Terminal):**

```sh
source myenv/bin/activate
```

Once activated, you should see `(myenv)` in your terminal.

---

## **4. Install Dependencies**

After activating the virtual environment, install the required dependencies:

```sh
pip install "requirements/requirements.txt"
```

---

## **5. Run the Application**

Once dependencies are installed, run your FastAPI script:

```sh
python main.py
```

---

## **6. Deactivate the Virtual Environment**

When you're done, deactivate the virtual environment by running:

```sh
deactivate
```

To use it again, simply **activate** it before running your script.

## Running the API

Start the API server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### `POST /process_musicxml`

Processes a MusicXML file to add predicted chords.

**Request**:

-   Content-Type: `application/xml`
-   Body: The processed MusicXML file with predicted chords

### `GET /`

Root endpoint with API information.

**Response**:

-   Content-Type: `application/json`
-   Body: Information about the API

## Usage Example

Using curl:

```bash
curl -X POST "http://localhost:8000/process_musicxml" \
  -H "accept: application/xml" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/file.musicxml" \
  --output processed_file.musicxml
```

Using Python requests:

```python
import requests

url = "http://localhost:8000/process_musicxml"
files = {"file": ("input.musicxml", open("path/to/your/file.musicxml", "rb"), "application/xml")}

response = requests.post(url, files=files)
with open("processed_file.musicxml", "wb") as f:
    f.write(response.content)
```

## Model Details

The chord prediction model is a transformer-based neural network with the following architecture:

-   Embedding layers for note tokens and octaves
-   Linear projection for note durations
-   Transformer encoder with self-attention mechanisms
-   Classification head for chord prediction
-   Regression head for chord duration prediction

The model was trained on a dataset of musical pieces to predict chord progressions based on melodic content.

## Customization

### Using a Different Model

To use a different chord prediction model:

1. Replace the model file at `transformersv2/chord_predictor.safetensors`
2. Update the model parameters in `model.py` if necessary
3. Update the chord and note mappings in `model.py` to match your model

### Extending the API

-   Add additional endpoints in `main.py`
-   Implement new processing functions in `processing.py`
-   Add helper functions in `utils.py`

## License

[MIT License](LICENSE)
