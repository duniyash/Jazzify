# Jazzify Project

Welcome to the Jazzify project – a final year project aimed at exploring and automating chord prediction for musical scores. This repository contains the backend code responsible for processing MusicXML files, extracting melody data, predicting chord symbols using a deep learning model, and compiling the final score.

---

## Project Structure

The main components of the project are organized under the `backend/` directory:

-   **main.py**

    -   Serves as the entry point to the FastAPI web application.
    -   Defines API endpoints to accept MusicXML file uploads and return processed files.
    -   Integrates with the processing module to execute the full MusicXML processing workflow.

-   **processing.py**

    -   Contains functions that parse MusicXML files using the `music21` library.
    -   Implements extraction of measures from a musical score, handling both notes and rests.
    -   Compiles the extracted melody data and predicted chord symbols into a final music21 score formatted for output.
    -   Manages error handling during file parsing with robust logging and fallback mechanisms.

-   **model.py**

    -   Defines the deep learning model (ChordPredictor) used for predicting chord symbols.
    -   Provides helper functions for tokenizing notes, loading the pre-trained model (from safetensors), and performing the chord prediction.
    -   Uses PyTorch to handle model computations and sequential data.

-   **.gitignore**
    -   Lists files and directories that should be ignored by Git (e.g., `myenv/` for local Python virtual environments).

_Additional files (such as `utils.py`, if present) may contain auxiliary functionality like logging configuration and other utility functions._

---

## Dependencies

This project relies on several external libraries and frameworks:

-   **Python 3.x**
-   **FastAPI** & **Uvicorn**
    -   Used to create and serve the API endpoints.
-   **music21**
    -   For parsing MusicXML files and manipulating musical scores.
-   **PyTorch**
    -   To implement and run the chord prediction model.
-   **Other dependencies**
    -   Standard libraries such as `logging`, `tempfile`, and `os` for file handling and debugging.

_Make sure to install the required packages, for example via:_

```bash
pip install fastapi uvicorn music21 torch
```

---

## Setup and Execution

1. **Clone the Repository:**  
   Clone the repository to your local machine and navigate to the project directory.

2. **Virtual Environment:**  
   Create and activate a Python virtual environment (recommended to keep dependencies isolated).

3. **Install Dependencies:**  
   Install the necessary packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

    _(Alternatively, install the main dependencies manually as listed above.)_

4. **Running the Server:**  
   Start the backend server using Uvicorn:

    ```bash
    uvicorn backend.main:app --reload
    ```

    The API will be available at `http://localhost:8000`.

5. **Using the API:**  
   To process a MusicXML file, send a POST request to the `/process_musicxml` endpoint with the file attached. The server will return the processed MusicXML file with chords integrated into the score.

---

## Chord Prediction Pipeline Overview

1. **Parsing:**  
   The input MusicXML file is parsed using `music21`. Robust error handling ensures that if the primary parser fails, a fallback parsing method is attempted.

2. **Melody Extraction:**  
   The melody is extracted (typically from the treble clef) and divided into measures. Each measure is represented as a sequence of notes (or rests) with their respective durations.

3. **Model Prediction:**  
   The extracted measure data is formatted and used as input into the ChordPredictor model (defined in `model.py`). The model, built using PyTorch, predicts a chord symbol for each measure.

4. **Score Compilation:**  
   Predicted chords are compiled into a new score using `music21`, aligning with the detected or default time signature. This new chord score is then merged with the extracted melody to form the final score.

5. **Output Generation:**  
   The final score with the integrated chord symbols is written out as a MusicXML stream, which can then be downloaded.

---

## Future Work & Contributions

Since the project is still under construction, there is plenty of room for improvement, including:

-   **Enhancing Multi-Part Support:**  
    Expanding the code to handle scores with multiple parts beyond the primary melody.

-   **Model Improvements:**  
    Refining the chord prediction model and exploring more advanced architectures.

-   **User Interface Enhancements:**  
    Potentially developing a front-end interface to interact with the API more seamlessly.

-   **Error Handling & Logging:**  
    Adding more detailed error handling and logging to improve debugging during development and in production.

Contributions, suggestions, and feedback are welcome to help enhance the Jazzify project further.

---

## License

_(Include licensing information here as applicable.)_

Enjoy exploring and contributing to Jazzify – where music meets machine learning!
