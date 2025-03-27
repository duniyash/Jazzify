"""
Jazzify API
-----------------------------
A FastAPI application that processes MusicXML files to predict and add chords.
"""

import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import uvicorn
from tempfile import NamedTemporaryFile
import shutil

from functions.processing import process_musicxml_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MusicXML Chord Prediction API",
    description="API for processing MusicXML files to predict and add chord notations",
    version="1.0.0"
)

@app.post("/process_musicxml", response_class=FileResponse)
async def process_musicxml(file: UploadFile = File(...)):
    """
    Process a MusicXML file to add predicted chords.
    
    Parameters:
    -----------
    file: UploadFile
        The MusicXML file to process
        
    Returns:
    --------
    FileResponse
        The processed MusicXML file with predicted chords
    """
    logger.info(f"Received file: {file.filename}")
    
    # Validate file extension
    if not file.filename.lower().endswith(('.xml', '.musicxml')):
        raise HTTPException(status_code=400, detail="Only MusicXML files are accepted (.xml, .musicxml)")
    
    # Create temporary files for input and output
    input_temp = NamedTemporaryFile(delete=False, suffix='.xml')
    output_temp = NamedTemporaryFile(delete=False, suffix='.xml')
    
    try:
        # Save uploaded file to temp file
        with input_temp as f:
            shutil.copyfileobj(file.file, f)
        
        # Process the file
        logger.info(f"Processing file: {file.filename}")
        process_musicxml_file(input_temp.name, output_temp.name)
        
        # Return the processed file
        return FileResponse(
            path=output_temp.name, 
            filename=f"processed_{file.filename}",
            media_type="application/xml"
        )
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    finally:
        # Clean up temp files
        file.file.close()
        if os.path.exists(input_temp.name):
            os.unlink(input_temp.name)
        # Output file will be cleaned up by FastAPI after sending the response

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Jazzify API",
        "usage": "POST a MusicXML file to /process_musicxml to get chord predictions"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)