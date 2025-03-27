from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

from functions.processing import process_musicxml_file
from functions.utils import setup_logging

# Set up logging
setup_logging()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_musicxml")
async def process_musicxml_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to process an uploaded MusicXML file.
    Validates the file type, processes the file, and returns a new MusicXML file with predicted chords.
    """
    # Validate file extension (basic check)
    if not (file.filename.endswith('.xml') or file.filename.endswith('.musicxml') or file.filename.endswith('.mxl')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a MusicXML file.")
    
    try:
        # Read the uploaded file's contents
        contents = await file.read()
        # Process the MusicXML file and get an in-memory output stream
        output_stream = process_musicxml_file(contents)
        
        # Return the output MusicXML as a downloadable file
        return StreamingResponse(
            output_stream,
            media_type="application/xml",
            headers={"Content-Disposition": "attachment; filename=processed.musicxml"}
        )
    
    except Exception as e:
        logging.exception("Error processing MusicXML file")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    # Run the app with uvicorn when executed directly
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
