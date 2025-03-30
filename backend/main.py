from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

from functions.processing import process_musicxml_file, maybe_convert_mxl
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

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/test")
async def test(file: UploadFile = File(...)):
    return {"message": "Test endpoint", "filename": file.filename}

@app.post("/process_musicxml")
async def process_musicxml_endpoint(file: UploadFile = File(...)):
    logging.info(f"Received file: {file.filename}, content type: {file.content_type}")
    try:
        contents = await file.read()
        logging.info(f"Read {len(contents)} bytes from file")
        output_stream = process_musicxml_file(contents)
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
