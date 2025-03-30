import io
import logging
from music21 import converter, chord, stream, clef, note, meter
from music21.harmony import ChordSymbol
import tempfile
import os
from model import *


def extract_measures(score):
    """
    Extracts measures from a music21 score.
    Returns a list where each element represents a measure as a list of (note, duration) tuples.
    Only note.Note and note.Rest elements are processed.
    """
    measures_data = []
    # Assume processing from the first part; adjust if multiple parts are needed.
    for measure in score.parts[0].getElementsByClass('Measure'):
        measure_data = []
        for element in measure:
            if isinstance(element, note.Note):
                measure_data.append((element.nameWithOctave, element.quarterLength))
            elif isinstance(element, note.Rest):
                measure_data.append(('Rest', element.quarterLength))
        measures_data.append(measure_data)
    return measures_data


def compile_chords_into_score(chords, time_sig=None):
    """
    Compiles a list of chord symbols into a new music21 score. Each chord is 
    realized as a chord (i.e. actual note objects) and is set to span the full 
    duration of the measure based on the detected time signature.
    
    Args:
        chords (list of str): A list of chord symbol strings.
        time_sig (music21.meter.TimeSignature, optional): The time signature to use.
            If None, defaults to 4/4.
    
    Returns:
        music21.stream.Score: A score with a part containing the realized chords in bass clef.
    """
    new_score = stream.Score()
    part = stream.Part()
    # Set the part to use the Bass Clef.
    part.append(clef.BassClef())
    
    # Use the provided time signature or default to 4/4.
    if time_sig is None:
        time_sig = meter.TimeSignature('4/4')
    part.append(time_sig)
    
    # Determine the bar duration (in quarter notes) from the time signature.
    bar_duration = time_sig.barDuration.quarterLength
    
    for chord_symbol in chords:
        measure = stream.Measure()
        cs = ChordSymbol(chord_symbol)
        # Realize the chord into actual pitches.
        realized_chord = chord.Chord(cs.pitches)
        # Set the chord to span the full measure.
        realized_chord.quarterLength = bar_duration
        
        measure.append(realized_chord)
        part.append(measure)
    
    new_score.append(part)
    return new_score


def process_musicxml_file(file_contents: bytes):
    """
    Full processing workflow for a MusicXML file:
      1. Parse the file with music21.
      2. Extract the melody (treble clef) from the score.
      3. Extract note and duration data per measure from the melody.
      4. Load the chord prediction model.
      5. Predict a chord for each measure.
      6. Auto-detect the time signature from the melody.
      7. Compile the predicted chords into a chord score (bass clef) with measures matching the time signature.
      8. Combine the melody and chord parts into a final score.
      9. Output the final score as a MusicXML file stream.
    """
    print("Processing musicxml file")
    
    # --- Parse the MusicXML file ---
    try:
        file_stream = io.BytesIO(file_contents)
        try:
            score = converter.parse(file_stream)
        except IndexError as e:
            logging.warning("Standard parse failed due to missing metadata. Falling back to parseData.")
            score = converter.parseData(file_contents, format='musicxml')
    except Exception as e:
        logging.exception("Error parsing MusicXML file")
        logging.error(f"File content (first 500 bytes): {file_contents[:500]}")
        raise Exception("Invalid MusicXML file provided.")
    
    # --- Extract the melody (treble clef) from the original score ---
    melody_score = melody_extractor(score)
    
    # --- Extract measures from the melody score ---
    measures_data = extract_measures(melody_score)
    
    # --- Load the chord prediction model ---
    model = load_model("backend/model/chord_predictor.safetensors")
    
    # --- Predict chords for each measure ---
    predicted_chords = []
    for measure_data in measures_data:
        chord_prediction = predict_chord(model, measure_data)
        predicted_chords.append(chord_prediction)
    
    # --- Auto-detect the time signature from the melody score ---
    # Here, we try to get the first TimeSignature object from the melody score.
    ts_elements = melody_score.flat.getElementsByClass(meter.TimeSignature)
    if ts_elements:
        detected_ts = ts_elements[0]
    else:
        detected_ts = None  # Will default to 4/4 in compile_chords_into_score
    
    # --- Compile the predicted chords into a chord score (bass clef) ---
    chord_score = compile_chords_into_score(predicted_chords, time_sig=detected_ts)
    
    # --- Combine the melody and chord parts into a final score ---
    final_score = stream.Score()
    final_score.append(melody_score.parts[0])
    final_score.append(chord_score.parts[0])
    
    # --- Write the final score to a temporary file and load it as BytesIO ---
    with tempfile.NamedTemporaryFile(delete=False, suffix='.musicxml') as tmp:
        temp_filename = tmp.name

    final_score.write('musicxml', fp=temp_filename)

    with open(temp_filename, 'rb') as f:
        output_data = f.read()
    os.remove(temp_filename)

    output_io = io.BytesIO(output_data)
    output_io.seek(0)
    return output_io



