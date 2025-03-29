import io
import logging
from music21 import converter, chord, stream, clef, note, meter

from model import load_model, predict_chord

def extract_measures(score):
    """
    Extracts measures from a music21 score.
    Returns a list where each element represents a measure as a list of (note, duration) tuples.
    """
    measures_data = []
    # Assume processing from the first part; adjust if multiple parts are needed.
    for measure in score.parts[0].getElementsByClass('Measure'):
        measure_data = []
        for element in measure:
            if isinstance(element, note.Note):
                measure_data.append((element.nameWithOctave, element.quarterLength))
            # Optionally include rests
            elif element.isRest:
                measure_data.append(('Rest', element.quarterLength))
        measures_data.append(measure_data)
    return measures_data

def compile_chords_into_score(chords):
    """
    Compiles a list of predicted chord symbols into a new music21 score.
    Each chord is added into its own measure in the bass clef.
    """
    new_score = stream.Score()
    part = stream.Part()
    part.append(clef.BassClef())
    # Optionally, set a default time signature
    part.append(meter.TimeSignature('4/4'))
    
    for chord_symbol in chords:
        m = stream.Measure()
        # Create a chord object; here, we assume chord_symbol is a valid chord string.
        chord_obj = chord.ChordSymbol(chord_symbol)
        m.append(chord_obj)
        part.append(m)
    new_score.append(part)
    return new_score

def process_musicxml_file(file_contents: bytes):
    """
    Full processing workflow for a MusicXML file:
      1. Parse the file with music21.
      2. Extract note and duration data per measure.
      3. Load the chord prediction model.
      4. Predict a chord for each measure.
      5. Compile the predicted chords into a new score (bass clef).
      6. Output the new score as a MusicXML file stream.
    """
    print("Processing musicxml file")
    try:
        # Parse the MusicXML file from bytes
        score = converter.parse(io.BytesIO(file_contents))
    except Exception as e:
        logging.exception("Error parsing MusicXML file")
        raise Exception("Invalid MusicXML file provided.")
    
    # Extract measures with notes and their durations
    measures_data = extract_measures(score)
    
    # Load the chord prediction model (dummy path provided)
    model = load_model("model/chord_predictor.safetensors")
    
    # Predict chords for each measure
    predicted_chords = []
    for measure_data in measures_data:
        # Preprocess if necessary before prediction
        chord_prediction = predict_chord(model, measure_data)
        predicted_chords.append(chord_prediction)
    
    # Compile the predicted chords into a new music21 score in the bass clef
    new_score = compile_chords_into_score(predicted_chords)
    
    # Write the new score to an in-memory MusicXML file
    output_io = io.BytesIO()
    new_score.write('musicxml', fp=output_io)
    output_io.seek(0)
    return output_io
