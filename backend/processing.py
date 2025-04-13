import io
import logging
from music21 import converter, chord, stream, clef, note, meter, expressions
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

def compile_chords_into_score(chords, time_sig=None, arpeggiate=True):
    """
    Compiles a list of chord symbol strings into a new music21 score.
    If arpeggiate is True, adds a visible arpeggio sign to each chord.
    """
    new_score = stream.Score()
    part = stream.Part()
    part.append(clef.BassClef())

    if time_sig is None:
        time_sig = meter.TimeSignature('4/4')
    part.append(time_sig)

    bar_duration = time_sig.barDuration.quarterLength

    for i, chord_symbol in enumerate(chords, start=1):
        measure = stream.Measure(number=i)
        cs = ChordSymbol(chord_symbol)
        realized_chord = chord.Chord(cs.pitches)
        realized_chord.quarterLength = bar_duration

        if arpeggiate:
            # Add arpeggio marking for visual notation
            arpeggio_mark = expressions.ArpeggioMark()
            realized_chord.expressions.append(arpeggio_mark)

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
      7. Compile the predicted chords into a chord score (bass clef) with full-measure chords
         and proper bar numbers.
      8. Combine the melody and chord parts into a final score.
      9. Output the final score as a MusicXML file stream.
    """
    print("Processing musicxml file")
    
    # --- Parse the MusicXML file with enhanced debugging ---
    try:
        file_stream = io.BytesIO(file_contents)
        try:
            score = converter.parse(file_stream)
            logging.info("Parsed MusicXML file using converter.parse successfully.")
        except Exception as e:
            logging.error(f"converter.parse failed with error: {e}")
            # Fallback: try using parseData
            try:
                score = converter.parseData(file_contents, format='musicxml')
                logging.info("Parsed MusicXML file using converter.parseData successfully.")
            except Exception as e2:
                logging.error(f"converter.parseData failed with error: {e2}")
                raise Exception("Invalid MusicXML file provided.")
    except Exception as e:
        logging.exception("Error parsing MusicXML file")
        logging.error(f"File content (first 500 bytes): {file_contents[:500]}")
        raise Exception("Invalid MusicXML file provided.")
    
    # --- Extract the melody (treble clef) from the original score ---
    melody_score = melody_extractor(score)
    
    # --- Extract measures from the melody score ---
    measures_data = extract_measures(melody_score)
    
    # --- Load the chord prediction model ---
    model = load_model("chord_predictor.safetensors")
    
    # --- Predict chords for each measure ---
    predicted_chords = []
    for measure_data in measures_data:
        chord_prediction = predict_chord(model, measure_data)
        predicted_chords.append(chord_prediction)
    
    # --- Auto-detect the time signature from the melody score ---
    ts_elements = melody_score.flat.getElementsByClass(meter.TimeSignature)
    detected_ts = ts_elements[0] if ts_elements else None
    
    # --- Compile the predicted chords into a chord score (bass clef) with proper bar numbers ---
    chord_score = compile_chords_into_score(predicted_chords, time_sig=detected_ts, arpeggiate=True)
    
    # --- Combine the melody and chord parts into a final score ---
    final_score = stream.Score()
    final_score.append(melody_score.parts[0])
    final_score.append(chord_score.parts[0])

    # --- Add swing feel text expression at the beginning ---
    swing_text = expressions.TextExpression("Swing")
    swing_text.placement = 'above'
    final_score.insert(0, swing_text)
    
    # --- Write the final score to a temporary MusicXML file and return as a BytesIO stream ---
    with tempfile.NamedTemporaryFile(delete=False, suffix='.musicxml') as tmp:
        temp_filename = tmp.name

    final_score.write('musicxml', fp=temp_filename)

    with open(temp_filename, 'rb') as f:
        output_data = f.read()
    os.remove(temp_filename)

    output_io = io.BytesIO(output_data)
    output_io.seek(0)
    return output_io
