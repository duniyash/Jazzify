import io
import logging
from music21 import converter, chord, stream, clef, note, meter
from music21.harmony import ChordSymbol
import tempfile
import os
from model import *

def extract_half_measures(score):
    """
    Extracts two halves from each measure in a music21 score.
    Returns a list of tuples, each containing two lists of (note, duration) tuples representing the halves.
    """
    half_measures_data = []
    for measure in score.parts[0].getElementsByClass('Measure'):
        total_duration = measure.quarterLength / 2
        first_half, second_half = [], []
        current_duration = 0.0

        for element in measure:
            if isinstance(element, (note.Note, note.Rest)):
                elem_data = (element.nameWithOctave if isinstance(element, note.Note) else 'Rest', element.quarterLength)
                if current_duration + element.quarterLength <= total_duration:
                    first_half.append(elem_data)
                    current_duration += element.quarterLength
                else:
                    second_half.append(elem_data)

        half_measures_data.append((first_half, second_half))
    return half_measures_data

def compile_double_chords_into_score(double_chords, time_sig=None):
    """
    Compiles a list of pairs of chord symbols into a new music21 score with two chords per measure.
    Each chord spans half a measure.
    """
    new_score = stream.Score()
    part = stream.Part()
    part.append(clef.BassClef())

    if time_sig is None:
        time_sig = meter.TimeSignature('4/4')
    part.append(time_sig)

    half_duration = time_sig.barDuration.quarterLength / 2

    for i, (chord1, chord2) in enumerate(double_chords, start=1):
        measure = stream.Measure(number=i)
        cs1 = ChordSymbol(chord1)
        cs2 = ChordSymbol(chord2)

        ch1 = chord.Chord(cs1.pitches)
        ch1.quarterLength = half_duration
        ch2 = chord.Chord(cs2.pitches)
        ch2.quarterLength = half_duration

        measure.append(ch1)
        measure.append(ch2)
        part.append(measure)

    new_score.append(part)
    return new_score


def process_musicxml_file(file_contents: bytes):
    print("Processing musicxml file")

    try:
        file_stream = io.BytesIO(file_contents)
        try:
            score = converter.parse(file_stream)
        except Exception:
            score = converter.parseData(file_contents, format='musicxml')
    except Exception:
        raise Exception("Invalid MusicXML file provided.")

    melody_score = melody_extractor(score)
    half_measure_data = extract_half_measures(melody_score)
    model = load_model("backend\chord_predictor.safetensors")

    predicted_chords = []
    for half1, half2 in half_measure_data:
        pred1 = predict_chord(model, half1)
        pred2 = predict_chord(model, half2)
        predicted_chords.append((pred1, pred2))

    ts_elements = melody_score.flat.getElementsByClass(meter.TimeSignature)
    detected_ts = ts_elements[0] if ts_elements else None

    chord_score = compile_double_chords_into_score(predicted_chords, time_sig=detected_ts)

    final_score = stream.Score()
    final_score.append(melody_score.parts[0])
    final_score.append(chord_score.parts[0])

    with tempfile.NamedTemporaryFile(delete=False, suffix='.musicxml') as tmp:
        temp_filename = tmp.name

    final_score.write('musicxml', fp=temp_filename)

    with open(temp_filename, 'rb') as f:
        output_data = f.read()
    os.remove(temp_filename)

    output_io = io.BytesIO(output_data)
    output_io.seek(0)
    return output_io