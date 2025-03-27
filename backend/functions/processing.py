"""
MusicXML Processing Module
-------------------------
Functions for parsing, processing, and generating MusicXML files.
"""

import logging
import music21
from fractions import Fraction
from model import predict_chord
from functions.utils import measure_to_features, is_valid_musicxml

logger = logging.getLogger(__name__)

def process_musicxml_file(input_path, output_path):
    """
    Process a MusicXML file to add predicted chords.
    
    Parameters:
    -----------
    input_path : str
        Path to the input MusicXML file
    output_path : str
        Path where the processed MusicXML file will be saved
        
    Returns:
    --------
    bool
        True if processing was successful, False otherwise
    """
    logger.info(f"Processing MusicXML file: {input_path}")
    
    # Validate input file
    if not is_valid_musicxml(input_path):
        raise ValueError(f"Invalid MusicXML file: {input_path}")
    
    try:
        # Parse the input MusicXML file
        score = music21.converter.parse(input_path)
        
        # Create a new score for output
        output_score = music21.stream.Score()
        
        # Process parts and add chord part
        process_score(score, output_score)
        
        # Write the processed score to the output file
        output_score.write('musicxml', fp=output_path)
        logger.info(f"Processed MusicXML saved to: {output_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing MusicXML: {str(e)}")
        raise RuntimeError(f"Failed to process MusicXML: {str(e)}")

def process_score(input_score, output_score):
    """
    Process a music21 score object to add predicted chords.
    
    Parameters:
    -----------
    input_score : music21.stream.Score
        The input score to process
    output_score : music21.stream.Score
        The output score to add processed parts to
    """
    # Get metadata from the original score
    transfer_metadata(input_score, output_score)
    
    # Create a part for the original melody
    melody_part = music21.stream.Part()
    melody_part.id = 'Melody'
    melody_part.partName = 'Melody'
    
    # Create a part for the chord accompaniment
    chord_part = music21.stream.Part()
    chord_part.id = 'Chords'
    chord_part.partName = 'Chords'
    
    # Add bass clef to chord part
    chord_part.append(music21.clef.BassClef())
    
    # Extract the main part from the input score (assumes single part for simplicity)
    # For multi-part scores, we might need more sophisticated logic
    if len(input_score.parts) > 0:
        main_part = input_score.parts[0]
    else:
        main_part = input_score  # If the score has no parts, treat it as a single part
    
    # Copy the melody part and add predicted chords
    process_part(main_part, melody_part, chord_part)
    
    # Add parts to the output score
    output_score.append(melody_part)
    output_score.append(chord_part)

def transfer_metadata(input_score, output_score):
    """
    Transfer metadata from input score to output score.
    
    Parameters:
    -----------
    input_score : music21.stream.Score
        The input score to get metadata from
    output_score : music21.stream.Score
        The output score to add metadata to
    """
    # Transfer title, composer, etc.
    if input_score.metadata is not None:
        output_score.metadata = input_score.metadata
    else:
        # Create default metadata if none exists
        metadata = music21.metadata.Metadata()
        metadata.title = "Processed Score with Predicted Chords"
        output_score.metadata = metadata

def process_part(input_part, melody_part, chord_part):
    """
    Process a music21 part to extract notes and predict chords.
    
    Parameters:
    -----------
    input_part : music21.stream.Part
        The input part to process
    melody_part : music21.stream.Part
        The output melody part to populate
    chord_part : music21.stream.Part
        The output chord part to populate with predicted chords
    """
    # Copy time signature, key signature, etc.
    for element in input_part.flatten().getElementsByClass('Measure')[0].getElementsByClass(['TimeSignature', 'KeySignature']):
        melody_part.append(element)
        chord_part.append(element.copy())
    
    # Process each measure
    for i, measure in enumerate(input_part.getElementsByClass('Measure')):
        # Create new measures for melody and chords
        new_melody_measure = music21.stream.Measure(number=measure.number)
        new_chord_measure = music21.stream.Measure(number=measure.number)
        
        # Copy measure attributes (barlines, etc.)
        for attr in ['leftBarline', 'rightBarline', 'mergedAttributes']:
            if hasattr(measure, attr) and getattr(measure, attr) is not None:
                setattr(new_melody_measure, attr, getattr(measure, attr))
                setattr(new_chord_measure, attr, getattr(measure, attr))
        
        # Extract notes, octaves, and durations from the measure
        notes, octaves, durations = extract_measure_features(measure)
        
        # Copy the original notes to the melody measure
        for element in measure:
            new_melody_measure.append(element)
        
        # Only predict chord if we have notes
        if notes:
            # Predict chord for this measure
            chord_name, chord_duration = predict_chord(notes, octaves, durations)
            
            # Create chord symbol and add to chord measure
            chord_symbol = create_chord(chord_name, chord_duration)
            new_chord_measure.append(chord_symbol)
            
            logger.info(f"Measure {measure.number}: Predicted chord {chord_name}")
        else:
            # Add rest if no notes to predict from
            rest = music21.note.Rest()
            rest.duration = music21.duration.Duration(4.0)  # Default to whole note rest
            new_chord_measure.append(rest)
        
        # Add measures to parts
        melody_part.append(new_melody_measure)
        chord_part.append(new_chord_measure)

def extract_measure_features(measure):
    """
    Extract notes, octaves, and durations from a measure.
    
    Parameters:
    -----------
    measure : music21.stream.Measure
        The measure to extract features from
        
    Returns:
    --------
    tuple
        (notes, octaves, durations)
    """
    return measure_to_features(measure)

def create_chord(chord_name, duration):
    """
    Create a music21 chord object from a chord name and duration.
    
    Parameters:
    -----------
    chord_name : str
        The name of the chord (e.g., 'C', 'G7', 'Dm')
    duration : float
        The duration of the chord in quarter notes
        
    Returns:
    --------
    music21.harmony.ChordSymbol
        A chord symbol object
    """
    # Create a chord symbol
    chord_symbol = music21.harmony.ChordSymbol(chord_name)
    
    # Set duration
    chord_symbol.duration = music21.duration.Duration(duration)
    
    # Create a realization of the chord (optional)
    chord_symbol.writeAsChord = True
    
    return chord_symbol