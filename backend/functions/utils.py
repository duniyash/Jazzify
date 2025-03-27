"""
Utility Functions
---------------
Helper functions for MusicXML processing and validation.
"""

import os
import logging
import music21

logger = logging.getLogger(__name__)

def is_valid_musicxml(file_path):
    """
    Check if a file is a valid MusicXML file.
    
    Parameters:
    -----------
    file_path : str
        Path to the file to check
        
    Returns:
    --------
    bool
        True if the file is a valid MusicXML file, False otherwise
    """
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False
    
    # Check file extension
    if not file_path.lower().endswith(('.xml', '.musicxml')):
        logger.error(f"File is not a MusicXML file: {file_path}")
        return False
    
    # Try to parse the file with music21
    try:
        music21.converter.parse(file_path)
        return True
    except Exception as e:
        logger.error(f"Error parsing MusicXML file: {str(e)}")
        return False

def measure_to_features(measure):
    """
    Extract notes, octaves, and durations from a measure for model input.
    
    Parameters:
    -----------
    measure : music21.stream.Measure
        The measure to extract features from
        
    Returns:
    --------
    tuple
        (notes, octaves, durations)
    """
    notes = []
    octaves = []
    durations = []
    
    # Process each note in the measure
    for element in measure.notes:
        # Handle both Note and Chord objects
        if element.isNote:
            process_note(element, notes, octaves, durations)
        elif element.isChord:
            # For chords, process each note separately
            for note in element.notes:
                process_note(note, notes, octaves, durations)
    
    return notes, octaves, durations

def process_note(note, notes_list, octaves_list, durations_list):
    """
    Process a single note and add its features to the provided lists.
    
    Parameters:
    -----------
    note : music21.note.Note
        The note to process
    notes_list : list
        List to append the note name to
    octaves_list : list
        List to append the octave to
    durations_list : list
        List to append the duration to
    """
    # Skip rests
    if note.isRest:
        return
    
    # Get note name
    note_name = note.name  # Returns note name like 'C', 'D', 'E', etc.
    
    # Get octave
    octave = note.octave
    
    # Get duration in quarter notes
    duration = note.duration.quarterLength
    
    # Append to lists
    notes_list.append(note_name)
    octaves_list.append(octave if octave is not None else 4)  # Default to octave 4 if None
    durations_list.append(float(duration))

def normalize_chord_name(chord_name):
    """
    Normalize chord names to a consistent format.
    
    Parameters:
    -----------
    chord_name : str
        The chord name to normalize
        
    Returns:
    --------
    str
        The normalized chord name
    """
    # Basic normalization rules - expand as needed
    chord_name = chord_name.replace('maj', 'M')
    chord_name = chord_name.replace('min', 'm')
    chord_name = chord_name.replace('dim', 'o')
    chord_name = chord_name.replace('aug', '+')
    
    return chord_name

def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it doesn't exist.
    
    Parameters:
    -----------
    directory_path : str
        Path to the directory to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")