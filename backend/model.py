import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from music21 import stream, clef, note, chord
from safetensors.torch import load_file


# ----- Example Encoding Dictionaries (replace with your actual mappings) -----
note2idx = {
    'A': 1, 'A#': 2, 'A-': 3, 'B': 4, 'B#': 5, 'B-': 6, 'B--': 7, 'C': 8, 'C#': 9, 'C-': 10,
    'D': 11, 'D#': 12, 'D-': 13, 'E': 14, 'E#': 15, 'E-': 16, 'E--': 17, 'F': 18, 'F#': 19,
    'F-': 20, 'G': 21, 'G#': 22, 'G-': 23, 'G--': 24, 'G---': 25
}  # 0 is reserved for padding
idx2note = {idx: note for note, idx in note2idx.items()}

chord2idx = {
    'A': 0, 'A#m7': 1, 'A-': 2, 'A-+': 3, 'A-/B': 4, 'A-/B-': 5, 'A-/E-': 6, 'A-/G-': 7, 'A-13': 8,
    'A-6': 9, 'A-7': 10, 'A-7 add #11': 11, 'A-7 add #9': 12, 'A-9': 13, 'A-M13': 14, 'A-M13 alter #11': 15,
    'A-M9': 16, 'A-dim': 17, 'A-m': 18, 'A-m/B-': 19, 'A-m/E-': 20, 'A-m11': 21, 'A-m7': 22, 'A-m7/B-': 23,
    'A-m7/D-': 24, 'A-m9': 25, 'A-maj7': 26, 'A-maj7/B-': 27, 'A-sus': 28, 'A-sus add 7': 29, 'A-sus/B- add 7': 30,
    'A/E': 31, 'A13': 32, 'A6': 33, 'A7': 34, 'A7 add #11': 35, 'A7 add #9': 36, 'A7 add b9': 37, 'A9': 38,
    'A9 add #11': 39, 'AM13': 40, 'AM9': 41, 'Adim': 42, 'Am': 43, 'Am11': 44, 'Am7': 45, 'Am7 alter b5': 46,
    'Am7/D': 47, 'Am7/G': 48, 'Am9': 49, 'Amaj7': 50, 'Ao7': 51, 'Asus': 52, 'B': 53, 'B add 9': 54, 'B-': 55,
    'B-/A': 56, 'B-/A-': 57, 'B-/B': 58, 'B-/C': 59, 'B-/D': 60, 'B-/F': 61, 'B-13': 62, 'B-13 alter #9': 63,
    'B-6': 64, 'B-7': 65, 'B-7 add #11': 66, 'B-7 add #9': 67, 'B-7/C': 68, 'B-7/D#': 69, 'B-9': 70, 'B-M9': 71,
    'B-m': 72, 'B-m11': 73, 'B-m6': 74, 'B-m7': 75, 'B-m7/E-': 76, 'B-m7/F': 77, 'B-m7/G': 78, 'B-m9': 79,
    'B-mM7': 80, 'B-maj7': 81, 'B-maj7/C': 82, 'B-o7': 83, 'B-sus add 7': 84, 'B-sus/C add 7': 85, 'B/D': 86,
    'B/D-': 87, 'B/E': 88, 'B/E-': 89, 'B13': 90, 'B7': 91, 'B7 add #9': 92, 'B7 add b9': 93, 'B7/A': 94,
    'B9': 95, 'B9 add #11': 96, 'B9/F#': 97, 'BM9': 98, 'Bdim': 99, 'Bm': 100, 'Bm/F#': 101, 'Bm11': 102,
    'Bm13': 103, 'Bm6': 104, 'Bm7': 105, 'Bm7 alter b5': 106, 'Bm7/A': 107, 'Bm9': 108, 'BmM7': 109,
    'Bmaj7': 110, 'Bmaj7 add #11': 111, 'Bmaj7/D#': 112, 'Bmaj7/F': 113, 'Bo7': 114, 'Bsus': 115, 'C': 116,
    'C#': 117, 'C#7': 118, 'C#7 add #9': 119, 'C#dim': 120, 'C#dim/B-': 121, 'C#m': 122, 'C#m11': 123,
    'C#m13': 124, 'C#m7': 125, 'C#maj7': 126, 'C#o7': 127, 'C#sus add 7': 128, 'C/A': 129, 'C/A-': 130,
    'C/B-': 131, 'C/D': 132, 'C/E': 133, 'C/F#': 134, 'C/G': 135, 'C13': 136, 'C6': 137, 'C7': 138,
    'C7 add #11': 139, 'C7 add #9': 140, 'C7 add b9': 141, 'C7+': 142, 'C7/E': 143, 'C7/G': 144, 'C9': 145,
    'CM9': 146, 'Cdim': 147, 'Cm': 148, 'Cm/B-': 149, 'Cm11': 150, 'Cm13': 151, 'Cm6': 152, 'Cm7': 153,
    'Cm7 alter b5': 154, 'Cm7/B-': 155, 'Cm9': 156, 'CmM7': 157, 'Cmaj7': 158, 'Cmaj7 add #11': 159, 'Co7': 160,
    'Co7/B-': 161, 'Csus': 162, 'Csus add 7': 163, 'D': 164, 'D#': 165, 'D#7': 166, 'D#9': 167, 'D#dim': 168,
    'D#m': 169, 'D#m7': 170, 'D#m7 alter b5': 171, 'D#o7': 172, 'D-': 173, 'D-/A-': 174, 'D-/E': 175,
    'D-/E-': 176, 'D-/G-': 177, 'D-13': 178, 'D-7': 179, 'D-7 add #11': 180, 'D-7 add #9': 181, 'D-9': 182,
    'D-M13': 183, 'D-M13 alter #11': 184, 'D-M9': 185, 'D-dim': 186, 'D-m': 187, 'D-m11': 188, 'D-m7': 189,
    'D-maj7': 190, 'D-maj7/C': 191, 'D-maj7/E-': 192, 'D-maj7/F': 193, 'D-sus add 7': 194, 'D/E': 195, 'D13': 196,
    'D6': 197, 'D7': 198, 'D7 add #11': 199, 'D7 add #9': 200, 'D7 add b9': 201, 'D7+': 202, 'D9': 203,
    'D9 add #11': 204, 'DM13': 205, 'Ddim': 206, 'Dm': 207, 'Dm/A': 208, 'Dm/E': 209, 'Dm11': 210, 'Dm13': 211,
    'Dm7': 212, 'Dm7 alter b5': 213, 'Dm9': 214, 'DmM7 add 9': 215, 'Dmaj7': 216, 'Dmaj7/E': 217, 'Do7': 218,
    'Dpower': 219, 'Dsus add 7': 220, 'E': 221, 'E-': 222, 'E-/B-': 223, 'E-/D-': 224, 'E-/E': 225, 'E-13': 226,
    'E-6': 227, 'E-7': 228, 'E-7 add #11': 229, 'E-7 add #9': 230, 'E-7 add #9 add #11': 231, 'E-7 alter b5': 232,
    'E-9': 233, 'E-9 add #11': 234, 'E-9 add 13': 235, 'E-M13': 236, 'E-M13 alter #11': 237, 'E-M9': 238,
    'E-dim': 239, 'E-m': 240, 'E-m11': 241, 'E-m6': 242, 'E-m7': 243, 'E-m7/D-': 244, 'E-m9': 245, 'E-maj7': 246,
    'E-maj7/F': 247, 'E-o7': 248, 'E/A': 249, 'E/B': 250, 'E/D': 251, 'E/F': 252, 'E/F#': 253, 'E/G#': 254,
    'E13': 255, 'E7': 256, 'E7 add #9': 257, 'E7 alter #5': 258, 'E7 alter b5': 259, 'E7+': 260, 'E7/F': 261,
    'E9': 262, 'EM13': 263, 'EM9': 264, 'Edim': 265, 'Em': 266, 'Em11': 267, 'Em13': 268, 'Em7': 269,
    'Em7 add 11': 270, 'Em7 alter b5': 271, 'Em7/D': 272, 'Em9': 273, 'Emaj7': 274, 'Emaj7/F#': 275, 'Eo7': 276,
    'Esus add 7': 277, 'F': 278, 'F#': 279, 'F#/E': 280, 'F#13': 281, 'F#7': 282, 'F#7 add #9': 283, 'F#7 alter b5': 284,
    'F#7/C#': 285, 'F#9': 286, 'F#dim': 287, 'F#m': 288, 'F#m/A': 289, 'F#m11': 290, 'F#m7': 291, 'F#m7 add 11': 292,
    'F#m7 alter b5': 293, 'F#m9': 294, 'F#maj7': 295, 'F#o7': 296, 'F#sus': 297, 'F#sus add 7': 298, 'F/B-': 299,
    'F/C': 300, 'F/E': 301, 'F/E-': 302, 'F/G': 303, 'F13': 304, 'F6': 305, 'F7': 306, 'F7 add #11': 307,
    'F7 add #9': 308, 'F7/A': 309, 'F9': 310, 'F9 add #11': 311, 'Fdim': 312, 'Fm': 313, 'Fm/C': 314, 'Fm11': 315,
    'Fm6': 316, 'Fm7': 317, 'Fm7 alter b5': 318, 'Fm7/E-': 319, 'Fm9': 320, 'FmM7': 321, 'FmM7 add 9': 322,
    'Fmaj7': 323, 'Fmaj7/G': 324, 'Fo7': 325, 'Fpower': 326, 'Fsus add 7': 327, 'G': 328, 'G#': 329, 'G#7': 330,
    'G#m7': 331, 'G#maj7': 332, 'G#o7': 333, 'G#sus': 334, 'G-': 335, 'G-13': 336, 'G-7': 337, 'G-7 add #11': 338,
    'G-9': 339, 'G-M13 alter #11': 340, 'G-dim': 341, 'G-m': 342, 'G-m11': 343, 'G-m7': 344, 'G-m9': 345,
    'G-maj7': 346, 'G-maj7/A-': 347, 'G-o7': 348, 'G-sus': 349, 'G/A': 350, 'G/B-': 351, 'G13': 352, 'G6': 353,
    'G7': 354, 'G7 add #11': 355, 'G7 add #9': 356, 'G7 add b9': 357, 'G7 alter #5': 358, 'G7 alter b5': 359,
    'G7/F': 360, 'G9': 361, 'GM13': 362, 'Gdim': 363, 'Gm': 364, 'Gm/B-': 365, 'Gm/E': 366, 'Gm/F': 367,
    'Gm/G-': 368, 'Gm11': 369, 'Gm13': 370, 'Gm6': 371, 'Gm7': 372, 'Gm7 alter b5': 373, 'Gm7/B-': 374,
    'Gm7/C': 375, 'Gm7/F': 376, 'Gm7/G-': 377, 'Gm9': 378, 'GmM7': 379, 'GmM7 add 9 add 11': 380, 'Gmaj7': 381,
    'Gmaj7/A': 382, 'Gsus': 383
}
idx2chord = {idx: chord for chord, idx in chord2idx.items()}

# ----- Hyperparameters -----
vocab_size = len(note2idx) + 1  # +1 for padding (0)
embed_dim = 32
num_heads = 4
hidden_dim = 256
num_layers = 4
num_classes = len(chord2idx)
max_seq_length = 32
num_octaves = 10

# ----- Model Definition -----
class ChordPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes, max_seq_length, num_octaves=10):
        """
        num_octaves: maximum number of octave categories (adjust if your octave range is larger)
        """
        super(ChordPredictor, self).__init__()
        # * Embedding for note tokens
        self.note_embed = nn.Embedding(vocab_size, embed_dim)
        # * Embedding for octave values (assumes octave values are small integers)
        self.octave_embed = nn.Embedding(num_octaves, embed_dim)
        # * Linear projection for note durations (continuous values)
        self.duration_linear = nn.Linear(1, embed_dim)

        # * Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))

        # * Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # * Additional fully connected layers after transformer pooling
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # * Two output heads:
        # * - For chord classification
        self.fc_class = nn.Linear(hidden_dim // 2, num_classes)
        # * - For chord duration regression
        self.fc_duration = nn.Linear(hidden_dim // 2, 1)

    def forward(self, tokens, octaves, note_durations):
        # ? 
        # ? tokens: LongTensor of shape [batch_size, seq_length]
        # ? octaves: LongTensor of shape [batch_size, seq_length]
        # ? note_durations: FloatTensor of shape [batch_size, seq_length]

        token_emb = self.note_embed(tokens)           # ! [B, L, embed_dim]
        octave_emb = self.octave_embed(octaves)         # ! [B, L, embed_dim]
        duration_emb = self.duration_linear(note_durations.unsqueeze(-1))  # ! [B, L, embed_dim]

        # * Sum embeddings and add positional encoding
        x = token_emb + octave_emb + duration_emb + self.pos_embedding  # ! [B, L, embed_dim]

        # * Transformer expects input of shape [L, B, embed_dim]
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)

        # * Use the first token's output as a pooled representation
        pooled = x[0]  # ! [B, embed_dim]

        # * Additional layers for further processing
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # * Final output heads
        chord_logits = self.fc_class(x)              # ! [B, num_classes]
        chord_duration = self.fc_duration(x).squeeze(-1)  # ! [B]
        return chord_logits, chord_duration

# ----- Helper Function: Parse a Note String -----
def parse_note(note_str):
    # * Parses a note string (e.g., 'C4', 'C#4', 'Rest') into its note name and octave.
    # * If the note is a 'Rest', returns ("Rest", 0).
    if note_str == "Rest":
        return "Rest", 0
    match = re.match(r"([A-G][#-]*)(\d+)", note_str)
    if match:
        note_name = match.group(1)
        octave = int(match.group(2))
        return note_name, octave
    else:
        # * If no octave is found, assume a default octave (e.g., 4)
        return note_str, 4

# ----- Load the Model -----
def load_model(filepath: str):
    model = ChordPredictor(
        vocab_size,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        num_classes,
        max_seq_length,
        num_octaves
    )
    # Load the state dictionary using the safetensors loader
    state_dict = load_file(filepath)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ----- Prediction Function -----
def predict_chord(model, measure_data):
    # ?
    # ? Predicts the chord for a given measure using the pre-trained model.
    
    # ? Args:
    # ?    model: The loaded ChordPredictor model.
    # ?    measure_data: List of tuples (note_str, duration) representing the measure.
        
    # ? Returns:
    # ?   predicted_chord: The predicted chord symbol as a string.
    
    tokens = []
    octaves = []
    durations = []
    
    # Process each note in the measure
    for note_str, dur in measure_data:
        note_name, octave = parse_note(note_str)
        if note_name == "Rest":
            tokens.append(0)  # 0 reserved for padding or rest
            octaves.append(0)
        else:
            token = note2idx.get(note_name, 0)
            tokens.append(token)
            octaves.append(octave)
        durations.append(dur)
    
    # Pad sequences to max_seq_length
    pad_length = max_seq_length - len(tokens)
    if pad_length > 0:
        tokens.extend([0] * pad_length)
        octaves.extend([0] * pad_length)
        durations.extend([0.0] * pad_length)
    else:
        tokens = tokens[:max_seq_length]
        octaves = octaves[:max_seq_length]
        durations = durations[:max_seq_length]
    
    # Convert lists to tensors (batch size = 1)
    tokens_tensor = torch.tensor([tokens], dtype=torch.long)
    octaves_tensor = torch.tensor([octaves], dtype=torch.long)
    durations_tensor = torch.tensor([durations], dtype=torch.float)
    
    with torch.no_grad():
        chord_logits, chord_duration = model(tokens_tensor, octaves_tensor, durations_tensor)
        # Choose the chord with the highest logit score
        predicted_chord_idx = torch.argmax(chord_logits, dim=-1).item()
        predicted_chord = idx2chord[predicted_chord_idx]
        predicted_duration = chord_duration.item()  # can be used if needed
    
    return predicted_chord

def melody_extractor(score):
    """
    Extracts the melody in the treble clef from the given MusicXML score,
    discarding non-musical elements (e.g., Clef or layout objects).

    Only note.Note, chord.Chord, and note.Rest elements are retained if they
    occur in a treble clef context. Measure attributes like TimeSignature and
    KeySignature are also copied.

    Args:
        score (music21.stream.Score): The input score.

    Returns:
        music21.stream.Score: A new score containing only the melody from the treble clef.
    """
    from music21 import stream, clef, note, chord

    # Create a new score and part for the melody
    melody_score = stream.Score()
    melody_part = stream.Part()
    # Insert a treble clef at the beginning of the part
    melody_part.append(clef.TrebleClef())

    # Assume the melody is in the first part of the score
    original_part = score.parts[0]

    # Iterate over each measure in the original part
    for measure in original_part.getElementsByClass('Measure'):
        new_measure = measure.__class__()  # create an empty measure
        for element in measure:
            # Skip any clef objects (and similar non-musical elements)
            if isinstance(element, clef.Clef):
                continue

            # Process only Note, Chord, and Rest objects
            if isinstance(element, (note.Note, chord.Chord, note.Rest)):
                # Determine the clef context; if it exists and is a TrebleClef, keep the element.
                current_clef = element.getContextByClass('Clef')
                if current_clef and isinstance(current_clef, clef.TrebleClef):
                    new_measure.append(element)
            # Also copy measure attributes like TimeSignature and KeySignature
            elif hasattr(element, 'classes') and element.classes and element.classes[0] in ['TimeSignature', 'KeySignature']:
                new_measure.insert(element.offset, element)
            # Otherwise, skip the element (e.g., SystemLayout, etc.)
        melody_part.append(new_measure)
    melody_score.append(melody_part)
    return melody_score
