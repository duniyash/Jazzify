"""
Chord Prediction Model
---------------------
Functions for loading and using the transformer-based chord prediction model.
"""

import os
import logging
import torch
import torch.nn as nn
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

# Define the ChordPredictor model class (must match the architecture in the saved file)
class ChordPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes, max_seq_length, num_octaves=10):
        """
        Initialize the ChordPredictor model.
        
        Parameters:
        -----------
        vocab_size : int
            Size of the note vocabulary (including padding token)
        embed_dim : int
            Dimension of the embedding vectors
        num_heads : int
            Number of attention heads in the transformer
        hidden_dim : int
            Dimension of the feed-forward network model
        num_layers : int
            Number of transformer encoder layers
        num_classes : int
            Number of chord classes to predict
        max_seq_length : int
            Maximum sequence length
        num_octaves : int
            Number of possible octave values
        """
        super(ChordPredictor, self).__init__()
        # Embedding for note tokens
        self.note_embed = nn.Embedding(vocab_size, embed_dim)
        # Embedding for octave values
        self.octave_embed = nn.Embedding(num_octaves, embed_dim)
        # Linear projection for note durations
        self.duration_linear = nn.Linear(1, embed_dim)

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Additional fully connected layers after transformer pooling
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Two output heads:
        # - For chord classification
        self.fc_class = nn.Linear(hidden_dim // 2, num_classes)
        # - For chord duration regression
        self.fc_duration = nn.Linear(hidden_dim // 2, 1)

    def forward(self, tokens, octaves, note_durations):
        """
        Forward pass of the model.
        
        Parameters:
        -----------
        tokens : torch.Tensor
            Tensor of note tokens [batch_size, seq_length]
        octaves : torch.Tensor
            Tensor of octave values [batch_size, seq_length]
        note_durations : torch.Tensor
            Tensor of note durations [batch_size, seq_length]
            
        Returns:
        --------
        tuple
            (chord_logits, chord_duration)
        """
        token_emb = self.note_embed(tokens)           # [B, L, embed_dim]
        octave_emb = self.octave_embed(octaves)         # [B, L, embed_dim]
        duration_emb = self.duration_linear(note_durations.unsqueeze(-1))  # [B, L, embed_dim]

        # Sum embeddings and add positional encoding
        x = token_emb + octave_emb + duration_emb + self.pos_embedding  # [B, L, embed_dim]

        # Transformer expects input of shape [L, B, embed_dim]
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)

        # Use the first token's output as a pooled representation
        pooled = x[0]  # [B, embed_dim]

        # Additional layers for further processing
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Final output heads
        chord_logits = self.fc_class(x)              # [B, num_classes]
        chord_duration = self.fc_duration(x).squeeze(-1)  # [B]
        return chord_logits, chord_duration


# Global model instance and mappings to avoid reloading
_model = None
_note2idx = None
_idx2chord = None

def load_model(model_path='model/chord_predictor.safetensors'):
    """
    Load the chord prediction model from a safetensors file.
    
    Parameters:
    -----------
    model_path : str
        Path to the safetensors model file
        
    Returns:
    --------
    tuple
        (model, note2idx, idx2chord)
    """
    global _model, _note2idx, _idx2chord
    
    if _model is not None:
        return _model, _note2idx, _idx2chord
    
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Define the note and chord mappings
        _note2idx = {
            'A': 1, 'A#': 2, 'A-': 3, 'B': 4, 'B#': 5, 'B-': 6, 'B--': 7, 
            'C': 8, 'C#': 9, 'C-': 10, 'D': 11, 'D#': 12, 'D-': 13, 
            'E': 14, 'E#': 15, 'E-': 16, 'E--': 17, 'F': 18, 'F#': 19, 'F-': 20, 
            'G': 21, 'G#': 22, 'G-': 23, 'G--': 24, 'G---': 25
        }
        
        # This is a simplified chord mapping for example purposes
        # In a real application, you would load the complete mapping from a file
        _idx2chord = {
            0: 'A', 34: 'A7', 43: 'Am', 45: 'Am7', 53: 'B', 91: 'B7', 100: 'Bm', 105: 'Bm7',
            116: 'C', 138: 'C7', 148: 'Cm', 153: 'Cm7', 164: 'D', 198: 'D7', 207: 'Dm', 212: 'Dm7',
            221: 'E', 256: 'E7', 266: 'Em', 269: 'Em7', 278: 'F', 306: 'F7', 313: 'Fm', 317: 'Fm7',
            328: 'G', 354: 'G7', 364: 'Gm', 372: 'Gm7'
        }
        
        # Model hyperparameters (should match the training parameters)
        vocab_size = len(_note2idx) + 1  # +1 for padding
        embed_dim = 32
        num_heads = 4
        hidden_dim = 192
        num_layers = 4
        num_classes = 384  # Number of chord classes
        max_seq_length = 32
        num_octaves = 10
        
        # Initialize the model
        _model = ChordPredictor(
            vocab_size, embed_dim, num_heads, hidden_dim, num_layers,
            num_classes, max_seq_length, num_octaves
        )
        
        # Load the model weights
        state_dict = load_file(model_path)
        _model.load_state_dict(state_dict)
        _model.eval()
        
        logger.info("Model loaded successfully")
        return _model, _note2idx, _idx2chord
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

def predict_chord(notes, octaves, durations):
    """
    Predict a chord based on the input notes, octaves, and durations.
    
    Parameters:
    -----------
    notes : list
        List of note names (e.g., ['C', 'E', 'G'])
    octaves : list
        List of octave values for each note
    durations : list
        List of durations for each note
        
    Returns:
    --------
    tuple
        (predicted_chord, predicted_duration)
    """
    model, note2idx, idx2chord = load_model()
    
    # Validate input lengths
    if not (len(notes) == len(octaves) == len(durations)):
        raise ValueError("Notes, octaves, and durations must have the same length")
    
    # Handle empty input
    if len(notes) == 0:
        return "N.C.", 4.0  # No Chord, default duration
    
    # Encode notes
    encoded_notes = []
    for note in notes:
        if note in note2idx:
            encoded_notes.append(note2idx[note])
        else:
            logger.warning(f"Unknown note: {note}, skipping")
    
    # Prepare model input
    max_seq_length = 32
    
    # Pad sequences to max_seq_length
    pad_length = max_seq_length - len(encoded_notes)
    tokens_padded = encoded_notes + [0] * pad_length if pad_length > 0 else encoded_notes[:max_seq_length]
    octaves_padded = octaves + [0] * pad_length if pad_length > 0 else octaves[:max_seq_length]
    durations_padded = durations + [0.0] * pad_length if pad_length > 0 else durations[:max_seq_length]
    
    # Convert to tensors (batch size = 1)
    tokens_tensor = torch.tensor([tokens_padded], dtype=torch.long)
    octaves_tensor = torch.tensor([octaves_padded], dtype=torch.long)
    durations_tensor = torch.tensor([durations_padded], dtype=torch.float)
    
    # Make prediction
    with torch.no_grad():
        chord_logits, chord_duration = model(tokens_tensor, octaves_tensor, durations_tensor)
        # Get predicted chord index
        predicted_chord_idx = torch.argmax(chord_logits, dim=-1).item()
        predicted_duration = chord_duration.item()
    
    # Convert prediction to chord name
    if predicted_chord_idx in idx2chord:
        chord_name = idx2chord[predicted_chord_idx]
    else:
        # Default to C major if chord index not found
        logger.warning(f"Chord index {predicted_chord_idx} not found in mapping, defaulting to C")
        chord_name = "C"
    
    return chord_name, predicted_duration