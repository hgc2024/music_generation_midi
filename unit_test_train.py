from mido import MidiFile, Message
import numpy as np
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from preprocess import MidiPreprocessor
from feature_extraction import MidiFeatureExtractor
from train import MidiDataset, MusicRNN, Trainer

def test_train():
    # Initialize the model - adjust parameters based on actual MusicRNN implementation
    input_dim = 128
    output_dim = 128 + 300  # Notes + duration values
    hidden_dim = 256
    
    model = MusicRNN(
        input_dim=input_dim, 
        output_dim=output_dim,
        hidden_dim=hidden_dim,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Data preparation
    midi_processed = r"C:\Users\henry-cao-local\Desktop\Personal_Projects\Music_Generation\music_generation_midi\processed_midi"
    
    all_piano_rolls = []
    all_durations = []
    all_chords = []
    
    for root, dirs, files in os.walk(midi_processed):
        for file in files:
            if file.endswith("_piano_roll.npy"):
                piano_roll = np.load(os.path.join(root, file))
                all_piano_rolls.append(piano_roll)
                
                duration_file = file.replace("_piano_roll.npy", "_durations.npy")
                chord_file = file.replace("_piano_roll.npy", "_chords.npy")
                
                if os.path.exists(os.path.join(root, duration_file)):
                    durations = np.load(os.path.join(root, duration_file))
                    all_durations.append(durations)
                
                if os.path.exists(os.path.join(root, chord_file)):
                    chords = np.load(os.path.join(root, chord_file))
                    all_chords.append(chords)
    
    if all_piano_rolls:
        print(f"Loaded {len(all_piano_rolls)} sequences")
        print(f"Dimensions of piano rolls: {all_piano_rolls[0].shape}")
        print(f"Dimensions of durations: {all_durations[0].shape if all_durations else None}")
        print(f"Dimensions of chords: {all_chords[0].shape if all_chords else None}")

        # Create dataset with appropriate parameters
        sequence_length = 64  # Adjust based on actual requirements
        dataset = MidiDataset(
            features=all_piano_rolls,
            labels=all_durations,
            midi_files=all_chords
        )
        
        # Training parameters
        batch_size = 16
        learning_rate = 1e-4
        num_epochs = 50
        
        # Initialize trainer with appropriate parameters
        trainer = Trainer(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=num_epochs,
            device=device
        )
        
        # Run training
        trainer.train()
    else:
        print("No data files found. Please check the processed_midi directory.")


if __name__ == "__main__":
    test_train()
