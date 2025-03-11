from mido import MidiFile, Message
import numpy as np
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from preprocess import MidiPreprocessor
from feature_extraction import MidiFeatureExtractor
from train import MidiDataset, MusicTransformer, Trainer

def test_train():
    model = MusicTransformer(input_dim=128, output_dim=128)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    preprocessor = MidiPreprocessor()
    midi_directory = r"C:\Users\henry-cao-local\Desktop\Personal_Projects\Music_Generation\data"
    midi_processed = r"C:\Users\henry-cao-local\Desktop\Personal_Projects\Music_Generation\music_generation_midi\processed_midi"

    extractor = MidiFeatureExtractor()

    for root, dirs, files, in os.walk(midi_directory):
        for file in files:
            if file.endswith(".mid"):
                midi_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, midi_directory)
                output_dir = os.path.join(midi_processed, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                # Load and parse MIDI file
                midi = preprocessor.load_midi(midi_path)
                notes, velocities, timings = preprocessor.parse_midi(midi)
                quantized_timings = preprocessor.quantize_timings(timings)

                # Save processed MIDI file
                output_midi_path = os.path.join(output_dir, file)
                preprocessor.save_midi(notes, velocities, quantized_timings, output_midi_path)

                # Extract features
                piano_roll = extractor.extract_piano_roll(notes, quantized_timings)
                durations = extractor.extract_note_durations(notes, quantized_timings)
                chords = extractor.extract_chord_sequences(notes)

                # Save features as numpy files
                np.save(os.path.join(output_dir, file.replace(".mid", "_piano_roll.npy")), piano_roll)
                np.save(os.path.join(output_dir, file.replace(".mid", "_durations.npy")), durations)
                np.save(os.path.join(output_dir, file.replace(".mid", "_chords.npy")), chords)

                print(f"Processed and saved: {midi_path}")



if __name__ == "__main__":
    test_train()