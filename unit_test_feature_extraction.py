from mido import MidiFile, Message
import numpy as np
import os
from preprocess import MidiPreprocessor
from feature_extraction import MidiFeatureExtractor

def test_feature_extraction():
    preprocessor = MidiPreprocessor()
    midi_directory = r"C:\Users\henry-cao-local\Desktop\Personal_Projects\Music_Generation\data"
    midi_test_file = r"C:\Users\henry-cao-local\Desktop\Personal_Projects\Music_Generation\data\albeniz\alb_esp1.mid"
    midi = preprocessor.load_midi(midi_test_file)
    assert isinstance(midi, MidiFile)
    parsed_midi = preprocessor.parse_midi(midi)
    assert len(parsed_midi) == 3
    quantized_timings = preprocessor.quantize_timings(parsed_midi[2])
    assert len(quantized_timings) == len(parsed_midi[2])
    save_directory = r"C:\Users\henry-cao-local\Desktop\Personal_Projects\Music_Generation\music_generation_midi\processed_midi"
    preprocessor.save_midi(parsed_midi[0], parsed_midi[1], quantized_timings, save_directory + r"\test.mid")
    assert os.path.exists(os.path.join(save_directory, "test.mid"))
    FeatureExtractor = MidiFeatureExtractor()
    piano_roll = FeatureExtractor.extract_piano_roll(parsed_midi[0], quantized_timings)
    assert piano_roll.shape[0] == 100
    durations = FeatureExtractor.extract_note_durations(parsed_midi[0], quantized_timings)
    assert len(durations) == len(parsed_midi[0]) - 1
    chords = FeatureExtractor.extract_chort_sequences(parsed_midi[0])
    assert chords.shape[1] == 4

if __name__ == "__main__":
    test_feature_extraction()