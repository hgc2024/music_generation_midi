from mido import MidiFile, Message
import numpy as np
import os
from preprocess import MidiPreprocessor

def test_load_midi():
    preprocessor = MidiPreprocessor()
    midi_directory = r"C:\Users\henry-cao-local\Desktop\Personal_Projects\Music_Generation\data"
    midi_test_file = r"C:\Users\henry-cao-local\Desktop\Personal_Projects\Music_Generation\data\albeniz\alb_esp1.mid"
    midi = preprocessor.load_midi(midi_test_file)
    assert isinstance(midi, MidiFile)
    parsed_midi = preprocessor.parse_midi(midi)
    print(parsed_midi)
    assert len(parsed_midi) == 3
    quantized_timings = preprocessor.quantize_timings(parsed_midi[2])
    assert len(quantized_timings) == len(parsed_midi[2])
    print(quantized_timings)
    save_directory = r"C:\Users\henry-cao-local\Desktop\Personal_Projects\Music_Generation\music_generation_midi\processed_midi"
    preprocessor.save_midi(parsed_midi[0], parsed_midi[1], quantized_timings, save_directory + r"\test.mid")
    assert os.path.exists(os.path.join(save_directory, "test.mid"))

if __name__ == "__main__":
    test_load_midi()