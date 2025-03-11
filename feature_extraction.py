import numpy as np

class MidiFeatureExtractor:
    def __init__(self, num_notes=128, resolution=16):
        self.num_notes = num_notes
        self.resolution = resolution
    
    def extract_piano_roll(self, notes, timings, sequence_length=100):
        """
        Convert MIDI notes and timings into a piano roll representation
        """
        piano_roll = np.zeros((sequence_length, self.num_notes))

        current_time = 0
        for i in range(sequence_length):
            if i < len(notes):
                note = notes[i]
                piano_roll[i, note] = 1 # Mark the note as active
                current_time += timings[i]
            else:
                break
        
        return piano_roll
    
    def extract_note_durations(self, notes, timings):
        """
        Calculate the duration of each note.
        """
        durations = []
        for i in range(len(notes) - 1):
            durations.append(timings[i + 1] - timings[i])
        return np.array(durations)
    
    def extract_chord_sequences(self, notes, window_size=4):
        """
        Extract sequences of chords from the MIDI data.
        """
        chords = []
        for i in range(len(notes) - window_size + 1):
            chord = notes[i:i + window_size]
            chords.append(chord)
        return np.array(chords)