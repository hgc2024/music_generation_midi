from mido import MidiFile, Message
import numpy as np

class MidiPreprocessor:
    def __init__(self, ticks_per_beat=480):
        self.ticks_per_beat = ticks_per_beat
    
    def load_midi(self, file_path):
        """
        Load a MIDI file uing mido.
        """
        midi = MidiFile(file_path)
        return midi

    def parse_midi(self, midi):
        """
        Extract notes, velocities, and timings from a MIDI file.
        """
        notes = []
        velocities = []
        timings = []

        for track in midi.tracks:
            for msg in track:
                if msg.type == 'note_on':
                    notes.append(msg.note)
                    velocities.append(msg.velocity)
                    timings.append(msg.time)
        
        return np.array(notes), np.array(velocities), np.array(timings)
    
    def quantize_timings(self, timings, resolution=16):
        """
        Quantize timings to a fixed resolution (e.g., 16th notes)
        """
        quantized = np.round(timings / (self.ticks_per_beat / resolution))
        return quantized.astype(int)
    
    def save_midi(self, notes, velocities, timings, output_path):
        """
        Save processed MIDI data to a new MIDI file.
        """
        midi = MidiFile()
        track = midi.add_track()

        for note, velocity, time in zip(notes, velocities, timings):
            track.append(Message('note_on', note=note, velocity=velocity, time=time))
        
        midi.save(output_path)