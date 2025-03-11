from mido import MidiFile, Message
import numpy as np
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from preprocess import MidiPreprocessor
from feature_extraction import MidiFeatureExtractor
from train import MidiDataset, MusicTransformer, Trainer

def test_train():
    # Make sure these parameters match the MusicTransformer definition in train.py
    model = MusicTransformer(input_dim=128, output_dim=428)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Data loading code remains the same...
    preprocessor = MidiPreprocessor()
    midi_directory = r"C:\Users\henry-cao-local\Desktop\Personal_Projects\Music_Generation\data"
    midi_processed = r"C:\Users\henry-cao-local\Desktop\Personal_Projects\Music_Generation\music_generation_midi\processed_midi"

    extractor = MidiFeatureExtractor()
    
    # Data loading section remains unchanged
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
        print(f"Dimensions of piano rolls: {all_piano_rolls[0].shape}")
        print(f"Dimensions of durations: {all_durations[0].shape if all_durations else None}")
        print(f"Dimensions of chords: {all_chords[0].shape if all_chords else None}")

        # Create dataset - make sure this matches MidiDataset's constructor parameters
        dataset = MidiDataset(all_piano_rolls, all_durations, all_chords)
        
        # Set up training parameters
        batch_size = 16
        learning_rate = 1e-4
        num_epochs = 5
        
        # Initialize trainer - make sure parameters match Trainer's constructor
        trainer = Trainer(
            dataset=dataset,
            model=model,
            epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device
        )
        
        # Train the model - ensure train method accepts num_epochs
        trainer.train()

        # # Save the trained model
        # # Create models directory if it doesn't exist
        # models_dir = os.path.join(r"C:\Users\henry-cao-local\Desktop\Personal_Projects\Music_Generation\music_generation_midi\models")
        # os.makedirs(models_dir, exist_ok=True)
        
        # # Find the highest existing version number
        # version = 1
        # for file in os.listdir(models_dir):
        #     if file.startswith("music_transformer_v") and file.endswith(".pth"):
        #         try:
        #             v = int(file.split("_v")[1].split(".")[0])
        #             version = max(version, v + 1)
        #         except:
        #             pass
                
        # # Create a unique model path with version number
        # model_path = os.path.join(models_dir, f"music_transformer_v{version}.pth")
        # print(f"Saving model to {model_path}")
        # os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # torch.save(model.state_dict(), model_path)
    else:
        print("No data files found. Please check the processed_midi directory.")


if __name__ == "__main__":
    test_train()
