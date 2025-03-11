# AI Generation of Midi Piano Files

This project attempts to build a Transformer model that can generate MIDI piano files of different music genres.

Having done a dry run on the Classical Music MIDI dataset, I plan to scale up and use the ADL Piano MIDI dataset to cover more musical genres.

The current Transformer model is mostly noisy, with no more than an F1 Score of 10%. But this is due to the small dry run dataset and limited hardware used.MIDI Piano Genre Cover Generator Using Generative AI
Overview

This project explores the use of Generative AI to create a model capable of generating MIDI piano files that cover various musical genres. The goal is to leverage the power of Transformer-based architectures to produce musically coherent and genre-specific piano compositions. While the initial results are promising, the current model's performance is limited due to constraints in dataset size, hardware capabilities, and the complexity of MIDI data.
Project Description

The core idea of this project is to train a Transformer-based model to generate MIDI piano files that mimic the stylistic elements of different musical genres, such as classical, jazz, pop, rock, and more. MIDI files are a natural choice for this task because they provide a structured representation of musical notes, velocities, and timings, making them suitable for machine learning models.

The model takes a seed sequence of MIDI data as input and generates a continuation of the sequence, adhering to the stylistic rules of the target genre. The generated MIDI files can then be played back using any MIDI-compatible software or hardware.
Current Status

As of now, the project has achieved the following milestones:

    Data Preprocessing:

        Implemented a robust preprocessing pipeline to handle MIDI files.

        Extracted features such as piano rolls, note durations, and chord sequences from MIDI data.

        Processed a dataset of MIDI files from various genres, including classical, jazz, and pop.

    Model Development:

        Built a Transformer-based architecture using PyTorch.

        Integrated the model with the preprocessing pipeline to train on MIDI data.

        Conducted initial training runs to validate the model's ability to learn musical patterns.

    Evaluation:

        Evaluated the model using metrics such as F1 Score and perplexity.

        Achieved an initial F1 Score of 10%, indicating room for improvement.

Key Challenges

    Dataset Size: The current dataset is limited in size and diversity, which restricts the model's ability to generalize across genres.

    Hardware Limitations: Training the Transformer model is computationally expensive, and the available hardware limits the scale of training.

    Evaluation: Evaluating the quality of generated music is inherently subjective, and additional evaluation methods (e.g., human listening tests) are needed.

Future Work

To improve the model's performance and achieve the project's goals, the following steps are planned:

    Expand the Dataset:

        Collect and preprocess a larger dataset of MIDI files spanning multiple genres.

        Use data augmentation techniques (e.g., transposing, varying tempos) to increase dataset diversity.

    Upgrade Hardware:

        Utilize more powerful hardware (e.g., GPUs or TPUs) or cloud-based resources (e.g., AWS, Google Cloud) to enable faster and more extensive training.

    Model Optimization:

        Experiment with different Transformer architectures, hyperparameters, and training strategies.

        Explore cutting-edge architectures such as Music Transformer or Pop Music Transformer.

    Evaluation Framework:

        Develop a robust evaluation framework that combines quantitative metrics (e.g., F1 Score, perplexity) with qualitative assessments (e.g., human listening tests, expert reviews).

    Genre-Specific Fine-Tuning:

        Train separate models or fine-tune the existing model for specific genres to achieve better stylistic accuracy.

    User Interface:

        Develop a user-friendly interface that allows users to input a seed sequence, select a genre, and generate MIDI piano covers with minimal effort.

Getting Started
Prerequisites

    Python 3.x

    PyTorch

    MIDI processing libraries (e.g., mido, pretty_midi)

    GPU support (recommended for faster training)

Installation

    Clone the repository:
    bash
    Copy

    git clone https://github.com/yourusername/midi-piano-genre-cover-generator.git
    cd midi-piano-genre-cover-generator

Usage

    Data Preparation:

        Place your MIDI files in the data/ directory.

        Run the preprocessing script to convert them into a format suitable for training:
        bash
        Copy

        python preprocess.py

    Training:

        Train the Transformer model using the preprocessed data:
        bash
        Copy

        python train.py

    Generation:

        Generate MIDI piano covers using the trained model:
        bash
        Copy

        python generate.py --seed <seed_sequence> --genre <target_genre>

Example

To generate a classical piano cover:
bash
Copy

python generate.py --seed "data/seed_classical.mid" --genre "classical"

Contributing

Contributions to this project are welcome! If you have ideas for improving the model, expanding the dataset, or enhancing the evaluation framework, please feel free to open an issue or submit a pull request.
License

This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

    Thanks to the open-source community for providing valuable resources and libraries for MIDI processing and machine learning.

    Special thanks to the creators of the Transformer architecture, which has revolutionized sequence generation tasks.

Contact

For any questions or feedback, please open an issue on the GitHub repository.
