import numpy as np
import random
from scipy.io import wavfile


def apply_envelope(signal, sample_rate, attack_time=0.01, release_time=0.01):
    """
    Apply attack and release envelope to avoid clicks.

    Parameters:
    signal: Input audio signal
    sample_rate: Sample rate in Hz
    attack_time: Attack time in seconds
    release_time: Release time in seconds
    """
    attack_samples = int(attack_time * sample_rate)
    release_samples = int(release_time * sample_rate)

    attack = np.linspace(0, 1, attack_samples)
    release = np.linspace(1, 0, release_samples)

    signal[:attack_samples] *= attack
    signal[-release_samples:] *= release

    return signal


def create_shepard_tone(base_freq, duration=1.0, sample_rate=44100):
    """
    Create a Shepard tone at a given base frequency.

    Parameters:
    base_freq: Base frequency in Hz
    duration: Duration of the tone in seconds
    sample_rate: Sample rate in Hz
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.zeros_like(t)

    # Create 5 octaves of the tone
    for octave in range(5):
        freq = base_freq * (2**octave)
        # Gaussian envelope centered at 3rd octave
        envelope = np.exp(-0.5 * ((octave - 3) / 2) ** 2)
        signal += envelope * np.sin(2 * np.pi * freq * t)

    # Normalize before applying envelope
    signal = signal / np.max(np.abs(signal))

    # Apply attack and release envelope
    signal = apply_envelope(signal, sample_rate)

    return signal


def create_tritone_pair(base_freq, duration=1.0, sample_rate=44100):
    """
    Create a pair of Shepard tones separated by a tritone.

    Parameters:
    base_freq: Base frequency of first tone in Hz
    duration: Duration of each tone in seconds
    sample_rate: Sample rate in Hz
    """
    # Create first tone
    tone1 = create_shepard_tone(base_freq, duration, sample_rate)

    # Create second tone (tritone = frequency * 2^(6/12))
    tritone_freq = base_freq * (2 ** (6 / 12))
    tone2 = create_shepard_tone(tritone_freq, duration, sample_rate)

    return tone1, tone2


def save_tritone_pair(filename, base_freq=440, duration=1.0, sample_rate=44100):
    """
    Generate and save a tritone pair to a WAV file.
    """
    tone1, tone2 = create_tritone_pair(base_freq, duration, sample_rate)

    # Combine tones
    combined = np.concatenate([tone1, tone2])

    # Convert to 16-bit integers
    audio_data = np.int16(combined * 32767)

    # Save to WAV file
    wavfile.write(filename, sample_rate, audio_data)


def save_all_tritone_pairs(
    filename, base_freq_map, duration=1.0, sample_rate=44100, gap_duration=1.0
):
    """
    Generate and save tritone pairs for all notes in the base frequency map.
    """
    combined = []
    gap = np.zeros(int(gap_duration * sample_rate))

    items = list(base_freq_map.items())
    random.shuffle(items)

    note_filename = filename.replace(".wav", ".txt")
    note_file = open(note_filename, "w")

    for note, freq in items:
        tone1, tone2 = create_tritone_pair(freq, duration, sample_rate)
        combined = np.concatenate([combined, gap, tone1, tone2])
        note_file.write(f"{note}\n")
    # Convert to 16-bit integers
    audio_data = np.int16(combined * 32767)
    wavfile.write(filename, sample_rate, audio_data)


# Frequency map for notes
base_freq_map = {
    "C": 65.41,
    "C#": 69.30,
    "D": 73.42,
    "D#": 77.78,
    "E": 82.41,
    "F": 87.31,
    "F#": 92.50,
    "G": 98.00,
    "G#": 103.83,
    "A": 110.00,
    "A#": 116.54,
    "B": 123.47,
}

save_all_tritone_pairs(
    "tritone_paradox_combined.wav", base_freq_map, duration=0.5, gap_duration=5.0
)
