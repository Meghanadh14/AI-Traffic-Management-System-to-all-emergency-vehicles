import librosa
import numpy as np
import os

# Function to detect siren sound
def detect_siren(audio_path, threshold=0.5):
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=22050)

        # Extract MFCC (Mel-Frequency Cepstral Coefficients) features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Compute the mean MFCC to use as a feature
        mfcc_mean = np.mean(mfcc, axis=1)

        # Simple heuristic: if high-energy frequencies are dominant, assume it's a siren
        siren_score = np.mean(mfcc_mean[1:5])  

        # Normalize score between 0 and 1
        normalized_score = (siren_score - np.min(mfcc_mean)) / (np.max(mfcc_mean) - np.min(mfcc_mean))

        if normalized_score > threshold:
            return {"siren_detected": True, "confidence": normalized_score}
        else:
            return {"siren_detected": False, "confidence": normalized_score}
    except Exception as e:
        return {"error": str(e)}

# Example Usage
if __name__ == "__main__":
    audio_file = "/Users/meghanadhkottana/Documents/pythonProjects/siren.mp3"
    if os.path.exists(audio_file):
        result = detect_siren(audio_file)
        print(result)
    else:
        print("Audio file not found.")
