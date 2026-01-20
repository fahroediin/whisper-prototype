import torchaudio

# Perbaikan untuk error torchaudio versi terbaru
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []
if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: None


import torch
from speechbrain.pretrained import SpeakerRecognition
from faster_whisper import WhisperModel
import numpy as np
import os

class MeetingEngine:
    def __init__(self):
        # Load model identitas suara
        self.spk_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="pretrained_models/spkrec"
        )
        # Load model transkripsi (ukuran tiny agar cepat untuk proto)
        self.stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        self.enrolled_users = {} # Simpan {nama: embedding}

    def enroll_user(self, name, audio_path):
        emb = self.spk_model.encode_batch(self.spk_model.load_audio(audio_path))
        self.enrolled_users[name] = emb[0][0]
        return f"User {name} enrolled successfully."

    def identify_speaker(self, audio_segment_path):
        # Ambil embedding dari potongan suara rapat
        test_emb = self.spk_model.encode_batch(self.spk_model.load_audio(audio_segment_path))[0][0]
        
        best_name = "Unknown"
        max_score = 0
        
        for name, saved_emb in self.enrolled_users.items():
            # Hitung skor kemiripan (Cosine Similarity)
            score = torch.nn.functional.cosine_similarity(test_emb, saved_emb, dim=0)
            if score > max_score and score > 0.25: # Threshold 0.25 - 0.35 biasanya ok
                max_score = score
                best_name = name
        
        return best_name

    def process_meeting(self, audio_path):
        # Transmisi dengan word-level timestamps untuk diarization sederhana
        segments, _ = self.stt_model.transcribe(audio_path, beam_size=5)
        
        results = []
        for segment in segments:
            # Di prototipe ini, kita identifikasi per baris kalimat
            # Dalam sistem nyata, kita potong audio berdasarkan segment.start & segment.end
            speaker = self.identify_speaker(audio_path) 
            results.append({
                "speaker": speaker,
                "text": segment.text.strip()
            })
        return results

engine = MeetingEngine()