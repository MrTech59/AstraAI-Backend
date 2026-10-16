import queue
import threading
import numpy as np
from typing import Any

from .runtime import AstraRuntimeError, _normalize_device_index

# Silence configuration
SILENCE_THRESHOLD_RMS = 0.01  # Audio energy threshold
SILENCE_DURATION_SEC = 1.0    # Seconds of silence to trigger transcription
MIN_SPEECH_DURATION_SEC = 0.5 # Minimum seconds of speech to transcribe

class MicListener:
    def __init__(
        self,
        stt_model_size: str,
        input_device_index: Any,
        text_queue: queue.Queue,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise AstraRuntimeError(
                "Missing dependency `faster-whisper`. Run `python -m pip install -r requirements.txt`."
            ) from exc
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AstraRuntimeError(
                "Missing dependency `sounddevice`. Run `python -m pip install -r requirements.txt`."
            ) from exc

        self._sd = sd
        self._text_queue = text_queue
        self._sample_rate = 16000
        self._channels = 1

        print(f"Loading STT model: {stt_model_size}...", flush=True)
        self._model = WhisperModel(stt_model_size, device="auto", compute_type="default")
        
        device_index = _normalize_device_index(input_device_index)
        
        self._audio_queue: queue.Queue = queue.Queue()
        self._running = True

        self._stream = self._sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            device=device_index,
            callback=self._audio_callback
        )
        
        self._thread = threading.Thread(target=self._process_audio, daemon=True, name="astra-mic-listener")

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"\n[Audio Input Status] {status}")
        self._audio_queue.put(indata.copy())

    def start(self):
        self._stream.start()
        self._thread.start()

    def stop(self):
        self._running = False
        self._stream.stop()
        self._stream.close()
        self._audio_queue.put(None)
        self._thread.join(timeout=2)

    def _process_audio(self):
        audio_buffer = []
        silence_frames = 0
        speech_frames = 0
        is_speaking = False
        
        # Frames per callback is usually based on default blocksize (often ~1024 frames)
        frames_per_sec = self._sample_rate
        
        while self._running:
            chunk = self._audio_queue.get()
            if chunk is None:
                break
                
            rms = np.sqrt(np.mean(chunk**2))
            
            if rms > SILENCE_THRESHOLD_RMS:
                is_speaking = True
                silence_frames = 0
                speech_frames += len(chunk)
                audio_buffer.append(chunk)
            elif is_speaking:
                silence_frames += len(chunk)
                audio_buffer.append(chunk)
                
                if silence_frames > SILENCE_DURATION_SEC * frames_per_sec:
                    # Speech stopped, time to transcribe
                    if speech_frames > MIN_SPEECH_DURATION_SEC * frames_per_sec:
                        audio_data = np.concatenate(audio_buffer).flatten()
                        self._transcribe(audio_data)
                    
                    # Reset state
                    is_speaking = False
                    silence_frames = 0
                    speech_frames = 0
                    audio_buffer = []
            else:
                # Not speaking, keep a tiny rolling buffer for smooth onset
                audio_buffer = [chunk]

    def _transcribe(self, audio_data: np.ndarray):
        segments, info = self._model.transcribe(
            audio_data, 
            beam_size=5, 
            language="en", 
            condition_on_previous_text=False
        )
        
        text = " ".join([segment.text for segment in segments]).strip()
        if text:
            # Send to main thread
            self._text_queue.put(("mic", text))
