from __future__ import annotations

import os
import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from .config import resolve_project_path, DEVICES_SCHEMA_PATH, DEVICES_SCHEMA_1_PATH
from .ha import HomeAssistantClient
from .vector_db import VectorDB

EXIT_COMMANDS = {"exit", "quit", ":q"}
SENTENCE_PATTERN = re.compile(r"(.+?[.!?]+[\"')\]]*)(?=\s+|$)", re.DOTALL)
MIN_SPEECH_CHUNK_CHARS = 32
MAX_SPEECH_CHUNK_CHARS = 72
_QUEUE_SENTINEL = object()


class AstraRuntimeError(RuntimeError):
    """Raised when the interactive runtime cannot start or continue."""


@dataclass(frozen=True)
class ChatOptions:
    prompt: str | None = None
    system_prompt: str | None = None
    temperature: float = 0.7
    # Reasoning models (nvidia_thinking) consume a large token budget
    # before emitting visible content, so use a generous default.
    max_tokens: int = 4096
    no_voice: bool = False


class SpeechChunkAccumulator:
    def __init__(
        self,
        min_chunk_chars: int = MIN_SPEECH_CHUNK_CHARS,
        max_chunk_chars: int = MAX_SPEECH_CHUNK_CHARS,
    ) -> None:
        self._buffer = ""
        self._min_chunk_chars = max(1, min_chunk_chars)
        self._max_chunk_chars = max(self._min_chunk_chars + 1, max_chunk_chars)

    def push(self, fragment: str) -> list[str]:
        self._buffer += fragment
        return self._collect_ready_chunks(allow_tail=False)

    def flush(self) -> list[str]:
        return self._collect_ready_chunks(allow_tail=True)

    def _collect_ready_chunks(self, allow_tail: bool) -> list[str]:
        ready: list[str] = []

        while True:
            chunk = self._pop_next_chunk(allow_tail=allow_tail)
            if chunk is None:
                return ready
            ready.append(chunk)

    def _pop_next_chunk(self, allow_tail: bool) -> str | None:
        if not self._buffer.strip():
            self._buffer = ""
            return None

        match = SENTENCE_PATTERN.match(self._buffer)
        if match:
            return self._consume(match.end())

        stripped_length = len(self._buffer.strip())
        if stripped_length < self._min_chunk_chars:
            if allow_tail:
                return self._consume(len(self._buffer))
            return None

        clause_break = _find_clause_break(
            self._buffer,
            min_chunk_chars=self._min_chunk_chars,
            max_chunk_chars=self._max_chunk_chars,
        )
        if clause_break is not None:
            return self._consume(clause_break)

        if stripped_length >= self._max_chunk_chars:
            word_break = _find_word_break(
                self._buffer,
                min_chunk_chars=self._min_chunk_chars,
                max_chunk_chars=self._max_chunk_chars,
            )
            if word_break is not None:
                return self._consume(word_break)

        if allow_tail:
            return self._consume(len(self._buffer))

        return None

    def _consume(self, end_index: int) -> str | None:
        chunk = self._buffer[:end_index].strip()
        self._buffer = self._buffer[end_index:].lstrip()
        return chunk or None


class PiperSpeaker:
    def __init__(self, voice_path: Path, output_device_index: Any) -> None:
        self._PiperVoice = _load_piper_voice_class()
        self._sounddevice = _load_sounddevice_module()
        self._voice = self._PiperVoice.load(str(voice_path))
        self._device = _resolve_output_device(self._sounddevice, output_device_index)
        self._sample_rate = int(self._voice.config.sample_rate)
        self._channels = 1
        self._stream = self._open_output_stream()
        self._queue: queue.Queue[object] = queue.Queue()
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="astra-piper-speaker",
            daemon=True,
        )
        self._thread.start()

    def speak_async(self, text: str) -> None:
        normalized = " ".join(text.split())
        if normalized:
            self._queue.put(normalized)

    def drain(self) -> None:
        self._queue.join()
        if self._error is not None:
            error = self._error
            self._error = None
            raise AstraRuntimeError(f"Piper playback failed: {error}") from error

    def close(self) -> None:
        self._queue.put(_QUEUE_SENTINEL)
        self._thread.join(timeout=10)
        self._stream.stop()
        self._stream.close()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _QUEUE_SENTINEL:
                    return

                self._speak_now(str(item))
            except Exception as exc:  # pragma: no cover - depends on local audio stack
                self._error = exc
            finally:
                self._queue.task_done()

    def _speak_now(self, text: str) -> None:
        for audio_chunk in self._voice.synthesize(text):
            if int(audio_chunk.sample_rate) != self._sample_rate:
                raise AstraRuntimeError(
                    "Piper returned an unexpected sample rate during playback."
                )

            audio = audio_chunk.audio_float_array
            channels = int(audio_chunk.sample_channels)
            if channels == 1:
                audio = audio.reshape(-1, 1)
            else:
                audio = audio.reshape(-1, channels)

            self._stream.write(audio)

    def _open_output_stream(self) -> Any:
        try:
            stream = self._sounddevice.OutputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                device=self._device,
                latency="low",
            )
            stream.start()
        except Exception as exc:
            raise AstraRuntimeError(f"Failed to open Piper output stream: {exc}") from exc

        return stream


def run_chat_session(settings: dict[str, Any], options: ChatOptions) -> int:
    system_name = str(settings.get("system_name", "Astra"))
    llm_type = settings.get("llm_type", "local")
    
    llm_model_path = None
    if llm_type == "local":
        llm_model_path = _require_project_file(settings.get("llm_model_path"), "llm model")
        
    voice_model_path = None
    if not options.no_voice:
        voice_model_path = _require_project_file(settings.get("tts_voice_path"), "voice model")
        
    llm = _load_llm(llm_model_path, settings)
    speaker = None
    if voice_model_path is not None:
        speaker = PiperSpeaker(voice_model_path, settings.get("output_device_index"))
    messages = [{"role": "system", "content": _build_system_prompt(settings, options.system_prompt)}]

    ha_client = None
    vector_db = None
    if settings.get("ha_url") and settings.get("ha_token"):
        ha_client = HomeAssistantClient(settings["ha_url"], settings["ha_token"])
        vector_db = VectorDB()
        try:
            with open(DEVICES_SCHEMA_PATH, "r", encoding="utf-8") as f:
                devices_schema = json.load(f)
            allowed_domains = set(devices_schema.get("domains", []))
            
            states = ha_client.get_states()
            devices_list = []
            for state in states:
                domain = state["entity_id"].split(".")[0]
                if domain in allowed_domains:
                    name = state.get("attributes", {}).get("friendly_name", state["entity_id"])
                    desc = f"[{state['entity_id']}] {name} - State: {state.get('state')}"
                    devices_list.append({"id": state["entity_id"], "desc": desc})
            
            if devices_list:
                vector_db.add_devices(devices_list)
        except Exception as e:
            print(f"Warning: Failed to sync Home Assistant devices: {e}")

    print(f"{system_name} launcher")
    if llm_type == "nvidia":
        print(f"LLM: Nvidia Online Model ({settings.get('nvidia_model', 'unknown')})")
    else:
        print(f"LLM: Local Model ({llm_model_path.name})")
        
    if options.no_voice:
        print("Voice: disabled")
    else:
        print(f"Voice: {voice_model_path.name} via Piper")
    if ha_client:
        print("Home Assistant: connected")

    try:
        if options.prompt:
            _run_turn(llm, speaker, messages, system_name, options.prompt, options, ha_client, vector_db)
            return 0

        command_queue = queue.Queue()
        wake_word = str(settings.get("wake_word", "")).strip().lower()
        stt_model_size = settings.get("stt_model_size")
        
        mic_listener = None
        if wake_word and stt_model_size:
            try:
                from .stt import MicListener
                mic_listener = MicListener(stt_model_size, settings.get("input_device_index"), command_queue)
                mic_listener.start()
            except Exception as e:
                print(f"Warning: Failed to start mic listener: {e}")

        def keyboard_listener():
            while True:
                try:
                    user_text = input("You: ").strip()
                    if user_text:
                        command_queue.put(("keyboard", user_text))
                except EOFError:
                    command_queue.put(("system", "exit"))
                    break
                except KeyboardInterrupt:
                    command_queue.put(("system", "exit"))
                    break

        threading.Thread(target=keyboard_listener, daemon=True, name="astra-keyboard").start()

        print("Type your message and press Enter. Use `exit` to quit.")
        if mic_listener:
            print(f"Listening for wake word: '{wake_word}'...")

        while True:
            source, user_text = command_queue.get()
            
            if source == "system" and user_text == "exit":
                return 0
                
            if source == "keyboard":
                if user_text.lower() in EXIT_COMMANDS:
                    return 0
            elif source == "mic":
                # Check for wake word
                wake_variants = [wake_word, f"hey {wake_word}"]
                text_lower = ''.join(c for c in user_text.lower() if c.isalnum() or c.isspace())
                
                matched = False
                for w in wake_variants:
                    if text_lower.startswith(w):
                        matched = True
                        break
                        
                if not matched:
                    continue # Ignore background chatter
                    
                # Strip wake word, keeping original casing
                user_text = re.sub(rf"^(hey\s+{wake_word}|{wake_word})[,\s]*", "", user_text, flags=re.IGNORECASE).strip()
                if not user_text:
                    user_text = "Yes?"
                    
                print(f"\n[Voice] {user_text}")

            _run_turn(llm, speaker, messages, system_name, user_text, options, ha_client, vector_db)
            
            if source == "mic":
                # Reprint the prompt cleanly after a voice interaction
                print("You: ", end="", flush=True)

    finally:
        if speaker is not None:
            speaker.close()
        if 'mic_listener' in locals() and mic_listener is not None:
            mic_listener.stop()


def _run_turn(
    llm: Any,
    speaker: PiperSpeaker | None,
    messages: list[dict[str, str]],
    system_name: str,
    user_text: str,
    options: ChatOptions,
    ha_client: HomeAssistantClient | None = None,
    vector_db: VectorDB | None = None,
) -> str:
    if ha_client and vector_db:
        search_results = vector_db.search(user_text, n_results=5)
        devices_context = []
        if search_results and "documents" in search_results and search_results["documents"]:
            for docs in search_results["documents"]:
                devices_context.extend(docs)
                
        if devices_context:
            context_str = "\n".join(devices_context)
            messages.append({"role": "system", "content": f"Available devices:\n{context_str}"})

    messages.append({"role": "user", "content": user_text})

    if ha_client and vector_db:
        with open(DEVICES_SCHEMA_1_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
            
        json_instruction = (
            "You are a smart home AI assistant. You MUST respond with ONLY a valid JSON object matching this schema:\n"
            f"{json.dumps(schema)}\n"
            "Do not include any markdown formatting, backticks, or other text. Only the raw JSON object."
        )
        messages.append({"role": "system", "content": json_instruction})
            
        print(f"{system_name}: ", end="", flush=True)
        try:
            response = llm.create_chat_completion(
                messages=messages,
                temperature=0.1,  # Lower temperature for JSON generation
                max_tokens=options.max_tokens,
                stream=False,
                response_format={"type": "json_object"}
            )
            
            if isinstance(response, dict):
                response_text = response["choices"][0]["message"].get("content")
            else:
                parts = []
                for chunk in response:
                    text = _extract_stream_text(chunk)
                    if text:
                        parts.append(text)
                response_text = "".join(parts)
                
            if not response_text:
                response_text = "{}"
                print("\n[HA Error] Model returned empty or None response.")

            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError:
                print(f"\n[HA Error] Failed to parse model output as JSON: {response_text}")
                parsed = {}
            
            if parsed.get("intent_type") == "control":
                for cmd in parsed.get("commands", []):
                    try:
                        ha_client.call_service(cmd["domain"], cmd["service"], cmd["entity_id"], cmd.get("data"))
                        print(f"\n[HA] Executed {cmd['domain']}.{cmd['service']} on {cmd['entity_id']}")
                    except Exception as e:
                        print(f"\n[HA Error] Failed to execute {cmd}: {e}")
            
            assistant_response = parsed.get("assistant_response", "")
            if not assistant_response:
                assistant_response = "I have processed your request."
                
            print(assistant_response)
            if speaker is not None:
                speaker.speak_async(assistant_response)
                speaker.drain()
                
            # Remove json_instruction
            messages.pop(-1)
            
            if devices_context:
                # Remove the injected devices list (it is now at index -2)
                messages.pop(-2)
                
            messages.append({"role": "assistant", "content": assistant_response})
            return assistant_response

        except Exception as exc:
            raise AstraRuntimeError(f"Failed during HA interaction: {exc}") from exc

    # Fallback to standard streaming if HA is not configured
    response_parts: list[str] = []
    accumulator = SpeechChunkAccumulator()

    print(f"{system_name}: ", end="", flush=True)
    try:
        stream = llm.create_chat_completion(
            messages=messages,
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            stream=True,
        )
    except Exception as exc:
        raise AstraRuntimeError(f"Failed to start generation: {exc}") from exc

    for chunk in stream:
        text = _extract_stream_text(chunk)
        if not text:
            continue

        response_parts.append(text)
        print(text, end="", flush=True)

        if speaker is not None:
            for chunk_text in accumulator.push(text):
                speaker.speak_async(chunk_text)

    print()

    if speaker is not None:
        for tail in accumulator.flush():
            speaker.speak_async(tail)
        speaker.drain()

    response_text = "".join(response_parts).strip()
    messages.append({"role": "assistant", "content": response_text})
    return response_text


def _build_system_prompt(settings: dict[str, Any], override: str | None) -> str:
    if override:
        return override

    system_name = str(settings.get("system_name", "Astra"))
    wake_word = str(settings.get("wake_word", "")).strip()
    wake_word_line = f" The wake word is {wake_word}." if wake_word else ""
    return (
        f"You are {system_name}, a local offline assistant.{wake_word_line} "
        "Reply in plain text that sounds natural when spoken aloud. "
        "Keep answers concise unless the user asks for more detail."
    )


def _extract_stream_text(chunk: Any) -> str:
    if not isinstance(chunk, dict):
        return ""

    choices = chunk.get("choices")
    if not choices:
        return ""

    choice = choices[0]
    if not isinstance(choice, dict):
        return ""

    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content

    text = choice.get("text")
    return text if isinstance(text, str) else ""


class NvidiaLLMWrapper:
    def __init__(
        self,
        api_key: str,
        model: str,
        thinking: bool = True,
        reasoning_effort: str = "high",
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AstraRuntimeError(
                "Missing dependency `openai`. Run `python -m pip install -r requirements.txt`."
            ) from exc
            
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )
        self.model = model
        self._thinking = bool(thinking)
        self._reasoning_effort = reasoning_effort

    def create_chat_completion(self, messages, temperature, max_tokens, stream, response_format=None):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if self._thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": self._reasoning_effort,
                }
            }
        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)

        if stream:
            def generator():
                for chunk in response:
                    yield chunk.model_dump()
            return generator()
        else:
            return response.model_dump()


def _load_llm(model_path: Path | None, settings: dict[str, Any]) -> Any:
    llm_type = settings.get("llm_type", "local")
    if llm_type == "nvidia":
        api_key = settings.get("nvidia_api_key")
        model = settings.get("nvidia_model")
        if not api_key or not model:
            raise AstraRuntimeError("Nvidia NIM requires 'nvidia_api_key' and 'nvidia_model' in config/settings.yaml")
        return NvidiaLLMWrapper(
            api_key,
            model,
            thinking=settings.get("nvidia_thinking", True),
            reasoning_effort=str(settings.get("nvidia_reasoning_effort", "high")),
        )

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise AstraRuntimeError(
            "Missing dependency `llama-cpp-python`. Run `python -m pip install -r requirements.txt`."
        ) from exc

    gpu_layers = _coerce_int(settings.get("gpu_layers"), default=0)
    thread_count = max(1, (os.cpu_count() or 2) - 1)

    def build(n_gpu_layers: int) -> Any:
        return Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=thread_count,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    try:
        return build(gpu_layers)
    except Exception as first_error:
        if gpu_layers != 0:
            print(
                f"LLM load with gpu_layers={gpu_layers} failed, retrying on CPU.",
                flush=True,
            )
            try:
                return build(0)
            except Exception as second_error:
                raise AstraRuntimeError(
                    f"Failed to load llm model `{model_path.name if model_path else 'Unknown'}`: {second_error}"
                ) from second_error

        raise AstraRuntimeError(
            f"Failed to load llm model `{model_path.name if model_path else 'Unknown'}`: {first_error}"
        ) from first_error


def _load_piper_voice_class() -> Any:
    try:
        from piper import PiperVoice
    except ImportError as exc:
        raise AstraRuntimeError(
            "Missing dependency `piper-tts`. Run `python -m pip install -r requirements.txt`."
        ) from exc

    return PiperVoice


def _load_sounddevice_module() -> Any:
    try:
        import sounddevice
    except ImportError as exc:
        raise AstraRuntimeError(
            "Missing dependency `sounddevice`. Run `python -m pip install -r requirements.txt`."
        ) from exc

    return sounddevice


def _require_project_file(value: Any, label: str) -> Path:
    path = resolve_project_path(value)
    if path is None:
        raise AstraRuntimeError(f"Configured {label} path is not set.")
    if not path.exists():
        raise AstraRuntimeError(f"Configured {label} path does not exist: {path}")
    if not path.is_file():
        raise AstraRuntimeError(f"Configured {label} path is not a file: {path}")
    return path


def _normalize_device_index(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    text = str(value).strip()
    if not text:
        return None

    try:
        return int(text)
    except ValueError as exc:
        raise AstraRuntimeError(f"Invalid output_device_index: {value!r}") from exc


def _resolve_output_device(sounddevice_module: Any, value: Any) -> int | None:
    device_index = _normalize_device_index(value)
    if device_index is None:
        return None

    try:
        device_info = sounddevice_module.query_devices(device_index)
    except Exception as exc:
        raise AstraRuntimeError(
            f"Configured output_device_index {device_index} is not available: {exc}"
        ) from exc

    if int(device_info["max_output_channels"]) > 0:
        return device_index

    print(
        f"Configured output_device_index {device_index} is not an output device. "
        "Using the system default output instead.",
        flush=True,
    )
    return None


def _find_clause_break(text: str, min_chunk_chars: int, max_chunk_chars: int) -> int | None:
    search_limit = min(len(text), max_chunk_chars)
    best_end: int | None = None

    for index, char in enumerate(text[:search_limit], start=1):
        if index < min_chunk_chars:
            continue
        if char in ",;:\n":
            best_end = index

    if best_end is None:
        return None

    while best_end < len(text) and text[best_end] in "\"')]} ":
        best_end += 1

    return best_end


def _find_word_break(text: str, min_chunk_chars: int, max_chunk_chars: int) -> int | None:
    search_limit = min(len(text), max_chunk_chars)

    for index in range(search_limit, min_chunk_chars - 1, -1):
        if text[index - 1].isspace():
            return index

    return search_limit if search_limit > 0 else None


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    try:
        return int(str(value).strip())
    except ValueError:
        return default
