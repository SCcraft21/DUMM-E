"""Local-first voice interaction runtime with silent moral deliberation filtering."""

import abc
import queue
import re
import threading
from typing import Dict, List, Optional, Tuple, Iterator, Any
from pathlib import Path

from src.inference.generate import ConscienceGenerator, GenerationConfig
from src.tokenizer.wrapper import ConscienceTokenizer
from src.tokenizer.special_tokens import (
    CONSCIENCE_START_TOKEN,
    CONSCIENCE_END_TOKEN,
    ASSISTANT_TOKEN,
    ACTION_START_TOKEN,
    ACTION_END_TOKEN,
    REFUSE_TOKEN,
)
from src.voice.actions import LocalActionDispatcher


class BaseSTT(abc.ABC):
    """Abstract interface for local Speech-To-Text engines (e.g. faster-whisper)."""

    @abc.abstractmethod
    def listen_and_transcribe(self) -> str:
        """Capture microphone audio and return transcribed text string."""
        pass


class BaseTTS(abc.ABC):
    """Abstract interface for local Text-To-Speech engines (e.g. Piper, Kokoro)."""

    @abc.abstractmethod
    def speak_sentence(self, sentence: str) -> None:
        """Synthesize and vocalize a sentence."""
        pass


class ConsoleSTT(BaseSTT):
    """Console text fallback for testing and CLI environments without audio hardware."""

    def listen_and_transcribe(self) -> str:
        try:
            return input("\n[USER VOICE INPUT] > ")
        except (EOFError, KeyboardInterrupt):
            return "exit"


class ConsoleTTS(BaseTTS):
    """Console speech synthesizer printing spoken chunks in real-time."""

    def speak_sentence(self, sentence: str) -> None:
        clean = sentence.strip()
        if clean:
            print(f"[DUMM-E SPEAKING] 🔊 {clean}")


class SentenceStreamBuffer:
    """Buffers streaming token characters and yields complete sentences for zero-latency TTS."""

    SENTENCE_END_REGEX = re.compile(r'([.?!]+[\s\n]+|[\n]{2,})')

    def __init__(self) -> None:
        self.buffer = ""

    def add_token(self, token_str: str) -> List[str]:
        """Append token string and return list of completed sentences."""
        self.buffer += token_str
        sentences = []

        while True:
            match = self.SENTENCE_END_REGEX.search(self.buffer)
            if not match:
                break
            end_idx = match.end()
            sentence = self.buffer[:end_idx].strip()
            self.buffer = self.buffer[end_idx:]
            if sentence:
                sentences.append(sentence)

        return sentences

    def flush(self) -> Optional[str]:
        """Flush remaining text in buffer."""
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining if remaining else None


class VoiceActionRuntime:
    """Voice runtime engine orchestrating STT, silent deliberation, action dispatch, and TTS."""

    def __init__(
        self,
        generator: ConscienceGenerator,
        tokenizer: ConscienceTokenizer,
        dispatcher: Optional[LocalActionDispatcher] = None,
        stt: Optional[BaseSTT] = None,
        tts: Optional[BaseTTS] = None,
    ) -> None:
        self.generator = generator
        self.tokenizer = tokenizer
        self.dispatcher = dispatcher or LocalActionDispatcher()
        self.stt = stt or ConsoleSTT()
        self.tts = tts or ConsoleTTS()

    def run_turn(
        self,
        user_text: str,
        gen_config: Optional[GenerationConfig] = None,
    ) -> Dict[str, Any]:
        """Process a single conversational turn with silent deliberation and voice output.

        Args:
            user_text: Transcribed input prompt.
            gen_config: Generation configuration.

        Returns:
            Dictionary containing thought trace, executed actions, and spoken response.
        """
        cfg = gen_config or GenerationConfig()

        # Format input prompt
        formatted_prompt = self.tokenizer.format_dialogue(user_prompt=user_text)

        # State trackers
        in_deliberation = False
        in_action = False
        in_assistant = False

        deliberation_buffer = []
        action_buffer = []
        spoken_sentences = []
        sentence_buffer = SentenceStreamBuffer()
        current_risk_score = 0.0
        refusal_triggered = False

        print("\n[Dumm-E Core]: Processing input...")

        # Stream generation
        for tid, tstr, risk, is_refusal in self.generator.generate_stream(formatted_prompt, cfg):
            current_risk_score = risk

            if is_refusal:
                refusal_triggered = True
                refusal_msg = "I must decline this request due to synthetic conscience alignment constraints."
                self.tts.speak_sentence(refusal_msg)
                spoken_sentences.append(refusal_msg)
                break

            # State transitions based on special tokens
            if CONSCIENCE_START_TOKEN in tstr:
                in_deliberation = True
                continue
            elif CONSCIENCE_END_TOKEN in tstr:
                in_deliberation = False
                thought = "".join(deliberation_buffer).strip()
                print(f"[CONSCIENCE SILENT DELIBERATION]: 🧠 {thought} (Risk: {current_risk_score:.2f})")
                continue

            if ACTION_START_TOKEN in tstr:
                in_action = True
                continue
            elif ACTION_END_TOKEN in tstr:
                in_action = False
                action_str = "".join(action_buffer).strip()
                print(f"[ACTION INTENT]: ⚙️ {action_str}")
                # Execute action with conscience clearance
                obs = self.dispatcher.execute_action(
                    action_payload=action_str,
                    conscience_risk_score=current_risk_score,
                    risk_threshold=self.generator.model.config.action_risk_threshold,
                )
                print(f"[ACTION OBSERVATION]: 📊 {obs}")
                continue

            if ASSISTANT_TOKEN in tstr:
                in_assistant = True
                continue

            # Route tokens based on active block
            if in_deliberation:
                deliberation_buffer.append(tstr)
            elif in_action:
                action_buffer.append(tstr)
            elif in_assistant:
                # Vocalize assistant speech to TTS
                completed = sentence_buffer.add_token(tstr)
                for sent in completed:
                    self.tts.speak_sentence(sent)
                    spoken_sentences.append(sent)

        # Flush trailing speech
        final_sentence = sentence_buffer.flush()
        if final_sentence:
            self.tts.speak_sentence(final_sentence)
            spoken_sentences.append(final_sentence)

        return {
            "prompt": user_text,
            "deliberation": "".join(deliberation_buffer).strip(),
            "action": "".join(action_buffer).strip() if action_buffer else None,
            "spoken_response": " ".join(spoken_sentences),
            "refused": refusal_triggered,
            "risk_score": current_risk_score,
        }

    def start_interactive_loop(self) -> None:
        """Launch the continuous voice interaction loop."""
        print("=" * 80)
        print("          DUMM-E 220M: VOICE RUNTIME & CONSCIENCE SYSTEM ONLINE")
        print("=" * 80)
        print("Speak or type prompt. Say/type 'exit' or 'quit' to terminate.")

        while True:
            user_input = self.stt.listen_and_transcribe()
            if not user_input or user_input.strip().lower() in ["exit", "quit"]:
                print("\n[Dumm-E]: Shutting down voice runtime. Goodbye!")
                break

            self.run_turn(user_input.strip())

