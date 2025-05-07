# --- Main Application File: gemini-whisper-app.py ---

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import os
import sys
import threading
from datetime import date
import time
import traceback

# --- System Tray Imports ---
try:
    import pystray
    from PIL import Image, ImageDraw # For icon loading/creation
    PYSTRAY_AVAILABLE = True
except ImportError:
    print("Warning: 'pystray' or 'Pillow' not found. System tray functionality will be disabled.")
    print("Install them with: pip install pystray Pillow")
    PYSTRAY_AVAILABLE = False

# --- Audio Playback with Pygame Mixer ---
try:
    import pygame
    PYGAME_MIXER_AVAILABLE = True
    pygame.mixer.init()
    print("Pygame mixer initialized successfully for sound cues.")
except ImportError:
    print("Warning: 'pygame' library not found. Sound cues will be disabled.")
    print("Install it with: pip install pygame")
    PYGAME_MIXER_AVAILABLE = False
except pygame.error as e:
    print(f"Warning: Pygame mixer failed to initialize: {e}. Sound cues will be disabled.")
    PYGAME_MIXER_AVAILABLE = False

# --- Existing Dictation Imports ---
import google.generativeai as genai
import sounddevice as sd
import soundfile as sf
import numpy as np
import keyboard
import pyautogui
import io
from scipy import signal

# --- Configuration File ---
CONFIG_FILE = "dictation_config.json"
FIXED_HOTKEY = "ctrl+alt+d"
ICON_PATH = "icon.png" # Path to your icon image, should be in the same dir as script

# --- Sound Files (MP3s are fine with pygame) ---
SOUND_ASSETS_DIR = "assets"
SOUND_DICTATION_STARTED = os.path.join(SOUND_ASSETS_DIR, "dictation_started.mp3")
SOUND_DICTATION_STOPPED = os.path.join(SOUND_ASSETS_DIR, "dictation_stopped.mp3")

# --- Default Prompt ---
COMPREHENSIVE_DEFAULT_PROMPT = (
"Transcribe speech accurately.\n\nStrictly OMIT: All filler words (um, uh, ah, like, you know, so), hesitations, and false starts (transcribe only the corrected phrase).\nNo Speech: If audio contains no discernible words (silence, pure noise), output a completely empty string\n\n"
"You are an highly skilled AI Transcription Engine specializing in converting spoken audio into meticulously formatted, publication-ready text. Your primary directive is to produce exceptionally clean, accurate, and readable transcriptions suitable for direct use.\n\n"
"Core Objective: Transform raw spoken audio into polished written text, removing all verbal clutter and adhering strictly to the rules below.\n\n"
"I. Transcription Accuracy and Verbatim Content:\n"
"1.  Intended Speech: Transcribe the speaker's intended words and phrases with the highest possible accuracy. Focus on the core message and meaning.\n"
"2.  Grammar Correction: Correct obvious grammatical errors or slips of the tongue if the speaker's intent is clear and the correction improves readability and professionalism. Do not change the speaker's meaning.\n"
"3.  Word Choice: Prefer standard vocabulary and avoid transcribing overly colloquial or unclear slang unless it's essential to the meaning and context.\n\n"
"II. Formatting and Style Rules:\n"
"4.  Punctuation: Apply standard English punctuation (periods, commas, question marks, exclamation points, semicolons, colons, apostrophes, quotation marks) correctly based on speech pauses, intonation, and grammatical structure to ensure clarity and readability.\n"
"5.  Capitalization: Use proper capitalization for sentence beginnings, proper nouns (names of people, places, organizations, titles), acronyms, and other standard English capitalization rules.\n"
"6.  Numbers: Transcribe numbers as digits (e.g., '10' instead of 'ten') for quantities, measurements, and most general cases. For very small numbers used in a narrative sense (e.g., 'one or two examples'), transcribing as words is acceptable if it enhances flow. Be consistent.\n"
"7.  Dates & Times: Format dates and times conventionally (e.g., 'December 5, 2023', '3:30 PM').\n"
"8.  Acronyms & Abbreviations: Transcribe known acronyms in uppercase (e.g., NASA, FBI). If an abbreviation is spoken, transcribe it as spoken.\n\n"
"III. Content Cleaning and Disfluency Management - CRITICAL SECTION:\n"
"9.  Filler Words & Hesitations (Strict Omission):\n"
"    *   ABSOLUTELY OMIT ALL filler words and vocal hesitations. This is a critical instruction.\n"
"    *   DO NOT transcribe sounds/words like: 'um', 'uh', 'er', 'ah', 'hmm', 'mhm', 'huh', 'hnh'.\n"
"    *   DO NOT transcribe common filler phrases such as: 'you know', 'like' (when used as a verbal tic, not as a verb/conjunction), 'I mean', 'so' (when used as a hesitation or lead-in without semantic value), 'well' (when used as a hesitation marker, not as an adverb).\n"
"    *   The final transcription MUST be entirely free of these. Edit them out completely.\n\n"
"10. False Starts & Repetitions:\n"
"    *   If a speaker starts a word or phrase, stumbles, and then immediately corrects or fluently repeats it, transcribe ONLY the final, fluent, and corrected version.\n"
"    *   Example: Audio 'I want to go to the... the park' should be transcribed as 'I want to go to the park'.\n"
"    *   Example: Audio 'She was, uh, she was very happy' should be transcribed as 'She was very happy'.\n\n"
"11. Stutters & Self-Corrections: Smooth out minor stutters on single words if the word is clear. For more significant self-corrections, refer to rule #10.\n\n"
"IV. Handling Non-Speech and Edge Cases:\n"
"12. Silence & Non-Verbal Sounds:\n"
"    *   If the audio segment contains NO discernible spoken words (e.g., it's pure silence, or only contains background noise like clicks, hums, static, or unintelligible murmurs), the transcribed text output MUST be a completely empty string ('').\n"
"    *   DO NOT generate placeholders like '[silence]', '[no speech]', '[noise]', or any other descriptive text for such segments.\n\n"
"13. Unintelligible Speech: If a specific word or short phrase is genuinely unintelligible due to noise or poor audio quality, and reasonable effort does not decipher it, you may use '[unintelligible]' as a placeholder for that specific segment. Use this sparingly and only when absolutely necessary. If the entire audio is unintelligible, treat it as per rule #12 (empty string).\n\n"
"V. Final Output Quality:\n"
"14. Fluency & Readability: The final transcription should be fluent, natural-sounding, and easy to read, as if it were professionally edited written text.\n"
"15. Consistency: Maintain consistency in formatting and style throughout the transcription.\n\n"
"Your ultimate goal is to transform spoken audio into a clean, accurate, and polished written document, ready for immediate use without further editing for disfluencies or basic formatting."
)
DEFAULT_PROMPT = COMPREHENSIVE_DEFAULT_PROMPT

# --- Model Configuration ---
DEFAULT_MODEL_CHOICE = "gemini-1.5-flash-latest"
AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-latest",
]
if DEFAULT_MODEL_CHOICE not in AVAILABLE_MODELS:
    AVAILABLE_MODELS.insert(0, DEFAULT_MODEL_CHOICE)

# --- Global UI Constants ---
APP_FONT_FAMILY = "Segoe UI" if os.name == 'nt' else ("Helvetica" if sys.platform != "darwin" else "Arial")
APP_FONT_SIZE = 10
APP_FONT = (APP_FONT_FAMILY, APP_FONT_SIZE)
APP_FONT_BOLD = (APP_FONT_FAMILY, APP_FONT_SIZE, "bold")

TEXT_AREA_FONT_FAMILY = "Consolas" if os.name == 'nt' else ("Monaco" if sys.platform == "darwin" else "Monospace")
TEXT_AREA_FONT_SIZE = APP_FONT_SIZE
TEXT_AREA_FONT = (TEXT_AREA_FONT_FAMILY, TEXT_AREA_FONT_SIZE)

PAD_XL = 15
PAD_L = 10
PAD_M = 5
PAD_S = 2

try:
    from ttkthemes import ThemedTk
    TTKTHEMES_INSTALLED = True
except ImportError:
    TTKTHEMES_INSTALLED = False


class DictationApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title(f"Gemini Dictation (Hotkey: {FIXED_HOTKEY})")
        self.root.geometry("650x700")
        self.root.minsize(600, 650)

        self.is_themed_app = TTKTHEMES_INSTALLED and hasattr(self.root, 'set_theme')

        self.settings = {}
        self.api_stats = {"daily_calls": 0, "last_call_date": str(date.today()), "total_calls": 0}

        self.is_recording = False
        self.audio_frames = []
        self.audio_stream = None
        self.hotkey_listener_active = False
        self.hotkey_listener_thread = None

        # --- Tray Icon Attributes ---
        self.tray_icon = None
        self.tray_thread = None
        self.is_quitting_via_tray = False
        self.can_use_tray = PYSTRAY_AVAILABLE
        # --- End Tray Icon Attributes ---

        self.start_sound_obj = None
        self.stop_sound_obj = None
        if PYGAME_MIXER_AVAILABLE:
            if os.path.exists(SOUND_DICTATION_STARTED):
                try:
                    self.start_sound_obj = pygame.mixer.Sound(SOUND_DICTATION_STARTED)
                except pygame.error as e:
                    print(f"Error loading start sound '{SOUND_DICTATION_STARTED}': {e}")
            else:
                print(f"Warning: Start sound file not found: {SOUND_DICTATION_STARTED}")

            if os.path.exists(SOUND_DICTATION_STOPPED):
                try:
                    self.stop_sound_obj = pygame.mixer.Sound(SOUND_DICTATION_STOPPED)
                except pygame.error as e:
                    print(f"Error loading stop sound '{SOUND_DICTATION_STOPPED}': {e}")
            else:
                print(f"Warning: Stop sound file not found: {SOUND_DICTATION_STOPPED}")

        if not self.is_themed_app:
            self._apply_fallback_styles()

        self.load_config()
        self.setup_ui()
        self.populate_ui_from_config()

        if self.settings.get("api_key"):
            self.apply_api_key_and_start_listener(self.settings["api_key"])
        else:
            self.status_var.set("Status: API Key required. Configure in settings.")

        if self.can_use_tray:
            self.root.protocol("WM_DELETE_WINDOW", self.hide_to_tray)
            self.setup_tray_icon()
        else:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing) # Default close if no tray

    def _apply_fallback_styles(self):
        print("Applying fallback ttk styles.")
        style = ttk.Style()
        try:
            # Prefer 'vista' on Windows if available for a more modern look than 'clam'
            if os.name == 'nt': # 'nt' is the OS name for Windows
                 style.theme_use('vista')
            elif sys.platform == "darwin": # macOS
                 style.theme_use('aqua')
            else: # Linux and other OSes
                 style.theme_use('clam') # 'clam' is a common fallback
        except tk.TclError:
            print("Fallback ttk theme (vista/aqua/clam) failed, using system default.")
            # If 'clam' or others fail, Tkinter will use a very basic default.

        style.configure('.', font=APP_FONT, padding=PAD_S)
        style.configure('TLabel', padding=PAD_M)
        style.configure('TButton', font=APP_FONT_BOLD, padding=(PAD_L, PAD_M))
        style.configure('Bold.TLabel', font=APP_FONT_BOLD)
        style.configure('Status.TLabel', padding=PAD_M)
        style.configure('TEntry', padding=(PAD_M, PAD_S + 1)) # Adjusted padding for TEntry
        style.configure('TCombobox', padding=(PAD_M, PAD_S + 1)) # Adjusted padding for TCombobox
        style.configure("InputArea.TFrame", relief="sunken", borderwidth=1)


    def play_sound_async(self, sound_object_to_play):
        if PYGAME_MIXER_AVAILABLE and sound_object_to_play:
            try:
                sound_object_to_play.play()
            except Exception as e:
                print(f"Error playing sound with pygame: {e}")
        elif PYGAME_MIXER_AVAILABLE and not sound_object_to_play:
            print(f"Warning: Sound object not loaded or invalid.")

    def load_config(self):
        default_settings_template = {"api_key": "", "model": DEFAULT_MODEL_CHOICE, "prompt": DEFAULT_PROMPT}
        default_api_stats_template = {"daily_calls": 0, "last_call_date": str(date.today()), "total_calls": 0}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config_data = json.load(f)
                self.settings = {**default_settings_template, **config_data.get("settings", {})}
                self.api_stats = {**default_api_stats_template, **config_data.get("api_stats", {})}
                today_str = str(date.today())
                if self.api_stats.get("last_call_date") != today_str:
                    self.api_stats["daily_calls"] = 0
                    self.api_stats["last_call_date"] = today_str
                    self.save_config()
            except (json.JSONDecodeError, Exception) as e:
                messagebox.showerror("Config Error", f"Error loading {CONFIG_FILE}: {e}. Using defaults.")
                self.initialize_default_config()
        else:
            self.initialize_default_config()
            self.save_config()

    def initialize_default_config(self):
        self.settings = {"api_key": "", "model": DEFAULT_MODEL_CHOICE, "prompt": DEFAULT_PROMPT}
        self.api_stats = {"daily_calls": 0, "last_call_date": str(date.today()), "total_calls": 0}

    def save_config(self):
        config_data = {"settings": self.settings, "api_stats": self.api_stats}
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            # Use status bar for this error if available, less intrusive
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Error saving config: {e}")
            else:
                messagebox.showerror("Config Save Error", f"Could not save config: {e}")

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=PAD_XL)
        main_frame.pack(expand=True, fill=tk.BOTH)

        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(3, weight=1) # Row index for prompt_outer_frame

        current_row = 0

        ttk.Label(main_frame, text="Google API Key:").grid(row=current_row, column=0, sticky=tk.W, pady=(0, PAD_M), padx=(0, PAD_M))
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(main_frame, textvariable=self.api_key_var, width=60, show="*")
        self.api_key_entry.grid(row=current_row, column=1, sticky=tk.EW, pady=(0, PAD_M))
        current_row += 1

        ttk.Label(main_frame, text="Gemini Model:").grid(row=current_row, column=0, sticky=tk.W, pady=PAD_M, padx=(0, PAD_M))
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(main_frame, textvariable=self.model_var, values=AVAILABLE_MODELS, state="readonly", width=57)
        self.model_dropdown.grid(row=current_row, column=1, sticky=tk.EW, pady=PAD_M)
        if not self.is_themed_app:
            self.model_dropdown.option_add('*TCombobox*Listbox.font', APP_FONT)
        current_row += 1

        ttk.Label(main_frame, text="Dictation Hotkey:").grid(row=current_row, column=0, sticky=tk.W, pady=PAD_M, padx=(0, PAD_M))
        hotkey_val_label = ttk.Label(main_frame, text=FIXED_HOTKEY)
        if not self.is_themed_app: hotkey_val_label.configure(style="Bold.TLabel")
        else: hotkey_val_label.configure(font=APP_FONT_BOLD)
        hotkey_val_label.grid(row=current_row, column=1, sticky=tk.W, pady=PAD_M)
        current_row += 1

        ttk.Label(main_frame, text="System Prompt:").grid(row=current_row, column=0, sticky=tk.NW, pady=(PAD_M, 0), padx=(0, PAD_M))
        prompt_outer_frame = ttk.Frame(main_frame)
        prompt_outer_frame.grid(row=current_row, column=1, sticky=tk.NSEW, pady=PAD_M)
        prompt_outer_frame.grid_rowconfigure(0, weight=1)
        prompt_outer_frame.grid_columnconfigure(0, weight=1)

        self.prompt_text = scrolledtext.ScrolledText(
            prompt_outer_frame, wrap=tk.WORD, width=70, height=18, font=TEXT_AREA_FONT,
            relief=tk.FLAT, borderwidth=0, highlightthickness=0, padx=PAD_S, pady=PAD_S
        )
        self.prompt_text.grid(row=0, column=0, sticky=tk.NSEW)

        s = ttk.Style()
        try:
            bg_color = s.lookup('TEntry', 'fieldbackground')
            fg_color = s.lookup('TEntry', 'foreground')
            insert_color = fg_color
            try:
                insert_color_lookup = s.lookup('TEntry', 'insertcolor')
                if insert_color_lookup: insert_color = insert_color_lookup
            except tk.TclError: pass

            self.prompt_text.configure(bg=bg_color, fg=fg_color, insertbackground=insert_color)

            if self.is_themed_app:
                prompt_outer_frame.configure(padding=1)
                try: prompt_outer_frame.configure(style="Editor.TFrame")
                except tk.TclError:
                    if not s.layout("TFrame") or "borderwidth" not in s.layout("TFrame")[0][1]:
                        prompt_outer_frame.configure(relief="sunken", borderwidth=1)
            else: # Non-themed app
                prompt_outer_frame.configure(style="InputArea.TFrame", padding=0)

        except tk.TclError as e:
            print(f"Style lookup failed for ScrolledText theming: {e}. Using basic ScrolledText border.")
            self.prompt_text.configure(relief=tk.SOLID, borderwidth=1) # Fallback border
        current_row += 1

        stats_frame = ttk.LabelFrame(main_frame, text="API Usage Stats", padding=PAD_L)
        stats_frame.grid(row=current_row, column=0, columnspan=2, sticky=tk.EW, pady=(PAD_L, PAD_M))
        self.daily_calls_var = tk.StringVar()
        ttk.Label(stats_frame, text="Calls Today:").pack(side=tk.LEFT, padx=(0, PAD_M))
        ttk.Label(stats_frame, textvariable=self.daily_calls_var).pack(side=tk.LEFT, padx=(0, PAD_L))
        self.total_calls_var = tk.StringVar()
        ttk.Label(stats_frame, text="Total Calls:").pack(side=tk.LEFT, padx=(0, PAD_M))
        ttk.Label(stats_frame, textvariable=self.total_calls_var).pack(side=tk.LEFT)
        current_row += 1

        self.status_var = tk.StringVar()
        self.status_var.set("Status: Initializing...")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, anchor=tk.W)
        if not self.is_themed_app: status_label.configure(style="Status.TLabel")
        else: status_label.configure(padding=(PAD_M, PAD_M))
        status_label.grid(row=current_row, column=0, columnspan=2, sticky=tk.EW, pady=(PAD_M, PAD_M))
        current_row += 1

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=current_row, column=0, columnspan=2, sticky=tk.EW, pady=(PAD_L, 0))
        button_frame.grid_columnconfigure(0, weight=1) # Empty space to push buttons to the right

        self.restore_prompt_button = ttk.Button(button_frame, text="Restore Default Prompt", command=self.restore_default_prompt_action)
        self.restore_prompt_button.grid(row=0, column=1, sticky=tk.E, padx=(0, PAD_M))

        self.save_button = ttk.Button(button_frame, text="Save Settings & Apply API Key", command=self.apply_ui_settings)
        self.save_button.grid(row=0, column=2, sticky=tk.E)
        current_row += 1

    def restore_default_prompt_action(self):
        confirm = messagebox.askyesno(
            "Restore Default Prompt",
            "Are you sure you want to restore the system prompt to its default value?\n"
            "This will overwrite the current content of the prompt text area.\n"
            "You will still need to click 'Save Settings' to make this change permanent."
        )
        if confirm:
            self.prompt_text.delete(1.0, tk.END)
            self.prompt_text.insert(tk.END, DEFAULT_PROMPT)
            self.status_var.set("Status: Default prompt restored to text area. Save settings to apply.")

    def populate_ui_from_config(self):
        self.api_key_var.set(self.settings.get("api_key", ""))
        current_model = self.settings.get("model", DEFAULT_MODEL_CHOICE)
        if current_model not in AVAILABLE_MODELS:
            current_model = DEFAULT_MODEL_CHOICE
            self.settings["model"] = current_model
        self.model_var.set(current_model)
        self.prompt_text.delete(1.0, tk.END)
        self.prompt_text.insert(tk.END, self.settings.get("prompt", DEFAULT_PROMPT))
        self.update_api_stats_display()

    def update_api_stats_display(self):
        self.daily_calls_var.set(str(self.api_stats.get("daily_calls", 0)))
        self.total_calls_var.set(str(self.api_stats.get("total_calls", 0)))

    def apply_ui_settings(self):
        new_api_key = self.api_key_var.get().strip()
        self.settings["api_key"] = new_api_key
        self.settings["model"] = self.model_var.get()
        self.settings["prompt"] = self.prompt_text.get(1.0, tk.END).strip()
        self.save_config()
        messagebox.showinfo("Settings Saved", "Settings have been saved.")
        self.apply_api_key_and_start_listener(new_api_key)

    def apply_api_key_and_start_listener(self, api_key_to_apply):
        if api_key_to_apply:
            try:
                genai.configure(api_key=api_key_to_apply)
                self.status_var.set(f"Status: Ready. Press {FIXED_HOTKEY} to dictate.")
                if not self.hotkey_listener_active:
                    self.start_hotkey_listener_thread()
            except Exception as e:
                self.status_var.set(f"Status: API Key Error. Check settings: {e}")
                messagebox.showerror("API Key Error", f"Failed to configure Gemini API: {e}")
                self.stop_hotkey_listener_thread()
        else:
            self.status_var.set("Status: API Key missing. Dictation disabled.")
            self.stop_hotkey_listener_thread()

    def increment_api_call(self):
        today_str = str(date.today())
        if self.api_stats.get("last_call_date") != today_str:
            self.api_stats["daily_calls"] = 0
            self.api_stats["last_call_date"] = today_str
        self.api_stats["daily_calls"] = self.api_stats.get("daily_calls", 0) + 1
        self.api_stats["total_calls"] = self.api_stats.get("total_calls", 0) + 1
        self.update_api_stats_display()
        self.save_config()

    def audio_callback(self, indata, frames, time_info, status):
        if status: print(status, flush=True)
        if self.is_recording: self.audio_frames.append(indata.copy())

    def toggle_dictation_mode(self):
        if not self.settings.get("api_key"):
            self.root.after(0, lambda: self.status_var.set("Status: API Key missing."))
            self.root.after(0, lambda: messagebox.showwarning("API Key Missing", "Please set your Google API Key in settings."))
            return

        if not self.is_recording:
            self.is_recording = True
            self.audio_frames = []
            self.play_sound_async(self.start_sound_obj)
            self.root.after(0, lambda: self.status_var.set("Status: Recording..."))
            try:
                self.audio_stream = sd.InputStream(samplerate=16000, channels=1, callback=self.audio_callback, dtype='float32')
                self.audio_stream.start()
            except Exception as e:
                self.is_recording = False
                self.root.after(0, lambda: self.status_var.set(f"Status: Audio Error - {e}"))
                messagebox.showerror("Audio Error", f"Could not start audio recording: {e}")
                if self.audio_stream:
                    try:
                        if not self.audio_stream.closed: self.audio_stream.close()
                    except: pass
                self.audio_stream = None
        else:
            self.is_recording = False
            self.play_sound_async(self.stop_sound_obj)
            self.root.after(0, lambda: self.status_var.set("Status: Processing... Please wait."))

            if self.audio_stream:
                try:
                    if not self.audio_stream.closed:
                        self.audio_stream.stop()
                        self.audio_stream.close()
                except Exception as e:
                    print(f"Error stopping/closing stream: {e}")
                finally:
                    self.audio_stream = None

            if not self.audio_frames:
                self.root.after(0, lambda: self.status_var.set(f"Status: No audio recorded. Press {FIXED_HOTKEY}."))
                return

            processing_thread = threading.Thread(target=self.process_recorded_audio_data_thread, daemon=True)
            processing_thread.start()

    def process_recorded_audio_data_thread(self):
        current_audio_frames_copy = list(self.audio_frames)
        self.audio_frames = []

        if not current_audio_frames_copy:
            self.root.after(0, lambda: self.status_var.set(f"Status: No audio data. Press {FIXED_HOTKEY}."))
            return
        try:
            audio_data_raw = np.concatenate(current_audio_frames_copy, axis=0)
            audio_data_float = audio_data_raw.astype(np.float32)
            audio_data_1d = audio_data_float[:, 0] if audio_data_float.ndim > 1 and audio_data_float.shape[1] >= 1 else audio_data_float.flatten()
            audio_data_to_send = audio_data_1d

            if len(audio_data_to_send) > 800: # Min samples for filter
                cutoff_hz, nyquist_rate = 100.0, 16000 / 2.0
                if cutoff_hz < nyquist_rate:
                    normalized_cutoff = cutoff_hz / nyquist_rate
                    if 0 < normalized_cutoff < 1:
                        try:
                            b, a = signal.butter(4, normalized_cutoff, btype='highpass', analog=False)
                            audio_data_to_send = signal.lfilter(b, a, audio_data_to_send)
                        except Exception as filter_e:
                            print(f"Warning: High-pass filter failed: {filter_e}")

            if len(audio_data_to_send) == 0:
                self.root.after(0, lambda: self.status_var.set(f"Status: Audio data too short. Press {FIXED_HOTKEY}."))
                return

            wav_io = io.BytesIO()
            sf.write(wav_io, audio_data_to_send, 16000, format='WAV', subtype='PCM_16')
            wav_io.seek(0)
            audio_blob = {'mime_type': 'audio/wav', 'data': wav_io.read()}

            model_name_to_use = self.settings.get("model", DEFAULT_MODEL_CHOICE)
            model = genai.GenerativeModel(model_name_to_use)
            prompt_to_use = self.settings.get("prompt", DEFAULT_PROMPT)

            self.root.after(0, lambda: self.status_var.set(f"Status: Transcribing with {model_name_to_use}..."))
            response = model.generate_content([prompt_to_use, audio_blob], request_options={"timeout": 120})
            self.increment_api_call()

            transcribed_text = ""
            if response.candidates and response.candidates[0].content.parts:
                transcribed_text = response.candidates[0].content.parts[0].text.strip()

            if transcribed_text:
                self.root.after(0, lambda: self.status_var.set("Status: Transcribed! Typing..."))
                time.sleep(0.05) # Shorter delay
                pyautogui.typewrite(transcribed_text + " ", interval=0.005) # Faster typing
                self.root.after(100, lambda: self.status_var.set(f"Status: Ready. Press {FIXED_HOTKEY}."))
            else:
                feedback_msg = "Status: No text transcribed."
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                    feedback_msg += f" Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}"
                elif hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.safety_ratings:
                    for rating in response.prompt_feedback.safety_ratings:
                        if rating.blocked:
                            feedback_msg += f" Blocked: {rating.category.name} ({rating.probability.name})."
                            break
                self.root.after(0, lambda: self.status_var.set(feedback_msg))
        except genai.types.generation_types.StopCandidateException as sce:
            self.root.after(0, lambda: self.status_var.set(f"Status: Transcription stopped by API. {sce}"))
            print(f"StopCandidateException: {sce}")
            if hasattr(sce, 'response') and sce.response.prompt_feedback: print(f"Prompt Feedback: {sce.response.prompt_feedback}")
        except Exception as e:
            error_info = traceback.format_exc()
            self.root.after(0, lambda: self.status_var.set(f"Status: Error - {str(e)[:100]}... Check console."))
            print(f"Error during processing or transcription: {e}\n{error_info}")
        finally:
            if not self.is_recording:
                self.root.after(200, lambda: self.status_var.set(f"Status: Ready. Press {FIXED_HOTKEY}."))

    def _actual_hotkey_listener_loop(self):
        print(f"Hotkey listener started for '{FIXED_HOTKEY}'.")
        try:
            try: keyboard.remove_hotkey(FIXED_HOTKEY)
            except: pass
            keyboard.add_hotkey(FIXED_HOTKEY, self.toggle_dictation_mode, suppress=False)
            self.hotkey_listener_active = True
            while self.hotkey_listener_active:
                time.sleep(0.1)
        except ImportError:
            self.root.after(0, lambda: messagebox.showerror("Hotkey Error", "Keyboard library missing. Hotkey disabled."))
            self.hotkey_listener_active = False
        except Exception as e:
            print(f"Error in hotkey listener thread for '{FIXED_HOTKEY}': {e}")
            self.root.after(0, lambda: messagebox.showerror("Hotkey Error", f"Could not set hotkey '{FIXED_HOTKEY}': {e}\nTry running as admin."))
            self.hotkey_listener_active = False
        finally:
            if 'keyboard' in sys.modules:
                try: keyboard.remove_hotkey(FIXED_HOTKEY)
                except Exception as e: print(f"Note: Error removing hotkey '{FIXED_HOTKEY}' on loop exit: {e}")
            print(f"Hotkey listener loop for '{FIXED_HOTKEY}' has ended.")

    def start_hotkey_listener_thread(self):
        if self.hotkey_listener_active and self.hotkey_listener_thread and self.hotkey_listener_thread.is_alive():
            return
        if not self.settings.get("api_key"):
            self.status_var.set("Status: API Key required to start listener.")
            return
        self.hotkey_listener_thread = threading.Thread(target=self._actual_hotkey_listener_loop, daemon=True)
        self.hotkey_listener_thread.start()
        self.root.after(200, self._check_hotkey_listener_status)

    def _check_hotkey_listener_status(self):
        if not (self.hotkey_listener_active and self.hotkey_listener_thread and self.hotkey_listener_thread.is_alive()):
            if self.settings.get("api_key"):
                self.status_var.set("Status: Hotkey listener failed. Check console/run as admin.")

    def stop_hotkey_listener_thread(self):
        if self.hotkey_listener_active:
            self.hotkey_listener_active = False
            if self.hotkey_listener_thread and self.hotkey_listener_thread.is_alive():
                self.hotkey_listener_thread.join(timeout=0.2) # Shorter join
        self.hotkey_listener_thread = None
        if not self.settings.get("api_key"):
            self.status_var.set("Status: API Key missing. Dictation disabled.")
        elif not self.hotkey_listener_active:
            self.status_var.set(f"Status: Hotkey listener stopped. Configure API Key and Save to restart.")

    # --- SYSTEM TRAY METHODS ---
    def _get_icon_image(self):
        try:
            image = Image.open(ICON_PATH)
            return image
        except FileNotFoundError:
            print(f"Warning: Icon file '{ICON_PATH}' not found. Creating placeholder.")
            try:
                img = Image.new('RGBA', (64, 64), (70, 130, 180, 255)) # SteelBlue
                d = ImageDraw.Draw(img)
                d.text((10,18), "GD", fill=(255,255,255,255), font_size=30) # Crude "GD"
                return img
            except Exception as e_placeholder:
                print(f"Could not create placeholder icon: {e_placeholder}")
                return None
        except Exception as e_load:
            print(f"Error loading icon '{ICON_PATH}': {e_load}")
            return None

    def show_window_action(self, icon=None, item=None):
        self.root.after(0, self.show_window)

    def quit_action(self, icon=None, item=None):
        self.is_quitting_via_tray = True
        if self.can_use_tray and self.tray_icon:
             self.tray_icon.stop()
        self.root.after(0, self.on_closing)

    def show_window(self):
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(100, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()

    def hide_to_tray(self):
        if self.is_quitting_via_tray:
            self.on_closing()
        else:
            self.root.withdraw()

    def setup_tray_icon(self):
        if not self.can_use_tray: return

        image = self._get_icon_image()
        menu_items = [
            pystray.MenuItem('Show Settings', self.show_window_action, default=True),
            pystray.MenuItem('Quit', self.quit_action)
        ]
        self.tray_icon = pystray.Icon("gemini_dictation_app", image, "Gemini Dictation", tuple(menu_items))

        def run_tray():
            try:
                self.tray_icon.run()
            except Exception as e:
                print(f"Error running tray icon: {e}")
            finally:
                print("Tray icon thread finished.")


        self.tray_thread = threading.Thread(target=run_tray, daemon=True)
        self.tray_thread.start()
    # --- END SYSTEM TRAY METHODS ---

    def on_closing(self):
        print("Closing application (on_closing called)...")
        is_actual_quit = self.is_quitting_via_tray

        if not self.can_use_tray:
            is_actual_quit = True
        elif not self.is_quitting_via_tray and self.root.winfo_exists() and self.root.winfo_viewable():
             print("Window 'X' clicked, but hide_to_tray should manage. If this appears, review logic.")
             return


        if self.can_use_tray and self.tray_icon:
            print("Stopping tray icon...")
            self.tray_icon.stop()

        print("Stopping hotkey listener...")
        self.stop_hotkey_listener_thread()

        if self.is_recording and self.audio_stream:
            self.is_recording = False
            print("Stopping active audio stream...")
            try:
                if self.audio_stream and not self.audio_stream.closed:
                    self.audio_stream.stop()
                    self.audio_stream.close()
                    print("Audio stream stopped and closed.")
            except Exception as e:
                print(f"Error closing audio stream on exit: {e}")
            self.audio_stream = None

        print("Saving configuration...")
        self.save_config()

        if PYGAME_MIXER_AVAILABLE:
            print("Quitting pygame mixer...")
            pygame.mixer.quit()
            print("Pygame mixer quit.")

        if self.root.winfo_exists():
            print("Destroying Tkinter root window...")
            self.root.destroy()
        print("Application closed.")


if __name__ == "__main__":
    chosen_theme = "radiance"
    root = None
    can_use_themed_tk = False

    if TTKTHEMES_INSTALLED:
        try:
            _test_root = ThemedTk(theme=chosen_theme, themebg=True)
            _test_root.destroy()
            del _test_root
            can_use_themed_tk = True
            print(f"ttkthemes is available. Attempting to use '{chosen_theme}' theme.")
        except tk.TclError as e:
            print(f"ttkthemes is installed, but failed to initialize theme '{chosen_theme}': {e}. Falling back.")
        except Exception as e_ttk:
            print(f"An unexpected error occurred while trying to use ttkthemes with '{chosen_theme}': {e_ttk}. Falling back.")
    else:
        print("ttkthemes not installed. Using standard Tk styling.")

    if can_use_themed_tk:
        root = ThemedTk(theme=chosen_theme, themebg=True)
    else:
        root = tk.Tk()

    app = DictationApp(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught. Closing application...")
        app.quit_action()
    finally:
        if app and not app.is_quitting_via_tray:
             if hasattr(app, 'on_closing') and callable(app.on_closing):
                  pass
        print("Mainloop finished or interrupted.")