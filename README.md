# Gemini Whisper Dictation App

A desktop application for real-time dictation and transcription using Google's Gemini API, with features designed for clean, publication-ready text output. Includes system tray integration and audio cues.

## Features

*   **Real-time Dictation:** Press a hotkey to start/stop recording your voice.
*   **Gemini Powered Transcription:** Utilizes Google's Gemini models for accurate speech-to-text.
*   **Advanced Prompting:** Employs a comprehensive system prompt to ensure clean transcriptions, removing filler words, stutters, and applying proper formatting.
*   **Automatic Text Insertion:** Transcribed text is automatically typed into the active application window.
*   **Customizable Model & Prompt:** Easily change the Gemini model and refine the system prompt through the UI.
*   **API Usage Tracking:** Basic tracking for daily and total API calls.
*   **System Tray Integration:** Minimize the app to the system tray for unobtrusive operation.
*   **Audio Cues:** Optional sound notifications for starting and stopping dictation.
*   **Cross-Platform (Intended):** Built with Tkinter, aiming for compatibility across Windows, macOS, and Linux (system tray and hotkeys might have platform-specific behaviors).

## Prerequisites

*   **Python 3.7+**
*   **Pip** (Python package installer)
*   **Git** (for cloning the repository)
*   **A Google Gemini API Key:** You'll need an API key from Google AI Studio or Google Cloud.
*   **PortAudio** (for `sounddevice`):
    *   **Windows:** Usually comes with Python.
    *   **macOS:** `brew install portaudio`
    *   **Linux (Debian/Ubuntu):** `sudo apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev`
    *   **Linux (Fedora):** `sudo dnf install portaudio-devel`

## Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Vansio3/gemini-whisper.git
    cd gemini-whisper
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Initial Configuration (API Key):**
    *   The `dictation_config.json` file (which stores your settings and API key) will be created automatically in the project's root directory when you first run the application or save settings. 
    *   When you run the application for the first time (see "Running the Application" below), the main window will appear.
    *   Enter your **Google Gemini API Key** into the "Google API Key" field.
    *   You can also select your preferred Gemini Model and review/customize the System Prompt at this stage.
    *   Click the **"Save Settings & Apply API Key"** button. This will save your API key and other settings to `dictation_config.json`.

5.  **Place Sound Assets (Optional):**
    If you want different sound cues for dictation start/stop:
    *   Upload sounds in the `assets` folder in the project's root directory.
    *   Place your `dictation_started.mp3` and `dictation_stopped.mp3` files into this `assets` folder.
    If these files are not present, sound cues will be disabled gracefully.

6.  **Place Icon (Optional for System Tray):**
    If you want a custom icon for the system tray:
    *   Place an `icon.png` file in the project's root directory.
    If not found, a placeholder icon will be used (if `pystray` and `Pillow` are installed).

## Running the Application

Once dependencies are installed and configuration is complete:
```bash
python gemini-whisper-app.py
