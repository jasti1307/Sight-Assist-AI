# Sight Assist AI

A vision-based AI assistant designed for visually impaired users. The system captures an image via webcam, accepts a spoken voice query, and responds in the user's desired language using text-to-speech output.

## Pipeline

```
Microphone → Whisper (speech-to-text) → Webcam → GPT-4 Vision (image + query) → T5 (translation) → gTTS (speech output)
```

## Features

- Real-time image capture via webcam (OpenCV)
- Voice query input transcribed using OpenAI Whisper
- Scene understanding and question answering via GPT-4 Vision
- Multilingual response translation using T5
- Text-to-speech output in user's preferred language (Telugu, Hindi, German, English, and more)

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/sight-assist-ai.git
cd sight-assist-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your OpenAI API key
Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_api_key_here
```

### 4. Run
```bash
python sight_assist.py
```

## Usage

1. Run the script
2. Speak your question when prompted (e.g. *"What is in front of me?"*)
3. Press **SPACE** to capture an image from the webcam
4. The assistant will describe the scene and answer your question
5. The response is spoken aloud in your configured language

To change the output language, edit the `main()` call at the bottom of the script:
```python
main(speech_lang="te")   # Telugu
main(speech_lang="hi")   # Hindi
main(speech_lang="de")   # German
main(speech_lang="en")   # English
```

## Tech Stack

| Component | Technology |
|---|---|
| Image Captioning | BLIP (Salesforce) |
| Speech to Text | OpenAI Whisper |
| Image + Query Understanding | GPT-4 Vision |
| Translation | T5 (Google) |
| Text to Speech | gTTS |
| Computer Vision | OpenCV |

## Requirements

See `requirements.txt`. Requires an OpenAI API key for Whisper and GPT-4 Vision.
