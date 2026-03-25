"""
Sight Assist AI
---------------
A vision-based AI assistant for visually impaired users.
Captures an image via webcam, accepts a spoken voice query,
and responds with GPT-4 Vision in the user's desired language via text-to-speech.

Pipeline:
    1. Record audio query (microphone)
    2. Transcribe audio using Whisper API
    3. Capture image from webcam (OpenCV)
    4. Analyze image + query using GPT-4 Vision
    5. Translate response using T5
    6. Speak response using gTTS
"""

import os
import wave
import base64

import cv2
import torch
import numpy as np
import sounddevice as sd
import sentencepiece

from PIL import Image
from gtts import gTTS
from IPython.display import Audio
from dotenv import load_dotenv, find_dotenv

import openai
from openai import OpenAI

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# ── Environment ──────────────────────────────────────────────────────────────

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=openai.api_key)

# ── Model Loading ─────────────────────────────────────────────────────────────

print("Loading BLIP image captioning model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

print("Loading T5 translation model...")
t5_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

# ── Audio Settings ────────────────────────────────────────────────────────────

SAMPLE_RATE = 44100
DURATION = 5  # seconds


# ── Functions ─────────────────────────────────────────────────────────────────

def record_audio(file_path="temp_audio.wav", duration=DURATION):
    """Record audio from microphone and save to file."""
    print("Recording... Speak now!")
    audio_data = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16
    )
    sd.wait()
    print("Recording complete!")

    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

    return file_path


def transcribe_audio(file_path):
    """Transcribe audio file to text using OpenAI Whisper."""
    with open(file_path, "rb") as audio_file:
        response = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return response.text


def capture_image(save_path="captured_image.jpg"):
    """Capture image from webcam. Press SPACE to capture."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        cv2.imshow("Press SPACE to Capture Image", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            print(f"Image captured! Saved to {save_path}")
            cv2.imwrite(save_path, frame)
            break

    cap.release()
    cv2.destroyAllWindows()
    return save_path


def generate_caption(image):
    """Generate a text caption for an image using BLIP."""
    inputs = blip_processor(image, return_tensors="pt")
    outputs = blip_model.generate(**inputs)
    caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
    return caption


def analyze_image_with_text(image_path, user_query):
    """Send image and user query to GPT-4 Vision and return response."""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for visually impaired users. Be concise and clear."},
            {"role": "user", "content": [
                {"type": "text", "text": user_query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=200,
        temperature=0.5
    )
    return response.choices[0].message.content


def text_translation(text, target_lang="German"):
    """Translate text to target language using T5."""
    input_text = f"translate English to {target_lang}: {text}"
    inputs = t5_tokenizer(input_text, return_tensors="pt")
    outputs = t5_model.generate(**inputs)
    translated_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def text_to_speech(text, lang="hi", output_path="output.mp3"):
    """Convert text to speech using gTTS and play audio."""
    if not text:
        raise ValueError("No text provided for speech synthesis.")
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_path)
    return Audio(output_path, autoplay=True)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main(speech_lang="te"):
    """
    Run the full Sight Assist pipeline.

    Args:
        speech_lang: Language code for speech output.
                     Examples: 'en' English, 'de' German,
                               'hi' Hindi, 'te' Telugu
    """
    # Step 1: Record voice query
    audio_file = record_audio()

    # Step 2: Transcribe query with Whisper
    transcribed_text = transcribe_audio(audio_file)
    print(f"Transcribed query: {transcribed_text}")

    # Step 3: Capture image from webcam
    image_path = capture_image()

    # Step 4: Analyze image + query with GPT-4 Vision
    response = analyze_image_with_text(image_path, user_query=transcribed_text)
    print(f"GPT-4 Response: {response}")

    # Step 5: Translate response
    translated_text = text_translation(text=response, target_lang="German")
    print(f"Translated: {translated_text}")

    # Step 6: Speak response
    return text_to_speech(text=translated_text, lang=speech_lang)


if __name__ == "__main__":
    main()
