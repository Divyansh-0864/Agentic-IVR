from gtts import gTTS
import os

os.makedirs("samples", exist_ok=True)

texts = {
    "hello_world.wav": "hello world this is a whisper benchmark test",
    "meeting_intro.wav": "good morning everyone welcome to the weekly meeting",
    "weather_report.wav": "today the weather is mostly sunny with a chance of light rain in the evening",
    "story_clip.wav": "once upon a time there was a little robot who wanted to learn to sing"
}

for filename, text in texts.items():
    tts = gTTS(text=text, lang='en')
    tts.save(os.path.join("data/samples", filename))
