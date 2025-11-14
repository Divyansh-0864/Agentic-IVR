from gtts import gTTS
import os

os.makedirs("data/samples", exist_ok=True)

texts = {
    "customer_support.wav": """Thank you for calling XYZ Bank’s customer service. 
    We appreciate your patience. Our representative will be with you shortly. 
    In the meantime, please have your account number ready for faster assistance. 
    You can also visit our website to check recent transactions or update your contact information.""",

    "travel_advisory.wav": """Good evening travelers. This is your daily travel advisory from the National Weather Center. 
    Heavy rainfall is expected across the northern region starting tonight, with possible flight delays at major airports. 
    Commuters are advised to plan extra travel time and stay updated on local transportation alerts.""",

    "tech_news.wav": """In today’s tech news, artificial intelligence continues to revolutionize industries from healthcare to entertainment. 
    Researchers have developed smaller, faster models capable of running efficiently on edge devices. 
    Experts believe this could open up a new generation of smart applications for everyday users.""",

    "story_clip.wav": """Once upon a time in a quiet coastal town, a young inventor dreamed of building a flying machine. 
    Every morning, he tested new designs on the windy cliffs, learning from each failure and trying again. 
    One summer afternoon, his creation finally lifted off the ground, soaring gracefully into the orange sunset.""",

    "ivr_demo.wav": """Welcome to the demo of our intelligent IVR system powered by speech recognition. 
    You can say things like, 'Check my account balance,' or 'Transfer money to savings.' 
    Our system will understand your intent and respond instantly, making phone banking simple and secure."""
}

for filename, text in texts.items():
    tts = gTTS(text=text, lang='en')
    tts.save(os.path.join("data/samples", filename))
