import pyperclip
import time
import os
import shutil
from PIL import Image
import hashlib
import base64
import json
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import anthropic

# Configuration
ASSETS_DIR = "/home/path/assets"
LOG_FILE = "/home/path/clipboard_log.txt"
DATASET_FILE = "/home/path/dataset.jsonl"
SCREENSHOTS_DIR = os.path.expanduser("~/Pictures/Screenshots")

# AI Configuration
AI_PROVIDER = "openai" 
AI_MODEL = "gpt-4o-mini" 
SYSTEM_MESSAGE = "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."

# Ensure directories exist
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Initialize AI client
if AI_PROVIDER == "openai":
    client = OpenAI()
elif AI_PROVIDER == "anthropic":
    client = anthropic.Anthropic()
else:
    raise ValueError(f"Unsupported AI provider: {AI_PROVIDER}")

def log_event(event_type, content):
    timestamp = datetime.now().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp} - {event_type}: {content}\n")

def get_image_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def save_image(image_path):
    image_hash = get_image_hash(image_path)
    new_path = os.path.join(ASSETS_DIR, f"{image_hash}.png")
    shutil.copy(image_path, new_path)
    return new_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_latest_screenshot():
    screenshots = sorted(Path(SCREENSHOTS_DIR).glob("*.png"), key=os.path.getmtime, reverse=True)
    return screenshots[0] if screenshots else None

def process_clipboard_content():
    text_content = pyperclip.paste()
    latest_screenshot = get_latest_screenshot()
    
    if text_content:
        return {"type": "text", "content": text_content}
    elif latest_screenshot:
        file_path = save_image(latest_screenshot)
        return {"type": "image", "content": file_path}
    else:
        return None

def query_ai(content):
    try:
        if AI_PROVIDER == "openai":
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": []}
            ]
            
            if content["type"] == "text":
                messages[1]["content"].append({"type": "text", "text": f"This is my most recently copied clipboard content: {content['content']}\nWhat do you think?"})
            elif content["type"] == "image":
                base64_image = encode_image(content["content"])
                messages[1]["content"].extend([
                    {"type": "text", "text": "This is my most recently saved screenshot. What do you think?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                ])
            
            completion = client.chat.completions.create(
                model=AI_MODEL,
                messages=messages,
                max_tokens=999
            )
            response = completion.choices[0].message.content
        
        elif AI_PROVIDER == "anthropic":
            if content["type"] == "text":
                prompt = f"This is my most recently copied clipboard content: {content['content']}\nWhat do you think?"
            elif content["type"] == "image":
                base64_image = encode_image(content["content"])
                prompt = f"This is my most recently saved screenshot: [image: data:image/png;base64,{base64_image}]\nWhat do you think?"
            
            message = client.messages.create(
                model=AI_MODEL,
                max_tokens=999,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ]
            )
            response = message.content
        
        print("AI says:", response)
        return response
    except Exception as e:
        print("Error querying AI:", str(e))
        return None

def save_to_dataset(content, ai_response):
    data = {
        "timestamp": datetime.now().isoformat(),
        "content": content,
        "ai_response": ai_response,
        "ai_provider": AI_PROVIDER,
        "ai_model": AI_MODEL,
        "system_message": SYSTEM_MESSAGE
    }
    with open(DATASET_FILE, "a") as f:
        json.dump(data, f)
        f.write("\n")

def main():
    print("Monitoring clipboard and screenshots... Press Ctrl+C to exit.")
    last_content = None
    try:
        while True:
            current_content = process_clipboard_content()
            if current_content and current_content != last_content:
                log_event("Content Changed", f"Type: {current_content['type']}")
                ai_response = query_ai(current_content)
                if ai_response:
                    log_event("AI Response", ai_response[:50] + "..." if len(ai_response) > 50 else ai_response)
                    save_to_dataset(current_content, ai_response)
                last_content = current_content
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()