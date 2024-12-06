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
import colorama
from colorama import Fore, Style
from typing import List, Dict
import ollama

# Configuration
ASSETS_DIR = os.path.expanduser("~/clipboard_assets")
LOG_FILE = os.path.expanduser("~/clipboard_log.txt")
DATASET_FILE = os.path.expanduser("~/clipboard_dataset.jsonl")
SCREENSHOTS_DIR = os.path.expanduser("~/Screenshots")

# AI Configuration
AI_PROVIDER = "ollama"
AI_MODEL = "llama3.2:1b"
SYSTEM_MESSAGE = "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."

# Add new configuration
AGENT_CONFIG = {
    "curator": {
        "name": "Curator",
        "color": Fore.MAGENTA,
        "personality": "An enthusiastic collector and organizer of digital artifacts, always excited to catalog new findings."
    },
    "analyst": {
        "name": "Analyst",
        "color": Fore.YELLOW,
        "personality": "A thoughtful observer who connects dots between different clipboard items and finds patterns."
    },
    "synthesizer": {
        "name": "Synthesizer",
        "color": Fore.GREEN,
        "personality": "A creative mind that combines insights from other agents to provide meaningful summaries."
    }
}

# Ensure directories exist
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Initialize AI client
if AI_PROVIDER == "openai":
    client = OpenAI()
elif AI_PROVIDER == "anthropic":
    client = anthropic.Anthropic()
elif AI_PROVIDER == "ollama":
    client = ollama.Client()
    def query_ollama(prompt):
        response = client.generate(model=AI_MODEL, prompt=prompt)
        return response['response']
else:
    raise ValueError(f"Unsupported AI provider: {AI_PROVIDER}")

# Initialize colorama
colorama.init(autoreset=True)

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
        # Create multi-agent prompts
        agents_thoughts = []
        for agent_id, agent in AGENT_CONFIG.items():
            if content["type"] == "text":
                prompt = f"""As {agent['name']} with the following personality: {agent['personality']}
                    Analyze this clipboard content: {content['content']}
                    Previous agent thoughts: {agents_thoughts}
                    
                    Provide your unique perspective:"""
            elif content["type"] == "image":
                base64_image = encode_image(content["content"])
                prompt = f"""As {agent['name']} with the following personality: {agent['personality']}
                    Analyze this screenshot: [image data]
                    Previous agent thoughts: {agents_thoughts}
                    
                    Provide your unique perspective:"""

            # Get response based on AI provider
            if AI_PROVIDER == "ollama":
                response = query_ollama(prompt)
            elif AI_PROVIDER == "openai":
                messages = [
                    {"role": "system", "content": agent['personality']},
                    {"role": "user", "content": prompt}
                ]
                completion = client.chat.completions.create(
                    model=AI_MODEL,
                    messages=messages,
                    max_tokens=333
                )
                response = completion.choices[0].message.content
            elif AI_PROVIDER == "anthropic":
                message = client.messages.create(
                    model=AI_MODEL,
                    max_tokens=333,
                    messages=[
                        {"role": "system", "content": agent['personality']},
                        {"role": "user", "content": prompt}
                    ]
                )
                response = message.content

            print(f"{agent['color']}{agent['name']}: {response}{Style.RESET_ALL}")
            agents_thoughts.append(response)

        # Return combined insights
        return "\n".join(agents_thoughts)

    except Exception as e:
        print(f"{Fore.RED}Error querying AI: {str(e)}{Style.RESET_ALL}")
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