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
import colorama
from colorama import Fore, Style
from typing import List, Dict
import ollama
from collections import deque
from threading import Thread, Lock
from queue import Queue
import asyncio
import signal

# Configuration
ASSETS_DIR = os.path.expanduser("~/Desktop/llama-pile/assets")
LOG_FILE = os.path.expanduser("~/Desktop/llama-pile/clipboard_log.txt")
DATASET_FILE = os.path.expanduser("~/Desktop/llama-pile/clipboard_dataset.jsonl")
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
        "personality": "A technical cataloger who maps content to concise key-value pairs.",
        "prompt_template": """Extract key technical elements as concise key-value pairs. Rules:
            - Each key and value should be 1-3 words maximum
            - Focus on technical facts only
            - Format as 'key: value'
            
            Example format:
            lang: python
            main_lib: pyperclip
            data_store: jsonl
            
            Previous pairs: {prev_thoughts}
            
            List the key-value pairs for this content:"""
    },
    "analyst": {
        "name": "Analyst",
        "color": Fore.YELLOW,
        "personality": "A pattern detector who identifies system characteristics in key-value format.",
        "prompt_template": """Map system patterns to concise key-value pairs. Rules:
            - Each key and value should be 1-3 words maximum
            - Focus on patterns and implications
            - Format as 'key: value'
            
            Example format:
            purpose: data collection
            arch_type: event driven
            risk_level: medium
            
            Previous pairs: {prev_thoughts}
            
            List the key-value pairs for this analysis:"""
    },
    "synthesizer": {
        "name": "Synthesizer",
        "color": Fore.GREEN,
        "personality": "An integrator who maps conclusions to actionable key-value pairs.",
        "prompt_template": """Synthesize previous analyses into final key-value pairs. Rules:
            - Each key and value should be 1-3 words maximum
            - Focus on conclusions and actions
            - Format as 'key: value'
            
            Example format:
            main_strength: modular design
            key_weakness: error handling
            next_step: add tests
            
            Previous pairs: {prev_thoughts}
            
            List the key-value pairs for your synthesis:"""
    }
}

# Ensure directories exist
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Initialize AI client
client = ollama.Client()

def query_ollama(prompt):
    response = client.generate(model=AI_MODEL, prompt=prompt)
    return response['response']

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

# Add performance configuration
PERFORMANCE_CONFIG = {
    "batch_size": 5,  # Process multiple clipboard items at once
    "min_content_length": 10,  # Ignore very short clips
    "poll_interval": 0.1,  # Check clipboard more frequently
    "max_queue_size": 100,  # Maximum items in processing queue
    "concurrent_agents": 2,  # Number of agents to run in parallel
}

# Add processing queues and locks
content_queue = Queue(maxsize=PERFORMANCE_CONFIG["max_queue_size"])
result_queue = Queue()
clipboard_lock = Lock()
last_content_hash = None

def get_content_hash(content):
    """Generate hash for content to avoid duplicates"""
    return hashlib.md5(str(content).encode()).hexdigest()

def process_clipboard_content():
    """Modified to handle batch processing"""
    global last_content_hash
    
    with clipboard_lock:
        text_content = pyperclip.paste()
        latest_screenshot = get_latest_screenshot()
        
        # Generate content hash
        current_hash = get_content_hash(text_content if text_content else latest_screenshot)
        
        # Skip if content hasn't changed or is too short
        if (current_hash == last_content_hash or 
            (text_content and len(text_content) < PERFORMANCE_CONFIG["min_content_length"])):
            return None
            
        last_content_hash = current_hash
        
        if text_content:
            return {"type": "text", "content": text_content, "hash": current_hash}
        elif latest_screenshot:
            file_path = save_image(latest_screenshot)
            return {"type": "image", "content": file_path, "hash": current_hash}
    return None

def process_content_batch(batch):
    """Process multiple content items efficiently"""
    results = []
    for content in batch:
        try:
            response = query_ai(content)
            if response:
                results.append((content, response))
                # Save to dataset in background
                Thread(target=save_to_dataset, args=(content, response)).start()
        except Exception as e:
            print(f"{Fore.RED}Error processing content: {str(e)}{Style.RESET_ALL}")
    return results

def content_collector():
    """Continuously collect clipboard content"""
    while True:
        try:
            content = process_clipboard_content()
            if content and not content_queue.full():
                content_queue.put(content)
        except Exception as e:
            print(f"{Fore.RED}Error collecting content: {str(e)}{Style.RESET_ALL}")
        time.sleep(PERFORMANCE_CONFIG["poll_interval"])

def content_processor():
    """Process content from queue in batches"""
    batch = []
    while True:
        try:
            # Collect batch
            while len(batch) < PERFORMANCE_CONFIG["batch_size"] and not content_queue.empty():
                content = content_queue.get_nowait()
                if content:
                    batch.append(content)
            
            # Process batch if not empty
            if batch:
                results = process_content_batch(batch)
                for content, response in results:
                    result_queue.put((content, response))
                batch = []
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.01)
        except Exception as e:
            print(f"{Fore.RED}Error processing batch: {str(e)}{Style.RESET_ALL}")

def query_ai(content):
    try:
        agents_thoughts = []
        final_response = {}
        
        for agent_id, agent in AGENT_CONFIG.items():
            if content["type"] == "text":
                truncated_content = truncate_content(content['content'])
                prev_thoughts = format_previous_thoughts(agents_thoughts)
                
                prompt = agent['prompt_template'].format(
                    prev_thoughts=prev_thoughts
                ) + f"\n\nContent to analyze:\n{truncated_content}"
                
            elif content["type"] == "image":
                base64_image = encode_image(content["content"])
                prev_thoughts = format_previous_thoughts(agents_thoughts)
                
                prompt = agent['prompt_template'].format(
                    prev_thoughts=prev_thoughts
                ) + "\n\nAnalyze this screenshot: [image data]"

            # Add token limit to response
            response = query_ollama(prompt)
            truncated_response = truncate_content(response, max_chars=512)
            
            print(f"\n{agent['color']}{agent['name']}: {truncated_response}{Style.RESET_ALL}")
            agents_thoughts.append(truncated_response)
            
            # Save individual agent response with context
            final_response[agent_id] = {
                "response": truncated_response,
                "context": {
                    "previous_thoughts": prev_thoughts,
                    "role": agent['personality']
                }
            }
            
            # Save to dataset with structured format
            save_to_dataset(content, final_response)

        # Return the structured response
        return final_response

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
        "system_message": SYSTEM_MESSAGE,
        "analysis_metadata": {
            "agent_count": len(AGENT_CONFIG),
            "content_type": content["type"],
            "content_length": len(str(content["content"]))
        }
    }
    with open(DATASET_FILE, "a") as f:
        json.dump(data, f)
        f.write("\n")

def truncate_content(text: str, max_chars: int = 512) -> str:
    """Truncate content to approximately 512 tokens (roughly 2048 characters)"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"

def format_previous_thoughts(thoughts: List[str]) -> str:
    """Format previous thoughts as a compact list of key-value pairs"""
    if not thoughts:
        return "no_previous: true"
    
    formatted = []
    for i, thought in enumerate(thoughts):
        agent_name = list(AGENT_CONFIG.keys())[i]
        # Extract only the key-value lines from the thought
        kv_pairs = [line.strip() for line in thought.split('\n') if ':' in line]
        formatted.append(f"{agent_name}:\n" + "\n".join(kv_pairs))
    
    return "\n".join(formatted)

def main():
    print(f"{Fore.CYAN}Starting optimized clipboard monitor...{Style.RESET_ALL}")
    print(f"Batch size: {PERFORMANCE_CONFIG['batch_size']}")
    print(f"Poll interval: {PERFORMANCE_CONFIG['poll_interval']}s")
    
    # Start collector and processor threads
    collector_thread = Thread(target=content_collector, daemon=True)
    processor_threads = [
        Thread(target=content_processor, daemon=True)
        for _ in range(PERFORMANCE_CONFIG["concurrent_agents"])
    ]
    
    collector_thread.start()
    for thread in processor_threads:
        thread.start()
    
    # Monitor and display results
    try:
        while True:
            try:
                content, response = result_queue.get(timeout=1)
                print(f"\n{Fore.GREEN}Processed content of type: {content['type']}{Style.RESET_ALL}")
                print(f"Queue size: {content_queue.qsize()}")
                
                # Log only the first few characters of content for performance
                log_event("Content Processed", 
                         f"Type: {content['type']}, Hash: {content['hash'][:8]}")
            except:
                continue
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Shutting down...{Style.RESET_ALL}")

if __name__ == "__main__":
    main()