from collections import Counter
import numpy as np

# Add to PERFORMANCE_CONFIG
PERFORMANCE_CONFIG.update({
    "novelty_weight": 0.3,  # How much to weight novelty (0-1)
    "history_size": 1000,   # How many past responses to track
    "min_similarity": 0.2   # Minimum similarity threshold
})

class NoveltyTracker:
    def __init__(self, history_size=1000):
        self.response_history = deque(maxlen=history_size)
        self.key_frequency = Counter()
        self.value_frequency = Counter()
        self.lock = Lock()

    def add_response(self, response_dict):
        """Track new response patterns"""
        with self.lock:
            for agent_responses in response_dict.values():
                if isinstance(agent_responses, dict) and 'response' in agent_responses:
                    # Extract key-value pairs from response
                    pairs = [line.strip().split(':') for line in 
                            agent_responses['response'].split('\n') 
                            if ':' in line]
                    
                    # Update frequencies
                    for k, v in pairs:
                        k, v = k.strip(), v.strip()
                        self.key_frequency[k] += 1
                        self.value_frequency[v] += 1
            
            self.response_history.append(response_dict)

    def get_novelty_score(self, response_text):
        """Calculate novelty score for a potential response"""
        pairs = [line.strip().split(':') for line in response_text.split('\n') 
                if ':' in line]
        
        if not pairs:
            return 0.0

        # Calculate average frequency of keys and values
        key_scores = []
        value_scores = []
        
        for k, v in pairs:
            k, v = k.strip(), v.strip()
            key_freq = self.key_frequency[k]
            value_freq = self.value_frequency[v]
            
            # Convert frequencies to novelty scores (less frequent = more novel)
            key_scores.append(1.0 / (key_freq + 1))
            value_scores.append(1.0 / (value_freq + 1))

        # Combine scores
        avg_novelty = (np.mean(key_scores) + np.mean(value_scores)) / 2
        return avg_novelty

# Initialize novelty tracker
novelty_tracker = NoveltyTracker(PERFORMANCE_CONFIG["history_size"])

def query_ollama(prompt):
    """Modified to generate multiple candidates and select based on novelty"""
    try:
        # Generate multiple responses
        responses = []
        for _ in range(3):  # Generate 3 candidates
            response = client.generate(model=AI_MODEL, prompt=prompt)
            responses.append(response['response'])

        # Score responses for novelty
        novelty_scores = [novelty_tracker.get_novelty_score(r) for r in responses]
        
        # Combine with base confidence scores (assuming all equal for now)
        base_scores = [1.0] * len(responses)
        
        # Weight novelty vs base confidence
        w = PERFORMANCE_CONFIG["novelty_weight"]
        final_scores = [(1-w)*b + w*n for b, n in zip(base_scores, novelty_scores)]
        
        # Select best response
        best_response = responses[np.argmax(final_scores)]
        return best_response

    except Exception as e:
        print(f"{Fore.RED}Error in query_ollama: {str(e)}{Style.RESET_ALL}")
        return responses[0] if responses else "Error generating response"

def query_ai(content):
    try:
        agents_thoughts = []
        final_response = {}
        
        for agent_id, agent in AGENT_CONFIG.items():
            # ... existing prompt construction ...
            
            response = query_ollama(prompt)
            truncated_response = truncate_content(response, max_chars=512)
            
            print(f"\n{agent['color']}{agent['name']}: {truncated_response}{Style.RESET_ALL}")
            agents_thoughts.append(truncated_response)
            
            final_response[agent_id] = {
                "response": truncated_response,
                "context": {
                    "previous_thoughts": prev_thoughts,
                    "role": agent['personality']
                }
            }
        
        # Track the response patterns
        novelty_tracker.add_response(final_response)
        
        # Save to dataset with structured format
        save_to_dataset(content, final_response)

        return final_response

    except Exception as e:
        print(f"{Fore.RED}Error querying AI: {str(e)}{Style.RESET_ALL}")
        return None