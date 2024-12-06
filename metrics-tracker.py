import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import json
import os
from datetime import datetime

@dataclass
class AttentionMetrics:
    mean_attention: float
    std_deviation: float
    max_attention: float
    min_attention: float
    attention_distribution: List[List[float]]
    timestamp: str

class MetricsTracker:
    def __init__(self, metrics_file: str):
        self.metrics_file = metrics_file
        self.metrics_history: List[AttentionMetrics] = []
        self.load_metrics()

    def load_metrics(self):
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
                self.metrics_history = [AttentionMetrics(**m) for m in data]

    def save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump([vars(m) for m in self.metrics_history], f, indent=2)

    def calculate_metrics(self, attention_matrix: List[List[float]]) -> AttentionMetrics:
        flat_scores = [score for row in attention_matrix for score in row]
        return AttentionMetrics(
            mean_attention=float(np.mean(flat_scores)),
            std_deviation=float(np.std(flat_scores)),
            max_attention=float(np.max(flat_scores)),
            min_attention=float(np.min(flat_scores)),
            attention_distribution=attention_matrix,
            timestamp=datetime.now().isoformat()
        )

    def add_attention_scores(self, attention_matrix: List[List[float]]):
        metrics = self.calculate_metrics(attention_matrix)
        self.metrics_history.append(metrics)
        self.save_metrics()
        return metrics

    def get_baseline_metrics(self) -> Dict:
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 entries
        return {
            'mean_attention': np.mean([m.mean_attention for m in recent_metrics]),
            'std_deviation': np.mean([m.std_deviation for m in recent_metrics]),
            'max_attention': np.max([m.max_attention for m in recent_metrics]),
            'min_attention': np.min([m.min_attention for m in recent_metrics]),
            'sample_size': len(recent_metrics)
        }

# Integration with existing code
def query_ollama(prompt):
    response = client.generate(model=AI_MODEL, prompt=prompt)
    
    # Extract attention scores from response
    attention_matrix = response.get('attention_scores', [])
    if attention_matrix:
        metrics = metrics_tracker.add_attention_scores(attention_matrix)
        print(f"{Fore.CYAN}Attention Metrics:{Style.RESET_ALL}")
        print(f"Mean: {metrics.mean_attention:.3f}")
        print(f"Std Dev: {metrics.std_deviation:.3f}")
    
    return response['response']

# Initialize tracker
metrics_tracker = MetricsTracker(os.path.join(os.path.dirname(DATASET_FILE), 'attention_metrics.json'))