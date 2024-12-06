import ollama
import networkx as nx
from typing import List, Tuple, Dict
import re
from colorama import init, Fore, Style
import threading
import time
import random

# Initialize colorama
init(autoreset=True)

class Agent:
    def __init__(self, name: str, model: str, color: str):
        self.name = name
        self.model = model
        self.client = ollama.Client()
        self.conversation_history: List[Dict[str, str]] = []
        self.background_thoughts: List[str] = []
        self.color = color

    def generate_response(self, user_input: str, context: str) -> str:
        prompt = f"""As {self.name}, consider the following:

        User Input: {user_input}
        Context: {context}
        Conversation History: {self.conversation_history[-5:]}

        Generate a response that builds upon the conversation and provides unique insights based on your role.
        """
        response = self.client.generate(self.model, prompt=prompt)
        return response['response']

    def background_process(self, other_thoughts: List[str]):
        prompt = f"""As {self.name}, reflect on the recent conversation and other agents' thoughts:
        
        Conversation history: {self.conversation_history[-5:]}
        Other agents' thoughts: {other_thoughts}

        Provide insights or ideas for improving future responses:"""
        
        response = self.client.generate(self.model, prompt=prompt)
        self.background_thoughts.append(response['response'])

class NeuroSymbolicEngine(Agent):
    def __init__(self, model: str):
        super().__init__("NeuroSymbolicEngine", model, Fore.MAGENTA)
        self.knowledge_graph = nx.Graph()

    def extract_symbolic_knowledge(self, text: str) -> List[Tuple[str, str, str]]:
        prompt = f"Extract symbolic knowledge triplets (subject, predicate, object) from the text. Format each triplet as 'subject,predicate,object' on a new line:\n\n{text}\n\nTriplets:"
        response = self.client.generate(self.model, prompt=prompt)
        triplets = []
        for line in response['response'].split('\n'):
            parts = line.split(',')
            if len(parts) == 3:
                triplets.append(tuple(part.strip() for part in parts))
        return triplets

    def update_knowledge_graph(self, triplets: List[Tuple[str, str, str]]):
        for triplet in triplets:
            if len(triplet) == 3:
                subject, predicate, obj = triplet
                self.knowledge_graph.add_edge(subject, obj, relation=predicate)
            else:
                print(f"{Fore.YELLOW}Skipping invalid triplet: {triplet}{Style.RESET_ALL}")

    def symbolic_reasoning(self, query: str) -> List[str]:
        if "path" in query.lower():
            start, end = re.findall(r'\b\w+\b', query)[-2:]
            try:
                path = nx.shortest_path(self.knowledge_graph, start, end)
                return [f"Path found: {' -> '.join(path)}"]
            except nx.NetworkXNoPath:
                return ["No path found"]
            except nx.NodeNotFound:
                return ["One or both nodes not found in the knowledge graph"]
        
        reasoning_prompt = f"Perform symbolic reasoning on the following query:\n\n{query}\n\nReasoning:"
        response = self.client.generate(self.model, prompt=reasoning_prompt)
        return [response['response']]

class ConversationalAgent(Agent):
    def __init__(self, model: str):
        super().__init__("ConversationalAgent", model, Fore.YELLOW)

class OverseerAgent(Agent):
    def __init__(self, model: str):
        super().__init__("OverseerAgent", model, Fore.GREEN)

    def make_decision(self, responses: List[str]) -> str:
        prompt = f"""As the OverseerAgent with inherent authority, review the following responses and make a decision:

        Responses:
        {responses}

        Provide a final decision or synthesis of the responses, along with any additional insights or directions for the conversation.
        """
        response = self.client.generate(self.model, prompt=prompt)
        return response['response']

class NeuroSymbolicConversationalSystem:
    def __init__(self, model: str):
        self.neuro_symbolic_engine = NeuroSymbolicEngine(model)
        self.conversational_agent = ConversationalAgent(model)
        self.overseer_agent = OverseerAgent(model)
        self.agents = [self.neuro_symbolic_engine, self.conversational_agent, self.overseer_agent]
        self.background_thread = None

    def initialize_knowledge(self, text: str):
        triplets = self.neuro_symbolic_engine.extract_symbolic_knowledge(text)
        self.neuro_symbolic_engine.update_knowledge_graph(triplets)
        print(f"{Fore.CYAN}Initialized knowledge graph with {len(triplets)} triplets.{Style.RESET_ALL}")

    def background_discussion(self):
        while True:
            for agent in self.agents:
                other_thoughts = [a.background_thoughts[-1] for a in self.agents if a != agent and a.background_thoughts]
                agent.background_process(other_thoughts)
            time.sleep(5)  # Wait for 5 seconds before the next background discussion

    def interactive_session(self):
        print(f"{Fore.CYAN}Welcome to the enhanced NeuroSymbolic AI session with three agents. Type 'exit' to end the conversation.{Style.RESET_ALL}")
        
        # Start the background discussion thread
        self.background_thread = threading.Thread(target=self.background_discussion, daemon=True)
        self.background_thread.start()

        while True:
            user_input = input(f"{Fore.BLUE}Human: {Style.RESET_ALL}")
            if user_input.lower() == 'exit':
                break

            context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.neuro_symbolic_engine.conversation_history[-5:]])
            
            responses = []
            for agent in self.agents[:2]:  # NeuroSymbolicEngine and ConversationalAgent
                response = agent.generate_response(user_input, context)
                responses.append(response)
                print(f"{agent.color}{agent.name}: {response}{Style.RESET_ALL}")

            overseer_decision = self.overseer_agent.make_decision(responses)
            print(f"{self.overseer_agent.color}{self.overseer_agent.name} (Decision): {overseer_decision}{Style.RESET_ALL}")

            for agent in self.agents:
                agent.conversation_history.append({"role": "human", "content": user_input})
                agent.conversation_history.append({"role": agent.name, "content": response})
            
            self.overseer_agent.conversation_history.append({"role": "OverseerAgent", "content": overseer_decision})

            # Display background thoughts periodically
            if random.random() < 0.3:  # 30% chance to show background thoughts
                print(f"\n{Fore.CYAN}Background Thoughts:{Style.RESET_ALL}")
                for agent in self.agents:
                    if agent.background_thoughts:
                        print(f"{agent.color}{agent.name}: {agent.background_thoughts[-1]}{Style.RESET_ALL}")
                        agent.background_thoughts.clear()

    def run(self):
        initial_knowledge = """
        The sky is blue. Water is composed of hydrogen and oxygen. 
        Trees produce oxygen through photosynthesis. The Earth orbits around the Sun.
        Humans need oxygen to breathe. Plants absorb carbon dioxide and release oxygen.
        The moon orbits around the Earth. Gravity keeps planets in orbit.
        """
        self.initialize_knowledge(initial_knowledge)
        self.interactive_session()

if __name__ == "__main__":
    system = NeuroSymbolicConversationalSystem("llama3.1")
    system.run()