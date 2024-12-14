import json
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import os

class TopicExpander:
    def __init__(self, embedding_model="sentence-transformers/all-mpnet-base-v2"):
        # Load environment variables
        load_dotenv()
        
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Initialize OpenAI client with API key from environment
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize and move to device
                inputs = self.tokenizer(
                    text, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Get model outputs
                outputs = self.model(**inputs)
                
                # Use mean pooling of last hidden states
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding.cpu().numpy())
        
        return np.vstack(embeddings)

    def cluster_topics(self, topics: List[str], n_clusters: int = 5) -> Dict[str, List[str]]:
        """Cluster topics using FAISS"""
        # Get embeddings
        embeddings = self.get_embeddings(topics)
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        d = embeddings.shape[1]
        kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=False)
        kmeans.train(embeddings)
        
        # Get cluster assignments
        _, cluster_ids = kmeans.index.search(embeddings, 1)
        
        # Group topics by cluster
        clusters = {}
        for topic, cluster_id in zip(topics, cluster_ids):
            cluster_id = int(cluster_id[0])
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(topic)
            
        return clusters

    def generate_related_topics(self, topic: str, n: int = 5) -> List[str]:
        """Generate related topics using GPT"""
        prompt = f"""Generate {n} specific topics related to "{topic}". 
        Make them detailed and diverse within the same general category.
        Return only the topics, one per line."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates related topics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        # Parse response and clean up
        topics = response.choices[0].message.content.strip().split('\n')
        topics = [t.strip() for t in topics if t.strip()]
        return topics

    def expand_topic_categories(self, data_path: str, topics_per_category: int = 10):
        """Expand existing topic categories with generated related topics"""
        # Load existing data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        expanded_topics = {}
        
        # Expand each category
        for category, topics in data['topics'].items():
            expanded_topics[category] = set(topics)  # Start with existing topics
            
            # Generate related topics for each existing topic
            for topic in topics:
                related = self.generate_related_topics(topic, n=2)
                expanded_topics[category].update(related)
            
            # Convert set to list and limit size
            expanded_topics[category] = list(expanded_topics[category])[:topics_per_category]
        
        # Update the data and save
        data['topics'] = expanded_topics
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        return expanded_topics

def main():
    # Initialize expander
    expander = TopicExpander()
    
    # Expand topics
    expanded = expander.expand_topic_categories('data/prompt_data.json')
    
    # Cluster and analyze
    all_topics = [topic for topics in expanded.values() for topic in topics]
    clusters = expander.cluster_topics(all_topics)
    
    print("\nTopic Clusters:")
    for cluster_id, topics in clusters.items():
        print(f"\nCluster {cluster_id}:")
        for topic in topics:
            print(f"  - {topic}")

if __name__ == "__main__":
    main() 