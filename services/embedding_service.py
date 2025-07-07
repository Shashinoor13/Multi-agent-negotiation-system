import faiss
import numpy as np
from .llm_service import LLMService

def save_embeddings(agent_card_generators):
    """
    Takes a list of agent_card generator functions, generates embeddings for each using LLMService,
    stacks them, and saves them in a Faiss index.
    Returns the Faiss index.
    """
    llm = LLMService()
    embeddings = []
    for generator in agent_card_generators:
        card = generator()
        # Convert card to text for embedding
        text_parts = []
        text_parts.append(card.get('name', ''))
        text_parts.append(card.get('description', ''))
        capabilities = card.get('capabilities', [])
        if capabilities:
            text_parts.append('Capabilities: ' + ', '.join(capabilities))
        tools = card.get('tools', [])
        if tools:
            text_parts.append('Tools: ' + ', '.join(tools))
        agent_text = ' '.join(text_parts)
        
        embedding = llm.generate_embeddings(agent_text)
        embeddings.append(embedding)
    all_embeddings = np.vstack(embeddings)
    index = faiss.IndexFlatL2(all_embeddings.shape[1])
    index.add(all_embeddings)
    return index
