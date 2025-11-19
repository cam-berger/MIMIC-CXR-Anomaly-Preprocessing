"""
Clinical note processing with NER, retrieval, and Claude summarization.
Implements the comprehensive text pipeline from Step 2 specification.
"""
import spacy
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import logging
import re

# LangChain and Anthropic imports
try:
    from langchain_anthropic import ChatAnthropic
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain/Anthropic not available. Install: pip install langchain langchain-anthropic anthropic")

logger = logging.getLogger(__name__)


class ClinicalNoteProcessor:
    """
    Process clinical notes using:
    1. Medical NER (scispacy)
    2. Entity-based retrieval
    3. Semantic similarity fallback
    4. Claude summarization
    5. ClinicalBERT tokenization
    """

    def __init__(self, config: Dict, anthropic_api_key: Optional[str] = None):
        self.config = config

        # Load NER model
        logger.info("Loading scispacy NER model...")
        try:
            self.nlp = spacy.load(config['text']['ner']['model'])
            logger.info(f"  Loaded: {config['text']['ner']['model']}")
        except OSError:
            logger.error(f"scispacy model not found. Install with:")
            logger.error(f"  python -m spacy download {config['text']['ner']['model']}")
            raise

        # Load sentence embedding model for semantic retrieval
        logger.info("Loading sentence embedding model...")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("  Loaded: all-MiniLM-L6-v2")

        # Setup Claude summarization
        if config['text']['summarization']['use_claude']:
            if not LANGCHAIN_AVAILABLE:
                raise ImportError("LangChain required for Claude integration")

            if anthropic_api_key is None:
                logger.warning("No Anthropic API key provided. Summarization will fail.")
                self.summarization_chain = None
            else:
                self._setup_claude_summarization(anthropic_api_key)
        else:
            self.summarization_chain = None

        # Load ClinicalBERT tokenizer
        logger.info("Loading ClinicalBERT tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['text']['tokenizer']['model']
        )
        logger.info(f"  Loaded: {config['text']['tokenizer']['model']}")

    def _setup_claude_summarization(self, api_key: str):
        """Setup LangChain with Claude for summarization"""
        model_config = self.config['text']['summarization']

        # Initialize Claude model
        self.llm = ChatAnthropic(
            model=model_config['model'],
            anthropic_api_key=api_key,
            temperature=model_config['temperature'],
            max_tokens=model_config['max_summary_length']
        )

        # Create summarization prompt
        prompt_template = """You are a medical expert analyzing clinical notes for chest X-ray interpretation.

Your task is to create a concise summary of the clinical note that captures:
- Patient's chief complaint and symptoms
- Relevant medical history (especially cardiopulmonary conditions)
- Physical exam findings related to chest/lungs
- Vital signs and lab abnormalities
- Working diagnosis or clinical concerns

CLINICAL NOTE EXCERPTS:
{context}

Provide a clear, concise summary in 3-5 sentences focusing on information relevant to interpreting the chest X-ray.
Summary:"""

        prompt = PromptTemplate(
            input_variables=["context"],
            template=prompt_template
        )

        self.summarization_chain = LLMChain(llm=self.llm, prompt=prompt)
        logger.info("Claude summarization chain initialized")

    def process_note(self, note_text: str) -> Dict:
        """
        Complete note processing pipeline.

        Args:
            note_text: Raw clinical note text

        Returns:
            Dict with summary, tokens, and metadata
        """
        if not note_text or len(note_text.strip()) == 0:
            return self._empty_note_result()

        # Step 1: Extract medical entities
        entities = self.extract_entities(note_text)

        # Step 2: Retrieve relevant sentences
        relevant_context = self.retrieve_relevant_sentences(note_text, entities)

        # Step 3: Summarize with Claude
        if self.summarization_chain is not None:
            summary = self.summarize_with_claude(relevant_context)
        else:
            # Fallback: use retrieved context as summary
            summary = " ".join(relevant_context[:5])  # First 5 sentences

        # Step 4: Tokenize with ClinicalBERT
        tokens = self.tokenize(summary)

        return {
            'summary': summary,
            'tokens': tokens,
            'num_entities': len(entities),
            'entities': entities[:20],  # Keep first 20 entities
            'context_sentences': len(relevant_context)
        }

    def extract_entities(self, note_text: str) -> List[str]:
        """
        Extract medical entities using scispacy NER.

        Returns:
            List of unique medical entity texts
        """
        doc = self.nlp(note_text)

        entities = []
        for ent in doc.ents:
            # Filter for medically relevant entity types
            if ent.label_ in ['DISEASE', 'SYMPTOM', 'TREATMENT', 'TEST', 'ANATOMY']:
                entities.append(ent.text.lower())

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for ent in entities:
            if ent not in seen:
                seen.add(ent)
                unique_entities.append(ent)

        logger.debug(f"Extracted {len(unique_entities)} unique entities")
        return unique_entities

    def retrieve_relevant_sentences(
        self,
        note_text: str,
        entities: List[str]
    ) -> List[str]:
        """
        Retrieve relevant sentences using entity-based and semantic methods.

        Implements hybrid retrieval as specified in Step 2.
        """
        # Split into sentences
        sentences = self._split_sentences(note_text)

        if len(sentences) == 0:
            return []

        config = self.config['text']['retrieval']
        max_sentences = config['max_sentences']

        # Method 1: Entity-based retrieval
        entity_sentences = []
        if config['use_entity_based'] and len(entities) > 0:
            entity_sentences = self._entity_based_retrieval(sentences, entities)

        # Method 2: Semantic similarity fallback
        semantic_sentences = []
        if config['use_semantic_fallback']:
            semantic_sentences = self._semantic_retrieval(
                sentences,
                threshold=config['similarity_threshold']
            )

        # Combine both methods (union)
        all_retrieved = list(set(entity_sentences + semantic_sentences))

        # Limit to max sentences
        return all_retrieved[:max_sentences]

    def _entity_based_retrieval(
        self,
        sentences: List[str],
        entities: List[str]
    ) -> List[str]:
        """Retrieve sentences containing medical entities"""
        relevant = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check if sentence contains any entity
            for entity in entities:
                if entity in sentence_lower:
                    relevant.append(sentence)
                    break  # One match is enough

        logger.debug(f"Entity-based retrieval: {len(relevant)} sentences")
        return relevant

    def _semantic_retrieval(
        self,
        sentences: List[str],
        threshold: float = 0.3
    ) -> List[str]:
        """
        Retrieve sentences based on semantic similarity to medical keywords.

        Uses sentence embeddings to find clinically relevant content.
        """
        if len(sentences) == 0:
            return []

        # Define query representing relevant medical information
        query = "chest pain shortness of breath fever cough medical history exam findings vital signs diagnosis"

        # Embed query and sentences
        query_embedding = self.embedder.encode([query])[0]
        sentence_embeddings = self.embedder.encode(sentences)

        # Calculate cosine similarities
        similarities = np.dot(sentence_embeddings, query_embedding) / (
            np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Filter by threshold and sort by similarity
        relevant_indices = np.where(similarities >= threshold)[0]
        sorted_indices = relevant_indices[np.argsort(similarities[relevant_indices])[::-1]]

        relevant = [sentences[i] for i in sorted_indices]

        logger.debug(f"Semantic retrieval: {len(relevant)} sentences")
        return relevant

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Use regex to split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter out very short sentences (likely not meaningful)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        return sentences

    def summarize_with_claude(self, context_sentences: List[str]) -> str:
        """
        Generate summary using Claude via LangChain.

        Args:
            context_sentences: Retrieved relevant sentences

        Returns:
            Summarized text
        """
        if self.summarization_chain is None:
            logger.warning("Claude summarization not available")
            return " ".join(context_sentences[:3])

        # Concatenate context
        context = "\n\n".join(context_sentences)

        try:
            # Run summarization
            result = self.summarization_chain.invoke({"context": context})

            # Extract summary text (LangChain returns dict)
            if isinstance(result, dict):
                summary = result.get('text', '')
            else:
                summary = str(result)

            logger.debug(f"Generated summary: {len(summary)} characters")
            return summary.strip()

        except Exception as e:
            logger.error(f"Error in Claude summarization: {e}")
            # Fallback to concatenation
            return " ".join(context_sentences[:3])

    def tokenize(self, text: str) -> Dict:
        """
        Tokenize text using ClinicalBERT tokenizer.

        Returns:
            Dict with input_ids, attention_mask, and metadata
        """
        config = self.config['text']['tokenizer']

        encoded = self.tokenizer(
            text,
            max_length=config['max_length'],
            padding=config['padding'],
            truncation=config['truncation'],
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'][0],  # Remove batch dimension
            'attention_mask': encoded['attention_mask'][0],
            'num_tokens': int(encoded['attention_mask'][0].sum()),
            'is_truncated': len(self.tokenizer.tokenize(text)) > config['max_length']
        }

    def _empty_note_result(self) -> Dict:
        """Return empty result for missing/invalid notes"""
        return {
            'summary': "",
            'tokens': {
                'input_ids': self.tokenizer("", return_tensors='pt')['input_ids'][0],
                'attention_mask': self.tokenizer("", return_tensors='pt')['attention_mask'][0],
                'num_tokens': 0,
                'is_truncated': False
            },
            'num_entities': 0,
            'entities': [],
            'context_sentences': 0
        }
