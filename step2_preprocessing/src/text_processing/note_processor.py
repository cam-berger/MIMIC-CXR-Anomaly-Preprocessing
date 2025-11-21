"""
Clinical note processing with NER, retrieval, and Claude summarization.
Implements the comprehensive text pipeline from Step 2 specification.
"""
import spacy
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from pathlib import Path
import logging
import re
import sys

# Add parent directory to path for base imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from base.processor import TextProcessor

# LangChain and Anthropic imports
try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain/Anthropic not available. Install: pip install langchain langchain-anthropic anthropic")

logger = logging.getLogger(__name__)


class ClinicalNoteProcessor(TextProcessor):
    """
    Process clinical notes using:
    1. Medical NER (scispacy)
    2. Entity-based retrieval
    3. Semantic similarity fallback
    4. Claude summarization
    5. ClinicalBERT tokenization
    """

    def __init__(self, config: Dict, anthropic_api_key: Optional[str] = None):
        super().__init__(config)

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

        # Setup Claude note rewriting (optional preprocessing)
        if config['text'].get('note_rewriting', {}).get('use_claude', False):
            if not LANGCHAIN_AVAILABLE:
                raise ImportError("LangChain required for note rewriting")

            if anthropic_api_key is None:
                logger.warning("No Anthropic API key provided. Note rewriting will be disabled.")
                self.rewriting_chain = None
            else:
                if config['text']['note_rewriting'].get('enabled', False):
                    self._setup_note_rewriting(anthropic_api_key)
                else:
                    self.rewriting_chain = None
        else:
            self.rewriting_chain = None

        # Load ClinicalBERT tokenizer
        logger.info("Loading ClinicalBERT tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['text']['tokenizer']['model']
        )
        logger.info(f"  Loaded: {config['text']['tokenizer']['model']}")

    def validate_config(self) -> None:
        """Validate text processing configuration"""
        # Check required keys
        self.get_config_value('text', 'ner', 'model', required=True)
        self.get_config_value('text', 'tokenizer', 'model', required=True)
        self.get_config_value('text', 'retrieval', 'use_entity_based', required=True)

    def process(self, note_text: str, **kwargs) -> Optional[Dict]:
        """Process clinical note (implements BaseProcessor.process)"""
        try:
            return self.process_note(note_text)
        except Exception as e:
            self._handle_error(e, "processing note")
            return None

    def _setup_claude_summarization(self, api_key: str):
        """Setup LangChain with Claude for summarization"""
        model_config = self.config['text']['summarization']

        # Initialize Claude model with retry and timeout settings
        self.llm = ChatAnthropic(
            model=model_config['model'],
            anthropic_api_key=api_key,
            temperature=model_config['temperature'],
            max_tokens=model_config['max_summary_length'],
            timeout=60,  # 60 second timeout
            max_retries=2  # Retry up to 2 times on failure
        )

        # Create summarization prompt using new ChatPromptTemplate
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

        self.prompt = ChatPromptTemplate.from_template(prompt_template)

        # Create chain using LCEL (pipe operator)
        self.summarization_chain = self.prompt | self.llm
        logger.info("Claude summarization chain initialized")

    def _setup_note_rewriting(self, api_key: str):
        """Setup LangChain with Claude for clinical note rewriting"""
        config = self.config['text']['note_rewriting']

        # Initialize Claude model for rewriting
        self.rewriting_llm = ChatAnthropic(
            model=config['model'],
            anthropic_api_key=api_key,
            temperature=config['temperature'],
            max_tokens=config['max_rewrite_length'],
            timeout=60,
            max_retries=2
        )

        # Create rewriting prompt template (from notebook implementation)
        rewriting_prompt_template = """You are a senior medical documentation assistant. Your job is to rewrite an unstructured clinical note into a complete, well-formatted form. Please preserve all factual details from the original note, without adding any new information. Follow these requirements when rewriting:

1. Expand all abbreviations and shorthand to their full meaning (e.g. convert abbreviations like "c/o" to "complains of", "c/p" to "chest pain", medication names to full generic names, etc.).
2. Normalize/Standardize the format: use complete sentences and a logical clinical narrative. If vital signs or measurements are present, format them consistently with units (e.g. blood pressure as "120/80 mmHg"). Use standard punctuation and grammar.
3. Use a professional clinical tone: write as if this is an official medical record. The tone should be formal and factual, using medical terminology appropriately (for example, use "hyperglycemia" instead of "high blood sugar" if needed).
4. Do NOT omit any information from the original note. Also, do NOT fabricate or infer facts that aren't in the note. If something is unclear or not provided, you may state it is unknown or leave it as is, but do not guess. (It's okay to add clarifying context in phrasing, but only if it's a standard interpretation of the given info.)
5. Maintain all numerical values exactly as written from the original note, do not change the numerical data.

Ensure the final output reads like a polished clinical note.

Clinical Note to Rewrite:
{note}"""

        self.rewriting_prompt = ChatPromptTemplate.from_template(rewriting_prompt_template)

        # Create chain using LCEL (pipe operator)
        self.rewriting_chain = self.rewriting_prompt | self.rewriting_llm
        logger.info("Claude note rewriting chain initialized")

    def rewrite_note(self, note_text: str) -> str:
        """
        Rewrite clinical note using Claude for standardization.
        Expands abbreviations and normalizes format.

        Args:
            note_text: Raw clinical note text

        Returns:
            Rewritten, standardized clinical note text
        """
        if self.rewriting_chain is None:
            logger.debug("Note rewriting not available, using original text")
            return note_text

        if not note_text or len(note_text.strip()) == 0:
            logger.warning("Empty note provided for rewriting")
            return ""

        try:
            # Run rewriting with LCEL API
            logger.debug(f"Rewriting note ({len(note_text)} chars)")
            result = self.rewriting_chain.invoke({"note": note_text})

            # Extract rewritten text (LCEL returns AIMessage object)
            if hasattr(result, 'content'):
                rewritten = result.content
            else:
                rewritten = str(result)

            logger.debug(f"Rewritten note: {len(note_text)} chars -> {len(rewritten)} chars")
            return rewritten.strip()

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            logger.error(f"Note rewriting failed ({error_type}): {error_msg}")
            logger.debug(f"Original note length: {len(note_text)} characters")
            logger.info("Using original note (rewriting fallback)")

            # Fallback to original text
            return note_text

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

        # Step 0: Optional note rewriting (if enabled)
        if self.config['text'].get('note_rewriting', {}).get('enabled', False):
            note_text = self.rewrite_note(note_text)
            logger.debug(f"Using rewritten note ({len(note_text)} chars) for processing")

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
            # en_core_sci_md labels all entities as "ENTITY"
            # Accept all since the model already filters for medical/scientific relevance
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

        if not context_sentences or len(context_sentences) == 0:
            logger.warning("No context sentences provided for summarization")
            return ""

        # Concatenate context
        context = "\n\n".join(context_sentences)

        try:
            # Run summarization with new LCEL API
            result = self.summarization_chain.invoke({"context": context})

            # Extract summary text (LCEL returns AIMessage object)
            if hasattr(result, 'content'):
                summary = result.content
            else:
                summary = str(result)

            logger.debug(f"Generated summary: {len(summary)} characters")
            return summary.strip()

        except Exception as e:
            # Provide detailed error information
            error_type = type(e).__name__
            error_msg = str(e)

            logger.error(f"Claude summarization failed ({error_type}): {error_msg}")
            logger.debug(f"Context length: {len(context)} characters, {len(context_sentences)} sentences")

            # Fallback to simple concatenation
            fallback_summary = " ".join(context_sentences[:3])
            logger.info(f"Using fallback summary ({len(fallback_summary)} chars)")
            return fallback_summary

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
