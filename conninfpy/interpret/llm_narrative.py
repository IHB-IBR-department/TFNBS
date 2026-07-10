import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from .evidence import validate_evidence

COGNITIVE_TERMS_TO_GUARD = {
    "memory", "attention", "executive", "motor", "visual", "auditory", "pain", "language",
    "emotion", "social", "reward", "decision", "default", "control", "salience", "somatomotor",
    "limbic", "cognitive", "working", "episodic", "semantic", "spatial", "fear", "anger",
    "sadness", "happiness", "disgust", "empathy", "inhibition", "shifting", "updating",
    "phonological", "syntactic", "lexical", "comprehension", "production", "calculation",
    "numerical", "reasoning", "problem", "solving", "planning", "monitoring", "flexibility",
    "creativity", "intelligence", "learning", "conditioning", "habituation", "sensitization",
    "retrieval", "encoding", "consolidation", "forgetting", "interference", "decay",
    "span", "capacity", "load", "maintenance", "manipulation", "rehearsal", "loop",
    "sketchpad", "buffer", "central", "executive", "focused", "divided", "sustained",
    "selective", "alerting", "orienting", "conflict", "stroop", "flanker", "simon",
    "go", "nogo", "stop", "signal", "delay", "discounting", "risk", "ambiguity",
    "utility", "probability", "value", "choice", "preference", "cooperation", "competition",
    "trust", "fairness", "altruism", "theory", "mind", "mentalizing", "perspective",
    "taking", "mirror", "neuron", "system", "imitation", "action", "perception",
    "face", "body", "scene", "object", "word", "letter", "color", "motion", "shape",
    "texture", "depth", "orientation", "frequency", "pitch", "timbre", "volume",
    "localization", "speech", "voice", "music", "noise", "touch", "temperature",
    "nociception", "itch", "proprioception", "vestibular", "taste", "smell", "olfaction",
    "gustation", "hunger", "thirst", "craving", "fatigue", "sleep", "arousal",
    "vigilance", "circadian", "dreaming", "hypnosis", "meditation", "mindfulness",
    "consciousness", "awareness", "self", "identity", "agency", "body", "ownership",
    "schema", "map", "navigation", "egocentric", "allocentric", "wayfinding", "orientation"
}

SAFE_STRUCTURAL_WORDS = {
    "the", "and", "a", "of", "to", "in", "is", "that", "for", "on", "with", "as", "by", "at", 
    "an", "be", "this", "are", "from", "it", "or", "have", "were", "was", "not", "which",
    "but", "association", "associations", "decodes", "decode", "decoding", "enriched", 
    "spatial", "literature", "neurosynth", "nimare", "evidence", "packet", "roi", "rois", 
    "network", "networks", "edge", "edges", "contrast", "atlas", "schaefer", "yeo", "bna", 
    "cautions", "caution", "cautious", "caveat", "caveats", "mechanistic", "mechanism", 
    "mechanisms", "claim", "claims", "score", "scores", "rank", "ranks", "ranked", 
    "associated", "showed", "showing", "strongest", "reflect", "reflects", "frequency", 
    "frequencies", "rather", "than", "prove", "proves", "proven", "task", "engagement", 
    "activation", "activations", "engaged", "summary", "interpretation", "report", "ready",
    "sentence", "cautious", "publication", "methods", "result", "inconclusive", "weak", 
    "sparse", "contradictory", "generic", "consistent", "standard", "average", "paired", 
    "group", "univariate", "threshold", "thresholding", "significant", "significance",
    "default", "control", "limitation", "limitations", "cognitive", "construct", "constructs",
    "working", "span", "spans", "spanning"
}

def load_dotenv_manually() -> None:
    """Scan directory hierarchy and load variables from .env if present."""
    search_dirs = [
        Path(os.getcwd()),
        Path(os.getcwd()).parent,
        Path(__file__).resolve().parents[2]
    ]
    for d in search_dirs:
        env_path = d / ".env"
        if env_path.exists():
            try:
                with open(env_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            # Strip quotes
                            v_clean = v.strip().strip("'").strip('"')
                            os.environ[k.strip()] = v_clean
                break
            except Exception:
                pass

def check_narrative_terms(narrative: str, evidence: Dict[str, Any]) -> List[str]:
    """Flag cognitive terms generated in the narrative that are not in the raw evidence."""
    words = set(re.findall(r'\b[a-zA-Z]{3,}\b', narrative.lower()))
    
    allow_list = set()
    
    # 1. Decoded terms
    for t in evidence.get("terms", []):
        allow_list.update(re.findall(r'\b[a-zA-Z]{3,}\b', t.get("term", "").lower()))
        allow_list.update(re.findall(r'\b[a-zA-Z]{3,}\b', t.get("roi_name", "").lower()))
        allow_list.update(re.findall(r'\b[a-zA-Z]{3,}\b', t.get("network", "").lower()))
        
    # 2. Network pair names
    for np_pair in evidence.get("query", {}).get("network_pairs", []):
        allow_list.update(re.findall(r'\b[a-zA-Z]{3,}\b', np_pair.get("source", "").lower()))
        allow_list.update(re.findall(r'\b[a-zA-Z]{3,}\b', np_pair.get("target", "").lower()))
        
    # 3. Method configuration
    decoder = evidence.get("decoder", {})
    allow_list.update(re.findall(r'\b[a-zA-Z]{3,}\b', decoder.get("backend", "").lower()))
    allow_list.update(re.findall(r'\b[a-zA-Z]{3,}\b', decoder.get("dataset", "").lower()))
    allow_list.update(re.findall(r'\b[a-zA-Z]{3,}\b', decoder.get("method", "").lower()))
    allow_list.update(re.findall(r'\b[a-zA-Z]{3,}\b', decoder.get("scoring", "").lower()))
    
    # Add structure words
    allow_list.update(SAFE_STRUCTURAL_WORDS)
    
    # 4. Expand common network and anatomical abbreviations to prevent false positives
    expanded = set()
    for w in allow_list:
        w_lower = w.lower()
        if "sommot" in w_lower or "motor" in w_lower:
            expanded.update(["somatomotor", "motor", "somatosensory", "sensory"])
        if "vis" in w_lower:
            expanded.update(["visual", "vision"])
        if "dorattn" in w_lower:
            expanded.update(["dorsal", "attention"])
        if "ventattn" in w_lower or "salventattn" in w_lower:
            expanded.update(["ventral", "attention", "salience"])
        if "frparietal" in w_lower:
            expanded.update(["frontoparietal", "frontal", "parietal"])
        if "aud" in w_lower:
            expanded.update(["auditory", "hearing", "sound"])
        if "lang" in w_lower:
            expanded.update(["language", "linguistic", "speech", "verbal"])
        if "mem" in w_lower or "memory" in w_lower:
            expanded.update(["memory", "working", "recall", "retrieval"])
    allow_list.update(expanded)
    
    flagged = []
    for word in words:
        if word in COGNITIVE_TERMS_TO_GUARD and word not in allow_list:
            flagged.append(word)
            
    return sorted(flagged)

class LLMNarrator:
    """Provider-neutral LLM client interface for generating interpretation text."""
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> None:
        load_dotenv_manually()
        self.provider = provider or os.getenv("CONNINFPY_LLM_PROVIDER", "openai").lower()
        self.model = model or os.getenv("CONNINFPY_LLM_MODEL")
        self.api_key = api_key
        
        if not self.api_key:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "openrouter":
                self.api_key = os.getenv("OPENROUTER_API_KEY")
            elif self.provider in ("google", "gemini"):
                self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        self.last_usage = None
                
    def generate(self, evidence: Dict[str, Any], *, style: str = "default") -> str:
        """Generate cautious interpretation text from the evidence packet."""
        validate_evidence(evidence)
        
        from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
        user_content = USER_PROMPT_TEMPLATE.format(evidence_json=json.dumps(evidence, indent=2))
        
        if self.provider == "mock":
            self.last_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0
            }
            return self._mock_response(evidence)
            
        if self.provider == "openai" or self.provider == "openrouter":
            if not self.api_key:
                raise ValueError(f"{self.provider.upper()}_API_KEY is not set.")
            try:
                import openai
            except ImportError:
                raise ImportError("openai package required. Run: pip install openai")
                
            if self.provider == "openrouter":
                base_url = "https://openrouter.ai/api/v1"
                model = self.model or "meta-llama/llama-3-8b-instruct:free"
                extra_headers = {
                    "HTTP-Referer": "https://github.com/IHB-IBR-department/ConnInfPy",
                    "X-Title": "ConnInfPy Dashboard"
                }
            else:
                base_url = None
                model = self.model or "gpt-4o-mini"
                extra_headers = {}
                
            client = openai.OpenAI(api_key=self.api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                extra_headers=extra_headers
            )
            
            # Extract usage statistics
            try:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                cost = self.estimate_cost(model, prompt_tokens, completion_tokens)
                self.last_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost_usd": cost
                }
            except Exception:
                self.last_usage = None
                
            return response.choices[0].message.content
            
        elif self.provider in ("google", "gemini"):
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
            try:
                from google import genai
            except ImportError:
                raise ImportError("google-genai package required. Run: pip install google-genai")
                
            model = self.model or "gemini-2.5-flash"
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=model,
                contents=user_content,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0.0
                }
            )
            
            # Extract usage statistics
            try:
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
                cost = self.estimate_cost(model, prompt_tokens, completion_tokens)
                self.last_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost_usd": cost
                }
            except Exception:
                self.last_usage = None
                
            return response.text
            
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
    def estimate_cost(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        # Default rates per 1,000 tokens (fallback: approximate at $0.0015 / 1K input, $0.002 / 1K output)
        input_rate = 0.0015 / 1000
        output_rate = 0.002 / 1000
        
        m_lower = model_name.lower() if model_name else ""
        
        # Custom rates for common models (per 1,000 tokens)
        if "gpt-4o-mini" in m_lower:
            input_rate = 0.15 / 1_000_000
            output_rate = 0.60 / 1_000_000
        elif "gpt-4o" in m_lower:
            input_rate = 2.50 / 1_000_000
            output_rate = 10.00 / 1_000_000
        elif "gemini-3.5-flash" in m_lower:
            input_rate = 0.075 / 1_000_000
            output_rate = 0.30 / 1_000_000
        elif "gemini-2.5-flash" in m_lower:
            input_rate = 0.075 / 1_000_000
            output_rate = 0.30 / 1_000_000
        elif "gemini-2.5-pro" in m_lower:
            input_rate = 1.25 / 1_000_000
            output_rate = 5.00 / 1_000_000
        elif "qwen3.7-max" in m_lower:
            input_rate = 1.20 / 1_000_000
            output_rate = 4.80 / 1_000_000
        elif "minimax-m3" in m_lower:
            input_rate = 0.55 / 1_000_000
            output_rate = 2.19 / 1_000_000
        elif "kimi-k2.6" in m_lower:
            input_rate = 0.30 / 1_000_000
            output_rate = 0.90 / 1_000_000
        elif "glm-5.2" in m_lower:
            input_rate = 0.30 / 1_000_000
            output_rate = 0.90 / 1_000_000
        elif "deepseek" in m_lower:
            input_rate = 0.14 / 1_000_000
            output_rate = 0.28 / 1_000_000
        elif "llama-3-8b" in m_lower or "free" in m_lower:
            input_rate = 0.0
            output_rate = 0.0
            
        return (prompt_tokens * input_rate) + (completion_tokens * output_rate)
            
    def _mock_response(self, evidence: Dict[str, Any]) -> str:
        """Generate a deterministic mock narrative for testing and keyless fallback."""
        top_terms = [t['term'] for t in evidence['terms'][:3]]
        terms_str = ", ".join([f"'{t['term']}' (score: {t['score']:.2f})" for t in evidence['terms'][:3]])
        atlas = evidence['query']['atlas']
        caveat = evidence['caveats'][0]
        
        return f"""### Decoding summary
NiMARE decoding of the contrast '{evidence['query']['contrast']}' using the {atlas} atlas identified significant spatial literature associations.

### Main associated terms
The top terms associated with the input regions are: {terms_str}.

### Network-level interpretation
The queried regions span the following networks: {", ".join(sorted(list(set(t['network'] for t in evidence['terms']))))}.

### Cautions
{caveat}

### Report-ready sentence
NiMARE decoding of the selected ConnInfPy regions in the {atlas} atlas showed strongest literature associations with {", ".join([f"'{t}'" for t in top_terms])}. These terms should be interpreted as spatial literature associations rather than mechanistic evidence for the contrast.
"""
