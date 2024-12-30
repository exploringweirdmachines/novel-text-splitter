from llama_cpp import Llama
import numpy as np
from typing import List


class LlamaChunkSplitter:
    """Sentence splitter using LlamaChunk logprob optimization approach"""
    
    def __init__(
        self,
        # model_path: str = "resources/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        # n_gpu_layers: int = 17,
        model_path: str = "resources/models/Llama-3.2-3B-Instruct-Q8_0.gguf",
        n_gpu_layers: int = 33,
        split_token: str = "段", # means 'section'
        n_ctx: int = 8192,
        seed=42
    ):
        self.prompt_format = """<|start_header_id|>system<|end_header_id|>

Your job is to act as a "Chunker", for use in RAG pipelines. The user will provide a long document.
You, the assistant, should repeat the exact same message verbatim. EXCEPT, you should insert split tokens throughout the passage.

For splits, use the {split_token} character as the separator. Add these splits at natural boundaries in the text, such as:
- Between major sections or topics
- Between sentences that are not tightly coupled
- After important headings or titles
- Between natural topic transitions

You may get a message that is unstructured or not cleanly formatted. Still try to split that input as best as you can.
You should prefer to wait until the end of a natural break point to split, rather than breaking mid-sentence.
Your input could be anything - code, HTML, markdown, etc. You MUST try to output SOME split regardless of the input.<|eot_id|><|start_header_id|>user<|end_header_id|>

Here is an example text:

{split_token}VI Polices and Terms

{split_token}1. INTELLECTUAL PROPERTY COMPLAINTS
{split_token}Amazon respects the intellectual property of others. If you believe that your intellectual property rights are being infringed, please follow our Notice and Procedure for Making Claims of Copyright Infringement.{split_token}

{split_token}2. RISK OF LOSS
{split_token}All purchases of physical items from Amazon are made pursuant to a shipment contract. This means that the risk of loss and title for such items pass to you upon our delivery to the carrier.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{split_token}VI Polices and Terms

{split_token}1. INTELLECTUAL PROPERTY COMPLAINTS
Amazon respects the intellectual property of others.{split_token} If you believe that your intellectual property rights are being infringed, please follow our Notice and Procedure for Making Claims of Copyright Infringement.

{split_token}2. RISK OF LOSS
All purchases of physical items from Amazon are made pursuant to a shipment contract.{split_token} This means that the risk of loss and title for such items pass to you upon our delivery to the carrier.<|eot_id|><|start_header_id|>user<|end_header_id|>

Here is the text to process:

{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{split_token}"""
        # Initialize llama model
        self.llm = Llama(
            model_path=model_path,
            logits_all=True,  # Need logits for all positions
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed
        )
        
        # Get index of split token in vocab
        self.split_token = split_token
        self.split_token_id = None
        for i in range(self.llm.n_vocab()):
            try:
                token = self.llm.detokenize([i]).decode("utf-8")
                if token == split_token:
                    self.split_token_id = i
                    break
            except UnicodeDecodeError:
                continue
                
        if self.split_token_id is None:
            raise ValueError(f"Could not find {split_token} in model vocabulary")

    def format_prompt(self, text: str) -> str:
        """Format input text with prompt template"""
        return self.prompt_format.format(
            split_token=self.split_token,
            input_text=text
        )

    def get_split_logprobs(self, text: str) -> np.ndarray:
        """Get logprobs for split token at each position"""
        # Format prompt and tokenize
        prompt = self.format_prompt(text)
        input_tokens = self.llm.tokenize(prompt.encode("utf-8"), special=True)
        
        # Add one split token to complete the sequence
        split_token_input = self.llm.tokenize(self.split_token.encode("utf-8"), special=True)
        input_tokens.extend(split_token_input)
        
        # Get logprobs by evaluating text with model
        self.llm.eval(input_tokens)
        
        # Get logprobs for split token at each position
        split_logprobs = []
        for i in range(len(input_tokens)):
            logprobs = self.llm.logits_to_logprobs(self.llm.scores[i])
            split_logprobs.append(float(logprobs[self.split_token_id]))
            
        return np.array(split_logprobs)

    def normalize_logprobs(self, logprobs: np.ndarray, window: int = 100) -> np.ndarray:
        """
        Normalize logprobs to account for decaying willpower over distance
        by subtracting local rolling mean
        """
        if len(logprobs) < window:
            return logprobs - np.mean(logprobs)
            
        # Use rolling window convolution for mean
        kernel_size = min(window, len(logprobs))
        kernel = np.ones(kernel_size) / kernel_size
        
        # Calculate rolling mean with same-length output
        rolling_mean = np.convolve(logprobs, kernel, mode='same')
        
        # Normalize
        return logprobs - rolling_mean

    def find_split_points(
        self,
        logprobs: np.ndarray,
        threshold: float = 0.4,
        min_distance: int = 10
    ) -> List[int]:
        """Find indices where splits should occur based on logprobs"""
        peaks = []
        last_peak = -min_distance
        
        for i in range(1, len(logprobs)-1):
            if i - last_peak < min_distance:
                continue
                
            if (logprobs[i] > threshold and
                logprobs[i] > logprobs[i-1] and 
                logprobs[i] > logprobs[i+1]):
                peaks.append(i)
                last_peak = i
                
        return peaks

    def split(
        self,
        text: str,
        threshold: float = 0.4,
        min_distance: int = 10
    ) -> List[str]:
        """Split text using LlamaChunk algorithm"""
        if not text.strip():
            return []
            
        # Get and normalize logprobs
        logprobs = self.get_split_logprobs(text)
        norm_logprobs = self.normalize_logprobs(logprobs)
        
        # Find split points
        split_points = self.find_split_points(
            norm_logprobs,
            threshold=threshold,
            min_distance=min_distance
        )
        
        # No splits found
        if not split_points:
            return [text.strip()] if text.strip() else []
        
        # Split text at split points
        chunks = []
        start = 0
        
        # Convert token indices to character positions using the tokenizer
        char_positions = []
        total_chars = 0
        tokens = self.llm.tokenize(text.encode("utf-8"), special=True)
        for token in tokens:
            try:
                token_text = self.llm.detokenize([token]).decode("utf-8")
                char_positions.append(total_chars)
                total_chars += len(token_text)
            except UnicodeDecodeError:
                char_positions.append(total_chars)
        
        # Split using character positions
        for pos in split_points:
            if pos >= len(char_positions):
                continue
            char_pos = char_positions[pos]
            chunk = text[start:char_pos].strip()
            if chunk:
                chunks.append(chunk)
            start = char_pos
            
        # Add final chunk
        final_chunk = text[start:].strip()
        if final_chunk:
            chunks.append(final_chunk)
            
        return chunks

def create_chunked_text(text: str, chunks: List[str], split_token: str = "段") -> str:
    """Create text with split tokens inserted at chunk boundaries"""
    if not text or not chunks:
        return text
        
    result = ""
    pos = 0
    
    for chunk in chunks:
        chunk_pos = text.find(chunk, pos)
        if chunk_pos == -1:
            continue
            
        if chunk_pos > pos:
            result += text[pos:chunk_pos]
            
        result += chunk + split_token
        pos = chunk_pos + len(chunk)
        
    if pos < len(text):
        result += text[pos:]
        
    return result


def get_text_chunks(text: str, chunks: List[str]) -> List[str]:
    """Convert split text into a list of chunks preserving original text positioning
    
    Args:
        text: Original input text
        chunks: Raw chunks from splitter
        
    Returns:
        List of text chunks that reconstruct the original text
    """
    if not text or not chunks:
        return [text] if text else []
        
    result = []
    pos = 0
    
    for chunk in chunks:
        # Find chunk in original text
        chunk_pos = text.find(chunk, pos)
        if chunk_pos == -1:
            continue
            
        # Add text before chunk if there is any
        if chunk_pos > pos:
            preceding_text = text[pos:chunk_pos].strip()
            if preceding_text:
                result.append(preceding_text)
            
        # Add the chunk itself
        result.append(chunk.strip())
        pos = chunk_pos + len(chunk)
        
    # Add any remaining text
    if pos < len(text):
        remaining = text[pos:].strip()
        if remaining:
            result.append(remaining)
        
    return [chunk for chunk in result if chunk]  # Filter out empty chunks



text = """
In the early night Frodo woke from deep sleep, suddenly, as if some sound or presence had disturbed him. He saw that Strider was sitting alert in his chair: his eyes gleamed in the light of the fire, which had been tended and was burning brightly; but he made no sign or movement.
     Frodo soon went to sleep again; but his dreams were again troubled with the noise of wind and of galloping hoofs. The wind seemed to be curling round the house and shaking it; and far off he heard a horn blowing wildly. He opened his eyes, and heard a cock crowing lustily in the inn-yard. Strider had drawn the curtains and pushed back the shutters with a clang. The first grey light of day was in the room, and a cold air was coming through the open window.
     As soon as Strider had roused them all, he led the way to their bedrooms. When they saw them they were glad that they had taken his advice: the windows had been forced open and were swinging, and the curtains were flapping; the beds were tossed about, and the bolsters slashed and flung upon the floor; the brown mat was torn to pieces.
     Strider immediately went to fetch the landlord. Poor Mr. Butterbur looked sleepy and frightened. He had hardly closed his eyes all night (so he said), but he had never heard a sound.
     'Never has such a thing happened in my time!' he cried, raising his hands in horror. 'Guests unable to sleep in their beds, and good bolsters ruined and all! What are we coming to?'
     'Dark times,' said Strider. 'But for the present you may be left in peace, when you have got rid of us. We will leave at once. Never mind about breakfast: a drink and a bite standing will have to do. We shall be packed in a few minutes.'
     Mr. Butterbur hurried off to see that their ponies were got ready, and to fetch them a 'bite'. But very soon he came back in dismay. The ponies had vanished! The stable-doors had all been opened in the night, and they were gone: not only Merry's ponies, but every other horse and beast in the place.
     Frodo was crushed by the news. How could they hope to reach Rivendell on foot, pursued by mounted enemies? They might as well set out for the Moon. Strider sat silent for a while, looking at the hobbits, as if he was weighing up their strength and courage.
     'Ponies would not help us to escape horsemen,' he said at last, thoughtfully, as if he guessed what Frodo had in mind. 'We should not go much slower on foot, not on the roads that I mean to take. I was going to walk in any case. It is the food and stores that trouble me. We cannot count on getting anything to eat between here and Rivendell, except what we take with us; and we ought to take plenty to spare; for we may be delayed, or forced to go round-about, far out of the direct way. How much are you prepared to carry on your backs?'
     'As much as we must,' said Pippin with a sinking heart, but trying to show that he was tougher than he looked (or felt).
"""

# Initialize the splitter
splitter = LlamaChunkSplitter()

chunks = splitter.split(text)

# Create text with split tokens inserted
chunked_text = create_chunked_text(text, chunks)

print(chunked_text)

result = get_text_chunks(text, chunks)
print(result)
