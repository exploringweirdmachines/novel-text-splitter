# LlamaChunkSplitter

LlamaChunkSplitter is a sophisticated text chunking tool that uses LLaMA language models to intelligently split text at natural boundaries. It's particularly useful for RAG (Retrieval-Augmented Generation) pipelines where effective document chunking is crucial for improved retrieval performance.
The LLM does NOT generate any text. This splitter is using logprobs to 'decide' where to split the given text.
## Features

- Intelligent text splitting using LLaMA model probabilities
- Natural boundary detection for various text formats (prose, code, HTML, markdown)
- Customizable split token and thresholds
- GPU acceleration support
- Preservation of original text structure and formatting

## Installation

### Prerequisites

- Python 3.6+
- llama-cpp-python
- numpy

```bash
pip install llama-cpp-python numpy
```

## Usage

### Basic Example

```python
class LlamaChunkSplitter:
    ...

# Initialize the splitter
splitter = LlamaChunkSplitter(
    model_path="path/to/your/llama/model.gguf",
    n_gpu_layers=-1
)

# Split text into chunks
text = "Your long text here..."
chunks = splitter.split(text)

# Get text with split tokens inserted (optional)
chunked_text = create_chunked_text(text, chunks)

# Get clean chunks that preserve original text positioning
result = get_text_chunks(text, chunks)
```

### Detailed Example with Literature Text

This will produce chunks that respect natural narrative boundaries, sentence structure, and dialogue breaks. The splitter will intelligently identify appropriate split points, such as:
- Between major scene transitions
- At paragraph boundaries
- Between dialogue exchanges
- At natural narrative breaks

Here's an example using a passage from "The Lord of the Rings":

```python

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
```

Text with spliting token '段' inserted
```bash
In the early night Frodo woke from deep sleep, suddenly, as if some sound or presence had disturbed him. He saw that Strider段 was sitting alert in his chair: his eyes gle段amed in the light of the fire, which had段 been tended and was burning brightly; but he made no段 sign or movement.
Frodo soon went to sleep again段; but his dreams were again troubled with the noise of wind and段 of galloping hoofs. The wind seemed to be curling round the house and段 shaking it; and far off he heard a horn段 blowing wildly. He opened his eyes, and heard段 a cock crowing lustily in the inn-yard. Str段ider had drawn the curtains and pushed back the shut段ters with a clang. The first grey light of段 day was in the room, and a cold air段 was coming through the open window.
As soon as Strider段 had roused them all, he led the way to段 their bedrooms. When they saw them they were glad段 that they had taken his advice: the windows had been forced open and were swinging,段 and the curtains were flapping; the beds were tossed about段, and the bolsters slashed and flung upon段 the floor; the brown mat was torn to pieces.
Strider immediately went段 to fetch the landlord. Poor Mr. Butterbur段 looked sleepy and frightened. He had hardly closed his eyes all段 night (so he said), but he had never heard a sound.
'Never has such a thing happened in my time!'段 he cried, raising his hands in horror. 'Guests段 unable to sleep in their beds, and good bolsters ruined and all段! What are we coming to?'
'Dark times,'段 said Strider. 'But for the present you段 may be left in peace, when you have got rid of us. We段 will leave at once. Never mind about breakfast: a drink and段 a bite standing will have to do. We shall段 be packed in a few minutes.'
Mr. Butterbur hurried段 off to see that their ponies were got ready段, and to fetch them a 'bite'. But very soon段 he came back in dismay. The ponies had vanished段! The stable-doors had all been opened in the night段, and they were gone: not only Merry's ponies段, but every other horse and beast in the place.
Frodo was crushed by the news.段 How could they hope to reach Rivendell on foot, pursued by mounted enemies段? They might as well set out for the Moon段. Strider sat silent for a while, looking段 at the hobbits, as if he was weighing up their strength and courage.
'Ponies would段 not help us to escape horsemen,' he said at last, thoughtfully段, as if he guessed what Frodo had in mind. '段We should not go much slower on foot, not段 on the roads that I mean to take. I was going to walk in any段 case. It is the food and stores that trouble me. We cannot count on段 getting anything to eat between here and Rivendell, except段 what we take with us; and we ought to段 take plenty to spare; for we may be delayed, or forced to go round-about段, far out of the direct way. How much are you prepared to carry on your段 backs?'
'As much as we must,' said P段ippin with a sinking heart, but trying to show that段 he was tougher than he looked (or felt).段
```
Result

```python
['In the early night Frodo woke from deep sleep, suddenly, as if some sound or presence had disturbed him. He saw that Strider',
'was sitting alert in his chair: his eyes gle',
'amed in the light of the fire, which had',
'been tended and was burning brightly; but he made no',
'sign or movement.\n     Frodo soon went to sleep again',
'; but his dreams were again troubled with the noise of wind and',
'of galloping hoofs. The wind seemed to be curling round the house and',
'shaking it; and far off he heard a horn', 'blowing wildly. He opened his eyes, and heard',
'a cock crowing lustily in the inn-yard. Str',
'ider had drawn the curtains and pushed back the shut',
'ters with a clang. The first grey light of',
'day was in the room, and a cold air',
'was coming through the open window.\n     As soon as Strider',
'had roused them all, he led the way to',
'their bedrooms. When they saw them they were glad',
'that they had taken his advice: the windows had been forced open and were swinging,',
'and the curtains were flapping; the beds were tossed about',
', and the bolsters slashed and flung upon',
'the floor; the brown mat was torn to pieces.\n     Strider immediately went',
'to fetch the landlord. Poor Mr. Butterbur',
'looked sleepy and frightened. He had hardly closed his eyes all', "night (so he said), but he had never heard a sound.\n     'Never has such a thing happened in my time!'", "he cried, raising his hands in horror. 'Guests",
'unable to sleep in their beds, and good bolsters ruined and all', "! What are we coming to?'\n     'Dark times,
'", "said Strider. 'But for the present you",
'may be left in peace, when you have got rid of us. We',
'will leave at once. Never mind about breakfast: a drink and',
'a bite standing will have to do. We shall', "be packed in a few minutes.'\n     Mr. Butterbur hurried",
'off to see that their ponies were got ready',
", and to fetch them a 'bite'. But very soon",
'he came back in dismay. The ponies had vanished',
'! The stable-doors had all been opened in the night',
", and they were gone: not only Merry's ponies",
', but every other horse and beast in the place.\n     Frodo was crushed by the news.',
'How could they hope to reach Rivendell on foot, pursued by mounted enemies',
'? They might as well set out for the Moon',
'. Strider sat silent for a while, looking',
"at the hobbits, as if he was weighing up their strength and courage.\n     'Ponies would",
"not help us to escape horsemen,' he said at last, thoughtfully",
", as if he guessed what Frodo had in mind. '",
'We should not go much slower on foot, not',
'on the roads that I mean to take. I was going to walk in any',
'case. It is the food and stores that trouble me. We cannot count on',
'getting anything to eat between here and Rivendell, except',
'what we take with us; and we ought to',
'take plenty to spare; for we may be delayed, or forced to go round-about',
', far out of the direct way. How much are you prepared to carry on your',
"backs?'\n     'As much as we must,' said P",
'ippin with a sinking heart, but trying to show that',
'he was tougher than he looked (or felt).']
```

The resulting chunks can then be used effectively in a RAG pipeline, maintaining context and readability while staying within token limits.

### Configuration Options

The `LlamaChunkSplitter` class accepts several initialization parameters:

- `model_path`: Path to the LLaMA model file (GGUF format)
- `n_gpu_layers`: Number of layers to offload to GPU (default: 33)
- `split_token`: Character used for splitting (default: "段")
- `n_ctx`: Context window size (default: 8192)
- `seed`: Random seed for reproducibility (default: 42)

### Splitting Parameters

The `split()` method accepts customization parameters:

- `threshold`: Minimum logprob threshold for split points (default: 0.4)
- `min_distance`: Minimum number of tokens between splits (default: 10)

## How It Works

1. **Prompt Engineering**: The system uses a carefully crafted prompt to instruct the LLaMA model about text splitting requirements.

2. **Logprob Analysis**: For each position in the text, the system calculates the probability of inserting a split token.

3. **Normalization**: Logprobs are normalized using a rolling window to account for varying context lengths.

4. **Peak Detection**: The system identifies optimal split points where normalized logprobs exceed the threshold and form local maxima.

5. **Text Reconstruction**: The original text structure is preserved while applying the identified split points.

## Utility Functions

- `create_chunked_text()`: Creates a version of the text with split tokens inserted
- `get_text_chunks()`: Converts split text into a list of chunks while preserving original positioning

## Limitations

- Requires a LLaMA model in GGUF format
- GPU memory usage depends on the number of offloaded layers
- Processing speed depends on model size and available computational resources

## Best Practices

1. **Model Selection**: Choose an appropriate model size based on your computational resources and accuracy requirements.

2. **Threshold Tuning**: Adjust the `threshold` parameter based on your specific use case:
   - Higher values (e.g., 0.5) result in fewer, larger chunks
   - Lower values (e.g., 0.3) create more frequent splits

3. **GPU Optimization**: Adjust `n_gpu_layers` based on your available GPU memory:
   - More layers generally means faster processing
   - Reduce if you encounter memory issues

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]
