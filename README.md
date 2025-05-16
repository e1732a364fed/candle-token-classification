# Candle Token Classification

This project provides support for using pre-trained token classification models of the BERT lineage (BERT, RoBERTa, DeBERTa, Electra, etc) via [Candle](https://github.com/huggingface/candle)

## Usage

```rust
use candle_token_classification::BertLikeTokenClassificationHead; // Import the token classifier trait from this library
use candle_token_classification::BertTokenClassificationHead; // Import the concrete classifier (BERT are provided)

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ... retrieve Config, VarBuilder & Tokenizer from filesystem, HF hub, etc

    let classifier = BertTokenClassificationHead::load(vb, &config)?;

    // retrieve an ordered list of labels for the chosen classifier
    let labels = config.id2label.values().cloned().collect();

    let output = classifier.classify( // classify some text (or use `classifier.forward` to get the output tensor)
        "This is the text we'd like to classify.",
        &tokenizer,
        &classifier.device
    );

    println!("{:?}", output) // view human-readable output or send for downstream processing, etc

    Ok(())
}
```

## Contributing

If you need other functionality, contributions are welcome :)
