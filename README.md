# ViIR: Information Retrieval Fine-tuning Framework for Vietnamese Documents

Fine-tuning sentence transformer models for Vietnamese information retrieval using any custom dataset.

## Overview

ViIR is a flexible framework for fine-tuning Bi-Encoder and other transformer models for Vietnamese information retrieval tasks. The framework supports three main fine-tuning strategies:

1. **Baseline**: Using pre-trained models without fine-tuning for benchmarking
2. **Positive-pair Tuning**: Fine-tuning with query-document positive pairs
3. **Hard Negative Tuning**: Advanced fine-tuning with hard negatives for improved discrimination

This framework is designed to work with any Vietnamese dataset that contains query-document pairs, making it adaptable for legal documents, news articles, medical information, and more.

## Installation

```bash
# Clone repository
git clone https://github.com/ntphuc149/ViIR.git
cd ViIR

# Install package and dependencies
pip install -e .
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.16+
- Sentence-transformers 2.2.0+
- Scikit-learn 1.0.0+
- Pandas & NumPy
- Tqdm
- PyYAML

## Usage

### 1. Data Preprocessing

The framework expects your dataset to have at least the following columns:
- `question` or `query`: The search query text
- `context` or `document`: The document text
- (Optional) `abstractive_answer`: The ground truth answer

```bash
python scripts/preprocess.py --input /path/to/your_dataset.csv --output data/processed/
```

### 2. Model Training

The framework supports three training strategies that can be easily selected through configuration files:

```bash
# Baseline (no fine-tuning)
python scripts/train.py --config config/baseline.yaml

# Positive-pair Tuning
python scripts/train.py --config config/positive_pair.yaml

# Hard Negative Tuning
python scripts/train.py --config config/hard_negative.yaml
```

You can directly specify model, batch size, learning rate and other parameters via command line:

```bash
# Using PhoBERT model with custom hyperparameters
python scripts/train.py --config config/hard_negative.yaml \
                        --model_name vinai/phobert-base \
                        --batch_size 32 \
                        --learning_rate 2e-5 \
                        --epochs 5
```

### 3. Model Evaluation

Evaluate your trained model with standard IR metrics including NDCG, MRR, precision, and recall:

```bash
python scripts/evaluate.py --model_path output/model/ --data_dir data/processed/ --split test
```

For comprehensive evaluation across all splits:

```bash
python scripts/evaluate.py --model_path output/model/ --data_dir data/processed/ --split all
```

### 4. Running the Complete Pipeline

For convenience, you can run the entire pipeline in one command:

```bash
python main.py --input /path/to/your_dataset.csv --strategy hard_negative
```

Switch between strategies and models:

```bash
# Using baseline strategy with default XLM-RoBERTa
python main.py --input /path/to/your_dataset.csv --strategy baseline

# Using positive-pair strategy with PhoBERT
python main.py --input /path/to/your_dataset.csv \
               --strategy positive_pair \
               --model_name vinai/phobert-base

# Using hard negative strategy with custom hyperparameters
python main.py --input /path/to/your_dataset.csv \
               --strategy hard_negative \
               --model_name vinai/phobert-base-v2 \
               --batch_size 16 \
               --learning_rate 3e-5 \
               --epochs 3
```

## Project Structure

```
ViIR/
├── config/                 # Configuration files for training strategies
├── data/                   # Data processing and dataset creation
├── models/                 # Model definitions
├── trainers/               # Training strategy implementations
├── utils/                  # Utility functions
├── evaluation/             # Model evaluation tools
├── scripts/                # Individual task scripts
└── main.py                 # Main entry point
```

## Customization

### Using Different Models

You can use any model from the Hugging Face hub either by specifying it in the command line or by changing the `model.name` parameter in the configuration files:

#### Command line method:
```bash
python main.py --input your_data.csv --strategy hard_negative --model_name vinai/phobert-base
```

#### Configuration file method:
```yaml
model:
  name: "vinai/phobert-base"  # Or any other Vietnamese language model
  max_seq_length: 512
  trust_remote_code: true
```

### Supported Vietnamese Models

The framework has been tested with the following Vietnamese models:
- `FacebookAI/xlm-roberta-base` (default)
- `FacebookAI/xlm-roberta-large` 
- `vinai/phobert-base-v2`
- `vinai/phobert-large`
- And other models compatible with Sentence Transformers

### Custom Dataset Format

If your dataset has a different format, you can modify the `data/processor.py` file to handle your specific data structure.

## Citation

If you use this framework in your research or applications, please cite:

```
@misc{viir,
  author = {Truong-Phuc Nguyen},
  title = {ViIR: The Unified Framework for Fine-tuning Vietnamese Information Retrieval Models with Various Tuning Strategies},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ntphuc149/ViIR}}
}
```

## License

MIT
