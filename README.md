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

You can customize hyperparameters by editing the YAML files or by providing command-line overrides:

```bash
python scripts/train.py --config config/hard_negative.yaml --data_dir path/to/data --output_dir path/to/output
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

Switch between strategies:

```bash
python main.py --input /path/to/your_dataset.csv --strategy baseline
python main.py --input /path/to/your_dataset.csv --strategy positive_pair
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

You can use any model from the Hugging Face hub by changing the `model.name` parameter in the configuration files:

```yaml
model:
  name: "vinai/phobert-base" # Or any other Vietnamese language model
  max_seq_length: 256
  trust_remote_code: true
```

### Custom Dataset Format

If your dataset has a different format, you can modify the `data/processor.py` file to handle your specific data structure.

## Citation

If you use this framework in your research or applications, please cite:

```
@misc{viir,
  author = {Truong-Phuc Nguyen},
  title = {ViIR: The unified framework for fine-tuning Vietnamese Information Retrieval models with various tuning strategies},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ntphuc149/ViIR}}
}
```

## License

MIT
