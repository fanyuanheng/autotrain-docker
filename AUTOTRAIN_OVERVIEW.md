# AutoTrain Advanced Documentation

## Overview

AutoTrain is a powerful tool created by Hugging Face that automates the process of training and fine-tuning state-of-the-art machine learning models. It's designed as a no-code/low-code solution, enabling users to train powerful models on their own data without writing complex training scripts. The "Advanced" version provides granular control over the entire training process, making it suitable for both beginners and experienced practitioners.

## Core Concepts

AutoTrain Advanced can be broken down into two main components:

1. **Tasks** - What you can do (the "What")
2. **Parameters** - How you control the process (the "How")

---

## Part 1: AutoTrain Tasks

The left-hand sidebar in the AutoTrain interface contains all the different types of models you can train. Here's a comprehensive overview of each task category:

### LLM Fine-tuning

Large Language Model fine-tuning tasks for adapting pre-trained language models to specific needs.

#### LLM SFT (Supervised Fine-Tuning)
- **Purpose**: The most common type of fine-tuning
- **Process**: Provide a dataset of prompts and desired responses (e.g., question-answer pairs, instructions)
- **Outcome**: Model learns to follow your specific format and style
- **Use Case**: Training models on custom datasets like `s32-training-data.csv`

#### LLM ORPO (Odds Ratio Preference Optimization)
- **Purpose**: Newer, efficient fine-tuning technique
- **Process**: Combines instruction tuning and preference alignment in one step
- **Advantage**: Often yields better results with less data and complexity than DPO
- **Use Case**: When you want to optimize for human preferences efficiently

#### LLM Generic
- **Purpose**: General-purpose fine-tuning task
- **Process**: Doesn't assume a specific prompt/response format
- **Use Case**: Flexible training when you don't have a specific format requirement

#### LLM DPO (Direct Preference Optimization)
- **Purpose**: Train models to align with human preferences
- **Process**: Provide data on which of two responses is better (a "chosen" and a "rejected" response)
- **Outcome**: Model becomes more helpful and less prone to generating bad content
- **Use Case**: Creating more aligned and helpful AI assistants

#### LLM Reward
- **Purpose**: Train a model to score responses
- **Process**: The "reward model" learns to predict which responses a human would prefer
- **Use Case**: Core component for advanced alignment techniques like Reinforcement Learning from Human Feedback (RLHF)

### VLM Fine-tuning

Vision-Language Model fine-tuning for models that understand both images and text.

#### VLM Captioning
- **Purpose**: Generate descriptive text captions for images
- **Use Case**: Creating image descriptions, accessibility tools

#### VLM VQA (Visual Question Answering)
- **Purpose**: Answer questions about provided images
- **Use Case**: Interactive image analysis, educational tools

### Sentence Transformers

Models specialized in understanding sentence meaning, excellent for semantic search, clustering, and sentence similarity tasks.

#### ST Pair / ST Triplet / ST Question Answering
- **Purpose**: Different training formats to teach models which sentences are similar or dissimilar
- **Use Cases**: 
  - Semantic search engines
  - Document clustering
  - Duplicate detection
  - Recommendation systems

### Other Text Tasks

Classic Natural Language Processing (NLP) tasks.

#### Text Classification
- **Purpose**: Categorize text into predefined labels
- **Examples**: Spam/not spam, positive/negative sentiment, topic classification

#### Text Regression
- **Purpose**: Predict a continuous numerical value from text
- **Examples**: Predicting star ratings from reviews, sentiment intensity scoring

#### Extractive Question Answering
- **Purpose**: Find and extract exact text spans that answer questions
- **Process**: Given a context paragraph and a question, extract the answer
- **Use Case**: Reading comprehension, information extraction

#### Sequence To Sequence
- **Purpose**: General category for text-to-text transformation
- **Examples**: Translation, summarization, text generation
- **Use Case**: Language translation, document summarization

#### Token Classification
- **Purpose**: Label individual words or "tokens" in a sentence
- **Examples**: Named Entity Recognition (names, locations, organizations)
- **Use Case**: Information extraction, entity recognition

### Image Tasks

Standard Computer Vision (CV) tasks.

#### Image Classification
- **Purpose**: Assign a label to an entire image
- **Examples**: "cat", "dog", "car", "building"
- **Use Case**: Content moderation, automated tagging

#### Image Scoring/Regression
- **Purpose**: Predict a numerical score for an image
- **Examples**: Aesthetic quality score, age estimation
- **Use Case**: Content curation, quality assessment

#### Object Detection
- **Purpose**: Draw bounding boxes around different objects in an image and label them
- **Use Case**: Autonomous vehicles, surveillance systems, retail analytics

### Tabular Tasks

For working with data in spreadsheets or CSV files.

#### Tabular Classification/Regression
- **Purpose**: Predict a category or numerical value based on rows of data with multiple features
- **Examples**: Customer churn prediction, house price prediction
- **Use Case**: Business analytics, financial modeling

---

## Part 2: Training Parameters

The right-hand panel controls the fine-tuning process. Here's a detailed explanation of each parameter category:

### Core Training Setup

#### Base Model
- **Purpose**: The pre-trained model you start with
- **Example**: `meta-llama/Llama-3.1-8B-Instruct`
- **Consideration**: Choose based on your task requirements and available computational resources

#### Distributed Backend
- **Purpose**: Training across multiple GPUs for speed
- **Option**: `ddp` (Distributed Data Parallel)
- **Use Case**: When you have multiple GPUs available

#### Mixed Precision
- **Purpose**: Use 16-bit floating-point numbers instead of 32-bit
- **Option**: `fp16`
- **Benefits**: 
  - Drastically reduces GPU memory usage
  - Speeds up training
  - Minimal loss in precision
- **Recommendation**: Almost always recommended if your GPU supports it

### Optimization & Strategy

#### Optimizer
- **Purpose**: Algorithm used to update model weights
- **Option**: `adamw_torch`
- **Benefits**: Robust, popular, and effective choice for most tasks

#### PEFT/LoRA
- **Purpose**: Parameter-Efficient Fine-Tuning
- **Process**: Instead of training all parameters, LoRA adds small, trainable "adapter" layers
- **Benefits**:
  - Dramatically lowers memory requirements
  - Enables fine-tuning massive models on consumer GPUs
  - Only trains small adapters while freezing the original model

#### Target Modules
- **Option**: `all-linear`
- **Purpose**: Tells LoRA where to add adapter layers
- **Effect**: Applies LoRA to all linear layers in the model

#### Scheduler
- **Purpose**: Controls how learning rate changes during training
- **Option**: `linear`
- **Process**: Gradually decreases learning rate from initial value to zero
- **Benefit**: Helps model converge better

### Training Hyperparameters

#### Epochs
- **Value**: 3
- **Definition**: One full pass through your entire training dataset
- **Guideline**: 3-5 epochs is a common starting point for fine-tuning

#### Batch Size
- **Value**: 2
- **Definition**: Number of data samples processed in one forward/backward pass
- **Consideration**: Larger batch size = more stable training but requires more VRAM

#### Gradient Accumulation
- **Value**: 4
- **Purpose**: Simulate larger batch size without using more memory
- **Process**: Accumulates gradients over 4 "mini-batches" before updating model
- **Effective Batch Size**: Batch Size × Gradient Accumulation = 2 × 4 = 8

#### Learning Rate
- **Value**: 0.00003 (3×10⁻⁵)
- **Purpose**: Step size the optimizer takes
- **Critical Factor**: Too high = unstable training; too low = slow training
- **Guideline**: Common and effective starting point for LoRA fine-tuning

#### Block Size / Model Max Length
- **Values**: 1024 / 2048
- **Purpose**: Maximum number of tokens the model can process in a single sequence
- **Also Known As**: "Context window"
- **Effect**: Ensures model can handle sequences up to 2048 tokens long

---

## Best Practices

### For Beginners
1. Start with SFT for most use cases
2. Use LoRA to reduce memory requirements
3. Begin with 3 epochs and adjust based on results
4. Use mixed precision (fp16) when possible

### For Advanced Users
1. Experiment with different PEFT methods
2. Try DPO for preference alignment
3. Use reward models for RLHF pipelines
4. Optimize hyperparameters based on your specific dataset

### Memory Management
1. Use LoRA to reduce VRAM usage
2. Adjust batch size based on available memory
3. Use gradient accumulation to simulate larger batches
4. Consider using smaller base models if memory is limited

---

## Troubleshooting Common Issues

### Out of Memory Errors
- Reduce batch size
- Enable LoRA
- Use gradient accumulation
- Switch to a smaller base model

### Poor Training Results
- Check your dataset quality
- Adjust learning rate
- Increase training epochs
- Verify data format matches task requirements

### Slow Training
- Enable mixed precision
- Use multiple GPUs if available
- Optimize data loading pipeline
- Consider using a smaller model

---

## Additional Resources

- [AutoTrain Advanced GitHub Repository](https://github.com/huggingface/autotrain-advanced)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [DPO Paper](https://arxiv.org/abs/2305.18290) 