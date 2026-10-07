# V2 Transformer Sentiment Analysis Pipeline (RoBERTa)

This module implements the modern Transformer-based sentiment analysis pipeline for the Sentiment Analyzer NLP project, completely decoupled from the classical V1 pipeline.

---

## 1. Pipeline Overview

- **Dataset**: `cardiffnlp/tweet_eval` (config: `sentiment`)
- **Labels**: 
  - `0` = Negative
  - `1` = Neutral
  - `2` = Positive
- **Pretrained Architecture**: `FacebookAI/roberta-base` (12 layers, 768 hidden dimension, 12 attention heads, 125M parameters)
- **Tokenization**: Byte-Pair Encoding (BPE, vocab size: 50,265) with `MAX_LENGTH = 128`
- **Training Strategy**: Epoch-based validation using Macro F1 as the model-selection metric, with dynamic batch padding via `DataCollatorWithPadding`.

---

## 2. Theoretical & Architectural Foundations

### 1. What RoBERTa's Pretrained Weights Provide
`FacebookAI/roberta-base` was pretrained on over 160 GB of diverse text (BookCorpus, English Wikipedia, CC-News, OpenWebText, Stories) using Masked Language Modeling (MLM).
Through multi-head self-attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
the model learns rich contextual syntactic and semantic word representations, cross-word dependencies, and nuances of grammar, slang, and context across 12 encoder layers before seeing any sentiment labels.

### 2. What the Classification Head Does
The base RoBERTa model outputs a sequence of 768-dimensional hidden vectors:
$$H \in \mathbb{R}^{L \times 768}$$
where the first vector $h_{\text{<s>}} \in \mathbb{R}^{768}$ corresponds to the classification token `<s>` (analogous to BERT's `[CLS]`).
The classification head consists of:
1. A dense projection layer with tanh activation and dropout: $z = \tanh(W_p h_{\text{<s>}} + b_p)$
2. A linear classification layer projecting from 768 dimensions down to the 3 target classes: $\text{logits} = W_c z + b_c \in \mathbb{R}^3$

### 3. What Logits Are
Logits are the raw, unnormalized real-valued outputs produced by the final linear layer:
$$z = [z_0, z_1, z_2] \in \mathbb{R}^3$$
They represent uncalibrated evidence scores for negative ($z_0$), neutral ($z_1$), and positive ($z_2$). They are not probabilities and can range from $-\infty$ to $+\infty$.

### 4. How Softmax Converts Logits into Probabilities
The Softmax function exponentiates the logits and normalizes them across all classes:
$$P(y = i \mid x) = \frac{e^{z_i}}{\sum_{j=0}^{2} e^{z_j}}$$
This guarantees that:
- Every class probability $P(y = i) \in (0, 1)$
- $\sum_{i=0}^{2} P(y = i) = 1.0$
- Relative differences in logits are magnified proportionally to their magnitudes.

### 5. What Cross-Entropy Loss Represents
Cross-entropy measures the divergence between the true one-hot target distribution $y \in \{0, 1\}^3$ and the predicted probability distribution $\hat{y}$:
$$\mathcal{L} = -\sum_{i=0}^{2} y_i \log(\hat{y}_i) = -\log(\hat{y}_{\text{true\_class}})$$
If the model assigns a high probability (near 1.0) to the correct label, $\mathcal{L} \approx 0$. If it assigns a low probability to the correct label, $\mathcal{L} \to \infty$, penalizing confident wrong predictions heavily.

### 6. How Backpropagation Updates Model Weights
During the backward pass:
1. The gradient of the loss with respect to the output logits is computed: $\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y$.
2. The chain rule propagates gradients backwards through the classification head and all 12 transformer encoder layers:
   $$\frac{\partial \mathcal{L}}{\partial W_l} = \frac{\partial \mathcal{L}}{\partial h_l} \frac{\partial h_l}{\partial W_l}$$
3. The AdamW optimizer updates each weight tensor:
   $$W \leftarrow W - \eta \left( \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} + \lambda W \right)$$
   where $\eta$ is the learning rate ($2 \times 10^{-5}$), $\lambda$ is weight decay ($0.01$), and $\hat{m}, \hat{v}$ are bias-corrected first and second moment estimates.

### 7. Why Fine-Tuning is Cheaper Than Pretraining
- **Pretraining**: Requires learning language from scratch across millions of documents over thousands of GPU hours ($100,000+).
- **Fine-Tuning**: Reuses the pretrained 125M parameters that already encode language syntax and semantics. We only need to adjust the weights slightly ($\text{LR} = 2 \times 10^{-5}$) and train the new 3-class classification head on TweetEval (45k tweets), taking only ~10–15 minutes on a modern T4/V100 GPU.

### 8. Why the Test Set Must NOT Be Used for Tuning
- The **Validation Split** (2,000 samples) is used for hyperparameter evaluation, early stopping, and selecting the best checkpoint (`metric_for_best_model="macro_f1"`).
- The **Test Split** (12,284 samples) must remain strictly unseen until final model selection is finalized. Evaluating multiple models on the test set causes *data snooping* and overfits to the test set, creating overly optimistic performance estimates that fail to generalize to real-world user inputs.

### 9. Why Full Training is Executed on Google Colab (GPU) vs Local CPU
- Local Machine: AMD Ryzen 7 5700U CPU (no NVIDIA CUDA).
- On CPU, full 4-epoch training on 45,615 samples requires forward and backward passes across 12 layers of 768-D attention matrices for ~5,700 optimizer steps, taking **4 to 6+ hours of 100% CPU utilization**.
- On Google Colab with an NVIDIA T4 / A100 GPU and Tensor Cores (FP16 mixed precision), the exact same run completes in **~12 minutes**.

---

---

## 3. Google Colab GPU Workflow (Step-by-Step)

Follow these exact steps to train on Google Colab with GPU acceleration and transfer the fine-tuned model back locally:

### Step 1: Enable T4 GPU Runtime
1. Open [Google Colab](https://colab.research.google.com) and create a New Notebook.
2. Navigate to **Runtime** → **Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU** (or A100 / V100 if available) and click **Save**.

### Step 2: Clone or Upload the Repository
In a Colab code cell:
```bash
# Option A: Clone from GitHub (replace with your repo URL)
!git clone https://github.com/<your-username>/<your-repo-name>.git
%cd <your-repo-name>

# Option B: Or if uploaded as a zip:
# !unzip sentiment_analyzer.zip -d sentiment_analyzer
# %cd sentiment_analyzer
```

### Step 3: Install Dependencies
```bash
!pip install -r ml_transformer/requirements.txt
```

### Step 4: Verify CUDA Availability
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Step 5: Run Smoke Test on GPU
Verify pipeline components before launching the full run:
```bash
!PYTHONPATH="." python ml_transformer/src/train.py --smoke-test
```

### Step 6: Launch Full Training
```bash
!PYTHONPATH="." python ml_transformer/src/train.py --full-train
```
*Expected training duration: ~10–14 minutes on T4 GPU (FP16 mixed precision, 4 epochs across 45,615 samples).*

### Step 7: Locate the Trained Model
Confirm that the model files and tokenizer are saved in `ml_transformer/models/roberta-tweeteval/`:
```bash
!ls -lh ml_transformer/models/roberta-tweeteval/
```
Expected files:
- `model.safetensors` (~499 MB)
- `config.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `special_tokens_map.json`
- `tokenizer.json`

### Step 8: Download the Trained Model
Zip and download the fine-tuned directory directly from Colab:
```python
import shutil
from google.colab import files

shutil.make_archive("roberta-tweeteval", "zip", "ml_transformer/models/roberta-tweeteval")
files.download("roberta-tweeteval.zip")
```

### Step 9: Place into Local Project
On your local development machine:
1. Extract `roberta-tweeteval.zip`.
2. Move the extracted folder contents into:
   `ml_transformer/models/roberta-tweeteval/`

### Step 10: Run Standalone Inference Locally
Run inference on your local machine using the fine-tuned weights:
```bash
PYTHONPATH="." python ml_transformer/src/test_model.py
```
The script will detect the fine-tuned checkpoint and output calibrated sentiment probabilities.

---

## 4. Local Execution Commands (Development Machine)

### Run Pipeline Smoke Test (Technical validation without multi-epoch training):
```bash
PYTHONPATH="." python ml_transformer/src/train.py --smoke-test
```

### Run Standalone Inference (Pre-training / Post-training verification):
```bash
PYTHONPATH="." python ml_transformer/src/test_model.py
```
