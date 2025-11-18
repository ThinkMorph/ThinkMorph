<p align="center">
    <img src="assets/logo.png" width="40%"> <br>
</p>


## Emergent Properties in Multimodal Interleaved Chain-of-Thought Reasoning

<p align="center">
  <a href="https://thinkmorph.github.io/">
    <img
      src="https://img.shields.io/badge/ThinkMorph-Website-0A66C2?logo=safari&logoColor=white"
      alt="ThinkMorph Website"
    />
  </a>
  <a href="https://arxiv.org/abs/2510.27492">
    <img
      src="https://img.shields.io/badge/ThinkMorph-Paper-red?logo=arxiv&logoColor=red"
      alt="ThinkMorph Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/ThinkMorph/ThinkMorph-7B">
    <img 
        src="https://img.shields.io/badge/ThinkMorph-Model-yellow?logo=huggingface&logoColor=yellow" 
        alt="ThinkMorph Model"
    />
  </a>
  <a href="https://huggingface.co/ThinkMorph">
    <img 
        src="https://img.shields.io/badge/ThinkMorph-Dataset-yellow?logo=huggingface&logoColor=yellow" 
        alt="ThinkMorph Dataset"
    />
  </a>
  <a href="https://x.com/Kuvvius/status/1985388865595851135">
    <img
      src="https://img.shields.io/badge/ThinkMorph-Twitter-blue?logo=x&logoColor=black"
      alt="BAGEL Demo"
    />
  </a>
</p>

🌟  This is the official repository which contains the training and inference code for ThinkMorph.

## 💥 News 
- **[2025.10.29]** Our model checkpoint and training data are now accessible at [Huggingface](https://huggingface.co/ThinkMorph).
- **[2025.10.29]** Our paper is now accessible at [arxiv](https://arxiv.org/abs/2510.27492).

## 👀 About ThinkMorph

<p align="center">
    <img src="assets/thinkmorph.jpg" width="100%"> <br>
</p>

We present **ThinkMorph**, a unified model fine-tuned on ∼24K high-quality interleaved reasoning traces across tasks, learning to generate progressive text–image reasoning steps that
concretely manipulate visual content while maintaining coherent verbal logic.

Beyond strong vision-benchmark performance and robust out-of-domain generalization, ThinkMorph demonstrates emergent multimodal intelligence, including novel visual manipulation skills and so on.
These findings suggest promising directions for characterizing the emergent capabilities of unified models for multimodal reasoning.

## 🔥 Quick Start

1️⃣  Set up environment
```bash
git clone https://github.com/ThinkMorph/ThinkMorph.git
cd ThinkMorph
conda create -n thinkmorph python=3.10 -y
conda activate thinkmorph
pip install -r requirements.txt
```

2️⃣  Download checkpoint
```bash
from huggingface_hub import snapshot_download

save_dir = "models/ThinkMorph-7B"
repo_id = "ThinkMorph/ThinkMorph-7B"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)

```

3️⃣ Use `inference.ipynb` to play with ThinkMorph!

## 🔥 Train & Eval

### Training Data prepration

We opensource our training data mentioned in our paper containing four tasks: **Jigsaw Assembly**, **Spatial Navigation**, **Visual Search** , and **Chart Refocus**. Here we show typical examples of four tasks. Training data can be downloaded from [Huggingface](https://huggingface.co/ThinkMorph).



1. **Download the training dataset**

   ```python
    from datasets import load_dataset

    # Jigsaw Assembly
    dataset = load_dataset("ThinkMorph/Jigsaw_Assembly", split="train")

    # Spatial Navigation
    dataset = load_dataset("ThinkMorph/Spatial_Navigation", split="train")

    # Visual Search
    dataset = load_dataset("ThinkMorph/Visual_Search", split="train")

    # Chart Refocus
    dataset = load_dataset("ThinkMorph/Chart_Refocus", split="train")
    ```

2. Convert the downloaded dataset into a data format suitable for model training. For details on the Bagel officially supported data formats, see in [Train](https://github.com/ByteDance-Seed/Bagel/blob/main/TRAIN.md). Based on Bagel's implementation, we modify the training code to support our interleaved data format, and an easy-to-understand example of a parquet file is shown below:

```python
{
    "image_list": [problem_image_0, reasoning_image_0],
    "instruction_list": [question],
    "output_text_list": [f"<think>{resoning_thought_0}</think><image_start>",f"<image_end><think>{resoning_thought_1}</think><answer>{answer}</answer>"],
}
```

3. Edit **`data/dataset_info.py`** with your own data path.

4. Edit **`configs/example.yaml`**. Additionally, we provide example configuration files corresponding to the different training settings in `data/configs`.

---

### Train

```bash
torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=8 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/interleaved_reasoning.yaml \
  --model_path $model_path \
  --layer_module Qwen2MoTDecoderLayer \
  --finetune_from_hf True \
  --auto_resume True \
  --finetune-from-ema True \
  --resume-from $model_path \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --lr 1e-5 \
  --num_worker 4 \
  --max_latent_size 64  \
  --max_num_tokens 32768 \
  --mse_weight 1 \
  --ce_weight 1 \
  --total_steps 8000 \

```

You can replace the variables in the script with your own before running. More training scripts are provided in `./script`. 
See Bagel's [TRAIN](https://github.com/ByteDance-Seed/Bagel/blob/main/TRAIN.md) for more details.

### Eval

All evaluations are conducted using the [`VLMEvalKit`](https://github.com/open-compass/VLMEvalKit) framework for consistency and reproducibility. The inference process can be referred to [infernece.ipynb](inference.ipynb)

## 📊 Benchmarks

| Model | Size |  | VSP | VisPuzzle | ChartQA | VStar | BLINK-J | MMVP | SAT | BLINK | CV-Bench |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-4o | – |  | 33.50 | 43.75 | 76.34 | 61.78 | 72.67 | 84.67 | 28.00 | 60.28 | 75.61 |
| GPT-5 | – |  | 57.33 | 78.00 | 80.85 | 71.73 | 77.33 | 86.33 | 73.30 | 69.86 | 85.46 |
| Gemini 2.5 Flash | – |  | 59.33 | 47.00 | 83.79 | 70.68 | 66.00 | 80.33 | 56.00 | 67.49 | 85.07 |
| InternVL3.5 | 8B |  | 8.17 | 34.75 | 76.26 | 68.59 | 71.33 | 76.33 | 45.33 | 59.60 | 81.99 |
|  | 38B |  | 20.16 | 36.50 | 80.44 | 76.96 | 80.67 | 80.33 | 49.33 | 62.65 | 85.96 |
| Qwen2.5-VL | 7B |  | 2.16 | 34.75 | 78.12 | 76.44 | 59.33 | 77.33 | 51.33 | 55.92 | 75.20 |
|  | 72B |  | 41.83 | 40.00 | 82.03 | 85.86 | 61.33 | 82.00 | 64.67 | 61.91 | 82.54 |
| Janus-pro | 7B |  | 0.00 | 33.50 | 43.08 | 38.22 | 50.67 | 63.33 | 22.00 | 38.51 | 67.83 |
| Chameleon | 7B |  | 0.83 | 30.50 | 5.74 | 28.27 | 0.67 | 47.67 | 10.67 | 16.52 | 36.52 |
| Bagel | 7B |  | 0.83* | 35.00* | 61.82 | 55.49 | 67.33 | 70.33 | 44.67 | 47.66 | 76.03 |
| **ThinkMorph** | **7B** |  | **75.83** | **79.00** | **78.10** | **67.02** | **72.00** | **80.33** | **52.67** | **60.07** | **80.82** |
| Δ (vs Bagel) |  |  | +75.00 | +44.00 | +16.28 | +11.53 | +4.67 | +10.00 | +8.00 | +12.41 | +4.79 |


## ✍️ Citation

```bibtex
@misc{gu2025thinkmorphemergentpropertiesmultimodal,
      title={ThinkMorph: Emergent Properties in Multimodal Interleaved Chain-of-Thought Reasoning}, 
      author={Jiawei Gu and Yunzhuo Hao and Huichen Will Wang and Linjie Li and Michael Qizhe Shieh and Yejin Choi and Ranjay Krishna and Yu Cheng},
      year={2025},
      eprint={2510.27492},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.27492}, 
}
```
