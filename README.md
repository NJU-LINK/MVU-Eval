<div align="center">
  <h1>
    <img src="assets/logo.png" width="200" alt="MVU-Eval Logo" style="vertical-align: middle; margin-bottom: 10px;"><br>
    MVU-Eval:<br>
    Towards Multi-Video Understanding Evaluation for Multimodal LLMs
  </h1>

  <p align="center">
    <a href="https://github.com/NJU-LINK/MVU-Eval"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
    <a href="https://arxiv.org/abs/2510.xxxxx"><img src="https://img.shields.io/badge/arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
    <a href="https://mvu-eval.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge" alt="Project Page"></a>
    <a href="https://huggingface.co/datasets/MVU-Eval-Team/MVU-Eval-Data"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow?style=for-the-badge" alt="Dataset"></a>
  </p>

  <p align="center">
    <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
  </p>
</div>

---

## 📋 Abstract

The advent of Multimodal Large Language Models (MLLMs) has expanded AI capabilities to visual modalities, yet existing benchmarks remain limited to **single-video understanding**.  
To address this gap, we introduce **MVU-Eval**, the **first comprehensive benchmark** for evaluating **multi-video understanding** in MLLMs.

MVU-Eval contains **1,824 carefully curated QA pairs** spanning **4,959 videos** from diverse domains, covering both fundamental perception and high-order reasoning tasks.  
It assesses eight core competencies: *Object Recognition, Spatial Understanding, Counting, Comparison, Knowledge-Intensive Reasoning, In-Context Learning, Retrieval-Augmented Generation,* and *Temporal Reasoning.*

<p align="center">
  <img src="assets/overview.png" width="800" alt="MVU-Eval Overview">
  <br>
  <em>Figure 1: Representative examples in MVU-Eval.</em>
</p>

---

## 🌟 Key Features

- **🎯 First Multi-Video Understanding Benchmark**  
  1,824 QA pairs and 4,959 videos across 8 task categories, bridging perception ↔ reasoning.

- **🧩 Eight Core Competencies**  
  Object Recognition (OR), Spatial Understanding (SU), Counting, Comparison, Knowledge-Intensive Reasoning (KIR), In-Context Learning (ICL), Retrieval-Augmented Generation (RAG), and Temporal Reasoning (TR).

- **⚙️ Rigorous Data Pipeline**  
  Automated QA generation + dual-round human verification + leakage and utility checks ensure quality and fairness.

- **📊 Comprehensive Evaluation**  
  Benchmarked on 30+ open/closed-source MLLMs (e.g., Gemini 2.5 Pro, GPT-4o, Qwen 2.5-VL, InternVL 3), revealing major performance gaps.

---

## 📈 Benchmark Statistics


<p align="center">
  <img src="assets/data_source.png" width="800" alt="MVU-Eval Statistics">
</p>


<p align="center">
  <img src="assets/statistics.png" width="300" alt="MVU-Eval Statistics">
</p>

---

## Experimental Results

<p align="center">
  <img src="assets/results.png" width="800" alt="MVU-Eval Statistics">
</p>


<p align="center">
  <img src="assets/scalinglaw.png" width="800" alt="MVU-Eval Statistics">
</p>

---

## 📦 Data

Please download the dataset from:  
👉 [https://huggingface.co/datasets/MVU-Eval-Team/MVU-Eval-Data](https://huggingface.co/datasets/MVU-Eval-Team/MVU-Eval-Data)

After downloading, extract the data into the `./MVU-Eval-Data/` directory.  
It contains all video clips (`.mp4`) and the corresponding QA annotation files (`.json`).

---

## 💻 Scripts

Below is an example of how to launch the **Qwen/Qwen2.5-VL-3B-Instruct** model using `vLLM`  
and run the inference script for evaluation.


### 1) Start the vLLM Server

```bash
# Start vLLM server (example: Qwen/Qwen2.5-VL-3B-Instruct)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --served-model-name Qwen/Qwen2.5-VL-3B-Instruct \
    --api-key sk-abc123 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --trust-remote-code \
    --dtype auto \
    --gpu-memory-utilization 0.85 \
    --port 8007 \
    --host localhost 
```

**Note:** Adjust --tensor-parallel-size to your GPU count and memory.
If you use another port, update --port in the next step accordingly.

### 2) Run Inference
Then navigate to the inference directory and run the main inference script:
```bash
cd inference

# Replace paths/filenames as needed:
python inference/main.py \
  --model_name Qwen/Qwen2.5-VL-3B-Instruct \
  --port 8007 \
  --data_filename QA_json_file.json \
  --data_root /path/to/MVU-Eval-Data/videos \
  --nframes 32 \
  --max_pixels 720
```

- --data_filename points to a JSON under QA_output/ (e.g., QA_json_file.json).
- --data_root is the root directory containing all videos used in the QA file.
- --nframes (default: 32) is the number of uniformly sampled frames per video.
- --max_pixels (default: 720) is the max side for frame resizing.

After execution, predictions will be saved under:
```
inference/Model_output/max_pixel_{max_pixels}_nframes_{nframes}/{QA_json_file_stem}/main/
```

### 3) Analyze Results
```bash
# Generate per-task and overall accuracy tables/plots from saved predictions
python inference/analyze.py
```

The analysis script will:
- Aggregate results from Model_output/…/*.json
- Compute overall and task-wise accuracy
- Export a markdown table and save comparison plots for reporting

---

## 🪶 Citation

If you find MVU-Eval useful for your research, please cite:

```
@inproceedings{
  peng2025mvueval,
  title={{MVU}-Eval: Towards Multi-Video Understanding Evaluation for Multimodal {LLM}s},
  author={Tianhao Peng and Haochen Wang and Yuanxing Zhang and Zekun Moore Wang and Zili Wang and Ge Zhang and Jian Yang and Shihao Li and Yanghai Wang and Xintao Wang and Houyi Li and Wei Ji and Pengfei Wan and Wenhao Huang and Zhaoxiang Zhang and Jiaheng Liu},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2025},
  url={https://openreview.net/forum?id=UZD5CQV6f9}
}
```
