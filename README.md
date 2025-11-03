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
