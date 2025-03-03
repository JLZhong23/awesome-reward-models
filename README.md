<div align="center">
<h2>
A Comprehensive Survey of Reward Models:  

Taxonomy, Usages, Evaluation, and Future
</h2>
</div>

<div align="center">
<b>Jialun Zhong</b><sup>1,4∗</sup>,
<b>Wei Shen</b><sup>2∗</sup>,
<b>Yanzeng Li</b><sup>1</sup>,
<b>Songyang Gao</b><sup>2</sup>,
<b>Hua Lu</b><sup>3</sup>,
<b>Yicheng Chen</b><sup>4</sup>,
<br/>
<b>Yang Zhang</b><sup>4</sup>,
<b>Jinjie Gu</b><sup>4</sup>,
<b>Wei Zhou</b><sup>4</sup>,
<b>Lei Zou</b><sup>1†</sup>
</div>

<div align="center">
<sup>1</sup>Peking University
</div>
<div align="center">
<sup>2</sup>Fudan University
</div>
<div align="center">
<sup>3</sup>Huazhong University of Science and Technology
</div>
<div align="center">
<sup>4</sup>Ant Group
</div>

## Paper List

### 🔍 Preference Collection

#### Human Preference

#### AI Preference

* Constitutional AI: Harmlessness from AI Feedback `2022` [[arxiv](https://arxiv.org/pdf/2212.08073)] 
* RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback `2024` [[ICML](https://openreview.net/pdf?id=uydQ2W41KO)]

### 🖥️ Reward Modeling

#### Type-Level

##### Discriminative Reward

##### Generative Reward

* Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena `2023` [[NeurIPS](https://openreview.net/pdf?id=uccHPGDlao)]
* Generative Judge for Evaluating Alignment `2024` [[ICLR](https://openreview.net/pdf?id=gtkFw6sZGS)]
* Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models `2024` [[EMNLP](https://aclanthology.org/2024.emnlp-main.248.pdf)]
* CompassJudger-1: All-in-one Judge Model Helps Model Evaluation and Evolution `2024` [[arxiv](https://arxiv.org/pdf/2410.16256)] 
* OffsetBias: Leveraging Debiased Data for Tuning Evaluators `2024` [[EMNLP Findings](https://aclanthology.org/2024.findings-emnlp.57.pdf)] 
* Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge `2024` [[arxiv](https://arxiv.org/pdf/2407.19594)]
* Self-Taught Evaluators `2024` [[arxiv](https://arxiv.org/pdf/2408.02666)]
* Self-Rewarding Language Models `2024` [[ICML](https://openreview.net/pdf?id=0NphYCmgua)]
* Direct Judgement Preference Optimization `2024` [[arxiv](https://arxiv.org/pdf/2409.14664)]
* Generative Reward Models `2024` [[arxiv](https://arxiv.org/pdf/2410.12832)]
* Generative Verifiers: Reward Modeling as Next-Token Prediction `2024` [[arxiv](https://arxiv.org/pdf/2408.15240)]
* Beyond Scalar Reward Model: Learning Generative Judge from Preference Data `2024` [[arxiv](https://arxiv.org/pdf/2410.03742)]
* Improving Large Language Models via Fine-grained Reinforcement Learning with Minimum Editing Constraint `2024` [[ACL Findings](https://aclanthology.org/2024.findings-acl.338.pdf)]

##### Implicit Reward

* Direct Preference Optimization: Your Language Model is Secretly a Reward Model `2023` [[NeurIPS](https://openreview.net/pdf?id=HPuSIXJaa9)]

#### Granularity-Level

##### Outcome Reward

##### Process Reward

### 📊 Reward Design

#### Point-wise

#### Binary

#### Ensemble

### 🛠️ Usage

#### Utilities



#### Applications

##### Harmless Dialogue

*Dialogue*

* Constitutional AI: Harmlessness from AI Feedback `2022` [[arxiv](https://arxiv.org/pdf/2212.08073)] 
* RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback `2024` [[ICML](https://openreview.net/pdf?id=uydQ2W41KO)]
* Deliberative Alignment: Reasoning Enables Safer Language Models `2024` [[arxiv](https://arxiv.org/pdf/2412.16339)]

##### Logical Reasoning

*Code*

*Math*
* Retrieval-Augmented Process Reward Model for Generalizable Mathematical Reasoning `2025` [[arxiv](https://arxiv.org/pdf/2502.14361)]

*QA*
* 

##### Retrieve & Recommendation

*Retrieve*
* Enhancing Generative Retrieval with Reinforcement Learning from Relevance Feedback `2023` [[EMNLP](https://aclanthology.org/2023.emnlp-main.768.pdf)]
* When Search Engine Services meet Large Language Models: Visions and Challenges `2024` [[arxiv](https://arxiv.org/pdf/2407.00128)]
* Syntriever: How to Train Your Retriever with Synthetic Data from LLMs `2025` [[arxiv](https://arxiv.org/pdf/2502.03824)]
* RAG-Gym: Optimizing Reasoning and Search Agents with Process Supervision `2025` [[arxiv](https://arxiv.org/pdf/2502.13957)]
* DeepRAG: Thinking to Retrieval Step by Step for Large Language Models `2025` [[arxiv](https://arxiv.org/pdf/2502.01142)]

*Recommendation*
* Reinforcement Learning-based Recommender Systems with Large Language Models for State Reward and Action Modeling `2024` [[SIGIR](https://dl.acm.org/doi/pdf/10.1145/3626772.3657767)]
* RLRF4Rec: Reinforcement Learning from Recsys Feedback for Enhanced Recommendation Reranking `2024` [[arxiv](https://arxiv.org/pdf/2410.05939)]
* Fine-Tuning Large Language Model Based Explainable Recommendation with Explainable Quality Reward `2025` [[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28777)]

##### Other Applications

*Music*
* MusicRL: Aligning Music Generation to Human Preferences `2024` [[ICML](https://openreview.net/pdf?id=EruV94XRDs)]

### 💯 Evaluation

#### Benchmarks

* RewardBench: Evaluating Reward Models for Language Modeling `2024` [[arxiv](https://arxiv.org/pdf/2403.13787)] [[Leaderboard](https://hf.co/spaces/allenai/reward-bench)]
* RM-Bench: Benchmarking Reward Models of Language Models with Subtlety and Style `2024` [[arxiv](https://arxiv.org/pdf/2410.16184)]
* RMB: comprehensively benchmarking reward models in LLM alignment `2024` [[arxiv](https://arxiv.org/pdf/2410.09893)]
* VL-RewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models `2024` [[arxiv](https://arxiv.org/pdf/2411.17451)] [[Leaderboard](https://huggingface.co/spaces/MMInstruction/VL-RewardBench)]
* How to Evaluate Reward Models for RLHF `2024` [[arxiv](https://arxiv.org/pdf/2410.14872)] [[Leaderboard](https://huggingface.co/spaces/lmarena-ai/preference-proxy-evaluations)]
* ProcessBench: Identifying Process Errors in Mathematical Reasoning `2024` [[arxiv](https://arxiv.org/pdf/2412.06559)]
* RAG-RewardBench: Benchmarking Reward Models in Retrieval Augmented Generation for Preference Alignment `2024` [[arxiv](https://arxiv.org/pdf/2412.13746)]
* MJ-Bench: Is Your Multimodal Reward Model Really a Good Judge for Text-to-Image Generation? `2024` [[arxiv](https://arxiv.org/pdf/2407.04842)] [[Leaderboard](https://huggingface.co/spaces/MJ-Bench/MJ-Bench-Leaderboard)]
* PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models `2025` [[arxiv](https://arxiv.org/pdf/2501.03124)]

#### Analysis

## Resources

### 🤖 Off-the-Shelf RMs

#### RMs

* (Nemotron) Nemotron-4 340B Technical Report `2024` [[arxiv](https://arxiv.org/pdf/2406.11704)]
* (GRM) Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs `2024` [[NeurIPS](https://openreview.net/pdf?id=jwh9MHEfmY)]
* (Starling-RM) Starling-7B: Improving Helpfulness and Harmlessness with RLAIF `2024` [[CoLM](https://openreview.net/pdf?id=GqDntYTTbk)]
* (Skywork-Reward) Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs `2024` [[arxiv](https://arxiv.org/pdf/2410.18451)]
* (ArmoRM) Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts `2024` [[arxiv](https://arxiv.org/pdf/2406.12845)]

#### PRMs

#### General Models

### 💿 Datasets

* HelpSteer2: Open-source dataset for training top-performing reward models `2024` [[arxiv](https://arxiv.org/pdf/2406.08673)] 

### 🌏 Blogs

* Illustrating Reinforcement Learning from Human Feedback (RLHF) [[Link](https://huggingface.co/blog/rlhf)]
* Why reward models are key for alignment [[Link](https://www.interconnects.ai/p/why-reward-models-matter)]

### 📚 Prior Survey

* A Survey on Interactive Reinforcement Learning: Design Principles and Open Challenges `2021` [[arxiv](https://arxiv.org/pdf/2105.12949)] 
* Reinforcement Learning With Human Advice: A Survey `2021` [[Frontiers Robotics AI](https://doi.org/10.3389/frobt.2021.584075)] 
* AI Alignment: A Comprehensive Survey `2023` [[arxiv](https://arxiv.org/pdf/2310.19852)] 
* A Survey of Reinforcement Learning from Human Feedback `2023` [[arxiv](https://arxiv.org/pdf/2312.14925)] 
* Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback `2023` [[TMLR](https://openreview.net/pdf?id=bx24KpJ4Eb)] 
* Human-in-the-Loop Reinforcement Learning: {A} Survey and Position on Requirements, Challenges, and Opportunities `2024` [[JAIR](https://jair.org/index.php/jair/article/view/15348/27006)]
* Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods `2024` [[arxiv](https://arxiv.org/pdf/2404.00282)]
* A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and More `2024` [[arxiv](https://arxiv.org/pdf/2407.16216)]
* Reinforcement Learning Enhanced LLMs: A Survey `2024` [[arxiv](https://arxiv.org/pdf/2412.10400)]
* Towards a Unified View of Preference Learning for Large Language Models: A Survey `2024` [[arxiv](https://arxiv.org/pdf/2409.02795)]

## Open Questions
