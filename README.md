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

* Deep Reinforcement Learning from Human Preferences `2017` [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2017/file/d5e2c0adad503c91f91df240d0cd4e49-Paper.pdf)]
* Batch Active Preference-Based Learning of Reward Functions `2018` [[CoRL](https://proceedings.mlr.press/v87/biyik18a/biyik18a.pdf)]
* Reward learning from human preferences and demonstrations in Atari `2018` [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2018/file/8cbe9ce23f42628c98f80fa0fac8b19a-Paper.pdf)]
* Active Preference-Based Gaussian Process Regression for Reward Learning `2020` [[RSS](https://www.roboticsproceedings.org/rss16/p041.pdf)]
* Information Directed Reward Learning for Reinforcement Learning `2021` [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2021/file/1fa6269f58898f0e809575c9a48747ef-Paper.pdf)]
* PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience and Unsupervised Pre-training `2021` [[ICML](https://proceedings.mlr.press/v139/lee21i/lee21i.pdf)]
* Improving alignment of dialogue agents via targeted human judgements `2022` [[arxiv](https://arxiv.org/pdf/2209.14375)]
* Training language models to follow instructions with human feedback `2022` [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf)]
* Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback `2022` [[arxiv](https://arxiv.org/pdf/2204.05862)] 
* Active Reward Learning from Multiple Teachers `2023` [[AAAI Workshop](https://ceur-ws.org/Vol-3381/48.pdf)] 
* RLHF-Blender: A Configurable Interactive Interface for Learning from Diverse Human Feedback `2023` [[ICML Workshop](https://openreview.net/pdf?id=JvkZtzJBFQ)] 
* Sequential Preference Ranking for Efficient Reinforcement Learning from Human Feedback `2023` [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/99766cda865be123d55a1d9666c7b9fc-Paper-Conference.pdf)]
* Uni-RLHF: Universal Platform and Benchmark Suite for Reinforcement Learning with Diverse Human Feedback `2024` [[ICLR](https://openreview.net/pdf?id=WesY0H9ghM)] 
* HelpSteer2: Open-source dataset for training top-performing reward models `2024` [[arxiv](https://arxiv.org/pdf/2406.08673)] 
* Batch Active Learning of Reward Functions from Human Preferences `2024` [[arxiv](https://arxiv.org/pdf/2402.15757)] 
* RLHF Workflow: From Reward Modeling to Online RLHF `2024` [[TMLR](https://openreview.net/pdf?id=a13aYUU9eU)] 

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
* LLM Critics Help Catch {LLM} Bugs `2024` [[arxiv](https://arxiv.org/pdf/2407.00215)] 
* LLM Critics Help Catch Bugs in Mathematics: Towards a Better Mathematical Verifier with Natural Language Feedback `2024` [[arxiv](https://arxiv.org/pdf/2406.14024)]
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


### 🦾 Workflow

#### Prompt Engineering

#### Trajectory Sampling / Data Synthetic 

#### Policy Training

#### Inference

### 📊 Ensemble

#### Reward Ensemble

#### Model Fusion

### 🛠️ Applications

#### Harmless Dialogue

*Dialogue*

* Safe RLHF: Safe Reinforcement Learning from Human Feedback
* Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback `2022` [[arxiv](https://arxiv.org/pdf/2204.05862)] 
* Constitutional AI: Harmlessness from AI Feedback `2022` [[arxiv](https://arxiv.org/pdf/2212.08073)] 
* Zhongjing: Enhancing the Chinese Medical Capabilities of Large Language Model through Expert Feedback and Real-world Multi-turn Dialogue `2023` [[arxiv](https://arxiv.org/pdf/2308.03549)]
* HuatuoGPT, Towards Taming Language Models To Be a Doctor `2023` [[EMNLP Findings](https://aclanthology.org/2023.findings-emnlp.725.pdf)]
* Empathy Level Alignment via Reinforcement Learning for Empathetic Response Generation `2024` [[arxiv](https://arxiv.org/pdf/2408.02976)]
* RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback `2024` [[ICML](https://openreview.net/pdf?id=uydQ2W41KO)]
* Deliberative Alignment: Reasoning Enables Safer Language Models `2024` [[arxiv](https://arxiv.org/pdf/2412.16339)]
* Training Dialogue Systems by AI Feedback for Improving Overall Dialogue Impression `2025` [[arxiv](https://arxiv.org/pdf/2501.12698)]

#### Logical Reasoning

*Math*
* Training Verifiers to Solve Math Word Problems `2022` [[arxiv](https://arxiv.org/pdf/2110.14168)]
* Solving math word problems with process- and outcome-based feedback `2022` [[arxiv](https://arxiv.org/pdf/2211.14275)]
* WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct `2023` [[arxiv](https://arxiv.org/pdf/2308.09583)]
* Let's Verify Step by Step `2024` [[arxiv](https://openreview.net/pdf?id=v8L0pN6EOi)]
* DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models `2024` [[arxiv](https://arxiv.org/pdf/2402.03300)]
* Improve Mathematical Reasoning in Language Models by Automated Process Supervision `2024` [[arxiv](https://arxiv.org/pdf/2406.06592)]
* Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations `2024` [[ACL](https://aclanthology.org/2024.acl-long.510.pdf)]
* The Lessons of Developing Process Reward Models in Mathematical Reasoning `2025` [[arxiv](https://arxiv.org/pdf/2501.07301)]
* Retrieval-Augmented Process Reward Model for Generalizable Mathematical Reasoning `2025` [[arxiv](https://arxiv.org/pdf/2502.14361)]

*Code*
* Applying RLAIF for Code Generation with API-usage in Lightweight LLMs `2024` [[arxiv](https://arxiv.org/pdf/2406.20060)]
* Process Supervision-Guided Policy Optimization for Code Generation `2024` [[arxiv](https://arxiv.org/pdf/2410.17621)]
* Performance-Aligned LLMs for Generating Fast Code `2024` [[arxiv](https://arxiv.org/pdf/2404.18864)]
* Policy Filtration in RLHF to Fine-Tune LLM for Code Generation `2024` [[arxiv](https://arxiv.org/pdf/2409.06957)]
* LLM Critics Help Catch LLM Bugs `2024` [[arxiv](https://arxiv.org/pdf/2407.00215)]

#### Retrieve & Recommendation

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

#### Other Applications

*Text to Audio*
* MusicRL: Aligning Music Generation to Human Preferences `2024` [[ICML](https://openreview.net/pdf?id=EruV94XRDs)]
* BATON: Aligning Text-to-Audio Model Using Human Preference Feedback `2024` [[IJCAI](https://www.ijcai.org/proceedings/2024/0502.pdf)]
* Reinforcement Learning for Fine-tuning Text-to-speech Diffusion Models `2024` [[arxiv](https://arxiv.org/pdf/2405.14632)]

*Text to Image*
* Aligning Text-to-Image Models using Human Feedback `2023` [[arxiv](https://arxiv.org/pdf/2302.12192)]
* ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation `2023` [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/33646ef0ed554145eab65f6250fab0c9-Paper-Conference.pdf)]
* DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models] `2023` [[arxiv](https://arxiv.org/pdf/2305.16381)]

*Text to Video*
* InstructVideo: Instructing Video Diffusion Models with Human Feedback `2024` [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Yuan_InstructVideo_Instructing_Video_Diffusion_Models_with_Human_Feedback_CVPR_2024_paper.pdf)]
* Boosting Text-to-Video Generative Model with MLLMs Feedback `2024` [[NeurIPS](https://openreview.net/pdf/4c9eebaad669788792e0a010be4031be5bdc426e.pdf)]
* Harness Local Rewards for Global Benefits: Effective Text-to-Video Generation Alignment with Patch-level Reward Models `2025` [[arxiv](https://arxiv.org/pdf/2502.06812)]

*Robotic*
* (x) Text2Reward: Reward Shaping with Language Models for Reinforcement Learning `2024` [[ICLR](https://openreview.net/pdf?id=tUM39YTRxH)]
* Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning `2024` [[ICLR](https://openreview.net/pdf?id=N0I2RtD8je)]

*Game*
* DIP-RL: Demonstration-Inferred Preference Learning in Minecraft `2025` [[arxiv](https://arxiv.org/pdf/2307.12158)]
* Process Reward Models for LLM Agents: Practical Framework and Directions `2025` [[arxiv](https://arxiv.org/pdf/2502.10325)]


### 💯 Evaluation

#### Benchmarks

* RewardBench: Evaluating Reward Models for Language Modeling `2024` [[arxiv](https://arxiv.org/pdf/2403.13787)] [[Leaderboard](https://hf.co/spaces/allenai/reward-bench)]
* RM-Bench: Benchmarking Reward Models of Language Models with Subtlety and Style `2024` [[arxiv](https://arxiv.org/pdf/2410.16184)]
* RMB: comprehensively benchmarking reward models in LLM alignment `2024` [[arxiv](https://arxiv.org/pdf/2410.09893)]
* VL-RewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models `2024` [[arxiv](https://arxiv.org/pdf/2411.17451)] [[Leaderboard](https://huggingface.co/spaces/MMInstruction/VL-RewardBench)]
* How to Evaluate Reward Models for RLHF `2024` [[arxiv](https://arxiv.org/pdf/2410.14872)] [[Leaderboard](https://huggingface.co/spaces/lmarena-ai/preference-proxy-evaluations)]
* ProcessBench: Identifying Process Errors in Mathematical Reasoning `2024` [[arxiv](https://arxiv.org/pdf/2412.06559)]
* RAG-RewardBench: Benchmarking Reward Models in Retrieval Augmented Generation for Preference Alignment `2024` [[arxiv](https://arxiv.org/pdf/2412.13746)]
* M-RewardBench: Evaluating Reward Models in Multilingual Settings `2024` [[arxiv](https://arxiv.org/abs/2410.15522)]
* MJ-Bench: Is Your Multimodal Reward Model Really a Good Judge for Text-to-Image Generation? `2024` [[arxiv](https://arxiv.org/pdf/2407.04842)] [[Leaderboard](https://huggingface.co/spaces/MJ-Bench/MJ-Bench-Leaderboard)]
* PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models `2025` [[arxiv](https://arxiv.org/pdf/2501.03124)]
* Multimodal RewardBench: Holistic Evaluation of Reward Models for Vision Language Models `2025` [[arxiv](https://arxiv.org/abs/2502.14191)]

#### Analysis

The Accuracy Paradox in RLHF: When Better Reward Models Don’t Yield Better Language Models

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
