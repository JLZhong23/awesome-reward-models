<div align="center">
<h2>
A Comprehensive Survey of Reward Models: 
  
Taxonomy, Applications, Challenges, and Future
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
<div align="center">
[<a href="https://arxiv.org/abs/2504.12328">arxiv</a>]
</div>
<br/>

😄 Welcome to recommend missing papers through **`Issues`** and **`Pull Requests`**. 

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
* SURF: Semi-supervised Reward Learning with Data Augmentation for Feedback-efficient Preference-based Reinforcement Learning `2022` [[ICLR](https://openreview.net/pdf?id=TfhfZLQ2EJO)]
* Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback `2022` [[arxiv](https://arxiv.org/pdf/2204.05862)] 
* Active Reward Learning from Multiple Teachers `2023` [[AAAI Workshop](https://ceur-ws.org/Vol-3381/48.pdf)] 
* RLHF-Blender: A Configurable Interactive Interface for Learning from Diverse Human Feedback `2023` [[ICML Workshop](https://openreview.net/pdf?id=JvkZtzJBFQ)] 
* Sequential Preference Ranking for Efficient Reinforcement Learning from Human Feedback `2023` [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/99766cda865be123d55a1d9666c7b9fc-Paper-Conference.pdf)]
* Fine-Grained Human Feedback Gives Better Rewards for Language Model Training `2023` [[NeurIPS](https://papers.neurips.cc/paper_files/paper/2023/file/b8c90b65739ae8417e61eadb521f63d5-Paper-Conference.pdf)]
* Uni-RLHF: Universal Platform and Benchmark Suite for Reinforcement Learning with Diverse Human Feedback `2024` [[ICLR](https://openreview.net/pdf?id=WesY0H9ghM)] 
* HelpSteer2: Open-source dataset for training top-performing reward models `2024` [[arxiv](https://arxiv.org/pdf/2406.08673)] 
* Batch Active Learning of Reward Functions from Human Preferences `2024` [[arxiv](https://arxiv.org/pdf/2402.15757)] 
* Towards Comprehensive Preference Data Collection for Reward Modeling `2024` [[arxiv](https://arxiv.org/pdf/2406.16486)]
* RLHF Workflow: From Reward Modeling to Online RLHF `2024` [[TMLR](https://openreview.net/pdf?id=a13aYUU9eU)] 
* Towards Comprehensive Preference Data Collection for Reward Modeling `2024` [[arxiv](https://arxiv.org/pdf/2406.16486)]
* Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data `2024` [[arxiv](https://openreview.net/pdf?id=bWNPx6t0sF)]
* Less is More: Improving LLM Alignment via Preference Data Selection `2025` [[arxiv](https://arxiv.org/pdf/2502.14560)]

#### AI Preference

* Constitutional AI: Harmlessness from AI Feedback `2022` [[arxiv](https://arxiv.org/pdf/2212.08073)] 
* AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback `2023` [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/5fc47800ee5b30b8777fdd30abcaaf3b-Paper-Conference.pdf)]
* Aligning Large Language Models through Synthetic Feedback `2023` [[EMNLP](https://aclanthology.org/2023.emnlp-main.844.pdf)]
* RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback `2024` [[ICML](https://openreview.net/pdf?id=uydQ2W41KO)]
* UltraFeedback: Boosting Language Models with Scaled AI Feedback `2024` [[ICML](https://openreview.net/forum?id=BOorDpKHiJ)]
* SALMON: Self-Alignment with Instructable Reward Models `2024` [[ICLR](https://openreview.net/pdf?id=xJbsmB8UMx)]
* Improving Reward Models with Synthetic Critiques `2024` [[arxiv](https://arxiv.org/pdf/2405.20850)]
* Self-Generated Critiques Boost Reward Modeling for Language Models `2024` [[arxiv](https://arxiv.org/pdf/2411.16646)]
* Safer-Instruct: Aligning Language Models with Automated Preference Data `2024` [[NAACL](https://aclanthology.org/2024.naacl-long.422.pdf)]
* Negating Negatives: Alignment with Human Negative Samples via Distributional Dispreference Optimization `2024` [[EMNLP Findings](https://aclanthology.org/2024.findings-emnlp.56.pdf)]
* Interpreting Language Model Preferences Through the Lens of Decision Trees `2025` [[Online](https://rlhflow.github.io/posts/2025-01-22-decision-tree-reward-model/)]

### 🖥️ Reward Modeling

#### Type-Level

##### Discriminative Reward

* LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion `2023` [[ACL](https://aclanthology.org/2023.acl-long.792.pdf)]
* InternLM2 Technical Report `2024` [[arxiv](https://arxiv.org/pdf/2403.17297)]
* Advancing LLM Reasoning Generalists with Preference Trees `2024` [[arxiv](https://arxiv.org/pdf/2404.02078)]
* Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs `2024` [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/71f7154547c748c8041505521ca433ab-Paper-Conference.pdf)]
* MetaMetrics: Calibrating Metrics For Generation Tasks Using Human Preferences `2024` [[ICLR](https://openreview.net/pdf?id=slO3xTt4CG)]
* Nemotron-4 340B Technical Report `2024` [[arxiv](https://arxiv.org/pdf/2406.11704)]
* Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts `2024` [[arxiv](https://arxiv.org/pdf/2406.12845)]

##### Generative Reward

* Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena `2023` [[NeurIPS](https://openreview.net/pdf?id=uccHPGDlao)]
* Generative Judge for Evaluating Alignment `2024` [[ICLR](https://openreview.net/pdf?id=gtkFw6sZGS)]
* Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models `2024` [[EMNLP](https://aclanthology.org/2024.emnlp-main.248.pdf)]
* CompassJudger-1: All-in-one Judge Model Helps Model Evaluation and Evolution `2024` [[arxiv](https://arxiv.org/pdf/2410.16256)] 
* LLM Critics Help Catch LLM Bugs `2024` [[arxiv](https://arxiv.org/pdf/2407.00215)] 
* LLM Critics Help Catch Bugs in Mathematics: Towards a Better Mathematical Verifier with Natural Language Feedback `2024` [[arxiv](https://arxiv.org/pdf/2406.14024)]
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
* SLiC-HF: Sequence Likelihood Calibration with Human Feedback `2023` [[arxiv](https://arxiv.org/pdf/2305.10425)]
* A General Theoretical Paradigm to Understand Learning from Human Preferences `2023` [[arxiv](https://arxiv.org/pdf/2310.12036)]
* A Minimaximalist Approach to Reinforcement Learning from Human Feedback `2024` [[ICML](https://openreview.net/pdf?id=5kVgd2MwMY)]
* Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive `2024` [[arxiv](https://arxiv.org/pdf/2402.13228)]
* From $r$ to $Q^*$: Your Language Model is Secretly a Q-Function `2024` [[COLM](https://openreview.net/pdf?id=kEVcNxtqXk)]
* Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs `2024` [[arxiv](https://arxiv.org/pdf/2406.18629)]
* Token-level Direct Preference Optimization `2024` [[ICML](https://openreview.net/pdf?id=1RZKuvqYCR)]
* $β$-DPO: Direct Preference Optimization with Dynamic $β$ `2024` [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/ea888178abdb6fc233226d12321d754f-Paper-Conference.pdf)]
* Generalized Preference Optimization: A Unified Approach to Offline Alignment `2024` [[ICML](https://openreview.net/pdf?id=gu3nacA9AH)]
* Contrastive Preference Optimization: Pushing the Boundaries of LLM `2024` [[ICML](https://openreview.net/pdf?id=51iwkioZpn)]
* Offline Regularised Reinforcement Learning for Large Language Models Alignment `2024` [[arxiv](https://arxiv.org/pdf/2405.19107)]
* Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences `2024` [[arxiv](https://arxiv.org/pdf/2404.03715)]
* ORPO: Monolithic Preference Optimization without Reference Model `2024` [[EMNLP](https://aclanthology.org/2024.emnlp-main.626.pdf)]
* Mixed Preference Optimization: A Two-stage Reinforcement Learning with Human Feedbacks `2024` [[arxiv](https://arxiv.org/pdf/2403.19443)]
* LiPO: Listwise Preference Optimization through Learning-to-Rank `2024` [[arxiv](https://arxiv.org/pdf/2402.01878)]
* Noise Contrastive Alignment of Language Models with Explicit Rewards `2024` [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/d5a58d198afa370a3dff0e1ca4fe1802-Paper-Conference.pdf)]
* SimPO: Simple Preference Optimization with a Reference-Free Reward `2024` [[NeurIPS](https://openreview.net/pdf?id=3Tzcot1LKb)]
* Direct Preference Optimization with an Offset `2024` [[ACL Findings](https://aclanthology.org/2024.findings-acl.592.pdf)]
* Statistical Rejection Sampling Improves Preference Optimization `2024` [[ICLR](https://openreview.net/pdf?id=xbjSwwrQOe)]
* sDPO: Don’t Use Your Data All at Once `2025` [[COLING Industry](https://aclanthology.org/2025.coling-industry.31.pdf)]
* Towards Robust Alignment of Language Models: Distributionally Robustifying Direct Preference Optimization `2025` [[ICLR](https://openreview.net/pdf?id=CbfsKHiWEn)]
* Self-Play Preference Optimization for Language Model Alignment `2025` [[ICLR](https://openreview.net/pdf?id=a3PmRgAB5T)]
* TIS-DPO: Token-level Importance Sampling for Direct Preference Optimization With Estimated Weights `2025` [[ICLR](https://openreview.net/pdf?id=oF6e2WwxX0)]

#### Granularity-Level

##### Outcome Reward

TBD

##### Process Reward

* Solving math word problems with process- and outcome-based feedback `2022` [[arxiv](https://arxiv.org/pdf/2211.14275)]
* GRACE: Discriminator-Guided Chain-of-Thought Reasoning `2023` [[EMNLP Findings](https://aclanthology.org/2023.findings-emnlp.1022.pdf)]
* Making Language Models Better Reasoners with Step-Aware Verifier `2023` [[ACL](https://aclanthology.org/2023.acl-long.291.pdf)]
* Let's reward step by step: Step-Level reward model as the Navigators for Reasoning `2023` [[arxiv](https://arxiv.org/pdf/2310.10080)]
* Let’s Reinforce Step by Step `2023` [[NeurIPS Workshop](https://openreview.net/pdf?id=QkdRqpClab)]
* Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations `2024` [[ACL](https://aclanthology.org/2024.acl-long.510.pdf)]
* Multi-step Problem Solving Through a Verifier: An Empirical Analysis on Model-induced Process Supervision `2024` [[EMNLP Findings](https://aclanthology.org/2024.findings-emnlp.429.pdf)]
* Improve Mathematical Reasoning in Language Models by Automated Process Supervision `2024` [[arxiv](https://arxiv.org/pdf/2406.06592)]
* Let's Verify Step by Step `2024` [[ICLR](https://openreview.net/pdf?id=v8L0pN6EOi)]
* AutoPSV: Automated Process-Supervised Verifier `2024` [[NeurIPS](https://openreview.net/pdf?id=eOAPWWOGs9)]
* Process Reward Model with Q-value Rankings `2025` [[ICLR](https://openreview.net/pdf?id=wQEdh2cgEk)]
* Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning `2025` [[ICLR](https://openreview.net/pdf?id=A6Y7AqlzLW)]
* Process Reinforcement through Implicit Rewards `2025` [[arxiv](https://arxiv.org/pdf/2502.01456)]
* The Lessons of Developing Process Reward Models in Mathematical Reasoning `2025` [[arxiv](https://arxiv.org/pdf/2501.07301)]
* AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence `2025` [[arxiv](https://arxiv.org/pdf/2502.13943)]
* Better Process Supervision with Bi-directional Rewarding Signals `2025` [[arxiv](https://arxiv.org/pdf/2503.04618)]

### 🦾 Usages

#### Data Selection

* RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment `2023` [[TMLR](https://openreview.net/pdf?id=m7p5O7zblY)]
* RRHF: Rank Responses to Align Language Models with Human Feedback without tears `2023` [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/23e6f78bdec844a9f7b6c957de2aae91-Paper-Conference.pdf)]
* Reinforced Self-Training (ReST) for Language Modeling `2023` [[arxiv](https://arxiv.org/pdf/2308.08998)]
* Iterative Reasoning Preference Optimization `2024` [[NeruIPS](https://openreview.net/pdf?id=4XIKfvNYvx)]
* Filtered Direct Preference Optimization `2024` [[EMNLP](https://aclanthology.org/2024.emnlp-main.1266.pdf)]

#### Policy Training

* Fine-Grained Human Feedback Gives Better Rewards for Language Model Training `2023` [[NeurIPS](https://papers.neurips.cc/paper_files/paper/2023/file/b8c90b65739ae8417e61eadb521f63d5-Paper-Conference.pdf)]
* Aligning Crowd Feedback via Distributional Preference Reward modeling `2024` [[arxiv](https://arxiv.org/pdf/2402.09764)]
* Reward-Robust RLHF in LLMs `2024` [[arxiv](https://arxiv.org/pdf/2409.15360)]
* Bayesian Reward Models for LLM Alignment `2024` [[arxiv](https://arxiv.org/pdf/2402.13210)]
* Prior Constraints-based Reward Model Training for Aligning Large Language Models `2024` [[CCL](https://aclanthology.org/2024.ccl-1.107.pdf)]
* ODIN: Disentangled Reward Mitigates Hacking in RLHF `2024` [[ICML](https://openreview.net/pdf?id=zcIV8OQFVF)]
* Disentangling Length from Quality in Direct Preference Optimization `2024` [[ACL Findings](https://aclanthology.org/2024.findings-acl.297.pdf)]
* WARM: On the Benefits of Weight Averaged Reward Models `2024` [[ICML](https://openreview.net/pdf?id=s7RDnNUJy6)]
* Improving Reinforcement Learning from Human Feedback with Efficient Reward Model Ensemble `2024` [[arxiv](https://arxiv.org/pdf/2401.16635)]
* RRM: Robust Reward Model Training Mitigates Reward Hacking `2025` [[ICLR](https://openreview.net/pdf?id=88AS5MQnmC)]
* Beyond Reward Hacking: Causal Rewards for Large Language Model Alignment `2025` [[arxiv](https://arxiv.org/pdf/2501.09620)]

#### Inference

* Let's reward step by step: Step-Level reward model as the Navigators for Reasoning `2023` [[arxiv](https://arxiv.org/pdf/2310.10080)]
* ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search `2024` [[arxiv](https://arxiv.org/pdf/2406.03816)]
* Advancing Process Verification for Large Language Models via Tree-Based Preference Learning `2024` [[arxiv](https://arxiv.org/pdf/2407.00390)]
* Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning `2025` [[ICLR](https://openreview.net/pdf?id=A6Y7AqlzLW)]
* Process Reward Models for LLM Agents: Practical Framework and Directions `2025` [[arxiv](https://arxiv.org/pdf/2502.10325)]
* Reward-Guided Speculative Decoding for Efficient LLM Reasoning `2025` [[arxiv](https://arxiv.org/pdf/2501.19324)]

### 🛠️ Applications

#### Harmless Dialogue

*Dialogue*

* Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback `2022` [[arxiv](https://arxiv.org/pdf/2204.05862)] 
* Constitutional AI: Harmlessness from AI Feedback `2022` [[arxiv](https://arxiv.org/pdf/2212.08073)] 
* Zhongjing: Enhancing the Chinese Medical Capabilities of Large Language Model through Expert Feedback and Real-world Multi-turn Dialogue `2023` [[arxiv](https://arxiv.org/pdf/2308.03549)]
* HuatuoGPT, Towards Taming Language Models To Be a Doctor `2023` [[EMNLP Findings](https://aclanthology.org/2023.findings-emnlp.725.pdf)]
* Empathy Level Alignment via Reinforcement Learning for Empathetic Response Generation `2024` [[arxiv](https://arxiv.org/pdf/2408.02976)]
* RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback `2024` [[ICML](https://openreview.net/pdf?id=uydQ2W41KO)]
* Safe RLHF: Safe Reinforcement Learning from Human Feedback `2024` [[ICLR](https://openreview.net/pdf?id=TyFrPOKYXw)]
* Deliberative Alignment: Reasoning Enables Safer Language Models `2024` [[arxiv](https://arxiv.org/pdf/2412.16339)]
* Training Dialogue Systems by AI Feedback for Improving Overall Dialogue Impression `2025` [[arxiv](https://arxiv.org/pdf/2501.12698)]

#### Logical Reasoning

*Math*
* Training Verifiers to Solve Math Word Problems `2022` [[arxiv](https://arxiv.org/pdf/2110.14168)]
* Solving math word problems with process- and outcome-based feedback `2022` [[arxiv](https://arxiv.org/pdf/2211.14275)]
* WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct `2023` [[arxiv](https://arxiv.org/pdf/2308.09583)]
* Let's Verify Step by Step `2024` [[ICLR](https://openreview.net/pdf?id=v8L0pN6EOi)]
* DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models `2024` [[arxiv](https://arxiv.org/pdf/2402.03300)]
* Improve Mathematical Reasoning in Language Models by Automated Process Supervision `2024` [[arxiv](https://arxiv.org/pdf/2406.06592)]
* Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations `2024` [[ACL](https://aclanthology.org/2024.acl-long.510.pdf)]
* The Lessons of Developing Process Reward Models in Mathematical Reasoning `2025` [[arxiv](https://arxiv.org/pdf/2501.07301)]
* Retrieval-Augmented Process Reward Model for Generalizable Mathematical Reasoning `2025` [[arxiv](https://arxiv.org/pdf/2502.14361)]

*Code*
* Let's reward step by step: Step-Level reward model as the Navigators for Reasoning `2023` [[arxiv](https://arxiv.org/pdf/2310.10080)]
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
* Efficient Preference-Based Reinforcement Learning Using Learned Dynamics Models `2023` [[ICRA](https://ieeexplore.ieee.org/document/10161081)]
* Accelerating Reinforcement Learning of Robotic Manipulations via Feedback from Large Language Models `2023` [[arxiv](https://arxiv.org/pdf/2311.02379)]
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
* VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models `2025` [[arxiv](https://arxiv.org/pdf/2503.07478)]

### 🤺 Challenges

#### Data

* Fine-Tuning Language Models from Human Preferences `2019` [[arxiv](https://arxiv.org/abs/1909.08593)]
* The Expertise Problem: Learning from Specialized Feedback `2022` [[arxiv](https://arxiv.org/abs/2211.06519)]
* Active Reward Learning from Multiple Teachers `2023` [[AAAI Workshop](https://ceur-ws.org/Vol-3381/48.pdf)] 
* Peering Through Preferences: Unraveling Feedback Acquisition for Aligning Large Language Models `2024` [[ICLR](https://openreview.net/forum?id=dKl6lMwbCy)]

#### Training

* Defining and Characterizing Reward Hacking `2022` [[NeurIPS](https://openreview.net/pdf?id=yb3HOXO3lX2)]
* A Baseline Analysis of Reward Models' Ability To Accurately Analyze Foundation Models Under Distribution Shift `2023` [[arxiv](https://arxiv.org/pdf/2311.14743)]
* Scaling Laws for Reward Model Overoptimization `2023` [[ICML](https://proceedings.mlr.press/v202/gao23h/gao23h.pdf)]
* Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback `2023` [[EMNLP Findings](https://aclanthology.org/2023.findings-emnlp.188.pdf)]
* Honesty to Subterfuge: In-Context Reinforcement Learning Can Make Honest Models Reward Hack `2024` [[arxiv](https://arxiv.org/pdf/2410.06491)]
* Sycophancy to Subterfuge: Investigating Reward-Tampering in Large Language Models `2024` [[arxiv](https://arxiv.org/pdf/2406.10162)]
* Language Models Learn to Mislead Humans via RLHF `2024` [[arxiv](https://arxiv.org/pdf/2409.12822)]
* Towards Understanding Sycophancy in Language Models `2024` [[ICLR](https://openreview.net/pdf?id=tvhaxkMKAn)]
* Reward Model Ensembles Help Mitigate Overoptimization `2024` [[ICLR](https://openreview.net/pdf?id=dcjtMYkpXx)]
* Catastrophic Goodhart: regularizing RLHF with KL divergence does not mitigate heavy-tailed reward misspecification `2024` [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/1a8189929f3d7bd6183718f42c3f4309-Paper-Conference.pdf)]
* Spontaneous Reward Hacking in Iterative Self-Refinement `2024` [[arxiv](https://arxiv.org/pdf/2407.04549)]
* Confronting Reward Model Overoptimization with Constrained RLHF `2024` [[ICLR](https://openreview.net/pdf?id=gkfUvn0fLU)]
* Rethinking Reward Model Evaluation: Are We Barking up the Wrong Tree? `2025` [[ICLR](https://openreview.net/pdf?id=Cnwz9jONi5)]
* Correlated Proxies: A New Definition and Improved Mitigation for Reward Hacking `2025` [[ICLR](https://openreview.net/pdf?id=msEr27EejF)]
* RRM: Robust Reward Model Training Mitigates Reward Hacking `2025` [[ICLR](https://openreview.net/pdf?id=88AS5MQnmC)]
* The Energy Loss Phenomenon in RLHF: A New Perspective on Mitigating Reward Hacking `2025` [[arxiv](https://arxiv.org/pdf/2501.19358)]

#### Evaluation

* An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Models are Task-specific Classifiers `2024` [[arxiv](https://arxiv.org/pdf/2403.02839)] 
* OffsetBias: Leveraging Debiased Data for Tuning Evaluators `2024` [[EMNLP Findings](https://aclanthology.org/2024.findings-emnlp.57.pdf)] 
* Preference Leakage: A Contamination Problem in LLM-as-a-judge `2025` [[arxiv](https://arxiv.org/pdf/2502.01534)]

### 📊 Analysis

* The Accuracy Paradox in RLHF: When Better Reward Models Don’t Yield Better Language Models `2024` [[EMNLP](https://aclanthology.org/2024.emnlp-main.174.pdf)]
* Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking `2024` [[COLM](https://openreview.net/pdf?id=5u1GpUkKtG)]
* Rethinking Bradley-Terry Models in Preference-Based Reward Modeling: Foundations, Theory, and Alternatives `2024` [[arxiv](https://arxiv.org/pdf/2411.04991)]
* RLHF Deciphered: A Critical Analysis of Reinforcement Learning from Human Feedback for LLMs `2024` [[arxiv](https://arxiv.org/pdf/2404.08555)]
* Towards Analyzing and Understanding the Limitations of DPO: A Theoretical Perspective `2024` [[arxiv](https://arxiv.org/pdf/2404.04626)]
* Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study `2024` [[ICML](https://openreview.net/forum?id=6XH8R7YrSk)]
* Rethinking Reward Model Evaluation: Are We Barking up the Wrong Tree? `2025` [[ICLR](https://openreview.net/pdf/7aa9cdaa8ae2a1fe57278fed0f70bed213ce9381.pdf)]
* All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning `2025` [[arxiv](https://arxiv.org/pdf/2503.01067)]

## Resources

### 🌏 Blogs

* Illustrating Reinforcement Learning from Human Feedback (RLHF) [[Link](https://huggingface.co/blog/rlhf)]
* Why reward models are key for alignment [[Link](https://www.interconnects.ai/p/why-reward-models-matter)]
* Reward Hacking in Reinforcement Learning [[Link](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)]

### 📚 Prior Survey

* A Survey on Interactive Reinforcement Learning: Design Principles and Open Challenges `2021` [[arxiv](https://arxiv.org/pdf/2105.12949)] 
* Reinforcement Learning With Human Advice: A Survey `2021` [[Frontiers Robotics AI](https://doi.org/10.3389/frobt.2021.584075)] 
* AI Alignment: A Comprehensive Survey `2023` [[arxiv](https://arxiv.org/pdf/2310.19852)] 
* A Survey of Reinforcement Learning from Human Feedback `2023` [[arxiv](https://arxiv.org/pdf/2312.14925)] 
* Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback `2023` [[TMLR](https://openreview.net/pdf?id=bx24KpJ4Eb)] 
* Human-in-the-Loop Reinforcement Learning: A Survey and Position on Requirements, Challenges, and Opportunities `2024` [[JAIR](https://jair.org/index.php/jair/article/view/15348/27006)]
* Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods `2024` [[arxiv](https://arxiv.org/pdf/2404.00282)]
* A Survey on Human Preference Learning for Large Language Models `2024` [[arxiv](https://arxiv.org/pdf/2406.11191)]
* A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and More `2024` [[arxiv](https://arxiv.org/pdf/2407.16216)]
* Reinforcement Learning Enhanced LLMs: A Survey `2024` [[arxiv](https://arxiv.org/pdf/2412.10400)]
* Towards a Unified View of Preference Learning for Large Language Models: A Survey `2024` [[arxiv](https://arxiv.org/pdf/2409.02795)]
* A Survey on Post-training of Large Language Models `2025` [[arxiv](https://arxiv.org/pdf/2503.06072)]
