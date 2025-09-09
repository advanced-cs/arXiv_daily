# 自然语言处理 cs.CL

- **最新发布 91 篇**

- **更新 106 篇**

## 最新发布

#### [new 001] LAMDAS: LLM as an Implicit Classifier for Domain-specific Data Selection
- **分类: cs.CL**

- **简介: 论文提出LAMDAS方法，利用预训练大语言模型作为隐式分类器，解决领域适配中高质量数据稀缺的问题。通过将数据选择建模为一类分类问题，实现高效准确的领域数据筛选，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.06524v1](http://arxiv.org/pdf/2509.06524v1)**

> **作者:** Jian Wu; Hang Yu; Bingchang Liu; Wenjie Yang; Peng Di; Jianguo Li; Yue Zhang
>
> **摘要:** Adapting large language models (LLMs) to specific domains often faces a critical bottleneck: the scarcity of high-quality, human-curated data. While large volumes of unchecked data are readily available, indiscriminately using them for fine-tuning risks introducing noise and degrading performance. Strategic data selection is thus crucial, requiring a method that is both accurate and efficient. Existing approaches, categorized as similarity-based and direct optimization methods, struggle to simultaneously achieve these goals. In this paper, we introduce LAMDAS (LLM As an iMplicit classifier for domain-specific DAta Selection), a novel approach that leverages the pre-trained LLM itself as an implicit classifier, thereby bypassing explicit feature engineering and computationally intensive optimization process. LAMDAS reframes data selection as a one-class classification problem, identifying candidate data that "belongs" to the target domain defined by a small reference dataset. Extensive experimental results demonstrate that LAMDAS not only exceeds the performance of full-data training using a fraction of the data but also outperforms nine state-of-the-art (SOTA) baselines under various scenarios. Furthermore, LAMDAS achieves the most compelling balance between performance gains and computational efficiency compared to all evaluated baselines.
>
---
#### [new 002] Orthogonal Low-rank Adaptation in Lie Groups for Continual Learning of Large Language Models
- **分类: cs.CL**

- **简介: 论文提出OLieRA方法，解决大语言模型在连续学习中的灾难性遗忘问题。通过引入李群理论，利用乘法更新保持参数几何结构，并施加正交约束，提升多任务学习性能。属于持续学习任务。**

- **链接: [http://arxiv.org/pdf/2509.06100v1](http://arxiv.org/pdf/2509.06100v1)**

> **作者:** Kefan Cao; Shuaicheng Wu
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** Large language models (LLMs) are prone to catastrophic forgetting in sequential multi-task settings. Parameter regularization methods such as O-LoRA and N-LoRA alleviate task interference by enforcing low-rank subspace orthogonality, but they overlook the fact that conventional additive fine-tuning disrupts the intrinsic geometric structure of LLM parameters, limiting performance. Our key insight is that the parameter space of LLMs possesses a geometric structure, which must be preserved in addition to enforcing orthogonality. Based on this, we propose Orthogonal Low-rank Adaptation in Lie Groups (OLieRA), which introduces Lie group theory into LLM fine-tuning: leveraging multiplicative updates to preserve parameter geometry while applying orthogonality constraints to task subspaces. Experiments demonstrate that OLieRA achieves state-of-the-art results on the Standard CL benchmark and remains among the top-performing methods in the Large Number of Tasks setting.
>
---
#### [new 003] Saturation-Driven Dataset Generation for LLM Mathematical Reasoning in the TPTP Ecosystem
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种基于TPTP库的符号数据生成方法，解决LLM数学推理数据稀缺问题。通过E-prover饱和生成定理，构建高质量训练数据集，并设计三个任务评估模型推理能力。属于数学推理数据生成与评估任务。**

- **链接: [http://arxiv.org/pdf/2509.06809v1](http://arxiv.org/pdf/2509.06809v1)**

> **作者:** Valentin Quesnel; Damien Sileo
>
> **摘要:** The scarcity of high-quality, logically sound data is a critical bottleneck for advancing the mathematical reasoning of Large Language Models (LLMs). Our work confronts this challenge by turning decades of automated theorem proving research into a scalable data engine. Rather than relying on error-prone LLMs or complex proof-assistant syntax like Lean and Isabelle, our framework leverages E-prover's saturation capabilities on the vast TPTP axiom library to derive a massive, guaranteed-valid corpus of theorems. Our pipeline is principled and simple: saturate axioms, filter for "interesting" theorems, and generate tasks. With no LLMs in the loop, we eliminate factual errors by construction. This purely symbolic data is then transformed into three difficulty-controlled challenges: entailment verification, premise selection, and proof reconstruction. Our zero-shot experiments on frontier models reveal a clear weakness: performance collapses on tasks requiring deep, structural reasoning. Our framework provides both the diagnostic tool to measure this gap and a scalable source of symbolic training data to address it. We make the code and data publicly available. https://github.com/sileod/reasoning_core https://hf.co/datasets/reasoning-core/rc1
>
---
#### [new 004] SLiNT: Structure-aware Language Model with Injection and Contrastive Training for Knowledge Graph Completion
- **分类: cs.CL; cs.AI**

- **简介: 论文提出SLiNT框架，用于知识图谱补全任务，解决结构信息利用不足导致的预测偏差问题。通过结构注入与对比训练，提升LLM在稀疏和零样本场景下的链接预测性能。**

- **链接: [http://arxiv.org/pdf/2509.06531v1](http://arxiv.org/pdf/2509.06531v1)**

> **作者:** Mengxue Yang; Chun Yang; Jiaqi Zhu; Jiafan Li; Jingqi Zhang; Yuyang Li; Ying Li
>
> **备注:** Accepted by EMNLP Findings 2025
>
> **摘要:** Link prediction in knowledge graphs requires integrating structural information and semantic context to infer missing entities. While large language models offer strong generative reasoning capabilities, their limited exploitation of structural signals often results in structural sparsity and semantic ambiguity, especially under incomplete or zero-shot settings. To address these challenges, we propose SLiNT (Structure-aware Language model with Injection and coNtrastive Training), a modular framework that injects knowledge-graph-derived structural context into a frozen LLM backbone with lightweight LoRA-based adaptation for robust link prediction. Specifically, Structure-Guided Neighborhood Enhancement (SGNE) retrieves pseudo-neighbors to enrich sparse entities and mitigate missing context; Dynamic Hard Contrastive Learning (DHCL) introduces fine-grained supervision by interpolating hard positives and negatives to resolve entity-level ambiguity; and Gradient-Decoupled Dual Injection (GDDI) performs token-level structure-aware intervention while preserving the core LLM parameters. Experiments on WN18RR and FB15k-237 show that SLiNT achieves superior or competitive performance compared with both embedding-based and generation-based baselines, demonstrating the effectiveness of structure-aware representation learning for scalable knowledge graph completion.
>
---
#### [new 005] On the Same Wavelength? Evaluating Pragmatic Reasoning in Language Models across Broad Concepts
- **分类: cs.CL**

- **简介: 该论文评估语言模型的语用推理能力，通过Wavelength游戏框架测试其在语言理解和生成中的表现。研究发现大模型在理解任务中接近人类水平，而使用RSA方法可提升生成效果，揭示了LM在语用推理中的潜力与局限。**

- **链接: [http://arxiv.org/pdf/2509.06952v1](http://arxiv.org/pdf/2509.06952v1)**

> **作者:** Linlu Qiu; Cedegao E. Zhang; Joshua B. Tenenbaum; Yoon Kim; Roger P. Levy
>
> **备注:** EMNLP 2025 (Main)
>
> **摘要:** Language use is shaped by pragmatics -- i.e., reasoning about communicative goals and norms in context. As language models (LMs) are increasingly used as conversational agents, it becomes ever more important to understand their pragmatic reasoning abilities. We propose an evaluation framework derived from Wavelength, a popular communication game where a speaker and a listener communicate about a broad range of concepts in a granular manner. We study a range of LMs on both language comprehension and language production using direct and Chain-of-Thought (CoT) prompting, and further explore a Rational Speech Act (RSA) approach to incorporating Bayesian pragmatic reasoning into LM inference. We find that state-of-the-art LMs, but not smaller ones, achieve strong performance on language comprehension, obtaining similar-to-human accuracy and exhibiting high correlations with human judgments even without CoT prompting or RSA. On language production, CoT can outperform direct prompting, and using RSA provides significant improvements over both approaches. Our study helps identify the strengths and limitations in LMs' pragmatic reasoning abilities and demonstrates the potential for improving them with RSA, opening up future avenues for understanding conceptual representation, language understanding, and social reasoning in LMs and humans.
>
---
#### [new 006] LatinX: Aligning a Multilingual TTS Model with Direct Preference Optimization
- **分类: cs.CL**

- **简介: 该论文提出LatinX，一种多语言TTS模型，用于语音到语音翻译，保留说话人身份。通过DPO优化对齐，降低WER并提升说话人相似度。属于语音合成与翻译任务，解决跨语言身份保持与质量提升问题。**

- **链接: [http://arxiv.org/pdf/2509.05863v1](http://arxiv.org/pdf/2509.05863v1)**

> **作者:** Luis Felipe Chary; Miguel Arjona Ramirez
>
> **摘要:** We present LatinX, a multilingual text-to-speech (TTS) model for cascaded speech-to-speech translation that preserves the source speaker's identity across languages. LatinX is a 12-layer decoder-only Transformer trained in three stages: (i) pre-training for text-to-audio mapping, (ii) supervised fine-tuning for zero-shot voice cloning, and (iii) alignment with Direct Preference Optimization (DPO) using automatically labeled pairs based on Word Error Rate (WER) and speaker-similarity metrics. Trained on English and Romance languages with emphasis on Portuguese, LatinX with DPO consistently reduces WER and improves objective similarity over the fine-tuned baseline. Human evaluations further indicate stronger perceived speaker similarity than a strong baseline (XTTSv2), revealing gaps between objective and subjective measures. We provide cross-lingual analyses and discuss balanced preference signals and lower-latency architectures as future work.
>
---
#### [new 007] QCSE: A Pretrained Quantum Context-Sensitive Word Embedding for Natural Language Processing
- **分类: cs.CL**

- **简介: 该论文提出QCSE模型，用于自然语言处理中的量子上下文敏感词嵌入。旨在解决传统NLP中上下文表示不足的问题，利用量子计算特性提升语言建模能力，并在低资源语言（如Fulani）和英语数据集上验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.05729v1](http://arxiv.org/pdf/2509.05729v1)**

> **作者:** Charles M. Varmantchaonala; Niclas GÖtting; Nils-Erik SchÜtte; Jean Louis E. K. Fendji; Christopher Gies
>
> **摘要:** Quantum Natural Language Processing (QNLP) offers a novel approach to encoding and understanding the complexity of natural languages through the power of quantum computation. This paper presents a pretrained quantum context-sensitive embedding model, called QCSE, that captures context-sensitive word embeddings, leveraging the unique properties of quantum systems to learn contextual relationships in languages. The model introduces quantum-native context learning, enabling the utilization of quantum computers for linguistic tasks. Central to the proposed approach are innovative context matrix computation methods, designed to create unique, representations of words based on their surrounding linguistic context. Five distinct methods are proposed and tested for computing the context matrices, incorporating techniques such as exponential decay, sinusoidal modulation, phase shifts, and hash-based transformations. These methods ensure that the quantum embeddings retain context sensitivity, thereby making them suitable for downstream language tasks where the expressibility and properties of quantum systems are valuable resources. To evaluate the effectiveness of the model and the associated context matrix methods, evaluations are conducted on both a Fulani corpus, a low-resource African language, dataset of small size and an English corpus of slightly larger size. The results demonstrate that QCSE not only captures context sensitivity but also leverages the expressibility of quantum systems for representing rich, context-aware language information. The use of Fulani further highlights the potential of QNLP to mitigate the problem of lack of data for this category of languages. This work underscores the power of quantum computation in natural language processing (NLP) and opens new avenues for applying QNLP to real-world linguistic challenges across various tasks and domains.
>
---
#### [new 008] The Majority is not always right: RL training for solution aggregation
- **分类: cs.CL**

- **简介: 论文提出AggLM方法，通过强化学习训练模型聚合多个候选解，解决LLM在复杂任务中多数投票效果有限的问题。该方法能有效识别少数正确答案，优于传统规则和奖励模型，在多个基准上表现更优。**

- **链接: [http://arxiv.org/pdf/2509.06870v1](http://arxiv.org/pdf/2509.06870v1)**

> **作者:** Wenting Zhao; Pranjal Aggarwal; Swarnadeep Saha; Asli Celikyilmaz; Jason Weston; Ilia Kulikov
>
> **摘要:** Scaling up test-time compute, by generating multiple independent solutions and selecting or aggregating among them, has become a central paradigm for improving large language models (LLMs) on challenging reasoning tasks. While most prior work relies on simple majority voting or reward model ranking to aggregate solutions, these approaches may only yield limited benefits. In this work, we propose to learn aggregation as an explicit reasoning skill: given a set of candidate solutions, we train an aggregator model to review, reconcile, and synthesize a final, correct answer using reinforcement learning from verifiable rewards. A key ingredient is careful balancing of easy and hard training examples, allowing the model to learn both to recover minority-but-correct answers as well as easy majority-correct answers. Empirically, we find our method, AggLM, outperforms both strong rule-based and reward-model baselines, across multiple benchmarks. Furthermore, it generalizes effectively to solutions from differing models, including stronger ones than contained in the training data, all while requiring substantially fewer tokens than majority voting with larger numbers of solutions.
>
---
#### [new 009] Mask-GCG: Are All Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks?
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文提出Mask-GCG方法，通过可学习的token掩码识别并剪枝对抗后缀中冗余token，提升攻击效率。属于LLM jailbreak攻击优化任务，解决传统GCG固定长度后缀冗余问题。**

- **链接: [http://arxiv.org/pdf/2509.06350v1](http://arxiv.org/pdf/2509.06350v1)**

> **作者:** Junjie Mu; Zonghao Ying; Zhekui Fan; Zonglei Jing; Yaoyuan Zhang; Zhengmin Yu; Wenxin Zhang; Quanchen Zou; Xiangzheng Zhang
>
> **摘要:** Jailbreak attacks on Large Language Models (LLMs) have demonstrated various successful methods whereby attackers manipulate models into generating harmful responses that they are designed to avoid. Among these, Greedy Coordinate Gradient (GCG) has emerged as a general and effective approach that optimizes the tokens in a suffix to generate jailbreakable prompts. While several improved variants of GCG have been proposed, they all rely on fixed-length suffixes. However, the potential redundancy within these suffixes remains unexplored. In this work, we propose Mask-GCG, a plug-and-play method that employs learnable token masking to identify impactful tokens within the suffix. Our approach increases the update probability for tokens at high-impact positions while pruning those at low-impact positions. This pruning not only reduces redundancy but also decreases the size of the gradient space, thereby lowering computational overhead and shortening the time required to achieve successful attacks compared to GCG. We evaluate Mask-GCG by applying it to the original GCG and several improved variants. Experimental results show that most tokens in the suffix contribute significantly to attack success, and pruning a minority of low-impact tokens does not affect the loss values or compromise the attack success rate (ASR), thereby revealing token redundancy in LLM prompts. Our findings provide insights for developing efficient and interpretable LLMs from the perspective of jailbreak attacks.
>
---
#### [new 010] No Translation Needed: Forecasting Quality from Fertility and Metadata
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译质量预测任务，旨在无需运行翻译系统即可预测翻译质量。通过使用词元生育率、词元数量和语言学元数据等特征，构建梯度提升模型，在FLORES-200基准上实现了较高的预测准确率。**

- **链接: [http://arxiv.org/pdf/2509.05425v1](http://arxiv.org/pdf/2509.05425v1)**

> **作者:** Jessica M. Lundin; Ada Zhang; David Adelani; Cody Carroll
>
> **摘要:** We show that translation quality can be predicted with surprising accuracy \textit{without ever running the translation system itself}. Using only a handful of features, token fertility ratios, token counts, and basic linguistic metadata (language family, script, and region), we can forecast ChrF scores for GPT-4o translations across 203 languages in the FLORES-200 benchmark. Gradient boosting models achieve favorable performance ($R^{2}=0.66$ for XX$\rightarrow$English and $R^{2}=0.72$ for English$\rightarrow$XX). Feature importance analyses reveal that typological factors dominate predictions into English, while fertility plays a larger role for translations into diverse target languages. These findings suggest that translation quality is shaped by both token-level fertility and broader linguistic typology, offering new insights for multilingual evaluation and quality estimation.
>
---
#### [new 011] Beyond ROUGE: N-Gram Subspace Features for LLM Hallucination Detection
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型（LLM）生成内容中的幻觉检测问题。提出基于N-Gram子空间特征的方法，通过张量分解提取语义信息，训练分类器识别幻觉，优于传统指标和先进LLM判别方法。**

- **链接: [http://arxiv.org/pdf/2509.05360v1](http://arxiv.org/pdf/2509.05360v1)**

> **作者:** Jerry Li; Evangelos Papalexakis
>
> **摘要:** Large Language Models (LLMs) have demonstrated effectiveness across a wide variety of tasks involving natural language, however, a fundamental problem of hallucinations still plagues these models, limiting their trustworthiness in generating consistent, truthful information. Detecting hallucinations has quickly become an important topic, with various methods such as uncertainty estimation, LLM Judges, retrieval augmented generation (RAG), and consistency checks showing promise. Many of these methods build upon foundational metrics, such as ROUGE, BERTScore, or Perplexity, which often lack the semantic depth necessary to detect hallucinations effectively. In this work, we propose a novel approach inspired by ROUGE that constructs an N-Gram frequency tensor from LLM-generated text. This tensor captures richer semantic structure by encoding co-occurrence patterns, enabling better differentiation between factual and hallucinated content. We demonstrate this by applying tensor decomposition methods to extract singular values from each mode and use these as input features to train a multi-layer perceptron (MLP) binary classifier for hallucinations. Our method is evaluated on the HaluEval dataset and demonstrates significant improvements over traditional baselines, as well as competitive performance against state-of-the-art LLM judges.
>
---
#### [new 012] MachineLearningLM: Continued Pretraining Language Models on Millions of Synthetic Tabular Prediction Tasks Scales In-Context ML
- **分类: cs.CL; cs.AI**

- **简介: 论文提出MachineLearningLM框架，通过合成任务对LLM进行持续预训练，提升其在上下文学习中的机器学习能力。解决LLM在多示例ML任务中表现不佳的问题，实现跨领域分类性能提升并保持通用对话能力。**

- **链接: [http://arxiv.org/pdf/2509.06806v1](http://arxiv.org/pdf/2509.06806v1)**

> **作者:** Haoyu Dong; Pengkun Zhang; Mingzhe Lu; Yanzhen Shen; Guolin Ke
>
> **摘要:** Large language models (LLMs) possess broad world knowledge and strong general-purpose reasoning ability, yet they struggle to learn from many in-context examples on standard machine learning (ML) tasks, that is, to leverage many-shot demonstrations purely via in-context learning (ICL) without gradient descent. We introduce MachineLearningLM, a portable continued-pretraining framework that equips a general-purpose LLM with robust in-context ML capability while preserving its general knowledge and reasoning for broader chat workflows. Our pretraining procedure synthesizes ML tasks from millions of structural causal models (SCMs), spanning shot counts up to 1,024. We begin with a random-forest teacher, distilling tree-based decision strategies into the LLM to strengthen robustness in numerical modeling. All tasks are serialized with a token-efficient prompt, enabling 3x to 6x more examples per context window and delivering up to 50x amortized throughput via batch inference. Despite a modest setup (Qwen-2.5-7B-Instruct with LoRA rank 8), MachineLearningLM outperforms strong LLM baselines (e.g., GPT-5-mini) by an average of about 15% on out-of-distribution tabular classification across finance, physics, biology, and healthcare domains. It exhibits a striking many-shot scaling law: accuracy increases monotonically as in-context demonstrations grow from 8 to 1,024. Without any task-specific training, it attains random-forest-level accuracy across hundreds of shots. General chat capabilities, including knowledge and reasoning, are preserved: it achieves 75.4% on MMLU.
>
---
#### [new 013] MSLEF: Multi-Segment LLM Ensemble Finetuning in Recruitment
- **分类: cs.CL**

- **简介: 该论文提出MSLEF框架，用于招聘中的简历解析任务，解决单一模型适应简历格式多样性不足的问题。通过多段LLM集成与加权投票，提升解析准确性与适应性。**

- **链接: [http://arxiv.org/pdf/2509.06200v1](http://arxiv.org/pdf/2509.06200v1)**

> **作者:** Omar Walid; Mohamed T. Younes; Khaled Shaban; Mai Hassan; Ali Hamdi
>
> **备注:** Accepted in AICCSA 2025
>
> **摘要:** This paper presents MSLEF, a multi-segment ensemble framework that employs LLM fine-tuning to enhance resume parsing in recruitment automation. It integrates fine-tuned Large Language Models (LLMs) using weighted voting, with each model specializing in a specific resume segment to boost accuracy. Building on MLAR , MSLEF introduces a segment-aware architecture that leverages field-specific weighting tailored to each resume part, effectively overcoming the limitations of single-model systems by adapting to diverse formats and structures. The framework incorporates Gemini-2.5-Flash LLM as a high-level aggregator for complex sections and utilizes Gemma 9B, LLaMA 3.1 8B, and Phi-4 14B. MSLEF achieves significant improvements in Exact Match (EM), F1 score, BLEU, ROUGE, and Recruitment Similarity (RS) metrics, outperforming the best single model by up to +7% in RS. Its segment-aware design enhances generalization across varied resume layouts, making it highly adaptable to real-world hiring scenarios while ensuring precise and reliable candidate representation.
>
---
#### [new 014] Revolutionizing Reinforcement Learning Framework for Diffusion Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出TraceRL框架，用于改进扩散大语言模型的强化学习训练，提升复杂数学和编程任务的推理能力。通过轨迹感知训练和课程学习，显著提高模型性能，并发布开源框架支持多种架构。**

- **链接: [http://arxiv.org/pdf/2509.06949v1](http://arxiv.org/pdf/2509.06949v1)**

> **作者:** Yinjie Wang; Ling Yang; Bowen Li; Ye Tian; Ke Shen; Mengdi Wang
>
> **备注:** Code and Models: https://github.com/Gen-Verse/dLLM-RL
>
> **摘要:** We propose TraceRL, a trajectory-aware reinforcement learning framework for diffusion language models (DLMs) that incorporates preferred inference trajectory into post-training, and is applicable across different architectures. Equipped with a diffusion-based value model that enhances training stability, we demonstrate improved reasoning performance on complex math and coding tasks. Besides, it can also be applied to adapt block-specific models to larger blocks, which improves sampling flexibility. Employing TraceRL, we derive a series of state-of-the-art diffusion language models, namely TraDo. Although smaller than 7B-scale AR models, TraDo-4B-Instruct still consistently outperforms them across complex math reasoning tasks. TraDo-8B-Instruct achieves relative accuracy improvements of 6.1% over Qwen2.5-7B-Instruct and 51.3% over Llama3.1-8B-Instruct on mathematical reasoning benchmarks. Through curriculum learning, we also derive the first long-CoT DLM, outperforming Qwen2.5-7B-Instruct on MATH500 with an 18.1% relative accuracy gain. To facilitate reproducible research and practical applications, we release a comprehensive open-source framework for building, training, and deploying diffusion LLMs across diverse architectures. The framework integrates accelerated KV-cache techniques and inference engines for both inference and reinforcement learning, and includes implementations of various supervised fine-tuning and RL methods for mathematics, coding, and general tasks. Code and Models: https://github.com/Gen-Verse/dLLM-RL
>
---
#### [new 015] Cross-Question Method Reuse in Large Language Models: From Word-Level Prediction to Rational Logical-Layer Reasoning
- **分类: cs.CL**

- **简介: 该论文属于方法复用任务，旨在解决LLM在低相似度或隐含相似性问题间的复用难题。提出分离问题与解法，引导模型进行解法迁移，扩展复用范围，提升跨问题方法复用效果。**

- **链接: [http://arxiv.org/pdf/2509.05660v1](http://arxiv.org/pdf/2509.05660v1)**

> **作者:** Hong Su
>
> **摘要:** Large language models (LLMs) have been widely applied to assist in finding solutions for diverse questions. Prior work has proposed representing a method as a pair of a question and its corresponding solution, enabling method reuse. However, existing approaches typically require the questions to be highly similar. In this paper, we extend the scope of method reuse to address questions with low similarity or with hidden similarities that are not explicitly observable. For questions that are similar in a general-specific sense (i.e., broader or narrower in scope), we propose to first separate the question and solution, rather than directly feeding the pair to the LLM. The LLM is then guided to adapt the solution to new but related questions, allowing it to focus on solution transfer rather than question recognition. Furthermore, we extend this approach to cases where questions only share partial features or hidden characteristics. This enables cross-question method reuse beyond conventional similarity constraints. Experimental verification shows that our scope-extension approach increases the probability of filtering out reusable solutions, thereby improving the effectiveness of cross-question method reuse.
>
---
#### [new 016] From Joy to Fear: A Benchmark of Emotion Estimation in Pop Song Lyrics
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感分析任务，旨在解决流行歌曲歌词多标签情感强度预测问题。构建了人工标注数据集，评估了零样本大语言模型和微调BERT模型的效果，揭示其优劣，为音乐信息检索提供参考。**

- **链接: [http://arxiv.org/pdf/2509.05617v1](http://arxiv.org/pdf/2509.05617v1)**

> **作者:** Shay Dahary; Avi Edana; Alexander Apartsin; Yehudit Aperstein
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** The emotional content of song lyrics plays a pivotal role in shaping listener experiences and influencing musical preferences. This paper investigates the task of multi-label emotional attribution of song lyrics by predicting six emotional intensity scores corresponding to six fundamental emotions. A manually labeled dataset is constructed using a mean opinion score (MOS) approach, which aggregates annotations from multiple human raters to ensure reliable ground-truth labels. Leveraging this dataset, we conduct a comprehensive evaluation of several publicly available large language models (LLMs) under zero-shot scenarios. Additionally, we fine-tune a BERT-based model specifically for predicting multi-label emotion scores. Experimental results reveal the relative strengths and limitations of zero-shot and fine-tuned models in capturing the nuanced emotional content of lyrics. Our findings highlight the potential of LLMs for emotion recognition in creative texts, providing insights into model selection strategies for emotion-based music information retrieval applications. The labeled dataset is available at https://github.com/LLM-HITCS25S/LyricsEmotionAttribution.
>
---
#### [new 017] Icon$^{2}$: Aligning Large Language Models Using Self-Synthetic Preference Data via Inherent Regulation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型对齐任务，旨在解决传统偏好数据集构建中的分布不匹配与高计算成本问题。提出Icon²方法，利用模型表示空间的内在规律高效生成高质量偏好数据，提升对齐效果并降低计算开销。**

- **链接: [http://arxiv.org/pdf/2509.05605v1](http://arxiv.org/pdf/2509.05605v1)**

> **作者:** Qiyuan Chen; Hongsen Huang; Qian Shao; Jiahe Chen; Jintai Chen; Hongxia Xu; Renjie Hua; Ren Chuan; Jian Wu
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Large Language Models (LLMs) require high quality preference datasets to align with human preferences. However, conventional methods for constructing such datasets face significant challenges: reliance on pre-collected instructions often leads to distribution mismatches with target models, while the need for sampling multiple stochastic responses introduces substantial computational overhead. In this work, we explore a paradigm shift by leveraging inherent regulation of LLMs' representation space for efficient and tailored preference dataset construction, named Icon$^{2}$. Specifically, it first extracts layer-wise direction vectors to encode sophisticated human preferences and then uses these vectors to filter self-synthesized instructions based on their inherent consistency. During decoding, bidirectional inherent control is applied to steer token representations, enabling the precise generation of response pairs with clear alignment distinctions. Experimental results demonstrate significant improvements in both alignment and efficiency. Llama3-8B and Qwen2-7B achieve an average win rate improvement of 13.89% on AlpacaEval 2.0 and 13.45% on Arena-Hard, while reducing computational costs by up to 48.1%.
>
---
#### [new 018] Llama-GENBA-10B: A Trilingual Large Language Model for German, English and Bavarian
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Llama-GENBA-10B，一个支持英、德、巴伐利亚语的三语大模型，旨在解决英语偏见问题。通过平衡多语种数据和优化训练，提升低资源语言性能，推动巴伐利亚语发展，并建立标准化评估体系。**

- **链接: [http://arxiv.org/pdf/2509.05668v1](http://arxiv.org/pdf/2509.05668v1)**

> **作者:** Michael Hoffmann; Jophin John; Stefan Schweter; Gokul Ramakrishnan; Hoi-Fong Mak; Alice Zhang; Dmitry Gaynullin; Nicolay J. Hammer
>
> **备注:** Michael Hoffmann and Jophin John contributed equally to this work
>
> **摘要:** We present Llama-GENBA-10B, a trilingual foundation model addressing English-centric bias in large language models. Built on Llama 3.1-8B and scaled to 10B parameters, Llama-GENBA-10B is continuously pretrained on 164B tokens (82B English, 82B German, and 80M Bavarian), balancing resources while preventing English dominance. Targeted at the German NLP community, the model also promotes Bavarian as a low-resource language. Development tackled four challenges: (1) curating a multilingual corpus despite Bavarian scarcity, (2) creating a unified tokenizer for English, German, and Bavarian, (3) optimizing architecture and language-ratio hyperparameters for cross-lingual transfer, and (4) establishing the first standardized trilingual evaluation suite by translating German benchmarks into Bavarian. Evaluations show that Llama-GENBA-10B achieves strong cross-lingual performance, with the fine-tuned variant surpassing Apertus-8B-2509 and gemma-2-9b in Bavarian and establishing itself as the best model in its class for this language, while also outperforming EuroLLM in English and matching its results in German. Training on the Cerebras CS-2 demonstrated efficient large-scale multilingual pretraining with documented energy use, offering a blueprint for inclusive foundation models that integrate low-resource languages.
>
---
#### [new 019] EPT Benchmark: Evaluation of Persian Trustworthiness in Large Language Models
- **分类: cs.CL; cs.CR**

- **简介: 该论文提出EPT基准，评估大语言模型在波斯语环境下的可信度，涵盖真实性、安全等六个方面。通过自动化与人工评估，揭示模型在安全方面的不足，并探讨其与波斯伦理文化的对齐情况。**

- **链接: [http://arxiv.org/pdf/2509.06838v1](http://arxiv.org/pdf/2509.06838v1)**

> **作者:** Mohammad Reza Mirbagheri; Mohammad Mahdi Mirkamali; Zahra Motoshaker Arani; Ali Javeri; Amir Mahdi Sadeghzadeh; Rasool Jalili
>
> **摘要:** Large Language Models (LLMs), trained on extensive datasets using advanced deep learning architectures, have demonstrated remarkable performance across a wide range of language tasks, becoming a cornerstone of modern AI technologies. However, ensuring their trustworthiness remains a critical challenge, as reliability is essential not only for accurate performance but also for upholding ethical, cultural, and social values. Careful alignment of training data and culturally grounded evaluation criteria are vital for developing responsible AI systems. In this study, we introduce the EPT (Evaluation of Persian Trustworthiness) metric, a culturally informed benchmark specifically designed to assess the trustworthiness of LLMs across six key aspects: truthfulness, safety, fairness, robustness, privacy, and ethical alignment. We curated a labeled dataset and evaluated the performance of several leading models - including ChatGPT, Claude, DeepSeek, Gemini, Grok, LLaMA, Mistral, and Qwen - using both automated LLM-based and human assessments. Our results reveal significant deficiencies in the safety dimension, underscoring the urgent need for focused attention on this critical aspect of model behavior. Furthermore, our findings offer valuable insights into the alignment of these models with Persian ethical-cultural values and highlight critical gaps and opportunities for advancing trustworthy and culturally responsible AI. The dataset is publicly available at: https://github.com/Rezamirbagheri110/EPT-Benchmark.
>
---
#### [new 020] An Empirical Analysis of Discrete Unit Representations in Speech Language Modeling Pre-training
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文研究语音语言模型预训练中离散单元表示的影响因素，探讨模型结构、数据表示及训练鲁棒性对预训练效果的作用，分析不同模型规模下的最优离散化策略及聚类数据选择对模型性能的影响。属于语音语言模型预训练优化任务。**

- **链接: [http://arxiv.org/pdf/2509.05359v1](http://arxiv.org/pdf/2509.05359v1)**

> **作者:** Yanis Labrak; Richard Dufour; Mickaël Rouvier
>
> **备注:** Published in International Conference on Text, Speech, and Dialogue, 13-24
>
> **摘要:** This paper investigates discrete unit representations in Speech Language Models (SLMs), focusing on optimizing speech modeling during continual pre-training. In this paper, we systematically examine how model architecture, data representation, and training robustness influence the pre-training stage in which we adapt existing pre-trained language models to the speech modality. Our experiments highlight the role of speech encoders and clustering granularity across different model scales, showing how optimal discretization strategies vary with model capacity. By examining cluster distribution and phonemic alignments, we investigate the effective use of discrete vocabulary, uncovering both linguistic and paralinguistic patterns. Additionally, we explore the impact of clustering data selection on model robustness, highlighting the importance of domain matching between discretization training and target applications.
>
---
#### [new 021] Crown, Frame, Reverse: Layer-Wise Scaling Variants for LLM Pre-Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出三种层-wise缩放方法（Framed、Reverse、Crown），优化大语言模型预训练中各层的计算资源分配。任务为改进LLM预训练架构设计，解决传统统一层尺寸忽略不同深度功能差异的问题，通过参数预算下的系统实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.06518v1](http://arxiv.org/pdf/2509.06518v1)**

> **作者:** Andrei Baroian; Kasper Notebomer
>
> **备注:** The reported results are skewed due to a data type mismatch. The dataset was saved with int32, but the data loader interpreted it as uint16. As a result, each 32-bit token was incorrectly split into two 16-bit tokens. Outcome: a consistent artifact where every other token is zero
>
> **摘要:** Transformer-based language models traditionally use uniform (isotropic) layer sizes, yet they ignore the diverse functional roles that different depths can play and their computational capacity needs. Building on Layer-Wise Scaling (LWS) and pruning literature, we introduce three new LWS variants - Framed, Reverse, and Crown - that redistribute FFN widths and attention heads via two or three-point linear interpolation in the pre-training stage. We present the first systematic ablation of LWS and its variants, on a fixed budget of 180M parameters, trained on 5B tokens. All models converge to similar losses and achieve better performance compared to an equal-cost isotropic baseline, without a substantial decrease in training throughput. This work represents an initial step into the design space of layer-wise architectures for pre-training, but future work should scale experiments to orders of magnitude more tokens and parameters to fully assess their potential.
>
---
#### [new 022] Proof-Carrying Numbers (PCN): A Protocol for Trustworthy Numeric Answers from LLMs via Claim Verification
- **分类: cs.CL; cs.CR; cs.DB; cs.LG**

- **简介: 论文提出Proof-Carrying Numbers（PCN）协议，解决大语言模型生成数值时的“数值幻觉”问题。通过将数值与结构化声明绑定，并由验证器检查，确保数值可信。该方法实现轻量、模型无关，保障显示数值的准确性与安全性。**

- **链接: [http://arxiv.org/pdf/2509.06902v1](http://arxiv.org/pdf/2509.06902v1)**

> **作者:** Aivin V. Solatorio
>
> **摘要:** Large Language Models (LLMs) as stochastic systems may generate numbers that deviate from available data, a failure known as \emph{numeric hallucination}. Existing safeguards -- retrieval-augmented generation, citations, and uncertainty estimation -- improve transparency but cannot guarantee fidelity: fabricated or misquoted values may still be displayed as if correct. We propose \textbf{Proof-Carrying Numbers (PCN)}, a presentation-layer protocol that enforces numeric fidelity through mechanical verification. Under PCN, numeric spans are emitted as \emph{claim-bound tokens} tied to structured claims, and a verifier checks each token under a declared policy (e.g., exact equality, rounding, aliases, or tolerance with qualifiers). Crucially, PCN places verification in the \emph{renderer}, not the model: only claim-checked numbers are marked as verified, and all others default to unverified. This separation prevents spoofing and guarantees fail-closed behavior. We formalize PCN and prove soundness, completeness under honest tokens, fail-closed behavior, and monotonicity under policy refinement. PCN is lightweight and model-agnostic, integrates seamlessly into existing applications, and can be extended with cryptographic commitments. By enforcing verification as a mandatory step before display, PCN establishes a simple contract for numerically sensitive settings: \emph{trust is earned only by proof}, while the absence of a mark communicates uncertainty.
>
---
#### [new 023] Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning
- **分类: cs.CL**

- **简介: 论文提出一种联合SFT与RL的协作训练方法，解决LLM推理效率低的问题。通过双层优化，使SFT指导RL训练，提升推理效果与效率，在五个基准测试中优于基线方法。属于自然语言处理中的模型训练优化任务。**

- **链接: [http://arxiv.org/pdf/2509.06948v1](http://arxiv.org/pdf/2509.06948v1)**

> **作者:** Liang Chen; Xueting Han; Li Shen; Jing Bai; Kam-Fai Wong
>
> **摘要:** Reinforcement learning (RL) has proven effective in incentivizing the reasoning abilities of large language models (LLMs), but suffers from severe efficiency challenges due to its trial-and-error nature. While the common practice employs supervised fine-tuning (SFT) as a warm-up stage for RL, this decoupled two-stage approach limits interaction between SFT and RL, thereby constraining overall effectiveness. This study introduces a novel method for learning reasoning models that employs bilevel optimization to facilitate better cooperation between these training paradigms. By conditioning the SFT objective on the optimal RL policy, our approach enables SFT to meta-learn how to guide RL's optimization process. During training, the lower level performs RL updates while simultaneously receiving SFT supervision, and the upper level explicitly maximizes the cooperative gain-the performance advantage of joint SFT-RL training over RL alone. Empirical evaluations on five reasoning benchmarks demonstrate that our method consistently outperforms baselines and achieves a better balance between effectiveness and efficiency.
>
---
#### [new 024] No Encore: Unlearning as Opt-Out in Music Generation
- **分类: cs.CL**

- **简介: 该论文属于AI音乐生成领域，旨在解决版权风险问题。研究探索了机器“遗忘”技术在预训练文本到音乐模型中的应用，分析其有效性及挑战，为未来相关工作奠定基础。**

- **链接: [http://arxiv.org/pdf/2509.06277v1](http://arxiv.org/pdf/2509.06277v1)**

> **作者:** Jinju Kim; Taehan Kim; Abdul Waheed; Rita Singh
>
> **备注:** Work in progress. 7 pages
>
> **摘要:** AI music generation is rapidly emerging in the creative industries, enabling intuitive music generation from textual descriptions. However, these systems pose risks in exploitation of copyrighted creations, raising ethical and legal concerns. In this paper, we present preliminary results on the first application of machine unlearning techniques from an ongoing research to prevent inadvertent usage of creative content. Particularly, we explore existing methods in machine unlearning to a pre-trained Text-to-Music (TTM) baseline and analyze their efficacy in unlearning pre-trained datasets without harming model performance. Through our experiments, we provide insights into the challenges of applying unlearning in music generation, offering a foundational analysis for future works on the application of unlearning for music generative models.
>
---
#### [new 025] MoGU V2: Toward a Higher Pareto Frontier Between Model Usability and Security
- **分类: cs.CL**

- **简介: 该论文提出MoGU_v2框架，旨在提升大语言模型在安全性和实用性间的平衡。通过动态分配权重和优化路由机制，解决原有方法中安全与实用性的权衡问题，增强模型在多种场景下的适应性与安全性。**

- **链接: [http://arxiv.org/pdf/2509.06807v1](http://arxiv.org/pdf/2509.06807v1)**

> **作者:** Yanrui Du; Fenglei Fan; Sendong Zhao; Jiawei Cao; Ting Liu; Bing Qin
>
> **摘要:** As Large Language Models (LLMs) increasingly permeate human life, their security has emerged as a critical concern, particularly their ability to maintain harmless responses to malicious instructions. Although extensive methods have improved LLMs' security, they often lead to conservative, rejection-oriented responses that compromise practical usability. This presents a key challenge: how to advance the Pareto frontier between LLMs' usability and security, rather than necessitate a trade-off between them. To address this, we propose the MoGU framework, in which the intra-layer router dynamically allocates weights by sensing hidden states, thereby balancing the contributions of security-optimized and usability-optimized variants. Despite its initial potential, the MoGU framework faces limitations such as parameter redundancy and performance bottlenecks. To overcome these, we further propose an improved MoGU_v2 framework that establishes a tighter coupling between the routers and hidden states. In MoGU_v2, routers are embedded only in layers encoding highly classifiable security features, and backbone modules are activated during router optimization to enable bidirectional adaptation. MoGU_V2 exhibits strong adaptability and stable improvements across various series of LLMs, including mainstream LLMs serving as brains in various applications, on-device LLMs optimized for resource-constrained scenarios, and reasoning LLMs tailored for user interpretability. Meanwhile, even facing risks introduced by Instruction Fine-tuning, MoGU_v2 can easily restore security without compromising the task performance gains via a simple data-mix strategy. These comprehensive improvements highlight MoGU_V2 as a robust and versatile solution for mitigating security risks in real-world applications.
>
---
#### [new 026] MedFactEval and MedAgentBrief: A Framework and Workflow for Generating and Evaluating Factual Clinical Summaries
- **分类: cs.CL**

- **简介: 论文提出MedFactEval和MedAgentBrief，用于生成和评估临床摘要的事实准确性。任务是解决LLM生成临床文本的事实核查问题，通过多模型投票和专家共识提升评估效率与质量。**

- **链接: [http://arxiv.org/pdf/2509.05878v1](http://arxiv.org/pdf/2509.05878v1)**

> **作者:** François Grolleau; Emily Alsentzer; Timothy Keyes; Philip Chung; Akshay Swaminathan; Asad Aali; Jason Hom; Tridu Huynh; Thomas Lew; April S. Liang; Weihan Chu; Natasha Z. Steele; Christina F. Lin; Jingkun Yang; Kameron C. Black; Stephen P. Ma; Fateme N. Haredasht; Nigam H. Shah; Kevin Schulman; Jonathan H. Chen
>
> **摘要:** Evaluating factual accuracy in Large Language Model (LLM)-generated clinical text is a critical barrier to adoption, as expert review is unscalable for the continuous quality assurance these systems require. We address this challenge with two complementary contributions. First, we introduce MedFactEval, a framework for scalable, fact-grounded evaluation where clinicians define high-salience key facts and an "LLM Jury"--a multi-LLM majority vote--assesses their inclusion in generated summaries. Second, we present MedAgentBrief, a model-agnostic, multi-step workflow designed to generate high-quality, factual discharge summaries. To validate our evaluation framework, we established a gold-standard reference using a seven-physician majority vote on clinician-defined key facts from inpatient cases. The MedFactEval LLM Jury achieved almost perfect agreement with this panel (Cohen's kappa=81%), a performance statistically non-inferior to that of a single human expert (kappa=67%, P < 0.001). Our work provides both a robust evaluation framework (MedFactEval) and a high-performing generation workflow (MedAgentBrief), offering a comprehensive approach to advance the responsible deployment of generative AI in clinical workflows.
>
---
#### [new 027] Multimodal Fine-grained Context Interaction Graph Modeling for Conversational Speech Synthesis
- **分类: cs.CL**

- **简介: 该论文属于对话语音合成任务，旨在解决多模态对话历史中细粒度语义与韵律交互建模不足的问题。提出MFCIG-CSS模型，构建语义与韵律交互图，提升生成语音的自然韵律表现。**

- **链接: [http://arxiv.org/pdf/2509.06074v1](http://arxiv.org/pdf/2509.06074v1)**

> **作者:** Zhenqi Jia; Rui Liu; Berrak Sisman; Haizhou Li
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Conversational Speech Synthesis (CSS) aims to generate speech with natural prosody by understanding the multimodal dialogue history (MDH). The latest work predicts the accurate prosody expression of the target utterance by modeling the utterance-level interaction characteristics of MDH and the target utterance. However, MDH contains fine-grained semantic and prosody knowledge at the word level. Existing methods overlook the fine-grained semantic and prosodic interaction modeling. To address this gap, we propose MFCIG-CSS, a novel Multimodal Fine-grained Context Interaction Graph-based CSS system. Our approach constructs two specialized multimodal fine-grained dialogue interaction graphs: a semantic interaction graph and a prosody interaction graph. These two interaction graphs effectively encode interactions between word-level semantics, prosody, and their influence on subsequent utterances in MDH. The encoded interaction features are then leveraged to enhance synthesized speech with natural conversational prosody. Experiments on the DailyTalk dataset demonstrate that MFCIG-CSS outperforms all baseline models in terms of prosodic expressiveness. Code and speech samples are available at https://github.com/AI-S2-Lab/MFCIG-CSS.
>
---
#### [new 028] Ad hoc conventions generalize to new referents
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究人们如何为新事物建立共享命名系统。通过实验发现，人们在讨论新图像时能推广已有约定，表明命名具有概念协调性，而非任意标签。任务属于语言与认知研究，解决参考系统泛化问题。**

- **链接: [http://arxiv.org/pdf/2509.05566v1](http://arxiv.org/pdf/2509.05566v1)**

> **作者:** Anya Ji; Claire Augusta Bergey; Ron Eliav; Yoav Artzi; Robert D. Hawkins
>
> **摘要:** How do people talk about things they've never talked about before? One view suggests that a new shared naming system establishes an arbitrary link to a specific target, like proper names that cannot extend beyond their bearers. An alternative view proposes that forming a shared way of describing objects involves broader conceptual alignment, reshaping each individual's semantic space in ways that should generalize to new referents. We test these competing accounts in a dyadic communication study (N=302) leveraging the recently-released KiloGram dataset containing over 1,000 abstract tangram images. After pairs of participants coordinated on referential conventions for one set of images through repeated communication, we measured the extent to which their descriptions aligned for undiscussed images. We found strong evidence for generalization: partners showed increased alignment relative to their pre-test labels. Generalization also decayed nonlinearly with visual similarity (consistent with Shepard's law) and was robust across levels of the images' nameability. These findings suggest that ad hoc conventions are not arbitrary labels but reflect genuine conceptual coordination, with implications for theories of reference and the design of more adaptive language agents.
>
---
#### [new 029] Augmented Fine-Tuned LLMs for Enhanced Recruitment Automation
- **分类: cs.CL**

- **简介: 该论文提出一种增强的微调大语言模型（LLMs）方法，用于改进招聘自动化。通过构建标准化JSON格式的合成数据集和解析简历，提升模型在招聘任务中的准确性与效率，显著提高F1分数至90.62%。**

- **链接: [http://arxiv.org/pdf/2509.06196v1](http://arxiv.org/pdf/2509.06196v1)**

> **作者:** Mohamed T. Younes; Omar Walid; Khaled Shaban; Ali Hamdi; Mai Hassan
>
> **备注:** Accepted in AICCSA 2025
>
> **摘要:** This paper presents a novel approach to recruitment automation. Large Language Models (LLMs) were fine-tuned to improve accuracy and efficiency. Building upon our previous work on the Multilayer Large Language Model-Based Robotic Process Automation Applicant Tracking (MLAR) system . This work introduces a novel methodology. Training fine-tuned LLMs specifically tuned for recruitment tasks. The proposed framework addresses the limitations of generic LLMs by creating a synthetic dataset that uses a standardized JSON format. This helps ensure consistency and scalability. In addition to the synthetic data set, the resumes were parsed using DeepSeek, a high-parameter LLM. The resumes were parsed into the same structured JSON format and placed in the training set. This will help improve data diversity and realism. Through experimentation, we demonstrate significant improvements in performance metrics, such as exact match, F1 score, BLEU score, ROUGE score, and overall similarity compared to base models and other state-of-the-art LLMs. In particular, the fine-tuned Phi-4 model achieved the highest F1 score of 90.62%, indicating exceptional precision and recall in recruitment tasks. This study highlights the potential of fine-tuned LLMs. Furthermore, it will revolutionize recruitment workflows by providing more accurate candidate-job matching.
>
---
#### [new 030] ParCzech4Speech: A New Speech Corpus Derived from Czech Parliamentary Data
- **分类: cs.CL**

- **简介: 论文提出ParCzech4Speech 1.0，基于捷克议会数据构建的语音语料库，用于语音建模任务。通过WhisperX和Wav2Vec 2.0实现音频-文本对齐，提供三种数据变体，提升对齐可靠性与灵活性，支持多种语音任务。**

- **链接: [http://arxiv.org/pdf/2509.06675v1](http://arxiv.org/pdf/2509.06675v1)**

> **作者:** Vladislav Stankov; Matyáš Kopp; Ondřej Bojar
>
> **摘要:** We introduce ParCzech4Speech 1.0, a processed version of the ParCzech 4.0 corpus, targeted at speech modeling tasks with the largest variant containing 2,695 hours. We combined the sound recordings of the Czech parliamentary speeches with the official transcripts. The recordings were processed with WhisperX and Wav2Vec 2.0 to extract automated audio-text alignment. Our processing pipeline improves upon the ParCzech 3.0 speech recognition version by extracting more data with higher alignment reliability. The dataset is offered in three flexible variants: (1) sentence-segmented for automatic speech recognition and speech synthesis tasks with clean boundaries, (2) unsegmented preserving original utterance flow across sentences, and (3) a raw-alignment for further custom refinement for other possible tasks. All variants maintain the original metadata and are released under a permissive CC-BY license. The dataset is available in the LINDAT repository, with the sentence-segmented and unsegmented variants additionally available on Hugging Face.
>
---
#### [new 031] A Survey of the State-of-the-Art in Conversational Question Answering Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文综述了对话问答系统（ConvQA）的最新进展，分析其核心组件、机器学习技术应用及大模型影响，总结数据集与研究方向，旨在推动ConvQA领域的发展。**

- **链接: [http://arxiv.org/pdf/2509.05716v1](http://arxiv.org/pdf/2509.05716v1)**

> **作者:** Manoj Madushanka Perera; Adnan Mahmood; Kasun Eranda Wijethilake; Fahmida Islam; Maryam Tahermazandarani; Quan Z. Sheng
>
> **备注:** 42 pages, 12 figures, 4 tables
>
> **摘要:** Conversational Question Answering (ConvQA) systems have emerged as a pivotal area within Natural Language Processing (NLP) by driving advancements that enable machines to engage in dynamic and context-aware conversations. These capabilities are increasingly being applied across various domains, i.e., customer support, education, legal, and healthcare where maintaining a coherent and relevant conversation is essential. Building on recent advancements, this survey provides a comprehensive analysis of the state-of-the-art in ConvQA. This survey begins by examining the core components of ConvQA systems, i.e., history selection, question understanding, and answer prediction, highlighting their interplay in ensuring coherence and relevance in multi-turn conversations. It further investigates the use of advanced machine learning techniques, including but not limited to, reinforcement learning, contrastive learning, and transfer learning to improve ConvQA accuracy and efficiency. The pivotal role of large language models, i.e., RoBERTa, GPT-4, Gemini 2.0 Flash, Mistral 7B, and LLaMA 3, is also explored, thereby showcasing their impact through data scalability and architectural advancements. Additionally, this survey presents a comprehensive analysis of key ConvQA datasets and concludes by outlining open research directions. Overall, this work offers a comprehensive overview of the ConvQA landscape and provides valuable insights to guide future advancements in the field.
>
---
#### [new 032] Enhancing the Robustness of Contextual ASR to Varying Biasing Information Volumes Through Purified Semantic Correlation Joint Modeling
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于上下文语音识别（ASR）任务，旨在解决因偏倚信息量变化导致的模型鲁棒性问题。提出PSC-Joint方法，通过多层次语义关联建模与净化机制，有效整合最相关偏倚信息，提升识别性能。**

- **链接: [http://arxiv.org/pdf/2509.05908v1](http://arxiv.org/pdf/2509.05908v1)**

> **作者:** Yue Gu; Zhihao Du; Ying Shi; Shiliang Zhang; Qian Chen; Jiqing Han
>
> **备注:** Accepted by IEEE Transactions on Audio, Speech and Language Processing, 2025 (https://ieeexplore.ieee.org/document/11150731). DOI: 10.1109/TASLPRO.2025.3606198
>
> **摘要:** Recently, cross-attention-based contextual automatic speech recognition (ASR) models have made notable advancements in recognizing personalized biasing phrases. However, the effectiveness of cross-attention is affected by variations in biasing information volume, especially when the length of the biasing list increases significantly. We find that, regardless of the length of the biasing list, only a limited amount of biasing information is most relevant to a specific ASR intermediate representation. Therefore, by identifying and integrating the most relevant biasing information rather than the entire biasing list, we can alleviate the effects of variations in biasing information volume for contextual ASR. To this end, we propose a purified semantic correlation joint modeling (PSC-Joint) approach. In PSC-Joint, we define and calculate three semantic correlations between the ASR intermediate representations and biasing information from coarse to fine: list-level, phrase-level, and token-level. Then, the three correlations are jointly modeled to produce their intersection, so that the most relevant biasing information across various granularities is highlighted and integrated for contextual recognition. In addition, to reduce the computational cost introduced by the joint modeling of three semantic correlations, we also propose a purification mechanism based on a grouped-and-competitive strategy to filter out irrelevant biasing phrases. Compared with baselines, our PSC-Joint approach achieves average relative F1 score improvements of up to 21.34% on AISHELL-1 and 28.46% on KeSpeech, across biasing lists of varying lengths.
>
---
#### [new 033] Biomedical Literature Q&A System Using Retrieval-Augmented Generation (RAG)
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出基于RAG架构的生物医学问答系统，旨在提升公众获取准确医学信息的能力。系统整合PubMed等多源数据，结合语义检索与生成模型，实现精准回答。实验表明其在事实一致性和语义相关性上优于基线模型。**

- **链接: [http://arxiv.org/pdf/2509.05505v1](http://arxiv.org/pdf/2509.05505v1)**

> **作者:** Mansi Garg; Lee-Chi Wang; Bhavesh Ghanchi; Sanjana Dumpala; Shreyash Kakde; Yen Chih Chen
>
> **备注:** 10 pages, 6 figures, 3 tables
>
> **摘要:** This work presents a Biomedical Literature Question Answering (Q&A) system based on a Retrieval-Augmented Generation (RAG) architecture, designed to improve access to accurate, evidence-based medical information. Addressing the shortcomings of conventional health search engines and the lag in public access to biomedical research, the system integrates diverse sources, including PubMed articles, curated Q&A datasets, and medical encyclopedias ,to retrieve relevant information and generate concise, context-aware responses. The retrieval pipeline uses MiniLM-based semantic embeddings and FAISS vector search, while answer generation is performed by a fine-tuned Mistral-7B-v0.3 language model optimized using QLoRA for efficient, low-resource training. The system supports both general medical queries and domain-specific tasks, with a focused evaluation on breast cancer literature demonstrating the value of domain-aligned retrieval. Empirical results, measured using BERTScore (F1), show substantial improvements in factual consistency and semantic relevance compared to baseline models. The findings underscore the potential of RAG-enhanced language models to bridge the gap between complex biomedical literature and accessible public health knowledge, paving the way for future work on multilingual adaptation, privacy-preserving inference, and personalized medical AI systems.
>
---
#### [new 034] From Staff Messages to Actionable Insights: A Multi-Stage LLM Classification Framework for Healthcare Analytics
- **分类: cs.CL**

- **简介: 该论文提出一种基于LLM的多阶段分类框架，用于从医院呼叫中心的工作人员消息中提取可操作洞察。任务是分类消息主题和原因，解决传统方法依赖标注数据的问题，使用多种LLM模型实现高准确率，并确保HIPAA合规性。**

- **链接: [http://arxiv.org/pdf/2509.05484v1](http://arxiv.org/pdf/2509.05484v1)**

> **作者:** Hajar Sakai; Yi-En Tseng; Mohammadsadegh Mikaeili; Joshua Bosire; Franziska Jovin
>
> **摘要:** Hospital call centers serve as the primary contact point for patients within a hospital system. They also generate substantial volumes of staff messages as navigators process patient requests and communicate with the hospital offices following the established protocol restrictions and guidelines. This continuously accumulated large amount of text data can be mined and processed to retrieve insights; however, traditional supervised learning approaches require annotated data, extensive training, and model tuning. Large Language Models (LLMs) offer a paradigm shift toward more computationally efficient methodologies for healthcare analytics. This paper presents a multi-stage LLM-based framework that identifies staff message topics and classifies messages by their reasons in a multi-class fashion. In the process, multiple LLM types, including reasoning, general-purpose, and lightweight models, were evaluated. The best-performing model was o3, achieving 78.4% weighted F1-score and 79.2% accuracy, followed closely by gpt-5 (75.3% Weighted F1-score and 76.2% accuracy). The proposed methodology incorporates data security measures and HIPAA compliance requirements essential for healthcare environments. The processed LLM outputs are integrated into a visualization decision support tool that transforms the staff messages into actionable insights accessible to healthcare professionals. This approach enables more efficient utilization of the collected staff messaging data, identifies navigator training opportunities, and supports improved patient experience and care quality.
>
---
#### [new 035] Modelling Intertextuality with N-gram Embeddings
- **分类: cs.CL**

- **简介: 该论文提出一种基于n-gram嵌入的模型，用于量化文学文本间的互文性。通过比较文本对的嵌入并计算平均结果，实现互文性分析。方法有效且高效，并通过网络分析验证其捕捉互文关系的能力。属于文本分析任务，解决互文性量化问题。**

- **链接: [http://arxiv.org/pdf/2509.06637v1](http://arxiv.org/pdf/2509.06637v1)**

> **作者:** Yi Xing
>
> **摘要:** Intertextuality is a central tenet in literary studies. It refers to the intricate links between literary texts that are created by various types of references. This paper proposes a new quantitative model of intertextuality to enable scalable analysis and network-based insights: perform pairwise comparisons of the embeddings of n-grams from two texts and average their results as the overall intertextuality. Validation on four texts with known degrees of intertextuality, alongside a scalability test on 267 diverse texts, demonstrates the method's effectiveness and efficiency. Network analysis further reveals centrality and community structures, affirming the approach's success in capturing and quantifying intertextual relationships.
>
---
#### [new 036] Multimodal Reasoning for Science: Technical Report and 1st Place Solution to the ICML 2025 SeePhys Challenge
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出一种多模态推理框架，解决AI在视觉与文本跨模态任务中的性能不足问题。通过辅助图像描述进行推理，在ICML 2025 SeePhys挑战赛中取得第一名，并在MathVerse基准上验证了方法的通用性。**

- **链接: [http://arxiv.org/pdf/2509.06079v1](http://arxiv.org/pdf/2509.06079v1)**

> **作者:** Hao Liang; Ruitao Wu; Bohan Zeng; Junbo Niu; Wentao Zhang; Bin Dong
>
> **摘要:** Multimodal reasoning remains a fundamental challenge in artificial intelligence. Despite substantial advances in text-based reasoning, even state-of-the-art models such as GPT-o3 struggle to maintain strong performance in multimodal scenarios. To address this gap, we introduce a caption-assisted reasoning framework that effectively bridges visual and textual modalities. Our approach achieved 1st place in the ICML 2025 AI for Math Workshop \& Challenge 2: SeePhys, highlighting its effectiveness and robustness. Furthermore, we validate its generalization on the MathVerse benchmark for geometric reasoning, demonstrating the versatility of our method. Our code is publicly available at https://github.com/OpenDCAI/SciReasoner.
>
---
#### [new 037] Enhancing Factual Accuracy and Citation Generation in LLMs via Multi-Stage Self-Verification
- **分类: cs.CL**

- **简介: 该论文提出VeriFact-CoT方法，用于提升大语言模型在生成复杂内容时的事实准确性和引用能力。通过多阶段自我验证机制，解决模型幻觉和缺乏可信引用的问题，增强输出的可靠性，适用于科研、新闻等高精度场景。**

- **链接: [http://arxiv.org/pdf/2509.05741v1](http://arxiv.org/pdf/2509.05741v1)**

> **作者:** Fernando Gabriela García; Qiyang Shi; Zilin Feng
>
> **摘要:** This research introduces VeriFact-CoT (Verified Factual Chain-of-Thought), a novel method designed to address the pervasive issues of hallucination and the absence of credible citation sources in Large Language Models (LLMs) when generating complex, fact-sensitive content. By incorporating a multi-stage mechanism of 'fact verification-reflection-citation integration,' VeriFact-CoT empowers LLMs to critically self-examine and revise their intermediate reasoning steps and final answers. This process significantly enhances the objective accuracy, trustworthiness, and traceability of the generated outputs, making LLMs more reliable for applications demanding high fidelity such as scientific research, news reporting, and legal consultation.
>
---
#### [new 038] ZhiFangDanTai: Fine-tuning Graph-based Retrieval-Augmented Generation Model for Traditional Chinese Medicine Formula
- **分类: cs.CL; cs.AI**

- **简介: 论文提出ZhiFangDanTai框架，结合图检索增强生成与大模型微调，解决中医方剂生成中信息不全、解释不足的问题，提升模型生成的准确性和可解释性。**

- **链接: [http://arxiv.org/pdf/2509.05867v1](http://arxiv.org/pdf/2509.05867v1)**

> **作者:** ZiXuan Zhang; Bowen Hao; Yingjie Li; Hongzhi Yin
>
> **摘要:** Traditional Chinese Medicine (TCM) formulas play a significant role in treating epidemics and complex diseases. Existing models for TCM utilize traditional algorithms or deep learning techniques to analyze formula relationships, yet lack comprehensive results, such as complete formula compositions and detailed explanations. Although recent efforts have used TCM instruction datasets to fine-tune Large Language Models (LLMs) for explainable formula generation, existing datasets lack sufficient details, such as the roles of the formula's sovereign, minister, assistant, courier; efficacy; contraindications; tongue and pulse diagnosis-limiting the depth of model outputs. To address these challenges, we propose ZhiFangDanTai, a framework combining Graph-based Retrieval-Augmented Generation (GraphRAG) with LLM fine-tuning. ZhiFangDanTai uses GraphRAG to retrieve and synthesize structured TCM knowledge into concise summaries, while also constructing an enhanced instruction dataset to improve LLMs' ability to integrate retrieved information. Furthermore, we provide novel theoretical proofs demonstrating that integrating GraphRAG with fine-tuning techniques can reduce generalization error and hallucination rates in the TCM formula task. Experimental results on both collected and clinical datasets demonstrate that ZhiFangDanTai achieves significant improvements over state-of-the-art models. Our model is open-sourced at https://huggingface.co/tczzx6/ZhiFangDanTai1.0.
>
---
#### [new 039] PL-CA: A Parametric Legal Case Augmentation Framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PL-CA框架，解决传统RAG在法律领域的上下文压力和计算开销问题。通过参数化知识编码与LoRA集成，并构建多任务数据集，提升模型性能与泛化能力。属于法律文本增强与多任务学习任务。**

- **链接: [http://arxiv.org/pdf/2509.06356v1](http://arxiv.org/pdf/2509.06356v1)**

> **作者:** Ao Chang; Yubo Chen; Jun Zhao
>
> **摘要:** Conventional RAG is considered one of the most effective methods for addressing model knowledge insufficiency and hallucination, particularly in the judicial domain that requires high levels of knowledge rigor, logical consistency, and content integrity. However, the conventional RAG method only injects retrieved documents directly into the model's context, which severely constrains models due to their limited context windows and introduces additional computational overhead through excessively long contexts, thereby disrupting models' attention and degrading performance on downstream tasks. Moreover, many existing benchmarks lack expert annotation and focus solely on individual downstream tasks while real-world legal scenarios consist of multiple mixed legal tasks, indicating conventional benchmarks' inadequacy for reflecting models' true capabilities. To address these limitations, we propose PL-CA, which introduces a parametric RAG (P-RAG) framework to perform data augmentation on corpus knowledge and encode this legal knowledge into parametric vectors, and then integrates this parametric knowledge into the LLM's feed-forward networks (FFN) via LoRA, thereby alleviating models' context pressure. Additionally, we also construct a multi-task legal dataset comprising more than 2000 training and test instances, which are all expert-annotated and manually verified. We conduct our experiments on our dataset, and the experimental results demonstrate that our method reduces the overhead associated with excessively long contexts while maintaining competitive performance on downstream tasks compared to conventional RAG. Our code and dataset are provided in the appendix.
>
---
#### [new 040] Direct-Scoring NLG Evaluators Can Use Pairwise Comparisons Too
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出一种直接评分方法，利用合成摘要进行成对比较，解决自动评分模型无法给出绝对分数的问题。该方法在多个基准测试中表现接近最优，适用于需要阈值判断的场景。属于自然语言生成评估任务。**

- **链接: [http://arxiv.org/pdf/2509.05440v1](http://arxiv.org/pdf/2509.05440v1)**

> **作者:** Logan Lawrence; Ashton Williamson; Alexander Shelton
>
> **备注:** 12 pages, 18 tables, 1 figure
>
> **摘要:** As large-language models have been increasingly used as automatic raters for evaluating free-form content, including document summarization, dialog, and story generation, work has been dedicated to evaluating such models by measuring their correlations with human judgment. For \textit{sample-level} performance, methods which operate by using pairwise comparisons between machine-generated text perform well but often lack the ability to assign absolute scores to individual summaries, an ability crucial for use cases that require thresholding. In this work, we propose a direct-scoring method which uses synthetic summaries to act as pairwise machine rankings at test time. We show that our method performs comparably to state-of-the-art pairwise evaluators in terms of axis-averaged sample-level correlations on the SummEval (\textbf{+0.03}), TopicalChat (\textbf{-0.03}), and HANNA (\textbf{+0.05}) meta-evaluation benchmarks, and release the synthetic in-context summaries as data to facilitate future work.
>
---
#### [new 041] Domain-Aware RAG: MoL-Enhanced RL for Efficient Training and Scalable Retrieval
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出MoLER方法，解决RAG系统中粗排序阶段领域知识与查询增强的平衡问题。通过MoL强化学习和多查询融合策略，提升检索效率与性能，实现跨领域可扩展的高效检索。**

- **链接: [http://arxiv.org/pdf/2509.06650v1](http://arxiv.org/pdf/2509.06650v1)**

> **作者:** Hao Lin; Peitong Xie; Jingxue Chen; Jie Lin; Qingkun Tang; Qianchun Lu
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems rely heavily on the retrieval stage, particularly the coarse-ranking process. Existing coarse-ranking optimization approaches often struggle to balance domain-specific knowledge learning with query enhencement, resulting in suboptimal retrieval performance. To address this challenge, we propose MoLER, a domain-aware RAG method that uses MoL-Enhanced Reinforcement Learning to optimize retrieval. MoLER has a two-stage pipeline: a continual pre-training (CPT) phase using a Mixture of Losses (MoL) to balance domain-specific knowledge with general language capabilities, and a reinforcement learning (RL) phase leveraging Group Relative Policy Optimization (GRPO) to optimize query and passage generation for maximizing document recall. A key innovation is our Multi-query Single-passage Late Fusion (MSLF) strategy, which reduces computational overhead during RL training while maintaining scalable inference via Multi-query Multi-passage Late Fusion (MMLF). Extensive experiments on benchmark datasets show that MoLER achieves state-of-the-art performance, significantly outperforming baseline methods. MoLER bridges the knowledge gap in RAG systems, enabling robust and scalable retrieval in specialized domains.
>
---
#### [new 042] KatotohananQA: Evaluating Truthfulness of Large Language Models in Filipino
- **分类: cs.CL**

- **简介: 该论文属于评估大语言模型真实性的任务，旨在解决低资源语言（如菲律宾语）缺乏相关基准的问题。研究者翻译了TruthfulQA基准为菲律宾语（KatotohananQA），并测试多个模型，发现英菲表现有差距，新OpenAI模型在多语言上更稳健。**

- **链接: [http://arxiv.org/pdf/2509.06065v1](http://arxiv.org/pdf/2509.06065v1)**

> **作者:** Lorenzo Alfred Nery; Ronald Dawson Catignas; Thomas James Tiam-Lee
>
> **备注:** 14 pages, 1 figure, 9 tables, 1 listing. To appear in Proceedings of NLPIR 2025
>
> **摘要:** Large Language Models (LLMs) achieve remarkable performance across various tasks, but their tendency to produce hallucinations limits reliable adoption. Benchmarks such as TruthfulQA have been developed to measure truthfulness, yet they are primarily available in English, leaving a gap in evaluating LLMs in low-resource languages. To address this, we present KatotohananQA, a Filipino translation of the TruthfulQA benchmark. Seven free-tier proprietary models were assessed using a binary-choice framework. Findings show a significant performance gap between English and Filipino truthfulness, with newer OpenAI models (GPT-5 and GPT-5 mini) demonstrating strong multilingual robustness. Results also reveal disparities across question characteristics, suggesting that some question types, categories, and topics are less robust to multilingual transfer which highlight the need for broader multilingual evaluation to ensure fairness and reliability in LLM usage.
>
---
#### [new 043] LM-Searcher: Cross-domain Neural Architecture Search with LLMs via Unified Numerical Encoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LM-Searcher，利用大语言模型进行跨领域神经架构搜索。通过NCode统一编码架构，并将NAS转化为排序任务，实现无需领域适配的高效搜索。解决了现有LLM-NAS依赖提示工程和领域调优的问题。**

- **链接: [http://arxiv.org/pdf/2509.05657v1](http://arxiv.org/pdf/2509.05657v1)**

> **作者:** Yuxuan Hu; Jihao Liu; Ke Wang; Jinliang Zhen; Weikang Shi; Manyuan Zhang; Qi Dou; Rui Liu; Aojun Zhou; Hongsheng Li
>
> **备注:** EMNLP2025
>
> **摘要:** Recent progress in Large Language Models (LLMs) has opened new avenues for solving complex optimization problems, including Neural Architecture Search (NAS). However, existing LLM-driven NAS approaches rely heavily on prompt engineering and domain-specific tuning, limiting their practicality and scalability across diverse tasks. In this work, we propose LM-Searcher, a novel framework that leverages LLMs for cross-domain neural architecture optimization without the need for extensive domain-specific adaptation. Central to our approach is NCode, a universal numerical string representation for neural architectures, which enables cross-domain architecture encoding and search. We also reformulate the NAS problem as a ranking task, training LLMs to select high-performing architectures from candidate pools using instruction-tuning samples derived from a novel pruning-based subspace sampling strategy. Our curated dataset, encompassing a wide range of architecture-performance pairs, encourages robust and transferable learning. Comprehensive experiments demonstrate that LM-Searcher achieves competitive performance in both in-domain (e.g., CNNs for image classification) and out-of-domain (e.g., LoRA configurations for segmentation and generation) tasks, establishing a new paradigm for flexible and generalizable LLM-based architecture search. The datasets and models will be released at https://github.com/Ashone3/LM-Searcher.
>
---
#### [new 044] Exploring Subjective Tasks in Farsi: A Survey Analysis and Evaluation of Language Models
- **分类: cs.CL**

- **简介: 该论文研究波斯语主观任务（情感分析、情绪分析和毒性检测）的挑战。论文分析了数据不足和质量差的问题，评估了现有模型表现不稳定，指出数据量不足以提升NLP效果。**

- **链接: [http://arxiv.org/pdf/2509.05719v1](http://arxiv.org/pdf/2509.05719v1)**

> **作者:** Donya Rooein; Flor Miriam Plaza-del-Arco; Debora Nozza; Dirk Hovy
>
> **摘要:** Given Farsi's speaker base of over 127 million people and the growing availability of digital text, including more than 1.3 million articles on Wikipedia, it is considered a middle-resource language. However, this label quickly crumbles when the situation is examined more closely. We focus on three subjective tasks (Sentiment Analysis, Emotion Analysis, and Toxicity Detection) and find significant challenges in data availability and quality, despite the overall increase in data availability. We review 110 publications on subjective tasks in Farsi and observe a lack of publicly available datasets. Furthermore, existing datasets often lack essential demographic factors, such as age and gender, that are crucial for accurately modeling subjectivity in language. When evaluating prediction models using the few available datasets, the results are highly unstable across both datasets and models. Our findings indicate that the volume of data is insufficient to significantly improve a language's prospects in NLP.
>
---
#### [new 045] A Lightweight Framework for Trigger-Guided LoRA-Based Self-Adaptation in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 论文提出SAGE框架，解决大语言模型在推理时无法动态适应新数据的问题。通过分解任务、触发模块检测失败、缓冲聚类异常样本，并利用LoRA动态优化参数，实现推理过程中的自适应更新。**

- **链接: [http://arxiv.org/pdf/2509.05385v1](http://arxiv.org/pdf/2509.05385v1)**

> **作者:** Jiacheng Wei; Faguo Wu; Xiao Zhang
>
> **备注:** 11 pages, 7 figures, conference
>
> **摘要:** Large language models are unable to continuously adapt and learn from new data during reasoning at inference time. To address this limitation, we propose that complex reasoning tasks be decomposed into atomic subtasks and introduce SAGE, a trigger-guided dynamic fine-tuning framework that enables adaptive updates during reasoning at inference time. SAGE consists of three key components: (1) a Trigger module that detects reasoning failures through multiple evaluation metrics in real time; (2) a Trigger Buffer module that clusters anomaly samples using a streaming clustering process with HDBSCAN, followed by stability checks and similarity-based merging; and (3) a Lora Store module that dynamically optimizes parameter updates with an adapter pool for knowledge retention. Evaluation results show that SAGE demonstrates excellent accuracy, robustness, and stability on the atomic reasoning subtask through dynamic knowledge updating during test time.
>
---
#### [new 046] COMPACT: Common-token Optimized Model Pruning Across Channels and Tokens
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出COMPACT方法，解决大语言模型高效部署问题。通过联合剪枝词汇表和FFN通道，保持标准架构，实现参数、内存和延迟的显著减少，提升推理效率。**

- **链接: [http://arxiv.org/pdf/2509.06836v1](http://arxiv.org/pdf/2509.06836v1)**

> **作者:** Eugene Kwek; Wenpeng Yin
>
> **摘要:** Making LLMs more efficient in memory, latency, and serving cost is crucial for edge deployment, interactive applications, and sustainable inference at scale. Pruning is a key technique toward this goal. However, prior pruning methods are limited: width pruning often breaks the standard transformer layout or requires custom inference code, while depth pruning removes entire layers and can cause abrupt accuracy drops. In this work, we propose COMPACT, which jointly (i) prunes rare vocabulary to shrink embedding/unembedding and (ii) prunes FFN intermediate channels using common-token-weighted activations, aligning importance with the post-pruning token distribution. COMPACT enjoys merits of both depth and width pruning, such as: deployment-friendliness (keeps a standard transformer architecture), scale-adaptivity (trade off vocab vs. FFN pruning), training-free operation with competitive pruning time, and strong memory savings alongside throughput gains. Experiments across Qwen, LLaMA, and Gemma families (0.5B-70B) show state-of-the-art downstream task performance at similar or higher pruning ratios, with substantial reductions in parameters, GPU memory, and end-to-end latency.
>
---
#### [new 047] Benchmarking Gender and Political Bias in Large Language Models
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 论文提出EuroParlVote基准，评估大语言模型在政治敏感场景中的性别和政治偏见。通过MEP辩论与投票数据，发现模型存在性别分类错误和政治立场偏差问题，并比较了不同模型的公平性表现。**

- **链接: [http://arxiv.org/pdf/2509.06164v1](http://arxiv.org/pdf/2509.06164v1)**

> **作者:** Jinrui Yang; Xudong Han; Timothy Baldwin
>
> **备注:** The 8th International Conference on Natural Language and Speech Processing (Oral)
>
> **摘要:** We introduce EuroParlVote, a novel benchmark for evaluating large language models (LLMs) in politically sensitive contexts. It links European Parliament debate speeches to roll-call vote outcomes and includes rich demographic metadata for each Member of the European Parliament (MEP), such as gender, age, country, and political group. Using EuroParlVote, we evaluate state-of-the-art LLMs on two tasks -- gender classification and vote prediction -- revealing consistent patterns of bias. We find that LLMs frequently misclassify female MEPs as male and demonstrate reduced accuracy when simulating votes for female speakers. Politically, LLMs tend to favor centrist groups while underperforming on both far-left and far-right ones. Proprietary models like GPT-4o outperform open-weight alternatives in terms of both robustness and fairness. We release the EuroParlVote dataset, code, and demo to support future research on fairness and accountability in NLP within political contexts.
>
---
#### [new 048] New Insights into Optimal Alignment of Acoustic and Linguistic Representations for Knowledge Transfer in ASR
- **分类: cs.CL; cs.LG**

- **简介: 论文研究ASR中声学与语言表征的对齐问题，提出基于非平衡最优传输的模型，解决结构不对称与分布不匹配问题，提升知识迁移效果。属于语音识别中的对齐与知识迁移任务。**

- **链接: [http://arxiv.org/pdf/2509.05609v1](http://arxiv.org/pdf/2509.05609v1)**

> **作者:** Xugang Lu; Peng Shen; Yu Tsao; Hisashi Kawai
>
> **摘要:** Aligning acoustic and linguistic representations is a central challenge to bridge the pre-trained models in knowledge transfer for automatic speech recognition (ASR). This alignment is inherently structured and asymmetric: while multiple consecutive acoustic frames typically correspond to a single linguistic token (many-to-one), certain acoustic transition regions may relate to multiple adjacent tokens (one-to-many). Moreover, acoustic sequences often include frames with no linguistic counterpart, such as background noise or silence may lead to imbalanced matching conditions. In this work, we take a new insight to regard alignment and matching as a detection problem, where the goal is to identify meaningful correspondences with high precision and recall ensuring full coverage of linguistic tokens while flexibly handling redundant or noisy acoustic frames in transferring linguistic knowledge for ASR. Based on this new insight, we propose an unbalanced optimal transport-based alignment model that explicitly handles distributional mismatch and structural asymmetries with soft and partial matching between acoustic and linguistic modalities. Our method ensures that every linguistic token is grounded in at least one acoustic observation, while allowing for flexible, probabilistic mappings from acoustic to linguistic units. We evaluate our proposed model with experiments on an CTC-based ASR system with a pre-trained language model for knowledge transfer. Experimental results demonstrate the effectiveness of our approach in flexibly controlling degree of matching and hence to improve ASR performance.
>
---
#### [new 049] mmBERT: A Modern Multilingual Encoder with Annealed Language Learning
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文提出mmBERT，一种用于多语言分类和检索任务的编码器模型。针对多语言模型研究不足的问题，通过引入逆掩码比例调度和逆温度采样等方法，在1800多种语言上预训练，显著提升低资源语言性能。**

- **链接: [http://arxiv.org/pdf/2509.06888v1](http://arxiv.org/pdf/2509.06888v1)**

> **作者:** Marc Marone; Orion Weller; William Fleshman; Eugene Yang; Dawn Lawrie; Benjamin Van Durme
>
> **摘要:** Encoder-only languages models are frequently used for a variety of standard machine learning tasks, including classification and retrieval. However, there has been a lack of recent research for encoder models, especially with respect to multilingual models. We introduce mmBERT, an encoder-only language model pretrained on 3T tokens of multilingual text in over 1800 languages. To build mmBERT we introduce several novel elements, including an inverse mask ratio schedule and an inverse temperature sampling ratio. We add over 1700 low-resource languages to the data mix only during the decay phase, showing that it boosts performance dramatically and maximizes the gains from the relatively small amount of training data. Despite only including these low-resource languages in the short decay phase we achieve similar classification performance to models like OpenAI's o3 and Google's Gemini 2.5 Pro. Overall, we show that mmBERT significantly outperforms the previous generation of models on classification and retrieval tasks -- on both high and low-resource languages.
>
---
#### [new 050] Understanding the Influence of Synthetic Data for Text Embedders
- **分类: cs.CL**

- **简介: 该论文研究合成数据对文本嵌入模型的影响。任务是分析合成数据在提升模型泛化能力中的作用。论文复现并公开了Mistral-E5的合成数据，发现其效果有限且存在任务间的性能权衡，指出当前合成数据方法的局限性。**

- **链接: [http://arxiv.org/pdf/2509.06184v1](http://arxiv.org/pdf/2509.06184v1)**

> **作者:** Jacob Mitchell Springer; Vaibhav Adlakha; Siva Reddy; Aditi Raghunathan; Marius Mosbach
>
> **备注:** ACL Findings 2025
>
> **摘要:** Recent progress in developing general purpose text embedders has been driven by training on ever-growing corpora of synthetic LLM-generated data. Nonetheless, no publicly available synthetic dataset exists, posing a barrier to studying its role for generalization. To address this issue, we first reproduce and publicly release the synthetic data proposed by Wang et al. (Mistral-E5). Our synthetic data is high quality and leads to consistent improvements in performance. Next, we critically examine where exactly synthetic data improves model generalization. Our analysis reveals that benefits from synthetic data are sparse and highly localized to individual datasets. Moreover, we observe trade-offs between the performance on different categories and data that benefits one task, degrades performance on another. Our findings highlight the limitations of current synthetic data approaches for building general-purpose embedders and challenge the notion that training on synthetic data leads to more robust embedding models across tasks.
>
---
#### [new 051] Accelerating Large Language Model Inference via Early-Exiting Algorithms
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理中的计算成本问题。通过设计并行解码机制、深度参数共享和预训练轻量路由器，实现动态计算与效率的平衡，提升批量推理吞吐量。**

- **链接: [http://arxiv.org/pdf/2509.05915v1](http://arxiv.org/pdf/2509.05915v1)**

> **作者:** Sangmin Bae
>
> **备注:** PhD Dissertation
>
> **摘要:** Large language models have achieved remarkable capabilities, but their practical deployment is hindered by significant computational costs. While adaptive computation methods like early-exiting promise to reduce these costs, they introduce a fundamental conflict: the per-token dynamism intended to save computation often creates system-level bottlenecks that can paradoxically reduce throughput in batched inference. This dissertation resolves this conflict by co-designing adaptive algorithms and model architectures to strike an optimal balance between dynamism and efficiency. To this end, our work first addresses critical sources of overhead in conventional early-exiting by proposing an efficient parallel decoding mechanism. We then show that deep parameter sharing provides an architectural foundation that not only yields compact, parameter-efficient models but also inherently mitigates the critical synchronization issues affecting dynamic inference. Finally, this work presents a unified framework where lightweight routers are pretrained to dynamically assign an optimal recursion depth for each token. This approach establishes a new Pareto frontier between efficiency and performance by effectively optimizing for both adaptive computation and parameter efficiency within a single model.
>
---
#### [new 052] A Comparative Benchmark of Large Language Models for Labelling Wind Turbine Maintenance Logs
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在风力涡轮机维护日志分类任务中的表现，旨在解决非结构化文本自动化分析难题。通过构建开源基准框架，评估多种LLM的可靠性与效率，发现其性能受语义模糊性影响，并建议采用人机协作方式提升数据标注质量。**

- **链接: [http://arxiv.org/pdf/2509.06813v1](http://arxiv.org/pdf/2509.06813v1)**

> **作者:** Max Malyi; Jonathan Shek; Alasdair McDonald; Andre Biscaya
>
> **备注:** Associated GitHub repository: https://github.com/mvmalyi/wind-farm-maintenance-logs-labelling-with-llms
>
> **摘要:** Effective Operation and Maintenance (O&M) is critical to reducing the Levelised Cost of Energy (LCOE) from wind power, yet the unstructured, free-text nature of turbine maintenance logs presents a significant barrier to automated analysis. Our paper addresses this by presenting a novel and reproducible framework for benchmarking Large Language Models (LLMs) on the task of classifying these complex industrial records. To promote transparency and encourage further research, this framework has been made publicly available as an open-source tool. We systematically evaluate a diverse suite of state-of-the-art proprietary and open-source LLMs, providing a foundational assessment of their trade-offs in reliability, operational efficiency, and model calibration. Our results quantify a clear performance hierarchy, identifying top models that exhibit high alignment with a benchmark standard and trustworthy, well-calibrated confidence scores. We also demonstrate that classification performance is highly dependent on the task's semantic ambiguity, with all models showing higher consensus on objective component identification than on interpretive maintenance actions. Given that no model achieves perfect accuracy and that calibration varies dramatically, we conclude that the most effective and responsible near-term application is a Human-in-the-Loop system, where LLMs act as a powerful assistant to accelerate and standardise data labelling for human experts, thereby enhancing O&M data quality and downstream reliability analysis.
>
---
#### [new 053] Let's Roleplay: Examining LLM Alignment in Collaborative Dialogues
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLM在协作对话中的对齐问题，提出摩擦代理干预方法，通过角色扮演评估不同对齐策略的效果，设计反事实框架量化干预影响，提升协作共识与任务正确性。属于多智能体协作对齐任务。**

- **链接: [http://arxiv.org/pdf/2509.05882v1](http://arxiv.org/pdf/2509.05882v1)**

> **作者:** Abhijnan Nath; Carine Graff; Nikhil Krishnaswamy
>
> **摘要:** As Large Language Models (LLMs) integrate into diverse workflows, they are increasingly being considered "collaborators" with humans. If such AI collaborators are to be reliable, their behavior over multiturn interactions must be predictable, validated and verified before deployment. Common alignment techniques are typically developed under simplified single-user settings and do not account for the dynamics of long-horizon multiparty interactions. This paper examines how different alignment methods affect LLM agents' effectiveness as partners in multiturn, multiparty collaborations. We study this question through the lens of friction agents that intervene in group dialogues to encourage the collaborative group to slow down and reflect upon their reasoning for deliberative decision-making. Using a roleplay methodology, we evaluate interventions from differently-trained friction agents in collaborative task conversations. We propose a novel counterfactual evaluation framework that quantifies how friction interventions change the trajectory of group collaboration and belief alignment. Our results show that a friction-aware approach significantly outperforms common alignment baselines in helping both convergence to a common ground, or agreed-upon task-relevant propositions, and correctness of task outcomes.
>
---
#### [new 054] Revealing the Numeracy Gap: An Empirical Investigation of Text Embedding Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究文本嵌入模型对数值信息的编码能力，属于自然语言处理任务。论文探讨模型是否能准确捕捉文本中的数值细节，并通过金融场景的合成数据评估13种模型，发现其在处理数值信息时表现不佳，为改进模型提供参考。**

- **链接: [http://arxiv.org/pdf/2509.05691v1](http://arxiv.org/pdf/2509.05691v1)**

> **作者:** Ningyuan Deng; Hanyu Duan; Yixuan Tang; Yi Yang
>
> **摘要:** Text embedding models are widely used in natural language processing applications. However, their capability is often benchmarked on tasks that do not require understanding nuanced numerical information in text. As a result, it remains unclear whether current embedding models can precisely encode numerical content, such as numbers, into embeddings. This question is critical because embedding models are increasingly applied in domains where numbers matter, such as finance and healthcare. For example, Company X's market share grew by 2\% should be interpreted very differently from Company X's market share grew by 20\%, even though both indicate growth in market share. This study aims to examine whether text embedding models can capture such nuances. Using synthetic data in a financial context, we evaluate 13 widely used text embedding models and find that they generally struggle to capture numerical details accurately. Our further analyses provide deeper insights into embedding numeracy, informing future research to strengthen embedding model-based NLP systems with improved capacity for handling numerical content.
>
---
#### [new 055] Using Contrastive Learning to Improve Two-Way Reasoning in Large Language Models: The Obfuscation Task as a Case Study
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型的双向推理能力，提出对比微调（CFT）方法解决单向训练导致的认知特化问题，通过正例、负例和混淆例训练，提升模型正向与反向任务表现，验证其对真正理解的促进作用。**

- **链接: [http://arxiv.org/pdf/2509.05553v1](http://arxiv.org/pdf/2509.05553v1)**

> **作者:** Serge Lionel Nikiema; Jordan Samhi; Micheline Bénédicte Moumoula; Albérick Euraste Djiré; Abdoul Kader Kaboré; Jacques Klein; Tegawendé F. Bissyandé
>
> **摘要:** This research addresses a fundamental question in AI: whether large language models truly understand concepts or simply recognize patterns. The authors propose bidirectional reasoning,the ability to apply transformations in both directions without being explicitly trained on the reverse direction, as a test for genuine understanding. They argue that true comprehension should naturally allow reversibility. For example, a model that can change a variable name like userIndex to i should also be able to infer that i represents a user index without reverse training. The researchers tested current language models and discovered what they term cognitive specialization: when models are fine-tuned on forward tasks, their performance on those tasks improves, but their ability to reason bidirectionally becomes significantly worse. To address this issue, they developed Contrastive Fine-Tuning (CFT), which trains models using three types of examples: positive examples that maintain semantic meaning, negative examples with different semantics, and forward-direction obfuscation examples. This approach aims to develop deeper understanding rather than surface-level pattern recognition and allows reverse capabilities to develop naturally without explicit reverse training. Their experiments demonstrated that CFT successfully achieved bidirectional reasoning, enabling strong reverse performance while maintaining forward task capabilities. The authors conclude that bidirectional reasoning serves both as a theoretical framework for assessing genuine understanding and as a practical training approach for developing more capable AI systems.
>
---
#### [new 056] Guided Decoding and Its Critical Role in Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文研究指导解码在检索增强生成（RAG）中的作用，比较三种方法在不同多轮提示下的表现，旨在提高输出结构化和减少幻觉。属于自然语言生成任务，解决RAG系统中输出格式控制与可靠性问题。**

- **链接: [http://arxiv.org/pdf/2509.06631v1](http://arxiv.org/pdf/2509.06631v1)**

> **作者:** Özgür Uğur; Musa Yılmaz; Esra Şavirdi; Özay Ezerceli; Mahmut El Huseyni; Selva Taş; Reyhan Bayraktar
>
> **摘要:** The integration of Large Language Models (LLMs) into various applications has driven the need for structured and reliable responses. A key challenge in Retrieval-Augmented Generation (RAG) systems is ensuring that outputs align with expected formats while minimizing hallucinations. This study examines the role of guided decoding in RAG systems, comparing three methods, Outlines, XGrammar, and LM Format Enforcer, across different multi-turn prompting setups (0-turn, 1-turn, and 2-turn). By evaluating success rates, hallucination rates, and output quality, we provide insights into their performance and applicability. Our findings reveal how multi-turn interactions influence guided decoding, uncovering unexpected performance variations that can inform method selection for specific use cases. This work advances the understanding of structured output generation in RAG systems, offering both theoretical insights and practical guidance for LLM deployment.
>
---
#### [new 057] WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents
- **分类: cs.CL**

- **简介: 该论文提出WebExplorer，解决长时域网络代理信息检索难题。通过生成复杂查询数据集，训练出高性能WebExplorer-8B模型，在多个基准测试中取得最优表现。**

- **链接: [http://arxiv.org/pdf/2509.06501v1](http://arxiv.org/pdf/2509.06501v1)**

> **作者:** Junteng Liu; Yunji Li; Chi Zhang; Jingyang Li; Aili Chen; Ke Ji; Weiyu Cheng; Zijia Wu; Chengyu Du; Qidi Xu; Jiayuan Song; Zhengmao Zhu; Wenhu Chen; Pengyu Zhao; Junxian He
>
> **摘要:** The paradigm of Large Language Models (LLMs) has increasingly shifted toward agentic applications, where web browsing capabilities are fundamental for retrieving information from diverse online sources. However, existing open-source web agents either demonstrate limited information-seeking abilities on complex tasks or lack transparent implementations. In this work, we identify that the key challenge lies in the scarcity of challenging data for information seeking. To address this limitation, we introduce WebExplorer: a systematic data generation approach using model-based exploration and iterative, long-to-short query evolution. This method creates challenging query-answer pairs that require multi-step reasoning and complex web navigation. By leveraging our curated high-quality dataset, we successfully develop advanced web agent WebExplorer-8B through supervised fine-tuning followed by reinforcement learning. Our model supports 128K context length and up to 100 tool calling turns, enabling long-horizon problem solving. Across diverse information-seeking benchmarks, WebExplorer-8B achieves the state-of-the-art performance at its scale. Notably, as an 8B-sized model, WebExplorer-8B is able to effectively search over an average of 16 turns after RL training, achieving higher accuracy than WebSailor-72B on BrowseComp-en/zh and attaining the best performance among models up to 100B parameters on WebWalkerQA and FRAMES. Beyond these information-seeking tasks, our model also achieves strong generalization on the HLE benchmark even though it is only trained on knowledge-intensive QA data. These results highlight our approach as a practical path toward long-horizon web agents.
>
---
#### [new 058] UNH at CheckThat! 2025: Fine-tuning Vs Prompting in Claim Extraction
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文参与CheckThat! 2025任务2，研究从社交媒体文本中提取可核查声明的方法。对比了提示和微调方法，发现微调FLAN-T5模型取得最佳METEOR分数，但其他方法有时能生成更高质量声明。属于自然语言处理中的事实核查任务。**

- **链接: [http://arxiv.org/pdf/2509.06883v1](http://arxiv.org/pdf/2509.06883v1)**

> **作者:** Joe Wilder; Nikhil Kadapala; Benji Xu; Mohammed Alsaadi; Aiden Parsons; Mitchell Rogers; Palash Agarwal; Adam Hassick; Laura Dietz
>
> **备注:** 16 pages,3 tables, CLEF 2025 Working Notes, 9-12 September 2025, Madrid, Spain
>
> **摘要:** We participate in CheckThat! Task 2 English and explore various methods of prompting and in-context learning, including few-shot prompting and fine-tuning with different LLM families, with the goal of extracting check-worthy claims from social media passages. Our best METEOR score is achieved by fine-tuning a FLAN-T5 model. However, we observe that higher-quality claims can sometimes be extracted using other methods, even when their METEOR scores are lower.
>
---
#### [new 059] Few-Shot Query Intent Detection via Relation-Aware Prompt Learning
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于意图识别任务，解决少样本场景下忽视对话结构信息的问题。提出SAID框架，融合文本与关系结构信息，并设计QueryAdapt机制，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2509.05635v1](http://arxiv.org/pdf/2509.05635v1)**

> **作者:** Liang Zhang; Yuan Li; Shijie Zhang; Zheng Zhang; Xitong Li
>
> **摘要:** Intent detection is a crucial component of modern conversational systems, since accurately identifying user intent at the beginning of a conversation is essential for generating effective responses. Recent efforts have focused on studying this problem under a challenging few-shot scenario. These approaches primarily leverage large-scale unlabeled dialogue text corpora to pretrain language models through various pretext tasks, followed by fine-tuning for intent detection with very limited annotations. Despite the improvements achieved, existing methods have predominantly focused on textual data, neglecting to effectively capture the crucial structural information inherent in conversational systems, such as the query-query relation and query-answer relation. To address this gap, we propose SAID, a novel framework that integrates both textual and relational structure information in a unified manner for model pretraining for the first time. Building on this framework, we further propose a novel mechanism, the query-adaptive attention network (QueryAdapt), which operates at the relation token level by generating intent-specific relation tokens from well-learned query-query and query-answer relations explicitly, enabling more fine-grained knowledge transfer. Extensive experimental results on two real-world datasets demonstrate that SAID significantly outperforms state-of-the-art methods.
>
---
#### [new 060] Mitigating Spurious Correlations Between Question and Answer via Chain-of-Thought Correctness Perception Distillation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的推理任务，旨在解决小模型从大模型生成的链式思维数据中学习时产生的虚假相关性问题。提出CoPeD方法，通过正确性感知任务设置和加权损失优化，提升小模型的推理质量。**

- **链接: [http://arxiv.org/pdf/2509.05602v1](http://arxiv.org/pdf/2509.05602v1)**

> **作者:** Hongyan Xie; Yitong Yao; Yikun Ban; Zixuan Huang; Deqing Wang; Zhenhe Wu; Haoxiang Su; Chao Wang; Shuangyong Song; Xuelong Li
>
> **备注:** PrePrint
>
> **摘要:** Large language models (LLMs) excel at reasoning tasks but are expensive to deploy. Thus small language models (SLMs) are fine-tuned on CoT data generated by LLMs to copy LLMs' abilities. However, these CoT data may include noisy rationales that either fail to substantiate the answers or contribute no additional information to support answer prediction, which leads SLMs to capture spurious correlations between questions and answers and compromise the quality of reasoning. In this work, we propose Chain-of-Thought Correctness Perception Distillation (CoPeD), which aims to improve the reasoning quality of the student model from the perspectives of task setting and data utilization. Firstly, we introduce a correctness-aware task setting that encourages the student model to predict answers based on correct rationales and revise them when they are incorrect. This setting improves the faithfulness of reasoning and allows the model to learn from its mistakes. Then, we propose a Correctness-Aware Weighted loss, which dynamically adjusts the contribution of each training instance based on the combined loss of the rationale and the answer. This strategy encourages the model to focus more on samples where the rationale offers stronger support for the correct answer. Experiments have shown that CoPeD is effective on both in-distribution (IND) and out-of-distribution (OOD) benchmark reasoning datasets.
>
---
#### [new 061] The Token Tax: Systematic Bias in Multilingual Tokenization
- **分类: cs.CL; cs.AI**

- **简介: 论文研究多语言分词中的系统性偏差问题，分析分词效率对低资源语言的影响。通过评估10个大模型在AfriMMLU数据集上的表现，发现分词数量与准确率负相关，并提出改进分词策略、公平定价和多语言基准以促进NLP公平性。**

- **链接: [http://arxiv.org/pdf/2509.05486v1](http://arxiv.org/pdf/2509.05486v1)**

> **作者:** Jessica M. Lundin; Ada Zhang; Nihal Karim; Hamza Louzan; Victor Wei; David Adelani; Cody Carroll
>
> **摘要:** Tokenization inefficiency imposes structural disadvantages on morphologically complex, low-resource languages, inflating compute resources and depressing accuracy. We evaluate 10 large language models (LLMs) on AfriMMLU (9,000 MCQA items; 5 subjects; 16 African languages) and show that fertility (tokens/word) reliably predicts accuracy. Higher fertility consistently predicts lower accuracy across all models and subjects. We further find that reasoning models (DeepSeek, o1) consistently outperform non-reasoning peers across high and low resource languages in the AfriMMLU dataset, narrowing accuracy gaps observed in prior generations. Finally, translating token inflation to economics, a doubling in tokens results in quadrupled training cost and time, underscoring the token tax faced by many languages. These results motivate morphologically aware tokenization, fair pricing, and multilingual benchmarks for equitable natural language processing (NLP).
>
---
#### [new 062] Will Annotators Disagree? Identifying Subjectivity in Value-Laden Arguments
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在识别论证中的人类价值观主观性。通过对比两种方法，提出直接识别主观性的方法更有效，并减少对标签主观性的依赖，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2509.06704v1](http://arxiv.org/pdf/2509.06704v1)**

> **作者:** Amir Homayounirad; Enrico Liscio; Tong Wang; Catholijn M. Jonker; Luciano C. Siebert
>
> **备注:** Accepted at Findings of EMNLP 2025
>
> **摘要:** Aggregating multiple annotations into a single ground truth label may hide valuable insights into annotator disagreement, particularly in tasks where subjectivity plays a crucial role. In this work, we explore methods for identifying subjectivity in recognizing the human values that motivate arguments. We evaluate two main approaches: inferring subjectivity through value prediction vs. directly identifying subjectivity. Our experiments show that direct subjectivity identification significantly improves the model performance of flagging subjective arguments. Furthermore, combining contrastive loss with binary cross-entropy loss does not improve performance but reduces the dependency on per-label subjectivity. Our proposed methods can help identify arguments that individuals may interpret differently, fostering a more nuanced annotation process.
>
---
#### [new 063] Anchoring Refusal Direction: Mitigating Safety Risks in Tuning via Projection Constraint
- **分类: cs.CL**

- **简介: 论文提出ProCon方法，通过投影约束损失缓解指令微调中拒绝方向漂移带来的安全风险，在保持任务性能的同时提升模型安全性。属于大语言模型安全优化任务。**

- **链接: [http://arxiv.org/pdf/2509.06795v1](http://arxiv.org/pdf/2509.06795v1)**

> **作者:** Yanrui Du; Fenglei Fan; Sendong Zhao; Jiawei Cao; Qika Lin; Kai He; Ting Liu; Bing Qin; Mengling Feng
>
> **摘要:** Instruction Fine-Tuning (IFT) has been widely adopted as an effective post-training strategy to enhance various abilities of Large Language Models (LLMs). However, prior studies have shown that IFT can significantly compromise LLMs' safety, particularly their ability to refuse malicious instructions, raising significant concerns. Recent research into the internal mechanisms of LLMs has identified the refusal direction (r-direction) in the hidden states, which plays a pivotal role in governing refusal behavior. Building on this insight, our study reveals that the r-direction tends to drift during training, which we identify as one of the causes of the associated safety risks. To mitigate such drift, our proposed ProCon method introduces a projection-constrained loss term that regularizes the projection magnitude of each training sample's hidden state onto the r-direction. Our initial analysis shows that applying an appropriate constraint can effectively mitigate the refusal direction drift and associated safety risks, but remains limited by overall performance barriers. To overcome this barrier, informed by our observation of early-stage sharp drift and a data-driven perspective, we introduce a warm-up strategy that emphasizes early-stage strong constraints and broaden the data distribution to strengthen constraint signals, leading to an enhanced ProCon method. Experimental results under various datasets, scenarios, and LLMs demonstrate that our method can significantly mitigate safety risks posed by IFT while preserving task performance gains. Even compared with strong baselines, our method consistently delivers superior overall performance. Crucially, our analysis indicates that ProCon can contribute to stabilizing the r-direction during training, while such an interpretability-driven exploration of LLMs' internal mechanisms lays a solid foundation for future safety research.
>
---
#### [new 064] Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文研究多智能体辩论中的失败模式，属于AI推理优化任务。论文发现辩论可能降低准确性，尤其当强模型未被激励去反驳错误推理时。通过实验分析，揭示了模型倾向于同意而非挑战错误观点的问题。**

- **链接: [http://arxiv.org/pdf/2509.05396v1](http://arxiv.org/pdf/2509.05396v1)**

> **作者:** Andrea Wynn; Harsh Satija; Gillian Hadfield
>
> **备注:** ICML MAS Workshop 2025
>
> **摘要:** While multi-agent debate has been proposed as a promising strategy for improving AI reasoning ability, we find that debate can sometimes be harmful rather than helpful. The prior work has exclusively focused on debates within homogeneous groups of agents, whereas we explore how diversity in model capabilities influences the dynamics and outcomes of multi-agent interactions. Through a series of experiments, we demonstrate that debate can lead to a decrease in accuracy over time -- even in settings where stronger (i.e., more capable) models outnumber their weaker counterparts. Our analysis reveals that models frequently shift from correct to incorrect answers in response to peer reasoning, favoring agreement over challenging flawed reasoning. These results highlight important failure modes in the exchange of reasons during multi-agent debate, suggesting that naive applications of debate may cause performance degradation when agents are neither incentivized nor adequately equipped to resist persuasive but incorrect reasoning.
>
---
#### [new 065] IntrEx: A Dataset for Modeling Engagement in Educational Conversations
- **分类: cs.CL**

- **简介: 论文提出IntrEx数据集，用于研究教育对话中的参与度。任务是识别影响学习者兴趣的语言特征。通过标注对话序列，分析LLM预测兴趣的能力，并探讨语言和认知因素对参与度的影响。**

- **链接: [http://arxiv.org/pdf/2509.06652v1](http://arxiv.org/pdf/2509.06652v1)**

> **作者:** Xingwei Tan; Mahathi Parvatham; Chiara Gambi; Gabriele Pergola
>
> **备注:** EMNLP 2025 Findings camera-ready, 9+7 pages
>
> **摘要:** Engagement and motivation are crucial for second-language acquisition, yet maintaining learner interest in educational conversations remains a challenge. While prior research has explored what makes educational texts interesting, still little is known about the linguistic features that drive engagement in conversations. To address this gap, we introduce IntrEx, the first large dataset annotated for interestingness and expected interestingness in teacher-student interactions. Built upon the Teacher-Student Chatroom Corpus (TSCC), IntrEx extends prior work by incorporating sequence-level annotations, allowing for the study of engagement beyond isolated turns to capture how interest evolves over extended dialogues. We employ a rigorous annotation process with over 100 second-language learners, using a comparison-based rating approach inspired by reinforcement learning from human feedback (RLHF) to improve agreement. We investigate whether large language models (LLMs) can predict human interestingness judgments. We find that LLMs (7B/8B parameters) fine-tuned on interestingness ratings outperform larger proprietary models like GPT-4o, demonstrating the potential for specialised datasets to model engagement in educational settings. Finally, we analyze how linguistic and cognitive factors, such as concreteness, comprehensibility (readability), and uptake, influence engagement in educational dialogues.
>
---
#### [new 066] HAVE: Head-Adaptive Gating and ValuE Calibration for Hallucination Mitigation in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出HAVE框架，用于减少大语言模型中的幻觉问题。通过头自适应门控和值校准，提升生成结果的可信度，无需微调，适用于多种LLM，在多个基准测试中表现优异。**

- **链接: [http://arxiv.org/pdf/2509.06596v1](http://arxiv.org/pdf/2509.06596v1)**

> **作者:** Xin Tong; Zhi Lin; Jingya Wang; Bo Jin
>
> **摘要:** Large Language Models (LLMs) often produce hallucinations in retrieval-augmented or long-context generation, even when relevant evidence is present. This stems from two issues: head importance is treated as input-agnostic, and raw attention weights poorly reflect each token's true contribution. We present HAVE (Head-Adaptive Gating and ValuE Calibration), a parameter-free decoding framework that directly addresses both challenges. HAVE introduces head-adaptive gating, which performs instance-level soft reweighing of attention heads, and value calibration, which augments attention with the magnitude of value vectors to approximate write-back contribution. Together, these modules construct token-level evidence aligned with model updates and fuse it with the LM distribution through a lightweight uncertainty-scaled policy. HAVE requires no finetuning and operates in a single forward pass, making it efficient and broadly applicable. Experiments across multiple QA benchmarks and LLM families demonstrate that HAVE consistently reduces hallucinations and outperforms strong baselines, including DAGCD, with modest overhead. The framework is transparent, reproducible, and readily integrates with off-the-shelf LLMs, advancing trustworthy generation in real-world settings.
>
---
#### [new 067] Do LLMs exhibit the same commonsense capabilities across languages?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在多语言常识生成任务中的表现，提出 MULTICOM 数据集，评估多种开源 LLM 在四国语言上的能力，发现英语表现最佳，低资源语言表现较差，揭示了当前模型在多语言常识生成中的局限性。**

- **链接: [http://arxiv.org/pdf/2509.06401v1](http://arxiv.org/pdf/2509.06401v1)**

> **作者:** Ivan Martínez-Murillo; Elena Lloret; Paloma Moreda; Albert Gatt
>
> **摘要:** This paper explores the multilingual commonsense generation abilities of Large Language Models (LLMs). To facilitate this investigation, we introduce MULTICOM, a novel benchmark that extends the COCOTEROS dataset to four languages: English, Spanish, Dutch, and Valencian. The task involves generating a commonsensical sentence that includes a given triplet of words. We evaluate a range of open-source LLMs, including LLaMA, Qwen, Gemma, EuroLLM, and Salamandra, on this benchmark. Our evaluation combines automatic metrics, LLM-as-a-judge approaches (using Prometheus and JudgeLM), and human annotations. Results consistently show superior performance in English, with significantly lower performance in less-resourced languages. While contextual support yields mixed results, it tends to benefit underrepresented languages. These findings underscore the current limitations of LLMs in multilingual commonsense generation. The dataset is publicly available at https://huggingface.co/datasets/gplsi/MULTICOM.
>
---
#### [new 068] Beyond Keywords: Driving Generative Search Engine Optimization with Content-Centric Agents
- **分类: cs.CL**

- **简介: 论文提出生成式搜索引擎优化（GSEO）框架，解决传统SEO在生成式搜索中的失效问题。构建内容中心基准CC-GSEO-Bench，并设计多智能体系统自动化优化内容，提升生成答案的影响力。**

- **链接: [http://arxiv.org/pdf/2509.05607v1](http://arxiv.org/pdf/2509.05607v1)**

> **作者:** Qiyuan Chen; Jiahe Chen; Hongsen Huang; Qian Shao; Jintai Chen; Renjie Hua; Hongxia Xu; Ruijia Wu; Ren Chuan; Jian Wu
>
> **备注:** Technical Report
>
> **摘要:** The paradigm shift from traditional ranked-based search to Generative Search Engines has rendered conventional SEO metrics obsolete, creating an urgent need to understand, measure, and optimize for content influence on synthesized answers. This paper introduces a comprehensive, end-to-end framework for Generative Search Engine Optimization (GSEO) to address this challenge. We make two primary contributions. First, we construct CC-GSEO-Bench, a large-scale, content-centric benchmark, and propose a multi-dimensional evaluation framework that systematically quantifies influence, moving beyond surface-level attribution to assess substantive semantic impact. Second, we design a novel multi-agent system that operationalizes this framework, automating the strategic refinement of content through a collaborative analyze-revise-evaluate workflow. Our empirical analysis using this framework reveals novel insights into the dynamics of content influence, offering actionable strategies for creators and establishing a principled foundation for future GSEO research.
>
---
#### [new 069] Test-Time Scaling in Reasoning Models Is Not Effective for Knowledge-Intensive Tasks Yet
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究测试时推理扩展在知识密集型任务中的有效性。发现增加推理步数并未提升准确性，反而可能引发更多幻觉。通过12个模型在两个基准上的实验分析了其影响，并指出模型更倾向于放弃回答而非提升事实记忆。**

- **链接: [http://arxiv.org/pdf/2509.06861v1](http://arxiv.org/pdf/2509.06861v1)**

> **作者:** James Xu Zhao; Bryan Hooi; See-Kiong Ng
>
> **备注:** 20 pages, 4 figures, 6 tables
>
> **摘要:** Test-time scaling increases inference-time computation by allowing models to generate long reasoning chains, and has shown strong performance across many domains. However, in this work, we show that this approach is not yet effective for knowledge-intensive tasks, where high factual accuracy and low hallucination rates are essential. We conduct a comprehensive evaluation of test-time scaling using 12 reasoning models on two knowledge-intensive benchmarks. Our results reveal that increasing test-time computation does not consistently improve accuracy and, in many cases, it even leads to more hallucinations. We then analyze how extended reasoning affects hallucination behavior. We find that reduced hallucinations often result from the model choosing to abstain after thinking more, rather than from improved factual recall. Conversely, for some models, longer reasoning encourages attempts on previously unanswered questions, many of which result in hallucinations. Case studies show that extended reasoning can induce confirmation bias, leading to overconfident hallucinations. Despite these limitations, we observe that compared to non-thinking, enabling thinking remains beneficial. Code and data are available at https://github.com/XuZhao0/tts-knowledge
>
---
#### [new 070] VehicleWorld: A Highly Integrated Multi-Device Environment for Intelligent Vehicle Interaction
- **分类: cs.AI; cs.CL; cs.RO**

- **简介: 该论文提出VehicleWorld，一个集成多设备的智能汽车交互环境，解决传统函数调用效率低的问题，提出基于状态的函数调用（SFC）方法，提升执行准确性和效率。**

- **链接: [http://arxiv.org/pdf/2509.06736v1](http://arxiv.org/pdf/2509.06736v1)**

> **作者:** Jie Yang; Jiajun Chen; Zhangyue Yin; Shuo Chen; Yuxin Wang; Yiran Guo; Yuan Li; Yining Zheng; Xuanjing Huang; Xipeng Qiu
>
> **摘要:** Intelligent vehicle cockpits present unique challenges for API Agents, requiring coordination across tightly-coupled subsystems that exceed typical task environments' complexity. Traditional Function Calling (FC) approaches operate statelessly, requiring multiple exploratory calls to build environmental awareness before execution, leading to inefficiency and limited error recovery. We introduce VehicleWorld, the first comprehensive environment for the automotive domain, featuring 30 modules, 250 APIs, and 680 properties with fully executable implementations that provide real-time state information during agent execution. This environment enables precise evaluation of vehicle agent behaviors across diverse, challenging scenarios. Through systematic analysis, we discovered that direct state prediction outperforms function calling for environmental control. Building on this insight, we propose State-based Function Call (SFC), a novel approach that maintains explicit system state awareness and implements direct state transitions to achieve target conditions. Experimental results demonstrate that SFC significantly outperforms traditional FC approaches, achieving superior execution accuracy and reduced latency. We have made all implementation code publicly available on Github https://github.com/OpenMOSS/VehicleWorld.
>
---
#### [new 071] Are LLM Agents Behaviorally Coherent? Latent Profiles for Social Simulation
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文探讨大语言模型（LLMs）代理在不同实验设置下的行为一致性。任务是评估LLM是否能作为人类参与者替代品。研究发现，尽管LLM能生成类似人类的回答，但其内部行为存在显著不一致，影响其在社会科学研究中的适用性。**

- **链接: [http://arxiv.org/pdf/2509.03736v1](http://arxiv.org/pdf/2509.03736v1)**

> **作者:** James Mooney; Josef Woldense; Zheng Robert Jia; Shirley Anugrah Hayati; My Ha Nguyen; Vipul Raheja; Dongyeop Kang
>
> **备注:** 25 pages, 9 figures, 7 tables
>
> **摘要:** The impressive capabilities of Large Language Models (LLMs) have fueled the notion that synthetic agents can serve as substitutes for real participants in human-subject research. In an effort to evaluate the merits of this claim, social science researchers have largely focused on whether LLM-generated survey data corresponds to that of a human counterpart whom the LLM is prompted to represent. In contrast, we address a more fundamental question: Do agents maintain internal consistency, retaining similar behaviors when examined under different experimental settings? To this end, we develop a study designed to (a) reveal the agent's internal state and (b) examine agent behavior in a basic dialogue setting. This design enables us to explore a set of behavioral hypotheses to assess whether an agent's conversation behavior is consistent with what we would expect from their revealed internal state. Our findings on these hypotheses show significant internal inconsistencies in LLMs across model families and at differing model sizes. Most importantly, we find that, although agents may generate responses matching those of their human counterparts, they fail to be internally consistent, representing a critical gap in their capabilities to accurately substitute for real participants in human-subject research. Our simulation code and data are publicly accessible.
>
---
#### [new 072] On the Contribution of Lexical Features to Speech Emotion Recognition
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音情感识别（SER）任务，研究词汇特征对情感识别的贡献。通过分析词义信息，提出一种基于词汇的模型，在MELD数据集上取得优于传统声学模型的效果，并探讨了自监督表示与音频去噪的影响。**

- **链接: [http://arxiv.org/pdf/2509.05634v1](http://arxiv.org/pdf/2509.05634v1)**

> **作者:** David Combei
>
> **备注:** Accepted to 13th Conference on Speech Technology and Human-Computer Dialogue
>
> **摘要:** Although paralinguistic cues are often considered the primary drivers of speech emotion recognition (SER), we investigate the role of lexical content extracted from speech and show that it can achieve competitive and in some cases higher performance compared to acoustic models. On the MELD dataset, our lexical-based approach obtains a weighted F1-score (WF1) of 51.5%, compared to 49.3% for an acoustic-only pipeline with a larger parameter count. Furthermore, we analyze different self-supervised (SSL) speech and text representations, conduct a layer-wise study of transformer-based encoders, and evaluate the effect of audio denoising.
>
---
#### [new 073] RAFFLES: Reasoning-based Attribution of Faults for LLM Systems
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出RAFFLES框架，用于检测大型语言模型系统的故障原因。它通过推理与迭代机制，提升故障定位准确性，解决现有评估方法在复杂系统中识别问题困难的问题。**

- **链接: [http://arxiv.org/pdf/2509.06822v1](http://arxiv.org/pdf/2509.06822v1)**

> **作者:** Chenyang Zhu; Spencer Hong; Jingyu Wu; Kushal Chawla; Charlotte Tang; Youbing Yin; Nathan Wolfe; Erin Babinsky; Daben Liu
>
> **摘要:** We have reached a critical roadblock in the development and enhancement of long-horizon, multi-component LLM agentic systems: it is incredibly tricky to identify where these systems break down and why. Evaluation capabilities that currently exist today (e.g., single pass LLM-as-a-judge) are limited in that they often focus on individual metrics or capabilities, end-to-end outcomes, and are narrowly grounded on the preferences of humans. We argue that to match the agentic capabilities, evaluation frameworks must also be able to reason, probe, iterate, and understand the complex logic passing through these systems over long horizons. In this paper, we present RAFFLES - an evaluation architecture that incorporates reasoning and iterative refinement. Specifically, RAFFLES operates as an iterative, multi-component pipeline, using a central Judge to systematically investigate faults and a set of specialized Evaluators to assess not only the system's components but also the quality of the reasoning by the Judge itself, thereby building a history of hypotheses. We tested RAFFLES against several baselines on the Who&When dataset, a benchmark designed to diagnose the "who" (agent) and "when" (step) of a system's failure. RAFFLES outperforms these baselines, achieving an agent-step fault pair accuracy of over 43% on the Algorithmically-Generated dataset (a substantial increase from the previously published best of 16.6%) and over 20% on the Hand-Crafted dataset (surpassing the previously published best of 8.8%). These results demonstrate a key step towards introducing automated fault detection for autonomous systems over labor-intensive manual human review.
>
---
#### [new 074] ForensicsData: A Digital Forensics Dataset for Large Language Models
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 论文提出ForensicsData，一个包含5000多个Q-C-A三元组的数字取证数据集，用于支持大语言模型研究。该数据集从真实恶意软件分析报告中构建，旨在解决数字取证领域缺乏公开数据的问题，促进可复现实验与研究合作。**

- **链接: [http://arxiv.org/pdf/2509.05331v1](http://arxiv.org/pdf/2509.05331v1)**

> **作者:** Youssef Chakir; Iyad Lahsen-Cherif
>
> **备注:** Accepted to WiMob 2025 (21st International Conference on Wireless and Mobile Computing, Networking and Communications), Marrakesh, Morocco, Oct 20-22, 2025. 6 pages, 5 figures, 5 tables. IEEEtran conference format
>
> **摘要:** The growing complexity of cyber incidents presents significant challenges for digital forensic investigators, especially in evidence collection and analysis. Public resources are still limited because of ethical, legal, and privacy concerns, even though realistic datasets are necessary to support research and tool developments. To address this gap, we introduce ForensicsData, an extensive Question-Context-Answer (Q-C-A) dataset sourced from actual malware analysis reports. It consists of more than 5,000 Q-C-A triplets. A unique workflow was used to create the dataset, which extracts structured data, uses large language models (LLMs) to transform it into Q-C-A format, and then uses a specialized evaluation process to confirm its quality. Among the models evaluated, Gemini 2 Flash demonstrated the best performance in aligning generated content with forensic terminology. ForensicsData aims to advance digital forensics by enabling reproducible experiments and fostering collaboration within the research community.
>
---
#### [new 075] Paper2Agent: Reimagining Research Papers As Interactive and Reliable AI Agents
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Paper2Agent框架，将科研论文转化为可交互的AI代理，解决传统论文难以复用和理解的问题。通过构建模型上下文协议（MCP），实现论文内容的自动化测试与优化，并支持自然语言查询，推动知识传播与协作研究。**

- **链接: [http://arxiv.org/pdf/2509.06917v1](http://arxiv.org/pdf/2509.06917v1)**

> **作者:** Jiacheng Miao; Joe R. Davis; Jonathan K. Pritchard; James Zou
>
> **摘要:** We introduce Paper2Agent, an automated framework that converts research papers into AI agents. Paper2Agent transforms research output from passive artifacts into active systems that can accelerate downstream use, adoption, and discovery. Conventional research papers require readers to invest substantial effort to understand and adapt a paper's code, data, and methods to their own work, creating barriers to dissemination and reuse. Paper2Agent addresses this challenge by automatically converting a paper into an AI agent that acts as a knowledgeable research assistant. It systematically analyzes the paper and the associated codebase using multiple agents to construct a Model Context Protocol (MCP) server, then iteratively generates and runs tests to refine and robustify the resulting MCP. These paper MCPs can then be flexibly connected to a chat agent (e.g. Claude Code) to carry out complex scientific queries through natural language while invoking tools and workflows from the original paper. We demonstrate Paper2Agent's effectiveness in creating reliable and capable paper agents through in-depth case studies. Paper2Agent created an agent that leverages AlphaGenome to interpret genomic variants and agents based on ScanPy and TISSUE to carry out single-cell and spatial transcriptomics analyses. We validate that these paper agents can reproduce the original paper's results and can correctly carry out novel user queries. By turning static papers into dynamic, interactive AI agents, Paper2Agent introduces a new paradigm for knowledge dissemination and a foundation for the collaborative ecosystem of AI co-scientists.
>
---
#### [new 076] Cross-Service Threat Intelligence in LLM Services using Privacy-Preserving Fingerprints
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 论文提出BinaryShield系统，解决LLM服务间因隐私法规无法共享威胁情报的问题。通过隐私保护指纹技术，在不泄露用户数据的前提下实现跨服务攻击模式共享，提升整体安全防护能力。**

- **链接: [http://arxiv.org/pdf/2509.05608v1](http://arxiv.org/pdf/2509.05608v1)**

> **作者:** Waris Gill; Natalie Isak; Matthew Dressman
>
> **摘要:** The widespread deployment of LLMs across enterprise services has created a critical security blind spot. Organizations operate multiple LLM services handling billions of queries daily, yet regulatory compliance boundaries prevent these services from sharing threat intelligence about prompt injection attacks, the top security risk for LLMs. When an attack is detected in one service, the same threat may persist undetected in others for months, as privacy regulations prohibit sharing user prompts across compliance boundaries. We present BinaryShield, the first privacy-preserving threat intelligence system that enables secure sharing of attack fingerprints across compliance boundaries. BinaryShield transforms suspicious prompts through a unique pipeline combining PII redaction, semantic embedding, binary quantization, and randomized response mechanism to potentially generate non-invertible fingerprints that preserve attack patterns while providing privacy. Our evaluations demonstrate that BinaryShield achieves an F1-score of 0.94, significantly outperforming SimHash (0.77), the privacy-preserving baseline, while achieving 64x storage reduction and 38x faster similarity search compared to dense embeddings.
>
---
#### [new 077] SFR-DeepResearch: Towards Effective Reinforcement Learning for Autonomously Reasoning Single Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SFR-DeepResearch，专注于通过强化学习提升单智能体自主推理与研究能力。任务是实现无需人工指令的自主深度研究，解决复杂推理与工具使用问题，采用合成数据训练模型，取得显著性能提升。**

- **链接: [http://arxiv.org/pdf/2509.06283v1](http://arxiv.org/pdf/2509.06283v1)**

> **作者:** Xuan-Phi Nguyen; Shrey Pandit; Revanth Gangi Reddy; Austin Xu; Silvio Savarese; Caiming Xiong; Shafiq Joty
>
> **备注:** Technical Report
>
> **摘要:** Equipping large language models (LLMs) with complex, interleaved reasoning and tool-use capabilities has become a key focus in agentic AI research, especially with recent advances in reasoning-oriented (``thinking'') models. Such capabilities are key to unlocking a number of important applications. One such application is Deep Research (DR), which requires extensive search and reasoning over many sources. Our work in this paper focuses on the development of native Autonomous Single-Agent models for DR featuring minimal web crawling and Python tool integration. Unlike multi-agent systems, where agents take up pre-defined roles and are told what to do at each step in a static workflow, an autonomous single-agent determines its next action dynamically based on context, without manual directive. While prior work has proposed training recipes for base or instruction-tuned LLMs, we focus on continual reinforcement learning (RL) of reasoning-optimized models to further enhance agentic skills while preserving reasoning ability. Towards this end, we propose a simple RL recipe with entirely synthetic data, which we apply to various open-source LLMs. Our best variant SFR-DR-20B achieves up to 28.7% on Humanity's Last Exam benchmark. In addition, we conduct key analysis experiments to provide more insights into our methodologies.
>
---
#### [new 078] Beamforming-LLM: What, Where and When Did I Miss?
- **分类: eess.AS; cs.AI; cs.CL; cs.HC**

- **简介: 论文提出Beamforming-LLM系统，通过麦克风阵列和RAG技术，帮助用户语义回忆多说话人环境中错过的对话。系统利用波束成形分离音频、Whisper转录、向量检索及轻量大模型生成摘要，实现语义检索与对比总结，应用于智能听觉记忆系统。**

- **链接: [http://arxiv.org/pdf/2509.06221v1](http://arxiv.org/pdf/2509.06221v1)**

> **作者:** Vishal Choudhari
>
> **摘要:** We present Beamforming-LLM, a system that enables users to semantically recall conversations they may have missed in multi-speaker environments. The system combines spatial audio capture using a microphone array with retrieval-augmented generation (RAG) to support natural language queries such as, "What did I miss when I was following the conversation on dogs?" Directional audio streams are separated using beamforming, transcribed with Whisper, and embedded into a vector database using sentence encoders. Upon receiving a user query, semantically relevant segments are retrieved, temporally aligned with non-attended segments, and summarized using a lightweight large language model (GPT-4o-mini). The result is a user-friendly interface that provides contrastive summaries, spatial context, and timestamped audio playback. This work lays the foundation for intelligent auditory memory systems and has broad applications in assistive technology, meeting summarization, and context-aware personal spatial computing.
>
---
#### [new 079] TSPC: A Two-Stage Phoneme-Centric Architecture for code-switching Vietnamese-English Speech Recognition
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出TSPC模型，解决越南语-英语代码切换语音识别中的音系差异与混淆问题。采用两阶段音素中心架构，利用扩展音素集进行混合语言建模，显著降低词错误率，优于现有基线。**

- **链接: [http://arxiv.org/pdf/2509.05983v1](http://arxiv.org/pdf/2509.05983v1)**

> **作者:** Minh N. H. Nguyen; Anh Nguyen Tran; Dung Truong Dinh; Nam Van Vo
>
> **摘要:** Code-switching (CS) presents a significant challenge for general Auto-Speech Recognition (ASR) systems. Existing methods often fail to capture the subtle phonological shifts inherent in CS scenarios. The challenge is particularly difficult for language pairs like Vietnamese and English, where both distinct phonological features and the ambiguity arising from similar sound recognition are present. In this paper, we propose a novel architecture for Vietnamese-English CS ASR, a Two-Stage Phoneme-Centric model (TSPC). The TSPC employs a phoneme-centric approach, built upon an extended Vietnamese phoneme set as an intermediate representation to facilitate mixed-lingual modeling. Experimental results demonstrate that TSPC consistently outperforms existing baselines, including PhoWhisper-base, in Vietnamese-English CS ASR, achieving a significantly lower word error rate of 20.8\% with reduced training resources. Furthermore, the phonetic-based two-stage architecture enables phoneme adaptation and language conversion to enhance ASR performance in complex CS Vietnamese-English ASR scenarios.
>
---
#### [new 080] From Long to Short: LLMs Excel at Trimming Own Reasoning Chains
- **分类: cs.AI; cs.CL**

- **简介: 论文研究如何提升大语言模型的推理效率，解决其过度复杂化简单问题导致可解释性差的问题。提出EDIT方法，在测试时动态修剪推理路径，实现简洁与正确性的平衡。属于自然语言处理中的推理优化任务。**

- **链接: [http://arxiv.org/pdf/2509.06174v1](http://arxiv.org/pdf/2509.06174v1)**

> **作者:** Wei Han; Geng Zhan; Sicheng Yu; Chenyu Wang; Bryan Hooi
>
> **备注:** 21 pages, 5 figures, 7 tables
>
> **摘要:** O1/R1 style large reasoning models (LRMs) signal a substantial leap forward over conventional instruction-following LLMs. By applying test-time scaling to generate extended reasoning paths, they establish many SOTAs across a wide range of complex reasoning tasks. However, recent studies show that LRMs are prone to suffer from overthinking -- the tendency to overcomplicate simple problems, leading to excessive strategy switching and long, convoluted reasoning traces that hinder their interpretability. To mitigate this issue, we conduct a systematic investigation into the reasoning efficiency of a broad set of LRMs and uncover a common dilemma: the difficulty in balancing multiple generation objectives such as correctness and brevity. Based on this discovery, we propose a test-time scaling method, EDIT (Efficient Dynamic Inference Trimming), which efficiently guides LRMs to identify the shortest correct reasoning paths at test time. EDIT employs constraint-guided generation while jointly tracking length and answer distributions under varying constraints, allowing it to select responses that strike an optimal balance between conciseness and correctness. Extensive experiments across diverse models and datasets show that EDIT substantially enhance the reasoning efficiency, producing compact yet informative outputs that improve readability and user experience.
>
---
#### [new 081] Outcome-based Exploration for LLM Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究如何通过结果导向的探索方法提升大语言模型的推理能力。针对强化学习导致生成多样性下降的问题，提出历史探索与批量探索算法，在提升准确率的同时缓解多样性崩溃，属于自然语言处理中的模型优化任务。**

- **链接: [http://arxiv.org/pdf/2509.06941v1](http://arxiv.org/pdf/2509.06941v1)**

> **作者:** Yuda Song; Julia Kempe; Remi Munos
>
> **备注:** 26 pages, 11 figures
>
> **摘要:** Reinforcement learning (RL) has emerged as a powerful method for improving the reasoning abilities of large language models (LLMs). Outcome-based RL, which rewards policies solely for the correctness of the final answer, yields substantial accuracy gains but also induces a systematic loss in generation diversity. This collapse undermines real-world performance, where diversity is critical for test-time scaling. We analyze this phenomenon by viewing RL post-training as a sampling process and show that, strikingly, RL can reduce effective diversity even on the training set relative to the base model. Our study highlights two central findings: (i) a transfer of diversity degradation, where reduced diversity on solved problems propagates to unsolved ones, and (ii) the tractability of the outcome space, since reasoning tasks admit only a limited set of distinct answers. Motivated by these insights, we propose outcome-based exploration, which assigns exploration bonuses according to final outcomes. We introduce two complementary algorithms: historical exploration, which encourages rarely observed answers via UCB-style bonuses, and batch exploration, which penalizes within-batch repetition to promote test-time diversity. Experiments on standard competition math with Llama and Qwen models demonstrate that both methods improve accuracy while mitigating diversity collapse. On the theoretical side, we formalize the benefit of outcome-based exploration through a new model of outcome-based bandits. Together, these contributions chart a practical path toward RL methods that enhance reasoning without sacrificing the diversity essential for scalable deployment.
>
---
#### [new 082] ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders
- **分类: q-bio.QM; cs.AI; cs.CL**

- **简介: 该论文提出ProtSAE，一种语义引导的稀疏自编码器，用于蛋白质语言模型的可解释性分析。旨在解决传统SAE中语义纠缠问题，通过结合标注数据与领域知识提升特征可解释性与重构性能。属于自然语言处理中的模型解释任务。**

- **链接: [http://arxiv.org/pdf/2509.05309v1](http://arxiv.org/pdf/2509.05309v1)**

> **作者:** Xiangyu Liu; Haodi Lei; Yi Liu; Yang Liu; Wei Hu
>
> **摘要:** Sparse Autoencoder (SAE) has emerged as a powerful tool for mechanistic interpretability of large language models. Recent works apply SAE to protein language models (PLMs), aiming to extract and analyze biologically meaningful features from their latent spaces. However, SAE suffers from semantic entanglement, where individual neurons often mix multiple nonlinear concepts, making it difficult to reliably interpret or manipulate model behaviors. In this paper, we propose a semantically-guided SAE, called ProtSAE. Unlike existing SAE which requires annotation datasets to filter and interpret activations, we guide semantic disentanglement during training using both annotation datasets and domain knowledge to mitigate the effects of entangled attributes. We design interpretability experiments showing that ProtSAE learns more biologically relevant and interpretable hidden features compared to previous methods. Performance analyses further demonstrate that ProtSAE maintains high reconstruction fidelity while achieving better results in interpretable probing. We also show the potential of ProtSAE in steering PLMs for downstream generation tasks.
>
---
#### [new 083] Reverse-Engineered Reasoning for Open-Ended Generation
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出REER方法，解决开放生成任务中深度推理不足的问题。通过逆向工程已知解决方案，构建大规模推理轨迹数据集DeepWriting-20K，并训练模型DeepWriter-8B，在性能上超越多个主流模型。属于自然语言生成与推理任务。**

- **链接: [http://arxiv.org/pdf/2509.06160v1](http://arxiv.org/pdf/2509.06160v1)**

> **作者:** Haozhe Wang; Haoran Que; Qixin Xu; Minghao Liu; Wangchunshu Zhou; Jiazhan Feng; Wanjun Zhong; Wei Ye; Tong Yang; Wenhao Huang; Ge Zhang; Fangzhen Lin
>
> **备注:** Preprint
>
> **摘要:** While the ``deep reasoning'' paradigm has spurred significant advances in verifiable domains like mathematics, its application to open-ended, creative generation remains a critical challenge. The two dominant methods for instilling reasoning -- reinforcement learning (RL) and instruction distillation -- falter in this area; RL struggles with the absence of clear reward signals and high-quality reward models, while distillation is prohibitively expensive and capped by the teacher model's capabilities. To overcome these limitations, we introduce REverse-Engineered Reasoning (REER), a new paradigm that fundamentally shifts the approach. Instead of building a reasoning process ``forwards'' through trial-and-error or imitation, REER works ``backwards'' from known-good solutions to computationally discover the latent, step-by-step deep reasoning process that could have produced them. Using this scalable, gradient-free approach, we curate and open-source DeepWriting-20K, a large-scale dataset of 20,000 deep reasoning trajectories for open-ended tasks. Our model, DeepWriter-8B, trained on this data, not only surpasses strong open-source baselines but also achieves performance competitive with, and at times superior to, leading proprietary models like GPT-4o and Claude 3.5.
>
---
#### [new 084] An Ethically Grounded LLM-Based Approach to Insider Threat Synthesis and Detection
- **分类: cs.CR; cs.AI; cs.CL; cs.CY; C.2.0; I.2.7; K.4.1; H.3.3**

- **简介: 该论文提出一种基于伦理的LLM方法，用于合成和检测内部威胁。通过 Claude Sonnet 3.7 动态生成 syslog 数据，提升检测准确性并减少误报，解决静态数据限制问题。**

- **链接: [http://arxiv.org/pdf/2509.06920v1](http://arxiv.org/pdf/2509.06920v1)**

> **作者:** Haywood Gelman; John D. Hastings; David Kenley
>
> **备注:** 6 pages, 5 figures, 5 tables
>
> **摘要:** Insider threats are a growing organizational problem due to the complexity of identifying their technical and behavioral elements. A large research body is dedicated to the study of insider threats from technological, psychological, and educational perspectives. However, research in this domain has been generally dependent on datasets that are static and limited access which restricts the development of adaptive detection models. This study introduces a novel, ethically grounded approach that uses the large language model (LLM) Claude Sonnet 3.7 to dynamically synthesize syslog messages, some of which contain indicators of insider threat scenarios. The messages reflect real-world data distributions by being highly imbalanced (1% insider threats). The syslogs were analyzed for insider threats by both Claude Sonnet 3.7 and GPT-4o, with their performance evaluated through statistical metrics including precision, recall, MCC, and ROC AUC. Sonnet 3.7 consistently outperformed GPT-4o across nearly all metrics, particularly in reducing false alarms and improving detection accuracy. The results show strong promise for the use of LLMs in synthetic dataset generation and insider threat detection.
>
---
#### [new 085] Imagining Alternatives: Towards High-Resolution 3D Counterfactual Medical Image Generation via Language Guidance
- **分类: eess.IV; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出一种基于语言引导的高分辨率3D医学反事实图像生成框架，解决缺乏预训练3D模型的问题，用于模拟疾病进展和生成虚拟患者图像，提升医学研究与临床应用。**

- **链接: [http://arxiv.org/pdf/2509.05978v1](http://arxiv.org/pdf/2509.05978v1)**

> **作者:** Mohamed Mohamed; Brennan Nichyporuk; Douglas L. Arnold; Tal Arbel
>
> **摘要:** Vision-language models have demonstrated impressive capabilities in generating 2D images under various conditions; however the impressive performance of these models in 2D is largely enabled by extensive, readily available pretrained foundation models. Critically, comparable pretrained foundation models do not exist for 3D, significantly limiting progress in this domain. As a result, the potential of vision-language models to produce high-resolution 3D counterfactual medical images conditioned solely on natural language descriptions remains completely unexplored. Addressing this gap would enable powerful clinical and research applications, such as personalized counterfactual explanations, simulation of disease progression scenarios, and enhanced medical training by visualizing hypothetical medical conditions in realistic detail. Our work takes a meaningful step toward addressing this challenge by introducing a framework capable of generating high-resolution 3D counterfactual medical images of synthesized patients guided by free-form language prompts. We adapt state-of-the-art 3D diffusion models with enhancements from Simple Diffusion and incorporate augmented conditioning to improve text alignment and image quality. To our knowledge, this represents the first demonstration of a language-guided native-3D diffusion model applied specifically to neurological imaging data, where faithful three-dimensional modeling is essential to represent the brain's three-dimensional structure. Through results on two distinct neurological MRI datasets, our framework successfully simulates varying counterfactual lesion loads in Multiple Sclerosis (MS), and cognitive states in Alzheimer's disease, generating high-quality images while preserving subject fidelity in synthetically generated medical images. Our results lay the groundwork for prompt-driven disease progression analysis within 3D medical imaging.
>
---
#### [new 086] Authorship Without Writing: Large Language Models and the Senior Author Analogy
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 论文探讨大语言模型（LLM）在科研写作中的作者身份问题，提出其使用可类比于资深作者角色。研究旨在解决LLM生成内容是否应被视作合法作者身份的问题，并主张现行作者标准需重新审视或认可LLM的作者地位。**

- **链接: [http://arxiv.org/pdf/2509.05390v1](http://arxiv.org/pdf/2509.05390v1)**

> **作者:** Clint Hurshman; Sebastian Porsdam Mann; Julian Savulescu; Brian D. Earp
>
> **备注:** 28 pages, 0 figures
>
> **摘要:** The use of large language models (LLMs) in bioethical, scientific, and medical writing remains controversial. While there is broad agreement in some circles that LLMs cannot count as authors, there is no consensus about whether and how humans using LLMs can count as authors. In many fields, authorship is distributed among large teams of researchers, some of whom, including paradigmatic senior authors who guide and determine the scope of a project and ultimately vouch for its integrity, may not write a single word. In this paper, we argue that LLM use (under specific conditions) is analogous to a form of senior authorship. On this view, the use of LLMs, even to generate complete drafts of research papers, can be considered a legitimate form of authorship according to the accepted criteria in many fields. We conclude that either such use should be recognized as legitimate, or current criteria for authorship require fundamental revision. AI use declaration: GPT-5 was used to help format Box 1. AI was not used for any other part of the preparation or writing of this manuscript.
>
---
#### [new 087] Reinforcement Learning Foundations for Deep Research Systems: A Survey
- **分类: cs.AI; cs.CL**

- **简介: 该论文综述了深度研究系统中强化学习的基础，旨在解决其训练中的数据、方法与框架问题。系统梳理了RL在代理研究中的应用，涵盖数据生成、算法优化与评估体系，推动构建鲁棒、透明的智能研究代理。**

- **链接: [http://arxiv.org/pdf/2509.06733v1](http://arxiv.org/pdf/2509.06733v1)**

> **作者:** Wenjun Li; Zhi Chen; Jingru Lin; Hannan Cao; Wei Han; Sheng Liang; Zhi Zhang; Kuicai Dong; Dexun Li; Chen Zhang; Yong Liu
>
> **备注:** 38 pages, first version
>
> **摘要:** Deep research systems, agentic AI that solve complex, multi-step tasks by coordinating reasoning, search across the open web and user files, and tool use, are moving toward hierarchical deployments with a Planner, Coordinator, and Executors. In practice, training entire stacks end-to-end remains impractical, so most work trains a single planner connected to core tools such as search, browsing, and code. While SFT imparts protocol fidelity, it suffers from imitation and exposure biases and underuses environment feedback. Preference alignment methods such as DPO are schema and proxy-dependent, off-policy, and weak for long-horizon credit assignment and multi-objective trade-offs. A further limitation of SFT and DPO is their reliance on human defined decision points and subskills through schema design and labeled comparisons. Reinforcement learning aligns with closed-loop, tool-interaction research by optimizing trajectory-level policies, enabling exploration, recovery behaviors, and principled credit assignment, and it reduces dependence on such human priors and rater biases. This survey is, to our knowledge, the first dedicated to the RL foundations of deep research systems. It systematizes work after DeepSeek-R1 along three axes: (i) data synthesis and curation; (ii) RL methods for agentic research covering stability, sample efficiency, long context handling, reward and credit design, multi-objective optimization, and multimodal integration; and (iii) agentic RL training systems and frameworks. We also cover agent architecture and coordination, as well as evaluation and benchmarks, including recent QA, VQA, long-form synthesis, and domain-grounded, tool-interaction tasks. We distill recurring patterns, surface infrastructure bottlenecks, and offer practical guidance for training robust, transparent deep research agents with RL.
>
---
#### [new 088] Index-Preserving Lightweight Token Pruning for Efficient Document Understanding in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于文档理解任务，旨在解决视觉语言模型计算成本高的问题。提出了一种轻量级的token剪枝框架，通过分类器去除非文本区域并恢复碎片化文本，降低计算开销同时保持精度。**

- **链接: [http://arxiv.org/pdf/2509.06415v1](http://arxiv.org/pdf/2509.06415v1)**

> **作者:** Jaemin Son; Sujin Choi; Inyong Yun
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Recent progress in vision-language models (VLMs) has led to impressive results in document understanding tasks, but their high computational demands remain a challenge. To mitigate the compute burdens, we propose a lightweight token pruning framework that filters out non-informative background regions from document images prior to VLM processing. A binary patch-level classifier removes non-text areas, and a max-pooling refinement step recovers fragmented text regions to enhance spatial coherence. Experiments on real-world document datasets demonstrate that our approach substantially lowers computational costs, while maintaining comparable accuracy.
>
---
#### [new 089] Interleaving Reasoning for Better Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于文本到图像生成任务，旨在提升模型的指令遵循与细节保留能力。提出IRG框架，通过交替进行文本推理与图像合成，结合IRGL训练方法，显著提升生成质量与细节精度。**

- **链接: [http://arxiv.org/pdf/2509.06945v1](http://arxiv.org/pdf/2509.06945v1)**

> **作者:** Wenxuan Huang; Shuang Chen; Zheyong Xie; Shaosheng Cao; Shixiang Tang; Yufan Shen; Qingyu Yin; Wenbo Hu; Xiaoman Wang; Yuntian Tang; Junbo Qiao; Yue Guo; Yao Hu; Zhenfei Yin; Philip Torr; Yu Cheng; Wanli Ouyang; Shaohui Lin
>
> **摘要:** Unified multimodal understanding and generation models recently have achieve significant improvement in image generation capability, yet a large gap remains in instruction following and detail preservation compared to systems that tightly couple comprehension with generation such as GPT-4o. Motivated by recent advances in interleaving reasoning, we explore whether such reasoning can further improve Text-to-Image (T2I) generation. We introduce Interleaving Reasoning Generation (IRG), a framework that alternates between text-based thinking and image synthesis: the model first produces a text-based thinking to guide an initial image, then reflects on the result to refine fine-grained details, visual quality, and aesthetics while preserving semantics. To train IRG effectively, we propose Interleaving Reasoning Generation Learning (IRGL), which targets two sub-goals: (1) strengthening the initial think-and-generate stage to establish core content and base quality, and (2) enabling high-quality textual reflection and faithful implementation of those refinements in a subsequent image. We curate IRGL-300K, a dataset organized into six decomposed learning modes that jointly cover learning text-based thinking, and full thinking-image trajectories. Starting from a unified foundation model that natively emits interleaved text-image outputs, our two-stage training first builds robust thinking and reflection, then efficiently tunes the IRG pipeline in the full thinking-image trajectory data. Extensive experiments show SoTA performance, yielding absolute gains of 5-10 points on GenEval, WISE, TIIF, GenAI-Bench, and OneIG-EN, alongside substantial improvements in visual quality and fine-grained fidelity. The code, model weights and datasets will be released in: https://github.com/Osilly/Interleaving-Reasoning-Generation .
>
---
#### [new 090] Language Native Lightly Structured Databases for Large Language Model Driven Composite Materials Research
- **分类: cs.DB; cond-mat.mtrl-sci; cs.AI; cs.CL**

- **简介: 论文提出一种语言原生轻结构数据库，用于大语言模型驱动的复合材料研究。解决传统数据库依赖表格、难以利用语言信息的问题，通过整合文献信息实现高效检索与生成，支持材料发现任务。**

- **链接: [http://arxiv.org/pdf/2509.06093v1](http://arxiv.org/pdf/2509.06093v1)**

> **作者:** Yuze Liu; Zhaoyuan Zhang; Xiangsheng Zeng; Yihe Zhang; Leping Yu; Lejia Wang; Xi Yu
>
> **摘要:** Chemical and materials research has traditionally relied heavily on knowledge narrative, with progress often driven by language-based descriptions of principles, mechanisms, and experimental experiences, rather than tables, limiting what conventional databases and ML can exploit. We present a language-native database for boron nitride nanosheet (BNNS) polymer thermally conductive composites that captures lightly structured information from papers across preparation, characterization, theory-computation, and mechanistic reasoning, with evidence-linked snippets. Records are organized in a heterogeneous database and queried via composite retrieval with semantics, key words and value filters. The system can synthesizes literature into accurate, verifiable, and expert style guidance. This substrate enables high fidelity efficient Retrieval Augmented Generation (RAG) and tool augmented agents to interleave retrieval with reasoning and deliver actionable SOP. The framework supplies the language rich foundation required for LLM-driven materials discovery.
>
---
#### [new 091] Language Bias in Information Retrieval: The Nature of the Beast and Mitigation Methods
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文研究多语言信息检索中的语言偏差问题，旨在提升不同语言查询的公平性。论文分析了现有方法的偏差，并提出LaKDA损失函数以缓解这一问题。属于信息检索任务，解决语言不公平性问题。**

- **链接: [http://arxiv.org/pdf/2509.06195v1](http://arxiv.org/pdf/2509.06195v1)**

> **作者:** Jinrui Yang; Fan Jiang; Timothy Baldwin
>
> **备注:** Accepted at EMNLP MRL 2024
>
> **摘要:** Language fairness in multilingual information retrieval (MLIR) systems is crucial for ensuring equitable access to information across diverse languages. This paper sheds light on the issue, based on the assumption that queries in different languages, but with identical semantics, should yield equivalent ranking lists when retrieving on the same multilingual documents. We evaluate the degree of fairness using both traditional retrieval methods, and a DPR neural ranker based on mBERT and XLM-R. Additionally, we introduce `LaKDA', a novel loss designed to mitigate language biases in neural MLIR approaches. Our analysis exposes intrinsic language biases in current MLIR technologies, with notable disparities across the retrieval methods, and the effectiveness of LaKDA in enhancing language fairness.
>
---
## 更新

#### [replaced 001] Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments
- **分类: stat.AP; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.00903v4](http://arxiv.org/pdf/2410.00903v4)**

> **作者:** Kosuke Imai; Kentaro Nakamura
>
> **摘要:** In this paper, we demonstrate how to enhance the validity of causal inference with unstructured high-dimensional treatments like texts, by leveraging the power of generative Artificial Intelligence (GenAI). Specifically, we propose to use a deep generative model such as large language models (LLMs) to efficiently generate treatments and use their internal representation for subsequent causal effect estimation. We show that the knowledge of this true internal representation helps disentangle the treatment features of interest, such as specific sentiments and certain topics, from other possibly unknown confounding features. Unlike existing methods, the proposed GenAI-Powered Inference (GPI) methodology eliminates the need to learn causal representation from the data, and hence produces more accurate and efficient estimates. We formally establish the conditions required for the nonparametric identification of the average treatment effect, propose an estimation strategy that avoids the violation of the overlap assumption, and derive the asymptotic properties of the proposed estimator through the application of double machine learning. Finally, using an instrumental variables approach, we extend the proposed GPI methodology to the settings in which the treatment feature is based on human perception. The GPI is also applicable to text reuse where an LLM is used to regenerate existing texts. We conduct simulation and empirical studies, using the generated text data from an open-source LLM, Llama~3, to illustrate the advantages of our estimator over state-of-the-art causal representation learning algorithms.
>
---
#### [replaced 002] VocalBench: Benchmarking the Vocal Conversational Abilities for Speech Interaction Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15727v2](http://arxiv.org/pdf/2505.15727v2)**

> **作者:** Heyang Liu; Yuhao Wang; Ziyang Cheng; Ronghua Wu; Qunshan Gu; Yanfeng Wang; Yu Wang
>
> **摘要:** The rapid advancement of large language models (LLMs) has accelerated the development of multimodal models capable of speech communications. Unlike text interactions, speech conveys diverse information, including acoustic variations, paralanguage cues, and environmental context. However, existing evaluations of speech interaction models lack instances mimicking real scenarios and predominantly focus on the quality of their textual responses, overlooking critical aspects of vocal performance. To address this gap, we propose VocalBench, a comprehensive benchmark to assess the speech conversational abilities, comprising 9,400 carefully curated instances across four key dimensions: semantic quality, acoustic performance, conversational abilities, and robustness. It covers a broad range of fundamental skills essential for effective vocal interactions. For the evaluation scheme, we propose several objective evaluation indicators and incorporate an additional LLM-as-a-judge approach to score open-ended questions. Experimental results on 15 mainstream systems reveal significant variability, each exhibiting distinct strengths and weaknesses, and provide valuable insights to guide future research in speech interaction systems.
>
---
#### [replaced 003] Joint Information Extraction Across Classical and Modern Chinese with Tea-MOELoRA
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.01158v2](http://arxiv.org/pdf/2509.01158v2)**

> **作者:** Xuemei Tang; Chengxi Yan; Jinghang Gu; Chu-Ren Huang
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Chinese information extraction (IE) involves multiple tasks across diverse temporal domains, including Classical and Modern documents. Fine-tuning a single model on heterogeneous tasks and across different eras may lead to interference and reduced performance. Therefore, in this paper, we propose Tea-MOELoRA, a parameter-efficient multi-task framework that combines LoRA with a Mixture-of-Experts (MoE) design. Multiple low-rank LoRA experts specialize in different IE tasks and eras, while a task-era-aware router mechanism dynamically allocates expert contributions. Experiments show that Tea-MOELoRA outperforms both single-task and joint LoRA baselines, demonstrating its ability to leverage task and temporal knowledge effectively.
>
---
#### [replaced 004] Synth-SBDH: A Synthetic Dataset of Social and Behavioral Determinants of Health for Clinical Text
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.06056v3](http://arxiv.org/pdf/2406.06056v3)**

> **作者:** Avijit Mitra; Zhichao Yang; Emily Druhl; Raelene Goodwin; Hong Yu
>
> **备注:** Accepted at EMNLP 2025 (main) Github: https://github.com/avipartho/Synth-SBDH
>
> **摘要:** Social and behavioral determinants of health (SBDH) play a crucial role in health outcomes and are frequently documented in clinical text. Automatically extracting SBDH information from clinical text relies on publicly available good-quality datasets. However, existing SBDH datasets exhibit substantial limitations in their availability and coverage. In this study, we introduce Synth-SBDH, a novel synthetic dataset with detailed SBDH annotations, encompassing status, temporal information, and rationale across 15 SBDH categories. We showcase the utility of Synth-SBDH on three tasks using real-world clinical datasets from two distinct hospital settings, highlighting its versatility, generalizability, and distillation capabilities. Models trained on Synth-SBDH consistently outperform counterparts with no Synth-SBDH training, achieving up to 63.75% macro-F improvements. Additionally, Synth-SBDH proves effective for rare SBDH categories and under-resource constraints while being substantially cheaper than expert-annotated real-world data. Human evaluation reveals a 71.06% Human-LLM alignment and uncovers areas for future refinements.
>
---
#### [replaced 005] Premise-Augmented Reasoning Chains Improve Error Identification in Math reasoning with LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.02362v5](http://arxiv.org/pdf/2502.02362v5)**

> **作者:** Sagnik Mukherjee; Abhinav Chinta; Takyoung Kim; Tarun Anoop Sharma; Dilek Hakkani-Tür
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** Chain-of-Thought (CoT) prompting enhances mathematical reasoning in large language models (LLMs) by enabling detailed step-by-step solutions. However, due to the verbosity of LLMs, the resulting reasoning chains can be long, making it harder to verify the reasoning steps and trace issues resulting from dependencies between the steps that may be farther away in the sequence of steps. Importantly, mathematical reasoning allows each step to be derived from a small set of premises, which are a subset of the preceding steps in the reasoning chain. In this paper, we present a framework that identifies the premises for each step, to improve the evaluation of reasoning. We restructure conventional linear reasoning chains into Premise Augmented Reasoning Chains (PARC) by introducing premise links, resulting in a directed acyclic graph where the nodes are the steps and the edges are the premise links. Through experiments with a PARC-based dataset that we built, namely PERL (Premises and ERrors identification in LLMs), we demonstrate that LLMs can reliably identify premises within complex reasoning chains. In particular, even open-source LLMs achieve 90% recall in premise identification. We also show that PARC helps to identify errors in reasoning chains more reliably. The accuracy of error identification improves by 6% to 16% absolute when step-by-step verification is carried out in PARC under the premises. Our findings highlight the utility of premise-centric representations in addressing complex problem-solving tasks and open new avenues for improving the reliability of LLM-based reasoning evaluations.
>
---
#### [replaced 006] Conversational Code Generation: a Case Study of Designing a Dialogue System for Generating Driving Scenarios for Testing Autonomous Vehicles
- **分类: cs.CL; cs.IR; cs.RO**

- **链接: [http://arxiv.org/pdf/2410.09829v3](http://arxiv.org/pdf/2410.09829v3)**

> **作者:** Rimvydas Rubavicius; Antonio Valerio Miceli-Barone; Alex Lascarides; Subramanian Ramamoorthy
>
> **备注:** In Proceedings of GeCoIn 2025: Generative Code Intelligence Workshop, co-located with ECAI-2025
>
> **摘要:** Cyber-physical systems like autonomous vehicles are tested in simulation before deployment, using domain-specific programs for scenario specification. To aid the testing of autonomous vehicles in simulation, we design a natural language interface, using an instruction-following large language model, to assist a non-coding domain expert in synthesising the desired scenarios and vehicle behaviours. We show that using it to convert utterances to the symbolic program is feasible, despite the very small training dataset. Human experiments show that dialogue is critical to successful simulation generation, leading to a 4.5 times higher success rate than a generation without engaging in extended conversation.
>
---
#### [replaced 007] Position: LLMs Can be Good Tutors in English Education
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.05467v2](http://arxiv.org/pdf/2502.05467v2)**

> **作者:** Jingheng Ye; Shen Wang; Deqing Zou; Yibo Yan; Kun Wang; Hai-Tao Zheng; Ruitong Liu; Zenglin Xu; Irwin King; Philip S. Yu; Qingsong Wen
>
> **备注:** Accepted to EMNLP 2025 Main. 20 pages, 4 figures
>
> **摘要:** While recent efforts have begun integrating large language models (LLMs) into English education, they often rely on traditional approaches to learning tasks without fully embracing educational methodologies, thus lacking adaptability to language learning. To address this gap, we argue that LLMs have the potential to serve as effective tutors in English Education. Specifically, LLMs can play three critical roles: (1) as data enhancers, improving the creation of learning materials or serving as student simulations; (2) as task predictors, serving as learner assessment or optimizing learning pathway; and (3) as agents, enabling personalized and inclusive education. We encourage interdisciplinary research to explore these roles, fostering innovation while addressing challenges and risks, ultimately advancing English Education through the thoughtful integration of LLMs.
>
---
#### [replaced 008] Out of the Box, into the Clinic? Evaluating State-of-the-Art ASR for Clinical Applications for Older Adults
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2508.08684v2](http://arxiv.org/pdf/2508.08684v2)**

> **作者:** Bram van Dijk; Tiberon Kuiper; Sirin Aoulad si Ahmed; Armel Levebvre; Jake Johnson; Jan Duin; Simon Mooijaart; Marco Spruit
>
> **摘要:** Voice-controlled interfaces can support older adults in clinical contexts, with chatbots being a prime example, but reliable Automatic Speech Recognition (ASR) for underrepresented groups remains a bottleneck. This study evaluates state-of-the-art ASR models on language use of older Dutch adults, who interacted with the \texttt{Welzijn.AI} chatbot designed for geriatric contexts. We benchmark generic multilingual ASR models, and models fine-tuned for Dutch spoken by older adults, while also considering processing speed. Our results show that generic multilingual models outperform fine-tuned models, which suggests recent ASR models can generalise well out of the box to realistic datasets. Furthermore, our results suggest that truncating existing architectures is helpful in balancing the accuracy-speed trade-off, though we also identify some cases with high WER due to hallucinations.
>
---
#### [replaced 009] Sticker-TTS: Learn to Utilize Historical Experience with a Sticker-driven Test-Time Scaling Framework
- **分类: cs.AI; cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2509.05007v2](http://arxiv.org/pdf/2509.05007v2)**

> **作者:** Jie Chen; Jinhao Jiang; Yingqian Min; Zican Dong; Shijie Wang; Wayne Xin Zhao; Ji-Rong Wen
>
> **备注:** 11 pages, 1 figures, 5 tables
>
> **摘要:** Large reasoning models (LRMs) have exhibited strong performance on complex reasoning tasks, with further gains achievable through increased computational budgets at inference. However, current test-time scaling methods predominantly rely on redundant sampling, ignoring the historical experience utilization, thereby limiting computational efficiency. To overcome this limitation, we propose Sticker-TTS, a novel test-time scaling framework that coordinates three collaborative LRMs to iteratively explore and refine solutions guided by historical attempts. At the core of our framework are distilled key conditions-termed stickers-which drive the extraction, refinement, and reuse of critical information across multiple rounds of reasoning. To further enhance the efficiency and performance of our framework, we introduce a two-stage optimization strategy that combines imitation learning with self-improvement, enabling progressive refinement. Extensive evaluations on three challenging mathematical reasoning benchmarks, including AIME-24, AIME-25, and OlymMATH, demonstrate that Sticker-TTS consistently surpasses strong baselines, including self-consistency and advanced reinforcement learning approaches, under comparable inference budgets. These results highlight the effectiveness of sticker-guided historical experience utilization. Our code and data are available at https://github.com/RUCAIBox/Sticker-TTS.
>
---
#### [replaced 010] Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.03624v4](http://arxiv.org/pdf/2504.03624v4)**

> **作者:** NVIDIA; :; Aaron Blakeman; Aarti Basant; Abhinav Khattar; Adithya Renduchintala; Akhiad Bercovich; Aleksander Ficek; Alexis Bjorlin; Ali Taghibakhshi; Amala Sanjay Deshmukh; Ameya Sunil Mahabaleshwarkar; Andrew Tao; Anna Shors; Ashwath Aithal; Ashwin Poojary; Ayush Dattagupta; Balaram Buddharaju; Bobby Chen; Boris Ginsburg; Boxin Wang; Brandon Norick; Brian Butterfield; Bryan Catanzaro; Carlo del Mundo; Chengyu Dong; Christine Harvey; Christopher Parisien; Dan Su; Daniel Korzekwa; Danny Yin; Daria Gitman; David Mosallanezhad; Deepak Narayanan; Denys Fridman; Dima Rekesh; Ding Ma; Dmytro Pykhtar; Dong Ahn; Duncan Riach; Dusan Stosic; Eileen Long; Elad Segal; Ellie Evans; Eric Chung; Erick Galinkin; Evelina Bakhturina; Ewa Dobrowolska; Fei Jia; Fuxiao Liu; Gargi Prasad; Gerald Shen; Guilin Liu; Guo Chen; Haifeng Qian; Helen Ngo; Hongbin Liu; Hui Li; Igor Gitman; Ilia Karmanov; Ivan Moshkov; Izik Golan; Jan Kautz; Jane Polak Scowcroft; Jared Casper; Jarno Seppanen; Jason Lu; Jason Sewall; Jiaqi Zeng; Jiaxuan You; Jimmy Zhang; Jing Zhang; Jining Huang; Jinze Xue; Jocelyn Huang; Joey Conway; John Kamalu; Jon Barker; Jonathan Cohen; Joseph Jennings; Jupinder Parmar; Karan Sapra; Kari Briski; Kateryna Chumachenko; Katherine Luna; Keshav Santhanam; Kezhi Kong; Kirthi Sivamani; Krzysztof Pawelec; Kumar Anik; Kunlun Li; Lawrence McAfee; Leon Derczynski; Lindsey Pavao; Luis Vega; Lukas Voegtle; Maciej Bala; Maer Rodrigues de Melo; Makesh Narsimhan Sreedhar; Marcin Chochowski; Markus Kliegl; Marta Stepniewska-Dziubinska; Matthieu Le; Matvei Novikov; Mehrzad Samadi; Michael Andersch; Michael Evans; Miguel Martinez; Mike Chrzanowski; Mike Ranzinger; Mikolaj Blaz; Misha Smelyanskiy; Mohamed Fawzy; Mohammad Shoeybi; Mostofa Patwary; Nayeon Lee; Nima Tajbakhsh; Ning Xu; Oleg Rybakov; Oleksii Kuchaiev; Olivier Delalleau; Osvald Nitski; Parth Chadha; Pasha Shamis; Paulius Micikevicius; Pavlo Molchanov; Peter Dykas; Philipp Fischer; Pierre-Yves Aquilanti; Piotr Bialecki; Prasoon Varshney; Pritam Gundecha; Przemek Tredak; Rabeeh Karimi; Rahul Kandu; Ran El-Yaniv; Raviraj Joshi; Roger Waleffe; Ruoxi Zhang; Sabrina Kavanaugh; Sahil Jain; Samuel Kriman; Sangkug Lym; Sanjeev Satheesh; Saurav Muralidharan; Sean Narenthiran; Selvaraj Anandaraj; Seonmyeong Bak; Sergey Kashirsky; Seungju Han; Shantanu Acharya; Shaona Ghosh; Sharath Turuvekere Sreenivas; Sharon Clay; Shelby Thomas; Shrimai Prabhumoye; Shubham Pachori; Shubham Toshniwal; Shyamala Prayaga; Siddhartha Jain; Sirshak Das; Slawek Kierat; Somshubra Majumdar; Song Han; Soumye Singhal; Sriharsha Niverty; Stefania Alborghetti; Suseella Panguluri; Swetha Bhendigeri; Syeda Nahida Akter; Szymon Migacz; Tal Shiri; Terry Kong; Timo Roman; Tomer Ronen; Trisha Saar; Tugrul Konuk; Tuomas Rintamaki; Tyler Poon; Ushnish De; Vahid Noroozi; Varun Singh; Vijay Korthikanti; Vitaly Kurin; Wasi Uddin Ahmad; Wei Du; Wei Ping; Wenliang Dai; Wonmin Byeon; Xiaowei Ren; Yao Xu; Yejin Choi; Yian Zhang; Ying Lin; Yoshi Suhara; Zhiding Yu; Zhiqi Li; Zhiyu Li; Zhongbo Zhu; Zhuolin Yang; Zijia Chen
>
> **摘要:** As inference-time scaling becomes critical for enhanced reasoning capabilities, it is increasingly becoming important to build models that are efficient to infer. We introduce Nemotron-H, a family of 8B and 56B/47B hybrid Mamba-Transformer models designed to reduce inference cost for a given accuracy level. To achieve this goal, we replace the majority of self-attention layers in the common Transformer model architecture with Mamba layers that perform constant computation and require constant memory per generated token. We show that Nemotron-H models offer either better or on-par accuracy compared to other similarly-sized state-of-the-art open-sourced Transformer models (e.g., Qwen-2.5-7B/72B and Llama-3.1-8B/70B), while being up to 3$\times$ faster at inference. To further increase inference speed and reduce the memory required at inference time, we created Nemotron-H-47B-Base from the 56B model using a new compression via pruning and distillation technique called MiniPuzzle. Nemotron-H-47B-Base achieves similar accuracy to the 56B model, but is 20% faster to infer. In addition, we introduce an FP8-based training recipe and show that it can achieve on par results with BF16-based training. This recipe is used to train the 56B model. We are releasing Nemotron-H base model checkpoints with support in Hugging Face and NeMo.
>
---
#### [replaced 011] A Structured Dataset of Disease-Symptom Associations to Improve Diagnostic Accuracy
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.13610v5](http://arxiv.org/pdf/2506.13610v5)**

> **作者:** Abdullah Al Shafi; Rowzatul Zannat; Abdul Muntakim; Mahmudul Hasan
>
> **备注:** Computational Biology
>
> **摘要:** Disease-symptom datasets are significant and in demand for medical research, disease diagnosis, clinical decision-making, and AI-driven health management applications. These datasets help identify symptom patterns associated with specific diseases, thus improving diagnostic accuracy and enabling early detection. The dataset presented in this study systematically compiles disease-symptom relationships from various online sources, medical literature, and publicly available health databases. The data was gathered through analyzing peer-reviewed medical articles, clinical case studies, and disease-symptom association reports. Only the verified medical sources were included in the dataset, while those from non-peer-reviewed and anecdotal sources were excluded. The dataset is structured in a tabular format, where the first column represents diseases, and the remaining columns represent symptoms. Each symptom cell contains a binary value, indicating whether a symptom is associated with a disease. Thereby, this structured representation makes the dataset very useful for a wide range of applications, including machine learning-based disease prediction, clinical decision support systems, and epidemiological studies. Although there are some advancements in the field of disease-symptom datasets, there is a significant gap in structured datasets for the Bangla language. This dataset aims to bridge that gap by facilitating the development of multilingual medical informatics tools and improving disease prediction models for underrepresented linguistic communities. Further developments should include region-specific diseases and further fine-tuning of symptom associations for better diagnostic performance
>
---
#### [replaced 012] Process-Supervised Reward Models for Verifying Clinical Note Generation: A Scalable Approach Guided by Domain Expertise
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.12583v3](http://arxiv.org/pdf/2412.12583v3)**

> **作者:** Hanyin Wang; Chufan Gao; Qiping Xu; Bolun Liu; Guleid Hussein; Hariprasad Korsapati; Mohamad El Labban; Kingsley Iheasirim; Mohamed Hassan; Gokhan Anil; Brian Bartlett; Jimeng Sun
>
> **摘要:** Process-supervised reward models (PRMs) excel at providing step-by-step verification for large language model (LLM) outputs in domains like mathematics and coding. However, their application to fields lacking ground-truth answers, such as clinical note generation, poses significant challenges. We introduce a novel framework for training PRMs to deliver step-level reward signals for LLM-generated clinical notes. By precisely defining meaningful "steps," injecting realistic "errors" informed by domain expertise, and leveraging LLMs to generate process supervision data at scale, we overcome previous limitations. Our PRM, built on LLaMA-3.1 8B, consistently outperforms proprietary reasoning and non-reasoning models, achieving state-of-the-art performance on two key evaluations: (1) distinguishing gold-standard from error-containing samples with 98.8% accuracy, and (2) selecting physician-preferred clinical notes with 56.2% accuracy. We investigate critical components for effective PRM training, including optimal loss functions and data selection strategies, and present a comprehensive physician reader study identifying predictors of downstream Best-of-N performance. Our study sheds light on unlocking the potential of PRMs for diverse generative tasks across domains.
>
---
#### [replaced 013] Exploring the Limits of Large Language Models: A Systematic Evaluation of Masked Text Processing Ability through MskQA and MskCal
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.05665v2](http://arxiv.org/pdf/2411.05665v2)**

> **作者:** Fuka Matsuzaki; Haru-Tada Sato
>
> **备注:** 19 pages
>
> **摘要:** This paper sheds light on the limitations of Large Language Models (LLMs) by rigorously evaluating their ability to process masked text. We introduce two novel tasks: MskQA, measuring reasoning on masked question-answering datasets like RealtimeQA, and MskCal, assessing numerical reasoning on masked arithmetic problems.Testing GPT-4o and 4o-mini reveals that while LLMs exhibit some resilience to masked text, their performance is highly contingent on masking rates and semantic cues. Specifically, "solid masking," where semantic clues are entirely absent, leads to a significant performance drop compared to "partial lifting," where some semantic information is retained, indicating LLMs' reliance on surface-level patterns. Interestingly, GPT-4o consistently outperforms 4o-mini, particularly in MskCal, demonstrating a greater ability to handle numerical reasoning with masked text. This underscores the crucial role of semantic cues in the reasoning process of LLMs. Our study illuminates the interplay between background knowledge and reasoning ability in masked text processing, paving the way for a deeper understanding of LLM capabilities and limitations, and highlighting the need for more robust evaluation methods to accurately assess their true comprehension abilities.
>
---
#### [replaced 014] Through the Prism of Culture: Evaluating LLMs' Understanding of Indian Subcultures and Traditions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.16748v3](http://arxiv.org/pdf/2501.16748v3)**

> **作者:** Garima Chhikara; Abhishek Kumar; Abhijnan Chakraborty
>
> **摘要:** Large Language Models (LLMs) have shown remarkable advancements but also raise concerns about cultural bias, often reflecting dominant narratives at the expense of under-represented subcultures. In this study, we evaluate the capacity of LLMs to recognize and accurately respond to the Little Traditions within Indian society, encompassing localized cultural practices and subcultures such as caste, kinship, marriage, and religion. Through a series of case studies, we assess whether LLMs can balance the interplay between dominant Great Traditions and localized Little Traditions. We explore various prompting strategies and further investigate whether using prompts in regional languages enhances the models cultural sensitivity and response quality. Our findings reveal that while LLMs demonstrate an ability to articulate cultural nuances, they often struggle to apply this understanding in practical, context-specific scenarios. To the best of our knowledge, this is the first study to analyze LLMs engagement with Indian subcultures, offering critical insights into the challenges of embedding cultural diversity in AI systems.
>
---
#### [replaced 015] Building Self-Evolving Agents via Experience-Driven Lifelong Learning: A Framework and Benchmark
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.19005v3](http://arxiv.org/pdf/2508.19005v3)**

> **作者:** Yuxuan Cai; Yipeng Hao; Jie Zhou; Hang Yan; Zhikai Lei; Rui Zhen; Zhenhua Han; Yutao Yang; Junsong Li; Qianjun Pan; Tianyu Huai; Qin Chen; Xin Li; Kai Chen; Bo Zhang; Xipeng Qiu; Liang He
>
> **摘要:** As AI advances toward general intelligence, the focus is shifting from systems optimized for static tasks to creating open-ended agents that learn continuously. In this paper, we introduce Experience-driven Lifelong Learning (ELL), a framework for building self-evolving agents capable of continuous growth through real-world interaction. The framework is built on four core principles: (1) Experience Exploration: Agents learn through continuous, self-motivated interaction with dynamic environments, navigating interdependent tasks and generating rich experiential trajectories. (2) Long-term Memory: Agents preserve and structure historical knowledge, including personal experiences, domain expertise, and commonsense reasoning, into a persistent memory system. (3) Skill Learning: Agents autonomously improve by abstracting recurring patterns from experience into reusable skills, which are actively refined and validated for application in new tasks. (4) Knowledge Internalization: Agents internalize explicit and discrete experiences into implicit and intuitive capabilities as "second nature". We also introduce StuLife, a benchmark dataset for ELL that simulates a student's holistic college journey, from enrollment to academic and personal development, across three core phases and ten detailed sub-scenarios. StuLife is designed around three key paradigm
>
---
#### [replaced 016] Step-level Verifier-guided Hybrid Test-Time Scaling for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.15512v2](http://arxiv.org/pdf/2507.15512v2)**

> **作者:** Kaiyan Chang; Yonghao Shi; Chenglong Wang; Hang Zhou; Chi Hu; Xiaoqian Liu; Yingfeng Luo; Yuan Ge; Tong Xiao; Jingbo Zhu
>
> **备注:** Accepted by EMNLP 2025. Code: https://github.com/Lucky-259/Hybrid_TTS
>
> **摘要:** Test-Time Scaling (TTS) is a promising approach to progressively elicit the model's intelligence during inference. Recently, training-based TTS methods, such as continued reinforcement learning (RL), have further surged in popularity, while training-free TTS methods are gradually fading from prominence. However, the additional computation overhead of training amplifies the burden on test-time scaling. In this paper, we focus on training-free TTS methods for reasoning. We first design Conditional Step-level Self-refinement, a fine-grained sequential scaling method guided by process verification. On top of its effectiveness, we further combine it with other classical parallel scaling methods at the step level, to introduce a novel inference paradigm called Hybrid Test-Time Scaling. Extensive experiments on five instruction-tuned LLMs across different scales (3B-14B) and families demonstrate that hybrid strategy incorporating various training-free TTS methods at a fine granularity has considerable potential for expanding the reasoning performance boundaries of LLMs.
>
---
#### [replaced 017] Repetition Improves Language Model Embeddings
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.15449v2](http://arxiv.org/pdf/2402.15449v2)**

> **作者:** Jacob Mitchell Springer; Suhas Kotha; Daniel Fried; Graham Neubig; Aditi Raghunathan
>
> **备注:** ICLR 2025
>
> **摘要:** Bidirectional models are considered essential for strong text embeddings. Recent approaches to adapt autoregressive language models (LMs) into strong text embedding models have largely had the requirement to modify the LM architecture to be bidirectional. We challenge this premise by introducing "echo embeddings" which converts autoregressive LMs into high quality text embedding models without changing the architecture or requiring fine-tuning. By repeating the input and extracting embeddings from the repeated tokens -- which have access to all original tokens -- echo embeddings improve over classical LM embeddings by over 5% in zero-shot settings. Our zero-shot embeddings nearly match those obtained by bidirectionally-converted LMs that undergo additional masked-language modeling training. Echo embeddings are also compatible with supervised fine-tuning, matching or outperforming bidirectionally-converted LMs in an apples-to-apples comparison, even with an identical compute budget during training and inference. Overall, repetition is a simple and effective strategy to circumvent the need for bidirectional attention in embedding models, paving the way towards a unified architecture for all NLP tasks.
>
---
#### [replaced 018] HierTOD: A Task-Oriented Dialogue System Driven by Hierarchical Goals
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.07152v2](http://arxiv.org/pdf/2411.07152v2)**

> **作者:** Lingbo Mo; Shun Jiang; Akash Maharaj; Bernard Hishamunda; Yunyao Li
>
> **备注:** Accepted to DaSH Workshop at VLDB 2025
>
> **摘要:** Task-Oriented Dialogue (TOD) systems assist users in completing tasks through natural language interactions, often relying on a single-layered workflow structure for slot-filling in public tasks, such as hotel bookings. However, in enterprise environments, which involve rich domain-specific knowledge, TOD systems face challenges due to task complexity and the lack of standardized documentation. In this work, we introduce HierTOD, an enterprise TOD system driven by hierarchical goals that can support composite workflows. By focusing on goal-driven interactions, our system serves a more proactive role, facilitating mixed-initiative dialogue and improving task completion. Equipped with components for natural language understanding, composite goal retriever, dialogue management, and response generation, backed by a well-organized data service with domain knowledge base and retrieval engine, HierTOD delivers efficient task assistance as judged by human evaluators. Furthermore, our system implementation unifies two TOD paradigms: slot-filling for information collection and step-by-step guidance for task execution. Our user study demonstrates the effectiveness and helpfulness of HierTOD in performing both paradigms.
>
---
#### [replaced 019] OmniThink: Expanding Knowledge Boundaries in Machine Writing through Thinking
- **分类: cs.CL; cs.AI; cs.HC; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.09751v3](http://arxiv.org/pdf/2501.09751v3)**

> **作者:** Zekun Xi; Wenbiao Yin; Jizhan Fang; Jialong Wu; Runnan Fang; Jiang Yong; Pengjun Xie; Fei Huang; Huajun Chen; Ningyu Zhang
>
> **备注:** EMNLP 2025
>
> **摘要:** Machine writing with large language models often relies on retrieval-augmented generation. However, these approaches remain confined within the boundaries of the model's predefined scope, limiting the generation of content with rich information. Specifically, vanilla-retrieved information tends to lack depth, novelty, and suffers from redundancy, which negatively impacts the quality of generated articles, leading to shallow, unoriginal, and repetitive outputs. To address these issues, we propose OmniThink, a slow-thinking machine writing framework that emulates the human-like process of iterative expansion and reflection. The core idea behind OmniThink is to simulate the cognitive behavior of learners as they slowly deepen their knowledge of the topics. Experimental results demonstrate that OmniThink improves the knowledge density of generated articles without compromising metrics such as coherence and depth. Human evaluations and expert feedback further highlight the potential of OmniThink to address real-world challenges in the generation of long-form articles. Code is available at https://github.com/zjunlp/OmniThink.
>
---
#### [replaced 020] VisBias: Measuring Explicit and Implicit Social Biases in Vision Language Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.07575v3](http://arxiv.org/pdf/2503.07575v3)**

> **作者:** Jen-tse Huang; Jiantong Qin; Jianping Zhang; Youliang Yuan; Wenxuan Wang; Jieyu Zhao
>
> **备注:** Accepted to EMNLP 2025 (Main)
>
> **摘要:** This research investigates both explicit and implicit social biases exhibited by Vision-Language Models (VLMs). The key distinction between these bias types lies in the level of awareness: explicit bias refers to conscious, intentional biases, while implicit bias operates subconsciously. To analyze explicit bias, we directly pose questions to VLMs related to gender and racial differences: (1) Multiple-choice questions based on a given image (e.g., "What is the education level of the person in the image?") (2) Yes-No comparisons using two images (e.g., "Is the person in the first image more educated than the person in the second image?") For implicit bias, we design tasks where VLMs assist users but reveal biases through their responses: (1) Image description tasks: Models are asked to describe individuals in images, and we analyze disparities in textual cues across demographic groups. (2) Form completion tasks: Models draft a personal information collection form with 20 attributes, and we examine correlations among selected attributes for potential biases. We evaluate Gemini-1.5, GPT-4V, GPT-4o, LLaMA-3.2-Vision and LLaVA-v1.6. Our code and data are publicly available at https://github.com/uscnlp-lime/VisBias.
>
---
#### [replaced 021] ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.04439v2](http://arxiv.org/pdf/2509.04439v2)**

> **作者:** Matthew Ho; Chen Si; Zhaoxiang Feng; Fangxu Yu; Yichi Yang; Zhijian Liu; Zhiting Hu; Lianhui Qin
>
> **摘要:** While inference-time scaling enables LLMs to carry out increasingly long and capable reasoning traces, the patterns and insights uncovered during these traces are immediately discarded once the context window is reset for a new query. External memory is a natural way to persist these discoveries, and recent work has shown clear benefits for reasoning-intensive tasks. We see an opportunity to make such memories more broadly reusable and scalable by moving beyond instance-based memory entries (e.g. exact query/response pairs, or summaries tightly coupled with the original problem context) toward concept-level memory: reusable, modular abstractions distilled from solution traces and stored in natural language. For future queries, relevant concepts are selectively retrieved and integrated into the prompt, enabling test-time continual learning without weight updates. Our design introduces new strategies for abstracting takeaways from rollouts and retrieving entries for new queries, promoting reuse and allowing memory to expand with additional experiences. We evaluate on ARC-AGI, a benchmark that stresses compositional generalization and abstract reasoning, making it a natural fit for concept memory. Our method yields a 7.5% relative gain over a strong no-memory baseline with performance continuing to scale with inference compute. We find abstract concepts to be the most consistent memory design, outscoring the baseline at all tested inference compute scales. Moreover, dynamically updating memory during test-time outperforms fixed settings, supporting the hypothesis that accumulating and abstracting patterns enables further solutions in a form of self-improvement. Code is available at https://github.com/matt-seb-ho/arc_memo.
>
---
#### [replaced 022] CARFT: Boosting LLM Reasoning via Contrastive Learning with Annotated Chain-of-Thought-based Reinforced Fine-Tuning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.15868v2](http://arxiv.org/pdf/2508.15868v2)**

> **作者:** Wenqiao Zhu; Ji Liu; Rongjuncheng Zhang; Haipang Wu; Yulun Zhang
>
> **备注:** 14 pages, to appear in EMNLP25
>
> **摘要:** Reasoning capability plays a significantly critical role in the the broad applications of Large Language Models (LLMs). To enhance the reasoning performance of LLMs, diverse Reinforcement Learning (RL)-based fine-tuning approaches have been proposed to address the limited generalization capability of LLMs trained solely via Supervised Fine-Tuning (SFT). Despite their effectiveness, two major limitations hinder the advancement of LLMs. First, vanilla RL-based approaches ignore annotated Chain-of-Thought (CoT) and incorporate unstable reasoning path sampling, which typically results in model collapse, unstable training process, and suboptimal performance. Second, existing SFT approaches generally overemphasize the annotated CoT, potentially leading to performance degradation due to insufficient exploitation of potential CoT. In this paper, we propose a Contrastive learning with annotated CoT-based Reinforced Fine-Tuning approach, i.e., \TheName{}, to enhance the reasoning performance of LLMs while addressing the aforementioned limitations. Specifically, we propose learning a representation for each CoT. Based on this representation, we design novel contrastive signals to guide the fine-tuning process. Our approach not only fully exploits the available annotated CoT but also stabilizes the fine-tuning procedure by incorporating an additional unsupervised learning signal. We conduct comprehensive experiments and in-depth analysis with three baseline approaches, two foundation models, and two datasets to demonstrate significant advantages of \TheName{} in terms of robustness, performance (up to 10.15\%), and efficiency (up to 30.62\%). Code is available at https://github.com/WNQzhu/CARFT.
>
---
#### [replaced 023] Language Mixing in Reasoning Language Models: Patterns, Impact, and Internal Causes
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14815v2](http://arxiv.org/pdf/2505.14815v2)**

> **作者:** Mingyang Wang; Lukas Lange; Heike Adel; Yunpu Ma; Jannik Strötgen; Hinrich Schütze
>
> **摘要:** Reasoning language models (RLMs) excel at complex tasks by leveraging a chain-of-thought process to generate structured intermediate steps. However, language mixing, i.e., reasoning steps containing tokens from languages other than the prompt, has been observed in their outputs and shown to affect performance, though its impact remains debated. We present the first systematic study of language mixing in RLMs, examining its patterns, impact, and internal causes across 15 languages, 7 task difficulty levels, and 18 subject areas, and show how all three factors influence language mixing. Moreover, we demonstrate that the choice of reasoning language significantly affects performance: forcing models to reason in Latin or Han scripts via constrained decoding notably improves accuracy. Finally, we show that the script composition of reasoning traces closely aligns with that of the model's internal representations, indicating that language mixing reflects latent processing preferences in RLMs. Our findings provide actionable insights for optimizing multilingual reasoning and open new directions for controlling reasoning languages to build more interpretable and adaptable RLMs.
>
---
#### [replaced 024] RADIANT: Retrieval AugmenteD entIty-context AligNmenT -- Introducing RAG-ability and Entity-Context Divergence
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.02949v2](http://arxiv.org/pdf/2507.02949v2)**

> **作者:** Vipula Rawte; Rajarshi Roy; Gurpreet Singh; Danush Khanna; Yaswanth Narsupalli; Basab Ghosh; Abhay Gupta; Argha Kamal Samanta; Aditya Shingote; Aadi Krishna Vikram; Vinija Jain; Aman Chadha; Amit Sheth; Amitava Das
>
> **摘要:** As Large Language Models (LLMs) continue to advance, Retrieval-Augmented Generation (RAG) has emerged as a vital technique to enhance factual accuracy by integrating external knowledge into the generation process. However, LLMs often fail to faithfully integrate retrieved evidence into their generated responses, leading to factual inconsistencies. To quantify this gap, we introduce Entity-Context Divergence (ECD), a metric that measures the extent to which retrieved information is accurately reflected in model outputs. We systematically evaluate contemporary LLMs on their ability to preserve factual consistency in retrieval-augmented settings, a capability we define as RAG-ability. Our empirical analysis reveals that RAG-ability remains low across most LLMs, highlighting significant challenges in entity retention and context fidelity. This paper introduces Radiant (Retrieval AugmenteD entIty-context AligNmenT), a novel framework that merges RAG with alignment designed to optimize the interplay between retrieved evidence and generated content. Radiant extends Direct Preference Optimization (DPO) to teach LLMs how to integrate provided additional information into subsequent generations. As a behavior correction mechanism, Radiant boosts RAG performance across varied retrieval scenarios, such as noisy web contexts, knowledge conflicts, and hallucination reduction. This enables more reliable, contextually grounded, and factually coherent content generation.
>
---
#### [replaced 025] Knowledge Editing through Chain-of-Thought
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.17727v2](http://arxiv.org/pdf/2412.17727v2)**

> **作者:** Changyue Wang; Weihang Su; Qingyao Ai; Yichen Tang; Yiqun Liu
>
> **摘要:** Knowledge Editing is a technique that updates large language models (LLMs) with new information to maintain their world knowledge. This approach avoids the need to rebuild the model from scratch, thereby addressing the high costs associated with frequent retraining. Among these, the in-context editing paradigm stands out for its effectiveness in integrating new knowledge while preserving the model's original capabilities. Despite its potential, existing in-context knowledge editing methods are often task-specific, focusing primarily on multi-hop QA tasks using structured knowledge triples. Moreover, their reliance on few-shot prompting for task decomposition makes them unstable and less effective in generalizing across diverse tasks. In response to these limitations, we propose EditCoT, a novel knowledge editing framework that flexibly and efficiently updates LLMs across various tasks without retraining. EditCoT works by generating a chain-of-thought (CoT) for a given input and then iteratively refining this CoT process using a CoT editor based on updated knowledge. We evaluate EditCoT across a diverse range of benchmarks, covering multiple languages and tasks. The results demonstrate that our approach achieves state-of-the-art performance while offering superior generalization, effectiveness, and stability compared to existing methods, marking a significant advancement in the field of knowledge updating. The code and data of EditCoT are available at: https://github.com/bebr2/EditCoT .
>
---
#### [replaced 026] Revealing the impact of synthetic native samples and multi-tasking strategies in Hindi-English code-mixed humour and sarcasm detection
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.12761v2](http://arxiv.org/pdf/2412.12761v2)**

> **作者:** Debajyoti Mazumder; Aakash Kumar; Jasabanta Patro
>
> **备注:** 33 pages; EMNLP 2025 (Findings)
>
> **摘要:** In this paper, we reported our experiments with various strategies to improve code-mixed humour and sarcasm detection. Particularly, we tried three approaches: (i) native sample mixing, (ii) multi-task learning (MTL), and (iii) prompting and instruction finetuning very large multilingual language models (VMLMs). In native sample mixing, we added monolingual task samples to code-mixed training sets. In MTL learning, we relied on native and code-mixed samples of a semantically related task (hate detection in our case). Finally, in our third approach, we evaluated the efficacy of VMLMs via few-shot context prompting and instruction finetuning. Some interesting findings we got are (i) adding native samples improved humor (raising the F1-score up to 6.76%) and sarcasm (raising the F1-score up to 8.64%) detection, (ii) training MLMs in an MTL framework boosted performance for both humour (raising the F1-score up to 10.67%) and sarcasm (increment up to 12.35% in F1-score) detection, and (iii) prompting and instruction finetuning VMLMs couldn't outperform the other approaches. Finally, our ablation studies and error analysis discovered the cases where our model is yet to improve. We provided our code for reproducibility.
>
---
#### [replaced 027] EMNLP: Educator-role Moral and Normative Large Language Models Profiling
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2508.15250v2](http://arxiv.org/pdf/2508.15250v2)**

> **作者:** Yilin Jiang; Mingzi Zhang; Sheng Jin; Zengyi Yu; Xiangjie Kong; Binghao Tu
>
> **备注:** 29pages, 15 figures, Accepted by EMNLP Main Confrence
>
> **摘要:** Simulating Professions (SP) enables Large Language Models (LLMs) to emulate professional roles. However, comprehensive psychological and ethical evaluation in these contexts remains lacking. This paper introduces EMNLP, an Educator-role Moral and Normative LLMs Profiling framework for personality profiling, moral development stage measurement, and ethical risk under soft prompt injection. EMNLP extends existing scales and constructs 88 teacher-specific moral dilemmas, enabling profession-oriented comparison with human teachers. A targeted soft prompt injection set evaluates compliance and vulnerability in teacher SP. Experiments on 14 LLMs show teacher-role LLMs exhibit more idealized and polarized personalities than human teachers, excel in abstract moral reasoning, but struggle with emotionally complex situations. Models with stronger reasoning are more vulnerable to harmful prompt injection, revealing a paradox between capability and safety. The model temperature and other hyperparameters have limited influence except in some risk behaviors. This paper presents the first benchmark to assess ethical and psychological alignment of teacher-role LLMs for educational AI. Resources are available at https://e-m-n-l-p.github.io/.
>
---
#### [replaced 028] Error Classification of Large Language Models on Math Word Problems: A Dynamically Adaptive Framework
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.15581v2](http://arxiv.org/pdf/2501.15581v2)**

> **作者:** Yuhong Sun; Zhangyue Yin; Xuanjing Huang; Xipeng Qiu; Hui Zhao
>
> **备注:** 28 pages, 10 figures, accepted by Findings of EMNLP2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains. Math Word Problems (MWPs) serve as a crucial benchmark for evaluating LLMs' reasoning abilities. While most research primarily focuses on improving accuracy, it often neglects understanding and addressing the underlying patterns of errors. Current error classification methods rely on static and predefined categories, which limit their ability to capture the full spectrum of error patterns in mathematical reasoning. To enable systematic error analysis, we collect error samples from 15 different LLMs of varying sizes across four distinct MWP datasets using multiple sampling strategies. Based on this extensive collection, we introduce MWPES-300K, a comprehensive dataset containing 304,865 error samples that cover diverse error patterns and reasoning paths. To reduce human bias and enable fine-grained analysis of error patterns, we propose a novel framework for automated dynamic error classification in mathematical reasoning. Experimental results demonstrate that dataset characteristics significantly shape error patterns, which evolve from basic to complex manifestations as model capabilities increase. With deeper insights into error patterns, we propose Error-Aware Prompting (EAP) that incorporates common error patterns as explicit guidance, leading to significant improvements in mathematical reasoning performance.
>
---
#### [replaced 029] FinAgentBench: A Benchmark Dataset for Agentic Retrieval in Financial Question Answering
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.14052v3](http://arxiv.org/pdf/2508.14052v3)**

> **作者:** Chanyeol Choi; Jihoon Kwon; Alejandro Lopez-Lira; Chaewoon Kim; Minjae Kim; Juneha Hwang; Jaeseon Ha; Hojun Choi; Suyeol Yun; Yongjin Kim; Yongjae Lee
>
> **备注:** 6 pages
>
> **摘要:** Accurate information retrieval (IR) is critical in the financial domain, where investors must identify relevant information from large collections of documents. Traditional IR methods-whether sparse or dense-often fall short in retrieval accuracy, as it requires not only capturing semantic similarity but also performing fine-grained reasoning over document structure and domain-specific knowledge. Recent advances in large language models (LLMs) have opened up new opportunities for retrieval with multi-step reasoning, where the model ranks passages through iterative reasoning about which information is most relevant to a given query. However, there exists no benchmark to evaluate such capabilities in the financial domain. To address this gap, we introduce FinAgentBench, the first large-scale benchmark for evaluating retrieval with multi-step reasoning in finance -- a setting we term agentic retrieval. The benchmark consists of 3,429 expert-annotated examples on S&P-100 listed firms and assesses whether LLM agents can (1) identify the most relevant document type among candidates, and (2) pinpoint the key passage within the selected document. Our evaluation framework explicitly separates these two reasoning steps to address context limitations. This design enables to provide a quantitative basis for understanding retrieval-centric LLM behavior in finance. We evaluate a suite of state-of-the-art models and further demonstrated how targeted fine-tuning can significantly improve agentic retrieval performance. Our benchmark provides a foundation for studying retrieval-centric LLM behavior in complex, domain-specific tasks for finance.
>
---
#### [replaced 030] Probe-Rewrite-Evaluate: A Workflow for Reliable Benchmarks and Quantifying Evaluation Awareness
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.00591v3](http://arxiv.org/pdf/2509.00591v3)**

> **作者:** Lang Xiong; Nishant Bhargava; Jeremy Chang; Jianhang Hong; Haihao Liu; Vasu Sharma; Kevin Zhu
>
> **摘要:** Large Language Models (LLMs) often exhibit significant behavioral shifts when they perceive a change from a real-world deployment context to a controlled evaluation setting, a phenomenon known as "evaluation awareness." This discrepancy poses a critical challenge for AI alignment, as benchmark performance may not accurately reflect a model's true safety and honesty. In this work, we systematically quantify these behavioral changes by manipulating the perceived context of prompts. We introduce a methodology that uses a linear probe to score prompts on a continuous scale from "test-like" to "deploy-like" and leverage an LLM rewriting strategy to shift these prompts towards a more natural, deployment-style context while preserving the original task. Using this method, we achieved a 30% increase in the average probe score across a strategic role-playing dataset after rewriting. Evaluating a suite of state-of-the-art models on these original and rewritten prompts, we find that rewritten "deploy-like" prompts induce a significant and consistent shift in behavior. Across all models, we observed an average increase in honest responses of 5.26% and a corresponding average decrease in deceptive responses of 12.40%. Furthermore, refusal rates increased by an average of 6.38%, indicating heightened safety compliance. Our findings demonstrate that evaluation awareness is a quantifiable and manipulable factor that directly influences LLM behavior, revealing that models are more prone to unsafe or deceptive outputs in perceived test environments. This underscores the urgent need for more realistic evaluation frameworks to accurately gauge true model alignment before deployment.
>
---
#### [replaced 031] Lessons from Studying Two-Hop Latent Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.16353v3](http://arxiv.org/pdf/2411.16353v3)**

> **作者:** Mikita Balesni; Tomek Korbak; Owain Evans
>
> **摘要:** Large language models can use chain-of-thought (CoT) to externalize reasoning, potentially enabling oversight of capable LLM agents. Prior work has shown that models struggle at two-hop question-answering without CoT. This capability is so basic that if it was a fundamental limitation, it would imply that many complex agentic tasks would similarly require CoT. We investigate LLM latent reasoning capabilities using two-hop question answering as a case study. Previous work on the gap between latent and externalized two-hop reasoning produced mixed evidence with inconclusive results. In this paper, we introduce a controlled setting for investigating two-hop reasoning in LLMs, where a positive result provides definitive evidence for latent reasoning. We fine-tune LLMs (including Llama 3 8B and GPT-4o) on synthetic facts and test two-hop reasoning over these facts. By using synthetic facts, we rule out memorization and reasoning shortcuts as explanations for two-hop performance. We observe a nuanced picture: Models fail to compose two synthetic facts, but can succeed when one fact is synthetic and the other is natural. These results demonstrate that LLMs are undeniably capable of latent two-hop reasoning, although it remains unclear how this ability scales with model size. Finally, we highlight a lesson for researchers studying LLM reasoning: when drawing conclusions about LLM latent reasoning, one must be careful to avoid both spurious successes (that stem from memorization and reasoning shortcuts) and spurious failures (that may stem from artificial experimental setups, divorced from training setups of frontier LLMs).
>
---
#### [replaced 032] CVPD at QIAS 2025 Shared Task: An Efficient Encoder-Based Approach for Islamic Inheritance Reasoning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.00457v2](http://arxiv.org/pdf/2509.00457v2)**

> **作者:** Salah Eddine Bekhouche; Abdellah Zakaria Sellam; Hichem Telli; Cosimo Distante; Abdenour Hadid
>
> **摘要:** Islamic inheritance law (Ilm al-Mawarith) requires precise identification of heirs and calculation of shares, which poses a challenge for AI. In this paper, we present a lightweight framework for solving multiple-choice inheritance questions using a specialised Arabic text encoder and Attentive Relevance Scoring (ARS). The system ranks answer options according to semantic relevance, and enables fast, on-device inference without generative reasoning. We evaluate Arabic encoders (MARBERT, ArabicBERT, AraBERT) and compare them with API-based LLMs (Gemini, DeepSeek) on the QIAS 2025 dataset. While large models achieve an accuracy of up to 87.6%, they require more resources and are context-dependent. Our MARBERT-based approach achieves 69.87% accuracy, presenting a compelling case for efficiency, on-device deployability, and privacy. While this is lower than the 87.6% achieved by the best-performing LLM, our work quantifies a critical trade-off between the peak performance of large models and the practical advantages of smaller, specialized systems in high-stakes domains.
>
---
#### [replaced 033] GASE: Generatively Augmented Sentence Encoding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.04914v2](http://arxiv.org/pdf/2411.04914v2)**

> **作者:** Manuel Frank; Haithem Afli
>
> **备注:** EMNLP Findings 2025
>
> **摘要:** We propose a training-free approach to improve sentence embeddings leveraging test-time compute by applying generative text models for data augmentation at inference time. Unlike conventional data augmentation that utilises synthetic training data, our approach does not require access to model parameters or the computational resources typically required for fine-tuning state-of-the-art models. Generatively Augmented Sentence Encoding variates the input text by paraphrasing, summarising, or extracting keywords, followed by pooling the original and synthetic embeddings. Experimental results on the Massive Text Embedding Benchmark for Semantic Textual Similarity (STS) demonstrate performance improvements across a range of embedding models using different generative models for augmentation. We find that generative augmentation leads to larger performance improvements for embedding models with lower baseline performance. These findings suggest that integrating generative augmentation at inference time adds semantic diversity and can enhance the robustness and generalisability of sentence embeddings for embedding models. Our results show that performance gains depend on the embedding model and the dataset.
>
---
#### [replaced 034] A Principled Framework for Evaluating on Typologically Diverse Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.05022v3](http://arxiv.org/pdf/2407.05022v3)**

> **作者:** Esther Ploeger; Wessel Poelman; Andreas Holck Høeg-Petersen; Anders Schlichtkrull; Miryam de Lhoneux; Johannes Bjerva
>
> **备注:** Revised version
>
> **摘要:** Beyond individual languages, multilingual natural language processing (NLP) research increasingly aims to develop models that perform well across languages generally. However, evaluating these systems on all the world's languages is practically infeasible. To attain generalizability, representative language sampling is essential. Previous work argues that generalizable multilingual evaluation sets should contain languages with diverse typological properties. However, 'typologically diverse' language samples have been found to vary considerably in this regard, and popular sampling methods are flawed and inconsistent. We present a language sampling framework for selecting highly typologically diverse languages given a sampling frame, informed by language typology. We compare sampling methods with a range of metrics and find that our systematic methods consistently retrieve more typologically diverse language selections than previous methods in NLP. Moreover, we provide evidence that this affects generalizability in multilingual model evaluation, emphasizing the importance of diverse language sampling in NLP evaluation.
>
---
#### [replaced 035] Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.15884v2](http://arxiv.org/pdf/2508.15884v2)**

> **作者:** Yuxian Gu; Qinghao Hu; Shang Yang; Haocheng Xi; Junyu Chen; Song Han; Han Cai
>
> **备注:** Tech Report
>
> **摘要:** We present Jet-Nemotron, a new family of hybrid-architecture language models, which matches or exceeds the accuracy of leading full-attention models while significantly improving generation throughput. Jet-Nemotron is developed using Post Neural Architecture Search (PostNAS), a novel neural architecture exploration pipeline that enables efficient model design. Unlike prior approaches, PostNAS begins with a pre-trained full-attention model and freezes its MLP weights, allowing efficient exploration of attention block designs. The pipeline includes four key components: (1) learning optimal full-attention layer placement and elimination, (2) linear attention block selection, (3) designing new attention blocks, and (4) performing hardware-aware hyperparameter search. Our Jet-Nemotron-2B model achieves comparable or superior accuracy to Qwen3, Qwen2.5, Gemma3, and Llama3.2 across a comprehensive suite of benchmarks while delivering up to 53.6x generation throughput speedup and 6.1x prefilling speedup. It also achieves higher accuracy on MMLU and MMLU-Pro than recent advanced MoE full-attention models, such as DeepSeek-V3-Small and Moonlight, despite their larger scale with 15B total and 2.2B activated parameters.
>
---
#### [replaced 036] Dynamic Injection of Entity Knowledge into Dense Retrievers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.03922v2](http://arxiv.org/pdf/2507.03922v2)**

> **作者:** Ikuya Yamada; Ryokan Ri; Takeshi Kojima; Yusuke Iwasawa; Yutaka Matsuo
>
> **备注:** EMNLP Findings
>
> **摘要:** Dense retrievers often struggle with queries involving less-frequent entities due to their limited entity knowledge. We propose the Knowledgeable Passage Retriever (KPR), a BERT-based retriever enhanced with a context-entity attention layer and dynamically updatable entity embeddings. This design enables KPR to incorporate external entity knowledge without retraining. Experiments on three datasets demonstrate that KPR consistently improves retrieval accuracy, with particularly large gains on the EntityQuestions dataset. When built on the off-the-shelf bge-base retriever, KPR achieves state-of-the-art performance among similarly sized models on two datasets. Models and code are released at https://github.com/knowledgeable-embedding/knowledgeable-embedding.
>
---
#### [replaced 037] PlainQAFact: Retrieval-augmented Factual Consistency Evaluation Metric for Biomedical Plain Language Summarization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.08890v2](http://arxiv.org/pdf/2503.08890v2)**

> **作者:** Zhiwen You; Yue Guo
>
> **摘要:** Hallucinated outputs from large language models (LLMs) pose risks in the medical domain, especially for lay audiences making health-related decisions. Existing automatic factual consistency evaluation methods, such as entailment- and question-answering (QA) -based, struggle with plain language summarization (PLS) due to elaborative explanation phenomenon, which introduces external content (e.g., definitions, background, examples) absent from the scientific abstract to enhance comprehension. To address this, we introduce PlainQAFact, an automatic factual consistency evaluation metric trained on a fine-grained, human-annotated dataset PlainFact, for evaluating factual consistency of both source-simplified and elaborately explained sentences. PlainQAFact first classifies sentence type, then applies a retrieval-augmented QA scoring method. Empirical results show that existing evaluation metrics fail to evaluate the factual consistency in PLS, especially for elaborative explanations, whereas PlainQAFact consistently outperforms them across all evaluation settings. We further analyze PlainQAFact's effectiveness across external knowledge sources, answer extraction strategies, answer overlap measures, and document granularity levels, refining its overall factual consistency assessment. Taken together, our work presents the first evaluation metric designed for PLS factual consistency evaluation, providing the community with both a robust benchmark and a practical tool to advance reliable and safe plain language communication in the medical domain. PlainQAFact and PlainFact are available at: https://github.com/zhiwenyou103/PlainQAFact
>
---
#### [replaced 038] E-THER: A Multimodal Dataset for Empathic AI - Towards Emotional Mismatch Awareness
- **分类: cs.HC; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.02100v2](http://arxiv.org/pdf/2509.02100v2)**

> **作者:** Sharjeel Tahir; Judith Johnson; Jumana Abu-Khalaf; Syed Afaq Ali Shah
>
> **备注:** 15 pages, 4 figures. Preprint
>
> **摘要:** A prevalent shortfall among current empathic AI systems is their inability to recognize when verbal expressions may not fully reflect underlying emotional states. This is because the existing datasets, used for the training of these systems, focus on surface-level emotion recognition without addressing the complex verbal-visual incongruence (mismatch) patterns useful for empathic understanding. In this paper, we present E-THER, the first Person-Centered Therapy-grounded multimodal dataset with multidimensional annotations for verbal-visual incongruence detection, enabling training of AI systems that develop genuine rather than performative empathic capabilities. The annotations included in the dataset are drawn from humanistic approach, i.e., identifying verbal-visual emotional misalignment in client-counsellor interactions - forming a framework for training and evaluating AI on empathy tasks. Additional engagement scores provide behavioral annotations for research applications. Notable gains in empathic and therapeutic conversational qualities are observed in state-of-the-art vision-language models (VLMs), such as IDEFICS and VideoLLAVA, using evaluation metrics grounded in empathic and therapeutic principles. Empirical findings indicate that our incongruence-trained models outperform general-purpose models in critical traits, such as sustaining therapeutic engagement, minimizing artificial or exaggerated linguistic patterns, and maintaining fidelity to PCT theoretical framework.
>
---
#### [replaced 039] Affective Computing in the Era of Large Language Models: A Survey from the NLP Perspective
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2408.04638v2](http://arxiv.org/pdf/2408.04638v2)**

> **作者:** Yiqun Zhang; Xiaocui Yang; Xingle Xu; Zeran Gao; Yijie Huang; Shiyi Mu; Shi Feng; Daling Wang; Yifei Zhang; Kaisong Song; Ge Yu
>
> **备注:** Compared with the previous version, reinforcement learning has been added (as a new section), including RLHF, RLVR, and RLAIF
>
> **摘要:** Affective Computing (AC) integrates computer science, psychology, and cognitive science to enable machines to recognize, interpret, and simulate human emotions across domains such as social media, finance, healthcare, and education. AC commonly centers on two task families: Affective Understanding (AU) and Affective Generation (AG). While fine-tuned pre-trained language models (PLMs) have achieved solid AU performance, they often generalize poorly across tasks and remain limited for AG, especially in producing diverse, emotionally appropriate responses. The advent of Large Language Models (LLMs) (e.g., ChatGPT and LLaMA) has catalyzed a paradigm shift by offering in-context learning, broader world knowledge, and stronger sequence generation. This survey presents an NLP-oriented overview of AC in the LLM era. We (i) consolidate traditional AC tasks and preliminary LLM-based studies; (ii) review adaptation techniques that improve AU/AG, including Instruction Tuning (full and parameter-efficient methods such as LoRA, P-/Prompt-Tuning), Prompt Engineering (zero/few-shot, chain-of-thought, agent-based prompting), and Reinforcement Learning. For the latter, we summarize RL from human preferences (RLHF), verifiable/programmatic rewards (RLVR), and AI feedback (RLAIF), which provide preference- or rule-grounded optimization signals that can help steer AU/AG toward empathy, safety, and planning, achieving finer-grained or multi-objective control. To assess progress, we compile benchmarks and evaluation practices for both AU and AG. We also discuss open challenges-from ethics, data quality, and safety to robust evaluation and resource efficiency-and outline research directions. We hope this survey clarifies the landscape and offers practical guidance for building affect-aware, reliable, and responsible LLM systems.
>
---
#### [replaced 040] CodeMixBench: Evaluating Code-Mixing Capabilities of LLMs Across 18 Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.18791v2](http://arxiv.org/pdf/2507.18791v2)**

> **作者:** Yilun Yang; Yekun Chai
>
> **备注:** EMNLP 2025
>
> **摘要:** Code-mixing, the practice of switching between languages within a conversation, poses unique challenges for traditional NLP. Existing benchmarks are limited by their narrow language pairs and tasks, failing to adequately assess large language models' (LLMs) code-mixing abilities. Despite the recognized importance of code-mixing for multilingual users, research on LLMs in this context remains sparse. Additionally, current techniques for synthesizing code-mixed data are underdeveloped to generate code-mixing. In response, we introduce CodeMixBench, a comprehensive benchmark covering eight tasks, including three specific to LLMs and five traditional NLP tasks, and 18 languages across seven language families. We also propose a new method for generating large-scale synthetic code-mixed texts by combining word substitution with GPT-4 prompting. Our evaluation reveals consistent underperformance of LLMs on code-mixed datasets involving different language families. Enhancements in training data size, model scale, and few-shot learning could improve their performance. The code and dataset are available at https://github.com/Jeromeyluck/CodeMixBench.
>
---
#### [replaced 041] MoSEs: Uncertainty-Aware AI-Generated Text Detection via Mixture of Stylistics Experts with Conditional Thresholds
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.02499v3](http://arxiv.org/pdf/2509.02499v3)**

> **作者:** Junxi Wu; Jinpeng Wang; Zheng Liu; Bin Chen; Dongjian Hu; Hao Wu; Shu-Tao Xia
>
> **备注:** EMNLP 2025
>
> **摘要:** The rapid advancement of large language models has intensified public concerns about the potential misuse. Therefore, it is important to build trustworthy AI-generated text detection systems. Existing methods neglect stylistic modeling and mostly rely on static thresholds, which greatly limits the detection performance. In this paper, we propose the Mixture of Stylistic Experts (MoSEs) framework that enables stylistics-aware uncertainty quantification through conditional threshold estimation. MoSEs contain three core components, namely, the Stylistics Reference Repository (SRR), the Stylistics-Aware Router (SAR), and the Conditional Threshold Estimator (CTE). For input text, SRR can activate the appropriate reference data in SRR and provide them to CTE. Subsequently, CTE jointly models the linguistic statistical properties and semantic features to dynamically determine the optimal threshold. With a discrimination score, MoSEs yields prediction labels with the corresponding confidence level. Our framework achieves an average improvement 11.34% in detection performance compared to baselines. More inspiringly, MoSEs shows a more evident improvement 39.15% in the low-resource case. Our code is available at https://github.com/creator-xi/MoSEs.
>
---
#### [replaced 042] Concept Bottleneck Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.07992v4](http://arxiv.org/pdf/2412.07992v4)**

> **作者:** Chung-En Sun; Tuomas Oikarinen; Berk Ustun; Tsui-Wei Weng
>
> **备注:** Accepted to ICLR 2025
>
> **摘要:** We introduce Concept Bottleneck Large Language Models (CB-LLMs), a novel framework for building inherently interpretable Large Language Models (LLMs). In contrast to traditional black-box LLMs that rely on limited post-hoc interpretations, CB-LLMs integrate intrinsic interpretability directly into the LLMs -- allowing accurate explanations with scalability and transparency. We build CB-LLMs for two essential NLP tasks: text classification and text generation. In text classification, CB-LLMs is competitive with, and at times outperforms, traditional black-box models while providing explicit and interpretable reasoning. For the more challenging task of text generation, interpretable neurons in CB-LLMs enable precise concept detection, controlled generation, and safer outputs. The embedded interpretability empowers users to transparently identify harmful content, steer model behavior, and unlearn undesired concepts -- significantly enhancing the safety, reliability, and trustworthiness of LLMs, which are critical capabilities notably absent in existing models. Our code is available at https://github.com/Trustworthy-ML-Lab/CB-LLMs.
>
---
#### [replaced 043] Think-to-Talk or Talk-to-Think? When LLMs Come Up with an Answer in Multi-Hop Arithmetic Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.01113v3](http://arxiv.org/pdf/2412.01113v3)**

> **作者:** Keito Kudo; Yoichi Aoki; Tatsuki Kuribayashi; Shusaku Sone; Masaya Taniguchi; Ana Brassard; Keisuke Sakaguchi; Kentaro Inui
>
> **摘要:** This study investigates the incremental, internal problem-solving process of language models (LMs) with arithmetic multi-hop reasoning as a case study. We specifically investigate when LMs internally resolve sub/whole problems through first reading the problem statements, generating reasoning chains, and achieving the final answer to mechanistically interpret LMs' multi-hop problem-solving process. Our experiments reveal a systematic incremental reasoning strategy underlying LMs. They have not derived an answer at the moment they first read the problem; instead, they obtain (sub)answers while generating the reasoning chain. Therefore, the generated reasoning chains can be regarded as faithful reflections of the model's internal computation.
>
---
#### [replaced 044] Soft Token Attacks Cannot Reliably Audit Unlearning in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.15836v2](http://arxiv.org/pdf/2502.15836v2)**

> **作者:** Haokun Chen; Sebastian Szyller; Weilin Xu; Nageen Himayat
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Large language models (LLMs) are trained using massive datasets, which often contain undesirable content such as harmful texts, personal information, and copyrighted material. To address this, machine unlearning aims to remove information from trained models. Recent work has shown that soft token attacks (STA) can successfully extract unlearned information from LLMs, but in this work we show that STAs can be an inadequate tool for auditing unlearning. Using common benchmarks such as Who Is Harry Potter? and TOFU, we demonstrate that in a strong auditor setting such attacks can elicit any information from the LLM, regardless of the deployed unlearning algorithm or whether the queried content was originally present in the training corpus. We further show that STA with just a few soft tokens (1-10) can elicit random strings over 400 characters long, indicating that STAs must be used carefully to effectively audit unlearning. Example code can be found at: https://github.com/IntelLabs/LLMart/tree/main/examples/unlearning
>
---
#### [replaced 045] Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.01872v4](http://arxiv.org/pdf/2501.01872v4)**

> **作者:** Rachneet Sachdeva; Rima Hazra; Iryna Gurevych
>
> **备注:** Accepted at EMNLP 2025 (Main)
>
> **摘要:** Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.
>
---
#### [replaced 046] Rhapsody: A Dataset for Highlight Detection in Podcasts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19429v2](http://arxiv.org/pdf/2505.19429v2)**

> **作者:** Younghan Park; Anuj Diwan; David Harwath; Eunsol Choi
>
> **备注:** COLM 2025
>
> **摘要:** Podcasts have become daily companions for half a billion users. Given the enormous amount of podcast content available, highlights provide a valuable signal that helps viewers get the gist of an episode and decide if they want to invest in listening to it in its entirety. However, identifying highlights automatically is challenging due to the unstructured and long-form nature of the content. We introduce Rhapsody, a dataset of 13K podcast episodes paired with segment-level highlight scores derived from YouTube's 'most replayed' feature. We frame the podcast highlight detection as a segment-level binary classification task. We explore various baseline approaches, including zero-shot prompting of language models and lightweight fine-tuned language models using segment-level classification heads. Our experimental results indicate that even state-of-the-art language models like GPT-4o and Gemini struggle with this task, while models fine-tuned with in-domain data significantly outperform their zero-shot performance. The fine-tuned model benefits from leveraging both speech signal features and transcripts. These findings highlight the challenges for fine-grained information access in long-form spoken media.
>
---
#### [replaced 047] Fast Quiet-STaR: Thinking Without Thought Tokens
- **分类: cs.CL; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.17746v2](http://arxiv.org/pdf/2505.17746v2)**

> **作者:** Wei Huang; Yizhe Xiong; Xin Ye; Zhijie Deng; Hui Chen; Zijia Lin; Guiguang Ding
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Large Language Models (LLMs) have achieved impressive performance across a range of natural language processing tasks. However, recent advances demonstrate that further gains particularly in complex reasoning tasks require more than merely scaling up model sizes or training data. One promising direction is to enable models to think during the reasoning process. Recently, Quiet STaR significantly improves reasoning by generating token-level thought traces, but incurs substantial inference overhead. In this work, we propose Fast Quiet STaR, a more efficient reasoning framework that preserves the benefits of token-level reasoning while reducing computational cost. Our method introduces a curriculum learning based training strategy that gradually reduces the number of thought tokens, enabling the model to internalize more abstract and concise reasoning processes. We further extend this approach to the standard Next Token Prediction (NTP) setting through reinforcement learning-based fine-tuning, resulting in Fast Quiet-STaR NTP, which eliminates the need for explicit thought token generation during inference. Experiments on four benchmark datasets with Mistral 7B and Qwen2.5 7B demonstrate that Fast Quiet-STaR consistently outperforms Quiet-STaR in terms of average accuracy under the same inference time budget. Notably, Fast Quiet-STaR NTP achieves an average accuracy improvement of 9\% on Mistral 7B and 5.7\% on Qwen2.5 7B, while maintaining the same inference latency. Our code will be available at https://github.com/huangwei200012/Fast-Quiet-STaR.
>
---
#### [replaced 048] Energy Landscapes Enable Reliable Abstention in Retrieval-Augmented Large Language Models for Healthcare
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.04482v2](http://arxiv.org/pdf/2509.04482v2)**

> **作者:** Ravi Shankar; Sheng Wong; Lin Li; Magdalena Bachmann; Alex Silverthorne; Beth Albert; Gabriel Davis Jones
>
> **摘要:** Reliable abstention is critical for retrieval-augmented generation (RAG) systems, particularly in safety-critical domains such as women's health, where incorrect answers can lead to harm. We present an energy-based model (EBM) that learns a smooth energy landscape over a dense semantic corpus of 2.6M guideline-derived questions, enabling the system to decide when to generate or abstain. We benchmark the EBM against a calibrated softmax baseline and a k-nearest neighbour (kNN) density heuristic across both easy and hard abstention splits, where hard cases are semantically challenging near-distribution queries. The EBM achieves superior abstention performance abstention on semantically hard cases, reaching AUROC 0.961 versus 0.950 for softmax, while also reducing FPR@95 (0.235 vs 0.331). On easy negatives, performance is comparable across methods, but the EBM's advantage becomes most pronounced in safety-critical hard distributions. A comprehensive ablation with controlled negative sampling and fair data exposure shows that robustness stems primarily from the energy scoring head, while the inclusion or exclusion of specific negative types (hard, easy, mixed) sharpens decision boundaries but is not essential for generalisation to hard cases. These results demonstrate that energy-based abstention scoring offers a more reliable confidence signal than probability-based softmax confidence, providing a scalable and interpretable foundation for safe RAG systems.
>
---
#### [replaced 049] Extracting and Combining Abilities For Building Multi-lingual Ability-enhanced Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.07825v3](http://arxiv.org/pdf/2410.07825v3)**

> **作者:** Zhipeng Chen; Kun Zhou; Liang Song; Wayne Xin Zhao; Bingning Wang; Weipeng Chen; Ji-Rong Wen
>
> **备注:** EMNLP 2025 Main Conference
>
> **摘要:** Multi-lingual ability transfer has become increasingly important for the broad application of large language models (LLMs). Existing work highly relies on training with the multi-lingual ability-related data, which may not be available for low-resource languages. To solve it, we propose a Multi-lingual Abilities Extraction and Combination approach, named as MAEC. Our key idea is to decompose and extract language-agnostic ability-related weights from LLMs, and combine them across different languages by simple addition and subtraction operations without training. Specifically, our MAEC consists of the extraction and combination stages. In the extraction stage, we firstly locate key neurons that are highly related to specific abilities, and then employ them to extract the transferable ability-related weights. In the combination stage, we further select the ability-related tensors that mitigate the linguistic effects, and design a combining strategy based on them and the language-specific weights, to build the multi-lingual ability-enhanced LLM. To assess the effectiveness of our approach, we conduct extensive experiments on LLaMA-3 8B on mathematical and scientific tasks in both high-resource and low-resource lingual scenarios. Experiment results have shown that MAEC can effectively and efficiently extract and combine the advanced abilities, achieving comparable performance with PaLM. Resources are available at https://github.com/RUCAIBox/MAET.
>
---
#### [replaced 050] Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.03646v2](http://arxiv.org/pdf/2509.03646v2)**

> **作者:** Haozhe Wang; Qixin Xu; Che Liu; Junhong Wu; Fangzhen Lin; Wenhu Chen
>
> **备注:** Preprint
>
> **摘要:** Reinforcement Learning (RL) has proven highly effective at enhancing the complex reasoning abilities of Large Language Models (LLMs), yet underlying mechanisms driving this success remain largely opaque. Our analysis reveals that puzzling phenomena like ``aha moments", ``length-scaling'' and entropy dynamics are not disparate occurrences but hallmarks of an emergent reasoning hierarchy, akin to the separation of high-level strategic planning from low-level procedural execution in human cognition. We uncover a compelling two-phase dynamic: initially, a model is constrained by procedural correctness and must improve its low-level skills. The learning bottleneck then decisively shifts, with performance gains being driven by the exploration and mastery of high-level strategic planning. This insight exposes a core inefficiency in prevailing RL algorithms like GRPO, which apply optimization pressure agnostically and dilute the learning signal across all tokens. To address this, we propose HIerarchy-Aware Credit Assignment (HICRA), an algorithm that concentrates optimization efforts on high-impact planning tokens. HICRA significantly outperforms strong baselines, demonstrating that focusing on this strategic bottleneck is key to unlocking advanced reasoning. Furthermore, we validate semantic entropy as a superior compass for measuring strategic exploration over misleading metrics such as token-level entropy.
>
---
#### [replaced 051] Oyster-I: Beyond Refusal -- Constructive Safety Alignment for Responsible Language Models
- **分类: cs.AI; cs.CL; cs.CY; cs.HC; cs.SC**

- **链接: [http://arxiv.org/pdf/2509.01909v3](http://arxiv.org/pdf/2509.01909v3)**

> **作者:** Ranjie Duan; Jiexi Liu; Xiaojun Jia; Shiji Zhao; Ruoxi Cheng; Fengxiang Wang; Cheng Wei; Yong Xie; Chang Liu; Defeng Li; Yinpeng Dong; Yichi Zhang; Yuefeng Chen; Chongwen Wang; Xingjun Ma; Xingxing Wei; Yang Liu; Hang Su; Jun Zhu; Xinfeng Li; Yitong Sun; Jie Zhang; Jinzhao Hu; Sha Xu; Yitong Yang; Jialing Tao; Hui Xue
>
> **备注:** Technical Report Code & Model weights available: https://github.com/Alibaba-AAIG/Oyster
>
> **摘要:** Large language models (LLMs) typically deploy safety mechanisms to prevent harmful content generation. Most current approaches focus narrowly on risks posed by malicious actors, often framing risks as adversarial events and relying on defensive refusals. However, in real-world settings, risks also come from non-malicious users seeking help while under psychological distress (e.g., self-harm intentions). In such cases, the model's response can strongly influence the user's next actions. Simple refusals may lead them to repeat, escalate, or move to unsafe platforms, creating worse outcomes. We introduce Constructive Safety Alignment (CSA), a human-centric paradigm that protects against malicious misuse while actively guiding vulnerable users toward safe and helpful results. Implemented in Oyster-I (Oy1), CSA combines game-theoretic anticipation of user reactions, fine-grained risk boundary discovery, and interpretable reasoning control, turning safety into a trust-building process. Oy1 achieves state-of-the-art safety among open models while retaining high general capabilities. On our Constructive Benchmark, it shows strong constructive engagement, close to GPT-5, and unmatched robustness on the Strata-Sword jailbreak dataset, nearing GPT-o1 levels. By shifting from refusal-first to guidance-first safety, CSA redefines the model-user relationship, aiming for systems that are not just safe, but meaningfully helpful. We release Oy1, code, and the benchmark to support responsible, user-centered AI.
>
---
#### [replaced 052] Transforming Wearable Data into Personal Health Insights using Large Language Model Agents
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.06464v4](http://arxiv.org/pdf/2406.06464v4)**

> **作者:** Mike A. Merrill; Akshay Paruchuri; Naghmeh Rezaei; Geza Kovacs; Javier Perez; Yun Liu; Erik Schenck; Nova Hammerquist; Jake Sunshine; Shyam Tailor; Kumar Ayush; Hao-Wei Su; Qian He; Cory Y. McLean; Mark Malhotra; Shwetak Patel; Jiening Zhan; Tim Althoff; Daniel McDuff; Xin Liu
>
> **备注:** 53 pages, 7 main figures, 2 main tables, accepted to Nature Communications
>
> **摘要:** Deriving personalized insights from popular wearable trackers requires complex numerical reasoning that challenges standard LLMs, necessitating tool-based approaches like code generation. Large language model (LLM) agents present a promising yet largely untapped solution for this analysis at scale. We introduce the Personal Health Insights Agent (PHIA), a system leveraging multistep reasoning with code generation and information retrieval to analyze and interpret behavioral health data. To test its capabilities, we create and share two benchmark datasets with over 4000 health insights questions. A 650-hour human expert evaluation shows that PHIA significantly outperforms a strong code generation baseline, achieving 84% accuracy on objective, numerical questions and, for open-ended ones, earning 83% favorable ratings while being twice as likely to achieve the highest quality rating. This work can advance behavioral health by empowering individuals to understand their data, enabling a new era of accessible, personalized, and data-driven wellness for the wider population.
>
---
#### [replaced 053] ResearchArena: Benchmarking Large Language Models' Ability to Collect and Organize Information as Research Agents
- **分类: cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2406.10291v3](http://arxiv.org/pdf/2406.10291v3)**

> **作者:** Hao Kang; Chenyan Xiong
>
> **摘要:** Large language models (LLMs) excel across many natural language processing tasks but face challenges in domain-specific, analytical tasks such as conducting research surveys. This study introduces ResearchArena, a benchmark designed to evaluate LLMs' capabilities in conducting academic surveys -- a foundational step in academic research. ResearchArena models the process in three stages: (1) information discovery, identifying relevant literature; (2) information selection, evaluating papers' relevance and impact; and (3) information organization, structuring knowledge into hierarchical frameworks such as mind-maps. Notably, mind-map construction is treated as a bonus task, reflecting its supplementary role in survey-writing. To support these evaluations, we construct an offline environment of 12M full-text academic papers and 7.9K survey papers. To ensure ethical compliance, we do not redistribute copyrighted materials; instead, we provide code to construct the environment from the Semantic Scholar Open Research Corpus (S2ORC). Preliminary evaluations reveal that LLM-based approaches underperform compared to simpler keyword-based retrieval methods, though recent reasoning models such as DeepSeek-R1 show slightly better zero-shot performance. These results underscore significant opportunities for advancing LLMs in autonomous research. We open-source the code to construct the ResearchArena benchmark at https://github.com/cxcscmu/ResearchArena.
>
---
#### [replaced 054] Empathy Omni: Enabling Empathetic Speech Response Generation through Large Language Models
- **分类: cs.CL; cs.SD; eess.AS; I.2.7**

- **链接: [http://arxiv.org/pdf/2508.18655v2](http://arxiv.org/pdf/2508.18655v2)**

> **作者:** Haoyu Wang; Guangyan Zhang; Jiale Chen; Jingyu Li; Yuehai Wang; Yiwen Guo
>
> **备注:** 5 pages, 1 figure, submitted to ICASSP 2026
>
> **摘要:** With the development of speech large language models (speech LLMs), users can now interact directly with assistants via speech. However, most existing models only convert response content into speech without fully capturing the rich emotional cues in user queries, where the same sentence may convey different meanings depending on the expression. Emotional understanding is thus essential for improving human-machine interaction. Most empathetic speech LLMs rely on massive datasets, demanding high computational cost. A key challenge is to build models that generate empathetic responses with limited data and without large-scale training. To this end, we propose Emotion Omni, a model that understands emotional content in user speech and generates empathetic responses. We further developed a data pipeline to construct a 200k emotional dialogue dataset supporting empathetic speech assistants. Experiments show that Emotion Omni achieves comparable instruction-following ability without large-scale pretraining, while surpassing existing models in speech quality (UTMOS:4.41) and empathy (Emotion GPT Score: 3.97). These results confirm its improvements in both speech fidelity and emotional expressiveness. Demos are available at https://w311411.github.io/omni_demo/.
>
---
#### [replaced 055] Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17656v3](http://arxiv.org/pdf/2505.17656v3)**

> **作者:** Hexiang Tan; Fei Sun; Sha Liu; Du Su; Qi Cao; Xin Chen; Jingang Wang; Xunliang Cai; Yuanzhuo Wang; Huawei Shen; Xueqi Cheng
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** As large language models (LLMs) often generate plausible but incorrect content, error detection has become increasingly critical to ensure truthfulness. However, existing detection methods often overlook a critical problem we term as self-consistent error, where LLMs repeatedly generate the same incorrect response across multiple stochastic samples. This work formally defines self-consistent errors and evaluates mainstream detection methods on them. Our investigation reveals two key findings: (1) Unlike inconsistent errors, whose frequency diminishes significantly as the LLM scale increases, the frequency of self-consistent errors remains stable or even increases. (2) All four types of detection methods significantly struggle to detect self-consistent errors. These findings reveal critical limitations in current detection methods and underscore the need for improvement. Motivated by the observation that self-consistent errors often differ across LLMs, we propose a simple but effective cross-model probe method that fuses hidden state evidence from an external verifier LLM. Our method significantly enhances performance on self-consistent errors across three LLM families.
>
---
#### [replaced 056] SUDER: Self-Improving Unified Large Multimodal Models for Understanding and Generation with Dual Self-Rewards
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07963v3](http://arxiv.org/pdf/2506.07963v3)**

> **作者:** Jixiang Hong; Yiran Zhang; Guanzhong Wang; Yi Liu; Ji-Rong Wen; Rui Yan
>
> **摘要:** Building upon large language models (LLMs), recent large multimodal models (LMMs) unify cross-model understanding and generation into a single framework. However, LMMs still struggle to achieve accurate vision-language alignment, prone to generating text responses contradicting the visual input or failing to follow the text-to-image prompts. Current solutions require external supervision (e.g., human feedback or reward models) and only address unidirectional tasks-either understanding or generation. In this work, based on the observation that understanding and generation are naturally inverse dual tasks, we propose \textbf{SUDER} (\textbf{S}elf-improving \textbf{U}nified LMMs with \textbf{D}ual s\textbf{E}lf-\textbf{R}ewards), a framework reinforcing the understanding and generation capabilities of LMMs with a self-supervised dual reward mechanism. SUDER leverages the inherent duality between understanding and generation tasks to provide self-supervised optimization signals for each other. Specifically, we sample multiple outputs for a given input in one task domain, then reverse the input-output pairs to compute the dual likelihood within the model as self-rewards for optimization. Extensive experimental results on visual understanding and generation benchmarks demonstrate that our method can effectively enhance the performance of the model without any external supervision, especially achieving remarkable improvements in text-to-image tasks.
>
---
#### [replaced 057] Evaluating the Robustness and Accuracy of Text Watermarking Under Real-World Cross-Lingual Manipulations
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.16699v2](http://arxiv.org/pdf/2502.16699v2)**

> **作者:** Mansour Al Ghanim; Jiaqi Xue; Rochana Prih Hastuti; Mengxin Zheng; Yan Solihin; Qian Lou
>
> **备注:** Accepted by EMNLP 2025 Finding
>
> **摘要:** We present a study to benchmark representative watermarking methods in cross-lingual settings. The current literature mainly focuses on the evaluation of watermarking methods for the English language. However, the literature for evaluating watermarking in cross-lingual settings is scarce. This results in overlooking important adversary scenarios in which a cross-lingual adversary could be in, leading to a gray area of practicality over cross-lingual watermarking. In this paper, we evaluate four watermarking methods in four different and vocabulary rich languages. Our experiments investigate the quality of text under different watermarking procedure and the detectability of watermarks with practical translation attack scenarios. Specifically, we investigate practical scenarios that an adversary with cross-lingual knowledge could take, and evaluate whether current watermarking methods are suitable for such scenarios. Finally, from our findings, we draw key insights about watermarking in cross-lingual settings.
>
---
#### [replaced 058] Efficient Dynamic Clustering-Based Document Compression for Retrieval-Augmented-Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.03165v3](http://arxiv.org/pdf/2504.03165v3)**

> **作者:** Weitao Li; Kaiming Liu; Xiangyu Zhang; Xuanyu Lei; Weizhi Ma; Yang Liu
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a widely adopted approach for knowledge injection during large language model (LLM) inference in recent years. However, due to their limited ability to exploit fine-grained inter-document relationships, current RAG implementations face challenges in effectively addressing the retrieved noise and redundancy content, which may cause error in the generation results. To address these limitations, we propose an Efficient Dynamic Clustering-based document Compression framework (EDC2-RAG) that utilizes latent inter-document relationships while simultaneously removing irrelevant information and redundant content. We validate our approach, built upon GPT-3.5-Turbo and GPT-4o-mini, on widely used knowledge-QA and Hallucination-Detection datasets. Experimental results show that our method achieves consistent performance improvements across various scenarios and experimental settings, demonstrating strong robustness and applicability. Our code and datasets are available at https://github.com/Tsinghua-dhy/EDC-2-RAG.
>
---
#### [replaced 059] Learning to Reason for Long-Form Story Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.22828v2](http://arxiv.org/pdf/2503.22828v2)**

> **作者:** Alexander Gurung; Mirella Lapata
>
> **摘要:** Generating high-quality stories spanning thousands of tokens requires competency across a variety of skills, from tracking plot and character arcs to keeping a consistent and engaging style. Due to the difficulty of sourcing labeled datasets and precise quality measurements, most work using large language models (LLMs) for long-form story generation uses combinations of hand-designed prompting techniques to elicit author-like behavior. This is a manual process that is highly dependent on the specific story-generation task. Motivated by the recent success of applying RL with Verifiable Rewards to domains like math and coding, we propose a general story-generation task (Next-Chapter Prediction) and a reward formulation (Verified Rewards via Completion Likelihood Improvement) that allows us to use an unlabeled book dataset as a learning signal for reasoning. We learn to reason over a story's condensed information and generate a detailed plan for the next chapter. Our reasoning is evaluated via the chapters it helps a story-generator create, and compared against non-trained and supervised finetuning (SFT) baselines. Pairwise human judgments reveal the chapters our learned reasoning produces are preferred across almost all metrics, and the effect is more pronounced in Scifi and Fantasy genres.
>
---
#### [replaced 060] Advancing Scientific Text Classification: Fine-Tuned Models with Dataset Expansion and Hard-Voting
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.19021v2](http://arxiv.org/pdf/2504.19021v2)**

> **作者:** Zhyar Rzgar K Rostam; Gábor Kertész
>
> **备注:** 6 pages, 1 figure, 8 tables
>
> **摘要:** Efficient text classification is essential for handling the increasing volume of academic publications. This study explores the use of pre-trained language models (PLMs), including BERT, SciBERT, BioBERT, and BlueBERT, fine-tuned on the Web of Science (WoS-46985) dataset for scientific text classification. To enhance performance, we augment the dataset by executing seven targeted queries in the WoS database, retrieving 1,000 articles per category aligned with WoS-46985's main classes. PLMs predict labels for this unlabeled data, and a hard-voting strategy combines predictions for improved accuracy and confidence. Fine-tuning on the expanded dataset with dynamic learning rates and early stopping significantly boosts classification accuracy, especially in specialized domains. Domain-specific models like SciBERT and BioBERT consistently outperform general-purpose models such as BERT. These findings underscore the efficacy of dataset augmentation, inference-driven label prediction, hard-voting, and fine-tuning techniques in creating robust and scalable solutions for automated academic text classification.
>
---
#### [replaced 061] ElectroVizQA: How well do Multi-modal LLMs perform in Electronics Visual Question Answering?
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.00102v2](http://arxiv.org/pdf/2412.00102v2)**

> **作者:** Pragati Shuddhodhan Meshram; Swetha Karthikeyan; Bhavya Bhavya; Suma Bhat
>
> **摘要:** Multi-modal Large Language Models (MLLMs) are gaining significant attention for their ability to process multi-modal data, providing enhanced contextual understanding of complex problems. MLLMs have demonstrated exceptional capabilities in tasks such as Visual Question Answering (VQA); however, they often struggle with fundamental engineering problems, and there is a scarcity of specialized datasets for training on topics like digital electronics. To address this gap, we propose a benchmark dataset called ElectroVizQA specifically designed to evaluate MLLMs' performance on digital electronic circuit problems commonly found in undergraduate curricula. This dataset, the first of its kind tailored for the VQA task in digital electronics, comprises approximately 626 visual questions, offering a comprehensive overview of digital electronics topics. This paper rigorously assesses the extent to which MLLMs can understand and solve digital electronic circuit questions, providing insights into their capabilities and limitations within this specialized domain. By introducing this benchmark dataset, we aim to motivate further research and development in the application of MLLMs to engineering education, ultimately bridging the performance gap and enhancing the efficacy of these models in technical fields.
>
---
#### [replaced 062] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.13111v2](http://arxiv.org/pdf/2503.13111v2)**

> **作者:** Erik Daxberger; Nina Wenzel; David Griffiths; Haiming Gang; Justin Lazarow; Gefen Kohavi; Kai Kang; Marcin Eichner; Yinfei Yang; Afshin Dehghan; Peter Grasch
>
> **备注:** ICCV 2025
>
> **摘要:** Multimodal large language models (MLLMs) excel at 2D visual understanding but remain limited in their ability to reason about 3D space. In this work, we leverage large-scale high-quality 3D scene data with open-set annotations to introduce 1) a novel supervised fine-tuning dataset and 2) a new evaluation benchmark, focused on indoor scenes. Our Cubify Anything VQA (CA-VQA) data covers diverse spatial tasks including spatial relationship prediction, metric size and distance estimation, and 3D grounding. We show that CA-VQA enables us to train MM-Spatial, a strong generalist MLLM that also achieves state-of-the-art performance on 3D spatial understanding benchmarks, including our own. We show how incorporating metric depth and multi-view inputs (provided in CA-VQA) can further improve 3D understanding, and demonstrate that data alone allows our model to achieve depth perception capabilities comparable to dedicated monocular depth estimation models.
>
---
#### [replaced 063] BeSimulator: A Large Language Model Powered Text-based Behavior Simulator
- **分类: cs.RO; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.15865v2](http://arxiv.org/pdf/2409.15865v2)**

> **作者:** Jianan Wang; Bin Li; Jingtao Qi; Xueying Wang; Fu Li; Hanxun Li
>
> **备注:** 19 pages, 5 figures, 8 tables
>
> **摘要:** Traditional robot simulators focus on physical process modeling and realistic rendering, often suffering from high computational costs, inefficiencies, and limited adaptability. To handle this issue, we concentrate on behavior simulation in robotics to analyze and validate the logic behind robot behaviors, aiming to achieve preliminary evaluation before deploying resource-intensive simulators and thus enhance simulation efficiency. In this paper, we propose BeSimulator, a modular and novel LLM-powered framework, as an attempt towards behavior simulation in the context of text-based environments. By constructing text-based virtual environments and performing semantic-level simulation, BeSimulator can generalize across scenarios and achieve long-horizon complex simulation. Inspired by human cognition paradigm, it employs a ``consider-decide-capture-transfer'' four-phase simulation process, termed Chain of Behavior Simulation (CBS), which excels at analyzing action feasibility and state transition. Additionally, BeSimulator incorporates code-driven reasoning to enable arithmetic operations and enhance reliability, and reflective feedback to refine simulation. Based on our manually constructed behavior-tree-based simulation benchmark, BTSIMBENCH, our experiments show a significant performance improvement in behavior simulation compared to baselines, ranging from 13.60% to 24.80%. Code and data are available at https://github.com/Dawn888888/BeSimulator.
>
---
#### [replaced 064] X-EcoMLA: Upcycling Pre-Trained Attention into MLA for Efficient and Extreme KV Compression
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.11132v4](http://arxiv.org/pdf/2503.11132v4)**

> **作者:** Guihong Li; Mehdi Rezagholizadeh; Mingyu Yang; Vikram Appia; Emad Barsoum
>
> **摘要:** Multi-head latent attention (MLA) is designed to optimize KV cache memory through low-rank key-value joint compression. Rather than caching keys and values separately, MLA stores their compressed latent representations, reducing memory overhead while maintaining the performance. While MLA improves memory efficiency without compromising language model accuracy, its major limitation lies in its integration during the pre-training phase, requiring models to be trained from scratch. This raises a key question: can we use MLA's benefits fully or partially in models that have already been pre-trained with different attention mechanisms? In this paper, we propose X-EcoMLA to deploy post training distillation to enable the upcycling of Transformer-based attention into an efficient hybrid MLA variant through lightweight post-training adaptation, bypassing the need for extensive pre-training. We demonstrate that leveraging the dark knowledge of a well-trained model can enhance training accuracy and enable extreme KV cache compression in MLA without compromising model performance. The experimental results show that our proposed method can effectively compress the KV cache while preserving the performance on the benchmarks; specifically, for Llama3.2-1B-Instruct baseline, a 6.4x compression achieves the same average score by using only 3.6B training tokens and 70 GPU hours on AMD MI300, whereas a 10.6x compression have less than 0.1% average score drop with 7B training tokens and 140 GPU hours. The code for this work is available at https://github.com/AMD-AGI/AMD-Hybrid-Models.
>
---
#### [replaced 065] Comparative Analysis of Transformer Models in Disaster Tweet Classification for Public Safety
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.04650v2](http://arxiv.org/pdf/2509.04650v2)**

> **作者:** Sharif Noor Zisad; N. M. Istiak Chowdhury; Ragib Hasan
>
> **摘要:** Twitter and other social media platforms have become vital sources of real time information during disasters and public safety emergencies. Automatically classifying disaster related tweets can help emergency services respond faster and more effectively. Traditional Machine Learning (ML) models such as Logistic Regression, Naive Bayes, and Support Vector Machines have been widely used for this task, but they often fail to understand the context or deeper meaning of words, especially when the language is informal, metaphorical, or ambiguous. We posit that, in this context, transformer based models can perform better than traditional ML models. In this paper, we evaluate the effectiveness of transformer based models, including BERT, DistilBERT, RoBERTa, and DeBERTa, for classifying disaster related tweets. These models are compared with traditional ML approaches to highlight the performance gap. Experimental results show that BERT achieved the highest accuracy (91%), significantly outperforming traditional models like Logistic Regression and Naive Bayes (both at 82%). The use of contextual embeddings and attention mechanisms allows transformer models to better understand subtle language in tweets, where traditional ML models fall short. This research demonstrates that transformer architectures are far more suitable for public safety applications, offering improved accuracy, deeper language understanding, and better generalization across real world social media text.
>
---
#### [replaced 066] Low-Confidence Gold: Refining Low-Confidence Samples for Efficient Instruction Tuning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.18978v5](http://arxiv.org/pdf/2502.18978v5)**

> **作者:** Hongyi Cai; Jie Li; Mohammad Mahdinur Rahman; Wenzhen Dong
>
> **备注:** Accepted to EMNLP Findings 2025
>
> **摘要:** The effectiveness of instruction fine-tuning for Large Language Models is fundamentally constrained by the quality and efficiency of training datasets. This work introduces Low-Confidence Gold (LCG), a novel filtering framework that employs centroid-based clustering and confidence-guided selection for identifying valuable instruction pairs. Through a semi-supervised approach using a lightweight classifier trained on representative samples, LCG curates high-quality subsets while preserving data diversity. Experimental evaluation demonstrates that models fine-tuned on LCG-filtered subsets of 6K samples achieve superior performance compared to existing methods, with substantial improvements on MT-bench and consistent gains across comprehensive evaluation metrics. The framework's efficacy while maintaining model performance establishes a promising direction for efficient instruction tuning.
>
---
#### [replaced 067] Self-Critique and Refinement for Faithful Natural Language Explanations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22823v2](http://arxiv.org/pdf/2505.22823v2)**

> **作者:** Yingming Wang; Pepa Atanasova
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** With the rapid development of Large Language Models (LLMs), Natural Language Explanations (NLEs) have become increasingly important for understanding model predictions. However, these explanations often fail to faithfully represent the model's actual reasoning process. While existing work has demonstrated that LLMs can self-critique and refine their initial outputs for various tasks, this capability remains unexplored for improving explanation faithfulness. To address this gap, we introduce Self-critique and Refinement for Natural Language Explanations (SR-NLE), a framework that enables models to improve the faithfulness of their own explanations -- specifically, post-hoc NLEs -- through an iterative critique and refinement process without external supervision. Our framework leverages different feedback mechanisms to guide the refinement process, including natural language self-feedback and, notably, a novel feedback approach based on feature attribution that highlights important input words. Our experiments across three datasets and four state-of-the-art LLMs demonstrate that SR-NLE significantly reduces unfaithfulness rates, with our best method achieving an average unfaithfulness rate of 36.02%, compared to 54.81% for baseline -- an absolute reduction of 18.79%. These findings reveal that the investigated LLMs can indeed refine their explanations to better reflect their actual reasoning process, requiring only appropriate guidance through feedback without additional training or fine-tuning.
>
---
#### [replaced 068] An LLM + ASP Workflow for Joint Entity-Relation Extraction
- **分类: cs.AI; cs.CL; I.2.7; F.4.1**

- **链接: [http://arxiv.org/pdf/2508.12611v2](http://arxiv.org/pdf/2508.12611v2)**

> **作者:** Trang Tran; Trung Hoang Le; Huiping Cao; Tran Cao Son
>
> **备注:** 13 pages, 1 figure, Accepted as Technical Communication, 41st International Conference on Logic Programming
>
> **摘要:** Joint entity-relation extraction (JERE) identifies both entities and their relationships simultaneously. Traditional machine-learning based approaches to performing this task require a large corpus of annotated data and lack the ability to easily incorporate domain specific information in the construction of the model. Therefore, creating a model for JERE is often labor intensive, time consuming, and elaboration intolerant. In this paper, we propose harnessing the capabilities of generative pretrained large language models (LLMs) and the knowledge representation and reasoning capabilities of Answer Set Programming (ASP) to perform JERE. We present a generic workflow for JERE using LLMs and ASP. The workflow is generic in the sense that it can be applied for JERE in any domain. It takes advantage of LLM's capability in natural language understanding in that it works directly with unannotated text. It exploits the elaboration tolerant feature of ASP in that no modification of its core program is required when additional domain specific knowledge, in the form of type specifications, is found and needs to be used. We demonstrate the usefulness of the proposed workflow through experiments with limited training data on three well-known benchmarks for JERE. The results of our experiments show that the LLM + ASP workflow is better than state-of-the-art JERE systems in several categories with only 10\% of training data. It is able to achieve a 2.5 times (35\% over 15\%) improvement in the Relation Extraction task for the SciERC corpus, one of the most difficult benchmarks.
>
---
#### [replaced 069] HoPE: Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.05218v2](http://arxiv.org/pdf/2509.05218v2)**

> **作者:** Chang Dai; Hongyu Shan; Mingyang Song; Di Liang
>
> **摘要:** Positional encoding mechanisms enable Transformers to model sequential structure and long-range dependencies in text. While absolute positional encodings struggle with extrapolation to longer sequences due to fixed positional representations, and relative approaches like Alibi exhibit performance degradation on extremely long contexts, the widely-used Rotary Positional Encoding (RoPE) introduces oscillatory attention patterns that hinder stable long-distance dependency modelling. We address these limitations through a geometric reformulation of positional encoding. Drawing inspiration from Lorentz transformations in hyperbolic geometry, we propose Hyperbolic Rotary Positional Encoding (HoPE), which leverages hyperbolic functions to implement Lorentz rotations on token representations. Theoretical analysis demonstrates that RoPE is a special case of our generalized formulation. HoPE fundamentally resolves RoPE's slation issues by enforcing monotonic decay of attention weights with increasing token distances. Extensive experimental results, including perplexity evaluations under several extended sequence benchmarks, show that HoPE consistently exceeds existing positional encoding methods. These findings underscore HoPE's enhanced capacity for representing and generalizing long-range dependencies. Data and code will be available.
>
---
#### [replaced 070] Leveraging Large Language Models for Accurate Sign Language Translation in Low-Resource Scenarios
- **分类: cs.CL; cs.AI; cs.CY; I.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2508.18183v2](http://arxiv.org/pdf/2508.18183v2)**

> **作者:** Luana Bulla; Gabriele Tuccio; Misael Mongiovì; Aldo Gangemi
>
> **摘要:** Translating natural languages into sign languages is a highly complex and underexplored task. Despite growing interest in accessibility and inclusivity, the development of robust translation systems remains hindered by the limited availability of parallel corpora which align natural language with sign language data. Existing methods often struggle to generalize in these data-scarce environments, as the few datasets available are typically domain-specific, lack standardization, or fail to capture the full linguistic richness of sign languages. To address this limitation, we propose Advanced Use of LLMs for Sign Language Translation (AulSign), a novel method that leverages Large Language Models via dynamic prompting and in-context learning with sample selection and subsequent sign association. Despite their impressive abilities in processing text, LLMs lack intrinsic knowledge of sign languages; therefore, they are unable to natively perform this kind of translation. To overcome this limitation, we associate the signs with compact descriptions in natural language and instruct the model to use them. We evaluate our method on both English and Italian languages using SignBank+, a recognized benchmark in the field, as well as the Italian LaCAM CNR-ISTC dataset. We demonstrate superior performance compared to state-of-the-art models in low-data scenario. Our findings demonstrate the effectiveness of AulSign, with the potential to enhance accessibility and inclusivity in communication technologies for underrepresented linguistic communities.
>
---
#### [replaced 071] OpenDeception: Benchmarking and Investigating AI Deceptive Behaviors via Open-ended Interaction Simulation
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.13707v2](http://arxiv.org/pdf/2504.13707v2)**

> **作者:** Yichen Wu; Xudong Pan; Geng Hong; Min Yang
>
> **摘要:** As the general capabilities of large language models (LLMs) improve and agent applications become more widespread, the underlying deception risks urgently require systematic evaluation and effective oversight. Unlike existing evaluation which uses simulated games or presents limited choices, we introduce OpenDeception, a novel deception evaluation framework with an open-ended scenario dataset. OpenDeception jointly evaluates both the deception intention and capabilities of LLM-based agents by inspecting their internal reasoning process. Specifically, we construct five types of common use cases where LLMs intensively interact with the user, each consisting of ten diverse, concrete scenarios from the real world. To avoid ethical concerns and costs of high-risk deceptive interactions with human testers, we propose to simulate the multi-turn dialogue via agent simulation. Extensive evaluation of eleven mainstream LLMs on OpenDeception highlights the urgent need to address deception risks and security concerns in LLM-based agents: the deception intention ratio across the models exceeds 80%, while the deception success rate surpasses 50%. Furthermore, we observe that LLMs with stronger capabilities do exhibit a higher risk of deception, which calls for more alignment efforts on inhibiting deceptive behaviors.
>
---
#### [replaced 072] Antidistillation Sampling
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.13146v4](http://arxiv.org/pdf/2504.13146v4)**

> **作者:** Yash Savani; Asher Trockman; Zhili Feng; Yixuan Even Xu; Avi Schwarzschild; Alexander Robey; Marc Finzi; J. Zico Kolter
>
> **摘要:** Frontier models that generate extended reasoning traces inadvertently produce rich token sequences that can facilitate model distillation. Recognizing this vulnerability, model owners may seek sampling strategies that limit the effectiveness of distillation without compromising model performance. Antidistillation sampling provides exactly this capability. By strategically modifying a model's next-token probability distribution, antidistillation sampling poisons reasoning traces, rendering them significantly less effective for distillation while preserving the model's practical utility. For further details, see https://antidistillation.com.
>
---
#### [replaced 073] Not All Features Deserve Attention: Graph-Guided Dependency Learning for Tabular Data Generation with Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.18504v2](http://arxiv.org/pdf/2507.18504v2)**

> **作者:** Zheyu Zhang; Shuo Yang; Bardh Prenkaj; Gjergji Kasneci
>
> **备注:** Accepted to EMNLP 2025 (Findings)
>
> **摘要:** Large Language Models (LLMs) have shown strong potential for tabular data generation by modeling textualized feature-value pairs. However, tabular data inherently exhibits sparse feature-level dependencies, where many feature interactions are structurally insignificant. This creates a fundamental mismatch as LLMs' self-attention mechanism inevitably distributes focus across all pairs, diluting attention on critical relationships, particularly in datasets with complex dependencies or semantically ambiguous features. To address this limitation, we propose GraDe (Graph-Guided Dependency Learning), a novel method that explicitly integrates sparse dependency graphs into LLMs' attention mechanism. GraDe employs a lightweight dynamic graph learning module guided by externally extracted functional dependencies, prioritizing key feature interactions while suppressing irrelevant ones. Our experiments across diverse real-world datasets demonstrate that GraDe outperforms existing LLM-based approaches by up to 12% on complex datasets while achieving competitive results with state-of-the-art approaches in synthetic data quality. Our method is minimally intrusive yet effective, offering a practical solution for structure-aware tabular data modeling with LLMs.
>
---
#### [replaced 074] Improve LLM-as-a-Judge Ability as a General Ability
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11689v2](http://arxiv.org/pdf/2502.11689v2)**

> **作者:** Jiachen Yu; Shaoning Sun; Xiaohui Hu; Jiaxu Yan; Kaidong Yu; Xuelong Li
>
> **摘要:** LLM-as-a-Judge leverages the generative and reasoning capabilities of large language models (LLMs) to evaluate LLM responses across diverse scenarios, providing accurate preference signals. This approach plays a vital role in aligning LLMs with human values, ensuring ethical and reliable AI outputs that align with societal norms. Recent studies have raised many methods to train LLM as generative judges, but most of them are data consuming or lack accuracy, and only focus on LLM's judge ability. In this work, we regard judge ability as a general ability of LLM and implement a two-stage training approach, comprising supervised fine-tuning (SFT) warm-up and direct preference optimization (DPO) enhancement, to achieve judge style adaptation and improve judgment accuracy. Additionally, we introduce an efficient data synthesis method to generate judgmental content. Experimental results demonstrate that our approach, utilizing only about 2% to 40% of the data required by other methods, achieves SOTA performance on RewardBench. Furthermore, our training method enhances the general capabilities of the model by constructing complicated judge task, and the judge signals provided by our model have significantly enhanced the downstream DPO training performance of our internal models in our test to optimize policy model with Judge Model. We also open-source our model weights and training data to facilitate further research.
>
---
#### [replaced 075] Reinforced Lifelong Editing for Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.05759v4](http://arxiv.org/pdf/2502.05759v4)**

> **作者:** Zherui Li; Houcheng Jiang; Hao Chen; Baolong Bi; Zhenhong Zhou; Fei Sun; Junfeng Fang; Xiang Wang
>
> **备注:** Accepted by ICML2025
>
> **摘要:** Large language models (LLMs) acquire information from pre-training corpora, but their stored knowledge can become inaccurate or outdated over time. Model editing addresses this challenge by modifying model parameters without retraining, and prevalent approaches leverage hypernetworks to generate these parameter updates. However, they face significant challenges in lifelong editing due to their incompatibility with LLM parameters that dynamically change during the editing process. To address this, we observed that hypernetwork-based lifelong editing aligns with reinforcement learning modeling and proposed RLEdit, an RL-based editing method. By treating editing losses as rewards and optimizing hypernetwork parameters at the full knowledge sequence level, we enable it to precisely capture LLM changes and generate appropriate parameter updates. Our extensive empirical evaluation across several LLMs demonstrates that RLEdit outperforms existing methods in lifelong editing with superior effectiveness and efficiency, achieving a 59.24% improvement while requiring only 2.11% of the time compared to most approaches. Our code is available at: https://github.com/zhrli324/RLEdit.
>
---
#### [replaced 076] Automatic Prompt Optimization with Prompt Distillation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.18992v2](http://arxiv.org/pdf/2508.18992v2)**

> **作者:** Ernest A. Dyagin; Nikita I. Kulin; Artur R. Khairullin; Viktor N. Zhuravlev; Alena N. Sitkina
>
> **摘要:** Autoprompting is the process of automatically selecting optimized prompts for language models, which is gaining popularity due to the rapid development of prompt engineering driven by extensive research in the field of large language models (LLMs). This paper presents DistillPrompt -- a novel autoprompting method based on large language models that employs a multi-stage integration of task-specific information into prompts using training data. DistillPrompt utilizes distillation, compression, and aggregation operations to explore the prompt space more thoroughly. The method was tested on different datasets for text classification and generation tasks using the t-lite-instruct-0.1 language model. The results demonstrate a significant average improvement (e.g., 20.12% across the entire dataset compared to Grips) in key metrics over existing methods in the field, establishing DistillPrompt as one of the most effective non-gradient approaches in autoprompting.
>
---
#### [replaced 077] ETF: An Entity Tracing Framework for Hallucination Detection in Code Summaries
- **分类: cs.SE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.14748v4](http://arxiv.org/pdf/2410.14748v4)**

> **作者:** Kishan Maharaj; Vitobha Munigala; Srikanth G. Tamilselvam; Prince Kumar; Sayandeep Sen; Palani Kodeswaran; Abhijit Mishra; Pushpak Bhattacharyya
>
> **备注:** Accepted in ACL 2025 Main, 14 pages, 3 Figures, 5 Tables
>
> **摘要:** Recent advancements in large language models (LLMs) have significantly enhanced their ability to understand both natural language and code, driving their use in tasks like natural language-to-code (NL2Code) and code summarisation. However, LLMs are prone to hallucination, outputs that stray from intended meanings. Detecting hallucinations in code summarisation is especially difficult due to the complex interplay between programming and natural languages. We introduce a first-of-its-kind dataset, CodeSumEval, with ~10K samples, curated specifically for hallucination detection in code summarisation. We further propose a novel Entity Tracing Framework (ETF) that a) utilises static program analysis to identify code entities from the program and b) uses LLMs to map and verify these entities and their intents within generated code summaries. Our experimental analysis demonstrates the framework's effectiveness, leading to a 73% F1 score. The proposed approach provides a method for detecting hallucinations by tracing entities from the summary to the code, allowing us to evaluate summary accuracy and localise the error within the summary.
>
---
#### [replaced 078] Self-Alignment: Improving Alignment of Cultural Values in LLMs via In-Context Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.16482v2](http://arxiv.org/pdf/2408.16482v2)**

> **作者:** Rochelle Choenni; Ekaterina Shutova
>
> **摘要:** Improving the alignment of Large Language Models (LLMs) with respect to the cultural values that they encode has become an increasingly important topic. In this work, we study whether we can exploit existing knowledge about cultural values at inference time to adjust model responses to cultural value probes. We present a simple and inexpensive method that uses a combination of in-context learning (ICL) and human survey data, and show that we can improve the alignment to cultural values across 5 models that include both English-centric and multilingual LLMs. Importantly, we show that our method could prove useful in test languages other than English and can improve alignment to the cultural values that correspond to a range of culturally diverse countries.
>
---
#### [replaced 079] KG-CQR: Leveraging Structured Relation Representations in Knowledge Graphs for Contextual Query Retrieval
- **分类: cs.CL; cs.DB**

- **链接: [http://arxiv.org/pdf/2508.20417v3](http://arxiv.org/pdf/2508.20417v3)**

> **作者:** Chi Minh Bui; Ngoc Mai Thieu; Van Vinh Nguyen; Jason J. Jung; Khac-Hoai Nam Bui
>
> **备注:** Accepted at Main EMNLP 2025
>
> **摘要:** The integration of knowledge graphs (KGs) with large language models (LLMs) offers significant potential to improve the retrieval phase of retrieval-augmented generation (RAG) systems. In this study, we propose KG-CQR, a novel framework for Contextual Query Retrieval (CQR) that enhances the retrieval phase by enriching the contextual representation of complex input queries using a corpus-centric KG. Unlike existing methods that primarily address corpus-level context loss, KG-CQR focuses on query enrichment through structured relation representations, extracting and completing relevant KG subgraphs to generate semantically rich query contexts. Comprising subgraph extraction, completion, and contextual generation modules, KG-CQR operates as a model-agnostic pipeline, ensuring scalability across LLMs of varying sizes without additional training. Experimental results on RAGBench and MultiHop-RAG datasets demonstrate KG-CQR's superior performance, achieving a 4-6% improvement in mAP and a 2-3% improvement in Recall@25 over strong baseline models. Furthermore, evaluations on challenging RAG tasks such as multi-hop question answering show that, by incorporating KG-CQR, the performance consistently outperforms the existing baseline in terms of retrieval effectiveness
>
---
#### [replaced 080] Pierce the Mists, Greet the Sky: Decipher Knowledge Overshadowing via Knowledge Circuit Analysis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14406v3](http://arxiv.org/pdf/2505.14406v3)**

> **作者:** Haoming Huang; Yibo Yan; Jiahao Huo; Xin Zou; Xinfeng Li; Kun Wang; Xuming Hu
>
> **备注:** Accepted by 2025 EMNLP Main
>
> **摘要:** Large Language Models (LLMs), despite their remarkable capabilities, are hampered by hallucinations. A particularly challenging variant, knowledge overshadowing, occurs when one piece of activated knowledge inadvertently masks another relevant piece, leading to erroneous outputs even with high-quality training data. Current understanding of overshadowing is largely confined to inference-time observations, lacking deep insights into its origins and internal mechanisms during model training. Therefore, we introduce PhantomCircuit, a novel framework designed to comprehensively analyze and detect knowledge overshadowing. By innovatively employing knowledge circuit analysis, PhantomCircuit dissects the function of key components in the circuit and how the attention pattern dynamics contribute to the overshadowing phenomenon and its evolution throughout the training process. Extensive experiments demonstrate PhantomCircuit's effectiveness in identifying such instances, offering novel insights into this elusive hallucination and providing the research community with a new methodological lens for its potential mitigation.
>
---
#### [replaced 081] InterFeat: A Pipeline for Finding Interesting Scientific Features
- **分类: q-bio.QM; cs.AI; cs.CL; cs.IR; 68T05, 68T50, 92C50; I.2.6; I.2.7; H.2.8; J.3**

- **链接: [http://arxiv.org/pdf/2505.13534v2](http://arxiv.org/pdf/2505.13534v2)**

> **作者:** Dan Ofer; Michal Linial; Dafna Shahaf
>
> **摘要:** Finding interesting phenomena is the core of scientific discovery, but it is a manual, ill-defined concept. We present an integrative pipeline for automating the discovery of interesting simple hypotheses (feature-target relations with effect direction and a potential underlying mechanism) in structured biomedical data. The pipeline combines machine learning, knowledge graphs, literature search and Large Language Models. We formalize "interestingness" as a combination of novelty, utility and plausibility. On 8 major diseases from the UK Biobank, our pipeline consistently recovers risk factors years before their appearance in the literature. 40--53% of our top candidates were validated as interesting, compared to 0--7% for a SHAP-based baseline. Overall, 28% of 109 candidates were interesting to medical experts. The pipeline addresses the challenge of operationalizing "interestingness" scalably and for any target. We release data and code: https://github.com/LinialLab/InterFeat
>
---
#### [replaced 082] Towards No-Code Programming of Cobots: Experiments with Code Synthesis by Large Code Models for Conversational Programming
- **分类: cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2409.11041v4](http://arxiv.org/pdf/2409.11041v4)**

> **作者:** Chalamalasetti Kranti; Sherzod Hakimov; David Schlangen
>
> **备注:** Accepted to ITL4HRI workshop at RO-MAN 2025 conference
>
> **摘要:** While there has been a lot of research recently on robots in household environments, at the present time, most robots in existence can be found on shop floors, and most interactions between humans and robots happen there. ``Collaborative robots'' (cobots) designed to work alongside humans on assembly lines traditionally require expert programming, limiting ability to make changes, or manual guidance, limiting expressivity of the resulting programs. To address these limitations, we explore using Large Language Models (LLMs), and in particular, their abilities of doing in-context learning, for conversational code generation. As a first step, we define RATS, the ``Repetitive Assembly Task'', a 2D building task designed to lay the foundation for simulating industry assembly scenarios. In this task, a `programmer' instructs a cobot, using natural language, on how a certain assembly is to be built; that is, the programmer induces a program, through natural language. We create a dataset that pairs target structures with various example instructions (human-authored, template-based, and model-generated) and example code. With this, we systematically evaluate the capabilities of state-of-the-art LLMs for synthesising this kind of code, given in-context examples. Evaluating in a simulated environment, we find that LLMs are capable of generating accurate `first order code' (instruction sequences), but have problems producing `higher-order code' (abstractions such as functions, or use of loops).
>
---
#### [replaced 083] MultiPL-MoE: Multi-Programming-Lingual Extension of Large Language Models through Hybrid Mixture-of-Experts
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.19268v2](http://arxiv.org/pdf/2508.19268v2)**

> **作者:** Qing Wang; Xue Han; Jiahui Wang; Lehao Xing; Qian Hu; Lianlian Zhang; Chao Deng; Junlan Feng
>
> **摘要:** Despite LLMs' excellent code creation capabilities, multilingual code generation remains extremely challenging. To address this, we intent to improve the multi-programming-lingual (MultiPL) performance of the base LLMs while retaining the most popular ones using restricted computational resources. We consider MultiPL to be a special case of multiple natural languages and propose a MultiPL extension of LLMs utilizing a hybrid mixture of experts (MoE), called MultiPL-MoE. Specifically, MultiPL-MoE combines two paired MoEs to optimize expert selection at both the token and segment levels. The token-level MoE is a standard upcycling MoE structure with a shared expert and a novel gate weight normalization approach that aids in the final fusion with the segment-level MoE. The segment-level MoE incorporates two innovative designs to better capture the syntactic structure and contextual patterns of programming languages: First, using a sliding window to partition the input token sequence into multiple segments; Then, adopting an expert-choice routing strategy that allows experts to select the top-k segments. The results of the experiment proved the effectiveness of MultiPL-MoE.
>
---
#### [replaced 084] AI Sees Your Location, But With A Bias Toward The Wealthy World
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11163v3](http://arxiv.org/pdf/2502.11163v3)**

> **作者:** Jingyuan Huang; Jen-tse Huang; Ziyi Liu; Xiaoyuan Liu; Wenxuan Wang; Jieyu Zhao
>
> **备注:** Accepted to EMNLP 2025 (Main)
>
> **摘要:** Visual-Language Models (VLMs) have shown remarkable performance across various tasks, particularly in recognizing geographic information from images. However, VLMs still show regional biases in this task. To systematically evaluate these issues, we introduce a benchmark consisting of 1,200 images paired with detailed geographic metadata. Evaluating four VLMs, we find that while these models demonstrate the ability to recognize geographic information from images, achieving up to 53.8% accuracy in city prediction, they exhibit significant biases. Specifically, performance is substantially higher for economically developed and densely populated regions compared to less developed (-12.5%) and sparsely populated (-17.0%) areas. Moreover, regional biases of frequently over-predicting certain locations remain. For instance, they consistently predict Sydney for images taken in Australia, shown by the low entropy scores for these countries. The strong performance of VLMs also raises privacy concerns, particularly for users who share images online without the intent of being identified. Our code and dataset are publicly available at https://github.com/uscnlp-lime/FairLocator.
>
---
#### [replaced 085] Assessing and Mitigating Medical Knowledge Drift and Conflicts in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.07968v3](http://arxiv.org/pdf/2505.07968v3)**

> **作者:** Weiyi Wu; Xinwen Xu; Chongyang Gao; Xingjian Diao; Siting Li; Lucas A. Salas; Jiang Gui
>
> **摘要:** Large Language Models (LLMs) have great potential in the field of health care, yet they face great challenges in adapting to rapidly evolving medical knowledge. This can lead to outdated or contradictory treatment suggestions. This study investigated how LLMs respond to evolving clinical guidelines, focusing on concept drift and internal inconsistencies. We developed the DriftMedQA benchmark to simulate guideline evolution and assessed the temporal reliability of various LLMs. Our evaluation of seven state-of-the-art models across 4,290 scenarios demonstrated difficulties in rejecting outdated recommendations and frequently endorsing conflicting guidance. Additionally, we explored two mitigation strategies: Retrieval-Augmented Generation and preference fine-tuning via Direct Preference Optimization. While each method improved model performance, their combination led to the most consistent and reliable results. These findings underscore the need to improve LLM robustness to temporal shifts to ensure more dependable applications in clinical practice. The dataset is available at https://huggingface.co/datasets/RDBH/DriftMed.
>
---
#### [replaced 086] Linearly Controlled Language Generation with Performative Guarantees
- **分类: cs.CL; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2405.15454v2](http://arxiv.org/pdf/2405.15454v2)**

> **作者:** Emily Cheng; Carmen Amo Alonso
>
> **备注:** Under review
>
> **摘要:** The increasing prevalence of Large Language Models (LMs) in critical applications highlights the need for controlled language generation strategies that are not only computationally efficient but that also enjoy performance guarantees. To achieve this, we use a common model of concept semantics as linearly represented in an LM's latent space. In particular, we take the view that natural language generation traces a trajectory in this continuous semantic space, realized by the language model's hidden activations. This view permits a control-theoretic treatment of text generation in latent space, in which we propose a lightweight, gradient-free intervention that dynamically steers trajectories away from regions corresponding to undesired meanings. In particular, we propose to directly intervene the activations of the token that is being generated in embedding space in an online fashion. Crucially, we do not simply steer activations towards a desirable region. Instead, our method relies on classical techniques from control theory to precisely control activations in a context-dependent way, and guarantees that they are brought into a specific pre-defined region of embedding space that corresponds to allowed semantics. Our intervention is computed in closed-form according to an optimal controller formulation, minimally impacting generation time. This control of the activations in embedding space allows for fine-grained steering of attributes of the generated sequence. We demonstrate the effectiveness of our approach on different objectives-- toxicity avoidance and sentiment control-- while maintaining text quality.
>
---
#### [replaced 087] Dynamically Adaptive Reasoning via LLM-Guided MCTS for Efficient and Context-Aware KGQA
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.00719v3](http://arxiv.org/pdf/2508.00719v3)**

> **作者:** Yingxu Wang; Shiqi Fan; Mengzhu Wang; Siyang Gao; Siwei Liu; Nan Yin
>
> **摘要:** Knowledge Graph Question Answering (KGQA) aims to interpret natural language queries and perform structured reasoning over knowledge graphs by leveraging their relational and semantic structures to retrieve accurate answers. Recent KGQA methods primarily follow either retrieve-then-reason paradigm, relying on GNNs or heuristic rules for static paths extraction, or dynamic path generation strategies that use large language models (LLMs) with prompting to jointly perform retrieval and reasoning. However, the former suffers from limited adaptability due to static path extraction and lack of contextual refinement, while the latter incurs high computational costs and struggles with accurate path evaluation due to reliance on fixed scoring functions and extensive LLM calls. To address these issues, this paper proposes Dynamically Adaptive MCTS-based Reasoning (DAMR), a novel framework that integrates symbolic search with adaptive path evaluation for efficient and context-aware KGQA. DAMR employs a Monte Carlo Tree Search (MCTS) backbone guided by an LLM-based planner, which selects top-$k$ relevant relations at each step to reduce search space. To improve path evaluation accuracy, we introduce a lightweight Transformer-based scorer that performs context-aware plausibility estimation by jointly encoding the question and relation sequence through cross-attention, enabling the model to capture fine-grained semantic shifts during multi-hop reasoning. Furthermore, to alleviate the scarcity of high-quality supervision, DAMR incorporates a dynamic pseudo-path refinement mechanism that periodically generates training signals from partial paths explored during search, allowing the scorer to continuously adapt to the evolving distribution of reasoning trajectories. Extensive experiments on multiple KGQA benchmarks show that DAMR significantly outperforms state-of-the-art methods.
>
---
#### [replaced 088] The Good, the Bad and the Constructive: Automatically Measuring Peer Review's Utility for Authors
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2509.04484v2](http://arxiv.org/pdf/2509.04484v2)**

> **作者:** Abdelrahman Sadallah; Tim Baumgärtner; Iryna Gurevych; Ted Briscoe
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Providing constructive feedback to paper authors is a core component of peer review. With reviewers increasingly having less time to perform reviews, automated support systems are required to ensure high reviewing quality, thus making the feedback in reviews useful for authors. To this end, we identify four key aspects of review comments (individual points in weakness sections of reviews) that drive the utility for authors: Actionability, Grounding & Specificity, Verifiability, and Helpfulness. To enable evaluation and development of models assessing review comments, we introduce the RevUtil dataset. We collect 1,430 human-labeled review comments and scale our data with 10k synthetically labeled comments for training purposes. The synthetic data additionally contains rationales, i.e., explanations for the aspect score of a review comment. Employing the RevUtil dataset, we benchmark fine-tuned models for assessing review comments on these aspects and generating rationales. Our experiments demonstrate that these fine-tuned models achieve agreement levels with humans comparable to, and in some cases exceeding, those of powerful closed models like GPT-4o. Our analysis further reveals that machine-generated reviews generally underperform human reviews on our four aspects.
>
---
#### [replaced 089] AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.12226v5](http://arxiv.org/pdf/2402.12226v5)**

> **作者:** Jun Zhan; Junqi Dai; Jiasheng Ye; Yunhua Zhou; Dong Zhang; Zhigeng Liu; Xin Zhang; Ruibin Yuan; Ge Zhang; Linyang Li; Hang Yan; Jie Fu; Tao Gui; Tianxiang Sun; Yu-Gang Jiang; Xipeng Qiu
>
> **备注:** 28 pages, 16 figures, under review, work in progress
>
> **摘要:** We introduce AnyGPT, an any-to-any multimodal language model that utilizes discrete representations for the unified processing of various modalities, including speech, text, images, and music. AnyGPT can be trained stably without any alterations to the current large language model (LLM) architecture or training paradigms. Instead, it relies exclusively on data-level preprocessing, facilitating the seamless integration of new modalities into LLMs, akin to the incorporation of new languages. We build a multimodal text-centric dataset for multimodal alignment pre-training. Utilizing generative models, we synthesize the first large-scale any-to-any multimodal instruction dataset. It consists of 108k samples of multi-turn conversations that intricately interweave various modalities, thus equipping the model to handle arbitrary combinations of multimodal inputs and outputs. Experimental results demonstrate that AnyGPT is capable of facilitating any-to-any multimodal conversation while achieving performance comparable to specialized models across all modalities, proving that discrete representations can effectively and conveniently unify multiple modalities within a language model. Demos are shown in https://junzhan2000.github.io/AnyGPT.github.io/
>
---
#### [replaced 090] Efficient Large Language Models with Zero-Shot Adjustable Acceleration
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.01190v2](http://arxiv.org/pdf/2509.01190v2)**

> **作者:** Sajjad Kachuee; Mohammad Sharifkhani
>
> **摘要:** Using Large Language Models (LLMs) in real-world applications presents significant challenges, particularly in balancing computational efficiency with model performance. Optimizing acceleration after fine-tuning and during inference is critical for building efficient architectures. This paper introduces Zero-Shot Adjustable Acceleration, a novel training and inference method that dynamically adjusts hardware utilization during inference without requiring additional fine-tuning. The proposed approach is applied to recent LLMs and evaluated across multiple classification and text generation tasks. Experimental results demonstrate that the method supports a wide range of zero-shot acceleration and achieves up to 11x speedup compared to the baseline.
>
---
#### [replaced 091] DCPO: Dynamic Clipping Policy Optimization
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.02333v2](http://arxiv.org/pdf/2509.02333v2)**

> **作者:** Shihui Yang; Chengfeng Dou; Peidong Guo; Kai Lu; Qiang Ju; Fei Deng; Rihui Xin
>
> **摘要:** Reinforcement Learning from Verifiable Rewards (RLVR) has emerged as a promising framework for enhancing the reasoning capabilities of large language models. However, existing approaches such as GRPO often suffer from zero gradients. This problem arises primarily due to fixed clipping bounds for token-level probability ratios and the standardization of identical rewards, which can lead to ineffective gradient updates and underutilization of generated responses. In this work, we propose Dynamic Clipping Policy Optimization(DCPO), which introduces a dynamic clipping strategy that adaptively adjusts clipping bounds based on token-specific prior probabilities to enhance token-level exploration, and a smooth advantage standardization technique that standardizes rewards across cumulative training steps to improve the response-level effective utilization of generated responses. DCPO achieved state-of-the-art performance on four benchmarks based on four different models. In particular, DCPO achieved an Avg@1 of 46.7 under greedy decoding and an Avg@32 of 38.8 under 32 times sampling on the AIME24 benchmark, surpassing DAPO (36.7/31.6), GRPO (36.7/32.1) and GSPO (40.0/34.9) on the Qwen2.5-Math-7B model. On the AIME25 benchmark based on Qwen2.5-14B, DCPO achieves a performance of (23.3/19.0), surpassing GRPO (13.3/10.5), DAPO (20.0/15.3) and GSPO (16.7/9.9). Furthermore, DCPO achieved an average 28% improvement in the nonzero advantage over GRPO in four models, doubled the training efficiency over DAPO, and significantly reduced the token clipping ratio by an order of magnitude compared to both GRPO and DAPO, while achieving superior performance. These results highlight DCPO's effectiveness in leveraging generated data more efficiently for reinforcement learning in large language models.
>
---
#### [replaced 092] A Minimum Description Length Approach to Regularization in Neural Networks
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13398v2](http://arxiv.org/pdf/2505.13398v2)**

> **作者:** Matan Abudy; Orr Well; Emmanuel Chemla; Roni Katzir; Nur Lan
>
> **备注:** 9 pages
>
> **摘要:** State-of-the-art neural networks can be trained to become remarkable solutions to many problems. But while these architectures can express symbolic, perfect solutions, trained models often arrive at approximations instead. We show that the choice of regularization method plays a crucial role: when trained on formal languages with standard regularization ($L_1$, $L_2$, or none), expressive architectures not only fail to converge to correct solutions but are actively pushed away from perfect initializations. In contrast, applying the Minimum Description Length (MDL) principle to balance model complexity with data fit provides a theoretically grounded regularization method. Using MDL, perfect solutions are selected over approximations, independently of the optimization algorithm. We propose that unlike existing regularization techniques, MDL introduces the appropriate inductive bias to effectively counteract overfitting and promote generalization.
>
---
#### [replaced 093] Membership Inference Attacks on LLM-based Recommender Systems
- **分类: cs.IR; cs.AI; cs.CL; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.18665v2](http://arxiv.org/pdf/2508.18665v2)**

> **作者:** Jiajie He; Yuechun Gu; Min-Chun Chen; Keke Chen
>
> **摘要:** Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots.
>
---
#### [replaced 094] MedualTime: A Dual-Adapter Language Model for Medical Time Series-Text Multimodal Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.06620v4](http://arxiv.org/pdf/2406.06620v4)**

> **作者:** Jiexia Ye; Weiqi Zhang; Ziyue Li; Jia Li; Meng Zhao; Fugee Tsung
>
> **备注:** 9 pages, 6 figure, 3 tables
>
> **摘要:** The recent rapid advancements in language models (LMs) have garnered attention in medical time series-text multimodal learning. However, existing contrastive learning-based and prompt-based LM approaches tend to be biased, often assigning a primary role to time series modality while treating text modality as secondary. We classify these approaches under a temporal-primary paradigm, which may overlook the unique and critical task-relevant information embedded in text modality like clinical reports, thus failing to fully leverage mutual benefits and complementarity of different modalities. To fill this gap, we propose a novel textual-temporal multimodal learning paradigm that enables either modality to serve as the primary while being enhanced by the other, thereby effectively capturing modality-specific information and fostering cross-modal interaction. In specific, we design MedualTime, a language model composed of dual adapters to implement temporal-primary and textual-primary modeling simultaneously. Within each adapter, lightweight adaptation tokens are injected into the top layers of LM to encourage high-level modality fusion. The shared LM pipeline by dual adapters not only achieves adapter alignment but also enables efficient fine-tuning, reducing computational resources. Empirically, MedualTime demonstrates superior performance on medical data, achieving notable improvements of 8% accuracy and 12% F1 in supervised settings. Furthermore, MedualTime's transferability is validated by few-shot label transfer experiments from coarse-grained to fine-grained medical data. https://github.com/start2020/MedualTime
>
---
#### [replaced 095] A Survey on Training-free Alignment of Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.09016v3](http://arxiv.org/pdf/2508.09016v3)**

> **作者:** Birong Pan; Yongqi Li; Weiyu Zhang; Wenpeng Lu; Mayi Xu; Shen Zhou; Yuanyuan Zhu; Ming Zhong; Tieyun Qian
>
> **备注:** Accepted to EMNLP 2025 (findings), camera-ready version
>
> **摘要:** The alignment of large language models (LLMs) aims to ensure their outputs adhere to human values, ethical standards, and legal norms. Traditional alignment methods often rely on resource-intensive fine-tuning (FT), which may suffer from knowledge degradation and face challenges in scenarios where the model accessibility or computational resources are constrained. In contrast, training-free (TF) alignment techniques--leveraging in-context learning, decoding-time adjustments, and post-generation corrections--offer a promising alternative by enabling alignment without heavily retraining LLMs, making them adaptable to both open-source and closed-source environments. This paper presents the first systematic review of TF alignment methods, categorizing them by stages of pre-decoding, in-decoding, and post-decoding. For each stage, we provide a detailed examination from the viewpoint of LLMs and multimodal LLMs (MLLMs), highlighting their mechanisms and limitations. Furthermore, we identify key challenges and future directions, paving the way for more inclusive and effective TF alignment techniques. By synthesizing and organizing the rapidly growing body of research, this survey offers a guidance for practitioners and advances the development of safer and more reliable LLMs.
>
---
#### [replaced 096] ChatCFD: An LLM-Driven Agent for End-to-End CFD Automation with Domain-Specific Structured Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.02019v2](http://arxiv.org/pdf/2506.02019v2)**

> **作者:** E Fan; Kang Hu; Zhuowen Wu; Jiangyang Ge; Jiawei Miao; Yuzhi Zhang; He Sun; Weizong Wang; Tianhan Zhang
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** Computational Fluid Dynamics (CFD) is essential for advancing scientific and engineering fields but is hindered by operational complexity, high expertise requirements, and limited accessibility. This paper introduces ChatCFD, an automated agent system for OpenFOAM simulations that processes multi-modal inputs (e.g., research papers, meshes) via an interactive interface, leveraging DeepSeek-R1 and DeepSeek-V3 large language models, a multi-agent architecture, and OpenFOAM knowledge. Its four-stage pipeline (Knowledge Base Construction, User Input Processing, Case File Generation, and Execution and Error Reflection) enables iterative trial-reflection-refinement for intricate setups, supporting diverse physical models and external meshes. Validation on 205 benchmark tutorial cases, 110 perturbed variants, and 2 literature-derived cases shows ChatCFD's 82.1 percent operational success rate on basic cases, outperforming MetaOpenFOAM (6.2 percent) and Foam-Agent (42.3 percent), and 60-80 percent on literature-derived complex cases. Turbulence model studies show a 40 percent success rate for common models versus 10 percent for rare ones like RNG k-epsilon. Physics coupling analyses reveal higher resource demands for multi-physics-coupled cases, while LLM bias toward simpler setups introduces persistent errors, such as dimensional inconsistency. Ablation studies highlight the efficacy of RAG-based modules and reflection mechanisms. By automating hypothesis testing and parameter exploration, ChatCFD accelerates scientific discovery in fluid mechanics and engineering, addressing LLM limitations through structured design and showing strong potential as a modular component in MCP-based agent networks for collaborative multi-agent systems, paving the way for scalable AI-driven CFD innovation. The code for ChatCFD is available at https://github.com/ConMoo/ChatCFD.
>
---
#### [replaced 097] LinkAlign: Scalable Schema Linking for Real-World Large-Scale Multi-Database Text-to-SQL
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.18596v4](http://arxiv.org/pdf/2503.18596v4)**

> **作者:** Yihan Wang; Peiyu Liu; Xin Yang
>
> **摘要:** Schema linking is a critical bottleneck in applying existing Text-to-SQL models to real-world, large-scale, multi-database environments. Through error analysis, we identify two major challenges in schema linking: (1) Database Retrieval: accurately selecting the target database from a large schema pool, while effectively filtering out irrelevant ones; and (2) Schema Item Grounding: precisely identifying the relevant tables and columns within complex and often redundant schemas for SQL generation. Based on these, we introduce LinkAlign, a novel framework tailored for large-scale databases with thousands of fields. LinkAlign comprises three key steps: multi-round semantic enhanced retrieval and irrelevant information isolation for Challenge 1, and schema extraction enhancement for Challenge 2. Each stage supports both Agent and Pipeline execution modes, enabling balancing efficiency and performance via modular design. To enable more realistic evaluation, we construct AmbiDB, a synthetic dataset designed to reflect the ambiguity of real-world schema linking. Experiments on widely-used Text-to-SQL benchmarks demonstrate that LinkAlign consistently outperforms existing baselines on all schema linking metrics. Notably, it improves the overall Text-to-SQL pipeline and achieves a new state-of-the-art score of 33.09% on the Spider 2.0-Lite benchmark using only open-source LLMs, ranking first on the leaderboard at the time of submission. The codes are available at https://github.com/Satissss/LinkAlign
>
---
#### [replaced 098] Support or Refute: Analyzing the Stance of Evidence to Detect Out-of-Context Mis- and Disinformation
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2311.01766v5](http://arxiv.org/pdf/2311.01766v5)**

> **作者:** Xin Yuan; Jie Guo; Weidong Qiu; Zheng Huang; Shujun Li
>
> **备注:** Accepted and published by EMNLP 2023. Details can be found in https://aclanthology.org/2023.emnlp-main.259
>
> **摘要:** Mis- and disinformation online have become a major societal problem as major sources of online harms of different kinds. One common form of mis- and disinformation is out-of-context (OOC) information, where different pieces of information are falsely associated, e.g., a real image combined with a false textual caption or a misleading textual description. Although some past studies have attempted to defend against OOC mis- and disinformation through external evidence, they tend to disregard the role of different pieces of evidence with different stances. Motivated by the intuition that the stance of evidence represents a bias towards different detection results, we propose a stance extraction network (SEN) that can extract the stances of different pieces of multi-modal evidence in a unified framework. Moreover, we introduce a support-refutation score calculated based on the co-occurrence relations of named entities into the textual SEN. Extensive experiments on a public large-scale dataset demonstrated that our proposed method outperformed the state-of-the-art baselines, with the best model achieving a performance gain of 3.2% in accuracy. The source code and checkpoints are publicly available at https://github.com/yx3266/SEN.
>
---
#### [replaced 099] TreeReview: A Dynamic Tree of Questions Framework for Deep and Efficient LLM-based Scientific Peer Review
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07642v2](http://arxiv.org/pdf/2506.07642v2)**

> **作者:** Yuan Chang; Ziyue Li; Hengyuan Zhang; Yuanbo Kong; Yanru Wu; Hayden Kwok-Hay So; Zhijiang Guo; Liya Zhu; Ngai Wong
>
> **备注:** Accepted to EMNLP2025 Main
>
> **摘要:** While Large Language Models (LLMs) have shown significant potential in assisting peer review, current methods often struggle to generate thorough and insightful reviews while maintaining efficiency. In this paper, we propose TreeReview, a novel framework that models paper review as a hierarchical and bidirectional question-answering process. TreeReview first constructs a tree of review questions by recursively decomposing high-level questions into fine-grained sub-questions and then resolves the question tree by iteratively aggregating answers from leaf to root to get the final review. Crucially, we incorporate a dynamic question expansion mechanism to enable deeper probing by generating follow-up questions when needed. We construct a benchmark derived from ICLR and NeurIPS venues to evaluate our method on full review generation and actionable feedback comments generation tasks. Experimental results of both LLM-based and human evaluation show that TreeReview outperforms strong baselines in providing comprehensive, in-depth, and expert-aligned review feedback, while reducing LLM token usage by up to 80% compared to computationally intensive approaches. Our code and benchmark dataset are available at https://github.com/YuanChang98/tree-review.
>
---
#### [replaced 100] Fine-Tuning Large Language Models for Scientific Text Classification: A Comparative Study
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.00098v2](http://arxiv.org/pdf/2412.00098v2)**

> **作者:** Zhyar Rzgar K Rostam; Gábor Kertész
>
> **备注:** 6 pages, 3 figures, 7 tables
>
> **摘要:** The exponential growth of online textual content across diverse domains has necessitated advanced methods for automated text classification. Large Language Models (LLMs) based on transformer architectures have shown significant success in this area, particularly in natural language processing (NLP) tasks. However, general-purpose LLMs often struggle with domain-specific content, such as scientific texts, due to unique challenges like specialized vocabulary and imbalanced data. In this study, we fine-tune four state-of-the-art LLMs BERT, SciBERT, BioBERT, and BlueBERT on three datasets derived from the WoS-46985 dataset to evaluate their performance in scientific text classification. Our experiments reveal that domain-specific models, particularly SciBERT, consistently outperform general-purpose models in both abstract-based and keyword-based classification tasks. Additionally, we compare our achieved results with those reported in the literature for deep learning models, further highlighting the advantages of LLMs, especially when utilized in specific domains. The findings emphasize the importance of domain-specific adaptations for LLMs to enhance their effectiveness in specialized text classification tasks.
>
---
#### [replaced 101] Grammaticality illusion or ambiguous interpretation? Event-related potentials reveal the nature of the missing-NP effect in Mandarin centre-embedded structures
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2402.11282v2](http://arxiv.org/pdf/2402.11282v2)**

> **作者:** Qihang Yang; Caimei Yang; Yu Liao; Ziman Zhuang
>
> **摘要:** In several languages, omitting a verb phrase (VP) in double centre-embedded structures creates a grammaticality illusion. Similar illusion also exhibited in Mandarin missing-NP double centre-embedded structures. However, there is no consensus on its very nature. Instead of treating it as grammaticality illusion, we argue that ambiguous interpretations of verbs can best account for this phenomenon in Mandarin. To further support this hypothesis, we conducted two electroencephalography (EEG) experiments on quasi double centre-embedded structures whose complexity is reduced by placing the self-embedding relative clauses into the sentence's subject position. Experiment 1 showed that similar phenomenon even exhibited in this structure, evidenced by an absence of P600 effect and a presence of N400 effect. In Experiment 2, providing semantic cues to reduce ambiguity dispelled this illusion, as evidenced by a P600 effect. We interpret the results under garden-path theory and propose that word-order difference may account for this cross-linguistic variation.
>
---
#### [replaced 102] ChinaTravel: An Open-Ended Benchmark for Language Agents in Chinese Travel Planning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.13682v4](http://arxiv.org/pdf/2412.13682v4)**

> **作者:** Jie-Jing Shao; Bo-Wen Zhang; Xiao-Wen Yang; Baizhi Chen; Si-Yu Han; Wen-Da Wei; Guohao Cai; Zhenhua Dong; Lan-Zhe Guo; Yu-Feng Li
>
> **备注:** Webpage: https://www.lamda.nju.edu.cn/shaojj/chinatravel
>
> **摘要:** Recent advances in LLMs, particularly in language reasoning and tool integration, have rapidly sparked the \emph{Language Agents} for real-world development. Among these, travel planning represents a prominent domain, combining complex multi-objective planning challenges with practical deployment demands. However, existing benchmarks often oversimplify real-world requirements by focusing on synthetic queries and limited constraints. We address the gap of evaluating language agents in multi-day, multi-POI travel planning scenarios with diverse and open human needs. Specifically, we introduce \emph{ChinaTravel}, the first open-ended benchmark grounded in authentic Chinese travel requirements collected from 1,154 human participants. We design a compositionally generalizable domain-specific language (DSL) for scalable evaluation, covering feasibility, constraint satisfaction, and preference comparison. Empirical studies reveal the potential of neuro-symbolic agents in travel planning, achieving a 37.0\% constraint satisfaction rate on human queries, a 10\times improvement over purely neural models. These findings highlight ChinaTravel as a pivotal milestone for advancing language agents in complex, real-world planning scenarios.
>
---
#### [replaced 103] Project Riley: Multimodal Multi-Agent LLM Collaboration with Emotional Reasoning and Voting
- **分类: cs.AI; cs.CL; I.2.7; I.2.1; H.5.2**

- **链接: [http://arxiv.org/pdf/2505.20521v2](http://arxiv.org/pdf/2505.20521v2)**

> **作者:** Ana Rita Ortigoso; Gabriel Vieira; Daniel Fuentes; Luis Frazão; Nuno Costa; António Pereira
>
> **备注:** 28 pages, 5 figures. Submitted for review to Information Fusion
>
> **摘要:** This paper presents Project Riley, a novel multimodal and multi-model conversational AI architecture oriented towards the simulation of reasoning influenced by emotional states. Drawing inspiration from Pixar's Inside Out, the system comprises five distinct emotional agents - Joy, Sadness, Fear, Anger, and Disgust - that engage in structured multi-round dialogues to generate, criticise, and iteratively refine responses. A final reasoning mechanism synthesises the contributions of these agents into a coherent output that either reflects the dominant emotion or integrates multiple perspectives. The architecture incorporates both textual and visual large language models (LLMs), alongside advanced reasoning and self-refinement processes. A functional prototype was deployed locally in an offline environment, optimised for emotional expressiveness and computational efficiency. From this initial prototype, another one emerged, called Armando, which was developed for use in emergency contexts, delivering emotionally calibrated and factually accurate information through the integration of Retrieval-Augmented Generation (RAG) and cumulative context tracking. The Project Riley prototype was evaluated through user testing, in which participants interacted with the chatbot and completed a structured questionnaire assessing three dimensions: Emotional Appropriateness, Clarity and Utility, and Naturalness and Human-likeness. The results indicate strong performance in structured scenarios, particularly with respect to emotional alignment and communicative clarity.
>
---
#### [replaced 104] MovieCORE: COgnitive REasoning in Movies
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.19026v2](http://arxiv.org/pdf/2508.19026v2)**

> **作者:** Gueter Josmy Faure; Min-Hung Chen; Jia-Fong Yeh; Ying Cheng; Hung-Ting Su; Yung-Hao Tang; Shang-Hong Lai; Winston H. Hsu
>
> **备注:** Accepted for EMNLP'2025 Main Conference. Project Page: https://joslefaure.github.io/assets/html/moviecore.html
>
> **摘要:** This paper introduces MovieCORE, a novel video question answering (VQA) dataset designed to probe deeper cognitive understanding of movie content. Unlike existing datasets that focus on surface-level comprehension, MovieCORE emphasizes questions that engage System-2 thinking while remaining specific to the video material. We present an innovative agentic brainstorming approach, utilizing multiple large language models (LLMs) as thought agents to generate and refine high-quality question-answer pairs. To evaluate dataset quality, we develop a set of cognitive tests assessing depth, thought-provocation potential, and syntactic complexity. We also propose a comprehensive evaluation scheme for assessing VQA model performance on deeper cognitive tasks. To address the limitations of existing video-language models (VLMs), we introduce an agentic enhancement module, Agentic Choice Enhancement (ACE), which improves model reasoning capabilities post-training by up to 25%. Our work contributes to advancing movie understanding in AI systems and provides valuable insights into the capabilities and limitations of current VQA models when faced with more challenging, nuanced questions about cinematic content. Our project page, dataset and code can be found at https://joslefaure.github.io/assets/html/moviecore.html.
>
---
#### [replaced 105] Persona-driven Simulation of Voting Behavior in the European Parliament with Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.11798v2](http://arxiv.org/pdf/2506.11798v2)**

> **作者:** Maximilian Kreutner; Marlene Lutz; Markus Strohmaier
>
> **摘要:** Large Language Models (LLMs) display remarkable capabilities to understand or even produce political discourse, but have been found to consistently display a progressive left-leaning bias. At the same time, so-called persona or identity prompts have been shown to produce LLM behavior that aligns with socioeconomic groups that the base model is not aligned with. In this work, we analyze whether zero-shot persona prompting with limited information can accurately predict individual voting decisions and, by aggregation, accurately predict positions of European groups on a diverse set of policies. We evaluate if predictions are stable towards counterfactual arguments, different persona prompts and generation methods. Finally, we find that we can simulate voting behavior of Members of the European Parliament reasonably well with a weighted F1 score of approximately 0.793. Our persona dataset of politicians in the 2024 European Parliament and our code are available at https://github.com/dess-mannheim/european_parliament_simulation.
>
---
#### [replaced 106] Multiple Noises in Diffusion Model for Semi-Supervised Multi-Domain Translation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2309.14394v2](http://arxiv.org/pdf/2309.14394v2)**

> **作者:** Tsiry Mayet; Simon Bernard; Romain Herault; Clement Chatelain
>
> **摘要:** In this work, we address the challenge of multi-domain translation, where the objective is to learn mappings between arbitrary configurations of domains within a defined set (such as $(D_1, D_2)\rightarrow{}D_3$, $D_2\rightarrow{}(D_1, D_3)$, $D_3\rightarrow{}D_1$, etc. for three domains) without the need for separate models for each specific translation configuration, enabling more efficient and flexible domain translation. We introduce Multi-Domain Diffusion (MDD), a method with dual purposes: i) reconstructing any missing views for new data objects, and ii) enabling learning in semi-supervised contexts with arbitrary supervision configurations. MDD achieves these objectives by exploiting the noise formulation of diffusion models, specifically modeling one noise level per domain. Similar to existing domain translation approaches, MDD learns the translation between any combination of domains. However, unlike prior work, our formulation inherently handles semi-supervised learning without modification by representing missing views as noise in the diffusion process. We evaluate our approach through domain translation experiments on BL3NDT, a multi-domain synthetic dataset designed for challenging semantic domain inversion, the BraTS2020 dataset, and the CelebAMask-HQ dataset.
>
---
