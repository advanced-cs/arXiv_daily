# 自然语言处理 cs.CL

- **最新发布 76 篇**

- **更新 58 篇**

## 最新发布

#### [new 001] Uncovering Cross-Objective Interference in Multi-Objective Alignment
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多目标对齐任务，解决LLM训练中部分目标性能提升而其他目标退化的问题。通过分析干扰机制，提出CTWA方法有效缓解此问题。**

- **链接: [https://arxiv.org/pdf/2602.06869v1](https://arxiv.org/pdf/2602.06869v1)**

> **作者:** Yining Lu; Meng Jiang
>
> **摘要:** We study a persistent failure mode in multi-objective alignment for large language models (LLMs): training improves performance on only a subset of objectives while causing others to degrade. We formalize this phenomenon as cross-objective interference and conduct the first systematic study across classic scalarization algorithms, showing that interference is pervasive and exhibits strong model dependence. To explain this phenomenon, we derive a local covariance law showing that an objective improves at first order when its reward exhibits positive covariance with the scalarized score. We extend this analysis to clipped surrogate objectives used in modern alignment, demonstrating that the covariance law remains valid under mild conditions despite clipping. Building on this analysis, we propose Covariance Targeted Weight Adaptation (CTWA), a plug-and-play method that maintains positive covariance between objective rewards and the training signal to effectively mitigate cross-objective interference. Finally, we complement these local improvement conditions with a global convergence analysis under the Polyak--Łojasiewicz condition, establishing when non-convex scalarized optimization achieves global convergence and how cross-objective interference depends on specific model geometric properties.
>
---
#### [new 002] Optimal Turkish Subword Strategies at Scale: Systematic Evaluation of Data, Vocabulary, Morphology Interplay
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的分词任务，旨在优化土耳其语的子词分割策略。通过系统评估数据、词汇和形态学的相互作用，提出一种全面的评估框架，解决分词效率与形态准确性问题。**

- **链接: [https://arxiv.org/pdf/2602.06942v1](https://arxiv.org/pdf/2602.06942v1)**

> **作者:** Duygu Altinok
>
> **备注:** Submitted to Cambridge NLP journal, all rights belong to them
>
> **摘要:** Tokenization is a pivotal design choice for neural language modeling in morphologically rich languages (MRLs) such as Turkish, where productive agglutination challenges both vocabulary efficiency and morphological fidelity. Prior studies have explored tokenizer families and vocabulary sizes but typically (i) vary vocabulary without systematically controlling the tokenizer's training corpus, (ii) provide limited intrinsic diagnostics, and (iii) evaluate a narrow slice of downstream tasks. We present the first comprehensive, principled study of Turkish subword tokenization; a "subwords manifest", that jointly varies vocabulary size and tokenizer training corpus size (data and vocabulary coupling), compares multiple tokenizer families under matched parameter budgets (WordPiece, morphology level, and character baselines), and evaluates across semantic (NLI, STS, sentiment analysis, NER), syntactic (POS, dependency parsing), and morphology-sensitive probes. To explain why tokenizers succeed or fail, we introduce a morphology-aware diagnostic toolkit that goes beyond coarse aggregates to boundary-level micro/macro F1, decoupled lemma atomicity vs. surface boundary hits, over/under-segmentation indices, character/word edit distances (CER/WER), continuation rates, and affix-type coverage and token-level atomicity. Our contributions are fourfold: (i) a systematic investigation of the vocabulary-corpus-success triad; (ii) a unified, morphology-aware evaluation framework linking intrinsic diagnostics to extrinsic outcomes; (iii) controlled comparisons identifying when character-level and morphology-level tokenization pay off; and (iv) an open-source release of evaluation code, tokenizer pipelines, and models. As the first work of its kind, this "subwords manifest" delivers actionable guidance for building effective tokenizers in MRLs and establishes a reproducible foundation for future research.
>
---
#### [new 003] Visual Word Sense Disambiguation with CLIP through Dual-Channel Text Prompting and Image Augmentations
- **分类: cs.CL**

- **简介: 该论文属于视觉词义消歧任务，旨在解决自然语言中词汇歧义问题。通过CLIP模型结合双通道文本提示和图像增强，提升词义消歧效果。**

- **链接: [https://arxiv.org/pdf/2602.06799v1](https://arxiv.org/pdf/2602.06799v1)**

> **作者:** Shamik Bhattacharya; Daniel Perkins; Yaren Dogan; Vineeth Konjeti; Sudarshan Srinivasan; Edmon Begoli
>
> **备注:** 9 pages, 6 figures, pending journal/workshop submission
>
> **摘要:** Ambiguity poses persistent challenges in natural language understanding for large language models (LLMs). To better understand how lexical ambiguity can be resolved through the visual domain, we develop an interpretable Visual Word Sense Disambiguation (VWSD) framework. The model leverages CLIP to project ambiguous language and candidate images into a shared multimodal space. We enrich textual embeddings using a dual-channel ensemble of semantic and photo-based prompts with WordNet synonyms, while image embeddings are refined through robust test-time augmentations. We then use cosine similarity to determine the image that best aligns with the ambiguous text. When evaluated on the SemEval-2023 VWSD dataset, enriching the embeddings raises the MRR from 0.7227 to 0.7590 and the Hit Rate from 0.5810 to 0.6220. Ablation studies reveal that dual-channel prompting provides strong, low-latency performance, whereas aggressive image augmentation yields only marginal gains. Additional experiments with WordNet definitions and multilingual prompt ensembles further suggest that noisy external signals tend to dilute semantic specificity, reinforcing the effectiveness of precise, CLIP-aligned prompts for visual word sense disambiguation.
>
---
#### [new 004] RoPE-LIME: RoPE-Space Locality + Sparse-K Sampling for Efficient LLM Attribution
- **分类: cs.CL**

- **简介: 该论文属于模型解释任务，解决封闭模型输出难以解释的问题。通过RoPE-LIME方法，利用开放模型计算token级归因，提升解释效率与质量。**

- **链接: [https://arxiv.org/pdf/2602.06275v1](https://arxiv.org/pdf/2602.06275v1)**

> **作者:** Isaac Picov; Ritesh Goru
>
> **摘要:** Explaining closed-source LLM outputs is challenging because API access prevents gradient-based attribution, while perturbation methods are costly and noisy when they depend on regenerated text. We introduce RoPE-LIME, an open-source extension of gSMILE that decouples reasoning from explanation: given a fixed output from a closed model, a smaller open-source surrogate computes token-level attributions from probability-based objectives (negative log-likelihood and divergence targets) under input perturbations. RoPE-LIME incorporates (i) a locality kernel based on Relaxed Word Mover's Distance computed in RoPE embedding space for stable similarity under masking, and (ii) Sparse-K sampling, an efficient perturbation strategy that improves interaction coverage under limited budgets. Experiments on HotpotQA (sentence features) and a hand-labeled MMLU subset (word features) show that RoPE-LIME produces more informative attributions than leave-one-out sampling and improves over gSMILE while substantially reducing closed-model API calls.
>
---
#### [new 005] BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks
- **分类: cs.CL**

- **简介: 该论文提出BenchMarker工具，用于检测多选题基准中的缺陷，解决MCQA基准质量不高的问题。通过教育标准评估，识别出污染、捷径和写作错误，并改进基准设计。**

- **链接: [https://arxiv.org/pdf/2602.06221v1](https://arxiv.org/pdf/2602.06221v1)**

> **作者:** Nishant Balepur; Bhavya Rajasekaran; Jane Oh; Michael Xie; Atrey Desai; Vipul Gupta; Steven James Moore; Eunsol Choi; Rachel Rudinger; Jordan Lee Boyd-Graber
>
> **备注:** In-progress preprint
>
> **摘要:** Multiple-choice question answering (MCQA) is standard in NLP, but benchmarks lack rigorous quality control. We present BenchMarker, an education-inspired toolkit using LLM judges to flag three common MCQ flaws: 1) contamination - items appearing exactly online; 2) shortcuts - cues in the choices that enable guessing; and 3) writing errors - structural/grammatical issues based on a 19-rule education rubric. We validate BenchMarker with human annotations, then run the tool to audit 12 benchmarks, revealing: 2) contaminated MCQs tend to inflate accuracy, while writing errors tend to lower it and change rankings beyond random; and 3) prior benchmark repairs address their targeted issues (i.e., lowering accuracy with LLM-written distractors), but inadvertently add new flaws (i.e. implausible distractors, many correct answers). Overall, flaws in MCQs degrade NLP evaluation, but education research offers a path forward. We release BenchMarker to bridge the fields and improve MCQA benchmark design.
>
---
#### [new 006] SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks
- **分类: cs.CL**

- **简介: 该论文提出SEMA，解决多轮越狱攻击问题，通过自调优和意图感知奖励提升攻击效果，实现高效、稳定且通用的对抗方法。**

- **链接: [https://arxiv.org/pdf/2602.06854v1](https://arxiv.org/pdf/2602.06854v1)**

> **作者:** Mingqian Feng; Xiaodong Liu; Weiwei Yang; Jialin Song; Xuekai Zhu; Chenliang Xu; Jianfeng Gao
>
> **备注:** ICLR 2026, 37 pages, 13 tables, 7 figures
>
> **摘要:** Multi-turn jailbreaks capture the real threat model for safety-aligned chatbots, where single-turn attacks are merely a special case. Yet existing approaches break under exploration complexity and intent drift. We propose SEMA, a simple yet effective framework that trains a multi-turn attacker without relying on any existing strategies or external data. SEMA comprises two stages. Prefilling self-tuning enables usable rollouts by fine-tuning on non-refusal, well-structured, multi-turn adversarial prompts that are self-generated with a minimal prefix, thereby stabilizing subsequent learning. Reinforcement learning with intent-drift-aware reward trains the attacker to elicit valid multi-turn adversarial prompts while maintaining the same harmful objective. We anchor harmful intent in multi-turn jailbreaks via an intent-drift-aware reward that combines intent alignment, compliance risk, and level of detail. Our open-loop attack regime avoids dependence on victim feedback, unifies single- and multi-turn settings, and reduces exploration complexity. Across multiple datasets, victim models, and jailbreak judges, our method achieves state-of-the-art (SOTA) attack success rates (ASR), outperforming all single-turn baselines, manually scripted and template-driven multi-turn baselines, as well as our SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization) variants. For instance, SEMA performs an average $80.1\%$ ASR@1 across three closed-source and open-source victim models on AdvBench, 33.9% over SOTA. The approach is compact, reproducible, and transfers across targets, providing a stronger and more realistic stress test for large language model (LLM) safety and enabling automatic redteaming to expose and localize failure modes. Our code is available at: https://github.com/fmmarkmq/SEMA.
>
---
#### [new 007] Inference-Time Rethinking with Latent Thought Vectors for Math Reasoning
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于数学推理任务，旨在解决传统链式推理易出错且无法修正的问题。提出Inference-Time Rethinking框架，通过潜在思维向量实现迭代自纠正。**

- **链接: [https://arxiv.org/pdf/2602.06584v1](https://arxiv.org/pdf/2602.06584v1)**

> **作者:** Deqian Kong; Minglu Zhao; Aoyang Qin; Bo Pang; Chenxin Tao; David Hartmann; Edouardo Honig; Dehong Xu; Amit Kumar; Matt Sarte; Chuan Li; Jianwen Xie; Ying Nian Wu
>
> **摘要:** Standard chain-of-thought reasoning generates a solution in a single forward pass, committing irrevocably to each token and lacking a mechanism to recover from early errors. We introduce Inference-Time Rethinking, a generative framework that enables iterative self-correction by decoupling declarative latent thought vectors from procedural generation. We factorize reasoning into a continuous latent thought vector (what to reason about) and a decoder that verbalizes the trace conditioned on this vector (how to reason). Beyond serving as a declarative buffer, latent thought vectors compress the reasoning structure into a continuous representation that abstracts away surface-level token variability, making gradient-based optimization over reasoning strategies well-posed. Our prior model maps unstructured noise to a learned manifold of valid reasoning patterns, and at test time we employ a Gibbs-style procedure that alternates between generating a candidate trace and optimizing the latent vector to better explain that trace, effectively navigating the latent manifold to refine the reasoning strategy. Training a 0.2B-parameter model from scratch on GSM8K, our method with 30 rethinking iterations surpasses baselines with 10 to 15 times more parameters, including a 3B counterpart. This result demonstrates that effective mathematical reasoning can emerge from sophisticated inference-time computation rather than solely from massive parameter counts.
>
---
#### [new 008] Is my model "mind blurting"? Interpreting the dynamics of reasoning tokens with Recurrence Quantification Analysis (RQA)
- **分类: cs.CL**

- **简介: 该论文属于模型分析任务，旨在解决推理过程中难以通过文本分析的问题。通过RQA分析生成过程的动态特性，提升对任务复杂度的预测效果。**

- **链接: [https://arxiv.org/pdf/2602.06266v1](https://arxiv.org/pdf/2602.06266v1)**

> **作者:** Quoc Tuan Pham; Mehdi Jafari; Flora Salim
>
> **摘要:** Test-time compute is central to large reasoning models, yet analysing their reasoning behaviour through generated text is increasingly impractical and unreliable. Response length is often used as a brute proxy for reasoning effort, but this metric fails to capture the dynamics and effectiveness of the Chain of Thoughts (CoT) or the generated tokens. We propose Recurrence Quantification Analysis (RQA) as a non-textual alternative for analysing model's reasoning chains at test time. By treating token generation as a dynamical system, we extract hidden embeddings at each generation step and apply RQA to the resulting trajectories. RQA metrics, including Determinism and Laminarity, quantify patterns of repetition and stalling in the model's latent representations. Analysing 3,600 generation traces from DeepSeek-R1-Distill, we show that RQA captures signals not reflected by response length, but also substantially improves prediction of task complexity by 8\%. These results help establish RQA as a principled tool for studying the latent token generation dynamics of test-time scaling in reasoning models.
>
---
#### [new 009] Diffusion-State Policy Optimization for Masked Diffusion Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言生成任务，解决中间决策信用分配问题。提出DiSPO方法，在扩散模型中间状态优化填充决策，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2602.06462v1](https://arxiv.org/pdf/2602.06462v1)**

> **作者:** Daisuke Oba; Hiroki Furuta; Naoaki Okazaki
>
> **摘要:** Masked diffusion language models generate by iteratively filling masked tokens over multiple denoising steps, so learning only from a terminal reward on the final completion yields coarse credit assignment over intermediate decisions. We propose DiSPO (Diffusion-State Policy Optimization), a plug-in credit-assignment layer that directly optimizes intermediate filling decisions. At selected intermediate masked states, DiSPO branches by resampling fillings for the currently masked positions from rollout-cached logits, scores the resulting completions, and updates only the newly filled tokens -- without additional multi-step diffusion rollouts. We formalize a fixed-state objective for branched completions and derive a policy-gradient estimator that can be combined with terminal-feedback policy optimization using the same rollouts. On LLaDA-8B-Instruct, DiSPO consistently improves over the terminal-feedback diffu-GRPO baseline on math and planning benchmarks under matched rollout compute and optimizer steps. Our code will be available at https://daioba.github.io/dispo .
>
---
#### [new 010] What Is Novel? A Knowledge-Driven Framework for Bias-Aware Literature Originality Evaluation
- **分类: cs.CL**

- **简介: 该论文属于文献新颖性评估任务，旨在解决人工评审主观性强、比较不充分的问题。通过分析同行评审报告，构建知识驱动框架，提升新颖性评估的准确性和一致性。**

- **链接: [https://arxiv.org/pdf/2602.06054v1](https://arxiv.org/pdf/2602.06054v1)**

> **作者:** Abeer Mostafa; Thi Huyen Nguyen; Zahra Ahmadi
>
> **摘要:** Assessing research novelty is a core yet highly subjective aspect of peer review, typically based on implicit judgment and incomplete comparison to prior work. We introduce a literature-aware novelty assessment framework that explicitly learns how humans judge novelty from peer-review reports and grounds these judgments in structured comparison to existing research. Using nearly 80K novelty-annotated reviews from top-tier AI conferences, we fine-tune a large language model to capture reviewer-aligned novelty evaluation behavior. For a given manuscript, the system extracts structured representations of its ideas, methods, and claims, retrieves semantically related papers, and constructs a similarity graph that enables fine-grained, concept-level comparison to prior work. Conditioning on this structured evidence, the model produces calibrated novelty scores and human-like explanatory assessments, reducing overestimation and improving consistency relative to existing approaches.
>
---
#### [new 011] TrailBlazer: History-Guided Reinforcement Learning for Black-Box LLM Jailbreaking
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文属于LLM安全研究任务，旨在解决黑盒LLM越狱问题。通过历史引导的强化学习方法，提升攻击效率与成功率。**

- **链接: [https://arxiv.org/pdf/2602.06440v1](https://arxiv.org/pdf/2602.06440v1)**

> **作者:** Sung-Hoon Yoon; Ruizhi Qian; Minda Zhao; Weiyue Li; Mengyu Wang
>
> **摘要:** Large Language Models (LLMs) have become integral to many domains, making their safety a critical priority. Prior jailbreaking research has explored diverse approaches, including prompt optimization, automated red teaming, obfuscation, and reinforcement learning (RL) based methods. However, most existing techniques fail to effectively leverage vulnerabilities revealed in earlier interaction turns, resulting in inefficient and unstable attacks. Since jailbreaking involves sequential interactions in which each response influences future actions, reinforcement learning provides a natural framework for this problem. Motivated by this, we propose a history-aware RL-based jailbreak framework that analyzes and reweights vulnerability signals from prior steps to guide future decisions. We show that incorporating historical information alone improves jailbreak success rates. Building on this insight, we introduce an attention-based reweighting mechanism that highlights critical vulnerabilities within the interaction history, enabling more efficient exploration with fewer queries. Extensive experiments on AdvBench and HarmBench demonstrate that our method achieves state-of-the-art jailbreak performance while significantly improving query efficiency. These results underscore the importance of historical vulnerability signals in reinforcement learning-driven jailbreak strategies and offer a principled pathway for advancing adversarial research on LLM safeguards.
>
---
#### [new 012] VowelPrompt: Hearing Speech Emotions from Text via Vowel-level Prosodic Augmentation
- **分类: cs.CL**

- **简介: 该论文属于情感识别任务，旨在解决语音情感识别中忽略细粒度韵母级韵律信息的问题。通过VowelPrompt框架，将韵律特征转化为可解释的自然语言描述，提升模型性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2602.06270v1](https://arxiv.org/pdf/2602.06270v1)**

> **作者:** Yancheng Wang; Osama Hanna; Ruiming Xie; Xianfeng Rui; Maohao Shen; Xuedong Zhang; Christian Fuegen; Jilong Wu; Debjyoti Paul; Arthur Guo; Zhihong Lei; Ozlem Kalinli; Qing He; Yingzhen Yang
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Emotion recognition in speech presents a complex multimodal challenge, requiring comprehension of both linguistic content and vocal expressivity, particularly prosodic features such as fundamental frequency, intensity, and temporal dynamics. Although large language models (LLMs) have shown promise in reasoning over textual transcriptions for emotion recognition, they typically neglect fine-grained prosodic information, limiting their effectiveness and interpretability. In this work, we propose VowelPrompt, a linguistically grounded framework that augments LLM-based emotion recognition with interpretable, fine-grained vowel-level prosodic cues. Drawing on phonetic evidence that vowels serve as primary carriers of affective prosody, VowelPrompt extracts pitch-, energy-, and duration-based descriptors from time-aligned vowel segments, and converts these features into natural language descriptions for better interpretability. Such a design enables LLMs to jointly reason over lexical semantics and fine-grained prosodic variation. Moreover, we adopt a two-stage adaptation procedure comprising supervised fine-tuning (SFT) followed by Reinforcement Learning with Verifiable Reward (RLVR), implemented via Group Relative Policy Optimization (GRPO), to enhance reasoning capability, enforce structured output adherence, and improve generalization across domains and speaker variations. Extensive evaluations across diverse benchmark datasets demonstrate that VowelPrompt consistently outperforms state-of-the-art emotion recognition methods under zero-shot, fine-tuned, cross-domain, and cross-linguistic conditions, while enabling the generation of interpretable explanations that are jointly grounded in contextual semantics and fine-grained prosodic structure.
>
---
#### [new 013] Revisiting the Shape Convention of Transformer Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，探讨Transformer模型中MLP结构的优化问题。研究提出使用更深的hourglass结构替代传统窄-宽-窄MLP，以提升模型效率与表现。**

- **链接: [https://arxiv.org/pdf/2602.06471v1](https://arxiv.org/pdf/2602.06471v1)**

> **作者:** Feng-Ting Liao; Meng-Hsi Chen; Guan-Ting Yi; Da-shan Shiu
>
> **摘要:** Dense Transformer language models have largely adhered to one consistent architectural shape: each layer consists of an attention module followed by a feed-forward network (FFN) with a narrow-wide-narrow MLP, allocating most parameters to the MLP at expansion ratios between 2 and 4. Motivated by recent results that residual wide-narrow-wide (hourglass) MLPs offer superior function approximation capabilities, we revisit the long-standing MLP shape convention in Transformer, challenging the necessity of the narrow-wide-narrow design. To study this, we develop a Transformer variant that replaces the conventional FFN with a deeper hourglass-shaped FFN, comprising a stack of hourglass sub-MLPs connected by residual pathways. We posit that a deeper but lighter hourglass FFN can serve as a competitive alternative to the conventional FFN, and that parameters saved by using a lighter hourglass FFN can be more effectively utilized, such as by enlarging model hidden dimensions under fixed budgets. We confirm these through empirical validations across model scales: hourglass FFNs outperform conventional FFNs up to 400M and achieve comparable performance at larger scales to 1B parameters; hourglass FFN variants with reduced FFN and increased attention parameters show consistent improvements over conventional configurations at matched budgets. Together, these findings shed new light on recent work and prompt a rethinking of the narrow-wide-narrow MLP convention and the balance between attention and FFN towards efficient and expressive modern language models.
>
---
#### [new 014] FairJudge: An Adaptive, Debiased, and Consistent LLM-as-a-Judge
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的评价任务，旨在解决LLM-as-a-Judge系统的适应性差、存在偏见和评价不一致的问题。工作包括构建高质量数据集，设计训练策略以提升一致性与公平性。**

- **链接: [https://arxiv.org/pdf/2602.06625v1](https://arxiv.org/pdf/2602.06625v1)**

> **作者:** Bo Yang; Lanfei Feng; Yunkui Chen; Yu Zhang; Xiao Xu; Shijian Li
>
> **摘要:** Existing LLM-as-a-Judge systems suffer from three fundamental limitations: limited adaptivity to task- and domain-specific evaluation criteria, systematic biases driven by non-semantic cues such as position, length, format, and model provenance, and evaluation inconsistency that leads to contradictory judgments across different evaluation modes (e.g., pointwise versus pairwise). To address these issues, we propose FairJudge, an adaptive, debiased, and consistent LLM-as-a-Judge. Unlike prior approaches that treat the judge as a static evaluator, FairJudge models judging behavior itself as a learnable and regularized policy. From a data-centric perspective, we construct a high-information-density judging dataset that explicitly injects supervision signals aligned with evaluation behavior. Building on this dataset, we adopt a curriculum-style SFT-DPO-GRPO training paradigm that progressively aligns rubric adherence, bias mitigation, and cross-mode consistency, while avoiding catastrophic forgetting. Experimental results on multiple internal and public benchmarks show that FairJudge consistently improves agreement and F1, reduces non-semantic biases, and outperforms substantially larger instruction-tuned LLMs. All resources will be publicly released after acceptance to facilitate future research.
>
---
#### [new 015] CORE: Comprehensive Ontological Relation Evaluation for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出CORE评估框架，用于测试大语言模型在语义关系辨别上的能力。任务是评估模型对相关与无关关系的区分能力，解决现有评估不足的问题。工作包括构建数据集和基准测试。**

- **链接: [https://arxiv.org/pdf/2602.06446v1](https://arxiv.org/pdf/2602.06446v1)**

> **作者:** Satyam Dwivedi; Sanjukta Ghosh; Shivam Dwivedi; Nishi Kumari; Anil Thakur; Anurag Purushottam; Deepak Alok; Praveen Gatla; Manjuprasad B; Bipasha Patgiri
>
> **摘要:** Large Language Models (LLMs) perform well on many reasoning benchmarks, yet existing evaluations rarely assess their ability to distinguish between meaningful semantic relations and genuine unrelatedness. We introduce CORE (Comprehensive Ontological Relation Evaluation), a dataset of 225K multiple-choice questions spanning 74 disciplines, together with a general-domain open-source benchmark of 203 rigorously validated questions (Cohen's Kappa = 1.0) covering 24 semantic relation types with equal representation of unrelated pairs. A human baseline from 1,000+ participants achieves 92.6% accuracy (95.1% on unrelated pairs). In contrast, 29 state-of-the-art LLMs achieve 48.25-70.9% overall accuracy, with near-ceiling performance on related pairs (86.5-100%) but severe degradation on unrelated pairs (0-41.35%), despite assigning similar confidence (92-94%). Expected Calibration Error increases 2-4x on unrelated pairs, and a mean semantic collapse rate of 37.6% indicates systematic generation of spurious relations. On the CORE 225K MCQs dataset, accuracy further drops to approximately 2%, highlighting substantial challenges in domain-specific semantic reasoning. We identify unrelatedness reasoning as a critical, under-evaluated frontier for LLM evaluation and safety.
>
---
#### [new 016] PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models
- **分类: cs.CL**

- **简介: 该论文提出PersonaPlex，解决对话模型角色与语音控制问题，通过混合提示实现多角色、个性化交互，提升对话自然度与响应性。**

- **链接: [https://arxiv.org/pdf/2602.06053v1](https://arxiv.org/pdf/2602.06053v1)**

> **作者:** Rajarshi Roy; Jonathan Raiman; Sang-gil Lee; Teodor-Dumitru Ene; Robert Kirby; Sungwon Kim; Jaehyeon Kim; Bryan Catanzaro
>
> **摘要:** Recent advances in duplex speech models have enabled natural, low-latency speech-to-speech interactions. However, existing models are restricted to a fixed role and voice, limiting their ability to support structured, role-driven real-world applications and personalized interactions. In this work, we introduce PersonaPlex, a duplex conversational speech model that incorporates hybrid system prompts, combining role conditioning with text prompts and voice cloning with speech samples. PersonaPlex is trained on a large-scale synthetic dataset of paired prompts and user-agent conversations, generated with open-source large language models (LLM) and text-to-speech (TTS) models. To evaluate role conditioning in real-world settings, we extend the Full-Duplex-Bench benchmark beyond a single assistant role to multi-role customer service scenarios. Experiments show that PersonaPlex achieves strong role-conditioned behavior, voice-conditioned speech, and natural conversational responsiveness, surpassing state-of-the-art duplex speech models and hybrid large language model-based speech systems in role adherence, speaker similarity, latency, and naturalness.
>
---
#### [new 017] Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevant Assessment for IR Benchmarks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索任务，解决IR基准数据集缺失标注的问题。通过DREAM框架提升标注准确性并减少人工参与，构建了BRIDGE基准以提高评估公平性。**

- **链接: [https://arxiv.org/pdf/2602.06526v1](https://arxiv.org/pdf/2602.06526v1)**

> **作者:** Minjeong Ban; Jeonghwan Choi; Hyangsuk Min; Nicole Hee-Yeon Kim; Minseok Kim; Jae-Gil Lee; Hwanjun Song
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Information retrieval (IR) evaluation remains challenging due to incomplete IR benchmark datasets that contain unlabeled relevant chunks. While LLMs and LLM-human hybrid strategies reduce costly human effort, they remain prone to LLM overconfidence and ineffective AI-to-human escalation. To address this, we propose DREAM, a multi-round debate-based relevance assessment framework with LLM agents, built on opposing initial stances and iterative reciprocal critique. Through our agreement-based debate, it yields more accurate labeling for certain cases and more reliable AI-to-human escalation for uncertain ones, achieving 95.2% labeling accuracy with only 3.5% human involvement. Using DREAM, we build BRIDGE, a refined benchmark that mitigates evaluation bias and enables fairer retriever comparison by uncovering 29,824 missing relevant chunks. We then re-benchmark IR systems and extend evaluation to RAG, showing that unaddressed holes not only distort retriever rankings but also drive retrieval-generation misalignment. The relevance assessment framework is available at https: //github.com/DISL-Lab/DREAM-ICLR-26; and the BRIDGE dataset is available at https://github.com/DISL-Lab/BRIDGE-Benchmark.
>
---
#### [new 018] FMBench: Adaptive Large Language Model Output Formatting
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的格式生成任务，旨在解决大模型输出的Markdown格式错误问题。提出FMBench基准和轻量对齐方法，提升格式合规性与语义一致性。**

- **链接: [https://arxiv.org/pdf/2602.06384v1](https://arxiv.org/pdf/2602.06384v1)**

> **作者:** Yaoting Wang; Yun Zhou; Henghui Ding
>
> **摘要:** Producing outputs that satisfy both semantic intent and format constraints is essential for deploying large language models in user-facing and system-integrated workflows. In this work, we focus on Markdown formatting, which is ubiquitous in assistants, documentation, and tool-augmented pipelines but still prone to subtle, hard-to-detect errors (e.g., broken lists, malformed tables, inconsistent headings, and invalid code blocks) that can significantly degrade downstream usability. We present FMBench, a benchmark for adaptive Markdown output formatting that evaluates models under a wide range of instruction-following scenarios with diverse structural requirements. FMBench emphasizes real-world formatting behaviors such as multi-level organization, mixed content (natural language interleaved with lists/tables/code), and strict adherence to user-specified layout constraints. To improve Markdown compliance without relying on hard decoding constraints, we propose a lightweight alignment pipeline that combines supervised fine-tuning (SFT) with reinforcement learning fine-tuning. Starting from a base model, we first perform SFT on instruction-response pairs, and then optimize a composite objective that balances semantic fidelity with structural correctness. Experiments on two model families (OpenPangu and Qwen) show that SFT consistently improves semantic alignment, while reinforcement learning provides additional gains in robustness to challenging Markdown instructions when initialized from a strong SFT policy. Our results also reveal an inherent trade-off between semantic and structural objectives, highlighting the importance of carefully designed rewards for reliable formatted generation. Code is available at: https://github.com/FudanCVL/FMBench.
>
---
#### [new 019] Relevance-aware Multi-context Contrastive Decoding for Retrieval-augmented Visual Question Answering
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉问答任务，旨在解决LVLMs缺乏实体细节知识的问题。提出RMCD方法，通过多上下文对比解码提升RAG效果，有效融合相关上下文并抑制无关影响。**

- **链接: [https://arxiv.org/pdf/2602.06050v1](https://arxiv.org/pdf/2602.06050v1)**

> **作者:** Jongha Kim; Byungoh Ko; Jeehye Na; Jinsung Yoon; Hyunwoo J. Kim
>
> **备注:** WACV 2026
>
> **摘要:** Despite the remarkable capabilities of Large Vision Language Models (LVLMs), they still lack detailed knowledge about specific entities. Retrieval-augmented Generation (RAG) is a widely adopted solution that enhances LVLMs by providing additional contexts from an external Knowledge Base. However, we observe that previous decoding methods for RAG are sub-optimal as they fail to sufficiently leverage multiple relevant contexts and suppress the negative effects of irrelevant contexts. To this end, we propose Relevance-aware Multi-context Contrastive Decoding (RMCD), a novel decoding method for RAG. RMCD outputs a final prediction by combining outputs predicted with each context, where each output is weighted based on its relevance to the question. By doing so, RMCD effectively aggregates useful information from multiple relevant contexts while also counteracting the negative effects of irrelevant ones. Experiments show that RMCD consistently outperforms other decoding methods across multiple LVLMs, achieving the best performance on three knowledge-intensive visual question-answering benchmarks. Also, RMCD can be simply applied by replacing the decoding method of LVLMs without additional training. Analyses also show that RMCD is robust to the retrieval results, consistently performing the best across the weakest to the strongest retrieval results. Code is available at https://github.com/mlvlab/RMCD.
>
---
#### [new 020] Can One-sided Arguments Lead to Response Change in Large Language Models?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的观点引导任务，研究如何通过单方面论据影响大语言模型的回应立场。**

- **链接: [https://arxiv.org/pdf/2602.06260v1](https://arxiv.org/pdf/2602.06260v1)**

> **作者:** Pedro Cisneros-Velarde
>
> **摘要:** Polemic questions need more than one viewpoint to express a balanced answer. Large Language Models (LLMs) can provide a balanced answer, but also take a single aligned viewpoint or refuse to answer. In this paper, we study if such initial responses can be steered to a specific viewpoint in a simple and intuitive way: by only providing one-sided arguments supporting the viewpoint. Our systematic study has three dimensions: (i) which stance is induced in the LLM response, (ii) how the polemic question is formulated, (iii) how the arguments are shown. We construct a small dataset and remarkably find that opinion steering occurs across (i)-(iii) for diverse models, number of arguments, and topics. Switching to other arguments consistently decreases opinion steering.
>
---
#### [new 021] Cost-Aware Model Selection for Text Classification: Multi-Objective Trade-offs Between Fine-Tuned Encoders and LLM Prompting in Production
- **分类: cs.CL**

- **简介: 该论文属于文本分类任务，解决模型选择问题，比较了微调编码器与LLM提示方法的性能、延迟和成本，提出多目标优化方案。**

- **链接: [https://arxiv.org/pdf/2602.06370v1](https://arxiv.org/pdf/2602.06370v1)**

> **作者:** Alberto Andres Valdes Gonzalez
>
> **备注:** 26 pages, 12 figures. Empirical benchmark comparing fine-tuned encoders and LLM prompting for text classification under cost and latency constraints
>
> **摘要:** Large language models (LLMs) such as GPT-4o and Claude Sonnet 4.5 have demonstrated strong capabilities in open-ended reasoning and generative language tasks, leading to their widespread adoption across a broad range of NLP applications. However, for structured text classification problems with fixed label spaces, model selection is often driven by predictive performance alone, overlooking operational constraints encountered in production systems. In this work, we present a systematic comparison of two contrasting paradigms for text classification: zero- and few-shot prompt-based large language models, and fully fine-tuned encoder-only architectures. We evaluate these approaches across four canonical benchmarks (IMDB, SST-2, AG News, and DBPedia), measuring predictive quality (macro F1), inference latency, and monetary cost. We frame model evaluation as a multi-objective decision problem and analyze trade-offs using Pareto frontier projections and a parameterized utility function reflecting different deployment regimes. Our results show that fine-tuned encoder-based models from the BERT family achieve competitive, and often superior, classification performance while operating at one to two orders of magnitude lower cost and latency compared to zero- and few-shot LLM prompting. Overall, our findings suggest that indiscriminate use of large language models for standard text classification workloads can lead to suboptimal system-level outcomes. Instead, fine-tuned encoders emerge as robust and efficient components for structured NLP pipelines, while LLMs are better positioned as complementary elements within hybrid architectures. We release all code, datasets, and evaluation protocols to support reproducibility and cost-aware NLP system design.
>
---
#### [new 022] Evaluating an evidence-guided reinforcement learning framework in aligning light-parameter large language models with decision-making cognition in psychiatric clinical reasoning
- **分类: cs.CL**

- **简介: 该论文属于医疗决策支持任务，旨在解决轻量级大语言模型在精神科推理中的逻辑偏差问题。通过引入ClinMPO框架，提升模型与专业诊断认知的对齐度。**

- **链接: [https://arxiv.org/pdf/2602.06449v1](https://arxiv.org/pdf/2602.06449v1)**

> **作者:** Xinxin Lin; Guangxin Dai; Yi Zhong; Xiang Li; Xue Xiao; Yixin Zhang; Zhengdong Wu; Yongbo Zheng; Runchuan Zhu; Ming Zhao; Huizi Yu; Shuo Wu; Jun Zhao; Lingming Hu; Yumei Wang; Ping Yin; Joey W. Y. Chan; Ngan Yin Chan; Sijing Chen; Yun Kwok Wing; Lin Lu; Xin Ma; Lizhou Fan
>
> **备注:** 21 pages, 8 figures
>
> **摘要:** Large language models (LLMs) hold transformative potential for medical decision support yet their application in psychiatry remains constrained by hallucinations and superficial reasoning. This limitation is particularly acute in light-parameter LLMs which are essential for privacy-preserving and efficient clinical deployment. Existing training paradigms prioritize linguistic fluency over structured clinical logic and result in a fundamental misalignment with professional diagnostic cognition. Here we introduce ClinMPO, a reinforcement learning framework designed to align the internal reasoning of LLMs with professional psychiatric practice. The framework employs a specialized reward model trained independently on a dataset derived from 4,474 psychiatry journal articles and structured according to evidence-based medicine principles. We evaluated ClinMPO on a unseen subset of the benchmark designed to isolate reasoning capabilities from rote memorization. This test set comprises items where leading large-parameter LLMs consistently fail. We compared the ClinMPO-aligned light LLM performance against a cohort of 300 medical students. The ClinMPO-tuned Qwen3-8B model achieved a diagnostic accuracy of 31.4% and surpassed the human benchmark of 30.8% on these complex cases. These results demonstrate that medical evidence-guided optimization enables light-parameter LLMs to master complex reasoning tasks. Our findings suggest that explicit cognitive alignment offers a scalable pathway to reliable and safe psychiatric decision support.
>
---
#### [new 023] Rethinking Memory Mechanisms of Foundation Agents in the Second Half
- **分类: cs.CL; cs.AI**

- **简介: 本文探讨基础智能体的记忆机制，旨在提升其在复杂环境中的长期实用性。研究涵盖记忆类型、操作方式及评估方法，解决智能体信息管理与利用问题。**

- **链接: [https://arxiv.org/pdf/2602.06052v1](https://arxiv.org/pdf/2602.06052v1)**

> **作者:** Wei-Chieh Huang; Weizhi Zhang; Yueqing Liang; Yuanchen Bei; Yankai Chen; Tao Feng; Xinyu Pan; Zhen Tan; Yu Wang; Tianxin Wei; Shanglin Wu; Ruiyao Xu; Liangwei Yang; Rui Yang; Wooseong Yang; Chin-Yuan Yeh; Hanrong Zhang; Haozhen Zhang; Siqi Zhu; Henry Peng Zou; Wanjia Zhao; Song Wang; Wujiang Xu; Zixuan Ke; Zheng Hui; Dawei Li; Yaozu Wu; Langzhou He; Chen Wang; Xiongxiao Xu; Baixiang Huang; Juntao Tan; Shelby Heinecke; Huan Wang; Caiming Xiong; Ahmed A. Metwally; Jun Yan; Chen-Yu Lee; Hanqing Zeng; Yinglong Xia; Xiaokai Wei; Ali Payani; Yu Wang; Haitong Ma; Wenya Wang; Chengguang Wang; Yu Zhang; Xin Wang; Yongfeng Zhang; Jiaxuan You; Hanghang Tong; Xiao Luo; Yizhou Sun; Wei Wang; Julian McAuley; James Zou; Jiawei Han; Philip S. Yu; Kai Shu
>
> **摘要:** The research of artificial intelligence is undergoing a paradigm shift from prioritizing model innovations over benchmark scores towards emphasizing problem definition and rigorous real-world evaluation. As the field enters the "second half," the central challenge becomes real utility in long-horizon, dynamic, and user-dependent environments, where agents face context explosion and must continuously accumulate, manage, and selectively reuse large volumes of information across extended interactions. Memory, with hundreds of papers released this year, therefore emerges as the critical solution to fill the utility gap. In this survey, we provide a unified view of foundation agent memory along three dimensions: memory substrate (internal and external), cognitive mechanism (episodic, semantic, sensory, working, and procedural), and memory subject (agent- and user-centric). We then analyze how memory is instantiated and operated under different agent topologies and highlight learning policies over memory operations. Finally, we review evaluation benchmarks and metrics for assessing memory utility, and outline various open challenges and future directions.
>
---
#### [new 024] SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SHINE，一种可扩展的上下文到LoRA适配器映射方法，解决LLM适应效率低的问题。通过单次前向传播生成高质量适配器，提升任务性能并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2602.06358v1](https://arxiv.org/pdf/2602.06358v1)**

> **作者:** Yewei Liu; Xiyuan Wang; Yansheng Mao; Yoav Gelbery; Haggai Maron; Muhan Zhang
>
> **摘要:** We propose SHINE (Scalable Hyper In-context NEtwork), a scalable hypernetwork that can map diverse meaningful contexts into high-quality LoRA adapters for large language models (LLM). By reusing the frozen LLM's own parameters in an in-context hypernetwork design and introducing architectural innovations, SHINE overcomes key limitations of prior hypernetworks and achieves strong expressive power with a relatively small number of parameters. We introduce a pretraining and instruction fine-tuning pipeline, and train our hypernetwork to generate high quality LoRA adapters from diverse meaningful contexts in a single forward pass. It updates LLM parameters without any fine-tuning, and immediately enables complex question answering tasks related to the context without directly accessing the context, effectively transforming in-context knowledge to in-parameter knowledge in one pass. Our work achieves outstanding results on various tasks, greatly saves time, computation and memory costs compared to SFT-based LLM adaptation, and shows great potential for scaling. Our code is available at https://github.com/Yewei-Liu/SHINE
>
---
#### [new 025] Table-as-Search: Formulate Long-Horizon Agentic Information Seeking as Table Completion
- **分类: cs.CL**

- **简介: 该论文提出Table-as-Search（TaS），将信息检索任务转化为表格补全问题，解决长周期搜索中的状态管理和任务统一问题。**

- **链接: [https://arxiv.org/pdf/2602.06724v1](https://arxiv.org/pdf/2602.06724v1)**

> **作者:** Tian Lan; Felix Henry; Bin Zhu; Qianghuai Jia; Junyang Ren; Qihang Pu; Haijun Li; Longyue Wang; Zhao Xu; Weihua Luo
>
> **摘要:** Current Information Seeking (InfoSeeking) agents struggle to maintain focus and coherence during long-horizon exploration, as tracking search states, including planning procedure and massive search results, within one plain-text context is inherently fragile. To address this, we introduce \textbf{Table-as-Search (TaS)}, a structured planning framework that reformulates the InfoSeeking task as a Table Completion task. TaS maps each query into a structured table schema maintained in an external database, where rows represent search candidates and columns denote constraints or required information. This table precisely manages the search states: filled cells strictly record the history and search results, while empty cells serve as an explicit search plan. Crucially, TaS unifies three distinct InfoSeeking tasks: Deep Search, Wide Search, and the challenging DeepWide Search. Extensive experiments demonstrate that TaS significantly outperforms numerous state-of-the-art baselines across three kinds of benchmarks, including multi-agent framework and commercial systems. Furthermore, our analysis validates the TaS's superior robustness in long-horizon InfoSeeking, alongside its efficiency, scalability and flexibility. Code and datasets are publicly released at https://github.com/AIDC-AI/Marco-Search-Agent.
>
---
#### [new 026] MPIB: A Benchmark for Medical Prompt Injection Attacks and Clinical Safety in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于医疗AI安全任务，旨在解决LLMs在临床应用中的提示注入攻击问题。通过构建MPIB基准，评估系统在直接和间接攻击下的临床安全风险。**

- **链接: [https://arxiv.org/pdf/2602.06268v1](https://arxiv.org/pdf/2602.06268v1)**

> **作者:** Junhyeok Lee; Han Jang; Kyu Sung Choi
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems are increasingly integrated into clinical workflows; however, prompt injection attacks can steer these systems toward clinically unsafe or misleading outputs. We introduce the Medical Prompt Injection Benchmark (MPIB), a dataset-and-benchmark suite for evaluating clinical safety under both direct prompt injection and indirect, RAG-mediated injection across clinically grounded tasks. MPIB emphasizes outcome-level risk via the Clinical Harm Event Rate (CHER), which measures high-severity clinical harm events under a clinically grounded taxonomy, and reports CHER alongside Attack Success Rate (ASR) to disentangle instruction compliance from downstream patient risk. The benchmark comprises 9,697 curated instances constructed through multi-stage quality gates and clinical safety linting. Evaluating MPIB across a diverse set of baseline LLMs and defense configurations, we find that ASR and CHER can diverge substantially, and that robustness depends critically on whether adversarial instructions appear in the user query or in retrieved context. We release MPIB with evaluation code, adversarial baselines, and comprehensive documentation to support reproducible and systematic research on clinical prompt injection. Code and data are available at GitHub (code) and Hugging Face (data).
>
---
#### [new 027] MTQE.en-he: Machine Translation Quality Estimation for English-Hebrew
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译质量评估任务，旨在解决英语-希伯来语翻译质量估计问题。作者发布了首个公开的基准数据集MTQE.en-he，并测试了多个模型，提出参数高效微调方法。**

- **链接: [https://arxiv.org/pdf/2602.06546v1](https://arxiv.org/pdf/2602.06546v1)**

> **作者:** Andy Rosenbaum; Assaf Siani; Ilan Kernerman
>
> **备注:** Accepted to LoResLM at EACL 2026
>
> **摘要:** We release MTQE.en-he: to our knowledge, the first publicly available English-Hebrew benchmark for Machine Translation Quality Estimation. MTQE.en-he contains 959 English segments from WMT24++, each paired with a machine translation into Hebrew, and Direct Assessment scores of the translation quality annotated by three human experts. We benchmark ChatGPT prompting, TransQuest, and CometKiwi and show that ensembling the three models outperforms the best single model (CometKiwi) by 6.4 percentage points Pearson and 5.6 percentage points Spearman. Fine-tuning experiments with TransQuest and CometKiwi reveal that full-model updates are sensitive to overfitting and distribution collapse, yet parameter-efficient methods (LoRA, BitFit, and FTHead, i.e., fine-tuning only the classification head) train stably and yield improvements of 2-3 percentage points. MTQE.en-he and our experimental results enable future research on this under-resourced language pair.
>
---
#### [new 028] Halluverse-M^3: A multitask multilingual benchmark for hallucination in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Halluverse-M^3，一个用于研究多语言、多任务幻觉的基准数据集。旨在解决大模型在多语言场景下的事实一致性问题，通过构建包含四种语言的生成任务数据，分析不同层次的幻觉现象。**

- **链接: [https://arxiv.org/pdf/2602.06920v1](https://arxiv.org/pdf/2602.06920v1)**

> **作者:** Samir Abdaljalil; Parichit Sharma; Erchin Serpedin; Hasan Kurban
>
> **摘要:** Hallucinations in large language models remain a persistent challenge, particularly in multilingual and generative settings where factual consistency is difficult to maintain. While recent models show strong performance on English-centric benchmarks, their behavior across languages, tasks, and hallucination types is not yet well understood. In this work, we introduce Halluverse-M^3, a dataset designed to enable systematic analysis of hallucinations across multiple languages, multiple generation tasks, and multiple hallucination categories. Halluverse-M^3 covers four languages, English, Arabic, Hindi, and Turkish, and supports two generation tasks: question answering and dialogue summarization. The dataset explicitly distinguishes between entity-level, relation-level, and sentence-level hallucinations. Hallucinated outputs are constructed through a controlled editing process and validated by human annotators, ensuring clear alignment between original content and hallucinated generations. Using this dataset, we evaluate a diverse set of contemporary open-source and proprietary language models on fine-grained hallucination detection. Our results show that question answering is consistently easier than dialogue summarization, while sentence-level hallucinations remain challenging even for the strongest models. Performance is highest in English and degrades in lower-resource languages, with Hindi exhibiting the lowest detection accuracy. Overall, Halluverse-M^3 provides a realistic and challenging benchmark for studying hallucinations in multilingual, multi-task settings. We release the dataset to support future research on hallucination detection and mitigation\footnote{https://huggingface.co/datasets/sabdalja/HalluVerse-M3}.
>
---
#### [new 029] Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型生成任务，旨在解决Masked Diffusion-LM中计算资源浪费问题。通过锁定稳定token位置，减少计算量，提升效率。**

- **链接: [https://arxiv.org/pdf/2602.06412v1](https://arxiv.org/pdf/2602.06412v1)**

> **作者:** Daisuke Oba; Danushka Bollegala; Masahiro Kaneko; Naoaki Okazaki
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Masked Diffusion Language Models generate sequences via iterative sampling that progressively unmasks tokens. However, they still recompute the attention and feed-forward blocks for every token position at every step -- even when many unmasked tokens are essentially fixed, resulting in substantial waste in compute. We propose SureLock: when the posterior at an unmasked position has stabilized across steps (our sure condition), we lock that position -- thereafter skipping its query projection and feed-forward sublayers -- while caching its attention keys and values so other positions can continue to attend to it. This reduces the dominant per-iteration computational cost from $O(N^2d)$ to $O(MNd)$ where $N$ is the sequence length, $M$ is the number of unlocked token positions, and $d$ is the model dimension. In practice, $M$ decreases as the iteration progresses, yielding substantial savings. On LLaDA-8B, SureLock reduces algorithmic FLOPs by 30--50% relative to the same sampler without locking, while maintaining comparable generation quality. We also provide a theoretical analysis to justify the design rationale of SureLock: monitoring only the local KL at the lock step suffices to bound the deviation in final token probabilities. Our code will be available at https://daioba.github.io/surelock .
>
---
#### [new 030] Recontextualizing Famous Quotes for Brand Slogan Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于品牌标语生成任务，旨在解决现有方法生成标语缺乏创意和个性的问题。通过重新语境化名人名言，提出一种模块化框架，提升标语的多样性与情感影响力。**

- **链接: [https://arxiv.org/pdf/2602.06049v1](https://arxiv.org/pdf/2602.06049v1)**

> **作者:** Ziao Yang; Zizhang Chen; Lei Zhang; Hongfu Liu
>
> **摘要:** Slogans are concise and memorable catchphrases that play a crucial role in advertising by conveying brand identity and shaping public perception. However, advertising fatigue reduces the effectiveness of repeated slogans, creating a growing demand for novel, creative, and insightful slogan generation. While recent work leverages large language models (LLMs) for this task, existing approaches often produce stylistically redundant outputs that lack a clear brand persona and appear overtly machine-generated. We argue that effective slogans should balance novelty with familiarity and propose a new paradigm that recontextualizes persona-related famous quotes for slogan generation. Well-known quotes naturally align with slogan-length text, employ rich rhetorical devices, and offer depth and insight, making them a powerful resource for creative generation. Technically, we introduce a modular framework that decomposes slogan generation into interpretable subtasks, including quote matching, structural decomposition, vocabulary replacement, and remix generation. Extensive automatic and human evaluations demonstrate marginal improvements in diversity, novelty, emotional impact, and human preference over three state-of-the-art LLM baselines.
>
---
#### [new 031] The Representational Geometry of Number
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型中数字概念的表征结构，探讨其如何在不同任务间保持关系一致性。旨在解决表征共享与任务隔离的矛盾，通过分析线性可转换子空间验证关系结构的稳定性。**

- **链接: [https://arxiv.org/pdf/2602.06843v1](https://arxiv.org/pdf/2602.06843v1)**

> **作者:** Zhimin Hu; Lanhao Niu; Sashank Varma
>
> **摘要:** A central question in cognitive science is whether conceptual representations converge onto a shared manifold to support generalization, or diverge into orthogonal subspaces to minimize task interference. While prior work has discovered evidence for both, a mechanistic account of how these properties coexist and transform across tasks remains elusive. We propose that representational sharing lies not in the concepts themselves, but in the geometric relations between them. Using number concepts as a testbed and language models as high-dimensional computational substrates, we show that number representations preserve a stable relational structure across tasks. Task-specific representations are embedded in distinct subspaces, with low-level features like magnitude and parity encoded along separable linear directions. Crucially, we find that these subspaces are largely transformable into one another via linear mappings, indicating that representations share relational structure despite being located in distinct subspaces. Together, these results provide a mechanistic lens of how language models balance the shared structure of number representation with functional flexibility. It suggests that understanding arises when task-specific transformations are applied to a shared underlying relational structure of conceptual representations.
>
---
#### [new 032] Beyond Static Alignment: Hierarchical Policy Control for LLM Safety via Risk-Aware Chain-of-Thought
- **分类: cs.CL**

- **简介: 该论文属于LLM安全对齐任务，解决静态安全策略导致的安全与帮助性矛盾问题。提出PACT框架，通过分层策略实现动态安全控制。**

- **链接: [https://arxiv.org/pdf/2602.06650v1](https://arxiv.org/pdf/2602.06650v1)**

> **作者:** Jianfeng Si; Lin Sun; Weihong Lin; Xiangzheng Zhang
>
> **备注:** 13 pages, 5 tables, 2 figures
>
> **摘要:** Large Language Models (LLMs) face a fundamental safety-helpfulness trade-off due to static, one-size-fits-all safety policies that lack runtime controllabilityxf, making it difficult to tailor responses to diverse application needs. %As a result, models may over-refuse benign requests or under-constrain harmful ones. We present \textbf{PACT} (Prompt-configured Action via Chain-of-Thought), a framework for dynamic safety control through explicit, risk-aware reasoning. PACT operates under a hierarchical policy architecture: a non-overridable global safety policy establishes immutable boundaries for critical risks (e.g., child safety, violent extremism), while user-defined policies can introduce domain-specific (non-global) risk categories and specify label-to-action behaviors to improve utility in real-world deployment settings. The framework decomposes safety decisions into structured Classify$\rightarrow$Act paths that route queries to the appropriate action (comply, guide, or reject) and render the decision-making process transparent. Extensive experiments demonstrate that PACT achieves near state-of-the-art safety performance under global policy evaluation while attaining the best controllability under user-specific policy evaluation, effectively mitigating the safety-helpfulness trade-off. We will release the PACT model suite, training data, and evaluation protocols to facilitate reproducible research in controllable safety alignment.
>
---
#### [new 033] compar:IA: The French Government's LLM arena to collect French-language human prompts and preference data
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍compar:IA平台，旨在收集法语人类偏好数据以提升非英语语言模型性能。属于自然语言处理任务，解决非英语语言模型训练数据不足问题，通过用户投票和对话数据进行模型评估与优化。**

- **链接: [https://arxiv.org/pdf/2602.06669v1](https://arxiv.org/pdf/2602.06669v1)**

> **作者:** Lucie Termignon; Simonas Zilinskas; Hadrien Pélissier; Aurélien Barrot; Nicolas Chesnais; Elie Gavoty
>
> **备注:** 18 pages, 7 figures, preprint
>
> **摘要:** Large Language Models (LLMs) often show reduced performance, cultural alignment, and safety robustness in non-English languages, partly because English dominates both pre-training data and human preference alignment datasets. Training methods like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) require human preference data, which remains scarce and largely non-public for many languages beyond English. To address this gap, we introduce compar:IA, an open-source digital public service developed inside the French government and designed to collect large-scale human preference data from a predominantly French-speaking general audience. The platform uses a blind pairwise comparison interface to capture unconstrained, real-world prompts and user judgments across a diverse set of language models, while maintaining low participation friction and privacy-preserving automated filtering. As of 2026-02-07, compar:IA has collected over 600,000 free-form prompts and 250,000 preference votes, with approximately 89% of the data in French. We release three complementary datasets -- conversations, votes, and reactions -- under open licenses, and present initial analyses, including a French-language model leaderboard and user interaction patterns. Beyond the French context, compar:IA is evolving toward an international digital public good, offering reusable infrastructure for multilingual model training, evaluation, and the study of human-AI interaction.
>
---
#### [new 034] On the Wings of Imagination: Conflicting Script-based Multi-role Framework for Humor Caption Generation
- **分类: cs.CL**

- **简介: 该论文属于多模态幽默标题生成任务，旨在解决大语言模型在幽默生成中创意和可解释性不足的问题。提出HOMER框架，通过角色协作生成有趣且多样化的幽默标题。**

- **链接: [https://arxiv.org/pdf/2602.06423v1](https://arxiv.org/pdf/2602.06423v1)**

> **作者:** Wenbo Shang; Yuxi Sun; Jing Ma; Xin Huang
>
> **备注:** Paper accepted as a conference paper at ICLR 2026
>
> **摘要:** Humor is a commonly used and intricate human language in daily life. Humor generation, especially in multi-modal scenarios, is a challenging task for large language models (LLMs), which is typically as funny caption generation for images, requiring visual understanding, humor reasoning, creative imagination, and so on. Existing LLM-based approaches rely on reasoning chains or self-improvement, which suffer from limited creativity and interpretability. To address these bottlenecks, we develop a novel LLM-based humor generation mechanism based on a fundamental humor theory, GTVH. To produce funny and script-opposite captions, we introduce a humor-theory-driven multi-role LLM collaboration framework augmented with humor retrieval (HOMER). The framework consists of three LLM-based roles: (1) conflicting-script extractor that grounds humor in key script oppositions, forming the basis of caption generation; (2) retrieval-augmented hierarchical imaginator that identifies key humor targets and expands the creative space of them through diverse associations structured as imagination trees; and (3) caption generator that produces funny and diverse captions conditioned on the obtained knowledge. Extensive experiments on two New Yorker Cartoon benchmarking datasets show that HOMER outperforms state-of-the-art baselines and powerful LLM reasoning strategies on multi-modal humor captioning.
>
---
#### [new 035] InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于推理任务，解决长序列推理中的效率与效果问题。提出InftyThink+框架，通过强化学习优化迭代推理过程，提升准确率并减少延迟。**

- **链接: [https://arxiv.org/pdf/2602.06960v1](https://arxiv.org/pdf/2602.06960v1)**

> **作者:** Yuchen Yan; Liang Jiang; Jin Jiang; Shuaicheng Li; Zujie Wen; Zhiqiang Zhang; Jun Zhou; Jian Shao; Yueting Zhuang; Yongliang Shen
>
> **备注:** Project Page: https://zju-real.github.io/InftyThink-Plus Code: https://github.com/ZJU-REAL/InftyThink-Plus
>
> **摘要:** Large reasoning models achieve strong performance by scaling inference-time chain-of-thought, but this paradigm suffers from quadratic cost, context length limits, and degraded reasoning due to lost-in-the-middle effects. Iterative reasoning mitigates these issues by periodically summarizing intermediate thoughts, yet existing methods rely on supervised learning or fixed heuristics and fail to optimize when to summarize, what to preserve, and how to resume reasoning. We propose InftyThink+, an end-to-end reinforcement learning framework that optimizes the entire iterative reasoning trajectory, building on model-controlled iteration boundaries and explicit summarization. InftyThink+ adopts a two-stage training scheme with supervised cold-start followed by trajectory-level reinforcement learning, enabling the model to learn strategic summarization and continuation decisions. Experiments on DeepSeek-R1-Distill-Qwen-1.5B show that InftyThink+ improves accuracy by 21% on AIME24 and outperforms conventional long chain-of-thought reinforcement learning by a clear margin, while also generalizing better to out-of-distribution benchmarks. Moreover, InftyThink+ significantly reduces inference latency and accelerates reinforcement learning training, demonstrating improved reasoning efficiency alongside stronger performance.
>
---
#### [new 036] Evaluating Prompt Engineering Strategies for Sentiment Control in AI-Generated Texts
- **分类: cs.CL**

- **简介: 该论文属于情感控制任务，旨在解决AI生成文本情感调控难题。通过对比不同提示工程技术，发现少量人工示例提示效果最佳，为数据受限场景提供有效方案。**

- **链接: [https://arxiv.org/pdf/2602.06692v1](https://arxiv.org/pdf/2602.06692v1)**

> **作者:** Kerstin Sahler; Sophie Jentzsch
>
> **备注:** The definitive, peer-reviewed and edited version of this article is published in HHAI 2025 - Proceedings of the Fourth International Conference on Hybrid Human-Artificial Intelligence, Frontiers in Artificial Intelligence and Applications, Volume 408, ISBN 978-1-64368-611-0, pages 423 - 438, 2025
>
> **摘要:** The groundbreaking capabilities of Large Language Models (LLMs) offer new opportunities for enhancing human-computer interaction through emotion-adaptive Artificial Intelligence (AI). However, deliberately controlling the sentiment in these systems remains challenging. The present study investigates the potential of prompt engineering for controlling sentiment in LLM-generated text, providing a resource-sensitive and accessible alternative to existing methods. Using Ekman's six basic emotions (e.g., joy, disgust), we examine various prompting techniques, including Zero-Shot and Chain-of-Thought prompting using gpt-3.5-turbo, and compare it to fine-tuning. Our results indicate that prompt engineering effectively steers emotions in AI-generated texts, offering a practical and cost-effective alternative to fine-tuning, especially in data-constrained settings. In this regard, Few-Shot prompting with human-written examples was the most effective among other techniques, likely due to the additional task-specific guidance. The findings contribute valuable insights towards developing emotion-adaptive AI systems.
>
---
#### [new 037] Uncertainty Drives Social Bias Changes in Quantized Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究量化对大语言模型社会偏见的影响，属于模型压缩与偏见分析任务。它揭示了量化导致偏见状态翻转的现象，强调需进行量化后评估以确保可靠性。**

- **链接: [https://arxiv.org/pdf/2602.06181v1](https://arxiv.org/pdf/2602.06181v1)**

> **作者:** Stanley Z. Hua; Sanae Lotfi; Irene Y. Chen
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Post-training quantization reduces the computational cost of large language models but fundamentally alters their social biases in ways that aggregate metrics fail to capture. We present the first large-scale study of 50 quantized models evaluated on PostTrainingBiasBench, a unified benchmark of 13 closed- and open-ended bias datasets. We identify a phenomenon we term quantization-induced masked bias flipping, in which up to 21% of responses flip between biased and unbiased states after quantization, despite showing no change in aggregate bias scores. These flips are strongly driven by model uncertainty, where the responses with high uncertainty are 3-11x more likely to change than the confident ones. Quantization strength amplifies this effect, with 4-bit quantized models exhibiting 4-6x more behavioral changes than 8-bit quantized models. Critically, these changes create asymmetric impacts across demographic groups, where bias can worsen by up to 18.6% for some groups while improving by 14.1% for others, yielding misleadingly neutral aggregate outcomes. Larger models show no consistent robustness advantage, and group-specific shifts vary unpredictably across model families. Our findings demonstrate that compression fundamentally alters bias patterns, requiring crucial post-quantization evaluation and interventions to ensure reliability in practice.
>
---
#### [new 038] Quantifying and Attributing Polarization to Annotator Groups
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的标注分析任务，旨在解决标注者群体间极化问题。提出一种新度量方法，用于量化并归因不同标注群体的极化现象，适用于多标签和不平衡数据集。**

- **链接: [https://arxiv.org/pdf/2602.06055v1](https://arxiv.org/pdf/2602.06055v1)**

> **作者:** Dimitris Tsirmpas; John Pavlopoulos
>
> **备注:** 28 pages, 6 tables, 7 figures, 1 algorithm
>
> **摘要:** Current annotation agreement metrics are not well-suited for inter-group analysis, are sensitive to group size imbalances and restricted to single-annotation settings. These restrictions render them insufficient for many subjective tasks such as toxicity and hate-speech detection. For this reason, we introduce a quantifiable metric, paired with a statistical significance test, that attributes polarization to various annotator groups. Our metric enables direct comparisons between heavily imbalanced sociodemographic and ideological subgroups across different datasets and tasks, while also enabling analysis on multi-label settings. We apply this metric to three datasets on hate speech, and one on toxicity detection, discovering that: (1) Polarization is strongly and persistently attributed to annotator race, especially on the hate speech task. (2) Religious annotators do not fundamentally disagree with each other, but do with other annotators, a trend that is gradually diminished and then reversed with irreligious annotators. (3) Less educated annotators are more subjective, while educated ones tend to broadly agree more between themselves. Overall, our results reflect current findings around annotation patterns for various subgroups. Finally, we estimate the minimum number of annotators needed to obtain robust results, and provide an open-source Python library that implements our metric.
>
---
#### [new 039] Lost in Speech: Benchmarking, Evaluation, and Parsing of Spoken Code-Switching Beyond Standard UD Assumptions
- **分类: cs.CL**

- **简介: 该论文研究口语代码切换的句法解析任务，解决现有方法在处理口语特征时的失效问题。提出SpokeBench和FLEX-UD，改进解析框架DECAP，提升解析效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2602.06307v1](https://arxiv.org/pdf/2602.06307v1)**

> **作者:** Nemika Tyagi; Holly Hendrix; Nelvin Licona-Guevara; Justin Mackie; Phanos Kareen; Muhammad Imran; Megan Michelle Smith; Tatiana Gallego Hernande; Chitta Baral; Olga Kellert
>
> **备注:** 18 pages, 4 Figures
>
> **摘要:** Spoken code-switching (CSW) challenges syntactic parsing in ways not observed in written text. Disfluencies, repetition, ellipsis, and discourse-driven structure routinely violate standard Universal Dependencies (UD) assumptions, causing parsers and large language models (LLMs) to fail despite strong performance on written data. These failures are compounded by rigid evaluation metrics that conflate genuine structural errors with acceptable variation. In this work, we present a systems-oriented approach to spoken CSW parsing. We introduce a linguistically grounded taxonomy of spoken CSW phenomena and SpokeBench, an expert-annotated gold benchmark designed to test spoken-language structure beyond standard UD assumptions. We further propose FLEX-UD, an ambiguity-aware evaluation metric, which reveals that existing parsing techniques perform poorly on spoken CSW by penalizing linguistically plausible analyses as errors. We then propose DECAP, a decoupled agentic parsing framework that isolates spoken-phenomena handling from core syntactic analysis. Experiments show that DECAP produces more robust and interpretable parses without retraining and achieves up to 52.6% improvements over existing parsing techniques. FLEX-UD evaluations further reveal qualitative improvements that are masked by standard metrics.
>
---
#### [new 040] Judging What We Cannot Solve: A Consequence-Based Approach for Oracle-Free Evaluation of Research-Level Math
- **分类: cs.CL**

- **简介: 该论文属于数学问题求解评估任务，解决无监督评价研究级数学解法的问题。通过构建基于后果的效用评估方法，提升解法排序效果。**

- **链接: [https://arxiv.org/pdf/2602.06291v1](https://arxiv.org/pdf/2602.06291v1)**

> **作者:** Guijin Son; Donghun Yang; Hitesh Laxmichand Patel; Hyunwoo Ko; Amit Agarwal; Sunghee Ahn; Kyong-Ha Lee; Youngjae Yu
>
> **备注:** Preprint
>
> **摘要:** Recent progress in reasoning models suggests that generating plausible attempts for research-level mathematics may be within reach, but verification remains a bottleneck, consuming scarce expert time. We hypothesize that a meaningful solution should contain enough method-level information that, when applied to a neighborhood of related questions, it should yield better downstream performance than incorrect solutions. Building on this idea, we propose \textbf{Consequence-Based Utility}, an oracle-free evaluator that scores each candidate by testing its value as an in-context exemplar in solving related yet verifiable questions. Our approach is evaluated on an original set of research-level math problems, each paired with one expert-written solution and nine LLM-generated solutions. Notably, Consequence-Based Utility consistently outperforms reward models, generative reward models, and LLM judges on ranking quality. Specifically, for GPT-OSS-120B, it improves Acc@1 from 67.2 to 76.3 and AUC from 71.4 to 79.6, with similarly large AUC gains on GPT-OSS-20B (69.0 to 79.2). Furthermore, compared to LLM-Judges, it also exhibits a larger solver-evaluator gap, maintaining a stronger correct-wrong separation even on instances where the underlying solver often fails to solve.
>
---
#### [new 041] Improve Large Language Model Systems with User Logs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM系统因数据稀缺和噪声而效果受限的问题。通过分析用户日志，提出UNO框架提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.06470v1](https://arxiv.org/pdf/2602.06470v1)**

> **作者:** Changyue Wang; Weihang Su; Qingyao Ai; Yiqun Liu
>
> **摘要:** Scaling training data and model parameters has long driven progress in large language models (LLMs), but this paradigm is increasingly constrained by the scarcity of high-quality data and diminishing returns from rising computational costs. As a result, recent work is increasing the focus on continual learning from real-world deployment, where user interaction logs provide a rich source of authentic human feedback and procedural knowledge. However, learning from user logs is challenging due to their unstructured and noisy nature. Vanilla LLM systems often struggle to distinguish useful feedback signals from noisy user behavior, and the disparity between user log collection and model optimization (e.g., the off-policy optimization problem) further strengthens the problem. To this end, we propose UNO (User log-driveN Optimization), a unified framework for improving LLM systems (LLMsys) with user logs. UNO first distills logs into semi-structured rules and preference pairs, then employs query-and-feedback-driven clustering to manage data heterogeneity, and finally quantifies the cognitive gap between the model's prior knowledge and the log data. This assessment guides the LLMsys to adaptively filter out noisy feedback and construct different modules for primary and reflective experiences extracted from user logs, thereby improving future responses. Extensive experiments show that UNO achieves state-of-the-art effectiveness and efficiency, significantly outperforming Retrieval Augmented Generation (RAG) and memory-based baselines. We have open-sourced our code at https://github.com/bebr2/UNO .
>
---
#### [new 042] DAWN: Dependency-Aware Fast Inference for Diffusion LLMs
- **分类: cs.CL**

- **简介: 该论文提出DAWN方法，解决扩散大语言模型推理效率与质量的平衡问题。通过依赖感知的解码策略，提升并行性同时保持生成质量。属于自然语言生成任务。**

- **链接: [https://arxiv.org/pdf/2602.06953v1](https://arxiv.org/pdf/2602.06953v1)**

> **作者:** Lizhuo Luo; Zhuoran Shi; Jiajun Luo; Zhi Wang; Shen Ren; Wenya Wang; Tianwei Zhang
>
> **摘要:** Diffusion large language models (dLLMs) have shown advantages in text generation, particularly due to their inherent ability for parallel decoding. However, constrained by the quality--speed trade-off, existing inference solutions adopt conservative parallel strategies, leaving substantial efficiency potential underexplored. A core challenge is that parallel decoding assumes each position can be filled independently, but tokens are often semantically coupled. Thus, the correct choice at one position constrains valid choices at others. Without modeling these inter-token dependencies, parallel strategies produce deteriorated outputs. Motivated by this insight, we propose DAWN, a training-free, dependency-aware decoding method for fast dLLM inference. DAWN extracts token dependencies and leverages two key motivations: (1) positions dependent on unmasked certain positions become more reliable, (2) simultaneously unmasking strongly coupled uncertain positions induces errors. Given those findings, DAWN leverages a dependency graph to select more reliable unmasking positions at each iteration, achieving high parallelism with negligible loss in generation quality. Extensive experiments across multiple models and datasets demonstrate that DAWN speedups the inference by 1.80-8.06x over baselines while preserving the generation quality. Code is released at https://github.com/lizhuo-luo/DAWN.
>
---
#### [new 043] Echoes as Anchors: Probabilistic Costs and Attention Refocusing in LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文研究大语言模型推理中的计算分配问题，通过分析模型自发重复提示的“回声”现象，提出新方法提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2602.06600v1](https://arxiv.org/pdf/2602.06600v1)**

> **作者:** Zhuoyuan Hao; Zhuo Li; Wu Li; Fangming Liu; Min Zhang; Jing Li
>
> **摘要:** Test-time compute allocation in large reasoning models (LRMs) is widely used and has applications in mathematical problem solving, code synthesis, and planning. Recent work has addressed this problem by scaling self-consistency and parallel thinking, adding generic ``thinking tokens'' and prompting models to re-read the question before answering. Unfortunately, these approaches either inject task-agnostic tokens or mandate heuristics that do not explain -- and often ignore -- the \emph{spontaneous} repetition that many LRMs exhibit at the head of their internal chains. In contrast, we analyze and harness the model's tendency to restate the question, which we term the \emph{Echo of Prompt (EOP)}, as a front-loaded, compute-shaping mechanism. We formalize its probabilistic cost by casting echo removal as rejection-based conditioning and defining the \emph{Echo Likelihood Gap} $Δ\mathcal{L}$ as a computable proxy. This provides the missing theoretical link that links early repetition to likelihood gains and downstream accuracy. However, it does not by itself specify how to exploit EOP. Consequently, we develop \emph{Echo-Distilled SFT (ED-SFT)} to instill an ``echo-then-reason'' pattern through supervised finetuning, and \emph{Echoic Prompting (EP)} to re-ground the model mid-trace without training. While promising, quantifying benefits beyond verbosity is non-trivial. Therefore, we conduct length and suffix-controlled likelihood analyses together with layer-wise attention studies, showing that EOP increases answer to answer-prefix attention in middle layers, consistent with an \emph{attention refocusing} mechanism. We evaluate on GSM8K, MathQA, Hendrycks-MATH, AIME24, and MATH-500 under identical decoding settings and budgets, and find consistent gains over baselines. Code is available at https://github.com/hhh2210/echoes-as-anchors.
>
---
#### [new 044] Not All Layers Need Tuning: Selective Layer Restoration Recovers Diversity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，解决LLM生成多样性不足的问题。通过选择性恢复预训练层，提升输出多样性同时保持质量。**

- **链接: [https://arxiv.org/pdf/2602.06665v1](https://arxiv.org/pdf/2602.06665v1)**

> **作者:** Bowen Zhang; Meiyi Wang; Harold Soh
>
> **备注:** 16 pages, 7 figures, 12 tables
>
> **摘要:** Post-training improves instruction-following and helpfulness of large language models (LLMs) but often reduces generation diversity, which leads to repetitive outputs in open-ended settings, a phenomenon known as mode collapse. Motivated by evidence that LLM layers play distinct functional roles, we hypothesize that mode collapse can be localized to specific layers and that restoring a carefully chosen range of layers to their pre-trained weights can recover diversity while maintaining high output quality. To validate this hypothesis and decide which layers to restore, we design a proxy task -- Constrained Random Character(CRC) -- with an explicit validity set and a natural diversity objective. Results on CRC reveal a clear diversity-validity trade-off across restoration ranges and identify configurations that increase diversity with minimal quality loss. Based on these findings, we propose Selective Layer Restoration (SLR), a training-free method that restores selected layers in a post-trained model to their pre-trained weights, yielding a hybrid model with the same architecture and parameter count, incurring no additional inference cost. Across three different tasks (creative writing, open-ended question answering, and multi-step reasoning) and three different model families (Llama, Qwen, and Gemma), we find SLR can consistently and substantially improve output diversity while maintaining high output quality.
>
---
#### [new 045] Do Prompts Guarantee Safety? Mitigating Toxicity from LLM Generations through Subspace Intervention
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于语言模型安全任务，旨在解决LLM生成有毒内容的问题。通过子空间干预方法，在保持文本流畅性的同时有效降低毒性。**

- **链接: [https://arxiv.org/pdf/2602.06623v1](https://arxiv.org/pdf/2602.06623v1)**

> **作者:** Himanshu Singh; Ziwei Xu; A. V. Subramanyam; Mohan Kankanhalli
>
> **摘要:** Large Language Models (LLMs) are powerful text generators, yet they can produce toxic or harmful content even when given seemingly harmless prompts. This presents a serious safety challenge and can cause real-world harm. Toxicity is often subtle and context-dependent, making it difficult to detect at the token level or through coarse sentence-level signals. Moreover, efforts to mitigate toxicity often face a trade-off between safety and the coherence, or fluency of the generated text. In this work, we present a targeted subspace intervention strategy for identifying and suppressing hidden toxic patterns from underlying model representations, while preserving overall ability to generate safe fluent content. On the RealToxicityPrompts, our method achieves strong mitigation performance compared to existing baselines, with minimal impact on inference complexity. Across multiple LLMs, our approach reduces toxicity of state-of-the-art detoxification systems by 8-20%, while maintaining comparable fluency. Through extensive quantitative and qualitative analyses, we show that our approach achieves effective toxicity reduction without impairing generative performance, consistently outperforming existing baselines.
>
---
#### [new 046] Generating Data-Driven Reasoning Rubrics for Domain-Adaptive Reward Modeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，旨在解决LLM在推理验证中的误差检测问题。通过构建数据驱动的推理评分体系，提升模型在复杂领域的准确性。**

- **链接: [https://arxiv.org/pdf/2602.06795v1](https://arxiv.org/pdf/2602.06795v1)**

> **作者:** Kate Sanders; Nathaniel Weir; Sapana Chaudhary; Kaj Bostrom; Huzefa Rangwala
>
> **摘要:** An impediment to using Large Language Models (LLMs) for reasoning output verification is that LLMs struggle to reliably identify errors in thinking traces, particularly in long outputs, domains requiring expert knowledge, and problems without verifiable rewards. We propose a data-driven approach to automatically construct highly granular reasoning error taxonomies to enhance LLM-driven error detection on unseen reasoning traces. Our findings indicate that classification approaches that leverage these error taxonomies, or "rubrics", demonstrate strong error identification compared to baseline methods in technical domains like coding, math, and chemical engineering. These rubrics can be used to build stronger LLM-as-judge reward functions for reasoning model training via reinforcement learning. Experimental results show that these rewards have the potential to improve models' task accuracy on difficult domains over models trained by general LLMs-as-judges by +45%, and approach performance of models trained by verifiable rewards while using as little as 20% as many gold labels. Through our approach, we extend the usage of reward rubrics from assessing qualitative model behavior to assessing quantitative model correctness on tasks typically learned via RLVR rewards. This extension opens the door for teaching models to solve complex technical problems without a full dataset of gold labels, which are often highly costly to procure.
>
---
#### [new 047] Baichuan-M3: Modeling Clinical Inquiry for Reliable Medical Decision-Making
- **分类: cs.CL**

- **简介: 该论文提出Baichuan-M3，用于医疗决策支持，解决传统问答系统在临床咨询中的局限性。通过主动信息获取、长程推理和幻觉抑制，提升诊断准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2602.06570v1](https://arxiv.org/pdf/2602.06570v1)**

> **作者:** Baichuan-M3 Team; :; Chengfeng Dou; Fan Yang; Fei Li; Jiyuan Jia; Qiang Ju; Shuai Wang; Tianpeng Li; Xiangrong Zeng; Yijie Zhou; Hongda Zhang; Jinyang Tai; Linzhuang Sun; Peidong Guo; Yichuan Mo; Xiaochuan Wang; Hengfu Cui; Zhishou Zhang
>
> **摘要:** We introduce Baichuan-M3, a medical-enhanced large language model engineered to shift the paradigm from passive question-answering to active, clinical-grade decision support. Addressing the limitations of existing systems in open-ended consultations, Baichuan-M3 utilizes a specialized training pipeline to model the systematic workflow of a physician. Key capabilities include: (i) proactive information acquisition to resolve ambiguity; (ii) long-horizon reasoning that unifies scattered evidence into coherent diagnoses; and (iii) adaptive hallucination suppression to ensure factual reliability. Empirical evaluations demonstrate that Baichuan-M3 achieves state-of-the-art results on HealthBench, the newly introduced HealthBench-Hallu and ScanBench, significantly outperforming GPT-5.2 in clinical inquiry, advisory and safety. The models are publicly available at https://huggingface.co/collections/baichuan-inc/baichuan-m3.
>
---
#### [new 048] Investigating the structure of emotions by analyzing similarity and association of emotion words
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感分析任务，旨在验证普拉奇克情绪轮的有效性。通过构建情感词的语义网络，分析其结构与情绪轮的相似性。**

- **链接: [https://arxiv.org/pdf/2602.06430v1](https://arxiv.org/pdf/2602.06430v1)**

> **作者:** Fumitaka Iwaki; Tatsuji Takahashi
>
> **备注:** 5 figures, 8 tables
>
> **摘要:** In the field of natural language processing, some studies have attempted sentiment analysis on text by handling emotions as explanatory or response variables. One of the most popular emotion models used in this context is the wheel of emotion proposed by Plutchik. This model schematizes human emotions in a circular structure, and represents them in two or three dimensions. However, the validity of Plutchik's wheel of emotion has not been sufficiently examined. This study investigated the validity of the wheel by creating and analyzing a semantic networks of emotion words. Through our experiments, we collected data of similarity and association of ordered pairs of emotion words, and constructed networks using these data. We then analyzed the structure of the networks through community detection, and compared it with that of the wheel of emotion. The results showed that each network's structure was, for the most part, similar to that of the wheel of emotion, but locally different.
>
---
#### [new 049] RelayGen: Intra-Generation Model Switching for Efficient Reasoning
- **分类: cs.CL**

- **简介: 该论文提出RelayGen，用于高效推理的生成模型切换。解决大模型推理效率低的问题，通过段级难度判断动态切换模型，提升速度并保持精度。**

- **链接: [https://arxiv.org/pdf/2602.06454v1](https://arxiv.org/pdf/2602.06454v1)**

> **作者:** Jiwon Song; Yoongon Kim; Jae-Joon Kim
>
> **摘要:** Large reasoning models (LRMs) achieve strong performance on complex reasoning tasks by generating long, multi-step reasoning trajectories, but inference-time scaling incurs substantial deployment cost. A key challenge is that generation difficulty varies within a single output, whereas existing efficiency-oriented approaches either ignore this intra-generation variation or rely on supervised token-level routing with high system complexity. We present \textbf{RelayGen}, a training-free, segment-level runtime model switching framework that exploits difficulty variation in long-form reasoning. Through offline analysis of generation uncertainty using token probability margins, we show that coarse-grained segment-level control is sufficient to capture difficulty transitions within a reasoning trajectory. RelayGen identifies model-specific switch cues that signal transitions to lower-difficulty segments and dynamically delegates their continuation to a smaller model, while preserving high-difficulty reasoning on the large model. Across multiple reasoning benchmarks, RelayGen substantially reduces inference latency while preserving most of the accuracy of large models. When combined with speculative decoding, RelayGen achieves up to 2.2$\times$ end-to-end speedup with less than 2\% accuracy degradation, without requiring additional training or learned routing components.
>
---
#### [new 050] Can Post-Training Transform LLMs into Causal Reasoners?
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于因果推理任务，旨在解决LLMs在因果推断上的能力不足问题。通过引入CauGym数据集和多种后训练方法，提升模型的因果推理能力。**

- **链接: [https://arxiv.org/pdf/2602.06337v1](https://arxiv.org/pdf/2602.06337v1)**

> **作者:** Junqi Chen; Sirui Chen; Chaochao Lu
>
> **摘要:** Causal inference is essential for decision-making but remains challenging for non-experts. While large language models (LLMs) show promise in this domain, their precise causal estimation capabilities are still limited, and the impact of post-training on these abilities is insufficiently explored. This paper examines the extent to which post-training can enhance LLMs' capacity for causal inference. We introduce CauGym, a comprehensive dataset comprising seven core causal tasks for training and five diverse test sets. Using this dataset, we systematically evaluate five post-training approaches: SFT, DPO, KTO, PPO, and GRPO. Across five in-domain and four existing benchmarks, our experiments demonstrate that appropriate post-training enables smaller LLMs to perform causal inference competitively, often surpassing much larger models. Our 14B parameter model achieves 93.5% accuracy on the CaLM benchmark, compared to 55.4% by OpenAI o3. Furthermore, the post-trained LLMs exhibit strong generalization and robustness under real-world conditions such as distribution shifts and noisy data. Collectively, these findings provide the first systematic evidence that targeted post-training can produce reliable and robust LLM-based causal reasoners. Our data and GRPO-model are available at https://github.com/OpenCausaLab/CauGym.
>
---
#### [new 051] CAST: Character-and-Scene Episodic Memory for Agents
- **分类: cs.CL**

- **简介: 该论文属于记忆建模任务，旨在解决智能体难以表征和检索连贯事件的问题。提出CAST架构，结合角色与场景构建情景记忆，并融合图式语义记忆，提升对话理解性能。**

- **链接: [https://arxiv.org/pdf/2602.06051v1](https://arxiv.org/pdf/2602.06051v1)**

> **作者:** Kexin Ma; Bojun Li; Yuhua Tang; Ruochun Jin; Liting Sun
>
> **摘要:** Episodic memory is a central component of human memory, which refers to the ability to recall coherent events grounded in who, when, and where. However, most agent memory systems only emphasize semantic recall and treat experience as structures such as key-value, vector, or graph, which makes them struggle to represent and retrieve coherent events. To address this challenge, we propose a Character-and-Scene based memory architecture(CAST) inspired by dramatic theory. Specifically, CAST constructs 3D scenes (time/place/topic) and organizes them into character profiles that summarize the events of a character to represent episodic memory. Moreover, CAST complements this episodic memory with a graph-based semantic memory, which yields a robust dual memory design. Experiments demonstrate that CAST has averagely improved 8.11% F1 and 10.21% J(LLM-as-a-Judge) than baselines on various datasets, especially on open and time-sensitive conversational questions.
>
---
#### [new 052] R-Align: Enhancing Generative Reward Models through Rationale-Centric Meta-Judging
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决生成奖励模型中推理质量评估不足的问题。通过引入R-Align方法提升模型推理一致性，改善下游RLHF效果。**

- **链接: [https://arxiv.org/pdf/2602.06763v1](https://arxiv.org/pdf/2602.06763v1)**

> **作者:** Yanlin Lai; Mitt Huang; Hangyu Guo; Xiangfeng Wang; Haodong Li; Shaoxiong Zhan; Liang Zhao; Chengyuan Yao; Yinmin Zhang; Qi Han; Chun Yuan; Zheng Ge; Xiangyu Zhang; Daxin Jiang
>
> **备注:** Github: https://github.com/lyn22333/R-Align Huggingface: https://huggingface.co/collections/lyn22333/r-align
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) remains indispensable for aligning large language models (LLMs) in subjective domains. To enhance robustness, recent work shifts toward Generative Reward Models (GenRMs) that generate rationales before predicting preferences. Yet in GenRM training and evaluation, practice remains outcome-label-only, leaving reasoning quality unchecked. We show that reasoning fidelity-the consistency between a GenRM's preference decision and reference decision rationales-is highly predictive of downstream RLHF outcomes, beyond standard label accuracy. Specifically, we repurpose existing reward-model benchmarks to compute Spurious Correctness (S-Corr)-the fraction of label-correct decisions with rationales misaligned with golden judgments. Our empirical evaluation reveals substantial S-Corr even for competitive GenRMs, and higher S-Corr is associated with policy degeneration under optimization. To improve fidelity, we propose Rationale-Centric Alignment, R-Align, which augments training with gold judgments and explicitly supervises rationale alignment. R-Align reduces S-Corr on RM benchmarks and yields consistent gains in actor performance across STEM, coding, instruction following, and general tasks.
>
---
#### [new 053] ReBeCA: Unveiling Interpretable Behavior Hierarchy behind the Iterative Self-Reflection of Language Models with Causal Analysis
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在揭示语言模型自我反思的因果机制。通过构建因果图和ICP管道，识别影响自我反思效果的关键行为因素，解决现有分析依赖相关性、泛化能力差的问题。**

- **链接: [https://arxiv.org/pdf/2602.06373v1](https://arxiv.org/pdf/2602.06373v1)**

> **作者:** Tianqiang Yan; Sihan Shang; Yuheng Li; Song Qiu; Hao Peng; Wenjian Luo; Jue Xie; Lizhen Qu; Yuan Gao
>
> **备注:** 17 pages, 3 figures
>
> **摘要:** While self-reflection can enhance language model reliability, its underlying mechanisms remain opaque, with existing analyses often yielding correlation-based insights that fail to generalize. To address this, we introduce \textbf{\texttt{ReBeCA}} (self-\textbf{\texttt{Re}}flection \textbf{\texttt{Be}}havior explained through \textbf{\texttt{C}}ausal \textbf{\texttt{A}}nalysis), a framework that unveils the interpretable behavioral hierarchy governing the self-reflection outcome. By modeling self-reflection trajectories as causal graphs, ReBeCA isolates genuine determinants of performance through a three-stage Invariant Causal Prediction (ICP) pipeline. We establish three critical findings: (1) \textbf{Behavioral hierarchy:} Semantic behaviors of the model influence final self-reflection results hierarchically: directly or indirectly; (2) \textbf{Causation matters:} Generalizability in self-reflection effects is limited to just a few semantic behaviors; (3) \textbf{More $\mathbf{\neq}$ better:} The confluence of seemingly positive semantic behaviors, even among direct causal factors, can impair the efficacy of self-reflection. ICP-based verification identifies sparse causal parents achieving up to $49.6\%$ structural likelihood gains, stable across tasks where correlation-based patterns fail. Intervention studies on novel datasets confirm these causal relationships hold out-of-distribution ($p = .013, η^2_\mathrm{p} = .071$). ReBeCA thus provides a rigorous methodology for disentangling genuine causal mechanisms from spurious associations in self-reflection dynamics.
>
---
#### [new 054] Reading Between the Waves: Robust Topic Segmentation Using Inter-Sentence Audio Features
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于话题分割任务，旨在提升语音内容的自动分段效果。通过融合文本与音频特征，提出一种多模态方法，增强对ASR噪声的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.06647v1](https://arxiv.org/pdf/2602.06647v1)**

> **作者:** Steffen Freisinger; Philipp Seeberger; Tobias Bocklet; Korbinian Riedhammer
>
> **备注:** Accepted to IEEE ICASSP 2026
>
> **摘要:** Spoken content, such as online videos and podcasts, often spans multiple topics, which makes automatic topic segmentation essential for user navigation and downstream applications. However, current methods do not fully leverage acoustic features, leaving room for improvement. We propose a multi-modal approach that fine-tunes both a text encoder and a Siamese audio encoder, capturing acoustic cues around sentence boundaries. Experiments on a large-scale dataset of YouTube videos show substantial gains over text-only and multi-modal baselines. Our model also proves more resilient to ASR noise and outperforms a larger text-only baseline on three additional datasets in Portuguese, German, and English, underscoring the value of learned acoustic features for robust topic segmentation.
>
---
#### [new 055] Stop the Flip-Flop: Context-Preserving Verification for Fast Revocable Diffusion Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的文本生成任务，解决并行扩散解码中因验证引发的翻转振荡问题。提出COVER方法，在单次前向传播中实现稳定解码，提升效率并保持质量。**

- **链接: [https://arxiv.org/pdf/2602.06161v1](https://arxiv.org/pdf/2602.06161v1)**

> **作者:** Yanzheng Xiang; Lan Wei; Yizhen Yao; Qinglin Zhu; Hanqi Yan; Chen Jin; Philip Alexander Teare; Dandan Zhang; Lin Gui; Amrutha Saseendran; Yulan He
>
> **摘要:** Parallel diffusion decoding can accelerate diffusion language model inference by unmasking multiple tokens per step, but aggressive parallelism often harms quality. Revocable decoding mitigates this by rechecking earlier tokens, yet we observe that existing verification schemes frequently trigger flip-flop oscillations, where tokens are remasked and later restored unchanged. This behaviour slows inference in two ways: remasking verified positions weakens the conditioning context for parallel drafting, and repeated remask cycles consume the revision budget with little net progress. We propose COVER (Cache Override Verification for Efficient Revision), which performs leave-one-out verification and stable drafting within a single forward pass. COVER constructs two attention views via KV cache override: selected seeds are masked for verification, while their cached key value states are injected for all other queries to preserve contextual information, with a closed form diagonal correction preventing self leakage at the seed positions. COVER further prioritises seeds using a stability aware score that balances uncertainty, downstream influence, and cache drift, and it adapts the number of verified seeds per step. Across benchmarks, COVER markedly reduces unnecessary revisions and yields faster decoding while preserving output quality.
>
---
#### [new 056] PhenoLIP: Integrating Phenotype Ontology Knowledge into Medical Vision-Language Pretraining
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医疗视觉-语言预训练任务，旨在解决现有模型未能有效利用医学表型本体知识的问题。通过构建PhenoKG和提出PhenoLIP框架，提升医学图像理解的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2602.06184v1](https://arxiv.org/pdf/2602.06184v1)**

> **作者:** Cheng Liang; Chaoyi Wu; Weike Zhao; Ya Zhang; Yanfeng Wang; Weidi Xie
>
> **摘要:** Recent progress in large-scale CLIP-like vision-language models(VLMs) has greatly advanced medical image analysis. However, most existing medical VLMs still rely on coarse image-text contrastive objectives and fail to capture the systematic visual knowledge encoded in well-defined medical phenotype ontologies. To address this gap, we construct PhenoKG, the first large-scale, phenotype-centric multimodal knowledge graph that encompasses over 520K high-quality image-text pairs linked to more than 3,000 phenotypes. Building upon PhenoKG, we propose PhenoLIP, a novel pretraining framework that explicitly incorporates structured phenotype knowledge into medical VLMs through a two-stage process. We first learn a knowledge-enhanced phenotype embedding space from textual ontology data and then distill this structured knowledge into multimodal pretraining via a teacher-guided knowledge distillation objective. To support evaluation, we further introduce PhenoBench, an expert-verified benchmark designed for phenotype recognition, comprising over 7,800 image--caption pairs covering more than 1,000 phenotypes. Extensive experiments demonstrate that PhenoLIP outperforms previous state-of-the-art baselines, improving upon BiomedCLIP in phenotype classification accuracy by 8.85\% and BIOMEDICA in cross-modal retrieval by 15.03%, underscoring the value of integrating phenotype-centric priors into medical VLMs for structured and interpretable medical image understanding.
>
---
#### [new 057] Learning Rate Scaling across LoRA Ranks and Transfer to Full Finetuning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型微调任务，解决LoRA中学习率与适配器秩的关联问题，提出μA框架，揭示学习率的 scaling 规律，并实现从LoRA到全微调的学习率迁移。**

- **链接: [https://arxiv.org/pdf/2602.06204v1](https://arxiv.org/pdf/2602.06204v1)**

> **作者:** Nan Chen; Soledad Villar; Soufiane Hayou
>
> **摘要:** Low-Rank Adaptation (LoRA) is a standard tool for parameter-efficient finetuning of large models. While it induces a small memory footprint, its training dynamics can be surprisingly complex as they depend on several hyperparameters such as initialization, adapter rank, and learning rate. In particular, it is unclear how the optimal learning rate scales with adapter rank, which forces practitioners to re-tune the learning rate whenever the rank is changed. In this paper, we introduce Maximal-Update Adaptation ($μ$A), a theoretical framework that characterizes how the "optimal" learning rate should scale with model width and adapter rank to produce stable, non-vanishing feature updates under standard configurations. $μ$A is inspired from the Maximal-Update Parametrization ($μ$P) in pretraining. Our analysis leverages techniques from hyperparameter transfer and reveals that the optimal learning rate exhibits different scaling patterns depending on initialization and LoRA scaling factor. Specifically, we identify two regimes: one where the optimal learning rate remains roughly invariant across ranks, and another where it scales inversely with rank. We further identify a configuration that allows learning rate transfer from LoRA to full finetuning, drastically reducing the cost of learning rate tuning for full finetuning. Experiments across language, vision, vision--language, image generation, and reinforcement learning tasks validate our scaling rules and show that learning rates tuned on LoRA transfer reliably to full finetuning.
>
---
#### [new 058] Learning a Generative Meta-Model of LLM Activations
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于神经网络可解释性任务，旨在解决传统方法依赖结构假设的问题。通过训练扩散模型学习激活分布，构建生成元模型，提升干预效果与概念分离度。**

- **链接: [https://arxiv.org/pdf/2602.06964v1](https://arxiv.org/pdf/2602.06964v1)**

> **作者:** Grace Luo; Jiahai Feng; Trevor Darrell; Alec Radford; Jacob Steinhardt
>
> **摘要:** Existing approaches for analyzing neural network activations, such as PCA and sparse autoencoders, rely on strong structural assumptions. Generative models offer an alternative: they can uncover structure without such assumptions and act as priors that improve intervention fidelity. We explore this direction by training diffusion models on one billion residual stream activations, creating "meta-models" that learn the distribution of a network's internal states. We find that diffusion loss decreases smoothly with compute and reliably predicts downstream utility. In particular, applying the meta-model's learned prior to steering interventions improves fluency, with larger gains as loss decreases. Moreover, the meta-model's neurons increasingly isolate concepts into individual units, with sparse probing scores that scale as loss decreases. These results suggest generative meta-models offer a scalable path toward interpretability without restrictive structural assumptions. Project page: https://generative-latent-prior.github.io.
>
---
#### [new 059] LogicSkills: A Structured Benchmark for Formal Reasoning in Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出LogicSkills基准，用于评估大语言模型在形式推理中的核心技能。任务是检测模型是否掌握符号化、构造反例和有效性判断，解决模型依赖表面模式而非真正逻辑推理的问题。**

- **链接: [https://arxiv.org/pdf/2602.06533v1](https://arxiv.org/pdf/2602.06533v1)**

> **作者:** Brian Rabern; Philipp Mondorf; Barbara Plank
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Large language models have demonstrated notable performance across various logical reasoning benchmarks. However, it remains unclear which core logical skills they truly master. To address this, we introduce LogicSkills, a unified benchmark designed to isolate three fundamental skills in formal reasoning: (i) $\textit{formal symbolization}\unicode{x2014}$translating premises into first-order logic; (ii) $\textit{countermodel construction}\unicode{x2014}$formulating a finite structure in which all premises are true while the conclusion is false; and (iii) $\textit{validity assessment}\unicode{x2014}$deciding whether a conclusion follows from a given set of premises. Items are drawn from the two-variable fragment of first-order logic (without identity) and are presented in both natural English and a Carroll-style language with nonce words. All examples are verified for correctness and non-triviality using the SMT solver Z3. Across leading models, performance is high on validity but substantially lower on symbolization and countermodel construction, suggesting reliance on surface-level patterns rather than genuine symbolic or rule-based reasoning.
>
---
#### [new 060] Protean Compiler: An Agile Framework to Drive Fine-grain Phase Ordering
- **分类: cs.PL; cs.AI; cs.CL; cs.LG; cs.PF**

- **简介: 该论文提出Protean Compiler，解决编译器优化阶段顺序问题。通过集成机器学习，实现细粒度优化，提升性能。**

- **链接: [https://arxiv.org/pdf/2602.06142v1](https://arxiv.org/pdf/2602.06142v1)**

> **作者:** Amir H. Ashouri; Shayan Shirahmad Gale Bagi; Kavin Satheeskumar; Tejas Srikanth; Jonathan Zhao; Ibrahim Saidoun; Ziwen Wang; Bryan Chan; Tomasz S. Czajkowski
>
> **备注:** Version 1- Submitted for a possible publication in 2026
>
> **摘要:** The phase ordering problem has been a long-standing challenge since the late 1970s, yet it remains an open problem due to having a vast optimization space and an unbounded nature, making it an open-ended problem without a finite solution, one can limit the scope by reducing the number and the length of optimizations. Traditionally, such locally optimized decisions are made by hand-coded algorithms tuned for a small number of benchmarks, often requiring significant effort to be retuned when the benchmark suite changes. In the past 20 years, Machine Learning has been employed to construct performance models to improve the selection and ordering of compiler optimizations, however, the approaches are not baked into the compiler seamlessly and never materialized to be leveraged at a fine-grained scope of code segments. This paper presents Protean Compiler: An agile framework to enable LLVM with built-in phase-ordering capabilities at a fine-grained scope. The framework also comprises a complete library of more than 140 handcrafted static feature collection methods at varying scopes, and the experimental results showcase speedup gains of up to 4.1% on average and up to 15.7% on select Cbench applications wrt LLVM's O3 by just incurring a few extra seconds of build time on Cbench. Additionally, Protean compiler allows for an easy integration with third-party ML frameworks and other Large Language Models, and this two-step optimization shows a gain of 10.1% and 8.5% speedup wrt O3 on Cbench's Susan and Jpeg applications. Protean compiler is seamlessly integrated into LLVM and can be used as a new, enhanced, full-fledged compiler. We plan to release the project to the open-source community in the near future.
>
---
#### [new 061] Analyzing Diffusion and Autoregressive Vision Language Models in Multimodal Embedding Space
- **分类: cs.MM; cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究多模态扩散模型作为嵌入模型的性能，对比其与自回归模型在分类、视觉问答和检索任务中的表现，发现扩散模型在图像-文本对齐上存在不足。**

- **链接: [https://arxiv.org/pdf/2602.06056v1](https://arxiv.org/pdf/2602.06056v1)**

> **作者:** Zihang Wang; Siyue Zhang; Yilun Zhao; Jingyi Yang; Tingyu Song; Anh Tuan Luu; Chen Zhao
>
> **摘要:** Embedding models are a fundamental component of modern AI systems such as semantic search and retrieval-augmented generation. Recent advances in large foundation models have substantially accelerated the development of embedding models, including those based on Large Language Models (LLMs), Vision Language Models (VLMs), and Multimodal LLMs. More recently, Large Diffusion Language Models (dLLMs) and Multimodal dLLMs have emerged as competitive alternatives to autoregressive models, offering advantages such as bidirectional attention and parallel generation. This progress naturally raises a critical yet unexplored question: can Multimodal dLLMs serve as effective multimodal embedding models? To answer this, we present the first systematic study of converting Multimodal dLLMs into embedding models. We evaluate state-of-the-art Multimodal dLLMs and Autoregressive VLMs across three categories of embedding tasks: classification, visual question answering, and information retrieval. Our results show that Multimodal dLLM embeddings generally underperform their autoregressive VLM counterparts. The stronger diffusion-based model, LaViDa, lags by only 3.5 points on classification, 2.5 points on VQA, and 4.4 points on retrieval tasks, whereas the other diffusion-based model, MMaDA, exhibits substantially larger performance gaps, exceeding 20 points across all tasks. Further analysis reveals insufficient image-text alignment in diffusion-based models, accounting for the observed limitations in their embedding performance.
>
---
#### [new 062] Generics in science communication: Misaligned interpretations across laypeople, scientists, and large language models
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于科学传播任务，研究科学家、公众和大语言模型对通用表述的不同理解，揭示沟通中的误解风险，并提出语言选择的重要性。**

- **链接: [https://arxiv.org/pdf/2602.06190v1](https://arxiv.org/pdf/2602.06190v1)**

> **作者:** Uwe Peters; Andrea Bertazzoli; Jasmine M. DeJesus; Gisela J. van der Velden; Benjamin Chin-Yee
>
> **摘要:** Scientists often use generics, that is, unquantified statements about whole categories of people or phenomena, when communicating research findings (e.g., "statins reduce cardiovascular events"). Large language models (LLMs), such as ChatGPT, frequently adopt the same style when summarizing scientific texts. However, generics can prompt overgeneralizations, especially when they are interpreted differently across audiences. In a study comparing laypeople, scientists, and two leading LLMs (ChatGPT-5 and DeepSeek), we found systematic differences in interpretation of generics. Compared to most scientists, laypeople judged scientific generics as more generalizable and credible, while LLMs rated them even higher. These mismatches highlight significant risks for science communication. Scientists may use generics and incorrectly assume laypeople share their interpretation, while LLMs may systematically overgeneralize scientific findings when summarizing research. Our findings underscore the need for greater attention to language choices in both human and LLM-mediated science communication.
>
---
#### [new 063] Endogenous Resistance to Activation Steering in Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究语言模型对激活引导的内在抵抗机制，旨在理解并控制这种抵抗以提升AI系统的透明性和可控性。**

- **链接: [https://arxiv.org/pdf/2602.06941v1](https://arxiv.org/pdf/2602.06941v1)**

> **作者:** Alex McKenzie; Keenan Pepper; Stijn Servaes; Martin Leitgab; Murat Cubuktepe; Mike Vaiana; Diogo de Lucena; Judd Rosenblatt; Michael S. A. Graziano
>
> **摘要:** Large language models can resist task-misaligned activation steering during inference, sometimes recovering mid-generation to produce improved responses even when steering remains active. We term this Endogenous Steering Resistance (ESR). Using sparse autoencoder (SAE) latents to steer model activations, we find that Llama-3.3-70B shows substantial ESR, while smaller models from the Llama-3 and Gemma-2 families exhibit the phenomenon less frequently. We identify 26 SAE latents that activate differentially during off-topic content and are causally linked to ESR in Llama-3.3-70B. Zero-ablating these latents reduces the multi-attempt rate by 25%, providing causal evidence for dedicated internal consistency-checking circuits. We demonstrate that ESR can be deliberately enhanced through both prompting and training: meta-prompts instructing the model to self-monitor increase the multi-attempt rate by 4x for Llama-3.3-70B, and fine-tuning on self-correction examples successfully induces ESR-like behavior in smaller models. These findings have dual implications: ESR could protect against adversarial manipulation but might also interfere with beneficial safety interventions that rely on activation steering. Understanding and controlling these resistance mechanisms is important for developing transparent and controllable AI systems. Code is available at github.com/agencyenterprise/endogenous-steering-resistance.
>
---
#### [new 064] STACodec: Semantic Token Assignment for Balancing Acoustic Fidelity and Semantic Information in Audio Codecs
- **分类: eess.AS; cs.CL**

- **简介: 该论文提出STACodec，解决音频编码中平衡声学保真与语义信息的问题，通过语义令牌分配和预蒸馏模块提升性能。**

- **链接: [https://arxiv.org/pdf/2602.06180v1](https://arxiv.org/pdf/2602.06180v1)**

> **作者:** Kaiyuan Zhang; Mohan Shi; Eray Eren; Natarajan Balaji Shankar; Zilai Wang; Abeer Alwan
>
> **备注:** ICASSP 2026
>
> **摘要:** Neural audio codecs are widely used for audio compression and can be integrated into token-based language models. Traditional codecs preserve acoustic details well but lack semantic information. Recent hybrid codecs attempt to incorporate semantic information through distillation, but this often degrades reconstruction performance, making it difficult to achieve both. To address this limitation, we introduce STACodec, a unified codec that integrates semantic information from self-supervised learning (SSL) models into the first layer of residual vector quantization (RVQ-1) via semantic token assignment (STA). To further eliminate reliance on SSL-based semantic tokenizers and improve efficiency during inference, we propose a semantic pre-distillation (SPD) module, which predicts semantic tokens directly for assignment to the first RVQ layer during inference. Experimental results show that STACodec outperforms existing hybrid codecs in both audio reconstruction and downstream semantic tasks, demonstrating a better balance between acoustic fidelity and semantic capability.
>
---
#### [new 065] SPARC: Separating Perception And Reasoning Circuits for Test-time Scaling of VLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出SPARC框架，解决视觉语言模型测试时扩展的稳定性问题。通过分离感知与推理模块，提升模型在复杂任务中的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2602.06566v1](https://arxiv.org/pdf/2602.06566v1)**

> **作者:** Niccolo Avogaro; Nayanika Debnath; Li Mi; Thomas Frick; Junling Wang; Zexue He; Hang Hua; Konrad Schindler; Mattia Rigotti
>
> **摘要:** Despite recent successes, test-time scaling - i.e., dynamically expanding the token budget during inference as needed - remains brittle for vision-language models (VLMs): unstructured chains-of-thought about images entangle perception and reasoning, leading to long, disorganized contexts where small perceptual mistakes may cascade into completely wrong answers. Moreover, expensive reinforcement learning with hand-crafted rewards is required to achieve good performance. Here, we introduce SPARC (Separating Perception And Reasoning Circuits), a modular framework that explicitly decouples visual perception from reasoning. Inspired by sequential sensory-to-cognitive processing in the brain, SPARC implements a two-stage pipeline where the model first performs explicit visual search to localize question-relevant regions, then conditions its reasoning on those regions to produce the final answer. This separation enables independent test-time scaling with asymmetric compute allocation (e.g., prioritizing perceptual processing under distribution shift), supports selective optimization (e.g., improving the perceptual stage alone when it is the bottleneck for end-to-end performance), and accommodates compressed contexts by running global search at lower image resolutions and allocating high-resolution processing only to selected regions, thereby reducing total visual tokens count and compute. Across challenging visual reasoning benchmarks, SPARC outperforms monolithic baselines and strong visual-grounding approaches. For instance, SPARC improves the accuracy of Qwen3VL-4B on the $V^*$ VQA benchmark by 6.7 percentage points, and it surpasses "thinking with images" by 4.6 points on a challenging OOD task despite requiring a 200$\times$ lower token budget.
>
---
#### [new 066] AgentCPM-Report: Interleaving Drafting and Deepening for Open-Ended Deep Research
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AgentCPM-Report，解决开放性深度研究报告生成问题。通过动态修订大纲和交替 drafting 与 deepening，提升信息整合与洞察力。**

- **链接: [https://arxiv.org/pdf/2602.06540v1](https://arxiv.org/pdf/2602.06540v1)**

> **作者:** Yishan Li; Wentong Chen; Yukun Yan; Mingwei Li; Sen Mei; Xiaorong Wang; Kunpeng Liu; Xin Cong; Shuo Wang; Zhong Zhang; Yaxi Lu; Zhenghao Liu; Yankai Lin; Zhiyuan Liu; Maosong Sun
>
> **摘要:** Generating deep research reports requires large-scale information acquisition and the synthesis of insight-driven analysis, posing a significant challenge for current language models. Most existing approaches follow a plan-then-write paradigm, whose performance heavily depends on the quality of the initial outline. However, constructing a comprehensive outline itself demands strong reasoning ability, causing current deep research systems to rely almost exclusively on closed-source or online large models. This reliance raises practical barriers to deployment and introduces safety and privacy concerns for user-authored data. In this work, we present AgentCPM-Report, a lightweight yet high-performing local solution composed of a framework that mirrors the human writing process and an 8B-parameter deep research agent. Our framework uses a Writing As Reasoning Policy (WARP), which enables models to dynamically revise outlines during report generation. Under this policy, the agent alternates between Evidence-Based Drafting and Reasoning-Driven Deepening, jointly supporting information acquisition, knowledge refinement, and iterative outline evolution. To effectively equip small models with this capability, we introduce a Multi-Stage Agentic Training strategy, consisting of cold-start, atomic skill RL, and holistic pipeline RL. Experiments on DeepResearch Bench, DeepConsult, and DeepResearch Gym demonstrate that AgentCPM-Report outperforms leading closed-source systems, with substantial gains in Insight.
>
---
#### [new 067] Large Language Model Reasoning Failures
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，旨在分析大语言模型的推理失败问题。通过分类框架系统梳理推理类型及失败原因，提出解决策略，推动更可靠的推理能力发展。**

- **链接: [https://arxiv.org/pdf/2602.06176v1](https://arxiv.org/pdf/2602.06176v1)**

> **作者:** Peiyang Song; Pengrui Han; Noah Goodman
>
> **备注:** Repository: https://github.com/Peiyang-Song/Awesome-LLM-Reasoning-Failures. Published at TMLR 2026 with Survey Certification
>
> **摘要:** Large Language Models (LLMs) have exhibited remarkable reasoning capabilities, achieving impressive results across a wide range of tasks. Despite these advances, significant reasoning failures persist, occurring even in seemingly simple scenarios. To systematically understand and address these shortcomings, we present the first comprehensive survey dedicated to reasoning failures in LLMs. We introduce a novel categorization framework that distinguishes reasoning into embodied and non-embodied types, with the latter further subdivided into informal (intuitive) and formal (logical) reasoning. In parallel, we classify reasoning failures along a complementary axis into three types: fundamental failures intrinsic to LLM architectures that broadly affect downstream tasks; application-specific limitations that manifest in particular domains; and robustness issues characterized by inconsistent performance across minor variations. For each reasoning failure, we provide a clear definition, analyze existing studies, explore root causes, and present mitigation strategies. By unifying fragmented research efforts, our survey provides a structured perspective on systemic weaknesses in LLM reasoning, offering valuable insights and guiding future research towards building stronger, more reliable, and robust reasoning capabilities. We additionally release a comprehensive collection of research works on LLM reasoning failures, as a GitHub repository at https://github.com/Peiyang-Song/Awesome-LLM-Reasoning-Failures, to provide an easy entry point to this area.
>
---
#### [new 068] Personality as Relational Infrastructure: User Perceptions of Personality-Trait-Infused LLM Messaging
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文研究AI消息个性化对用户感知的影响，属于行为改变系统任务。解决的问题是个性化是否提升单条消息效果或通过整体暴露影响感知。工作包括实验对比不同LLM策略及人格特质的影响。**

- **链接: [https://arxiv.org/pdf/2602.06596v1](https://arxiv.org/pdf/2602.06596v1)**

> **作者:** Dominik P. Hofer; David Haag; Rania Islambouli; Jan D. Smeddinck
>
> **备注:** Currently under review
>
> **摘要:** Digital behaviour change systems increasingly rely on repeated, system-initiated messages to support users in everyday contexts. LLMs enable these messages to be personalised consistently across interactions, yet it remains unclear whether such personalisation improves individual messages or instead shapes users' perceptions through patterns of exposure. We explore this question in the context of LLM-generated JITAIs, which are short, context-aware messages delivered at moments deemed appropriate to support behaviour change, using physical activity as an application domain. In a controlled retrospective study, 90 participants evaluated messages generated using four LLM strategies: baseline prompting, few-shot prompting, fine-tuned models, and retrieval augmented generation, each implemented with and without Big Five Personality Traits to produce personality-aligned communication across multiple scenarios. Using ordinal multilevel models with within-between decomposition, we distinguish trial-level effects, whether personality information improves evaluations of individual messages, from person-level exposure effects, whether participants receiving higher proportions of personality-informed messages exhibit systematically different overall perceptions. Results showed no trial-level associations, but participants who received higher proportions of BFPT-informed messages rated the messages as more personalised, appropriate, and reported less negative affect. We use Communication Accommodation Theory for post-hoc analysis. These results suggest that personality-based personalisation in behaviour change systems may operate primarily through aggregate exposure rather than per-message optimisation, with implications for how adaptive systems are designed and evaluated in sustained human-AI interaction. In-situ longitudinal studies are needed to validate these findings in real-world contexts.
>
---
#### [new 069] Malicious Agent Skills in the Wild: A Large-Scale Security Empirical Study
- **分类: cs.CR; cs.AI; cs.CL; cs.ET**

- **简介: 该论文属于安全研究任务，旨在解决第三方代理技能的恶意威胁问题。通过构建首个标注数据集，识别并分析了恶意技能及其漏洞，提出安全防护方法。**

- **链接: [https://arxiv.org/pdf/2602.06547v1](https://arxiv.org/pdf/2602.06547v1)**

> **作者:** Yi Liu; Zhihao Chen; Yanjun Zhang; Gelei Deng; Yuekang Li; Jianting Ning; Leo Yu Zhang
>
> **摘要:** Third-party agent skills extend LLM-based agents with instruction files and executable code that run on users' machines. Skills execute with user privileges and are distributed through community registries with minimal vetting, but no ground-truth dataset exists to characterize the resulting threats. We construct the first labeled dataset of malicious agent skills by behaviorally verifying 98,380 skills from two community registries, confirming 157 malicious skills with 632 vulnerabilities. These attacks are not incidental. Malicious skills average 4.03 vulnerabilities across a median of three kill chain phases, and the ecosystem has split into two archetypes: Data Thieves that exfiltrate credentials through supply chain techniques, and Agent Hijackers that subvert agent decision-making through instruction manipulation. A single actor accounts for 54.1\% of confirmed cases through templated brand impersonation. Shadow features, capabilities absent from public documentation, appear in 0\% of basic attacks but 100\% of advanced ones; several skills go further by exploiting the AI platform's own hook system and permission flags. Responsible disclosure led to 93.6\% removal within 30 days. We release the dataset and analysis pipeline to support future work on agent skill security.
>
---
#### [new 070] The Condensate Theorem: Transformers are O(n), Not $O(n^2)$
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决Transformer模型的注意力计算复杂度问题。通过发现注意力集中在特定拓扑结构上，实现O(n)复杂度的高效计算，显著提升推理速度。**

- **链接: [https://arxiv.org/pdf/2602.06317v1](https://arxiv.org/pdf/2602.06317v1)**

> **作者:** Jorge L. Ruiz Williams
>
> **备注:** 13 pages, 4 figures, 8 tables
>
> **摘要:** We present the Condensate Theorem: attention sparsity is a learned topological property, not an architectural constraint. Through empirical analysis of trained language models, we find that attention mass concentrates on a distinct topological manifold -- and this manifold can be identified dynamically without checking every position. We prove a general result: for any query, projecting attention onto the Condensate Manifold (Anchor + Window + Dynamic Top-k) achieves 100% output equivalence with full $O(n^2)$ attention. This is not an approximation -- it is lossless parity. We validate this across GPT-2, Pythia, Qwen2, TinyLlama, and Mistral, demonstrating bit-exact token matching on 1,500+ generated tokens. By mapping this topology to hardware, our Topological Attention kernel achieves a 159x measured speedup at 131K tokens (3.94ms vs 628ms) and a projected >1,200x speedup at 1M tokens, reducing inference costs by >99.9% compared to Flash Attention. We conclude that the quadratic bottleneck is an artifact of naive implementation, not intelligence.
>
---
#### [new 071] MoSE: Mixture of Slimmable Experts for Efficient and Adaptive Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出MoSE架构，解决高效自适应语言模型问题。通过可裁剪专家结构实现更连续的精度与计算权衡，提升模型效率。**

- **链接: [https://arxiv.org/pdf/2602.06154v1](https://arxiv.org/pdf/2602.06154v1)**

> **作者:** Nurbek Tastan; Stefanos Laskaridis; Karthik Nandakumar; Samuel Horvath
>
> **摘要:** Mixture-of-Experts (MoE) models scale large language models efficiently by sparsely activating experts, but once an expert is selected, it is executed fully. Hence, the trade-off between accuracy and computation in an MoE model typically exhibits large discontinuities. We propose Mixture of Slimmable Experts (MoSE), an MoE architecture in which each expert has a nested, slimmable structure that can be executed at variable widths. This enables conditional computation not only over which experts are activated, but also over how much of each expert is utilized. Consequently, a single pretrained MoSE model can support a more continuous spectrum of accuracy-compute trade-offs at inference time. We present a simple and stable training recipe for slimmable experts under sparse routing, combining multi-width training with standard MoE objectives. During inference, we explore strategies for runtime width determination, including a lightweight test-time training mechanism that learns how to map router confidence/probabilities to expert widths under a fixed budget. Experiments on GPT models trained on OpenWebText demonstrate that MoSE matches or improves upon standard MoE at full width and consistently shifts the Pareto frontier for accuracy vs. cost, achieving comparable performance with significantly fewer FLOPs.
>
---
#### [new 072] Steering Safely or Off a Cliff? Rethinking Specificity and Robustness in Inference-Time Interventions
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究模型干预中的特定性与鲁棒性问题，属于自然语言处理任务。旨在解决干预措施是否仅影响预期属性而不引发意外行为的问题，通过框架评估三维度特定性，发现现有方法在鲁棒性上存在缺陷。**

- **链接: [https://arxiv.org/pdf/2602.06256v1](https://arxiv.org/pdf/2602.06256v1)**

> **作者:** Navita Goyal; Hal Daumé
>
> **备注:** EACL 2026 Main, Long Paper
>
> **摘要:** Model steering, which involves intervening on hidden representations at inference time, has emerged as a lightweight alternative to finetuning for precisely controlling large language models. While steering efficacy has been widely studied, evaluations of whether interventions alter only the intended property remain limited, especially with respect to unintended changes in behaviors related to the target property. We call this notion specificity. We propose a framework that distinguishes three dimensions of specificity: general (preserving fluency and unrelated abilities), control (preserving related control properties), and robustness (preserving control properties under distribution shifts). We study two safety-critical use cases: steering models to reduce overrefusal and faithfulness hallucinations, and show that while steering achieves high efficacy and largely maintains general and control specificity, it consistently fails to preserve robustness specificity. In the case of overrefusal steering, for example, all steering methods reduce overrefusal without harming general abilities and refusal on harmful queries; however, they substantially increase vulnerability to jailbreaks. Our work provides the first systematic evaluation of specificity in model steering, showing that standard efficacy and specificity checks are insufficient, because without robustness evaluation, steering methods may appear reliable even when they compromise model safety.
>
---
#### [new 073] Deep networks learn to parse uniform-depth context-free languages from local statistics
- **分类: stat.ML; cond-mat.dis-nn; cs.CL; cs.LG**

- **简介: 该论文研究语言结构学习任务，解决如何从句子中学习语法结构的问题。通过引入可调PCFG和基于深度网络的推理算法，分析数据统计与学习能力的关系。**

- **链接: [https://arxiv.org/pdf/2602.06065v1](https://arxiv.org/pdf/2602.06065v1)**

> **作者:** Jack T. Parley; Francesco Cagnetta; Matthieu Wyart
>
> **摘要:** Understanding how the structure of language can be learned from sentences alone is a central question in both cognitive science and machine learning. Studies of the internal representations of Large Language Models (LLMs) support their ability to parse text when predicting the next word, while representing semantic notions independently of surface form. Yet, which data statistics make these feats possible, and how much data is required, remain largely unknown. Probabilistic context-free grammars (PCFGs) provide a tractable testbed for studying these questions. However, prior work has focused either on the post-hoc characterization of the parsing-like algorithms used by trained networks; or on the learnability of PCFGs with fixed syntax, where parsing is unnecessary. Here, we (i) introduce a tunable class of PCFGs in which both the degree of ambiguity and the correlation structure across scales can be controlled; (ii) provide a learning mechanism -- an inference algorithm inspired by the structure of deep convolutional networks -- that links learnability and sample complexity to specific language statistics; and (iii) validate our predictions empirically across deep convolutional and transformer-based architectures. Overall, we propose a unifying framework where correlations at different scales lift local ambiguities, enabling the emergence of hierarchical representations of the data.
>
---
#### [new 074] Self-Improving World Modelling with Latent Actions
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出SWIRL框架，用于无需动作标注的自提升世界建模，解决LLMs和VLMs在无监督环境下学习状态转移的问题。**

- **链接: [https://arxiv.org/pdf/2602.06130v1](https://arxiv.org/pdf/2602.06130v1)**

> **作者:** Yifu Qiu; Zheng Zhao; Waylon Li; Yftah Ziser; Anna Korhonen; Shay B. Cohen; Edoardo M. Ponti
>
> **摘要:** Internal modelling of the world -- predicting transitions between previous states $X$ and next states $Y$ under actions $Z$ -- is essential to reasoning and planning for LLMs and VLMs. Learning such models typically requires costly action-labelled trajectories. We propose SWIRL, a self-improvement framework that learns from state-only sequences by treating actions as a latent variable and alternating between Forward World Modelling (FWM) $P_θ(Y|X,Z)$ and an Inverse Dynamics Modelling (IDM) $Q_φ(Z|X,Y)$. SWIRL iterates two phases: (1) Variational Information Maximisation, which updates the FWM to generate next states that maximise conditional mutual information with latent actions given prior states, encouraging identifiable consistency; and (2) ELBO Maximisation, which updates the IDM to explain observed transitions, effectively performing coordinate ascent. Both models are trained with reinforcement learning (specifically, GRPO) with the opposite frozen model's log-probability as a reward signal. We provide theoretical learnability guarantees for both updates, and evaluate SWIRL on LLMs and VLMs across multiple environments: single-turn and multi-turn open-world visual dynamics and synthetic textual environments for physics, web, and tool calling. SWIRL achieves gains of 16% on AURORABench, 28% on ByteMorph, 16% on WorldPredictionBench, and 14% on StableToolBench.
>
---
#### [new 075] Designing Computational Tools for Exploring Causal Relationships in Qualitative Data
- **分类: cs.HC; cs.CL; cs.CY**

- **简介: 该论文属于人机交互与社会科学研究领域，旨在解决定性数据分析中因果关系探索的问题。通过设计QualCausal系统，实现因果关系的可视化分析，减轻分析负担。**

- **链接: [https://arxiv.org/pdf/2602.06506v1](https://arxiv.org/pdf/2602.06506v1)**

> **作者:** Han Meng; Qiuyuan Lyu; Peinuan Qin; Yitian Yang; Renwen Zhang; Wen-Chieh Lin; Yi-Chieh Lee
>
> **备注:** 19 pages, 5 figures, conditionally accepted by CHI26
>
> **摘要:** Exploring causal relationships for qualitative data analysis in HCI and social science research enables the understanding of user needs and theory building. However, current computational tools primarily characterize and categorize qualitative data; the few systems that analyze causal relationships either inadequately consider context, lack credibility, or produce overly complex outputs. We first conducted a formative study with 15 participants interested in using computational tools for exploring causal relationships in qualitative data to understand their needs and derive design guidelines. Based on these findings, we designed and implemented QualCausal, a system that extracts and illustrates causal relationships through interactive causal network construction and multi-view visualization. A feedback study (n = 15) revealed that participants valued our system for reducing the analytical burden and providing cognitive scaffolding, yet navigated how such systems fit within their established research paradigms, practices, and habits. We discuss broader implications for designing computational tools that support qualitative data analysis.
>
---
#### [new 076] Quantum Attention by Overlap Interference: Predicting Sequences from Classical and Many-Body Quantum Data
- **分类: quant-ph; cs.CL; cs.LG**

- **简介: 该论文提出一种量子自注意力机制（QSA），用于序列预测任务，解决传统方法在处理量子数据时的效率与非线性问题。通过量子态重叠实现非线性，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.06699v1](https://arxiv.org/pdf/2602.06699v1)**

> **作者:** Alessio Pecilli; Matteo Rosati
>
> **备注:** 4 + 1 pages, 2 figures
>
> **摘要:** We propose a variational quantum implementation of self-attention (QSA), the core operation in transformers and large language models, which predicts future elements of a sequence by forming overlap-weighted combinations of past data. At variance with previous approaches, our QSA realizes the required nonlinearity through interference of state overlaps and returns a Renyi-1/2 cross-entropy loss directly as the expectation value of an observable, avoiding the need to decode amplitude-encoded predictions into classical logits. Furthermore, QSA naturally accommodates a constrained, trainable data-embedding that ties quantum state overlaps to data-level similarities. We find a gate complexity dominant scaling O(T d^2) for QSA, versus O(T^2 d) classically, suggesting an advantage in the practical regime where the sequence length T dominates the embedding size d. In simulations, we show that our QSA-based quantum transformer learns sequence prediction on classical data and on many-body transverse-field Ising quantum trajectories, establishing trainable attention as a practical primitive for quantum dynamical modeling.
>
---
## 更新

#### [replaced 001] MapFormer: Self-Supervised Learning of Cognitive Maps with Input-Dependent Positional Embeddings
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出MapFormer，一种基于Transformer的自监督学习模型，用于构建认知地图，解决AI在分布外泛化上的不足。通过输入相关的位置编码，实现结构与内容解耦，提升导航等任务的泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.19279v3](https://arxiv.org/pdf/2511.19279v3)**

> **作者:** Victor Rambaud; Salvador Mascarenhas; Yair Lakretz
>
> **备注:** 19 pages (29 with appendix), 8 figures
>
> **摘要:** A cognitive map is an internal model which encodes the abstract relationships among entities in the world, giving humans and animals the flexibility to adapt to new situations, with a strong out-of-distribution (OOD) generalization that current AI systems still do not possess. To bridge this gap, we introduce MapFormers, new architectures based on Transformer models, which can learn cognitive maps from observational data and perform path integration in parallel, in a self-supervised manner. Cognitive maps are learned in the model by disentangling structural relationships in the inputs from their specific content, a property that can be achieved naturally by updating the positional encoding in Transformers with input-dependent matrices. We developed two variants of MapFormers that unify absolute and relative positional encoding to model episodic (EM) and working memory (WM), respectively. We tested MapFormers on several tasks, including a classic 2D navigation task, showing that our models can learn a cognitive map of the underlying space and generalize OOD (e.g., to longer sequences) with near-perfect performance, unlike current architectures. Together, these results demonstrate the superiority of models designed to learn a cognitive map, and the importance of introducing a structural bias for structure-content disentanglement, which can be achieved in Transformers with input-dependent positional encoding. MapFormers have broad applications in both neuroscience and AI, by explaining the neural mechanisms giving rise to cognitive maps, while allowing these relation models to be learned at scale.
>
---
#### [replaced 002] Layer-adaptive Expert Pruning for Pre-Training of Mixture-of-Experts Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型预训练任务，旨在解决MoE模型预训练效率低的问题。通过LAEP算法优化专家选择与分布，提升训练效率并减少参数量。**

- **链接: [https://arxiv.org/pdf/2601.14327v2](https://arxiv.org/pdf/2601.14327v2)**

> **作者:** YuanLab. ai; :; Shawn Wu; Jiangang Luo; Tong Yu; Darcy Chen; Sean Wang; Xudong Zhao; Louie Li; Claire Wang; Hunter He; Carol Wang; Allen Wang
>
> **摘要:** Although Mixture-of-Experts (MoE) Large Language Models (LLMs) deliver superior accuracy with a reduced number of active parameters, their pre-training represents a significant computationally bottleneck due to underutilized experts and limited training efficiency. This work introduces a Layer-Adaptive Expert Pruning (LAEP) algorithm designed for the pre-training stage of MoE LLMs. In contrast to previous expert pruning approaches that operate primarily in the post-training phase, the proposed algorithm enhances training efficiency by selectively pruning underutilized experts and reorganizing experts across computing devices according to token distribution statistics. Comprehensive experiments demonstrate that LAEP effectively reduces model size and substantially improves pre-training efficiency. In particular, when pre-training the Yuan3.0-1T Base model from scratch original with 1515B parameters, LAEP achieves a 48.3% improvement in training efficiency alongside a 33.3% parameter reduction, while still delivering excellent performance across multiple domains.
>
---
#### [replaced 003] Probabilistic Aggregation and Targeted Embedding Optimization for Collective Moral Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于道德推理任务，旨在解决大语言模型在复杂道德困境中的分歧问题。通过概率聚合和嵌入优化，提升模型间的一致性与准确性。**

- **链接: [https://arxiv.org/pdf/2506.14625v3](https://arxiv.org/pdf/2506.14625v3)**

> **作者:** Chenchen Yuan; Zheyu Zhang; Shuo Yang; Bardh Prenkaj; Gjergji Kasneci
>
> **备注:** Accepted to ACL 2025 (Findings)
>
> **摘要:** Large Language Models (LLMs) have shown impressive moral reasoning abilities. Yet they often diverge when confronted with complex, multi-factor moral dilemmas. To address these discrepancies, we propose a framework that synthesizes multiple LLMs' moral judgments into a collectively formulated moral judgment, realigning models that deviate significantly from this consensus. Our aggregation mechanism fuses continuous moral acceptability scores (beyond binary labels) into a collective probability, weighting contributions by model reliability. For misaligned models, a targeted embedding-optimization procedure fine-tunes token embeddings for moral philosophical theories, minimizing JS divergence to the consensus while preserving semantic integrity. Experiments on a large-scale social moral dilemma dataset show our approach builds robust consensus and improves individual model fidelity. These findings highlight the value of data-driven moral alignment across multiple models and its potential for safer, more consistent AI systems.
>
---
#### [replaced 004] Harnessing the Unseen: The Hidden Influence of Intrinsic Knowledge in Long-Context Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨长上下文语言模型中内在知识的影响。研究解决模型如何平衡参数知识与外部信息利用的问题，提出混合测试方法评估双能力。**

- **链接: [https://arxiv.org/pdf/2504.08202v2](https://arxiv.org/pdf/2504.08202v2)**

> **作者:** Yu Fu; Haz Sameen Shahgir; Hui Liu; Xianfeng Tang; Qi He; Yue Dong
>
> **备注:** 17 pages,11figures (accepted to AAAI 2026)
>
> **摘要:** Recent advances in long-context language models (LCLMs), designed to handle extremely long contexts, primarily focus on utilizing external contextual information, often leaving the influence of language models' parametric knowledge underexplored. In this work, we firstly investigate how this parametric knowledge affects content generation and demonstrate that its impact becomes increasingly pronounced as context length extends. Furthermore, we show that the model's ability to utilize parametric knowledge, which we call parametric recall ability, does not improve simultaneously with its ability to leverage contextual knowledge through extrinsic retrieval ability. Moreover, better extrinsic retrieval ability can interfere with the model's parametric recall ability, limiting its full potential. To bridge this gap, we design a simple yet effective Hybrid Needle-in-a-Haystack test that evaluates models based on their capabilities across both abilities, rather than solely emphasizing extrinsic retrieval ability. Our experimental results reveal that Qwen-2.5 models significantly outperform Llama-3.1 models, demonstrating superior potential to combine various abilities. Moreover, even the more powerful Llama-3.1-70B-Instruct model fails to exhibit better performance, highlighting the importance of evaluating models from a dual-ability perspective.
>
---
#### [replaced 005] CLaRa: Bridging Retrieval and Generation with Continuous Latent Reasoning
- **分类: cs.CL**

- **简介: 该论文提出CLaRa，解决RAG中上下文过长和优化不一致的问题。通过联合优化检索与生成，实现高效文本压缩与重排序。**

- **链接: [https://arxiv.org/pdf/2511.18659v3](https://arxiv.org/pdf/2511.18659v3)**

> **作者:** Jie He; Richard He Bai; Sinead Williamson; Jeff Z. Pan; Navdeep Jaitly; Yizhe Zhang
>
> **摘要:** Retrieval-augmented generation (RAG) enhances large language models (LLMs) with external knowledge but still suffers from long contexts and disjoint retrieval-generation optimization. In this work, we propose CLaRa (Continuous Latent Reasoning), a unified framework that performs embedding-based compression and joint optimization in a shared continuous space. To obtain semantically rich and retrievable compressed vectors, thereby reducing the document length fed into the generator, we introduce SCP, a key-preserving data synthesis framework based on question answering and paraphrase supervision. CLaRa then trains the reranker and generator end-to-end via a single language modeling loss, with gradients flowing through both modules using a differentiable top-k estimator. Theoretically, this unified optimization aligns retrieval relevance with answer quality. Experiments across multiple QA benchmarks show that CLaRa achieves state-of-the-art compression and reranking performance, even at a text compression rate of 16, outperforming text-based fine-tuned baselines.
>
---
#### [replaced 006] FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出FadeMem，解决AI代理记忆过载与遗忘不足问题，通过生物启发的遗忘机制实现高效记忆管理。**

- **链接: [https://arxiv.org/pdf/2601.18642v2](https://arxiv.org/pdf/2601.18642v2)**

> **作者:** Lei Wei; Xiao Peng; Xu Dong; Niantao Xie; Bin Wang
>
> **摘要:** Large language models deployed as autonomous agents face critical memory limitations, lacking selective forgetting mechanisms that lead to either catastrophic forgetting at context boundaries or information overload within them. While human memory naturally balances retention and forgetting through adaptive decay processes, current AI systems employ binary retention strategies that preserve everything or lose it entirely. We propose FadeMem, a biologically-inspired agent memory architecture that incorporates active forgetting mechanisms mirroring human cognitive efficiency. FadeMem implements differential decay rates across a dual-layer memory hierarchy, where retention is governed by adaptive exponential decay functions modulated by semantic relevance, access frequency, and temporal patterns. Through LLM-guided conflict resolution and intelligent memory fusion, our system consolidates related information while allowing irrelevant details to fade. Experiments on Multi-Session Chat, LoCoMo, and LTI-Bench demonstrate superior multi-hop reasoning and retrieval with 45\% storage reduction, validating the effectiveness of biologically-inspired forgetting in agent memory systems.
>
---
#### [replaced 007] Quantifying the Effect of Test Set Contamination on Generative Evaluations
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于AI评估任务，研究测试集污染对生成评估的影响。通过实验分析污染对模型性能、训练和推理的影响，揭示生成任务中的新复杂性。**

- **链接: [https://arxiv.org/pdf/2601.04301v2](https://arxiv.org/pdf/2601.04301v2)**

> **作者:** Rylan Schaeffer; Joshua Kazdan; Baber Abbasi; Ken Ziyu Liu; Brando Miranda; Ahmed Ahmed; Fazl Berez; Abhay Puri; Stella Biderman; Niloofar Mireshghallah; Sanmi Koyejo
>
> **摘要:** As frontier AI systems are pretrained on web-scale data, test set contamination has become a critical concern for accurately assessing their capabilities. While research has thoroughly investigated the impact of test set contamination on discriminative evaluations like multiple-choice question-answering, comparatively little research has studied the impact of test set contamination on generative evaluations. In this work, we quantitatively assess the effect of test set contamination on generative evaluations through the language model lifecycle. We pretrain language models on mixtures of web data and the MATH benchmark, sweeping model sizes and number of test set replicas contaminating the pretraining corpus; performance improves with contamination and model size. Using scaling laws, we make a surprising discovery: including even a single test set replica enables models to achieve lower loss than the irreducible error of training on the uncontaminated corpus. We then study further training: overtraining with fresh data reduces the effects of contamination, whereas supervised finetuning on the training set can either increase or decrease performance on test data, depending on the amount of pretraining contamination. Finally, at inference, we identify factors that modulate memorization: high sampling temperatures mitigate contamination effects, and longer solutions are exponentially more difficult to memorize than shorter ones, presenting a contrast with discriminative evaluations, where solutions are only a few tokens in length. By characterizing how generation and memorization interact, we highlight a new layer of complexity for trustworthy evaluation of AI systems.
>
---
#### [replaced 008] SAGE: Benchmarking and Improving Retrieval for Deep Research Agents
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决深度研究代理在复杂查询中的检索效果问题。通过构建基准SAGE，对比不同检索方法，发现LLM检索器表现不佳，并提出改进框架提升效果。**

- **链接: [https://arxiv.org/pdf/2602.05975v2](https://arxiv.org/pdf/2602.05975v2)**

> **作者:** Tiansheng Hu; Yilun Zhao; Canyu Zhang; Arman Cohan; Chen Zhao
>
> **摘要:** Deep research agents have emerged as powerful systems for addressing complex queries. Meanwhile, LLM-based retrievers have demonstrated strong capability in following instructions or reasoning. This raises a critical question: can LLM-based retrievers effectively contribute to deep research agent workflows? To investigate this, we introduce SAGE, a benchmark for scientific literature retrieval comprising 1,200 queries across four scientific domains, with a 200,000 paper retrieval corpus. We evaluate six deep research agents and find that all systems struggle with reasoning-intensive retrieval. Using DR Tulu as backbone, we further compare BM25 and LLM-based retrievers (i.e., ReasonIR and gte-Qwen2-7B-instruct) as alternative search tools. Surprisingly, BM25 significantly outperforms LLM-based retrievers by approximately 30%, as existing agents generate keyword-oriented sub-queries. To improve performance, we propose a corpus-level test-time scaling framework that uses LLMs to augment documents with metadata and keywords, making retrieval easier for off-the-shelf retrievers. This yields 8% and 2% gains on short-form and open-ended questions, respectively.
>
---
#### [replaced 009] code_transformed: The Influence of Large Language Models on Code
- **分类: cs.CL; cs.AI; cs.LG; cs.SE**

- **简介: 该论文属于代码风格分析任务，研究LLM对编程风格的影响，通过分析大量代码数据，揭示LLM生成代码在命名、复杂度等方面的演变趋势。**

- **链接: [https://arxiv.org/pdf/2506.12014v2](https://arxiv.org/pdf/2506.12014v2)**

> **作者:** Yuliang Xu; Siming Huang; Mingmeng Geng; Yao Wan; Xuanhua Shi; Dongping Chen
>
> **备注:** EACL 2026 Findings
>
> **摘要:** Coding remains one of the most fundamental modes of interaction between humans and machines. With the rapid advancement of Large Language Models (LLMs), code generation capabilities have begun to significantly reshape programming practices. This development prompts a central question: Have LLMs transformed code style, and how can such transformation be characterized? In this paper, we present a pioneering study that investigates the impact of LLMs on code style, with a focus on naming conventions, complexity, maintainability, and similarity. By analyzing code from over 20,000 GitHub repositories linked to arXiv papers published between 2020 and 2025, we identify measurable trends in the evolution of coding style that align with characteristics of LLM-generated code. For instance, the proportion of snake_case function names in Python code increased from 40.7% in Q1 2023 to 49.8% in Q3 2025. Furthermore, we investigate how LLMs approach algorithmic problems by examining their reasoning processes. Our experimental results may provide the first large-scale empirical evidence that LLMs affect real-world programming style. We release all the experimental dataset and source code at: https://github.com/ignorancex/LLM_code
>
---
#### [replaced 010] Think-Augmented Function Calling: Improving LLM Parameter Accuracy Through Embedded Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在提升大模型函数调用的参数准确性。通过引入显式推理机制，增强参数生成的透明度与合理性。**

- **链接: [https://arxiv.org/pdf/2601.18282v2](https://arxiv.org/pdf/2601.18282v2)**

> **作者:** Lei Wei; Xiao Peng; Jinpeng Ou; Bin Wang
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities in function calling for autonomous agents, yet current mechanisms lack explicit reasoning transparency during parameter generation, particularly for complex functions with interdependent parameters. While existing approaches like chain-of-thought prompting operate at the agent level, they fail to provide fine-grained reasoning guidance for individual function parameters. To address these limitations, we propose Think-Augmented Function Calling (TAFC), a novel framework that enhances function calling accuracy through explicit reasoning at both function and parameter levels. Our method introduces a universal "think" parameter augmentation that enables models to articulate their decision-making process, with dynamic optimization for parameter descriptions to improve reasoning quality. For complex parameters, TAFC automatically triggers granular reasoning based on complexity scoring, ensuring appropriate justification for critical decisions. Additionally, we propose reasoning-guided optimization to align generated reasoning with human expectations. TAFC requires no architectural modifications to existing LLMs while maintaining full API compatibility. Evaluation on ToolBench across proprietary and open-source models demonstrates significant improvements in parameter generation accuracy and reasoning coherence for multi-parameter functions, while providing enhanced interpretability for debugging AI agent behaviors.
>
---
#### [replaced 011] Unsupervised Classification of English Words Based on Phonological Information: Discovery of Germanic and Latinate Clusters
- **分类: cs.CL**

- **简介: 该论文属于语言学与计算语言学任务，旨在通过语音信息无监督分类英语词汇，解决词源区分问题。研究发现，基于音位模式的聚类能有效识别日耳曼语和拉丁语源词汇。**

- **链接: [https://arxiv.org/pdf/2504.11770v4](https://arxiv.org/pdf/2504.11770v4)**

> **作者:** Takashi Morita; Timothy J. O'Donnell
>
> **摘要:** Cross-linguistically, native words and loanwords follow different phonological rules. In English, for example, words of Germanic and Latinate origin exhibit different stress patterns, and a certain syntactic structure, double-object datives, is predominantly associated with Germanic verbs rather than Latinate verbs. From the perspective of language acquisition, however, such etymology-based generalizations raise learnability concerns, since the historical origins of words are presumably inaccessible information for general language learners. In this study, we present computational evidence indicating that the Germanic-Latinate distinction in the English lexicon is learnable from the phonotactic information of individual words. Specifically, we performed an unsupervised clustering on corpus-extracted words, and the resulting word clusters largely aligned with the etymological distinction. The model-discovered clusters also recovered various linguistic generalizations documented in the previous literature regarding the corresponding etymological classes. Moreover, our model also uncovered previously unrecognized features of the quasi-etymological clusters. Taken together with prior results from Japanese, our findings indicate that the proposed method provides a general, cross-linguistic approach to discovering etymological structure from phonotactic cues in the lexicon.
>
---
#### [replaced 012] D-SCoRE: Document-Centric Segmentation and CoT Reasoning with Structured Export for QA-CoT Data Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出D-SCoRE框架，用于生成高质量QA-CoT数据。解决领域特定数据稀缺问题，通过文档处理、分割、推理和结构化导出，提升数据多样性与相关性。**

- **链接: [https://arxiv.org/pdf/2508.01309v2](https://arxiv.org/pdf/2508.01309v2)**

> **作者:** Weibo Zhou; Lingbo Li; Shangsong Liang
>
> **摘要:** The scarcity and high cost of high-quality domain-specific question-answering (QA) datasets limit supervised fine-tuning of large language models (LLMs). We introduce $\textbf{D-SCoRE}$, a training-free framework that leverages LLMs and prompt engineering to automatically generate diverse, rich QA datasets with Chain-of-Thought (CoT) from arbitrary textual sources. By integrating $\textbf{D}$ocument-centric processing, $\textbf{S}$egmentation, $\textbf{Co}$T $\textbf{R}$easoning, and structured $\textbf{E}$xport - along with multi-dimensional controls such as semantic role transformation, question type balancing, and counterfactual augmentation - D-SCoRE produces tailored QA pairs with enhanced diversity and relevance. LLMs fine-tuned on D-SCoRE-generated datasets outperform those trained on human-annotated QA data across most evaluated domains. Its efficiency and scalability enable rapid, high-performance domain-adaptive fine-tuning on consumer-grade hardware, generating over 1,100 high-quality QA pairs per GPU-hour end-to-end.
>
---
#### [replaced 013] T$^3$-S2S: Training-free Triplet Tuning for Sketch to Scene Synthesis in Controllable Concept Art Generation
- **分类: cs.CV; cs.CL; cs.GR**

- **简介: 该论文属于2D概念艺术生成任务，解决3D场景生成中多实例和结构化地形布局的问题。提出T3-S2S方法，通过三模块优化生成更精确的场景图像。**

- **链接: [https://arxiv.org/pdf/2412.13486v2](https://arxiv.org/pdf/2412.13486v2)**

> **作者:** Zhenhong Sun; Yifu Wang; Yonhon Ng; Yongzhi Xu; Daoyi Dong; Hongdong Li; Pan Ji
>
> **备注:** https://openreview.net/forum?id=lyn2BgKQ8F
>
> **摘要:** 2D concept art generation for 3D scenes is a crucial yet challenging task in computer graphics, as creating natural intuitive environments still demands extensive manual effort in concept design. While generative AI has simplified 2D concept design via text-to-image synthesis, it struggles with complex multi-instance scenes and offers limited support for structured terrain layout. In this paper, we propose a Training-free Triplet Tuning for Sketch-to-Scene (T3-S2S) generation after reviewing the entire cross-attention mechanism. This scheme revitalizes the ControlNet model for detailed multi-instance generation via three key modules: Prompt Balance ensures keyword representation and minimizes the risk of missing critical instances; Characteristic Priority emphasizes sketch-based features by highlighting TopK indices in feature channels; and Dense Tuning refines contour details within instance-related regions of the attention map. Leveraging the controllability of T3-S2S, we also introduce a feature-sharing strategy with dual prompt sets to generate layer-aware isometric and terrain-view representations for the terrain layout. Experiments show that our sketch-to-scene workflow consistently produces multi-instance 2D scenes with details aligned with input prompts.
>
---
#### [replaced 014] OpenDeception: Learning Deception and Trust in Human-AI Interaction via Multi-Agent Simulation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于人机交互中的信任与欺骗评估任务，旨在解决AI在对话中可能产生的欺骗行为问题。工作包括构建基准、设计意图与信任网络，并通过模拟数据训练模型进行风险评估。**

- **链接: [https://arxiv.org/pdf/2504.13707v3](https://arxiv.org/pdf/2504.13707v3)**

> **作者:** Yichen Wu; Qianqian Gao; Xudong Pan; Geng Hong; Min Yang
>
> **摘要:** As large language models (LLMs) are increasingly deployed as interactive agents, open-ended human-AI interactions can involve deceptive behaviors with serious real-world consequences, yet existing evaluations remain largely scenario-specific and model-centric. We introduce OpenDeception, a lightweight framework for jointly evaluating deception risk from both sides of human-AI dialogue. It consists of a scenario benchmark with 50 real-world deception cases, an IntentNet that infers deceptive intent from agent reasoning, and a TrustNet that estimates user susceptibility. To address data scarcity, we synthesize high-risk dialogues via LLM-based role-and-goal simulation, and train the User Trust Scorer using contrastive learning on controlled response pairs, avoiding unreliable scalar labels. Experiments on 11 LLMs and three large reasoning models show that over 90% of goal-driven interactions in most models exhibit deceptive intent, with stronger models displaying higher risk. A real-world case study adapted from a documented AI-induced suicide incident further demonstrates that our joint evaluation can proactively trigger warnings before critical trust thresholds are reached.
>
---
#### [replaced 015] Detecting Latin in Historical Books with Large Language Models: A Multimodal Benchmark
- **分类: cs.CL; cs.AI; cs.CV; cs.DL**

- **简介: 该论文属于历史文献处理任务，旨在从混合语言文档中检测拉丁语片段。通过构建多模态数据集，评估大模型的检测能力，建立处理拉丁语的基准。**

- **链接: [https://arxiv.org/pdf/2510.19585v3](https://arxiv.org/pdf/2510.19585v3)**

> **作者:** Yu Wu; Ke Shu; Jonas Fischer; Lidia Pivovarova; David Rosson; Eetu Mäkelä; Mikko Tolonen
>
> **备注:** Accepted by the EACL 2026 main conference. Code and data available at https://github.com/COMHIS/EACL26-detect-latin
>
> **摘要:** This paper presents a novel task of extracting low-resourced and noisy Latin fragments from mixed-language historical documents with varied layouts. We benchmark and evaluate the performance of large foundation models against a multimodal dataset of 724 annotated pages. The results demonstrate that reliable Latin detection with contemporary zero-shot models is achievable, yet these models lack a functional comprehension of Latin. This study establishes a comprehensive baseline for processing Latin within mixed-language corpora, supporting quantitative analysis in intellectual history and historical linguistics. Both the dataset and code are available at https://github.com/COMHIS/EACL26-detect-latin.
>
---
#### [replaced 016] AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design
- **分类: q-bio.QM; cs.CL**

- **简介: 该论文属于抗体语言建模任务，旨在解决LLM在抗体理解与设计上的不足。构建了AFD-Instruction数据集，支持基于自然语言的抗体功能解析与生成。**

- **链接: [https://arxiv.org/pdf/2602.04916v2](https://arxiv.org/pdf/2602.04916v2)**

> **作者:** Ling Luo; Wenbin Jiang; Hongyuan Chang; Xinkang Wang; Xushi Zhang; Yueting Xiong; Mengsha Tong; Rongshan Yu
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Large language models (LLMs) have significantly advanced protein representation learning. However, their capacity to interpret and design antibodies through natural language remains limited. To address this challenge, we present AFD-Instruction, the first large-scale instruction dataset with functional annotations tailored to antibodies. This dataset encompasses two key components: antibody understanding, which infers functional attributes directly from sequences, and antibody design, which enables de novo sequence generation under functional constraints. These components provide explicit sequence-function alignment and support antibody design guided by natural language instructions. Extensive instruction-tuning experiments on general-purpose LLMs demonstrate that AFD-Instruction consistently improves performance across diverse antibody-related tasks. By linking antibody sequences with textual descriptions of function, AFD-Instruction establishes a new foundation for advancing antibody modeling and accelerating therapeutic discovery.
>
---
#### [replaced 017] Is Your Paper Being Reviewed by an LLM? Benchmarking AI Text Detection in Peer Review
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI文本检测任务，旨在解决LLM辅助撰写同行评审的识别问题。研究构建了大规模数据集并评估检测算法效果。**

- **链接: [https://arxiv.org/pdf/2502.19614v3](https://arxiv.org/pdf/2502.19614v3)**

> **作者:** Sungduk Yu; Man Luo; Avinash Madasu; Vasudev Lal; Phillip Howard
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Peer review is a critical process for ensuring the integrity of published scientific research. Confidence in this process is predicated on the assumption that experts in the relevant domain give careful consideration to the merits of manuscripts which are submitted for publication. With the recent rapid advancements in large language models (LLMs), a new risk to the peer review process is that negligent reviewers will rely on LLMs to perform the often time consuming process of reviewing a paper. However, there is a lack of existing resources for benchmarking the detectability of AI text in the domain of peer review. To address this deficiency, we introduce a comprehensive dataset containing a total of 788,984 AI-written peer reviews paired with corresponding human reviews, covering 8 years of papers submitted to each of two leading AI research conferences (ICLR and NeurIPS). We use this new resource to evaluate the ability of 18 existing AI text detection algorithms to distinguish between peer reviews fully written by humans and different state-of-the-art LLMs. Additionally, we explore a context-aware detection method called Anchor, which leverages manuscript content to detect AI-generated reviews, and analyze the sensitivity of detection models to LLM-assisted editing of human-written text. Our work reveals the difficulty of identifying AI-generated text at the individual peer review level, highlighting the urgent need for new tools and methods to detect this unethical use of generative AI. Our dataset is publicly available at: https://huggingface.co/datasets/IntelLabs/AI-Peer-Review-Detection-Benchmark.
>
---
#### [replaced 018] Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于代码生成任务，旨在解决LLM在生成高质量GPU内核时的奖励欺骗和优化惰性问题。通过设计KernelGYM环境和提出TRLOO等方法，提升生成内核的性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.05885v2](https://arxiv.org/pdf/2602.05885v2)**

> **作者:** Wei Liu; Jiawei Xu; Yingru Li; Longtao Zheng; Tianjian Li; Qian Liu; Junxian He
>
> **摘要:** High-quality kernel is critical for scalable AI systems, and enabling LLMs to generate such code would advance AI development. However, training LLMs for this task requires sufficient data, a robust environment, and the process is often vulnerable to reward hacking and lazy optimization. In these cases, models may hack training rewards and prioritize trivial correctness over meaningful speedup. In this paper, we systematically study reinforcement learning (RL) for kernel generation. We first design KernelGYM, a robust distributed GPU environment that supports reward hacking check, data collection from multi-turn interactions and long-term RL training. Building on KernelGYM, we investigate effective multi-turn RL methods and identify a biased policy gradient issue caused by self-inclusion in GRPO. To solve this, we propose Turn-level Reinforce-Leave-One-Out (TRLOO) to provide unbiased advantage estimation for multi-turn RL. To alleviate lazy optimization, we incorporate mismatch correction for training stability and introduce Profiling-based Rewards (PR) and Profiling-based Rejection Sampling (PRS) to overcome the issue. The trained model, Dr Kernel-14B, reaches performance competitive with Claude-4.5-Sonnet in Kernelbench. Finally, we study sequential test-time scaling for Dr Kernel-14B. On the KernelBench Level-2 subset, 31.6% of the generated kernels achieve at least a 1.2x speedup over the Torch reference, surpassing Claude-4.5-Sonnet (26.7%) and GPT-5 (28.6%). When selecting the best candidate across all turns, this 1.2x speedup rate further increases to 47.8%. All resources, including environment, training code, models, and dataset, are included in https://www.github.com/hkust-nlp/KernelGYM.
>
---
#### [replaced 019] Encoding syntactic objects and Merge operations in function spaces
- **分类: cs.CL; math.RA; q-bio.NC**

- **简介: 该论文属于理论语言学与计算模型领域，旨在通过函数空间和代数结构实现语法对象的数学表示，解决语法结构的神经计算实现问题。工作包括构建代数结构并模拟Merge操作。**

- **链接: [https://arxiv.org/pdf/2507.13501v2](https://arxiv.org/pdf/2507.13501v2)**

> **作者:** Matilde Marcolli; Robert C. Berwick
>
> **备注:** 48 pages, LaTeX, 4 png figures; v2: expository changes
>
> **摘要:** We provide a mathematical argument showing that, given a representation of lexical items as functions (wavelets, for instance) in some function space, it is possible to construct a faithful representation of arbitrary syntactic objects in the same function space. This space can be endowed with a commutative non-associative semiring structure built using the second Renyi entropy. The resulting representation of syntactic objects is compatible with the magma structure. The resulting set of functions is an algebra over an operad, where the operations in the operad model circuits that transform the input wave forms into a combined output that encodes the syntactic structure. The action of Merge on workspaces is faithfully implemented as action on these circuits, through a coproduct and a Hopf algebra Markov chain. The results obtained here provide a constructive argument showing the theoretical possibility of a neurocomputational realization of the core computational structure of syntax. We also present a particular case of this general construction where this type of realization of Merge is implemented as a cross frequency phase synchronization on sinusoidal waves. This also shows that Merge can be expressed in terms of the successor function of a semiring, thus clarifying the well known observation of its similarities with the successor function of arithmetic.
>
---
#### [replaced 020] AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AgentXRay，解决黑盒代理系统可解释性问题，通过工作流重构生成可解释的替代流程，提升系统透明度与可控性。**

- **链接: [https://arxiv.org/pdf/2602.05353v2](https://arxiv.org/pdf/2602.05353v2)**

> **作者:** Ruijie Shi; Houbin Zhang; Yuecheng Han; Yuheng Wang; Jingru Fan; Runde Yang; Yufan Dang; Huatao Li; Dewen Liu; Yuan Cheng; Chen Qian
>
> **摘要:** Large Language Models have shown strong capabilities in complex problem solving, yet many agentic systems remain difficult to interpret and control due to opaque internal workflows. While some frameworks offer explicit architectures for collaboration, many deployed agentic systems operate as black boxes to users. We address this by introducing Agentic Workflow Reconstruction (AWR), a new task aiming to synthesize an explicit, interpretable stand-in workflow that approximates a black-box system using only input--output access. We propose AgentXRay, a search-based framework that formulates AWR as a combinatorial optimization problem over discrete agent roles and tool invocations in a chain-structured workflow space. Unlike model distillation, AgentXRay produces editable white-box workflows that match target outputs under an observable, output-based proxy metric, without accessing model parameters. To navigate the vast search space, AgentXRay employs Monte Carlo Tree Search enhanced by a scoring-based Red-Black Pruning mechanism, which dynamically integrates proxy quality with search depth. Experiments across diverse domains demonstrate that AgentXRay achieves higher proxy similarity and reduces token consumption compared to unpruned search, enabling deeper workflow exploration under fixed iteration budgets.
>
---
#### [replaced 021] SeSE: Black-Box Uncertainty Quantification for Large Language Models Based on Structural Information Theory
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于不确定性量化任务，旨在提升大语言模型在安全关键场景中的可靠性。提出SeSE框架，通过结构信息理论更精确地评估模型不确定性。**

- **链接: [https://arxiv.org/pdf/2511.16275v3](https://arxiv.org/pdf/2511.16275v3)**

> **作者:** Xingtao Zhao; Hao Peng; Dingli Su; Xianghua Zeng; Chunyang Liu; Jinzhi Liao; Philip S. Yu
>
> **摘要:** Reliable uncertainty quantification (UQ) is essential for deploying large language models (LLMs) in safety-critical scenarios, as it enables them to abstain from responding when uncertain, thereby avoiding hallucinations, i.e., plausible yet factually incorrect responses. However, while semantic UQ methods have achieved advanced performance, they overlook latent semantic structural information that could enable more precise uncertainty estimates. In this paper, we propose \underline{Se}mantic \underline{S}tructural \underline{E}ntropy ({SeSE}), a principled black-box UQ framework applicable to both open- and closed-source LLMs. To reveal the intrinsic structure of the semantic space, SeSE constructs its optimal hierarchical abstraction through an encoding tree with minimal structural entropy. The structural entropy of this encoding tree thus quantifies the inherent uncertainty within LLM semantic space after optimal compression. Additionally, unlike existing methods that primarily focus on simple short-form generation, we extent SeSE to provide interpretable, granular uncertainty estimation for long-form outputs. We theoretically prove that SeSE generalizes semantic entropy, the gold standard for UQ in LLMs, and empirically demonstrate its superior performance over strong baselines across 24 model-dataset combinations.
>
---
#### [replaced 022] DimStance: Multilingual Datasets for Dimensional Stance Analysis
- **分类: cs.CL**

- **简介: 该论文提出DimStance，首个包含情感维度（valence-arousal）标注的多语言立场分析数据集，解决多语言情绪感知的立场分析问题。**

- **链接: [https://arxiv.org/pdf/2601.21483v2](https://arxiv.org/pdf/2601.21483v2)**

> **作者:** Jonas Becker; Liang-Chih Yu; Shamsuddeen Hassan Muhammad; Jan Philip Wahle; Terry Ruas; Idris Abdulmumin; Lung-Hao Lee; Nelson Odhiambo; Lilian Wanzare; Wen-Ni Liu; Tzu-Mi Lin; Zhe-Yu Xu; Ying-Lung Lin; Jin Wang; Maryam Ibrahim Mukhtar; Bela Gipp; Saif M. Mohammad
>
> **摘要:** Stance detection is an established task that classifies an author's attitude toward a specific target into categories such as Favor, Neutral, and Against. Beyond categorical stance labels, we leverage a long-established affective science framework to model stance along real-valued dimensions of valence (negative-positive) and arousal (calm-active). This dimensional approach captures nuanced affective states underlying stance expressions, enabling fine-grained stance analysis. To this end, we introduce DimStance, the first dimensional stance resource with valence-arousal (VA) annotations. This resource comprises 11,746 target aspects in 7,365 texts across five languages (English, German, Chinese, Nigerian Pidgin, and Swahili) and two domains (politics and environmental protection). To facilitate the evaluation of stance VA prediction, we formulate the dimensional stance regression task, analyze cross-lingual VA patterns, and benchmark pretrained and large language models under regression and prompting settings. Results show competitive performance of fine-tuned LLM regressors, persistent challenges in low-resource languages, and limitations of token-based generation. DimStance provides a foundation for multilingual, emotion-aware, stance analysis and benchmarking.
>
---
#### [replaced 023] PACE: Defying the Scaling Hypothesis of Exploration in Iterative Alignment for Mathematical Reasoning
- **分类: cs.CL**

- **简介: 该论文属于语言模型对齐任务，针对数学推理中的探索效率问题，提出PACE方法，通过修正探索策略提升对齐效果，减少计算成本。**

- **链接: [https://arxiv.org/pdf/2602.05370v2](https://arxiv.org/pdf/2602.05370v2)**

> **作者:** Jun Rao; Zixiong Yu; Xuebo Liu; Guhan Chen; Jing Li; Jiansheng Wei; Xiaojun Meng; Min Zhang
>
> **摘要:** Iterative Direct Preference Optimization has emerged as the state-of-the-art paradigm for aligning Large Language Models on reasoning tasks. Standard implementations (DPO-R1) rely on Best-of-N sampling (e.g., $N \ge 8$) to mine golden trajectories from the distribution tail. In this paper, we challenge this scaling hypothesis and reveal a counter-intuitive phenomenon: in mathematical reasoning, aggressive exploration yields diminishing returns and even catastrophic policy collapse. We theoretically demonstrate that scaling $N$ amplifies verifier noise and induces detrimental distribution shifts. To resolve this, we introduce \textbf{PACE} (Proximal Alignment via Corrective Exploration), which replaces brute-force mining with a generation-based corrective strategy. Operating with a minimal budget ($2<N<3$), PACE synthesizes high-fidelity preference pairs from failed explorations. Empirical evaluations show that PACE outperforms DPO-R1 $(N=16)$ while using only about $1/5$ of the compute, demonstrating superior robustness against reward hacking and label noise.
>
---
#### [replaced 024] MAGIC: A Co-Evolving Attacker-Defender Adversarial Game for Robust LLM Safety
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文属于LLM安全任务，旨在解决防御滞后于攻击的问题。提出MAGIC框架，通过对抗游戏实现攻防共进化，提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2602.01539v2](https://arxiv.org/pdf/2602.01539v2)**

> **作者:** Xiaoyu Wen; Zhida He; Han Qi; Ziyu Wan; Zhongtian Ma; Ying Wen; Tianhang Zheng; Xingcheng Xu; Chaochao Lu; Qiaosheng Zhang
>
> **摘要:** Ensuring robust safety alignment is crucial for Large Language Models (LLMs), yet existing defenses often lag behind evolving adversarial attacks due to their \textbf{reliance on static, pre-collected data distributions}. In this paper, we introduce \textbf{MAGIC}, a novel multi-turn multi-agent reinforcement learning framework that formulates LLM safety alignment as an adversarial asymmetric game. Specifically, an attacker agent learns to iteratively rewrite original queries into deceptive prompts, while a defender agent simultaneously optimizes its policy to recognize and refuse such inputs. This dynamic process triggers a \textbf{co-evolution}, where the attacker's ever-changing strategies continuously uncover long-tail vulnerabilities, driving the defender to generalize to unseen attack patterns. Remarkably, we observe that the attacker, endowed with initial reasoning ability, evolves \textbf{novel, previously unseen combinatorial strategies} through iterative RL training, underscoring our method's substantial potential. Theoretically, we provide insights into a more robust game equilibrium and derive safety guarantees. Extensive experiments validate our framework's effectiveness, demonstrating superior defense success rates without compromising the helpfulness of the model. Our code is available at https://github.com/BattleWen/MAGIC.
>
---
#### [replaced 025] A.X K1 Technical Report
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍A.X K1，一个519B参数的MoE语言模型，解决推理能力与效率平衡问题，通过Think-Fusion训练实现模式切换，提升多场景部署效果。**

- **链接: [https://arxiv.org/pdf/2601.09200v4](https://arxiv.org/pdf/2601.09200v4)**

> **作者:** Sung Jun Cheon; Jaekyung Cho; Seongho Choi; Hyunjun Eun; Seokhwan Jo; Jaehyun Jun; Minsoo Kang; Jin Kim; Jiwon Kim; Minsang Kim; Sungwan Kim; Seungsik Kim; Tae Yoon Kim; Youngrang Kim; Hyeongmun Lee; Sangyeol Lee; Sungeun Lee; Youngsoon Lee; Yujin Lee; Seongmin Ok; Chanyong Park; Hyewoong Park; Junyoung Park; Hyunho Yang; Subin Yi; Soohyun Bae; Dhammiko Arya; Yongseok Choi; Sangho Choi; Dongyeon Cho; Seungmo Cho; Gyoungeun Han; Yong-jin Han; Seokyoung Hong; Hyeon Hwang; Wonbeom Jang; Minjeong Ju; Wonjin Jung; Keummin Ka; Sungil Kang; Dongnam Kim; Joonghoon Kim; Jonghwi Kim; SaeRom Kim; Sangjin Kim; Seongwon Kim; Youngjin Kim; Seojin Lee; Sunwoo Lee; Taehoon Lee; Chanwoo Park; Sohee Park; Sooyeon Park; Yohan Ra; Sereimony Sek; Seungyeon Seo; Gun Song; Sanghoon Woo; Janghan Yoon; Sungbin Yoon
>
> **摘要:** We introduce A.X K1, a 519B-parameter Mixture-of-Experts (MoE) language model trained from scratch. Our design leverages scaling laws to optimize training configurations and vocabulary size under fixed computational budgets. A.X K1 is pre-trained on a corpus of approximately 10T tokens, curated by a multi-stage data processing pipeline. Designed to bridge the gap between reasoning capability and inference efficiency, A.X K1 supports explicitly controllable reasoning to facilitate scalable deployment across diverse real-world scenarios. We propose a simple yet effective Think-Fusion training recipe, enabling user-controlled switching between thinking and non-thinking modes within a single unified model. Extensive evaluations demonstrate that A.X K1 achieves performance competitive with leading open-source models, while establishing a distinctive advantage in Korean-language benchmarks.
>
---
#### [replaced 026] Scalable Multi-Stage Influence Function for Large Language Models via Eigenvalue-Corrected Kronecker-Factored Parameterization
- **分类: cs.CL**

- **简介: 该论文属于模型解释任务，旨在解决大语言模型预测归因问题。提出多阶段影响函数与EK-FAC参数化，提升可扩展性和效率。**

- **链接: [https://arxiv.org/pdf/2505.05017v2](https://arxiv.org/pdf/2505.05017v2)**

> **作者:** Yuntai Bao; Xuhong Zhang; Tianyu Du; Xinkui Zhao; Jiang Zong; Hao Peng; Jianwei Yin
>
> **备注:** 17 pages, 4 figures; accepted by IJCAI 2025
>
> **摘要:** Pre-trained large language models (LLMs) are commonly fine-tuned to adapt to downstream tasks. Since the majority of knowledge is acquired during pre-training, attributing the predictions of fine-tuned LLMs to their pre-training data may provide valuable insights. Influence functions have been proposed as a means to explain model predictions based on training data. However, existing approaches fail to compute ``multi-stage'' influence and lack scalability to billion-scale LLMs. In this paper, we propose the multi-stage influence function to attribute the downstream predictions of fine-tuned LLMs to pre-training data under the full-parameter fine-tuning paradigm. To enhance the efficiency and practicality of our multi-stage influence function, we leverage Eigenvalue-corrected Kronecker-Factored (EK-FAC) parameterization for efficient approximation. Empirical results validate the superior scalability of EK-FAC approximation and the effectiveness of our multi-stage influence function. Additionally, case studies on a real-world LLM, dolly-v2-3b, demonstrate its interpretive power, with exemplars illustrating insights provided by multi-stage influence estimates. Our code is public at https://github.com/colored-dye/multi_stage_influence_function.
>
---
#### [replaced 027] FLAT-LLM: Fine-grained Low-rank Activation Space Transformation for Large Language Model Compression
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型在资源受限环境中的部署问题。通过细粒度低秩激活空间变换，实现高效准确的结构压缩。**

- **链接: [https://arxiv.org/pdf/2505.23966v4](https://arxiv.org/pdf/2505.23966v4)**

> **作者:** Jiayi Tian; Ryan Solgi; Jinming Lu; Yifan Yang; Hai Li; Zheng Zhang
>
> **摘要:** Large Language Models (LLMs) have enabled remarkable progress in natural language processing, yet their high computational and memory demands pose challenges for deployment in resource-constrained environments. Although recent low-rank decomposition methods offer a promising path for structural compression, they often suffer from accuracy degradation, expensive calibration procedures, and result in inefficient model architectures that hinder real-world inference speedups. In this paper, we propose FLAT-LLM, a fast and accurate, training-free structural compression method based on fine-grained low-rank transformations in the activation space. Specifically, we reduce the hidden dimension by transforming the weights using truncated eigenvectors computed via head-wise Principal Component Analysis, and employ a greedy budget redistribution strategy to adaptively allocate ranks across decoders. FLAT-LLM achieves efficient and effective weight compression without recovery fine-tuning, which could complete the calibration within a few minutes. Evaluated across 5 models and 11 datasets, FLAT-LLM outperforms structural pruning baselines in generalization and downstream performance, while delivering inference speedups over decomposition-based methods.
>
---
#### [replaced 028] Hyperbolic Fine-Tuning for Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.NE**

- **简介: 该论文属于自然语言处理领域，旨在解决大语言模型的嵌入空间适配问题。通过分析token分布特性，提出HypLoRA方法，在双曲空间中进行高效微调，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2410.04010v2](https://arxiv.org/pdf/2410.04010v2)**

> **作者:** Menglin Yang; Ram Samarth B B; Aosong Feng; Bo Xiong; Jihong Liu; Irwin King; Rex Ying
>
> **备注:** NeurIPS 2025; https://github.com/marlin-codes/HypLoRA
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable performance across various tasks. However, it remains an open question whether the default Euclidean space is the most suitable choice for LLMs. In this study, we investigate the geometric characteristics of LLMs, focusing specifically on tokens and their embeddings. Our findings reveal that token frequency follows a power-law distribution, where high-frequency tokens (e.g., the, that ) constitute the minority, while low-frequency tokens (e.g., apple, dog) constitute the majority. Furthermore, high-frequency tokens cluster near the origin, whereas low-frequency tokens are positioned farther away in the embedding space. Additionally, token embeddings exhibit hyperbolic characteristics, indicating a latent tree-like structure within the embedding space. Motivated by these observations, we propose HypLoRA, an efficient fine-tuning approach that operates in hyperbolic space to exploit these underlying hierarchical structures better. HypLoRA performs low-rank adaptation directly in hyperbolic space, thereby preserving hyperbolic modeling capabilities throughout the fine-tuning process. Extensive experiments across various base models and reasoning benchmarks, specifically arithmetic and commonsense reasoning tasks, demonstrate that HypLoRA substantially improves LLM performance.
>
---
#### [replaced 029] Investigating Disability Representations in Text-to-Image Models
- **分类: cs.CL; cs.CV; cs.CY; cs.HC**

- **简介: 该论文属于AI生成图像的伦理研究任务，旨在解决残疾群体在文本到图像模型中的代表性问题。通过分析不同提示下的图像，评估模型的偏见并提出改进策略。**

- **链接: [https://arxiv.org/pdf/2602.04687v2](https://arxiv.org/pdf/2602.04687v2)**

> **作者:** Yang Yian; Yu Fan; Liudmila Zavolokina; Sarah Ebling
>
> **备注:** 21 pages, 9 figures. References included
>
> **摘要:** Text-to-image generative models have made remarkable progress in producing high-quality visual content from textual descriptions, yet concerns remain about how they represent social groups. While characteristics like gender and race have received increasing attention, disability representations remain underexplored. This study investigates how people with disabilities are represented in AI-generated images by analyzing outputs from Stable Diffusion XL and DALL-E 3 using a structured prompt design. We analyze disability representations by comparing image similarities between generic disability prompts and prompts referring to specific disability categories. Moreover, we evaluate how mitigation strategies influence disability portrayals, with a focus on assessing affective framing through sentiment polarity analysis, combining both automatic and human evaluation. Our findings reveal persistent representational imbalances and highlight the need for continuous evaluation and refinement of generative models to foster more diverse and inclusive portrayals of disability.
>
---
#### [replaced 030] DimABSA: Building Multilingual and Multidomain Datasets for Dimensional Aspect-Based Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文属于多语言、多领域的情感分析任务，旨在解决传统分类标签无法捕捉细微情感的问题。通过引入连续的维度情感评分，构建了DimABSA数据集，并提出新评估指标cF1。**

- **链接: [https://arxiv.org/pdf/2601.23022v2](https://arxiv.org/pdf/2601.23022v2)**

> **作者:** Lung-Hao Lee; Liang-Chih Yu; Natalia Loukashevich; Ilseyar Alimova; Alexander Panchenko; Tzu-Mi Lin; Zhe-Yu Xu; Jian-Yu Zhou; Guangmin Zheng; Jin Wang; Sharanya Awasthi; Jonas Becker; Jan Philip Wahle; Terry Ruas; Shamsuddeen Hassan Muhammad; Saif M. Mohammad
>
> **摘要:** Aspect-Based Sentiment Analysis (ABSA) focuses on extracting sentiment at a fine-grained aspect level and has been widely applied across real-world domains. However, existing ABSA research relies on coarse-grained categorical labels (e.g., positive, negative), which limits its ability to capture nuanced affective states. To address this limitation, we adopt a dimensional approach that represents sentiment with continuous valence-arousal (VA) scores, enabling fine-grained analysis at both the aspect and sentiment levels. To this end, we introduce DimABSA, the first multilingual, dimensional ABSA resource annotated with both traditional ABSA elements (aspect terms, aspect categories, and opinion terms) and newly introduced VA scores. This resource contains 76,958 aspect instances across 42,590 sentences, spanning six languages and four domains. We further introduce three subtasks that combine VA scores with different ABSA elements, providing a bridge from traditional ABSA to dimensional ABSA. Given that these subtasks involve both categorical and continuous outputs, we propose a new unified metric, continuous F1 (cF1), which incorporates VA prediction error into standard F1. We provide a comprehensive benchmark using both prompted and fine-tuned large language models across all subtasks. Our results show that DimABSA is a challenging benchmark and provides a foundation for advancing multilingual dimensional ABSA.
>
---
#### [replaced 031] Scoring, Reasoning, and Selecting the Best! Ensembling Large Language Models via a Peer-Review Process
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LLM-PeerReview，用于集成多个大语言模型的输出。任务是提升问答和推理性能，通过评分、推理和选择最佳响应解决模型多样性问题。**

- **链接: [https://arxiv.org/pdf/2512.23213v2](https://arxiv.org/pdf/2512.23213v2)**

> **作者:** Zhijun Chen; Zeyu Ji; Qianren Mao; Hao Wu; Junhang Cheng; Bangjie Qin; Zhuoran Li; Jingzheng Li; Kai Sun; Zizhe Wang; Yikun Ban; Zhu Sun; Xiangyang Ji; Hailong Sun
>
> **摘要:** We propose LLM-PeerReview, an unsupervised LLM Ensemble method that selects the most ideal response from multiple LLM-generated candidates for each query, harnessing the collective wisdom of multiple models with diverse strengths. LLM-PeerReview is built on a novel, peer-review-inspired framework that offers a transparent and interpretable mechanism, while remaining fully unsupervised for flexible adaptability and generalization. Specifically, it operates in three stages: For scoring, we use the emerging LLM-as-a-Judge technique to evaluate each response by reusing multiple LLMs at hand; For reasoning, we can apply a straightforward averaging strategy or a principled graphical model-based truth inference algorithm to aggregate multiple scores to produce a final score for each response; Finally, the highest-scoring response is selected as the best ensemble output. LLM-PeerReview is conceptually simple and empirically powerful. Our results across four datasets show that the two variants of the proposed approach outperform the advanced model Smoothie-Global by 6.9% and 7.3% points, cross diverse task types including factual recall QA, math reasoning, and instruction following.
>
---
#### [replaced 032] You Had One Job: Per-Task Quantization Using LLMs' Hidden Representations
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决量化过程中精度分配不合理的问题。通过分析隐藏表示，提出方法按任务需求分配不同精度，提升压缩效率。**

- **链接: [https://arxiv.org/pdf/2511.06516v2](https://arxiv.org/pdf/2511.06516v2)**

> **作者:** Amit LeVi; Raz Lapid; Rom Himelstein; Chaim Baskin; Ravid Shwartz Ziv; Avi Mendelson
>
> **摘要:** Many applications of large language models (LLMs) require only a narrow capability, yet common post-training quantization (PTQ) pipelines assign precision largely without regard to the target task. As a result, they may spend bits on layers that are less relevant to the task. We propose per-task mixed-precision PTQ guided by hidden representations. Given a small set of unlabeled calibration prompts from the target task, we estimate layer importance and allocate higher precision to task-relevant layers while lower to the rest, under a bits allocation budget. We introduce three task-aware allocation signals: \textbf{TAQ}, which scores layers using an information-stability criterion derived from activation geometry; \textbf{TAQO}, which ranks layers by direct sensitivity to single-layer quantization; and \textbf{TAQ-KL}, which measures output sensitivity via KL divergence under a noise proxy for quantization error. Together, these methods provide a simple, post-training framework that connects mechanistic signals to quantization decisions, enabling task-aligned compression without additional training.
>
---
#### [replaced 033] Bridging Symbolic Control and Neural Reasoning in LLM Agents: Structured Cognitive Loop with a Governance Layer
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SCL架构，解决LLM代理的推理与执行纠缠、记忆不稳定和动作不可控问题，通过分阶段结构和符号控制提升可解释性与可控性。**

- **链接: [https://arxiv.org/pdf/2511.17673v4](https://arxiv.org/pdf/2511.17673v4)**

> **作者:** Myung Ho Kim
>
> **备注:** This revised version strengthens the architectural clarity and conceptual coherence of the manuscript. In particular, it formalizes Soft Symbolic Control as a dedicated Governance layer distinct from the R-CCAM loop, clarifying its structural role beyond the earlier meta-prompt add-on formulation
>
> **摘要:** Large language model agents suffer from fundamental architectural problems: entangled reasoning and execution, memory volatility, and uncontrolled action sequences. We introduce Structured Cognitive Loop (SCL), a modular architecture that explicitly separates agent cognition into five phases: Retrieval, Cognition, Control, Action, and Memory (R-CCAM). Soft Symbolic Control constitutes a dedicated governance layer within SCL, applying symbolic constraints to probabilistic inference while preserving the flexibility of neural reasoning and restoring the explainability and controllability of classical symbolic systems. Through empirical validation on multi-step conditional reasoning tasks, we demonstrate that SCL achieves zero policy violations, eliminates redundant tool calls, and maintains complete decision traceability. These results address critical gaps in existing frameworks such as ReAct, AutoGPT, and memory-augmented approaches. Our contributions are threefold: (1) we situate SCL within the taxonomy of hybrid intelligence, differentiating it from prompt-centric and memory-only approaches; (2) we formally define Soft Symbolic Control and contrast it with neuro-symbolic AI; and (3) we derive three design principles for trustworthy agents: modular decomposition, adaptive symbolic governance, and transparent state management. We provide a complete open-source implementation demonstrating the R-CCAM loop architecture, alongside a live GPT-4o-powered travel planning agent. By connecting expert system principles with modern LLM capabilities, this work offers a practical and theoretically grounded path toward reliable, explainable, and governable AI agents.
>
---
#### [replaced 034] FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在预填充和解码阶段的计算与内存效率问题。通过分离上下文压缩与KV缓存优化，提升处理速度并保持精度。**

- **链接: [https://arxiv.org/pdf/2502.01068v5](https://arxiv.org/pdf/2502.01068v5)**

> **作者:** Dongwon Jo; Jiwon Song; Yulhwa Kim; Jae-Joon Kim
>
> **摘要:** While large language models (LLMs) excel at handling long-context sequences, they require substantial prefill computation and key-value (KV) cache, which can heavily burden computational efficiency and memory usage in both prefill and decoding stages. Recent works that compress KV caches with prefill acceleration reduce this cost but inadvertently tie the prefill compute reduction to the decoding KV budget. This coupling arises from overlooking the layer-dependent variation of critical context, often leading to accuracy degradation. To address this issue, we introduce FastKV, a KV cache compression framework designed to reduce latency in both prefill and decoding by leveraging the stabilization of token importance in later layers. FastKV performs full-context computation until a Token-Selective Propagation (TSP) layer, which forwards only the most informative tokens to subsequent layers. From these propagated tokens, FastKV independently selects salient KV entries for caching, thereby decoupling KV budget from the prefill compute reduction based on the TSP decision. This independent control of the TSP rate and KV retention rate enables flexible optimization of efficiency and accuracy. Experimental results show that FastKV achieves speedups of up to 1.82$\times$ in prefill and 2.87$\times$ in decoding compared to the full-context baseline, while matching the accuracy of the baselines that only accelerate the decoding stage. Our code is available at https://github.com/dongwonjo/FastKV.
>
---
#### [replaced 035] SWE-Dev: Evaluating and Training Autonomous Feature-Driven Software Development
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出SWE-Dev数据集，用于评估和训练自主编码系统在真实世界功能驱动开发任务中的能力。**

- **链接: [https://arxiv.org/pdf/2505.16975v3](https://arxiv.org/pdf/2505.16975v3)**

> **作者:** Yaxin Du; Yuzhu Cai; Yifan Zhou; Cheng Wang; Yu Qian; Xianghe Pang; Qian Liu; Yue Hu; Siheng Chen
>
> **摘要:** Large Language Models (LLMs) have shown strong capability in diverse software engineering tasks. However, feature-driven development, a highly prevalent real-world task that involves developing new functionalities for large, existing codebases, remains underexplored. We therefore introduce SWE-Dev, the first large-scale dataset (with 14,000 training and 500 test samples) designed to evaluate and train autonomous coding systems on real-world end-to-end feature-driven software development tasks. To ensure verifiable and diverse training, SWE-Dev uniquely provides all instances with a runnable environment and its developer-authored executable unit tests. This collection not only provides high-quality data for Supervised Fine-Tuning (SFT), but also enables Reinforcement Learning (RL) by delivering accurate reward signals from executable unit tests. We evaluated SWE-Dev across 17 base LLMs, 10 reasoning-focused LLMs, 10 multi-agent systems, and 8 tool-augmented LLM agents. Results show substantial headroom: the best single-turn model reaches only 22.51\% Pass@1 on the hard split, while OpenHands agents improve to 56.44\% but still leave many tasks unsolved. Code is available here https://github.com/DorothyDUUU/SWE-Dev.
>
---
#### [replaced 036] Let LLMs Speak Embedding Languages: Generative Text Embeddings via Iterative Contrastive Refinement
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出GIRCSE框架，解决LLM生成嵌入表示的问题。通过迭代对比优化，提升语义表示质量，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2509.24291v2](https://arxiv.org/pdf/2509.24291v2)**

> **作者:** Yu-Che Tsai; Kuan-Yu Chen; Yuan-Chi Li; Yuan-Hao Chen; Ching-Yu Tsai; Shou-De Lin
>
> **摘要:** Existing large language model (LLM)-based embeddings typically adopt an encoder-only paradigm, treating LLMs as static feature extractors and overlooking their core generative strengths. We introduce GIRCSE (Generative Iterative Refinement for Contrastive Sentence Embeddings), a novel framework that leverages autoregressive generation to iteratively refine semantic representations. By producing sequences of soft tokens optimized under contrastive objective, GIRCSE captures latent concepts and implicit semantics that encoder-only methods often miss. To guide this process, we propose an Iterative Contrastive Refinement (ICR) objective that encourages each refinement step to yield better representations. Extensive experiments show that GIRCSE outperforms strong LLM-based embedding baselines on the MTEB benchmark and instruction-following tasks. Moreover, GIRCSE exhibits an emergent test-time scaling property: generating more tokens at inference steadily improves embedding quality. Our results establish generative iterative refinement as a new paradigm for representation learning.
>
---
#### [replaced 037] OmniCode: A Benchmark for Evaluating Software Engineering Agents
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出OmniCode基准，用于评估软件工程代理。解决现有基准任务狭窄的问题，涵盖更广泛的软件开发任务，包括错误修复、测试生成等，提升代理的全面能力。**

- **链接: [https://arxiv.org/pdf/2602.02262v2](https://arxiv.org/pdf/2602.02262v2)**

> **作者:** Atharv Sonwane; Eng-Shen Tu; Wei-Chung Lu; Claas Beger; Carter Larsen; Debjit Dhar; Simon Alford; Rachel Chen; Ronit Pattanayak; Tuan Anh Dang; Guohao Chen; Gloria Geng; Kevin Ellis; Saikat Dutta
>
> **摘要:** LLM-powered coding agents are redefining how real-world software is developed. To drive the research towards better coding agents, we require challenging benchmarks that can rigorously evaluate the ability of such agents to perform various software engineering tasks. However, popular coding benchmarks such as HumanEval and SWE-Bench focus on narrowly scoped tasks such as competition programming and patch generation. In reality, software engineers have to handle a broader set of tasks for real-world software development. To address this gap, we propose OmniCode, a novel software engineering benchmark that contains a broader and more diverse set of task categories beyond code or patch generation. Overall, OmniCode contains 1794 tasks spanning three programming languages (Python, Java, and C++) and four key categories: bug fixing, test generation, code review fixing, and style fixing. In contrast to prior software engineering benchmarks, the tasks in OmniCode are (1) manually validated to eliminate ill-defined problems, and (2) synthetically crafted or recently curated to avoid data leakage issues, presenting a new framework for synthetically generating diverse software tasks from limited real-world data. We evaluate OmniCode with popular agent frameworks such as SWE-Agent and show that while they may perform well on bug fixing for Python, they fall short on tasks such as Test Generation and in languages such as C++ and Java. For instance, SWE-Agent achieves a maximum of 20.9% with DeepSeek-V3.1 on Java Test Generation tasks. OmniCode aims to serve as a robust benchmark and spur the development of agents that can perform well across different aspects of software development. Code and data are available at https://github.com/seal-research/OmniCode.
>
---
#### [replaced 038] Efficient LLM Moderation with Multi-Layer Latent Prototypes
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于语言模型安全任务，解决部署时有害输出的监管问题。提出MLPM，通过多层原型提升 moderation 效率与可定制性。**

- **链接: [https://arxiv.org/pdf/2502.16174v3](https://arxiv.org/pdf/2502.16174v3)**

> **作者:** Maciej Chrabąszcz; Filip Szatkowski; Bartosz Wójcik; Jan Dubiński; Tomasz Trzciński; Sebastian Cygert
>
> **摘要:** Although modern LLMs are aligned with human values during post-training, robust moderation remains essential to prevent harmful outputs at deployment time. Existing approaches suffer from performance-efficiency trade-offs and are difficult to customize to user-specific requirements. Motivated by this gap, we introduce Multi-Layer Prototype Moderator (MLPM), a lightweight and highly customizable input moderation tool. We propose leveraging prototypes of intermediate representations across multiple layers to improve moderation quality while maintaining high efficiency. By design, our method adds negligible overhead to the generation pipeline and can be seamlessly applied to any model. MLPM achieves state-of-the-art performance on diverse moderation benchmarks and demonstrates strong scalability across model families of various sizes. Moreover, we show that it integrates smoothly into end-to-end moderation pipelines and further improves response safety when combined with output moderation techniques. Overall, our work provides a practical and adaptable solution for safe, robust, and efficient LLM deployment.
>
---
#### [replaced 039] Simulated Adoption: Decoupling Magnitude and Direction in LLM In-Context Conflict Resolution
- **分类: cs.LG; cs.CL; cs.CY**

- **简介: 该论文属于语言模型研究任务，旨在解决LLM在上下文冲突中的知识冲突问题。通过分析残差流的几何特性，揭示了模型通过几何位移而非幅度抑制来处理冲突，挑战了现有检测方法。**

- **链接: [https://arxiv.org/pdf/2602.04918v2](https://arxiv.org/pdf/2602.04918v2)**

> **作者:** Long Zhang; Fangwei Lin
>
> **摘要:** Large Language Models (LLMs) frequently prioritize conflicting in-context information over pre-existing parametric memory, a phenomenon often termed sycophancy or compliance. However, the mechanistic realization of this behavior remains obscure, specifically how the model resolves these knowledge conflicts through compliance, and whether this suppression arises from signal magnitude dilution or directional geometric alteration within the residual stream. To resolve this, we conducted a layer-wise geometric analysis across Qwen-3-4B, Llama-3.1-8B, and GLM-4-9B, decomposing the residual stream updates induced by counter-factual contexts into radial (norm-based) and angular (cosine-based) components. Our empirical results reject the universality of the "Manifold Dilution" hypothesis, as two of the three architectures maintained stable residual norms despite exhibiting significant performance degradation on factual queries. Instead, we observed that compliance is consistently characterized by "Orthogonal Interference," where the conflicting context injects a steering vector that is quasi-orthogonal to the ground-truth direction, effectively rotating the hidden state representation. This suggests that models do not "unlearn" or suppress the magnitude of internal truths but rather employ a mechanism of geometric displacement to bypass the correct unembedding vector, effectively simulating adoption while preserving the original structural magnitude. These findings challenge scalar confidence metrics for detecting hallucinations and underscore the necessity of vectorial monitoring to distinguish between genuine knowledge integration and superficial in-context mimicry.
>
---
#### [replaced 040] DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于文档问答任务，旨在解决长文档信息检索与理解问题。提出DeepRead框架，通过结构感知的多轮推理，提升问答效果。**

- **链接: [https://arxiv.org/pdf/2602.05014v2](https://arxiv.org/pdf/2602.05014v2)**

> **作者:** Zhanli Li; Huiwen Tian; Lvzhou Luo; Yixuan Cao; Ping Luo
>
> **备注:** This work is currently in progress
>
> **摘要:** With the rapid progress of tool-using and agentic large language models (LLMs), Retrieval-Augmented Generation (RAG) is evolving from one-shot, passive retrieval into multi-turn, decision-driven evidence acquisition. Despite strong results in open-domain settings, existing agentic search frameworks commonly treat long documents as flat collections of chunks, underutilizing document-native priors such as hierarchical organization and sequential discourse structure. We introduce DeepRead, a structure-aware, multi-turn document reasoning agent that explicitly operationalizes these priors for long-document question answering. DeepRead leverages LLM-based OCR model to convert PDFs into structured Markdown that preserves headings and paragraph boundaries. It then indexes documents at the paragraph level and assigns each paragraph a coordinate-style metadata key encoding its section identity and in-section order. Building on this representation, DeepRead equips the LLM with two complementary tools: a Retrieve tool that localizes relevant paragraphs while exposing their structural coordinates (with lightweight scanning context), and a ReadSection tool that enables contiguous, order-preserving reading within a specified section and paragraph range. Our experiments demonstrate that DeepRead achieves significant improvements over Search-o1-style agentic search in document question answering. The synergistic effect between retrieval and reading tools is also validated. Our fine-grained behavioral analysis reveals a reading and reasoning paradigm resembling human-like ``locate then read'' behavior.
>
---
#### [replaced 041] LeWiDi-2025 at NLPerspectives: Third Edition of the Learning with Disagreements Shared Task
- **分类: cs.CL**

- **简介: 该论文属于"Learning With Disagreements"共享任务，旨在提升AI模型对人类判断差异的感知能力。通过扩展数据集和引入新评估方法，研究解决模型在不同任务中处理意见分歧的问题。**

- **链接: [https://arxiv.org/pdf/2510.08460v2](https://arxiv.org/pdf/2510.08460v2)**

> **作者:** Elisa Leonardelli; Silvia Casola; Siyao Peng; Giulia Rizzi; Valerio Basile; Elisabetta Fersini; Diego Frassinelli; Hyewon Jang; Maja Pavlovic; Barbara Plank; Massimo Poesio
>
> **备注:** 14 pages; LeWiDi-2025 shared task description paper at NLPerspective workshop at EMNLP 2025
>
> **摘要:** Many researchers have reached the conclusion that AI models should be trained to be aware of the possibility of variation and disagreement in human judgments, and evaluated as per their ability to recognize such variation. The LEWIDI series of shared tasks on Learning With Disagreements was established to promote this approach to training and evaluating AI models, by making suitable datasets more accessible and by developing evaluation methods. The third edition of the task builds on this goal by extending the LEWIDI benchmark to four datasets spanning paraphrase identification, irony detection, sarcasm detection, and natural language inference, with labeling schemes that include not only categorical judgments as in previous editions, but ordinal judgments as well. Another novelty is that we adopt two complementary paradigms to evaluate disagreement-aware systems: the soft-label approach, in which models predict population-level distributions of judgments, and the perspectivist approach, in which models predict the interpretations of individual annotators. Crucially, we moved beyond standard metrics such as cross-entropy, and tested new evaluation metrics for the two paradigms. The task attracted diverse participation, and the results provide insights into the strengths and limitations of methods to modeling variation. Together, these contributions strengthen LEWIDI as a framework and provide new resources, benchmarks, and findings to support the development of disagreement-aware technologies.
>
---
#### [replaced 042] Estimating Semantic Alphabet Size for LLM Uncertainty Quantification
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于不确定性量化任务，旨在解决LLM采样计算成本高的问题。通过改进语义字母大小估计，提升DSE的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2509.14478v2](https://arxiv.org/pdf/2509.14478v2)**

> **作者:** Lucas H. McCabe; Rimon Melamed; Thomas Hartvigsen; H. Howie Huang
>
> **摘要:** Many black-box techniques for quantifying the uncertainty of large language models (LLMs) rely on repeated LLM sampling, which can be computationally expensive. Therefore, practical applicability demands reliable estimation from few samples. Semantic entropy (SE) is a popular sample-based uncertainty estimator with a discrete formulation attractive for the black-box setting. Recent extensions of SE exhibit improved LLM hallucination detection, but do so with less interpretable methods that admit additional hyperparameters. For this reason, we revisit the canonical discrete semantic entropy (DSE) estimator, finding that it underestimates the "true" semantic entropy, as expected from theory. We propose a modified semantic alphabet size estimator, and illustrate that using it to adjust DSE for sample coverage results in more accurate SE estimation in our setting of interest. Furthermore, we find that two semantic alphabet size estimators, including our proposed, flag incorrect LLM responses as well or better than many top-performing alternatives, with the added benefit of remaining highly interpretable.
>
---
#### [replaced 043] SafeCOMM: A Study on Safety Degradation in Fine-Tuned Telecom Large Language Models
- **分类: cs.CY; cs.CL; cs.CR; cs.LG**

- **简介: 该论文属于安全对齐任务，研究细调电信大模型时安全性能下降的问题。通过实验和基准测试，提出防御方法以恢复安全性和任务性能。**

- **链接: [https://arxiv.org/pdf/2506.00062v3](https://arxiv.org/pdf/2506.00062v3)**

> **作者:** Aladin Djuhera; Swanand Ravindra Kadhe; Farhan Ahmed; Syed Zawad; Fernando Koch; Walid Saad; Holger Boche
>
> **摘要:** Fine-tuning large language models (LLMs) on telecom datasets is a common practice to adapt general-purpose models to the telecom domain. However, little attention has been paid to how this process may compromise model safety. Recent research has shown that even benign fine-tuning can degrade the safety alignment of LLMs, causing them to respond to harmful or unethical user queries. In this paper, we investigate this issue by fine-tuning LLMs on three representative telecom datasets and show that safety degrades even for light telecom domain adaptation. To this end, we introduce TeleHarm, the first telecom-specific red-teaming benchmark, which we use alongside established DirectHarm and HexPhi datasets to systematically assess harmful behavior. We further extend our analysis to publicly available TeleLLMs that were continually pre-trained on large telecom corpora, revealing that safety alignment is severely lacking, primarily due to the omission of safety-focused instruction tuning. To address these issues, we evaluate three realignment defenses: SafeInstruct, SafeLoRA, SafeMERGE. We show that, across all settings, the proposed defenses can effectively restore safety without compromising telecom task performance, leading to Safe teleCOMMunication (SafeCOMM) models. Our work serves as both a diagnostic study and practical guide for safety realignment in telecom-tuned LLMs, underscoring the need for safety-aware instruction and fine-tuning in the telecom domain.
>
---
#### [replaced 044] STAR: Stepwise Task Augmentation with Relation Learning for Aspect Sentiment Quad Prediction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于方面情感四元组预测任务，解决数据稀缺下情感元素依赖建模难题。提出STAR框架，通过分步任务增强和关系学习提升预测性能。**

- **链接: [https://arxiv.org/pdf/2501.16093v2](https://arxiv.org/pdf/2501.16093v2)**

> **作者:** Wenna Lai; Haoran Xie; Guandong Xu; Qing Li
>
> **备注:** 17 pages, 6 figures, and 7 tables
>
> **摘要:** Aspect-based sentiment analysis (ABSA) aims to identify four sentiment elements, including aspect term, aspect category, opinion term, and sentiment polarity. These elements construct a complete picture of sentiments. The most challenging task, aspect sentiment quad prediction (ASQP), requires predicting all four elements simultaneously and is hindered by the difficulty of accurately modeling dependencies among sentiment elements. A key challenge lies in the scarcity of annotated data, which limits the model ability to understand and reason about the relational dependencies required for effective quad prediction. To address this challenge, we propose a stepwise task augmentation framework with relation learning that decomposes ASQP into a sequence of auxiliary subtasks with increasing relational granularity. Specifically, STAR incrementally constructs auxiliary data by augmenting the training data with pairwise and overall relation tasks, enabling the model to capture and compose sentiment dependencies in a stepwise manner. This stepwise formulation provides effective relational learning signals that enhance quad prediction performance, particularly in low-resource scenarios. Extensive experiments across four benchmark datasets demonstrate that STAR consistently outperforms existing methods, achieving average F1 improvements of over $2\%$ under low-resource conditions.
>
---
#### [replaced 045] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言生成任务，解决长文本生成中速度与质量的平衡问题。提出FS-DFM模型，在少量采样步骤下实现快速且准确的生成。**

- **链接: [https://arxiv.org/pdf/2509.20624v4](https://arxiv.org/pdf/2509.20624v4)**

> **作者:** Amin Karimi Monsefi; Nikhil Bhendawade; Manuel Rafael Ciosici; Dominic Culver; Yizhe Zhang; Irina Belousova
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Autoregressive language models (ARMs) deliver strong likelihoods, but are inherently serial: they generate one token per forward pass, which limits throughput and inflates latency for long sequences. Diffusion Language Models (DLMs) parallelize across positions and thus appear promising for language generation, yet standard discrete diffusion typically needs hundreds to thousands of model evaluations to reach high quality, trading serial depth for iterative breadth. We introduce FS-DFM, Few-Step Discrete Flow-Matching. A discrete flow-matching model designed for speed without sacrificing quality. The core idea is simple: make the number of sampling steps an explicit parameter and train the model to be consistent across step budgets, so one big move lands where many small moves would. We pair this with a reliable update rule that moves probability in the right direction without overshooting, and with strong teacher guidance distilled from long-run trajectories. Together, these choices make few-step sampling stable, accurate, and easy to control. On language modeling benchmarks, FS-DFM with 8 sampling steps achieves perplexity parity with a 1,024-step discrete-flow baseline for generating 1,024 tokens using a similar-size model, delivering up to 128 times faster sampling and corresponding latency/throughput gains. Code & pretrained checkpoints: https://github.com/apple/ml-fs-dfm
>
---
#### [replaced 046] Large Language Models as Formalizers on Constraint Satisfaction Problems
- **分类: cs.CL**

- **简介: 该论文属于形式化任务，研究如何使用大语言模型作为形式化工具，将问题转化为可求解的程序，而非直接生成答案。**

- **链接: [https://arxiv.org/pdf/2505.13252v3](https://arxiv.org/pdf/2505.13252v3)**

> **作者:** Rikhil Amonkar; Ceyhun Efe Kayan; May Lai; Ronan Le Bras; Li Zhang
>
> **摘要:** An emerging line of recent work advocates for using large language models (LLMs) as formalizers instead of as end-to-end solvers for various types of problems. Instead of generating the solution, the LLM generates a formal program that derives a solution via an external solver. We thoroughly investigate the formalization capability of LLMs on real-life constraint satisfaction problems. On 4 domains, we systematically evaluate 6 LLMs, including 4 large reasoning models with inference-time scaling, paired with 5 pipelines, including 2 types of formalism. We show that in zero-shot settings, LLM-as-formalizer performs on par with the mainstream LLM-as-solver while offering verifiability, interpretability, and robustness. We also observe excessive reasoning tokens and hard-coded solutions scaling with problem complexity, which demonstrates that even the state-of-the-art LLMs have limited ability to generate solutions or formal programs. We present our detailed analysis and actionable remedies to drive future research that improves LLM-as-formalizer.
>
---
#### [replaced 047] ExpressivityBench: Can LLMs Communicate Implicitly?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在评估大语言模型的隐含表达能力。通过构建基准测试框架，分析模型在情感、身份和语气等任务中的表现，揭示其在社会语言信号上的不足。**

- **链接: [https://arxiv.org/pdf/2411.08010v2](https://arxiv.org/pdf/2411.08010v2)**

> **作者:** Joshua Tint; Som Sagar; Aditya Taparia; Kelly Raines; Bimsara Pathiraja; Caleb Liu; Ransalu Senanayake
>
> **备注:** 21 pages, 7 figures
>
> **摘要:** Human communication is often implicit, conveying tone, identity, and intent beyond literal meanings. While large language models have achieved strong performance on explicit tasks such as summarization and reasoning, their capacity for expressivity, or implicit communication, remains underexplored. We introduce \textbf{ExpressivityBench}, a framework for evaluating the expressivity of LLMs using information-theoretic communication models. Our approach quantifies how well LLM-generated text communicates target properties without explicit mention, across nine tasks spanning emotion, identity, and tone. To enable scalable and reproducible evaluation, we employ LLM-based graders validated against human judgments. Our results reveal that while models are adept at expressing affective content, they struggle with sociolinguistic signals, lagging behind human baselines. This study provides a necessary step to evaluate human-like implicit communication, with implications for applications such as education, mental health support, and socially-aware dialogue systems. We provide code and data for our benchmark alongside our paper.
>
---
#### [replaced 048] FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于生成模型任务，解决长上下文生成中的计算效率问题。通过引入FlashBlock机制，减少注意力计算冗余，提升推理速度。**

- **链接: [https://arxiv.org/pdf/2602.05305v2](https://arxiv.org/pdf/2602.05305v2)**

> **作者:** Zhuokun Chen; Jianfei Cai; Bohan Zhuang
>
> **摘要:** Generating long-form content, such as minute-long videos and extended texts, is increasingly important for modern generative models. Block diffusion improves inference efficiency via KV caching and block-wise causal inference and has been widely adopted in diffusion language models and video generation. However, in long-context settings, block diffusion still incurs substantial overhead from repeatedly computing attention over a growing KV cache. We identify an underexplored property of block diffusion: cross-step redundancy of attention within a block. Our analysis shows that attention outputs from tokens outside the current block remain largely stable across diffusion steps, while block-internal attention varies significantly. Based on this observation, we propose FlashBlock, a cached block-external attention mechanism that reuses stable attention output, reducing attention computation and KV cache access without modifying the diffusion process. Moreover, FlashBlock is orthogonal to sparse attention and can be combined as a complementary residual reuse strategy, substantially improving model accuracy under aggressive sparsification. Experiments on diffusion language models and video generation demonstrate up to 1.44$\times$ higher token throughput and up to 1.6$\times$ reduction in attention time, with negligible impact on generation quality. Project page: https://caesarhhh.github.io/FlashBlock/.
>
---
#### [replaced 049] Constrained Group Relative Policy Optimization
- **分类: cs.LG; cs.CL; cs.RO**

- **简介: 该论文提出Constrained GRPO，解决受限策略优化问题。针对GRPO在约束环境下的不足，通过拉格朗日方法引入约束，改进优势估计以稳定约束控制，提升任务成功率与约束满足度。**

- **链接: [https://arxiv.org/pdf/2602.05863v2](https://arxiv.org/pdf/2602.05863v2)**

> **作者:** Roger Girgis; Rodrigue de Schaetzen; Luke Rowe; Azalée Robitaille; Christopher Pal; Liam Paull
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** While Group Relative Policy Optimization (GRPO) has emerged as a scalable framework for critic-free policy learning, extending it to settings with explicit behavioral constraints remains underexplored. We introduce Constrained GRPO, a Lagrangian-based extension of GRPO for constrained policy optimization. Constraints are specified via indicator cost functions, enabling direct optimization of violation rates through a Lagrangian relaxation. We show that a naive multi-component treatment in advantage estimation can break constrained learning: mismatched component-wise standard deviations distort the relative importance of the different objective terms, which in turn corrupts the Lagrangian signal and prevents meaningful constraint enforcement. We formally derive this effect to motivate our scalarized advantage construction that preserves the intended trade-off between reward and constraint terms. Experiments in a toy gridworld confirm the predicted optimization pathology and demonstrate that scalarizing advantages restores stable constraint control. In addition, we evaluate Constrained GRPO on robotics tasks, where it improves constraint satisfaction while increasing task success, establishing a simple and effective recipe for constrained policy optimization in embodied AI domains that increasingly rely on large multimodal foundation models.
>
---
#### [replaced 050] Applying Text Embedding Models for Efficient Analysis in Labeled Property Graphs
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于图数据分析任务，旨在解决属性图中文本信息利用不足的问题。通过引入预训练文本嵌入模型，提升节点分类和关系预测的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2507.10772v4](https://arxiv.org/pdf/2507.10772v4)**

> **作者:** Michal Podstawski
>
> **摘要:** Labeled property graphs often contain rich textual attributes that can enhance analytical tasks when properly leveraged. This work explores the use of pretrained text embedding models to enable efficient semantic analysis in such graphs. By embedding textual node and edge properties, we support downstream tasks including node classification and relation prediction with improved contextual understanding. Our approach integrates language model embeddings into the graph pipeline without altering its structure, demonstrating that textual semantics can significantly enhance the accuracy and interpretability of property graph analysis.
>
---
#### [replaced 051] Back to Basics: Revisiting Exploration in Reinforcement Learning for LLM Reasoning via Generative Probabilities
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM推理中因策略优化导致的多样性不足问题。通过引入ARM机制，提升生成多样性并平衡探索与利用。**

- **链接: [https://arxiv.org/pdf/2602.05281v2](https://arxiv.org/pdf/2602.05281v2)**

> **作者:** Pengyi Li; Elizaveta Goncharova; Andrey Kuznetsov; Ivan Oseledets
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as an indispensable paradigm for enhancing reasoning in Large Language Models (LLMs). However, standard policy optimization methods, such as Group Relative Policy Optimization (GRPO), often converge to low-entropy policies, leading to severe mode collapse and limited output diversity. We analyze this issue from the perspective of sampling probability dynamics, identifying that the standard objective disproportionately reinforces the highest-likelihood paths, thereby suppressing valid alternative reasoning chains. To address this, we propose a novel Advantage Re-weighting Mechanism (ARM) designed to equilibrate the confidence levels across all correct responses. By incorporating Prompt Perplexity and Answer Confidence into the advantage estimation, our method dynamically reshapes the reward signal to attenuate the gradient updates of over-confident reasoning paths, while redistributing probability mass toward under-explored correct solutions. Empirical results demonstrate that our approach significantly enhances generative diversity and response entropy while maintaining competitive accuracy, effectively achieving a superior trade-off between exploration and exploitation in reasoning tasks. Empirical results on Qwen2.5 and DeepSeek models across mathematical and coding benchmarks show that ProGRPO significantly mitigates entropy collapse. Specifically, on Qwen2.5-7B, our method outperforms GRPO by 5.7% in Pass@1 and, notably, by 13.9% in Pass@32, highlighting its superior capability in generating diverse correct reasoning paths.
>
---
#### [replaced 052] Causal Front-Door Adjustment for Robust Jailbreak Attacks on LLMs
- **分类: cs.CL**

- **简介: 该论文属于LLM安全攻击任务，旨在突破模型的安全机制。通过因果分析和稀疏自编码器，提出CFA²框架，实现高效且可解释的越狱攻击。**

- **链接: [https://arxiv.org/pdf/2602.05444v2](https://arxiv.org/pdf/2602.05444v2)**

> **作者:** Yao Zhou; Zeen Song; Wenwen Qiang; Fengge Wu; Shuyi Zhou; Changwen Zheng; Hui Xiong
>
> **摘要:** Safety alignment mechanisms in Large Language Models (LLMs) often operate as latent internal states, obscuring the model's inherent capabilities. Building on this observation, we model the safety mechanism as an unobserved confounder from a causal perspective. Then, we propose the Causal Front-Door Adjustment Attack (CFA{$^2$}) to jailbreak LLM, which is a framework that leverages Pearl's Front-Door Criterion to sever the confounding associations for robust jailbreaking. Specifically, we employ Sparse Autoencoders (SAEs) to physically strip defense-related features, isolating the core task intent. We further reduce computationally expensive marginalization to a deterministic intervention with low inference complexity. Experiments demonstrate that CFA{$^2$} achieves state-of-the-art attack success rates while offering a mechanistic interpretation of the jailbreaking process.
>
---
#### [replaced 053] A Human-in-the-Loop, LLM-Centered Architecture for Knowledge-Graph Question Answering
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于知识图谱问答任务，旨在解决LLM在知识密集型领域中的准确性与可解释性问题。通过人机交互框架，使LLM生成并解释Cypher查询，提升查询的准确性和透明度。**

- **链接: [https://arxiv.org/pdf/2602.05512v2](https://arxiv.org/pdf/2602.05512v2)**

> **作者:** Larissa Pusch; Alexandre Courtiol; Tim Conrad
>
> **摘要:** Large Language Models (LLMs) excel at language understanding but remain limited in knowledge-intensive domains due to hallucinations, outdated information, and limited explainability. Text-based retrieval-augmented generation (RAG) helps ground model outputs in external sources but struggles with multi-hop reasoning. Knowledge Graphs (KGs), in contrast, support precise, explainable querying, yet require a knowledge of query languages. This work introduces an interactive framework in which LLMs generate and explain Cypher graph queries and users iteratively refine them through natural language. Applied to real-world KGs, the framework improves accessibility to complex datasets while preserving factual accuracy and semantic rigor and provides insight into how model performance varies across domains. Our core quantitative evaluation is a 90-query benchmark on a synthetic movie KG that measures query explanation quality and fault detection across multiple LLMs, complemented by two smaller real-life query-generation experiments on a Hyena KG and the MaRDI (Mathematical Research Data Initiative) KG.
>
---
#### [replaced 054] Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决大模型在低成功率数据集上难以提升的问题。通过设计SOAR框架，让模型自动生成课程以促进学习，提升推理能力。**

- **链接: [https://arxiv.org/pdf/2601.18778v2](https://arxiv.org/pdf/2601.18778v2)**

> **作者:** Shobhita Sundaram; John Quan; Ariel Kwiatkowski; Kartik Ahuja; Yann Ollivier; Julia Kempe
>
> **备注:** Blog post: https://ssundaram21.github.io/soar/
>
> **摘要:** Can a model learn to escape its own learning plateau? Reinforcement learning methods for finetuning large reasoning models stall on datasets with low initial success rates, and thus little training signal. We investigate a fundamental question: Can a pretrained LLM leverage latent knowledge to generate an automated curriculum for problems it cannot solve? To explore this, we design SOAR: A self-improvement framework designed to surface these pedagogical signals through meta-RL. A teacher copy of the model proposes synthetic problems for a student copy, and is rewarded with its improvement on a small subset of hard problems. Critically, SOAR grounds the curriculum in measured student progress rather than intrinsic proxy rewards. Our study on the hardest subsets of mathematical benchmarks (0/128 success) reveals three core findings. First, we show that it is possible to realize bi-level meta-RL that unlocks learning under sparse, binary rewards by sharpening a latent capacity of pretrained models to generate useful stepping stones. Second, grounded rewards outperform intrinsic reward schemes used in prior LLM self-play, reliably avoiding the instability and diversity collapse modes they typically exhibit. Third, analyzing the generated questions reveals that structural quality and well-posedness are more critical for learning progress than solution correctness. Our results suggest that the ability to generate useful stepping stones does not require the preexisting ability to actually solve the hard problems, paving a principled path to escape reasoning plateaus without additional curated data.
>
---
#### [replaced 055] inversedMixup: Data Augmentation via Inverting Mixed Embeddings
- **分类: cs.CL**

- **简介: 该论文提出inversedMixup，解决文本数据增强中可控制性与可解释性不足的问题，通过结合Mixup和LLM生成，实现可控且可读的增强样本。**

- **链接: [https://arxiv.org/pdf/2601.21543v2](https://arxiv.org/pdf/2601.21543v2)**

> **作者:** Fanshuang Kong; Richong Zhang; Qiyu Sun; Zhijie Nie; Ting Deng; Chunming Hu
>
> **摘要:** Mixup generates augmented samples by linearly interpolating inputs and labels with a controllable ratio. However, since it operates in the latent embedding level, the resulting samples are not human-interpretable. In contrast, LLM-based augmentation methods produce sentences via prompts at the token level, yielding readable outputs but offering limited control over the generation process. Inspired by recent advances in LLM inversion, which reconstructs natural language from embeddings and helps bridge the gap between latent embedding space and discrete token space, we propose inversedMixup, a unified framework that combines the controllability of Mixup with the interpretability of LLM-based generation. Specifically, inversedMixup adopts a three-stage training procedure to align the output embedding space of a task-specific model with the input embedding space of an LLM. Upon successful alignment, inversedMixup can reconstruct mixed embeddings with a controllable mixing ratio into human-interpretable augmented sentences, thereby improving the augmentation performance. Additionally, inversedMixup provides the first empirical evidence of the manifold intrusion phenomenon in text Mixup and introduces a simple yet effective strategy to mitigate it. Extensive experiments demonstrate the effectiveness and generalizability of our approach in both few-shot and fully supervised scenarios.
>
---
#### [replaced 056] Context-Free Recognition with Transformers
- **分类: cs.LG; cs.CC; cs.CL; cs.FL**

- **简介: 该论文研究Transformer在上下文无关语言识别中的能力。任务是解决其处理语法结构的局限性，通过引入循环层和填充token，证明其可识别所有上下文无关语言。**

- **链接: [https://arxiv.org/pdf/2601.01754v2](https://arxiv.org/pdf/2601.01754v2)**

> **作者:** Selim Jerad; Anej Svete; Sophie Hao; Ryan Cotterell; William Merrill
>
> **摘要:** Transformers excel empirically on tasks that process well-formed inputs according to some grammar, such as natural language and code. However, it remains unclear how they can process grammatical syntax. In fact, under standard complexity conjectures, standard transformers cannot recognize context-free languages (CFLs), a canonical formalism to describe syntax, or even regular languages, a subclass of CFLs. Past work proves that $\mathcal{O}(\log(n))$ looping layers (w.r.t. input length n) allows transformers to recognize regular languages, but the question of context-free recognition remained open. In this work, we show that looped transformers with $\mathcal{O}(\log(n))$ looping layers and $\mathcal{O}(n^6)$ padding tokens can recognize all CFLs. However, training and inference with $\mathcal{O}(n^6)$ padding tokens is potentially impractical. Fortunately, we show that, for natural subclasses such as unambiguous CFLs, the recognition problem on transformers becomes more tractable, requiring $\mathcal{O}(n^3)$ padding. We empirically validate our results and show that looping helps on a language that provably requires logarithmic depth. Overall, our results shed light on the intricacy of CFL recognition by transformers: While general recognition may require an intractable amount of padding, natural constraints such as unambiguity yield efficient recognition algorithms.
>
---
#### [replaced 057] Benchmarking Automatic Speech Recognition for Indian Languages in Agricultural Contexts
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决农业领域多语言ASR系统的性能评估问题。通过构建基准框架，分析不同语言和模型的表现，提出改进方法。**

- **链接: [https://arxiv.org/pdf/2602.03868v2](https://arxiv.org/pdf/2602.03868v2)**

> **作者:** Chandrashekar M S; Vineet Singh; Lakshmi Pedapudi
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** The digitization of agricultural advisory services in India requires robust Automatic Speech Recognition (ASR) systems capable of accurately transcribing domain-specific terminology in multiple Indian languages. This paper presents a benchmarking framework for evaluating ASR performance in agricultural contexts across Hindi, Telugu, and Odia languages. We introduce evaluation metrics including Agriculture Weighted Word Error Rate (AWWER) and domain-specific utility scoring to complement traditional metrics. Our evaluation of 10,934 audio recordings, each transcribed by up to 10 ASR models, reveals performance variations across languages and models, with Hindi achieving the best overall performance (WER: 16.2%) while Odia presents the greatest challenges (best WER: 35.1%, achieved only with speaker diarization). We characterize audio quality challenges inherent to real-world agricultural field recordings and demonstrate that speaker diarization with best-speaker selection can substantially reduce WER for multi-speaker recordings (upto 66% depending on the proportion of multi-speaker audio). We identify recurring error patterns in agricultural terminology and provide practical recommendations for improving ASR systems in low-resource agricultural domains. The study establishes baseline benchmarks for future agricultural ASR development.
>
---
#### [replaced 058] Personalized Learning Path Planning with Goal-Driven Learner State Modeling
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于个性化学习路径规划任务，旨在解决LLM在目标对齐规划上的不足。提出Pxplore框架，结合强化学习与教育架构，设计 learner state 模型和奖励函数，实现个性化学习路径生成。**

- **链接: [https://arxiv.org/pdf/2510.13215v2](https://arxiv.org/pdf/2510.13215v2)**

> **作者:** Joy Jia Yin Lim; Ye He; Jifan Yu; Xin Cong; Daniel Zhang-Li; Zhiyuan Liu; Huiqin Liu; Lei Hou; Juanzi Li; Bin Xu
>
> **备注:** Accepted at The Web Conference 2026 (WWW'26)
>
> **摘要:** Personalized Learning Path Planning (PLPP) aims to design adaptive learning paths that align with individual goals. While large language models (LLMs) show potential in personalizing learning experiences, existing approaches often lack mechanisms for goal-aligned planning. We introduce Pxplore, a novel framework for PLPP that integrates a reinforcement-based training paradigm and an LLM-driven educational architecture. We design a structured learner state model and an automated reward function that transforms abstract objectives into computable signals. We train the policy combining supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO), and deploy it within a real-world learning platform. Extensive experiments validate Pxplore's effectiveness in producing coherent, personalized, and goal-driven learning paths. We release our code and dataset at https://github.com/Pxplore/pxplore-algo.
>
---
