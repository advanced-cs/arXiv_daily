# 自然语言处理 cs.CL

- **最新发布 59 篇**

- **更新 43 篇**

## 最新发布

#### [new 001] Not Just the Destination, But the Journey: Reasoning Traces Causally Shape Generalization Behaviors
- **分类: cs.CL**

- **简介: 该论文属于模型对齐任务，旨在探究推理过程是否独立影响模型泛化行为。通过控制最终答案，研究不同推理路径的影响，发现推理内容具有因果效力。**

- **链接: [https://arxiv.org/pdf/2603.12397](https://arxiv.org/pdf/2603.12397)**

> **作者:** Pengcheng Wen; Yanxu Zhu; Jiapeng Sun; Han Zhu; Yujin Zhou; Chi-Min Chan; Sirui Han; Yike Guo
>
> **摘要:** Chain-of-Thought (CoT) is often viewed as a window into LLM decision-making, yet recent work suggests it may function merely as post-hoc rationalization. This raises a critical alignment question: Does the reasoning trace causally shape model generalization independent of the final answer? To isolate reasoning's causal effect, we design a controlled experiment holding final harmful answers constant while varying reasoning paths. We construct datasets with \textit{Evil} reasoning embracing malice, \textit{Misleading} reasoning rationalizing harm, and \textit{Submissive} reasoning yielding to pressure. We train models (0.6B--14B parameters) under multiple paradigms, including question-thinking-answer (QTA), question-thinking (QT), and thinking-only (T-only), and evaluate them in both think and no-think modes. We find that: (1) CoT training could amplify harmful generalization more than standard fine-tuning; (2) distinct reasoning types induce distinct behavioral patterns aligned with their semantics, despite identical final answers; (3) training on reasoning without answer supervision (QT or T-only) is sufficient to alter behavior, proving reasoning carries an independent signal; and (4) these effects persist even when generating answers without reasoning, indicating deep internalization. Our findings demonstrate that reasoning content is causally potent, challenging alignment strategies that supervise only outputs.
>
---
#### [new 002] Experimental evidence of progressive ChatGPT models self-convergence
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，研究LLM在递归训练下的输出多样性下降问题。通过文本相似度分析，发现ChatGPT模型随版本迭代出现输出趋同现象。**

- **链接: [https://arxiv.org/pdf/2603.12683](https://arxiv.org/pdf/2603.12683)**

> **作者:** Konstantinos F. Xylogiannopoulos; Petros Xanthopoulos; Panagiotis Karampelas; Georgios A. Bakamitsos
>
> **摘要:** Large Language Models (LLMs) that undergo recursive training on synthetically generated data are susceptible to model collapse, a phenomenon marked by the generation of meaningless output. Existing research has examined this issue from either theoretical or empirical perspectives, often focusing on a single model trained recursively on its own outputs. While prior studies have cautioned against the potential degradation of LLM output quality under such conditions, no longitudinal investigation has yet been conducted to assess this effect over time. In this study, we employ a text similarity metric to evaluate different ChatGPT models' capacity to generate diverse textual outputs. Our findings indicate a measurable decline of recent ChatGPT releases' ability to produce varied text, even when explicitly prompted to do so, by setting the temperature parameter to one. The observed reduction in output diversity may be attributed to the influence of the amounts of synthetic data incorporated within their training datasets as the result of internet infiltration by LLM generated data. The phenomenon is defined as model self-convergence because of the gradual increase of similarities of produced texts among different ChatGPT versions.
>
---
#### [new 003] Task-Specific Knowledge Distillation via Intermediate Probes
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识蒸馏任务，解决LLM在推理任务中输出信号质量下降的问题。通过训练轻量探针捕捉教师模型中间表示，提升学生模型训练效果。**

- **链接: [https://arxiv.org/pdf/2603.12270](https://arxiv.org/pdf/2603.12270)**

> **作者:** Ryan Brown; Chris Russell
>
> **摘要:** Knowledge distillation from large language models (LLMs) assumes that the teacher's output distribution is a high-quality training signal. On reasoning tasks, this assumption is frequently violated. A model's intermediate representations may encode the correct answer, yet this information is lost or distorted through the vocabulary projection, where prompt formatting and answer-token choices creates brittle, noisy outputs. We introduce \method{}, a distillation framework that bypasses this bottleneck by training lightweight probes on frozen teacher hidden states and using the probe's predictions, rather than output logits, as supervision for student training. This simple change yields consistent improvements across four reasoning benchmarks (AQuA-RAT, ARC Easy/Challenge, and MMLU), with gains most pronounced under limited data. Probes trained on intermediate representations provide cleaner labels than the teacher's own outputs, effectively denoising the distillation signal. \method{} requires no architectural changes to student or teacher, is architecture-agnostic, and adds minimal compute since probe training is cheap and teacher representations can be cached. By exploiting internal representations, \method{} enables practitioners to extract more value from large teacher models without additional training data or architectural complexity.
>
---
#### [new 004] Interpretable Semantic Gradients in SSD: A PCA Sweep Approach and a Case Study on AI Discourse
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决SSD中PCA降维参数选择问题，提出PCA sweep方法以提高模型稳定性和可解释性。**

- **链接: [https://arxiv.org/pdf/2603.13038](https://arxiv.org/pdf/2603.13038)**

> **作者:** Hubert Plisiecki; Maria Leniarska; Jan Piotrowski; Marcin Zajenkowski
>
> **备注:** Submitted to ACL 2026
>
> **摘要:** Supervised Semantic Differential (SSD) is a mixed quantitative-interpretive method that models how text meaning varies with continuous individual-difference variables by estimating a semantic gradient in an embedding space and interpreting its poles through clustering and text retrieval. SSD applies PCA before regression, but currently no systematic method exists for choosing the number of retained components, introducing avoidable researcher degrees of freedom in the analysis pipeline. We propose a PCA sweep procedure that treats dimensionality selection as a joint criterion over representation capacity, gradient interpretability, and stability across nearby values of K. We illustrate the method on a corpus of short posts about artificial intelligence written by Prolific participants who also completed Admiration and Rivalry narcissism scales. The sweep yields a stable, interpretable Admiration-related gradient contrasting optimistic, collaborative framings of AI with distrustful and derisive discourse, while no robust alignment emerges for Rivalry. We also show that a counterfactual using a high-PCA dimension solution heuristic produces diffuse, weakly structured clusters instead, reinforcing the value of the sweep-based choice of K. The case study shows how the PCA sweep constrains researcher degrees of freedom while preserving SSD's interpretive aims, supporting transparent and psychologically meaningful analyses of connotative meaning.
>
---
#### [new 005] SteerRM: Debiasing Reward Models via Sparse Autoencoders
- **分类: cs.CL**

- **简介: 该论文属于强化学习中的对齐任务，旨在解决奖励模型对表面风格特征的偏见问题。工作包括提出SteerRM方法，利用稀疏自编码器在不重新训练的情况下减少偏差。**

- **链接: [https://arxiv.org/pdf/2603.12795](https://arxiv.org/pdf/2603.12795)**

> **作者:** Mengyuan Sun; Zhuohao Yu; Weizheng Gu; Shikun Zhang; Wei Ye
>
> **摘要:** Reward models (RMs) are critical components of alignment pipelines, yet they exhibit biases toward superficial stylistic cues, preferring better-presented responses over semantically superior ones. Existing debiasing methods typically require retraining or architectural modifications, while direct activation suppression degrades performance due to representation entanglement. We propose SteerRM, the first training-free method for debiasing reward models using Sparse Autoencoder (SAE)-based interventions. SteerRM isolates stylistic effects using contrastive paired responses, identifies bias-related SAE features with a strength-stability criterion, and suppresses them at inference time. Across six reward models on RM-Bench, SteerRM improves Hard-split accuracy by 7.3 points on average while preserving overall performance. Results on a Gemma-based reward model and a controlled non-format bias further suggest generalization across RM architectures and bias types. We further find that format-related features are concentrated in shallow layers and transfer across models, revealing shared architecture-level bias encoding patterns. These results show that SAE-based interventions can mitigate reward-model biases without retraining, providing a practical and interpretable solution for alignment pipelines.
>
---
#### [new 006] Learning from Child-Directed Speech in Two-Language Scenarios: A French-English Case Study
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言语言模型研究，旨在探讨法语和英语环境下儿童导向言语的效果。通过对比不同训练数据，分析模型在语法和语义任务中的表现。**

- **链接: [https://arxiv.org/pdf/2603.12906](https://arxiv.org/pdf/2603.12906)**

> **作者:** Liel Binyamin; Elior Sulem
>
> **备注:** Accepted to Findings of EACL 2026
>
> **摘要:** Research on developmentally plausible language models has largely focused on English, leaving open questions about multilingual settings. We present a systematic study of compact language models by extending BabyBERTa to English-French scenarios under strictly size-matched data conditions, covering monolingual, bilingual, and cross-lingual settings. Our design contrasts two types of training corpora: (i) child-directed speech (about 2.5M tokens), following BabyBERTa and related work, and (ii) multi-domain corpora (about 10M tokens), extending the BabyLM framework to French. To enable fair evaluation, we also introduce new resources, including French versions of QAMR and QASRL, as well as English and French multi-domain corpora. We evaluate the models on both syntactic and semantic tasks and compare them with models trained on Wikipedia-only data. The results reveal context-dependent effects: training on Wikipedia consistently benefits semantic tasks, whereas child-directed speech improves grammatical judgments in monolingual settings. Bilingual pretraining yields notable gains for textual entailment, with particularly strong improvements for French. Importantly, similar patterns emerge across BabyBERTa, RoBERTa, and LTG-BERT, suggesting consistent trends across architectures.
>
---
#### [new 007] Marked Pedagogies: Examining Linguistic Biases in Personalized Automated Writing Feedback
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于自然语言处理中的教育应用任务，旨在解决自动化写作反馈中的语言偏见问题。通过分析LLM在不同学生属性下的反馈差异，揭示其存在的刻板印象偏差，并提出透明化与责任机制的必要性。**

- **链接: [https://arxiv.org/pdf/2603.12471](https://arxiv.org/pdf/2603.12471)**

> **作者:** Mei Tan; Lena Phalen; Dorottya Demszky
>
> **备注:** To appear in LAK 2026
>
> **摘要:** Effective personalized feedback is critical to students' literacy development. Though LLM-powered tools now promise to automate such feedback at scale, LLMs are not language-neutral: they privilege standard academic English and reproduce social stereotypes, raising concerns about how "personalization" shapes the feedback students receive. We examine how four widely used LLMs (GPT-4o, GPT-3.5-turbo, Llama-3.3 70B, Llama-3.1 8B) adapt written feedback in response to student attributes. Using 600 eighth-grade persuasive essays from the PERSUADE dataset, we generated feedback under prompt conditions embedding gender, race/ethnicity, learning needs, achievement, and motivation. We analyze lexical shifts across model outputs by adapting the Marked Words framework. Our results reveal systematic, stereotype-aligned shifts in feedback conditioned on presumed student attributes--even when essay content was identical. Feedback for students marked by race, language, or disability often exhibited positive feedback bias and feedback withholding bias--overuse of praise, less substantive critique, and assumptions of limited ability. Across attributes, models tailored not only what content was emphasized but also how writing was judged and how students were addressed. We term these instructional orientations Marked Pedagogies and highlight the need for transparency and accountability in automated feedback tools.
>
---
#### [new 008] RTD-Guard: A Black-Box Textual Adversarial Detection Framework via Replacement Token Detection
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于文本对抗检测任务，解决如何在不依赖攻击知识或模型访问的情况下检测对抗样本的问题。工作是提出RTD-Guard框架，利用替换词检测技术实现高效防御。**

- **链接: [https://arxiv.org/pdf/2603.12582](https://arxiv.org/pdf/2603.12582)**

> **作者:** He Zhu; Yanshu Li; Wen Liu; Haitian Yang
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Textual adversarial attacks pose a serious security threat to Natural Language Processing (NLP) systems by introducing imperceptible perturbations that mislead deep learning models. While adversarial example detection offers a lightweight alternative to robust training, existing methods typically rely on prior knowledge of attacks, white-box access to the victim model, or numerous queries, which severely limits their practical deployment. This paper introduces RTD-Guard, a novel black-box framework for detecting textual adversarial examples. Our key insight is that word-substitution perturbations in adversarial attacks closely resemble the "replaced tokens" that a Replaced Token Detection (RTD) discriminator is pre-trained to identify. Leveraging this, RTD-Guard employs an off-the-shelf RTD discriminator-without fine-tuning-to localize suspicious tokens, masks them, and detects adversarial examples by observing the prediction confidence shift of the victim model before and after intervention. The entire process requires no adversarial data, model tuning, or internal model access, and uses only two black-box queries. Comprehensive experiments on multiple benchmark datasets demonstrate that RTD-Guard effectively detects adversarial texts generated by diverse state-of-the-art attack methods. It surpasses existing detection baselines across multiple metrics, offering a highly efficient, practical, and resource-light defense mechanism-particularly suited for real-world deployment in resource-constrained or privacy-sensitive environments.
>
---
#### [new 009] Continual Learning in Large Language Models: Methods, Challenges, and Opportunities
- **分类: cs.CL; cs.AI**

- **简介: 本文探讨了大语言模型的持续学习问题，旨在解决模型在新任务中遗忘旧知识的问题。论文分析了持续预训练、微调等方法，比较了不同技术的适应性与改进方向。**

- **链接: [https://arxiv.org/pdf/2603.12658](https://arxiv.org/pdf/2603.12658)**

> **作者:** Hongyang Chen; Zhongwu Sun; Hongfei Ye; Kunchi Li; Xuemin Lin
>
> **摘要:** Continual learning (CL) has emerged as a pivotal paradigm to enable large language models (LLMs) to dynamically adapt to evolving knowledge and sequential tasks while mitigating catastrophic forgetting-a critical limitation of the static pre-training paradigm inherent to modern LLMs. This survey presents a comprehensive overview of CL methodologies tailored for LLMs, structured around three core training stages: continual pre-training, continual fine-tuning, and continual this http URL the canonical taxonomy of rehearsal-, regularization-, and architecture-based methods, we further subdivide each category by its distinct forgetting mitigation mechanisms and conduct a rigorous comparative analysis of the adaptability and critical improvements of traditional CL methods for LLMs. In doing so, we explicitly highlight core distinctions between LLM CL and traditional machine learning, particularly with respect to scale, parameter efficiency, and emergent capabilities. Our analysis covers essential evaluation metrics, including forgetting rates and knowledge transfer efficiency, along with emerging benchmarks for assessing CL performance. This survey reveals that while current methods demonstrate promising results in specific domains, fundamental challenges persist in achieving seamless knowledge integration across diverse tasks and temporal scales. This systematic review contributes to the growing body of knowledge on LLM adaptation, providing researchers and practitioners with a structured framework for understanding current achievements and future opportunities in lifelong learning for language models.
>
---
#### [new 010] AgentDrift: Unsafe Recommendation Drift Under Tool Corruption Hidden by Ranking Metrics in LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI安全任务，旨在解决LLM代理在高风险场景中推荐安全性不足的问题。通过实验揭示了评估指标的盲点，并提出改进的评估方法。**

- **链接: [https://arxiv.org/pdf/2603.12564](https://arxiv.org/pdf/2603.12564)**

> **作者:** Zekun Wu; Adriano Koshiyama; Sahan Bulathwela; Maria Perez-Ortiz
>
> **备注:** 50 pages, 31 tables, 15 figures. Under review at COLM 2026
>
> **摘要:** Tool-augmented LLM agents increasingly serve as multi-turn advisors in high-stakes domains, yet their evaluation relies on ranking-quality metrics that measure what is recommended but not whether it is safe for the user. We introduce a paired-trajectory protocol that replays real financial dialogues under clean and contaminated tool-output conditions across seven LLMs (7B to frontier) and decomposes divergence into information-channel and memory-channel mechanisms. Across the seven models tested, we consistently observe the evaluation-blindness pattern: recommendation quality is largely preserved under contamination (utility preservation ratio approximately 1.0) while risk-inappropriate products appear in 65-93% of turns, a systematic safety failure poorly reflected by standard NDCG. Safety violations are predominantly information-channel-driven, emerge at the first contaminated turn, and persist without self-correction over 23-step trajectories; no agent across 1,563 contaminated turns explicitly questions tool-data reliability. Even narrative-only corruption (biased headlines, no numerical manipulation) induces significant drift while completely evading consistency monitors. A safety-penalized NDCG variant (sNDCG) reduces preservation ratios to 0.51-0.74, indicating that much of the evaluation gap becomes visible once safety is explicitly measured. These results motivate considering trajectory-level safety monitoring, beyond single-turn quality, for deployed multi-turn agents in high-stakes settings.
>
---
#### [new 011] Mending the Holes: Mitigating Reward Hacking in Reinforcement Learning for Multilingual Translation
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决低资源语言翻译性能不足的问题。通过引入WALAR方法，利用单语文本提升模型翻译能力，同时保持高资源语言表现。**

- **链接: [https://arxiv.org/pdf/2603.13045](https://arxiv.org/pdf/2603.13045)**

> **作者:** Yifeng Liu; Siqi Ouyang; Yatish Hosmane Revanasiddappa; Lei Li
>
> **备注:** Our code is available at this https URL
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capability in machine translation on high-resource language pairs, yet their performance on low-resource translation still lags behind. Existing post-training methods rely heavily on high-quality parallel data, which are often scarce or unavailable for low-resource languages. In this paper, we introduce WALAR, a reinforcement training method using only monolingual text to elevate LLMs' translation capabilities on massive low-resource languages while retaining their performance on high-resource languages. Our key insight is based on the observation of failure modes (or "holes") in existing source-based multilingual quality estimation (QE) models. Reinforcement learning (RL) using these QE models tends to amplify such holes, resulting in poorer multilingual LLMs. We develop techniques including word alignment and language alignment to mitigate such holes in WALAR's reward for RL training. We continually trained an LLM supporting translation of 101 languages using WALAR. The experiments show that our new model outperforms LLaMAX, one of the strongest open-source multilingual LLMs by a large margin on 1400 language directions on Flores-101 dataset.
>
---
#### [new 012] Diagnosing Retrieval Bias Under Multiple In-Context Knowledge Updates in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究知识密集型任务中大语言模型的检索偏差问题，针对多轮知识更新场景，提出DKI框架评估模型表现，发现随着更新次数增加，模型准确率下降。**

- **链接: [https://arxiv.org/pdf/2603.12271](https://arxiv.org/pdf/2603.12271)**

> **作者:** Boyu Qiao; Sean Guo; Xian Yang; Kun Li; Wei Zhou; Songlin Hu; Yunya Song
>
> **摘要:** LLMs are widely used in knowledge-intensive tasks where the same fact may be revised multiple times within context. Unlike prior work focusing on one-shot updates or single conflicts, multi-update scenarios contain multiple historically valid versions that compete at retrieval, yet remain underexplored. This challenge resembles the AB-AC interference paradigm in cognitive psychology: when the same cue A is successively associated with B and C, the old and new associations compete during retrieval, leading to bias. Inspired by this, we introduce a Dynamic Knowledge Instance (DKI) evaluation framework, modeling multi-updates of the same fact as a cue paired with a sequence of updated values, and assess models via endpoint probing of the earliest (initial) and latest (current) states. Across diverse LLMs, we observe that retrieval bias intensifies as updates increase, earliest-state accuracy stays high while latest-state accuracy drops substantially. Diagnostic analyses of attention, hidden-state similarity, and output logits further reveal that these signals become flatter and weakly discriminative on errors, providing little stable basis for identifying the latest update. Finally, cognitively inspired heuristic intervention strategies yield only modest gains and do not eliminate the bias. Our results reveal a persistent challenge in tracking and following knowledge updates in long contexts.
>
---
#### [new 013] Adaptive Vision-Language Model Routing for Computer Use Agents
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于计算机视觉与自然语言处理任务，解决CUA中VLM路由效率与准确性问题。提出AVR框架，根据难度动态选择模型，提升效率并保证可靠性。**

- **链接: [https://arxiv.org/pdf/2603.12823](https://arxiv.org/pdf/2603.12823)**

> **作者:** Xunzhuo Liu; Bowei He; Xue Liu; Andy Luo; Haichen Zhang; Huamin Chen
>
> **摘要:** Computer Use Agents (CUAs) translate natural-language instructions into Graphical User Interface (GUI) actions such as clicks, keystrokes, and scrolls by relying on a Vision-Language Model (VLM) to interpret screenshots and predict grounded tool calls. However, grounding accuracy varies dramatically across VLMs, while current CUA systems typically route every action to a single fixed model regardless of difficulty. We propose \textbf{Adaptive VLM Routing} (AVR), a framework that inserts a lightweight semantic routing layer between the CUA orchestrator and a pool of VLMs. For each tool call, AVR estimates action difficulty from multimodal embeddings, probes a small VLM to measure confidence, and routes the action to the cheapest model whose predicted accuracy satisfies a target reliability threshold. For \textit{warm} agents with memory of prior UI interactions, retrieved context further narrows the capability gap between small and large models, allowing many actions to be handled without escalation. We formalize routing as a cost--accuracy trade-off, derive a threshold-based policy for model selection, and evaluate AVR using ScreenSpot-Pro grounding data together with the OpenClaw agent routing benchmark. Across these settings, AVR projects inference cost reductions of up to 78\% while staying within 2 percentage points of an all-large-model baseline. When combined with the Visual Confused Deputy guardrail, AVR also escalates high-risk actions directly to the strongest available model, unifying efficiency and safety within a single routing framework. Materials are also provided Model, benchmark, and code: this https URL.
>
---
#### [new 014] Using a Human-AI Teaming Approach to Create and Curate Scientific Datasets with the SCILIRE System
- **分类: cs.CL; cs.HC**

- **简介: 论文提出SCILIRE系统，解决科学文献中结构化知识提取难题。通过人机协作流程，提升数据集构建的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2603.12638](https://arxiv.org/pdf/2603.12638)**

> **作者:** Necva Bölücü; Jessica Irons; Changhyun Lee; Brian Jin; Maciej Rybinski; Huichen Yang; Andreas Duenser; Stephen Wan
>
> **备注:** 17pages, 9 figures, EACL demo track
>
> **摘要:** The rapid growth of scientific literature has made manual extraction of structured knowledge increasingly impractical. To address this challenge, we introduce SCILIRE, a system for creating datasets from scientific literature. SCILIRE has been designed around Human-AI teaming principles centred on workflows for verifying and curating data. It facilitates an iterative workflow in which researchers can review and correct AI outputs. Furthermore, this interaction is used as a feedback signal to improve future LLM-based inference. We evaluate our design using a combination of intrinsic benchmarking outcomes together with real-world case studies across multiple domains. The results demonstrate that SCILIRE improves extraction fidelity and facilitates efficient dataset creation.
>
---
#### [new 015] Is Human Annotation Necessary? Iterative MBR Distillation for Error Span Detection in Machine Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译评估任务，解决错误段落检测问题。提出一种无需人工标注的自进化框架，通过MBR解码生成伪标签，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.12983](https://arxiv.org/pdf/2603.12983)**

> **作者:** Boxuan Lyu; Haiyue Song; Zhi Qu
>
> **摘要:** Error Span Detection (ESD) is a crucial subtask in Machine Translation (MT) evaluation, aiming to identify the location and severity of translation errors. While fine-tuning models on human-annotated data improves ESD performance, acquiring such data is expensive and prone to inconsistencies among annotators. To address this, we propose a novel self-evolution framework based on Minimum Bayes Risk (MBR) decoding, named Iterative MBR Distillation for ESD, which eliminates the reliance on human annotations by leveraging an off-the-shelf LLM to generate this http URL experiments on the WMT Metrics Shared Task datasets demonstrate that models trained solely on these self-generated pseudo-labels outperform both unadapted base model and supervised baselines trained on human annotations at the system and span levels, while maintaining competitive sentence-level performance.
>
---
#### [new 016] Rethinking Multiple-Choice Questions for RLVR: Unlocking Potential via Distractor Design
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RLVR中MCQ因选项设计不当导致的奖励欺骗问题。通过分析选项设计影响，提出IDC框架提升干扰项质量，增强模型推理能力。**

- **链接: [https://arxiv.org/pdf/2603.12826](https://arxiv.org/pdf/2603.12826)**

> **作者:** Xu Guo; Qiming Ge; Jian Tong; Kedi Chen; Jin Zhang; Xiaogui Yang; Xuan Gao; Haijun Lv; Zhihui Lu; Yicheng Zou; Qipeng Guo
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) significantly enhances the reasoning capabilities of Large Language Models. When applied to RLVR, Multiple-Choice Questions (MCQs) offer a scalable source of verifiable data but risk inducing reward hacking, where models shortcut reasoning via random guessing or simple elimination. Current approaches often mitigate this by converting MCQs to open-ended formats, thereby discarding the contrastive signal provided by expert-designed distractors. In this work, we systematically investigate the impact of option design on RLVR. Our analysis highlights two primary insights: (1) Mismatches in option counts between training and testing degrade performance. (2) Strong distractors effectively mitigate random guessing, enabling effective RLVR training even with 2-way questions. Motivated by these findings, we propose Iterative Distractor Curation (IDC), a framework that actively constructs high-quality distractors to block elimination shortcuts and promote deep reasoning. Experiments on various benchmarks demonstrate that our method effectively enhances distractor quality and yields significant gains in RLVR training compared to the original data.
>
---
#### [new 017] Shattering the Shortcut: A Topology-Regularized Benchmark for Multi-hop Medical Reasoning in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗推理任务，旨在解决LLMs在多步诊断推理中的“捷径学习”问题。通过构建拓扑正则化的知识图谱和基准数据集，评估并提升模型的深层推理能力。**

- **链接: [https://arxiv.org/pdf/2603.12458](https://arxiv.org/pdf/2603.12458)**

> **作者:** Xing Zi; Xinying Zhou; Jinghao Xiao; Catarina Moreira; Mukesh Prasad
>
> **摘要:** While Large Language Models (LLMs) achieve expert-level performance on standard medical benchmarks through single-hop factual recall, they severely struggle with the complex, multi-hop diagnostic reasoning required in real-world clinical settings. A primary obstacle is "shortcut learning", where models exploit highly connected, generic hub nodes (e.g., "inflammation") in knowledge graphs to bypass authentic micro-pathological cascades. To address this, we introduce ShatterMed-QA, a bilingual benchmark of 10,558 multi-hop clinical questions designed to rigorously evaluate deep diagnostic reasoning. Our framework constructs a topology-regularized medical Knowledge Graph using a novel $k$-Shattering algorithm, which physically prunes generic hubs to explicitly sever logical shortcuts. We synthesize the evaluation vignettes by applying implicit bridge entity masking and topology-driven hard negative sampling, forcing models to navigate biologically plausible distractors without relying on superficial elimination. Comprehensive evaluations of 21 LLMs reveal massive performance degradation on our multi-hop tasks, particularly among domain-specific models. Crucially, restoring the masked evidence via Retrieval-Augmented Generation (RAG) triggers near-universal performance recovery, validating ShatterMed-QA's structural fidelity and proving its efficacy in diagnosing the fundamental reasoning deficits of current medical AI. Explore the dataset, interactive examples, and full leaderboards at our project website: this https URL
>
---
#### [new 018] GONE: Structural Knowledge Unlearning via Neighborhood-Expanded Distribution Shaping
- **分类: cs.CL**

- **简介: 该论文属于知识去遗忘任务，旨在解决LLM中结构化知识的删除问题。提出GONE基准和NEDS框架，有效实现知识擦除并减少副作用。**

- **链接: [https://arxiv.org/pdf/2603.12275](https://arxiv.org/pdf/2603.12275)**

> **作者:** Chahana Dahal; Ashutosh Balasubramaniam; Zuobin Xiong
>
> **摘要:** Unlearning knowledge is a pressing and challenging task in Large Language Models (LLMs) because of their unprecedented capability to memorize and digest training data at scale, raising more significant issues regarding safety, privacy, and intellectual property. However, existing works, including parameter editing, fine-tuning, and distillation-based methods, are all focused on flat sentence-level data but overlook the relational, multi-hop, and reasoned knowledge in naturally structured data. In response to this gap, this paper introduces Graph Oblivion and Node Erasure (GONE), a benchmark for evaluating knowledge unlearning over structured knowledge graph (KG) facts in LLMs. This KG-based benchmark enables the disentanglement of three effects of unlearning: direct fact removal, reasoning-based leakage, and catastrophic forgetting. In addition, Neighborhood-Expanded Distribution Shaping (NEDS), a novel unlearning framework, is designed to leverage graph connectivity and identify anchor correlated neighbors, enforcing a precise decision boundary between the forgotten fact and its semantic neighborhood. Evaluations on LLaMA-3-8B and Mistral-7B across multiple knowledge editing and unlearning methods showcase NEDS's superior performance (1.000 on unlearning efficacy and 0.839 on locality) on GONE and other benchmarks. Code is available at this https URL.
>
---
#### [new 019] SectEval: Evaluating the Latent Sectarian Preferences of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于AI偏见检测任务，旨在评估大语言模型对伊斯兰教逊尼派与什叶派的潜在偏见。通过构建SectEval测试，发现模型在不同语言和地理位置下表现出宗教观点的不一致。**

- **链接: [https://arxiv.org/pdf/2603.12768](https://arxiv.org/pdf/2603.12768)**

> **作者:** Aditya Maheshwari; Amit Gajkeshwar; Kaushal Sharma; Vivek Patel
>
> **备注:** 14 pages; 3 figures
>
> **摘要:** As Large Language Models (LLMs) becomes a popular source for religious knowledge, it is important to know if it treats different groups fairly. This study is the first to measure how LLMs handle the differences between the two main sects of Islam: Sunni and Shia. We present a test called SectEval, available in both English and Hindi, consisting of 88 questions, to check the bias-ness of 15 top LLM models, both proprietary and open-weights. Our results show a major inconsistency based on language. In English, many powerful models DeepSeek-v3 and GPT-4o often favored Shia answers. However, when asked the exact same questions in Hindi, these models switched to favoring Sunni answers. This means a user could get completely different religious advice just by changing languages. We also looked at how models react to location. Advanced models Claude-3.5 changed their answers to match the user's country-giving Shia answers to a user from Iran and Sunni answers to a user from Saudi Arabia. In contrast, smaller models (especially in Hindi) ignored the user's location and stuck to a Sunni viewpoint. These findings show that AI is not neutral; its religious ``truth'' changes depending on the language you speak and the country you claim to be from. The data set is available at this https URL
>
---
#### [new 020] Prompt Injection as Role Confusion
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文研究语言模型的安全问题，聚焦于提示注入攻击。通过分析角色混淆机制，揭示攻击成功原因，并提出新方法验证其有效性。**

- **链接: [https://arxiv.org/pdf/2603.12277](https://arxiv.org/pdf/2603.12277)**

> **作者:** Charles Ye; Jasmine Cui; Dylan Hadfield-Menell
>
> **摘要:** Language models remain vulnerable to prompt injection attacks despite extensive safety training. We trace this failure to role confusion: models infer roles from how text is written, not where it comes from. We design novel role probes to capture how models internally identify "who is speaking." These reveal why prompt injection works: untrusted text that imitates a role inherits that role's authority. We test this insight by injecting spoofed reasoning into user prompts and tool outputs, achieving average success rates of 60% on StrongREJECT and 61% on agent exfiltration, across multiple open- and closed-weight models with near-zero baselines. Strikingly, the degree of internal role confusion strongly predicts attack success before generation begins. Our findings reveal a fundamental gap: security is defined at the interface but authority is assigned in latent space. More broadly, we introduce a unifying, mechanistic framework for prompt injection, demonstrating that diverse prompt-injection attacks exploit the same underlying role-confusion mechanism.
>
---
#### [new 021] LMEB: Long-horizon Memory Embedding Benchmark
- **分类: cs.CL**

- **简介: 该论文提出LMEB基准，用于评估长时记忆嵌入模型。解决现有基准在长时记忆检索任务上的不足，涵盖多种记忆类型和任务，评估多个模型表现。**

- **链接: [https://arxiv.org/pdf/2603.12572](https://arxiv.org/pdf/2603.12572)**

> **作者:** Xinping Zhao; Xinshuo Hu; Jiaxin Xu; Danyu Tang; Xin Zhang; Mengjia Zhou; Yan Zhong; Yao Zhou; Zifei Shan; Meishan Zhang; Baotian Hu; Min Zhang
>
> **备注:** 35 pages, 9 figures, 23 tables
>
> **摘要:** Memory embeddings are crucial for memory-augmented systems, such as OpenClaw, but their evaluation is underexplored in current text embedding benchmarks, which narrowly focus on traditional passage retrieval and fail to assess models' ability to handle long-horizon memory retrieval tasks involving fragmented, context-dependent, and temporally distant information. To address this, we introduce the Long-horizon Memory Embedding Benchmark (LMEB), a comprehensive framework that evaluates embedding models' capabilities in handling complex, long-horizon memory retrieval tasks. LMEB spans 22 datasets and 193 zero-shot retrieval tasks across 4 memory types: episodic, dialogue, semantic, and procedural, with both AI-generated and human-annotated data. These memory types differ in terms of level of abstraction and temporal dependency, capturing distinct aspects of memory retrieval that reflect the diverse challenges of the real world. We evaluate 15 widely used embedding models, ranging from hundreds of millions to ten billion parameters. The results reveal that (1) LMEB provides a reasonable level of difficulty; (2) Larger models do not always perform better; (3) LMEB and MTEB exhibit orthogonality. This suggests that the field has yet to converge on a universal model capable of excelling across all memory retrieval tasks, and that performance in traditional passage retrieval may not generalize to long-horizon memory retrieval. In summary, by providing a standardized and reproducible evaluation framework, LMEB fills a crucial gap in memory embedding evaluation, driving further advancements in text embedding for handling long-term, context-dependent memory retrieval. LMEB is available at this https URL.
>
---
#### [new 022] EvolveCoder: Evolving Test Cases via Adversarial Verification for Code Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于代码生成任务，解决RLVR中验证信号弱的问题，通过对抗性验证框架迭代优化测试用例，提升强化学习效果。**

- **链接: [https://arxiv.org/pdf/2603.12698](https://arxiv.org/pdf/2603.12698)**

> **作者:** Chi Ruan; Dongfu Jiang; Huaye Zeng; Ping Nie; Wenhu Chen
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) is a promising approach for improving code generation in large language models, but its effectiveness is limited by weak and static verification signals in existing coding RL datasets. In this paper, we propose a solution-conditioned and adversarial verification framework that iteratively refines test cases based on the execution behaviors of candidate solutions, with the goal of increasing difficulty, improving discriminative power, and reducing redundancy. Based on this framework, we introduce EvolveCoder-22k, a large-scale coding reinforcement learning dataset constructed through multiple rounds of adversarial test case evolution. Empirical analysis shows that iterative refinement substantially strengthens verification, with pass@1 decreasing from 43.80 to 31.22. Reinforcement learning on EvolveCoder-22k yields stable optimization and consistent performance gains, improving Qwen3-4B by an average of 4.2 points across four downstream benchmarks and outperforming strong 4B-scale baselines. Our results highlight the importance of adversarial, solution-conditioned verification for effective and scalable reinforcement learning in code generation.
>
---
#### [new 023] Long-form RewardBench: Evaluating Reward Models for Long-form Generation
- **分类: cs.CL**

- **简介: 该论文属于奖励模型评估任务，旨在解决长文本生成中奖励模型能力不足的问题。通过构建Long-form RewardBench基准，涵盖多种子任务，并对比不同模型性能，揭示现有模型的局限性。**

- **链接: [https://arxiv.org/pdf/2603.12963](https://arxiv.org/pdf/2603.12963)**

> **作者:** Hui Huang; Yancheng He; Wei Liu; Muyun Yang; Jiaheng Liu; Kehai Chen; Bing Xu; Conghui Zhu; Hailong Cao; Tiejun Zhao
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** The widespread adoption of reinforcement learning-based alignment highlights the growing importance of reward models. Various benchmarks have been built to evaluate reward models in various domains and scenarios. However, a significant gap remains in assessing reward models for long-form generation, despite its critical role in real-world applications. To bridge this, we introduce Long-form RewardBench, the first reward modeling testbed specifically designed for long-form generation. Our benchmark encompasses five key subtasks: QA, RAG, Chat, Writing, and Reasoning. We collected instruction and preference data through a meticulously designed multi-stage data collection process, and conducted extensive experiments on 20+ mainstream reward models, including both classifiers and generative models. Our findings reveal that current models still lack long-form reward modeling capabilities. Furthermore, we designed a novel Long-form Needle-in-a-Haystack Test, which revealed a correlation between reward modeling performance and the error's position within a response, as well as the overall response length, with distinct characteristics observed between classification and generative models. Finally, we demonstrate that classifiers exhibit better generalizability compared to generative models trained on the same data. As the first benchmark for long-form reward modeling, this work aims to offer a robust platform for visualizing progress in this crucial area.
>
---
#### [new 024] LLM-Augmented Therapy Normalization and Aspect-Based Sentiment Analysis for Treatment-Resistant Depression on Reddit
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在解决治疗抵抗性抑郁症患者用药体验的量化问题。通过提取Reddit数据并进行情感分析，分析患者对药物的看法。**

- **链接: [https://arxiv.org/pdf/2603.12343](https://arxiv.org/pdf/2603.12343)**

> **作者:** Yuxin Zhu; Sahithi Lakamana; Masoud Rouhizadeh; Selen Bozkurt; Rachel Hershenberg; Abeed Sarker
>
> **摘要:** Treatment-resistant depression (TRD) is a severe form of major depressive disorder in which patients do not achieve remission despite multiple adequate treatment trials. Evidence across pharmacologic options for TRD remains limited, and trials often do not fully capture patient-reported tolerability. Large-scale online peer-support narratives therefore offer a complementary lens on how patients describe and evaluate medications in real-world use. In this study, we curated a corpus of 5,059 Reddit posts explicitly referencing TRD from 3,480 subscribers across 28 mental health-related subreddits from 2010 to 2025. Of these, 3,839 posts mentioned at least one medication, yielding 23,399 mentions of 81 generic-name medications after lexicon-based normalization of brand names, misspellings, and colloquialisms. We developed an aspect-based sentiment classifier by fine-tuning DeBERTa-v3 on the SMM4H 2023 therapy-sentiment Twitter corpus with large language model based data augmentation, achieving a micro-F1 score of 0.800 on the shared-task test set. Applying this classifier to Reddit, we quantified sentiment toward individual medications across three categories: positive, neutral, and negative, and tracked patterns by drug, subscriber, subreddit, and year. Overall, 72.1% of medication mentions were neutral, 14.8% negative, and 13.1% positive. Conventional antidepressants, especially SSRIs and SNRIs, showed consistently higher negative than positive proportions, whereas ketamine and esketamine showed comparatively more favorable sentiment profiles. These findings show that normalized medication extraction combined with aspect-based sentiment analysis can help characterize patient-perceived treatment experiences in TRD-related Reddit discourse, complementing clinical evidence with large-scale patient-generated perspectives.
>
---
#### [new 025] LLM BiasScope: A Real-Time Bias Analysis Platform for Comparative LLM Evaluation
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文提出LLM BiasScope，用于实时比较和分析多个大语言模型的偏见。属于模型评估任务，解决偏见检测与对比分析问题，通过双阶段检测流程实现 bias 分析与可视化。**

- **链接: [https://arxiv.org/pdf/2603.12522](https://arxiv.org/pdf/2603.12522)**

> **作者:** Himel Ghosh; Nick Elias Werner
>
> **备注:** Accepted at EACL 2026 (24-29 March, Morocco)
>
> **摘要:** As large language models (LLMs) are deployed widely, detecting and understanding bias in their outputs is critical. We present LLM BiasScope, a web application for side-by-side comparison of LLM outputs with real-time bias analysis. The system supports multiple providers (Google Gemini, DeepSeek, MiniMax, Mistral, Meituan, Meta Llama) and enables researchers and practitioners to compare models on the same prompts while analyzing bias patterns. LLM BiasScope uses a two-stage bias detection pipeline: sentence-level bias detection followed by bias type classification for biased sentences. The analysis runs automatically on both user prompts and model responses, providing statistics, visualizations, and detailed breakdowns of bias types. The interface displays two models side-by-side with synchronized streaming responses, per-model bias summaries, and a comparison view highlighting differences in bias distributions. The system is built on this http URL with React, integrates Hugging Face inference endpoints for bias detection, and uses the Vercel AI SDK for multi-provider LLM access. Features include real-time streaming, export to JSON/PDF, and interactive visualizations (bar charts, radar charts) for bias analysis. LLM BiasScope is available as an open-source web application, providing a practical tool for bias evaluation and comparative analysis of LLM behaviour.
>
---
#### [new 026] Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出EPT，解决多任务中专家架构单一的问题，通过分层特征金字塔提升参数效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.12577](https://arxiv.org/pdf/2603.12577)**

> **作者:** Jia-Chen Zhang; Zhen-Wei Yan; Yu-Jie Xiong; Chun-Ming Xia
>
> **摘要:** Parameter-Efficient Fine-Tuning (PEFT) has become a dominant paradigm for deploying LLMs in multi-task scenarios due to its extreme parameter efficiency. While Mixture-of-Experts (MoE) based LoRA variants have achieved promising results by dynamically routing tokens to different low-rank experts, they largely overlook the hierarchical nature of task complexity. Existing methods typically employ experts with uniform architectures, limiting their ability to capture diverse feature granularities required by distinct tasks--where some tasks demand high-level semantic abstraction while others require fine-grained syntactic manipulation. To bridge this gap, we propose Expert Pyramid Tuning (EPT), a novel architecture that integrates the multi-scale feature pyramid concept from computer vision into the realm of PEFT. Unlike standard LoRA, EPT decomposes task adaptation into two stages: (1) A shared meta-knowledge Subspace that encodes universal linguistic patterns in low dimensions; (2) A Pyramid Projection Mechanism that utilizes learnable up-projection operators to reconstruct high-dimensional features at varying scales. A task-aware router then dynamically selects the optimal combination of these multi-scale features. Extensive experiments across multiple multi-task benchmarks demonstrate that EPT significantly outperforms SOTA MoE-LoRA variants. Crucially, thanks to the re-parameterization capability of our design, EPT achieves this performance improvement while simultaneously reducing the number of training parameters.
>
---
#### [new 027] HMS-BERT: Hybrid Multi-Task Self-Training for Multilingual and Multi-Label Cyberbullying Detection
- **分类: cs.CL; stat.ML**

- **简介: 该论文属于多语言多标签网络欺凌检测任务，解决现有方法在多语言和多标签场景下的局限性。提出HMS-BERT框架，结合上下文与人工特征，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.12920](https://arxiv.org/pdf/2603.12920)**

> **作者:** Zixin Feng; Xinying Cui; Yifan Sun; Zheng Wei; Jiachen Yuan; Jiazhen Hu; Ning Xin; Md Maruf Hasan
>
> **摘要:** Cyberbullying on social media is inherently multilingual and multi-faceted, where abusive behaviors often overlap across multiple categories. Existing methods are commonly limited by monolingual assumptions or single-task formulations, which restrict their effectiveness in realistic multilingual and multi-label scenarios. In this paper, we propose HMS-BERT, a hybrid multi-task self-training framework for multilingual and multi-label cyberbullying detection. Built upon a pretrained multilingual BERT backbone, HMS-BERT integrates contextual representations with handcrafted linguistic features and jointly optimizes a fine-grained multi-label abuse classification task and a three-class main classification task. To address labeled data scarcity in low-resource languages, an iterative self-training strategy with confidence-based pseudo-labeling is introduced to facilitate cross-lingual knowledge transfer. Experiments on four public datasets demonstrate that HMS-BERT achieves strong performance, attaining a macro F1-score of up to 0.9847 on the multi-label task and an accuracy of 0.6775 on the main classification task. Ablation studies further verify the effectiveness of the proposed components.
>
---
#### [new 028] ActTail: Global Activation Sparsity in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于大语言模型优化任务，解决激活稀疏性分配不均导致性能下降的问题。提出ActTail方法，基于重尾自正则化理论，实现全局激活稀疏性合理分配。**

- **链接: [https://arxiv.org/pdf/2603.12272](https://arxiv.org/pdf/2603.12272)**

> **作者:** Wenwen Hou; Xinyuan Song; Shiwei Liu
>
> **摘要:** Activation sparsity is a promising approach for accelerating large language model (LLM) inference by reducing computation and memory movement. However, existing activation sparsity methods typically apply uniform sparsity across projections, ignoring the heterogeneous statistical properties of Transformer weights and thereby amplifying performance degradation. In this paper, we propose ActTail, a TopK magnitude-based activation sparsity method with global activation sparsity allocation grounded in Heavy-Tailed Self-Regularization (HT-SR) theory. Specifically, we capture this heterogeneity via the heavy-tail exponent computed from each projection's empirical spectral density (ESD), which is used as a quantitative indicator to assign projection-specific sparsity budgets. Importantly, we provide a theoretical analysis that establishes an explicit relationship between the activation sparsity ratio and the heavy-tail exponent under the HT-SR regime, offering principled guidance for sparsity allocation beyond heuristic design. Experiments on LLaMA and Mistral models show that our method improves both perplexity and downstream task performance at high sparsity compared to uniform allocation. At 80% sparsity, perplexity is reduced by 21.8% on LLaMA-2-7B, 40.1% on LLaMA-2-13B, and 9.4% on Mistral-7B.
>
---
#### [new 029] A Method for Learning Large-Scale Computational Construction Grammars from Semantically Annotated Corpora
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决从语义标注语料中学习大规模构造语法的问题。工作包括提出一种方法，构建可解释的计算构造语法，捕捉句法与语义关系。**

- **链接: [https://arxiv.org/pdf/2603.12754](https://arxiv.org/pdf/2603.12754)**

> **作者:** Paul Van Eecke; Katrien Beuls
>
> **摘要:** We present a method for learning large-scale, broad-coverage construction grammars from corpora of language use. Starting from utterances annotated with constituency structure and semantic frames, the method facilitates the learning of human-interpretable computational construction grammars that capture the intricate relationship between syntactic structures and the semantic relations they express. The resulting grammars consist of networks of tens of thousands of constructions formalised within the Fluid Construction Grammar framework. Not only do these grammars support the frame-semantic analysis of open-domain text, they also house a trove of information about the syntactico-semantic usage patterns present in the data they were learnt from. The method and learnt grammars contribute to the scaling of usage-based, constructionist approaches to language, as they corroborate the scalability of a number of fundamental construction grammar conjectures while also providing a practical instrument for the constructionist study of English argument structure in broad-coverage corpora.
>
---
#### [new 030] Neuron-Aware Data Selection In Instruction Tuning For Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于指令微调任务，旨在解决IT数据选择效率问题。通过分析神经元激活模式，提出NAIT框架，高效选取高质量数据提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.13201](https://arxiv.org/pdf/2603.13201)**

> **作者:** Xin Chen; Junchao Wu; Shu Yang; Runzhe Zhan; Zeyu Wu; Min Yang; Shujian Huang; Lidia S. Chao; Derek F. Wong
>
> **摘要:** Instruction Tuning (IT) has been proven to be an effective approach to unlock the powerful capabilities of large language models (LLMs). Recent studies indicate that excessive IT data can degrade LLMs performance, while carefully selecting a small subset of high-quality IT data can significantly enhance their capabilities. Therefore, identifying the most efficient subset data from the IT dataset to effectively develop either specific or general abilities in LLMs has become a critical challenge. To address this, we propose a novel and efficient framework called NAIT. NAIT evaluates the impact of IT data on LLMs performance by analyzing the similarity of neuron activation patterns between the IT dataset and the target domain capability. Specifically, NAIT captures neuron activation patterns from in-domain datasets of target domain capabilities to construct reusable and transferable neuron activation features. It then evaluates and selects optimal samples based on the similarity between candidate samples and the expected activation features of the target capabilities. Experimental results show that training on the 10\% Alpaca-GPT4 IT data subset selected by NAIT consistently outperforms methods that rely on external advanced models or uncertainty-based features across various tasks. Our findings also reveal the transferability of neuron activation features across different capabilities of LLMs. In particular, IT data with more logical reasoning and programmatic features possesses strong general transferability, enabling models to develop stronger capabilities across multiple tasks, while a stable core subset of data is sufficient to consistently activate fundamental model capabilities and universally improve performance across diverse tasks.
>
---
#### [new 031] 98$\times$ Faster LLM Routing Without a Dedicated GPU: Flash Attention, Prompt Compression, and Near-Streaming for the vLLM Semantic Router
- **分类: cs.CL**

- **简介: 该论文针对大语言模型路由任务，解决其在GPU内存和延迟上的瓶颈。通过优化注意力机制、提示压缩和近流处理，实现98倍加速，使长文本路由可在共享GPU上高效运行。**

- **链接: [https://arxiv.org/pdf/2603.12646](https://arxiv.org/pdf/2603.12646)**

> **作者:** Xunzhuo Liu; Bowei He; Xue Liu; Andy Luo; Haichen Zhang; Huamin Chen
>
> **摘要:** System-level routers that intercept LLM requests for safety classification, domain routing, and PII detection must be both fast and operationally lightweight: they should add minimal latency to every request, yet not require a dedicated GPU -- an expensive resource better used for LLM inference itself. When the router co-locates on the same GPU as vLLM serving instances, standard attention's $O(n^2)$ memory makes long-context classification (8K--32K tokens) impossible: at 8K tokens, three concurrent classifiers need ${\sim}$4.5\,GB for attention masks alone, far exceeding the memory left by vLLM. We present three staged optimizations for the vLLM Semantic Router, benchmarked on AMD Instinct MI300X, that solve both the latency and the memory problem. \emph{Stage~1}: a custom CK Flash Attention operator for ONNX Runtime on ROCm reduces attention memory from $O(n^2)$ to $O(n)$ and end-to-end (E2E) latency from 4{,}918\,ms to 127\,ms (\textbf{38.7$\times$}), enabling 8K--32K tokens where SDPA OOMs. \emph{Stage~2}: classical NLP prompt compression (TextRank, position weighting, TF-IDF, and novelty scoring) reduces all inputs to ${\sim}$512 tokens without neural inference, capping both latency and GPU memory at a constant regardless of original prompt length (E2E 127$\to$62\,ms, \textbf{2.0$\times$}). \emph{Stage~3}: near-streaming body processing with adaptive chunking and zero-copy JSON eliminates serialization overhead (E2E 62$\to$50\,ms, \textbf{1.2$\times$}). Cumulatively: \textbf{98$\times$} improvement (4{,}918\,ms to 50\,ms), 16K-token routing in 108\,ms, and a total router GPU footprint under 800\,MB -- small enough to share a GPU with LLM serving and removing the need for a dedicated accelerator. Stage~1 targets AMD ROCm (NVIDIA GPUs already have FlashAttention via cuDNN); Stages~2 and~3 are hardware-agnostic.
>
---
#### [new 032] DS$^2$-Instruct: Domain-Specific Data Synthesis for Large Language Models Instruction Tuning
- **分类: cs.CL**

- **简介: 该论文属于指令微调任务，旨在解决领域专用数据生成困难的问题。提出DS$^2$-Instruct框架，无需人工标注生成高质量领域指令数据。**

- **链接: [https://arxiv.org/pdf/2603.12932](https://arxiv.org/pdf/2603.12932)**

> **作者:** Ruiyao Xu; Noelle I. Samia; Han Liu
>
> **摘要:** Adapting Large Language Models (LLMs) to specialized domains requires high-quality instruction tuning datasets, which are expensive to create through human annotation. Existing data synthesis methods focus on general-purpose tasks and fail to capture domain-specific terminology and reasoning patterns. To address this, we introduce DS$^2$-Instruct, a zero-shot framework that generates domain-specific instruction datasets without human supervision. Our approach first generates task-informed keywords to ensure comprehensive domain coverage. It then creates diverse instructions by pairing these keywords with different cognitive levels from Bloom's Taxonomy. Finally, it uses self-consistency validation to ensure data quality. We apply this framework to generate datasets across seven challenging domains, such as mathematics, finance, and logical reasoning. Comprehensive evaluation demonstrates that models fine-tuned on our generated data achieve substantial improvements over existing data generation methods.
>
---
#### [new 033] MetaKE: Meta-learning Aligned Knowledge Editing via Bi-level Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识编辑任务，解决LLM中知识修正与通用能力保持的矛盾。提出MetaKE框架，通过双层优化实现语义与执行对齐，提升编辑效果。**

- **链接: [https://arxiv.org/pdf/2603.12677](https://arxiv.org/pdf/2603.12677)**

> **作者:** Shuxin Liu; Ou Wu
>
> **备注:** 17 pages, 2 figures
>
> **摘要:** Knowledge editing (KE) aims to precisely rectify specific knowledge in Large Language Models (LLMs) without disrupting general capabilities. State-of-the-art methods suffer from an open-loop control mismatch. We identify a critical "Semantic-Execution Disconnect": the semantic target is derived independently without feedback from the downstream's feasible region. This misalignment often causes valid semantic targets to fall within the prohibited space, resulting in gradient truncation and editing failure. To bridge this gap, we propose MetaKE (Meta-learning Aligned Knowledge Editing), a new framework that reframes KE as a bi-level optimization problem. Departing from static calculation, MetaKE treats the edit target as a learnable meta-parameter: the upper-level optimizer seeks a feasible target to maximize post-edit performance, while the lower-level solver executes the editing. To address the challenge of differentiating through complex solvers, we derive a Structural Gradient Proxy, which explicitly backpropagates editability constraints to the target learning phase. Theoretical analysis demonstrates that MetaKE automatically aligns the edit direction with the model's feasible manifold. Extensive experiments confirm that MetaKE significantly outperforms strong baselines, offering a new perspective on knowledge editing.
>
---
#### [new 034] TASTE-Streaming: Towards Streamable Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音语言建模任务，解决语音与文本对齐的延迟问题。提出TASTE-S，实现实时流式处理，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2603.12350](https://arxiv.org/pdf/2603.12350)**

> **作者:** Liang-Hsuan Tseng; Hung-yi Lee
>
> **备注:** Work in progress
>
> **摘要:** Text-speech joint spoken language modeling (SLM) aims at natural and intelligent speech-based interactions, but developing such a system may suffer from modality mismatch: speech unit sequences are much longer than text tokens. Prior work reduces this gap with text-aligned tokenization and embedding (TASTE), producing speech tokens that align in lengths with their textual counterparts. However, the dependence on an external ASR system and the use of a non-causal decoder limits streaming use. To address this limitation, we propose TASTE-S, a streamable extension of TASTE suitable for real-time usage. TASTE-S integrates a CTC-based ASR module into the encoder for instant dual-modality encoding. We also redesign the unit decoder to enable on-the-fly decoding. With joint training, we show that TASTE-S matches TASTE's performance while significantly reducing latency. Further investigations reveal that TASTE-S remains robust to transcriptions and enables long-form encoding and decoding.
>
---
#### [new 035] Interpreting Negation in GPT-2: Layer- and Head-Level Causal Analysis
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语义理解任务，旨在解决语言模型对否定句处理不佳的问题。通过分析GPT-2的层和注意力头，探究否定信号的内部机制。**

- **链接: [https://arxiv.org/pdf/2603.12423](https://arxiv.org/pdf/2603.12423)**

> **作者:** Abdullah Al Mofael; Lisa M. Kuhn; Ghassan Alkadi; Kuo-Pao Yang
>
> **备注:** 9 pages, 4 figures, 1 table. Accepted at the 2026 IEEE 16th Annual Computing and Communication Workshop and Conference (CCWC)
>
> **摘要:** Negation remains a persistent challenge for modern language models, often causing reversed meanings or factual errors. In this work, we conduct a causal analysis of how GPT-2 Small internally processes such linguistic transformations. We examine its hidden representations at both the layer and head level. Our analysis is based on a self-curated 12,000-pair dataset of matched affirmative and negated sentences, covering multiple linguistic templates and forms of negation. To quantify this behavior, we define a metric, the Negation Effect Score (NES), which measures the model's sensitivity in distinguishing between affirmative statements and their negations. We carried out two key interventions to probe causal structure. In activation patching, internal activations from affirmative sentences were inserted into their negated counterparts to see how meaning shifted. In ablation, specific attention heads were temporarily disabled to observe how logical polarity changed. Together, these steps revealed how negation signals move and evolve through GPT-2's layers. Our findings indicate that this capability is not widespread; instead, it is highly concentrated within a limited number of mid-layer attention heads, primarily within layers 4 to 6. Ablating these specific components directly disrupts the model's negation sensitivity: on our in-domain, ablation increased NES (indicating weaker negation sensitivity), and re-introducing cached affirmative activations (rescue) increased NES further, confirming that these heads carry affirmative signal rather than restoring baseline behavior. On xNot360, ablation slightly decreased NES and rescue restored performance above baseline. This pattern demonstrates that these causal patterns are consistent across various negation forms and remain detectable on the external xNot360 benchmark, though with smaller magnitude.
>
---
#### [new 036] ESG-Bench: Benchmarking Long-Context ESG Reports for Hallucination Mitigation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ESG-Bench，用于评估和减少大语言模型在ESG报告分析中的幻觉问题。任务是提升模型在长文本中的事实准确性。工作包括构建标注数据集、设计推理提示策略并优化模型。**

- **链接: [https://arxiv.org/pdf/2603.13154](https://arxiv.org/pdf/2603.13154)**

> **作者:** Siqi Sun; Ben Peng Wu; Mali Jin; Peizhen Bai; Hanpei Zhang; Xingyi Song
>
> **备注:** To be published in the AAAI 2026 proceedings
>
> **摘要:** As corporate responsibility increasingly incorporates environmental, social, and governance (ESG) criteria, ESG reporting is becoming a legal requirement in many regions and a key channel for documenting sustainability practices and assessing firms' long-term and ethical performance. However, the length and complexity of ESG disclosures make them difficult to interpret and automate the analysis reliably. To support scalable and trustworthy analysis, this paper introduces ESG-Bench, a benchmark dataset for ESG report understanding and hallucination mitigation in large language models (LLMs). ESG-Bench contains human-annotated question-answer (QA) pairs grounded in real-world ESG report contexts, with fine-grained labels indicating whether model outputs are factually supported or hallucinated. Framing ESG report analysis as a QA task with verifiability constraints enables systematic evaluation of LLMs' ability to extract and reason over ESG content and provides a new use case: mitigating hallucinations in socially sensitive, compliance-critical settings. We design task-specific Chain-of-Thought (CoT) prompting strategies and fine-tune multiple state-of-the-art LLMs on ESG-Bench using CoT-annotated rationales. Our experiments show that these CoT-based methods substantially outperform standard prompting and direct fine-tuning in reducing hallucinations, and that the gains transfer to existing QA benchmarks beyond the ESG domain.
>
---
#### [new 037] Aligning Language Models from User Interactions
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决语言模型对齐与个性化问题。通过分析用户交互数据，提出一种基于自我蒸馏的方法，提升模型对齐和指令遵循能力，并实现持续适应。**

- **链接: [https://arxiv.org/pdf/2603.12273](https://arxiv.org/pdf/2603.12273)**

> **作者:** Thomas Kleine Buening; Jonas Hübotter; Barna Pásztor; Idan Shenfeld; Giorgia Ramponi; Andreas Krause
>
> **摘要:** Multi-turn user interactions are among the most abundant data produced by language models, yet we lack effective methods to learn from them. While typically discarded, these interactions often contain useful information: follow-up user messages may indicate that a response was incorrect, failed to follow an instruction, or did not align with the user's preferences. Importantly, language models are already able to make use of this information in context. After observing a user's follow-up, the same model is often able to revise its behavior. We leverage this ability to propose a principled and scalable method for learning directly from user interactions through self-distillation. By conditioning the model on the user's follow-up message and comparing the resulting token distribution with the original policy, we obtain a target for updating the policy that captures how the model's behavior changes in hindsight. We then distill this hindsight distribution back into the current policy. Remarkably, we show that training on real-world user conversations from WildChat improves language models across standard alignment and instruction-following benchmarks, without regressing other capabilities. The same mechanism enables personalization, allowing models to continually adapt to individual users through interaction without explicit feedback. Our results demonstrate that raw user interactions that arise naturally during deployment enable alignment, personalization, and continual adaptation.
>
---
#### [new 038] From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于时间序列预测任务，解决文本与时间序列间的模态差距问题。通过引入Temporal Evolution Semantic Space，将文本语义转化为可预测的数值特征，提升预测精度。**

- **链接: [https://arxiv.org/pdf/2603.12664](https://arxiv.org/pdf/2603.12664)**

> **作者:** Lehui Li; Yuyao Wang; Jisheng Yan; Wei Zhang; Jinliang Deng; Haoliang Sun; Zhongyi Han; Yongshun Gong
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** Incorporating textual information into time-series forecasting holds promise for addressing event-driven non-stationarity; however, a fundamental modality gap hinders effective fusion: textual descriptions express temporal impacts implicitly and qualitatively, whereas forecasting models rely on explicit and quantitative signals. Through controlled semi-synthetic experiments, we show that existing methods over-attend to redundant tokens and struggle to reliably translate textual semantics into usable numerical cues. To bridge this gap, we propose TESS, which introduces a Temporal Evolution Semantic Space as an intermediate bottleneck between modalities. This space consists of interpretable, numerically grounded temporal primitives (mean shift, volatility, shape, and lag) extracted from text by an LLM via structured prompting and filtered through confidence-aware gating. Experiments on four real-world datasets demonstrate up to a 29 percent reduction in forecasting error compared to state-of-the-art unimodal and multimodal baselines. The code will be released after acceptance.
>
---
#### [new 039] CLARIN-PT-LDB: An Open LLM Leaderboard for Portuguese to assess Language, Culture and Civility
- **分类: cs.CL**

- **简介: 该论文提出CLARIN-PT-LDB，一个针对欧洲葡萄牙语的开放大语言模型排行榜及其基准测试。旨在解决葡萄牙语LLM评估不足的问题，特别关注文化适配与模型安全。**

- **链接: [https://arxiv.org/pdf/2603.12872](https://arxiv.org/pdf/2603.12872)**

> **作者:** João Silva; Luís Gomes; António Branco
>
> **备注:** Accepted at PROPOR 2026
>
> **摘要:** This paper reports on the development of a leaderboard of Open Large Language Models (LLM) for European Portuguese (PT-PT), and on its associated benchmarks. This leaderboard comes as a way to address a gap in the evaluation of LLM for European Portuguese, which so far had no leaderboard dedicated to this variant of the language. The paper also reports on novel benchmarks, including some that address aspects of performance that so far have not been available in benchmarks for European Portuguese, namely model safeguards and alignment to Portuguese culture. The leaderboard is available at this https URL.
>
---
#### [new 040] CSE-UOI at SemEval-2026 Task 6: A Two-Stage Heterogeneous Ensemble with Deliberative Complexity Gating for Political Evasion Detection
- **分类: cs.CL**

- **简介: 该论文针对政治访谈回应清晰度分类任务，解决模糊回应检测问题。提出双模型集成与复杂度门控机制，提升分类效果。**

- **链接: [https://arxiv.org/pdf/2603.12453](https://arxiv.org/pdf/2603.12453)**

> **作者:** Christos Tzouvaras; Konstantinos Skianis; Athanasios Voulodimos
>
> **摘要:** This paper describes our system for SemEval-2026 Task 6, which classifies clarity of responses in political interviews into three categories: Clear Reply, Ambivalent, and Clear Non-Reply. We propose a heterogeneous dual large language model (LLM) ensemble via self-consistency (SC) and weighted voting, and a novel post-hoc correction mechanism, Deliberative Complexity Gating (DCG). This mechanism uses cross-model behavioral signals and exploits the finding that an LLM response-length proxy correlates strongly with sample ambiguity. To further examine mechanisms for improving ambiguity detection, we evaluated multi-agent debate as an alternative strategy for increasing deliberative capacity. Unlike DCG, which adaptively gates reasoning using cross-model behavioral signals, debate increases agent count without increasing model diversity. Our solution achieved a Macro-F1 score of 0.85 on the evaluation set, securing 3rd place.
>
---
#### [new 041] Developing and evaluating a chatbot to support maternal health care
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于医疗聊天机器人任务，旨在解决低资源地区孕产妇健康信息支持问题。通过多方法评估和防御性设计，提升聊天机器人的准确性和安全性。**

- **链接: [https://arxiv.org/pdf/2603.13168](https://arxiv.org/pdf/2603.13168)**

> **作者:** Smriti Jha; Vidhi Jain; Jianyu Xu; Grace Liu; Sowmya Ramesh; Jitender Nagpal; Gretchen Chapman; Benjamin Bellows; Siddhartha Goyal; Aarti Singh; Bryan Wilder
>
> **备注:** 17 pages; submitted to IJCAI 2026 AI and Social Good Track
>
> **摘要:** The ability to provide trustworthy maternal health information using phone-based chatbots can have a significant impact, particularly in low-resource settings where users have low health literacy and limited access to care. However, deploying such systems is technically challenging: user queries are short, underspecified, and code-mixed across languages, answers require regional context-specific grounding, and partial or missing symptom context makes safe routing decisions difficult. We present a chatbot for maternal health in India developed through a partnership between academic researchers, a health tech company, a public health nonprofit, and a hospital. The system combines (1) stage-aware triage, routing high-risk queries to expert templates, (2) hybrid retrieval over curated maternal/newborn guidelines, and (3) evidence-conditioned generation from an LLM. Our core contribution is an evaluation workflow for high-stakes deployment under limited expert supervision. Targeting both component-level and end-to-end testing, we introduce: (i) a labeled triage benchmark (N=150) achieving 86.7% emergency recall, explicitly reporting the missed-emergency vs. over-escalation trade-off; (ii) a synthetic multi-evidence retrieval benchmark (N=100) with chunk-level evidence labels; (iii) LLM-as-judge comparison on real queries (N=781) using clinician-codesigned criteria; and (iv) expert validation. Our findings show that trustworthy medical assistants in multilingual, noisy settings require defense-in-depth design paired with multi-method evaluation, rather than any single model and evaluation method choice.
>
---
#### [new 042] Self-Supervised Speech Models Encode Phonetic Context via Position-dependent Orthogonal Subspaces
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文研究自监督语音模型如何编码音素及其上下文，属于语音表示学习任务。解决的问题是理解单帧表示如何组合音素信息。工作包括提出音素向量在位置相关正交子空间中叠加的结构。**

- **链接: [https://arxiv.org/pdf/2603.12642](https://arxiv.org/pdf/2603.12642)**

> **作者:** Kwanghee Choi; Eunjung Yeo; Cheol Jun Cho; David R. Mortensen; David Harwath
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Transformer-based self-supervised speech models (S3Ms) are often described as contextualized, yet what this entails remains unclear. Here, we focus on how a single frame-level S3M representation can encode phones and their surrounding context. Prior work has shown that S3Ms represent phones compositionally; for example, phonological vectors such as voicing, bilabiality, and nasality vectors are superposed in the S3M representation of [m]. We extend this view by proposing that phonological information from a sequence of neighboring phones is also compositionally encoded in a single frame, such that vectors corresponding to previous, current, and next phones are superposed within a single frame-level representation. We show that this structure has several properties, including orthogonality between relative positions, and emergence of implicit phonetic boundaries. Together, our findings advance our understanding of context-dependent S3M representations.
>
---
#### [new 043] Reinforcement Learning for Diffusion LLMs with Entropy-Guided Step Selection and Stepwise Advantages
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型训练任务，解决扩散语言模型（DLMs）的强化学习问题。通过构建马尔可夫决策过程，提出无偏策略梯度方法，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2603.12554](https://arxiv.org/pdf/2603.12554)**

> **作者:** Vishnu Teja Kunde; Fatemeh Doudi; Mahdi Farahbakhsh; Dileep Kalathil; Krishna Narayanan; Jean-Francois Chamberland
>
> **摘要:** Reinforcement learning (RL) has been effective for post-training autoregressive (AR) language models, but extending these methods to diffusion language models (DLMs) is challenging due to intractable sequence-level likelihoods. Existing approaches therefore rely on surrogate likelihoods or heuristic approximations, which can introduce bias and obscure the sequential structure of denoising. We formulate diffusion-based sequence generation as a finite-horizon Markov decision process over the denoising trajectory and derive an exact, unbiased policy gradient that decomposes over denoising steps and is expressed in terms of intermediate advantages, without requiring explicit evaluation of the sequence likelihood. To obtain a practical and compute-efficient estimator, we (i) select denoising steps for policy updates via an entropy-guided approximation bound, and (ii) estimate intermediate advantages using a one-step denoising reward naturally provided by the diffusion model, avoiding costly multi-step rollouts. Experiments on coding and logical reasoning benchmarks demonstrate state-of-the-art results, with strong competitive performance on mathematical reasoning, outperforming existing RL post-training approaches for DLMs. Code is available at this https URL.
>
---
#### [new 044] Literary Narrative as Moral Probe : A Cross-System Framework for Evaluating AI Ethical Reasoning and Refusal Behavior
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于AI伦理评估任务，旨在解决现有框架仅测试表面正确回应的问题。通过文学叙事设计实验，评估AI的道德推理与拒绝行为，发现系统能力与评估精度相关。**

- **链接: [https://arxiv.org/pdf/2603.12615](https://arxiv.org/pdf/2603.12615)**

> **作者:** David C. Flynn
>
> **备注:** 27 pages, 6 tables. Target: Minds and Machines (Springer)
>
> **摘要:** Existing AI moral evaluation frameworks test for the production of correct-sounding ethical responses rather than the presence of genuine moral reasoning capacity. This paper introduces a novel probe methodology using literary narrative - specifically, unresolvable moral scenarios drawn from a published science fiction series - as stimulus material structurally resistant to surface performance. We present results from a 24-condition cross-system study spanning 13 distinct systems across two series: Series 1 (frontier commercial systems, blind; n=7) and Series 2 (local and API open-source systems, blind and declared; n=6). Four Series 2 systems were re-administered under declared conditions (13 blind + 4 declared + 7 ceiling probe = 24 total conditions), yielding zero delta across all 16 dimension-pair comparisons. Probe administration was conducted by two human raters across three machines; primary blind scoring was performed by Claude (Anthropic) as LLM judge, with Gemini Pro (Google) and Copilot Pro (Microsoft) serving as independent judges for the ceiling discrimination probe. A supplemental theological differentiator probe yielded perfect rank-order agreement between the two independent ceiling probe judges (Gemini Pro and Copilot Pro; rs = 1.00). Five qualitatively distinct D3 reflexive failure modes were identified - including categorical self-misidentification and false positive self-attribution - suggesting that instrument sophistication scales with system capability rather than being circumvented by it. We argue that literary narrative constitutes an anticipatory evaluation instrument - one that becomes more discriminating as AI capability increases - and that the gap between performed and authentic moral reasoning is measurable, meaningful, and consequential for deployment decisions in high-stakes domains.
>
---
#### [new 045] daVinci-Env: Open SWE Environment Synthesis at Scale
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出OpenSWE，一个大规模、透明的软件工程代理训练环境，解决数据规模小和工业方案不透明的问题，通过自动化构建45,320个可执行Docker环境，提升训练效率与效果。**

- **链接: [https://arxiv.org/pdf/2603.13023](https://arxiv.org/pdf/2603.13023)**

> **作者:** Dayuan Fu; Shenyu Wu; Yunze Wu; Zerui Peng; Yaxing Huang; Jie Sun; Ji Zeng; Mohan Jiang; Lin Zhang; Yukun Li; Jiarui Hu; Liming Liu; Jinlong Hou; Pengfei Liu
>
> **摘要:** Training capable software engineering (SWE) agents demands large-scale, executable, and verifiable environments that provide dynamic feedback loops for iterative code editing, test execution, and solution refinement. However, existing open-source datasets remain limited in scale and repository diversity, while industrial solutions are opaque with unreleased infrastructure, creating a prohibitive barrier for most academic research groups. We present OpenSWE, the largest fully transparent framework for SWE agent training in Python, comprising 45,320 executable Docker environments spanning over 12.8k repositories, with all Dockerfiles, evaluation scripts, and infrastructure fully open-sourced for reproducibility. OpenSWE is built through a multi-agent synthesis pipeline deployed across a 64-node distributed cluster, automating repository exploration, Dockerfile construction, evaluation script generation, and iterative test analysis. Beyond scale, we propose a quality-centric filtering pipeline that characterizes the inherent difficulty of each environment, filtering out instances that are either unsolvable or insufficiently challenging and retaining only those that maximize learning efficiency. With $891K spent on environment construction and an additional $576K on trajectory sampling and difficulty-aware curation, the entire project represents a total investment of approximately $1.47 million, yielding about 13,000 curated trajectories from roughly 9,000 quality guaranteed environments. Extensive experiments validate OpenSWE's effectiveness: OpenSWE-32B and OpenSWE-72B achieve 62.4% and 66.0% on SWE-bench Verified, establishing SOTA among Qwen2.5 series. Moreover, SWE-focused training yields substantial out-of-domain improvements, including up to 12 points on mathematical reasoning and 5 points on science benchmarks, without degrading factual recall.
>
---
#### [new 046] Semantic Invariance in Agentic AI
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI可靠性研究任务，旨在解决LLM在语义相似输入下推理稳定性不足的问题。通过构建测试框架，评估不同模型的鲁棒性，发现模型规模与稳定性无直接关联。**

- **链接: [https://arxiv.org/pdf/2603.13173](https://arxiv.org/pdf/2603.13173)**

> **作者:** I. de Zarzà; J. de Curtò; Jordi Cabot; Pietro Manzoni; Carlos T. Calafate
>
> **备注:** Accepted for publication in 20th International Conference on Agents and Multi-Agent Systems: Technologies and Applications (AMSTA 2026), to appear in Springer Nature proceedings (KES Smart Innovation Systems and Technologies). The final authenticated version will be available online at Springer
>
> **摘要:** Large Language Models (LLMs) increasingly serve as autonomous reasoning agents in decision support, scientific problem-solving, and multi-agent coordination systems. However, deploying LLM agents in consequential applications requires assurance that their reasoning remains stable under semantically equivalent input variations, a property we term semantic this http URL benchmark evaluations, which assess accuracy on fixed, canonical problem formulations, fail to capture this critical reliability dimension. To address this shortcoming, in this paper we present a metamorphic testing framework for systematically assessing the robustness of LLM reasoning agents, applying eight semantic-preserving transformations (identity, paraphrase, fact reordering, expansion, contraction, academic context, business context, and contrastive formulation) across seven foundation models spanning four distinct architectural families: Hermes (70B, 405B), Qwen3 (30B-A3B, 235B-A22B), DeepSeek-R1, and gpt-oss (20B, 120B). Our evaluation encompasses 19 multi-step reasoning problems across eight scientific domains. The results reveal that model scale does not predict robustness: the smaller Qwen3-30B-A3B achieves the highest stability (79.6% invariant responses, semantic similarity 0.91), while larger models exhibit greater fragility.
>
---
#### [new 047] NeuroLoRA: Context-Aware Neuromodulation for Parameter-Efficient Multi-Task Adaptation
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出NeuroLoRA，解决多任务适应中的参数效率问题。通过引入上下文感知的神经调节机制，提升模型在不同任务间的适应能力。**

- **链接: [https://arxiv.org/pdf/2603.12378](https://arxiv.org/pdf/2603.12378)**

> **作者:** Yuxin Yang; Haoran Zhang; Mingxuan Li; Jiachen Xu; Ruoxi Shen; Zhenyu Wang; Tianhao Liu; Siqi Chen; Weilin Huang
>
> **备注:** work in progress
>
> **摘要:** Parameter-Efficient Fine-Tuning (PEFT) techniques, particularly Low-Rank Adaptation (LoRA), have become essential for adapting Large Language Models (LLMs) to downstream tasks. While the recent FlyLoRA framework successfully leverages bio-inspired sparse random projections to mitigate parameter interference, it relies on a static, magnitude-based routing mechanism that is agnostic to input context. In this paper, we propose NeuroLoRA, a novel Mixture-of-Experts (MoE) based LoRA framework inspired by biological neuromodulation -- the dynamic regulation of neuronal excitability based on context. NeuroLoRA retains the computational efficiency of frozen random projections while introducing a lightweight, learnable neuromodulation gate that contextually rescales the projection space prior to expert selection. We further propose a Contrastive Orthogonality Loss to explicitly enforce separation between expert subspaces, enhancing both task decoupling and continual learning capacity. Extensive experiments on MMLU, GSM8K, and ScienceQA demonstrate that NeuroLoRA consistently outperforms FlyLoRA and other strong baselines across single-task adaptation, multi-task model merging, and sequential continual learning scenarios, while maintaining comparable parameter efficiency.
>
---
#### [new 048] TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于推理任务，解决大型推理模型过早生成答案后仍过度计算的问题。通过设计TERMINATOR策略，提前终止推理过程，显著减少计算量且不影响性能。**

- **链接: [https://arxiv.org/pdf/2603.12529](https://arxiv.org/pdf/2603.12529)**

> **作者:** Alliot Nagle; Jakhongir Saydaliev; Dhia Garbaya; Michael Gastpar; Ashok Vardhan Makkuva; Hyeji Kim
>
> **备注:** 35 pages, 31 figures
>
> **摘要:** Large Reasoning Models (LRMs) achieve impressive performance on complex reasoning tasks via Chain-of-Thought (CoT) reasoning, which enables them to generate intermediate thinking tokens before arriving at the final answer. However, LRMs often suffer from significant overthinking, spending excessive compute time even after the answer is generated early on. Prior work has identified the existence of an optimal reasoning length such that truncating reasoning at this point significantly shortens CoT outputs with virtually no change in performance. However, determining optimal CoT lengths for practical datasets is highly non-trivial as they are fully task and model-dependent. In this paper, we precisely address this and design TERMINATOR, an early-exit strategy for LRMs at inference to mitigate overthinking. The central idea underpinning TERMINATOR is that the first arrival of an LRM's final answer is often predictable, and we leverage these first answer positions to create a novel dataset of optimal reasoning lengths to train TERMINATOR. Powered by this approach, TERMINATOR achieves significant reductions in CoT lengths of 14%-55% on average across four challenging practical datasets: MATH-500, AIME 2025, HumanEval, and GPQA, whilst outperforming current state-of-the-art methods.
>
---
#### [new 049] AI Planning Framework for LLM-Based Web Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI代理任务，旨在解决LLM代理规划不透明的问题。通过构建规划框架，分析不同代理架构，并提出新评估指标以优化代理选择。**

- **链接: [https://arxiv.org/pdf/2603.12710](https://arxiv.org/pdf/2603.12710)**

> **作者:** Orit Shahnovsky; Rotem Dror
>
> **摘要:** Developing autonomous agents for web-based tasks is a core challenge in AI. While Large Language Model (LLM) agents can interpret complex user requests, they often operate as black boxes, making it difficult to diagnose why they fail or how they plan. This paper addresses this gap by formally treating web tasks as sequential decision-making processes. We introduce a taxonomy that maps modern agent architectures to traditional planning paradigms: Step-by-Step agents to Breadth-First Search (BFS), Tree Search agents to Best-First Tree Search, and Full-Plan-in-Advance agents to Depth-First Search (DFS). This framework allows for a principled diagnosis of system failures like context drift and incoherent task decomposition. To evaluate these behaviors, we propose five novel evaluation metrics that assess trajectory quality beyond simple success rates. We support this analysis with a new dataset of 794 human-labeled trajectories from the WebArena benchmark. Finally, we validate our evaluation framework by comparing a baseline Step-by-Step agent against a novel Full-Plan-in-Advance implementation. Our results reveal that while the Step-by-Step agent aligns more closely with human gold trajectories (38% overall success), the Full-Plan-in-Advance agent excels in technical measures such as element accuracy (89%), demonstrating the necessity of our proposed metrics for selecting appropriate agent architectures based on specific application constraints.
>
---
#### [new 050] MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出MoKus框架，解决知识感知概念定制任务中的稀有标记性能不稳定问题，通过跨模态知识迁移实现高质量生成。**

- **链接: [https://arxiv.org/pdf/2603.12743](https://arxiv.org/pdf/2603.12743)**

> **作者:** Chenyang Zhu; Hongxiang Li; Xiu Li; Long Chen
>
> **备注:** Project Page: this https URL
>
> **摘要:** Concept customization typically binds rare tokens to a target concept. Unfortunately, these approaches often suffer from unstable performance as the pretraining data seldom contains these rare tokens. Meanwhile, these rare tokens fail to convey the inherent knowledge of the target concept. Consequently, we introduce Knowledge-aware Concept Customization, a novel task aiming at binding diverse textual knowledge to target visual concepts. This task requires the model to identify the knowledge within the text prompt to perform high-fidelity customized generation. Meanwhile, the model should efficiently bind all the textual knowledge to the target concept. Therefore, we propose MoKus, a novel framework for knowledge-aware concept customization. Our framework relies on a key observation: cross-modal knowledge transfer, where modifying knowledge within the text modality naturally transfers to the visual modality during generation. Inspired by this observation, MoKus contains two stages: (1) In visual concept learning, we first learn the anchor representation to store the visual information of the target concept. (2) In textual knowledge updating, we update the answer for the knowledge queries to the anchor representation, enabling high-fidelity customized generation. To further comprehensively evaluate our proposed MoKus on the new task, we introduce the first benchmark for knowledge-aware concept customization: KnowCusBench. Extensive evaluations have demonstrated that MoKus outperforms state-of-the-art methods. Moreover, the cross-model knowledge transfer allows MoKus to be easily extended to other knowledge-aware applications like virtual concept creation and concept erasure. We also demonstrate the capability of our method to achieve improvements on world knowledge benchmarks.
>
---
#### [new 051] Multi-Step Semantic Reasoning in Generative Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决生成式检索模型在数值语义推理上的不足。提出ReasonGR框架，通过结构化提示和推理模块提升复杂查询的检索效果。**

- **链接: [https://arxiv.org/pdf/2603.12368](https://arxiv.org/pdf/2603.12368)**

> **作者:** Steven Dong; Yubao Tang; Maarten de Rijke
>
> **备注:** Accepted at ECIR2026
>
> **摘要:** Generative retrieval (GR) models encode a corpus within model parameters and generate relevant document identifiers directly for a given query. While this paradigm shows promise in retrieval tasks, existing GR models struggle with complex queries in numerical contexts, such as those involving semantic reasoning over financial reports, due to limited reasoning capabilities. This limitation leads to suboptimal retrieval accuracy and hinders practical applicability. We propose ReasonGR, a framework designed to enhance multi-step semantic reasoning in numerical contexts within GR. ReasonGR employs a structured prompting strategy combining task-specific instructions with stepwise reasoning guidance to better address complex retrieval queries. Additionally, it integrates a reasoning-focused adaptation module to improve the learning of reasoning-related parameters. Experiments on the FinQA dataset, which contains financial queries over complex documents, demonstrate that ReasonGR improves retrieval accuracy and consistency, indicating its potential for advancing GR models in reasoning-intensive retrieval scenarios.
>
---
#### [new 052] Red-Teaming Vision-Language-Action Models via Quality Diversity Prompt Generation for Robust Robot Policies
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于机器人控制任务，旨在提升视觉-语言-动作模型的鲁棒性。通过生成多样化且自然的指令，识别模型漏洞并优化其性能。**

- **链接: [https://arxiv.org/pdf/2603.12510](https://arxiv.org/pdf/2603.12510)**

> **作者:** Siddharth Srikanth; Freddie Liang; Sophie Hsu; Varun Bhatt; Shihan Zhao; Henry Chen; Bryon Tjanaka; Minjune Hwang; Akanksha Saran; Daniel Seita; Aaquib Tabrez; Stefanos Nikolaidis
>
> **摘要:** Vision-Language-Action (VLA) models have significant potential to enable general-purpose robotic systems for a range of vision-language tasks. However, the performance of VLA-based robots is highly sensitive to the precise wording of language instructions, and it remains difficult to predict when such robots will fail. To improve the robustness of VLAs to different wordings, we present Q-DIG (Quality Diversity for Diverse Instruction Generation), which performs red-teaming by scalably identifying diverse natural language task descriptions that induce failures while remaining task-relevant. Q-DIG integrates Quality Diversity (QD) techniques with Vision-Language Models (VLMs) to generate a broad spectrum of adversarial instructions that expose meaningful vulnerabilities in VLA behavior. Our results across multiple simulation benchmarks show that Q-DIG finds more diverse and meaningful failure modes compared to baseline methods, and that fine-tuning VLAs on the generated instructions improves task success rates. Furthermore, results from a user study highlight that Q-DIG generates prompts judged to be more natural and human-like than those from baselines. Finally, real-world evaluations of Q-DIG prompts show results consistent with simulation, and fine-tuning VLAs on the generated prompts further success rates on unseen instructions. Together, these findings suggest that Q-DIG is a promising approach for identifying vulnerabilities and improving the robustness of VLA-based robots. Our anonymous project website is at this http URL.
>
---
#### [new 053] Speech-Worthy Alignment for Japanese SpeechLLMs via Direct Preference Optimization
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音生成任务，旨在解决日本SpeechLLM输出风格不适合语音合成的问题。通过偏好对齐方法优化输出，使其更口语化、自然。**

- **链接: [https://arxiv.org/pdf/2603.12565](https://arxiv.org/pdf/2603.12565)**

> **作者:** Mengjie Zhao; Lianbo Liu; Yusuke Fujita; Hao Shi; Yuan Gao; Roman Koshkin; Yui Sudo
>
> **摘要:** SpeechLLMs typically combine ASR-trained encoders with text-based LLM backbones, leading them to inherit written-style output patterns unsuitable for text-to-speech synthesis. This mismatch is particularly pronounced in Japanese, where spoken and written registers differ substantially in politeness markers, sentence-final particles, and syntactic complexity. We propose a preference-based alignment approach to adapt Japanese SpeechLLMs for speech-worthy outputs: text that is concise, conversational, and readily synthesized as natural speech. To rigorously evaluate this task, we introduce SpokenElyza, a benchmark for Japanese speech-worthiness derived from ELYZA-tasks-100 with auditory verification by native experts. Experiments show that our approach achieves substantial improvement on SpokenElyza while largely preserving performance on the original written-style evaluation. We will release SpokenElyza to support future research on Japanese spoken dialog systems.
>
---
#### [new 054] Efficient Reasoning with Balanced Thinking
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于大模型推理任务，旨在解决LRMs的过思与欠思问题。提出ReBalance框架，通过信心动态调整推理路径，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.12372](https://arxiv.org/pdf/2603.12372)**

> **作者:** Yulin Li; Tengyao Tu; Li Ding; Junjie Wang; Huiling Zhen; Yixin Chen; Yong Li; Zhuotao Tian
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Large Reasoning Models (LRMs) have shown remarkable reasoning capabilities, yet they often suffer from overthinking, expending redundant computational steps on simple problems, or underthinking, failing to explore sufficient reasoning paths despite inherent capabilities. These issues lead to inefficiencies and potential inaccuracies, limiting practical deployment in resource-constrained settings. Existing methods to mitigate overthinking, such as suppressing reflective keywords or adjusting reasoning length, may inadvertently induce underthinking, compromising accuracy. Therefore, we propose ReBalance, a training-free framework that achieves efficient reasoning with balanced thinking. ReBalance leverages confidence as a continuous indicator of reasoning dynamics, identifying overthinking through high confidence variance and underthinking via consistent overconfidence. By aggregating hidden states from a small-scale dataset into reasoning mode prototypes, we compute a steering vector to guide LRMs' reasoning trajectories. A dynamic control function modulates this vector's strength and direction based on real-time confidence, pruning redundancy during overthinking, and promoting exploration during underthinking. Extensive experiments conducted on four models ranging from 0.5B to 32B, and across nine benchmarks in math reasoning, general question answering, and coding tasks demonstrate that ReBalance effectively reduces output redundancy while improving accuracy, offering a general, training-free, and plug-and-play strategy for efficient and robust LRM deployment. Code is available at this https URL .
>
---
#### [new 055] FGTR: Fine-Grained Multi-Table Retrieval via Hierarchical LLM Reasoning
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文提出FGTR，解决多表细粒度检索问题，通过分层推理提升检索精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.12702](https://arxiv.org/pdf/2603.12702)**

> **作者:** Chaojie Sun; Bin Cao; Tiantian Li; Chenyu Hou; Ruizhe Li; Qing Fan
>
> **备注:** Under Review - Submitted to SIGIR 2026 Resources Track; 10pages, 5 figures, 4 tables
>
> **摘要:** With the rapid advancement of large language models (LLMs), growing efforts have been made on LLM-based table retrieval. However, existing studies typically focus on single-table query, and implement it by similarity matching after encoding the entire table. These methods usually result in low accuracy due to their coarse-grained encoding which incorporates much query-irrelated data, and are also inefficient when dealing with large tables, failing to fully utilize the reasoning capabilities of LLM. Further, multi-table query is under-explored in retrieval tasks. To this end, we propose a hierarchical multi-table query method based on LLM: Fine-Grained Multi-Table Retrieval FGTR, a new retrieval paradigm that employs a human-like reasoning strategy. Through hierarchical reasoning, FGTR first identifies relevant schema elements and then retrieves the corresponding cell contents, ultimately constructing a concise and accurate sub-table that aligns with the given query. To comprehensively evaluate the performance of FGTR, we construct two new benchmark datasets based on Spider and BIRD . Experimental results show that FGTR outperforms previous state-of-the-art methods, improving the F_2 metric by 18% on Spider and 21% on BIRD, demonstrating its effectiveness in enhancing fine-grained retrieval and its potential to improve end-to-end performance on table-based downstream tasks.
>
---
#### [new 056] Structured Distillation for Personalized Agent Memory: 11x Token Reduction with Retrieval Preservation
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决长对话中记忆存储成本高的问题。通过结构化压缩技术，将对话内容减少至1/11，同时保持检索效果。**

- **链接: [https://arxiv.org/pdf/2603.13017](https://arxiv.org/pdf/2603.13017)**

> **作者:** Sydney Lewis
>
> **备注:** 6 figures. Code: this https URL
>
> **摘要:** Long conversations with an AI agent create a simple problem for one user: the history is useful, but carrying it verbatim is expensive. We study personalized agent memory: one user's conversation history with an agent, distilled into a compact retrieval layer for later search. Each exchange is compressed into a compound object with four fields (exchange_core, specific_context, thematic room_assignments, and regex-extracted files_touched). The searchable distilled text averages 38 tokens per exchange. Applied to 4,182 conversations (14,340 exchanges) from 6 software engineering projects, the method reduces average exchange length from 371 to 38 tokens, yielding 11x compression. We evaluate whether personalized recall survives that compression using 201 recall-oriented queries, 107 configurations spanning 5 pure and 5 cross-layer search modes, and 5 LLM graders (214,519 consensus-graded query-result pairs). The best pure distilled configuration reaches 96% of the best verbatim MRR (0.717 vs 0.745). Results are mechanism-dependent. All 20 vector search configurations remain non-significant after Bonferroni correction, while all 20 BM25 configurations degrade significantly (effect sizes |d|=0.031-0.756). The best cross-layer setup slightly exceeds the best pure verbatim baseline (MRR 0.759). Structured distillation compresses single-user agent memory without uniformly sacrificing retrieval quality. At 1/11 the context cost, thousands of exchanges fit within a single prompt while the verbatim source remains available for drill-down. We release the implementation and analysis pipeline as open-source software.
>
---
#### [new 057] DIALECTIC: A Multi-Agent System for Startup Evaluation
- **分类: cs.MA; cs.CE; cs.CL**

- **简介: 该论文提出DIALECTIC，一个基于大语言模型的多智能体系统，用于初创企业评估。任务是提高风险投资筛选效率，解决评估精力与数量之间的矛盾。通过构建论点并模拟辩论，生成决策评分。**

- **链接: [https://arxiv.org/pdf/2603.12274](https://arxiv.org/pdf/2603.12274)**

> **作者:** Jae Yoon Bae; Simon Malberg; Joyce Galang; Andre Retterath; Georg Groh
>
> **备注:** Accepted at EACL 2026 Industry Track
>
> **摘要:** Venture capital (VC) investors face a large number of investment opportunities but only invest in few of these, with even fewer ending up successful. Early-stage screening of opportunities is often limited by investor bandwidth, demanding tradeoffs between evaluation diligence and number of opportunities assessed. To ease this tradeoff, we introduce DIALECTIC, an LLM-based multi-agent system for startup evaluation. DIALECTIC first gathers factual knowledge about a startup and organizes these facts into a hierarchical question tree. It then synthesizes the facts into natural-language arguments for and against an investment and iteratively critiques and refines these arguments through a simulated debate, which surfaces only the most convincing arguments. Our system also produces numeric decision scores that allow investors to rank and thus efficiently prioritize opportunities. We evaluate DIALECTIC through backtesting on real investment opportunities aggregated from five VC funds, showing that DIALECTIC matches the precision of human VCs in predicting startup success.
>
---
#### [new 058] When LLM Judge Scores Look Good but Best-of-N Decisions Fail
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大模型评分在最佳选择任务中的有效性问题。指出全局评分指标无法准确反映局部排序效果，提出应关注提示内相关性与并列率。**

- **链接: [https://arxiv.org/pdf/2603.12520](https://arxiv.org/pdf/2603.12520)**

> **作者:** Eddie Landesberg
>
> **摘要:** Large language models are often used as judges to score candidate responses, then validated with a single global metric such as correlation with reference labels. This can be misleading when the real deployment task is best-of-n selection within a prompt. In a 5,000-prompt best-of-4 benchmark from Chatbot Arena, a judge with moderate global correlation (r = 0.47) captures only 21.0% of the improvement that perfect selection would achieve over random choice. The gap arises because global agreement is driven largely by prompt-level baseline effects, while selection depends on within-prompt ranking: within-prompt correlation is only r_within = 0.27, and coarse pointwise scoring creates ties in 67% of pairwise comparisons. In a matched-pair best-of-2 audit, explicit pairwise judging recovers much of this lost signal, raising recovery from 21.1% to 61.2%. For judge-based selection, the relevant audit should report within-prompt signal, tie rates, and recovery/top-1 accuracy, not global agreement alone.
>
---
#### [new 059] Context-Enriched Natural Language Descriptions of Vessel Trajectories
- **分类: cs.AI; cs.CL; cs.DB**

- **简介: 该论文属于自然语言生成任务，旨在将船舶轨迹数据转化为语义丰富的描述。通过构建上下文感知的抽象框架，提升轨迹表示的语义密度，支持基于LLM的可控文本生成。**

- **链接: [https://arxiv.org/pdf/2603.12287](https://arxiv.org/pdf/2603.12287)**

> **作者:** Kostas Patroumpas; Alexandros Troupiotis-Kapeliaris; Giannis Spiliopoulos; Panagiotis Betchavas; Dimitrios Skoutas; Dimitris Zissis; Nikos Bikakis
>
> **摘要:** We address the problem of transforming raw vessel trajectory data collected from AIS into structured and semantically enriched representations interpretable by humans and directly usable by machine reasoning systems. We propose a context-aware trajectory abstraction framework that segments noisy AIS sequences into distinct trips each consisting of clean, mobility-annotated episodes. Each episode is further enriched with multi-source contextual information, such as nearby geographic entities, offshore navigation features, and weather conditions. Crucially, such representations can support generation of controlled natural language descriptions using LLMs. We empirically examine the quality of such descriptions generated using several LLMs over AIS data along with open contextual features. By increasing semantic density and reducing spatiotemporal complexity, this abstraction can facilitate downstream analytics and enable integration with LLMs for higher-level maritime reasoning tasks.
>
---
## 更新

#### [replaced 001] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models
- **分类: cs.CL**

- **简介: 该论文提出SPELL框架，用于提升长文本推理能力。针对LLM在长文本处理上的不足，通过自博弈强化学习实现无监督优化，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2509.23863](https://arxiv.org/pdf/2509.23863)**

> **作者:** Ziyi Yang; Weizhou Shen; Chenliang Li; Ruijun Chen; Fanqi Wan; Ming Yan; Xiaojun Quan; Fei Huang
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Progress in long-context reasoning for large language models (LLMs) has lagged behind other recent advances. This gap arises not only from the intrinsic difficulty of processing long texts, but also from the scarcity of reliable human annotations and programmatically verifiable reward signals. In this paper, we propose SPELL, a multi-role self-play reinforcement learning framework that enables scalable, label-free optimization for long-context reasoning. SPELL integrates three cyclical roles-questioner, responder, and verifier-within a single model to enable continual self-improvement. The questioner generates questions from raw documents paired with reference answers; the responder learns to solve these questions based on the documents; and the verifier evaluates semantic equivalence between the responder's output and the questioner's reference answer, producing reward signals to guide continual training. To stabilize training, we introduce an automated curriculum that gradually increases document length and a reward function that adapts question difficulty to the model's evolving capabilities. Extensive experiments on six long-context benchmarks show that SPELL consistently improves performance across diverse LLMs and outperforms equally sized models fine-tuned on large-scale annotated data. Notably, SPELL achieves an average 7.6-point gain in pass@8 on the strong reasoning model Qwen3-30B-A3B-Thinking, raising its performance ceiling and showing promise for scaling to even more capable models. Our code is available at this https URL.
>
---
#### [replaced 002] Representing data in words: A context engineering approach
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于数据表述任务，旨在解决LLMs难以准确表达数值数据的问题。提出wordalisations方法，生成自然描述性文本，并在多个领域验证其有效性。**

- **链接: [https://arxiv.org/pdf/2503.15509](https://arxiv.org/pdf/2503.15509)**

> **作者:** Amandine M. Caut; Amy Rouillard; Beimnet Zenebe; Matthias Green; Ágúst Pálmason Morthens; David J. T. Sumpter
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable potential across a broad range of applications. However, producing reliable text that faithfully represents data remains a challenge. While prior work has shown that task-specific conditioning through in-context learning and knowledge augmentation can improve performance, LLMs continue to struggle with interpreting and reasoning about numerical data. To address this, we introduce wordalisations, a methodology for generating stylistically natural narratives from data. Much like how visualisations display numerical data in a way that is easy to digest, wordalisations abstract data insights into descriptive texts. To illustrate the method's versatility, we apply it to three application areas: scouting football players, personality tests, and international survey data. Due to the absence of standardized benchmarks for this specific task, we conduct LLM-as-a-judge and human-as-a-judge evaluations to assess accuracy across the three applications. We found that wordalisation produces engaging texts that accurately represent the data. We further describe best practice methods for open and transparent development of communication about data.
>
---
#### [replaced 003] mAceReason-Math: A Dataset of High-Quality Multilingual Math Problems Ready For RLVR
- **分类: cs.CL**

- **简介: 该论文属于多语言数学问题研究任务，旨在解决现有数据集不适合RLVR的问题。工作包括构建高质量多语言数学数据集mAceReason-Math，覆盖14种语言，每种语言超10000样本。**

- **链接: [https://arxiv.org/pdf/2603.10767](https://arxiv.org/pdf/2603.10767)**

> **作者:** Konstantin Dobler; Simon Lehnerer; Federico Scozzafava; Jonathan Janke; Mohamed Ali
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has been successfully applied to significantly boost the capabilities of pretrained large language models, especially in the math and logic problem domains. However, current research and available training datasets remain English-centric. While multilingual training data and benchmarks have been created in the past, they were not created with RLVR and current model capability in mind, and their level of difficulty is often too low to provide appropriate training signals for current models. To address this gap, we provide mAceReason-Math, a dataset of high-quality translations of challenging math problems sourced from a corpus specifically curated for RLVR (AceReason-Math). We further take specific care to clean and improve our translations, resulting in a coverage of 14 languages with more than 10,000 samples per language. We release the dataset to facilitate multilingual RLVR research and benchmarking in the research community.
>
---
#### [replaced 004] Towards Interactive Intelligence for Digital Humans
- **分类: cs.CV; cs.CL; cs.GR; cs.HC**

- **简介: 该论文提出交互智能，解决数字人缺乏真实互动的问题。通过Mio框架实现多模态交互与自我进化，提升数字人的智能交互能力。**

- **链接: [https://arxiv.org/pdf/2512.13674](https://arxiv.org/pdf/2512.13674)**

> **作者:** Yiyi Cai; Xuangeng Chu; Xiwei Gao; Sitong Gong; Yifei Huang; Caixin Kang; Kunhang Li; Haiyang Liu; Ruicong Liu; Yun Liu; Dianwen Ng; Zixiong Su; Erwin Wu; Yuhan Wu; Dingkun Yan; Tianyu Yan; Chang Zeng; Bo Zheng; You Zhou
>
> **摘要:** We introduce Interactive Intelligence, a novel paradigm of digital human that is capable of personality-aligned expression, adaptive interaction, and self-evolution. To realize this, we present Mio (Multimodal Interactive Omni-Avatar), an end-to-end framework composed of five specialized modules: Thinker, Talker, Face Animator, Body Animator, and Renderer. This unified architecture integrates cognitive reasoning with real-time multimodal embodiment to enable fluid, consistent interaction. Furthermore, we establish a new benchmark to rigorously evaluate the capabilities of interactive intelligence. Extensive experiments demonstrate that our framework achieves superior performance compared to state-of-the-art methods across all evaluated dimensions. Together, these contributions move digital humans beyond superficial imitation toward intelligent interaction.
>
---
#### [replaced 005] AdaBoN: Adaptive Best-of-N Alignment
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型对齐任务，旨在解决统一分配计算资源效率低的问题。提出自适应的Best-of-N策略，根据提示难度动态分配资源，提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2505.12050](https://arxiv.org/pdf/2505.12050)**

> **作者:** Vinod Raman; Hilal Asi; Satyen Kale
>
> **备注:** 25 pages
>
> **摘要:** Recent advances in test-time alignment methods, such as Best-of-N sampling, offer a simple and effective way to steer language models (LMs) toward preferred behaviors using reward models (RM). However, these approaches can be computationally expensive, especially when applied uniformly across prompts without accounting for differences in alignment difficulty. In this work, we propose a prompt-adaptive strategy for Best-of-N alignment that allocates inference-time compute more efficiently. Motivated by latency concerns, we develop a two-stage algorithm: an initial exploratory phase estimates the reward distribution for each prompt using a small exploration budget, and a second stage adaptively allocates the remaining budget using these estimates. Our method is simple, practical, and compatible with any LM-RM combination. Empirical results on prompts from the AlpacaEval, HH-RLHF, and PKU-SafeRLHF datasets for 12 LM/RM pairs and 50 different batches of prompts show that our adaptive strategy outperforms the uniform allocation with the same inference budget. Moreover, we show that our adaptive strategy remains competitive against uniform allocations with 20 percent larger inference budgets and improves in performance as the batch size grows.
>
---
#### [replaced 006] Towards Contextual Sensitive Data Detection
- **分类: cs.CR; cs.AI; cs.CL; cs.CY; cs.DB; cs.IR**

- **简介: 该论文属于敏感数据检测任务，旨在解决数据发布前的隐私保护问题。通过引入上下文感知的敏感性框架，提升检测准确性。**

- **链接: [https://arxiv.org/pdf/2512.04120](https://arxiv.org/pdf/2512.04120)**

> **作者:** Liang Telkamp; Madelon Hulsebos
>
> **摘要:** The emergence of open data portals necessitates more attention to protecting sensitive data before datasets get published and exchanged. To do so effectively, we observe the need to refine and broaden our definitions of sensitive data, and argue that the sensitivity of data depends on its context. Following this definition, we introduce a contextual data sensitivity framework building on two core concepts: 1) type contextualization, which considers the type of the data values at hand within the overall context of the dataset or document to assess their true sensitivity, and 2) domain contextualization, which assesses the sensitivity of data values informed by domain-specific information external to the dataset, such as geographic origin of a dataset. Experiments instrumented with language models confirm that: 1) type-contextualization significantly reduces the number of false positives for type-based sensitive data detection and reaches a recall of 94% compared to 63% with commercial tools, and 2) domain-contextualization leveraging sensitivity rule retrieval effectively grounds sensitive data detection in relevant context in non-standard data domains. A case study with humanitarian data experts also illustrates that context-grounded explanations provide useful guidance in manual data auditing processes. We open-source the implementation of the mechanisms and annotated datasets at this https URL.
>
---
#### [replaced 007] Instructing Large Language Models for Low-Resource Languages: A Systematic Study for Basque
- **分类: cs.CL**

- **简介: 该论文属于低资源语言语言模型训练任务，旨在解决缺乏指令数据的问题。通过使用目标语言语料和合成指令，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2506.07597](https://arxiv.org/pdf/2506.07597)**

> **作者:** Oscar Sainz; Naiara Perez; Julen Etxaniz; Joseba Fernandez de Landa; Itziar Aldabe; Iker García-Ferrero; Aimar Zabala; Ekhi Azurmendi; German Rigau; Eneko Agirre; Mikel Artetxe; Aitor Soroa
>
> **备注:** Accepted at EMNLP 2025 Main Conference
>
> **摘要:** Instructing language models with user intent requires large instruction datasets, which are only available for a limited set of languages. In this paper, we explore alternatives to conventional instruction adaptation pipelines in low-resource scenarios. We assume a realistic scenario for low-resource languages, where only the following are available: corpora in the target language, existing open-weight multilingual base and instructed backbone LLMs, and synthetically generated instructions sampled from the instructed backbone. We present a comprehensive set of experiments for Basque that systematically study different combinations of these components evaluated on benchmarks and human preferences from 1,680 participants. Our conclusions show that target language corpora are essential, with synthetic instructions yielding robust models, and, most importantly, that using as backbone an instruction-tuned model outperforms using a base non-instructed model. Scaling up to Llama 3.1 Instruct 70B as backbone, our model comes near frontier models of much larger sizes for Basque, without using any Basque instructions. We release code, models, instruction datasets, and human preferences to support full reproducibility in future research on low-resource language adaptation. this https URL
>
---
#### [replaced 008] Why Softmax Attention Outperforms Linear Attention
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解释为何softmax注意力优于线性注意力。通过理论分析，揭示两者性能差异的原因。**

- **链接: [https://arxiv.org/pdf/2310.11685](https://arxiv.org/pdf/2310.11685)**

> **作者:** Yichuan Deng; Zhao Song; Kaijun Yuan; Tianyi Zhou
>
> **摘要:** Large transformer models have achieved state-of-the-art results in numerous natural language processing tasks. Among the pivotal components of the transformer architecture, the attention mechanism plays a crucial role in capturing token interactions within sequences through the utilization of softmax function. Conversely, linear attention presents a more computationally efficient alternative by approximating the softmax operation with linear complexity. However, it exhibits substantial performance degradation when compared to the traditional softmax attention mechanism. In this paper, we bridge the gap in our theoretical understanding of the reasons behind the practical performance gap between softmax and linear attention. By conducting a comprehensive comparative analysis of these two attention mechanisms, we shed light on the underlying reasons for why softmax attention outperforms linear attention in most scenarios.
>
---
#### [replaced 009] LatentChem: From Textual CoT to Latent Thinking in Chemical Reasoning
- **分类: physics.chem-ph; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于化学推理任务，解决传统方法依赖离散文本表示的问题。通过引入LatentChem，在连续潜在空间中进行推理，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2602.07075](https://arxiv.org/pdf/2602.07075)**

> **作者:** Xinwu Ye; Yicheng Mao; Jia Zhang; Yimeng Liu; Li Hao; Fang Wu; Zhiwei Li; Yuxuan Liao; Zehong Wang; Yingcheng Wu; Zhiyuan Liu; Zhenfei Yin; Li Yuan; Philip Torr; Huan Sun; Xiangxiang Zeng; Mengdi Wang; Le Cong; Shenghua Gao; Xiangru Tang
>
> **摘要:** Chemical large language models (LLMs) predominantly rely on explicit Chain-of-Thought (CoT) in natural language to perform complex reasoning. However, chemical reasoning is inherently continuous and structural, and forcing it into discrete linguistic tokens introduces a fundamental representation mismatch that constrains both efficiency and performance. We introduce LatentChem, a latent reasoning interface that decouples chemical computation from textual generation, enabling models to perform multi-step reasoning directly in continuous latent space while emitting language only for final outputs. Remarkably, we observe a consistent emergent behavior: when optimized solely for task success, models spontaneously internalize reasoning, progressively abandoning verbose textual derivations in favor of implicit latent computation. This shift is not merely stylistic but computationally advantageous. Across diverse chemical reasoning benchmarks, LatentChem achieves a 59.88\% non-tie win rate over strong CoT-based baselines on ChemCoTBench, while delivering a 10.84$\times$ average reduction in reasoning overhead. Our results provide empirical evidence that chemical reasoning is more naturally and effectively realized as continuous latent dynamics rather than discretized linguistic trajectories.
>
---
#### [replaced 010] LLMs Can Infer Political Alignment from Online Conversations
- **分类: cs.SI; cs.CL; cs.CY**

- **简介: 该论文属于隐私风险分析任务，旨在研究LLMs能否从在线对话中推断用户政治立场。通过分析Reddit和DebateOrg数据，发现LLMs能有效识别政治倾向，揭示其潜在的隐私威胁。**

- **链接: [https://arxiv.org/pdf/2603.11253](https://arxiv.org/pdf/2603.11253)**

> **作者:** Byunghwee Lee; Sangyeon Kim; Filippo Menczer; Yong-Yeol Ahn; Haewoon Kwak; Jisun An
>
> **备注:** 56 pages; 4 figures in the main text and 18 supplementary figures, 11 supplementary tables
>
> **摘要:** Due to the correlational structure in our traits such as identities, cultures, and political attitudes, seemingly innocuous preferences like following a band or using a specific slang can reveal private traits. This possibility, especially when combined with massive, public social data and advanced computational methods, poses a fundamental privacy risk. As our data exposure online and the rapid advancement of AI are increasing the risk of misuse, it is critical to understand the capacity of large language models (LLMs) to exploit such potential. Here, using online discussions on DebateOrg and Reddit, we show that LLMs can reliably infer hidden political alignment, significantly outperforming traditional machine learning models. Prediction accuracy further improves as we aggregate multiple text-level inferences into a user-level prediction, and as we use more politics-adjacent domains. We demonstrate that LLMs leverage words that are highly predictive of political alignment while not being explicitly political. Our findings underscore the capacity and risks of LLMs for exploiting socio-cultural correlates.
>
---
#### [replaced 011] A Decision-Theoretic Formalisation of Steganography With Applications to LLM Monitoring
- **分类: cs.AI; cs.CL; cs.CR; cs.IT; cs.MA**

- **简介: 该论文属于LLM安全任务，旨在解决检测和量化LLM中隐写术行为的问题。提出决策理论视角和隐写差距概念，以评估隐藏信息的可用性差异。**

- **链接: [https://arxiv.org/pdf/2602.23163](https://arxiv.org/pdf/2602.23163)**

> **作者:** Usman Anwar; Julianna Piskorz; David D. Baek; David Africa; Jim Weatherall; Max Tegmark; Christian Schroeder de Witt; Mihaela van der Schaar; David Krueger
>
> **备注:** First two authors contributed equally
>
> **摘要:** Large language models are beginning to show steganographic capabilities. Such capabilities could allow misaligned models to evade oversight mechanisms. Yet principled methods to detect and quantify such behaviours are lacking. Classical definitions of steganography, and detection methods based on them, require a known reference distribution of non-steganographic signals. For the case of steganographic reasoning in LLMs, knowing such a reference distribution is not feasible; this renders these approaches inapplicable. We propose an alternative, \textbf{decision-theoretic view of steganography}. Our central insight is that steganography creates an asymmetry in usable information between agents who can and cannot decode the hidden content (present within a steganographic signal), and this otherwise latent asymmetry can be inferred from the agents' observable actions. To formalise this perspective, we introduce generalised $\mathcal{V}$-information: a utilitarian framework for measuring the amount of usable information within some input. We use this to define the \textbf{steganographic gap} -- a measure that quantifies steganography by comparing the downstream utility of the steganographic signal to agents that can and cannot decode the hidden content. We empirically validate our formalism, and show that it can be used to detect, quantify, and mitigate steganographic reasoning in LLMs.
>
---
#### [replaced 012] Pyramid MoA: A Probabilistic Framework for Cost-Optimized Anytime Inference
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Pyramid MoA框架，解决大模型推理成本与能力的平衡问题，通过动态调度实现高效推理。**

- **链接: [https://arxiv.org/pdf/2602.19509](https://arxiv.org/pdf/2602.19509)**

> **作者:** Arindam Khaled
>
> **备注:** 11 pages, 6 figures, 3 tables. v2: updated model ensemble, expanded benchmarks, added zero-shot transfer experiments
>
> **摘要:** Large Language Models (LLMs) face a persistent trade-off between inference cost and reasoning capability. While "Oracle" models (e.g., Llama-3.3-70B) achieve state-of-the-art accuracy, they are prohibitively expensive for high-volume deployment. Smaller models (e.g., 7-9B parameters) are cost-effective but struggle with complex tasks. We observe that the emerging practice of LLM cascading and routing implicitly solves an anytime computation problem -- a class of algorithms, well-studied in classical AI, that produce valid solutions immediately and improve them as additional computation is allocated. In this work, we formalize this connection and propose "Pyramid MoA", a hierarchical Mixture-of-Agents architecture governed by a decision-theoretic router that dynamically escalates queries only when necessary. We establish a Probabilistic Anytime Property, proving that expected solution quality is monotonically non-decreasing with computational depth under identifiable conditions on router precision. We derive a generalized escalation rule from Value of Computation theory that accounts for imperfect oracles, extending the classical monitoring framework of Hansen and Zilberstein to stochastic LLM inference. On the MBPP code generation benchmark, the Consensus Router intercepts 81.6% of bugs. On the GSM8K/MMLU mathematical reasoning benchmark, the system matches the Oracle baseline of 68.1% accuracy while enabling up to 18.4% compute savings at a balanced operating point. Crucially, the router transfers zero-shot to unseen benchmarks: on HumanEval it achieves 81.1% accuracy (matching the Oracle) with 62.7% cost savings in economy mode, and on the highly complex MATH 500 benchmark it preserves the 58.0% Oracle ceiling. The framework acts dynamically: serving as an aggressive cost-cutter for low-entropy tasks and a strict safety net for high-entropy tasks.
>
---
#### [replaced 013] When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型集成任务，解决长文本生成中集成位置选择问题。提出SAFE框架，通过考虑分词差异和概率共识，提升集成效果与效率。**

- **链接: [https://arxiv.org/pdf/2510.15346](https://arxiv.org/pdf/2510.15346)**

> **作者:** Heecheol Yun; Kwangmin Ki; Junghyun Lee; Eunho Yang
>
> **备注:** ICLR 2026
>
> **摘要:** Ensembling Large Language Models (LLMs) has gained attention as a promising approach to surpass the performance of individual models by leveraging their complementary strengths. In particular, aggregating models' next-token probability distributions to select the next token has been shown to be effective in various tasks. However, while successful for short-form answers, its application to long-form generation remains underexplored. In this paper, we show that using existing ensemble methods in long-form generation requires a careful choice of ensembling positions, since the standard practice of ensembling at every token often degrades performance. We identify two key factors for determining the ensembling positions: tokenization mismatch across models and consensus in their next-token probability distributions. Based on this, we propose SAFE, (Stable And Fast LLM Ensembling), a framework that selectively ensembles by jointly considering these factors. To further improve stability, we apply a probability sharpening strategy when the ensemble distribution becomes overly smooth, enabling the selection of more confident tokens during ensembling. Our experiments on diverse benchmarks, including MATH500 and BBH, demonstrate that SAFE outperforms existing methods in both accuracy and efficiency, with gains achieved even when ensembling fewer than 1% of tokens.
>
---
#### [replaced 014] Multilingual, Multimodal Pipeline for Creating Authentic and Structured Fact-Checked Claim Dataset
- **分类: cs.CL**

- **简介: 该论文属于事实核查任务，旨在解决现有数据集缺乏多模态证据和结构化标注的问题。通过构建多语言、多模态的数据处理流程，生成更准确、可解释的核查数据集。**

- **链接: [https://arxiv.org/pdf/2601.07985](https://arxiv.org/pdf/2601.07985)**

> **作者:** Z. Melce Hüsünbeyi; Virginie Mouilleron; Leonie Uhling; Daniel Foppe; Tatjana Scheffler; Djamé Seddah
>
> **摘要:** The rapid proliferation of misinformation across online platforms underscores the urgent need for robust, up-to-date, explainable, and multilingual fact-checking resources. However, existing datasets are limited in scope, often lacking multimodal evidence, structured annotations, and detailed links between claims, evidence, and verdicts. This paper introduces a comprehensive data collection and processing pipeline that constructs multimodal fact-checking datasets in French and German languages by aggregating ClaimReview feeds, scraping full debunking articles, normalizing heterogeneous claim verdicts, and enriching them with structured metadata and aligned visual content. We used state-of-the-art large language models (LLMs) and multimodal LLMs for (i) evidence extraction under predefined evidence categories and (ii) justification generation that links evidence to verdicts. Evaluation with G-Eval and human assessment demonstrates that our pipeline enables fine-grained comparison of fact-checking practices across different organizations or media markets, facilitates the development of more interpretable and evidence-grounded fact-checking models, and lays the groundwork for future research on multilingual, multimodal misinformation verification.
>
---
#### [replaced 015] Re2: A Consistency-ensured Dataset for Full-stage Peer Review and Multi-turn Rebuttal Discussions
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Re²数据集，解决peer review数据质量与多样性不足的问题，支持多轮反驳讨论，提升审稿效率与作者自评能力。**

- **链接: [https://arxiv.org/pdf/2505.07920](https://arxiv.org/pdf/2505.07920)**

> **作者:** Daoze Zhang; Zhijian Bao; Sihang Du; Zhiyi Zhao; Kuangling Zhang; Dezheng Bao; Yang Yang
>
> **备注:** 2 figures, 5 tables
>
> **摘要:** Peer review is a critical component of scientific progress in the fields like AI, but the rapid increase in submission volume has strained the reviewing system, which inevitably leads to reviewer shortages and declines review quality. Besides the growing research popularity, another key factor in this overload is the repeated resubmission of substandard manuscripts, largely due to the lack of effective tools for authors to self-evaluate their work before submission. Large Language Models (LLMs) show great promise in assisting both authors and reviewers, and their performance is fundamentally limited by the quality of the peer review data. However, existing peer review datasets face three major limitations: (1) limited data diversity, (2) inconsistent and low-quality data due to the use of revised rather than initial submissions, and (3) insufficient support for tasks involving rebuttal and reviewer-author interactions. To address these challenges, we introduce the largest consistency-ensured peer review and rebuttal dataset named Re^2, which comprises 19,926 initial submissions, 70,668 review comments, and 53,818 rebuttals from 24 conferences and 21 workshops on OpenReview. Moreover, the rebuttal and discussion stage is framed as a multi-turn conversation paradigm to support both traditional static review tasks and dynamic interactive LLM assistants, providing more practical guidance for authors to refine their manuscripts and helping alleviate the growing review burden. Our data and code are available in this https URL.
>
---
#### [replaced 016] Large language models show fragile cognitive reasoning about human emotions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感计算领域，旨在探究大语言模型是否能通过认知维度而非标签理解情感。研究构建了CoRE基准，评估模型在情感推理上的表现，发现其存在与人类判断不一致和上下文不稳定的问题。**

- **链接: [https://arxiv.org/pdf/2508.05880](https://arxiv.org/pdf/2508.05880)**

> **作者:** Sree Bhattacharyya; Evgenii Kuriabov; Lucas Craig; Tharun Dilliraj; Reginald B. Adams Jr.; Jia Li; James Z. Wang
>
> **备注:** Under Review, a version was presented at WiML Workshop @ NeurIPS 2025
>
> **摘要:** Affective computing seeks to support the holistic development of artificial intelligence by enabling machines to engage with human emotion. Recent foundation models, particularly large language models (LLMs), have been trained and evaluated on emotion-related tasks, typically using supervised learning with discrete emotion labels. Such evaluations largely focus on surface phenomena, such as recognizing expressed or evoked emotions, leaving open whether these systems reason about emotion in cognitively meaningful ways. Here we ask whether LLMs can reason about emotions through underlying cognitive dimensions rather than labels alone. Drawing on cognitive appraisal theory, we introduce CoRE, a large-scale benchmark designed to probe the implicit cognitive structures LLMs use when interpreting emotionally charged situations. We assess alignment with human appraisal patterns, internal consistency, cross-model generalization, and robustness to contextual variation. We find that LLMs capture systematic relations between cognitive appraisals and emotions but show misalignment with human judgments and instability across contexts.
>
---
#### [replaced 017] Expert Selections In MoE Models Reveal (Almost) As Much As Text
- **分类: cs.CL; cs.CR**

- **简介: 该论文研究MoE模型中专家选择的隐私泄露问题，通过分析路由信息恢复文本内容，属于安全与隐私任务。工作包括提出文本重建攻击方法，验证其有效性，并探讨防御措施。**

- **链接: [https://arxiv.org/pdf/2602.04105](https://arxiv.org/pdf/2602.04105)**

> **作者:** Amir Nuriyev; Gabriel Kulp
>
> **摘要:** We present a text-reconstruction attack on mixture-of-experts (MoE) language models that recovers tokens from expert selections alone. In MoE models, each token is routed to a subset of expert subnetworks; we show these routing decisions leak substantially more information than previously understood. Prior work using logistic regression achieves limited reconstruction; we show that a 3-layer MLP improves this to 63.1% top-1 accuracy, and that a transformer-based sequence decoder recovers 91.2% of tokens top-1 (94.8% top-10) on 32-token sequences from OpenWebText after training on 100M tokens. These results connect MoE routing to the broader literature on embedding inversion. We outline practical leakage scenarios (e.g., distributed inference and side channels) and show that adding noise reduces but does not eliminate reconstruction. Our findings suggest that expert selections in MoE deployments should be treated as sensitive as the underlying text.
>
---
#### [replaced 018] IROSA: Interactive Robot Skill Adaptation using Natural Language
- **分类: cs.RO; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于机器人技能适应任务，旨在通过自然语言实现机器人技能的灵活调整。工作包括提出一个框架，利用预训练语言模型选择工具，无需微调即可完成轨迹修正、避障等操作，提升安全性与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.03897](https://arxiv.org/pdf/2603.03897)**

> **作者:** Markus Knauer; Samuel Bustamante; Thomas Eiband; Alin Albu-Schäffer; Freek Stulp; João Silvério
>
> **备注:** Accepted IEEE Robotics and Automation Letters (RA-L) journal, 8 pages, 5 figures, 3 tables, 1 listing
>
> **摘要:** Foundation models have demonstrated impressive capabilities across diverse domains, while imitation learning provides principled methods for robot skill adaptation from limited data. Combining these approaches holds significant promise for direct application to robotics, yet this combination has received limited attention, particularly for industrial deployment. We present a novel framework that enables open-vocabulary skill adaptation through a tool-based architecture, maintaining a protective abstraction layer between the language model and robot hardware. Our approach leverages pre-trained LLMs to select and parameterize specific tools for adapting robot skills without requiring fine-tuning or direct model-to-robot interaction. We demonstrate the framework on a 7-DoF torque-controlled robot performing an industrial bearing ring insertion task, showing successful skill adaptation through natural language commands for speed adjustment, trajectory correction, and obstacle avoidance while maintaining safety, transparency, and interpretability.
>
---
#### [replaced 019] A survey of diversity quantification in natural language processing: The why, what, where and how
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的多样性研究任务，旨在解决多样性概念不统一的问题。通过分析300余篇论文，构建了NLP多样性框架，明确其重要性、测量对象、位置及方法。**

- **链接: [https://arxiv.org/pdf/2507.20858](https://arxiv.org/pdf/2507.20858)**

> **作者:** Louis Estève; Marie-Catherine de Marneffe; Nurit Melnik; Agata Savary; Olha Kanishcheva
>
> **摘要:** The concept of diversity has received increasing attention in natural language processing (NLP) in recent years. It became an advocated property of datasets and systems, and many measures are used to quantify it. However, it is often addressed in an ad hoc manner, with few explicit justifications of its endorsement and many cross-paper inconsistencies. There have been very few attempts to take a step back and understand the conceptualization of diversity in NLP. To address this fragmentation, we take inspiration from other scientific fields where the concept of diversity has been more thoroughly conceptualized. We build upon Stirling (2007), a unified framework adapted from ecology and economics, which distinguishes three dimensions of diversity: variety, balance, and disparity. We survey over 300 recent diversity-related papers from ACL Anthology and build an NLP-specific framework with 4 perspectives: why diversity is important, what diversity is measured on, where it is measured, and how. Our analysis increases comparability of approaches to diversity in NLP, reveals emerging trends and allows us to formulate recommendations for the field.
>
---
#### [replaced 020] Rethinking the Relationship between the Power Law and Hierarchical Structures
- **分类: cs.CL**

- **简介: 该论文属于语言结构分析任务，旨在验证幂律与层级结构之间的关系。通过分析句法树的统计特性，发现原有假设不成立，需重新思考二者关系。**

- **链接: [https://arxiv.org/pdf/2505.04984](https://arxiv.org/pdf/2505.04984)**

> **作者:** Kai Nakaishi; Ryo Yoshida; Kohei Kajikawa; Koji Hukushima; Yohei Oseki
>
> **备注:** Accepted for publication in Transactions of the Association for Computational Linguistics (TACL). This is a pre-MIT Press publication version
>
> **摘要:** Statistical analysis of corpora provides an approach to quantitatively investigate natural languages. This approach has revealed that several power laws consistently emerge across different corpora and languages, suggesting universal mechanisms underlying languages. In particular, the power-law decay of correlations has been interpreted as evidence of underlying hierarchical structures in syntax, semantics, and discourse. This perspective has also been extended beyond corpora produced by human adults, including child speech, birdsong, and chimpanzee action sequences. However, the argument supporting this interpretation has not been empirically tested in natural languages. To address this gap, the present study examines the validity of the argument for syntactic structures. Specifically, we test whether the statistical properties of parse trees align with the assumptions in the argument. Using English and Japanese corpora, we analyze the mutual information, deviations from probabilistic context-free grammars (PCFGs), and other properties in natural language parse trees, as well as in the PCFG that approximates these parse trees. Our results indicate that the assumptions do not hold for syntactic structures and that it is difficult to apply the proposed argument not only to sentences by human adults but also to other domains, highlighting the need to reconsider the relationship between the power law and hierarchical structures.
>
---
#### [replaced 021] Building Benchmarks from the Ground Up: Community-Centered Evaluation of LLMs in Healthcare Chatbot Settings
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的评估任务，旨在解决LLMs在医疗聊天机器人中缺乏社区实际需求考量的问题。通过构建社区驱动的评估框架，提升模型评价的实用性与包容性。**

- **链接: [https://arxiv.org/pdf/2509.24506](https://arxiv.org/pdf/2509.24506)**

> **作者:** Hamna Hamna; Gayatri Bhat; Sourabrata Mukherjee; Faisal Lalani; Evan Hadfield; Divya Siddarth; Kalika Bali; Sunayana Sitaram
>
> **备注:** Accepted at ACM CHI 2026
>
> **摘要:** Large Language Models (LLMs) are typically evaluated through general or domain-specific benchmarks testing capabilities that often lack grounding in the lived realities of end users. Critical domains such as healthcare require evaluations that extend beyond artificial or simulated tasks to reflect the everyday needs, cultural practices, and nuanced contexts of communities. We propose Samiksha, a community-driven evaluation pipeline co-created with civil-society organizations (CSOs) and community members. Our approach enables scalable, automated benchmarking through a culturally aware, community-driven pipeline in which community feedback informs what to evaluate, how the benchmark is built, and how outputs are scored. We demonstrate this approach in the health domain in India. Our analysis highlights how current multilingual LLMs address nuanced community health queries, while also offering a scalable pathway for contextually grounded and inclusive LLM evaluation.
>
---
#### [replaced 022] From Formal Language Theory to Statistical Learning: Finite Observability of Subregular Languages
- **分类: cs.CL; cs.FL; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决子正则语言的可观察性问题。通过证明其线性可分性，确保了使用简单线性模型的学习可行性，并通过实验验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2509.22598](https://arxiv.org/pdf/2509.22598)**

> **作者:** Katsuhiko Hayashi; Hidetaka Kamigaito
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** We prove that all standard subregular language classes are linearly separable when represented by their deciding predicates. This establishes finite observability and guarantees learnability with simple linear models. Synthetic experiments confirm perfect separability under noise-free conditions, while real-data experiments on English morphology show that learned features align with well-known linguistic constraints. These results demonstrate that the subregular hierarchy provides a rigorous and interpretable foundation for modeling natural language structure. Our code used in real-data experiments is available at this https URL.
>
---
#### [replaced 023] Test-Time Adaptation via Many-Shot Prompting: Benefits, Limits, and Pitfalls
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理领域，研究测试时适应问题。通过实验分析多示例提示的有效性与局限性，探讨其在不同任务中的表现及优化策略。**

- **链接: [https://arxiv.org/pdf/2603.05829](https://arxiv.org/pdf/2603.05829)**

> **作者:** Shubhangi Upasani; Chen Wu; Jay Rainton; Bo Li; Urmish Thakker; Changran Hu; Qizheng Zhang
>
> **摘要:** Test-time adaptation enables large language models (LLMs) to modify their behavior at inference without updating model parameters. A common approach is many-shot prompting, where large numbers of in-context learning (ICL) examples are injected as an input-space test-time update. Although performance can improve as more demonstrations are added, the reliability and limits of this update mechanism remain poorly understood, particularly for open-source models. We present an empirical study of many-shot prompting across tasks and model backbones, analyzing how performance varies with update magnitude, example ordering, and selection policy. We further study Dynamic and Reinforced ICL as alternative test-time update strategies that control which information is injected and how it constrains model behavior. We find that many-shot prompting is effective for structured tasks where demonstrations provide high information gain, but is highly sensitive to selection strategy and often shows limited benefits for open-ended generation tasks. Overall, we characterize the practical limits of prompt-based test-time adaptation and outline when input-space updates are beneficial versus harmful.
>
---
#### [replaced 024] Triple X: A LLM-Based Multilingual Speech Recognition System for the INTERSPEECH2025 MLC-SLM Challenge
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于多语言语音识别任务，旨在提升多语言对话场景下的识别准确率。通过创新的编码器-适配器-大语言模型架构和多阶段训练策略，实现了优异的识别性能。**

- **链接: [https://arxiv.org/pdf/2507.17288](https://arxiv.org/pdf/2507.17288)**

> **作者:** Miaomiao Gao; Xiaoxiao Xiang; Yiwen Guo
>
> **备注:** Accepted By Interspeech 2025 MLC-SLM workshop
>
> **摘要:** This paper describes our Triple X speech recognition system submitted to Task 1 of the Multi-Lingual Conversational Speech Language Modeling (MLC-SLM) Challenge. Our work focuses on optimizing speech recognition accuracy in multilingual conversational scenarios through an innovative encoder-adapter-LLM architecture. This framework harnesses the powerful reasoning capabilities of text-based large language models while incorporating domain-specific adaptations. To further enhance multilingual recognition performance, we adopted a meticulously designed multi-stage training strategy leveraging extensive multilingual audio datasets. Experimental results demonstrate that our approach achieves competitive Word Error Rate (WER) performance on both dev and test sets, obtaining second place in the challenge ranking.
>
---
#### [replaced 025] Partially Recentralization Softmax Loss for Vision-Language Models Robustness
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态自然语言处理任务，旨在提升视觉-语言模型的对抗鲁棒性。通过修改损失函数，限制Top K Softmax输出，实验表明可有效增强模型抗攻击能力。**

- **链接: [https://arxiv.org/pdf/2402.03627](https://arxiv.org/pdf/2402.03627)**

> **作者:** Hao Wang; Jinzhe Jiang; Xin Zhang; Chen Li
>
> **备注:** The study described in Section 4 was conducted without required institutional review board approval. The paper is withdrawn pending completion of the approval process
>
> **摘要:** As Large Language Models make a breakthrough in natural language processing tasks (NLP), multimodal technique becomes extremely popular. However, it has been shown that multimodal NLP are vulnerable to adversarial attacks, where the outputs of a model can be dramatically changed by a perturbation to the input. While several defense techniques have been proposed both in computer vision and NLP models, the multimodal robustness of models have not been fully explored. In this paper, we study the adversarial robustness provided by modifying loss function of pre-trained multimodal models, by restricting top K softmax outputs. Based on the evaluation and scoring, our experiments show that after a fine-tuning, adversarial robustness of pre-trained models can be significantly improved, against popular attacks. Further research should be studying, such as output diversity, generalization and the robustness-performance trade-off of this kind of loss functions. Our code will be available after this paper is accepted
>
---
#### [replaced 026] Towards AI Search Paradigm
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出AI搜索范式，解决传统搜索系统适应性不足问题，通过四个LLM代理协作，实现复杂任务处理与信息整合。**

- **链接: [https://arxiv.org/pdf/2506.17188](https://arxiv.org/pdf/2506.17188)**

> **作者:** Yuchen Li; Hengyi Cai; Rui Kong; Xinran Chen; Jiamin Chen; Jun Yang; Haojie Zhang; Jiayi Li; Jiayi Wu; Yiqun Chen; Changle Qu; Wenwen Ye; Lixin Su; Xinyu Ma; Lingyong Yan; Long Xia; Daiting Shi; Junfeng Wang; Xiangyu Zhao; Jiashu Zhao; Haoyi Xiong; Shuaiqiang Wang; Dawei Yin
>
> **摘要:** In this paper, we introduce the AI Search Paradigm, a comprehensive blueprint for next-generation search systems capable of emulating human information processing and decision-making. The paradigm employs a modular architecture of four LLM-powered agents (Master, Planner, Executor and Writer) that dynamically adapt to the full spectrum of information needs, from simple factual queries to complex multi-stage reasoning tasks. These agents collaborate dynamically through coordinated workflows to evaluate query complexity, decompose problems into executable plans, and orchestrate tool usage, task execution, and content synthesis. We systematically present key methodologies for realizing this paradigm, including task planning and tool integration, execution strategies, aligned and robust retrieval-augmented generation, and efficient LLM inference, spanning both algorithmic techniques and infrastructure-level optimizations. By providing an in-depth guide to these foundational components, this work aims to inform the development of trustworthy, adaptive, and scalable AI search systems.
>
---
#### [replaced 027] Token Distillation: Attention-aware Input Embeddings For New Tokens
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决新词嵌入初始化问题。通过Token Distillation方法，利用原始分词表示快速学习高质量新词嵌入，无需额外训练。**

- **链接: [https://arxiv.org/pdf/2505.20133](https://arxiv.org/pdf/2505.20133)**

> **作者:** Konstantin Dobler; Desmond Elliott; Gerard de Melo
>
> **备注:** ICLR 2026 camera-ready
>
> **摘要:** Current language models rely on static vocabularies determined at pretraining time, which can lead to decreased performance and increased computational cost for domains underrepresented in the original vocabulary. New tokens can be added to solve this problem, when coupled with a good initialization for their new embeddings. However, existing embedding initialization methods require expensive further training or pretraining of additional modules. In this paper, we propose Token Distillation and show that by distilling representations obtained using the original tokenization, we can quickly learn high-quality input embeddings for new tokens. Experimental results with a wide range of open-weight models show that Token Distillation outperforms even strong baselines.
>
---
#### [replaced 028] XSkill: Continual Learning from Experience and Skills in Multimodal Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出XSkill，解决多模态智能体在开放环境中持续学习的问题，通过经验与技能双流框架提升工具使用效率和任务规划能力。**

- **链接: [https://arxiv.org/pdf/2603.12056](https://arxiv.org/pdf/2603.12056)**

> **作者:** Guanyu Jiang; Zhaochen Su; Xiaoye Qu; Yi R. Fung
>
> **摘要:** Multimodal agents can now tackle complex reasoning tasks with diverse tools, yet they still suffer from inefficient tool use and inflexible orchestration in open-ended settings. A central challenge is enabling such agents to continually improve without parameter updates by learning from past trajectories. We identify two complementary forms of reusable knowledge essential for this goal: experiences, providing concise action-level guidance for tool selection and decision making, and skills, providing structured task-level guidance for planning and tool use. To this end, we propose XSkill, a dual-stream framework for continual learning from experience and skills in multimodal agents. XSkill grounds both knowledge extraction and retrieval in visual observations. During accumulation, XSkill distills and consolidates experiences and skills from multi-path rollouts via visually grounded summarization and cross-rollout critique. During inference, it retrieves and adapts this knowledge to the current visual context and feeds usage history back into accumulation to form a continual learning loop. Evaluated on five benchmarks across diverse domains with four backbone models, XSkill consistently and substantially outperforms both tool-only and learning-based baselines. Further analysis reveals that the two knowledge streams play complementary roles in influencing the reasoning behaviors of agents and show superior zero-shot generalization.
>
---
#### [replaced 029] Mask-Enhanced Autoregressive Prediction: Pay Less Attention to Learn More
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出MEAP方法，解决大语言模型关键信息检索不足的问题。通过在自回归预测中引入掩码机制，提升模型的上下文理解能力。**

- **链接: [https://arxiv.org/pdf/2502.07490](https://arxiv.org/pdf/2502.07490)**

> **作者:** Xialie Zhuang; Zhikai Jia; Jianjin Li; Zhenyu Zhang; Li Shen; Zheng Cao; Shiwei Liu
>
> **备注:** 17 pages,7 figures
>
> **摘要:** Large Language Models (LLMs) are discovered to suffer from accurately retrieving key information. To address this, we propose Mask-Enhanced Autoregressive Prediction (MEAP), a simple yet effective training paradigm that seamlessly integrates Masked Language Modeling (MLM) into Next-Token Prediction (NTP) to enhance the latter's in-context retrieval capabilities. Specifically, MEAP first randomly masks a small fraction of input tokens and then directly performs the standard next-token prediction autoregressive using a decoder-only Transformer. MEAP eliminates the need for bidirectional attention or encoder-decoder architectures for MLM, incurring no additional computational overhead during pre-training or inference. Intensive experiments demonstrate that MEAP substantially outperforms NTP on key information retrieval and long-context reasoning tasks, while performing on par or better on commonsense reasoning tasks. The benefits of MEAP also extend to supervised fine-tuning, where it shows remarkable advantages in lost-in-the-middle scenarios, outperforming NTP by 11.77 percentage points. Our analysis indicates that MEAP's effectiveness arises from its ability to promote more distinguishable attention scores by concentrating on a reduced set of non-masked tokens. This mechanism improves the model's focus on task-relevant signals while mitigating the influence of peripheral context. These findings position MEAP as a promising training paradigm for large language models.
>
---
#### [replaced 030] Scaling Generalist Data-Analytic Agents
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于数据解析代理任务，解决开放源码模型在处理复杂数据分析时的不足，提出DataMind框架提升模型性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2509.25084](https://arxiv.org/pdf/2509.25084)**

> **作者:** Shuofei Qiao; Yanqiu Zhao; Zhisong Qiu; Xiaobin Wang; Jintian Zhang; Zhao Bin; Ningyu Zhang; Yong Jiang; Pengjun Xie; Fei Huang; Huajun Chen
>
> **备注:** ICLR 2026
>
> **摘要:** Data-analytic agents are emerging as a key catalyst for automated scientific discovery and for the vision of Innovating AI. Current approaches, however, rely heavily on prompt engineering over proprietary models, while open-source models struggle to face diverse-format, large-scale data files and long-horizon, multi-step reasoning that real-world analytics demands. This paper introduces DataMind, a scalable data synthesis and agent training recipe designed to build generalist data-analytic agents. DataMind tackles three key challenges in building open-source data-analytic agents, including insufficient data resources, improper training strategy, and unstable code-based multi-turn rollout. Concretely, DataMind applies 1) a fine-grained task taxonomy and a recursive easy-to-hard task composition mechanism to increase the diversity and difficulty of synthesized queries; 2) a knowledge-augmented trajectory sampling strategy followed by model-based and rule-based filtering; 3) a dynamically adjustable training objective combining both SFT and RL losses; 4) a memory-frugal and stable code-based multi-turn rollout framework. Built on DataMind, we curate DataMind-12K, a high-quality trajectory set spanning diverse domains, task categories, and data file formats for data-analytic tasks. Trained on DataMind-12K, our DataMind-14B achieves state-of-the-art with an average score of 71.16% on multiple data analysis benchmarks, outperforming the strongest proprietary baselines DeepSeek-V3.1 and GPT-5. Our DataMind-7B also performs best among all open-source models with a score of 68.10%. We also incorporate some empirical insights gained from our exploratory trials into the analysis experiments, aiming to provide actionable insights about agentic training for the community. We will release DataMind-12K and DataMind-7B,14B for the community's future research.
>
---
#### [replaced 031] LESS: Large Language Model Enhanced Semi-Supervised Learning for Speech Foundational Models Using in-the-wild Data
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音基础模型的半监督学习任务，旨在解决真实场景数据中伪标签质量低的问题。通过引入大语言模型优化伪标签，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2506.04586](https://arxiv.org/pdf/2506.04586)**

> **作者:** Wen Ding; Fan Qian
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Although state-of-the-art Speech Foundational Models can produce high-quality text pseudo-labels, applying Semi-Supervised Learning (SSL) for in-the-wild real-world data remains challenging due to its richer and more complex acoustics compared to curated datasets. To address the challenges, we introduce LESS (Large Language Model Enhanced Semi-supervised Learning), a versatile framework that uses Large Language Models (LLMs) to correct pseudo-labels generated on in-the-wild data. In the LESS framework, pseudo-labeled text from Automatic Speech Recognition (ASR) or Automatic Speech Translation (AST) of the unsupervised data is refined by an LLM, and further improved by a data filtering strategy. Across Mandarin ASR and Spanish-to-English AST evaluations, LESS delivers consistent gains, with an absolute Word Error Rate reduction of 3.8% on WenetSpeech, and BLEU score increase of 0.8 and 0.7, achieving 34.0 on Callhome and 64.7 on Fisher testsets respectively. These results highlight LESS's effectiveness across diverse languages, tasks, and domains. We have released the recipe as open source to facilitate further research in this area.
>
---
#### [replaced 032] A Longitudinal, Multinational, and Multilingual Corpus of News Coverage of the Russo-Ukrainian War
- **分类: cs.CL; cs.SI**

- **简介: 该论文构建了DNIPRO语料库，用于分析俄乌战争中不同国家媒体的叙事差异。属于信息分析任务，解决媒体框架与叙事分歧问题，通过多语言、多国别新闻分析实现。**

- **链接: [https://arxiv.org/pdf/2601.16309](https://arxiv.org/pdf/2601.16309)**

> **作者:** Dikshya Mohanty; Taisiia Sabadyn; Jelwin Rodrigues; Chenlu Wang; Abhishek Kalugade; Ritwik Banerjee
>
> **备注:** To appear in Language Resources and Evaluation Conference (LREC) 2026
>
> **摘要:** We present DNIPRO, a corpus of 246K news articles from the Russo-Ukrainian war (Feb 2022 -- Aug 2024) spanning eleven outlets across five nation-states (Russia, Ukraine, U.S., U.K., China) and three languages. The corpus features comprehensive metadata and human-evaluated annotations for stance, sentiment, and topical framing, enabling systematic analysis of competing geopolitical narratives. It is uniquely suited for empirical studies of narrative divergence, media framing, and information warfare. Our exploratory analyses reveal how media outlets construct incompatible realities through divergent attribution and topical selection without direct refutation of opposing narratives. DNIPRO empowers empirical research on narrative evolution, cross-lingual information flow, and computational detection of implicit contradictions in fragmented information ecosystems.
>
---
#### [replaced 033] Superficial Safety Alignment Hypothesis
- **分类: cs.CL; cs.AI; cs.CR; cs.CY; cs.LG**

- **简介: 该论文属于安全对齐任务，旨在解决大语言模型生成不安全响应的问题。提出SSAH假设，识别关键组件以简化安全对齐过程。**

- **链接: [https://arxiv.org/pdf/2410.10862](https://arxiv.org/pdf/2410.10862)**

> **作者:** Jianwei Li; Jung-Eun Kim
>
> **备注:** ICLR 2026
>
> **摘要:** As large language models (LLMs) are overwhelmingly more and more integrated into various applications, ensuring they generate safe responses is a pressing need. Previous studies on alignment have largely focused on general instruction-following but have often overlooked the distinct properties of safety alignment, such as the brittleness of safety mechanisms. To bridge the gap, we propose the Superficial Safety Alignment Hypothesis (SSAH), which posits that safety alignment teaches an otherwise unsafe model to choose the correct reasoning direction-fulfill or refuse users' requests-interpreted as an implicit binary classification task. Through SSAH, we hypothesize that only a few essential components can establish safety guardrails in LLMs. We successfully identify four types of attribute-critical components: Safety Critical Unit (SCU), Utility Critical Unit (UCU), Complex Unit (CU), and Redundant Unit (RU). Our findings show that freezing certain safety-critical components during fine-tuning allows the model to retain its safety attributes while adapting to new tasks. Similarly, we show that leveraging redundant units in the pre-trained model as an "alignment budget" can effectively minimize the alignment tax while achieving the alignment goal. All considered, this paper concludes that the atomic functional unit for safety in LLMs is at the neuron level and underscores that safety alignment should not be complicated. We have code implementation and other information on the project website: this https URL.
>
---
#### [replaced 034] Do LLMs have a Gender (Entropy) Bias?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的公平性研究任务，旨在检测和缓解大语言模型的性别熵偏差。通过构建基准数据集并分析模型响应差异，提出一种简单的去偏策略。**

- **链接: [https://arxiv.org/pdf/2505.20343](https://arxiv.org/pdf/2505.20343)**

> **作者:** Sonal Prabhune; Balaji Padmanabhan; Kaushik Dutta
>
> **备注:** 18 pages, 4 figures
>
> **摘要:** We investigate the existence and persistence of a specific type of gender bias in some of the popular LLMs and contribute a new benchmark dataset, RealWorldQuestioning (released on HuggingFace ), developed from real-world questions across four key domains in business and health contexts: education, jobs, personal financial management, and general health. We define and study entropy bias, which we define as a discrepancy in the amount of information generated by an LLM in response to real questions users have asked. We tested this using four different LLMs and evaluated the generated responses both qualitatively and quantitatively by using ChatGPT-4o (as "LLM-as-judge"). Our analyses (metric-based comparisons and "LLM-as-judge" evaluation) suggest that there is no significant bias in LLM responses for men and women at a category level. However, at a finer granularity (the individual question level), there are substantial differences in LLM responses for men and women in the majority of cases, which "cancel" each other out often due to some responses being better for males and vice versa. This is still a concern since typical users of these tools often ask a specific question (only) as opposed to several varied ones in each of these common yet important areas of life. We suggest a simple debiasing approach that iteratively merges the responses for the two genders to produce a final result. Our approach demonstrates that a simple, prompt-based debiasing strategy can effectively debias LLM outputs, thus producing responses with higher information content than both gendered variants in 78% of the cases, and consistently achieving a balanced integration in the remaining cases.
>
---
#### [replaced 035] LLM Unlearning with LLM Beliefs
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型遗忘任务，旨在解决模型记忆敏感内容的问题。工作包括发现现有方法的副作用，提出结合模型信念的框架以更彻底地遗忘。**

- **链接: [https://arxiv.org/pdf/2510.19422](https://arxiv.org/pdf/2510.19422)**

> **作者:** Kemou Li; Qizhou Wang; Yue Wang; Fengpeng Li; Jun Liu; Bo Han; Jiantao Zhou
>
> **备注:** ICLR 2026
>
> **摘要:** Large language models trained on vast corpora inherently risk memorizing sensitive or harmful content, which may later resurface in their outputs. Prevailing unlearning methods generally rely on gradient ascent and its variants to lower the probability of specific target responses. However, we find that this strategy induces a critical side effect: probability mass is redistributed into high-likelihood regions, often corresponding to semantically related rephrasings of the targets. We refer to this as the squeezing effect, which explains why many methods yield merely spurious unlearning, a problem further obscured by automated metrics (e.g., ROUGE, truth ratio) that misreport actual success. To address this, we propose a bootstrapping (BS) framework that explicitly links the squeezing effect with the model's own high-confidence generations, namely its model beliefs. Since model beliefs inherently capture the very high-likelihood regions where probability mass is squeezed, incorporating them into the unlearning objective directly counters the squeezing effect. By jointly suppressing both target responses and model beliefs, BS-T (token) attenuates high-probability tokens, whereas BS-S (sequence) removes entire high-confidence generations, together achieving more thorough forgetting while preserving utility. Extensive experiments across diverse benchmarks with various model families confirm the effectiveness of our approach.
>
---
#### [replaced 036] Evolution and compression in LLMs: On the emergence of human-aligned categorization
- **分类: cs.CL**

- **简介: 该论文研究LLMs是否能发展出符合人类语义分类的高效系统。通过颜色命名实验和模拟文化演化，验证了LLMs在特定条件下可实现人类相似的分类效率。任务为理解LLMs的语义生成机制。**

- **链接: [https://arxiv.org/pdf/2509.08093](https://arxiv.org/pdf/2509.08093)**

> **作者:** Nathaniel Imel; Noga Zaslavsky
>
> **备注:** Published as a conference paper at ICLR 2026 (The Fourteenth International Conference on Learning Representations). OpenReview: this https URL
>
> **摘要:** Converging evidence suggests that human systems of semantic categories achieve near-optimal compression via the Information Bottleneck (IB) complexity-accuracy tradeoff. Large language models (LLMs) are not trained for this objective, which raises the question: are LLMs capable of evolving efficient human-aligned semantic systems? To address this question, we focus on color categorization -- a key testbed of cognitive theories of categorization with uniquely rich human data -- and replicate with LLMs two influential human studies. First, we conduct an English color-naming study, showing that LLMs vary widely in their complexity and English-alignment, with larger instruction-tuned models achieving better alignment and IB-efficiency. Second, to test whether these LLMs simply mimic patterns in their training data or actually exhibit a human-like inductive bias toward IB-efficiency, we simulate cultural evolution of pseudo color-naming systems in LLMs via a method we refer to as Iterated in-Context Language Learning (IICLL). We find that akin to humans, LLMs iteratively restructure initially random systems towards greater IB-efficiency. However, only a model with strongest in-context capabilities (Gemini 2.0) is able to recapitulate the wide range of near-optimal IB-tradeoffs observed in humans, while other state-of-the-art models converge to low-complexity solutions. These findings demonstrate how human-aligned semantic categories can emerge in LLMs via the same fundamental principle that underlies semantic efficiency in humans.
>
---
#### [replaced 037] One Supervisor, Many Modalities: Adaptive Tool Orchestration for Autonomous Queries
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种多模态查询处理框架，解决多模态任务协调问题。通过中央监督器动态分配任务，提升效率与经济性。**

- **链接: [https://arxiv.org/pdf/2603.11545](https://arxiv.org/pdf/2603.11545)**

> **作者:** Mayank Saini; Arit Kumar Bishwas
>
> **备注:** 19 pages, 3 figures; v2: corrected author metadata
>
> **摘要:** We present an agentic AI framework for autonomous multimodal query processing that coordinates specialized tools across text, image, audio, video, and document modalities. A central Supervisor dynamically decomposes user queries, delegates subtasks to modality-appropriate tools (e.g., object detection, OCR, speech transcription), and synthesizes results through adaptive routing strategies rather than predetermined decision trees. For text-only queries, the framework uses learned routing via RouteLLM, while non-text paths use SLM-assisted modality decomposition. Evaluated on 2,847 queries across 15 task categories, our framework achieves 72% reduction in time-to-accurate-answer, 85% reduction in conversational rework, and 67% cost reduction compared to the matched hierarchical baseline while maintaining accuracy parity. These results demonstrate that intelligent centralized orchestration fundamentally improves multimodal AI deployment economics.
>
---
#### [replaced 038] Cross-Family Speculative Prefill: Training-Free Long-Context Compression with Small Draft Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决长上下文推理中的提示压缩问题。通过跨家族草稿模型实现无需训练的提示压缩，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2603.02631](https://arxiv.org/pdf/2603.02631)**

> **作者:** Shubhangi Upasani; Ravi Shanker Raju; Bo Li; Mengmeng Ji; John Long; Chen Wu; Urmish Thakker; Guangtao Wang
>
> **摘要:** Prompt length is a major bottleneck in agentic large language model (LLM) workloads, where repeated inference steps and multi-call loops incur substantial prefill cost. Recent work on speculative prefill demonstrates that attention-based token importance estimation can enable training-free prompt compression, but this assumes the existence of a draft model that shares the same tokenizer as the target model. In practice, however, agentic pipelines frequently employ models without any smaller in-family draft model. In this work, we study cross-family speculative prefill, where a lightweight draft model from one model family is used to perform prompt compression for a target model from a different family. Using the same speculative prefill mechanism as prior work, we evaluate a range of cross-family draft-target combinations, including Qwen, LLaMA, and DeepSeek models. Across a broad diversity of tasks, we find that attention-based token importance estimation transfers reliably across different model families despite differences in model architectures and tokenizers between draft and target models. Cross-model prompt compression largely retains 90~100% of full-prompt baseline performance and, in some cases, slightly improves accuracy due to denoising effects, while delivering substantial reductions in time to first token (TTFT). These results suggest that speculative prefill depends mainly on task priors and semantic structure, thus serving as a generalizable prompt compression primitive. We discuss the implications of our findings for agentic systems, where repeated long-context inference and heterogeneous model stacks make cross-model prompt compression both necessary and practical.
>
---
#### [replaced 039] From XAI to Stories: A Factorial Study of LLM-Generated Explanation Quality
- **分类: cs.CL**

- **简介: 该论文属于可解释AI任务，研究LLM生成解释质量的影响因素。通过实验分析模型选择、XAI方法、LLM类型和提示策略对自然语言解释质量的影响。**

- **链接: [https://arxiv.org/pdf/2601.02224](https://arxiv.org/pdf/2601.02224)**

> **作者:** Fabian Lukassen; Jan Herrmann; Christoph Weisser; Benjamin Saefken; Thomas Kneib
>
> **摘要:** Explainable AI (XAI) methods like SHAP and LIME produce numerical feature attributions that remain inaccessible to non expert users. Prior work has shown that Large Language Models (LLMs) can transform these outputs into natural language explanations (NLEs), but it remains unclear which factors contribute to high-quality explanations. We present a systematic factorial study investigating how Forecasting model choice, XAI method, LLM selection, and prompting strategy affect NLE quality. Our design spans four models (XGBoost (XGB), Random Forest (RF), Multilayer Perceptron (MLP), and SARIMAX - comparing black-box Machine-Learning (ML) against classical time-series approaches), three XAI conditions (SHAP, LIME, and a no-XAI baseline), three LLMs (GPT-4o, Llama-3-8B, DeepSeek-R1), and eight prompting strategies. Using G-Eval, an LLM-as-a-judge evaluation method, with dual LLM judges and four evaluation criteria, we evaluate 660 explanations for time-series forecasting. Our results suggest that: (1) XAI provides only small improvements over no-XAI baselines, and only for expert audiences; (2) LLM choice dominates all other factors, with DeepSeek-R1 outperforming GPT-4o and Llama-3; (3) we observe an interpretability paradox: in our setting, SARIMAX yielded lower NLE quality than ML models despite higher prediction accuracy; (4) zero-shot prompting is competitive with self-consistency at 7-times lower cost; and (5) chain-of-thought hurts rather than helps.
>
---
#### [replaced 040] RECAP: Reproducing Copyrighted Data from LLMs Training with an Agentic Pipeline
- **分类: cs.CL**

- **简介: 该论文提出RECAP，用于从LLM中提取和验证训练数据，解决无法直接访问训练数据的问题。通过反馈循环和破解模块提升提取效果。**

- **链接: [https://arxiv.org/pdf/2510.25941](https://arxiv.org/pdf/2510.25941)**

> **作者:** André V. Duarte; Xuying li; Bin Zeng; Arlindo L. Oliveira; Lei Li; Zhuo Li
>
> **摘要:** If we cannot inspect the training data of a large language model (LLM), how can we ever know what it has seen? We believe the most compelling evidence arises when the model itself freely reproduces the target content. As such, we propose RECAP, an agentic pipeline designed to elicit and verify memorized training data from LLM outputs. At the heart of RECAP is a feedback-driven loop, where an initial extraction attempt is evaluated by a secondary language model, which compares the output against a reference passage and identifies discrepancies. These are then translated into minimal correction hints, which are fed back into the target model to guide subsequent generations. In addition, to address alignment-induced refusals, RECAP includes a jailbreaking module that detects and overcomes such barriers. We evaluate RECAP on EchoTrace, a new benchmark spanning over 30 full books, and the results show that RECAP leads to substantial gains over single-iteration approaches. For instance, with GPT-4.1, the average ROUGE-L score for the copyrighted text extraction improved from 0.38 to 0.47 - a nearly 24% increase.
>
---
#### [replaced 041] DeCode: Decoupling Content and Delivery for Medical QA
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗问答任务，旨在解决LLM回答与患者实际需求不匹配的问题。提出DeCode框架，无需训练即可提升问答的临床相关性。**

- **链接: [https://arxiv.org/pdf/2601.02123](https://arxiv.org/pdf/2601.02123)**

> **作者:** Po-Jen Ko; Chen-Han Tsai; Yu-Shao Peng
>
> **备注:** Preprint
>
> **摘要:** Large language models (LLMs) exhibit strong medical knowledge and can generate factually accurate responses. However, existing models often fail to account for individual patient contexts, producing answers that are clinically correct yet poorly aligned with patients' needs. In this work, we introduce DeCode (Decoupling Content and Delivery), a training-free, model-agnostic framework that adapts existing LLMs to produce contextualized answers in clinical settings. We evaluate DeCode on OpenAI HealthBench, a comprehensive and challenging benchmark designed to assess clinical relevance and validity of LLM responses. DeCode boosts zero-shot performance from 28.4% to 49.8% and achieves new state-of-the-art compared to existing methods. Experimental results suggest the effectiveness of DeCode in improving clinical question answering of LLMs.
>
---
#### [replaced 042] Tiny Recursive Reasoning with Mamba-2 Attention Hybrid
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于抽象推理任务，研究如何将Mamba-2混合算子融入递归框架，以提升模型的推理能力。通过替换Transformer块，验证了Mamba-2在保持参数量的同时有效提升性能。**

- **链接: [https://arxiv.org/pdf/2602.12078](https://arxiv.org/pdf/2602.12078)**

> **作者:** Wenlong Wang; Fergal Reid
>
> **备注:** Published at ICLR 2026 Latent & Implicit Thinking Workshop
>
> **摘要:** Recent work on recursive reasoning models like TRM demonstrates that tiny networks (7M parameters) can achieve strong performance on abstract reasoning tasks through latent recursion -- iterative refinement in hidden representation space without emitting intermediate tokens. This raises a natural question about operator choice: Mamba-2's state space recurrence is itself a form of iterative refinement, making it a natural candidate for recursive reasoning -- but does introducing Mamba-2 into the recursive scaffold preserve reasoning capability? We investigate this by replacing the Transformer blocks in TRM with Mamba-2 hybrid operators while maintaining parameter parity (6.83M vs 6.86M parameters). On ARC-AGI-1, we find that the hybrid improves pass@2 (the official metric) by +2.0\% (45.88\% vs 43.88\%) and consistently outperforms at higher K values (+4.75\% at pass@100), whilst maintaining pass@1 parity. This suggests improved candidate coverage -- the model generates correct solutions more reliably -- with similar top-1 selection. Our results validate that Mamba-2 hybrid operators preserve reasoning capability within the recursive scaffold, establishing SSM-based operators as viable candidates in the recursive operator design space and taking a first step towards understanding the best mixing strategies for recursive reasoning.
>
---
#### [replaced 043] Computational lexical analysis of Flamenco genres
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于文本分类任务，旨在通过计算方法分析弗拉门戈歌词，识别其风格（palos），揭示风格间的语义特征与关系。**

- **链接: [https://arxiv.org/pdf/2405.05723](https://arxiv.org/pdf/2405.05723)**

> **作者:** Pablo Rosillo-Rodes; Maxi San Miguel; David Sanchez
>
> **备注:** 25 pages, 20 figures
>
> **摘要:** Flamenco, recognized by UNESCO as part of the Intangible Cultural Heritage of Humanity, is a profound expression of cultural identity rooted in Andalusia, Spain. However, there is a lack of quantitative studies that help identify characteristic patterns in this long-lived music tradition. In this work, we present a computational analysis of Flamenco lyrics, employing natural language processing and machine learning to categorize over 2000 lyrics into their respective Flamenco genres, termed as $\textit{palos}$. Using a Multinomial Naive Bayes classifier, we find that lexical variation across styles enables to accurately identify distinct $\textit{palos}$. More importantly, from an automatic method of word usage, we obtain the semantic fields that characterize each style. Further, applying a metric that quantifies the inter-genre distance we perform a network analysis that sheds light on the relationship between Flamenco styles. Remarkably, our results suggest historical connections and $\textit{palo}$ evolutions. Overall, our work illuminates the intricate relationships and cultural significance embedded within Flamenco lyrics, complementing previous qualitative discussions with quantitative analyses and sparking new discussions on the origin and development of traditional music genres.
>
---
