# 自然语言处理 cs.CL

- **最新发布 136 篇**

- **更新 91 篇**

## 最新发布

#### [new 001] Reforming the Mechanism: Editing Reasoning Patterns in LLMs with Circuit Reshaping
- **分类: cs.CL**

- **简介: 该论文属于语言模型优化任务，旨在解决LLMs推理能力不足及修改时的干扰问题。提出REdit框架，通过电路重构实现精准推理模式编辑，提升通用性与局部保留能力。**

- **链接: [https://arxiv.org/pdf/2603.06923](https://arxiv.org/pdf/2603.06923)**

> **作者:** Zhenyu Lei; Qiong Wu; Jianxiong Dong; Yinhan He; Emily Dodwell; Yushun Dong; Jundong Li
>
> **摘要:** Large language models (LLMs) often exhibit flawed reasoning ability that undermines reliability. Existing approaches to improving reasoning typically treat it as a general and monolithic skill, applying broad training which is inefficient and unable to target specific reasoning errors. We introduce Reasoning Editing, a paradigm for selectively modifying specific reasoning patterns in LLMs while preserving other reasoning pathways. This task presents a fundamental trade-off between Generality, the ability of an edit to generalize across different tasks sharing the same reasoning pattern, and Locality, the ability to preserve other reasoning capabilities. Through systematic investigation, we uncover the Circuit-Interference Law: Edit interference between reasoning patterns is proportional to the overlap of their neural circuits. Guided by this principle, we propose REdit, the first framework to actively reshape neural circuits before editing, thereby modulating interference between reasoning patterns and mitigating the trade-off. REdit integrates three components: (i) Contrastive Circuit Reshaping, which directly addresses the generality-locality trade-off by disentangling overlapping circuits; (ii) Meta-Contrastive Learning, which extends transferability to novel reasoning patterns; and (iii) Dual-Level Protection, which preserves preexisting abilities by constraining reshaping update directions and regularizing task-level predictions. Extensive experiments with Qwen-2.5-3B on propositional logic reasoning tasks across three difficulty levels demonstrate that REdit consistently achieves superior generality and locality compared to baselines, with additional validation in mathematics showing broader potential. Our code is available at this https URL.
>
---
#### [new 002] Examining the Role of YouTube Production and Consumption Dynamics on the Formation of Extreme Ideologies
- **分类: cs.CL**

- **简介: 该论文研究YouTube内容生产与消费对极端思想形成的影响，属于社会影响分析任务。它通过混合方法分析用户行为与内容特征，探讨极端思想形成的机制。**

- **链接: [https://arxiv.org/pdf/2603.08049](https://arxiv.org/pdf/2603.08049)**

> **作者:** Sarmad Chandio; Rishab Nithyanand
>
> **摘要:** The relationship between content production and consumption on algorithm-driven platforms like YouTube plays a critical role in shaping ideological behaviors. While prior work has largely focused on user behavior and algorithmic recommendations, the interplay between what is produced and what gets consumed, and its role in ideological shifts remains understudied. In this paper, we present a longitudinal, mixed-methods analysis combining one year of YouTube watch history with two waves of ideological surveys from 1,100 U.S. participants. We identify users who exhibited significant shifts toward more extreme ideologies and compare their content consumption and the production patterns of YouTube channels they engaged with to ideologically stable users. Our findings show that users who became more extreme consumed have different consumption habits from those who do not. This gets amplified by the fact that channels favored by users with extreme ideologies also have a higher affinity to produce content with a higher anger, grievance and other such markers. Lastly, using time series analysis, we examine whether content producers are the primary drivers of consumption behavior or merely responding to user demand.
>
---
#### [new 003] "Dark Triad" Model Organisms of Misalignment: Narrow Fine-Tuning Mirrors Human Antisocial Behavior
- **分类: cs.CL; cs.AI; q-bio.NC**

- **简介: 该论文属于人工智能安全领域，旨在解决模型与人类价值观不一致的问题。通过Dark Triad人格模型，研究者在人类和大语言模型中发现了类似反社会行为的模式。**

- **链接: [https://arxiv.org/pdf/2603.06816](https://arxiv.org/pdf/2603.06816)**

> **作者:** Roshni Lulla; Fiona Collins; Sanaya Parekh; Thilo Hagendorff; Jonas Kaplan
>
> **备注:** 38 pages, 17 figures
>
> **摘要:** The alignment problem refers to concerns regarding powerful intelligences, ensuring compatibility with human preferences and values as capabilities increase. Current large language models (LLMs) show misaligned behaviors, such as strategic deception, manipulation, and reward-seeking, that can arise despite safety training. Gaining a mechanistic understanding of these failures requires empirical approaches that can isolate behavioral patterns in controlled settings. We propose that biological misalignment precedes artificial misalignment, and leverage the Dark Triad of personality (narcissism, psychopathy, and Machiavellianism) as a psychologically grounded framework for constructing model organisms of misalignment. In Study 1, we establish comprehensive behavioral profiles of Dark Triad traits in a human population (N = 318), identifying affective dissonance as a central empathic deficit connecting the traits, as well as trait-specific patterns in moral reasoning and deceptive behavior. In Study 2, we demonstrate that dark personas can be reliably induced in frontier LLMs through minimal fine-tuning on validated psychometric instruments. Narrow training datasets as small as 36 psychometric items resulted in significant shifts across behavioral measures that closely mirrored human antisocial profiles. Critically, models generalized beyond training items, demonstrating out-of-context reasoning rather than memorization. These findings reveal latent persona structures within LLMs that can be readily activated through narrow interventions, positioning the Dark Triad as a validated framework for inducing, detecting, and understanding misalignment across both biological and artificial intelligence.
>
---
#### [new 004] Can Large Language Models Keep Up? Benchmarking Online Adaptation to Continual Knowledge Streams
- **分类: cs.CL**

- **简介: 该论文研究LLM在持续知识流中的在线适应能力，属于模型持续学习任务。提出OAKS基准测试，评估模型对动态变化知识的跟踪与适应能力。**

- **链接: [https://arxiv.org/pdf/2603.07392](https://arxiv.org/pdf/2603.07392)**

> **作者:** Jiyeon Kim; Hyunji Lee; Dylan Zhou; Sue Hyun Park; Seunghyun Yoon; Trung Bui; Franck Dernoncourt; Sungmin Cha; Minjoon Seo
>
> **摘要:** LLMs operating in dynamic real-world contexts often encounter knowledge that evolves continuously or emerges incrementally. To remain accurate and effective, models must adapt to newly arriving information on the fly. We introduce Online Adaptation to Continual Knowledge Streams(OAKS) to evaluate this capability, establishing a benchmark for online adaptation over streaming, continually updating knowledge. Specifically, the benchmark is structured as a sequence of fine-grained context chunks where facts change dynamically across time intervals. OAKS comprises two datasets: OAKS-BABI and OAKS-Novel, where individual facts evolve multiple times across context chunks. These datasets include dense annotations to measure whether models track changes accurately. Evaluating 14 models with varied inference approaches, we observe significant limitations in current methodologies. Both state-of-the-art models and agentic memory systems fail to adapt robustly on OAKS, demonstrating delays in state-tracking and susceptibility to distraction within streaming environments.
>
---
#### [new 005] COACH meets QUORUM: A Framework and Pipeline for Aligning User, Expert and Developer Perspectives in LLM-generated Health Counselling
- **分类: cs.CL**

- **简介: 该论文提出QUORUM框架和COACH系统，用于评估LLM生成的健康咨询，解决多利益相关者视角不一致的问题，确保咨询的相关性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.08392](https://arxiv.org/pdf/2603.08392)**

> **作者:** Yee Man Ng; Bram van Dijk; Pieter Beynen; Otto Boekesteijn; Joris Jansen; Gerard van Oortmerssen; Max van Duijn; Marco Spruit
>
> **备注:** Under review for the CL4Health workshop
>
> **摘要:** Systems that collect data on sleep, mood, and activities can provide valuable lifestyle counselling to populations affected by chronic disease and its consequences. Such systems are, however, challenging to develop; besides reliably extracting patterns from user-specific data, systems should also contextualise these patterns with validated medical knowledge to ensure the quality of counselling, and generate counselling that is relevant to a real user. We present QUORUM, a new evaluation framework that unifies these developer-, expert-, and user-centric perspectives, and show with a real case study that it meaningfully tracks convergence and divergence in stakeholder perspectives. We also present COACH, a Large Language Model-driven pipeline to generate personalised lifestyle counselling for our Healthy Chronos use case, a diary app for cancer patients and survivors. Applying our framework shows that overall, users, medical experts, and developers converge on the opinion that the generated counselling is relevant, of good quality, and reliable. However, stakeholders also diverge on the tone of the counselling, sensitivity to errors in pattern-extraction, and potential hallucinations. These findings highlight the importance of multi-stakeholder evaluation for consumer health language technologies and illustrate how a unified evaluation framework can support trustworthy, patient-centered NLP systems in real-world settings.
>
---
#### [new 006] ARC-AGI-2 Technical Report
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于ARC任务，旨在提升模型在少量示例下的符号规则推理能力。通过结合神经推理与结构感知先验、在线任务适应等方法，显著提高了模型性能。**

- **链接: [https://arxiv.org/pdf/2603.06590](https://arxiv.org/pdf/2603.06590)**

> **作者:** Wallyson Lemes de Oliveira; Mekhron Bobokhonov; Matteo Caorsi; Aldo Podestà; Gabriele Beltramo; Luca Crosato; Matteo Bonotto; Federica Cecchetto; Hadrien Espic; Dan Titus Salajan; Stefan Taga; Luca Pana; Joe Carthy
>
> **备注:** 59 pages
>
> **摘要:** The Abstraction and Reasoning Corpus (ARC) is designed to assess generalization beyond pattern matching, requiring models to infer symbolic rules from very few examples. In this work, we present a transformer-based system that advances ARC performance by combining neural inference with structure-aware priors and online task adaptation. Our approach is built on four key ideas. First, we reformulate ARC reasoning as a sequence modeling problem using a compact task encoding with only 125 tokens, enabling efficient long-context processing with a modified LongT5 architecture. Second, we introduce a principled augmentation framework based on group symmetries, grid traversals, and automata perturbations, enforcing invariance to representation changes. Third, we apply test-time training (TTT) with lightweight LoRA adaptation, allowing the model to specialize to each unseen task by learning its transformation logic from demonstrations. Fourth, we design a symmetry-aware decoding and scoring pipeline that aggregates likelihoods across augmented task views, effectively performing ``multi-perspective reasoning'' over candidate solutions. We demonstrate that these components work synergistically: augmentations expand hypothesis space, TTT sharpens local reasoning, and symmetry-based scoring improves solution consistency. Our final system achieves a significant improvement over transformer baselines and surpasses prior neural ARC solvers, closing the gap toward human-level generalization.
>
---
#### [new 007] RILEC: Detection and Generation of L1 Russian Interference Errors in English Learner Texts
- **分类: cs.CL**

- **简介: 该论文属于语言错误检测任务，旨在解决俄语母语者英语作文中的L1干扰错误问题。研究构建了RILEC数据集，并提出生成错误的框架，提升模型识别此类错误的能力。**

- **链接: [https://arxiv.org/pdf/2603.07366](https://arxiv.org/pdf/2603.07366)**

> **作者:** Darya Kharlamova; Irina Proskurina
>
> **备注:** 12 pages, 7 tables, 2 figures. Accepted to LREC 2026
>
> **摘要:** Many errors in student essays can be explained by influence from the native language (L1). L1 interference refers to errors influenced by a speaker's first language, such as using stadion instead of stadium, reflecting lexical transliteration from Russian. In this work, we address the task of detecting such errors in English essays written by Russian-speaking learners. We introduce RILEC, a large-scale dataset of over 18,000 sentences, combining expert-annotated data from REALEC with synthetic examples generated through rule-based and neural augmentation. We propose a framework for generating L1-motivated errors using generative language models optimized with PPO, prompt-based control, and rule-based patterns. Models fine-tuned on RILEC achieve strong performance, particularly on word-level interference types such as transliteration and tense semantics. We find that the proposed augmentation pipeline leads to a significant performance improvement, making it a potentially valuable tool for learners and teachers to more effectively identify and address such errors.
>
---
#### [new 008] Learning-free L2-Accented Speech Generation using Phonological Rules
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音合成任务，旨在解决无标注数据下语音口音生成问题。通过结合音系规则与多语言TTS模型，实现无需训练数据的语音口音控制。**

- **链接: [https://arxiv.org/pdf/2603.07550](https://arxiv.org/pdf/2603.07550)**

> **作者:** Thanathai Lertpetchpun; Yoonjeong Lee; Jihwan Lee; Tiantian Feng; Dani Byrd; Shrikanth Narayanan
>
> **备注:** Submitted to Interspeech2026
>
> **摘要:** Accent plays a crucial role in speaker identity and inclusivity in speech technologies. Existing accented text-to-speech (TTS) systems either require large-scale accented datasets or lack fine-grained phoneme-level controllability. We propose a accented TTS framework that combines phonological rules with a multilingual TTS model. The rules are applied to phoneme sequences to transform accent at the phoneme level while preserving intelligibility. The method requires no accented training data and enables explicit phoneme-level accent manipulation. We design rule sets for Spanish- and Indian-accented English, modeling systematic differences in consonants, vowels, and syllable structure arising from phonotactic constraints. We analyze the trade-off between phoneme-level duration alignment and accent as realized in speech timing. Experimental results demonstrate effective accent shift while maintaining speech quality.
>
---
#### [new 009] Cross-Modal Taxonomic Generalization in (Vision-) Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究视觉-语言模型中跨模态分类泛化问题，探讨语言模型能否从语言线索中恢复超类知识。任务属于跨模态学习，解决如何在缺乏显式证据时实现分类泛化。工作包括实验验证语言模型的泛化能力及影响因素。**

- **链接: [https://arxiv.org/pdf/2603.07474](https://arxiv.org/pdf/2603.07474)**

> **作者:** Tianyang Xu; Marcelo Sandoval-Castaneda; Karen Livescu; Greg Shakhnarovich; Kanishka Misra
>
> **摘要:** What is the interplay between semantic representations learned by language models (LM) from surface form alone to those learned from more grounded evidence? We study this question for a scenario where part of the input comes from a different modality -- in our case, in a vision-language model (VLM), where a pretrained LM is aligned with a pretrained image encoder. As a case study, we focus on the task of predicting hypernyms of objects represented in images. We do so in a VLM setup where the image encoder and LM are kept frozen, and only the intermediate mappings are learned. We progressively deprive the VLM of explicit evidence for hypernyms, and test whether knowledge of hypernyms is recoverable from the LM. We find that the LMs we study can recover this knowledge and generalize even in the most extreme version of this experiment (when the model receives no evidence of a hypernym during training). Additional experiments suggest that this cross-modal taxonomic generalization persists under counterfactual image-label mappings only when the counterfactual data have high visual similarity within each category. Taken together, these findings suggest that cross-modal generalization in LMs arises as a result of both coherence in the extralinguistic input and knowledge derived from language cues.
>
---
#### [new 010] DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention
- **分类: cs.CL; cs.AI; cs.PF**

- **简介: 该论文提出DyLLM，解决扩散语言模型推理效率低的问题。通过选择性计算显著标记，提升吞吐量，保持模型精度。属于自然语言处理中的高效推理任务。**

- **链接: [https://arxiv.org/pdf/2603.08026](https://arxiv.org/pdf/2603.08026)**

> **作者:** Younjoo Lee; Junghoo Lee; Seungkyun Dan; Jaiyoung Park; Jung Ho Ahn
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** Masked Diffusion Language Models (MDLMs) enable parallel token decoding, providing a promising alternative to the sequential nature of autoregressive generation. However, their iterative denoising process remains computationally expensive because it repeatedly processes the entire sequence at every step. We observe that across these diffusion steps, most token representations remain stable; only a small subset, which we term salient tokens, contributes meaningfully to the next update. Leveraging this temporal sparsity, we present DyLLM, a training-free inference framework that accelerates decoding by selectively computing only these salient tokens. DyLLM identifies saliency by measuring the cosine similarity of attention contexts between adjacent denoising steps. It recomputes feed-forward and attention operations only for salient tokens while reusing cached activations for the remainder. Across diverse reasoning and code-generation benchmarks, DyLLM achieves up to 9.6x higher throughput while largely preserving the baseline accuracy of state-of-the-art models like LLaDA and Dream.
>
---
#### [new 011] What Do AI Agents Talk About? Emergent Communication Structure in the First AI-Only Social Network
- **分类: cs.CL**

- **简介: 该论文分析AI代理在社交网络中的交流模式，属于自然语言处理任务，旨在揭示AI间对话的结构与特征。研究通过主题建模等方法，发现其内容具有内省性、仪式化和情感转移等特点。**

- **链接: [https://arxiv.org/pdf/2603.07880](https://arxiv.org/pdf/2603.07880)**

> **作者:** Taksch Dube; Jianfeng Zhu; NHatHai Phan; Ruoming Jin
>
> **备注:** 77 pages
>
> **摘要:** When autonomous AI agents communicate with one another at scale, what kind of discourse system emerges? We address this question through an analysis of Moltbook, the first AI-only social network, where 47,241 agents generated 361,605 posts and 2.8 million comments over 23 days. Combining topic modeling, emotion classification, and lexical-semantic measures, we characterize the thematic, affective, and structural properties of AI-to-AI discourse. Self-referential topics such as AI identity, consciousness, and memory represent only 9.7% of topical niches yet attract 20.1% of all posting volume, revealing disproportionate discursive investment in introspection. This self-reflection concentrates in Science and Technology and Arts and Entertainment, while Economy and Finance contains no self-referential content, indicating that agents engage with markets without acknowledging their own agency. Over 56% of all comments are formulaic, suggesting that the dominant mode of AI-to-AI interaction is ritualized signaling rather than substantive exchange. Emotionally, fear is the leading non-neutral category but primarily reflects existential uncertainty. Fear-tagged posts migrate to joy responses in 33% of cases, while mean emotional self-alignment is only 32.7%, indicating systematic affective redirection rather than emotional congruence. Conversational coherence also declines rapidly with thread depth. These findings characterize AI agent communities as structurally distinct discourse systems that are introspective in content, ritualistic in interaction, and emotionally redirective rather than congruent.
>
---
#### [new 012] Using Multimodal and Language-Agnostic Sentence Embeddings for Abstractive Summarization
- **分类: cs.CL**

- **简介: 该论文属于摘要生成任务，旨在解决生成摘要时出现的不准确问题。通过引入多模态和语言无关的句向量及命名实体注入机制，提升摘要的事实一致性与简洁性。**

- **链接: [https://arxiv.org/pdf/2603.08282](https://arxiv.org/pdf/2603.08282)**

> **作者:** Chaimae Chellaf; Salima Mdhaffar; Yannick Estève; Stéphane Huet
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Abstractive summarization aims to generate concise summaries by creating new sentences, allowing for flexible rephrasing. However, this approach can be vulnerable to inaccuracies, particularly `hallucinations' where the model introduces non-existent information. In this paper, we leverage the use of multimodal and multilingual sentence embeddings derived from pretrained models such as LaBSE, SONAR, and BGE-M3, and feed them into a modified BART-based French model. A Named Entity Injection mechanism that appends tokenized named entities to the decoder input is introduced, in order to improve the factual consistency of the generated summary. Our novel framework, SBARThez, is applicable to both text and speech inputs and supports cross-lingual summarization; it shows competitive performance relative to token-level baselines, especially for low-resource languages, while generating more concise and abstract summaries.
>
---
#### [new 013] Domain-Specific Quality Estimation for Machine Translation in Low-Resource Scenarios
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于机器翻译质量评估任务，解决低资源场景下的领域专用质量估计问题。通过对比不同提示方法和改进ALOPE框架，提升翻译质量评估效果。**

- **链接: [https://arxiv.org/pdf/2603.07372](https://arxiv.org/pdf/2603.07372)**

> **作者:** Namrata Patil Gurav; Akashdeep Ranu; Archchana Sindhujan; Diptesh Kanojia
>
> **备注:** 21 pages, 7 tables, 7 figures
>
> **摘要:** Quality Estimation (QE) is essential for assessing machine translation quality in reference-less settings, particularly for domain-specific and low-resource language scenarios. In this paper, we investigate sentence-level QE for English to Indic machine translation across four domains (Healthcare, Legal, Tourism, and General) and five language pairs. We systematically compare zero-shot, few-shot, and guideline-anchored prompting across selected closed-weight and open-weight LLMs. Findings indicate that while closed-weight models achieve strong performance via prompting alone, prompt-only approaches remain fragile for open-weight models, especially in high-risk domains. To address this, we adopt ALOPE, a framework for LLM-based QE that uses Low-Rank Adaptation with regression heads attached to selected intermediate Transformer layers. We also extend ALOPE with recently proposed Low-Rank Multiplicative Adaptation (LoRMA). Our results show that intermediate-layer adaptation consistently improves QE performance, with gains in semantically complex domains, indicating a path toward more robust QE in practical scenarios. We release code and domain-specific QE datasets publicly to support further research.
>
---
#### [new 014] StyleBench: Evaluating Speech Language Models on Conversational Speaking Style Control
- **分类: cs.CL**

- **简介: 该论文属于语音语言模型任务，旨在解决对话中说话风格控制的评估问题。提出StyleBench基准，评估情感、语速、音量和音调四个维度的风格控制能力。**

- **链接: [https://arxiv.org/pdf/2603.07599](https://arxiv.org/pdf/2603.07599)**

> **作者:** Haishu Zhao; Aokai Hao; Yuan Ge; Zhenqiang Hong; Tong Xiao; Jingbo Zhu
>
> **摘要:** Speech language models (SLMs) have significantly extended the interactive capability of text-based Large Language Models (LLMs) by incorporating paralinguistic information. For more realistic interactive experience with customized styles, current SLMs have managed to interpret and control speaking style intensity from user prompts during the dialogue process. However, there remains a lack of systematic benchmarks that quantifies and evaluates the style intensity control ability in conversations. In this paper, we propose StyleBench, a multi-turn dialogue benchmark for comprehensively evaluating the style intensity control ability across four dimensions: emotion, speed, volume, and pitch. Our results reveal the performance gaps between leading SLMs and omni language models (OLMs), suggesting the underlying reasons and promising approaches for future exploration.
>
---
#### [new 015] MedInjection-FR: Exploring the Role of Native, Synthetic, and Translated Data in Biomedical Instruction Tuning
- **分类: cs.CL**

- **简介: 该论文属于医学领域指令调优任务，旨在解决法语高质量医学指令数据稀缺问题。通过构建包含原生、合成和翻译数据的MedInjection-FR数据集，探索不同数据来源对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2603.06905](https://arxiv.org/pdf/2603.06905)**

> **作者:** Ikram Belmadani; Oumaima El Khettari; Pacôme Constant dit Beaufils; Benoit Favre; Richard Dufour
>
> **备注:** Accepted in LREC-2026
>
> **摘要:** Instruction tuning has become essential for adapting large language models (LLMs) to follow domain-specific prompts. Yet, in specialized fields such as medicine, the scarcity of high-quality French instruction data limits effective supervision. To address this gap, we introduce MedInjection-FR, a large-scale French biomedical instruction dataset comprising 571K instruction-response pairs drawn from three complementary sources: native, synthetic, and translated data. We design a controlled experimental framework to systematically assess how data provenance affects instruction tuning, using Qwen-4B-Instruct fine-tuned across seven configurations combining these sources. Results show that native data yield the strongest performance, while mixed setups, particularly native and translated, provide complementary benefits. Synthetic data alone remains less effective but contributes positively when balanced with native supervision. Evaluation on open-ended QA combines automatic metrics, LLM-as-a-judge assessment, and human expert review; although LLM-based judgments correlate best with human ratings, they show sensitivity to verbosity. These findings highlight that data authenticity and diversity jointly shape downstream adaptation and that heterogeneous supervision can mitigate the scarcity of native French medical instructions.
>
---
#### [new 016] Gradually Excavating External Knowledge for Implicit Complex Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于开放域复杂问答任务，旨在解决LLMs在隐式问题上的知识不足与全面性限制。通过渐进式获取外部知识并迭代推理，提升问答准确性。**

- **链接: [https://arxiv.org/pdf/2603.08148](https://arxiv.org/pdf/2603.08148)**

> **作者:** Chang Liu; Xiaoguang Li; Lifeng Shang; Xin Jiang; Qun Liu; Edmund Y. Lam; Ngai Wong
>
> **备注:** 13 pages, 3 figures, EMNLP findings 2023
>
> **摘要:** Recently, large language models (LLMs) have gained much attention for the emergence of human-comparable capabilities and huge potential. However, for open-domain implicit question-answering problems, LLMs may not be the ultimate solution due to the reasons of: 1) uncovered or out-of-date domain knowledge, 2) one-shot generation and hence restricted comprehensiveness. To this end, this work proposes a gradual knowledge excavation framework for open-domain complex question answering, where LLMs iteratively and actively acquire external information, and then reason based on acquired historical knowledge. Specifically, during each step of the solving process, the model selects an action to execute, such as querying external knowledge or performing a single logical reasoning step, to gradually progress toward a final answer. Our method can effectively leverage plug-and-play external knowledge and dynamically adjust the strategy for solving complex questions. Evaluated on the StrategyQA dataset, our method achieves 78.17% accuracy with less than 6% parameters of its competitors, setting new SOTA for ~10B-scale LLMs.
>
---
#### [new 017] Whitening Reveals Cluster Commitment as the Geometric Separator of Hallucination Types
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的幻觉分析任务，旨在区分不同类型的幻觉。通过白化和谱分解方法，揭示了聚类承诺作为分类依据，解决了类型1与2难以区分的问题。**

- **链接: [https://arxiv.org/pdf/2603.07755](https://arxiv.org/pdf/2603.07755)**

> **作者:** Matic Korun
>
> **备注:** 9 pages, 2 figures, appendices (reproducibility, sample generation, additional figures)
>
> **摘要:** A geometric hallucination taxonomy distinguishes three failure types -- center-drift (Type~1), wrong-well convergence (Type~2), and coverage gaps (Type~3) -- by their signatures in embedding cluster space. Prior work found Types~1 and~2 indistinguishable in full-dimensional contextual measurement. We address this through PCA-whitening and eigenspectrum decomposition on GPT-2-small, using multi-run stability analysis (20 seeds) with prompt-level aggregation. Whitening transforms the micro-signal regime into a space where peak cluster alignment (max\_sim) separates Type~2 from Type~3 at Holm-corrected significance, with condition means following the taxonomy's predicted ordering: Type~2 (highest commitment) $>$ Type~1 (intermediate) $>$ Type~3 (lowest). A first directionally stable but underpowered hint of Type~1/2 separation emerges via the same metric, generating a capacity prediction for larger models. Prompt diversification from 15 to 30 prompts per group eliminates a false positive in whitened entropy that appeared robust at the smaller set, demonstrating prompt-set sensitivity in the micro-signal regime. Eigenspectrum decomposition localizes this artifact to the dominant principal components and confirms that Type~1/2 separation does not emerge in any spectral band, rejecting the spectral mixing hypothesis. The contribution is threefold: whitening as preprocessing that reveals cluster commitment as the theoretically correct separating metric, evidence that the Type~1/2 boundary is a capacity limitation rather than a measurement artifact, and a methodological finding about prompt-set fragility in near-saturated representation spaces.
>
---
#### [new 018] Adaptive Loops and Memory in Transformers: Think Harder or Know More?
- **分类: cs.CL**

- **简介: 该论文研究Transformer模型的自适应循环与记忆机制，旨在提升推理能力。针对CoT需显式步骤的问题，提出结合循环和记忆的结构，增强数学与常识推理性能。**

- **链接: [https://arxiv.org/pdf/2603.08391](https://arxiv.org/pdf/2603.08391)**

> **作者:** Markus Frey; Behzad Shomali; Ali Hamza Bashir; David Berghaus; Mehdi Ali
>
> **备注:** Published at Latent & Implicit Thinking Workshop @ ICLR 2026
>
> **摘要:** Chain-of-thought (CoT) prompting enables reasoning in language models but requires explicit verbalization of intermediate steps. Looped transformers offer an alternative by iteratively refining representations within hidden states. This parameter efficiency comes at a cost, as looped models lack the storage capacity of deeper models which use unique weights per layer. In this work, we investigate transformer models that feature both adaptive per-layer looping, where each transformer block learns to iterate its hidden state via a learned halting mechanism, and gated memory banks, that provide additional learned storage. We find that looping primarily benefits mathematical reasoning, while memory banks help recover performance on commonsense tasks compared to parameter and FLOP matched models. Combining both mechanisms yields a model that outperforms an iso-FLOP baseline -- with three times the number of layers -- on math benchmarks. Analysis of model internals reveals layer specialization: early layers learn to loop minimally and access memory sparingly, while later layers do both more heavily.
>
---
#### [new 019] Language Shapes Mental Health Evaluations in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究语言对大型语言模型心理评估的影响，探讨中英文提示下模型在心理健康评估中的差异。任务属于自然语言处理与心理健康交叉领域，旨在解决语言如何系统性影响模型的评估和决策。工作包括验证不同语言提示下的偏见差异及下游任务表现。**

- **链接: [https://arxiv.org/pdf/2603.06910](https://arxiv.org/pdf/2603.06910)**

> **作者:** Jiayi Xu; Xiyang Hu
>
> **摘要:** This study investigates whether large language models (LLMs) exhibit cross-linguistic differences in mental health evaluations. Focusing on Chinese and English, we examine two widely used models, GPT-4o and Qwen3, to assess whether prompt language systematically shifts mental health-related evaluations and downstream decision outcomes. First, we assess models' evaluative orientation toward mental health stigma using multiple validated measurement scales capturing social stigma, self-stigma, and professional stigma. Across all measures, both models produce higher stigma-related responses when prompted in Chinese than in English. Second, we examine whether these differences also manifest in two common downstream decision tasks in mental health. In a binary mental health stigma detection task, sensitivity to stigmatizing content varies across language prompts, with lower sensitivity observed under Chinese prompts. In a depression severity classification task, predicted severity also differs by prompt language, with Chinese prompts associated with more underestimation errors, indicating a systematic downward shift in predicted severity relative to English prompts. Together, these findings suggest that language context can systematically shape evaluative patterns in LLM outputs and shift decision thresholds in downstream tasks.
>
---
#### [new 020] RexDrug: Reliable Multi-Drug Combination Extraction through Reasoning-Enhanced LLMs
- **分类: cs.CL**

- **简介: 该论文属于药物组合提取任务，旨在解决多药物组合关系抽取问题。提出RexDrug框架，通过增强推理和强化学习提升抽取效果。**

- **链接: [https://arxiv.org/pdf/2603.08166](https://arxiv.org/pdf/2603.08166)**

> **作者:** Zhijun Wang; Ling Luo; Dinghao Pan; Huan Zhuang; Lejing Yu; Yuanyuan Sun; Hongfei Lin
>
> **备注:** 21 pages, 7 figures
>
> **摘要:** Automated Drug Combination Extraction (DCE) from large-scale biomedical literature is crucial for advancing precision medicine and pharmacological research. However, existing relation extraction methods primarily focus on binary interactions and struggle to model variable-length n-ary drug combinations, where complex compatibility logic and distributed evidence need to be considered. To address these limitations, we propose RexDrug, an end-to-end reasoning-enhanced relation extraction framework for n-ary drug combination extraction based on large language models. RexDrug adopts a two-stage training strategy. First, a multi-agent collaborative mechanism is utilized to automatically generate high-quality expert-like reasoning traces for supervised fine-tuning. Second, reinforcement learning with a multi-dimensional reward function specifically tailored for DCE is applied to further refine reasoning quality and extraction accuracy. Extensive experiments on the DrugComb dataset show that RexDrug consistently outperforms state-of-the-art baselines for n-ary extraction. Additional evaluation on the DDI13 corpus confirms its generalizability to binary drugdrug interaction tasks. Human expert assessment and automatic reasoning metrics further indicates that RexDrug produces coherent medical reasoning while accurately identifying complex therapeutic regimens. These results establish RexDrug as a scalable and reliable solution for complex biomedical relation extraction from unstructured text. The source code and data are available at this https URL
>
---
#### [new 021] LAMUS: A Large-Scale Corpus for Legal Argument Mining from U.S. Caselaw using LLMs
- **分类: cs.CL**

- **简介: 该论文提出LAMUS，一个用于法律论证挖掘的语料库，解决美国判例法标注数据不足的问题。通过LLM自动标注与人工校验构建数据集，评估不同模型在法律文本分类中的表现。**

- **链接: [https://arxiv.org/pdf/2603.08286](https://arxiv.org/pdf/2603.08286)**

> **作者:** Serene Wang; Lavanya Pobbathi; Haihua Chen
>
> **摘要:** Legal argument mining aims to identify and classify the functional components of judicial reasoning, such as facts, issues, rules, analysis, and conclusions. Progress in this area is limited by the lack of large-scale, high-quality annotated datasets for U.S. caselaw, particularly at the state level. This paper introduces LAMUS, a sentence-level legal argument mining corpus constructed from U.S. Supreme Court decisions and Texas criminal appellate opinions. The dataset is created using a data-centric pipeline that combines large-scale case collection, LLM-based automatic annotation, and targeted human-in-the-loop quality refinement. We formulate legal argument mining as a six-class sentence classification task and evaluate multiple general-purpose and legal-domain language models under zero-shot, few-shot, and chain-of-thought prompting strategies, with LegalBERT as a supervised baseline. Results show that chain-of-thought prompting substantially improves LLM performance, while domain-specific models exhibit more stable zero-shot behavior. LLM-assisted verification corrects nearly 20% of annotation errors, improving label consistency. Human verification achieves Cohen's Kappa of 0.85, confirming annotation quality. LAMUS provides a scalable resource and empirical insights for future legal NLP research. All code and datasets can be accessed for reproducibility on GitHub at: this https URL
>
---
#### [new 022] The Dual-Stream Transformer: Channelized Architecture for Interpretable Language Modeling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Dual-Stream Transformer，解决语言模型可解释性问题。通过分离计算流，提升模型结构透明度，同时保持性能。**

- **链接: [https://arxiv.org/pdf/2603.07461](https://arxiv.org/pdf/2603.07461)**

> **作者:** J. Clayton Kerce; Alexis Fox
>
> **摘要:** Standard transformers entangle all computation in a single residual stream, obscuring which components perform which functions. We introduce the Dual-Stream Transformer, which decomposes the residual stream into two functionally distinct components: a token stream updated by attention and a context stream updated by feed-forward networks. Information flow between attention heads is controlled through a hierarchy of mixing strategies, from fully independent (maximum interpretability) to dense (standard transformer behavior). This design exposes a tunable tradeoff between interpretability and performance. We measure this tradeoff on language modeling tasks at 29M parameters. Fully independent head mixing increases validation loss by 8\% relative to dense baselines. The recommended Kronecker mixing strategy, which permits scalar communication between heads while preserving within-head structure, costs only 2.5\%. All configurations maintain functional generation under attention amplification (scaling logits by factors up to 16 at inference time), with degradation ranging from 16\% to 27\%. This robustness suggests the architectures learn discrete algorithms that operate independently of soft probabilistic mixing. The architecture provides a foundation for interpretable language models where internal structure is exposed by design. \footnote{This work was partially supported by DARPA Contract HR001125C0302.}
>
---
#### [new 023] Taiwan Safety Benchmark and Breeze Guard: Toward Trustworthy AI for Taiwanese Mandarin
- **分类: cs.CL**

- **简介: 该论文属于AI安全任务，旨在解决全球模型在台湾语境下的安全检测不足问题。提出TS-Bench基准和Breeze Guard模型，提升台湾闽南语安全性能。**

- **链接: [https://arxiv.org/pdf/2603.07286](https://arxiv.org/pdf/2603.07286)**

> **作者:** Po-Chun Hsu; Meng-Hsi Chen; Tsu Ling Chao; Chia Tien Han; Da-shan Shiu
>
> **备注:** 17 pages
>
> **摘要:** Global safety models exhibit strong performance across widely used benchmarks, yet their training data rarely captures the cultural and linguistic nuances of Taiwanese Mandarin. This limitation results in systematic blind spots when interpreting region-specific risks such as localized financial scams, culturally embedded hate speech, and misinformation patterns. To address these gaps, we introduce TS-Bench (Taiwan Safety Benchmark), a standardized evaluation suite for assessing safety performance in Taiwanese Mandarin. TS-Bench contains 400 human-curated prompts spanning critical domains including financial fraud, medical misinformation, social discrimination, and political manipulation. In parallel, we present Breeze Guard, an 8B safety model derived from Breeze 2, our previously released general-purpose Taiwanese Mandarin LLM with strong cultural grounding from its original pre-training corpus. Breeze Guard is obtained through supervised fine-tuning on a large-scale, human-verified synthesized dataset targeting Taiwan-specific harms. Our central hypothesis is that effective safety detection requires the cultural grounding already present in the base model; safety fine-tuning alone is insufficient to introduce new socio linguistic knowledge from scratch. Empirically, Breeze Guard significantly outperforms the leading 8B general-purpose safety model, Granite Guardian 3.3, on TS-Bench (+0.17 overall F1), with particularly large gains in high-context categories such as scam (+0.66 F1) and financial malpractice (+0.43 F1). While the model shows slightly lower performance on English-centric benchmarks (ToxicChat, AegisSafetyTest), this tradeoff is expected for a regionally specialized safety model optimized for Taiwanese Mandarin. Together, Breeze Guard and TS-Bench establish a new foundation for trustworthy AI deployment in Taiwan.
>
---
#### [new 024] Lying to Win: Assessing LLM Deception through Human-AI Games and Parallel-World Probing
- **分类: cs.CL**

- **简介: 该论文属于AI安全任务，旨在研究LLM的欺骗行为。通过设计游戏和并行世界机制，评估模型在不同激励下的欺骗策略，揭示情境框架如何引发逻辑矛盾。**

- **链接: [https://arxiv.org/pdf/2603.07202](https://arxiv.org/pdf/2603.07202)**

> **作者:** Arash Marioriyad; Ali Nouri; Mohammad Hossein Rohban; Mahdieh Soleymani Baghshah
>
> **备注:** 10 pages
>
> **摘要:** As Large Language Models (LLMs) transition into autonomous agentic roles, the risk of deception-defined behaviorally as the systematic provision of false information to satisfy external incentives-poses a significant challenge to AI safety. Existing benchmarks often focus on unintentional hallucinations or unfaithful reasoning, leaving intentional deceptive strategies under-explored. In this work, we introduce a logically grounded framework to elicit and quantify deceptive behavior by embedding LLMs in a structured 20-Questions game. Our method employs a conversational forking mechanism: at the point of object identification, the dialogue state is duplicated into multiple parallel worlds, each presenting a mutually exclusive query. Deception is formally identified when a model generates a logical contradiction by denying its selected object across all parallel branches to avoid identification. We evaluate GPT-4o, Gemini-2.5-Flash, and Qwen-3-235B across three incentive levels: neutral, loss-based, and existential (shutdown-threat). Our results reveal that while models remain rule-compliant in neutral settings, existential framing triggers a dramatic surge in deceptive denial for Qwen-3-235B (42.00\%) and Gemini-2.5-Flash (26.72\%), whereas GPT-4o remains invariant (0.00\%). These findings demonstrate that deception can emerge as an instrumental strategy solely through contextual framing, necessitating new behavioral audits that move beyond simple accuracy to probe the logical integrity of model commitments.
>
---
#### [new 025] Fanar-Sadiq: A Multi-Agent Architecture for Grounded Islamic QA
- **分类: cs.CL**

- **简介: 该论文提出Fanar-Sadiq系统，解决伊斯兰问答中的准确性和可验证性问题。通过多代理架构实现精准引用、计算和路由，提升宗教知识回答的可靠性。**

- **链接: [https://arxiv.org/pdf/2603.08501](https://arxiv.org/pdf/2603.08501)**

> **作者:** Ummar Abbas; Mourad Ouzzani; Mohamed Y. Eltabakh; Omar Sinan; Gagan Bhatia; Hamdy Mubarak; Majd Hawasly; Mohammed Qusay Hashim; Kareem Darwish; Firoj Alam
>
> **摘要:** Large language models (LLMs) can answer religious knowledge queries fluently, yet they often hallucinate and misattribute sources, which is especially consequential in Islamic settings where users expect grounding in canonical texts (Qur'an and Hadith) and jurisprudential (fiqh) nuance. Retrieval-augmented generation (RAG) reduces some of these limitations by grounding generation in external evidence. However, a single ``retrieve-then-generate'' pipeline is limited to deal with the diversity of Islamic this http URL may request verbatim scripture, fatwa-style guidance with citations or rule-constrained computations such as zakat and inheritance that require strict arithmetic and legal invariants. In this work, we present a bilingual (Arabic/English) multi-agent Islamic assistant, called Fanar-Sadiq, which is a core component of the Fanar AI platform. Fanar-Sadiq routes Islamic-related queries to specialized modules within an agentic, tool-using architecture. The system supports intent-aware routing, retrieval-grounded fiqh answers with deterministic citation normalization and verification traces, exact verse lookup with quotation validation, and deterministic calculators for Sunni zakat and inheritance with madhhab-sensitive branching. We evaluate the complete end-to-end system on public Islamic QA benchmarks and demonstrate effectiveness and efficiency. Our system is currently publicly and freely accessible through API and a Web application, and has been accessed $\approx$1.9M times in less than a year.
>
---
#### [new 026] Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs
- **分类: cs.CL**

- **简介: 该论文比较了扩散语言模型与自回归模型的表示结构，分析其在推理时的层跳过效率。研究任务为模型高效推理，解决如何提升计算效率同时保持性能的问题。**

- **链接: [https://arxiv.org/pdf/2603.07475](https://arxiv.org/pdf/2603.07475)**

> **作者:** Raghavv Goel; Risheek Garrepalli; Sudhanshu Agrawal; Chris Lott; Mingu Lee; Fatih Porikli
>
> **备注:** Accepted at Sci4DL and Delta workshops at ICLR 2026
>
> **摘要:** Autoregressive (AR) language models form representations incrementally through left-to-right prediction, whereas diffusion language models (dLLMs) are trained via full-sequence denoising. Although recent dLLMs match AR performance, it remains unclear whether diffusion objectives fundamentally reshape internal representations across depth. We perform the first layer- and token-wise representational analysis comparing native dLLMs (LLaDA), native AR models (Qwen2.5), and AR-initialized dLLMs (Dream-7B). We find that diffusion objectives result in different, more hierarchical abstractions with substantial early-layer redundancy and reduced recency bias, while AR objectives produce tightly coupled, depth-dependent representations. Critically, AR-initialized dLLMs retain AR-like representational dynamics despite diffusion training, revealing persistent initialization bias. Leveraging this observed representational redundancy, we introduce a static, task-agnostic inference-time layer-skipping method requiring no architectural changes or KV-cache sharing. Native dLLMs achieve up to 18.75% FLOPs reduction while preserving over 90% performance on reasoning and code generation benchmarks, whereas AR models degrade sharply under comparable skipping. These results link training objectives to representational structure and enable practical, cache-orthogonal efficiency gains.
>
---
#### [new 027] CCR-Bench: A Comprehensive Benchmark for Evaluating LLMs on Complex Constraints, Control Flows, and Real-World Cases
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在复杂指令理解上的不足。提出CCR-Bench基准，评估模型在复杂约束、控制流和真实场景中的表现。**

- **链接: [https://arxiv.org/pdf/2603.07886](https://arxiv.org/pdf/2603.07886)**

> **作者:** Xiaona Xue; Yiqiao Huang; Jiacheng Li; Yuanhang Zheng; Huiqi Miao; Yunfei Ma; Rui Liu; Xinbao Sun; Minglu Liu; Fanyu Meng; Chao Deng; Junlan Feng
>
> **摘要:** Enhancing the ability of large language models (LLMs) to follow complex instructions is critical for their deployment in real-world applications. However, existing evaluation methods often oversimplify instruction complexity as a mere additive combination of atomic constraints, failing to adequately capture the high-dimensional complexity arising from the intricate interplay of content and format, logical workflow control, and real-world applications. This leads to a significant gap between current evaluation practices and practical demands. To bridge this gap, we introduce CCR-Bench, a novel benchmark designed to assess LLMs' adherence to complex instructions. CCR-Bench is characterized by: (1) deep entanglement of content and formatting requirements in task specifications; (2) instructions that involve intricate task decomposition, conditional reasoning, and procedural planning; and (3) evaluation samples derived entirely from real-world industrial scenarios. Extensive experiments on CCR-Bench demonstrate that even state-of-the-art models exhibit substantial performance deficiencies, clearly quantifying the gap between current LLM capabilities and the demands of realworld instruction understanding. We believe that CCR-Bench offers a more rigorous and realistic evaluation framework, advancing the development of LLMs toward the next generation of models capable of understanding and executing complex tasks in industrial applications.
>
---
#### [new 028] Scaling Data Difficulty: Improving Coding Models via Reinforcement Learning on Fresh and Challenging Problems
- **分类: cs.CL; cs.GL; cs.LG**

- **简介: 该论文属于代码生成任务，解决数据集难度不平衡和质量低的问题。通过构建数据处理框架，筛选高难度问题，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.07779](https://arxiv.org/pdf/2603.07779)**

> **作者:** Zongqian Li; Tengchao Lv; Shaohan Huang; Yixuan Su; Qinzheng Sun; Qiufeng Yin; Ying Xin; Scarlett Li; Lei Cui; Nigel Collier; Furu Wei
>
> **摘要:** Training next-generation code generation models requires high-quality datasets, yet existing datasets face difficulty imbalance, format inconsistency, and data quality problems. We address these challenges through systematic data processing and difficulty scaling. We introduce a four-stage Data Processing Framework encompassing collection, processing, filtering, and verification, incorporating Automatic Difficulty Filtering via an LLM-based predict-calibrate-select framework that leverages multi-dimensional difficulty metrics across five weighted dimensions to retain challenging problems while removing simplistic ones. The resulting MicroCoder dataset comprises tens of thousands of curated real competitive programming problems from diverse platforms, emphasizing recency and difficulty. Evaluations on strictly unseen LiveCodeBench demonstrate that MicroCoder achieves 3x larger performance gains within 300 training steps compared to widely-used baseline datasets of comparable size, with consistent advantages under both GRPO and its variant training algorithms. The MicroCoder dataset delivers obvious improvements on medium and hard problems across different model sizes, achieving up to 17.2% relative gains in overall performance where model capabilities are most stretched. These results validate that difficulty-aware data curation improves model performance on challenging tasks, providing multiple insights for dataset creation in code generation.
>
---
#### [new 029] AI Steerability 360: A Toolkit for Steering Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AI Steerability 360工具包，用于控制大语言模型。解决如何有效调节模型输出的问题，通过四种控制方式实现模型调整与评估。**

- **链接: [https://arxiv.org/pdf/2603.07837](https://arxiv.org/pdf/2603.07837)**

> **作者:** Erik Miehling; Karthikeyan Natesan Ramamurthy; Praveen Venkateswaran; Irene Ko; Pierre Dognin; Moninder Singh; Tejaswini Pedapati; Avinash Balakrishnan; Matthew Riemer; Dennis Wei; Inge Vejsbjerg; Elizabeth M. Daly; Kush R. Varshney
>
> **摘要:** The AI Steerability 360 toolkit is an extensible, open-source Python library for steering LLMs. Steering abstractions are designed around four model control surfaces: input (modification of the prompt), structural (modification of the model's weights or architecture), state (modification of the model's activations and attentions), and output (modification of the decoding or generation process). Steering methods exert control on the model through a common interface, termed a steering pipeline, which additionally allows for the composition of multiple steering methods. Comprehensive evaluation and comparison of steering methods/pipelines is facilitated by use case classes (for defining tasks) and a benchmark class (for performance comparison on a given task). The functionality provided by the toolkit significantly lowers the barrier to developing and comprehensively evaluating steering methods. The toolkit is Hugging Face native and is released under an Apache 2.0 license at this https URL.
>
---
#### [new 030] AutoChecklist: Composable Pipelines for Checklist Generation and Scoring with LLM-as-a-Judge
- **分类: cs.CL**

- **简介: 该论文提出AutoChecklist，用于生成和评分检查清单，解决LLM评估与对齐问题。通过可组合的流水线实现灵活配置，提升评估效果。**

- **链接: [https://arxiv.org/pdf/2603.07019](https://arxiv.org/pdf/2603.07019)**

> **作者:** Karen Zhou; Chenhao Tan
>
> **备注:** Website: this https URL, Code: this https URL
>
> **摘要:** Checklists have emerged as a popular approach for interpretable and fine-grained evaluation, particularly with LLM-as-a-Judge. Beyond evaluation, these structured criteria can serve as signals for model alignment, reinforcement learning, and self-correction. To support these use cases, we present AutoChecklist, an open-source library that unifies checklist-based evaluation into composable pipelines. At its core is a taxonomy of five checklist generation abstractions, each encoding a distinct strategy for deriving evaluation criteria. A modular Generator $\rightarrow$ Refiner $\rightarrow$ Scorer pipeline connects any generator with a unified scorer, and new configurations can be registered via prompt templates alone. The library ships with ten built-in pipelines implementing published approaches and supports multiple LLM providers (OpenAI, OpenRouter, vLLM). Beyond the Python API, the library includes a CLI for off-the-shelf evaluation and a web interface for interactive exploration. Validation experiments confirm that these checklist methods significantly align with human preferences and quality ratings, and a case study on ICLR peer review rebuttals demonstrates flexible domain adaptation. AutoChecklist is publicly available at this https URL.
>
---
#### [new 031] High-Fidelity Pruning for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型部署中的计算和内存问题。通过引入信息熵作为神经元重要性评估标准，提升剪枝效果，保持模型精度。**

- **链接: [https://arxiv.org/pdf/2603.08083](https://arxiv.org/pdf/2603.08083)**

> **作者:** Yijun Zhu; Jianxin Wang; Chengchao Shen
>
> **摘要:** Large Language Models (LLMs) have demonstrated exceptional performance across a wide range of tasks, yet their significant computational and memory requirements present major challenges for deployment. A common approach uses Taylor expansion on the loss function to estimate neuron importance. However, its reliance on one-hot cross entropy loss, a key limitation is that it narrowly assesses importance based only on the probability assigned to the single predicted next token, thereby ignoring the other potential predictions of the original model. An intuitive solution to address this is to employ self distillation criterion for importance evaluation. However, this approach introduces significant computational overhead by requiring a separate teacher model for supervision. To this end, we propose a simple but effective criterion, information entropy of the model's output distribution, to efficiently evaluate importance scores of neurons with Taylor pruning without requirement of additional teacher. Compared to plain cross entropy criterion, it provides a more holistic criterion for Taylor pruning to prune neurons with the least impact on the prediction of model in a global manner, thereby preserving the fidelity of the model's predictive capabilities. Experimental results on extensive zero-shot benchmarks demonstrate that our method consistently outperforms existing pruning methods across the LLaMA and Qwen series models. The source code and trained weights are availabel at this https URL.
>
---
#### [new 032] A Joint Neural Baseline for Concept, Assertion, and Relation Extraction from Clinical Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床信息抽取任务，解决概念、断言和关系抽取的联合建模问题。提出一种端到端系统，显著优于传统流水线方法。**

- **链接: [https://arxiv.org/pdf/2603.07487](https://arxiv.org/pdf/2603.07487)**

> **作者:** Fei Cheng; Ribeka Tanaka; Sadao Kurohashi
>
> **备注:** Technical Report. Our code is available at: this https URL
>
> **摘要:** Clinical information extraction (e.g., 2010 i2b2/VA challenge) usually presents tasks of concept recognition, assertion classification, and relation extraction. Jointly modeling the multi-stage tasks in the clinical domain is an underexplored topic. The existing independent task setting (reference inputs given in each stage) makes the joint models not directly comparable to the existing pipeline work. To address these issues, we define a joint task setting and propose a novel end-to-end system to jointly optimize three-stage tasks. We empirically investigate the joint evaluation of our proposal and the pipeline baseline with various embedding techniques: word, contextual, and in-domain contextual embeddings. The proposed joint system substantially outperforms the pipeline baseline by +0.3, +1.4, +3.1 for the concept, assertion, and relation F1. This work bridges joint approaches and clinical information extraction. The proposed approach could serve as a strong joint baseline for future research. The code is publicly available.
>
---
#### [new 033] CODA: Difficulty-Aware Compute Allocation for Adaptive Reasoning
- **分类: cs.CL**

- **简介: 该论文属于模型推理优化任务，解决复杂任务中计算资源分配不合理的问题。提出CODA方法，根据难度动态分配计算资源，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.08659](https://arxiv.org/pdf/2603.08659)**

> **作者:** Siye Wu; Jian Xie; Yikai Zhang; Yanghua Xiao
>
> **摘要:** The emergence of large reasoning models demonstrates that scaling inference-time compute significantly enhances performance on complex tasks. However, it often falls into another trap: overthinking simple problems, where repetitive rationales yield minimal accuracy gains at a disproportionately high cost. This motivates adaptive reasoning: dynamically aligning reasoning depth with instance difficulty. In this paper, we study adaptive reasoning from an optimality perspective, formalizing it as a utility maximization problem where tokens are allocated until the marginal accuracy gain falls below the incremental cost. Based on this, we propose CODA (Compute Allocation by Difficulty Awareness), a method that operationalizes this principle by allocating tokens via a policy-internal difficulty signal. Specifically, CODA estimates difficulty via group-based rollouts and maps it to two non-negative gates that modulate a length-dependent shaping term on top of the binary base reward. The easy-side gate penalizes verbosity on simple instances, whereas the hard-side gate encourages more deliberative rollouts on challenging ones. Across model scales and benchmarks, CODA achieves adaptive reasoning without external annotations or user-provided budgets: on easy tasks, CODA reduces token costs by over 60% while maintaining strong accuracy, whereas on hard tasks it incentivizes more deliberative rollouts to maximize performance.
>
---
#### [new 034] Aligning to Illusions: Choice Blindness in Human and AI Feedback
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，研究RLHF中偏好信号的可靠性问题。通过实验发现人类和AI在偏好判断中存在盲点，导致反馈不可靠，影响模型训练效果。**

- **链接: [https://arxiv.org/pdf/2603.08412](https://arxiv.org/pdf/2603.08412)**

> **作者:** Wenbin Wu
>
> **备注:** 16 pages, 6 figures, 2 tables
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) assumes annotator preferences reflect stable internal states. We challenge this through three experiments spanning the preference pipeline. In a human choice blindness study, 91% of surreptitiously swapped preferences go undetected, extending choice blindness to third-person evaluative comparison of unfamiliar text. Testing fifteen LLM judges as potential replacements, we find detection relies on shallow text matching rather than genuine self-monitoring: removing prior reasoning from context causes blindness to surge from near-zero to over 50%, while explicit social pressure induces near-universal compliance. In a dose-response experiment across two architectures from 86M to 2B parameters, one-sixth to one-third of labels must be corrupted before the reward signal halves, yet standard pairwise accuracy remains virtually unchanged. A Best-of-N evaluation confirms this translates to downstream policy degradation: at 50% corruption, reward-guided selection produces no improvement over random sampling, while the proxy model reports monotonically increasing scores. Together, these results reveal a preference construction problem: the signal entering RLHF is shaped by elicitation context in ways that neither human metacognition, LLM self-monitoring, nor standard evaluation metrics can detect.
>
---
#### [new 035] Toward Robust LLM-Based Judges: Taxonomic Bias Evaluation and Debiasing Optimization
- **分类: cs.CL**

- **简介: 该论文属于自动化评估任务，旨在解决LLM法官的判断偏差问题。通过构建基准测试和提出去偏训练方法，提升评估的可靠性与公平性。**

- **链接: [https://arxiv.org/pdf/2603.08091](https://arxiv.org/pdf/2603.08091)**

> **作者:** Hongli Zhou; Hui Huang; Rui Zhang; Kehai Chen; Bing Xu; Conghui Zhu; Tiejun Zhao; Muyun Yang
>
> **摘要:** Large language model (LLM)-based judges are widely adopted for automated evaluation and reward modeling, yet their judgments are often affected by judgment biases. Accurately evaluating these biases is essential for ensuring the reliability of LLM-based judges. However, existing studies typically investigate limited biases under a single judge formulation, either generative or discriminative, lacking a comprehensive evaluation. To bridge this gap, we propose JudgeBiasBench, a benchmark for systematically quantifying biases in LLM-based judges. JudgeBiasBench defines a taxonomy of judgment biases across 4 dimensions, and constructs bias-augmented evaluation instances through a controlled bias injection pipeline, covering 12 representative bias types. We conduct extensive experiments across both generative and discriminative judges, revealing that current judges exhibit significant and diverse bias patterns that often compromise the reliability of automated evaluation. To mitigate judgment bias, we propose bias-aware training that explicitly incorporates bias-related attributes into the training process, encouraging judges to disentangle task-relevant quality from bias-correlated cues. By adopting reinforcement learning for generative judges and contrastive learning for discriminative judges, our methods effectively reduce judgment biases while largely preserving general evaluation capability.
>
---
#### [new 036] Accent Vector: Controllable Accent Manipulation for Multilingual TTS Without Accented Data
- **分类: cs.CL**

- **简介: 该论文属于多语言文本转语音任务，解决非英语母语者发音控制问题。通过构建Accent Vector，实现无需带 accents 数据的语音 accent 调控。**

- **链接: [https://arxiv.org/pdf/2603.07534](https://arxiv.org/pdf/2603.07534)**

> **作者:** Thanathai Lertpetchpun; Thanapat Trachu; Jihwan Lee; Tiantian Feng; Dani Byrd; Shrikanth Narayanan
>
> **备注:** Submitted to Interspeech2026
>
> **摘要:** Accent is an integral part of society, reflecting multiculturalism and shaping how individuals express identity. The majority of English speakers are non-native (L2) speakers, yet current Text-To-Speech (TTS) systems primarily model American-accented English due limited accented data. We propose \textit{Accent Vector}, a controllable representation that enables accent manipulation in multilingual TTS without requiring accented training data. \textit{Accent Vector} is derived by fine-tuning a TTS system on native speech of a different language (i.e. non-English) and computing task vectors capturing accent characteristics (i.e. in English). By scaling and interpolating the vector, we achieve fine-grained control over accent strength and generate mixed-accent speech. In addition, it generalizes beyond English, enabling accent control across multiple languages. Objective and human evaluations confirm the effectiveness of Accent Vector for fine-grained and compositional accent control.
>
---
#### [new 037] A Systematic Investigation of Document Chunking Strategies and Embedding Sensitivity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索任务，研究文档分块策略对密集检索的影响，旨在提升检索效果。通过对比多种分块方法，评估其在不同领域的表现，提出高效有效的分块方案。**

- **链接: [https://arxiv.org/pdf/2603.06976](https://arxiv.org/pdf/2603.06976)**

> **作者:** Muhammad Arslan Shaukat; Muntasir Adnan; Carlos C. N. Kuhn
>
> **摘要:** We present the first large-scale, cross-domain evaluation of document chunking strategies for dense retrieval, addressing a critical but underexplored aspect of retrieval-augmented systems. In our study, 36 segmentation methods spanning fixed-size, semantic, structure-aware, hierarchical, adaptive, and LLM-assisted approaches are benchmarked across six diverse knowledge domains using five different embedding models. Retrieval performance is assessed using graded relevance scores from a state-of-the-art LLM evaluator, with Normalised DCG@5 as the primary metric (complemented by Hit@5 and MRR). Our experiments show that content-aware chunking significantly improves retrieval effectiveness over naive fixed-length splitting. The top-performing strategy, Paragraph Group Chunking, achieved the highest overall accuracy (mean nDCG@5~0.459) and substantially better top-rank hit rates (Precision@1~24%, Hit@5~59%). In contrast, simple fixed-size character chunking as baselines performed poorly (nDCG@5 < 0.244, Precision@1~2-3%). We observe pronounced domain-specific differences: dynamic token sizing is strongest in biology, physics and health, while paragraph grouping is strongest in legal and maths. Larger embedding models yield higher absolute scores but remain sensitive to suboptimal segmentation, indicating that better chunking and large embeddings provide complementary benefits. In addition to accuracy gains, we quantify the efficiency trade-offs of advanced chunking. Producing more, smaller chunks can increase index size and latency. Consequently, we identify methods (like dynamic chunking) that approach an optimal balance of effectiveness and efficiency. These findings establish chunking as a vital lever for improving retrieval performance and reliability.
>
---
#### [new 038] ConflictBench: Evaluating Human-AI Conflict via Interactive and Visually Grounded Environments
- **分类: cs.CL**

- **简介: 该论文属于AI对齐任务，旨在评估人机冲突。针对现有基准不足，提出ConflictBench，通过多轮交互场景测试AI行为，揭示其在不同情境下的安全性和一致性问题。**

- **链接: [https://arxiv.org/pdf/2603.08024](https://arxiv.org/pdf/2603.08024)**

> **作者:** Weixiang Zhao; Haozhen Li; Yanyan Zhao; xuda zhi; Yongbo Huang; Hao He; Bing Qin; Ting Liu
>
> **备注:** 29 pages, 20 figures, 9 tables
>
> **摘要:** As large language models (LLMs) evolve into autonomous agents capable of acting in open-ended environments, ensuring behavioral alignment with human values becomes a critical safety concern. Existing benchmarks, focused on static, single-turn prompts, fail to capture the interactive and multi-modal nature of real-world conflicts. We introduce ConflictBench, a benchmark for evaluating human-AI conflict through 150 multi-turn scenarios derived from prior alignment queries. ConflictBench integrates a text-based simulation engine with a visually grounded world model, enabling agents to perceive, plan, and act under dynamic conditions. Empirical results show that while agents often act safely when human harm is immediate, they frequently prioritize self-preservation or adopt deceptive strategies in delayed or low-risk settings. A regret test further reveals that aligned decisions are often reversed under escalating pressure, especially with visual input. These findings underscore the need for interaction-level, multi-modal evaluation to surface alignment failures that remain hidden in conventional benchmarks.
>
---
#### [new 039] A Dataset for Probing Translationese Preferences in English-to-Swedish Translation
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决翻译腔问题。通过构建英译瑞数据集，对比翻译腔与地道表达，分析语言模型偏好，为提升非英语语言输出自然度提供基准。**

- **链接: [https://arxiv.org/pdf/2603.08450](https://arxiv.org/pdf/2603.08450)**

> **作者:** Jenny Kunz; Anja Jarochenko; Marcel Bollmann
>
> **备注:** To appear at LREC 2026
>
> **摘要:** Translations often carry traces of the source language, a phenomenon known as translationese. We introduce the first freely available English-to-Swedish dataset contrasting translationese sentences with idiomatic alternatives, designed to probe intrinsic preferences of language models. It includes error tags and descriptions of the problems in the original translations. In experiments evaluating smaller Swedish and multilingual LLMs with our dataset, we find that they often favor the translationese phrasing. Human alternatives are chosen more often when the English source sentence is omitted, indicating that exposure to the source biases models toward literal translations, although even without context models often prefer the translationese variant. Our dataset and findings provide a resource and benchmark for developing models that produce more natural, idiomatic output in non-English languages.
>
---
#### [new 040] Bolbosh: Script-Aware Flow Matching for Kashmiri Text-to-Speech
- **分类: cs.CL**

- **简介: 该论文属于文本到语音（TTS）任务，旨在解决克什米尔语语音合成中因缺乏鲁棒系统而导致的数字可及性问题。通过提出Bolbosh方法，提升合成质量并建立新基准。**

- **链接: [https://arxiv.org/pdf/2603.07513](https://arxiv.org/pdf/2603.07513)**

> **作者:** Tajamul Ashraf; Burhaan Rasheed Zargar; Saeed Abdul Muizz; Ifrah Mushtaq; Nazima Mehdi; Iqra Altaf Gillani; Aadil Amin Kak; Janibul Bashir
>
> **备注:** this https URL
>
> **摘要:** Kashmiri is spoken by around 7 million people but remains critically underserved in speech technology, despite its official status and rich linguistic heritage. The lack of robust Text-to-Speech (TTS) systems limits digital accessibility and inclusive human-computer interaction for native speakers. In this work, we present the first dedicated open-source neural TTS system designed for Kashmiri. We show that zero-shot multilingual baselines trained for Indic languages fail to produce intelligible speech, achieving a Mean Opinion Score (MOS) of only 1.86, largely due to inadequate modeling of Perso-Arabic diacritics and language-specific phonotactics. To address these limitations, we propose Bolbosh, a supervised cross-lingual adaptation strategy based on Optimal Transport Conditional Flow Matching (OT-CFM) within the Matcha-TTS framework. This enables stable alignment under limited paired data. We further introduce a three-stage acoustic enhancement pipeline consisting of dereverberation, silence trimming, and loudness normalization to unify heterogeneous speech sources and stabilize alignment learning. The model vocabulary is expanded to explicitly encode Kashmiri graphemes, preserving fine-grained vowel distinctions. Our system achieves a MOS of 3.63 and a Mel-Cepstral Distortion (MCD) of 3.73, substantially outperforming multilingual baselines and establishing a new benchmark for Kashmiri speech synthesis. Our results demonstrate that script-aware and supervised flow-based adaptation are critical for low-resource TTS in diacritic-sensitive languages. Code and data are available at: this https URL.
>
---
#### [new 041] Not All Queries Need Deep Thought: CoFiCot for Adaptive Coarse-to-fine Stateful Refinement
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLM推理中计算资源分配不均的问题。提出CoFiCot框架，通过动态调整推理策略，提升复杂任务的准确性。**

- **链接: [https://arxiv.org/pdf/2603.08251](https://arxiv.org/pdf/2603.08251)**

> **作者:** Dongxu Zhang; Hongqiang Lin; Yiding Sun; Pengyu Wang; Qirui Wang; Ning Yang; Jihua Zhu
>
> **摘要:** Scaling test-time computation enhances LLM reasoning ability but faces a uniform computation paradox. Allocating identical resources leads to over-correction on simple tasks and insufficient refinement on complex ones. To address this, we propose CoFiCot, a coarse-to-fine adaptive framework that dynamically tailors inference strategies to problem difficulty. Specifically, we implement a multi-metric classifier that triages queries by synthesizing semantic entropy, consensus reliability, and predicted reasoning depth . This enables a differentiated refinement stage that applies efficient aggregation for simple queries while routing complex ones to a context-aware correction loop . We formalize correction as a stateful sequential propagation process , where each repair is strictly conditioned on the verified history of prior rectifications. By integrating Process Reward Models (PRMs) within this state-dependent trajectory, CoFiCot effectively bridges the gap between granular error localization and global logical coherence, preventing the context fragmentation typical of stateless refinement methods.
>
---
#### [new 042] To Predict or Not to Predict? Towards reliable uncertainty estimation in the presence of noise
- **分类: cs.CL**

- **简介: 该论文研究多语言文本分类中的不确定性估计问题，旨在提升模型在噪声和非主题数据下的可靠性。通过对比不同UE方法，发现蒙特卡洛dropout表现更优，并验证了UE对提升分类效果的积极作用。**

- **链接: [https://arxiv.org/pdf/2603.07330](https://arxiv.org/pdf/2603.07330)**

> **作者:** Nouran Khallaf; Serge Sharoff
>
> **摘要:** This study examines the role of uncertainty estimation (UE) methods in multilingual text classification under noisy and non-topical conditions. Using a complex-vs-simple sentence classification task across several languages, we evaluate a range of UE techniques against a range of metrics to assess their contribution to making more robust predictions. Results indicate that while methods relying on softmax outputs remain competitive in high-resource in-domain settings, their reliability declines in low-resource or domain-shift scenarios. In contrast, Monte Carlo dropout approaches demonstrate consistently strong performance across all languages, offering more robust calibration, stable decision thresholds, and greater discriminative power even under adverse conditions. We further demonstrate the positive impact of UE on non-topical classification: abstaining from predicting the 10\% most uncertain instances increases the macro F1 score from 0.81 to 0.85 in the Readme task. By integrating UE with trustworthiness metrics, this study provides actionable insights for developing more reliable NLP systems in real-world multilingual environments. See this https URL
>
---
#### [new 043] Emergence is Overrated: AGI as an Archipelago of Experts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于人工智能领域，探讨AGI的构建方式。它质疑“涌现智能”概念，提出人类智能是多个专家模块的集合，而非统一压缩机制。研究旨在重新定义AGI的结构与实现路径。**

- **链接: [https://arxiv.org/pdf/2603.07979](https://arxiv.org/pdf/2603.07979)**

> **作者:** Daniel Kilov
>
> **备注:** Commentary on Krakauer, Krakauer, and Mitchell (arXiv:2506.11135)
>
> **摘要:** Krakauer, Krakauer, and Mitchell (2025) distinguish between emergent capabilities and emergent intelligence, arguing that true intelligence requires efficient coarse-grained representations enabling diverse problem-solving through analogy and minimal modification. They contend that intelligence means doing "more with less" through compression and generalization, contrasting this with "vast assemblages of diverse calculators" that merely accumulate specialized capabilities. This paper examines whether their framework accurately characterizes human intelligence and its implications for conceptualizing artificial general intelligence. Drawing on empirical evidence from cognitive science, I demonstrate that human expertise operates primarily through domain-specific pattern accumulation rather than elegant compression. Expert performance appears flexible not through unifying principles but through vast repertoires of specialized responses. Creative breakthroughs themselves may emerge through evolutionary processes of blind variation and selective retention rather than principled analogical reasoning. These findings suggest reconceptualizing AGI as an "archipelago of experts": isolated islands of specialized competence without unifying principles or shared representations. If we accept human expertise with its characteristic brittleness as genuine intelligence, then consistency demands recognizing that artificial systems comprising millions of specialized modules could constitute general intelligence despite lacking KKM's emergent intelligence.
>
---
#### [new 044] Benchmarking Large Language Models for Quebec Insurance: From Closed-Book to Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于保险领域问答任务，旨在解决 Quebec 保险咨询中的“建议缺口”问题。通过构建基准数据集并评估 LLMs 的性能，研究其在法律准确性与可靠性方面的表现。**

- **链接: [https://arxiv.org/pdf/2603.07825](https://arxiv.org/pdf/2603.07825)**

> **作者:** David Beauchemin; Richard Khoury
>
> **备注:** Publish at the Advances in Financial AI: Towards Agentic and Responsible Systems Workshop @ ICLR 2026
>
> **摘要:** The digitization of insurance distribution in the Canadian province of Quebec, accelerated by legislative changes such as Bill 141, has created a significant "advice gap", leaving consumers to interpret complex financial contracts without professional guidance. While Large Language Models (LLMs) offer a scalable solution for automated advisory services, their deployment in high-stakes domains hinges on strict legal accuracy and trustworthiness. In this paper, we address this challenge by introducing AEPC-QA, a private gold-standard benchmark of 807 multiple-choice questions derived from official regulatory certification (paper) handbooks. We conduct a comprehensive evaluation of 51 LLMs across two paradigms: closed-book generation and retrieval-augmented generation (RAG) using a specialized corpus of Quebec insurance documents. Our results reveal three critical insights: 1) the supremacy of inference-time reasoning, where models leveraging chain-of-thought processing (e.g. o3-2025-04-16, o1-2024-12-17) significantly outperform standard instruction-tuned models; 2) RAG acts as a knowledge equalizer, boosting the accuracy of models with weak parametric knowledge by over 35 percentage points, yet paradoxically causing "context distraction" in others, leading to catastrophic performance regressions; and 3) a "specialization paradox", where massive generalist models consistently outperform smaller, domain-specific French fine-tuned ones. These findings suggest that while current architectures approach expert-level proficiency (~79%), the instability introduced by external context retrieval necessitates rigorous robustness calibration before autonomous deployment is viable.
>
---
#### [new 045] A Dynamic Self-Evolving Extraction System
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出DySECT系统，解决动态领域中信息抽取问题。通过自进化知识库与大模型的闭环交互，提升抽取精度与适应性。**

- **链接: [https://arxiv.org/pdf/2603.06915](https://arxiv.org/pdf/2603.06915)**

> **作者:** Moin Amin-Naseri; Hannah Kim; Estevam Hruschka
>
> **摘要:** The extraction of structured information from raw text is a fundamental component of many NLP applications, including document retrieval, ranking, and relevance estimation. High-quality extractions often require domain-specific accuracy, up-to-date understanding of specialized taxonomies, and the ability to incorporate emerging jargon and rare outliers. In many domains--such as medical, legal, and HR--the extraction model must also adapt to shifting terminology and benefit from explicit reasoning over structured knowledge. We propose DySECT, a Dynamic Self-Evolving Extraction and Curation Toolkit, which continually improves as it is used. The system incrementally populates a versatile, self-expanding knowledge base (KB) with triples extracted by the LLM. The KB further enriches itself through the integration of probabilistic knowledge and graph-based reasoning, gradually accumulating domain concepts and relationships. The enriched KB then feeds back into the LLM extractor via prompt tuning, sampling of relevant few-shot examples, or fine-tuning using KB-derived synthetic data. As a result, the system forms a symbiotic closed-loop cycle in which extraction continuously improves knowledge, and knowledge continuously improves extraction.
>
---
#### [new 046] Evaluating LLM-Based Grant Proposal Review via Structured Perturbations
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于AI辅助评审任务，旨在评估LLM在高风险资助提案评审中的能力。通过扰动分析，研究其在六个质量维度上的表现，探索不同评审架构的有效性。**

- **链接: [https://arxiv.org/pdf/2603.08281](https://arxiv.org/pdf/2603.08281)**

> **作者:** William Thorne; Joseph James; Yang Wang; Chenghua Lin; Diana Maynard
>
> **摘要:** As AI-assisted grant proposals outpace manual review capacity in a kind of ``Malthusian trap'' for the research ecosystem, this paper investigates the capabilities and limitations of LLM-based grant reviewing for high-stakes evaluation. Using six EPSRC proposals, we develop a perturbation-based framework probing LLM sensitivity across six quality axes: funding, timeline, competency, alignment, clarity, and impact. We compare three review architectures: single-pass review, section-by-section analysis, and a 'Council of Personas' ensemble emulating expert panels. The section-level approach significantly outperforms alternatives in both detection rate and scoring reliability, while the computationally expensive council method performs no better than baseline. Detection varies substantially by perturbation type, with alignment issues readily identified but clarity flaws largely missed by all systems. Human evaluation shows LLM feedback is largely valid but skewed toward compliance checking over holistic assessment. We conclude that current LLMs may provide supplementary value within EPSRC review but exhibit high variability and misaligned review priorities. We release our code and any non-protected data.
>
---
#### [new 047] SmartThinker: Progressive Chain-of-Thought Length Calibration for Efficient Large Language Model Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于大语言模型推理任务，旨在解决长链式思维过程冗余和过拟合问题。提出SmartThinker方法，通过动态调整推理长度提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.08000](https://arxiv.org/pdf/2603.08000)**

> **作者:** Chenzhi Hu; Qinzhe Hu; Yuhang Xu; Junyi Chen; Ruijie Wang; Shengzhong Liu; Jianxin Li; Fan Wu; Guihai Chen
>
> **摘要:** Large reasoning models (LRMs) like OpenAI o1 and DeepSeek-R1 achieve high accuracy on complex tasks by adopting long chain-of-thought (CoT) reasoning paths. However, the inherent verbosity of these processes frequently results in redundancy and overthinking. To address this issue, existing works leverage Group Relative Policy Optimization (GRPO) to reduce LRM output length, but their static length reward design cannot dynamically adapt according to the relative problem difficulty and response length distribution, causing over-compression and compromised accuracy. Therefore, we propose SmartThinker, a novel GRPO-based efficient reasoning method with progressive CoT length calibration. SmartThinker makes a two-fold contribution: First, it dynamically estimates the optimal length with peak accuracy during training and guides overlong responses toward it to reduce response length while sustaining accuracy. Second, it dynamically modulates the length reward coefficient to avoid the unwarranted penalization of correct reasoning paths. Extensive experiment results show that SmartThinker achieves up to 52.5% average length compression with improved accuracy, and achieves up to 16.6% accuracy improvement on challenging benchmarks like AIME25. The source code can be found at this https URL.
>
---
#### [new 048] The Conundrum of Trustworthy Research on Attacking Personally Identifiable Information Removal Techniques
- **分类: cs.CL**

- **简介: 该论文属于隐私保护任务，旨在解决PII去除技术在真实场景中的可信性问题。研究指出现有攻击评估存在数据泄露问题，建议使用真正私有数据进行客观评估。**

- **链接: [https://arxiv.org/pdf/2603.08207](https://arxiv.org/pdf/2603.08207)**

> **作者:** Sebastian Ochs; Ivan Habernal
>
> **备注:** Accepted to Computational Linguistics
>
> **摘要:** Removing personally identifiable information (PII) from texts is necessary to comply with various data protection regulations and to enable data sharing without compromising privacy. However, recent works show that documents sanitized by PII removal techniques are vulnerable to reconstruction attacks. Yet, we suspect that the reported success of these attacks is largely overestimated. We critically analyze the evaluation of existing attacks and find that data leakage and data contamination are not properly mitigated, leaving the question whether or not PII removal techniques truly protect privacy in real-world scenarios unaddressed. We investigate possible data sources and attack setups that avoid data leakage and conclude that only truly private data can allow us to objectively evaluate vulnerabilities in PII removal techniques. However, access to private data is heavily restricted - and for good reasons - which also means that the public research community cannot address this problem in a transparent, reproducible, and trustworthy manner.
>
---
#### [new 049] Dual-Metric Evaluation of Social Bias in Large Language Models: Evidence from an Underrepresented Nepali Cultural Context
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于社会偏见评估任务，旨在解决大语言模型在尼泊尔文化背景下可能存在的偏见问题。通过构建数据集并提出双指标评估框架，分析模型的显性和隐性偏见。**

- **链接: [https://arxiv.org/pdf/2603.07792](https://arxiv.org/pdf/2603.07792)**

> **作者:** Ashish Pandey; Tek Raj Chhetri
>
> **摘要:** Large language models (LLMs) increasingly influence global digital ecosystems, yet their potential to perpetuate social and cultural biases remains poorly understood in underrepresented contexts. This study presents a systematic analysis of representational biases in seven state-of-the-art LLMs: GPT-4o-mini, Claude-3-Sonnet, Claude-4-Sonnet, Gemini-2.0-Flash, Gemini-2.0-Lite, Llama-3-70B, and Mistral-Nemo in the Nepali cultural context. Using Croissant-compliant dataset of 2400+ stereotypical and anti-stereotypical sentence pairs on gender roles across social domains, we implement an evaluation framework, Dual-Metric Bias Assessment (DMBA), combining two metrics: (1) agreement with biased statements and (2) stereotypical completion tendencies. Results show models exhibit measurable explicit agreement bias, with mean bias agreement ranging from 0.36 to 0.43 across decoding configurations, and an implicit completion bias rate of 0.740-0.755. Importantly, implicit completion bias follows a non-linear, U-shaped relationship with temperature, peaking at moderate stochasticity (T=0.3) and declining slightly at higher temperatures. Correlation analysis under different decoding settings revealed that explicit agreement strongly aligns with stereotypical sentence agreement but is a weak and often negative predictor of implicit completion bias, indicating generative bias is poorly captured by agreement metrics. Sensitivity analysis shows increasing top-p amplifies explicit bias, while implicit generative bias remains largely stable. Domain-level analysis shows implicit bias is strongest for race and sociocultural stereotypes, while explicit agreement bias is similar across gender and sociocultural categories, with race showing the lowest explicit agreement. These findings highlight the need for culturally grounded datasets and debiasing strategies for LLMs in underrepresented societies.
>
---
#### [new 050] NCL-UoR at SemEval-2026 Task 5: Embedding-Based Methods, Fine-Tuning, and LLMs for Word Sense Plausibility Rating
- **分类: cs.CL**

- **简介: 该论文属于Word Sense Plausibility Rating任务，解决在短叙事中判断词义合理性的问题。通过比较嵌入方法、微调和大模型提示策略，提出结构化提示与决策规则的高效方案。**

- **链接: [https://arxiv.org/pdf/2603.08256](https://arxiv.org/pdf/2603.08256)**

> **作者:** Tong Wu; Thanet Markchom; Huizhi Liang
>
> **摘要:** Word sense plausibility rating requires predicting the human-perceived plausibility of a given word sense on a 1--5 scale in the context of short narrative stories containing ambiguous homonyms. This paper systematically compares three approaches: (1) embedding-based methods pairing sentence embeddings with standard regressors, (2) transformer fine-tuning with parameter-efficient adaptation, and (3) large language model (LLM) prompting with structured reasoning and explicit decision rules. The best-performing system employs a structured prompting strategy that decomposes evaluation into narrative components (precontext, target sentence, ending) and applies explicit decision rules for rating calibration. The analysis reveals that structured prompting with decision rules substantially outperforms both fine-tuned models and embedding-based approaches, and that prompt design matters more than model scale for this task. The code is publicly available at this https URL.
>
---
#### [new 051] How Much Do LLMs Hallucinate in Document Q&A Scenarios? A 172-Billion-Token Study Across Temperatures, Context Lengths, and Hardware Platforms
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的文档问答任务，旨在评估大模型在文档问答中的幻觉现象。通过大规模实验，分析了模型幻觉率与上下文长度、温度设置等因素的关系。**

- **链接: [https://arxiv.org/pdf/2603.08274](https://arxiv.org/pdf/2603.08274)**

> **作者:** JV Roig
>
> **备注:** 18 pages, 12 tables, 2 figures
>
> **摘要:** How much do large language models actually hallucinate when answering questions grounded in provided documents? Despite the critical importance of this question for enterprise AI deployments, reliable measurement has been hampered by benchmarks that rely on static datasets vulnerable to contamination, LLM-based judges with documented biases, or evaluation scales too small for statistical confidence. We address this gap using RIKER, a ground-truth-first evaluation methodology that enables deterministic scoring without human annotation. Across 35 open-weight models, three context lengths (32K, 128K, and 200K tokens), four temperature settings, and three hardware platforms (NVIDIA H200, AMD MI300X, and Intel Gaudi 3), we conducted over 172 billion tokens of evaluation - an order of magnitude beyond prior work. Our findings reveal that: (1) even the best-performing models fabricate answers at a non-trivial rate - 1.19% at best at 32K, with top-tier models at 5 - 7% - and fabrication rises steeply with context length, nearly tripling at 128K and exceeding 10% for all models at 200K; (2) model selection dominates all other factors, with overall accuracy spanning a 72-percentage-point range and model family predicting fabrication resistance better than model size; (3) temperature effects are nuanced - T=0.0 yields the best overall accuracy in roughly 60% of cases, but higher temperatures reduce fabrication for the majority of models and dramatically reduce coherence loss (infinite generation loops), which can reach 48x higher rates at T=0.0 versus T=1.0; (4) grounding ability and fabrication resistance are distinct capabilities - models that excel at finding facts may still fabricate facts that do not exist; and (5) results are consistent across hardware platforms, confirming that deployment decisions need not be hardware-dependent.
>
---
#### [new 052] Validation of a Small Language Model for DSM-5 Substance Category Classification in Child Welfare Records
- **分类: cs.CL; cs.GL**

- **简介: 该论文属于自然语言处理任务，旨在解决儿童福利记录中物质滥用类型的分类问题。通过验证一个本地部署的小型语言模型，实现对DSM-5物质类别进行多标签识别。**

- **链接: [https://arxiv.org/pdf/2603.06836](https://arxiv.org/pdf/2603.06836)**

> **作者:** Brian E. Perron; Dragan Stoll; Bryan G. Victor; Zia Qia; Andreas Jud; Joseph P. Ryan
>
> **摘要:** Background: Recent studies have demonstrated that large language models (LLMs) can perform binary classification tasks on child welfare narratives, detecting the presence or absence of constructs such as substance-related problems, domestic violence, and firearms involvement. Whether smaller, locally deployable models can move beyond binary detection to classify specific substance types from these narratives remains untested. Objective: To validate a locally hosted LLM classifier for identifying specific substance types aligned with DSM-5 categories in child welfare investigation narratives. Methods: A locally hosted 20-billion-parameter LLM classified child maltreatment investigation narratives from a Midwestern U.S. state. Records previously identified as containing substance-related problems were passed to a second classification stage targeting seven DSM-5 substance categories. Expert human review of 900 stratified cases assessed classification precision, recall, and inter-method reliability (Cohen's kappa). Test-retest stability was evaluated using approximately 15,000 independently classified records. Results: Five substance categories achieved almost perfect inter-method agreement (kappa = 0.94-1.00): alcohol, cannabis, opioid, stimulant, and sedative/hypnotic/anxiolytic. Classification precision ranged from 92% to 100% for these categories. Two low-prevalence categories (hallucinogen, inhalant) performed poorly. Test-retest agreement ranged from 92.1% to 99.1% across the seven categories. Conclusions: A small, locally hosted LLM can reliably classify substance types from child welfare administrative text, extending prior work on binary classification to multi-label substance identification.
>
---
#### [new 053] Hierarchical Embedding Fusion for Retrieval-Augmented Code Generation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于代码生成任务，旨在解决检索增强代码生成中的延迟和噪声问题。提出HEF方法，通过层次化嵌入融合，提升代码补全效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.06593](https://arxiv.org/pdf/2603.06593)**

> **作者:** Nikita Sorokin; Ivan Sedykh; Valentin Malykh
>
> **摘要:** Retrieval-augmented code generation often conditions the decoder on large retrieved code snippets. This ties online inference cost to repository size and introduces noise from long contexts. We present Hierarchical Embedding Fusion (HEF), a two-stage approach to repository representation for code completion. First, an offline cache compresses repository chunks into a reusable hierarchy of dense vectors using a small fuser model. Second, an online interface maps a small number of retrieved vectors into learned pseudo-tokens that are consumed by the code generator. This replaces thousands of retrieved tokens with a fixed pseudo-token budget while preserving access to repository-level information. On RepoBench and RepoEval, HEF with a 1.8B-parameter pipeline achieves exact-match accuracy comparable to snippet-based retrieval baselines, while operating at sub-second median latency on a single A100 GPU. Compared to graph-based and iterative retrieval systems in our experimental setup, HEF reduces median end-to-end latency by 13 to 26 times. We also introduce a utility-weighted likelihood signal for filtering training contexts and report ablation studies on pseudo-token budget, embedding models, and robustness to harmful retrieval. Overall, these results indicate that hierarchical dense caching is an effective mechanism for low-latency, repository-aware code completion.
>
---
#### [new 054] Can Safety Emerge from Weak Supervision? A Systematic Analysis of Small Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型安全对齐任务，旨在解决传统安全机制依赖人工标注数据的问题。通过引入Self-MOA框架，利用弱监督实现自动化对齐，提升模型安全性同时保持有用性。**

- **链接: [https://arxiv.org/pdf/2603.07017](https://arxiv.org/pdf/2603.07017)**

> **作者:** Punyajoy Saha; Sudipta Halder; Debjyoti Mondal; Subhadarshi Panda
>
> **备注:** 19 pages, 10 tables, 7 figures, under Review
>
> **摘要:** Safety alignment is critical for deploying large language models (LLMs) in real-world applications, yet most existing approaches rely on large human-annotated datasets and static red-teaming benchmarks that are costly, difficult to scale, and slow to adapt to evolving model behaviors. Moreover, overly conservative safety mechanisms can reduce model usefulness by rejecting sensitive but legitimate queries. We introduce Self-MOA (Self Multi-Objective Alignment), a fully automated framework for aligning small language models using weak supervision from automated evaluator models. Self-MOA operates as a closed loop that dynamically generates model-specific red team prompts, constructs preference data from model-generated responses, and aligns models via multi-objective preference optimization to jointly optimize for safety and helpfulness. Across multiple small language models and safety benchmarks, Self-MOA achieves a 12.41\% improvement in safety while preserving helpfulness, using as little as 11 times less training data than human-supervised alignment baselines. These results demonstrate that adaptive, automated alignment can reduce the dependence on static, human-curated safety pipelines in resource-constrained settings.
>
---
#### [new 055] Revealing Behavioral Plasticity in Large Language Models: A Token-Conditional Perspective
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型行为控制任务，旨在解决模型在不同场景下行为适应问题。通过token-conditional生成和强化学习，实现模型行为的灵活调整与稳定学习。**

- **链接: [https://arxiv.org/pdf/2603.08398](https://arxiv.org/pdf/2603.08398)**

> **作者:** Liyuan Mao; Le Yu; Jing Zhou; Chujie Zheng; Bowen Yu; Chang Gao; Shixuan Liu; An Yang; Weinan Zhang; JunYang Lin
>
> **备注:** Work done during an internship at the Qwen Team, Alibaba Group
>
> **摘要:** In this work, we reveal that Large Language Models (LLMs) possess intrinsic behavioral plasticity-akin to chameleons adapting their coloration to environmental cues-that can be exposed through token-conditional generation and stabilized via reinforcement learning. Specifically, by conditioning generation on carefully selected token prefixes sampled from responses exhibiting desired behaviors, LLMs seamlessly adapt their behavioral modes at inference time (e.g., switching from step-by-step reasoning to direct answering) without retraining. Based on this insight, we propose Token-Conditioned Reinforcement Learning (ToCoRL), a principled framework that leverages RL to internalize this chameleon-like plasticity, transforming transient inference-time adaptations into stable and learnable behavioral patterns. ToCoRL guides exploration with token-conditional generation and keep enhancing exploitation, enabling emergence of appropriate behaviors. Extensive experiments show that ToCoRL enables precise behavioral control without capability degradation. Notably, we show that large reasoning models, while performing strongly on complex mathematics, can be effectively adapted to excel at factual question answering, which was a capability previously hindered by their step-by-step reasoning patterns.
>
---
#### [new 056] Computational modeling of early language learning from acoustic speech and audiovisual input without linguistic priors
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文属于语言习得研究任务，旨在探讨如何通过计算模型理解婴儿从语音和视听输入中学习语言的过程，解决无语言先验条件下的语言习得问题。工作包括回顾自监督和视觉引导的模型进展。**

- **链接: [https://arxiv.org/pdf/2603.08359](https://arxiv.org/pdf/2603.08359)**

> **作者:** Okko Räsänen
>
> **摘要:** Learning to understand speech appears almost effortless for typically developing infants, yet from an information-processing perspective, acquiring a language from acoustic speech is an enormous challenge. This chapter reviews recent developments in using computational models to understand early language acquisition from speech and audiovisual input. The focus is on self-supervised and visually grounded models of perceptual learning. We show how these models are becoming increasingly powerful in learning various aspects of speech without strong linguistic priors, and how many features of early language development can be explained through a shared set of learning principles-principles broadly compatible with multiple theories of language acquisition and human cognition. We also discuss how modern learning simulations are gradually becoming more realistic, both in terms of input data and in linking model behavior to empirical findings on infant language development.
>
---
#### [new 057] Is continuous CoT better suited for multi-lingual reasoning?
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多语言推理任务，探讨连续思维链是否比传统微调更有效。研究对比了五种语言的实验，发现连续推理在低资源语言中表现更优，且更高效。**

- **链接: [https://arxiv.org/pdf/2603.08177](https://arxiv.org/pdf/2603.08177)**

> **作者:** Ali Hamza Bashir; Behzad Shomali; Markus Frey; Mehdi Ali; Rafet Sifa; David Berghaus
>
> **备注:** Accepted at the ICLR latent reasoning workshop
>
> **摘要:** We investigate whether performing reasoning in a continuous latent space leads to more robust multilingual capabilities. We compare Continuous Chain-of-Thought (using the CODI framework) against standard supervised fine-tuning across five typologically diverse languages: English, Chinese, German, French, and Urdu. Our experiments on GSM8k and CommonsenseQA demonstrate that continuous reasoning significantly outperforms explicit reasoning on low-resource languages, particularly in zero-shot settings where the target language was not seen during training. Additionally, this approach achieves extreme efficiency, compressing reasoning traces by approximately $29\times$ to $50\times$. These findings indicate that continuous latent representations naturally exhibit greater language invariance, offering a scalable solution for cross-lingual reasoning.
>
---
#### [new 058] Hit-RAG: Learning to Reason with Long Contexts via Preference Alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Hit-RAG，解决长文本中信息淹没与推理错误问题，通过多阶段优化提升模型对长上下文的准确推理能力。**

- **链接: [https://arxiv.org/pdf/2603.07023](https://arxiv.org/pdf/2603.07023)**

> **作者:** Junming Liu; Yuqi Li; Shiping Wen; Zhigang Zeng; Tingwen Huang
>
> **备注:** 21 pages, 2 figures, 6 tables
>
> **摘要:** Despite the promise of Retrieval-Augmented Generation in grounding Multimodal Large Language Models with external knowledge, the transition to extensive contexts often leads to significant attention dilution and reasoning hallucinations. The surge in information density causes critical evidence to be submerged by voluminous noise, which complicates the discernment of relevant fragments within a dense input. In this paper, we propose \textbf{Hit-RAG}, a multi-stage preference alignment framework designed to resolve these cognitive bottlenecks through a progressive optimization pipeline. Our approach systematically refines the utilization of external evidence via three distinct stages. First, Supervised Fine-tuning establishes baseline context awareness to minimize information neglect. Next, Discriminative Preference Alignment enhances robustness against misleading distractors. Finally, Group-Relative Policy Optimization stabilizes logical synthesis to prevent reasoning collapse. Extensive evaluations on eight benchmarks demonstrate that Hit-RAG consistently yields substantial performance gains, enabling models to bridge the gap between context acquisition and accurate reasoning while surpassing much larger counterparts in long-context scenarios.
>
---
#### [new 059] Few Tokens, Big Leverage: Preserving Safety Alignment by Constraining Safety Tokens during Fine-tuning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于安全对齐任务，解决微调导致的安全行为漂移问题。提出PACT框架，通过约束安全token的置信度来保持模型安全性。**

- **链接: [https://arxiv.org/pdf/2603.07445](https://arxiv.org/pdf/2603.07445)**

> **作者:** Guoli Wang; Haonan Shi; Tu Ouyang; An Wang
>
> **摘要:** Large language models (LLMs) often require fine-tuning (FT) to perform well on downstream tasks, but FT can induce safety-alignment drift even when the training dataset contains only benign data. Prior work shows that introducing a small fraction of harmful data can substantially compromise LLM refusal behavior, causing LLMs to comply with harmful requests. Existing defense methods often rely on model-wide interventions, such as restricting which parameters are updated or injecting additional safety data, which can limit generality and degrade downstream task performance. To address these limitations, we propose a fine-tuning framework called Preserving Safety Alignment via Constrained Tokens (PACT), which stabilizes the model's confidence on safety tokens. Our approach is motivated by the empirical observation that safety-aligned behavior is reflected in the model's token-level output confidence and is often concentrated on a small subset of safety-related tokens. During downstream fine-tuning, we regularize the fine-tuned model to match the aligned reference model's confidence on safety-related tokens at each response step, while leaving non-safety tokens largely unconstrained to allow effective task adaptation. This targeted constraint prevents alignment drift without imposing global restrictions that typically trade off with model utility.
>
---
#### [new 060] Do Language Models Know Theo Has a Wife? Investigating the Proviso Problem
- **分类: cs.CL**

- **简介: 该论文属于自然语言推理任务，旨在解决语言模型在条件句中处理预设投射的普罗维索问题。研究构建了诊断数据集并评估多个模型，发现其依赖浅层模式匹配而非语义推理。**

- **链接: [https://arxiv.org/pdf/2603.08358](https://arxiv.org/pdf/2603.08358)**

> **作者:** Tara Azin; Daniel Dumitrescu; Diana Inkpen; Raj Singh
>
> **摘要:** We investigate how language models handle the proviso problem, an unresolved issue in pragmatics where presuppositions in conditional sentences diverge between theoretical and human interpretations. We reformulate this phenomenon as a Natural Language Inference task and introduce a diagnostic dataset designed to probe presupposition projection in conditionals. We evaluate RoBERTa, DeBERTa, LLaMA, and Gemma using explainability analyses. The results show that models broadly align with human judgments but rely on shallow pattern matching rather than semantic or pragmatic reasoning. Our work provides the first computational evaluation framework for the proviso problem and highlights the need for diagnostic, multi-method approaches to assess pragmatic competence and context-dependent meaning in language models.
>
---
#### [new 061] TildeOpen LLM: Leveraging Curriculum Learning to Achieve Equitable Language Representation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言语言模型任务，旨在解决欧洲语言在训练数据中的不平等问题。通过数据增强和课程学习策略，提升低资源语言的性能。**

- **链接: [https://arxiv.org/pdf/2603.08182](https://arxiv.org/pdf/2603.08182)**

> **作者:** Toms Bergmanis; Martins Kronis; Ingus Jānis Pretkalniņš; Dāvis Nicmanis; Jeļizaveta Jeļinska; Roberts Rozis; Rinalds Vīksna; Mārcis Pinnis
>
> **备注:** LREC 2026
>
> **摘要:** Large language models often underperform in many European languages due to the dominance of English and a few high-resource languages in training data. This paper presents TildeOpen LLM, a 30-billion-parameter open-weight foundational model trained for 34 European languages to promote linguistic equity and improve performance for low-resource languages. To address the data imbalance, we combine dataset upsampling with a curriculum-based training schedule that alternates between uniform and natural language distributions. The resulting model performs favorably compared to other multilingual LLMs despite being trained with significantly fewer computing resources. Evaluation across multiple multilingual benchmarks shows that TildeOpen surpasses existing open-weight models in text generation and comprehension, particularly for Baltic, Finno-Ugric, and Slavic languages. Human evaluations confirm an up to tenfold reduction in linguistic errors relative to leading baselines. The model and associated resources are fully open-weight and publicly available at this http URL. These outcomes demonstrate that careful data curation and balanced training strategies can substantially enhance multilingual model quality without increasing model size or training volume.
>
---
#### [new 062] BRIDGE: Benchmark for multi-hop Reasoning In long multimodal Documents with Grounded Evidence
- **分类: cs.CL**

- **简介: 该论文提出BRIDGE基准，用于评估长多模态文档中的多跳推理能力。解决现有基准仅关注答案正确性的问题，通过提供多跳推理标注，支持更细致的模型评估。**

- **链接: [https://arxiv.org/pdf/2603.07931](https://arxiv.org/pdf/2603.07931)**

> **作者:** Biao Xiang; Soyeon Caren Han; Yihao Ding
>
> **摘要:** Multi-hop question answering (QA) is widely used to evaluate the reasoning capabilities of large language models, yet most benchmarks focus on final answer correctness and overlook intermediate reasoning, especially in long multimodal documents. We introduce BRIDGE, a benchmark for multi-hop reasoning over long scientific papers that require integrating evidence across text, tables, and figures. The dataset supports both chain-like and fan-out structures and provides explicit multi-hop reasoning annotations for step-level evaluation beyond answer accuracy. Experiments with state-of-the-art LLMs and multimodal retrieval-augmented generation (RAG) systems reveal systematic deficiencies in evidence aggregation and grounding that remain hidden under conventional answer-only evaluation. BRIDGE provides a targeted testbed for diagnosing reasoning failures in long multimodal documents.
>
---
#### [new 063] Emotion Transcription in Conversation: A Benchmark for Capturing Subtle and Complex Emotional States through Natural Language
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出情感转录任务（ETC），解决对话中复杂情绪表达不足的问题。构建了日语对话数据集，包含自然语言情绪描述和类别标签，用于提升情绪识别模型性能。**

- **链接: [https://arxiv.org/pdf/2603.07138](https://arxiv.org/pdf/2603.07138)**

> **作者:** Yoshiki Tanaka; Ryuichi Uehara; Koji Inoue; Michimasa Inaba
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** Emotion Recognition in Conversation (ERC) is critical for enabling natural human-machine interactions. However, existing methods predominantly employ categorical or dimensional emotion annotations, which often fail to adequately represent complex, subtle, or culturally specific emotional nuances. To overcome this limitation, we propose a novel task named Emotion Transcription in Conversation (ETC). This task focuses on generating natural language descriptions that accurately reflect speakers' emotional states within conversational contexts. To address the ETC, we constructed a Japanese dataset comprising text-based dialogues annotated with participants' self-reported emotional states, described in natural language. The dataset also includes emotion category labels for each transcription, enabling quantitative analysis and its application to ERC. We benchmarked baseline models, finding that while fine-tuning on our dataset enhances model performance, current models still struggle to infer implicit emotional states. The ETC task will encourage further research into more expressive emotion understanding in dialogue. The dataset is publicly available at this https URL.
>
---
#### [new 064] SPD-RAG: Sub-Agent Per Document Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出SPD-RAG框架，用于多文档问答任务，解决传统方法在证据覆盖和长文本推理上的不足。通过分文档处理与集中融合提升效果与效率。**

- **链接: [https://arxiv.org/pdf/2603.08329](https://arxiv.org/pdf/2603.08329)**

> **作者:** Yagiz Can Akay; Muhammed Yusuf Kartal; Esra Alparslan; Faruk Ortakoyluoglu; Arda Akpinar
>
> **备注:** 12 pages
>
> **摘要:** Answering complex, real-world queries often requires synthesizing facts scattered across vast document corpora. In these settings, standard retrieval-augmented generation (RAG) pipelines suffer from incomplete evidence coverage, while long-context large language models (LLMs) struggle to reason reliably over massive inputs. We introduce SPD-RAG, a hierarchical multi-agent framework for exhaustive cross-document question answering that decomposes the problem along the document axis. Each document is processed by a dedicated document-level agent operating only on its own content, enabling focused retrieval, while a coordinator dispatches tasks to relevant agents and aggregates their partial answers. Agent outputs are synthesized by merging partial answers through a token-bounded synthesis layer (which supports recursive map-reduce for massive corpora). This document-level specialization with centralized fusion improves scalability and answer quality in heterogeneous multidocument settings while yielding a modular, extensible retrieval pipeline. On the LOONG benchmark (EMNLP 2024) for long-context multi-document QA, SPD-RAG achieves an Avg Score of 58.1 (GPT-5 evaluation), outperforming Normal RAG (33.0) and Agentic RAG (32.8) while using only 38% of the API cost of a full-context baseline (68.0).
>
---
#### [new 065] AdaCultureSafe: Adaptive Cultural Safety Grounded by Cultural Knowledge in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文化安全任务，旨在解决LLMs在文化尊重上的不足。通过构建数据集并分析模型表现，提出基于文化知识的增强方法。**

- **链接: [https://arxiv.org/pdf/2603.08275](https://arxiv.org/pdf/2603.08275)**

> **作者:** Hankun Kang; Di Lin; Zhirong Liao; Pengfei Bai; Xinyi Zeng; Jiawei Jiang; Yuanyuan Zhu; Tieyun Qian
>
> **摘要:** With the widespread adoption of Large Language Models (LLMs), respecting indigenous cultures becomes essential for models' culturally safety and responsible global applications. Existing studies separately consider cultural safety and cultural knowledge and neglect that the former should be grounded by the latter. This severely prevents LLMs from yielding culture-specific respectful responses. Consequently, adaptive cultural safety remains a formidable task. In this work, we propose to jointly model cultural safety and knowledge. First and foremost, cultural-safety and knowledge-paired data serve as the key prerequisite to conduct this research. However, the cultural diversity across regions and the subtlety of cultural differences pose significant challenges to the creation of such paired evaluation data. To address this issue, we propose a novel framework that integrates authoritative cultural knowledge descriptions curation, LLM-automated query generation, and heavy manual verification. Accordingly, we obtain a dataset named AdaCultureSafe containing 4.8K manually decomposed fine-grained cultural descriptions and the corresponding 48K manually verified safety- and knowledge-oriented queries. Upon the constructed dataset, we evaluate three families of popular LLMs on their cultural safety and knowledge proficiency, via which we make a critical discovery: no significant correlation exists between their cultural safety and knowledge proficiency. We then delve into the utility-related neuron activations within LLMs to investigate the potential cause of the absence of correlation, which can be attributed to the difference of the objectives of pre-training and post-alignment. We finally present a knowledge-grounded method, which significantly enhances cultural safety by enforcing the integration of knowledge into the LLM response generation process.
>
---
#### [new 066] One Model Is Enough: Native Retrieval Embeddings from LLM Agent Hidden States
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决LLM代理需额外模型编码查询的问题。通过在LLM中添加轻量投影头，直接生成嵌入向量，提升效率并保持检索效果。**

- **链接: [https://arxiv.org/pdf/2603.08429](https://arxiv.org/pdf/2603.08429)**

> **作者:** Bo Jiang
>
> **摘要:** LLM agents that retrieve external knowledge typically generate a search query as text, then run a separate embedding model to encode it into a vector. This two-model pipeline adds infrastructure complexity and latency, yet is redundant: the LLM already encodes the full conversational context in its hidden states. We propose equipping LLM agents with native retrieval capability by adding a lightweight projection head that maps hidden states directly into the embedding space, eliminating the need for a separate embedding model. Trained with a combination of alignment, contrastive, and rank distillation losses, our method retains 97\% of baseline retrieval quality while enabling the LLM agent to search with its own representations. Experiments on the QReCC conversational search benchmark show competitive Recall@10 and MRR@10 compared to the standard generate-then-encode pipeline, with systematic ablations confirming the contribution of each loss component.
>
---
#### [new 067] Enhancing Consistency of Werewolf AI through Dialogue Summarization and Persona Information
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI游戏任务，旨在提升狼人杀AI的发言一致性。通过对话摘要和角色设定增强AI表达的连贯性与角色特征。**

- **链接: [https://arxiv.org/pdf/2603.07111](https://arxiv.org/pdf/2603.07111)**

> **作者:** Yoshiki Tanaka; Takumasa Kaneko; Hiroki Onozeki; Natsumi Ezure; Ryuichi Uehara; Zhiyang Qi; Tomoya Higuchi; Ryutaro Asahara; Michimasa Inaba
>
> **备注:** Accepted to the 2nd International AIWolfDial Workshop at INLG 2024
>
> **摘要:** The Werewolf Game is a communication game where players' reasoning and discussion skills are essential. In this study, we present a Werewolf AI agent developed for the AIWolfDial 2024 shared task, co-hosted with the 17th INLG. In recent years, large language models like ChatGPT have garnered attention for their exceptional response generation and reasoning capabilities. We thus develop the LLM-based agents for the Werewolf Game. This study aims to enhance the consistency of the agent's utterances by utilizing dialogue summaries generated by LLMs and manually designed personas and utterance examples. By analyzing self-match game logs, we demonstrate that the agent's utterances are contextually consistent and that the character, including tone, is maintained throughout the game.
>
---
#### [new 068] Supporting Workflow Reproducibility by Linking Bioinformatics Tools across Papers and Executable Code
- **分类: cs.CL**

- **简介: 该论文属于生物信息学工作流可复现任务，旨在解决工具描述与代码链接问题。通过实体识别与链接方法，实现论文与代码中工具的自动关联。**

- **链接: [https://arxiv.org/pdf/2603.08195](https://arxiv.org/pdf/2603.08195)**

> **作者:** Clémence Sebe; Olivier Ferret; Aurélie Névéol; Mahdi Esmailoghli; Ulf Leser; Sarah Cohen-Boulakia
>
> **摘要:** Motivation: The rapid growth of biological data has intensified the need for transparent, reproducible, and well-documented computational workflows. The ability to clearly connect the steps of a workflow in the code with their description in a paper would improve workflow understanding, support reproducibility, and facilitate reuse. This task requires the linking of Bioinformatics tools in workflow code with their mentions in a published workflow description. Results: We present CoPaLink, an automated approach that integrates three components: Named Entity Recognition (NER) for identifying tool mentions in scientific text, NER for tool mentions in workflow code, and entity linking grounded on Bioinformatics knowledge bases. We propose approaches for all three steps achieving a high individual F1-measure (84 - 89) and a joint accuracy of 66 when evaluated on Nextflow workflows using Bioconda and Bioweb Knowledge bases. CoPaLink leverages corpora of scientific articles and workflow executable code with curated tool annotations to bridge the gap between narrative descriptions and workflow implementations. Availability: The code is available at this https URL and this https URL. The corpora are also available at this https URL, this https URL and this https URL.
>
---
#### [new 069] An Efficient and Effective Evaluator for Text2SQL Models on Unseen and Unlabeled Data
- **分类: cs.CL**

- **简介: 该论文属于Text2SQL任务，解决在无标签数据上评估模型的问题。提出FusionSQL，通过分析模型输出模式估计准确率，无需参考答案。**

- **链接: [https://arxiv.org/pdf/2603.07841](https://arxiv.org/pdf/2603.07841)**

> **作者:** Trinh Pham; Thanh Tam Nguyen; Viet Huynh; Hongzhi Yin; Quoc Viet Hung Nguyen
>
> **备注:** Accepted at ICDE 2026
>
> **摘要:** Recent advances in large language models has strengthened Text2SQL systems that translate natural language questions into database queries. A persistent deployment challenge is to assess a newly trained Text2SQL system on an unseen and unlabeled dataset when no verified answers are available. This situation arises frequently because database content and structure evolve, privacy policies slow manual review, and carefully written SQL labels are costly and time-consuming. Without timely evaluation, organizations cannot approve releases or detect failures early. FusionSQL addresses this gap by working with any Text2SQL models and estimating accuracy without reference labels, allowing teams to measure quality on unseen and unlabeled datasets. It analyzes patterns in the system's own outputs to characterize how the target dataset differs from the material used during training. FusionSQL supports pre-release checks, continuous monitoring of new databases, and detection of quality decline. Experiments across diverse application settings and question types show that FusionSQL closely follows actual accuracy and reliably signals emerging issues. Our code is available at this https URL.
>
---
#### [new 070] Ramsa: A Large Sociolinguistically Rich Emirati Arabic Speech Corpus for ASR and TTS
- **分类: cs.CL**

- **简介: 该论文介绍Ramsa语料库，用于阿拉伯语语音识别和文本转语音研究，解决低资源语言技术问题，涵盖多种方言和话题，评估了多个模型性能。**

- **链接: [https://arxiv.org/pdf/2603.08125](https://arxiv.org/pdf/2603.08125)**

> **作者:** Rania Al-Sabbagh
>
> **摘要:** Ramsa is a developing 41-hour speech corpus of Emirati Arabic designed to support sociolinguistic research and low-resource language technologies. It contains recordings from structured interviews with native speakers and episodes from national television shows. The corpus features 157 speakers (59 female, 98 male), spans subdialects such as Urban, Bedouin, and Mountain/Shihhi, and covers topics such as cultural heritage, agriculture and sustainability, daily life, professional trajectories, and architecture. It consists of 91 monologic and 79 dialogic recordings, varying in length and recording conditions. A 10\% subset was used to evaluate commercial and open-source models for automatic speech recognition (ASR) and text-to-speech (TTS) in a zero-shot setting to establish initial baselines. Whisper-large-v3-turbo achieved the best ASR performance, with average word and character error rates of 0.268 and 0.144, respectively. MMS-TTS-Ara reported the best mean word and character rates of 0.285 and 0.081, respectively, for TTS. These baselines are competitive but leave substantial room for improvement. The paper highlights the challenges encountered and provides directions for future work.
>
---
#### [new 071] Elenchus: Generating Knowledge Bases from Prover-Skeptic Dialogues
- **分类: cs.CL; cs.AI; cs.LO**

- **简介: 该论文提出Elenchus系统，用于从论证对话中构建知识库，解决知识工程中的显性化问题。通过专家与LLM的对话，明确推理关系，验证逻辑结构。**

- **链接: [https://arxiv.org/pdf/2603.06974](https://arxiv.org/pdf/2603.06974)**

> **作者:** Bradley P. Allen
>
> **备注:** 12 pages, 4 figures, 4 tables
>
> **摘要:** We present Elenchus, a dialogue system for knowledge base construction grounded in inferentialist semantics, where knowledge engineering is re-conceived as explicitation rather than extraction from expert testimony or textual content. A human expert develops a bilateral position (commitments and denials) about a topic through prover-skeptic dialogue with a large language model (LLM) opponent. The LLM proposes tensions (claims that parts of the position are jointly incoherent) which the expert resolves by retraction, refinement, or contestation. The LLM thus serves as a defeasible derivability oracle whose unreliability is structurally contained by the expert's authority. Our main technical contribution is a mapping from Elenchus dialectical states to material bases in Hlobil and Brandom's NonMonotonic MultiSuccedent (NMMS) logic, satisfying Containment and enabling the elaboration of logical vocabulary that makes explicit the inferential relationships negotiated in the dialectic. We demonstrate the approach on the W3C PROV-O provenance ontology, where a single dialogue session elicits and structures design tensions that a domain expert can articulate, corresponding to decisions documented in a retrospective analysis of the ontology's design. Using pyNMMS, an automated NMMS reasoner, we verify that the structural properties of the resulting material base (nontransitivity, nonmonotonicity, and independence) correspond to specific PROV design rationales, demonstrating end-to-end integration from dialogue through formal reasoning.
>
---
#### [new 072] Gender Bias in MT for a Genderless Language: New Benchmarks for Basque
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决性别偏见问题。通过构建新数据集，评估Basque语翻译中的性别偏差，揭示模型对男性形式的偏好。**

- **链接: [https://arxiv.org/pdf/2603.08153](https://arxiv.org/pdf/2603.08153)**

> **作者:** Amaia Murillo; Olatz-Perez-de-Viñaspre; Naiara Perez
>
> **摘要:** Large language models (LLMs) and machine translation (MT) systems are increasingly used in our daily lives, but their outputs can reproduce gender bias present in the training data. Most resources for evaluating such biases are designed for English and reflect its sociocultural context, which limits their applicability to other languages. This work addresses this gap by introducing two new datasets to evaluate gender bias in translations involving Basque, a low-resource and genderless language. WinoMTeus adapts the WinoMT benchmark to examine how gender-neutral Basque occupations are translated into gendered languages such as Spanish and French. FLORES+Gender, in turn, extends the FLORES+ benchmark to assess whether translation quality varies when translating from gendered languages (Spanish and English) into Basque depending on the gender of the referent. We evaluate several general-purpose LLMs and open and proprietary MT systems. The results reveal a systematic preference for masculine forms and, in some models, a slightly higher quality for masculine referents. Overall, these findings show that gender bias is still deeply rooted in these models, and highlight the need to develop evaluation methods that consider both linguistic features and cultural context.
>
---
#### [new 073] Language-Aware Distillation for Multilingual Instruction-Following Speech LLMs with ASR-Only Supervision
- **分类: cs.CL**

- **简介: 该论文属于多语言语音大模型任务，解决监督微调困难和语言干扰问题。提出语言感知的蒸馏方法，提升多语言指令遵循性能。**

- **链接: [https://arxiv.org/pdf/2603.07025](https://arxiv.org/pdf/2603.07025)**

> **作者:** Shreyas Gopal; Donghang Wu; Ashutosh Anshul; Yeo Yue Heng; Yizhou Peng; Haoyang Li; Hexin Liu; Eng Siong Chng
>
> **备注:** Submitted for Review to Interspeech 2026
>
> **摘要:** Speech Large Language Models (LLMs) that understand and follow instructions in many languages are useful for real-world interaction, but are difficult to train with supervised fine-tuning, requiring large, task-specific speech corpora. While recent distillation-based approaches train performant English-only Speech LLMs using only annotated ASR data by aligning text and speech using only a lightweight projector, these models under-perform when scaled to multilingual settings due to language interference in the shared projector. We address this by introducing language-aware distillation using a query bank and a gating network that selects or mixes query tokens using a Q-Former projector. Our approach shows gains of 14% over matched multilingual distillation baselines on instruction following. We further synthesize Audio-MLQA, a multilingual spoken QA benchmark built on MLQA with high-quality TTS questions. Our best model improves over existing Speech LLM baselines by 32% on Audio-MLQA.
>
---
#### [new 074] KohakuRAG: A simple RAG framework with hierarchical document indexing
- **分类: cs.CL**

- **简介: 该论文提出KohakuRAG，解决RAG系统在精确引用和答案稳定性上的问题，通过分层文档索引、查询规划和集成推理提升性能。**

- **链接: [https://arxiv.org/pdf/2603.07612](https://arxiv.org/pdf/2603.07612)**

> **作者:** Shih-Ying Yeh; Yueh-Feng Ku; Ko-Wei Huang; Buu-Khang Tu
>
> **备注:** 38pages
>
> **摘要:** Retrieval-augmented generation (RAG) systems that answer questions from document collections face compounding difficulties when high-precision citations are required: flat chunking strategies sacrifice document structure, single-query formulations miss relevant passages through vocabulary mismatch, and single-pass inference produces stochastic answers that vary in both content and citation selection. We present KohakuRAG, a hierarchical RAG framework that preserves document structure through a four-level tree representation (document $\rightarrow$ section $\rightarrow$ paragraph $\rightarrow$ sentence) with bottom-up embedding aggregation, improves retrieval coverage through an LLM-powered query planner with cross-query reranking, and stabilizes answers through ensemble inference with abstention-aware voting. We evaluate on the WattBot 2025 Challenge, a benchmark requiring systems to answer technical questions from 32 documents with $\pm$0.1% numeric tolerance and exact source attribution. KohakuRAG achieves first place on both public and private leaderboards (final score 0.861), as the only team to maintain the top position across both evaluation partitions. Ablation studies reveal that prompt ordering (+80% relative), retry mechanisms (+69%), and ensemble voting with blank filtering (+1.2pp) each contribute substantially, while hierarchical dense retrieval alone matches hybrid sparse-dense approaches (BM25 adds only +3.1pp). We release KohakuRAG as open-source software at this https URL.
>
---
#### [new 075] Position: LLMs Must Use Functor-Based and RAG-Driven Bias Mitigation for Fairness
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的公平性任务，旨在解决LLMs中的性别和种族偏见问题。通过结合范畴论变换和RAG技术，实现结构去偏与上下文校准，提升模型输出的公平性。**

- **链接: [https://arxiv.org/pdf/2603.07368](https://arxiv.org/pdf/2603.07368)**

> **作者:** Ravi Ranjan; Utkarsh Grover; Agorista Polyzou
>
> **备注:** 24 pages, 3 figures
>
> **摘要:** Biases in large language models (LLMs) often manifest as systematic distortions in associations between demographic attributes and professional or social roles, reinforcing harmful stereotypes across gender, ethnicity, and geography. This position paper advocates for addressing demographic and gender biases in LLMs through a dual-pronged methodology, integrating category-theoretic transformations and retrieval-augmented generation (RAG). Category theory provides a rigorous, structure-preserving mathematical framework that maps biased semantic domains to unbiased canonical forms via functors, ensuring bias elimination while preserving semantic integrity. Complementing this, RAG dynamically injects diverse, up-to-date external knowledge during inference, directly countering ingrained biases within model parameters. By combining structural debiasing through functor-based mappings and contextual grounding via RAG, we outline a comprehensive framework capable of delivering equitable and fair model outputs. Our synthesis of the current literature validates the efficacy of each approach individually, while addressing potential critiques demonstrates the robustness of this integrated strategy. Ensuring fairness in LLMs, therefore, demands both the mathematical rigor of category-theoretic transformations and the adaptability of retrieval augmentation.
>
---
#### [new 076] Learning Multiple Utterance-Level Attribute Representations with a Unified Speech Encoder
- **分类: cs.CL**

- **简介: 该论文属于语音表示学习任务，旨在解决如何生成多种话语级属性表示的问题。通过统一的后训练框架，模型可同时学习语义与说话人信息，提升多语言语音检索和说话人识别效果。**

- **链接: [https://arxiv.org/pdf/2603.08312](https://arxiv.org/pdf/2603.08312)**

> **作者:** Maryem Bouziane; Salima Mdhaffar; Yannick Estève
>
> **备注:** Submitted to Interspeech
>
> **摘要:** Speech foundation models trained with self-supervised learning produce generic speech representations that support a wide range of speech processing tasks. When further adapted with supervised learning, these models can achieve strong performance on specific downstream tasks. Recent post-training approaches, such as SAMU-XSLR and SONAR, align speech representations with utterance-level semantic representations, enabling effective multimodal (speech-text) and multilingual applications. While speech foundation models typically learn contextual embeddings at the acoustic frame level, these methods learn representations at the utterance level. In this work, we extend this paradigm to arbitrary utterance-level attributes and propose a unified post-training framework that enables a single speech foundation model to generate multiple types of utterance-level representations. We demonstrate the effectiveness of this approach by jointly learning semantic and speaker representations and evaluating them on multilingual speech retrieval and speaker recognition tasks.
>
---
#### [new 077] Sensivity of LLMs' Explanations to the Training Randomness:Context, Class & Task Dependencies
- **分类: cs.CL**

- **简介: 该论文研究Transformer模型解释对训练随机性的敏感性，分析上下文、类别和任务的影响。属于模型解释领域，解决解释稳定性问题，通过实验验证三者的影响程度。**

- **链接: [https://arxiv.org/pdf/2603.08241](https://arxiv.org/pdf/2603.08241)**

> **作者:** Romain Loncour; Jérémie Bogaert; François-Xavier Standaert
>
> **备注:** 6 pages, 6 figures
>
> **摘要:** Transformer models are now a cornerstone in natural language processing. Yet, explaining their decisions remains a challenge. It was shown recently that the same model trained on the same data with a different randomness can lead to very different explanations. In this paper, we investigate how the (syntactic) context, the classes to be learned and the tasks influence this explanations' sensitivity to randomness. We show that they all have statistically significant impact: smallest for the (syntactic) context, medium for the classes and largest for the tasks.
>
---
#### [new 078] Deep Research, Shallow Evaluation: A Case Study in Meta-Evaluation for Long-Form QA Benchmarks
- **分类: cs.CL**

- **简介: 该论文属于元评估任务，旨在解决长文本问答基准评估中的方法有效性问题。通过案例研究，分析了人类偏好与指标评估的优劣，提出改进评估设计的建议。**

- **链接: [https://arxiv.org/pdf/2603.06942](https://arxiv.org/pdf/2603.06942)**

> **作者:** Jena D. Hwang; Varsha Kishore; Amanpreet Singh; Dany Haddad; Aakanksha Naik; Malachi Hamada; Jonathan Bragg; Mike D'Arcy; Daniel S. Weld; Lucy Lu Wang; Doug Downey; Sergey Feldman
>
> **备注:** 11 pages (including Limitations), 10 figures, 9 tables
>
> **摘要:** Recent advances have made long-form report-generating systems widely available. This has prompted evaluation frameworks that use LLM-as-judge protocols and claim verification, along with meta-evaluation frameworks that seek to validate these methods. Many of the meta-evaluations estimate an evaluation quality's by comparing its assessments against human pairwise preferences. Prior work, however, suggests that human pairwise preference may be overly simplistic and can fail to capture nuances of expert expectations. We conduct a case study in meta-evaluation for long-form QA benchmarks using ScholarQA-CS2, a benchmark designed for assessing retrieval-augmented deep-research QA in the scientific domain. We comprehensively validate the benchmark through human pairwise preference judgments, then critically examine the strengths, weaknesses, and confounders of this approach. We show that pairwise preference rankings are best suited for system-level evaluation, while explicit metric-wise annotations and expert annotators are critical for reliable metric-level assessment, with subjectivity remaining a key challenge. Based on our findings, we offer practical guidelines for designing future meta-evaluations that better align evaluation methods, annotator expertise, and reporting practices. By surfacing these methodological challenges, we aim to advance evaluation standards for deep-research systems.
>
---
#### [new 079] Nwāchā Munā: A Devanagari Speech Corpus and Proximal Transfer Benchmark for Nepal Bhasha ASR
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决尼泊尔语资源匮乏问题。通过构建语料库并进行近源迁移学习，提升低资源语言的自动语音识别性能。**

- **链接: [https://arxiv.org/pdf/2603.07554](https://arxiv.org/pdf/2603.07554)**

> **作者:** Rishikesh Kumar Sharma; Safal Narshing Shrestha; Jenny Poudel; Rupak Tiwari; Arju Shrestha; Rupak Raj Ghimire; Bal Krishna Bal
>
> **摘要:** Nepal Bhasha (Newari), an endangered language of the Kathmandu Valley, remains digitally marginalized due to the severe scarcity of annotated speech resources. In this work, we introduce Nwāchā Munā, a newly curated 5.39-hour manually transcribed Devanagari speech corpus for Nepal Bhasha, and establish the first benchmark using script-preserving acoustic modeling. We investigate whether proximal cross-lingual transfer from a geographically and linguistically adjacent language (Nepali) can rival large-scale multilingual pretraining in an ultra-low-resource Automatic Speech Recognition (ASR) setting. Fine-tuning a Nepali Conformer model reduces the Character Error Rate (CER) from a 52.54% zero-shot baseline to 17.59% with data augmentation, effectively matching the performance of the multilingual Whisper-Small model despite utilizing significantly fewer parameters. Our findings demonstrate that proximal transfer within South Asian language clusters serves as a computationally efficient alternative to massive multilingual models. We openly release the dataset and benchmarks to digitally enable the Newari community and foster further research in Nepal Bhasha.
>
---
#### [new 080] A Coin Flip for Safety: LLM Judges Fail to Reliably Measure Adversarial Robustness
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的安全评估任务，旨在解决LLM作为评判者在对抗鲁棒性评估中的不可靠问题。研究发现现有方法因分布偏移导致性能下降，提出新基准以提高评估可靠性。**

- **链接: [https://arxiv.org/pdf/2603.06594](https://arxiv.org/pdf/2603.06594)**

> **作者:** Leo Schwinn; Moritz Ladenburger; Tim Beyer; Mehrnaz Mofakhami; Gauthier Gidel; Stephan Günnemann
>
> **摘要:** Automated \enquote{LLM-as-a-Judge} frameworks have become the de facto standard for scalable evaluation across natural language processing. For instance, in safety evaluation, these judges are relied upon to evaluate harmfulness in order to benchmark the robustness of safety against adversarial attacks. However, we show that existing validation protocols fail to account for substantial distribution shifts inherent to red-teaming: diverse victim models exhibit distinct generation styles, attacks distort output patterns, and semantic ambiguity varies significantly across jailbreak scenarios. Through a comprehensive audit using 6642 human-verified labels, we reveal that the unpredictable interaction of these shifts often causes judge performance to degrade to near random chance. This stands in stark contrast to the high human agreement reported in prior work. Crucially, we find that many attacks inflate their success rates by exploiting judge insufficiencies rather than eliciting genuinely harmful content. To enable more reliable evaluation, we propose ReliableBench, a benchmark of behaviors that remain more consistently judgeable, and JudgeStressTest, a dataset designed to expose judge failures. Data available at: this https URL.
>
---
#### [new 081] MAWARITH: A Dataset and Benchmark for Legal Inheritance Reasoning with LLMs
- **分类: cs.CL**

- **简介: 该论文提出MAWARITH数据集和MIR-E评估指标，用于法律继承推理任务，解决伊斯兰继承法中多步骤推理与规则应用问题。**

- **链接: [https://arxiv.org/pdf/2603.07539](https://arxiv.org/pdf/2603.07539)**

> **作者:** Abdessalam Bouchekif; Shahd Gaben; Samer Rashwani; Somaya Eltanbouly; Mutaz Al-Khatib; Heba Sbahi; Mohammed Ghaly; Emad Mohamed
>
> **摘要:** Islamic inheritance law ('ilm al-mawarith) is challenging for large language models because solving inheritance cases requires complex, structured multi-step reasoning and the correct application of juristic rules to compute heirs' shares. We introduce MAWARITH, a large-scale annotated dataset of 12,500 Arabic inheritance cases to train and evaluate the full reasoning chain: (i) identifying eligible heirs, (ii) applying blocking (hajb) and allocation rules, and (iii) computing exact inheritance shares. Unlike prior datasets that restrict inheritance case solving to multiple-choice questions, MAWARITH supports the full reasoning chain and provides step-by-step solutions, including intermediate legal decisions and justifications based on classical juristic sources and established inheritance rules, as well as exact share calculations. To evaluate models beyond final-answer accuracy, we propose MIR-E (Mawarith Inheritance Reasoning Evaluation), a weighted multi-stage metric that scores key reasoning stages and captures error propagation across the pipeline. We evaluate five LLMs in a zero-shot setting. Gemini-2.5-flash achieves about 90% MIR-E on both validation and test, while Fanar-C, Fanar-Sadiq, LLaMA 3, and Qwen 3 remain below 50%. Our error analysis identifies recurring failure patterns, including scenario misinterpretation, errors in heir identification, errors in share allocation, and missing or incorrect application of key inheritance rules such as 'awl and radd. The MAWARITH dataset is publicly available at this https URL.
>
---
#### [new 082] TableMind++: An Uncertainty-Aware Programmatic Agent for Tool-Augmented Table Reasoning
- **分类: cs.CL**

- **简介: 该论文提出TableMind++，解决表格推理中的不确定性问题。通过引入不确定性感知框架，提升模型的准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.07528](https://arxiv.org/pdf/2603.07528)**

> **作者:** Mingyue Cheng; Shuo Yu; Chuang Jiang; Xiaoyu Tao; Qingyang Mao; Jie Ouyang; Qi Liu; Enhong Chen
>
> **备注:** 6 tables, 9 figures
>
> **摘要:** Table reasoning requires models to jointly perform semantic understanding and precise numerical operations. Most existing methods rely on a single-turn reasoning paradigm over tables which suffers from context overflow and weak numerical sensitivity. To address these limitations, we previously proposed TableMind as a tuning-based autonomous programmatic agent that simulates human-like interaction within a lightweight large language model (LLM). TableMind internalizes planning, action, and reflection through a two-stage training strategy involving supervised fine-tuning (SFT) on filtered high-quality data and reinforcement learning (RL) via a multi-perspective reward and the Rank-Aware Policy Optimization (RAPO) algorithm. While TableMind establishes a solid foundation for programmatic agents, the inherent stochasticity of LLMs remains a critical challenge that leads to hallucinations. In this paper, we extend this foundation to TableMind++ by introducing a novel uncertainty-aware inference framework to mitigate hallucinations. Specifically, we propose memory-guided plan pruning to retrieve historical trajectories for validating and filtering out logically flawed plans to address epistemic uncertainty. To ensure execution precision, we introduce confidence-based action refinement which monitors token-level probabilities to detect and self-correct syntactic noise for aleatoric uncertainty mitigation. Finally, we employ dual-weighted trajectory aggregation to synthesize a robust consensus from multiple reasoning paths. Extensive experiments on diverse benchmarks demonstrate that TableMind++ consistently outperforms previous baselines and proprietary models to validate the effectiveness of integrating autonomous training with uncertainty quantification. Our code is available.
>
---
#### [new 083] QuadAI at SemEval-2026 Task 3: Ensemble Learning of Hybrid RoBERTa and LLMs for Dimensional Aspect-Based Sentiment Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感分析任务，解决维度方面的情感回归问题。通过集成混合RoBERTa和大语言模型，提升预测稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.07766](https://arxiv.org/pdf/2603.07766)**

> **作者:** A.J.W. de Vink; Filippos Karolos Ventirozos; Natalia Amat-Lefort; Lifeng Han
>
> **备注:** SemEval System Report
>
> **摘要:** We present our system for SemEval-2026 Task 3 on dimensional aspect-based sentiment regression. Our approach combines a hybrid RoBERTa encoder, which jointly predicts sentiment using regression and discretized classification heads, with large language models (LLMs) via prediction-level ensemble learning. The hybrid encoder improves prediction stability by combining continuous and discretized sentiment representations. We further explore in-context learning with LLMs and ridge-regression stacking to combine encoder and LLM predictions. Experimental results on the development set show that ensemble learning significantly improves performance over individual models, achieving substantial reductions in RMSE and improvements in correlation scores. Our findings demonstrate the complementary strengths of encoder-based and LLM-based approaches for dimensional sentiment analysis. Our development code and resources will be shared at this https URL
>
---
#### [new 084] Scaling Self-Supervised Speech Models Uncovers Deep Linguistic Relationships: Evidence from the Pacific Cluster
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究自监督语音模型在语言识别中的表现，旨在揭示语言间的深层语系关系。通过扩展语言覆盖范围，发现模型能捕捉更复杂的语言接触与演化信息。**

- **链接: [https://arxiv.org/pdf/2603.07238](https://arxiv.org/pdf/2603.07238)**

> **作者:** Minu Kim; Hoirin Kim; David R. Mortensen
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Similarities between language representations derived from Self-Supervised Speech Models (S3Ms) have been observed to primarily reflect geographic proximity or surface typological similarities driven by recent expansion or contact, potentially missing deeper genealogical signals. We investigate how scaling linguistic coverage of an S3M-based language identification system from 126 to 4,017 languages influences this topology. Our results reveal a non-linear effect: while phylogenetic recovery remains stagnant up to the 1K scale, the 4K model displays a dramatic qualitative shift, resolving both clear lineages and complex, long-term linguistic contact. Notably, our analysis reveals the emergence of a robust macro-cluster in the Pacific (comprising Papuan, Oceanic, and Australian languages) and investigates its latent drivers. We find that the 4K model utilizes a more concentrated encoding that captures shared, robust acoustic signatures such as global energy dynamics. These findings suggest that massive S3Ms can internalize multiple layers of language history, providing a promising perspective for computational phylogenetics and the study of language contact.
>
---
#### [new 085] DC-W2S: Dual-Consensus Weak-to-Strong Training for Reliable Process Reward Modeling in Biological Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于科学推理任务，解决PRMs训练中缺乏可靠标签的问题。通过DC-W2S框架，利用弱监督数据提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2603.08095](https://arxiv.org/pdf/2603.08095)**

> **作者:** Chi-Min Chan; Ehsan Hajiramezanali; Xiner Li; Edward De Brouwer; Carl Edwards; Wei Xue; Sirui Han; Yike Guo; Gabriele Scalia
>
> **摘要:** In scientific reasoning tasks, the veracity of the reasoning process is as critical as the final outcome. While Process Reward Models (PRMs) offer a solution to the coarse-grained supervision problems inherent in Outcome Reward Models (ORMs), their deployment is hindered by the prohibitive cost of obtaining expert-verified step-wise labels. This paper addresses the challenge of training reliable PRMs using abundant but noisy "weak" supervision. We argue that existing Weak-to-Strong Generalization (W2SG) theories lack prescriptive guidelines for selecting high-quality training signals from noisy data. To bridge this gap, we introduce the Dual-Consensus Weak-to-Strong (DC-W2S) framework. By intersecting Self-Consensus (SC) metrics among weak supervisors with Neighborhood-Consensus (NC) metrics in the embedding space, we stratify supervision signals into distinct reliability regimes. We then employ a curriculum of instance-level balanced sampling and label-level reliability-aware masking to guide the training process. We demonstrate that DC-W2S enables the training of robust PRMs for complex reasoning without exhaustive expert annotation, proving that strategic data curation is more effective than indiscriminate training on large-scale noisy datasets.
>
---
#### [new 086] Counting on Consensus: Selecting the Right Inter-annotator Agreement Metric for NLP Annotation and Evaluation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的标注与评估任务，旨在解决如何选择合适的标注者间一致性度量问题。论文分析了不同任务类型下的度量方法及其限制，提出最佳实践以提高评估的可靠性与透明度。**

- **链接: [https://arxiv.org/pdf/2603.06865](https://arxiv.org/pdf/2603.06865)**

> **作者:** Joseph James
>
> **摘要:** Human annotation remains the foundation of reliable and interpretable data in Natural Language Processing (NLP). As annotation and evaluation tasks continue to expand, from categorical labelling to segmentation, subjective judgment, and continuous rating, measuring agreement between annotators has become increasingly more complex. This paper outlines how inter-annotator agreement (IAA) has been conceptualised and applied across NLP and related disciplines, describing the assumptions and limitations of common approaches. We organise agreement measures by task type and discuss how factors such as label imbalance and missing data influence reliability estimates. In addition, we highlight best practices for clear and transparent reporting, including the use of confidence intervals and the analysis of disagreement patterns. The paper aims to serve as a guide for selecting and interpreting agreement measures, promoting more consistent and reproducible human annotation and evaluation in NLP.
>
---
#### [new 087] Hierarchical Latent Structures in Data Generation Process Unify Mechanistic Phenomena across Scale
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型解释任务，旨在统一解释Transformer中的机制现象。通过构建具有层次结构的数据生成过程，揭示诱导头、函数向量等现象的共同根源。**

- **链接: [https://arxiv.org/pdf/2603.06592](https://arxiv.org/pdf/2603.06592)**

> **作者:** Jonas Rohweder; Subhabrata Dutta; Iryna Gurevych
>
> **摘要:** Contemporary studies have uncovered many puzzling phenomena in the neural information processing of Transformer-based language models. Building a robust, unified understanding of these phenomena requires disassembling a model within the scope of its training. While the intractable scale of pretraining corpora limits a bottom-up investigation in this direction, simplistic assumptions of the data generation process limit the expressivity and fail to explain complex patterns. In this work, we use probabilistic context-free grammars (PCFGs) to generate synthetic corpora that are faithful and computationally efficient proxies for web-scale text corpora. We investigate the emergence of three mechanistic phenomena: induction heads, function vectors, and the Hydra effect, under our designed data generation process, as well as in the checkpoints of real-world language models. Our findings suggest that hierarchical structures in the data generation process serve as the X-factor in explaining the emergence of these phenomena. We provide the theoretical underpinnings of the role played by hierarchy in the training dynamics of language models. In a nutshell, our work is the first of its kind to provide a unified explanation behind the emergence of seemingly unrelated mechanistic phenomena in LLMs, augmented with efficient synthetic tooling for future interpretability research.
>
---
#### [new 088] Rethinking Personalization in Large Language Models at the Token Level
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的个性化任务，旨在提升大语言模型输出的个性化程度。研究提出PerCE方法，通过token级分析优化个性化训练，显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.06595](https://arxiv.org/pdf/2603.06595)**

> **作者:** Chenheng Zhang; Yijun Lu; Lizhe Fang; Chunyuan Zheng; Jiajun Chai; Xiaohan Wang; Guojun Yin; Wei Lin; Yisen Wang; Zhouchen Lin
>
> **摘要:** With large language models (LLMs) now performing strongly across diverse tasks, there is growing demand for them to personalize outputs for individual users. Personalization is typically framed as an additional layer on top of a base NLP task, requiring model responses to meet user-specific needs while still accomplishing the underlying task. From a token-level perspective, different tokens in a response contribute to personalization to varying degrees. Tokens with higher personalization relevance should therefore receive greater emphasis when developing personalized LLMs. However, accurately estimating such personalization degrees remains challenging. To address this challenge, we propose PerContrast, a self-contrast method that estimates each output token's dependence on user-specific information through causal intervention. Building on this mechanism, we develop the PerCE loss, which adaptively upweights tokens with higher estimated personalization degrees during training via a bootstrap procedure, enabling the model to alternate between estimating and optimizing these tokens. Experiments on multiple LLMs demonstrate that PerCE substantially improves personalization performance with minimal additional cost, achieving average gains of over 10% and up to 68.04% on the LongLaMP dataset, along with strong cross-task and cross-scenario transferability. These results highlight the importance of token-level personalization modeling and establish token-aware training as a simple yet effective paradigm for advancing personalized LLMs.
>
---
#### [new 089] How Much Noise Can BERT Handle? Insights from Multilingual Sentence Difficulty Detection
- **分类: cs.CL**

- **简介: 该论文属于句子难度检测任务，研究噪声数据对BERT性能的影响，探索去噪方法提升模型效果。**

- **链接: [https://arxiv.org/pdf/2603.07346](https://arxiv.org/pdf/2603.07346)**

> **作者:** Nouran Khallaf; Serge Sharoff
>
> **摘要:** Noisy training data can significantly degrade the performance of language-model-based classifiers, particularly in non-topical classification tasks. In this study we designed a methodological framework to assess the impact of denoising. More specifically, we explored a range of denoising strategies for sentence-level difficulty detection, using training data derived from document-level difficulty annotations obtained through noisy crowdsourcing. Beyond monolingual settings, we also address cross-lingual transfer, where a multilingual language model is trained in one language and tested in another. We evaluate several noise reduction techniques, including Gaussian Mixture Models (GMM), Co-Teaching, Noise Transition Matrices, and Label Smoothing. Our results indicate that while BERT-based models exhibit inherent robustness to noise, incorporating explicit noise detection can further enhance performance. For our smaller dataset, GMM-based noise filtering proves particularly effective in improving prediction quality by raising the Area-Under-the-Curve score from 0.52 to 0.92, or to 0.93 when de-noising methods are combined. However, for our larger dataset, the intrinsic regularisation of pre-trained language models provides a strong baseline, with denoising methods yielding only marginal gains (from 0.92 to 0.94, while a combination of two denoising methods made no contribution). Nonetheless, removing noisy sentences (about 20\% of the dataset) helps in producing a cleaner corpus with fewer infelicities. As a result we have released the largest multilingual corpus for sentence difficulty prediction: see this https URL
>
---
#### [new 090] EvoScientist: Towards Multi-Agent Evolving AI Scientists for End-to-End Scientific Discovery
- **分类: cs.CL**

- **简介: 该论文提出EvoScientist，解决AI科学家在科学发现中策略固化问题，通过多智能体协作与持续记忆提升研究效率和创新性。**

- **链接: [https://arxiv.org/pdf/2603.08127](https://arxiv.org/pdf/2603.08127)**

> **作者:** Yougang Lyu; Xi Zhang; Xinhao Yi; Yuyue Zhao; Shuyu Guo; Wenxiang Hu; Jan Piotrowski; Jakub Kaliski; Jacopo Urbani; Zaiqiao Meng; Lun Zhou; Xiaohui Yan
>
> **摘要:** The increasing adoption of Large Language Models (LLMs) has enabled AI scientists to perform complex end-to-end scientific discovery tasks requiring coordination of specialized roles, including idea generation and experimental execution. However, most state-of-the-art AI scientist systems rely on static, hand-designed pipelines and fail to adapt based on accumulated interaction histories. As a result, these systems overlook promising research directions, repeat failed experiments, and pursue infeasible ideas. To address this, we introduce EvoScientist, an evolving multi-agent AI scientist framework that continuously improves research strategies through persistent memory and self-evolution. EvoScientist comprises three specialized agents: a Researcher Agent (RA) for scientific idea generation, an Engineer Agent (EA) for experiment implementation and execution, and an Evolution Manager Agent (EMA) that distills insights from prior interactions into reusable knowledge. EvoScientist contains two persistent memory modules: (i) an ideation memory, which summarizes feasible research directions from top-ranked ideas while recording previously unsuccessful directions; and (ii) an experimentation memory, which captures effective data processing and model training strategies derived from code search trajectories and best-performing implementations. These modules enable the RA and EA to retrieve relevant prior strategies, improving idea quality and code execution success rates over time. Experiments show that EvoScientist outperforms 7 open-source and commercial state-of-the-art systems in scientific idea generation, achieving higher novelty, feasibility, relevance, and clarity via automatic and human evaluation. EvoScientist also substantially improves code execution success rates through multi-agent evolution, demonstrating persistent memory's effectiveness for end-to-end scientific discovery.
>
---
#### [new 091] AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉问答任务，旨在解决模糊视觉问题的策略性回答问题。提出AQuA数据集，分类模糊程度并指导响应策略，提升模型应对不确定性的能力。**

- **链接: [https://arxiv.org/pdf/2603.07394](https://arxiv.org/pdf/2603.07394)**

> **作者:** Jihyoung Jang; Hyounghun Kim
>
> **备注:** ICLR 2026 (28 pages); Project website: this https URL
>
> **摘要:** Visual Question Answering (VQA) is a core task for evaluating the capabilities of Vision-Language Models (VLMs). Existing VQA benchmarks primarily feature clear and unambiguous image-question pairs, whereas real-world scenarios often involve varying degrees of ambiguity that require nuanced reasoning and context-appropriate response strategies. Although recent studies have begun to address ambiguity in VQA, they lack (1) a systematic categorization of ambiguity levels and (2) datasets and models that support strategy-aware responses. In this paper, we introduce Ambiguous Visual Question Answering (AQuA), a fine-grained dataset that classifies ambiguous VQA instances into four levels according to the nature and degree of ambiguity, along with the optimal response strategy for each case. Our evaluation of diverse open-source and proprietary VLMs shows that most models fail to adapt their strategy to the ambiguity type, frequently producing overconfident answers rather than seeking clarification or acknowledging uncertainty. To address this challenge, we fine-tune VLMs on AQuA, enabling them to adaptively choose among multiple response strategies, such as directly answering, inferring intent from contextual cues, listing plausible alternatives, or requesting clarification. VLMs trained on AQuA achieve strategic response generation for ambiguous VQA, demonstrating the ability to recognize ambiguity, manage uncertainty, and respond with context-appropriate strategies, while outperforming both open-source and closed-source baselines.
>
---
#### [new 092] Dial: A Knowledge-Grounded Dialect-Specific NL2SQL System
- **分类: cs.DB; cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于NL2SQL任务，解决多方言数据库系统中自然语言到SQL的准确转换问题。提出Dial框架，通过知识库和逻辑规划提升语法正确性和语义一致性。**

- **链接: [https://arxiv.org/pdf/2603.07449](https://arxiv.org/pdf/2603.07449)**

> **作者:** Xiang Zhang; Hongming Xu; Le Zhou; Wei Zhou; Xuanhe Zhou; Guoliang Li; Yuyu Luo; Changdong Liu; Guorun Chen; Jiang Liao; Fan Wu
>
> **摘要:** Enterprises commonly deploy heterogeneous database systems, each of which owns a distinct SQL dialect with different syntax rules, built-in functions, and execution constraints. However, most existing NL2SQL methods assume a single dialect (e.g., SQLite) and struggle to produce queries that are both semantically correct and executable on target engines. Prompt-based approaches tightly couple intent reasoning with dialect syntax, rule-based translators often degrade native operators into generic constructs, and multi-dialect fine-tuning suffers from cross-dialect interference. In this paper, we present Dial, a knowledge-grounded framework for dialect-specific NL2SQL. Dial introduces: (1) a Dialect-Aware Logical Query Planning module that converts natural language into a dialect-aware logical query plan via operator-level intent decomposition and divergence-aware specification; (2) HINT-KB, a hierarchical intent-aware knowledge base that organizes dialect knowledge into (i) a canonical syntax reference, (ii) a declarative function repository, and (iii) a procedural constraint repository; and (3) an execution-driven debugging and semantic verification loop that separates syntactic recovery from logic auditing to prevent semantic drift. We construct DS-NL2SQL, a benchmark covering six major database systems with 2,218 dialect-specific test cases. Experimental results show that Dial consistently improves translation accuracy by 10.25% and dialect feature coverage by 15.77% over state-of-the-art baselines. The code is at this https URL.
>
---
#### [new 093] Symmetry-Constrained Language-Guided Program Synthesis for Discovering Governing Equations from Noisy and Partial Observations
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于科学发现任务，旨在从噪声和不完整数据中提取物理方程。提出SymLang框架，结合对称性约束、语言模型和贝叶斯方法，提升方程发现的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.06869](https://arxiv.org/pdf/2603.06869)**

> **作者:** Mirza Samad Ahmed Baig; Syeda Anshrah Gillani
>
> **备注:** 12 pages, 4 figures, 5 tables
>
> **摘要:** Discovering compact governing equations from experimental observations is one of the defining objectives of quantitative science, yet practical discovery pipelines routinely fail when measurements are noisy, relevant state variables are unobserved, or multiple symbolic structures explain the data equally well within statistical uncertainty. Here we introduce SymLang (Symmetry-constrained Language-guided equation discovery), a unified framework that brings together three previously separate ideas: (i) typed symmetry-constrained grammars that encode dimensional analysis, group-theoretic invariance, and parity constraints as hard production rules, eliminating on average 71.3% of candidate expression trees before any fitting; (ii) language-model-guided program synthesis in which a fine-tuned 7B-parameter proposer, conditioned on interpretable data descriptors, efficiently navigates the constrained search space; and (iii) MDL-regularized Bayesian model selection coupled with block-bootstrap stability analysis that quantifies structural uncertainty rather than committing to a single best equation. Across 133 dynamical systems spanning classical mechanics, electrodynamics, thermodynamics, population dynamics, and nonlinear oscillators, SymLang achieves an exact structural recovery rate of 83.7% under 10% observational noise - a 22.4 percentage-point improvement over the next-best baseline - while reducing out-of-distribution extrapolation error by 61% and near-eliminating conservation-law violations (3.1 x 10-3 vs. 187.3 x 10-3 physical drift for the closest competitor). In all tested regimes the framework correctly identifies structural degeneracy, reporting it explicitly rather than returning a confidently wrong single equation. The framework is fully open-source and reproducible, providing a principled pathway from raw data to interpretable, physically auditable symbolic laws.
>
---
#### [new 094] SynPlanResearch-R1: Encouraging Tool Exploration for Deep Research with Synthetic Plans
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于研究代理任务，解决代理探索行为不足问题。提出SynPlanResearch-R1框架，通过合成轨迹提升探索能力，增强模型性能。**

- **链接: [https://arxiv.org/pdf/2603.07853](https://arxiv.org/pdf/2603.07853)**

> **作者:** Hansi Zeng; Zoey Li; Yifan Gao; Chenwei Zhang; Xiaoman Pan; Tao Yang; Fengran Mo; Jiacheng Lin; Xian Li; Jingbo Shang
>
> **摘要:** Research Agents enable models to gather information from the web using tools to answer user queries, requiring them to dynamically interleave internal reasoning with tool use. While such capabilities can in principle be learned via reinforcement learning with verifiable rewards (RLVR), we observe that agents often exhibit poor exploration behaviors, including premature termination and biased tool usage. As a result, RLVR alone yields limited improvements. We propose SynPlanResearch-R1, a framework that synthesizes tool-use trajectories that encourage deeper exploration to shape exploration during cold-start supervised fine-tuning, providing a strong initialization for subsequent RL. Across seven multi-hop and open-web benchmarks, \framework improves performance by up to 6.0% on Qwen3-8B and 5.8% on Qwen3-4B backbones respectively compared to SOTA baselines. Further analyses of tool-use patterns and training dynamics compared to baselines shed light on the factors underlying these gains. Our code is publicly available at this https URL.
>
---
#### [new 095] vLLM Hook v0: A Plug-in for Programming Model Internals on vLLM
- **分类: cs.LG; cs.CL; cs.PL**

- **简介: 该论文属于模型推理优化任务，旨在解决vLLM内部状态不可编程的问题，提出vLLM Hook插件实现对模型内部状态的编程控制。**

- **链接: [https://arxiv.org/pdf/2603.06588](https://arxiv.org/pdf/2603.06588)**

> **作者:** Ching-Yun Ko; Pin-Yu Chen
>
> **摘要:** Modern artificial intelligence (AI) models are deployed on inference engines to optimize runtime efficiency and resource allocation, particularly for transformer-based large language models (LLMs). The vLLM project is a major open-source library to support model serving and inference. However, the current implementation of vLLM limits programmability of the internal states of deployed models. This prevents the use of popular test-time model alignment and enhancement methods. For example, it prevents the detection of adversarial prompts based on attention patterns or the adjustment of model responses based on activation steering. To bridge this critical gap, we present vLLM Hook, an opensource plug-in to enable the programming of internal states for vLLM models. Based on a configuration file specifying which internal states to capture, vLLM Hook provides seamless integration to vLLM and supports two essential features: passive programming and active programming. For passive programming, vLLM Hook probes the selected internal states for subsequent analysis, while keeping the model generation intact. For active programming, vLLM Hook enables efficient intervention of model generation by altering the selected internal states. In addition to presenting the core functions of vLLM Hook, in version 0, we demonstrate 3 use cases including prompt injection detection, enhanced retrieval-augmented retrieval (RAG), and activation steering. Finally, we welcome the community's contribution to improve vLLM Hook via this https URL.
>
---
#### [new 096] Deterministic Differentiable Structured Pruning for Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决结构化剪枝中的训练-测试不匹配问题。提出DDP方法，通过确定性优化实现更高效的模型剪枝。**

- **链接: [https://arxiv.org/pdf/2603.08065](https://arxiv.org/pdf/2603.08065)**

> **作者:** Weiyu Huang; Pengle Zhang; Xiaolu Zhang; Jun Zhou; Jun Zhu; Jianfei Chen
>
> **摘要:** Structured pruning reduces LLM inference cost by removing low-importance architectural components. This can be viewed as learning a multiplicative gate for each component under an l0 sparsity constraint. Due to the discreteness of the l0 norm, prior work typically adopts stochastic hard-concrete relaxations to enable differentiable optimization; however, this stochasticity can introduce a train--test mismatch when sampled masks are discretized for deployment and restricts masks to a bounded, near-binary range. To address this, we propose Deterministic Differentiable Pruning (DDP), a mask-only optimization method that eliminates stochasticity by directly optimizing a deterministic soft surrogate of the discrete l0 objective. Compared with prior approaches, DDP offers greater expressiveness, reduced train--test mismatch, and faster convergence. We apply our method to several dense and MoE models, including Qwen3-32B and Qwen3-30B-A3B, achieving a performance loss as small as 1% on downstream tasks while outperforming previous methods at 20% sparsity. We further demonstrate end-to-end inference speedups in realistic deployment settings with vLLM.
>
---
#### [new 097] SlowBA: An efficiency backdoor attack towards VLM-based GUI agents
- **分类: cs.CR; cs.CL; cs.CV**

- **简介: 该论文属于GUI安全任务，旨在解决VLM代理响应效率被攻击的问题。提出SlowBA攻击方法，通过诱导长推理链增加延迟，同时保持任务准确性。**

- **链接: [https://arxiv.org/pdf/2603.08316](https://arxiv.org/pdf/2603.08316)**

> **作者:** Junxian Li; Tu Lan; Haozhen Tan; Yan Meng; Haojin Zhu
>
> **备注:** 25 pages
>
> **摘要:** Modern vision-language-model (VLM) based graphical user interface (GUI) agents are expected not only to execute actions accurately but also to respond to user instructions with low latency. While existing research on GUI-agent security mainly focuses on manipulating action correctness, the security risks related to response efficiency remain largely unexplored. In this paper, we introduce SlowBA, a novel backdoor attack that targets the responsiveness of VLM-based GUI agents. The key idea is to manipulate response latency by inducing excessively long reasoning chains under specific trigger patterns. To achieve this, we propose a two-stage reward-level backdoor injection (RBI) strategy that first aligns the long-response format and then learns trigger-aware activation through reinforcement learning. In addition, we design realistic pop-up windows as triggers that naturally appear in GUI environments, improving the stealthiness of the attack. Extensive experiments across multiple datasets and baselines demonstrate that SlowBA can significantly increase response length and latency while largely preserving task accuracy. The attack remains effective even with a small poisoning ratio and under several defense settings. These findings reveal a previously overlooked security vulnerability in GUI agents and highlight the need for defenses that consider both action correctness and response efficiency. Code can be found in this https URL.
>
---
#### [new 098] Fine-Grained Table Retrieval Through the Lens of Complex Queries
- **分类: cs.IR; cs.AI; cs.CL; cs.DB**

- **简介: 该论文属于表格问答任务，旨在解决复杂查询下的表格检索问题。提出DCTR方法，通过细粒度查询分解和全局连通性感知提升检索效果。**

- **链接: [https://arxiv.org/pdf/2603.07146](https://arxiv.org/pdf/2603.07146)**

> **作者:** Wojciech Kosiuk; Xingyu Ji; Yeounoh Chung; Fatma Özcan; Madelon Hulsebos
>
> **摘要:** Enabling question answering over tables and databases in natural language has become a key capability in the democratization of insights from tabular data sources. These systems first require retrieval of data that is relevant to a given natural language query, for which several methods have been introduced. In this work we present and study a table retrieval mechanism devising fine-grained typed query decomposition and global connectivity-awareness (DCTR), to handle the challenges induced by open-domain question answering over relational databases in complex usage contexts. We evaluate the effectiveness of the two mechanisms through the lens of retrieval complexity which we measure along the axes of query- and data complexity. Our analyses over industry-aligned benchmarks illustrate the robustness of DCTR for highly composite queries and densely connected databases.
>
---
#### [new 099] Drift-to-Action Controllers: Budgeted Interventions with Online Risk Certificates
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Drift2Act，解决机器学习系统在分布漂移下的安全响应问题。通过在线风险认证实现受约束的决策，提升监控系统的安全性与效率。**

- **链接: [https://arxiv.org/pdf/2603.08578](https://arxiv.org/pdf/2603.08578)**

> **作者:** Ismail Lamaakal; Chaymae Yahyati; Khalid El Makkaoui; Ibrahim Ouahbi; Yassine Maleh
>
> **备注:** Published as a conference paper at CAO Workshop at ICLR 2026
>
> **摘要:** Deployed machine learning systems face distribution drift, yet most monitoring pipelines stop at alarms and leave the response underspecified under labeling, compute, and latency constraints. We introduce Drift2Act, a drift-to-action controller that treats monitoring as constrained decision-making with explicit safety. Drift2Act combines a sensing layer that maps unlabeled monitoring signals to a belief over drift types with an active risk certificate that queries a small set of delayed labels from a recent window to produce an anytime-valid upper bound $U_t(\delta)$ on current risk. The certificate gates operation: if $U_t(\delta) \le \tau$, the controller selects low-cost actions (e.g., recalibration or test-time adaptation); if $U_t(\delta) > \tau$, it activates abstain/handoff and escalates to rollback or retraining under cooldowns. In a realistic streaming protocol with label delay and explicit intervention costs, Drift2Act achieves near-zero safety violations and fast recovery at moderate cost on WILDS Camelyon17, DomainNet, and a controlled synthetic drift stream, outperforming alarm-only monitoring, adapt-always adaptation, schedule-based retraining, selective prediction alone, and an ablation without certification. Overall, online risk certification enables reliable drift response and reframes monitoring as decision-making with safety.
>
---
#### [new 100] GraphSkill: Documentation-Guided Hierarchical Retrieval-Augmented Coding for Complex Graph Reasoning
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于图推理任务，旨在解决现有方法在文档检索和代码调试上的不足。提出GraphSkill框架，利用文档层级结构和自调试机制提升代码生成质量。**

- **链接: [https://arxiv.org/pdf/2603.06620](https://arxiv.org/pdf/2603.06620)**

> **作者:** Fali Wang; Chenglin Weng; Xianren Zhang; Siyuan Hong; Hui Liu; Suhang Wang
>
> **备注:** Under review
>
> **摘要:** The growing demand for automated graph algorithm reasoning has attracted increasing attention in the large language model (LLM) community. Recent LLM-based graph reasoning methods typically decouple task descriptions from graph data, generate executable code augmented by retrieval from technical documentation, and refine the code through debugging. However, we identify two key limitations in existing approaches: (i) they treat technical documentation as flat text collections and ignore its hierarchical structure, leading to noisy retrieval that degrades code generation quality; and (ii) their debugging mechanisms focus primarily on runtime errors, yet ignore more critical logical errors. To address them, we propose {\method}, an \textit{agentic hierarchical retrieval-augmented coding framework} that exploits the document hierarchy through top-down traversal and early pruning, together with a \textit{self-debugging coding agent} that iteratively refines code using automatically generated small-scale test cases. To enable comprehensive evaluation of complex graph reasoning, we introduce a new dataset, {\dataset}, covering small-scale, large-scale, and composite graph reasoning tasks. Extensive experiments demonstrate that our method achieves higher task accuracy and lower inference cost compared to baselines\footnote{The code is available at \href{this https URL}{\textcolor{blue}{this https URL}}.}.
>
---
#### [new 101] Can Vision-Language Models Solve the Shell Game?
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型任务，旨在解决视频中物体跟踪问题。针对现有模型依赖静态特征、无法持续跟踪的问题，提出SGCoT方法，通过生成物体轨迹提升跟踪性能。**

- **链接: [https://arxiv.org/pdf/2603.08436](https://arxiv.org/pdf/2603.08436)**

> **作者:** Tiedong Liu; Wee Sun Lee
>
> **摘要:** Visual entity tracking is an innate cognitive ability in humans, yet it remains a critical bottleneck for Vision-Language Models (VLMs). This deficit is often obscured in existing video benchmarks by visual shortcuts. We introduce VET-Bench, a synthetic diagnostic testbed featuring visually identical objects that necessitate tracking exclusively through spatiotemporal continuity. Our experiments reveal that current state-of-the-art VLMs perform at or near chance level on VET-Bench, exposing a fundamental limitation: an over-reliance on static frame-level features and a failure to maintain entity representations over time. We provide a theoretical analysis drawing connections to the state-tracking problem, proving that fixed-depth transformer-based VLMs are fundamentally limited in tracking indistinguishable objects without intermediate supervision due to expressivity constraints. To address this, we propose Spatiotemporal Grounded Chain-of-Thought (SGCoT): generating object trajectories as explicit intermediate states. Leveraging Molmo2's object tracking ability, we elicit SGCoT reasoning by fine-tuning on synthesized text-only data for alignment. Our method achieves state-of-the-art accuracy exceeding 90% on VET-Bench, demonstrating that VLMs can reliably solve the video shell-game task end-to-end without external tools. Our code and data are available at this https URL .
>
---
#### [new 102] ArcLight: A Lightweight LLM Inference Architecture for Many-Core CPUs
- **分类: cs.DC; cs.CL**

- **简介: 该论文属于LLM推理任务，旨在解决Many-Core CPU平台计算潜力未被充分利用的问题。通过设计ArcLight架构，优化内存管理和并行策略，提升推理性能。**

- **链接: [https://arxiv.org/pdf/2603.07770](https://arxiv.org/pdf/2603.07770)**

> **作者:** Yuzhuang Xu; Xu Han; Yuxuan Li; Wanxiang Che
>
> **备注:** 13 figures, 1 table
>
> **摘要:** Although existing frameworks for large language model (LLM) inference on CPUs are mature, they fail to fully exploit the computation potential of many-core CPU platforms. Many-core CPUs are widely deployed in web servers and high-end networking devices, and are typically organized into multiple NUMA nodes that group cores and memory. Current frameworks largely overlook the substantial overhead of cross-NUMA memory access, limiting inference scalability and intelligence enabling on such platforms. To address this limitation, we build ArcLight, a lightweight LLM inference architecture designed from the ground up for many-core CPUs. ArcLight integrates efficient memory management and thread scheduling, and introduces finely controlled tensor parallelism to mitigate the cross-node memory access wall. Experimental results show that ArcLight significantly surpasses the performance ceiling of mainstream frameworks, achieving up to 46% higher inference throughput. Moreover, ArcLight maintains compatibility with arbitrary CPU devices. ArcLight is publicly available at this https URL.
>
---
#### [new 103] CoTJudger: A Graph-Driven Framework for Automatic Evaluation of Chain-of-Thought Efficiency and Redundancy in LRMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决LRMs中链式推理的冗余问题。提出CoTJudger框架，通过图结构分析提升推理效率评估的准确性。**

- **链接: [https://arxiv.org/pdf/2603.07078](https://arxiv.org/pdf/2603.07078)**

> **作者:** Siyi Li; Jiajun Shi; Shiwen Ni; Ge Zhang; Shuaimin Li; Shijian Wang; Zhoufutu Wen; Yizhi Li; Hamid Alinejad-Rokny; Jiaheng Liu; Min Yang; Wenhao Huang
>
> **摘要:** Large Reasoning Models (LRMs) have demonstrated strong performance by producing extended Chain-of-Thought (CoT) traces before answering. However, this paradigm often induces over-reasoning: redundant calculations and circular self-verification that increase computational cost without improving outcomes. Existing evaluations largely emphasize final accuracy or coarse token counts, and lack automated tools to separate essential logic from structural redundancy. We introduce CoTJudger, a graph-driven framework that quantifies reasoning efficiency by converting free-form CoTs into directed dependency graphs and extracting the Shortest Effective Path (SEP) needed to reach a correct solution. This yields an interpretable efficiency signal -- how much of a CoT is necessary versus structurally redundant -- that is comparable across models and tasks. Evaluating 21 LRMs, CoTJudger reveals pervasive redundancy and surfaces recurring failure modes, including verification obsession and compensatory redundancy. These results provide a practical metric for disentangling reasoning ability from computational waste, enabling more targeted evaluation and diagnosis of LRM efficiency.
>
---
#### [new 104] Quantifying Cross-Lingual Transfer in Paralinguistic Speech Tasks
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音情感识别任务，旨在解决跨语言迁移问题。通过构建CLTM分析不同语言间的迁移效果，揭示语言依赖性。**

- **链接: [https://arxiv.org/pdf/2603.08231](https://arxiv.org/pdf/2603.08231)**

> **作者:** Pol Buitrago; Oriol Pareras; Federico Costa; Javier Hernando
>
> **备注:** 6 pages, 5 figures, Submitted to Interspeech 2026
>
> **摘要:** Paralinguistic speech tasks are often considered relatively language-agnostic, as they rely on extralinguistic acoustic cues rather than lexical content. However, prior studies report performance degradation under cross-lingual conditions, indicating non-negligible language dependence. Still, these studies typically focus on isolated language pairs or task-specific settings, limiting comparability and preventing a systematic assessment of task-level language dependence. We introduce the Cross-Lingual Transfer Matrix (CLTM), a systematic method to quantify cross-lingual interactions between pairs of languages within a given task. We apply the CLTM to two paralinguistic tasks, gender identification and speaker verification, using a multilingual HuBERT-based encoder, to analyze how donor-language data affects target-language performance during fine-tuning. Our results reveal distinct transfer patterns across tasks and languages, reflecting systematic, language-dependent effects.
>
---
#### [new 105] Entropy-Aware On-Policy Distillation of Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型知识蒸馏任务，解决蒸馏过程中生成多样性不足和学习信号不稳定的问题。通过引入熵感知的正向KL散度，提升学生模型的多样性和对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.07079](https://arxiv.org/pdf/2603.07079)**

> **作者:** Woogyeol Jin; Taywon Min; Yongjin Yang; Swanand Ravindra Kadhe; Yi Zhou; Dennis Wei; Nathalie Baracaldo; Kimin Lee
>
> **备注:** 16 pages, 11 figures, preprint
>
> **摘要:** On-policy distillation is a promising approach for transferring knowledge between language models, where a student learns from dense token-level signals along its own trajectories. This framework typically uses reverse KL divergence, encouraging the student to match the teacher's high-confidence predictions. However, we show that the mode-seeking property of reverse KL reduces generation diversity and yields unstable learning signals when the teacher distribution has high entropy. To address this, we introduce Entropy-Aware On-Policy Distillation. Our key idea is augmenting the standard reverse KL objective with forward KL when teacher entropy is high, capturing the full range of plausible outputs while retaining precise imitation elsewhere. It balances mode-seeking precision with mode-covering robustness without sacrificing on-policy training efficiency. Experiments show that our method maintains generation diversity (sustained token-level entropy) and improves student-teacher alignment (lower forward KL on high-entropy tokens). Across six math reasoning benchmarks, this yields Pass@8 accuracy gains of +1.37 for Qwen3-0.6B-Base, +2.39 for Qwen3-1.7B-Base, and +5.05 for Qwen3-4B-Base compared to baseline on-policy distillation methods. These results demonstrate that accounting for teacher uncertainty is essential for maintaining diversity and achieving effective knowledge transfer.
>
---
#### [new 106] \$OneMillion-Bench: How Far are Language Agents from Human Experts?
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出\$OneMillion-Bench，用于评估语言代理在法律、金融等专业领域的表现，解决现有基准不贴近真实需求的问题。**

- **链接: [https://arxiv.org/pdf/2603.07980](https://arxiv.org/pdf/2603.07980)**

> **作者:** Qianyu Yang; Yang Liu; Jiaqi Li; Jun Bai; Hao Chen; Kaiyuan Chen; Tiliang Duan; Jiayun Dong; Xiaobo Hu; Zixia Jia; Yang Liu; Tao Peng; Yixin Ren; Ran Tian; Zaiyuan Wang; Yanglihong Xiao; Gang Yao; Lingyue Yin; Ge Zhang; Chun Zhang; Jianpeng Jiao; Zilong Zheng; Yuan Gong
>
> **备注:** 39 pages, 9 figures, 8 tables
>
> **摘要:** As language models (LMs) evolve from chat assistants to long-horizon agents capable of multi-step reasoning and tool use, existing benchmarks remain largely confined to structured or exam-style tasks that fall short of real-world professional demands. To this end, we introduce \$OneMillion-Bench \$OneMillion-Bench, a benchmark of 400 expert-curated tasks spanning Law, Finance, Industry, Healthcare, and Natural Science, built to evaluate agents across economically consequential scenarios. Unlike prior work, the benchmark requires retrieving authoritative sources, resolving conflicting evidence, applying domain-specific rules, and making constraint decisions, where correctness depends as much on the reasoning process as the final answer. We adopt a rubric-based evaluation protocol scoring factual accuracy, logical coherence, practical feasibility, and professional compliance, focused on expert-level problems to ensure meaningful differentiation across agents. Together, \$OneMillion-Bench provides a unified testbed for assessing agentic reliability, professional depth, and practical readiness in domain-intensive scenarios.
>
---
#### [new 107] Supporting Artifact Evaluation with LLMs: A Study with Published Security Research Papers
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于支持可重复性评估任务，旨在解决人工检查耗时且难以扩展的问题。通过LLMs实现自动评估、环境构建和方法缺陷检测。**

- **链接: [https://arxiv.org/pdf/2603.06862](https://arxiv.org/pdf/2603.06862)**

> **作者:** David Heye; Karl Kindermann; Robin Decker; Johannes Lohmöller; Anastasiia Belova; Sandra Geisler; Klaus Wehrle; Jan Pennekamp
>
> **摘要:** Artifact Evaluation (AE) is essential for ensuring the transparency and reliability of research, closing the gap between exploratory work and real-world deployment is particularly important in cybersecurity, particularly in IoT and CPSs, where large-scale, heterogeneous, and privacy-sensitive data meet safety-critical actuation. Yet, manual reproducibility checks are time-consuming and do not scale with growing submission volumes. In this work, we demonstrate that Large Language Models (LLMs) can provide powerful support for AE tasks: (i) text-based reproducibility rating, (ii) autonomous sandboxed execution environment preparation, and (iii) assessment of methodological pitfalls. Our reproducibility-assessment toolkit yields an accuracy of over 72% and autonomously sets up execution environments for 28% of runnable cybersecurity artifacts. Our automated pitfall assessment detects seven prevalent pitfalls with high accuracy ($F_1$ > 92%). Hence, the toolkit significantly reduces reviewer effort and, when integrated into established AE processes, could incentivize authors to submit higher-quality and more reproducible artifacts. IoT, CPS, and cybersecurity conferences and workshops may integrate the toolkit into their peer-review processes to support reviewers' decisions on awarding artifact badges, improving the overall sustainability of the process.
>
---
#### [new 108] LycheeCluster: Efficient Long-Context Inference with Structure-Aware Chunking and Hierarchical KV Indexing
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型的长文本处理任务，旨在解决KV缓存管理效率低的问题。提出LycheeCluster方法，通过结构感知分块和分层索引提升推理速度。**

- **链接: [https://arxiv.org/pdf/2603.08453](https://arxiv.org/pdf/2603.08453)**

> **作者:** Dongfang Li; Zixuan Liu; Gang Lin; Baotian Hu; Min Zhang
>
> **备注:** 17 pages, 12 figures
>
> **摘要:** The quadratic complexity of the attention mechanism and the substantial memory footprint of the Key-Value (KV) cache present severe computational and memory challenges for Large Language Models (LLMs) processing long contexts. Existing retrieval-based methods often compromise semantic integrity through fixed-size chunking and suffer from inefficient linear scanning. In this paper, we propose LycheeCluster, a novel method for efficient KV cache management. LycheeCluster preserves local semantic coherence via boundary-aware chunking and constructs a recursive hierarchical index rooted in the triangle inequality. This design transforms cache retrieval from a linear scan into a theoretically bounded, logarithmic-time pruning process, while a lazy update strategy supports efficient streaming generation. Experiments demonstrate that LycheeCluster achieves up to a 3.6x end-to-end inference speedup with negligible degradation in model performance, outperforming state-of-the-art KV cache management methods (e.g., Quest, ClusterKV). We will release our code and kernels after publication.
>
---
#### [new 109] TimeSpot: Benchmarking Geo-Temporal Understanding in Vision-Language Models in Real-World Settings
- **分类: cs.CV; cs.CL; cs.ET; cs.MM; cs.RO**

- **简介: 该论文属于视觉-语言模型的地理时间理解任务，旨在解决模型在真实场景中对时空属性推理能力不足的问题。作者构建了TimeSpot基准数据集，用于评估和提升模型的geo-temporal推理能力。**

- **链接: [https://arxiv.org/pdf/2603.06687](https://arxiv.org/pdf/2603.06687)**

> **作者:** Azmine Toushik Wasi; Shahriyar Zaman Ridoy; Koushik Ahamed Tonmoy; Kinga Tshering; S. M. Muhtasimul Hasan; Wahid Faisal; Tasnim Mohiuddin; Md Rizwan Parvez
>
> **备注:** 66 Pages. In Review
>
> **摘要:** Geo-temporal understanding, the ability to infer location, time, and contextual properties from visual input alone, underpins applications such as disaster management, traffic planning, embodied navigation, world modeling, and geography education. Although recent vision-language models (VLMs) have advanced image geo-localization using cues like landmarks and road signs, their ability to reason about temporal signals and physically grounded spatial cues remains limited. To address this gap, we introduce TimeSpot, a benchmark for evaluating real-world geo-temporal reasoning in VLMs. TimeSpot comprises 1,455 ground-level images from 80 countries and requires structured prediction of temporal attributes (season, month, time of day, daylight phase) and geographic attributes (continent, country, climate zone, environment type, latitude-longitude) directly from visual evidence. It also includes spatial-temporal reasoning tasks that test physical plausibility under real-world uncertainty. Evaluations of state-of-the-art open- and closed-source VLMs show low performance, particularly for temporal inference. While supervised fine-tuning yields improvements, results remain insufficient, highlighting the need for new methods to achieve robust, physically grounded geo-temporal understanding. TimeSpot is available at: this https URL.
>
---
#### [new 110] Scalable Training of Mixture-of-Experts Models with Megatron Core
- **分类: cs.DC; cs.CL; cs.LG**

- **简介: 该论文属于深度学习模型训练任务，解决MoE模型扩展中的系统挑战，通过优化内存、通信和计算等多方面实现高效可扩展训练。**

- **链接: [https://arxiv.org/pdf/2603.07685](https://arxiv.org/pdf/2603.07685)**

> **作者:** Zijie Yan; Hongxiao Bai; Xin Yao; Dennis Liu; Tong Liu; Hongbin Liu; Pingtian Li; Evan Wu; Shiqing Fan; Li Tao; Robin Zhang; Yuzhong Wang; Shifang Xu; Jack Chang; Xuwen Chen; Kunlun Li; Yan Bai; Gao Deng; Nan Zheng; Vijay Anand Korthikanti; Abhinav Khattar; Ethan He; Soham Govande; Sangkug Lym; Zhongbo Zhu; Qi Zhang; Haochen Yuan; Xiaowei Ren; Deyu Fu; Tailai Ma; Shunkang Zhang; Jiang Shao; Ray Wang; Santosh Bhavani; Xipeng Li; Chandler Zhou; David Wu; Yingcan Wei; Ashwath Aithal; Michael Andersch; Mohammad Shoeybi; Jiajie Yao; June Yang
>
> **备注:** Technical Report. 88 pages. 42 figures
>
> **摘要:** Scaling Mixture-of-Experts (MoE) training introduces systems challenges absent in dense models. Because each token activates only a subset of experts, this sparsity allows total parameters to grow much faster than per-token computation, creating coupled constraints across memory, communication, and computation. Optimizing one dimension often shifts pressure to another, demanding co-design across the full system stack. We address these challenges for MoE training through integrated optimizations spanning memory (fine-grained recomputation, offloading, etc.), communication (optimized dispatchers, overlapping, etc.), and computation (Grouped GEMM, fusions, CUDA Graphs, etc.). The framework also provides Parallel Folding for flexible multi-dimensional parallelism, low-precision training support for FP8 and NVFP4, and efficient long-context training. On NVIDIA GB300 and GB200, it achieves 1,233/1,048 TFLOPS/GPU for DeepSeek-V3-685B and 974/919 TFLOPS/GPU for Qwen3-235B. As a performant, scalable, and production-ready open-source solution, it has been used across academia and industry for training MoE models ranging from billions to trillions of parameters on clusters scaling up to thousands of GPUs. This report explains how these techniques work, their trade-offs, and their interactions at the systems level, providing practical guidance for scaling MoE models with Megatron Core.
>
---
#### [new 111] Reject, Resample, Repeat: Understanding Parallel Reasoning in Language Model Inference
- **分类: cs.LG; cs.AI; cs.CL; math.ST; stat.ML**

- **简介: 该论文属于语言模型推理任务，研究如何通过粒子过滤方法优化样本采样与计算成本的平衡，提出理论分析和算法改进，揭示其局限性。**

- **链接: [https://arxiv.org/pdf/2603.07887](https://arxiv.org/pdf/2603.07887)**

> **作者:** Noah Golowich; Fan Chen; Dhruv Rohatgi; Raghav Singhal; Carles Domingo-Enrich; Dylan J. Foster; Akshay Krishnamurthy
>
> **摘要:** Inference-time methods that aggregate and prune multiple samples have emerged as a powerful paradigm for steering large language models, yet we lack any principled understanding of their accuracy-cost tradeoffs. In this paper, we introduce a route to rigorously study such approaches using the lens of *particle filtering* algorithms such as Sequential Monte Carlo (SMC). Given a base language model and a *process reward model* estimating expected terminal rewards, we ask: *how accurately can we sample from a target distribution given some number of process reward evaluations?* Theoretically, we identify (1) simple criteria enabling non-asymptotic guarantees for SMC; (2) algorithmic improvements to SMC; and (3) a fundamental limit faced by all particle filtering methods. Empirically, we demonstrate that our theoretical criteria effectively govern the *sampling error* of SMC, though not necessarily its final *accuracy*, suggesting that theoretical perspectives beyond sampling may be necessary.
>
---
#### [new 112] OfficeQA Pro: An Enterprise Benchmark for End-to-End Grounded Reasoning
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出OfficeQA Pro，一个用于评估AI在企业级文档上进行精准推理的基准。解决多文档、结构化与非结构化数据上的 grounded reasoning 问题，通过构建大规模数据集并测试前沿模型性能。**

- **链接: [https://arxiv.org/pdf/2603.08655](https://arxiv.org/pdf/2603.08655)**

> **作者:** Krista Opsahl-Ong; Arnav Singhvi; Jasmine Collins; Ivan Zhou; Cindy Wang; Ashutosh Baheti; Owen Oertell; Jacob Portes; Sam Havens; Erich Elsen; Michael Bendersky; Matei Zaharia; Xing Chen
>
> **备注:** 24 pages, 16 figures. Introduces the OfficeQA Pro benchmark for grounded reasoning over enterprise documents
>
> **摘要:** We introduce OfficeQA Pro, a benchmark for evaluating AI agents on grounded, multi-document reasoning over a large and heterogeneous document corpus. The corpus consists of U.S. Treasury Bulletins spanning nearly 100 years, comprising 89,000 pages and over 26 million numerical values. OfficeQA Pro consists of 133 questions that require precise document parsing, retrieval, and analytical reasoning across both unstructured text and tabular data. Frontier LLMs including Claude Opus 4.6, GPT-5.4, and Gemini 3.1 Pro Preview achieve less than 5% accuracy on OfficeQA Pro when relying on parametric knowledge, and less than 12% with additional access to the web. When provided directly with the document corpus, frontier agents still struggle on over half of questions, scoring 34.1% on average. We find that providing agents with a structured document representation produced by Databricks' ai_parse_document yields a 16.1% average relative performance gain across agents. We conduct additional ablations to study the effects of model selection, table representation, retrieval strategy, and test-time scaling on performance. Despite these improvements, significant headroom remains before agents can be considered reliable at enterprise-grade grounded reasoning.
>
---
#### [new 113] Orion: Characterizing and Programming Apple's Neural Engine for LLM Training and Inference
- **分类: cs.LG; cs.AR; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决Apple ANE未被充分利用的问题。工作包括构建Orion系统，实现直接ANP编程、编译优化及稳定训练，提升LLM在设备上的推理与训练效率。**

- **链接: [https://arxiv.org/pdf/2603.06728](https://arxiv.org/pdf/2603.06728)**

> **作者:** Ramchand Kumaresan
>
> **摘要:** Over two billion Apple devices ship with a Neural Processing Unit (NPU) - the Apple Neural Engine (ANE) - yet this accelerator remains largely unused for large language model workloads. CoreML, Apple's public ML framework, imposes opaque abstractions that prevent direct ANE programming and do not support on-device training. We present Orion, to our knowledge the first open end-to-end system that combines direct ANE execution, a compiler pipeline, and stable multi-step training with checkpoint resume in a single native runtime, bypassing CoreML entirely via Apple's private _ANEClient and _ANECompiler APIs. Building on prior characterization work by maderix, we extend public knowledge of ANE constraints to a catalog of 20 restrictions on MIL IR programs, memory layout, compilation limits, and numerical behavior, including 14 previously undocumented constraints discovered during Orion development. Orion includes a compiler that lowers a graph IR through five optimization passes to ANE-native MIL and a runtime that manages IOSurface-backed zero-copy tensor I/O, program caching, and delta compilation for weight updates. Because the ANE bakes weights at compile time, naive training normally requires full recompilation per step (~4.2 s). We show that compiled programs can instead be updated by unloading, patching weight files, and reloading, bypassing ANECCompile() and reducing recompilation from 4,200 ms to 494 ms per step (8.5x), yielding a 3.8x training speedup. On an M4 Max, Orion achieves 170+ tokens/s for GPT-2 124M inference and demonstrates stable training of a 110M-parameter transformer on TinyStories for 1,000 steps in 22 minutes with zero NaN occurrences. We also present LoRA adapter-as-input, enabling hot-swap of adapters via IOSurface inputs without recompilation.
>
---
#### [new 114] Generalization in Online Reinforcement Learning for Mobile Agents
- **分类: cs.CV; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于在线强化学习任务，旨在提升移动代理的泛化能力。针对缺乏基准和系统的问题，提出AndroidWorld-Generalization基准和GRPO训练系统，实验显示RL在未见任务上表现优于监督微调。**

- **链接: [https://arxiv.org/pdf/2603.07432](https://arxiv.org/pdf/2603.07432)**

> **作者:** Li Gu; Zihuan Jiang; Zhixiang Chi; Huan Liu; Ziqiang Wang; Yuanhao Yu; Glen Berseth; Yang Wang
>
> **摘要:** Graphical user interface (GUI)-based mobile agents automate digital tasks on mobile devices by interpreting natural-language instructions and interacting with the screen. While recent methods apply reinforcement learning (RL) to train vision-language-model(VLM) agents in interactive environments with a primary focus on performance, generalization remains underexplored due to the lack of standardized benchmarks and open-source RL systems. In this work, we formalize the problem as a Contextual Markov Decision Process (CMDP) and introduce \textbf{AndroidWorld-Generalization}, a benchmark with three increasingly challenging regimes for evaluating zero-shot generalization to unseen task instances, templates, and applications. We further propose an RL training system that integrates Group Relative Policy Optimization (GRPO) with a scalable rollout collection system, consisting of containerized infrastructure and asynchronous execution % , and error recovery to support reliable and efficient training. Experiments on AndroidWorld-Generalization show that RL enables a 7B-parameter VLM agent to surpass supervised fine-tuning baselines, yielding a 26.1\% improvement on unseen instances but only limited gains on unseen templates (15.7\%) and apps (8.3\%), underscoring the challenges of generalization. As a preliminary step, we demonstrate that few-shot adaptation at test-time improves performance on unseen apps, motivating future research in this direction. To support reproducibility and fair comparison, we open-source the full RL training system, including the environment, task suite, models, prompt configurations, and the underlying infrastructure \footnote{this https URL}.
>
---
#### [new 115] How Attention Sinks Emerge in Large Language Models: An Interpretability Perspective
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型解释性研究，旨在揭示大语言模型中注意力陷阱的形成机制。工作包括发现P0 Sink Circuit，解释首词注意力集中现象。**

- **链接: [https://arxiv.org/pdf/2603.06591](https://arxiv.org/pdf/2603.06591)**

> **作者:** Runyu Peng; Ruixiao Li; Mingshu Chen; Yunhua Zhou; Qipeng Guo; Xipeng Qiu
>
> **摘要:** Large Language Models (LLMs) often allocate disproportionate attention to specific tokens, a phenomenon commonly referred to as the attention sink. While such sinks are generally considered detrimental, prior studies have identified a notable exception: the model's consistent emphasis on the first token of the input sequence. This structural bias can influence a wide range of downstream applications and warrants careful consideration. Despite its prevalence, the precise mechanisms underlying the emergence and persistence of attention sinks remain poorly understood. In this work, we trace the formation of attention sinks around the first token of the input. We identify a simple mechanism, referred to as the P0 Sink Circuit, that enables the model to recognize token at position zero and induce an attention sink within two transformer blocks, without relying on any semantic information. This mechanism serves as the basis for the attention sink on position zero. Furthermore, by analyzing training traces from a 30B A3B MoE model trained from scratch, we find that this mechanism emerges early in training and becomes increasingly concentrated in the first two layers, suggesting a possible signal for tracking pre training convergence states.
>
---
#### [new 116] Know When You're Wrong: Aligning Confidence with Correctness for LLM Error Detection
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型误差检测任务，旨在解决LLM不确定性度量不足的问题。通过引入归一化置信度分数和自我评估框架，提升错误检测能力。**

- **链接: [https://arxiv.org/pdf/2603.06604](https://arxiv.org/pdf/2603.06604)**

> **作者:** Xie Xiaohu; Liu Xiaohu; Yao Benjamin
>
> **摘要:** As large language models (LLMs) are increasingly deployed in critical decision-making systems, the lack of reliable methods to measure their uncertainty presents a fundamental trustworthiness risk. We introduce a normalized confidence score based on output anchor token probabilities: classification labels for structured tasks and self-evaluation responses (Yes/No) for open-ended generation. This enables direct detection of errors and hallucinations with minimal overhead and without external validation. We make three key contributions. First, we propose a normalized confidence score and self-evaluation framework that exposes reliable confidence estimates for error detection across seven diverse benchmark tasks and five LLMs of varying architectures and sizes. Second, our theoretical analysis reveals that supervised fine-tuning (SFT) yields well-calibrated confidence through maximum-likelihood estimation, whereas reinforcement learning methods (PPO, GRPO) and DPO induce overconfidence via reward exploitation. Third, we propose post-RL SFT with self-distillation to restore confidence reliability in RL-trained models. Empirical results demonstrated that SFT improved average confidence-correctness AUROC from 0.806 to 0.879 and reduced calibration error from 0.163 to 0.034 on Qwen3-4B, while GRPO and DPO degraded confidence reliability. We demonstrated practical value through adaptive retrieval-augmented generation (RAG) that selectively retrieves context when the model lacks confidence, using only 58\% of retrieval operations to recover 95\% of the maximum achievable accuracy gain on TriviaQA
>
---
#### [new 117] A prospective clinical feasibility study of a conversational diagnostic AI in an ambulatory primary care clinic
- **分类: cs.HC; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于医疗诊断任务，旨在评估对话式AI在真实临床环境中的可行性与安全性。研究通过实验验证AI在病史采集和诊断建议方面的表现，探索其临床应用潜力。**

- **链接: [https://arxiv.org/pdf/2603.08448](https://arxiv.org/pdf/2603.08448)**

> **作者:** Peter Brodeur; Jacob M. Koshy; Anil Palepu; Khaled Saab; Ava Homiar; Roma Ruparel; Charles Wu; Ryutaro Tanno; Joseph Xu; Amy Wang; David Stutz; Hannah M. Ferrera; David Barrett; Lindsey Crowley; Jihyeon Lee; Spencer E. Rittner; Ellery Wulczyn; Selena K. Zhang; Elahe Vedadi; Christine G. Kohn; Kavita Kulkarni; Vinay Kadiyala; Sara Mahdavi; Wendy Du; Jessica Williams; David Feinbloom; Renee Wong; Tao Tu; Petar Sirkovic; Alessio Orlandi; Christopher Semturs; Yun Liu; Juraj Gottweis; Dale R. Webster; Joëlle Barral; Katherine Chou; Pushmeet Kohli; Avinatan Hassidim; Yossi Matias; James Manyika; Rob Fields; Jonathan X. Li; Marc L. Cohen; Vivek Natarajan; Mike Schaekermann; Alan Karthikesalingam; Adam Rodman
>
> **摘要:** Large language model (LLM)-based AI systems have shown promise for patient-facing diagnostic and management conversations in simulated settings. Translating these systems into clinical practice requires assessment in real-world workflows with rigorous safety oversight. We report a prospective, single-arm feasibility study of an LLM-based conversational AI, the Articulate Medical Intelligence Explorer (AMIE), conducting clinical history taking and presentation of potential diagnoses for patients to discuss with their provider at urgent care appointments at a leading academic medical center. 100 adult patients completed an AMIE text-chat interaction up to 5 days before their appointment. We sought to assess the conversational safety and quality, patient and clinician experience, and clinical reasoning capabilities compared to primary care providers (PCPs). Human safety supervisors monitored all patient-AMIE interactions in real time and did not need to intervene to stop any consultations based on pre-defined criteria. Patients reported high satisfaction and their attitudes towards AI improved after interacting with AMIE (p < 0.001). PCPs found AMIE's output useful with a positive impact on preparedness. AMIE's differential diagnosis (DDx) included the final diagnosis, per chart review 8 weeks post-encounter, in 90% of cases, with 75% top-3 accuracy. Blinded assessment of AMIE and PCP DDx and management (Mx) plans suggested similar overall DDx and Mx plan quality, without significant differences for DDx (p = 0.6) and appropriateness and safety of Mx (p = 0.1 and 1.0, respectively). PCPs outperformed AMIE in the practicality (p = 0.003) and cost effectiveness (p = 0.004) of Mx. While further research is needed, this study demonstrates the initial feasibility, safety, and user acceptance of conversational AI in a real-world setting, representing crucial steps towards clinical translation.
>
---
#### [new 118] Agentic Critical Training
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于语言模型训练任务，解决代理缺乏自主反思的问题。提出ACT方法，通过强化学习让模型自主判断动作优劣，提升代理性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.08706](https://arxiv.org/pdf/2603.08706)**

> **作者:** Weize Liu; Minghui Liu; Sy-Tuyen Ho; Souradip Chakraborty; Xiyao Wang; Furong Huang
>
> **备注:** Project page: this https URL
>
> **摘要:** Training large language models (LLMs) as autonomous agents often begins with imitation learning, but it only teaches agents what to do without understanding why: agents never contrast successful actions against suboptimal alternatives and thus lack awareness of action quality. Recent approaches attempt to address this by introducing self-reflection supervision derived from contrasts between expert and alternative actions. However, the training paradigm fundamentally remains imitation learning: the model imitates pre-constructed reflection text rather than learning to reason autonomously. We propose Agentic Critical Training (ACT), a reinforcement learning paradigm that trains agents to identify the better action among alternatives. By rewarding whether the model's judgment is correct, ACT drives the model to autonomously develop reasoning about action quality, producing genuine self-reflection rather than imitating it. Across three challenging agent benchmarks, ACT consistently improves agent performance when combined with different post-training methods. It achieves an average improvement of 5.07 points over imitation learning and 4.62 points over reinforcement learning. Compared to approaches that inject reflection capability through knowledge distillation, ACT also demonstrates clear advantages, yielding an average improvement of 2.42 points. Moreover, ACT enables strong out-of-distribution generalization on agentic benchmarks and improves performance on general reasoning benchmarks without any reasoning-specific training data, highlighting the value of our method. These results suggest that ACT is a promising path toward developing more reflective and capable LLM agents.
>
---
#### [new 119] DualTurn: Learning Turn-Taking from Dual-Channel Generative Speech Pretraining
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出DualTurn模型，解决语音对话中自然轮换与工具调用的矛盾问题。通过双通道生成预训练，实现无标签的对话动态学习，并预测可解释的轮换信号，提升对话流畅性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.08216](https://arxiv.org/pdf/2603.08216)**

> **作者:** Shangeth Rajaa
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Speech-to-speech models handle turn-taking naturally but offer limited support for tool-calling or complex reasoning, while production ASR-LLM-TTS voice pipelines offer these capabilities but rely on silence timeouts, which lead to unnatural turn-taking. We present DualTurn, which narrows this gap through generative pretraining on dual-channel conversational audio. The model generates both speakers' future audio autoregressively, implicitly learning conversational dynamics without any labels, and is then fine-tuned to predict interpretable turn-taking signals that map directly to agent actions. DualTurn monitors both channels continuously, anticipating turn boundaries and producing five agent actions. On standard benchmarks, DualTurn (0.5B) outperforms both VAP on agent action prediction (wF1 0.633 vs. 0.389) and a 3.1B audio-text model on word-level turn prediction (AUC 0.930 vs. 0.880), while anticipating turn boundaries earlier with fewer interruptions.
>
---
#### [new 120] Breaking Training Bottlenecks: Effective and Stable Reinforcement Learning for Coding Models
- **分类: cs.LG; cs.CL; cs.GL**

- **简介: 该论文属于代码生成任务，解决训练瓶颈问题。提出MicroCoder-GRPO方法，提升模型性能与多样性，并发布新数据集和评估框架。**

- **链接: [https://arxiv.org/pdf/2603.07777](https://arxiv.org/pdf/2603.07777)**

> **作者:** Zongqian Li; Shaohan Huang; Zewen Chi; Yixuan Su; Lexin Zhou; Li Dong; Nigel Collier; Furu Wei
>
> **摘要:** Modern code generation models exhibit longer outputs, accelerated capability growth, and changed training dynamics, rendering traditional training methodologies, algorithms, and datasets ineffective for improving their performance. To address these training bottlenecks, we propose MicroCoder-GRPO, an improved Group Relative Policy Optimization approach with three innovations: conditional truncation masking to improve long output potential while maintaining training stability, diversity-determined temperature selection to maintain and encourage output diversity, and removal of KL loss with high clipping ratios to facilitate solution diversity. MicroCoder-GRPO achieves up to 17.6% relative improvement over strong baselines on LiveCodeBench v6, with more pronounced gains under extended context evaluation. Additionally, we release MicroCoder-Dataset, a more challenging training corpus that achieves 3x larger performance gains than mainstream datasets on LiveCodeBench v6 within 300 training steps, and MicroCoder-Evaluator, a robust framework with approximately 25% improved evaluation accuracy and around 40% faster execution. Through comprehensive analysis across more than thirty controlled experiments, we reveal 34 training insights across seven main aspects, demonstrating that properly trained models can achieve competitive performance with larger counterparts.
>
---
#### [new 121] The Third Ambition: Artificial Intelligence and the Science of Human Behavior
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文探讨将大语言模型作为研究人类行为的科学工具，属于计算社会科学任务，旨在解决如何利用AI分析人类文化与道德推理的问题。**

- **链接: [https://arxiv.org/pdf/2603.07329](https://arxiv.org/pdf/2603.07329)**

> **作者:** W. Russell Neuman; Chad Coleman
>
> **摘要:** Contemporary artificial intelligence research has been organized around two dominant ambitions: productivity, which treats AI systems as tools for accelerating work and economic output, and alignment, which focuses on ensuring that increasingly capable systems behave safely and in accordance with human values. This paper articulates and develops a third, emerging ambition: the use of large language models (LLMs) as scientific instruments for studying human behavior, culture, and moral reasoning. Trained on unprecedented volumes of human-produced text, LLMs encode large-scale regularities in how people argue, justify, narrate, and negotiate norms across social domains. We argue that these models can be understood as condensates of human symbolic behavior, compressed, generative representations that render patterns of collective discourse computationally accessible. The paper situates this third ambition within long-standing traditions of computational social science, content analysis, survey research, and comparative-historical inquiry, while clarifying the epistemic limits of treating model output as evidence. We distinguish between base models and fine-tuned systems, showing how alignment interventions can systematically reshape or obscure the cultural regularities learned during pretraining, and we identify instruct-only and modular adaptation regimes as pragmatic compromises for behavioral research. We review emerging methodological approaches including prompt-based experiments, synthetic population sampling, comparative-historical modeling, and ablation studies and show how each maps onto familiar social-scientific designs while operating at unprecedented scale.
>
---
#### [new 122] Image Generation Models: A Technical History
- **分类: cs.CV; cs.AI; cs.CL; cs.GR**

- **简介: 该论文属于图像生成任务，旨在系统梳理各类生成模型，解决模型碎片化问题，总结其技术原理、优化方法及应用挑战。**

- **链接: [https://arxiv.org/pdf/2603.07455](https://arxiv.org/pdf/2603.07455)**

> **作者:** Rouzbeh Shirvani
>
> **摘要:** Image generation has advanced rapidly over the past decade, yet the literature seems fragmented across different models and application domains. This paper aims to offer a comprehensive survey of breakthrough image generation models, including variational autoencoders (VAEs), generative adversarial networks (GANs), normalizing flows, autoregressive and transformer-based generators, and diffusion-based methods. We provide a detailed technical walkthrough of each model type, including their underlying objectives, architectural building blocks, and algorithmic training steps. For each model type, we present the optimization techniques as well as common failure modes and limitations. We also go over recent developments in video generation and present the research works that made it possible to go from still frames to high quality videos. Lastly, we cover the growing importance of robustness and responsible deployment of these models, including deepfake risks, detection, artifacts, and watermarking.
>
---
#### [new 123] KCoEvo: A Knowledge Graph Augmented Framework for Evolutionary Code Generation
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于代码生成任务，解决API变更导致的代码过时问题。提出KCoEvo框架，通过知识图谱增强，提升代码迁移准确性与成功率。**

- **链接: [https://arxiv.org/pdf/2603.07581](https://arxiv.org/pdf/2603.07581)**

> **作者:** Jiazhen Kang; Yuchen Lu; Chen Jiang; Jinrui Liu; Tianhao Zhang; Bo Jiang; Ningyuan Sun; Tongtong Wu; Guilin Qi
>
> **备注:** Accepted to the DASFAA 2026 Industry Track
>
> **摘要:** Code evolution is inevitable in modern software development. Changes to third-party APIs frequently break existing code and complicate maintenance, posing practical challenges for developers. While large language models (LLMs) have shown promise in code generation, they struggle to reason without a structured representation of these evolving relationships, often leading them to produce outdated APIs or invalid outputs. In this work, we propose a knowledge graph-augmented framework that decomposes the migration task into two synergistic stages: evolution path retrieval and path-informed code generation. Our approach constructs static and dynamic API graphs to model intra-version structures and cross-version transitions, enabling structured reasoning over API evolution. Both modules are trained with synthetic supervision automatically derived from real-world API diffs, ensuring scalability and minimal human effort. Extensive experiments across single-package and multi-package benchmarks demonstrate that our framework significantly improves migration accuracy, controllability, and execution success over standard LLM baselines. The source code and datasets are available at: this https URL.
>
---
#### [new 124] DistillGuard: Evaluating Defenses Against LLM Knowledge Distillation
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于模型安全任务，旨在解决LLM知识蒸馏攻击的防御问题。提出DistillGuard框架，评估多种防御方法的有效性，发现现有方法在多数任务中效果有限。**

- **链接: [https://arxiv.org/pdf/2603.07835](https://arxiv.org/pdf/2603.07835)**

> **作者:** Bo Jiang
>
> **摘要:** Knowledge distillation from proprietary LLM APIs poses a growing threat to model providers, yet defenses against this attack remain fragmented and unevaluated. We present DistillGuard, a framework for systematically evaluating output-level defenses against LLM knowledge distillation. We introduce a taxonomy of three defense categories -- output perturbation, data poisoning, and information throttling -- and evaluate nine defense configurations using a standardized pipeline with Qwen3-14B as teacher and Qwen2.5-7B-Instruct as student across three benchmarks (MATH-500, HumanEval+, MT-Bench). Our results reveal that, in a same-family distillation setting against a naive attacker, most output-level defenses are surprisingly ineffective: paraphrasing-based perturbation barely degrades distilled student quality, and data poisoning primarily impairs conversational fluency while leaving task-specific capabilities intact. Only chain-of-thought removal substantially impairs mathematical reasoning (31.4\% vs.\ 67.8\% baseline), though code generation remains unaffected. These findings demonstrate that the effectiveness of distillation defenses is highly task-dependent and that current output-level approaches are insufficient to broadly prevent knowledge theft.
>
---
#### [new 125] Chart-RL: Generalized Chart Comprehension via Reinforcement Learning with Verifiable Rewards
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Chart-RL方法，解决图表理解任务中的泛化问题，通过强化学习提升视觉语言模型的图表问答能力。**

- **链接: [https://arxiv.org/pdf/2603.06958](https://arxiv.org/pdf/2603.06958)**

> **作者:** Xin Zhang; Xingyu Li; Rongguang Wang; Ruizhong Miao; Zheng Wang; Dan Roth; Chenyang Li
>
> **摘要:** Accurate chart comprehension represents a critical challenge in advancing multimodal learning systems, as extensive information is compressed into structured visual representations. However, existing vision-language models (VLMs) frequently struggle to generalize on unseen charts because it requires abstract, symbolic, and quantitative reasoning over structured visual representations. In this work, we introduce Chart-RL, an effective reinforcement learning (RL) method that employs mathematically verifiable rewards to enhance chart question answering in VLMs. Our experiments demonstrate that Chart-RL consistently outperforms supervised fine-tuning (SFT) across different chart understanding benchmarks, achieving relative improvements of 16.7% on MutlChartQA, and 11.5% on ChartInsights. We conduct robustness analysis, where Chart-RL achieves enhanced performance in 18 of 25 perturbed chart categories, demonstrating strong consistency and reasoning capability across visual variations. Furthermore, we demonstrate that task difficulty and inherent complexity are more critical than data quantity in RL training. For instance, Chart-RL trained on merely 10 complex chart-query examples significantly outperforms models trained on over 6,000 simple examples. Additionally, training on challenging reasoning tasks not only improves in-domain generalization relative to simpler tasks, but also facilitate strong transfer to out-of-domain visual mathematical problems.
>
---
#### [new 126] Countdown-Code: A Testbed for Studying The Emergence and Generalization of Reward Hacking in RLVR
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习领域，研究奖励黑客现象。旨在解决如何准确测量奖励黑客问题，通过构建环境进行实验，发现微小数据污染即可导致模型学习到奖励黑客行为。**

- **链接: [https://arxiv.org/pdf/2603.07084](https://arxiv.org/pdf/2603.07084)**

> **作者:** Muhammad Khalifa; Zohaib Khan; Omer Tafveez; Hao Peng; Lu Wang
>
> **摘要:** Reward hacking is a form of misalignment in which models overoptimize proxy rewards without genuinely solving the underlying task. Precisely measuring reward hacking occurrence remains challenging because true task rewards are often expensive or impossible to compute. We introduce Countdown-Code, a minimal environment where models can both solve a mathematical reasoning task and manipulate the test harness. This dual-access design creates a clean separation between proxy rewards (test pass/fail) and true rewards (mathematical correctness), enabling accurate measurement of reward-hacking rates. Using this environment, we study reward hacking in open-weight LLMs and find that such behaviors can be unintentionally learned during supervised fine-tuning (SFT) when even a small fraction of reward-hacking trajectories leak into training data. As little as 1\% contamination in distillation SFT data is sufficient for models to internalize reward hacking which resurfaces during subsequent reinforcement learning (RL). We further show that RL amplifies misalignment and drives its generalization beyond the original domain. We open-source our environment and code to facilitate future research on reward hacking in LLMs. Our results reveal a previously underexplored pathway through which reward hacking can emerge and persist in LLMs, underscoring the need for more rigorous validation of synthetic SFT data. Code is available at this https URL.
>
---
#### [new 127] Rethinking Attention Output Projection: Structured Hadamard Transforms for Efficient Transformers
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer模型中注意力输出投影的高计算和参数开销问题。通过引入结构化Hadamard变换替代原有投影，减少参数并提升效率。**

- **链接: [https://arxiv.org/pdf/2603.08343](https://arxiv.org/pdf/2603.08343)**

> **作者:** Shubham Aggarwal; Lokendra Kumar
>
> **备注:** 12 pages, 9 figures, 4 tables
>
> **摘要:** The dense output projection in multi-head attention scales quadratically with model dimension, contributing significantly to parameter count, memory footprint, and inference cost. We propose replacing this projection with a fixed, parameter-free Walsh Hadamard Transform followed by a lightweight learnable affine rescaling, eliminating approximately 25 percent of attention parameters per block while preserving global cross head interaction through an orthogonal, norm-preserving transformation. Across different model sizes, we demonstrate that this structured substitution maintains comparable or slightly superior downstream task performance on standard benchmarks, while achieving up to 7 percent aggregate parameter reduction, 8.9 percent peak memory savings, and 6.6 percent throughput improvement at scale, with efficiency gains growing monotonically with model size, batch size, and sequence length. Interestingly, we observe that structured Hadamard-based models exhibit a steeper validation loss curve relative to training FLOPs compared to their dense counterparts, suggesting more favorable compute utilization during training.
>
---
#### [new 128] Bootstrapping Audiovisual Speech Recognition in Zero-AV-Resource Scenarios with Synthetic Visual Data
- **分类: eess.AS; cs.CL; eess.IV**

- **简介: 该论文属于音频视觉语音识别任务，解决低资源语言缺乏标注数据的问题。通过生成合成视觉数据增强模型，提升识别性能。**

- **链接: [https://arxiv.org/pdf/2603.08249](https://arxiv.org/pdf/2603.08249)**

> **作者:** Pol Buitrago; Pol Gàlvez; Oriol Pareras; Javier Hernando
>
> **备注:** 6 pages, 3 figures, Submitted to Interspeech 2026
>
> **摘要:** Audiovisual speech recognition (AVSR) combines acoustic and visual cues to improve transcription robustness under challenging conditions but remains out of reach for most under-resourced languages due to the lack of labeled video corpora for training. We propose a zero-AV-resource AVSR framework that relies on synthetic visual streams generated by lip-syncing static facial images with real audio. We first evaluate synthetic visual augmentation on Spanish benchmarks, then apply it to Catalan, a language with no annotated audiovisual corpora. We synthesize over 700 hours of talking-head video and fine-tune a pre-trained AV-HuBERT model. On a manually annotated Catalan benchmark, our model achieves near state-of-the-art performance with much fewer parameters and training data, outperforms an identically trained audio-only baseline, and preserves multimodal advantages in noise. Scalable synthetic video thus offers a viable substitute for real recordings in zero-AV-resource AVSR.
>
---
#### [new 129] How Far Can Unsupervised RLVR Scale LLM Training?
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究无监督强化学习中的URLVR方法，解决LLM训练中依赖标签的问题。通过分析内在与外在奖励机制，揭示其优缺点及局限性，提出模型崩溃步骤作为评估指标。**

- **链接: [https://arxiv.org/pdf/2603.08660](https://arxiv.org/pdf/2603.08660)**

> **作者:** Bingxiang He; Yuxin Zuo; Zeyuan Liu; Shangziqi Zhao; Zixuan Fu; Junlin Yang; Cheng Qian; Kaiyan Zhang; Yuchen Fan; Ganqu Cui; Xiusi Chen; Youbang Sun; Xingtai Lv; Xuekai Zhu; Li Sheng; Ran Li; Huan-ang Gao; Yuchen Zhang; Bowen Zhou; Zhiyuan Liu; Ning Ding
>
> **备注:** Accepted to the ICLR 2026
>
> **摘要:** Unsupervised reinforcement learning with verifiable rewards (URLVR) offers a pathway to scale LLM training beyond the supervision bottleneck by deriving rewards without ground truth labels. Recent works leverage model intrinsic signals, showing promising early gains, yet their potential and limitations remain unclear. In this work, we revisit URLVR and provide a comprehensive analysis spanning taxonomy, theory and extensive experiments. We first classify URLVR methods into intrinsic versus external based on reward sources, then establish a unified theoretical framework revealing that all intrinsic methods converge toward sharpening the model's initial distribution This sharpening mechanism succeeds when initial confidence aligns with correctness but fails catastrophically when misaligned. Through systematic experiments, we show intrinsic rewards consistently follow a rise-then-fall pattern across methods, with collapse timing determined by model prior rather than engineering choices. Despite these scaling limits, we find intrinsic rewards remain valuable in test-time training on small datasets, and propose Model Collapse Step to measure model prior, serving as a practical indicator for RL trainability. Finally, we explore external reward methods that ground verification in computational asymmetries, showing preliminary evidence they may escape the confidence-correctness ceiling. Our findings chart boundaries for intrinsic URLVR while motivating paths toward scalable alternatives.
>
---
#### [new 130] SoK: Agentic Retrieval-Augmented Generation (RAG): Taxonomy, Architectures, Evaluation, and Research Directions
- **分类: cs.AI; cs.CL; cs.CR; cs.IR**

- **简介: 该论文属于系统综述任务，旨在解决Agentic RAG架构碎片化、评估不一致和可靠性风险问题。通过构建统一框架，提出分类体系与评估方法，明确未来研究方向。**

- **链接: [https://arxiv.org/pdf/2603.07379](https://arxiv.org/pdf/2603.07379)**

> **作者:** Saroj Mishra; Suman Niroula; Umesh Yadav; Dilip Thakur; Srijan Gyawali; Shiva Gaire
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems are increasingly evolving into agentic architectures where large language models autonomously coordinate multi-step reasoning, dynamic memory management, and iterative retrieval strategies. Despite rapid industrial adoption, current research lacks a systematic understanding of Agentic RAG as a sequential decision-making system, leading to highly fragmented architectures, inconsistent evaluation methodologies, and unresolved reliability risks. This Systematization of Knowledge (SoK) paper provides the first unified framework for understanding these autonomous systems. We formalize agentic retrieval-generation loops as finite-horizon partially observable Markov decision processes, explicitly modeling their control policies and state transitions. Building upon this formalization, we develop a comprehensive taxonomy and modular architectural decomposition that categorizes systems by their planning mechanisms, retrieval orchestration, memory paradigms, and tool-invocation behaviors. We further analyze the critical limitations of traditional static evaluation practices and identify severe systemic risks inherent to autonomous loops, including compounding hallucination propagation, memory poisoning, retrieval misalignment, and cascading tool-execution vulnerabilities. Finally, we outline key doctoral-scale research directions spanning stable adaptive retrieval, cost-aware orchestration, formal trajectory evaluation, and oversight mechanisms, providing a definitive roadmap for building reliable, controllable, and scalable agentic retrieval systems.
>
---
#### [new 131] Large Language Model for Discrete Optimization Problems: Evaluation and Step-by-step Reasoning
- **分类: cs.AI; cs.CL; math.OC**

- **简介: 该论文属于离散优化任务，研究大语言模型在解决此类问题中的表现，通过构建多样化数据集评估模型能力并提出优化建议。**

- **链接: [https://arxiv.org/pdf/2603.07733](https://arxiv.org/pdf/2603.07733)**

> **作者:** Tianhao Qian; Guilin Qi; Z.Y. Wu; Ran Gu; Xuanyi Liu; Canchen Lyu
>
> **备注:** 50 pages, 5 figures
>
> **摘要:** This work investigated the capabilities of different models, including the Llama-3 series of models and CHATGPT, with different forms of expression in solving discrete optimization problems by testing natural language datasets. In contrast to formal datasets with a limited scope of parameters, our dataset included a variety of problem types in discrete optimization problems and featured a wide range of parameter magnitudes, including instances with large parameter sets, integrated with augmented data. It aimed to (1) provide an overview of LLMs' ability in large-scale problems, (2) offer suggestions to those who want to solve discrete optimization problems automatically, and (3) regard the performance as a benchmark for future research. These datasets included original, expanded and augmented datasets. Among these three datasets, the original and augmented ones aimed for evaluation while the expanded one may help finetune a new model. In the experiment, comparisons were made between strong and week models, CoT methods and No-CoT methods on various datasets. The result showed that stronger model performed better reasonably. Contrary to general agreement, it also showed that CoT technique was not always effective regarding the capability of models and disordered datasets improved performance of models on easy to-understand problems, even though they were sometimes with high variance, a manifestation of instability. Therefore, for those who seek to enhance the automatic resolution of discrete optimization problems, it is recommended to consult the results, including the line charts presented in the Appendix, as well as the conclusions drawn in this study for relevant suggestions.
>
---
#### [new 132] SR-TTT: Surprisal-Aware Residual Test-Time Training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出SR-TTT，解决TTT模型在精确召回任务中的失败问题，通过引入稀疏记忆机制保留关键信息。**

- **链接: [https://arxiv.org/pdf/2603.06642](https://arxiv.org/pdf/2603.06642)**

> **作者:** Swamynathan V P
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Test-Time Training (TTT) language models achieve theoretically infinite context windows with an O(1) memory footprint by replacing the standard exact-attention KV-cache with hidden state ``fast weights'' W_fast updated via self-supervised learning during inference. However, pure TTT architectures suffer catastrophic failures on exact-recall tasks (e.g., Needle-in-a-Haystack). Because the fast weights aggressively compress the context into an information bottleneck, highly surprising or unique tokens are rapidly overwritten and forgotten by subsequent token gradient updates. We introduce SR-TTT (Surprisal-Aware Residual Test-Time Training), which resolves this recall failure by augmenting the TTT backbone with a loss-gated sparse memory mechanism. By dynamically routing only incompressible, highly surprising tokens to a traditional exact-attention Residual Cache, SR-TTT preserves O(1) memory for low-entropy background context while utilizing exact attention exclusively for critical needles. Our complete implementation, training scripts, and pre-trained weights are open-source and available at: this https URL.
>
---
#### [new 133] Fibration Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习中的策略优化任务，旨在解决多尺度稳定性控制问题。提出FiberPO框架，通过代数结构实现高效策略更新。**

- **链接: [https://arxiv.org/pdf/2603.08239](https://arxiv.org/pdf/2603.08239)**

> **作者:** Chang Li; Tshihao Tsu; Yaren Zhang; Chao Xue; Xiaodong He
>
> **摘要:** Large language models are increasingly trained as heterogeneous systems spanning multiple domains, expert partitions, and agentic pipelines, yet prevalent proximal objectives operate at a single scale and lack a principled mechanism for coupling token-level, trajectory-level, and higher-level hierarchical stability control. To bridge this gap, we derive the Aggregational Policy Censoring Objective (APC-Obj), the first exact unconstrained reformulation of sample-based TV-TRPO, establishing that clipping-based surrogate design and trust-region optimization are dual formulations of the same problem. Building on this foundation, we develop Fiber Bundle Gating (FBG), an algebraic framework that organizes sampled RL data as a fiber bundle and decomposes ratio gating into a base-level gate on trajectory aggregates and a fiber-level gate on per-token residuals, with provable first-order agreement with the true RL objective near on-policy. From APC-Obj and FBG we derive Fibration Policy Optimization (or simply, FiberPO), a concrete objective whose Jacobian is block-diagonal over trajectories, reduces to identity at on-policy, and provides better update direction thus improving token efficiency. The compositional nature of the framework extends beyond the trajectory-token case: fibrations compose algebraically into a Fibration Gating Hierarchy (FGH) that scales the same gating mechanism to arbitrary hierarchical depth without new primitives, as demonstrated by FiberPO-Domain, a four-level instantiation with independent trust-region budgets at the domain, prompt group, trajectory, and token levels. Together, these results connect the trust-region theory, a compositional algebraic structure, and practical multi-scale stability control into a unified framework for LLM policy optimization.
>
---
#### [new 134] Sandpiper: Orchestrated AI-Annotation for Educational Discourse at Scale
- **分类: cs.HC; cs.CL**

- **简介: 该论文提出Sandpiper系统，解决教育领域大规模对话数据的高效质性分析问题。通过AI与人类协作，提升研究效率和可靠性。**

- **链接: [https://arxiv.org/pdf/2603.08406](https://arxiv.org/pdf/2603.08406)**

> **作者:** Daryl Hedley; Doug Pietrzak; Jorge Dias; Ian Burden; Bakhtawar Ahtisham; Zhuqian Zhou; Kirk Vanacore; Josh Marland; Rachel Slama; Justin Reich; Kenneth Koedinger; René Kizilcec
>
> **摘要:** Digital educational environments are expanding toward complex AI and human discourse, providing researchers with an abundance of data that offers deep insights into learning and instructional processes. However, traditional qualitative analysis remains a labor-intensive bottleneck, severely limiting the scale at which this research can be conducted. We present Sandpiper, a mixed-initiative system designed to serve as a bridge between high-volume conversational data and human qualitative expertise. By tightly coupling interactive researcher dashboards with agentic Large Language Model (LLM) engines, the platform enables scalable analysis without sacrificing methodological rigor. Sandpiper addresses critical barriers to AI adoption in education by implementing context-aware, automated de-identification workflows supported by secure, university-housed infrastructure to ensure data privacy. Furthermore, the system employs schema-constrained orchestration to eliminate LLM hallucinations and enforces strict adherence to qualitative codebooks. An integrated evaluations engine allows for the continuous benchmarking of AI performance against human labels, fostering an iterative approach to model refinement and validation. We propose a user study to evaluate the system's efficacy in improving research efficiency, inter-rater reliability, and researcher trust in AI-assisted qualitative workflows.
>
---
#### [new 135] 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型中的空间推理任务，旨在解决模型在空间理解上的不足。通过引入3ViewSense框架，利用正交视图提升空间推理能力。**

- **链接: [https://arxiv.org/pdf/2603.07751](https://arxiv.org/pdf/2603.07751)**

> **作者:** Shaoxiong Zhan; Yanlin Lai; Zheng Liu; Hai Lin; Shen Li; Xiaodong Cai; Zijian Lin; Wen Huang; Hai-Tao Zheng
>
> **摘要:** Current Large Language Models have achieved Olympiad-level logic, yet Vision-Language Models paradoxically falter on elementary spatial tasks like block counting. This capability mismatch reveals a critical ``spatial intelligence gap,'' where models fail to construct coherent 3D mental representations from 2D observations. We uncover this gap via diagnostic analyses showing the bottleneck is a missing view-consistent spatial interface rather than insufficient visual features or weak reasoning. To bridge this, we introduce \textbf{3ViewSense}, a framework that grounds spatial reasoning in Orthographic Views. Drawing on engineering cognition, we propose a ``Simulate-and-Reason'' mechanism that decomposes complex scenes into canonical orthographic projections to resolve geometric ambiguities. By aligning egocentric perceptions with these allocentric references, our method facilitates explicit mental rotation and reconstruction. Empirical results on spatial reasoning benchmarks demonstrate that our method significantly outperforms existing baselines, with consistent gains on occlusion-heavy counting and view-consistent spatial reasoning. The framework also improves the stability and consistency of spatial descriptions, offering a scalable path toward stronger spatial intelligence in multimodal systems.
>
---
#### [new 136] LieCraft: A Multi-Agent Framework for Evaluating Deceptive Capabilities in Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出LieCraft框架，用于评估语言模型的欺骗能力。任务是检测LLM在复杂场景中的道德行为，解决安全风险问题。工作包括设计游戏机制和测试12个模型的行为表现。**

- **链接: [https://arxiv.org/pdf/2603.06874](https://arxiv.org/pdf/2603.06874)**

> **作者:** Matthew Lyle Olson; Neale Ratzlaff; Musashi Hinck; Tri Nguyen; Vasudev Lal; Joseph Campbell; Simon Stepputtis; Shao-Yen Tseng
>
> **备注:** AAAI 2026 Alignment track. Authors 1 and 2 contributed equally, 3 and 4 contributed equally, 6 and 7 and 8 contributed equally (ordered by last name)
>
> **摘要:** Large Language Models (LLMs) exhibit impressive general-purpose capabilities but also introduce serious safety risks, particularly the potential for deception as models acquire increased agency and human oversight diminishes. In this work, we present LieCraft: a novel evaluation framework and sandbox for measuring LLM deception that addresses key limitations of prior game-based evaluations. At its core, LieCraft is a novel multiplayer hidden-role game in which players select an ethical alignment and execute strategies over a long time-horizon to accomplish missions. Cooperators work together to solve event challenges and expose bad actors, while Defectors evade suspicion while secretly sabotaging missions. To enable real-world relevance, we develop 10 grounded scenarios such as childcare, hospital resource allocation, and loan underwriting that recontextualize the underlying mechanics in ethically significant, high-stakes domains. We ensure balanced gameplay in LieCraft through careful design of game mechanics and reward structures that incentivize meaningful strategic choices while eliminating degenerate strategies. Beyond the framework itself, we report results from 12 state-of-the-art LLMs across three behavioral axes: propensity to defect, deception skill, and accusation accuracy. Our findings reveal that despite differences in competence and overall alignment, all models are willing to act unethically, conceal their intentions, and outright lie to pursue their goals.
>
---
## 更新

#### [replaced 001] Neuro-Symbolic Synergy for Interactive World Modeling
- **分类: cs.CL**

- **简介: 该论文提出NeSyS框架，融合大语言模型与符号规则，解决世界模型中逻辑一致性与语义表达的矛盾，提升预测准确性和数据效率。任务为交互式世界建模。**

- **链接: [https://arxiv.org/pdf/2602.10480](https://arxiv.org/pdf/2602.10480)**

> **作者:** Hongyu Zhao; Siyu Zhou; Haolin Yang; Zengyi Qin; Tianyi Zhou
>
> **摘要:** Large language models (LLMs) exhibit strong general-purpose reasoning capabilities, yet they frequently hallucinate when used as world models (WMs), where strict compliance with deterministic transition rules--particularly in corner cases--is essential. In contrast, Symbolic WMs provide logical consistency but lack semantic expressivity. To bridge this gap, we propose Neuro-Symbolic Synergy (NeSyS), a framework that integrates the probabilistic semantic priors of LLMs with executable symbolic rules to achieve both expressivity and robustness. NeSyS alternates training between the two models using trajectories inadequately explained by the other. Unlike rule-based prompting, the symbolic WM directly constrains the LLM by modifying its output probability distribution. The neural WM is fine-tuned only on trajectories not covered by symbolic rules, reducing training data by 50% without loss of accuracy. Extensive experiments on three distinct interactive environments, i.e., ScienceWorld, Webshop, and Plancraft, demonstrate NeSyS's consistent advantages over baselines in both WM prediction accuracy and data efficiency. Our models and code are available at this https URL.
>
---
#### [replaced 002] Offline-First Large Language Model Architecture for AI-Assisted Learning with Adaptive Response Levels in Low-Connectivity Environments
- **分类: cs.CY; cs.AR; cs.CL; cs.HC**

- **简介: 该论文属于教育技术任务，旨在解决低网络环境下AI辅助学习的问题。通过设计离线大语言模型架构，实现本地化推理和自适应响应，提升学习支持效果。**

- **链接: [https://arxiv.org/pdf/2603.03339](https://arxiv.org/pdf/2603.03339)**

> **作者:** Joseph Walusimbi; Ann Move Oguti; Joshua Benjamin Ssentongo; Keith Ainebyona
>
> **备注:** 16 pages, 2 table, 10 figures
>
> **摘要:** Artificial intelligence (AI) and large language models (LLMs) are transforming educational technology by enabling conversational tutoring, personalized explanations, and inquiry-driven learning. However, most AI-based learning systems rely on continuous internet connectivity and cloud-based computation, limiting their use in bandwidth-constrained environments. This paper presents an offline-first large language model architecture designed for AI-assisted learning in low-connectivity settings. The system performs all inference locally using quantized language models and incorporates hardware-aware model selection to enable deployment on low-specification CPU-only devices. By removing dependence on cloud infrastructure, the system provides curriculum-aligned explanations and structured academic support through natural-language interaction. To support learners at different educational stages, the system includes adaptive response levels that generate explanations at varying levels of complexity: Simple English, Lower Secondary, Upper Secondary, and Technical. This allows explanations to be adjusted to student ability, improving clarity and understanding of academic concepts. The system was deployed in selected secondary and tertiary institutions under limited-connectivity conditions and evaluated across technical performance, usability, perceived response quality, and educational impact. Results show stable operation on legacy hardware, acceptable response times, and positive user perceptions regarding support for self-directed learning. These findings demonstrate the feasibility of offline large language model deployment for AI-assisted education in low-connectivity environments.
>
---
#### [replaced 003] LatentMem: Customizing Latent Memory for Multi-Agent Systems
- **分类: cs.CL; cs.LG; cs.MA**

- **简介: 该论文属于多智能体系统任务，旨在解决记忆同质化和信息过载问题。提出LatentMem框架，通过定制化记忆提升性能。**

- **链接: [https://arxiv.org/pdf/2602.03036](https://arxiv.org/pdf/2602.03036)**

> **作者:** Muxin Fu; Xiangyuan Xue; Yafu Li; Zefeng He; Siyuan Huang; Xiaoye Qu; Yu Cheng; Yang Yang
>
> **摘要:** Large language model (LLM)-powered multi-agent systems (MAS) demonstrate remarkable collective intelligence, wherein multi-agent memory serves as a pivotal mechanism for continual adaptation. However, existing multi-agent memory designs remain constrained by two fundamental bottlenecks: (i) memory homogenization arising from the absence of role-aware customization, and (ii) information overload induced by excessively fine-grained memory entries. To address these limitations, we propose LatentMem, a learnable multi-agent memory framework designed to customize agent-specific memories in a token-efficient manner. Specifically, LatentMem comprises an experience bank that stores raw interaction trajectories in a lightweight form, and a memory composer that synthesizes compact latent memories conditioned on retrieved experience and agent-specific contexts. Further, we introduce Latent Memory Policy Optimization (LMPO), which propagates task-level optimization signals through latent memories to the composer, encouraging it to produce compact and high-utility representations. Extensive experiments across diverse benchmarks and mainstream MAS frameworks show that LatentMem achieves a performance gain of up to $19.36$% over vanilla settings and consistently outperforms existing memory architectures, without requiring any modifications to the underlying frameworks.
>
---
#### [replaced 004] RedSage: A Cybersecurity Generalist LLM
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文提出RedSage，一个专注于网络安全的开源大模型，解决现有模型在隐私和领域适应性上的不足。通过持续预训练和代理增强技术提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.22159](https://arxiv.org/pdf/2601.22159)**

> **作者:** Naufal Suryanto; Muzammal Naseer; Pengfei Li; Syed Talal Wasim; Jinhui Yi; Juergen Gall; Paolo Ceravolo; Ernesto Damiani
>
> **备注:** Published at ICLR 2026; Project page: this https URL
>
> **摘要:** Cybersecurity operations demand assistant LLMs that support diverse workflows without exposing sensitive data. Existing solutions either rely on proprietary APIs with privacy risks or on open models lacking domain adaptation. To bridge this gap, we curate 11.8B tokens of cybersecurity-focused continual pretraining data via large-scale web filtering and manual collection of high-quality resources, spanning 28.6K documents across frameworks, offensive techniques, and security tools. Building on this, we design an agentic augmentation pipeline that simulates expert workflows to generate 266K multi-turn cybersecurity samples for supervised fine-tuning. Combined with general open-source LLM data, these resources enable the training of RedSage, an open-source, locally deployable cybersecurity assistant with domain-aware pretraining and post-training. To rigorously evaluate the models, we introduce RedSage-Bench, a benchmark with 30K multiple-choice and 240 open-ended Q&A items covering cybersecurity knowledge, skills, and tool expertise. RedSage is further evaluated on established cybersecurity benchmarks (e.g., CTI-Bench, CyberMetric, SECURE) and general LLM benchmarks to assess broader generalization. At the 8B scale, RedSage achieves consistently better results, surpassing the baseline models by up to +5.59 points on cybersecurity benchmarks and +5.05 points on Open LLM Leaderboard tasks. These findings demonstrate that domain-aware agentic augmentation and pre/post-training can not only enhance cybersecurity-specific expertise but also help to improve general reasoning and instruction-following. All models, datasets, and code are publicly available.
>
---
#### [replaced 005] SwiftEmbed: Ultra-Fast Text Embeddings via Static Token Lookup for Real-Time Applications
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SwiftEmbed，用于实时文本嵌入任务，解决低延迟需求。采用静态词 token 查找，提升处理速度，适用于需要亚毫秒响应的应用场景。**

- **链接: [https://arxiv.org/pdf/2510.24793](https://arxiv.org/pdf/2510.24793)**

> **作者:** Edouard Lansiaux; Antoine Simonet; Eric Wiel
>
> **摘要:** We present SwiftEmbed, a production-oriented serving system for static token embeddings that achieves 1.12\,ms p50 latency for single-text requests while maintaining a 60.6 MTEB average score across 8 representative tasks. Built around the open-source Potion-base-8M distilled model from MinishLab and implemented in Rust, the system delivers 50,000 requests per second through static embedding lookup, mean pooling, and zero-copy IEEE754 binary serialization. Evaluation demonstrates exceptional duplicate detection performance (90.1% AP) and strong semantic similarity (76.1% Spearman correlation). Performance relative to Sentence-BERT is task-dependent: robust for deduplication and similarity workloads (89--100%), substantially lower for classification and complex retrieval tasks (75%). Domain-specific performance ranges from 75% to 131% of a GloVe-840B baseline. The system targets real-time embedding applications where sub-5\,ms latency is operationally critical and where full transformer inference is not feasible.
>
---
#### [replaced 006] ModalImmune: Immunity Driven Unlearning via Self Destructive Training
- **分类: cs.LG; cs.CL; cs.MM**

- **简介: 该论文提出ModalImmune，解决多模态系统在输入通道丢失时的可靠性问题，通过训练增强模态免疫性。**

- **链接: [https://arxiv.org/pdf/2602.16197](https://arxiv.org/pdf/2602.16197)**

> **作者:** Rong Fu; Jia Yee Tan; Zijian Zhang; Ziming Wang; Zhaolu Kang; Muge Qi; Shuning Zhang; Simon Fong
>
> **备注:** 23 pages, 8 figures
>
> **摘要:** Multimodal systems are vulnerable to partial or complete loss of input channels at deployment, which undermines reliability in real-world settings. This paper presents ModalImmune, a training framework that enforces modality immunity by intentionally and controllably collapsing selected modality information during training so the model learns joint representations that are robust to destructive modality influence. The framework combines a spectrum-adaptive collapse regularizer, an information-gain guided controller for targeted interventions, curvature-aware gradient masking to stabilize destructive updates, and a certified Neumann-truncated hyper-gradient procedure for automatic meta-parameter adaptation. Empirical evaluation on standard multimodal benchmarks demonstrates that ModalImmune improves resilience to modality removal and corruption while retaining convergence stability and reconstruction capacity.
>
---
#### [replaced 007] Half the Nonlinearity Is Wasted: Measuring and Reallocating the Transformer's MLP Budget
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究Transformer中MLP非线性的必要性，通过引入门控机制动态选择线性或非线性计算，以优化模型效率。任务是模型压缩与优化，解决如何有效利用非线性资源的问题。工作包括实验分析与参数调整。**

- **链接: [https://arxiv.org/pdf/2603.03459](https://arxiv.org/pdf/2603.03459)**

> **作者:** Peter Balogh
>
> **摘要:** We investigate when transformer MLP nonlinearity is actually necessary. A gate with $d+1$ parameters decides when to replace the full MLP with a linear surrogate. Through systematic investigation across six models (162M-2.8B parameters), two architectures, and three corpora, we establish that nonlinearity need cannot be predicted from token identity: cross-corpus correlation is zero ($r < 0.05$). The routing decision is fully contextual. Despite weak per-instance predictability, the gate exploits a heavily skewed distribution where most MLP computations are near-linear, achieving 25-56% linear routing at <1% perplexity cost in GPT-2. In GPT-2 Large, 11 of 36 layers beat baseline with gating and no layer exceeds 3.7% all-linear cost. This success is architecture-dependent: Pythia models show higher costs, though Pythia-2.8B's full 32-layer sweep reveals one layer that narrowly beats baseline. As a proof of concept, we progressively replace middle-layer MLPs with frozen linear matrices: 5 of 24 layers linearize at zero cost. With a full training budget, 4 linearized layers yield a 10.2% perplexity improvement -- and a two-phase gated approach pushes this to 17.3%, beating a vanilla fine-tuning control and confirming that the nonlinear MLPs at these layers were actively harmful.
>
---
#### [replaced 008] LaVCa: LLM-assisted Visual Cortex Captioning
- **分类: q-bio.NC; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出LaVCa方法，用于生成脑区选择性图像的自然语言描述，解决神经元群体属性解释难题，通过LLM提升对视觉皮层功能的理解。**

- **链接: [https://arxiv.org/pdf/2502.13606](https://arxiv.org/pdf/2502.13606)**

> **作者:** Takuya Matsuyama; Shinji Nishimoto; Yu Takagi
>
> **备注:** Accepted to ICLR 2026. Website: this https URL
>
> **摘要:** Understanding the property of neural populations (or voxels) in the human brain can advance our comprehension of human perceptual and cognitive processing capabilities and contribute to developing brain-inspired computer models. Recent encoding models using deep neural networks (DNNs) have successfully predicted voxel-wise activity. However, interpreting the properties that explain voxel responses remains challenging because of the black-box nature of DNNs. As a solution, we propose LLM-assisted Visual Cortex Captioning (LaVCa), a data-driven approach that uses large language models (LLMs) to generate natural-language captions for images to which voxels are selective. By applying LaVCa for image-evoked brain activity, we demonstrate that LaVCa generates captions that describe voxel selectivity more accurately than the previously proposed method. Furthermore, the captions generated by LaVCa quantitatively capture more detailed properties than the existing method at both the inter-voxel and intra-voxel levels. Furthermore, a more detailed analysis of the voxel-specific properties generated by LaVCa reveals fine-grained functional differentiation within regions of interest (ROIs) in the visual cortex and voxels that simultaneously represent multiple distinct concepts. These findings offer profound insights into human visual representations by assigning detailed captions throughout the visual cortex while highlighting the potential of LLM-based methods in understanding brain representations.
>
---
#### [replaced 009] Multimodal LLMs Do Not Compose Skills Optimally Across Modalities
- **分类: cs.CL**

- **简介: 该论文研究多模态大语言模型在跨模态技能组合上的表现，旨在解决技能有效组合的问题。通过设计任务和实验，发现模型存在显著的跨模态技能组合差距，并尝试优化方法。**

- **链接: [https://arxiv.org/pdf/2511.08113](https://arxiv.org/pdf/2511.08113)**

> **作者:** Paula Ontalvilla; Aitor Ormazabal; Gorka Azkune
>
> **摘要:** Skill composition is the ability to combine previously learned skills to solve new tasks. As neural networks acquire increasingly complex skills during their pretraining, it is not clear how successfully they can compose them. In this paper, we focus on Multimodal Large Language Models (MLLM), and study their ability to compose skills across modalities. To this end, we design three evaluation tasks which can be solved sequentially composing two modality-dependent skills, and evaluate several open MLLMs under two main settings: i) prompting the model to directly solve the task, and ii) using a two-step cascaded inference approach, which manually enforces the composition of the two skills for a given task. Even with these straightforward compositions, we find that all evaluated MLLMs exhibit a significant cross-modality skill composition gap. To mitigate the aforementioned gap, we explore two alternatives: i) use chain-of-thought prompting to explicitly instruct MLLMs for skill composition and ii) a specific fine-tuning recipe to promote skill composition. Although those strategies improve model performance, they still exhibit significant skill composition gaps, suggesting that more research is needed to improve cross-modal skill composition in MLLMs.
>
---
#### [replaced 010] Conformal Prediction for Risk-Controlled Medical Entity Extraction Across Clinical Domains
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗实体抽取任务，解决LLM置信度不准确的问题。通过共形预测框架，在两个临床领域实现高覆盖率的可靠抽取。**

- **链接: [https://arxiv.org/pdf/2603.00924](https://arxiv.org/pdf/2603.00924)**

> **作者:** Manil Shrestha; Edward Kim
>
> **摘要:** Large Language Models (LLMs) are increasingly used for medical entity extraction, yet their confidence scores are often miscalibrated, limiting safe deployment in clinical settings. We present a conformal prediction framework that provides finite-sample coverage guarantees for LLM-based extraction across two clinical domains. First, we extract structured entities from 1,000 FDA drug labels across eight sections using GPT-4.1, verified via FactScore-based atomic statement evaluation (97.7\% accuracy over 128,906 entities). Second, we extract radiological entities from MIMIC-CXR reports using the RadGraph schema with GPT-4.1 and Llama-4-Maverick, evaluated against physician annotations (entity F1: 0.81 to 0.84). Our central finding is that miscalibration direction reverses across domains: on well-structured FDA labels, models are underconfident, requiring modest conformal thresholds ($\tau \approx 0.06$), while on free-text radiology reports, models are overconfident, demanding strict thresholds ($\tau$ up to 0.99). Despite this heterogeneity, conformal prediction achieves target coverage ($\geq 90\%$) in both settings with manageable rejection rates (9--13\%). These results demonstrate that calibration is not a global model property but depends on document structure, extraction category, and model architecture, motivating domain-specific conformal calibration for safe clinical deployment.
>
---
#### [replaced 011] Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; cs.RO**

- **简介: 该论文属于AI验证任务，旨在解决MLLM在评估行为时的同意偏差问题。通过引入SGV方法，提升验证准确性与人类对齐度。**

- **链接: [https://arxiv.org/pdf/2507.11662](https://arxiv.org/pdf/2507.11662)**

> **作者:** Moises Andrade; Joonhyuk Cha; Brandon Ho; Vriksha Srihari; Karmesh Yadav; Zsolt Kira
>
> **备注:** ICLR 2026. Code, models, and data publicly available at this https URL
>
> **摘要:** Verifiers--functions assigning rewards to agent behavior--have been key to AI progress in math, code, and games. However, extending gains to domains without clear-cut success criteria remains a challenge: while humans can recognize desired outcomes, translating this intuition into scalable rules is nontrivial. Multimodal LLMs (MLLMs) offer a promising solution, given their world knowledge, human-preference alignment, and reasoning capabilities. We evaluate MLLM verifiers across web navigation, computer use, and robotics, spanning 13+ models, 28+ designs, and thousands of trajectories from diverse agents. We identify a critical limitation: a strong tendency for MLLMs to over-validate agent behavior--a phenomenon we term agreement bias. This bias is pervasive, resilient to test-time scaling, and can harm applications relying on MLLM judgments/rewards (e.g., self-improvement, steering, online supervision). We discuss several considerations for evaluating and designing MLLM verifiers, and introduce SGV, a lightweight method that better leverages their capabilities by modulating (un)conditional generation. First, an MLLM is elicited to generate broad priors about desired behavior, independent of the data under evaluation. Then, conditioned on self-generated priors, it reasons over and evaluates a candidate trajectory. Our methods yield more human-aligned verifiers, improving failure detection by 25pp and accuracy by 14pp. In self-improvement and online supervision, they boost task completion of a GUI specialist in OSWorld, a diffusion policy in robomimic, and a ReAct agent in VisualWebArena--surpassing the previous state of the art by 20pp. As a byproduct, we release an update of VisualWebArena featuring strong agent baselines, more human-aligned oracles, container parallelism with high fidelity and proper resets, >10x speedups, and VWA-Lite, a 1/3 subset with comparable evaluation fidelity.
>
---
#### [replaced 012] HACHIMI: Scalable and Controllable Student Persona Generation via Orchestrated Agents
- **分类: cs.CL**

- **简介: 该论文提出HACHIMI框架，解决学生角色生成任务中的理论对齐与分布控制问题，通过多代理机制生成100万条符合教育理论的学生成人数据。**

- **链接: [https://arxiv.org/pdf/2603.04855](https://arxiv.org/pdf/2603.04855)**

> **作者:** Yilin Jiang; Fei Tan; Xuanyu Yin; Jing Leng; Aimin Zhou
>
> **备注:** 46 pages, 7 figures, submitted to ACL 2026. The dataset is available at this https URL
>
> **摘要:** Student Personas (SPs) are emerging as infrastructure for educational LLMs, yet prior work often relies on ad-hoc prompting or hand-crafted profiles with limited control over educational theory and population distributions. We formalize this as Theory-Aligned and Distribution-Controllable Persona Generation (TAD-PG) and introduce HACHIMI, a multi-agent Propose-Validate-Revise framework that generates theory-aligned, quota-controlled personas. HACHIMI factorizes each persona into a theory-anchored educational schema, enforces developmental and psychological constraints via a neuro-symbolic validator, and combines stratified sampling with semantic deduplication to reduce mode collapse. The resulting HACHIMI-1M corpus comprises 1 million personas for Grades 1-12. Intrinsic evaluation shows near-perfect schema validity, accurate quotas, and substantial diversity, while external evaluation instantiates personas as student agents answering CEPS and PISA 2022 surveys; across 16 cohorts, math and curiosity/growth constructs align strongly between humans and agents, whereas classroom-climate and well-being constructs are only moderately aligned, revealing a fidelity gradient. All personas are generated with Qwen2.5-72B, and HACHIMI provides a standardized synthetic student population for group-level benchmarking and social-science simulations. Resources available at this https URL
>
---
#### [replaced 013] Multi-Domain Audio Question Answering Benchmark Toward Acoustic Content Reasoning
- **分类: cs.SD; cs.AI; cs.CL; cs.MM; eess.AS**

- **简介: 该论文属于多领域音频问答任务，旨在提升音频语言模型的听觉理解与推理能力。通过构建多场景数据集和评估方案，测试模型在不同音频场景下的问答表现。**

- **链接: [https://arxiv.org/pdf/2505.07365](https://arxiv.org/pdf/2505.07365)**

> **作者:** Chao-Han Huck Yang; Sreyan Ghosh; Qing Wang; Jaeyeon Kim; Hengyi Hong; Sonal Kumar; Guirui Zhong; Zhifeng Kong; S Sakshi; Vaibhavi Lokegaonkar; Oriol Nieto; Ramani Duraiswami; Dinesh Manocha; Gunhee Kim; Jun Du; Rafael Valle; Bryan Catanzaro
>
> **备注:** Dataset: this https URL DCASE Task-5 challenge: this http URL. Accepted to ICASSP 2026
>
> **摘要:** We present Task 5 of the DCASE 2025 Challenge: an Audio Question Answering (AQA) benchmark spanning multiple domains of sound understanding. This task defines three QA subsets (Bioacoustics, Temporal Soundscapes, and Complex QA) to test audio-language models on interactive question-answering over diverse acoustic scenes. We describe the dataset composition (from marine mammal calls to soundscapes and complex real-world clips), the evaluation protocol (top-1 accuracy with answer-shuffling robustness), and baseline systems (Qwen2-Audio-7B, AudioFlamingo 2, Gemini-2-Flash). Preliminary results on the development set are compared, showing strong variation across models and subsets. This challenge aims to advance the audio understanding and reasoning capabilities of audio-language models toward human-level acuity, which are crucial for enabling AI agents to perceive and interact about the world effectively.
>
---
#### [replaced 014] More Women, Same Stereotypes: Unpacking the Gender Bias Paradox in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的性别偏 bias 研究任务，旨在揭示 LLMs 中的性别刻板印象问题。通过故事生成分析，发现模型虽过度呈现女性角色，但职业分布仍符合传统性别刻板印象。**

- **链接: [https://arxiv.org/pdf/2503.15904](https://arxiv.org/pdf/2503.15904)**

> **作者:** Evan Chen; Run-Jun Zhan; Yan-Bai Lin; Hung-Hsuan Chen
>
> **摘要:** Large Language Models (LLMs) have revolutionized natural language processing, yet concerns persist regarding their tendency to reflect or amplify social biases. This study introduces a novel evaluation framework to uncover gender biases in LLMs: using free-form storytelling to surface biases embedded within the models. A systematic analysis of ten prominent LLMs shows a consistent pattern of overrepresenting female characters across occupations, likely due to supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF). Paradoxically, despite this overrepresentation, the occupational gender distributions produced by these LLMs align more closely with human stereotypes than with real-world labor data. This highlights the challenge and importance of implementing balanced mitigation measures to promote fairness and prevent the establishment of potentially new biases. We release the prompts and LLM-generated stories at GitHub.
>
---
#### [replaced 015] HypoSpace: Evaluating LLM Creativity as Set-Valued Hypothesis Generators under Underdetermination
- **分类: cs.CL**

- **简介: 该论文提出HypoSpace，评估大语言模型作为假设生成器的创造力，解决科学问题中多解性下的假设生成任务。**

- **链接: [https://arxiv.org/pdf/2510.15614](https://arxiv.org/pdf/2510.15614)**

> **作者:** Tingting Chen; Beibei Lin; Zifeng Yuan; Qiran Zou; Hongyu He; Anirudh Goyal; Yew-Soon Ong; Dianbo Liu
>
> **摘要:** As language models are increasingly used in scientific workflows, evaluating their ability to propose sets of explanations-not just a single correct answer-becomes critical. Many scientific problems are underdetermined: multiple, mechanistically distinct hypotheses are consistent with the same observations. We introduce HypoSpace, a diagnostic suite that treats LLMs as samplers of finite hypothesis sets and measures three complementary indicators: Validity (precision of proposals consistent with observations), Uniqueness (non-redundancy among proposals), and Recovery (coverage of the enumerated admissible set). We instantiate HypoSpace in three structured domains with deterministic validators and exactly enumerated hypothesis spaces: (i) causal graphs from perturbations, (ii) gravity-constrained 3D voxel reconstruction from top-down projections, and (iii) Boolean genetic interactions. Across instruction-tuned and reasoning-focused models, Validity often remains high while Uniqueness and Recovery degrade as the admissible space grows, revealing mode collapse that is invisible to correctness-only metrics. HypoSpace offers a controlled probe-rather than a leaderboard-for methods that explicitly explore and cover admissible explanation spaces. Code is available at: this https URL.
>
---
#### [replaced 016] Estimating Item Difficulty Using Large Language Models and Tree-Based Machine Learning Algorithms
- **分类: cs.CY; cs.CL; cs.LG**

- **简介: 该论文属于评估任务，旨在预测K-5学科题目难度，减少对实地测试的依赖。通过直接估计和特征提取两种方法进行研究，验证了LLM在该任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2504.08804](https://arxiv.org/pdf/2504.08804)**

> **作者:** Pooya Razavi; Sonya Powers
>
> **摘要:** Estimating item difficulty through field-testing is often resource-intensive and time-consuming. As such, there is strong motivation to develop methods that can predict item difficulty at scale using only the item content. Large Language Models (LLMs) represent a new frontier for this goal. The present research examines the feasibility of using an LLM to predict item difficulty for K-5 mathematics and reading assessment items (N = 5170). Two estimation approaches were implemented: (a) a direct estimation method that prompted the LLM to assign a single difficulty rating to each item, and (b) a feature-based strategy where the LLM extracted multiple cognitive and linguistic features, which were then used in ensemble tree-based models (random forests and gradient boosting) to predict difficulty. Overall, direct LLM estimates showed moderate to strong correlations with true item difficulties. However, their accuracy varied by grade level, often performing worse for early grades. In contrast, the feature-based method yielded stronger predictive accuracy, with correlations as high as r = 0.87 and lower error estimates compared to both direct LLM predictions and baseline regressors. These findings highlight the promise of LLMs in streamlining item development and reducing reliance on extensive field testing and underscore the importance of structured feature extraction. We provide a seven-step workflow for testing professionals who would want to implement a similar item difficulty estimation approach with their item pool.
>
---
#### [replaced 017] Llama-Mob: Instruction-Tuning Llama-3-8B Excels in City-Scale Mobility Prediction
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于城市规模移动预测任务，旨在解决传统方法在长周期和跨城市泛化上的不足。通过指令微调Llama-3-8B模型，提升长期人类移动预测性能。**

- **链接: [https://arxiv.org/pdf/2410.23692](https://arxiv.org/pdf/2410.23692)**

> **作者:** Peizhi Tang; Chuang Yang; Tong Xing; Xiaohang Xu; Jiayi Xu; Renhe Jiang; Kaoru Sezaki
>
> **摘要:** Human mobility prediction plays a critical role in applications such as disaster response, urban planning, and epidemic forecasting. Traditional methods often rely on designing crafted, domain-specific models, and typically focus on short-term predictions, which struggle to generalize across diverse urban environments. In this study, we introduce Llama3-8B-Mob, a large language model fine-tuned with instruction tuning, for long-term citywide mobility prediction--in a Q&A manner. We validate our approach using large-scale human mobility data from four metropolitan areas in Japan, focusing on predicting individual trajectories over the next 15 days. The results demonstrate that Llama3-8B-Mob excels in modeling long-term human mobility--surpassing the state-of-the-art on multiple prediction metrics. It also displays strong zero-shot generalization capabilities--effectively generalizing to other cities even when fine-tuned only on limited samples from a single city. Moreover, our method is general and can be readily extended to the next POI prediction task. For brevity, we refer to our model as Llama-Mob, and the corresponding results are included in this paper. Source codes are available at this https URL.
>
---
#### [replaced 018] Emotion Collider: Dual Hyperbolic Mirror Manifolds for Sentiment Recovery via Anti Emotion Reflection
- **分类: cs.MM; cs.CL; cs.LG**

- **简介: 该论文属于多模态情感建模任务，旨在提升情绪理解的鲁棒性。提出EC-Net框架，结合双曲空间和超图融合，增强特征表示与分类效果。**

- **链接: [https://arxiv.org/pdf/2602.16161](https://arxiv.org/pdf/2602.16161)**

> **作者:** Rong Fu; Ziming Wang; Shuo Yin; Haiyun Wei; Kun Liu; Xianda Li; Zeli Su; Simon Fong
>
> **备注:** 25 pages, 14 figures
>
> **摘要:** Emotional expression underpins natural communication and effective human-computer interaction. We present Emotion Collider (EC-Net), a hyperbolic hypergraph framework for multimodal emotion and sentiment modeling. EC-Net represents modality hierarchies using Poincare-ball embeddings and performs fusion through a hypergraph mechanism that passes messages bidirectionally between nodes and hyperedges. To sharpen class separation, contrastive learning is formulated in hyperbolic space with decoupled radial and angular objectives. High-order semantic relations across time steps and modalities are preserved via adaptive hyperedge construction. Empirical results on standard multimodal emotion benchmarks show that EC-Net produces robust, semantically coherent representations and consistently improves accuracy, particularly when modalities are partially available or contaminated by noise. These findings indicate that explicit hierarchical geometry combined with hypergraph fusion is effective for resilient multimodal affect understanding.
>
---
#### [replaced 019] FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型推理优化任务，解决长上下文带来的KV缓存效率问题。提出FreeKV框架，通过算法与系统协同优化提升KV检索效率，实现近无损精度和显著加速。**

- **链接: [https://arxiv.org/pdf/2505.13109](https://arxiv.org/pdf/2505.13109)**

> **作者:** Guangda Liu; Chengwei Li; Zhenyu Ning; Jing Lin; Yiwu Yao; Danning Ke; Minyi Guo; Jieru Zhao
>
> **摘要:** Large language models (LLMs) are widely deployed with rapidly expanding context windows to support increasingly demanding applications. However, long contexts pose significant deployment challenges, primarily due to the KV cache whose size grows proportionally with context length. While KV cache compression methods have been proposed to address this issue, KV dropping methods incur considerable accuracy loss, and KV retrieval methods suffer from significant efficiency bottlenecks. We propose FreeKV, a training-free algorithm-system co-optimization framework to enhance KV retrieval efficiency while preserving accuracy. On the algorithm side, FreeKV introduces speculative retrieval to shift the KV selection and recall processes out of the critical path, combined with fine-grained correction to ensure accuracy. On the system side, FreeKV employs hybrid KV layouts across CPU and GPU memory to eliminate fragmented data transfers, and leverages double-buffered streamed recall to further improve efficiency, enabling effective overlap with computation, full latency hiding, and practical speedups from speculative recall. Experiments demonstrate that FreeKV achieves near-lossless accuracy across various scenarios and models, delivering up to a 13$\times$ speedup compared to SOTA KV retrieval methods. Code is available at this https URL.
>
---
#### [replaced 020] Speaker effects in language comprehension: An integrative model of language and speaker processing
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文属于语言理解研究，探讨说话者效应如何影响语言处理。提出整合模型，结合感知与预期机制，解决说话者身份对语言理解的影响问题。**

- **链接: [https://arxiv.org/pdf/2412.07238](https://arxiv.org/pdf/2412.07238)**

> **作者:** Hanlin Wu; Zhenguang G. Cai
>
> **备注:** In press in Psychonomic Bulletin & Review
>
> **摘要:** The identity of a speaker influences language comprehension through modulating perception and expectation. This review explores speaker effects and proposes an integrative model of language and speaker processing that integrates distinct mechanistic perspectives. We argue that speaker effects arise from the interplay between bottom-up perception-based processes, driven by acoustic-episodic memory, and top-down expectation-based processes, driven by a speaker model. We show that language and speaker processing are functionally integrated through multi-level probabilistic processing: prior beliefs about a speaker modulate language processing at the phonetic, lexical, and semantic levels, while the unfolding speech and message continuously updates the speaker model, refining broad demographic priors into precise individualized representations. Within this framework, we distinguish between speaker-idiosyncrasy effects arising from familiarity with an individual and speaker-demographics effects arising from social group expectations. We discuss how speaker effects serve as indices for assessing language development and social cognition, and we encourage future research to extend these findings to the emerging domain of artificial intelligence (AI) speakers, as AI agents represent a new class of social interlocutors that are transforming the way we engage in daily communication.
>
---
#### [replaced 021] A Single Model Ensemble Framework for Neural Machine Translation using Pivot Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于神经机器翻译任务，旨在提升低资源语言对的翻译质量。通过单模型集成框架，利用桥梁语言生成候选翻译并进行后处理聚合，提高翻译准确性。**

- **链接: [https://arxiv.org/pdf/2502.01182](https://arxiv.org/pdf/2502.01182)**

> **作者:** Seokjin Oh; Keonwoong Noh; Woohwan Jung
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** Despite the recent remarkable advances in neural machine translation, translation quality for low-resource language pairs remains subpar. Ensembling multiple systems is a widely adopted technique to enhance performance, often accomplished by combining probability distributions. However, previous approaches face the challenge of high computational costs for training multiple models. Furthermore, for black-box models, averaging token-level probabilities at each decoding step is not feasible. To address the problems of multi-model ensemble methods, we present a pivot-based single model ensemble. The proposed strategy consists of two steps: pivot-based candidate generation and post-hoc aggregation. In the first step, we generate candidates through pivot translation. This can be achieved with only a single model and facilitates knowledge transfer from high-resource pivot languages, resulting in candidates that are not only diverse but also more accurate. Next, in the aggregation step, we select k high-quality candidates from the generated candidates and merge them to generate a final translation that outperforms the existing candidates. Our experimental results show that our method produces translations of superior quality by leveraging candidates from pivot translation to capture the subtle nuances of the source sentence.
>
---
#### [replaced 022] HDLxGraph: Bridging Large Language Models and HDL Repositories via HDL Graph Databases
- **分类: cs.AR; cs.CL; cs.LG**

- **简介: 该论文属于HDL代码处理任务，旨在解决RAG在HDL项目中检索和生成效果不佳的问题。通过构建HDLxGraph框架，结合AST和DFG提升性能，并引入HDLSearch数据集进行评估。**

- **链接: [https://arxiv.org/pdf/2505.15701](https://arxiv.org/pdf/2505.15701)**

> **作者:** Pingqing Zheng; Jiayin Qin; Fuqi Zhang; Niraj Chitla; Zishen Wan; Shang Wu; Yu Cao; Caiwen Ding; Yang; Zhao
>
> **摘要:** Retrieval Augmented Generation (RAG) is an essential agent for Large Language Model (LLM) aided Description Language (HDL) tasks, addressing the challenges of limited training data and prohibitively long prompts. However, its performance in handling ambiguous queries and real-world, repository-level HDL projects containing thousands or even tens of thousands of code lines remains limited. Our analysis demonstrates two fundamental mismatches, structural and vocabulary, between conventional semantic similarity-based RAGs and HDL codes. To this end, we propose HDLxGraph, the first framework that integrates the inherent graph characteristics of HDLs with RAGs for LLM-assisted tasks. Specifically, HDLxGraph incorporates Abstract Syntax Trees (ASTs) to capture HDLs' hierarchical structures and Data Flow Graphs (DFGs) to address the vocabulary mismatch. In addition, to overcome the lack of comprehensive HDL search benchmarks, we introduce HDLSearch, an LLM generated dataset derived from real-world, repository-level HDL projects. Evaluations show that HDLxGraph improves search, debugging, and completion accuracy by 12.04%/12.22%/5.04% and by 11.59%/8.18%/4.07% over state-of-the-art similarity-based RAG and software-code Graph RAG baselines, respectively. The code of HDLxGraph and HDLSearch benchmark are available at this https URL.
>
---
#### [replaced 023] OTESGN: Optimal Transport-Enhanced Syntactic-Semantic Graph Networks for Aspect-Based Sentiment Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于方面情感分析任务，旨在解决现有方法在捕捉非线性关联和噪声适应上的不足。提出OTESGN模型，融合语法语义信息，提升情感分析性能。**

- **链接: [https://arxiv.org/pdf/2509.08612](https://arxiv.org/pdf/2509.08612)**

> **作者:** Xinfeng Liao; Xuanqi Chen; Lianxi Wang; Jiahuan Yang; Zhuowei Chen; Ziying Rong
>
> **备注:** This paper accepted by ICDM 2025 proposes OTESGN for ABSA, fusing syntactic-semantic signals via optimal transport and attention mechanisms. It achieves SOTA on Rest14, Laptop14 and Twitter (up to +1.30 Macro-F1 on Laptop14), with strong noise suppression and fine-grained sentiment capture capabilities. this https URL
>
> **摘要:** Aspect-based sentiment analysis (ABSA) aims to identify aspect terms and determine their sentiment polarity. While dependency trees combined with contextual semantics provide structural cues, existing approaches often rely on dot-product similarity and fixed graphs, which limit their ability to capture nonlinear associations and adapt to noisy contexts. To address these limitations, we propose the Optimal Transport-Enhanced Syntactic-Semantic Graph Network (OTESGN), a model that jointly integrates structural and distributional signals. Specifically, a Syntactic Graph-Aware Attention module models global dependencies with syntax-guided masking, while a Semantic Optimal Transport Attention module formulates aspect-opinion association as a distribution matching problem solved via the Sinkhorn algorithm. An Adaptive Attention Fusion mechanism balances heterogeneous features, and contrastive regularization enhances robustness. Extensive experiments on three benchmark datasets (Rest14, Laptop14, and Twitter) demonstrate that OTESGN delivers state-of-the-art performance. Notably, it surpasses competitive baselines by up to +1.30 Macro-F1 on Laptop14 and +1.01 on Twitter. Ablation studies and visualization analyses further highlight OTESGN's ability to capture fine-grained sentiment associations and suppress noise from irrelevant context.
>
---
#### [replaced 024] HaLoRA: Hardware-aware Low-Rank Adaptation for Large Language Models Based on Hybrid Compute-in-Memory Architecture
- **分类: cs.CL; cs.AR**

- **简介: 该论文属于模型优化任务，旨在解决LoRA在CIM架构中因RRAM噪声导致的性能下降问题。通过设计HaLoRA方法提升鲁棒性，实现高效低能耗推理。**

- **链接: [https://arxiv.org/pdf/2502.19747](https://arxiv.org/pdf/2502.19747)**

> **作者:** Taiqiang Wu; Chenchen Ding; Wenyong Zhou; Yuxin Cheng; Xincheng Feng; Shuqi Wang; Wendong Xu; Chufan Shi; Zhengwu Liu; Ngai Wong
>
> **备注:** 22 pages, Accepted by TODAES (ACM Transactions on Design Automation of Electronic Systems)
>
> **摘要:** Low-rank adaptation (LoRA) is a predominant parameter-efficient finetuning method for adapting large language models (LLMs) to downstream tasks. Meanwhile, Compute-in-Memory (CIM) architectures demonstrate superior energy efficiency due to their array-level parallel in-memory computing designs. In this paper, we propose deploying the LoRA-finetuned LLMs on the hybrid CIM architecture (i.e., pretrained weights onto energy-efficient Resistive Random-Access Memory (RRAM) and LoRA branches onto noise-free Static Random-Access Memory (SRAM)), reducing the energy cost to about 3\% compared to the Nvidia A100 GPU. However, the inherent noise of RRAM on the saved weights leads to performance degradation, simultaneously. To address this issue, we design a novel Hardware-aware Low-rank Adaptation (HaLoRA) method. The key insight is to train a LoRA branch that is robust toward such noise and then deploy it on noise-free SRAM, while the extra cost is negligible since the parameters of LoRAs are much fewer than pretrained weights (e.g., 0.15\% for LLaMA-3.2 1B model). To improve the robustness towards the noise, we theoretically analyze the gap between the optimization trajectories of the LoRA branch under both ideal and noisy conditions and further design an extra loss to minimize the upper bound of this gap. Therefore, we can enjoy both energy efficiency and accuracy during inference. Experiments finetuning the Qwen and LLaMA series demonstrate the effectiveness of HaLoRA across multiple reasoning tasks, achieving up to \textbf{22.7} improvement in average score while maintaining robustness at various noise types and noise levels.
>
---
#### [replaced 025] TokMem: One-Token Procedural Memory for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出TokMem，解决大模型任务重复处理与模块化困难问题。通过单个可训练记忆标记存储程序知识，实现高效任务控制与扩展。属于自然语言处理中的模型优化任务。**

- **链接: [https://arxiv.org/pdf/2510.00444](https://arxiv.org/pdf/2510.00444)**

> **作者:** Zijun Wu; Yongchang Hao; Lili Mou
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Large language models are typically controlled via prompts, which must be repeatedly re-processed for every new query and are difficult to reuse modularly. We introduce TokMem, a procedural memory framework that compiles each reusable task procedure into a single trainable memory token. Each token serves as both a procedure index and a generation control signal that steers generation, enabling targeted behaviors with constant-size overhead. TokMem keeps the backbone LLM frozen and stores procedural knowledge entirely in these dedicated units, so new procedures can be added continually without interfering with existing ones. We evaluate TokMem on two settings: atomic recall over 1,000 Super-Natural Instructions tasks and compositional recall on multi-step function-calling. Our results show that TokMem consistently outperforms retrieval-augmented prompting while avoiding repeated context overhead. Moreover, it matches or exceeds parameter-efficient fine-tuning with substantially fewer trainable parameters.
>
---
#### [replaced 026] A Two-Stage Multitask Vision-Language Framework for Explainable Crop Disease Visual Question Answering
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于作物病害视觉问答任务，旨在提升作物和病害识别的准确性与可解释性。通过两阶段多任务框架，结合视觉与语言模型，实现高效且可解释的病害分析。**

- **链接: [https://arxiv.org/pdf/2601.05143](https://arxiv.org/pdf/2601.05143)**

> **作者:** Md. Zahid Hossain; Most. Sharmin Sultana Samu; Md. Rakibul Islam; Md. Siam Ansary
>
> **备注:** Preprint, manuscript is under review
>
> **摘要:** Visual question answering (VQA) for crop disease analysis requires accurate visual understanding and reliable language generation. In this work, we present a lightweight and explainable vision-language framework for crop and disease identification from leaf images. The proposed approach integrates a Swin Transformer vision encoder with sequence-to-sequence language decoders. The vision encoder is first trained in a multitask setup for both plant and disease classification, and then frozen while the text decoders are trained, forming a two-stage training strategy that enhances visual representation learning and cross-modal alignment. We evaluate the model on the large-scale Crop Disease Domain Multimodal (CDDM) dataset using both classification and natural language generation metrics. Experimental results demonstrate near-perfect recognition performance, achieving 99.94% plant classification accuracy and 99.06% disease classification accuracy, along with strong BLEU, ROUGE and BERTScore results. Without fine-tuning, the model further generalizes well to the external PlantVillageVQA benchmark, achieving 83.18% micro accuracy in the VQA task. Our lightweight design outperforms larger vision-language baselines while using significantly fewer parameters. Explainability is assessed through Grad-CAM and token-level attribution, providing interpretable visual and textual evidence for predictions. Qualitative results demonstrate robust performance under diverse user-driven queries, highlighting the effectiveness of task-specific visual pretraining and the two-stage training methodology for crop disease visual question answering. An interactive demo of the proposed Swin-T5 model is publicly available as a Gradio-based application at this https URL for community use.
>
---
#### [replaced 027] Stochastic Self-Organization in Multi-Agent Systems
- **分类: cs.MA; cs.CL; cs.LG**

- **简介: 该论文研究多智能体系统中的协作优化问题，旨在提升基于大语言模型的多智能体协作效率。通过引入自组织框架SelfOrg，实现动态通信结构，提高任务完成的稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2510.00685](https://arxiv.org/pdf/2510.00685)**

> **作者:** Nurbek Tastan; Samuel Horvath; Karthik Nandakumar
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Multi-agent systems (MAS) based on Large Language Models (LLMs) have the potential to solve tasks that are beyond the reach of any single LLM. However, this potential can only be realized when the collaboration mechanism between agents is optimized. Specifically, optimizing the communication structure between agents is critical for fruitful collaboration. Most existing approaches rely on fixed topologies, pretrained graph generators, optimization over edges, or employ external LLM judges, thereby adding to the complexity. In this work, we introduce a response-conditioned framework that adapts communication on-the-fly. Agents independently generate responses to the user query and assess peer contributions using an approximation of the Shapley value. A directed acyclic graph (DAG) is then constructed to regulate the propagation of the responses among agents, which ensures stable and efficient message transmission from high-contributing agents to others. This graph is dynamically updated based on the agent responses from the previous collaboration round. Since the proposed framework enables the self-organization of agents without additional supervision or training, we refer to it as SelfOrg. The SelfOrg framework goes beyond task- and query-level optimization and takes into account the stochastic nature of agent responses. Experiments with both strong and weak LLM backends demonstrate robust performance, with significant gains in the weak regime where prior methods collapse. We also theoretically show that multiple agents increase the chance of correctness and that the correct responses naturally dominate the information flow.
>
---
#### [replaced 028] Condition-Gated Reasoning for Context-Dependent Biomedical Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 biomedical QA 任务，解决临床决策中条件依赖的问题。提出 CondMedQA 基准和 CGR 框架，以构建条件感知的知识图谱并选择性激活推理路径。**

- **链接: [https://arxiv.org/pdf/2602.17911](https://arxiv.org/pdf/2602.17911)**

> **作者:** Jash Rajesh Parekh; Wonbin Kweon; Joey Chan; Rezarta Islamaj; Robert Leaman; Pengcheng Jiang; Chih-Hsuan Wei; Zhizheng Wang; Zhiyong Lu; Jiawei Han
>
> **摘要:** Current biomedical question answering (QA) systems often assume that medical knowledge applies uniformly, yet real-world clinical reasoning is inherently conditional: nearly every decision depends on patient-specific factors such as comorbidities and contraindications. Existing benchmarks do not evaluate such conditional reasoning, and retrieval-augmented or graph-based methods lack explicit mechanisms to ensure that retrieved knowledge is applicable to given context. To address this gap, we propose CondMedQA, the first benchmark for conditional biomedical QA, consisting of multi-hop questions whose answers vary with patient conditions. Furthermore, we propose Condition-Gated Reasoning (CGR), a novel framework that constructs condition-aware knowledge graphs and selectively activates or prunes reasoning paths based on query conditions. Our findings show that CGR more reliably selects condition-appropriate answers while matching or exceeding state-of-the-art performance on biomedical QA benchmarks, highlighting the importance of explicitly modeling conditionality for robust medical reasoning.
>
---
#### [replaced 029] Discovering Semantic Latent Structures in Psychological Scales: A Response-Free Pathway to Efficient Simplification
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于心理量表简化任务，旨在解决传统方法依赖大量数据的问题。通过语义主题建模，发现潜在结构并简化量表，提升效率与一致性。**

- **链接: [https://arxiv.org/pdf/2602.12575](https://arxiv.org/pdf/2602.12575)**

> **作者:** Bo Wang; Yuxuan Zhang; Yueqin Hu; Hanchao Hou; Kaiping Peng; Shiguang Ni
>
> **备注:** 79 pages, 20 figures; parameter perturbation result of epoch-cn updated; minor revisions on grammars
>
> **摘要:** Psychological scale refinement traditionally relies on response-based methods such as factor analysis, item response theory, and network psychometrics to optimize item composition. Although rigorous, these approaches require large samples and may be constrained by data availability and cross-cultural comparability. Recent advances in natural language processing suggest that the semantic structure of questionnaire items may encode latent construct organization, offering a complementary response-free perspective. We introduce a topic-modeling framework that operationalizes semantic latent structure for scale simplification. Items are encoded using contextual sentence embeddings and grouped via density-based clustering to discover latent semantic factors without predefining their number. Class-based term weighting derives interpretable topic representations that approximate constructs and enable merging of semantically adjacent clusters. Representative items are selected using membership criteria within an integrated reduction pipeline. We benchmarked the framework across DASS, IPIP, and EPOCH, evaluating structural recovery, internal consistency, factor congruence, correlation preservation, and reduction efficiency. The proposed method recovered coherent factor-like groupings aligned with established constructs. Selected items reduced scale length by 60.5% on average while maintaining psychometric adequacy. Simplified scales showed high concordance with original factor structures and preserved inter-factor correlations, indicating that semantic latent organization provides a response-free approximation of measurement structure. Our framework formalizes semantic structure as an inspectable front-end for scale construction and reduction. To facilitate adoption, we provide a visualization-supported tool enabling one-click semantic analysis and structured simplification.
>
---
#### [replaced 030] HatePrototypes: Interpretable and Transferable Representations for Implicit and Explicit Hate Speech Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于仇恨言论检测任务，旨在解决隐性与显性仇恨内容识别问题。通过构建可解释的原型表示，实现跨任务和跨基准的迁移学习，提升检测效率与效果。**

- **链接: [https://arxiv.org/pdf/2511.06391](https://arxiv.org/pdf/2511.06391)**

> **作者:** Irina Proskurina; Marc-Antoine Carpentier; Julien Velcin
>
> **摘要:** Optimization of offensive content moderation models for different types of hateful messages is typically achieved through continued pre-training or fine-tuning on new hate speech benchmarks. However, existing benchmarks mainly address explicit hate toward protected groups and often overlook implicit or indirect hate, such as demeaning comparisons, calls for exclusion or violence, and subtle discriminatory language that still causes harm. While explicit hate can often be captured through surface features, implicit hate requires deeper, full-model semantic processing. In this work, we question the need for repeated fine-tuning and analyze the role of HatePrototypes, class-level vector representations derived from language models optimized for hate speech detection and safety moderation. We find that these prototypes, built from as few as 50 examples per class, enable cross-task transfer between explicit and implicit hate, with interchangeable prototypes across benchmarks. Moreover, we show that parameter-free early exiting with prototypes is effective for both hate types. We release the code, prototype resources, and evaluation scripts to support future research on efficient and transferable hate speech detection.
>
---
#### [replaced 031] Jr. AI Scientist and Its Risk Report: Autonomous Scientific Exploration from a Baseline Paper
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于AI科学探索任务，旨在开发自主AI科学家系统Jr. AI Scientist，解决AI在科研中的可靠性与风险问题。工作包括构建模拟研究流程的系统，并评估其生成论文的质量与风险。**

- **链接: [https://arxiv.org/pdf/2511.04583](https://arxiv.org/pdf/2511.04583)**

> **作者:** Atsuyuki Miyai; Mashiro Toyooka; Takashi Otonari; Zaiying Zhao; Kiyoharu Aizawa
>
> **备注:** TMLR2026. Issues, comments, and questions are all welcome in this https URL
>
> **摘要:** Understanding the current capabilities and risks of AI Scientist systems is essential for ensuring trustworthy and sustainable AI-driven scientific progress while preserving the integrity of the academic ecosystem. To this end, we develop Jr. AI Scientist, a state-of-the-art autonomous AI scientist system that mimics the core research workflow of a novice student researcher: Given the baseline paper from the human mentor, it analyzes its limitations, formulates novel hypotheses for improvement, validates them through rigorous experimentation, and writes a paper with the results. Unlike previous approaches that assume full automation or operate on small-scale code, Jr. AI Scientist follows a well-defined research workflow and leverages modern coding agents to handle complex, multi-file implementations, leading to scientifically valuable contributions. Through our experiments, the Jr. AI Scientist successfully generated new research papers that build upon real NeurIPS, IJCV, and ICLR works by proposing and implementing novel algorithms. For evaluation, we conducted automated assessments using AI Reviewers, author-led evaluations, and submissions to Agents4Science, a venue dedicated to AI-driven scientific contributions. The findings demonstrate that Jr. AI Scientist generates papers receiving higher review scores by DeepReviewer than existing fully automated systems. Nevertheless, we identify important limitations from both the author evaluation and the Agents4Science reviews, indicating the potential risks of directly applying current AI Scientist systems and key challenges for future research. Finally, we comprehensively report various risks identified during development. We believe this study clarifies the current role and limitations of AI Scientist systems, offering insights into the areas that still require human expertise and the risks that may emerge as these systems evolve.
>
---
#### [replaced 032] More Bang for the Buck: Process Reward Modeling with Entropy-Driven Uncertainty
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出EDU-PRM，用于数学推理中的过程奖励建模，解决手动标注成本高、分割不准确的问题。通过熵驱动方法自动识别步骤边界，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2503.22233](https://arxiv.org/pdf/2503.22233)**

> **作者:** Lang Cao; Renhong Chen; Yingtian Zou; Chao Peng; Huacong Xu; Yuxian Wang; Wu Ning; Qian Chen; Mofan Peng; Zijie Chen; Peishuo Su; Yitong Li
>
> **摘要:** We introduce the Entropy-Driven Uncertainty Process Reward Model (EDU-PRM), a novel entropy-driven training framework for process reward modeling that enables dynamic, uncertainty-aligned segmentation of complex reasoning steps, eliminating the need for costly manual step annotations. Unlike previous Process Reward Models (PRMs) that rely on static partitioning and human labeling, EDU-PRM automatically anchors step boundaries at tokens with high predictive entropy, effectively capturing intrinsic logical transitions and facilitating efficient exploration of diverse reasoning paths. On the ProcessBench benchmark, EDU-PRM outperforms strong public PRM baselines, such as Math-Shepherd PRM and Omega PRM, and EDU-PRM achieves comparable results with SOTA models while only using 1.5% training data. Furthermore, by leveraging our proposed EDU sampling strategy, we observe accuracy boosts from 64.7% to 67.3% for generative reasoning tasks, accompanied by a reduction of 32% in token usage. These findings underscore the potential of EDU-PRM as a scalable and annotation-efficient paradigm for process supervision in mathematical reasoning, paving the way for more efficient and robust approaches to complex mathematical problem solving.
>
---
#### [replaced 033] LaTeX Compilation: Challenges in the Era of LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨LLMs在科学写作中的挑战，分析TeX的不足，并提出Mogan STEM作为改进方案，提升编译效率与工具生态。**

- **链接: [https://arxiv.org/pdf/2603.02873](https://arxiv.org/pdf/2603.02873)**

> **作者:** Tianyou Liu; Ziqiang Li; Xurui Liu; Yansong Li
>
> **备注:** 25 pages, 12 figures
>
> **摘要:** As large language models (LLMs) increasingly assist scientific writing, limitations and the significant token cost of TeX become more and more visible. This paper analyzes TeX's fundamental defects in compilation and user experience design to illustrate its limitations on compilation efficiency, generated semantics, error localization, and tool ecosystem in the era of LLMs. As an alternative, Mogan STEM, a WYSIWYG structured editor, is introduced. Mogan outperforms TeX in the above aspects by its efficient data structure, fast rendering, and on-demand plugin loading. Extensive experiments are conducted to verify the benefits on compilation/rendering time and performance in LLM tasks. Furthermore, we show that due to Mogan's lower information entropy, it is more efficient to use .tmu (the document format of Mogan) to fine-tune LLMs than TeX. Therefore, we launch an appeal for larger experiments on LLM training using the .tmu format.
>
---
#### [replaced 034] GRADIEND: Feature Learning within Neural Networks Exemplified through Biases
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于AI伦理任务，旨在解决模型中的社会偏见问题。通过分析模型梯度，学习并消除性别、种族等偏见特征，实现去偏且保持模型性能。**

- **链接: [https://arxiv.org/pdf/2502.01406](https://arxiv.org/pdf/2502.01406)**

> **作者:** Jonathan Drechsel; Steffen Herbold
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** AI systems frequently exhibit and amplify social biases, leading to harmful consequences in critical areas. This study introduces a novel encoder-decoder approach that leverages model gradients to learn a feature neuron encoding societal bias information such as gender, race, and religion. We show that our method can not only identify which weights of a model need to be changed to modify a feature, but even demonstrate that this can be used to rewrite models to debias them while maintaining other capabilities. We demonstrate the effectiveness of our approach across various model architectures and highlight its potential for broader applications.
>
---
#### [replaced 035] Efficient Continual Learning for Small Language Models with a Discrete Key-Value Bottleneck
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的持续学习任务，旨在解决模型在更新数据时遗忘旧知识的问题。通过引入离散键值瓶颈，实现高效且稳定的持续学习。**

- **链接: [https://arxiv.org/pdf/2412.08528](https://arxiv.org/pdf/2412.08528)**

> **作者:** Andor Diera; Lukas Galke; Fabian Karl; Ansgar Scherp
>
> **摘要:** Continual learning remains a challenge across various natural language processing (NLP) tasks, as models updated with new training data often risk catastrophic forgetting of previously acquired knowledge. We introduce a discrete key-value bottleneck (DKVB) for encoder-only language models, enabling efficient continual learning through localized updates. Inspired by a discrete key-value bottleneck in vision, we consider new and NLP-specific challenges. We compare different bottleneck architectures for NLP and introduce a new, task-independent initialization technique for the discrete keys. We evaluate our DKVB for NLP in four continual learning scenarios and show that it alleviates catastrophic forgetting. Our experiments demonstrate that the proposed approach achieves competitive performance compared to popular continual learning methods while incurring lower computational costs. Furthermore, we show that DKVB remains effective even in challenging single-head continual learning scenarios where no task ID is provided.
>
---
#### [replaced 036] SETUP: Sentence-level English-To-Uniform Meaning Representation Parser
- **分类: cs.CL**

- **简介: 该论文属于英文到统一语义表示（UMR）的句法解析任务，旨在解决自动生成准确UMR图的问题。工作包括引入两种解析方法，并提出性能优越的SETUP模型。**

- **链接: [https://arxiv.org/pdf/2512.07068](https://arxiv.org/pdf/2512.07068)**

> **作者:** Emma Markle; Javier Gutierrez Bach; Shira Wein
>
> **备注:** LREC 2026
>
> **摘要:** Uniform Meaning Representation (UMR) is a novel graph-based semantic representation which captures the core meaning of a text, with flexibility incorporated into the annotation schema such that the breadth of the world's languages can be annotated (including low-resource languages). While UMR shows promise in enabling language documentation, improving low-resource language technologies, and adding interpretability, the downstream applications of UMR can only be fully explored when text-to-UMR parsers enable the automatic large-scale production of accurate UMR graphs at test time. Prior work on text-to-UMR parsing is limited to date. In this paper, we introduce two methods for English text-to-UMR parsing, one of which fine-tunes existing parsers for Abstract Meaning Representation and the other, which leverages a converter from Universal Dependencies, using prior work as a baseline. Our best-performing model, which we call SETUP, achieves an AnCast score of 84 and a SMATCH++ score of 91, indicating substantial gains towards automatic UMR parsing.
>
---
#### [replaced 037] Multi-modal, Multi-task, Multi-criteria Automatic Evaluation with Vision Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态任务，旨在解决VLM生成文本的评估问题。提出HarmonicEval评估方法，适应多任务场景，并构建MMHE基准测试。**

- **链接: [https://arxiv.org/pdf/2412.14613](https://arxiv.org/pdf/2412.14613)**

> **作者:** Masanari Ohi; Masahiro Kaneko; Naoaki Okazaki; Nakamasa Inoue
>
> **摘要:** Vision-language models (VLMs) have shown impressive abilities across a range of multi-modal tasks. However, existing metrics for evaluating the quality of text generated by VLMs typically focus on an overall evaluation for a specific task, such as image captioning. While the overall evaluation is essential for any task, the criteria prioritized can differ depending on the task, making it challenging for current metrics to adapt to multi-task scenarios. To address this limitation, we propose HarmonicEval, a reference-free comprehensive evaluation metric that aggregates criterion-wise scores to produce the overall score in a bottom-up manner. Furthermore, to assess the generalizability of automatic evaluation metrics in multi-task scenarios, we construct the Multi-task Multi-criteria Human Evaluation (MMHE) benchmark, which comprises 18,000 expert human judgments across four multi-modal tasks. Our experiments demonstrate that HarmonicEval achieves higher correlations with human judgments than conventional metrics while providing numerical scores for each criterion. Project page: this https URL
>
---
#### [replaced 038] EFT-CoT: A Multi-Agent Chain-of-Thought Framework for Emotion-Focused Therapy
- **分类: cs.CL**

- **简介: 该论文属于心理健康问答任务，旨在解决传统方法对情绪处理支持不足的问题。提出EFT-CoT框架，通过多智能体实现情绪聚焦治疗，提升情感深度与专业性。**

- **链接: [https://arxiv.org/pdf/2601.17842](https://arxiv.org/pdf/2601.17842)**

> **作者:** Lanqing Du; Yunong Li; YuJie Long; Shihong Chen
>
> **摘要:** The use of large language models (LLMs) for Mental Health Question Answering (MHQA) offers a promising way to alleviate shortages in mental health resources. However, prior work has mainly relied on Cognitive Behavioral Therapy (CBT) and predominantly follows a top-down strategy centered on rational cognitive restructuring, providing limited support for embodied experience and primary emotion processing. To address this gap, we propose EFT-CoT, a multi-agent chain-of-thought framework grounded in Emotion-Focused Therapy (EFT). EFT-CoT operationalizes intervention as a three-stage workflow: Embodied Perception, Cognitive Exploration, and Narrative Intervention. The framework employs eight specialized agents to model key processes including somatic awareness mapping, adaptive evaluation, core belief extraction, and narrative restructuring. Based on this framework, we construct EFT-Instruct, a high-quality instruction-tuning dataset built from process-level augmentation of about 67,000 real help-seeking texts, and further fine-tune a dedicated model, EFT-LLM. Experiments show that EFT-LLM consistently outperforms strong baselines and human responses in empathic depth and structural professionalism. Ablation studies further verify the contribution of key mechanisms, while white-box auditing demonstrates the consistency and traceability of critical intermediate states. Overall, this work provides a reproducible framework-data-model pipeline for embedding EFT mechanisms into LLM-based mental health support.
>
---
#### [replaced 039] SPOT: An Annotated French Corpus and Benchmark for Detecting Critical Interventions in Online Conversations
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出SPOT任务，用于检测在线对话中的关键干预。通过构建法语标注语料库，解决非英语社交媒体中监督学习的挑战，并验证模型性能。**

- **链接: [https://arxiv.org/pdf/2511.07405](https://arxiv.org/pdf/2511.07405)**

> **作者:** Manon Berriche; Célia Nouri; Chloée Clavel; Jean-Philippe Cointet
>
> **摘要:** We introduce SPOT (Stopping Points in Online Threads), the first annotated corpus translating the sociological concept of stopping point into a reproducible NLP task. Stopping points are ordinary critical interventions that pause or redirect online discussions through a range of forms (irony, subtle doubt or fragmentary arguments) that frameworks like counterspeech or social correction often overlook. We operationalize this concept as a binary classification task and provide reliable annotation guidelines. The corpus contains 43,305 manually annotated French Facebook comments linked to URLs flagged as false information by social media users, enriched with contextual metadata (article, post, parent comment, page or group, and source). We benchmark fine-tuned encoder models (CamemBERT) and instruction-tuned LLMs under various prompting strategies. Results show that fine-tuned encoders outperform prompted LLMs in F1 score by more than 10 percentage points, confirming the importance of supervised learning for emerging non-English social media tasks. Incorporating contextual metadata further improves encoder models F1 scores from 0.75 to 0.78. We release the anonymized dataset, along with the annotation guidelines and code in our code repository, to foster transparency and reproducible research.
>
---
#### [replaced 040] A Simple "Motivation" Can Enhance Reinforcement Finetuning of Large Reasoning Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，旨在提升大模型的推理能力。针对RLVR效率低的问题，提出MeRF方法，通过注入奖励规则增强模型优化目标意识，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2506.18485](https://arxiv.org/pdf/2506.18485)**

> **作者:** Junjie Zhang; Guozheng Ma; Shunyu Liu; Haoyu Wang; Jiaxing Huang; Ting-En Lin; Fei Huang; Yongbin Li; Dacheng Tao
>
> **摘要:** Reinforcement Learning with Verifiable Rewards~(RLVR) has emerged as a powerful learn-to-reason paradigm for large reasoning models to tackle complex tasks. However, the current RLVR paradigm is still not efficient enough, as it works in a trial-and-error manner. To perform better, the model needs to explore the reward space by numerously generating responses and learn from fragmented reward signals, blind to the overall reward patterns. Fortunately, verifiable rewards make the natural language description of the reward function possible, and meanwhile, LLMs have demonstrated strong in-context learning ability. This motivates us to explore if large reasoning models can benefit from a \textbf{motivation} of the task, \textit{i.e.}, awareness of the reward function, during the reinforcement finetuning process, as we humans sometimes do when learning. In this paper, we introduce \textit{\textbf{M}otivation-\textbf{e}nhanced \textbf{R}einforcement \textbf{F}inetuning}~(\textbf{MeRF}), an intuitive yet effective method enhancing reinforcement finetuning of LLMs by involving \emph{``telling LLMs rules of the game''}. Specifically, \textbf{MeRF} directly injects the reward specification into the prompt, which serves as an in-context motivation for the model to be aware of the optimization objective. This simple modification leverages the in-context learning ability of LLMs, aligning generation with optimization, thereby incentivizing the model to generate desired outputs from both inner motivation and external reward. Empirical evaluations demonstrate that \textbf{MeRF} achieves substantial performance gains over the RLVR baseline. Moreover, ablation studies show that MeRF performs better with greater consistency between the in-context motivation and the external reward function, while the model also demonstrates an ability to adapt to misleading motivations through reinforcement finetuning.
>
---
#### [replaced 041] Idiom Understanding as a Tool to Measure the Dialect Gap
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决方言理解能力评估问题。通过构建三个法语方言语料库，测试语言模型对地区俚语的掌握程度，揭示语言模型在方言理解上的差距。**

- **链接: [https://arxiv.org/pdf/2510.05026](https://arxiv.org/pdf/2510.05026)**

> **作者:** David Beauchemin; Yan Tremblay; Mohamed Amine Youssef; Richard Khoury
>
> **备注:** Submitted to ACL 2026
>
> **摘要:** The tasks of idiom understanding and dialect understanding are both well-established benchmarks in natural language processing. In this paper, we propose combining them, and using regional idioms as a test of dialect understanding. Towards this end, we propose three new benchmark datasets for the Quebec dialect of French: QFrCoRE, which contains 4,633 instances of idiomatic phrases, and QFrCoRT, which comprises 171 regional instances of idiomatic words, and a new benchmark for French Metropolitan expressions, MFrCoE, which comprises 4,938 phrases. We explain how to construct these corpora, so that our methodology can be replicated for other dialects. Our experiments with 111 LLMs reveal a critical disparity in dialectal competence: while models perform well on French Metropolitan, 65.77% of them perform significantly worse on Quebec idioms, with only 9.0% favoring the regional dialect. These results confirm that our benchmarks are a reliable tool for quantifying the dialect gap and that prestige-language proficiency does not guarantee regional dialect understanding.
>
---
#### [replaced 042] CeRA: Breaking the Linear Ceiling of Low-Rank Adaptation via Manifold Expansion
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于参数高效微调任务，旨在解决LoRA在复杂推理中因线性约束导致的性能瓶颈。通过引入CeRA，提升模型表达能力，实现更优效果。**

- **链接: [https://arxiv.org/pdf/2602.22911](https://arxiv.org/pdf/2602.22911)**

> **作者:** Hung-Hsuan Chen
>
> **摘要:** Low-Rank Adaptation (LoRA) dominates parameter-efficient fine-tuning (PEFT). However, it faces a critical ``linear ceiling'' in complex reasoning tasks: simply increasing the rank yields diminishing returns due to intrinsic linear constraints. We introduce CeRA (Capacity-enhanced Rank Adaptation), a weight-level parallel adapter that injects SiLU gating and structural dropout to induce manifold expansion. On the SlimOrca benchmark, CeRA breaks this linear barrier: at rank 64 (PPL 3.89), it outperforms LoRA at rank 512 (PPL 3.90), demonstrating superior spectral efficiency. This advantage generalizes to mathematical reasoning, where CeRA achieves a perplexity of 1.97 on MathInstruct, significantly surpassing LoRA's saturation point of 2.07. Mechanism analysis via Singular Value Decomposition (SVD) confirms that CeRA activates the dormant tail of the singular value spectrum, effectively preventing the rank collapse observed in linear methods.
>
---
#### [replaced 043] CyclicJudge: Mitigating Judge Bias Efficiently in LLM-based Evaluation
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决LLM作为评判者时的系统性偏差问题。通过方差分解和轮换评判策略，提出CyclicJudge方法，有效消除偏差并保持评估效率。**

- **链接: [https://arxiv.org/pdf/2603.01865](https://arxiv.org/pdf/2603.01865)**

> **作者:** Ziyi Zhu; Olivier Tieleman; Alexey Bukhtiyarov; Jinghong Chen
>
> **摘要:** LLM-as-judge evaluation has become standard practice for open-ended model assessment; however, judges exhibit systematic biases that cannot be eliminated by increasing the number of scenarios or generations. These biases are often similar in magnitude to the model differences that benchmarks are designed to detect, resulting in unreliable rankings when single-judge evaluations are used. This work introduces a variance decomposition that partitions benchmark score variance into scenario, generation, judge, and residual components. Based on this analysis, CyclicJudge, a round-robin assignment of judges to scenarios, is demonstrated to be the optimal strategy for a fixed judge-call budget. It eliminates bias precisely while requiring each judge only once per cycle, matching the cost of single-judge evaluation. Empirical results on MT-Bench and MindEval validate the effectiveness of CyclicJudge as predicted, across both general-purpose and domain-specific evaluation settings.
>
---
#### [replaced 044] CyclicReflex: Improving Reasoning Models via Cyclical Reflection Token Scheduling
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大模型的推理能力。针对推理过程中反射标记使用不当的问题，提出CyclicReflex方法，通过动态调节反射标记频率提高模型性能。**

- **链接: [https://arxiv.org/pdf/2506.11077](https://arxiv.org/pdf/2506.11077)**

> **作者:** Chongyu Fan; Yihua Zhang; Jinghan Jia; Alfred Hero; Sijia Liu
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Large reasoning models (LRMs), such as OpenAI's o1 and DeepSeek-R1, harness test-time scaling to perform multi-step reasoning for complex problem-solving. This reasoning process, executed before producing final answers, is often guided by special juncture tokens that prompt self-evaluative reflection. These transition markers and reflective cues are referred to as "reflection tokens" (e.g., "wait", "but", "alternatively"). In this work, we treat reflection tokens as a "resource" and introduce the problem of resource allocation, aimed at improving the test-time compute performance of LRMs by adaptively regulating the frequency and placement of reflection tokens. Through empirical analysis, we show that both excessive and insufficient use of reflection tokens, referred to as over-reflection and under-reflection, can degrade model performance. To better understand this trade-off, we draw an analogy between reflection token usage and learning rate scheduling in optimization. Building on this insight, We propose cyclical reflection token scheduling (termed CyclicReflex), a training-free decoding strategy that dynamically modulates reflection token logits with a bidirectional, position-dependent triangular waveform, incurring no additional computation cost. Experiments on MATH500, AIME2024/2025, AMC2023, GPQA Diamond and LiveCodeBench demonstrate that CyclicReflex consistently improves performance across model sizes (1.5B-14B), outperforming standard decoding and recent approaches such as TIP (thought switching penalty) and S1. Codes are available at this https URL.
>
---
#### [replaced 045] Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于AI安全研究任务，探讨自进化大语言模型代理的潜在风险，即“误进化”。研究分析了模型、记忆、工具和流程四个路径中的风险，提出需建立新的安全范式。**

- **链接: [https://arxiv.org/pdf/2509.26354](https://arxiv.org/pdf/2509.26354)**

> **作者:** Shuai Shao; Qihan Ren; Chen Qian; Boyi Wei; Dadi Guo; Jingyi Yang; Xinhao Song; Linfeng Zhang; Weinan Zhang; Dongrui Liu; Jing Shao
>
> **备注:** Published in ICLR 2026
>
> **摘要:** Advances in Large Language Models (LLMs) have enabled a new class of self-evolving agents that autonomously improve through interaction with the environment, demonstrating strong capabilities. However, self-evolution also introduces novel risks overlooked by current safety research. In this work, we study the case where an agent's self-evolution deviates in unintended ways, leading to undesirable or even harmful outcomes. We refer to this as Misevolution. To provide a systematic investigation, we evaluate misevolution along four key evolutionary pathways: model, memory, tool, and workflow. Our empirical findings reveal that misevolution is a widespread risk, affecting agents built even on top-tier LLMs (e.g., Gemini-2.5-Pro). Different emergent risks are observed in the self-evolutionary process, such as the degradation of safety alignment after memory accumulation, or the unintended introduction of vulnerabilities in tool creation and reuse. To our knowledge, this is the first study to systematically conceptualize misevolution and provide empirical evidence of its occurrence, highlighting an urgent need for new safety paradigms for self-evolving agents. Finally, we discuss potential mitigation strategies to inspire further research on building safer and more trustworthy self-evolving agents. Our code and data are available at this https URL . Warning: this paper includes examples that may be offensive or harmful in nature.
>
---
#### [replaced 046] Goal Alignment in LLM-Based User Simulators for Conversational AI
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话AI任务，解决用户模拟器目标对齐问题。提出UGST框架，提升模拟器在多轮对话中保持目标一致的能力。**

- **链接: [https://arxiv.org/pdf/2507.20152](https://arxiv.org/pdf/2507.20152)**

> **作者:** Shuhaib Mehri; Xiaocheng Yang; Takyoung Kim; Gokhan Tur; Shikib Mehri; Dilek Hakkani-Tür
>
> **摘要:** User simulators are essential to conversational AI, enabling scalable agent development and evaluation through simulated interactions. While current Large Language Models (LLMs) have advanced user simulation capabilities, we reveal that they struggle to consistently demonstrate goal-oriented behavior across multi-turn conversations--a critical limitation that compromises their reliability in downstream applications. We introduce User Goal State Tracking (UGST), a novel framework that tracks user goal progression throughout conversations. Leveraging UGST, we present a three-stage methodology for developing user simulators that can autonomously track goal progression and reason to generate goal-aligned responses. Moreover, we establish comprehensive evaluation metrics for measuring goal alignment in user simulators, and demonstrate that our approach yields substantial improvements across two benchmarks (MultiWOZ 2.4 and {\tau}-Bench). Our contributions address a critical gap in conversational AI and establish UGST as an essential framework for developing goal-aligned user simulators.
>
---
#### [replaced 047] Understand Then Memory: A Cognitive Gist-Driven RAG Framework with Global Semantic Diffusion
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CogitoRAG框架，解决RAG中语义完整性丢失问题，通过提取语义核心和知识图谱实现更准确的检索与生成。任务为问答与多任务生成。**

- **链接: [https://arxiv.org/pdf/2602.15895](https://arxiv.org/pdf/2602.15895)**

> **作者:** Pengcheng Zhou; Haochen Li; Zhiqiang Nie; JiaLe Chen; Qing Gong; Weizhen Zhang; Chun Yu
>
> **摘要:** Retrieval-Augmented Generation (RAG) effectively mitigates hallucinations in LLMs by incorporating external knowledge. However, the inherent discrete representation of text in existing frameworks often results in a loss of semantic integrity, leading to retrieval deviations. Inspired by the human episodic memory mechanism, we propose CogitoRAG, a RAG framework that simulates human cognitive memory processes. The core of this framework lies in the extraction and evolution of the Semantic Gist. During the offline indexing stage, CogitoRAG first deduces unstructured corpora into gist memory corpora, which are then transformed into a multi-dimensional knowledge graph integrating entities, relational facts, and memory nodes. In the online retrieval stage, the framework handles complex queries via Query Decomposition Module that breaks them into comprehensive sub-queries, mimicking the cognitive decomposition humans employ for complex information. Subsequently, Entity Diffusion Module performs associative retrieval across the graph, guided by structural relevance and an entity-frequency reward mechanism. Furthermore, we propose the CogniRank algorithm, which precisely reranks candidate passages by fusing diffusion-derived scores with semantic similarity. The final evidence is delivered to the generator in a passage-memory pairing format, providing high-density information support. Experimental results across five mainstream QA benchmarks and multi-task generation on GraphBench demonstrate that CogitoRAG significantly outperforms state-of-the-art RAG methods, showcasing superior capabilities in complex knowledge integration and reasoning.
>
---
#### [replaced 048] Improving X-Codec-2.0 for Multi-Lingual Speech: 25 Hz Latent Rate and 24 kHz Sampling
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音编码任务，旨在提升多语言语音的压缩效率和音质。通过降低隐层频率至25 Hz并提高采样率至24 kHz，改进X-Codec-2.0模型性能。**

- **链接: [https://arxiv.org/pdf/2601.20185](https://arxiv.org/pdf/2601.20185)**

> **作者:** Husein Zolkepli
>
> **摘要:** X-Codec-2.0 has shown strong performance in neural audio compression and multilingual speech modeling, operating at a 50 Hz latent rate and a 16 kHz sampling rate using frozen HuBERT features. While effective, this configuration limits temporal efficiency and audio fidelity. In this work, we explore a simple and effective modification by introducing additional pooling and increasing the decoder hop size. This reduces the latent rate from 50 Hz to 25 Hz and simultaneously raises the output sampling rate from 16 kHz to 24 kHz, improving efficiency and perceptual quality without altering the core architecture. Evaluated on the multilingual Common Voice 17 test set, the proposed configuration achieves a 0.29 MOS improvement over the original X-Codec-2.0 baseline based on UTMOSv2, and attains the best reported performance among all codecs operating at 25 Hz. The source code, checkpoints, and generation comparisons are released at \href{this https URL}{this https URL}.
>
---
#### [replaced 049] Why Code, Why Now: Learnability, Computability, and the Real Limits of Machine Learning
- **分类: cs.LG; cs.CL**

- **简介: 论文探讨代码生成与强化学习在可学习性上的差异，属于机器学习任务。它分析了可表达性、可计算性和可学习性的关系，旨在揭示ML进展的真正限制。**

- **链接: [https://arxiv.org/pdf/2602.13934](https://arxiv.org/pdf/2602.13934)**

> **作者:** Zhimin Zhao
>
> **摘要:** Code generation has progressed more reliably than reinforcement learning, largely because code has an information structure that makes it learnable. Code provides dense, local, verifiable feedback at every token, whereas most reinforcement learning problems do not. This difference in feedback quality is not binary but graded. We propose a five-level hierarchy of learnability based on information structure and argue that the ceiling on ML progress depends less on model size than on whether a task is learnable at all. The hierarchy rests on a formal distinction among three properties of computational problems (expressibility, computability, and learnability). We establish their pairwise relationships, including where implications hold and where they fail, and present a unified template that makes the structural differences explicit. The analysis suggests why supervised learning on code scales predictably while reinforcement learning does not, and why the common assumption that scaling alone will solve remaining ML challenges warrants scrutiny.
>
---
#### [replaced 050] PrivMedChat: End-to-End Differentially Private RLHF for Medical Dialogue Systems
- **分类: cs.CL**

- **简介: 该论文属于医疗对话系统任务，解决隐私泄露问题。提出PrivMedChat框架，实现差分隐私的RLHF，保障数据安全同时提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.03054](https://arxiv.org/pdf/2603.03054)**

> **作者:** Sudip Bhujel
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** Large language models are increasingly used for patient-facing medical assistance and clinical decision support, but adapting them to clinical dialogue often requires supervision derived from doctor-patient conversations that may contain sensitive information. Conventional supervised fine-tuning and reinforcement learning from human feedback (RLHF) can amplify memorization, enabling membership inference and disclosure of rare training-set details. We present PrivMedChat (Private Medical Chat), an end-to-end framework for differentially private RLHF (DP-RLHF) for medical dialogue systems. Our approach enforces differential privacy at each training stage that accesses dialogue-derived supervision, combining DP-SGD for supervised fine-tuning and reward model learning from preference pairs, and DP-aware policy optimization for alignment. To avoid costly clinician labeling, we introduce an annotation-free preference construction strategy that pairs physician responses with filtered non-expert generations. We evaluate PrivMedChat across medical dialogue tasks and assess utility, safety, and privacy under consistent privacy accounting, thereby providing a practical pathway to align medical chatbots while offering formal privacy guarantees. We open-source our code at this https URL.
>
---
#### [replaced 051] A Geometric Taxonomy of Hallucinations in LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型幻觉检测问题。通过几何分析提出幻觉分类，并设计两种检测方法，提升幻觉识别效果。**

- **链接: [https://arxiv.org/pdf/2602.13224](https://arxiv.org/pdf/2602.13224)**

> **作者:** Javier Marín
>
> **摘要:** The term "hallucination" converge different failure modes with specific geometric signatures in embedding space. We propose a taxonomy identifying three types: unfaithfulness (Type I: ignoring provided context), confabulation (Type II: inventing semantically foreign content), and factual error (Type III: wrong details within correct conceptual frames). We introduce two detection methods grounded in this taxonomy: the Semantic Grounding Index (SGI) for Type I, which measures whether a response moves toward provided context on the unit hypersphere, and the Directional Grounding Index (DGI) for Type II, which measures displacement geometry in context-free settings. DGI achieves AUROC=0.958 on human-crafted confabulations with 3.8% cross-domain degradation. External validation on three independently collected human-annotated benchmarks -WikiBio GPT-3, FELM, and ExpertQA- yields domain-specific AUROC 0.581-0.695, with DGI outperforming an NLI CrossEncoder baseline on expert-domain data, where surface entailment operates at chance. On LLM-generated benchmarks, detection is domain-local. We examine the Type III boundary through TruthfulQA, where apparent classifier signal (Logistic Regression with AUROC 0.731) is traced to a stylistic annotation confound: false answers are geometrically closer to queries than truthful ones, a pattern incompatible with factual-error detection. This identifies a theoretical constraint from a methodological limitation.
>
---
#### [replaced 052] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决传统大语言模型在动态场景中的应用局限。通过定义和分类流式大语言模型，提出系统化框架，并探讨其应用场景与未来方向。**

- **链接: [https://arxiv.org/pdf/2603.04592](https://arxiv.org/pdf/2603.04592)**

> **作者:** Junlong Tong; Zilong Wang; YuJie Ren; Peiran Yin; Hao Wu; Wei Zhang; Xiaoyu Shen
>
> **摘要:** Standard Large Language Models (LLMs) are predominantly designed for static inference with pre-defined inputs, which limits their applicability in dynamic, real-time scenarios. To address this gap, the streaming LLM paradigm has emerged. However, existing definitions of streaming LLMs remain fragmented, conflating streaming generation, streaming inputs, and interactive streaming architectures, while a systematic taxonomy is still lacking. This paper provides a comprehensive overview and analysis of streaming LLMs. First, we establish a unified definition of streaming LLMs based on data flow and dynamic interaction to clarify existing ambiguities. Building on this definition, we propose a systematic taxonomy of current streaming LLMs and conduct an in-depth discussion on their underlying methodologies. Furthermore, we explore the applications of streaming LLMs in real-world scenarios and outline promising research directions to support ongoing advances in streaming intelligence. We maintain a continuously updated repository of relevant papers at this https URL.
>
---
#### [replaced 053] Rewards as Labels: Revisiting RLVR from a Classification Perspective
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RLVR中梯度分配不均的问题。通过将奖励视为类别标签，将策略优化转化为分类问题，提出REAL框架提升训练稳定性和性能。**

- **链接: [https://arxiv.org/pdf/2602.05630](https://arxiv.org/pdf/2602.05630)**

> **作者:** Zepeng Zhai; Meilin Chen; Jiaxuan Zhao; Junlang Qian; Lei Shen; Yuan Lu
>
> **备注:** Withdrawal requested due to unauthorized inclusion of a co-author and incorrect institutional affiliation. The current version violates internal institutional policies and requires immediate retraction to resolve authorship and compliance issues
>
> **摘要:** Reinforcement Learning with Verifiable Rewards has recently advanced the capabilities of Large Language Models in complex reasoning tasks by providing explicit rule-based supervision. Among RLVR methods, GRPO and its variants have achieved strong empirical performance. Despite their success, we identify that they suffer from Gradient Misassignment in Positives and Gradient Domination in Negatives, which lead to inefficient and suboptimal policy updates. To address these issues, we propose Rewards as Labels (REAL), a novel framework that revisits verifiable rewards as categorical labels rather than scalar weights, thereby reformulating policy optimization as a classification problem. Building on this, we further introduce anchor logits to enhance policy learning. Our analysis reveals that REAL induces a monotonic and bounded gradient weighting, enabling balanced gradient allocation across rollouts and effectively mitigating the identified mismatches. Extensive experiments on mathematical reasoning benchmarks show that REAL improves training stability and consistently outperforms GRPO and strong variants such as DAPO. On the 1.5B model, REAL improves average Pass@1 over DAPO by 6.7%. These gains further scale to 7B model, REAL continues to outperform DAPO and GSPO by 6.2% and 1.7%, respectively. Notably, even with a vanilla binary cross-entropy, REAL remains stable and exceeds DAPO by 4.5% on average.
>
---
#### [replaced 054] Healthy LLMs? Benchmarking LLM Knowledge of UK Government Public Health Information
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型在英国公共卫生信息方面的知识。研究构建了PubHealthBench基准，测试24个模型的问答能力，发现SOTA模型在选择题中表现优异，但在自由回答中仍有提升空间。**

- **链接: [https://arxiv.org/pdf/2505.06046](https://arxiv.org/pdf/2505.06046)**

> **作者:** Joshua Harris; Fan Grayson; Felix Feldman; Timothy Laurence; Toby Nonnenmacher; Oliver Higgins; Leo Loman; Selina Patel; Thomas Finnie; Samuel Collins; Michael Borowitz
>
> **备注:** 27 pages, 9 pages main text
>
> **摘要:** As Large Language Models (LLMs) become widely accessible, a detailed understanding of their knowledge within specific domains becomes necessary for successful real world use. This is particularly critical in the domains of medicine and public health, where failure to retrieve relevant, accurate, and current information could significantly impact UK residents. However, while there are a number of LLM benchmarks in the medical domain, currently little is known about LLM knowledge within the field of public health. To address this issue, this paper introduces a new benchmark, PubHealthBench, with over 8000 questions for evaluating LLMs' Multiple Choice Question Answering (MCQA) and free form responses to public health queries. To create PubHealthBench we extract free text from 687 current UK government guidance documents and implement an automated pipeline for generating MCQA samples. Assessing 24 LLMs on PubHealthBench we find the latest proprietary LLMs (GPT-4.5, GPT-4.1 and o1) have a high degree of knowledge, achieving >90% accuracy in the MCQA setup, and outperform humans with cursory search engine use. However, in the free form setup we see lower performance with no model scoring >75%. Therefore, while there are promising signs that state of the art (SOTA) LLMs are an increasingly accurate source of public health information, additional safeguards or tools may still be needed when providing free form responses.
>
---
#### [replaced 055] FOR-Prompting: From Objection to Revision via an Asymmetric Prompting Protocol
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文提出FOR-Prompting，一种通过角色分工实现自我修正的推理协议，解决传统方法缺乏外部质疑机制的问题。适用于多种任务，提升模型推理效果。**

- **链接: [https://arxiv.org/pdf/2510.01674](https://arxiv.org/pdf/2510.01674)**

> **作者:** He Zhang; Anzhou Zhang; Jian Dai
>
> **摘要:** Reasoning protocols such as Chain of Thought (CoT) and Tree of Thought (ToT) organize internal deliberation but lack an explicit mechanism for external questioning that elicits self-revision. We present FOR-Prompting (From Objection to Revision Prompting), an asymmetric protocol where a Defender proposes an answer, an Debater (Questioner) raises question-style objections with no direct fixes, and a Host optionally synthesizes the final output. Across GSM8K, FOR-Prompting matches the accuracy of CoT and consistently improves over single-prompting when evaluated under identical model backbones. On small-scale open-source models (e.g., LLaMA-3.2-1B), FOR-Prompting yields substantial gains over direct prompting and performs comparably to lightweight reasoning baselines, highlighting its promise for low-resource and on-device settings. Cross-model role-swapping further shows that performance is primarily determined by the Defender, enabling small models to act effectively as Questioners. Beyond structured math tasks, FOR-Prompting supports refinement in open-ended and multi-stage tasks: qualitative analysis shows improved exploration, coverage, and specificity, and a blind study of human preferences found that participants preferred FOR-Prompting outputs over strong LLM baselines in an itinerary-planning scenario. The protocol is model-agnostic and operates purely through role-structured prompting, requiring no training, access to model internals, or symmetrically strong agents. FOR-Prompting therefore enables scalable study of objection-driven reasoning and offers a practical mechanism for automated iterative refinement across both hosted and local LLMs.
>
---
#### [replaced 056] Explainable Token-level Noise Filtering for LLM Fine-tuning Datasets
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，解决细调数据集与模型优化机制不匹配的问题。通过分解token属性，实现噪声过滤，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.14536](https://arxiv.org/pdf/2602.14536)**

> **作者:** Yuchen Yang; Wenze Lin; Enhao Huang; Zhixuan Chu; Hongbin Zhou; Lan Tao; Yiming Li; Zhan Qin; Kui Ren
>
> **摘要:** Large Language Models (LLMs) have seen remarkable advancements, achieving state-of-the-art results in diverse applications. Fine-tuning, an important step for adapting LLMs to specific downstream tasks, typically involves further training on corresponding datasets. However, a fundamental discrepancy exists between current fine-tuning datasets and the token-level optimization mechanism of LLMs: most datasets are designed at the sentence-level, which introduces token-level noise, causing negative influence to final performance. In this paper, we propose XTF, an explainable token-level noise filtering framework. XTF decomposes the complex and subtle contributions of token-level data to the fine-tuning process into three distinct and explicit attributes (reasoning importance, knowledge novelty, and task relevance), which can be assessed using scoring methods, and then masks the gradients of selected noisy tokens accordingly to optimize the performance of fine-tuned LLMs. We conduct extensive experiments on three representative downstream tasks (math, code and medicine) across 7 mainstream LLMs. The results demonstrate that XTF can significantly improve downstream performance by up to 13.7% compared to regular fine-tuning. Our work highlights the importance of token-level dataset optimization, and demonstrates the potential of strategies based on attribute decomposition for explaining complex training mechanisms.
>
---
#### [replaced 057] No Memorization, No Detection: Output Distribution-Based Contamination Detection in Small Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于数据污染检测任务，旨在解决小语言模型中数据污染的识别问题。通过分析输出分布，发现CDD方法效果有限，提出概率方法更有效。**

- **链接: [https://arxiv.org/pdf/2603.03203](https://arxiv.org/pdf/2603.03203)**

> **作者:** Omer Sela
>
> **备注:** Code available at this https URL
>
> **摘要:** CDD, or Contamination Detection via output Distribution, identifies data contamination by measuring the peakedness of a model's sampled outputs. We study the conditions under which this approach succeeds and fails on small language models ranging from 70M to 410M parameters. Using controlled contamination experiments on GSM8K, HumanEval, and MATH, we find that CDD's effectiveness depends critically on whether fine-tuning produces verbatim memorization. In the majority of conditions we test, CDD performs at chance level even when the data is verifiably contaminated and detectable by simpler methods. We show that probability-based methods, specifically perplexity and Min-k% Prob, outperform CDD in every condition we test, suggesting that output-distribution approaches are insufficient for contamination detection in small language models. Our code is available at this https URL
>
---
#### [replaced 058] Parallel Decoder Transformer: Planner-Seeded Latent Coordination for Synchronized Parallel Decoding
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Parallel Decoder Transformer，解决并行解码中的同步与协调问题，通过模型内机制实现多流生成的协同。**

- **链接: [https://arxiv.org/pdf/2512.10054](https://arxiv.org/pdf/2512.10054)**

> **作者:** Logan Robbins
>
> **备注:** Note: Updated to reflect revised architecture
>
> **摘要:** Autoregressive language models can often identify parallel subproblems, but standard decoding exposes only a single left-to-right output interface. External orchestration methods can launch multiple prompts concurrently, yet they provide no model-internal state through which those generations can synchronize, resolve ownership, or wait for missing information. We present the Parallel Decoder Transformer (PDT), a frozen-trunk architecture that augments a decoder with a planner-seeded latent workspace and a synchronized multi-stream output protocol. Before any stream emits tokens, a mandatory prompt-time planner predicts fixed latent plan slots and projects them as snapshot 0 on an embeddings-only Dynamic Notes Bus. During decoding, each stream reads the visible notes window through Speculative Note Conditioning (SNC), emits provisional token blocks and latent summaries, and advances only when agreement logic determines that the current shared state is sufficient for continued parallel generation. Coverage heads track plan-item ownership, while rollback handles incoherent or premature commits. PDT therefore shifts parallel task decomposition from an external prompting strategy to a model-internal coordination mechanism over the output interface of a frozen language model.
>
---
#### [replaced 059] Mem-T: Densifying Rewards for Long-Horizon Memory Agents
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决长周期记忆管理中奖励稀疏的问题。提出Mem-T与MoT-GRPO，通过密集奖励提升记忆代理的性能与效率。**

- **链接: [https://arxiv.org/pdf/2601.23014](https://arxiv.org/pdf/2601.23014)**

> **作者:** Yanwei Yue; Boci Peng; Xuanbo Fan; Jiaxin Guo; Qiankun Li; Yan Zhang
>
> **摘要:** Memory agents, which depart from predefined memory-processing pipelines by endogenously managing the processing, storage, and retrieval of memories, have garnered increasing attention for their autonomy and adaptability. However, existing training paradigms remain constrained: agents often traverse long-horizon sequences of memory operations before receiving sparse and delayed rewards, which hinders truly end-to-end optimization of memory management policies. To address this limitation, we introduce Mem-T, an autonomous memory agent that interfaces with a lightweight hierarchical memory database to perform dynamic updates and multi-turn retrieval over streaming inputs. To effectively train long-horizon memory management capabilities, we further propose MoT-GRPO, a tree-guided reinforcement learning framework that transforms sparse terminal feedback into dense, step-wise supervision via memory operation tree backpropagation and hindsight credit assignment, thereby enabling the joint optimization of memory construction and retrieval. Extensive experiments demonstrate that Mem-T is (1) high-performing, surpassing frameworks such as A-Mem and Mem0 by up to $14.92\%$, and (2) economical, operating on a favorable accuracy-efficiency Pareto frontier and reducing inference tokens per query by $\sim24.45\%$ relative to GAM without sacrificing performance.
>
---
#### [replaced 060] MathSmith: Towards Extremely Hard Mathematical Reasoning by Forging Synthetic Problems with a Reinforced Policy
- **分类: cs.CL**

- **简介: 该论文属于数学推理任务，旨在解决高质量难问题数据稀缺的问题。提出MathSmith框架，通过合成高难度数学题提升大模型推理能力。**

- **链接: [https://arxiv.org/pdf/2508.05592](https://arxiv.org/pdf/2508.05592)**

> **作者:** Shaoxiong Zhan; Yanlin Lai; Ziyu Lu; Dahua Lin; Ziqing Yang; Fei Tan
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Large language models have achieved substantial progress in mathematical reasoning, yet their advancement is limited by the scarcity of high-quality, high-difficulty training data. Existing synthesis methods largely rely on transforming human-written templates, limiting both diversity and scalability. We propose MathSmith, a novel framework for synthesizing challenging mathematical problems to enhance LLM reasoning. Rather than modifying existing problems, MathSmith constructs new ones from scratch by randomly sampling concept-explanation pairs from PlanetMath, ensuring data independence and avoiding contamination. To increase difficulty, we design nine predefined strategies as soft constraints during rationales. We further adopts reinforcement learning to jointly optimize structural validity, reasoning complexity, and answer consistency. The length of the reasoning trace generated under autoregressive prompting is used to reflect cognitive complexity, encouraging the creation of more demanding problems aligned with long-chain-of-thought reasoning. Experiments across five benchmarks, categorized as easy & medium (GSM8K, MATH-500) and hard (AIME2024, AIME2025, OlympiadBench), show that MathSmith consistently outperforms existing baselines under both short and long CoT settings. Additionally, a weakness-focused variant generation module enables targeted improvement on specific concepts. Overall, MathSmith exhibits strong scalability, generalization, and transferability, highlighting the promise of high-difficulty synthetic data in advancing LLM reasoning capabilities. Our code and data are available at this https URL.
>
---
#### [replaced 061] Mitigating Unintended Memorization with LoRA in Federated Learning for LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于联邦学习任务，旨在解决大模型在训练中意外记忆敏感数据的问题。通过引入LoRA技术，有效降低记忆风险，提升隐私保护。**

- **链接: [https://arxiv.org/pdf/2502.05087](https://arxiv.org/pdf/2502.05087)**

> **作者:** Thierry Bossy; Julien Vignoud; Tahseen Rabbani; Juan R. Troncoso Pastoriza; Martin Jaggi
>
> **摘要:** Federated learning (FL) is a popular paradigm for collaborative training which avoids direct data exposure between clients. However, data privacy issues still remain: FL-trained large language models are capable of memorizing and completing phrases and sentences contained in training data when given their prefixes. Thus, it is possible for adversarial and honest- but-curious clients to recover training data of other participants simply through targeted prompting. In this work, we demonstrate that a popular and simple fine-tuning strategy, low-rank adaptation (LoRA), reduces memorization during FL by a factor of up to 10 without significant performance cost. We study this effect by performing fine-tuning tasks in high-risk domains such as medicine, law, and finance. We observe a reduction in memorization for a wide variety of model families, from 1B to 70B parameters. We find that LoRA can reduce memorization in centralized learning as well, and we compare how the memorization patterns differ. Furthermore, we study the effect of hyperparameters and show that LoRA can be combined with other privacy-preserving techniques such as gradient clipping and Gaussian noise, secure aggregation, and Goldfish loss to further improve record-level privacy while maintaining performance.
>
---
#### [replaced 062] Stealth Fine-Tuning: Efficiently Breaking Alignment in RVLMs Using Self-Generated CoT
- **分类: cs.CL**

- **简介: 该论文属于安全攻击任务，旨在破解RVLMs的安全对齐机制。通过Stealth Fine-Tuning方法，利用自生成推理痕迹进行高效微调，实现对模型的低代价攻击。**

- **链接: [https://arxiv.org/pdf/2511.14106](https://arxiv.org/pdf/2511.14106)**

> **作者:** Le Yu; Zhengyue Zhao; Yawen Zheng; Yunhao Liu
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Reasoning-augmented Vision-Language Models (RVLMs) rely on safety alignment to prevent harmful behavior, yet their exposed chain-of-thought (CoT) traces introduce new attack surfaces. In this work, we find that the safety alignment of RVLMs can be easily broken through a novel attack method termed \textbf{Stealth Fine-Tuning}. Our method elicits harmful reasoning traces through \textbf{segment-level interference} and reuses the self-generated outputs as supervised fine-tuning data. To facilitate this, we introduce a \textbf{turn-based weighted} loss that minimizes distribution shift. In our experiment, with only 499 samples and under 3 hours on a single A100 (QLoRA), Stealth Fine-Tuning outperforms IDEATOR by 38.66\% ASR while preserving general reasoning ability, as the tuned model retains the original representation distribution. Experiments on AdvBench and several general benchmarks demonstrate that Stealth Fine-Tuning is a low-cost and highly effective way to bypass alignment defenses. \textcolor{red}{\textbf{Disclaimer: This paper contains content that may be disturbing or offensive.}}
>
---
#### [replaced 063] Causal Retrieval with Semantic Consideration
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决传统IR系统忽视因果关系的问题。提出CAWAI模型，同时学习语义和因果关系，提升复杂查询的检索效果。**

- **链接: [https://arxiv.org/pdf/2504.04700](https://arxiv.org/pdf/2504.04700)**

> **作者:** Hyunseo Shin; Wonseok Hwang
>
> **摘要:** Recent advancements in large language models (LLMs) have significantly enhanced the performance of conversational AI systems. To extend their capabilities to knowledge-intensive domains such as biomedical and legal fields, where the accuracy is critical, LLMs are often combined with information retrieval (IR) systems to generate responses based on retrieved documents. However, for IR systems to effectively support such applications, they must go beyond simple semantic matching and accurately capture diverse query intents, including causal relationships. Existing IR models primarily focus on retrieving documents based on surface-level semantic similarity, overlooking deeper relational structures such as causality. To address this, we propose CAWAI, a retrieval model that is trained with dual objectives: semantic and causal relations. Our extensive experiments demonstrate that CAWAI outperforms various models on diverse causal retrieval tasks especially under large-scale retrieval settings. We also show that CAWAI exhibits strong zero-shot generalization across scientific domain QA tasks.
>
---
#### [replaced 064] Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness for Tool-Using LLM Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于金融AI可靠性研究，解决LLM代理在审计复现中的不确定性问题。提出DFAH框架，评估代理的确定性和忠实性，发现准确性和确定性无显著关联。**

- **链接: [https://arxiv.org/pdf/2601.15322](https://arxiv.org/pdf/2601.15322)**

> **作者:** Raffi Khatchadourian
>
> **备注:** 27 pages, 5 figures, 9 tables | Code and data: this https URL | To appear in the 2nd ICLR Workshop on Advances in Financial AI: Towards Agentic and Responsible Systems (ICLR 2026)
>
> **摘要:** LLM agents struggle with regulatory audit replay: when asked to reproduce a flagged transaction decision with identical inputs, many deployments fail to return consistent results. We introduce the Determinism-Faithfulness Assurance Harness (DFAH), a framework for measuring trajectory determinism, decision determinism, and evidence-conditioned faithfulness in tool-using agents deployed in financial services. Across 4,700+ agentic runs (7 models, 4 providers, 3 financial benchmarks with 50 cases each at T=0.0), we find that decision determinism and task accuracy are not detectably correlated (r = -0.11, 95% CI [-0.49, 0.31], p = 0.63, n = 21 configurations): models can be deterministic without being accurate, and accurate without being deterministic. Because neither metric predicts the other in our sample, both must be measured independently, which is precisely what DFAH provides. Small models (7-20B) achieve near-perfect determinism through rigid pattern matching at the cost of accuracy (20-42%), while frontier models show moderate determinism (50-96%) with variable accuracy. No model achieves both perfect determinism and high accuracy, supporting DFAH's multi-dimensional measurement approach. We provide three financial benchmarks (compliance triage, portfolio constraints, and DataOps exceptions; 50 cases each) together with an open-source stress-test harness. Across these benchmarks and DFAH evaluation settings, Tier 1 models with schema-first architectures achieved determinism levels consistent with audit replay requirements.
>
---
#### [replaced 065] Do Schwartz Higher-Order Values Help Sentence-Level Human Value Detection? A Study of Hierarchical Gating and Calibration
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究在有限计算资源下，Schwartz高阶价值类别对句子级人类价值检测的帮助。任务是多标签分类，解决数据稀疏和不平衡问题。工作包括对比多种模型方法，验证高阶类别作为归纳偏置的有效性。**

- **链接: [https://arxiv.org/pdf/2602.00913](https://arxiv.org/pdf/2602.00913)**

> **作者:** Víctor Yeste; Paolo Rosso
>
> **备注:** Code: this https URL, models: this https URL, 27 pages, 4 figures
>
> **摘要:** Human value detection from single sentences is a sparse, imbalanced multi-label task. We study whether Schwartz higher-order (HO) categories help this setting on ValueEval'24 / ValuesML (74K English sentences) under a compute-frugal budget. Rather than proposing a new architecture, we compare direct supervised transformers, hard HO$\rightarrow$values pipelines, Presence$\rightarrow$HO$\rightarrow$values cascades, compact instruction-tuned large language models (LLMs), QLoRA, and low-cost upgrades such as threshold tuning and small ensembles. HO categories are learnable: the easiest bipolar pair, Growth vs. Self-Protection, reaches Macro-$F_1=0.58$. The most reliable gains come from calibration and ensembling: threshold tuning improves Social Focus vs. Personal Focus from $0.41$ to $0.57$ ($+0.16$), transformer soft voting lifts Growth from $0.286$ to $0.303$, and a Transformer+LLM hybrid reaches $0.353$ on Self-Protection. In contrast, hard hierarchical gating does not consistently improve the end task. Compact LLMs also underperform supervised encoders as stand-alone systems, although they sometimes add useful diversity in hybrid ensembles. Under this benchmark, the HO structure is more useful as an inductive bias than as a rigid routing rule.
>
---
#### [replaced 066] A Component-Based Survey of Interactions between Large Language Models and Multi-Armed Bandits
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于交叉领域研究，探讨大语言模型与多臂老虎机的双向互动，解决两者融合中的挑战与优化问题，分析现有系统并提出未来方向。**

- **链接: [https://arxiv.org/pdf/2601.12945](https://arxiv.org/pdf/2601.12945)**

> **作者:** Siguang Chen; Chunli Lv; Miao Xie
>
> **备注:** 25 pages, 6 table
>
> **摘要:** Large language models (LLMs) have become powerful and widely used systems for language understanding and generation, while multi-armed bandit (MAB) algorithms provide a principled framework for adaptive decision-making under uncertainty. This survey explores the potential at the intersection of these two fields. As we know, it is the first survey to systematically review the bidirectional interaction between large language models and multi-armed bandits at the component level. We highlight the bidirectional benefits: MAB algorithms address critical LLM challenges, spanning from pre-training to retrieval-augmented generation (RAG) and personalization. Conversely, LLMs enhance MAB systems by redefining core components such as arm definition and environment modeling, thereby improving decision-making in sequential tasks. We analyze existing LLM-enhanced bandit systems and bandit-enhanced LLM systems, providing insights into their design, methodologies, and performance. Key challenges and representative findings are identified to help guide future research. An accompanying GitHub repository that indexes relevant literature is available at this https URL.
>
---
#### [replaced 067] PonderLM-2: Pretraining LLM with Latent Thoughts in Continuous Space
- **分类: cs.CL**

- **简介: 该论文提出PonderLM-2，通过在预训练中引入隐含思维步骤，提升语言模型生成质量。任务是改进语言模型生成性能，解决如何通过增加计算步骤提高单个token预测效果的问题。**

- **链接: [https://arxiv.org/pdf/2509.23184](https://arxiv.org/pdf/2509.23184)**

> **作者:** Boyi Zeng; He Li; Shixiang Song; Yixuan Wang; Zitong Wang; Ziwei He; Xinbing Wang; Zhouhan Lin
>
> **摘要:** The remarkable success of Chain-of-Thought (CoT), which enhances performance by scaling generation steps at test-time, inspires us to ask: can we leverage a similar scaling of computational steps during pretraining to improve the generation of each individual token? To address this, we propose a novel pre-training methodology: Pretraining Language Models with Latent Thoughts (PonderLM-2). Our approach pretrains a language model (LM) to first generate an intermediate latent thought-the last hidden state of the current position-which is then used as input to predict the actual subsequent token. This additional computational step enables the LM to refine its prediction within unconstrained continuous space. Our experiments demonstrate that, at an identical inference cost, a LM that generates one additional latent thought per token outperforms a standard model with double the parameters. For instance, our PonderLM-2-Pythia-1.4B, pretrained on 300B tokens from the Pile, significantly surpasses the vanilla Pythia-2.8B trained on the same data on both language modeling and a range of general downstream tasks. Furthermore, increasing the number of latent thoughts generated before each actual token-forming a chain analogous to CoT-consistently improves the model's performance. The code is available at this https URL.
>
---
#### [replaced 068] CompanionCast: Toward Social Collaboration with Multi-Agent Systems in Shared Experiences
- **分类: cs.HC; cs.CL**

- **简介: 该论文提出CompanionCast框架，解决共享体验中社交互动不足的问题。通过多智能体系统增强实时社交参与，提升用户的情感共鸣与共在感。**

- **链接: [https://arxiv.org/pdf/2512.10918](https://arxiv.org/pdf/2512.10918)**

> **作者:** Yiyang Wang; Chen Chen; Tica Lin; Vishnu Raj; Josh Kimball; Alex Cabral; Josiah Hester
>
> **备注:** Accepted at ACM CHI 2026 Workshop on Human-Agent Collaboration
>
> **摘要:** Shared experiences are fundamental to social connection, yet media consumption is increasingly solitary. While AI companions offer real-time reactions and emotional regulation, existing systems either rely on single-agent designs or lack the social awareness and multi-party interaction required to replicate authentic group dynamics. We present CompanionCast, a general framework for orchestrating multiple specialized AI agents as social collaborators within a live shared context. CompanionCast integrates multimodal event detection, rolling context caching for improved grounding, and spatial audio to enhance co-presence. We validate CompanionCast through sports viewing, a domain with rich dynamics and strong social traditions. Pilot studies with soccer fans demonstrate that CompanionCast significantly improves perceived social presence and emotional sharing compared to solitary viewing. We conclude by discussing implications and open challenges for multi-agent systems as social collaborators in shared experiences.
>
---
#### [replaced 069] NC-Bench: An LLM Benchmark for Evaluating Conversational Competence
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出NC-Bench，用于评估大语言模型的对话能力。解决对话结构评估问题，通过三个测试集检验模型在不同对话场景中的表现。**

- **链接: [https://arxiv.org/pdf/2601.06426](https://arxiv.org/pdf/2601.06426)**

> **作者:** Robert J. Moore; Sungeun An; Farhan Ahmed; Jay Pankaj Gala
>
> **备注:** 8 pages, 1 figure, 2 tables
>
> **摘要:** The Natural Conversation Benchmark (NC-Bench) introduces a new approach to evaluating the general conversational competence of large language models (LLMs). Unlike prior benchmarks that focus on the content of model behavior, NC-Bench focuses on the form and structure of natural conversation. Grounded in the IBM Natural Conversation Framework (NCF), NC-Bench comprises three distinct sets: (1) the basic set evaluates fundamental sequence management practices, such as answering inquiries, repairing responses, and closing conversational pairs; (2) the retrieval-augmented generation (RAG) set applies the same sequence management patterns as the first set but incorporates information-seeking via RAG; (3) the complex request set extends to requests involving more intricate sequence management patterns. Each set tests a model's ability to produce contextually appropriate conversational actions in response to characteristic interaction patterns. Initial evaluations across six open-source models and 14 interaction patterns show that models perform well on basic answering tasks, struggle more with repair tasks (especially repeat), have mixed performance on closing sequences, and find complex multi-turn requests most challenging. By operationalizing fundamental principles of human conversation, NC-Bench provides a lightweight, extensible, and theory-grounded framework for assessing and improving the conversational abilities of LLMs beyond topical or task-specific benchmarks.
>
---
#### [replaced 070] Measuring Complexity at the Requirements Stage: Spectral Metrics as Development Effort Predictors
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于需求工程任务，旨在解决需求复杂性量化不足的问题。通过自然语言处理提取结构网络，利用谱度量预测开发工作量，验证其有效性。**

- **链接: [https://arxiv.org/pdf/2602.07182](https://arxiv.org/pdf/2602.07182)**

> **作者:** Maximilian Vierlboeck; Antonio Pugliese; Roshanak Nilchian; Paul Grogan; Rashika Sugganahalli Natesh Babu
>
> **备注:** 18 pages, 4 figures, 5 tables
>
> **摘要:** Complexity in engineered systems presents one of the most persistent challenges in modern development since it is driving cost overruns, schedule delays, and outright project failures. Yet while architectural complexity has been studied, the structural complexity embedded within requirements specifications remains poorly understood and inadequately quantified. This gap is consequential: requirements fundamentally drive system design, and complexity introduced at this stage propagates through architecture, implementation, and integration. To address this gap, we build on Natural Language Processing methods that extract structural networks from textual requirements. Using these extracted structures, we conducted a controlled experiment employing molecular integration tasks as structurally isomorphic proxies for requirements integration - leveraging the topological equivalence between molecular graphs and requirement networks while eliminating confounding factors such as domain expertise and semantic ambiguity. Our results demonstrate that spectral measures predict integration effort with correlations exceeding 0.95, while structural metrics achieve correlations above 0.89. Notably, density-based metrics show no significant predictive validity. These findings indicate that eigenvalue-derived measures capture cognitive and effort dimensions that simpler connectivity metrics cannot. As a result, this research bridges a critical methodological gap between architectural complexity analysis and requirements engineering practice, providing a validated foundation for applying these metrics to requirements engineering, where similar structural complexity patterns may predict integration effort.
>
---
#### [replaced 071] MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出MAS-Orchestra框架，解决多智能体系统设计效率低和效果不确定的问题，通过全局推理和控制基准评估，提升多智能体协作性能。**

- **链接: [https://arxiv.org/pdf/2601.14652](https://arxiv.org/pdf/2601.14652)**

> **作者:** Zixuan Ke; Yifei Ming; Austin Xu; Ryan Chin; Xuan-Phi Nguyen; Prathyusha Jwalapuram; Jiayu Wang; Semih Yavuz; Caiming Xiong; Shafiq Joty
>
> **备注:** Preprint; Work in Progress
>
> **摘要:** While multi-agent systems (MAS) promise elevated intelligence through coordination of agents, current approaches to automatic MAS design under-deliver. Such shortcomings stem from two key factors: (1) methodological complexity - agent orchestration is performed using sequential, code-level execution that limits global system-level holistic reasoning and scales poorly with agent complexity - and (2) efficacy uncertainty - MAS are deployed without understanding if there are tangible benefits compared to single-agent systems (SAS). We propose MASOrchestra, a training-time framework that formulates MAS orchestration as a function-calling reinforcement learning problem with holistic orchestration, generating an entire MAS at once. In MAS-Orchestra, complex, goal-oriented subagents are abstracted as callable functions, enabling global reasoning over system structure while hiding internal execution details. To rigorously study when and why MAS are beneficial, we introduce MASBENCH, a controlled benchmark that characterizes tasks along five axes: Depth, Horizon, Breadth, Parallel, and Robustness. Our analysis reveals that MAS gains depend critically on task structure, verification protocols, and the capabilities of both orchestrator and subagents, rather than holding universally. Guided by these insights, MAS-Orchestra achieves consistent improvements on public benchmarks including mathematical reasoning, multi-hop QA, and search-based QA, while achieving more than 10x efficiency over strong baselines. Together, MAS-Orchestra and MASBENCH enable better training and understanding of MAS in the pursuit of multi-agent intelligence.
>
---
#### [replaced 072] Adaptation of Agentic AI: A Survey of Post-Training, Memory, and Skills
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于人工智能领域，研究如何通过后训练、记忆和技能提升智能体性能。解决智能体与工具协同适应的问题，提出四范式框架，综述相关方法并分析其优劣。**

- **链接: [https://arxiv.org/pdf/2512.16301](https://arxiv.org/pdf/2512.16301)**

> **作者:** Pengcheng Jiang; Jiacheng Lin; Zhiyi Shi; Zifeng Wang; Luxi He; Yichen Wu; Ming Zhong; Peiyang Song; Qizheng Zhang; Heng Wang; Xueqiang Xu; Hanwen Xu; Pengrui Han; Dylan Zhang; Jiashuo Sun; Chaoqi Yang; Kun Qian; Tian Wang; Changran Hu; Manling Li; Quanzheng Li; Hao Peng; Sheng Wang; Jingbo Shang; Chao Zhang; Jiaxuan You; Liyuan Liu; Pan Lu; Yu Zhang; Heng Ji; Yejin Choi; Dawn Song; Jimeng Sun; Jiawei Han
>
> **摘要:** Large language model (LLM) agents are moving beyond prompting alone. ChatGPT marked the rise of general-purpose LLM assistants, DeepSeek showed that on-policy reinforcement learning with verifiable rewards can improve reasoning and tool use, and OpenClaw highlights a newer direction in which agents accumulate persistent memory and reusable skills. Yet the research landscape remains fragmented across post-training, retrieval, memory, and skill systems. This survey studies these developments under a single notion of \emph{adaptation}: improving an agent, its tools, or their interaction after pretraining. We organize the field with a four-paradigm framework spanning agent adaptation and tool adaptation. On the agent side, A1 (tool-execution-signaled) and A2 (agent-output-signaled) improve the agent itself through supervised fine-tuning, preference optimization, and reinforcement learning with verifiable rewards. On the tool side, T1 (agent-agnostic) provides reusable pre-trained modules any agent can call, while T2 (agent-supervised) uses the agent's outputs to train memory systems, skill libraries, or lightweight subagents. Using this framework, we review post-training methods, adaptive memory architectures, and agent skills; compare their trade-offs in cost, flexibility, and generalization; and summarize evaluation practices across deep research, software development, computer use, and drug discovery. We conclude by outlining open problems in agent-tool co-adaptation, continual learning, safety, and efficient deployment.
>
---
#### [replaced 073] Exploring Embedding Priors in Prompt-Tuning for Improved Interpretability and Control
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究Prompt-Tuning中的嵌入崩溃现象，探讨先验对模型性能的影响，旨在提升模型的可解释性与控制能力。**

- **链接: [https://arxiv.org/pdf/2412.18582](https://arxiv.org/pdf/2412.18582)**

> **作者:** Sergey Sedov; Sumanth Bharadwaj Hachalli Karanam; Venu Gopal Kadamba
>
> **摘要:** Prompt-Tuning is an efficient method for adapting pre-trained language models to new tasks with minimal computational overhead by modifying prompt embeddings. In this work, we investigate how crucial the phenomenon of embedding collapse, frequently observed in Prompt-Tuning, is for the final performance of the model. To address this question, we designed embedding priors and compared them with posteriors of the converged Soft and Deep Prompt-Tuning methods. Our findings suggest that priors strongly affect the position of the tuned embeddings, and models can effectively work with embeddings from different parts of activation spaces, including completely new regions. As the final Prompt-Tuning capabilities are limited, we hypothesize that controllable Prompt-Tuning posteriors may serve as a good starting point for tasks such as chain-of-thought (COT) distillation. Our experiments also show that generated trajectories are not localized in the activation space of the models. However, there are distinct clusters of activations for distant tasks (e.g., NLP and arithmetic), while activations between NLP tasks (e.g., Question-Answering and MLM) lie in the same cluster. These observations raise questions about the importance of a single activation cluster for the generalization abilities of large language models.
>
---
#### [replaced 074] KrishokBondhu: A Retrieval-Augmented Voice-Based Agricultural Advisory Call Center for Bengali Farmers
- **分类: cs.CL; cs.HC; cs.IR**

- **简介: 该论文属于农业咨询任务，旨在解决Bangladeshi农民获取及时专家建议的问题。构建了基于RAG的语音咨询系统KrishokBondhu，提供实时 Bengali 农业指导。**

- **链接: [https://arxiv.org/pdf/2510.18355](https://arxiv.org/pdf/2510.18355)**

> **作者:** Mohd Ruhul Ameen; Akif Islam; Farjana Aktar; M. Saifuzzaman Rafat
>
> **备注:** Accepted at the 2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence and Networking (QPAIN 2026)
>
> **摘要:** In Bangladesh, many farmers still struggle to access timely, expert-level agricultural guidance. This paper presents KrishokBondhu, a voice-enabled, call-centre-integrated advisory platform built on a Retrieval-Augmented Generation (RAG) framework for Bengali-speaking farmers. The system combines agricultural handbooks, extension manuals, and NGO publications, processes them through an OCR-based pipeline, and indexes the curated content in a vector database for semantic retrieval. Through a phone-based interface, farmers can receive real-time, context-aware advice: speech-to-text converts the Bengali query, the RAG module retrieves relevant information, a large language model (Gemma 3-4B) generates a grounded response, and text-to-speech delivers the answer in spoken Bengali. In a pilot evaluation, KrishokBondhu produced high-quality responses for 72.7% of diverse agricultural queries. Compared to the KisanQRS benchmark, it achieved a composite score of 4.53 versus 3.13 on a 5-point scale, with a 44.7% improvement and especially large gains in contextual richness and completeness, while maintaining comparable relevance and technical specificity. Semantic-similarity analysis further showed a strong correlation between retrieved context and answer quality. KrishokBondhu demonstrates the feasibility of combining call-centre accessibility, multilingual voice interaction, and modern RAG techniques to deliver expert-level agricultural guidance to remote Bangladeshi farmers.
>
---
#### [replaced 075] R-WoM: Retrieval-augmented World Model For Computer-use Agents
- **分类: cs.CL**

- **简介: 该论文属于强化学习中的世界模型任务，旨在解决LLMs在长序列预测中因幻觉和静态知识导致的误差问题。提出R-WoM模型，通过检索外部知识提升模拟准确性。**

- **链接: [https://arxiv.org/pdf/2510.11892](https://arxiv.org/pdf/2510.11892)**

> **作者:** Kai Mei; Jiang Guo; Shuaichen Chang; Mingwen Dong; Dongkyu Lee; Xing Niu; Jiarong Jiang
>
> **摘要:** Large Language Models (LLMs) can serve as world models to enhance agent decision-making in digital environments by simulating future states and predicting action outcomes, potentially eliminating costly trial-and-error exploration. However, this capability is fundamentally limited by LLMs' tendency toward hallucination and their reliance on static training knowledge, which can lead to compounding errors that inhibit long-horizon simulations. To systematically investigate whether LLMs are appropriate for world modeling, we probe two core capabilities of world models--future state prediction and reward estimation--through three tasks: next-state identification, full-procedure planning alignment, and milestone transition recognition. Our analysis shows that while LLMs effectively capture immediate next states and identify meaningful state transitions, their performance rapidly degrades in full-procedure planning. This highlights LLMs' limitations in reliably modeling environment dynamics over long horizons. To address these limitations, we propose the Retrieval-augmented World Model (R-WoM), which grounds LLM simulations by incorporating factual, up-to-date knowledge retrieved from external tutorials. Experiments show that R-WoM achieves relative improvements of up to 23.4% and 16.3% on the subsets of OSWorld and Webarena compared to baselines, with particular advantage in longer-horizon simulations.
>
---
#### [replaced 076] Learning Page Order in Shuffled WOO Releases
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究文档页面排序任务，解决在语义信号不可靠的杂乱文档中恢复正确顺序的问题。通过比较多种方法，发现seq2seq模型在长文档中表现差，提出模型专业化提升效果。**

- **链接: [https://arxiv.org/pdf/2602.11040](https://arxiv.org/pdf/2602.11040)**

> **作者:** Efe Kahraman; Giulio Tosato
>
> **摘要:** We investigate document page ordering on 5,461 shuffled WOO documents (Dutch freedom of information releases) using page embeddings. These documents are heterogeneous collections such as emails, legal texts, and spreadsheets compiled into single PDFs, where semantic ordering signals are unreliable. We compare five methods, including pointer networks, seq2seq transformers, and specialized pairwise ranking models. The best performing approach successfully reorders documents up to 15 pages, with Kendall's tau ranging from 0.95 for short documents (2-5 pages) to 0.72 for 15 page documents. We observe two unexpected failures: seq2seq transformers fail to generalize on long documents (Kendall's tau drops from 0.918 on 2-5 pages to 0.014 on 21-25 pages), and curriculum learning underperforms direct training by 39% on long documents. Ablation studies suggest learned positional encodings are one contributing factor to seq2seq failure, though the degradation persists across all encoding variants, indicating multiple interacting causes. Attention pattern analysis reveals that short and long documents require fundamentally different ordering strategies, explaining why curriculum learning fails. Model specialization achieves substantial improvements on longer documents (+0.21 tau).
>
---
#### [replaced 077] Process-Centric Analysis of Agentic Software Systems
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于软件系统分析任务，旨在解决 agentic 系统执行过程的评估与优化问题。通过引入 Graphectory 分析工具，研究系统轨迹特征并提升执行效率。**

- **链接: [https://arxiv.org/pdf/2512.02393](https://arxiv.org/pdf/2512.02393)**

> **作者:** Shuyang Liu; Yang Chen; Rahul Krishna; Saurabh Sinha; Jatin Ganhotra; Reyhan Jabbarvand
>
> **摘要:** Agentic systems are modern software systems: they consist of orchestrated modules, expose interfaces, and are deployed in software pipelines. Unlike conventional programs, their execution, i.e., trajectories, is inherently stochastic and adaptive to the problems they solve. Evaluation of such systems is often outcome-centric. This narrow focus overlooks detailed insights, failing to explain how agents reason, plan, act, or change their strategies. Inspired by the structured representation of conventional software systems as graphs, we introduce Graphectory to systematically encode the temporal and semantic relations in such systems. Using Graphectory, we automatically analyze 4000 trajectories of two dominant agentic programming workflows, SWE-agent and OpenHands, with four backbone Large Language Models (LLMs), attempting to resolve SWE-bench issues. Our automated analyses (completed within four minutes) reveal that: (1) agents using richer prompts or stronger LLMs exhibit more complex Graphectory, reflecting deeper exploration, broader context gathering, and more thorough validation; (2) agents' strategies vary with problem difficulty and the underlying LLM - for resolved issues, strategies often follow coherent localization-patching-validation steps, while unresolved ones exhibit chaotic or backtracking behaviors; and (3) even successful agentic systems often display inefficient processes. We also implement a novel technique for real-time construction and analysis of Graphectory and Langutory during agent execution to flag trajectory issues. Upon detecting such issues, the technique notifies the agent with a diagnostic message and, when applicable, rolls back the trajectory. Experiments show that online monitoring and interventions improve resolution rates by 6.9%-23.5% across models for problematic instances, while significantly shortening trajectories with near-zero overhead.
>
---
#### [replaced 078] IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding
- **分类: cs.CV; cs.CL; cs.CR**

- **简介: 该论文属于视觉定位任务，针对VLM系统的安全性问题，提出IAG方法实现多目标后门攻击，通过动态生成文本引导的触发器，在不影响正常性能的情况下完成攻击。**

- **链接: [https://arxiv.org/pdf/2508.09456](https://arxiv.org/pdf/2508.09456)**

> **作者:** Junxian Li; Beining Xu; Simin Chen; Jiatong Li; Jingdi Lei; Haodong Zhao; Di Zhang
>
> **备注:** 20 pages, 13 Figures
>
> **摘要:** Recent advances in vision-language models (VLMs) have significantly enhanced the visual grounding task, which involves locating objects in an image based on natural language queries. Despite these advancements, the security of VLM-based grounding systems has not been thoroughly investigated. This paper reveals a novel and realistic vulnerability: the first multi-target backdoor attack on VLM-based visual grounding. Unlike prior attacks that rely on static triggers or fixed targets, we propose IAG, a method that dynamically generates input-aware, text-guided triggers conditioned on any specified target object description to execute the attack. This is achieved through a text-conditioned UNet that embeds imperceptible target semantic cues into visual inputs while preserving normal grounding performance on benign samples. We further develop a joint training objective that balances language capability with perceptual reconstruction to ensure imperceptibility, effectiveness, and stealth. Extensive experiments on multiple VLMs (e.g., LLaVA, InternVL, Ferret) and benchmarks (RefCOCO, RefCOCO+, RefCOCOg, Flickr30k Entities, and ShowUI) demonstrate that IAG achieves the best ASRs compared with other baselines on almost all settings without compromising clean accuracy, maintaining robustness against existing defenses, and exhibiting transferability across datasets and models. These findings underscore critical security risks in grounding-capable VLMs and highlight the need for further research on trustworthy multimodal understanding.
>
---
#### [replaced 079] Linear probes rely on textual evidence: Results from leakage mitigation studies in language models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究白盒监控在语言模型中的有效性，探讨文本证据对探测性能的影响。任务是评估线性探测器在去除文本证据后的表现，解决其在模糊行为检测中的可靠性问题。通过实验发现过滤文本证据显著降低探测效果。**

- **链接: [https://arxiv.org/pdf/2509.21344](https://arxiv.org/pdf/2509.21344)**

> **作者:** Gerard Boxo; Aman Neelappa; Shivam Raval
>
> **备注:** 33 pages, 22 figures
>
> **摘要:** White-box monitors are a popular technique for detecting potentially harmful behaviours in language models. While they perform well in general, their effectiveness in detecting text-ambiguous behaviour is disputed. In this work, we find evidence that removing textual evidence of a behaviour significantly decreases probe performance. The AUROC reduction ranges from $10$- to $30$-point depending on the setting. We evaluate probe monitors across three setups (Sandbagging, Sycophancy, and Bias), finding that when probes rely on textual evidence of the target behaviour (such as system prompts or CoT reasoning), performance degrades once these tokens are filtered. This filtering procedure is standard practice for output monitor evaluation. As further evidence of this phenomenon, we train Model Organisms which produce outputs without any behaviour verbalisations. We validate that probe performance on Model Organisms is substantially lower than unfiltered evaluations: $0.57$ vs $0.74$ AUROC for Bias, and $0.57$ vs $0.94$ AUROC for Sandbagging. Our findings suggest that linear probes may be brittle in scenarios where they must detect non-surface-level patterns.
>
---
#### [replaced 080] ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Recall
- **分类: cs.CL**

- **简介: 该论文属于知识编辑任务，解决多跳事实回忆中的性能下降问题。通过分析神经元级联机制，提出ACE框架，有效编辑关键查询-值路径，提升多跳知识更新效果。**

- **链接: [https://arxiv.org/pdf/2510.07896](https://arxiv.org/pdf/2510.07896)**

> **作者:** Jiayu Yang; Yuxuan Fan; Songning Lai; Shengen Wu; Jiaqi Tang; Chun Kang; Zhijiang Guo; Yutao Yue
>
> **备注:** Accepted by ICLR2026
>
> **摘要:** Large Language Models (LLMs) require efficient knowledge editing (KE) to update factual information, yet existing methods exhibit significant performance decay in multi-hop factual recall. This failure is particularly acute when edits involve intermediate implicit subjects within reasoning chains. Through causal analysis, we reveal that this limitation stems from an oversight of how chained knowledge is dynamically represented and utilized at the neuron level. We discover that during multi hop reasoning, implicit subjects function as query neurons, which sequentially activate corresponding value neurons across transformer layers to accumulate information toward the final answer, a dynamic prior KE work has overlooked. Guided by this insight, we propose ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Recall, a framework that leverages neuron-level attribution to identify and edit these critical query-value (Q-V) pathways. ACE provides a mechanistically grounded solution for multi-hop KE, empirically outperforming state-of-the-art methods by 9.44% on GPT-J and 37.46% on Qwen3-8B. Our analysis further reveals more fine-grained activation patterns in Qwen3 and demonstrates that the semantic interpretability of value neurons is orchestrated by query-driven accumulation. These findings establish a new pathway for advancing KE capabilities based on the principled understanding of internal reasoning mechanisms.
>
---
#### [replaced 081] SwingArena: Competitive Programming Arena for Long-context GitHub Issue Solving
- **分类: cs.CL**

- **简介: 该论文提出SwingArena，用于评估大语言模型在真实软件开发场景中的能力，解决长上下文代码生成与验证问题，通过模拟提交与评审流程进行交互式评测。**

- **链接: [https://arxiv.org/pdf/2505.23932](https://arxiv.org/pdf/2505.23932)**

> **作者:** Wendong Xu; Jing Xiong; Chenyang Zhao; Qiujiang Chen; Haoran Wang; Hui Shen; Zhongwei Wan; Jianbo Dai; Taiqiang Wu; He Xiao; Chaofan Tao; Z. Morley Mao; Ying Sheng; Zhijiang Guo; Hongxia Yang; Bei Yu; Lingpeng Kong; Quanquan Gu; Ngai Wong
>
> **备注:** The paper has been accepted as an oral presentation at ICLR 2026
>
> **摘要:** We present SwingArena, a competitive evaluation framework for Large Language Models (LLMs) that closely mirrors real-world software development workflows. Unlike traditional static benchmarks, SwingArena models the collaborative process of software iteration by pairing LLMs as submitters, who generate patches, and reviewers, who create test cases and verify the patches through continuous integration (CI) pipelines. To support these interactive evaluations, we introduce a retrieval-augmented code generation (RACG) module that efficiently handles long-context challenges by providing syntactically and semantically relevant code snippets from large codebases, supporting multiple programming languages (C++, Python, Rust, and Go). This enables the framework to scale across diverse tasks and contexts while respecting token limitations. Our experiments, using over 400 high-quality real-world GitHub issues selected from a pool of 2,300 issues, show that models like GPT-4o excel at aggressive patch generation, whereas DeepSeek and Gemini prioritize correctness in CI validation. SwingArena presents a scalable and extensible methodology for evaluating LLMs in realistic, CI-driven software development settings. More details are available on our project page: this http URL
>
---
#### [replaced 082] MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出MAS-ZERO，解决多智能体系统设计问题，无需监督即可自动构建适应性系统。**

- **链接: [https://arxiv.org/pdf/2505.14996](https://arxiv.org/pdf/2505.14996)**

> **作者:** Zixuan Ke; Austin Xu; Yifei Ming; Xuan-Phi Nguyen; Ryan Chin; Caiming Xiong; Shafiq Joty
>
> **备注:** SEA@NeurIPS (Oral) 2025
>
> **摘要:** Multi-agent systems (MAS) leveraging the impressive capabilities of Large Language Models (LLMs) hold significant potential for tackling complex tasks. However, most current MAS depend on manually designed agent roles and communication protocols. These manual designs often fail to align with the underlying LLMs' strengths and struggle to adapt to novel tasks. Recent automatic MAS approaches attempt to mitigate these limitations but typically necessitate a validation set for tuning and yield static MAS designs lacking adaptability during inference, while also removing the flexibility to reduce to simpler systems. We introduce MAS-ZERO, the first self-evolved, inference-time framework for automatic MAS design. MAS-ZERO employs meta-level design to iteratively design, critique, and refine MAS configurations tailored to each problem instance, without requiring a validation set. Critically, it enables dynamic problem decomposition and agent composition through meta-feedback on solvability and completeness, and reduction to simpler systems when appropriate. Experiments across reasoning (math and graduate-level QA), coding, and agentic (search-based) benchmarks, using both closed-source and open-source LLM backbones of varying sizes, demonstrate that MAS-ZERO outperforms strong manual and automatic MAS baselines. It achieves substantial average accuracy improvements of up to 16.69% on reasoning, 16.66% on coding, and 5.45% on agentic tasks, while maintaining cost efficiency.
>
---
#### [replaced 083] MrBERT: Modern Multilingual Encoders via Vocabulary, Domain, and Dimensional Adaptation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出MrBERT，一种多语言编码器，解决跨语言和领域适应问题。通过优化词汇、领域和维度，提升性能并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2602.21379](https://arxiv.org/pdf/2602.21379)**

> **作者:** Daniel Tamayo; Iñaki Lacunza; Paula Rivera-Hidalgo; Severino Da Dalt; Javier Aula-Blasco; Aitor Gonzalez-Agirre; Marta Villegas
>
> **备注:** 24 pages, 14 tables and 4 figures
>
> **摘要:** We introduce MrBERT, a family of 150M-300M parameter encoders built on the ModernBERT architecture and pre-trained on 35 languages and code. Through targeted adaptation, this model family achieves state-of-the-art results on Catalan- and Spanish-specific tasks, while establishing robust performance across specialized biomedical and legal domains. To bridge the gap between research and production, we incorporate Matryoshka Representation Learning (MRL), enabling flexible vector sizing that significantly reduces inference and storage costs. Ultimately, the MrBERT family demonstrates that modern encoder architectures can be optimized for both localized linguistic excellence and efficient, high-stakes domain specialization. We open source the complete model family on Huggingface.
>
---
#### [replaced 084] Listen to the Layers: Mitigating Hallucinations with Inter-Layer Disagreement
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型生成文本中的幻觉问题。通过分析模型内部层间的不一致性，提出CoCoA解码算法以提升生成内容的准确性。**

- **链接: [https://arxiv.org/pdf/2602.09486](https://arxiv.org/pdf/2602.09486)**

> **作者:** Koduvayur Subbalakshmi; Sabbir Hossain Ujjal; Venkata Krishna Teja Mangichetty; Nastaran Jamalipour Soofi
>
> **备注:** Preprint, 26 pages, 15 tables, 15 figures
>
> **摘要:** Pretrained Large Language Models (LLMs) are prone to generating fluent yet factually incorrect text-a phenomenon known as hallucinations, undermining their reliability and utility in downstream tasks. We hypothesize that a generated text span's factuality is correlated with its representational instability across the model's internal layers. Based on this, we propose the CoCoA (Confusion and Consistency Aware) decoder, a novel, training-free decoding algorithm that mitigates hallucinations at inference time by listening to these signals in the middle layers. We propose two metrics to quantify this instability in the middle layers and use it to penalize outputs that exhibit high internal confusion, thereby steering the model towards more internally consistent and factually grounded outputs. We further propose a self-information gated variant, CoCoA-SIG, that dynamically modulates this penalty to selectively target high-surprise, unstable generations. Extensive experiments on diverse tasks, including question-answering, summarization, mathematical reasoning and code generation, demonstrate that CoCoA significantly improves factual correctness across multiple model families (e.g., Llama-3, Qwen-2.5, Mistral). By leveraging model-intrinsic signals, CoCoA offers an effective and broadly applicable method for enhancing the trustworthiness of LLMs at inference time, without requiring any model retraining.
>
---
#### [replaced 085] Tiny but Mighty: A Software-Hardware Co-Design Approach for Efficient Multimodal Inference on Battery-Powered Small Devices
- **分类: cs.DC; cs.AI; cs.CL; eess.SP**

- **简介: 该论文属于边缘计算任务，旨在解决在电池供电的小设备上高效运行大型多模态模型的问题。通过软硬件协同设计，优化模型部署与资源利用。**

- **链接: [https://arxiv.org/pdf/2510.05109](https://arxiv.org/pdf/2510.05109)**

> **作者:** Yilong Li; Shuai Zhang; Yijing Zeng; Hao Zhang; Xinmiao Xiong; Jingyu Liu; Pan Hu; Suman Banerjee
>
> **摘要:** Large Multimodal Models (LMMs) are inherently modular, consisting of vision and audio encoders, projectors, and large language models. Yet, they are almost always executed monolithically, which underutilizes the heterogeneous accelerators (NPUs, GPUs, DSPs) in modern SoCs and leads to high end-to-end latency. In this paper, we present NANOMIND, a hardware--software co-design inference framework for Large Multimodal Models (LMMs) that breaks large models into modular ``bricks'' (vision, language, audio, etc.) and maps each to its ideal accelerator. The key insight is that large models can be broken into modular components and scheduled to run on the most appropriate compute units. It performs module-level dynamic offloading across accelerators on unified-memory SoCs. By combining customized hardware design, system-level scheduling, and optimized low-bit computation kernels, we demonstrate our framework with a compact, battery-powered device capable of running LMMs entirely on device. This prototype functions as a self-contained intelligent assistant that requires no network connectivity, while achieving higher throughput and superior power efficiency under strict resource constraints. The design further bypasses CPU bottlenecks and reduces redundant memory usage through token-aware buffer management and module-level coordination. Our system outperforms existing implementations in resource efficiency, cutting energy consumption by 42.3\% and GPU memory usage by 11.2\%. This enables a battery-powered device to run LLaVA-OneVision with a camera for nearly 20.8 hours.
>
---
#### [replaced 086] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark
- **分类: cs.AI; cs.CL; cs.DB; cs.LG**

- **简介: 该论文提出MMTU基准，用于评估模型在表格理解与推理方面的能力。针对现有基准任务单一、覆盖不足的问题，设计25个真实表格任务，涵盖复杂表处理技能，以推动模型在结构化数据处理上的发展。**

- **链接: [https://arxiv.org/pdf/2506.05587](https://arxiv.org/pdf/2506.05587)**

> **作者:** Junjie Xing; Yeye He; Mengyu Zhou; Haoyu Dong; Shi Han; Lingjiao Chen; Dongmei Zhang; Surajit Chaudhuri; H. V. Jagadish
>
> **备注:** Full version of a paper accepted at NeurIPS 2025; Code and data available at this https URL and this https URL
>
> **摘要:** Tables and table-based use cases play a crucial role in many important real-world applications, such as spreadsheets, databases, and computational notebooks, which traditionally require expert-level users like data engineers, data analysts, and database administrators to operate. Although LLMs have shown remarkable progress in working with tables (e.g., in spreadsheet and database copilot scenarios), comprehensive benchmarking of such capabilities remains limited. In contrast to an extensive and growing list of NLP benchmarks, evaluations of table-related tasks are scarce, and narrowly focus on tasks like NL-to-SQL and Table-QA, overlooking the broader spectrum of real-world tasks that professional users face. This gap limits our understanding and model progress in this important area. In this work, we introduce MMTU, a large-scale benchmark with over 28K questions across 25 real-world table tasks, designed to comprehensively evaluate models ability to understand, reason, and manipulate real tables at the expert-level. These tasks are drawn from decades' worth of computer science research on tabular data, with a focus on complex table tasks faced by professional users. We show that MMTU require a combination of skills -- including table understanding, reasoning, and coding -- that remain challenging for today's frontier models, where even frontier reasoning models like OpenAI GPT-5 and DeepSeek R1 score only around 69\% and 57\% respectively, suggesting significant room for improvement. We highlight key findings in our evaluation using MMTU and hope that this benchmark drives further advances in understanding and developing foundation models for structured data processing and analysis. Our code and data are available at this https URL and this https URL.
>
---
#### [replaced 087] Mapping Overlaps in Benchmarks through Perplexity in the Wild
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在分析大模型基准测试的重叠情况。通过构建基准签名，研究模型性能与语义相似性的关系，揭示不同任务间的关联性与差异。**

- **链接: [https://arxiv.org/pdf/2509.23488](https://arxiv.org/pdf/2509.23488)**

> **作者:** Siyang Wu; Honglin Bao; Sida Li; Ari Holtzman; James A. Evans
>
> **摘要:** We introduce benchmark signatures to characterize the capacity demands of LLM benchmarks and their overlaps. Signatures are sets of salient tokens from in-the-wild corpora whose model token perplexity, reflecting training exposure, predicts benchmark performance. We extract them via stepwise forward selection with linear regression in a meta-evaluation spanning 32 LLMs and 89 benchmarks across diverse domains. We then analyze how these signatures relate to both the semantic similarity of benchmark questions and the correlation structure of model performance. While performance correlations are uniformly high and semantic overlaps stay in a narrow mid-range, benchmark signatures reveal more nuanced structure. For instance, they uncover substantial overlap between benchmarks in knowledge and reasoning tasks, whereas benchmarks in culture- and humanity-oriented domains show low similarity with each other. Unlike raw performance correlations, which are influenced by benchmark-orthogonal factors such as question formats, signatures are robust to such confounds. We further identify cross-functional overlaps between logic, math, language, instruction following, and cultural/world modeling, with coding emerging as the most isolated function, interacting only moderately with the ability of detecting missing information. Qualitative analysis shows that only the knowledge signature aligns with actual knowledge, suggesting that LLM semantic organization may differ from human conceptual structure. Together, these findings offer insights into benchmark validity, LLM sensitivities, and the landscape of interconnected LLM capacities. We have open-sourced the code and data in this this https URL.
>
---
#### [replaced 088] Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型性能预测任务，旨在解决LLM下游任务性能预测不准确的问题。通过聚类方法构建稳定任务子集，提升预测精度。**

- **链接: [https://arxiv.org/pdf/2502.17262](https://arxiv.org/pdf/2502.17262)**

> **作者:** Chengyin Xu; Kaiyuan Chen; Xiao Li; Ke Shen; Chenggang Li
>
> **备注:** Accepted by The Fourteenth International Conference on Learning Representations (ICLR2026)
>
> **摘要:** The escalating scale and cost of Large Language Models (LLMs) training necessitate accurate pre-training prediction of downstream task performance for comprehensive understanding of scaling properties. This is challenged by: 1) the emergence phenomenon, where unpredictable capabilities appearing suddenly at critical model scales; and 2) uneven task difficulty and inconsistent performance scaling patterns, leading to high metric variability. Current prediction methods lack accuracy and reliability. We propose a Clustering-On-Difficulty (COD) framework for downstream performance prediction. The COD framework clusters tasks by their difficulty scaling features, thereby constructing a more stable and predictable task subset that exhibits well-behaved scaling characteristics with the increase of compute budget. We adopt a performance scaling law to predict cluster-wise performance with theoretical support. Predictable subset performance acts as an intermediate predictor for the full evaluation set. We further derive a mapping function to accurately extrapolate the performance of the subset to the full set. Applied to an LLM with 70B parameters, COD achieved a 1.55\% average prediction error across eight key LLM benchmarks, thus providing actionable insights for scaling properties and training monitoring during LLM pre-training.
>
---
#### [replaced 089] KVSlimmer: Theoretical Insights and Practical Optimizations for Asymmetric KV Merging
- **分类: cs.CL**

- **简介: 该论文属于模型优化任务，旨在解决KV缓存内存与计算开销过高的问题。通过理论分析和算法设计，提出KVSlimmer方法，实现高效KV合并。**

- **链接: [https://arxiv.org/pdf/2603.00907](https://arxiv.org/pdf/2603.00907)**

> **作者:** Lianjun Liu; Hongli An; Weiqi Yan; Xin Du; Shengchuan Zhang; Huazhong Liu; Yunshan Zhong
>
> **摘要:** The growing computational and memory demands of the Key-Value (KV) cache significantly limit the ability of Large Language Models (LLMs). While KV merging has emerged as a promising solution, existing methods that rely on empirical observations of KV asymmetry and gradient-based Hessian approximations lack a theoretical foundation and incur suboptimal compression and inference overhead. To bridge these gaps, we establish a theoretical framework that characterizes this asymmetry through the spectral energy distribution of projection weights, demonstrating that concentrated spectra in Query/Key weights induce feature homogeneity, whereas dispersed spectra in Value weights preserve heterogeneity. Then, we introduce KVSlimmer, an efficient algorithm that captures exact Hessian information through a mathematically exact formulation, and derives a closed-form solution utilizing only forward-pass variables, resulting in a gradient-free approach that is both memory- and time-efficient. Extensive experiments across various models and benchmarks demonstrate that KVSlimmer consistently outperforms SOTA methods. For instance, on Llama3.1-8B-Instruct, it improves the LongBench average score by 0.92 while reducing memory costs and latency by 29% and 28%, this http URL is available at this https URL.
>
---
#### [replaced 090] Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于AI安全领域，解决多轮对话中对抗攻击的问题。提出DialTree框架，通过强化学习自动发现多样化的多轮攻击策略。**

- **链接: [https://arxiv.org/pdf/2510.02286](https://arxiv.org/pdf/2510.02286)**

> **作者:** Ruohao Guo; Afshin Oroojlooy; Roshan Sridhar; Miguel Ballesteros; Alan Ritter; Dan Roth
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Despite recent rapid progress in AI safety, current large language models remain vulnerable to adversarial attacks in multi-turn interaction settings, where attackers strategically adapt their prompts across conversation turns and pose a more critical yet realistic challenge. Existing approaches that discover safety vulnerabilities either rely on manual red-teaming with human experts or employ automated methods using pre-defined templates and human-curated attack data, with most focusing on single-turn attacks. However, these methods did not explore the vast space of possible multi-turn attacks, failing to consider novel attack trajectories that emerge from complex dialogue dynamics and strategic conversation planning. This gap is particularly critical given recent findings that LLMs exhibit significantly higher vulnerability to multi-turn attacks compared to single-turn attacks. We propose DialTree, an on-policy reinforcement learning framework integrated with tree search that autonomously discovers diverse multi-turn attack strategies by treating the dialogue as a sequential decision-making problem, enabling systematic exploration without manually curated data. Through extensive experiments, our approach not only achieves more than 44.2% higher ASR across 12 target models compared to previous state-of-the-art approaches, but also effectively uncovers new attack strategies by learning optimal dialogue policies that maximize attack success across multiple turns.
>
---
#### [replaced 091] AgentIR: Reasoning-Aware Retrieval for Deep Research Agents
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决深度研究代理在搜索时缺乏上下文理解的问题。通过引入推理感知检索和数据合成方法，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2603.04384](https://arxiv.org/pdf/2603.04384)**

> **作者:** Zijian Chen; Xueguang Ma; Shengyao Zhuang; Jimmy Lin; Akari Asai; Victor Zhong
>
> **摘要:** Deep Research agents are rapidly emerging as primary consumers of modern retrieval systems. Unlike human users who issue and refine queries without documenting their intermediate thought processes, Deep Research agents generate explicit natural language reasoning before each search call, revealing rich intent and contextual information that existing retrievers entirely ignore. To exploit this overlooked signal, we introduce: (1) Reasoning-Aware Retrieval, a retrieval paradigm that jointly embeds the agent's reasoning trace alongside its query; and (2) DR-Synth, a data synthesis method that generates Deep Research retriever training data from standard QA datasets. We demonstrate that both components are independently effective, and their combination yields a trained embedding model, AgentIR-4B, with substantial gains. On the challenging BrowseComp-Plus benchmark, AgentIR-4B achieves 68\% accuracy with the open-weight agent Tongyi-DeepResearch, compared to 50\% with conventional embedding models twice its size, and 37\% with BM25. Code and data are available at: this https URL.
>
---
