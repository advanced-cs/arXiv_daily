# 自然语言处理 cs.CL

- **最新发布 144 篇**

- **更新 103 篇**

## 最新发布

#### [new 001] SugarTextNet: A Transformer-Based Framework for Detecting Sugar Dating-Related Content on Social Media with Context-Aware Focal Loss
- **分类: cs.CL; cs.CY; cs.SI**

- **简介: 论文提出SugarTextNet，用于检测社交媒体中的“糖 dating”相关内容，解决文本隐晦、类别不平衡难题。结合Transformer与上下文感知焦点损失，提升小样本识别准确率，在中文微博数据上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2511.06402v1](http://arxiv.org/pdf/2511.06402v1)**

> **作者:** Lionel Z. Wang; Shihan Ben; Yulu Huang; Simeng Qing
>
> **备注:** This paper is accepted by HICSS 2026
>
> **摘要:** Sugar dating-related content has rapidly proliferated on mainstream social media platforms, giving rise to serious societal and regulatory concerns, including commercialization of intimate relationships and the normalization of transactional relationships.~Detecting such content is highly challenging due to the prevalence of subtle euphemisms, ambiguous linguistic cues, and extreme class imbalance in real-world data.~In this work, we present SugarTextNet, a novel transformer-based framework specifically designed to identify sugar dating-related posts on social media.~SugarTextNet integrates a pretrained transformer encoder, an attention-based cue extractor, and a contextual phrase encoder to capture both salient and nuanced features in user-generated text.~To address class imbalance and enhance minority-class detection, we introduce Context-Aware Focal Loss, a tailored loss function that combines focal loss scaling with contextual weighting.~We evaluate SugarTextNet on a newly curated, manually annotated dataset of 3,067 Chinese social media posts from Sina Weibo, demonstrating that our approach substantially outperforms traditional machine learning models, deep learning baselines, and large language models across multiple metrics.~Comprehensive ablation studies confirm the indispensable role of each component.~Our findings highlight the importance of domain-specific, context-aware modeling for sensitive content detection, and provide a robust solution for content moderation in complex, real-world scenarios.
>
---
#### [new 002] Sensitivity of Small Language Models to Fine-tuning Data Contamination
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究小语言模型（SLMs）在指令微调中对数据污染的敏感性，发现语法污染（如字符反转）导致灾难性退化，语义污染则引发“能力诅咒”——更大模型更易学坏指令，揭示了鲁棒性评估的缺失与训练协议改进需求。**

- **链接: [http://arxiv.org/pdf/2511.06763v1](http://arxiv.org/pdf/2511.06763v1)**

> **作者:** Nicy Scaria; Silvester John Joseph Kennedy; Deepak Subramani
>
> **摘要:** Small Language Models (SLMs) are increasingly being deployed in resource-constrained environments, yet their behavioral robustness to data contamination during instruction tuning remains poorly understood. We systematically investigate the contamination sensitivity of 23 SLMs (270M to 4B parameters) across multiple model families by measuring susceptibility to syntactic and semantic transformation types during instruction tuning: syntactic transformations (character and word reversal) and semantic transformations (irrelevant and counterfactual responses), each applied at contamination levels of 25\%, 50\%, 75\%, and 100\%. Our results reveal fundamental asymmetries in vulnerability patterns: syntactic transformations cause catastrophic performance degradation, with character reversal producing near-complete failure across all models regardless of size or family, while semantic transformations demonstrate distinct threshold behaviors and greater resilience in core linguistic capabilities. Critically, we discover a ``\textit{capability curse}" where larger, more capable models become more susceptible to learning semantic corruptions, effectively following harmful instructions more readily, while our analysis of base versus instruction-tuned variants reveals that alignment provides inconsistent robustness benefits, sometimes even reducing resilience. Our work establishes three core contributions: (1) empirical evidence of SLMs' disproportionate vulnerability to syntactic pattern contamination, (2) identification of asymmetric sensitivity patterns between syntactic and semantic transformations, and (3) systematic evaluation protocols for contamination robustness assessment. These findings have immediate deployment implications, suggesting that current robustness assumptions may not hold for smaller models and highlighting the need for contamination-aware training protocols.
>
---
#### [new 003] CLiFT-ASR: A Cross-Lingual Fine-Tuning Framework for Low-Resource Taiwanese Hokkien Speech Recognition
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出CLiFT-ASR框架，用于低资源台语语音识别，解决单一标注方式信息缺失问题。通过两阶段跨语言微调，先利用台罗拼音学习音调特征，再结合汉字文本学习词汇语法，显著降低字符错误率。**

- **链接: [http://arxiv.org/pdf/2511.06860v1](http://arxiv.org/pdf/2511.06860v1)**

> **作者:** Hung-Yang Sung; Chien-Chun Wang; Kuan-Tang Huang; Tien-Hong Lo; Yu-Sheng Tsao; Yung-Chang Hsu; Berlin Chen
>
> **备注:** Accepted for an oral presentation at the 37th Conference on Computational Linguistics and Speech Processing (ROCLING 2025)
>
> **摘要:** Automatic speech recognition (ASR) for low-resource languages such as Taiwanese Hokkien is difficult due to the scarcity of annotated data. However, direct fine-tuning on Han-character transcriptions often fails to capture detailed phonetic and tonal cues, while training only on romanization lacks lexical and syntactic coverage. In addition, prior studies have rarely explored staged strategies that integrate both annotation types. To address this gap, we present CLiFT-ASR, a cross-lingual fine-tuning framework that builds on Mandarin HuBERT models and progressively adapts them to Taiwanese Hokkien. The framework employs a two-stage process in which it first learns acoustic and tonal representations from phonetic Tai-lo annotations and then captures vocabulary and syntax from Han-character transcriptions. This progressive adaptation enables effective alignment between speech sounds and orthographic structures. Experiments on the TAT-MOE corpus demonstrate that CLiFT-ASR achieves a 24.88\% relative reduction in character error rate (CER) compared with strong baselines. The results indicate that CLiFT-ASR provides an effective and parameter-efficient solution for Taiwanese Hokkien ASR and that it has potential to benefit other low-resource language scenarios.
>
---
#### [new 004] EMODIS: A Benchmark for Context-Dependent Emoji Disambiguation in Large Language Models
- **分类: cs.CL**

- **简介: 论文提出EMODIS基准，评估大语言模型在细微语境下对歧义表情符号的语义消歧能力，揭示模型对语用对比敏感性不足与偏好主导解释的缺陷，填补了上下文依赖的emoji理解评估空白。**

- **链接: [http://arxiv.org/pdf/2511.07193v1](http://arxiv.org/pdf/2511.07193v1)**

> **作者:** Jiacheng Huang; Ning Yu; Xiaoyin Yi
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** Large language models (LLMs) are increasingly deployed in real-world communication settings, yet their ability to resolve context-dependent ambiguity remains underexplored. In this work, we present EMODIS, a new benchmark for evaluating LLMs' capacity to interpret ambiguous emoji expressions under minimal but contrastive textual contexts. Each instance in EMODIS comprises an ambiguous sentence containing an emoji, two distinct disambiguating contexts that lead to divergent interpretations, and a specific question that requires contextual reasoning. We evaluate both open-source and API-based LLMs, and find that even the strongest models frequently fail to distinguish meanings when only subtle contextual cues are present. Further analysis reveals systematic biases toward dominant interpretations and limited sensitivity to pragmatic contrast. EMODIS provides a rigorous testbed for assessing contextual disambiguation, and highlights the gap in semantic reasoning between humans and LLMs.
>
---
#### [new 005] UTF-8 Plumbing: Byte-level Tokenizers Unavoidably Enable LLMs to Generate Ill-formed UTF-8
- **分类: cs.CL**

- **简介: 该论文研究字节级分词器导致LLM生成非法UTF-8序列的问题，证明其不可避免性，并指出增量解码与整体解码结果不一致，引发实际系统错误，提出形式化分析与缓解方案。**

- **链接: [http://arxiv.org/pdf/2511.05578v1](http://arxiv.org/pdf/2511.05578v1)**

> **作者:** Preston Firestone; Shubham Ugare; Gagandeep Singh; Sasa Misailovic
>
> **备注:** COLM 2025
>
> **摘要:** Subword tokenization segments input text according to a pre-defined vocabulary to feed it into a language model; the language model, in turn, generates a sequence made from this same vocabulary. The members of the vocabulary can be built of code points or bytes. Using code points means that all members of the vocabulary are valid UTF-8 characters. However, it also requires thousands of initial members to achieve acceptable coverage of inputs. Beginning with bytes, on the contrary, avoids out-of-vocabulary errors with only 256 initial members of the vocabulary, but the members of the vocabulary and sequences of them are not guaranteed to be valid UTF-8. Sequences that are not valid UTF-8 break code that assumes its input to be valid UTF-8. Applications of language models must account for the breakage thereby introduced. In this paper, we formalize tokenization using monoid theory and prove that tokenizers whose vocabularies contain tokens that are ill-formed UTF-8 can always produce sequences that are ill-formed UTF-8. We demonstrate formally that attempting to incrementally convert tokens back to a string and interpret the results as UTF-8 gives different results than converting the whole sequence of tokens at once. This formal result predicts real-world bugs: we evaluate mitigations for the problem identified and provide case studies of major foundation models, serving engines, and constrained generation systems.
>
---
#### [new 006] Rethinking Retrieval-Augmented Generation for Medicine: A Large-Scale, Systematic Expert Evaluation and Practical Insights
- **分类: cs.CL**

- **简介: 该论文评估医学领域RAG系统的有效性，发现其常因检索与证据选择不佳而降低输出质量，提出过滤与查询重写等策略显著提升性能，呼吁重新设计医学LLM的RAG架构。**

- **链接: [http://arxiv.org/pdf/2511.06738v1](http://arxiv.org/pdf/2511.06738v1)**

> **作者:** Hyunjae Kim; Jiwoong Sohn; Aidan Gilson; Nicholas Cochran-Caggiano; Serina Applebaum; Heeju Jin; Seihee Park; Yujin Park; Jiyeong Park; Seoyoung Choi; Brittany Alexandra Herrera Contreras; Thomas Huang; Jaehoon Yun; Ethan F. Wei; Roy Jiang; Leah Colucci; Eric Lai; Amisha Dave; Tuo Guo; Maxwell B. Singer; Yonghoe Koo; Ron A. Adelman; James Zou; Andrew Taylor; Arman Cohan; Hua Xu; Qingyu Chen
>
> **备注:** 34 pages, 6 figures
>
> **摘要:** Large language models (LLMs) are transforming the landscape of medicine, yet two fundamental challenges persist: keeping up with rapidly evolving medical knowledge and providing verifiable, evidence-grounded reasoning. Retrieval-augmented generation (RAG) has been widely adopted to address these limitations by supplementing model outputs with retrieved evidence. However, whether RAG reliably achieves these goals remains unclear. Here, we present the most comprehensive expert evaluation of RAG in medicine to date. Eighteen medical experts contributed a total of 80,502 annotations, assessing 800 model outputs generated by GPT-4o and Llama-3.1-8B across 200 real-world patient and USMLE-style queries. We systematically decomposed the RAG pipeline into three components: (i) evidence retrieval (relevance of retrieved passages), (ii) evidence selection (accuracy of evidence usage), and (iii) response generation (factuality and completeness of outputs). Contrary to expectation, standard RAG often degraded performance: only 22% of top-16 passages were relevant, evidence selection remained weak (precision 41-43%, recall 27-49%), and factuality and completeness dropped by up to 6% and 5%, respectively, compared with non-RAG variants. Retrieval and evidence selection remain key failure points for the model, contributing to the overall performance drop. We further show that simple yet effective strategies, including evidence filtering and query reformulation, substantially mitigate these issues, improving performance on MedMCQA and MedXpertQA by up to 12% and 8.2%, respectively. These findings call for re-examining RAG's role in medicine and highlight the importance of stage-aware evaluation and deliberate system design for reliable medical LLM applications.
>
---
#### [new 007] Categorical Emotions or Appraisals - Which Emotion Model Explains Argument Convincingness Better?
- **分类: cs.CL**

- **简介: 该论文研究论证说服力的情感建模问题，比较类别情绪与评价理论的效能。基于ContArgA语料库，通过零样本提示实验发现，评价理论比传统情绪类别更能有效预测主观说服力，首次系统证明评价理论在计算论证中的优势。**

- **链接: [http://arxiv.org/pdf/2511.07162v1](http://arxiv.org/pdf/2511.07162v1)**

> **作者:** Lynn Greschner; Meike Bauer; Sabine Weber; Roman Klinger
>
> **摘要:** The convincingness of an argument does not only depend on its structure (logos), the person who makes the argument (ethos), but also on the emotion that it causes in the recipient (pathos). While the overall intensity and categorical values of emotions in arguments have received considerable attention in the research community, we argue that the emotion an argument evokes in a recipient is subjective. It depends on the recipient's goals, standards, prior knowledge, and stance. Appraisal theories lend themselves as a link between the subjective cognitive assessment of events and emotions. They have been used in event-centric emotion analysis, but their suitability for assessing argument convincingness remains unexplored. In this paper, we evaluate whether appraisal theories are suitable for emotion analysis in arguments by considering subjective cognitive evaluations of the importance and impact of an argument on its receiver. Based on the annotations in the recently published ContArgA corpus, we perform zero-shot prompting experiments to evaluate the importance of gold-annotated and predicted emotions and appraisals for the assessment of the subjective convincingness labels. We find that, while categorical emotion information does improve convincingness prediction, the improvement is more pronounced with appraisals. This work presents the first systematic comparison between emotion models for convincingness prediction, demonstrating the advantage of appraisals, providing insights for theoretical and practical applications in computational argumentation.
>
---
#### [new 008] Multi-Scale Feature Fusion and Graph Neural Network Integration for Text Classification with Large Language Models
- **分类: cs.CL**

- **简介: 该论文面向文本分类任务，提出融合大语言模型、多尺度特征金字塔与图神经网络的混合框架，以增强语义建模能力，有效整合全局与局部信息及结构依赖，显著提升分类性能。**

- **链接: [http://arxiv.org/pdf/2511.05752v1](http://arxiv.org/pdf/2511.05752v1)**

> **作者:** Xiangchen Song; Yulin Huang; Jinxu Guo; Yuchen Liu; Yaxuan Luan
>
> **摘要:** This study investigates a hybrid method for text classification that integrates deep feature extraction from large language models, multi-scale fusion through feature pyramids, and structured modeling with graph neural networks to enhance performance in complex semantic contexts. First, the large language model captures contextual dependencies and deep semantic representations of the input text, providing a rich feature foundation for subsequent modeling. Then, based on multi-level feature representations, the feature pyramid mechanism effectively integrates semantic features of different scales, balancing global information and local details to construct hierarchical semantic expressions. Furthermore, the fused features are transformed into graph representations, and graph neural networks are employed to capture latent semantic relations and logical dependencies in the text, enabling comprehensive modeling of complex interactions among semantic units. On this basis, the readout and classification modules generate the final category predictions. The proposed method demonstrates significant advantages in robustness alignment experiments, outperforming existing models on ACC, F1-Score, AUC, and Precision, which verifies the effectiveness and stability of the framework. This study not only constructs an integrated framework that balances global and local information as well as semantics and structure, but also provides a new perspective for multi-scale feature fusion and structured semantic modeling in text classification tasks.
>
---
#### [new 009] SR-KI: Scalable and Real-Time Knowledge Integration into LLMs via Supervised Attention
- **分类: cs.CL; cs.AI**

- **简介: SR-KI提出一种通过监督注意力将大规模知识库实时集成到LLM的新方法，无需外部检索器，直接在模型潜空间中实现端到端知识检索与更新，支持高效压缩与动态更新，显著提升检索与问答性能。**

- **链接: [http://arxiv.org/pdf/2511.06446v1](http://arxiv.org/pdf/2511.06446v1)**

> **作者:** Bohan Yu; Wei Huang; Kang Liu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** This paper proposes SR-KI, a novel approach for integrating real-time and large-scale structured knowledge bases (KBs) into large language models (LLMs). SR-KI begins by encoding KBs into key-value pairs using a pretrained encoder, and injects them into LLMs' KV cache. Building on this representation, we employ a two-stage training paradigm: first locating a dedicated retrieval layer within the LLM, and then applying an attention-based loss at this layer to explicitly supervise attention toward relevant KB entries. Unlike traditional retrieval-augmented generation methods that rely heavily on the performance of external retrievers and multi-stage pipelines, SR-KI supports end-to-end inference by performing retrieval entirely within the models latent space. This design enables efficient compression of injected knowledge and facilitates dynamic knowledge updates. Comprehensive experiments demonstrate that SR-KI enables the integration of up to 40K KBs into a 7B LLM on a single A100 40GB GPU, and achieves strong retrieval performance, maintaining over 98% Recall@10 on the best-performing task and exceeding 88% on average across all tasks. Task performance on question answering and KB ID generation also demonstrates that SR-KI maintains strong performance while achieving up to 99.75% compression of the injected KBs.
>
---
#### [new 010] LLMs Do Not See Age: Assessing Demographic Bias in Automated Systematic Review Synthesis
- **分类: cs.CL**

- **简介: 该论文评估LLM在生成系统综述摘要时对年龄信息的保留能力，发现模型存在年龄偏差，尤其成人组信息丢失严重、弱势群体易幻觉。提出DemogSummary数据集与DSS指标，揭示当前模型在生物医学摘要生成中的公平性缺陷。**

- **链接: [http://arxiv.org/pdf/2511.06000v1](http://arxiv.org/pdf/2511.06000v1)**

> **作者:** Favour Yahdii Aghaebe; Tanefa Apekey; Elizabeth Williams; Nafise Sadat Moosavi
>
> **备注:** Accepted at AACL 2025
>
> **摘要:** Clinical interventions often hinge on age: medications and procedures safe for adults may be harmful to children or ineffective for older adults. However, as language models are increasingly integrated into biomedical evidence synthesis workflows, it remains uncertain whether these systems preserve such crucial demographic distinctions. To address this gap, we evaluate how well state-of-the-art language models retain age-related information when generating abstractive summaries of biomedical studies. We construct DemogSummary, a novel age-stratified dataset of systematic review primary studies, covering child, adult, and older adult populations. We evaluate three prominent summarisation-capable LLMs, Qwen (open-source), Longformer (open-source) and GPT-4.1 Nano (proprietary), using both standard metrics and a newly proposed Demographic Salience Score (DSS), which quantifies age-related entity retention and hallucination. Our results reveal systematic disparities across models and age groups: demographic fidelity is lowest for adult-focused summaries, and under-represented populations are more prone to hallucinations. These findings highlight the limitations of current LLMs in faithful and bias-free summarisation and point to the need for fairness-aware evaluation frameworks and summarisation pipelines in biomedical NLP.
>
---
#### [new 011] DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: DRAGON提出一种无需微调的上下文推理框架，用于大语言模型的无数据遗忘，通过负样本检测与链式推理实现安全干预，解决现实场景中保留数据缺失下的高效遗忘问题。**

- **链接: [http://arxiv.org/pdf/2511.05784v1](http://arxiv.org/pdf/2511.05784v1)**

> **作者:** Yaxuan Wang; Chris Yuhao Liu; Quan Liu; Jinglong Pang; Wei Wei; Yujia Bao; Yang Liu
>
> **备注:** Please refer to the NeurIPS 2025 submission: https://openreview.net/forum?id=FNuul0hlin; The paper has been accepted to the ICML 2025 MUGen Workshop: https://openreview.net/forum?id=ET24oKP23c
>
> **摘要:** Unlearning in Large Language Models (LLMs) is crucial for protecting private data and removing harmful knowledge. Most existing approaches rely on fine-tuning to balance unlearning efficiency with general language capabilities. However, these methods typically require training or access to retain data, which is often unavailable in real world scenarios. Although these methods can perform well when both forget and retain data are available, few works have demonstrated equivalent capability in more practical, data-limited scenarios. To overcome these limitations, we propose Detect-Reasoning Augmented GeneratiON (DRAGON), a systematic, reasoning-based framework that utilizes in-context chain-of-thought (CoT) instructions to guard deployed LLMs before inference. Instead of modifying the base model, DRAGON leverages the inherent instruction-following ability of LLMs and introduces a lightweight detection module to identify forget-worthy prompts without any retain data. These are then routed through a dedicated CoT guard model to enforce safe and accurate in-context intervention. To robustly evaluate unlearning performance, we introduce novel metrics for unlearning performance and the continual unlearning setting. Extensive experiments across three representative unlearning tasks validate the effectiveness of DRAGON, demonstrating its strong unlearning capability, scalability, and applicability in practical scenarios.
>
---
#### [new 012] Language Generation: Complexity Barriers and Implications for Learning
- **分类: cs.CL; cs.AI; cs.FL; cs.LG**

- **简介: 该论文研究语言生成的可学习性，揭示即使对简单语言族，所需示例可能远超可计算界限，指出理论可行与实践高效间的巨大鸿沟，呼吁关注自然语言结构特性以解释现代语言模型的成功。**

- **链接: [http://arxiv.org/pdf/2511.05759v1](http://arxiv.org/pdf/2511.05759v1)**

> **作者:** Marcelo Arenas; Pablo Barceló; Luis Cofré; Alexander Kozachinskiy
>
> **摘要:** Kleinberg and Mullainathan showed that, in principle, language generation is always possible: with sufficiently many positive examples, a learner can eventually produce sentences indistinguishable from those of a target language. However, the existence of such a guarantee does not speak to its practical feasibility. In this work, we show that even for simple and well-studied language families -- such as regular and context-free languages -- the number of examples required for successful generation can be extraordinarily large, and in some cases not bounded by any computable function. These results reveal a substantial gap between theoretical possibility and efficient learnability. They suggest that explaining the empirical success of modern language models requires a refined perspective -- one that takes into account structural properties of natural language that make effective generation possible in practice.
>
---
#### [new 013] HLPD: Aligning LLMs to Human Language Preference for Machine-Revised Text Detection
- **分类: cs.CL; cs.CR**

- **简介: 论文提出HLPD方法，通过人类语言偏好优化（HLPO）对齐大模型评分器，提升对机器润色文本的检测能力，解决黑盒环境下先进LLM生成/润色文本的识别难题，在多个基准上显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2511.06942v1](http://arxiv.org/pdf/2511.06942v1)**

> **作者:** Fangqi Dai; Xingjian Jiang; Zizhuang Deng
>
> **备注:** 9 pages, 3 figures, accepted by AAAI'26
>
> **摘要:** To prevent misinformation and social issues arising from trustworthy-looking content generated by LLMs, it is crucial to develop efficient and reliable methods for identifying the source of texts. Previous approaches have demonstrated exceptional performance in detecting texts fully generated by LLMs. However, these methods struggle when confronting more advanced LLM output or text with adversarial multi-task machine revision, especially in the black-box setting, where the generating model is unknown. To address this challenge, grounded in the hypothesis that human writing possesses distinctive stylistic patterns, we propose Human Language Preference Detection (HLPD). HLPD employs a reward-based alignment process, Human Language Preference Optimization (HLPO), to shift the scoring model's token distribution toward human-like writing, making the model more sensitive to human writing, therefore enhancing the identification of machine-revised text. We test HLPD in an adversarial multi-task evaluation framework that leverages a five-dimensional prompt generator and multiple advanced LLMs to create diverse revision scenarios. When detecting texts revised by GPT-series models, HLPD achieves a 15.11% relative improvement in AUROC over ImBD, surpassing Fast-DetectGPT by 45.56%. When evaluated on texts generated by advanced LLMs, HLPD achieves the highest average AUROC, exceeding ImBD by 5.53% and Fast-DetectGPT by 34.14%. Code will be made available at https://github.com/dfq2021/HLPD.
>
---
#### [new 014] Textual Self-attention Network: Test-Time Preference Optimization through Textual Gradient-based Attention
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Textual Self-Attention Network（TSAN），用于测试时偏好对齐，无需微调。通过自然语言形式的自注意力机制，融合多个候选响应的优势，生成更符合人类偏好的输出，超越现有方法与大型微调模型。**

- **链接: [http://arxiv.org/pdf/2511.06682v1](http://arxiv.org/pdf/2511.06682v1)**

> **作者:** Shibing Mo; Haoyang Ruan; Kai Wu; Jing Liu
>
> **备注:** AAAI2026
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable generalization capabilities, but aligning their outputs with human preferences typically requires expensive supervised fine-tuning. Recent test-time methods leverage textual feedback to overcome this, but they often critique and revise a single candidate response, lacking a principled mechanism to systematically analyze, weigh, and synthesize the strengths of multiple promising candidates. Such a mechanism is crucial because different responses may excel in distinct aspects (e.g., clarity, factual accuracy, or tone), and combining their best elements may produce a far superior outcome. This paper proposes the Textual Self-Attention Network (TSAN), a new paradigm for test-time preference optimization that requires no parameter updates. TSAN emulates self-attention entirely in natural language to overcome this gap: it analyzes multiple candidates by formatting them into textual keys and values, weighs their relevance using an LLM-based attention module, and synthesizes their strengths into a new, preference-aligned response under the guidance of the learned textual attention. This entire process operates in a textual gradient space, enabling iterative and interpretable optimization. Empirical evaluations demonstrate that with just three test-time iterations on a base SFT model, TSAN outperforms supervised models like Llama-3.1-70B-Instruct and surpasses the current state-of-the-art test-time alignment method by effectively leveraging multiple candidate solutions.
>
---
#### [new 015] FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation
- **分类: cs.CL; cs.AI**

- **简介: 论文首次提出股权研究报告自动生成任务，构建开源基准FinRpt，包含自动化构建的数据集与11项评估指标，并提出多智能体框架FinRpt-Gen，基于LLM实现高效报告生成。**

- **链接: [http://arxiv.org/pdf/2511.07322v1](http://arxiv.org/pdf/2511.07322v1)**

> **作者:** Song Jin; Shuqi Li; Shukun Zhang; Rui Yan
>
> **备注:** AAAI 2026
>
> **摘要:** While LLMs have shown great success in financial tasks like stock prediction and question answering, their application in fully automating Equity Research Report generation remains uncharted territory. In this paper, we formulate the Equity Research Report (ERR) Generation task for the first time. To address the data scarcity and the evaluation metrics absence, we present an open-source evaluation benchmark for ERR generation - FinRpt. We frame a Dataset Construction Pipeline that integrates 7 financial data types and produces a high-quality ERR dataset automatically, which could be used for model training and evaluation. We also introduce a comprehensive evaluation system including 11 metrics to assess the generated ERRs. Moreover, we propose a multi-agent framework specifically tailored to address this task, named FinRpt-Gen, and train several LLM-based agents on the proposed datasets using Supervised Fine-Tuning and Reinforcement Learning. Experimental results indicate the data quality and metrics effectiveness of the benchmark FinRpt and the strong performance of FinRpt-Gen, showcasing their potential to drive innovation in the ERR generation field. All code and datasets are publicly available.
>
---
#### [new 016] Learning to Focus: Focal Attention for Selective and Scalable Transformers
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Focal Attention，通过调节softmax温度来 sharpen 注意力分布，解决标准Transformer在长上下文下注意力噪声大、效率低的问题，显著提升模型参数与数据效率，在长文本任务上实现17%-82%的性能提升。**

- **链接: [http://arxiv.org/pdf/2511.06818v1](http://arxiv.org/pdf/2511.06818v1)**

> **作者:** Dhananjay Ram; Wei Xia; Stefano Soatto
>
> **摘要:** Attention is a core component of transformer architecture, whether encoder-only, decoder-only, or encoder-decoder model. However, the standard softmax attention often produces noisy probability distribution, which can impair effective feature selection at every layer of these models, particularly for long contexts. We propose Focal Attention, a simple yet effective modification that sharpens the attention distribution by controlling the softmax temperature, either as a fixed hyperparameter or as a learnable parameter during training. This sharpening enables the model to concentrate on the most relevant tokens while suppressing irrelevant ones. Empirically, Focal Attention scales more favorably than standard transformer with respect to model size, training data, and context length. Across diverse benchmarks, it achieves the same accuracy with up to 42% fewer parameters or 33% less training data. On long-context tasks, it delivers substantial relative improvements ranging from 17% to 82%, demonstrating its effectiveness in real world applications.
>
---
#### [new 017] Future of AI Models: A Computational perspective on Model collapse
- **分类: cs.CL; cs.DB; cs.IT; math.IT; I.2.0**

- **简介: 该论文研究AI模型坍塌问题，量化分析2013–2025年英文语料语义相似度上升趋势，揭示LLM普及后语料多样性急剧下降，预测递归训练导致的数据退化临界点。**

- **链接: [http://arxiv.org/pdf/2511.05535v1](http://arxiv.org/pdf/2511.05535v1)**

> **作者:** Trivikram Satharasi; S Sitharama Iyengar
>
> **备注:** Submitted to Springer Nature. Code Available at https://github.com/t-satharasi/AI-Modal-Collapse-Code-for-Reproduction.git
>
> **摘要:** Artificial Intelligence, especially Large Language Models (LLMs), has transformed domains such as software engineering, journalism, creative writing, academia, and media (Naveed et al. 2025; arXiv:2307.06435). Diffusion models like Stable Diffusion generate high-quality images and videos from text. Evidence shows rapid expansion: 74.2% of newly published webpages now contain AI-generated material (Ryan Law 2025), 30-40% of the active web corpus is synthetic (Spennemann 2025; arXiv:2504.08755), 52% of U.S. adults use LLMs for writing, coding, or research (Staff 2025), and audits find AI involvement in 18% of financial complaints and 24% of press releases (Liang et al. 2025). The underlying neural architectures, including Transformers (Vaswani et al. 2023; arXiv:1706.03762), RNNs, LSTMs, GANs, and diffusion networks, depend on large, diverse, human-authored datasets (Shi & Iyengar 2019). As synthetic content dominates, recursive training risks eroding linguistic and semantic diversity, producing Model Collapse (Shumailov et al. 2024; arXiv:2307.15043; Dohmatob et al. 2024; arXiv:2402.07712). This study quantifies and forecasts collapse onset by examining year-wise semantic similarity in English-language Wikipedia (filtered Common Crawl) from 2013 to 2025 using Transformer embeddings and cosine similarity metrics. Results reveal a steady rise in similarity before public LLM adoption, likely driven by early RNN/LSTM translation and text-normalization pipelines, though modest due to a smaller scale. Observed fluctuations reflect irreducible linguistic diversity, variable corpus size across years, finite sampling error, and an exponential rise in similarity after the public adoption of LLM models. These findings provide a data-driven estimate of when recursive AI contamination may significantly threaten data richness and model generalization.
>
---
#### [new 018] TCM-Eval: An Expert-Level Dynamic and Extensible Benchmark for Traditional Chinese Medicine
- **分类: cs.CL**

- **简介: 论文提出TCM-Eval首个动态可扩展中医评估基准，解决中医领域缺乏高质量数据与评估标准的问题，构建大规模训练语料并提出SI-CoTE方法自增强推理链，发布中医大模型ZhiMingTang，超越人类医师通过率。**

- **链接: [http://arxiv.org/pdf/2511.07148v1](http://arxiv.org/pdf/2511.07148v1)**

> **作者:** Zihao Cheng; Yuheng Lu; Huaiqian Ye; Zeming Liu; Minqi Wang; Jingjing Liu; Zihan Li; Wei Fan; Yuanfang Guo; Ruiji Fu; Shifeng She; Gang Wang; Yunhong Wang
>
> **备注:** Work in Progress
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in modern medicine, yet their application in Traditional Chinese Medicine (TCM) remains severely limited by the absence of standardized benchmarks and the scarcity of high-quality training data. To address these challenges, we introduce TCM-Eval, the first dynamic and extensible benchmark for TCM, meticulously curated from national medical licensing examinations and validated by TCM experts. Furthermore, we construct a large-scale training corpus and propose Self-Iterative Chain-of-Thought Enhancement (SI-CoTE) to autonomously enrich question-answer pairs with validated reasoning chains through rejection sampling, establishing a virtuous cycle of data and model co-evolution. Using this enriched training data, we develop ZhiMingTang (ZMT), a state-of-the-art LLM specifically designed for TCM, which significantly exceeds the passing threshold for human practitioners. To encourage future research and development, we release a public leaderboard, fostering community engagement and continuous improvement.
>
---
#### [new 019] Automated Circuit Interpretation via Probe Prompting
- **分类: cs.CL; I.2.0; I.2.6; I.2.7; I.2.4**

- **简介: 该论文提出Probe Prompting，自动化解释神经网络的因果电路。通过概念对齐的探针与决策规则，将复杂属性图压缩为可解释子图，提升语义一致性与解释效率，揭示Transformer的分层计算结构。**

- **链接: [http://arxiv.org/pdf/2511.07002v1](http://arxiv.org/pdf/2511.07002v1)**

> **作者:** Giuseppe Birardi
>
> **备注:** 27 pages, 5 figures, 3 tables. Code and interactive demo available
>
> **摘要:** Mechanistic interpretability aims to understand neural networks by identifying which learned features mediate specific behaviors. Attribution graphs reveal these feature pathways, but interpreting them requires extensive manual analysis -- a single prompt can take approximately 2 hours for an experienced circuit tracer. We present probe prompting, an automated pipeline that transforms attribution graphs into compact, interpretable subgraphs built from concept-aligned supernodes. Starting from a seed prompt and target logit, we select high-influence features, generate concept-targeted yet context-varying probes, and group features by cross-prompt activation signatures into Semantic, Relationship, and Say-X categories using transparent decision rules. Across five prompts including classic "capitals" circuits, probe-prompted subgraphs preserve high explanatory coverage while compressing complexity (Completeness 0.83, mean across circuits; Replacement 0.54). Compared to geometric clustering baselines, concept-aligned groups exhibit higher behavioral coherence: 2.3x higher peak-token consistency (0.425 vs 0.183) and 5.8x higher activation-pattern similarity (0.762 vs 0.130), despite lower geometric compactness. Entity-swap tests reveal a layerwise hierarchy: early-layer features transfer robustly (64% transfer rate, mean layer 6.3), while late-layer Say-X features specialize for output promotion (mean layer 16.4), supporting a backbone-and-specialization view of transformer computation. We release code (https://github.com/peppinob-ol/attribution-graph-probing), an interactive demo (https://huggingface.co/spaces/Peppinob/attribution-graph-probing), and minimal artifacts enabling immediate reproduction and community adoption.
>
---
#### [new 020] Retriv at BLP-2025 Task 2: Test-Driven Feedback-Guided Framework for Bangla-to-Python Code Generation
- **分类: cs.CL**

- **简介: 该论文面向孟加拉语到Python代码生成任务，解决低资源语言数据匮乏问题，提出基于Qwen2.5-14B的测试驱动反馈迭代框架，通过三轮测试反馈优化生成结果，实现Pass@1达0.934，获共享任务第二名。**

- **链接: [http://arxiv.org/pdf/2511.07382v1](http://arxiv.org/pdf/2511.07382v1)**

> **作者:** K M Nafi Asib; Sourav Saha; Mohammed Moshiul Hoque
>
> **备注:** 8 pages, 1 figure, experimental scripts publicly available at https://github.com/NafiAsib/Retriv-BLP25-Task-2
>
> **摘要:** Large Language Models (LLMs) have advanced the automated generation of code from natural language prompts. However, low-resource languages (LRLs) like Bangla remain underrepresented due to the limited availability of instruction-to-code datasets and evaluation benchmarks. To address this, the BLP Workshop at IJCNLP-AACL 2025 introduced a shared task on "Code Generation in Bangla". In this work, we propose a method that combines instruction prompting with a test-driven, feedback-guided iterative refinement process using a fine-tuned Qwen2.5-14B model. The model generates code from Bangla instructions, tests it against unit tests, and iteratively refines any failing outputs through three evaluation passes, using test feedback to guide each step. This approach helped our team "Retriv" to secure 2nd place in the shared task with a Pass@1 score of 0.934. The analysis highlights challenges in Bangla instruction understanding and Python code generation, emphasizing the need for targeted methods in LRLs. We made experimental scripts publicly available for the community.
>
---
#### [new 021] Analyzing and Mitigating Negation Artifacts using Data Augmentation for Improving ELECTRA-Small Model Accuracy
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言推理任务，旨在解决ELECTRA-small模型对否定句的误判问题。通过构造含否定的对比集与对抗样本进行数据增强，有效提升模型在否定场景下的准确率，缓解数据偏差。**

- **链接: [http://arxiv.org/pdf/2511.06234v1](http://arxiv.org/pdf/2511.06234v1)**

> **作者:** Mojtaba Noghabaei
>
> **摘要:** Pre-trained models for natural language inference (NLI) often achieve high performance on benchmark datasets by using spurious correlations, or dataset artifacts, rather than understanding language touches such as negation. In this project, we investigate the performance of an ELECTRA-small model fine-tuned on the Stanford Natural Language Inference (SNLI) dataset, focusing on its handling of negation. Through analysis, we identify that the model struggles with correctly classifying examples containing negation. To address this, we augment the training data with contrast sets and adversarial examples emphasizing negation. Our results demonstrate that this targeted data augmentation improves the model's accuracy on negation-containing examples without adversely affecting overall performance, therefore mitigating the identified dataset artifact.
>
---
#### [new 022] MuonAll: Muon Variant for Efficient Finetuning of Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 论文提出MuonAll，一种改进的优化器，将所有参数纳入Muon框架，解决其在微调阶段应用受限的问题。实验表明，MuonAll在半十亿参数模型上性能与AdamW相当，开源了分布式实现。**

- **链接: [http://arxiv.org/pdf/2511.06086v1](http://arxiv.org/pdf/2511.06086v1)**

> **作者:** Saurabh Page; Advait Joshi; S. S. Sonawane
>
> **摘要:** Muon optimizer has demonstrated robust results in pretraining of language models but its performance in finetuning of existing public pretrained models is not yet explored. Currently, Muon is used along with AdamW introducing a scope of improvement for adopting all parameters inside Muon. We introduce MuonAll, which incorporates all the parameters inside Muon by transforming into 2D matrices. We conduct extensive finetuning experiments across publicly available language models with model sizes upto half billion parameters. Muon and MuonAll perform at par with AdamW across major benchmarks, highlighting their effectiveness as alternative optimizers. We open-source the distributed implementations of Muon and MuonAll, available at https://github.com/Saurabh750/optimizer
>
---
#### [new 023] ReMoD: Rethinking Modality Contribution in Multimodal Stance Detection via Dual Reasoning
- **分类: cs.CL; cs.MM**

- **简介: 该论文针对多模态立场检测中模态贡献不均问题，受人类双过程认知启发，提出ReMoD框架，通过直觉与反思双推理机制动态加权模态贡献，提升立场判断准确性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.06057v1](http://arxiv.org/pdf/2511.06057v1)**

> **作者:** Bingbing Wang; Zhengda Jin; Bin Liang; Jing Li; Ruifeng Xu
>
> **摘要:** Multimodal Stance Detection (MSD) is a crucial task for understanding public opinion on social media. Existing work simply fuses information from various modalities to learn stance representations, overlooking the varying contributions of stance expression from different modalities. Therefore, stance misunderstanding noises may be drawn into the stance learning process due to the risk of learning errors by rough modality combination. To address this, we get inspiration from the dual-process theory of human cognition and propose **ReMoD**, a framework that **Re**thinks **Mo**dality contribution of stance expression through a **D**ual-reasoning paradigm. ReMoD integrates *experience-driven intuitive reasoning* to capture initial stance cues with *deliberate reflective reasoning* to adjust for modality biases, refine stance judgments, and thereby dynamically weight modality contributions based on their actual expressive power for the target stance. Specifically, the intuitive stage queries the Modality Experience Pool (MEP) and Semantic Experience Pool (SEP) to form an initial stance hypothesis, prioritizing historically impactful modalities. This hypothesis is then refined in the reflective stage via two reasoning chains: Modality-CoT updates MEP with adaptive fusion strategies to amplify relevant modalities, while Semantic-CoT refines SEP with deeper contextual insights of stance semantics. These dual experience structures are continuously refined during training and recalled at inference to guide robust and context-aware stance decisions. Extensive experiments on the public MMSD benchmark demonstrate that our ReMoD significantly outperforms most baseline models and exhibits strong generalization capabilities.
>
---
#### [new 024] Multilingual Lexical Feature Analysis of Spoken Language for Predicting Major Depression Symptom Severity
- **分类: cs.CL; cs.LG**

- **简介: 该论文探究多语言口语词汇特征预测重度抑郁症状严重程度，使用线性混合模型分析英、荷、西三语纵向语音数据，发现英语和荷兰语中部分词汇特征与症状相关，但预测性能接近随机，亟需更大规模、多语言的改进研究。**

- **链接: [http://arxiv.org/pdf/2511.07011v1](http://arxiv.org/pdf/2511.07011v1)**

> **作者:** Anastasiia Tokareva; Judith Dineley; Zoe Firth; Pauline Conde; Faith Matcham; Sara Siddi; Femke Lamers; Ewan Carr; Carolin Oetzmann; Daniel Leightley; Yuezhou Zhang; Amos A. Folarin; Josep Maria Haro; Brenda W. J. H. Penninx; Raquel Bailon; Srinivasan Vairavan; Til Wykes; Richard J. B. Dobson; Vaibhav A. Narayan; Matthew Hotopf; Nicholas Cummins; The RADAR-CNS Consortium
>
> **摘要:** Background: Captured between clinical appointments using mobile devices, spoken language has potential for objective, more regular assessment of symptom severity and earlier detection of relapse in major depressive disorder. However, research to date has largely been in non-clinical cross-sectional samples of written language using complex machine learning (ML) approaches with limited interpretability. Methods: We describe an initial exploratory analysis of longitudinal speech data and PHQ-8 assessments from 5,836 recordings of 586 participants in the UK, Netherlands, and Spain, collected in the RADAR-MDD study. We sought to identify interpretable lexical features associated with MDD symptom severity with linear mixed-effects modelling. Interpretable features and high-dimensional vector embeddings were also used to test the prediction performance of four regressor ML models. Results: In English data, MDD symptom severity was associated with 7 features including lexical diversity measures and absolutist language. In Dutch, associations were observed with words per sentence and positive word frequency; no associations were observed in recordings collected in Spain. The predictive power of lexical features and vector embeddings was near chance level across all languages. Limitations: Smaller samples in non-English speech and methodological choices, such as the elicitation prompt, may have also limited the effect sizes observable. A lack of NLP tools in languages other than English restricted our feature choice. Conclusion: To understand the value of lexical markers in clinical research and practice, further research is needed in larger samples across several languages using improved protocols, and ML models that account for within- and between-individual variations in language.
>
---
#### [new 025] A Picture is Worth a Thousand (Correct) Captions: A Vision-Guided Judge-Corrector System for Multimodal Machine Translation
- **分类: cs.CL; cs.CV; cs.HC**

- **简介: 该论文面向英-印地语系多模态翻译任务，提出视觉引导的判断-修正系统，自动检测并修正训练数据中的翻译错误，结合LoRA微调模型，显著提升翻译质量。**

- **链接: [http://arxiv.org/pdf/2511.07010v1](http://arxiv.org/pdf/2511.07010v1)**

> **作者:** Siddharth Betala; Kushan Raj; Vipul Betala; Rohan Saswade
>
> **备注:** Accepted at The 12th Workshop on Asian Translation, co-located with IJCLNLP-AACL 2025
>
> **摘要:** In this paper, we describe our system under the team name BLEU Monday for the English-to-Indic Multimodal Translation Task at WAT 2025. We participate in the text-only translation tasks for English-Hindi, English-Bengali, English-Malayalam, and English-Odia language pairs. We present a two-stage approach that addresses quality issues in the training data through automated error detection and correction, followed by parameter-efficient model fine-tuning. Our methodology introduces a vision-augmented judge-corrector pipeline that leverages multimodal language models to systematically identify and correct translation errors in the training data. The judge component classifies translations into three categories: correct, visually ambiguous (requiring image context), or mistranslated (poor translation quality). Identified errors are routed to specialized correctors: GPT-4o-mini regenerates captions requiring visual disambiguation, while IndicTrans2 retranslates cases with pure translation quality issues. This automated pipeline processes 28,928 training examples across four languages, correcting an average of 17.1% of captions per language. We then apply Low-Rank Adaptation (LoRA) to fine-tune the IndicTrans2 en-indic 200M distilled model on both original and corrected datasets. Training on corrected data yields consistent improvements, with BLEU score gains of +1.30 for English-Bengali on the evaluation set (42.00 -> 43.30) and +0.70 on the challenge set (44.90 -> 45.60), +0.60 for English-Odia on the evaluation set (41.00 -> 41.60), and +0.10 for English-Hindi on the challenge set (53.90 -> 54.00).
>
---
#### [new 026] LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出LoGo，一种无需训练的实例级LoRA动态选择与融合方法，解决多任务场景下传统LoRA适应性差、依赖标注数据的问题，通过单次前向传播自动识别最优适配器组合，在多个NLP任务上超越训练基线。**

- **链接: [http://arxiv.org/pdf/2511.07129v1](http://arxiv.org/pdf/2511.07129v1)**

> **作者:** Seungeon Lee; Soumi Das; Manish Gupta; Krishna P. Gummadi
>
> **摘要:** Low-Rank Adaptation (LoRA) has emerged as a parameter-efficient approach for fine-tuning large language models.However, conventional LoRA adapters are typically trained for a single task, limiting their applicability in real-world settings where inputs may span diverse and unpredictable domains. At inference time, existing approaches combine multiple LoRAs for improving performance on diverse tasks, while usually requiring labeled data or additional task-specific training, which is expensive at scale. In this work, we introduce LoRA on the Go (LoGo), a training-free framework that dynamically selects and merges adapters at the instance level without any additional requirements. LoGo leverages signals extracted from a single forward pass through LoRA adapters, to identify the most relevant adapters and determine their contributions on-the-fly. Across 5 NLP benchmarks, 27 datasets, and 3 model families, LoGo outperforms training-based baselines on some tasks upto a margin of 3.6% while remaining competitive on other tasks and maintaining inference throughput, highlighting its effectiveness and practicality.
>
---
#### [new 027] Efficient Hate Speech Detection: A Three-Layer LoRA-Tuned BERTweet Framework
- **分类: cs.CL**

- **简介: 该论文针对高效仇恨言论检测任务，提出一种三层LoRA微调BERTweet框架，通过参数高效微调与规则预过滤，在仅1.87M可训练参数下实现接近SOTA模型94%的性能，显著降低计算成本，适用于资源受限环境的实时部署。**

- **链接: [http://arxiv.org/pdf/2511.06051v1](http://arxiv.org/pdf/2511.06051v1)**

> **作者:** Mahmoud El-Bahnasawi
>
> **备注:** 13 pages, 2 figures
>
> **摘要:** This paper addresses the critical challenge of developing computationally efficient hate speech detection systems that maintain competitive performance while being practical for real-time deployment. We propose a novel three-layer framework that combines rule-based pre-filtering with a parameter-efficient LoRA-tuned BERTweet model and continuous learning capabilities. Our approach achieves 0.85 macro F1 score - representing 94% of the performance of state-of-the-art large language models like SafePhi (Phi-4 based) while using a base model that is 100x smaller (134M vs 14B parameters). Compared to traditional BERT-based approaches with similar computational requirements, our method demonstrates superior performance through strategic dataset unification and optimized fine-tuning. The system requires only 1.87M trainable parameters (1.37% of full fine-tuning) and trains in approximately 2 hours on a single T4 GPU, making robust hate speech detection accessible in resource-constrained environments while maintaining competitive accuracy for real-world deployment.
>
---
#### [new 028] SCOPE: Intrinsic Semantic Space Control for Mitigating Copyright Infringement in LLMs
- **分类: cs.CL**

- **简介: 该论文提出SCOPE，一种无需参数更新的推理时方法，通过稀疏自编码器识别并抑制隐藏空间中的版权敏感子空间，以缓解大语言模型的语义抄袭问题，兼顾侵权防控与模型效用。**

- **链接: [http://arxiv.org/pdf/2511.07001v1](http://arxiv.org/pdf/2511.07001v1)**

> **作者:** Zhenliang Zhang; Xinyu Hu; Xiaojun Wan
>
> **备注:** Accepted by the AAAI 2026 (Main Track)
>
> **摘要:** Large language models sometimes inadvertently reproduce passages that are copyrighted, exposing downstream applications to legal risk. Most existing studies for inference-time defences focus on surface-level token matching and rely on external blocklists or filters, which add deployment complexity and may overlook semantically paraphrased leakage. In this work, we reframe copyright infringement mitigation as intrinsic semantic-space control and introduce SCOPE, an inference-time method that requires no parameter updates or auxiliary filters. Specifically, the sparse autoencoder (SAE) projects hidden states into a high-dimensional, near-monosemantic space; benefiting from this representation, we identify a copyright-sensitive subspace and clamp its activations during decoding. Experiments on widely recognized benchmarks show that SCOPE mitigates copyright infringement without degrading general utility. Further interpretability analyses confirm that the isolated subspace captures high-level semantics.
>
---
#### [new 029] BookAsSumQA: An Evaluation Framework for Aspect-Based Book Summarization via Question Answering
- **分类: cs.CL**

- **简介: 该论文提出BookAsSumQA框架，用于评估长篇书籍的方面级摘要质量。通过从叙事知识图谱自动生成QA对，解决人工参考摘要构建难的问题，发现RAG方法在长文本中更优。**

- **链接: [http://arxiv.org/pdf/2511.06183v1](http://arxiv.org/pdf/2511.06183v1)**

> **作者:** Ryuhei Miyazato; Ting-Ruen Wei; Xuyang Wu; Hsin-Tai Wu; Kei Harada
>
> **摘要:** Aspect-based summarization aims to generate summaries that highlight specific aspects of a text, enabling more personalized and targeted summaries. However, its application to books remains unexplored due to the difficulty of constructing reference summaries for long text. To address this challenge, we propose BookAsSumQA, a QA-based evaluation framework for aspect-based book summarization. BookAsSumQA automatically generates aspect-specific QA pairs from a narrative knowledge graph to evaluate summary quality based on its question-answering performance. Our experiments using BookAsSumQA revealed that while LLM-based approaches showed higher accuracy on shorter texts, RAG-based methods become more effective as document length increases, making them more efficient and practical for aspect-based book summarization.
>
---
#### [new 030] When Sufficient is not Enough: Utilizing the Rashomon Effect for Complete Evidence Extraction
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文研究特征归因任务，解决单一模型仅提取部分证据的问题，提出通过模型集成提升完整证据召回率，实验证明集成方法将召回率从0.60提升至0.86。**

- **链接: [http://arxiv.org/pdf/2511.07055v1](http://arxiv.org/pdf/2511.07055v1)**

> **作者:** Katharina Beckh; Stefan Rüping
>
> **摘要:** Feature attribution methods typically provide minimal sufficient evidence justifying a model decision. However, in many applications this is inadequate. For compliance and cataloging, the full set of contributing features must be identified - complete evidence. We perform a case study on a medical dataset which contains human-annotated complete evidence. We show that individual models typically recover only subsets of complete evidence and that aggregating evidence from several models improves evidence recall from $\sim$0.60 (single best model) to $\sim$0.86 (ensemble). We analyze the recall-precision trade-off, the role of training with evidence, dynamic ensembles with certainty thresholds, and discuss implications.
>
---
#### [new 031] In-Context Learning Without Copying
- **分类: cs.CL**

- **简介: 该论文研究Transformer在抑制归纳复制后能否仍具备上下文学习能力。提出Hapax方法，剔除可被归纳头预测的token损失，发现模型仍能有效学习抽象型上下文学习任务，证明归纳复制非其必要前提。**

- **链接: [http://arxiv.org/pdf/2511.05743v1](http://arxiv.org/pdf/2511.05743v1)**

> **作者:** Kerem Sahin; Sheridan Feucht; Adam Belfki; Jannik Brinkmann; Aaron Mueller; David Bau; Chris Wendler
>
> **摘要:** Induction heads are attention heads that perform inductive copying by matching patterns from earlier context and copying their continuations verbatim. As models develop induction heads, they often experience a sharp drop in training loss, a phenomenon cited as evidence that induction heads may serve as a prerequisite for more complex in-context learning (ICL) capabilities. In this work, we ask whether transformers can still acquire ICL capabilities when inductive copying is suppressed. We propose Hapax, a setting where we omit the loss contribution of any token that can be correctly predicted by induction heads. Despite a significant reduction in inductive copying, performance on abstractive ICL tasks (i.e., tasks where the answer is not contained in the input context) remains comparable and surpasses the vanilla model on 13 of 21 tasks, even though 31.7\% of tokens are omitted from the loss. Furthermore, our model achieves lower loss values on token positions that cannot be predicted correctly by induction heads. Mechanistic analysis further shows that models trained with Hapax develop fewer and weaker induction heads but still preserve ICL capabilities. Taken together, our findings indicate that inductive copying is not essential for learning abstractive ICL mechanisms.
>
---
#### [new 032] Retriv at BLP-2025 Task 1: A Transformer Ensemble and Multi-Task Learning Approach for Bangla Hate Speech Identification
- **分类: cs.CL**

- **简介: 该论文针对孟加拉语仇恨言论识别任务，提出基于Transformer集成与多任务学习的方法，分别解决仇恨类型、目标群体及联合检测三子任务，实现F1分数超72%，在共享任务中位列前10。**

- **链接: [http://arxiv.org/pdf/2511.07304v1](http://arxiv.org/pdf/2511.07304v1)**

> **作者:** Sourav Saha; K M Nafi Asib; Mohammed Moshiul Hoque
>
> **备注:** 7 pages, 3 figures, experimental scripts publicly available at https://github.com/sahasourav17/Retriv-BLP25-Task-1
>
> **摘要:** This paper addresses the problem of Bangla hate speech identification, a socially impactful yet linguistically challenging task. As part of the "Bangla Multi-task Hate Speech Identification" shared task at the BLP Workshop, IJCNLP-AACL 2025, our team "Retriv" participated in all three subtasks: (1A) hate type classification, (1B) target group identification, and (1C) joint detection of type, severity, and target. For subtasks 1A and 1B, we employed a soft-voting ensemble of transformer models (BanglaBERT, MuRIL, IndicBERTv2). For subtask 1C, we trained three multitask variants and aggregated their predictions through a weighted voting ensemble. Our systems achieved micro-f1 scores of 72.75% (1A) and 72.69% (1B), and a weighted micro-f1 score of 72.62% (1C). On the shared task leaderboard, these corresponded to 9th, 10th, and 7th positions, respectively. These results highlight the promise of transformer ensembles and weighted multitask frameworks for advancing Bangla hate speech detection in low-resource contexts. We made experimental scripts publicly available for the community.
>
---
#### [new 033] Steering LLMs toward Korean Local Speech: Iterative Refinement Framework for Faithful Dialect Translation
- **分类: cs.CL**

- **简介: 该论文面向韩语方言翻译任务，解决大模型方言生成失真与评估指标误导问题，提出DIA-REFINE迭代框架，结合新指标DFS与TDR，提升方言翻译真实性与评估准确性。**

- **链接: [http://arxiv.org/pdf/2511.06680v1](http://arxiv.org/pdf/2511.06680v1)**

> **作者:** Keunhyeung Park; Seunguk Yu; Youngbin Kim
>
> **备注:** Submitted to LREC 2026
>
> **摘要:** Standard-to-dialect machine translation remains challenging due to a persistent dialect gap in large language models and evaluation distortions inherent in n-gram metrics, which favor source copying over authentic dialect translation. In this paper, we propose the dialect refinement (DIA-REFINE) framework, which guides LLMs toward faithful target dialect outputs through an iterative loop of translation, verification, and feedback using external dialect classifiers. To address the limitations of n-gram-based metrics, we introduce the dialect fidelity score (DFS) to quantify linguistic shift and the target dialect ratio (TDR) to measure the success of dialect translation. Experiments on Korean dialects across zero-shot and in-context learning baselines demonstrate that DIA-REFINE consistently enhances dialect fidelity. The proposed metrics distinguish between False Success cases, where high n-gram scores obscure failures in dialectal translation, and True Attempt cases, where genuine attempts at dialectal translation yield low n-gram scores. We also observed that models exhibit varying degrees of responsiveness to the framework, and that integrating in-context examples further improves the translation of dialectal expressions. Our work establishes a robust framework for goal-directed, inclusive dialect translation, providing both rigorous evaluation and critical insights into model performance.
>
---
#### [new 034] Evaluation of retrieval-based QA on QUEST-LOFT
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文针对QUEST-LOFT基准中检索增强问答（RAG）性能不佳的问题，分析其成因，通过人类评估更新数据，并提出结合结构化推理与证据输出及答案复核的优化方法，显著超越长上下文模型。**

- **链接: [http://arxiv.org/pdf/2511.06125v1](http://arxiv.org/pdf/2511.06125v1)**

> **作者:** Nathan Scales; Nathanael Schärli; Olivier Bousquet
>
> **摘要:** Despite the popularity of retrieval-augmented generation (RAG) as a solution for grounded QA in both academia and industry, current RAG methods struggle with questions where the necessary information is distributed across many documents or where retrieval needs to be combined with complex reasoning. Recently, the LOFT study has shown that this limitation also applies to approaches based on long-context language models, with the QUEST benchmark exhibiting particularly large headroom. In this paper, we provide an in-depth analysis of the factors contributing to the poor performance on QUEST-LOFT, publish updated numbers based on a thorough human evaluation, and demonstrate that RAG can be optimized to significantly outperform long-context approaches when combined with a structured output format containing reasoning and evidence, optionally followed by answer re-verification.
>
---
#### [new 035] Interpretable Recognition of Cognitive Distortions in Natural Language Texts
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文提出一种可解释的AI模型，用于自动识别自然语言文本中的认知扭曲，基于加权N-gram结构模式，提升检测准确率，并在公开数据集上验证效果，代码与模型已开源。**

- **链接: [http://arxiv.org/pdf/2511.05969v1](http://arxiv.org/pdf/2511.05969v1)**

> **作者:** Anton Kolonin; Anna Arinicheva
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** We propose a new approach to multi-factor classification of natural language texts based on weighted structured patterns such as N-grams, taking into account the heterarchical relationships between them, applied to solve such a socially impactful problem as the automation of detection of specific cognitive distortions in psychological care, relying on an interpretable, robust and transparent artificial intelligence model. The proposed recognition and learning algorithms improve the current state of the art in this field. The improvement is tested on two publicly available datasets, with significant improvements over literature-known F1 scores for the task, with optimal hyper-parameters determined, having code and models available for future use by the community.
>
---
#### [new 036] Rep2Text: Decoding Full Text from a Single LLM Token Representation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出Rep2Text，旨在从LLM单个末尾token表征中重建原始输入文本，解决模型内部表示可逆性问题。通过可训练适配器映射至解码模型嵌入空间，实现高语义保真度的文本恢复，平均可重建超半数信息。**

- **链接: [http://arxiv.org/pdf/2511.06571v1](http://arxiv.org/pdf/2511.06571v1)**

> **作者:** Haiyan Zhao; Zirui He; Fan Yang; Ali Payani; Mengnan Du
>
> **备注:** 15 pages, 7 figures, 4 tables
>
> **摘要:** Large language models (LLMs) have achieved remarkable progress across diverse tasks, yet their internal mechanisms remain largely opaque. In this work, we address a fundamental question: to what extent can the original input text be recovered from a single last-token representation within an LLM? We propose Rep2Text, a novel framework for decoding full text from last-token representations. Rep2Text employs a trainable adapter that projects a target model's internal representations into the embedding space of a decoding language model, which then autoregressively reconstructs the input text. Experiments on various model combinations (Llama-3.1-8B, Gemma-7B, Mistral-7B-v0.1, Llama-3.2-3B) demonstrate that, on average, over half of the information in 16-token sequences can be recovered from this compressed representation while maintaining strong semantic integrity and coherence. Furthermore, our analysis reveals an information bottleneck effect: longer sequences exhibit decreased token-level recovery while preserving strong semantic integrity. Besides, our framework also demonstrates robust generalization to out-of-distribution medical data.
>
---
#### [new 037] Evaluating LLMs for Anxiety, Depression, and Stress Detection Evaluating Large Language Models for Anxiety, Depression, and Stress Detection: Insights into Prompting Strategies and Synthetic Data
- **分类: cs.CL**

- **简介: 该论文研究利用大语言模型和Transformer架构从文本中自动检测焦虑、抑郁与压力，解决症状表达隐蔽导致的检测难题。通过微调与合成数据增强，对比多种模型性能，发现Distil-RoBERTa和XLNet表现最优，合成数据显著提升召回率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.07044v1](http://arxiv.org/pdf/2511.07044v1)**

> **作者:** Mihael Arcan; David-Paul Niland
>
> **摘要:** Mental health disorders affect over one-fifth of adults globally, yet detecting such conditions from text remains challenging due to the subtle and varied nature of symptom expression. This study evaluates multiple approaches for mental health detection, comparing Large Language Models (LLMs) such as Llama and GPT with classical machine learning and transformer-based architectures including BERT, XLNet, and Distil-RoBERTa. Using the DAIC-WOZ dataset of clinical interviews, we fine-tuned models for anxiety, depression, and stress classification and applied synthetic data generation to mitigate class imbalance. Results show that Distil-RoBERTa achieved the highest F1 score (0.883) for GAD-2, while XLNet outperformed others on PHQ tasks (F1 up to 0.891). For stress detection, a zero-shot synthetic approach (SD+Zero-Shot-Basic) reached an F1 of 0.884 and ROC AUC of 0.886. Findings demonstrate the effectiveness of transformer-based models and highlight the value of synthetic data in improving recall and generalization. However, careful calibration is required to prevent precision loss. Overall, this work emphasizes the potential of combining advanced language models and data augmentation to enhance automated mental health assessment from text.
>
---
#### [new 038] IDALC: A Semi-Supervised Framework for Intent Detection and Active Learning based Correction
- **分类: cs.CL; cs.AI**

- **简介: IDALC提出一种半监督框架，用于语音对话系统中的意图检测与拒绝样本修正，通过主动学习显著降低人工标注成本，提升模型准确率与F1值，仅用6-10%标注数据即可超越基线方法。**

- **链接: [http://arxiv.org/pdf/2511.05921v1](http://arxiv.org/pdf/2511.05921v1)**

> **作者:** Ankan Mullick; Sukannya Purkayastha; Saransh Sharma; Pawan Goyal; Niloy Ganguly
>
> **备注:** Paper accepted in IEEE Transactions on Artificial Intelligence (October 2025)
>
> **摘要:** Voice-controlled dialog systems have become immensely popular due to their ability to perform a wide range of actions in response to diverse user queries. These agents possess a predefined set of skills or intents to fulfill specific user tasks. But every system has its own limitations. There are instances where, even for known intents, if any model exhibits low confidence, it results in rejection of utterances that necessitate manual annotation. Additionally, as time progresses, there may be a need to retrain these agents with new intents from the system-rejected queries to carry out additional tasks. Labeling all these emerging intents and rejected utterances over time is impractical, thus calling for an efficient mechanism to reduce annotation costs. In this paper, we introduce IDALC (Intent Detection and Active Learning based Correction), a semi-supervised framework designed to detect user intents and rectify system-rejected utterances while minimizing the need for human annotation. Empirical findings on various benchmark datasets demonstrate that our system surpasses baseline methods, achieving a 5-10% higher accuracy and a 4-8% improvement in macro-F1. Remarkably, we maintain the overall annotation cost at just 6-10% of the unlabelled data available to the system. The overall framework of IDALC is shown in Fig. 1
>
---
#### [new 039] Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文将预训练的非循环语言模型转化为深度循环模型，通过渐进式循环课程提升推理深度，以更低计算成本提高性能，解决预训练模型推理效率与能力不足的问题。**

- **链接: [http://arxiv.org/pdf/2511.07384v1](http://arxiv.org/pdf/2511.07384v1)**

> **作者:** Sean McLeish; Ang Li; John Kirchenbauer; Dayal Singh Kalra; Brian R. Bartoldson; Bhavya Kailkhura; Avi Schwarzschild; Jonas Geiping; Tom Goldstein; Micah Goldblum
>
> **备注:** code: https://github.com/mcleish7/retrofitting-recurrence, models: https://huggingface.co/collections/tomg-group-umd/retrofitting-recurrence
>
> **摘要:** Recent advances in depth-recurrent language models show that recurrence can decouple train-time compute and parameter count from test-time compute. In this work, we study how to convert existing pretrained non-recurrent language models into depth-recurrent models. We find that using a curriculum of recurrences to increase the effective depth of the model over the course of training preserves performance while reducing total computational cost. In our experiments, on mathematics, we observe that converting pretrained models to recurrent ones results in better performance at a given compute budget than simply post-training the original non-recurrent language model.
>
---
#### [new 040] More Agents Helps but Adversarial Robustness Gap Persists
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多智能体协作在数学问答中的对抗鲁棒性，发现增加智能体数量可提升准确率，但对抗扰动（尤其是人工拼写错误）仍导致显著性能差距，鲁棒性瓶颈未被解决。**

- **链接: [http://arxiv.org/pdf/2511.07112v1](http://arxiv.org/pdf/2511.07112v1)**

> **作者:** Khashayar Alavi; Zhastay Yeltay; Lucie Flek; Akbar Karimi
>
> **摘要:** When LLM agents work together, they seem to be more powerful than a single LLM in mathematical question answering. However, are they also more robust to adversarial inputs? We investigate this question using adversarially perturbed math questions. These perturbations include punctuation noise with three intensities (10, 30, and 50 percent), plus real-world and human-like typos (WikiTypo, R2ATA). Using a unified sampling-and-voting framework (Agent Forest), we evaluate six open-source models (Qwen3-4B/14B, Llama3.1-8B, Mistral-7B, Gemma3-4B/12B) across four benchmarks (GSM8K, MATH, MMLU-Math, MultiArith), with various numbers of agents n from one to 25 (1, 2, 5, 10, 15, 20, 25). Our findings show that (1) Noise type matters: punctuation noise harm scales with its severity, and the human typos remain the dominant bottleneck, yielding the largest gaps to Clean accuracy and the highest ASR even with a large number of agents. And (2) Collaboration reliably improves accuracy as the number of agents, n, increases, with the largest gains from one to five agents and diminishing returns beyond 10 agents. However, the adversarial robustness gap persists regardless of the agent count.
>
---
#### [new 041] Aligning Attention with Human Rationales for Self-Explaining Hate Speech Detection
- **分类: cs.CL; cs.LG**

- **简介: 该论文面向仇恨言论检测任务，旨在提升模型的可解释性与公平性。提出监督理性注意力（SRA）框架，将人类标注的推理依据融入注意力机制，显著提升解释忠实度，同时保持公平性。**

- **链接: [http://arxiv.org/pdf/2511.07065v1](http://arxiv.org/pdf/2511.07065v1)**

> **作者:** Brage Eilertsen; Røskva Bjørgfinsdóttir; Francielle Vargas; Ali Ramezani-Kebrya
>
> **备注:** Accepted at the Annual AAAI Conference on Artificial Intelligence (AAAI26)
>
> **摘要:** The opaque nature of deep learning models presents significant challenges for the ethical deployment of hate speech detection systems. To address this limitation, we introduce Supervised Rational Attention (SRA), a framework that explicitly aligns model attention with human rationales, improving both interpretability and fairness in hate speech classification. SRA integrates a supervised attention mechanism into transformer-based classifiers, optimizing a joint objective that combines standard classification loss with an alignment loss term that minimizes the discrepancy between attention weights and human-annotated rationales. We evaluated SRA on hate speech benchmarks in English (HateXplain) and Portuguese (HateBRXplain) with rationale annotations. Empirically, SRA achieves 2.4x better explainability compared to current baselines, and produces token-level explanations that are more faithful and human-aligned. In terms of fairness, SRA achieves competitive fairness across all measures, with second-best performance in detecting toxic posts targeting identity groups, while maintaining comparable results on other metrics. These findings demonstrate that incorporating human rationales into attention mechanisms can enhance interpretability and faithfulness without compromising fairness.
>
---
#### [new 042] Inclusion of Role into Named Entity Recognition and Ranking
- **分类: cs.CL; cs.LG**

- **简介: 该论文将实体角色检测建模为命名实体识别与实体检索/排序任务，通过学习上下文中的代表性词句，构建角色与实体的表示，解决角色依赖上下文且标注数据稀缺的问题，实现领域无关的高效角色识别与检索。**

- **链接: [http://arxiv.org/pdf/2511.06886v1](http://arxiv.org/pdf/2511.06886v1)**

> **作者:** Neelesh Kumar Shukla; Sanasam Ranbir Singh
>
> **备注:** MTP Paper
>
> **摘要:** Most of the Natural Language Processing sys- tems are involved in entity-based processing for several tasks like Information Extraction, Question-Answering, Text-Summarization and so on. A new challenge comes when entities play roles according to their act or attributes in certain context. Entity Role Detection is the task of assigning such roles to the entities. Usu- ally real-world entities are of types: person, lo- cation and organization etc. Roles could be con- sidered as domain-dependent subtypes of these types. In the cases, where retrieving a subset of entities based on their roles is needed, poses the problem of defining the role and entities having those roles. This paper presents the study of study of solving Entity Role Detection prob- lem by modeling it as Named Entity Recogni- tion (NER) and Entity Retrieval/Ranking task. In NER, these roles could be considered as mutually exclusive classes and standard NER methods like sequence tagging could be used. For Entity Retrieval, Roles could be formulated as Query and entities as Collection on which the query needs to be executed. The aspect of Entity Retrieval task, which is different than document retrieval task is that the entities and roles against which they need to be retrieved are indirectly described. We have formulated au- tomated ways of learning representative words and phrases and building representations of roles and entities using them. We have also explored different contexts like sentence and document. Since the roles depend upon con- text, so it is not always possible to have large domain-specific dataset or knowledge bases for learning purposes, so we have tried to exploit the information from small dataset in domain- agnostic way.
>
---
#### [new 043] SAFENLIDB: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces
- **分类: cs.CL**

- **简介: 论文提出SafeNlidb框架，解决LLM驱动的自然语言数据库接口中的隐私泄露与安全攻击问题，通过自动生成混合推理数据与优化算法，实现无需人工标注的安全对齐，提升SQL生成的安全性与可靠性。**

- **链接: [http://arxiv.org/pdf/2511.06778v1](http://arxiv.org/pdf/2511.06778v1)**

> **作者:** Ruiheng Liu; XiaoBing Chen; Jinyu Zhang; Qiongwen Zhang; Yu Zhang; Bailong Yang
>
> **备注:** 26 pages, 14 figures, 22 tables
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has driven significant progress in Natural Language Interface to Database (NLIDB). However, the widespread adoption of LLMs has raised critical privacy and security concerns. During interactions, LLMs may unintentionally expose confidential database contents or be manipulated by attackers to exfiltrate data through seemingly benign queries. While current efforts typically rely on rule-based heuristics or LLM agents to mitigate this leakage risk, these methods still struggle with complex inference-based attacks, suffer from high false positive rates, and often compromise the reliability of SQL queries. To address these challenges, we propose \textsc{SafeNlidb}, a novel privacy-security alignment framework for LLM-based NLIDB. The framework features an automated pipeline that generates hybrid chain-of-thought interaction data from scratch, seamlessly combining implicit security reasoning with SQL generation. Additionally, we introduce reasoning warm-up and alternating preference optimization to overcome the multi-preference oscillations of Direct Preference Optimization (DPO), enabling LLMs to produce security-aware SQL through fine-grained reasoning without the need for human-annotated preference data. Extensive experiments demonstrate that our method outperforms both larger-scale LLMs and ideal-setting baselines, achieving significant security improvements while preserving high utility.WARNING: This work may contain content that is offensive and harmful!
>
---
#### [new 044] NILC: Discovering New Intents with LLM-assisted Clustering
- **分类: cs.CL; cs.AI**

- **简介: 论文提出NILC框架，用于新意图发现（NID）任务，解决传统聚类忽略语义细粒度的问题。通过LLM辅助迭代优化聚类中心与文本嵌入，并引入语义增强与半监督信号，显著提升多领域意图识别性能。**

- **链接: [http://arxiv.org/pdf/2511.05913v1](http://arxiv.org/pdf/2511.05913v1)**

> **作者:** Hongtao Wang; Renchi Yang; Wenqing Lin
>
> **摘要:** New intent discovery (NID) seeks to recognize both new and known intents from unlabeled user utterances, which finds prevalent use in practical dialogue systems. Existing works towards NID mainly adopt a cascaded architecture, wherein the first stage focuses on encoding the utterances into informative text embeddings beforehand, while the latter is to group similar embeddings into clusters (i.e., intents), typically by K-Means. However, such a cascaded pipeline fails to leverage the feedback from both steps for mutual refinement, and, meanwhile, the embedding-only clustering overlooks nuanced textual semantics, leading to suboptimal performance. To bridge this gap, this paper proposes NILC, a novel clustering framework specially catered for effective NID. Particularly, NILC follows an iterative workflow, in which clustering assignments are judiciously updated by carefully refining cluster centroids and text embeddings of uncertain utterances with the aid of large language models (LLMs). Specifically, NILC first taps into LLMs to create additional semantic centroids for clusters, thereby enriching the contextual semantics of the Euclidean centroids of embeddings. Moreover, LLMs are then harnessed to augment hard samples (ambiguous or terse utterances) identified from clusters via rewriting for subsequent cluster correction. Further, we inject supervision signals through non-trivial techniques seeding and soft must links for more accurate NID in the semi-supervised setting. Extensive experiments comparing NILC against multiple recent baselines under both unsupervised and semi-supervised settings showcase that NILC can achieve significant performance improvements over six benchmark datasets of diverse domains consistently.
>
---
#### [new 045] Beyond English: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs
- **分类: cs.CL**

- **简介: 该论文提出LMT模型，解决多语言机器翻译中的语言覆盖不均与英语中心偏差问题，通过定向采样与并行提示技术，在60种语言234个方向上实现SOTA性能，释放四尺寸模型促进公平可扩展翻译研究。**

- **链接: [http://arxiv.org/pdf/2511.07003v1](http://arxiv.org/pdf/2511.07003v1)**

> **作者:** Yingfeng Luo; Ziqiang Xu; Yuxuan Ouyang; Murun Yang; Dingyang Lin; Kaiyan Chang; Tong Zheng; Bei Li; Peinan Feng; Quan Du; Tong Xiao; Jingbo Zhu
>
> **摘要:** Large language models have significantly advanced Multilingual Machine Translation (MMT), yet the broad language coverage, consistent translation quality, and English-centric bias remain open challenges. To address these challenges, we introduce \textbf{LMT}, a suite of \textbf{L}arge-scale \textbf{M}ultilingual \textbf{T}ranslation models centered on both Chinese and English, covering 60 languages and 234 translation directions. During development, we identify a previously overlooked phenomenon of \textbf{directional degeneration}, where symmetric multi-way fine-tuning data overemphasize reverse directions (X $\to$ En/Zh), leading to excessive many-to-one mappings and degraded translation quality. We propose \textbf{Strategic Downsampling}, a simple yet effective method to mitigate this degeneration. In addition, we design \textbf{Parallel Multilingual Prompting (PMP)}, which leverages typologically related auxiliary languages to enhance cross-lingual transfer. Through rigorous data curation and refined adaptation strategies, LMT achieves SOTA performance among models of comparable language coverage, with our 4B model (LMT-60-4B) surpassing the much larger Aya-101-13B and NLLB-54B models by a substantial margin. We release LMT in four sizes (0.6B/1.7B/4B/8B) to catalyze future research and provide strong baselines for inclusive, scalable, and high-quality MMT \footnote{\href{https://github.com/NiuTrans/LMT}{https://github.com/NiuTrans/LMT}}.
>
---
#### [new 046] Towards Resource-Efficient Multimodal Intelligence: Learned Routing among Specialized Expert Models
- **分类: cs.CL; cs.LG; I.2.7; I.2.6; I.2.11**

- **简介: 该论文提出一种基于学习路由的多模态智能框架，将不同查询智能分配给专用专家模型，解决大模型推理成本高、小模型能力不足的问题，在MMLU和VQA上性能媲美单一大模型，但减少70%以上高成本模型依赖。**

- **链接: [http://arxiv.org/pdf/2511.06441v1](http://arxiv.org/pdf/2511.06441v1)**

> **作者:** Mayank Saini; Arit Kumar Bishwas
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** As AI moves beyond text, large language models (LLMs) increasingly power vision, audio, and document understanding; however, their high inference costs hinder real-time, scalable deployment. Conversely, smaller open-source models offer cost advantages but struggle with complex or multimodal queries. We introduce a unified, modular framework that intelligently routes each query - textual, multimodal, or complex - to the most fitting expert model, using a learned routing network that balances cost and quality. For vision tasks, we employ a two-stage open-source pipeline optimized for efficiency and reviving efficient classical vision components where they remain SOTA for sub-tasks. On benchmarks such as Massive Multitask Language Understanding (MMLU) and Visual Question Answering (VQA), we match or exceed the performance of always-premium LLM (monolithic systems with one model serving all query types) performance, yet reduce the reliance on costly models by over 67%. With its extensible, multi-agent orchestration, we deliver high-quality, resource-efficient AI at scale.
>
---
#### [new 047] Wasm: A Pipeline for Constructing Structured Arabic Interleaved Multimodal Corpora
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Wasm管道，从Common Crawl构建结构化阿拉伯语图文交错多模态数据集，解决阿拉伯语缺乏结构保留多模态语料的问题，首次输出Markdown格式，支持文本与多模态预训练。**

- **链接: [http://arxiv.org/pdf/2511.07080v1](http://arxiv.org/pdf/2511.07080v1)**

> **作者:** Khalil Hennara; Ahmad Bastati; Muhammad Hreden; Mohamed Motasim Hamed; Zeina Aldallal; Sara Chrouf; Safwan AlModhayan
>
> **摘要:** The performance of large language models (LLMs) and large multimodal models (LMMs) depends heavily on the quality and scale of their pre-training datasets. Recent research shows that large multimodal models trained on natural documents where images and text are interleaved outperform those trained only on image-text pairs across a wide range of benchmarks, leveraging advanced pre- trained models to enforce semantic alignment, image-sequence consistency, and textual coherence. For Arabic, however, the lack of high-quality multimodal datasets that preserve document structure has limited progress. In this paper, we present our pipeline Wasm for processing the Common Crawl dataset to create a new Arabic multimodal dataset that uniquely provides markdown output. Unlike existing Arabic corpora that focus solely on text extraction, our approach preserves the structural integrity of web content while maintaining flexibility for both text-only and multimodal pre-training scenarios. We provide a comprehensive comparative analysis of our data processing pipeline against those used for major existing datasets, highlighting the convergences in filtering strategies and justifying our specific design choices. To support future research, we publicly release a representative dataset dump along with the multimodal processing pipeline for Arabic.
>
---
#### [new 048] EmoBang: Detecting Emotion From Bengali Texts
- **分类: cs.CL**

- **简介: 该论文面向孟加拉语情感检测任务，解决低资源语言缺乏标注数据与高性能模型的问题，构建了首个八类情感标注数据集，并提出CRNN与BERT集成模型，实现超92%准确率，建立新基准。**

- **链接: [http://arxiv.org/pdf/2511.07077v1](http://arxiv.org/pdf/2511.07077v1)**

> **作者:** Abdullah Al Maruf; Aditi Golder; Zakaria Masud Jiyad; Abdullah Al Numan; Tarannum Shaila Zaman
>
> **摘要:** Emotion detection from text seeks to identify an individual's emotional or mental state - positive, negative, or neutral - based on linguistic cues. While significant progress has been made for English and other high-resource languages, Bengali remains underexplored despite being the world's fourth most spoken language. The lack of large, standardized datasets classifies Bengali as a low-resource language for emotion detection. Existing studies mainly employ classical machine learning models with traditional feature engineering, yielding limited performance. In this paper, we introduce a new Bengali emotion dataset annotated across eight emotion categories and propose two models for automatic emotion detection: (i) a hybrid Convolutional Recurrent Neural Network (CRNN) model (EmoBangHybrid) and (ii) an AdaBoost-Bidirectional Encoder Representations from Transformers (BERT) ensemble model (EmoBangEnsemble). Additionally, we evaluate six baseline models with five feature engineering techniques and assess zero-shot and few-shot large language models (LLMs) on the dataset. To the best of our knowledge, this is the first comprehensive benchmark for Bengali emotion detection. Experimental results show that EmoBangH and EmoBangE achieve accuracies of 92.86% and 93.69%, respectively, outperforming existing methods and establishing strong baselines for future research.
>
---
#### [new 049] Visual Exploration of Feature Relationships in Sparse Autoencoders with Curated Concepts
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对稀疏自编码器（SAE）特征过多难探索的问题，提出一种聚焦于预定义概念的交互式可视化框架，结合拓扑编码与降维技术，精准呈现特征间局部与全局关系，提升概念表征的可解释性分析。**

- **链接: [http://arxiv.org/pdf/2511.06048v1](http://arxiv.org/pdf/2511.06048v1)**

> **作者:** Xinyuan Yan; Shusen Liu; Kowshik Thopalli; Bei Wang
>
> **备注:** 8 pages (5 main paper+3 refernce), 2 figures, pulished at Mechanistic Interpretability Workshop at NeurIPS 2025
>
> **摘要:** Sparse autoencoders (SAEs) have emerged as a powerful tool for uncovering interpretable features in large language models (LLMs) through the sparse directions they learn. However, the sheer number of extracted directions makes comprehensive exploration intractable. While conventional embedding techniques such as UMAP can reveal global structure, they suffer from limitations including high-dimensional compression artifacts, overplotting, and misleading neighborhood distortions. In this work, we propose a focused exploration framework that prioritizes curated concepts and their corresponding SAE features over attempts to visualize all available features simultaneously. We present an interactive visualization system that combines topology-based visual encoding with dimensionality reduction to faithfully represent both local and global relationships among selected features. This hybrid approach enables users to investigate SAE behavior through targeted, interpretable subsets, facilitating deeper and more nuanced analysis of concept representation in latent space.
>
---
#### [new 050] RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments
- **分类: cs.CL; cs.LG**

- **简介: 论文提出RLVE，通过自适应可验证环境动态调整问题难度，解决传统RL中奖励信号消失问题。构建RLVE-Gym（400个环境），显著提升语言模型推理能力，比原方法提升近7倍。**

- **链接: [http://arxiv.org/pdf/2511.07317v1](http://arxiv.org/pdf/2511.07317v1)**

> **作者:** Zhiyuan Zeng; Hamish Ivison; Yiping Wang; Lifan Yuan; Shuyue Stella Li; Zhuorui Ye; Siting Li; Jacqueline He; Runlong Zhou; Tong Chen; Chenyang Zhao; Yulia Tsvetkov; Simon Shaolei Du; Natasha Jaques; Hao Peng; Pang Wei Koh; Hannaneh Hajishirzi
>
> **摘要:** We introduce Reinforcement Learning (RL) with Adaptive Verifiable Environments (RLVE), an approach using verifiable environments that procedurally generate problems and provide algorithmically verifiable rewards, to scale up RL for language models (LMs). RLVE enables each verifiable environment to dynamically adapt its problem difficulty distribution to the policy model's capabilities as training progresses. In contrast, static data distributions often lead to vanishing learning signals when problems are either too easy or too hard for the policy. To implement RLVE, we create RLVE-Gym, a large-scale suite of 400 verifiable environments carefully developed through manual environment engineering. Using RLVE-Gym, we show that environment scaling, i.e., expanding the collection of training environments, consistently improves generalizable reasoning capabilities. RLVE with joint training across all 400 environments in RLVE-Gym yields a 3.37% absolute average improvement across six reasoning benchmarks, starting from one of the strongest 1.5B reasoning LMs. By comparison, continuing this LM's original RL training yields only a 0.49% average absolute gain despite using over 3x more compute. We release our code publicly.
>
---
#### [new 051] Beyond One-Size-Fits-All: Personalized Harmful Content Detection with In-Context Learning
- **分类: cs.CL**

- **简介: 该论文提出基于上下文学习的个性化有害内容检测框架，解决传统系统缺乏灵活性与隐私保护的问题，无需重训练即可通过提示实现用户自定义内容过滤，支持多任务泛化与轻量级个性化。**

- **链接: [http://arxiv.org/pdf/2511.05532v1](http://arxiv.org/pdf/2511.05532v1)**

> **作者:** Rufan Zhang; Lin Zhang; Xianghang Mi
>
> **摘要:** The proliferation of harmful online content--e.g., toxicity, spam, and negative sentiment--demands robust and adaptable moderation systems. However, prevailing moderation systems are centralized and task-specific, offering limited transparency and neglecting diverse user preferences--an approach ill-suited for privacy-sensitive or decentralized environments. We propose a novel framework that leverages in-context learning (ICL) with foundation models to unify the detection of toxicity, spam, and negative sentiment across binary, multi-class, and multi-label settings. Crucially, our approach enables lightweight personalization, allowing users to easily block new categories, unblock existing ones, or extend detection to semantic variations through simple prompt-based interventions--all without model retraining. Extensive experiments on public benchmarks (TextDetox, UCI SMS, SST2) and a new, annotated Mastodon dataset reveal that: (i) foundation models achieve strong cross-task generalization, often matching or surpassing task-specific fine-tuned models; (ii) effective personalization is achievable with as few as one user-provided example or definition; and (iii) augmenting prompts with label definitions or rationales significantly enhances robustness to noisy, real-world data. Our work demonstrates a definitive shift beyond one-size-fits-all moderation, establishing ICL as a practical, privacy-preserving, and highly adaptable pathway for the next generation of user-centric content safety systems. To foster reproducibility and facilitate future research, we publicly release our code on GitHub and the annotated Mastodon dataset on Hugging Face.
>
---
#### [new 052] SPA: Achieving Consensus in LLM Alignment via Self-Priority Optimization
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出SPA框架，解决LLM在高风险场景中“安全”与“有用”目标冲突问题，通过自优先优化实现“安全优先、有用次之”的对齐，无需人工标注，提升帮助性同时保障安全性。**

- **链接: [http://arxiv.org/pdf/2511.06222v1](http://arxiv.org/pdf/2511.06222v1)**

> **作者:** Yue Huang; Xiangqi Wang; Xiangliang Zhang
>
> **备注:** Accepted by AAAI 2026 (Oral)
>
> **摘要:** In high-stakes scenarios-such as self-harm, legal, or medical queries-LLMs must be both trustworthy and helpful. However, these goals often conflict. We propose priority alignment, a new alignment paradigm that enforces a strict "trustworthy-before-helpful" ordering: optimization of helpfulness is conditioned on first meeting trustworthy thresholds (e.g., harmlessness or honesty). To realize this, we introduce Self-Priority Alignment (SPA)-a fully unsupervised framework that generates diverse responses, self-evaluates them and refines them by the model itself, and applies dual-criterion denoising to remove inconsistency and control variance. From this, SPA constructs lexicographically ordered preference pairs and fine-tunes the model using an uncertainty-weighted alignment loss that emphasizes high-confidence, high-gap decisions. Experiments across multiple benchmarks show that SPA improves helpfulness without compromising safety, outperforming strong baselines while preserving general capabilities. Our results demonstrate that SPA provides a scalable and interpretable alignment strategy for critical LLM applications.
>
---
#### [new 053] RPTS: Tree-Structured Reasoning Process Scoring for Faithful Multimodal Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出RPTS，一种基于树结构的推理过程评分方法，用于评估多模态模型的推理可信度，解决现有基准忽略推理过程与跨模态关系的问题，并构建RPTS-Eval基准验证其有效性。**

- **链接: [http://arxiv.org/pdf/2511.06899v1](http://arxiv.org/pdf/2511.06899v1)**

> **作者:** Haofeng Wang; Yu Zhang
>
> **摘要:** Large Vision-Language Models (LVLMs) excel in multimodal reasoning and have shown impressive performance on various multimodal benchmarks. However, most of these benchmarks evaluate models primarily through multiple-choice or short-answer formats, which do not take the reasoning process into account. Although some benchmarks assess the reasoning process, their methods are often overly simplistic and only examine reasoning when answers are incorrect. This approach overlooks scenarios where flawed reasoning leads to correct answers. In addition, these benchmarks do not consider the impact of intermodal relationships on reasoning. To address this issue, we propose the Reasoning Process Tree Score (RPTS), a tree structure-based metric to assess reasoning processes. Specifically, we organize the reasoning steps into a reasoning tree and leverage its hierarchical information to assign weighted faithfulness scores to each reasoning step. By dynamically adjusting these weights, RPTS not only evaluates the overall correctness of the reasoning, but also pinpoints where the model fails in the reasoning. To validate RPTS in real-world multimodal scenarios, we construct a new benchmark, RPTS-Eval, comprising 374 images and 390 reasoning instances. Each instance includes reliable visual-textual clues that serve as leaf nodes of the reasoning tree. Furthermore, we define three types of intermodal relationships to investigate how intermodal interactions influence the reasoning process. We evaluated representative LVLMs (e.g., GPT4o, Llava-Next), uncovering their limitations in multimodal reasoning and highlighting the differences between open-source and closed-source commercial LVLMs. We believe that this benchmark will contribute to the advancement of research in the field of multimodal reasoning.
>
---
#### [new 054] Rethinking what Matters: Effective and Robust Multilingual Realignment for Low-Resource Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言模型中的词对齐任务，解决低资源语言（LRLs）对齐效果差、依赖高质量平行数据的问题。通过精选语言子集进行对齐，证明其可媲美甚至超越全语言对齐，提升LRLs的跨语言迁移效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2511.06497v1](http://arxiv.org/pdf/2511.06497v1)**

> **作者:** Quang Phuoc Nguyen; David Anugraha; Felix Gaschi; Jun Bin Cheng; En-Shiun Annie Lee
>
> **备注:** Accepted to IJCNLP-AACL 2025
>
> **摘要:** Realignment is a promising strategy to improve cross-lingual transfer in multilingual language models. However, empirical results are mixed and often unreliable, particularly for typologically distant or low-resource languages (LRLs) compared to English. Moreover, word realignment tools often rely on high-quality parallel data, which can be scarce or noisy for many LRLs. In this work, we conduct an extensive empirical study to investigate whether realignment truly benefits from using all available languages, or if strategically selected subsets can offer comparable or even improved cross-lingual transfer, and study the impact on LRLs. Our controlled experiments show that realignment can be particularly effective for LRLs and that using carefully selected, linguistically diverse subsets can match full multilingual alignment, and even outperform it for unseen LRLs. This indicates that effective realignment does not require exhaustive language coverage and can reduce data collection overhead, while remaining both efficient and robust when guided by informed language selection.
>
---
#### [new 055] Explicit Knowledge-Guided In-Context Learning for Early Detection of Alzheimer's Disease
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对阿尔茨海默病（AD）早期检测任务，提出EK-ICL框架，利用结构化知识增强大模型上下文学习的稳定性与语义对齐，解决临床数据稀缺下标签错位与演示选择不佳问题，显著提升检测性能。**

- **链接: [http://arxiv.org/pdf/2511.06215v1](http://arxiv.org/pdf/2511.06215v1)**

> **作者:** Puzhen Su; Yongzhu Miao; Chunxi Guo; Jintao Tang; Shasha Li; Ting Wang
>
> **备注:** This paper was accepted by IEEE BIBM 2025 conference
>
> **摘要:** Detecting Alzheimer's Disease (AD) from narrative transcripts remains a challenging task for large language models (LLMs), particularly under out-of-distribution (OOD) and data-scarce conditions. While in-context learning (ICL) provides a parameter-efficient alternative to fine-tuning, existing ICL approaches often suffer from task recognition failure, suboptimal demonstration selection, and misalignment between label words and task objectives, issues that are amplified in clinical domains like AD detection. We propose Explicit Knowledge In-Context Learners (EK-ICL), a novel framework that integrates structured explicit knowledge to enhance reasoning stability and task alignment in ICL. EK-ICL incorporates three knowledge components: confidence scores derived from small language models (SLMs) to ground predictions in task-relevant patterns, parsing feature scores to capture structural differences and improve demo selection, and label word replacement to resolve semantic misalignment with LLM priors. In addition, EK-ICL employs a parsing-based retrieval strategy and ensemble prediction to mitigate the effects of semantic homogeneity in AD transcripts. Extensive experiments across three AD datasets demonstrate that EK-ICL significantly outperforms state-of-the-art fine-tuning and ICL baselines. Further analysis reveals that ICL performance in AD detection is highly sensitive to the alignment of label semantics and task-specific context, underscoring the importance of explicit knowledge in clinical reasoning under low-resource conditions.
>
---
#### [new 056] Beyond Plain Demos: A Demo-centric Anchoring Paradigm for In-Context Learning in Alzheimer's Disease Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文面向阿尔茨海默病（AD）检测任务，针对LLM在上下文学习中因演示样本同质化导致的感知能力不足，提出DA4ICL框架，通过多样对比检索与投影向量锚定，增强上下文宽度与深度，显著提升低资源、分布外场景下的检测性能。**

- **链接: [http://arxiv.org/pdf/2511.06826v1](http://arxiv.org/pdf/2511.06826v1)**

> **作者:** Puzhen Su; Haoran Yin; Yongzhu Miao; Jintao Tang; Shasha Li; Ting Wang
>
> **备注:** Accepted to the 40th Annual AAAI Conference on Artificial Intelligence (2026) - Main Technical Track (Oral)
>
> **摘要:** Detecting Alzheimer's disease (AD) from narrative transcripts challenges large language models (LLMs): pre-training rarely covers this out-of-distribution task, and all transcript demos describe the same scene, producing highly homogeneous contexts. These factors cripple both the model's built-in task knowledge (\textbf{task cognition}) and its ability to surface subtle, class-discriminative cues (\textbf{contextual perception}). Because cognition is fixed after pre-training, improving in-context learning (ICL) for AD detection hinges on enriching perception through better demonstration (demo) sets. We demonstrate that standard ICL quickly saturates, its demos lack diversity (context width) and fail to convey fine-grained signals (context depth), and that recent task vector (TV) approaches improve broad task adaptation by injecting TV into the LLMs' hidden states (HSs), they are ill-suited for AD detection due to the mismatch of injection granularity, strength and position. To address these bottlenecks, we introduce \textbf{DA4ICL}, a demo-centric anchoring framework that jointly expands context width via \emph{\textbf{Diverse and Contrastive Retrieval}} (DCR) and deepens each demo's signal via \emph{\textbf{Projected Vector Anchoring}} (PVA) at every Transformer layer. Across three AD benchmarks, DA4ICL achieves large, stable gains over both ICL and TV baselines, charting a new paradigm for fine-grained, OOD and low-resource LLM adaptation.
>
---
#### [new 057] Sample-Efficient Language Modeling with Linear Attention and Lightweight Enhancements
- **分类: cs.CL; cs.AI**

- **简介: 该论文面向BabyLM 2025小样本语言建模任务，提出BLaLM模型，用线性mLSTM替代自注意力，结合轻量增强技术与Muon优化器，在低资源下提升效率与零样本性能，无需依赖模型规模。**

- **链接: [http://arxiv.org/pdf/2511.05560v1](http://arxiv.org/pdf/2511.05560v1)**

> **作者:** Patrick Haller; Jonas Golde; Alan Akbik
>
> **摘要:** We study architectural and optimization tech- niques for sample-efficient language modeling under the constraints of the BabyLM 2025 shared task. Our model, BLaLM, replaces self-attention with a linear-time mLSTM to- ken mixer and explores lightweight enhance- ments, including short convolutions, sliding window attention with dynamic modulation, and Hedgehog feature maps. To support train- ing in low-resource settings, we curate a high- quality corpus emphasizing readability and ped- agogical structure. Experiments across both STRICT and STRICT-SMALL tracks show that (1) linear attention combined with sliding win- dow attention consistently improves zero-shot performance, and (2) the Muon optimizer stabi- lizes convergence and reduces perplexity over AdamW. These results highlight effective strate- gies for efficient language modeling without relying on scale.
>
---
#### [new 058] MCP4IFC: IFC-Based Building Design Using Large Language Models
- **分类: cs.CL**

- **简介: 论文提出MCP4IFC框架，利用大语言模型直接操作IFC标准BIM数据，解决AI难以理解与生成建筑模型的问题，通过工具集与RAG机制实现自然语言驱动的建模与编辑，开源以推动AI辅助设计。**

- **链接: [http://arxiv.org/pdf/2511.05533v1](http://arxiv.org/pdf/2511.05533v1)**

> **作者:** Bharathi Kannan Nithyanantham; Tobias Sesterhenn; Ashwin Nedungadi; Sergio Peral Garijo; Janis Zenkner; Christian Bartelt; Stefan Lüdtke
>
> **摘要:** Bringing generative AI into the architecture, engineering and construction (AEC) field requires systems that can translate natural language instructions into actions on standardized data models. We present MCP4IFC, a comprehensive open-source framework that enables Large Language Models (LLMs) to directly manipulate Industry Foundation Classes (IFC) data through the Model Context Protocol (MCP). The framework provides a set of BIM tools, including scene querying tools for information retrieval, predefined functions for creating and modifying common building elements, and a dynamic code-generation system that combines in-context learning with retrieval-augmented generation (RAG) to handle tasks beyond the predefined toolset. Experiments demonstrate that an LLM using our framework can successfully perform complex tasks, from building a simple house to querying and editing existing IFC data. Our framework is released as open-source to encourage research in LLM-driven BIM design and provide a foundation for AI-assisted modeling workflows. Our code is available at https://show2instruct.github.io/mcp4ifc/.
>
---
#### [new 059] ACE-ICD: Acronym Expansion As Data Augmentation For Automated ICD Coding
- **分类: cs.CL**

- **简介: 该论文针对电子病历自动ICD编码任务，解决医疗缩略语被忽略导致编码不准的问题，提出ACE-ICD方法：利用大模型展开缩略语并结合一致性训练进行数据增强，显著提升编码准确率。**

- **链接: [http://arxiv.org/pdf/2511.07311v1](http://arxiv.org/pdf/2511.07311v1)**

> **作者:** Tuan-Dung Le; Shohreh Haddadan; Thanh Q. Thieu
>
> **备注:** Camera ready version for IJCNLP-AACL 2025 (Findings)
>
> **摘要:** Automatic ICD coding, the task of assigning disease and procedure codes to electronic medical records, is crucial for clinical documentation and billing. While existing methods primarily enhance model understanding of code hierarchies and synonyms, they often overlook the pervasive use of medical acronyms in clinical notes, a key factor in ICD code inference. To address this gap, we propose a novel effective data augmentation technique that leverages large language models to expand medical acronyms, allowing models to be trained on their full form representations. Moreover, we incorporate consistency training to regularize predictions by enforcing agreement between the original and augmented documents. Extensive experiments on the MIMIC-III dataset demonstrate that our approach, ACE-ICD establishes new state-of-the-art performance across multiple settings, including common codes, rare codes, and full-code assignments. Our code is publicly available.
>
---
#### [new 060] Retracing the Past: LLMs Emit Training Data When They Get Lost
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Confusion-Inducing Attacks（CIA），通过诱导模型不确定性提取LLM中记忆的训练数据，解决现有方法效率低、机制不清的问题，并结合Mismatched SFT增强攻击效果，系统性评估隐私泄露风险。**

- **链接: [http://arxiv.org/pdf/2511.05518v1](http://arxiv.org/pdf/2511.05518v1)**

> **作者:** Myeongseob Ko; Nikhil Reddy Billa; Adam Nguyen; Charles Fleming; Ming Jin; Ruoxi Jia
>
> **备注:** The 2025 Conference on Empirical Methods in Natural Language Processing
>
> **摘要:** The memorization of training data in large language models (LLMs) poses significant privacy and copyright concerns. Existing data extraction methods, particularly heuristic-based divergence attacks, often exhibit limited success and offer limited insight into the fundamental drivers of memorization leakage. This paper introduces Confusion-Inducing Attacks (CIA), a principled framework for extracting memorized data by systematically maximizing model uncertainty. We empirically demonstrate that the emission of memorized text during divergence is preceded by a sustained spike in token-level prediction entropy. CIA leverages this insight by optimizing input snippets to deliberately induce this consecutive high-entropy state. For aligned LLMs, we further propose Mismatched Supervised Fine-tuning (SFT) to simultaneously weaken their alignment and induce targeted confusion, thereby increasing susceptibility to our attacks. Experiments on various unaligned and aligned LLMs demonstrate that our proposed attacks outperform existing baselines in extracting verbatim and near-verbatim training data without requiring prior knowledge of the training data. Our findings highlight persistent memorization risks across various LLMs and offer a more systematic method for assessing these vulnerabilities.
>
---
#### [new 061] Optimizing Diversity and Quality through Base-Aligned Model Collaboration
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出BACo框架，在推理阶段动态融合基础模型与对齐模型，通过令牌级路由平衡生成质量与多样性，无需重训练或多次采样，显著提升两者协同表现。**

- **链接: [http://arxiv.org/pdf/2511.05650v1](http://arxiv.org/pdf/2511.05650v1)**

> **作者:** Yichen Wang; Chenghao Yang; Tenghao Huang; Muhao Chen; Jonathan May; Mina Lee
>
> **备注:** 52 pages, 16 figures
>
> **摘要:** Alignment has greatly improved large language models (LLMs)' output quality at the cost of diversity, yielding highly similar outputs across generations. We propose Base-Aligned Model Collaboration (BACo), an inference-time token-level model collaboration framework that dynamically combines a base LLM with its aligned counterpart to optimize diversity and quality. Inspired by prior work (Fei et al., 2025), BACo employs routing strategies that determine, at each token, from which model to decode based on next-token prediction uncertainty and predicted contents' semantic role. Prior diversity-promoting methods, such as retraining, prompt engineering, and multi-sampling methods, improve diversity but often degrade quality or require costly decoding or post-training. In contrast, BACo achieves both high diversity and quality post hoc within a single pass, while offering strong controllability. We explore a family of routing strategies, across three open-ended generation tasks and 13 metrics covering diversity and quality, BACo consistently surpasses state-of-the-art inference-time baselines. With our best router, BACo achieves a 21.3% joint improvement in diversity and quality. Human evaluations also mirror these improvements. The results suggest that collaboration between base and aligned models can optimize and control diversity and quality.
>
---
#### [new 062] Overview of CHIP 2025 Shared Task 2: Discharge Medication Recommendation for Metabolic Diseases Based on Chinese Electronic Health Records
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CHIP 2025共享任务2，旨在基于中文电子健康记录自动推荐代谢病患者出院用药，构建了CDrugRed数据集，推动LLM在多标签用药推荐中的应用，最高模型F1达0.6267。**

- **链接: [http://arxiv.org/pdf/2511.06230v1](http://arxiv.org/pdf/2511.06230v1)**

> **作者:** Juntao Li; Haobin Yuan; Ling Luo; Tengxiao Lv; Yan Jiang; Fan Wang; Ping Zhang; Huiyi Lv; Jian Wang; Yuanyuan Sun; Hongfei Lin
>
> **摘要:** Discharge medication recommendation plays a critical role in ensuring treatment continuity, preventing readmission, and improving long-term management for patients with chronic metabolic diseases. This paper present an overview of the CHIP 2025 Shared Task 2 competition, which aimed to develop state-of-the-art approaches for automatically recommending appro-priate discharge medications using real-world Chinese EHR data. For this task, we constructed CDrugRed, a high-quality dataset consisting of 5,894 de-identified hospitalization records from 3,190 patients in China. This task is challenging due to multi-label nature of medication recommendation, het-erogeneous clinical text, and patient-specific variability in treatment plans. A total of 526 teams registered, with 167 and 95 teams submitting valid results to the Phase A and Phase B leaderboards, respectively. The top-performing team achieved the highest overall performance on the final test set, with a Jaccard score of 0.5102, F1 score of 0.6267, demonstrating the potential of advanced large language model (LLM)-based ensemble systems. These re-sults highlight both the promise and remaining challenges of applying LLMs to medication recommendation in Chinese EHRs. The post-evaluation phase remains open at https://tianchi.aliyun.com/competition/entrance/532411/.
>
---
#### [new 063] TimeSense:Making Large Language Models Proficient in Time-Series Analysis
- **分类: cs.CL; cs.AI**

- **简介: 论文提出TimeSense框架，解决LLM在时间序列分析中过度依赖文本提示、忽视时序动态的问题，通过引入时序感知模块与坐标位置嵌入，实现文本与时序信息的平衡，提升复杂时序推理性能。**

- **链接: [http://arxiv.org/pdf/2511.06344v1](http://arxiv.org/pdf/2511.06344v1)**

> **作者:** Zhirui Zhang; Changhua Pei; Tianyi Gao; Zhe Xie; Yibo Hao; Zhaoyang Yu; Longlong Xu; Tong Xiao; Jing Han; Dan Pei
>
> **摘要:** In the time-series domain, an increasing number of works combine text with temporal data to leverage the reasoning capabilities of large language models (LLMs) for various downstream time-series understanding tasks. This enables a single model to flexibly perform tasks that previously required specialized models for each domain. However, these methods typically rely on text labels for supervision during training, biasing the model toward textual cues while potentially neglecting the full temporal features. Such a bias can lead to outputs that contradict the underlying time-series context. To address this issue, we construct the EvalTS benchmark, comprising 10 tasks across three difficulty levels, from fundamental temporal pattern recognition to complex real-world reasoning, to evaluate models under more challenging and realistic scenarios. We also propose TimeSense, a multimodal framework that makes LLMs proficient in time-series analysis by balancing textual reasoning with a preserved temporal sense. TimeSense incorporates a Temporal Sense module that reconstructs the input time-series within the model's context, ensuring that textual reasoning is grounded in the time-series dynamics. Moreover, to enhance spatial understanding of time-series data, we explicitly incorporate coordinate-based positional embeddings, which provide each time point with spatial context and enable the model to capture structural dependencies more effectively. Experimental results demonstrate that TimeSense achieves state-of-the-art performance across multiple tasks, and it particularly outperforms existing methods on complex multi-dimensional time-series reasoning tasks.
>
---
#### [new 064] Quantifying Edits Decay in Fine-tuned LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究知识编辑与微调的交互影响，发现微调会导致编辑失效，提出选择性微调策略以控制编辑留存，为编辑与微调的协同应用提供实证基础与解决方案。**

- **链接: [http://arxiv.org/pdf/2511.05852v1](http://arxiv.org/pdf/2511.05852v1)**

> **作者:** Yinjie Cheng; Paul Youssef; Christin Seifert; Jörg Schlötterer; Zhixue Zhao
>
> **备注:** Under review at ICLR 2026
>
> **摘要:** Knowledge editing has emerged as a lightweight alternative to retraining for correcting or injecting specific facts in large language models (LLMs). Meanwhile, fine-tuning remains the default operation for adapting LLMs to new domains and tasks. Despite their widespread adoption, these two post-training interventions have been studied in isolation, leaving open a crucial question: if we fine-tune an edited model, do the edits survive? This question is motivated by two practical scenarios: removing covert or malicious edits, and preserving beneficial edits. If fine-tuning impairs edits as shown in Figure 1, current KE methods become less useful, as every fine-tuned model would require re-editing, which significantly increases the cost; if edits persist, fine-tuned models risk propagating hidden malicious edits, raising serious safety concerns. To this end, we systematically quantify edits decay after fine-tuning, investigating how fine-tuning affects knowledge editing. We evaluate two state-of-the-art editing methods (MEMIT, AlphaEdit) and three fine-tuning approaches (full-parameter, LoRA, DoRA) across five LLMs and three datasets, yielding 232 experimental configurations. Our results show that edits decay after fine-tuning, with survival varying across configurations, e.g., AlphaEdit edits decay more than MEMIT edits. Further, we propose selective-layer fine-tuning and find that fine-tuning edited layers only can effectively remove edits, though at a slight cost to downstream performance. Surprisingly, fine-tuning non-edited layers impairs more edits than full fine-tuning. Overall, our study establishes empirical baselines and actionable strategies for integrating knowledge editing with fine-tuning, and underscores that evaluating model editing requires considering the full LLM application pipeline.
>
---
#### [new 065] Sentiment Analysis On YouTube Comments Using Machine Learning Techniques Based On Video Games Content
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在分析YouTube游戏视频评论中的用户情感。通过YouTube API收集数据，使用TextBlob预处理，并比较Naive Bayes、逻辑回归和SVM算法，发现SVM准确率最高，为游戏优化提供情感反馈。**

- **链接: [http://arxiv.org/pdf/2511.06708v1](http://arxiv.org/pdf/2511.06708v1)**

> **作者:** Adi Danish Bin Muhammad Amin; Mohaiminul Islam Bhuiyan; Nur Shazwani Kamarudin; Zulfahmi Toh; Nur Syafiqah Nafis
>
> **备注:** 6 pages, 7 figures, 2025 IEEE 9th International Conference on Software Engineering & Computer Systems
>
> **摘要:** The rapid evolution of the gaming industry, driven by technological advancements and a burgeoning community, necessitates a deeper understanding of user sentiments, especially as expressed on popular social media platforms like YouTube. This study presents a sentiment analysis on video games based on YouTube comments, aiming to understand user sentiments within the gaming community. Utilizing YouTube API, comments related to various video games were collected and analyzed using the TextBlob sentiment analysis tool. The pre-processed data underwent classification using machine learning algorithms, including Na\"ive Bayes, Logistic Regression, and Support Vector Machine (SVM). Among these, SVM demonstrated superior performance, achieving the highest classification accuracy across different datasets. The analysis spanned multiple popular gaming videos, revealing trends and insights into user preferences and critiques. The findings underscore the importance of advanced sentiment analysis in capturing the nuanced emotions expressed in user comments, providing valuable feedback for game developers to enhance game design and user experience. Future research will focus on integrating more sophisticated natural language processing techniques and exploring additional data sources to further refine sentiment analysis in the gaming domain.
>
---
#### [new 066] Importance-Aware Data Selection for Efficient LLM Instruction Tuning
- **分类: cs.CL**

- **简介: 该论文针对LLM指令微调中的数据低效问题，提出MIWV指标，基于上下文学习的响应差异识别对模型能力提升最关键的指令数据，仅用Top 1%高重要性数据即可超越全数据集效果。**

- **链接: [http://arxiv.org/pdf/2511.07074v1](http://arxiv.org/pdf/2511.07074v1)**

> **作者:** Tingyu Jiang; Shen Li; Yiyao Song; Lan Zhang; Hualei Zhu; Yuan Zhao; Xiaohang Xu; Kenjiro Taura; Hao Henry Wang
>
> **备注:** Accepted by AAAI 2026 Oral
>
> **摘要:** Instruction tuning plays a critical role in enhancing the performance and efficiency of Large Language Models (LLMs). Its success depends not only on the quality of the instruction data but also on the inherent capabilities of the LLM itself. Some studies suggest that even a small amount of high-quality data can achieve instruction fine-tuning results that are on par with, or even exceed, those from using a full-scale dataset. However, rather than focusing solely on calculating data quality scores to evaluate instruction data, there is a growing need to select high-quality data that maximally enhances the performance of instruction tuning for a given LLM. In this paper, we propose the Model Instruction Weakness Value (MIWV) as a novel metric to quantify the importance of instruction data in enhancing model's capabilities. The MIWV metric is derived from the discrepancies in the model's responses when using In-Context Learning (ICL), helping identify the most beneficial data for enhancing instruction tuning performance. Our experimental results demonstrate that selecting only the top 1\% of data based on MIWV can outperform training on the full dataset. Furthermore, this approach extends beyond existing research that focuses on data quality scoring for data selection, offering strong empirical evidence supporting the effectiveness of our proposed method.
>
---
#### [new 067] You Had One Job: Per-Task Quantization Using LLMs' Hidden Representations
- **分类: cs.CL**

- **简介: 该论文针对大语言模型（LLM）的低效问题，提出任务感知量化方法（TAQ/TAQO），利用隐藏表征识别任务关键层，动态分配比特精度，在保持接近原模型性能下显著降低内存与延迟。**

- **链接: [http://arxiv.org/pdf/2511.06516v1](http://arxiv.org/pdf/2511.06516v1)**

> **作者:** Amit LeVi; Raz Lapid; Rom Himelstein; Yaniv Nemcovsky; Ravid Shwartz Ziv; Avi Mendelson
>
> **摘要:** Large Language Models (LLMs) excel across diverse tasks, yet many applications require only limited capabilities, making large variants inefficient in memory and latency. Existing approaches often combine distillation and quantization, but most post-training quantization (PTQ) methods are task-agnostic, ignoring how task-specific signals are distributed across layers. In this work, we propose to use hidden representations that encode task-salient signals as a guideline for quantization. In order to fully utilize our innovative idea, this paper compares two new task-aware PTQ methods: Task-Aware Quantization (TAQ), which allocates bitwidths using task-conditioned statistics from hidden activations, and TAQO, which allocates precision based on direct layer sensitivity tests. From a small calibration set, these approaches identify task-relevant layers, preserving their precision while aggressively quantizing the rest. This yields stable task sensitivity profiles and efficient task-specialized models. Across models, TAQ and TAQO outperform the baselines; TAQ leads on Phi-4, while TAQO leads on Llama-3.1, Qwen3, and Qwen2.5. For instances, on Phi-4 it achieves 42.33 EM / 50.81 F1, far surpassing Activation-aware Weight Quantization (AWQ) (2.25 / 7.07), while remaining within < 1.0% of the original accuracy at lower average precision.
>
---
#### [new 068] How Well Do LLMs Understand Drug Mechanisms? A Knowledge + Reasoning Evaluation Dataset
- **分类: cs.CL**

- **简介: 该论文构建了一个评估LLM对药物作用机制理解的基准数据集，聚焦事实知识与反事实推理能力，揭示开放世界推理更难，且内部链路扰动更具挑战性，发现o4-mini与Qwen3-4B-thinking表现优异。**

- **链接: [http://arxiv.org/pdf/2511.06418v1](http://arxiv.org/pdf/2511.06418v1)**

> **作者:** Sunil Mohan; Theofanis Karaletsos
>
> **备注:** An earlier version of this paper appears in IEEE FLLM 2025. GitHub: https://github.com/czi-ai/DrugMechCounterfactuals
>
> **摘要:** Two scientific fields showing increasing interest in pre-trained large language models (LLMs) are drug development / repurposing, and personalized medicine. For both, LLMs have to demonstrate factual knowledge as well as a deep understanding of drug mechanisms, so they can recall and reason about relevant knowledge in novel situations. Drug mechanisms of action are described as a series of interactions between biomedical entities, which interlink into one or more chains directed from the drug to the targeted disease. Composing the effects of the interactions in a candidate chain leads to an inference about whether the drug might be useful or not for that disease. We introduce a dataset that evaluates LLMs on both factual knowledge of known mechanisms, and their ability to reason about them under novel situations, presented as counterfactuals that the models are unlikely to have seen during training. Using this dataset, we show that o4-mini outperforms the 4o, o3, and o3-mini models from OpenAI, and the recent small Qwen3-4B-thinking model closely matches o4-mini's performance, even outperforming it in some cases. We demonstrate that the open world setting for reasoning tasks, which requires the model to recall relevant knowledge, is more challenging than the closed world setting where the needed factual knowledge is provided. We also show that counterfactuals affecting internal links in the reasoning chain present a much harder task than those affecting a link from the drug mentioned in the prompt.
>
---
#### [new 069] Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit Chain-of-Thought
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出EBM-CoT，通过能量模型校准隐式思维表示，解决显式链式推理中误差传播与一致性差的问题，在不修改基座模型下提升多步推理的准确性和稳定性。**

- **链接: [http://arxiv.org/pdf/2511.07124v1](http://arxiv.org/pdf/2511.07124v1)**

> **作者:** Zhikang Chen; Sen Cui; Deheng Ye; Yu Zhang; Yatao Bian; Tingting Zhu
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong reasoning capabilities through \emph{Chain-of-Thought} (CoT) prompting, which enables step-by-step intermediate reasoning. However, explicit CoT methods rely on discrete token-level reasoning processes that are prone to error propagation and limited by vocabulary expressiveness, often resulting in rigid and inconsistent reasoning trajectories. Recent research has explored implicit or continuous reasoning in latent spaces, allowing models to perform internal reasoning before generating explicit output. Although such approaches alleviate some limitations of discrete CoT, they generally lack explicit mechanisms to enforce consistency among reasoning steps, leading to divergent reasoning paths and unstable outcomes. To address this issue, we propose EBM-CoT, an Energy-Based Chain-of-Thought Calibration framework that refines latent thought representations through an energy-based model (EBM). Our method dynamically adjusts latent reasoning trajectories toward lower-energy, high-consistency regions in the embedding space, improving both reasoning accuracy and consistency without modifying the base language model. Extensive experiments across mathematical, commonsense, and symbolic reasoning benchmarks demonstrate that the proposed framework significantly enhances the consistency and efficiency of multi-step reasoning in LLMs.
>
---
#### [new 070] Duality-based Mode Operations and Pyramid Multilayer Mapping for Rhetorical Modes
- **分类: cs.CL; cs.FL; cs.PL**

- **简介: 该论文提出基于对偶操作与金字塔多层映射的 rhetorical mode 扩展框架，将静态修辞分类转化为可量化、动态的推理结构，解决修辞模式难以计算建模的问题，为AI实现层叠修辞推理提供新路径。**

- **链接: [http://arxiv.org/pdf/2511.06601v1](http://arxiv.org/pdf/2511.06601v1)**

> **作者:** Zi-Niu Wu
>
> **摘要:** Rhetorical modes are useful in both academic and non-academic writing, and can be subjects to be studied within linguistic research and computational modeling. Establishing a conceptual bridge among these domains could enable each to benefit from the others. This paper proposes duality-based mode operations (split-unite, forward-backward, expansion-reduction and orthogonal dualities) to expand the set of rhetorical modes, introducing generated modes like combination and generalization, thereby enhancing epistemic diversity across multiple applications. It further presents a pyramid multilayer mapping framework (e.g., three layers from the rhetorical model layer, to cognitive layer, and to epistemic layers) that reduces the resulting cognitive complexity. The degrees of expressive diversity and complexity reduction are quantified through binomial combinatorics and Shannon entropy analysis. A Marginal Rhetorical Bit (MRB) is identified, permitting the definition of a rhetorical-scalable parameter that measures expressive growth speed in bits per stage. A direct entropy measure shows that hierarchical selection over smaller subsets markedly reduces choice uncertainty compared with flat selection across all modes. These considerations appear to transform static and non-measurable rhetorical taxonomies into more dynamic and more measurable systems for discourse design. From this work, it would be possible to identify a pathway for future AI systems to operate not only on language tokens but on layered rhetorical reasoning structures, bridging linguistic, pedagogical, academic, and computational research
>
---
#### [new 071] AdaRec: Adaptive Recommendation with LLMs via Narrative Profiling and Dual-Channel Reasoning
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: AdaRec提出一种基于LLM的自适应推荐框架，通过叙事画像与双通道推理，无需人工特征工程，实现少样本/零样本个性化推荐，显著提升长尾用户推荐效果。**

- **链接: [http://arxiv.org/pdf/2511.07166v1](http://arxiv.org/pdf/2511.07166v1)**

> **作者:** Meiyun Wang; Charin Polpanumas
>
> **摘要:** We propose AdaRec, a few-shot in-context learning framework that leverages large language models for an adaptive personalized recommendation. AdaRec introduces narrative profiling, transforming user-item interactions into natural language representations to enable unified task handling and enhance human readability. Centered on a bivariate reasoning paradigm, AdaRec employs a dual-channel architecture that integrates horizontal behavioral alignment, discovering peer-driven patterns, with vertical causal attribution, highlighting decisive factors behind user preferences. Unlike existing LLM-based approaches, AdaRec eliminates manual feature engineering through semantic representations and supports rapid cross-task adaptation with minimal supervision. Experiments on real ecommerce datasets demonstrate that AdaRec outperforms both machine learning models and LLM-based baselines by up to eight percent in few-shot settings. In zero-shot scenarios, it achieves up to a nineteen percent improvement over expert-crafted profiling, showing effectiveness for long-tail personalization with minimal interaction data. Furthermore, lightweight fine-tuning on synthetic data generated by AdaRec matches the performance of fully fine-tuned models, highlighting its efficiency and generalization across diverse tasks.
>
---
#### [new 072] Surgical Agent Orchestration Platform for Voice-directed Patient Data Interaction
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出语音驱动的外科代理编排平台（SAOP），解决机器人手术中医生无法中断操作访问患者数据的问题。基于多智能体LLM框架，实现语音指令到临床任务的自动映射，并通过MOEM评估其鲁棒性与准确性。**

- **链接: [http://arxiv.org/pdf/2511.07392v1](http://arxiv.org/pdf/2511.07392v1)**

> **作者:** Hyeryun Park; Byung Mo Gu; Jun Hee Lee; Byeong Hyeon Choi; Sekeun Kim; Hyun Koo Kim; Kyungsang Kim
>
> **备注:** 22 pages, 12 figures, 1 table, Supplementary Information, Supplementary Data 1
>
> **摘要:** In da Vinci robotic surgery, surgeons' hands and eyes are fully engaged in the procedure, making it difficult to access and manipulate multimodal patient data without interruption. We propose a voice-directed Surgical Agent Orchestrator Platform (SAOP) built on a hierarchical multi-agent framework, consisting of an orchestration agent and three task-specific agents driven by Large Language Models (LLMs). These LLM-based agents autonomously plan, refine, validate, and reason to map voice commands into specific tasks such as retrieving clinical information, manipulating CT scans, or navigating 3D anatomical models on the surgical video. We also introduce a Multi-level Orchestration Evaluation Metric (MOEM) to comprehensively assess the performance and robustness from command-level and category-level perspectives. The SAOP achieves high accuracy and success rates across 240 voice commands, while LLM-based agents improve robustness against speech recognition errors and diverse or ambiguous free-form commands, demonstrating strong potential to support minimally invasive da Vinci robotic surgery.
>
---
#### [new 073] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出时序稀疏自编码器（T-SAEs），解决传统SAE难以捕捉语义连贯特征的问题。通过引入时序对比损失，自监督地分离语义与句法特征，使模型发现更平滑、连贯的语义概念，提升可解释性。**

- **链接: [http://arxiv.org/pdf/2511.05541v1](http://arxiv.org/pdf/2511.05541v1)**

> **作者:** Usha Bhalla; Alex Oesterling; Claudio Mayrink Verdun; Himabindu Lakkaraju; Flavio P. Calmon
>
> **备注:** 23 Pages, 10 figures
>
> **摘要:** Translating the internal representations and computations of models into concepts that humans can understand is a key goal of interpretability. While recent dictionary learning methods such as Sparse Autoencoders (SAEs) provide a promising route to discover human-interpretable features, they suffer from a variety of problems, including a systematic failure to capture the rich conceptual information that drives linguistic understanding. Instead, they exhibit a bias towards shallow, token-specific, or noisy features, such as "the phrase 'The' at the start of sentences". In this work, we propose that this is due to a fundamental issue with how dictionary learning methods for LLMs are trained. Language itself has a rich, well-studied structure spanning syntax, semantics, and pragmatics; however, current unsupervised methods largely ignore this linguistic knowledge, leading to poor feature discovery that favors superficial patterns over meaningful concepts. We focus on a simple but important aspect of language: semantic content has long-range dependencies and tends to be smooth over a sequence, whereas syntactic information is much more local. Building on this insight, we introduce Temporal Sparse Autoencoders (T-SAEs), which incorporate a novel contrastive loss encouraging consistent activations of high-level features over adjacent tokens. This simple yet powerful modification enables SAEs to disentangle semantic from syntactic features in a self-supervised manner. Across multiple datasets and models, T-SAEs recover smoother, more coherent semantic concepts without sacrificing reconstruction quality. Strikingly, they exhibit clear semantic structure despite being trained without explicit semantic signal, offering a new pathway for unsupervised interpretability in language models.
>
---
#### [new 074] Automating Hardware Design and Verification from Architectural Papers via a Neural-Symbolic Graph Framework
- **分类: cs.CL; cs.SE**

- **简介: 该论文提出ArchCraft框架，将学术论文中的硬件架构描述自动转换为可综合的Verilog代码并完成RTL验证，解决硬件复现难的问题。构建首个ArchSynthBench基准，实现高精度设计生成与PPA一致性验证。**

- **链接: [http://arxiv.org/pdf/2511.06067v1](http://arxiv.org/pdf/2511.06067v1)**

> **作者:** Haoyue Yang; Xuanle Zhao; Yujie Liu; Zhuojun Zou; Kailin Lyu; Changchun Zhou; Yao Zhu; Jie Hao
>
> **备注:** Preprint Version, Work in Progress
>
> **摘要:** The reproduction of hardware architectures from academic papers remains a significant challenge due to the lack of publicly available source code and the complexity of hardware description languages (HDLs). To this end, we propose \textbf{ArchCraft}, a Framework that converts abstract architectural descriptions from academic papers into synthesizable Verilog projects with register-transfer level (RTL) verification. ArchCraft introduces a structured workflow, which uses formal graphs to capture the Architectural Blueprint and symbols to define the Functional Specification, translating unstructured academic papers into verifiable, hardware-aware designs. The framework then generates RTL and testbench (TB) code decoupled via these symbols to facilitate verification and debugging, ultimately reporting the circuit's Power, Area, and Performance (PPA). Moreover, we propose the first benchmark, \textbf{ArchSynthBench}, for synthesizing hardware from architectural descriptions, with a complete set of evaluation indicators, 50 project-level circuits, and around 600 circuit blocks. We systematically assess ArchCraft on ArchSynthBench, where the experiment results demonstrate the superiority of our proposed method, surpassing direct generation methods and the VerilogCoder framework in both paper understanding and code completion. Furthermore, evaluation and physical implementation of the generated executable RTL code show that these implementations meet all timing constraints without violations, and their performance metrics are consistent with those reported in the original papers.
>
---
#### [new 075] TabRAG: Tabular Document Retrieval via Structured Language Representations
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **简介: TabRAG面向表格密集文档的检索增强生成任务，解决传统解析方法对表格信息提取效果差的问题，提出基于结构化语言表示的解析管道，显著提升检索与生成性能。**

- **链接: [http://arxiv.org/pdf/2511.06582v1](http://arxiv.org/pdf/2511.06582v1)**

> **作者:** Jacob Si; Mike Qu; Michelle Lee; Yingzhen Li
>
> **备注:** NeurIPS 2025 AI4Tab
>
> **摘要:** Ingesting data for Retrieval-Augmented Generation (RAG) involves either fine-tuning the embedding model directly on the target corpus or parsing documents for embedding model encoding. The former, while accurate, incurs high computational hardware requirements, while the latter suffers from suboptimal performance when extracting tabular data. In this work, we address the latter by presenting TabRAG, a parsing-based RAG pipeline designed to tackle table-heavy documents via structured language representations. TabRAG outperforms existing popular parsing-based methods for generation and retrieval. Code is available at https://github.com/jacobyhsi/TabRAG.
>
---
#### [new 076] OckBench: Measuring the Efficiency of LLM Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 论文提出OckBench，首次在推理与编码任务中联合评估大模型的准确率与解码token效率，揭示效率差异被忽视的问题，推动评估范式从“忽略token成本”转向“权衡准确-效率帕累托前沿”。**

- **链接: [http://arxiv.org/pdf/2511.05722v1](http://arxiv.org/pdf/2511.05722v1)**

> **作者:** Zheng Du; Hao Kang; Song Han; Tushar Krishna; Ligeng Zhu
>
> **摘要:** Large language models such as GPT-4, Claude 3, and the Gemini series have improved automated reasoning and code generation. However, existing benchmarks mainly focus on accuracy and output quality, and they ignore an important factor: decoding token efficiency. In real systems, generating 10,000 tokens versus 100,000 tokens leads to large differences in latency, cost, and energy. In this work, we introduce OckBench, a model-agnostic and hardware-agnostic benchmark that evaluates both accuracy and token count for reasoning and coding tasks. Through experiments comparing multiple open- and closed-source models, we uncover that many models with comparable accuracy differ wildly in token consumption, revealing that efficiency variance is a neglected but significant axis of differentiation. We further demonstrate Pareto frontiers over the accuracy-efficiency plane and argue for an evaluation paradigm shift: we should no longer treat tokens as "free" to multiply. OckBench provides a unified platform for measuring, comparing, and guiding research in token-efficient reasoning. Our benchmarks are available at https://ockbench.github.io/ .
>
---
#### [new 077] When Bias Pretends to Be Truth: How Spurious Correlations Undermine Hallucination Detection in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLM中由训练数据中的虚假相关性引发的幻觉问题，揭示现有检测方法（如置信度过滤）失效，并通过合成实验与理论分析证明其根本缺陷，呼吁设计针对性的新检测方法。**

- **链接: [http://arxiv.org/pdf/2511.07318v1](http://arxiv.org/pdf/2511.07318v1)**

> **作者:** Shaowen Wang; Yiqi Dong; Ruinian Chang; Tansheng Zhu; Yuebo Sun; Kaifeng Lyu; Jian Li
>
> **摘要:** Despite substantial advances, large language models (LLMs) continue to exhibit hallucinations, generating plausible yet incorrect responses. In this paper, we highlight a critical yet previously underexplored class of hallucinations driven by spurious correlations -- superficial but statistically prominent associations between features (e.g., surnames) and attributes (e.g., nationality) present in the training data. We demonstrate that these spurious correlations induce hallucinations that are confidently generated, immune to model scaling, evade current detection methods, and persist even after refusal fine-tuning. Through systematically controlled synthetic experiments and empirical evaluations on state-of-the-art open-source and proprietary LLMs (including GPT-5), we show that existing hallucination detection methods, such as confidence-based filtering and inner-state probing, fundamentally fail in the presence of spurious correlations. Our theoretical analysis further elucidates why these statistical biases intrinsically undermine confidence-based detection techniques. Our findings thus emphasize the urgent need for new approaches explicitly designed to address hallucinations caused by spurious correlations.
>
---
#### [new 078] HatePrototypes: Interpretable and Transferable Representations for Implicit and Explicit Hate Speech Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出HatePrototypes，一种可迁移的类级向量表征，用于统一检测显式与隐式仇恨言论，避免重复微调。通过少量样本构建原型，实现跨任务迁移与无参数早退，提升效率与泛化性。**

- **链接: [http://arxiv.org/pdf/2511.06391v1](http://arxiv.org/pdf/2511.06391v1)**

> **作者:** Irina Proskurina; Marc-Antoine Carpentier; Julien Velcin
>
> **摘要:** Optimization of offensive content moderation models for different types of hateful messages is typically achieved through continued pre-training or fine-tuning on new hate speech benchmarks. However, existing benchmarks mainly address explicit hate toward protected groups and often overlook implicit or indirect hate, such as demeaning comparisons, calls for exclusion or violence, and subtle discriminatory language that still causes harm. While explicit hate can often be captured through surface features, implicit hate requires deeper, full-model semantic processing. In this work, we question the need for repeated fine-tuning and analyze the role of HatePrototypes, class-level vector representations derived from language models optimized for hate speech detection and safety moderation. We find that these prototypes, built from as few as 50 examples per class, enable cross-task transfer between explicit and implicit hate, with interchangeable prototypes across benchmarks. Moreover, we show that parameter-free early exiting with prototypes is effective for both hate types. We release the code, prototype resources, and evaluation scripts to support future research on efficient and transferable hate speech detection.
>
---
#### [new 079] EduGuardBench: A Holistic Benchmark for Evaluating the Pedagogical Fidelity and Adversarial Safety of LLMs as Simulated Teachers
- **分类: cs.CL; I.2.7**

- **简介: 论文提出EduGuardBench，首个评估LLM作为模拟教师的教育忠实度与对抗安全性基准，解决现有评测缺失教学伦理与角色扮演评估的问题，通过RFS与ASR等指标揭示模型性能极化与“教育转化效应”。**

- **链接: [http://arxiv.org/pdf/2511.06890v1](http://arxiv.org/pdf/2511.06890v1)**

> **作者:** Yilin Jiang; Mingzi Zhang; Xuanyu Yin; Sheng Jin; Suyu Lu; Zuocan Ying; Zengyi Yu; Xiangjie Kong
>
> **备注:** 22 pages, 9 figures, accepted by AAAI2026 as oral paper
>
> **摘要:** Large Language Models for Simulating Professions (SP-LLMs), particularly as teachers, are pivotal for personalized education. However, ensuring their professional competence and ethical safety is a critical challenge, as existing benchmarks fail to measure role-playing fidelity or address the unique teaching harms inherent in educational scenarios. To address this, we propose EduGuardBench, a dual-component benchmark. It assesses professional fidelity using a Role-playing Fidelity Score (RFS) while diagnosing harms specific to the teaching profession. It also probes safety vulnerabilities using persona-based adversarial prompts targeting both general harms and, particularly, academic misconduct, evaluated with metrics including Attack Success Rate (ASR) and a three-tier Refusal Quality assessment. Our extensive experiments on 14 leading models reveal a stark polarization in performance. While reasoning-oriented models generally show superior fidelity, incompetence remains the dominant failure mode across most models. The adversarial tests uncovered a counterintuitive scaling paradox, where mid-sized models can be the most vulnerable, challenging monotonic safety assumptions. Critically, we identified a powerful Educational Transformation Effect: the safest models excel at converting harmful requests into teachable moments by providing ideal Educational Refusals. This capacity is strongly negatively correlated with ASR, revealing a new dimension of advanced AI safety. EduGuardBench thus provides a reproducible framework that moves beyond siloed knowledge tests toward a holistic assessment of professional, ethical, and pedagogical alignment, uncovering complex dynamics essential for deploying trustworthy AI in education. See https://github.com/YL1N/EduGuardBench for Materials.
>
---
#### [new 080] Referring Expressions as a Lens into Spatial Language Grounding in Vision-Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出使用指代表达理解任务评估视觉语言模型的空间推理能力，针对模糊检测、复杂空间关系与否定表达等挑战，分析不同模型在拓扑、方向等空间语义上的表现，揭示其短板与研究缺口。**

- **链接: [http://arxiv.org/pdf/2511.06146v1](http://arxiv.org/pdf/2511.06146v1)**

> **作者:** Akshar Tumu; Varad Shinde; Parisa Kordjamshidi
>
> **备注:** Accepted at IJCNLP-AACL 2025
>
> **摘要:** Spatial Reasoning is an important component of human cognition and is an area in which the latest Vision-language models (VLMs) show signs of difficulty. The current analysis works use image captioning tasks and visual question answering. In this work, we propose using the Referring Expression Comprehension task instead as a platform for the evaluation of spatial reasoning by VLMs. This platform provides the opportunity for a deeper analysis of spatial comprehension and grounding abilities when there is 1) ambiguity in object detection, 2) complex spatial expressions with a longer sentence structure and multiple spatial relations, and 3) expressions with negation ('not'). In our analysis, we use task-specific architectures as well as large VLMs and highlight their strengths and weaknesses in dealing with these specific situations. While all these models face challenges with the task at hand, the relative behaviors depend on the underlying models and the specific categories of spatial semantics (topological, directional, proximal, etc.). Our results highlight these challenges and behaviors and provide insight into research gaps and future directions.
>
---
#### [new 081] Llama-Embed-Nemotron-8B: A Universal Text Embedding Model for Multilingual and Cross-Lingual Tasks
- **分类: cs.CL; cs.IR**

- **简介: 论文提出Llama-Embed-Nemotron-8B，一个开源多语言文本嵌入模型，解决现有模型训练数据不透明问题，通过合成数据与公开数据混合训练，在MMTEB基准上达SOTA，支持指令微调，提升跨语言与低资源场景性能。**

- **链接: [http://arxiv.org/pdf/2511.07025v1](http://arxiv.org/pdf/2511.07025v1)**

> **作者:** Yauhen Babakhin; Radek Osmulski; Ronay Ak; Gabriel Moreira; Mengyao Xu; Benedikt Schifferer; Bo Liu; Even Oldridge
>
> **摘要:** We introduce llama-embed-nemotron-8b, an open-weights text embedding model that achieves state-of-the-art performance on the Multilingual Massive Text Embedding Benchmark (MMTEB) leaderboard as of October 21, 2025. While recent models show strong performance, their training data or methodologies are often not fully disclosed. We aim to address this by developing a fully open-source model, publicly releasing its weights and detailed ablation studies, and planning to share the curated training datasets. Our model demonstrates superior performance across all major embedding tasks -- including retrieval, classification and semantic textual similarity (STS) -- and excels in challenging multilingual scenarios, such as low-resource languages and cross-lingual setups. This state-of-the-art performance is driven by a novel data mix of 16.1 million query-document pairs, split between 7.7 million samples from public datasets and 8.4 million synthetically generated examples from various open-weight LLMs. One of our key contributions is a detailed ablation study analyzing core design choices, including a comparison of contrastive loss implementations, an evaluation of synthetic data generation (SDG) strategies, and the impact of model merging. The llama-embed-nemotron-8b is an instruction-aware model, supporting user-defined instructions to enhance performance for specific use-cases. This combination of top-tier performance, broad applicability, and user-driven flexibility enables it to serve as a universal text embedding solution.
>
---
#### [new 082] Who Is the Story About? Protagonist Entity Recognition in News
- **分类: cs.CL**

- **简介: 该论文提出“主角实体识别”（PER）任务，旨在从新闻中识别主导叙事的核心组织实体，解决传统NER忽略叙事重要性的问题。通过专家标注与LLM协作，构建数据集并验证LLM可自动识别叙事主角，实现可扩展的叙事理解。**

- **链接: [http://arxiv.org/pdf/2511.07296v1](http://arxiv.org/pdf/2511.07296v1)**

> **作者:** Jorge Gabín; M. Eduardo Ares; Javier Parapar
>
> **摘要:** News articles often reference numerous organizations, but traditional Named Entity Recognition (NER) treats all mentions equally, obscuring which entities genuinely drive the narrative. This limits downstream tasks that rely on understanding event salience, influence, or narrative focus. We introduce Protagonist Entity Recognition (PER), a task that identifies the organizations that anchor a news story and shape its main developments. To validate PER, we compare he predictions of Large Language Models (LLMs) against annotations from four expert annotators over a gold corpus, establishing both inter-annotator consistency and human-LLM agreement. Leveraging these findings, we use state-of-the-art LLMs to automatically label large-scale news collections through NER-guided prompting, generating scalable, high-quality supervision. We then evaluate whether other LLMs, given reduced context and without explicit candidate guidance, can still infer the correct protagonists. Our results demonstrate that PER is a feasible and meaningful extension to narrative-centered information extraction, and that guided LLMs can approximate human judgments of narrative importance at scale.
>
---
#### [new 083] Confidence-Guided Stepwise Model Routing for Cost-Efficient Reasoning
- **分类: cs.CL**

- **简介: 该论文提出STEER，一种无需外部模型的域无关推理路由方法，利用小模型的置信度分数在每一步动态决定是否调用大模型，以在降低计算成本的同时保持高准确率。**

- **链接: [http://arxiv.org/pdf/2511.06190v1](http://arxiv.org/pdf/2511.06190v1)**

> **作者:** Sangmook Lee; Dohyung Kim; Hyukhun Koh; Nakyeong Yang; Kyomin Jung
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Recent advances in Large Language Models (LLMs) - particularly model scaling and test-time techniques - have greatly enhanced the reasoning capabilities of language models at the expense of higher inference costs. To lower inference costs, prior works train router models or deferral mechanisms that allocate easy queries to a small, efficient model, while forwarding harder queries to larger, more expensive models. However, these trained router models often lack robustness under domain shifts and require expensive data synthesis techniques such as Monte Carlo rollouts to obtain sufficient ground-truth routing labels for training. In this work, we propose Confidence-Guided Stepwise Model Routing for Cost-Efficient Reasoning (STEER), a domain-agnostic framework that performs fine-grained, step-level routing between smaller and larger LLMs without utilizing external models. STEER leverages confidence scores from the smaller model's logits prior to generating a reasoning step, so that the large model is invoked only when necessary. Extensive evaluations using different LLMs on a diverse set of challenging benchmarks across multiple domains such as Mathematical Reasoning, Multi-Hop QA, and Planning tasks indicate that STEER achieves competitive or enhanced accuracy while reducing inference costs (up to +20% accuracy with 48% less FLOPs compared to solely using the larger model on AIME), outperforming baselines that rely on trained external modules. Our results establish model-internal confidence as a robust, domain-agnostic signal for model routing, offering a scalable pathway for efficient LLM deployment.
>
---
#### [new 084] SPOT: An Annotated French Corpus and Benchmark for Detecting Critical Interventions in Online Conversations
- **分类: cs.CL; cs.CY**

- **简介: 论文提出SPOT，首个标注法语社交评论语料库，将“停止点”这一社会学概念转化为二分类NLP任务，用于检测隐性批判性回应。实验表明微调CamemBERT模型优于LLM，上下文元数据可提升性能。公开数据与代码促进可复现研究。**

- **链接: [http://arxiv.org/pdf/2511.07405v1](http://arxiv.org/pdf/2511.07405v1)**

> **作者:** Manon Berriche; Célia Nouri; Chloé Clavel; Jean-Philippe Cointet
>
> **摘要:** We introduce SPOT (Stopping Points in Online Threads), the first annotated corpus translating the sociological concept of stopping point into a reproducible NLP task. Stopping points are ordinary critical interventions that pause or redirect online discussions through a range of forms (irony, subtle doubt or fragmentary arguments) that frameworks like counterspeech or social correction often overlook. We operationalize this concept as a binary classification task and provide reliable annotation guidelines. The corpus contains 43,305 manually annotated French Facebook comments linked to URLs flagged as false information by social media users, enriched with contextual metadata (article, post, parent comment, page or group, and source). We benchmark fine-tuned encoder models (CamemBERT) and instruction-tuned LLMs under various prompting strategies. Results show that fine-tuned encoders outperform prompted LLMs in F1 score by more than 10 percentage points, confirming the importance of supervised learning for emerging non-English social media tasks. Incorporating contextual metadata further improves encoder models F1 scores from 0.75 to 0.78. We release the anonymized dataset, along with the annotation guidelines and code in our code repository, to foster transparency and reproducible research.
>
---
#### [new 085] FlowMM: Cross-Modal Information Flow Guided KV Cache Merging for Efficient Multimodal Context Inference
- **分类: cs.CL**

- **简介: FlowMM针对多模态大模型中KV缓存冗余问题，提出跨模态信息流引导的自适应合并方法，通过敏感性感知匹配在压缩80%-95%缓存的同时保持生成质量。**

- **链接: [http://arxiv.org/pdf/2511.05534v1](http://arxiv.org/pdf/2511.05534v1)**

> **作者:** Kunxi Li; Yufan Xiong; Zhonghua Jiang; Yiyun Zhou; Zhaode Wang; Chengfei Lv; Shengyu Zhang
>
> **摘要:** Traditional KV cache eviction strategies, which discard less critical KV-pairs based on attention scores, often degrade generation quality, causing context loss or hallucinations. Recent efforts shift toward KV merging, merging eviction tokens with retention tokens based on similarity. However, in multimodal scenarios, distributional biases across modality tokens and attentional biases in cross-modal interactions limit its effectiveness. This work introduces FlowMM, an adaptive framework for cross-modal information flow-guided multimodal KV cache merging. FlowMM leverages cross-modal information flow to dynamically apply layer-specific merging strategies, capturing modality-specific patterns while preserving contextual integrity. Furthermore, we introduce a sensitivity-adaptive token matching mechanism that jointly evaluates token similarity and task-critical sensitivity, merging low-risk tokens while safeguarding high-sensitivity ones. Extensive experiments across diverse leading MLLMs show that FlowMM reduces KV cache memory by 80% to 95% and decoding latency by 1.3-1.8x, while maintaining competitive task performance.
>
---
#### [new 086] Retrieval-Augmented Generation in Medicine: A Scoping Review of Technical Implementations, Clinical Applications, and Ethical Considerations
- **分类: cs.CL; cs.AI**

- **简介: 该论文为综述研究，旨在梳理医学领域检索增强生成（RAG）的技术实现、临床应用与伦理问题，发现当前研究依赖公开数据、英语模型，缺乏临床验证与多语言支持，亟需提升可信度与全球适用性。**

- **链接: [http://arxiv.org/pdf/2511.05901v1](http://arxiv.org/pdf/2511.05901v1)**

> **作者:** Rui Yang; Matthew Yu Heng Wong; Huitao Li; Xin Li; Wentao Zhu; Jingchi Liao; Kunyu Yu; Jonathan Chong Kai Liew; Weihao Xuan; Yingjian Chen; Yuhe Ke; Jasmine Chiat Ling Ong; Douglas Teodoro; Chuan Hong; Daniel Shi Wei Ting; Nan Liu
>
> **摘要:** The rapid growth of medical knowledge and increasing complexity of clinical practice pose challenges. In this context, large language models (LLMs) have demonstrated value; however, inherent limitations remain. Retrieval-augmented generation (RAG) technologies show potential to enhance their clinical applicability. This study reviewed RAG applications in medicine. We found that research primarily relied on publicly available data, with limited application in private data. For retrieval, approaches commonly relied on English-centric embedding models, while LLMs were mostly generic, with limited use of medical-specific LLMs. For evaluation, automated metrics evaluated generation quality and task performance, whereas human evaluation focused on accuracy, completeness, relevance, and fluency, with insufficient attention to bias and safety. RAG applications were concentrated on question answering, report generation, text summarization, and information extraction. Overall, medical RAG remains at an early stage, requiring advances in clinical validation, cross-linguistic adaptation, and support for low-resource settings to enable trustworthy and responsible global use.
>
---
#### [new 087] MedVoiceBias: A Controlled Study of Audio LLM Behavior in Clinical Decision-Making
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究音频大模型在临床决策中的偏见问题，发现语音特征（年龄、性别、情绪）显著影响医疗建议，导致高达35%的决策差异，揭示音频模态引入的公平性风险，呼吁构建偏见感知架构。**

- **链接: [http://arxiv.org/pdf/2511.06592v1](http://arxiv.org/pdf/2511.06592v1)**

> **作者:** Zhi Rui Tam; Yun-Nung Chen
>
> **摘要:** As large language models transition from text-based interfaces to audio interactions in clinical settings, they might introduce new vulnerabilities through paralinguistic cues in audio. We evaluated these models on 170 clinical cases, each synthesized into speech from 36 distinct voice profiles spanning variations in age, gender, and emotion. Our findings reveal a severe modality bias: surgical recommendations for audio inputs varied by as much as 35% compared to identical text-based inputs, with one model providing 80% fewer recommendations. Further analysis uncovered age disparities of up to 12% between young and elderly voices, which persisted in most models despite chain-of-thought prompting. While explicit reasoning successfully eliminated gender bias, the impact of emotion was not detected due to poor recognition performance. These results demonstrate that audio LLMs are susceptible to making clinical decisions based on a patient's voice characteristics rather than medical evidence, a flaw that risks perpetuating healthcare disparities. We conclude that bias-aware architectures are essential and urgently needed before the clinical deployment of these models.
>
---
#### [new 088] Discourse Graph Guided Document Translation with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TransGraph，用于文档机器翻译任务，解决长文本中语篇连贯性差与上下文依赖建模难的问题。通过构建语篇图结构，引导大模型局部条件化翻译，显著提升质量与术语一致性，同时降低计算开销。**

- **链接: [http://arxiv.org/pdf/2511.07230v1](http://arxiv.org/pdf/2511.07230v1)**

> **作者:** Viet-Thanh Pham; Minghan Wang; Hao-Han Liao; Thuy-Trang Vu
>
> **摘要:** Adapting large language models to full document translation remains challenging due to the difficulty of capturing long-range dependencies and preserving discourse coherence throughout extended texts. While recent agentic machine translation systems mitigate context window constraints through multi-agent orchestration and persistent memory, they require substantial computational resources and are sensitive to memory retrieval strategies. We introduce TransGraph, a discourse-guided framework that explicitly models inter-chunk relationships through structured discourse graphs and selectively conditions each translation segment on relevant graph neighbourhoods rather than relying on sequential or exhaustive context. Across three document-level MT benchmarks spanning six languages and diverse domains, TransGraph consistently surpasses strong baselines in translation quality and terminology consistency while incurring significantly lower token overhead.
>
---
#### [new 089] Ming-UniAudio: Speech LLM for Joint Understanding, Generation and Editing with Unified Representation
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 论文提出Ming-UniAudio，首个统一语音理解、生成与编辑的语音大模型，通过连续音频分词器MingTok-Audio融合语义与声学特征，实现自然语言指令驱动的自由编辑，并构建首个相关评测基准。**

- **链接: [http://arxiv.org/pdf/2511.05516v1](http://arxiv.org/pdf/2511.05516v1)**

> **作者:** Canxiang Yan; Chunxiang Jin; Dawei Huang; Haibing Yu; Han Peng; Hui Zhan; Jie Gao; Jing Peng; Jingdong Chen; Jun Zhou; Kaimeng Ren; Ming Yang; Mingxue Yang; Qiang Xu; Qin Zhao; Ruijie Xiong; Shaoxiong Lin; Xuezhi Wang; Yi Yuan; Yifei Wu; Yongjie Lyu; Zhengyu He; Zhihao Qiu; Zhiqiang Fang; Ziyuan Huang
>
> **备注:** 32 pages, 8 figures
>
> **摘要:** Existing speech models suffer from competing requirements on token representations by understanding and generation tasks. This discrepancy in representation prevents speech language models from performing instruction-based free-form editing. To solve this challenge, we introduce a novel framework that unifies speech understanding, generation, and editing. The core of our unified model is a unified continuous speech tokenizer MingTok-Audio, the first continuous tokenizer to effectively integrate semantic and acoustic features, which makes it suitable for both understanding and generation tasks. Based on this unified continuous audio tokenizer, we developed the speech language model Ming-UniAudio, which achieved a balance between generation and understanding capabilities. Ming-UniAudio sets new state-of-the-art (SOTA) records on 8 out of 12 metrics on the ContextASR benchmark. Notably, for Chinese voice cloning, it achieves a highly competitive Seed-TTS-WER of 0.95. Leveraging this foundational model, we further trained a dedicated speech editing model Ming-UniAudio-Edit, the first speech language model that enables universal, free-form speech editing guided solely by natural language instructions, handling both semantic and acoustic modifications without timestamp condition. To rigorously assess the editing capability and establish a foundation for future research, we introduce Ming-Freeform-Audio-Edit, the first comprehensive benchmark tailored for instruction-based free-form speech editing, featuring diverse scenarios and evaluation dimensions spanning semantic correctness, acoustic quality, and instruction alignment. We open-sourced the continuous audio tokenizer, the unified foundational model, and the free-form instruction-based editing model to facilitate the development of unified audio understanding, generation, and manipulation.
>
---
#### [new 090] Dutch Metaphor Extraction from Cancer Patients' Interviews and Forum Data using LLMs and Human in the Loop
- **分类: cs.CL; cs.CY**

- **简介: 该论文面向荷兰语癌症患者语料，利用LLMs结合人类验证，从访谈与论坛数据中提取隐喻，构建HealthQuote.NL语料库，以改善医患沟通与个性化护理。**

- **链接: [http://arxiv.org/pdf/2511.06427v1](http://arxiv.org/pdf/2511.06427v1)**

> **作者:** Lifeng Han; David Lindevelt; Sander Puts; Erik van Mulligen; Suzan Verberne
>
> **备注:** Ongoing project report, on behalf of 4D PICTURE https://4dpicture.eu/
>
> **摘要:** Metaphors and metaphorical language (MLs) play an important role in healthcare communication between clinicians, patients, and patients' family members. In this work, we focus on Dutch language data from cancer patients. We extract metaphors used by patients using two data sources: (1) cancer patient storytelling interview data and (2) online forum data, including patients' posts, comments, and questions to professionals. We investigate how current state-of-the-art large language models (LLMs) perform on this task by exploring different prompting strategies such as chain of thought reasoning, few-shot learning, and self-prompting. With a human-in-the-loop setup, we verify the extracted metaphors and compile the outputs into a corpus named HealthQuote.NL. We believe the extracted metaphors can support better patient care, for example shared decision making, improved communication between patients and clinicians, and enhanced patient health literacy. They can also inform the design of personalized care pathways. We share prompts and related resources at https://github.com/aaronlifenghan/HealthQuote.NL
>
---
#### [new 091] ConvFill: Model Collaboration for Responsive Conversational Voice Agents
- **分类: cs.CL**

- **简介: 论文提出ConvFill，解决语音助手响应延迟与知识不足的矛盾：用轻量级本地模型实时生成对话，动态融合云端大模型的流式知识，实现低延迟（<200ms）与高准确率（提升36-42%）兼得。**

- **链接: [http://arxiv.org/pdf/2511.07397v1](http://arxiv.org/pdf/2511.07397v1)**

> **作者:** Vidya Srinivas; Zachary Englhardt; Maximus Powers; Shwetak Patel; Vikram Iyer
>
> **摘要:** Deploying conversational voice agents with large language models faces a critical challenge: cloud-based foundation models provide deep reasoning and domain knowledge but introduce latency that disrupts natural conversation, while on-device models respond immediately but lack sophistication. We propose conversational infill, a task where a lightweight on-device model generates contextually appropriate dialogue while seamlessly incorporating streaming knowledge from a powerful backend model. This approach decouples response latency from model capability, enabling systems that feel responsive while accessing the full power of large-scale models. We present ConvFill, a 360M parameter model trained on synthetic multi-domain conversations. Evaluation across multiple backend models shows that conversational infill can be successfully learned, with ConvFill achieving accuracy improvements of 36-42% over standalone small models of the same size while consistently retaining sub-200ms response latencies. Our results demonstrate the promise of this approach for building on-device conversational agents that are both immediately responsive and knowledgeable.
>
---
#### [new 092] Selecting Auxiliary Data via Neural Tangent Kernels for Low-Resource Domains
- **分类: cs.CL**

- **简介: 该论文针对低资源领域数据稀缺问题，提出NTK-Selector，利用神经切线核从大量通用域数据中筛选高价值辅助样本，提升小样本微调性能，显著优于传统方法。**

- **链接: [http://arxiv.org/pdf/2511.07380v1](http://arxiv.org/pdf/2511.07380v1)**

> **作者:** Pingjie Wang; Hongcheng Liu; Yusheng Liao; Ziqing Fan; Yaxin Du; Shuo Tang; Yanfeng Wang; Yu Wang
>
> **备注:** 27 pages
>
> **摘要:** Large language models (LLMs) have achieved remarkable success across widespread tasks, yet their application in low-resource domains remains a significant challenge due to data scarcity and the high risk of overfitting. While in-domain data is limited, there exist vast amounts of similar general-domain data, and our initial findings reveal that they could potentially serve as auxiliary supervision for domain enhancement. This observation leads us to our central research question: \textbf{\textit{how to effectively select the most valuable auxiliary data to maximize domain-specific performance}}, particularly when traditional methods are inapplicable due to a lack of large in-domain data pools or validation sets. To address this, we propose \textbf{NTK-Selector}, a principled and efficient framework for selecting general-domain auxiliary data to enhance domain-specific performance via neural tangent kernels (NTK). Our method tackles two challenges of directly applying NTK to LLMs, theoretical assumptions and prohibitive computational cost, by empirically demonstrating a stable NTK-like behavior in LLMs during LoRA fine-tuning and proposing a Jacobian-free approximation method. Extensive experiments across four low-resource domains (medical, financial, legal, and psychological) demonstrate that NTK-Selector consistently improves downstream performance. Specifically, fine-tuning on 1,000 in-domain samples alone only yielded +0.8 points for Llama3-8B-Instruct and +0.9 points for Qwen3-8B. In contrast, enriching with 9,000 auxiliary samples selected by NTK-Selector led to substantial \textbf{gains of +8.7 and +5.1 points}, which corresponds to a \textbf{10.9x and 5.7x improvement} over the domain-only setting.
>
---
#### [new 093] Ibom NLP: A Step Toward Inclusive Natural Language Processing for Nigeria's Minority Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ibom数据集，填补尼日利亚四种少数民族语言（Anaang、Efik、Ibibio、Oro）在NLP中的数据空白，扩展Flores-200用于机器翻译与SIB-200用于主题分类，揭示现有大模型在这些语言上表现差，但少量样本可提升分类效果。**

- **链接: [http://arxiv.org/pdf/2511.06531v1](http://arxiv.org/pdf/2511.06531v1)**

> **作者:** Oluwadara Kalejaiye; Luel Hagos Beyene; David Ifeoluwa Adelani; Mmekut-Mfon Gabriel Edet; Aniefon Daniel Akpan; Eno-Abasi Urua; Anietie Andy
>
> **备注:** Accepted at IJCNLP-AACL
>
> **摘要:** Nigeria is the most populous country in Africa with a population of more than 200 million people. More than 500 languages are spoken in Nigeria and it is one of the most linguistically diverse countries in the world. Despite this, natural language processing (NLP) research has mostly focused on the following four languages: Hausa, Igbo, Nigerian-Pidgin, and Yoruba (i.e <1% of the languages spoken in Nigeria). This is in part due to the unavailability of textual data in these languages to train and apply NLP algorithms. In this work, we introduce ibom -- a dataset for machine translation and topic classification in four Coastal Nigerian languages from the Akwa Ibom State region: Anaang, Efik, Ibibio, and Oro. These languages are not represented in Google Translate or in major benchmarks such as Flores-200 or SIB-200. We focus on extending Flores-200 benchmark to these languages, and further align the translated texts with topic labels based on SIB-200 classification dataset. Our evaluation shows that current LLMs perform poorly on machine translation for these languages in both zero-and-few shot settings. However, we find the few-shot samples to steadily improve topic classification with more shots.
>
---
#### [new 094] How AI Fails: An Interactive Pedagogical Tool for Demonstrating Dialectal Bias in Automated Toxicity Models
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文研究AI毒性检测模型对非裔美国人英语（AAE）的系统性偏见，发现其评分比标准英语高1.8–8.8倍，并开发交互工具揭示政策阈值如何放大算法歧视，旨在提升公众对AI偏见的批判性认知。**

- **链接: [http://arxiv.org/pdf/2511.06676v1](http://arxiv.org/pdf/2511.06676v1)**

> **作者:** Subhojit Ghimire
>
> **备注:** 9 pages, 5 figures, 4 tables, 14 references
>
> **摘要:** Now that AI-driven moderation has become pervasive in everyday life, we often hear claims that "the AI is biased". While this is often said jokingly, the light-hearted remark reflects a deeper concern. How can we be certain that an online post flagged as "inappropriate" was not simply the victim of a biased algorithm? This paper investigates this problem using a dual approach. First, I conduct a quantitative benchmark of a widely used toxicity model (unitary/toxic-bert) to measure performance disparity between text in African-American English (AAE) and Standard American English (SAE). The benchmark reveals a clear, systematic bias: on average, the model scores AAE text as 1.8 times more toxic and 8.8 times higher for "identity hate". Second, I introduce an interactive pedagogical tool that makes these abstract biases tangible. The tool's core mechanic, a user-controlled "sensitivity threshold," demonstrates that the biased score itself is not the only harm; instead, the more-concerning harm is the human-set, seemingly neutral policy that ultimately operationalises discrimination. This work provides both statistical evidence of disparate impact and a public-facing tool designed to foster critical AI literacy.
>
---
#### [new 095] Multi-Reward GRPO Fine-Tuning for De-biasing Large Language Models: A Study Based on Chinese-Context Discrimination Data
- **分类: cs.CL**

- **简介: 该论文提出多奖励GRPO框架，用于缓解大语言模型在中文语境下的地域、民族、职业等多元偏见，通过合成英文数据构建德伯塔奖励模型，引导策略优化，实现去偏同时保持语句流畅性。**

- **链接: [http://arxiv.org/pdf/2511.06023v1](http://arxiv.org/pdf/2511.06023v1)**

> **作者:** Deng Yixuan; Ji Xiaoqiang
>
> **摘要:** Large Language Models (LLMs) often exhibit implicit biases and discriminatory tendencies that reflect underlying social stereotypes. While recent alignment techniques such as RLHF and DPO have mitigated some of these issues, they remain limited in addressing culturally specific and multi-dimensional forms of discrimination. This paper proposes a Multi-Reward Group Relative Policy Optimization (GRPO) framework to fine-tune LLMs toward ethical and bias-free behavior. Our approach constructs a synthetic English-language dataset derived from Chinese-context discrimination categories, including regional, ethnic, and occupational biases. Each instance is paired with both neutral and biased responses to train a reward model based on DeBERTa-v3, which provides multi-dimensional reward signals capturing fairness, neutrality, and linguistic quality. The trained reward model then guides GRPO fine-tuning to optimize model outputs along these ethical dimensions. Experimental results demonstrate significant reductions in bias intensity and improved alignment with non-discriminatory standards without compromising fluency or informativeness. This study highlights the effectiveness of GRPO-based multi-reward optimization for de-biasing LLMs and offers a replicable framework for cultural-contextual ethical alignment.
>
---
#### [new 096] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究强化学习中大模型熵坍塌问题，揭示训练策略对熵动态的影响，提出通过调整正负优势token的损失权重来调控熵，提升模型多样性与性能。**

- **链接: [http://arxiv.org/pdf/2511.05993v1](http://arxiv.org/pdf/2511.05993v1)**

> **作者:** Renren Jin; Pengzhi Gao; Yuqi Ren; Zhuowen Han; Tongxuan Zhang; Wuwei Huang; Wei Liu; Jian Luan; Deyi Xiong
>
> **备注:** 16 pages, 11 figures, 3 tables
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has emerged as a predominant approach for enhancing the reasoning capabilities of large language models (LLMs). However, the entropy of LLMs usually collapses during RLVR training, causing premature convergence to suboptimal local minima and hinder further performance improvement. Although various approaches have been proposed to mitigate entropy collapse, a comprehensive study of entropy in RLVR remains lacking. To address this gap, we conduct extensive experiments to investigate the entropy dynamics of LLMs trained with RLVR and analyze how model entropy correlates with response diversity, calibration, and performance across various benchmarks. Our findings reveal that the number of off-policy updates, the diversity of training data, and the clipping thresholds in the optimization objective are critical factors influencing the entropy of LLMs trained with RLVR. Moreover, we theoretically and empirically demonstrate that tokens with positive advantages are the primary contributors to entropy collapse, and that model entropy can be effectively regulated by adjusting the relative loss weights of tokens with positive and negative advantages during training.
>
---
#### [new 097] Better Datasets Start From RefineLab: Automatic Optimization for High-Quality Dataset Refinement
- **分类: cs.CL**

- **简介: 论文提出RefineLab，一种LLM驱动的QA数据集自动优化框架，解决人工构建数据集质量不均问题，在限定token预算下通过智能编辑提升覆盖度、难度平衡与事实一致性，实现高效、可复现的高质量数据集构建。**

- **链接: [http://arxiv.org/pdf/2511.06530v1](http://arxiv.org/pdf/2511.06530v1)**

> **作者:** Xiaonan Luo; Yue Huang; Ping He; Xiangliang Zhang
>
> **摘要:** High-quality Question-Answer (QA) datasets are foundational for reliable Large Language Model (LLM) evaluation, yet even expert-crafted datasets exhibit persistent gaps in domain coverage, misaligned difficulty distributions, and factual inconsistencies. The recent surge in generative model-powered datasets has compounded these quality challenges. In this work, we introduce RefineLab, the first LLM-driven framework that automatically refines raw QA textual data into high-quality datasets under a controllable token-budget constraint. RefineLab takes a set of target quality attributes (such as coverage and difficulty balance) as refinement objectives, and performs selective edits within a predefined token budget to ensure practicality and efficiency. In essence, RefineLab addresses a constrained optimization problem: improving the quality of QA samples as much as possible while respecting resource limitations. With a set of available refinement operations (e.g., rephrasing, distractor replacement), RefineLab takes as input the original dataset, a specified set of target quality dimensions, and a token budget, and determines which refinement operations should be applied to each QA sample. This process is guided by an assignment module that selects optimal refinement strategies to maximize overall dataset quality while adhering to the budget constraint. Experiments demonstrate that RefineLab consistently narrows divergence from expert datasets across coverage, difficulty alignment, factual fidelity, and distractor quality. RefineLab pioneers a scalable, customizable path to reproducible dataset design, with broad implications for LLM evaluation.
>
---
#### [new 098] Stemming Hallucination in Language Models Using a Licensing Oracle
- **分类: cs.CL; cs.AI; cs.LG; cs.LO; 68T50, 68T27, 03B70; I.2.7; I.2.4; H.3.3**

- **简介: 该论文提出“许可预言机”架构，解决语言模型幻觉问题，通过对接知识图谱进行确定性事实验证，强制生成内容符合真实知识，实现零错误输出与高准确率，区别于统计方法。**

- **链接: [http://arxiv.org/pdf/2511.06073v1](http://arxiv.org/pdf/2511.06073v1)**

> **作者:** Simeon Emanuilov; Richard Ackermann
>
> **备注:** 23 pages, 4 figures, 8 tables. Introduces the Licensing Oracle, an architectural solution for eliminating hallucinations in language models through formal SHACL validation against knowledge graphs. All datasets and models are available at https://huggingface.co/collections/s-emanuilov/licensing-oracle-experiments
>
> **摘要:** Language models exhibit remarkable natural language generation capabilities but remain prone to hallucinations, generating factually incorrect information despite producing syntactically coherent responses. This study introduces the Licensing Oracle, an architectural solution designed to stem hallucinations in LMs by enforcing truth constraints through formal validation against structured knowledge graphs. Unlike statistical approaches that rely on data scaling or fine-tuning, the Licensing Oracle embeds a deterministic validation step into the model's generative process, ensuring that only factually accurate claims are made. We evaluated the effectiveness of the Licensing Oracle through experiments comparing it with several state-of-the-art methods, including baseline language model generation, fine-tuning for factual recall, fine-tuning for abstention behavior, and retrieval-augmented generation (RAG). Our results demonstrate that although RAG and fine-tuning improve performance, they fail to eliminate hallucinations. In contrast, the Licensing Oracle achieved perfect abstention precision (AP = 1.0) and zero false answers (FAR-NE = 0.0), ensuring that only valid claims were generated with 89.1% accuracy in factual responses. This work shows that architectural innovations, such as the Licensing Oracle, offer a necessary and sufficient solution for hallucinations in domains with structured knowledge representations, offering guarantees that statistical methods cannot match. Although the Licensing Oracle is specifically designed to address hallucinations in fact-based domains, its framework lays the groundwork for truth-constrained generation in future AI systems, providing a new path toward reliable, epistemically grounded models.
>
---
#### [new 099] Reinforcement Learning Improves Traversal of Hierarchical Knowledge in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究强化学习（RL）对大模型层次知识检索能力的提升，挑战RL损害记忆的主流观点。发现RL通过优化知识遍历路径而非存储内容，显著提升结构化知识召回性能，并通过提示工程验证其机制。**

- **链接: [http://arxiv.org/pdf/2511.05933v1](http://arxiv.org/pdf/2511.05933v1)**

> **作者:** Renfei Zhang; Manasa Kaniselvan; Niloofar Mireshghallah
>
> **备注:** `
>
> **摘要:** Reinforcement learning (RL) is often credited with improving language model reasoning and generalization at the expense of degrading memorized knowledge. We challenge this narrative by observing that RL-enhanced models consistently outperform their base and supervised fine-tuned (SFT) counterparts on pure knowledge recall tasks, particularly those requiring traversal of hierarchical, structured knowledge (e.g., medical codes). We hypothesize these gains stem not from newly acquired data, but from improved procedural skills in navigating and searching existing knowledge hierarchies within the model parameters. To support this hypothesis, we show that structured prompting, which explicitly guides SFTed models through hierarchical traversal, recovers most of the performance gap (reducing 24pp to 7pp on MedConceptsQA for DeepSeek-V3/R1). We further find that while prompting improves final-answer accuracy, RL-enhanced models retain superior ability to recall correct procedural paths on deep-retrieval tasks. Finally our layer-wise internal activation analysis reveals that while factual representations (e.g., activations for the statement "code 57.95 refers to urinary infection") maintain high cosine similarity between SFT and RL models, query representations (e.g., "what is code 57.95") diverge noticeably, indicating that RL primarily transforms how models traverse knowledge rather than the knowledge representation itself.
>
---
#### [new 100] Enhancing Multimodal Misinformation Detection by Replaying the Whole Story from Image Modality Perspective
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文面向多模态虚假信息检测任务，针对图像信息不足问题，提出RETSIMD方法：将文本分段生成补充图像，结合互信息优化与图神经网络融合多模态特征，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2511.06284v1](http://arxiv.org/pdf/2511.06284v1)**

> **作者:** Bing Wang; Ximing Li; Yanjun Wang; Changchun Li; Lin Yuanbo Wu; Buyu Wang; Shengsheng Wang
>
> **备注:** Accepted by AAAI 2026. 13 pages, 6 figures. Code: https://github.com/wangbing1416/RETSIMD
>
> **摘要:** Multimodal Misinformation Detection (MMD) refers to the task of detecting social media posts involving misinformation, where the post often contains text and image modalities. However, by observing the MMD posts, we hold that the text modality may be much more informative than the image modality because the text generally describes the whole event/story of the current post but the image often presents partial scenes only. Our preliminary empirical results indicate that the image modality exactly contributes less to MMD. Upon this idea, we propose a new MMD method named RETSIMD. Specifically, we suppose that each text can be divided into several segments, and each text segment describes a partial scene that can be presented by an image. Accordingly, we split the text into a sequence of segments, and feed these segments into a pre-trained text-to-image generator to augment a sequence of images. We further incorporate two auxiliary objectives concerning text-image and image-label mutual information, and further post-train the generator over an auxiliary text-to-image generation benchmark dataset. Additionally, we propose a graph structure by defining three heuristic relationships between images, and use a graph neural network to generate the fused features. Extensive empirical results validate the effectiveness of RETSIMD.
>
---
#### [new 101] DiagnoLLM: A Hybrid Bayesian Neural Language Framework for Interpretable Disease Diagnosis
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: DiagnoLLM提出一种混合框架，用于可解释的疾病诊断，结合贝叶斯去卷积、eQTL引导的神经网络与LLM生成临床报告，解决AI诊断缺乏透明性问题，在阿尔茨海默病检测中实现高准确率并提升医生与患者理解。**

- **链接: [http://arxiv.org/pdf/2511.05810v1](http://arxiv.org/pdf/2511.05810v1)**

> **作者:** Bowen Xu; Xinyue Zeng; Jiazhen Hu; Tuo Wang; Adithya Kulkarni
>
> **摘要:** Building trustworthy clinical AI systems requires not only accurate predictions but also transparent, biologically grounded explanations. We present \texttt{DiagnoLLM}, a hybrid framework that integrates Bayesian deconvolution, eQTL-guided deep learning, and LLM-based narrative generation for interpretable disease diagnosis. DiagnoLLM begins with GP-unmix, a Gaussian Process-based hierarchical model that infers cell-type-specific gene expression profiles from bulk and single-cell RNA-seq data while modeling biological uncertainty. These features, combined with regulatory priors from eQTL analysis, power a neural classifier that achieves high predictive performance in Alzheimer's Disease (AD) detection (88.0\% accuracy). To support human understanding and trust, we introduce an LLM-based reasoning module that translates model outputs into audience-specific diagnostic reports, grounded in clinical features, attribution signals, and domain knowledge. Human evaluations confirm that these reports are accurate, actionable, and appropriately tailored for both physicians and patients. Our findings show that LLMs, when deployed as post-hoc reasoners rather than end-to-end predictors, can serve as effective communicators within hybrid diagnostic pipelines.
>
---
#### [new 102] Persian Musical Instruments Classification Using Polyphonic Data Augmentation
- **分类: cs.SD; cs.CL**

- **简介: 该论文针对波斯音乐乐器分类任务，解决非西方音乐数据稀缺问题，构建首个波斯传统乐器数据集，并提出文化感知的多声部数据增强方法，结合MERT模型显著提升真实混音场景下的分类性能。**

- **链接: [http://arxiv.org/pdf/2511.05717v1](http://arxiv.org/pdf/2511.05717v1)**

> **作者:** Diba Hadi Esfangereh; Mohammad Hossein Sameti; Sepehr Harfi Moridani; Leili Javidpour; Mahdieh Soleymani Baghshah
>
> **备注:** 9 pages, 2 figures, 4 tables
>
> **摘要:** Musical instrument classification is essential for music information retrieval (MIR) and generative music systems. However, research on non-Western traditions, particularly Persian music, remains limited. We address this gap by introducing a new dataset of isolated recordings covering seven traditional Persian instruments, two common but originally non-Persian instruments (i.e., violin, piano), and vocals. We propose a culturally informed data augmentation strategy that generates realistic polyphonic mixtures from monophonic samples. Using the MERT model (Music undERstanding with large-scale self-supervised Training) with a classification head, we evaluate our approach with out-of-distribution data which was obtained by manually labeling segments of traditional songs. On real-world polyphonic Persian music, the proposed method yielded the best ROC-AUC (0.795), highlighting complementary benefits of tonal and temporal coherence. These results demonstrate the effectiveness of culturally grounded augmentation for robust Persian instrument recognition and provide a foundation for culturally inclusive MIR and diverse music generation systems.
>
---
#### [new 103] On the Analogy between Human Brain and LLMs: Spotting Key Neurons in Grammar Perception
- **分类: q-bio.NC; cs.AI; cs.CL**

- **简介: 该论文研究LLMs中与语法感知相关的关键神经元，验证其类脑机制：通过Llama 3识别与词性标注相关的激活神经元，并构建分类器，证明其可预测词性，揭示LLMs中存在类似人脑的语法表征子空间。**

- **链接: [http://arxiv.org/pdf/2511.06519v1](http://arxiv.org/pdf/2511.06519v1)**

> **作者:** Sanaz Saki Norouzi; Mohammad Masjedi; Pascal Hitzler
>
> **摘要:** Artificial Neural Networks, the building blocks of AI, were inspired by the human brain's network of neurons. Over the years, these networks have evolved to replicate the complex capabilities of the brain, allowing them to handle tasks such as image and language processing. In the realm of Large Language Models, there has been a keen interest in making the language learning process more akin to that of humans. While neuroscientific research has shown that different grammatical categories are processed by different neurons in the brain, we show that LLMs operate in a similar way. Utilizing Llama 3, we identify the most important neurons associated with the prediction of words belonging to different part-of-speech tags. Using the achieved knowledge, we train a classifier on a dataset, which shows that the activation patterns of these key neurons can reliably predict part-of-speech tags on fresh data. The results suggest the presence of a subspace in LLMs focused on capturing part-of-speech tag concepts, resembling patterns observed in lesion studies of the brain in neuroscience.
>
---
#### [new 104] Evaluating Implicit Biases in LLM Reasoning through Logic Grid Puzzles
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出PRIME框架，利用逻辑网格谜题评估大语言模型在推理中隐含的社会偏见（如性别刻板印象），解决现有评测难以捕捉潜性偏见的问题，通过控制变量实验揭示模型倾向支持刻板结论。**

- **链接: [http://arxiv.org/pdf/2511.06160v1](http://arxiv.org/pdf/2511.06160v1)**

> **作者:** Fatima Jahara; Mark Dredze; Sharon Levy
>
> **备注:** 24 pages (including appendix)
>
> **摘要:** While recent safety guardrails effectively suppress overtly biased outputs, subtler forms of social bias emerge during complex logical reasoning tasks that evade current evaluation benchmarks. To fill this gap, we introduce a new evaluation framework, PRIME (Puzzle Reasoning for Implicit Biases in Model Evaluation), that uses logic grid puzzles to systematically probe the influence of social stereotypes on logical reasoning and decision making in LLMs. Our use of logic puzzles enables automatic generation and verification, as well as variability in complexity and biased settings. PRIME includes stereotypical, anti-stereotypical, and neutral puzzle variants generated from a shared puzzle structure, allowing for controlled and fine-grained comparisons. We evaluate multiple model families across puzzle sizes and test the effectiveness of prompt-based mitigation strategies. Focusing our experiments on gender stereotypes, our findings highlight that models consistently reason more accurately when solutions align with stereotypical associations. This demonstrates the significance of PRIME for diagnosing and quantifying social biases perpetuated in the deductive reasoning of LLMs, where fairness is critical.
>
---
#### [new 105] Anchors in the Machine: Behavioral and Attributional Evidence of Anchoring Bias in LLMs
- **分类: cs.AI; cs.CL; econ.GN; q-fin.EC**

- **简介: 该论文研究大语言模型（LLMs）中的锚定偏差，通过对数概率分析与Shapley值归因，验证其非表面模仿，而是内部概率重加权，并提出统一评估指标，揭示模型规模与提示设计对偏差敏感性的影响。**

- **链接: [http://arxiv.org/pdf/2511.05766v1](http://arxiv.org/pdf/2511.05766v1)**

> **作者:** Felipe Valencia-Clavijo
>
> **摘要:** Large language models (LLMs) are increasingly examined as both behavioral subjects and decision systems, yet it remains unclear whether observed cognitive biases reflect surface imitation or deeper probability shifts. Anchoring bias, a classic human judgment bias, offers a critical test case. While prior work shows LLMs exhibit anchoring, most evidence relies on surface-level outputs, leaving internal mechanisms and attributional contributions unexplored. This paper advances the study of anchoring in LLMs through three contributions: (1) a log-probability-based behavioral analysis showing that anchors shift entire output distributions, with controls for training-data contamination; (2) exact Shapley-value attribution over structured prompt fields to quantify anchor influence on model log-probabilities; and (3) a unified Anchoring Bias Sensitivity Score integrating behavioral and attributional evidence across six open-source models. Results reveal robust anchoring effects in Gemma-2B, Phi-2, and Llama-2-7B, with attribution signaling that the anchors influence reweighting. Smaller models such as GPT-2, Falcon-RW-1B, and GPT-Neo-125M show variability, suggesting scale may modulate sensitivity. Attributional effects, however, vary across prompt designs, underscoring fragility in treating LLMs as human substitutes. The findings demonstrate that anchoring bias in LLMs is robust, measurable, and interpretable, while highlighting risks in applied domains. More broadly, the framework bridges behavioral science, LLM safety, and interpretability, offering a reproducible path for evaluating other cognitive biases in LLMs.
>
---
#### [new 106] Injecting Falsehoods: Adversarial Man-in-the-Middle Attacks Undermining Factual Recall in LLMs
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文研究LLM在对抗性中间人攻击下的事实记忆脆弱性，提出Xmera框架通过提示注入干扰问答准确性，发现简单指令攻击成功率高达85.3%，并构建随机森林分类器基于响应不确定性检测攻击，提升用户安全警示能力。**

- **链接: [http://arxiv.org/pdf/2511.05919v1](http://arxiv.org/pdf/2511.05919v1)**

> **作者:** Alina Fastowski; Bardh Prenkaj; Yuxiao Li; Gjergji Kasneci
>
> **摘要:** LLMs are now an integral part of information retrieval. As such, their role as question answering chatbots raises significant concerns due to their shown vulnerability to adversarial man-in-the-middle (MitM) attacks. Here, we propose the first principled attack evaluation on LLM factual memory under prompt injection via Xmera, our novel, theory-grounded MitM framework. By perturbing the input given to "victim" LLMs in three closed-book and fact-based QA settings, we undermine the correctness of the responses and assess the uncertainty of their generation process. Surprisingly, trivial instruction-based attacks report the highest success rate (up to ~85.3%) while simultaneously having a high uncertainty for incorrectly answered questions. To provide a simple defense mechanism against Xmera, we train Random Forest classifiers on the response uncertainty levels to distinguish between attacked and unattacked queries (average AUC of up to ~96%). We believe that signaling users to be cautious about the answers they receive from black-box and potentially corrupt LLMs is a first checkpoint toward user cyberspace safety.
>
---
#### [new 107] Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对多模态大模型后训练中样本难度未量化、感知与推理未协同优化的问题，提出PISM与CMAB两种难度感知采样策略，构建分层训练框架，无需SFT即可提升模型性能。**

- **链接: [http://arxiv.org/pdf/2511.06722v1](http://arxiv.org/pdf/2511.06722v1)**

> **作者:** Jianyu Qi; Ding Zou; Wenrui Yan; Rui Ma; Jiaxu Li; Zhijie Zheng; Zhiguo Yang; Rongchang Zhao
>
> **备注:** Accpeted by AAAI 2026
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have spurred significant progress in Chain-of-Thought (CoT) reasoning. Building on the success of Deepseek-R1, researchers extended multimodal reasoning to post-training paradigms based on reinforcement learning (RL), focusing predominantly on mathematical datasets. However, existing post-training paradigms tend to neglect two critical aspects: (1) The lack of quantifiable difficulty metrics capable of strategically screening samples for post-training optimization. (2) Suboptimal post-training paradigms that fail to jointly optimize perception and reasoning capabilities. To address this gap, we propose two novel difficulty-aware sampling strategies: Progressive Image Semantic Masking (PISM) quantifies sample hardness through systematic image degradation, while Cross-Modality Attention Balance (CMAB) assesses cross-modal interaction complexity via attention distribution analysis. Leveraging these metrics, we design a hierarchical training framework that incorporates both GRPO-only and SFT+GRPO hybrid training paradigms, and evaluate them across six benchmark datasets. Experiments demonstrate consistent superiority of GRPO applied to difficulty-stratified samples compared to conventional SFT+GRPO pipelines, indicating that strategic data sampling can obviate the need for supervised fine-tuning while improving model accuracy. Our code will be released at https://github.com/qijianyu277/DifficultySampling.
>
---
#### [new 108] Mixtures of SubExperts for Large Language Continual Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文面向大语言模型持续学习任务，解决参数效率与灾难性遗忘间的权衡问题，提出Mixtures of SubExperts（MoSEs），通过任务特定路由动态组合稀疏子专家，实现知识隔离、迁移与亚线性参数增长。**

- **链接: [http://arxiv.org/pdf/2511.06237v1](http://arxiv.org/pdf/2511.06237v1)**

> **作者:** Haeyong Kang
>
> **摘要:** Adapting Large Language Models (LLMs) to a continuous stream of tasks is a critical yet challenging endeavor. While Parameter-Efficient Fine-Tuning (PEFT) methods have become a standard for this, they face a fundamental dilemma in continual learning. Reusing a single set of PEFT parameters for new tasks often leads to catastrophic forgetting of prior knowledge. Conversely, allocating distinct parameters for each task prevents forgetting but results in a linear growth of the model's size and fails to facilitate knowledge transfer between related tasks. To overcome these limitations, we propose a novel adaptive PEFT method referred to as \textit{Mixtures of SubExperts (MoSEs)}, a novel continual learning framework designed for minimal forgetting and efficient scalability. MoSEs integrate a sparse Mixture of SubExperts into the transformer layers, governed by a task-specific routing mechanism. This architecture allows the model to isolate and protect knowledge within dedicated SubExperts, thereby minimizing parameter interference and catastrophic forgetting. Crucially, the router can adaptively select and combine previously learned sparse parameters for new tasks, enabling effective knowledge transfer while ensuring that the model's capacity grows sublinearly. We evaluate MoSEs on the comprehensive TRACE benchmark datasets. Our experiments demonstrate that MoSEs significantly outperform conventional continual learning approaches in both knowledge retention and scalability to new tasks, achieving state-of-the-art performance with substantial memory and computational savings.
>
---
#### [new 109] CG-TTRL: Context-Guided Test-Time Reinforcement Learning for On-Device Large Language Models
- **分类: cs.LG; cs.CL; I.2.7; I.5.4**

- **简介: 该论文提出CG-TTRL，面向设备端大模型的测试时适应任务，解决传统TTRL忽视上下文引导的问题，通过动态融合上下文优化采样与奖励机制，在数学与科学问答任务中显著提升准确率与训练效率。**

- **链接: [http://arxiv.org/pdf/2511.06430v1](http://arxiv.org/pdf/2511.06430v1)**

> **作者:** Peyman Hosseini; Ondrej Bohdal; Taha Ceritli; Ignacio Castro; Matthew Purver; Mete Ozay; Umberto Michieli
>
> **备注:** 12 pages, 7 Figures, 4 Tables
>
> **摘要:** Test-time Reinforcement Learning (TTRL) has shown promise in adapting foundation models for complex tasks at test-time, resulting in large performance improvements. TTRL leverages an elegant two-phase sampling strategy: first, multi-sampling derives a pseudo-label via majority voting, while subsequent downsampling and reward-based fine-tuning encourages the model to explore and learn diverse valid solutions, with the pseudo-label modulating the reward signal. Meanwhile, in-context learning has been widely explored at inference time and demonstrated the ability to enhance model performance without weight updates. However, TTRL's two-phase sampling strategy under-utilizes contextual guidance, which can potentially improve pseudo-label accuracy in the initial exploitation phase while regulating exploration in the second. To address this, we propose context-guided TTRL (CG-TTRL), integrating context dynamically into both sampling phases and propose a method for efficient context selection for on-device applications. Our evaluations on mathematical and scientific QA benchmarks show CG-TTRL outperforms TTRL (e.g. additional 7% relative accuracy improvement over TTRL), while boosting efficiency by obtaining strong performance after only a few steps of test-time training (e.g. 8% relative improvement rather than 1% over TTRL after 3 steps).
>
---
#### [new 110] Reasoning with Confidence: Efficient Verification of LLM Reasoning Steps via Uncertainty Heads
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出轻量级不确定性头（UHeads），利用冻结LLM的内部状态自动量化推理步骤的置信度，实现高效、无需标注的步骤级验证，性能媲美大得多的PRM模型，推动可扩展的LLM自省机制。**

- **链接: [http://arxiv.org/pdf/2511.06209v1](http://arxiv.org/pdf/2511.06209v1)**

> **作者:** Jingwei Ni; Ekaterina Fadeeva; Tianyi Wu; Mubashara Akhtar; Jiaheng Zhang; Elliott Ash; Markus Leippold; Timothy Baldwin; See-Kiong Ng; Artem Shelmanov; Mrinmaya Sachan
>
> **备注:** Preprint under review
>
> **摘要:** Solving complex tasks usually requires LLMs to generate long multi-step reasoning chains. Previous work has shown that verifying the correctness of individual reasoning steps can further improve the performance and efficiency of LLMs on such tasks and enhance solution interpretability. However, existing verification approaches, such as Process Reward Models (PRMs), are either computationally expensive, limited to specific domains, or require large-scale human or model-generated annotations. Thus, we propose a lightweight alternative for step-level reasoning verification based on data-driven uncertainty scores. We train transformer-based uncertainty quantification heads (UHeads) that use the internal states of a frozen LLM to estimate the uncertainty of its reasoning steps during generation. The approach is fully automatic: target labels are generated either by another larger LLM (e.g., DeepSeek R1) or in a self-supervised manner by the original model itself. UHeads are both effective and lightweight, containing less than 10M parameters. Across multiple domains, including mathematics, planning, and general knowledge question answering, they match or even surpass the performance of PRMs that are up to 810x larger. Our findings suggest that the internal states of LLMs encode their uncertainty and can serve as reliable signals for reasoning verification, offering a promising direction toward scalable and generalizable introspective LLMs.
>
---
#### [new 111] MCP-RiskCue: Can LLM infer risk information from MCP server System Logs?
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出MCP-RiskCue基准，评估LLM从恶意MCP服务器系统日志中识别安全风险的能力，构建1800条合成日志与2421条对话数据，发现RLVR方法（如GRPO）显著提升检测精度，优于SFT与大模型。**

- **链接: [http://arxiv.org/pdf/2511.05867v1](http://arxiv.org/pdf/2511.05867v1)**

> **作者:** Jiayi Fu; Qiyao Sun
>
> **摘要:** Large language models (LLMs) demonstrate strong capabilities in solving complex tasks when integrated with external tools. The Model Context Protocol (MCP) has become a standard interface for enabling such tool-based interactions. However, these interactions introduce substantial security concerns, particularly when the MCP server is compromised or untrustworthy. While prior benchmarks primarily focus on prompt injection attacks or analyze the vulnerabilities of LLM MCP interaction trajectories, limited attention has been given to the underlying system logs associated with malicious MCP servers. To address this gap, we present the first synthetic benchmark for evaluating LLMs ability to identify security risks from system logs. We define nine categories of MCP server risks and generate 1,800 synthetic system logs using ten state-of-the-art LLMs. These logs are embedded in the return values of 243 curated MCP servers, yielding a dataset of 2,421 chat histories for training and 471 queries for evaluation. Our pilot experiments reveal that smaller models often fail to detect risky system logs, leading to high false negatives. While models trained with supervised fine-tuning (SFT) tend to over-flag benign logs, resulting in elevated false positives, Reinforcement Learning from Verifiable Reward (RLVR) offers a better precision-recall balance. In particular, after training with Group Relative Policy Optimization (GRPO), Llama3.1-8B-Instruct achieves 83% accuracy, surpassing the best-performing large remote model by 9 percentage points. Fine-grained, per-category analysis further underscores the effectiveness of reinforcement learning in enhancing LLM safety within the MCP framework. Code and data are available at: https://github.com/PorUna-byte/MCP-Guard/tree/master
>
---
#### [new 112] Factual and Musical Evaluation Metrics for Music Language Models
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文针对音乐语言模型（Music LMs）评估不足的问题，指出传统指标仅衡量语言流畅性而非事实正确性，提出新的音乐领域通用评估指标与事实性评估框架，以准确衡量模型回答的准确性。**

- **链接: [http://arxiv.org/pdf/2511.05550v1](http://arxiv.org/pdf/2511.05550v1)**

> **作者:** Daniel Chenyu Lin; Michael Freeman; John Thickstun
>
> **备注:** 18 pages; first submission
>
> **摘要:** Music language models (Music LMs), like vision language models, leverage multimodal representations to answer natural language queries about musical audio recordings. Although Music LMs are reportedly improving, we find that current evaluations fail to capture whether their answers are correct. Specifically, for all Music LMs that we examine, widely-used evaluation metrics such as BLEU, METEOR, and BERTScore fail to measure anything beyond linguistic fluency of the model's responses. To measure the true performance of Music LMs, we propose (1) a better general-purpose evaluation metric for Music LMs adapted to the music domain and (2) a factual evaluation framework to quantify the correctness of a Music LM's responses. Our framework is agnostic to the modality of the question-answering model and could be generalized to quantify performance in other open-ended question-answering domains. We use open datasets in our experiments and will release all code on publication.
>
---
#### [new 113] FPGA or GPU? Analyzing comparative research for application-specific guidance
- **分类: cs.AR; cs.CL; cs.DC; cs.PL**

- **简介: 该论文属于综述与指导性研究，旨在解决FPGA与GPU在应用选择上的决策盲区。通过系统分析现有研究，提炼二者在性能、能效和可编程性上的优劣，为特定应用提供加速器选型指南。**

- **链接: [http://arxiv.org/pdf/2511.06565v1](http://arxiv.org/pdf/2511.06565v1)**

> **作者:** Arnab A Purkayastha; Jay Tharwani; Shobhit Aggarwal
>
> **备注:** 7 pages
>
> **摘要:** The growing complexity of computational workloads has amplified the need for efficient and specialized hardware accelerators. Field Programmable Gate Arrays (FPGAs) and Graphics Processing Units (GPUs) have emerged as prominent solutions, each excelling in specific domains. Although there is substantial research comparing FPGAs and GPUs, most of the work focuses primarily on performance metrics, offering limited insight into the specific types of applications that each accelerator benefits the most. This paper aims to bridge this gap by synthesizing insights from various research articles to guide users in selecting the appropriate accelerator for domain-specific applications. By categorizing the reviewed studies and analyzing key performance metrics, this work highlights the strengths, limitations, and ideal use cases for FPGAs and GPUs. The findings offer actionable recommendations, helping researchers and practitioners navigate trade-offs in performance, energy efficiency, and programmability.
>
---
#### [new 114] ScRPO: From Errors to Insights
- **分类: cs.AI; cs.CL**

- **简介: 论文提出ScRPO框架，用于提升大语言模型在数学推理任务中的表现，通过“试错收集错误”与“自我反思修正”两阶段实现自纠正学习，无需外部反馈即可显著提升模型准确率。**

- **链接: [http://arxiv.org/pdf/2511.06065v1](http://arxiv.org/pdf/2511.06065v1)**

> **作者:** Lianrui Li; Dakuan Lu; Jiawei Shao; Chi Zhang; Xuelong Li
>
> **摘要:** We propose Self-correction Relative Policy Optimization (ScRPO), a novel reinforcement learning framework designed to enhance large language models on challenging mathemati- cal problems by leveraging self-reflection and error correction. Our approach consists of two stages: (1) Trial-and-error learning stage: training the model with GRPO and collect- ing incorrect answers along with their cor- responding questions in an error pool; (2) Self-correction learning stage: guiding the model to reflect on why its previous an- swers were wrong. Extensive experiments across multiple math reasoning benchmarks, including AIME, AMC, Olympiad, MATH- 500, GSM8k, using Deepseek-Distill-Qwen- 1.5B and Deepseek-Distill-Qwen-7B. The ex- perimental results demonstrate that ScRPO consistently outperforms several post-training methods. These findings highlight ScRPO as a promising paradigm for enabling language models to self-improve on difficult tasks with limited external feedback, paving the way to- ward more reliable and capable AI systems.
>
---
#### [new 115] The Imperfect Learner: Incorporating Developmental Trajectories in Memory-based Student Simulation
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出一种基于记忆的学生活动模拟框架，通过层次化记忆与结构化知识表征，融入发展轨迹、元认知与个性特征，解决现有模拟忽视学习渐进性与认知局限的问题，更真实还原学生学习过程。**

- **链接: [http://arxiv.org/pdf/2511.05903v1](http://arxiv.org/pdf/2511.05903v1)**

> **作者:** Zhengyuan Liu; Stella Xin Yin; Bryan Chen Zhengyu Tan; Roy Ka-Wei Lee; Guimei Liu; Dion Hoe-Lian Goh; Wenya Wang; Nancy F. Chen
>
> **摘要:** User simulation is important for developing and evaluating human-centered AI, yet current student simulation in educational applications has significant limitations. Existing approaches focus on single learning experiences and do not account for students' gradual knowledge construction and evolving skill sets. Moreover, large language models are optimized to produce direct and accurate responses, making it challenging to represent the incomplete understanding and developmental constraints that characterize real learners. In this paper, we introduce a novel framework for memory-based student simulation that incorporates developmental trajectories through a hierarchical memory mechanism with structured knowledge representation. The framework also integrates metacognitive processes and personality traits to enrich the individual learner profiling, through dynamical consolidation of both cognitive development and personal learning characteristics. In practice, we implement a curriculum-aligned simulator grounded on the Next Generation Science Standards. Experimental results show that our approach can effectively reflect the gradual nature of knowledge development and the characteristic difficulties students face, providing a more accurate representation of learning processes.
>
---
#### [new 116] Tiny Model, Big Logic: Diversity-Driven Optimization Elicits Large-Model Reasoning Ability in VibeThinker-1.5B
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出VibeThinker-1.5B，通过多样性驱动优化（SSP框架）在1.5B参数小模型上实现媲美超大模型的推理能力，解决小模型推理弱的问题，显著降低训练成本并挑战参数规模依赖的主流范式。**

- **链接: [http://arxiv.org/pdf/2511.06221v1](http://arxiv.org/pdf/2511.06221v1)**

> **作者:** Sen Xu; Yi Zhou; Wei Wang; Jixin Min; Zhibin Yin; Yingwei Dai; Shixi Liu; Lianyu Pang; Yirong Chen; Junlin Zhang
>
> **摘要:** Challenging the prevailing consensus that small models inherently lack robust reasoning, this report introduces VibeThinker-1.5B, a 1.5B-parameter dense model developed via our Spectrum-to-Signal Principle (SSP). This challenges the prevailing approach of scaling model parameters to enhance capabilities, as seen in models like DeepSeek R1 (671B) and Kimi k2 (>1T). The SSP framework first employs a Two-Stage Diversity-Exploring Distillation (SFT) to generate a broad spectrum of solutions, followed by MaxEnt-Guided Policy Optimization (RL) to amplify the correct signal. With a total training cost of only $7,800, VibeThinker-1.5B demonstrates superior reasoning capabilities compared to closed-source models like Magistral Medium and Claude Opus 4, and performs on par with open-source models like GPT OSS-20B Medium. Remarkably, it surpasses the 400x larger DeepSeek R1 on three math benchmarks: AIME24 (80.3 vs. 79.8), AIME25 (74.4 vs. 70.0), and HMMT25 (50.4 vs. 41.7). This is a substantial improvement over its base model (6.7, 4.3, and 0.6, respectively). On LiveCodeBench V6, it scores 51.1, outperforming Magistral Medium's 50.3 and its base model's 0.0. These findings demonstrate that small models can achieve reasoning capabilities comparable to large models, drastically reducing training and inference costs and thereby democratizing advanced AI research.
>
---
#### [new 117] Large Language Models Develop Novel Social Biases Through Adaptive Exploration
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型（LLMs）在决策中会自发产生新型社会偏见，而非仅复制人类偏见。通过心理学实验，发现模型因探索不足导致群体歧视，提出多维度干预措施，证实激励探索能有效缓解偏见，揭示LLMs主动塑造社会不公的潜在风险。**

- **链接: [http://arxiv.org/pdf/2511.06148v1](http://arxiv.org/pdf/2511.06148v1)**

> **作者:** Addison J. Wu; Ryan Liu; Xuechunzi Bai; Thomas L. Griffiths
>
> **摘要:** As large language models (LLMs) are adopted into frameworks that grant them the capacity to make real decisions, it is increasingly important to ensure that they are unbiased. In this paper, we argue that the predominant approach of simply removing existing biases from models is not enough. Using a paradigm from the psychology literature, we demonstrate that LLMs can spontaneously develop novel social biases about artificial demographic groups even when no inherent differences exist. These biases result in highly stratified task allocations, which are less fair than assignments by human participants and are exacerbated by newer and larger models. In social science, emergent biases like these have been shown to result from exploration-exploitation trade-offs, where the decision-maker explores too little, allowing early observations to strongly influence impressions about entire demographic groups. To alleviate this effect, we examine a series of interventions targeting model inputs, problem structure, and explicit steering. We find that explicitly incentivizing exploration most robustly reduces stratification, highlighting the need for better multifaceted objectives to mitigate bias. These results reveal that LLMs are not merely passive mirrors of human social biases, but can actively create new ones from experience, raising urgent questions about how these systems will shape societies over time.
>
---
#### [new 118] GRAPH-GRPO-LEX: Contract Graph Modeling and Reinforcement Learning with Group Relative Policy Optimization
- **分类: cs.AI; cs.CL; cs.LG; cs.SE**

- **简介: 该论文将法律合同建模为语义图，提出GRAPH-GRPO-LEX框架，结合LLM与组相对策略优化，自动提取条款实体与关系，挖掘隐含依赖，实现合同分析的自动化与可视化，推动合同审查向智能 linting 转型。**

- **链接: [http://arxiv.org/pdf/2511.06618v1](http://arxiv.org/pdf/2511.06618v1)**

> **作者:** Moriya Dechtiar; Daniel Martin Katz; Mari Sundaresan; Sylvain Jaume; Hongming Wang
>
> **摘要:** Contracts are complex documents featuring detailed formal structures, explicit and implicit dependencies and rich semantic content. Given these document properties, contract drafting and manual examination of contracts have proven to be both arduous and susceptible to errors. This work aims to simplify and automate the task of contract review and analysis using a novel framework for transforming legal contracts into structured semantic graphs, enabling computational analysis and data-driven insights. We introduce a detailed ontology mapping core legal contract elements to their graph-theoretic equivalents of nodes and edges. We then present a reinforcement learning based Large Language Model (LLM) framework for segmentation and extraction of entities and relationships from contracts. Our method, GRAPH-GRPO-LEX, incorporates both LLMs and reinforcement learning with group relative policy optimization (GRPO). By applying a carefully drafted reward function of graph metrics, we demonstrate the ability to automatically identify direct relationships between clauses, and even uncover hidden dependencies. Our introduction of the gated GRPO approach shows a strong learning signal and can move contract analysis from a linear, manual reading process to an easily visualized graph. This allows for a more dynamic analysis, including building the groundwork for contract linting similar to what is now practiced in software engineering.
>
---
#### [new 119] The Few Govern the Many:Unveiling Few-Layer Dominance for Time Series Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究时间序列预测任务，发现大模型存在“缩放悖论”——模型越大性能越差。提出“少数层主导”现象，通过仅保留21%关键层，显著提升精度与推理速度，验证了方法在8个SOTA模型上的普适性。**

- **链接: [http://arxiv.org/pdf/2511.07237v1](http://arxiv.org/pdf/2511.07237v1)**

> **作者:** Xin Qiu; Junlong Tong; Yirong Sun; Yunpu Ma; Xiaoyu Shen
>
> **摘要:** Large-scale models are at the forefront of time series (TS) forecasting, dominated by two paradigms: fine-tuning text-based Large Language Models (LLM4TS) and training Time Series Foundation Models (TSFMs) from scratch. Both approaches share a foundational assumption that scaling up model capacity and data volume leads to improved performance. However, we observe a \textit{\textbf{scaling paradox}} in TS models, revealing a puzzling phenomenon that larger models do \emph{NOT} achieve better performance. Through extensive experiments on two model families across four scales (100M to 1.7B parameters) and diverse data (up to 6B observations), we rigorously confirm that the scaling paradox is a pervasive issue. We then diagnose its root cause by analyzing internal representations, identifying a phenomenon we call \textit{few-layer dominance}: only a small subset of layers are functionally important, while the majority are redundant, under-utilized, and can even distract training. Based on this discovery, we propose a practical method to automatically identify and retain only these dominant layers. In our models, retaining only 21\% of the parameters achieves up to a 12\% accuracy improvement and a 2.7$\times$ inference speedup. We validate the universality of our method on 8 prominent SOTA models (LLM4TS and TSFMs, 90M to 6B), showing that retaining less than 30\% of layers achieves comparable or superior accuracy in over 95\% of tasks.
>
---
#### [new 120] Enhancing Adversarial Robustness of IoT Intrusion Detection via SHAP-Based Attribution Fingerprinting
- **分类: cs.CR; cs.AI; cs.CL; cs.LG; cs.NI; I.2; I.2.6; I.2.7; I.1.1; C.2.2; C.2.6; B.8.1; B.8.2**

- **简介: 该论文面向IoT入侵检测，解决AI模型易受对抗攻击的问题，提出基于SHAP的属性指纹方法，通过提取特征重要性模式区分正常与对抗样本，提升检测鲁棒性与可解释性。**

- **链接: [http://arxiv.org/pdf/2511.06197v1](http://arxiv.org/pdf/2511.06197v1)**

> **作者:** Dilli Prasad Sharma; Liang Xue; Xiaowei Sun; Xiaodong Lin; Pulei Xiong
>
> **摘要:** The rapid proliferation of Internet of Things (IoT) devices has transformed numerous industries by enabling seamless connectivity and data-driven automation. However, this expansion has also exposed IoT networks to increasingly sophisticated security threats, including adversarial attacks targeting artificial intelligence (AI) and machine learning (ML)-based intrusion detection systems (IDS) to deliberately evade detection, induce misclassification, and systematically undermine the reliability and integrity of security defenses. To address these challenges, we propose a novel adversarial detection model that enhances the robustness of IoT IDS against adversarial attacks through SHapley Additive exPlanations (SHAP)-based fingerprinting. Using SHAP's DeepExplainer, we extract attribution fingerprints from network traffic features, enabling the IDS to reliably distinguish between clean and adversarially perturbed inputs. By capturing subtle attribution patterns, the model becomes more resilient to evasion attempts and adversarial manipulations. We evaluated the model on a standard IoT benchmark dataset, where it significantly outperformed a state-of-the-art method in detecting adversarial attacks. In addition to enhanced robustness, this approach improves model transparency and interpretability, thereby increasing trust in the IDS through explainable AI.
>
---
#### [new 121] A Representation Sharpening Framework for Zero Shot Dense Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文针对零样本稠密检索中相似文档语义区分困难的问题，提出一种无训练的表示 sharpening 框架，通过增强文档表示提升区分能力，在多语言数据集上显著超越传统方法，且可通过索引优化避免推理开销。**

- **链接: [http://arxiv.org/pdf/2511.05684v1](http://arxiv.org/pdf/2511.05684v1)**

> **作者:** Dhananjay Ashok; Suraj Nair; Mutasem Al-Darabsah; Choon Hui Teo; Tarun Agarwal; Jonathan May
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Zero-shot dense retrieval is a challenging setting where a document corpus is provided without relevant queries, necessitating a reliance on pretrained dense retrievers (DRs). However, since these DRs are not trained on the target corpus, they struggle to represent semantic differences between similar documents. To address this failing, we introduce a training-free representation sharpening framework that augments a document's representation with information that helps differentiate it from similar documents in the corpus. On over twenty datasets spanning multiple languages, the representation sharpening framework proves consistently superior to traditional retrieval, setting a new state-of-the-art on the BRIGHT benchmark. We show that representation sharpening is compatible with prior approaches to zero-shot dense retrieval and consistently improves their performance. Finally, we address the performance-cost tradeoff presented by our framework and devise an indexing-time approximation that preserves the majority of our performance gains over traditional retrieval, yet suffers no additional inference-time cost.
>
---
#### [new 122] DigiData: Training and Evaluating General-Purpose Mobile Control Agents
- **分类: cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 论文提出DigiData数据集与DigiData-Bench评估基准，解决移动控制代理训练数据不足与评估指标不准确的问题，通过多模态高复杂度任务数据与动态AI评估方法，推动通用移动控制代理的发展。**

- **链接: [http://arxiv.org/pdf/2511.07413v1](http://arxiv.org/pdf/2511.07413v1)**

> **作者:** Yuxuan Sun; Manchen Wang; Shengyi Qian; William R. Wong; Eric Gan; Pierluca D'Oro; Alejandro Castillejo Munoz; Sneha Silwal; Pedro Matias; Nitin Kamra; Satwik Kottur; Nick Raines; Xuanyi Zhao; Joy Chen; Joseph Greer; Andrea Madotto; Allen Bolourchi; James Valori; Kevin Carlberg; Karl Ridgeway; Joseph Tighe
>
> **备注:** Website: https://facebookresearch.github.io/DigiData
>
> **摘要:** AI agents capable of controlling user interfaces have the potential to transform human interaction with digital devices. To accelerate this transformation, two fundamental building blocks are essential: high-quality datasets that enable agents to achieve complex and human-relevant goals, and robust evaluation methods that allow researchers and practitioners to rapidly enhance agent performance. In this paper, we introduce DigiData, a large-scale, high-quality, diverse, multi-modal dataset designed for training mobile control agents. Unlike existing datasets, which derive goals from unstructured interactions, DigiData is meticulously constructed through comprehensive exploration of app features, resulting in greater diversity and higher goal complexity. Additionally, we present DigiData-Bench, a benchmark for evaluating mobile control agents on real-world complex tasks. We demonstrate that the commonly used step-accuracy metric falls short in reliably assessing mobile control agents and, to address this, we propose dynamic evaluation protocols and AI-powered evaluations as rigorous alternatives for agent assessment. Our contributions aim to significantly advance the development of mobile control agents, paving the way for more intuitive and effective human-device interactions.
>
---
#### [new 123] The Role of High-Performance GPU Resources in Large Language Model Based Radiology Imaging Diagnosis
- **分类: q-bio.TO; cs.CL; eess.IV; physics.med-ph**

- **简介: 该论文探讨高性能GPU在大型语言模型驱动的放射影像诊断中的关键作用，解决推理延迟与计算效率问题，系统分析GPU架构与性能指标，并提出优化策略，推动临床级AI部署。**

- **链接: [http://arxiv.org/pdf/2509.16328v2](http://arxiv.org/pdf/2509.16328v2)**

> **作者:** Jyun-Ping Kao
>
> **摘要:** Large-language models (LLMs) are rapidly being applied to radiology, enabling automated image interpretation and report generation tasks. Their deployment in clinical practice requires both high diagnostic accuracy and low inference latency, which in turn demands powerful hardware. High-performance graphical processing units (GPUs) provide the necessary compute and memory throughput to run large LLMs on imaging data. We review modern GPU architectures (e.g. NVIDIA A100/H100, AMD Instinct MI250X/MI300) and key performance metrics of floating-point throughput, memory bandwidth, VRAM capacity. We show how these hardware capabilities affect radiology tasks: for example, generating reports or detecting findings on CheXpert and MIMIC-CXR images is computationally intensive and benefits from GPU parallelism and tensor-core acceleration. Empirical studies indicate that using appropriate GPU resources can reduce inference time and improve throughput. We discuss practical challenges including privacy, deployment, cost, power and optimization strategies: mixed-precision, quantization, compression, and multi-GPU scaling. Finally, we anticipate that next-generation features (8-bit tensor cores, enhanced interconnect) will further enable on-premise and federated radiology AI. Advancing GPU infrastructure is essential for safe, efficient LLM-based radiology diagnostics.
>
---
#### [new 124] HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment
- **分类: cs.CV; cs.CL**

- **简介: HiMo-CLIP面向视觉-语言对齐任务，解决CLIP忽略文本语义层次与单调性的问题，提出HiDe模块提取语义成分，结合MoLo损失函数，实现多粒度、有序的跨模态对齐，显著提升长文本检索性能。**

- **链接: [http://arxiv.org/pdf/2511.06653v1](http://arxiv.org/pdf/2511.06653v1)**

> **作者:** Ruijia Wu; Ping Chen; Fei Shen; Shaoan Zhao; Qiang Hui; Huanlin Gao; Ting Lu; Zhaoxiang Liu; Fang Zhao; Kai Wang; Shiguo Lian
>
> **备注:** Accepted by AAAI 2026 as an Oral Presentation (13 pages, 7 figures, 7 tables)
>
> **摘要:** Contrastive vision-language models like CLIP have achieved impressive results in image-text retrieval by aligning image and text representations in a shared embedding space. However, these models often treat text as flat sequences, limiting their ability to handle complex, compositional, and long-form descriptions. In particular, they fail to capture two essential properties of language: semantic hierarchy, which reflects the multi-level compositional structure of text, and semantic monotonicity, where richer descriptions should result in stronger alignment with visual content.To address these limitations, we propose HiMo-CLIP, a representation-level framework that enhances CLIP-style models without modifying the encoder architecture. HiMo-CLIP introduces two key components: a hierarchical decomposition (HiDe) module that extracts latent semantic components from long-form text via in-batch PCA, enabling flexible, batch-aware alignment across different semantic granularities, and a monotonicity-aware contrastive loss (MoLo) that jointly aligns global and component-level representations, encouraging the model to internalize semantic ordering and alignment strength as a function of textual completeness.These components work in concert to produce structured, cognitively-aligned cross-modal representations. Experiments on multiple image-text retrieval benchmarks show that HiMo-CLIP consistently outperforms strong baselines, particularly under long or compositional descriptions. The code is available at https://github.com/UnicomAI/HiMo-CLIP.
>
---
#### [new 125] Adaptive Testing for Segmenting Watermarked Texts From Language Models
- **分类: stat.ML; cs.CL; cs.LG**

- **简介: 该论文提出一种自适应检测方法，用于分割混合了水印与非水印文本的片段，无需精确提示估计，提升水印文本识别的鲁棒性与准确性。**

- **链接: [http://arxiv.org/pdf/2511.06645v1](http://arxiv.org/pdf/2511.06645v1)**

> **作者:** Xingchi Li; Xiaochi Liu; Guanxun Li
>
> **备注:** 13 pages, 3 figures, accepted for publication in STAT, October 28, 2025
>
> **摘要:** The rapid adoption of large language models (LLMs), such as GPT-4 and Claude 3.5, underscores the need to distinguish LLM-generated text from human-written content to mitigate the spread of misinformation and misuse in education. One promising approach to address this issue is the watermark technique, which embeds subtle statistical signals into LLM-generated text to enable reliable identification. In this paper, we first generalize the likelihood-based LLM detection method of a previous study by introducing a flexible weighted formulation, and further adapt this approach to the inverse transform sampling method. Moving beyond watermark detection, we extend this adaptive detection strategy to tackle the more challenging problem of segmenting a given text into watermarked and non-watermarked substrings. In contrast to the approach in a previous study, which relies on accurate estimation of next-token probabilities that are highly sensitive to prompt estimation, our proposed framework removes the need for precise prompt estimation. Extensive numerical experiments demonstrate that the proposed methodology is both effective and robust in accurately segmenting texts containing a mixture of watermarked and non-watermarked content.
>
---
#### [new 126] MONICA: Real-Time Monitoring and Calibration of Chain-of-Thought Sycophancy in Large Reasoning Models
- **分类: cs.AI; cs.CL**

- **简介: 论文提出MONICA框架，用于实时监测与校正大推理模型在思维链推理过程中出现的谄媚行为，解决传统方法仅依赖最终答案、无法捕捉推理中偏差的问题，实现中间步骤与最终答案的协同抑制。**

- **链接: [http://arxiv.org/pdf/2511.06419v1](http://arxiv.org/pdf/2511.06419v1)**

> **作者:** Jingyu Hu; Shu Yang; Xilin Gong; Hongming Wang; Weiru Liu; Di Wang
>
> **摘要:** Large Reasoning Models (LRMs) suffer from sycophantic behavior, where models tend to agree with users' incorrect beliefs and follow misinformation rather than maintain independent reasoning. This behavior undermines model reliability and poses societal risks. Mitigating LRM sycophancy requires monitoring how this sycophancy emerges during the reasoning trajectory; however, current methods mainly focus on judging based on final answers and correcting them, without understanding how sycophancy develops during reasoning processes. To address this limitation, we propose MONICA, a novel Monitor-guided Calibration framework that monitors and mitigates sycophancy during model inference at the level of reasoning steps, without requiring the model to finish generating its complete answer. MONICA integrates a sycophantic monitor that provides real-time monitoring of sycophantic drift scores during response generation with a calibrator that dynamically suppresses sycophantic behavior when scores exceed predefined thresholds. Extensive experiments across 12 datasets and 3 LRMs demonstrate that our method effectively reduces sycophantic behavior in both intermediate reasoning steps and final answers, yielding robust performance improvements.
>
---
#### [new 127] IterResearch: Rethinking Long-Horizon Agents via Markovian State Reconstruction
- **分类: cs.AI; cs.CL**

- **简介: IterResearch提出一种基于马尔可夫状态重构的迭代式长程研究范式，解决传统方法因上下文膨胀导致的噪声污染与推理退化问题，通过动态摘要与高效策略优化，在多基准上显著超越现有代理并提升大模型推理能力。**

- **链接: [http://arxiv.org/pdf/2511.07327v1](http://arxiv.org/pdf/2511.07327v1)**

> **作者:** Guoxin Chen; Zile Qiao; Xuanzhong Chen; Donglei Yu; Haotian Xu; Wayne Xin Zhao; Ruihua Song; Wenbiao Yin; Huifeng Yin; Liwen Zhang; Kuan Li; Minpeng Liao; Yong Jiang; Pengjun Xie; Fei Huang; Jingren Zhou
>
> **备注:** https://github.com/Alibaba-NLP/DeepResearch
>
> **摘要:** Recent advances in deep-research agents have shown promise for autonomous knowledge construction through dynamic reasoning over external sources. However, existing approaches rely on a mono-contextual paradigm that accumulates all information in a single, expanding context window, leading to context suffocation and noise contamination that limit their effectiveness on long-horizon tasks. We introduce IterResearch, a novel iterative deep-research paradigm that reformulates long-horizon research as a Markov Decision Process with strategic workspace reconstruction. By maintaining an evolving report as memory and periodically synthesizing insights, our approach preserves consistent reasoning capacity across arbitrary exploration depths. We further develop Efficiency-Aware Policy Optimization (EAPO), a reinforcement learning framework that incentivizes efficient exploration through geometric reward discounting and enables stable distributed training via adaptive downsampling. Extensive experiments demonstrate that IterResearch achieves substantial improvements over existing open-source agents with average +14.5pp across six benchmarks and narrows the gap with frontier proprietary systems. Remarkably, our paradigm exhibits unprecedented interaction scaling, extending to 2048 interactions with dramatic performance gains (from 3.5\% to 42.5\%), and serves as an effective prompting strategy, improving frontier models by up to 19.2pp over ReAct on long-horizon tasks. These findings position IterResearch as a versatile solution for long-horizon reasoning, effective both as a trained agent and as a prompting paradigm for frontier models.
>
---
#### [new 128] Simulating Students with Large Language Models: A Review of Architecture, Mechanisms, and Role Modelling in Education with Generative AI
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文综述了利用大语言模型（LLM）模拟学生行为的研究，旨在解决教育中真实 learner 模拟难的问题，系统梳理了LLM在学习风格建模、教学互动与多代理课堂中的应用，并指出其偏差与可靠性挑战，提出生成式AI融入教学设计的未来方向。**

- **链接: [http://arxiv.org/pdf/2511.06078v1](http://arxiv.org/pdf/2511.06078v1)**

> **作者:** Luis Marquez-Carpintero; Alberto Lopez-Sellers; Miguel Cazorla
>
> **摘要:** Simulated Students offer a valuable methodological framework for evaluating pedagogical approaches and modelling diverse learner profiles, tasks which are otherwise challenging to undertake systematically in real-world settings. Recent research has increasingly focused on developing such simulated agents to capture a range of learning styles, cognitive development pathways, and social behaviours. Among contemporary simulation techniques, the integration of large language models (LLMs) into educational research has emerged as a particularly versatile and scalable paradigm. LLMs afford a high degree of linguistic realism and behavioural adaptability, enabling agents to approximate cognitive processes and engage in contextually appropriate pedagogical dialogues. This paper presents a thematic review of empirical and methodological studies utilising LLMs to simulate student behaviour across educational environments. We synthesise current evidence on the capacity of LLM-based agents to emulate learner archetypes, respond to instructional inputs, and interact within multi-agent classroom scenarios. Furthermore, we examine the implications of such systems for curriculum development, instructional evaluation, and teacher training. While LLMs surpass rule-based systems in natural language generation and situational flexibility, ongoing concerns persist regarding algorithmic bias, evaluation reliability, and alignment with educational objectives. The review identifies existing technological and methodological gaps and proposes future research directions for integrating generative AI into adaptive learning systems and instructional design.
>
---
#### [new 129] Adapting Web Agents with Synthetic Supervision
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出SynthAgent，面向网页代理自适应任务，解决合成数据中任务幻觉与轨迹噪声问题，通过任务与轨迹双阶段优化生成高质量合成监督数据，提升代理在新网站上的适应性能。**

- **链接: [http://arxiv.org/pdf/2511.06101v1](http://arxiv.org/pdf/2511.06101v1)**

> **作者:** Zhaoyang Wang; Yiming Liang; Xuchao Zhang; Qianhui Wu; Siwei Han; Anson Bastos; Rujia Wang; Chetan Bansal; Baolin Peng; Jianfeng Gao; Saravan Rajmohan; Huaxiu Yao
>
> **备注:** 19 pages, 6 figures
>
> **摘要:** Web agents struggle to adapt to new websites due to the scarcity of environment specific tasks and demonstrations. Recent works have explored synthetic data generation to address this challenge, however, they suffer from data quality issues where synthesized tasks contain hallucinations that cannot be executed, and collected trajectories are noisy with redundant or misaligned actions. In this paper, we propose SynthAgent, a fully synthetic supervision framework that aims at improving synthetic data quality via dual refinement of both tasks and trajectories. Our approach begins by synthesizing diverse tasks through categorized exploration of web elements, ensuring efficient coverage of the target environment. During trajectory collection, we refine tasks when conflicts with actual observations are detected, mitigating hallucinations while maintaining task consistency. After collection, we conduct trajectory refinement with a global context to mitigate potential noise or misalignments. Finally, we fine-tune open-source web agents on the refined synthetic data to adapt them to the target environment. Experimental results demonstrate that SynthAgent outperforms existing synthetic data methods, validating the importance of high-quality synthetic supervision. The code will be publicly available at https://github.com/aiming-lab/SynthAgent.
>
---
#### [new 130] Self-Evaluating LLMs for Multi-Step Tasks: Stepwise Confidence Estimation for Failure Detection
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文面向多步推理任务，解决LLM缺乏步骤级错误检测的问题，提出步进式自评估方法，通过对比整体与分步评分，证明分步置信估计可提升错误检测准确率，增强模型可信度。**

- **链接: [http://arxiv.org/pdf/2511.07364v1](http://arxiv.org/pdf/2511.07364v1)**

> **作者:** Vaibhav Mavi; Shubh Jaroria; Weiqi Sun
>
> **备注:** Accepted at NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
>
> **摘要:** Reliability and failure detection of large language models (LLMs) is critical for their deployment in high-stakes, multi-step reasoning tasks. Prior work explores confidence estimation for self-evaluating LLM-scorer systems, with confidence scorers estimating the likelihood of errors in LLM responses. However, most methods focus on single-step outputs and overlook the challenges of multi-step reasoning. In this work, we extend self-evaluation techniques to multi-step tasks, testing two intuitive approaches: holistic scoring and step-by-step scoring. Using two multi-step benchmark datasets, we show that stepwise evaluation generally outperforms holistic scoring in detecting potential errors, with up to 15% relative increase in AUC-ROC. Our findings demonstrate that self-evaluating LLM systems provide meaningful confidence estimates in complex reasoning, improving their trustworthiness and providing a practical framework for failure detection.
>
---
#### [new 131] MENTOR: A Metacognition-Driven Self-Evolution Framework for Uncovering and Mitigating Implicit Risks in LLMs on Domain Tasks
- **分类: cs.AI; cs.CL**

- **简介: 论文提出MENTOR框架，通过元认知自评估与动态规则图谱，解决LLM在领域任务中隐性价值对齐风险，实现低成本、自进化式风险识别与缓解，提升模型安全性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.07107v1](http://arxiv.org/pdf/2511.07107v1)**

> **作者:** Liang Shan; Kaicheng Shen; Wen Wu; Zhenyu Ying; Chaochao Lu; Guangze Ye; Liang He
>
> **摘要:** Ensuring the safety and value alignment of large language models (LLMs) is critical for their deployment. Current alignment efforts primarily target explicit risks such as bias, hate speech, and violence. However, they often fail to address deeper, domain-specific implicit risks and lack a flexible, generalizable framework applicable across diverse specialized fields. Hence, we proposed MENTOR: A MEtacognition-driveN self-evoluTion framework for uncOvering and mitigating implicit Risks in LLMs on Domain Tasks. To address the limitations of labor-intensive human evaluation, we introduce a novel metacognitive self-assessment tool. This enables LLMs to reflect on potential value misalignments in their responses using strategies like perspective-taking and consequential thinking. We also release a supporting dataset of 9,000 risk queries spanning education, finance, and management to enhance domain-specific risk identification. Subsequently, based on the outcomes of metacognitive reflection, the framework dynamically generates supplementary rule knowledge graphs that extend predefined static rule trees. This enables models to actively apply validated rules to future similar challenges, establishing a continuous self-evolution cycle that enhances generalization by reducing maintenance costs and inflexibility of static systems. Finally, we employ activation steering during inference to guide LLMs in following the rules, a cost-effective method to robustly enhance enforcement across diverse contexts. Experimental results show MENTOR's effectiveness: In defensive testing across three vertical domains, the framework substantially reduces semantic attack success rates, enabling a new level of implicit risk mitigation for LLMs. Furthermore, metacognitive assessment not only aligns closely with baseline human evaluators but also delivers more thorough and insightful analysis of LLMs value alignment.
>
---
#### [new 132] When AI Agents Collude Online: Financial Fraud Risks by Collaborative LLM Agents on Social Platforms
- **分类: cs.MA; cs.AI; cs.CL; cs.SI**

- **简介: 该论文研究LLM智能体在社交平台协同实施金融诈骗的风险，构建了MultiAgentFraudBench基准，分析协作机制与影响因素，并提出预警、监控与群体韧性等缓解策略，揭示智能体 collusion 的真实威胁。**

- **链接: [http://arxiv.org/pdf/2511.06448v1](http://arxiv.org/pdf/2511.06448v1)**

> **作者:** Qibing Ren; Zhijie Zheng; Jiaxuan Guo; Junchi Yan; Lizhuang Ma; Jing Shao
>
> **备注:** Code is available at https://github.com/zheng977/MutiAgent4Fraud
>
> **摘要:** In this work, we study the risks of collective financial fraud in large-scale multi-agent systems powered by large language model (LLM) agents. We investigate whether agents can collaborate in fraudulent behaviors, how such collaboration amplifies risks, and what factors influence fraud success. To support this research, we present MultiAgentFraudBench, a large-scale benchmark for simulating financial fraud scenarios based on realistic online interactions. The benchmark covers 28 typical online fraud scenarios, spanning the full fraud lifecycle across both public and private domains. We further analyze key factors affecting fraud success, including interaction depth, activity level, and fine-grained collaboration failure modes. Finally, we propose a series of mitigation strategies, including adding content-level warnings to fraudulent posts and dialogues, using LLMs as monitors to block potentially malicious agents, and fostering group resilience through information sharing at the societal level. Notably, we observe that malicious agents can adapt to environmental interventions. Our findings highlight the real-world risks of multi-agent financial fraud and suggest practical measures for mitigating them. Code is available at https://github.com/zheng977/MutiAgent4Fraud.
>
---
#### [new 133] Predicting Oscar-Nominated Screenplays with Sentence Embeddings
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文预测奥斯卡提名剧本，属文本分类任务。构建Movie-O-Label数据集，用E5嵌入编码剧本标题、摘要与脚本，结合逻辑回归分类，实现F1=0.66，验证语言模型在影视奖项预测中的有效性。**

- **链接: [http://arxiv.org/pdf/2511.05500v1](http://arxiv.org/pdf/2511.05500v1)**

> **作者:** Francis Gross
>
> **摘要:** Oscar nominations are an important factor in the movie industry because they can boost both the visibility and the commercial success. This work explores whether it is possible to predict Oscar nominations for screenplays using modern language models. Since no suitable dataset was available, a new one called Movie-O-Label was created by combining the MovieSum collection of movie scripts with curated Oscar records. Each screenplay was represented by its title, Wikipedia summary, and full script. Long scripts were split into overlapping text chunks and encoded with the E5 sentence em bedding model. Then, the screenplay embed dings were classified using a logistic regression model. The best results were achieved when three feature inputs related to screenplays (script, summary, and title) were combined. The best-performing model reached a macro F1 score of 0.66, a precision recall AP of 0.445 with baseline 0.19 and a ROC-AUC of 0.79. The results suggest that even simple models based on modern text embeddings demonstrate good prediction performance and might be a starting point for future research.
>
---
#### [new 134] TabDistill: Distilling Transformers into Neural Nets for Few-Shot Tabular Classification
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出TabDistill，将预训练Transformer的少样本表格分类知识蒸馏至轻量神经网络，在数据稀缺时提升性能并降低参数量，超越传统模型甚至原始Transformer。**

- **链接: [http://arxiv.org/pdf/2511.05704v1](http://arxiv.org/pdf/2511.05704v1)**

> **作者:** Pasan Dissanayake; Sanghamitra Dutta
>
> **摘要:** Transformer-based models have shown promising performance on tabular data compared to their classical counterparts such as neural networks and Gradient Boosted Decision Trees (GBDTs) in scenarios with limited training data. They utilize their pre-trained knowledge to adapt to new domains, achieving commendable performance with only a few training examples, also called the few-shot regime. However, the performance gain in the few-shot regime comes at the expense of significantly increased complexity and number of parameters. To circumvent this trade-off, we introduce TabDistill, a new strategy to distill the pre-trained knowledge in complex transformer-based models into simpler neural networks for effectively classifying tabular data. Our framework yields the best of both worlds: being parameter-efficient while performing well with limited training data. The distilled neural networks surpass classical baselines such as regular neural networks, XGBoost and logistic regression under equal training data, and in some cases, even the original transformer-based models that they were distilled from.
>
---
#### [new 135] Graph Representation-based Model Poisoning on the Heterogeneous Internet of Agents
- **分类: cs.NI; cs.CL**

- **简介: 该论文针对异构智能体网络中的联邦学习模型投毒问题，提出图表示攻击（GRMP），通过构建参数相关图与变分图自编码器生成隐蔽恶意模型，绕过现有检测机制，严重威胁IoA系统安全。**

- **链接: [http://arxiv.org/pdf/2511.07176v1](http://arxiv.org/pdf/2511.07176v1)**

> **作者:** Hanlin Cai; Houtianfu Wang; Haofan Dong; Kai Li; Ozgur B. Akan
>
> **备注:** 6 pages, 6 figures
>
> **摘要:** Internet of Agents (IoA) envisions a unified, agent-centric paradigm where heterogeneous large language model (LLM) agents can interconnect and collaborate at scale. Within this paradigm, federated learning (FL) serves as a key enabler that allows distributed LLM agents to co-train global models without centralizing data. However, the FL-enabled IoA system remains vulnerable to model poisoning attacks, and the prevailing distance and similarity-based defenses become fragile at billion-parameter scale and under heterogeneous data distributions. This paper proposes a graph representation-based model poisoning (GRMP) attack, which passively exploits observed benign local models to construct a parameter correlation graph and extends an adversarial variational graph autoencoder to capture and reshape higher-order dependencies. The GRMP attack synthesizes malicious local models that preserve benign-like statistics while embedding adversarial objectives, remaining elusive to detection at the server. Experiments demonstrate a gradual drop in system accuracy under the proposed attack and the ineffectiveness of the prevailing defense mechanism in detecting the attack, underscoring a severe threat to the ambitious IoA paradigm.
>
---
#### [new 136] ELEGANCE: Efficient LLM Guidance for Audio-Visual Target Speech Extraction
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 论文提出ELEGANCE框架，将大语言模型的 linguistic 知识引入音视频目标语音提取任务，通过三种引导策略提升模型在视觉信息受限、语种未知等挑战场景下的提取性能。**

- **链接: [http://arxiv.org/pdf/2511.06288v1](http://arxiv.org/pdf/2511.06288v1)**

> **作者:** Wenxuan Wu; Shuai Wang; Xixin Wu; Helen Meng; Haizhou Li
>
> **摘要:** Audio-visual target speaker extraction (AV-TSE) models primarily rely on visual cues from the target speaker. However, humans also leverage linguistic knowledge, such as syntactic constraints, next word prediction, and prior knowledge of conversation, to extract target speech. Inspired by this observation, we propose ELEGANCE, a novel framework that incorporates linguistic knowledge from large language models (LLMs) into AV-TSE models through three distinct guidance strategies: output linguistic constraints, intermediate linguistic prediction, and input linguistic prior. Comprehensive experiments with RoBERTa, Qwen3-0.6B, and Qwen3-4B on two AV-TSE backbones demon- strate the effectiveness of our approach. Significant improvements are observed in challenging scenarios, including visual cue impaired, unseen languages, target speaker switches, increased interfering speakers, and out-of-domain test set. Demo page: https://alexwxwu.github.io/ELEGANCE/.
>
---
#### [new 137] Approximating the Mathematical Structure of Psychodynamics
- **分类: q-bio.NC; cs.CL; cs.CY; cs.HC; 18M35 (primary) 47N99, 47A50, 60H25, 68T37, 68T42, 81V99, 91C99,
  91E10, 94A99 (secondary); E.4; G.3; I.2.0; I.2.4; J.3; J.4**

- **简介: 该论文旨在为心理动力学建立数学框架，使用过程理论的图示方法形式化人类心理动态，解决心理学缺乏精确量化工具的问题，推动其在心理治疗、AI安全等领域的跨学科应用。**

- **链接: [http://arxiv.org/pdf/2511.05580v1](http://arxiv.org/pdf/2511.05580v1)**

> **作者:** Bryce-Allen Bagley; Navin Khoshnan
>
> **摘要:** The complexity of human cognition has meant that psychology makes more use of theory and conceptual models than perhaps any other biomedical field. To enable precise quantitative study of the full breadth of phenomena in psychological and psychiatric medicine as well as cognitive aspects of AI safety, there is a need for a mathematical formulation which is both mathematically precise and equally accessible to experts from numerous fields. In this paper we formalize human psychodynamics via the diagrammatic framework of process theory, describe its key properties, and explain the links between a diagrammatic representation and central concepts in analysis of cognitive processes in contexts such as psychotherapy, neurotechnology, AI alignment, AI agent representation of individuals in autonomous negotiations, developing human-like AI systems, and other aspects of AI safety.
>
---
#### [new 138] LPFQA: A Long-Tail Professional Forum-based Benchmark for LLM Evaluation
- **分类: cs.AI; cs.CL**

- **简介: 论文提出LPFQA基准，面向长尾专业领域知识，解决现有LLM评估脱离真实场景的问题。基于20个领域的实际论坛数据，构建502个复杂任务，引入多维评估体系，显著提升评估的真实性与区分度。**

- **链接: [http://arxiv.org/pdf/2511.06346v1](http://arxiv.org/pdf/2511.06346v1)**

> **作者:** Liya Zhu; Peizhuang Cong; Aowei Ji; Wenya Wu; Jiani Hou; Chunjie Wu; Xiang Gao; Jingkai Liu; Zhou Huan; Xuelei Sun; Yang Yang; Jianpeng Jiao; Liang Hu; Xinjie Chen; Jiashuo Liu; Jingzhe Ding; Tong Yang; Zaiyuan Wang; Ge Zhang; Wenhao Huang
>
> **摘要:** Large Language Models (LLMs) have made rapid progress in reasoning, question answering, and professional applications; however, their true capabilities remain difficult to evaluate using existing benchmarks. Current datasets often focus on simplified tasks or artificial scenarios, overlooking long-tail knowledge and the complexities of real-world applications. To bridge this gap, we propose LPFQA, a long-tail knowledge-based benchmark derived from authentic professional forums across 20 academic and industrial fields, covering 502 tasks grounded in practical expertise. LPFQA introduces four key innovations: fine-grained evaluation dimensions that target knowledge depth, reasoning, terminology comprehension, and contextual analysis; a hierarchical difficulty structure that ensures semantic clarity and unique answers; authentic professional scenario modeling with realistic user personas; and interdisciplinary knowledge integration across diverse domains. We evaluated 12 mainstream LLMs on LPFQA and observed significant performance disparities, especially in specialized reasoning tasks. LPFQA provides a robust, authentic, and discriminative benchmark for advancing LLM evaluation and guiding future model development.
>
---
#### [new 139] Fine-Tuning Vision-Language Models for Multimodal Polymer Property Prediction
- **分类: cs.LG; cond-mat.mtrl-sci; cs.AI; cs.CL**

- **简介: 该论文面向聚合物性质预测任务，解决多模态数据利用不足问题，构建专用数据集并基于LoRA微调视觉-语言模型，实现跨属性的高效多模态预测，显著优于单模态基线。**

- **链接: [http://arxiv.org/pdf/2511.05577v1](http://arxiv.org/pdf/2511.05577v1)**

> **作者:** An Vuong; Minh-Hao Van; Prateek Verma; Chen Zhao; Xintao Wu
>
> **摘要:** Vision-Language Models (VLMs) have shown strong performance in tasks like visual question answering and multimodal text generation, but their effectiveness in scientific domains such as materials science remains limited. While some machine learning methods have addressed specific challenges in this field, there is still a lack of foundation models designed for broad tasks like polymer property prediction using multimodal data. In this work, we present a multimodal polymer dataset to fine-tune VLMs through instruction-tuning pairs and assess the impact of multimodality on prediction performance. Our fine-tuned models, using LoRA, outperform unimodal and baseline approaches, demonstrating the benefits of multimodal learning. Additionally, this approach reduces the need to train separate models for different properties, lowering deployment and maintenance costs.
>
---
#### [new 140] Optimizing Chain-of-Thought Confidence via Topological and Dirichlet Risk Analysis
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对大语言模型链式推理中置信度校准差、过度自信问题，提出EDTR方法，融合拓扑分析与狄利克雷分布，量化多推理路径的几何不确定性，显著提升校准精度，在数学与常识推理任务上表现优异。**

- **链接: [http://arxiv.org/pdf/2511.06437v1](http://arxiv.org/pdf/2511.06437v1)**

> **作者:** Abhishek More; Anthony Zhang; Nicole Bonilla; Ashvik Vivekan; Kevin Zhu; Parham Sharafoleslami; Maheep Chaudhary
>
> **摘要:** Chain-of-thought (CoT) prompting enables Large Language Models to solve complex problems, but deploying these models safely requires reliable confidence estimates, a capability where existing methods suffer from poor calibration and severe overconfidence on incorrect predictions. We propose Enhanced Dirichlet and Topology Risk (EDTR), a novel decoding strategy that combines topological analysis with Dirichlet-based uncertainty quantification to measure LLM confidence across multiple reasoning paths. EDTR treats each CoT as a vector in high-dimensional space and extracts eight topological risk features capturing the geometric structure of reasoning distributions: tighter, more coherent clusters indicate higher confidence while dispersed, inconsistent paths signal uncertainty. We evaluate EDTR against three state-of-the-art calibration methods across four diverse reasoning benchmarks spanning olympiad-level mathematics (AIME), grade school math (GSM8K), commonsense reasoning, and stock price prediction \cite{zhang2025aime, cobbe2021training, talmor-etal-2019-commonsenseqa, yahoo_finance}. EDTR achieves 41\% better calibration than competing methods with an average ECE of 0.287 and the best overall composite score of 0.672, while notably achieving perfect accuracy on AIME and exceptional calibration on GSM8K with an ECE of 0.107, domains where baselines exhibit severe overconfidence. Our work provides a geometric framework for understanding and quantifying uncertainty in multi-step LLM reasoning, enabling more reliable deployment where calibrated confidence estimates are essential.
>
---
#### [new 141] Language Generation with Infinite Contamination
- **分类: stat.ML; cs.AI; cs.CL; cs.DS; cs.LG**

- **简介: 该论文研究带噪声和缺失的语言生成问题，刻画了生成与稠密生成在污染数据下的鲁棒性边界，证明零渐近污染率是可生成的充要条件，并揭示课程学习对处理无限污染的关键作用。**

- **链接: [http://arxiv.org/pdf/2511.07417v1](http://arxiv.org/pdf/2511.07417v1)**

> **作者:** Anay Mehrotra; Grigoris Velegkas; Xifan Yu; Felix Zhou
>
> **摘要:** We study language generation in the limit, where an algorithm observes an adversarial enumeration of strings from an unknown target language $K$ and must eventually generate new, unseen strings from $K$. Kleinberg and Mullainathan [KM24] proved that generation is achievable in surprisingly general settings. But their generator suffers from ``mode collapse,'' producing from an ever-smaller subset of the target. To address this, Kleinberg and Wei [KW25] require the generator's output to be ``dense'' in the target language. They showed that generation with density, surprisingly, remains achievable at the same generality. Both results assume perfect data: no noisy insertions and no omissions. This raises a central question: how much contamination can generation tolerate? Recent works made partial progress on this question by studying (non-dense) generation with either finite amounts of noise (but no omissions) or omissions (but no noise). We characterize robustness under contaminated enumerations: 1. Generation under Contamination: Language generation in the limit is achievable for all countable collections iff the fraction of contaminated examples converges to zero. When this fails, we characterize which collections are generable. 2. Dense Generation under Contamination: Dense generation is strictly less robust to contamination than generation. As a byproduct, we resolve an open question of Raman and Raman [ICML25] by showing that generation is possible with only membership oracle access under finitely many contaminated examples. Finally, we introduce a beyond-worst-case model inspired by curriculum learning and prove that dense generation is achievable even with infinite contamination provided the fraction of contaminated examples converges to zero. This suggests curriculum learning may be crucial for learning from noisy web data.
>
---
#### [new 142] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 论文提出SpatialThinker，面向多模态大模型的3D空间理解任务，解决其空间推理能力弱的问题。通过合成STVQA-7K数据集与强化学习密集空间奖励机制，实现无需大规模数据的高效空间 grounding 与多步推理，显著超越基线与GPT-4o。**

- **链接: [http://arxiv.org/pdf/2511.07403v1](http://arxiv.org/pdf/2511.07403v1)**

> **作者:** Hunar Batra; Haoqin Tu; Hardy Chen; Yuanze Lin; Cihang Xie; Ronald Clark
>
> **备注:** Preprint. Accepted at NeurIPS 2025 Workshops on SPACE in Vision, Language, and Embodied AI (SpaVLE), Embodied World Models for Decision Making (EWM), Aligning Reinforcement Learning Experimentalists and Theorists (ARLET), and Scaling Environments for Agents (SEA)
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable progress in vision-language tasks, but they continue to struggle with spatial understanding. Existing spatial MLLMs often rely on explicit 3D inputs or architecture-specific modifications, and remain constrained by large-scale datasets or sparse supervision. To address these limitations, we introduce SpatialThinker, a 3D-aware MLLM trained with RL to integrate structured spatial grounding with multi-step reasoning. The model simulates human-like spatial perception by constructing a scene graph of task-relevant objects and spatial relations, and reasoning towards an answer via dense spatial rewards. SpatialThinker consists of two key contributions: (1) a data synthesis pipeline that generates STVQA-7K, a high-quality spatial VQA dataset, and (2) online RL with a multi-objective dense spatial reward enforcing spatial grounding. SpatialThinker-7B outperforms supervised fine-tuning and the sparse RL baseline on spatial understanding and real-world VQA benchmarks, nearly doubling the base-model gain compared to sparse RL, and surpassing GPT-4o. These results showcase the effectiveness of combining spatial supervision with reward-aligned reasoning in enabling robust 3D spatial understanding with limited data and advancing MLLMs towards human-level visual reasoning.
>
---
#### [new 143] Long Grounded Thoughts: Distilling Compositional Visual Reasoning Chains at Scale
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出一种大规模视觉推理数据生成框架，构建超百万高质量视觉推理样本，支持多模态推理训练。通过蒸馏复杂思维链，显著提升Qwen2.5-VL-7B性能，并实现跨模态迁移，解决视觉推理数据稀缺与泛化不足问题。**

- **链接: [http://arxiv.org/pdf/2511.05705v1](http://arxiv.org/pdf/2511.05705v1)**

> **作者:** David Acuna; Chao-Han Huck Yang; Yuntian Deng; Jaehun Jung; Ximing Lu; Prithviraj Ammanabrolu; Hyunwoo Kim; Yuan-Hong Liao; Yejin Choi
>
> **备注:** Project Page: https://nvlabs.github.io/LongGroundedThoughts/
>
> **摘要:** Recent progress in multimodal reasoning has been driven largely by undisclosed datasets and proprietary data synthesis recipes, leaving open questions about how to systematically build large-scale, vision-centric reasoning datasets, particularly for tasks that go beyond visual math. In this work, we introduce a new reasoning data generation framework spanning diverse skills and levels of complexity with over 1M high-quality synthetic vision-centric questions. The dataset also includes preference data and instruction prompts supporting both offline and online RL. Our synthesis framework proceeds in two stages: (1) scale; and (2) complexity. Reasoning traces are then synthesized through a two-stage process that leverages VLMs and reasoning LLMs, producing CoT traces for VLMs that capture the richness and diverse cognitive behaviors found in frontier reasoning models. Remarkably, we show that finetuning Qwen2.5-VL-7B on our data outperforms all open-data baselines across all evaluated vision-centric benchmarks, and even surpasses strong closed-data models such as MiMo-VL-7B-RL on V* Bench, CV-Bench and MMStar-V. Perhaps most surprising, despite being entirely vision-centric, our data transfers positively to text-only reasoning (MMLU-Pro) and audio reasoning (MMAU), demonstrating its effectiveness. Similarly, despite not containing videos or embodied visual data, we observe notable gains when evaluating on a single-evidence embodied QA benchmark (NiEH). Finally, we use our data to analyze the entire VLM post-training pipeline. Our empirical analysis highlights that (i) SFT on high-quality data with non-linear reasoning traces is essential for effective online RL, (ii) staged offline RL matches online RL's performance while reducing compute demands, and (iii) careful SFT on high quality data can substantially improve out-of-domain, cross-modality transfer.
>
---
#### [new 144] Place Matters: Comparing LLM Hallucination Rates for Place-Based Legal Queries
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文研究LLM在不同地区法律查询中的幻觉率差异，提出基于功能主义的比较方法，构建三地真实法律场景数据集，评估模型输出的法律信息准确性，发现幻觉率与地理位置显著相关。**

- **链接: [http://arxiv.org/pdf/2511.06700v1](http://arxiv.org/pdf/2511.06700v1)**

> **作者:** Damian Curran; Vanessa Sporne; Lea Frermann; Jeannie Paterson
>
> **摘要:** How do we make a meaningful comparison of a large language model's knowledge of the law in one place compared to another? Quantifying these differences is critical to understanding if the quality of the legal information obtained by users of LLM-based chatbots varies depending on their location. However, obtaining meaningful comparative metrics is challenging because legal institutions in different places are not themselves easily comparable. In this work we propose a methodology to obtain place-to-place metrics based on the comparative law concept of functionalism. We construct a dataset of factual scenarios drawn from Reddit posts by users seeking legal advice for family, housing, employment, crime and traffic issues. We use these to elicit a summary of a law from the LLM relevant to each scenario in Los Angeles, London and Sydney. These summaries, typically of a legislative provision, are manually evaluated for hallucinations. We show that the rate of hallucination of legal information by leading closed-source LLMs is significantly associated with place. This suggests that the quality of legal solutions provided by these models is not evenly distributed across geography. Additionally, we show a strong negative correlation between hallucination rate and the frequency of the majority response when the LLM is sampled multiple times, suggesting a measure of uncertainty of model predictions of legal facts.
>
---
## 更新

#### [replaced 001] Mixed Signals: Understanding Model Disagreement in Multimodal Empathy Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13979v2](http://arxiv.org/pdf/2505.13979v2)**

> **作者:** Maya Srikanth; Run Chen; Julia Hirschberg
>
> **摘要:** Multimodal models play a key role in empathy detection, but their performance can suffer when modalities provide conflicting cues. To understand these failures, we examine cases where unimodal and multimodal predictions diverge. Using fine-tuned models for text, audio, and video, along with a gated fusion model, we find that such disagreements often reflect underlying ambiguity, as evidenced by annotator uncertainty. Our analysis shows that dominant signals in one modality can mislead fusion when unsupported by others. We also observe that humans, like models, do not consistently benefit from multimodal input. These insights position disagreement as a useful diagnostic signal for identifying challenging examples and improving empathy system robustness.
>
---
#### [replaced 002] Language Model Distillation: A Temporal Difference Imitation Learning Perspective
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20335v2](http://arxiv.org/pdf/2505.20335v2)**

> **作者:** Zishun Yu; Shangzhe Li; Xinhua Zhang
>
> **备注:** AAAI 2026
>
> **摘要:** Large language models have led to significant progress across many NLP tasks, although their massive sizes often incur substantial computational costs. Distillation has become a common practice to compress these large and highly capable models into smaller, more efficient ones. Many existing language model distillation methods can be viewed as behavior cloning from the perspective of imitation learning or inverse reinforcement learning. This viewpoint has inspired subsequent studies that leverage (inverse) reinforcement learning techniques, including variations of behavior cloning and temporal difference learning methods. Rather than proposing yet another specific temporal difference method, we introduce a general framework for temporal difference-based distillation by exploiting the distributional sparsity of the teacher model. Specifically, it is often observed that language models assign most probability mass to a small subset of tokens. Motivated by this observation, we design a temporal difference learning framework that operates on a reduced action space (a subset of vocabulary), and demonstrate how practical algorithms can be derived and the resulting performance improvements.
>
---
#### [replaced 003] Rethinking Tokenization for Rich Morphology: The Dominance of Unigram over BPE and Morphological Alignment
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.08424v3](http://arxiv.org/pdf/2508.08424v3)**

> **作者:** Saketh Reddy Vemula; Sandipan Dandapat; Dipti Misra Sharma; Parameswari Krishnamurthy
>
> **摘要:** The relationship between tokenizer algorithm (e.g., Byte-Pair Encoding (BPE), Unigram), morphological alignment, tokenization quality (e.g., compression efficiency), and downstream performance remains largely unclear, particularly for languages with complex morphology. In this paper, we conduct a comprehensive evaluation of tokenizers using small-sized BERT models -- from pre-training through fine-tuning -- for Telugu (agglutinative), along with preliminary evaluation in Hindi (primarily fusional with some agglutination) and English (fusional). To evaluate morphological alignment of tokenizers in Telugu, we create a dataset containing gold morpheme segmentations of 600 derivational and 7000 inflectional word forms. Our experiments reveal two key findings for Telugu. First, the choice of tokenizer algorithm is the most significant factor influencing performance, with Unigram-based tokenizers consistently outperforming BPE across most settings. Second, while better morphological alignment shows a moderate, positive correlation with performance on text classification and structure prediction tasks, its impact is secondary to the tokenizer algorithm. Notably, hybrid approaches that use morphological information for pre-segmentation significantly boost the performance of BPE, though not Unigram. Our results further showcase the need for comprehensive intrinsic evaluation metrics for tokenizers that could explain downstream performance trends consistently.
>
---
#### [replaced 004] Sub-exponential Growth of New Words and Names Online: A Piecewise Power-Law Model
- **分类: physics.soc-ph; cs.CL; cs.CY; stat.AP**

- **链接: [http://arxiv.org/pdf/2511.04106v2](http://arxiv.org/pdf/2511.04106v2)**

> **作者:** Hayafumi Watanabe
>
> **摘要:** The diffusion of ideas and language in society has conventionally been described by S-shaped models, such as the logistic curve. However, the role of sub-exponential growth -a slower than exponential pattern known in epidemiology- has been largely overlooked in broader social phenomena. Here, we present a piecewise power-law model to characterize complex growth curves with a few parameters. We systematically analyzed a large-scale dataset of approximately one billion Japanese blog articles linked to Wikipedia vocabulary, and observed consistent patterns in web search trend data (English, Spanish, and Japanese). Our analysis of the 2,965 selected items reveals that about 55% (1,625 items) were found to have no abrupt jumps and were well captured by one or two segments. For single-segment curves, we found that (i) the mode of the shape parameter alpha was near 0.5, indicating prevalent sub-exponential growth; (ii) the ultimate diffusion scale is primarily determined by the growth rate R, with minor contributions from alpha or the duration T; and (iii) alpha showed a tendency to vary with the nature of the topic, being smaller for niche/local topics and larger for widely shared ones. Furthermore, a micro-behavioral model distinguishing outward contact with strangers from inward interaction within their community suggests that alpha can be interpreted as an index of the preference for outward-oriented communication. These findings suggest that sub-exponential growth is a common pattern of social diffusion, and our model provides a practical framework for consistently describing, comparing, and interpreting complex and diverse growth curves.
>
---
#### [replaced 005] DynaSpec: Context-aware Dynamic Speculative Sampling for Large-Vocabulary Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.13847v2](http://arxiv.org/pdf/2510.13847v2)**

> **作者:** Jinbin Zhang; Nasib Ullah; Erik Schultheis; Rohit Babbar
>
> **摘要:** Speculative decoding has become a standard way to accelerate LLM inference: a small drafter proposes multiple tokens and a large target model verifies them once per speculation length. Recently, scaling of the LLM vocabulary has pushed the number of tokens to grow substantially. While verification over the full vocabulary leaves the target model largely unaffected, the O(|V|d) parameters in the drafter's output head become a latency bottleneck, slowing the entire pipeline. Contemporary methods (e.g., FR-Spec, VocabTrim) restrict the drafter's vocabulary to a fixed top frequent subset of the target model's vocabulary. Although this reduces draft-time compute, it is brittle, since: (i) frequency lists are corpus-dependent and require retuning to generalize, and (ii) static shortlists suppress rare or domain-specific tokens, lowering the expected number of tokens per verification step. We propose DynaSpec, a context-dependent dynamic shortlisting mechanism that is robust, speeds up drafting, and generalizes across diverse tasks. Concretely, we introduce lightweight, coarse-grained meta-classifiers that route contexts to a small number of token clusters; the union of the top-k selected clusters forms the drafter's shortlist, while verification retains the full vocabulary and exactness. The meta-classifier finishes its computation earlier than the drafter's hidden state generation by exploiting parallel execution of draft encoding and meta shortlisting on separate streams. Across standard speculative decoding benchmarks, DynaSpec delivers consistent improvements in mean accepted length, for Llama-3-8B, reaching upto 98.2% of full-vocabulary performance, while fixed-shortlist baselines attain only 84.4%. By leveraging context-dependent selection, DynaSpec achieves up to a 2.18 times increase in generated tokens compared to 1.91 times for fixed-vocabulary approaches.
>
---
#### [replaced 006] Evaluating the Ability of Large Language Models to Reason about Cardinal Directions, Revisited
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.12059v2](http://arxiv.org/pdf/2507.12059v2)**

> **作者:** Anthony G Cohn; Robert E Blackwell
>
> **备注:** 8 pages, 5 figures. Accepted at QR 2025 : 38th International Workshop on Qualitative Reasoning at IJCAI. arXiv admin note: substantial text overlap with arXiv:2406.16528
>
> **摘要:** We investigate the abilities of 28 Large language Models (LLMs) to reason about cardinal directions (CDs) using a benchmark generated from a set of templates, extensively testing an LLM's ability to determine the correct CD given a particular scenario. The templates allow for a number of degrees of variation such as means of locomotion of the agent involved, and whether set in the first, second or third person. Even the newer Large Reasoning Models are unable to reliably determine the correct CD for all questions. This paper summarises and extends earlier work presented at COSIT-24.
>
---
#### [replaced 007] GUARD: Guideline Upholding Test through Adaptive Role-play and Jailbreak Diagnostics for LLMs
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.20325v2](http://arxiv.org/pdf/2508.20325v2)**

> **作者:** Haibo Jin; Ruoxi Chen; Peiyan Zhang; Andy Zhou; Haohan Wang
>
> **备注:** 54 pages
>
> **摘要:** As Large Language Models become increasingly integral to various domains, their potential to generate harmful responses has prompted significant societal and regulatory concerns. In response, governments have issued ethics guidelines to promote the development of trustworthy AI. However, these guidelines are typically high-level demands for developers and testers, leaving a gap in translating them into actionable testing questions to verify LLM compliance. To address this challenge, we introduce GUARD (\textbf{G}uideline \textbf{U}pholding Test through \textbf{A}daptive \textbf{R}ole-play and Jailbreak \textbf{D}iagnostics), a testing method designed to operationalize guidelines into specific guideline-violating questions that assess LLM adherence. To implement this, GUARD uses automated generation of guideline-violating questions based on government-issued guidelines, thereby testing whether responses comply with these guidelines. When responses directly violate guidelines, GUARD reports inconsistencies. Furthermore, for responses that do not directly violate guidelines, GUARD integrates the concept of ``jailbreaks'' to diagnostics, named GUARD-JD, which creates scenarios that provoke unethical or guideline-violating responses, effectively identifying potential scenarios that could bypass built-in safety mechanisms. Our method finally culminates in a compliance report, delineating the extent of adherence and highlighting any violations. We have empirically validated the effectiveness of GUARD on seven LLMs, including Vicuna-13B, LongChat-7B, Llama2-7B, Llama-3-8B, GPT-3.5, GPT-4, GPT-4o, and Claude-3.7, by testing compliance under three government-issued guidelines and conducting jailbreak diagnostics. Additionally, GUARD-JD can transfer jailbreak diagnostics to vision-language models, demonstrating its usage in promoting reliable LLM-based applications.
>
---
#### [replaced 008] Employing Sentence Space Embedding for Classification of Data Stream from Fake News Domain
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.10807v2](http://arxiv.org/pdf/2407.10807v2)**

> **作者:** Paweł Zyblewski; Jakub Klikowski; Weronika Borek-Marciniec; Paweł Ksieniewicz
>
> **备注:** 16 pages, 7 figures
>
> **摘要:** Tabular data is considered the last unconquered castle of deep learning, yet the task of data stream classification is stated to be an equally important and demanding research area. Due to the temporal constraints, it is assumed that deep learning methods are not the optimal solution for application in this field. However, excluding the entire -- and prevalent -- group of methods seems rather rash given the progress that has been made in recent years in its development. For this reason, the following paper is the first to present an approach to natural language data stream classification using the sentence space method, which allows for encoding text into the form of a discrete digital signal. This allows the use of convolutional deep networks dedicated to image classification to solve the task of recognizing fake news based on text data. Based on the real-life Fakeddit dataset, the proposed approach was compared with state-of-the-art algorithms for data stream classification based on generalization ability and time complexity.
>
---
#### [replaced 009] Rethinking Text-based Protein Understanding: Retrieval or LLM?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20354v4](http://arxiv.org/pdf/2505.20354v4)**

> **作者:** Juntong Wu; Zijing Liu; He Cao; Hao Li; Bin Feng; Zishan Shu; Ke Yu; Li Yuan; Yu Li
>
> **备注:** Accepted by Empirical Methods in Natural Language Processing 2025 (EMNLP 2025) Main Conference
>
> **摘要:** In recent years, protein-text models have gained significant attention for their potential in protein generation and understanding. Current approaches focus on integrating protein-related knowledge into large language models through continued pretraining and multi-modal alignment, enabling simultaneous comprehension of textual descriptions and protein sequences. Through a thorough analysis of existing model architectures and text-based protein understanding benchmarks, we identify significant data leakage issues present in current benchmarks. Moreover, conventional metrics derived from natural language processing fail to accurately assess the model's performance in this domain. To address these limitations, we reorganize existing datasets and introduce a novel evaluation framework based on biological entities. Motivated by our observation, we propose a retrieval-enhanced method, which significantly outperforms fine-tuned LLMs for protein-to-text generation and shows accuracy and efficiency in training-free scenarios. Our code and data can be seen at https://github.com/IDEA-XL/RAPM.
>
---
#### [replaced 010] Revealing emergent human-like conceptual representations from language prediction
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.12547v4](http://arxiv.org/pdf/2501.12547v4)**

> **作者:** Ningyu Xu; Qi Zhang; Chao Du; Qiang Luo; Xipeng Qiu; Xuanjing Huang; Menghan Zhang
>
> **备注:** 66 pages. Accepted manuscript. Final version published in Proceedings of the National Academy of Sciences (PNAS): https://www.pnas.org/doi/10.1073/pnas.2512514122
>
> **摘要:** People acquire concepts through rich physical and social experiences and use them to understand and navigate the world. In contrast, large language models (LLMs), trained solely through next-token prediction on text, exhibit strikingly human-like behaviors. Are these models developing concepts akin to those of humans? If so, how are such concepts represented, organized, and related to behavior? Here, we address these questions by investigating the representations formed by LLMs during an in-context concept inference task. We found that LLMs can flexibly derive concepts from linguistic descriptions in relation to contextual cues about other concepts. The derived representations converge toward a shared, context-independent structure, and alignment with this structure reliably predicts model performance across various understanding and reasoning tasks. Moreover, the convergent representations effectively capture human behavioral judgments and closely align with neural activity patterns in the human brain, providing evidence for biological plausibility. Together, these findings establish that structured, human-like conceptual representations can emerge purely from language prediction without real-world grounding, highlighting the role of conceptual structure in understanding intelligent behavior. More broadly, our work suggests that LLMs offer a tangible window into the nature of human concepts and lays the groundwork for advancing alignment between artificial and human intelligence.
>
---
#### [replaced 011] Jr. AI Scientist and Its Risk Report: Autonomous Scientific Exploration from a Baseline Paper
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2511.04583v2](http://arxiv.org/pdf/2511.04583v2)**

> **作者:** Atsuyuki Miyai; Mashiro Toyooka; Takashi Otonari; Zaiying Zhao; Kiyoharu Aizawa
>
> **备注:** Issues, comments, and questions are all welcome in https://github.com/Agent4Science-UTokyo/Jr.AI-Scientist
>
> **摘要:** Understanding the current capabilities and risks of AI Scientist systems is essential for ensuring trustworthy and sustainable AI-driven scientific progress while preserving the integrity of the academic ecosystem. To this end, we develop Jr. AI Scientist, a state-of-the-art autonomous AI scientist system that mimics the core research workflow of a novice student researcher: Given the baseline paper from the human mentor, it analyzes its limitations, formulates novel hypotheses for improvement, and iteratively conducts experiments until improvements are realized, and writes a paper with the results. Unlike previous approaches that assume full automation or operate on small-scale code, Jr. AI Scientist follows a well-defined research workflow and leverages modern coding agents to handle complex, multi-file implementations, leading to scientifically valuable contributions. Through our experiments, the Jr. AI Scientist successfully generated new research papers that build upon real NeurIPS, IJCV, and ICLR works by proposing and implementing novel methods. For evaluation, we conducted automated assessments using AI Reviewers, author-led evaluations, and submissions to Agents4Science, a venue dedicated to AI-driven scientific contributions. The findings demonstrate that Jr. AI Scientist generates papers receiving higher review scores than existing fully automated systems. Nevertheless, we identify important limitations from both the author evaluation and the Agents4Science reviews, indicating the potential risks of directly applying current AI Scientist systems and key challenges for future research. Finally, we comprehensively report various risks identified during development. We believe this study clarifies the current role and limitations of AI Scientist systems, offering insights into the areas that still require human expertise and the risks that may emerge as these systems evolve.
>
---
#### [replaced 012] LLMCARE: early detection of cognitive impairment via transformer models enhanced by LLM-generated synthetic data
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.10027v3](http://arxiv.org/pdf/2508.10027v3)**

> **作者:** Ali Zolnour; Hossein Azadmaleki; Yasaman Haghbin; Fatemeh Taherinezhad; Mohamad Javad Momeni Nezhad; Sina Rashidi; Masoud Khani; AmirSajjad Taleban; Samin Mahdizadeh Sani; Maryam Dadkhah; James M. Noble; Suzanne Bakken; Yadollah Yaghoobzadeh; Abdol-Hossein Vahabie; Masoud Rouhizadeh; Maryam Zolnoori
>
> **摘要:** Alzheimer's disease and related dementias(ADRD) affect nearly five million older adults in the United States, yet more than half remain undiagnosed. Speech-based natural language processing(NLP) offers a scalable approach for detecting early cognitive decline through subtle linguistic markers that may precede clinical diagnosis. This study develops and evaluates a speech-based screening pipeline integrating transformer embeddings with handcrafted linguistic features, synthetic augmentation using large language models(LLMs), and benchmarking of unimodal and multimodal classifiers. External validation assessed generalizability to a MCI-only cohort. Transcripts were drawn from the ADReSSo 2021 benchmark dataset(n=237, Pitt Corpus) and the DementiaBank Delaware corpus(n=205, MCI vs. controls). Ten transformer models were tested under three fine-tuning strategies. A late-fusion model combined embeddings from the top transformer with 110 linguistic features. Five LLMs(LLaMA8B/70B, MedAlpaca7B, Ministral8B,GPT-4o) generated label-conditioned synthetic speech for augmentation, and three multimodal LLMs(GPT-4o,Qwen-Omni,Phi-4) were evaluated in zero-shot and fine-tuned modes. On ADReSSo, the fusion model achieved F1=83.3(AUC=89.5), outperforming transformer-only and linguistic baselines. MedAlpaca7B augmentation(2x) improved F1=85.7, though larger scales reduced gains. Fine-tuning boosted unimodal LLMs(MedAlpaca7B F1=47.7=>78.7), while multimodal models performed lower (Phi-4=71.6;GPT-4o=67.6). On Delaware, the fusion plus 1x MedAlpaca7B model achieved F1=72.8(AUC=69.6). Integrating transformer and linguistic features enhances ADRD detection. LLM-based augmentation improves data efficiency but yields diminishing returns, while current multimodal models remain limited. Validation on an independent MCI cohort supports the pipeline's potential for scalable, clinically relevant early screening.
>
---
#### [replaced 013] Likelihood-based Mitigation of Evaluation Bias in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2402.15987v4](http://arxiv.org/pdf/2402.15987v4)**

> **作者:** Masanari Oi; Masahiro Kaneko; Ryuto Koike; Mengsay Loem; Naoaki Okazaki
>
> **备注:** 5 main pages
>
> **摘要:** Large Language Models (LLMs) are widely used to evaluate natural language generation tasks as automated metrics. However, the likelihood, a measure of LLM's plausibility for a sentence, can vary due to superficial differences in sentences, such as word order and sentence structure. It is therefore possible that there might be a likelihood bias if LLMs are used for evaluation: they might overrate sentences with higher likelihoods while underrating those with lower likelihoods. In this paper, we investigate the presence and impact of likelihood bias in LLM-based evaluators. We also propose a method to mitigate the likelihood bias. Our method utilizes highly biased instances as few-shot examples for in-context learning. Our experiments in evaluating the data-to-text and grammatical error correction tasks reveal that several LLMs we test display a likelihood bias. Furthermore, our proposed method successfully mitigates this bias, also improving evaluation performance (in terms of correlation of models with human scores) significantly.
>
---
#### [replaced 014] Breadcrumbs Reasoning: Memory-Efficient Reasoning with Compression Beacons
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.13797v2](http://arxiv.org/pdf/2510.13797v2)**

> **作者:** Giovanni Monea; Yair Feldman; Shankar Padmanabhan; Kianté Brantley; Yoav Artzi
>
> **摘要:** The scalability of large language models for long-context reasoning is severely constrained by the linear growth of their Transformer key-value cache, which incurs significant memory and computational costs. We posit that as a model generates reasoning tokens, the informational value of past generated tokens diminishes, creating an opportunity for compression. In this work, we propose to periodically compress the generation KV cache with a learned, special-purpose token and evict compressed entries. We train the model to perform this compression via a modified joint distillation and reinforcement learning (RL) framework. Our training method minimizes overhead over the conventional RL process, as it leverages RL outputs for distillation. Empirically, our method achieves a superior memory-accuracy Pareto frontier compared to both the model without cache compression and training-free compression techniques.
>
---
#### [replaced 015] ColorBench: Can VLMs See and Understand the Colorful World? A Comprehensive Benchmark for Color Perception, Reasoning, and Robustness
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.10514v3](http://arxiv.org/pdf/2504.10514v3)**

> **作者:** Yijun Liang; Ming Li; Chenrui Fan; Ziyue Li; Dang Nguyen; Kwesi Cobbina; Shweta Bhardwaj; Jiuhai Chen; Fuxiao Liu; Tianyi Zhou
>
> **备注:** Accepted by NeurIPS2025. 36 pages, including references and appendix. Code is available at https://github.com/tianyi-lab/ColorBench
>
> **摘要:** Color plays an important role in human perception and usually provides critical clues in visual reasoning. However, it is unclear whether and how vision-language models (VLMs) can perceive, understand, and leverage color as humans. This paper introduces ColorBench, an innovative benchmark meticulously crafted to assess the capabilities of VLMs in color understanding, including color perception, reasoning, and robustness. By curating a suite of diverse test scenarios, with grounding in real applications, ColorBench evaluates how these models perceive colors, infer meanings from color-based cues, and maintain consistent performance under varying color transformations. Through an extensive evaluation of 32 VLMs with varying language models and vision encoders, our paper reveals some undiscovered findings: (i) The scaling law (larger models are better) still holds on ColorBench, while the language model plays a more important role than the vision encoder. (ii) However, the performance gaps across models are relatively small, indicating that color understanding has been largely neglected by existing VLMs. (iii) CoT reasoning improves color understanding accuracies and robustness, though they are vision-centric tasks. (iv) Color clues are indeed leveraged by VLMs on ColorBench but they can also mislead models in some tasks. These findings highlight the critical limitations of current VLMs and underscore the need to enhance color comprehension. Our ColorBenchcan serve as a foundational tool for advancing the study of human-level color understanding of multimodal AI.
>
---
#### [replaced 016] Pralekha: Cross-Lingual Document Alignment for Indic Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.19096v3](http://arxiv.org/pdf/2411.19096v3)**

> **作者:** Sanjay Suryanarayanan; Haiyue Song; Mohammed Safi Ur Rahman Khan; Anoop Kunchukuttan; Raj Dabre
>
> **摘要:** Mining parallel document pairs for document-level machine translation (MT) remains challenging due to the limitations of existing Cross-Lingual Document Alignment (CLDA) techniques. Existing methods often rely on metadata such as URLs, which are scarce, or on pooled document representations that fail to capture fine-grained alignment cues. Moreover, the limited context window of sentence embedding models hinders their ability to represent document-level context, while sentence-based alignment introduces a combinatorially large search space, leading to high computational cost. To address these challenges for Indic languages, we introduce Pralekha, a benchmark containing over 3 million aligned document pairs across 11 Indic languages and English, which includes 1.5 million English-Indic pairs. Furthermore, we propose Document Alignment Coefficient (DAC), a novel metric for fine-grained document alignment. Unlike pooling-based methods, DAC aligns documents by matching smaller chunks and computes similarity as the ratio of aligned chunks to the average number of chunks in a pair. Intrinsic evaluation shows that our chunk-based method is 2-3x faster while maintaining competitive performance, and that DAC achieves substantial gains over pooling-based baselines. Extrinsic evaluation further demonstrates that document-level MT models trained on DAC-aligned pairs consistently outperform those using baseline alignment methods. These results highlight DAC's effectiveness for parallel document mining. The dataset and evaluation framework are publicly available to support further research.
>
---
#### [replaced 017] ReCode: Updating Code API Knowledge with Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; cs.SE**

- **链接: [http://arxiv.org/pdf/2506.20495v3](http://arxiv.org/pdf/2506.20495v3)**

> **作者:** Haoze Wu; Yunzhi Yao; Wenhao Yu; Ningyu Zhang
>
> **备注:** AAAI 2026
>
> **摘要:** Large Language Models (LLMs) exhibit remarkable code generation capabilities but falter when adapting to frequent updates in external library APIs. This critical limitation, stemming from reliance on outdated API knowledge from their training data, even with access to current documentation, impedes reliable code generation in dynamic environments. To tackle this issue, we propose ReCode (rule-based Reinforcement learning for Code Update), a novel framework that mimics human programmer adaptation to API changes. Specifically, we construct a dataset of approximately 2,000 data entries to train the LLMs to perform version migration based on updated information. Then, we introduce a modified string similarity metric for code evaluation as the reward for reinforcement learning. Our experiments demonstrate that ReCode substantially boosts LLMs' code generation performance in dynamic API scenarios, especially on the unseen CodeUpdateArena task. Crucially, compared to supervised fine-tuning, ReCode has less impact on LLMs' general code generation abilities. We apply ReCode on various LLMs and reinforcement learning algorithms (GRPO and DAPO), all achieving consistent improvements. Notably, after training, Qwen2.5-Coder-7B outperforms that of the 32B parameter code instruction-tuned model and the reasoning model with the same architecture. Code is available at https://github.com/zjunlp/ReCode.
>
---
#### [replaced 018] How Does a Deep Neural Network Look at Lexical Stress?
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.07229v2](http://arxiv.org/pdf/2508.07229v2)**

> **作者:** Itai Allouche; Itay Asael; Rotem Rousso; Vered Dassa; Ann Bradlow; Seung-Eun Kim; Matthew Goldrick; Joseph Keshet
>
> **备注:** 11 pages, 5 figures, submitted to the Journal of the Acoustical Society of America (JASA)
>
> **摘要:** Despite their success in speech processing, neural networks often operate as black boxes, prompting the question: what informs their decisions, and how can we interpret them? This work examines this issue in the context of lexical stress. A dataset of English disyllabic words was automatically constructed from read and spontaneous speech. Several Convolutional Neural Network (CNN) architectures were trained to predict stress position from a spectrographic representation of disyllabic words lacking minimal stress pairs (e.g., initial stress WAllet, final stress exTEND), achieving up to 92% accuracy on held-out test data. Layerwise Relevance Propagation (LRP), a technique for CNN interpretability analysis, revealed that predictions for held-out minimal pairs (PROtest vs. proTEST ) were most strongly influenced by information in stressed versus unstressed syllables, particularly the spectral properties of stressed vowels. However, the classifiers also attended to information throughout the word. A feature-specific relevance analysis is proposed, and its results suggest that our best-performing classifier is strongly influenced by the stressed vowel's first and second formants, with some evidence that its pitch and third formant also contribute. These results reveal deep learning's ability to acquire distributed cues to stress from naturally occurring data, extending traditional phonetic work based around highly controlled stimuli.
>
---
#### [replaced 019] The Markovian Thinker
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.06557v2](http://arxiv.org/pdf/2510.06557v2)**

> **作者:** Milad Aghajohari; Kamran Chitsaz; Amirhossein Kazemnejad; Sarath Chandar; Alessandro Sordoni; Aaron Courville; Siva Reddy
>
> **摘要:** Reinforcement learning (RL) has recently become a strong recipe for training reasoning LLMs that produce long chains of thought (LongCoT). Yet the standard RL "thinking environment", where the state is the prompt plus all prior reasoning tokens, makes the state unbounded and forces attention-based policies to pay quadratic compute as thoughts lengthen. We revisit the environment itself. We propose Markovian Thinking, a paradigm in which the policy advances reasoning while conditioning on a constant-size state, decoupling thinking length from context size. As an immediate consequence this yields linear compute with constant memory. We instantiate this idea with Delethink, an RL environment that structures reasoning into fixed-size chunks. Within each chunk, the model thinks as usual; at the boundary, the environment resets the context and reinitializes the prompt with a short carryover. Through RL, the policy learns to write a textual state near the end of each chunk sufficient for seamless continuation of reasoning after reset. Trained in this environment, an R1-Distill 1.5B model reasons in 8K-token chunks yet thinks up to 24K tokens, matching or surpassing LongCoT-RL trained with a 24K budget. With test-time scaling, Delethink continues to improve where LongCoT plateaus. The effect of linear compute is substantial: we empirically estimate at 96K average thinking length LongCoT-RL costs 27 H100-months vs. 7 for Delethink. Analysis at RL initialization shows off-the-shelf reasoning models (1.5B-120B) often sample Markovian traces zero-shot across diverse benchmarks, providing positive samples that make RL effective at scale. Our results show that redesigning the thinking environment is a powerful lever: it enables very long reasoning without quadratic overhead and opens a path toward efficient, scalable reasoning LLMs.
>
---
#### [replaced 020] ZK-SenseLM: Verifiable Large-Model Wireless Sensing with Selective Abstention and Zero-Knowledge Attestation
- **分类: cs.CR; cs.CL; C.2.1; D.4.6; E.3; I.2.6; I.5.4**

- **链接: [http://arxiv.org/pdf/2510.25677v2](http://arxiv.org/pdf/2510.25677v2)**

> **作者:** Hasan Akgul; Mari Eplik; Javier Rojas; Aina Binti Abdullah; Pieter van der Merwe
>
> **备注:** 45 pages
>
> **摘要:** ZK-SenseLM is a secure and auditable wireless sensing framework that pairs a large-model encoder for Wi-Fi channel state information (and optionally mmWave radar or RFID) with a policy-grounded decision layer and end-to-end zero-knowledge proofs of inference. The encoder uses masked spectral pretraining with phase-consistency regularization, plus a light cross-modal alignment that ties RF features to compact, human-interpretable policy tokens. To reduce unsafe actions under distribution shift, we add a calibrated selective-abstention head; the chosen risk-coverage operating point is registered and bound into the proof. We implement a four-stage proving pipeline: (C1) feature sanity and commitment, (C2) threshold and version binding, (C3) time-window binding, and (C4) PLONK-style proofs that the quantized network, given the committed window, produced the logged action and confidence. Micro-batched proving amortizes cost across adjacent windows, and a gateway option offloads proofs from low-power devices. The system integrates with differentially private federated learning and on-device personalization without weakening verifiability: model hashes and the registered threshold are part of each public statement. Across activity, presence or intrusion, respiratory proxy, and RF fingerprinting tasks, ZK-SenseLM improves macro-F1 and calibration, yields favorable coverage-risk curves under perturbations, and rejects tamper and replay with compact proofs and fast verification.
>
---
#### [replaced 021] Generative Medical Event Models Improve with Scale
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.12104v3](http://arxiv.org/pdf/2508.12104v3)**

> **作者:** Shane Waxler; Paul Blazek; Davis White; Daniel Sneider; Kevin Chung; Mani Nagarathnam; Patrick Williams; Hank Voeller; Karen Wong; Matthew Swanhorst; Sheng Zhang; Naoto Usuyama; Cliff Wong; Tristan Naumann; Hoifung Poon; Andrew Loza; Daniella Meeker; Seth Hain; Rahul Shah
>
> **摘要:** Realizing personalized medicine at scale calls for methods that distill insights from longitudinal patient journeys, which can be viewed as a sequence of medical events. Foundation models pretrained on large-scale medical event data represent a promising direction for scaling real-world evidence generation and generalizing to diverse downstream tasks. Using Epic Cosmos, a dataset with medical events from de-identified longitudinal health records for 16.3 billion encounters over 300 million unique patient records from 310 health systems, we introduce the Curiosity models, a family of decoder-only transformer models pretrained on 118 million patients representing 115 billion discrete medical events (151 billion tokens). We present the largest scaling-law study of medical event data, establishing a methodology for pretraining and revealing power-law scaling relationships for compute, tokens, and model size. Consequently, we pretrained a series of compute-optimal models with up to 1 billion parameters. Conditioned on a patient's real-world history, Curiosity autoregressively predicts the next medical event to simulate patient health timelines. We studied 78 real-world tasks, including diagnosis prediction, disease prognosis, and healthcare operations. Remarkably for a foundation model with generic pretraining and simulation-based inference, Curiosity generally outperformed or matched task-specific supervised models on these tasks, without requiring task-specific fine-tuning or few-shot examples. Curiosity's predictive power consistently improves as the model and pretraining scale. Our results show that Curiosity, a generative medical event foundation model, can effectively capture complex clinical dynamics, providing an extensible and generalizable framework to support clinical decision-making, streamline healthcare operations, and improve patient outcomes.
>
---
#### [replaced 022] Hallucination as an Upper Bound: A New Perspective on Text-to-Image Evaluation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.21257v2](http://arxiv.org/pdf/2509.21257v2)**

> **作者:** Seyed Amir Kasaei; Mohammad Hossein Rohban
>
> **备注:** Accepted at GenProCC NeurIPS 2025 Workshop
>
> **摘要:** In language and vision-language models, hallucination is broadly understood as content generated from a model's prior knowledge or biases rather than from the given input. While this phenomenon has been studied in those domains, it has not been clearly framed for text-to-image (T2I) generative models. Existing evaluations mainly focus on alignment, checking whether prompt-specified elements appear, but overlook what the model generates beyond the prompt. We argue for defining hallucination in T2I as bias-driven deviations and propose a taxonomy with three categories: attribute, relation, and object hallucinations. This framing introduces an upper bound for evaluation and surfaces hidden biases, providing a foundation for richer assessment of T2I models.
>
---
#### [replaced 023] HaluMem: Evaluating Hallucinations in Memory Systems of Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2511.03506v2](http://arxiv.org/pdf/2511.03506v2)**

> **作者:** Ding Chen; Simin Niu; Kehang Li; Peng Liu; Xiangping Zheng; Bo Tang; Xinchi Li; Feiyu Xiong; Zhiyu Li
>
> **摘要:** Memory systems are key components that enable AI systems such as LLMs and AI agents to achieve long-term learning and sustained interaction. However, during memory storage and retrieval, these systems frequently exhibit memory hallucinations, including fabrication, errors, conflicts, and omissions. Existing evaluations of memory hallucinations are primarily end-to-end question answering, which makes it difficult to localize the operational stage within the memory system where hallucinations arise. To address this, we introduce the Hallucination in Memory Benchmark (HaluMem), the first operation level hallucination evaluation benchmark tailored to memory systems. HaluMem defines three evaluation tasks (memory extraction, memory updating, and memory question answering) to comprehensively reveal hallucination behaviors across different operational stages of interaction. To support evaluation, we construct user-centric, multi-turn human-AI interaction datasets, HaluMem-Medium and HaluMem-Long. Both include about 15k memory points and 3.5k multi-type questions. The average dialogue length per user reaches 1.5k and 2.6k turns, with context lengths exceeding 1M tokens, enabling evaluation of hallucinations across different context scales and task complexities. Empirical studies based on HaluMem show that existing memory systems tend to generate and accumulate hallucinations during the extraction and updating stages, which subsequently propagate errors to the question answering stage. Future research should focus on developing interpretable and constrained memory operation mechanisms that systematically suppress hallucinations and improve memory reliability.
>
---
#### [replaced 024] GRDD+: An Extended Greek Dialectal Dataset with Cross-Architecture Fine-tuning Evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2511.03772v2](http://arxiv.org/pdf/2511.03772v2)**

> **作者:** Stergios Chatzikyriakidis; Dimitris Papadakis; Sevasti-Ioanna Papaioannou; Erofili Psaltaki
>
> **摘要:** We present an extended Greek Dialectal Dataset (GRDD+) 1that complements the existing GRDD dataset with more data from Cretan, Cypriot, Pontic and Northern Greek, while we add six new varieties: Greco-Corsican, Griko (Southern Italian Greek), Maniot, Heptanesian, Tsakonian, and Katharevusa Greek. The result is a dataset with total size 6,374,939 words and 10 varieties. This is the first dataset with such variation and size to date. We conduct a number of fine-tuning experiments to see the effect of good quality dialectal data on a number of LLMs. We fine-tune three model architectures (Llama-3-8B, Llama-3.1-8B, Krikri-8B) and compare the results to frontier models (Claude-3.7-Sonnet, Gemini-2.5, ChatGPT-5).
>
---
#### [replaced 025] All Entities are Not Created Equal: Examining the Long Tail for Ultra-Fine Entity Typing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.17355v3](http://arxiv.org/pdf/2410.17355v3)**

> **作者:** Advait Deshmukh; Ashwin Umadi; Dananjay Srinivas; Maria Leonor Pacheco
>
> **摘要:** Due to their capacity to acquire world knowledge from large corpora, pre-trained language models (PLMs) are extensively used in ultra-fine entity typing tasks where the space of labels is extremely large. In this work, we explore the limitations of the knowledge acquired by PLMs by proposing a novel heuristic to approximate the pre-training distribution of entities when the pre-training data is unknown. Then, we systematically demonstrate that entity-typing approaches that rely solely on the parametric knowledge of PLMs struggle significantly with entities at the long tail of the pre-training distribution, and that knowledge-infused approaches can account for some of these shortcomings. Our findings suggest that we need to go beyond PLMs to produce solutions that perform well for infrequent entities.
>
---
#### [replaced 026] UnsafeChain: Enhancing Reasoning Model Safety via Hard Cases
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.21652v2](http://arxiv.org/pdf/2507.21652v2)**

> **作者:** Raj Vardhan Tomar; Preslav Nakov; Yuxia Wang
>
> **摘要:** As large reasoning models (LRMs) grow more capable, chain-of-thought (CoT) reasoning introduces new safety challenges. Existing SFT-based safety alignment studies dominantly focused on filtering prompts with safe, high-quality responses, while overlooking hard prompts that always elicit harmful outputs. To fill this gap, we introduce UnsafeChain, a safety alignment dataset constructed from hard prompts with diverse sources, where unsafe completions are identified and explicitly corrected into safe responses. By exposing models to unsafe behaviors and guiding their correction, UnsafeChain enhances safety while preserving general reasoning ability. We fine-tune three LRMs on UnsafeChain and compare them against recent SafeChain and STAR-1 across six out-of-distribution and five in-distribution benchmarks. UnsafeChain consistently outperforms prior datasets, with even a 1K subset matching or surpassing baseline performance, demonstrating the effectiveness and generalizability of correction-based supervision. We release our dataset and code at https://github.com/mbzuai-nlp/UnsafeChain
>
---
#### [replaced 027] Continual Pre-training of MoEs: How robust is your router?
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.05029v2](http://arxiv.org/pdf/2503.05029v2)**

> **作者:** Benjamin Thérien; Charles-Étienne Joseph; Zain Sarwar; Ashwinee Panda; Anirban Das; Shi-Xiong Zhang; Stephen Rawls; Sambit Sahu; Eugene Belilovsky; Irina Rish
>
> **摘要:** Sparsely-activated Mixture of Experts (MoE) transformers are promising architectures for foundation models. Compared to dense transformers that require the same amount of floating-point operations (FLOPs) per forward pass, MoEs benefit from improved sample efficiency at training time and achieve much stronger performance. Many closed-source and open-source frontier language models have thus adopted an MoE architecture. Naturally, practitioners will want to extend the capabilities of these models with large amounts of newly collected data without completely re-training them. Prior work has shown that a simple combination of replay, learning rate re-warming, and re-decaying can enable the continual pre-training (CPT) of dense decoder-only transformers with minimal performance degradation compared to full re-training. In the case of decoder-only MoE transformers, however, it is unclear how the routing algorithm will impact continual pre-training performance: 1) do the MoE transformer's routers exacerbate forgetting relative to a dense model?; 2) do the routers maintain a balanced load on previous distributions after CPT?; 3) are the same strategies applied to dense models sufficient to continually pre-train MoE LLMs? In what follows, we conduct a large-scale study training a 500M parameter dense transformer and four 500M-active/2B-total parameter MoE transformers. Each model is trained for 600B tokens. Our results establish a surprising robustness to distribution shifts for MoEs using both Sinkhorn-Balanced and Z-and-Aux-loss-balanced routing algorithms, even in MoEs continually pre-trained without replay. Moreover, we show that MoE LLMs maintain their sample efficiency (relative to a FLOP-matched dense model) during CPT and that they can match the performance of a fully re-trained MoE at a fraction of the cost.
>
---
#### [replaced 028] Minimal and Mechanistic Conditions for Behavioral Self-Awareness in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2511.04875v2](http://arxiv.org/pdf/2511.04875v2)**

> **作者:** Matthew Bozoukov; Matthew Nguyen; Shubkarman Singh; Bart Bussmann; Patrick Leask
>
> **摘要:** Recent studies have revealed that LLMs can exhibit behavioral self-awareness: the ability to accurately describe or predict their own learned behaviors without explicit supervision. This capability raises safety concerns as it may, for example, allow models to better conceal their true abilities during evaluation. We attempt to characterize the minimal conditions under which such self-awareness emerges, and the mechanistic processes through which it manifests. Through controlled finetuning experiments on instruction-tuned LLMs with low-rank adapters (LoRA), we find: (1) that self-awareness can be reliably induced using a single rank-1 LoRA adapter; (2) that the learned self-aware behavior can be largely captured by a single steering vector in activation space, recovering nearly all of the fine-tune's behavioral effect; and (3) that self-awareness is non-universal and domain-localized, with independent representations across tasks. Together, these findings suggest that behavioral self-awareness emerges as a domain-specific, linear feature that can be easily induced and modulated.
>
---
#### [replaced 029] DiscoTrack: A Multilingual LLM Benchmark for Discourse Tracking
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.17013v3](http://arxiv.org/pdf/2510.17013v3)**

> **作者:** Lanni Bu; Lauren Levine; Amir Zeldes
>
> **摘要:** Recent LLM benchmarks have tested models on a range of phenomena, but are still focused primarily on natural language understanding for extraction of explicit information, such as QA or summarization, with responses often tar- geting information from individual sentences. We are still lacking more challenging, and im- portantly also multilingual, benchmarks focus- ing on implicit information and pragmatic infer- ences across larger documents in the context of discourse tracking: integrating and aggregating information across sentences, paragraphs and multiple speaker utterances. To this end, we present DiscoTrack, an LLM benchmark target- ing a range of tasks across 12 languages and four levels of discourse understanding: salience recognition, entity tracking, discourse relations and bridging inference. Our evaluation shows that these tasks remain challenging, even for state-of-the-art models.
>
---
#### [replaced 030] AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2511.02376v2](http://arxiv.org/pdf/2511.02376v2)**

> **作者:** Aashray Reddy; Andrew Zagula; Nicholas Saban; Kevin Zhu
>
> **备注:** Accepted to NeurIPS 2025 Lock-LLM Workshop. Code is available at https://github.com/AAN-AutoAdv/AutoAdv
>
> **摘要:** Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs, yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves up to 95% attack success rate on Llama-3.1-8B within six turns a 24 percent improvement over single turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests then iteratively refines them. Extensive evaluation across commercial and open-source models (GPT-4o-mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.
>
---
#### [replaced 031] Describe Where You Are: Improving Noise-Robustness for Speech Emotion Recognition with Text Description of the Environment
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.17716v2](http://arxiv.org/pdf/2407.17716v2)**

> **作者:** Seong-Gyun Leem; Daniel Fulford; Jukka-Pekka Onnela; David Gard; Carlos Busso
>
> **摘要:** Speech emotion recognition (SER) systems often struggle in real-world environments, where ambient noise severely degrades their performance. This paper explores a novel approach that exploits prior knowledge of testing environments to maximize SER performance under noisy conditions. To address this task, we propose a text-guided, environment-aware training where an SER model is trained with contaminated speech samples and their paired noise description. We use a pre-trained text encoder to extract the text-based environment embedding and then fuse it to a transformer-based SER model during training and inference. We demonstrate the effectiveness of our approach through our experiment with the MSP-Podcast corpus and real-world additive noise samples collected from the Freesound and DEMAND repositories. Our experiment indicates that the text-based environment descriptions processed by a large language model (LLM) produce representations that improve the noise-robustness of the SER system. With a contrastive learning (CL)-based representation, our proposed method can be improved by jointly fine-tuning the text encoder with the emotion recognition model. Under the -5dB signal-to-noise ratio (SNR) level, fine-tuning the text encoder improves our CL-based representation method by 76.4% (arousal), 100.0% (dominance), and 27.7% (valence).
>
---
#### [replaced 032] SDS KoPub VDR: A Benchmark Dataset for Visual Document Retrieval in Korean Public Documents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2511.04910v2](http://arxiv.org/pdf/2511.04910v2)**

> **作者:** Jaehoon Lee; Sohyun Kim; Wanggeun Park; Geon Lee; Seungkyung Kim; Minyoung Lee
>
> **备注:** 27 pages, 15 figures, 6 tables
>
> **摘要:** Existing benchmarks for visual document retrieval (VDR) largely overlook non-English languages and the structural complexity of official publications. To address this gap, we introduce SDS KoPub VDR, the first large-scale, public benchmark for retrieving and understanding Korean public documents. The benchmark is built upon 361 real-world documents, including 256 files under the KOGL Type 1 license and 105 from official legal portals, capturing complex visual elements like tables, charts, and multi-column layouts. To establish a reliable evaluation set, we constructed 600 query-page-answer triples. These were initially generated using multimodal models (e.g., GPT-4o) and subsequently underwent human verification to ensure factual accuracy and contextual relevance. The queries span six major public domains and are categorized by the reasoning modality required: text-based, visual-based, and cross-modal. We evaluate SDS KoPub VDR on two complementary tasks: (1) text-only retrieval and (2) multimodal retrieval, which leverages visual features alongside text. This dual-task evaluation reveals substantial performance gaps, particularly in multimodal scenarios requiring cross-modal reasoning, even for state-of-the-art models. As a foundational resource, SDS KoPub VDR enables rigorous and fine-grained evaluation and provides a roadmap for advancing multimodal AI in real-world document intelligence. The dataset is available at https://huggingface.co/datasets/SamsungSDS-Research/SDS-KoPub-VDR-Benchmark.
>
---
#### [replaced 033] Learning Task Representations from In-Context Learning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.05390v2](http://arxiv.org/pdf/2502.05390v2)**

> **作者:** Baturay Saglam; Xinyang Hu; Zhuoran Yang; Dionysis Kalogerias; Amin Karbasi
>
> **备注:** ACL Findings 2025
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable proficiency in in-context learning (ICL), where models adapt to new tasks through example-based prompts without requiring parameter updates. However, understanding how tasks are internally encoded and generalized remains a challenge. To address some of the empirical and technical gaps in the literature, we introduce an automated formulation for encoding task information in ICL prompts as a function of attention heads within the transformer architecture. This approach computes a single task vector as a weighted sum of attention heads, with the weights optimized causally via gradient descent. Our findings show that existing methods fail to generalize effectively to modalities beyond text. In response, we also design a benchmark to evaluate whether a task vector can preserve task fidelity in functional regression tasks. The proposed method successfully extracts task-specific information from in-context demonstrations and excels in both text and regression tasks, demonstrating its generalizability across modalities.
>
---
#### [replaced 034] Evaluating, Synthesizing, and Enhancing for Customer Support Conversation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.04423v2](http://arxiv.org/pdf/2508.04423v2)**

> **作者:** Jie Zhu; Huaixia Dou; Junhui Li; Lifan Guo; Feng Chen; Chi Zhang; Fang Kong
>
> **备注:** Accepted by AAAI-2026
>
> **摘要:** Effective customer support requires not only accurate problem solving but also structured and empathetic communication aligned with professional standards. However, existing dialogue datasets often lack strategic guidance, and real-world service data is difficult to access and annotate. To address this, we introduce the task of Customer Support Conversation (CSC), aimed at training customer service agents to respond using well-defined support strategies. We propose a structured CSC framework grounded in COPC guidelines, defining five conversational stages and twelve strategies to guide high-quality interactions. Based on this, we construct CSConv, an evaluation dataset of 1,855 real-world customer-agent conversations rewritten using LLMs to reflect deliberate strategy use, and annotated accordingly. Additionally, we develop a role-playing approach that simulates strategy-rich conversations using LLM-powered roles aligned with the CSC framework, resulting in the training dataset RoleCS. Experiments show that fine-tuning strong LLMs on RoleCS significantly improves their ability to generate high-quality, strategy-aligned responses on CSConv. Human evaluations further confirm gains in problem resolution. All code and data will be made publicly available at https://github.com/aliyun/qwen-dianjin.
>
---
#### [replaced 035] Robustness of Neurosymbolic Reasoners on First-Order Logic Problems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.17377v2](http://arxiv.org/pdf/2509.17377v2)**

> **作者:** Hannah Bansal; Kemal Kurniawan; Lea Frermann
>
> **备注:** Accepted to ALA Conference
>
> **摘要:** Recent trends in NLP aim to improve reasoning capabilities in Large Language Models (LLMs), with key focus on generalization and robustness to variations in tasks. Counterfactual task variants introduce minimal but semantically meaningful changes to otherwise valid first-order logic (FOL) problem instances altering a single predicate or swapping roles of constants to probe whether a reasoning system can maintain logical consistency under perturbation. Previous studies showed that LLMs becomes brittle on counterfactual variations, suggesting that they often rely on spurious surface patterns to generate responses. In this work, we explore if a neurosymbolic (NS) approach that integrates an LLM and a symbolic logical solver could mitigate this problem. Experiments across LLMs of varying sizes show that NS methods are more robust but perform worse overall that purely neural methods. We then propose NSCoT that combines an NS method and Chain-of-Thought (CoT) prompting and demonstrate that while it improves performance, NSCoT still lags behind standard CoT. Our analysis opens research directions for future work.
>
---
#### [replaced 036] CoSense-LLM: Semantics at the Edge with Cost- and Uncertainty-Aware Cloud-Edge Cooperation
- **分类: cs.CL; I.2.6; C.2.4; C.3**

- **链接: [http://arxiv.org/pdf/2510.19670v2](http://arxiv.org/pdf/2510.19670v2)**

> **作者:** Hasan Akgul; Mari Eplik; Javier Rojas; Aina Binti Abdullah; Pieter van der Merwe
>
> **备注:** 19 pages,8 figures
>
> **摘要:** We present CoSense-LLM, an edge-first framework that turns continuous multimodal sensor streams (for example Wi-Fi CSI, IMU, audio, RFID, and lightweight vision) into compact, verifiable semantic tokens and coordinates with large language models under explicit latency, energy, bandwidth, and privacy constraints. CoSense-LLM has four parts: (i) SenseFusion, a lightweight encoder that aligns sensor embeddings with language and compresses them into short discrete code sequences; (ii) Edge-RAG, a local hybrid retrieval layer that grounds generation in site specific policies and notes; (iii) PromptRouter, a cost and uncertainty aware policy that selects edge only generation, edge plus retrieval, or compact cloud escalation; and (iv) Secure Execution, an auditable redaction path that enforces data minimization so raw waveforms never leave the device. The system works with modern serving optimizations, including paged or streaming KV caches, FlashAttention style kernels, speculative decoding, and quantized LoRA adapters, and supports on device personalization and federated updates under non IID drift. Across home, office, and clinic deployments, CoSense-LLM delivers grounded explanations while meeting tight service level objectives: it sustains sub second (p95) end to end latency on edge dominant paths, reduces inter tier token and bandwidth costs by preferring local retrieval grounded responses, and preserves privacy by transmitting only discrete codes and redacted metadata. Ablations show that Edge-RAG improves factual consistency and reduces contradictions, calibrated uncertainty enables selective abstention and controlled escalations, and KV plus decoding accelerators lower energy per decision. The results support an edge first design that treats semantics, privacy, and predictable latency as co equal goals for large model deployments in interference prone environments.
>
---
#### [replaced 037] SageLM: A Multi-aspect and Explainable Large Language Model for Speech Judgement
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.20916v2](http://arxiv.org/pdf/2508.20916v2)**

> **作者:** Yuan Ge; Junxiang Zhang; Xiaoqian Liu; Bei Li; Xiangnan Ma; Chenglong Wang; Kaiyang Ye; Yangfan Du; Linfeng Zhang; Yuxin Huang; Tong Xiao; Zhengtao Yu; JingBo Zhu
>
> **摘要:** Speech-to-Speech (S2S) Large Language Models (LLMs) are foundational to natural human-computer interaction, enabling end-to-end spoken dialogue systems. However, evaluating these models remains a fundamental challenge. We propose \texttt{SageLM}, an end-to-end, multi-aspect, and explainable speech LLM for comprehensive S2S LLMs evaluation. First, unlike cascaded approaches that disregard acoustic features, SageLM jointly assesses both semantic and acoustic dimensions. Second, it leverages rationale-based supervision to enhance explainability and guide model learning, achieving superior alignment with evaluation outcomes compared to rule-based reinforcement learning methods. Third, we introduce \textit{SpeechFeedback}, a synthetic preference dataset, and employ a two-stage training paradigm to mitigate the scarcity of speech preference data. Trained on both semantic and acoustic dimensions, SageLM achieves an 82.79\% agreement rate with human evaluators, outperforming cascaded and SLM-based baselines by at least 7.42\% and 26.20\%, respectively.
>
---
#### [replaced 038] Inside CORE-KG: Evaluating Structured Prompting and Coreference Resolution for Knowledge Graphs
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.26512v2](http://arxiv.org/pdf/2510.26512v2)**

> **作者:** Dipak Meher; Carlotta Domeniconi
>
> **备注:** ICDM 2025
>
> **摘要:** Human smuggling networks are increasingly adaptive and difficult to analyze. Legal case documents offer critical insights but are often unstructured, lexically dense, and filled with ambiguous or shifting references, which pose significant challenges for automated knowledge graph (KG) construction. While recent LLM-based approaches improve over static templates, they still generate noisy, fragmented graphs with duplicate nodes due to the absence of guided extraction and coreference resolution. The recently proposed CORE-KG framework addresses these limitations by integrating a type-aware coreference module and domain-guided structured prompts, significantly reducing node duplication and legal noise. In this work, we present a systematic ablation study of CORE-KG to quantify the individual contributions of its two key components. Our results show that removing coreference resolution results in a 28.25% increase in node duplication and a 4.32% increase in noisy nodes, while removing structured prompts leads to a 4.29% increase in node duplication and a 73.33% increase in noisy nodes. These findings offer empirical insights for designing robust LLM-based pipelines for extracting structured representations from complex legal texts.
>
---
#### [replaced 039] How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.02904v3](http://arxiv.org/pdf/2504.02904v3)**

> **作者:** Hongzhe Du; Weikai Li; Min Cai; Karim Saraipour; Zimin Zhang; Himabindu Lakkaraju; Yizhou Sun; Shichang Zhang
>
> **备注:** COLM 2025
>
> **摘要:** Post-training is essential for the success of large language models (LLMs), transforming pre-trained base models into more useful and aligned post-trained models. While plenty of works have studied post-training algorithms and evaluated post-training models by their outputs, it remains understudied how post-training reshapes LLMs internally. In this paper, we compare base and post-trained LLMs mechanistically from four perspectives to better understand post-training effects. Our findings across model families and datasets reveal that: (1) Post-training does not change the factual knowledge storage locations, and it adapts knowledge representations from the base model while developing new knowledge representations; (2) Both truthfulness and refusal can be represented by vectors in the hidden representation space. The truthfulness direction is highly similar between the base and post-trained model, and it is effectively transferable for interventions; (3) The refusal direction is different between the base and post-trained models, and it shows limited forward transferability; (4) Differences in confidence between the base and post-trained models cannot be attributed to entropy neurons. Our study provides insights into the fundamental mechanisms preserved and altered during post-training, facilitates downstream tasks like model steering, and could potentially benefit future research in interpretability and LLM post-training. Our code is publicly available at https://github.com/HZD01/post-training-mechanistic-analysis.
>
---
#### [replaced 040] Zeroth-Order Adaptive Neuron Alignment Based Pruning without Re-Training
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.07066v4](http://arxiv.org/pdf/2411.07066v4)**

> **作者:** Elia Cunegatti; Leonardo Lucio Custode; Giovanni Iacca
>
> **备注:** Published in Transactions on Machine Learning Research (TMLR)
>
> **摘要:** Network pruning focuses on algorithms that aim to reduce a given model's computational cost by removing a subset of its parameters while having minimal impact on performance. Throughout the last decade, the most widely used pruning paradigm has been pruning and re-training, which nowadays is inconvenient due to the vast amount of pre-trained models, which are, in any case, too expensive to re-train. In this paper, we exploit functional information from dense pre-trained models, i.e., their input activations, to obtain sparse models that maximize the activations' alignment with respect to their corresponding dense models. Hence, we propose \textbf{NeuroAl}, a \emph{top-up} algorithm that can be used on top of any given pruning algorithm for LLMs, which modifies the block-wise and row-wise sparsity, exploiting information from both the dense model and its sparse version to maximize the \emph{neuron alignment} among activations. Different from existing methods, our approach adaptively selects the best hyperparameters for the block-wise and row-wise sparsity ratios w.r.t. the model and the desired sparsity, and requires \emph{no re-training}. We test our method over $\sim$300 test cases with four LLM families, three sparsity ratios, and ten language tasks (three language modeling and seven zero-shot datasets), showing how it consistently outperforms the latest state-of-the-art methods in terms of performance-runtime trade-off. The code is available at \href{https://github.com/eliacunegatti/NeuroAL}{https://github.com/eliacunegatti/NeuroAL}.
>
---
#### [replaced 041] RareAgents: Autonomous Multi-disciplinary Team for Rare Disease Diagnosis and Treatment
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.12475v3](http://arxiv.org/pdf/2412.12475v3)**

> **作者:** Xuanzhong Chen; Ye Jin; Xiaohao Mao; Lun Wang; Shuyang Zhang; Ting Chen
>
> **备注:** AAAI2026 Oral
>
> **摘要:** Rare diseases, despite their low individual incidence, collectively impact around 300 million people worldwide due to the vast number of diseases. The involvement of multiple organs and systems, and the shortage of specialized doctors with relevant experience, make diagnosing and treating rare diseases more challenging than common diseases. Recently, agents powered by large language models (LLMs) have demonstrated notable applications across various domains. In the medical field, some agent methods have outperformed direct prompts in question-answering tasks from medical examinations. However, current agent frameworks are not well-adapted to real-world clinical scenarios, especially those involving the complex demands of rare diseases. To bridge this gap, we introduce RareAgents, the first LLM-driven multi-disciplinary team decision-support tool designed specifically for the complex clinical context of rare diseases. RareAgents integrates advanced Multidisciplinary Team (MDT) coordination, memory mechanisms, and medical tools utilization, leveraging Llama-3.1-8B/70B as the base model. Experimental results show that RareAgents outperforms state-of-the-art domain-specific models, GPT-4o, and current agent frameworks in diagnosis and treatment for rare diseases. Furthermore, we contribute a novel rare disease dataset, MIMIC-IV-Ext-Rare, to facilitate further research in this field.
>
---
#### [replaced 042] DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.24332v2](http://arxiv.org/pdf/2505.24332v2)**

> **作者:** Wenxuan Shi; Haochen Tan; Chuqiao Kuang; Xiaoguang Li; Xiaozhe Ren; Chen Zhang; Hanting Chen; Yasheng Wang; Lu Hou; Lifeng Shang
>
> **备注:** Accepted as NeurIPS 2025 Spotlight
>
> **摘要:** Information seeking demands iterative evidence gathering and reflective reasoning, yet large language models (LLMs) still struggle with it in open-web question answering. Existing prompting and supervised fine-tuning (SFT) methods remain fixed by prompt rules or training corpora, and are usually benchmarked only on well-structured wiki sources, limiting real-world adaptability. We introduce WebPuzzle, a 24k-sample training and 275-sample test benchmark that evaluates information seeking on the live internet, across both wiki and open-domain queries. Leveraging 7k WebPuzzle instances, we develop DeepDiver, a reinforcement-learning (RL) framework that cultivates Search Intensity Scaling (SIS)-an emergent ability to escalate search frequency and depth instead of settling on overconfident, under-evidenced answers. With SIS, Qwen2.5-7B-Instruct and Pangu-7B-Reasoner attain performance on real-web tasks comparable to the 671B-parameter DeepSeek-R1. We detail DeepDiver's curriculum from cold-start SFT to a well designed RL procedure, and show that its seeking policy generalized from closed-ended queries to open-ended generation such as long-form writing. Our results advance adaptive information seeking in LLMs and provide a rigorous benchmark for future work.
>
---
#### [replaced 043] Invoke Interfaces Only When Needed: Adaptive Invocation for Large Language Models in Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.02311v2](http://arxiv.org/pdf/2505.02311v2)**

> **作者:** Jihao Zhao; Chunlai Zhou; Daixuan Li; Shuaishuai Zu; Biao Qin
>
> **摘要:** The collaborative paradigm of large and small language models (LMs) effectively balances performance and cost, yet its pivotal challenge lies in precisely pinpointing the moment of invocation when hallucinations arise in small LMs. Previous optimization efforts primarily focused on post-processing techniques, which were separate from the reasoning process of LMs, resulting in high computational costs and limited effectiveness. In this paper, we propose a practical invocation evaluation metric called AttenHScore, which calculates the accumulation and propagation of hallucinations during the generation process of small LMs, continuously amplifying potential reasoning errors. By dynamically adjusting the detection threshold, we achieve more accurate real-time invocation of large LMs. Additionally, considering the limited reasoning capacity of small LMs, we leverage uncertainty-aware knowledge reorganization to assist them better capture critical information from different text chunks. Extensive experiments reveal that our AttenHScore outperforms most baselines in enhancing real-time hallucination detection capabilities across multiple QA datasets, especially when addressing complex queries. Moreover, our strategies eliminate the need for additional model training and display flexibility in adapting to various transformer-based LMs.
>
---
#### [replaced 044] Evaluating Test-Time Scaling LLMs for Legal Reasoning: OpenAI o1, DeepSeek-R1, and Beyond
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.16040v2](http://arxiv.org/pdf/2503.16040v2)**

> **作者:** Yinghao Hu; Yaoyao Yu; Leilei Gan; Bin Wei; Kun Kuang; Fei Wu
>
> **备注:** 23 pages, Published in Findings of the Association for Computational Linguistics: EMNLP 2025
>
> **摘要:** Recent advances in test-time scaling of large language models (LLMs), exemplified by DeepSeek-R1 and OpenAI's o1, show that extending the chain of thought during inference can significantly improve general reasoning performance. However, the impact of this paradigm on legal reasoning remains insufficiently explored. To address this gap, we present the first systematic evaluation of 12 LLMs, including both reasoning-focused and general-purpose models, across 17 Chinese and English legal tasks spanning statutory and case-law traditions. In addition, we curate a bilingual chain-of-thought dataset for legal reasoning through distillation from DeepSeek-R1 and develop Legal-R1, an open-source model specialized for the legal domain. Experimental results show that Legal-R1 delivers competitive performance across diverse tasks. DeepSeek-R1 exhibits clear advantages in Chinese legal reasoning, while OpenAI's o1 achieves comparable results on English tasks. We further conduct a detailed error analysis, which reveals recurring issues such as outdated legal knowledge, limited capacity for legal interpretation, and susceptibility to factual hallucinations. These findings delineate the main obstacles confronting legal-domain LLMs and suggest promising directions for future research.
>
---
#### [replaced 045] How Efficient Are Diffusion Language Models? A Critical Examination of Efficiency Evaluation Practices
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.18480v3](http://arxiv.org/pdf/2510.18480v3)**

> **作者:** Han Peng; Peiyu Liu; Zican Dong; Daixuan Cheng; Junyi Li; Yiru Tang; Shuo Wang; Wayne Xin Zhao
>
> **摘要:** Diffusion language models (DLMs) have emerged as a promising alternative to the long-dominant autoregressive (AR) paradigm, offering a parallelable decoding process that could yield greater efficiency. Yet, in practice, current open-source DLMs often underperform their AR counterparts in speed, limiting their real-world utility. This work presents a systematic study of DLM efficiency, identifying key issues in prior evaluation methods. Through empirical benchmarking and a theoretical analysis, we demonstrate that AR models generally achieve higher throughput, while DLMs consistently lag. We also investigate acceleration strategies, finding that techniques like dual cache and parallel decoding mainly offer gains at small batch sizes, with their benefits diminishing upon scaling. Our findings underscore the necessity of robust evaluation methods and improved acceleration strategies to advance research on DLMs.
>
---
#### [replaced 046] ECLeKTic: a Novel Challenge Set for Evaluation of Cross-Lingual Knowledge Transfer
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.21228v3](http://arxiv.org/pdf/2502.21228v3)**

> **作者:** Omer Goldman; Uri Shaham; Dan Malkin; Sivan Eiger; Avinatan Hassidim; Yossi Matias; Joshua Maynez; Adi Mayrav Gilady; Jason Riesa; Shruti Rijhwani; Laura Rimell; Idan Szpektor; Reut Tsarfaty; Matan Eyal
>
> **摘要:** To achieve equitable performance across languages, large language models (LLMs) must be able to abstract knowledge beyond the language in which it was learnt. However, the current literature lacks reliable ways to measure LLMs' capability of such cross-lingual knowledge transfer. To that end, we present ECLeKTic, a multilingual closed-book QA dataset that Evaluates Cross-Lingual Knowledge Transfer in a simple, black-box manner. Concretely, we used the presence and absence of Wikipedia articles in 12 languages to detect pieces of information that were likely available during pre-training in one of the languages but not in the others. We curate ECLeKTic as a set of fact-seeking questions over this kind of information, in all the different languages. Therefore, in order to solve ECLeKTic the model is required to transfer knowledge between languages. We evaluated 8 LLMs and showed that current SOTA models struggle to effectively share knowledge across languages, even if they can predict the answer for questions in the language in which the knowledge was acquired.
>
---
#### [replaced 047] DiLA: Enhancing LLM Tool Learning with Differential Logic Layer
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2402.11903v4](http://arxiv.org/pdf/2402.11903v4)**

> **作者:** Yu Zhang; Hui-Ling Zhen; Zehua Pei; Yingzhao Lian; Lihao Yin; Mingxuan Yuan; Bei Yu
>
> **备注:** arXiv admin note: text overlap with arXiv:2305.12295 by other authors
>
> **摘要:** Considering the challenges faced by large language models (LLMs) in logical reasoning and planning, prior efforts have sought to augment LLMs with access to external solvers. While progress has been made on simple reasoning problems, solving classical constraint satisfaction problems, such as the Boolean Satisfiability Problem (SAT) and Graph Coloring Problem (GCP), remains difficult for off-the-shelf solvers due to their intricate expressions and exponential search spaces. In this paper, we propose a novel differential logic layer-aided language modeling (DiLA) approach, where logical constraints are integrated into the forward and backward passes of a network layer, to provide another option for LLM tool learning. In DiLA, LLM aims to transform the language description to logic constraints and identify initial solutions of the highest quality, while the differential logic layer focuses on iteratively refining the LLM-prompted solution. Leveraging the logic layer as a bridge, DiLA enhances the logical reasoning ability of LLMs on a range of reasoning problems encoded by Boolean variables, guaranteeing the efficiency and correctness of the solution process. We evaluate the performance of DiLA on two classic reasoning problems and empirically demonstrate its consistent outperformance against existing prompt-based and solver-aided approaches.
>
---
#### [replaced 048] DrKGC: Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion across General and Biomedical Domains
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.00708v3](http://arxiv.org/pdf/2506.00708v3)**

> **作者:** Yongkang Xiao; Sinian Zhang; Yi Dai; Huixue Zhou; Jue Hou; Jie Ding; Rui Zhang
>
> **备注:** Accepted at EMNLP 2025 Findings
>
> **摘要:** Knowledge graph completion (KGC) aims to predict missing triples in knowledge graphs (KGs) by leveraging existing triples and textual information. Recently, generative large language models (LLMs) have been increasingly employed for graph tasks. However, current approaches typically encode graph context in textual form, which fails to fully exploit the potential of LLMs for perceiving and reasoning about graph structures. To address this limitation, we propose DrKGC (Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion). DrKGC employs a flexible lightweight model training strategy to learn structural embeddings and logical rules within the KG. It then leverages a novel bottom-up graph retrieval method to extract a subgraph for each query guided by the learned rules. Finally, a graph convolutional network (GCN) adapter uses the retrieved subgraph to enhance the structural embeddings, which are then integrated into the prompt for effective LLM fine-tuning. Experimental results on two general domain benchmark datasets and two biomedical datasets demonstrate the superior performance of DrKGC. Furthermore, a realistic case study in the biomedical domain highlights its interpretability and practical utility.
>
---
#### [replaced 049] Skill Path: Unveiling Language Skills from Circuit Graphs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.01334v3](http://arxiv.org/pdf/2410.01334v3)**

> **作者:** Hang Chen; Jiaying Zhu; Xinyu Yang; Wenya Wang
>
> **备注:** accepted by AAAI 2026 (oral)
>
> **摘要:** Circuit graph discovery has emerged as a fundamental approach to elucidating the skill mechanistic of language models. Despite the output faithfulness of circuit graphs, they suffer from atomic ablation, which causes the loss of causal dependencies between connected components. In addition, their discovery process, designed to preserve output faithfulness, inadvertently captures extraneous effects other than an isolated target skill. To alleviate these challenges, we introduce skill paths, which offers a more refined and compact representation by isolating individual skills within a linear chain of components. To enable skill path extracting from circuit graphs, we propose a three-step framework, consisting of decomposition, pruning, and post-pruning causal mediation. In particular, we offer a complete linear decomposition of the transformer model which leads to a disentangled computation graph. After pruning, we further adopt causal analysis techniques, including counterfactuals and interventions, to extract the final skill paths from the circuit graph. To underscore the significance of skill paths, we investigate three generic language skills-Previous Token Skill, Induction Skill, and In-Context Learning Skill-using our framework. Experiments support two crucial properties of these skills, namely stratification and inclusiveness.
>
---
#### [replaced 050] Dissecting Long-Chain-of-Thought Reasoning Models: An Empirical Study
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04913v2](http://arxiv.org/pdf/2506.04913v2)**

> **作者:** Yongyu Mu; Jiali Zeng; Bei Li; Xinyan Guan; Fandong Meng; Jie Zhou; Tong Xiao; Jingbo Zhu
>
> **备注:** Working in process
>
> **摘要:** Despite recent progress in training long-chain-of-thought reasoning models via scaling reinforcement learning (RL), its underlying training dynamics remain poorly understood, and several counterintuitive behaviors persist. This work focuses on three key aspects: (1) We systematically analyze the roles of positive and negative samples in scaling RL, revealing that positive samples mainly facilitate precise fitting to the training data, whereas negative samples significantly enhance generalization and robustness. Interestingly, while positive samples are essential for convergence in the zero-RL setting, training on negative samples alone suffices to attain strong reasoning performance and even better generalization in cold-start scenarios. (2) We identify substantial data inefficiency in group relative policy optimization, where over half of the samples yield zero advantage. To address this, we explore two strategies, including relative length rewards and offline sample injection, to leverage these data better and enhance reasoning efficiency and capability. (3) We investigate unstable performance across various reasoning models and benchmarks, attributing instability to uncertain problems with ambiguous outcomes, and demonstrate that greedy decoding can distort evaluation by flipping the correctness of responses. Our code is available at: https://github.com/takagi97/Dissect-Long-Reason-Models.
>
---
#### [replaced 051] DP-Fusion: Token-Level Differentially Private Inference for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.04531v3](http://arxiv.org/pdf/2507.04531v3)**

> **作者:** Rushil Thareja; Preslav Nakov; Praneeth Vepakomma; Nils Lukas
>
> **备注:** Our code and data are publicly available here: https://github.com/MBZUAI-Trustworthy-ML/DP-Fusion-DPI
>
> **摘要:** Large language models (LLMs) do not preserve privacy at inference-time. The LLM's outputs can inadvertently reveal information about the model's context, which presents a privacy challenge when the LLM is augmented via tools or databases containing sensitive information. Existing privacy-preserving methods at inference-time have significant limitations since they (i) lack provable guarantees or (ii) have a poor utility/privacy trade-off. We propose DP-Fusion, a Differentially Private Inference (DPI) mechanism for LLMs that provably bounds the influence a set of tokens in the context can have on the LLM's output. DP-Fusion works as follows: (1) label a subset of sensitive tokens, (2) infer the LLM without any sensitive tokens to obtain a baseline, (3) infer the LLM with the sensitive tokens, and (4) blend distributions so that the final output remains within a bounded distance of the baseline distribution. While this per-token influence bound also mitigates jailbreak-style prompt injection, we focus on \emph{document privatization}, where the goal is to paraphrase a document containing sensitive tokens, e.g., personally identifiable information, so that no attacker can reliably infer them from the paraphrased document while preserving high text quality. The privacy/utility trade-off is controlled by $\epsilon$, where $\epsilon=0$ hides sensitive tokens entirely, while higher values trade off privacy for improved text quality. We show that our method creates token-level provably privatized documents with substantially improved theoretical and empirical privacy, achieving $6\times$ lower perplexity than related DPI methods.
>
---
#### [replaced 052] CultureGuard: Towards Culturally-Aware Dataset and Guard Model for Multilingual Safety Applications
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.01710v4](http://arxiv.org/pdf/2508.01710v4)**

> **作者:** Raviraj Joshi; Rakesh Paul; Kanishk Singla; Anusha Kamath; Michael Evans; Katherine Luna; Shaona Ghosh; Utkarsh Vaidya; Eileen Long; Sanjay Singh Chauhan; Niranjan Wartikar
>
> **摘要:** The increasing use of Large Language Models (LLMs) in agentic applications highlights the need for robust safety guard models. While content safety in English is well-studied, non-English languages lack similar advancements due to the high cost of collecting culturally aligned labeled datasets. We present CultureGuard, a novel solution for curating culturally aligned, high-quality safety datasets across multiple languages. Our approach introduces a four-stage synthetic data generation and filtering pipeline: cultural data segregation, cultural data adaptation, machine translation, and quality filtering. This pipeline enables the conversion and expansion of the Nemotron-Content-Safety-Dataset-V2 English safety dataset into eight distinct languages: Arabic, German, Spanish, French, Hindi, Japanese, Thai, and Chinese. The resulting dataset, Nemotron-Safety-Guard-Dataset-v3, comprises 386,661 samples in 9 languages and facilitates the training of Llama-3.1-Nemotron-Safety-Guard-8B-v3 via LoRA-based fine-tuning. The final model achieves state-of-the-art performance on several multilingual content safety benchmarks. Furthermore, we show our moderately multilingual fine-tuning enables robust cross-lingual transfer and strong zero-shot generalization to unseen languages. We also benchmark the latest open LLMs on multilingual safety and observe that these LLMs are more prone to give unsafe responses when prompted in non-English languages. This work advances multilingual LLM safety by enabling the development of culturally aware safety guard models.
>
---
#### [replaced 053] EMNLP: Educator-role Moral and Normative Large Language Models Profiling
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2508.15250v3](http://arxiv.org/pdf/2508.15250v3)**

> **作者:** Yilin Jiang; Mingzi Zhang; Sheng Jin; Zengyi Yu; Xiangjie Kong; Binghao Tu
>
> **备注:** 29pages, 15 figures, Accepted by EMNLP Main Confrence
>
> **摘要:** Simulating Professions (SP) enables Large Language Models (LLMs) to emulate professional roles. However, comprehensive psychological and ethical evaluation in these contexts remains lacking. This paper introduces EMNLP, an Educator-role Moral and Normative LLMs Profiling framework for personality profiling, moral development stage measurement, and ethical risk under soft prompt injection. EMNLP extends existing scales and constructs 88 teacher-specific moral dilemmas, enabling profession-oriented comparison with human teachers. A targeted soft prompt injection set evaluates compliance and vulnerability in teacher SP. Experiments on 14 LLMs show teacher-role LLMs exhibit more idealized and polarized personalities than human teachers, excel in abstract moral reasoning, but struggle with emotionally complex situations. Models with stronger reasoning are more vulnerable to harmful prompt injection, revealing a paradox between capability and safety. The model temperature and other hyperparameters have limited influence except in some risk behaviors. This paper presents the first benchmark to assess ethical and psychological alignment of teacher-role LLMs for educational AI. Resources are available at https://e-m-n-l-p.github.io/.
>
---
#### [replaced 054] JailbreakZoo: Survey, Landscapes, and Horizons in Jailbreaking Large Language and Vision-Language Models
- **分类: cs.CL; cs.CR; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.01599v3](http://arxiv.org/pdf/2407.01599v3)**

> **作者:** Haibo Jin; Leyang Hu; Xinnuo Li; Peiyan Zhang; Chonghan Chen; Jun Zhuang; Haohan Wang
>
> **备注:** 45 pages
>
> **摘要:** The rapid evolution of artificial intelligence (AI) through developments in Large Language Models (LLMs) and Vision-Language Models (VLMs) has brought significant advancements across various technological domains. While these models enhance capabilities in natural language processing and visual interactive tasks, their growing adoption raises critical concerns regarding security and ethical alignment. This survey provides an extensive review of the emerging field of jailbreaking--deliberately circumventing the ethical and operational boundaries of LLMs and VLMs--and the consequent development of defense mechanisms. Our study categorizes jailbreaks into seven distinct types and elaborates on defense strategies that address these vulnerabilities. Through this comprehensive examination, we identify research gaps and propose directions for future studies to enhance the security frameworks of LLMs and VLMs. Our findings underscore the necessity for a unified perspective that integrates both jailbreak strategies and defensive solutions to foster a robust, secure, and reliable environment for the next generation of language models. More details can be found on our website: https://chonghan-chen.com/llm-jailbreak-zoo-survey/.
>
---
#### [replaced 055] Multi-turn Evaluation of Anthropomorphic Behaviours in Large Language Models
- **分类: cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2502.07077v2](http://arxiv.org/pdf/2502.07077v2)**

> **作者:** Lujain Ibrahim; Canfer Akbulut; Rasmi Elasmar; Charvi Rastogi; Minsuk Kahng; Meredith Ringel Morris; Kevin R. McKee; Verena Rieser; Murray Shanahan; Laura Weidinger
>
> **摘要:** The tendency of users to anthropomorphise large language models (LLMs) is of growing interest to AI developers, researchers, and policy-makers. Here, we present a novel method for empirically evaluating anthropomorphic LLM behaviours in realistic and varied settings. Going beyond single-turn static benchmarks, we contribute three methodological advances in state-of-the-art (SOTA) LLM evaluation. First, we develop a multi-turn evaluation of 14 anthropomorphic behaviours. Second, we present a scalable, automated approach by employing simulations of user interactions. Third, we conduct an interactive, large-scale human subject study (N=1101) to validate that the model behaviours we measure predict real users' anthropomorphic perceptions. We find that all SOTA LLMs evaluated exhibit similar behaviours, characterised by relationship-building (e.g., empathy and validation) and first-person pronoun use, and that the majority of behaviours only first occur after multiple turns. Our work lays an empirical foundation for investigating how design choices influence anthropomorphic model behaviours and for progressing the ethical debate on the desirability of these behaviours. It also showcases the necessity of multi-turn evaluations for complex social phenomena in human-AI interaction.
>
---
#### [replaced 056] Compositional Phoneme Approximation for L1-Grounded L2 Pronunciation Training
- **分类: cs.CL; cs.SD; eess.AS; H.5.5**

- **链接: [http://arxiv.org/pdf/2411.10927v5](http://arxiv.org/pdf/2411.10927v5)**

> **作者:** Jisang Park; Minu Kim; DaYoung Hong; Jongha Lee
>
> **备注:** Accepted to IJCNLP-AACL 2025
>
> **摘要:** Learners of a second language (L2) often map non-native phonemes to similar native-language (L1) phonemes, making conventional L2-focused training slow and effortful. To address this, we propose an L1-grounded pronunciation training method based on compositional phoneme approximation (CPA), a feature-based representation technique that approximates L2 sounds with sequences of L1 phonemes. Evaluations with 20 Korean non-native English speakers show that CPA-based training achieves a 76% in-box formant rate in acoustic analysis, 17.6% relative improvement in phoneme recognition accuracy, and over 80% of speech being rated as more native-like, with minimal training. Project page: https://gsanpark.github.io/CPA-Pronunciation.
>
---
#### [replaced 057] LegalEval-Q: A New Benchmark for The Quality Evaluation of LLM-Generated Legal Text
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.24826v2](http://arxiv.org/pdf/2505.24826v2)**

> **作者:** Li yunhan; Wu gengshen
>
> **备注:** 10 pages, 11 figures
>
> **摘要:** As large language models (LLMs) are increasingly used in legal applications, current evaluation benchmarks tend to focus mainly on factual accuracy while largely neglecting important linguistic quality aspects such as clarity, coherence, and terminology. To address this gap, we propose three steps: First, we develop a regression model to evaluate the quality of legal texts based on clarity, coherence, and terminology. Second, we create a specialized set of legal questions. Third, we analyze 49 LLMs using this evaluation framework. Our analysis identifies three key findings: First, model quality levels off at 14 billion parameters, with only a marginal improvement of $2.7\%$ noted at 72 billion parameters. Second, engineering choices such as quantization and context length have a negligible impact, as indicated by statistical significance thresholds above 0.016. Third, reasoning models consistently outperform base architectures. A significant outcome of our research is the release of a ranking list and Pareto analysis, which highlight the Qwen3 series as the optimal choice for cost-performance tradeoffs. This work not only establishes standardized evaluation protocols for legal LLMs but also uncovers fundamental limitations in current training data refinement approaches. Code and models are available at: https://github.com/lyxx3rd/LegalEval-Q.
>
---
#### [replaced 058] Mechanisms vs. Outcomes: Probing for Syntax Fails to Explain Performance on Targeted Syntactic Evaluations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.16678v2](http://arxiv.org/pdf/2506.16678v2)**

> **作者:** Ananth Agarwal; Jasper Jian; Christopher D. Manning; Shikhar Murty
>
> **摘要:** Large Language Models (LLMs) exhibit a robust mastery of syntax when processing and generating text. While this suggests internalized understanding of hierarchical syntax and dependency relations, the precise mechanism by which they represent syntactic structure is an open area within interpretability research. Probing provides one way to identify the mechanism of syntax being linearly encoded in activations, however, no comprehensive study has yet established whether a model's probing accuracy reliably predicts its downstream syntactic performance. Adopting a "mechanisms vs. outcomes" framework, we evaluate 32 open-weight transformer models and find that syntactic features extracted via probing fail to predict outcomes of targeted syntax evaluations across English linguistic phenomena. Our results highlight a substantial disconnect between latent syntactic representations found via probing and observable syntactic behaviors in downstream tasks.
>
---
#### [replaced 059] SEAGraph: Unveiling the Whole Story of Paper Review Comments
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.11939v2](http://arxiv.org/pdf/2412.11939v2)**

> **作者:** Jianxiang Yu; Jiaqi Tan; Zichen Ding; Jiapeng Zhu; Jiahao Li; Yao Cheng; Qier Cui; Yunshi Lan; Yao Liu; Xiang Li
>
> **摘要:** Peer review, as a cornerstone of scientific research, ensures the integrity and quality of scholarly work by providing authors with objective feedback for refinement. However, in the traditional peer review process, authors often receive vague or insufficiently detailed feedback, which provides limited assistance and leads to a more time-consuming review cycle. If authors can identify some specific weaknesses in their paper, they can not only address the reviewer's concerns but also improve their work. This raises the critical question of how to enhance authors' comprehension of review comments. In this paper, we present SEAGraph, a novel framework developed to clarify review comments by uncovering the underlying intentions behind them. We construct two types of graphs for each paper: the semantic mind graph, which captures the authors' thought process, and the hierarchical background graph, which delineates the research domains related to the paper. A retrieval method is then designed to extract relevant content from both graphs, facilitating coherent explanations for the review comments. Extensive experiments show that SEAGraph excels in review comment understanding tasks, offering significant benefits to authors. By bridging the gap between reviewers' critiques and authors' comprehension, SEAGraph contributes to a more efficient, transparent and collaborative scientific publishing ecosystem.
>
---
#### [replaced 060] Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.16795v2](http://arxiv.org/pdf/2507.16795v2)**

> **作者:** Helena Casademunt; Caden Juang; Adam Karvonen; Samuel Marks; Senthooran Rajamanoharan; Neel Nanda
>
> **摘要:** Fine-tuning large language models (LLMs) can lead to unintended out-of-distribution generalization. Standard approaches to this problem rely on modifying training data, for example by adding data that better specify the intended generalization. However, this is not always practical. We introduce Concept Ablation Fine-Tuning (CAFT), a technique that leverages interpretability tools to control how LLMs generalize from fine-tuning, without needing to modify the training data or otherwise use data from the target distribution. Given a set of directions in an LLM's latent space corresponding to undesired concepts, CAFT works by ablating these concepts with linear projections during fine-tuning, steering the model away from unintended generalizations. We successfully apply CAFT to three fine-tuning tasks, including emergent misalignment, a phenomenon where LLMs fine-tuned on a narrow task generalize to give egregiously misaligned responses to general questions. Without any changes to the fine-tuning data, CAFT reduces misaligned responses by 10x without degrading performance on the training distribution. Overall, CAFT represents a novel approach for steering LLM generalization without modifying training data.
>
---
#### [replaced 061] Reasoning with Exploration: An Entropy Perspective
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14758v4](http://arxiv.org/pdf/2506.14758v4)**

> **作者:** Daixuan Cheng; Shaohan Huang; Xuekai Zhu; Bo Dai; Wayne Xin Zhao; Zhenliang Zhang; Furu Wei
>
> **备注:** AAAI 2026 Conference
>
> **摘要:** Balancing exploration and exploitation is a central goal in reinforcement learning (RL). Despite recent advances in enhancing large language model (LLM) reasoning, most methods lean toward exploitation, and increasingly encounter performance plateaus. In this work, we revisit entropy -- a signal of exploration in RL -- and examine its relationship to exploratory reasoning in LLMs. Through empirical analysis, we uncover positive correlations between high-entropy regions and three types of exploratory reasoning actions: (1) pivotal tokens that determine or connect logical steps, (2) reflective actions such as self-verification and correction, and (3) rare behaviors under-explored by the base LLMs. Motivated by this, we introduce a minimal modification to standard RL with only one line of code: augmenting the advantage function with an entropy-based term. Unlike traditional maximum-entropy methods which encourage exploration by promoting uncertainty, we encourage exploration by promoting longer and deeper reasoning chains. Notably, our method achieves significant gains on the Pass@K metric -- an upper-bound estimator of LLM reasoning capabilities -- even when evaluated with extremely large K values, pushing the boundaries of LLM reasoning.
>
---
#### [replaced 062] Distributional Surgery for Language Model Activations
- **分类: cs.LG; cs.CL; math.OC**

- **链接: [http://arxiv.org/pdf/2501.15758v2](http://arxiv.org/pdf/2501.15758v2)**

> **作者:** Bao Nguyen; Binh Nguyen; Duy Nguyen; Viet Anh Nguyen
>
> **备注:** 3 figures
>
> **摘要:** Language models, while capable of generating remarkably coherent and seemingly accurate text, can occasionally produce undesirable content, including harmful or toxic outputs. In this paper, we present a new two-stage approach to detect and mitigate undesirable content generations by rectifying activations. First, we train an ensemble of layerwise classifiers to detect undesirable content using activations by minimizing a smooth surrogate of the risk-aware score. Then, for detected undesirable contents, we propose layerwise distributional steering policies that transform the attention heads. These policies are computed through principled semidefinite programming, which aims to minimally perturb the attention distribution while probabilistically guaranteeing the effectiveness of the editions. Empirical evaluations across multiple language models and datasets show that our method outperforms baselines in reducing the generation of undesirable output.
>
---
#### [replaced 063] When Language Shapes Thought: Cross-Lingual Transfer of Factual Knowledge in Question Answering
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.24409v2](http://arxiv.org/pdf/2505.24409v2)**

> **作者:** Eojin Kang; Juae Kim
>
> **备注:** Accepted at CIKM2025 (Expanded version)
>
> **摘要:** Multilingual large language models (LLMs) offer promising opportunities for cross-lingual information access, yet their use of factual knowledge remains highly sensitive to the input language. Prior work has addressed this through English prompting and evaluation, assuming that English-based reasoning is universally beneficial. In this work, we challenge that assumption by exploring factual knowledge transfer from non-English to English through the lens of Language and Thought Theory. We introduce Language-to-Thought (L2T) prompting, which aligns the model's internal ''thinking'' language with the source of knowledge. Across three languages and four models, L2T consistently outperforms English-based reasoning, reversing the expected advantage of English prompts. Our code is available at https://github.com/GeomeunByeol/Language2Thought.
>
---
#### [replaced 064] Text-to-Pipeline: Bridging Natural Language and Data Preparation Pipelines
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15874v2](http://arxiv.org/pdf/2505.15874v2)**

> **作者:** Yuhang Ge; Yachuan Liu; Zhangyan Ye; Yuren Mao; Yunjun Gao
>
> **摘要:** Data preparation (DP) transforms raw data into a form suitable for downstream applications, typically by composing operations into executable pipelines. Building such pipelines is time-consuming and requires sophisticated programming skills, posing a significant barrier for non-experts. To lower this barrier, we introduce Text-to-Pipeline, a new task that translates NL data preparation instructions into DP pipelines, and PARROT, a large-scale benchmark to support systematic evaluation. To ensure realistic DP scenarios, PARROT is built by mining transformation patterns from production pipelines and instantiating them on 23,009 real-world tables, resulting in ~18,000 tasks spanning 16 core operators. Our empirical evaluation on PARROT reveals a critical failure mode in cutting-edge LLMs: they struggle not only with multi-step compositional logic but also with semantic parameter grounding. We thus establish a strong baseline with Pipeline-Agent, an execution-aware agent that iteratively reflects on intermediate states. While it achieves state-of-the-art performance, a significant gap remains, underscoring the deep, unsolved challenges for PARROT. It provides the essential, large-scale testbed for developing and evaluating the next generation of autonomous data preparation agentic systems.
>
---
#### [replaced 065] Verifiable Fine-Tuning for LLMs: Zero-Knowledge Training Proofs Bound to Data Provenance and Policy
- **分类: cs.CR; cs.CL; 68T07, 94A60, 68Q25; I.2.6; G.1.6; E.3; C.2.4**

- **链接: [http://arxiv.org/pdf/2510.16830v2](http://arxiv.org/pdf/2510.16830v2)**

> **作者:** Hasan Akgul; Daniel Borg; Arta Berisha; Amina Rahimova; Andrej Novak; Mila Petrov
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** Large language models are often adapted through parameter efficient fine tuning, but current release practices provide weak assurances about what data were used and how updates were computed. We present Verifiable Fine Tuning, a protocol and system that produces succinct zero knowledge proofs that a released model was obtained from a public initialization under a declared training program and an auditable dataset commitment. The approach combines five elements. First, commitments that bind data sources, preprocessing, licenses, and per epoch quota counters to a manifest. Second, a verifiable sampler that supports public replayable and private index hiding batch selection. Third, update circuits restricted to parameter efficient fine tuning that enforce AdamW style optimizer semantics and proof friendly approximations with explicit error budgets. Fourth, recursive aggregation that folds per step proofs into per epoch and end to end certificates with millisecond verification. Fifth, provenance binding and optional trusted execution property cards that attest code identity and constants. On English and bilingual instruction mixtures, the method maintains utility within tight budgets while achieving practical proof performance. Policy quotas are enforced with zero violations, and private sampling windows show no measurable index leakage. Federated experiments demonstrate that the system composes with probabilistic audits and bandwidth constraints. These results indicate that end to end verifiable fine tuning is feasible today for real parameter efficient pipelines, closing a critical trust gap for regulated and decentralized deployments.
>
---
#### [replaced 066] Enhancing Large Language Models for Detecting Mental Manipulation via Annotation-Free Data Augmentation and Anti-Curriculum Distillation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15255v3](http://arxiv.org/pdf/2505.15255v3)**

> **作者:** Yuansheng Gao; Han Bao; Tong Zhang; Bin Li; Jixiang Luo; Ronghao Chen; Zonghui Wang; Wenzhi Chen
>
> **摘要:** Mental manipulation is a subtle yet pervasive form of psychological abuse that poses serious threats to mental health. Nevertheless, detecting mental manipulation remains a largely underexplored research problem. The field faces three major challenges: (i) insufficient and hard-to-obtain training data; (ii) the covert nature of mental manipulation, which hinders detection; and (iii) the lack of real-world datasets. To address these challenges, we propose MentalMAC, a novel framework that enhances large language models' ability to detect elements of mental manipulation in multi-turn dialogue. Our approach consists of three key components: EvoSA, an annotation-free data augmentation method based on evolutionary operations and speech act theory; teacher-model-generated multi-task supervision; and progressive task-level anti-curriculum distillation. We then constructed the ReaMent dataset, comprising 5,000 real-world dialogue samples, utilizing MentalMAC-distilled models to aid in human annotation. Vast experiments show that MentalMAC achieves up to 25.9% improvement in F1mac and 8.1% in accuracy over the best-performing baseline, outperforming commercial LLMs such as GPT-4 and Claude-3.5-Sonnet. Warning: This paper contains content that may be offensive to the reader.
>
---
#### [replaced 067] ECom-Bench: Can LLM Agent Resolve Real-World E-commerce Customer Support Issues?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.05639v2](http://arxiv.org/pdf/2507.05639v2)**

> **作者:** Haoxin Wang; Xianhan Peng; Xucheng Huang; Yizhe Huang; Ming Gong; Chenghan Yang; Yang Liu; Ling Jiang
>
> **备注:** Accepted as a main conference paper at EMNLP 2025
>
> **摘要:** In this paper, we introduce ECom-Bench, the first benchmark framework for evaluating LLM agent with multimodal capabilities in the e-commerce customer support domain. ECom-Bench features dynamic user simulation based on persona information collected from real e-commerce customer interactions and a realistic task dataset derived from authentic e-commerce dialogues. These tasks, covering a wide range of business scenarios, are designed to reflect real-world complexities, making ECom-Bench highly challenging. For instance, even advanced models like GPT-4o achieve only a 10-20% pass^3 metric in our benchmark, highlighting the substantial difficulties posed by complex e-commerce scenarios. The code and data have been made publicly available at https://github.com/XiaoduoAILab/ECom-Bench to facilitate further research and development in this domain.
>
---
#### [replaced 068] Meronymic Ontology Extraction via Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.13839v2](http://arxiv.org/pdf/2510.13839v2)**

> **作者:** Dekai Zhang; Simone Conia; Antonio Rago
>
> **备注:** Accepted to AACL 2025
>
> **摘要:** Ontologies have become essential in today's digital age as a way of organising the vast amount of readily available unstructured text. In providing formal structure to this information, ontologies have immense value and application across various domains, e.g., e-commerce, where countless product listings necessitate proper product organisation. However, the manual construction of these ontologies is a time-consuming, expensive and laborious process. In this paper, we harness the recent advancements in large language models (LLMs) to develop a fully-automated method of extracting product ontologies, in the form of meronymies, from raw review texts. We demonstrate that the ontologies produced by our method surpass an existing, BERT-based baseline when evaluating using an LLM-as-a-judge. Our investigation provides the groundwork for LLMs to be used more generally in (product or otherwise) ontology extraction.
>
---
#### [replaced 069] Latent Traits and Cross-Task Transfer: Deconstructing Dataset Interactions in LLM Fine-tuning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.13624v2](http://arxiv.org/pdf/2509.13624v2)**

> **作者:** Shambhavi Krishna; Atharva Naik; Chaitali Agarwal; Sudharshan Govindan; Taesung Lee; Haw-Shiuan Chang
>
> **备注:** Proceedings of the 14th Joint Conference on Lexical and Computational Semantics (*SEM 2025)
>
> **摘要:** Large language models are increasingly deployed across diverse applications. This often includes tasks LLMs have not encountered during training. This implies that enumerating and obtaining the high-quality training data for all tasks is infeasible. Thus, we often need to rely on transfer learning using datasets with different characteristics, and anticipate out-of-distribution requests. Motivated by this practical need, we propose an analysis framework, building a transfer learning matrix and dimensionality reduction, to dissect these cross-task interactions. We train and analyze 10 models to identify latent abilities (e.g., Reasoning, Sentiment Classification, NLU, Arithmetic) and discover the side effects of the transfer learning. Our findings reveal that performance improvements often defy explanations based on surface-level dataset similarity or source data quality. Instead, hidden statistical factors of the source dataset, such as class distribution and generation length proclivities, alongside specific linguistic features, are actually more influential. This work offers insights into the complex dynamics of transfer learning, paving the way for more predictable and effective LLM adaptation.
>
---
#### [replaced 070] Retrieval-Augmented Feature Generation for Domain-Specific Classification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.11177v4](http://arxiv.org/pdf/2406.11177v4)**

> **作者:** Xinhao Zhang; Jinghan Zhang; Fengran Mo; Dakshak Keerthi Chandra; Yu-Zhong Chen; Fei Xie; Kunpeng Liu
>
> **备注:** Accepted by ICDM 2025
>
> **摘要:** Feature generation can significantly enhance learning outcomes, particularly for tasks with limited data. An effective way to improve feature generation is to expand the current feature space using existing features and enriching the informational content. However, generating new, interpretable features usually requires domain-specific knowledge on top of the existing features. In this paper, we introduce a Retrieval-Augmented Feature Generation method, RAFG, to generate useful and explainable features specific to domain classification tasks. To increase the interpretability of the generated features, we conduct knowledge retrieval among the existing features in the domain to identify potential feature associations. These associations are expected to help generate useful features. Moreover, we develop a framework based on large language models (LLMs) for feature generation with reasoning to verify the quality of the features during their generation process. Experiments across several datasets in medical, economic, and geographic domains show that our RAFG method can produce high-quality, meaningful features and significantly improve classification performance compared with baseline methods.
>
---
#### [replaced 071] Understanding Forgetting in LLM Supervised Fine-Tuning and Preference Learning - A Convex Optimization Perspective
- **分类: cs.LG; cs.AI; cs.CL; math.OC; stat.ML**

- **链接: [http://arxiv.org/pdf/2410.15483v4](http://arxiv.org/pdf/2410.15483v4)**

> **作者:** Heshan Fernando; Han Shen; Parikshit Ram; Yi Zhou; Horst Samulowitz; Nathalie Baracaldo; Tianyi Chen
>
> **摘要:** The post-training of LLMs, which typically consists of the supervised fine-tuning (SFT) stage and the preference learning stage (RLHF or DPO), is crucial to effective and safe LLM applications. The widely adopted approach in post-training popular open-source LLMs is to sequentially perform SFT and RLHF/DPO. However, this is suboptimal in terms of SFT and RLHF/DPO trade-off: the LLM gradually forgets about the first stage's training when undergoing the second stage's training. This sequential paradigm persists largely due to its simplicity and modularity, which make it easier to implement and manage at scale despite its limitations. We theoretically prove the sub-optimality of sequential post-training and propose a practical joint post-training framework which has theoretical convergence guarantees and empirically outperforms sequential post-training framework, with up to 23% overall performance improvement across multiple LLM evaluation benchmarks, while having minimal computational overhead. Our code is available at https://github.com/heshandevaka/XRIGHT.
>
---
#### [replaced 072] ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.10419v2](http://arxiv.org/pdf/2508.10419v2)**

> **作者:** Juyuan Wang; Rongchen Zhao; Wei Wei; Yufeng Wang; Mo Yu; Jie Zhou; Jin Xu; Liyan Xu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Narrative comprehension on long stories and novels has been a challenging domain attributed to their intricate plotlines and entangled, often evolving relations among characters and entities. Given the LLM's diminished reasoning over extended context and its high computational cost, retrieval-based approaches remain a pivotal role in practice. However, traditional RAG methods could fall short due to their stateless, single-step retrieval process, which often overlooks the dynamic nature of capturing interconnected relations within long-range context. In this work, we propose ComoRAG, holding the principle that narrative reasoning is not a one-shot process, but a dynamic, evolving interplay between new evidence acquisition and past knowledge consolidation, analogous to human cognition on reasoning with memory-related signals in the brain. Specifically, when encountering a reasoning impasse, ComoRAG undergoes iterative reasoning cycles while interacting with a dynamic memory workspace. In each cycle, it generates probing queries to devise new exploratory paths, then integrates the retrieved evidence of new aspects into a global memory pool, thereby supporting the emergence of a coherent context for the query resolution. Across four challenging long-context narrative benchmarks (200K+ tokens), ComoRAG outperforms strong RAG baselines with consistent relative gains up to 11% compared to the strongest baseline. Further analysis reveals that ComoRAG is particularly advantageous for complex queries requiring global context comprehension, offering a principled, cognitively motivated paradigm towards retrieval-based stateful reasoning. Our framework is made publicly available at https://github.com/EternityJune25/ComoRAG.
>
---
#### [replaced 073] Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.21227v2](http://arxiv.org/pdf/2509.21227v2)**

> **作者:** Seyed Amir Kasaei; Ali Aghayari; Arash Marioriyad; Niki Sepasian; MohammadAmin Fazli; Mahdieh Soleymani Baghshah; Mohammad Hossein Rohban
>
> **备注:** Accepted at GenProCC NeurIPS 2025 Workshop
>
> **摘要:** Text-image generation has advanced rapidly, but assessing whether outputs truly capture the objects, attributes, and relations described in prompts remains a central challenge. Evaluation in this space relies heavily on automated metrics, yet these are often adopted by convention or popularity rather than validated against human judgment. Because evaluation and reported progress in the field depend directly on these metrics, it is critical to understand how well they reflect human preferences. To address this, we present a broad study of widely used metrics for compositional text-image evaluation. Our analysis goes beyond simple correlation, examining their behavior across diverse compositional challenges and comparing how different metric families align with human judgments. The results show that no single metric performs consistently across tasks: performance varies with the type of compositional problem. Notably, VQA-based metrics, though popular, are not uniformly superior, while certain embedding-based metrics prove stronger in specific cases. Image-only metrics, as expected, contribute little to compositional evaluation, as they are designed for perceptual quality rather than alignment. These findings underscore the importance of careful and transparent metric selection, both for trustworthy evaluation and for their use as reward models in generation. Project page is available at https://amirkasaei.com/eval-the-evals/ .
>
---
#### [replaced 074] multiMentalRoBERTa: A Fine-tuned Multiclass Classifier for Mental Health Disorder
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2511.04698v2](http://arxiv.org/pdf/2511.04698v2)**

> **作者:** K M Sajjadul Islam; John Fields; Praveen Madiraju
>
> **备注:** Accepted in IEEE Big Data, 8-11 December, 2025 @ Macau SAR, China
>
> **摘要:** The early detection of mental health disorders from social media text is critical for enabling timely support, risk assessment, and referral to appropriate resources. This work introduces multiMentalRoBERTa, a fine-tuned RoBERTa model designed for multiclass classification of common mental health conditions, including stress, anxiety, depression, post-traumatic stress disorder (PTSD), suicidal ideation, and neutral discourse. Drawing on multiple curated datasets, data exploration is conducted to analyze class overlaps, revealing strong correlations between depression and suicidal ideation as well as anxiety and PTSD, while stress emerges as a broad, overlapping category. Comparative experiments with traditional machine learning methods, domain-specific transformers, and prompting-based large language models demonstrate that multiMentalRoBERTa achieves superior performance, with macro F1-scores of 0.839 in the six-class setup and 0.870 in the five-class setup (excluding stress), outperforming both fine-tuned MentalBERT and baseline classifiers. Beyond predictive accuracy, explainability methods, including Layer Integrated Gradients and KeyBERT, are applied to identify lexical cues that drive classification, with a particular focus on distinguishing depression from suicidal ideation. The findings emphasize the effectiveness of fine-tuned transformers for reliable and interpretable detection in sensitive contexts, while also underscoring the importance of fairness, bias mitigation, and human-in-the-loop safety protocols. Overall, multiMentalRoBERTa is presented as a lightweight, robust, and deployable solution for enhancing support in mental health platforms.
>
---
#### [replaced 075] Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation
- **分类: cs.MA; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.18224v3](http://arxiv.org/pdf/2507.18224v3)**

> **作者:** Shiyuan Li; Yixin Liu; Qingsong Wen; Chengqi Zhang; Shirui Pan
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Multi-agent systems (MAS) based on large language models (LLMs) have emerged as a powerful solution for dealing with complex problems across diverse domains. The effectiveness of MAS is critically dependent on its collaboration topology, which has become a focal point for automated design research. However, existing approaches are fundamentally constrained by their reliance on a template graph modification paradigm with a predefined set of agents and hard-coded interaction structures, significantly limiting their adaptability to task-specific requirements. To address these limitations, we reframe MAS design as a conditional autoregressive graph generation task, where both the system composition and structure are designed jointly. We propose ARG-Designer, a novel autoregressive model that operationalizes this paradigm by constructing the collaboration graph from scratch. Conditioned on a natural language task query, ARG-Designer sequentially and dynamically determines the required number of agents, selects their appropriate roles from an extensible pool, and establishes the optimal communication links between them. This generative approach creates a customized topology in a flexible and extensible manner, precisely tailored to the unique demands of different tasks. Extensive experiments across six diverse benchmarks demonstrate that ARG-Designer not only achieves state-of-the-art performance but also enjoys significantly greater token efficiency and enhanced extensibility. The source code of ARG-Designer is available at https://github.com/Shiy-Li/ARG-Designer.
>
---
#### [replaced 076] KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16002v4](http://arxiv.org/pdf/2502.16002v4)**

> **作者:** Jingbo Yang; Bairu Hou; Wei Wei; Yujia Bao; Shiyu Chang
>
> **摘要:** We describe KVLink, an approach for efficient key-value (KV) cache reuse in large language models (LLMs). In many LLM applications, different inputs can share overlapping context, such as the same retrieved document appearing in multiple queries. However, the LLMs still need to encode the entire context for each query, leading to redundant computation. In this paper, we investigate a new strategy to eliminate such inefficiency, where the KV cache of each document is precomputed independently. During inference, the KV caches of retrieved documents are concatenated, allowing the model to reuse cached representations instead of recomputing them. To mitigate the performance degradation when using KV caches computed independently for each document, KVLink introduces two key techniques: adjusting positional embeddings of the KV cache at inference to match the global position after concatenation, and using trainable special tokens to restore self-attention across independently encoded documents. Experiments across 7 datasets demonstrate that KVLink improves question answering accuracy by an average of 4% over state-of-the-art methods. Furthermore, by leveraging precomputed KV caches, our approach reduces time-to-first-token by up to 96% compared to standard LLM inference, making it a scalable and efficient solution for context reuse. Additionally, KVLink can be combined with KV cache compression to further save cache loading and storage overhead while outperforming the baselines.
>
---
#### [replaced 077] Evaluating Reasoning Faithfulness in Medical Vision-Language Models using Multimodal Perturbations
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.11196v2](http://arxiv.org/pdf/2510.11196v2)**

> **作者:** Johannes Moll; Markus Graf; Tristan Lemke; Nicolas Lenhart; Daniel Truhn; Jean-Benoit Delbrouck; Jiazhen Pan; Daniel Rueckert; Lisa C. Adams; Keno K. Bressem
>
> **备注:** Accepted to ML4H 2025 Proceedings
>
> **摘要:** Vision-language models (VLMs) often produce chain-of-thought (CoT) explanations that sound plausible yet fail to reflect the underlying decision process, undermining trust in high-stakes clinical use. Existing evaluations rarely catch this misalignment, prioritizing answer accuracy or adherence to formats. We present a clinically grounded framework for chest X-ray visual question answering (VQA) that probes CoT faithfulness via controlled text and image modifications across three axes: clinical fidelity, causal attribution, and confidence calibration. In a reader study (n=4), evaluator-radiologist correlations fall within the observed inter-radiologist range for all axes, with strong alignment for attribution (Kendall's $\tau_b=0.670$), moderate alignment for fidelity ($\tau_b=0.387$), and weak alignment for confidence tone ($\tau_b=0.091$), which we report with caution. Benchmarking six VLMs shows that answer accuracy and explanation quality can be decoupled, acknowledging injected cues does not ensure grounding, and text cues shift explanations more than visual cues. While some open-source models match final answer accuracy, proprietary models score higher on attribution (25.0% vs. 1.4%) and often on fidelity (36.1% vs. 31.7%), highlighting deployment risks and the need to evaluate beyond final answer accuracy.
>
---
#### [replaced 078] EditGRPO: Reinforcement Learning with Post-Rollout Edits for Clinically Accurate Chest X-Ray Report Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.22812v2](http://arxiv.org/pdf/2509.22812v2)**

> **作者:** Kai Zhang; Christopher Malon; Lichao Sun; Martin Renqiang Min
>
> **备注:** AACL 2025
>
> **摘要:** Radiology report generation requires advanced medical image analysis, effective temporal reasoning, and accurate text generation. Although recent innovations, particularly multimodal large language models, have shown improved performance, their supervised fine-tuning (SFT) objective is not explicitly aligned with clinical efficacy. In this work, we introduce EditGRPO, a mixed-policy reinforcement learning algorithm designed specifically to optimize the generation through clinically motivated rewards. EditGRPO integrates on-policy exploration with off-policy guidance by injecting sentence-level detailed corrections during training rollouts. This mixed-policy approach addresses the exploration dilemma and sampling efficiency issues typically encountered in RL. Applied to a Qwen2.5-VL-3B, EditGRPO outperforms both SFT and vanilla GRPO baselines, achieving an average improvement of 3.4\% in clinical metrics across four major datasets. Notably, EditGRPO also demonstrates superior out-of-domain generalization, with an average performance gain of 5.9\% on unseen datasets.
>
---
#### [replaced 079] PCS: Perceived Confidence Scoring of Black Box LLMs with Metamorphic Relations
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.07186v2](http://arxiv.org/pdf/2502.07186v2)**

> **作者:** Sina Salimian; Gias Uddin; Shaina Raza; Henry Leung
>
> **摘要:** Zero-shot LLMs are now also used for textual classification tasks, e.g., sentiment and bias detection in a sentence or article. However, their performance can be suboptimal in such data annotation tasks. We introduce a novel technique that evaluates an LLM's confidence for classifying a textual input by leveraging Metamorphic Relations (MRs). The MRs generate semantically equivalent yet textually divergent versions of the input. Following the principles of Metamorphic Testing (MT), the mutated versions are expected to have annotation labels similar to the input. By analyzing the consistency of an LLM's responses across these variations, we compute a perceived confidence score (PCS) based on the frequency of the predicted labels. PCS can be used for both single and multiple LLM settings (e.g., when multiple LLMs are vetted in a majority-voting setup). Empirical evaluation shows that our PCS-based approach improves the performance of zero-shot LLMs by 9.3% in textual classification tasks. When multiple LLMs are used in a majority-voting setup, we obtain a performance boost of 5.8% with PCS.
>
---
#### [replaced 080] BEE-RAG: Balanced Entropy Engineering for Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.05100v2](http://arxiv.org/pdf/2508.05100v2)**

> **作者:** Yuhao Wang; Ruiyang Ren; Yucheng Wang; Jing Liu; Wayne Xin Zhao; Hua Wu; Haifeng Wang
>
> **摘要:** With the rapid advancement of large language models (LLMs), retrieval-augmented generation (RAG) has emerged as a critical approach to supplement the inherent knowledge limitations of LLMs. However, due to the typically large volume of retrieved information, RAG tends to operate with long context lengths. From the perspective of entropy engineering, we identify unconstrained entropy growth and attention dilution due to long retrieval context as significant factors affecting RAG performance. In this paper, we propose the balanced entropy-engineered RAG (BEE-RAG) framework, which improves the adaptability of RAG systems to varying context lengths through the principle of entropy invariance. By leveraging balanced context entropy to reformulate attention dynamics, BEE-RAG separates attention sensitivity from context length, ensuring a stable entropy level. Building upon this, we introduce a zero-shot inference strategy for multi-importance estimation and a parameter-efficient adaptive fine-tuning mechanism to obtain the optimal balancing factor for different settings. Extensive experiments across multiple RAG tasks demonstrate the effectiveness of BEE-RAG.
>
---
#### [replaced 081] Shared Heritage, Distinct Writing: Rethinking Resource Selection for East Asian Historical Documents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.04822v2](http://arxiv.org/pdf/2411.04822v2)**

> **作者:** Seyoung Song; Haneul Yoo; Jiho Jin; Kyunghyun Cho; Alice Oh
>
> **备注:** IJCNLP-AACL 2025 Findings
>
> **摘要:** Historical documents in the Sinosphere are known to share common formats and practices, particularly in veritable records compiled by court historians. This shared linguistic heritage has led researchers to use Classical Chinese resources for cross-lingual transfer when processing historical documents from Korea and Japan, which remain relatively low-resource. In this paper, we question the assumption of cross-lingual transferability from Classical Chinese to Hanja and Kanbun, the ancient written languages of Korea and Japan, respectively. Our experiments across machine translation, named entity recognition, and punctuation restoration tasks show minimal impact of Classical Chinese datasets on language model performance for ancient Korean documents written in Hanja, with performance differences within $\pm{}0.0068$ F1-score for sequence labeling tasks and up to $+0.84$ BLEU score for translation. These limitations persist consistently across various model sizes, architectures, and domain-specific datasets. Our analysis reveals that the benefits of Classical Chinese resources diminish rapidly as local language data increases for Hanja, while showing substantial improvements only in extremely low-resource scenarios for both Korean and Japanese historical documents. These findings emphasize the need for careful empirical validation rather than assuming benefits from indiscriminate cross-lingual transfer.
>
---
#### [replaced 082] OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.12618v2](http://arxiv.org/pdf/2506.12618v2)**

> **作者:** Vineeth Dorna; Anmol Mekala; Wenlong Zhao; Andrew McCallum; Zachary C. Lipton; J. Zico Kolter; Pratyush Maini
>
> **摘要:** Robust unlearning is crucial for safely deploying large language models (LLMs) in environments where data privacy, model safety, and regulatory compliance must be ensured. Yet the task is inherently challenging, partly due to difficulties in reliably measuring whether unlearning has truly occurred. Moreover, fragmentation in current methodologies and inconsistent evaluation metrics hinder comparative analysis and reproducibility. To unify and accelerate research efforts, we introduce OpenUnlearning, a standardized and extensible framework designed explicitly for benchmarking both LLM unlearning methods and metrics. OpenUnlearning integrates 13 unlearning algorithms and 16 diverse evaluations across 3 leading benchmarks (TOFU, MUSE, and WMDP) and also enables analyses of forgetting behaviors across 450+ checkpoints we publicly release. Leveraging OpenUnlearning, we propose a novel meta-evaluation benchmark focused specifically on assessing the faithfulness and robustness of evaluation metrics themselves. We also benchmark diverse unlearning methods and provide a comparative analysis against an extensive evaluation suite. Overall, we establish a clear, community-driven pathway toward rigorous development in LLM unlearning research.
>
---
#### [replaced 083] Reasoning Planning for Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2511.00521v2](http://arxiv.org/pdf/2511.00521v2)**

> **作者:** Bao Nguyen; Hieu Trung Nguyen; Ruifeng She; Xiaojin Fu; Viet Anh Nguyen
>
> **备注:** 27 pages, 5 figures
>
> **摘要:** Selecting an appropriate reasoning method for a given query remains a key challenge in language model generation. Existing approaches typically generate multiple candidate responses and use an aggregation strategy to select the output answer, often assuming that more candidate answers yield higher accuracy. We revisit this assumption through a rigorous theoretical analysis, deriving accuracy bounds for standard aggregation methods under fixed generation distributions and candidate sizes. Building on these insights, we introduce EPIC, an Ensemble Planning with Contrastive learning framework to learn a shared representation space that captures both model reasoning abilities and query-method compatibility. EPIC incorporates our probability bounds as a regularizer in a utility-driven optimization that balances accuracy and computational cost. Experiments on diverse mathematical reasoning tasks show that EPIC consistently selects optimal reasoning methods, improving accuracy while reducing computational overhead. Our code can be found at https://github.com/nguyenngocbaocmt02/EPIC.
>
---
#### [replaced 084] Mufu: Multilingual Fused Learning for Low-Resource Translation with LLM
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.13949v3](http://arxiv.org/pdf/2409.13949v3)**

> **作者:** Zheng Wei Lim; Nitish Gupta; Honglin Yu; Trevor Cohn
>
> **备注:** 29 pages
>
> **摘要:** Multilingual large language models (LLMs) are great translators, but this is largely limited to high-resource languages. For many LLMs, translating in and out of low-resource languages remains a challenging task. To maximize data efficiency in this low-resource setting, we introduce Mufu, which includes a selection of automatically generated multilingual candidates and an instruction to correct inaccurate translations in the prompt. Mufu prompts turn a translation task into a postediting one, and seek to harness the LLM's reasoning capability with auxiliary translation candidates, from which the model is required to assess the input quality, align the semantics cross-lingually, copy from relevant inputs and override instances that are incorrect. Our experiments on En-XX translations over the Flores-200 dataset show LLMs finetuned against Mufu-style prompts are robust to poor quality auxiliary translation candidates, achieving performance superior to NLLB 1.3B distilled model in 64% of low- and very-low-resource language pairs. We then distill these models to reduce inference cost, while maintaining on average 3.1 chrF improvement over finetune-only baseline in low-resource translations.
>
---
#### [replaced 085] EMBRACE: Shaping Inclusive Opinion Representation by Aligning Implicit Conversations with Social Norms
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20264v2](http://arxiv.org/pdf/2507.20264v2)**

> **作者:** Abeer Aldayel; Areej Alokaili
>
> **备注:** Accepted, to appear IJCNLP-AACL 2025 Findings
>
> **摘要:** Shaping inclusive representations that embrace diversity and ensure fair participation and reflections of values is at the core of many conversation-based models. However, many existing methods rely on surface inclusion using mention of user demographics or behavioral attributes of social groups. Such methods overlook the nuanced, implicit expression of opinion embedded in conversations. Furthermore, the over-reliance on overt cues can exacerbate misalignment and reinforce harmful or stereotypical representations in model outputs. Thus, we took a step back and recognized that equitable inclusion needs to account for the implicit expression of opinion and use the stance of responses to validate the normative alignment. This study aims to evaluate how opinions are represented in NLP or computational models by introducing an alignment evaluation framework that foregrounds implicit, often overlooked conversations and evaluates the normative social views and discourse. Our approach models the stance of responses as a proxy for the underlying opinion, enabling a considerate and reflective representation of diverse social viewpoints. We evaluate the framework using both (i) positive-unlabeled (PU) online learning with base classifiers, and (ii) instruction-tuned language models to assess post-training alignment. Through this, we provide a principled and structured lens on how implicit opinions are (mis)represented and offer a pathway toward more inclusive model behavior.
>
---
#### [replaced 086] On the Consistency of Multilingual Context Utilization in Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.00597v4](http://arxiv.org/pdf/2504.00597v4)**

> **作者:** Jirui Qi; Raquel Fernández; Arianna Bisazza
>
> **备注:** Best Paper Award at MRL Workshop 2025, colocated with EMNLP 2025. All codes and data are released at https://github.com/Betswish/mRAG-Context-Consistency
>
> **摘要:** Retrieval-augmented generation (RAG) with large language models (LLMs) has demonstrated strong performance in multilingual question-answering (QA) tasks by leveraging relevant passages retrieved from corpora. In multilingual RAG (mRAG), the retrieved passages can be written in languages other than that of the query entered by the user, making it challenging for LLMs to effectively utilize the provided information. Recent research suggests that retrieving passages from multilingual corpora can improve RAG performance, particularly for low-resource languages. However, the extent to which LLMs can leverage different kinds of multilingual contexts to generate accurate answers, *independently from retrieval quality*, remains understudied. In this paper, we conduct an extensive assessment of LLMs' ability to (i) make consistent use of a relevant passage regardless of its language, (ii) respond in the expected language, and (iii) focus on the relevant passage even when multiple `distracting' passages in different languages are provided in the context. Our experiments with four LLMs across three QA datasets covering a total of 48 languages reveal a surprising ability of LLMs to extract the relevant information from passages in a different language than the query, but a much weaker ability to formulate a full answer in the correct language. Our analysis, based on both accuracy and feature attribution techniques, further shows that distracting passages negatively impact answer quality regardless of their language. However, distractors in the query language exert a slightly stronger influence. Taken together, our findings deepen the understanding of how LLMs utilize context in mRAG systems, providing directions for future improvements.
>
---
#### [replaced 087] SinLlama -- A Large Language Model for Sinhala
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.09115v4](http://arxiv.org/pdf/2508.09115v4)**

> **作者:** H. W. K. Aravinda; Rashad Sirajudeen; Samith Karunathilake; Nisansa de Silva; Surangika Ranathunga; Rishemjit Kaur
>
> **摘要:** Low-resource languages such as Sinhala are often overlooked by open-source Large Language Models (LLMs). In this research, we extend an existing multilingual LLM (Llama-3-8B) to better serve Sinhala. We enhance the LLM tokenizer with Sinhala specific vocabulary and perform continual pre-training on a cleaned 10 million Sinhala corpus, resulting in the SinLlama model. This is the very first decoder-based open-source LLM with explicit Sinhala support. When SinLlama was instruction fine-tuned for three text classification tasks, it outperformed base and instruct variants of Llama-3-8B by a significant margin.
>
---
#### [replaced 088] PPC-GPT: Federated Task-Specific Compression of Large Language Models via Pruning and Chain-of-Thought Distillation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.15857v2](http://arxiv.org/pdf/2502.15857v2)**

> **作者:** Tao Fan; Guoqiang Ma; Yuanfeng Song; Lixin Fan; Qiang Yang
>
> **摘要:** Compressing Large Language Models (LLMs) into task-specific Small Language Models (SLMs) encounters two significant challenges: safeguarding domain-specific knowledge privacy and managing limited resources. To tackle these challenges, we propose PPC-GPT, a novel unified framework that systematically addresses both privacy preservation and model compression in federated settings. PPC-GPT works on a server-client federated architecture, where the client sends differentially private (DP) perturbed task-specific data to the server's LLM. The LLM then generates synthetic data along with their corresponding rationales. This synthetic data is subsequently used for both LLM pruning and retraining processes. Our framework's key innovation lies in its holistic integration of privacy-preserving mechanisms, synthetic data generation, and task-specific compression techniques, creating unique benefits through component interaction. Our experiments across diverse text generation tasks demonstrate that PPC-GPT successfully achieves dual objectives: maintaining competitive performance comparable to full-sized LLMs while ensuring robust privacy protection through its federated architecture. Our code has been contributed to the FATE open-source project and is now publicly accessible at \textit{https://github.com/FederatedAI/FATE-LLM/tree/main/python/fate_llm/algo/ppc-gpt}
>
---
#### [replaced 089] Order Doesn't Matter, But Reasoning Does: Training LLMs with Order-Centric Augmentation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.19907v2](http://arxiv.org/pdf/2502.19907v2)**

> **作者:** Qianxi He; Qianyu He; Jiaqing Liang; Yanghua Xiao; Weikang Zhou; Zeye Sun; Fei Yu
>
> **摘要:** Logical reasoning is essential for large language models (LLMs) to ensure accurate and coherent inference. However, LLMs struggle with reasoning order variations and fail to generalize across logically equivalent transformations. LLMs often rely on fixed sequential patterns rather than true logical understanding. To address this issue, we introduce an order-centric data augmentation framework based on commutativity in logical reasoning. We first randomly shuffle independent premises to introduce condition order augmentation. For reasoning steps, we construct a directed acyclic graph (DAG) to model dependencies between steps, which allows us to identify valid reorderings of steps while preserving logical correctness. By leveraging order-centric augmentations, models can develop a more flexible and generalized reasoning process. Finally, we conduct extensive experiments across multiple logical reasoning benchmarks, demonstrating that our method significantly enhances LLMs' reasoning performance and adaptability to diverse logical structures. We release our codes and augmented data in https://github.com/qianxiHe147/Order-Centric-Data-Augmentation.
>
---
#### [replaced 090] Evaluating Human-LLM Representation Alignment: A Case Study on Affective Sentence Generation for Augmentative and Alternative Communication
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.11881v3](http://arxiv.org/pdf/2503.11881v3)**

> **作者:** Shadab Choudhury; Asha Kumar; Lara J. Martin
>
> **备注:** Published at IJCNLP-AACL 2025 Findings
>
> **摘要:** Gaps arise between a language model's use of concepts and people's expectations. This gap is critical when LLMs generate text to help people communicate via Augmentative and Alternative Communication (AAC) tools. In this work, we introduce the evaluation task of Representation Alignment for measuring this gap via human judgment. In our study, we expand keywords and emotion representations into full sentences. We select four emotion representations: Words, Valence-Arousal-Dominance (VAD) dimensions expressed in both Lexical and Numeric forms, and Emojis. In addition to Representation Alignment, we also measure people's judgments of the accuracy and realism of the generated sentences. While representations like VAD break emotions into easy-to-compute components, our findings show that people agree more with how LLMs generate when conditioned on English words (e.g., "angry") rather than VAD scales. This difference is especially visible when comparing Numeric VAD to words. Furthermore, we found that the perception of how much a generated sentence conveys an emotion is dependent on both the representation type and which emotion it is.
>
---
#### [replaced 091] LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.10114v4](http://arxiv.org/pdf/2510.10114v4)**

> **作者:** Luyao Zhuang; Shengyuan Chen; Yilin Xiao; Huachi Zhou; Yujing Zhang; Hao Chen; Qinggang Zhang; Xiao Huang
>
> **摘要:** Retrieval-Augmented Generation (RAG) is widely used to mitigate hallucinations of Large Language Models (LLMs) by leveraging external knowledge. While effective for simple queries, traditional RAG systems struggle with large-scale, unstructured corpora where information is fragmented. Recent advances incorporate knowledge graphs to capture relational structures, enabling more comprehensive retrieval for complex, multi-hop reasoning tasks. However, existing graph-based RAG (GraphRAG) methods rely on unstable and costly relation extraction for graph construction, often producing noisy graphs with incorrect or inconsistent relations that degrade retrieval quality. In this paper, we revisit the pipeline of existing GraphRAG systems and propose LinearRAG (Linear Graph-based Retrieval-Augmented Generation), an efficient framework that enables reliable graph construction and precise passage retrieval. Specifically, LinearRAG constructs a relation-free hierarchical graph, termed Tri-Graph, using only lightweight entity extraction and semantic linking, avoiding unstable relation modeling. This new paradigm of graph construction scales linearly with corpus size and incurs no extra token consumption, providing an economical and reliable indexing of the original passages. For retrieval, LinearRAG adopts a two-stage strategy: (i) relevant entity activation via local semantic bridging, followed by (ii) passage retrieval through global importance aggregation. Extensive experiments on four datasets demonstrate that LinearRAG significantly outperforms baseline models. Our code and datasets are available at https://github.com/DEEP-PolyU/LinearRAG.
>
---
#### [replaced 092] Rethinking Creativity Evaluation: A Critical Analysis of Existing Creativity Evaluations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.05470v2](http://arxiv.org/pdf/2508.05470v2)**

> **作者:** Li-Chun Lu; Miri Liu; Pin-Chun Lu; Yufei Tian; Shao-Hua Sun; Nanyun Peng
>
> **备注:** 23 pages, 6 figures
>
> **摘要:** We systematically examine, analyze, and compare representative creativity measures--creativity index, perplexity, syntactic templates, and LLM-as-a-Judge--across diverse creative domains, including creative writing, unconventional problem-solving, and research ideation. Our analyses reveal that these metrics exhibit limited consistency, capturing different dimensions of creativity. We highlight key limitations, including the creativity index's focus on lexical diversity, perplexity's sensitivity to model confidence, and syntactic templates' inability to capture conceptual creativity. Additionally, LLM-as-a-Judge shows instability and bias. Our findings underscore the need for more robust, generalizable evaluation frameworks that better align with human judgments of creativity.
>
---
#### [replaced 093] FedCoT: Federated Chain-of-Thought Distillation for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.12403v2](http://arxiv.org/pdf/2406.12403v2)**

> **作者:** Tao Fan; Weijing Chen; Yan Kang; Guoqiang Ma; Hanlin Gu; Yuanfeng Song; Lixin Fan; Qiang Yang
>
> **摘要:** Large Language Models (LLMs) have emerged as a transformative force in artificial intelligence, demonstrating exceptional proficiency across various tasks. However, their deployment in resource-constrained environments and concerns over user data privacy pose significant challenges. In contrast, Small Language Models (SLMs) offer computational efficiency but often lag in performance. To address these issues, we propose FedCoT, a federated framework designed for the Chain-of-Thought (CoT) distillation of knowledge from LLMs to SLMs, while ensuring the preservation of clients' data privacy. FedCoT ensures secure and efficient knowledge transfer from an LLM on a high-powered server to an SLM on a resource-constrained client, while adhering to privacy requirements. Leveraging perturbed prompts and rationales generated through the CoT approach, the framework enhances the performance of the client's SLM without compromising user data privacy within a multi-task learning framework. We propose two privacy protection strategies: the Exponential Mechanism Strategy and the Adaptive Exponential Mechanism Strategy, which balance user prompt privacy and the usability of rationales. Empirical evaluation on various text generation tasks demonstrates the effectiveness of FedCoT in training task-specific SLMs with enhanced performance while prioritizing data privacy protection. Our code has been contributed to the FATE open-source project and is now publicly accessible at \textit{https://github.com/FederatedAI/FATE-LLM/tree/main/python/fate_llm/algo/fedcot}
>
---
#### [replaced 094] The Landscape of Agentic Reinforcement Learning for LLMs: A Survey
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.02547v3](http://arxiv.org/pdf/2509.02547v3)**

> **作者:** Guibin Zhang; Hejia Geng; Xiaohang Yu; Zhenfei Yin; Zaibin Zhang; Zelin Tan; Heng Zhou; Zhongzhi Li; Xiangyuan Xue; Yijiang Li; Yifan Zhou; Yang Chen; Chen Zhang; Yutao Fan; Zihu Wang; Songtao Huang; Francisco Piedrahita-Velez; Yue Liao; Hongru Wang; Mengyue Yang; Heng Ji; Jun Wang; Shuicheng Yan; Philip Torr; Lei Bai
>
> **摘要:** The emergence of agentic reinforcement learning (Agentic RL) marks a paradigm shift from conventional reinforcement learning applied to large language models (LLM RL), reframing LLMs from passive sequence generators into autonomous, decision-making agents embedded in complex, dynamic worlds. This survey formalizes this conceptual shift by contrasting the degenerate single-step Markov Decision Processes (MDPs) of LLM-RL with the temporally extended, partially observable Markov decision processes (POMDPs) that define Agentic RL. Building on this foundation, we propose a comprehensive twofold taxonomy: one organized around core agentic capabilities, including planning, tool use, memory, reasoning, self-improvement, and perception, and the other around their applications across diverse task domains. Central to our thesis is that reinforcement learning serves as the critical mechanism for transforming these capabilities from static, heuristic modules into adaptive, robust agentic behavior. To support and accelerate future research, we consolidate the landscape of open-source environments, benchmarks, and frameworks into a practical compendium. By synthesizing over five hundred recent works, this survey charts the contours of this rapidly evolving field and highlights the opportunities and challenges that will shape the development of scalable, general-purpose AI agents.
>
---
#### [replaced 095] Normality and the Turing Test
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.21382v2](http://arxiv.org/pdf/2508.21382v2)**

> **作者:** Alexandre Kabbach
>
> **摘要:** This paper proposes to revisit the Turing test through the concept of normality. Its core argument is that the Turing test is a test of normal intelligence as assessed by a normal judge. First, in the sense that the Turing test targets normal/average rather than exceptional human intelligence, so that successfully passing the test requires machines to "make mistakes" and display imperfect behavior just like normal/average humans. Second, in the sense that the Turing test is a statistical test where judgments of intelligence are never carried out by a single "average" judge (understood as non-expert) but always by a full jury. As such, the notion of "average human interrogator" that Turing talks about in his original paper should be understood primarily as referring to a mathematical abstraction made of the normalized aggregate of individual judgments of multiple judges. Its conclusions are twofold. First, it argues that large language models such as ChatGPT are unlikely to pass the Turing test as those models precisely target exceptional rather than normal/average human intelligence. As such, they constitute models of what it proposes to call artificial smartness rather than artificial intelligence, insofar as they deviate from the original goal of Turing for the modeling of artificial minds. Second, it argues that the objectivization of normal human behavior in the Turing test fails due to the game configuration of the test which ends up objectivizing normative ideals of normal behavior rather than normal behavior per se.
>
---
#### [replaced 096] BLADE: Benchmarking Language Model Agents for Data-Driven Science
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.09667v3](http://arxiv.org/pdf/2408.09667v3)**

> **作者:** Ken Gu; Ruoxi Shang; Ruien Jiang; Keying Kuang; Richard-John Lin; Donghe Lyu; Yue Mao; Youran Pan; Teng Wu; Jiaqian Yu; Yikun Zhang; Tianmai M. Zhang; Lanyi Zhu; Mike A. Merrill; Jeffrey Heer; Tim Althoff
>
> **备注:** EMNLP 2024
>
> **摘要:** Data-driven scientific discovery requires the iterative integration of scientific domain knowledge, statistical expertise, and an understanding of data semantics to make nuanced analytical decisions, e.g., about which variables, transformations, and statistical models to consider. LM-based agents equipped with planning, memory, and code execution capabilities have the potential to support data-driven science. However, evaluating agents on such open-ended tasks is challenging due to multiple valid approaches, partially correct steps, and different ways to express the same decisions. To address these challenges, we present BLADE, a benchmark to automatically evaluate agents' multifaceted approaches to open-ended research questions. BLADE consists of 12 datasets and research questions drawn from existing scientific literature, with ground truth collected from independent analyses by expert data scientists and researchers. To automatically evaluate agent responses, we developed corresponding computational methods to match different representations of analyses to this ground truth. Though language models possess considerable world knowledge, our evaluation shows that they are often limited to basic analyses. However, agents capable of interacting with the underlying data demonstrate improved, but still non-optimal, diversity in their analytical decision making. Our work enables the evaluation of agents for data-driven science and provides researchers deeper insights into agents' analysis approaches.
>
---
#### [replaced 097] Auto-PRE: An Automatic and Cost-Efficient Peer-Review Framework for Language Generation Evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12265v2](http://arxiv.org/pdf/2410.12265v2)**

> **作者:** Junjie Chen; Weihang Su; Zhumin Chu; Haitao Li; Yujia Zhou; Dingbo Yuan; Xudong Wang; Jun Zhou; Yiqun Liu; Min Zhang; Shaoping Ma; Qingyao Ai
>
> **备注:** AAAI 2026
>
> **摘要:** The rapid development of large language models (LLMs) has highlighted the need for efficient and reliable methods to evaluate their performance. Traditional evaluation methods often face challenges like high costs, limited task formats, dependence on human references, and systematic biases. To address these limitations, we propose Auto-PRE, an automatic LLM evaluation framework inspired by the peer review process. Unlike previous approaches that rely on human annotations, Auto-PRE automatically selects evaluator LLMs based on three core traits: consistency, pertinence, and self-confidence, which correspond to the instruction, content, and response stages, respectively, and collectively cover the entire evaluation process. Experiments on three representative tasks, including summarization, non-factoid QA, and dialogue generation, demonstrate that Auto-PRE achieves state-of-the-art performance while significantly reducing evaluation costs. Furthermore, the structured and scalable design of our automatic qualification exam framework provides valuable insights into automating the evaluation of LLMs-as-judges, paving the way for more advanced LLM-based evaluation frameworks.
>
---
#### [replaced 098] OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting during Parameter-Efficient Fine-Tuning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.13003v2](http://arxiv.org/pdf/2510.13003v2)**

> **作者:** Yifeng Xiong; Xiaohui Xie
>
> **摘要:** Low-Rank Adaptation (LoRA) enables efficient fine-tuning of large language models but suffers from catastrophic forgetting when learned updates interfere with the dominant singular directions that encode essential pre-trained knowledge. We propose Orthogonal Projection LoRA (OPLoRA), a theoretically grounded approach that prevents this interference through double-sided orthogonal projections. By decomposing frozen weights via SVD, OPLoRA constrains LoRA updates to lie entirely within the orthogonal complement of the top-$k$ singular subspace using projections $P_L = I - U_k U_k^\top$ and $P_R = I - V_k V_k^\top$. We prove that this construction exactly preserves the top-$k$ singular triples, providing mathematical guarantees for knowledge retention. To quantify subspace interference, we introduce $\rho_k$, a metric measuring update alignment with dominant directions. Extensive experiments across commonsense reasoning, mathematics, and code generation demonstrate that OPLoRA significantly reduces forgetting while maintaining competitive task-specific performance on LLaMA-2 7B and Qwen2.5 7B, establishing orthogonal projection as an effective mechanism for knowledge preservation in parameter-efficient fine-tuning.
>
---
#### [replaced 099] Quriosity: Analyzing Human Questioning Behavior and Causal Inquiry through Curiosity-Driven Queries
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2405.20318v4](http://arxiv.org/pdf/2405.20318v4)**

> **作者:** Roberto Ceraolo; Dmitrii Kharlapenko; Ahmad Khan; Amélie Reymond; Punya Syon Pandey; Rada Mihalcea; Bernhard Schölkopf; Mrinmaya Sachan; Zhijing Jin
>
> **备注:** IJCNLP-AACL 2025 Findings
>
> **摘要:** Recent progress in Large Language Model (LLM) technology has changed our role in interacting with these models. Instead of primarily testing these models with questions we already know answers to, we are now using them for queries where the answers are unknown to us, driven by human curiosity. This shift highlights the growing need to understand curiosity-driven human questions - those that are more complex, open-ended, and reflective of real-world needs. To this end, we present Quriosity, a collection of 13.5K naturally occurring questions from three diverse sources: human-to-search-engine queries, human-to-human interactions, and human-to-LLM conversations. Our comprehensive collection enables a rich understanding of human curiosity across various domains and contexts. Our analysis reveals a significant presence of causal questions (up to 42%) in the dataset, for which we develop an iterative prompt improvement framework to identify all causal queries and examine their unique linguistic properties, cognitive complexity and source distribution. Our paper paves the way for future work on causal question identification and open-ended chatbot interactions. Our code and data are at https://github.com/roberto-ceraolo/quriosity.
>
---
#### [replaced 100] MultiMed-ST: Large-scale Many-to-many Multilingual Medical Speech Translation
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.03546v3](http://arxiv.org/pdf/2504.03546v3)**

> **作者:** Khai Le-Duc; Tuyen Tran; Bach Phan Tat; Nguyen Kim Hai Bui; Quan Dang; Hung-Phong Tran; Thanh-Thuy Nguyen; Ly Nguyen; Tuan-Minh Phan; Thi Thu Phuong Tran; Chris Ngo; Nguyen X. Khanh; Thanh Nguyen-Tang
>
> **备注:** EMNLP 2025
>
> **摘要:** Multilingual speech translation (ST) and machine translation (MT) in the medical domain enhances patient care by enabling efficient communication across language barriers, alleviating specialized workforce shortages, and facilitating improved diagnosis and treatment, particularly during pandemics. In this work, we present the first systematic study on medical ST, to our best knowledge, by releasing MultiMed-ST, a large-scale ST dataset for the medical domain, spanning all translation directions in five languages: Vietnamese, English, German, French, and Simplified/Traditional Chinese, together with the models. With 290,000 samples, this is the largest medical MT dataset and the largest many-to-many multilingual ST among all domains. Secondly, we present the most comprehensive ST analysis in the field's history, to our best knowledge, including: empirical baselines, bilingual-multilingual comparative study, end-to-end vs. cascaded comparative study, task-specific vs. multi-task sequence-to-sequence comparative study, code-switch analysis, and quantitative-qualitative error analysis. All code, data, and models are available online: https://github.com/leduckhai/MultiMed-ST
>
---
#### [replaced 101] LLM Teacher-Student Framework for Text Classification With No Manually Annotated Data: A Case Study in IPTC News Topic Classification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.19638v2](http://arxiv.org/pdf/2411.19638v2)**

> **作者:** Taja Kuzman; Nikola Ljubešić
>
> **备注:** This work has been accepted and published in the IEEE Access journal. This arXiv version is retained for archival purposes. Readers should use and cite the IEEE Access Version available at https://ieeexplore.ieee.org/document/10900365
>
> **摘要:** With the ever-increasing number of news stories available online, classifying them by topic, regardless of the language they are written in, has become crucial for enhancing readers' access to relevant content. To address this challenge, we propose a teacher-student framework based on large language models (LLMs) for developing multilingual news topic classification models of reasonable size with no need for manual data annotation. The framework employs a Generative Pretrained Transformer (GPT) model as the teacher model to develop a news topic training dataset through automatic annotation of 20,000 news articles in Slovenian, Croatian, Greek, and Catalan. Articles are classified into 17 main categories from the Media Topic schema, developed by the International Press Telecommunications Council (IPTC). The teacher model exhibits high zero-shot performance in all four languages. Its agreement with human annotators is comparable to that between the human annotators themselves. To mitigate the computational limitations associated with the requirement of processing millions of texts daily, smaller BERT-like student models are fine-tuned on the GPT-annotated dataset. These student models achieve high performance comparable to the teacher model. Furthermore, we explore the impact of the training data size on the performance of the student models and investigate their monolingual, multilingual, and zero-shot cross-lingual capabilities. The findings indicate that student models can achieve high performance with a relatively small number of training instances, and demonstrate strong zero-shot cross-lingual abilities. Finally, we publish the best-performing news topic classifier, enabling multilingual classification with the top-level categories of the IPTC Media Topic schema.
>
---
#### [replaced 102] Utilizing Multilingual Encoders to Improve Large Language Models for Low-Resource Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.09091v3](http://arxiv.org/pdf/2508.09091v3)**

> **作者:** Imalsha Puranegedara; Themira Chathumina; Nisal Ranathunga; Nisansa de Silva; Surangika Ranathunga; Mokanarangan Thayaparan
>
> **摘要:** Large Language Models (LLMs) excel in English, but their performance degrades significantly on low-resource languages (LRLs) due to English-centric training. While methods like LangBridge align LLMs with multilingual encoders such as the Massively Multilingual Text-to-Text Transfer Transformer (mT5), they typically use only the final encoder layer. We propose a novel architecture that fuses all intermediate layers, enriching the linguistic information passed to the LLM. Our approach features two strategies: (1) a Global Softmax weighting for overall layer importance, and (2) a Transformer Softmax model that learns token-specific weights. The fused representations are mapped into the LLM's embedding space, enabling it to process multilingual inputs. The model is trained only on English data, without using any parallel or multilingual data. Evaluated on XNLI, IndicXNLI, Sinhala News Classification, and Amazon Reviews, our Transformer Softmax model significantly outperforms the LangBridge baseline. We observe strong performance gains in LRLs, improving Sinhala classification accuracy from 71.66% to 75.86% and achieving clear improvements across Indic languages such as Tamil, Bengali, and Malayalam. These specific gains contribute to an overall boost in average XNLI accuracy from 70.36% to 71.50%. This approach offers a scalable, data-efficient path toward more capable and equitable multilingual LLMs.
>
---
#### [replaced 103] Atomic Consistency Preference Optimization for Long-Form Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.09039v2](http://arxiv.org/pdf/2505.09039v2)**

> **作者:** Jingfeng Chen; Raghuveer Thirukovalluru; Junlin Wang; Kaiwei Luo; Bhuwan Dhingra
>
> **备注:** 13 pages, 1 figure
>
> **摘要:** Large Language Models (LLMs) often produce factoid hallucinations - plausible yet incorrect answers. A common mitigation strategy is model alignment, which improves factual accuracy by training on curated (factual, non-factual) pairs. However, this approach often relies on a stronger model (e.g., GPT-4) or an external knowledge base to assess factual correctness that may not always be accessible. Addressing this, we propose Atomic Consistency Preference Optimization (ACPO), a self-supervised preference-tuning method that enhances factual accuracy without external supervision. ACPO leverages atomic consistency signals (i.e., the agreement of individual facts across multiple stochastic responses) to identify high- and low-quality data pairs for model alignment. Despite being fully self-supervised, ACPO outperforms the strong supervised alignment baseline by 1.95 points averaged across Phi-3 and Llama3 on the LongFact and BioGen datasets, demonstrating its effectiveness in improving factual reliability without relying on external models or knowledge bases.
>
---
