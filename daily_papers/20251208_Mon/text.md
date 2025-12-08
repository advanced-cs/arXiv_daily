# 自然语言处理 cs.CL

- **最新发布 44 篇**

- **更新 30 篇**

## 最新发布

#### [new 001] Transformer-Enabled Diachronic Analysis of Vedic Sanskrit: Neural Methods for Quantifying Types of Language Change
- **分类: cs.CL**

- **简介: 该论文属于计算语言学任务，旨在解决低资源、形态丰富的梵语历时演变分析问题。作者提出混合神经-符号方法，利用弱监督和正则表达式生成伪标签，结合多语言BERT与置信度加权集成，量化语言变化，揭示梵语形态复杂性动态重分布而非简化。**

- **链接: [https://arxiv.org/pdf/2512.05364v1](https://arxiv.org/pdf/2512.05364v1)**

> **作者:** Ananth Hariharan; David Mortensen
>
> **摘要:** This study demonstrates how hybrid neural-symbolic methods can yield significant new insights into the evolution of a morphologically rich, low-resource language. We challenge the naive assumption that linguistic change is simplification by quantitatively analyzing over 2,000 years of Sanskrit, demonstrating how weakly-supervised hybrid methods can yield new insights into the evolution of morphologically rich, low-resource languages. Our approach addresses data scarcity through weak supervision, using 100+ high-precision regex patterns to generate pseudo-labels for fine-tuning a multilingual BERT. We then fuse symbolic and neural outputs via a novel confidence-weighted ensemble, creating a system that is both scalable and interpretable. Applying this framework to a 1.47-million-word diachronic corpus, our ensemble achieves a 52.4% overall feature detection rate. Our findings reveal that Sanskrit's overall morphological complexity does not decrease but is instead dynamically redistributed: while earlier verbal features show cyclical patterns of decline, complexity shifts to other domains, evidenced by a dramatic expansion in compounding and the emergence of new philosophical terminology. Critically, our system produces well-calibrated uncertainty estimates, with confidence strongly correlating with accuracy (Pearson r = 0.92) and low overall calibration error (ECE = 0.043), bolstering the reliability of these findings for computational philology.
>
---
#### [new 002] Grounded Multilingual Medical Reasoning for Question Answering with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦多语言医学问答任务，旨在提升大模型在医疗领域的推理可靠性。通过构建基于维基医学知识的检索增强生成方法，生成英意西三种语言的50万条推理链，并发布翻译数据集与微调模型，显著提升问答性能。**

- **链接: [https://arxiv.org/pdf/2512.05658v1](https://arxiv.org/pdf/2512.05658v1)**

> **作者:** Pietro Ferrazzi; Aitor Soroa; Rodrigo Agerri
>
> **备注:** Under Review
>
> **摘要:** Large Language Models (LLMs) with reasoning capabilities have recently demonstrated strong potential in medical Question Answering (QA). Existing approaches are largely English-focused and primarily rely on distillation from general-purpose LLMs, raising concerns about the reliability of their medical knowledge. In this work, we present a method to generate multilingual reasoning traces grounded in factual medical knowledge. We produce 500k traces in English, Italian, and Spanish, using a retrievalaugmented generation approach over medical information from Wikipedia. The traces are generated to solve medical questions drawn from MedQA and MedMCQA, which we extend to Italian and Spanish. We test our pipeline in both in-domain and outof-domain settings across Medical QA benchmarks, and demonstrate that our reasoning traces improve performance both when utilized via in-context learning (few-shot) and supervised fine-tuning, yielding state-of-the-art results among 8B-parameter LLMs. We believe that these resources can support the development of safer, more transparent clinical decision-support tools in multilingual settings. We release the full suite of resources: reasoning traces, translated QA datasets, Medical-Wikipedia, and fine-tuned models.
>
---
#### [new 003] LYNX: Learning Dynamic Exits for Confidence-Controlled Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于推理效率优化任务，旨在解决大模型推理中的“过思考”问题。作者提出LYNX方法，利用模型自身隐状态和自然推理线索实现在线早停，通过一次训练即可跨任务复用，并提供可调的置信度控制，在多任务上显著减少计算量且保持或提升准确率。**

- **链接: [https://arxiv.org/pdf/2512.05325v1](https://arxiv.org/pdf/2512.05325v1)**

> **作者:** Ömer Faruk Akgül; Yusuf Hakan Kalaycı; Rajgopal Kannan; Willie Neiswanger; Viktor Prasanna
>
> **摘要:** Large reasoning models achieve strong performance on complex tasks by generating extended chains of thought, but they often "overthink": continuing to reason long after they have enough information to answer correctly. This wastes inference-time compute and can hurt accuracy. Existing attempts to stop early either manipulate decoding with extra sampling and heuristics, rely on auxiliary verifier models, or operate only as post-hoc analysis pipelines without formal guarantees. We introduce LYNX, an online early-exit mechanism that turns a model's own hidden-state awareness into confidence-controlled stopping decisions. LYNX attaches exit decisions to naturally occurring reasoning cues (e.g., "hmm", "wait") during generation, trains a lightweight probe on hidden states at those cue tokens using supervision from forced exits, and wraps the resulting scores in split conformal prediction to obtain distribution-free control over premature exits. Crucially, we train and calibrate this probe once on a generic mathematical corpus and reuse it unchanged across benchmarks, decoding temperatures, and even non-mathematical tasks. Across three model families spanning 1.5B to 32B parameters, a single mathematically trained probe per base model yields strong accuracy--efficiency tradeoffs. On GSM8K, LYNX matches or improves baseline accuracy while reducing tokens by 40--65\%; on MATH-500 it improves accuracy by up to 12 points with roughly 35--60\% fewer tokens; on AIME 2024 it recovers baseline accuracy with more than 50\% token savings; and on CommonsenseQA, a non-math benchmark, it transfers zero-shot with modest accuracy gains and up to 70\% fewer tokens. Compared to state-of-the-art early-exit methods, LYNX offers competitive or superior Pareto frontiers while remaining fully online, requiring no proxy models at inference, and providing explicit, user-tunable confidence guarantees.
>
---
#### [new 004] Interleaved Latent Visual Reasoning with Selective Perceptual Modeling
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究多模态大模型中的视觉推理任务，旨在解决现有方法在计算成本、感知精度与动态推理间的权衡问题。作者提出ILVR框架，通过交错文本生成与潜变量视觉表征，结合选择性自监督机制，实现高效且精细的多模态推理。**

- **链接: [https://arxiv.org/pdf/2512.05665v1](https://arxiv.org/pdf/2512.05665v1)**

> **作者:** Shuai Dong; Siyuan Wang; Xingyu Liu; Zhongyu Wei
>
> **备注:** 11 pages, 6 figures. Code available at https://github.com/XD111ds/ILVR
>
> **摘要:** Interleaved reasoning paradigms enhance Multimodal Large Language Models (MLLMs) with visual feedback but are hindered by the prohibitive computational cost of repeatedly re-encoding pixel-dense images. A promising alternative, latent visual reasoning, circumvents this bottleneck yet currently forces a critical trade-off: methods either sacrifice precise perceptual modeling by over-compressing features or fail to model dynamic problems due to static, non-interleaved structures. We introduce Interleaved Latent Visual Reasoning (ILVR), a framework that unifies dynamic state evolution with precise perceptual modeling. ILVR interleaves textual generation with latent visual representations that act as specific, evolving cues for subsequent reasoning. To enable this, we employ a self-supervision strategy where a Momentum Teacher Model selectively distills relevant features from helper images into sparse supervision targets. This adaptive selection mechanism guides the model to autonomously generate context-aware visual signals. Extensive experiments on multimodal reasoning benchmarks demonstrate that ILVR significantly outperforms existing approaches, effectively bridging the gap between fine-grained perception and sequential multimodal reasoning.
>
---
#### [new 005] Exposing Pink Slime Journalism: Linguistic Signatures and Robust Detection Against LLM-Generated Threats
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本检测任务，旨在识别“粉红 slime 新闻”（低质伪本地新闻）。研究分析其语言特征，揭示大模型篡改对检测系统的破坏，并提出一种鲁棒学习框架，提升对抗生成内容的检测能力。**

- **链接: [https://arxiv.org/pdf/2512.05331v1](https://arxiv.org/pdf/2512.05331v1)**

> **作者:** Sadat Shahriar; Navid Ayoobi; Arjun Mukherjee; Mostafa Musharrat; Sai Vishnu Vamsi
>
> **备注:** Published in RANLP 2025
>
> **摘要:** The local news landscape, a vital source of reliable information for 28 million Americans, faces a growing threat from Pink Slime Journalism, a low-quality, auto-generated articles that mimic legitimate local reporting. Detecting these deceptive articles requires a fine-grained analysis of their linguistic, stylistic, and lexical characteristics. In this work, we conduct a comprehensive study to uncover the distinguishing patterns of Pink Slime content and propose detection strategies based on these insights. Beyond traditional generation methods, we highlight a new adversarial vector: modifications through large language models (LLMs). Our findings reveal that even consumer-accessible LLMs can significantly undermine existing detection systems, reducing their performance by up to 40% in F1-score. To counter this threat, we introduce a robust learning framework specifically designed to resist LLM-based adversarial attacks and adapt to the evolving landscape of automated pink slime journalism, and showed and improvement by up to 27%.
>
---
#### [new 006] Capturing Classic Authorial Style in Long-Form Story Generation with GRPO Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文研究长篇故事生成中的作者风格控制问题，提出基于GRPO微调的框架，结合细粒度风格奖励（来自作者验证信号）与内容完整性优化。实验以马克·吐温风格为例，8B模型在风格相似度上超越GPT-4o等大模型，验证了小模型实现风格化生成的可行性。**

- **链接: [https://arxiv.org/pdf/2512.05747v1](https://arxiv.org/pdf/2512.05747v1)**

> **作者:** Jinlong Liu; Mohammed Bahja; Venelin Kovatchev; Mark Lee
>
> **摘要:** Recent advances in large language models (LLMs) show impressive performance in open-ended story generation, but fine-grained stylistic control remains limited. Existing methods often rely on shallow cues (e.g., names or topics) to simulate authorial style, without robust evaluation. In this work, we present a training framework for style-conditioned story generation using Group Relative Policy Optimization (GRPO) and a custom multi-reward setup. The style reward is derived from a fine-tuned sentence transformer using authorship verification (AV) signals, combined with content and completeness scores to stabilize long-form narrative generation. We conduct experiments using fiction by Mark Twain, a prominent 19th-century American author, with The Adventures of Huckleberry Finn serving as the reference style exemplar. Our 8B model outperforms larger baselines such as GPT-4o and Claude Sonnet 4 in AV-style metrics, achieving a style score of 0.628 and competitive content quality. Results demonstrate the feasibility of agentic stylistic generation with moderate model size and task-specific training. While the output is clearly style-aligned, narrative completeness remains a challenge, indicating future work is needed to better model global coherence and story resolution.
>
---
#### [new 007] Structured Reasoning with Tree-of-Thoughts for Bengali Math Word Problems
- **分类: cs.CL**

- **简介: 该论文研究孟加拉语数学应用题求解，针对链式思维推理易传播错误的问题，提出采用树状思维（ToT）方法。基于SOMADHAN数据集，在多种大模型上实验表明，ToT相比标准和链式提示显著提升准确率，尤其在大模型上有效，验证了其在低资源语言数学推理中的优越性。**

- **链接: [https://arxiv.org/pdf/2512.05580v1](https://arxiv.org/pdf/2512.05580v1)**

> **作者:** Aurprita Mahmood; Sabrin alam; Neloy kumer Sagor; Md. Abdul Hadi; Md. Sehab Al Islam; Minhajul Islam
>
> **摘要:** Mathematical Word Problems (MWPs) are among the most challenging tasks in natural language processing because they require both linguistic understanding and multi-step numerical reasoning. While Chain-of-Thought (CoT) prompting has shown promise, its linear structure often propagates errors, limiting overall effectiveness. To address this limitation, we present the a systematic study of Tree-of-Thought (ToT) reasoning for Bengali MWPs using the SOMADHAN dataset. Owing to computational and token-cost constraints, we evaluate a curated set of 100 representative problems across multiple large language models (LLMs), including GPT-OSS and LLaMA variants, under standard prompting, CoT, and ToT strategies. Our results show that CoT improves baseline accuracy from 78% (standard prompting) to 83% on average, while ToT further increases performance by up to 5 percentage points, achieving 88% accuracy with GPT-OSS-120B. These improvements highlight that ToT is particularly effective in medium-to-large-scale models but may offer less advantage for smaller ones. Overall, our findings establish ToT as a robust framework for solving mathematical problems in low-resource languages such as Bengali. More broadly, this study shows that structured reasoning methods like ToT can provide more reliable and globally consistent outcomes than CoT, paving the way for better reasoning strategies in multilingual NLP.
>
---
#### [new 008] To Think or Not to Think: The Hidden Cost of Meta-Training with Excessive CoT Examples
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究元训练中过多思维链（CoT）示例的负面影响，旨在提升大模型在新任务上的少样本推理能力。提出CoT-Recipe方法，调控CoT与非CoT示例比例，显著提升模型在无CoT上下文时的推理准确率。**

- **链接: [https://arxiv.org/pdf/2512.05318v1](https://arxiv.org/pdf/2512.05318v1)**

> **作者:** Vignesh Kothapalli; Ata Fatahibaarzi; Hamed Firooz; Maziar Sanjabi
>
> **备注:** 26 pages, 45 figures, 3 tables
>
> **摘要:** Chain-of-thought (CoT) prompting combined with few-shot in-context learning (ICL) has unlocked significant reasoning capabilities in large language models (LLMs). However, ICL with CoT examples is ineffective on novel tasks when the pre-training knowledge is insufficient. We study this problem in a controlled setting using the CoT-ICL Lab framework, and propose meta-training techniques to learn novel abstract reasoning tasks in-context. Although CoT examples facilitate reasoning, we noticed that their excessive inclusion during meta-training degrades performance when CoT supervision is limited. To mitigate such behavior, we propose CoT-Recipe, a formal approach to modulate the mix of CoT and non-CoT examples in meta-training sequences. We demonstrate that careful modulation via CoT-Recipe can increase the accuracy of transformers on novel tasks by up to 300% even when there are no CoT examples available in-context. We confirm the broader effectiveness of these techniques by applying them to pretrained LLMs (Qwen2.5 series) for symbolic reasoning tasks and observing gains of up to 130% in accuracy.
>
---
#### [new 009] Automated Identification of Incidentalomas Requiring Follow-Up: A Multi-Anatomy Evaluation of LLM-Based and Supervised Approaches
- **分类: cs.CL**

- **简介: 该论文研究如何自动识别放射报告中需随访的偶发瘤。针对现有方法仅做文档级分类的问题，提出基于大语言模型和监督模型的病灶级检测，引入解剖结构提示和病灶标记输入，显著提升了细粒度识别性能。**

- **链接: [https://arxiv.org/pdf/2512.05537v1](https://arxiv.org/pdf/2512.05537v1)**

> **作者:** Namu Park; Farzad Ahmed; Zhaoyi Sun; Kevin Lybarger; Ethan Breinhorst; Julie Hu; Ozlem Uzuner; Martin Gunn; Meliha Yetisgen
>
> **摘要:** Objective: To evaluate large language models (LLMs) against supervised baselines for fine-grained, lesion-level detection of incidentalomas requiring follow-up, addressing the limitations of current document-level classification systems. Methods: We utilized a dataset of 400 annotated radiology reports containing 1,623 verified lesion findings. We compared three supervised transformer-based encoders (BioClinicalModernBERT, ModernBERT, Clinical Longformer) against four generative LLM configurations (Llama 3.1-8B, GPT-4o, GPT-OSS-20b). We introduced a novel inference strategy using lesion-tagged inputs and anatomy-aware prompting to ground model reasoning. Performance was evaluated using class-specific F1-scores. Results: The anatomy-informed GPT-OSS-20b model achieved the highest performance, yielding an incidentaloma-positive macro-F1 of 0.79. This surpassed all supervised baselines (maximum macro-F1: 0.70) and closely matched the inter-annotator agreement of 0.76. Explicit anatomical grounding yielded statistically significant performance gains across GPT-based models (p < 0.05), while a majority-vote ensemble of the top systems further improved the macro-F1 to 0.90. Error analysis revealed that anatomy-aware LLMs demonstrated superior contextual reasoning in distinguishing actionable findings from benign lesions. Conclusion: Generative LLMs, when enhanced with structured lesion tagging and anatomical context, significantly outperform traditional supervised encoders and achieve performance comparable to human experts. This approach offers a reliable, interpretable pathway for automated incidental finding surveillance in radiology workflows.
>
---
#### [new 010] Optimizing Medical Question-Answering Systems: A Comparative Study of Fine-Tuned and Zero-Shot Large Language Models with RAG Framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究医学问答任务，旨在提升大语言模型在临床领域的事实准确性和可靠性。通过结合检索增强生成（RAG）与微调开源大模型（LLaMA2、Falcon），利用医学文献检索结果辅助回答，显著提高准确率并减少幻觉。**

- **链接: [https://arxiv.org/pdf/2512.05863v1](https://arxiv.org/pdf/2512.05863v1)**

> **作者:** Tasnimul Hassan; Md Faisal Karim; Haziq Jeelani; Elham Behnam; Robert Green; Fayeq Jeelani Syed
>
> **摘要:** Medical question-answering (QA) systems can benefit from advances in large language models (LLMs), but directly applying LLMs to the clinical domain poses challenges such as maintaining factual accuracy and avoiding hallucinations. In this paper, we present a retrieval-augmented generation (RAG) based medical QA system that combines domain-specific knowledge retrieval with open-source LLMs to answer medical questions. We fine-tune two state-of-the-art open LLMs (LLaMA~2 and Falcon) using Low-Rank Adaptation (LoRA) for efficient domain specialization. The system retrieves relevant medical literature to ground the LLM's answers, thereby improving factual correctness and reducing hallucinations. We evaluate the approach on benchmark datasets (PubMedQA and MedMCQA) and show that retrieval augmentation yields measurable improvements in answer accuracy compared to using LLMs alone. Our fine-tuned LLaMA~2 model achieves 71.8% accuracy on PubMedQA, substantially improving over the 55.4% zero-shot baseline, while maintaining transparency by providing source references. We also detail the system design and fine-tuning methodology, demonstrating that grounding answers in retrieved evidence reduces unsupported content by approximately 60%. These results highlight the potential of RAG-augmented open-source LLMs for reliable biomedical QA, pointing toward practical clinical informatics applications.
>
---
#### [new 011] Prompting Science Report 4: Playing Pretend: Expert Personas Don't Improve Factual Accuracy
- **分类: cs.CL**

- **简介: 该论文研究专家角色设定是否提升AI在高难度客观题中的准确率。通过测试六种模型在科学、工程、法律等领域的表现，发现赋予专家身份对事实准确性无显著帮助，错配或低知识角色甚至降低性能。**

- **链接: [https://arxiv.org/pdf/2512.05858v1](https://arxiv.org/pdf/2512.05858v1)**

> **作者:** Savir Basil; Ina Shapiro; Dan Shapiro; Ethan Mollick; Lilach Mollick; Lennart Meincke
>
> **摘要:** This is the fourth in a series of short reports that help business, education, and policy leaders understand the technical details of working with AI through rigorous testing. Here, we ask whether assigning personas to models improves performance on difficult objective multiple-choice questions. We study both domain-specific expert personas and low-knowledge personas, evaluating six models on GPQA Diamond (Rein et al. 2024) and MMLU-Pro (Wang et al. 2024), graduate-level questions spanning science, engineering, and law. We tested three approaches: -In-Domain Experts: Assigning the model an expert persona ("you are a physics expert") matched to the problem type (physics problems) had no significant impact on performance (with the exception of the Gemini 2.0 Flash model). -Off-Domain Experts (Domain-Mismatched): Assigning the model an expert persona ("you are a physics expert") not matched to the problem type (law problems) resulted in marginal differences. -Low-Knowledge Personas: We assigned the model negative capability personas (layperson, young child, toddler), which were generally harmful to benchmark accuracy. Across both benchmarks, persona prompts generally did not improve accuracy relative to a no-persona baseline. Expert personas showed no consistent benefit across models, with few exceptions. Domain-mismatched expert personas sometimes degraded performance. Low-knowledge personas often reduced accuracy. These results are about the accuracy of answers only; personas may serve other purposes (such as altering the tone of outputs), beyond improving factual performance.
>
---
#### [new 012] Efficient Text Classification with Conformal In-Context Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究文本分类任务，旨在解决大语言模型在上下文学习中依赖提示设计且计算成本高的问题。作者提出并评估了CICLe框架，结合轻量分类器与保形预测，自适应减少候选类别，提升效率与性能，尤其在类别不平衡场景下表现优越。**

- **链接: [https://arxiv.org/pdf/2512.05732v1](https://arxiv.org/pdf/2512.05732v1)**

> **作者:** Ippokratis Pantelidis; Korbinian Randl; Aron Henriksson
>
> **备注:** 10 pages, 4 tables, 2 figures
>
> **摘要:** Large Language Models (LLMs) demonstrate strong in-context learning abilities, yet their effectiveness in text classification depends heavily on prompt design and incurs substantial computational cost. Conformal In-Context Learning (CICLe) has been proposed as a resource-efficient framework that integrates a lightweight base classifier with Conformal Prediction to guide LLM prompting by adaptively reducing the set of candidate classes. However, its broader applicability and efficiency benefits beyond a single domain have not yet been systematically explored. In this paper, we present a comprehensive evaluation of CICLe across diverse NLP classification benchmarks. The results show that CICLe consistently improves over its base classifier and outperforms few-shot prompting baselines when the sample size is sufficient for training the base classifier, and performs comparably in low-data regimes. In terms of efficiency, CICLe reduces the number of shots and prompt length by up to 34.45% and 25.16%, respectively, and enables the use of smaller models with competitive performance. CICLe is furthermore particularly advantageous for text classification tasks with high class imbalance. These findings highlight CICLe as a practical and scalable approach for efficient text classification, combining the robustness of traditional classifiers with the adaptability of LLMs, and achieving substantial gains in data and computational efficiency.
>
---
#### [new 013] SEA-SafeguardBench: Evaluating AI Safety in SEA Languages and Cultures
- **分类: cs.CL**

- **简介: 该论文聚焦AI安全评估任务，针对现有基准英语中心化和忽视文化差异的问题，构建了首个覆盖八种东南亚语言、经人工验证的安全基准SEA-SafeguardBench，旨在更准确评估大模型在本地文化语境下的有害内容识别能力。**

- **链接: [https://arxiv.org/pdf/2512.05501v1](https://arxiv.org/pdf/2512.05501v1)**

> **作者:** Panuthep Tasawong; Jian Gang Ngui; Alham Fikri Aji; Trevor Cohn; Peerat Limkonchotiwat
>
> **备注:** Under review
>
> **摘要:** Safeguard models help large language models (LLMs) detect and block harmful content, but most evaluations remain English-centric and overlook linguistic and cultural diversity. Existing multilingual safety benchmarks often rely on machine-translated English data, which fails to capture nuances in low-resource languages. Southeast Asian (SEA) languages are underrepresented despite the region's linguistic diversity and unique safety concerns, from culturally sensitive political speech to region-specific misinformation. Addressing these gaps requires benchmarks that are natively authored to reflect local norms and harm scenarios. We introduce SEA-SafeguardBench, the first human-verified safety benchmark for SEA, covering eight languages, 21,640 samples, across three subsets: general, in-the-wild, and content generation. The experimental results from our benchmark demonstrate that even state-of-the-art LLMs and guardrails are challenged by SEA cultural and harm scenarios and underperform when compared to English texts.
>
---
#### [new 014] LMSpell: Neural Spell Checking for Low-Resource Languages
- **分类: cs.CL**

- **简介: 该论文聚焦低资源语言的拼写纠错任务，旨在解决现有预训练语言模型在此类语言上应用不足的问题。作者系统评估了不同模型的效果，提出LMSpell工具包，并通过僧伽罗语案例验证其有效性。**

- **链接: [https://arxiv.org/pdf/2512.05414v1](https://arxiv.org/pdf/2512.05414v1)**

> **作者:** Akesh Gunathilakea; Nadil Karunarathnea; Tharusha Bandaranayakea; Nisansa de Silvaa; Surangika Ranathunga
>
> **摘要:** Spell correction is still a challenging problem for low-resource languages (LRLs). While pretrained language models (PLMs) have been employed for spell correction, their use is still limited to a handful of languages, and there has been no proper comparison across PLMs. We present the first empirical study on the effectiveness of PLMs for spell correction, which includes LRLs. We find that Large Language Models (LLMs) outperform their counterparts (encoder-based and encoder-decoder) when the fine-tuning dataset is large. This observation holds even in languages for which the LLM is not pre-trained. We release LMSpell, an easy- to use spell correction toolkit across PLMs. It includes an evaluation function that compensates for the hallucination of LLMs. Further, we present a case study with Sinhala to shed light on the plight of spell correction for LRLs.
>
---
#### [new 015] Heard or Halted? Gender, Interruptions, and Emotional Tone in U.S. Supreme Court Oral Arguments
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究美国最高法院口头辩论中性别与打断现象的关系，探讨打断是否影响论点内容及情感倾向。基于语料库，采用句子嵌入和情感词典分析，发现打断未显著改变语义，但针对女性律师的打断更具负面情绪。**

- **链接: [https://arxiv.org/pdf/2512.05832v1](https://arxiv.org/pdf/2512.05832v1)**

> **作者:** Yifei Tong
>
> **备注:** 12 pages, 5 figures, 1 table. Includes appendix. Code available at: https://github.com/1TSHARUKA/Emotional_Interruption_Analysis
>
> **摘要:** This study examines how interruptions during U.S. Supreme Court oral arguments shape both the semantic content and emotional tone of advocates' speech, with a focus on gendered dynamics in judicial discourse. Using the ConvoKit Supreme Court Corpus (2010-2019), we analyze 12,663 speech chunks from advocate-justice interactions to assess whether interruptions alter the meaning of an advocate's argument and whether interruptions toward female advocates exhibit more negative emotional valence. Semantic shifts are quantified using GloVe-based sentence embeddings, while sentiment is measured through lexicon-based analysis. We find that semantic similarity between pre- and post-interruption speech remains consistently high, suggesting that interruptions do not substantially alter argumentative content. However, interruptions directed at female advocates contain significantly higher levels of negative sentiment. These results deepen empirical understanding of gendered communication in elite institutional settings and demonstrate the value of computational linguistic methods for studying power, discourse, and equity in judicial proceedings.
>
---
#### [new 016] Decoding the Black Box: Discerning AI Rhetorics About and Through Poetic Prompting
- **分类: cs.CL; cs.CY**

- **简介: 该论文探讨诗歌提示模式在提示工程中的应用，旨在揭示大语言模型的算法倾向与偏见。通过诗意提示分析模型对著名诗人作品的描述与改写，检验其适应与重构创意文本的能力，属于AI生成内容与批判性评估任务。**

- **链接: [https://arxiv.org/pdf/2512.05243v1](https://arxiv.org/pdf/2512.05243v1)**

> **作者:** P. D. Edgar; Alia Hall
>
> **备注:** Late-Breaking Paper accepted to IEEE SSCI 2025 NLP & Social Media Track as extended abstract and presented in Trondheim, Norway 17-20 March 2025 as Poster Presentation
>
> **摘要:** Prompt engineering has emerged as a useful way studying the algorithmic tendencies and biases of large language models. Meanwhile creatives and academics have leveraged LLMs to develop creative works and explore the boundaries of their writing capabilities through text generation and code. This study suggests that creative text prompting, specifically Poetry Prompt Patterns, may be a useful addition to the toolbox of the prompt engineer, and outlines the process by which this approach may be taken. Then, the paper uses poetic prompts to assess descriptions and evaluations of three models of a renowned poet and test the consequences of the willingness of models to adapt or rewrite original creative works for presumed audiences.
>
---
#### [new 017] Retrieving Semantically Similar Decisions under Noisy Institutional Labels: Robust Comparison of Embedding Methods
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律文本检索任务，旨在解决因司法数据库标签噪声导致的案例相似性检索难题。作者比较了通用与领域专用嵌入模型在捷克宪法法院判例中的表现，提出一种抗噪评估框架，发现通用模型显著优于领域预训练模型。**

- **链接: [https://arxiv.org/pdf/2512.05681v1](https://arxiv.org/pdf/2512.05681v1)**

> **作者:** Tereza Novotna; Jakub Harasta
>
> **备注:** The manuscript has been accepted for presentation as a short paper at the 38th International Conference on Legal Knowledge and Information Systems (JURIX 2025) in Torino, Italy
>
> **摘要:** Retrieving case law is a time-consuming task predominantly carried out by querying databases. We provide a comparison of two models in three different settings for Czech Constitutional Court decisions: (i) a large general-purpose embedder (OpenAI), (ii) a domain-specific BERT-trained from scratch on ~30,000 decisions using sliding windows and attention pooling. We propose a noise-aware evaluation including IDF-weighted keyword overlap as graded relevance, binarization via two thresholds (0.20 balanced, 0.28 strict), significance via paired bootstrap, and an nDCG diagnosis supported with qualitative analysis. Despite modest absolute nDCG (expected under noisy labels), the general OpenAI embedder decisively outperforms the domain pre-trained BERT in both settings at @10/@20/@100 across both thresholds; differences are statistically significant. Diagnostics attribute low absolutes to label drift and strong ideals rather than lack of utility. Additionally, our framework is robust enough to be used for evaluation under a noisy gold dataset, which is typical when handling data with heterogeneous labels stemming from legacy judicial databases.
>
---
#### [new 018] A Greek Government Decisions Dataset for Public-Sector Analysis and Insight
- **分类: cs.CL**

- **简介: 该论文构建了一个包含百万条希腊政府决策的开源语料库，旨在提升公共部门信息透明度。属于数据集构建与应用任务，解决了政府文档机器可读性与信息检索问题，提出了RAG问答任务并验证其潜力。**

- **链接: [https://arxiv.org/pdf/2512.05647v1](https://arxiv.org/pdf/2512.05647v1)**

> **作者:** Giorgos Antoniou; Giorgos Filandrianos; Aggelos Vlachos; Giorgos Stamou; Lampros Kollimenos; Konstantinos Skianis; Michalis Vazirgiannis
>
> **摘要:** We introduce an open, machine-readable corpus of Greek government decisions sourced from the national transparency platform Diavgeia. The resource comprises 1 million decisions, featuring and high-quality raw text extracted from PDFs. It is released with raw extracted text in Markdown format, alongside a fully reproducible extraction pipeline. Beyond the core dataset, we conduct qualitative analyses to explore boilerplate patterns and design a retrieval-augmented generation (RAG) task by formulating a set of representative questions, creating high-quality answers, and evaluating a baseline RAG system on its ability to retrieve and reason over public decisions. This evaluation demonstrates the potential of large-scale public-sector corpora to support advanced information access and transparency through structured retrieval and reasoning over governmental documents, and highlights how such a RAG pipeline could simulate a chat-based assistant capable of interactively answering questions about public decisions. Due to its scale, quality, and domain coverage, the corpus can also serve as high-value pre-training or fine-tuning material for new Language Models (LMs) and Large Language Models (LLMs) respectively, including specialized models for legal and governmental domains, and as a foundation for novel approaches in domain adaptation, knowledge-grounded generation, and explainable AI. Finally, we discuss limitations, outline future directions, and make both the data and the code accessible.
>
---
#### [new 019] MedTutor-R1: Socratic Personalized Medical Teaching with Multi-Agent Simulation
- **分类: cs.CL**

- **简介: 该论文针对临床教学中师资不足与协作训练缺失的问题，提出多智能体教学模拟器ClinEdu，构建群体苏格拉底教学数据集ClinTeach，并训练出支持一对多的多模态导师模型MedTutor-R1，显著提升医学教育的可扩展性与教学效果。**

- **链接: [https://arxiv.org/pdf/2512.05671v1](https://arxiv.org/pdf/2512.05671v1)**

> **作者:** Zhitao He; Haolin Yang; Zeyu Qin; Yi R Fung
>
> **备注:** Work In Progress
>
> **摘要:** The significant gap between rising demands for clinical training and the scarcity of expert instruction poses a major challenge to medical education. With powerful capabilities in personalized guidance, Large Language Models (LLMs) offer a promising solution to bridge this gap. However, current research focuses mainly on one-on-one knowledge instruction, overlooking collaborative reasoning, a key skill for students developed in teamwork like ward rounds. To this end, we develop ClinEdu, a multi-agent pedagogical simulator with personality-driven patients and diverse student cohorts, enabling controlled testing of complex pedagogical processes and scalable generation of teaching data. Based on ClinEdu, we construct ClinTeach, a large Socratic teaching dialogue dataset that captures the complexities of group instruction. We then train MedTutor-R1, the first multimodal Socratic tutor designed for one-to-many instruction in clinical medical education. MedTutor-R1 is first instruction-tuned on our ClinTeach dataset and then optimized with reinforcement learning, using rewards derived from a three-axis rubric, covering structural fidelity, analytical quality, and clinical safety, to refine its adaptive Socratic strategies. For authentic in-situ assessment, we use simulation-based interactive evaluation that redeploys the tutor back into ClinEdu. Experimental results demonstrate that our MedTutor-R1 outperforms the base model by over 20% in average pedagogical score and is comparable to o3, while also exhibiting high adaptability in handling a varying number of students. This promising performance underscores the effectiveness of our pedagogical simulator, ClinEdu.
>
---
#### [new 020] Enhancing Clinical Note Generation with ICD-10, Clinical Ontology Knowledge Graphs, and Chain-of-Thought Prompting Using GPT-4
- **分类: cs.CL; q-bio.QM**

- **简介: 该论文属临床文本生成任务，旨在缓解医生书写电子病历耗时问题。作者提出结合ICD编码、临床知识图谱与思维链提示的方法，利用GPT-4生成更高质量临床笔记，实验表明其优于标准单样本提示。**

- **链接: [https://arxiv.org/pdf/2512.05256v1](https://arxiv.org/pdf/2512.05256v1)**

> **作者:** Ivan Makohon; Mohamad Najafi; Jian Wu; Mathias Brochhausen; Yaohang Li
>
> **摘要:** In the past decade a surge in the amount of electronic health record (EHR) data in the United States, attributed to a favorable policy environment created by the Health Information Technology for Economic and Clinical Health (HITECH) Act of 2009 and the 21st Century Cures Act of 2016. Clinical notes for patients' assessments, diagnoses, and treatments are captured in these EHRs in free-form text by physicians, who spend a considerable amount of time entering and editing them. Manually writing clinical notes takes a considerable amount of a doctor's valuable time, increasing the patient's waiting time and possibly delaying diagnoses. Large language models (LLMs) possess the ability to generate news articles that closely resemble human-written ones. We investigate the usage of Chain-of-Thought (CoT) prompt engineering to improve the LLM's response in clinical note generation. In our prompts, we use as input International Classification of Diseases (ICD) codes and basic patient information. We investigate a strategy that combines the traditional CoT with semantic search results to improve the quality of generated clinical notes. Additionally, we infuse a knowledge graph (KG) built from clinical ontology to further enrich the domain-specific knowledge of generated clinical notes. We test our prompting technique on six clinical cases from the CodiEsp test dataset using GPT-4 and our results show that it outperformed the clinical notes generated by standard one-shot prompts.
>
---
#### [new 021] Dynamic Alignment for Collective Agency: Toward a Scalable Self-Improving Framework for Open-Ended LLM Alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型对齐任务，旨在解决传统对齐方法难以扩展且价值观局限的问题。提出“集体能动性”目标和“动态对齐”框架，通过自生成数据与自我奖励实现可扩展的自我改进对齐。**

- **链接: [https://arxiv.org/pdf/2512.05464v1](https://arxiv.org/pdf/2512.05464v1)**

> **作者:** Panatchakorn Anantaprayoon; Nataliia Babina; Jad Tarifi; Nima Asgharbeygi
>
> **备注:** 8 pages, 4 figures, to appear in AAAI 2026 AIGOV Workshop
>
> **摘要:** Large Language Models (LLMs) are typically aligned with human values using preference data or predefined principles such as helpfulness, honesty, and harmlessness. However, as AI systems progress toward Artificial General Intelligence (AGI) and Artificial Superintelligence (ASI), such value systems may become insufficient. In addition, human feedback-based alignment remains resource-intensive and difficult to scale. While AI-feedback-based self-improving alignment methods have been explored as a scalable alternative, they have largely remained constrained to conventional alignment values. In this work, we explore both a more holistic alignment objective and a scalable, self-improving alignment approach. Aiming to transcend conventional alignment norms, we introduce Collective Agency (CA)-a unified and open-ended alignment value that encourages integrated agentic capabilities. We also propose Dynamic Alignment-an alignment framework that enables an LLM to iteratively align itself. Dynamic Alignment comprises two key components: (1) automated training dataset generation with LLMs, and (2) a self-rewarding mechanism, where the policy model evaluates its own output candidates and assigns rewards for GRPO-based learning. Experimental results demonstrate that our approach successfully aligns the model to CA while preserving general NLP capabilities.
>
---
#### [new 022] ArtistMus: A Globally Diverse, Artist-Centric Benchmark for Retrieval-Augmented Music Question Answering
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文聚焦音乐领域问答任务，旨在解决大模型在音乐知识推理中因预训练数据不足导致的准确性低问题。作者构建了音乐维基数据库MusWikiDB和艺术家中心型评测集ArtistMus，提出检索增强生成方法提升事实准确性和上下文理解能力。**

- **链接: [https://arxiv.org/pdf/2512.05430v1](https://arxiv.org/pdf/2512.05430v1)**

> **作者:** Daeyong Kwon; SeungHeon Doh; Juhan Nam
>
> **备注:** Submitted to LREC 2026. This work is an evolution of our earlier preprint arXiv:2507.23334
>
> **摘要:** Recent advances in large language models (LLMs) have transformed open-domain question answering, yet their effectiveness in music-related reasoning remains limited due to sparse music knowledge in pretraining data. While music information retrieval and computational musicology have explored structured and multimodal understanding, few resources support factual and contextual music question answering (MQA) grounded in artist metadata or historical context. We introduce MusWikiDB, a vector database of 3.2M passages from 144K music-related Wikipedia pages, and ArtistMus, a benchmark of 1,000 questions on 500 diverse artists with metadata such as genre, debut year, and topic. These resources enable systematic evaluation of retrieval-augmented generation (RAG) for MQA. Experiments show that RAG markedly improves factual accuracy; open-source models gain up to +56.8 percentage points (for example, Qwen3 8B improves from 35.0 to 91.8), approaching proprietary model performance. RAG-style fine-tuning further boosts both factual recall and contextual reasoning, improving results on both in-domain and out-of-domain benchmarks. MusWikiDB also yields approximately 6 percentage points higher accuracy and 40% faster retrieval than a general-purpose Wikipedia corpus. We release MusWikiDB and ArtistMus to advance research in music information retrieval and domain-specific question answering, establishing a foundation for retrieval-augmented reasoning in culturally rich domains such as music.
>
---
#### [new 023] SQ-format: A Unified Sparse-Quantized Hardware-friendly Data Format for LLMs
- **分类: cs.CL**

- **简介: 该论文提出SQ-format，一种统一的稀疏-量化硬件友好格式，旨在解决低比特量化与稀疏化在硬件支持下难兼顾精度与效率的问题，实现LLM后训练量化的性能与吞吐帕累托优化。**

- **链接: [https://arxiv.org/pdf/2512.05409v1](https://arxiv.org/pdf/2512.05409v1)**

> **作者:** Ruixuan Huang; Hao Zeng; Hantao Huang; Jinyuan Shi; Minghui Yu; Ian En-Hsu Yen; Shuai Wang
>
> **摘要:** Post-training quantization (PTQ) plays a crucial role in the democratization of large language models (LLMs). However, existing low-bit quantization and sparsification techniques are difficult to balance accuracy and efficiency due to the limited hardware support. For example, W4A8 can only achieve the same peak TOPS as W8A8 whereas the GPU-supported sparse data format (2:4 semi-structure sparse) is seldomly adopted due to the loss of accuracy. To bridge this gap, in this paper, we propose the Sparse-Quantized Format (SQ-format), which is a unified data format for quantization and sparsification potentially easily supported by new hardware and existing GPUs. SQ-format makes use of the fact that sparse matrix can be accelerated in high-precision, and low-precision matrix multiplication can also be accelerated accordingly. As such, SQ-format is proposed to achieve Pareto improvement between performance and throughput. This format is particularly suitable for activations with outlier inequality status and makes their static compression possible. We show the state-of-the-art PTQ performance with SQ-format, propose the hardware required to support it, and further offer the design exploration and insights for the next-generation AI accelerators.
>
---
#### [new 024] Unveiling Affective Polarization Trends in Parliamentary Proceedings
- **分类: cs.CL**

- **简介: 该论文提出基于情感特征（效价、唤醒度、支配度）量化议会话语中的情感极化，旨在揭示以色列议会中执政党与反对党间随时间加剧的情感对立趋势，属于计算社会科学中的极化分析任务。**

- **链接: [https://arxiv.org/pdf/2512.05231v1](https://arxiv.org/pdf/2512.05231v1)**

> **作者:** Gili Goldin; Ella Rabinovich; Shuly Wintner
>
> **备注:** pre-MIT Press publication version
>
> **摘要:** Recent years have seen an increase in polarized discourse worldwide, on various platforms. We propose a novel method for quantifying polarization, based on the emotional style of the discourse rather than on differences in ideological stands. Using measures of Valence, Arousal and Dominance, we detect signals of emotional discourse and use them to operationalize the concept of affective polarization. Applying this method to a recently released corpus of proceedings of the Knesset, the Israeli parliament (in Hebrew), we find that the emotional style of members of government differs from that of opposition members; and that the level of affective polarization, as reflected by this style, is significantly increasing with time.
>
---
#### [new 025] Faithfulness metric fusion: Improving the evaluation of LLM trustworthiness across domains
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于LLM可信度评估任务，旨在提升事实一致性评价的准确性。通过融合多种基础指标，构建基于树模型的综合度量方法，并结合人类判断进行验证，在多领域中实现了与人类评价更高相关性的自动评估。**

- **链接: [https://arxiv.org/pdf/2512.05700v1](https://arxiv.org/pdf/2512.05700v1)**

> **作者:** Ben Malin; Tatiana Kalganova; Nikolaos Boulgouris
>
> **备注:** 9 pages, conference paper
>
> **摘要:** We present a methodology for improving the accuracy of faithfulness evaluation in Large Language Models (LLMs). The proposed methodology is based on the combination of elementary faithfulness metrics into a combined (fused) metric, for the purpose of improving the faithfulness of LLM outputs. The proposed strategy for metric fusion deploys a tree-based model to identify the importance of each metric, which is driven by the integration of human judgements evaluating the faithfulness of LLM responses. This fused metric is demonstrated to correlate more strongly with human judgements across all tested domains for faithfulness. Improving the ability to evaluate the faithfulness of LLMs, allows for greater confidence to be placed within models, allowing for their implementation in a greater diversity of scenarios. Additionally, we homogenise a collection of datasets across question answering and dialogue-based domains and implement human judgements and LLM responses within this dataset, allowing for the reproduction and trialling of faithfulness evaluation across domains.
>
---
#### [new 026] Mitigating Self-Preference by Authorship Obfuscation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型作为评测者时的自偏好偏见问题，旨在通过黑盒扰动（如同义词替换）隐藏回答来源，降低其自我识别能力以减轻偏见，发现简单扰动有效但完全消除仍具挑战。**

- **链接: [https://arxiv.org/pdf/2512.05379v1](https://arxiv.org/pdf/2512.05379v1)**

> **作者:** Taslim Mahbub; Shi Feng
>
> **摘要:** Language models (LMs) judges are widely used to evaluate the quality of LM outputs. Despite many advantages, LM judges display concerning biases that can impair their integrity in evaluations. One such bias is self-preference: LM judges preferring their own answers over those produced by other LMs or humans. The bias is hard to eliminate as frontier LM judges can distinguish their own outputs from those of others, even when the evaluation candidates are not labeled with their sources. In this paper, we investigate strategies to mitigate self-preference by reducing the LM judges' ability to recognize their own outputs. We apply black-box perturbations to evaluation candidates in pairwise comparison to obfuscate the authorship and reduce self-recognition. We find that perturbations as simple as synonym replacement for a few words predictably reduce self-preference. However, we also uncover fundamental challenges to eliminating the bias: when we extrapolate our perturbations to a more complete neutralization of stylistic differences between the evaluation candidates, self-preference recovers. Our findings suggest that self-recognition and self-preference can happen on many semantic levels, and complete mitigation remains challenging despite promising initial results.
>
---
#### [new 027] Fine-Tuning BERT for Domain-Specific Question Answering: Toward Educational NLP Resources at University Scale
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦教育领域问答任务，旨在解决大学课程信息查询的自动化问题。通过构建1,203个标注样本，微调BERT模型，提升其在特定学术领域的问答性能，验证了基础模型在高校教育知识系统中应用的可行性。**

- **链接: [https://arxiv.org/pdf/2512.05179v1](https://arxiv.org/pdf/2512.05179v1)**

> **作者:** Aurélie Montfrond
>
> **备注:** 4 pages, 2 figures
>
> **摘要:** Prior work on scientific question answering has largely emphasized chatbot-style systems, with limited exploration of fine-tuning foundation models for domain-specific reasoning. In this study, we developed a chatbot for the University of Limerick's Department of Electronic and Computer Engineering to provide course information to students. A custom dataset of 1,203 question-answer pairs in SQuAD format was constructed using the university book of modules, supplemented with manually and synthetically generated entries. We fine-tuned BERT (Devlin et al., 2019) using PyTorch and evaluated performance with Exact Match and F1 scores. Results show that even modest fine-tuning improves hypothesis framing and knowledge extraction, demonstrating the feasibility of adapting foundation models to educational domains. While domain-specific BERT variants such as BioBERT and SciBERT exist for biomedical and scientific literature, no foundation model has yet been tailored to university course materials. Our work addresses this gap by showing that fine-tuning BERT with academic QA pairs yields effective results, highlighting the potential to scale towards the first domain-specific QA model for universities and enabling autonomous educational knowledge systems.
>
---
#### [new 028] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文聚焦多语言多模态检索增强生成（RAG）任务，旨在解决现有VLMs因静态训练数据导致的知识局限问题。作者构建了大规模基准M4-RAG，涵盖42种语言、56种方言及8万余图文对，评估跨语言文化场景下的RAG性能，发现当前RAG难以有效扩展至大模型，揭示模型规模与检索效果间的不匹配问题。**

- **链接: [https://arxiv.org/pdf/2512.05959v1](https://arxiv.org/pdf/2512.05959v1)**

> **作者:** David Anugraha; Patrick Amadeus Irawan; Anshul Singh; En-Shiun Annie Lee; Genta Indra Winata
>
> **备注:** Preprint
>
> **摘要:** Vision-language models (VLMs) have achieved strong performance in visual question answering (VQA), yet they remain constrained by static training data. Retrieval-Augmented Generation (RAG) mitigates this limitation by enabling access to up-to-date, culturally grounded, and multilingual information; however, multilingual multimodal RAG remains largely underexplored. We introduce M4-RAG, a massive-scale benchmark covering 42 languages and 56 regional dialects and registers, comprising over 80,000 culturally diverse image-question pairs for evaluating retrieval-augmented VQA across languages and modalities. To balance realism with reproducibility, we build a controlled retrieval environment containing millions of carefully curated multilingual documents relevant to the query domains, approximating real-world retrieval conditions while ensuring consistent experimentation. Our systematic evaluation reveals that although RAG consistently benefits smaller VLMs, it fails to scale to larger models and often even degrades their performance, exposing a critical mismatch between model size and current retrieval effectiveness. M4-RAG provides a foundation for advancing next-generation RAG systems capable of reasoning seamlessly across languages, modalities, and cultural contexts.
>
---
#### [new 029] Learning from Self Critique and Refinement for Faithful LLM Summarization
- **分类: cs.CL**

- **简介: 该论文聚焦于提升大语言模型在摘要生成中的忠实性，解决幻觉问题。提出自批判与精炼偏好优化（SCRPO）框架，通过自我批判构建偏好数据并进行偏好学习，实现无需额外模型或测试时计算的高效训练，提升摘要忠实性与整体质量。**

- **链接: [https://arxiv.org/pdf/2512.05387v1](https://arxiv.org/pdf/2512.05387v1)**

> **作者:** Ting-Yao Hu; Hema Swetha Koppula; Hadi Pouransari; Cem Koc; Oncel Tuzel; Raviteja Vemulapalli
>
> **摘要:** Large Language Models (LLMs) often suffer from hallucinations: output content that is not grounded in the input context, when performing long-form text generation tasks such as summarization. Prior works have shown that hallucinations can be reduced by iteratively critiquing and refining previously generated outputs using either the same model or a more powerful teacher model as the critique. However, these approaches either require additional test-time compute or assume access to more powerful teacher models, making them costly and less practical. In this work, we propose Self Critique and Refinement-based Preference Optimization (SCRPO), which is a self-supervised training framework that first constructs a preference dataset by leveraging the LLM's own critique and refinement capabilities, and then applies preference learning to improve the same LLM for faithful summarization. Experiments on three summarization benchmarks (XSUM CNNDM and SAMSum), demonstrate that our approach outperforms state-of-the-art self-supervised learning methods in terms of faithfulness metrics while either maintaining or improving other metrics that measure the overall quality of the summary. Moreover, compared to test-time refinement, our approach not only improves efficiency but also results in more faithful summaries.
>
---
#### [new 030] SyncVoice: Towards Video Dubbing with Vision-Augmented Pretrained TTS Model
- **分类: eess.AS; cs.AI; cs.CL; cs.CV; cs.MM; cs.SD**

- **简介: 该论文研究视频配音任务，旨在解决现有方法在语音自然度、音画同步及多语言支持上的不足。作者提出SyncVoice框架，基于预训练TTS模型引入视觉信息并设计双说话人编码器，提升跨语言合成效果与音画一致性。**

- **链接: [https://arxiv.org/pdf/2512.05126v1](https://arxiv.org/pdf/2512.05126v1)**

> **作者:** Kaidi Wang; Yi He; Wenhao Guan; Weijie Wu; Hongwu Ding; Xiong Zhang; Di Wu; Meng Meng; Jian Luan; Lin Li; Qingyang Hong
>
> **摘要:** Video dubbing aims to generate high-fidelity speech that is precisely temporally aligned with the visual content. Existing methods still suffer from limitations in speech naturalness and audio-visual synchronization, and are limited to monolingual settings. To address these challenges, we propose SyncVoice, a vision-augmented video dubbing framework built upon a pretrained text-to-speech (TTS) model. By fine-tuning the TTS model on audio-visual data, we achieve strong audiovisual consistency. We propose a Dual Speaker Encoder to effectively mitigate inter-language interference in cross-lingual speech synthesis and explore the application of video dubbing in video translation scenarios. Experimental results show that SyncVoice achieves high-fidelity speech generation with strong synchronization performance, demonstrating its potential in video dubbing tasks.
>
---
#### [new 031] Zoom in, Click out: Unlocking and Evaluating the Potential of Zooming for GUI Grounding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究GUI界面元素定位任务，提出无需训练的ZoomClick方法，利用缩放操作的先验特性实现精准元素定位，并构建GUIZoom-Bench基准测试，提升模型在不同场景下的适应性与定位精度。**

- **链接: [https://arxiv.org/pdf/2512.05941v1](https://arxiv.org/pdf/2512.05941v1)**

> **作者:** Zhiyuan Jiang; Shenghao Xie; Wenyi Li; Wenqiang Zu; Peihang Li; Jiahao Qiu; Siqi Pei; Lei Ma; Tiejun Huang; Mengdi Wang; Shilong Liu
>
> **备注:** Code is available at https://github.com/Princeton-AI2-Lab/ZoomClick
>
> **摘要:** Grounding is a fundamental capability for building graphical user interface (GUI) agents. Although existing approaches rely on large-scale bounding box supervision, they still face various challenges, such as cross-platform generalization, complex layout analysis, and fine-grained element localization. In this paper, we investigate zoom as a strong yet underexplored prior for GUI grounding, and propose a training-free method, ZoomClick. By characterizing four key properties of zoom (i.e., pre-zoom, depth, shrink size, minimal crop size), we unlock its full capabilities for dynamic spatial focusing and adaptive context switching. Experiments demonstrate that our method significantly boosts the performance of both general vision-language and specialized GUI grounding models, achieving state-of-the-art results on several mainstream benchmarks; for example, UI-Venus-72B attains a 73.1% success rate on ScreenSpot-Pro. Furthermore, we present GUIZoom-Bench, a benchmark for evaluating model adaptability to zoom, aiming to inspire future research on improving zoom for further training and test-time scaling in GUI grounding tasks.
>
---
#### [new 032] Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations
- **分类: cs.AI; cs.CL; cs.IT; cs.LG; q-fin.CP**

- **简介: 该论文属LLM可信度评估任务，旨在解决生成内容的幻觉问题。提出基于信息论与热力学的语义保真度（SF）和语义熵产生（SEP）指标，通过QCA三元组建模与KL散度优化量化模型忠实性，并用于检测与控制幻觉。**

- **链接: [https://arxiv.org/pdf/2512.05156v1](https://arxiv.org/pdf/2512.05156v1)**

> **作者:** Igor Halperin
>
> **备注:** 23 pages, 6 figures
>
> **摘要:** Evaluating faithfulness of Large Language Models (LLMs) to a given task is a complex challenge. We propose two new unsupervised metrics for faithfulness evaluation using insights from information theory and thermodynamics. Our approach treats an LLM as a bipartite information engine where hidden layers act as a Maxwell demon controlling transformations of context $C $ into answer $A$ via prompt $Q$. We model Question-Context-Answer (QCA) triplets as probability distributions over shared topics. Topic transformations from $C$ to $Q$ and $A$ are modeled as transition matrices ${\bf Q}$ and ${\bf A}$ encoding the query goal and actual result, respectively. Our semantic faithfulness (SF) metric quantifies faithfulness for any given QCA triplet by the Kullback-Leibler (KL) divergence between these matrices. Both matrices are inferred simultaneously via convex optimization of this KL divergence, and the final SF metric is obtained by mapping the minimal divergence onto the unit interval [0,1], where higher scores indicate greater faithfulness. Furthermore, we propose a thermodynamics-based semantic entropy production (SEP) metric in answer generation, and show that high faithfulness generally implies low entropy production. The SF and SEP metrics can be used jointly or separately for LLM evaluation and hallucination control. We demonstrate our framework on LLM summarization of corporate SEC 10-K filings.
>
---
#### [new 033] On the Computability of Artificial General Intelligence
- **分类: cs.AI; cs.CL**

- **简介: 该论文探讨人工通用智能（AGI）的可计算性，旨在界定算法在创造力上的极限。作者基于已有AGI定义，证明任何算法无法产生全新的功能能力，从而得出AI无法真正创新的结论，并讨论其对AI发展与人类智能起源的意义。**

- **链接: [https://arxiv.org/pdf/2512.05212v1](https://arxiv.org/pdf/2512.05212v1)**

> **作者:** Georgios Mappouras; Charalambos Rossides
>
> **摘要:** In recent years we observed rapid and significant advancements in artificial intelligence (A.I.). So much so that many wonder how close humanity is to developing an A.I. model that can achieve human level of intelligence, also known as artificial general intelligence (A.G.I.). In this work we look at this question and we attempt to define the upper bounds, not just of A.I., but rather of any machine-computable process (a.k.a. an algorithm). To answer this question however, one must first precisely define A.G.I. We borrow prior work's definition of A.G.I. [1] that best describes the sentiment of the term, as used by the leading developers of A.I. That is, the ability to be creative and innovate in some field of study in a way that unlocks new and previously unknown functional capabilities in that field. Based on this definition we draw new bounds on the limits of computation. We formally prove that no algorithm can demonstrate new functional capabilities that were not already present in the initial algorithm itself. Therefore, no algorithm (and thus no A.I. model) can be truly creative in any field of study, whether that is science, engineering, art, sports, etc. In contrast, A.I. models can demonstrate existing functional capabilities, as well as combinations and permutations of existing functional capabilities. We conclude this work by discussing the implications of this proof both as it regards to the future of A.I. development, as well as to what it means for the origins of human intelligence.
>
---
#### [new 034] Natural Language Summarization Enables Multi-Repository Bug Localization by LLMs in Microservice Architectures
- **分类: cs.SE; cs.AI; cs.CL; cs.IR**

- **简介: 该论文针对微服务架构中跨仓库的缺陷定位难题，提出将代码转化为分层自然语言摘要，通过两阶段自然语言搜索实现高效定位，提升了大模型在多仓库环境下的缺陷定位效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.05908v1](https://arxiv.org/pdf/2512.05908v1)**

> **作者:** Amirkia Rafiei Oskooei; S. Selcan Yukcu; Mehmet Cevheri Bozoglan; Mehmet S. Aktas
>
> **备注:** Accepted at LLM4Code Workshop, ICSE 2026
>
> **摘要:** Bug localization in multi-repository microservice architectures is challenging due to the semantic gap between natural language bug reports and code, LLM context limitations, and the need to first identify the correct repository. We propose reframing this as a natural language reasoning task by transforming codebases into hierarchical NL summaries and performing NL-to-NL search instead of cross-modal retrieval. Our approach builds context-aware summaries at file, directory, and repository levels, then uses a two-phase search: first routing bug reports to relevant repositories, then performing top-down localization within those repositories. Evaluated on DNext, an industrial system with 46 repositories and 1.1M lines of code, our method achieves Pass@10 of 0.82 and MRR of 0.50, significantly outperforming retrieval baselines and agentic RAG systems like GitHub Copilot and Cursor. This work demonstrates that engineered natural language representations can be more effective than raw source code for scalable bug localization, providing an interpretable repository -> directory -> file search path, which is vital for building trust in enterprise AI tools by providing essential transparency.
>
---
#### [new 035] Vague Knowledge: Information without Transitivity and Partitions
- **分类: econ.TH; cs.CL; math.LO; q-fin.GN**

- **简介: 该论文研究经济模型中知识的模糊性，放松传递性和划分结构假设，形式化非传递性不可区分知识。表明模糊知识虽不划分状态空间但仍具信息性，需通过边界模糊的语言表达，为现实世界中自然语言和定性推理的普遍性提供微观基础。**

- **链接: [https://arxiv.org/pdf/2512.05833v1](https://arxiv.org/pdf/2512.05833v1)**

> **作者:** Kerry Xiao
>
> **摘要:** I relax the standard assumptions of transitivity and partition structure in economic models of information to formalize vague knowledge: non-transitive indistinguishability over states. I show that vague knowledge, while failing to partition the state space, remains informative by distinguishing some states from others. Moreover, it can only be faithfully expressed through vague communication with blurred boundaries. My results provide microfoundations for the prevalence of natural language communication and qualitative reasoning in the real world, where knowledge is often vague.
>
---
#### [new 036] The Effect of Document Summarization on LLM-Based Relevance Judgments
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文研究文档摘要对基于大语言模型（LLM）的相关性判断的影响，属于信息检索评估任务。旨在解决人工标注成本高的问题，探讨使用LLM生成的摘要替代全文进行自动评估的可行性及影响。**

- **链接: [https://arxiv.org/pdf/2512.05334v1](https://arxiv.org/pdf/2512.05334v1)**

> **作者:** Samaneh Mohtadi; Kevin Roitero; Stefano Mizzaro; Gianluca Demartini
>
> **摘要:** Relevance judgments are central to the evaluation of Information Retrieval (IR) systems, but obtaining them from human annotators is costly and time-consuming. Large Language Models (LLMs) have recently been proposed as automated assessors, showing promising alignment with human annotations. Most prior studies have treated documents as fixed units, feeding their full content directly to LLM assessors. We investigate how text summarization affects the reliability of LLM-based judgments and their downstream impact on IR evaluation. Using state-of-the-art LLMs across multiple TREC collections, we compare judgments made from full documents with those based on LLM-generated summaries of different lengths. We examine their agreement with human labels, their effect on retrieval effectiveness evaluation, and their influence on IR systems' ranking stability. Our findings show that summary-based judgments achieve comparable stability in systems' ranking to full-document judgments, while introducing systematic shifts in label distributions and biases that vary by model and dataset. These results highlight summarization as both an opportunity for more efficient large-scale IR evaluation and a methodological choice with important implications for the reliability of automatic judgments.
>
---
#### [new 037] RAG-IGBench: Innovative Evaluation for RAG-based Interleaved Generation in Open-domain Question Answering
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文聚焦检索增强的图文交错生成任务，旨在解决现有基准缺乏多模态评估指标的问题。作者提出RAG-IGBench，构建新数据集并设计综合评测方法，评估文本、图像及一致性，验证了其与人工评价的高度相关性。**

- **链接: [https://arxiv.org/pdf/2512.05119v1](https://arxiv.org/pdf/2512.05119v1)**

> **作者:** Rongyang Zhang; Yuqing Huang; Chengqiang Lu; Qimeng Wang; Yan Gao; Yi Wu; Yao Hu; Yin Xu; Wei Wang; Hao Wang; Enhong Chen
>
> **备注:** 26 pages, 6 figures, NeurIPS 2025 D&B Track poster
>
> **摘要:** In real-world scenarios, providing user queries with visually enhanced responses can considerably benefit understanding and memory, underscoring the great value of interleaved image-text generation. Despite recent progress, like the visual autoregressive model that unifies text and image processing in a single transformer architecture, generating high-quality interleaved content remains challenging. Moreover, evaluations of these interleaved sequences largely remain underexplored, with existing benchmarks often limited by unimodal metrics that inadequately assess the intricacies of combined image-text outputs. To address these issues, we present RAG-IGBench, a thorough benchmark designed specifically to evaluate the task of Interleaved Generation based on Retrieval-Augmented Generation (RAG-IG) in open-domain question answering. RAG-IG integrates multimodal large language models (MLLMs) with retrieval mechanisms, enabling the models to access external image-text information for generating coherent multimodal content. Distinct from previous datasets, RAG-IGBench draws on the latest publicly available content from social platforms and introduces innovative evaluation metrics that measure the quality of text and images, as well as their consistency. Through extensive experiments with state-of-the-art MLLMs (both open-source and proprietary) on RAG-IGBench, we provide an in-depth analysis examining the capabilities and limitations of these models. Additionally, we validate our evaluation metrics by demonstrating their high correlation with human assessments. Models fine-tuned on RAG-IGBench's training set exhibit improved performance across multiple benchmarks, confirming both the quality and practical utility of our dataset. Our benchmark is available at https://github.com/USTC-StarTeam/RAG-IGBench.
>
---
#### [new 038] To Err Is Human: Systematic Quantification of Errors in Published AI Papers via LLM Analysis
- **分类: cs.AI; cs.CL**

- **简介: 该论文旨在量化AI顶会论文中的客观错误。作者构建基于GPT-5的纠错系统，自动检测公式、图表等错误，发现错误率随时间上升，平均精度达83.2%，并能为75.8%的错误提供修正，提升研究可复现性。**

- **链接: [https://arxiv.org/pdf/2512.05925v1](https://arxiv.org/pdf/2512.05925v1)**

> **作者:** Federico Bianchi; Yongchan Kwon; Zachary Izzo; Linjun Zhang; James Zou
>
> **摘要:** How many mistakes do published AI papers contain? Peer-reviewed publications form the foundation upon which new research and knowledge are built. Errors that persist in the literature can propagate unnoticed, creating confusion in follow-up studies and complicating reproducibility. The accelerating pace of research and the increasing demands on the peer-review system make such mistakes harder to detect and avoid. To address this, we developed a Paper Correctness Checker based on GPT-5 to systematically identify mistakes in papers previously published at top AI conferences and journals. Our analysis focuses on objective mistakes-e.g., errors in formulas, derivations, calculations, figures, and tables-that have a clearly verifiable ground truth. We intentionally exclude subjective considerations such as novelty, importance, or writing quality. We find that published papers contain a non-negligible number of objective mistakes and that the average number of mistakes per paper has increased over time-from 3.8 in NeurIPS 2021 to 5.9 in NeurIPS 2025 (55.3% increase); from 4.1 in ICLR 2018 to 5.2 in ICLR 2025; and from 5.0 in TMLR 2022/23 to 5.5 in TMLR 2025. Human experts reviewed 316 potential mistakes identified by the AI Checker and confirmed that 263 were actual mistakes, corresponding to a precision of 83.2%. While most identified issues are relatively minor, correcting them would reduce confusion in the literature and strengthen reproducibility. The AI Checker also surfaced potentially more substantive mistakes that could affect the interpretation of results. Moreover, we show that the AI Checker can propose correct fixes for 75.8% of the identified mistakes. Overall, this study highlights the potential of frontier LLMs to detect and correct objective mistakes in published papers, helping to establish a firmer foundation of knowledge.
>
---
#### [new 039] Active Video Perception: Iterative Evidence Seeking for Agentic Long Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦长视频理解任务，旨在解决现有方法因被动感知导致的计算冗余和信息模糊问题。作者提出Active Video Perception（AVP）框架，通过MLLM代理迭代执行“规划-观察-反思”流程，主动获取与查询相关的时空证据，显著提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2512.05774v1](https://arxiv.org/pdf/2512.05774v1)**

> **作者:** Ziyang Wang; Honglu Zhou; Shijie Wang; Junnan Li; Caiming Xiong; Silvio Savarese; Mohit Bansal; Michael S. Ryoo; Juan Carlos Niebles
>
> **备注:** Website: https://activevideoperception.github.io/
>
> **摘要:** Long video understanding (LVU) is challenging because answering real-world queries often depends on sparse, temporally dispersed cues buried in hours of mostly redundant and irrelevant content. While agentic pipelines improve video reasoning capabilities, prevailing frameworks rely on a query-agnostic captioner to perceive video information, which wastes computation on irrelevant content and blurs fine-grained temporal and spatial information. Motivated by active perception theory, we argue that LVU agents should actively decide what, when, and where to observe, and continuously assess whether the current observation is sufficient to answer the query. We present Active Video Perception (AVP), an evidence-seeking framework that treats the video as an interactive environment and acquires compact, queryrelevant evidence directly from pixels. Concretely, AVP runs an iterative plan-observe-reflect process with MLLM agents. In each round, a planner proposes targeted video interactions, an observer executes them to extract time-stamped evidence, and a reflector evaluates the sufficiency of the evidence for the query, either halting with an answer or triggering further observation. Across five LVU benchmarks, AVP achieves highest performance with significant improvements. Notably, AVP outperforms the best agentic method by 5.7% in average accuracy while only requires 18.4% inference time and 12.4% input tokens.
>
---
#### [new 040] Text Rationalization for Robust Causal Effect Estimation
- **分类: cs.LG; cs.AI; cs.CL; stat.ME; stat.ML**

- **简介: 该论文属于因果推断与文本分析交叉任务，旨在解决高维文本作为混杂因素时因冗余特征导致的 positivity 违背问题。作者提出 CATR 框架，通过残差独立性诊断筛选关键文本标记，在保留混杂信息的同时提升因果效应估计的准确性与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.05373v1](https://arxiv.org/pdf/2512.05373v1)**

> **作者:** Lijinghua Zhang; Hengrui Cai
>
> **摘要:** Recent advances in natural language processing have enabled the increasing use of text data in causal inference, particularly for adjusting confounding factors in treatment effect estimation. Although high-dimensional text can encode rich contextual information, it also poses unique challenges for causal identification and estimation. In particular, the positivity assumption, which requires sufficient treatment overlap across confounder values, is often violated at the observational level, when massive text is represented in feature spaces. Redundant or spurious textual features inflate dimensionality, producing extreme propensity scores, unstable weights, and inflated variance in effect estimates. We address these challenges with Confounding-Aware Token Rationalization (CATR), a framework that selects a sparse necessary subset of tokens using a residual-independence diagnostic designed to preserve confounding information sufficient for unconfoundedness. By discarding irrelevant texts while retaining key signals, CATR mitigates observational-level positivity violations and stabilizes downstream causal effect estimators. Experiments on synthetic data and a real-world study using the MIMIC-III database demonstrate that CATR yields more accurate, stable, and interpretable causal effect estimates than existing baselines.
>
---
#### [new 041] Ontology Learning with LLMs: A Benchmark Study on Axiom Identification
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究利用大语言模型（LLMs）进行本体学习中的公理识别任务，旨在自动识别定义类与属性间逻辑关系的公理。作者构建了OntoAxiom基准，评估多种LLM在不同提示策略下的表现，发现逐条提示效果更优，且模型规模和领域影响性能，虽未实现完全自动化，但可辅助本体工程。**

- **链接: [https://arxiv.org/pdf/2512.05594v1](https://arxiv.org/pdf/2512.05594v1)**

> **作者:** Roos M. Bakker; Daan L. Di Scala; Maaike H. T. de Boer; Stephan A. Raaijmakers
>
> **备注:** Submitted to Semantic Web Journal, under review
>
> **摘要:** Ontologies are an important tool for structuring domain knowledge, but their development is a complex task that requires significant modelling and domain expertise. Ontology learning, aimed at automating this process, has seen advancements in the past decade with the improvement of Natural Language Processing techniques, and especially with the recent growth of Large Language Models (LLMs). This paper investigates the challenge of identifying axioms: fundamental ontology components that define logical relations between classes and properties. In this work, we introduce an Ontology Axiom Benchmark OntoAxiom, and systematically test LLMs on that benchmark for axiom identification, evaluating different prompting strategies, ontologies, and axiom types. The benchmark consists of nine medium-sized ontologies with together 17.118 triples, and 2.771 axioms. We focus on subclass, disjoint, subproperty, domain, and range axioms. To evaluate LLM performance, we compare twelve LLMs with three shot settings and two prompting strategies: a Direct approach where we query all axioms at once, versus an Axiom-by-Axiom (AbA) approach, where each prompt queries for one axiom only. Our findings show that the AbA prompting leads to higher F1 scores than the direct approach. However, performance varies across axioms, suggesting that certain axioms are more challenging to identify. The domain also influences performance: the FOAF ontology achieves a score of 0.642 for the subclass axiom, while the music ontology reaches only 0.218. Larger LLMs outperform smaller ones, but smaller models may still be viable for resource-constrained settings. Although performance overall is not high enough to fully automate axiom identification, LLMs can provide valuable candidate axioms to support ontology engineers with the development and refinement of ontologies.
>
---
#### [new 042] Big Tech-Funded AI Papers Have Higher Citation Impact, Greater Insularity, and Larger Recency Bias
- **分类: cs.DL; cs.AI; cs.CL**

- **简介: 该论文研究大科技公司资助的AI论文影响力。旨在分析行业资助对论文引用、多样性和时效性的影响。通过分析近5万篇论文，揭示行业资助论文引用更高、更封闭且偏重新近研究。**

- **链接: [https://arxiv.org/pdf/2512.05714v1](https://arxiv.org/pdf/2512.05714v1)**

> **作者:** Max Martin Gnewuch; Jan Philip Wahle; Terry Ruas; Bela Gipp
>
> **备注:** Published at IEEE (ACDSA)
>
> **摘要:** Over the past four decades, artificial intelligence (AI) research has flourished at the nexus of academia and industry. However, Big Tech companies have increasingly acquired the edge in computational resources, big data, and talent. So far, it has been largely unclear how many papers the industry funds, how their citation impact compares to non-funded papers, and what drives industry interest. This study fills that gap by quantifying the number of industry-funded papers at 10 top AI conferences (e.g., ICLR, CVPR, AAAI, ACL) and their citation influence. We analyze about 49.8K papers, about 1.8M citations from AI papers to other papers, and about 2.3M citations from other papers to AI papers from 1998-2022 in Scopus. Through seven research questions, we examine the volume and evolution of industry funding in AI research, the citation impact of funded papers, the diversity and temporal range of their citations, and the subfields in which industry predominantly acts. Our findings reveal that industry presence has grown markedly since 2015, from less than 2 percent to more than 11 percent in 2020. Between 2018 and 2022, 12 percent of industry-funded papers achieved high citation rates as measured by the h5-index, compared to 4 percent of non-industry-funded papers and 2 percent of non-funded papers. Top AI conferences engage more with industry-funded research than non-funded research, as measured by our newly proposed metric, the Citation Preference Ratio (CPR). We show that industry-funded research is increasingly insular, citing predominantly other industry-funded papers while referencing fewer non-funded papers. These findings reveal new trends in AI research funding, including a shift towards more industry-funded papers and their growing citation impact, greater insularity of industry-funded work than non-funded work, and a preference of industry-funded research to cite recent work.
>
---
#### [new 043] Entropy Ratio Clipping as a Soft Global Constraint for Stable Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对强化学习中因分布偏移导致的训练不稳定问题，提出熵比剪裁（ERC）机制，通过约束新旧策略的熵比，实现全局分布层面的稳定更新，增强PPO-Clip对未采样动作的调控能力，并在DAPO和GPPO中验证了有效性。**

- **链接: [https://arxiv.org/pdf/2512.05591v1](https://arxiv.org/pdf/2512.05591v1)**

> **作者:** Zhenpeng Su; Leiyu Pan; Minxuan Lv; Tiehua Mei; Zijia Lin; Yuntao Li; Wenping Hu; Ruiming Tang; Kun Gai; Guorui Zhou
>
> **摘要:** Large language model post-training relies on reinforcement learning to improve model capability and alignment quality. However, the off-policy training paradigm introduces distribution shift, which often pushes the policy beyond the trust region, leading to training instabilities manifested as fluctuations in policy entropy and unstable gradients. Although PPO-Clip mitigates this issue through importance clipping, it still overlooks the global distributional shift of actions. To address these challenges, we propose using the entropy ratio between the current and previous policies as a new global metric that effectively quantifies the relative change in policy exploration throughout updates. Building on this metric, we introduce an \textbf{Entropy Ratio Clipping} (ERC) mechanism that imposes bidirectional constraints on the entropy ratio. This stabilizes policy updates at the global distribution level and compensates for the inability of PPO-clip to regulate probability shifts of un-sampled actions. We integrate ERC into both DAPO and GPPO reinforcement learning algorithms. Experiments across multiple benchmarks show that ERC consistently improves performance.
>
---
#### [new 044] Enhancing Retrieval-Augmented Generation with Entity Linking for Educational Platforms
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于教育领域的问答系统任务，旨在解决RAG在专业领域因术语歧义导致的事实错误问题。作者提出结合实体链接的增强RAG架构，通过三种重排序策略提升事实准确性，实验证明其在领域适配和混合排序上的有效性。**

- **链接: [https://arxiv.org/pdf/2512.05967v1](https://arxiv.org/pdf/2512.05967v1)**

> **作者:** Francesco Granata; Francesco Poggi; Misael Mongiovì
>
> **摘要:** In the era of Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) architectures are gaining significant attention for their ability to ground language generation in reliable knowledge sources. Despite their impressive effectiveness in many areas, RAG systems based solely on semantic similarity often fail to ensure factual accuracy in specialized domains, where terminological ambiguity can affect retrieval relevance. This study proposes an enhanced RAG architecture that integrates a factual signal derived from Entity Linking to improve the accuracy of educational question-answering systems in Italian. The system includes a Wikidata-based Entity Linking module and implements three re-ranking strategies to combine semantic and entity-based information: a hybrid score weighting model, reciprocal rank fusion, and a cross-encoder re-ranker. Experiments were conducted on two benchmarks: a custom academic dataset and the standard SQuAD-it dataset. Results show that, in domain-specific contexts, the hybrid schema based on reciprocal rank fusion significantly outperforms both the baseline and the cross-encoder approach, while the cross-encoder achieves the best results on the general-domain dataset. These findings confirm the presence of an effect of domain mismatch and highlight the importance of domain adaptation and hybrid ranking strategies to enhance factual precision and reliability in retrieval-augmented generation. They also demonstrate the potential of entity-aware RAG systems in educational environments, fostering adaptive and reliable AI-based tutoring tools.
>
---
## 更新

#### [replaced 001] Towards Data-efficient Customer Intent Recognition with Prompt-based Learning Paradigm
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理中的客户意图识别任务，旨在解决标注数据不足的问题。作者提出基于提示学习的框架，结合答案映射、主动采样与集成学习，提升小模型在少数据下的性能，并验证其在零样本场景的潜力。**

- **链接: [https://arxiv.org/pdf/2309.14779v2](https://arxiv.org/pdf/2309.14779v2)**

> **作者:** Hengyu Luo; Peng Liu; Stefan Esping
>
> **摘要:** Recognizing customer intent accurately with language models based on customer-agent conversational data is essential in today's digital customer service marketplace, but it is often hindered by the lack of sufficient labeled data. In this paper, we introduce the prompt-based learning paradigm that significantly reduces the dependency on extensive datasets. Utilizing prompted training combined with answer mapping techniques, this approach allows small language models to achieve competitive intent recognition performance with only a minimal amount of training data. Furthermore, We enhance the performance by integrating active sampling and ensemble learning strategies in the prompted training pipeline. Additionally, preliminary tests in a zero-shot setting demonstrate that, with well-crafted and detailed prompts, small language models show considerable instruction-following potential even without any further training. These results highlight the viability of semantic modeling of conversational data in a more data-efficient manner with minimal data use, paving the way for advancements in AI-driven customer service.
>
---
#### [replaced 002] ORFuzz: Fuzzing the "Other Side" of LLM Safety -- Testing Over-Refusal
- **分类: cs.SE; cs.AI; cs.CL; cs.IR**

- **简介: 该论文聚焦LLM过度拒绝问题，提出首个进化测试框架ORFuzz，通过智能种子选择、自适应变异和精准判定模型OR-Judge，系统检测并分析过度拒绝现象，生成高效测试用例集ORFuzzSet，显著提升检测率与基准质量。**

- **链接: [https://arxiv.org/pdf/2508.11222v2](https://arxiv.org/pdf/2508.11222v2)**

> **作者:** Haonan Zhang; Dongxia Wang; Yi Liu; Kexin Chen; Jiashui Wang; Xinlei Ying; Long Liu; Wenhai Wang
>
> **备注:** Accepted by ASE 2025
>
> **摘要:** Large Language Models (LLMs) increasingly exhibit over-refusal - erroneously rejecting benign queries due to overly conservative safety measures - a critical functional flaw that undermines their reliability and usability. Current methods for testing this behavior are demonstrably inadequate, suffering from flawed benchmarks and limited test generation capabilities, as highlighted by our empirical user study. To the best of our knowledge, this paper introduces the first evolutionary testing framework, ORFuzz, for the systematic detection and analysis of LLM over-refusals. ORFuzz uniquely integrates three core components: (1) safety category-aware seed selection for comprehensive test coverage, (2) adaptive mutator optimization using reasoning LLMs to generate effective test cases, and (3) OR-Judge, a human-aligned judge model validated to accurately reflect user perception of toxicity and refusal. Our extensive evaluations demonstrate that ORFuzz generates diverse, validated over-refusal instances at a rate (6.98% average) more than double that of leading baselines, effectively uncovering vulnerabilities. Furthermore, ORFuzz's outputs form the basis of ORFuzzSet, a new benchmark of 1,855 highly transferable test cases that achieves a superior 63.56% average over-refusal rate across 10 diverse LLMs, significantly outperforming existing datasets. ORFuzz and ORFuzzSet provide a robust automated testing framework and a valuable community resource, paving the way for developing more reliable and trustworthy LLM-based software systems.
>
---
#### [replaced 003] IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.RO**

- **简介: 该论文聚焦VLM驱动具身智能体在家庭任务中的交互安全评估。针对现有静态评测无法捕捉动态风险的问题，提出IS-Bench，首个支持多模态、过程导向的交互安全评测基准，包含161个场景与388种安全风险，揭示当前模型缺乏交互安全意识，并推动更安全AI系统发展。**

- **链接: [https://arxiv.org/pdf/2506.16402v3](https://arxiv.org/pdf/2506.16402v3)**

> **作者:** Xiaoya Lu; Zeren Chen; Xuhao Hu; Yijin Zhou; Weichen Zhang; Dongrui Liu; Lu Sheng; Jing Shao
>
> **摘要:** Flawed planning from VLM-driven embodied agents poses significant safety hazards, hindering their deployment in real-world household tasks. However, existing static, non-interactive evaluation paradigms fail to adequately assess risks within these interactive environments, since they cannot simulate dynamic risks that emerge from an agent's actions and rely on unreliable post-hoc evaluations that ignore unsafe intermediate steps. To bridge this critical gap, we propose evaluating an agent's interactive safety: its ability to perceive emergent risks and execute mitigation steps in the correct procedural order. We thus present IS-Bench, the first multi-modal benchmark designed for interactive safety, featuring 161 challenging scenarios with 388 unique safety risks instantiated in a high-fidelity simulator. Crucially, it facilitates a novel process-oriented evaluation that verifies whether risk mitigation actions are performed before/after specific risk-prone steps. Extensive experiments on leading VLMs, including the GPT-4o and Gemini-2.5 series, reveal that current agents lack interactive safety awareness, and that while safety-aware Chain-of-Thought can improve performance, it often compromises task completion. By highlighting these critical limitations, IS-Bench provides a foundation for developing safer and more reliable embodied AI systems. Code and data are released under https://github.com/AI45Lab/IS-Bench.
>
---
#### [replaced 004] Optimizing Fine-Tuning through Advanced Initialization Strategies for Low-Rank Adaptation
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于参数高效微调任务，旨在解决LoRA因初始化为零矩阵导致激活原模型权重能力不足的问题。作者提出IniLoRA，通过更优初始化逼近原权重，并设计两种变体，提升微调性能。**

- **链接: [https://arxiv.org/pdf/2510.03731v3](https://arxiv.org/pdf/2510.03731v3)**

> **作者:** Yongfu Xue
>
> **摘要:** The rapid development of parameter-efficient fine-tuning methods has noticeably improved the efficiency of adapting large language models. Among these, LoRA has gained widespread popularity due to its strong balance of effectiveness and parameter efficiency. However, LoRA relies on initializing two low-rank matrices whose product is zero, which limits its ability to effectively activate and leverage the original model weights-creating a potential bottleneck for optimal performance. To address this limitation, we propose \textbf{IniLoRA}, a novel initialization strategy that initializes the low-rank matrices to closely approximate the original model weights. Experimental results indicate that IniLoRA achieves better performance than LoRA across a range of models and tasks. Additionally, we introduce two variants, IniLoRA-$α$ and IniLoRA-$β$, both leveraging distinct initialization methods to enhance performance further.
>
---
#### [replaced 005] LittleBit: Ultra Low-Bit Quantization via Latent Factorization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型的极低比特量化任务，旨在解决亚1比特下的性能退化问题。提出LittleBit方法，通过潜在因子分解与二值化，结合多尺度补偿机制和Dual-SVID初始化，实现高压缩比下优异性能。**

- **链接: [https://arxiv.org/pdf/2506.13771v3](https://arxiv.org/pdf/2506.13771v3)**

> **作者:** Banseok Lee; Dongkyu Kim; Youngcheon You; Youngmin Kim
>
> **备注:** Accepted to NeurIPS 2025. Banseok Lee and Dongkyu Kim contributed equally
>
> **摘要:** Deploying large language models (LLMs) often faces challenges from substantial memory and computational costs. Quantization offers a solution, yet performance degradation in the sub-1-bit regime remains particularly difficult. This paper introduces LittleBit, a novel method for extreme LLM compression. It targets levels like 0.1 bits per weight (BPW), achieving nearly 31$\times$ memory reduction, e.g., Llama2-13B to under 0.9 GB. LittleBit represents weights in a low-rank form using latent matrix factorization, subsequently binarizing these factors. To counteract information loss from this extreme precision, it integrates a multi-scale compensation mechanism. This includes row, column, and an additional latent dimension that learns per-rank importance. Two key contributions enable effective training: Dual Sign-Value-Independent Decomposition (Dual-SVID) for quantization-aware training (QAT) initialization, and integrated Residual Compensation to mitigate errors. Extensive experiments confirm LittleBit's superiority in sub-1-bit quantization: e.g., its 0.1 BPW performance on Llama2-7B surpasses the leading method's 0.7 BPW. LittleBit establishes a new, viable size-performance trade-off--unlocking a potential 11.6$\times$ speedup over FP16 at the kernel level--and makes powerful LLMs practical for resource-constrained environments. Our code can be found at https://github.com/SamsungLabs/LittleBit.
>
---
#### [replaced 006] Reinforce-Ada: An Adaptive Sampling Framework under Non-linear RL Objectives
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文针对大语言模型在强化学习推理中因采样不足导致的信号丢失问题，提出Reinforce-Ada框架。通过非线性目标引导自适应采样，动态分配计算资源，优先处理难提示，提升学习效率与收敛速度。**

- **链接: [https://arxiv.org/pdf/2510.04996v3](https://arxiv.org/pdf/2510.04996v3)**

> **作者:** Wei Xiong; Chenlu Ye; Baohao Liao; Hanze Dong; Xinxing Xu; Christof Monz; Jiang Bian; Nan Jiang; Tong Zhang
>
> **备注:** 27 pages, 10 figures
>
> **摘要:** Reinforcement learning (RL) for large language model reasoning is frequently hindered by signal loss, a phenomenon where standard uniform sampling with small group sizes fails to uncover informative learning signals for difficult prompts. We demonstrate that this collapse is a statistical artifact of undersampling rather than an inherent model limitation. To address this systematically, we introduce a theoretical framework based on optimizing a non-linear RL objective (e.g., log-likelihood). We show that this objective naturally induces a weighted gradient estimator that prioritizes difficult prompts, which can be robustly realized through adaptive sampling. Guided by this framework, we propose Reinforce-Ada, a family of algorithms that dynamically allocates inference budgets based on prompt difficulty, effectively scaling up RL compute to where it is needed most. Unlike passive filtering methods that discard low-signal prompts, Reinforce-Ada actively invests compute to recover them. We introduce two efficient realizations: an estimation-based approach and a model-free sequential sampling approach. Extensive experiments across multiple benchmarks show that Reinforce-Ada significantly outperforms uniform baselines like GRPO, recovering lost signals and accelerating convergence by up to $2\times$ while maintaining the same total inference budget. Code is available at https://github.com/RLHFlow/Reinforce-Ada.
>
---
#### [replaced 007] SAE-SSV: Supervised Steering in Sparse Representation Spaces for Reliable Control of Language Models
- **分类: cs.CL**

- **简介: 该论文属语言模型控制任务，旨在可靠调控大模型生成行为。提出SAE-SSV方法：利用稀疏自编码器解耦语义属性，训练线性分类器识别关键隐空间维度，并在该子空间学习受监督的导向向量，实现高效、可解释的生成控制。**

- **链接: [https://arxiv.org/pdf/2505.16188v2](https://arxiv.org/pdf/2505.16188v2)**

> **作者:** Zirui He; Mingyu Jin; Bo Shen; Ali Payani; Yongfeng Zhang; Mengnan Du
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Large language models (LLMs) have demonstrated impressive capabilities in natural language understanding and generation, but controlling their behavior reliably remains challenging, especially in open-ended generation settings. This paper introduces a novel supervised steering approach that operates in sparse, interpretable representation spaces. We employ sparse autoencoders (SAEs) to obtain sparse latent representations that aim to disentangle semantic attributes from model activations. Then we train linear classifiers to identify a small subspace of task-relevant dimensions in latent representations. Finally, we learn supervised steering vectors constrained to this subspace, optimized to align with target behaviors. Experiments across sentiment, truthfulness, and political polarity steering tasks with multiple LLMs demonstrate that our supervised steering vectors achieve higher success rates with minimal degradation in generation quality compared to existing methods. Further analysis reveals that a notably small subspace is sufficient for effective steering, enabling more targeted and interpretable interventions. Our implementation is publicly available at https://github.com/Ineedanamehere/SAE-SSV.
>
---
#### [replaced 008] CodeNER: Code Prompting for Named Entity Recognition
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究命名实体识别（NER）任务，旨在解决现有大模型方法仅依赖上下文、缺乏明确标注结构的问题。提出CodeNER，通过代码提示嵌入BIO标注规范，提升模型对标签逻辑的理解，在多语言基准上超越传统文本提示，并结合思维链进一步优化性能。**

- **链接: [https://arxiv.org/pdf/2507.20423v2](https://arxiv.org/pdf/2507.20423v2)**

> **作者:** Sungwoo Han; Jingun Kwon; Hidetaka Kamigaito; Manabu Okumura
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** Recent studies have explored various approaches for treating candidate named entity spans as both source and target sequences in named entity recognition (NER) by leveraging large language models (LLMs). Although previous approaches have successfully generated candidate named entity spans with suitable labels, they rely solely on input context information when using LLMs, particularly, ChatGPT. However, NER inherently requires capturing detailed labeling requirements with input context information. To address this issue, we propose a novel method that leverages code-based prompting to improve the capabilities of LLMs in understanding and performing NER. By embedding code within prompts, we provide detailed BIO schema instructions for labeling, thereby exploiting the ability of LLMs to comprehend long-range scopes in programming languages. Experimental results demonstrate that the proposed code-based prompting method outperforms conventional text-based prompting on ten benchmarks across English, Arabic, Finnish, Danish, and German datasets, indicating the effectiveness of explicitly structuring NER instructions. We also verify that combining the proposed code-based prompting method with the chain-of-thought prompting further improves performance.
>
---
#### [replaced 009] REINA: Regularized Entropy Information-Based Loss for Efficient Simultaneous Speech Translation
- **分类: cs.LG; cs.CL; eess.AS**

- **简介: 该论文研究同步语音翻译（SimulST），旨在平衡翻译质量与延迟。提出基于信息熵的正则化损失REINA，指导模型仅在获取新信息时等待输入，优化权衡。基于开源或合成数据训练，实现多语言SOTA性能，并提升流式效率达21%。**

- **链接: [https://arxiv.org/pdf/2508.04946v3](https://arxiv.org/pdf/2508.04946v3)**

> **作者:** Nameer Hirschkind; Joseph Liu; Xiao Yu; Mahesh Kumar Nandwana
>
> **备注:** Accepted to AAAI 2026 (Oral Track)
>
> **摘要:** Simultaneous Speech Translation (SimulST) systems stream in audio while simultaneously emitting translated text or speech. Such systems face the significant challenge of balancing translation quality and latency. We introduce a strategy to optimize this tradeoff: wait for more input only if you gain information by doing so. Based on this strategy, we present Regularized Entropy INformation Adaptation (REINA), a novel loss to train an adaptive policy using an existing non-streaming translation model. We derive REINA from information theory principles and show that REINA helps push the reported Pareto frontier of the latency/quality tradeoff over prior works. Utilizing REINA, we train a SimulST model on French, Spanish and German, both from and into English. Training on only open source or synthetically generated data, we achieve state-of-the-art (SOTA) streaming results for models of comparable size. We also introduce a metric for streaming efficiency, quantitatively showing REINA improves the latency/quality trade-off by as much as 21% compared to prior approaches, normalized against non-streaming baseline BLEU scores.
>
---
#### [replaced 010] From Simulation to Strategy: Automating Personalized Interaction Planning for Conversational Agents
- **分类: cs.CL**

- **简介: 该论文研究面向销售的对话系统，旨在通过用户模拟器优化对话策略。基于用户职业等画像信息，提出轻量级的职业条件策略，提升对话效率与成功率，实现个性化交互规划。**

- **链接: [https://arxiv.org/pdf/2510.08621v2](https://arxiv.org/pdf/2510.08621v2)**

> **作者:** Wen-Yu Chang; Tzu-Hung Huang; Chih-Ho Chen; Yun-Nung Chen
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** Amid the rapid rise of agentic dialogue models, realistic user-simulator studies are essential for tuning effective conversation strategies. This work investigates a sales-oriented agent that adapts its dialogue based on user profiles spanning age, gender, and occupation. While age and gender influence overall performance, occupation produces the most pronounced differences in conversational intent. Leveraging this insight, we introduce a lightweight, occupation-conditioned strategy that guides the agent to prioritize intents aligned with user preferences, resulting in shorter and more successful dialogues. Our findings highlight the importance of rich simulator profiles and demonstrate how simple persona-informed strategies can enhance the effectiveness of sales-oriented dialogue systems.
>
---
#### [replaced 011] HARP: Hallucination Detection via Reasoning Subspace Projection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属幻觉检测任务，旨在解决大语言模型在关键决策中产生幻觉的问题。提出HARP框架，通过分解Unembedding层的语义与推理子空间，利用推理子空间投影降维并提升检测鲁棒性，实现SOTA性能。**

- **链接: [https://arxiv.org/pdf/2509.11536v2](https://arxiv.org/pdf/2509.11536v2)**

> **作者:** Junjie Hu; Gang Tu; ShengYu Cheng; Jinxin Li; Jinting Wang; Rui Chen; Zhilong Zhou; Dongbo Shan
>
> **摘要:** Hallucinations in Large Language Models (LLMs) pose a major barrier to their reliable use in critical decision-making. Although existing hallucination detection methods have improved accuracy, they still struggle with disentangling semantic and reasoning information and maintaining robustness. To address these challenges, we propose HARP (Hallucination detection via reasoning subspace projection), a novel hallucination detection framework. HARP establishes that the hidden state space of LLMs can be decomposed into a direct sum of a semantic subspace and a reasoning subspace, where the former encodes linguistic expression and the latter captures internal reasoning processes. Moreover, we demonstrate that the Unembedding layer can disentangle these subspaces, and by applying Singular Value Decomposition (SVD) to its parameters, the basis vectors spanning the semantic and reasoning subspaces are obtained. Finally, HARP projects hidden states onto the basis vectors of the reasoning subspace, and the resulting projections are then used as input features for hallucination detection in LLMs. By using these projections, HARP reduces the dimension of the feature to approximately 5% of the original, filters out most noise, and achieves enhanced robustness. Experiments across multiple datasets show that HARP achieves state-of-the-art hallucination detection performance; in particular, it achieves an AUROC of 92.8% on TriviaQA, outperforming the previous best method by 7.5%.
>
---
#### [replaced 012] OpenMMReasoner: Pushing the Frontiers for Multimodal Reasoning with an Open and General Recipe
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦多模态推理任务，旨在解决现有方法在数据构建与训练策略上缺乏透明性和可复现性的问题。作者提出OpenMMReasoner，包含两阶段开源训练方案：基于874K样本的监督微调与74K样本的强化学习，显著提升推理性能，并推动可复现研究。**

- **链接: [https://arxiv.org/pdf/2511.16334v4](https://arxiv.org/pdf/2511.16334v4)**

> **作者:** Kaichen Zhang; Keming Wu; Zuhao Yang; Bo Li; Kairui Hu; Bin Wang; Ziwei Liu; Xingxuan Li; Lidong Bing
>
> **摘要:** Recent advancements in large reasoning models have fueled growing interest in extending such capabilities to multimodal domains. However, despite notable progress in visual reasoning, the lack of transparent and reproducible data curation and training strategies remains a major barrier to scalable research. In this work, we introduce OpenMMReasoner, a fully transparent two-stage recipe for multimodal reasoning spanning supervised fine-tuning (SFT) and reinforcement learning (RL). In the SFT stage, we construct an 874K-sample cold-start dataset with rigorous step-by-step validation, providing a strong foundation for reasoning capabilities. The subsequent RL stage leverages a 74K-sample dataset across diverse domains to further sharpen and stabilize these abilities, resulting in a more robust and efficient learning process. Extensive evaluations demonstrate that our training recipe not only surpasses strong baselines but also highlights the critical role of data quality and training design in shaping multimodal reasoning performance. Notably, our method achieves a 11.6% improvement over the Qwen2.5-VL-7B-Instruct baseline across nine multimodal reasoning benchmarks, establishing a solid empirical foundation for future large-scale multimodal reasoning research. We open-sourced all our codes, pipeline, and data at https://github.com/EvolvingLMMs-Lab/OpenMMReasoner.
>
---
#### [replaced 013] Fair Text Classification via Transferable Representations
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于文本分类任务，旨在解决敏感属性带来的群体不公平问题。通过扩展Wasserstein依赖度量，结合对抗训练和领域自适应，实现无需敏感属性即可学习公平的文本表示。**

- **链接: [https://arxiv.org/pdf/2503.07691v2](https://arxiv.org/pdf/2503.07691v2)**

> **作者:** Thibaud Leteno; Michael Perrot; Charlotte Laclau; Antoine Gourru; Christophe Gravier
>
> **备注:** arXiv admin note: text overlap with arXiv:2311.12689
>
> **摘要:** Group fairness is a central research topic in text classification, where reaching fair treatment between sensitive groups (e.g., women and men) remains an open challenge. We propose an approach that extends the use of the Wasserstein Dependency Measure for learning unbiased neural text classifiers. Given the challenge of distinguishing fair from unfair information in a text encoder, we draw inspiration from adversarial training by inducing independence between representations learned for the target label and those for a sensitive attribute. We further show that Domain Adaptation can be efficiently leveraged to remove the need for access to the sensitive attributes in the dataset we cure. We provide both theoretical and empirical evidence that our approach is well-founded.
>
---
#### [replaced 014] Experiments with Large Language Models on Retrieval-Augmented Generation for Closed-Source Simulation Software
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究检索增强生成（RAG）在闭源仿真软件中的应用，旨在缓解大语言模型的幻觉问题。通过本地小模型实验，探索提升响应质量的方法，验证了RAG在保护数据安全前提下的潜力与挑战。**

- **链接: [https://arxiv.org/pdf/2502.03916v2](https://arxiv.org/pdf/2502.03916v2)**

> **作者:** Andreas Baumann; Peter Eberhard
>
> **备注:** 16 pages, 6 tables, 2 figures
>
> **摘要:** Large Language Models (LLMs) are tools that have become indispensable in development and programming. However, they suffer from hallucinations, especially when dealing with unknown knowledge. This is particularly the case when LLMs are to be used to support closed-source software applications. Retrieval-Augmented Generation (RAG) offers an approach to use additional knowledge alongside the pre-trained knowledge of the LLM to respond to user prompts. Possible tasks range from a smart-autocomplete, text extraction for question answering, model summarization, component explaining, compositional reasoning, to creation of simulation components and complete input models. This work tests existing RAG systems for closed-source simulation frameworks, in our case the mesh-free simulation software Pasimodo. Since data protection and intellectual property rights are particularly important for problems solved with closed-source software, the tests focus on execution using local LLMs. In order to enable smaller institutions to use the systems, smaller language models will be tested first. The systems show impressive results, but often fail due to insufficient information. Different approaches for improving response quality are tested. In particular, tailoring the information provided to the LLMs dependent to the prompts proves to be a significant improvement. This demonstrates the great potential and the further work needed to improve information retrieval for closed-source simulation models.
>
---
#### [replaced 015] Pet-Bench: Benchmarking the Abilities of Large Language Models as E-Pets in Social Network Services
- **分类: cs.CL**

- **简介: 该论文属于AI评测任务，旨在解决大语言模型在社交网络中作为电子宠物的陪伴能力缺乏系统评估的问题。作者提出Pet-Bench基准，涵盖自交互与人交互维度，设计7500+实例评估28个模型，推动情感化人机互动发展。**

- **链接: [https://arxiv.org/pdf/2506.03761v2](https://arxiv.org/pdf/2506.03761v2)**

> **作者:** Hongcheng Guo; Zheyong Xie; Shaosheng Cao; Boyang Wang; Weiting Liu; Zheyu Ye; Zhoujun Li; Zuozhu Liu; Wei Liu
>
> **摘要:** As interest in using Large Language Models for interactive and emotionally rich experiences grows, virtual pet companionship emerges as a novel yet underexplored application. Existing approaches focus on basic pet role-playing interactions without systematically benchmarking LLMs for comprehensive companionship. In this paper, we introduce Pet-Bench, a dedicated benchmark that evaluates LLMs across both self-interaction and human-interaction dimensions. Unlike prior work, Pet-Bench emphasizes self-evolution and developmental behaviors alongside interactive engagement, offering a more realistic reflection of pet companionship. It features diverse tasks such as intelligent scheduling, memory-based dialogues, and psychological conversations, with over 7,500 interaction instances designed to simulate pet behaviors. Evaluation of 28 LLMs reveals significant performance variations linked to model size and inherent capabilities, underscoring the need for specialized optimization in this domain. Pet-Bench serves as a foundational resource for benchmarking pet-related LLM abilities and advancing emotionally immersive human-pet interactions.
>
---
#### [replaced 016] Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match
- **分类: cs.CL**

- **简介: 该论文属推理加速任务，旨在解决传统推测解码因严格匹配丢弃语义正确草案的问题。提出无需训练的宽松推测解码FLy，利用目标模型自校正能力判断草案语义有效性，提升生成效率且保持高准确率。**

- **链接: [https://arxiv.org/pdf/2511.22972v2](https://arxiv.org/pdf/2511.22972v2)**

> **作者:** Jinze Li; Yixing Xu; Guanchen Li; Shuo Yang; Jinfeng Xu; Xuanwu Yin; Dong Li; Edith C. H. Ngai; Emad Barsoum
>
> **备注:** Under review
>
> **摘要:** Large language models (LLMs) achieve strong performance across diverse tasks but suffer from high inference latency due to their autoregressive generation. Speculative Decoding (SPD) mitigates this issue by verifying candidate tokens in parallel from a smaller draft model, yet its strict exact-match verification discards many semantically valid continuations. Moreover, existing training-based SPD methods often suffer from performance degradation on out-of-distribution (OOD) tasks. To this end, we propose Training-Free Loosely Speculative Decoding (FLy), a novel method that loosens the rigid verification criterion by leveraging the target model's self-corrective behavior to judge whether a draft-target mismatch remains semantically valid. FLy introduces a two-tier mechanism: an entropy-level gate that identifies whether the current token allows multiple plausible alternatives or is nearly deterministic, and a token-level deferred window that distinguishes genuine errors from differently worded yet semantically correct variants. To further reduce latency, we design a multi-level acceleration strategy that accelerates not only the target model but also the drafter itself. Owing to its training-free design, FLy composes seamlessly with arbitrary draft-target pairs and generalizes across models and domains without hyperparameter re-tuning. Experiments show that FLy preserves more than 99% of the target model's accuracy while achieving an average 2.81x speedup on Llama-3.1-70B-Instruct and 5.07x speedup on the 405B variant. Notably, on out-of-domain datasets, our method remains highly effective and outperforms the training-based method EAGLE-3 by 1.62x.
>
---
#### [replaced 017] Chinese Discharge Drug Recommendation in Metabolic Diseases with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究基于大语言模型的中文出院药物推荐任务，旨在提升代谢性疾病临床决策效率。针对中文临床用药推荐研究不足的问题，系统评估了多种LLM及提示方法，分析其性能与局限性。**

- **链接: [https://arxiv.org/pdf/2510.21084v2](https://arxiv.org/pdf/2510.21084v2)**

> **作者:** Juntao Li; Haobin Yuan; Ling Luo; Yan Jiang; Fan Wang; Ping Zhang; Huiyi Lv; Jian Wang; Yuanyuan Sun; Hongfei Lin
>
> **摘要:** Intelligent drug recommendation based on Electronic Health Records (EHRs) is critical for improving the quality and efficiency of clinical decision-making. By leveraging large-scale patient data, drug recommendation systems can assist physicians in selecting the most appropriate medications according to a patient's medical history, diagnoses, laboratory results, and comorbidities. Recent advances in large language models (LLMs) have shown remarkable capabilities in complex reasoning and medical text understanding, making them promising tools for drug recommendation tasks. However, the application of LLMs for Chinese clinical medication recommendation remains largely unexplored. In this work, we conduct a systematic investigation of LLM-based methodologies for Chinese discharge medication recommendation. We evaluate several representative LLM families (GLM, Llama, Qwen) under a unified methodological framework including zero-shot prompting, in-context learning, chain-of-thought prompting, and supervised fine-tuning using LoRA. We analyze model behavior across reasoning styles, error patterns, domain adaptation mechanisms, and robustness. Experimental results show that while supervised fine-tuning improves model performance, there remains substantial room for improvement, with the best model achieving the F1 score of 0.5648 and Jaccard score of 0.4477. Our findings highlight both the potential and limitations of LLMs for Chinese drug recommendation.
>
---
#### [replaced 018] Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究稀疏自编码器（SAE）中L0超参数对特征提取的影响，指出错误的L0会导致特征混淆。作者提出应正确设置L0以解耦LLM内部特征，并设计代理指标指导L0选择，发现常用SAE的L0普遍偏低。**

- **链接: [https://arxiv.org/pdf/2508.16560v3](https://arxiv.org/pdf/2508.16560v3)**

> **作者:** David Chanin; Adrià Garriga-Alonso
>
> **摘要:** Sparse Autoencoders (SAEs) extract features from LLM internal activations, meant to correspond to interpretable concepts. A core SAE training hyperparameter is L0: how many SAE features should fire per token on average. Existing work compares SAE algorithms using sparsity-reconstruction tradeoff plots, implying L0 is a free parameter with no single correct value aside from its effect on reconstruction. In this work we study the effect of L0 on SAEs, and show that if L0 is not set correctly, the SAE fails to disentangle the underlying features of the LLM. If L0 is too low, the SAE will mix correlated features to improve reconstruction. If L0 is too high, the SAE finds degenerate solutions that also mix features. Further, we present a proxy metric that can help guide the search for the correct L0 for an SAE on a given training distribution. We show that our method finds the correct L0 in toy models and coincides with peak sparse probing performance in LLM SAEs. We find that most commonly used SAEs have an L0 that is too low. Our work shows that L0 must be set correctly to train SAEs with correct features.
>
---
#### [replaced 019] Decoding inner speech with an end-to-end brain-to-text neural interface
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于脑机接口中的语音解码任务，旨在解决传统级联框架无法端到端优化的问题。作者提出名为BIT的端到端神经网络框架，通过跨任务预训练编码器和音频大语言模型结合对比学习，显著降低词错误率，并实现尝试与想象语音的统一表征。**

- **链接: [https://arxiv.org/pdf/2511.21740v2](https://arxiv.org/pdf/2511.21740v2)**

> **作者:** Yizi Zhang; Linyang He; Chaofei Fan; Tingkai Liu; Han Yu; Trung Le; Jingyuan Li; Scott Linderman; Lea Duncker; Francis R Willett; Nima Mesgarani; Liam Paninski
>
> **摘要:** Speech brain-computer interfaces (BCIs) aim to restore communication for people with paralysis by translating neural activity into text. Most systems use cascaded frameworks that decode phonemes before assembling sentences with an n-gram language model (LM), preventing joint optimization of all stages simultaneously. Here, we introduce an end-to-end Brain-to-Text (BIT) framework that translates neural activity into coherent sentences using a single differentiable neural network. Central to our approach is a cross-task, cross-species pretrained neural encoder, whose representations transfer to both attempted and imagined speech. In a cascaded setting with an n-gram LM, the pretrained encoder establishes a new state-of-the-art (SOTA) on the Brain-to-Text '24 and '25 benchmarks. Integrated end-to-end with audio large language models (LLMs) and trained with contrastive learning for cross-modal alignment, BIT reduces the word error rate (WER) of the prior end-to-end method from 24.69% to 10.22%. Notably, we find that small-scale audio LLMs markedly improve end-to-end decoding. Beyond record-setting performance, BIT aligns attempted and imagined speech embeddings to enable cross-task generalization. Altogether, our approach advances the integration of large, diverse neural datasets, paving the way for an end-to-end decoding framework that supports seamless, differentiable optimization.
>
---
#### [replaced 020] AURA: A Diagnostic Framework for Tracking User Satisfaction of Interactive Planning Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AURA框架，旨在诊断交互式任务规划智能体的用户满意度。针对现有基准仅关注任务完成率的问题，AURA通过细粒度评估标准分析智能体在各行为阶段的表现，揭示用户满意度受结果与过程共同影响，帮助识别智能体优缺点。**

- **链接: [https://arxiv.org/pdf/2505.01592v2](https://arxiv.org/pdf/2505.01592v2)**

> **作者:** Takyoung Kim; Janvijay Singh; Shuhaib Mehri; Emre Can Acikgoz; Sagnik Mukherjee; Nimet Beyza Bozdag; Sumuk Shashidhar; Gokhan Tur; Dilek Hakkani-Tür
>
> **备注:** NeurIPS 2025 MTI-LLM Workshop. Full version is under review
>
> **摘要:** The growing capabilities of large language models (LLMs) in instruction-following and context-understanding lead to the era of agents with numerous applications. Among these, task planning agents have become especially prominent in realistic scenarios involving complex internal pipelines, such as context understanding, tool management, and response generation. However, existing benchmarks predominantly evaluate agent performance based on task completion as a proxy for overall effectiveness. We hypothesize that merely improving task completion is misaligned with maximizing user satisfaction, as users interact with the entire agentic process and not only the end result. To address this gap, we propose AURA, an Agent-User inteRaction Assessment framework that conceptualizes the behavioral stages of interactive task planning agents. AURA offers a comprehensive assessment of agent through a set of atomic LLM evaluation criteria, allowing researchers and practitioners to diagnose specific strengths and weaknesses within the agent's decision-making pipeline. Our analyses show that agents excel in different behavioral stages, with user satisfaction shaped by both outcomes and intermediate behaviors. We also highlight future directions, including systems that leverage multiple agents and the limitations of user simulators in task planning.
>
---
#### [replaced 021] A quantitative analysis of semantic information in deep representations of text and images
- **分类: cs.CL; cs.LG; physics.comp-ph**

- **简介: 该论文研究深度模型中文本与图像语义表示的量化分析，旨在揭示跨模态和跨语言的语义信息编码机制。通过分析大模型多层表示，定位富含语义的层级，比较不同规模模型的信息提取能力，并探究图文间的信息不对称性。**

- **链接: [https://arxiv.org/pdf/2505.17101v4](https://arxiv.org/pdf/2505.17101v4)**

> **作者:** Santiago Acevedo; Andrea Mascaretti; Riccardo Rende; Matéo Mahaut; Marco Baroni; Alessandro Laio
>
> **摘要:** Deep neural networks are known to develop similar representations for semantically related data, even when they belong to different domains, such as an image and its description, or the same text in different languages. We present a method for quantitatively investigating this phenomenon by measuring the relative information content of the representations of semantically related data and probing how it is encoded into multiple tokens of large language models (LLMs) and vision transformers. Looking first at how LLMs process pairs of translated sentences, we identify inner ``semantic'' layers containing the most language-transferable information. We find moreover that, on these layers, a larger LLM (DeepSeek-V3) extracts significantly more general information than a smaller one (Llama3.1-8B). Semantic information of English text is spread across many tokens and it is characterized by long-distance correlations between tokens and by a causal left-to-right (i.e., past-future) asymmetry. We also identify layers encoding semantic information within visual transformers. We show that caption representations in the semantic layers of LLMs predict visual representations of the corresponding images. We observe significant and model-dependent information asymmetries between image and text representations.
>
---
#### [replaced 022] HalluClean: A Unified Framework to Combat Hallucinations in LLMs
- **分类: cs.CL**

- **简介: 该论文针对大语言模型易产生事实性错误（幻觉）的问题，提出HalluClean框架，通过推理增强的三阶段流程，在无需外部知识或监督的情况下实现跨任务的幻觉检测与修正，提升生成内容的事实一致性。**

- **链接: [https://arxiv.org/pdf/2511.08916v4](https://arxiv.org/pdf/2511.08916v4)**

> **作者:** Yaxin Zhao; Yu Zhang
>
> **摘要:** Large language models (LLMs) have achieved impressive performance across a wide range of natural language processing tasks, yet they often produce hallucinated content that undermines factual reliability. To address this challenge, we introduce HalluClean, a lightweight and task-agnostic framework for detecting and correcting hallucinations in LLM-generated text. HalluClean adopts a reasoning-enhanced paradigm, explicitly decomposing the process into planning, execution, and revision stages to identify and refine unsupported claims. It employs minimal task-routing prompts to enable zero-shot generalization across diverse domains, without relying on external knowledge sources or supervised detectors. We conduct extensive evaluations on five representative tasks-question answering, dialogue, summarization, math word problems, and contradiction detection. Experimental results show that HalluClean significantly improves factual consistency and outperforms competitive baselines, demonstrating its potential to enhance the trustworthiness of LLM outputs in real-world applications.
>
---
#### [replaced 023] Characterizing Language Use in a Collaborative Situated Game
- **分类: cs.CL**

- **简介: 该论文聚焦协作式情境游戏中的语言使用，旨在分析复杂环境中人类对话特征。研究采集并发布了《Portal 2》合作模式下的口语语料库（11.5小时，24.5K语句），包含多模态数据与标注，揭示了空间指代、会话修复等独特语言现象，服务于未来协作对话系统研究。**

- **链接: [https://arxiv.org/pdf/2512.03381v2](https://arxiv.org/pdf/2512.03381v2)**

> **作者:** Nicholas Tomlin; Naitian Zhou; Eve Fleisig; Liangyuan Chen; Téa Wright; Lauren Vinh; Laura X. Ma; Seun Eisape; Ellie French; Tingting Du; Tianjiao Zhang; Alexander Koller; Alane Suhr
>
> **摘要:** Cooperative video games, where multiple participants must coordinate by communicating and reasoning under uncertainty in complex environments, yield a rich source of language data. We collect the Portal Dialogue Corpus: a corpus of 11.5 hours of spoken human dialogue in the co-op mode of the popular Portal 2 virtual puzzle game, comprising 24.5K total utterances. We analyze player language and behavior, identifying a number of linguistic phenomena that rarely appear in most existing chitchat or task-oriented dialogue corpora, including complex spatial reference, clarification and repair, and ad-hoc convention formation. To support future analyses of language use in complex, situated, collaborative problem-solving scenarios, we publicly release the corpus, which comprises player videos, audio, transcripts, game state data, and both manual and automatic annotations of language data.
>
---
#### [replaced 024] MindEval: Benchmarking Language Models on Multi-turn Mental Health Support
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI心理健康支持任务，旨在解决现有评测基准无法反映真实多轮治疗对话复杂性的问题。作者提出MindEval框架，联合临床心理学家设计，通过模拟患者和自动评估，实现对大模型在多轮心理对话中表现的可靠评测，并发现当前模型普遍存在缺陷。**

- **链接: [https://arxiv.org/pdf/2511.18491v3](https://arxiv.org/pdf/2511.18491v3)**

> **作者:** José Pombal; Maya D'Eon; Nuno M. Guerreiro; Pedro Henrique Martins; António Farinhas; Ricardo Rei
>
> **摘要:** Demand for mental health support through AI chatbots is surging, though current systems present several limitations, like sycophancy or overvalidation, and reinforcement of maladaptive beliefs. A core obstacle to the creation of better systems is the scarcity of benchmarks that capture the complexity of real therapeutic interactions. Most existing benchmarks either only test clinical knowledge through multiple-choice questions or assess single responses in isolation. To bridge this gap, we present MindEval, a framework designed in collaboration with Ph.D-level Licensed Clinical Psychologists for automatically evaluating language models in realistic, multi-turn mental health therapy conversations. Through patient simulation and automatic evaluation with LLMs, our framework balances resistance to gaming with reproducibility via its fully automated, model-agnostic design. We begin by quantitatively validating the realism of our simulated patients against human-generated text and by demonstrating strong correlations between automatic and human expert judgments. Then, we evaluate 12 state-of-the-art LLMs and show that all models struggle, scoring below 4 out of 6, on average, with particular weaknesses in problematic AI-specific patterns of communication. Notably, reasoning capabilities and model scale do not guarantee better performance, and systems deteriorate with longer interactions or when supporting patients with severe symptoms. We release all code, prompts, and human evaluation data.
>
---
#### [replaced 025] LAET: A Layer-wise Adaptive Ensemble Tuning Framework for Pretrained Language Models
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: 该论文提出LAET框架，针对金融NLP任务中大模型计算开销高的问题，通过分析隐藏状态选择性微调关键层，冻结其余层，在降低计算成本的同时提升小模型性能，实现高效部署。**

- **链接: [https://arxiv.org/pdf/2511.11315v2](https://arxiv.org/pdf/2511.11315v2)**

> **作者:** Jawad Ibn Ahad; Muhammad Rafsan Kabir; Robin Krambroeckers; Sifat Momen; Nabeel Mohammed; Shafin Rahman
>
> **摘要:** Natural Language Processing (NLP) has transformed the financial industry, enabling advancements in areas such as textual analysis, risk management, and forecasting. Large language models (LLMs) like BloombergGPT and FinMA have set new benchmarks across various financial NLP tasks, including sentiment analysis, stock movement prediction, and credit risk assessment. Furthermore, FinMA-ES, a bilingual financial LLM, has also demonstrated strong performance using the FLARE and FLARE-ES benchmarks. However, the high computational demands of these models limit the accessibility of many organizations. To address this, we propose Layer-wise Adaptive Ensemble Tuning (LAET), a novel strategy that selectively fine-tunes the most effective layers of pre-trained LLMs by analyzing hidden state representations while freezing less critical layers. LAET significantly reduces computational overhead while enhancing task-specific performance. Our approach shows strong results in financial NLP tasks, outperforming existing benchmarks and state-of-the-art LLMs such as GPT-4, even with smaller LLMs ($\sim$3B parameters). This work bridges cutting-edge financial NLP research and real-world deployment with efficient and scalable models for financial applications.
>
---
#### [replaced 026] Analysing Moral Bias in Finetuned LLMs through Mechanistic Interpretability
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究微调大模型中的道德偏差（Knobe效应），通过机械可解释性方法定位其在模型层中的来源。实验表明，仅修补少数关键层即可消除偏差，证明社会偏见可被定位并干预，无需重新训练。**

- **链接: [https://arxiv.org/pdf/2510.12229v2](https://arxiv.org/pdf/2510.12229v2)**

> **作者:** Bianca Raimondi; Daniela Dalbagno; Maurizio Gabbrielli
>
> **备注:** Preprint. Under review
>
> **摘要:** Large language models (LLMs) have been shown to internalize human-like biases during finetuning, yet the mechanisms by which these biases manifest remain unclear. In this work, we investigated whether the well-known Knobe effect, a moral bias in intentionality judgements, emerges in finetuned LLMs and whether it can be traced back to specific components of the model. We conducted a Layer-Patching analysis across 3 open-weights LLMs and demonstrated that the bias is not only learned during finetuning but also localized in a specific set of layers. Surprisingly, we found that patching activations from the corresponding pretrained model into just a few critical layers is sufficient to eliminate the effect. Our findings offer new evidence that social biases in LLMs can be interpreted, localized, and mitigated through targeted interventions, without the need for model retraining.
>
---
#### [replaced 027] Hierarchical Mamba Meets Hyperbolic Geometry: A New Paradigm for Structured Language Embeddings
- **分类: cs.CL; cs.LG**

- **简介: 该论文聚焦语言表示中的层次结构建模问题，提出将Mamba2与双曲几何结合的Hierarchical Mamba（HiM），通过可学习曲率在Poincaré球或Lorentz流形中学习层次感知的语言嵌入，提升长程推理与多跳分类性能。**

- **链接: [https://arxiv.org/pdf/2505.18973v3](https://arxiv.org/pdf/2505.18973v3)**

> **作者:** Sarang Patil; Ashish Parmanand Pandey; Ioannis Koutis; Mengjia Xu
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Selective state-space models excel at long-sequence modeling, but their capacity for language representation -- in complex hierarchical reasoning -- remains underexplored. Most large language models rely on \textit{flat} Euclidean embeddings, limiting their ability to capture latent hierarchies. To address this, we propose {\it Hierarchical Mamba (HiM)}, integrating efficient Mamba2 with hyperbolic geometry to learn hierarchy-aware language embeddings for deeper linguistic understanding. Mamba2-processed sequences are projected to the Poincaré ball or Lorentzian manifold with ``learnable'' curvature, optimized with a hyperbolic loss. Our HiM model facilitates the capture of relational distances across varying hierarchical levels, enabling effective long-range reasoning for tasks like mixed-hop prediction and multi-hop inference in hierarchical classification. Experimental results show both HiM variants effectively capture hierarchical relationships across four linguistic and medical datasets, surpassing Euclidean baselines, with HiM-Poincaré providing fine-grained distinctions with higher h-norms, while HiM-Lorentz offers more stable, compact, and hierarchy-preserving embeddings-favoring robustness. The source code is publicly available at https://github.com/BerryByte/HiM.
>
---
#### [replaced 028] Vision-centric Token Compression in Large Language Model
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于大语言模型高效推理任务，旨在解决长上下文带来的高计算与内存开销问题。提出Vision-centric Token Compression（Vist），通过慢-快路径将低显著性上下文转为图像压缩表示，提升效率。**

- **链接: [https://arxiv.org/pdf/2502.00791v4](https://arxiv.org/pdf/2502.00791v4)**

> **作者:** Ling Xing; Alex Jinpeng Wang; Rui Yan; Xiangbo Shu; Jinhui Tang
>
> **备注:** NeurIPS 2025 spotlight
>
> **摘要:** Real-world applications are stretching context windows to hundreds of thousand of tokens while Large Language Models (LLMs) swell from billions to trillions of parameters. This dual expansion send compute and memory costs skyrocketing, making token compression indispensable. We introduce Vision Centric Token Compression (Vist), a slow-fast compression framework that mirrors human reading: the fast path renders distant tokens into images, letting a frozen, lightweight vision encoder skim the low-salience context; the slow path feeds the proximal window into the LLM for fine-grained reasoning. A Probability-Informed Visual Enhancement (PVE) objective masks high-frequency tokens during training, steering the Resampler to concentrate on semantically rich regions-just as skilled reader gloss over function words. On eleven in-context learning benchmarks, Vist achieves the same accuracy with 2.3 times fewer tokens, cutting FLOPs by 16% and memory by 50%. This method delivers remarkable results, outperforming the strongest text encoder-based compression method CEPE by 7.6% on average over benchmarks like TriviaQA, NQ, PopQA, NLUI, and CLIN, setting a new standard for token efficiency in LLMs. The project is at https://github.com/CSU-JPG/VIST.
>
---
#### [replaced 029] The AI Productivity Index (APEX)
- **分类: econ.GN; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于AI评估任务，旨在衡量前沿模型在投资银行、管理咨询等四个专业岗位中完成经济价值任务的能力。通过扩展测试集和改进评分方法，发布APEX-v1-extended基准与新排行榜，并开源部分案例与评测工具。**

- **链接: [https://arxiv.org/pdf/2509.25721v4](https://arxiv.org/pdf/2509.25721v4)**

> **作者:** Bertie Vidgen; Abby Fennelly; Evan Pinnix; Julien Benchek; Daniyal Khan; Zach Richards; Austin Bridges; Calix Huang; Ben Hunsberger; Isaac Robinson; Akul Datta; Chirag Mahapatra; Dominic Barton; Cass R. Sunstein; Eric Topol; Brendan Foody; Osvald Nitski
>
> **摘要:** We present an extended version of the AI Productivity Index (APEX-v1-extended), a benchmark for assessing whether frontier models are capable of performing economically valuable tasks in four jobs: investment banking associate, management consultant, big law associate, and primary care physician (MD). This technical report details the extensions to APEX-v1, including an increase in the held-out evaluation set from n = 50 to n = 100 cases per job (n = 400 total) and updates to the grading methodology. We present a new leaderboard, where GPT5 (Thinking = High) remains the top performing model with a score of 67.0%. APEX-v1-extended shows that frontier models still have substantial limitations when performing typical professional tasks. To support further research, we are open sourcing n = 25 non-benchmark example cases per role (n = 100 total) along with our evaluation harness.
>
---
#### [replaced 030] A Survey on Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于综述任务，旨在系统梳理扩散语言模型（DLMs）的研究进展。它分析了DLMs的原理、发展、技术分类、推理优化及多模态应用，对比自回归模型，探讨其优势与挑战，并展望未来方向。**

- **链接: [https://arxiv.org/pdf/2508.10875v2](https://arxiv.org/pdf/2508.10875v2)**

> **作者:** Tianyi Li; Mingda Chen; Bowei Guo; Zhiqiang Shen
>
> **摘要:** Diffusion Language Models (DLMs) are rapidly emerging as a powerful and promising alternative to the dominant autoregressive (AR) paradigm. By generating tokens in parallel through an iterative denoising process, DLMs possess inherent advantages in reducing inference latency and capturing bidirectional context, thereby enabling fine-grained control over the generation process. While achieving a several-fold speed-up, recent advancements have allowed DLMs to show performance comparable to their autoregressive counterparts, making them a compelling choice for various natural language processing tasks. In this survey, we provide a holistic overview of the current DLM landscape. We trace its evolution and relationship with other paradigms, such as autoregressive and masked language models, and cover both foundational principles and state-of-the-art models. Our work offers an up-to-date, comprehensive taxonomy and an in-depth analysis of current techniques, from pre-training strategies to advanced post-training methods. Another contribution of this survey is a thorough review of DLM inference strategies and optimizations, including improvements in decoding parallelism, caching mechanisms, and generation quality. We also highlight the latest approaches to multimodal extensions of DLMs and delineate their applications across various practical scenarios. Furthermore, our discussion addresses the limitations and challenges of DLMs, including efficiency, long-sequence handling, and infrastructure requirements, while outlining future research directions to sustain progress in this rapidly evolving field. Project GitHub is available at https://github.com/VILA-Lab/Awesome-DLMs.
>
---
