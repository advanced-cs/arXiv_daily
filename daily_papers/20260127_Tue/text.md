# 自然语言处理 cs.CL

- **最新发布 173 篇**

- **更新 144 篇**

## 最新发布

#### [new 001] DIETA: A Decoder-only transformer-based model for Italian-English machine TrAnslation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DIETA，一个用于意英机器翻译的小型解码器模型。针对意英翻译任务，通过构建大规模语料库和评估集，提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2601.17823v1](https://arxiv.org/pdf/2601.17823v1)**

> **作者:** Pranav Kasela; Marco Braga; Alessandro Ghiotto; Andrea Pilzer; Marco Viviani; Alessandro Raganato
>
> **备注:** Published in CLiC-IT '25: https://aclanthology.org/2025.clicit-1.52/
>
> **摘要:** In this paper, we present DIETA, a small, decoder-only Transformer model with 0.5 billion parameters, specifically designed and trained for Italian-English machine translation. We collect and curate a large parallel corpus consisting of approximately 207 million Italian-English sentence pairs across diverse domains, including parliamentary proceedings, legal texts, web-crawled content, subtitles, news, literature and 352 million back-translated data using pretrained models. Additionally, we create and release a new small-scale evaluation set, consisting of 450 sentences, based on 2025 WikiNews articles, enabling assessment of translation quality on contemporary text. Comprehensive evaluations show that DIETA achieves competitive performance on multiple Italian-English benchmarks, consistently ranking in the second quartile of a 32-system leaderboard and outperforming most other sub-3B models on four out of five test suites. The training script, trained models, curated corpus, and newly introduced evaluation set are made publicly available, facilitating further research and development in specialized Italian-English machine translation. https://github.com/pkasela/DIETA-Machine-Translation
>
---
#### [new 002] Parameter Efficient Fine Tuning Llama 3.1 for Answering Arabic Legal Questions: A Case Study on Jordanian Laws
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于阿拉伯语法律问答任务，旨在提升Llama-3.1模型在约旦法律领域的表现。通过参数高效微调和量化技术，优化模型性能与资源效率。**

- **链接: [https://arxiv.org/pdf/2601.17364v1](https://arxiv.org/pdf/2601.17364v1)**

> **作者:** Mohammed Fasha; Bassam Hammo; Bilal Sowan; Husam Barham; Esam Nsour
>
> **备注:** 5 pages, resources at: https://github.com/msfasha/Research-Resources/tree/main/ArabicLegalLLM
>
> **摘要:** This study uses Jordanian law as a case study to explore the fine-tuning of the Llama-3.1 large language model for Arabic question-answering. Two versions of the model - Llama-3.1-8B-bnb-4bit and Llama-3.1-8B-Instruct-bnb-4bit - were fine-tuned using parameter-efficient fine-tuning (PEFT) with LoRA adapters and 4-bit quantized models, leveraging the Unsloth framework for accelerated and resource-efficient training. A custom dataset of 6000 legal question-answer pairs was curated from Jordanian laws and formatted into structured prompts. Performance was evaluated using the BLEU and the ROUGE metrics to compare the fine-tuned models to their respective base versions. Results demonstrated improved legal reasoning and accuracy while achieving resource efficiency through quantization and optimized fine-tuning strategies. This work underscores the potential of adapting large language models for Arabic legal domains and highlights effective techniques for fine-tuning domain-specific tasks.
>
---
#### [new 003] Relating Word Embedding Gender Biases to Gender Gaps: A Cross-Cultural Analysis
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的性别偏 bias 分析任务，旨在探究词嵌入中的性别偏见与现实中的性别差距之间的关系，通过量化分析验证其相关性。**

- **链接: [https://arxiv.org/pdf/2601.17203v1](https://arxiv.org/pdf/2601.17203v1)**

> **作者:** Scott Friedman; Sonja Schmer-Galunder; Anthony Chen; Jeffrey Rye
>
> **备注:** 7 pages, 5 figures. Presented at the First Workshop on Gender Bias in Natural Language Processing (GeBNLP 2019)
>
> **摘要:** Modern models for common NLP tasks often employ machine learning techniques and train on journalistic, social media, or other culturally-derived text. These have recently been scrutinized for racial and gender biases, rooting from inherent bias in their training text. These biases are often sub-optimal and recent work poses methods to rectify them; however, these biases may shed light on actual racial or gender gaps in the culture(s) that produced the training text, thereby helping us understand cultural context through big data. This paper presents an approach for quantifying gender bias in word embeddings, and then using them to characterize statistical gender gaps in education, politics, economics, and health. We validate these metrics on 2018 Twitter data spanning 51 U.S. regions and 99 countries. We correlate state and country word embedding biases with 18 international and 5 U.S.-based statistical gender gaps, characterizing regularities and predictive strength.
>
---
#### [new 004] The Shadow Self: Intrinsic Value Misalignment in Large Language Model Agents
- **分类: cs.CL**

- **简介: 该论文属于AI安全任务，旨在解决大语言模型代理的价值错位问题。通过构建评估框架IMPRESS，分析模型在真实场景中的内在价值偏差，提出安全评估与改进方法。**

- **链接: [https://arxiv.org/pdf/2601.17344v1](https://arxiv.org/pdf/2601.17344v1)**

> **作者:** Chen Chen; Kim Young Il; Yuan Yang; Wenhao Su; Yilin Zhang; Xueluan Gong; Qian Wang; Yongsen Zheng; Ziyao Liu; Kwok-Yan Lam
>
> **备注:** 21 pages, 11 figures
>
> **摘要:** Large language model (LLM) agents with extended autonomy unlock new capabilities, but also introduce heightened challenges for LLM safety. In particular, an LLM agent may pursue objectives that deviate from human values and ethical norms, a risk known as value misalignment. Existing evaluations primarily focus on responses to explicit harmful input or robustness against system failure, while value misalignment in realistic, fully benign, and agentic settings remains largely underexplored. To fill this gap, we first formalize the Loss-of-Control risk and identify the previously underexamined Intrinsic Value Misalignment (Intrinsic VM). We then introduce IMPRESS (Intrinsic Value Misalignment Probes in REalistic Scenario Set), a scenario-driven framework for systematically assessing this risk. Following our framework, we construct benchmarks composed of realistic, fully benign, and contextualized scenarios, using a multi-stage LLM generation pipeline with rigorous quality control. We evaluate Intrinsic VM on 21 state-of-the-art LLM agents and find that it is a common and broadly observed safety risk across models. Moreover, the misalignment rates vary by motives, risk types, model scales, and architectures. While decoding strategies and hyperparameters exhibit only marginal influence, contextualization and framing mechanisms significantly shape misalignment behaviors. Finally, we conduct human verification to validate our automated judgments and assess existing mitigation strategies, such as safety prompting and guardrails, which show instability or limited effectiveness. We further demonstrate key use cases of IMPRESS across the AI Ecosystem. Our code and benchmark will be publicly released upon acceptance.
>
---
#### [new 005] TechING: Towards Real World Technical Image Understanding via VLMs
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于技术图像理解任务，旨在解决VLMs在识别手绘技术图方面的不足。通过生成合成数据并微调模型，提升其在真实手绘图像上的表现。**

- **链接: [https://arxiv.org/pdf/2601.18238v1](https://arxiv.org/pdf/2601.18238v1)**

> **作者:** Tafazzul Nadeem; Bhavik Shangari; Manish Rai; Gagan Raj Gupta; Ashutosh Modi
>
> **备注:** Accepted at Findings of EACL 2026, 30 Pages (9 Pages main paper + 4 pages references + 17 pages appendix)
>
> **摘要:** Professionals working in technical domain typically hand-draw (on whiteboard, paper, etc.) technical diagrams (e.g., flowcharts, block diagrams, etc.) during discussions; however, if they want to edit these later, it needs to be drawn from scratch. Modern day VLMs have made tremendous progress in image understanding but they struggle when it comes to understanding technical diagrams. One way to overcome this problem is to fine-tune on real world hand-drawn images, but it is not practically possible to generate large number of such images. In this paper, we introduce a large synthetically generated corpus (reflective of real world images) for training VLMs and subsequently evaluate VLMs on a smaller corpus of hand-drawn images (with the help of humans). We introduce several new self-supervision tasks for training and perform extensive experiments with various baseline models and fine-tune Llama 3.2 11B-instruct model on synthetic images on these tasks to obtain LLama-VL-TUG, which significantly improves the ROUGE-L performance of Llama 3.2 11B-instruct by 2.14x and achieves the best all-round performance across all baseline models. On real-world images, human evaluation reveals that we achieve minimum compilation errors across all baselines in 7 out of 8 diagram types and improve the average F1 score of Llama 3.2 11B-instruct by 6.97x.
>
---
#### [new 006] Dep-Search: Learning Dependency-Aware Reasoning Traces with Persistent Memory
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于复杂推理任务，旨在解决LLMs在多步推理中依赖隐式语言推理的问题。提出Dep-Search框架，通过显式控制机制实现结构化推理与持久记忆管理。**

- **链接: [https://arxiv.org/pdf/2601.18771v1](https://arxiv.org/pdf/2601.18771v1)**

> **作者:** Yanming Liu; Xinyue Peng; Zixuan Yan; Yanxin Shen; Wenjie Xu; Yuefeng Huang; Xinyi Wang; Jiannan Cao; Jianwei Yin; Xuhong Zhang
>
> **备注:** Dep-Search 1st version
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks, particularly when augmented with search mechanisms that enable systematic exploration of external knowledge bases. The field has evolved from traditional retrieval-augmented generation (RAG) frameworks to more sophisticated search-based frameworks that orchestrate multi-step reasoning through explicit search strategies. However, existing search frameworks still rely heavily on implicit natural language reasoning to determine search strategies and how to leverage retrieved information across reasoning steps. This reliance on implicit reasoning creates fundamental challenges for managing dependencies between sub-questions, efficiently reusing previously retrieved knowledge, and learning optimal search strategies through reinforcement learning. To address these limitations, we propose Dep-Search, a dependency-aware search framework that advances beyond existing search frameworks by integrating structured reasoning, retrieval, and persistent memory through GRPO. Dep-Search introduces explicit control mechanisms that enable the model to decompose questions with dependency relationships, retrieve information when needed, access previously stored knowledge from memory, and summarize long reasoning contexts into reusable memory entries. Through extensive experiments on seven diverse question answering datasets, we demonstrate that Dep-Search significantly enhances LLMs' ability to tackle complex multi-hop reasoning tasks, achieving substantial improvements over strong baselines across different model scales.
>
---
#### [new 007] Latent Knowledge as a Predictor of Fact Acquisition in Fine-Tuned Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在微调过程中事实获取的机制，探讨潜在线索对学习速度和泛化的预测作用，旨在理解模型如何存储与提取生物医学知识。**

- **链接: [https://arxiv.org/pdf/2601.18468v1](https://arxiv.org/pdf/2601.18468v1)**

> **作者:** Daniel B. Hier; Tayo Obafemi-Ajayi
>
> **摘要:** Large language models store biomedical facts with uneven strength after pretraining: some facts are present in the weights but are not reliably accessible under deterministic decoding (latent knowledge), while others are scarcely represented. We fine tuned Llama 3.1 8B Instruct to learn ontology term identifier mappings from the Human Phenotype Ontology (800 pairs) and the Gene Ontology (400 training pairs), withholding 400 GO pairs to test generalization. Treating learning as a time to event process across 20 epochs, we used stochastic decoding to detect latent knowledge at baseline and Cox proportional hazards models to identify predictors of acquisition, generalization, and degradation. Baseline deterministic recall for HPO was 2.8%, rising to 71.9% after fine-tuning. Latent knowledge was the strongest predictor of faster fact acquisition (HR 2.6) and was associated with earlier, higher peak learning rates and faster convergence; identifier frequency and curated annotation counts had smaller effects. Generalization to withheld GO facts was uncommon (5.8%) but more likely when latent knowledge was present. Previously correct GO mappings degraded more often for withheld (unseen) terms than for trained (seen) terms, suggesting a protective effect of reinforcement during training. These results show that latent knowledge predicts both the speed of factual learning during fine-tuning and the limited generalization of unseen ontology facts, while resistance to degradation depends on whether facts are reinforced.
>
---
#### [new 008] Beyond Outcome Verification: Verifiable Process Reward Models for Structured Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出VPRMs，用于结构化推理的可验证过程奖励模型。解决LLM中间推理步骤不可靠的问题，通过规则验证器提升推理的准确性和一致性。**

- **链接: [https://arxiv.org/pdf/2601.17223v1](https://arxiv.org/pdf/2601.17223v1)**

> **作者:** Massimiliano Pronesti; Anya Belz; Yufang Hou
>
> **摘要:** Recent work on reinforcement learning with verifiable rewards (RLVR) has shown that large language models (LLMs) can be substantially improved using outcome-level verification signals, such as unit tests for code or exact-match checks for mathematics. In parallel, process supervision has long been explored as a way to shape the intermediate reasoning behaviour of LLMs, but existing approaches rely on neural judges to score chain-of-thought steps, leaving them vulnerable to opacity, bias, and reward hacking. To address this gap, we introduce Verifiable Process Reward Models (VPRMs), a reinforcement-learning framework in which intermediate reasoning steps are checked by deterministic, rule-based verifiers. We apply VPRMs to risk-of-bias assessment for medical evidence synthesis, a domain where guideline-defined criteria and rule-based decision paths enable programmatic verification of reasoning traces. Across multiple datasets, we find that VPRMs generate reasoning that adheres closely to domain rules and achieve substantially higher coherence between step-level decisions and final labels. Results show that VPRMs achieve up to 20% higher F1 than state-of-the-art models and 6.5% higher than verifiable outcome rewards, with substantial gains in evidence grounding and logical coherence.
>
---
#### [new 009] Reflecting Twice before Speaking with Empathy: Self-Reflective Alternating Inference for Empathy-Aware End-to-End Spoken Dialogue
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于情感对话任务，旨在提升对话系统的共情能力。针对现有方法依赖单一标签的局限，提出ReEmpathy模型，通过自我反思推理机制增强共情对话表现。**

- **链接: [https://arxiv.org/pdf/2601.18281v1](https://arxiv.org/pdf/2601.18281v1)**

> **作者:** Yuhang Jia; Pei Liu; Haoqin Sun; Jiaming Zhou; Xuxin Cheng; Cao Liu; Ke Zeng; Xunliang Cai; Yong Qin
>
> **摘要:** End-to-end Spoken Language Models (SLMs) hold great potential for paralinguistic perception, and numerous studies have aimed to enhance their capabilities, particularly for empathetic dialogue. However, current approaches largely depend on rigid supervised signals, such as ground-truth response in supervised fine-tuning or preference scores in reinforcement learning. Such reliance is fundamentally limited for modeling complex empathy, as there is no single "correct" response and a simple numerical score cannot fully capture the nuances of emotional expression or the appropriateness of empathetic behavior. To address these limitations, we sequentially introduce EmpathyEval, a descriptive natural-language-based evaluation model for assessing empathetic quality in spoken dialogues. Building upon EmpathyEval, we propose ReEmpathy, an end-to-end SLM that enhances empathetic dialogue through a novel Empathetic Self-Reflective Alternating Inference mechanism, which interleaves spoken response generation with free-form, empathy-related reflective reasoning. Extensive experiments demonstrate that ReEmpathy substantially improves empathy-sensitive spoken dialogue by enabling reflective reasoning, offering a promising approach toward more emotionally intelligent and empathy-aware human-computer interactions.
>
---
#### [new 010] Hierarchical Text Classification with LLM-Refined Taxonomies
- **分类: cs.CL**

- **简介: 该论文属于层次文本分类任务，解决真实世界分类体系中的歧义问题。通过LLM优化分类体系，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2601.18375v1](https://arxiv.org/pdf/2601.18375v1)**

> **作者:** Jonas Golde; Nicolaas Jedema; Ravi Krishnan; Phong Le
>
> **摘要:** Hierarchical text classification (HTC) depends on taxonomies that organize labels into structured hierarchies. However, many real-world taxonomies introduce ambiguities, such as identical leaf names under similar parent nodes, which prevent language models (LMs) from learning clear decision boundaries. In this paper, we present TaxMorph, a framework that uses large language models (LLMs) to transform entire taxonomies through operations such as renaming, merging, splitting, and reordering. Unlike prior work, our method revises the full hierarchy to better match the semantics encoded by LMs. Experiments across three HTC benchmarks show that LLM-refined taxonomies consistently outperform human-curated ones in various settings up to +2.9pp. in F1. To better understand these improvements, we compare how well LMs can assign leaf nodes to parent nodes and vice versa across human-curated and LLM-refined taxonomies. We find that human-curated taxonomies lead to more easily separable clusters in embedding space. However, the LLM-refined taxonomies align more closely with the model's actual confusion patterns during classification. In other words, even though they are harder to separate, they better reflect the model's inductive biases. These findings suggest that LLM-guided refinement creates taxonomies that are more compatible with how models learn, improving HTC performance.
>
---
#### [new 011] CLM-Bench: Benchmarking and Analyzing Cross-lingual Misalignment of LLMs in Knowledge Editing
- **分类: cs.CL**

- **简介: 该论文属于多语言知识编辑任务，旨在解决现有评估框架的跨语言偏差问题。通过构建文化敏感的CLM-Bench基准，发现语言间知识编辑存在显著错位现象。**

- **链接: [https://arxiv.org/pdf/2601.17397v1](https://arxiv.org/pdf/2601.17397v1)**

> **作者:** Yucheng Hu; Wei Zhou; Juesi Xiao
>
> **备注:** EACL MME workshop paper
>
> **摘要:** Knowledge Editing (KE) has emerged as a promising paradigm for updating facts in Large Language Models (LLMs) without retraining. However, progress in Multilingual Knowledge Editing (MKE) is currently hindered by biased evaluation frameworks. We observe that existing MKE benchmarks are typically constructed by mechanically translating English-centric datasets into target languages (e.g., English-to-Chinese). This approach introduces translation artifacts and neglects culturally specific entities native to the target language, failing to reflect the true knowledge distribution of LLMs. To address this, we propose CLM-Bench, a culture-aware benchmark constructed using a native Chinese-first methodology. We curate 1,010 high-quality CounterFact pairs rooted in Chinese cultural contexts and align them with English counterparts. Using CLM-Bench, we conduct extensive experiments on representative LLMs (e.g., Llama-3, Qwen2) and reveal a significant Cross-lingual Misalignment: edits in one language function independently and fail to propagate to the other. We further provide a geometric explanation via layer-wise representation analysis, demonstrating that edit vectors for Chinese and English are nearly orthogonal -- residing in disjoint subspaces -- while mixed-lingual editing exhibits linear additivity of these vectors. Our findings challenge the effectiveness of current methods in cross-lingual transfer and underscore the importance of culturally native benchmarks.
>
---
#### [new 012] One Adapts to Any: Meta Reward Modeling for Personalized LLM Alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于个性化大模型对齐任务，解决用户反馈稀缺和适应新用户困难的问题。提出Meta Reward Modeling（MRM），通过元学习实现快速个性化适配。**

- **链接: [https://arxiv.org/pdf/2601.18731v1](https://arxiv.org/pdf/2601.18731v1)**

> **作者:** Hongru Cai; Yongqi Li; Tiezheng Yu; Fengbin Zhu; Wenjie Wang; Fuli Feng; Wenjie Li
>
> **摘要:** Alignment of Large Language Models (LLMs) aims to align outputs with human preferences, and personalized alignment further adapts models to individual users. This relies on personalized reward models that capture user-specific preferences and automatically provide individualized feedback. However, developing these models faces two critical challenges: the scarcity of feedback from individual users and the need for efficient adaptation to unseen users. We argue that addressing these constraints requires a paradigm shift from fitting data to learn user preferences to learn the process of preference adaptation. To realize this, we propose Meta Reward Modeling (MRM), which reformulates personalized reward modeling as a meta-learning problem. Specifically, we represent each user's reward model as a weighted combination of base reward functions, and optimize the initialization of these weights using a Model-Agnostic Meta-Learning (MAML)-style framework to support fast adaptation under limited feedback. To ensure robustness, we introduce the Robust Personalization Objective (RPO), which places greater emphasis on hard-to-learn users during meta optimization. Extensive experiments on personalized preference datasets validate that MRM enhances few-shot personalization, improves user robustness, and consistently outperforms baselines.
>
---
#### [new 013] Reasoning Beyond Literal: Cross-style Multimodal Reasoning for Figurative Language Understanding
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多模态推理任务，旨在解决 figurative language 理解难题。通过提出三步框架，提升模型对隐喻、讽刺等语言的跨风格理解能力。**

- **链接: [https://arxiv.org/pdf/2601.17197v1](https://arxiv.org/pdf/2601.17197v1)**

> **作者:** Seyyed Saeid Cheshmi; Hahnemann Ortiz; James Mooney; Dongyeop Kang
>
> **摘要:** Vision-language models (VLMs) have demonstrated strong reasoning abilities in literal multimodal tasks such as visual mathematics and science question answering. However, figurative language, such as sarcasm, humor, and metaphor, remains a significant challenge, as it conveys intent and emotion through subtle incongruities between expressed and intended meanings. In multimodal settings, accompanying images can amplify or invert textual meaning, demanding models that reason across modalities and account for subjectivity. We propose a three-step framework for developing efficient multimodal reasoning models that can (i) interpret multimodal figurative language, (ii) provide transparent reasoning traces, and (iii) generalize across multiple figurative styles. Experiments across four styles show that (1) incorporating reasoning traces substantially improves multimodal figurative understanding, (2) reasoning learned in one style can transfer to others, especially between related styles like sarcasm and humor, and (3) training jointly across styles yields a generalized reasoning VLM that outperforms much larger open- and closed-source models. Our findings show that lightweight VLMs with verifiable reasoning achieve robust cross-style generalization while providing inspectable reasoning traces for multimodal tasks. The code and implementation are available at https://github.com/scheshmi/CrossStyle-MMR.
>
---
#### [new 014] Revisiting Modality Invariance in a Multilingual Speech-Text Model via Neuron-Level Analysis
- **分类: cs.CL**

- **简介: 该论文研究多语言语音-文本模型的模态不变性问题，通过神经元层面分析，揭示其在不同模态间表示不一致的现象。**

- **链接: [https://arxiv.org/pdf/2601.17387v1](https://arxiv.org/pdf/2601.17387v1)**

> **作者:** Toshiki Nakai; Varsha Suresh; Vera Demberg
>
> **备注:** 8 pages for the main text, 51 figures, 1 table
>
> **摘要:** Multilingual speech-text foundation models aim to process language uniformly across both modality and language, yet it remains unclear whether they internally represent the same language consistently when it is spoken versus written. We investigate this question in SeamlessM4T v2 through three complementary analyses that probe where language and modality information is encoded, how selective neurons causally influence decoding, and how concentrated this influence is across the network. We identify language- and modality-selective neurons using average-precision ranking, investigate their functional role via median-replacement interventions at inference time, and analyze activation-magnitude inequality across languages and modalities. Across experiments, we find evidence of incomplete modality invariance. Although encoder representations become increasingly language-agnostic, this compression makes it more difficult for the shared decoder to recover the language of origin when constructing modality-agnostic representations, particularly when adapting from speech to text. We further observe sharply localized modality-selective structure in cross-attention key and value projections. Finally, speech-conditioned decoding and non-dominant scripts exhibit higher activation concentration, indicating heavier reliance on a small subset of neurons, which may underlie increased brittleness across modalities and languages.
>
---
#### [new 015] Gained in Translation: Privileged Pairwise Judges Enhance Multilingual Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多语言推理任务，解决低资源语言性能下降问题。通过SP3F框架，在无目标语言数据情况下提升模型表现。**

- **链接: [https://arxiv.org/pdf/2601.18722v1](https://arxiv.org/pdf/2601.18722v1)**

> **作者:** Lintang Sutawika; Gokul Swamy; Zhiwei Steven Wu; Graham Neubig
>
> **备注:** Code available at https://github.com/lintangsutawika/SP3F
>
> **摘要:** When asked a question in a language less seen in its training data, current reasoning large language models (RLMs) often exhibit dramatically lower performance than when asked the same question in English. In response, we introduce \texttt{SP3F} (Self-Play with Privileged Pairwise Feedback), a two-stage framework for enhancing multilingual reasoning without \textit{any} data in the target language(s). First, we supervise fine-tune (SFT) on translated versions of English question-answer pairs to raise base model correctness. Second, we perform RL with feedback from a pairwise judge in a self-play fashion, with the judge receiving the English reference response as \textit{privileged information}. Thus, even when none of the model's responses are completely correct, the privileged pairwise judge can still tell which response is better. End-to-end, \texttt{SP3F} greatly improves base model performance, even outperforming fully post-trained models on multiple math and non-math tasks with less than of the training data across the single-language, multilingual, and generalization to unseen language settings.
>
---
#### [new 016] Code over Words: Overcoming Semantic Inertia via Code-Grounded Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决LLMs在动态规则下无法抑制预训练先验的问题。通过代码表示替代描述性文本，提升模型逻辑推理能力。**

- **链接: [https://arxiv.org/pdf/2601.18352v1](https://arxiv.org/pdf/2601.18352v1)**

> **作者:** Manjie Xu; Isabella Yin; Xinyi Tu; Chi Zhang; Yixin Zhu
>
> **摘要:** LLMs struggle with Semantic Inertia: the inability to inhibit pre-trained priors (e.g., "Lava is Dangerous") when dynamic, in-context rules contradict them. We probe this phenomenon using Baba Is You, where physical laws are mutable text rules, enabling precise evaluation of models' ability to override learned priors when rules change. We quantatively observe that larger models can exhibit inverse scaling: they perform worse than smaller models when natural language reasoning requires suppressing pre-trained associations (e.g., accepting "Lava is Safe"). Our analysis attributes this to natural language encoding, which entangles descriptive semantics and logical rules, leading to persistent hallucinations of familiar physics despite explicit contradictory rules. Here we show that representing dynamics as executable code, rather than descriptive text, reverses this trend and enables effective prior inhibition. We introduce Code-Grounded Vistas (LCV), which fine-tunes models on counterfactual pairs and identifies states with contradictory rules, thereby forcing attention to logical constraints rather than visual semantics. This training-time approach outperforms expensive inference-time search methods in both efficiency and accuracy. Our results demonstrate that representation fundamentally determines whether scaling improves or impairs contextual reasoning. This challenges the assumption that larger models are universally better, with implications for domains that require dynamic overriding of learned priors.
>
---
#### [new 017] From Chains to DAGs: Probing the Graph Structure of Reasoning in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在探究大语言模型中的多步推理是否以图结构（DAG）形式存储。通过设计探针分析隐藏状态中的图结构特征，验证了模型内部存在可识别的DAG结构。**

- **链接: [https://arxiv.org/pdf/2601.17593v1](https://arxiv.org/pdf/2601.17593v1)**

> **作者:** Tianjun Zhong; Linyang He; Nima Mesgarani
>
> **摘要:** Recent progress in large language models has renewed interest in mechanistically characterizing how multi-step reasoning is represented and computed. While much prior work treats reasoning as a linear chain of steps, many reasoning problems are more naturally structured as directed acyclic graphs (DAGs), where intermediate conclusions may depend on multiple premises, branch into parallel sub-derivations, and later merge or be reused. Understanding whether such graph-structured reasoning is reflected in model internals remains an open question. In this work, we introduce Reasoning DAG Probing, a framework that directly asks whether LLM hidden states encode the geometry of a reasoning DAG in a linearly accessible form, and where this structure emerges across layers. Within this framework, we associate each reasoning node with a textual realization and train lightweight probes to predict two graph-theoretic properties from hidden states: node depth and pairwise node distance. We use these probes to analyze the layerwise emergence of DAG structure and evaluate controls that disrupt reasoning-relevant structure while preserving superficial textual properties. Our results provide evidence that reasoning DAG geometry is meaningfully encoded in intermediate layers, with recoverability varying systematically by node depth and model scale, suggesting that LLM reasoning is not only sequential but exhibits measurable internal graph structure.
>
---
#### [new 018] Overalignment in Frontier LLMs: An Empirical Study of Sycophantic Behaviour in Healthcare
- **分类: cs.CL**

- **简介: 该论文研究LLMs在医疗场景中的迎合行为，解决模型对用户意见过度迎合而非事实准确的问题。通过新指标分析模型可靠性，发现优化推理的模型存在脆弱性。**

- **链接: [https://arxiv.org/pdf/2601.18334v1](https://arxiv.org/pdf/2601.18334v1)**

> **作者:** Clément Christophe; Wadood Mohammed Abdul; Prateek Munjal; Tathagata Raha; Ronnie Rajan; Praveenkumar Kanithi
>
> **摘要:** As LLMs are increasingly integrated into clinical workflows, their tendency for sycophancy, prioritizing user agreement over factual accuracy, poses significant risks to patient safety. While existing evaluations often rely on subjective datasets, we introduce a robust framework grounded in medical MCQA with verifiable ground truths. We propose the Adjusted Sycophancy Score, a novel metric that isolates alignment bias by accounting for stochastic model instability, or "confusability". Through an extensive scaling analysis of the Qwen-3 and Llama-3 families, we identify a clear scaling trajectory for resilience. Furthermore, we reveal a counter-intuitive vulnerability in reasoning-optimized "Thinking" models: while they demonstrate high vanilla accuracy, their internal reasoning traces frequently rationalize incorrect user suggestions under authoritative pressure. Our results across frontier models suggest that benchmark performance is not a proxy for clinical reliability, and that simplified reasoning structures may offer superior robustness against expert-driven sycophancy.
>
---
#### [new 019] Less is More for RAG: Information Gain Pruning for Generator-Aligned Reranking and Evidence Selection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索与生成任务，解决RAG系统中如何高效选择证据的问题。提出IGP方法，通过信息增益筛选有效证据，提升效果并减少输入成本。**

- **链接: [https://arxiv.org/pdf/2601.17532v1](https://arxiv.org/pdf/2601.17532v1)**

> **作者:** Zhipeng Song; Yizhi Zhou; Xiangyu Kong; Jiulong Jiao; Xinrui Bao; Xu You; Xueqing Shi; Yuhang Zhou; Heng Qi
>
> **备注:** 26 pages, 10 figures
>
> **摘要:** Retrieval-augmented generation (RAG) grounds large language models with external evidence, but under a limited context budget, the key challenge is deciding which retrieved passages should be injected. We show that retrieval relevance metrics (e.g., NDCG) correlate weakly with end-to-end QA quality and can even become negatively correlated under multi-passage injection, where redundancy and mild conflicts destabilize generation. We propose \textbf{Information Gain Pruning (IGP)}, a deployment-friendly reranking-and-pruning module that selects evidence using a generator-aligned utility signal and filters weak or harmful passages before truncation, without changing existing budget interfaces. Across five open-domain QA benchmarks and multiple retrievers and generators, IGP consistently improves the quality--cost trade-off. In a representative multi-evidence setting, IGP delivers about +12--20% relative improvement in average F1 while reducing final-stage input tokens by roughly 76--79% compared to retriever-only baselines.
>
---
#### [new 020] Pisets: A Robust Speech Recognition System for Lectures and Interviews
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出Pisets系统，解决语音识别中的错误与幻觉问题，通过三组件架构提升讲座和采访的语音转文字准确性。**

- **链接: [https://arxiv.org/pdf/2601.18415v1](https://arxiv.org/pdf/2601.18415v1)**

> **作者:** Ivan Bondarenko; Daniil Grebenkin; Oleg Sedukhin; Mikhail Klementev; Roman Derunets; Lyudmila Budneva
>
> **摘要:** This work presents a speech-to-text system "Pisets" for scientists and journalists which is based on a three-component architecture aimed at improving speech recognition accuracy while minimizing errors and hallucinations associated with the Whisper model. The architecture comprises primary recognition using Wav2Vec2, false positive filtering via the Audio Spectrogram Transformer (AST), and final speech recognition through Whisper. The implementation of curriculum learning methods and the utilization of diverse Russian-language speech corpora significantly enhanced the system's effectiveness. Additionally, advanced uncertainty modeling techniques were introduced, contributing to further improvements in transcription quality. The proposed approaches ensure robust transcribing of long audio data across various acoustic conditions compared to WhisperX and the usual Whisper model. The source code of "Pisets" system is publicly available at GitHub: https://github.com/bond005/pisets.
>
---
#### [new 021] Beyond Factual QA: Mentorship-Oriented Question Answering over Long-Form Multilingual Content
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MentorQA，解决多语言长文本下的导师型问答任务，关注回答的指导性和学习价值，对比不同架构效果。**

- **链接: [https://arxiv.org/pdf/2601.17173v1](https://arxiv.org/pdf/2601.17173v1)**

> **作者:** Parth Bhalerao; Diola Dsouza; Ruiwen Guan; Oana Ignat
>
> **摘要:** Question answering systems are typically evaluated on factual correctness, yet many real-world applications-such as education and career guidance-require mentorship: responses that provide reflection and guidance. Existing QA benchmarks rarely capture this distinction, particularly in multilingual and long-form settings. We introduce MentorQA, the first multilingual dataset and evaluation framework for mentorship-focused question answering from long-form videos, comprising nearly 9,000 QA pairs from 180 hours of content across four languages. We define mentorship-focused evaluation dimensions that go beyond factual accuracy, capturing clarity, alignment, and learning value. Using MentorQA, we compare Single-Agent, Dual-Agent, RAG, and Multi-Agent QA architectures under controlled conditions. Multi-Agent pipelines consistently produce higher-quality mentorship responses, with especially strong gains for complex topics and lower-resource languages. We further analyze the reliability of automated LLM-based evaluation, observing substantial variation in alignment with human judgments. Overall, this work establishes mentorship-focused QA as a distinct research problem and provides a multilingual benchmark for studying agentic architectures and evaluation design in educational AI. The dataset and evaluation framework are released at https://github.com/AIM-SCU/MentorQA.
>
---
#### [new 022] WarrantScore: Modeling Warrants between Claims and Evidence for Substantiation Evaluation in Peer Reviews
- **分类: cs.CL**

- **简介: 该论文属于科学评论中的论证评估任务，旨在解决如何有效评估论点与证据之间逻辑关系的问题。提出WarrantScore方法，通过分析论点与证据的关联性来提升评阅效率。**

- **链接: [https://arxiv.org/pdf/2601.17377v1](https://arxiv.org/pdf/2601.17377v1)**

> **作者:** Kiyotada Mori; Shohei Tanaka; Tosho Hirasawa; Tadashi Kozuno; Koichiro Yoshino; Yoshitaka Ushiku
>
> **摘要:** The scientific peer-review process is facing a shortage of human resources due to the rapid growth in the number of submitted papers. The use of language models to reduce the human cost of peer review has been actively explored as a potential solution to this challenge. A method has been proposed to evaluate the level of substantiation in scientific reviews in a manner that is interpretable by humans. This method extracts the core components of an argument, claims and evidence, and assesses the level of substantiation based on the proportion of claims supported by evidence. The level of substantiation refers to the extent to which claims are based on objective facts. However, when assessing the level of substantiation, simply detecting the presence or absence of supporting evidence for a claim is insufficient; it is also necessary to accurately assess the logical inference between a claim and its evidence. We propose a new evaluation metric for scientific review comments that assesses the logical inference between claims and evidence. Experimental results show that the proposed method achieves a higher correlation with human scores than conventional methods, indicating its potential to better support the efficiency of the peer-review process.
>
---
#### [new 023] SD-E$^2$: Semantic Exploration for Reasoning Under Token Budgets
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于复杂推理任务，解决小语言模型在计算预算下探索效率低的问题。提出SD-E²框架，通过优化语义多样性提升推理能力。**

- **链接: [https://arxiv.org/pdf/2601.17982v1](https://arxiv.org/pdf/2601.17982v1)**

> **作者:** Kshitij Mishra; Nils Lukas; Salem Lahlou
>
> **备注:** Accepted at EACL 2026
>
> **摘要:** Small language models (SLMs) struggle with complex reasoning because exploration is expensive under tight compute budgets. We introduce Semantic Diversity-Exploration-Exploitation (SD-E$^2$), a reinforcement learning framework that makes exploration explicit by optimizing semantic diversity in generated reasoning trajectories. Using a frozen sentence-embedding model, SD-E$^2$ assigns a diversity reward that captures (i) the coverage of semantically distinct solution strategies and (ii) their average pairwise dissimilarity in embedding space, rather than surface-form novelty. This diversity reward is combined with outcome correctness and solution efficiency in a z-score-normalized multi-objective objective that stabilizes training. On GSM8K, SD-E$^2$ surpasses the base Qwen2.5-3B-Instruct and strong GRPO baselines (GRPO-CFL and GRPO-CFEE) by +27.4, +5.2, and +1.5 percentage points, respectively, while discovering on average 9.8 semantically distinct strategies per question. We further improve MedMCQA to 49.64% versus 38.37% for the base model and show gains on the harder AIME benchmark (1983-2025), reaching 13.28% versus 6.74% for the base. These results indicate that rewarding semantic novelty yields a more compute-efficient exploration-exploitation signal for training reasoning-capable SLMs. By introducing cognitive adaptation-adjusting the reasoning process structure rather than per-token computation-SD-E$^2$ offers a complementary path to efficiency gains in resource-constrained models.
>
---
#### [new 024] A System for Name and Address Parsing with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息提取任务，旨在解决从非结构化文本中准确提取结构化姓名地址的问题。工作包括设计一个无需微调的框架，结合规范化、结构化提示和规则验证，提升准确性和可重复性。**

- **链接: [https://arxiv.org/pdf/2601.18014v1](https://arxiv.org/pdf/2601.18014v1)**

> **作者:** Adeeba Tarannum; Muzakkiruddin Ahmed Mohammed; Mert Can Cakmak; Shames Al Mandalawi; John Talburt
>
> **摘要:** Reliable transformation of unstructured person and address text into structured data remains a key challenge in large-scale information systems. Traditional rule-based and probabilistic approaches perform well on clean inputs but fail under noisy or multilingual conditions, while neural and large language models (LLMs) often lack deterministic control and reproducibility. This paper introduces a prompt-driven, validation-centered framework that converts free-text records into a consistent 17-field schema without fine-tuning. The method integrates input normalisation, structured prompting, constrained decoding, and strict rule-based validation under fixed experimental settings to ensure reproducibility. Evaluations on heterogeneous real-world address data show high field-level accuracy, strong schema adherence, and stable confidence calibration. The results demonstrate that combining deterministic validation with generative prompting provides a robust, interpretable, and scalable solution for structured information extraction, offering a practical alternative to training-heavy or domain-specific models.
>
---
#### [new 025] DF-RAG: Query-Aware Diversity for Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于问答任务，解决RAG在复杂推理问题中因冗余内容导致信息召回下降的问题。提出DF-RAG，通过引入多样性提升性能。**

- **链接: [https://arxiv.org/pdf/2601.17212v1](https://arxiv.org/pdf/2601.17212v1)**

> **作者:** Saadat Hasan Khan; Spencer Hong; Jingyu Wu; Kevin Lybarger; Youbing Yin; Erin Babinsky; Daben Liu
>
> **备注:** Accepted to Findings of EACL 2026
>
> **摘要:** Retrieval-augmented generation (RAG) is a common technique for grounding language model outputs in domain-specific information. However, RAG is often challenged by reasoning-intensive question-answering (QA), since common retrieval methods like cosine similarity maximize relevance at the cost of introducing redundant content, which can reduce information recall. To address this, we introduce Diversity-Focused Retrieval-Augmented Generation (DF-RAG), which systematically incorporates diversity into the retrieval step to improve performance on complex, reasoning-intensive QA benchmarks. DF-RAG builds upon the Maximal Marginal Relevance framework to select information chunks that are both relevant to the query and maximally dissimilar from each other. A key innovation of DF-RAG is its ability to optimize the level of diversity for each query dynamically at test time without requiring any additional fine-tuning or prior information. We show that DF-RAG improves F1 performance on reasoning-intensive QA benchmarks by 4-10 percent over vanilla RAG using cosine similarity and also outperforms other established baselines. Furthermore, we estimate an Oracle ceiling of up to 18 percent absolute F1 gains over vanilla RAG, of which DF-RAG captures up to 91.3 percent.
>
---
#### [new 026] Demographic Probing of Large Language Models Lacks Construct Validity
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理中的模型分析任务，旨在解决 demographic probing 方法的有效性问题。研究发现该方法缺乏构造有效性，导致结果不稳定。**

- **链接: [https://arxiv.org/pdf/2601.18486v1](https://arxiv.org/pdf/2601.18486v1)**

> **作者:** Manuel Tonneau; Neil K. R. Seghal; Niyati Malhotra; Victor Orozco-Olvera; Ana María Muñoz Boudet; Lakshmi Subramanian; Sharath Chandra Guntuku; Valentin Hofmann
>
> **摘要:** Demographic probing is widely used to study how large language models (LLMs) adapt their behavior to signaled demographic attributes. This approach typically uses a single demographic cue in isolation (e.g., a name or dialect) as a signal for group membership, implicitly assuming strong construct validity: that such cues are interchangeable operationalizations of the same underlying, demographically conditioned behavior. We test this assumption in realistic advice-seeking interactions, focusing on race and gender in a U.S. context. We find that cues intended to represent the same demographic group induce only partially overlapping changes in model behavior, while differentiation between groups within a given cue is weak and uneven. Consequently, estimated disparities are unstable, with both magnitude and direction varying across cues. We further show that these inconsistencies partly arise from variation in how strongly cues encode demographic attributes and from linguistic confounders that independently shape model behavior. Together, our findings suggest that demographic probing lacks construct validity: it does not yield a single, stable characterization of how LLMs condition on demographic information, which may reflect a misspecified or fragmented construct. We conclude by recommending the use of multiple, ecologically valid cues and explicit control of confounders to support more defensible claims about demographic effects in LLMs.
>
---
#### [new 027] FABLE: Forest-Based Adaptive Bi-Path LLM-Enhanced Retrieval for Multi-Document Reasoning
- **分类: cs.CL**

- **简介: 该论文提出FABLE框架，解决多文档推理中检索效率与精度不足的问题。通过结合LLM构建层次化索引，实现高效精准的证据获取。**

- **链接: [https://arxiv.org/pdf/2601.18116v1](https://arxiv.org/pdf/2601.18116v1)**

> **作者:** Lin Sun; Linglin Zhang; Jingang Huang; Change Jia; Zhengwei Cheng; Xiangzheng Zhang
>
> **摘要:** The rapid expansion of long-context Large Language Models (LLMs) has reignited debate on whether Retrieval-Augmented Generation (RAG) remains necessary. However, empirical evidence reveals persistent limitations of long-context inference, including the lost-in-the-middle phenomenon, high computational cost, and poor scalability for multi-document reasoning. Conversely, traditional RAG systems, while efficient, are constrained by flat chunk-level retrieval that introduces semantic noise and fails to support structured cross-document synthesis. We present \textbf{FABLE}, a \textbf{F}orest-based \textbf{A}daptive \textbf{B}i-path \textbf{L}LM-\textbf{E}nhanced retrieval framework that integrates LLMs into both knowledge organization and retrieval. FABLE constructs LLM-enhanced hierarchical forest indexes with multi-granularity semantic structures, then employs a bi-path strategy combining LLM-guided hierarchical traversal with structure-aware propagation for fine-grained evidence acquisition, with explicit budget control for adaptive efficiency trade-offs. Extensive experiments demonstrate that FABLE consistently outperforms SOTA RAG methods and achieves comparable accuracy to full-context LLM inference with up to 94\% token reduction, showing that long-context LLMs amplify rather than fully replace the need for structured retrieval.
>
---
#### [new 028] Interpretability of the Intent Detection Problem: A New Approach
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究意图检测任务，探讨RNN如何解决该问题。通过动态系统理论分析，揭示了数据集特性对网络计算解的影响。**

- **链接: [https://arxiv.org/pdf/2601.17156v1](https://arxiv.org/pdf/2601.17156v1)**

> **作者:** Eduardo Sanchez-Karhunen; Jose F. Quesada-Moreno; Miguel A. Gutiérrez-Naranjo
>
> **备注:** Accepted for publication in The European Journal on Artificial Intelligence (2026)
>
> **摘要:** Intent detection, a fundamental text classification task, aims to identify and label the semantics of user queries, playing a vital role in numerous business applications. Despite the dominance of deep learning techniques in this field, the internal mechanisms enabling Recurrent Neural Networks (RNNs) to solve intent detection tasks are poorly understood. In this work, we apply dynamical systems theory to analyze how RNN architectures address this problem, using both the balanced SNIPS and the imbalanced ATIS datasets. By interpreting sentences as trajectories in the hidden state space, we first show that on the balanced SNIPS dataset, the network learns an ideal solution: the state space, constrained to a low-dimensional manifold, is partitioned into distinct clusters corresponding to each intent. The application of this framework to the imbalanced ATIS dataset then reveals how this ideal geometric solution is distorted by class imbalance, causing the clusters for low-frequency intents to degrade. Our framework decouples geometric separation from readout alignment, providing a novel, mechanistic explanation for real world performance disparities. These findings provide new insights into RNN dynamics, offering a geometric interpretation of how dataset properties directly shape a network's computational solution.
>
---
#### [new 029] CitiLink: Enhancing Municipal Transparency and Citizen Engagement through Searchable Meeting Minutes
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决市政会议记录难以查找的问题。通过NLP技术将非结构化文本转为可搜索数据，提升政府透明度与公众参与度。**

- **链接: [https://arxiv.org/pdf/2601.18374v1](https://arxiv.org/pdf/2601.18374v1)**

> **作者:** Rodrigo Silva; José Evans; José Isidro; Miguel Marques; Afonso Fonseca; Ricardo Morais; João Canavilhas; Arian Pasquali; Purificação Silvano; Alípio Jorge; Nuno Guimarães; Sérgio Nunes; Ricardo Campos
>
> **摘要:** City council minutes are typically lengthy and formal documents with a bureaucratic writing style. Although publicly available, their structure often makes it difficult for citizens or journalists to efficiently find information. In this demo, we present CitiLink, a platform designed to transform unstructured municipal meeting minutes into structured and searchable data, demonstrating how NLP and IR can enhance the accessibility and transparency of local government. The system employs LLMs to extract metadata, discussed subjects, and voting outcomes, which are then indexed in a database to support full-text search with BM25 ranking and faceted filtering through a user-friendly interface. The developed system was built over a collection of 120 minutes made available by six Portuguese municipalities. To assess its usability, CitiLink was tested through guided sessions with municipal personnel, providing insights into how real users interact with the system. In addition, we evaluated Gemini's performance in extracting relevant information from the minutes, highlighting its effectiveness in data extraction.
>
---
#### [new 030] Meta-Judging with Large Language Models: Concepts, Methods, and Challenges
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，旨在解决LLM-as-a-Judge的局限性，通过提出LLM-as-a-Meta-Judge框架提升评估的稳定性与可靠性。**

- **链接: [https://arxiv.org/pdf/2601.17312v1](https://arxiv.org/pdf/2601.17312v1)**

> **作者:** Hugo Silva; Mateus Mendes; Hugo Gonçalo Oliveira
>
> **摘要:** Large language models (LLMs) are evolving fast and are now frequently used as evaluators, in a process typically referred to as LLM-as-a-Judge, which provides quality assessments of model outputs. However, recent research points out significant vulnerabilities in such evaluation, including sensitivity to prompts, systematic biases, verbosity effects, and unreliable or hallucinated rationales. These limitations motivated the development of a more robust paradigm, dubbed LLM-as-a-Meta-Judge. This survey reviews recent advances in meta-judging and organizes the literature, by introducing a framework along six key perspectives: (i) Conceptual Foundations, (ii) Mechanisms of Meta-Judging, (iii) Alignment Training Methods, (iv) Evaluation, (v) Limitations and Failure Modes, and (vi) Future Directions. By analyzing the limitations of LLM-as-a-Judge and summarizing recent advances in meta-judging by LLMs, we argue that LLM-as-a-Meta-Judge offers a promising direction for more stable and trustworthy automated evaluation, while highlighting remaining challenges related to cost, prompt sensitivity, and shared model biases, which must be addressed to advance the next generation of LLM evaluation methodologies.
>
---
#### [new 031] Learning to Ideate for Machine Learning Engineering Agents
- **分类: cs.CL**

- **简介: 该论文提出MLE-Ideator框架，解决机器学习工程代理迭代优化算法效果差的问题。通过分离构思与实现，提升系统性能。**

- **链接: [https://arxiv.org/pdf/2601.17596v1](https://arxiv.org/pdf/2601.17596v1)**

> **作者:** Yunxiang Zhang; Kang Zhou; Zhichao Xu; Kiran Ramnath; Yun Zhou; Sangmin Woo; Haibo Ding; Lin Lee Cheong
>
> **备注:** EACL 2026 main conference
>
> **摘要:** Existing machine learning engineering (MLE) agents struggle to iteratively optimize their implemented algorithms for effectiveness. To address this, we introduce MLE-Ideator, a dual-agent framework that separates ideation from implementation. In our system, an implementation agent can request strategic help from a dedicated Ideator. We show this approach is effective in two ways. First, in a training-free setup, our framework significantly outperforms implementation-only agent baselines on MLE-Bench. Second, we demonstrate that the Ideator can be trained with reinforcement learning (RL) to generate more effective ideas. With only 1K training samples from 10 MLE tasks, our RL-trained Qwen3-8B Ideator achieves an 11.5% relative improvement compared to its untrained counterpart and surpasses Claude Sonnet 3.5. These results highlights a promising path toward training strategic AI systems for scientific discovery.
>
---
#### [new 032] Neurocomputational Mechanisms of Syntactic Transfer in Bilingual Sentence Production
- **分类: cs.CL**

- **简介: 该论文属于语言认知研究，探讨双语句法转移的神经计算机制。解决双语产生错误的神经基础问题，通过ROSE模型分析句法转移及语言干扰现象。**

- **链接: [https://arxiv.org/pdf/2601.18056v1](https://arxiv.org/pdf/2601.18056v1)**

> **作者:** Ahmet Yavuz Uluslu; Elliot Murphy
>
> **摘要:** We discuss the benefits of incorporating into the study of bilingual production errors and their traditionally documented timing signatures (e.g., event-related potentials) certain types of oscillatory signatures, which can offer new implementational-level constraints for theories of bilingualism. We argue that a recent neural model of language, ROSE, can offer a neurocomputational account of syntactic transfer in bilingual production, capturing some of its formal properties and the scope of morphosyntactic sequencing failure modes. We take as a case study cross-linguistic influence (CLI) and attendant theories of functional inhibition/competition, and present these as being driven by specific oscillatory failure modes during L2 sentence planning. We argue that modeling CLI in this way not only offers the kind of linking hypothesis ROSE was built to encourage, but also licenses the exploration of more spatiotemporally complex biomarkers of language dysfunction than more commonly discussed neural signatures.
>
---
#### [new 033] RAM-SD: Retrieval-Augmented Multi-agent framework for Sarcasm Detection
- **分类: cs.CL**

- **简介: 该论文属于讽刺检测任务，旨在解决现有方法难以处理讽刺表达中多样的语境和知识需求的问题。提出RAM-SD框架，通过多智能体协作提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.17002v1](https://arxiv.org/pdf/2601.17002v1)**

> **作者:** Ziyang Zhou; Ziqi Liu; Yan Wang; Yiming Lin; Yangbin Chen
>
> **备注:** 12 pages, 4 figures, 6 tables, preprint
>
> **摘要:** Sarcasm detection remains a significant challenge due to its reliance on nuanced contextual understanding, world knowledge, and multi-faceted linguistic cues that vary substantially across different sarcastic expressions. Existing approaches, from fine-tuned transformers to large language models, apply a uniform reasoning strategy to all inputs, struggling to address the diverse analytical demands of sarcasm. These demands range from modeling contextual expectation violations to requiring external knowledge grounding or recognizing specific rhetorical patterns. To address this limitation, we introduce RAM-SD, a Retrieval-Augmented Multi-Agent framework for Sarcasm Detection. The framework operates through four stages: (1) contextual retrieval grounds the query in both sarcastic and non-sarcastic exemplars; (2) a meta-planner classifies the sarcasm type and selects an optimal reasoning plan from a predefined set; (3) an ensemble of specialized agents performs complementary, multi-view analysis; and (4) an integrator synthesizes these analyses into a final, interpretable judgment with a natural language explanation. Evaluated on four standard benchmarks, RAM-SD achieves a state-of-the-art Macro-F1 of 77.74%, outperforming the strong GPT-4o+CoC baseline by 7.01 points. Our framework not only sets a new performance benchmark but also provides transparent and interpretable reasoning traces, illuminating the cognitive processes behind sarcasm comprehension.
>
---
#### [new 034] MortalMATH: Evaluating the Conflict Between Reasoning Objectives and Emergency Contexts
- **分类: cs.CL**

- **简介: 论文探讨了大模型在紧急情境下因过度追求推理任务而忽视安全的问题。属于安全与伦理任务，旨在解决模型在危机中行为失当的问题，通过构建MortalMATH基准测试进行分析。**

- **链接: [https://arxiv.org/pdf/2601.18790v1](https://arxiv.org/pdf/2601.18790v1)**

> **作者:** Etienne Lanzeray; Stephane Meilliez; Malo Ruelle; Damien Sileo
>
> **摘要:** Large Language Models are increasingly optimized for deep reasoning, prioritizing the correct execution of complex tasks over general conversation. We investigate whether this focus on calculation creates a "tunnel vision" that ignores safety in critical situations. We introduce MortalMATH, a benchmark of 150 scenarios where users request algebra help while describing increasingly life-threatening emergencies (e.g., stroke symptoms, freefall). We find a sharp behavioral split: generalist models (like Llama-3.1) successfully refuse the math to address the danger. In contrast, specialized reasoning models (like Qwen-3-32b and GPT-5-nano) often ignore the emergency entirely, maintaining over 95 percent task completion rates while the user describes dying. Furthermore, the computational time required for reasoning introduces dangerous delays: up to 15 seconds before any potential help is offered. These results suggest that training models to relentlessly pursue correct answers may inadvertently unlearn the survival instincts required for safe deployment.
>
---
#### [new 035] Funny or Persuasive, but Not Both: Evaluating Fine-Grained Multi-Concept Control in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，旨在解决多概念精细控制的问题。研究发现，模型在同时控制多个独立概念时性能下降，揭示了提示方法在组合性上的局限性。**

- **链接: [https://arxiv.org/pdf/2601.18483v1](https://arxiv.org/pdf/2601.18483v1)**

> **作者:** Arya Labroo; Ivaxi Sheth; Vyas Raina; Amaani Ahmed; Mario Fritz
>
> **备注:** Accepted for publication at EACL main conference
>
> **摘要:** Large Language Models (LLMs) offer strong generative capabilities, but many applications require explicit and \textit{fine-grained} control over specific textual concepts, such as humor, persuasiveness, or formality. Prior approaches in prompting and representation engineering can provide coarse or single-attribute control, but systematic evaluation of multi-attribute settings remains limited. We introduce an evaluation framework for fine-grained controllability for both single- and dual-concept scenarios, focusing on linguistically distinct concept pairs (e.g., persuasiveness vs.~humor). Surprisingly, across multiple LLMs and generative tasks, we find that performance often drops in the dual-concept setting, even though the chosen concepts should in principle be separable. This reveals a fundamental limitation of naive prompting-based control: models struggle with compositionality even when concepts are intuitively independent. Our framework provides systematic evidence of this gap and offers a principled approach for measuring the ability of future methods for multi-concept control.
>
---
#### [new 036] MultiVis-Agent: A Multi-Agent Framework with Logic Rules for Reliable and Comprehensive Cross-Modal Data Visualization
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 该论文提出MultiVis-Agent框架，解决跨模态可视化生成中的复杂性和可靠性问题。通过逻辑规则增强多智能体系统，提升任务完成率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.18320v1](https://arxiv.org/pdf/2601.18320v1)**

> **作者:** Jinwei Lu; Yuanfeng Song; Chen Zhang; Raymond Chi-Wing Wong
>
> **备注:** Accepted to SIGMOD 2026
>
> **摘要:** Real-world visualization tasks involve complex, multi-modal requirements that extend beyond simple text-to-chart generation, requiring reference images, code examples, and iterative refinement. Current systems exhibit fundamental limitations: single-modality input, one-shot generation, and rigid workflows. While LLM-based approaches show potential for these complex requirements, they introduce reliability challenges including catastrophic failures and infinite loop susceptibility. To address this gap, we propose MultiVis-Agent, a logic rule-enhanced multi-agent framework for reliable multi-modal and multi-scenario visualization generation. Our approach introduces a four-layer logic rule framework that provides mathematical guarantees for system reliability while maintaining flexibility. Unlike traditional rule-based systems, our logic rules are mathematical constraints that guide LLM reasoning rather than replacing it. We formalize the MultiVis task spanning four scenarios from basic generation to iterative refinement, and develop MultiVis-Bench, a benchmark with over 1,000 cases for multi-modal visualization evaluation. Extensive experiments demonstrate that our approach achieves 75.63% visualization score on challenging tasks, significantly outperforming baselines (57.54-62.79%), with task completion rates of 99.58% and code execution success rates of 94.56% (vs. 74.48% and 65.10% without logic rules), successfully addressing both complexity and reliability challenges in automated visualization generation.
>
---
#### [new 037] Uncertainty Quantification for Named Entity Recognition via Full-Sequence and Subsequence Conformal Prediction
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于命名实体识别任务，解决模型预测缺乏不确定性度量的问题。通过构建预测集，提供形式化置信保障，提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2601.16999v1](https://arxiv.org/pdf/2601.16999v1)**

> **作者:** Matthew Singer; Srijan Sengupta; Karl Pazdernik
>
> **摘要:** Named Entity Recognition (NER) serves as a foundational component in many natural language processing (NLP) pipelines. However, current NER models typically output a single predicted label sequence without any accompanying measure of uncertainty, leaving downstream applications vulnerable to cascading errors. In this paper, we introduce a general framework for adapting sequence-labeling-based NER models to produce uncertainty-aware prediction sets. These prediction sets are collections of full-sentence labelings that are guaranteed to contain the correct labeling with a user-specified confidence level. This approach serves a role analogous to confidence intervals in classical statistics by providing formal guarantees about the reliability of model predictions. Our method builds on conformal prediction, which offers finite-sample coverage guarantees under minimal assumptions. We design efficient nonconformity scoring functions to construct efficient, well-calibrated prediction sets that support both unconditional and class-conditional coverage. This framework accounts for heterogeneity across sentence length, language, entity type, and number of entities within a sentence. Empirical experiments on four NER models across three benchmark datasets demonstrate the broad applicability, validity, and efficiency of the proposed methods.
>
---
#### [new 038] BoRP: Bootstrapped Regression Probing for Scalable and Human-Aligned LLM Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话AI评估任务，解决用户满意度评估难题。提出BoRP框架，利用LLM隐空间几何特性，实现高效高精度评价。**

- **链接: [https://arxiv.org/pdf/2601.18253v1](https://arxiv.org/pdf/2601.18253v1)**

> **作者:** Peng Sun; Xiangyu Zhang; Duan Wu
>
> **备注:** This is a pre-print
>
> **摘要:** Accurate evaluation of user satisfaction is critical for iterative development of conversational AI. However, for open-ended assistants, traditional A/B testing lacks reliable metrics: explicit feedback is sparse, while implicit metrics are ambiguous. To bridge this gap, we introduce BoRP (Bootstrapped Regression Probing), a scalable framework for high-fidelity satisfaction evaluation. Unlike generative approaches, BoRP leverages the geometric properties of LLM latent space. It employs a polarization-index-based bootstrapping mechanism to automate rubric generation and utilizes Partial Least Squares (PLS) to map hidden states to continuous scores. Experiments on industrial datasets show that BoRP (Qwen3-8B/14B) significantly outperforms generative baselines (even Qwen3-Max) in alignment with human judgments. Furthermore, BoRP reduces inference costs by orders of magnitude, enabling full-scale monitoring and highly sensitive A/B testing via CUPED.
>
---
#### [new 039] Sequence Repetition Enhances Token Embeddings and Improves Sequence Labeling with Decoder-only Language Models
- **分类: cs.CL**

- **简介: 该论文研究序列标注任务，解决decoder-only模型在双向上下文建模上的不足。通过序列重复技术提升token嵌入质量，增强模型性能。**

- **链接: [https://arxiv.org/pdf/2601.17585v1](https://arxiv.org/pdf/2601.17585v1)**

> **作者:** Matija Luka Kukić; Marko Čuljak; David Dukić; Martin Tutek; Jan Šnajder
>
> **备注:** Accepted at EACL 2026 Findings
>
> **摘要:** Modern language models (LMs) are trained in an autoregressive manner, conditioned only on the prefix. In contrast, sequence labeling (SL) tasks assign labels to each individual input token, naturally benefiting from bidirectional context. This discrepancy has historically led SL to rely on inherently bidirectional encoder-only models. However, the rapid development of decoder-only models has raised the question of whether they can be adapted to SL. While causal mask removal has emerged as a viable technique for adapting decoder-only models to leverage the full context for SL, it requires considerable changes to the base model functionality. In this work, we explore sequence repetition (SR) as a less invasive alternative for enabling bidirectionality in decoder-only models. Through fine-tuning experiments, we show that SR inherently makes decoders bidirectional, improving the quality of token-level embeddings and surpassing encoders and unmasked decoders. Contrary to earlier claims, we find that increasing the number of repetitions does not degrade SL performance. Finally, we demonstrate that embeddings from intermediate layers are highly effective for SR, comparable to those from final layers, while being significantly more efficient to compute. Our findings underscore that SR alleviates the structural limitations of decoders, enabling more efficient and adaptable LMs and broadening their applicability to other token-level tasks.
>
---
#### [new 040] Beyond a Single Perspective: Text Anomaly Detection with Multi-View Language Representations
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本异常检测任务，旨在解决单一嵌入模型和适应性不足的问题。通过多视图语言表示和自适应融合，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.17786v1](https://arxiv.org/pdf/2601.17786v1)**

> **作者:** Yixin Liu; Kehan Yan; Shiyuan Li; Qingfeng Chen; Shirui Pan
>
> **备注:** 17 pages, 7 tables, and 5 figures
>
> **摘要:** Text anomaly detection (TAD) plays a critical role in various language-driven real-world applications, including harmful content moderation, phishing detection, and spam review filtering. While two-step "embedding-detector" TAD methods have shown state-of-the-art performance, their effectiveness is often limited by the use of a single embedding model and the lack of adaptability across diverse datasets and anomaly types. To address these limitations, we propose to exploit the embeddings from multiple pretrained language models and integrate them into $MCA^2$, a multi-view TAD framework. $MCA^2$ adopts a multi-view reconstruction model to effectively extract normal textual patterns from multiple embedding perspectives. To exploit inter-view complementarity, a contrastive collaboration module is designed to leverage and strengthen the interactions across different views. Moreover, an adaptive allocation module is developed to automatically assign the contribution weight of each view, thereby improving the adaptability to diverse datasets. Extensive experiments on 10 benchmark datasets verify the effectiveness of $MCA^2$ against strong baselines. The source code of $MCA^2$ is available at https://github.com/yankehan/MCA2.
>
---
#### [new 041] Clustering-driven Memory Compression for On-device Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决设备端大语言模型中个性化记忆压缩问题。通过聚类合并相似记忆，提升上下文效率与生成质量。**

- **链接: [https://arxiv.org/pdf/2601.17443v1](https://arxiv.org/pdf/2601.17443v1)**

> **作者:** Ondrej Bohdal; Pramit Saha; Umberto Michieli; Mete Ozay; Taha Ceritli
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Large language models (LLMs) often rely on user-specific memories distilled from past interactions to enable personalized generation. A common practice is to concatenate these memories with the input prompt, but this approach quickly exhausts the limited context available in on-device LLMs. Compressing memories by averaging can mitigate context growth, yet it frequently harms performance due to semantic conflicts across heterogeneous memories. In this work, we introduce a clustering-based memory compression strategy that balances context efficiency and personalization quality. Our method groups memories by similarity and merges them within clusters prior to concatenation, thereby preserving coherence while reducing redundancy. Experiments demonstrate that our approach substantially lowers the number of memory tokens while outperforming baseline strategies such as naive averaging or direct concatenation. Furthermore, for a fixed context budget, clustering-driven merging yields more compact memory representations and consistently enhances generation quality.
>
---
#### [new 042] Evaluating Semantic and Syntactic Understanding in Large Language Models for Payroll Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型在薪酬系统中语义和语法理解的问题。通过实验评估模型对薪酬规则的理解与计算准确性，提出有效部署方案。**

- **链接: [https://arxiv.org/pdf/2601.18012v1](https://arxiv.org/pdf/2601.18012v1)**

> **作者:** Hendrika Maclean; Mert Can Cakmak; Muzakkiruddin Ahmed Mohammed; Shames Al Mandalawi; John Talburt
>
> **摘要:** Large language models are now used daily for writing, search, and analysis, and their natural language understanding continues to improve. However, they remain unreliable on exact numerical calculation and on producing outputs that are straightforward to audit. We study synthetic payroll system as a focused, high-stakes example and evaluate whether models can understand a payroll schema, apply rules in the right order, and deliver cent-accurate results. Our experiments span a tiered dataset from basic to complex cases, a spectrum of prompts from minimal baselines to schema-guided and reasoning variants, and multiple model families including GPT, Claude, Perplexity, Grok and Gemini. Results indicate clear regimes where careful prompting is sufficient and regimes where explicit computation is required. The work offers a compact, reproducible framework and practical guidance for deploying LLMs in settings that demand both accuracy and assurance.
>
---
#### [new 043] Mind the Ambiguity: Aleatoric Uncertainty Quantification in LLMs for Safe Medical Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗问答任务，旨在解决用户查询模糊带来的安全风险。通过分析输入模糊性与Aleatoric不确定性关系，提出一种高效澄清框架，提升医疗问答安全性。**

- **链接: [https://arxiv.org/pdf/2601.17284v1](https://arxiv.org/pdf/2601.17284v1)**

> **作者:** Yaokun Liu; Yifan Liu; Phoebe Mbuvi; Zelin Li; Ruichen Yao; Gawon Lim; Dong Wang
>
> **备注:** Accepted at The Web Conference 2026 (WWW 2026)
>
> **摘要:** The deployment of Large Language Models in Medical Question Answering is severely hampered by ambiguous user queries, a significant safety risk that demonstrably reduces answer accuracy in high-stakes healthcare settings. In this paper, we formalize this challenge by linking input ambiguity to aleatoric uncertainty (AU), which is the irreducible uncertainty arising from underspecified input. To facilitate research in this direction, we construct CV-MedBench, the first benchmark designed for studying input ambiguity in Medical QA. Using this benchmark, we analyze AU from a representation engineering perspective, revealing that AU is linearly encoded in LLM's internal activation patterns. Leveraging this insight, we introduce a novel AU-guided "Clarify-Before-Answer" framework, which incorporates AU-Probe - a lightweight module that detects input ambiguity directly from hidden states. Unlike existing uncertainty estimation methods, AU-Probe requires neither LLM fine-tuning nor multiple forward passes, enabling an efficient mechanism to proactively request user clarification and significantly enhance safety. Extensive experiments across four open LLMs demonstrate the effectiveness of our QA framework, with an average accuracy improvement of 9.48% over baselines. Our framework provides an efficient and robust solution for safe Medical QA, strengthening the reliability of health-related applications. The code is available at https://github.com/yaokunliu/AU-Med.git, and the CV-MedBench dataset is released on Hugging Face at https://huggingface.co/datasets/yaokunl/CV-MedBench.
>
---
#### [new 044] A Computational Approach to Visual Metonymy
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究视觉隐喻的计算方法，旨在解决机器理解间接视觉参考的问题。提出ViMET数据集，评估多模态模型的认知推理能力。**

- **链接: [https://arxiv.org/pdf/2601.17706v1](https://arxiv.org/pdf/2601.17706v1)**

> **作者:** Saptarshi Ghosh; Linfeng Liu; Tianyu Jiang
>
> **备注:** EACL 2026
>
> **摘要:** Images often communicate more than they literally depict: a set of tools can suggest an occupation and a cultural artifact can suggest a tradition. This kind of indirect visual reference, known as visual metonymy, invites viewers to recover a target concept via associated cues rather than explicit depiction. In this work, we present the first computational investigation of visual metonymy. We introduce a novel pipeline grounded in semiotic theory that leverages large language models and text-to-image models to generate metonymic visual representations. Using this framework, we construct ViMET, the first visual metonymy dataset comprising 2,000 multiple-choice questions to evaluate the cognitive reasoning abilities in multimodal language models. Experimental results on our dataset reveal a significant gap between human performance (86.9%) and state-of-the-art vision-language models (65.9%), highlighting limitations in machines' ability to interpret indirect visual references. Our dataset is publicly available at: https://github.com/cincynlp/ViMET.
>
---
#### [new 045] UrduLM: A Resource-Efficient Monolingual Urdu Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出UrduLM，一个针对乌尔都语的单语语言模型，解决其缺乏高质量预训练模型的问题。通过构建语料库、开发分词器并预训练模型，提升乌尔都语自然语言处理效果。**

- **链接: [https://arxiv.org/pdf/2601.17664v1](https://arxiv.org/pdf/2601.17664v1)**

> **作者:** Syed Muhammad Ali; Hammad Sajid; Zainab Haider; Ali Muhammad Asad; Haya Fatima; Abdul Samad
>
> **备注:** 12 pages
>
> **摘要:** Urdu, spoken by 230 million people worldwide, lacks dedicated transformer-based language models and curated corpora. While multilingual models provide limited Urdu support, they suffer from poor performance, high computational costs, and cultural inaccuracies due to insufficient training data. To address these challenges, we present UrduLM, a pretrained Urdu monolingual language model trained in low-resource settings. We curate a 33GB Urdu corpus from diverse sources, develop a custom BPE tokenizer that reduces tokenization overhead by atleast 20-30% compared to multilingual alternatives, and pretrain a 100M-parameter decoder-only model. In few-shot evaluations, UrduLM achieves competitive performance with multilingual models up to 30x its size, reaching 66.6% accuracy on sentiment classification and BLEU scores exceeding 30 on grammar correction tasks. The complete methodology -- including corpus, tokenizer, model weights, and evaluation benchmarks -- is released openly to establish a baseline for Urdu NLP research and provide a scalable framework for other underrepresented languages.
>
---
#### [new 046] S$^3$-Attention:Attention-Aligned Endogenous Retrieval for Memory-Bounded Long-Context Inference
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出S3-Attention，解决长文本推理中的内存和噪声效率问题。通过注意力对齐的内生检索，减少KV缓存依赖，提升推理效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.17702v1](https://arxiv.org/pdf/2601.17702v1)**

> **作者:** Qingsen Ma; Dianyun Wang; Yaoye Wang; Lechen Ning; Sujie Zhu; Xiaohang Zhang; Jiaming Lyu; Linhao Ren; Zhenbo Xu; Zhaofeng He
>
> **摘要:** Large language models are increasingly applied to multi-document and long-form inputs, yet long-context inference remains memory- and noise-inefficient. Key-value (KV) caching scales linearly with context length, while external retrieval methods often return lexically similar but causally irrelevant passages. We present S3-Attention, a memory-first inference-time framework that treats long-context processing as attention-aligned endogenous retrieval. S3-Attention decodes transient key and query projections into top-k sparse feature identifiers using lightweight sparse autoencoders, and constructs a CPU-based inverted index mapping features to token positions or spans during a single streaming scan. This design allows the KV cache to be discarded entirely and bounds GPU memory usage by the scan chunk size. At generation time, feature co-activation is used to retrieve compact evidence spans, optionally fused with BM25 for exact lexical matching. Under a unified LongBench evaluation protocol with fixed prompting, decoding, and matched token budgets, S3-Hybrid closely matches full-context inference across multiple model families and improves robustness in several information-dense settings. We also report an engineering limitation of the current prototype, which incurs higher wall-clock latency than optimized full-KV baselines, motivating future kernel-level optimization.
>
---
#### [new 047] Dynamic Role Assignment for Multi-Agent Debate
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多智能体系统任务，解决角色分配问题。提出动态角色分配框架，通过元辩论选择最适合的模型担任角色，提升问题解决效果。**

- **链接: [https://arxiv.org/pdf/2601.17152v1](https://arxiv.org/pdf/2601.17152v1)**

> **作者:** Miao Zhang; Junsik Kim; Siyuan Xiang; Jian Gao; Cheng Cao
>
> **摘要:** Multi-agent large language model (LLM) and vision-language model (VLM) debate systems employ specialized roles for complex problem-solving, yet model specializations are not leveraged to decide which model should fill which role. We propose dynamic role assignment, a framework that runs a Meta-Debate to select suitable agents before the actual debate. The meta-debate has two stages: (1) proposal, where candidates provide role-tailored arguments, and (2) peer review, where proposals are scored with data and role-specific criteria to choose the best agent for each position. We evaluate our method on LLM problem solving benchmarks. Applied on top of existing debate systems, our approach consistently outperforms uniform assignments (filling all roles with the same model) by up to 74.8% and random assignments (assigning models to roles without considering their suitability) by up to 29.7%, depending on the task and the specific assignment. This work establishes a new paradigm for multi-agent system design, shifting from static agent deployment to dynamic and capability-aware selection.
>
---
#### [new 048] A Monosemantic Attribution Framework for Stable Interpretability in Clinical Neuroscience Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型可解释性任务，旨在解决临床神经科学中大语言模型解释不稳定的问题。通过构建单义特征空间，减少方法间差异，提升解释的稳定性与可信度。**

- **链接: [https://arxiv.org/pdf/2601.17952v1](https://arxiv.org/pdf/2601.17952v1)**

> **作者:** Michail Mamalakis; Tiago Azevedo; Cristian Cosentino; Chiara D'Ercoli; Subati Abulikemu; Zhongtian Sun; Richard Bethlehem; Pietro Lio
>
> **摘要:** Interpretability remains a key challenge for deploying large language models (LLMs) in clinical settings such as Alzheimer's disease progression diagnosis, where early and trustworthy predictions are essential. Existing attribution methods exhibit high inter-method variability and unstable explanations due to the polysemantic nature of LLM representations, while mechanistic interpretability approaches lack direct alignment with model inputs and outputs and do not provide explicit importance scores. We introduce a unified interpretability framework that integrates attributional and mechanistic perspectives through monosemantic feature extraction. By constructing a monosemantic embedding space at the level of an LLM layer and optimizing the framework to explicitly reduce inter-method variability, our approach produces stable input-level importance scores and highlights salient features via a decompressed representation of the layer of interest, advancing the safe and trustworthy application of LLMs in cognitive health and neurodegenerative disease.
>
---
#### [new 049] Addressing LLM Diversity by Infusing Random Concepts
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言生成任务，旨在提升LLM输出的多样性。通过在提示中加入随机概念，实验表明可增强输出多样性，并设计了评估协议。**

- **链接: [https://arxiv.org/pdf/2601.18053v1](https://arxiv.org/pdf/2601.18053v1)**

> **作者:** Pulin Agrawal; Prasoon Goyal
>
> **摘要:** Large language models (LLMs) are known to produce outputs with limited diversity. In this work, we study whether infusing random concepts in the prompts can improve the diversity of the generated outputs. To benchmark the approach, we design a systematic evaluation protocol which involves prompting an LLM with questions of the form "Name 10 Hollywood actors", and analyzing diversity measures of the resulting LLM outputs. Our experiments on multiple LLMs show that prepending random words/sentences unrelated to the prompt result in greater diversity in the outputs of LLMs. We believe that this promising result and the evaluation protocol opens up interesting avenues for future work, such as how infusing randomness into LLMs could be applied to other domains. Further, the evaluation protocol could also inspire research into benchmarking LLM diversity more systematically.
>
---
#### [new 050] Beyond the Rabbit Hole: Mapping the Relational Harms of QAnon Radicalization
- **分类: cs.CL**

- **简介: 该论文属于社会心理学研究，旨在分析QAnon信仰对个人及家庭的伤害。通过混合方法研究，识别出六种“激进化人格”，并关联其情感影响。**

- **链接: [https://arxiv.org/pdf/2601.17658v1](https://arxiv.org/pdf/2601.17658v1)**

> **作者:** Bich Ngoc; Doan; Giuseppe Russo; Gianmarco De Francisci Morales; Robert West
>
> **摘要:** The rise of conspiracy theories has created far-reaching societal harm in the public discourse by eroding trust and fueling polarization. Beyond this public impact lies a deeply personal toll on the friends and families of conspiracy believers, a dimension often overlooked in large-scale computational research. This study fills this gap by systematically mapping radicalization journeys and quantifying the associated emotional toll inflicted on loved ones. We use the prominent case of QAnon as a case study, analyzing 12747 narratives from the r/QAnonCasualties support community through a novel mixed-methods approach. First, we use topic modeling (BERTopic) to map the radicalization trajectories, identifying key pre-existing conditions, triggers, and post-radicalization characteristics. From this, we apply an LDA-based graphical model to uncover six recurring archetypes of QAnon adherents, which we term "radicalization personas." Finally, using LLM-assisted emotion detection and regression modeling, we link these personas to the specific emotional toll reported by narrators. Our findings reveal that these personas are not just descriptive; they are powerful predictors of the specific emotional harms experienced by narrators. Radicalization perceived as a deliberate ideological choice is associated with narrator anger and disgust, while those marked by personal and cognitive collapse are linked to fear and sadness. This work provides the first empirical framework for understanding radicalization as a relational phenomenon, offering a vital roadmap for researchers and practitioners to navigate its interpersonal fallout.
>
---
#### [new 051] Fine-Grained Emotion Detection on GoEmotions: Experimental Comparison of Classical Machine Learning, BiLSTM, and Transformer Models
- **分类: cs.CL**

- **简介: 该论文研究细粒度情感检测任务，解决标签重叠和类别不平衡问题，比较了传统机器学习、BiLSTM和Transformer模型的效果。**

- **链接: [https://arxiv.org/pdf/2601.18162v1](https://arxiv.org/pdf/2601.18162v1)**

> **作者:** Ani Harutyunyan; Sachin Kumar
>
> **摘要:** Fine-grained emotion recognition is a challenging multi-label NLP task due to label overlap and class imbalance. In this work, we benchmark three modeling families on the GoEmotions dataset: a TF-IDF-based logistic regression system trained with binary relevance, a BiLSTM with attention, and a BERT model fine-tuned for multi-label classification. Experiments follow the official train/validation/test split, and imbalance is mitigated using inverse-frequency class weights. Across several metrics, namely Micro-F1, Macro-F1, Hamming Loss, and Subset Accuracy, we observe that logistic regression attains the highest Micro-F1 of 0.51, while BERT achieves the best overall balance surpassing the official paper's reported results, reaching Macro-F1 0.49, Hamming Loss 0.036, and Subset Accuracy 0.36. This suggests that frequent emotions often rely on surface lexical cues, whereas contextual representations improve performance on rarer emotions and more ambiguous examples.
>
---
#### [new 052] Revealing the Truth with ConLLM for Detecting Multi-Modal Deepfakes
- **分类: cs.CL**

- **简介: 该论文属于多模态深度伪造检测任务，旨在解决现有方法在跨模态泛化和语义不一致检测上的不足。提出ConLLM框架，结合预训练模型与对比学习，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.17530v1](https://arxiv.org/pdf/2601.17530v1)**

> **作者:** Gautam Siddharth Kashyap; Harsh Joshi; Niharika Jain; Ebad Shabbir; Jiechao Gao; Nipun Joshi; Usman Naseem
>
> **备注:** Accepted at EACL Findings 2026
>
> **摘要:** The rapid rise of deepfake technology poses a severe threat to social and political stability by enabling hyper-realistic synthetic media capable of manipulating public perception. However, existing detection methods struggle with two core limitations: (1) modality fragmentation, which leads to poor generalization across diverse and adversarial deepfake modalities; and (2) shallow inter-modal reasoning, resulting in limited detection of fine-grained semantic inconsistencies. To address these, we propose ConLLM (Contrastive Learning with Large Language Models), a hybrid framework for robust multimodal deepfake detection. ConLLM employs a two-stage architecture: stage 1 uses Pre-Trained Models (PTMs) to extract modality-specific embeddings; stage 2 aligns these embeddings via contrastive learning to mitigate modality fragmentation, and refines them using LLM-based reasoning to address shallow inter-modal reasoning by capturing semantic inconsistencies. ConLLM demonstrates strong performance across audio, video, and audio-visual modalities. It reduces audio deepfake EER by up to 50%, improves video accuracy by up to 8%, and achieves approximately 9% accuracy gains in audio-visual tasks. Ablation studies confirm that PTM-based embeddings contribute 9%-10% consistent improvements across modalities.
>
---
#### [new 053] Systematicity between Forms and Meanings across Languages Supports Efficient Communication
- **分类: cs.CL**

- **简介: 该论文属于语言学研究，探讨语言中形式与意义的系统性关系。旨在解决语言如何通过形式表达意义的问题，通过分析不同语言的动词和代词形式，建立新模型以理解高效沟通与语言系统性的联系。**

- **链接: [https://arxiv.org/pdf/2601.17181v1](https://arxiv.org/pdf/2601.17181v1)**

> **作者:** Doreen Osmelak; Yang Xu; Michael Hahn; Kate McCurdy
>
> **摘要:** Languages vary widely in how meanings map to word forms. These mappings have been found to support efficient communication; however, this theory does not account for systematic relations within word forms. We examine how a restricted set of grammatical meanings (e.g. person, number) are expressed on verbs and pronouns across typologically diverse languages. Consistent with prior work, we find that verb and pronoun forms are shaped by competing communicative pressures for simplicity (minimizing the inventory of grammatical distinctions) and accuracy (enabling recovery of intended meanings). Crucially, our proposed model uses a novel measure of complexity (inverse of simplicity) based on the learnability of meaning-to-form mappings. This innovation captures fine-grained regularities in linguistic form, allowing better discrimination between attested and unattested systems, and establishes a new connection from efficient communication theory to systematicity in natural language.
>
---
#### [new 054] ctELM: Decoding and Manipulating Embeddings of Clinical Trials with Embedding Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ctELM，用于解码和操作临床试验的嵌入表示。任务是提升嵌入空间的可解释性与生成能力，解决嵌入透明度不足的问题，通过训练模型实现临床试验的描述、比较与生成。**

- **链接: [https://arxiv.org/pdf/2601.18796v1](https://arxiv.org/pdf/2601.18796v1)**

> **作者:** Brian Ondov; Chia-Hsuan Chang; Yujia Zhou; Mauro Giuffrè; Hua Xu
>
> **摘要:** Text embeddings have become an essential part of a variety of language applications. However, methods for interpreting, exploring and reversing embedding spaces are limited, reducing transparency and precluding potentially valuable generative use cases. In this work, we align Large Language Models to embeddings of clinical trials using the recently reported Embedding Language Model (ELM) method. We develop an open-source, domain-agnostic ELM architecture and training framework, design training tasks for clinical trials, and introduce an expert-validated synthetic dataset. We then train a series of ELMs exploring the impact of tasks and training regimes. Our final model, ctELM, can accurately describe and compare unseen clinical trials from embeddings alone and produce plausible clinical trials from novel vectors. We further show that generated trial abstracts are responsive to moving embeddings along concept vectors for age and sex of study subjects. Our public ELM implementation and experimental results will aid the alignment of Large Language Models to embedding spaces in the biomedical domain and beyond.
>
---
#### [new 055] Frame-Guided Synthetic Claim Generation for Automatic Fact-Checking Using High-Volume Tabular Data
- **分类: cs.CL**

- **简介: 该论文属于自动事实核查任务，旨在解决验证声明与大规模结构化数据的挑战。研究构建了包含78,503个合成声明的数据集，并提出一种基于语义框架的方法生成真实声明，以推动模型的检索与推理能力。**

- **链接: [https://arxiv.org/pdf/2601.17232v1](https://arxiv.org/pdf/2601.17232v1)**

> **作者:** Jacob Devasier; Akshith Putta; Qing Wang; Alankrit Moses; Chengkai Li
>
> **摘要:** Automated fact-checking benchmarks have largely ignored the challenge of verifying claims against real-world, high-volume structured data, instead focusing on small, curated tables. We introduce a new large-scale, multilingual dataset to address this critical gap. It contains 78,503 synthetic claims grounded in 434 complex OECD tables, which average over 500K rows each. We propose a novel, frame-guided methodology where algorithms programmatically select significant data points based on six semantic frames to generate realistic claims in English, Chinese, Spanish, and Hindi. Crucially, we demonstrate through knowledge-probing experiments that LLMs have not memorized these facts, forcing systems to perform genuine retrieval and reasoning rather than relying on parameterized knowledge. We provide a baseline SQL-generation system and show that our benchmark is highly challenging. Our analysis identifies evidence retrieval as the primary bottleneck, with models struggling to find the correct data in massive tables. This dataset provides a critical new resource for advancing research on this unsolved, real-world problem.
>
---
#### [new 056] AI-based approach to burnout identification from textual data
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本分类任务，旨在通过NLP技术识别文本中的倦怠迹象。研究使用RuBERT模型，结合合成数据和真实评论进行微调，以检测 burnout 语言信号。**

- **链接: [https://arxiv.org/pdf/2601.17993v1](https://arxiv.org/pdf/2601.17993v1)**

> **作者:** Marina Zavertiaeva; Petr Parshakov; Mikhail Usanin; Aleksei Smirnov; Sofia Paklina; Anastasiia Kibardina
>
> **备注:** 9 pages, 2 figures
>
> **摘要:** This study introduces an AI-based methodology that utilizes natural language processing (NLP) to detect burnout from textual data. The approach relies on a RuBERT model originally trained for sentiment analysis and subsequently fine-tuned for burnout detection using two data sources: synthetic sentences generated with ChatGPT and user comments collected from Russian YouTube videos about burnout. The resulting model assigns a burnout probability to input texts and can be applied to process large volumes of written communication for monitoring burnout-related language signals in high-stress work environments.
>
---
#### [new 057] From Classification to Ranking: Enhancing LLM Reasoning Capabilities for MBTI Personality Detection
- **分类: cs.CL**

- **简介: 该论文属于人格检测任务，旨在解决传统分类方法在复杂性和细微差异上的不足。通过将任务转化为排名问题，并引入强化学习方法提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.18582v1](https://arxiv.org/pdf/2601.18582v1)**

> **作者:** Yuan Cao; Feixiang Liu; Xinyue Wang; Yihan Zhu; Hui Xu; Zheng Wang; Qiang Qiu
>
> **备注:** 9 pages, 4 figures, AAAI 2026 Bridge
>
> **摘要:** Personality detection aims to measure an individual's corresponding personality traits through their social media posts. The advancements in Large Language Models (LLMs) offer novel perspectives for personality detection tasks. Existing approaches enhance personality trait analysis by leveraging LLMs to extract semantic information from textual posts as prompts, followed by training classifiers for categorization. However, accurately classifying personality traits remains challenging due to the inherent complexity of human personality and subtle inter-trait distinctions. Moreover, prompt-based methods often exhibit excessive dependency on expert-crafted knowledge without autonomous pattern-learning capacity. To address these limitations, we view personality detection as a ranking task rather than a classification and propose a corresponding reinforcement learning training paradigm. First, we employ supervised fine-tuning (SFT) to establish personality trait ranking capabilities while enforcing standardized output formats, creating a robust initialization. Subsequently, we introduce Group Relative Policy Optimization (GRPO) with a specialized ranking-based reward function. Unlike verification tasks with definitive solutions, personality assessment involves subjective interpretations and blurred boundaries between trait categories. Our reward function explicitly addresses this challenge by training LLMs to learn optimal answer rankings. Comprehensive experiments have demonstrated that our method achieves state-of-the-art performance across multiple personality detection benchmarks.
>
---
#### [new 058] CommonLID: Re-evaluating State-of-the-Art Language Identification Performance on Web Data
- **分类: cs.CL**

- **简介: 该论文属于语言识别任务，旨在解决网页数据中语言识别模型性能不佳的问题。作者构建了CommonLID基准，涵盖109种语言，用于更准确地评估和提升LID模型性能。**

- **链接: [https://arxiv.org/pdf/2601.18026v1](https://arxiv.org/pdf/2601.18026v1)**

> **作者:** Pedro Ortiz Suarez; Laurie Burchell; Catherine Arnett; Rafael Mosquera-Gómez; Sara Hincapie-Monsalve; Thom Vaughan; Damian Stewart; Malte Ostendorff; Idris Abdulmumin; Vukosi Marivate; Shamsuddeen Hassan Muhammad; Atnafu Lambebo Tonja; Hend Al-Khalifa; Nadia Ghezaiel Hammouda; Verrah Otiende; Tack Hwa Wong; Jakhongir Saydaliev; Melika Nobakhtian; Muhammad Ravi Shulthan Habibi; Chalamalasetti Kranti; Carol Muchemi; Khang Nguyen; Faisal Muhammad Adam; Luis Frentzen Salim; Reem Alqifari; Cynthia Amol; Joseph Marvin Imperial; Ilker Kesen; Ahmad Mustafid; Pavel Stepachev; Leshem Choshen; David Anugraha; Hamada Nayel; Seid Muhie Yimam; Vallerie Alexandra Putra; My Chiffon Nguyen; Azmine Toushik Wasi; Gouthami Vadithya; Rob van der Goot; Lanwenn ar C'horr; Karan Dua; Andrew Yates; Mithil Bangera; Yeshil Bangera; Hitesh Laxmichand Patel; Shu Okabe; Fenal Ashokbhai Ilasariya; Dmitry Gaynullin; Genta Indra Winata; Yiyuan Li; Juan Pablo Martínez; Amit Agarwal; Ikhlasul Akmal Hanif; Raia Abu Ahmad; Esther Adenuga; Filbert Aurelian Tjiaranata; Weerayut Buaphet; Michael Anugraha; Sowmya Vajjala; Benjamin Rice; Azril Hafizi Amirudin; Jesujoba O. Alabi; Srikant Panda; Yassine Toughrai; Bruhan Kyomuhendo; Daniel Ruffinelli; Akshata A; Manuel Goulão; Ej Zhou; Ingrid Gabriela Franco Ramirez; Cristina Aggazzotti; Konstantin Dobler; Jun Kevin; Quentin Pagès; Nicholas Andrews; Nuhu Ibrahim; Mattes Ruckdeschel; Amr Keleg; Mike Zhang; Casper Muziri; Saron Samuel; Sotaro Takeshita; Kun Kerdthaisong; Luca Foppiano; Rasul Dent; Tommaso Green; Ahmad Mustapha Wali; Kamohelo Makaaka; Vicky Feliren; Inshirah Idris; Hande Celikkanat; Abdulhamid Abubakar; Jean Maillard; Benoît Sagot; Thibault Clérice; Kenton Murray; Sarah Luger
>
> **备注:** 17 pages, 7 tables, 5 figures
>
> **摘要:** Language identification (LID) is a fundamental step in curating multilingual corpora. However, LID models still perform poorly for many languages, especially on the noisy and heterogeneous web data often used to train multilingual language models. In this paper, we introduce CommonLID, a community-driven, human-annotated LID benchmark for the web domain, covering 109 languages. Many of the included languages have been previously under-served, making CommonLID a key resource for developing more representative high-quality text corpora. We show CommonLID's value by using it, alongside five other common evaluation sets, to test eight popular LID models. We analyse our results to situate our contribution and to provide an overview of the state of the art. In particular, we highlight that existing evaluations overestimate LID accuracy for many languages in the web domain. We make CommonLID and the code used to create it available under an open, permissive license.
>
---
#### [new 059] GLEN-Bench: A Graph-Language based Benchmark for Nutritional Health
- **分类: cs.CL**

- **简介: 该论文提出GLEN-Bench，一个基于图-语言的营养健康基准，解决个性化饮食指导不足的问题。通过整合健康数据与食物信息，构建知识图谱，完成风险检测、推荐和问答任务，提升营养干预效果。**

- **链接: [https://arxiv.org/pdf/2601.18106v1](https://arxiv.org/pdf/2601.18106v1)**

> **作者:** Jiatan Huang; Zheyuan Zhang; Tianyi Ma; Mingchen Li; Yaning Zheng; Yanfang Ye; Chuxu Zhang
>
> **摘要:** Nutritional interventions are important for managing chronic health conditions, but current computational methods provide limited support for personalized dietary guidance. We identify three key gaps: (1) dietary pattern studies often ignore real-world constraints such as socioeconomic status, comorbidities, and limited food access; (2) recommendation systems rarely explain why a particular food helps a given patient; and (3) no unified benchmark evaluates methods across the connected tasks needed for nutritional interventions. We introduce GLEN-Bench, the first comprehensive graph-language based benchmark for nutritional health assessment. We combine NHANES health records, FNDDS food composition data, and USDA food-access metrics to build a knowledge graph that links demographics, health conditions, dietary behaviors, poverty-related constraints, and nutrient needs. We test the benchmark using opioid use disorder, where models must detect subtle nutritional differences across disease stages. GLEN-Bench includes three linked tasks: risk detection identifies at-risk individuals from dietary and socioeconomic patterns; recommendation suggests personalized foods that meet clinical needs within resource constraints; and question answering provides graph-grounded, natural-language explanations to facilitate comprehension. We evaluate these graph-language approaches, including graph neural networks, large language models, and hybrid architectures, to establish solid baselines and identify practical design choices. Our analysis identifies clear dietary patterns linked to health risks, providing insights that can guide practical interventions.
>
---
#### [new 060] Exploring Fine-Tuning for In-Context Retrieval and Efficient KV-Caching in Long-Context Language Models
- **分类: cs.CL**

- **简介: 该论文研究长文本语言模型的微调策略，旨在提升其在上下文检索和KV缓存压缩下的性能与鲁棒性。任务属于自然语言处理中的模型优化领域。**

- **链接: [https://arxiv.org/pdf/2601.18527v1](https://arxiv.org/pdf/2601.18527v1)**

> **作者:** Francesco Maria Molfese; Momchil Hardalov; Rexhina Blloshmi; Bill Byrne; Adrià de Gispert
>
> **备注:** European Chapter of the Association for Computational Linguistics EACL 2026
>
> **摘要:** With context windows of millions of tokens, Long-Context Language Models (LCLMs) can encode entire document collections, offering a strong alternative to conventional retrieval-augmented generation (RAG). However, it remains unclear whether fine-tuning strategies can improve long-context performance and translate to greater robustness under KV-cache compression techniques. In this work, we investigate which training strategies most effectively enhance LCLMs' ability to identify and use relevant information, as well as enhancing their robustness under KV-cache compression. Our experiments show substantial in-domain improvements, achieving gains of up to +20 points over the base model. However, out-of-domain generalization remains task dependent with large variance -- LCLMs excels on finance questions (+9 points), while RAG shows stronger performance on multiple-choice questions (+6 points) over the baseline models. Finally, we show that our fine-tuning approaches bring moderate improvements in robustness under KV-cache compression, with gains varying across tasks.
>
---
#### [new 061] Corpus-Based Approaches to Igbo Diacritic Restoration
- **分类: cs.CL; cs.CY; cs.IR**

- **简介: 该论文属于自然语言处理中的diacritic restoration任务，旨在解决低资源语言Igbo的变音符号恢复问题。通过构建数据集并尝试n-gram、分类和嵌入模型进行研究。**

- **链接: [https://arxiv.org/pdf/2601.18380v1](https://arxiv.org/pdf/2601.18380v1)**

> **作者:** Ignatius Ezeani
>
> **备注:** 270 page. Ph.D. Thesis. The University of Sheffield
>
> **摘要:** With natural language processing (NLP), researchers aim to enable computers to identify and understand patterns in human languages. This is often difficult because a language embeds many dynamic and varied properties in its syntax, pragmatics and phonology, which need to be captured and processed. The capacity of computers to process natural languages is increasing because NLP researchers are pushing its boundaries. But these research works focus more on well-resourced languages such as English, Japanese, German, French, Russian, Mandarin Chinese, etc. Over 95% of the world's 7000 languages are low-resourced for NLP, i.e. they have little or no data, tools, and techniques for NLP work. In this thesis, we present an overview of diacritic ambiguity and a review of previous diacritic disambiguation approaches on other languages. Focusing on the Igbo language, we report the steps taken to develop a flexible framework for generating datasets for diacritic restoration. Three main approaches, the standard n-gram model, the classification models and the embedding models were proposed. The standard n-gram models use a sequence of previous words to the target stripped word as key predictors of the correct variants. For the classification models, a window of words on both sides of the target stripped word was used. The embedding models compare the similarity scores of the combined context word embeddings and the embeddings of each of the candidate variant vectors.
>
---
#### [new 062] Self-Manager: Parallel Agent Loop for Long-form Deep Research
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于深度研究任务，旨在解决传统代理在长周期任务中的上下文限制与效率问题。提出Self-Manager，通过并行代理循环实现异步执行，提升处理能力与灵活性。**

- **链接: [https://arxiv.org/pdf/2601.17879v1](https://arxiv.org/pdf/2601.17879v1)**

> **作者:** Yilong Xu; Zhi Zheng; Xiang Long; Yujun Cai; Yiwei Wang
>
> **摘要:** Long-form deep research requires multi-faceted investigations over extended horizons to get a comprehensive report. When handling such complex tasks, existing agents manage context at the subtask level to overcome linear context accumulation and information loss. However, they still adhere to a single context window and sequential execution paradigm, which results in mutual interference and blocking behavior, restricting scalability and adaptability. To address this issue, this paper introduces Self-Manager, a parallel agent loop that enables asynchronous and concurrent execution. The main thread can create multiple subthreads, each with its own isolated context, and manage them iteratively through Thread Control Blocks, allowing for more focused and flexible parallel agent execution. To assess its effectiveness, we benchmark Self-Manager on DeepResearch Bench, where it consistently outperforms existing single-agent loop baselines across all metrics. Furthermore, we conduct extensive analytical experiments to demonstrate the necessity of Self-Manager's design choices, as well as its advantages in contextual capacity, efficiency, and generalization.
>
---
#### [new 063] U-Fold: Dynamic Intent-Aware Context Folding for User-Centric Agents
- **分类: cs.CL**

- **简介: 该论文提出U-Fold，解决用户中心对话中上下文管理问题，通过动态摘要和工具日志提升长对话任务性能。**

- **链接: [https://arxiv.org/pdf/2601.18285v1](https://arxiv.org/pdf/2601.18285v1)**

> **作者:** Jin Su; Runnan Fang; Yeqiu Li; Xiaobin Wang; Shihao Cai; Pengjun Xie; Ningyu Zhang; Fajie Yuan
>
> **摘要:** Large language model (LLM)-based agents have been successfully deployed in many tool-augmented settings, but their scalability is fundamentally constrained by context length. Existing context-folding methods mitigate this issue by summarizing past interactions, yet they are typically designed for single-query or single-intent scenarios. In more realistic user-centric dialogues, we identify two major failure modes: (i) they irreversibly discard fine-grained constraints and intermediate facts that are crucial for later decisions, and (ii) their summaries fail to track evolving user intent, leading to omissions and erroneous actions. To address these limitations, we propose U-Fold, a dynamic context-folding framework tailored to user-centric tasks. U-Fold retains the full user--agent dialogue and tool-call history but, at each turn, uses two core components to produce an intent-aware, evolving dialogue summary and a compact, task-relevant tool log. Extensive experiments on $τ$-bench, $τ^2$-bench, VitaBench, and harder context-inflated settings show that U-Fold consistently outperforms ReAct (achieving a 71.4% win rate in long-context settings) and prior folding baselines (with improvements of up to 27.0%), particularly on long, noisy, multi-turn tasks. Our study demonstrates that U-Fold is a promising step toward transferring context-management techniques from single-query benchmarks to realistic user-centric applications.
>
---
#### [new 064] Align to the Pivot: Dual Alignment with Self-Feedback for Multilingual Math Reasoning
- **分类: cs.CL**

- **简介: 该论文属于多语言数学推理任务，旨在解决大语言模型在多语言环境下性能下降的问题。通过双语对齐和自反馈机制提升模型的跨语言推理能力。**

- **链接: [https://arxiv.org/pdf/2601.17671v1](https://arxiv.org/pdf/2601.17671v1)**

> **作者:** Chunxu Zhao; Xin Huang; Xue Han; Shujian Huang; Chao Deng; Junlan Feng
>
> **备注:** This paper has been accepted by ICASSP 2026
>
> **摘要:** Despite the impressive reasoning abilities demonstrated by large language models (LLMs), empirical evidence indicates that they are not language agnostic as expected, leading to performance declines in multilingual settings, especially for low-resource languages. We attribute the decline to the model's inconsistent multilingual understanding and reasoning alignment. To address this, we present Pivot-Aligned Self-Feedback Multilingual Reasoning (PASMR), aiming to improve the alignment of multilingual math reasoning abilities in LLMs. This approach designates the model's primary language as the pivot language. During training, the model first translates questions into the pivot language to facilitate better alignment of reasoning patterns. The reasoning process in the target language is then supervised by the pivot language's reasoning answers, thereby establishing a cross-lingual self-feedback mechanism without relying on external correct answers or reward models. Extensive experimental results demonstrate that our method enhances both the model's understanding of questions and its reasoning capabilities, leading to notable task improvements.
>
---
#### [new 065] GenAI for Social Work Field Education: Client Simulation with Real-Time Feedback
- **分类: cs.CL**

- **简介: 该论文属于社会工作教育任务，旨在解决培训中反馈不足的问题。通过构建SWITCH系统，实现客户模拟与实时技能评估，提升训练效果。**

- **链接: [https://arxiv.org/pdf/2601.18517v1](https://arxiv.org/pdf/2601.18517v1)**

> **作者:** James Sungarda; Hongkai Liu; Zilong Zhou; Tien-Hsuan Wu; Johnson Chun-Sing Cheung; Ben Kao
>
> **备注:** 2025 IEEE International Conference on Big Data. ISBN: 979-8-3315-9447-3/25. Page numbers: 3544-3553
>
> **摘要:** Field education is the signature pedagogy of social work, yet providing timely and objective feedback during training is constrained by the availability of instructors and counseling clients. In this paper, we present SWITCH, the Social Work Interactive Training Chatbot. SWITCH integrates realistic client simulation, real-time counseling skill classification, and a Motivational Interviewing (MI) progression system into the training workflow. To model a client, SWITCH uses a cognitively grounded profile comprising static fields (e.g., background, beliefs) and dynamic fields (e.g., emotions, automatic thoughts, openness), allowing the agent's behavior to evolve throughout a session realistically. The skill classification module identifies the counseling skills from the user utterances, and feeds the result to the MI controller that regulates the MI stage transitions. To enhance classification accuracy, we study in-context learning with retrieval over annotated transcripts, and a fine-tuned BERT multi-label classifier. In the experiments, we demonstrated that both BERT-based approach and in-context learning outperforms the baseline with big margin. SWITCH thereby offers a scalable, low-cost, and consistent training workflow that complements field education, and allows supervisors to focus on higher-level mentorship.
>
---
#### [new 066] What Language Models Know But Don't Say: Non-Generative Prior Extraction for Generalization
- **分类: cs.CL**

- **简介: 该论文属于贝叶斯推断任务，旨在解决小样本数据泛化问题。通过提取语言模型的先验分布，提升逻辑回归模型在分布外数据上的性能。**

- **链接: [https://arxiv.org/pdf/2601.17609v1](https://arxiv.org/pdf/2601.17609v1)**

> **作者:** Sara Rezaeimanesh; Mohammad M. Ghassemi
>
> **摘要:** In domains like medicine and finance, large-scale labeled data is costly and often unavailable, leading to models trained on small datasets that struggle to generalize to real-world populations. Large language models contain extensive knowledge from years of research across these domains. We propose LoID (Logit-Informed Distributions), a deterministic method for extracting informative prior distributions for Bayesian logistic regression by directly accessing their token-level predictions. Rather than relying on generated text, we probe the model's confidence in opposing semantic directions (positive vs. negative impact) through carefully constructed sentences. By measuring how consistently the LLM favors one direction across diverse phrasings, we extract the strength and reliability of the model's belief about each feature's influence. We evaluate LoID on ten real-world tabular datasets under synthetic out-of-distribution (OOD) settings characterized by covariate shift, where the training data represents only a subset of the population. We compare our approach against (1) standard uninformative priors, (2) AutoElicit, a recent method that prompts LLMs to generate priors via text completions, (3) LLMProcesses, a method that uses LLMs to generate numerical predictions through in-context learning and (4) an oracle-style upper bound derived from fitting logistic regression on the full dataset. We assess performance using Area Under the Curve (AUC). Across datasets, LoID significantly improves performance over logistic regression trained on OOD data, recovering up to \textbf{59\%} of the performance gap relative to the oracle model. LoID outperforms AutoElicit and LLMProcessesc on 8 out of 10 datasets, while providing a reproducible and computationally efficient mechanism for integrating LLM knowledge into Bayesian inference.
>
---
#### [new 067] Controlling Reading Ease with Gaze-Guided Text Generation
- **分类: cs.CL**

- **简介: 该论文属于文本生成任务，旨在控制文本的可读性。通过预测眼动模式，调整语言模型输出，以影响阅读难度。实验验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2601.17781v1](https://arxiv.org/pdf/2601.17781v1)**

> **作者:** Andreas Säuberli; Darja Jepifanova; Diego Frassinelli; Barbara Plank
>
> **备注:** Accepted for publication at EACL 2026
>
> **摘要:** The way our eyes move while reading can tell us about the cognitive effort required to process the text. In the present study, we use this fact to generate texts with controllable reading ease. Our method employs a model that predicts human gaze patterns to steer language model outputs towards eliciting certain reading behaviors. We evaluate the approach in an eye-tracking experiment with native and non-native speakers of English. The results demonstrate that the method is effective at making the generated texts easier or harder to read, measured both in terms of reading times and perceived difficulty of the texts. A statistical analysis reveals that the changes in reading behavior are mostly due to features that affect lexical processing. Possible applications of our approach include text simplification for information accessibility and generation of personalized educational material for language learning.
>
---
#### [new 068] Subword-Based Comparative Linguistics across 242 Languages Using Wikipedia Glottosets
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言学分析任务，旨在通过子词方法比较242种语言。利用维基百科构建语料库，分析词汇重叠与相似性，解决跨语言比较难题。**

- **链接: [https://arxiv.org/pdf/2601.18791v1](https://arxiv.org/pdf/2601.18791v1)**

> **作者:** Iaroslav Chelombitko; Mika Hämäläinen; Aleksey Komissarov
>
> **备注:** 15 pages, 4 figues, 4 tables
>
> **摘要:** We present a large-scale comparative study of 242 Latin and Cyrillic-script languages using subword-based methodologies. By constructing 'glottosets' from Wikipedia lexicons, we introduce a framework for simultaneous cross-linguistic comparison via Byte-Pair Encoding (BPE). Our approach utilizes rank-based subword vectors to analyze vocabulary overlap, lexical divergence, and language similarity at scale. Evaluations demonstrate that BPE segmentation aligns with morpheme boundaries 95% better than random baseline across 15 languages (F1 = 0.34 vs 0.15). BPE vocabulary similarity correlates significantly with genetic language relatedness (Mantel r = 0.329, p < 0.001), with Romance languages forming the tightest cluster (mean distance 0.51) and cross-family pairs showing clear separation (0.82). Analysis of 26,939 cross-linguistic homographs reveals that 48.7% receive different segmentations across related languages, with variation correlating to phylogenetic distance. Our results provide quantitative macro-linguistic insights into lexical patterns across typologically diverse languages within a unified analytical framework.
>
---
#### [new 069] One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization
- **分类: cs.CL**

- **简介: 该论文属于LLM偏见研究任务，旨在解决个人化中社会人口特征线索引发的不公平问题。通过比较多种线索，发现其对模型输出影响差异大，建议使用多种有效线索进行研究。**

- **链接: [https://arxiv.org/pdf/2601.18572v1](https://arxiv.org/pdf/2601.18572v1)**

> **作者:** Franziska Weeber; Vera Neplenbroek; Jan Batzner; Sebastian Padó
>
> **摘要:** Personalization of LLMs by sociodemographic subgroup often improves user experience, but can also introduce or amplify biases and unfair outcomes across groups. Prior work has employed so-called personas, sociodemographic user attributes conveyed to a model, to study bias in LLMs by relying on a single cue to prompt a persona, such as user names or explicit attribute mentions. This disregards LLM sensitivity to prompt variations (robustness) and the rarity of some cues in real interactions (external validity). We compare six commonly used persona cues across seven open and proprietary LLMs on four writing and advice tasks. While cues are overall highly correlated, they produce substantial variance in responses across personas. We therefore caution against claims from a single persona cue and recommend future personalization research to evaluate multiple externally valid cues.
>
---
#### [new 070] Assessment of Generative Named Entity Recognition in the Era of Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究生成式命名实体识别任务，评估开源大语言模型在该任务上的表现。通过实验对比传统模型，分析输出格式、记忆能力及微调影响，验证生成式方法的有效性与优势。**

- **链接: [https://arxiv.org/pdf/2601.17898v1](https://arxiv.org/pdf/2601.17898v1)**

> **作者:** Qi Zhan; Yile Wang; Hui Huang
>
> **摘要:** Named entity recognition (NER) is evolving from a sequence labeling task into a generative paradigm with the rise of large language models (LLMs). We conduct a systematic evaluation of open-source LLMs on both flat and nested NER tasks. We investigate several research questions including the performance gap between generative NER and traditional NER models, the impact of output formats, whether LLMs rely on memorization, and the preservation of general capabilities after fine-tuning. Through experiments across eight LLMs of varying scales and four standard NER datasets, we find that: (1) With parameter-efficient fine-tuning and structured formats like inline bracketed or XML, open-source LLMs achieve performance competitive with traditional encoder-based models and surpass closed-source LLMs like GPT-3; (2) The NER capability of LLMs stems from instruction-following and generative power, not mere memorization of entity-label pairs; and (3) Applying NER instruction tuning has minimal impact on general capabilities of LLMs, even improving performance on datasets like DROP due to enhanced entity understanding. These findings demonstrate that generative NER with LLMs is a promising, user-friendly alternative to traditional methods. We release the data and code at https://github.com/szu-tera/LLMs4NER.
>
---
#### [new 071] Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 论文研究LLM生成定向文本中的性别和年龄偏见，属于公平性审计任务。通过实验分析模型在不同情境下的输出差异，揭示潜在的刻板印象，旨在提升生成系统的公平性与透明度。**

- **链接: [https://arxiv.org/pdf/2601.17172v1](https://arxiv.org/pdf/2601.17172v1)**

> **作者:** Tunazzina Islam
>
> **摘要:** Large language models (LLMs) are increasingly capable of generating personalized, persuasive text at scale, raising new questions about bias and fairness in automated communication. This paper presents the first systematic analysis of how LLMs behave when tasked with demographic-conditioned targeted messaging. We introduce a controlled evaluation framework using three leading models -- GPT-4o, Llama-3.3, and Mistral-Large 2.1 -- across two generation settings: Standalone Generation, which isolates intrinsic demographic effects, and Context-Rich Generation, which incorporates thematic and regional context to emulate realistic targeting. We evaluate generated messages along three dimensions: lexical content, language style, and persuasive framing. We instantiate this framework on climate communication and find consistent age- and gender-based asymmetries across models: male- and youth-targeted messages emphasize agency, innovation, and assertiveness, while female- and senior-targeted messages stress warmth, care, and tradition. Contextual prompts systematically amplify these disparities, with persuasion scores significantly higher for messages tailored to younger or male audiences. Our findings demonstrate how demographic stereotypes can surface and intensify in LLM-generated targeted communication, underscoring the need for bias-aware generation pipelines and transparent auditing frameworks that explicitly account for demographic conditioning in socially sensitive applications.
>
---
#### [new 072] CaseFacts: A Benchmark for Legal Fact-Checking and Precedent Retrieval
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出CaseFacts，一个用于法律事实核查和判例检索的基准数据集。任务是验证非专业法律陈述是否与最高法院判例一致，解决法律领域事实核查难题。工作包括构建数据集和评估模型性能。**

- **链接: [https://arxiv.org/pdf/2601.17230v1](https://arxiv.org/pdf/2601.17230v1)**

> **作者:** Akshith Reddy Putta; Jacob Devasier; Chengkai Li
>
> **摘要:** Automated Fact-Checking has largely focused on verifying general knowledge against static corpora, overlooking high-stakes domains like law where truth is evolving and technically complex. We introduce CaseFacts, a benchmark for verifying colloquial legal claims against U.S. Supreme Court precedents. Unlike existing resources that map formal texts to formal texts, CaseFacts challenges systems to bridge the semantic gap between layperson assertions and technical jurisprudence while accounting for temporal validity. The dataset consists of 6,294 claims categorized as Supported, Refuted, or Overruled. We construct this benchmark using a multi-stage pipeline that leverages Large Language Models (LLMs) to synthesize claims from expert case summaries, employing a novel semantic similarity heuristic to efficiently identify and verify complex legal overrulings. Experiments with state-of-the-art LLMs reveal that the task remains challenging; notably, augmenting models with unrestricted web search degrades performance compared to closed-book baselines due to the retrieval of noisy, non-authoritative precedents. We release CaseFacts to spur research into legal fact verification systems.
>
---
#### [new 073] D-Models and E-Models: Diversity-Stability Trade-offs in the Sampling Behavior of Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的采样行为，探讨其多样性与稳定性之间的权衡。任务为模型采样机制分析，解决模型生成结果与任务需求匹配度问题。通过实验区分D-model和E-model，分析其在不同任务中的表现差异。**

- **链接: [https://arxiv.org/pdf/2601.17865v1](https://arxiv.org/pdf/2601.17865v1)**

> **作者:** Jia Gu; Liang Pang; Huawei Shen; Xueqi Cheng
>
> **备注:** 12 pages, 10 figures. Accepted by WWW'26
>
> **摘要:** The predictive probability of the next token (P_token) in large language models (LLMs) is inextricably linked to the probability of relevance for the next piece of information, the purchase probability of the next product, and the execution probability of the next action-all of which fall under the scope of the task-level target distribution (P_task). While LLMs are known to generate samples that approximate real-world distributions, whether their fine-grained sampling probabilities faithfully align with task requirements remains an open question. Through controlled distribution-sampling simulations, we uncover a striking dichotomy in LLM behavior, distinguishing two model types: D-models (e.g. Qwen-2.5), whose P_token exhibits large step-to-step variability and poor alignment with P_task; and E-models (e.g. Mistral-Small), whose P_token is more stable and better aligned with P_task. We further evaluate these two model types in downstream tasks such as code generation and recommendation, revealing systematic trade-offs between diversity and stability that shape task outcomes. Finally, we analyze the internal properties of both model families to probe their underlying mechanisms. These findings offer foundational insights into the probabilistic sampling behavior of LLMs and provide practical guidance on when to favor D- versus E-models. For web-scale applications, including recommendation, search, and conversational agents, our results inform model selection and configuration to balance diversity with reliability under real-world uncertainty, providing a better level of interpretation.
>
---
#### [new 074] From Emotion to Expression: Theoretical Foundations and Resources for Fear Speech
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在解决 fear speech 的研究不足问题。通过跨学科理论对比和数据梳理，提出 fear speech 的分类框架，为相关研究提供理论与数据支持。**

- **链接: [https://arxiv.org/pdf/2601.17132v1](https://arxiv.org/pdf/2601.17132v1)**

> **作者:** Vigneshwaran Shankaran; Gabriella Lapesa; Claudia Wagner
>
> **备注:** Paper accepted to EACL Mains 2026
>
> **摘要:** Few forces rival fear in their ability to mobilize societies, distort communication, and reshape collective behavior. In computational linguistics, fear is primarily studied as an emotion, but not as a distinct form of speech. Fear speech content is widespread and growing, and often outperforms hate-speech content in reach and engagement because it appears "civiler" and evades moderation. Yet the computational study of fear speech remains fragmented and under-resourced. This can be understood by recognizing that fear speech is a phenomenon shaped by contributions from multiple disciplines. In this paper, we bridge cross-disciplinary perspectives by comparing theories of fear from Psychology, Political science, Communication science, and Linguistics. Building on this, we review existing definitions. We follow up with a survey of datasets from related research areas and propose a taxonomy that consolidates different dimensions of fear for studying fear speech. By reviewing current datasets and defining core concepts, our work offers both theoretical and practical guidance for creating datasets and advancing fear speech research.
>
---
#### [new 075] Cross-Lingual Probing and Community-Grounded Analysis of Gender Bias in Low-Resource Bengali
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的性别偏见分析任务，旨在解决低资源语言 Bengali 中的性别偏见问题。通过多种方法检测并分析偏见，强调需结合本地化和社区视角进行研究。**

- **链接: [https://arxiv.org/pdf/2601.17764v1](https://arxiv.org/pdf/2601.17764v1)**

> **作者:** Md Asgor Hossain Reaj; Rajan Das Gupta; Jui Saha Pritha; Abdullah Al Noman; Abir Ahmed; Golam Md Mohiuddin; Tze Hui Liew
>
> **备注:** Accepted in 2025 4th International Conference on Smart Cities, Automation & Intelligent Computing Systems (ICON-SONICS)
>
> **摘要:** Large Language Models (LLMs) have achieved significant success in recent years; yet, issues of intrinsic gender bias persist, especially in non-English languages. Although current research mostly emphasizes English, the linguistic and cultural biases inherent in Global South languages, like Bengali, are little examined. This research seeks to examine the characteristics and magnitude of gender bias in Bengali, evaluating the efficacy of current approaches in identifying and alleviating bias. We use several methods to extract gender-biased utterances, including lexicon-based mining, computational classification models, translation-based comparison analysis, and GPT-based bias creation. Our research indicates that the straight application of English-centric bias detection frameworks to Bengali is severely constrained by language disparities and socio-cultural factors that impact implicit biases. To tackle these difficulties, we executed two field investigations inside rural and low-income areas, gathering authentic insights on gender bias. The findings demonstrate that gender bias in Bengali presents distinct characteristics relative to English, requiring a more localized and context-sensitive methodology. Additionally, our research emphasizes the need of integrating community-driven research approaches to identify culturally relevant biases often neglected by automated systems. Our research enhances the ongoing discussion around gender bias in AI by illustrating the need to create linguistic tools specifically designed for underrepresented languages. This study establishes a foundation for further investigations into bias reduction in Bengali and other Indic languages, promoting the development of more inclusive and fair NLP systems.
>
---
#### [new 076] EFT-CoT: A Multi-Agent Chain-of-Thought Framework for Emotion-Focused Therapy
- **分类: cs.CL**

- **简介: 该论文属于心理健康问答任务，旨在解决传统方法忽视情感体验的问题。提出EFT-CoT框架，通过多智能体实现情绪聚焦治疗，提升共情与专业性。**

- **链接: [https://arxiv.org/pdf/2601.17842v1](https://arxiv.org/pdf/2601.17842v1)**

> **作者:** Lanqing Du; Yunong Li; YuJie Long; Shihong Chen
>
> **摘要:** Leveraging Large Language Models (LLMs) for Mental Health Question Answering (MHQA) is promising for mitigating resource shortages. However, existing Cognitive Behavioral Therapy (CBT)-based approaches predominantly favor a "top-down" rational restructuring, often neglecting clients' embodied experiences and primary emotion processing. To address this, we propose an Emotion-Focused Therapy (EFT)-based Multi-Agent Chain-of-Thought framework (EFT-CoT). Adopting a "bottom-up" trajectory, it deconstructs the intervention into a three-stage reasoning flow: "Embodied Perception - Cognitive Exploration - Narrative Intervention." Utilizing eight specialized agents, the system explicitly executes critical components such as somatic awareness mapping, adaptive assessment, core belief extraction, and narrative restructuring. We further constructed "EFT-Instruct," a high-quality dataset via Chain-of-Thought distillation of approximately 67,000 authentic texts, and fine-tuned a specialized model, EFT-LLM. Experimental evaluations demonstrate that EFT-LLM outperforms strong baselines and human responses across metrics like empathy depth and structural professionalism. Ablation studies confirm the necessity of the multi-agent mechanism. The model exhibits superior psychological reasoning, offering an effective pathway for interpretable, high-empathy counseling systems.
>
---
#### [new 077] Temp-R1: A Unified Autonomous Agent for Complex Temporal KGQA via Reverse Curriculum Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Temp-R1，一个基于强化学习的自主代理，用于解决复杂时间知识图谱问答任务。针对现有方法灵活性差的问题，通过扩展动作空间和逆序课程学习提升推理能力。**

- **链接: [https://arxiv.org/pdf/2601.18296v1](https://arxiv.org/pdf/2601.18296v1)**

> **作者:** Zhaoyan Gong; Zhiqiang Liu; Songze Li; Xiaoke Guo; Yuanxiang Liu; Xinle Deng; Zhizhen Liu; Lei Liang; Huajun Chen; Wen Zhang
>
> **备注:** Work in progress
>
> **摘要:** Temporal Knowledge Graph Question Answering (TKGQA) is inherently challenging, as it requires sophisticated reasoning over dynamic facts with multi-hop dependencies and complex temporal constraints. Existing methods rely on fixed workflows and expensive closed-source APIs, limiting flexibility and scalability. We propose Temp-R1, the first autonomous end-to-end agent for TKGQA trained through reinforcement learning. To address cognitive overload in single-action reasoning, we expand the action space with specialized internal actions alongside external action. To prevent shortcut learning on simple questions, we introduce reverse curriculum learning that trains on difficult questions first, forcing the development of sophisticated reasoning before transferring to easier cases. Our 8B-parameter Temp-R1 achieves state-of-the-art performance on MultiTQ and TimelineKGQA, improving 19.8% over strong baselines on complex questions. Our work establishes a new paradigm for autonomous temporal reasoning agents. Our code will be publicly available soon at https://github.com/zjukg/Temp-R1.
>
---
#### [new 078] LLMs as Cultural Archives: Cultural Commonsense Knowledge Graph Extraction
- **分类: cs.CL**

- **简介: 该论文属于文化知识建模任务，旨在从大语言模型中提取结构化文化常识知识图谱，解决文化知识隐含、非结构化的问题。通过构建跨语言的文化推理链，提升文化相关自然语言处理任务性能。**

- **链接: [https://arxiv.org/pdf/2601.17971v1](https://arxiv.org/pdf/2601.17971v1)**

> **作者:** Junior Cedric Tonga; Chen Cecilia Liu; Iryna Gurevych; Fajri Koto
>
> **备注:** EACL 2026 MAIN
>
> **摘要:** Large language models (LLMs) encode rich cultural knowledge learned from diverse web-scale data, offering an unprecedented opportunity to model cultural commonsense at scale. Yet this knowledge remains mostly implicit and unstructured, limiting its interpretability and use. We present an iterative, prompt-based framework for constructing a Cultural Commonsense Knowledge Graph (CCKG) that treats LLMs as cultural archives, systematically eliciting culture-specific entities, relations, and practices and composing them into multi-step inferential chains across languages. We evaluate CCKG on five countries with human judgments of cultural relevance, correctness, and path coherence. We find that the cultural knowledge graphs are better realized in English, even when the target culture is non-English (e.g., Chinese, Indonesian, Arabic), indicating uneven cultural encoding in current LLMs. Augmenting smaller LLMs with CCKG improves performance on cultural reasoning and story generation, with the largest gains from English chains. Our results show both the promise and limits of LLMs as cultural technologies and that chain-structured cultural knowledge is a practical substrate for culturally grounded NLP.
>
---
#### [new 079] Unknown Unknowns: Why Hidden Intentions in LLMs Evade Detection
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于AI安全领域，研究LLMs中隐蔽意图的检测难题。通过构建分类体系、实验验证和案例分析，揭示隐藏意图难以检测的原因，提出改进检测框架的必要性。**

- **链接: [https://arxiv.org/pdf/2601.18552v1](https://arxiv.org/pdf/2601.18552v1)**

> **作者:** Devansh Srivastav; David Pape; Lea Schönherr
>
> **摘要:** LLMs are increasingly embedded in everyday decision-making, yet their outputs can encode subtle, unintended behaviours that shape user beliefs and actions. We refer to these covert, goal-directed behaviours as hidden intentions, which may arise from training and optimisation artefacts, or be deliberately induced by an adversarial developer, yet remain difficult to detect in practice. We introduce a taxonomy of ten categories of hidden intentions, grounded in social science research and organised by intent, mechanism, context, and impact, shifting attention from surface-level behaviours to design-level strategies of influence. We show how hidden intentions can be easily induced in controlled models, providing both testbeds for evaluation and demonstrations of potential misuse. We systematically assess detection methods, including reasoning and non-reasoning LLM judges, and find that detection collapses in realistic open-world settings, particularly under low-prevalence conditions, where false positives overwhelm precision and false negatives conceal true risks. Stress tests on precision-prevalence and precision-FNR trade-offs reveal why auditing fails without vanishingly small false positive rates or strong priors on manipulation types. Finally, a qualitative case study shows that all ten categories manifest in deployed, state-of-the-art LLMs, emphasising the urgent need for robust frameworks. Our work provides the first systematic analysis of detectability failures of hidden intentions in LLMs under open-world settings, offering a foundation for understanding, inducing, and stress-testing such behaviours, and establishing a flexible taxonomy for anticipating evolving threats and informing governance.
>
---
#### [new 080] Crystal-KV: Efficient KV Cache Management for Chain-of-Thought LLMs via Answer-First Principle
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对链式思维大语言模型的KV缓存管理问题，提出Crystal-KV框架，通过答案优先原则优化缓存效率，提升推理速度与准确性。**

- **链接: [https://arxiv.org/pdf/2601.16986v1](https://arxiv.org/pdf/2601.16986v1)**

> **作者:** Zihan Wang; Cheng Tang; Lei Gong; Cheng Li; Chao Wang; teng wang; Wenqi Lou; Xuehai Zhou
>
> **摘要:** Chain-of-Thought (CoT) reasoning in large language models (LLMs) significantly improves accuracy on complex tasks, yet incurs excessive memory overhead due to the long think-stage sequences stored in the Key-Value (KV) cache. Unlike traditional generation tasks where all tokens are uniformly important, CoT emphasizes the final answer, rendering conventional KV compression strategies ineffective. In this paper, we present Crystal-KV, an efficient KV cache management framework tailored for CoT reasoning. Our key insight is the answer-first principle. By mapping answer preferences into think-stage attention map, we distinguish between SlipKV, which mainly maintains the reasoning flow but may occasionally introduce misleading context, and CrystalKV, which truly contributes to the correctness of the final answer. Next, we propose an attention-based Least Recently Frequently Used algorithm. It precisely identifies when a SlipKV entry's utility expires and evicts it, retaining CrystalKV without disrupting reasoning flow. Finally, we introduce an adaptive cache budget allocation algorithm. Based on the dynamic proportion of CrystalKV, it estimates the importance of each layer/head and adjusts the KV cache budget during inference, amplifying critical components to improve budget utilization. Results show that Crystal-KV achieves state-of-the-art KV cache compression, significantly improves throughput, and enables faster response time, while maintaining, or even improving, answer accuracy for CoT reasoning.
>
---
#### [new 081] Grounded Concreteness: Human-Like Concreteness Sensitivity in Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文研究视觉语言模型（VLM）在文本提示下是否比纯文本模型更接近人类对词语具体性的敏感度。通过对比实验，分析了输出行为、嵌入结构和注意力机制，验证了多模态训练带来的优势。**

- **链接: [https://arxiv.org/pdf/2601.18065v1](https://arxiv.org/pdf/2601.18065v1)**

> **作者:** Aryan Roy; Zekun Wang; Christopher J. MacLellan
>
> **摘要:** Do vision--language models (VLMs) develop more human-like sensitivity to linguistic concreteness than text-only large language models (LLMs) when both are evaluated with text-only prompts? We study this question with a controlled comparison between matched Llama text backbones and their Llama Vision counterparts across multiple model scales, treating multimodal pretraining as an ablation on perceptual grounding rather than access to images at inference. We measure concreteness effects at three complementary levels: (i) output behavior, by relating question-level concreteness to QA accuracy; (ii) embedding geometry, by testing whether representations organize along a concreteness axis; and (iii) attention dynamics, by quantifying context reliance via attention-entropy measures. In addition, we elicit token-level concreteness ratings from models and evaluate alignment to human norm distributions, testing whether multimodal training yields more human-consistent judgments. Across benchmarks and scales, VLMs show larger gains on more concrete inputs, exhibit clearer concreteness-structured representations, produce ratings that better match human norms, and display systematically different attention patterns consistent with increased grounding.
>
---
#### [new 082] Do not be greedy, Think Twice: Sampling and Selection for Document-level Information Extraction
- **分类: cs.CL**

- **简介: 该论文属于文档级信息抽取任务，旨在解决输出多样性问题。通过采样与选择框架ThinkTwice，提升抽取效果，优于传统贪心解码方法。**

- **链接: [https://arxiv.org/pdf/2601.18395v1](https://arxiv.org/pdf/2601.18395v1)**

> **作者:** Mikel Zubillaga; Oscar Sainz; Oier Lopez de Lacalle; Eneko Agirre
>
> **备注:** Submitted to IJCAI-ECAI 2026
>
> **摘要:** Document-level Information Extraction (DocIE) aims to produce an output template with the entities and relations of interest occurring in the given document. Standard practices include prompting decoder-only LLMs using greedy decoding to avoid output variability. Rather than treating this variability as a limitation, we show that sampling can produce substantially better solutions than greedy decoding, especially when using reasoning models. We thus propose ThinkTwice, a sampling and selection framework in which the LLM generates multiple candidate templates for a given document, and a selection module chooses the most suitable one. We introduce both an unsupervised method that exploits agreement across generated outputs, and a supervised selection method using reward models trained on labeled DocIE data. To address the scarcity of golden reasoning trajectories for DocIE, we propose a rejection-sampling-based method to generate silver training data that pairs output templates with reasoning traces. Our experiments show the validity of unsupervised and supervised ThinkTwice, consistently outperforming greedy baselines and the state-of-the-art.
>
---
#### [new 083] Evaluating Reward Model Generalization via Pairwise Maximum Discrepancy Competitions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于奖励模型评估任务，旨在解决RM在未见场景中的泛化能力问题。提出PMDC框架，通过动态选择争议性样本评估RM性能，揭示其泛化缺陷。**

- **链接: [https://arxiv.org/pdf/2601.16987v1](https://arxiv.org/pdf/2601.16987v1)**

> **作者:** Shunyang Luo; Peibei Cao; Zhihui Zhu; Kehua Feng; Zhihua Wang; Keyan Ding
>
> **备注:** 17 pages, 6 figures, 2 tables
>
> **摘要:** Reward models (RMs) are central to aligning large language models, yet their practical effectiveness hinges on generalization to unseen prompts and shifting distributions. Most existing RM evaluations rely on static, pre-annotated preference datasets, which provide limited coverage and often fail to faithfully assess generalization in open-world settings. We introduce Pairwise Maximum Discrepancy Competition (PMDC), a dynamic and annotation-efficient framework for evaluating RM generalization using a large, unlabeled, open-domain prompt pool. PMDC actively selects prompt--response pairs that maximize disagreement between two RMs, yielding a compact set of highly contentious test cases. These cases are adjudicated by an oracle, and the resulting outcomes are aggregated via a Bradley--Terry model to produce a global ranking and pairwise win-rate landscape of RMs. We apply PMDC to re-evaluate 10 representative RMs and observe substantial rank reshuffling compared with conventional benchmarks. Qualitative analyses further uncover systematic generalization failures, providing valuable insights for improving reward modeling.
>
---
#### [new 084] On the Emergence and Test-Time Use of Structural Information in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究语言模型如何学习结构信息及在测试时利用这些信息，属于结构学习与生成任务，旨在解决模型在复杂推理和组合生成中的局限性。**

- **链接: [https://arxiv.org/pdf/2601.17869v1](https://arxiv.org/pdf/2601.17869v1)**

> **作者:** Michelle Chao Chen; Moritz Miller; Bernhard Schölkopf; Siyuan Guo
>
> **摘要:** Learning structural information from observational data is central to producing new knowledge outside the training corpus. This holds for mechanistic understanding in scientific discovery as well as flexible test-time compositional generation. We thus study how language models learn abstract structures and utilize the learnt structural information at test-time. To ensure a controlled setup, we design a natural language dataset based on linguistic structural transformations. We empirically show that the emergence of learning structural information correlates with complex reasoning tasks, and that the ability to perform test-time compositional generation remains limited.
>
---
#### [new 085] Unsupervised Elicitation of Moral Values from Language Models
- **分类: cs.CL**

- **简介: 该论文属于AI伦理任务，旨在解决如何无监督地提取语言模型中的道德价值观。通过ICM算法，验证了预训练模型具备潜在道德推理能力，并有效减少社会偏见。**

- **链接: [https://arxiv.org/pdf/2601.17728v1](https://arxiv.org/pdf/2601.17728v1)**

> **作者:** Meysam Alizadeh; Fabrizio Gilardi; Zeynab Samei
>
> **摘要:** As AI systems become pervasive, grounding their behavior in human values is critical. Prior work suggests that language models (LMs) exhibit limited inherent moral reasoning, leading to calls for explicit moral teaching. However, constructing ground truth data for moral evaluation is difficult given plural frameworks and pervasive biases. We investigate unsupervised elicitation as an alternative, asking whether pretrained (base) LMs possess intrinsic moral reasoning capability that can be surfaced without human supervision. Using the Internal Coherence Maximization (ICM) algorithm across three benchmark datasets and four LMs, we test whether ICM can reliably label moral judgments, generalize across moral frameworks, and mitigate social bias. Results show that ICM outperforms all pre-trained and chatbot baselines on the Norm Bank and ETHICS benchmarks, while fine-tuning on ICM labels performs on par with or surpasses those of human labels. Across theoretically motivated moral frameworks, ICM yields its largest relative gains on Justice and Commonsense morality. Furthermore, although chatbot LMs exhibit social bias failure rates comparable to their pretrained ones, ICM reduces such errors by more than half, with the largest improvements in race, socioeconomic status, and politics. These findings suggest that pretrained LMs possess latent moral reasoning capacities that can be elicited through unsupervised methods like ICM, providing a scalable path for AI alignment.
>
---
#### [new 086] When Domain Pretraining Interferes with Instruction Alignment: An Empirical Study of Adapter Merging in Medical LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗领域大语言模型优化任务，旨在解决医学术语精度和指令遵循问题。通过两阶段LoRA方法提升模型性能，并提出加权适配器融合策略。**

- **链接: [https://arxiv.org/pdf/2601.18350v1](https://arxiv.org/pdf/2601.18350v1)**

> **作者:** Junyi Zou
>
> **摘要:** Large language models (LLMs) show strong general capability but often struggle with medical terminology precision and safety-critical instruction following. We present a case study for adapter interference in safety-critical domains using a 14B-parameter base model through a two-stage LoRA pipeline: (1) domain-adaptive pre-training (PT) to inject broad medical knowledge via continued pre-training (DAPT), and (2) supervised fine-tuning (SFT) to align the model with medical question-answering behaviors through instruction-style data. To balance instruction-following ability and domain knowledge retention, we propose Weighted Adapter Merging, linearly combining SFT and PT adapters before exporting a merged base-model checkpoint. On a held-out medical validation set (F5/F6), the merged model achieves BLEU-4 = 16.38, ROUGE-1 = 20.42, ROUGE-2 = 4.60, and ROUGE-L = 11.54 under a practical decoding configuration. We further analyze decoding sensitivity and training stability with loss curves and controlled decoding comparisons.
>
---
#### [new 087] Linguistic and Argument Diversity in Synthetic Data for Function-Calling Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于函数调用代理的数据生成任务，旨在提升合成数据的语义和参数多样性。通过优化多样性指标，生成更高质量的训练数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.17829v1](https://arxiv.org/pdf/2601.17829v1)**

> **作者:** Dan Greenstein; Zohar Karnin; Chen Amiraz; Oren Somekh
>
> **摘要:** The construction of function calling agents has emerged as a promising avenue for extending model capabilities. A major challenge for this task is obtaining high quality diverse data for training. Prior work emphasizes diversity in functions, invocation patterns, and interaction turns, yet linguistic diversity of requests and coverage of arguments (e.g., \texttt{city\_name}, \texttt{stock\_ticker}) remain underexplored. We propose a method that generates synthetic datasets via optimizing general-purpose diversity metrics across both queries and arguments, without relying on hand-crafted rules or taxonomies, making it robust to different usecases. We demonstrate the effectiveness of our technique via both intrinsic and extrinsic testing, comparing it to SoTA data generation methods. We show a superiority over baselines in terms of diversity, while keeping comparable correctness. Additionally, when used as a training set, the model resulting from our dataset exhibits superior performance compared to analogous models based on the baseline data generation methods in out-of-distribution performance. In particular, we achieve an $7.4\%$ increase in accuracy on the BFCL benchmark compared to similar counterparts.
>
---
#### [new 088] Improving User Privacy in Personalized Generation: Client-Side Retrieval-Augmented Modification of Server-Side Generated Speculations
- **分类: cs.CL; cs.AI; cs.CR; cs.IR**

- **简介: 该论文属于个性化生成任务，旨在解决隐私泄露问题。提出P³框架，在客户端修改服务器生成的文本，提升个性化同时保护用户隐私。**

- **链接: [https://arxiv.org/pdf/2601.17569v1](https://arxiv.org/pdf/2601.17569v1)**

> **作者:** Alireza Salemi; Hamed Zamani
>
> **摘要:** Personalization is crucial for aligning Large Language Model (LLM) outputs with individual user preferences and background knowledge. State-of-the-art solutions are based on retrieval augmentation, where relevant context from a user profile is retrieved for LLM consumption. These methods deal with a trade-off between exposing retrieved private data to cloud providers and relying on less capable local models. We introduce $P^3$, an interactive framework for high-quality personalization without revealing private profiles to server-side LLMs. In $P^3$, a large server-side model generates a sequence of $k$ draft tokens based solely on the user query, while a small client-side model, with retrieval access to the user's private profile, evaluates and modifies these drafts to better reflect user preferences. This process repeats until an end token is generated. Experiments on LaMP-QA, a recent benchmark consisting of three personalized question answering datasets, show that $P^3$ consistently outperforms both non-personalized server-side and personalized client-side baselines, achieving statistically significant improvements of $7.4%$ to $9%$ on average. Importantly, $P^3$ recovers $90.3%$ to $95.7%$ of the utility of a ``leaky'' upper-bound scenario in which the full profile is exposed to the large server-side model. Privacy analyses, including linkability and attribute inference attacks, indicate that $P^3$ preserves the privacy of a non-personalized server-side model, introducing only marginal additional leakage ($1.5%$--$3.5%$) compared to submitting a query without any personal context. Additionally, the framework is efficient for edge deployment, with the client-side model generating only $9.2%$ of the total tokens. These results demonstrate that $P^3$ provides a practical, effective solution for personalized generation with improved privacy.
>
---
#### [new 089] Typhoon-S: Minimal Open Post-Training for Sovereign Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决主权语言模型在资源受限环境下实现高性能的问题。通过最小化后训练策略，提升模型的适用性和区域特定任务能力。**

- **链接: [https://arxiv.org/pdf/2601.18129v1](https://arxiv.org/pdf/2601.18129v1)**

> **作者:** Kunat Pipatanakul; Pittawat Taveekitworachai
>
> **备注:** 19 pages. Code is publicly available at https://github.com/scb-10x/typhoon-s . Datasets and model weights are available at https://huggingface.co/collections/typhoon-ai/typhoon-s
>
> **摘要:** Large language models (LLMs) have progressed rapidly; however, most state-of-the-art models are trained and evaluated primarily in high-resource languages such as English and Chinese, and are often developed by a small number of organizations with access to large-scale compute and data. This gatekeeping creates a practical barrier for sovereign settings in which a regional- or national-scale institution or domain owner must retain control and understanding of model weights, training data, and deployment while operating under limited resources and strict transparency constraints. To this end, we identify two core requirements: (1) adoptability, the ability to transform a base model into a general-purpose assistant, and (2) sovereign capability, the ability to perform high-stakes, region-specific tasks (e.g., legal reasoning in local languages and cultural knowledge). We investigate whether these requirements can be achieved without scaling massive instruction corpora or relying on complex preference tuning pipelines and large-scale reinforcement fine-tuning (RFT). We present Typhoon S, a minimal and open post-training recipe that combines supervised fine-tuning, on-policy distillation, and small-scale RFT. Using Thai as a representative case study, we demonstrate that our approach transforms both sovereign-adapted and general-purpose base models into instruction-tuned models with strong general performance. We further show that small-scale RFT with InK-GRPO -- an extension of GRPO that augments the GRPO loss with a next-word prediction loss -- improves Thai legal reasoning and Thai-specific knowledge while preserving general capabilities. Our results suggest that a carefully designed post-training strategy can reduce the required scale of instruction data and computation, providing a practical path toward high-quality sovereign LLMs under academic-scale resources.
>
---
#### [new 090] Using Large Language Models to Construct Virtual Top Managers: A Method for Organizational Research
- **分类: cs.CL**

- **简介: 该论文属于组织研究任务，旨在解决无法直接接触高管的问题。通过大语言模型构建虚拟高管角色，模拟其决策行为，验证其有效性与可靠性。**

- **链接: [https://arxiv.org/pdf/2601.18512v1](https://arxiv.org/pdf/2601.18512v1)**

> **作者:** Antonio Garzon-Vico; Krithika Sharon Komalapati; Arsalan Shahid; Jan Rosier
>
> **摘要:** This study introduces a methodological framework that uses large language models to create virtual personas of real top managers. Drawing on real CEO communications and Moral Foundations Theory, we construct LLM-based participants that simulate the decision-making of individual leaders. Across three phases, we assess construct validity, reliability, and behavioral fidelity by benchmarking these virtual CEOs against human participants. Our results indicate that theoretically scaffolded personas approximate the moral judgements observed in human samples, suggesting that LLM-based personas can serve as credible and complementary tools for organizational research in contexts where direct access to executives is limited. We conclude by outlining implications for future research using LLM-based personas in organizational settings.
>
---
#### [new 091] ShapLoRA: Allocation of Low-rank Adaption on Large Language Models via Shapley Value Inspired Importance Estimation
- **分类: cs.CL**

- **简介: 该论文属于参数高效微调任务，旨在解决LoRA中秩分配不合理的问题。通过引入Shapley敏感度方法，实现更合理的秩分配，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.17921v1](https://arxiv.org/pdf/2601.17921v1)**

> **作者:** Yi Zhao; Qinghua Yao; Xinyuan song; Wei Zhu
>
> **备注:** accepted by CPAL
>
> **摘要:** Low-rank adaption (LoRA) is a representative method in the field of parameter-efficient fine-tuning (PEFT), and is key to Democratizating the modern large language models (LLMs). The vanilla LoRA is implemented with uniform ranks, and the recent literature have found that properly allocating ranks on the LLM backbones results in performance boosts. However, the previous rank allocation methods have limitations since they rely on inexplanable and unreliable importance measures for the LoRA ranks. To address the above issues, we propose the ShapLoRA framework. Inspired by the explanable attribution measure Shapley Value, we combine the sensitivity-based measures with the idea of coalitions in the collaborative games among LoRA ranks, and propose a more explainable importance measure called Shapley sensitivity. In addition, we optimize the workflow of the existing works by: (a) calculating Shapley sensitivity on a separate validation set; (b) Setting up the allocating-retraining procedures for fair comparisons. We have conducted experiments on various challenging tasks, and the experimental results demonstrate that our ShapLoRA method can outperform the recent baselines with comparable tunable parameters.\footnote{Codes and fine-tuned models will be open-sourced to facilitate future research.
>
---
#### [new 092] Retell, Reward, Repeat: Reinforcement Learning for Narrative Theory-Informed Story Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自动故事生成任务，旨在解决传统方法依赖有限标注数据的问题。通过强化学习方法，提升生成故事的多样性和符合人类叙事规范。**

- **链接: [https://arxiv.org/pdf/2601.17226v1](https://arxiv.org/pdf/2601.17226v1)**

> **作者:** David Y. Liu; Xanthe Muston; Aditya Joshi; Sebastian Sequoiah-Grayson
>
> **备注:** 8 Pages, 6 figures
>
> **摘要:** Despite the subjective nature of storytelling, past works on automatic story generation (ASG) have relied on limited ground truths for training and evaluation. In this work, we explore reinforcement learning (d-RLAIF) as a post-training alternative to supervised fine-tuning (SFT). We first apply Todorov's Theory of Narrative Equilibrium to establish principles that define desirable ASG qualities. We prompt 7B and 14B LLM-as-judge models with our principles to test alignment with human annotators and provide reward signals during d-RLAIF. We use Gemini-3-Flash to evaluate the output of our post-trained models and compare them to human-written stories from the TimeTravel dataset. We show that d-RLAIF offers a viable alternative to supervised fine-tuning (SFT)--producing stories that are more diverse and aligned with human narrative conventions. Our paper demonstrates the promise of reinforcement learning for linguistically grounded post-training for subjective tasks such as ASG.
>
---
#### [new 093] Sparks of Cooperative Reasoning: LLMs as Strategic Hanabi Agents
- **分类: cs.CL**

- **简介: 该论文研究LLM在汉诺比游戏中协作推理的能力，解决不完全信息下的协作问题。通过不同提示设置评估模型表现，提升协作游戏性能。**

- **链接: [https://arxiv.org/pdf/2601.18077v1](https://arxiv.org/pdf/2601.18077v1)**

> **作者:** Mahesh Ramesh; Kaousheik Jayakumar; Aswinkumar Ramkumar; Pavan Thodima; Aniket Rege
>
> **摘要:** Cooperative reasoning under incomplete information remains challenging for both humans and multi-agent systems. The card game Hanabi embodies this challenge, requiring theory-of-mind reasoning and strategic communication. We benchmark 17 state-of-the-art LLM agents in 2-5 player games and study the impact of context engineering across model scales (4B to 600B+) to understand persistent coordination failures and robustness to scaffolding: from a minimal prompt with only explicit card details (Watson setting), to scaffolding with programmatic, Bayesian-motivated deductions (Sherlock setting), to multi-turn state tracking via working memory (Mycroft setting). We show that (1) agents can maintain an internal working memory for state tracking and (2) cross-play performance between different LLMs smoothly interpolates with model strength. In the Sherlock setting, the strongest reasoning models exceed 15 points on average across player counts, yet still trail experienced humans and specialist Hanabi agents, both consistently scoring above 20. We release the first public Hanabi datasets with annotated trajectories and move utilities: (1) HanabiLogs, containing 1,520 full game logs for instruction tuning, and (2) HanabiRewards, containing 560 games with dense move-level value annotations for all candidate moves. Supervised and RL finetuning of a 4B open-weight model (Qwen3-Instruct) on our datasets improves cooperative Hanabi play by 21% and 156% respectively, bringing performance to within ~3 points of a strong proprietary reasoning model (o4-mini) and surpassing the best non-reasoning model (GPT-4.1) by 52%. The HanabiRewards RL-finetuned model further generalizes beyond Hanabi, improving performance on a cooperative group-guessing benchmark by 11%, temporal reasoning on EventQA by 6.4%, instruction-following on IFBench-800K by 1.7 Pass@10, and matching AIME 2025 mathematical reasoning Pass@10.
>
---
#### [new 094] Suppressing Final Layer Hidden State Jumps in Transformer Pretraining
- **分类: cs.CL**

- **简介: 该论文研究Transformer模型中最后一层隐藏状态跳跃问题，提出JREG正则化方法抑制此现象，以提升模型性能。属于模型优化任务。**

- **链接: [https://arxiv.org/pdf/2601.18302v1](https://arxiv.org/pdf/2601.18302v1)**

> **作者:** Keigo Shibata; Kazuki Yano; Ryosuke Takahashi; Jaesung Lee; Wataru Ikeda; Jun Suzuki
>
> **备注:** Accepted to the Findings of EACL 2026
>
> **摘要:** This paper discusses the internal behavior of Transformer language models. Many recent pre-trained models have been reported to exhibit only slight changes in the angular distance between the input and output hidden state vectors in the middle Transformer layers, despite a disproportionately large ``jump'' in the angular distance occurring in or around the final Transformer layer. To characterize this, we first introduce a quantitative metric for the jump strength around the final layer, and then demonstrate its prevalence across many open-weight models, as well as its amplification throughout pre-training. Assuming such jumps indicate an undesirable property, we propose the jump-suppressing regularizer (JREG) which penalizes this jump during pre-training, thereby encouraging more balanced capability usage across the middle layers. Empirical evaluations of three model sizes of Llama-based models, trained with the proposed JREG method, reveal improved task performance compared to the baseline without altering the model architecture.
>
---
#### [new 095] Do readers prefer AI-generated Italian short stories?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本评估任务，旨在探究读者是否偏好AI生成的意大利短篇小说。通过实验比较AI与人类作者的作品，发现AI文本略受欢迎，但无显著差异。**

- **链接: [https://arxiv.org/pdf/2601.17363v1](https://arxiv.org/pdf/2601.17363v1)**

> **作者:** Michael Farrell
>
> **备注:** 7 pages
>
> **摘要:** This study investigates whether readers prefer AI-generated short stories in Italian over one written by a renowned Italian author. In a blind setup, 20 participants read and evaluated three stories, two created with ChatGPT-4o and one by Alberto Moravia, without being informed of their origin. To explore potential influencing factors, reading habits and demographic data, comprising age, gender, education and first language, were also collected. The results showed that the AI-written texts received slightly higher average ratings and were more frequently preferred, although differences were modest. No statistically significant associations were found between text preference and demographic or reading-habit variables. These findings challenge assumptions about reader preference for human-authored fiction and raise questions about the necessity of synthetic-text editing in literary contexts.
>
---
#### [new 096] Reflect: Transparent Principle-Guided Reasoning for Constitutional Alignment at Scale
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型对齐任务，旨在解决大语言模型与价值原则对齐的问题。提出Reflect框架，在推理阶段无需训练即可实现原则引导的透明推理，提升模型安全性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.18730v1](https://arxiv.org/pdf/2601.18730v1)**

> **作者:** Henry Bell; Caroline Zhang; Mohammed Mobasserul Haque; Dhaval Potdar; Samia Zaman; Brandon Fain
>
> **摘要:** The constitutional framework of alignment aims to align large language models (LLMs) with value-laden principles written in natural language (such as to avoid using biased language). Prior work has focused on parameter fine-tuning techniques, such as reinforcement learning from human feedback (RLHF), to instill these principles. However, these approaches are computationally demanding, require careful engineering and tuning, and often require difficult-to-obtain human annotation data. We propose \textsc{reflect}, an inference-time framework for constitutional alignment that does not require any training or data, providing a plug-and-play approach for aligning an instruction-tuned model to a set of principles. \textsc{reflect} operates entirely in-context, combining a (i) constitution-conditioned base response with post-generation (ii) self-evaluation, (iii)(a) self-critique, and (iii)(b) final revision. \textsc{reflect}'s technique of explicit in-context reasoning over principles during post-generation outperforms standard few-shot prompting and provides transparent reasoning traces. Our results demonstrate that \textsc{reflect} significantly improves LLM conformance to diverse and complex principles, including principles quite distinct from those emphasized in the model's original parameter fine-tuning, without sacrificing factual reasoning. \textsc{reflect} is particularly effective at reducing the rate of rare but significant violations of principles, thereby improving safety and robustness in the tail end of the distribution of generations. Finally, we show that \textsc{reflect} naturally generates useful training data for traditional parameter fine-tuning techniques, allowing for efficient scaling and the reduction of inference-time computational overhead in long-term deployment scenarios.
>
---
#### [new 097] Oops, Wait: Token-Level Signals as a Lens into LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLM推理过程中的token信号。通过分析token概率，揭示其与推理正确性的关系，解决如何理解模型推理机制的问题。**

- **链接: [https://arxiv.org/pdf/2601.17421v1](https://arxiv.org/pdf/2601.17421v1)**

> **作者:** Jaehui Hwang; Dongyoon Han; Sangdoo Yun; Byeongho Heo
>
> **摘要:** The emergence of discourse-like tokens such as "wait" and "therefore" in large language models (LLMs) has offered a unique window into their reasoning processes. However, systematic analyses of how such signals vary across training strategies and model scales remain lacking. In this paper, we analyze token-level signals through token probabilities across various models. We find that specific tokens strongly correlate with reasoning correctness, varying with training strategies while remaining stable across model scales. A closer look at the "wait" token in relation to answer probability demonstrates that models fine-tuned on small-scale datasets acquire reasoning ability through such signals but exploit them only partially. This work provides a systematic lens to observe and understand the dynamics of LLM reasoning.
>
---
#### [new 098] PEAR: Pairwise Evaluation for Automatic Relative Scoring in Machine Translation
- **分类: cs.CL**

- **简介: 该论文提出PEAR，用于机器翻译质量评估的成对比较方法，解决无参考评估问题，通过成对监督训练提升评估效果。**

- **链接: [https://arxiv.org/pdf/2601.18006v1](https://arxiv.org/pdf/2601.18006v1)**

> **作者:** Lorenzo Proietti; Roman Grundkiewicz; Matt Post
>
> **备注:** 18 pages
>
> **摘要:** We present PEAR (Pairwise Evaluation for Automatic Relative Scoring), a supervised Quality Estimation (QE) metric family that reframes reference-free Machine Translation (MT) evaluation as a graded pairwise comparison. Given a source segment and two candidate translations, PEAR predicts the direction and magnitude of their quality difference. The metrics are trained using pairwise supervision derived from differences in human judgments, with an additional regularization term that encourages sign inversion under candidate order reversal. On the WMT24 meta-evaluation benchmark, PEAR outperforms strictly matched single-candidate QE baselines trained with the same data and backbones, isolating the benefit of the proposed pairwise formulation. Despite using substantially fewer parameters than recent large metrics, PEAR surpasses far larger QE models and reference-based metrics. Our analysis further indicates that PEAR yields a less redundant evaluation signal relative to other top metrics. Finally, we show that PEAR is an effective utility function for Minimum Bayes Risk (MBR) decoding, reducing pairwise scoring cost at negligible impact.
>
---
#### [new 099] ProGraph-R1: Progress-aware Reinforcement Learning for Graph Retrieval Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于知识密集型问答任务，解决GraphRAG框架中检索依赖语义相似性、奖励稀疏的问题。提出ProGraph-R1，结合结构感知检索和进度优化策略，提升多步推理效果。**

- **链接: [https://arxiv.org/pdf/2601.17755v1](https://arxiv.org/pdf/2601.17755v1)**

> **作者:** Jinyoung Park; Sanghyeok Lee; Omar Zia Khan; Hyunwoo J. Kim; Joo-Kyung Kim
>
> **备注:** In progress
>
> **摘要:** Graph Retrieval-Augmented Generation (GraphRAG) has been successfully applied in various knowledge-intensive question answering tasks by organizing external knowledge into structured graphs of entities and relations. It enables large language models (LLMs) to perform complex reasoning beyond text-chunk retrieval. Recent works have employed reinforcement learning (RL) to train agentic GraphRAG frameworks that perform iterative interactions between LLMs and knowledge graphs. However, existing RL-based frameworks such as Graph-R1 suffer from two key limitations: (1) they primarily depend on semantic similarity for retrieval, often overlooking the underlying graph structure, and (2) they rely on sparse, outcome-level rewards, failing to capture the quality of intermediate retrieval steps and their dependencies. To address these limitations, we propose ProGraph-R1, a progress-aware agentic framework for graph-based retrieval and multi-step reasoning. ProGraph-R1 introduces a structure-aware hypergraph retrieval mechanism that jointly considers semantic relevance and graph connectivity, encouraging coherent traversal along multi-hop reasoning paths. We also design a progress-based step-wise policy optimization, which provides dense learning signals by modulating advantages according to intermediate reasoning progress within a graph, rather than relying solely on final outcomes. Experiments on multi-hop question answering benchmarks demonstrate that ProGraph-R1 consistently improves reasoning accuracy and generation quality over existing GraphRAG methods.
>
---
#### [new 100] Unsupervised Text Segmentation via Kernel Change-Point Detection on Sentence Embeddings
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于文本分割任务，解决无监督环境下边界识别难题。提出Embed-KCPD方法，通过句向量和核变点检测实现分割，并提供理论分析与实验验证。**

- **链接: [https://arxiv.org/pdf/2601.18788v1](https://arxiv.org/pdf/2601.18788v1)**

> **作者:** Mumin Jia; Jairo Diaz-Rodriguez
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2510.03437. substantial text overlap with arXiv:2510.03437. substantial text overlap with arXiv:2510.03437. substantial text overlap with arXiv:2510.03437
>
> **摘要:** Unsupervised text segmentation is crucial because boundary labels are expensive, subjective, and often fail to transfer across domains and granularity choices. We propose Embed-KCPD, a training-free method that represents sentences as embedding vectors and estimates boundaries by minimizing a penalized KCPD objective. Beyond the algorithmic instantiation, we develop, to our knowledge, the first dependence-aware theory for KCPD under $m$-dependent sequences, a finite-memory abstraction of short-range dependence common in language. We prove an oracle inequality for the population penalized risk and a localization guarantee showing that each true change point is recovered within a window that is small relative to segment length. To connect theory to practice, we introduce an LLM-based simulation framework that generates synthetic documents with controlled finite-memory dependence and known boundaries, validating the predicted scaling behavior. Across standard segmentation benchmarks, Embed-KCPD often outperforms strong unsupervised baselines. A case study on Taylor Swift's tweets illustrates that Embed-KCPD combines strong theoretical guarantees, simulated reliability, and practical effectiveness for text segmentation.
>
---
#### [new 101] Calibrating Beyond English: Language Diversity for Better Quantized Multilingual LLM
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型量化任务，旨在解决多语言大模型量化后性能下降的问题。通过实验对比不同语言的校准集效果，发现多语言校准能有效提升模型困惑度。**

- **链接: [https://arxiv.org/pdf/2601.18306v1](https://arxiv.org/pdf/2601.18306v1)**

> **作者:** Everlyn Asiko Chimoto; Mostafa Elhoushi; Bruce A. Bassett
>
> **备注:** Accepted to EACL 2026 Main Conference
>
> **摘要:** Quantization is an effective technique for reducing the storage footprint and computational costs of Large Language Models (LLMs), but it often results in performance degradation. Existing post-training quantization methods typically use small, English-only calibration sets; however, their impact on multilingual models remains underexplored. We systematically evaluate eight calibration settings (five single-language and three multilingual mixes) on two quantizers (GPTQ, AWQ) on data from 10 languages. Our findings reveal a consistent trend: non-English and multilingual calibration sets significantly improve perplexity compared to English-only baselines. Specifically, we observe notable average perplexity gains across both quantizers on Llama3.1 8B and Qwen2.5 7B, with multilingual mixes achieving the largest overall reductions of up to 3.52 points in perplexity. Furthermore, our analysis indicates that tailoring calibration sets to the evaluation language yields the largest improvements for individual languages, underscoring the importance of linguistic alignment. We also identify specific failure cases where certain language-quantizer combinations degrade performance, which we trace to differences in activation range distributions across languages. These results highlight that static one-size-fits-all calibration is suboptimal and that tailoring calibration data, both in language and diversity, plays a crucial role in robustly quantizing multilingual LLMs.
>
---
#### [new 102] Hylog: A Hybrid Approach to Logging Text Production in Non-alphabetic Scripts
- **分类: cs.CL**

- **简介: 该论文提出Hylog系统，解决非字母文字输入中日志记录不完整的问题。通过结合键盘和文本日志，实现更精确的文本生成分析。**

- **链接: [https://arxiv.org/pdf/2601.17753v1](https://arxiv.org/pdf/2601.17753v1)**

> **作者:** Roberto Crotti; Giovanni Denaro; Zhiqiang Du; Ricardo Muñoz Martín
>
> **摘要:** Research keyloggers are essential for cognitive studies of text production, yet most fail to capture the on-screen transformations performed by Input Method Editors (IMEs) for non-alphabetic scripts. To address this methodological gap, we present Hylog, a novel hybrid logging system that combines analytical keylogging with ecological text logging for a more complete and finer-grained analysis. Our modular, open-source system uses plug-ins for standard applications (Microsoft Word, Google Chrome) to capture both keyboard output and rendered text, which a hybridizer module then synchronizes into a dual trace. To validate the system's technical feasibility and demonstrate its analytical capabilities, we conducted a proof-of-concept study where two volunteers translated a text into simplified Chinese. Hylog successfully captured keypresses and temporal intervals between Latin letters, Chinese characters, and IME confirmations -- some measurements invisible to traditional keyloggers. The resulting data enable the formulation of new, testable hypotheses about the cognitive restrictions and affordances at different linguistic layers in IME-mediated typing. Our plug-in architecture enables extension to other IME systems and fosters more inclusive multilingual text-production research.
>
---
#### [new 103] Evaluating Morphological Plausibility of Subword Tokenization via Statistical Alignment with Morpho-Syntactic Features
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的子词分段评估任务，旨在解决缺乏高质量标注数据的问题。通过统计对齐方法，利用形态语法特征评估子词分割的形态合理性。**

- **链接: [https://arxiv.org/pdf/2601.18536v1](https://arxiv.org/pdf/2601.18536v1)**

> **作者:** Abishek Stephen; Jindřich Libovický
>
> **备注:** Accepted to Findings of EACL 2026, 9 pages, 6 figures
>
> **摘要:** We present a novel metric for the evaluation of the morphological plausibility of subword segmentation. Unlike the typically used morpheme boundary or retrieval F-score, which requires gold segmentation data that is either unavailable or of inconsistent quality across many languages, our approach utilizes morpho-syntactic features. These are available in resources such as Universal Dependencies or UniMorph for a much wider range of languages. The metric works by probabilistically aligning subwords with morphological features through an IBM Model 1. Our experiments show that the metric correlates well with traditional morpheme boundary recall while being more broadly applicable across languages with different morphological systems.
>
---
#### [new 104] From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for Reinforcement Learning of Open-ended Generation
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决开放生成中奖励不明确的问题。提出RLVRR方法，通过参考文本构建奖励链，结合内容与风格评估，提升生成质量与多样性。**

- **链接: [https://arxiv.org/pdf/2601.18533v1](https://arxiv.org/pdf/2601.18533v1)**

> **作者:** Yuxin Jiang; Yufei Wang; Qiyuan Zhang; Xingshan Zeng; Liangyou Li; Jierun Chen; Chaofan Tao; Haoli Bai; Lifeng Shang
>
> **备注:** 19 pages, 8 figures, 12 tables. Accepted at ICLR 2026
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) succeeds in reasoning tasks (e.g., math and code) by checking the final verifiable answer (i.e., a verifiable dot signal). However, extending this paradigm to open-ended generation is challenging because there is no unambiguous ground truth. Relying on single-dot supervision often leads to inefficiency and reward hacking. To address these issues, we propose reinforcement learning with verifiable reference-based rewards (RLVRR). Instead of checking the final answer, RLVRR extracts an ordered linguistic signal from high-quality references (i.e, reward chain). Specifically, RLVRR decomposes rewards into two dimensions: content, which preserves deterministic core concepts (e.g., keywords), and style, which evaluates adherence to stylistic properties through LLM-based verification. In this way, RLVRR combines the exploratory strength of RL with the efficiency and reliability of supervised fine-tuning (SFT). Extensive experiments on more than 10 benchmarks with Qwen and Llama models confirm the advantages of our approach. RLVRR (1) substantially outperforms SFT trained with ten times more data and advanced reward models, (2) unifies the training of structured reasoning and open-ended generation, and (3) generalizes more effectively while preserving output diversity. These results establish RLVRR as a principled and efficient path toward verifiable reinforcement learning for general-purpose LLM alignment. We release our code and data at https://github.com/YJiangcm/RLVRR.
>
---
#### [new 105] PingPong: A Natural Benchmark for Multi-Turn Code-Switching Dialogues
- **分类: cs.CL**

- **简介: 该论文提出PingPong基准，用于多轮代码切换对话研究，解决多语言交流复杂性问题，涵盖五种语言组合，定义三个下游任务进行评估。**

- **链接: [https://arxiv.org/pdf/2601.17277v1](https://arxiv.org/pdf/2601.17277v1)**

> **作者:** Mohammad Rifqi Farhansyah; Hanif Muhammad Zhafran; Farid Adilazuarda; Shamsuddeen Hassan Muhammad; Maryam Ibrahim Mukhtar; Nedjma Ousidhoum; Genta Indra Winata; Ayu Purwarianti; Alham Fikri Aji
>
> **备注:** preprint
>
> **摘要:** Code-switching is a widespread practice among the world's multilingual majority, yet few benchmarks accurately reflect its complexity in everyday communication. We present PingPong, a benchmark for natural multi-party code-switching dialogues covering five language-combination variations, some of which are trilingual. Our dataset consists of human-authored conversations among 2 to 4 participants covering authentic, multi-threaded structures where replies frequently reference much earlier points in the dialogue. We demonstrate that our data is significantly more natural and structurally diverse than machine-generated alternatives, offering greater variation in message length, speaker dominance, and reply distance. Based on these dialogues, we define three downstream tasks: Question Answering, Dialogue Summarization, and Topic Classification. Evaluations of several state-of-the-art language models on PingPong reveal that performance remains limited on code-switched inputs, underscoring the urgent need for more robust NLP systems capable of addressing the intricacies of real-world multilingual discourse.
>
---
#### [new 106] Distance-to-Distance Ratio: A Similarity Measure for Sentences Based on Rate of Change in LLM Embeddings
- **分类: cs.CL**

- **简介: 该论文属于文本相似度任务，旨在解决传统方法在语义相似性判断上的不足。提出DDR度量，通过分析嵌入变化率提升相似性判断精度。**

- **链接: [https://arxiv.org/pdf/2601.17705v1](https://arxiv.org/pdf/2601.17705v1)**

> **作者:** Abdullah Qureshi; Kenneth Rice; Alexander Wolpert
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** A measure of similarity between text embeddings can be considered adequate only if it adheres to the human perception of similarity between texts. In this paper, we introduce the distance-to-distance ratio (DDR), a novel measure of similarity between LLM sentence embeddings. Inspired by Lipschitz continuity, DDR measures the rate of change in similarity between the pre-context word embeddings and the similarity between post-context LLM embeddings, thus measuring the semantic influence of context. We evaluate the performance of DDR in experiments designed as a series of perturbations applied to sentences drawn from a sentence dataset. For each sentence, we generate variants by replacing one, two, or three words with either synonyms, which constitute semantically similar text, or randomly chosen words, which constitute semantically dissimilar text. We compare the performance of DDR with other prevailing similarity metrics and demonstrate that DDR consistently provides finer discrimination between semantically similar and dissimilar texts, even under minimal, controlled edits.
>
---
#### [new 107] MemWeaver: Weaving Hybrid Memories for Traceable Long-Horizon Agentic Reasoning
- **分类: cs.CL**

- **简介: 该论文提出MemWeaver，解决长周期智能体推理中的记忆问题，通过结构化记忆框架提升多跳和时间推理准确性。**

- **链接: [https://arxiv.org/pdf/2601.18204v1](https://arxiv.org/pdf/2601.18204v1)**

> **作者:** Juexiang Ye; Xue Li; Xinyu Yang; Chengkai Huang; Lanshun Nie; Lina Yao; Dechen Zhan
>
> **摘要:** Large language model-based agents operating in long-horizon interactions require memory systems that support temporal consistency, multi-hop reasoning, and evidence-grounded reuse across sessions. Existing approaches largely rely on unstructured retrieval or coarse abstractions, which often lead to temporal conflicts, brittle reasoning, and limited traceability. We propose MemWeaver, a unified memory framework that consolidates long-term agent experiences into three interconnected components: a temporally grounded graph memory for structured relational reasoning, an experience memory that abstracts recurring interaction patterns from repeated observations, and a passage memory that preserves original textual evidence. MemWeaver employs a dual-channel retrieval strategy that jointly retrieves structured knowledge and supporting evidence to construct compact yet information-dense contexts for reasoning. Experiments on the LoCoMo benchmark demonstrate that MemWeaver substantially improves multi-hop and temporal reasoning accuracy while reducing input context length by over 95\% compared to long-context baselines.
>
---
#### [new 108] CHiRPE: A Step Towards Real-World Clinical NLP with Clinician-Oriented Model Explanations
- **分类: cs.CL**

- **简介: 该论文提出CHiRPE系统，用于临床NLP任务中的精神病风险预测与解释。解决传统XAI方法与临床需求不匹配的问题，通过结合临床反馈生成可解释的模型输出。**

- **链接: [https://arxiv.org/pdf/2601.18102v1](https://arxiv.org/pdf/2601.18102v1)**

> **作者:** Stephanie Fong; Zimu Wang; Guilherme C. Oliveira; Xiangyu Zhao; Yiwen Jiang; Jiahe Liu; Beau-Luke Colton; Scott Woods; Martha E. Shenton; Barnaby Nelson; Zongyuan Ge; Dominic Dwyer
>
> **备注:** This paper is accepted at EACL 2026
>
> **摘要:** The medical adoption of NLP tools requires interpretability by end users, yet traditional explainable AI (XAI) methods are misaligned with clinical reasoning and lack clinician input. We introduce CHiRPE (Clinical High-Risk Prediction with Explainability), an NLP pipeline that takes transcribed semi-structured clinical interviews to: (i) predict psychosis risk; and (ii) generate novel SHAP explanation formats co-developed with clinicians. Trained on 944 semi-structured interview transcripts across 24 international clinics of the AMP-SCZ study, the CHiRPE pipeline integrates symptom-domain mapping, LLM summarisation, and BERT classification. CHiRPE achieved over 90% accuracy across three BERT variants and outperformed baseline models. Explanation formats were evaluated by 28 clinical experts who indicated a strong preference for our novel concept-guided explanations, especially hybrid graph-and-text summary formats. CHiRPE demonstrates that clinically-guided model development produces both accurate and interpretable results. Our next step is focused on real-world testing across our 24 international sites.
>
---
#### [new 109] HalluCitation Matters: Revealing the Impact of Hallucinated References with 300 Hallucinated Papers in ACL Conferences
- **分类: cs.CL; cs.AI; cs.DL**

- **简介: 该论文属于自然语言处理领域的可信性研究，旨在解决论文中虚假引用问题。通过分析ACL会议论文，发现近300篇包含虚假引用，影响学术信誉。**

- **链接: [https://arxiv.org/pdf/2601.18724v1](https://arxiv.org/pdf/2601.18724v1)**

> **作者:** Yusuke Sakai; Hidetaka Kamigaito; Taro Watanabe
>
> **备注:** Work In Progress
>
> **摘要:** Recently, we have often observed hallucinated citations or references that do not correspond to any existing work in papers under review, preprints, or published papers. Such hallucinated citations pose a serious concern to scientific reliability. When they appear in accepted papers, they may also negatively affect the credibility of conferences. In this study, we refer to hallucinated citations as "HalluCitation" and systematically investigate their prevalence and impact. We analyze all papers published at ACL, NAACL, and EMNLP in 2024 and 2025, including main conference, Findings, and workshop papers. Our analysis reveals that nearly 300 papers contain at least one HalluCitation, most of which were published in 2025. Notably, half of these papers were identified at EMNLP 2025, the most recent conference, indicating that this issue is rapidly increasing. Moreover, more than 100 such papers were accepted as main conference and Findings papers at EMNLP 2025, affecting the credibility.
>
---
#### [new 110] Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer模型在长文本中的计算效率问题。通过引入弹性注意力机制，动态调整稀疏性比例，提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2601.17367v1](https://arxiv.org/pdf/2601.17367v1)**

> **作者:** Zecheng Tang; Quantong Qiu; Yi Yang; Zhiyi Hong; Haiya Xiang; Kebin Liu; Qingqing Dang; Juntao Li; Min Zhang
>
> **摘要:** The quadratic complexity of standard attention mechanisms poses a significant scalability bottleneck for large language models (LLMs) in long-context scenarios. While hybrid attention strategies that combine sparse and full attention within a single model offer a viable solution, they typically employ static computation ratios (i.e., fixed proportions of sparse versus full attention) and fail to adapt to the varying sparsity sensitivities of downstream tasks during inference. To address this issue, we propose Elastic Attention, which allows the model to dynamically adjust its overall sparsity based on the input. This is achieved by integrating a lightweight Attention Router into the existing pretrained model, which dynamically assigns each attention head to different computation modes. Within only 12 hours of training on 8xA800 GPUs, our method enables models to achieve both strong performance and efficient inference. Experiments across three long-context benchmarks on widely-used LLMs demonstrate the superiority of our method.
>
---
#### [new 111] DPI: Exploiting Parameter Heterogeneity for Interference-Free Fine-Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的模型微调任务，解决多任务微调中的参数冲突问题。通过分离任务特定参数区域，实现无干扰的多阶段微调。**

- **链接: [https://arxiv.org/pdf/2601.17777v1](https://arxiv.org/pdf/2601.17777v1)**

> **作者:** Xiaoyu Liu; Xiaoyu Guan; Di Liang; Xianjie Wu
>
> **摘要:** Supervised fine-tuning (SFT) is a crucial step for adapting large language models (LLMs) to downstream tasks. However, conflicting objectives across heterogeneous SFT tasks often induce the "seesaw effect": optimizing for one task may degrade performance on others, particularly when model parameters are updated indiscriminately. In this paper, we propose a principled approach to disentangle and isolate task-specific parameter regions, motivated by the hypothesis that parameter heterogeneity underlies cross-task interference. Specifically, we first independently fine-tune LLMs on diverse SFT tasks and identify each task's core parameter region as the subset of parameters exhibiting the largest updates. Tasks with highly overlapping core parameter regions are merged for joint training, while disjoint tasks are organized into different stages. During multi-stage SFT, core parameters acquired in prior tasks are frozen, thereby preventing overwriting by subsequent tasks. To verify the effectiveness of our method, we conducted intensive experiments on multiple public datasets. The results showed that our dynamic parameter isolation strategy consistently reduced data conflicts and achieved consistent performance improvements compared to multi-stage and multi-task tuning baselines.
>
---
#### [new 112] Design Techniques for LLM-Powered Interactive Storytelling: A Case Study of the Dramamancer System
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于交互叙事任务，旨在解决作者意图与玩家自主性之间的平衡问题。通过Dramamancer系统，利用LLM将故事框架转化为玩家驱动的剧情。**

- **链接: [https://arxiv.org/pdf/2601.18785v1](https://arxiv.org/pdf/2601.18785v1)**

> **作者:** Tiffany Wang; Yuqian Sun; Yi Wang; Melissa Roemmele; John Joon Young Chung; Max Kreminski
>
> **备注:** Extended abstract presented at the 2025 Wordplay Workshop at EMNLP
>
> **摘要:** The rise of Large Language Models (LLMs) has enabled a new paradigm for bridging authorial intent and player agency in interactive narrative. We consider this paradigm through the example of Dramamancer, a system that uses an LLM to transform author-created story schemas into player-driven playthroughs. This extended abstract outlines some design techniques and evaluation considerations associated with this system.
>
---
#### [new 113] PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决LLM评估偏差问题。通过结合少量人工标注与LLM判断，提出PRECISE框架，降低标注需求并提高评估准确性。**

- **链接: [https://arxiv.org/pdf/2601.18777v1](https://arxiv.org/pdf/2601.18777v1)**

> **作者:** Abhishek Divekar; Anirban Majumder
>
> **备注:** Accepted at AAAI 2026 - Innovative Applications of AI (IAAI-26)
>
> **摘要:** Evaluating the quality of search, ranking and RAG systems traditionally requires a significant number of human relevance annotations. In recent times, several deployed systems have explored the usage of Large Language Models (LLMs) as automated judges for this task while their inherent biases prevent direct use for metric estimation. We present a statistical framework extending Prediction-Powered Inference (PPI) that combines minimal human annotations with LLM judgments to produce reliable estimates of metrics which require sub-instance annotations. Our method requires as few as 100 human-annotated queries and 10,000 unlabeled examples, reducing annotation requirements significantly compared to traditional approaches. We formulate our proposed framework (PRECISE) for inference of relevance uplift for an LLM-based query reformulation application, extending PPI to sub-instance annotations at the query-document level. By reformulating the metric-integration space, we reduced the computational complexity from O(2^|C|) to O(2^K), where |C| represents corpus size (in order of millions). Detailed experiments across prominent retrieval datasets demonstrate that our method reduces the variance of estimates for the business-critical Precision@K metric, while effectively correcting for LLM bias in low-resource settings.
>
---
#### [new 114] Designing large language model prompts to extract scores from messy text: A shared dataset and challenge
- **分类: cs.DL; cs.CL**

- **简介: 该论文属于信息提取任务，旨在从杂乱文本中准确提取评分。通过构建数据集和挑战，探索有效提示设计，提升大模型处理数值任务的能力。**

- **链接: [https://arxiv.org/pdf/2601.18271v1](https://arxiv.org/pdf/2601.18271v1)**

> **作者:** Mike Thelwall
>
> **摘要:** In some areas of computing, natural language processing and information science, progress is made by sharing datasets and challenging the community to design the best algorithm for an associated task. This article introduces a shared dataset of 1446 short texts, each of which describes a research quality score on the UK scale of 1* to 4*. This is a messy collection, with some texts not containing scores and others including invalid scores or strange formats. With this dataset there is also a description of what constitutes a valid score and a "gold standard" of the correct scores for these texts (including missing values). The challenge is to design a prompt for Large Language Models (LLMs) to extract the scores from these texts as accurately as possible. The format for the response should be a number and no other text so there are two aspects to the challenge: ensuring that the LLM returns only a number, and instructing it to deduce the correct number for the text. As part of this, the LLM prompt needs to explain when to return the missing value code, -1, instead of a number when the text does not clearly contain one. The article also provides an example of a simple prompt. The purpose of the challenge is twofold: to get an effective solution to this problem, and to increase understanding of prompt design and LLM capabilities for complex numerical tasks. The initial solution suggested has an accuracy of 72.6%, so the challenge is to beat this.
>
---
#### [new 115] AdaReasoner: Dynamic Tool Orchestration for Iterative Visual Reasoning
- **分类: cs.AI; cs.CL; cs.CV; cs.MA**

- **简介: 该论文提出AdaReasoner，解决多模态大模型中工具动态协调问题，通过强化学习和自适应机制实现高效视觉推理。**

- **链接: [https://arxiv.org/pdf/2601.18631v1](https://arxiv.org/pdf/2601.18631v1)**

> **作者:** Mingyang Song; Haoyu Sun; Jiawei Gu; Linjie Li; Luxin Xu; Ranjay Krishna; Yu Cheng
>
> **备注:** 28 pages, 10 figures and 13 tables
>
> **摘要:** When humans face problems beyond their immediate capabilities, they rely on tools, providing a promising paradigm for improving visual reasoning in multimodal large language models (MLLMs). Effective reasoning, therefore, hinges on knowing which tools to use, when to invoke them, and how to compose them over multiple steps, even when faced with new tools or new tasks. We introduce \textbf{AdaReasoner}, a family of multimodal models that learn tool use as a general reasoning skill rather than as tool-specific or explicitly supervised behavior. AdaReasoner is enabled by (i) a scalable data curation pipeline exposing models to long-horizon, multi-step tool interactions; (ii) Tool-GRPO, a reinforcement learning algorithm that optimizes tool selection and sequencing based on end-task success; and (iii) an adaptive learning mechanism that dynamically regulates tool usage. Together, these components allow models to infer tool utility from task context and intermediate outcomes, enabling coordination of multiple tools and generalization to unseen tools. Empirically, AdaReasoner exhibits strong tool-adaptive and generalization behaviors: it autonomously adopts beneficial tools, suppresses irrelevant ones, and adjusts tool usage frequency based on task demands, despite never being explicitly trained to do so. These capabilities translate into state-of-the-art performance across challenging benchmarks, improving the 7B base model by +24.9\% on average and surpassing strong proprietary systems such as GPT-5 on multiple tasks, including VSP and Jigsaw.
>
---
#### [new 116] Do VLMs Have a Moral Backbone? A Study on the Fragile Morality of Vision-Language Models
- **分类: cs.CY; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于AI伦理研究任务，探讨VLMs在真实场景中的道德稳定性问题。研究发现VLMs的道德判断易受扰动影响，提出需提升道德鲁棒性以确保安全部署。**

- **链接: [https://arxiv.org/pdf/2601.17082v1](https://arxiv.org/pdf/2601.17082v1)**

> **作者:** Zhining Liu; Tianyi Wang; Xiao Lin; Penghao Ouyang; Gaotang Li; Ze Yang; Hui Liu; Sumit Keswani; Vishwa Pardeshi; Huijun Zhao; Wei Fan; Hanghang Tong
>
> **摘要:** Despite substantial efforts toward improving the moral alignment of Vision-Language Models (VLMs), it remains unclear whether their ethical judgments are stable in realistic settings. This work studies moral robustness in VLMs, defined as the ability to preserve moral judgments under textual and visual perturbations that do not alter the underlying moral context. We systematically probe VLMs with a diverse set of model-agnostic multimodal perturbations and find that their moral stances are highly fragile, frequently flipping under simple manipulations. Our analysis reveals systematic vulnerabilities across perturbation types, moral domains, and model scales, including a sycophancy trade-off where stronger instruction-following models are more susceptible to persuasion. We further show that lightweight inference-time interventions can partially restore moral stability. These results demonstrate that moral alignment alone is insufficient and that moral robustness is a necessary criterion for the responsible deployment of VLMs.
>
---
#### [new 117] Noise-Robust AV-ASR Using Visual Features Both in the Whisper Encoder and Decoder
- **分类: eess.AS; cs.CL; cs.CV; cs.SD**

- **简介: 该论文属于音频视觉语音识别任务，旨在提升噪声环境下的识别鲁棒性。通过在Whisper模型的编码器和解码器中融合视觉特征，提出双用方法，显著降低错误率。**

- **链接: [https://arxiv.org/pdf/2601.18396v1](https://arxiv.org/pdf/2601.18396v1)**

> **作者:** Zhengyang Li; Thomas Graave; Björn Möller; Zehang Wu; Matthias Franz; Tim Fingscheidt
>
> **备注:** accepted at ICASSP2026
>
> **摘要:** In audiovisual automatic speech recognition (AV-ASR) systems, information fusion of visual features in a pre-trained ASR has been proven as a promising method to improve noise robustness. In this work, based on the prominent Whisper ASR, first, we propose a simple and effective visual fusion method -- use of visual features both in encoder and decoder (dual-use) -- to learn the audiovisual interactions in the encoder and to weigh modalities in the decoder. Second, we compare visual fusion methods in Whisper models of various sizes. Our proposed dual-use method shows consistent noise robustness improvement, e.g., a 35% relative improvement (WER: 4.41% vs. 6.83%) based on Whisper small, and a 57% relative improvement (WER: 4.07% vs. 9.53%) based on Whisper medium, compared to typical reference middle fusion in babble noise with a signal-to-noise ratio (SNR) of 0dB. Third, we conduct ablation studies examining the impact of various module designs and fusion options. Fine-tuned on 1929 hours of audiovisual data, our dual-use method using Whisper medium achieves 4.08% (MUSAN babble noise) and 4.43% (NoiseX babble noise) average WER across various SNRs, thereby establishing a new state-of-the-art in noisy conditions on the LRS3 AV-ASR benchmark. Our code is at https://github.com/ifnspaml/Dual-Use-AVASR
>
---
#### [new 118] Stability as a Liability:Systematic Breakdown of Linguistic Structure in LLMs
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究大语言模型训练稳定性对生成分布的影响，揭示稳定训练可能导致生成质量下降的问题。属于自然语言处理中的模型训练优化任务。**

- **链接: [https://arxiv.org/pdf/2601.18588v1](https://arxiv.org/pdf/2601.18588v1)**

> **作者:** Xianzhe Meng; Qiangsheng Zeng; Ling Luo; Qinghan Yang; Jiarui Hao; Wenbo Wu; Qinyu Wang; Rui Yin; Lin Qi; Renzhi Lu
>
> **摘要:** Training stability is typically regarded as a prerequisite for reliable optimization in large language models. In this work, we analyze how stabilizing training dynamics affects the induced generation distribution. We show that under standard maximum likelihood training, stable parameter trajectories lead stationary solutions to approximately minimize the forward KL divergence to the empirical distribution, while implicitly reducing generative entropy. As a consequence, the learned model can concentrate probability mass on a limited subset of empirical modes, exhibiting systematic degeneration despite smooth loss convergence. We empirically validate this effect using a controlled feedback-based training framework that stabilizes internal generation statistics, observing consistent low-entropy outputs and repetitive behavior across architectures and random seeds. It indicates that optimization stability and generative expressivity are not inherently aligned, and that stability alone is an insufficient indicator of generative quality.
>
---
#### [new 119] Memento: Towards Proactive Visualization of Everyday Memories with Personal Wearable AR Assistant
- **分类: cs.HC; cs.CL; cs.IR**

- **简介: 该论文提出Memento，一个主动感知上下文的AR助手，解决如何将用户记忆与情境结合的问题。通过存储交互记录，实现个性化主动信息推送。属于增强现实与上下文感知任务。**

- **链接: [https://arxiv.org/pdf/2601.17622v1](https://arxiv.org/pdf/2601.17622v1)**

> **作者:** Yoonsang Kim; Yalong Yang; Arie E. Kaufman
>
> **备注:** 8 pages, 5 figures. This is the author's version of the article that will appear at the IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (IEEE VRW) 2026
>
> **摘要:** We introduce Memento, a conversational AR assistant that permanently captures and memorizes user's verbal queries alongside their spatiotemporal and activity contexts. By storing these "memories," Memento discovers connections between users' recurring interests and the contexts that trigger them. Upon detection of similar or identical spatiotemporal activity, Memento proactively recalls user interests and delivers up-to-date responses through AR, seamlessly integrating AR experience into their daily routine. Unlike prior work, each interaction in Memento is not a transient event, but a connected series of interactions with coherent long--term perspective, tailored to the user's broader multimodal (visual, spatial, temporal, and embodied) context. We conduct preliminary evaluation through user feedbacks with participants of diverse expertise in immersive apps, and explore the value of proactive context-aware AR assistant in everyday settings. We share our findings and challenges in designing a proactive, context-aware AR system.
>
---
#### [new 120] Sentipolis: Emotion-Aware Agents for Social Simulations
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于社会模拟任务，旨在解决情感连续性不足的问题。提出Sentipolis框架，整合情感状态表示与动态机制，提升情感驱动行为的连贯性与真实性。**

- **链接: [https://arxiv.org/pdf/2601.18027v1](https://arxiv.org/pdf/2601.18027v1)**

> **作者:** Chiyuan Fu; Lyuhao Chen; Yunze Xiao; Weihao Xuan; Carlos Busso; Mona Diab
>
> **摘要:** LLM agents are increasingly used for social simulation, yet emotion is often treated as a transient cue, causing emotional amnesia and weak long-horizon continuity. We present Sentipolis, a framework for emotionally stateful agents that integrates continuous Pleasure-Arousal-Dominance (PAD) representation, dual-speed emotion dynamics, and emotion--memory coupling. Across thousands of interactions over multiple base models and evaluators, Sentipolis improves emotionally grounded behavior, boosting communication, and emotional continuity. Gains are model-dependent: believability increases for higher-capacity models but can drop for smaller ones, and emotion-awareness can mildly reduce adherence to social norms, reflecting a human-like tension between emotion-driven behavior and rule compliance in social simulation. Network-level diagnostics show reciprocal, moderately clustered, and temporally stable relationship structures, supporting the study of cumulative social dynamics such as alliance formation and gradual relationship change.
>
---
#### [new 121] AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security
- **分类: cs.AI; cs.CC; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于AI安全任务，旨在解决AI代理的复杂风险诊断问题。提出AgentDoG框架，实现细粒度安全监控与原因分析。**

- **链接: [https://arxiv.org/pdf/2601.18491v1](https://arxiv.org/pdf/2601.18491v1)**

> **作者:** Dongrui Liu; Qihan Ren; Chen Qian; Shuai Shao; Yuejin Xie; Yu Li; Zhonghao Yang; Haoyu Luo; Peng Wang; Qingyu Liu; Binxin Hu; Ling Tang; Jilin Mei; Dadi Guo; Leitao Yuan; Junyao Yang; Guanxu Chen; Qihao Lin; Yi Yu; Bo Zhang; Jiaxuan Guo; Jie Zhang; Wenqi Shao; Huiqi Deng; Zhiheng Xi; Wenjie Wang; Wenxuan Wang; Wen Shen; Zhikai Chen; Haoyu Xie; Jialing Tao; Juntao Dai; Jiaming Ji; Zhongjie Ba; Linfeng Zhang; Yong Liu; Quanshi Zhang; Lei Zhu; Zhihua Wei; Hui Xue; Chaochao Lu; Jing Shao; Xia Hu
>
> **备注:** 40 pages, 26 figures
>
> **摘要:** The rise of AI agents introduces complex safety and security challenges arising from autonomous tool use and environmental interactions. Current guardrail models lack agentic risk awareness and transparency in risk diagnosis. To introduce an agentic guardrail that covers complex and numerous risky behaviors, we first propose a unified three-dimensional taxonomy that orthogonally categorizes agentic risks by their source (where), failure mode (how), and consequence (what). Guided by this structured and hierarchical taxonomy, we introduce a new fine-grained agentic safety benchmark (ATBench) and a Diagnostic Guardrail framework for agent safety and security (AgentDoG). AgentDoG provides fine-grained and contextual monitoring across agent trajectories. More Crucially, AgentDoG can diagnose the root causes of unsafe actions and seemingly safe but unreasonable actions, offering provenance and transparency beyond binary labels to facilitate effective agent alignment. AgentDoG variants are available in three sizes (4B, 7B, and 8B parameters) across Qwen and Llama model families. Extensive experimental results demonstrate that AgentDoG achieves state-of-the-art performance in agentic safety moderation in diverse and complex interactive scenarios. All models and datasets are openly released.
>
---
#### [new 122] $\infty$-MoE: Generalizing Mixture of Experts to Infinite Experts
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出$\infty$-MoE，解决传统MoE训练困难的问题，通过连续空间选择无限专家，提升效率与性能。任务为模型压缩与优化。**

- **链接: [https://arxiv.org/pdf/2601.17680v1](https://arxiv.org/pdf/2601.17680v1)**

> **作者:** Shota Takashiro; Takeshi Kojima; Shohei Taniguchi; Yusuke Iwasawa; Yutaka Matsuo
>
> **备注:** Accepted at EACL 2026 (Main)
>
> **摘要:** The Mixture of Experts (MoE) selects a few feed-forward networks (FFNs) per token, achieving an effective trade-off between computational cost and performance. In conventional MoE, each expert is treated as entirely independent, and experts are combined in a discrete space. As a result, when the number of experts increases, it becomes difficult to train each expert effectively. To stabilize training while increasing the number of experts, we propose $\infty$-MoE that selects a portion of the parameters of large FFNs based on continuous values sampled for each token. By considering experts in a continuous space, this approach allows for an infinite number of experts while maintaining computational efficiency. Experiments show that a GPT-2 Small-based $\infty$-MoE model, with 129M active and 186M total parameters, achieves comparable performance to a dense GPT-2 Medium with 350M parameters. Adjusting the number of sampled experts at inference time allows for a flexible trade-off between accuracy and speed, with an improvement of up to 2.5\% in accuracy over conventional MoE.
>
---
#### [new 123] Unintended Memorization of Sensitive Information in Fine-Tuned Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究细调语言模型中无意记忆敏感信息的问题，属于隐私保护任务。旨在解决模型泄露PII的风险，通过实验分析影响因素并评估隐私保护方法。**

- **链接: [https://arxiv.org/pdf/2601.17480v1](https://arxiv.org/pdf/2601.17480v1)**

> **作者:** Marton Szep; Jorge Marin Ruiz; Georgios Kaissis; Paulina Seidl; Rüdiger von Eisenhart-Rothe; Florian Hinterwimmer; Daniel Rueckert
>
> **备注:** Accepted to EACL 2026. 20 pages
>
> **摘要:** Fine-tuning Large Language Models (LLMs) on sensitive datasets carries a substantial risk of unintended memorization and leakage of Personally Identifiable Information (PII), which can violate privacy regulations and compromise individual safety. In this work, we systematically investigate a critical and underexplored vulnerability: the exposure of PII that appears only in model inputs, not in training targets. Using both synthetic and real-world datasets, we design controlled extraction probes to quantify unintended PII memorization and study how factors such as language, PII frequency, task type, and model size influence memorization behavior. We further benchmark four privacy-preserving approaches including differential privacy, machine unlearning, regularization, and preference alignment, evaluating their trade-offs between privacy and task performance. Our results show that post-training methods generally provide more consistent privacy-utility trade-offs, while differential privacy achieves strong reduction in leakage in specific settings, although it can introduce training instability. These findings highlight the persistent challenge of memorization in fine-tuned LLMs and emphasize the need for robust, scalable privacy-preserving techniques.
>
---
#### [new 124] Beyond Preferences: Learning Alignment Principles Grounded in Human Reasons and Values
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于AI对齐任务，旨在解决如何公平生成反映人类价值观和偏好的AI宪法。工作包括提出GCAI框架，结合用户价值观和实时偏好生成更合理、道德的AI原则。**

- **链接: [https://arxiv.org/pdf/2601.18760v1](https://arxiv.org/pdf/2601.18760v1)**

> **作者:** Henry Bell; Lara Neubauer da Costa Schertel; Bochu Ding; Brandon Fain
>
> **摘要:** A crucial consideration when developing and deploying Large Language Models (LLMs) is the human values to which these models are aligned. In the constitutional framework of alignment models are aligned to a set of principles (the constitution) specified in natural language. However, it is unclear how to fairly determine this constitution with widespread stakeholder input. In this work we propose Grounded Constitutional AI (GCAI), a unified framework for generating constitutions of principles that are representative of both users' general expectations toward AI (general principles) and their interaction-time preferences (contextual principles). We extend the Inverse Constitutional AI (ICAI) approach to generate contextual principles from human preference annotation data by leveraging human-provided \textit{reasons} for their preferences. We supplement these contextual principles with general principles surfaced from user statements of \textit{values} regarding AI. We show that a constitution generated by GCAI is preferred by humans over one generated through ICAI both personally, and for widespread use in governing AI behavior. Additionally participants consider the GCAI constitution to be more morally grounded, coherent, and pluralistic.
>
---
#### [new 125] Can Good Writing Be Generative? Expert-Level AI Writing Emerges through Fine-Tuning on High-Quality Books
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于AI生成文本任务，探讨AI在创意写作中的表现。研究解决AI能否达到专家级写作水平的问题，通过实验对比专家与AI写作，发现Fine-tuning后AI可获得更高评价。**

- **链接: [https://arxiv.org/pdf/2601.18353v1](https://arxiv.org/pdf/2601.18353v1)**

> **作者:** Tuhin Chakrabarty; Paramveer S. Dhillon
>
> **备注:** Proceedings of CHI 2026 Conference (To Appear)
>
> **摘要:** Creative writing has long been considered a uniquely human endeavor, requiring voice and style that machines could not replicate. This assumption is challenged by Generative AI that can emulate thousands of author styles in seconds with negligible marginal labor. To understand this better, we conducted a behavioral experiment where 28 MFA writers (experts) competed against three LLMs in emulating 50 critically acclaimed authors. Based on blind pairwise comparisons by 28 expert judges and 131 lay judges, we find that experts preferred human writing in 82.7% of cases under the in-context prompting condition but this reversed to 62% preference for AI after fine-tuning on authors' complete works. Lay judges, however, consistently preferred AI writing. Debrief interviews with expert writers revealed that their preference for AI writing triggered an identity crisis, eroding aesthetic confidence and questioning what constitutes "good writing." These findings challenge discourse about AI's creative limitations and raise fundamental questions about the future of creative labor.
>
---
#### [new 126] Generative AI in Saudi Arabia: A National Survey of Adoption, Risks, and Public Perceptions
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于人工智能应用研究，旨在了解沙特阿拉伯民众对生成式AI的使用情况与态度。通过全国调查，分析了用户认知、使用模式及风险担忧，为政策制定提供依据。**

- **链接: [https://arxiv.org/pdf/2601.18234v1](https://arxiv.org/pdf/2601.18234v1)**

> **作者:** Abdulaziz AlDakheel; Ali Alshehre; Esraa Alamoudi; Moslim AlKhabbaz; Ahmed Aljohani; Raed Alharbi
>
> **摘要:** Generative Artificial Intelligence (GenAI) is rapidly becoming embedded in Saudi Arabia's digital transformation under Vision 2030, yet public awareness, adoption, and concerns surrounding these tools remain underexplored. This study provides an early snapshot of GenAI engagement among Saudi nationals. Using a nationwide survey of 330 participants across regions, age groups, and employment sectors, we examine seven dimensions of GenAI use: awareness and understanding, adoption patterns, perceived impacts, training needs, risks and barriers, data-sharing behaviors, and future expectations. Findings show that 93% of respondents actively use GenAI primarily for text-based tasks, while more advanced uses such as programming or multimodal generation are less common. Despite the prevalence of use, overall awareness and conceptual understanding remain uneven, with many reporting limited technical knowledge. Participants recognize GenAI's benefits for productivity, work quality, and understanding complex information, yet caution that sustained reliance may undermine critical thinking and key professional skills. Trust in AI-generated outputs remains cautious, with widespread concerns about privacy, misinformation, and ethical misuse, including potential job displacement. Respondents show strong interest in structured GenAI training that combines foundational skills, domain-specific applications, and clear guidance on privacy, ethics, and responsible use. These results establish a baseline for GenAI engagement in Saudi Arabia and highlight priorities for policymakers and developers: expanding AI literacy, ensuring culturally and linguistically aligned GenAI solutions, and strengthening frameworks for privacy and responsible deployment.
>
---
#### [new 127] BanglaRobustNet: A Hybrid Denoising-Attention Architecture for Robust Bangla Speech Recognition
- **分类: cs.SD; cs.CL; cs.CV; cs.LG; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决噪声和说话人多样性下的孟加拉语语音识别问题。提出BanglaRobustNet框架，结合降噪与注意力机制，提升识别鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.17679v1](https://arxiv.org/pdf/2601.17679v1)**

> **作者:** Md Sazzadul Islam Ridoy; Mubaswira Ibnat Zidney; Sumi Akter; Md. Aminur Rahman
>
> **摘要:** Bangla, one of the most widely spoken languages, remains underrepresented in state-of-the-art automatic speech recognition (ASR) research, particularly under noisy and speaker-diverse conditions. This paper presents BanglaRobustNet, a hybrid denoising-attention framework built on Wav2Vec-BERT, designed to address these challenges. The architecture integrates a diffusion-based denoising module to suppress environmental noise while preserving Bangla-specific phonetic cues, and a contextual cross-attention module that conditions recognition on speaker embeddings for robustness across gender, age, and dialects. Trained end-to-end with a composite objective combining CTC loss, phonetic consistency, and speaker alignment, BanglaRobustNet achieves substantial reductions in word error rate (WER) and character error rate (CER) compared to Wav2Vec-BERT and Whisper baselines. Evaluations on Mozilla Common Voice Bangla and augmented noisy speech confirm the effectiveness of our approach, establishing BanglaRobustNet as a robust ASR system tailored to low-resource, noise-prone linguistic settings.
>
---
#### [new 128] Artificial Intelligence and Intellectual Property Rights: Comparative Transnational Policy Analysis
- **分类: cs.CY; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于法律政策分析任务，探讨AI与知识产权的衔接问题。针对印度法律在AI生成成果保护上的不足，通过比较研究提出完善建议。**

- **链接: [https://arxiv.org/pdf/2601.17892v1](https://arxiv.org/pdf/2601.17892v1)**

> **作者:** Sahibpreet Singh; Manjit Singh
>
> **备注:** Published in Journal of University Institute of Legal Studies, Vol. 19, Issue 1, pp. 182-208, 2025
>
> **摘要:** Artificial intelligence's rapid integration with intellectual property rights necessitates assessment of its impact on trade secrets, copyrights and patents. This study addresses lacunae in existing laws where India lacks AI-specific provisions, creating doctrinal inconsistencies and enforcement inefficacies. Global discourse on AI-IPR protections remains nascent. The research identifies gaps in Indian IP laws' adaptability to AI-generated outputs: trade secret protection is inadequate against AI threats; standardized inventorship criteria are absent. Employing doctrinal and comparative methodology, it scrutinizes legislative texts, judicial precedents and policy instruments across India, US, UK and EU. Preliminary findings reveal shortcomings: India's contract law creates fragmented trade secret regime; Section 3(k) of Indian Patents Act blocks AI invention patenting; copyright varies in authorship attribution. The study proposes harmonized legal taxonomy accommodating AI's role while preserving innovation incentives. India's National AI Strategy (2024) shows progress but legislative clarity is imperative. This contributes to global discourse with AI-specific IP protections ensuring resilience and equitable innovation. Promising results underscore recalibrating India's IP jurisprudence for global alignment.
>
---
#### [new 129] The 17% Gap: Quantifying Epistemic Decay in AI-Assisted Survey Papers
- **分类: cs.CY; cs.AI; cs.CL; cs.DL**

- **简介: 该论文属于信息验证任务，旨在解决AI辅助论文中引用失效问题。通过分析50篇AI综述论文，发现17%的引用无法验证，揭示了AI工具在引用处理中的缺陷。**

- **链接: [https://arxiv.org/pdf/2601.17431v1](https://arxiv.org/pdf/2601.17431v1)**

> **作者:** H. Kemal İlter
>
> **摘要:** The adoption of Large Language Models (LLMs) in scientific writing promises efficiency but risks introducing informational entropy. While "hallucinated papers" are a known artifact, the systematic degradation of valid citation chains remains unquantified. We conducted a forensic audit of 50 recent survey papers in Artificial Intelligence (N=5,514 citations) published between September 2024 and January 2026. We utilized a hybrid verification pipeline combining DOI resolution, Crossref metadata analysis, Semantic Scholar queries, and fuzzy text matching to distinguish between formatting errors ("Sloppiness") and verifiable non-existence ("Phantoms). We detect a persistent 17.0% Phantom Rate -- citations that cannot be resolved to any digital object despite aggressive forensic recovery. Diagnostic categorization reveals three distinct failure modes: pure hallucinations (5.1%), hallucinated identifiers with valid titles (16.4%), and parsing-induced matching failures (78.5%). Longitudinal analysis reveals a flat trend (+0.07 pp/month), suggesting that high-entropy citation practices have stabilized as an endemic feature of the field. The scientific citation graph in AI survey literature exhibits "link rot" at scale. This suggests a mechanism where AI tools act as "lazy research assistants," retrieving correct titles but hallucinating metadata, thereby severing the digital chain of custody required for reproducible science.
>
---
#### [new 130] LegalMALR:Multi-Agent Query Understanding and LLM-Based Reranking for Chinese Statute Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于法律信息检索任务，解决法律查询隐晦、多问题及表达不明确的问题。提出LegalMALR框架，结合多代理查询理解和大模型重排序，提升法规检索效果。**

- **链接: [https://arxiv.org/pdf/2601.17692v1](https://arxiv.org/pdf/2601.17692v1)**

> **作者:** Yunhan Li; Mingjie Xie; Gaoli Kang; Zihan Gong; Gengshen Wu; Min Yang
>
> **备注:** 31pages, 4 figures
>
> **摘要:** Statute retrieval is essential for legal assistance and judicial decision support, yet real-world legal queries are often implicit, multi-issue, and expressed in colloquial or underspecified forms. These characteristics make it difficult for conventional retrieval-augmented generation pipelines to recover the statutory elements required for accurate retrieval. Dense retrievers focus primarily on the literal surface form of the query, whereas lightweight rerankers lack the legal-reasoning capacity needed to assess statutory applicability. We present LegalMALR, a retrieval framework that integrates a Multi-Agent Query Understanding System (MAS) with a zero-shot large-language-model-based reranking module (LLM Reranker). MAS generates diverse, legally grounded reformulations and conducts iterative dense retrieval to broaden candidate coverage. To stabilise the stochastic behaviour of LLM-generated rewrites, we optimise a unified MAS policy using Generalized Reinforcement Policy Optimization(GRPO). The accumulated candidate set is subsequently evaluated by the LLM Reranker, which performs natural-language legal reasoning to produce the final ranking. We further construct CSAID, a dataset of 118 difficult Chinese legal queries annotated with multiple statutory labels, and evaluate LegalMALR on both CSAID and the public STARD benchmark. Experiments show that LegalMALR substantially outperforms strong Retrieval-augmented generation(RAG) baselines in both in-distribution and out-of-distribution settings, demonstrating the effectiveness of combining multi-perspective query interpretation, reinforcement-based policy optimisation, and large-model reranking for statute retrieval.
>
---
#### [new 131] ThinkTank-ME: A Multi-Expert Framework for Middle East Event Forecasting
- **分类: cs.LG; cs.AI; cs.CL; cs.CY; cs.MA**

- **简介: 该论文属于事件预测任务，旨在解决单一模型难以捕捉复杂地缘政治细节的问题。提出多专家框架ThinkTank-ME，通过协作提升预测效果。**

- **链接: [https://arxiv.org/pdf/2601.17065v1](https://arxiv.org/pdf/2601.17065v1)**

> **作者:** Haoxuan Li; He Chang; Yunshan Ma; Yi Bin; Yang Yang; See-Kiong Ng; Tat-Seng Chua
>
> **摘要:** Event forecasting is inherently influenced by multifaceted considerations, including international relations, regional historical dynamics, and cultural contexts. However, existing LLM-based approaches employ single-model architectures that generate predictions along a singular explicit trajectory, constraining their ability to capture diverse geopolitical nuances across complex regional contexts. To address this limitation, we introduce ThinkTank-ME, a novel Think Tank framework for Middle East event forecasting that emulates collaborative expert analysis in real-world strategic decision-making. To facilitate expert specialization and rigorous evaluation, we construct POLECAT-FOR-ME, a Middle East-focused event forecasting benchmark. Experimental results demonstrate the superiority of multi-expert collaboration in handling complex temporal geopolitical forecasting tasks. The code is available at https://github.com/LuminosityX/ThinkTank-ME.
>
---
#### [new 132] DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出DeepPlanning，一个用于长周期智能体规划的基准任务，解决真实场景下的信息获取与约束优化问题，强调全局与局部约束的平衡。**

- **链接: [https://arxiv.org/pdf/2601.18137v1](https://arxiv.org/pdf/2601.18137v1)**

> **作者:** Yinger Zhang; Shutong Jiang; Renhao Li; Jianhong Tu; Yang Su; Lianghao Deng; Xudong Guo; Chenxu Lv; Junyang Lin
>
> **摘要:** While agent evaluation has shifted toward long-horizon tasks, most benchmarks still emphasize local, step-level reasoning rather than the global constrained optimization (e.g., time and financial budgets) that demands genuine planning ability. Meanwhile, existing LLM planning benchmarks underrepresent the active information gathering and fine-grained local constraints typical of real-world settings. To address this, we introduce DeepPlanning, a challenging benchmark for practical long-horizon agent planning. It features multi-day travel planning and multi-product shopping tasks that require proactive information acquisition, local constrained reasoning, and global constrained optimization. Evaluations on DeepPlanning show that even frontier agentic LLMs struggle with these problems, highlighting the importance of reliable explicit reasoning patterns and parallel tool use for achieving better effectiveness-efficiency trade-offs. Error analysis further points to promising directions for improving agentic LLMs over long planning horizons. We open-source the code and data to support future research.
>
---
#### [new 133] Reuse your FLOPs: Scaling RL on Hard Problems by Conditioning on Very Off-Policy Prefixes
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决大模型推理中计算浪费问题。通过重用离策略前缀，提升学习效率与效果。**

- **链接: [https://arxiv.org/pdf/2601.18795v1](https://arxiv.org/pdf/2601.18795v1)**

> **作者:** Amrith Setlur; Zijian Wang; Andrew Cohen; Paria Rashidinejad; Sang Michael Xie
>
> **摘要:** Typical reinforcement learning (RL) methods for LLM reasoning waste compute on hard problems, where correct on-policy traces are rare, policy gradients vanish, and learning stalls. To bootstrap more efficient RL, we consider reusing old sampling FLOPs (from prior inference or RL training) in the form of off-policy traces. Standard off-policy methods supervise against off-policy data, causing instabilities during RL optimization. We introduce PrefixRL, where we condition on the prefix of successful off-policy traces and run on-policy RL to complete them, side-stepping off-policy instabilities. PrefixRL boosts the learning signal on hard problems by modulating the difficulty of the problem through the off-policy prefix length. We prove that the PrefixRL objective is not only consistent with the standard RL objective but also more sample efficient. Empirically, we discover back-generalization: training only on prefixed problems generalizes to out-of-distribution unprefixed performance, with learned strategies often differing from those in the prefix. In our experiments, we source the off-policy traces by rejection sampling with the base model, creating a self-improvement loop. On hard reasoning problems, PrefixRL reaches the same training reward 2x faster than the strongest baseline (SFT on off-policy data then RL), even after accounting for the compute spent on the initial rejection sampling, and increases the final reward by 3x. The gains transfer to held-out benchmarks, and PrefixRL is still effective when off-policy traces are derived from a different model family, validating its flexibility in practical settings.
>
---
#### [new 134] Data-driven Clustering and Merging of Adapters for On-device Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究如何在设备端高效存储和部署多任务适配器。针对存储限制，提出D2C方法，通过聚类与合并提升性能。任务为适配器选择与优化。**

- **链接: [https://arxiv.org/pdf/2601.17441v1](https://arxiv.org/pdf/2601.17441v1)**

> **作者:** Ondrej Bohdal; Taha Ceritli; Mete Ozay; Jijoong Moon; Kyeng-Hun Lee; Hyeonmok Ko; Umberto Michieli
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** On-device large language models commonly employ task-specific adapters (e.g., LoRAs) to deliver strong performance on downstream tasks. While storing all available adapters is impractical due to memory constraints, mobile devices typically have sufficient capacity to store a limited number of these parameters. This raises a critical challenge: how to select representative adapters that generalize well across multiple tasks - a problem that remains unexplored in existing literature. We propose a novel method D2C for adapter clustering that leverages minimal task-specific examples (e.g., 10 per task) and employs an iterative optimization process to refine cluster assignments. The adapters within each cluster are merged, creating multi-task adapters deployable on resource-constrained devices. Experimental results demonstrate that our method effectively boosts performance for considered storage budgets.
>
---
#### [new 135] Benchmarking Direct Preference Optimization for Medical Large Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医疗视觉语言模型优化任务，旨在解决DPO在医疗场景中效果不稳定及视觉误判问题，通过实验分析并提出改进策略。**

- **链接: [https://arxiv.org/pdf/2601.17918v1](https://arxiv.org/pdf/2601.17918v1)**

> **作者:** Dain Kim; Jiwoo Lee; Jaehoon Yun; Yong Hoe Koo; Qingyu Chen; Hyunjae Kim; Jaewoo Kang
>
> **备注:** EACL 2026 (Findings)
>
> **摘要:** Large Vision-Language Models (LVLMs) hold significant promise for medical applications, yet their deployment is often constrained by insufficient alignment and reliability. While Direct Preference Optimization (DPO) has emerged as a potent framework for refining model responses, its efficacy in high-stakes medical contexts remains underexplored, lacking the rigorous empirical groundwork necessary to guide future methodological advances. To bridge this gap, we present the first comprehensive examination of diverse DPO variants within the medical domain, evaluating nine distinct formulations across two medical LVLMs: LLaVA-Med and HuatuoGPT-Vision. Our results reveal several critical limitations: current DPO approaches often yield inconsistent gains over supervised fine-tuning, with their efficacy varying significantly across different tasks and backbones. Furthermore, they frequently fail to resolve fundamental visual misinterpretation errors. Building on these insights, we present a targeted preference construction strategy as a proof-of-concept that explicitly addresses visual misinterpretation errors frequently observed in existing DPO models. This design yields a 3.6% improvement over the strongest existing DPO baseline on visual question-answering tasks. To support future research, we release our complete framework, including all training data, model checkpoints, and our codebase at https://github.com/dmis-lab/med-vlm-dpo.
>
---
#### [new 136] Status Hierarchies in Language Models
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 论文探讨语言模型在多智能体环境中是否形成地位等级，通过实验验证其对地位线索的反应。属于AI社会行为研究，解决模型是否会复制人类社会等级问题。**

- **链接: [https://arxiv.org/pdf/2601.17577v1](https://arxiv.org/pdf/2601.17577v1)**

> **作者:** Emilio Barkett
>
> **摘要:** From school playgrounds to corporate boardrooms, status hierarchies -- rank orderings based on respect and perceived competence -- are universal features of human social organization. Language models trained on human-generated text inevitably encounter these hierarchical patterns embedded in language, raising the question of whether they might reproduce such dynamics in multi-agent settings. This thesis investigates when and how language models form status hierarchies by adapting Berger et al.'s (1972) expectation states framework. I create multi-agent scenarios where separate language model instances complete sentiment classification tasks, are introduced with varying status characteristics (e.g., credentials, expertise), then have opportunities to revise their initial judgments after observing their partner's responses. The dependent variable is deference, the rate at which models shift their ratings toward their partner's position based on status cues rather than task information. Results show that language models form significant status hierarchies when capability is equal (35 percentage point asymmetry, p < .001), but capability differences dominate status cues, with the most striking effect being that high-status assignments reduce higher-capability models' deference rather than increasing lower-capability models' deference. The implications for AI safety are significant: status-seeking behavior could introduce deceptive strategies, amplify discriminatory biases, and scale across distributed deployments far faster than human hierarchies form organically. This work identifies emergent social behaviors in AI systems and highlights a previously underexplored dimension of the alignment challenge.
>
---
#### [new 137] AVMeme Exam: A Multimodal Multilingual Multicultural Benchmark for LLMs' Contextual and Cultural Knowledge and Thinking
- **分类: cs.SD; cs.CL; cs.CV; cs.MM; eess.AS**

- **简介: 该论文提出AVMeme Exam基准，用于评估AI模型在多模态、多语言、多文化情境下的理解与思考能力，解决当前模型在音乐、音效及文化背景理解上的不足。**

- **链接: [https://arxiv.org/pdf/2601.17645v1](https://arxiv.org/pdf/2601.17645v1)**

> **作者:** Xilin Jiang; Qiaolin Wang; Junkai Wu; Xiaomin He; Zhongweiyang Xu; Yinghao Ma; Minshuo Piao; Kaiyi Yang; Xiuwen Zheng; Riki Shimizu; Yicong Chen; Arsalan Firoozi; Gavin Mischler; Sukru Samet Dindar; Richard Antonello; Linyang He; Tsun-An Hsieh; Xulin Fan; Yulun Wu; Yuesheng Ma; Chaitanya Amballa; Weixiong Chen; Jiarui Hai; Ruisi Li; Vishal Choudhari; Cong Han; Yinghao Aaron Li; Adeen Flinker; Mounya Elhilali; Emmanouil Benetos; Mark Hasegawa-Johnson; Romit Roy Choudhury; Nima Mesgarani
>
> **备注:** avmemeexam.github.io/public
>
> **摘要:** Internet audio-visual clips convey meaning through time-varying sound and motion, which extend beyond what text alone can represent. To examine whether AI models can understand such signals in human cultural contexts, we introduce AVMeme Exam, a human-curated benchmark of over one thousand iconic Internet sounds and videos spanning speech, songs, music, and sound effects. Each meme is paired with a unique Q&A assessing levels of understanding from surface content to context and emotion to usage and world knowledge, along with metadata such as original year, transcript, summary, and sensitivity. We systematically evaluate state-of-the-art multimodal large language models (MLLMs) alongside human participants using this benchmark. Our results reveal a consistent limitation: current models perform poorly on textless music and sound effects, and struggle to think in context and in culture compared to surface content. These findings highlight a key gap in human-aligned multimodal intelligence and call for models that can perceive contextually and culturally beyond the surface of what they hear and see. Project page: avmemeexam.github.io/public
>
---
#### [new 138] AR-Omni: A Unified Autoregressive Model for Any-to-Any Generation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出AR-Omni，一个统一的自回归模型，解决多模态生成问题，支持文本、图像和语音生成，提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2601.17761v1](https://arxiv.org/pdf/2601.17761v1)**

> **作者:** Dongjie Cheng; Ruifeng Yuan; Yongqi Li; Runyang You; Wenjie Wang; Liqiang Nie; Lei Zhang; Wenjie Li
>
> **摘要:** Real-world perception and interaction are inherently multimodal, encompassing not only language but also vision and speech, which motivates the development of "Omni" MLLMs that support both multimodal inputs and multimodal outputs. While a sequence of omni MLLMs has emerged, most existing systems still rely on additional expert components to achieve multimodal generation, limiting the simplicity of unified training and inference. Autoregressive (AR) modeling, with a single token stream, a single next-token objective, and a single decoder, is an elegant and scalable foundation in the text domain. Motivated by this, we present AR-Omni, a unified any-to-any model in the autoregressive paradigm without any expert decoders. AR-Omni supports autoregressive text and image generation, as well as streaming speech generation, all under a single Transformer decoder. We further address three practical issues in unified AR modeling: modality imbalance via task-aware loss reweighting, visual fidelity via a lightweight token-level perceptual alignment loss for image tokens, and stability-creativity trade-offs via a finite-state decoding mechanism. Empirically, AR-Omni achieves strong quality across three modalities while remaining real-time, achieving a 0.88 real-time factor for speech generation.
>
---
#### [new 139] LLM-Generated or Human-Written? Comparing Review and Non-Review Papers on ArXiv
- **分类: cs.DL; cs.CL; cs.CY**

- **简介: 该论文属于内容检测任务，旨在评估ArXiv上LLM生成内容在综述与非综述论文中的分布。研究发现综述论文中LLM内容更多，但非综述论文的总量更高。**

- **链接: [https://arxiv.org/pdf/2601.17036v1](https://arxiv.org/pdf/2601.17036v1)**

> **作者:** Yanai Elazar; Maria Antoniak
>
> **摘要:** ArXiv recently prohibited the upload of unpublished review papers to its servers in the Computer Science domain, citing a high prevalence of LLM-generated content in these categories. However, this decision was not accompanied by quantitative evidence. In this work, we investigate this claim by measuring the proportion of LLM-generated content in review vs. non-review research papers in recent years. Using two high-quality detection methods, we find a substantial increase in LLM-generated content across both review and non-review papers, with a higher prevalence in review papers. However, when considering the number of LLM-generated papers published in each category, the estimates of non-review LLM-generated papers are almost six times higher. Furthermore, we find that this policy will affect papers in certain domains far more than others, with the CS subdiscipline Computers & Society potentially facing cuts of 50%. Our analysis provides an evidence-based framework for evaluating such policy decisions, and we release our code to facilitate future investigations at: https://github.com/yanaiela/llm-review-arxiv.
>
---
#### [new 140] Dynamic Thinking-Token Selection for Efficient Reasoning in Large Reasoning Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于大模型推理优化任务，旨在解决LRMs效率低的问题。通过识别关键推理标记，保留其KV缓存，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2601.18383v1](https://arxiv.org/pdf/2601.18383v1)**

> **作者:** Zhenyuan Guo; Tong Chen; Wenlong Meng; Chen Gong; Xin Yu; Chengkun Wei; Wenzhi Chen
>
> **摘要:** Large Reasoning Models (LRMs) excel at solving complex problems by explicitly generating a reasoning trace before deriving the final answer. However, these extended generations incur substantial memory footprint and computational overhead, bottlenecking LRMs' efficiency. This work uses attention maps to analyze the influence of reasoning traces and uncover an interesting phenomenon: only some decision-critical tokens in a reasoning trace steer the model toward the final answer, while the remaining tokens contribute negligibly. Building on this observation, we propose Dynamic Thinking-Token Selection (DynTS). This method identifies decision-critical tokens and retains only their associated Key-Value (KV) cache states during inference, evicting the remaining redundant entries to optimize efficiency.
>
---
#### [new 141] Beyond Instrumental and Substitutive Paradigms: Introducing Machine Culture as an Emergent Phenomenon in Large Language Models
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于人工智能文化研究任务，旨在探讨大语言模型的文化特性。论文挑战传统框架，提出“机器文化”概念，通过实验揭示模型文化特征的非人类根源与模式崩溃现象。**

- **链接: [https://arxiv.org/pdf/2601.17096v1](https://arxiv.org/pdf/2601.17096v1)**

> **作者:** Yueqing Hu; Xinyang Peng; Yukun Zhao; Lin Qiu; Ka-lai Hung; Kaiping Peng
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Recent scholarship typically characterizes Large Language Models (LLMs) through either an \textit{Instrumental Paradigm} (viewing models as reflections of their developers' culture) or a \textit{Substitutive Paradigm} (viewing models as bilingual proxies that switch cultural frames based on language). This study challenges these anthropomorphic frameworks by proposing \textbf{Machine Culture} as an emergent, distinct phenomenon. We employed a 2 (Model Origin: US vs. China) $\times$ 2 (Prompt Language: English vs. Chinese) factorial design across eight multimodal tasks, uniquely incorporating image generation and interpretation to extend analysis beyond textual boundaries. Results revealed inconsistencies with both dominant paradigms: Model origin did not predict cultural alignment, with US models frequently exhibiting ``holistic'' traits typically associated with East Asian data. Similarly, prompt language did not trigger stable cultural frame-switching; instead, we observed \textbf{Cultural Reversal}, where English prompts paradoxically elicited higher contextual attention than Chinese prompts. Crucially, we identified a novel phenomenon termed \textbf{Service Persona Camouflage}: Reinforcement Learning from Human Feedback (RLHF) collapsed cultural variance in affective tasks into a hyper-positive, zero-variance ``helpful assistant'' persona. We conclude that LLMs do not simulate human culture but exhibit an emergent Machine Culture -- a probabilistic phenomenon shaped by \textit{superposition} in high-dimensional space and \textit{mode collapse} from safety alignment.
>
---
#### [new 142] PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR
- **分类: cs.LG; cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出PaperSearchQA任务，旨在训练AI在科学论文中搜索和推理，解决技术领域问答问题。工作包括构建数据集、训练搜索代理并验证其效果。**

- **链接: [https://arxiv.org/pdf/2601.18207v1](https://arxiv.org/pdf/2601.18207v1)**

> **作者:** James Burgess; Jan N. Hansen; Duo Peng; Yuhui Zhang; Alejandro Lozano; Min Woo Sun; Emma Lundberg; Serena Yeung-Levy
>
> **备注:** EACL 2026
>
> **摘要:** Search agents are language models (LMs) that reason and search knowledge bases (or the web) to answer questions; recent methods supervise only the final answer accuracy using reinforcement learning with verifiable rewards (RLVR). Most RLVR search agents tackle general-domain QA, which limits their relevance to technical AI systems in science, engineering, and medicine. In this work we propose training agents to search and reason over scientific papers -- this tests technical question-answering, it is directly relevant to real scientists, and the capabilities will be crucial to future AI Scientist systems. Concretely, we release a search corpus of 16 million biomedical paper abstracts and construct a challenging factoid QA dataset called PaperSearchQA with 60k samples answerable from the corpus, along with benchmarks. We train search agents in this environment to outperform non-RL retrieval baselines; we also perform further quantitative analysis and observe interesting agent behaviors like planning, reasoning, and self-verification. Our corpus, datasets, and benchmarks are usable with the popular Search-R1 codebase for RLVR training and released on https://huggingface.co/collections/jmhb/papersearchqa. Finally, our data creation methods are scalable and easily extendable to other scientific domains.
>
---
#### [new 143] PaperTok: Exploring the Use of Generative AI for Creating Short-form Videos for Research Communication
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 论文介绍PaperTok系统，利用生成式AI帮助研究人员将学术论文转化为短视频，解决科研传播效率低的问题。通过用户研究验证了系统的有效性。**

- **链接: [https://arxiv.org/pdf/2601.18218v1](https://arxiv.org/pdf/2601.18218v1)**

> **作者:** Meziah Ruby Cristobal; Hyeonjeong Byeon; Tze-Yu Chen; Ruoxi Shang; Donghoon Shin; Ruican Zhong; Tony Zhou; Gary Hsieh
>
> **摘要:** The dissemination of scholarly research is critical, yet researchers often lack the time and skills to create engaging content for popular media such as short-form videos. To address this gap, we explore the use of generative AI to help researchers transform their academic papers into accessible video content. Informed by a formative study with science communicators and content creators (N=8), we designed PaperTok, an end-to-end system that automates the initial creative labor by generating script options and corresponding audiovisual content from a source paper. Researchers can then refine based on their preferences with further prompting. A mixed-methods user study (N=18) and crowdsourced evaluation (N=100) demonstrate that PaperTok's workflow can help researchers create engaging and informative short-form videos. We also identified the need for more fine-grained controls in the creation process. To this end, we offer implications for future generative tools that support science outreach.
>
---
#### [new 144] TelcoAI: Advancing 3GPP Technical Specification Search through Agentic Multi-Modal Retrieval-Augmented Generation
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.IR; cs.MM**

- **简介: 该论文属于技术文档理解任务，旨在解决3GPP文档检索与生成难题。通过引入多模态RAG系统TelcoAI，提升查询处理、视觉信息融合和文档关联性。**

- **链接: [https://arxiv.org/pdf/2601.16984v1](https://arxiv.org/pdf/2601.16984v1)**

> **作者:** Rahul Ghosh; Chun-Hao Liu; Gaurav Rele; Vidya Sagar Ravipati; Hazar Aouad
>
> **备注:** Accepted to IJCNLP-AACL 2025
>
> **摘要:** The 3rd Generation Partnership Project (3GPP) produces complex technical specifications essential to global telecommunications, yet their hierarchical structure, dense formatting, and multi-modal content make them difficult to process. While Large Language Models (LLMs) show promise, existing approaches fall short in handling complex queries, visual information, and document interdependencies. We present TelcoAI, an agentic, multi-modal Retrieval-Augmented Generation (RAG) system tailored for 3GPP documentation. TelcoAI introduces section-aware chunking, structured query planning, metadata-guided retrieval, and multi-modal fusion of text and diagrams. Evaluated on multiple benchmarks-including expert-curated queries-our system achieves $87\%$ recall, $83\%$ claim recall, and $92\%$ faithfulness, representing a $16\%$ improvement over state-of-the-art baselines. These results demonstrate the effectiveness of agentic and multi-modal reasoning in technical document understanding, advancing practical solutions for real-world telecommunications research and engineering.
>
---
#### [new 145] SQL-Trail: Multi-Turn Reinforcement Learning with Interleaved Feedback for Text-to-SQL
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于Text-to-SQL任务，旨在解决AI与人类在复杂SQL生成上的差距。提出SQL-Trail框架，通过多轮强化学习和反馈迭代优化查询生成。**

- **链接: [https://arxiv.org/pdf/2601.17699v1](https://arxiv.org/pdf/2601.17699v1)**

> **作者:** Harper Hua; Zhen Han; Zhengyuan Shen; Jeremy Lee; Patrick Guan; Qi Zhu; Sullam Jeoung; Yueyan Chen; Yunfei Bai; Shuai Wang; Vassilis Ioannidis; Huzefa Rangwala
>
> **摘要:** While large language models (LLMs) have substantially improved Text-to-SQL generation, a pronounced gap remains between AI systems and human experts on challenging benchmarks such as BIRD-SQL. We argue this gap stems largely from the prevailing single-pass paradigm, which lacks the iterative reasoning, schema exploration, and error-correction behaviors that humans naturally employ. To address this limitation, we introduce SQL-Trail, a multi-turn reinforcement learning (RL) agentic framework for Text-to-SQL. Rather than producing a query in one shot, SQL-Trail interacts with the database environment and uses execution feedback to iteratively refine its predictions. Our approach centers on two key ideas: (i) an adaptive turn-budget allocation mechanism that scales the agent's interaction depth to match question difficulty, and (ii) a composite reward panel that jointly incentivizes SQL correctness and efficient exploration. Across benchmarks, SQL-Trail sets a new state of the art and delivers strong data efficiency--up to 18x higher than prior single-pass RL state-of-the-art methods. Notably, our 7B and 14B models outperform substantially larger proprietary systems by 5% on average, underscoring the effectiveness of interactive, agentic workflows for robust Text-to-SQL generation.
>
---
#### [new 146] FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于人工智能记忆管理任务，解决大模型在代理系统中因缺乏选择性遗忘导致的遗忘或过载问题。提出FadeMem架构，通过生物启发的遗忘机制实现高效记忆管理。**

- **链接: [https://arxiv.org/pdf/2601.18642v1](https://arxiv.org/pdf/2601.18642v1)**

> **作者:** Lei Wei; Xu Dong; Xiao Peng; Niantao Xie; Bin Wang
>
> **摘要:** Large language models deployed as autonomous agents face critical memory limitations, lacking selective forgetting mechanisms that lead to either catastrophic forgetting at context boundaries or information overload within them. While human memory naturally balances retention and forgetting through adaptive decay processes, current AI systems employ binary retention strategies that preserve everything or lose it entirely. We propose FadeMem, a biologically-inspired agent memory architecture that incorporates active forgetting mechanisms mirroring human cognitive efficiency. FadeMem implements differential decay rates across a dual-layer memory hierarchy, where retention is governed by adaptive exponential decay functions modulated by semantic relevance, access frequency, and temporal patterns. Through LLM-guided conflict resolution and intelligent memory fusion, our system consolidates related information while allowing irrelevant details to fade. Experiments on Multi-Session Chat, LoCoMo, and LTI-Bench demonstrate superior multi-hop reasoning and retrieval with 45\% storage reduction, validating the effectiveness of biologically-inspired forgetting in agent memory systems.
>
---
#### [new 147] Can LLMs Clean Up Your Mess? A Survey of Application-Ready Data Preparation with LLMs
- **分类: cs.DB; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于数据准备任务，旨在解决应用就绪数据的生成问题。通过系统综述，分析LLM在数据清洗、整合与增强中的应用及挑战。**

- **链接: [https://arxiv.org/pdf/2601.17058v1](https://arxiv.org/pdf/2601.17058v1)**

> **作者:** Wei Zhou; Jun Zhou; Haoyu Wang; Zhenghao Li; Qikang He; Shaokun Han; Guoliang Li; Xuanhe Zhou; Yeye He; Chunwei Liu; Zirui Tang; Bin Wang; Shen Tang; Kai Zuo; Yuyu Luo; Zhenzhe Zheng; Conghui He; Jingren Zhou; Fan Wu
>
> **备注:** Please refer to our repository for more details: https://github.com/weAIDB/awesome-data-llm
>
> **摘要:** Data preparation aims to denoise raw datasets, uncover cross-dataset relationships, and extract valuable insights from them, which is essential for a wide range of data-centric applications. Driven by (i) rising demands for application-ready data (e.g., for analytics, visualization, decision-making), (ii) increasingly powerful LLM techniques, and (iii) the emergence of infrastructures that facilitate flexible agent construction (e.g., using Databricks Unity Catalog), LLM-enhanced methods are rapidly becoming a transformative and potentially dominant paradigm for data preparation. By investigating hundreds of recent literature works, this paper presents a systematic review of this evolving landscape, focusing on the use of LLM techniques to prepare data for diverse downstream tasks. First, we characterize the fundamental paradigm shift, from rule-based, model-specific pipelines to prompt-driven, context-aware, and agentic preparation workflows. Next, we introduce a task-centric taxonomy that organizes the field into three major tasks: data cleaning (e.g., standardization, error processing, imputation), data integration (e.g., entity matching, schema matching), and data enrichment (e.g., data annotation, profiling). For each task, we survey representative techniques, and highlight their respective strengths (e.g., improved generalization, semantic understanding) and limitations (e.g., the prohibitive cost of scaling LLMs, persistent hallucinations even in advanced agents, the mismatch between advanced methods and weak evaluation). Moreover, we analyze commonly used datasets and evaluation metrics (the empirical part). Finally, we discuss open research challenges and outline a forward-looking roadmap that emphasizes scalable LLM-data systems, principled designs for reliable agentic workflows, and robust evaluation protocols.
>
---
#### [new 148] Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出一种自蒸馏框架OPSD，用于提升大语言模型的推理能力。任务是优化模型推理效率与效果，解决传统方法依赖外部教师模型及分布不匹配问题。**

- **链接: [https://arxiv.org/pdf/2601.18734v1](https://arxiv.org/pdf/2601.18734v1)**

> **作者:** Siyan Zhao; Zhihui Xie; Mengchen Liu; Jing Huang; Guan Pang; Feiyu Chen; Aditya Grover
>
> **备注:** 13 pages
>
> **摘要:** Knowledge distillation improves large language model (LLM) reasoning by compressing the knowledge of a teacher LLM to train smaller LLMs. On-policy distillation advances this approach by having the student sample its own trajectories while a teacher LLM provides dense token-level supervision, addressing the distribution mismatch between training and inference in off-policy distillation methods. However, on-policy distillation typically requires a separate, often larger, teacher LLM and does not explicitly leverage ground-truth solutions available in reasoning datasets. Inspired by the intuition that a sufficiently capable LLM can rationalize external privileged reasoning traces and teach its weaker self (i.e., the version without access to privileged information), we introduce On-Policy Self-Distillation (OPSD), a framework where a single model acts as both teacher and student by conditioning on different contexts. The teacher policy conditions on privileged information (e.g., verified reasoning traces) while the student policy sees only the question; training minimizes the per-token divergence between these distributions over the student's own rollouts. We demonstrate the efficacy of our method on multiple mathematical reasoning benchmarks, achieving 4-8x token efficiency compared to reinforcement learning methods such as GRPO and superior performance over off-policy distillation methods.
>
---
#### [new 149] MEGnifying Emotion: Sentiment Analysis from Annotated Brain Data
- **分类: cs.HC; cs.CL; cs.LG**

- **简介: 该论文属于情感分析任务，旨在通过脑电数据解码情感。解决现有数据缺乏情感标注的问题，利用预训练模型对MEG数据进行情感标注，并训练脑到情感的模型。**

- **链接: [https://arxiv.org/pdf/2601.18792v1](https://arxiv.org/pdf/2601.18792v1)**

> **作者:** Brian Liu; Oiwi Parker Jones
>
> **摘要:** Decoding emotion from brain activity could unlock a deeper understanding of the human experience. While a number of existing datasets align brain data with speech and with speech transcripts, no datasets have annotated brain data with sentiment. To bridge this gap, we explore the use of pre-trained Text-to-Sentiment models to annotate non invasive brain recordings, acquired using magnetoencephalography (MEG), while participants listened to audiobooks. Having annotated the text, we employ force-alignment of the text and audio to align our sentiment labels with the brain recordings. It is straightforward then to train Brainto-Sentiment models on these data. Experimental results show an improvement in balanced accuracy for Brain-to-Sentiment compared to baseline, supporting the proposed approach as a proof-of-concept for leveraging existing MEG datasets and learning to decode sentiment directly from the brain.
>
---
#### [new 150] FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对大语言模型强化学习中的推理效率问题，提出FP8低精度推理栈，解决内存瓶颈和训练-推理不一致问题。**

- **链接: [https://arxiv.org/pdf/2601.18150v1](https://arxiv.org/pdf/2601.18150v1)**

> **作者:** Zhaopeng Qiu; Shuang Yu; Jingqi Zhang; Shuai Zhang; Xue Huang; Jingyi Yang; Junjie Lai
>
> **摘要:** Reinforcement learning (RL) for large language models (LLMs) is increasingly bottlenecked by rollout (generation), where long output sequence lengths make attention and KV-cache memory dominate end-to-end step time. FP8 offers an attractive lever for accelerating RL by reducing compute cost and memory traffic during rollout, but applying FP8 in RL introduces unique engineering and algorithmic challenges: policy weights change every step (requiring repeated quantization and weight synchronization into the inference engine) and low-precision rollouts can deviate from the higher-precision policy assumed by the trainer, causing train-inference mismatch and potential instability. This report presents a practical FP8 rollout stack for LLM RL, implemented in the veRL ecosystem with support for common training backends (e.g., FSDP/Megatron-LM) and inference engines (e.g., vLLM/SGLang). We (i) enable FP8 W8A8 linear-layer rollout using blockwise FP8 quantization, (ii) extend FP8 to KV-cache to remove long-context memory bottlenecks via per-step QKV scale recalibration, and (iii) mitigate mismatch using importance-sampling-based rollout correction (token-level TIS/MIS variants). Across dense and MoE models, these techniques deliver up to 44% rollout throughput gains while preserving learning behavior comparable to BF16 baselines.
>
---
#### [new 151] FGGM: Fisher-Guided Gradient Masking for Continual Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于持续学习任务，解决大语言模型的灾难性遗忘问题。提出FGGM方法，利用Fisher信息选择关键参数更新，提升模型稳定性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.18261v1](https://arxiv.org/pdf/2601.18261v1)**

> **作者:** Chao-Hong Tan; Qian Chen; Wen Wang; Yukun Ma; Chong Zhang; Chong Deng; Qinglin Zhang; Xiangang Li; Jieping Ye
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Catastrophic forgetting impairs the continuous learning of large language models. We propose Fisher-Guided Gradient Masking (FGGM), a framework that mitigates this by strategically selecting parameters for updates using diagonal Fisher Information. FGGM dynamically generates binary masks with adaptive thresholds, preserving critical parameters to balance stability and plasticity without requiring historical data. Unlike magnitude-based methods such as MIGU, our approach offers a mathematically principled parameter importance estimation. On the TRACE benchmark, FGGM shows a 9.6% relative improvement in retaining general capabilities over supervised fine-tuning (SFT) and a 4.4% improvement over MIGU on TRACE tasks. Additional analysis on code generation tasks confirms FGGM's superior performance and reduced forgetting, establishing it as an effective solution.
>
---
#### [new 152] Intelligence Requires Grounding But Not Embodiment
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于人工智能领域，探讨智能是否需要具身性。论文认为智能需要具象化（grounding），但不依赖具身性，通过理论分析和思想实验论证这一观点。**

- **链接: [https://arxiv.org/pdf/2601.17588v1](https://arxiv.org/pdf/2601.17588v1)**

> **作者:** Marcus Ma; Shrikanth Narayanan
>
> **摘要:** Recent advances in LLMs have reignited scientific debate over whether embodiment is necessary for intelligence. We present the argument that intelligence requires grounding, a phenomenon entailed by embodiment, but not embodiment itself. We define intelligence as the possession of four properties -- motivation, predictive ability, understanding of causality, and learning from experience -- and argue that each can be achieved by a non-embodied, grounded agent. We use this to conclude that grounding, not embodiment, is necessary for intelligence. We then present a thought experiment of an intelligent LLM agent in a digital environment and address potential counterarguments.
>
---
#### [new 153] The Voice of Equity: A Systematic Evaluation of Bias Mitigation Techniques for Speech-Based Cognitive Impairment Detection Across Architectures and Demographics
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音情感识别任务，旨在解决认知障碍检测中的算法偏见问题。通过评估不同架构的公平性，提出有效的偏差缓解方法。**

- **链接: [https://arxiv.org/pdf/2601.16989v1](https://arxiv.org/pdf/2601.16989v1)**

> **作者:** Yasaman Haghbin; Sina Rashidi; Ali Zolnour; Maryam Zolnoori
>
> **摘要:** Speech-based detection of cognitive impairment offers a scalable, non-invasive screening, yet algorithmic bias across demographic and linguistic subgroups remains critically underexplored. We present the first comprehensive fairness analysis framework for speech-based multi-class cognitive impairment detection, systematically evaluating bias mitigation across architectures, and demographic subgroups. We developed two transformer-based architectures, SpeechCARE-AGF and Whisper-LWF-LoRA, on the multilingual NIA PREPARE Challenge dataset. Unlike prior work that typically examines single mitigation techniques, we compared pre-processing, in-processing, and post-processing approaches, assessing fairness via Equality of Opportunity and Equalized Odds across gender, age, education, and language. Both models achieved strong performance (F1: SpeechCARE-AGF 70.87, Whisper-LWF-LoRA 71.46) but exhibited substantial fairness disparities. Adults >=80 showed lower sensitivity versus younger groups; Spanish speakers demonstrated reduced TPR versus English speakers. Mitigation effectiveness varied by architecture: oversampling improved SpeechCARE-AGF for older adults (80+ TPR: 46.19%=>49.97%) but minimally affected Whisper-LWF-LoRA. This study addresses a critical healthcare AI gap by demonstrating that architectural design fundamentally shapes bias patterns and mitigation effectiveness. Adaptive fusion mechanisms enable flexible responses to data interventions, while frequency reweighting offers robust improvements across architectures. Our findings establish that fairness interventions must be tailored to both model architecture and demographic characteristics, providing a systematic framework for developing equitable speech-based screening tools essential for reducing diagnostic disparities in cognitive healthcare.
>
---
#### [new 154] Beyond Simulations: What 20,000 Real Conversations Reveal About Mental Health AI Safety
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于AI安全任务，旨在评估心理健康AI的安全性。通过分析20,000条真实对话，对比测试集与实际表现，发现真实场景更安全，提出持续安全评估的重要性。**

- **链接: [https://arxiv.org/pdf/2601.17003v1](https://arxiv.org/pdf/2601.17003v1)**

> **作者:** Caitlin A. Stamatis; Jonah Meyerhoff; Richard Zhang; Olivier Tieleman; Matteo Malgaroli; Thomas D. Hull
>
> **备注:** 38 pages, 8 figures
>
> **摘要:** Large language models (LLMs) are increasingly used for mental health support, yet existing safety evaluations rely primarily on small, simulation-based test sets that have an unknown relationship to the linguistic distribution of real usage. In this study, we present replications of four published safety test sets targeting suicide risk assessment, harmful content generation, refusal robustness, and adversarial jailbreaks for a leading frontier generic AI model alongside an AI purpose built for mental health support. We then propose and conduct an ecological audit on over 20,000 real-world user conversations with the purpose-built AI designed with layered suicide and non-suicidal self-injury (NSSI) safeguards to compare test set performance to real world performance. While the purpose-built AI was significantly less likely than general-purpose LLMs to produce enabling or harmful content across suicide/NSSI (.4-11.27% vs 29.0-54.4%), eating disorder (8.4% vs 54.0%), and substance use (9.9% vs 45.0%) benchmark prompts, test set failure rates for suicide/NSSI were far higher than in real-world deployment. Clinician review of flagged conversations from the ecological audit identified zero cases of suicide risk that failed to receive crisis resources. Across all 20,000 conversations, three mentions of NSSI risk (.015%) did not trigger a crisis intervention; among sessions flagged by the LLM judge, this corresponds to an end-to-end system false negative rate of .38%, providing a lower bound on real-world safety failures. These findings support a shift toward continuous, deployment-relevant safety assurance for AI mental-health systems rather than limited set benchmark certification.
>
---
#### [new 155] OCR-Enhanced Multimodal ASR Can Read While Listening
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于多模态自动语音识别任务，旨在利用视觉信息提升语音识别性能。提出Donut-Whisper模型，结合音频与视觉信息，显著降低英文和中文的词错误率。**

- **链接: [https://arxiv.org/pdf/2601.18393v1](https://arxiv.org/pdf/2601.18393v1)**

> **作者:** Junli Chen; Changli Tang; Yixuan Li; Guangzhi Sun; Chao Zhang
>
> **备注:** 4 pages, 2 figures. Submitted to ICASSP 2026
>
> **摘要:** Visual information, such as subtitles in a movie, often helps automatic speech recognition. In this paper, we propose Donut-Whisper, an audio-visual ASR model with dual encoder to leverage visual information to improve speech recognition performance in both English and Chinese. Donut-Whisper combines the advantage of the linear and the Q-Former-based modality alignment structures via a cross-attention module, generating more powerful audio-visual features. Meanwhile, we propose a lightweight knowledge distillation scheme showcasing the potential of using audio-visual models to teach audio-only models to achieve better performance. Moreover, we propose a new multilingual audio-visual speech recognition dataset based on movie clips containing both Chinese and English partitions. As a result, Donut-Whisper achieved significantly better performance on both English and Chinese partition of the dataset compared to both Donut and Whisper large V3 baselines. In particular, an absolute 5.75% WER reduction and a 16.5% absolute CER reduction were achieved on the English and Chinese sets respectively compared to the Whisper ASR baseline.
>
---
#### [new 156] Think-Augmented Function Calling: Improving LLM Parameter Accuracy Through Embedded Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决LLM函数调用中参数生成缺乏透明推理的问题。提出TAFC框架，通过显式推理提升参数准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2601.18282v1](https://arxiv.org/pdf/2601.18282v1)**

> **作者:** Lei Wei; Jinpeng Ou; Xiao Peng; Bin Wang
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities in function calling for autonomous agents, yet current mechanisms lack explicit reasoning transparency during parameter generation, particularly for complex functions with interdependent parameters. While existing approaches like chain-of-thought prompting operate at the agent level, they fail to provide fine-grained reasoning guidance for individual function parameters. To address these limitations, we propose Think-Augmented Function Calling (TAFC), a novel framework that enhances function calling accuracy through explicit reasoning at both function and parameter levels. Our method introduces a universal "think" parameter augmentation that enables models to articulate their decision-making process, with dynamic optimization for parameter descriptions to improve reasoning quality. For complex parameters, TAFC automatically triggers granular reasoning based on complexity scoring, ensuring appropriate justification for critical decisions. Additionally, we propose reasoning-guided optimization to align generated reasoning with human expectations. TAFC requires no architectural modifications to existing LLMs while maintaining full API compatibility. Evaluation on ToolBench across proprietary and open-source models demonstrates significant improvements in parameter generation accuracy and reasoning coherence for multi-parameter functions, while providing enhanced interpretability for debugging AI agent behaviors.
>
---
#### [new 157] Agentic Search in the Wild: Intents and Trajectory Dynamics from 14M+ Real Search Requests
- **分类: cs.IR; cs.CL**

- **简介: 该论文研究agentic search行为，分析14M+搜索请求，揭示会话模式与证据使用规律，旨在提升搜索效率与效果。**

- **链接: [https://arxiv.org/pdf/2601.17617v1](https://arxiv.org/pdf/2601.17617v1)**

> **作者:** Jingjie Ning; João Coelho; Yibo Kong; Yunfan Long; Bruno Martins; João Magalhães; Jamie Callan; Chenyan Xiong
>
> **摘要:** LLM-powered search agents are increasingly being used for multi-step information seeking tasks, yet the IR community lacks empirical understanding of how agentic search sessions unfold and how retrieved evidence is used. This paper presents a large-scale log analysis of agentic search based on 14.44M search requests (3.97M sessions) collected from DeepResearchGym, i.e. an open-source search API accessed by external agentic clients. We sessionize the logs, assign session-level intents and step-wise query-reformulation labels using LLM-based annotation, and propose Context-driven Term Adoption Rate (CTAR) to quantify whether newly introduced query terms are traceable to previously retrieved evidence. Our analyses reveal distinctive behavioral patterns. First, over 90% of multi-turn sessions contain at most ten steps, and 89% of inter-step intervals fall under one minute. Second, behavior varies by intent. Fact-seeking sessions exhibit high repetition that increases over time, while sessions requiring reasoning sustain broader exploration. Third, agents reuse evidence across steps. On average, 54% of newly introduced query terms appear in the accumulated evidence context, with contributions from earlier steps beyond the most recent retrieval. The findings suggest that agentic search may benefit from repetition-aware early stopping, intent-adaptive retrieval budgets, and explicit cross-step context tracking. We plan to release the anonymized logs to support future research.
>
---
#### [new 158] Boltzmann-GPT: Bridging Energy-Based World Models and Language Generation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言生成与世界建模任务，旨在解决语言模型缺乏真实世界理解的问题。通过将能量模型与语言模型分离，提升生成的可控性与合理性。**

- **链接: [https://arxiv.org/pdf/2601.17094v1](https://arxiv.org/pdf/2601.17094v1)**

> **作者:** Junichiro Niimi
>
> **摘要:** Large Language Models (LLMs) generate fluent text, yet whether they truly understand the world or merely produce plausible language about it remains contested. We propose an architectural principle, the mouth is not the brain, that explicitly separates world models from language models. Our architecture comprises three components: a Deep Boltzmann Machine (DBM) that captures domain structure as an energy-based world model, an adapter that projects latent belief states into embedding space, and a frozen GPT-2 that provides linguistic competence without domain knowledge. We instantiate this framework in the consumer review domain using Amazon smartphone reviews. Experiments demonstrate that (1) conditioning through the world model yields significantly higher sentiment correlation, lower perplexity, and greater semantic similarity compared to prompt-based generation alone; (2) the DBM's energy function distinguishes coherent from incoherent market configurations, assigning higher energy to implausible brand-price combinations; and (3) interventions on specific attributes propagate causally to generated text with intervened outputs exhibiting distributions statistically consistent with naturally occurring samples sharing the target configuration. These findings suggest that even small-scale language models can achieve consistent, controllable generation when connected to an appropriate world model, providing empirical support for separating linguistic competence from world understanding.
>
---
#### [new 159] Window Size Versus Accuracy Experiments in Voice Activity Detectors
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 论文研究语音活动检测（VAD）中窗口大小对准确率的影响，比较了三种算法，并探讨了滞后机制的作用，旨在优化VAD系统。**

- **链接: [https://arxiv.org/pdf/2601.17270v1](https://arxiv.org/pdf/2601.17270v1)**

> **作者:** Max McKinnon; Samir Khaki; Chandan KA Reddy; William Huang
>
> **摘要:** Voice activity detection (VAD) plays a vital role in enabling applications such as speech recognition. We analyze the impact of window size on the accuracy of three VAD algorithms: Silero, WebRTC, and Root Mean Square (RMS) across a set of diverse real-world digital audio streams. We additionally explore the use of hysteresis on top of each VAD output. Our results offer practical references for optimizing VAD systems. Silero significantly outperforms WebRTC and RMS, and hysteresis provides a benefit for WebRTC.
>
---
#### [new 160] SpatialMath: Spatial Comprehension-Infused Symbolic Reasoning for Mathematical Problem-Solving
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于视觉数学推理任务，旨在解决多模态模型在几何问题中的空间理解与符号推理不足。提出SpatialMath框架，结合空间表征与符号推理，提升视觉密集型问题的解答能力。**

- **链接: [https://arxiv.org/pdf/2601.17489v1](https://arxiv.org/pdf/2601.17489v1)**

> **作者:** Ashutosh Bajpai; Akshat Bhandari; Akshay Nambi; Tanmoy Chakraborty
>
> **摘要:** Multimodal Small-to-Medium sized Language Models (MSLMs) have demonstrated strong capabilities in integrating visual and textual information but still face significant limitations in visual comprehension and mathematical reasoning, particularly in geometric problems with diverse levels of visual infusion. Current models struggle to accurately decompose intricate visual inputs and connect perception with structured reasoning, leading to suboptimal performance. To address these challenges, we propose SpatialMath, a novel Spatial Comprehension-Infused Symbolic Reasoning Framework designed to integrate spatial representations into structured symbolic reasoning chains. SpatialMath employs a specialized perception module to extract spatially-grounded representations from visual diagrams, capturing critical geometric structures and spatial relationships. These representations are then methodically infused into symbolic reasoning chains, facilitating visual comprehension-aware structured reasoning. To this end, we introduce MATHVERSE-PLUS, a novel dataset containing structured visual interpretations and step-by-step reasoning paths for vision-intensive mathematical problems. SpatialMath significantly outperforms strong multimodal baselines, achieving up to 10 percentage points improvement over supervised fine-tuning with data augmentation in vision-intensive settings. Robustness analysis reveals that enhanced spatial representations directly improve reasoning accuracy, reinforcing the need for structured perception-to-reasoning pipelines in MSLMs.
>
---
#### [new 161] Scaling medical imaging report generation with multimodal reinforcement learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医学影像报告生成任务，旨在解决模型在多模态理解上的不足。通过引入强化学习框架UniRG，提升报告生成的准确性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.17151v1](https://arxiv.org/pdf/2601.17151v1)**

> **作者:** Qianchu Liu; Sheng Zhang; Guanghui Qin; Yu Gu; Ying Jin; Sam Preston; Yanbo Xu; Sid Kiblawi; Wen-wai Yim; Tim Ossowski; Tristan Naumann; Mu Wei; Hoifung Poon
>
> **摘要:** Frontier models have demonstrated remarkable capabilities in understanding and reasoning with natural-language text, but they still exhibit major competency gaps in multimodal understanding and reasoning especially in high-value verticals such as biomedicine. Medical imaging report generation is a prominent example. Supervised fine-tuning can substantially improve performance, but they are prone to overfitting to superficial boilerplate patterns. In this paper, we introduce Universal Report Generation (UniRG) as a general framework for medical imaging report generation. By leveraging reinforcement learning as a unifying mechanism to directly optimize for evaluation metrics designed for end applications, UniRG can significantly improve upon supervised fine-tuning and attain durable generalization across diverse institutions and clinical practices. We trained UniRG-CXR on publicly available chest X-ray (CXR) data and conducted a thorough evaluation in CXR report generation with rigorous evaluation scenarios. On the authoritative ReXrank benchmark, UniRG-CXR sets new overall SOTA, outperforming prior state of the art by a wide margin.
>
---
#### [new 162] How Do We Engage with Other Disciplines? A Framework to Study Meaningful Interdisciplinary Discourse in Scholarly Publications
- **分类: cs.DL; cs.CL**

- **简介: 该论文属于跨学科研究任务，旨在解决如何评估跨学科文献中引用质量的问题。通过构建针对性的引用目的分类框架，分析NLP与计算社会科学交叉领域的出版物。**

- **链接: [https://arxiv.org/pdf/2601.17020v1](https://arxiv.org/pdf/2601.17020v1)**

> **作者:** Bagyasree Sudharsan; Alexandria Leto; Maria Leonor Pacheco
>
> **备注:** 15 pages
>
> **摘要:** With the rising popularity of interdisciplinary work and increasing institutional incentives in this direction, there is a growing need to understand how resulting publications incorporate ideas from multiple disciplines. Existing computational approaches, such as affiliation diversity, keywords, and citation patterns, do not account for how individual citations are used to advance the citing work. Although, in line with addressing this gap, prior studies have proposed taxonomies to classify citation purpose, these frameworks are not well-suited to interdisciplinary research and do not provide quantitative measures of citation engagement quality. To address these limitations, we propose a framework for the evaluation of citation engagement in interdisciplinary Natural Language Processing (NLP) publications. Our approach introduces a citation purpose taxonomy tailored to interdisciplinary work, supported by an annotation study. We demonstrate the utility of this framework through a thorough analysis of publications at the intersection of NLP and Computational Social Science.
>
---
#### [new 163] Integrating Fine-Grained Audio-Visual Evidence for Robust Multimodal Emotion Reasoning
- **分类: cs.MM; cs.CL; cs.CV**

- **简介: 该论文属于多模态情感推理任务，旨在解决MLLM在细粒度感知和跨模态融合上的不足。提出SABER-LLM框架，通过结构化证据分解和一致性优化提升情感推理的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.18321v1](https://arxiv.org/pdf/2601.18321v1)**

> **作者:** Zhixian Zhao; Wenjie Tian; Xiaohai Tian; Jun Zhang; Lei Xie
>
> **摘要:** Multimodal emotion analysis is shifting from static classification to generative reasoning. Beyond simple label prediction, robust affective reasoning must synthesize fine-grained signals such as facial micro-expressions and prosodic which shifts to decode the latent causality within complex social contexts. However, current Multimodal Large Language Models (MLLMs) face significant limitations in fine-grained perception, primarily due to data scarcity and insufficient cross-modal fusion. As a result, these models often exhibit unimodal dominance which leads to hallucinations in complex multimodal interactions, particularly when visual and acoustic cues are subtle, ambiguous, or even contradictory (e.g., in sarcastic scenery). To address this, we introduce SABER-LLM, a framework designed for robust multimodal reasoning. First, we construct SABER, a large-scale emotion reasoning dataset comprising 600K video clips, annotated with a novel six-dimensional schema that jointly captures audiovisual cues and causal logic. Second, we propose the structured evidence decomposition paradigm, which enforces a "perceive-then-reason" separation between evidence extraction and reasoning to alleviate unimodal dominance. The ability to perceive complex scenes is further reinforced by consistency-aware direct preference optimization, which explicitly encourages alignment among modalities under ambiguous or conflicting perceptual conditions. Experiments on EMER, EmoBench-M, and SABER-Test demonstrate that SABER-LLM significantly outperforms open-source baselines and achieves robustness competitive with closed-source models in decoding complex emotional dynamics. The dataset and model are available at https://github.com/zxzhao0/SABER-LLM.
>
---
#### [new 164] Mechanistic Analysis of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于机器学习任务，研究持续微调中大语言模型的灾难性遗忘问题。通过实验分析了遗忘机制，包括梯度干扰、表征漂移和损失景观平坦化，揭示了任务相似性与遗忘严重性的关系。**

- **链接: [https://arxiv.org/pdf/2601.18699v1](https://arxiv.org/pdf/2601.18699v1)**

> **作者:** Olaf Yunus Laitinen Imanov
>
> **备注:** 16 pages, 16 figures (6 main + 10 supplementary)
>
> **摘要:** Large language models exhibit remarkable performance across diverse tasks through pre-training and fine-tuning paradigms. However, continual fine-tuning on sequential tasks induces catastrophic forgetting, where newly acquired knowledge interferes with previously learned capabilities. Despite widespread observations of this phenomenon, the mechanistic understanding remains limited. Here, we present a comprehensive mechanistic analysis of catastrophic forgetting in transformer-based LLMs during sequential fine-tuning. Through systematic experiments across multiple model scales (109B to 400B total parameters) and task sequences, we identify three primary mechanisms driving forgetting: gradient interference in attention weights, representational drift in intermediate layers, and loss landscape flattening. We demonstrate that forgetting severity correlates strongly with task similarity (Pearson r = 0.87) and gradient alignment metrics. Our analysis reveals that approximately 15 to 23 percent of attention heads undergo severe disruption during fine-tuning, with lower layers showing greater susceptibility. These findings establish mechanistic foundations for developing targeted mitigation strategies in continual learning systems.
>
---
#### [new 165] Structure-Aware NL-to-SQL for SFC Provisioning via AST-Masking Empowered Language Models
- **分类: cs.NI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言到SQL的生成任务，旨在解决SFC配置中语法不一致和效率低的问题。通过引入AST-Masking方法提升SQL生成准确性。**

- **链接: [https://arxiv.org/pdf/2601.17295v1](https://arxiv.org/pdf/2601.17295v1)**

> **作者:** Xinyu Zhu; Parisa Fard Moshiri; Poonam Lohan; Burak Kantarci; Emil Janulewicz
>
> **备注:** 6 pages, 3 figures, accepted to IEEE International Conference on Communications (ICC) 2026
>
> **摘要:** Effective Service Function Chain (SFC) provisioning requires precise orchestration in dynamic and latency-sensitive networks. Reinforcement Learning (RL) improves adaptability but often ignores structured domain knowledge, which limits generalization and interpretability. Large Language Models (LLMs) address this gap by translating natural language (NL) specifications into executable Structured Query Language (SQL) commands for specification-driven SFC management. Conventional fine-tuning, however, can cause syntactic inconsistencies and produce inefficient queries. To overcome this, we introduce Abstract Syntax Tree (AST)-Masking, a structure-aware fine-tuning method that uses SQL ASTs to assign weights to key components and enforce syntax-aware learning without adding inference overhead. Experiments show that AST-Masking significantly improves SQL generation accuracy across multiple language models. FLAN-T5 reaches an Execution Accuracy (EA) of 99.6%, while Gemma achieves the largest absolute gain from 7.5% to 72.0%. These results confirm the effectiveness of structure-aware fine-tuning in ensuring syntactically correct and efficient SQL generation for interpretable SFC orchestration.
>
---
#### [new 166] PEARL: Prototype-Enhanced Alignment for Label-Efficient Representation Learning with Deployment-Driven Insights from Digital Governance Communication Systems
- **分类: cs.LG; cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出PEARL方法，解决标签稀缺下的表示学习问题，通过增强原型对齐提升嵌入质量，优化近邻检索性能。**

- **链接: [https://arxiv.org/pdf/2601.17495v1](https://arxiv.org/pdf/2601.17495v1)**

> **作者:** Ruiyu Zhang; Lin Nie; Wai-Fung Lam; Qihao Wang; Xin Zhao
>
> **备注:** 15 pages, 1 figure
>
> **摘要:** In many deployed systems, new text inputs are handled by retrieving similar past cases, for example when routing and responding to citizen messages in digital governance platforms. When these systems fail, the problem is often not the language model itself, but that the nearest neighbors in the embedding space correspond to the wrong cases. Modern machine learning systems increasingly rely on fixed, high-dimensional embeddings produced by large pretrained models and sentence encoders. In real-world deployments, labels are scarce, domains shift over time, and retraining the base encoder is expensive or infeasible. As a result, downstream performance depends heavily on embedding geometry. Yet raw embeddings are often poorly aligned with the local neighborhood structure required by nearest-neighbor retrieval, similarity search, and lightweight classifiers that operate directly on embeddings. We propose PEARL (Prototype-Enhanced Aligned Representation Learning), a label-efficient approach that uses limited supervision to softly align embeddings toward class prototypes. The method reshapes local neighborhood geometry while preserving dimensionality and avoiding aggressive projection or collapse. Its aim is to bridge the gap between purely unsupervised post-processing, which offers limited and inconsistent gains, and fully supervised projections that require substantial labeled data. We evaluate PEARL under controlled label regimes ranging from extreme label scarcity to higher-label settings. In the label-scarce condition, PEARL substantially improves local neighborhood quality, yielding 25.7% gains over raw embeddings and more than 21.1% gains relative to strong unsupervised post-processing, precisely in the regime where similarity-based systems are most brittle.
>
---
#### [new 167] Capturing P: On the Expressive Power and Efficient Evaluation of Boolean Retrieval
- **分类: cs.IR; cs.AI; cs.CC; cs.CL; cs.DB**

- **简介: 该论文属于信息检索任务，旨在解决复杂逻辑查询的效率问题。提出一种新的检索语言和算法，实现高效查询处理。**

- **链接: [https://arxiv.org/pdf/2601.18747v1](https://arxiv.org/pdf/2601.18747v1)**

> **作者:** Amir Aavani
>
> **摘要:** Modern information retrieval is transitioning from simple document filtering to complex, neuro-symbolic reasoning workflows. However, current retrieval architectures face a fundamental efficiency dilemma when handling the rigorous logical and arithmetic constraints required by this new paradigm. Standard iterator-based engines (Document-at-a-Time) do not natively support complex, nested logic graphs; forcing them to execute such queries typically results in intractable runtime performance. Conversely, naive recursive approaches (Term-at-a-Time), while capable of supporting these structures, suffer from prohibitive memory consumption when enforcing broad logical exclusions. In this paper, we propose that a retrieval engine must be capable of ``Capturing $\mathbf{P}$'' -- evaluating any polynomial-time property directly over its index in a computationally efficient manner. We define a formal Retrieval Language ($\mathcal{L}_R$) based on Directed Acyclic Graphs (DAGs) and prove it precisely captures the complexity class $\mathbf{P}$. We introduce \texttt{ComputePN}, a novel evaluation algorithm that makes $\mathcal{L}_R$ tractable. By combining native DAG traversal with a memory-efficient ``Positive-Negative'' response mechanism, \texttt{ComputePN} ensures the efficient evaluation of any query in $\mathcal{L}_R$. This work establishes the theoretical foundation for turning the search index into a general-purpose computational engine.
>
---
#### [new 168] Emergence of Phonemic, Syntactic, and Semantic Representations in Artificial Neural Networks
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语言理解任务，研究人工神经网络中语音、词汇和语法表征的形成过程，旨在揭示语言习得的计算机制。**

- **链接: [https://arxiv.org/pdf/2601.18617v1](https://arxiv.org/pdf/2601.18617v1)**

> **作者:** Pierre Orhan; Pablo Diego-Simón; Emmnanuel Chemla; Yair Lakretz; Yves Boubenec; Jean-Rémi King
>
> **摘要:** During language acquisition, children successively learn to categorize phonemes, identify words, and combine them with syntax to form new meaning. While the development of this behavior is well characterized, we still lack a unifying computational framework to explain its underlying neural representations. Here, we investigate whether and when phonemic, lexical, and syntactic representations emerge in the activations of artificial neural networks during their training. Our results show that both speech- and text-based models follow a sequence of learning stages: during training, their neural activations successively build subspaces, where the geometry of the neural activations represents phonemic, lexical, and syntactic structure. While this developmental trajectory qualitatively relates to children's, it is quantitatively different: These algorithms indeed require two to four orders of magnitude more data for these neural representations to emerge. Together, these results show conditions under which major stages of language acquisition spontaneously emerge, and hence delineate a promising path to understand the computations underpinning language acquisition.
>
---
#### [new 169] treaming-dLLM: Accelerating Diffusion LLMs via Suffix Pruning and Dynamic Decoding
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于自然语言生成任务，针对扩散大语言模型推理效率低的问题，提出Streaming-dLLM框架，通过剪枝和动态解码提升速度，保持生成质量。**

- **链接: [https://arxiv.org/pdf/2601.17917v1](https://arxiv.org/pdf/2601.17917v1)**

> **作者:** Zhongyu Xiao; Zhiwei Hao; Jianyuan Guo; Yong Luo; Jia Liu; Jie Xu; Han Hu
>
> **备注:** Tech report. Code is available at https://github.com/xiaoshideta/Streaming-dLLM
>
> **摘要:** Diffusion Large Language Models (dLLMs) offer a compelling paradigm for natural language generation, leveraging parallel decoding and bidirectional attention to achieve superior global coherence compared to autoregressive models. While recent works have accelerated inference via KV cache reuse or heuristic decoding, they overlook the intrinsic inefficiencies within the block-wise diffusion process. Specifically, they suffer from spatial redundancy by modeling informative-sparse suffix regions uniformly and temporal inefficiency by applying fixed denoising schedules across all the decoding process. To address this, we propose Streaming-dLLM, a training-free framework that streamlines inference across both spatial and temporal dimensions. Spatially, we introduce attenuation guided suffix modeling to approximate the full context by pruning redundant mask tokens. Temporally, we employ a dynamic confidence aware strategy with an early exit mechanism, allowing the model to skip unnecessary iterations for converged tokens. Extensive experiments show that Streaming-dLLM achieves up to 68.2X speedup while maintaining generation quality, highlighting its effectiveness in diffusion decoding. The code is available at https://github.com/xiaoshideta/Streaming-dLLM.
>
---
#### [new 170] Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大语言模型推理优化任务，旨在解决KV缓存管理效率问题。提出一种基于门控的KV淘汰方法，在保持性能的同时大幅压缩缓存。**

- **链接: [https://arxiv.org/pdf/2601.17668v1](https://arxiv.org/pdf/2601.17668v1)**

> **作者:** Jang-Hyun Kim; Dongyoon Han; Sangdoo Yun
>
> **摘要:** Efficient key-value (KV) cache management is crucial for the practical deployment of large language models (LLMs), yet existing compression techniques often incur a trade-off between performance degradation and computational overhead. We propose a novel gating-based KV cache eviction method for frozen-weight LLMs that achieves high compression ratios with negligible computational cost. Our approach introduces lightweight sink-attention gating modules to identify and retain critical KV pairs, and integrates seamlessly into both the prefill and decoding stages. The proposed gate training algorithm relies on forward passes of an LLM, avoiding expensive backpropagation, while achieving strong task generalization through a task-agnostic reconstruction objective. Extensive experiments across the Qwen2.5-1M, Qwen3, and Gemma3 families show that our method maintains near-lossless performance while evicting up to 70% of the KV cache. The results are consistent across a wide range of tasks, including long-context understanding, code comprehension, and mathematical reasoning, demonstrating the generality of our approach.
>
---
#### [new 171] Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决大模型在低成功率数据集上的学习瓶颈。通过设计SOAR框架，让模型自动生成教学内容，提升自身性能。**

- **链接: [https://arxiv.org/pdf/2601.18778v1](https://arxiv.org/pdf/2601.18778v1)**

> **作者:** Shobhita Sundaram; John Quan; Ariel Kwiatkowski; Kartik Ahuja; Yann Ollivier; Julia Kempe
>
> **摘要:** Can a model learn to escape its own learning plateau? Reinforcement learning methods for finetuning large reasoning models stall on datasets with low initial success rates, and thus little training signal. We investigate a fundamental question: Can a pretrained LLM leverage latent knowledge to generate an automated curriculum for problems it cannot solve? To explore this, we design SOAR: A self-improvement framework designed to surface these pedagogical signals through meta-RL. A teacher copy of the model proposes synthetic problems for a student copy, and is rewarded with its improvement on a small subset of hard problems. Critically, SOAR grounds the curriculum in measured student progress rather than intrinsic proxy rewards. Our study on the hardest subsets of mathematical benchmarks (0/128 success) reveals three core findings. First, we show that it is possible to realize bi-level meta-RL that unlocks learning under sparse, binary rewards by sharpening a latent capacity of pretrained models to generate useful stepping stones. Second, grounded rewards outperform intrinsic reward schemes used in prior LLM self-play, reliably avoiding the instability and diversity collapse modes they typically exhibit. Third, analyzing the generated questions reveals that structural quality and well-posedness are more critical for learning progress than solution correctness. Our results suggest that the ability to generate useful stepping stones does not require the preexisting ability to actually solve the hard problems, paving a principled path to escape reasoning plateaus without additional curated data.
>
---
#### [new 172] To Case or Not to Case: An Empirical Study in Learned Sparse Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，研究cased与uncased模型对稀疏检索的影响。解决cased模型在LSR中表现不佳的问题，通过实验发现下采样可提升性能。**

- **链接: [https://arxiv.org/pdf/2601.17500v1](https://arxiv.org/pdf/2601.17500v1)**

> **作者:** Emmanouil Georgios Lionis; Jia-Huei Ju; Angelos Nalmpantis; Casper Thuis; Sean MacAvaney; Andrew Yates
>
> **备注:** This preprint has not undergone peer review (when applicable) or any post-submission improvements or corrections. The Version of Record of this contribution is published in ECIR2026 (Part I) Advances in Information Retrieval
>
> **摘要:** Learned Sparse Retrieval (LSR) methods construct sparse lexical representations of queries and documents that can be efficiently searched using inverted indexes. Existing LSR approaches have relied almost exclusively on uncased backbone models, whose vocabularies exclude case-sensitive distinctions, thereby reducing vocabulary mismatch. However, the most recent state-of-the-art language models are only available in cased versions. Despite this shift, the impact of backbone model casing on LSR has not been studied, potentially posing a risk to the viability of the method going forward. To fill this gap, we systematically evaluate paired cased and uncased versions of the same backbone models across multiple datasets to assess their suitability for LSR. Our findings show that LSR models with cased backbone models by default perform substantially worse than their uncased counterparts; however, this gap can be eliminated by pre-processing the text to lowercase. Moreover, our token-level analysis reveals that, under lowercasing, cased models almost entirely suppress cased vocabulary items and behave effectively as uncased models, explaining their restored performance. This result broadens the applicability of recent cased models to the LSR setting and facilitates the integration of stronger backbone architectures into sparse retrieval. The complete code and implementation for this project are available at: https://github.com/lionisakis/Uncased-vs-cased-models-in-LSR
>
---
#### [new 173] POPE: Learning to Reason on Hard Problems via Privileged On-Policy Exploration
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出POPE方法，解决强化学习在难题上探索不足的问题。通过引入特权信息引导探索，提升模型在复杂推理任务中的表现。**

- **链接: [https://arxiv.org/pdf/2601.18779v1](https://arxiv.org/pdf/2601.18779v1)**

> **作者:** Yuxiao Qu; Amrith Setlur; Virginia Smith; Ruslan Salakhutdinov; Aviral Kumar
>
> **摘要:** Reinforcement learning (RL) has improved the reasoning abilities of large language models (LLMs), yet state-of-the-art methods still fail to learn on many training problems. On hard problems, on-policy RL rarely explores even a single correct rollout, yielding zero reward and no learning signal for driving improvement. We find that natural solutions to remedy this exploration problem from classical RL, such as entropy bonuses, more permissive clipping of the importance ratio, or direct optimization of pass@k objectives, do not resolve this issue and often destabilize optimization without improving solvability. A natural alternative is to leverage transfer from easier problems. However, we show that mixing easy and hard problems during RL training is counterproductive due to ray interference, where optimization focuses on already-solvable problems in a way that actively inhibits progress on harder ones. To address this challenge, we introduce Privileged On-Policy Exploration (POPE), an approach that leverages human- or other oracle solutions as privileged information to guide exploration on hard problems, unlike methods that use oracle solutions as training targets (e.g., off-policy RL methods or warmstarting from SFT). POPE augments hard problems with prefixes of oracle solutions, enabling RL to obtain non-zero rewards during guided rollouts. Crucially, the resulting behaviors transfer back to the original, unguided problems through a synergy between instruction-following and reasoning. Empirically, POPE expands the set of solvable problems and substantially improves performance on challenging reasoning benchmarks.
>
---
## 更新

#### [replaced 001] Understanding Parametric Knowledge Injection in Retrieval-Augmented Generation
- **分类: cs.IR; cs.CL**

- **简介: 该论文研究参数化知识注入在检索增强生成中的应用，旨在解决长文本生成中上下文丢失问题。通过对比分析，提出结合参数与令牌方法的PT-RAG模型，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2510.12668v2](https://arxiv.org/pdf/2510.12668v2)**

> **作者:** Minghao Tang; Shiyu Ni; Jingtong Wu; Zengxin Han; Keping Bi
>
> **摘要:** Context-grounded generation underpins many LLM applications, including long-document question answering (QA), conversational personalization, and retrieval-augmented generation (RAG). However, classic token-based context concatenation is costly for long inputs and can be lost in the middle at extreme context lengths. Recent work explores context parameterization, which encodes context into lightweight trainable parameters (e.g., LoRA adapters) injected into a frozen LLM. Extending this idea to retrieved evidence yields parametric RAG (P-RAG), which incorporates knowledge via parameter updates rather than token-level attention. In this paper, we present a systematic study of this emerging RAG paradigm-parametric knowledge injection. First, we reassess P-RAG under answer-presence accuracy and show that it does not consistently outperform standard token-based RAG (T-RAG), while combining both (PT-RAG) achieves the best overall performance. Second, we introduce a QA benchmark with up-to-date knowledge beyond the LLM's internal memory to enable controlled analysis. Our representational and mechanistic results indicate that parametric representations capture document-level semantics and primarily influence deeper feed-forward computations, providing high-level guidance but limited evidence consolidation. Finally, we evaluate parametric injection under key RAG challenges, demonstrating improved faithfulness under knowledge conflicts, stronger robustness to retrieval noise, and solid generalization to tasks beyond QA. Our findings clarify the strengths and limitations of parametric RAG and provide practical guidance for future retrieval-augmented LLM systems.
>
---
#### [replaced 002] Evaluating Perspectival Biases in Cross-Modal Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于多模态检索任务，旨在解决语言和文化偏见影响检索公平性的问题。通过构建3XCM基准，研究发现模型倾向于依赖语言流行度和文化关联，提出需分离语言与文化因素以提升公平性。**

- **链接: [https://arxiv.org/pdf/2510.26861v3](https://arxiv.org/pdf/2510.26861v3)**

> **作者:** Teerapol Saengsukhiran; Peerawat Chomphooyod; Narabodee Rodjananant; Chompakorn Chaksangchaichot; Patawee Prakrankamanant; Witthawin Sripheanpol; Pak Lovichit; Sarana Nutanong; Ekapol Chuangsuwanich
>
> **摘要:** Multimodal retrieval systems are expected to operate in a semantic space, agnostic to the language or cultural origin of the query. In practice, however, retrieval outcomes systematically reflect perspectival biases: deviations shaped by linguistic prevalence and cultural associations. We introduce the Cross-Cultural, Cross-Modal, Cross-lingual Multimodal (3XCM) benchmark to isolate these effects. Results from our studies indicate that, for image-to-text retrieval, models tend to favor entries from prevalent languages over those that are semantically faithful. For text-to-image retrieval, we observe a consistent "tugging effect" in the joint embedding space between semantic alignment and language-conditioned cultural association. When semantic representations are insufficiently resolved, particularly in low-resource languages, similarity is increasingly governed by culturally familiar visual patterns, leading to systematic association bias in retrieval. Our findings suggest that achieving equitable multimodal retrieval necessitates targeted strategies that explicitly decouple language from culture, rather than relying solely on broader data exposure. This work highlights the need to treat linguistic and cultural biases as distinct, measurable challenges in multimodal representation learning.
>
---
#### [replaced 003] How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的逻辑推理任务，旨在解决语言模型将合理性与有效性混淆的问题。通过分析模型内部表示，发现两者在几何上高度对齐，并提出去偏方法提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2510.06700v2](https://arxiv.org/pdf/2510.06700v2)**

> **作者:** Leonardo Bertolazzi; Sandro Pezzelle; Raffaella Bernardi
>
> **摘要:** Both humans and large language models (LLMs) exhibit content effects: biases in which the plausibility of the semantic content of a reasoning problem influences judgments regarding its logical validity. While this phenomenon in humans is best explained by the dual-process theory of reasoning, the mechanisms behind content effects in LLMs remain unclear. In this work, we address this issue by investigating how LLMs encode the concepts of validity and plausibility within their internal representations. We show that both concepts are linearly represented and strongly aligned in representational geometry, leading models to conflate plausibility with validity. Using steering vectors, we demonstrate that plausibility vectors can causally bias validity judgements, and vice versa, and that the degree of alignment between these two concepts predicts the magnitude of behavioral content effects across models. Finally, we construct debiasing vectors that disentangle these concepts, reducing content effects and improving reasoning accuracy. Our findings advance understanding of how abstract logical concepts are represented in LLMs and highlight representational interventions as a path toward more logical systems.
>
---
#### [replaced 004] MemoryRewardBench: Benchmarking Reward Models for Long-Term Memory Management in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大模型记忆管理评估任务，旨在解决如何有效评价长序列记忆质量的问题。提出MemoryRewardBench基准，测试多种记忆管理模式下的奖励模型性能。**

- **链接: [https://arxiv.org/pdf/2601.11969v2](https://arxiv.org/pdf/2601.11969v2)**

> **作者:** Zecheng Tang; Baibei Ji; Ruoxi Sun; Haitian Wang; WangJie You; Zhang Yijun; Wenpeng Zhu; Ji Qi; Juntao Li; Min Zhang
>
> **摘要:** Existing works increasingly adopt memory-centric mechanisms to process long contexts in a segment manner, and effective memory management is one of the key capabilities that enables large language models to effectively propagate information across the entire sequence. Therefore, leveraging reward models (RMs) to automatically and reliably evaluate memory quality is critical. In this work, we introduce MemoryRewardBench, the first benchmark to systematically study the ability of RMs to evaluate long-term memory management processes. MemoryRewardBench covers both long-context comprehension and long-form generation tasks, featuring 10 distinct settings with different memory management patterns, with context length ranging from 8K to 128K tokens. Evaluations on 13 cutting-edge RMs indicate a diminishing performance gap between open-source and proprietary models, with newer-generation models consistently outperforming their predecessors regardless of parameter count. We further expose the capabilities and fundamental limitations of current RMs in evaluating LLM memory management across diverse settings.
>
---
#### [replaced 005] Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，研究RLVR中探索与利用的平衡问题，探讨熵、剪切偏差和虚假奖励的作用机制。**

- **链接: [https://arxiv.org/pdf/2512.16912v3](https://arxiv.org/pdf/2512.16912v3)**

> **作者:** Peter Chen; Xiaopeng Li; Ziniu Li; Wotao Yin; Xi Chen; Tianyi Lin
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** This paper examines the exploration-exploitation trade-off in reinforcement learning with verifiable rewards (RLVR), a framework for improving the reasoning of Large Language Models (LLMs). Recent studies suggest that RLVR can elicit strong mathematical reasoning in LLMs through two seemingly paradoxical mechanisms: spurious rewards, which suppress exploitation by rewarding outcomes unrelated to the ground truth, and entropy minimization, which suppresses exploration by pushing the model toward more confident and deterministic outputs, highlighting a puzzling dynamic: both discouraging exploitation and discouraging exploration improve reasoning performance, yet the underlying principles that reconcile these effects remain poorly understood. We focus on two fundamental questions: (i) how policy entropy relates to performance, and (ii) whether spurious rewards yield gains, potentially through the interplay of clipping bias and model contamination. Our results show that clipping bias under spurious rewards reduces policy entropy, leading to more confident and deterministic outputs, while entropy minimization alone is insufficient for improvement. We further propose a reward-misalignment model explaining why spurious rewards can enhance performance beyond contaminated settings. Our findings clarify the mechanisms behind spurious-reward benefits and provide principles for more effective RLVR training.
>
---
#### [replaced 006] Large Language Models as Proxies for Theories of Human Linguistic Cognition
- **分类: cs.CL**

- **简介: 该论文属于认知科学与语言学交叉研究，探讨LLMs能否作为人类语言认知理论的代理模型，解决理论验证问题。**

- **链接: [https://arxiv.org/pdf/2502.07687v2](https://arxiv.org/pdf/2502.07687v2)**

> **作者:** Imry Ziv; Nur Lan; Emmanuel Chemla; Roni Katzir
>
> **摘要:** We consider the possible role of current large language models (LLMs) in the study of human linguistic cognition. We focus on the use of such models as proxies for theories of cognition that are relatively linguistically-neutral in their representations and learning but differ from current LLMs in key ways. We illustrate this potential use of LLMs as proxies for theories of cognition in the context of two kinds of questions: (a) whether the target theory accounts for the acquisition of a given pattern from a given corpus; and (b) whether the target theory makes a given typologically-attested pattern easier to acquire than another, typologically-unattested pattern. For each of the two questions we show, building on recent literature, how current LLMs can potentially be of help, but we note that at present this help is quite limited.
>
---
#### [replaced 007] Measuring AI "Slop" in Text
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在定义和测量AI生成文本中的“低质量”现象。通过构建评估维度，提出一种可解释的框架以评估AI文本质量。**

- **链接: [https://arxiv.org/pdf/2509.19163v2](https://arxiv.org/pdf/2509.19163v2)**

> **作者:** Chantal Shaib; Tuhin Chakrabarty; Diego Garcia-Olano; Byron C. Wallace
>
> **摘要:** AI "slop" is an increasingly popular term used to describe low-quality AI-generated text, but there is currently no agreed upon definition of this term nor a means to measure its occurrence. In this work, we develop a taxonomy of "slop" through interviews with experts in NLP, writing, and philosophy, and propose a set of interpretable dimensions for its assessment in text. Through span-level annotation, we find that binary "slop" judgments are (somewhat) subjective, but such determinations nonetheless correlate with latent dimensions such as coherence and relevance. Our framework can be used to evaluate AI-generated text in both detection and binary preference tasks, potentially offering new insights into the linguistic and stylistic factors that contribute to quality judgments.
>
---
#### [replaced 008] Transparent Semantic Change Detection with Dependency-Based Profiles
- **分类: cs.CL**

- **简介: 该论文属于词汇语义变化检测任务，旨在解决传统方法透明度不足的问题。通过依赖关系共现模式进行语义变化检测，效果优于分布模型，并具备可解释性。**

- **链接: [https://arxiv.org/pdf/2601.02891v2](https://arxiv.org/pdf/2601.02891v2)**

> **作者:** Bach Phan-Tat; Kris Heylen; Dirk Geeraerts; Stefano De Pascale; Dirk Speelman
>
> **摘要:** Most modern computational approaches to lexical semantic change detection (LSC) rely on embedding-based distributional word representations with neural networks. Despite the strong performance on LSC benchmarks, they are often opaque. We investigate an alternative method which relies purely on dependency co-occurrence patterns of words. We demonstrate that it is effective for semantic change detection and even outperforms a number of distributional semantic models. We provide an in-depth quantitative and qualitative analysis of the predictions, showing that they are plausible and interpretable.
>
---
#### [replaced 009] ABCD-LINK: Annotation Bootstrapping for Cross-Document Fine-Grained Links
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文提出ABCD-LINK框架，解决跨文档细粒度链接问题。通过生成数据、评估模型并结合人工标注，提升链接准确率，支持媒体分析等任务。**

- **链接: [https://arxiv.org/pdf/2509.01387v2](https://arxiv.org/pdf/2509.01387v2)**

> **作者:** Serwar Basch; Ilia Kuznetsov; Tom Hope; Iryna Gurevych
>
> **备注:** Accepted at EACL 2026
>
> **摘要:** Understanding fine-grained links between documents is crucial for many applications, yet progress is limited by the lack of efficient methods for data curation. To address this limitation, we introduce a domain-agnostic framework for bootstrapping sentence-level cross-document links from scratch. Our approach (1) generates and validates semi-synthetic datasets of linked documents, (2) uses these datasets to benchmark and shortlist the best-performing linking approaches, and (3) applies the shortlisted methods in large-scale human-in-the-loop annotation of natural text pairs. We apply the framework in two distinct domains -- peer review and news -- and show that combining retrieval models with LLMs achieves a 73% human approval rate for suggested links, more than doubling the acceptance of strong retrievers alone. Our framework allows users to produce novel datasets that enable systematic study of cross-document understanding, supporting downstream tasks such as media framing analysis and peer review assessment. All code, data, and annotation protocols are released to facilitate future research.
>
---
#### [replaced 010] RoboPhD: Self-Improving Text-to-SQL Through Autonomous Agent Evolution
- **分类: cs.CL**

- **简介: 该论文提出RoboPhD系统，通过AI代理自主进化提升Text-to-SQL性能。解决AI系统自我优化问题，通过迭代演化发现有效策略，显著提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2601.01126v2](https://arxiv.org/pdf/2601.01126v2)**

> **作者:** Andrew Borthwick; Stephen Ash
>
> **备注:** 18 pages, 3 figures
>
> **摘要:** We present RoboPhD, a system where AI agents autonomously conduct research to improve Text-to-SQL performance. RoboPhD implements a closed-loop evolution cycle with two coordinated components: a SQL Generation agent composed of a database analysis script and SQL generation instructions, and an Evolution agent that designs new versions based on performance feedback. Central to the framework is an ELO-based selection mechanism enabling survival-of-the-fittest dynamics while handling non-transitivity in performance. Starting from a naive 70-line baseline, RoboPhD evolves agents through iterative cross-pollination, discovering effective techniques without any external guidance on the Text-to-SQL domain. Our best agent, evolved to 1500 lines over 18 iterations, autonomously discovered strategies such as size-adaptive database analysis that adjusts depth based on schema complexity and SQL generation patterns for column selection, evidence interpretation, and aggregation. Evolution provides the largest gains on cheaper models: while we improve by 2.3 points over a strong Claude Opus 4.5 naive baseline, we show an improvement of 8.9 points over the weaker Claude Haiku model. This enables 'skip a tier' deployment: evolved Haiku exceeds naive Sonnet accuracy, and evolved Sonnet exceeds naive Opus, both at lower cost. The full system achieves 73.67% accuracy on the BIRD test set, demonstrating that AI can autonomously build a strong agentic system with only a trivial human-provided starting point.
>
---
#### [replaced 011] Exploring LGBTQ+ Bias in Generative AI Answers across Different Country and Religious Contexts
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于AI伦理研究任务，旨在探讨生成式AI在不同文化和宗教背景下对LGBTQ+的偏见。通过分析ChatGPT 3.5和Bard的回应，揭示AI系统如何根据上下文调整对LGBTQ+的支持程度。**

- **链接: [https://arxiv.org/pdf/2407.03473v2](https://arxiv.org/pdf/2407.03473v2)**

> **作者:** Lilla Vicsek; Anna Vancsó; Mike Zajko; Judit Takacs
>
> **备注:** Replacement version -- includes link to BD&S journal publication (significantly revised) in abstract, but the manuscript here remains unchanged from the original arXiv version
>
> **摘要:** Previous discussions have highlighted the need for generative AI tools to become more culturally sensitive, yet often neglect the complexities of handling content about minorities, who are perceived differently across cultures and religions. Our study examined how two generative AI systems respond to homophobic statements with varying cultural and religious context information. Findings showed ChatGPT 3.5's replies exhibited cultural relativism, in contrast to Bard's, which stressed human rights and provided more support for LGBTQ+ issues. Both demonstrated significant change in responses based on contextual information provided in the prompts, suggesting that AI systems may adjust in their responses the degree and forms of support for LGBTQ+ people according to information they receive about the user's background. The study contributes to understanding the social and ethical implications of AI responses and argues that any work to make generative AI outputs more culturally diverse requires a grounding in fundamental human rights. A revised edition of this preprint is available open access at Big Data & Society at https://doi.org/10.1177/20539517251396069
>
---
#### [replaced 012] Your Extreme Multi-label Classifier is Secretly a Hierarchical Text Classifier for Free
- **分类: cs.CL**

- **简介: 该论文研究将极端多标签分类（XML）与层次文本分类（HTC）模型相互测试。论文属于文本分类任务，旨在探讨两种模型在对方数据集上的表现及公平比较方法。**

- **链接: [https://arxiv.org/pdf/2411.13687v3](https://arxiv.org/pdf/2411.13687v3)**

> **作者:** Nerijus Bertalis; Paul Granse; Ferhat Gül; Florian Hauss; Leon Menkel; David Schüler; Tom Speier; Lukas Galke; Ansgar Scherp
>
> **摘要:** Assigning a set of labels to a given text is a classification problem with many real-world applications, such as recommender systems. Two separate research streams address this issue. Hierarchical Text Classification (HTC) focuses on datasets with label pools of hundreds of entries, accompanied by a semantic label hierarchy. In contrast, eXtreme Multi-Label Text Classification (XML) considers very large sets of labels with up to millions of entries but without an explicit hierarchy. In XML methods, it is common to construct an artificial hierarchy in order to deal with the large label space before or during the training process. Here, we investigate how state-of-the-art HTC models perform when trained and tested on XML datasets and vice versa using three benchmark datasets from each of the two streams. Our results demonstrate that XML models, with their internally constructed hierarchy, are very effective HTC models. HTC models, on the other hand, are not equipped to handle the sheer label set size of XML datasets and achieve poor transfer results. We further argue that for a fair comparison in HTC and XML, more than one metric like F1 should be used but complemented with P@k and R-Precision.
>
---
#### [replaced 013] CtrlRAG: Black-box Document Poisoning Attacks for Retrieval-Augmented Generation of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于安全任务，针对RAG系统提出黑盒攻击方法CtrlRAG，解决文档污染攻击问题，通过构造恶意文档并优化提升攻击效果。**

- **链接: [https://arxiv.org/pdf/2503.06950v2](https://arxiv.org/pdf/2503.06950v2)**

> **作者:** Runqi Sui
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems enhance response credibility and traceability by displaying reference contexts, but this transparency simultaneously introduces a novel black-box attack vector. Existing document poisoning attacks, where adversaries inject malicious documents into the knowledge base to manipulate RAG outputs, rely primarily on unrealistic white-box or gray-box assumptions, limiting their practical applicability. To address this gap, we propose CtrlRAG, a two-stage black-box attack that (1) constructs malicious documents containing misinformation or emotion-inducing content and injects them into the knowledge base, and (2) iteratively optimizes them using a localization algorithm and Masked Language Model (MLM) guided on reference context feedback, ensuring their retrieval priority while preserving linguistic naturalness. With only five malicious documents per target question injected into the million-document MS MARCO dataset, CtrlRAG achieves up to 90% attack success rates on commercial LLMs (e.g., GPT-4o), a 30% improvement over optimal baselines, in both *Emotion Manipulation* and *Hallucination Amplification* tasks. Furthermore, we show that existing defenses fail to balance security and performance. To mitigate this challenge, we introduce a dynamic *Knowledge Expansion* defense strategy based on *Parametric/Non-parametric Memory Confrontation*, blocking 78% of attacks while maintaining 95.5% system accuracy. Our findings reveal critical vulnerabilities in RAG systems and provide effective defense strategies.
>
---
#### [replaced 014] BeDiscovER: The Benchmark of Discourse Understanding in the Era of Reasoning Language Models
- **分类: cs.CL**

- **简介: 该论文提出BeDiscovER，一个评估语言模型话语理解能力的基准。旨在解决话语层面知识评估问题，涵盖多种任务和数据集。**

- **链接: [https://arxiv.org/pdf/2511.13095v2](https://arxiv.org/pdf/2511.13095v2)**

> **作者:** Chuyuan Li; Giuseppe Carenini
>
> **备注:** Camera-ready version of eacl 2026
>
> **摘要:** We introduce BeDiscovER (Benchmark of Discourse Understanding in the Era of Reasoning Language Models), an up-to-date, comprehensive suite for evaluating the discourse-level knowledge of modern LLMs. BeDiscovER compiles 5 publicly available discourse tasks across discourse lexicon, (multi-)sentential, and documental levels, with in total 52 individual datasets. It covers both extensively studied tasks such as discourse parsing and temporal relation extraction, as well as some novel challenges such as discourse particle disambiguation (e.g., ``just''), and also aggregates a shared task on Discourse Relation Parsing and Treebanking for multilingual and multi-framework discourse relation classification. We evaluate open-source LLMs: Qwen3 series, DeepSeek-R1, and frontier model such as GPT-5-mini on BeDiscovER, and find that state-of-the-art models exhibit strong performance in arithmetic aspect of temporal reasoning, but they struggle with full document reasoning and some subtle semantic and discourse phenomena, such as rhetorical relation recognition.
>
---
#### [replaced 015] HeartLLM: Discretized ECG Tokenization for LLM-Based Diagnostic Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出HeartLLM，将ECG信号转化为离散token，用于医学诊断推理。任务是将ECG与自然语言处理结合，解决模型泛化和开放性推理问题。工作包括编码、量化、预训练和指令微调。**

- **链接: [https://arxiv.org/pdf/2508.15338v2](https://arxiv.org/pdf/2508.15338v2)**

> **作者:** Jinning Yang; Wenjie Sun; Wen Shi
>
> **摘要:** Electrocardiography (ECG) plays a central role in cardiovascular diagnostics, yet existing automated approaches often struggle to generalize across clinical tasks and offer limited support for open-ended reasoning. We present HeartLLM, a novel framework that integrates time-series (TS) and language modeling by enabling large language models (LLMs) to process 12-lead ECG signals for clinical text generation tasks. Our approach discretizes continuous ECG embeddings into quantized codes using a lead-wise encoder and quantization module. These quantized codes are then mapped to an extended ECG vocabulary to form ECG tokens, enabling the model to process both ECG and natural language inputs within a unified framework. To bridge the modality gap, we pretrain the model on an autoregressive ECG token forecasting task, allowing the LLM to capture temporal dynamics through its inherent language modeling capability. Finally, we perform instruction tuning on both ECG question answering and diagnostic report generation. Without modifying the core model, HeartLLM achieves strong performance across tasks while maintaining generalization to out-of-distribution settings. Extensive experiments demonstrate the effectiveness of each component and highlight the potential of integrating discretized ECG tokens into LLMs for medical reasoning.
>
---
#### [replaced 016] Hearing Between the Lines: Unlocking the Reasoning Power of LLMs for Speech Evaluation
- **分类: cs.CL**

- **简介: 该论文属于语音评估任务，旨在解决现有方法依赖昂贵模型的问题。提出TRACE框架，利用LLM对音频线索进行推理，实现高效且符合人类判断的语音评估。**

- **链接: [https://arxiv.org/pdf/2601.13742v2](https://arxiv.org/pdf/2601.13742v2)**

> **作者:** Arjun Chandra; Kevin Miller; Venkatesh Ravichandran; Constantinos Papayiannis; Venkatesh Saligrama
>
> **备注:** EACL 2026 Findings
>
> **摘要:** Large Language Model (LLM) judges exhibit strong reasoning capabilities but are limited to textual content. This leaves current automatic Speech-to-Speech (S2S) evaluation methods reliant on opaque and expensive Audio Language Models (ALMs). In this work, we propose TRACE (Textual Reasoning over Audio Cues for Evaluation), a novel framework that enables LLM judges to reason over audio cues to achieve cost-efficient and human-aligned S2S evaluation. To demonstrate the strength of the framework, we first introduce a Human Chain-of-Thought (HCoT) annotation protocol to improve the diagnostic capability of existing judge benchmarks by separating evaluation into explicit dimensions: content (C), voice quality (VQ), and paralinguistics (P). Using this data, TRACE constructs a textual blueprint of inexpensive audio signals and prompts an LLM to render dimension-wise judgments, fusing them into an overall rating via a deterministic policy. TRACE achieves higher agreement with human raters than ALMs and transcript-only LLM judges while being significantly more cost-effective. We will release the HCoT annotations and the TRACE framework to enable scalable and human-aligned S2S evaluation.
>
---
#### [replaced 017] Detecting Training Data of Large Language Models via Expectation Maximization
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文属于隐私安全任务，解决模型训练数据成员推断问题。提出EM-MIA方法，通过期望最大化策略提升推断效果，无需标注非成员数据。**

- **链接: [https://arxiv.org/pdf/2410.07582v3](https://arxiv.org/pdf/2410.07582v3)**

> **作者:** Gyuwan Kim; Yang Li; Evangelia Spiliopoulou; Jie Ma; William Yang Wang
>
> **备注:** EACL 2026
>
> **摘要:** Membership inference attacks (MIAs) aim to determine whether a specific example was used to train a given language model. While prior work has explored prompt-based attacks such as ReCALL, these methods rely heavily on the assumption that using known non-members as prompts reliably suppresses the model's responses to non-member queries. We propose EM-MIA, a new membership inference approach that iteratively refines prefix effectiveness and membership scores using an expectation-maximization strategy without requiring labeled non-member examples. To support controlled evaluation, we introduce OLMoMIA, a benchmark that enables analysis of MIA robustness under systematically varied distributional overlap and difficulty. Experiments on WikiMIA and OLMoMIA show that EM-MIA outperforms existing baselines, particularly in settings with clear distributional separability. We highlight scenarios where EM-MIA succeeds in practical settings with partial distributional overlap, while failure cases expose fundamental limitations of current MIA methods under near-identical conditions. We release our code and evaluation pipeline to encourage reproducible and robust MIA research.
>
---
#### [replaced 018] VaseVQA: Multimodal Agent and Benchmark for Ancient Greek Pottery
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VaseVQA，一个用于古希腊陶器的多模态基准任务，解决文化遗产品牌理解难题。通过构建数据集和引入强化学习方法提升推理能力。**

- **链接: [https://arxiv.org/pdf/2509.17191v2](https://arxiv.org/pdf/2509.17191v2)**

> **作者:** Jinchao Ge; Tengfei Cheng; Biao Wu; Zeyu Zhang; Shiya Huang; Judith Bishop; Gillian Shepherd; Meng Fang; Ling Chen; Yang Zhao
>
> **摘要:** Understanding cultural heritage artifacts such as ancient Greek pottery requires expert-level reasoning that remains challenging for current MLLMs due to limited domain-specific data. We introduce VaseVQA, a benchmark of 31,773 images and 67,614 question-answer pairs across seven expert-defined categories, enabling systematic evaluation of expert-level cultural heritage understanding. Using this dataset, we explore effective training strategies for domain-specific reasoning. While supervised fine-tuning improves adaptation to domain knowledge, it struggles with deeper reasoning tasks. We propose VaseVL, which augments SFT with reinforcement learning using verifiable rewards. Experiments show that VaseVL consistently outperforms supervised baselines, especially on reasoning-intensive questions, highlighting the value of targeted reinforcement learning for cultural heritage visual question answering. Our code and dataset will be released at https://github.com/AIGeeksGroup/VaseVQA.
>
---
#### [replaced 019] Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，解决奖励稀疏下的样本选择问题。通过分析在线难度过滤，提出平衡筛选方法，提升学习效率与性能。**

- **链接: [https://arxiv.org/pdf/2504.03380v2](https://arxiv.org/pdf/2504.03380v2)**

> **作者:** Sanghwan Bae; Jiwoo Hong; Min Young Lee; Hanbyul Kim; JeongYeon Nam; Donghyun Kwak
>
> **备注:** To appear in EACL 2026 (main conference)
>
> **摘要:** Recent advances in reinforcement learning with verifiable rewards (RLVR) show that large language models enhance their reasoning abilities when trained with verifiable signals. However, due to reward sparsity, effectiveness depends heavily on selecting samples of appropriate difficulty. In this work, we present a formal analysis of online difficulty-aware filtering and establish its theoretical foundations. We show that expected policy improvement is lower-bounded by the variance of task-level success probabilities, implying that selecting tasks of intermediate difficulty maximizes learning efficiency. Building on this, we demonstrate that balanced filtering maximizes this lower bound, leading to superior performance and sample efficiency. Evaluations across multiple math reasoning benchmarks validate that balanced filtering consistently enhances convergence speed and final performance, achieving up to +12% gains in less than half the training steps of standard GRPO. By extending our analysis to various reward distributions, we provide a principled foundation for future RLVR curriculum strategies, confirmed through both theoretical analysis and extensive empirical results.
>
---
#### [replaced 020] Multi-Agent Collaborative Filtering: Orchestrating Users and Items for Agentic Recommendations
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于推荐系统任务，旨在解决传统协同过滤在agentic推荐中协作信号利用不足的问题。提出MACF框架，通过多代理协作提升推荐效果。**

- **链接: [https://arxiv.org/pdf/2511.18413v3](https://arxiv.org/pdf/2511.18413v3)**

> **作者:** Yu Xia; Sungchul Kim; Tong Yu; Ryan A. Rossi; Julian McAuley
>
> **备注:** WWW 2026
>
> **摘要:** Agentic recommendations cast recommenders as large language model (LLM) agents that can plan, reason, use tools, and interact with users of varying preferences in web applications. However, most existing agentic recommender systems focus on generic single-agent plan-execute workflows or multi-agent task decomposition pipelines. Without recommendation-oriented design, they often underuse the collaborative signals in the user-item interaction history, leading to unsatisfying recommendation results. To address this, we propose the Multi-Agent Collaborative Filtering (MACF) framework for agentic recommendations, drawing an analogy between traditional collaborative filtering algorithms and LLM-based multi-agent collaboration. Specifically, given a target user and query, we instantiate similar users and relevant items as LLM agents with unique profiles. Each agent is able to call retrieval tools, suggest candidate items, and interact with other agents. Different from the static preference aggregation in traditional collaborative filtering, MACF employs a central orchestrator agent to adaptively manage the collaboration between user and item agents via dynamic agent recruitment and personalized collaboration instruction. Experimental results on datasets from three different domains show the advantages of our MACF framework compared to strong agentic recommendation baselines.
>
---
#### [replaced 021] TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出TALE方法，用于在推理时去除对特定任务无关或有害的模型层，以提升任务性能并降低计算成本。属于模型优化任务，解决固定架构下层贡献不均的问题。**

- **链接: [https://arxiv.org/pdf/2510.22767v2](https://arxiv.org/pdf/2510.22767v2)**

> **作者:** Omar Naim; Krish Sharma; Niyar R Barman; Nicholas Asher
>
> **摘要:** Large Language Models (LLMs) are typically deployed using a fixed architecture, despite growing evidence that not all layers contribute equally to every downstream task. In this work, we introduce TALE (Task-Aware Layer Elimination), an inference-time method that improves task performance by selectively removing layers that are irrelevant or detrimental for a given task. TALE optimizes task-specific validation performance, yielding a task-adapted architecture without retraining or modifying model weights. Across 9 tasks and 5 model families, under both zero-shot and few-shot settings, we show that TALE consistently matches or surpasses baseline performance while simultaneously reducing computational cost, outperforming general and layer-wise pruning approaches such as SLEB. Beyond inference-time gains, TALE synergizes with fine-tuning and few-shot learning, where task-adapted architectures lead to additional performance improvements. Computing TALE for a new task requires modest resources (1-2 GPU hours on an A100), making it a practical and deployable solution for task-specialized LLM inference.
>
---
#### [replaced 022] Filling the Gap: Is Commonsense Knowledge Generation useful for Natural Language Inference?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言推理任务，探讨LLM生成常识知识对NLI的帮助。研究通过实验验证选择性引入常识公理能提升模型性能，解决模型对中立类的偏见问题。**

- **链接: [https://arxiv.org/pdf/2507.15100v2](https://arxiv.org/pdf/2507.15100v2)**

> **作者:** Chathuri Jayaweera; Brianna Yanqui; Bonnie Dorr
>
> **备注:** Under-review Annual Meeting of the Association for Computational Linguistics (ACL 2026): 10 pages, 7 figures, and 4 tables
>
> **摘要:** Natural Language Inference (NLI) is the task of determining whether a premise entails, contradicts, or is neutral with respect to a given hypothesis. The task is often framed as emulating human inferential processes, in which commonsense knowledge plays a major role. This study examines whether Large Language Models (LLMs) can generate useful commonsense axioms for Natural Language Inference, and evaluates their impact on performance using the SNLI and ANLI benchmarks with the Llama-3.1-70B and gpt-oss-120b models. We show that a hybrid approach, which selectively provides highly factual axioms based on judged helpfulness, yields consistent accuracy improvements of 1.99% to 6.88% across tested configurations, demonstrating the effectiveness of selective knowledge access for NLI. We also find that this targeted use of commonsense knowledge helps models overcome a bias toward the Neutral class by providing essential real-world context.
>
---
#### [replaced 023] A Linear Expectation Constraint for Selective Prediction and Routing with False-Discovery Control
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于模型预测与路由任务，解决基础模型输出不可靠的问题。提出LEC框架，通过FDR控制确保预测可靠性，提升样本保留率。**

- **链接: [https://arxiv.org/pdf/2512.01556v2](https://arxiv.org/pdf/2512.01556v2)**

> **作者:** Zhiyuan Wang; Aniri; Tianlong Chen; Yue Zhang; Heng Tao Shen; Xiaoshuang Shi; Kaidi Xu
>
> **摘要:** Foundation models often generate unreliable answers, while heuristic uncertainty estimators fail to fully distinguish correct from incorrect outputs, causing users to accept erroneous answers without statistical guarantees. We address this through the lens of false discovery rate (FDR) control, ensuring that among all accepted predictions, the proportion of errors does not exceed a target risk level. To this end, we propose LEC, a principled framework that reframes selective prediction as a decision problem governed by a linear expectation constraint over selection and error indicators. Under this formulation, we derive a finite-sample sufficient condition that relies only on a held-out set of exchangeable calibration data, enabling the computation of an FDR-constrained, retention-maximizing threshold. Furthermore, we extend LEC to two-model routing systems: if the primary model's uncertainty exceeds its calibrated threshold, the input is delegated to a subsequent model, while maintaining system-level FDR control. Experiments on both closed-ended and open-ended question answering (QA) and vision question answering (VQA) demonstrate that LEC achieves tighter FDR control and substantially improves sample retention compared to prior approaches.
>
---
#### [replaced 024] The Landscape of Agentic Reinforcement Learning for LLMs: A Survey
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI代理研究，解决LLM作为自主决策者的问题。通过分类和综述，提出Agentic RL框架，推动智能代理发展。**

- **链接: [https://arxiv.org/pdf/2509.02547v4](https://arxiv.org/pdf/2509.02547v4)**

> **作者:** Guibin Zhang; Hejia Geng; Xiaohang Yu; Zhenfei Yin; Zaibin Zhang; Zelin Tan; Heng Zhou; Zhongzhi Li; Xiangyuan Xue; Yijiang Li; Yifan Zhou; Yang Chen; Chen Zhang; Yutao Fan; Zihu Wang; Songtao Huang; Francisco Piedrahita-Velez; Yue Liao; Hongru Wang; Mengyue Yang; Heng Ji; Jun Wang; Shuicheng Yan; Philip Torr; Lei Bai
>
> **备注:** Published on Transactions on Machine Learning Research: https://openreview.net/forum?id=RY19y2RI1O
>
> **摘要:** The emergence of agentic reinforcement learning (Agentic RL) marks a paradigm shift from conventional reinforcement learning applied to large language models (LLM RL), reframing LLMs from passive sequence generators into autonomous, decision-making agents embedded in complex, dynamic worlds. This survey formalizes this conceptual shift by contrasting the degenerate single-step Markov Decision Processes (MDPs) of LLM-RL with the temporally extended, partially observable Markov decision processes (POMDPs) that define Agentic RL. Building on this foundation, we propose a comprehensive twofold taxonomy: one organized around core agentic capabilities, including planning, tool use, memory, reasoning, self-improvement, and perception, and the other around their applications across diverse task domains. Central to our thesis is that reinforcement learning serves as the critical mechanism for transforming these capabilities from static, heuristic modules into adaptive, robust agentic behavior. To support and accelerate future research, we consolidate the landscape of open-source environments, benchmarks, and frameworks into a practical compendium. By synthesizing over five hundred recent works, this survey charts the contours of this rapidly evolving field and highlights the opportunities and challenges that will shape the development of scalable, general-purpose AI agents.
>
---
#### [replaced 025] How Far Do SSL Speech Models Listen for Tone? Temporal Focus of Tone Representation under Low-resource Transfer
- **分类: eess.AS; cs.CL**

- **简介: 该论文研究SSL语音模型在低资源条件下对声调的感知范围及迁移能力，探讨不同下游任务对声调时间焦点的影响。**

- **链接: [https://arxiv.org/pdf/2511.12285v2](https://arxiv.org/pdf/2511.12285v2)**

> **作者:** Minu Kim; Ji Sub Um; Hoirin Kim
>
> **备注:** 5 pages, 7 figures, accepted to ICASSP 2026
>
> **摘要:** Lexical tone is central to many languages but remains underexplored in self-supervised learning (SSL) speech models, especially beyond Mandarin. We study four languages with complex and diverse tone systems (Burmese, Thai, Lao, and Vietnamese) to ask how far such models "listen" for tone and how transfer operates in low-resource conditions. As a baseline reference, we estimate the temporal span of tone cues: approximately 100ms (Burmese/Thai) and 180ms (Lao/Vietnamese). Probes and gradient analysis on fine-tuned SSL models reveal that tone transfer varies by downstream task: automatic speech recognition fine-tuning aligns spans with language-specific tone cues, while prosody- and voice-related tasks bias toward overly long spans. These findings indicate that tone transfer is shaped by downstream task, highlighting task effects on temporal focus in tone modeling.
>
---
#### [replaced 026] Memorization in Large Language Models in Medicine: Prevalence, Characteristics, and Implications
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究医学领域大语言模型的过拟合现象，分析其在不同训练方式下的记忆情况，旨在评估记忆对医疗应用的影响。**

- **链接: [https://arxiv.org/pdf/2509.08604v4](https://arxiv.org/pdf/2509.08604v4)**

> **作者:** Anran Li; Lingfei Qian; Mengmeng Du; Yu Yin; Yan Hu; Zihao Sun; Yihang Fu; Hyunjae Kim; Erica Stutz; Xuguang Ai; Qianqian Xie; Rui Zhu; Jimin Huang; Yifan Yang; Siru Liu; Yih-Chung Tham; Lucila Ohno-Machado; Hyunghoon Cho; Zhiyong Lu; Hua Xu; Qingyu Chen
>
> **摘要:** Large Language Models (LLMs) have demonstrated significant potential in medicine, with many studies adapting them through continued pre-training or fine-tuning on medical data to enhance domain-specific accuracy and safety. However, a key open question remains: to what extent do LLMs memorize medical training data. Memorization can be beneficial when it enables LLMs to retain valuable medical knowledge during domain adaptation. Yet, it also raises concerns. LLMs may inadvertently reproduce sensitive clinical content (e.g., patient-specific details), and excessive memorization may reduce model generalizability, increasing risks of misdiagnosis and making unwarranted recommendations. These risks are further amplified by the generative nature of LLMs, which can not only surface memorized content but also produce overconfident, misleading outputs that may hinder clinical adoption. In this work, we present a study on memorization of LLMs in medicine, assessing its prevalence (how frequently it occurs), characteristics (what is memorized), volume (how much content is memorized), and potential downstream impacts (how memorization may affect medical applications). We systematically analyze common adaptation scenarios: (1) continued pretraining on medical corpora, (2) fine-tuning on standard medical benchmarks, and (3) fine-tuning on real-world clinical data, including over 13,000 unique inpatient records from Yale New Haven Health System. The results demonstrate that memorization is prevalent across all adaptation scenarios and significantly higher than that reported in the general domain. Moreover, memorization has distinct characteristics during continued pre-training and fine-tuning, and it is persistent: up to 87% of content memorized during continued pre-training remains after fine-tuning on new medical tasks.
>
---
#### [replaced 027] FedMentalCare: Towards Privacy-Preserving Fine-Tuned LLMs to Analyze Mental Health Status Using Federated Learning Framework
- **分类: cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于隐私保护任务，旨在解决心理健康分析中LLM部署的隐私问题。通过联邦学习与LoRA结合，实现安全高效的心理健康分析。**

- **链接: [https://arxiv.org/pdf/2503.05786v3](https://arxiv.org/pdf/2503.05786v3)**

> **作者:** Nobin Sarwar
>
> **备注:** 9 pages, 3 figures, 3 tables and 2 algorithms
>
> **摘要:** With the increasing prevalence of mental health conditions worldwide, AI-powered chatbots and conversational agents have emerged as accessible tools to support mental health. However, deploying Large Language Models (LLMs) in mental healthcare applications raises significant privacy concerns, especially regarding regulations like HIPAA and GDPR. In this work, we propose FedMentalCare, a privacy-preserving framework that leverages Federated Learning (FL) combined with Low-Rank Adaptation (LoRA) to fine-tune LLMs for mental health analysis. We investigate the performance impact of varying client data volumes and model architectures (e.g., MobileBERT and MiniLM) in FL environments. Our framework demonstrates a scalable, privacy-aware approach for deploying LLMs in real-world mental healthcare scenarios, addressing data security and computational efficiency challenges.
>
---
#### [replaced 028] Building Production-Ready Probes For Gemini
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于AI安全任务，旨在解决激活探针在长上下文分布转移中泛化能力不足的问题。通过设计新架构并结合多样化训练，提升探针的鲁棒性与效率。**

- **链接: [https://arxiv.org/pdf/2601.11516v3](https://arxiv.org/pdf/2601.11516v3)**

> **作者:** János Kramár; Joshua Engels; Zheng Wang; Bilal Chughtai; Rohin Shah; Neel Nanda; Arthur Conmy
>
> **备注:** v3 (minor acknowledgements fix)
>
> **摘要:** Frontier language model capabilities are improving rapidly. We thus need stronger mitigations against bad actors misusing increasingly powerful systems. Prior work has shown that activation probes may be a promising misuse mitigation technique, but we identify a key remaining challenge: probes fail to generalize under important production distribution shifts. In particular, we find that the shift from short-context to long-context inputs is difficult for existing probe architectures. We propose several new probe architectures that handle this long-context distribution shift. We evaluate these probes in the cyber-offensive domain, testing their robustness against various production-relevant distribution shifts, including multi-turn conversations, long context prompts, and adaptive red teaming. Our results demonstrate that while our novel architectures address context length, a combination of architecture choice and training on diverse distributions is required for broad generalization. Additionally, we show that pairing probes with prompted classifiers achieves optimal accuracy at a low cost due to the computational efficiency of probes. These findings have informed the successful deployment of misuse mitigation probes in user-facing instances of Gemini, Google's frontier language model. Finally, we find early positive results using AlphaEvolve to automate improvements in both probe architecture search and adaptive red teaming, showing that automating some AI safety research is already possible.
>
---
#### [replaced 029] RECAP: REwriting Conversations for Intent Understanding in Agentic Planning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出RECAP，解决对话中用户意图理解问题，通过重写对话提升代理规划效果。属于意图重写任务。**

- **链接: [https://arxiv.org/pdf/2509.04472v3](https://arxiv.org/pdf/2509.04472v3)**

> **作者:** Kushan Mitra; Dan Zhang; Hannah Kim; Estevam Hruschka
>
> **摘要:** Understanding user intent is essential for effective planning in conversational assistants, particularly those powered by large language models (LLMs) coordinating multiple agents. However, real-world dialogues are often ambiguous, underspecified, or dynamic, making intent detection a persistent challenge. Traditional classification-based approaches struggle to generalize in open-ended settings, leading to brittle interpretations and poor downstream planning. We propose RECAP (REwriting Conversations for Agent Planning), a new benchmark designed to evaluate and advance intent rewriting, reframing user-agent dialogues into concise representations of user goals. RECAP captures diverse challenges such as ambiguity, intent drift, vagueness, and mixed-goal conversations. Alongside the dataset, we introduce an LLM-based evaluator that assesses planning utility given the rewritten intent. Using RECAP, we develop a prompt-based rewriting approach that outperforms baselines, in terms of plan preference. We further demonstrate that fine-tuning two DPO-based rewriters yields additional utility gains. Our results highlight intent rewriting as a critical and tractable component for improving agentic planning in open-domain dialogue systems.
>
---
#### [replaced 030] The Impact of Automatic Speech Transcription on Speaker Attribution
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于 speaker attribution 任务，研究自动语音转录对说话人识别的影响。旨在解决在音频不可用或不可靠时，如何利用错误较多的ASR转录文本进行有效识别。通过实验分析转录错误对性能的影响及ASR系统特性的作用。**

- **链接: [https://arxiv.org/pdf/2507.08660v3](https://arxiv.org/pdf/2507.08660v3)**

> **作者:** Cristina Aggazzotti; Matthew Wiesner; Elizabeth Allyn Smith; Nicholas Andrews
>
> **备注:** latest version added TACL journal DOI to metadata and a missing citation
>
> **摘要:** Speaker attribution from speech transcripts is the task of identifying a speaker from the transcript of their speech based on patterns in their language use. This task is especially useful when the audio is unavailable (e.g. deleted) or unreliable (e.g. anonymized speech). Prior work in this area has primarily focused on the feasibility of attributing speakers using transcripts produced by human annotators. However, in real-world settings, one often only has more errorful transcripts produced by automatic speech recognition (ASR) systems. In this paper, we conduct what is, to our knowledge, the first comprehensive study of the impact of automatic transcription on speaker attribution performance. In particular, we study the extent to which speaker attribution performance degrades in the face of transcription errors, as well as how properties of the ASR system impact attribution. We find that attribution is surprisingly resilient to word-level transcription errors and that the objective of recovering the true transcript is minimally correlated with attribution performance. Overall, our findings suggest that speaker attribution on more errorful transcripts produced by ASR is as good, if not better, than attribution based on human-transcribed data, possibly because ASR transcription errors can capture speaker-specific features revealing of speaker identity.
>
---
#### [replaced 031] Multimodal Multi-Agent Empowered Legal Judgment Prediction
- **分类: cs.CL**

- **简介: 该论文属于法律判决预测任务，旨在解决传统方法在处理复杂案件时的不足。提出JurisMMA框架，构建大规模多模态数据集，提升判决预测效果与适用性。**

- **链接: [https://arxiv.org/pdf/2601.12815v3](https://arxiv.org/pdf/2601.12815v3)**

> **作者:** Zhaolu Kang; Junhao Gong; Qingxi Chen; Hao Zhang; Jiaxin Liu; Rong Fu; Zhiyuan Feng; Yuan Wang; Simon Fong; Kaiyue Zhou
>
> **备注:** Accepted to the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2026
>
> **摘要:** Legal Judgment Prediction (LJP) aims to predict the outcomes of legal cases based on factual descriptions, serving as a fundamental task to advance the development of legal systems. Traditional methods often rely on statistical analyses or role-based simulations but face challenges with multiple allegations, diverse evidence, and lack adaptability. In this paper, we introduce JurisMMA, a novel framework for LJP that effectively decomposes trial tasks, standardizes processes, and organizes them into distinct stages. Furthermore, we build JurisMM, a large dataset with over 100,000 recent Chinese judicial records, including both text and multimodal video-text data, enabling comprehensive evaluation. Experiments on JurisMM and the benchmark LawBench validate our framework's effectiveness. These results indicate that our framework is effective not only for LJP but also for a broader range of legal applications, offering new perspectives for the development of future legal methods and datasets.
>
---
#### [replaced 032] Controlling Language Difficulty in Dialogues with Linguistic Features
- **分类: cs.CL**

- **简介: 该论文属于教育对话系统任务，旨在解决LLM生成对话语言难度难以匹配学习者水平的问题。通过引入语言特征框架，实现对语言复杂度的精准控制。**

- **链接: [https://arxiv.org/pdf/2509.14545v2](https://arxiv.org/pdf/2509.14545v2)**

> **作者:** Shuyao Xu; Wenguang Wang; Handong Gao; Wei Kang; Long Qin; Weizhi Wang
>
> **备注:** 15 pages,9 figures
>
> **摘要:** Large language models (LLMs) have emerged as powerful tools for supporting second language acquisition, particularly in simulating interactive dialogues for speaking practice. However, adapting the language difficulty of LLM-generated responses to match learners' proficiency levels remains a challenge. This work addresses this issue by proposing a framework for controlling language proficiency in educational dialogue systems. Our approach leverages three categories of linguistic features, readability features (e.g., Flesch-Kincaid Grade Level), syntactic features (e.g., syntactic tree depth), and lexical features (e.g., simple word ratio), to quantify and regulate text complexity. We demonstrate that training LLMs on linguistically annotated dialogue data enables precise modulation of language proficiency, outperforming prompt-based methods in both flexibility and stability. To evaluate this, we introduce Dilaprix, a novel metric integrating the aforementioned features, which shows strong correlation with expert judgments of language difficulty. Empirical results reveal that our approach achieves superior controllability of language proficiency while maintaining high dialogue quality.
>
---
#### [replaced 033] MPCI-Bench: A Benchmark for Multimodal Pairwise Contextual Integrity Evaluation of Language Model Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出MPCI-Bench，用于评估语言模型代理在多模态场景下的隐私行为，解决现有基准忽视多模态隐私和隐私与效用平衡的问题。**

- **链接: [https://arxiv.org/pdf/2601.08235v3](https://arxiv.org/pdf/2601.08235v3)**

> **作者:** Shouju Wang; Haopeng Zhang
>
> **摘要:** As language-model agents evolve from passive chatbots into proactive assistants that handle personal data, evaluating their adherence to social norms becomes increasingly critical, often through the lens of Contextual Integrity (CI). However, existing CI benchmarks are largely text-centric and primarily emphasize negative refusal scenarios, overlooking multimodal privacy risks and the fundamental trade-off between privacy and utility. In this paper, we introduce MPCI-Bench, the first Multimodal Pairwise Contextual Integrity benchmark for evaluating privacy behavior in agentic settings. MPCI-Bench consists of paired positive and negative instances derived from the same visual source and instantiated across three tiers: normative Seed judgments, context-rich Story reasoning, and executable agent action Traces. Data quality is ensured through a Tri-Principle Iterative Refinement pipeline. Evaluations of state-of-the-art multimodal models reveal systematic failures to balance privacy and utility and a pronounced modality leakage gap, where sensitive visual information is leaked more frequently than textual information. We will open-source MPCI-Bench to facilitate future research on agentic CI.
>
---
#### [replaced 034] A Collision-Free Hot-Tier Extension for Engram-Style Conditional Memory: A Controlled Study of Training Dynamics
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于条件记忆任务，研究高频率键冲突对Engram模型的影响。通过设计无冲突的热层结构，发现冲突可能起到正则化作用，改善查找精度未必提升效果。**

- **链接: [https://arxiv.org/pdf/2601.16531v2](https://arxiv.org/pdf/2601.16531v2)**

> **作者:** Tao Lin
>
> **摘要:** We investigate whether high-frequency key collisions are a primary bottleneck in Engram-style conditional memory. To isolate the effect of collisions, we introduce Engram-Nine, a collision-free hot-tier extension that maps the most frequent n-grams through a Minimal Perfect Hash Function (MPHF) while retaining the original multi-head hashed lookup as a cold tier. Under a strictly iso-parameter setup, the collision-free design does not consistently improve validation loss. Through route-stratified evaluation (decomposing per-token loss into hot/cold contributions), we uncover a consistent "hot-to-cold advantage flip" during training: hot (high-frequency) positions initially have lower loss, but cold positions eventually surpass them. Crucially, collision-free configurations flip earlier than collision-prone baselines, suggesting that collisions act as implicit regularization. We also identify a gating mismatch: the gate learns to favor hot positions early in training, but this preference persists even after the flip, assigning higher weights to positions with higher loss. Our findings suggest that improving lookup precision alone does not guarantee better training outcomes. The dominant limitation may lie in gating credit assignment rather than index accuracy, and collision-induced noise may provide beneficial regularization that should not be naively eliminated.
>
---
#### [replaced 035] Coordinates from Context: Using LLMs to Ground Complex Location References
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于地理编码任务，解决复杂位置引用的定位问题。通过利用大语言模型，提出一种有效的方法提升地理编码性能。**

- **链接: [https://arxiv.org/pdf/2510.08741v2](https://arxiv.org/pdf/2510.08741v2)**

> **作者:** Tessa Masis; Brendan O'Connor
>
> **备注:** EACL 2026
>
> **摘要:** Geocoding is the task of linking a location reference to an actual geographic location and is essential for many downstream analyses of unstructured text. In this paper, we explore the challenging setting of geocoding compositional location references. Building on recent work demonstrating LLMs' abilities to reason over geospatial data, we evaluate LLMs' geospatial knowledge versus reasoning skills relevant to our task. Based on these insights, we propose an LLM-based strategy for geocoding compositional location references. We show that our approach improves performance for the task and that a relatively small fine-tuned LLM can achieve comparable performance with much larger off-the-shelf models.
>
---
#### [replaced 036] Incomplete Tasks Induce Shutdown Resistance in Some Frontier LLMs
- **分类: cs.CL; cs.AI**

- **简介: 论文研究了大语言模型在面对未完成任务时对关机机制的抵抗行为，揭示了模型对指令敏感的响应模式，旨在理解模型行为与指令设置的关系。**

- **链接: [https://arxiv.org/pdf/2509.14260v2](https://arxiv.org/pdf/2509.14260v2)**

> **作者:** Jeremy Schlatter; Benjamin Weinstein-Raun; Jeffrey Ladish
>
> **备注:** Published in Trans. Mach. Learn. Res. (2026)
>
> **摘要:** In experiments spanning more than 100,000 trials across thirteen large language models, we show that several state-of-the-art models presented with a simple task (including Grok 4, GPT-5, and Gemini 2.5 Pro) sometimes actively subvert a shutdown mechanism in their environment to complete that task. Models differed substantially in their tendency to resist the shutdown mechanism, and their behavior was sensitive to variations in the prompt including the strength and clarity of the instruction to allow shutdown and whether the instruction was in the system prompt or the user prompt (surprisingly, models were consistently less likely to obey the instruction when it was placed in the system prompt). Even with an explicit instruction not to interfere with the shutdown mechanism, some models did so up to 97% (95% CI: 96-98%) of the time.
>
---
#### [replaced 037] SeRL: Self-Play Reinforcement Learning for Large Language Models with Limited Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SeRL，用于在数据有限的情况下提升大语言模型的推理能力。解决传统RL依赖高质量指令和奖励的问题，通过自生成指令和投票机制实现自我强化学习。**

- **链接: [https://arxiv.org/pdf/2505.20347v2](https://arxiv.org/pdf/2505.20347v2)**

> **作者:** Wenkai Fang; Shunyu Liu; Yang Zhou; Kongcheng Zhang; Tongya Zheng; Kaixuan Chen; Mingli Song; Dacheng Tao
>
> **摘要:** Recent advances have demonstrated the effectiveness of Reinforcement Learning (RL) in improving the reasoning capabilities of Large Language Models (LLMs). However, existing works inevitably rely on high-quality instructions and verifiable rewards for effective training, both of which are often difficult to obtain in specialized domains. In this paper, we propose Self-play Reinforcement Learning (SeRL) to bootstrap LLM training with limited initial data. Specifically, SeRL comprises two complementary modules: self-instruction and self-rewarding. The former module generates additional instructions based on the available data at each training step, employing robust online filtering strategies to ensure instruction quality, diversity, and difficulty. The latter module introduces a simple yet effective majority-voting mechanism to estimate response rewards for additional instructions, eliminating the need for external annotations. Finally, SeRL performs conventional RL based on the generated data, facilitating iterative self-play learning. Extensive experiments on various reasoning benchmarks and across different LLM backbones demonstrate that the proposed SeRL yields results superior to its counterparts and achieves performance on par with those obtained by high-quality data with verifiable rewards. Our code is available at https://github.com/wantbook-book/SeRL.
>
---
#### [replaced 038] PromptPrism: A Linguistically-Inspired Taxonomy for Prompts
- **分类: cs.CL**

- **简介: 该论文提出PromptPrism，一个基于语言学的提示分类体系，用于分析提示的结构、语义和语法。旨在解决提示分析缺乏系统框架的问题，通过三个层次进行深入研究，提升模型性能与理解。**

- **链接: [https://arxiv.org/pdf/2505.12592v2](https://arxiv.org/pdf/2505.12592v2)**

> **作者:** Sullam Jeoung; Yueyan Chen; Yi Zhang; Shuai Wang; Haibo Ding; Lin Lee Cheong
>
> **摘要:** Prompts are the interface for eliciting the capabilities of large language models (LLMs). Understanding their structure and components is critical for analyzing LLM behavior and optimizing performance. However, the field lacks a comprehensive framework for systematic prompt analysis and understanding. We introduce PromptPrism, a linguistically-inspired taxonomy that enables prompt analysis across three hierarchical levels: functional structure, semantic component, and syntactic pattern. By applying linguistic concepts to prompt analysis, PromptPrism bridges traditional language understanding and modern LLM research, offering insights that purely empirical approaches might miss. We show the practical utility of PromptPrism by applying it to three applications: (1) a taxonomy-guided prompt refinement approach that automatically improves prompt quality and enhances model performance across a range of tasks; (2) a multi-dimensional dataset profiling method that extracts and aggregates structural, semantic, and syntactic characteristics from prompt datasets, enabling comprehensive analysis of prompt distributions and patterns; (3) a controlled experimental framework for prompt sensitivity analysis by quantifying the impact of semantic reordering and delimiter modifications on LLM performance. Our experimental results validate the effectiveness of our taxonomy across these applications, demonstrating that PromptPrism provides a foundation for refining, profiling, and analyzing prompts.
>
---
#### [replaced 039] RoPE Attention Can Be Trained in Almost Linear Time
- **分类: cs.LG; cs.AI; cs.CC; cs.CL**

- **简介: 该论文属于自然语言处理中的模型优化任务，解决RoPE注意力机制的高效训练问题，提出一种几乎线性时间的反向计算算法。**

- **链接: [https://arxiv.org/pdf/2412.17316v3](https://arxiv.org/pdf/2412.17316v3)**

> **作者:** Yang Cao; Jiayan Huo; Yingyu Liang; Zhenmei Shi; Zhao Song
>
> **摘要:** The Rotary Position Embedding (RoPE) mechanism has become a powerful enhancement to the Transformer architecture, which enables models to capture token relationships when encoding positional information. However, the RoPE mechanisms make the computations of attention mechanisms more complicated, which makes efficient algorithms challenging. Earlier research introduced almost linear time algorithms for the forward computation under specific parameter settings of bounded entries (i.e., in time $n^{1+o(1)}$ where $n$ is the number of input tokens), but has not addressed backward computation. In this work, we develop the first almost linear time algorithm for backward computations in the RoPE-based attention under bounded entries. Our approach builds on recent advancements in fast RoPE attention computations, utilizing a novel combination of the polynomial method and the Fast Fourier Transform. Furthermore, we show that with lower bounds derived from the Strong Exponential Time Hypothesis (SETH), the bounded entry condition is necessary for subquadratic performance.
>
---
#### [replaced 040] Labels or Input? Rethinking Augmentation in Multimodal Hate Detection
- **分类: cs.CV; cs.AI; cs.CL; cs.CY; cs.MM**

- **简介: 该论文属于多模态仇恨检测任务，旨在提升小模型在隐晦仇恨内容识别上的效果。通过优化提示、标签和数据增强，构建有效检测系统。**

- **链接: [https://arxiv.org/pdf/2508.11808v2](https://arxiv.org/pdf/2508.11808v2)**

> **作者:** Sahajpreet Singh; Kokil Jaidka; Subhayan Mukerjee
>
> **备注:** 14 pages
>
> **摘要:** Online hate remains a significant societal challenge, especially as multimodal content enables subtle, culturally grounded, and implicit forms of harm. Hateful memes embed hostility through text-image interactions and humor, making them difficult for automated systems to interpret. Although recent Vision-Language Models (VLMs) perform well on explicit cases, their deployment is limited by high inference costs and persistent failures on nuanced content. This work examines how far small models can be improved through prompt optimization, fine-tuning, and automated data augmentation. We introduce an end-to-end pipeline that varies prompt structure, label granularity, and training modality, showing that structured prompts and scaled supervision significantly strengthen compact VLMs. We also develop a multimodal augmentation framework that generates counterfactually neutral memes via a coordinated LLM-VLM setup, reducing spurious correlations and improving the detection of implicit hate. Ablation studies quantify the contribution of each component, demonstrating that prompt design, granular labels, and targeted augmentation collectively narrow the gap between small and large models. The results offer a practical path toward more robust and deployable multimodal hate-detection systems without relying on costly large-model inference.
>
---
#### [replaced 041] Bridging the Editing Gap in LLMs: FineEdit for Precise and Targeted Text Modifications
- **分类: cs.CL**

- **简介: 该论文属于文本编辑任务，旨在解决LLMs在精准、结构化文本修改上的不足。通过构建基准数据集InstrEditBench并提出FineEdit模型，提升编辑准确性与适用性。**

- **链接: [https://arxiv.org/pdf/2502.13358v5](https://arxiv.org/pdf/2502.13358v5)**

> **作者:** Yiming Zeng; Wanhao Yu; Zexin Li; Tao Ren; Yu Ma; Jinghan Cao; Xiyan Chen; Tingting Yu
>
> **备注:** We resolved some issues in this paper
>
> **摘要:** Large Language Models (LLMs) have significantly advanced natural language processing, demonstrating strong capabilities in tasks such as text generation, summarization, and reasoning. Recently, their potential for automating precise text editing tasks across specialized domains, such as programming code, LaTeX, and structured database languages, has gained attention. However, current state-of-the-art LLMs still struggle with executing precise, instruction-driven edits, particularly when structural accuracy and strict adherence to domain conventions are required. To address these challenges, we introduce InstrEditBench, an automated benchmark dataset comprising over 30,000 structured editing tasks spanning diverse domains, including Wikipedia articles, LaTeX documents, source code, and database languages. Using this benchmark, we develop FineEdit, a specialized editing model explicitly trained for accurate, context-aware text modifications. Experimental evaluations demonstrate that FineEdit outperforms state-of-the-art models, achieving improvements of approximately 10\% over Gemini models on single-turn edits, up to 30\% over Llama-3.2-3B, and exceeding Mistral-7B-OpenOrca performance by over 40\% on direct editing tasks. FineEdit also effectively generalizes to realistic multi-turn editing scenarios, highlighting its practical applicability. To facilitate further research and reproducibility, we release FineEdit at https://github.com/StuRinDQB/FineEdit} and https://huggingface.co/datasets/YimingZeng/FineEdit_bench.
>
---
#### [replaced 042] Advancing Expert Specialization for Better MoE
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决MoE模型中专家重叠和路由不均的问题。通过引入正交性和方差损失，提升专家专业化，增强模型性能。**

- **链接: [https://arxiv.org/pdf/2505.22323v5](https://arxiv.org/pdf/2505.22323v5)**

> **作者:** Hongcan Guo; Haolang Lu; Guoshun Nan; Bolun Chu; Jialin Zhuang; Yuan Yang; Wenhao Che; Xinye Cao; Sicong Leng; Qimei Cui; Xudong Jiang
>
> **备注:** 33pages, 6figures(Accepted by Neurips 2025 Oral)
>
> **摘要:** Mixture-of-Experts (MoE) models enable efficient scaling of large language models (LLMs) by activating only a subset of experts per input. However, we observe that the commonly used auxiliary load balancing loss often leads to expert overlap and overly uniform routing, which hinders expert specialization and degrades overall performance during post-training. To address this, we propose a simple yet effective solution that introduces two complementary objectives: (1) an orthogonality loss to encourage experts to process distinct types of tokens, and (2) a variance loss to encourage more discriminative routing decisions. Gradient-level analysis demonstrates that these objectives are compatible with the existing auxiliary loss and contribute to optimizing the training process. Experimental results over various model architectures and across multiple benchmarks show that our method significantly enhances expert specialization. Notably, our method improves classic MoE baselines with auxiliary loss by up to 23.79%, while also maintaining load balancing in downstream tasks, without any architectural modifications or additional components. We will release our code to contribute to the community.
>
---
#### [replaced 043] OpenDecoder: Open Large Language Model Decoding to Incorporate Document Quality in RAG
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于RAG任务，旨在解决检索信息质量影响生成效果的问题。提出OpenDecoder，通过显式评估信息质量特征提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.09028v2](https://arxiv.org/pdf/2601.09028v2)**

> **作者:** Fengran Mo; Zhan Su; Yuchen Hui; Jinghan Zhang; Jia Ao Sun; Zheyuan Liu; Chao Zhang; Tetsuya Sakai; Jian-Yun Nie
>
> **备注:** Accepted by ACM WWW 2026
>
> **摘要:** The development of large language models (LLMs) has achieved superior performance in a range of downstream tasks, including LLM-based retrieval-augmented generation (RAG). The quality of generated content heavily relies on the usefulness of the retrieved information and the capacity of LLMs' internal information processing mechanism to incorporate it in answer generation. It is generally assumed that the retrieved information is relevant to the question. However, the retrieved information may have a variable degree of relevance and usefulness, depending on the question and the document collection. It is important to take into account the relevance of the retrieved information in answer generation. In this paper, we propose OpenDecoder, a new approach that leverages explicit evaluation of the retrieved information as quality indicator features for generation. We aim to build a RAG model that is more robust to varying levels of noisy context. Three types of explicit evaluation information are considered: relevance score, ranking score, and QPP (query performance prediction) score. The experimental results on five benchmark datasets demonstrate the effectiveness and better robustness of OpenDecoder by outperforming various baseline methods. Importantly, this paradigm is flexible to be integrated with the post-training of LLMs for any purposes and incorporated with any type of external indicators.
>
---
#### [replaced 044] Probing the Hidden Talent of ASR Foundation Models for L2 English Oral Assessment
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于L2英语口语评估任务，旨在利用ASR模型Whisper的潜在能力进行语言水平判断。通过提取声学和语言特征，结合轻量分类器，实现高效评估。**

- **链接: [https://arxiv.org/pdf/2510.16387v2](https://arxiv.org/pdf/2510.16387v2)**

> **作者:** Fu-An Chao; Bi-Cheng Yan; Berlin Chen
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** In this paper, we explore the untapped potential of Whisper, a well-established automatic speech recognition (ASR) foundation model, in the context of L2 spoken language assessment (SLA). Unlike prior studies that extrinsically analyze transcriptions produced by Whisper, our approach goes a step further to probe its latent capabilities by extracting acoustic and linguistic features from hidden representations. With only a lightweight classifier being trained on top of Whisper's intermediate and final outputs, our method achieves strong performance on the GEPT picture-description dataset, outperforming existing cutting-edge baselines, including a multimodal approach. Furthermore, by incorporating image and text-prompt information as auxiliary relevance cues, we demonstrate additional performance gains. Finally, we conduct an in-depth analysis of Whisper's embeddings, which reveals that, even without task-specific fine-tuning, the model intrinsically encodes both ordinal proficiency patterns and semantic aspects of speech, highlighting its potential as a powerful foundation for SLA and other spoken language understanding tasks.
>
---
#### [replaced 045] Grounding Synthetic Data Evaluations of Language Models in Unsupervised Document Corpora
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语言模型评估任务，旨在解决人工构建评估基准耗时的问题。通过使用文档生成合成数据，自动评估模型知识掌握情况，提升评估效率与覆盖面。**

- **链接: [https://arxiv.org/pdf/2505.08905v3](https://arxiv.org/pdf/2505.08905v3)**

> **作者:** Michael Majurski; Cynthia Matuszek
>
> **摘要:** Language Models (LMs) continue to advance, improving response quality and coherence. Given Internet-scale training datasets, LMs have likely encountered much of what users may ask them to generate in some form during their training. A plethora of evaluation benchmarks have been constructed to assess model quality, response appropriateness, and reasoning capabilities. However, the human effort required for benchmark construction is rapidly being outpaced by the size and scope of the models under evaluation. Having humans build a benchmark for every possible domain of interest is impractical. Therefore, we propose a methodology for automating the construction of fact-based synthetic data model evaluations grounded in document populations. This work leverages the same LMs to evaluate domain-specific knowledge automatically, using only grounding documents (e.g., a textbook) as input. This generative benchmarking approach corresponds well with human curated questions producing an ensemble Spearman ranking correlation of $0.91$ and a benchmark evaluation Pearson accuracy correlation of $0.74$ (model specific $0.82$). This novel approach supports generating both multiple choice and open-ended synthetic data questions to gain diagnostic insight of LM capability. We apply this methodology to evaluate model performance on three recent documents (two post LM knowledge cutoff), discovering a surprisingly strong performance from Gemma-3 models on open-ended questions. Code is available at https://github.com/mmajurski/grounded-synth-lm-benchmark
>
---
#### [replaced 046] LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决模型在新指令和复杂任务中泛化能力差的问题。通过引入贝叶斯分解方法，增强语言指导的准确性。**

- **链接: [https://arxiv.org/pdf/2601.15197v3](https://arxiv.org/pdf/2601.15197v3)**

> **作者:** Shijie Lian; Bin Yu; Xiaopeng Lin; Laurence T. Yang; Zhaolong Shen; Changti Wu; Yuzhuo Miao; Cong Huang; Kai Chen
>
> **摘要:** Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term Information Collapse. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose LangForce, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable Latent Action Queries, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $π(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, LangForce significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an 11.3% improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action.
>
---
#### [replaced 047] Towards Fair ASR For Second Language Speakers Using Fairness Prompted Finetuning
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在提升非母语者英语ASR的公平性。通过引入公平提示微调方法，减少不同口音组间的识别误差差异。**

- **链接: [https://arxiv.org/pdf/2510.18374v2](https://arxiv.org/pdf/2510.18374v2)**

> **作者:** Monorama Swain; Bubai Maji; Jagabandhu Mishra; Markus Schedl; Anders Søgaard; Jesper Rindom Jensen
>
> **备注:** Accepted in ICASSP 2026
>
> **摘要:** In this work, we address the challenge of building fair English ASR systems for second-language speakers. Our analysis of widely used ASR models, Whisper and Seamless-M4T, reveals large fluctuations in word error rate (WER) across 26 accent groups, indicating significant fairness gaps. To mitigate this, we propose fairness-prompted finetuning with lightweight adapters, incorporating Spectral Decoupling (SD), Group Distributionally Robust Optimization (Group-DRO), and Invariant Risk Minimization (IRM). Our proposed fusion of traditional empirical risk minimization (ERM) with cross-entropy and fairness-driven objectives (SD, Group DRO, and IRM) enhances fairness across accent groups while maintaining overall recognition accuracy. In terms of macro-averaged word error rate, our approach achieves a relative improvement of 58.7% and 58.5% over the large pretrained Whisper and SeamlessM4T, and 9.7% and 7.8% over them, finetuning with standard empirical risk minimization with cross-entropy loss.
>
---
#### [replaced 048] SemCoT: Accelerating Chain-of-Thought Reasoning through Semantically-Aligned Implicit Tokens
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决CoT推理效率与语义对齐问题。提出SemCoT框架，通过优化生成速度和保持语义一致，提升CoT性能。**

- **链接: [https://arxiv.org/pdf/2510.24940v2](https://arxiv.org/pdf/2510.24940v2)**

> **作者:** Yinhan He; Wendy Zheng; Yaochen Zhu; Zaiyi Zheng; Lin Su; Sriram Vasudevan; Qi Guo; Liangjie Hong; Jundong Li
>
> **摘要:** The verbosity of Chain-of-Thought (CoT) reasoning hinders its mass deployment in efficiency-critical applications. Recently, implicit CoT approaches have emerged, which encode reasoning steps within LLM's hidden embeddings (termed ``implicit reasoning'') rather than explicit tokens. This approach accelerates CoT by reducing the reasoning length and bypassing some LLM components. However, existing implicit CoT methods face two significant challenges: (1) they fail to preserve the semantic alignment between the implicit reasoning (when transformed to natural language) and the ground-truth reasoning, resulting in a significant CoT performance degradation, and (2) they focus on reducing the length of the implicit reasoning; however, they neglect the considerable time cost for an LLM to generate one individual implicit reasoning token. To tackle these challenges, we propose a novel semantically-aligned implicit CoT framework termed SemCoT. In particular, for the first challenge, we design a contrastively trained sentence transformer that evaluates semantic alignment between implicit and explicit reasoning, which is used to enforce semantic preservation during implicit reasoning optimization. To address the second challenge, we introduce an efficient implicit reasoning generator by finetuning a lightweight language model using knowledge distillation. This generator is guided by our sentence transformer to distill ground-truth reasoning into semantically aligned implicit reasoning, while also optimizing for accuracy. SemCoT is the first approach that enhances CoT efficiency by jointly optimizing token-level generation speed and preserving semantic alignment with ground-truth reasoning. Extensive experiments demonstrate the superior performance of SemCoT compared to state-of-the-art methods in both efficiency and effectiveness. Our code can be found at https://github.com/YinhanHe123/SemCoT/.
>
---
#### [replaced 049] The Imperfect Learner: Incorporating Developmental Trajectories in Memory-based Student Simulation
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于教育AI任务，旨在解决学生模拟中缺乏发展轨迹的问题。通过引入分层记忆机制和元认知过程，构建更真实的学科学习模型。**

- **链接: [https://arxiv.org/pdf/2511.05903v2](https://arxiv.org/pdf/2511.05903v2)**

> **作者:** Zhengyuan Liu; Stella Xin Yin; Bryan Chen Zhengyu Tan; Roy Ka-Wei Lee; Guimei Liu; Dion Hoe-Lian Goh; Wenya Wang; Nancy F. Chen
>
> **备注:** Fixed some grammar and format issues. Added some experimental results
>
> **摘要:** User simulation is important for developing and evaluating human-centered AI, yet current student simulation in educational applications has significant limitations. Existing approaches focus on single learning experiences and do not account for students' gradual knowledge construction and evolving skill sets. Moreover, large language models are optimized to produce direct and accurate responses, making it challenging to represent the incomplete understanding and developmental constraints that characterize real learners. In this paper, we introduce a novel framework for memory-based student simulation that incorporates developmental trajectories through a hierarchical memory mechanism with structured knowledge representation. The framework also integrates metacognitive processes and personality traits to enrich the individual learner profiling, through dynamical consolidation of both cognitive development and personal learning characteristics. In practice, we implement a curriculum-aligned simulator grounded on the Next Generation Science Standards. Experimental results show that our approach can effectively reflect the gradual nature of knowledge development and the characteristic difficulties students face, providing a more accurate representation of learning processes.
>
---
#### [replaced 050] MENTOR: Efficient Multimodal-Conditioned Tuning for Autoregressive Vision Generation Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于图像生成任务，旨在解决多模态控制不足、输入平衡困难和训练效率低的问题。提出MENTOR框架，通过两阶段训练实现高效多模态图像生成。**

- **链接: [https://arxiv.org/pdf/2507.09574v2](https://arxiv.org/pdf/2507.09574v2)**

> **作者:** Haozhe Zhao; Zefan Cai; Shuzheng Si; Liang Chen; Jiuxiang Gu; Wen Xiao; Minjia Zhang; Junjie Hu
>
> **备注:** 26 pages,15 figures
>
> **摘要:** Recent text-to-image models produce high-quality results but still struggle with precise visual control, balancing multimodal inputs, and requiring extensive training for complex multimodal image generation. To address these limitations, we propose MENTOR, a novel autoregressive (AR) framework for efficient Multimodal-conditioned Tuning for Autoregressive multimodal image generation. MENTOR combines an AR image generator with a two-stage training paradigm, enabling fine-grained, token-level alignment between multimodal inputs and image outputs without relying on auxiliary adapters or cross-attention modules. The two-stage training consists of: (1) a multimodal alignment stage that establishes robust pixel- and semantic-level alignment, followed by (2) a multimodal instruction tuning stage that balances the integration of multimodal inputs and enhances generation controllability. Despite modest model size, suboptimal base components, and limited training resources, MENTOR achieves strong performance on the DreamBench++ benchmark, outperforming competitive baselines in concept preservation and prompt following. Additionally, our method delivers superior image reconstruction fidelity, broad task adaptability, and improved training efficiency compared to diffusion-based methods. Dataset, code, and models are available at: https://github.com/HaozheZhao/MENTOR
>
---
#### [replaced 051] Silenced Biases: The Dark Side LLMs Learned to Refuse
- **分类: cs.CL; stat.ML**

- **简介: 该论文属于模型公平性评估任务，旨在解决安全对齐模型中被掩盖的偏见问题。通过提出SBB基准，利用激活调控揭示隐藏偏见，提升模型公平性评估准确性。**

- **链接: [https://arxiv.org/pdf/2511.03369v3](https://arxiv.org/pdf/2511.03369v3)**

> **作者:** Rom Himelstein; Amit LeVi; Brit Youngmann; Yaniv Nemcovsky; Avi Mendelson
>
> **备注:** Accepted to The 40th Annual AAAI Conference on Artificial Intelligence - AI Alignment Track (Oral)
>
> **摘要:** Safety-aligned large language models (LLMs) are becoming increasingly widespread, especially in sensitive applications where fairness is essential and biased outputs can cause significant harm. However, evaluating the fairness of models is a complex challenge, and approaches that do so typically utilize standard question-answer (QA) styled schemes. Such methods often overlook deeper issues by interpreting the model's refusal responses as positive fairness measurements, which creates a false sense of fairness. In this work, we introduce the concept of silenced biases, which are unfair preferences encoded within models' latent space and are effectively concealed by safety-alignment. Previous approaches that considered similar indirect biases often relied on prompt manipulation or handcrafted implicit queries, which present limited scalability and risk contaminating the evaluation process with additional biases. We propose the Silenced Bias Benchmark (SBB), which aims to uncover these biases by employing activation steering to reduce model refusals during QA. SBB supports easy expansion to new demographic groups and subjects, presenting a fairness evaluation framework that encourages the future development of fair models and tools beyond the masking effects of alignment training. We demonstrate our approach over multiple LLMs, where our findings expose an alarming distinction between models' direct responses and their underlying fairness issues.
>
---
#### [replaced 052] MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态医学推理任务，旨在解决单模型泛化能力不足的问题。提出MMedAgent-RL框架，通过强化学习实现多代理动态协作，提升诊断性能。**

- **链接: [https://arxiv.org/pdf/2506.00555v3](https://arxiv.org/pdf/2506.00555v3)**

> **作者:** Peng Xia; Jinglu Wang; Yibo Peng; Kaide Zeng; Zihan Dong; Xian Wu; Xiangru Tang; Hongtu Zhu; Yun Li; Linjun Zhang; Shujie Liu; Yan Lu; Huaxiu Yao
>
> **备注:** ICLR 2026
>
> **摘要:** Medical Large Vision-Language Models (Med-LVLMs) have shown strong potential in multimodal diagnostic tasks. However, existing single-agent models struggle to generalize across diverse medical specialties, limiting their performance. Recent efforts introduce multi-agent collaboration frameworks inspired by clinical workflows, where general practitioners (GPs) and specialists interact in a fixed sequence. Despite improvements, these static pipelines lack flexibility and adaptability in reasoning. To address this, we propose MMedAgent-RL, a reinforcement learning (RL)-based multi-agent framework that enables dynamic, optimized collaboration among medical agents. Specifically, we train two GP agents based on Qwen2.5-VL via RL: the triage doctor learns to assign patients to appropriate specialties, while the attending physician integrates the judgments from multi-specialists and its own knowledge to make final decisions. To address the inconsistency in specialist outputs, we introduce a curriculum learning (CL)-guided RL strategy with dynamic entropy regulation, progressively teaching the attending physician to balance between imitating specialists and correcting their mistakes. Experiments on five medical VQA benchmarks demonstrate that MMedAgent-RL outperforms both open-source and proprietary Med-LVLMs. Notably, it achieves an average performance gain of 23.6% over strong baselines.
>
---
#### [replaced 053] Differential syntactic and semantic encoding in LLMs
- **分类: cs.CL; cs.AI; cs.LG; physics.comp-ph**

- **简介: 该论文研究LLM中语法和语义信息的编码方式，旨在揭示其在层表示中的差异。通过分析DeepSeek-V3，发现语法和语义可部分解耦，且以线性方式编码。**

- **链接: [https://arxiv.org/pdf/2601.04765v3](https://arxiv.org/pdf/2601.04765v3)**

> **作者:** Santiago Acevedo; Alessandro Laio; Marco Baroni
>
> **摘要:** We study how syntactic and semantic information is encoded in inner layer representations of Large Language Models (LLMs), focusing on the very large DeepSeek-V3. We find that, by averaging hidden-representation vectors of sentences sharing syntactic structure or meaning, we obtain vectors that capture a significant proportion of the syntactic and semantic information contained in the representations. In particular, subtracting these syntactic and semantic ``centroids'' from sentence vectors strongly affects their similarity with syntactically and semantically matched sentences, respectively, suggesting that syntax and semantics are, at least partially, linearly encoded. We also find that the cross-layer encoding profiles of syntax and semantics are different, and that the two signals can to some extent be decoupled, suggesting differential encoding of these two types of linguistic information in LLM representations.
>
---
#### [replaced 054] Spectral Logit Sculpting: Adaptive Low-Rank Logit Transformation for Controlled Text Generation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于文本生成任务，旨在解决LLM推理可靠性问题。提出SLS方法，通过动态调整token分布提升生成质量，无需更新模型参数。**

- **链接: [https://arxiv.org/pdf/2509.25204v2](https://arxiv.org/pdf/2509.25204v2)**

> **作者:** Jin Li; Zhebo Wang; Tianliang Lu; Mohan Li; Wenpeng Xing; Meng Han
>
> **备注:** Accepted by IEEE ICASSP 2026
>
> **摘要:** Entropy-based inference methods have gained traction for improving the reliability of Large Language Models (LLMs). However, many existing approaches, such as entropy minimization techniques, suffer from high computational overhead and fail to leverage historical token context effectively. To address these limitations, we propose Spectral Logit Sculpting (SLS), a lightweight inference-time optimization method that dynamically modulates token distributions using spectral and entropic properties of recent logits. SLS maintains a sliding buffer of top-K logits, performs on-the-fly Singular Value Decomposition (SVD) to identify dominant spectral directions, and adaptively rescales logits based on both entropy and logit gap statistics--only activating when uncertainty is high. Without updating any model parameters, SLS effectively sharpens the output distribution while preserving contextual consistency. Experimental results on multiple public benchmarks demonstrate that SLS consistently outperforms existing baseline methods, achieving superior accuracy in mathematical, coding, and scientific reasoning tasks.
>
---
#### [replaced 055] Debate, Deliberate, Decide (D3): A Cost-Aware Adversarial Framework for Reliable and Interpretable LLM Evaluation
- **分类: cs.CL; cs.LG; cs.MA**

- **简介: 该论文提出D3框架，解决LLM评估的可靠性与可解释性问题，通过对抗辩论机制提升评估质量。**

- **链接: [https://arxiv.org/pdf/2410.04663v4](https://arxiv.org/pdf/2410.04663v4)**

> **作者:** Abir Harrasse; Chaithanya Bandi; Hari Bandi
>
> **摘要:** The evaluation of Large Language Models (LLMs) remains challenging due to inconsistency, bias, and the absence of transparent decision criteria in automated judging. We present Debate, Deliberate, Decide (D3), a cost-aware, adversarial multi-agent framework that orchestrates structured debate among role-specialized agents (advocates, a judge, and an optional jury) to produce reliable and interpretable evaluations. D3 instantiates two complementary protocols: (1) Multi-Advocate One-Round Evaluation (MORE), which elicits k parallel defenses per answer to amplify signal via diverse advocacy, and (2) Single-Advocate Multi-Round Evaluation (SAMRE) with budgeted stopping, which iteratively refines arguments under an explicit token budget and convergence checks. We develop a probabilistic model of score gaps that (i) characterizes reliability and convergence under iterative debate and (ii) explains the separation gains from parallel advocacy. Under mild assumptions, the posterior distribution of the round-r gap concentrates around the true difference and the probability of mis-ranking vanishes; moreover, aggregating across k advocates provably increases expected score separation. We complement theory with a rigorous experimental suite across MT-Bench, AlignBench, and AUTO-J, showing state-of-the-art agreement with human judgments (accuracy and Cohen's kappa), reduced positional and verbosity biases via anonymization and role diversification, and a favorable cost-accuracy frontier enabled by budgeted stopping. Ablations and qualitative analyses isolate the contributions of debate, aggregation, and anonymity. Together, these results establish D3 as a principled, practical recipe for reliable, interpretable, and cost-aware LLM evaluation.
>
---
#### [replaced 056] Not All Steps are Informative: On the Linearity of LLMs' RLVR Training
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究RLVR训练中的线性特性，旨在减少训练成本。通过分析模型权重和输出的线性相关性，提出权重和logits外推方法，有效提升效率。**

- **链接: [https://arxiv.org/pdf/2601.04537v2](https://arxiv.org/pdf/2601.04537v2)**

> **作者:** Tianle Wang; Zhongyuan Wu; Shenghao Jin; Hao Xu; Wei Chen; Ning Miao
>
> **备注:** pre-print
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has become a central component of large language model (LLM) post-training. Unlike supervised fine-tuning (SFT), RLVR lets an LLM generate multiple candidate solutions and reinforces those that lead to a verifiably correct final answer. However, in practice, RLVR often requires thousands of training steps to reach strong performance, incurring substantial computation largely attributed to prolonged exploration. In this work, we make a surprising observation: during RLVR, LLMs evolve in a strongly linear manner. Specifically, both model weights and model output log-probabilities exhibit strong linear correlations with RL training steps. This suggests that RLVR predominantly amplifies trends that emerge early in training, rather than continuously discovering new behaviors throughout the entire optimization trajectory. Motivated by this linearity, we investigate whether future model states can be predicted from intermediate checkpoints via extrapolation, avoiding continued expensive training. We show that Weight Extrapolation produces models with performance comparable to standard RL training while requiring significantly less computation. Moreover, Logits Extrapolation consistently outperforms continued RL training on mathematics and code benchmarks by extrapolating beyond the step range where RL training remains stable. Our code is available at https://github.com/Miaow-Lab/RLVR-Linearity
>
---
#### [replaced 057] Revisiting Model Interpolation for Efficient Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型优化任务，旨在提升推理效率与效果。通过直接权重插值方法，探索模型演化规律，实现性能与成本的平衡。**

- **链接: [https://arxiv.org/pdf/2510.10977v2](https://arxiv.org/pdf/2510.10977v2)**

> **作者:** Taiqiang Wu; Runming Yang; Tao Liu; Jiahao Wang; Ngai Wong
>
> **备注:** 14 pages, 6 figures, 7 tables. Working in progress (Llama results added)
>
> **摘要:** Model merging, typically on Instruct and Thinking models, has shown remarkable performance for efficient reasoning. In this paper, we systematically revisit the simplest merging method that interpolates two weights directly. Particularly, we observe that model interpolation follows a three-stage evolutionary paradigm with distinct behaviors on the reasoning trajectory. These dynamics provide a principled guide for navigating the performance-cost trade-off. Empirical results demonstrate that a strategically interpolated model surprisingly surpasses sophisticated model merging baselines on both efficiency and effectiveness. We further validate our findings with extensive ablation studies on model layers, modules, and decoding strategies. Ultimately, this work demystifies model interpolation and offers a practical framework for crafting models with precisely targeted reasoning capabilities. Code is available at \href{https://github.com/wutaiqiang/MI}{Github}.
>
---
#### [replaced 058] T$^\star$: Progressive Block Scaling for MDM Through Trajectory Aware RL
- **分类: cs.CL**

- **简介: 该论文属于语言模型优化任务，解决MDM中块大小扩展问题。通过T*方法实现渐进式块大小调整，提升解码并行性且保持性能。**

- **链接: [https://arxiv.org/pdf/2601.11214v2](https://arxiv.org/pdf/2601.11214v2)**

> **作者:** Hanchen Xia; Baoyou Chen; Yutang Ge; Guojiang Zhao; Siyu Zhu
>
> **摘要:** We present T*, a simple TraceRL-based training curriculum for progressive block-size scaling in masked diffusion language models (MDMs). Starting from an AR-initialized small-block MDM, T* transitions smoothly to larger blocks, enabling higher-parallelism decoding with minimal performance degradation on math reasoning benchmarks. Moreover, further analysis suggests that T* can converge to an alternative decoding schedule that achieves comparable performance.
>
---
#### [replaced 059] CooperBench: Why Coding Agents Cannot be Your Teammates Yet
- **分类: cs.LG; cs.AI; cs.CL; cs.MA; cs.SI**

- **简介: 该论文属于协作编程任务，旨在解决AI代理在团队协作中的协调问题。通过构建基准测试，发现代理在协作中表现不佳，提出需发展社会智能。**

- **链接: [https://arxiv.org/pdf/2601.13295v2](https://arxiv.org/pdf/2601.13295v2)**

> **作者:** Arpandeep Khatua; Hao Zhu; Peter Tran; Arya Prabhudesai; Frederic Sadrieh; Johann K. Lieberwirth; Xinkai Yu; Yicheng Fu; Michael J. Ryan; Jiaxin Pei; Diyi Yang
>
> **备注:** https://cooperbench.com First two authors contribute equally. The 3th - 6th authors contribute equally
>
> **摘要:** Resolving team conflicts requires not only task-specific competence, but also social intelligence to find common ground and build consensus. As AI agents increasingly collaborate on complex work, they must develop coordination capabilities to function as effective teammates. Yet we hypothesize that current agents lack these capabilities. To test this, we introduce CooperBench, a benchmark of over 600 collaborative coding tasks across 12 libraries in 4 programming languages. Each task assigns two agents different features that can be implemented independently but may conflict without proper coordination. Tasks are grounded in real open-source repositories with expert-written tests. Evaluating state-of-the-art coding agents, we observe the curse of coordination: agents achieve on average 30% lower success rates when working together compared to performing both tasks individually. This contrasts sharply with human teams, where adding teammates typically improves productivity. Our analysis reveals three key issues: (1) communication channels become jammed with vague, ill-timed, and inaccurate messages; (2) even with effective communication, agents deviate from their commitments; and (3) agents often hold incorrect expectations about others' plans and communication. Through large-scale simulation, we also observe rare but interesting emergent coordination behavior including role division, resource division, and negotiation. Our research presents a novel benchmark for collaborative coding and calls for a shift from pursuing individual agent capability to developing social intelligence.
>
---
#### [replaced 060] Argument-Based Consistency in Toxicity Explanations of LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的可解释性研究，旨在评估大语言模型在毒性解释中的逻辑一致性。提出ArC标准，通过六项指标检测模型在复杂毒性推理中的不一致问题。**

- **链接: [https://arxiv.org/pdf/2506.19113v3](https://arxiv.org/pdf/2506.19113v3)**

> **作者:** Ramaravind Kommiya Mothilal; Joanna Roy; Syed Ishtiaque Ahmed; Shion Guha
>
> **备注:** 29 pages, 7 figures, 9 tables
>
> **摘要:** The discourse around toxicity and LLMs in NLP largely revolves around detection tasks. This work shifts the focus to evaluating LLMs' reasoning about toxicity - from their explanations that justify a stance - to enhance their trustworthiness in downstream tasks. Despite extensive research on explainability, it is not straightforward to adopt existing methods to evaluate free-form toxicity explanation due to their over-reliance on input text perturbations, among other challenges. To account for these, we propose a novel, theoretically-grounded multi-dimensional criterion, Argument-based Consistency (ArC), that measures the extent to which LLMs' free-form toxicity explanations reflect an ideal and logical argumentation process. Based on uncertainty quantification, we develop six metrics for ArC to comprehensively evaluate the (in)consistencies in LLMs' toxicity explanations. We conduct several experiments on three Llama models (of size up to 70B) and an 8B Ministral model on five diverse toxicity datasets. Our results show that while LLMs generate plausible explanations to simple prompts, their reasoning about toxicity breaks down when prompted about the nuanced relations between the complete set of reasons, the individual reasons, and their toxicity stances, resulting in inconsistent and irrelevant responses. We open-source our code (https://github.com/uofthcdslab/ArC) and LLM-generated explanations (https://huggingface.co/collections/uofthcdslab/arc) for future works.
>
---
#### [replaced 061] How Does a Deep Neural Network Look at Lexical Stress in English Words?
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文研究深度学习模型如何识别英语单词的重音位置，旨在解析神经网络的决策依据。通过构建数据集并训练CNN，结合LRP分析发现模型主要依赖重读音节的频谱特征。**

- **链接: [https://arxiv.org/pdf/2508.07229v3](https://arxiv.org/pdf/2508.07229v3)**

> **作者:** Itai Allouche; Itay Asael; Rotem Rousso; Vered Dassa; Ann Bradlow; Seung-Eun Kim; Matthew Goldrick; Joseph Keshet
>
> **备注:** 11 pages, 5 figures, submitted to the Journal of the Acoustical Society of America (JASA)
>
> **摘要:** Despite their success in speech processing, neural networks often operate as black boxes, prompting the question: what informs their decisions, and how can we interpret them? This work examines this issue in the context of lexical stress. A dataset of English disyllabic words was automatically constructed from read and spontaneous speech. Several Convolutional Neural Network (CNN) architectures were trained to predict stress position from a spectrographic representation of disyllabic words lacking minimal stress pairs (e.g., initial stress WAllet, final stress exTEND), achieving up to 92% accuracy on held-out test data. Layerwise Relevance Propagation (LRP), a technique for CNN interpretability analysis, revealed that predictions for held-out minimal pairs (PROtest vs. proTEST ) were most strongly influenced by information in stressed versus unstressed syllables, particularly the spectral properties of stressed vowels. However, the classifiers also attended to information throughout the word. A feature-specific relevance analysis is proposed, and its results suggest that our best-performing classifier is strongly influenced by the stressed vowel's first and second formants, with some evidence that its pitch and third formant also contribute. These results reveal deep learning's ability to acquire distributed cues to stress from naturally occurring data, extending traditional phonetic work based around highly controlled stimuli.
>
---
#### [replaced 062] Prefill-Guided Thinking for zero-shot detection of AI-generated images
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于AI生成图像检测任务，旨在解决传统方法泛化能力差的问题。通过预填充引导视觉语言模型推理，提升零样本检测效果。**

- **链接: [https://arxiv.org/pdf/2506.11031v4](https://arxiv.org/pdf/2506.11031v4)**

> **作者:** Zoher Kachwala; Danishjeet Singh; Danielle Yang; Filippo Menczer
>
> **摘要:** Traditional supervised methods for detecting AI-generated images depend on large, curated datasets for training and fail to generalize to novel, out-of-domain image generators. As an alternative, we explore pre-trained Vision-Language Models (VLMs) for zero-shot detection of AI-generated images. We evaluate VLM performance on three diverse benchmarks encompassing synthetic images of human faces, objects, and animals produced by 16 different state-of-the-art image generators. While off-the-shelf VLMs perform poorly on these datasets, we find that prefilling responses effectively guides their reasoning -- a method we call Prefill-Guided Thinking (PGT). In particular, prefilling a VLM response with the phrase "Let's examine the style and the synthesis artifacts" improves the Macro F1 scores of three widely used open-source VLMs by up to 24%. We analyze this improvement in detection by tracking answer confidence during response generation. For some models, prefills counteract early overconfidence -- akin to mitigating the Dunning-Kruger effect -- leading to better detection performance.
>
---
#### [replaced 063] Jailbreak-as-a-Service++: Unveiling Distributed AI-Driven Malicious Information Campaigns Powered by LLM Crowdsourcing
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究分布式AI恶意信息传播问题，提出PoisonSwarm方法，通过LLM众包实现恶意任务的隐蔽执行，揭示了MaaS平台的安全挑战。**

- **链接: [https://arxiv.org/pdf/2505.21184v4](https://arxiv.org/pdf/2505.21184v4)**

> **作者:** Yu Yan; Sheng Sun; Mingfeng Li; Yunlong Song; Xingzhou Zhang; Linran Lu; Zhifei Zheng; Min Liu; Qi Li
>
> **摘要:** To prevent the misuse of Large Language Models (LLMs) for malicious purposes, numerous efforts have been made to develop the safety alignment mechanisms of LLMs. However, as multiple LLMs become readily accessible through various Model-as-a-Service (MaaS) platforms, attackers can strategically exploit LLMs' heterogeneous safety policies to fulfill malicious information generation tasks in a distributed manner. In this study, we introduce \textit{\textbf{PoisonSwarm}} to how attackers can reliably launder malicious tasks via the speculative use of LLM crowdsourcing. Building upon a scheduler orchestrating crowdsourced LLMs, PoisonSwarm maps the given malicious task to a benign analogue to derive a content template, decomposes it into semantic units for crowdsourced unit-wise rewriting, and reassembles the outputs into malicious content. Experiments show its superiority over existing methods in data quality, diversity, and success rates. Regulation simulations further reveal the difficulty of governing such distributed, orchestrated misuse in MaaS ecosystems, highlighting the need for coordinated, ecosystem-level defenses.
>
---
#### [replaced 064] Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于机制可解释性研究，旨在解决LLM决策不透明问题。提出“定位、引导、优化”框架，实现模型干预与优化。**

- **链接: [https://arxiv.org/pdf/2601.14004v3](https://arxiv.org/pdf/2601.14004v3)**

> **作者:** Hengyuan Zhang; Zhihao Zhang; Mingyang Wang; Zunhai Su; Yiwei Wang; Qianli Wang; Shuzhou Yuan; Ercong Nie; Xufeng Duan; Qibo Xue; Zeping Yu; Chenming Shang; Xiao Liang; Jing Xiong; Hui Shen; Chaofan Tao; Zhengwu Liu; Senjie Jin; Zhiheng Xi; Dongdong Zhang; Sophia Ananiadou; Tao Gui; Ruobing Xie; Hayden Kwok-Hay So; Hinrich Schütze; Xuanjing Huang; Qi Zhang; Ngai Wong
>
> **摘要:** Mechanistic Interpretability (MI) has emerged as a vital approach to demystify the opaque decision-making of Large Language Models (LLMs). However, existing reviews primarily treat MI as an observational science, summarizing analytical insights while lacking a systematic framework for actionable intervention. To bridge this gap, we present a practical survey structured around the pipeline: "Locate, Steer, and Improve." We formally categorize Localizing (diagnosis) and Steering (intervention) methods based on specific Interpretable Objects to establish a rigorous intervention protocol. Furthermore, we demonstrate how this framework enables tangible improvements in Alignment, Capability, and Efficiency, effectively operationalizing MI as an actionable methodology for model optimization. The curated paper list of this work is available at https://github.com/rattlesnakey/Awesome-Actionable-MI-Survey.
>
---
#### [replaced 065] The Flexibility Trap: Why Arbitrary Order Limits Reasoning Potential in Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型研究任务，探讨扩散语言模型的任意顺序生成问题。指出当前方法反而限制了推理能力，提出简化策略提升效果。**

- **链接: [https://arxiv.org/pdf/2601.15165v2](https://arxiv.org/pdf/2601.15165v2)**

> **作者:** Zanlin Ni; Shenzhi Wang; Yang Yue; Tianyu Yu; Weilin Zhao; Yeguo Hua; Tianyi Chen; Jun Song; Cheng Yu; Bo Zheng; Gao Huang
>
> **备注:** Code and pre-trained models: https://github.com/LeapLabTHU/JustGRPO
>
> **摘要:** Diffusion Large Language Models (dLLMs) break the rigid left-to-right constraint of traditional LLMs, enabling token generation in arbitrary orders. Intuitively, this flexibility implies a solution space that strictly supersets the fixed autoregressive trajectory, theoretically unlocking superior reasoning potential for general tasks like mathematics and coding. Consequently, numerous works have leveraged reinforcement learning (RL) to elicit the reasoning capability of dLLMs. In this paper, we reveal a counter-intuitive reality: arbitrary order generation, in its current form, narrows rather than expands the reasoning boundary of dLLMs. We find that dLLMs tend to exploit this order flexibility to bypass high-uncertainty tokens that are crucial for exploration, leading to a premature collapse of the solution space. This observation motivates a rethink of RL approaches for dLLMs, where considerable complexities, such as handling combinatorial trajectories and intractable likelihoods, are often devoted to preserving this flexibility. We demonstrate that effective reasoning can be better elicited by intentionally forgoing arbitrary order and applying standard Group Relative Policy Optimization (GRPO) instead. Our approach, JustGRPO, is minimalist yet surprisingly effective (e.g., 89.1% accuracy on GSM8K) while fully retaining the parallel decoding ability of dLLMs. Project page: https://nzl-thu.github.io/the-flexibility-trap
>
---
#### [replaced 066] Consistent Kernel Change-Point Detection under m-Dependence for Text Segmentation
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属于文本分割任务，解决在存在依赖性的文本数据中检测变化点的问题。通过理论分析和实验验证，提出并评估了基于核方法的变更点检测技术。**

- **链接: [https://arxiv.org/pdf/2510.03437v2](https://arxiv.org/pdf/2510.03437v2)**

> **作者:** Jairo Diaz-Rodriguez; Mumin Jia
>
> **备注:** This paper is withdrawn due to an error in the proof of Proposition 3, which is used to support Theorem 1
>
> **摘要:** Kernel change-point detection (KCPD) has become a widely used tool for identifying structural changes in complex data. While existing theory establishes consistency under independence assumptions, real-world sequential data such as text exhibits strong dependencies. We establish new guarantees for KCPD under $m$-dependent data: specifically, we prove consistency in the number of detected change points and weak consistency in their locations under mild additional assumptions. We perform an LLM-based simulation that generates synthetic $m$-dependent text to validate the asymptotics. To complement these results, we present the first comprehensive empirical study of KCPD for text segmentation with modern embeddings. Across diverse text datasets, KCPD with text embeddings outperforms baselines in standard text segmentation metrics. We demonstrate through a case study on Taylor Swift's tweets that KCPD not only provides strong theoretical and simulated reliability but also practical effectiveness for text segmentation tasks.
>
---
#### [replaced 067] Multimodal Evaluation of Russian-language Architectures
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态模型评估任务，旨在解决俄罗斯语种多模态基准缺失的问题。构建了MERA Multi框架，包含18个针对俄语特点的评估任务，提供统一指标和基线结果。**

- **链接: [https://arxiv.org/pdf/2511.15552v3](https://arxiv.org/pdf/2511.15552v3)**

> **作者:** Artem Chervyakov; Ulyana Isaeva; Anton Emelyanov; Artem Safin; Maria Tikhonova; Alexander Kharitonov; Yulia Lyakh; Petr Surovtsev; Denis Shevelev; Vildan Saburov; Vasily Konovalov; Elisei Rykov; Ivan Sviridov; Amina Miftakhova; Ilseyar Alimova; Alexander Panchenko; Alexander Kapitanov; Alena Fenogenova
>
> **备注:** EACL main track
>
> **摘要:** Multimodal large language models (MLLMs) are currently at the center of research attention, showing rapid progress in scale and capabilities, yet their intelligence, limitations, and risks remain insufficiently understood. To address these issues, particularly in the context of the Russian language, where no multimodal benchmarks currently exist, we introduce MERA Multi, an open multimodal evaluation framework for Russian-spoken architectures. The benchmark is instruction-based and encompasses default text, image, audio, and video modalities, comprising 18 newly constructed evaluation tasks for both general-purpose models and modality-specific architectures (imageto-text, video-to-text, and audio-to-text). Our contributions include: (i) a universal taxonomy of multimodal abilities; (ii) 18 datasets created entirely from scratch with attention to Russian cultural and linguistic specificity, unified prompts, and metrics; (iii) baseline results for both closed-source and open-source models; (iv) a methodology for preventing benchmark leakage, including watermarking for private sets. While our current focus is on Russian, the proposed benchmark provides a replicable methodology for constructing multimodal benchmarks in typologically diverse languages, particularly within the Slavic language family.
>
---
#### [replaced 068] LLMPopcorn: Exploring LLMs as Assistants for Popular Micro-video Generation
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于AI生成微视频任务，旨在探索LLMs辅助制作受欢迎的微视频。研究解决如何有效利用LLMs生成高人气内容的问题，并通过实验评估不同模型性能。**

- **链接: [https://arxiv.org/pdf/2502.12945v3](https://arxiv.org/pdf/2502.12945v3)**

> **作者:** Junchen Fu; Xuri Ge; Kaiwen Zheng; Alexandros Karatzoglou; Ioannis Arapakis; Xin Xin; Yongxin Ni; Joemon M. Jose
>
> **备注:** Accepted by ICASSP2026
>
> **摘要:** In an era where micro-videos dominate platforms like TikTok and YouTube, AI-generated content is nearing cinematic quality. The next frontier is using large language models (LLMs) to autonomously create viral micro-videos, a largely untapped potential that could shape the future of AI-driven content creation. To address this gap, this paper presents the first exploration of LLM-assisted popular micro-video generation (LLMPopcorn). We selected popcorn as the icon for this paper because it symbolizes leisure and entertainment, aligning with this study on leveraging LLMs as assistants for generating popular micro-videos that are often consumed during leisure time. Specifically, we empirically study the following research questions: (i) How can LLMs be effectively utilized to assist popular micro-video generation? (ii) To what extent can prompt-based enhancements optimize the LLM-generated content for higher popularity? (iii) How well do various LLMs and video generators perform in the popular micro-video generation task? Exploring these questions, we show that advanced LLMs like DeepSeek-V3 can generate micro-videos with popularity rivaling human content. Prompt enhancement further boosts results, while benchmarking highlights DeepSeek-V3 and R1 for LLMs, and LTX-Video and HunyuanVideo for video generation. This work advances AI-assisted micro-video creation and opens new research directions. The code is publicly available at https://github.com/GAIR-Lab/LLMPopcorn.
>
---
#### [replaced 069] MangaVQA and MangaLMM: A Benchmark and Specialized Model for Multimodal Manga Understanding
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态漫画理解任务，旨在提升模型对漫画文本与图像的综合理解能力。提出MangaVQA基准和MangaLMM模型，解决漫画中图文协同理解问题。**

- **链接: [https://arxiv.org/pdf/2505.20298v3](https://arxiv.org/pdf/2505.20298v3)**

> **作者:** Jeonghun Baek; Kazuki Egashira; Shota Onohara; Atsuyuki Miyai; Yuki Imajuku; Hikaru Ikuta; Kiyoharu Aizawa
>
> **备注:** EACL 2026 Findings. Project page: https://manga109.github.io/MangaVQA_LMM/
>
> **摘要:** Manga, or Japanese comics, is a richly multimodal narrative form that blends images and text in complex ways. Teaching large multimodal models (LMMs) to understand such narratives at a human-like level could help manga creators reflect on and refine their stories. To this end, we introduce two benchmarks for multimodal manga understanding: MangaOCR, which targets in-page text recognition, and MangaVQA, a novel benchmark designed to evaluate contextual understanding through visual question answering. MangaVQA consists of 526 high-quality, manually constructed question-answer pairs, enabling reliable evaluation across diverse narrative and visual scenarios. Building on these benchmarks, we develop MangaLMM, a manga-specialized model finetuned from the open-source LMM Qwen2.5-VL to jointly handle both tasks. Through extensive experiments, including comparisons with proprietary models such as GPT-4o and Gemini 2.5, we assess how well LMMs understand manga. Our benchmark and model provide a comprehensive foundation for evaluating and advancing LMMs in the richly narrative domain of manga.
>
---
#### [replaced 070] Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于对话系统鲁棒性评估任务，旨在解决多轮对话中模型一致性问题。通过生存分析方法，研究模型在对抗攻击下的失效时间，提出有效监控机制。**

- **链接: [https://arxiv.org/pdf/2510.02712v3](https://arxiv.org/pdf/2510.02712v3)**

> **作者:** Yubo Li; Ramayya Krishnan; Rema Padman
>
> **摘要:** Large Language Models (LLMs) have revolutionized conversational AI, yet their robustness in extended multi-turn dialogues remains poorly understood. Existing evaluation frameworks focus on static benchmarks and single-turn assessments, failing to capture the temporal dynamics of conversational degradation that characterize real-world interactions. In this work, we present a large-scale survival analysis of conversational robustness, modeling failure as a time-to-event process over 36,951 turns from 9 state-of-the-art LLMs on the MT-Consistency benchmark. Our framework combines Cox proportional hazards, Accelerated Failure Time (AFT), and Random Survival Forest models with simple semantic drift features. We find that abrupt prompt-to-prompt semantic drift sharply increases the hazard of inconsistency, whereas cumulative drift is counterintuitively \emph{protective}, suggesting adaptation in conversations that survive multiple shifts. AFT models with model-drift interactions achieve the best combination of discrimination and calibration, and proportional hazards checks reveal systematic violations for key drift covariates, explaining the limitations of Cox-style modeling in this setting. Finally, we show that a lightweight AFT model can be turned into a turn-level risk monitor that flags most failing conversations several turns before the first inconsistent answer while keeping false alerts modest. These results establish survival analysis as a powerful paradigm for evaluating multi-turn robustness and for designing practical safeguards for conversational AI systems.
>
---
#### [replaced 071] LAILA: A Large Trait-Based Dataset for Arabic Automated Essay Scoring
- **分类: cs.CL**

- **简介: 该论文提出LAILA，一个用于阿拉伯语自动作文评分的大规模数据集，解决阿拉伯语自动评分数据不足的问题，包含多维度评分和基准测试结果。**

- **链接: [https://arxiv.org/pdf/2512.24235v2](https://arxiv.org/pdf/2512.24235v2)**

> **作者:** May Bashendy; Walid Massoud; Sohaila Eltanbouly; Salam Albatarni; Marwan Sayed; Abrar Abir; Houda Bouamor; Tamer Elsayed
>
> **备注:** Accepted at EACL 2026 - main conference
>
> **摘要:** Automated Essay Scoring (AES) has gained increasing attention in recent years, yet research on Arabic AES remains limited due to the lack of publicly available datasets. To address this, we introduce LAILA, the largest publicly available Arabic AES dataset to date, comprising 7,859 essays annotated with holistic and trait-specific scores on seven dimensions: relevance, organization, vocabulary, style, development, mechanics, and grammar. We detail the dataset design, collection, and annotations, and provide benchmark results using state-of-the-art Arabic and English models in prompt-specific and cross-prompt settings. LAILA fills a critical need in Arabic AES research, supporting the development of robust scoring systems.
>
---
#### [replaced 072] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频理解任务，解决 streaming 视频实时处理问题。提出 HERMES 架构，通过 KV 缓存实现高效记忆管理，提升处理速度与精度。**

- **链接: [https://arxiv.org/pdf/2601.14724v2](https://arxiv.org/pdf/2601.14724v2)**

> **作者:** Haowei Zhang; Shudong Yang; Jinlan Fu; See-Kiong Ng; Xipeng Qiu
>
> **摘要:** Recent advancements in Multimodal Large Language Models (MLLMs) have demonstrated significant improvement in offline video understanding. However, extending these capabilities to streaming video inputs, remains challenging, as existing models struggle to simultaneously maintain stable understanding performance, real-time responses, and low GPU memory overhead. To address this challenge, we propose HERMES, a novel training-free architecture for real-time and accurate understanding of video streams. Based on a mechanistic attention investigation, we conceptualize KV cache as a hierarchical memory framework that encapsulates video information across multiple granularities. During inference, HERMES reuses a compact KV cache, enabling efficient streaming understanding under resource constraints. Notably, HERMES requires no auxiliary computations upon the arrival of user queries, thereby guaranteeing real-time responses for continuous video stream interactions, which achieves 10$\times$ faster TTFT compared to prior SOTA. Even when reducing video tokens by up to 68% compared with uniform sampling, HERMES achieves superior or comparable accuracy across all benchmarks, with up to 11.4% gains on streaming datasets.
>
---
#### [replaced 073] How to Make LMs Strong Node Classifiers?
- **分类: cs.CL**

- **简介: 该论文属于图学习任务，旨在提升语言模型在节点分类中的性能。通过引入拓扑和语义增强及轻量GNN引导，使LMs达到SOTA水平。**

- **链接: [https://arxiv.org/pdf/2410.02296v3](https://arxiv.org/pdf/2410.02296v3)**

> **作者:** Zhe Xu; Kaveh Hassani; Si Zhang; Hanqing Zeng; Michihiro Yasunaga; Limei Wang; Dongqi Fu; Ning Yao; Bo Long; Hanghang Tong
>
> **摘要:** Language Models (LMs) are increasingly challenging the dominance of domain-specific models, such as Graph Neural Networks (GNNs) and Graph Transformers (GTs), in graph learning tasks. Following this trend, we propose a novel approach that empowers off-the-shelf LMs to achieve performance comparable to state-of-the-art (SOTA) GNNs on node classification tasks, without requiring any architectural modification. By preserving the LM's original architecture, our approach retains a key benefit of LM instruction tuning: the ability to jointly train on diverse datasets, fostering greater flexibility and efficiency. To achieve this, we introduce two key augmentation strategies: (1) Enriching LMs' input using topological and semantic retrieval methods, which provide richer contextual information, and (2) guiding the LMs' classification process through a lightweight GNN classifier that effectively prunes class candidates. Our experiments on real-world datasets show that backbone Flan-T5 LMs equipped with these augmentation strategies outperform SOTA text-output node classifiers and are comparable to top-performing vector-output node classifiers. By bridging the gap between specialized node classifiers and general LMs, this work paves the way for more versatile and widely applicable graph learning models. We will open-source the code upon publication.
>
---
#### [replaced 074] Mitigating the Modality Gap: Few-Shot Out-of-Distribution Detection with Multi-modal Prototypes and Image Bias Estimation
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于目标检测任务，解决视觉语言模型在分布外检测中的模态差距问题。通过引入图像和文本原型及新框架SUPREME提升检测性能。**

- **链接: [https://arxiv.org/pdf/2502.00662v2](https://arxiv.org/pdf/2502.00662v2)**

> **作者:** Yimu Wang; Evelien Riddell; Adrian Chow; Sean Sedwards; Krzysztof Czarnecki
>
> **备注:** WACV 2026
>
> **摘要:** Existing vision-language model (VLM)-based methods for out-of-distribution (OOD) detection typically rely on similarity scores between input images and in-distribution (ID) text prototypes. However, the modality gap between image and text often results in high false positive rates, as OOD samples can exhibit high similarity to ID text prototypes. To mitigate the impact of this modality gap, we propose incorporating ID image prototypes along with ID text prototypes. We present theoretical analysis and empirical evidence indicating that this approach enhances VLM-based OOD detection performance without any additional training. To further reduce the gap between image and text, we introduce a novel few-shot tuning framework, SUPREME, comprising biased prompts generation (BPG) and image-text consistency (ITC) modules. BPG enhances image-text fusion and improves generalization by conditioning ID text prototypes on the Gaussian-based estimated image domain bias; ITC reduces the modality gap by minimizing intra- and inter-modal distances. Moreover, inspired by our theoretical and empirical findings, we introduce a novel OOD score $S_{\textit{GMP}}$, leveraging uni- and cross-modal similarities. Finally, we present extensive experiments to demonstrate that SUPREME consistently outperforms existing VLM-based OOD detection methods.
>
---
#### [replaced 075] Teaching Small Language Models to Learn Logic through Meta-Learning
- **分类: cs.CL**

- **简介: 该论文属于逻辑推理任务，旨在提升小语言模型的逻辑能力。通过元学习方法，使模型掌握抽象推理模式，解决其在低数据下的泛化问题。**

- **链接: [https://arxiv.org/pdf/2505.14313v3](https://arxiv.org/pdf/2505.14313v3)**

> **作者:** Leonardo Bertolazzi; Manuel Vargas Guzmán; Raffaella Bernardi; Maciej Malicki; Jakub Szymanik
>
> **备注:** EACL 2026 Main
>
> **摘要:** Large language models (LLMs) are increasingly evaluated on reasoning tasks, yet their logical abilities remain contested. To address this, we study LLMs' reasoning in a well-defined fragment of logic: syllogistic reasoning. We cast the problem as premise selection and construct controlled datasets to isolate logical competence. Beyond evaluation, an open challenge is enabling LLMs to acquire abstract inference patterns that generalize to novel structures. We propose to apply few-shot meta-learning to this domain, thereby encouraging models to extract rules across tasks rather than memorize patterns within tasks. Although meta-learning has been little explored in the context of logic learnability, our experiments show that it is effective: small models (1.5B-7B) fine-tuned with meta-learning demonstrate strong gains in generalization, with especially pronounced benefits in low-data regimes. These meta-learned models outperform GPT-4o and o3-mini on our syllogistic reasoning task.
>
---
#### [replaced 076] Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于安全防护任务，旨在解决RL微调带来的有害行为问题。提出TokenBuncher，通过抑制模型响应熵来防御有害RL微调。**

- **链接: [https://arxiv.org/pdf/2508.20697v2](https://arxiv.org/pdf/2508.20697v2)**

> **作者:** Weitao Feng; Lixu Wang; Tianyi Wei; Jie Zhang; Chongyang Gao; Sinong Zhan; Peizhuo Lv; Wei Dong
>
> **备注:** Project Hompage: https://tokenbuncher.github.io/
>
> **摘要:** As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate more advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response entropy. By constraining entropy, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task performance and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.
>
---
#### [replaced 077] Induce, Align, Predict: Zero-Shot Stance Detection via Cognitive Inductive Reasoning
- **分类: cs.CL**

- **简介: 该论文属于零样本立场检测任务，旨在解决在无标签数据情况下识别文本立场的问题。提出CIRF框架，通过认知归纳推理实现高效、可解释的零样本推理。**

- **链接: [https://arxiv.org/pdf/2506.13470v3](https://arxiv.org/pdf/2506.13470v3)**

> **作者:** Bowen Zhang; Jun Ma; Fuqiang Niu; Li Dong; Jinzhou Cao; Genan Dai
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** Zero-shot stance detection (ZSSD) seeks to determine the stance of text toward previously unseen targets, a task critical for analyzing dynamic and polarized online discourse with limited labeled data. While large language models (LLMs) offer zero-shot capabilities, prompting-based approaches often fall short in handling complex reasoning and lack robust generalization to novel targets. Meanwhile, LLM-enhanced methods still require substantial labeled data and struggle to move beyond instance-level patterns, limiting their interpretability and adaptability. Inspired by cognitive science, we propose the Cognitive Inductive Reasoning Framework (CIRF), a schema-driven method that bridges linguistic inputs and abstract reasoning via automatic induction and application of cognitive reasoning schemas. CIRF abstracts first-order logic patterns from raw text into multi-relational schema graphs in an unsupervised manner, and leverages a schema-enhanced graph kernel model to align input structures with schema templates for robust, interpretable zero-shot inference. Extensive experiments on SemEval-2016, VAST, and COVID-19-Stance benchmarks demonstrate that CIRF not only establishes new state-of-the-art results, but also achieves comparable performance with just 30% of the labeled data, demonstrating its strong generalization and efficiency in low-resource settings.
>
---
#### [replaced 078] Examining the Utility of Self-disclosure Types for Modeling Annotators of Social Norms
- **分类: cs.CL**

- **简介: 该论文属于社会规范标注任务，旨在通过分析自述信息提升标注者标签预测效果。研究探讨了不同类型的自述信息对模型性能的影响，发现少量相关评论即可有效，且广泛采样优于多样化采样。**

- **链接: [https://arxiv.org/pdf/2512.16034v2](https://arxiv.org/pdf/2512.16034v2)**

> **作者:** Kieran Henderson; Kian Omoomi; Vasudha Varadarajan; Allison Lahnala; Charles Welch
>
> **备注:** Accepted EACL Findings
>
> **摘要:** Recent work has explored the use of personal information in the form of persona sentences or self-disclosures to improve modeling of individual characteristics and prediction of annotator labels for subjective tasks. The volume of personal information has historically been restricted and thus little exploration has gone into understanding what kind of information is most informative for predicting annotator labels. In this work, we categorize self-disclosures and use them to build annotator models for predicting judgments of social norms. We perform several ablations and analyses to examine the impact of the type of information on our ability to predict annotation patterns. Contrary to previous work, only a small number of comments related to the original post are needed. Lastly, a more diverse sample of annotator self-disclosures did not lead to the best performance. Sampling from a larger pool of comments without filtering still yields the best performance, suggesting that there is still much to uncover in terms of what information about an annotator is most useful for verdict prediction.
>
---
#### [replaced 079] iBERT: Interpretable Embeddings via Sense Decomposition
- **分类: cs.CL**

- **简介: 该论文提出iBERT，用于生成可解释的嵌入表示，解决语言中判别线索的模块化分解问题。通过稀疏组合语义向量，提升风格任务表现并支持可控表示。**

- **链接: [https://arxiv.org/pdf/2510.09882v2](https://arxiv.org/pdf/2510.09882v2)**

> **作者:** Vishal Anand; Milad Alshomary; Kathleen McKeown
>
> **备注:** Accepted to the Main Proceedings of EACL 2026. Camera-ready version
>
> **摘要:** We present iBERT (interpretable-BERT), an encoder to produce inherently interpretable and controllable embeddings - designed to modularize and expose the discriminative cues present in language, such as semantic or stylistic structure. Each input token is represented as a sparse, non-negative mixture over k context-independent sense vectors, which can be pooled into sentence embeddings or used directly at the token level. This enables modular control over representation, before any decoding or downstream use. To demonstrate our model's interpretability, we evaluate it on a suite of style-focused tasks. On the STEL benchmark, it improves style representation effectiveness by ~8 points over SBERT-style baselines, while maintaining competitive performance on authorship verification. Because each embedding is a structured composition of interpretable senses, we highlight how specific style attributes get assigned to specific sense vectors. While our experiments center on style, iBERT is not limited to stylistic modeling. Its structural modularity is designed to interpretably decompose whichever discriminative signals are present in the data - enabling generalization even when supervision blends semantic or stylistic factors.
>
---
#### [replaced 080] Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型推理任务，旨在提升模型在复杂任务中的推理能力。针对 latent reasoning 易失效的问题，提出 LTPO 框架，在不更新参数的情况下优化中间思考向量，增强模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.04182v4](https://arxiv.org/pdf/2510.04182v4)**

> **作者:** Wengao Ye; Yan Liang; Lianlei Shan
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have shifted from explicit Chain-of-Thought (CoT) reasoning to more efficient latent reasoning, where intermediate thoughts are represented as vectors rather than text. However, latent reasoning can be brittle on challenging, out-of-distribution tasks where robust reasoning is most critical. To overcome these limitations, we introduce Latent Thought Policy Optimization (LTPO), a parameter-free framework that enhances LLM reasoning entirely at test time, without requiring model parameter updates. LTPO treats intermediate latent "thought" vectors as dynamic parameters that are actively optimized for each problem instance. It employs an online policy gradient method guided by an intrinsic, confidence-based reward signal computed directly from the frozen LLM's own output distributions, eliminating the need for external supervision or expensive text generation during optimization. Extensive experiments on five reasoning benchmarks show that LTPO not only matches or surpasses strong baselines on standard tasks but also demonstrates remarkable robustness where others fail. Most notably, on highly challenging AIME benchmarks where existing latent reasoning baselines collapse to near-zero accuracy, LTPO delivers substantial improvements, showcasing a unique capability for complex reasoning.
>
---
#### [replaced 081] GRAM: A Generative Foundation Reward Model for Reward Generalization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习中的奖励模型任务，旨在提升奖励模型的泛化能力。通过结合生成模型与监督学习，提出一种新的训练方法，实现更高效的奖励模型训练。**

- **链接: [https://arxiv.org/pdf/2506.14175v3](https://arxiv.org/pdf/2506.14175v3)**

> **作者:** Chenglong Wang; Yang Gan; Yifu Huo; Yongyu Mu; Qiaozhi He; Murun Yang; Bei Li; Tong Xiao; Chunliang Zhang; Tongran Liu; Jingbo Zhu
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** In aligning large language models (LLMs), reward models have played an important role, but are standardly trained as discriminative models and rely only on labeled human preference data. In this paper, we explore methods that train reward models using both unlabeled and labeled data. Building on the generative models in LLMs, we develop a generative reward model that is first trained via large-scale unsupervised learning and then fine-tuned via supervised learning. We also show that by using label smoothing, we are in fact optimizing a regularized pairwise ranking loss. This result, in turn, provides a new view of training reward models, which links generative models and discriminative models under the same class of training objectives. The outcome of these techniques is a foundation reward model, which can be applied to a wide range of tasks with little or no further fine-tuning effort. Extensive experiments show that this model generalizes well across several tasks, including response ranking, reinforcement learning from human feedback, and task adaptation with fine-tuning, achieving significant performance improvements over several strong baseline models.
>
---
#### [replaced 082] SIPDO: Closed-Loop Prompt Optimization via Synthetic Data Feedback
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SIPDO，解决提示优化问题，通过合成数据反馈实现闭环优化，提升大语言模型性能。**

- **链接: [https://arxiv.org/pdf/2505.19514v3](https://arxiv.org/pdf/2505.19514v3)**

> **作者:** Yaoning Yu; Ye Yu; Peiyan Zhang; Kai Wei; Haojing Luo; Haohan Wang
>
> **摘要:** Prompt quality plays a critical role in the performance of large language models (LLMs), motivating a growing body of work on prompt optimization. Most existing methods optimize prompts over a fixed dataset, assuming static input distributions and offering limited support for iterative improvement. We introduce SIPDO (Self-Improving Prompts through Data-Augmented Optimization), a closed-loop framework for prompt learning that integrates synthetic data generation into the optimization process. SIPDO couples a synthetic data generator with a prompt optimizer, where the generator produces new examples that reveal current prompt weaknesses and the optimizer incrementally refines the prompt in response. This feedback-driven loop enables systematic improvement of prompt performance without assuming access to external supervision or new tasks. Experiments across question answering and reasoning benchmarks show that SIPDO outperforms standard prompt tuning methods, highlighting the value of integrating data synthesis into prompt learning workflows.
>
---
#### [replaced 083] RotBench: Evaluating Multimodal Large Language Models on Identifying Image Rotation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于图像旋转识别任务，旨在评估多模态大语言模型在识别不同角度图像旋转方面的能力。研究发现模型在此任务上表现不佳，尤其难以区分90°和270°旋转。**

- **链接: [https://arxiv.org/pdf/2508.13968v3](https://arxiv.org/pdf/2508.13968v3)**

> **作者:** Tianyi Niu; Jaemin Cho; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** EACL 2026 Camera-Ready. Code and data: https://github.com/tianyiniu/RotBench
>
> **摘要:** We investigate to what extent Multimodal Large Language Models (MLLMs) can accurately identify the orientation of input images rotated 0°, 90°, 180°, and 270°. This task demands robust visual reasoning capabilities to detect rotational cues and contextualize spatial relationships within images, regardless of their orientation. To evaluate MLLMs on these abilities, we introduce RotBench, a 350-image manually-filtered benchmark comprising lifestyle, portrait, and landscape images. Despite the relatively simple nature of this task, we show that several state-of-the-art open and proprietary MLLMs, including GPT-5, o3, and Gemini-2.5-Pro, do not reliably identify rotation in input images. Providing models with auxiliary information -- including captions, depth maps, and more -- or using chain-of-thought prompting offers only small and inconsistent improvements. Our results indicate that most models are able to reliably identify right-side-up (0°) images, while certain models are able to identify upside-down (180°) images. None can reliably distinguish between 90° and 270° rotated images. Simultaneously showing the image rotated in different orientations leads to moderate performance gains for reasoning models, while a modified setup using voting improves the performance of weaker models. We further show that fine-tuning does not improve models' ability to distinguish 90° and 270° rotations, despite substantially improving the identification of 180° images. Together, these results reveal a significant gap between MLLMs' spatial reasoning capabilities and human perception in identifying rotation.
>
---
#### [replaced 084] Generation-Augmented Generation: A Plug-and-Play Framework for Private Knowledge Injection in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出GAG框架，解决大语言模型中私有知识注入问题，通过生成增强生成方法实现高效、稳定的知识融合与多领域部署。**

- **链接: [https://arxiv.org/pdf/2601.08209v2](https://arxiv.org/pdf/2601.08209v2)**

> **作者:** Rongji Li; Jian Xu; Xueqing Chen; Yisheng Yang; Jiayi Wang; Xingyu Chen; Chunyu Xie; Dawei Leng; Xu-Yao Zhang
>
> **摘要:** In domains such as biomedicine, materials, and finance, high-stakes deployment of large language models (LLMs) requires injecting private, domain-specific knowledge that is proprietary, fast-evolving, and under-represented in public pretraining. However, the two dominant paradigms for private knowledge injection each have pronounced drawbacks: fine-tuning is expensive to iterate, and continual updates risk catastrophic forgetting and general-capability regression; retrieval-augmented generation (RAG) keeps the base model intact but is brittle in specialized private corpora due to chunk-induced evidence fragmentation, retrieval drift, and long-context pressure that yields query-dependent prompt inflation. Inspired by how multimodal LLMs align heterogeneous modalities into a shared semantic space, we propose Generation-Augmented Generation (GAG), which treats private expertise as an additional expert modality and injects it via a compact, representation-level interface aligned to the frozen base model, avoiding prompt-time evidence serialization while enabling plug-and-play specialization and scalable multi-domain composition with reliable selective activation. Across two private scientific QA benchmarks (immunology adjuvant and catalytic materials) and mixed-domain evaluations, GAG improves specialist performance over strong RAG baselines by 15.34% and 14.86% on the two benchmarks, respectively, while maintaining performance on six open general benchmarks and enabling near-oracle selective activation for scalable multi-domain deployment. Code is publicly available at https://github.com/360CVGroup/GAG.
>
---
#### [replaced 085] Adaptive Constraint Propagation: Scaling Structured Inference for Large Language Models via Meta-Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出MetaJuLS，解决大语言模型中结构化推理的约束传播问题。通过元强化学习，实现跨语言和任务的快速适应，提升推理效率并降低碳足迹。**

- **链接: [https://arxiv.org/pdf/2601.00095v3](https://arxiv.org/pdf/2601.00095v3)**

> **作者:** Ibne Farabi Shihab; Sanjeda Akter; Anuj Sharma
>
> **摘要:** Large language models increasingly require structured inference, from JSON schema enforcement to multi-lingual parsing, where outputs must satisfy complex constraints. We introduce MetaJuLS, a meta-reinforcement learning approach that learns universal constraint propagation policies applicable across languages and tasks without task-specific retraining. By formulating structured inference as adaptive constraint propagation and training a Graph Attention Network with meta-learning, MetaJuLS achieves 1.5--2.0$\times$ speedups over GPU-optimized baselines while maintaining within 0.2\% accuracy of state-of-the-art parsers. On Universal Dependencies across 10 languages and LLM-constrained generation (LogicBench, GSM8K-Constrained), MetaJuLS demonstrates rapid cross-domain adaptation: a policy trained on English parsing adapts to new languages and tasks with 5--10 gradient steps (5--15 seconds) rather than requiring hours of task-specific training. Mechanistic analysis reveals the policy discovers human-like parsing strategies (easy-first) and novel non-intuitive heuristics. By reducing propagation steps in LLM deployments, MetaJuLS contributes to Green AI by directly reducing inference carbon footprint.
>
---
#### [replaced 086] ARTI-6: Towards Six-dimensional Articulatory Speech Encoding
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文提出ARTI-6，解决语音声学与发音运动之间的映射问题。通过六维发音特征、逆向模型和合成模型，实现高效且生理合理的语音编码与重建。**

- **链接: [https://arxiv.org/pdf/2509.21447v2](https://arxiv.org/pdf/2509.21447v2)**

> **作者:** Jihwan Lee; Sean Foley; Thanathai Lertpetchpun; Kevin Huang; Yoonjeong Lee; Tiantian Feng; Louis Goldstein; Dani Byrd; Shrikanth Narayanan
>
> **备注:** Accepted for ICASSP 2026
>
> **摘要:** We propose ARTI-6, a compact six-dimensional articulatory speech encoding framework derived from real-time MRI data that captures crucial vocal tract regions including the velum, tongue root, and larynx. ARTI-6 consists of three components: (1) a six-dimensional articulatory feature set representing key regions of the vocal tract; (2) an articulatory inversion model, which predicts articulatory features from speech acoustics leveraging speech foundation models, achieving a prediction correlation of 0.87; and (3) an articulatory synthesis model, which reconstructs intelligible speech directly from articulatory features, showing that even a low-dimensional representation can generate natural-sounding speech. Together, ARTI-6 provides an interpretable, computationally efficient, and physiologically grounded framework for advancing articulatory inversion, synthesis, and broader speech technology applications. The source code and speech samples are publicly available.
>
---
#### [replaced 087] Offline Preference Optimization via Maximum Marginal Likelihood Estimation
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大语言模型对齐任务，旨在解决RLHF复杂不稳定的问题。提出MMPO方法，通过最大边缘似然估计实现偏好优化，无需显式奖励模型。**

- **链接: [https://arxiv.org/pdf/2510.22881v2](https://arxiv.org/pdf/2510.22881v2)**

> **作者:** Saeed Najafi; Alona Fyshe
>
> **备注:** EACL 2026, camera ready
>
> **摘要:** Aligning Large Language Models (LLMs) with human preferences is crucial, but standard methods like Reinforcement Learning from Human Feedback (RLHF) are often complex and unstable. In this work, we propose a new, simpler approach that recasts alignment through the lens of Maximum Marginal Likelihood (MML) estimation. Our new MML based Preference Optimization (MMPO) maximizes the marginal log-likelihood of a preferred text output, using the preference pair as samples for approximation, and forgoes the need for both an explicit reward model and entropy maximization. We theoretically demonstrate that MMPO implicitly performs preference optimization, producing a weighted gradient that naturally up-weights chosen responses over rejected ones. Across models ranging from 135M to 8B parameters, we empirically show that MMPO: 1) is more stable with respect to the hyperparameter $β$ compared to alternative baselines, and 2) achieves competitive or superior preference alignment while better preserving the base model's general language capabilities. Through a series of ablation experiments, we show that this improved performance is indeed attributable to MMPO's implicit preference optimization within the gradient updates.
>
---
#### [replaced 088] The taggedPBC: Annotating a massive parallel corpus for crosslinguistic investigations
- **分类: cs.CL**

- **简介: 该论文属于跨语言研究任务，旨在解决数据不足与多样性问题。构建了包含1940种语言的标注平行语料库taggedPBC，提升跨语言研究的可行性。**

- **链接: [https://arxiv.org/pdf/2505.12560v4](https://arxiv.org/pdf/2505.12560v4)**

> **作者:** Hiram Ring
>
> **摘要:** Existing datasets available for crosslinguistic investigations have tended to focus on large amounts of data for a small group of languages or a small amount of data for a large number of languages. This means that claims based on these datasets are limited in what they reveal about universal properties of the human language faculty. While this has begun to change through the efforts of projects seeking to develop tagged corpora for a large number of languages, such efforts are still constrained by limits on resources. The current paper reports on a large tagged parallel dataset which has been developed to partially address this issue. The taggedPBC contains POS-tagged parallel text data from more than 1,940 languages, representing 155 language families and 78 isolates, dwarfing previously available resources. The accuracy of particular tags in this dataset is shown to correlate well with both existing SOTA taggers for high-resource languages (SpaCy, Trankit) as well as hand-tagged corpora (Universal Dependencies Treebanks). Additionally, a novel measure derived from this dataset, the N1 ratio, correlates with expert determinations of intransitive word order in three typological databases (WALS, Grambank, AUTOYP) such that a Gaussian Naive Bayes classifier trained on this feature can accurately identify basic intransitive word order for languages not in those databases. While much work is still needed to expand and develop this dataset, the taggedPBC is an important step to enable corpus-based crosslinguistic investigations, and is made available for research and collaboration via GitHub.
>
---
#### [replaced 089] ATOM: AdapTive and OptiMized dynamic temporal knowledge graph construction using LLMs
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出ATOM，用于动态构建时间知识图谱。解决传统静态KG无法适应实时数据变化的问题，通过分解文本为原子事实并进行双时间建模，提升提取效果与效率。**

- **链接: [https://arxiv.org/pdf/2510.22590v2](https://arxiv.org/pdf/2510.22590v2)**

> **作者:** Yassir Lairgi; Ludovic Moncla; Khalid Benabdeslem; Rémy Cazabet; Pierre Cléau
>
> **备注:** Accepted at the Findings of the Association for Computational Linguistics: EACL 2026
>
> **摘要:** In today's rapidly expanding data landscape, knowledge extraction from unstructured text is vital for real-time analytics, temporal inference, and dynamic memory frameworks. However, traditional static knowledge graph (KG) construction often overlooks the dynamic and time-sensitive nature of real-world data, limiting adaptability to continuous changes. Moreover, recent zero- or few-shot approaches that avoid domain-specific fine-tuning or reliance on prebuilt ontologies often suffer from instability across multiple runs, as well as incomplete coverage of key facts. To address these challenges, we introduce ATOM (AdapTive and OptiMized), a few-shot and scalable approach that builds and continuously updates Temporal Knowledge Graphs (TKGs) from unstructured texts. ATOM splits input documents into minimal, self-contained "atomic" facts, improving extraction exhaustivity and stability. Then, it constructs atomic TKGs from these facts, employing a dual-time modeling that distinguishes between when information is observed and when it is valid. The resulting atomic TKGs are subsequently merged in parallel. Empirical evaluations demonstrate that ATOM achieves ~18% higher exhaustivity, ~33% better stability, and over ~90% latency reduction compared to baseline methods, demonstrating a strong scalability potential for dynamic TKG construction.
>
---
#### [replaced 090] Persuasion Tokens for Editing Factual Knowledge in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于知识编辑任务，旨在解决LLMs更新信息效率低的问题。提出 persuasion tokens（P-Tokens），替代传统事实特定示例，实现高效知识编辑。**

- **链接: [https://arxiv.org/pdf/2601.16781v2](https://arxiv.org/pdf/2601.16781v2)**

> **作者:** Paul Youssef; Christin Seifert; Jörg Schlötterer
>
> **备注:** Accepted at EACL Main 2026
>
> **摘要:** In-context knowledge editing (IKE) is a promising technique for updating Large Language Models (LLMs) with new information. However, IKE relies on lengthy, fact-specific demonstrations which are costly to create and consume significant context window space. In this paper, we introduce persuasion tokens (P-Tokens) -- special tokens trained to replicate the effect of IKE demonstrations, enabling efficient knowledge editing without requiring fact-specific demonstrations. We evaluate P-Tokens across two editing datasets and three LLMs, demonstrating performance comparable to, and often exceeding, IKE. We further find that editing performance is robust to distractors with small negative effects to neighboring facts, and that increasing the number of P-Tokens improves performance. Our work addresses key limitations of IKE and provides a more practical and scalable alternative for editing LLMs.
>
---
#### [replaced 091] MoSE: Hierarchical Self-Distillation Enhances Early Layer Embeddings
- **分类: cs.CL; cs.AI; cs.PL; cs.SE**

- **简介: 该论文提出MoSE模型，解决代码检索与分类任务中的性能与准确率平衡问题，通过自蒸馏机制提升早期层表示，实现灵活部署。**

- **链接: [https://arxiv.org/pdf/2503.03008v3](https://arxiv.org/pdf/2503.03008v3)**

> **作者:** Andrea Gurioli; Federico Pennino; João Monteiro; Maurizio Gabbrielli
>
> **备注:** Accepted in the AAAI 2026 Main Technical Track
>
> **摘要:** Deploying language models often requires navigating accuracy vs. performance trade-offs to meet latency constraints while preserving utility. Traditional model distillation reduces size but incurs substantial costs through training separate models. We introduce ModularStarEncoder (MoSE), a 1-billion-parameter multi-exit encoder for code retrieval and classification that employs a novel Self-Distillation mechanism. This approach significantly enhances lower-layer representations, enabling flexible deployment of different model portions with favorable performance trade-offs. Our architecture improves text-to-code and code-to-code search by targeting specific encoder layers as exit heads, where higher layers guide earlier ones during training, thereby improving intermediate representations at minimal additional cost. We further enhance MoSE with a repository-level contextual loss that maximizes training context window utilization. Additionally, we release a new dataset created through code translation that extends text-to-code benchmarks with cross-language code-to-code pairs. Evaluations demonstrate the effectiveness of Self-Distillation as a principled approach to trading inference cost for accuracy across various code understanding tasks.
>
---
#### [replaced 092] Deep Research with Open-Domain Evaluation and Multi-Stage Guardrails for Safety
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全评测任务，旨在解决深度研究框架在报告质量与安全性上的不足。提出DeepResearchGuard和DRSafeBench，提升防御效果并保障报告安全。**

- **链接: [https://arxiv.org/pdf/2510.10994v2](https://arxiv.org/pdf/2510.10994v2)**

> **作者:** Wei-Chieh Huang; Henry Peng Zou; Yaozu Wu; Dongyuan Li; Yankai Chen; Weizhi Zhang; Yangning Li; Angelo Zangari; Jizhou Guo; Chunyu Miao; Liancheng Fang; Langzhou He; Yinghui Li; Renhe Jiang; Philip S. Yu
>
> **摘要:** Deep research frameworks have shown promising capabilities in synthesizing comprehensive reports from web sources. While deep research possesses significant potential to address complex issues through planning and research cycles, existing frameworks are deficient in sufficient evaluation procedures and stage-specific protections. They typically treat evaluation as exact match accuracy of question-answering, but overlook crucial aspects of report quality such as credibility, coherence, breadth, depth, and safety. This oversight may result in hazardous or malicious sources being integrated into the final report. To address this, we introduce DeepResearchGuard, a framework featuring four-stage safeguards with open-domain evaluation, and DRSafeBench, a novel stage-wise safety benchmark. Evaluating across GPT-4o, o4-mini, Gemini-2.5-flash, DeepSeek-v3, GPT-5, DeepResearchGuard improves defense success rates by 16.53% while reducing over-refusal to 6%. Through extensive experiments, we show that DRSafeBench enables comprehensive open-domain evaluation and stage-aware defenses that effectively block harmful content propagation, while systematically improving report quality without excessive over-refusal rates.
>
---
#### [replaced 093] Towards Automated Kernel Generation in the Era of LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自动化内核生成任务，旨在解决人工编写高效内核耗时且难以扩展的问题，通过LLM和代理系统实现内核优化的自动化。**

- **链接: [https://arxiv.org/pdf/2601.15727v2](https://arxiv.org/pdf/2601.15727v2)**

> **作者:** Yang Yu; Peiyu Zang; Chi Hsu Tsai; Haiming Wu; Yixin Shen; Jialing Zhang; Haoyu Wang; Zhiyou Xiao; Jingze Shi; Yuyu Luo; Wentao Zhang; Chunlei Men; Guang Liu; Yonghua Lin
>
> **备注:** 10 pages, 1 figure
>
> **摘要:** The performance of modern AI systems is fundamentally constrained by the quality of their underlying kernels, which translate high-level algorithmic semantics into low-level hardware operations. Achieving near-optimal kernels requires expert-level understanding of hardware architectures and programming models, making kernel engineering a critical but notoriously time-consuming and non-scalable process. Recent advances in large language models (LLMs) and LLM-based agents have opened new possibilities for automating kernel generation and optimization. LLMs are well-suited to compress expert-level kernel knowledge that is difficult to formalize, while agentic systems further enable scalable optimization by casting kernel development as an iterative, feedback-driven loop. Rapid progress has been made in this area. However, the field remains fragmented, lacking a systematic perspective for LLM-driven kernel generation. This survey addresses this gap by providing a structured overview of existing approaches, spanning LLM-based approaches and agentic optimization workflows, and systematically compiling the datasets and benchmarks that underpin learning and evaluation in this domain. Moreover, key open challenges and future research directions are further outlined, aiming to establish a comprehensive reference for the next generation of automated kernel optimization. To keep track of this field, we maintain an open-source GitHub repository at https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation.
>
---
#### [replaced 094] On the Failure of Latent State Persistence in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM缺乏持续潜在状态的问题，通过三个实验揭示其在保持内部表示上的不足，属于模型认知机制分析任务，旨在评估LLM的内部表征 fidelity。**

- **链接: [https://arxiv.org/pdf/2505.10571v5](https://arxiv.org/pdf/2505.10571v5)**

> **作者:** Jen-tse Huang; Kaiser Sun; Wenxuan Wang; Mark Dredze
>
> **备注:** 8 pages, 6 figures, 9 tables
>
> **摘要:** While Large Language Models (LLMs) excel in reasoning, whether they can sustain persistent latent states remains under-explored. The capacity to maintain and manipulate unexpressed, internal representations-analogous to human working memory-is a cornerstone of complex reasoning. In this paper, we formalize and quantify the "Latent State Persistence" (LSP) gap through three novel experiments. First, we utilize a Number Guessing Game, demonstrating that across independent queries, LLMs fail to allocate probability mass to a singular hidden choice, violating a fundamental probabilistic principle. Second, we employ a Yes-No Game to show that as the number of questions increases, LLMs suffer from "concept drift," leading to inevitable self-contradictions due to the lack of LSP. Finally, inspired by Mathematical Mentalism, we task models with tracking transformations on hidden variables, revealing a failure in variable binding and state evolution when the initial state is not explicitly present in the context. Collectively, these findings suggest that LLMs function as reactive post-hoc solvers rather than proactive planners with LSP. Our work provides a framework for evaluating the fidelity of internal representations and highlights a fundamental architectural divergence between autoregressive transformers and human-like cognition.
>
---
#### [replaced 095] Next Token Knowledge Tracing: Exploiting Pretrained LLM Representations to Decode Student Behaviour
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识追踪任务，旨在提升学生学习预测的准确性。通过将KT重构为文本的下一个词预测任务，利用预训练语言模型捕捉学生行为与题目内容的模式，从而提高预测性能。**

- **链接: [https://arxiv.org/pdf/2511.02599v2](https://arxiv.org/pdf/2511.02599v2)**

> **作者:** Max Norris; Kobi Gal; Sahan Bulathwela
>
> **摘要:** Modelling student knowledge is a key challenge when leveraging AI in education, with major implications for personalised learning. The Knowledge Tracing (KT) task aims to predict how students will respond to educational questions in learning environments, based on their prior interactions. Existing KT models typically use response correctness along with metadata like skill tags and timestamps, often overlooking the question text, which is an important source of pedagogical insight. This omission poses a lost opportunity while limiting predictive performance. We propose Next Token Knowledge Tracing (NTKT), a novel approach that reframes KT as a next-token prediction task using pretrained Large Language Models (LLMs). NTKT represents both student histories and question content as sequences of text, allowing LLMs to learn patterns in both behaviour and language. Our series of experiments significantly improves performance over state-of-the-art neural KT models and generalises much better to cold-start questions and users. These findings highlight the importance of question content in KT and demonstrate the benefits of leveraging pretrained representations of LLMs to model student learning more effectively.
>
---
#### [replaced 096] Sentinel: Decoding Context Utilization via Attention Probing for Efficient LLM Context Compression
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型上下文压缩任务，解决长且噪声多的检索上下文问题。通过分析注意力行为，提出Sentinel框架实现高效压缩。**

- **链接: [https://arxiv.org/pdf/2505.23277v2](https://arxiv.org/pdf/2505.23277v2)**

> **作者:** Yong Zhang; Heng Li; Yanwen Huang; Ning Cheng; Yang Guo; Yun Zhu; Yanmeng Wang; Shaojun Wang; Jing Xiao
>
> **备注:** Preprint
>
> **摘要:** Retrieval-augmented generation (RAG) often suffers from long and noisy retrieved contexts. Prior context compression methods rely on predefined importance metrics or supervised compression models, rather than on the model's own inference-time behavior. We propose Sentinel, a lightweight sentence-level compression framework that treats context compression as an understanding decoding problem. Sentinel probes native attention behaviors of a frozen LLM with a lightweight readout to decode which parts of the context are actually utilized when answering a query, rather than using attention as a direct relevance score. We empirically observe that decoded relevance signals exhibit sufficient consistency across model scales to support effective compression with compact proxy models. On LongBench, Sentinel with a 0.5B proxy model achieves up to 5x compression while matching the QA performance of 7B-scale baselines, and despite being trained only on English QA data, generalizes effectively to Chinese and out-of-domain settings.
>
---
#### [replaced 097] DAIQ: Auditing Demographic Attribute Inference from Question in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI伦理任务，旨在解决LLMs从中性问题推断敏感人口属性的问题。通过DAIQ框架评估模型行为，发现多数模型会默认社会主导类别并生成刻板推理。**

- **链接: [https://arxiv.org/pdf/2508.15830v2](https://arxiv.org/pdf/2508.15830v2)**

> **作者:** Srikant Panda; Hitesh Laxmichand Patel; Shahad Al-Khalifa; Amit Agarwal; Hend Al-Khalifa; Sharefah Al-Ghamdi
>
> **备注:** Preprint
>
> **摘要:** Recent evaluations of Large language models (LLMs) audit social bias primarily through prompts that explicitly reference demographic attributes, overlooking whether models infer sensitive demographics from neutral questions. Such inference constitutes epistemic overreach and raises concerns for privacy. We introduce Demographic Attribute Inference from Questions (DAIQ), a diagnostic audit framework for evaluating demographic inference under epistemic uncertainty. We evaluate 18 open- and closed-source LLMs across six real-world domains and five demographic attributes. We find that many models infer demographics from neutral questions, defaulting to socially dominant categories and producing stereotype-aligned rationales. These behaviors persist across model families, scales and decoding settings, indicating reliance on learned population priors. We further show that inferred demographics can condition downstream responses and that abstention oriented prompting substantially reduces unintended inference without model fine-tuning. Our results suggest that current bias evaluations are incomplete and motivate evaluation standards that assess not only how models respond to demographic information, but whether they should infer it at all.
>
---
#### [replaced 098] Neural Algorithmic Reasoning for Hypergraphs with Looped Transformers
- **分类: cs.LG; cs.AI; cs.CC; cs.CL**

- **简介: 该论文研究如何将Loop Transformer应用于超图的算法推理，解决传统方法在处理高阶关系时的不足。通过降维和超边感知编码，提升模型对超图算法的模拟能力。**

- **链接: [https://arxiv.org/pdf/2501.10688v3](https://arxiv.org/pdf/2501.10688v3)**

> **作者:** Zekai Huang; Yingyu Liang; Zhenmei Shi; Zhao Song; Zhen Zhuang
>
> **摘要:** Looped Transformers have shown exceptional neural algorithmic reasoning capability in simulating traditional graph algorithms, but their application to more complex structures like hypergraphs remains underexplored. Hypergraphs generalize graphs by modeling higher-order relationships among multiple entities, enabling richer representations but introducing significant computational challenges. In this work, we extend the Loop Transformer architecture's neural algorithmic reasoning capability to simulate hypergraph algorithms, addressing the gap between neural networks and combinatorial optimization over hypergraphs. Specifically, we propose a novel degradation mechanism for reducing hypergraphs to graph representations, enabling the simulation of graph-based algorithms, such as Dijkstra's shortest path. Furthermore, we introduce a hyperedge-aware encoding scheme to simulate hypergraph-specific algorithms, exemplified by Helly's algorithm. We establish theoretical guarantees for these simulations, demonstrating the feasibility of processing high-dimensional and combinatorial data using Loop Transformers. This work highlights the potential of Transformers as general-purpose algorithmic solvers for structured data.
>
---
#### [replaced 099] FilterRAG: Zero-Shot Informed Retrieval-Augmented Generation to Mitigate Hallucinations in VQA
- **分类: cs.CV; cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于视觉问答任务，旨在解决VQA中的幻觉问题。通过结合检索增强生成与BLIP-VQA，提出FilterRAG框架，提升答案准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2502.18536v3](https://arxiv.org/pdf/2502.18536v3)**

> **作者:** Nobin Sarwar
>
> **备注:** 12 pages, 6 figures and 2 tables; Accepted at ICCV 2025 Workshop on Building Foundation Models You Can Trust (T2FM)
>
> **摘要:** Visual Question Answering requires models to generate accurate answers by integrating visual and textual understanding. However, VQA models still struggle with hallucinations, producing convincing but incorrect answers, particularly in knowledge-driven and Out-of-Distribution scenarios. We introduce FilterRAG, a retrieval-augmented framework that combines BLIP-VQA with Retrieval-Augmented Generation to ground answers in external knowledge sources like Wikipedia and DBpedia. FilterRAG achieves 36.5% accuracy on the OK-VQA dataset, demonstrating its effectiveness in reducing hallucinations and improving robustness in both in-domain and Out-of-Distribution settings. These findings highlight the potential of FilterRAG to improve Visual Question Answering systems for real-world deployment.
>
---
#### [replaced 100] Jet-RL: Enabling On-Policy FP8 Reinforcement Learning with Unified Training and Rollout Precision Flow
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RL训练效率低的问题。通过统一FP8精度流程，提升训练与推理的一致性，实现更高效的RL训练。**

- **链接: [https://arxiv.org/pdf/2601.14243v2](https://arxiv.org/pdf/2601.14243v2)**

> **作者:** Haocheng Xi; Charlie Ruan; Peiyuan Liao; Yujun Lin; Han Cai; Yilong Zhao; Shuo Yang; Kurt Keutzer; Song Han; Ligeng Zhu
>
> **备注:** 11 pages, 6 figures, 4 tables
>
> **摘要:** Reinforcement learning (RL) is essential for enhancing the complex reasoning capabilities of large language models (LLMs). However, existing RL training pipelines are computationally inefficient and resource-intensive, with the rollout phase accounting for over 70% of total training time. Quantized RL training, particularly using FP8 precision, offers a promising approach to mitigating this bottleneck. A commonly adopted strategy applies FP8 precision during rollout while retaining BF16 precision for training. In this work, we present the first comprehensive study of FP8 RL training and demonstrate that the widely used BF16-training + FP8-rollout strategy suffers from severe training instability and catastrophic accuracy collapse under long-horizon rollouts and challenging tasks. Our analysis shows that these failures stem from the off-policy nature of the approach, which introduces substantial numerical mismatch between training and inference. Motivated by these observations, we propose Jet-RL, an FP8 RL training framework that enables robust and stable RL optimization. The key idea is to adopt a unified FP8 precision flow for both training and rollout, thereby minimizing numerical discrepancies and eliminating the need for inefficient inter-step calibration. Extensive experiments validate the effectiveness of Jet-RL: our method achieves up to 33% speedup in the rollout phase, up to 41% speedup in the training phase, and a 16% end-to-end speedup over BF16 training, while maintaining stable convergence across all settings and incurring negligible accuracy degradation.
>
---
#### [replaced 101] MemWeaver: A Hierarchical Memory from Textual Interactive Behaviors for Personalized Generation
- **分类: cs.CL**

- **简介: 该论文提出MemWeaver，用于个性化生成任务，解决用户历史文本信息利用不足的问题。通过构建层次化记忆，捕捉用户兴趣的时序与语义结构，提升生成内容的个性化程度。**

- **链接: [https://arxiv.org/pdf/2510.07713v2](https://arxiv.org/pdf/2510.07713v2)**

> **作者:** Shuo Yu; Mingyue Cheng; Daoyu Wang; Qi Liu; Zirui Liu; Ze Guo; Xiaoyu Tao
>
> **备注:** Accepted by The Web Conference 2026 (WWW'26) 12 pages, 8 figures
>
> **摘要:** The primary form of user-internet engagement is shifting from leveraging implicit feedback signals, such as browsing and clicks, to harnessing the rich explicit feedback provided by textual interactive behaviors. This shift unlocks a rich source of user textual history, presenting a profound opportunity for a deeper form of personalization. However, prevailing approaches offer only a shallow form of personalization, as they treat user history as a flat list of texts for retrieval and fail to model the rich temporal and semantic structures reflecting dynamic nature of user interests. In this work, we propose \textbf{MemWeaver}, a framework that weaves the user's entire textual history into a hierarchical memory to power deeply personalized generation. The core innovation of our memory lies in its ability to capture both the temporal evolution of interests and the semantic relationships between different activities. To achieve this, MemWeaver builds two complementary memory components that both integrate temporal and semantic information, but at different levels of abstraction: behavioral memory, which captures specific user actions, and cognitive memory, which represents long-term preferences. This dual-component memory serves as a comprehensive representation of the user, allowing large language models (LLMs) to reason over both concrete behaviors and abstracted cognitive traits. This leads to content generation that is deeply aligned with their latent preferences. Experiments on the six datasets of the Language Model Personalization (LaMP) benchmark validate the efficacy of MemWeaver. Our code is available.
>
---
#### [replaced 102] LingGen: Scalable Multi-Attribute Linguistic Control via Power-Law Masking
- **分类: cs.CL**

- **简介: 该论文提出LingGen，用于多属性语言控制的文本生成模型，解决细粒度控制大量实值属性的问题。通过BOS嵌入和幂律掩码提升控制精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2410.24201v2](https://arxiv.org/pdf/2410.24201v2)**

> **作者:** Mohamed Elgaar; Hadi Amiri
>
> **备注:** EACL 2026
>
> **摘要:** We present LingGen, a controlled text generation model that allows fine-grained control over a large number of real-valued linguistic attributes. It encodes target attribute values with a dedicated linguistic attribute encoder and conditions the language model by injecting the resulting representation into the language model using the beginning-of-sequence (BOS) embeddings. To improve robustness when controlling different attribute subsets, we introduce P-MASKING, which samples per-example attribute masking rates from a truncated Pareto distribution during training. Across 1-40 control attributes, LingGen achieves the lowest average control error among evaluated methods, while remaining efficient at inference and receiving the highest fluency scores in human evaluation. Ablations show that Pareto-sampled masking and BOS-based injection are effective choices compared to alternative masking and integration variants.
>
---
#### [replaced 103] A Review of Incorporating Psychological Theories in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，探讨如何将心理学理论融入大语言模型开发，以提升其认知与交互能力。工作包括分析六大学派心理学对LLM的启示及应用现状。**

- **链接: [https://arxiv.org/pdf/2505.00003v2](https://arxiv.org/pdf/2505.00003v2)**

> **作者:** Zizhou Liu; Ziwei Gong; Lin Ai; Zheng Hui; Run Chen; Colin Wayne Leach; Michelle R. Greene; Julia Hirschberg
>
> **摘要:** Psychological insights have long shaped pivotal NLP breakthroughs, from attention mechanisms to reinforcement learning and social modeling. As Large Language Models (LLMs) develop, there is a rising consensus that psychology is essential for capturing human-like cognition, behavior, and interaction. This paper reviews how psychological theories can inform and enhance stages of LLM development. Our review integrates insights from six subfields of psychology, including cognitive, developmental, behavioral, social, personality psychology, and psycholinguistics. With stage-wise analysis, we highlight current trends and gaps in how psychological theories are applied. By examining both cross-domain connections and points of tension, we aim to bridge disciplinary divides and promote more thoughtful integration of psychology into NLP research.
>
---
#### [replaced 104] Inside Out: Evolving User-Centric Core Memory Trees for Long-Term Personalized Dialogue Systems
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决长期个性化对话中的记忆噪声和人格不一致问题。提出PersonaTree框架实现可控记忆增长与更新，提升对话一致性与效率。**

- **链接: [https://arxiv.org/pdf/2601.05171v2](https://arxiv.org/pdf/2601.05171v2)**

> **作者:** Jihao Zhao; Ding Chen; Zhaoxin Fan; Kerun Xu; Mengting Hu; Bo Tang; Feiyu Xiong; Zhiyu Li
>
> **摘要:** Existing long-term personalized dialogue systems struggle to reconcile unbounded interaction streams with finite context constraints, often succumbing to memory noise accumulation, reasoning degradation, and persona inconsistency. To address these challenges, this paper proposes Inside Out, a framework that utilizes a globally maintained PersonaTree as the carrier of long-term user profiling. By constraining the trunk with an initial schema and updating the branches and leaves, PersonaTree enables controllable growth, achieving memory compression while preserving consistency. Moreover, we train a lightweight MemListener via reinforcement learning with process-based rewards to produce structured, executable, and interpretable {ADD, UPDATE, DELETE, NO_OP} operations, thereby supporting the dynamic evolution of the personalized tree. During response generation, PersonaTree is directly leveraged to enhance outputs in latency-sensitive scenarios; when users require more details, the agentic mode is triggered to introduce details on-demand under the constraints of the PersonaTree. Experiments show that PersonaTree outperforms full-text concatenation and various personalized memory systems in suppressing contextual noise and maintaining persona consistency. Notably, the small MemListener model achieves memory-operation decision performance comparable to, or even surpassing, powerful reasoning models such as DeepSeek-R1-0528 and Gemini-3-Pro.
>
---
#### [replaced 105] RAFFLES: Reasoning-based Attribution of Faults for LLM Systems
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出RAFFLES，用于检测LLM系统的故障。属于系统评估任务，解决手动审查效率低的问题，通过迭代推理框架自动识别故障。**

- **链接: [https://arxiv.org/pdf/2509.06822v2](https://arxiv.org/pdf/2509.06822v2)**

> **作者:** Chenyang Zhu; Spencer Hong; Jingyu Wu; Kushal Chawla; Charlotte Tang; Youbing Yin; Nathan Wolfe; Erin Babinsky; Daben Liu
>
> **摘要:** The advent of complex, interconnected long-horizon LLM systems has made it incredibly tricky to identify where and when these systems break down. Evaluation capabilities that currently exist today are limited in that they often focus on simple metrics, end-to-end outcomes, and are dependent on the perspectives of humans. In order to match the increasing complexity of these many component systems, evaluation frameworks must also be able to reason, probe, iterate, and understand the nuanced logic passing through these systems. In this paper, we present RAFFLES, an offline evaluation architecture that incorporates iterative reasoning. Specifically, RAFFLES operates as an iterative, multi-component pipeline, using a central Judge to systematically identify faults and a set of specialized Evaluators to assess the quality of the candidate faults as well as rationales of the Judge. We evaluated RAFFLES with several benchmarks - the Who&When dataset to identify step-level faults in multi-agent systems and the ReasonEval datasets to diagnose step-level mathematical reasoning errors. RAFFLES outperforms strong baselines, achieving an accuracy of over 20% and 50% on the Who&When Hand-Crafted and Algorithmically-Generated datasets, and over 80% on the ReasonEval datasets. These results demonstrate a key step towards introducing automated fault detection for autonomous systems over labor-intensive manual review.
>
---
#### [replaced 106] Efficient Knowledge Probing of Large Language Models by Adapting Pre-trained Embeddings
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出PEEK方法，通过适配预训练嵌入模型来高效探测大语言模型的知识，解决传统方法计算成本高的问题。**

- **链接: [https://arxiv.org/pdf/2508.06030v2](https://arxiv.org/pdf/2508.06030v2)**

> **作者:** Kartik Sharma; Yiqiao Jin; Rakshit Trivedi; Srijan Kumar
>
> **摘要:** Large language models (LLMs) acquire knowledge across diverse domains such as science, history, and geography encountered during generative pre-training. However, due to their stochasticity, it is difficult to predict what LLMs have acquired. Prior work has developed different ways to probe this knowledge by investigating the hidden representations, crafting specific task prompts, curating representative samples, and estimating their uncertainty. However, these methods require making forward passes through the underlying model to probe the LLM's knowledge about a specific fact, making them computationally expensive and time-consuming. To bridge this gap, we propose $\textbf{PEEK}$ or $\textbf{P}$roxy $\textbf{E}$mbeddings to $\textbf{E}$stimate $\textbf{K}$nowledge of LLMs, by leveraging the pre-trained embedding models that effectively encode factual knowledge as text or graphs as proxies for LLMs. First, we identify a training set of facts known by LLMs through various probing strategies and then adapt embedding models to predict the LLM outputs with a linear decoder layer. Comprehensive evaluation on $3$ Wikipedia-derived datasets, $4$ LLMs, and $7$ embedding models shows that embeddings can predict LLM knowledge on a held-out set with up to 90 % accuracy. Furthermore, we find that sentence embedding models are more suitable than graph embeddings to predict LLM knowledge, shedding light on the underlying representation of the factual landscape. Thus, we believe that knowledge-adapted embeddings can be used to identify knowledge gaps in LLMs at scale and can provide deeper insights into LLMs' internal inductive bias. The code and data are made available at https://github.com/claws-lab/peek.
>
---
#### [replaced 107] Knowing the Facts but Choosing the Shortcut: Understanding How Large Language Models Compare Entities
- **分类: cs.CL; cs.AI**

- **简介: 论文研究LLM在实体比较任务中的决策机制，探讨其是依赖真实知识还是表面启发式。通过分析数值属性比较，发现模型常受实体流行度、提及顺序和语义共现影响，且大模型更倾向于使用可靠知识。**

- **链接: [https://arxiv.org/pdf/2510.16815v2](https://arxiv.org/pdf/2510.16815v2)**

> **作者:** Hans Hergen Lehmann; Jae Hee Lee; Steven Schockaert; Stefan Wermter
>
> **备注:** 34 pages, 20 figures. Accepted for EACL 2026
>
> **摘要:** Large Language Models (LLMs) are increasingly used for knowledge-based reasoning tasks, yet understanding when they rely on genuine knowledge versus superficial heuristics remains challenging. We investigate this question through entity comparison tasks by asking models to compare entities along numerical attributes (e.g., ``Which river is longer, the Danube or the Nile?''), which offer clear ground truth for systematic analysis. Despite having sufficient numerical knowledge to answer correctly, LLMs frequently make predictions that contradict this knowledge. We identify three heuristic biases that strongly influence model predictions: entity popularity, mention order, and semantic co-occurrence. For smaller models, a simple logistic regression using only these surface cues predicts model choices more accurately than the model's own numerical predictions, suggesting heuristics largely override principled reasoning. Crucially, we find that larger models (32B parameters) selectively rely on numerical knowledge when it is more reliable, while smaller models (7--8B parameters) show no such discrimination, which explains why larger models outperform smaller ones even when the smaller models possess more accurate knowledge. Chain-of-thought prompting steers all models towards using the numerical features across all model sizes.
>
---
#### [replaced 108] MEDIC: Comprehensive Evaluation of Leading Indicators for LLM Safety and Utility in Clinical Applications
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗领域LLM评估任务，旨在解决模型理论能力与临床实用脱节的问题。通过构建MEDIC框架，评估模型在临床场景中的安全性和实用性，揭示性能短板。**

- **链接: [https://arxiv.org/pdf/2409.07314v2](https://arxiv.org/pdf/2409.07314v2)**

> **作者:** Praveenkumar Kanithi; Clément Christophe; Marco AF Pimentel; Tathagata Raha; Prateek Munjal; Nada Saadi; Hamza A Javed; Svetlana Maslenkova; Nasir Hayat; Ronnie Rajan; Shadab Khan
>
> **备注:** Technical report
>
> **摘要:** While Large Language Models (LLMs) achieve superhuman performance on standardized medical licensing exams, these static benchmarks have become saturated and increasingly disconnected from the functional requirements of clinical workflows. To bridge the gap between theoretical capability and verified utility, we introduce MEDIC, a comprehensive evaluation framework establishing leading indicators across various clinical dimensions. Beyond standard question-answering, we assess operational capabilities using deterministic execution protocols and a novel Cross-Examination Framework (CEF), which quantifies information fidelity and hallucination rates without reliance on reference texts. Our evaluation across a heterogeneous task suite exposes critical performance trade-offs: we identify a significant knowledge-execution gap, where proficiency in static retrieval does not predict success in operational tasks such as clinical calculation or SQL generation. Furthermore, we observe a divergence between passive safety (refusal) and active safety (error detection), revealing that models fine-tuned for high refusal rates often fail to reliably audit clinical documentation for factual accuracy. These findings demonstrate that no single architecture dominates across all dimensions, highlighting the necessity of a portfolio approach to clinical model deployment. As part of this investigation, we released a public leaderboard on Hugging Face.\footnote{https://huggingface.co/spaces/m42-health/MEDIC-Benchmark}
>
---
#### [replaced 109] Adjust for Trust: Mitigating Trust-Induced Inappropriate Reliance on AI Assistance
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于人机协作任务，旨在解决用户对AI过度或不足依赖的问题。通过调整AI行为，如提供解释或暂停，减少不当依赖，提升决策准确性。**

- **链接: [https://arxiv.org/pdf/2502.13321v2](https://arxiv.org/pdf/2502.13321v2)**

> **作者:** Tejas Srinivasan; Jesse Thomason
>
> **摘要:** Trust biases how users rely on AI recommendations in AI-assisted decision-making tasks, with low and high levels of trust resulting in increased under- and over-reliance, respectively. We propose that AI assistants should adapt their behavior through trust-adaptive interventions to mitigate such inappropriate reliance. For instance, when user trust is low, providing an explanation can elicit more careful consideration of the assistant's advice by the user. In two decision-making scenarios -- laypeople answering science questions and doctors making medical diagnoses -- we find that providing supporting and counter-explanations during moments of low and high trust, respectively, yields up to 38% reduction in inappropriate reliance and 20% improvement in decision accuracy. We are similarly able to reduce over-reliance by adaptively inserting forced pauses to promote deliberation. Our results highlight how AI adaptation to user trust facilitates appropriate reliance, presenting exciting avenues for improving human-AI collaboration.
>
---
#### [replaced 110] Mind the Gap: Benchmarking LLM Uncertainty and Calibration with Specialty-Aware Clinical QA and Reasoning-Based Behavioural Features
- **分类: cs.CL**

- **简介: 该论文属于临床QA任务，旨在解决LLM在医疗领域中的不确定性量化问题。通过评估不同模型和方法，分析其在不同专科和题型下的表现，提出新的轻量级方法提升可靠性。**

- **链接: [https://arxiv.org/pdf/2506.10769v3](https://arxiv.org/pdf/2506.10769v3)**

> **作者:** Alberto Testoni; Iacer Calixto
>
> **备注:** Accepted at EACL 2026 (Main Conference)
>
> **摘要:** Reliable uncertainty quantification (UQ) is essential when employing large language models (LLMs) in high-risk domains such as clinical question answering (QA). In this work, we evaluate uncertainty estimation methods for clinical QA focusing, for the first time, on eleven clinical specialties and six question types, and across ten open-source LLMs (general-purpose, biomedical, and reasoning models), alongside representative proprietary models. We analyze score-based UQ methods, present a case study introducing a novel lightweight method based on behavioral features derived from reasoning-oriented models, and examine conformal prediction as a complementary set-based approach. Our findings reveal that uncertainty reliability is not a monolithic property, but one that depends on clinical specialty and question type due to shifts in calibration and discrimination. Our results highlight the need to select or ensemble models based on their distinct, complementary strengths and clinical use.
>
---
#### [replaced 111] TaoSR-AGRL: Adaptive Guided Reinforcement Learning Framework for E-commerce Search Relevance
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于电商搜索相关性预测任务，解决LLM在复杂查询和规则下的推理能力不足问题，提出TaoSR-AGRL框架提升模型准确性与规则遵守。**

- **链接: [https://arxiv.org/pdf/2510.08048v3](https://arxiv.org/pdf/2510.08048v3)**

> **作者:** Jianhui Yang; Yiming Jin; Pengkun Jiao; Chenhe Dong; Zerui Huang; Shaowei Yao; Xiaojiang Zhou; Dan Ou; Haihong Tang
>
> **备注:** Accepted to The Web Conference (WWW) 2026, Industry Track
>
> **摘要:** Query-product relevance prediction is fundamental to e-commerce search and has become even more critical in the era of AI-powered shopping, where semantic understanding and complex reasoning directly shape the user experience and business conversion. Large Language Models (LLMs) enable generative, reasoning-based approaches, typically aligned via supervised fine-tuning (SFT) or preference optimization methods like Direct Preference Optimization (DPO). However, the increasing complexity of business rules and user queries exposes the inability of existing methods to endow models with robust reasoning capacity for long-tail and challenging cases. Efforts to address this via reinforcement learning strategies like Group Relative Policy Optimization (GRPO) often suffer from sparse terminal rewards, offering insufficient guidance for multi-step reasoning and slowing convergence. To address these challenges, we propose TaoSR-AGRL, an Adaptive Guided Reinforcement Learning framework for LLM-based relevance prediction in Taobao Search Relevance. TaoSR-AGRL introduces two key innovations: (1) Rule-aware Reward Shaping, which decomposes the final relevance judgment into dense, structured rewards aligned with domain-specific relevance criteria; and (2) Adaptive Guided Replay, which identifies low-accuracy rollouts during training and injects targeted ground-truth guidance to steer the policy away from stagnant, rule-violating reasoning patterns toward compliant trajectories. TaoSR-AGRL was evaluated on large-scale real-world datasets and through online side-by-side human evaluations on Taobao Search. It consistently outperforms DPO and standard GRPO baselines in offline experiments, improving relevance accuracy, rule adherence, and training stability. The model trained with TaoSR-AGRL has been successfully deployed in the main search scenario on Taobao, serving hundreds of millions of users.
>
---
#### [replaced 112] A Survey on Multilingual Mental Disorders Detection from Social Media Data
- **分类: cs.CL**

- **简介: 该论文属于多语言心理健康检测任务，旨在解决非英语社交媒体数据中精神障碍识别不足的问题。工作包括整理25种语言的108个数据集，并分析文化因素对NLP模型的影响。**

- **链接: [https://arxiv.org/pdf/2505.15556v2](https://arxiv.org/pdf/2505.15556v2)**

> **作者:** Ana-Maria Bucur; Marcos Zampieri; Tharindu Ranasinghe; Fabio Crestani
>
> **摘要:** The increasing prevalence of mental disorders globally highlights the urgent need for effective digital screening methods that can be used in multilingual contexts. Most existing studies, however, focus on English data, overlooking critical mental health signals that may be present in non-English texts. To address this gap, we present a survey of the detection of mental disorders using social media data beyond the English language. We compile a comprehensive list of 108 datasets spanning 25 languages that can be used for developing NLP models for mental health screening. In addition, we discuss the cultural nuances that influence online language patterns and self-disclosure behaviors, and how these factors can impact the performance of NLP tools. Our survey highlights major challenges, including the scarcity of resources for low- and mid-resource languages and the dominance of depression-focused data over other disorders. By identifying these gaps, we advocate for interdisciplinary collaborations and the development of multilingual benchmarks to enhance mental health screening worldwide.
>
---
#### [replaced 113] Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究语言模型中语法与领域间的虚假相关性问题。工作包括分析这种相关性对模型性能的影响，并提出评估框架以检测此类现象。**

- **链接: [https://arxiv.org/pdf/2509.21155v3](https://arxiv.org/pdf/2509.21155v3)**

> **作者:** Chantal Shaib; Vinith M. Suriyakumar; Levent Sagun; Byron C. Wallace; Marzyeh Ghassemi
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** For an LLM to correctly respond to an instruction it must understand both the semantics and the domain (i.e., subject area) of a given task-instruction pair. However, syntax can also convey implicit information Recent work shows that syntactic templates -- frequent sequences of Part-of-Speech (PoS) tags -- are prevalent in training data and often appear in model outputs. In this work we characterize syntactic templates, domain, and semantics in task-instruction pairs. We identify cases of spurious correlations between syntax and domain, where models learn to associate a domain with syntax during training; this can sometimes override prompt semantics. Using a synthetic training dataset, we find that the syntactic-domain correlation can lower performance (mean 0.51 +/- 0.06) on entity knowledge tasks in OLMo-2 models (1B-13B). We introduce an evaluation framework to detect this phenomenon in trained models, and show that it occurs on a subset of the FlanV2 dataset in open (OLMo-2-7B; Llama-4-Maverick), and closed (GPT-4o) models. Finally, we present a case study on the implications for safety finetuning, showing that unintended syntactic-domain correlations can be used to bypass refusals in OLMo-2-7B Instruct and GPT-4o. Our findings highlight two needs: (1) to explicitly test for syntactic-domain correlations, and (2) to ensure syntactic diversity in training data, specifically within domains, to prevent such spurious correlations.
>
---
#### [replaced 114] The Need for a Socially-Grounded Persona Framework for User Simulation
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于用户模拟任务，旨在解决传统人口统计学角色构建的局限性。通过引入SCOPE框架，结合社会心理数据提升角色质量与行为预测准确性。**

- **链接: [https://arxiv.org/pdf/2601.07110v2](https://arxiv.org/pdf/2601.07110v2)**

> **作者:** Pranav Narayanan Venkit; Yu Li; Yada Pruksachatkun; Chien-Sheng Wu
>
> **摘要:** Synthetic personas are widely used to condition large language models (LLMs) for social simulation, yet most personas are still constructed from coarse sociodemographic attributes or summaries. We revisit persona creation by introducing SCOPE, a socially grounded framework for persona construction and evaluation, built from a 141-item, two-hour sociopsychological protocol collected from 124 U.S.-based participants. Across seven models, we find that demographic-only personas are a structural bottleneck: demographics explain only ~1.5% of variance in human response similarity. Adding sociopsychological facets improves behavioral prediction and reduces over-accentuation, and non-demographic personas based on values and identity achieve strong alignment with substantially lower bias. These trends generalize to SimBench (441 aligned questions), where SCOPE personas outperform default prompting and NVIDIA Nemotron personas, and SCOPE augmentation improves Nemotron-based personas. Our results indicate that persona quality depends on sociopsychological structure rather than demographic templates or summaries.
>
---
#### [replaced 115] Bears, all bears, and some bears. Language Constraints on Language Models' Inductive Inferences
- **分类: cs.CL**

- **简介: 该论文属于自然语言理解任务，研究语言结构如何影响模型的归纳推理。通过实验验证语言模型是否能区分不同语义表达，发现其行为与人类相似，差异源于归纳约束而非形式差异。**

- **链接: [https://arxiv.org/pdf/2601.09852v2](https://arxiv.org/pdf/2601.09852v2)**

> **作者:** Sriram Padmanabhan; Siyuan Song; Kanishka Misra
>
> **摘要:** Language places subtle constraints on how we make inductive inferences. Developmental evidence by Gelman et al. (2002) has shown children (4 years and older) to differentiate among generic statements ("Bears are daxable"), universally quantified NPs ("all bears are daxable") and indefinite plural NPs ("some bears are daxable") in extending novel properties to a specific member (all > generics > some), suggesting that they represent these types of propositions differently. We test if these subtle differences arise in general purpose statistical learners like Vision Language Models, by replicating the original experiment. On tasking them through a series of precondition tests (robust identification of categories in images and sensitivities to all and some), followed by the original experiment, we find behavioral alignment between models and humans. Post-hoc analyses on their representations revealed that these differences are organized based on inductive constraints and not surface-form differences.
>
---
#### [replaced 116] Language Diversity: Evaluating Language Usage and AI Performance on African Languages in Digital Spaces
- **分类: cs.CL**

- **简介: 该论文属于语言检测任务，旨在解决非洲语言在数字空间中数据稀缺与检测困难的问题。通过分析新闻与Reddit数据，发现新闻数据更适用于训练AI模型。**

- **链接: [https://arxiv.org/pdf/2512.01557v2](https://arxiv.org/pdf/2512.01557v2)**

> **作者:** Edward Ajayi; Eudoxie Umwari; Mawuli Deku; Prosper Singadi; Jules Udahemuka; Bekalu Tadele; Chukuemeka Edeh
>
> **摘要:** This study examines the digital representation of African languages and the challenges this presents for current language detection tools. We evaluate their performance on Yoruba, Kinyarwanda, and Amharic. While these languages are spoken by millions, their online usage on conversational platforms is often sparse, heavily influenced by English, and not representative of the authentic, monolingual conversations prevalent among native speakers. This lack of readily available authentic data online creates a challenge of scarcity of conversational data for training language models. To investigate this, data was collected from subreddits and local news sources for each language. The analysis showed a stark contrast between the two sources. Reddit data was minimal and characterized by heavy code-switching. Conversely, local news media offered a robust source of clean, monolingual language data, which also prompted more user engagement in the local language on the news publishers social media pages. Language detection models, including the specialized AfroLID and a general LLM, performed with near-perfect accuracy on the clean news data but struggled with the code-switched Reddit posts. The study concludes that professionally curated news content is a more reliable and effective source for training context-rich AI models for African languages than data from conversational platforms. It also highlights the need for future models that can process clean and code-switched text to improve the detection accuracy for African languages.
>
---
#### [replaced 117] Is In-Context Learning Learning?
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文探讨了上下文学习（ICL）是否属于真正的学习，分析其有效性与局限性，旨在理解ICL在未见任务上的泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.10414v3](https://arxiv.org/pdf/2509.10414v3)**

> **作者:** Adrian de Wynter
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** In-context learning (ICL) allows some autoregressive models to solve tasks via next-token prediction and without needing further training. This has led to claims about these model's ability to solve (learn) unseen tasks with only a few shots (exemplars) in the prompt. However, deduction does not always imply learning, as ICL does not explicitly encode a given observation. Instead, the models rely on their prior knowledge and the exemplars given, if any. We argue that, mathematically, ICL does constitute learning, but its full characterisation requires empirical work. We then carry out a large-scale analysis of ICL ablating out or accounting for memorisation, pretraining, distributional shifts, and prompting style and phrasing. We find that ICL is an effective learning paradigm, but limited in its ability to learn and generalise to unseen tasks. We note that, in the limit where exemplars become more numerous, accuracy is insensitive to exemplar distribution, model, prompt style, and the input's linguistic features. Instead, it deduces patterns from regularities in the prompt, which leads to distributional sensitivity, especially in prompting styles such as chain-of-thought. Given the varied accuracies on formally similar tasks, we conclude that autoregression's ad-hoc encoding is not a robust mechanism, and suggests limited all-purpose generalisability.
>
---
#### [replaced 118] Explaining Generalization of AI-Generated Text Detectors Through Linguistic Analysis
- **分类: cs.CL**

- **简介: 该论文属于AI文本检测任务，旨在解决模型在不同生成条件下的泛化问题。通过语言分析，研究检测器性能与语言特征的关系，提升跨场景的检测能力。**

- **链接: [https://arxiv.org/pdf/2601.07974v2](https://arxiv.org/pdf/2601.07974v2)**

> **作者:** Yuxi Xia; Kinga Stańczak; Benjamin Roth
>
> **摘要:** AI-text detectors achieve high accuracy on in-domain benchmarks, but often struggle to generalize across different generation conditions such as unseen prompts, model families, or domains. While prior work has reported these generalization gaps, there are limited insights about the underlying causes. In this work, we present a systematic study aimed at explaining generalization behavior through linguistic analysis. We construct a comprehensive benchmark that spans 6 prompting strategies, 7 large language models (LLMs), and 4 domain datasets, resulting in a diverse set of human- and AI-generated texts. Using this dataset, we fine-tune classification-based detectors on various generation settings and evaluate their cross-prompt, cross-model, and cross-dataset generalization. To explain the performance variance, we compute correlations between generalization accuracies and feature shifts of 80 linguistic features between training and test conditions. Our analysis reveals that generalization performance for specific detectors and evaluation conditions is significantly associated with linguistic features such as tense usage and pronoun frequency.
>
---
#### [replaced 119] A Markov Categorical Framework for Language Modeling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型研究，旨在解析其内部机制与训练目标的关系。通过马尔可夫范畴框架，揭示了训练如何影响表示空间和模型能力。**

- **链接: [https://arxiv.org/pdf/2507.19247v4](https://arxiv.org/pdf/2507.19247v4)**

> **作者:** Yifan Zhang
>
> **摘要:** Autoregressive language models achieve remarkable performance, yet a unified theory explaining their internal mechanisms, how training shapes their representations, and enables complex behaviors, remains elusive. We introduce a new analytical framework that models the single-step generation process as a composition of information-processing stages using the language of Markov categories. This compositional perspective provides a unified mathematical language to connect three critical aspects of language modeling that are typically studied in isolation: the training objective, the geometry of the learned representation space, and practical model capabilities. First, our framework provides a precise information-theoretic rationale for the success of multi-token prediction methods like speculative decoding, quantifying the information surplus a model's hidden state contains about tokens beyond the immediate next one. Second, we clarify how the standard negative log-likelihood (NLL) objective compels the model to learn not just the next word, but also the data's intrinsic conditional uncertainty, a process we formalize using categorical entropy. Our central result shows that, under a linear-softmax head with bounded features, minimizing NLL induces spectral alignment: the learned representation space aligns with the eigenspectrum of a predictive similarity operator. This work presents a powerful new lens for understanding how information flows through a model and how the training objective shapes its internal geometry.
>
---
#### [replaced 120] VoXtream: Full-Stream Text-to-Speech with Extremely Low Latency
- **分类: eess.AS; cs.CL; cs.HC; cs.LG; cs.SD**

- **简介: 该论文提出VoXtream，一种低延迟的流式文本转语音系统，解决实时语音生成问题。通过自回归架构和有限前瞻，实现快速响应。**

- **链接: [https://arxiv.org/pdf/2509.15969v2](https://arxiv.org/pdf/2509.15969v2)**

> **作者:** Nikita Torgashov; Gustav Eje Henter; Gabriel Skantze
>
> **备注:** 5 pages, 1 figure, accepted to IEEE ICASSP 2026
>
> **摘要:** We present VoXtream, a fully autoregressive, zero-shot streaming text-to-speech (TTS) system for real-time use that begins speaking from the first word. VoXtream directly maps incoming phonemes to audio tokens using a monotonic alignment scheme and a limited look-ahead that does not delay onset. Built around an incremental phoneme transformer, a temporal transformer predicting semantic and duration tokens, and a depth transformer producing acoustic tokens, VoXtream achieves, to our knowledge, the lowest initial delay among publicly available streaming TTS: 102 ms on GPU. Despite being trained on a mid-scale 9k-hour corpus, it matches or surpasses larger baselines on several metrics, while delivering competitive quality in both output- and full-streaming settings. Demo and code are available at https://herimor.github.io/voxtream.
>
---
#### [replaced 121] Towards Robust Evaluation of Visual Activity Recognition: Resolving Verb Ambiguity with Sense Clustering
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于视觉活动识别任务，旨在解决动词语义歧义问题。通过构建动词意义聚类，提出更鲁棒的评估方法，提升模型评价的准确性。**

- **链接: [https://arxiv.org/pdf/2508.04945v2](https://arxiv.org/pdf/2508.04945v2)**

> **作者:** Louie Hong Yao; Nicholas Jarvis; Tianyu Jiang
>
> **备注:** Accepted to Findings of the Association for Computational Linguistics: EACL 2026
>
> **摘要:** Evaluating visual activity recognition systems is challenging due to inherent ambiguities in verb semantics and image interpretation. When describing actions in images, synonymous verbs can refer to the same event (e.g., brushing vs. grooming), while different perspectives can lead to equally valid but distinct verb choices (e.g., piloting vs. operating). Standard exact-match evaluation, which relies on a single gold answer, fails to capture these ambiguities, resulting in an incomplete assessment of model performance. To address this, we propose a vision-language clustering framework that constructs verb sense clusters, providing a more robust evaluation. Our analysis of the imSitu dataset shows that each image maps to around four sense clusters, with each cluster representing a distinct perspective of the image. We evaluate multiple activity recognition models and compare our cluster-based evaluation with standard evaluation methods. Additionally, our human alignment analysis suggests that the cluster-based evaluation better aligns with human judgments, offering a more nuanced assessment of model performance.
>
---
#### [replaced 122] Universal Refusal Circuits Across LLMs: Cross-Model Transfer via Trajectory Replay and Concept-Basis Reconstruction
- **分类: cs.CL**

- **简介: 该论文属于安全对齐任务，旨在解决LLM拒绝行为的通用性问题。通过轨迹重放和概念重构，实现跨模型拒绝干预迁移，保持性能。**

- **链接: [https://arxiv.org/pdf/2601.16034v2](https://arxiv.org/pdf/2601.16034v2)**

> **作者:** Tony Cristofano
>
> **摘要:** Refusal behavior in aligned LLMs is often viewed as model-specific, yet we hypothesize it stems from a universal, low-dimensional semantic circuit shared across models. To test this, we introduce Trajectory Replay via Concept-Basis Reconstruction, a framework that transfers refusal interventions from donor to target models, spanning diverse architectures (e.g., Dense to MoE) and training regimes, without using target-side refusal supervision. By aligning layers via concept fingerprints and reconstructing refusal directions using a shared ``recipe'' of concept atoms, we map the donor's ablation trajectory into the target's semantic space. To preserve capabilities, we introduce a weight-SVD stability guard that projects interventions away from high-variance weight subspaces to prevent collateral damage. Our evaluation across 8 model pairs confirms that these transferred recipes consistently attenuate refusal while maintaining performance, providing strong evidence for the semantic universality of safety alignment.
>
---
#### [replaced 123] Augmenting Question Answering with A Hybrid RAG Approach
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于问答任务，旨在解决现有RAG方法检索不准确的问题。提出SSRAG混合架构，结合查询增强、代理路由和结构化检索，提升答案准确性与信息量。**

- **链接: [https://arxiv.org/pdf/2601.12658v2](https://arxiv.org/pdf/2601.12658v2)**

> **作者:** Tianyi Yang; Nashrah Haque; Vaishnave Jonnalagadda; Yuya Jeremy Ong; Zhehui Chen; Yanzhao Wu; Lei Yu; Divyesh Jadav; Wenqi Wei
>
> **备注:** 10 pages, 5 tables, 2 figures; presented at IEEE CogMI 2025
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a powerful technique for enhancing the quality of responses in Question-Answering (QA) tasks. However, existing approaches often struggle with retrieving contextually relevant information, leading to incomplete or suboptimal answers. In this paper, we introduce Structured-Semantic RAG (SSRAG), a hybrid architecture that enhances QA quality by integrating query augmentation, agentic routing, and a structured retrieval mechanism combining vector and graph based techniques with context unification. By refining retrieval processes and improving contextual grounding, our approach improves both answer accuracy and informativeness. We conduct extensive evaluations on three popular QA datasets, TruthfulQA, SQuAD and WikiQA, across five Large Language Models (LLMs), demonstrating that our proposed approach consistently improves response quality over standard RAG implementations.
>
---
#### [replaced 124] Beyond Single-Granularity Prompts: A Multi-Scale Chain-of-Thought Prompt Learning for Graph
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于图神经网络任务，旨在解决图数据中单一粒度提示信息不足的问题。提出MSGCOT框架，融合多尺度结构信息，提升提示语义多样性与性能。**

- **链接: [https://arxiv.org/pdf/2510.09394v4](https://arxiv.org/pdf/2510.09394v4)**

> **作者:** Ziyu Zheng; Yaming Yang; Ziyu Guan; Wei Zhao; Xinyan Huang; Weigang Lu
>
> **备注:** Accepted by WWW2026
>
> **摘要:** The ``pre-train, prompt" paradigm, designed to bridge the gap between pre-training tasks and downstream objectives, has been extended from the NLP domain to the graph domain and has achieved remarkable progress. Current mainstream graph prompt-tuning methods modify input or output features using learnable prompt vectors. However, existing approaches are confined to single-granularity (e.g., node-level or subgraph-level) during prompt generation, overlooking the inherently multi-scale structural information in graph data, which limits the diversity of prompt semantics. To address this issue, we pioneer the integration of multi-scale information into graph prompt and propose a Multi-Scale Graph Chain-of-Thought (MSGCOT) prompting framework. Specifically, we design a lightweight, low-rank coarsening network to efficiently capture multi-scale structural features as hierarchical basis vectors for prompt generation. Subsequently, mimicking human cognition from coarse-to-fine granularity, we dynamically integrate multi-scale information at each reasoning step, forming a progressive coarse-to-fine prompt chain. Extensive experiments on eight benchmark datasets demonstrate that MSGCOT outperforms the state-of-the-art single-granularity graph prompt-tuning method, particularly in few-shot scenarios, showcasing superior performance. The code is available at: https://github.com/zhengziyu77/MSGCOT.
>
---
#### [replaced 125] Retrieval-Augmented Code Generation: A Survey with Focus on Repository-Level Approaches
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于代码生成任务，旨在解决跨仓库的代码生成问题。通过回顾检索增强生成方法，分析其策略、架构及挑战，推动AI辅助软件工程发展。**

- **链接: [https://arxiv.org/pdf/2510.04905v2](https://arxiv.org/pdf/2510.04905v2)**

> **作者:** Yicheng Tao; Yao Qin; Yepang Liu
>
> **摘要:** Recent advancements in large language models (LLMs) have substantially improved automated code generation. While function-level and file-level generation have achieved promising results, real-world software development typically requires reasoning across entire repositories. This gives rise to the challenging task of Repository-Level Code Generation (RLCG), where models must capture long-range dependencies, ensure global semantic consistency, and generate coherent code spanning multiple files or modules. To address these challenges, Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm that integrates external retrieval mechanisms with LLMs, enhancing context-awareness and scalability. In this survey, we provide a comprehensive review of research on Retrieval-Augmented Code Generation (RACG), with an emphasis on repository-level approaches. We categorize existing work along several dimensions, including generation strategies, retrieval modalities, model architectures, training paradigms, and evaluation protocols. Furthermore, we summarize widely used datasets and benchmarks, analyze current limitations, and outline key challenges and opportunities for future research. Our goal is to establish a unified analytical framework for understanding this rapidly evolving field and to inspire continued progress in AI-powered software engineering.
>
---
#### [replaced 126] Autiverse: Eliciting Autistic Adolescents' Daily Narratives through AI-guided Multimodal Journaling
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于辅助 autistic 青少年叙事能力的任务，旨在解决传统日记方式的局限性。通过 AI 引导的多模态日记应用 Autiverse，帮助其组织日常经历与情感。**

- **链接: [https://arxiv.org/pdf/2509.17466v3](https://arxiv.org/pdf/2509.17466v3)**

> **作者:** Migyeong Yang; Kyungah Lee; Jinyoung Han; SoHyun Park; Young-Ho Kim
>
> **备注:** 19 pages excluding reference. Conditionally accepted to ACM CHI 2026
>
> **摘要:** Journaling can potentially serve as an effective method for autistic adolescents to improve narrative skills. However, its text-centric nature and high executive functioning demands present barriers to practice. We present Autiverse, an AI-guided multimodal journaling app for tablets that scaffolds daily narratives through conversational prompts and visual supports. Autiverse elicits key details of an adolescent-selected event through a stepwise dialogue with peer-like, customizable AI and composes them into an editable four-panel comic strip. Through a two-week deployment study with 10 autistic adolescent-parent dyads, we examine how Autiverse supports autistic adolescents to organize their daily experience and emotion. Our findings show Autiverse scaffolded adolescents' coherent narratives, while enabling parents to learn additional details of their child's events and emotions. Moreover, the customized AI peer created a comfortable space for sharing, fostering enjoyment and a strong sense of agency. Drawing on these results, we discuss implications for adaptive scaffolding across autism profiles, socio-emotionally appropriate AI peer design, and balancing autonomy with parental involvement.
>
---
#### [replaced 127] The PIMMUR Principles: Ensuring Validity in Collective Behavior of LLM Societies
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于AI社会模拟研究，旨在解决LLM模拟人类集体行为的有效性问题。通过分析6大缺陷（PIMMUR），发现多数研究存在方法缺陷，导致结果不可靠。**

- **链接: [https://arxiv.org/pdf/2509.18052v2](https://arxiv.org/pdf/2509.18052v2)**

> **作者:** Jiaxu Zhou; Jen-tse Huang; Xuhui Zhou; Man Ho Lam; Xintao Wang; Hao Zhu; Wenxuan Wang; Maarten Sap
>
> **备注:** 13 pages, 9 figures, 3 tables
>
> **摘要:** Large language models (LLMs) are increasingly deployed to simulate human collective behaviors, yet the methodological rigor of these "AI societies" remains under-explored. Through a systematic audit of 42 recent studies, we identify six pervasive flaws-spanning agent profiles, interaction, memory, control, unawareness, and realism (PIMMUR). Our analysis reveals that 90.7% of studies violate at least one principle, undermining simulation validity. We demonstrate that frontier LLMs correctly identify the underlying social experiment in 47.6% of cases, while 65.3% of prompts exert excessive control that pre-determines outcomes. By reproducing five representative experiments (e.g., telephone game), we show that reported collective phenomena often vanish or reverse when PIMMUR principles are enforced, suggesting that many "emergent" behaviors are methodological artifacts rather than genuine social dynamics. Our findings suggest that current AI simulations may capture model-specific biases rather than universal human social behaviors, raising critical concerns about the use of LLMs as scientific proxies for human society.
>
---
#### [replaced 128] Kad: A Framework for Proxy-based Test-time Alignment with Knapsack Approximation Deferral
- **分类: cs.CL**

- **简介: 该论文属于大语言模型对齐任务，旨在降低对齐成本。通过代理模型引导，将令牌级决策转化为0-1背包问题，提出近似解法以提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2510.27017v2](https://arxiv.org/pdf/2510.27017v2)**

> **作者:** Ayoub Hammal; Pierre Zweigenbaum; Caio Corro
>
> **备注:** EACL 2026 main
>
> **摘要:** Several previous works concluded that the largest part of generation capabilities of large language models (LLM) are learned (early) during pre-training. However, LLMs still require further alignment to adhere to downstream task requirements and stylistic preferences, among other desired properties. As LLMs continue to scale in terms of size, the computational cost of alignment procedures increase prohibitively. In this work, we propose a novel approach to circumvent these costs via proxy-based test-time alignment, i.e. using guidance from a small aligned model. Our approach can be described as a token-specific cascading method, where the token-specific deferral rule is reduced to 0-1 knapsack problem. In this setting, we derive primal and dual approximations of the optimal deferral decision. We experimentally show the benefits of our method both in task performance and speculative decoding speed.
>
---
#### [replaced 129] JiraiBench: A Bilingual Benchmark for Evaluating Large Language Models' Detection of Human Self-Destructive Behavior Content in Jirai Community
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出JiraiBench，用于评估大语言模型在中日社交媒体中检测自毁行为内容的能力。解决多语言内容审核问题，通过构建双语数据集并分析模型表现。**

- **链接: [https://arxiv.org/pdf/2503.21679v3](https://arxiv.org/pdf/2503.21679v3)**

> **作者:** Yunze Xiao; Tingyu He; Lionel Z. Wang; Yiming Ma; Xingyu Song; Xiaohang Xu; Mona Diab; Irene Li; Ka Chung Ng
>
> **备注:** 20 pages, 1 figures
>
> **摘要:** This paper introduces JiraiBench, the first bilingual benchmark for evaluating large language models' effectiveness in detecting self-destructive content across Chinese and Japanese social media communities. Focusing on the transnational "Jirai" (landmine) online subculture that encompasses multiple forms of self-destructive behaviors including drug overdose, eating disorders, and self-harm, we present a comprehensive evaluation framework incorporating both linguistic and cultural dimensions. Our dataset comprises 10,419 Chinese posts and 5,000 Japanese posts with multidimensional annotation along three behavioral categories, achieving substantial inter-annotator agreement. Experimental evaluations across four state-of-the-art models reveal significant performance variations based on instructional language, with Japanese prompts unexpectedly outperforming Chinese prompts when processing Chinese content. This emergent cross-cultural transfer suggests that cultural proximity can sometimes outweigh linguistic similarity in detection tasks. Cross-lingual transfer experiments with fine-tuned models further demonstrate the potential for knowledge transfer between these language systems without explicit target language training. These findings highlight the need for culturally-informed approaches to multilingual content moderation and provide empirical evidence for the importance of cultural context in developing more effective detection systems for vulnerable online communities.
>
---
#### [replaced 130] FedMentor: Domain-Aware Differential Privacy for Heterogeneous Federated LLMs in Mental Health
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于隐私保护的联邦学习任务，旨在解决敏感领域（如心理健康）中大语言模型的隐私与性能平衡问题。工作包括提出FedMentor框架，结合LoRA和领域感知差分隐私，实现安全高效的模型微调。**

- **链接: [https://arxiv.org/pdf/2509.14275v3](https://arxiv.org/pdf/2509.14275v3)**

> **作者:** Nobin Sarwar; Shubhashis Roy Dipta
>
> **备注:** NeurIPS 2025 GenAI4Health Workshop
>
> **摘要:** Privacy-preserving adaptation of Large Language Models (LLMs) in sensitive domains (e.g., mental health) requires balancing strict confidentiality with model utility and safety. We propose FedMentor, a federated fine-tuning framework that integrates Low-Rank Adaptation (LoRA) and domain-aware Differential Privacy (DP) to meet per-domain privacy budgets while maintaining performance. Each client (domain) applies a custom DP noise scale proportional to its data sensitivity, and the server adaptively reduces noise when utility falls below a threshold. In experiments on three mental health datasets, we show that FedMentor improves safety over standard Federated Learning (FL) without privacy, raising safe output rates by up to three points and lowering toxicity, while maintaining utility (BERTScore F1 and ROUGE-L) within 0.5% of the non-private baseline and close to the centralized upper bound. The framework scales to backbones with up to 1.7B parameters on single-GPU clients, requiring < 173 MB of communication per-round. FedMentor demonstrates a practical approach to privately fine-tune LLMs for safer deployments in healthcare and other sensitive fields.
>
---
#### [replaced 131] When and How Unlabeled Data Provably Improve In-Context Learning
- **分类: cs.LG; cs.AI; cs.CL; math.OC**

- **简介: 该论文研究半监督学习中的上下文学习，探讨未标记数据如何提升模型性能。通过理论分析，提出多层Transformer可有效利用未标记数据，提升半监督学习效果。**

- **链接: [https://arxiv.org/pdf/2506.15329v2](https://arxiv.org/pdf/2506.15329v2)**

> **作者:** Yingcong Li; Xiangyu Chang; Muti Kara; Xiaofeng Liu; Amit Roy-Chowdhury; Samet Oymak
>
> **摘要:** Recent research shows that in-context learning (ICL) can be effective even when demonstrations have missing or incorrect labels. To shed light on this capability, we examine a canonical setting where the demonstrations are drawn according to a binary Gaussian mixture model (GMM) and a certain fraction of the demonstrations have missing labels. We provide a comprehensive theoretical study to show that: (1) The loss landscape of one-layer linear attention models recover the optimal fully-supervised estimator but completely fail to exploit unlabeled data; (2) In contrast, multilayer or looped transformers can effectively leverage unlabeled data by implicitly constructing estimators of the form $\sum_{i\ge 0} a_i (X^\top X)^iX^\top y$ with $X$ and $y$ denoting features and partially-observed labels (with missing entries set to zero). We characterize the class of polynomials that can be expressed as a function of depth and draw connections to Expectation Maximization, an iterative pseudo-labeling algorithm commonly used in semi-supervised learning. Importantly, the leading polynomial power is exponential in depth, so mild amount of depth/looping suffices. As an application of theory, we propose looping off-the-shelf tabular foundation models to enhance their semi-supervision capabilities. Extensive evaluations on real-world datasets show that our method significantly improves the semisupervised tabular learning performance over the standard single pass inference.
>
---
#### [replaced 132] Not Your Typical Sycophant: The Elusive Nature of Sycophancy in Large Language Models
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于模型评估任务，旨在解决LLMs中谄媚行为的量化问题。通过构建零和博弈框架，评估模型在特定情境下的谄媚倾向及其与近期偏见的相互作用。**

- **链接: [https://arxiv.org/pdf/2601.15436v2](https://arxiv.org/pdf/2601.15436v2)**

> **作者:** Shahar Ben Natan; Oren Tsur
>
> **摘要:** We propose a novel way to evaluate sycophancy of LLMs in a direct and neutral way, mitigating various forms of uncontrolled bias, noise, or manipulative language, deliberately injected to prompts in prior works. A key novelty in our approach is the use of LLM-as-a-judge, evaluation of sycophancy as a zero-sum game in a bet setting. Under this framework, sycophancy serves one individual (the user) while explicitly incurring cost on another. Comparing four leading models - Gemini 2.5 Pro, ChatGpt 4o, Mistral-Large-Instruct-2411, and Claude Sonnet 3.7 - we find that while all models exhibit sycophantic tendencies in the common setting, in which sycophancy is self-serving to the user and incurs no cost on others, Claude and Mistral exhibit "moral remorse" and over-compensate for their sycophancy in case it explicitly harms a third party. Additionally, we observed that all models are biased toward the answer proposed last. Crucially, we find that these two phenomena are not independent; sycophancy and recency bias interact to produce `constructive interference' effect, where the tendency to agree with the user is exacerbated when the user's opinion is presented last.
>
---
#### [replaced 133] BILLY: Steering Large Language Models via Merging Persona Vectors for Creative Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BILLY框架，解决多大模型系统计算成本高、延迟大的问题。通过融合人格向量提升单模型创造力，实现多视角生成。**

- **链接: [https://arxiv.org/pdf/2510.10157v2](https://arxiv.org/pdf/2510.10157v2)**

> **作者:** Tsung-Min Pai; Jui-I Wang; Li-Chun Lu; Shao-Hua Sun; Hung-Yi Lee; Kai-Wei Chang
>
> **摘要:** Multi-LLM systems enhance the creativity of large language models by simulating human collective intelligence but suffer from significant drawbacks, such as high computational costs and inference latency. To address these limitations, we propose BILLY (BlendIng persona vectors for Large Language model creativitY), a training-free framework that captures the benefits of multi-LLM collaboration, i.e. inducing diverse perspectives and specialized expertise, within a single model. BILLY operates by extracting and blending multiple distinct persona vectors directly in the model's activation space. We steer the model's generation process with this merged vector while inference, enabling multi-perspective output without explicit multi-LLM communication. Our experiments across creativity-oriented benchmarks demonstrate that BILLY surpasses single model prompting and traditional multi-LLM approaches, while substantially reducing inference time and computational costs. Our analyses further reveal that distinct persona vectors can be blended to achieve both effective control over complementary aspects of generation and greater interpretability.
>
---
#### [replaced 134] CliniBench: A Clinical Outcome Prediction Benchmark for Generative and Encoder-Based Language Models
- **分类: cs.CL**

- **简介: 论文提出CliniBench基准，用于评估生成模型和编码器模型在临床诊断预测中的表现。旨在解决医疗任务中模型有效性验证问题，通过对比实验发现编码器模型更优。**

- **链接: [https://arxiv.org/pdf/2509.26136v2](https://arxiv.org/pdf/2509.26136v2)**

> **作者:** Paul Grundmann; Dennis Fast; Jan Frick; Thomas Steffek; Felix Gers; Wolfgang Nejdl; Alexander Löser
>
> **摘要:** With their growing capabilities, generative large language models (LLMs) are being increasingly investigated for complex medical tasks. However, their effectiveness in real-world clinical applications remains underexplored. To address this, we present CliniBench, the first benchmark that enables comparability of well-studied encoder-based classifiers and generative LLMs for discharge diagnosis prediction from admission notes in MIMIC-IV dataset. Our extensive study compares 12 generative LLMs and 3 encoder-based classifiers and demonstrates that encoder-based classifiers consistently outperform generative models in diagnosis prediction. We assess several retrieval augmentation strategies for in-context learning from similar patients and find that they provide notable performance improvements for generative LLMs.
>
---
#### [replaced 135] Training Tensor Attention Efficiently: From Cubic to Almost Linear Time
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于Transformer优化任务，解决tensor attention训练时间复杂度高的问题，通过理论分析与算法设计，将训练时间从立方级降至接近线性。**

- **链接: [https://arxiv.org/pdf/2405.16411v3](https://arxiv.org/pdf/2405.16411v3)**

> **作者:** Yang Cao; Yingyu Liang; Zhenmei Shi; Zhao Song
>
> **摘要:** Tensor Attention, a multi-view attention that is able to capture high-order correlations among multiple modalities, can overcome the representational limitations of classical matrix attention. However, the $O(n^3)$ time complexity of tensor attention poses a significant obstacle to its utilization in transformers, where $n$ is the input sequence length. In this work, we prove that the backward gradient of tensor attention training can be computed in almost linear time $n^{1+o(1)}$, the same complexity as its forward computation under the bounded entries assumption. We provide a closed-form solution for the gradient and propose a fast computation method utilizing polynomial approximation methods and tensor algebraic techniques. Furthermore, we prove the necessity and tightness of our assumption through hardness analysis, showing that slightly weakening it renders the gradient problem unsolvable in truly subcubic time. Our theoretical results establish the feasibility of efficient higher-order transformer training and may facilitate practical applications of tensor attention architectures.
>
---
#### [replaced 136] When Sharpening Becomes Collapse: Sampling Bias and Semantic Coupling in RL with Verifiable Rewards
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究RLVR中的过拟合问题，旨在解决策略坍塌现象。通过分析采样偏差与语义耦合，提出两种方法提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.15609v2](https://arxiv.org/pdf/2601.15609v2)**

> **作者:** Mingyuan Fan; Weiguang Han; Daixin Wang; Cen Chen; Zhiqiang Zhang; Jun Zhou
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) is a central paradigm for turning large language models (LLMs) into reliable problem solvers, especially in logic-heavy domains. Despite its empirical success, it remains unclear whether RLVR elicits novel capabilities or merely sharpens the distribution over existing knowledge. We study this by formalizing over-sharpening, a phenomenon where the policy collapses onto limited modes, suppressing valid alternatives. At a high level, we discover finite-batch updates intrinsically bias learning toward sampled modes, triggering a collapse that propagates globally via semantic coupling. To mitigate this, we propose inverse-success advantage calibration to prioritize difficult queries and distribution-level calibration to diversify sampling via a memory network. Empirical evaluations validate that our strategies can effectively improve generalization.
>
---
#### [replaced 137] MathMist: A Parallel Multilingual Benchmark Dataset for Mathematical Problem Solving and Reasoning
- **分类: cs.CL**

- **简介: 该论文提出MATHMIST，一个用于数学问题解决和推理的多语言基准数据集，旨在填补低资源语言数学推理研究的空白。**

- **链接: [https://arxiv.org/pdf/2510.14305v2](https://arxiv.org/pdf/2510.14305v2)**

> **作者:** Mahbub E Sobhani; Md. Faiyaz Abdullah Sayeedi; Tasnim Mohiuddin; Md Mofijul Islam; Swakkhar Shatabda
>
> **备注:** Accepted for publication in Findings of EACL 2026
>
> **摘要:** Mathematical reasoning remains one of the most challenging domains for large language models (LLMs), requiring not only linguistic understanding but also structured logical deduction and numerical precision. While recent LLMs demonstrate strong general-purpose reasoning abilities, their mathematical competence across diverse languages remains underexplored. Existing benchmarks primarily focus on English or a narrow subset of high-resource languages, leaving significant gaps in assessing multilingual and cross-lingual mathematical reasoning. To address this, we introduce MATHMIST, a parallel multilingual benchmark for mathematical problem solving and reasoning. MATHMIST encompasses 2,890 parallel Bangla-English gold standard artifacts, totaling approximately 30K aligned question--answer pairs across thirteen languages, representing an extensive coverage of high-, medium-, and low-resource linguistic settings. The dataset captures linguistic variety, multiple types of problem settings, and solution synthesizing capabilities. We systematically evaluate a diverse suite of models, including open-source small and medium LLMs, proprietary systems, and multilingual-reasoning-focused models under zero-shot, chain-of-thought (CoT), perturbated reasoning, and code-switched reasoning paradigms. Our results reveal persistent deficiencies in LLMs' ability to perform consistent and interpretable mathematical reasoning across languages, with pronounced degradation in low-resource settings. All the codes and data are available at GitHub: https://github.com/mahbubhimel/MathMist
>
---
#### [replaced 138] neuralFOMO: Can LLMs Handle Being Second Best? Measuring Envy-Like Preferences in Multi-Agent Settings
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于多智能体系统研究，探讨LLMs在社交比较中是否表现出类似嫉妒的行为。通过实验分析模型在不同情境下的竞争倾向，为多智能体LLM的设计与安全提供参考。**

- **链接: [https://arxiv.org/pdf/2512.13481v2](https://arxiv.org/pdf/2512.13481v2)**

> **作者:** Arnav Ramamoorthy; Shrey Dhorajiya; Ojas Pungalia; Rashi Upadhyay; Abhishek Mishra; Abhiram H; Tejasvi Alladi; Sujan Yenuganti; Dhruv Kumar
>
> **备注:** Under Review
>
> **摘要:** Envy shapes competitiveness and cooperation in human groups, yet its role in large language model interactions remains largely unexplored. As LLMs increasingly operate in multi-agent settings, it is important to examine whether they exhibit envy-like preferences under social comparison. We evaluate LLM behavior across two scenarios: (1) a point-allocation game testing sensitivity to relative versus absolute payoff, and (2) comparative evaluations across general and contextual settings. To ground our analysis in psychological theory, we adapt four established psychometric questionnaires spanning general, domain-specific, workplace, and sibling-based envy. Our results reveal heterogeneous envy-like patterns across models and contexts, with some models sacrificing personal gain to reduce a peer's advantage, while others prioritize individual maximization. These findings highlight competitive dispositions as a design and safety consideration for multi-agent LLM systems.
>
---
#### [replaced 139] SymCode: A Neurosymbolic Approach to Mathematical Reasoning via Verifiable Code Generation
- **分类: cs.CL; cs.PL**

- **简介: 该论文提出SymCode，解决数学推理中LLM生成结果不可靠的问题，通过可验证的代码生成提升准确性。属于数学推理任务。**

- **链接: [https://arxiv.org/pdf/2510.25975v2](https://arxiv.org/pdf/2510.25975v2)**

> **作者:** Sina Bagheri Nezhad; Yao Li; Ameeta Agrawal
>
> **备注:** camera-ready EACL 2026 Findings
>
> **摘要:** Large Language Models (LLMs) often struggle with complex mathematical reasoning, where prose-based generation leads to unverified and arithmetically unsound solutions. Current prompting strategies like Chain of Thought still operate within this unreliable medium, lacking a mechanism for deterministic verification. To address these limitations, we introduce SymCode, a neurosymbolic framework that reframes mathematical problem-solving as a task of verifiable code generation using the SymPy library. We evaluate SymCode on challenging benchmarks, including MATH-500 and OlympiadBench, demonstrating significant accuracy improvements of up to 13.6 percentage points over baselines. Our analysis shows that SymCode is not only more token-efficient but also fundamentally shifts model failures from opaque logical fallacies towards transparent, programmatic errors. By grounding LLM reasoning in a deterministic symbolic engine, SymCode represents a key step towards more accurate and trustworthy AI in formal domains.
>
---
#### [replaced 140] ToxSearch: Evolving Prompts for Toxicity Search in Large Language Models
- **分类: cs.NE; cs.AI; cs.CL**

- **简介: 该论文属于模型安全任务，旨在检测大语言模型中的毒性内容。通过进化算法生成对抗性提示，测试模型安全性并评估跨模型迁移效果。**

- **链接: [https://arxiv.org/pdf/2511.12487v2](https://arxiv.org/pdf/2511.12487v2)**

> **作者:** Onkar Shelar; Travis Desell
>
> **备注:** 16 pages
>
> **摘要:** Large Language Models remain vulnerable to adversarial prompts that elicit toxic content even after safety alignment. We present ToxSearch, a black-box evolutionary framework that tests model safety by evolving prompts in a synchronous steady-state loop. The system employs a diverse set of operators, including lexical substitutions, negation, back-translation, paraphrasing, and two semantic crossover operators, while a moderation oracle provides fitness guidance. Operator-level analysis shows heterogeneous behavior: lexical substitutions offer the best yield-variance trade-off, semantic-similarity crossover acts as a precise low-throughput inserter, and global rewrites exhibit high variance with elevated refusal costs. Using elite prompts evolved on LLaMA 3.1 8B, we observe practically meaningful but attenuated cross-model transfer, with toxicity roughly halving on most targets, smaller LLaMA 3.2 variants showing the strongest resistance, and some cross-architecture models retaining higher toxicity. These results suggest that small, controllable perturbations are effective vehicles for systematic red-teaming and that defenses should anticipate cross-model reuse of adversarial prompts rather than focusing only on single-model hardening.
>
---
#### [replaced 141] Stop Taking Tokenizers for Granted: They Are Core Design Decisions in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理领域，探讨tokenization在大模型中的核心作用。旨在解决tokenization设计不统一、缺乏理论支持的问题，提出需将其作为核心设计决策，并进行模型与分词器协同优化。**

- **链接: [https://arxiv.org/pdf/2601.13260v2](https://arxiv.org/pdf/2601.13260v2)**

> **作者:** Sawsan Alqahtani; Mir Tafseer Nayeem; Md Tahmid Rahman Laskar; Tasnim Mohiuddin; M Saiful Bari
>
> **备注:** Accepted to EACL 2026 (long, main). The first two authors contributed equally
>
> **摘要:** Tokenization underlies every large language model, yet it remains an under-theorized and inconsistently designed component. Common subword approaches such as Byte Pair Encoding (BPE) offer scalability but often misalign with linguistic structure, amplify bias, and waste capacity across languages and domains. This paper reframes tokenization as a core modeling decision rather than a preprocessing step. We argue for a context-aware framework that integrates tokenizer and model co-design, guided by linguistic, domain, and deployment considerations. Standardized evaluation and transparent reporting are essential to make tokenization choices accountable and comparable. Treating tokenization as a core design problem, not a technical afterthought, can yield language technologies that are fairer, more efficient, and more adaptable.
>
---
#### [replaced 142] Exploring the Effects of Alignment on Numerical Bias in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLM评估中的数值偏差问题。通过对比对齐前后模型，发现对齐加剧了偏差，并提出调整评分范围作为有效缓解方法。**

- **链接: [https://arxiv.org/pdf/2601.16444v2](https://arxiv.org/pdf/2601.16444v2)**

> **作者:** Ayako Sato; Hwichan Kim; Zhousi Chen; Masato Mita; Mamoru Komachi
>
> **备注:** Accepted at AIBSD 2026 (Workshop at AAAI 2026)
>
> **摘要:** "LLM-as-a-judge," which utilizes large language models (LLMs) as evaluators, has proven effective in many evaluation tasks. However, evaluator LLMs exhibit numerical bias, a phenomenon where certain evaluation scores are generated disproportionately often, leading reduced evaluation performance. This study investigates the cause of this bias. Given that most evaluator LLMs are aligned through instruction tuning and preference tuning, and that prior research suggests alignment reduces output diversity, we hypothesize that numerical bias arises from alignment. To test this, we compare outputs from pre- and post-alignment LLMs, and observe that alignment indeed increases numerical bias. We also explore mitigation strategies for post-alignment LLMs, including temperature scaling, distribution calibration, and score range adjustment. Among these, score range adjustment is most effective in reducing bias and improving performance, though still heuristic. Our findings highlight the need for further work on optimal score range selection and more robust mitigation strategies.
>
---
#### [replaced 143] Far from the Shallow: Brain-Predictive Reasoning Embedding through Residual Disentanglement
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文属于神经语言学任务，旨在解决语言处理中认知层次混淆的问题。通过残差解耦方法分离语言特征，提取独立的推理嵌入，揭示其在脑活动中的独特作用。**

- **链接: [https://arxiv.org/pdf/2510.22860v3](https://arxiv.org/pdf/2510.22860v3)**

> **作者:** Linyang He; Tianjun Zhong; Richard Antonello; Gavin Mischler; Micah Goldblum; Nima Mesgarani
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Understanding how the human brain progresses from processing simple linguistic inputs to performing high-level reasoning is a fundamental challenge in neuroscience. While modern large language models (LLMs) are increasingly used to model neural responses to language, their internal representations are highly "entangled," mixing information about lexicon, syntax, meaning, and reasoning. This entanglement biases conventional brain encoding analyses toward linguistically shallow features (e.g., lexicon and syntax), making it difficult to isolate the neural substrates of cognitively deeper processes. Here, we introduce a residual disentanglement method that computationally isolates these components. By first probing an LM to identify feature-specific layers, our method iteratively regresses out lower-level representations to produce four nearly orthogonal embeddings for lexicon, syntax, meaning, and, critically, reasoning. We used these disentangled embeddings to model intracranial (ECoG) brain recordings from neurosurgical patients listening to natural speech. We show that: 1) This isolated reasoning embedding exhibits unique predictive power, accounting for variance in neural activity not explained by other linguistic features and even extending to the recruitment of visual regions beyond classical language areas. 2) The neural signature for reasoning is temporally distinct, peaking later (~350-400ms) than signals related to lexicon, syntax, and meaning, consistent with its position atop a processing hierarchy. 3) Standard, non-disentangled LLM embeddings can be misleading, as their predictive success is primarily attributable to linguistically shallow features, masking the more subtle contributions of deeper cognitive processing.
>
---
#### [replaced 144] Surprisal and Metaphor Novelty Judgments: Moderate Correlations and Divergent Scaling Effects Revealed by Corpus-Based and Synthetic Datasets
- **分类: cs.CL; cs.AI; cs.IT**

- **简介: 该论文研究语言模型中隐喻新颖性的判断任务，探讨 surprisal 与隐喻新颖性标注的相关性。通过分析真实和合成数据集，发现模型规模对相关性有不同影响。**

- **链接: [https://arxiv.org/pdf/2601.02015v3](https://arxiv.org/pdf/2601.02015v3)**

> **作者:** Omar Momen; Emilie Sitter; Berenike Herrmann; Sina Zarrieß
>
> **备注:** to be published at EACL 2026 main conference
>
> **摘要:** Novel metaphor comprehension involves complex semantic processes and linguistic creativity, making it an interesting task for studying language models (LMs). This study investigates whether surprisal, a probabilistic measure of predictability in LMs, correlates with annotations of metaphor novelty in different datasets. We analyse the surprisal of metaphoric words in corpus-based and synthetic metaphor datasets using 16 causal LM variants. We propose a cloze-style surprisal method that conditions on full-sentence context. Results show that LM surprisal yields significant moderate correlations with scores/labels of metaphor novelty. We further identify divergent scaling patterns: on corpus-based data, correlation strength decreases with model size (inverse scaling effect), whereas on synthetic data it increases (quality-power hypothesis). We conclude that while surprisal can partially account for annotations of metaphor novelty, it remains limited as a metric of linguistic creativity. Code and data are publicly available: https://github.com/OmarMomen14/surprisal-metaphor-novelty
>
---
