# 自然语言处理 cs.CL

- **最新发布 40 篇**

- **更新 38 篇**

## 最新发布

#### [new 001] CLINIC: Evaluating Multilingual Trustworthiness in Language Models for Healthcare
- **分类: cs.CL**

- **简介: 该论文提出CLINIC，一个面向医疗领域多语言可信度评估的基准。针对现有模型在低资源语言中可信度不足的问题，系统评估了语言模型在真实性、公平性、安全性、鲁棒性和隐私性五个维度的表现，覆盖15种语言和18项任务，揭示了当前模型在多语言医疗应用中的关键缺陷。**

- **链接: [https://arxiv.org/pdf/2512.11437v1](https://arxiv.org/pdf/2512.11437v1)**

> **作者:** Akash Ghosh; Srivarshinee Sridhar; Raghav Kaushik Ravi; Muhsin Muhsin; Sriparna Saha; Chirag Agarwal
>
> **备注:** 49 pages, 31 figures
>
> **摘要:** Integrating language models (LMs) in healthcare systems holds great promise for improving medical workflows and decision-making. However, a critical barrier to their real-world adoption is the lack of reliable evaluation of their trustworthiness, especially in multilingual healthcare settings. Existing LMs are predominantly trained in high-resource languages, making them ill-equipped to handle the complexity and diversity of healthcare queries in mid- and low-resource languages, posing significant challenges for deploying them in global healthcare contexts where linguistic diversity is key. In this work, we present CLINIC, a Comprehensive Multilingual Benchmark to evaluate the trustworthiness of language models in healthcare. CLINIC systematically benchmarks LMs across five key dimensions of trustworthiness: truthfulness, fairness, safety, robustness, and privacy, operationalized through 18 diverse tasks, spanning 15 languages (covering all the major continents), and encompassing a wide array of critical healthcare topics like disease conditions, preventive actions, diagnostic tests, treatments, surgeries, and medications. Our extensive evaluation reveals that LMs struggle with factual correctness, demonstrate bias across demographic and linguistic groups, and are susceptible to privacy breaches and adversarial attacks. By highlighting these shortcomings, CLINIC lays the foundation for enhancing the global reach and safety of LMs in healthcare across diverse languages.
>
---
#### [new 002] CIP: A Plug-and-Play Causal Prompting Framework for Mitigating Hallucinations under Long-Context Noise
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型在长而噪声的上下文中易产生幻觉的问题，提出CIP因果提示框架。通过构建因果序列并注入提示，抑制虚假相关，提升事实性和推理效率，属可信语言模型任务。**

- **链接: [https://arxiv.org/pdf/2512.11282v1](https://arxiv.org/pdf/2512.11282v1)**

> **作者:** Qingsen Ma; Dianyun Wang; Ran Jing; Yujun Sun; Zhenbo Xu
>
> **摘要:** Large language models often hallucinate when processing long and noisy retrieval contexts because they rely on spurious correlations rather than genuine causal relationships. We propose CIP, a lightweight and plug-and-play causal prompting framework that mitigates hallucinations at the input stage. CIP constructs a causal relation sequence among entities, actions, and events and injects it into the prompt to guide reasoning toward causally relevant evidence. Through causal intervention and counterfactual reasoning, CIP suppresses non causal reasoning paths, improving factual grounding and interpretability. Experiments across seven mainstream language models, including GPT-4o, Gemini 2.0 Flash, and Llama 3.1, show that CIP consistently enhances reasoning quality and reliability, achieving 2.6 points improvement in Attributable Rate, 0.38 improvement in Causal Consistency Score, and a fourfold increase in effective information density. API level profiling further shows that CIP accelerates contextual understanding and reduces end to end response latency by up to 55.1 percent. These results suggest that causal reasoning may serve as a promising paradigm for improving the explainability, stability, and efficiency of large language models.
>
---
#### [new 003] Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究减少大模型幻觉的技术对创造力的影响，旨在平衡科学发现中事实准确性与创造性。通过评估CoVe、DoLa和RAG在多模型上的表现，发现不同方法对发散性思维有正负或中性影响。**

- **链接: [https://arxiv.org/pdf/2512.11509v1](https://arxiv.org/pdf/2512.11509v1)**

> **作者:** Mohor Banerjee; Nadya Yuki Wangsajaya; Syed Ali Redha Alsagoff; Min Sen Tan; Zachary Choy Kit Chun; Alvin Chan Guo Wei
>
> **摘要:** Large Language Models (LLMs) exhibit remarkable capabilities in natural language understanding and reasoning, but suffer from hallucination: the generation of factually incorrect content. While numerous methods have been developed to reduce hallucinations, their impact on creative generations remains unexplored. This gap is particularly critical for AI-assisted scientific discovery, which requires both factual accuracy and creative hypothesis generation. We investigate how three hallucination-reduction techniques: Chain of Verification (CoVe), Decoding by Contrasting Layers (DoLa), and Retrieval-Augmented Generation (RAG), affect creativity in LLMs. Evaluating multiple model families (LLaMA, Qwen, Mistral) at varying scales (1B - 70B parameters) on two creativity benchmarks (NeoCoder and CS4), we find that these methods have opposing effects on divergent creativity. CoVe enhances divergent thinking, DoLa suppresses it, and RAG shows minimal impact. Our findings provide guidance for selecting appropriate hallucination-reduction methods in scientific applications, where the balance between factual accuracy and creative exploration is crucial.
>
---
#### [new 004] qa-FLoRA: Data-free query-adaptive Fusion of LoRAs for LLMs
- **分类: cs.CL**

- **简介: 该论文研究大型语言模型中LoRA适配器的融合任务，旨在解决多领域复合查询下静态融合效果差、动态融合依赖训练数据的问题。提出qa-FLoRA方法，无需数据和训练，通过度量分布差异自适应计算层级别融合权重，实现高效多域适应。**

- **链接: [https://arxiv.org/pdf/2512.11366v1](https://arxiv.org/pdf/2512.11366v1)**

> **作者:** Shreya Shukla; Aditya Sriram; Milinda Kuppur Narayanaswamy; Hiteshi Jain
>
> **备注:** Accepted at AAAI 2026 (Main Technical Track)
>
> **摘要:** The deployment of large language models for specialized tasks often requires domain-specific parameter-efficient finetuning through Low-Rank Adaptation (LoRA) modules. However, effectively fusing these adapters to handle complex, multi-domain composite queries remains a critical challenge. Existing LoRA fusion approaches either use static weights, which assign equal relevance to each participating LoRA, or require data-intensive supervised training for every possible LoRA combination to obtain respective optimal fusion weights. We propose qa-FLoRA, a novel query-adaptive data-and-training-free method for LoRA fusion that dynamically computes layer-level fusion weights by measuring distributional divergence between the base model and respective adapters. Our approach eliminates the need for composite training data or domain-representative samples, making it readily applicable to existing adapter collections. Extensive experiments across nine multilingual composite tasks spanning mathematics, coding, and medical domains, show that qa-FLoRA outperforms static fusion by ~5% with LLaMA-2 and ~6% with LLaMA-3, and the training-free baselines by ~7% with LLaMA-2 and ~10% with LLaMA-3, while significantly closing the gap with supervised baselines. Further, layer-level analysis of our fusion weights reveals interpretable fusion patterns, demonstrating the effectiveness of our approach for robust multi-domain adaptation.
>
---
#### [new 005] Automating Historical Insight Extraction from Large-Scale Newspaper Archives via Neural Topic Modeling
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属文本挖掘任务，旨在解决传统方法难以捕捉历史报刊中动态话题演变的问题。作者采用BERTopic模型，分析1955–2018年核能与核安全 discourse 的主题演化，揭示公众讨论的长期趋势与主题关联变化，展现神经话题建模在历史研究中的优势。**

- **链接: [https://arxiv.org/pdf/2512.11635v1](https://arxiv.org/pdf/2512.11635v1)**

> **作者:** Keerthana Murugaraj; Salima Lamsiyah; Marten During; Martin Theobald
>
> **备注:** This is a preprint of a manuscript submitted to Digital Scholarship in the Humanities (Oxford University Press). The paper is currently under peer review
>
> **摘要:** Extracting coherent and human-understandable themes from large collections of unstructured historical newspaper archives presents significant challenges due to topic evolution, Optical Character Recognition (OCR) noise, and the sheer volume of text. Traditional topic-modeling methods, such as Latent Dirichlet Allocation (LDA), often fall short in capturing the complexity and dynamic nature of discourse in historical texts. To address these limitations, we employ BERTopic. This neural topic-modeling approach leverages transformerbased embeddings to extract and classify topics, which, despite its growing popularity, still remains underused in historical research. Our study focuses on articles published between 1955 and 2018, specifically examining discourse on nuclear power and nuclear safety. We analyze various topic distributions across the corpus and trace their temporal evolution to uncover long-term trends and shifts in public discourse. This enables us to more accurately explore patterns in public discourse, including the co-occurrence of themes related to nuclear power and nuclear weapons and their shifts in topic importance over time. Our study demonstrates the scalability and contextual sensitivity of BERTopic as an alternative to traditional approaches, offering richer insights into historical discourses extracted from newspaper archives. These findings contribute to historical, nuclear, and social-science research while reflecting on current limitations and proposing potential directions for future work.
>
---
#### [new 006] Benchmarking Automatic Speech Recognition Models for African Languages
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决非洲语言因资源稀缺导致的模型选择与优化问题。作者系统评测了四种主流ASR模型在13种非洲语言上的表现，分析不同数据规模和解码策略下的性能差异，揭示模型特性与资源条件的相互作用，为低资源语言提供实用设计指导。**

- **链接: [https://arxiv.org/pdf/2512.10968v1](https://arxiv.org/pdf/2512.10968v1)**

> **作者:** Alvin Nahabwe; Sulaiman Kagumire; Denis Musinguzi; Bruno Beijuka; Jonah Mubuuke Kyagaba; Peter Nabende; Andrew Katumba; Joyce Nakatumba-Nabende
>
> **备注:** 19 pages, 8 figures, Deep Learning Indiba, Proceedings of Machine Learning Research
>
> **摘要:** Automatic speech recognition (ASR) for African languages remains constrained by limited labeled data and the lack of systematic guidance on model selection, data scaling, and decoding strategies. Large pre-trained systems such as Whisper, XLS-R, MMS, and W2v-BERT have expanded access to ASR technology, but their comparative behavior in African low-resource contexts has not been studied in a unified and systematic way. In this work, we benchmark four state-of-the-art ASR models across 13 African languages, fine-tuning them on progressively larger subsets of transcribed data ranging from 1 to 400 hours. Beyond reporting error rates, we provide new insights into why models behave differently under varying conditions. We show that MMS and W2v-BERT are more data efficient in very low-resource regimes, XLS-R scales more effectively as additional data becomes available, and Whisper demonstrates advantages in mid-resource conditions. We also analyze where external language model decoding yields improvements and identify cases where it plateaus or introduces additional errors, depending on the alignment between acoustic and text resources. By highlighting the interaction between pre-training coverage, model architecture, dataset domain, and resource availability, this study offers practical and insights into the design of ASR systems for underrepresented languages.
>
---
#### [new 007] LegalRikai: Open Benchmark -- A Benchmark for Complex Japanese Corporate Legal Tasks
- **分类: cs.CL**

- **简介: 该论文提出LegalRikai: Open Benchmark，针对日本企业法律实务构建包含四项复杂任务的评测基准，旨在解决现有短文本评测无法暴露模型在长文档结构化输出与编辑中的缺陷问题，通过人工与自动评估结合，推动法律领域更贴近实践的AI研究。**

- **链接: [https://arxiv.org/pdf/2512.11297v1](https://arxiv.org/pdf/2512.11297v1)**

> **作者:** Shogo Fujita; Yuji Naraki; Yiqing Zhu; Shinsuke Mori
>
> **摘要:** This paper introduces LegalRikai: Open Benchmark, a new benchmark comprising four complex tasks that emulate Japanese corporate legal practices. The benchmark was created by legal professionals under the supervision of an attorney. This benchmark has 100 samples that require long-form, structured outputs, and we evaluated them against multiple practical criteria. We conducted both human and automated evaluations using leading LLMs, including GPT-5, Gemini 2.5 Pro, and Claude Opus 4.1. Our human evaluation revealed that abstract instructions prompted unnecessary modifications, highlighting model weaknesses in document-level editing that were missed by conventional short-text tasks. Furthermore, our analysis reveals that automated evaluation aligns well with human judgment on criteria with clear linguistic grounding, and assessing structural consistency remains a challenge. The result demonstrates the utility of automated evaluation as a screening tool when expert availability is limited. We propose a dataset evaluation framework to promote more practice-oriented research in the legal domain.
>
---
#### [new 008] Mistake Notebook Learning: Selective Batch-Wise Context Optimization for In-Context Learning
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的上下文学习任务，旨在解决ICL中错误无法有效学习的问题。提出Mistake Notebook Learning（MNL），通过批量错误抽象与动态知识库存储可泛化的纠错策略，在无需训练的前提下实现稳定提升，性能接近监督微调，显著优于现有无训练方法。**

- **链接: [https://arxiv.org/pdf/2512.11485v1](https://arxiv.org/pdf/2512.11485v1)**

> **作者:** Xuanbo Su; Yingfang Zhang; Hao Luo; Xiaoteng Liu; Leo Huang
>
> **摘要:** Large language models (LLMs) adapt to tasks via gradient fine-tuning (heavy computation, catastrophic forgetting) or In-Context Learning (ICL: low robustness, poor mistake learning). To fix this, we introduce Mistake Notebook Learning (MNL), a training-free framework with a persistent knowledge base of abstracted error patterns. Unlike prior instance/single-trajectory memory methods, MNL uses batch-wise error abstraction: it extracts generalizable guidance from multiple failures, stores insights in a dynamic notebook, and retains only baseline-outperforming guidance via hold-out validation (ensuring monotonic improvement). We show MNL nearly matches Supervised Fine-Tuning (93.9% vs 94.3% on GSM8K) and outperforms training-free alternatives on GSM8K, Spider, AIME, and KaggleDBQA. On KaggleDBQA (Qwen3-8B), MNL hits 28% accuracy (47% relative gain), outperforming Memento (15.1%) and Training-Free GRPO (22.1) - proving it's a strong training-free alternative for complex reasoning.
>
---
#### [new 009] ASR Under the Stethoscope: Evaluating Biases in Clinical Speech Recognition across Indian Languages
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文评估印度多语言临床场景中语音识别（ASR）系统的偏见问题，属语音技术公平性研究。针对患者与医生、不同性别及语言群体，测试多种ASR模型在真实临床对话中的表现，揭示跨语言、角色与性别的系统性性能差距，推动医疗ASR的包容性发展。**

- **链接: [https://arxiv.org/pdf/2512.10967v1](https://arxiv.org/pdf/2512.10967v1)**

> **作者:** Subham Kumar; Prakrithi Shivaprakash; Abhishek Manoharan; Astut Kurariya; Diptadhi Mukherjee; Lekhansh Shukla; Animesh Mukherjee; Prabhat Chand; Pratima Murthy
>
> **摘要:** Automatic Speech Recognition (ASR) is increasingly used to document clinical encounters, yet its reliability in multilingual and demographically diverse Indian healthcare contexts remains largely unknown. In this study, we conduct the first systematic audit of ASR performance on real world clinical interview data spanning Kannada, Hindi, and Indian English, comparing leading models including Indic Whisper, Whisper, Sarvam, Google speech to text, Gemma3n, Omnilingual, Vaani, and Gemini. We evaluate transcription accuracy across languages, speakers, and demographic subgroups, with a particular focus on error patterns affecting patients vs. clinicians and gender based or intersectional disparities. Our results reveal substantial variability across models and languages, with some systems performing competitively on Indian English but failing on code mixed or vernacular speech. We also uncover systematic performance gaps tied to speaker role and gender, raising concerns about equitable deployment in clinical settings. By providing a comprehensive multilingual benchmark and fairness analysis, our work highlights the need for culturally and demographically inclusive ASR development for healthcare ecosystem in India.
>
---
#### [new 010] PIAST: Rapid Prompting with In-context Augmentation for Scarce Training data
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的提示工程任务，旨在解决少样本场景下提示设计困难的问题。作者提出PIAST方法，通过蒙特卡洛Shapley估计自动筛选增广的少量示例，提升大模型在数据稀缺下的表现，兼顾效率与效果。**

- **链接: [https://arxiv.org/pdf/2512.11013v1](https://arxiv.org/pdf/2512.11013v1)**

> **作者:** Pawel Batorski; Paul Swoboda
>
> **摘要:** LLMs are highly sensitive to prompt design, but handcrafting effective prompts is difficult and often requires intricate crafting of few-shot examples. We propose a fast automatic prompt construction algorithm that augments human instructions by generating a small set of few shot examples. Our method iteratively replaces/drops/keeps few-shot examples using Monte Carlo Shapley estimation of example utility. For faster execution, we use aggressive subsampling and a replay buffer for faster evaluations. Our method can be run using different compute time budgets. On a limited budget, we outperform existing automatic prompting methods on text simplification and GSM8K and obtain second best results on classification and summarization. With an extended, but still modest compute budget we set a new state of the art among automatic prompting methods on classification, simplification and GSM8K. Our results show that carefully constructed examples, rather than exhaustive instruction search, are the dominant lever for fast and data efficient prompt engineering. Our code is available at https://github.com/Batorskq/PIAST.
>
---
#### [new 011] MultiScript30k: Leveraging Multilingual Embeddings to Extend Cross Script Parallel Data
- **分类: cs.CL; cs.AI; cs.LG; cs.MM**

- **简介: 该论文属多语言机器翻译任务，旨在解决Multi30k数据集语言和脚本多样性不足问题。作者提出MultiScript30k，利用NLLB200-3.3B模型将英文数据扩展至阿拉伯语、西班牙语、乌克兰语及中文（简繁体），并验证其语义一致性。**

- **链接: [https://arxiv.org/pdf/2512.11074v1](https://arxiv.org/pdf/2512.11074v1)**

> **作者:** Christopher Driggers-Ellis; Detravious Brinkley; Ray Chen; Aashish Dhawan; Daisy Zhe Wang; Christan Grant
>
> **备注:** 7 pages, 2 figures, 5 tables. Not published at any conference at this time
>
> **摘要:** Multi30k is frequently cited in the multimodal machine translation (MMT) literature, offering parallel text data for training and fine-tuning deep learning models. However, it is limited to four languages: Czech, English, French, and German. This restriction has led many researchers to focus their investigations only on these languages. As a result, MMT research on diverse languages has been stalled because the official Multi30k dataset only represents European languages in Latin scripts. Previous efforts to extend Multi30k exist, but the list of supported languages, represented language families, and scripts is still very short. To address these issues, we propose MultiScript30k, a new Multi30k dataset extension for global languages in various scripts, created by translating the English version of Multi30k (Multi30k-En) using NLLB200-3.3B. The dataset consists of over \(30000\) sentences and provides translations of all sentences in Multi30k-En into Ar, Es, Uk, Zh\_Hans and Zh\_Hant. Similarity analysis shows that Multi30k extension consistently achieves greater than \(0.8\) cosine similarity and symmetric KL divergence less than \(0.000251\) for all languages supported except Zh\_Hant which is comparable to the previous Multi30k extensions ArEnMulti30k and Multi30k-Uk. COMETKiwi scores reveal mixed assessments of MultiScript30k as a translation of Multi30k-En in comparison to the related work. ArEnMulti30k scores nearly equal MultiScript30k-Ar, but Multi30k-Uk scores $6.4\%$ greater than MultiScript30k-Uk per split.
>
---
#### [new 012] Explanation Bias is a Product: Revealing the Hidden Lexical and Position Preferences in Post-Hoc Feature Attribution
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究后验特征归因方法的解释偏差问题，旨在揭示不同方法在词汇和位置上的隐性偏好。通过模型无关的评估框架，在人工与自然语言任务中系统分析两种Transformer的偏差表现，发现各类方法在两类偏好上存在结构性失衡。**

- **链接: [https://arxiv.org/pdf/2512.11108v1](https://arxiv.org/pdf/2512.11108v1)**

> **作者:** Jonathan Kamp; Roos Bakker; Dominique Blok
>
> **摘要:** Good quality explanations strengthen the understanding of language models and data. Feature attribution methods, such as Integrated Gradient, are a type of post-hoc explainer that can provide token-level insights. However, explanations on the same input may vary greatly due to underlying biases of different methods. Users may be aware of this issue and mistrust their utility, while unaware users may trust them inadequately. In this work, we delve beyond the superficial inconsistencies between attribution methods, structuring their biases through a model- and method-agnostic framework of three evaluation metrics. We systematically assess both the lexical and position bias (what and where in the input) for two transformers; first, in a controlled, pseudo-random classification task on artificial data; then, in a semi-controlled causal relation detection task on natural data. We find that lexical and position biases are structurally unbalanced in our model comparison, with models that score high on one type score low on the other. We also find signs that methods producing anomalous explanations are more likely to be biased themselves.
>
---
#### [new 013] Improving Translation Quality by Selecting Better Data for LLM Fine-Tuning: A Comparative Analysis
- **分类: cs.CL**

- **简介: 该论文研究数据选择对开源大语言模型机器翻译微调的影响，旨在提升翻译质量。通过比较五种数据选择方法，发现语义类方法效果更优，且少量数据差异也会显著影响性能。**

- **链接: [https://arxiv.org/pdf/2512.11388v1](https://arxiv.org/pdf/2512.11388v1)**

> **作者:** Felipe Ribeiro Fujita de Mello; Hideyuki Takada
>
> **备注:** To appear at IEEE Big Data 2025
>
> **摘要:** We investigated the impact of data selection on machine translation fine-tuning for open LLMs. Using Japanese-English corpora, we compare five selectors: TF-IDF, COMET Kiwi, QuRate, FD-Score, and random selection, under controlled training conditions. We observed that semantic selectors consistently outperform lexical and geometry-based heuristics, and that even when the selected data differ by less than 3%, the impact on model performance is substantial, underscoring the sensitivity of fine-tuning to data quality.
>
---
#### [new 014] When Actions Teach You to Think: Reasoning-Action Synergy via Reinforcement Learning in Conversational Agents
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究如何通过强化学习提升对话智能体的推理与行动能力。针对监督微调泛化性差、高质量推理标注难的问题，提出基于奖励信号（工具准确性和答案正确性）的GRPO方法，让模型从任务结果中自主学习推理策略，实现推理与行动协同优化。**

- **链接: [https://arxiv.org/pdf/2512.11277v1](https://arxiv.org/pdf/2512.11277v1)**

> **作者:** Mrinal Rawat; Arkajyoti Chakraborty; Neha Gupta; Roberto Pieraccini
>
> **摘要:** Supervised fine-tuning (SFT) has emerged as one of the most effective ways to improve the performance of large language models (LLMs) in downstream tasks. However, SFT can have difficulty generalizing when the underlying data distribution changes, even when the new data does not fall completely outside the training domain. Recent reasoning-focused models such as o1 and R1 have demonstrated consistent gains over their non-reasoning counterparts, highlighting the importance of reasoning for improved generalization and reliability. However, collecting high-quality reasoning traces for SFT remains challenging -- annotations are costly, subjective, and difficult to scale. To address this limitation, we leverage Reinforcement Learning (RL) to enable models to learn reasoning strategies directly from task outcomes. We propose a pipeline in which LLMs generate reasoning steps that guide both the invocation of tools (e.g., function calls) and the final answer generation for conversational agents. Our method employs Group Relative Policy Optimization (GRPO) with rewards designed around tool accuracy and answer correctness, allowing the model to iteratively refine its reasoning and actions. Experimental results demonstrate that our approach improves both the quality of reasoning and the precision of tool invocations, achieving a 1.5% relative improvement over the SFT model (trained without explicit thinking) and a 40% gain compared to the base of the vanilla Qwen3-1.7B model. These findings demonstrate the promise of unifying reasoning and action learning through RL to build more capable and generalizable conversational agents.
>
---
#### [new 015] SUMFORU: An LLM-Based Review Summarization Framework for Personalized Purchase Decision Support
- **分类: cs.CL**

- **简介: 该论文提出SUMFORU，解决现有评论摘要模型缺乏个性化的问题。属于个性化决策支持任务，通过构建用户画像感知的两阶段对齐框架，实现更符合个人偏好的商品评论摘要生成。**

- **链接: [https://arxiv.org/pdf/2512.11755v1](https://arxiv.org/pdf/2512.11755v1)**

> **作者:** Yuming Feng; Xinrui Jiang
>
> **备注:** Code available at https://github.com/Harry20030331/SumForU
>
> **摘要:** Online product reviews contain rich but noisy signals that overwhelm users and hinder effective decision-making. Existing LLM-based summarizers remain generic and fail to account for individual preferences, limiting their practical utility. We propose SUMFORU, a steerable review summarization framework that aligns outputs with explicit user personas to support personalized purchase decisions. Our approach integrates a high-quality data pipeline built from the Amazon 2023 Review Dataset with a two-stage alignment procedure: (1) persona-aware Supervised Fine-Tuning (SFT) via asymmetric knowledge distillation, and (2) Reinforcement Learning with AI Feedback (RLAIF) using a preference estimator to capture fine-grained, persona-relevant signals. We evaluate the model across rule-based, LLM-based, and human-centered metrics, demonstrating consistent improvements in consistency, grounding, and preference alignment. Our framework achieves the highest performance across all evaluation settings and generalizes effectively to unseen product categories. Our results highlight the promise of steerable pluralistic alignment for building next-generation personalized decision-support systems.
>
---
#### [new 016] Leveraging LLMs for Title and Abstract Screening for Systematic Review: A Cost-Effective Dynamic Few-Shot Learning Approach
- **分类: cs.CL**

- **简介: 该论文针对系统评价中标题与摘要筛选耗时费力的问题，提出一种基于大语言模型的两阶段动态少样本学习方法。通过低成本模型初筛、高精度模型复核低置信度样本，提升效率并控制成本，实验证明其具有良好的通用性与成本效益。**

- **链接: [https://arxiv.org/pdf/2512.11261v1](https://arxiv.org/pdf/2512.11261v1)**

> **作者:** Yun-Chung Liu; Rui Yang; Jonathan Chong Kai Liew; Ziran Yin; Henry Foote; Christopher J. Lindsell; Chuan Hong
>
> **备注:** 22 pages, 3 figures
>
> **摘要:** Systematic reviews are a key component of evidence-based medicine, playing a critical role in synthesizing existing research evidence and guiding clinical decisions. However, with the rapid growth of research publications, conducting systematic reviews has become increasingly burdensome, with title and abstract screening being one of the most time-consuming and resource-intensive steps. To mitigate this issue, we designed a two-stage dynamic few-shot learning (DFSL) approach aimed at improving the efficiency and performance of large language models (LLMs) in the title and abstract screening task. Specifically, this approach first uses a low-cost LLM for initial screening, then re-evaluates low-confidence instances using a high-performance LLM, thereby enhancing screening performance while controlling computational costs. We evaluated this approach across 10 systematic reviews, and the results demonstrate its strong generalizability and cost-effectiveness, with potential to reduce manual screening burden and accelerate the systematic review process in practical applications.
>
---
#### [new 017] FIBER: A Multilingual Evaluation Resource for Factual Inference Bias
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出FIBER，一个多语言评测基准，旨在评估大模型在单/多实体事实推理中的偏差问题。研究考察提示语言对实体选择的影响及多实体推理难度，发现语言和模型规模显著影响性能与偏差，揭示了多语言环境下事实可靠性的挑战。**

- **链接: [https://arxiv.org/pdf/2512.11110v1](https://arxiv.org/pdf/2512.11110v1)**

> **作者:** Evren Ayberk Munis; Deniz Yılmaz; Arianna Muti; Çağrı Toraman
>
> **摘要:** Large language models are widely used across domains, yet there are concerns about their factual reliability and biases. Factual knowledge probing offers a systematic means to evaluate these aspects. Most existing benchmarks focus on single-entity facts and monolingual data. We therefore present FIBER, a multilingual benchmark for evaluating factual knowledge in single- and multi-entity settings. The dataset includes sentence completion, question-answering, and object-count prediction tasks in English, Italian, and Turkish. Using FIBER, we examine whether the prompt language induces inference bias in entity selection and how large language models perform on multi-entity versus single-entity questions. The results indicate that the language of the prompt can influence the model's generated output, particularly for entities associated with the country corresponding to that language. However, this effect varies across different topics such that 31% of the topics exhibit factual inference bias score greater than 0.5. Moreover, the level of bias differs across languages such that Turkish prompts show higher bias compared to Italian in 83% of the topics, suggesting a language-dependent pattern. Our findings also show that models face greater difficulty when handling multi-entity questions than the single-entity questions. Model performance differs across both languages and model sizes. The highest mean average precision is achieved in English, while Turkish and Italian lead to noticeably lower scores. Larger models, including Llama-3.1-8B and Qwen-2.5-7B, show consistently better performance than smaller 3B-4B models.
>
---
#### [new 018] Applying NLP to iMessages: Understanding Topic Avoidance, Responsiveness, and Sentiment
- **分类: cs.CL; cs.CY; stat.AP; stat.OT**

- **简介: 该论文属于NLP应用研究，旨在通过分析iMessage数据解决话题回避、响应速度、情感倾向等问题。作者开发了一个文本分析工具，探索本地消息数据在话题建模、响应时间与情感分析中的潜力。**

- **链接: [https://arxiv.org/pdf/2512.11079v1](https://arxiv.org/pdf/2512.11079v1)**

> **作者:** Alan Gerber; Sam Cooperman
>
> **备注:** 11 pages, 18 figures, https://github.com/Alanshnir/imessage-analyzer/blob/main/Research/NLP-iMessage-Analyzer%20Findings.pdf
>
> **摘要:** What is your messaging data used for? While many users do not often think about the information companies can gather based off of their messaging platform of choice, it is nonetheless important to consider as society increasingly relies on short-form electronic communication. While most companies keep their data closely guarded, inaccessible to users or potential hackers, Apple has opened a door to their walled-garden ecosystem, providing iMessage users on Mac with one file storing all their messages and attached metadata. With knowledge of this locally stored file, the question now becomes: What can our data do for us? In the creation of our iMessage text message analyzer, we set out to answer five main research questions focusing on topic modeling, response times, reluctance scoring, and sentiment analysis. This paper uses our exploratory data to show how these questions can be answered using our analyzer and its potential in future studies on iMessage data.
>
---
#### [new 019] Building Patient Journeys in Hebrew: A Language Model for Clinical Timeline Extraction
- **分类: cs.CL**

- **简介: 该论文属于临床信息抽取任务，旨在构建希伯来语患者诊疗时间线。作者基于DictaBERT 2.0持续预训练五百万匿名病历，提出新模型，并发布两个标注数据集。实验证明模型在时序关系提取上表现优异，且词汇适配提升效率，去标识化不影响性能。**

- **链接: [https://arxiv.org/pdf/2512.11502v1](https://arxiv.org/pdf/2512.11502v1)**

> **作者:** Kai Golan Hashiloni; Brenda Kasabe Nokai; Michal Shevach; Esthy Shemesh; Ronit Bartin; Anna Bergrin; Liran Harel; Nachum Dershowitz; Liat Nadai Arad; Kfir Bar
>
> **备注:** In Proceedings of the Workshop on Large Language Models and Generative AI for Health Informatics 2025, IJCAI 2025, Montreal, Canada
>
> **摘要:** We present a new Hebrew medical language model designed to extract structured clinical timelines from electronic health records, enabling the construction of patient journeys. Our model is based on DictaBERT 2.0 and continually pre-trained on over five million de-identified hospital records. To evaluate its effectiveness, we introduce two new datasets -- one from internal medicine and emergency departments, and another from oncology -- annotated for event temporal relations. Our results show that our model achieves strong performance on both datasets. We also find that vocabulary adaptation improves token efficiency and that de-identification does not compromise downstream performance, supporting privacy-conscious model development. The model is made available for research use under ethical restrictions.
>
---
#### [new 020] SciLaD: A Large-Scale, Transparent, Reproducible Dataset for Natural Scientific Language Processing
- **分类: cs.CL**

- **简介: 该论文构建了一个大规模、开源的科学语言数据集SciLaD，旨在提升科学自然语言处理的透明性与可复现性。作者使用公开数据和工具创建数据集，并发布生成流程与预训练模型，验证了其在多项基准上的有效性。**

- **链接: [https://arxiv.org/pdf/2512.11192v1](https://arxiv.org/pdf/2512.11192v1)**

> **作者:** Luca Foppiano; Sotaro Takeshita; Pedro Ortiz Suarez; Ekaterina Borisova; Raia Abu Ahmad; Malte Ostendorff; Fabio Barth; Julian Moreno-Schneider; Georg Rehm
>
> **备注:** 12 pages, 2 figures, 3 tables
>
> **摘要:** SciLaD is a novel, large-scale dataset of scientific language constructed entirely using open-source frameworks and publicly available data sources. It comprises a curated English split containing over 10 million scientific publications and a multilingual, unfiltered TEI XML split including more than 35 million publications. We also publish the extensible pipeline for generating SciLaD. The dataset construction and processing workflow demonstrates how open-source tools can enable large-scale, scientific data curation while maintaining high data quality. Finally, we pre-train a RoBERTa model on our dataset and evaluate it across a comprehensive set of benchmarks, achieving performance comparable to other scientific language models of similar size, validating the quality and utility of SciLaD. We publish the dataset and evaluation pipeline to promote reproducibility, transparency, and further research in natural scientific language processing and understanding including scholarly document processing.
>
---
#### [new 021] Mining Legal Arguments to Study Judicial Formalism
- **分类: cs.CL; cs.CY**

- **简介: 该论文属计算法律研究任务，旨在通过法律论据挖掘分析司法形式主义。针对中欧司法形式化争议，构建标注数据集MADON，结合自然语言处理技术训练模型，实现对捷克法院判决中论证类型与形式主义倾向的自动分类，挑战既有观点，推动可解释、可复现的司法哲学分析。**

- **链接: [https://arxiv.org/pdf/2512.11374v1](https://arxiv.org/pdf/2512.11374v1)**

> **作者:** Tomáš Koref; Lena Held; Mahammad Namazov; Harun Kumru; Yassine Thlija; Christoph Burchard; Ivan Habernal
>
> **备注:** pre-print under review
>
> **摘要:** Courts must justify their decisions, but systematically analyzing judicial reasoning at scale remains difficult. This study refutes claims about formalistic judging in Central and Eastern Europe (CEE) by developing automated methods to detect and classify judicial reasoning in Czech Supreme Courts' decisions using state-of-the-art natural language processing methods. We create the MADON dataset of 272 decisions from two Czech Supreme Courts with expert annotations of 9,183 paragraphs with eight argument types and holistic formalism labels for supervised training and evaluation. Using a corpus of 300k Czech court decisions, we adapt transformer LLMs for Czech legal domain by continued pretraining and experiment with methods to address dataset imbalance including asymmetric loss and class weighting. The best models successfully detect argumentative paragraphs (82.6\% macro-F1), classify traditional types of legal argument (77.5\% macro-F1), and classify decisions as formalistic/non-formalistic (83.2\% macro-F1). Our three-stage pipeline combining ModernBERT, Llama 3.1, and traditional feature-based machine learning achieves promising results for decision classification while reducing computational costs and increasing explainability. Empirically, we challenge prevailing narratives about CEE formalism. This work shows that legal argument mining enables reliable judicial philosophy classification and shows the potential of legal argument mining for other important tasks in computational legal studies. Our methodology is easily replicable across jurisdictions, and our entire pipeline, datasets, guidelines, models, and source codes are available at https://github.com/trusthlt/madon.
>
---
#### [new 022] Visualizing token importance for black-box language models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型可解释性任务，旨在解决黑盒大语言模型中输入token对输出影响的分析难题。作者提出DBSA方法，无需梯度或分布假设，即可可视化各输入token的敏感性，帮助用户快速发现模型依赖的关键输入片段。**

- **链接: [https://arxiv.org/pdf/2512.11573v1](https://arxiv.org/pdf/2512.11573v1)**

> **作者:** Paulius Rauba; Qiyao Wei; Mihaela van der Schaar
>
> **摘要:** We consider the problem of auditing black-box large language models (LLMs) to ensure they behave reliably when deployed in production settings, particularly in high-stakes domains such as legal, medical, and regulatory compliance. Existing approaches for LLM auditing often focus on isolated aspects of model behavior, such as detecting specific biases or evaluating fairness. We are interested in a more general question -- can we understand how the outputs of black-box LLMs depend on each input token? There is a critical need to have such tools in real-world applications that rely on inaccessible API endpoints to language models. However, this is a highly non-trivial problem, as LLMs are stochastic functions (i.e. two outputs will be different by chance), while computing prompt-level gradients to approximate input sensitivity is infeasible. To address this, we propose Distribution-Based Sensitivity Analysis (DBSA), a lightweight model-agnostic procedure to evaluate the sensitivity of the output of a language model for each input token, without making any distributional assumptions about the LLM. DBSA is developed as a practical tool for practitioners, enabling quick, plug-and-play visual exploration of LLMs reliance on specific input tokens. Through illustrative examples, we demonstrate how DBSA can enable users to inspect LLM inputs and find sensitivities that may be overlooked by existing LLM interpretability methods.
>
---
#### [new 023] MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MedBioRAG，面向生物医学问答任务，解决现有大模型在专业领域准确性不足的问题。通过结合语义与词汇检索、文档重排序及监督微调，提升检索与生成效果，在多项生物医学QA任务上超越现有方法。**

- **链接: [https://arxiv.org/pdf/2512.10996v1](https://arxiv.org/pdf/2512.10996v1)**

> **作者:** Seonok Kim
>
> **备注:** Submitted to ACL 2025. 9 pages, 4 figures, 5 tables (including 2 appendix tables)
>
> **摘要:** Recent advancements in retrieval-augmented generation (RAG) have significantly enhanced the ability of large language models (LLMs) to perform complex question-answering (QA) tasks. In this paper, we introduce MedBioRAG, a retrieval-augmented model designed to improve biomedical QA performance through a combination of semantic and lexical search, document retrieval, and supervised fine-tuning. MedBioRAG efficiently retrieves and ranks relevant biomedical documents, enabling precise and context-aware response generation. We evaluate MedBioRAG across text retrieval, close-ended QA, and long-form QA tasks using benchmark datasets such as NFCorpus, TREC-COVID, MedQA, PubMedQA, and BioASQ. Experimental results demonstrate that MedBioRAG outperforms previous state-of-the-art (SoTA) models and the GPT-4o base model in all evaluated tasks. Notably, our approach improves NDCG and MRR scores for document retrieval, while achieving higher accuracy in close-ended QA and ROUGE scores in long-form QA. Our findings highlight the effectiveness of semantic search-based retrieval and LLM fine-tuning in biomedical applications.
>
---
#### [new 024] Multi-Intent Spoken Language Understanding: Methods, Trends, and Challenges
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多意图口语理解任务，旨在解决同一语句中包含多个意图时的意图检测与槽位填充问题。作者系统综述了现有方法，从解码范式和建模角度分析进展，比较模型性能，并探讨挑战与未来方向。**

- **链接: [https://arxiv.org/pdf/2512.11258v1](https://arxiv.org/pdf/2512.11258v1)**

> **作者:** Di Wu; Ruiyu Fang; Liting Jiang; Shuangyong Song; Xiaomeng Huang; Shiquan Wang; Zhongqiu Li; Lingling Shi; Mengjiao Bao; Yongxiang Li; Hao Huang
>
> **摘要:** Multi-intent spoken language understanding (SLU) involves two tasks: multiple intent detection and slot filling, which jointly handle utterances containing more than one intent. Owing to this characteristic, which closely reflects real-world applications, the task has attracted increasing research attention, and substantial progress has been achieved. However, there remains a lack of a comprehensive and systematic review of existing studies on multi-intent SLU. To this end, this paper presents a survey of recent advances in multi-intent SLU. We provide an in-depth overview of previous research from two perspectives: decoding paradigms and modeling approaches. On this basis, we further compare the performance of representative models and analyze their strengths and limitations. Finally, we discuss the current challenges and outline promising directions for future research. We hope this survey will offer valuable insights and serve as a useful reference for advancing research in multi-intent SLU.
>
---
#### [new 025] Extending a Parliamentary Corpus with MPs' Tweets: Automatic Annotation and Evaluation Using MultiParTweet
- **分类: cs.CL; cs.MM**

- **简介: 该论文构建多语言推文语料库MultiParTweet，关联德国议会语料，通过自动模型标注情感、情绪和主题，并用人工标注验证。提出数据采集工具TTLABTweetCrawler，验证多模态标注更贴近人类理解。**

- **链接: [https://arxiv.org/pdf/2512.11567v1](https://arxiv.org/pdf/2512.11567v1)**

> **作者:** Mevlüt Bagci; Ali Abusaleh; Daniel Baumartz; Giueseppe Abrami; Maxim Konca; Alexander Mehler
>
> **备注:** Submitted to LREC 2026
>
> **摘要:** Social media serves as a critical medium in modern politics because it both reflects politicians' ideologies and facilitates communication with younger generations. We present MultiParTweet, a multilingual tweet corpus from X that connects politicians' social media discourse with German political corpus GerParCor, thereby enabling comparative analyses between online communication and parliamentary debates. MultiParTweet contains 39 546 tweets, including 19 056 media items. Furthermore, we enriched the annotation with nine text-based models and one vision-language model (VLM) to annotate MultiParTweet with emotion, sentiment, and topic annotations. Moreover, the automated annotations are evaluated against a manually annotated subset. MultiParTweet can be reconstructed using our tool, TTLABTweetCrawler, which provides a framework for collecting data from X. To demonstrate a methodological demonstration, we examine whether the models can predict each other using the outputs of the remaining models. In summary, we provide MultiParTweet, a resource integrating automatic text and media-based annotations validated with human annotations, and TTLABTweetCrawler, a general-purpose X data collection tool. Our analysis shows that the models are mutually predictable. In addition, VLM-based annotation were preferred by human annotators, suggesting that multimodal representations align more with human interpretation.
>
---
#### [new 026] Unifying Dynamic Tool Creation and Cross-Task Experience Sharing through Cognitive Memory Architecture
- **分类: cs.CL**

- **简介: 该论文研究LLM智能体的动态工具创建与跨任务经验共享问题，提出SMITH架构，通过分层记忆整合工具生成与经验复用，结合课程学习策略，在GAIA基准上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.11303v1](https://arxiv.org/pdf/2512.11303v1)**

> **作者:** Jiarun Liu; Shiyue Xu; Yang Li; Shangkun Liu; Yongli Yu; Peng Cao
>
> **摘要:** Large Language Model agents face fundamental challenges in adapting to novel tasks due to limitations in tool availability and experience reuse. Existing approaches either rely on predefined tools with limited coverage or build tools from scratch without leveraging past experiences, leading to inefficient exploration and suboptimal performance. We introduce SMITH (Shared Memory Integrated Tool Hub), a unified cognitive architecture that seamlessly integrates dynamic tool creation with cross-task experience sharing through hierarchical memory organization. SMITH organizes agent memory into procedural, semantic, and episodic components, enabling systematic capability expansion while preserving successful execution patterns. Our approach formalizes tool creation as iterative code generation within controlled sandbox environments and experience sharing through episodic memory retrieval with semantic similarity matching. We further propose a curriculum learning strategy based on agent-ensemble difficulty re-estimation. Extensive experiments on the GAIA benchmark demonstrate SMITH's effectiveness, achieving 81.8% Pass@1 accuracy and outperforming state-of-the-art baselines including Alita (75.2%) and Memento (70.9%). Our work establishes a foundation for building truly adaptive agents that continuously evolve their capabilities through principled integration of tool creation and experience accumulation.
>
---
#### [new 027] Speculative Decoding Speed-of-Light: Optimal Lower Bounds via Branching Random Walks
- **分类: cs.CL**

- **简介: 该论文研究大语言模型中推测解码的加速极限，通过分支随机游走建模，首次给出确定性推测生成算法的最优下界，揭示并行生成的理论性能极限，并经实验证实其紧致性。**

- **链接: [https://arxiv.org/pdf/2512.11718v1](https://arxiv.org/pdf/2512.11718v1)**

> **作者:** Sergey Pankratov; Dan Alistarh
>
> **摘要:** Speculative generation has emerged as a promising technique to accelerate inference in large language models (LLMs) by leveraging parallelism to verify multiple draft tokens simultaneously. However, the fundamental limits on the achievable speedup remain poorly understood. In this work, we establish the first ``tight'' lower bounds on the runtime of any deterministic speculative generation algorithm. This is achieved by drawing a parallel between the token generation process and branching random walks, which allows us to analyze the optimal draft tree selection problem. We prove, under basic assumptions, that the expected number of tokens successfully predicted per speculative iteration is bounded as $\mathbb{E}[X] \leq (μ+ μ_{(2)})\log(P )/μ^2 + O(1)$, where $P$ is the verifier's capacity, $μ$ is the expected entropy of the verifier's output distribution, and $μ_{(2)}$ is the expected second log-moment. This result provides new insights into the limits of parallel token generation, and could guide the design of future speculative decoding systems. Empirical evaluations on Llama models validate our theoretical predictions, confirming the tightness of our bounds in practical settings.
>
---
#### [new 028] KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering
- **分类: cs.CL**

- **简介: 该论文研究知识库问答（KBQA）任务，旨在解决大模型在生成逻辑形式时的幻觉或僵化推理问题。提出KBQA-R1框架，通过强化学习优化交互过程，并设计RRS方法合成高质量推理数据，提升模型基于执行反馈的决策能力。**

- **链接: [https://arxiv.org/pdf/2512.10999v1](https://arxiv.org/pdf/2512.10999v1)**

> **作者:** Xin Sun; Zhongqi Chen; Xing Zheng; Qiang Liu; Shu Wu; Bowen Song; Zilei Wang; Weiqiang Wang; Liang Wang
>
> **摘要:** Knowledge Base Question Answering (KBQA) challenges models to bridge the gap between natural language and strict knowledge graph schemas by generating executable logical forms. While Large Language Models (LLMs) have advanced this field, current approaches often struggle with a dichotomy of failure: they either generate hallucinated queries without verifying schema existence or exhibit rigid, template-based reasoning that mimics synthesized traces without true comprehension of the environment. To address these limitations, we present \textbf{KBQA-R1}, a framework that shifts the paradigm from text imitation to interaction optimization via Reinforcement Learning. Treating KBQA as a multi-turn decision process, our model learns to navigate the knowledge base using a list of actions, leveraging Group Relative Policy Optimization (GRPO) to refine its strategies based on concrete execution feedback rather than static supervision. Furthermore, we introduce \textbf{Referenced Rejection Sampling (RRS)}, a data synthesis method that resolves cold-start challenges by strictly aligning reasoning traces with ground-truth action sequences. Extensive experiments on WebQSP, GrailQA, and GraphQuestions demonstrate that KBQA-R1 achieves state-of-the-art performance, effectively grounding LLM reasoning in verifiable execution.
>
---
#### [new 029] Bounding Hallucinations: Information-Theoretic Guarantees for RAG Systems via Merlin-Arthur Protocols
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于RAG系统可靠性任务，旨在解决LLM在生成时依赖不可靠检索导致的幻觉问题。作者提出基于Merlin-Arthur协议的训练框架，通过对抗性上下文与可解释性方法，使生成器学会依据真实证据回答或拒绝，提升生成的可信度与可验证性。**

- **链接: [https://arxiv.org/pdf/2512.11614v1](https://arxiv.org/pdf/2512.11614v1)**

> **作者:** Björn Deiseroth; Max Henning Höth; Kristian Kersting; Letitia Parcalabescu
>
> **备注:** 34 pages, 19 figures
>
> **摘要:** Retrieval-augmented generation (RAG) models rely on retrieved evidence to guide large language model (LLM) generators, yet current systems treat retrieval as a weak heuristic rather than verifiable evidence. As a result, LLMs answer without support, hallucinate under incomplete or misleading context, and rely on spurious evidence. We introduce a training framework that treats the entire RAG pipeline -- both the retriever and the generator -- as an interactive proof system via an adaptation of the Merlin-Arthur (M/A) protocol. Arthur (the generator LLM) trains on questions of unkown provenance: Merlin provides helpful evidence, while Morgana injects adversarial, misleading context. Both use a linear-time XAI method to identify and modify the evidence most influential to Arthur. Consequently, Arthur learns to (i) answer when the context support the answer, (ii) reject when evidence is insufficient, and (iii) rely on the specific context spans that truly ground the answer. We further introduce a rigorous evaluation framework to disentangle explanation fidelity from baseline predictive errors. This allows us to introduce and measure the Explained Information Fraction (EIF), which normalizes M/A certified mutual-information guarantees relative to model capacity and imperfect benchmarks. Across three RAG datasets and two model families of varying sizes, M/A-trained LLMs show improved groundedness, completeness, soundness, and reject behavior, as well as reduced hallucinations -- without needing manually annotated unanswerable questions. The retriever likewise improves recall and MRR through automatically generated M/A hard positives and negatives. Our results demonstrate that autonomous interactive-proof-style supervision provides a principled and practical path toward reliable RAG systems that treat retrieved documents not as suggestions, but as verifiable evidence.
>
---
#### [new 030] Minimal Clips, Maximum Salience: Long Video Summarization via Key Moment Extraction
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视频摘要任务，旨在解决长视频中关键信息易丢失且计算成本高的问题。作者提出一种基于关键片段提取的方法，通过轻量级模型生成片段描述，利用大语言模型选择最具信息量的片段，实现高效、准确的多模态视频摘要。**

- **链接: [https://arxiv.org/pdf/2512.11399v1](https://arxiv.org/pdf/2512.11399v1)**

> **作者:** Galann Pennec; Zhengyuan Liu; Nicholas Asher; Philippe Muller; Nancy F. Chen
>
> **摘要:** Vision-Language Models (VLMs) are able to process increasingly longer videos. Yet, important visual information is easily lost throughout the entire context and missed by VLMs. Also, it is important to design tools that enable cost-effective analysis of lengthy video content. In this paper, we propose a clip selection method that targets key video moments to be included in a multimodal summary. We divide the video into short clips and generate compact visual descriptions of each using a lightweight video captioning model. These are then passed to a large language model (LLM), which selects the K clips containing the most relevant visual information for a multimodal summary. We evaluate our approach on reference clips for the task, automatically derived from full human-annotated screenplays and summaries in the MovieSum dataset. We further show that these reference clips (less than 6% of the movie) are sufficient to build a complete multimodal summary of the movies in MovieSum. Using our clip selection method, we achieve a summarization performance close to that of these reference clips while capturing substantially more relevant video information than random clip selection. Importantly, we maintain low computational cost by relying on a lightweight captioning model.
>
---
#### [new 031] AdaSD: Adaptive Speculative Decoding for Efficient Language Model Inference
- **分类: cs.CL**

- **简介: 该论文研究大语言模型推理加速任务，提出自适应推测解码（AdaSD），通过动态调整生成长度与接受阈值，无需调参或训练即可提升推理效率。实验表明其可显著加速推理并保持高精度。**

- **链接: [https://arxiv.org/pdf/2512.11280v1](https://arxiv.org/pdf/2512.11280v1)**

> **作者:** Kuan-Wei Lu; Ding-Yong Hong; Pangfeng Liu
>
> **摘要:** Large language models (LLMs) have achieved remarkable performance across a wide range of tasks, but their increasing parameter sizes significantly slow down inference. Speculative decoding mitigates this issue by leveraging a smaller draft model to predict candidate tokens, which are then verified by a larger target model. However, existing approaches often require additional training, extensive hyperparameter tuning, or prior analysis of models and tasks before deployment. In this paper, we propose Adaptive Speculative Decoding (AdaSD), a hyperparameter-free decoding scheme that dynamically adjusts generation length and acceptance criteria during inference. AdaSD introduces two adaptive thresholds: one to determine when to stop candidate token generation and another to decide token acceptance, both updated in real time based on token entropy and Jensen-Shannon distance. This approach eliminates the need for pre-analysis or fine-tuning and is compatible with off-the-shelf models. Experiments on benchmark datasets demonstrate that AdaSD achieves up to 49\% speedup over standard speculative decoding while limiting accuracy degradation to under 2\%, making it a practical solution for efficient and adaptive LLM inference.
>
---
#### [new 032] SCOUT: A Defense Against Data Poisoning Attacks in Fine-Tuned Language Models
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于AI安全任务，旨在防御语言模型中的数据投毒后门攻击。针对现有防御无法检测上下文合理触发器的问题，提出SCOUT框架，通过显著性分析识别隐蔽触发词，有效防御新型语义连贯的攻击，同时保持对正常输入的准确率。**

- **链接: [https://arxiv.org/pdf/2512.10998v1](https://arxiv.org/pdf/2512.10998v1)**

> **作者:** Mohamed Afane; Abhishek Satyam; Ke Chen; Tao Li; Junaid Farooq; Juntao Chen
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Backdoor attacks create significant security threats to language models by embedding hidden triggers that manipulate model behavior during inference, presenting critical risks for AI systems deployed in healthcare and other sensitive domains. While existing defenses effectively counter obvious threats such as out-of-context trigger words and safety alignment violations, they fail against sophisticated attacks using contextually-appropriate triggers that blend seamlessly into natural language. This paper introduces three novel contextually-aware attack scenarios that exploit domain-specific knowledge and semantic plausibility: the ViralApp attack targeting social media addiction classification, the Fever attack manipulating medical diagnosis toward hypertension, and the Referral attack steering clinical recommendations. These attacks represent realistic threats where malicious actors exploit domain-specific vocabulary while maintaining semantic coherence, demonstrating how adversaries can weaponize contextual appropriateness to evade conventional detection methods. To counter both traditional and these sophisticated attacks, we present \textbf{SCOUT (Saliency-based Classification Of Untrusted Tokens)}, a novel defense framework that identifies backdoor triggers through token-level saliency analysis rather than traditional context-based detection methods. SCOUT constructs a saliency map by measuring how the removal of individual tokens affects the model's output logits for the target label, enabling detection of both conspicuous and subtle manipulation attempts. We evaluate SCOUT on established benchmark datasets (SST-2, IMDB, AG News) against conventional attacks (BadNet, AddSent, SynBkd, StyleBkd) and our novel attacks, demonstrating that SCOUT successfully detects these sophisticated threats while preserving accuracy on clean inputs.
>
---
#### [new 033] HFS: Holistic Query-Aware Frame Selection for Efficient Video Reasoning
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文研究视频理解中的关键帧选择任务，旨在解决传统方法因独立打分导致的冗余和静态伪标签问题。提出端到端可训练框架HFS，通过查询感知的隐式向量、集级别优化与师生互学习，实现任务自适应的高效帧选择。**

- **链接: [https://arxiv.org/pdf/2512.11534v1](https://arxiv.org/pdf/2512.11534v1)**

> **作者:** Yiqing Yang; Kin-Man Lam
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** Key frame selection in video understanding presents significant challenges. Traditional top-K selection methods, which score frames independently, often fail to optimize the selection as a whole. This independent scoring frequently results in selecting frames that are temporally clustered and visually redundant. Additionally, training lightweight selectors using pseudo labels generated offline by Multimodal Large Language Models (MLLMs) prevents the supervisory signal from dynamically adapting to task objectives. To address these limitations, we propose an end-to-end trainable, task-adaptive framework for frame selection. A Chain-of-Thought approach guides a Small Language Model (SLM) to generate task-specific implicit query vectors, which are combined with multimodal features to enable dynamic frame scoring. We further define a continuous set-level objective function that incorporates relevance, coverage, and redundancy, enabling differentiable optimization via Gumbel-Softmax to select optimal frame combinations at the set level. Finally, student-teacher mutual learning is employed, where the student selector (SLM) and teacher reasoner (MLLM) are trained to align their frame importance distributions via KL divergence. Combined with cross-entropy loss, this enables end-to-end optimization, eliminating reliance on static pseudo labels. Experiments across various benchmarks, including Video-MME, LongVideoBench, MLVU, and NExT-QA, demonstrate that our method significantly outperforms existing approaches.
>
---
#### [new 034] From Signal to Turn: Interactional Friction in Modular Speech-to-Speech Pipelines
- **分类: cs.HC; cs.AI; cs.CL; cs.SE**

- **简介: 该论文研究模块化语音到语音生成系统中的交互摩擦问题，属于人机对话任务。针对系统响应不自然的问题，分析发现时间错位、表达扁平化和纠错僵化三类摩擦，并指出其为模块化设计的结构性结果，强调需从整体架构优化交互流畅性。**

- **链接: [https://arxiv.org/pdf/2512.11724v1](https://arxiv.org/pdf/2512.11724v1)**

> **作者:** Titaya Mairittha; Tanakon Sawanglok; Panuwit Raden; Jirapast Buntub; Thanapat Warunee; Napat Asawachaisuvikrom; Thanaphum Saiwongin
>
> **备注:** 6 pages, 1 figure
>
> **摘要:** While voice-based AI systems have achieved remarkable generative capabilities, their interactions often feel conversationally broken. This paper examines the interactional friction that emerges in modular Speech-to-Speech Retrieval-Augmented Generation (S2S-RAG) pipelines. By analyzing a representative production system, we move beyond simple latency metrics to identify three recurring patterns of conversational breakdown: (1) Temporal Misalignment, where system delays violate user expectations of conversational rhythm; (2) Expressive Flattening, where the loss of paralinguistic cues leads to literal, inappropriate responses; and (3) Repair Rigidity, where architectural gating prevents users from correcting errors in real-time. Through system-level analysis, we demonstrate that these friction points should not be understood as defects or failures, but as structural consequences of a modular design that prioritizes control over fluidity. We conclude that building natural spoken AI is an infrastructure design challenge, requiring a shift from optimizing isolated components to carefully choreographing the seams between them.
>
---
#### [new 035] TV2TV: A Unified Framework for Interleaved Language and Video Generation
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出TV2TV框架，属于视频生成任务，旨在解决复杂语义推理与多分支视频生成难题。通过融合语言与视频的交错生成，利用语言模型进行高层推理，提升生成质量与可控性。**

- **链接: [https://arxiv.org/pdf/2512.05103v2](https://arxiv.org/pdf/2512.05103v2)**

> **作者:** Xiaochuang Han; Youssef Emad; Melissa Hall; John Nguyen; Karthik Padthe; Liam Robbins; Amir Bar; Delong Chen; Michal Drozdzal; Maha Elbayad; Yushi Hu; Shang-Wen Li; Sreya Dutta Roy; Jakob Verbeek; XuDong Wang; Marjan Ghazvininejad; Luke Zettlemoyer; Emily Dinan
>
> **摘要:** Video generation models are rapidly advancing, but can still struggle with complex video outputs that require significant semantic branching or repeated high-level reasoning about what should happen next. In this paper, we introduce a new class of omni video-text models that integrate ideas from recent LM reasoning advances to address this challenge. More specifically, we present TV2TV, a unified generative modeling framework which decomposes video generation into an interleaved text and video generation process. TV2TV jointly learns language modeling (next-token prediction) and video flow matching (next-frame prediction) using a Mixture-of-Transformers (MoT) architecture. At inference time, TV2TV decides when to alternate between generating text and video frames, allowing the model to "think in words" about subsequent content before ``acting in pixels'' to produce frames. This design offloads much of the responsibility for deciding what should happen next to the language modeling tower, enabling improved visual quality and prompt alignment of generated videos. It also enables fine-grained controllability, allowing users to modify the video generation trajectory through text interventions at any point in the process. In controlled experiments on video game data, TV2TV demonstrates substantial improvements in both visual quality and controllability. TV2TV also scales to natural videos, as we show by augmenting sports videos with interleaved natural language action descriptions using vision-language models (VLMs). Training TV2TV on this corpus yields strong visual quality and prompt alignment, showcasing the model's ability to reason about and generate complex real-world action sequences. Together, these results highlight TV2TV as a promising step toward video generation with open-ended textual reasoning and control.
>
---
#### [new 036] Adaptive Soft Rolling KV Freeze with Entropy-Guided Recovery: Sublinear Memory Growth for Efficient LLM Inference
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大语言模型推理时内存消耗大的问题，提出一种无需训练的推理框架ASR-KF-EGR。通过自适应软冻结低重要性KV缓存并按需恢复，实现亚线性内存增长，在减少55-67%显存占用的同时保持生成质量。**

- **链接: [https://arxiv.org/pdf/2512.11221v1](https://arxiv.org/pdf/2512.11221v1)**

> **作者:** Adilet Metinov; Gulida M. Kudakeeva; Bolotbek uulu Nursultan; Gulnara D. Kabaeva
>
> **备注:** 6 pages, 3 tables , 1 figure
>
> **摘要:** We present Adaptive Soft Rolling KV Freeze with Entropy-Guided Recovery (ASR-KF-EGR), a training-free inference-time framework for efficient large language model generation. Our method introduces a reversible soft-freeze mechanism that temporarily suspends key-value (KV) updates for low-importance tokens identified within a sliding attention window. Unlike eviction-based approaches that permanently discard context, ASR-KF-EGR preserves all tokens in off-GPU storage and restores them on demand. We extend the framework with sublinear freeze scheduling, where freeze duration grows sublinearly with repeated low-importance detections, preventing over-aggressive compression. Preliminary experiments on LLaMA-3 8B demonstrate 55-67% reduction in active KV cache size while maintaining generation quality and passing needle-in-haystack retrieval tests. The method is architecture-agnostic, requires no fine-tuning, and provides a practical solution for memory-constrained deployment of long-context LLMs.
>
---
#### [new 037] FutureWeaver: Planning Test-Time Compute for Multi-Agent Systems with Modularized Collaboration
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究多智能体系统中推理时计算资源的分配问题，提出FutureWeaver框架，通过模块化协作和双层规划架构，在固定预算下优化计算资源分配，提升多智能体协同性能。**

- **链接: [https://arxiv.org/pdf/2512.11213v1](https://arxiv.org/pdf/2512.11213v1)**

> **作者:** Dongwon Jung; Peng Shi; Yi Zhang
>
> **摘要:** Scaling test-time computation improves large language model performance without additional training. Recent work demonstrates that techniques such as repeated sampling, self-verification, and self-reflection can significantly enhance task success by allocating more inference-time compute. However, applying these techniques across multiple agents in a multi-agent system is difficult: there does not exist principled mechanisms to allocate compute to foster collaboration among agents, to extend test-time scaling to collaborative interactions, or to distribute compute across agents under explicit budget constraints. To address this gap, we propose FutureWeaver, a framework for planning and optimizing test-time compute allocation in multi-agent systems under fixed budgets. FutureWeaver introduces modularized collaboration, formalized as callable functions that encapsulate reusable multi-agent workflows. These modules are automatically derived through self-play reflection by abstracting recurring interaction patterns from past trajectories. Building on these modules, FutureWeaver employs a dual-level planning architecture that optimizes compute allocation by reasoning over the current task state while also speculating on future steps. Experiments on complex agent benchmarks demonstrate that FutureWeaver consistently outperforms baselines across diverse budget settings, validating its effectiveness for multi-agent collaboration in inference-time optimization.
>
---
#### [new 038] Task-Specific Sparse Feature Masks for Molecular Toxicity Prediction with Chemical Language Models
- **分类: cs.CE; cs.AI; cs.CL; cs.LG; q-bio.BM**

- **简介: 该论文属分子毒性预测任务，旨在解决模型可解释性差的问题。作者提出多任务学习框架，结合化学语言模型与任务特定的稀疏注意力模块，通过L1正则化识别关键分子片段，在提升预测性能的同时增强化学可解释性。**

- **链接: [https://arxiv.org/pdf/2512.11412v1](https://arxiv.org/pdf/2512.11412v1)**

> **作者:** Kwun Sy Lee; Jiawei Chen; Fuk Sheng Ford Chung; Tianyu Zhao; Zhenyuan Chen; Debby D. Wang
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** Reliable in silico molecular toxicity prediction is a cornerstone of modern drug discovery, offering a scalable alternative to experimental screening. However, the black-box nature of state-of-the-art models remains a significant barrier to adoption, as high-stakes safety decisions demand verifiable structural insights alongside predictive performance. To address this, we propose a novel multi-task learning (MTL) framework designed to jointly enhance accuracy and interpretability. Our architecture integrates a shared chemical language model with task-specific attention modules. By imposing an L1 sparsity penalty on these modules, the framework is constrained to focus on a minimal set of salient molecular fragments for each distinct toxicity endpoint. The resulting framework is trained end-to-end and is readily adaptable to various transformer-based backbones. Evaluated on the ClinTox, SIDER, and Tox21 benchmark datasets, our approach consistently outperforms both single-task and standard MTL baselines. Crucially, the sparse attention weights provide chemically intuitive visualizations that reveal the specific fragments influencing predictions, thereby enhancing insight into the model's decision-making process.
>
---
#### [new 039] Rethinking Expert Trajectory Utilization in LLM Post-training
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究大语言模型后训练中专家轨迹的利用问题，提出“可塑性-上限”框架，通过理论分析与实验验证，确立SFT-RL分阶段流程的优越性，并给出数据规模、难度与验证损失的实用优化准则。**

- **链接: [https://arxiv.org/pdf/2512.11470v1](https://arxiv.org/pdf/2512.11470v1)**

> **作者:** Bowen Ding; Yuhan Chen; Jiayang Lv; Jiyao Yuan; Qi Zhu; Shuangshuang Tian; Dantong Zhu; Futing Wang; Heyuan Deng; Fei Mi; Lifeng Shang; Tao Lin
>
> **备注:** 24 pages, 5 figures, under review
>
> **摘要:** While effective post-training integrates Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), the optimal mechanism for utilizing expert trajectories remains unresolved. We propose the Plasticity-Ceiling Framework to theoretically ground this landscape, decomposing performance into foundational SFT performance and the subsequent RL plasticity. Through extensive benchmarking, we establish the Sequential SFT-then-RL pipeline as the superior standard, overcoming the stability deficits of synchronized approaches. Furthermore, we derive precise scaling guidelines: (1) Transitioning to RL at the SFT Stable or Mild Overfitting Sub-phase maximizes the final ceiling by securing foundational SFT performance without compromising RL plasticity; (2) Refuting ``Less is More'' in the context of SFT-then-RL scaling, we demonstrate that Data Scale determines the primary post-training potential, while Trajectory Difficulty acts as a performance multiplier; and (3) Identifying that the Minimum SFT Validation Loss serves as a robust indicator for selecting the expert trajectories that maximize the final performance ceiling. Our findings provide actionable guidelines for maximizing the value extracted from expert trajectories.
>
---
#### [new 040] DentalGPT: Incentivizing Multimodal Complex Reasoning in Dentistry
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出DentalGPT，旨在提升牙科多模态复杂推理能力。针对现有模型在牙科视觉细节理解和推理上的不足，构建了12万+标注数据集，并结合领域知识注入与强化学习，显著提升疾病分类与牙科视觉问答性能。**

- **链接: [https://arxiv.org/pdf/2512.11558v1](https://arxiv.org/pdf/2512.11558v1)**

> **作者:** Zhenyang Cai; Jiaming Zhang; Junjie Zhao; Ziyi Zeng; Yanchao Li; Jingyi Liang; Junying Chen; Yunjin Yang; Jiajun You; Shuzhi Deng; Tongfei Wang; Wanting Chen; Chunxiu Hao; Ruiqi Xie; Zhenwei Wen; Xiangyi Feng; Zou Ting; Jin Zou Lin; Jianquan Li; Guangjun Yu; Liangyi Chen; Junwen Wang; Shan Jiang; Benyou Wang
>
> **摘要:** Reliable interpretation of multimodal data in dentistry is essential for automated oral healthcare, yet current multimodal large language models (MLLMs) struggle to capture fine-grained dental visual details and lack sufficient reasoning ability for precise diagnosis. To address these limitations, we present DentalGPT, a specialized dental MLLM developed through high-quality domain knowledge injection and reinforcement learning. Specifically, the largest annotated multimodal dataset for dentistry to date was constructed by aggregating over 120k dental images paired with detailed descriptions that highlight diagnostically relevant visual features, making it the multimodal dataset with the most extensive collection of dental images to date. Training on this dataset significantly enhances the MLLM's visual understanding of dental conditions, while the subsequent reinforcement learning stage further strengthens its capability for multimodal complex reasoning. Comprehensive evaluations on intraoral and panoramic benchmarks, along with dental subsets of medical VQA benchmarks, show that DentalGPT achieves superior performance in disease classification and dental VQA tasks, outperforming many state-of-the-art MLLMs despite having only 7B parameters. These results demonstrate that high-quality dental data combined with staged adaptation provides an effective pathway for building capable and domain-specialized dental MLLMs.
>
---
## 更新

#### [replaced 001] ReCode: Unify Plan and Action for Universal Granularity Control
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究LLM智能体的多粒度决策问题，提出ReCode框架，通过递归代码生成统一规划与动作。将高层计划视为抽象函数，逐步分解至基本动作，实现灵活粒度控制，并提升推理性能与数据效率。**

- **链接: [https://arxiv.org/pdf/2510.23564v3](https://arxiv.org/pdf/2510.23564v3)**

> **作者:** Zhaoyang Yu; Jiayi Zhang; Huixue Su; Yufan Zhao; Yifan Wu; Mingyi Deng; Jinyu Xiang; Yizhang Lin; Lingxiao Tang; Yuyu Luo; Bang Liu; Chenglin Wu
>
> **摘要:** Real-world tasks require decisions at varying granularities, and humans excel at this by leveraging a unified cognitive representation where planning is fundamentally understood as a high-level form of action. However, current Large Language Model (LLM)-based agents lack this crucial capability to operate fluidly across decision granularities. This limitation stems from existing paradigms that enforce a rigid separation between high-level planning and low-level action, which impairs dynamic adaptability and limits generalization. We propose ReCode (Recursive Code Generation), a novel paradigm that addresses this limitation by unifying planning and action within a single code representation. In this representation, ReCode treats high-level plans as abstract placeholder functions, which the agent then recursively decomposes into finer-grained sub-functions until reaching primitive actions. This recursive approach dissolves the rigid boundary between plan and action, enabling the agent to dynamically control its decision granularity. Furthermore, the recursive structure inherently generates rich, multi-granularity training data, enabling models to learn hierarchical decision-making processes. Extensive experiments show ReCode significantly surpasses advanced baselines in inference performance and demonstrates exceptional data efficiency in training, validating our core insight that unifying planning and action through recursive code generation is a powerful and effective approach to achieving universal granularity control. The code is available at https://github.com/FoundationAgents/ReCode.
>
---
#### [replaced 002] Scalable Best-of-N Selection for Large Language Models via Self-Certainty
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属LLM推理优化任务，旨在解决现有Best-of-N方法依赖高成本奖励模型或难以扩展的问题。作者提出“自确定性”指标，利用LLM输出概率分布评估响应质量，无需外部奖励模型，实现高效、可扩展的推理提升。**

- **链接: [https://arxiv.org/pdf/2502.18581v3](https://arxiv.org/pdf/2502.18581v3)**

> **作者:** Zhewei Kang; Xuandong Zhao; Dawn Song
>
> **备注:** NeurIPS 2025
>
> **摘要:** Best-of-N selection is a key technique for improving the reasoning performance of Large Language Models (LLMs) through increased test-time computation. Current state-of-the-art methods often employ computationally intensive reward models for response evaluation and selection. Reward-free alternatives, like self-consistency and universal self-consistency, are limited in their ability to handle open-ended generation tasks or scale effectively. To address these limitations, we propose self-certainty, a novel and efficient metric that leverages the inherent probability distribution of LLM outputs to estimate response quality without requiring external reward models. We hypothesize that higher distributional self-certainty, aggregated across multiple samples, correlates with improved response accuracy, as it reflects greater confidence in the generated output. Through extensive experiments on various reasoning tasks, we demonstrate that self-certainty (1) scales effectively with increasing sample size N, akin to reward models but without the computational overhead; (2) complements chain-of-thought, improving reasoning performance beyond greedy decoding; and (3) generalizes to open-ended tasks where traditional self-consistency methods fall short. Our findings establish self-certainty as a practical and efficient way for improving LLM reasoning capabilities. The code is available at https://github.com/backprop07/Self-Certainty
>
---
#### [replaced 003] Large Continual Instruction Assistant
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究持续指令微调任务，旨在缓解模型在持续学习中遗忘旧知识的问题。提出基于梯度与参数的平衡系数，动态调整稳定性与可塑性，并结合语义相似性决定参数重用或扩展，提升抗遗忘能力与整体性能。**

- **链接: [https://arxiv.org/pdf/2410.10868v5](https://arxiv.org/pdf/2410.10868v5)**

> **作者:** Jingyang Qiao; Zhizhong Zhang; Xin Tan; Yanyun Qu; Shouhong Ding; Yuan Xie
>
> **摘要:** Continual Instruction Tuning (CIT) is adopted to continually instruct Large Models to follow human intent data by data. It is observed that existing gradient update would heavily destroy the performance on previous datasets during CIT process. Instead, Exponential Moving Average (EMA), owns the ability to trace previous parameters, which can aid in decreasing forgetting. Nonetheless, its stable balance weight fails to deal with the ever-changing datasets, leading to the out-of-balance between plasticity and stability. In this paper, we propose a general continual instruction tuning framework to address the challenge. Starting from the trade-off prerequisite and EMA update, we propose the plasticity and stability ideal condition. Based on Taylor expansion in the loss function, we find the optimal balance weight can be automatically determined by the gradients and learned parameters. Therefore, we propose a stable-plasticity balanced coefficient to avoid knowledge interference. Based on the semantic similarity of the instructions, we can determine whether to retrain or expand the training parameters and allocate the most suitable parameters for the testing instances. Extensive experiments across multiple continual instruction tuning benchmarks demonstrate that our approach not only enhances anti-forgetting capabilities but also significantly improves overall continual tuning performance. Our code is available at https://github.com/JingyangQiao/CoIN.
>
---
#### [replaced 004] M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出M3-Embedding模型，解决多语言、多功能、多粒度文本嵌入问题。通过自知识蒸馏和优化 batching 策略，统一支持百种语言、多种检索功能及长短文本，提升语义检索性能，在多语言和长文档基准上达到SOTA。**

- **链接: [https://arxiv.org/pdf/2402.03216v5](https://arxiv.org/pdf/2402.03216v5)**

> **作者:** Jianlv Chen; Shitao Xiao; Peitian Zhang; Kun Luo; Defu Lian; Zheng Liu
>
> **摘要:** In this paper, we introduce a new embedding model called M3-Embedding, which is distinguished for its versatility in \textit{Multi-Linguality}, \textit{Multi-Functionality}, and \textit{Multi-Granularity}. It provides a uniform support for the semantic retrieval of more than 100 working languages. It can simultaneously accomplish the three common retrieval functionalities: dense retrieval, multi-vector retrieval, and sparse retrieval. Besides, it is also capable of processing inputs of different granularities, spanning from short sentences to long documents of up to 8,192 tokens. The effective training of M3-Embedding presents a series of technical contributions. Notably, we propose a novel self-knowledge distillation approach, where the relevance scores from different retrieval functionalities can be integrated as the teacher signal to enhance the training quality. We also optimize the batching strategy, which enables a large batch size and high training throughput to improve the discriminativeness of embeddings. M3-Embedding exhibits a superior performance in our experiment, leading to new state-of-the-art results on multilingual, cross-lingual, and long-document retrieval benchmarks.
>
---
#### [replaced 005] Dynamics of Spontaneous Topic Changes in Next Token Prediction with Self-Attention
- **分类: cs.CL; cs.AI; stat.ML**

- **简介: 该论文研究自注意力模型在下一词预测中自发话题转换的机制，揭示其与人类思维差异。通过理论建模与实证分析，表明模型缺乏真正 spontaneity，上下文越长或主题越模糊，话题切换越难，凸显AI与人类认知的根本区别。**

- **链接: [https://arxiv.org/pdf/2501.06382v4](https://arxiv.org/pdf/2501.06382v4)**

> **作者:** Mumin Jia; Jairo Diaz-Rodriguez
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Human cognition is punctuated by abrupt, spontaneous shifts between topics-driven by emotional, contextual, or associative cues-a phenomenon known as spontaneous thought in neuroscience. In contrast, self-attention based models depend on structured patterns over their inputs to predict each next token, lacking spontaneity. Motivated by this distinction, we characterize spontaneous topic changes in self-attention architectures, revealing both their similarities and their divergences from spontaneous human thought. First, we establish theoretical results under a simplified, single-layer self-attention model with suitable conditions by defining the topic as a set of Token Priority Graphs (TPGs). Specifically, we demonstrate that (1) the model maintains the priority order of tokens related to the input topic, (2) a spontaneous topic change can occur only if lower-priority tokens outnumber all higher-priority tokens of the input topic, and (3) unlike human cognition, the longer context length or the more ambiguous input topic reduces the likelihood of spontaneous change. Second, we empirically validate that these dynamics persist in modern, state-of-the-art LLMs, underscoring a fundamental disparity between human cognition and AI behaviour in the context of spontaneous topic changes. To the best of our knowledge, no prior work has explored these questions with a focus as closely aligned to human thought.
>
---
#### [replaced 006] MixtureVitae: Open Web-Scale Pretraining Dataset With High Quality Instruction and Reasoning Data Built from Permissive-First Text Sources
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦开放大模型预训练数据构建任务，旨在解决法律风险与数据质量的平衡问题。作者提出MixtureVitae，采用许可优先策略，融合公有领域、宽松授权文本及低风险来源，构建高质量、可追溯的指令与推理数据集，并验证其在多规模模型上的有效性。**

- **链接: [https://arxiv.org/pdf/2509.25531v3](https://arxiv.org/pdf/2509.25531v3)**

> **作者:** Huu Nguyen; Victor May; Harsh Raj; Marianna Nezhurina; Yishan Wang; Yanqi Luo; Minh Chien Vu; Taishi Nakamura; Ken Tsui; Van Khue Nguyen; David Salinas; Aleksandra Krasnodębska; Christoph Schuhmann; Mats Leon Richter; Xuan-Son; Vu; Jenia Jitsev
>
> **备注:** Code: \url{https://github.com/ontocord/mixturevitae}
>
> **摘要:** We present MixtureVitae, an open-access pretraining corpus built to minimize legal risk while providing strong model performance. MixtureVitae follows a risk-mitigated sourcing strategy that combines public-domain and permissively licensed text (e.g., CC-BY/Apache) with carefully justified low-risk additions (e.g., government works and EU TDM-eligible sources), alongside targeted instruction, reasoning and synthetic data with documented provenance. We detail a transparent, multi-stage pipeline for license-aware filtering, safety and quality screening, and domain-aware mixing, and we release the dataset and curation recipes to support reproducible research. In controlled experiments using the open-sci-ref training protocol (fixed architectures at 130M/400M/1.3B/1.7B parameters; training budgets of 50B and 300B tokens), models trained on MixtureVitae consistently outperform other permissive datasets across a suite of standard benchmarks, and at the 1.7B/300B setting they surpass FineWeb-Edu and approach DCLM in the later stages of training. Performance is particularly strong on math/code and competitive on QA tasks. These results demonstrate that permissive-first, risk-mitigated data provides a practical and legally mitigated foundation for training capable LLMs, reducing reliance on indiscriminate web scraping without sacrificing competitiveness. Code: https://github.com/ontocord/mixturevitae
>
---
#### [replaced 007] Long-horizon Reasoning Agent for Olympiad-Level Mathematical Problem Solving
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦奥林匹克数学难题求解，旨在突破大模型在长程推理中的上下文限制。提出Intern-S1-MO代理，构建基于多智能体的分层推理系统，并设计OREAL-H框架进行强化学习训练，实现IMO级别问题的有效求解。**

- **链接: [https://arxiv.org/pdf/2512.10739v2](https://arxiv.org/pdf/2512.10739v2)**

> **作者:** Songyang Gao; Yuzhe Gu; Zijian Wu; Lingkai Kong; Wenwei Zhang; Zhongrui Cai; Fan Zheng; Tianyou Ma; Junhao Shen; Haiteng Zhao; Duanyang Zhang; Huilun Zhang; Kuikun Liu; Chengqi Lyu; Yanhui Duan; Chiyu Chen; Ningsheng Ma; Jianfei Gao; Han Lyu; Dahua Lin; Kai Chen
>
> **摘要:** Large Reasoning Models (LRMs) have expanded the mathematical reasoning frontier through Chain-of-Thought (CoT) techniques and Reinforcement Learning with Verifiable Rewards (RLVR), capable of solving AIME-level problems. However, the performance of LRMs is heavily dependent on the extended reasoning context length. For solving ultra-hard problems like those in the International Mathematical Olympiad (IMO), the required reasoning complexity surpasses the space that an LRM can explore in a single round. Previous works attempt to extend the reasoning context of LRMs but remain prompt-based and built upon proprietary models, lacking systematic structures and training pipelines. Therefore, this paper introduces Intern-S1-MO, a long-horizon math agent that conducts multi-round hierarchical reasoning, composed of an LRM-based multi-agent system including reasoning, summary, and verification. By maintaining a compact memory in the form of lemmas, Intern-S1-MO can more freely explore the lemma-rich reasoning spaces in multiple reasoning stages, thereby breaking through the context constraints for IMO-level math problems. Furthermore, we propose OREAL-H, an RL framework for training the LRM using the online explored trajectories to simultaneously bootstrap the reasoning ability of LRM and elevate the overall performance of Intern-S1-MO. Experiments show that Intern-S1-MO can obtain 26 out of 35 points on the non-geometry problems of IMO2025, matching the performance of silver medalists. It also surpasses the current advanced LRMs on inference benchmarks such as HMMT2025, AIME2025, and CNMO2025. In addition, our agent officially participates in CMO2025 and achieves a score of 102/126 under the judgment of human experts, reaching the gold medal level.
>
---
#### [replaced 008] BAMBO: Construct Ability and Efficiency LLM Pareto Set via Bayesian Adaptive Multi-objective Block-wise Optimization
- **分类: cs.LG; cs.CL; cs.NE**

- **简介: 该论文属多目标优化任务，旨在解决大模型能力与效率权衡问题。提出BAMBO框架，通过块级贝叶斯自适应优化，结合动态规划分块与qEHVI进化搜索，自动构建高质量Pareto解集。**

- **链接: [https://arxiv.org/pdf/2512.09972v2](https://arxiv.org/pdf/2512.09972v2)**

> **作者:** Kesheng Chen; Wenjian Luo; Zhenqian Zhu; Yamin Hu; Yiya Xi
>
> **摘要:** Constructing a Pareto set is pivotal for navigating the capability-efficiency trade-offs in Large Language Models (LLMs); however, existing merging techniques remain inadequate for this task. Coarse-grained, model-level methods yield only a sparse set of suboptimal solutions, while fine-grained, layer-wise approaches suffer from the "curse of dimensionality," rendering the search space computationally intractable. To resolve this dichotomy, we propose BAMBO (Bayesian Adaptive Multi-objective Block-wise Optimization), a novel framework that automatically constructs the LLM Pareto set. BAMBO renders the search tractable by introducing a Hybrid Optimal Block Partitioning strategy. Formulated as a 1D clustering problem, this strategy leverages a dynamic programming approach to optimally balance intra-block homogeneity and inter-block information distribution, thereby dramatically reducing dimensionality without sacrificing critical granularity. The entire process is automated within an evolutionary loop driven by the q-Expected Hypervolume Improvement (qEHVI) acquisition function. Experiments demonstrate that BAMBO discovers a superior and more comprehensive Pareto frontier than baselines, enabling agile model selection tailored to diverse operational constraints. Code is available at: https://github.com/xin8coder/BAMBO.
>
---
#### [replaced 009] Counterfactual Segmentation Reasoning: Diagnosing and Mitigating Pixel-Grounding Hallucination
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究分割视觉语言模型的像素错觉问题，提出反事实分割推理任务及新基准HalluSegBench，诊断并缓解视觉驱动的幻觉。通过反事实微调训练RobustSeg模型，在减少30%幻觉的同时提升分割性能。**

- **链接: [https://arxiv.org/pdf/2506.21546v3](https://arxiv.org/pdf/2506.21546v3)**

> **作者:** Xinzhuo Li; Adheesh Juvekar; Jiaxun Zhang; Xingyou Liu; Muntasir Wahed; Kiet A. Nguyen; Yifan Shen; Tianjiao Yu; Ismini Lourentzou
>
> **备注:** Project webpage: https://plan-lab.github.io/hallusegbench/
>
> **摘要:** Segmentation Vision-Language Models (VLMs) have significantly advanced grounded visual understanding, yet they remain prone to pixel-grounding hallucinations, producing masks for incorrect objects or for objects that are entirely absent. Existing evaluations rely almost entirely on text- or label-based perturbations, which check only whether the predicted mask matches the queried label. Such evaluations overlook the spatial footprint and severity of hallucination and therefore fail to reveal vision-driven hallucinations, which are more challenging and more prevalent. To address this gap, we formalize the task of Counterfactual Segmentation Reasoning (CSR), where a model must segment the referenced object in the factual image and abstain in its counterfactual counterpart. To support this task, we curate HalluSegBench, the first large-scale benchmark to diagnose referring and reasoning expression segmentation hallucinations using controlled visual counterfactuals, alongside new evaluation metrics that measure hallucination severity and disentangle vision- and language-driven failure modes. We further introduce RobustSeg, a segmentation VLM trained with counterfactual fine-tuning (CFT) to learn when to segment and when to abstain. Experimental results confirm RobustSeg reduces hallucinations by 30%, while improving segmentation performance on FP-RefCOCO(+/g).
>
---
#### [replaced 010] Statistical Analysis of Sentence Structures through ASCII, Lexical Alignment and PCA
- **分类: cs.CL**

- **简介: 该论文属NLP任务，旨在不依赖传统句法工具的情况下分析句子结构平衡。提出用ASCII码表示文本，结合词汇对齐与PCA降维，通过统计检验和可视化分析11个语料库的结构分布，验证方法在评估文本平衡性上的有效性。**

- **链接: [https://arxiv.org/pdf/2503.10470v2](https://arxiv.org/pdf/2503.10470v2)**

> **作者:** Abhijeet Sahdev
>
> **摘要:** While utilizing syntactic tools such as parts-of-speech (POS) tagging has helped us understand sentence structures and their distribution across diverse corpora, it is quite complex and poses a challenge in natural language processing (NLP). This study focuses on understanding sentence structure balance - usages of nouns, verbs, determiners, etc - harmoniously without relying on such tools. It proposes a novel statistical method that uses American Standard Code for Information Interchange (ASCII) codes to represent text of 11 text corpora from various sources and their lexical category alignment after using their compressed versions through PCA, and analyzes the results through histograms and normality tests such as Shapiro-Wilk and Anderson-Darling Tests. By focusing on ASCII codes, this approach simplifies text processing, although not replacing any syntactic tools but complementing them by offering it as a resource-efficient tool for assessing text balance. The story generated by Grok shows near normality indicating balanced sentence structures in LLM outputs, whereas 4 out of the remaining 10 pass the normality tests. Further research could explore potential applications in text quality evaluation and style analysis with syntactic integration for more broader tasks.
>
---
#### [replaced 011] Mind the Confidence Gap: Overconfidence, Calibration, and Distractor Effects in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型的置信度校准问题，旨在缓解其预测置信度与实际准确性不匹配的风险。通过引入干扰项提示并评估九个模型在问答任务中的表现，发现结构化提示可显著改善校准效果，并提出针对性优化建议。**

- **链接: [https://arxiv.org/pdf/2502.11028v3](https://arxiv.org/pdf/2502.11028v3)**

> **作者:** Prateek Chhikara
>
> **备注:** Published in Transactions on Machine Learning Research (TMLR)
>
> **摘要:** Large Language Models (LLMs) show remarkable proficiency in natural language tasks, yet their frequent overconfidence-misalignment between predicted confidence and true correctness-poses significant risks in critical decision-making applications. We present a comprehensive analysis on calibration in LLMs across nine LLMs and three factual Question-Answering (QA) datasets, systematically comparing standard free-generation settings against structured distractor-augmented prompts. Our evaluation reveals that explicitly incorporating distractors can substantially mitigate miscalibration, achieving relative accuracy improvements up to 460% and ECE reductions up to 90%. Despite general trends, we uncover nuanced findings: large RLHF-tuned models display inherent calibration strengths but can paradoxically suffer increased miscalibration on easier queries, whereas smaller models benefit disproportionately from distractor prompts but remain significantly miscalibrated. Through detailed analyses across question types, we identify persistent calibration failures, particularly in person-based queries. We conclude with concrete recommendations-targeted fine-tuning, structured prompting, and strategic model choice-to ensure reliable, trustworthy LLM deployments.
>
---
#### [replaced 012] PUMA: Discovery of Protein Units via Mutation-Aware Merging
- **分类: cs.CL; q-bio.QM**

- **简介: 该论文提出PUMA方法，旨在发现蛋白质中具有进化意义的功能单元。通过突变感知的迭代合并，构建反映演化关系的蛋白单元层级体系，并验证其在临床变异、功能注释和语言模型中的生物学意义。**

- **链接: [https://arxiv.org/pdf/2503.08838v2](https://arxiv.org/pdf/2503.08838v2)**

> **作者:** Burak Suyunu; Özdeniz Dolu; Ibukunoluwa Abigail Olaosebikan; Hacer Karatas Bristow; Arzucan Özgür
>
> **备注:** 18 pages, 12 figures, 1 table, 1 algorithm
>
> **摘要:** Proteins are the essential drivers of biological processes. At the molecular level, they are chains of amino acids that can be viewed through a linguistic lens where the twenty standard residues serve as an alphabet combining to form a complex language, referred to as the language of life. To understand this language, we must first identify its fundamental units. Analogous to words, these units are hypothesized to represent an intermediate layer between single residues and larger domains. Crucially, just as protein diversity arises from evolution, these units should inherently reflect evolutionary relationships. We introduce PUMA (Protein Units via Mutation-Aware Merging) to discover these evolutionarily meaningful units. PUMA employs an iterative merging algorithm guided by substitution matrices to identify protein units and organize them into families linked by plausible mutations. This process creates a hierarchical genealogy where parent units and their mutational variants coexist, simultaneously producing a unit vocabulary and the genealogical structure connecting them. We validate that PUMA families are biologically meaningful; mutations within a PUMA family correlate with clinically benign variants and with high-scoring mutations in high-throughput assays. Furthermore, these units align with the contextual preferences of protein language models and map to known functional annotations. PUMA's genealogical framework provides evolutionarily grounded units, offering a structured approach for understanding the language of life.
>
---
#### [replaced 013] Mirror Speculative Decoding: Breaking the Serial Barrier in LLM Inference
- **分类: cs.CL**

- **简介: 该论文属于LLM推理加速任务，旨在解决 speculative decoding 中延迟与接受率的权衡问题。提出 Mirror-SD，通过异构并行执行和多令牌流式推测，实现高接受率与低开销，显著提升端到端推理速度。**

- **链接: [https://arxiv.org/pdf/2510.13161v2](https://arxiv.org/pdf/2510.13161v2)**

> **作者:** Nikhil Bhendawade; Kumari Nishu; Arnav Kundu; Chris Bartels; Minsik Cho; Irina Belousova
>
> **摘要:** Speculative decoding accelerates LLM inference by using a draft model to look ahead, but gains are capped by the cost of autoregressive draft generation: increasing draft size elevates acceptance rates but introduces additional latency overhead exacerbating the speed-accuracy tradeoff. Prior methods (Medusa, Hydra, EAGLE) partially reduce draft cost but either degrade acceptance or introduce overheads that limit scaling. We present Mirror Speculative Decoding (Mirror-SD), an inference algorithm that breaks the latency-acceptance tradeoff. Mirror-SD launches branch-complete rollouts from early-exit signals in parallel with the target model's suffix and explicitly maps computation across heterogeneous accelerators (GPU and NPU) to exploit cross-device parallelism. The draft speculates forward continuations for the target to verify, while the target simultaneously speculates correction paths for the draft, converting speculation into two complementary execution pipelines. To further cut draft latency without weakening acceptance semantics, we add speculative streaming so the draft emits multiple tokens per step. This dual strategy of parallel heterogeneous execution plus multi-token speculative streaming pushes speculative decoding toward its ideal regime of high acceptance with low overhead. On SpecBench with server-scale models from 14B to 66B parameters, Mirror-SD delivers consistent end-to-end gains, achieving 2.8x-5.8x wall-time speedups across diverse tasks and a 30% average relative improvement over the strongest baseline, EAGLE3.
>
---
#### [replaced 014] Beyond Early-Token Bias: Model-Specific and Language-Specific Position Effects in Multilingual LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究多语言大模型中的位置偏差问题，探究不同语言和模型架构下上下文位置对信息利用的影响。通过多语言实验，揭示模型特异性和语言特异性的位置偏好，挑战早期token偏好的普遍假设，并分析提示策略对准确率和输出熵的影响。**

- **链接: [https://arxiv.org/pdf/2505.16134v3](https://arxiv.org/pdf/2505.16134v3)**

> **作者:** Mikhail Menschikov; Alexander Kharitonov; Maiia Kotyga; Vadim Porvatov; Anna Zhukovskaya; David Kagramanyan; Egor Shvetsov; Evgeny Burnaev
>
> **摘要:** Large Language Models (LLMs) exhibit position bias systematically underweighting information based on its location in the context but how this bias varies across languages and models remains unclear. We conduct a multilingual study across five typologically diverse languages (English, Russian, German, Hindi, Vietnamese) and five model architectures, analyzing how position bias interacts with prompting strategies and affects output entropy. Our key findings are: (1) Position bias is primarily model-driven but shows language-specific nuances. Notably, Qwen2.5-7B-Instruct, DeepSeek 7B Chat and Mistral 7B consistently favor late positions challenging the common assumption of universal early-token preference. (2) Explicitly instructing the model, in the presence of irrelevant distractors, that "the most relevant context to the query is marked as 1" unexpectedly reduces accuracy across all languages, questioning standard prompt-engineering practices. (3) Accuracy consistently drops most when relevant information appears in the middle of the context, yet this is not reflected in a corresponding increase in output entropy, suggesting the model remains confident even when it fails to use mid-context cues.
>
---
#### [replaced 015] The Expressive Capacity of State Space Models: A Formal Language Perspective
- **分类: cs.CL; cs.FL; cs.LG**

- **简介: 该论文研究线性状态空间模型（SSM）在语言建模中的表达能力，从形式语言角度对比其与Transformer和RNN的能力。发现SSM在星自由状态跟踪和层次结构建模中具优势，但存在设计局限，提出改进方向，并在Mamba上验证。**

- **链接: [https://arxiv.org/pdf/2405.17394v3](https://arxiv.org/pdf/2405.17394v3)**

> **作者:** Yash Sarrof; Yana Veitsman; Michael Hahn
>
> **备注:** Published in NeurIPS 2024
>
> **摘要:** Recently, recurrent models based on linear state space models (SSMs) have shown promising performance in language modeling (LM), competititve with transformers. However, there is little understanding of the in-principle abilities of such models, which could provide useful guidance to the search for better LM architectures. We present a comprehensive theoretical study of the capacity of such SSMs as it compares to that of transformers and traditional RNNs. We find that SSMs and transformers have overlapping but distinct strengths. In star-free state tracking, SSMs implement straightforward and exact solutions to problems that transformers struggle to represent exactly. They can also model bounded hierarchical structure with optimal memory even without simulating a stack. On the other hand, we identify a design choice in current SSMs that limits their expressive power. We discuss implications for SSM and LM research, and verify results empirically on a recent SSM, Mamba.
>
---
#### [replaced 016] Joint Learning of Wording and Formatting for Singable Melody-to-Lyric Generation
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究旋律到歌词生成任务，旨在提升生成歌词的可唱性。通过联合学习用词与格式（如行数、音节数），引入音乐学启发的辅助目标，增强模型对韵律与结构的建模，显著提升格式准确率与人工评价得分。**

- **链接: [https://arxiv.org/pdf/2307.02146v3](https://arxiv.org/pdf/2307.02146v3)**

> **作者:** Longshen Ou; Xichu Ma; Ye Wang
>
> **备注:** An extension of our previous work arXiv:2305.16816 [cs.CL]
>
> **摘要:** Despite progress in melody-to-lyric generation, a substantial singability gap remains between machine-generated lyrics and those written by human lyricists. In this work, we aim to narrow this gap by jointly learning both wording and formatting for melody-to-lyric generation. After general-domain pretraining, our model acquires length awareness through an self-supervised stage trained on a large text-only lyric corpus. During supervised melody-to-lyric training, we introduce multiple auxiliary supervision objective informed by musicological findings on melody--lyric relationships, encouraging the model to capture fine-grained prosodic and structural patterns. Compared with naïve fine-tuning, our approach improves adherence to line-count and syllable-count requirements by 3.8% and 21.4% absolute, respectively, without degrading text quality. In human evaluation, it achieves 42.2% and 74.2% relative gains in overall quality over two task-specific baselines, underscoring the importance of formatting-aware training for generating singable lyrics.
>
---
#### [replaced 017] Large Language Model Agent for Modular Task Execution in Drug Discovery
- **分类: cs.LG; cs.CL; q-bio.BM**

- **简介: 该论文提出一种基于大语言模型的模块化智能体，用于自动化药物发现中的多步骤任务。它整合领域工具，完成分子生成、性质预测、优化及蛋白-配体结构构建，提升分子质量与药物相似性，支持高效、可扩展的AI辅助药物研发。**

- **链接: [https://arxiv.org/pdf/2507.02925v3](https://arxiv.org/pdf/2507.02925v3)**

> **作者:** Janghoon Ock; Radheesh Sharma Meda; Srivathsan Badrinarayanan; Neha S. Aluru; Achuth Chandrasekhar; Amir Barati Farimani
>
> **摘要:** We present a modular framework powered by large language models (LLMs) that automates and streamlines key tasks across the early-stage computational drug discovery pipeline. By combining LLM reasoning with domain-specific tools, the framework performs biomedical data retrieval, literature-grounded question answering via retrieval-augmented generation, molecular generation, multi-property prediction, property-aware molecular refinement, and 3D protein-ligand structure generation. The agent autonomously retrieved relevant biomolecular information, including FASTA sequences, SMILES representations, and literature, and answered mechanistic questions with improved contextual accuracy compared to standard LLMs. It then generated chemically diverse seed molecules and predicted 75 properties, including ADMET-related and general physicochemical descriptors, which guided iterative molecular refinement. Across two refinement rounds, the number of molecules with QED > 0.6 increased from 34 to 55. The number of molecules satisfying empirical drug-likeness filters also rose; for example, compliance with the Ghose filter increased from 32 to 55 within a pool of 100 molecules. The framework also employed Boltz-2 to generate 3D protein-ligand complexes and provide rapid binding affinity estimates for candidate compounds. These results demonstrate that the approach effectively supports molecular screening, prioritization, and structure evaluation. Its modular design enables flexible integration of evolving tools and models, providing a scalable foundation for AI-assisted therapeutic discovery.
>
---
#### [replaced 018] MapFormer: Self-Supervised Learning of Cognitive Maps with Input-Dependent Positional Embeddings
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出MapFormers，旨在通过输入依赖的位置编码，在Transformer中自监督学习认知地图，解决AI在出分布泛化和结构-内容解耦上的不足，实现类似生物的灵活导航能力。**

- **链接: [https://arxiv.org/pdf/2511.19279v2](https://arxiv.org/pdf/2511.19279v2)**

> **作者:** Victor Rambaud; Salvador Mascarenhas; Yair Lakretz
>
> **备注:** 19 pages (29 with appendix), 8 figures
>
> **摘要:** A cognitive map is an internal model which encodes the abstract relationships among entities in the world, giving humans and animals the flexibility to adapt to new situations, with a strong out-of-distribution (OOD) generalization that current AI systems still do not possess. To bridge this gap, we introduce MapFormers, new architectures based on Transformer models, which can learn cognitive maps from observational data and perform path integration in parallel, in a self-supervised manner. Cognitive maps are learned in the model by disentangling structural relationships in the inputs from their specific content, a property that can be achieved naturally by updating the positional encoding in Transformers with input-dependent matrices. We developed two variants of MapFormers that unify absolute and relative positional encoding to model episodic (EM) and working memory (WM), respectively. We tested MapFormers on several tasks, including a classic 2D navigation task, showing that our models can learn a cognitive map of the underlying space and generalize OOD (e.g., to longer sequences) with near-perfect performance, unlike current architectures. Together, these results demonstrate the superiority of models designed to learn a cognitive map, and the importance of introducing a structural bias for structure-content disentanglement, which can be achieved in Transformers with input-dependent positional encoding. MapFormers have broad applications in both neuroscience and AI, by explaining the neural mechanisms giving rise to cognitive maps, while allowing these relation models to be learned at scale.
>
---
#### [replaced 019] Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond
- **分类: cs.CL**

- **简介: 该论文研究LoRA的泛化能力与损失曲面平坦性的关系，提出FMLoRA和高效版EFMLoRA，通过理论证明将全参数空间扰动迁移至低秩子空间，实现对平坦极小值的优化，提升模型泛化性能。**

- **链接: [https://arxiv.org/pdf/2508.00522v2](https://arxiv.org/pdf/2508.00522v2)**

> **作者:** Jiaxin Deng; Qingcheng Zhu; Junbiao Pang; Linlin Yang; Zhongqian Fu; Baochang Zhang
>
> **摘要:** Little research explores the correlation between the expressive ability and generalization ability of the low-rank adaptation (LoRA). Sharpness-Aware Minimization (SAM) improves model generalization for both Convolutional Neural Networks (CNNs) and Transformers by encouraging convergence to locally flat minima. However, the connection between sharpness and generalization has not been fully explored for LoRA due to the lack of tools to either empirically seek flat minima or develop theoretical methods. In this work, we propose Flat Minima LoRA (FMLoRA) and its efficient version, i.e., EFMLoRA, to seek flat minima for LoRA. Concretely, we theoretically demonstrate that perturbations in the full parameter space can be transferred to the low-rank subspace. This approach eliminates the potential interference introduced by perturbations across multiple matrices in the low-rank subspace. Our extensive experiments on large language models and vision-language models demonstrate that EFMLoRA achieves optimize efficiency comparable to that of LoRA while simultaneously attaining comparable or even better performance. For example, on the GLUE dataset with RoBERTa-large, EFMLoRA outperforms LoRA and full fine-tuning by 1.0% and 0.5% on average, respectively. On vision-language models, e.g., Qwen-VL-Chat, there are performance improvements of 1.5% and 1.0% on the SQA and VizWiz datasets, respectively. These empirical results also verify that the generalization of LoRA is closely related to sharpness, which is omitted by previous methods.
>
---
#### [replaced 020] Confucius Code Agent: An Open-sourced AI Software Engineer at Industrial Scale
- **分类: cs.CL; cs.AI; cs.LG; cs.SE**

- **简介: 该论文聚焦AI软件工程代理任务，旨在解决开源代理在工业规模下能力不足的问题。作者提出Confucius SDK平台与基于其构建的CCA代理，通过分层记忆、持续学习和模块化工具协调，实现可扩展、透明且高性能的代码生成，在SWE-Bench-Pro上达到54.3%的SOTA性能。**

- **链接: [https://arxiv.org/pdf/2512.10398v2](https://arxiv.org/pdf/2512.10398v2)**

> **作者:** Zhaodong Wang; Zhenting Qi; Sherman Wong; Nathan Hu; Samuel Lin; Jun Ge; Erwin Gao; Yining Yang; Ben Maurer; Wenlin Chen; David Recordon; Yilun Du; Minlan Yu; Ying Zhang
>
> **备注:** Meta requires more thorough internal review process to ensure paper quality and experiments as well as compliance with the internal research publishing process
>
> **摘要:** Real-world AI software engineering demands coding agents that can reason over massive repositories, maintain durable memory across and within long sessions, and robustly coordinate complex toolchains at test time. Existing open-source coding agents provide transparency but frequently fall short when pushed to these industrial-scale workloads, while proprietary coding agents offer strong practical performance but limited extensibility, interpretability, and controllability. We present the Confucius Code Agent (CCA), an open-sourced AI software engineer that can operate at an industrial scale. CCA is built atop the Confucius SDK, an open-sourced agent development platform designed around three complementary perspectives: Agent Experience (AX), User Experience (UX), and Developer Experience (DX). The SDK introduces a unified orchestrator with hierarchical working memory for long-context reasoning, a persistent note-taking system for cross-session continual learning, and a modular extension module for robust tool use. Moreover, a meta-agent automates the synthesis, evaluation, and refinement of agent configurations through a build-test-improve loop, enabling rapid agent development on new tasks, environments, and tool stacks. Instantiated on Confucius SDK with these mechanisms, CCA delivers strong performance on real-world software engineering tasks. On SWE-Bench-Pro, CCA achieves a state-of-the-art Resolve@1 performance of 54.3%, substantially improving over prior coding agents. Together, the Confucius SDK and CCA provide a transparent, extensible, and reproducible foundation for AI agents, bridge gaps between research prototypes and production-grade systems, and support agent development and deployment at industrial scale.
>
---
#### [replaced 021] DeepSeek's WEIRD Behavior: The cultural alignment of Large Language Models and the effects of prompt language and cultural prompting
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的文化对齐问题，旨在分析不同模型在中美文化维度上的响应差异。通过 Hofstede 理论和文化提示策略，评估 DeepSeek 与 GPT 系列模型在不同提示语言和文化引导下的对齐表现。**

- **链接: [https://arxiv.org/pdf/2512.09772v2](https://arxiv.org/pdf/2512.09772v2)**

> **作者:** James Luther; Donald Brown
>
> **摘要:** Culture is a core component of human-to-human interaction and plays a vital role in how we perceive and interact with others. Advancements in the effectiveness of Large Language Models (LLMs) in generating human-sounding text have greatly increased the amount of human-to-computer interaction. As this field grows, the cultural alignment of these human-like agents becomes an important field of study. Our work uses Hofstede's VSM13 international surveys to understand the cultural alignment of the following models: DeepSeek-V3, V3.1, GPT-4, GPT-4.1, GPT-4o, and GPT-5. We use a combination of prompt language and cultural prompting, a strategy that uses a system prompt to shift a model's alignment to reflect a specific country, to align these LLMs with the United States and China. Our results show that DeepSeek-V3, V3.1, and OpenAI's GPT-5 exhibit a close alignment with the survey responses of the United States and do not achieve a strong or soft alignment with China, even when using cultural prompts or changing the prompt language. We also find that GPT-4 exhibits an alignment closer to China when prompted in English, but cultural prompting is effective in shifting this alignment closer to the United States. Other low-cost models, GPT-4o and GPT-4.1, respond to the prompt language used (i.e., English or Simplified Chinese) and cultural prompting strategies to create acceptable alignments with both the United States and China.
>
---
#### [replaced 022] Idea-Gated Transformers: Enforcing Semantic Coherence via Differentiable Vocabulary Pruning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言建模任务，旨在解决自回归模型因局部预测导致的主题漂移问题。作者提出“概念门控”机制，通过一个辅助“思想头”预测语义向量，动态抑制无关词元，增强生成的语义连贯性与领域保持能力。**

- **链接: [https://arxiv.org/pdf/2512.03343v2](https://arxiv.org/pdf/2512.03343v2)**

> **作者:** Darshan Fofadiya
>
> **备注:** Code available at https://github.com/DarshanFofadiya/idea-gated-transformers/tree/main
>
> **摘要:** Autoregressive Language Models (LLMs) trained on Next-Token Prediction (NTP) often suffer from Topic Drift where the generation wanders away from the initial prompt due to a reliance on local associations rather than global planning. While scaling model size mitigates this, the fundamental myopia of the NTP objective remains. In this work, we introduce the Idea-Gated Transformer, a novel architecture that separates semantic planning from syntactic generation. We introduce an auxiliary Idea Head trained to predict the bag-of-words distribution for a future context window, creating a latent ``Concept Vector'' that actively gates the main vocabulary during generation. We propose a differentiable gating mechanism that suppresses semantically irrelevant tokens, effectively pruning the search space in real-time. Experiments on WikiText-103 demonstrate that while the Idea-Gated model achieves comparable validation perplexity to a standard GPT-2 baseline, it exhibits significantly superior Domain Retention. Qualitative and quantitative analysis reveals that the gating mechanism successfully locks generation into specific semantic clusters (e.g., Finance, Science) and resists associative drift, offering a parameter-efficient path toward more controllable language modeling.
>
---
#### [replaced 023] Textual Self-attention Network: Test-Time Preference Optimization through Textual Gradient-based Attention
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型对齐任务，旨在解决测试时偏好优化中缺乏多候选方案综合分析的问题。作者提出TSAN，通过文本化自注意力机制，在无需参数更新的情况下，利用多个候选响应的优点生成更优输出。**

- **链接: [https://arxiv.org/pdf/2511.06682v2](https://arxiv.org/pdf/2511.06682v2)**

> **作者:** Shibing Mo; Haoyang Ruan; Kai Wu; Jing Liu
>
> **备注:** AAAI2026
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable generalization capabilities, but aligning their outputs with human preferences typically requires expensive supervised fine-tuning. Recent test-time methods leverage textual feedback to overcome this, but they often critique and revise a single candidate response, lacking a principled mechanism to systematically analyze, weigh, and synthesize the strengths of multiple promising candidates. Such a mechanism is crucial because different responses may excel in distinct aspects (e.g., clarity, factual accuracy, or tone), and combining their best elements may produce a far superior outcome. This paper proposes the Textual Self-Attention Network (TSAN), a new paradigm for test-time preference optimization that requires no parameter updates. TSAN emulates self-attention entirely in natural language to overcome this gap: it analyzes multiple candidates by formatting them into textual keys and values, weighs their relevance using an LLM-based attention module, and synthesizes their strengths into a new, preference-aligned response under the guidance of the learned textual attention. This entire process operates in a textual gradient space, enabling iterative and interpretable optimization. Empirical evaluations demonstrate that with just three test-time iterations on a base SFT model, TSAN outperforms supervised models like Llama-3.1-70B-Instruct and surpasses the current state-of-the-art test-time alignment method by effectively leveraging multiple candidate solutions.
>
---
#### [replaced 024] SDialog: A Python Toolkit for End-to-End Agent Building, User Simulation, Dialog Generation, and Evaluation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SDialog，一个统一的Python工具包，旨在解决LLM对话系统构建、仿真、生成与评估的集成问题。它提供标准化对话表示，支持多智能体模拟、综合评估、可解释性分析和音频仿真，促进对话系统的端到端研究与分析。**

- **链接: [https://arxiv.org/pdf/2506.10622v2](https://arxiv.org/pdf/2506.10622v2)**

> **作者:** Sergio Burdisso; Séverin Baroudi; Yanis Labrak; David Grunert; Pawel Cyrta; Yiyang Chen; Srikanth Madikeri; Esaú Villatoro-Tello; Thomas Schaaf; Ricard Marxer; Petr Motlicek
>
> **备注:** Pre-print submitted to EACL System Demonstration (under review)
>
> **摘要:** We present SDialog, an MIT-licensed open-source Python toolkit that unifies dialog generation, evaluation and mechanistic interpretability into a single end-to-end framework for building and analyzing LLM-based conversational agents. Built around a standardized \texttt{Dialog} representation, SDialog provides: (1) persona-driven multi-agent simulation with composable orchestration for controlled, synthetic dialog generation, (2) comprehensive evaluation combining linguistic metrics, LLM-as-a-judge and functional correctness validators, (3) mechanistic interpretability tools for activation inspection and steering via feature ablation and induction, and (4) audio generation with full acoustic simulation including 3D room modeling and microphone effects. The toolkit integrates with all major LLM backends, enabling mixed-backend experiments under a unified API. By coupling generation, evaluation, and interpretability in a dialog-centric architecture, SDialog enables researchers to build, benchmark and understand conversational systems more systematically.
>
---
#### [replaced 025] LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大语言模型在推理过程中难以整体修正思维链的问题，提出LaDiR框架。通过变分自编码器构建隐式思维空间，结合潜在扩散模型实现迭代优化，提升文本推理的准确性、多样性和可解释性。**

- **链接: [https://arxiv.org/pdf/2510.04573v4](https://arxiv.org/pdf/2510.04573v4)**

> **作者:** Haoqiang Kang; Yizhe Zhang; Nikki Lijing Kuang; Nicklas Majamaki; Navdeep Jaitly; Yi-An Ma; Lianhui Qin
>
> **摘要:** Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design allows efficient parallel generation of diverse reasoning trajectories, allowing the model to plan and revise the reasoning process holistically. We conduct evaluations on a suite of mathematical reasoning and planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion.
>
---
#### [replaced 026] Textual Data Bias Detection and Mitigation -- An Extensible Pipeline with Experimental Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦文本数据中的偏见检测与缓解，属数据去偏任务。针对表征偏见和显式刻板印象，提出包含四组件的可扩展流水线，结合LLM生成词表、量化评分、语言学过滤与数据增强，在性别、宗教、年龄上实验验证了数据去偏有效性，但发现模型偏见评测仍存方法论缺口。**

- **链接: [https://arxiv.org/pdf/2512.10734v2](https://arxiv.org/pdf/2512.10734v2)**

> **作者:** Rebekka Görge; Sujan Sai Gannamaneni; Tabea Naeven; Hammam Abdelwahab; Héctor Allende-Cid; Armin B. Cremers; Lennard Helmer; Michael Mock; Anna Schmitz; Songkai Xue; Elif Yildirir; Maximilian Poretschkin; Stefan Wrobel
>
> **摘要:** Textual data used to train large language models (LLMs) exhibits multifaceted bias manifestations encompassing harmful language and skewed demographic distributions. Regulations such as the European AI Act require identifying and mitigating biases against protected groups in data, with the ultimate goal of preventing unfair model outputs. However, practical guidance and operationalization are lacking. We propose a comprehensive data bias detection and mitigation pipeline comprising four components that address two data bias types, namely representation bias and (explicit) stereotypes for a configurable sensitive attribute. First, we leverage LLM-generated word lists created based on quality criteria to detect relevant group labels. Second, representation bias is quantified using the Demographic Representation Score. Third, we detect and mitigate stereotypes using sociolinguistically informed filtering. Finally, we compensate representation bias through Grammar- and Context-Aware Counterfactual Data Augmentation. We conduct a two-fold evaluation using the examples of gender, religion and age. First, the effectiveness of each individual component on data debiasing is evaluated through human validation and baseline comparison. The findings demonstrate that we successfully reduce representation bias and (explicit) stereotypes in a text dataset. Second, the effect of data debiasing on model bias reduction is evaluated by bias benchmarking of several models (0.6B-8B parameters), fine-tuned on the debiased text dataset. This evaluation reveals that LLMs fine-tuned on debiased data do not consistently show improved performance on bias benchmarks, exposing critical gaps in current evaluation methodologies and highlighting the need for targeted data manipulation to address manifested model bias.
>
---
#### [replaced 027] Sorting the Babble in Babel: Assessing the Performance of Language Identification Algorithms on the OpenAlex Database
- **分类: cs.CL**

- **简介: 该论文属于语言识别任务，旨在优化OpenAlex数据库的语种标注。通过比较多种算法在不同文本片段上的精度、召回率和速度，发现FastSpell在标题上表现最优，尤其在重视召回或效率时。**

- **链接: [https://arxiv.org/pdf/2502.03627v3](https://arxiv.org/pdf/2502.03627v3)**

> **作者:** Maxime Holmberg Sainte-Marie; Diego Kozlowski; Lucía Céspedes; Vincent Larivière
>
> **备注:** 32 pages, 4 figures
>
> **摘要:** This project aims to optimize the linguistic indexing of the OpenAlex database by comparing the performance of various Python-based language identification procedures on different metadata corpora extracted from a manually-annotated article sample. The precision and recall performance of each algorithm, corpus, and language is first analyzed, followed by an assessment of processing speeds recorded for each algorithm and corpus type. These different performance measures are then simulated at the database level using probabilistic confusion matrices for each algorithm, corpus, and language, as well as a probabilistic modeling of relative article language frequencies for the whole OpenAlex database. Results show that procedure performance strongly depends on the importance given to each of the measures implemented: for contexts where precision is preferred, using the LangID algorithm on the greedy corpus gives the best results; however, for all cases where recall is considered at least slightly more important than precision or as soon as processing times are given any kind of consideration, the procedure that consists in the application of the FastSpell algorithm on the Titles corpus outperforms all other alternatives. Given the lack of truly multilingual large-scale bibliographic databases, it is hoped that these results help confirm and foster the unparalleled potential of the OpenAlex database for cross-linguistic and comprehensive measurement and evaluation.
>
---
#### [replaced 028] Enhancing Instruction-Following Capabilities in Seq2Seq Models: DoLA Adaptations for T5
- **分类: cs.CL**

- **简介: 该论文针对Seq2Seq模型（如FLAN-T5）在指令冲突时生成不稳定的问题，研究其解码器表示演化，提出基于梯度的激活引导方法，在中间层注入“指令遵从”方向，显著提升指令遵循能力，尤其在MemoTrap任务上效果显著。**

- **链接: [https://arxiv.org/pdf/2512.03803v2](https://arxiv.org/pdf/2512.03803v2)**

> **作者:** Huey Sun; Anabel Yong; Lorenzo Gilly; Felipe Jin
>
> **摘要:** Encoder-decoder models such as FLAN-T5 are finetuned to follow instructions, but often fail when the instructions conflict with memorized continuations ingrained during training. To understand this behavior, we adapt DoLa to FLAN-T5 and examine how representations evolve in the decoder. Our findings show that T5's intermediate layers undergo rapid shifts driven by cross-attention to the encoder. When projected through the language modeling head, each depth presents highly volatile token preferences, leading to unreliable behavior with contrastive decoding. Motivated by this, we introduce a gradient-based activation-steering method that injects an "instruction-compliance" direction into mid-decoder layers, where the representation is both meaningful and still malleable. This intervention dramatically improves MemoTrap performance (52% to 99.7%), demonstrating that mechanistic steering can succeed where contrastive decoding fails in Seq2Seq architectures.
>
---
#### [replaced 029] Empirical Analysis of the Effect of Context in the Task of Automated Essay Scoring in Transformer-Based Models
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于自动作文评分（AES）任务，旨在解决Transformer模型在此任务中表现不足的问题。通过引入多种上下文因素增强模型，提升了评分性能，尤其在部分数据集上接近甚至超过现有最佳模型，且方法可通用。**

- **链接: [https://arxiv.org/pdf/2508.16638v2](https://arxiv.org/pdf/2508.16638v2)**

> **作者:** Abhirup Chakravarty
>
> **备注:** MSc Dissertation
>
> **摘要:** Automated Essay Scoring (AES) has emerged to prominence in response to the growing demand for educational automation. Providing an objective and cost-effective solution, AES standardises the assessment of extended responses. Although substantial research has been conducted in this domain, recent investigations reveal that alternative deep-learning architectures outperform transformer-based models. Despite the successful dominance in the performance of the transformer architectures across various other tasks, this discrepancy has prompted a need to enrich transformer-based AES models through contextual enrichment. This study delves into diverse contextual factors using the ASAP-AES dataset, analysing their impact on transformer-based model performance. Our most effective model, augmented with multiple contextual dimensions, achieves a mean Quadratic Weighted Kappa score of 0.823 across the entire essay dataset and 0.8697 when trained on individual essay sets. Evidently surpassing prior transformer-based models, this augmented approach only underperforms relative to the state-of-the-art deep learning model trained essay-set-wise by an average of 3.83\% while exhibiting superior performance in three of the eight sets. Importantly, this enhancement is orthogonal to architecture-based advancements and seamlessly adaptable to any AES model. Consequently, this contextual augmentation methodology presents a versatile technique for refining AES capabilities, contributing to automated grading and evaluation evolution in educational settings.
>
---
#### [replaced 030] The Landscape of Memorization in LLMs: Mechanisms, Measurement, and Mitigation
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文综述大语言模型中的记忆现象，属技术与隐私交叉任务。旨在分析训练数据记忆的成因、检测与缓解方法，探讨其机制、测量手段及伦理风险，提出通过数据清洗、差分隐私等策略降低隐私泄露，平衡模型性能与安全。**

- **链接: [https://arxiv.org/pdf/2507.05578v2](https://arxiv.org/pdf/2507.05578v2)**

> **作者:** Alexander Xiong; Xuandong Zhao; Aneesh Pappu; Dawn Song
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet they also exhibit memorization of their training data. This phenomenon raises critical questions about model behavior, privacy risks, and the boundary between learning and memorization. Addressing these concerns, this paper synthesizes recent studies and investigates the landscape of memorization, the factors influencing it, and methods for its detection and mitigation. We explore key drivers, including training data duplication, training dynamics, and fine-tuning procedures that influence data memorization. In addition, we examine methodologies such as prefix-based extraction, membership inference, and adversarial prompting, assessing their effectiveness in detecting and measuring memorized content. Beyond technical analysis, we also explore the broader implications of memorization, including the legal and ethical implications. Finally, we discuss mitigation strategies, including data cleaning, differential privacy, and post-training unlearning, while highlighting open challenges in balancing the need to minimize harmful memorization with model utility. This paper provides a comprehensive overview of the current state of research on LLM memorization across technical, privacy, and performance dimensions, identifying critical directions for future work.
>
---
#### [replaced 031] Annotation-Free Reinforcement Learning Query Rewriting via Verifiable Search Reward
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文研究检索增强生成（RAG）中的查询重写任务，旨在解决依赖人工标注数据的问题。提出RL-QR框架，利用可验证的搜索奖励实现免标注强化学习，提升多模态检索性能。**

- **链接: [https://arxiv.org/pdf/2507.23242v2](https://arxiv.org/pdf/2507.23242v2)**

> **作者:** Sungguk Cha; DongWook Kim; Taeseung Hahn; Mintae Kim; Youngsub Han; Byoung-Ki Jeon
>
> **摘要:** Optimizing queries for Retrieval-Augmented Generation (RAG) systems poses a significant challenge, particularly across diverse modal indices. We introduce RL-QR, a novel annotation-free reinforcement learning framework for query rewriting that eliminates the need for costly human-annotated data. By leveraging verifiable search rewards derived from index-aligned synthetic queries, RL-QR overcomes human-annotation dependencies, extending its applicability to various modalities and index domains. Experimental results demonstrate the framework's robustness, achieving substantial retrieval performance gains of up to 3.9$\times$ on lexical retrievers and 3.5$\times$ on semantic retrievers on the MTEB VIDORE V2 benchmark for unstructured visual documents, along with consistent 5\% to 10\% improvements on MS MARCO v2.1 and internal industrial datasets.
>
---
#### [replaced 032] The Illusion of Readiness in Health AI
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于AI医疗应用评估任务，旨在揭示大模型在医学基准测试中的表面性能与实际可靠性间的差距。作者通过对抗性压力测试，发现模型对输入扰动敏感、推理不稳健，并指出当前基准无法准确衡量真实医疗能力，强调需更严格评估以确保AI在医疗场景的可信与安全。**

- **链接: [https://arxiv.org/pdf/2509.18234v3](https://arxiv.org/pdf/2509.18234v3)**

> **作者:** Yu Gu; Jingjing Fu; Xiaodong Liu; Jeya Maria Jose Valanarasu; Noel CF Codella; Reuben Tan; Qianchu Liu; Ying Jin; Sheng Zhang; Jinyu Wang; Rui Wang; Lei Song; Guanghui Qin; Naoto Usuyama; Cliff Wong; Hao Cheng; HoHin Lee; Praneeth Sanapathi; Sarah Hilado; Tristan Naumann; Javier Alvarez-Valle; Jiang Bian; Mu Wei; Khalil Malik; Lidong Zhou; Jianfeng Gao; Eric Horvitz; Matthew P. Lungren; Doug Burger; Eric Topol; Hoifung Poon; Paul Vozila
>
> **摘要:** Large language models have demonstrated remarkable performance in a wide range of medical benchmarks. Yet underneath the seemingly promising results lie salient growth areas, especially in cutting-edge frontiers such as multimodal reasoning. In this paper, we introduce a series of adversarial stress tests to systematically assess the robustness of flagship models and medical benchmarks. Our study reveals prevalent brittleness in the presence of simple adversarial transformations: leading systems can guess the right answer even with key inputs removed, yet may get confused by the slightest prompt alterations, while fabricating convincing yet flawed reasoning traces. Using clinician-guided rubrics, we demonstrate that popular medical benchmarks vary widely in what they truly measure. Our study reveals significant competency gaps of frontier AI in attaining real-world readiness for health applications. If we want AI to earn trust in healthcare, we must demand more than leaderboard wins and must hold AI systems accountable to ensure robustness, sound reasoning, and alignment with real medical demands.
>
---
#### [replaced 033] MOAT: Evaluating LMMs for Capability Integration and Instruction Grounding
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出MOAT基准，评估大视觉语言模型在复杂现实任务中的能力整合与指令 grounding 问题。针对现有模型在多能力协同和复杂指令理解上的不足，构建了含1005题的测试集，揭示当前最优模型准确率仅44%，并分析瓶颈原因，推动模型改进。**

- **链接: [https://arxiv.org/pdf/2503.09348v2](https://arxiv.org/pdf/2503.09348v2)**

> **作者:** Zhoutong Ye; Mingze Sun; Huan-ang Gao; Xutong Wang; Xiangyang Wang; Yu Mei; Chang Liu; Qinwei Li; Chengwen Zhang; Qinghuan Lan; Chun Yu; Yuanchun Shi
>
> **备注:** Project page: https://cambrian-yzt.github.io/MOAT
>
> **摘要:** Large multimodal models (LMMs) have demonstrated significant potential as generalists in vision-language (VL) tasks. However, adoption of LMMs in real-world tasks is hindered by their poor performance in tasks that require a combination of VL capabilities, as well as in tasks that involve the grounding of complex text or visual instructions. To thoroughly investigate this gap and its underlying causes, we propose MOAT, a diverse benchmark with 1005 complex real-world vision questions that are straightforward for humans but challenging for LMMs. Specifically, the tasks in MOAT require LMMs to engage in generalist problem solving by integrating VL capabilities such as reading text, counting, understanding spatial relations, grounding textual and visual instructions, etc. All these abilities fit into a taxonomy proposed by us that contains 9 VL capabilities, enabling MOAT to provide a fine-grained view of LMMs' strengths and weaknesses. Besides, MOAT is the first benchmark to explicitly evaluate LMMs' ability to ground complex text and visual instructions, which is essential for many real-world applications. We evaluated 17 proprietary and open source LMMs, finding that the best performing LMM (Gemini 2.5 Pro) achieved only 44% accuracy, far below what would be acceptable in real-world applications. To guide future model development, we analyze common trends in our results and discuss the underlying causes of poor performance, focusing on the impact of text-centric reasoning, which VL capabilities form bottlenecks in complex tasks, and the potential harmful effects of tiling. Code and data are available at https://cambrian-yzt.github.io/MOAT/.
>
---
#### [replaced 034] HyperAdaLoRA: Accelerating LoRA Rank Allocation During Training via Hypernetworks without Sacrificing Performance
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于参数高效微调任务，旨在解决AdaLoRA训练收敛慢、计算开销高的问题。作者提出HyperAdaLoRA，利用基于注意力的超网络动态生成SVD参数，并通过超网络输出剪枝实现动态秩分配，加速收敛且不损失性能。**

- **链接: [https://arxiv.org/pdf/2510.02630v2](https://arxiv.org/pdf/2510.02630v2)**

> **作者:** Hao Zhang; Zhenjia Li; Runfeng Bao; Yifan Gao; Xi Xiao; Heng Zhang; Shuyang Zhang; Bo Huang; Yuhang Wu; Tianyang Wang; Hao Xu
>
> **备注:** 13 pages
>
> **摘要:** Parameter-Efficient Fine-Tuning (PEFT), especially Low-Rank Adaptation (LoRA), has emerged as a promising approach to fine-tuning large language models(LLMs) while reducing computational and memory overhead. However, LoRA assumes a uniform rank \textit{r} for each incremental matrix, not accounting for the varying significance of weight matrices across different modules and layers. AdaLoRA leverages Singular Value Decomposition (SVD) to parameterize updates and employs pruning of singular values to introduce dynamic rank allocation, thereby enhancing adaptability. However, during the training process, it often encounters issues of slow convergence speed and high computational overhead. To address this issue, we propose HyperAdaLoRA, a novel framework that accelerates the convergence of AdaLoRA by leveraging a hypernetwork. Instead of directly optimizing the components of Singular Value Decomposition $(P, Λ, Q)$, HyperAdaLoRA employs a hypernetwork based on attention mechanisms to dynamically generate these parameters. By pruning the outputs of the hypernetwork that generates the singular values, dynamic rank allocation is achieved. Comprehensive experiments on various datasets and models demonstrate that our method achieves faster convergence without sacrificing performance. Additionally, further extension experiments on other LoRA-based approaches validate the broad applicability of our method.
>
---
#### [replaced 035] Grammar-Aligned Decoding
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究语法约束下的文本生成任务，解决传统约束解码扭曲语言模型分布导致生成质量低的问题。提出ASAp算法，通过自适应采样和未来概率估计，确保输出符合语法且保持模型原始条件概率，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2405.21047v3](https://arxiv.org/pdf/2405.21047v3)**

> **作者:** Kanghee Park; Jiayu Wang; Taylor Berg-Kirkpatrick; Nadia Polikarpova; Loris D'Antoni
>
> **备注:** Accepted to NeurIPS 2024
>
> **摘要:** Large Language Models (LLMs) struggle with reliably generating highly structured outputs, such as program code, mathematical formulas, or well-formed markup. Constrained decoding approaches mitigate this problem by greedily restricting what tokens an LLM can output at each step to guarantee that the output matches a given constraint. Specifically, in grammar-constrained decoding (GCD), the LLM's output must follow a given grammar. In this paper, we demonstrate that GCD techniques (and in general constrained decoding techniques) can distort the LLM's distribution, leading to outputs that are grammatical but appear with likelihoods that are not proportional to the ones given by the LLM, and so ultimately are low-quality. We call the problem of aligning sampling with a grammar constraint, grammar-aligned decoding (GAD), and propose adaptive sampling with approximate expected futures (ASAp), a decoding algorithm that guarantees the output to be grammatical while provably producing outputs that match the conditional probability of the LLM's distribution conditioned on the given grammar constraint. Our algorithm uses prior sample outputs to soundly overapproximate the future grammaticality of different output prefixes. Our evaluation on code generation and structured NLP tasks shows how ASAp often produces outputs with higher likelihood (according to the LLM's distribution) than existing GCD techniques, while still enforcing the desired grammatical constraints.
>
---
#### [replaced 036] Less Is More for Multi-Step Logical Reasoning of LLM Generalisation Under Rule Removal, Paraphrasing, and Compression
- **分类: cs.AI; cs.CL; cs.LG; cs.LO**

- **简介: 该论文研究大语言模型在逻辑规则扰动下的多步推理泛化能力。通过规则删除、矛盾注入和逻辑等价改写等压力测试，评估模型对结构变化的鲁棒性，揭示其在证据缺失或逻辑复合变换时的脆弱性，并提出一个诊断框架以检验深层逻辑一致性。**

- **链接: [https://arxiv.org/pdf/2512.06393v2](https://arxiv.org/pdf/2512.06393v2)**

> **作者:** Qiming Bao; Xiaoxuan Fu
>
> **摘要:** Large language models (LLMs) achieve strong performance on many natural language tasks, yet their generalisation under structured perturbations of logical rule systems remains insufficiently characterised. We present a controlled evaluation framework that probes reasoning reliability through four stress tests: (1) rule deletion, removing redundant versus essential rules from a multi-step inference chain; (2) contradictory evidence injection; (3) logic-preserving rewrites based on equivalence laws (contraposition, double negation, implication-to-disjunction, De Morgan, identity, and commutativity); and (4) multi-law equivalence stacking that composes 2--5 transformations. Across three representative model families -- BERT, Qwen2, and LLaMA-like models -- all models attain Acc$=1.0000$ on the base split and show no degradation under redundant rule deletion. In contrast, essential rule deletion yields a pronounced decrease to near-chance performance, and injecting explicit contradictions reduces accuracy to 0.0000. Under logic-preserving rewrites, accuracy is largely preserved for single-law transformations with only small degradations in a few cases, whereas multi-law stacking exposes model-dependent sensitivity: BERT matches the base condition, TinyLlama shows only marginal degradation, and Qwen2 exhibits a substantial drop. Overall, the results indicate that contemporary LLMs are generally stable under semantic-preserving reformulations, yet remain brittle to missing or inconsistent evidence and may degrade under composed logical transformations depending on the model family. The proposed framework provides a concise diagnostic tool for isolating these failure modes and for evaluating logical generalisation beyond surface-form variation.
>
---
#### [replaced 037] Uncertainty Distillation: Teaching Language Models to Express Semantic Confidence
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型不确定性量化任务，旨在解决大模型在问答中表达的置信度与实际错误率不匹配的问题。作者提出“不确定性蒸馏”方法，通过监督微调教会模型表达校准后的语义置信度，使置信度更准确且高效，适用于黑箱模型。**

- **链接: [https://arxiv.org/pdf/2503.14749v3](https://arxiv.org/pdf/2503.14749v3)**

> **作者:** Sophia Hager; David Mueller; Kevin Duh; Nicholas Andrews
>
> **摘要:** As large language models (LLMs) are increasingly used for factual question-answering, it becomes more important for LLMs to have the capability to communicate the likelihood that their answer is correct. For these verbalized expressions of uncertainty to be meaningful, they should reflect the error rates at the expressed level of confidence. However, when prompted to express confidence, the error rates of current LLMs are inconsistent with their communicated confidences, highlighting the need for uncertainty quantification methods. Many prior methods calculate lexical uncertainty, estimating a model's confidence in the specific string it generated. In some cases, however, it may be more useful to estimate semantic uncertainty, or the model's confidence in the answer regardless of how it is verbalized. We propose a simple procedure, uncertainty distillation, to teach an LLM to verbalize calibrated semantic confidences. Using held-out data to map initial uncertainty estimates to meaningful probabilities, we create examples annotated with verbalized probabilities for supervised fine-tuning. We find that our method yields verbalized confidences that correlate well with observed error rates, even when compared to strong baselines, some of which are more than twenty times slower at inference time. Additionally, we demonstrate that our method can be applied to black-box models that allow API-based fine-tuning, resulting in estimates of uncertainty that are both more effective and more efficient than any of our baselines.
>
---
#### [replaced 038] RECAP: REwriting Conversations for Intent Understanding in Agentic Planning
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦开放域对话中的意图理解任务，旨在解决现实对话中意图模糊、动态变化导致的规划困难问题。作者提出RECAP基准和LLM评估器，构建数据集并设计基于提示和微调的重写方法，提升代理规划效果。**

- **链接: [https://arxiv.org/pdf/2509.04472v2](https://arxiv.org/pdf/2509.04472v2)**

> **作者:** Kushan Mitra; Dan Zhang; Hannah Kim; Estevam Hruschka
>
> **摘要:** Understanding user intent is essential for effective planning in conversational assistants, particularly those powered by large language models (LLMs) coordinating multiple agents. However, real-world dialogues are often ambiguous, underspecified, or dynamic, making intent detection a persistent challenge. Traditional classification-based approaches struggle to generalize in open-ended settings, leading to brittle interpretations and poor downstream planning. We propose RECAP (REwriting Conversations for Agent Planning), a new benchmark designed to evaluate and advance intent rewriting, reframing user-agent dialogues into concise representations of user goals. RECAP captures diverse challenges such as ambiguity, intent drift, vagueness, and mixed-goal conversations. Alongside the dataset, we introduce an LLM-based evaluator that assesses planning utility given the rewritten intent. Using RECAP, we develop a prompt-based rewriting approach that outperforms baselines, in terms of plan preference. We further demonstrate that fine-tuning two DPO-based rewriters yields additional utility gains. Our results highlight intent rewriting as a critical and tractable component for improving agentic planning in open-domain dialogue systems.
>
---
