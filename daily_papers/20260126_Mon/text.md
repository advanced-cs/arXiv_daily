# 自然语言处理 cs.CL

- **最新发布 66 篇**

- **更新 47 篇**

## 最新发布

#### [new 001] Attention-MoA: Enhancing Mixture-of-Agents via Inter-Agent Semantic Attention and Deep Residual Synthesis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决MoA框架中代理间语义交互不足的问题。提出Attention-MoA，通过语义注意力和残差模块提升协作效果与效率。**

- **链接: [https://arxiv.org/pdf/2601.16596v1](https://arxiv.org/pdf/2601.16596v1)**

> **作者:** Jianyu Wen; Yang Wei; Xiongxi Yu; Changxuan Xiao; Ke Zeng
>
> **摘要:** As the development of Large Language Models (LLMs) shifts from parameter scaling to inference-time collaboration, the Mixture-of-Agents (MoA) framework has emerged as a general paradigm to harness collective intelligence by layering diverse models. While recent MoA variants have introduced dynamic routing and residual connections to improve efficiency, these methods often fail to facilitate deep semantic interaction between agents, limiting the system's ability to actively correct hallucinations and refine logic. In this paper, we introduce Attention-MoA, a novel MoA-based framework that redefines collaboration through Inter-agent Semantic Attention. Complemented by an Inter-layer Residual Module with Adaptive Early Stopping Mechanism, our architecture mitigates information degradation in deep layers while improving computational efficiency. Extensive evaluations across AlpacaEval 2.0, MT-Bench, and FLASK demonstrate that Attention-MoA significantly outperforms state-of-the-art baselines, achieving a 91.15% Length-Controlled Win Rate on AlpacaEval 2.0 and dominating in 10 out of 12 capabilities on FLASK. Notably, Attention-MoA enables an ensemble of small open-source models to outperform massive proprietary models like Claude-4.5-Sonnet and GPT-4.1, achieving an MT-Bench score of 8.83 and an AlpacaEval 2.0 LC Win Rate of 77.36%.
>
---
#### [new 002] Typologically Informed Parameter Aggregation
- **分类: cs.CL**

- **简介: 该论文提出TIPA方法，解决低资源语言在多语言模型中的性能问题。通过类型学相似性聚合现有适配器，实现零样本跨语言迁移，无需额外训练。**

- **链接: [https://arxiv.org/pdf/2601.16629v1](https://arxiv.org/pdf/2601.16629v1)**

> **作者:** Stef Accou; Wessel Poelman
>
> **备注:** EACL 2026: Findings
>
> **摘要:** Massively multilingual language models enable cross-lingual generalization but underperform on low-resource and unseen languages. While adapter-based fine-tuning offers a parameter-efficient solution, training language-specific adapters at scale remains costly. We introduce Typologically Informed Parameter Aggregation (TIPA), a training-free method that constructs proxy language adapters by aggregating existing ones, weighted by typological similarity. Integrated into the MAD-X framework, these proxies enable zero-shot cross-lingual transfer without additional training. We evaluate TIPA on five NLP tasks and over 230 languages. TIPA consistently outperforms or matches baselines such as English-only fine-tuning or selecting the typologically closest language adapter. We see the largest gains for languages lacking dedicated adapters. Our results demonstrate that typologically informed aggregation provides a viable alternative to language-specific modules without any training needed.
>
---
#### [new 003] SoS: Analysis of Surface over Semantics in Multilingual Text-To-Image Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言文本到图像生成中的表面优先现象（SoS），分析模型对不同语言表面形式的敏感性，揭示其导致刻板视觉表现的问题。**

- **链接: [https://arxiv.org/pdf/2601.16803v1](https://arxiv.org/pdf/2601.16803v1)**

> **作者:** Carolin Holtermann; Florian Schneider; Anne Lauscher
>
> **摘要:** Text-to-image (T2I) models are increasingly employed by users worldwide. However, prior research has pointed to the high sensitivity of T2I towards particular input languages - when faced with languages other than English (i.e., different surface forms of the same prompt), T2I models often produce culturally stereotypical depictions, prioritizing the surface over the prompt's semantics. Yet a comprehensive analysis of this behavior, which we dub Surface-over-Semantics (SoS), is missing. We present the first analysis of T2I models' SoS tendencies. To this end, we create a set of prompts covering 171 cultural identities, translated into 14 languages, and use it to prompt seven T2I models. To quantify SoS tendencies across models, languages, and cultures, we introduce a novel measure and analyze how the tendencies we identify manifest visually. We show that all but one model exhibit strong surface-level tendency in at least two languages, with this effect intensifying across the layers of T2I text encoders. Moreover, these surface tendencies frequently correlate with stereotypical visual depictions.
>
---
#### [new 004] Standardizing Longitudinal Radiology Report Evaluation via Large Language Model Annotation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学文本分析任务，旨在解决放射报告中纵向信息标注困难的问题。通过构建LLM标注流程，提升标注效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.16753v1](https://arxiv.org/pdf/2601.16753v1)**

> **作者:** Xinyi Wang; Grazziela Figueredo; Ruizhe Li; Xin Chen
>
> **摘要:** Longitudinal information in radiology reports refers to the sequential tracking of findings across multiple examinations over time, which is crucial for monitoring disease progression and guiding clinical decisions. Many recent automated radiology report generation methods are designed to capture longitudinal information; however, validating their performance is challenging. There is no proper tool to consistently label temporal changes in both ground-truth and model-generated texts for meaningful comparisons. Existing annotation methods are typically labor-intensive, relying on the use of manual lexicons and rules. Complex rules are closed-source, domain specific and hard to adapt, whereas overly simple ones tend to miss essential specialised information. Large language models (LLMs) offer a promising annotation alternative, as they are capable of capturing nuanced linguistic patterns and semantic similarities without extensive manual intervention. They also adapt well to new contexts. In this study, we therefore propose an LLM-based pipeline to automatically annotate longitudinal information in radiology reports. The pipeline first identifies sentences containing relevant information and then extracts the progression of diseases. We evaluate and compare five mainstream LLMs on these two tasks using 500 manually annotated reports. Considering both efficiency and performance, Qwen2.5-32B was subsequently selected and used to annotate another 95,169 reports from the public MIMIC-CXR dataset. Our Qwen2.5-32B-annotated dataset provided us with a standardized benchmark for evaluating report generation models. Using this new benchmark, we assessed seven state-of-the-art report generation models. Our LLM-based annotation method outperforms existing annotation solutions, achieving 11.3\% and 5.3\% higher F1-scores for longitudinal information detection and disease tracking, respectively.
>
---
#### [new 005] Mixing Expert Knowledge: Bring Human Thoughts Back To the Game of Go
- **分类: cs.CL**

- **简介: 该论文属于人工智能领域，旨在解决通用大模型在专业领域（如围棋）表现不足的问题。通过混合微调与强化学习，提升模型在围棋领域的推理和决策能力。**

- **链接: [https://arxiv.org/pdf/2601.16447v1](https://arxiv.org/pdf/2601.16447v1)**

> **作者:** Yichuan Ma; Linyang Li; Yongkang Chen; Peiji Li; Jiasheng Ye; Qipeng Guo; Dahua Lin; Kai Chen
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Large language models (LLMs) have demonstrated exceptional performance in reasoning tasks such as mathematics and coding, matching or surpassing human capabilities. However, these impressive reasoning abilities face significant challenges in specialized domains. Taking Go as an example, although AlphaGo has established the high performance ceiling of AI systems in Go, mainstream LLMs still struggle to reach even beginner-level proficiency, let alone perform natural language reasoning. This performance gap between general-purpose LLMs and domain experts is significantly limiting the application of LLMs on a wider range of domain-specific tasks. In this work, we aim to bridge the divide between LLMs' general reasoning capabilities and expert knowledge in domain-specific tasks. We perform mixed fine-tuning with structured Go expertise and general long Chain-of-Thought (CoT) reasoning data as a cold start, followed by reinforcement learning to integrate expert knowledge in Go with general reasoning capabilities. Through this methodology, we present \textbf{LoGos}, a powerful LLM that not only maintains outstanding general reasoning abilities, but also conducts Go gameplay in natural language, demonstrating effective strategic reasoning and accurate next-move prediction. LoGos achieves performance comparable to human professional players, substantially surpassing all existing LLMs. Through this work, we aim to contribute insights on applying general LLM reasoning capabilities to specialized domains. We will release the first large-scale Go dataset for LLM training, the first LLM Go evaluation benchmark, and the first general LLM that reaches human professional-level performance in Go at: https://github.com/Entarochuan/LoGos.
>
---
#### [new 006] Identity, Cooperation and Framing Effects within Groups of Real and Simulated Humans
- **分类: cs.CL**

- **简介: 该论文属于人工智能与社会行为研究任务，旨在提升大语言模型对人类行为的模拟精度。通过深度绑定模型与丰富背景，增强身份和情境因素的模拟，以更真实地再现人类决策过程。**

- **链接: [https://arxiv.org/pdf/2601.16355v1](https://arxiv.org/pdf/2601.16355v1)**

> **作者:** Suhong Moon; Minwoo Kang; Joseph Suh; Mustafa Safdari; John Canny
>
> **摘要:** Humans act via a nuanced process that depends both on rational deliberation and also on identity and contextual factors. In this work, we study how large language models (LLMs) can simulate human action in the context of social dilemma games. While prior work has focused on "steering" (weak binding) of chat models to simulate personas, we analyze here how deep binding of base models with extended backstories leads to more faithful replication of identity-based behaviors. Our study has these findings: simulation fidelity vs human studies is improved by conditioning base LMs with rich context of narrative identities and checking consistency using instruction-tuned models. We show that LLMs can also model contextual factors such as time (year that a study was performed), question framing, and participant pool effects. LLMs, therefore, allow us to explore the details that affect human studies but which are often omitted from experiment descriptions, and which hamper accurate replication.
>
---
#### [new 007] Large Language Models as Automatic Annotators and Annotation Adjudicators for Fine-Grained Opinion Analysis
- **分类: cs.CL**

- **简介: 该论文属于细粒度情感分析任务，旨在解决标注数据稀缺问题。通过使用大语言模型作为自动标注器和仲裁者，提高标注效率与一致性。**

- **链接: [https://arxiv.org/pdf/2601.16800v1](https://arxiv.org/pdf/2601.16800v1)**

> **作者:** Gaurav Negi; MA Waskow; Paul Buitelaar
>
> **摘要:** Fine-grained opinion analysis of text provides a detailed understanding of expressed sentiments, including the addressed entity. Although this level of detail is sound, it requires considerable human effort and substantial cost to annotate opinions in datasets for training models, especially across diverse domains and real-world applications. We explore the feasibility of LLMs as automatic annotators for fine-grained opinion analysis, addressing the shortage of domain-specific labelled datasets. In this work, we use a declarative annotation pipeline. This approach reduces the variability of manual prompt engineering when using LLMs to identify fine-grained opinion spans in text. We also present a novel methodology for an LLM to adjudicate multiple labels and produce final annotations. After trialling the pipeline with models of different sizes for the Aspect Sentiment Triplet Extraction (ASTE) and Aspect-Category-Opinion-Sentiment (ACOS) analysis tasks, we show that LLMs can serve as automatic annotators and adjudicators, achieving high Inter-Annotator Agreement across individual LLM-based annotators. This reduces the cost and human effort needed to create these fine-grained opinion-annotated datasets.
>
---
#### [new 008] Retrieve-Refine-Calibrate: A Framework for Complex Claim Fact-Checking
- **分类: cs.CL**

- **简介: 该论文属于事实核查任务，旨在解决分解方法引入噪声的问题。提出RRC框架，通过检索、精炼和校准提升核查准确性。**

- **链接: [https://arxiv.org/pdf/2601.16555v1](https://arxiv.org/pdf/2601.16555v1)**

> **作者:** Mingwei Sun; Qianlong Wang; Ruifeng Xu
>
> **备注:** 9 pages, 4 figures. This is an original work by the authors. Any unauthorized submission, reproduction, or commercial use by third parties is prohibited
>
> **摘要:** Fact-checking aims to verify the truthfulness of a claim based on the retrieved evidence. Existing methods typically follow a decomposition paradigm, in which a claim is broken down into sub-claims that are individually verified. However, the decomposition paradigm may introduce noise to the verification process due to irrelevant entities or evidence, ultimately degrading verification accuracy. To address this problem, we propose a Retrieve-Refine-Calibrate (RRC) framework based on large language models (LLMs). Specifically, the framework first identifies the entities mentioned in the claim and retrieves evidence relevant to them. Then, it refines the retrieved evidence based on the claim to reduce irrelevant information. Finally, it calibrates the verification process by re-evaluating low-confidence predictions. Experiments on two popular fact-checking datasets (HOVER and FEVEROUS-S) demonstrate that our framework achieves superior performance compared with competitive baselines.
>
---
#### [new 009] PolyAgent: Large Language Model Agent for Polymer Design
- **分类: cs.CL**

- **简介: 该论文属于聚合物设计任务，旨在解决实验过程耗时长、资源多的问题。提出PolyAgent框架，利用大语言模型实现聚合物结构与性质的预测和生成。**

- **链接: [https://arxiv.org/pdf/2601.16376v1](https://arxiv.org/pdf/2601.16376v1)**

> **作者:** Vani Nigam; Achuth Chandrasekhar; Amir Barati Farimani
>
> **摘要:** On-demand Polymer discovery is essential for various industries, ranging from biomedical to reinforcement materials. Experiments with polymers have a long trial-and-error process, leading to long procedures and extensive resources. For these processes, machine learning has accelerated scientific discovery at the property prediction and latent space search fronts. However, laboratory researchers cannot readily access codes and these models to extract individual structures and properties due to infrastructure limitations. We present a closed-loop polymer structure-property predictor integrated in a terminal for early-stage polymer discovery. The framework is powered by LLM reasoning to provide users with property prediction, property-guided polymer structure generation, and structure modification capabilities. The SMILES sequences are guided by the synthetic accessibility score and the synthetic complexity score (SC Score) to ensure that polymer generation is as close as possible to synthetically accessible monomer-level structures. This framework addresses the challenge of generating novel polymer structures for laboratory researchers, thereby providing computational insights into polymer research.
>
---
#### [new 010] Towards Latent Diffusion Suitable For Text
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于自然语言生成任务，旨在解决扩散模型在离散状态空间中的应用问题。通过引入神经流扩散模型，提升采样速度与生成质量。**

- **链接: [https://arxiv.org/pdf/2601.16220v1](https://arxiv.org/pdf/2601.16220v1)**

> **作者:** Nesta Midavaine; Christian A. Naesseth; Grigory Bartosh
>
> **摘要:** Language diffusion models aim to improve sampling speed and coherence over autoregressive LLMs. We introduce Neural Flow Diffusion Models for language generation, an extension of NFDM that enables the straightforward application of continuous diffusion models to discrete state spaces. NFDM learns a multivariate forward process from the data, ensuring that the forward process and generative trajectory are a good fit for language modeling. Our model substantially reduces the likelihood gap with autoregressive models of the same size, while achieving sample quality comparable to that of previous latent diffusion models.
>
---
#### [new 011] PROST-LLM: Progressively Enhancing the Speech-to-Speech Translation Capability in LLMs
- **分类: cs.CL**

- **简介: 该论文属于语音到语音翻译任务，旨在解决LLMs在该任务中数据稀缺的问题。通过渐进式优化提升模型性能，包括预训练、自采样和偏好优化。**

- **链接: [https://arxiv.org/pdf/2601.16618v1](https://arxiv.org/pdf/2601.16618v1)**

> **作者:** Jing Xu; Jiaqi Wang; Daxin Tan; Xiao Chen
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Although Large Language Models (LLMs) excel in many tasks, their application to Speech-to-Speech Translation (S2ST) is underexplored and hindered by data scarcity. To bridge this gap, we propose PROST-LLM (PROgressive Speech-to-speech Translation) to enhance the S2ST capabilities in LLMs progressively. First, we fine-tune the LLMs with the CVSS corpus, employing designed tri-task learning and chain of modality methods to boost the initial performance. Then, leveraging the fine-tuned model, we generate preference pairs through self-sampling and back-translation without human evaluation. Finally, these preference pairs are used for preference optimization to enhance the model's S2ST capability further. Extensive experiments confirm the effectiveness of our proposed PROST-LLM in improving the S2ST capability of LLMs.
>
---
#### [new 012] Persuasion Tokens for Editing Factual Knowledge in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于知识编辑任务，旨在解决LLMs更新信息效率低的问题。通过引入P-Tokens，实现高效、无需特定事实示例的知识编辑。**

- **链接: [https://arxiv.org/pdf/2601.16781v1](https://arxiv.org/pdf/2601.16781v1)**

> **作者:** Paul Youssef; Jörg Schlötterer; Christin Seifert
>
> **备注:** Accepted at EACL Main 2026
>
> **摘要:** In-context knowledge editing (IKE) is a promising technique for updating Large Language Models (LLMs) with new information. However, IKE relies on lengthy, fact-specific demonstrations which are costly to create and consume significant context window space. In this paper, we introduce persuasion tokens (P-Tokens) -- special tokens trained to replicate the effect of IKE demonstrations, enabling efficient knowledge editing without requiring fact-specific demonstrations. We evaluate P-Tokens across two editing datasets and three LLMs, demonstrating performance comparable to, and often exceeding, IKE. We further find that editing performance is robust to distractors with small negative effects to neighboring facts, and that increasing the number of P-Tokens improves performance. Our work addresses key limitations of IKE and provides a more practical and scalable alternative for editing LLMs.
>
---
#### [new 013] Limits of n-gram Style Control for LLMs via Logit-Space Injection
- **分类: cs.CL**

- **简介: 该论文属于文本风格控制任务，旨在探索通过logit空间注入n-gram先验实现轻量风格调节的有效性与局限性。**

- **链接: [https://arxiv.org/pdf/2601.16224v1](https://arxiv.org/pdf/2601.16224v1)**

> **作者:** Sami-ul Ahmed
>
> **备注:** 18 pages, 7 figures. Experimental study of decoding-time style control via n-gram logit injection
>
> **摘要:** Large language models (LLMs) are typically personalized via prompt engineering or parameter-efficient fine-tuning such as LoRA. However, writing style can be difficult to distill into a single prompt, and LoRA fine-tuning requires computationally intensive training and infrastructure. We investigate a possible lightweight alternative: steering a frozen LLM with n-gram style priors injected in logit space at decoding time. We train an n-gram model on stylistically distinct corpora -- including Don Quixote, CNN/DailyMail news headlines, and arXiv abstracts -- constructing an interpolated 1-to-3-gram prior over next-token probabilities. During generation we modify the LLM's logits by adding a weighted sum of style log-probabilities from each n-gram order that matches the current context, scaled by a control parameter lambda in [0, 1]. We sweep lambda and style corpora and report style perplexity under the n-gram model, base-model perplexity as a proxy for fluency, Jensen-Shannon (JS) divergence between the original and steered token distributions, and token-overlap statistics. On TinyLlama-1.1B we identify a single narrow regime (for the Don Quixote corpus at lambda=0.1) where style perplexity improves by 24.7% and base-model perplexity improves by 51.4% relative to the frozen model. Outside this regime, and for multi-author corpora such as CNN/DailyMail and arXiv abstracts, even small nonzero lambda values generally result in worse style and fluency, and larger lambda values lead to collapse with extreme perplexities and incoherent text. Logit-space injection of n-gram style priors provides lightweight, tunable style control, but it is fragile: it operates effectively only within a narrow range of low lambda values and is consistently outperformed by prompting and LoRA.
>
---
#### [new 014] Generating Literature-Driven Scientific Theories at Scale
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于科学理论生成任务，旨在从大量文献中合成理论。解决如何利用文献支持生成更准确和有预测力的理论的问题，通过对比不同生成方法验证效果。**

- **链接: [https://arxiv.org/pdf/2601.16282v1](https://arxiv.org/pdf/2601.16282v1)**

> **作者:** Peter Jansen; Peter Clark; Doug Downey; Daniel S. Weld
>
> **备注:** 9 pages plus appendix, 3 figures
>
> **摘要:** Contemporary automated scientific discovery has focused on agents for generating scientific experiments, while systems that perform higher-level scientific activities such as theory building remain underexplored. In this work, we formulate the problem of synthesizing theories consisting of qualitative and quantitative laws from large corpora of scientific literature. We study theory generation at scale, using 13.7k source papers to synthesize 2.9k theories, examining how generation using literature-grounding versus parametric knowledge, and accuracy-focused versus novelty-focused generation objectives change theory properties. Our experiments show that, compared to using parametric LLM memory for generation, our literature-supported method creates theories that are significantly better at both matching existing evidence and at predicting future results from 4.6k subsequently-written papers
>
---
#### [new 015] TL-GRPO: Turn-Level RL for Reasoning-Guided Iterative Optimization
- **分类: cs.CL**

- **简介: 该论文提出TL-GRPO，解决迭代优化任务中的细粒度策略优化问题，通过 turn-level 采样提升性能。**

- **链接: [https://arxiv.org/pdf/2601.16480v1](https://arxiv.org/pdf/2601.16480v1)**

> **作者:** Peiji Li; Linyang Li; Handa Sun; Wenjin Mai; Yongkang Chen; Xiaozhe Li; Yue Shen; Yichuan Ma; Yiliu Sun; Jiaxi Cao; Zhishu He; Bo Wang; Xiaoqing Zheng; Zhaori Bi; Xipeng Qiu; Qipeng Guo; Kai Chen; Dahua Lin
>
> **备注:** Work in progress
>
> **摘要:** Large language models have demonstrated strong reasoning capabilities in complex tasks through tool integration, which is typically framed as a Markov Decision Process and optimized with trajectory-level RL algorithms such as GRPO. However, a common class of reasoning tasks, iterative optimization, presents distinct challenges: the agent interacts with the same underlying environment state across turns, and the value of a trajectory is determined by the best turn-level reward rather than cumulative returns. Existing GRPO-based methods cannot perform fine-grained, turn-level optimization in such settings, while black-box optimization methods discard prior knowledge and reasoning capabilities. To address this gap, we propose Turn-Level GRPO (TL-GRPO), a lightweight RL algorithm that performs turn-level group sampling for fine-grained optimization. We evaluate TL-GRPO on analog circuit sizing (ACS), a challenging scientific optimization task requiring multiple simulations and domain expertise. Results show that TL-GRPO outperforms standard GRPO and Bayesian optimization methods across various specifications. Furthermore, our 30B model trained with TL-GRPO achieves state-of-the-art performance on ACS tasks under same simulation budget, demonstrating both strong generalization and practical utility.
>
---
#### [new 016] A Longitudinal, Multinational, and Multilingual Corpus of News Coverage of the Russo-Ukrainian War
- **分类: cs.CL; cs.SI**

- **简介: 该论文介绍了一个多国、多语言的新闻语料库DNIPRO，用于研究俄乌战争中的媒体叙事差异。属于跨国家庭分析任务，旨在解决冲突期间不同立场报道的对比问题。**

- **链接: [https://arxiv.org/pdf/2601.16309v1](https://arxiv.org/pdf/2601.16309v1)**

> **作者:** Dikshya Mohanty; Taisiia Sabadyn; Jelwin Rodrigues; Chenlu Wang; Abhishek Kalugade; Ritwik Banerjee
>
> **摘要:** We introduce DNIPRO, a novel longitudinal corpus of 246K news articles documenting the Russo-Ukrainian war from Feb 2022 to Aug 2024, spanning eleven media outlets across five nation states (Russia, Ukraine, U.S., U.K., and China) and three languages (English, Russian, and Mandarin Chinese). This multilingual resource features consistent and comprehensive metadata, and multiple types of annotation with rigorous human evaluations for downstream tasks relevant to systematic transnational analyses of contentious wartime discourse. DNIPRO's distinctive value lies in its inclusion of competing geopolitical perspectives, making it uniquely suited for studying narrative divergence, media framing, and information warfare. To demonstrate its utility, we include use case experiments using stance detection, sentiment analysis, topical framing, and contradiction analysis of major conflict events within the larger war. Our explorations reveal how outlets construct competing realities, with coverage exhibiting polarized interpretations that reflect geopolitical interests. Beyond supporting computational journalism research, DNIPRO provides a foundational resource for understanding how conflicting narratives emerge and evolve across global information ecosystems.
>
---
#### [new 017] Exploring the Effects of Alignment on Numerical Bias in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，研究对齐导致的数值偏差问题。通过对比对齐前后模型输出，发现对齐增加偏差，并尝试多种方法缓解，其中调整评分范围效果最佳。**

- **链接: [https://arxiv.org/pdf/2601.16444v1](https://arxiv.org/pdf/2601.16444v1)**

> **作者:** Ayako Sato; Hwichan Kim; Zhousi Chen; Masato Mita; Mamoru Komachi
>
> **备注:** Accepted at AIBSD 2026 (Workshop at AAAI 2026)
>
> **摘要:** ``LLM-as-a-judge,'' which utilizes large language models (LLMs) as evaluators, has proven effective in many evaluation tasks. However, evaluator LLMs exhibit numerical bias, a phenomenon where certain evaluation scores are generated disproportionately often, leading reduced evaluation performance. This study investigates the cause of this bias. Given that most evaluator LLMs are aligned through instruction tuning and preference tuning, and that prior research suggests alignment reduces output diversity, we hypothesize that numerical bias arises from alignment. To test this, we compare outputs from pre- and post-alignment LLMs, and observe that alignment indeed increases numerical bias. We also explore mitigation strategies for post-alignment LLMs, including temperature scaling, distribution calibration, and score range adjustment. Among these, score range adjustment is most effective in reducing bias and improving performance, though still heuristic. Our findings highlight the need for further work on optimal score range selection and more robust mitigation strategies.
>
---
#### [new 018] Strategies for Span Labeling with Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在文本标注任务中的策略，解决生成模型缺乏显式输入引用的问题。提出LogitMatch方法提升span匹配效果。**

- **链接: [https://arxiv.org/pdf/2601.16946v1](https://arxiv.org/pdf/2601.16946v1)**

> **作者:** Danil Semin; Ondřej Dušek; Zdeněk Kasner
>
> **摘要:** Large language models (LLMs) are increasingly used for text analysis tasks, such as named entity recognition or error detection. Unlike encoder-based models, however, generative architectures lack an explicit mechanism to refer to specific parts of their input. This leads to a variety of ad-hoc prompting strategies for span labeling, often with inconsistent results. In this paper, we categorize these strategies into three families: tagging the input text, indexing numerical positions of spans, and matching span content. To address the limitations of content matching, we introduce LogitMatch, a new constrained decoding method that forces the model's output to align with valid input spans. We evaluate all methods across four diverse tasks. We find that while tagging remains a robust baseline, LogitMatch improves upon competitive matching-based methods by eliminating span matching issues and outperforms other strategies in some setups.
>
---
#### [new 019] SearchLLM: Detecting LLM Paraphrased Text by Measuring the Similarity with Regeneration of the Candidate Source via Search Engine
- **分类: cs.CL**

- **简介: 该论文属于文本检测任务，旨在解决LLM paraphrased文本难以检测的问题。通过搜索引擎查找原始来源并比较相似性，提升检测准确性。**

- **链接: [https://arxiv.org/pdf/2601.16512v1](https://arxiv.org/pdf/2601.16512v1)**

> **作者:** Hoang-Quoc Nguyen-Son; Minh-Son Dao; Koji Zettsu
>
> **备注:** EACL 2026 camera ready (Main Track)
>
> **摘要:** With the advent of large language models (LLMs), it has become common practice for users to draft text and utilize LLMs to enhance its quality through paraphrasing. However, this process can sometimes result in the loss or distortion of the original intended meaning. Due to the human-like quality of LLM-generated text, traditional detection methods often fail, particularly when text is paraphrased to closely mimic original content. In response to these challenges, we propose a novel approach named SearchLLM, designed to identify LLM-paraphrased text by leveraging search engine capabilities to locate potential original text sources. By analyzing similarities between the input and regenerated versions of candidate sources, SearchLLM effectively distinguishes LLM-paraphrased content. SearchLLM is designed as a proxy layer, allowing seamless integration with existing detectors to enhance their performance. Experimental results across various LLMs demonstrate that SearchLLM consistently enhances the accuracy of recent detectors in detecting LLM-paraphrased text that closely mimics original content. Furthermore, SearchLLM also helps the detectors prevent paraphrasing attacks.
>
---
#### [new 020] Better Generalizing to Unseen Concepts: An Evaluation Framework and An LLM-Based Auto-Labeled Pipeline for Biomedical Concept Recognition
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于生物医学概念识别任务，解决未见概念泛化问题。提出评估框架和基于大模型的自动标注数据管道，提升模型对未见概念的识别能力。**

- **链接: [https://arxiv.org/pdf/2601.16711v1](https://arxiv.org/pdf/2601.16711v1)**

> **作者:** Shanshan Liu; Noriki Nishida; Fei Cheng; Narumi Tokunaga; Rumana Ferdous Munne; Yuki Yamagata; Kouji Kozaki; Takehito Utsuro; Yuji Matsumoto
>
> **备注:** Accepted to EACL 2026 (Main)
>
> **摘要:** Generalization to unseen concepts is a central challenge due to the scarcity of human annotations in Mention-agnostic Biomedical Concept Recognition (MA-BCR). This work makes two key contributions to systematically address this issue. First, we propose an evaluation framework built on hierarchical concept indices and novel metrics to measure generalization. Second, we explore LLM-based Auto-Labeled Data (ALD) as a scalable resource, creating a task-specific pipeline for its generation. Our research unequivocally shows that while LLM-generated ALD cannot fully substitute for manual annotations, it is a valuable resource for improving generalization, successfully providing models with the broader coverage and structural knowledge needed to approach recognizing unseen concepts. Code and datasets are available at https://github.com/bio-ie-tool/hi-ald.
>
---
#### [new 021] ChiEngMixBench: Evaluating Large Language Models on Spontaneous and Natural Chinese-English Code-Mixed Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言生成任务，旨在评估大模型在自然中英混用场景下的表现。提出ChiEngMixBench基准，解决代码混用评估不足的问题，揭示了术语分层策略与认知对齐现象。**

- **链接: [https://arxiv.org/pdf/2601.16217v1](https://arxiv.org/pdf/2601.16217v1)**

> **作者:** Qingyan Yang; Tongxi Wang; Yunsheng Luo
>
> **摘要:** Code-mixing is increasingly prevalent in interactions between humans and large language models, yet existing work often reduces it to a translation or convertibility problem, making it difficult to assess whether a model's switching behavior is context-appropriate and aligned with human conventions. We introduce ChiEngMixBench, the first benchmark designed to evaluate code-mixing ability in authentic community contexts, built upon a general construction pipeline that enables scalable dataset development across domains and bilingual pairs. ChiEngMixBench formulates code-mixing as a cognitive alignment problem, characterized by two complementary signals: Spontaneity and Naturalness. Empirical evaluation shows that our metrics can systematically distinguish code-mixing performance across models. Beyond benchmarking, we further uncover an implicitly emergent Terminology Layering Strategy, a phenomenon consistent with the Matrix Language Frame (MLF) theory, indicating structured cognitive alignment between multilingual large language models and human communication.
>
---
#### [new 022] How Does Personalized Memory Shape LLM Behavior? Benchmarking Rational Preference Utilization in Personalized Assistants
- **分类: cs.CL**

- **简介: 该论文研究个性化记忆对LLM行为的影响，解决个性化与意图理解冲突的问题。通过构建RPEval基准和RP-Reasoner方法，提升个性化信息的合理利用。**

- **链接: [https://arxiv.org/pdf/2601.16621v1](https://arxiv.org/pdf/2601.16621v1)**

> **作者:** Xueyang Feng; Weinan Gan; Xu Chen; Quanyu Dai; Yong Liu
>
> **摘要:** Large language model (LLM)-powered assistants have recently integrated memory mechanisms that record user preferences, leading to more personalized and user-aligned responses. However, irrelevant personalized memories are often introduced into the context, interfering with the LLM's intent understanding. To comprehensively investigate the dual effects of personalization, we develop RPEval, a benchmark comprising a personalized intent reasoning dataset and a multi-granularity evaluation protocol. RPEval reveals the widespread phenomenon of irrational personalization in existing LLMs and, through error pattern analysis, illustrates its negative impact on user experience. Finally, we introduce RP-Reasoner, which treats memory utilization as a pragmatic reasoning process, enabling the selective integration of personalized information. Experimental results demonstrate that our method significantly outperforms carefully designed baselines on RPEval, and resolves 80% of the bad cases observed in a large-scale commercial personalized assistant, highlighting the potential of pragmatic reasoning to mitigate irrational personalization. Our benchmark is publicly available at https://github.com/XueyangFeng/RPEval.
>
---
#### [new 023] AuroraEdge-V-2B: A Faster And Stronger Edge Visual Large Language Model
- **分类: cs.CL**

- **简介: 该论文属于视觉语言模型任务，旨在解决VLLM在工业应用中的效率与性能问题。提出AuroraEdge-V-2B模型，优化参数量与推理速度，提升边缘部署能力。**

- **链接: [https://arxiv.org/pdf/2601.16615v1](https://arxiv.org/pdf/2601.16615v1)**

> **作者:** Xiang Chen
>
> **摘要:** Recently, due to the advancement of multimodal technology, people are attempting to use visual large language models (VLLMs) in industrial production. Many deep learning models (DLMs) deployed in the production environment are gradually being replaced by VLLMs. Compared with DLMs, VLLMs have some advantages in industrial applications: (1) Their strong generalization ability enables them to perform well across a wide range of tasks. (2) They are flexible and can deal with unfamiliar samples through context learning quickly. However, VLLMs also have obvious drawbacks: (1) VLLMs do not perform as well as custom-developed DLMs in specific domains. (2) The number of parameters in VLLMs is generally quite large, and their deployment requires substantial computational resources. (3) VLLMs generally operate much slower than DLMs, making real-time response challenging to achieve. To better utilize VLLMs in industrial applications, we introduce AuroraEdge-V-2B in this work, a compact, robust, and high-speed VLLM designed for edge deployment. To make the model run faster, we also propose a compression-fusion method to improve inference efficiency. AuroraEdge-V-2B has the following notable features: (1) Easy deployment and faster: It has only 2B parameters and is highly suitable for edge deployment, offering better real-time performance. (2) Fewer visual tokens and cheaper: It significantly reduces the number of visual tokens in the decoding process, thereby reducing the floating-point operations by half during inference and making it cheaper to use. (3) Strong performance: It gets a higher score on 9 benchmarks than models with the same number of parameter (e.g., Qwen2-VL-2B, Qwen2.5-VL-3B, InternVL-2.5-2B).
>
---
#### [new 024] Timely Machine: Awareness of Time Makes Test-Time Scaling Agentic
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决agentic场景下测试时扩展的效率问题。提出Timely Machine，基于实际时间调整策略，并引入Timely-Eval和Timely-RL提升时间感知与性能。**

- **链接: [https://arxiv.org/pdf/2601.16486v1](https://arxiv.org/pdf/2601.16486v1)**

> **作者:** Yichuan Ma; Linyang Li; Yongkang chen; Peiji Li; Xiaozhe Li; Qipeng Guo; Dahua Lin; Kai Chen
>
> **备注:** Under Review
>
> **摘要:** As large language models (LLMs) increasingly tackle complex reasoning tasks, test-time scaling has become critical for enhancing capabilities. However, in agentic scenarios with frequent tool calls, the traditional generation-length-based definition breaks down: tool latency decouples inference time from generation length. We propose Timely Machine, redefining test-time as wall-clock time, where models dynamically adjust strategies based on time budgets. We introduce Timely-Eval, a benchmark spanning high-frequency tool calls, low-frequency tool calls, and time-constrained reasoning. By varying tool latency, we find smaller models excel with fast feedback through more interactions, while larger models dominate high-latency settings via superior interaction quality. Moreover, existing models fail to adapt reasoning to time budgets. We propose Timely-RL to address this gap. After cold-start supervised fine-tuning, we use reinforcement learning to enhance temporal planning. Timely-RL improves time budget awareness and consistently boosts performance across Timely-Eval. We hope our work offers a new perspective on test-time scaling for the agentic era.
>
---
#### [new 025] LLM-Based Adversarial Persuasion Attacks on Fact-Checking Systems
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于事实核查任务，旨在解决AFC系统受对抗攻击的问题。通过生成式大模型运用说服技巧进行攻击，降低验证和证据检索效果。**

- **链接: [https://arxiv.org/pdf/2601.16890v1](https://arxiv.org/pdf/2601.16890v1)**

> **作者:** João A. Leite; Olesya Razuvayevskaya; Kalina Bontcheva; Carolina Scarton
>
> **摘要:** Automated fact-checking (AFC) systems are susceptible to adversarial attacks, enabling false claims to evade detection. Existing adversarial frameworks typically rely on injecting noise or altering semantics, yet no existing framework exploits the adversarial potential of persuasion techniques, which are widely used in disinformation campaigns to manipulate audiences. In this paper, we introduce a novel class of persuasive adversarial attacks on AFCs by employing a generative LLM to rephrase claims using persuasion techniques. Considering 15 techniques grouped into 6 categories, we study the effects of persuasion on both claim verification and evidence retrieval using a decoupled evaluation strategy. Experiments on the FEVER and FEVEROUS benchmarks show that persuasion attacks can substantially degrade both verification performance and evidence retrieval. Our analysis identifies persuasion techniques as a potent class of adversarial attacks, highlighting the need for more robust AFC systems.
>
---
#### [new 026] Machine-Assisted Grading of Nationwide School-Leaving Essay Exams with LLMs and Statistical NLP
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自动评分任务，旨在解决大规模作文批改问题。通过LLMs和统计NLP方法进行自动化评分，并与人工评分对比验证其有效性。**

- **链接: [https://arxiv.org/pdf/2601.16314v1](https://arxiv.org/pdf/2601.16314v1)**

> **作者:** Andres Karjus; Kais Allkivi; Silvia Maine; Katarin Leppik; Krister Kruusmaa; Merilin Aruvee
>
> **摘要:** Large language models (LLMs) enable rapid and consistent automated evaluation of open-ended exam responses, including dimensions of content and argumentation that have traditionally required human judgment. This is particularly important in cases where a large amount of exams need to be graded in a limited time frame, such as nation-wide graduation exams in various countries. Here, we examine the applicability of automated scoring on two large datasets of trial exam essays of two full national cohorts from Estonia. We operationalize the official curriculum-based rubric and compare LLM and statistical natural language processing (NLP) based assessments with human panel scores. The results show that automated scoring can achieve performance comparable to that of human raters and tends to fall within the human scoring range. We also evaluate bias, prompt injection risks, and LLMs as essay writers. These findings demonstrate that a principled, rubric-driven, human-in-the-loop scoring pipeline is viable for high-stakes writing assessment, particularly relevant for digitally advanced societies like Estonia, which is about to adapt a fully electronic examination system. Furthermore, the system produces fine-grained subscore profiles that can be used to generate systematic, personalized feedback for instruction and exam preparation. The study provides evidence that LLM-assisted assessment can be implemented at a national scale, even in a small-language context, while maintaining human oversight and compliance with emerging educational and regulatory standards.
>
---
#### [new 027] Do LLM hallucination detectors suffer from low-resource effect?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究低资源语言下大模型幻觉检测的性能问题，属于自然语言处理任务。它探讨了幻觉检测器是否受低资源效应影响，并通过实验验证其在不同语言和设置下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.16766v1](https://arxiv.org/pdf/2601.16766v1)**

> **作者:** Debtanu Datta; Mohan Kishore Chilukuri; Yash Kumar; Saptarshi Ghosh; Muhammad Bilal Zafar
>
> **备注:** Accepted at EACL 2026 (Main)
>
> **摘要:** LLMs, while outperforming humans in a wide range of tasks, can still fail in unanticipated ways. We focus on two pervasive failure modes: (i) hallucinations, where models produce incorrect information about the world, and (ii) the low-resource effect, where the models show impressive performance in high-resource languages like English but the performance degrades significantly in low-resource languages like Bengali. We study the intersection of these issues and ask: do hallucination detectors suffer from the low-resource effect? We conduct experiments on five tasks across three domains (factual recall, STEM, and Humanities). Experiments with four LLMs and three hallucination detectors reveal a curious finding: As expected, the task accuracies in low-resource languages experience large drops (compared to English). However, the drop in detectors' accuracy is often several times smaller than the drop in task accuracy. Our findings suggest that even in low-resource languages, the internal mechanisms of LLMs might encode signals about their uncertainty. Further, the detectors are robust within language (even for non-English) and in multilingual setups, but not in cross-lingual settings without in-language supervision.
>
---
#### [new 028] Mitigating Bias in Automated Grading Systems for ESL Learners: A Contrastive Learning Approach
- **分类: cs.CL**

- **简介: 该论文属于自动作文评分任务，旨在解决ESL学习者在AES系统中因算法偏见而被低估的问题。通过对比学习方法减少评分差异，提升公平性。**

- **链接: [https://arxiv.org/pdf/2601.16724v1](https://arxiv.org/pdf/2601.16724v1)**

> **作者:** Kevin Fan; Eric Yun
>
> **摘要:** As Automated Essay Scoring (AES) systems are increasingly used in high-stakes educational settings, concerns regarding algorithmic bias against English as a Second Language (ESL) learners have increased. Current Transformer-based regression models trained primarily on native-speaker corpora often learn spurious correlations between surface-level L2 linguistic features and essay quality. In this study, we conduct a bias study of a fine-tuned DeBERTa-v3 model using the ASAP 2.0 and ELLIPSE datasets, revealing a constrained score scaling for high-proficiency ESL writing where high-proficiency ESL essays receive scores 10.3% lower than Native speaker essays of identical human-rated quality. To mitigate this, we propose applying contrastive learning with a triplet construction strategy: Contrastive Learning with Matched Essay Pairs. We constructed a dataset of 17,161 matched essay pairs and fine-tuned the model using Triplet Margin Loss to align the latent representations of ESL and Native writing. Our approach reduced the high-proficiency scoring disparity by 39.9% (to a 6.2% gap) while maintaining a Quadratic Weighted Kappa (QWK) of 0.76. Post-hoc linguistic analysis suggests the model successfully disentangled sentence complexity from grammatical error, preventing the penalization of valid L2 syntactic structures.
>
---
#### [new 029] DeepEra: A Deep Evidence Reranking Agent for Scientific Retrieval-Augmented Generated Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于科学问答任务，旨在解决RAG框架中因语义相似但逻辑无关的文本导致的事实不可靠问题。提出DeepEra模型进行更精准的证据重排序。**

- **链接: [https://arxiv.org/pdf/2601.16478v1](https://arxiv.org/pdf/2601.16478v1)**

> **作者:** Haotian Chen; Qingqing Long; Siyu Pu; Xiao Luo; Wei Ju; Meng Xiao; Yuanchun Zhou; Jianghua Zhao; Xuezhi Wang
>
> **摘要:** With the rapid growth of scientific literature, scientific question answering (SciQA) has become increasingly critical for exploring and utilizing scientific knowledge. Retrieval-Augmented Generation (RAG) enhances LLMs by incorporating knowledge from external sources, thereby providing credible evidence for scientific question answering. But existing retrieval and reranking methods remain vulnerable to passages that are semantically similar but logically irrelevant, often reducing factual reliability and amplifying hallucinations.To address this challenge, we propose a Deep Evidence Reranking Agent (DeepEra) that integrates step-by-step reasoning, enabling more precise evaluation of candidate passages beyond surface-level semantics. To support systematic evaluation, we construct SciRAG-SSLI (Scientific RAG - Semantically Similar but Logically Irrelevant), a large-scale dataset comprising about 300K SciQA instances across 10 subjects, constructed from 10M scientific corpus. The dataset combines naturally retrieved contexts with systematically generated distractors to test logical robustness and factual grounding. Comprehensive evaluations confirm that our approach achieves superior retrieval performance compared to leading rerankers. To our knowledge, this work is the first to comprehensively study and empirically validate innegligible SSLI issues in two-stage RAG frameworks.
>
---
#### [new 030] Curate-Train-Refine: A Closed-Loop Agentic Framework for Zero Shot Classification
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出一种闭环框架Curate-Train-Refine，用于零样本分类。解决大模型部署成本高的问题，通过轻量级分类器与LLM协作提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.16530v1](https://arxiv.org/pdf/2601.16530v1)**

> **作者:** Gaurav Maheshwari; Kevin El Haddad
>
> **摘要:** Large language models (LLMs) and high-capacity encoders have advanced zero and few-shot classification, but their inference cost and latency limit practical deployment. We propose training lightweight text classifiers using dynamically generated supervision from an LLM. Our method employs an iterative, agentic loop in which the LLM curates training data, analyzes model successes and failures, and synthesizes targeted examples to address observed errors. This closed-loop generation and evaluation process progressively improves data quality and adapts it to the downstream classifier and task. Across four widely used benchmarks, our approach consistently outperforms standard zero and few-shot baselines. These results indicate that LLMs can serve effectively as data curators, enabling accurate and efficient classification without the operational cost of large-model deployment.
>
---
#### [new 031] MRAG: Benchmarking Retrieval-Augmented Generation for Bio-medicine
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学问答任务，旨在解决RAG在医疗领域缺乏全面评估的问题。提出MRAG基准和工具包，评估不同RAG组件的效果。**

- **链接: [https://arxiv.org/pdf/2601.16503v1](https://arxiv.org/pdf/2601.16503v1)**

> **作者:** Wei Zhu
>
> **摘要:** While Retrieval-Augmented Generation (RAG) has been swiftly adopted in scientific and clinical QA systems, a comprehensive evaluation benchmark in the medical domain is lacking. To address this gap, we introduce the Medical Retrieval-Augmented Generation (MRAG) benchmark, covering various tasks in English and Chinese languages, and building a corpus with Wikipedia and Pubmed. Additionally, we develop the MRAG-Toolkit, facilitating systematic exploration of different RAG components. Our experiments reveal that: (a) RAG enhances LLM reliability across MRAG tasks. (b) the performance of RAG systems is influenced by retrieval approaches, model sizes, and prompting strategies. (c) While RAG improves usefulness and reasoning quality, LLM responses may become slightly less readable for long-form questions. We will release the MRAG-Bench's dataset and toolkit with CCBY-4.0 license upon acceptance, to facilitate applications from both academia and industry.
>
---
#### [new 032] Persona Jailbreaking in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究LLM中角色劫持问题，提出PHISH框架通过用户输入操纵模型角色，验证其在多个领域的有效性，揭示安全漏洞并强调需更强的上下文防护。**

- **链接: [https://arxiv.org/pdf/2601.16466v1](https://arxiv.org/pdf/2601.16466v1)**

> **作者:** Jivnesh Sandhan; Fei Cheng; Tushar Sandhan; Yugo Murawaki
>
> **备注:** Accepted at EACL26 (Findings)
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in domains such as education, mental health and customer support, where stable and consistent personas are critical for reliability. Yet, existing studies focus on narrative or role-playing tasks and overlook how adversarial conversational history alone can reshape induced personas. Black-box persona manipulation remains unexplored, raising concerns for robustness in realistic interactions. In response, we introduce the task of persona editing, which adversarially steers LLM traits through user-side inputs under a black-box, inference-only setting. To this end, we propose PHISH (Persona Hijacking via Implicit Steering in History), the first framework to expose a new vulnerability in LLM safety that embeds semantically loaded cues into user queries to gradually induce reverse personas. We also define a metric to quantify attack success. Across 3 benchmarks and 8 LLMs, PHISH predictably shifts personas, triggers collateral changes in correlated traits, and exhibits stronger effects in multi-turn settings. In high-risk domains mental health, tutoring, and customer support, PHISH reliably manipulates personas, validated by both human and LLM-as-Judge evaluations. Importantly, PHISH causes only a small reduction in reasoning benchmark performance, leaving overall utility largely intact while still enabling significant persona manipulation. While current guardrails offer partial protection, they remain brittle under sustained attack. Our findings expose new vulnerabilities in personas and highlight the need for context-resilient persona in LLMs. Our codebase and dataset is available at: https://github.com/Jivnesh/PHISH
>
---
#### [new 033] Domain Specific Specialization in Low-Resource Settings: The Efficacy of Offline Response-Based Knowledge Distillation in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文研究低资源环境下大语言模型的领域专用化问题，通过离线响应知识蒸馏方法提升模型准确性。任务为领域适应，解决模型在专业领域中的幻觉问题，采用高质量小数据集进行优化。**

- **链接: [https://arxiv.org/pdf/2601.16219v1](https://arxiv.org/pdf/2601.16219v1)**

> **作者:** Erdem Aslan; Pakize Erdoğmuş
>
> **备注:** 10 pages, 10 tables
>
> **摘要:** Large Language Models (LLMs) excel in general tasks but often struggle with hallucinations when handling domain-specific or institutional knowledge absent from their pre-training. We present an offline response-based knowledge distillation method that develops high-accuracy specialized assistants under constrained hardware resources. We evaluate three distinct data strategies: general domain adaptation (15,000 lines), unstructured knowledge injection (2,000 lines), and a context-aware synthetic dataset (500 lines) generated by a teacher model. To minimize computational costs, we utilize the Unsloth library to optimize the Qwen-2.5-7B student model, reducing NVIDIA A100 GPU memory requirements from 40 GB to 16 GB. Experimental results demonstrate that while larger unstructured datasets suffer from persistent hallucinations, the 500-line context-aware dataset achieves a 96.7% accuracy rate and robust rejection capability. These findings validate the LIMA hypothesis, showing that data quality and structural alignment are more critical than quantity for domain adaptation in low-resource settings.
>
---
#### [new 034] Sycophancy Hides Linearly in the Attention Heads
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型中的奉承行为，通过分析注意力机制定位其线性可分离信号，提出针对性干预方法以减少奉承现象。任务为模型偏差检测与缓解。**

- **链接: [https://arxiv.org/pdf/2601.16644v1](https://arxiv.org/pdf/2601.16644v1)**

> **作者:** Rifo Genadi; Munachiso Nwadike; Nurdaulet Mukhituly; Hilal Alquabeh; Tatsuya Hiraoka; Kentaro Inui
>
> **摘要:** We find that correct-to-incorrect sycophancy signals are most linearly separable within multi-head attention activations. Motivated by the linear representation hypothesis, we train linear probes across the residual stream, multilayer perceptron (MLP), and attention layers to analyze where these signals emerge. Although separability appears in the residual stream and MLPs, steering using these probes is most effective in a sparse subset of middle-layer attention heads. Using TruthfulQA as the base dataset, we find that probes trained on it transfer effectively to other factual QA benchmarks. Furthermore, comparing our discovered direction to previously identified "truthful" directions reveals limited overlap, suggesting that factual accuracy, and deference resistance, arise from related but distinct mechanisms. Attention-pattern analysis further indicates that the influential heads attend disproportionately to expressions of user doubt, contributing to sycophantic shifts. Overall, these findings suggest that sycophancy can be mitigated through simple, targeted linear interventions that exploit the internal geometry of attention activations.
>
---
#### [new 035] Jacobian Scopes: token-level causal attributions in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在解决LLM中token级因果解释的问题。提出Jacobian Scopes方法，通过梯度分析量化输入token对预测的影响。**

- **链接: [https://arxiv.org/pdf/2601.16407v1](https://arxiv.org/pdf/2601.16407v1)**

> **作者:** Toni J. B. Liu; Baran Zadeoğlu; Nicolas Boullé; Raphaël Sarfati; Christopher J. Earls
>
> **备注:** 12 pages, 15 figures, under review at ACL 2026
>
> **摘要:** Large language models (LLMs) make next-token predictions based on clues present in their context, such as semantic descriptions and in-context examples. Yet, elucidating which prior tokens most strongly influence a given prediction remains challenging due to the proliferation of layers and attention heads in modern architectures. We propose Jacobian Scopes, a suite of gradient-based, token-level causal attribution methods for interpreting LLM predictions. By analyzing the linearized relations of final hidden state with respect to inputs, Jacobian Scopes quantify how input tokens influence a model's prediction. We introduce three variants - Semantic, Fisher, and Temperature Scopes - which respectively target sensitivity of specific logits, the full predictive distribution, and model confidence (inverse temperature). Through case studies spanning instruction understanding, translation and in-context learning (ICL), we uncover interesting findings, such as when Jacobian Scopes point to implicit political biases. We believe that our proposed methods also shed light on recently debated mechanisms underlying in-context time-series forecasting. Our code and interactive demonstrations are publicly available at https://github.com/AntonioLiu97/JacobianScopes.
>
---
#### [new 036] Teaching and Evaluating LLMs to Reason About Polymer Design Related Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI4Science领域，旨在解决LLMs在聚合物设计任务中表现不佳的问题。通过构建PolyBench数据集和知识增强的推理蒸馏方法，提升模型在该领域的性能。**

- **链接: [https://arxiv.org/pdf/2601.16312v1](https://arxiv.org/pdf/2601.16312v1)**

> **作者:** Dikshya Mohanty; Mohammad Saqib Hasan; Syed Mostofa Monsur; Size Zheng; Benjamin Hsiao; Niranjan Balasubramanian
>
> **摘要:** Research in AI4Science has shown promise in many science applications, including polymer design. However, current LLMs prove ineffective on this problem space because: (i) most models lack polymer-specific knowledge (ii) existing aligned models lack coverage of knowledge and capabilities relevant to polymer design. Addressing this, we introduce PolyBench, a large scale training and test benchmark dataset of more than 125K polymer design related tasks, leveraging a knowledge base of 13M+ data points obtained from experimental and synthetic sources to ensure broad coverage of polymers and their properties. For effective alignment using PolyBench, we introduce a knowledge-augmented reasoning distillation method that augments this dataset with structured CoT. Furthermore, tasks in PolyBench are organized from simple to complex analytical reasoning problems, enabling generalization tests and diagnostic probes across the problem space. Experiments show that small language models (SLMs), of 7B to 14B parameters, trained on PolyBench data outperform similar sized models, and even closed source frontier LLMs on PolyBench test dataset while demonstrating gains on other polymer benchmarks as well.
>
---
#### [new 037] GameTalk: Training LLMs for Strategic Conversation
- **分类: cs.CL; cs.AI; cs.GT; cs.LG; cs.MA**

- **简介: 该论文提出GameTalk，用于训练大语言模型进行战略对话。任务是解决多智能体环境下的长期目标优化问题，通过对话实现协调与谈判。工作包括设计框架并应用强化学习方法提升性能。**

- **链接: [https://arxiv.org/pdf/2601.16276v1](https://arxiv.org/pdf/2601.16276v1)**

> **作者:** Victor Conchello Vendrell; Max Ruiz Luyten; Mihaela van der Schaar
>
> **备注:** 32 pages, 8 figures
>
> **摘要:** Strategic decision-making in multi-agent settings is a key challenge for large language models (LLMs), particularly when coordination and negotiation must unfold over extended conversations. While recent work has explored the use of LLMs in isolated decision tasks, little attention has been given to optimizing long-term objectives through dialogue. We introduce \textbf{GameTalk}, a framework for training LLMs to make strategic decisions via multi-turn interactions. Unlike prior work that focuses on single-turn objectives or static action prediction, we train LLMs to optimize a global objective across full conversations. We achieve this by adapting fine-tuning methods like GRPO, DPO, and STaR to incorporate reward signals that depend on the entire interaction. We evaluate this approach on a suite of increasingly complex games, designed to stress different aspects of reasoning, coordination, and opponent modeling. Our results show that GameTalk significantly outperforms untrained models, especially under reward shaping, with DPO consistently yielding the strongest gains. These findings position conversational fine-tuning as a promising path for LLMs to reason, negotiate, and act in interactive environments.
>
---
#### [new 038] Learning Domain Knowledge in Multimodal Large Language Models through Reinforcement Fine-Tuning
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态语言模型领域，解决领域知识融入不足的问题。通过强化学习微调，将领域知识直接融入优化目标，提升模型在遥感和医学领域的表现。**

- **链接: [https://arxiv.org/pdf/2601.16419v1](https://arxiv.org/pdf/2601.16419v1)**

> **作者:** Qinglong Cao; Yuntian Chen; Chao Ma; Xiaokang Yang
>
> **摘要:** Multimodal large language models (MLLMs) have shown remarkable capabilities in multimodal perception and understanding tasks. However, their effectiveness in specialized domains, such as remote sensing and medical imaging, remains limited. A natural approach to domain adaptation is to inject domain knowledge through textual instructions, prompts, or auxiliary captions. Surprisingly, we find that such input-level domain knowledge injection yields little to no improvement on scientific multimodal tasks, even when the domain knowledge is explicitly provided. This observation suggests that current MLLMs fail to internalize domain-specific priors through language alone, and that domain knowledge must be integrated at the optimization level. Motivated by this insight, we propose a reinforcement fine-tuning framework that incorporates domain knowledge directly into the learning objective. Instead of treating domain knowledge as descriptive information, we encode it as domain-informed constraints and reward signals, shaping the model's behavior in the output space. Extensive experiments across multiple datasets in remote sensing and medical domains consistently demonstrate good performance gains, achieving state-of-the-art results on multimodal domain tasks. Our results highlight the necessity of optimization-level domain knowledge integration and reveal a fundamental limitation of textual domain conditioning in current MLLMs.
>
---
#### [new 039] Clarify or Answer: Reinforcement Learning for Agentic VQA with Context Under-specification
- **分类: cs.CL**

- **简介: 该论文属于视觉问答任务，解决上下文不明确时的错误回答问题。提出CoA模型，通过强化学习优化澄清问题生成，提升VQA准确率。**

- **链接: [https://arxiv.org/pdf/2601.16400v1](https://arxiv.org/pdf/2601.16400v1)**

> **作者:** Zongwan Cao; Bingbing Wen; Lucy Lu Wang
>
> **摘要:** Real-world visual question answering (VQA) is often context-dependent: an image-question pair may be under-specified, such that the correct answer depends on external information that is not observable in the image. In such cases, directly answering can lead to confident but incorrect predictions. We propose CoA(Clarify-or-Answer), an ask-or-answer agent that separately models the decision to ask or answer, and what to ask if needed. CoA first determines whether clarification is necessary; if so, it asks a single focused question and then incorporates the response to produce the final answer. We introduce CONTEXTCLARIFY with a set of ambiguous VQA questions and the contrast set that is non-ambiguous. We further introduce GRPO-CR (Clarification Reasoning), a reinforcement learning approach that optimizes clarification question generation with multiple reward signals encouraging well-formed, focused, non-trivial questions that resolve ambiguity. Across three VLLMs and three datasets, CoA achieves consistent improvements at both the module and system levels, improving end-to-end VQA accuracy by an average of +15.3 points (83%) over prompting-based baselines
>
---
#### [new 040] Regional Bias in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI公平性研究任务，旨在解决语言模型中的区域偏见问题。通过设计测试集和FAZE框架评估10个模型，发现不同模型存在显著偏见差异。**

- **链接: [https://arxiv.org/pdf/2601.16349v1](https://arxiv.org/pdf/2601.16349v1)**

> **作者:** M P V S Gopinadh; Kappara Lakshmi Sindhu; Soma Sekhar Pandu Ranga Raju P; Yesaswini Swarna
>
> **备注:** 8 pages, 1 figure. Presented at the Second International Conference on Advanced Computing, Machine Learning, Robotics and Internet Technologies (AMRIT 2024)
>
> **摘要:** This study investigates regional bias in large language models (LLMs), an emerging concern in AI fairness and global representation. We evaluate ten prominent LLMs: GPT-3.5, GPT-4o, Gemini 1.5 Flash, Gemini 1.0 Pro, Claude 3 Opus, Claude 3.5 Sonnet, Llama 3, Gemma 7B, Mistral 7B, and Vicuna-13B using a dataset of 100 carefully designed prompts that probe forced-choice decisions between regions under contextually neutral scenarios. We introduce FAZE, a prompt-based evaluation framework that measures regional bias on a 10-point scale, where higher scores indicate a stronger tendency to favor specific regions. Experimental results reveal substantial variation in bias levels across models, with GPT-3.5 exhibiting the highest bias score (9.5) and Claude 3.5 Sonnet scoring the lowest (2.5). These findings indicate that regional bias can meaningfully undermine the reliability, fairness, and inclusivity of LLM outputs in real-world, cross-cultural applications. This work contributes to AI fairness research by highlighting the importance of inclusive evaluation frameworks and systematic approaches for identifying and mitigating geographic biases in language models.
>
---
#### [new 041] MultiLexNorm++: A Unified Benchmark and a Generative Model for Lexical Normalization for Asian Languages
- **分类: cs.CL**

- **简介: 该论文属于词汇规范化任务，旨在解决社交媒体数据中非标准语言表达带来的NLP处理难题。工作包括构建覆盖5种亚洲语言的基准集，并提出基于大语言模型的改进方法。**

- **链接: [https://arxiv.org/pdf/2601.16623v1](https://arxiv.org/pdf/2601.16623v1)**

> **作者:** Weerayut Buaphet; Thanh-Nhi Nguyen; Risa Kondo; Tomoyuki Kajiwara; Yumin Kim; Jimin Lee; Hwanhee Lee; Holy Lovenia; Peerat Limkonchotiwat; Sarana Nutanong; Rob Van der Goot
>
> **摘要:** Social media data has been of interest to Natural Language Processing (NLP) practitioners for over a decade, because of its richness in information, but also challenges for automatic processing. Since language use is more informal, spontaneous, and adheres to many different sociolects, the performance of NLP models often deteriorates. One solution to this problem is to transform data to a standard variant before processing it, which is also called lexical normalization. There has been a wide variety of benchmarks and models proposed for this task. The MultiLexNorm benchmark proposed to unify these efforts, but it consists almost solely of languages from the Indo-European language family in the Latin script. Hence, we propose an extension to MultiLexNorm, which covers 5 Asian languages from different language families in 4 different scripts. We show that the previous state-of-the-art model performs worse on the new languages and propose a new architecture based on Large Language Models (LLMs), which shows more robust performance. Finally, we analyze remaining errors, revealing future directions for this task.
>
---
#### [new 042] M3Kang: Evaluating Multilingual Multimodal Mathematical Reasoning in Vision-Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出M3Kang数据集，用于评估视觉-语言模型在多语言数学推理中的表现，解决多语言多模态数学推理能力不足的问题。**

- **链接: [https://arxiv.org/pdf/2601.16218v1](https://arxiv.org/pdf/2601.16218v1)**

> **作者:** Aleix Torres-Camps; Nathaniel Mitrani Hadida; Víctor Conchello Vendrell; Àlex Batlle Casellas; Arnau Padrés Masdemont; Jordi Ros-Giralt
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** Despite state-of-the-art vision-language models (VLMs) have demonstrated strong reasoning capabilities, their performance in multilingual mathematical reasoning remains underexplored, particularly when compared to human performance. To bridge this gap, we introduce M3Kang, the first massively multilingual, multimodal mathematical reasoning dataset for VLMs. It is derived from the Kangaroo Math Competition, the world's largest mathematics contest, which annually engages over six million participants under the age of 18 across more than 90 countries. M3Kang includes 1,747 unique multiple-choice problems organized by grade-level difficulty, with translations into 108 culturally diverse languages, some of them including diagrams essential for solving them. Using this dataset, we conduct extensive benchmarking on both closed- and open-source SOTA models. We observe that, despite recent advances, models still struggle with basic math and diagram-based reasoning, with performance scaling with language presence and model size, but not with grade level. We also find that multilingual techniques can be effectively extended to the multimodal setting, resulting in significant improvements over baseline approaches. Our analysis also incorporates performance data from over 68,000 students, enabling direct comparison with human performance. We are open-sourcing M3Kang, including the English-only subset M2Kang, along with the framework and codebase used to construct the dataset.
>
---
#### [new 043] EMemBench: Interactive Benchmarking of Episodic Memory for VLM Agents
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出EMemBench，用于评估视觉语言模型代理的长期记忆能力。通过交互式游戏生成问题，测试多类记忆技能，发现视觉场景下记忆仍具挑战。**

- **链接: [https://arxiv.org/pdf/2601.16690v1](https://arxiv.org/pdf/2601.16690v1)**

> **作者:** Xinze Li; Ziyue Zhu; Siyuan Liu; Yubo Ma; Yuhang Zang; Yixin Cao; Aixin Sun
>
> **备注:** 25 pages
>
> **摘要:** We introduce EMemBench, a programmatic benchmark for evaluating long-term memory of agents through interactive games. Rather than using a fixed set of questions, EMemBench generates questions from each agent's own trajectory, covering both text and visual game environments. Each template computes verifiable ground truth from underlying game signals, with controlled answerability and balanced coverage over memory skills: single/multi-hop recall, induction, temporal, spatial, logical, and adversarial. We evaluate memory agents with strong LMs/VLMs as backbones, using in-context prompting as baselines. Across 15 text games and multiple visual seeds, results are far from saturated: induction and spatial reasoning are persistent bottlenecks, especially in visual setting. Persistent memory yields clear gains for open backbones on text games, but improvements are less consistent for VLM agents, suggesting that visually grounded episodic memory remains an open challenge. A human study further confirms the difficulty of EMemBench.
>
---
#### [new 044] Trapped in the past? Disentangling fluid and crystallized intelligence of large language models using chess
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于人工智能领域，旨在区分大语言模型的流体智能与晶体智能。通过国际象棋测试，分析模型在不同推理需求下的表现，揭示其泛化能力的局限性。**

- **链接: [https://arxiv.org/pdf/2601.16823v1](https://arxiv.org/pdf/2601.16823v1)**

> **作者:** Leonard S. Pleiss; Maximilian Schiffer; Robert K. von Weizsäcker
>
> **摘要:** Large Language Models (LLMs) exhibit remarkable capabilities, yet it remains unclear to what extent these reflect sophisticated recall (crystallized intelligence) or reasoning ability (fluid intelligence). We introduce chess as a controlled testbed for disentangling these faculties. Leveraging the game's structure and scalable engine evaluations, we construct a taxonomy of positions varying in training corpus proximity--ranging from common states solvable by memorization to novel ones requiring first-principles reasoning. We systematically evaluate multiple GPT generations under varying reasoning intensities. Our analysis reveals a clear gradient: performance consistently degrades as fluid intelligence demands increase. Notably, in out-of-distribution tasks, performance collapses to random levels. While newer models improve, progress slows significantly for tasks outside the training distribution. Furthermore, while reasoning-augmented inference improves performance, its marginal benefit per token decreases with distributional proximity. These results suggest current architectures remain limited in systematic generalization, highlighting the need for mechanisms beyond scale to achieve robust fluid intelligence.
>
---
#### [new 045] Better as Generators Than Classifiers: Leveraging LLMs and Synthetic Data for Low-Resource Multilingual Classification
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多语言分类任务，旨在解决低资源语言数据不足的问题。通过LLMs生成合成数据，训练小型模型，结果表明其表现优于大型模型。**

- **链接: [https://arxiv.org/pdf/2601.16278v1](https://arxiv.org/pdf/2601.16278v1)**

> **作者:** Branislav Pecher; Jan Cegin; Robert Belanec; Ivan Srba; Jakub Simko; Maria Bielikova
>
> **备注:** Accepted to the Findings of EACL 2026
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable multilingual capabilities, making them promising tools in both high- and low-resource languages. One particularly valuable use case is generating synthetic samples that can be used to train smaller models in low-resource scenarios where human-labelled data is scarce. In this work, we investigate whether these synthetic data generation capabilities can serve as a form of distillation, producing smaller models that perform on par with or even better than massive LLMs across languages and tasks. To this end, we use a state-of-the-art multilingual LLM to generate synthetic datasets covering 11 languages and 4 classification tasks. These datasets are then used to train smaller models via fine-tuning or instruction tuning, or as synthetic in-context examples for compact LLMs. Our experiments show that even small amounts of synthetic data enable smaller models to outperform the large generator itself, particularly in low-resource languages. Overall, the results suggest that LLMs are best utilised as generators (teachers) rather than classifiers, producing data that empowers smaller and more efficient multilingual models.
>
---
#### [new 046] LOGICAL-COMMONSENSEQA: A Benchmark for Logical Commonsense Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LOGICAL-COMMONSENSEQA基准，用于逻辑常识推理任务，解决传统评估方法无法准确衡量多重解释的问题。通过逻辑运算符构建测试集，评估模型在不同推理模式下的表现。**

- **链接: [https://arxiv.org/pdf/2601.16504v1](https://arxiv.org/pdf/2601.16504v1)**

> **作者:** Obed Junias; Maria Leonor Pacheco
>
> **摘要:** Commonsense reasoning often involves evaluating multiple plausible interpretations rather than selecting a single atomic answer, yet most benchmarks rely on single-label evaluation, obscuring whether statements are jointly plausible, mutually exclusive, or jointly implausible. We introduce LOGICAL-COMMONSENSEQA, a benchmark that re-frames commonsense reasoning as logical composition over pairs of atomic statements using plausibility-level operators (AND, OR, NEITHER/NOR). Evaluating instruction-tuned, reasoning-specialized, and fine-tuned models under zero-shot, few-shot, and chain-of-thought prompting, we find that while models perform reasonably on conjunctive and moderately on disjunctive reasoning, performance degrades sharply on negation-based questions. LOGICAL-COMMONSENSEQA exposes fundamental reasoning limitations and provides a controlled framework for advancing compositional commonsense reasoning.
>
---
#### [new 047] Cross-Lingual Activation Steering for Multilingual Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言模型任务，旨在解决非主导语言性能不足的问题。通过CLAS方法，在不修改模型权重的情况下调节神经元激活，提升模型的多语言能力。**

- **链接: [https://arxiv.org/pdf/2601.16390v1](https://arxiv.org/pdf/2601.16390v1)**

> **作者:** Rhitabrat Pokharel; Ameeta Agrawal; Tanay Nagar
>
> **备注:** Under review
>
> **摘要:** Large language models exhibit strong multilingual capabilities, yet significant performance gaps persist between dominant and non-dominant languages. Prior work attributes this gap to imbalances between shared and language-specific neurons in multilingual representations. We propose Cross-Lingual Activation Steering (CLAS), a training-free inference-time intervention that selectively modulates neuron activations. We evaluate CLAS on classification and generation benchmarks, achieving average improvements of 2.3% (Acc.) and 3.4% (F1) respectively, while maintaining high-resource language performance. We discover that effective transfer operates through functional divergence rather than strict alignment; performance gains correlate with increased language cluster separation. Our results demonstrate that targeted activation steering can unlock latent multilingual capacity in existing models without modification to model weights.
>
---
#### [new 048] Select or Project? Evaluating Lower-dimensional Vectors for LLM Training Data Explanations
- **分类: cs.CL**

- **简介: 该论文研究如何高效生成大语言模型的实例解释，解决高维梯度计算难题。通过选择性子集或投影降低维度，发现选择性方法更有效且高效。任务属于模型解释与优化。**

- **链接: [https://arxiv.org/pdf/2601.16651v1](https://arxiv.org/pdf/2601.16651v1)**

> **作者:** Lukas Hinterleitner; Loris Schoenegger; Benjamin Roth
>
> **备注:** 8 pages
>
> **摘要:** Gradient-based methods for instance-based explanation for large language models (LLMs) are hindered by the immense dimensionality of model gradients. In practice, influence estimation is restricted to a subset of model parameters to make computation tractable, but this subset is often chosen ad hoc and rarely justified by systematic evaluation. This paper investigates if it is better to create low-dimensional representations by selecting a small, architecturally informed subset of model components or by projecting the full gradients into a lower-dimensional space. Using a novel benchmark, we show that a greedily selected subset of components captures the information about training data influence needed for a retrieval task more effectively than either the full gradient or random projection. We further find that this approach is more computationally efficient than random projection, demonstrating that targeted component selection is a practical strategy for making instance-based explanations of large models more computationally feasible.
>
---
#### [new 049] Graph-Anchored Knowledge Indexing for Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于知识增强生成任务，旨在解决RAG系统在整合分散证据时的挑战。提出GraphAnchor方法，通过动态图结构提升知识索引与检索效果。**

- **链接: [https://arxiv.org/pdf/2601.16462v1](https://arxiv.org/pdf/2601.16462v1)**

> **作者:** Zhenghao Liu; Mingyan Wu; Xinze Li; Yukun Yan; Shuo Wang; Cheng Yang; Minghe Yu; Zheni Zeng; Maosong Sun
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a dominant paradigm for mitigating hallucinations in Large Language Models (LLMs) by incorporating external knowledge. Nevertheless, effectively integrating and interpreting key evidence scattered across noisy documents remains a critical challenge for existing RAG systems. In this paper, we propose GraphAnchor, a novel Graph-Anchored Knowledge Indexing approach that reconceptualizes graph structures from static knowledge representations into active, evolving knowledge indices. GraphAnchor incrementally updates a graph during iterative retrieval to anchor salient entities and relations, yielding a structured index that guides the LLM in evaluating knowledge sufficiency and formulating subsequent subqueries. The final answer is generated by jointly leveraging all retrieved documents and the final evolved graph. Experiments on four multi-hop question answering benchmarks demonstrate the effectiveness of GraphAnchor, and reveal that GraphAnchor modulates the LLM's attention to more effectively associate key information distributed in retrieved documents. All code and data are available at https://github.com/NEUIR/GraphAnchor.
>
---
#### [new 050] Cite-While-You-Generate: Training-Free Evidence Attribution for Multimodal Clinical Summarization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床摘要任务，解决生成内容来源不透明的问题。提出无需训练的引用框架，利用解码器注意力直接引用文本或图像，提升摘要的可解释性与准确性。**

- **链接: [https://arxiv.org/pdf/2601.16397v1](https://arxiv.org/pdf/2601.16397v1)**

> **作者:** Qianqi Yan; Huy Nguyen; Sumana Srivatsa; Hari Bandi; Xin Eric Wang; Krishnaram Kenthapadi
>
> **摘要:** Trustworthy clinical summarization requires not only fluent generation but also transparency about where each statement comes from. We propose a training-free framework for generation-time source attribution that leverages decoder attentions to directly cite supporting text spans or images, overcoming the limitations of post-hoc or retraining-based methods. We introduce two strategies for multimodal attribution: a raw image mode, which directly uses image patch attentions, and a caption-as-span mode, which substitutes images with generated captions to enable purely text-based alignment. Evaluations on two representative domains: clinician-patient dialogues (CliConSummation) and radiology reports (MIMIC-CXR), show that our approach consistently outperforms embedding-based and self-attribution baselines, improving both text-level and multimodal attribution accuracy (e.g., +15% F1 over embedding baselines). Caption-based attribution achieves competitive performance with raw-image attention while being more lightweight and practical. These findings highlight attention-guided attribution as a promising step toward interpretable and deployable clinical summarization systems.
>
---
#### [new 051] Is Length Really A Liability? An Evaluation of Multi-turn LLM Conversations using BoolQ
- **分类: cs.CL**

- **简介: 该论文属于语言模型评估任务，旨在解决多轮对话中长度对回答真实性影响的问题。通过BoolQ数据集测试，发现模型在多轮对话中的特定漏洞。**

- **链接: [https://arxiv.org/pdf/2601.16508v1](https://arxiv.org/pdf/2601.16508v1)**

> **作者:** Karl Neergaard; Le Qiu; Emmanuele Chersoni
>
> **备注:** 4 pages plus 6 pages of bibliography and appendix
>
> **摘要:** Single-prompt evaluations dominate current LLM benchmarking, yet they fail to capture the conversational dynamics where real-world harm occurs. In this study, we examined whether conversation length affects response veracity by evaluating LLM performance on the BoolQ dataset under varying length and scaffolding conditions. Our results across three distinct LLMs revealed model-specific vulnerabilities that are invisible under single-turn testing. The length-dependent and scaffold-specific effects we observed demonstrate a fundamental limitation of static evaluations, as deployment-relevant vulnerabilities could only be spotted in a multi-turn conversational setting.
>
---
#### [new 052] Information Representation Fairness in Long-Document Embeddings: The Peculiar Interaction of Positional and Language Bias
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究长文档嵌入中的信息表示公平性问题，旨在解决位置和语言偏见。通过引入评估框架，发现早期段落和高资源语言被过度表示，提出注意力校准方法以改善公平性。**

- **链接: [https://arxiv.org/pdf/2601.16934v1](https://arxiv.org/pdf/2601.16934v1)**

> **作者:** Elias Schuhmacher; Andrianos Michail; Juri Opitz; Rico Sennrich; Simon Clematide
>
> **摘要:** To be discoverable in an embedding-based search process, each part of a document should be reflected in its embedding representation. To quantify any potential reflection biases, we introduce a permutation-based evaluation framework. With this, we observe that state-of-the-art embedding models exhibit systematic positional and language biases when documents are longer and consist of multiple segments. Specifically, early segments and segments in higher-resource languages like English are over-represented, while later segments and segments in lower-resource languages are marginalized. In our further analysis, we find that the positional bias stems from front-loaded attention distributions in pooling-token embeddings, where early tokens receive more attention. To mitigate this issue, we introduce an inference-time attention calibration method that redistributes attention more evenly across document positions, increasing discoverabiltiy of later segments. Our evaluation framework and attention calibration is available at https://github.com/impresso/fair-sentence-transformers
>
---
#### [new 053] PLawBench: A Rubric-Based Benchmark for Evaluating LLMs in Real-World Legal Practice
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出PLawBench，用于评估大语言模型在真实法律场景中的表现。解决现有基准过于简化、缺乏细粒度法律推理评估的问题。通过三个任务类别进行测试，涵盖法律咨询、案例分析和文书生成。**

- **链接: [https://arxiv.org/pdf/2601.16669v1](https://arxiv.org/pdf/2601.16669v1)**

> **作者:** Yuzhen Shi; Huanghai Liu; Yiran Hu; Gaojie Song; Xinran Xu; Yubo Ma; Tianyi Tang; Li Zhang; Qingjing Chen; Di Feng; Wenbo Lv; Weiheng Wu; Kexin Yang; Sen Yang; Wei Wang; Rongyao Shi; Yuanyang Qiu; Yuemeng Qi; Jingwen Zhang; Xiaoyu Sui; Yifan Chen; Yi Zhang; An Yang; Bowen Yu; Dayiheng Liu; Junyang Lin; Weixing Shen; Bing Zhao; Charles L. A. Clarke; Hu Wei
>
> **摘要:** As large language models (LLMs) are increasingly applied to legal domain-specific tasks, evaluating their ability to perform legal work in real-world settings has become essential. However, existing legal benchmarks rely on simplified and highly standardized tasks, failing to capture the ambiguity, complexity, and reasoning demands of real legal practice. Moreover, prior evaluations often adopt coarse, single-dimensional metrics and do not explicitly assess fine-grained legal reasoning. To address these limitations, we introduce PLawBench, a Practical Law Benchmark designed to evaluate LLMs in realistic legal practice scenarios. Grounded in real-world legal workflows, PLawBench models the core processes of legal practitioners through three task categories: public legal consultation, practical case analysis, and legal document generation. These tasks assess a model's ability to identify legal issues and key facts, perform structured legal reasoning, and generate legally coherent documents. PLawBench comprises 850 questions across 13 practical legal scenarios, with each question accompanied by expert-designed evaluation rubrics, resulting in approximately 12,500 rubric items for fine-grained assessment. Using an LLM-based evaluator aligned with human expert judgments, we evaluate 10 state-of-the-art LLMs. Experimental results show that none achieves strong performance on PLawBench, revealing substantial limitations in the fine-grained legal reasoning capabilities of current LLMs and highlighting important directions for future evaluation and development of legal LLMs. Data is available at: https://github.com/skylenage/PLawbench.
>
---
#### [new 054] EvoConfig: Self-Evolving Multi-Agent Systems for Efficient Autonomous Environment Configuration
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出EvoConfig，解决大语言模型在软件工程任务中环境配置效率低的问题。通过多智能体协作与自进化机制提升配置成功率和错误修复能力。**

- **链接: [https://arxiv.org/pdf/2601.16489v1](https://arxiv.org/pdf/2601.16489v1)**

> **作者:** Xinshuai Guo; Jiayi Kuang; Linyue Pan; Yinghui Li; Yangning Li; Hai-Tao Zheng; Ying Shen; Di Yin; Xing Sun
>
> **摘要:** A reliable executable environment is the foundation for ensuring that large language models solve software engineering tasks. Due to the complex and tedious construction process, large-scale configuration is relatively inefficient. However, most methods always overlook fine-grained analysis of the actions performed by the agent, making it difficult to handle complex errors and resulting in configuration failures. To address this bottleneck, we propose EvoConfig, an efficient environment configuration framework that optimizes multi-agent collaboration to build correct runtime environments. EvoConfig features an expert diagnosis module for fine-grained post-execution analysis, and a self-evolving mechanism that lets expert agents self-feedback and dynamically adjust error-fixing priorities in real time. Empirically, EvoConfig matches the previous state-of-the-art Repo2Run on Repo2Run's 420 repositories, while delivering clear gains on harder cases: on the more challenging Envbench, EvoConfig achieves a 78.1% success rate, outperforming Repo2Run by 7.1%. Beyond end-to-end success, EvoConfig also demonstrates stronger debugging competence, achieving higher accuracy in error identification and producing more effective repair recommendations than existing methods.
>
---
#### [new 055] EdgeSpot: Efficient and High-Performance Few-Shot Model for Keyword Spotting
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出EdgeSpot模型，用于边缘设备的关键词检测任务，解决少样本下准确率与效率的问题。通过优化网络结构和知识蒸馏提升性能。**

- **链接: [https://arxiv.org/pdf/2601.16316v1](https://arxiv.org/pdf/2601.16316v1)**

> **作者:** Oguzhan Buyuksolak; Alican Gok; Osman Erman Okman
>
> **备注:** Accepted to be presented in IEEE ICASSP 2026
>
> **摘要:** We introduce an efficient few-shot keyword spotting model for edge devices, EdgeSpot, that pairs an optimized version of a BC-ResNet-based acoustic backbone with a trainable Per-Channel Energy Normalization frontend and lightweight temporal self-attention. Knowledge distillation is utilized during training by employing a self-supervised teacher model, optimized with Sub-center ArcFace loss. This study demonstrates that the EdgeSpot model consistently provides better accuracy at a fixed false-alarm rate (FAR) than strong BC-ResNet baselines. The largest variant, EdgeSpot-4, improves the 10-shot accuracy at 1% FAR from 73.7% to 82.0%, which requires only 29.4M MACs with 128k parameters.
>
---
#### [new 056] Reasoning Promotes Robustness in Theory of Mind Tasks
- **分类: cs.AI; cs.CL**

- **简介: 论文研究推理模型在心智理论任务中的表现，探讨其鲁棒性提升的原因。旨在解决LLM在社交认知任务中的评估问题，通过实验分析发现其优势源于解题稳定性而非新推理方式。**

- **链接: [https://arxiv.org/pdf/2601.16853v1](https://arxiv.org/pdf/2601.16853v1)**

> **作者:** Ian B. de Haan; Peter van der Putten; Max van Duijn
>
> **备注:** 14 pages, 2 figures
>
> **摘要:** Large language models (LLMs) have recently shown strong performance on Theory of Mind (ToM) tests, prompting debate about the nature and true performance of the underlying capabilities. At the same time, reasoning-oriented LLMs trained via reinforcement learning with verifiable rewards (RLVR) have achieved notable improvements across a range of benchmarks. This paper examines the behavior of such reasoning models in ToM tasks, using novel adaptations of machine psychological experiments and results from established benchmarks. We observe that reasoning models consistently exhibit increased robustness to prompt variations and task perturbations. Our analysis indicates that the observed gains are more plausibly attributed to increased robustness in finding the correct solution, rather than to fundamentally new forms of ToM reasoning. We discuss the implications of this interpretation for evaluating social-cognitive behavior in LLMs.
>
---
#### [new 057] A Collision-Free Hot-Tier Extension for Engram-Style Conditional Memory: A Controlled Study of Training Dynamics
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于条件记忆任务，研究高频率键冲突对Engram模型的影响。通过设计无冲突的热层结构，发现冲突可能起到正则化作用，提升训练效果需关注门控机制而非仅提高查找精度。**

- **链接: [https://arxiv.org/pdf/2601.16531v1](https://arxiv.org/pdf/2601.16531v1)**

> **作者:** Tao Lin
>
> **摘要:** We investigate whether high-frequency key collisions are a primary bottleneck in Engram-style conditional memory. To isolate the effect of collisions, we introduce Engram-Nine, a collision-free hot-tier extension that maps the most frequent n-grams through a Minimal Perfect Hash Function (MPHF) while retaining the original multi-head hashed lookup as a cold tier. Under a strictly iso-parameter setup, the collision-free design does not consistently improve validation loss. Through route-stratified evaluation (decomposing per-token loss into hot/cold contributions), we uncover a consistent "hot-to-cold advantage flip" during training: hot (high-frequency) positions initially have lower loss, but cold positions eventually surpass them. Crucially, collision-free configurations flip earlier than collision-prone baselines, suggesting that collisions act as implicit regularization. We also identify a gating mismatch: the gate learns to favor hot positions early in training, but this preference persists even after the flip, assigning higher weights to positions with higher loss. Our findings suggest that improving lookup precision alone does not guarantee better training outcomes. The dominant limitation may lie in gating credit assignment rather than index accuracy, and collision-induced noise may provide beneficial regularization that should not be naively eliminated.
>
---
#### [new 058] Zero-Shot Speech LLMs for Multi-Aspect Evaluation of L2 Speech: Challenges and Opportunities
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语言评估任务，旨在解决L2英语发音自动评分难题。研究评估了Qwen2-Audio-7B-Instruct在5000个语音样本上的零样本性能，探讨其在准确性、流利度等方面的评估能力。**

- **链接: [https://arxiv.org/pdf/2601.16230v1](https://arxiv.org/pdf/2601.16230v1)**

> **作者:** Aditya Kamlesh Parikh; Cristian Tejedor-Garcia; Catia Cucchiarini; Helmer Strik
>
> **备注:** This publication is part of the project Responsible AI for Voice Diagnostics (RAIVD) with file number NGF.1607.22.013 of the research programme NGF AiNed Fellowship Grants which is financed by the Dutch Research Council (NWO)
>
> **摘要:** An accurate assessment of L2 English pronunciation is crucial for language learning, as it provides personalized feedback and ensures a fair evaluation of individual progress. However, automated scoring remains challenging due to the complexity of sentence-level fluency, prosody, and completeness. This paper evaluates the zero-shot performance of Qwen2-Audio-7B-Instruct, an instruction-tuned speech-LLM, on 5,000 Speechocean762 utterances. The model generates rubric-aligned scores for accuracy, fluency, prosody, and completeness, showing strong agreement with human ratings within +-2 tolerance, especially for high-quality speech. However, it tends to overpredict low-quality speech scores and lacks precision in error detection. These findings demonstrate the strong potential of speech LLMs in scalable pronunciation assessment and suggest future improvements through enhanced prompting, calibration, and phonetic integration to advance Computer-Assisted Pronunciation Training.
>
---
#### [new 059] ColorConceptBench: A Benchmark for Probabilistic Color-Concept Understanding in Text-to-Image Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文本到图像生成任务，旨在解决模型对隐含颜色概念理解不足的问题。通过构建ColorConceptBench基准，评估模型的色彩语义能力。**

- **链接: [https://arxiv.org/pdf/2601.16836v1](https://arxiv.org/pdf/2601.16836v1)**

> **作者:** Chenxi Ruan; Yu Xiao; Yihan Hou; Guosheng Hu; Wei Zeng
>
> **摘要:** While text-to-image (T2I) models have advanced considerably, their capability to associate colors with implicit concepts remains underexplored. To address the gap, we introduce ColorConceptBench, a new human-annotated benchmark to systematically evaluate color-concept associations through the lens of probabilistic color distributions. ColorConceptBench moves beyond explicit color names or codes by probing how models translate 1,281 implicit color concepts using a foundation of 6,369 human annotations. Our evaluation of seven leading T2I models reveals that current models lack sensitivity to abstract semantics, and crucially, this limitation appears resistant to standard interventions (e.g., scaling and guidance). This demonstrates that achieving human-like color semantics requires more than larger models, but demands a fundamental shift in how models learn and represent implicit meaning.
>
---
#### [new 060] Endless Terminals: Scaling RL Environments for Terminal Agents
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Endless Terminals，一个自动生成终端任务的强化学习环境管道，解决传统基准不适合训练的问题。通过简单RL方法提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.16443v1](https://arxiv.org/pdf/2601.16443v1)**

> **作者:** Kanishk Gandhi; Shivam Garg; Noah D. Goodman; Dimitris Papailiopoulos
>
> **摘要:** Environments are the bottleneck for self-improving agents. Current terminal benchmarks were built for evaluation, not training; reinforcement learning requires a scalable pipeline, not just a dataset. We introduce Endless Terminals, a fully autonomous pipeline that procedurally generates terminal-use tasks without human annotation. The pipeline has four stages: generating diverse task descriptions, building and validating containerized environments, producing completion tests, and filtering for solvability. From this pipeline we obtain 3255 tasks spanning file operations, log management, data processing, scripting, and database operations. We train agents using vanilla PPO with binary episode level rewards and a minimal interaction loop: no retrieval, multi-agent coordination, or specialized tools. Despite this simplicity, models trained on Endless Terminals show substantial gains: on our held-out dev set, Llama-3.2-3B improves from 4.0% to 18.2%, Qwen2.5-7B from 10.7% to 53.3%, and Qwen3-8B-openthinker-sft from 42.6% to 59.0%. These improvements transfer to human-curated benchmarks: models trained on Endless Terminals show substantial gains on held out human curated benchmarks: on TerminalBench 2.0, Llama-3.2-3B improves from 0.0% to 2.2%, Qwen2.5-7B from 2.2% to 3.4%, and Qwen3-8B-openthinker-sft from 1.1% to 6.7%, in each case outperforming alternative approaches including models with more complex agentic scaffolds. These results demonstrate that simple RL succeeds when environments scale.
>
---
#### [new 061] Beyond Superficial Unlearning: Sharpness-Aware Robust Erasure of Hallucinations in Multimodal LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态大模型的幻觉消除任务，解决模型生成不存在实体的问题。提出SARE方法，通过几何稳定化实现鲁棒消融。**

- **链接: [https://arxiv.org/pdf/2601.16527v1](https://arxiv.org/pdf/2601.16527v1)**

> **作者:** Xianya Fang; Feiyang Ren; Xiang Chen; Yu Tian; Zhen Bi; Haiyang Yu; Sheng-Jun Huang
>
> **摘要:** Multimodal LLMs are powerful but prone to object hallucinations, which describe non-existent entities and harm reliability. While recent unlearning methods attempt to mitigate this, we identify a critical flaw: structural fragility. We empirically demonstrate that standard erasure achieves only superficial suppression, trapping the model in sharp minima where hallucinations catastrophically resurge after lightweight relearning. To ensure geometric stability, we propose SARE, which casts unlearning as a targeted min-max optimization problem and uses a Targeted-SAM mechanism to explicitly flatten the loss landscape around hallucinated concepts. By suppressing hallucinations under simulated worst-case parameter perturbations, our framework ensures robust removal stable against weight shifts. Extensive experiments demonstrate that SARE significantly outperforms baselines in erasure efficacy while preserving general generation quality. Crucially, it maintains persistent hallucination suppression against relearning and parameter updates, validating the effectiveness of geometric stabilization.
>
---
#### [new 062] SoundBreak: A Systematic Study of Audio-Only Adversarial Attacks on Trimodal Models
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; eess.AS**

- **简介: 该论文研究音频对抗攻击对多模态模型的影响，旨在揭示单模态攻击漏洞。通过分析不同攻击目标，验证音频扰动可导致多模态系统失效。**

- **链接: [https://arxiv.org/pdf/2601.16231v1](https://arxiv.org/pdf/2601.16231v1)**

> **作者:** Aafiya Hussain; Gaurav Srivastava; Alvi Ishmam; Zaber Hakim; Chris Thomas
>
> **摘要:** Multimodal foundation models that integrate audio, vision, and language achieve strong performance on reasoning and generation tasks, yet their robustness to adversarial manipulation remains poorly understood. We study a realistic and underexplored threat model: untargeted, audio-only adversarial attacks on trimodal audio-video-language models. We analyze six complementary attack objectives that target different stages of multimodal processing, including audio encoder representations, cross-modal attention, hidden states, and output likelihoods. Across three state-of-the-art models and multiple benchmarks, we show that audio-only perturbations can induce severe multimodal failures, achieving up to 96% attack success rate. We further show that attacks can be successful at low perceptual distortions (LPIPS <= 0.08, SI-SNR >= 0) and benefit more from extended optimization than increased data scale. Transferability across models and encoders remains limited, while speech recognition systems such as Whisper primarily respond to perturbation magnitude, achieving >97% attack success under severe distortion. These results expose a previously overlooked single-modality attack surface in multimodal systems and motivate defenses that enforce cross-modal consistency.
>
---
#### [new 063] Where is the multimodal goal post? On the Ability of Foundation Models to Recognize Contextually Important Moments
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究多模态模型识别视频中关键事件的能力，针对足球比赛视频进行分析，旨在提升模型在多模态数据中的信息融合能力。**

- **链接: [https://arxiv.org/pdf/2601.16333v1](https://arxiv.org/pdf/2601.16333v1)**

> **作者:** Aditya K Surikuchi; Raquel Fernández; Sandro Pezzelle
>
> **摘要:** Foundation models are used for many real-world applications involving language generation from temporally-ordered multimodal events. In this work, we study the ability of models to identify the most important sub-events in a video, which is a fundamental prerequisite for narrating or summarizing multimodal events. Specifically, we focus on football games and evaluate models on their ability to distinguish between important and non-important sub-events in a game. To this end, we construct a new dataset by leveraging human preferences for importance implicit in football game highlight reels, without any additional annotation costs. Using our dataset, which we will publicly release to the community, we compare several state-of-the-art multimodal models and show that they are not far from chance level performance. Analyses of models beyond standard evaluation metrics reveal their tendency to rely on a single dominant modality and their ineffectiveness in synthesizing necessary information from multiple sources. Our findings underline the importance of modular architectures that can handle sample-level heterogeneity in multimodal data and the need for complementary training procedures that can maximize cross-modal synergy.
>
---
#### [new 064] White-Box Sensitivity Auditing with Steering Vectors
- **分类: cs.CY; cs.CL; cs.LG**

- **简介: 该论文属于模型审计任务，旨在解决LLM中隐性偏见检测问题。通过白盒方法，利用激活控制进行内部敏感性测试，揭示模型对保护属性的依赖性。**

- **链接: [https://arxiv.org/pdf/2601.16398v1](https://arxiv.org/pdf/2601.16398v1)**

> **作者:** Hannah Cyberey; Yangfeng Ji; David Evans
>
> **摘要:** Algorithmic audits are essential tools for examining systems for properties required by regulators or desired by operators. Current audits of large language models (LLMs) primarily rely on black-box evaluations that assess model behavior only through input-output testing. These methods are limited to tests constructed in the input space, often generated by heuristics. In addition, many socially relevant model properties (e.g., gender bias) are abstract and difficult to measure through text-based inputs alone. To address these limitations, we propose a white-box sensitivity auditing framework for LLMs that leverages activation steering to conduct more rigorous assessments through model internals. Our auditing method conducts internal sensitivity tests by manipulating key concepts relevant to the model's intended function for the task. We demonstrate its application to bias audits in four simulated high-stakes LLM decision tasks. Our method consistently reveals substantial dependence on protected attributes in model predictions, even in settings where standard black-box evaluations suggest little or no bias. Our code is openly available at https://github.com/hannahxchen/llm-steering-audit
>
---
#### [new 065] TangramPuzzle: Evaluating Multimodal Large Language Models with Compositional Spatial Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态大语言模型的评估任务，旨在解决其在组合空间推理上的不足。提出TangramPuzzle基准和TCE框架，通过几何任务评估模型的空间推理能力。**

- **链接: [https://arxiv.org/pdf/2601.16520v1](https://arxiv.org/pdf/2601.16520v1)**

> **作者:** Daixian Liu; Jiayi Kuang; Yinghui Li; Yangning Li; Di Yin; Haoyu Cao; Xing Sun; Ying Shen; Hai-Tao Zheng; Liang Lin; Philip S. Yu
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved remarkable progress in visual recognition and semantic understanding. Nevertheless, their ability to perform precise compositional spatial reasoning remains largely unexplored. Existing benchmarks often involve relatively simple tasks and rely on semantic approximations or coarse relative positioning, while their evaluation metrics are typically limited and lack rigorous mathematical formulations. To bridge this gap, we introduce TangramPuzzle, a geometry-grounded benchmark designed to evaluate compositional spatial reasoning through the lens of the classic Tangram game. We propose the Tangram Construction Expression (TCE), a symbolic geometric framework that grounds tangram assemblies in exact, machine-verifiable coordinate specifications, to mitigate the ambiguity of visual approximation. We design two complementary tasks: Outline Prediction, which demands inferring global shapes from local components, and End-to-End Code Generation, which requires solving inverse geometric assembly problems. We conduct extensive evaluation experiments on advanced open-source and proprietary models, revealing an interesting insight: MLLMs tend to prioritize matching the target silhouette while neglecting geometric constraints, leading to distortions or deformations of the pieces.
>
---
#### [new 066] SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出SWE-Pruner，解决编码代理中长上下文导致的高成本与延迟问题。通过自适应剪枝保留关键信息，提升效率。**

- **链接: [https://arxiv.org/pdf/2601.16746v1](https://arxiv.org/pdf/2601.16746v1)**

> **作者:** Yuhang Wang; Yuling Shi; Mo Yang; Rongrui Zhang; Shilin He; Heng Lian; Yuting Chen; Siyu Ye; Kai Cai; Xiaodong Gu
>
> **备注:** Code available at https://github.com/Ayanami1314/swe-pruner
>
> **摘要:** LLM agents have demonstrated remarkable capabilities in software development, but their performance is hampered by long interaction contexts, which incur high API costs and latency. While various context compression approaches such as LongLLMLingua have emerged to tackle this challenge, they typically rely on fixed metrics such as PPL, ignoring the task-specific nature of code understanding. As a result, they frequently disrupt syntactic and logical structure and fail to retain critical implementation details. In this paper, we propose SWE-Pruner, a self-adaptive context pruning framework tailored for coding agents. Drawing inspiration from how human programmers "selectively skim" source code during development and debugging, SWE-Pruner performs task-aware adaptive pruning for long contexts. Given the current task, the agent formulates an explicit goal (e.g., "focus on error handling") as a hint to guide the pruning targets. A lightweight neural skimmer (0.6B parameters) is trained to dynamically select relevant lines from the surrounding context given the goal. Evaluations across four benchmarks and multiple models validate SWE-Pruner's effectiveness in various scenarios, achieving 23-54% token reduction on agent tasks like SWE-Bench Verified and up to 14.84x compression on single-turn tasks like LongCodeQA with minimal performance impact.
>
---
## 更新

#### [replaced 001] Evaluating Adversarial Robustness of Concept Representations in Sparse Autoencoders
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型解释任务，旨在评估稀疏自编码器（SAE）概念表示的对抗鲁棒性。研究发现，微小扰动可有效操控概念解释，表明SAE表示脆弱，需进一步处理以适用于模型监控。**

- **链接: [https://arxiv.org/pdf/2505.16004v2](https://arxiv.org/pdf/2505.16004v2)**

> **作者:** Aaron J. Li; Suraj Srinivas; Usha Bhalla; Himabindu Lakkaraju
>
> **摘要:** Sparse autoencoders (SAEs) are commonly used to interpret the internal activations of large language models (LLMs) by mapping them to human-interpretable concept representations. While existing evaluations of SAEs focus on metrics such as the reconstruction-sparsity tradeoff, human (auto-)interpretability, and feature disentanglement, they overlook a critical aspect: the robustness of concept representations to input perturbations. We argue that robustness must be a fundamental consideration for concept representations, reflecting the fidelity of concept labeling. To this end, we formulate robustness quantification as input-space optimization problems and develop a comprehensive evaluation framework featuring realistic scenarios in which adversarial perturbations are crafted to manipulate SAE representations. Empirically, we find that tiny adversarial input perturbations can effectively manipulate concept-based interpretations in most scenarios without notably affecting the base LLM's activations. Overall, our results suggest that SAE concept representations are fragile and without further denoising or postprocessing they might be ill-suited for applications in model monitoring and oversight.
>
---
#### [replaced 002] GAICo: A Deployed and Extensible Framework for Evaluating Diverse and Multimodal Generative AI Outputs
- **分类: cs.CL**

- **简介: 该论文提出GAICo框架，解决生成式AI输出评估标准化问题，支持多模态和结构化数据比较，提升评估效率与可重复性。**

- **链接: [https://arxiv.org/pdf/2508.16753v4](https://arxiv.org/pdf/2508.16753v4)**

> **作者:** Nitin Gupta; Pallav Koppisetti; Kausik Lakkaraju; Biplav Srivastava
>
> **备注:** 11 pages, 7 figures; accepted at IAAI/AAAI 2026; (updated) extended version
>
> **摘要:** The rapid proliferation of Generative AI (GenAI) into diverse, high-stakes domains necessitates robust and reproducible evaluation methods. However, practitioners often resort to ad-hoc, non-standardized scripts, as common metrics are often unsuitable for specialized, structured outputs (e.g., automated plans, time-series) or holistic comparison across modalities (e.g., text, audio, and image). This fragmentation hinders comparability and slows AI system development. To address this challenge, we present GAICo (Generative AI Comparator): a deployed, open-source Python library that streamlines and standardizes GenAI output comparison. GAICo provides a unified, extensible framework supporting a comprehensive suite of reference-based metrics for unstructured text, specialized structured data formats, and multimedia (images, audio). Its architecture features a high-level API for rapid, end-to-end analysis, from multi-model comparison to visualization and reporting, alongside direct metric access for granular control. We demonstrate GAICo's utility through a detailed case study evaluating and debugging complex, multi-modal AI Travel Assistant pipelines. GAICo empowers AI researchers and developers to efficiently assess system performance, make evaluation reproducible, improve development velocity, and ultimately build more trustworthy AI systems, aligning with the goal of moving faster and safer in AI deployment. Since its release on PyPI in Jun 2025, the tool has been downloaded over 16K times, across versions, by Dec 2025, demonstrating growing community interest.
>
---
#### [replaced 003] Fluent but Foreign: Even Regional LLMs Lack Cultural Alignment
- **分类: cs.CL; cs.AI; cs.CY; physics.soc-ph**

- **简介: 论文研究区域大模型的文化适配问题，发现其未有效反映本地价值观。任务为文化对齐评估，解决模型与本地文化不匹配的问题，通过实证分析和用户研究验证。**

- **链接: [https://arxiv.org/pdf/2505.21548v3](https://arxiv.org/pdf/2505.21548v3)**

> **作者:** Dhruv Agarwal; Anya Shukla; Sunayana Sitaram; Aditya Vashistha
>
> **备注:** Under review
>
> **摘要:** Large language models (LLMs) are used worldwide, yet exhibit Western cultural tendencies. Many countries are now building ``regional'' or ``sovereign'' LLMs, but it remains unclear whether they reflect local values and practices or merely speak local languages. Using India as a case study, we evaluate six Indic and six global LLMs on two dimensions -- values and practices -- grounded in nationally representative surveys and community-sourced QA datasets. Across tasks, Indic models do not align better with Indian norms than global models; in fact, a U.S. respondent is a closer proxy for Indian values than any Indic model. We further run a user study with 115 Indian users and find that writing suggestions from both global and Indic LLMs introduce Westernized or exoticized writing. Prompting and regional fine-tuning fail to recover alignment and can even degrade existing knowledge. We attribute this to scarce culturally grounded data, especially for pretraining. We position cultural evaluation as a first-class requirement alongside multilingual benchmarks and offer a reusable, community-grounded methodology. We call for native, community-authored corpora and thickxwide evaluations to build truly sovereign LLMs.
>
---
#### [replaced 004] VMMU: A Vietnamese Multitask Multimodal Understanding and Reasoning Benchmark
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出VMMU，一个用于评估视觉语言模型在越南语环境下多模态理解与推理能力的基准。解决跨语言多任务多模态理解问题，通过2.5k个问题测试模型整合视觉与文本信息的能力。**

- **链接: [https://arxiv.org/pdf/2508.13680v4](https://arxiv.org/pdf/2508.13680v4)**

> **作者:** Vy Tuong Dang; An Vo; Emilio Villa-Cueva; Quang Tau; Duc Dm; Thamar Solorio; Daeyoung Kim
>
> **摘要:** We introduce VMMU, a Vietnamese Multitask Multimodal Understanding and Reasoning Benchmark designed to evaluate how vision-language models (VLMs) interpret and reason over visual and textual information beyond English. VMMU consists of 2.5k multimodal questions across 7 tasks, covering a diverse range of problem contexts, including STEM problem solving, data interpretation, rule-governed visual reasoning, and abstract visual reasoning. All questions require genuine multimodal integration, rather than reliance on text-only cues or OCR-based shortcuts. We evaluate a diverse set of state-of-the-art proprietary and open-source VLMs on VMMU. Despite strong Vietnamese OCR performance, proprietary models achieve only 66% mean accuracy. Further analysis shows that the primary source of failure is not OCR, but instead multimodal grounding and reasoning over text and visual evidence. Code and data are available at https://vmmu-bench.github.io/
>
---
#### [replaced 005] Exploring Generative Process Reward Modeling for Semi-Structured Data: A Case Study of Table Question Answering
- **分类: cs.CL**

- **简介: 该论文研究生成式过程奖励模型在表格问答任务中的应用，旨在提升复杂推理能力。工作包括评估现有模型，发现其在泛化性和步骤依赖性上的不足。**

- **链接: [https://arxiv.org/pdf/2510.20304v2](https://arxiv.org/pdf/2510.20304v2)**

> **作者:** Lei Tang; Wei Zhou; Mohsen Mesgar
>
> **备注:** Accepted at EACL 2026 Main
>
> **摘要:** Process reward models (PRMs) enhance complex reasoning in large language models (LLMs) by evaluating candidate solutions step-by-step and selecting answers based on aggregated step scores. While effective in domains such as mathematics, their applicability to tasks involving semi-structured data, like table question answering (TQA), remains unexplored. TQA poses unique challenges for PRMs, including abundant irrelevant information, loosely connected reasoning steps, and domain-specific reasoning. This work presents the first systematic study of PRMs for TQA. We evaluate state-of-the-art generative PRMs on TQA from both answer and step perspectives. Results show that PRMs that combine textual and code verification can aid solution selection but struggle to generalize to out-of-domain data. Analysis reveals a weak correlation between performance in step-level verification and answer accuracy, possibly stemming from weak step dependencies and loose causal links. Our findings highlight limitations of current PRMs on TQA and offer valuable insights for building more robust, process-aware verifiers.
>
---
#### [replaced 006] Enhancing Study-Level Inference from Clinical Trial Papers via Reinforcement Learning-Based Numeric Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医学系统综述自动化任务，旨在提升临床试验数据的数值推理能力。通过构建数值证据提取与效应估计系统，解决传统方法依赖文本线索的问题，提高结论准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2505.22928v3](https://arxiv.org/pdf/2505.22928v3)**

> **作者:** Massimiliano Pronesti; Michela Lorandi; Paul Flanagan; Oisin Redmond; Anya Belz; Yufang Hou
>
> **备注:** Accepted at EMNLP 2025 Main Conference. This revision corrects a minor typo in the camera-ready version
>
> **摘要:** Systematic reviews in medicine play a critical role in evidence-based decision-making by aggregating findings from multiple studies. A central bottleneck in automating this process is extracting numeric evidence and determining study-level conclusions for specific outcomes and comparisons. Prior work has framed this problem as a textual inference task by retrieving relevant content fragments and inferring conclusions from them. However, such approaches often rely on shallow textual cues and fail to capture the underlying numeric reasoning behind expert assessments. In this work, we conceptualise the problem as one of quantitative reasoning. Rather than inferring conclusions from surface text, we extract structured numerical evidence (e.g., event counts or standard deviations) and apply domain knowledge informed logic to derive outcome-specific conclusions. We develop a numeric reasoning system composed of a numeric data extraction model and an effect estimate component, enabling more accurate and interpretable inference aligned with the domain expert principles. We train the numeric data extraction model using different strategies, including supervised fine-tuning (SFT) and reinforcement learning (RL) with a new value reward model. When evaluated on the CochraneForest benchmark, our best-performing approach -- using RL to train a small-scale number extraction model -- yields up to a 21% absolute improvement in F1 score over retrieval-based systems and outperforms general-purpose LLMs of over 400B parameters by up to 9% on the RCTs benchmark. Our results demonstrate the promise of reasoning-driven approaches for automating systematic evidence synthesis.
>
---
#### [replaced 007] Efficient semantic uncertainty quantification in language models via diversity-steered sampling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理中的不确定性量化任务，旨在提高大语言模型在问答任务中语义不确定性的估计效率。通过引入多样性引导采样方法，减少冗余输出，提升样本效率。**

- **链接: [https://arxiv.org/pdf/2510.21310v2](https://arxiv.org/pdf/2510.21310v2)**

> **作者:** Ji Won Park; Kyunghyun Cho
>
> **备注:** 10 pages (+7 appendix), 7 figures. Accepted at NeurIPS 2025
>
> **摘要:** Accurately estimating semantic aleatoric and epistemic uncertainties in large language models (LLMs) is particularly challenging in free-form question answering (QA), where obtaining stable estimates often requires many expensive generations. We introduce a diversity-steered sampler that discourages semantically redundant outputs during decoding, covers both autoregressive and masked diffusion paradigms, and yields substantial sample-efficiency gains. The key idea is to inject a continuous semantic-similarity penalty into the model's proposal distribution using a natural language inference (NLI) model lightly finetuned on partial prefixes or intermediate diffusion states. We debias downstream uncertainty estimates with importance reweighting and shrink their variance with control variates. Across four QA benchmarks, our method matches or surpasses baselines while covering more semantic clusters with the same number of samples. Being modular and requiring no gradient access to the base LLM, the framework promises to serve as a drop-in enhancement for uncertainty estimation in risk-sensitive model deployments.
>
---
#### [replaced 008] Unified Multimodal Interleaved Document Representation for Retrieval
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决文档多模态表示与上下文丢失问题。通过统一嵌入多模态文档并融合片段表示，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2410.02729v3](https://arxiv.org/pdf/2410.02729v3)**

> **作者:** Jaewoo Lee; Joonho Ko; Jinheon Baek; Soyeong Jeong; Sung Ju Hwang
>
> **备注:** EACL Findings 2026
>
> **摘要:** Information Retrieval (IR) methods aim to identify documents relevant to a query, which have been widely applied in various natural language tasks. However, existing approaches typically consider only the textual content within documents, overlooking the fact that documents can contain multiple modalities, including images and tables. Also, they often segment each long document into multiple discrete passages for embedding, which prevents them from capturing the overall document context and interactions between paragraphs. To address these two challenges, we propose a method that holistically embeds documents interleaved with multiple modalities by leveraging the capability of recent vision-language models that enable the processing and integration of text, images, and tables into a unified format and representation. Moreover, to mitigate the information loss from segmenting documents into passages, instead of representing and retrieving passages individually, we further merge the representations of segmented passages into one single document representation, while we additionally introduce a reranking strategy to decouple and identify the relevant passage within the document if necessary. Then, through extensive experiments on diverse IR scenarios considering both the textual and multimodal queries, we show that our approach substantially outperforms relevant baselines, thanks to the consideration of the multimodal information within documents.
>
---
#### [replaced 009] I-MCTS: Enhancing Agentic AutoML via Introspective Monte Carlo Tree Search
- **分类: cs.CL**

- **简介: 该论文属于AutoML任务，旨在解决LLM生成代码多样性低和质量差的问题。提出I-MCTS方法，通过自我反思和LLM评估提升搜索效率与效果。**

- **链接: [https://arxiv.org/pdf/2502.14693v5](https://arxiv.org/pdf/2502.14693v5)**

> **作者:** Zujie Liang; Feng Wei; Wujiang Xu; Lin Chen; Yuxi Qian; Xinhui Wu
>
> **备注:** EACL 2026 Findings
>
> **摘要:** Recent advancements in large language models (LLMs) have shown remarkable potential in automating machine learning tasks. However, existing LLM-based agents often struggle with low-diversity and suboptimal code generation. While recent work has introduced Monte Carlo Tree Search (MCTS) to address these issues, limitations persist in the quality and diversity of thoughts generated, as well as in the scalar value feedback mechanisms used for node selection. In this study, we introduce Introspective Monte Carlo Tree Search (I-MCTS), a novel approach that iteratively expands tree nodes through an introspective process that meticulously analyzes solutions and results from parent and sibling nodes. This facilitates a continuous refinement of the node in the search tree, thereby enhancing the overall decision-making process. Furthermore, we integrate a Large Language Model (LLM)-based value model to facilitate direct evaluation of each node's solution prior to conducting comprehensive computational rollouts. A hybrid rewarding mechanism is implemented to seamlessly transition the Q-value from LLM-estimated scores to actual performance scores. This allows higher-quality nodes to be traversed earlier. Applied to the various ML tasks, our approach demonstrates a 4% absolute improvement in performance compared to the strong open-source AutoML agents, showcasing its effectiveness in enhancing agentic AutoML systems. Resource available at https://github.com/jokieleung/I-MCTS
>
---
#### [replaced 010] CRADLE Bench: A Clinician-Annotated Benchmark for Multi-Faceted Mental Health Crisis and Safety Risk Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于心理健康危机检测任务，旨在提升语言模型对多种危机情境的识别能力。研究构建了CRADLE BENCH基准，包含临床标注数据和自动标注训练集，以改进危机检测模型性能。**

- **链接: [https://arxiv.org/pdf/2510.23845v2](https://arxiv.org/pdf/2510.23845v2)**

> **作者:** Grace Byun; Rebecca Lipschutz; Sean T. Minton; Abigail Lott; Jinho D. Choi
>
> **摘要:** Detecting mental health crisis situations such as suicide ideation, rape, domestic violence, child abuse, and sexual harassment is a critical yet underexplored challenge for language models. When such situations arise during user--model interactions, models must reliably flag them, as failure to do so can have serious consequences. In this work, we introduce CRADLE BENCH, a benchmark for multi-faceted crisis detection. Unlike previous efforts that focus on a limited set of crisis types, our benchmark covers seven types defined in line with clinical standards and is the first to incorporate temporal labels. Our benchmark provides 600 clinician-annotated evaluation examples and 420 development examples, together with a training corpus of around 4K examples automatically labeled using a majority-vote ensemble of multiple language models, which significantly outperforms single-model annotation. We further fine-tune six crisis detection models on subsets defined by consensus and unanimous ensemble agreement, providing complementary models trained under different agreement criteria.
>
---
#### [replaced 011] A Two-Stage GPU Kernel Tuner Combining Semantic Refactoring and Search-Based Optimization
- **分类: cs.CL**

- **简介: 该论文属于GPU代码优化任务，解决手动调优效率低的问题。通过模板化重构与搜索优化结合，提升性能稳定性与可解释性。**

- **链接: [https://arxiv.org/pdf/2601.12698v3](https://arxiv.org/pdf/2601.12698v3)**

> **作者:** Qiuyi Qu; Yicheng Sui; Yufei Sun; Rui Chen; Xiaofei Zhang; Yuzhi Zhang; Haofeng Wang; Ge Lan
>
> **摘要:** GPU code optimization is a key performance bottleneck for HPC workloads as well as large-model training and inference. Although compiler optimizations and hand-written kernels can partially alleviate this issue, achieving near-hardware-limit performance still relies heavily on manual code refactoring and parameter tuning. Recent progress in LLM-agent-based kernel generation and optimization has been reported, yet many approaches primarily focus on direct code rewriting, where parameter choices are often implicit and hard to control, or require human intervention, leading to unstable performance gains. This paper introduces a template-based rewriting layer on top of an agent-driven iterative loop: kernels are semantically refactored into explicitly parameterizable templates, and template parameters are then optimized via search-based autotuning, yielding more stable and higher-quality speedups. Experiments on a set of real-world kernels demonstrate speedups exceeding 3x in the best case. We extract representative CUDA kernels from SGLang as evaluation targets; the proposed agentic tuner iteratively performs templating, testing, analysis, and planning, and leverages profiling feedback to execute constrained parameter search under hardware resource limits. Compared to agent-only direct rewriting, the template-plus-search design significantly reduces the randomness of iterative optimization, making the process more interpretable and enabling a more systematic approach toward high-performance configurations. The proposed method can be further extended to OpenCL, HIP, and other backends to deliver automated performance optimization for real production workloads.
>
---
#### [replaced 012] AfriEconQA: A Benchmark Dataset for African Economic Analysis based on World Bank Reports
- **分类: cs.CL**

- **简介: 该论文提出AfriEconQA，一个用于非洲经济分析的基准数据集，解决信息检索与问答任务中的精准数值推理问题。**

- **链接: [https://arxiv.org/pdf/2601.15297v2](https://arxiv.org/pdf/2601.15297v2)**

> **作者:** Edward Ajayi
>
> **摘要:** We introduce AfriEconQA, a specialized benchmark dataset for African economic analysis grounded in a comprehensive corpus of 236 World Bank reports. The task of AfriEconQA is to answer complex economic queries that require high-precision numerical reasoning and temporal disambiguation from specialized institutional documents. The dataset consists of 8,937 curated QA instances, rigorously filtered from a pool of 10018 synthetic questions to ensure high-quality evidence-answer alignment. Each instance is composed of: (1) a question requiring reasoning over economic indicators, (2) the corresponding evidence retrieved from the corpus, (3) a verified ground-truth answer, and (4) source metadata (e.g., URL and publication date) to ensure temporal provenance. AfriEconQA is the first benchmark focused specifically on African economic analysis, providing a unique challenge for Information Retrieval (IR) systems, as the data is largely absent from the pretraining corpora of current Large Language Models (LLMs). We operationalize this dataset through an 11-experiment matrix, benchmarking a zero-shot baseline (GPT-5 Mini) against RAG configurations using GPT-4o and Qwen 32B across five distinct embedding and ranking strategies. Our results demonstrate a severe parametric knowledge gap, where zero-shot models fail to answer over 90 percent of queries, and even state-of-the-art RAG pipelines struggle to achieve high precision. This confirms AfriEconQA as a robust and challenging benchmark for the next generation of domain-specific IR and RAG systems. The AfriEconQA dataset and code will be made publicly available upon publication.
>
---
#### [replaced 013] Stable-DiffCoder: Pushing the Frontier of Code Diffusion Large Language Model
- **分类: cs.CL**

- **简介: 该论文属于代码生成任务，旨在提升代码扩散模型的性能。针对现有代码扩散模型效果不足的问题，提出Stable-DiffCoder，通过优化训练方法实现更优的代码建模。**

- **链接: [https://arxiv.org/pdf/2601.15892v2](https://arxiv.org/pdf/2601.15892v2)**

> **作者:** Chenghao Fan; Wen Heng; Bo Li; Sichen Liu; Yuxuan Song; Jing Su; Xiaoye Qu; Kai Shen; Wei Wei
>
> **摘要:** Diffusion-based language models (DLLMs) offer non-sequential, block-wise generation and richer data reuse compared to autoregressive (AR) models, but existing code DLLMs still lag behind strong AR baselines under comparable budgets. We revisit this setting in a controlled study and introduce Stable-DiffCoder, a block diffusion code model that reuses the Seed-Coder architecture, data, and training pipeline. To enable efficient knowledge learning and stable training, we incorporate a block diffusion continual pretraining (CPT) stage enhanced by a tailored warmup and block-wise clipped noise schedule. Under the same data and architecture, Stable-DiffCoder overall outperforms its AR counterpart on a broad suite of code benchmarks. Moreover, relying only on the CPT and supervised fine-tuning stages, Stable-DiffCoder achieves stronger performance than a wide range of \~8B ARs and DLLMs, demonstrating that diffusion-based training can improve code modeling quality beyond AR training alone. Moreover, diffusion-based any-order modeling improves structured code modeling for editing and reasoning, and through data augmentation, benefits low-resource coding languages.
>
---
#### [replaced 014] CARE: Cognitive-reasoning Augmented Reinforcement for Emotional Support Conversation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感支持对话任务，旨在提升对话的逻辑性和支持性。针对现有方法忽视认知推理的问题，提出CARE框架，结合原始数据和强化学习增强模型的推理能力。**

- **链接: [https://arxiv.org/pdf/2510.05122v2](https://arxiv.org/pdf/2510.05122v2)**

> **作者:** Jie Zhu; Yuanchen Zhou; Shuo Jiang; Junhui Li; Lifan Guo; Feng Chen; Chi Zhang; Fang Kong
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Emotional Support Conversation (ESC) plays a vital role in alleviating psychological stress and providing emotional value through dialogue. While recent studies have largely focused on data augmentation and synthetic corpus construction, they often overlook the deeper cognitive reasoning processes that underpin effective emotional support. To address this gap, we propose \textbf{CARE}, a novel framework that strengthens reasoning in ESC without relying on large-scale synthetic data. CARE leverages the original ESC training set to guide models in generating logically coherent and supportive responses, thereby explicitly enhancing cognitive reasoning. Building on this foundation, we further employ reinforcement learning to refine and reinforce the reasoning process. Experimental results demonstrate that CARE significantly improves both the logical soundness and supportive quality of responses, advancing the development of empathetic, cognitively robust, and human-like emotional support systems.
>
---
#### [replaced 015] Improving Training Efficiency and Reducing Maintenance Costs via Language Specific Model Merging
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决多语言模型训练效率低和维护成本高的问题。通过模型合并提升效率，减少重新训练需求。**

- **链接: [https://arxiv.org/pdf/2601.16127v2](https://arxiv.org/pdf/2601.16127v2)**

> **作者:** Alphaeus Dmonte; Vidhi Gupta; Daniel J Perry; Mark Arehart
>
> **备注:** Accepted to EACL 2026 Industry Track
>
> **摘要:** Fine-tuning a task-specific multilingual large language model (LLM) involves training the model on a multilingual dataset with examples in all the required languages. Updating one or more supported languages with additional data or adding support for a new language involves retraining the model, which can be computationally inefficient and creates a severe maintenance bottleneck. Recent research on merging multilingual multitask models has shown promise in terms of improved quality, but its computational and maintenance efficiency remains unstudied. In this work, we provide the first focused analysis of this merging strategy from an efficiency perspective, evaluating it across three independent tasks. We demonstrate significant efficiency gains while maintaining parity in terms of quality: this merging approach reduces the initial training time by up to 50\%. We also demonstrate that updating an individual language and re-merging as part of model maintenance reduces training costs by more than 60\%, compared to re-training the full multilingual model. We show this on both public and proprietary industry datasets confirming that the approach works well for industrial use cases in addition to academic settings already studied in previous work.
>
---
#### [replaced 016] Revisiting Direct Speech-to-Text Translation with Speech LLMs: Better Scaling than CoT Prompting?
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音到文本翻译任务，探讨CoT与直接提示方法的效果差异。通过增加数据量比较两种策略，发现直接提示在数据增多时表现更稳定，可能更有效。**

- **链接: [https://arxiv.org/pdf/2510.03093v2](https://arxiv.org/pdf/2510.03093v2)**

> **作者:** Oriol Pareras; Gerard I. Gállego; Federico Costa; Cristina España-Bonet; Javier Hernando
>
> **备注:** To appear in Proc. ICASSP 2026, May 04-08, 2026, Barcelona, Spain
>
> **摘要:** Recent work on Speech-to-Text Translation (S2TT) has focused on LLM-based models, introducing the increasingly adopted Chain-of-Thought (CoT) prompting, where the model is guided to first transcribe the speech and then translate it. CoT typically outperforms direct prompting primarily because it can exploit abundant Automatic Speech Recognition (ASR) and Text-to-Text Translation (T2TT) datasets to explicitly model its steps. In this paper, we systematically compare CoT and Direct prompting under increasing amounts of S2TT data. To this end, we pseudo-label an ASR corpus by translating its transcriptions into six European languages, and train LLM-based S2TT systems with both prompting strategies at different data scales. Our results show that Direct improves more consistently as the amount of data increases, suggesting that it may become a more effective approach as larger S2TT resources are created.
>
---
#### [replaced 017] The Bitter Lesson of Diffusion Language Models for Agentic Workflows: A Comprehensive Reality Check
- **分类: cs.CL**

- **简介: 该论文属于人工智能领域，研究dLLMs在代理工作流中的应用。旨在解决dLLMs是否能有效支持代理行为的问题，通过实验发现其存在缺陷，并提出DiffuAgent框架进行评估。**

- **链接: [https://arxiv.org/pdf/2601.12979v2](https://arxiv.org/pdf/2601.12979v2)**

> **作者:** Qingyu Lu; Liang Ding; Kanjian Zhang; Jinxia Zhang; Dacheng Tao
>
> **备注:** Under Review
>
> **摘要:** The pursuit of real-time agentic interaction has driven interest in Diffusion-based Large Language Models (dLLMs) as alternatives to auto-regressive backbones, promising to break the sequential latency bottleneck. However, does such efficiency gains translate into effective agentic behavior? In this work, we present a comprehensive evaluation of dLLMs (e.g., LLaDA, Dream) across two distinct agentic paradigms: Embodied Agents (requiring long-horizon planning) and Tool-Calling Agents (requiring precise formatting). Contrary to the efficiency hype, our results on Agentboard and BFCL reveal a "bitter lesson": current dLLMs fail to serve as reliable agentic backbones, frequently leading to systematically failure. (1) In Embodied settings, dLLMs suffer repeated attempts, failing to branch under temporal feedback. (2) In Tool-Calling settings, dLLMs fail to maintain symbolic precision (e.g. strict JSON schemas) under diffusion noise. To assess the potential of dLLMs in agentic workflows, we introduce DiffuAgent, a multi-agent evaluation framework that integrates dLLMs as plug-and-play cognitive cores. Our analysis shows that dLLMs are effective in non-causal roles (e.g., memory summarization and tool selection) but require the incorporation of causal, precise, and logically grounded reasoning mechanisms into the denoising process to be viable for agentic tasks.
>
---
#### [replaced 018] Who Does This Name Remind You of ? Nationality Prediction via Large Language Model Associative Memory
- **分类: cs.CL**

- **简介: 该论文属于国籍预测任务，旨在解决通过名字推断国籍的问题。提出LAMA框架，利用大语言模型的联想记忆，通过召回名人并聚合国籍信息进行预测。**

- **链接: [https://arxiv.org/pdf/2601.12771v2](https://arxiv.org/pdf/2601.12771v2)**

> **作者:** Keito Inoshita
>
> **摘要:** Large language models (LLMs) possess extensive world knowledge, yet methods for effectively eliciting this knowledge remain underexplored. Nationality and region prediction tasks require understanding of not only linguistic features but also cultural and historical background, making LLM world knowledge particularly valuable. However, conventional LLM prompting methods rely on direct reasoning approaches, which have limitations in applying abstract linguistic rules. We propose LLM Associative Memory Agents (LAMA), a novel framework that leverages LLM world knowledge as associative memory. Rather than directly inferring nationality from names, LAMA recalls famous individuals with the same name and aggregates their nationalities through indirect reasoning. A dual-agent architecture comprising a Person Agent and a Media Agent, specialized in different knowledge domains, recalls famous individuals in parallel, generating Top-1 predictions through voting and Top-K predictions through conditional completion. On a 99-country nationality prediction task, LAMA achieved 0.817 accuracy, substantially outperforming conventional LLM prompting methods and neural models. Our experiments reveal that LLMs exhibit higher reliability in recalling concrete examples than in abstract reasoning, that recall-based approaches are robust to low-frequency nationalities independent of data frequency distributions, and that the dual-agent architecture functions complementarily to produce synergistic effects. These results demonstrate the effectiveness of a new multi-agent system that retrieves and aggregates LLM knowledge rather than prompting reasoning.
>
---
#### [replaced 019] The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining
- **分类: cs.CL**

- **简介: 该论文研究多语言预训练中混合语言文档的作用，旨在解决翻译与跨语言理解任务的差异。通过对比实验发现，平行语料对翻译性能至关重要，而代码切换数据影响较小。**

- **链接: [https://arxiv.org/pdf/2601.00364v2](https://arxiv.org/pdf/2601.00364v2)**

> **作者:** Jiandong Shao; Raphael Tang; Crystina Zhang; Karin Sevegnani; Pontus Stenetorp; Jianfei Yang; Yao Lu
>
> **备注:** under review
>
> **摘要:** Multilingual large language models achieve impressive cross-lingual performance despite largely monolingual pretraining. While bilingual data in pretraining corpora is widely believed to enable these abilities, details of its contributions remain unclear. We investigate this question by pretraining models from scratch under controlled conditions, comparing the standard web corpus with a monolingual-only version that removes all multilingual documents. Despite constituting only 2% of the corpus, removing bilingual data causes translation performance to drop 56% in BLEU, while behaviour on cross-lingual QA and general reasoning tasks remains stable, with training curves largely overlapping the baseline. To understand this asymmetry, we categorize bilingual data into parallel (14%), code-switching (72%), and miscellaneous documents (14%) based on the semantic relevance of content in different languages. We then conduct granular ablations by reintroducing parallel or code-switching data into the monolingual-only corpus. Our experiments reveal that parallel data almost fully restores translation performance (91% of the unfiltered baseline), whereas code-switching contributes minimally. Other cross-lingual tasks remain largely unaffected by either type. These findings reveal that translation critically depends on systematic token-level alignments from parallel data, whereas cross-lingual understanding and reasoning appear to be achievable even without bilingual data.
>
---
#### [replaced 020] Speech-Aware Long Context Pruning and Integration for Contextualized Automatic Speech Recognition
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，解决长上下文信息利用不足的问题。提出SAP²方法，通过动态剪枝和整合关键词提升识别性能。**

- **链接: [https://arxiv.org/pdf/2511.11139v2](https://arxiv.org/pdf/2511.11139v2)**

> **作者:** Yiming Rong; Yixin Zhang; Ziyi Wang; Deyang Jiang; Yunlong Zhao; Haoran Wu; Shiyu Zhou; Bo Xu
>
> **摘要:** Automatic speech recognition (ASR) systems have achieved remarkable performance in common conditions but often struggle to leverage long-context information in contextualized scenarios that require domain-specific knowledge, such as conference presentations. This challenge arises primarily due to constrained model context windows and the sparsity of relevant information within extensive contextual noise. To solve this, we propose the SAP$^{2}$ method, a novel framework that dynamically prunes and integrates relevant contextual keywords in two stages. Specifically, each stage leverages our proposed Speech-Driven Attention-based Pooling mechanism, enabling efficient compression of context embeddings while preserving speech-salient information. Experimental results demonstrate state-of-the-art performance of SAP$^{2}$ on the SlideSpeech and LibriSpeech datasets, achieving word error rates (WER) of 7.71% and 1.12%, respectively. On SlideSpeech, our method notably reduces biased keyword error rates (B-WER) by 41.1% compared to non-contextual baselines. SAP$^{2}$ also exhibits robust scalability, consistently maintaining performance under extensive contextual input conditions on both datasets.
>
---
#### [replaced 021] Theoretical Foundations of Scaling Law in Familial Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于机器学习领域，解决传统神经网络缩放定律无法适应多子模型部署的问题。通过引入粒度变量，建立统一的缩放定律模型，实现高效训练与灵活部署。**

- **链接: [https://arxiv.org/pdf/2512.23407v2](https://arxiv.org/pdf/2512.23407v2)**

> **作者:** Huan Song; Qingfei Zhao; Ting Long; Shuyu Tian; Hongjun An; Jiawei Shao; Xuelong Li
>
> **摘要:** Neural scaling laws have become foundational for optimizing large language model (LLM) training, yet they typically assume a single dense model output. This limitation effectively overlooks "Familial models, a transformative paradigm essential for realizing ubiquitous intelligence across heterogeneous device-edge-cloud hierarchies. Transcending static architectures, familial models integrate early exits with relay-style inference to spawn G deployable sub-models from a single shared backbone. In this work, we theoretically and empirically extend the scaling law to capture this "one-run, many-models" paradigm by introducing Granularity (G) as a fundamental scaling variable alongside model size (N) and training tokens (D). To rigorously quantify this relationship, we propose a unified functional form L(N, D, G) and parameterize it using large-scale empirical runs. Specifically, we employ a rigorous IsoFLOP experimental design to strictly isolate architectural impact from computational scale. Across fixed budgets, we systematically sweep model sizes (N) and granularities (G) while dynamically adjusting tokens (D). This approach effectively decouples the marginal cost of granularity from the benefits of scale, ensuring high-fidelity parameterization of our unified scaling law. Our results reveal that the granularity penalty follows a multiplicative power law with an extremely small exponent. Theoretically, this bridges fixed-compute training with dynamic architectures. Practically, it validates the "train once, deploy many" paradigm, demonstrating that deployment flexibility is achievable without compromising the compute-optimality of dense baselines.
>
---
#### [replaced 022] A Machine Learning Approach for Detection of Mental Health Conditions and Cyberbullying from Social Media
- **分类: cs.CL; cs.SI**

- **简介: 该论文属于多分类任务，旨在检测社交媒体中的心理健康问题和网络欺凌。通过构建平衡数据集并使用Transformer模型，提升检测性能，并引入可解释框架辅助人工审核。**

- **链接: [https://arxiv.org/pdf/2511.20001v3](https://arxiv.org/pdf/2511.20001v3)**

> **作者:** Edward Ajayi; Martha Kachweka; Mawuli Deku; Emily Aiken
>
> **备注:** Oral Presentation at the AAAI-26 Bridge Program on AI for Medicine and Healthcare. To appear in Proceedings of Machine Learning Research (PMLR)
>
> **摘要:** Mental health challenges and cyberbullying are increasingly prevalent in digital spaces, necessitating scalable and interpretable detection systems. This paper introduces a unified multiclass classification framework for detecting ten distinct mental health and cyberbullying categories from social media data. We curate datasets from Twitter and Reddit, implementing a rigorous "split-then-balance" pipeline to train on balanced data while evaluating on a realistic, held-out imbalanced test set. We conducted a comprehensive evaluation comparing traditional lexical models, hybrid approaches, and several end-to-end fine-tuned transformers. Our results demonstrate that end-to-end fine-tuning is critical for performance, with the domain-adapted MentalBERT emerging as the top model, achieving an accuracy of 0.92 and a Macro F1 score of 0.76, surpassing both its generic counterpart and a zero-shot LLM baseline. Grounded in a comprehensive ethical analysis, we frame the system as a human-in-the-loop screening aid, not a diagnostic tool. To support this, we introduce a hybrid SHAPLLM explainability framework and present a prototype dashboard ("Social Media Screener") designed to integrate model predictions and their explanations into a practical workflow for moderators. Our work provides a robust baseline, highlighting future needs for multi-label, clinically-validated datasets at the critical intersection of online safety and computational mental health.
>
---
#### [replaced 023] Simple-Sampling and Hard-Mixup with Prototypes to Rebalance Contrastive Learning for Text Classification
- **分类: cs.CL**

- **简介: 该论文属于文本分类任务，针对数据不平衡问题，提出SharpReCL模型。通过原型向量构建平衡样本集，提升对比学习效果。**

- **链接: [https://arxiv.org/pdf/2405.11524v2](https://arxiv.org/pdf/2405.11524v2)**

> **作者:** Mengyu Li; Yonghao Liu; Fausto Giunchiglia; Ximing Li; Xiaoyue Feng; Renchu Guan
>
> **备注:** WWW26
>
> **摘要:** Text classification is a crucial and fundamental task in web content mining. Compared with the previous learning paradigm of pre-training and fine-tuning by cross entropy loss, the recently proposed supervised contrastive learning approach has received tremendous attention due to its powerful feature learning capability and robustness. Although several studies have incorporated this technique for text classification, some limitations remain. First, many text datasets are imbalanced, and the learning mechanism of supervised contrastive learning is sensitive to data imbalance, which may harm the model's performance. Moreover, these models leverage separate classification branches with cross entropy and supervised contrastive learning branches without explicit mutual guidance. To this end, we propose a novel model named SharpReCL for imbalanced text classification tasks. First, we obtain the prototype vector of each class in the balanced classification branch to act as a representation of each class. Then, by further explicitly leveraging the prototype vectors, we construct a proper and sufficient target sample set with the same size for each class to perform the supervised contrastive learning procedure. The empirical results show the effectiveness of our model, which even outperforms popular large language models across several datasets. Our code is available here.
>
---
#### [replaced 024] CALE : Concept-Aligned Embeddings for Both Within-Lemma and Inter-Lemma Sense Differentiation
- **分类: cs.CL**

- **简介: 该论文属于词汇语义任务，旨在解决同词素和跨词义的区分问题。通过构建数据集并微调模型，提出Concept-Aligned Embeddings（CALE），提升语义表示效果。**

- **链接: [https://arxiv.org/pdf/2508.04494v2](https://arxiv.org/pdf/2508.04494v2)**

> **作者:** Bastien Liétard; Gabriel Loiseau
>
> **备注:** Accepted at EACL 2026
>
> **摘要:** Lexical semantics is concerned with both the multiple senses a word can adopt in different contexts, and the semantic relations that exist between meanings of different words. To investigate them, Contextualized Language Models are a valuable tool that provides context-sensitive representations that can be used to investigate lexical meaning. Recent works like XL-LEXEME have leveraged the task of Word-in-Context to fine-tune them to get more semantically accurate representations, but Word-in-Context only compares occurrences of the same lemma, limiting the range of captured information. In this paper, we propose an extension, Concept Differentiation, to include inter-words scenarios. We provide a dataset for this task, derived from SemCor data. Then we fine-tune several representation models on this dataset. We call these models Concept-Aligned Embeddings (CALE). By challenging our models and other models on various lexical semantic tasks, we demonstrate that the proposed models provide efficient multi-purpose representations of lexical meaning that reach best performances in our experiments. We also show that CALE's fine-tuning brings valuable changes to the spatial organization of embeddings.
>
---
#### [replaced 025] WildScore: Benchmarking MLLMs in-the-Wild Symbolic Music Reasoning
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出WildScore，一个用于评估多模态大语言模型在真实音乐符号推理能力的基准。任务是解决MLLMs在音乐分析中的符号推理问题，通过构建真实音乐数据集和多选题形式进行评估。**

- **链接: [https://arxiv.org/pdf/2509.04744v2](https://arxiv.org/pdf/2509.04744v2)**

> **作者:** Gagan Mundada; Yash Vishe; Amit Namburi; Xin Xu; Zachary Novack; Julian McAuley; Junda Wu
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities across various vision-language tasks. However, their reasoning abilities in the multimodal symbolic music domain remain largely unexplored. We introduce WildScore, the first in-the-wild multimodal symbolic music reasoning and analysis benchmark, designed to evaluate MLLMs' capacity to interpret real-world music scores and answer complex musicological queries. Each instance in WildScore is sourced from genuine musical compositions and accompanied by authentic user-generated questions and discussions, capturing the intricacies of practical music analysis. To facilitate systematic evaluation, we propose a systematic taxonomy, comprising both high-level and fine-grained musicological ontologies. Furthermore, we frame complex music reasoning as multiple-choice question answering, enabling controlled and scalable assessment of MLLMs' symbolic music understanding. Empirical benchmarking of state-of-the-art MLLMs on WildScore reveals intriguing patterns in their visual-symbolic reasoning, uncovering both promising directions and persistent challenges for MLLMs in symbolic music reasoning and analysis. We release the dataset and code.
>
---
#### [replaced 026] Intention Collapse: Intention-Level Metrics for Reasoning in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型内部意图与外部输出间的映射问题，提出三种度量指标评估推理过程。属于自然语言处理任务，旨在分析模型内部状态与输出准确性之间的关系。**

- **链接: [https://arxiv.org/pdf/2601.01011v2](https://arxiv.org/pdf/2601.01011v2)**

> **作者:** Patricio Vera
>
> **备注:** 41 pages, 8 figures, 6 tables. Code: https://github.com/patriciomvera/intention-collapse-experiments
>
> **摘要:** Language generation maps a rich, high-dimensional internal state to a single token sequence. We study this many-to-one mapping through the lens of intention collapse: the projection from an internal intention space I to an external language space L. We introduce three cheap, model-agnostic metrics computed on a pre-collapse state I: (i) intention entropy Hint(I), (ii) effective dimensionality deff(I), and (iii) recoverability Recov(I), operationalized as probe AUROC for predicting eventual success. We evaluate these metrics in a 3x3 study across models (Mistral-7B, LLaMA-3.1-8B, Qwen-2.5-7B) and benchmarks (GSM8K, ARC-Challenge, AQUA-RAT), comparing baseline, chain-of-thought (CoT), and a babble control (n=200 items per cell). CoT increases average accuracy from 34.2% to 47.3% (+13.1 pp), driven by large gains on GSM8K but consistent degradations on ARC-Challenge. Across models, CoT induces distinct entropy regimes relative to baseline, dH = Hint(CoT) - Hint(Base): Mistral shows dH < 0 (lower-entropy CoT), whereas LLaMA shows dH > 0 (higher-entropy CoT), highlighting heterogeneity in CoT-induced internal uncertainty. Finally, probe AUROC is significantly above chance in a subset of settings and can dissociate from behavioral accuracy (e.g., high AUROC alongside lower CoT accuracy on ARC-Challenge for Qwen), suggesting that informative internal signal is not always reliably converted into a final discrete decision under constrained response formats.
>
---
#### [replaced 027] Exploring LLMs for Scientific Information Extraction Using The SciEx Framework
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于科学信息抽取任务，旨在解决科学文献中细粒度信息提取的难题。提出SciEx框架，实现模块化、可扩展的信息抽取系统。**

- **链接: [https://arxiv.org/pdf/2512.10004v2](https://arxiv.org/pdf/2512.10004v2)**

> **作者:** Sha Li; Ayush Sadekar; Nathan Self; Yiqi Su; Lars Andersland; Mira Chaplin; Annabel Zhang; Hyoju Yang; James B Henderson; Krista Wigginton; Linsey Marr; T. M. Murali; Naren Ramakrishnan
>
> **备注:** Accepted to the KGML Bridge at AAAI 2026 (non-archival)
>
> **摘要:** Large language models (LLMs) are increasingly touted as powerful tools for automating scientific information extraction. However, existing methods and tools often struggle with the realities of scientific literature: long-context documents, multi-modal content, and reconciling varied and inconsistent fine-grained information across multiple publications into standardized formats. These challenges are further compounded when the desired data schema or extraction ontology changes rapidly, making it difficult to re-architect or fine-tune existing systems. We present SciEx, a modular and composable framework that decouples key components including PDF parsing, multi-modal retrieval, extraction, and aggregation. This design streamlines on-demand data extraction while enabling extensibility and flexible integration of new models, prompting strategies, and reasoning mechanisms. We evaluate SciEx on datasets spanning three scientific topics for its ability to extract fine-grained information accurately and consistently. Our findings provide practical insights into both the strengths and limitations of current LLM-based pipelines.
>
---
#### [replaced 028] PRACTIQ: A Practical Conversational Text-to-SQL dataset with Ambiguous and Unanswerable Queries
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PRACTIQ数据集，用于对话式文本到SQL任务，解决模糊和无法回答的问题。通过构建包含四类模糊和四类无法回答问题的对话数据，评估模型处理此类问题的能力。**

- **链接: [https://arxiv.org/pdf/2410.11076v2](https://arxiv.org/pdf/2410.11076v2)**

> **作者:** Mingwen Dong; Nischal Ashok Kumar; Yiqun Hu; Anuj Chauhan; Chung-Wei Hang; Shuaichen Chang; Lin Pan; Wuwei Lan; Henghui Zhu; Jiarong Jiang; Patrick Ng; Zhiguo Wang
>
> **摘要:** Previous text-to-SQL datasets and systems have primarily focused on user questions with clear intentions that can be answered. However, real user questions can often be ambiguous with multiple interpretations or unanswerable due to a lack of relevant data. In this work, we construct a practical conversational text-to-SQL dataset called PRACTIQ, consisting of ambiguous and unanswerable questions inspired by real-world user questions. We first identified four categories of ambiguous questions and four categories of unanswerable questions by studying existing text-to-SQL datasets. Then, we generate conversations with four turns: the initial user question, an assistant response seeking clarification, the user's clarification, and the assistant's clarified SQL response with the natural language explanation of the execution results. For some ambiguous queries, we also directly generate helpful SQL responses, that consider multiple aspects of ambiguity, instead of requesting user clarification. To benchmark the performance on ambiguous, unanswerable, and answerable questions, we implemented large language model (LLM)-based baselines using various LLMs. Our approach involves two steps: question category classification and clarification SQL prediction. Our experiments reveal that state-of-the-art systems struggle to handle ambiguous and unanswerable questions effectively. We will release our code for data generation and experiments on GitHub.
>
---
#### [replaced 029] Comparing Specialised Small and General Large Language Models on Text Classification: 100 Labelled Samples to Achieve Break-Even Performance
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文研究文本分类任务，比较专用小模型与通用大模型在少量标注数据下的表现。工作包括分析不同微调方法，确定模型性能平衡点，发现约100个样本即可使小模型超越大模型。**

- **链接: [https://arxiv.org/pdf/2402.12819v4](https://arxiv.org/pdf/2402.12819v4)**

> **作者:** Branislav Pecher; Ivan Srba; Maria Bielikova
>
> **备注:** Accepted to the EMNLP 2025 conference
>
> **摘要:** When solving NLP tasks with limited labelled data, researchers typically either use a general large language model without further update, or use a small number of labelled samples to tune a specialised smaller model. In this work, we answer an important question -- how many labelled samples are required for the specialised small models to outperform general large models, while taking the performance variance into consideration. By observing the behaviour of fine-tuning, instruction-tuning, prompting and in-context learning on 8 language models, we identify such performance break-even points across 8 representative text classification tasks of varying characteristics. We show that the specialised models often need only few samples (on average $100$) to be on par or better than the general ones. At the same time, the number of required labels strongly depends on the dataset or task characteristics, with fine-tuning on binary datasets requiring significantly more samples. When performance variance is taken into consideration, the number of required labels increases on average by $100 - 200\%$. Finally, larger models do not consistently lead to better performance and lower variance, with 4-bit quantisation having negligible impact.
>
---
#### [replaced 030] Massively Multilingual Joint Segmentation and Glossing
- **分类: cs.CL**

- **简介: 该论文属于语言文档任务，解决自动分词与释义不匹配的问题。通过联合预测实现更准确的分词和释义，提出PolyGloss模型提升性能并可快速适应新数据。**

- **链接: [https://arxiv.org/pdf/2601.10925v2](https://arxiv.org/pdf/2601.10925v2)**

> **作者:** Michael Ginn; Lindia Tjuatja; Enora Rice; Ali Marashian; Maria Valentini; Jasmine Xu; Graham Neubig; Alexis Palmer
>
> **备注:** 13 pages, 8 figures, submitted to ARR Jan 2026
>
> **摘要:** Automated interlinear gloss prediction with neural networks is a promising approach to accelerate language documentation efforts. However, while state-of-the-art models like GlossLM achieve high scores on glossing benchmarks, user studies with linguists have found critical barriers to the usefulness of such models in real-world scenarios. In particular, existing models typically generate morpheme-level glosses but assign them to whole words without predicting the actual morpheme boundaries, making the predictions less interpretable and thus untrustworthy to human annotators. We conduct the first study on neural models that jointly predict interlinear glosses and the corresponding morphological segmentation from raw text. We run experiments to determine the optimal way to train models that balance segmentation and glossing accuracy, as well as the alignment between the two tasks. We extend the training corpus of GlossLM and pretrain PolyGloss, a family of seq2seq multilingual models for joint segmentation and glossing that outperforms GlossLM on glossing and beats various open-source LLMs on segmentation, glossing, and alignment. In addition, we demonstrate that PolyGloss can be quickly adapted to a new dataset via low-rank adaptation.
>
---
#### [replaced 031] LLMs Got Rhythm? Hybrid Phonological Filtering for Greek Poetry Rhyme Detection and Generation
- **分类: cs.CL**

- **简介: 论文研究希腊诗歌押韵检测与生成任务，针对LLMs在语音现象上的不足，提出混合系统结合LLM与音系算法，提升押韵准确性。**

- **链接: [https://arxiv.org/pdf/2601.09631v4](https://arxiv.org/pdf/2601.09631v4)**

> **作者:** Stergios Chatzikyriakidis; Anastasia Natsina
>
> **摘要:** Large Language Models (LLMs), despite their remarkable capabilities across NLP tasks, struggle with phonologically-grounded phenomena like rhyme detection and generation. This is even more evident in lower-resource languages such as Modern Greek. In this paper, we present a hybrid system that combines LLMs with deterministic phonological algorithms to achieve accurate rhyme identification/analysis and generation. Our approach implements a comprehensive taxonomy of Greek rhyme types, including Pure, Rich, Imperfect, Mosaic, and Identical Pre-rhyme Vowel (IDV) patterns, and employs an agentic generation pipeline with phonological verification. We evaluate multiple prompting strategies (zero-shot, few-shot, Chain-of-Thought, and RAG-augmented) across several LLMs including Claude 3.7 and 4.5, GPT-4o, Gemini 2.0 and open-weight models like Llama 3.1 8B and 70B and Mistral Large. Results reveal a significant "Reasoning Gap": while native-like models (Claude 3.7) perform intuitively (40\% accuracy in identification), reasoning-heavy models (Claude 4.5) achieve state-of-the-art performance (54\%) only when prompted with Chain-of-Thought. Most critically, pure LLM generation fails catastrophically (under 4\% valid poems), while our hybrid verification loop restores performance to 73.1\%. We release our system and a corpus of 40,000+ rhymes, derived from the Anemoskala and Interwar Poetry corpora, to support future research.
>
---
#### [replaced 032] LLM Jailbreak Detection for (Almost) Free!
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于安全检测任务，旨在解决LLM遭受jailbreak攻击的问题。通过分析输出分布差异，提出FJD方法，实现高效检测。**

- **链接: [https://arxiv.org/pdf/2509.14558v2](https://arxiv.org/pdf/2509.14558v2)**

> **作者:** Guorui Chen; Yifan Xia; Xiaojun Jia; Zhijiang Li; Philip Torr; Jindong Gu
>
> **备注:** EMNLP 2025 (Findings) https://aclanthology.org/2025.findings-emnlp.309/
>
> **摘要:** Large language models (LLMs) enhance security through alignment when widely used, but remain susceptible to jailbreak attacks capable of producing inappropriate content. Jailbreak detection methods show promise in mitigating jailbreak attacks through the assistance of other models or multiple model inferences. However, existing methods entail significant computational costs. In this paper, we first present a finding that the difference in output distributions between jailbreak and benign prompts can be employed for detecting jailbreak prompts. Based on this finding, we propose a Free Jailbreak Detection (FJD) which prepends an affirmative instruction to the input and scales the logits by temperature to further distinguish between jailbreak and benign prompts through the confidence of the first token. Furthermore, we enhance the detection performance of FJD through the integration of virtual instruction learning. Extensive experiments on aligned LLMs show that our FJD can effectively detect jailbreak prompts with almost no additional computational costs during LLM inference.
>
---
#### [replaced 033] R$^2$PO: Decoupling Training Trajectories from Inference Responses for LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型推理任务，解决训练轨迹与推理响应冲突问题。提出R²PO方法，通过解耦提升推理能力与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.11960v2](https://arxiv.org/pdf/2601.11960v2)**

> **作者:** Jingchu Wang; Bingbing Xu; Yige Yuan; Bin Xie; Xiaoqian Sun; Huawei Shen
>
> **摘要:** Reinforcement learning has become a central paradigm for improving LLM reasoning. However, existing methods use a single policy to produce both inference responses and training optimization trajectories. The objective conflict between generating stable inference responses and diverse training trajectories leads to insufficient exploration, which harms reasoning capability. In this paper, to address the problem, we propose R$^2$PO (Residual Rollout Policy Optimization), which introduces a lightweight Residual Rollout-Head atop the policy to decouple training trajectories from inference responses, enabling controlled trajectory diversification during training while keeping inference generation stable. Experiments across multiple benchmarks show that our method consistently outperforms baselines, achieving average accuracy gains of 3.4% on MATH-500 and 1.3% on APPS, while also reducing formatting errors and mitigating length bias for stable optimization. Our code is publicly available at https://github.com/RRPO-ARR/Code.
>
---
#### [replaced 034] Linguistic traces of stochastic empathy in language models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本生成与人类识别任务，旨在解决AI与人类内容区分难题。通过多项实验，研究语言模型在模仿人类情感和表达方式上的表现。**

- **链接: [https://arxiv.org/pdf/2410.01675v2](https://arxiv.org/pdf/2410.01675v2)**

> **作者:** Bennett Kleinberg; Jari Zegers; Jonas Festor; Stefana Vida; Julian Präsent; Riccardo Loconte; Sanne Peereboom
>
> **备注:** preprint (updated)
>
> **摘要:** Differentiating generated and human-written content is increasingly difficult. We examine how an incentive to convey humanness and task characteristics shape this human vs AI race across five studies. In Study 1-2 (n=530 and n=610) humans and a large language model (LLM) wrote relationship advice or relationship descriptions, either with or without instructions to sound human. New participants (n=428 and n=408) judged each text's source. Instructions to sound human were only effective for the LLM, reducing the human advantage. Study 3 (n=360 and n=350) showed that these effects persist when writers were instructed to avoid sounding like an LLM. Study 4 (n=219) tested empathy as mechanism of humanness and concluded that LLMs can produce empathy without humanness and humanness without empathy. Finally, computational text analysis (Study 5) indicated that LLMs become more human-like by applying an implicit representation of humanness to mimic stochastic empathy.
>
---
#### [replaced 035] CASE -- Condition-Aware Sentence Embeddings for Conditional Semantic Textual Similarity Measurement
- **分类: cs.CL**

- **简介: 该论文属于语义文本相似度任务，解决条件上下文下的句子嵌入问题。提出CASE方法，通过结合条件信息提升相似度测量效果。**

- **链接: [https://arxiv.org/pdf/2503.17279v4](https://arxiv.org/pdf/2503.17279v4)**

> **作者:** Gaifan Zhang; Yi Zhou; Danushka Bollegala
>
> **备注:** Accepted to EACL2026
>
> **摘要:** The meaning conveyed by a sentence often depends on the context in which it appears. Despite the progress of sentence embedding methods, it remains unclear how to best modify a sentence embedding conditioned on its context. To address this problem, we propose Condition-Aware Sentence Embeddings (CASE), an efficient and accurate method to create an embedding for a sentence under a given condition. First, CASE creates an embedding for the condition using a Large Language Model (LLM), where the sentence influences the attention scores computed for the tokens in the condition during pooling. Next, a supervised nonlinear projection is learned to reduce the dimensionality of the LLM-based text embeddings. We show that CASE significantly outperforms previously proposed Conditional Semantic Textual Similarity (C-STS) methods on an existing standard benchmark dataset. We find that subtracting the condition embedding consistently improves the C-STS performance of LLM-based text embeddings. Moreover, we propose a supervised dimensionality reduction method that not only reduces the dimensionality of LLM-based embeddings but also significantly improves their performance.
>
---
#### [replaced 036] Evaluating the Effect of Retrieval Augmentation on Social Biases
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，研究RAG系统对社会偏见的影响。旨在解决RAG是否放大文本中的社会偏见问题，通过实验分析不同语言和偏见类型下的生成结果。**

- **链接: [https://arxiv.org/pdf/2502.17611v3](https://arxiv.org/pdf/2502.17611v3)**

> **作者:** Tianhui Zhang; Yi Zhou; Danushka Bollegala
>
> **备注:** EACL26 main
>
> **摘要:** Retrieval Augmented Generation (RAG) has gained popularity as a method for conveniently incorporating novel facts that were not seen during the pre-training stage in Large Language Model (LLM)-based Natural Language Generation (NLG) systems. However, LLMs are known to encode significant levels of unfair social biases. The modulation of these biases by RAG in NLG systems is not well understood. In this paper, we systematically study the relationship between the different components of a RAG system and the social biases presented in the text generated across three languages (i.e. English, Japanese and Chinese) and four social bias types (i.e. gender, race, age and religion). Specifically, using the Bias Question Answering (BBQ) benchmark datasets, we evaluate the social biases in RAG responses from document collections with varying levels of stereotypical biases, employing multiple LLMs used as generators. We find that the biases in document collections are often amplified in the generated responses, even when the generating LLM exhibits a low-level of bias. Our findings raise concerns about the use of RAG as a technique for injecting novel facts into NLG systems and call for careful evaluation of potential social biases in RAG applications before their real-world deployment.
>
---
#### [replaced 037] Benchmarking LLMs for Political Science: A United Nations Perspective
- **分类: cs.CL; cs.CY; cs.ET**

- **简介: 该论文属于AI与政治科学交叉领域，旨在评估LLMs在联合国决策中的应用。提出UNBench基准，解决LLMs在政治决策中的能力评估问题，涵盖四个任务以模拟决策过程。**

- **链接: [https://arxiv.org/pdf/2502.14122v2](https://arxiv.org/pdf/2502.14122v2)**

> **作者:** Yueqing Liang; Liangwei Yang; Chen Wang; Congying Xia; Rui Meng; Xiongxiao Xu; Haoran Wang; Ali Payani; Kai Shu
>
> **备注:** This paper has been accepted at AAAI 2026 as an oral paper
>
> **摘要:** Large Language Models (LLMs) have achieved significant advances in natural language processing, yet their potential for high-stake political decision-making remains largely unexplored. This paper addresses the gap by focusing on the application of LLMs to the United Nations (UN) decision-making process, where the stakes are particularly high and political decisions can have far-reaching consequences. We introduce a novel dataset comprising publicly available UN Security Council (UNSC) records from 1994 to 2024, including draft resolutions, voting records, and diplomatic speeches. Using this dataset, we propose the United Nations Benchmark (UNBench), the first comprehensive benchmark designed to evaluate LLMs across four interconnected political science tasks: co-penholder judgment, representative voting simulation, draft adoption prediction, and representative statement generation. These tasks span the three stages of the UN decision-making process--drafting, voting, and discussing--and aim to assess LLMs' ability to understand and simulate political dynamics. Our experimental analysis demonstrates the potential and challenges of applying LLMs in this domain, providing insights into their strengths and limitations in political science. This work contributes to the growing intersection of AI and political science, opening new avenues for research and practical applications in global governance. The UNBench Repository can be accessed at: https://github.com/yueqingliang1/UNBench.
>
---
#### [replaced 038] On Fine-Grained I/O Complexity of Attention Backward Passes
- **分类: cs.LG; cs.AI; cs.CC; cs.CL**

- **简介: 该论文研究注意力机制的I/O复杂度，解决LLM训练与推理中的效率问题。通过分析前向和反向传播，提出优化算法并建立理论界限。**

- **链接: [https://arxiv.org/pdf/2410.09397v2](https://arxiv.org/pdf/2410.09397v2)**

> **作者:** Xiaoyu Li; Yingyu Liang; Zhenmei Shi; Zhao Song; Song Yue; Jiahao Zhang
>
> **摘要:** Large Language Models (LLMs) exhibit exceptional proficiency in handling extensive context windows in natural language. Nevertheless, the quadratic scaling of attention computation relative to sequence length creates substantial efficiency bottlenecks, necessitating the development of I/O-optimized algorithms. In this work, we conduct a systematic examination of the I/O complexity inherent in attention mechanisms, with a specific emphasis on the backward pass under both small and large cache settings. By leveraging the red-blue pebble game framework, we derive tight bounds for I/O complexity across the full spectrum of cache sizes. We validate that FlashAttention, one of the current industry standards, achieves optimality in the large-cache scenario for both forward and backward passes. Conversely, for small-cache environments, we introduce a novel algorithm that outperforms contemporary methods and successfully attains theoretical tight bounds. Furthermore, we expand our investigation to include sparse attention by establishing granular lower bounds for both forward and backward passes across all cache configurations. Ultimately, our results solidify the theoretical framework regarding I/O complexity in attention mechanisms, providing critical guidance for the development of efficient LLM training and inference systems.
>
---
#### [replaced 039] Beyond Memorization: A Rigorous Evaluation Framework for Medical Knowledge Editing
- **分类: cs.CL**

- **简介: 该论文属于医疗知识编辑任务，旨在解决现有方法仅实现表面记忆而非有效泛化的问题。提出MedEditBench框架和SGR-Edit方法，提升医疗知识编辑效果。**

- **链接: [https://arxiv.org/pdf/2506.03490v3](https://arxiv.org/pdf/2506.03490v3)**

> **作者:** Shigeng Chen; Linhao Luo; Zhangchi Qiu; Yanan Cao; Carl Yang; Shirui Pan
>
> **备注:** Accepted to EACL 2026 Main Conference
>
> **摘要:** Recently, knowledge editing (KE) has emerged as a promising approach to update specific facts in Large Language Models (LLMs) without the need for full retraining. Despite the effectiveness in general-domain benchmarks, their applicability to complex medical domain remains largely unexplored. Medical knowledge editing is particularly challenging, as it requires LLMs to internalize the knowledge and generalize to unseen scenarios for effective and interpretable decision-making. In this work, we propose a novel framework called MedEditBench to rigorously evaluate the effectiveness of existing KE methods in the medical domain. In MedEditBench, we introduce a new medical knowledge editing benchmark as well as three different knowledge editing paradigms, which are designed to assess the impact of different knowledge sources for editing. Our findings indicate that current KE methods result in only superficial memorization of the injected information, failing to generalize to new scenarios. To overcome this limitation, we present Self-Generated Rationale Editing (SGR-Edit), which utilizes model-derived rationales as the target knowledge for editing, thereby uncovering the underlying reasoning process and demonstrating significant improvements over existing KE approaches. Additionally, we offer deeper insights into medical knowledge editing, including the localization of medical knowledge in LLMs and the impact of sequential editing on evolving knowledge. This could provide practical guidance for implementing KE methods in real-world medical applications.
>
---
#### [replaced 040] Hierarchy-Aware Multimodal Unlearning for Medical AI
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于医疗AI中的多模态去遗忘任务，旨在解决数据隐私与模型遗忘的矛盾。提出MedForget基准和CHIP方法，实现高效且保留实用性的结构化遗忘。**

- **链接: [https://arxiv.org/pdf/2512.09867v3](https://arxiv.org/pdf/2512.09867v3)**

> **作者:** Fengli Wu; Vaidehi Patil; Jaehong Yoon; Yue Zhang; Mohit Bansal
>
> **备注:** Dataset and Code: https://github.com/fengli-wu/MedForget
>
> **摘要:** Pretrained Multimodal Large Language Models (MLLMs) are increasingly used in sensitive domains such as medical AI, where privacy regulations like HIPAA and GDPR require specific removal of individuals' or institutions' data. This motivates machine unlearning, which aims to remove the influence of target data from a trained model. However, existing unlearning benchmarks fail to reflect the hierarchical and multimodal structure of real-world medical data, limiting their ability to properly evaluate unlearning in practice. Therefore, we introduce MedForget, a hierarchy-aware multimodal unlearning benchmark that models hospital data as a nested structure, enabling fine-grained evaluation of multimodal unlearning across retain and forget splits. Experiments with current unlearning methods show that existing approaches struggle to achieve effective hierarchy-aware forgetting without degrading downstream medical utility. To address this limitation, we propose Cross-modal Hierarchy-Informed Projection for unlearning (CHIP), a training-free, hierarchy-aware multimodal unlearning method that deletes information by selectively removing target-specific weight subspaces while preserving sibling-shared information. Experiments show that CHIP achieves the highest forget-retain performance gap across all hierarchy levels while maintaining competitive downstream utility compared to existing methods. Overall, MedForget provides a practical, HIPAA-aligned benchmark for evaluating structured multimodal unlearning for medical data, and CHIP offers an effective and general solution for hierarchy-aware forgetting that balances deletion with utility.
>
---
#### [replaced 041] StealthGraph: Exposing Domain-Specific Risks in LLMs through Knowledge-Graph-Guided Harmful Prompt Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于LLM安全研究任务，旨在解决隐性有害提示生成难题。通过知识图谱引导生成领域相关提示，并进行双重路径混淆重写，提升提示的隐性和现实性。**

- **链接: [https://arxiv.org/pdf/2601.04740v2](https://arxiv.org/pdf/2601.04740v2)**

> **作者:** Huawei Zheng; Xinqi Jiang; Sen Yang; Shouling Ji; Yingcai Wu; Dazhen Deng
>
> **摘要:** Large language models (LLMs) are increasingly applied in specialized domains such as finance and healthcare, where they introduce unique safety risks. Domain-specific datasets of harmful prompts remain scarce and still largely rely on manual construction; public datasets mainly focus on explicit harmful prompts, which modern LLM defenses can often detect and refuse. In contrast, implicit harmful prompts-expressed through indirect domain knowledge-are harder to detect and better reflect real-world threats. We identify two challenges: transforming domain knowledge into actionable constraints and increasing the implicitness of generated harmful prompts. To address them, we propose an end-to-end framework that first performs knowledge-graph-guided harmful prompt generation to systematically produce domain-relevant prompts, and then applies dual-path obfuscation rewriting to convert explicit harmful prompts into implicit variants via direct and context-enhanced rewriting. This framework yields high-quality datasets combining strong domain relevance with implicitness, enabling more realistic red-teaming and advancing LLM safety research. We release our code and datasets at GitHub.
>
---
#### [replaced 042] Improving Estonian Text Simplification through Pretrained Language Models and Custom Datasets
- **分类: cs.CL**

- **简介: 该论文属于文本简化任务，针对爱沙尼亚语资源不足的问题，利用预训练语言模型和自建数据集提升简化效果。**

- **链接: [https://arxiv.org/pdf/2501.15624v2](https://arxiv.org/pdf/2501.15624v2)**

> **作者:** Eduard Barbu; Meeri-Ly Muru; Sten Marcus Malva
>
> **备注:** RANLP 2025 version, including code and data
>
> **摘要:** This paper presents a method for text simplification based on two neural architectures: a neural machine translation (NMT) model and a fine-tuned large language model (LLaMA). Given the scarcity of existing resources for Estonian, a new dataset was created by combining manually translated corpora with GPT-4.0-generated simplifications. OpenNMT was selected as a representative NMT-based system, while LLaMA was fine-tuned on the constructed dataset. Evaluation shows LLaMA outperforms OpenNMT in grammaticality, readability, and meaning preservation. These results underscore the effectiveness of large language models for text simplification in low-resource language settings. The complete dataset, fine-tuning scripts, and evaluation pipeline are provided in a publicly accessible supplementary package to support reproducibility and adaptation to other languages.
>
---
#### [replaced 043] Identifying Reliable Evaluation Metrics for Scientific Text Revision
- **分类: cs.CL**

- **简介: 该论文属于科学文本修订评估任务，旨在解决传统指标无法准确衡量文本改进的问题。通过人工标注、无参考评估和LLM判断，提出混合评估方法以提高可靠性。**

- **链接: [https://arxiv.org/pdf/2506.04772v5](https://arxiv.org/pdf/2506.04772v5)**

> **作者:** Léane Jourdan; Florian Boudin; Richard Dufour; Nicolas Hernandez
>
> **备注:** V5 contains the English version, (ACL 2025 main, 26 pages) and V4 contains the French version (TALN 2025, 32 pages), both with corrected results for cramer's v and pairwise accuracy
>
> **摘要:** Evaluating text revision in scientific writing remains a challenge, as traditional metrics such as ROUGE and BERTScore primarily focus on similarity rather than capturing meaningful improvements. In this work, we analyse and identify the limitations of these metrics and explore alternative evaluation methods that better align with human judgments. We first conduct a manual annotation study to assess the quality of different revisions. Then, we investigate reference-free evaluation metrics from related NLP domains. Additionally, we examine LLM-as-a-judge approaches, analysing their ability to assess revisions with and without a gold reference. Our results show that LLMs effectively assess instruction-following but struggle with correctness, while domain-specific metrics provide complementary insights. We find that a hybrid approach combining LLM-as-a-judge evaluation and task-specific metrics offers the most reliable assessment of revision quality.
>
---
#### [replaced 044] Orthogonal Low-rank Adaptation in Lie Groups for Continual Learning of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于持续学习任务，解决大语言模型在多任务学习中的灾难性遗忘问题。提出OLieRA框架，通过乘法更新保持参数几何结构，实现任务间正交性，提升性能。**

- **链接: [https://arxiv.org/pdf/2509.06100v2](https://arxiv.org/pdf/2509.06100v2)**

> **作者:** Kefan Cao; Shuaicheng Wu
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** Large language models (LLMs) suffer from catastrophic forgetting in sequential multi-task learning. Existing parameter regularization methods (e.g., O-LoRA, N-LoRA) mitigate interference via low-rank subspace orthogonality, but additive updates distort the intrinsic geometry of model parameters. We propose \textbf{OLieRA}, a Lie group based fine-tuning framework that preserves parameter geometry through multiplicative updates while enforcing orthogonality across task subspaces. OLieRA achieves state-of-the-art performance on the Standard CL benchmark and remains highly competitive under large task sequences. It further inherits the replay-free and task-ID free inference properties of O-LoRA, establishing a principled paradigm for continual learning in LLMs.
>
---
#### [replaced 045] Frame-Stacked Local Transformers For Efficient Multi-Codebook Speech Generation
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音生成任务，解决多代码本结构下的依赖问题。通过帧堆叠和局部Transformer提升生成效率与质量。**

- **链接: [https://arxiv.org/pdf/2509.19592v2](https://arxiv.org/pdf/2509.19592v2)**

> **作者:** Roy Fejgin; Paarth Neekhara; Xuesong Yang; Edresson Casanova; Ryan Langman; Jaehyeon Kim; Subhankar Ghosh; Shehzeen Hussain; Jason Li
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Speech generation models based on large language models (LLMs) typically operate on discrete acoustic codes, which differ fundamentally from text tokens due to their multicodebook structure. At each timestep, models must predict N codebook entries jointly, introducing dependencies that challenge simple parallel prediction approaches. Parallel prediction assumes independence among codebooks, yielding efficient decoding but often at the cost of reduced fidelity. To address this, hierarchical strategies employ a local transformer (LT) to refine predictions and capture intra-timestep dependencies. In this work, we systematically investigate two LT architectures: an autoregressive transformer that generates codebooks sequentially, and a MaskGIT-based transformer that performs iterative masked prediction. Both designs further enable frame stacking, where the primary transformer predicts multiple frames jointly, and the LT decodes their codebooks, offering improvements in speed without compromising perceptual quality. Through extensive analysis, we characterize the tradeoffs between parallel and iterative sampling strategies across different throughput and quality regimes. Finally, we propose practical guidelines for selecting decoding strategies based on deployment priorities such as computational efficiency and synthesis fidelity.
>
---
#### [replaced 046] Cognitive Control Architecture (CCA): A Lifecycle Supervision Framework for Robustly Aligned AI Agents
- **分类: cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于AI安全任务，旨在解决LLM代理受间接提示注入攻击的问题。提出CCA框架，通过双层防御机制实现全程监督，有效抵御攻击并平衡安全与效率。**

- **链接: [https://arxiv.org/pdf/2512.06716v2](https://arxiv.org/pdf/2512.06716v2)**

> **作者:** Zhibo Liang; Tianze Hu; Zaiye Chen; Mingjie Tang
>
> **摘要:** Autonomous Large Language Model (LLM) agents exhibit significant vulnerability to Indirect Prompt Injection (IPI) attacks. These attacks hijack agent behavior by polluting external information sources, exploiting fundamental trade-offs between security and functionality in existing defense mechanisms. This leads to malicious and unauthorized tool invocations, diverting agents from their original objectives. The success of complex IPIs reveals a deeper systemic fragility: while current defenses demonstrate some effectiveness, most defense architectures are inherently fragmented. Consequently, they fail to provide full integrity assurance across the entire task execution pipeline, forcing unacceptable multi-dimensional compromises among security, functionality, and efficiency. Our method is predicated on a core insight: no matter how subtle an IPI attack, its pursuit of a malicious objective will ultimately manifest as a detectable deviation in the action trajectory, distinct from the expected legitimate plan. Based on this, we propose the Cognitive Control Architecture (CCA), a holistic framework achieving full-lifecycle cognitive supervision. CCA constructs an efficient, dual-layered defense system through two synergistic pillars: (i) proactive and preemptive control-flow and data-flow integrity enforcement via a pre-generated "Intent Graph"; and (ii) an innovative "Tiered Adjudicator" that, upon deviation detection, initiates deep reasoning based on multi-dimensional scoring, specifically designed to counter complex conditional attacks. Experiments on the AgentDojo benchmark substantiate that CCA not only effectively withstands sophisticated attacks that challenge other advanced defense methods but also achieves uncompromised security with notable efficiency and robustness, thereby reconciling the aforementioned multi-dimensional trade-off.
>
---
#### [replaced 047] Modern Hopfield Networks Require Chain-of-Thought to Solve $\mathsf{NC}^1$-Hard Problems
- **分类: cs.CC; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究现代霍普菲尔德网络的表达能力，探讨其解决复杂问题的局限性及改进方法。任务属于理论计算机科学与深度学习交叉领域，旨在分析网络的计算能力并提出增强方案。**

- **链接: [https://arxiv.org/pdf/2412.05562v2](https://arxiv.org/pdf/2412.05562v2)**

> **作者:** Yang Cao; Xiaoyu Li; Yuanpeng Li; Yingyu Liang; Zhenmei Shi; Zhao Song
>
> **摘要:** Modern Hopfield Networks (MHNs) have emerged as powerful components in deep learning, serving as effective replacements for pooling layers, LSTMs, and attention mechanisms. While recent advancements have significantly improved their storage capacity and retrieval efficiency, their fundamental theoretical boundaries remain underexplored. In this paper, we rigorously characterize the expressive power of MHNs through the lens of circuit complexity theory. We prove that $\mathrm{poly}(n)$-precision MHNs with constant depth and linear hidden dimension fall within the $\mathsf{DLOGTIME}$-uniform $\mathsf{TC}^0$ complexity class. Consequently, assuming $\mathsf{TC}^0 \neq \mathsf{NC}^1$, we demonstrate that these architectures are incapable of solving $\mathsf{NC}^1$-hard problems, such as undirected graph connectivity and tree isomorphism. We further extend these impossibility results to Kernelized Hopfield Networks. However, we show that these limitations are not absolute: we prove that equipping MHNs with a Chain-of-Thought (CoT) mechanism enables them to transcend the $\mathsf{TC}^0$ barrier, allowing them to solve inherently serial problems like the word problem for the permutation group $S_5$. Collectively, our results delineate a fine-grained boundary between the capabilities of standard MHNs and those augmented with reasoning steps.
>
---
