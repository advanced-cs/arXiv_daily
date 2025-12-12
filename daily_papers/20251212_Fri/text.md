# 自然语言处理 cs.CL

- **最新发布 48 篇**

- **更新 35 篇**

## 最新发布

#### [new 001] Unforgotten Safety: Preserving Safety Alignment of Large Language Models with Continual Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在持续学习中因微调导致的安全对齐退化问题，旨在防止安全遗忘。通过适配多种持续学习方法，在不同数据场景和模型上验证其有效性，结果表明DER方法能有效保持安全性和任务性能。**

- **链接: [https://arxiv.org/pdf/2512.10150v1](https://arxiv.org/pdf/2512.10150v1)**

> **作者:** Lama Alssum; Hani Itani; Hasan Abed Al Kader Hammoud; Philip Torr; Adel Bibi; Bernard Ghanem
>
> **摘要:** The safety alignment of large language models (LLMs) is becoming increasingly important with their democratization. In this paper, we study the safety degradation that comes with adapting LLMs to new tasks. We attribute this safety compromise to catastrophic forgetting and frame the problem of preserving safety when fine-tuning as a continual learning (CL) problem. We consider the fine-tuning-as-a-service setup where the user uploads their data to a service provider to get a customized model that excels on the user's selected task. We adapt several CL approaches from the literature and systematically evaluate their ability to mitigate safety degradation. These include regularization-based, memory-based, and model merging approaches. We consider two scenarios, (1) benign user data and (2) poisoned user data. Our results demonstrate that CL approaches consistently achieve lower attack success rates than standard fine-tuning. Among these, DER outperforms both other CL methods and existing safety-preserving baselines while maintaining task utility. These findings generalize across three downstream tasks (GSM8K, SST2, Code) and three model families (LLaMA2-7B, Mistral-7B, Gemma-2B), establishing CL as a practical solution to preserve safety.
>
---
#### [new 002] What Kind of Reasoning (if any) is an LLM actually doing? On the Stochastic Nature and Abductive Appearance of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨大语言模型（LLM）是否真正在推理，分析其基于统计模式生成文本的随机本质，指出其看似溯因推理实为模仿人类文本结构。研究揭示LLM不具备真实理解与验证能力，强调输出需谨慎评估，属于对LLM推理能力的本质辨析与批判性评价任务。**

- **链接: [https://arxiv.org/pdf/2512.10080v1](https://arxiv.org/pdf/2512.10080v1)**

> **作者:** Luciano Floridi; Jessica Morley; Claudio Novelli; David Watson
>
> **摘要:** This article looks at how reasoning works in current Large Language Models (LLMs) that function using the token-completion method. It examines their stochastic nature and their similarity to human abductive reasoning. The argument is that these LLMs create text based on learned patterns rather than performing actual abductive reasoning. When their output seems abductive, this is largely because they are trained on human-generated texts that include reasoning structures. Examples are used to show how LLMs can produce plausible ideas, mimic commonsense reasoning, and give explanatory answers without being grounded in truth, semantics, verification, or understanding, and without performing any real abductive reasoning. This dual nature, where the models have a stochastic base but appear abductive in use, has important consequences for how LLMs are evaluated and applied. They can assist with generating ideas and supporting human thinking, but their outputs must be critically assessed because they cannot identify truth or verify their explanations. The article concludes by addressing five objections to these points, noting some limitations in the analysis, and offering an overall evaluation.
>
---
#### [new 003] Multilingual VLM Training: Adapting an English-Trained VLM to French
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言视觉-语言模型研究任务，旨在解决英文VLM向法语迁移的挑战。作者比较了翻译、LoRA微调和两阶段微调方法，发现数据质量是主要瓶颈，强调需加强目标语言数据建设。**

- **链接: [https://arxiv.org/pdf/2512.10336v1](https://arxiv.org/pdf/2512.10336v1)**

> **作者:** Jules Lahmi; Alexis Roger
>
> **摘要:** Artificial intelligence has made great progress in recent years, particularly in the development of Vision--Language Models (VLMs) that understand both visual and textual data. However, these advancements remain largely limited to English, reducing their accessibility for non--English speakers. It is essential to extend these capabilities to a broader range of languages. This paper explores the challenges of adapting an English-trained VLM to different languages. To this end, we will explore and compare different methods for their performance and computational cost. We consider a translation-based pipeline, LoRA finetuning, and a two-stage finetuning strategy that separates vision adaptation from language adaptation. To evaluate these methods, we use a combination of standard multimodal benchmarks translated into the target language and manual assessments by native experts. The results reveal that dataset translation remains a major bottleneck in multilingual VLM performance, with data quality limiting the effectiveness of training and evaluation. These findings suggest that future efforts should focus on native-language dataset collection and improved translation strategies.
>
---
#### [new 004] Grow Up and Merge: Scaling Strategies for Efficient Language Adaptation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究如何通过模型扩展（scaling）高效地将预训练语言模型适配到中低资源语言。任务是语言适应与多语言模型构建，旨在解决小规模下多语言模型性能不足及灾难性遗忘问题，提出扩展模型并探索合并策略以提升数据效率和多语言能力。**

- **链接: [https://arxiv.org/pdf/2512.10772v1](https://arxiv.org/pdf/2512.10772v1)**

> **作者:** Kevin Glocker; Kätriin Kukk; Romina Oji; Marcel Bollmann; Marco Kuhlmann; Jenny Kunz
>
> **摘要:** Achieving high-performing language models which include medium- and lower-resource languages remains a challenge. Massively multilingual models still underperform compared to language-specific adaptations, especially at smaller model scales. In this work, we investigate scaling as an efficient strategy for adapting pretrained models to new target languages. Through comprehensive scaling ablations with approximately FLOP-matched models, we test whether upscaling an English base model enables more effective and resource-efficient adaptation than standard continued pretraining. We find that, once exposed to sufficient target-language data, larger upscaled models can match or surpass the performance of smaller models continually pretrained on much more data, demonstrating the benefits of scaling for data efficiency. Scaling also helps preserve the base model's capabilities in English, thus reducing catastrophic forgetting. Finally, we explore whether such scaled, language-specific models can be merged to construct modular and flexible multilingual systems. We find that while merging remains less effective than joint multilingual training, upscaled merges perform better than smaller ones. We observe large performance differences across merging methods, suggesting potential for improvement through merging approaches specialized for language-level integration.
>
---
#### [new 005] Enhancing Next-Generation Language Models with Knowledge Graphs: Extending Claude, Mistral IA, and GPT-4 via KG-BERT
- **分类: cs.CL**

- **简介: 该论文属知识增强语言模型任务，旨在解决大语言模型事实性错误问题。通过引入KG-BERT融合知识图谱，提升Claude、Mistral IA和GPT-4在问答与实体链接等任务中的事实准确性和推理能力。**

- **链接: [https://arxiv.org/pdf/2512.10440v1](https://arxiv.org/pdf/2512.10440v1)**

> **作者:** Nour El Houda Ben Chaabene; Hamza Hammami
>
> **备注:** This paper was accepted and scheduled for inclusion in the ICALT 2025 proceedings but was ultimately not published due to absence from the conference presentation. It appears in the official program booklet. Conference: 2025 IEEE International Conference on Advanced Learning Technologies (ICALT)
>
> **摘要:** Large language models (LLMs) like Claude, Mistral IA, and GPT-4 excel in NLP but lack structured knowledge, leading to factual inconsistencies. We address this by integrating Knowledge Graphs (KGs) via KG-BERT to enhance grounding and reasoning. Experiments show significant gains in knowledge-intensive tasks such as question answering and entity linking. This approach improves factual reliability and enables more context-aware next-generation LLMs.
>
---
#### [new 006] PARAN: Persona-Augmented Review ANswering system on Food Delivery Review Dataset
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究自动化评论回复生成任务，旨在解决用户信息少导致回复泛化的问题。提出两阶段提示框架，从短评中推断显式与隐式用户画像，并融入生成提示以提升回复个性化，无需模型微调，在真实外卖数据上验证了方法有效性。**

- **链接: [https://arxiv.org/pdf/2512.10148v1](https://arxiv.org/pdf/2512.10148v1)**

> **作者:** Moonsoo Park; Jeongseok Yun; Bohyung Kim
>
> **摘要:** Personalized review response generation presents a significant challenge in domains where user information is limited, such as food delivery platforms. While large language models (LLMs) offer powerful text generation capabilities, they often produce generic responses when lacking contextual user data, reducing engagement and effectiveness. In this work, we propose a two-stage prompting framework that infers both explicit (e.g., user-stated preferences) and implicit (e.g., demographic or stylistic cues) personas directly from short review texts. These inferred persona attributes are then incorporated into the response generation prompt to produce user-tailored replies. To encourage diverse yet faithful generations, we adjust decoding temperature during inference. We evaluate our method using a real-world dataset collected from a Korean food delivery app, and assess its impact on precision, diversity, and semantic consistency. Our findings highlight the effectiveness of persona-augmented prompting in enhancing the relevance and personalization of automated responses without requiring model fine-tuning.
>
---
#### [new 007] AgriGPT-Omni: A Unified Speech-Vision-Text Framework for Multilingual Agricultural Intelligence
- **分类: cs.CL**

- **简介: 该论文提出AgriGPT-Omni，解决多语言农业智能中缺乏语音数据、统一架构和评测基准的问题。构建了大规模多语言农业语音数据集，训练首个融合语音、视觉、文本的农业大模型，并发布首个三模态农业评测基准AgriBench-Omni-2K。**

- **链接: [https://arxiv.org/pdf/2512.10624v1](https://arxiv.org/pdf/2512.10624v1)**

> **作者:** Bo Yang; Lanfei Feng; Yunkui Chen; Yu Zhang; Jianyu Zhang; Xiao Xu; Nueraili Aierken; Shijian Li
>
> **摘要:** Despite rapid advances in multimodal large language models, agricultural applications remain constrained by the lack of multilingual speech data, unified multimodal architectures, and comprehensive evaluation benchmarks. To address these challenges, we present AgriGPT-Omni, an agricultural omni-framework that integrates speech, vision, and text in a unified framework. First, we construct a scalable data synthesis and collection pipeline that converts agricultural texts and images into training data, resulting in the largest agricultural speech dataset to date, including 492K synthetic and 1.4K real speech samples across six languages. Second, based on this, we train the first agricultural omni-model via a three-stage paradigm: textual knowledge injection, progressive multimodal alignment, and GRPO-based reinforcement learning, enabling unified reasoning across languages and modalities. Third, we propose AgriBench-Omni-2K, the first tri-modal benchmark for agriculture, covering diverse speech-vision-text tasks and multilingual slices, with standardized protocols and reproducible tools. Experiments show that AgriGPT-Omni significantly outperforms general-purpose baselines on multilingual and multimodal reasoning as well as real-world speech understanding. All models, data, benchmarks, and code will be released to promote reproducible research, inclusive agricultural intelligence, and sustainable AI development for low-resource regions.
>
---
#### [new 008] Workflow is All You Need: Escaping the "Statistical Smoothing Trap" via High-Entropy Information Foraging and Adversarial Pacing
- **分类: cs.CL; cs.AI; cs.CY; q-fin.GN**

- **简介: 该论文针对长文本生成中低幻觉、高逻辑性与个性化难以兼顾的问题，提出DeepNews框架，通过信息觅食、结构化规划与对抗性提示，模拟专家写作认知过程，显著提升生成质量与真实性。**

- **链接: [https://arxiv.org/pdf/2512.10121v1](https://arxiv.org/pdf/2512.10121v1)**

> **作者:** Zhongjie Jiang
>
> **备注:** 22 pages, 8 figures. Includes an ecological validity blind test where the Agentic Workflow achieved a 25% acceptance rate in top-tier media, decisively outperforming the SOTA Zero-shot baseline (0%). Features the DNFO-v5 ontology
>
> **摘要:** Central to long-form text generation in vertical domains is the "impossible trinity" confronting current large language models (LLMs): the simultaneous achievement of low hallucination, deep logical coherence, and personalized expression. This study establishes that this bottleneck arises from existing generative paradigms succumbing to the Statistical Smoothing Trap, a phenomenon that overlooks the high-entropy information acquisition and structured cognitive processes integral to expert-level writing. To address this limitation, we propose the DeepNews Framework, an agentic workflow that explicitly models the implicit cognitive processes of seasoned financial journalists. The framework integrates three core modules: first, a dual-granularity retrieval mechanism grounded in information foraging theory, which enforces a 10:1 saturated information input ratio to mitigate hallucinatory outputs; second, schema-guided strategic planning, a process leveraging domain expert knowledge bases (narrative schemas) and Atomic Blocks to forge a robust logical skeleton; third, adversarial constraint prompting, a technique deploying tactics including Rhythm Break and Logic Fog to disrupt the probabilistic smoothness inherent in model-generated text. Experiments delineate a salient Knowledge Cliff in deep financial reporting: content truthfulness collapses when retrieved context falls below 15,000 characters, while a high-redundancy input exceeding 30,000 characters stabilizes the Hallucination-Free Rate (HFR) above 85%. In an ecological validity blind test conducted with a top-tier Chinese technology media outlet, the DeepNews system--built on a previous-generation model (DeepSeek-V3-0324)-achieved a 25% submission acceptance rate, significantly outperforming the 0% acceptance rate of zero-shot generation by a state-of-the-art (SOTA) model (GPT-5).
>
---
#### [new 009] Textual Data Bias Detection and Mitigation - An Extensible Pipeline with Experimental Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本数据去偏任务，旨在解决训练数据中的表征偏差与显性刻板印象问题。作者提出一个可扩展的四阶段管道，结合LLM生成词表、统计评分、语言学过滤与数据增强，在性别、宗教和年龄上验证了数据去偏效果，但发现模型层面的偏见缓解仍存在挑战。**

- **链接: [https://arxiv.org/pdf/2512.10734v1](https://arxiv.org/pdf/2512.10734v1)**

> **作者:** Rebekka Görge; Sujan Sai Gannamaneni; Tabea Naeven; Hammam Abdelwahab; Héctor Allende-Cid; Armin B. Cremers; Lennard Helmer; Michael Mock; Anna Schmitz; Songkai Xue; Elif Yildirir; Maximilian Poretschkin; Stefan Wrobel
>
> **摘要:** Textual data used to train large language models (LLMs) exhibits multifaceted bias manifestations encompassing harmful language and skewed demographic distributions. Regulations such as the European AI Act require identifying and mitigating biases against protected groups in data, with the ultimate goal of preventing unfair model outputs. However, practical guidance and operationalization are lacking. We propose a comprehensive data bias detection and mitigation pipeline comprising four components that address two data bias types, namely representation bias and (explicit) stereotypes for a configurable sensitive attribute. First, we leverage LLM-generated word lists created based on quality criteria to detect relevant group labels. Second, representation bias is quantified using the Demographic Representation Score. Third, we detect and mitigate stereotypes using sociolinguistically informed filtering. Finally, we compensate representation bias through Grammar- and Context-Aware Counterfactual Data Augmentation. We conduct a two-fold evaluation using the examples of gender, religion and age. First, the effectiveness of each individual component on data debiasing is evaluated through human validation and baseline comparison. The findings demonstrate that we successfully reduce representation bias and (explicit) stereotypes in a text dataset. Second, the effect of data debiasing on model bias reduction is evaluated by bias benchmarking of several models (0.6B-8B parameters), fine-tuned on the debiased text dataset. This evaluation reveals that LLMs fine-tuned on debiased data do not consistently show improved performance on bias benchmarks, exposing critical gaps in current evaluation methodologies and highlighting the need for targeted data manipulation to address manifested model bias.
>
---
#### [new 010] TRIDENT: A Redundant Architecture for Caribbean-Accented Emergency Speech Triage
- **分类: cs.CL**

- **简介: 该论文提出TRIDENT架构，解决加勒比口音紧急语音识别性能差的问题。通过融合口音适配的语音识别、大语言模型实体抽取与生物声学 distress 检测，为调度员提供三重辅助信号，提升对非标准英语紧急呼叫的公平处理能力。**

- **链接: [https://arxiv.org/pdf/2512.10741v1](https://arxiv.org/pdf/2512.10741v1)**

> **作者:** Elroy Galbraith; Chadwick Sutherland; Donahue Morgan
>
> **摘要:** Emergency speech recognition systems exhibit systematic performance degradation on non-standard English varieties, creating a critical gap in services for Caribbean populations. We present TRIDENT (Transcription and Routing Intelligence for Dispatcher-Empowered National Triage), a three-layer dispatcher-support architecture designed to structure emergency call inputs for human application of established triage protocols (the ESI for routine operations and START for mass casualty events), even when automatic speech recognition fails. The system combines Caribbean-accent-tuned ASR, local entity extraction via large language models, and bio-acoustic distress detection to provide dispatchers with three complementary signals: transcription confidence, structured clinical entities, and vocal stress indicators. Our key insight is that low ASR confidence, rather than representing system failure, serves as a valuable queue prioritization signal -- particularly when combined with elevated vocal distress markers indicating a caller in crisis whose speech may have shifted toward basilectal registers. A complementary insight drives the entity extraction layer: trained responders and composed bystanders may report life-threatening emergencies without elevated vocal stress, requiring semantic analysis to capture clinical indicators that paralinguistic features miss. We describe the architectural design, theoretical grounding in psycholinguistic research on stress-induced code-switching, and deployment considerations for offline operation during disaster scenarios. This work establishes a framework for accent-resilient emergency AI that ensures Caribbean voices receive equitable access to established national triage protocols. Empirical validation on Caribbean emergency calls remains future work.
>
---
#### [new 011] Decoding Student Minds: Leveraging Conversational Agents for Psychological and Learning Analysis
- **分类: cs.CL**

- **简介: 该论文提出一种融合心理感知的对话系统，旨在同时提升学习效果与情绪健康。通过结合LLM、KG-BERT与带注意力的双向LSTM，利用多模态数据实时识别学生的认知与情感状态，实现个性化教育干预。**

- **链接: [https://arxiv.org/pdf/2512.10441v1](https://arxiv.org/pdf/2512.10441v1)**

> **作者:** Nour El Houda Ben Chaabene; Hamza Hammami; Laid Kahloul
>
> **备注:** This manuscript is currently under peer review in Expert Systems with Applications
>
> **摘要:** This paper presents a psychologically-aware conversational agent designed to enhance both learning performance and emotional well-being in educational settings. The system combines Large Language Models (LLMs), a knowledge graph-enhanced BERT (KG-BERT), and a bidirectional Long Short-Term Memory (LSTM) with attention to classify students' cognitive and affective states in real time. Unlike prior chatbots limited to either tutoring or affective support, our approach leverages multimodal data-including textual semantics, prosodic speech features, and temporal behavioral trends-to infer engagement, stress, and conceptual understanding. A pilot study with university students demonstrated improved motivation, reduced stress, and moderate academic gains compared to baseline methods. These results underline the promise of integrating semantic reasoning, multimodal fusion, and temporal modeling to support adaptive, student-centered educational interventions.
>
---
#### [new 012] Generate-Then-Validate: A Novel Question Generation Approach Using Small Language Models
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于自动问答生成任务，旨在利用小语言模型（SLM）生成高质量问题以补充大模型的不足。提出“生成-验证”新方法，先生成候选问题，再通过概率推理筛选，经人工与大模型评估，验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2512.10110v1](https://arxiv.org/pdf/2512.10110v1)**

> **作者:** Yumou Wei; John Stamper; Paulo F. Carvalho
>
> **备注:** Accepted as a full research paper for the 16th International Conference on Learning Analytics and Knowledge (LAK'26)
>
> **摘要:** We explore the use of small language models (SLMs) for automatic question generation as a complement to the prevalent use of their large counterparts in learning analytics research. We present a novel question generation pipeline that leverages both the text generation and the probabilistic reasoning abilities of SLMs to generate high-quality questions. Adopting a "generate-then-validate" strategy, our pipeline first performs expansive generation to create an abundance of candidate questions and refine them through selective validation based on novel probabilistic reasoning. We conducted two evaluation studies, one with seven human experts and the other with a large language model (LLM), to assess the quality of the generated questions. Most judges (humans or LLMs) agreed that the generated questions had clear answers and generally aligned well with the intended learning objectives. Our findings suggest that an SLM can effectively generate high-quality questions when guided by a well-designed pipeline that leverages its strengths.
>
---
#### [new 013] Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于问答任务，旨在解决现有检索增强生成（RAG）方法在单跳与多跳问答中检索错误和幻觉问题。提出CoopRAG框架，通过子问题分解、层间对比重排序和推理链重建，实现检索器与大模型的协同优化，提升问答与检索效果。**

- **链接: [https://arxiv.org/pdf/2512.10422v1](https://arxiv.org/pdf/2512.10422v1)**

> **作者:** Youmin Ko; Sungjong Seo; Hyunjoon Kim
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Since large language models (LLMs) have a tendency to generate factually inaccurate output, retrieval-augmented generation (RAG) has gained significant attention as a key means to mitigate this downside of harnessing only LLMs. However, existing RAG methods for simple and multi-hop question answering (QA) are still prone to incorrect retrievals and hallucinations. To address these limitations, we propose CoopRAG, a novel RAG framework for the question answering task in which a retriever and an LLM work cooperatively with each other by exchanging informative knowledge, and the earlier and later layers of the retriever model work cooperatively with each other to accurately rank the retrieved documents relevant to a given query. In this framework, we (i) unroll a question into sub-questions and a reasoning chain in which uncertain positions are masked, (ii) retrieve the documents relevant to the question augmented with the sub-questions and the reasoning chain, (iii) rerank the documents by contrasting layers of the retriever, and (iv) reconstruct the reasoning chain by filling the masked positions via the LLM. Our experiments demonstrate that CoopRAG consistently outperforms state-of-the-art QA methods on three multi-hop QA datasets as well as a simple QA dataset in terms of both the retrieval and QA performances. Our code is available.\footnote{https://github.com/meaningful96/CoopRAG}
>
---
#### [new 014] Computational emotion analysis with multimodal LLMs: Current evidence on an emerging methodological opportunity
- **分类: cs.CL**

- **简介: 该论文属于计算社会科学任务，旨在评估多模态大语言模型（mLLMs）在视频情感分析中的有效性。研究通过对比人工标注数据，检验mLLMs对政治演讲中情绪唤醒度的识别能力，发现其在理想条件下可靠，但在真实议会场景中表现不佳，提示需谨慎应用于实证研究。**

- **链接: [https://arxiv.org/pdf/2512.10882v1](https://arxiv.org/pdf/2512.10882v1)**

> **作者:** Hauke Licht
>
> **摘要:** Emotions are central to politics and analyzing their role in political communication has a long tradition. As research increasingly leverages audio-visual materials to analyze the display of emotions, the emergence of multimodal generative AI promises great advances. However, we lack evidence about the effectiveness of multimodal AI in emotion analysis. This paper addresses this gap by evaluating current multimodal large language models (mLLMs) in video-based analysis of emotional arousal in two complementary data sets of human-labeled video recordings. I find that under ideal circumstances, mLLMs' emotional arousal ratings are highly reliable and show little to know indication of demographic bias. However, in recordings of speakers in real-world parliamentary debates, mLLMs' arousal ratings fail to deliver on this promise with potential negative consequences for downstream statistical inferences. This study therefore underscores the need for continued, thorough evaluation of emerging generative AI methods in political analysis and contributes a suitable replicable framework.
>
---
#### [new 015] XDoGE: Multilingual Data Reweighting to Enhance Language Inclusivity in LLMs
- **分类: cs.CL**

- **简介: 该论文属多语言大模型优化任务，旨在缓解主流语言主导导致的低资源语言性能下降问题。提出XDoGE方法，通过数据重加权与持续预训练，提升模型对伊比利亚语系等低资源语言的支持能力，并发布新模型IberianLLM-7B-Instruct。**

- **链接: [https://arxiv.org/pdf/2512.10545v1](https://arxiv.org/pdf/2512.10545v1)**

> **作者:** Iñaki Lacunza; José Javier Saiz; Alexander Shvets; Aitor Gonzalez-Agirre; Marta Villegas
>
> **备注:** Accepted and presented at the LLMs4All workshop at the IEEE BigData 2025 Conference, Macau - December 8-11, 2025
>
> **摘要:** Current large language models (LLMs) are trained on massive amounts of text data, primarily from a few dominant languages. Studies suggest that this over-reliance on high-resource languages, such as English, hampers LLM performance in mid- and low-resource languages. To mitigate this problem, we propose to (i) optimize the language distribution by training a small proxy model within a domain-reweighing DoGE algorithm that we extend to XDoGE for a multilingual setup, and (ii) rescale the data and train a full-size model with the established language weights either from scratch or within a continual pre-training phase (CPT). We target six languages possessing a variety of geographic and intra- and inter-language-family relations, namely, English and Spanish (high-resource), Portuguese and Catalan (mid-resource), Galician and Basque (low-resource). We experiment with Salamandra-2b, which is a promising model for these languages. We investigate the effects of substantial data repetition on minor languages and under-sampling on dominant languages using the IberoBench framework for quantitative evaluation. Finally, we release a new promising IberianLLM-7B-Instruct model centering on Iberian languages and English that we pretrained from scratch and further improved using CPT with the XDoGE weights.
>
---
#### [new 016] T-pro 2.0: An Efficient Russian Hybrid-Reasoning Model and Playground
- **分类: cs.CL**

- **简介: 该论文聚焦于构建高效的俄语大语言模型，解决推理效率与资源开放问题。作者提出T-pro 2.0模型，支持混合推理与快速解码，并开源模型权重、数据集、基准与推理工具，推动可复现的俄语AI研究。**

- **链接: [https://arxiv.org/pdf/2512.10430v1](https://arxiv.org/pdf/2512.10430v1)**

> **作者:** Dmitrii Stoianov; Danil Taranets; Olga Tsymboi; Ramil Latypov; Almaz Dautov; Vladislav Kruglikov; Nikita Surkov; German Abramov; Pavel Gein; Dmitry Abulkhanov; Mikhail Gashkov; Viktor Zelenkovskiy; Artem Batalov; Aleksandr Medvedev; Anatolii Potapov
>
> **摘要:** We introduce T-pro 2.0, an open-weight Russian LLM for hybrid reasoning and efficient inference. The model supports direct answering and reasoning-trace generation, using a Cyrillic-dense tokenizer and an adapted EAGLE speculative-decoding pipeline to reduce latency. To enable reproducible and extensible research, we release the model weights, the T-Wix 500k instruction corpus, the T-Math reasoning benchmark, and the EAGLE weights on Hugging Face. These resources allow users to study Russian-language reasoning and to extend or adapt both the model and the inference pipeline. A public web demo exposes reasoning and non-reasoning modes and illustrates the speedups achieved by our inference stack across domains. T-pro 2.0 thus serves as an accessible open system for building and evaluating efficient, practical Russian LLM applications.
>
---
#### [new 017] Sliding Window Attention Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何将全注意力预训练的大语言模型适配滑动窗口注意力，以降低长文本推理成本。提出SWAA方法组合，解决训练与推理不匹配导致的性能下降问题，实现在无需重新预训练下有效恢复长上下文性能。**

- **链接: [https://arxiv.org/pdf/2512.10411v1](https://arxiv.org/pdf/2512.10411v1)**

> **作者:** Yijiong Yu; Jiale Liu; Qingyun Wu; Huazheng Wang; Ji Pei
>
> **摘要:** The self-attention mechanism in Transformer-based Large Language Models (LLMs) scales quadratically with input length, making long-context inference expensive. Sliding window attention (SWA) reduces this cost to linear complexity, but naively enabling complete SWA at inference-time for models pretrained with full attention (FA) causes severe long-context performance degradation due to training-inference mismatch. This makes us wonder: Can FA-pretrained LLMs be well adapted to SWA without pretraining? We investigate this by proposing Sliding Window Attention Adaptation (SWAA), a set of practical recipes that combine five methods for better adaptation: (1) applying SWA only during prefilling; (2) preserving "sink" tokens; (3) interleaving FA/SWA layers; (4) chain-of-thought (CoT); and (5) fine-tuning. Our experiments show that SWA adaptation is feasible while non-trivial: no single method suffices, yet specific synergistic combinations effectively recover the original long-context performance. We further analyze the performance-efficiency trade-offs of different SWAA configurations and provide recommended recipes for diverse scenarios. Our code is available at https://github.com/yuyijiong/sliding-window-attention-adaptation
>
---
#### [new 018] RoleRMBench & RoleRM: Towards Reward Modeling for Profile-Based Role Play in Dialogue Systems
- **分类: cs.CL**

- **简介: 该论文聚焦角色扮演对话中的奖励建模任务，旨在解决现有模型难以捕捉主观、人格化人类判断的问题。作者构建了首个基准RoleRMBench，并提出新模型RoleRM，采用连续隐式偏好训练，在叙事连贯性和风格保真上显著超越现有方法。**

- **链接: [https://arxiv.org/pdf/2512.10575v1](https://arxiv.org/pdf/2512.10575v1)**

> **作者:** Hang Ding; Qiming Feng; Dongqi Liu; Qi Zhao; Tao Yao; Shuo Wang; Dongsheng Chen; Jian Li; Zhenye Gan; Jiangning Zhang; Chengjie Wang; Yabiao Wang
>
> **摘要:** Reward modeling has become a cornerstone of aligning large language models (LLMs) with human preferences. Yet, when extended to subjective and open-ended domains such as role play, existing reward models exhibit severe degradation, struggling to capture nuanced and persona-grounded human judgments. To address this gap, we introduce RoleRMBench, the first systematic benchmark for reward modeling in role-playing dialogue, covering seven fine-grained capabilities from narrative management to role consistency and engagement. Evaluation on RoleRMBench reveals large and consistent gaps between general-purpose reward models and human judgment, particularly in narrative and stylistic dimensions. We further propose RoleRM, a reward model trained with Continuous Implicit Preferences (CIP), which reformulates subjective evaluation as continuous consistent pairwise supervision under multiple structuring strategies. Comprehensive experiments show that RoleRM surpasses strong open- and closed-source reward models by over 24% on average, demonstrating substantial gains in narrative coherence and stylistic fidelity. Our findings highlight the importance of continuous preference representation and annotation consistency, establishing a foundation for subjective alignment in human-centered dialogue systems.
>
---
#### [new 019] OPV: Outcome-based Process Verifier for Efficient Long Chain-of-Thought Verification
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于推理验证任务，旨在解决长思维链中中间步骤验证难的问题。提出 outcome-based process verifier（OPV），结合结果与过程验证，通过迭代主动学习降低标注成本，提升验证准确率与效率。**

- **链接: [https://arxiv.org/pdf/2512.10756v1](https://arxiv.org/pdf/2512.10756v1)**

> **作者:** Zijian Wu; Lingkai Kong; Wenwei Zhang; Songyang Gao; Yuzhe Gu; Zhongrui Cai; Tianyou Ma; Yuhong Liu; Zhi Wang; Runyuan Ma; Guangyu Wang; Wei Li; Conghui He; Dahua Lin; Kai Chen
>
> **摘要:** Large language models (LLMs) have achieved significant progress in solving complex reasoning tasks by Reinforcement Learning with Verifiable Rewards (RLVR). This advancement is also inseparable from the oversight automated by reliable verifiers. However, current outcome-based verifiers (OVs) are unable to inspect the unreliable intermediate steps in the long reasoning chains of thought (CoTs). Meanwhile, current process-based verifiers (PVs) have difficulties in reliably detecting errors in the complex long CoTs, limited by the scarcity of high-quality annotations due to the prohibitive costs of human annotations. Therefore, we propose the Outcome-based Process Verifier (OPV), which verifies the rationale process of summarized outcomes from long CoTs to achieve both accurate and efficient verification and enable large-scale annotation. To empower the proposed verifier, we adopt an iterative active learning framework with expert annotations to progressively improve the verification capability of OPV with fewer annotation costs. Specifically, in each iteration, the most uncertain cases of the current best OPV are annotated and then subsequently used to train a new OPV through Rejection Fine-Tuning (RFT) and RLVR for the next round. Extensive experiments demonstrate OPV's superior performance and broad applicability. It achieves new state-of-the-art results on our held-out OPV-Bench, outperforming much larger open-source models such as Qwen3-Max-Preview with an F1 score of 83.1 compared to 76.3. Furthermore, OPV effectively detects false positives within synthetic dataset, closely align with expert assessment. When collaborating with policy models, OPV consistently yields performance gains, e.g., raising the accuracy of DeepSeek-R1-Distill-Qwen-32B from 55.2% to 73.3% on AIME2025 as the compute budget scales.
>
---
#### [new 020] Long-horizon Reasoning Agent for Olympiad-Level Mathematical Problem Solving
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究数学推理任务，旨在解决长链推理中验证难的问题。提出结果导向的过程验证器（OPV），结合主动学习与专家标注，提升验证准确性和标注效率，显著增强大模型在奥数级数学问题上的求解能力。**

- **链接: [https://arxiv.org/pdf/2512.10739v1](https://arxiv.org/pdf/2512.10739v1)**

> **作者:** Songyang Gao; Yuzhe Gu; Zijian Wu; Lingkai Kong; Wenwei Zhang; Zhongrui Cai; Fan Zheng; Tianyou Ma; Junhao Shen; Haiteng Zhao; Duanyang Zhang; Huilun Zhang; Kuikun Liu; Chengqi Lyu; Yanhui Duan; Chiyu Chen; Ningsheng Ma; Jianfei Gao; Han Lyu; Dahua Lin; Kai Chen
>
> **摘要:** Large language models (LLMs) have achieved significant progress in solving complex reasoning tasks by Reinforcement Learning with Verifiable Rewards (RLVR). This advancement is also inseparable from the oversight automated by reliable verifiers. However, current outcome-based verifiers (OVs) are unable to inspect the unreliable intermediate steps in the long reasoning chains of thought (CoTs). Meanwhile, current process-based verifiers (PVs) have difficulties in reliably detecting errors in the complex long CoTs, limited by the scarcity of high-quality annotations due to the prohibitive costs of human annotations. Therefore, we propose the \textbf{O}utcome-based \textbf{P}rocess \textbf{V}erifier (OPV), which verifies the rationale process of summarized outcomes from long CoTs to achieve both accurate and efficient verification and enable large-scale annotation. To empower the proposed verifier, we adopt an iterative active learning framework with expert annotations to progressively improve the verification capability of OPV with fewer annotation costs. Specifically, in each iteration, the most uncertain cases of the current best OPV are annotated and then subsequently used to train a new OPV through Rejection Fine-Tuning (RFT) and RLVR for the next round. Extensive experiments demonstrate OPV's superior performance and broad applicability. It achieves new state-of-the-art results on our held-out \textsc{\thisbench}, outperforming much larger open-source models such as Qwen3-Max-Preview with an F1 score of 83.1 compared to 76.3. Furthermore, OPV effectively detects false positives within synthetic dataset, closely align with expert assessment. When collaborating with policy models, OPV consistently yields performance gains, e.g., raising the accuracy of DeepSeek-R1-Distill-Qwen-32B from 55.2\% to 73.3\% on AIME2025 as the compute budget scales.
>
---
#### [new 021] LabelFusion: Learning to Fuse LLMs and Transformer Classifiers for Robust Text Classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦文本分类任务，旨在融合传统Transformer模型与大语言模型（LLM）的优势，提升分类鲁棒性并兼顾成本。提出LabelFusion框架，通过拼接嵌入与LLM生成的类别得分，经可学习的多层感知机融合输出，实现高精度、低成本的多类与多标签分类。**

- **链接: [https://arxiv.org/pdf/2512.10793v1](https://arxiv.org/pdf/2512.10793v1)**

> **作者:** Michael Schlee; Christoph Weisser; Timo Kivimäki; Melchizedek Mashiku; Benjamin Saefken
>
> **摘要:** LabelFusion is a fusion ensemble for text classification that learns to combine a traditional transformer-based classifier (e.g., RoBERTa) with one or more Large Language Models (LLMs such as OpenAI GPT, Google Gemini, or DeepSeek) to deliver accurate and cost-aware predictions across multi-class and multi-label tasks. The package provides a simple high-level interface (AutoFusionClassifier) that trains the full pipeline end-to-end with minimal configuration, and a flexible API for advanced users. Under the hood, LabelFusion integrates vector signals from both sources by concatenating the ML backbone's embeddings with the LLM-derived per-class scores -- obtained through structured prompt-engineering strategies -- and feeds this joint representation into a compact multi-layer perceptron (FusionMLP) that produces the final prediction. This learned fusion approach captures complementary strengths of LLM reasoning and traditional transformer-based classifiers, yielding robust performance across domains -- achieving 92.4% accuracy on AG News and 92.3% on 10-class Reuters 21578 topic classification -- while enabling practical trade-offs between accuracy, latency, and cost.
>
---
#### [new 022] AutoMedic: An Automated Evaluation Framework for Clinical Conversational Agents with Medical Dataset Grounding
- **分类: cs.CL; cs.LG; cs.MA**

- **简介: 该论文提出AutoMedic，旨在解决临床对话智能体在动态交互场景中的评估难题。通过将静态医疗问答数据转化为虚拟患者，构建多智能体仿真框架，实现对LLM在准确性、效率、共情和鲁棒性等方面的自动化多维度评估。**

- **链接: [https://arxiv.org/pdf/2512.10195v1](https://arxiv.org/pdf/2512.10195v1)**

> **作者:** Gyutaek Oh; Sangjoon Park; Byung-Hoon Kim
>
> **摘要:** Evaluating large language models (LLMs) has recently emerged as a critical issue for safe and trustworthy application of LLMs in the medical domain. Although a variety of static medical question-answering (QA) benchmarks have been proposed, many aspects remain underexplored, such as the effectiveness of LLMs in generating responses in dynamic, interactive clinical multi-turn conversation situations and the identification of multi-faceted evaluation strategies beyond simple accuracy. However, formally evaluating a dynamic, interactive clinical situation is hindered by its vast combinatorial space of possible patient states and interaction trajectories, making it difficult to standardize and quantitatively measure such scenarios. Here, we introduce AutoMedic, a multi-agent simulation framework that enables automated evaluation of LLMs as clinical conversational agents. AutoMedic transforms off-the-shelf static QA datasets into virtual patient profiles, enabling realistic and clinically grounded multi-turn clinical dialogues between LLM agents. The performance of various clinical conversational agents is then assessed based on our CARE metric, which provides a multi-faceted evaluation standard of clinical conversational accuracy, efficiency/strategy, empathy, and robustness. Our findings, validated by human experts, demonstrate the validity of AutoMedic as an automated evaluation framework for clinical conversational agents, offering practical guidelines for the effective development of LLMs in conversational medical applications.
>
---
#### [new 023] From Data Scarcity to Data Care: Reimagining Language Technologies for Serbian and other Low-Resource Languages
- **分类: cs.CL; cs.CY**

- **简介: 该论文探讨塞尔维亚语等低资源语言在大模型中的边缘化问题，揭示历史与技术因素导致的语言偏见。通过访谈研究，提出“数据关怀”框架，倡导以CARE原则指导语料建设，推动公平、可持续的语言技术发展。**

- **链接: [https://arxiv.org/pdf/2512.10630v1](https://arxiv.org/pdf/2512.10630v1)**

> **作者:** Smiljana Antonijevic Ubois
>
> **摘要:** Large language models are commonly trained on dominant languages like English, and their representation of low resource languages typically reflects cultural and linguistic biases present in the source language materials. Using the Serbian language as a case, this study examines the structural, historical, and sociotechnical factors shaping language technology development for low resource languages in the AI age. Drawing on semi structured interviews with ten scholars and practitioners, including linguists, digital humanists, and AI developers, it traces challenges rooted in historical destruction of Serbian textual heritage, intensified by contemporary issues that drive reductive, engineering first approaches prioritizing functionality over linguistic nuance. These include superficial transliteration, reliance on English-trained models, data bias, and dataset curation lacking cultural specificity. To address these challenges, the study proposes Data Care, a framework grounded in CARE principles (Collective Benefit, Authority to Control, Responsibility, and Ethics), that reframes bias mitigation from a post hoc technical fix to an integral component of corpus design, annotation, and governance, and positions Data Care as a replicable model for building inclusive, sustainable, and culturally grounded language technologies in contexts where traditional LLM development reproduces existing power imbalances and cultural blind spots.
>
---
#### [new 024] Semantic Reconstruction of Adversarial Plagiarism: A Context-Aware Framework for Detecting and Restoring "Tortured Phrases" in Scientific Literature
- **分类: cs.CL**

- **简介: 该论文属于科学文本中的对抗性抄袭检测任务，旨在识别并恢复被改写工具扭曲的“ tortured phrases”。作者提出SRAP框架，结合领域语言模型与语义检索，实现异常检测与原始术语的语义重建，提升检测与源追溯能力。**

- **链接: [https://arxiv.org/pdf/2512.10435v1](https://arxiv.org/pdf/2512.10435v1)**

> **作者:** Agniva Maiti; Prajwal Panth; Suresh Chandra Satapathy
>
> **备注:** 10 pages, 5 figures; unpublished manuscript; submitted to arXiv for dissemination
>
> **摘要:** The integrity and reliability of scientific literature is facing a serious threat by adversarial text generation techniques, specifically from the use of automated paraphrasing tools to mask plagiarism. These tools generate "tortured phrases", statistically improbable synonyms (e.g. "counterfeit consciousness" for "artificial intelligence"), that preserve the local grammar while obscuring the original source. Most existing detection methods depend heavily on static blocklists or general-domain language models, which suffer from high false-negative rates for novel obfuscations and cannot determine the source of the plagiarized content. In this paper, we propose Semantic Reconstruction of Adversarial Plagiarism (SRAP), a framework designed not only to detect these anomalies but to mathematically recover the original terminology. We use a two-stage architecture: (1) statistical anomaly detection with a domain-specific masked language model (SciBERT) using token-level pseudo-perplexity, and (2) source-based semantic reconstruction using dense vector retrieval (FAISS) and sentence-level alignment (SBERT). Experiments on a parallel corpus of adversarial scientific text show that while zero-shot baselines fail completely (0.00 percent restoration accuracy), our retrieval-augmented approach achieves 23.67 percent restoration accuracy, significantly outperforming baseline methods. We also show that static decision boundaries are necessary for robust detection in jargon-heavy scientific text, since dynamic thresholding fails under high variance. SRAP enables forensic analysis by linking obfuscated expressions back to their most probable source documents.
>
---
#### [new 025] Quantifying Emotional Tone in Tolkien's The Hobbit: Dialogue Sentiment Analysis with RegEx, NRC-VAD, and Python
- **分类: cs.CL**

- **简介: 该论文属计算文学研究，旨在分析《霍比特人》对话的情感基调。通过正则表达式提取对话，结合NRC-VAD词典与Python进行情感评分，揭示文本中情感的动态变化，展现故事张力与舒缓的循环节奏。**

- **链接: [https://arxiv.org/pdf/2512.10865v1](https://arxiv.org/pdf/2512.10865v1)**

> **作者:** Lilin Qiu
>
> **摘要:** This study analyzes the emotional tone of dialogue in J. R. R. Tolkien's The Hobbit (1937) using computational text analysis. Dialogue was extracted with regular expressions, then preprocessed, and scored using the NRC-VAD lexicon to quantify emotional dimensions. The results show that the dialogue maintains a generally positive (high valence) and calm (low arousal) tone, with a gradually increasing sense of agency (dominance) as the story progresses. These patterns reflect the novel's emotional rhythm: moments of danger and excitement are regularly balanced by humor, camaraderie, and relief. Visualizations -- including emotional trajectory graphs and word clouds -- highlight how Tolkien's language cycles between tension and comfort. By combining computational tools with literary interpretation, this study demonstrates how digital methods can uncover subtle emotional structures in literature, revealing the steady rhythm and emotional modulation that shape the storytelling in The Hobbit.
>
---
#### [new 026] Grammaticality Judgments in Humans and Language Models: Revisiting Generative Grammar with LLMs
- **分类: cs.CL**

- **简介: 该论文探究大语言模型是否能基于表面形式学习句法结构。通过测试GPT-4和LLaMA-3对主语-助动词倒装和寄生空位的可接受性判断，发现模型能区分语法性差异，表明其在无显式语法输入下仍能敏感于句法结构。**

- **链接: [https://arxiv.org/pdf/2512.10453v1](https://arxiv.org/pdf/2512.10453v1)**

> **作者:** Lars G. B. Johnsen
>
> **备注:** 2 figures
>
> **摘要:** What counts as evidence for syntactic structure? In traditional generative grammar, systematic contrasts in grammaticality such as subject-auxiliary inversion and the licensing of parasitic gaps are taken as evidence for an internal, hierarchical grammar. In this paper, we test whether large language models (LLMs), trained only on surface forms, reproduce these contrasts in ways that imply an underlying structural representation. We focus on two classic constructions: subject-auxiliary inversion (testing recognition of the subject boundary) and parasitic gap licensing (testing abstract dependency structure). We evaluate models including GPT-4 and LLaMA-3 using prompts eliciting acceptability ratings. Results show that LLMs reliably distinguish between grammatical and ungrammatical variants in both constructions, and as such support that they are sensitive to structure and not just linear order. Structural generalizations, distinct from cognitive knowledge, emerge from predictive training on surface forms, suggesting functional sensitivity to syntax without explicit encoding.
>
---
#### [new 027] Confucius Code Agent: An Open-sourced AI Software Engineer at Industrial Scale
- **分类: cs.CL; cs.AI; cs.LG; cs.SE**

- **简介: 该论文提出Confucius Code Agent（CCA），旨在解决开源AI编码代理在工业级软件工程任务中性能不足的问题。基于Confucius SDK，构建具备长时记忆、持续学习与强工具协同能力的开源AI工程师，在SWE-Bench-Pro上达到54.3%的SOTA性能。**

- **链接: [https://arxiv.org/pdf/2512.10398v1](https://arxiv.org/pdf/2512.10398v1)**

> **作者:** Zhaodong Wang; Zhenting Qi; Sherman Wong; Nathan Hu; Samuel Lin; Jun Ge; Erwin Gao; Yining Yang; Ben Maurer; Wenlin Chen; David Recordon; Yilun Du; Minlan Yu; Ying Zhang
>
> **摘要:** Real-world AI software engineering demands coding agents that can reason over massive repositories, maintain durable memory across and within long sessions, and robustly coordinate complex toolchains at test time. Existing open-source coding agents provide transparency but frequently fall short when pushed to these industrial-scale workloads, while proprietary coding agents offer strong practical performance but limited extensibility, interpretability, and controllability. We present the Confucius Code Agent (CCA), an open-sourced AI software engineer that can operate at an industrial scale. CCA is built atop the Confucius SDK, an open-sourced agent development platform designed around three complementary perspectives: Agent Experience (AX), User Experience (UX), and Developer Experience (DX). The SDK introduces a unified orchestrator with hierarchical working memory for long-context reasoning, a persistent note-taking system for cross-session continual learning, and a modular extension module for robust tool use. Moreover, a meta-agent automates the synthesis, evaluation, and refinement of agent configurations through a build-test-improve loop, enabling rapid agent development on new tasks, environments, and tool stacks. Instantiated on Confucius SDK with these mechanisms, CCA delivers strong performance on real-world software engineering tasks. On SWE-Bench-Pro, CCA achieves a state-of-the-art Resolve@1 performance of 54.3%, substantially improving over prior coding agents. Together, the Confucius SDK and CCA provide a transparent, extensible, and reproducible foundation for AI agents, bridge gaps between research prototypes and production-grade systems, and support agent development and deployment at industrial scale.
>
---
#### [new 028] Script Gap: Evaluating LLM Triage on Indian Languages in Native vs Roman Scripts in a Real World Setting
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型在印度本土语言不同书写形式（原生文字 vs 罗马化）下的临床分诊性能。针对母婴健康领域的真实用户查询，发现模型对罗马化文本的分类效果显著下降，虽能理解语义，但输出易受拼写噪声影响，导致误判风险上升。**

- **链接: [https://arxiv.org/pdf/2512.10780v1](https://arxiv.org/pdf/2512.10780v1)**

> **作者:** Manurag Khullar; Utkarsh Desai; Poorva Malviya; Aman Dalmia; Zheyuan Ryan Shi
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in high-stakes clinical applications in India. In many such settings, speakers of Indian languages frequently communicate using romanized text rather than native scripts, yet existing research rarely evaluates this orthographic variation using real-world data. We investigate how romanization impacts the reliability of LLMs in a critical domain: maternal and newborn healthcare triage. We benchmark leading LLMs on a real-world dataset of user-generated queries spanning five Indian languages and Nepali. Our results reveal consistent degradation in performance for romanized messages, with F1 scores trailing those of native scripts by 5-12 points. At our partner maternal health organization in India, this gap could cause nearly 2 million excess errors in triage. Crucially, this performance gap by scripts is not due to a failure in clinical reasoning. We demonstrate that LLMs often correctly infer the semantic intent of romanized queries. Nevertheless, their final classification outputs remain brittle in the presence of orthographic noise in romanized inputs. Our findings highlight a critical safety blind spot in LLM-based health systems: models that appear to understand romanized input may still fail to act on it reliably.
>
---
#### [new 029] The FACTS Leaderboard: A Comprehensive Benchmark for Large Language Model Factuality
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型评测任务，旨在解决模型生成文本的事实准确性评估问题。作者提出FACTS Leaderboard，包含四个子榜单，分别评估多模态、知识记忆、检索增强和文档 grounding 场景下的事实性，采用自动化裁判模型打分，综合衡量模型事实一致性。**

- **链接: [https://arxiv.org/pdf/2512.10791v1](https://arxiv.org/pdf/2512.10791v1)**

> **作者:** Aileen Cheng; Alon Jacovi; Amir Globerson; Ben Golan; Charles Kwong; Chris Alberti; Connie Tao; Eyal Ben-David; Gaurav Singh Tomar; Lukas Haas; Yonatan Bitton; Adam Bloniarz; Aijun Bai; Andrew Wang; Anfal Siddiqui; Arturo Bajuelos Castillo; Aviel Atias; Chang Liu; Corey Fry; Daniel Balle; Deepanway Ghosal; Doron Kukliansky; Dror Marcus; Elena Gribovskaya; Eran Ofek; Honglei Zhuang; Itay Laish; Jan Ackermann; Lily Wang; Meg Risdal; Megan Barnes; Michael Fink; Mohamed Amin; Moran Ambar; Natan Potikha; Nikita Gupta; Nitzan Katz; Noam Velan; Ofir Roval; Ori Ram; Polina Zablotskaia; Prathamesh Bang; Priyanka Agrawal; Rakesh Ghiya; Sanjay Ganapathy; Simon Baumgartner; Sofia Erell; Sushant Prakash; Thibault Sellam; Vikram Rao; Xuanhui Wang; Yaroslav Akulov; Yulong Yang; Zhen Yang; Zhixin Lai; Zhongru Wu; Anca Dragan; Avinatan Hassidim; Fernando Pereira; Slav Petrov; Srinivasan Venkatachary; Tulsee Doshi; Yossi Matias; Sasha Goldshtein; Dipanjan Das
>
> **摘要:** We introduce The FACTS Leaderboard, an online leaderboard suite and associated set of benchmarks that comprehensively evaluates the ability of language models to generate factually accurate text across diverse scenarios. The suite provides a holistic measure of factuality by aggregating the performance of models on four distinct sub-leaderboards: (1) FACTS Multimodal, which measures the factuality of responses to image-based questions; (2) FACTS Parametric, which assesses models' world knowledge by answering closed-book factoid questions from internal parameters; (3) FACTS Search, which evaluates factuality in information-seeking scenarios, where the model must use a search API; and (4) FACTS Grounding (v2), which evaluates whether long-form responses are grounded in provided documents, featuring significantly improved judge models. Each sub-leaderboard employs automated judge models to score model responses, and the final suite score is an average of the four components, designed to provide a robust and balanced assessment of a model's overall factuality. The FACTS Leaderboard Suite will be actively maintained, containing both public and private splits to allow for external participation while guarding its integrity. It can be found at https://www.kaggle.com/benchmarks/google/facts .
>
---
#### [new 030] Causal Reasoning Favors Encoders: On The Limits of Decoder-Only Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究因果推理任务中不同模型架构的性能差异，探讨编码器、编码器-解码器与纯解码器模型在多步因果推理中的表现。通过对比微调和上下文学习效果，发现编码器类模型更鲁棒，尤其在分布偏移和非自然语言场景下优于纯解码器模型。**

- **链接: [https://arxiv.org/pdf/2512.10561v1](https://arxiv.org/pdf/2512.10561v1)**

> **作者:** Amartya Roy; Elamparithy M; Kripabandhu Ghosh; Ponnurangam Kumaraguru; Adrian de Wynter
>
> **摘要:** In context learning (ICL) underpins recent advances in large language models (LLMs), although its role and performance in causal reasoning remains unclear. Causal reasoning demands multihop composition and strict conjunctive control, and reliance on spurious lexical relations of the input could provide misleading results. We hypothesize that, due to their ability to project the input into a latent space, encoder and encoder decoder architectures are better suited for said multihop conjunctive reasoning versus decoder only models. To do this, we compare fine-tuned versions of all the aforementioned architectures with zero and few shot ICL in both natural language and non natural language scenarios. We find that ICL alone is insufficient for reliable causal reasoning, often overfocusing on irrelevant input features. In particular, decoder only models are noticeably brittle to distributional shifts, while finetuned encoder and encoder decoder models can generalize more robustly across our tests, including the non natural language split. Both architectures are only matched or surpassed by decoder only architectures at large scales. We conclude by noting that for cost effective, short horizon robust causal reasoning, encoder or encoder decoder architectures with targeted finetuning are preferable.
>
---
#### [new 031] GPG: Generalized Policy Gradient Theorem for Transformer-based Policies
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出广义策略梯度（GPG）定理，专为基于Transformer的策略设计。它统一了标准策略梯度与GRPO，属于强化学习中策略优化任务，旨在提升大语言模型训练效率。**

- **链接: [https://arxiv.org/pdf/2512.10365v1](https://arxiv.org/pdf/2512.10365v1)**

> **作者:** Hangyu Mao; Guangting Dong; Zhicheng Dou
>
> **摘要:** We present the Generalized Policy Gradient (GPG) Theorem, specifically designed for Transformer-based policies. Notably, we demonstrate that both standard Policy Gradient Theorem and GRPO emerge as special cases within our GPG framework. Furthermore, we explore its practical applications in training Large Language Models (LLMs), offering new insights into efficient policy optimization.
>
---
#### [new 032] Parallel Decoder Transformer: Model-Internal Parallel Decoding with Speculative Invariance via Note Conditioning
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对大语言模型自回归生成的序列瓶颈，提出并行解码架构PDT。通过在冻结模型中引入轻量级适配器，实现多流并行生成与语义同步，解决传统方法的连贯性漂移问题，无需微调即可高效生成结构化文本。**

- **链接: [https://arxiv.org/pdf/2512.10054v1](https://arxiv.org/pdf/2512.10054v1)**

> **作者:** Logan Robbins
>
> **摘要:** Autoregressive decoding in Large Language Models (LLMs) is inherently sequential, creating a latency bottleneck that scales linearly with output length. While ``Decomposition-and-Fill'' methods like Skeleton-of-Thought attempt to parallelize generation via external orchestration, they suffer from \textit{coherence drift} due to the lack of cross-stream communication. In this work, we introduce the \textbf{Parallel Decoder Transformer (PDT)}, a parameter-efficient architecture that embeds coordination primitives directly into the inference process of a frozen pre-trained model. Instead of retraining the base model, PDT injects lightweight \textit{Speculative Note Conditioning (SNC)} adapters that allow parallel decoding streams to synchronize via a shared, dynamic latent space. We formulate coordination as a \textit{speculative consensus} problem, where sibling streams broadcast semantic ``notes'' to a global bus, gated by a learned verification head. We validate our approach on a 50,000-step curriculum using a frozen 20B-parameter backbone. Our results demonstrate that PDT achieves effective self-correction, reaching \textbf{77.8\% precision} in coverage prediction and recovering approximate serial semantics without modifying the trunk weights. This establishes PDT as a scalable, efficient alternative to full model fine-tuning for structured parallel generation.
>
---
#### [new 033] MotionEdit: Benchmarking and Learning Motion-Centric Image Editing
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究运动中心的图像编辑任务，旨在修改主体动作并保持身份与物理合理性。提出MotionEdit数据集和MotionEdit-Bench评测基准，发现现有模型表现不佳，进而提出MotionNFT微调框架，提升运动编辑质量与保真度。**

- **链接: [https://arxiv.org/pdf/2512.10284v1](https://arxiv.org/pdf/2512.10284v1)**

> **作者:** Yixin Wan; Lei Ke; Wenhao Yu; Kai-Wei Chang; Dong Yu
>
> **摘要:** We introduce MotionEdit, a novel dataset for motion-centric image editing-the task of modifying subject actions and interactions while preserving identity, structure, and physical plausibility. Unlike existing image editing datasets that focus on static appearance changes or contain only sparse, low-quality motion edits, MotionEdit provides high-fidelity image pairs depicting realistic motion transformations extracted and verified from continuous videos. This new task is not only scientifically challenging but also practically significant, powering downstream applications such as frame-controlled video synthesis and animation. To evaluate model performance on the novel task, we introduce MotionEdit-Bench, a benchmark that challenges models on motion-centric edits and measures model performance with generative, discriminative, and preference-based metrics. Benchmark results reveal that motion editing remains highly challenging for existing state-of-the-art diffusion-based editing models. To address this gap, we propose MotionNFT (Motion-guided Negative-aware Fine Tuning), a post-training framework that computes motion alignment rewards based on how well the motion flow between input and model-edited images matches the ground-truth motion, guiding models toward accurate motion transformations. Extensive experiments on FLUX.1 Kontext and Qwen-Image-Edit show that MotionNFT consistently improves editing quality and motion fidelity of both base models on the motion editing task without sacrificing general editing ability, demonstrating its effectiveness.
>
---
#### [new 034] Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven Agent Evolution
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究LLM代理的经验驱动进化，解决现有记忆系统静态、被动的问题。提出ReMe框架，通过多维度蒸馏、上下文自适应复用和效用化精炼，实现动态记忆管理，提升代理性能与效率。**

- **链接: [https://arxiv.org/pdf/2512.10696v1](https://arxiv.org/pdf/2512.10696v1)**

> **作者:** Zouying Cao; Jiaji Deng; Li Yu; Weikang Zhou; Zhaoyang Liu; Bolin Ding; Hai Zhao
>
> **备注:** 16 pages, 9 figures, 9 tables
>
> **摘要:** Procedural memory enables large language model (LLM) agents to internalize "how-to" knowledge, theoretically reducing redundant trial-and-error. However, existing frameworks predominantly suffer from a "passive accumulation" paradigm, treating memory as a static append-only archive. To bridge the gap between static storage and dynamic reasoning, we propose $\textbf{ReMe}$ ($\textit{Remember Me, Refine Me}$), a comprehensive framework for experience-driven agent evolution. ReMe innovates across the memory lifecycle via three mechanisms: 1) $\textit{multi-faceted distillation}$, which extracts fine-grained experiences by recognizing success patterns, analyzing failure triggers and generating comparative insights; 2) $\textit{context-adaptive reuse}$, which tailors historical insights to new contexts via scenario-aware indexing; and 3) $\textit{utility-based refinement}$, which autonomously adds valid memories and prunes outdated ones to maintain a compact, high-quality experience pool. Extensive experiments on BFCL-V3 and AppWorld demonstrate that ReMe establishes a new state-of-the-art in agent memory system. Crucially, we observe a significant memory-scaling effect: Qwen3-8B equipped with ReMe outperforms larger, memoryless Qwen3-14B, suggesting that self-evolving memory provides a computation-efficient pathway for lifelong learning. We release our code and the $\texttt{reme.library}$ dataset to facilitate further research.
>
---
#### [new 035] BAMBO: Construct Ability and Efficiency LLM Pareto Set via Bayesian Adaptive Multi-objective Block-wise Optimization
- **分类: cs.LG; cs.CL; cs.NE**

- **简介: 该论文属多目标优化任务，旨在解决大模型能力与效率权衡的Pareto集构建问题。提出BAMBO框架，通过块级自适应划分与贝叶斯优化，自动化高效搜索最优解集，提升性能与效率平衡。**

- **链接: [https://arxiv.org/pdf/2512.09972v1](https://arxiv.org/pdf/2512.09972v1)**

> **作者:** Kesheng Chen; Wenjian Luo; Zhenqian Zhu; Yamin Hu; Yiya Xi
>
> **摘要:** Constructing a Pareto set is pivotal for navigating the capability-efficiency trade-offs in Large Language Models (LLMs); however, existing merging techniques remain inadequate for this task. Coarse-grained, model-level methods yield only a sparse set of suboptimal solutions, while fine-grained, layer-wise approaches suffer from the "curse of dimensionality," rendering the search space computationally intractable. To resolve this dichotomy, we propose BAMBO (Bayesian Adaptive Multi-objective Block-wise Optimization), a novel framework that automatically constructs the LLM Pareto set. BAMBO renders the search tractable by introducing a Hybrid Optimal Block Partitioning strategy. Formulated as a 1D clustering problem, this strategy leverages a dynamic programming approach to optimally balance intra-block homogeneity and inter-block information distribution, thereby dramatically reducing dimensionality without sacrificing critical granularity. The entire process is automated within an evolutionary loop driven by the q-Expected Hypervolume Improvement (qEHVI) acquisition function. Experiments demonstrate that BAMBO discovers a superior and more comprehensive Pareto frontier than baselines, enabling agile model selection tailored to diverse operational constraints. Code is available at: https://github.com/xin8coder/BAMBO.
>
---
#### [new 036] CompanionCast: A Multi-Agent Conversational AI Framework with Spatial Audio for Social Co-Viewing Experiences
- **分类: cs.HC; cs.CL**

- **简介: 该论文提出CompanionCast框架，属多智能体对话AI任务，旨在解决单人观看内容时缺乏社交临场感的问题。通过角色化AI代理、空间音频与LLM评分机制，模拟多人共看体验，提升用户社交感知。**

- **链接: [https://arxiv.org/pdf/2512.10918v1](https://arxiv.org/pdf/2512.10918v1)**

> **作者:** Yiyang Wang; Chen Chen; Tica Lin; Vishnu Raj; Josh Kimball; Alex Cabral; Josiah Hester
>
> **备注:** 11 pages
>
> **摘要:** Social presence is central to the enjoyment of watching content together, yet modern media consumption is increasingly solitary. We investigate whether multi-agent conversational AI systems can recreate the dynamics of shared viewing experiences across diverse content types. We present CompanionCast, a general framework for orchestrating multiple role-specialized AI agents that respond to video content using multimodal inputs, speech synthesis, and spatial audio. Distinctly, CompanionCast integrates an LLM-as-a-Judge module that iteratively scores and refines conversations across five dimensions (relevance, authenticity, engagement, diversity, personality consistency). We validate this framework through sports viewing, a domain with rich dynamics and strong social traditions, where a pilot study with soccer fans suggests that multi-agent interaction improves perceived social presence compared to solo viewing. We contribute: (1) a generalizable framework for orchestrating multi-agent conversations around multimodal video content, (2) a novel evaluator-agent pipeline for conversation quality control, and (3) exploratory evidence of increased social presence in AI-mediated co-viewing. We discuss challenges and future directions for applying this approach to diverse viewing contexts including entertainment, education, and collaborative watching experiences.
>
---
#### [new 037] Planning, Living and Judging: A Multi-agent LLM-based Framework for Cyclical Urban Planning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出基于大语言模型的多智能体框架CUP，用于循环城市规划。通过“规划-居住-评判”闭环，实现城市方案的生成、模拟与优化，解决城市更新中动态适应性不足的问题，提升规划的持续性和响应能力。**

- **链接: [https://arxiv.org/pdf/2412.20505v1](https://arxiv.org/pdf/2412.20505v1)**

> **作者:** Hang Ni; Yuzhi Wang; Hao Liu
>
> **备注:** 4 pages, 2 figures, accepted by The 1st Workshop on AI for Urban Planning (AAAI 2025's Workshop)
>
> **摘要:** Urban regeneration presents significant challenges within the context of urbanization, requiring adaptive approaches to tackle evolving needs. Leveraging advancements in large language models (LLMs), we propose Cyclical Urban Planning (CUP), a new paradigm that continuously generates, evaluates, and refines urban plans in a closed-loop. Specifically, our multi-agent LLM-based framework consists of three key components: (1) Planning, where LLM agents generate and refine urban plans based on contextual data; (2) Living, where agents simulate the behaviors and interactions of residents, modeling life in the urban environment; and (3) Judging, which involves evaluating plan effectiveness and providing iterative feedback for improvement. The cyclical process enables a dynamic and responsive planning approach. Experiments on the real-world dataset demonstrate the effectiveness of our framework as a continuous and adaptive planning process.
>
---
#### [new 038] Are We Ready for RL in Text-to-3D Generation? A Progressive Investigation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究强化学习（RL）在文本到3D生成中的应用，旨在解决3D生成中几何一致性与纹理精细度对奖励设计和算法敏感的问题。作者系统评估了奖励机制、RL算法，提出新基准MME-3DR与分层RL方法Hi-GRPO，并推出首个RL增强的文本到3D模型AR3D-R1。**

- **链接: [https://arxiv.org/pdf/2512.10949v1](https://arxiv.org/pdf/2512.10949v1)**

> **作者:** Yiwen Tang; Zoey Guo; Kaixin Zhu; Ray Zhang; Qizhi Chen; Dongzhi Jiang; Junli Liu; Bohan Zeng; Haoming Song; Delin Qu; Tianyi Bai; Dan Xu; Wentao Zhang; Bin Zhao
>
> **备注:** Code is released at https://github.com/Ivan-Tang-3D/3DGen-R1
>
> **摘要:** Reinforcement learning (RL), earlier proven to be effective in large language and multi-modal models, has been successfully extended to enhance 2D image generation recently. However, applying RL to 3D generation remains largely unexplored due to the higher spatial complexity of 3D objects, which require globally consistent geometry and fine-grained local textures. This makes 3D generation significantly sensitive to reward designs and RL algorithms. To address these challenges, we conduct the first systematic study of RL for text-to-3D autoregressive generation across several dimensions. (1) Reward designs: We evaluate reward dimensions and model choices, showing that alignment with human preference is crucial, and that general multi-modal models provide robust signal for 3D attributes. (2) RL algorithms: We study GRPO variants, highlighting the effectiveness of token-level optimization, and further investigate the scaling of training data and iterations. (3) Text-to-3D Benchmarks: Since existing benchmarks fail to measure implicit reasoning abilities in 3D generation models, we introduce MME-3DR. (4) Advanced RL paradigms: Motivated by the natural hierarchy of 3D generation, we propose Hi-GRPO, which optimizes the global-to-local hierarchical 3D generation through dedicated reward ensembles. Based on these insights, we develop AR3D-R1, the first RL-enhanced text-to-3D model, expert from coarse shape to texture refinement. We hope this study provides insights into RL-driven reasoning for 3D generation. Code is released at https://github.com/Ivan-Tang-3D/3DGen-R1.
>
---
#### [new 039] CIEGAD: Cluster-Conditioned Interpolative and Extrapolative Framework for Geometry-Aware and Domain-Aligned Data Augmentation
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出CIEGAD框架，用于数据增强任务，解决数据稀缺与分布不平衡导致的语义覆盖不全问题。通过聚类条件生成、几何感知插值与外推、域对齐及质量控制，提升模型在类别边界和外围区域的分类性能。**

- **链接: [https://arxiv.org/pdf/2512.10178v1](https://arxiv.org/pdf/2512.10178v1)**

> **作者:** Keito Inoshita; Xiaokang Zhou; Akira Kawai; Katsutoshi Yada
>
> **摘要:** In practical deep learning deployment, the scarcity of data and the imbalance of label distributions often lead to semantically uncovered regions within the real-world data distribution, hindering model training and causing misclassification near class boundaries as well as unstable behaviors in peripheral areas. Although recent large language models (LLMs) show promise for data augmentation, an integrated framework that simultaneously achieves directional control of generation, domain alignment, and quality control has not yet been fully established. To address these challenges, we propose a Cluster-conditioned Interpolative and Extrapolative framework for Geometry-Aware and Domain-aligned data augmentation (CIEGAD), which systematically complements both in-distribution and out-of-distribution semantically uncovered regions. CIEGAD constructs domain profiles through cluster conditioning, allocates generation with a hierarchical frequency-geometric allocation integrating class frequency and geometric indicators, and finely controls generation directions via the coexistence of interpolative and extrapolative synthesis. It further performs quality control through geometry-constrained filtering combined with an LLM-as-a-Judge mechanism. Experiments on multiple classification tasks demonstrate that CIEGAD effectively extends the periphery of real-world data distributions while maintaining high alignment between generated and real-world data as well as semantic diversity. In particular, for long-tailed and multi-class classification tasks, CIEGAD consistently improves F1 and recall, validating the triple harmony of distributional consistency, diversity, and quality. These results indicate that CIEGAD serves as a practically oriented data augmentation framework that complements underrepresented regions while preserving alignment with real-world data.
>
---
#### [new 040] Stronger Normalization-Free Transformers
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究无需归一化层的Transformer模型，旨在寻找优于现有方法（如LayerNorm、DyT）的激活函数。通过大规模搜索，提出Derf函数，其基于误差函数并提升模型泛化能力，在多领域任务中表现更优。**

- **链接: [https://arxiv.org/pdf/2512.10938v1](https://arxiv.org/pdf/2512.10938v1)**

> **作者:** Mingzhi Chen; Taiming Lu; Jiachen Zhu; Mingjie Sun; Zhuang Liu
>
> **摘要:** Although normalization layers have long been viewed as indispensable components of deep learning architectures, the recent introduction of Dynamic Tanh (DyT) has demonstrated that alternatives are possible. The point-wise function DyT constrains extreme values for stable convergence and reaches normalization-level performance; this work seeks further for function designs that can surpass it. We first study how the intrinsic properties of point-wise functions influence training and performance. Building on these findings, we conduct a large-scale search for a more effective function design. Through this exploration, we introduce $\mathrm{Derf}(x) = \mathrm{erf}(αx + s)$, where $\mathrm{erf}(x)$ is the rescaled Gaussian cumulative distribution function, and identify it as the most performant design. Derf outperforms LayerNorm, RMSNorm, and DyT across a wide range of domains, including vision (image recognition and generation), speech representation, and DNA sequence modeling. Our findings suggest that the performance gains of Derf largely stem from its improved generalization rather than stronger fitting capacity. Its simplicity and stronger performance make Derf a practical choice for normalization-free Transformer architectures.
>
---
#### [new 041] Exploring LLMs for Scientific Information Extraction Using The SciEx Framework
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于科学信息抽取任务，旨在解决现有方法在处理长文本、多模态内容和动态数据模式时的局限性。作者提出SciEx框架，模块化解耦解析、检索、抽取与聚合，提升灵活性与可扩展性，并在多领域数据集上验证其有效性。**

- **链接: [https://arxiv.org/pdf/2512.10004v1](https://arxiv.org/pdf/2512.10004v1)**

> **作者:** Sha Li; Ayush Sadekar; Nathan Self; Yiqi Su; Lars Andersland; Mira Chaplin; Annabel Zhang; Hyoju Yang; James B Henderson; Krista Wigginton; Linsey Marr; T. M. Murali; Naren Ramakrishnan
>
> **摘要:** Large language models (LLMs) are increasingly touted as powerful tools for automating scientific information extraction. However, existing methods and tools often struggle with the realities of scientific literature: long-context documents, multi-modal content, and reconciling varied and inconsistent fine-grained information across multiple publications into standardized formats. These challenges are further compounded when the desired data schema or extraction ontology changes rapidly, making it difficult to re-architect or fine-tune existing systems. We present SciEx, a modular and composable framework that decouples key components including PDF parsing, multi-modal retrieval, extraction, and aggregation. This design streamlines on-demand data extraction while enabling extensibility and flexible integration of new models, prompting strategies, and reasoning mechanisms. We evaluate SciEx on datasets spanning three scientific topics for its ability to extract fine-grained information accurately and consistently. Our findings provide practical insights into both the strengths and limitations of current LLM-based pipelines.
>
---
#### [new 042] Replace, Don't Expand: Mitigating Context Dilution in Multi-Hop RAG via Fixed-Budget Evidence Assembly
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于检索增强生成（RAG）任务，旨在解决多跳查询中因检索缺失导致的上下文稀释问题。作者提出SEAL-RAG方法，通过“替换而非扩展”的策略，在固定检索深度下动态替换无关证据，提升答案准确性和证据精度。**

- **链接: [https://arxiv.org/pdf/2512.10787v1](https://arxiv.org/pdf/2512.10787v1)**

> **作者:** Moshe Lahmy; Roi Yozevitch
>
> **备注:** 24 pages, 2 figures
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems often fail on multi-hop queries when the initial retrieval misses a bridge fact. Prior corrective approaches, such as Self-RAG, CRAG, and Adaptive-$k$, typically address this by \textit{adding} more context or pruning existing lists. However, simply expanding the context window often leads to \textbf{context dilution}, where distractors crowd out relevant information. We propose \textbf{SEAL-RAG}, a training-free controller that adopts a \textbf{``replace, don't expand''} strategy to fight context dilution under a fixed retrieval depth $k$. SEAL executes a (\textbf{S}earch $\rightarrow$ \textbf{E}xtract $\rightarrow$ \textbf{A}ssess $\rightarrow$ \textbf{L}oop) cycle: it performs on-the-fly, entity-anchored extraction to build a live \textit{gap specification} (missing entities/relations), triggers targeted micro-queries, and uses \textit{entity-first ranking} to actively swap out distractors for gap-closing evidence. We evaluate SEAL-RAG against faithful re-implementations of Basic RAG, CRAG, Self-RAG, and Adaptive-$k$ in a shared environment on \textbf{HotpotQA} and \textbf{2WikiMultiHopQA}. On HotpotQA ($k=3$), SEAL improves answer correctness by \textbf{+3--13 pp} and evidence precision by \textbf{+12--18 pp} over Self-RAG. On 2WikiMultiHopQA ($k=5$), it outperforms Adaptive-$k$ by \textbf{+8.0 pp} in accuracy and maintains \textbf{96\%} evidence precision compared to 22\% for CRAG. These gains are statistically significant ($p<0.001$). By enforcing fixed-$k$ replacement, SEAL yields a predictable cost profile while ensuring the top-$k$ slots are optimized for precision rather than mere breadth. We release our code and data at https://github.com/mosherino/SEAL-RAG.
>
---
#### [new 043] When Reject Turns into Accept: Quantifying the Vulnerability of LLM-Based Scientific Reviewers to Indirect Prompt Injection
- **分类: cs.AI; cs.CL; cs.CR**

- **简介: 该论文研究LLM在科学评审中被间接提示注入攻击的脆弱性，旨在量化“拒稿变接收”的风险。提出WAVS指标，构建数据集与攻击框架，评估多种模型在15种攻击下的表现，揭示现有系统的安全隐患。**

- **链接: [https://arxiv.org/pdf/2512.10449v1](https://arxiv.org/pdf/2512.10449v1)**

> **作者:** Devanshu Sahoo; Manish Prasad; Vasudev Majhi; Jahnvi Singh; Vinay Chamola; Yash Sinha; Murari Mandal; Dhruv Kumar
>
> **摘要:** The landscape of scientific peer review is rapidly evolving with the integration of Large Language Models (LLMs). This shift is driven by two parallel trends: the widespread individual adoption of LLMs by reviewers to manage workload (the "Lazy Reviewer" hypothesis) and the formal institutional deployment of AI-powered assessment systems by conferences like AAAI and Stanford's Agents4Science. This study investigates the robustness of these "LLM-as-a-Judge" systems (both illicit and sanctioned) to adversarial PDF manipulation. Unlike general jailbreaks, we focus on a distinct incentive: flipping "Reject" decisions to "Accept," for which we develop a novel evaluation metric which we term as WAVS (Weighted Adversarial Vulnerability Score). We curated a dataset of 200 scientific papers and adapted 15 domain-specific attack strategies to this task, evaluating them across 13 Language Models, including GPT-5, Claude Haiku, and DeepSeek. Our results demonstrate that obfuscation strategies like "Maximum Mark Magyk" successfully manipulate scores, achieving alarming decision flip rates even in large-scale models. We will release our complete dataset and injection framework to facilitate more research on this topic.
>
---
#### [new 044] Asynchronous Reasoning: Training-Free Interactive Thinking LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对推理型大模型响应延迟高、交互性差的问题，提出无需训练的异步推理方法，利用旋转变换嵌入使模型能同时思考、接收输入和生成输出，显著降低响应延迟，提升实时交互能力。**

- **链接: [https://arxiv.org/pdf/2512.10931v1](https://arxiv.org/pdf/2512.10931v1)**

> **作者:** George Yakushev; Nataliia Babina; Masoud Vahid Dastgerdi; Vyacheslav Zhdanovskiy; Alina Shutova; Denis Kuznedelev
>
> **备注:** Preprint, work in progress
>
> **摘要:** Many state-of-the-art LLMs are trained to think before giving their answer. Reasoning can greatly improve language model capabilities and safety, but it also makes them less interactive: given a new input, a model must stop thinking before it can respond. Real-world use cases such as voice-based or embedded assistants require an LLM agent to respond and adapt to additional information in real time, which is incompatible with sequential interactions. In contrast, humans can listen, think, and act asynchronously: we begin thinking about the problem while reading it and continue thinking while formulating the answer. In this work, we augment LLMs capable of reasoning to operate in a similar way without additional training. Our method uses the properties of rotary embeddings to enable LLMs built for sequential interactions to simultaneously think, listen, and generate outputs. We evaluate our approach on math, commonsense, and safety reasoning and find that it can generate accurate thinking-augmented answers in real time, reducing time to first non-thinking token from minutes to <= 5s. and the overall real-time delays by 6-11x.
>
---
#### [new 045] Offscript: Automated Auditing of Instruction Adherence in LLMs
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于指令遵循评估任务，旨在解决大语言模型（LLM）是否有效遵循用户自定义指令的问题。作者提出Offscript，一种自动化审计工具，用于检测指令遵循失败，并通过实验证明其有效性。**

- **链接: [https://arxiv.org/pdf/2512.10172v1](https://arxiv.org/pdf/2512.10172v1)**

> **作者:** Nicholas Clark; Ryan Bai; Tanu Mitra
>
> **摘要:** Large Language Models (LLMs) and generative search systems are increasingly used for information seeking by diverse populations with varying preferences for knowledge sourcing and presentation. While users can customize LLM behavior through custom instructions and behavioral prompts, no mechanism exists to evaluate whether these instructions are being followed effectively. We present Offscript, an automated auditing tool that efficiently identifies potential instruction following failures in LLMs. In a pilot study analyzing custom instructions sourced from Reddit, Offscript detected potential deviations from instructed behavior in 86.4% of conversations, 22.2% of which were confirmed as material violations through human review. Our findings suggest that automated auditing serves as a viable approach for evaluating compliance to behavioral instructions related to information seeking.
>
---
#### [new 046] Diffusion Is Your Friend in Show, Suggest and Tell
- **分类: cs.CV; cs.CL**

- **简介: 该论文聚焦图像描述生成任务，旨在解决扩散模型在离散文本生成中不及自回归模型的问题。提出“展示、建议、告知”（SST）框架，利用扩散模型为自回归生成提供建议，结合二者优势，显著提升生成质量，在COCO数据集上取得最优性能。**

- **链接: [https://arxiv.org/pdf/2512.10038v1](https://arxiv.org/pdf/2512.10038v1)**

> **作者:** Jia Cheng Hu; Roberto Cavicchioli; Alessandro Capotondi
>
> **摘要:** Diffusion Denoising models demonstrated impressive results across generative Computer Vision tasks, but they still fail to outperform standard autoregressive solutions in the discrete domain, and only match them at best. In this work, we propose a different paradigm by adopting diffusion models to provide suggestions to the autoregressive generation rather than replacing them. By doing so, we combine the bidirectional and refining capabilities of the former with the strong linguistic structure provided by the latter. To showcase its effectiveness, we present Show, Suggest and Tell (SST), which achieves State-of-the-Art results on COCO, among models in a similar setting. In particular, SST achieves 125.1 CIDEr-D on the COCO dataset without Reinforcement Learning, outperforming both autoregressive and diffusion model State-of-the-Art results by 1.5 and 2.5 points. On top of the strong results, we performed extensive experiments to validate the proposal and analyze the impact of the suggestion module. Results demonstrate a positive correlation between suggestion and caption quality, overall indicating a currently underexplored but promising research direction. Code will be available at: https://github.com/jchenghu/show\_suggest\_tell.
>
---
#### [new 047] BRACE: A Benchmark for Robust Audio Caption Quality Evaluation
- **分类: cs.SD; cs.CL**

- **简介: 该论文聚焦音频描述质量评估任务，旨在解决无参考文本时的评估难题。作者提出BRACE基准，包含两个子任务，用于评测音频描述对齐与幻觉检测，通过实验揭示现有CLAPScore与大模型的局限性。**

- **链接: [https://arxiv.org/pdf/2512.10403v1](https://arxiv.org/pdf/2512.10403v1)**

> **作者:** Tianyu Guo; Hongyu Chen; Hao Liang; Meiyi Qiang; Bohan Zeng; Linzhuang Sun; Bin Cui; Wentao Zhang
>
> **摘要:** Automatic audio captioning is essential for audio understanding, enabling applications such as accessibility and content indexing. However, evaluating the quality of audio captions remains a major challenge, especially in reference-free settings where high-quality ground-truth captions are unavailable. While CLAPScore is currently the most widely used reference-free Audio Caption Evaluation Metric(ACEM), its robustness under diverse conditions has not been systematically validated. To address this gap, we introduce BRACE, a new benchmark designed to evaluate audio caption alignment quality in a reference-free setting. BRACE is primarily designed for assessing ACEMs, and can also be extended to measure the modality alignment abilities of Large Audio Language Model(LALM). BRACE consists of two sub-benchmarks: BRACE-Main for fine-grained caption comparison and BRACE-Hallucination for detecting subtle hallucinated content. We construct these datasets through high-quality filtering, LLM-based corruption, and human annotation. Given the widespread adoption of CLAPScore as a reference-free ACEM and the increasing application of LALMs in audio-language tasks, we evaluate both approaches using the BRACE benchmark, testing CLAPScore across various CLAP model variants and assessing multiple LALMs. Notably, even the best-performing CLAP-based ACEM achieves only a 70.01 F1-score on the BRACE-Main benchmark, while the best LALM reaches just 63.19. By revealing the limitations of CLAP models and LALMs, our BRACE benchmark offers valuable insights into the direction of future research.
>
---
#### [new 048] Watermarks for Language Models via Probabilistic Automata
- **分类: cs.CR; cs.CL**

- **简介: 该论文研究语言模型水印技术，旨在解决现有方法生成多样性低、检测开销大及易被探测的问题。提出基于概率自动机的新水印方案，兼具高生成多样性、高效性，并实现形式化不可探测性保证。**

- **链接: [https://arxiv.org/pdf/2512.10185v1](https://arxiv.org/pdf/2512.10185v1)**

> **作者:** Yangkun Wang; Jingbo Shang
>
> **摘要:** A recent watermarking scheme for language models achieves distortion-free embedding and robustness to edit-distance attacks. However, it suffers from limited generation diversity and high detection overhead. In parallel, recent research has focused on undetectability, a property ensuring that watermarks remain difficult for adversaries to detect and spoof. In this work, we introduce a new class of watermarking schemes constructed through probabilistic automata. We present two instantiations: (i) a practical scheme with exponential generation diversity and computational efficiency, and (ii) a theoretical construction with formal undetectability guarantees under cryptographic assumptions. Extensive experiments on LLaMA-3B and Mistral-7B validate the superior performance of our scheme in terms of robustness and efficiency.
>
---
## 更新

#### [replaced 001] A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents
- **分类: cs.CL**

- **简介: 该论文针对大模型代理在长时对话中记忆保持困难的问题，提出一种基于事件语义的结构化记忆方法。将对话分解为带实体和时间信息的单元，构建异构图实现高效检索与关联，提升长期记忆性能。**

- **链接: [https://arxiv.org/pdf/2511.17208v2](https://arxiv.org/pdf/2511.17208v2)**

> **作者:** Sizhe Zhou; Jiawei Han
>
> **备注:** Work in progress
>
> **摘要:** LLM-based conversational agents still struggle to maintain coherent, personalized interaction over many sessions: fixed context windows limit how much history can be kept in view, and most external memory approaches trade off between coarse retrieval over large chunks and fine-grained but fragmented views of the dialogue. Motivated by neo-Davidsonian event semantics, we propose an event-centric alternative that represents conversational history as short, event-like propositions which bundle together participants, temporal cues, and minimal local context, rather than as independent relation triples or opaque summaries. In contrast to work that aggressively compresses or forgets past content, our design aims to preserve information in a non-compressive form and make it more accessible, rather than more lossy. Concretely, we instruct an LLM to decompose each session into enriched elementary discourse units (EDUs) -- self-contained statements with normalized entities and source turn attributions -- and organize sessions, EDUs, and their arguments in a heterogeneous graph that supports associative recall. On top of this representation we build two simple retrieval-based variants that use dense similarity search and LLM filtering, with an optional graph-based propagation step to connect and aggregate evidence across related EDUs. Experiments on the LoCoMo and LongMemEval$_S$ benchmarks show that these event-centric memories match or surpass strong baselines, while operating with much shorter QA contexts. Our results suggest that structurally simple, event-level memory provides a principled and practical foundation for long-horizon conversational agents. Our code and data will be released at https://github.com/KevinSRR/EMem.
>
---
#### [replaced 002] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型推理中的KL正则策略梯度算法设计，旨在统一并修正现有方法在离策略场景下的梯度偏差问题。提出RPG框架，明确不同KL变体的权重选择，修正GRPO错误，并引入RPG-Style Clip提升训练稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2505.17508v3](https://arxiv.org/pdf/2505.17508v3)**

> **作者:** Yifan Zhang; Yifeng Liu; Huizhuo Yuan; Yang Yuan; Quanquan Gu; Andrew Chi-Chih Yao
>
> **备注:** Project Page: https://github.com/complex-reasoning/RPG
>
> **摘要:** Policy gradient algorithms have been successfully applied to enhance the reasoning capabilities of large language models (LLMs). KL regularization is ubiquitous, yet the design surface, choice of KL direction (forward vs. reverse), normalization (normalized vs. unnormalized), and estimator ($k_1/k_2/k_3$), is scattered across the literature and often intertwined with off-policy estimation. We ask a focused question: under the off-policy setting, what weighting is required for each KL variant so that the surrogate we optimize yields the exact gradient of the intended KL-regularized objective? We answer this with a compact, unified derivation we call the Regularized Policy Gradient (RPG) view. RPG (i) unifies normalized and unnormalized KL variants and shows that the widely-used $k_3$ penalty is exactly the unnormalized KL; (ii) specifies conditions under which REINFORCE-style losses with stop-gradient are gradient-equivalent to fully differentiable surrogates; (iii) identifies and corrects an off-policy importance-weighting mismatch in GRPO's KL term; and (iv) introduces RPG-Style Clip, a clipped-importance-sampling step within RPG-REINFORCE that enables stable, off-policy policy-gradient training at scale. On mathematical reasoning benchmarks (AIME24, AIME25), RPG-REINFORCE with RPG-Style Clip improves accuracy by up to $+6$ absolute percentage points over DAPO. We extend our experiments to 8K context length, and RPG-REINFORCE with RPG-Style Clip achieves 52% accuracy on AIME25, surpassing the official Qwen3-4B-Instruct model (47%). Notably, RPG is a stable and scalable RL algorithm for LLM reasoning, realized via (a) a KL-correct objective, (b) clipped importance sampling, and (c) an iterative reference-policy update scheme.
>
---
#### [replaced 003] Forensic deepfake audio detection using segmental speech features
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文研究深伪音频检测，提出利用可解释性强的音段语音特征（与发音机制相关）进行识别。工作包括验证此类特征在区分真/假音频中的有效性，并提出面向个体说话人的检测框架，以提升法医场景下的可解释性与敏感性。**

- **链接: [https://arxiv.org/pdf/2505.13847v3](https://arxiv.org/pdf/2505.13847v3)**

> **作者:** Tianle Yang; Chengzhe Sun; Siwei Lyu; Phil Rose
>
> **备注:** Accepted for publication in Forensic Science International
>
> **摘要:** This study explores the potential of using acoustic features of segmental speech sounds to detect deepfake audio. These features are highly interpretable because of their close relationship with human articulatory processes and are expected to be more difficult for deepfake models to replicate. The results demonstrate that certain segmental features commonly used in forensic voice comparison (FVC) are effective in identifying deep-fakes, whereas some global features provide little value. These findings underscore the need to approach audio deepfake detection using methods that are distinct from those employed in traditional FVC, and offer a new perspective on leveraging segmental features for this purpose. In addition, the present study proposes a speaker-specific framework for deepfake detection, which differs fundamentally from the speaker-independent systems that dominate current benchmarks. While speaker-independent frameworks aim at broad generalization, the speaker-specific approach offers advantages in forensic contexts where case-by-case interpretability and sensitivity to individual phonetic realization are essential.
>
---
#### [replaced 004] TheMCPCompany: Creating General-purpose Agents with Task-specific Tools
- **分类: cs.CL**

- **简介: 该论文聚焦工具调用代理任务，旨在解决现有代理在复杂真实服务环境中工具利用效率低的问题。作者构建了包含1.8万余工具的基准TheMCPCompany，评估代理在任务特定工具下的表现，揭示先进模型在复杂企业环境中的推理与检索瓶颈。**

- **链接: [https://arxiv.org/pdf/2510.19286v2](https://arxiv.org/pdf/2510.19286v2)**

> **作者:** Reza Esfandiarpoor; Vishwas Suryanarayanan; Stephen H. Bach; Vishal Chowdhary; Anthony Aue
>
> **备注:** Code: https://github.com/Reza-esfandiarpoor/the-mcp-company
>
> **摘要:** Since the introduction of the Model Context Protocol (MCP), the number of available tools for Large Language Models (LLMs) has increased significantly. These task-specific tool sets offer an alternative to general-purpose tools such as web browsers, while being easier to develop and maintain than GUIs. However, current general-purpose agents predominantly rely on web browsers for interacting with the environment. Here, we introduce TheMCPCompany, a benchmark for evaluating tool-calling agents on tasks that involve interacting with various real-world services. We use the REST APIs of these services to create MCP servers, which include over 18,000 tools. We also provide manually annotated ground-truth tools for each task. In our experiments, we use the ground truth tools to show the potential of tool-calling agents for both improving performance and reducing costs assuming perfect tool retrieval. Next, we explore agent performance using tool retrieval to study the real-world practicality of tool-based agents. While all models with tool retrieval perform similarly or better than browser-based agents, smaller models cannot take full advantage of the available tools through retrieval. On the other hand, GPT-5's performance with tool retrieval is very close to its performance with ground-truth tools. Overall, our work shows that the most advanced reasoning models are effective at discovering tools in simpler environments, but seriously struggle with navigating complex enterprise environments. TheMCPCompany reveals that navigating tens of thousands of tools and combining them in non-trivial ways to solve complex problems is still a challenging task for current models and requires both better reasoning and better retrieval models.
>
---
#### [replaced 005] LLMs in Interpreting Legal Documents
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨大语言模型在法律文本解读中的应用，属于法律科技任务。旨在解决法律文书分析、摘要与检索等问题，提升效率与准确性，同时分析算法偏见、幻觉及合规性挑战，并提出两个基准测试。**

- **链接: [https://arxiv.org/pdf/2512.09830v2](https://arxiv.org/pdf/2512.09830v2)**

> **作者:** Simone Corbo
>
> **摘要:** This chapter explores the application of Large Language Models in the legal domain, showcasing their potential to optimise and augment traditional legal tasks by analysing possible use cases, such as assisting in interpreting statutes, contracts, and case law, enhancing clarity in legal summarisation, contract negotiation, and information retrieval. There are several challenges that can arise from the application of such technologies, such as algorithmic monoculture, hallucinations, and compliance with existing regulations, including the EU's AI Act and recent U.S. initiatives, alongside the emerging approaches in China. Furthermore, two different benchmarks are presented.
>
---
#### [replaced 006] Teaching Language Models to Evolve with Users: Dynamic Profile Modeling for Personalized Alignment
- **分类: cs.CL**

- **简介: 该论文研究个性化对齐任务，旨在解决现有方法在冷启动和长期个性化中的静态局限。提出RLPA框架，通过强化学习动态建模用户画像，实现更精准、持续的个性化对话生成。**

- **链接: [https://arxiv.org/pdf/2505.15456v2](https://arxiv.org/pdf/2505.15456v2)**

> **作者:** Weixiang Zhao; Xingyu Sui; Yulin Hu; Jiahe Guo; Haixiao Liu; Biye Li; Yanyan Zhao; Bing Qin; Ting Liu
>
> **备注:** NeurIPS 2025 Camera-ready
>
> **摘要:** Personalized alignment is essential for enabling large language models (LLMs) to engage effectively in user-centric dialogue. While recent prompt-based and offline optimization methods offer preliminary solutions, they fall short in cold-start scenarios and long-term personalization due to their inherently static and shallow designs. In this work, we introduce the Reinforcement Learning for Personalized Alignment (RLPA) framework, in which an LLM interacts with a simulated user model to iteratively infer and refine user profiles through dialogue. The training process is guided by a dual-level reward structure: the Profile Reward encourages accurate construction of user representations, while the Response Reward incentivizes generation of responses consistent with the inferred profile. We instantiate RLPA by fine-tuning Qwen-2.5-3B-Instruct, resulting in Qwen-RLPA, which achieves state-of-the-art performance in personalized dialogue. Empirical evaluations demonstrate that Qwen-RLPA consistently outperforms prompting and offline fine-tuning baselines, and even surpasses advanced commercial models such as Claude-3.5 and GPT-4o. Further analysis highlights Qwen-RLPA's robustness in reconciling conflicting user preferences, sustaining long-term personalization and delivering more efficient inference compared to recent reasoning-focused LLMs. These results emphasize the potential of dynamic profile inference as a more effective paradigm for building personalized dialogue systems.
>
---
#### [replaced 007] Reparameterized LLM Training via Orthogonal Equivalence Transformation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型训练优化任务，旨在解决大语言模型训练不稳定和泛化能力差的问题。作者提出POET算法，通过正交等价变换重参数化神经元，保持权重矩阵谱特性，提升训练稳定性与泛化性能，并设计高效近似方法以支持大规模应用。**

- **链接: [https://arxiv.org/pdf/2506.08001v4](https://arxiv.org/pdf/2506.08001v4)**

> **作者:** Zeju Qiu; Simon Buchholz; Tim Z. Xiao; Maximilian Dax; Bernhard Schölkopf; Weiyang Liu
>
> **备注:** NeurIPS 2025 (40 pages, 26 figures, project page: https://spherelab.ai/poet/, v4: added experiments of finetuning and larger-scale pretraining)
>
> **摘要:** While large language models (LLMs) are driving the rapid advancement of artificial intelligence, effectively and reliably training these large models remains one of the field's most significant challenges. To address this challenge, we propose POET, a novel reParameterized training algorithm that uses Orthogonal Equivalence Transformation to optimize neurons. Specifically, POET reparameterizes each neuron with two learnable orthogonal matrices and a fixed random weight matrix. Because of its provable preservation of spectral properties of weight matrices, POET can stably optimize the objective function with improved generalization. We further develop efficient approximations that make POET flexible and scalable for training large-scale neural networks. Extensive experiments validate the effectiveness and scalability of POET in training LLMs.
>
---
#### [replaced 008] Examining the Metrics for Document-Level Claim Extraction in Czech and Slovak
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究文档级主张提取的评估方法，针对捷克语和斯洛伐克语新闻评论中的非正式语言和语境难题，提出基于对齐的相似性评分，用于衡量模型提取结果与人工标注的一致性，旨在构建可靠的评估框架并揭示现有方法在语义相似性和主张特性评价上的不足。**

- **链接: [https://arxiv.org/pdf/2511.14566v2](https://arxiv.org/pdf/2511.14566v2)**

> **作者:** Lucia Makaiova; Martin Fajcik; Antonin Jarolim
>
> **摘要:** Document-level claim extraction remains an open challenge in the field of fact-checking, and subsequently, methods for evaluating extracted claims have received limited attention. In this work, we explore approaches to aligning two sets of claims pertaining to the same source document and computing their similarity through an alignment score. We investigate techniques to identify the best possible alignment and evaluation method between claim sets, with the aim of providing a reliable evaluation framework. Our approach enables comparison between model-extracted and human-annotated claim sets, serving as a metric for assessing the extraction performance of models and also as a possible measure of inter-annotator agreement. We conduct experiments on newly collected dataset-claims extracted from comments under Czech and Slovak news articles-domains that pose additional challenges due to the informal language, strong local context, and subtleties of these closely related languages. The results draw attention to the limitations of current evaluation approaches when applied to document-level claim extraction and highlight the need for more advanced methods-ones able to correctly capture semantic similarity and evaluate essential claim properties such as atomicity, checkworthiness, and decontextualization.
>
---
#### [replaced 009] Towards Personalized Deep Research: Benchmarks and Evaluations
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文聚焦个性化深度研究代理（DRA）评估，旨在解决现有基准忽视个性化的问题。作者提出PDR-Bench，首个涵盖250个真实用户-任务查询的个性化评测集，并设计PQR框架评估个性化对齐、内容质量与事实可靠性，推动个性化AI研究助手发展。**

- **链接: [https://arxiv.org/pdf/2509.25106v2](https://arxiv.org/pdf/2509.25106v2)**

> **作者:** Yuan Liang; Jiaxian Li; Yuqing Wang; Piaohong Wang; Motong Tian; Pai Liu; Shuofei Qiao; Runnan Fang; He Zhu; Ge Zhang; Minghao Liu; Yuchen Eleanor Jiang; Ningyu Zhang; Wangchunshu Zhou
>
> **摘要:** Deep Research Agents (DRAs) can autonomously conduct complex investigations and generate comprehensive reports, demonstrating strong real-world potential. However, existing benchmarks primarily evaluate DRAs on generic quality metrics and overlook personalization, a critical dimension for individual users. However, existing evaluations mostly rely on close-ended benchmarks, while open-ended deep research benchmarks remain scarce and typically neglect personalized scenarios. To bridge this gap, we introduce Personalized Deep Research Bench (PDR-Bench), the first benchmark for evaluating personalization in DRAs. It pairs 50 diverse research tasks across 10 domains with 25 authentic user profiles that combine structured persona attributes with dynamic real-world contexts, yielding 250 realistic user-task queries. To assess system performance, we propose the PQR Evaluation Framework, which jointly measures Personalization Alignment, Content Quality, and Factual Reliability. Our experiments on a range of systems highlight current capabilities and limitations in handling personalized deep research. This work establishes a rigorous foundation for developing and evaluating the next generation of truly personalized AI research assistants.
>
---
#### [replaced 010] Forest vs Tree: The $(N, K)$ Trade-off in Reproducible ML Evaluation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究机器学习评测中样本数量（N）与每样本标注次数（K）的权衡，旨在提升可复现性。针对人类标注分歧问题，分析多标注数据，探讨固定预算下最优（N, K）配置，发现较高K常更优，且效果依赖于评估指标特性。**

- **链接: [https://arxiv.org/pdf/2508.03663v2](https://arxiv.org/pdf/2508.03663v2)**

> **作者:** Deepak Pandita; Flip Korn; Chris Welty; Christopher M. Homan
>
> **备注:** Accepted at AAAI-26
>
> **摘要:** Reproducibility is a cornerstone of scientific validation and of the authority it confers on its results. Reproducibility in machine learning evaluations leads to greater trust, confidence, and value. However, the ground truth responses used in machine learning often necessarily come from humans, among whom disagreement is prevalent, and surprisingly little research has studied the impact of effectively ignoring disagreement in these responses, as is typically the case. One reason for the lack of research is that budgets for collecting human-annotated evaluation data are limited, and obtaining more samples from multiple raters for each example greatly increases the per-item annotation costs. We investigate the trade-off between the number of items ($N$) and the number of responses per item ($K$) needed for reliable machine learning evaluation. We analyze a diverse collection of categorical datasets for which multiple annotations per item exist, and simulated distributions fit to these datasets, to determine the optimal $(N, K)$ configuration, given a fixed budget ($N \times K$), for collecting evaluation data and reliably comparing the performance of machine learning models. Our findings show, first, that accounting for human disagreement may come with $N \times K$ at no more than 1000 (and often much lower) for every dataset tested on at least one metric. Moreover, this minimal $N \times K$ almost always occurred for $K > 10$. Furthermore, the nature of the tradeoff between $K$ and $N$, or if one even existed, depends on the evaluation metric, with metrics that are more sensitive to the full distribution of responses performing better at higher levels of $K$. Our methods can be used to help ML practitioners get more effective test data by finding the optimal metrics and number of items and annotations per item to collect to get the most reliability for their budget.
>
---
#### [replaced 011] GTPO: Stabilizing Group Relative Policy Optimization via Gradient and Entropy Control
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大语言模型对齐中的Group Relative Policy Optimization（GRPO）方法，解决其训练不稳定和策略崩溃问题。提出GTPO方法，通过梯度控制与熵过滤提升稳定性，无需KL正则与参考模型，实验验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2508.03772v5](https://arxiv.org/pdf/2508.03772v5)**

> **作者:** Marco Simoni; Aleksandar Fontana; Giulio Rossolini; Andrea Saracino; Paolo Mori
>
> **摘要:** Group Relative Policy Optimization (GRPO) is a promising policy-based approach for Large Language Model alignment, yet its performance is often limited by training instability and suboptimal convergence. In this paper, we identify and analyze two main GRPO issues: (i) the token-level penalization, where valuable tokens shared across different responses receive contradictory feedback signals, leading to conflicting gradient updates that can reduce their likelihood; and (ii) the policy collapse, where negatively rewarded completions may penalize confident responses and shift model decisions toward unlikely tokens, destabilizing training process. To address these issues we introduce GTPO (Group-relative Trajectory-based Policy Optimization), which prevents conflicting gradients on valuable tokens by skipping negative updates while amplifying positive ones and filters out completions whose entropy exceeds a provable threshold, to prevent policy collapse. Unlike GRPO, GTPO does not rely on KL-divergence regularization, eliminating the need for a reference model during training, while still ensuring greater training stability and improved performance, as validated through multiple experiments on GSM8K, MATH, AIME 2024, AIME 2025 and AMC 2023.
>
---
#### [replaced 012] LMSpell: Neural Spell Checking for Low-Resource Languages
- **分类: cs.CL**

- **简介: 该论文聚焦低资源语言的拼写纠错任务，旨在解决现有预训练模型在此类语言上应用不足的问题。作者系统评估了不同预训练模型的效果，发现大语言模型在大数据微调下表现更优，并发布了包含评估功能的工具包LMSpell，辅以僧伽罗语案例研究。**

- **链接: [https://arxiv.org/pdf/2512.05414v3](https://arxiv.org/pdf/2512.05414v3)**

> **作者:** Akesh Gunathilake; Nadil Karunarathna; Tharusha Bandaranayake; Nisansa de Silva; Surangika Ranathunga
>
> **摘要:** Spell correction is still a challenging problem for low-resource languages (LRLs). While pretrained language models (PLMs) have been employed for spell correction, their use is still limited to a handful of languages, and there has been no proper comparison across PLMs. We present the first empirical study on the effectiveness of PLMs for spell correction, which includes LRLs. We find that Large Language Models (LLMs) outperform their counterparts (encoder-based and encoder-decoder) when the fine-tuning dataset is large. This observation holds even in languages for which the LLM is not pre-trained. We release LMSpell, an easy- to use spell correction toolkit across PLMs. It includes an evaluation function that compensates for the hallucination of LLMs. Further, we present a case study with Sinhala to shed light on the plight of spell correction for LRLs.
>
---
#### [replaced 013] Luxical: High-Speed Lexical-Dense Text Embeddings
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本表示任务，旨在解决现有文本嵌入方法在速度与灵活性间的权衡问题。作者提出Luxical，结合TF-IDF、小神经网络与知识蒸馏，实现接近大模型质量但更快速的lexical-dense嵌入，适用于大规模文本组织。**

- **链接: [https://arxiv.org/pdf/2512.09015v2](https://arxiv.org/pdf/2512.09015v2)**

> **作者:** DatologyAI; :; Luke Merrick; Alex Fang; Aldo Carranza; Alvin Deng; Amro Abbas; Brett Larsen; Cody Blakeney; Darren Teh; David Schwab; Fan Pan; Haakon Mongstad; Haoli Yin; Jack Urbanek; Jason Lee; Jason Telanoff; Josh Wills; Kaleigh Mentzer; Paul Burstein; Parth Doshi; Paul Burnstein; Pratyush Maini; Ricardo Monti; Rishabh Adiga; Scott Loftin; Siddharth Joshi; Spandan Das; Tony Jiang; Vineeth Dorna; Zhengping Wang; Bogdan Gaza; Ari Morcos; Matthew Leavitt
>
> **备注:** 9 pages, 6 figures (v2 fixes typos only)
>
> **摘要:** Frontier language model quality increasingly hinges on our ability to organize web-scale text corpora for training. Today's dominant tools trade off speed and flexibility: lexical classifiers (e.g., FastText) are fast but limited to producing classification output scores, while the vector-valued outputs of transformer text embedding models flexibly support numerous workflows (e.g., clustering, classification, and retrieval) but are computationally expensive to produce. We introduce Luxical, a library for high-speed "lexical-dense" text embeddings that aims to recover the best properties of both approaches for web-scale text organization. Luxical combines sparse TF--IDF features, a small ReLU network, and a knowledge distillation training regimen to approximate large transformer embedding models at a fraction of their operational cost. In this technical report, we describe the Luxical architecture and training objective and evaluate a concrete Luxical model in two disparate applications: a targeted webcrawl document retrieval test and an end-to-end language model data curation task grounded in text classification. In these tasks we demonstrate speedups ranging from 3x to 100x over varying-sized neural baselines, and comparable to FastText model inference during the data curation task. On these evaluations, the tested Luxical model illustrates favorable compute/quality trade-offs for large-scale text organization, matching the quality of neural baselines. Luxical is available as open-source software at https://github.com/datologyai/luxical.
>
---
#### [replaced 014] The LLM Wears Prada: Analysing Gender Bias and Stereotypes through Online Shopping Data
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属性别偏见分析任务，旨在探究大语言模型是否通过网购数据推断性别并受刻板印象影响。作者利用美国用户购物记录，评估六种LLM的性别分类能力及推理逻辑，发现模型依赖性别刻板关联，且去偏指令难消除此类模式。**

- **链接: [https://arxiv.org/pdf/2504.01951v2](https://arxiv.org/pdf/2504.01951v2)**

> **作者:** Massimiliano Luca; Ciro Beneduce; Bruno Lepri; Jacopo Staiano
>
> **摘要:** With the wide and cross-domain adoption of Large Language Models, it becomes crucial to assess to which extent the statistical correlations in training data, which underlie their impressive performance, hide subtle and potentially troubling biases. Gender bias in LLMs has been widely investigated from the perspectives of works, hobbies, and emotions typically associated with a specific gender. In this study, we introduce a novel perspective. We investigate whether LLMs can predict an individual's gender based solely on online shopping histories and whether these predictions are influenced by gender biases and stereotypes. Using a dataset of historical online purchases from users in the United States, we evaluate the ability of six LLMs to classify gender and we then analyze their reasoning and products-gender co-occurrences. Results indicate that while models can infer gender with moderate accuracy, their decisions are often rooted in stereotypical associations between product categories and gender. Furthermore, explicit instructions to avoid bias reduce the certainty of model predictions, but do not eliminate stereotypical patterns. Our findings highlight the persistent nature of gender biases in LLMs and emphasize the need for robust bias-mitigation strategies.
>
---
#### [replaced 015] OutSafe-Bench: A Benchmark for Multimodal Offensive Content Detection in Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文聚焦多模态大语言模型的安全内容检测任务，旨在解决现有基准在模态覆盖和风险评估上的局限。作者构建了涵盖文本、图像、音频、视频的多语言安全评测集OutSafe-Bench，提出多维交叉风险评分MCRS与公平评估框架FairScore，系统评估主流MLLM的安全漏洞。**

- **链接: [https://arxiv.org/pdf/2511.10287v3](https://arxiv.org/pdf/2511.10287v3)**

> **作者:** Yuping Yan; Yuhan Xie; Yuanshuai Li; Yingchao Yu; Lingjuan Lyu; Yaochu Jin
>
> **摘要:** Since Multimodal Large Language Models (MLLMs) are increasingly being integrated into everyday tools and intelligent agents, growing concerns have arisen regarding their possible output of unsafe contents, ranging from toxic language and biased imagery to privacy violations and harmful misinformation. Current safety benchmarks remain highly limited in both modality coverage and performance evaluations, often neglecting the extensive landscape of content safety. In this work, we introduce OutSafe-Bench, the first most comprehensive content safety evaluation test suite designed for the multimodal era. OutSafe-Bench includes a large-scale dataset that spans four modalities, featuring over 18,000 bilingual (Chinese and English) text prompts, 4,500 images, 450 audio clips and 450 videos, all systematically annotated across nine critical content risk categories. In addition to the dataset, we introduce a Multidimensional Cross Risk Score (MCRS), a novel metric designed to model and assess overlapping and correlated content risks across different categories. To ensure fair and robust evaluation, we propose FairScore, an explainable automated multi-reviewer weighted aggregation framework. FairScore selects top-performing models as adaptive juries, thereby mitigating biases from single-model judgments and enhancing overall evaluation reliability. Our evaluation of nine state-of-the-art MLLMs reveals persistent and substantial safety vulnerabilities, underscoring the pressing need for robust safeguards in MLLMs.
>
---
#### [replaced 016] Heard or Halted? Gender, Interruptions, and Emotional Tone in U.S. Supreme Court Oral Arguments
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究美国最高法院口头辩论中性别与打断的关系，探讨打断是否影响论点内容及情感倾向。基于语料库和计算语言学方法，分析发现打断未显著改变语义，但针对女性律师的打断更具负面情绪。**

- **链接: [https://arxiv.org/pdf/2512.05832v2](https://arxiv.org/pdf/2512.05832v2)**

> **作者:** Yifei Tong
>
> **备注:** 12 pages, 5 figures, 1 table. Includes appendix. Code available at: https://github.com/1TSHARUKA/Emotional_Interruption_Analysis
>
> **摘要:** This study examines how interruptions during U.S. Supreme Court oral arguments shape both the semantic content and emotional tone of advocates' speech, with a focus on gendered dynamics in judicial discourse. Using the ConvoKit Supreme Court Corpus (2010-2019), we analyze 12,663 speech chunks from advocate-justice interactions to assess whether interruptions alter the meaning of an advocate's argument and whether interruptions toward female advocates exhibit more negative emotional valence. Semantic shifts are quantified using GloVe-based sentence embeddings, while sentiment is measured through lexicon-based analysis. We find that semantic similarity between pre- and post-interruption speech remains consistently high, suggesting that interruptions do not substantially alter argumentative content. However, interruptions directed at female advocates contain significantly higher levels of negative sentiment. These results deepen empirical understanding of gendered communication in elite institutional settings and demonstrate the value of computational linguistic methods for studying power, discourse, and equity in judicial proceedings.
>
---
#### [replaced 017] Leveraging language models for summarizing mental state examinations: A comprehensive evaluation and dataset release
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理在心理健康领域的应用任务，旨在解决发展中国家精神卫生资源不足导致的评估效率低下问题。作者构建了包含9720条语句的MSE对话数据集，评估了多种语言模型自动生成精神状态检查摘要的效果，并公开发布数据与模型。**

- **链接: [https://arxiv.org/pdf/2403.20145v3](https://arxiv.org/pdf/2403.20145v3)**

> **作者:** Nilesh Kumar Sahu; Manjeet Yadav; Mudita Chaturvedi; Snehil Gupta; Haroon R Lone
>
> **备注:** Appeared in: Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025), Abu Dhabi, UAE. ACL Anthology ID: 2025.coling-main.182. (https://aclanthology.org/2025.coling-main.182/)
>
> **摘要:** Mental health disorders affect a significant portion of the global population, with diagnoses primarily conducted through Mental State Examinations (MSEs). MSEs serve as structured assessments to evaluate behavioral and cognitive functioning across various domains, aiding mental health professionals in diagnosis and treatment monitoring. However, in developing countries, access to mental health support is limited, leading to an overwhelming demand for mental health professionals. Resident doctors often conduct initial patient assessments and create summaries for senior doctors, but their availability is constrained, resulting in extended patient wait times. This study addresses the challenge of generating concise summaries from MSEs through the evaluation of various language models. Given the scarcity of relevant mental health conversation datasets, we developed a 12-item descriptive MSE questionnaire and collected responses from 405 participants, resulting in 9720 utterances covering diverse mental health aspects. Subsequently, we assessed the performance of five well-known pre-trained summarization models, both with and without fine-tuning, for summarizing MSEs. Our comprehensive evaluation, leveraging metrics such as ROUGE, SummaC, and human evaluation, demonstrates that language models can generate automated coherent MSE summaries for doctors. With this paper, we release our collected conversational dataset and trained models publicly for the mental health research community.
>
---
#### [replaced 018] When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners
- **分类: cs.CL**

- **简介: 该论文研究多语言推理任务，旨在解决大模型在低资源语言上表现差的问题。受认知科学启发，提出通过推理与语言表征解耦，消融语言特定表示以提升多语言推理能力，无需训练且效果优于微调方法。**

- **链接: [https://arxiv.org/pdf/2505.15257v2](https://arxiv.org/pdf/2505.15257v2)**

> **作者:** Weixiang Zhao; Jiahe Guo; Yang Deng; Tongtong Wu; Wenxuan Zhang; Yulin Hu; Xingyu Sui; Yanyan Zhao; Wanxiang Che; Bing Qin; Tat-Seng Chua; Ting Liu
>
> **备注:** NeurIPS 2025 Camera-ready
>
> **摘要:** Multilingual reasoning remains a significant challenge for large language models (LLMs), with performance disproportionately favoring high-resource languages. Drawing inspiration from cognitive neuroscience, which suggests that human reasoning functions largely independently of language processing, we hypothesize that LLMs similarly encode reasoning and language as separable components that can be disentangled to enhance multilingual reasoning. To evaluate this, we perform a causal intervention by ablating language-specific representations at inference time. Experiments on 10 open-weight LLMs spanning 11 typologically diverse languages show that this language-specific ablation consistently boosts multilingual reasoning performance. Layer-wise analyses further confirm that language and reasoning representations can be effectively disentangled throughout the model, yielding improved multilingual reasoning capabilities, while preserving top-layer language features remains essential for maintaining linguistic fidelity. Compared to post-training methods such as supervised fine-tuning or reinforcement learning, our training-free language-reasoning disentanglement achieves comparable or superior results with minimal computational overhead. These findings shed light on the internal mechanisms underlying multilingual reasoning in LLMs and suggest a lightweight and interpretable strategy for improving cross-lingual generalization.
>
---
#### [replaced 019] Advancing AI Research Assistants with Expert-Involved Learning
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文聚焦AI在生物医学研究中的可靠性问题，提出ARIEL框架，通过专家参与的评估与优化，提升大模型在文章摘要和图表理解上的表现，推动可信赖AI在生物医学中的应用。**

- **链接: [https://arxiv.org/pdf/2505.04638v3](https://arxiv.org/pdf/2505.04638v3)**

> **作者:** Tianyu Liu; Simeng Han; Hanchen Wang; Xiao Luo; Pan Lu; Biqing Zhu; Yuge Wang; Keyi Li; Jiapeng Chen; Rihao Qu; Yufeng Liu; Xinyue Cui; Aviv Yaish; Yuhang Chen; Minsheng Hao; Chuhan Li; Kexing Li; Arman Cohan; Hua Xu; Mark Gerstein; James Zou; Hongyu Zhao
>
> **备注:** 36 pages, 7 figures
>
> **摘要:** Large language models (LLMs) and large multimodal models (LMMs) promise to accelerate biomedical discovery, yet their reliability remains unclear. We introduce ARIEL (AI Research Assistant for Expert-in-the-Loop Learning), an open-source evaluation and optimization framework that pairs a curated multimodal biomedical corpus with expert-vetted tasks to probe two capabilities: full-length article summarization and fine-grained figure interpretation. Using uniform protocols and blinded PhD-level evaluation, we find that state-of-the-art models generate fluent but incomplete summaries, whereas LMMs struggle with detailed visual reasoning. We later observe that prompt engineering and lightweight fine-tuning substantially improve textual coverage, and a compute-scaled inference strategy enhances visual question answering. We build an ARIEL agent that integrates textual and visual cues, and we show it can propose testable mechanistic hypotheses. ARIEL delineates current strengths and limitations of foundation models, and provides a reproducible platform for advancing trustworthy AI in biomedicine.
>
---
#### [replaced 020] Emotional Support with LLM-based Empathetic Dialogue Generation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于情感支持对话任务，旨在通过大语言模型生成共情且恰当的回应。作者结合提示工程与高效微调方法，提升模型表现，其方案在NLPCC 2025评测中排名第二，展示了大模型在情感支持对话中的潜力。**

- **链接: [https://arxiv.org/pdf/2507.12820v2](https://arxiv.org/pdf/2507.12820v2)**

> **作者:** Shiquan Wang; Ruiyu Fang; Zhongjiang He; Shuangyong Song; Yongxiang Li
>
> **摘要:** Emotional Support Conversation (ESC) aims to provide empathetic and effective emotional assistance through dialogue, addressing the growing demand for mental health support. This paper presents our solution for the NLPCC 2025 Task 8 ESC evaluation, where we leverage large-scale language models enhanced by prompt engineering and finetuning techniques. We explore both parameter-efficient Low-Rank Adaptation and full-parameter fine-tuning strategies to improve the model's ability to generate supportive and contextually appropriate responses. Our best model ranked second in the competition, highlighting the potential of combining LLMs with effective adaptation methods for ESC tasks. Future work will focus on further enhancing emotional understanding and response personalization to build more practical and reliable emotional support systems.
>
---
#### [replaced 021] V-VAE: A Variational Auto Encoding Framework Towards Fine-Grained Control over Human-Like Chat
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究角色化对话生成任务，旨在解决现有方法因静态设定和低质数据导致难以建模细粒度人类对话特征的问题。提出V-VAE框架与高质量数据集HumanChatData，实现对说话风格、互动模式等细粒度控制，提升类人对话生成效果。**

- **链接: [https://arxiv.org/pdf/2506.01524v2](https://arxiv.org/pdf/2506.01524v2)**

> **作者:** Qi Lin; Weikai Xu; Lisi Chen; Bin Dai
>
> **摘要:** With the continued proliferation of Large Language Model (LLM) based chatbots, there is a growing demand for generating responses that are not only linguistically fluent but also consistently aligned with persona-specific traits in conversations. However, existing role-play and persona-based chat approaches rely heavily on static role descriptions, coarse-grained signal space, and low-quality synthetic data, which fail to capture dynamic fine-grained details in human-like chat. Human-like chat requires modeling subtle latent traits, such as emotional tone, situational awareness, and evolving personality, which are difficult to predefine and cannot be easily learned from synthetic or distillation-based data. To address these limitations, we propose a Verbal Variational Auto-Encoding (V-VAE) framework, containing a variational auto-encoding module and fine-grained control space which dynamically adapts dialogue behaviour based on fine-grained, interpretable latent variables across talking style, interaction patterns, and personal attributes. We also construct a high-quality dataset, HumanChatData, and benchmark HumanChatBench to address the scarcity of high-quality data in the human-like domain. Experiments show that LLMs based on V-VAE consistently outperform standard baselines on HumanChatBench and DialogBench, which further demonstrates the effectiveness of V-VAE and HumanChatData.
>
---
#### [replaced 022] UltraCUA: A Foundation Model for Computer Use Agents with Hybrid Action
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究计算机使用代理任务，旨在解决仅依赖低级GUI操作导致的脆弱性问题。提出UltraCUA模型，通过融合GUI操作与高级工具调用的混合动作范式，提升代理的鲁棒性与效率。**

- **链接: [https://arxiv.org/pdf/2510.17790v2](https://arxiv.org/pdf/2510.17790v2)**

> **作者:** Yuhao Yang; Zhen Yang; Zi-Yi Dou; Anh Nguyen; Keen You; Omar Attia; Andrew Szot; Michael Feng; Ram Ramrakhya; Alexander Toshev; Chao Huang; Yinfei Yang; Zhe Gan
>
> **摘要:** Computer-use agents face a fundamental limitation. They rely exclusively on primitive GUI actions (click, type, scroll), creating brittle execution chains prone to cascading failures. While API-driven agents harness rich capabilities through structured interfaces and tools, computer-use agents remain constrained to low-level visual interactions. We present UltraCUA, a foundation model that transcends this limitation through hybrid action-seamlessly unifying primitive GUI operations with high-level tool execution. Our innovation rests on four critical advances. First, an automated pipeline extracts and scales tool capabilities from software documentation and code repositories. Second, a synthetic data engine produces 17,000+ verifiable tasks capturing real-world computer-use complexity. Third, comprehensive hybrid action trajectory collection incorporates both GUI primitives and strategic tool calls. Fourth, a two-stage training methodology combines supervised fine-tuning with online reinforcement learning, enabling intelligent action selection between GUI and API. Evaluation with our 7B and 32B UltraCUA models reveals transformative performance gains. On OSWorld, UltraCUA achieves 22% relative improvement while executing 11% faster than existing approaches, averagely. Cross-domain validation on WindowsAgentArena demonstrates robust generalization with 21.7% success rate, surpassing Windows-trained baselines. The hybrid action paradigm proves essential, reducing error propagation while improving execution efficiency. This work establishes a scalable paradigm bridging primitive GUI interactions and high-level tool intelligence, enabling more resilient and adaptable computer use agents for diverse environments and complex real-world tasks.
>
---
#### [replaced 023] The Spatial Semantics of Iconic Gesture
- **分类: cs.CL**

- **简介: 该论文探讨多模态语言中手势的意义问题，提出一种空间手势语义框架。通过将手势形式转化为向量序列、在空间域中进行变换与嵌入，并结合信息性评价，实现手势与言语意义的整合。**

- **链接: [https://arxiv.org/pdf/2404.18708v2](https://arxiv.org/pdf/2404.18708v2)**

> **作者:** Andy Lücking; Alexander Henlein; Alexander Mehler
>
> **备注:** 52 pages, 38 figures, in review
>
> **摘要:** The current multimodal turn in linguistic theory leaves a crucial question unanswered: what is the meaning of iconic gestures, and how does it compose with speech meaning? We argue for a separation of linguistic and visual levels of meaning and introduce a spatial gesture semantics that closes this gap. Iconicity is differentiated into three aspects: Firstly, an interpretation of the form of a gesture in terms of a translation from kinematic gesture annotations into vector sequences (iconic model). Secondly, a truth-functional evaluation of the iconic model within spatially extended domains (embedding). Since a simple embedding is too strong, we identify a number of transformations that can be applied to iconic models, namely rotation, scaling, perspective fixation, and quotation of handshape. Thirdly, the linguistic description or classification of an iconic model (informational evaluation). Since the informational evaluation of an iconic gesture is a heuristic act, it needs a place in a semantic theory of visual communication. Informational evaluation lifts a gesture to a quasi-linguistic level that can interact with verbal content. This interaction is either vacuous, or regimented by usual lexicon-driven inferences discussed in dynamic semantic frameworks.
>
---
#### [replaced 024] Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何无需训练地将非文本模态表征融入大语言模型。提出In-Context Representation Learning（ICRL），通过上下文学习机制，用非文本基础模型表征替代文本输入，实现跨模态推理，验证了其在分子任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2509.17552v3](https://arxiv.org/pdf/2509.17552v3)**

> **作者:** Tianle Zhang; Wanlong Fang; Jonathan Woo; Paridhi Latawa; Deepak A. Subramanian; Alvin Chan
>
> **备注:** NeurIPS 2025
>
> **摘要:** The remarkable performance of Large Language Models (LLMs) can be enhanced with test-time computation, which relies on external tools and even other deep learning models. However, existing approaches for integrating non-text modality representations into LLMs typically require additional costly supervised training, restricting on-the-fly adaptation to new domains and modalities. In this work, we explore the feasibility of integrating representations from non-text foundational models (FMs) into text-based LLMs in a training-free manner. We propose In-Context Representation Learning (ICRL) as a proof-of-concept to allow LLMs to adaptively utilize non-text modality representations with few-shot learning. Unlike traditional in-context learning, which incorporates text-label pairs, ICRL replaces text inputs with FM representations, enabling the LLM to perform multi-modal inference without fine-tuning. We evaluate ICRL on a suite of tasks in the molecular domain, investigating three core research questions: (i) how to map FM representations into LLMs in a training-free manner, (ii) what factors influence ICRL performance, and (iii) what mechanisms underlie the effectiveness of ICRL. To the best of our knowledge, ICRL is the first training-free framework for integrating non-text modality representations into text-based LLMs, presenting a promising direction for adaptable, multi-modal generalization.
>
---
#### [replaced 025] Vision-centric Token Compression in Large Language Model
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对大语言模型中上下文过长导致的高计算与内存开销问题，提出一种视觉中心的双通路压缩框架Vist，通过将低显著性上下文转为图像由轻量视觉编码器处理，保留关键文本输入LLM，显著降低资源消耗并提升压缩效率。**

- **链接: [https://arxiv.org/pdf/2502.00791v5](https://arxiv.org/pdf/2502.00791v5)**

> **作者:** Ling Xing; Alex Jinpeng Wang; Rui Yan; Xiangbo Shu; Jinhui Tang
>
> **备注:** NeurIPS 2025 spotlight
>
> **摘要:** Real-world applications are stretching context windows to hundreds of thousand of tokens while Large Language Models (LLMs) swell from billions to trillions of parameters. This dual expansion send compute and memory costs skyrocketing, making token compression indispensable. We introduce Vision Centric Token Compression (Vist), a slow-fast compression framework that mirrors human reading: the fast path renders distant tokens into images, letting a frozen, lightweight vision encoder skim the low-salience context; the slow path feeds the proximal window into the LLM for fine-grained reasoning. A Probability-Informed Visual Enhancement (PVE) objective masks high-frequency tokens during training, steering the Resampler to concentrate on semantically rich regions-just as skilled reader gloss over function words. On eleven in-context learning benchmarks, Vist achieves the same accuracy with 2.3 times fewer tokens, cutting FLOPs by 16% and memory by 50%. This method delivers remarkable results, outperforming the strongest text encoder-based compression method CEPE by 7.6% on average over benchmarks like TriviaQA, NQ, PopQA, NLUI, and CLIN, setting a new standard for token efficiency in LLMs. The project is at https://github.com/CSU-JPG/VIST.
>
---
#### [replaced 026] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于检索增强生成（RAG）任务，旨在解决现有方法难以建模知识间复杂关系及图结构噪声问题。作者提出GFM-RAG，首个用于RAG的图基础模型，通过创新图神经网络和两阶段大规模训练，实现跨数据集零样本迁移，提升多跳问答与领域推理性能。**

- **链接: [https://arxiv.org/pdf/2502.01113v3](https://arxiv.org/pdf/2502.01113v3)**

> **作者:** Linhao Luo; Zicheng Zhao; Gholamreza Haffari; Dinh Phung; Chen Gong; Shirui Pan
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Retrieval-augmented generation (RAG) has proven effective in integrating knowledge into large language models (LLMs). However, conventional RAGs struggle to capture complex relationships between pieces of knowledge, limiting their performance in intricate reasoning that requires integrating knowledge from multiple sources. Recently, graph-enhanced retrieval augmented generation (GraphRAG) builds graph structure to explicitly model these relationships, enabling more effective and efficient retrievers. Nevertheless, its performance is still hindered by the noise and incompleteness within the graph structure. To address this, we introduce GFM-RAG, a novel graph foundation model (GFM) for retrieval augmented generation. GFM-RAG is powered by an innovative graph neural network that reasons over graph structure to capture complex query-knowledge relationships. The GFM with 8M parameters undergoes a two-stage training process on large-scale datasets, comprising 60 knowledge graphs with over 14M triples and 700k documents. This results in impressive performance and generalizability for GFM-RAG, making it the first graph foundation model applicable to unseen datasets for retrieval without any fine-tuning required. Extensive experiments on three multi-hop QA datasets and seven domain-specific RAG datasets demonstrate that GFM-RAG achieves state-of-the-art performance while maintaining efficiency and alignment with neural scaling laws, highlighting its potential for further improvement.
>
---
#### [replaced 027] SCALE: Upscaled Continual Learning of Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的持续预训练任务，旨在缓解灾难性遗忘。提出SCALE架构，通过宽度扩展和参数冻结，在不干扰原模型的前提下提升容量，平衡知识保留与新知识学习。**

- **链接: [https://arxiv.org/pdf/2511.03270v2](https://arxiv.org/pdf/2511.03270v2)**

> **作者:** Jin-woo Lee; Junhwa Choi; Bongkyu Hwang; Jinho Choo; Bogun Kim; JeongSeon Yi; Joonseok Lee; DongYoung Jung; Jaeseon Park; Kyoungwon Park; Suk-hoon Jung
>
> **摘要:** We revisit continual pre-training for large language models and argue that progress now depends more on scaling the right structure than on scaling parameters alone. We introduce SCALE, a width upscaling architecture that inserts lightweight expansion into linear modules while freezing all pre-trained parameters. This preserves the residual and attention topologies and increases capacity without perturbing the base model's original functionality. SCALE is guided by two principles: Persistent Preservation, which maintains the base model's behavior via preservation-oriented initialization and freezing of the pre-trained weights, and Collaborative Adaptation, which selectively trains a subset of expansion components to acquire new knowledge with minimal interference. We instantiate these ideas as SCALE-Preserve (preservation-first), SCALE-Adapt (adaptation-first), and SCALE-Route, an optional routing extension that performs token-level routing between preservation and adaptation heads. On a controlled synthetic biography benchmark, SCALE mitigates the severe forgetting observed with depth expansion while still acquiring new knowledge. In continual pre-training on a Korean corpus, SCALE variants achieve less forgetting on English evaluations and competitive gains on Korean benchmarks, with these variants offering the best overall stability-plasticity trade-off. Accompanying analysis clarifies when preservation provably holds and why the interplay between preservation and adaptation stabilizes optimization compared to standard continual learning setups.
>
---
#### [replaced 028] Benchmarking Multimodal LLMs on Recognition and Understanding over Chemical Tables
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦科学领域多模态大模型评估，针对现有基准缺乏领域特性的不足，构建了含精细标注的化学表格基准ChemTable，支持表格识别与理解两大任务，揭示模型在分子结构和符号理解上的瓶颈。**

- **链接: [https://arxiv.org/pdf/2506.11375v2](https://arxiv.org/pdf/2506.11375v2)**

> **作者:** Yitong Zhou; Mingyue Cheng; Qingyang Mao; Yucong Luo; Qi Liu; Yupeng Li; Xiaohan Zhang; Deguang Liu; Xin Li; Enhong Chen
>
> **摘要:** With the widespread application of multimodal large language models in scientific intelligence, there is an urgent need for more challenging evaluation benchmarks to assess their ability to understand complex scientific data. Scientific tables, as core carriers of knowledge representation, combine text, symbols, and graphics, forming a typical multimodal reasoning scenario. However, existing benchmarks are mostly focused on general domains, failing to reflect the unique structural complexity and domain-specific semantics inherent in scientific research. Chemical tables are particularly representative: they intertwine structured variables such as reagents, conditions, and yields with visual symbols like molecular structures and chemical formulas, posing significant challenges to models in cross-modal alignment and semantic parsing. To address this, we propose ChemTable-a large scale benchmark of chemical tables constructed from real-world literature, containing expert-annotated cell layouts, logical structures, and domain-specific labels. It supports two core tasks: (1) table recognition (structure and content extraction); and (2) table understanding (descriptive and reasoning-based question answering). Evaluation on ChemTable shows that while mainstream multimodal models perform reasonably well in layout parsing, they still face significant limitations when handling critical elements such as molecular structures and symbolic conventions. Closed-source models lead overall but still fall short of human-level performance. This work provides a realistic testing platform for evaluating scientific multimodal understanding, revealing the current bottlenecks in domain-specific reasoning and advancing the development of intelligent systems for scientific research.
>
---
#### [replaced 029] A Greek Government Decisions Dataset for Public-Sector Analysis and Insight
- **分类: cs.CL**

- **简介: 该论文构建了包含百万条希腊政府决策的开源语料库，旨在提升公共部门信息透明与访问。通过高质量文本提取与可复现流程，支持检索增强生成任务，探索政府文档的问答应用，并为法律领域大模型提供训练资源。**

- **链接: [https://arxiv.org/pdf/2512.05647v2](https://arxiv.org/pdf/2512.05647v2)**

> **作者:** Giorgos Antoniou; Giorgos Filandrianos; Aggelos Vlachos; Giorgos Stamou; Lampros Kollimenos; Konstantinos Skianis; Michalis Vazirgiannis
>
> **摘要:** We introduce an open, machine-readable corpus of Greek government decisions sourced from the national transparency platform Diavgeia. The resource comprises 1 million decisions, featuring and high-quality raw text extracted from PDFs. It is released with raw extracted text in Markdown format, alongside a fully reproducible extraction pipeline. Beyond the core dataset, we conduct qualitative analyses to explore boilerplate patterns and design a retrieval-augmented generation (RAG) task by formulating a set of representative questions, creating high-quality answers, and evaluating a baseline RAG system on its ability to retrieve and reason over public decisions. This evaluation demonstrates the potential of large-scale public-sector corpora to support advanced information access and transparency through structured retrieval and reasoning over governmental documents, and highlights how such a RAG pipeline could simulate a chat-based assistant capable of interactively answering questions about public decisions. Due to its scale, quality, and domain coverage, the corpus can also serve as high-value pre-training or fine-tuning material for new Language Models (LMs) and Large Language Models (LLMs) respectively, including specialized models for legal and governmental domains, and as a foundation for novel approaches in domain adaptation, knowledge-grounded generation, and explainable AI. Finally, we discuss limitations, outline future directions, and make both the data and the code accessible.
>
---
#### [replaced 030] Better Language Model Inversion by Compactly Representing Next-Token Distributions
- **分类: cs.CL**

- **简介: 该论文研究语言模型反演任务，旨在通过模型输出恢复隐藏提示。提出PILS方法，利用低维子空间压缩下一词概率序列，提升反演准确率，显著优于现有方法，揭示了logprob序列的脆弱性。**

- **链接: [https://arxiv.org/pdf/2506.17090v3](https://arxiv.org/pdf/2506.17090v3)**

> **作者:** Murtaza Nazir; Matthew Finlayson; John X. Morris; Xiang Ren; Swabha Swayamdipta
>
> **摘要:** Language model inversion seeks to recover hidden prompts using only language model outputs. This capability has implications for security and accountability in language model deployments, such as leaking private information from an API-protected language model's system message. We propose a new method -- prompt inversion from logprob sequences (PILS) -- that recovers hidden prompts by gleaning clues from the model's next-token probabilities over the course of multiple generation steps. Our method is enabled by a key insight: The vector-valued outputs of a language model occupy a low-dimensional subspace. This enables us to losslessly compress the full next-token probability distribution over multiple generation steps using a linear map, allowing more output information to be used for inversion. Our approach yields massive gains over previous state-of-the-art methods for recovering hidden prompts, achieving 2--3.5 times higher exact recovery rates across test sets, in one case increasing the recovery rate from 17% to 60%. Our method also exhibits surprisingly good generalization behavior; for instance, an inverter trained on 16 generations steps gets 5--27 points higher prompt recovery when we increase the number of steps to 32 at test time. Furthermore, we demonstrate strong performance of our method on the more challenging task of recovering hidden system messages. We also analyze the role of verbatim repetition in prompt recovery and propose a new method for cross-family model transfer for logit-based inverters. Our findings show that next-token probabilities are a considerably more vulnerable attack surface for inversion attacks than previously known.
>
---
#### [replaced 031] Beyond Over-Refusal: Scenario-Based Diagnostics and Post-Hoc Mitigation for Exaggerated Refusals in LLMs
- **分类: cs.CL**

- **简介: 该论文针对大语言模型对安全相关词汇的过度拒绝问题，构建了单轮与多轮评测基准，提出无需重训练的三种轻量级缓解方法，提升模型对良性请求的响应能力，同时保持安全防护。**

- **链接: [https://arxiv.org/pdf/2510.08158v2](https://arxiv.org/pdf/2510.08158v2)**

> **作者:** Shuzhou Yuan; Ercong Nie; Yinuo Sun; Chenxuan Zhao; William LaCroix; Michael Färber
>
> **备注:** Errors in the paper
>
> **摘要:** Large language models (LLMs) frequently produce false refusals, declining benign requests that contain terms resembling unsafe queries. We address this challenge by introducing two comprehensive benchmarks: the Exaggerated Safety Benchmark (XSB) for single-turn prompts, annotated with "Focus" keywords that identify refusal-inducing triggers, and the Multi-turn Scenario-based Exaggerated Safety Benchmark (MS-XSB), which systematically evaluates refusal calibration in realistic, context-rich dialog settings. Our benchmarks reveal that exaggerated refusals persist across diverse recent LLMs and are especially pronounced in complex, multi-turn scenarios. To mitigate these failures, we leverage post-hoc explanation methods to identify refusal triggers and deploy three lightweight, model-agnostic approaches, ignore-word instructions, prompt rephrasing, and attention steering, at inference time, all without retraining or parameter access. Experiments on four instruction-tuned Llama models demonstrate that these strategies substantially improve compliance on safe prompts while maintaining robust safety protections. Our findings establish a reproducible framework for diagnosing and mitigating exaggerated refusals, highlighting practical pathways to safer and more helpful LLM deployments.
>
---
#### [replaced 032] Can LLMs Detect Their Confabulations? Estimating Reliability in Uncertainty-Aware Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLM）能否检测自身生成的错误内容（虚构）。针对多轮对话中上下文误导导致的置信度与正确性错位问题，提出一种基于词元级不确定性的可靠性估计方法，通过聚合隐藏状态提升对不可靠输出的识别能力。**

- **链接: [https://arxiv.org/pdf/2508.08139v2](https://arxiv.org/pdf/2508.08139v2)**

> **作者:** Tianyi Zhou; Johanne Medina; Sanjay Chawla
>
> **摘要:** Large Language Models (LLMs) are prone to generating fluent but incorrect content, known as confabulation, which poses increasing risks in multi-turn or agentic applications where outputs may be reused as context. In this work, we investigate how in-context information influences model behavior and whether LLMs can identify their unreliable responses. We propose a reliability estimation that leverages token-level uncertainty to guide the aggregation of internal model representations. Specifically, we compute aleatoric and epistemic uncertainty from output logits to identify salient tokens and aggregate their hidden states into compact representations for response-level reliability prediction. Through controlled experiments on open QA benchmarks, we find that correct in-context information improves both answer accuracy and model confidence, while misleading context often induces confidently incorrect responses, revealing a misalignment between uncertainty and correctness. Our probing-based method captures these shifts in model behavior and improves the detection of unreliable outputs across multiple open-source LLMs. These results underscore the limitations of direct uncertainty signals and highlight the potential of uncertainty-guided probing for reliability-aware generation.
>
---
#### [replaced 033] ARE: Scaling Up Agent Environments and Evaluations
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ARE平台与Gaia2基准，旨在解决智能体环境构建与综合能力评估难题。通过可扩展的环境抽象和异步动态评测，支持复杂任务模拟与持续扩展，推动AI系统在真实场景中的全面发展。**

- **链接: [https://arxiv.org/pdf/2509.17158v2](https://arxiv.org/pdf/2509.17158v2)**

> **作者:** Romain Froger; Pierre Andrews; Matteo Bettini; Amar Budhiraja; Ricardo Silveira Cabral; Virginie Do; Emilien Garreau; Jean-Baptiste Gaya; Hugo Laurençon; Maxime Lecanu; Kunal Malkan; Dheeraj Mekala; Pierre Ménard; Gerard Moreno-Torres Bertran; Ulyana Piterbarg; Mikhail Plekhanov; Mathieu Rita; Andrey Rusakov; Vladislav Vorotilov; Mengjue Wang; Ian Yu; Amine Benhalloum; Grégoire Mialon; Thomas Scialom
>
> **备注:** Updated authors order and acknowledgement
>
> **摘要:** We introduce Meta Agents Research Environments (ARE), a research platform for scalable creation of environments, integration of synthetic or real applications, and execution of agentic orchestrations. ARE provides simple abstractions to build complex and diverse environments, each with their own rules, tools, content, and verifiers, helping to bridge the gap between model development and real-world deployment. We also propose Gaia2, a benchmark built in ARE and designed to measure general agent capabilities. Beyond search and execution, Gaia2 requires agents to handle ambiguities and noise, adapt to dynamic environments, collaborate with other agents, and operate under temporal constraints. Unlike prior benchmarks, Gaia2 runs asynchronously, surfacing new failure modes that are invisible in static settings. Our experiments show that no system dominates across the intelligence spectrum: stronger reasoning often comes at the cost of efficiency, and budget scaling curves plateau, highlighting the need for new architectures and adaptive compute strategies. Perhaps more importantly, ARE abstractions enable continuous extension of Gaia2 to other environments, empowering the community to rapidly create new benchmarks tailored to their domains. In AI's second half, progress increasingly depends on defining meaningful tasks and robust evaluations to drive frontier capabilities forward.
>
---
#### [replaced 034] SyGra: A Unified Graph-Based Framework for Scalable Generation, Quality Tagging, and Management of Synthetic Data
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出SyGra框架，属数据生成与管理任务，旨在解决大模型训练中高质量合成数据获取难的问题。通过图结构建模对话流，结合双阶段质量标注，实现可扩展、高保真的SFT与DPO数据生成与管理。**

- **链接: [https://arxiv.org/pdf/2508.15432v3](https://arxiv.org/pdf/2508.15432v3)**

> **作者:** Bidyapati Pradhan; Surajit Dasgupta; Amit Kumar Saha; Omkar Anustoop; Sriram Puttagunta; Vipul Mittal; Gopal Sarda
>
> **摘要:** The advancement of large language models (LLMs) is critically dependent on the availability of high-quality datasets for Supervised Fine-Tuning (SFT), alignment tasks like Direct Preference Optimization (DPO), etc. In this work, we present a comprehensive synthetic data generation framework that facilitates scalable, configurable, and high-fidelity generation of synthetic data tailored for these training paradigms. Our approach employs a modular and configuration-based pipeline capable of modeling complex dialogue flows with minimal manual intervention. This framework uses a dual-stage quality tagging mechanism, combining heuristic rules and LLM-based evaluations, to automatically filter and score data extracted from OASST-formatted conversations, ensuring the curation of high-quality dialogue samples. The resulting datasets are structured under a flexible schema supporting both SFT and DPO use cases, enabling seamless integration into diverse training workflows. Together, these innovations offer a robust solution for generating and managing synthetic conversational data at scale, significantly reducing the overhead of data preparation in LLM training pipelines.
>
---
#### [replaced 035] Anthropocentric bias in language model evaluation
- **分类: cs.CL**

- **简介: 该论文探讨语言模型评估中的人类中心偏见问题，指出忽略辅助因素影响和机械策略差异两类偏差。通过行为实验与机制研究结合，提出应以实证迭代方式准确评估模型认知能力。**

- **链接: [https://arxiv.org/pdf/2407.03859v3](https://arxiv.org/pdf/2407.03859v3)**

> **作者:** Raphaël Millière; Charles Rathkopf
>
> **备注:** Published in Computational Linguistics
>
> **摘要:** Evaluating the cognitive capacities of large language models (LLMs) requires overcoming not only anthropomorphic but also anthropocentric biases. This article identifies two types of anthropocentric bias that have been neglected: overlooking how auxiliary factors can impede LLM performance despite competence ("auxiliary oversight"), and dismissing LLM mechanistic strategies that differ from those of humans as not genuinely competent ("mechanistic chauvinism"). Mitigating these biases necessitates an empirically-driven, iterative approach to mapping cognitive tasks to LLM-specific capacities and mechanisms, which can be done by supplementing carefully designed behavioral experiments with mechanistic studies.
>
---
