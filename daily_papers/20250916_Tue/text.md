# 自然语言处理 cs.CL

- **最新发布 129 篇**

- **更新 81 篇**

## 最新发布

#### [new 001] CognitiveSky: Scalable Sentiment and Narrative Analysis for Decentralized Social Media
- **分类: cs.CL; cs.SI**

- **简介: 该论文提出CognitiveSky框架，用于在去中心化社交媒体Bluesky上进行情感与叙事分析。其任务是实时分析公众讨论，解决传统平台数据获取与分析的挑战。通过API采集数据，利用Transformer模型生成结构化输出，并构建可视化仪表盘，实现低成本、高可扩展性分析。**

- **链接: [http://arxiv.org/pdf/2509.11444v1](http://arxiv.org/pdf/2509.11444v1)**

> **作者:** Gaurab Chhetri; Anandi Dutta; Subasish Das
>
> **备注:** This is the author's preprint version of a paper accepted for presentation at HICSS 59 (Hawaii International Conference on System Sciences), 2026, Hawaii, USA. The final published version will appear in the official conference proceedings. Conference site: https://hicss.hawaii.edu/
>
> **摘要:** The emergence of decentralized social media platforms presents new opportunities and challenges for real-time analysis of public discourse. This study introduces CognitiveSky, an open-source and scalable framework designed for sentiment, emotion, and narrative analysis on Bluesky, a federated Twitter or X.com alternative. By ingesting data through Bluesky's Application Programming Interface (API), CognitiveSky applies transformer-based models to annotate large-scale user-generated content and produces structured and analyzable outputs. These summaries drive a dynamic dashboard that visualizes evolving patterns in emotion, activity, and conversation topics. Built entirely on free-tier infrastructure, CognitiveSky achieves both low operational cost and high accessibility. While demonstrated here for monitoring mental health discourse, its modular design enables applications across domains such as disinformation detection, crisis response, and civic sentiment analysis. By bridging large language models with decentralized networks, CognitiveSky offers a transparent, extensible tool for computational social science in an era of shifting digital ecosystems.
>
---
#### [new 002] Text2Mem: A Unified Memory Operation Language for Memory Operating System
- **分类: cs.CL**

- **简介: 论文提出Text2Mem，一种统一的内存操作语言，解决大语言模型代理在长期交互中缺乏标准化内存操作的问题。通过定义规范指令集、验证机制和适配器，实现安全、可移植的内存控制，为内存管理提供标准化基础。**

- **链接: [http://arxiv.org/pdf/2509.11145v1](http://arxiv.org/pdf/2509.11145v1)**

> **作者:** Felix Wang; Boyu Chen; Kerun Xu; Bo Tang; Feiyu Xiong; Zhiyu Li
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Large language model agents increasingly depend on memory to sustain long horizon interaction, but existing frameworks remain limited. Most expose only a few basic primitives such as encode, retrieve, and delete, while higher order operations like merge, promote, demote, split, lock, and expire are missing or inconsistently supported. Moreover, there is no formal and executable specification for memory commands, leaving scope and lifecycle rules implicit and causing unpredictable behavior across systems. We introduce Text2Mem, a unified memory operation language that provides a standardized pathway from natural language to reliable execution. Text2Mem defines a compact yet expressive operation set aligned with encoding, storage, and retrieval. Each instruction is represented as a JSON based schema instance with required fields and semantic invariants, which a parser transforms into typed operation objects with normalized parameters. A validator ensures correctness before execution, while adapters map typed objects either to a SQL prototype backend or to real memory frameworks. Model based services such as embeddings or summarization are integrated when required. All results are returned through a unified execution contract. This design ensures safety, determinism, and portability across heterogeneous backends. We also outline Text2Mem Bench, a planned benchmark that separates schema generation from backend execution to enable systematic evaluation. Together, these components establish the first standardized foundation for memory control in agents.
>
---
#### [new 003] Preservation of Language Understanding Capabilities in Speech-aware Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出C3T基准，用于评估语音感知大语言模型的语言理解能力保留情况。通过文本任务和语音合成模型，测试模型在语音输入下的性能、公平性及跨模态鲁棒性，属于模型评估任务。**

- **链接: [http://arxiv.org/pdf/2509.12171v1](http://arxiv.org/pdf/2509.12171v1)**

> **作者:** Marek Kubis; Paweł Skórzewski; Iwona Christop; Mateusz Czyżnikiewicz; Jakub Kubiak; Łukasz Bondaruk; Marcin Lewandowski
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** The paper presents C3T (Cross-modal Capabilities Conservation Test), a new benchmark for assessing the performance of speech-aware large language models. The benchmark utilizes textual tasks and a voice cloning text-to-speech model to quantify the extent to which language understanding capabilities are preserved when the model is accessed via speech input. C3T quantifies the fairness of the model for different categories of speakers and its robustness across text and speech modalities.
>
---
#### [new 004] XplaiNLP at CheckThat! 2025: Multilingual Subjectivity Detection with Finetuned Transformers and Prompt-Based Inference with Large Language Models
- **分类: cs.CL**

- **简介: 该论文参与CheckThat! 2025多语言主观性检测任务，提出两种方法：微调Transformer模型和基于提示的LLM推理。在多个语言子任务中取得优异成绩，尤其在意大利语和罗马尼亚语中表现突出。**

- **链接: [http://arxiv.org/pdf/2509.12130v1](http://arxiv.org/pdf/2509.12130v1)**

> **作者:** Ariana Sahitaj; Jiaao Li; Pia Wenzel Neves; Fedor Splitt; Premtim Sahitaj; Charlott Jakob; Veronika Solopova; Vera Schmitt
>
> **摘要:** This notebook reports the XplaiNLP submission to the CheckThat! 2025 shared task on multilingual subjectivity detection. We evaluate two approaches: (1) supervised fine-tuning of transformer encoders, EuroBERT, XLM-RoBERTa, and German-BERT, on monolingual and machine-translated training data; and (2) zero-shot prompting using two LLMs: o3-mini for Annotation (rule-based labelling) and gpt-4.1-mini for DoubleDown (contrastive rewriting) and Perspective (comparative reasoning). The Annotation Approach achieves 1st place in the Italian monolingual subtask with an F_1 score of 0.8104, outperforming the baseline of 0.6941. In the Romanian zero-shot setting, the fine-tuned XLM-RoBERTa model obtains an F_1 score of 0.7917, ranking 3rd and exceeding the baseline of 0.6461. The same model also performs reliably in the multilingual task and improves over the baseline in Greek. For German, a German-BERT model fine-tuned on translated training data from typologically related languages yields competitive performance over the baseline. In contrast, performance in the Ukrainian and Polish zero-shot settings falls slightly below the respective baselines, reflecting the challenge of generalization in low-resource cross-lingual scenarios.
>
---
#### [new 005] Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
- **分类: cs.CL**

- **简介: 论文提出OBR框架，解决LLM压缩中量化与稀疏化的冲突问题。通过误差补偿对齐两者，实现高效训练-free压缩，在保持性能的同时显著提升速度和减少内存占用。**

- **链接: [http://arxiv.org/pdf/2509.11177v1](http://arxiv.org/pdf/2509.11177v1)**

> **作者:** Hang Guo; Yawei Li; Luca Benini
>
> **备注:** Preprint
>
> **摘要:** Recent advances in Large Language Model (LLM) compression, such as quantization and pruning, have achieved notable success. However, as these techniques gradually approach their respective limits, relying on a single method for further compression has become increasingly challenging. In this work, we explore an alternative solution by combining quantization and sparsity. This joint approach, though promising, introduces new difficulties due to the inherently conflicting requirements on weight distributions: quantization favors compact ranges, while pruning benefits from high variance. To attack this problem, we propose Optimal Brain Restoration (OBR), a general and training-free framework that aligns pruning and quantization by error compensation between both. OBR minimizes performance degradation on downstream tasks by building on a second-order Hessian objective, which is then reformulated into a tractable problem through surrogate approximation and ultimately reaches a closed-form solution via group error compensation. Experiments show that OBR enables aggressive W4A4KV4 quantization with 50% sparsity on existing LLMs, and delivers up to 4.72x speedup and 6.4x memory reduction compared to the FP16-dense baseline.
>
---
#### [new 006] Reasoning Under Uncertainty: Exploring Probabilistic Reasoning Capabilities of LLMs
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在概率推理任务中的表现，探讨其处理显式离散概率分布的能力。通过设计三个任务评估模型的推理能力，发现模型性能随规模增大而提升，但也存在对符号表示敏感和上下文长度影响大的问题。**

- **链接: [http://arxiv.org/pdf/2509.10739v1](http://arxiv.org/pdf/2509.10739v1)**

> **作者:** Mobina Pournemat; Keivan Rezaei; Gaurang Sriramanan; Arman Zarei; Jiaxiang Fu; Yang Wang; Hamid Eghbalzadeh; Soheil Feizi
>
> **备注:** 25 pages, 4 figures, 6 tables
>
> **摘要:** Despite widespread success in language understanding and generation, large language models (LLMs) exhibit unclear and often inconsistent behavior when faced with tasks that require probabilistic reasoning. In this work, we present the first comprehensive study of the reasoning capabilities of LLMs over explicit discrete probability distributions. Given observations from a probability distribution, we evaluate models on three carefully designed tasks, mode identification, maximum likelihood estimation, and sample generation, by prompting them to provide responses to queries about either the joint distribution or its conditionals. These tasks thus probe a range of probabilistic skills, including frequency analysis, marginalization, and generative behavior. Through comprehensive empirical evaluations, we demonstrate that there exists a clear performance gap between smaller and larger models, with the latter demonstrating stronger inference and surprising capabilities in sample generation. Furthermore, our investigations reveal notable limitations, including sensitivity to variations in the notation utilized to represent probabilistic outcomes and performance degradation of over 60% as context length increases. Together, our results provide a detailed understanding of the probabilistic reasoning abilities of LLMs and identify key directions for future improvement.
>
---
#### [new 007] No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes
- **分类: cs.CL; cs.AI**

- **简介: 论文研究大语言模型在读取问题后能否预测自身回答的准确性。通过训练线性探针，利用问题激活状态预测答案正确性，发现中间层具有最强预测能力，并揭示了模型自我评估机制及对数学推理的局限性。属于模型内部机制分析任务。**

- **链接: [http://arxiv.org/pdf/2509.10625v1](http://arxiv.org/pdf/2509.10625v1)**

> **作者:** Iván Vicente Moreno Cencerrado; Arnau Padrés Masdemont; Anton Gonzalvez Hawthorne; David Demitri Africa; Lorenzo Pacchiardi
>
> **摘要:** Do large language models (LLMs) anticipate when they will answer correctly? To study this, we extract activations after a question is read but before any tokens are generated, and train linear probes to predict whether the model's forthcoming answer will be correct. Across three open-source model families ranging from 7 to 70 billion parameters, projections on this "in-advance correctness direction" trained on generic trivia questions predict success in distribution and on diverse out-of-distribution knowledge datasets, outperforming black-box baselines and verbalised predicted confidence. Predictive power saturates in intermediate layers, suggesting that self-assessment emerges mid-computation. Notably, generalisation falters on questions requiring mathematical reasoning. Moreover, for models responding "I don't know", doing so strongly correlates with the probe score, indicating that the same direction also captures confidence. By complementing previous results on truthfulness and other behaviours obtained with probes and sparse auto-encoders, our work contributes essential findings to elucidate LLM internals.
>
---
#### [new 008] Pun Unintended: LLMs and the Illusion of Humor Understanding
- **分类: cs.CL; cs.AI; 68T50; I.2.7**

- **简介: 论文研究大语言模型（LLMs）对双关语的理解能力，指出其理解浅显，缺乏人类的细腻把握。通过构建新基准和实验分析，揭示LLMs在处理双关语时的鲁棒性问题。属于自然语言理解任务，旨在提升模型对幽默语言的深层理解能力。**

- **链接: [http://arxiv.org/pdf/2509.12158v1](http://arxiv.org/pdf/2509.12158v1)**

> **作者:** Alessandro Zangari; Matteo Marcuzzo; Andrea Albarelli; Mohammad Taher Pilehvar; Jose Camacho-Collados
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Puns are a form of humorous wordplay that exploits polysemy and phonetic similarity. While LLMs have shown promise in detecting puns, we show in this paper that their understanding often remains shallow, lacking the nuanced grasp typical of human interpretation. By systematically analyzing and reformulating existing pun benchmarks, we demonstrate how subtle changes in puns are sufficient to mislead LLMs. Our contributions include comprehensive and nuanced pun detection benchmarks, human evaluation across recent LLMs, and an analysis of the robustness challenges these models face in processing puns.
>
---
#### [new 009] In-domain SSL pre-training and streaming ASR
- **分类: cs.CL; cs.AI**

- **简介: 论文研究在航空管制（ATC）场景下，利用领域自监督预训练提升离线与流式语音识别（ASR）性能。通过BEST-RQ模型在4.5k小时未标注数据上预训练，并引入分块注意力和动态卷积实现低延迟推理，显著降低词错误率，适用于实时航空安全应用。**

- **链接: [http://arxiv.org/pdf/2509.12101v1](http://arxiv.org/pdf/2509.12101v1)**

> **作者:** Jarod Duret; Salima Mdhaffar; Gaëlle Laperrière; Ryan Whetten; Audrey Galametz; Catherine Kobus; Marion-Cécile Martin; Jo Oleiwan; Yannick Estève
>
> **备注:** Accepted to SPECOM 2025
>
> **摘要:** In this study, we investigate the benefits of domain-specific self-supervised pre-training for both offline and streaming ASR in Air Traffic Control (ATC) environments. We train BEST-RQ models on 4.5k hours of unlabeled ATC data, then fine-tune on a smaller supervised ATC set. To enable real-time processing, we propose using chunked attention and dynamic convolutions, ensuring low-latency inference. We compare these in-domain SSL models against state-of-the-art, general-purpose speech encoders such as w2v-BERT 2.0 and HuBERT. Results show that domain-adapted pre-training substantially improves performance on standard ATC benchmarks, significantly reducing word error rates when compared to models trained on broad speech corpora. Furthermore, the proposed streaming approach further improves word error rate under tighter latency constraints, making it particularly suitable for safety-critical aviation applications. These findings highlight that specializing SSL representations for ATC data is a practical path toward more accurate and efficient ASR systems in real-world operational settings.
>
---
#### [new 010] HARP: Hallucination Detection via Reasoning Subspace Projection
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出HARP框架，用于检测大语言模型中的幻觉。通过分解隐藏状态空间为语义与推理子空间，并利用SVD提取基向量进行投影，降低噪声并提升鲁棒性。实验表明其在多个数据集上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2509.11536v1](http://arxiv.org/pdf/2509.11536v1)**

> **作者:** Junjie Hu; Gang Tu; ShengYu Cheng; Jinxin Li; Jinting Wang; Rui Chen; Zhilong Zhou; Dongbo Shan
>
> **摘要:** Hallucinations in Large Language Models (LLMs) pose a major barrier to their reliable use in critical decision-making. Although existing hallucination detection methods have improved accuracy, they still struggle with disentangling semantic and reasoning information and maintaining robustness. To address these challenges, we propose HARP (Hallucination detection via reasoning subspace projection), a novel hallucination detection framework. HARP establishes that the hidden state space of LLMs can be decomposed into a direct sum of a semantic subspace and a reasoning subspace, where the former encodes linguistic expression and the latter captures internal reasoning processes. Moreover, we demonstrate that the Unembedding layer can disentangle these subspaces, and by applying Singular Value Decomposition (SVD) to its parameters, the basis vectors spanning the semantic and reasoning subspaces are obtained. Finally, HARP projects hidden states onto the basis vectors of the reasoning subspace, and the resulting projections are then used as input features for hallucination detection in LLMs. By using these projections, HARP reduces the dimension of the feature to approximately 5% of the original, filters out most noise, and achieves enhanced robustness. Experiments across multiple datasets show that HARP achieves state-of-the-art hallucination detection performance; in particular, it achieves an AUROC of 92.8% on TriviaQA, outperforming the previous best method by 7.5%.
>
---
#### [new 011] A Survey on Retrieval And Structuring Augmented Generation with Large Language Models
- **分类: cs.CL**

- **简介: 该论文综述了基于大语言模型的检索与结构增强生成方法，旨在解决幻觉、知识过时等问题。研究涵盖检索机制、文本结构化技术及与LLM的融合方法，分析挑战并指出未来方向。属于自然语言处理任务。**

- **链接: [http://arxiv.org/pdf/2509.10697v1](http://arxiv.org/pdf/2509.10697v1)**

> **作者:** Pengcheng Jiang; Siru Ouyang; Yizhu Jiao; Ming Zhong; Runchu Tian; Jiawei Han
>
> **备注:** KDD'25 survey track
>
> **摘要:** Large Language Models (LLMs) have revolutionized natural language processing with their remarkable capabilities in text generation and reasoning. However, these models face critical challenges when deployed in real-world applications, including hallucination generation, outdated knowledge, and limited domain expertise. Retrieval And Structuring (RAS) Augmented Generation addresses these limitations by integrating dynamic information retrieval with structured knowledge representations. This survey (1) examines retrieval mechanisms including sparse, dense, and hybrid approaches for accessing external knowledge; (2) explore text structuring techniques such as taxonomy construction, hierarchical classification, and information extraction that transform unstructured text into organized representations; and (3) investigate how these structured representations integrate with LLMs through prompt-based methods, reasoning frameworks, and knowledge embedding techniques. It also identifies technical challenges in retrieval efficiency, structure quality, and knowledge integration, while highlighting research opportunities in multimodal retrieval, cross-lingual structures, and interactive systems. This comprehensive overview provides researchers and practitioners with insights into RAS methods, applications, and future directions.
>
---
#### [new 012] Designing LLMs for cultural sensitivity: Evidence from English-Japanese translation
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 论文研究大语言模型在英日职场邮件翻译中的文化适应性。任务是评估不同提示策略对文化敏感度的影响。通过混合方法分析，发现定制化提示可提升文化契合度，为设计包容性多语LLMs提供建议。**

- **链接: [http://arxiv.org/pdf/2509.11921v1](http://arxiv.org/pdf/2509.11921v1)**

> **作者:** Helene Tenzer; Oumnia Abidi; Stefan Feuerriegel
>
> **摘要:** Large language models (LLMs) are increasingly used in everyday communication, including multilingual interactions across different cultural contexts. While LLMs can now generate near-perfect literal translations, it remains unclear whether LLMs support culturally appropriate communication. In this paper, we analyze the cultural sensitivity of different LLM designs when applied to English-Japanese translations of workplace e-mails. Here, we vary the prompting strategies: (1) naive "just translate" prompts, (2) audience-targeted prompts specifying the recipient's cultural background, and (3) instructional prompts with explicit guidance on Japanese communication norms. Using a mixed-methods study, we then analyze culture-specific language patterns to evaluate how well translations adapt to cultural norms. Further, we examine the appropriateness of the tone of the translations as perceived by native speakers. We find that culturally-tailored prompting can improve cultural fit, based on which we offer recommendations for designing culturally inclusive LLMs in multilingual settings.
>
---
#### [new 013] CultureSynth: A Hierarchical Taxonomy-Guided and Retrieval-Augmented Framework for Cultural Question-Answer Synthesis
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CultureSynth框架，用于合成文化相关问答对，解决现有文化评估数据碎片化、依赖人工标注的问题。构建多语言文化分类体系，并利用检索增强生成技术，生成跨语言文化问答基准，提升大模型文化理解能力。**

- **链接: [http://arxiv.org/pdf/2509.10886v1](http://arxiv.org/pdf/2509.10886v1)**

> **作者:** Xinyu Zhang; Pei Zhang; Shuang Luo; Jialong Tang; Yu Wan; Baosong Yang; Fei Huang
>
> **备注:** Accepted as a Findings paper at EMNLP 2025
>
> **摘要:** Cultural competence, defined as the ability to understand and adapt to multicultural contexts, is increasingly vital for large language models (LLMs) in global environments. While several cultural benchmarks exist to assess LLMs' cultural competence, current evaluations suffer from fragmented taxonomies, domain specificity, and heavy reliance on manual data annotation. To address these limitations, we introduce CultureSynth, a novel framework comprising (1) a comprehensive hierarchical multilingual cultural taxonomy covering 12 primary and 130 secondary topics, and (2) a Retrieval-Augmented Generation (RAG)-based methodology leveraging factual knowledge to synthesize culturally relevant question-answer pairs. The CultureSynth-7 synthetic benchmark contains 19,360 entries and 4,149 manually verified entries across 7 languages. Evaluation of 14 prevalent LLMs of different sizes reveals clear performance stratification led by ChatGPT-4o-Latest and Qwen2.5-72B-Instruct. The results demonstrate that a 3B-parameter threshold is necessary for achieving basic cultural competence, models display varying architectural biases in knowledge processing, and significant geographic disparities exist across models. We believe that CultureSynth offers a scalable framework for developing culturally aware AI systems while reducing reliance on manual annotation\footnote{Benchmark is available at https://github.com/Eyr3/CultureSynth.}.
>
---
#### [new 014] ToolRM: Outcome Reward Models for Tool-Calling Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出ToolRM，解决工具调用大模型的奖励建模问题。通过构建FC-RewardBench基准和基于结果的奖励模型训练框架，提升模型在工具使用场景下的性能，实现下游任务平均25%的性能提升。**

- **链接: [http://arxiv.org/pdf/2509.11963v1](http://arxiv.org/pdf/2509.11963v1)**

> **作者:** Mayank Agarwal; Ibrahim Abdelaziz; Kinjal Basu; Merve Unuvar; Luis A. Lastras; Yara Rizk; Pavan Kapanipathi
>
> **摘要:** As large language models (LLMs) increasingly interact with external tools, reward modeling for tool use has become a critical yet underexplored area. Existing reward models, trained primarily on natural language outputs, struggle to evaluate tool-based reasoning and execution. To quantify this gap, we introduce FC-RewardBench, the first benchmark designed to systematically assess reward models' performance in tool-calling scenarios. Our analysis shows that current reward models often miss key signals of effective tool use, highlighting the need for domain-specific modeling. To address this, we propose a training framework for outcome-based reward models using data synthesized from permissively licensed, open-weight LLMs. We train models ranging from 1.7B to 14B parameters and evaluate them across seven out-of-domain benchmarks. These models consistently outperform general-purpose baselines, achieving up to 25\% average improvement in downstream task performance and enabling data-efficient fine-tuning through reward-guided filtering.
>
---
#### [new 015] GTA: Supervised-Guided Reinforcement Learning for Text Classification with Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出GTA框架，解决文本分类中SFT与RL的效率与性能权衡问题。结合SFT的高效与RL的能力提升，通过猜-思-答流程及奖励机制，加速收敛并提升性能。**

- **链接: [http://arxiv.org/pdf/2509.12108v1](http://arxiv.org/pdf/2509.12108v1)**

> **作者:** Min Zeng; Jinfei Sun; Xueyou Luo; Caiquan Liu; Shiqi Zhang; Li Xie; Xiaoxin Chen
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** In natural language processing tasks, pure reinforcement learning (RL) fine-tuning methods often suffer from inefficient exploration and slow convergence; while supervised fine-tuning (SFT) methods, although efficient in training, have limited performance ceiling and less solid theoretical foundation compared to RL. To address efficiency-capability trade-off, we propose the Guess-Think-Answer (GTA) framework that combines the efficiency of SFT with the capability gains of RL in a unified training paradigm. GTA works by having the model first produce a provisional guess (optimized via cross-entropy loss), then reflect on this guess before generating the final answer, with RL rewards shaping both the final output and the format of the entire GTA structure. This hybrid approach achieves both faster convergence than pure RL and higher performance ceiling than pure SFT. To mitigate gradient conflicts between the two training signals, we employ loss masking and gradient constraints. Empirical results on four text classification benchmarks demonstrate that GTA substantially accelerates convergence while outperforming both standalone SFT and RL baselines.
>
---
#### [new 016] Struct-Bench: A Benchmark for Differentially Private Structured Text Generation
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Struct-Bench，用于评估差分隐私结构化文本生成。任务是解决现有方法难以评估结构化数据合成质量的问题。工作包括构建包含CFG的基准数据集、设计评估指标及发布排行榜。**

- **链接: [http://arxiv.org/pdf/2509.10696v1](http://arxiv.org/pdf/2509.10696v1)**

> **作者:** Shuaiqi Wang; Vikas Raunak; Arturs Backurs; Victor Reis; Pei Zhou; Sihao Chen; Longqi Yang; Zinan Lin; Sergey Yekhanin; Giulia Fanti
>
> **摘要:** Differentially private (DP) synthetic data generation is a promising technique for utilizing private datasets that otherwise cannot be exposed for model training or other analytics. While much research literature has focused on generating private unstructured text and image data, in enterprise settings, structured data (e.g., tabular) is more common, often including natural language fields or components. Existing synthetic data evaluation techniques (e.g., FID) struggle to capture the structural properties and correlations of such datasets. In this work, we propose Struct-Bench, a framework and benchmark for evaluating synthetic datasets derived from structured datasets that contain natural language data. The Struct-Bench framework requires users to provide a representation of their dataset structure as a Context-Free Grammar (CFG). Our benchmark comprises 5 real-world and 2 synthetically generated datasets, each annotated with CFGs. We show that these datasets demonstrably present a great challenge even for state-of-the-art DP synthetic data generation methods. Struct-Bench also includes reference implementations of different metrics and a leaderboard, thereby providing researchers a standardized evaluation platform to benchmark and investigate privacy-preserving synthetic data generation methods. Further, we also present a case study showing how to use Struct-Bench to improve the synthetic data quality of Private Evolution (PE) on structured data. The benchmark and the leaderboard have been publicly made available at https://struct-bench.github.io.
>
---
#### [new 017] A Transformer-Based Cross-Platform Analysis of Public Discourse on the 15-Minute City Paradigm
- **分类: cs.CL; cs.SI**

- **简介: 该论文进行跨平台情感分析，研究公众对“15分钟城市”概念的看法。任务是分类社交媒体和新闻中的情感倾向，解决多源文本异构性问题，采用压缩Transformer模型实现高效、可复现的跨平台分析。**

- **链接: [http://arxiv.org/pdf/2509.11443v1](http://arxiv.org/pdf/2509.11443v1)**

> **作者:** Gaurab Chhetri; Darrell Anderson; Boniphace Kutela; Subasish Das
>
> **备注:** This is the author's preprint version of a paper accepted for presentation at the 24th International Conference on Machine Learning and Applications (ICMLA 2025), December 3-5, 2025, Florida, USA. The final published version will appear in the official IEEE proceedings. Conference site: https://www.icmla-conference.org/icmla25/
>
> **摘要:** This study presents the first multi-platform sentiment analysis of public opinion on the 15-minute city concept across Twitter, Reddit, and news media. Using compressed transformer models and Llama-3-8B for annotation, we classify sentiment across heterogeneous text domains. Our pipeline handles long-form and short-form text, supports consistent annotation, and enables reproducible evaluation. We benchmark five models (DistilRoBERTa, DistilBERT, MiniLM, ELECTRA, TinyBERT) using stratified 5-fold cross-validation, reporting F1-score, AUC, and training time. DistilRoBERTa achieved the highest F1 (0.8292), TinyBERT the best efficiency, and MiniLM the best cross-platform consistency. Results show News data yields inflated performance due to class imbalance, Reddit suffers from summarization loss, and Twitter offers moderate challenge. Compressed models perform competitively, challenging assumptions that larger models are necessary. We identify platform-specific trade-offs and propose directions for scalable, real-world sentiment classification in urban planning discourse.
>
---
#### [new 018] SCDTour: Embedding Axis Ordering and Merging for Interpretable Semantic Change Detection
- **分类: cs.CL**

- **简介: 该论文提出SCDTour方法，解决语义变化检测（SCD）中模型性能与可解释性难以兼顾的问题。通过排序和合并可解释轴，提升模型性能并保持高可解释性，实现语义变化的有意义解释。**

- **链接: [http://arxiv.org/pdf/2509.11818v1](http://arxiv.org/pdf/2509.11818v1)**

> **作者:** Taichi Aida; Danushka Bollegala
>
> **备注:** Findings of EMNLP2025
>
> **摘要:** In Semantic Change Detection (SCD), it is a common problem to obtain embeddings that are both interpretable and high-performing. However, improving interpretability often leads to a loss in the SCD performance, and vice versa. To address this problem, we propose SCDTour, a method that orders and merges interpretable axes to alleviate the performance degradation of SCD. SCDTour considers both (a) semantic similarity between axes in the embedding space, as well as (b) the degree to which each axis contributes to semantic change. Experimental results show that SCDTour preserves performance in semantic change detection while maintaining high interpretability. Moreover, agglomerating the sorted axes produces a more refined set of word senses, which achieves comparable or improved performance against the original full-dimensional embeddings in the SCD task. These findings demonstrate that SCDTour effectively balances interpretability and SCD performance, enabling meaningful interpretation of semantic shifts through a small number of refined axes. Source code is available at https://github.com/LivNLP/svp-tour .
>
---
#### [new 019] Analyzing Information-Seeking Behaviors in a Hakka AI Chatbot: A Cognitive-Pragmatic Study
- **分类: cs.CL**

- **简介: 该论文研究用户在Hakka AI聊天机器人TALKA中的信息搜索行为，结合认知与语用分析框架，分析7,077条用户语句，探讨AI如何支持濒危语言学习与文化认同。任务是探索AI辅助语言学习的有效性及对语言保护的贡献。**

- **链接: [http://arxiv.org/pdf/2509.11591v1](http://arxiv.org/pdf/2509.11591v1)**

> **作者:** Chu-Hsuan Lee; Chen-Chi Chang; Hung-Shin Lee; Yun-Hsiang Hsu; Ching-Yuan Chen
>
> **备注:** Accepted to HICSS-59 (2026)
>
> **摘要:** With many endangered languages at risk of disappearing, efforts to preserve them now rely more than ever on using technology alongside culturally informed teaching strategies. This study examines user behaviors in TALKA, a generative AI-powered chatbot designed for Hakka language engagement, by employing a dual-layered analytical framework grounded in Bloom's Taxonomy of cognitive processes and dialogue act categorization. We analyzed 7,077 user utterances, each carefully annotated according to six cognitive levels and eleven dialogue act types. These included a variety of functions, such as asking for information, requesting translations, making cultural inquiries, and using language creatively. Pragmatic classifications further highlight how different types of dialogue acts--such as feedback, control commands, and social greetings--align with specific cognitive intentions. The results suggest that generative AI chatbots can support language learning in meaningful ways--especially when they are designed with an understanding of how users think and communicate. They may also help learners express themselves more confidently and connect with their cultural identity. The TALKA case provides empirical insights into how AI-mediated dialogue facilitates cognitive development in low-resource language learners, as well as pragmatic negotiation and socio-cultural affiliation. By focusing on AI-assisted language learning, this study offers new insights into how technology can support language preservation and educational practice.
>
---
#### [new 020] Bhaasha, Bhasa, Zaban: A Survey for Low-Resourced Languages in South Asia -- Current Stage and Challenges
- **分类: cs.CL**

- **简介: 该论文综述了南亚低资源语言的NLP研究现状与挑战，聚焦数据、模型与任务三方面，指出数据缺失、代码混合和评估基准不足等问题，旨在推动更公平的语言模型发展。**

- **链接: [http://arxiv.org/pdf/2509.11570v1](http://arxiv.org/pdf/2509.11570v1)**

> **作者:** Sampoorna Poria; Xiaolei Huang
>
> **摘要:** Rapid developments of large language models have revolutionized many NLP tasks for English data. Unfortunately, the models and their evaluations for low-resource languages are being overlooked, especially for languages in South Asia. Although there are more than 650 languages in South Asia, many of them either have very limited computational resources or are missing from existing language models. Thus, a concrete question to be answered is: Can we assess the current stage and challenges to inform our NLP community and facilitate model developments for South Asian languages? In this survey, we have comprehensively examined current efforts and challenges of NLP models for South Asian languages by retrieving studies since 2020, with a focus on transformer-based models, such as BERT, T5, & GPT. We present advances and gaps across 3 essential aspects: data, models, & tasks, such as available data sources, fine-tuning strategies, & domain applications. Our findings highlight substantial issues, including missing data in critical domains (e.g., health), code-mixing, and lack of standardized evaluation benchmarks. Our survey aims to raise awareness within the NLP community for more targeted data curation, unify benchmarks tailored to cultural and linguistic nuances of South Asia, and encourage an equitable representation of South Asian languages. The complete list of resources is available at: https://github.com/trust-nlp/LM4SouthAsia-Survey.
>
---
#### [new 021] The Prompt Engineering Report Distilled: Quick Start Guide for Life Sciences
- **分类: cs.CL**

- **简介: 该论文总结2025年《Prompt Report》中的6种核心提示工程技巧，旨在为生命科学领域提供高效、系统的提示设计指南，解决LLM应用中效率低、质量差的问题，提升研究质量。**

- **链接: [http://arxiv.org/pdf/2509.11295v1](http://arxiv.org/pdf/2509.11295v1)**

> **作者:** Valentin Romanov; Steven A Niederer
>
> **摘要:** Developing effective prompts demands significant cognitive investment to generate reliable, high-quality responses from Large Language Models (LLMs). By deploying case-specific prompt engineering techniques that streamline frequently performed life sciences workflows, researchers could achieve substantial efficiency gains that far exceed the initial time investment required to master these techniques. The Prompt Report published in 2025 outlined 58 different text-based prompt engineering techniques, highlighting the numerous ways prompts could be constructed. To provide actionable guidelines and reduce the friction of navigating these various approaches, we distil this report to focus on 6 core techniques: zero-shot, few-shot approaches, thought generation, ensembling, self-criticism, and decomposition. We breakdown the significance of each approach and ground it in use cases relevant to life sciences, from literature summarization and data extraction to editorial tasks. We provide detailed recommendations for how prompts should and shouldn't be structured, addressing common pitfalls including multi-turn conversation degradation, hallucinations, and distinctions between reasoning and non-reasoning models. We examine context window limitations, agentic tools like Claude Code, while analyzing the effectiveness of Deep Research tools across OpenAI, Google, Anthropic and Perplexity platforms, discussing current limitations. We demonstrate how prompt engineering can augment rather than replace existing established individual practices around data processing and document editing. Our aim is to provide actionable guidance on core prompt engineering principles, and to facilitate the transition from opportunistic prompting to an effective, low-friction systematic practice that contributes to higher quality research.
>
---
#### [new 022] Towards Automated Error Discovery: A Study in Conversational AI
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 论文提出Automated Error Discovery框架及SEEED方法，用于检测对话AI中的错误，解决LLM难以识别未明确指定错误的问题，提升未知错误检测准确率。属于错误检测任务。**

- **链接: [http://arxiv.org/pdf/2509.10833v1](http://arxiv.org/pdf/2509.10833v1)**

> **作者:** Dominic Petrak; Thy Thy Tran; Iryna Gurevych
>
> **备注:** Accepted to EMNLP 2025 main conference
>
> **摘要:** Although LLM-based conversational agents demonstrate strong fluency and coherence, they still produce undesirable behaviors (errors) that are challenging to prevent from reaching users during deployment. Recent research leverages large language models (LLMs) to detect errors and guide response-generation models toward improvement. However, current LLMs struggle to identify errors not explicitly specified in their instructions, such as those arising from updates to the response-generation model or shifts in user behavior. In this work, we introduce Automated Error Discovery, a framework for detecting and defining errors in conversational AI, and propose SEEED (Soft Clustering Extended Encoder-Based Error Detection), as an encoder-based approach to its implementation. We enhance the Soft Nearest Neighbor Loss by amplifying distance weighting for negative samples and introduce Label-Based Sample Ranking to select highly contrastive examples for better representation learning. SEEED outperforms adapted baselines -- including GPT-4o and Phi-4 -- across multiple error-annotated dialogue datasets, improving the accuracy for detecting unknown errors by up to 8 points and demonstrating strong generalization to unknown intent detection.
>
---
#### [new 023] RAGs to Riches: RAG-like Few-shot Learning for Large Language Model Role-playing
- **分类: cs.CL; cs.AI**

- **简介: 论文提出RAGs-to-Riches框架，解决LLM角色扮演中易偏离设定的问题。通过检索增强生成方法，利用参考示例提升角色一致性，实验显示其在对抗性交互中表现更优。属于自然语言处理中的角色扮演任务。**

- **链接: [http://arxiv.org/pdf/2509.12168v1](http://arxiv.org/pdf/2509.12168v1)**

> **作者:** Timothy Rupprecht; Enfu Nan; Arash Akbari; Arman Akbari; Lei Lu; Priyanka Maan; Sean Duffy; Pu Zhao; Yumei He; David Kaeli; Yanzhi Wang
>
> **摘要:** Role-playing Large language models (LLMs) are increasingly deployed in high-stakes domains such as healthcare, education, and governance, where failures can directly impact user trust and well-being. A cost effective paradigm for LLM role-playing is few-shot learning, but existing approaches often cause models to break character in unexpected and potentially harmful ways, especially when interacting with hostile users. Inspired by Retrieval-Augmented Generation (RAG), we reformulate LLM role-playing into a text retrieval problem and propose a new prompting framework called RAGs-to-Riches, which leverages curated reference demonstrations to condition LLM responses. We evaluate our framework with LLM-as-a-judge preference voting and introduce two novel token-level ROUGE metrics: Intersection over Output (IOO) to quantity how much an LLM improvises and Intersection over References (IOR) to measure few-shot demonstrations utilization rate during the evaluation tasks. When simulating interactions with a hostile user, our prompting strategy incorporates in its responses during inference an average of 35% more tokens from the reference demonstrations. As a result, across 453 role-playing interactions, our models are consistently judged as being more authentic, and remain in-character more often than zero-shot and in-context Learning (ICL) methods. Our method presents a scalable strategy for building robust, human-aligned LLM role-playing frameworks.
>
---
#### [new 024] D$^2$HScore: Reasoning-Aware Hallucination Detection via Semantic Breadth and Depth Analysis in LLMs
- **分类: cs.CL**

- **简介: 该论文提出D$^2$HScore，用于检测大语言模型生成中的幻觉。通过分析语义广度和深度，从模型结构和生成动态角度解决输出不可靠问题，实现无需训练和标签的轻量检测方法。**

- **链接: [http://arxiv.org/pdf/2509.11569v1](http://arxiv.org/pdf/2509.11569v1)**

> **作者:** Yue Ding; Xiaofang Zhu; Tianze Xia; Junfei Wu; Xinlong Chen; Qiang Liu; Liang Wang
>
> **备注:** under review
>
> **摘要:** Although large Language Models (LLMs) have achieved remarkable success, their practical application is often hindered by the generation of non-factual content, which is called "hallucination". Ensuring the reliability of LLMs' outputs is a critical challenge, particularly in high-stakes domains such as finance, security, and healthcare. In this work, we revisit hallucination detection from the perspective of model architecture and generation dynamics. Leveraging the multi-layer structure and autoregressive decoding process of LLMs, we decompose hallucination signals into two complementary dimensions: the semantic breadth of token representations within each layer, and the semantic depth of core concepts as they evolve across layers. Based on this insight, we propose \textbf{D$^2$HScore (Dispersion and Drift-based Hallucination Score)}, a training-free and label-free framework that jointly measures: (1) \textbf{Intra-Layer Dispersion}, which quantifies the semantic diversity of token representations within each layer; and (2) \textbf{Inter-Layer Drift}, which tracks the progressive transformation of key token representations across layers. To ensure drift reflects the evolution of meaningful semantics rather than noisy or redundant tokens, we guide token selection using attention signals. By capturing both the horizontal and vertical dynamics of representation during inference, D$^2$HScore provides an interpretable and lightweight proxy for hallucination detection. Extensive experiments across five open-source LLMs and five widely used benchmarks demonstrate that D$^2$HScore consistently outperforms existing training-free baselines.
>
---
#### [new 025] User eXperience Perception Insights Dataset (UXPID): Synthetic User Feedback from Public Industrial Forums
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出UXPID数据集，通过合成工业论坛用户反馈解决传统分析方法在处理非结构化、领域特定内容时的困难。利用大语言模型进行标注，支持UX分析与AI反馈处理研究。**

- **链接: [http://arxiv.org/pdf/2509.11777v1](http://arxiv.org/pdf/2509.11777v1)**

> **作者:** Mikhail Kulyabin; Jan Joosten; Choro Ulan uulu; Nuno Miguel Martins Pacheco; Fabian Ries; Filippos Petridis; Jan Bosch; Helena Holmström Olsson
>
> **摘要:** Customer feedback in industrial forums reflect a rich but underexplored source of insight into real-world product experience. These publicly shared discussions offer an organic view of user expectations, frustrations, and success stories shaped by the specific contexts of use. Yet, harnessing this information for systematic analysis remains challenging due to the unstructured and domain-specific nature of the content. The lack of structure and specialized vocabulary makes it difficult for traditional data analysis techniques to accurately interpret, categorize, and quantify the feedback, thereby limiting its potential to inform product development and support strategies. To address these challenges, this paper presents the User eXperience Perception Insights Dataset (UXPID), a collection of 7130 artificially synthesized and anonymized user feedback branches extracted from a public industrial automation forum. Each JavaScript object notation (JSON) record contains multi-post comments related to specific hardware and software products, enriched with metadata and contextual conversation data. Leveraging a large language model (LLM), each branch is systematically analyzed and annotated for UX insights, user expectations, severity and sentiment ratings, and topic classifications. The UXPID dataset is designed to facilitate research in user requirements, user experience (UX) analysis, and AI-driven feedback processing, particularly where privacy and licensing restrictions limit access to real-world data. UXPID supports the training and evaluation of transformer-based models for tasks such as issue detection, sentiment analysis, and requirements extraction in the context of technical forums.
>
---
#### [new 026] Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Judge Q方法，通过训练可学习的查询 tokens 来优化KV缓存淘汰策略，提升信息保留效果。属于LLM缓存管理任务，解决传统方法忽略全局信息的问题，提升解码质量。**

- **链接: [http://arxiv.org/pdf/2509.10798v1](http://arxiv.org/pdf/2509.10798v1)**

> **作者:** Yijun Liu; Yixuan Wang; Yuzhuang Xu; Shiyu Ji; Yang Xu; Qingfu Zhu; Wanxiang Che
>
> **备注:** preprint
>
> **摘要:** Large language models (LLMs) utilize key-value (KV) cache to store historical information during sequence processing. The size of KV cache grows linearly as the length of the sequence extends, which seriously affects memory usage and decoding efficiency. Current methods for KV cache eviction typically utilize the last window from the pre-filling phase as queries to compute the KV importance scores for eviction. Although this scheme is simple to implement, it tends to overly focus on local information, potentially leading to the neglect or omission of crucial global information. To mitigate this issue, we propose Judge Q, a novel training method which incorporates a soft token list. This method only tunes the model's embedding layer at a low training cost. By concatenating the soft token list at the end of the input sequence, we train these tokens' attention map to the original input sequence to align with that of the actual decoded tokens. In this way, the queries corresponding to the soft tokens can effectively capture global information and better evaluate the importance of the keys and values within the KV cache, thus maintaining decoding quality when KV cache is evicted. Under the same eviction budget, our method exhibits less performance degradation compared to existing eviction approaches. We validate our approach through experiments conducted on models such as Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3, using benchmarks including LongBench, RULER, and Needle-in-a-Haystack. Results indicate an improvement of approximately 1 point on the LongBench and over 3 points on RULER. This proposed methodology can be seamlessly integrated into existing open-source models with minimal training overhead, thereby enhancing performance in KV cache eviction scenarios.
>
---
#### [new 027] Quantifier Scope Interpretation in Language Learners and LLMs
- **分类: cs.CL**

- **简介: 该论文研究LLMs在处理英语和汉语量化词范围解释时的表现，探讨其与人类的相似性。通过概率和HS分数评估模型表现，发现多数LLMs偏好表层解释，部分体现语言差异，模型结构和预训练数据对其影响显著。任务为量化词歧义解析，解决跨语言模型表现差异问题。**

- **链接: [http://arxiv.org/pdf/2509.10860v1](http://arxiv.org/pdf/2509.10860v1)**

> **作者:** Shaohua Fang; Yue Li; Yan Cong
>
> **摘要:** Sentences with multiple quantifiers often lead to interpretive ambiguities, which can vary across languages. This study adopts a cross-linguistic approach to examine how large language models (LLMs) handle quantifier scope interpretation in English and Chinese, using probabilities to assess interpretive likelihood. Human similarity (HS) scores were used to quantify the extent to which LLMs emulate human performance across language groups. Results reveal that most LLMs prefer the surface scope interpretations, aligning with human tendencies, while only some differentiate between English and Chinese in the inverse scope preferences, reflecting human-similar patterns. HS scores highlight variability in LLMs' approximation of human behavior, but their overall potential to align with humans is notable. Differences in model architecture, scale, and particularly models' pre-training data language background, significantly influence how closely LLMs approximate human quantifier scope interpretations.
>
---
#### [new 028] A funny companion: Distinct neural responses to perceived AI- versus humangenerated humor
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究人类对AI与人类幽默的神经反应差异。通过EEG发现，尽管两者笑点相似，但AI幽默引发更小的N400和更大的LPP，显示认知负荷低但情感反应强。研究揭示了大脑如何动态适应AI幽默，挑战“算法厌恶”观念。**

- **链接: [http://arxiv.org/pdf/2509.10847v1](http://arxiv.org/pdf/2509.10847v1)**

> **作者:** Xiaohui Rao; Hanlin Wu; Zhenguang G. Cai
>
> **摘要:** As AI companions become capable of human-like communication, including telling jokes, understanding how people cognitively and emotionally respond to AI humor becomes increasingly important. This study used electroencephalography (EEG) to compare how people process humor from AI versus human sources. Behavioral analysis revealed that participants rated AI and human humor as comparably funny. However, neurophysiological data showed that AI humor elicited a smaller N400 effect, suggesting reduced cognitive effort during the processing of incongruity. This was accompanied by a larger Late Positive Potential (LPP), indicating a greater degree of surprise and emotional response. This enhanced LPP likely stems from the violation of low initial expectations regarding AI's comedic capabilities. Furthermore, a key temporal dynamic emerged: human humor showed habituation effects, marked by an increasing N400 and a decreasing LPP over time. In contrast, AI humor demonstrated increasing processing efficiency and emotional reward, with a decreasing N400 and an increasing LPP. This trajectory reveals how the brain can dynamically update its predictive model of AI capabilities. This process of cumulative reinforcement challenges "algorithm aversion" in humor, as it demonstrates how cognitive adaptation to AI's language patterns can lead to an intensified emotional reward. Additionally, participants' social attitudes toward AI modulated these neural responses, with higher perceived AI trustworthiness correlating with enhanced emotional engagement. These findings indicate that the brain responds to AI humor with surprisingly positive and intense reactions, highlighting humor's potential for fostering genuine engagement in human-AI social interaction.
>
---
#### [new 029] RanAT4BIE: Random Adversarial Training for Biomedical Information Extraction
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出RAT框架，用于生物医学信息抽取任务，旨在提升模型性能并降低计算成本。基于PubMedBERT，通过随机对抗训练机制，在保持模型鲁棒性的同时显著减少计算开销。**

- **链接: [http://arxiv.org/pdf/2509.11191v1](http://arxiv.org/pdf/2509.11191v1)**

> **作者:** Jian Chen; Shengyi Lv; Leilei Su
>
> **备注:** Accepted for publication at the International Joint Conference on Neural Networks (IJCNN) 2025
>
> **摘要:** We introduce random adversarial training (RAT), a novel framework successfully applied to biomedical information extraction (BioIE) tasks. Building on PubMedBERT as the foundational architecture, our study first validates the effectiveness of conventional adversarial training in enhancing pre-trained language models' performance on BioIE tasks. While adversarial training yields significant improvements across various performance metrics, it also introduces considerable computational overhead. To address this limitation, we propose RAT as an efficiency solution for biomedical information extraction. This framework strategically integrates random sampling mechanisms with adversarial training principles, achieving dual objectives: enhanced model generalization and robustness while significantly reducing computational costs. Through comprehensive evaluations, RAT demonstrates superior performance compared to baseline models in BioIE tasks. The results highlight RAT's potential as a transformative framework for biomedical natural language processing, offering a balanced solution to the model performance and computational efficiency.
>
---
#### [new 030] EmoBench-Reddit: A Hierarchical Benchmark for Evaluating the Emotional Intelligence of Multimodal Large Language Models
- **分类: cs.CL**

- **简介: 论文提出EmoBench-Reddit，用于评估多模态大模型的情感理解能力。针对现有基准忽视主观情绪的问题，构建包含图像、文本和情感标签的数据集，并设计由浅入深的层次任务框架，提升模型在感知与认知层面的情感分析能力。**

- **链接: [http://arxiv.org/pdf/2509.11101v1](http://arxiv.org/pdf/2509.11101v1)**

> **作者:** Haokun Li; Yazhou Zhang; Jizhi Ding; Qiuchi Li; Peng Zhang
>
> **摘要:** With the rapid advancement of Multimodal Large Language Models (MLLMs), they have demonstrated exceptional capabilities across a variety of vision-language tasks. However, current evaluation benchmarks predominantly focus on objective visual question answering or captioning, inadequately assessing the models' ability to understand complex and subjective human emotions. To bridge this gap, we introduce EmoBench-Reddit, a novel, hierarchical benchmark for multimodal emotion understanding. The dataset comprises 350 meticulously curated samples from the social media platform Reddit, each containing an image, associated user-provided text, and an emotion category (sad, humor, sarcasm, happy) confirmed by user flairs. We designed a hierarchical task framework that progresses from basic perception to advanced cognition, with each data point featuring six multiple-choice questions and one open-ended question of increasing difficulty. Perception tasks evaluate the model's ability to identify basic visual elements (e.g., colors, objects), while cognition tasks require scene reasoning, intent understanding, and deep empathy integrating textual context. We ensured annotation quality through a combination of AI assistance (Claude 4) and manual verification.
>
---
#### [new 031] PeruMedQA: Benchmarking Large Language Models (LLMs) on Peruvian Medical Exams -- Dataset Construction and Evaluation
- **分类: cs.CL; cs.LG**

- **简介: 该论文构建PeruMedQA数据集，用于评估大语言模型在秘鲁医学考试中的表现。任务是测试LLMs在西班牙语和拉丁美洲医学问题上的性能。研究通过微调提升模型准确率，发现medgemma-27b-text-it表现最佳。**

- **链接: [http://arxiv.org/pdf/2509.11517v1](http://arxiv.org/pdf/2509.11517v1)**

> **作者:** Rodrigo M. Carrillo-Larco; Jesus Lovón Melgarejo; Manuel Castillo-Cara; Gusseppe Bravo-Rocca
>
> **备注:** https://github.com/rodrigo-carrillo/PeruMedQA
>
> **摘要:** BACKGROUND: Medical large language models (LLMS) have demonstrated remarkable performance in answering medical examinations. However, the extent to which this high performance is transferable to medical questions in Spanish and from a Latin American country remains unexplored. This knowledge is crucial as LLM-based medical applications gain traction in Latin America. AIMS: to build a dataset of questions from medical examinations taken by Peruvian physicians pursuing specialty training; to fine-tune a LLM on this dataset; to evaluate and compare the performance in terms of accuracy between vanilla LLMs and the fine-tuned LLM. METHODS: We curated PeruMedQA, a multiple-choice question-answering (MCQA) datasets containing 8,380 questions spanning 12 medical domains (2018-2025). We selected eight medical LLMs including medgemma-4b-it and medgemma-27b-text-it, and developed zero-shot task-specific prompts to answer the questions appropriately. We employed parameter-efficient fine tuning (PEFT)and low-rant adaptation (LoRA) to fine-tune medgemma-4b-it utilizing all questions except those from 2025 (test set). RESULTS: medgemma-27b-text-it outperformed all other models, achieving a proportion of correct answers exceeding 90% in several instances. LLMs with <10 billion parameters exhibited <60% of correct answers, while some exams yielded results <50%. The fine-tuned version of medgemma-4b-it emerged victorious agains all LLMs with <10 billion parameters and rivaled a LLM with 70 billion parameters across various examinations. CONCLUSIONS: For medical AI application and research that require knowledge bases from Spanish-speaking countries and those exhibiting similar epidemiological profiles to Peru's, interested parties should utilize medgemma-27b-text-it or a fine-tuned version of medgemma-4b-it.
>
---
#### [new 032] Interdisciplinary Research in Conversation: A Case Study in Computational Morphology for Language Documentation
- **分类: cs.CL**

- **简介: 论文探讨计算形态学在语言记录中的应用，指出其与实际需求脱节。通过GlossLM案例和用户研究，提出需引入用户中心设计以提升工具实用性，并引发新研究方向。**

- **链接: [http://arxiv.org/pdf/2509.10644v1](http://arxiv.org/pdf/2509.10644v1)**

> **作者:** Enora Rice; Katharina von der Wense; Alexis Palmer
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Computational morphology has the potential to support language documentation through tasks like morphological segmentation and the generation of Interlinear Glossed Text (IGT). However, our research outputs have seen limited use in real-world language documentation settings. This position paper situates the disconnect between computational morphology and language documentation within a broader misalignment between research and practice in NLP and argues that the field risks becoming decontextualized and ineffectual without systematic integration of User-Centered Design (UCD). To demonstrate how principles from UCD can reshape the research agenda, we present a case study of GlossLM, a state-of-the-art multilingual IGT generation model. Through a small-scale user study with three documentary linguists, we find that despite strong metric based performance, the system fails to meet core usability needs in real documentation contexts. These insights raise new research questions around model constraints, label standardization, segmentation, and personalization. We argue that centering users not only produces more effective tools, but surfaces richer, more relevant research directions
>
---
#### [new 033] An Interpretable Benchmark for Clickbait Detection and Tactic Attribution
- **分类: cs.CL**

- **简介: 该论文属于点击诱饵检测任务，旨在解决检测与解释点击诱饵策略的问题。研究提出一个可解释框架，包含检测与策略归因两阶段，并构建合成数据集辅助分析，提升AI系统的透明性与可信度。**

- **链接: [http://arxiv.org/pdf/2509.10937v1](http://arxiv.org/pdf/2509.10937v1)**

> **作者:** Lihi Nofar; Tomer Portal; Aviv Elbaz; Alexander Apartsin; Yehudit Aperstein
>
> **备注:** 7 pages
>
> **摘要:** The proliferation of clickbait headlines poses significant challenges to the credibility of information and user trust in digital media. While recent advances in machine learning have improved the detection of manipulative content, the lack of explainability limits their practical adoption. This paper presents a model for explainable clickbait detection that not only identifies clickbait titles but also attributes them to specific linguistic manipulation strategies. We introduce a synthetic dataset generated by systematically augmenting real news headlines using a predefined catalogue of clickbait strategies. This dataset enables controlled experimentation and detailed analysis of model behaviour. We present a two-stage framework for automatic clickbait analysis comprising detection and tactic attribution. In the first stage, we compare a fine-tuned BERT classifier with large language models (LLMs), specifically GPT-4.0 and Gemini 2.4 Flash, under both zero-shot prompting and few-shot prompting enriched with illustrative clickbait headlines and their associated persuasive tactics. In the second stage, a dedicated BERT-based classifier predicts the specific clickbait strategies present in each headline. This work advances the development of transparent and trustworthy AI systems for combating manipulative media content. We share the dataset with the research community at https://github.com/LLM-HITCS25S/ClickbaitTacticsDetection
>
---
#### [new 034] Joint Effects of Argumentation Theory, Audio Modality and Data Enrichment on LLM-Based Fallacy Classification
- **分类: cs.CL**

- **简介: 该论文研究LLM在政治辩论中分类谬误的任务，探讨论证理论、音频模态和数据增强的影响。通过不同提示策略测试Qwen-3模型，发现增加上下文和情感元数据可能降低性能，基础提示更优。**

- **链接: [http://arxiv.org/pdf/2509.11127v1](http://arxiv.org/pdf/2509.11127v1)**

> **作者:** Hongxu Zhou; Hylke Westerdijk; Khondoker Ittehadul Islam
>
> **摘要:** This study investigates how context and emotional tone metadata influence large language model (LLM) reasoning and performance in fallacy classification tasks, particularly within political debate settings. Using data from U.S. presidential debates, we classify six fallacy types through various prompting strategies applied to the Qwen-3 (8B) model. We introduce two theoretically grounded Chain-of-Thought frameworks: Pragma-Dialectics and the Periodic Table of Arguments, and evaluate their effectiveness against a baseline prompt under three input settings: text-only, text with context, and text with both context and audio-based emotional tone metadata. Results suggest that while theoretical prompting can improve interpretability and, in some cases, accuracy, the addition of context and especially emotional tone metadata often leads to lowered performance. Emotional tone metadata biases the model toward labeling statements as \textit{Appeal to Emotion}, worsening logical reasoning. Overall, basic prompts often outperformed enhanced ones, suggesting that attention dilution from added inputs may worsen rather than improve fallacy classification in LLMs.
>
---
#### [new 035] Pluralistic Alignment for Healthcare: A Role-Driven Framework
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出EthosAgents框架，解决医疗领域大语言模型输出缺乏多元价值观的问题。通过角色驱动方式模拟多样视角，提升模型在不同规模模型中的多元对齐效果，为高风险领域提供更具包容性的解决方案。**

- **链接: [http://arxiv.org/pdf/2509.10685v1](http://arxiv.org/pdf/2509.10685v1)**

> **作者:** Jiayou Zhong; Anudeex Shetty; Chao Jia; Xuanrui Lin; Usman Naseem
>
> **备注:** Accepted to EMNLP 2025 (Main Proceedings)
>
> **摘要:** As large language models are increasingly deployed in sensitive domains such as healthcare, ensuring their outputs reflect the diverse values and perspectives held across populations is critical. However, existing alignment approaches, including pluralistic paradigms like Modular Pluralism, often fall short in the health domain, where personal, cultural, and situational factors shape pluralism. Motivated by the aforementioned healthcare challenges, we propose a first lightweight, generalizable, pluralistic alignment approach, EthosAgents, designed to simulate diverse perspectives and values. We empirically show that it advances the pluralistic alignment for all three modes across seven varying-sized open and closed models. Our findings reveal that health-related pluralism demands adaptable and normatively aware approaches, offering insights into how these models can better respect diversity in other high-stakes domains.
>
---
#### [new 036] RECAP: Transparent Inference-Time Emotion Alignment for Medical Dialogue Systems
- **分类: cs.CL**

- **简介: 论文提出RECAP框架，在不重新训练模型的前提下，提升医疗对话系统的情感推理能力。通过分解共情为透明阶段并暴露情感信号，增强模型的同理心与可审计性，解决医疗AI情感表达不足的问题。**

- **链接: [http://arxiv.org/pdf/2509.10746v1](http://arxiv.org/pdf/2509.10746v1)**

> **作者:** Adarsh Srinivasan; Jacob Dineen; Muhammad Umar Afzal; Muhammad Uzair Sarfraz; Irbaz B. Riaz; Ben Zhou
>
> **摘要:** Large language models in healthcare often miss critical emotional cues, delivering medically sound but emotionally flat advice. This is especially problematic in clinical contexts where patients are distressed and vulnerable, and require empathic communication to support safety, adherence, and trust. We present RECAP (Reflect-Extract-Calibrate-Align-Produce), an inference-time framework that adds structured emotional reasoning without retraining. By decomposing empathy into transparent appraisal-theoretic stages and exposing per-dimension Likert signals, RECAP produces nuanced, auditable responses. Across EmoBench, SECEU, and EQ-Bench, RECAP improves emotional reasoning by 22-28% on 8B models and 10-13% on larger models over zero-shot baselines. Clinician evaluations further confirm superior empathetic communication. RECAP shows that modular, theory-grounded prompting can systematically enhance emotional intelligence in medical AI while preserving the accountability required for deployment.
>
---
#### [new 037] From Fuzzy Speech to Medical Insight: Benchmarking LLMs on Noisy Patient Narratives
- **分类: cs.CL**

- **简介: 该论文提出一个合成数据集NDB，用于评估大语言模型在处理模糊、嘈杂的患者叙述中的诊断能力。任务是测试LLMs在真实医疗场景下的表现，解决现有基准缺乏噪声数据的问题，工作包括构建数据集并评估多个模型。**

- **链接: [http://arxiv.org/pdf/2509.11803v1](http://arxiv.org/pdf/2509.11803v1)**

> **作者:** Eden Mama; Liel Sheri; Yehudit Aperstein; Alexander Apartsin
>
> **备注:** 6 pages, 1 figure
>
> **摘要:** The widespread adoption of large language models (LLMs) in healthcare raises critical questions about their ability to interpret patient-generated narratives, which are often informal, ambiguous, and noisy. Existing benchmarks typically rely on clean, structured clinical text, offering limited insight into model performance under realistic conditions. In this work, we present a novel synthetic dataset designed to simulate patient self-descriptions characterized by varying levels of linguistic noise, fuzzy language, and layperson terminology. Our dataset comprises clinically consistent scenarios annotated with ground-truth diagnoses, spanning a spectrum of communication clarity to reflect diverse real-world reporting styles. Using this benchmark, we fine-tune and evaluate several state-of-the-art models (LLMs), including BERT-based and encoder-decoder T5 models. To support reproducibility and future research, we release the Noisy Diagnostic Benchmark (NDB), a structured dataset of noisy, synthetic patient descriptions designed to stress-test and compare the diagnostic capabilities of large language models (LLMs) under realistic linguistic conditions. We made the benchmark available for the community: https://github.com/lielsheri/PatientSignal
>
---
#### [new 038] Unsupervised Candidate Ranking for Lexical Substitution via Holistic Sentence Semantics
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于词替换任务，旨在解决候选词排序问题。提出基于注意力权重和集成梯度的方法，衡量上下文对目标词的影响，提升语义相似度排序效果。实验表明方法有效。**

- **链接: [http://arxiv.org/pdf/2509.11513v1](http://arxiv.org/pdf/2509.11513v1)**

> **作者:** Zhongyang Hu; Naijie Gu; Xiangzhi Tao; Tianhui Gu; Yibing Zhou
>
> **摘要:** A key subtask in lexical substitution is ranking the given candidate words. A common approach is to replace the target word with a candidate in the original sentence and feed the modified sentence into a model to capture semantic differences before and after substitution. However, effectively modeling the bidirectional influence of candidate substitution on both the target word and its context remains challenging. Existing methods often focus solely on semantic changes at the target position or rely on parameter tuning over multiple evaluation metrics, making it difficult to accurately characterize semantic variation. To address this, we investigate two approaches: one based on attention weights and another leveraging the more interpretable integrated gradients method, both designed to measure the influence of context tokens on the target token and to rank candidates by incorporating semantic similarity between the original and substituted sentences. Experiments on the LS07 and SWORDS datasets demonstrate that both approaches improve ranking performance.
>
---
#### [new 039] CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; I.2.7; I.2.10**

- **简介: 论文提出CoachMe模型，用于生成精准的运动指导指令。该任务旨在解决运动教学中如何有效识别错误并提供改进方法的问题。通过对比学习者与参考动作，在滑冰和拳击等运动中生成高质量指导，优于GPT-4o。**

- **链接: [http://arxiv.org/pdf/2509.11698v1](http://arxiv.org/pdf/2509.11698v1)**

> **作者:** Wei-Hsin Yeh; Yu-An Su; Chih-Ning Chen; Yi-Hsueh Lin; Calvin Ku; Wen-Hsin Chiu; Min-Chun Hu; Lun-Wei Ku
>
> **备注:** Published in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025. Official version: https://doi.org/10.18653/v1/2025.acl-long.1413
>
> **摘要:** Motion instruction is a crucial task that helps athletes refine their technique by analyzing movements and providing corrective guidance. Although recent advances in multimodal models have improved motion understanding, generating precise and sport-specific instruction remains challenging due to the highly domain-specific nature of sports and the need for informative guidance. We propose CoachMe, a reference-based model that analyzes the differences between a learner's motion and a reference under temporal and physical aspects. This approach enables both domain-knowledge learning and the acquisition of a coach-like thinking process that identifies movement errors effectively and provides feedback to explain how to improve. In this paper, we illustrate how CoachMe adapts well to specific sports such as skating and boxing by learning from general movements and then leveraging limited data. Experiments show that CoachMe provides high-quality instructions instead of directions merely in the tone of a coach but without critical information. CoachMe outperforms GPT-4o by 31.6% in G-Eval on figure skating and by 58.3% on boxing. Analysis further confirms that it elaborates on errors and their corresponding improvement methods in the generated instructions. You can find CoachMe here: https://motionxperts.github.io/
>
---
#### [new 040] MOOM: Maintenance, Organization and Optimization of Memory in Ultra-Long Role-Playing Dialogues
- **分类: cs.CL**

- **简介: 该论文提出MOOM，用于解决超长角色扮演对话中记忆失控问题。通过双分支结构提取情节与角色信息，并引入遗忘机制控制记忆容量。同时构建ZH-4O数据集，实验表明MOOM在减少模型调用次数的同时保持可控记忆。属于对话系统中的记忆管理任务。**

- **链接: [http://arxiv.org/pdf/2509.11860v1](http://arxiv.org/pdf/2509.11860v1)**

> **作者:** Weishu Chen; Jinyi Tang; Zhouhui Hou; Shihao Han; Mingjie Zhan; Zhiyuan Huang; Delong Liu; Jiawei Guo; Zhicheng Zhao; Fei Su
>
> **摘要:** Memory extraction is crucial for maintaining coherent ultra-long dialogues in human-robot role-playing scenarios. However, existing methods often exhibit uncontrolled memory growth. To address this, we propose MOOM, the first dual-branch memory plugin that leverages literary theory by modeling plot development and character portrayal as core storytelling elements. Specifically, one branch summarizes plot conflicts across multiple time scales, while the other extracts the user's character profile. MOOM further integrates a forgetting mechanism, inspired by the ``competition-inhibition'' memory theory, to constrain memory capacity and mitigate uncontrolled growth. Furthermore, we present ZH-4O, a Chinese ultra-long dialogue dataset specifically designed for role-playing, featuring dialogues that average 600 turns and include manually annotated memory information. Experimental results demonstrate that MOOM outperforms all state-of-the-art memory extraction methods, requiring fewer large language model invocations while maintaining a controllable memory capacity.
>
---
#### [new 041] Spec-LLaVA: Accelerating Vision-Language Models with Dynamic Tree-Based Speculative Decoding
- **分类: cs.CL**

- **简介: 该论文提出Spec-LLaVA，通过动态树状推测解码加速视觉语言模型推理，解决其生成速度慢的问题。使用轻量级草案模型与主模型协同，提升效率且不损失质量，适用于实时多模态应用。**

- **链接: [http://arxiv.org/pdf/2509.11961v1](http://arxiv.org/pdf/2509.11961v1)**

> **作者:** Mingxiao Huo; Jiayi Zhang; Hewei Wang; Jinfeng Xu; Zheyu Chen; Huilin Tai; Yijun Chen
>
> **备注:** 7pages, accepted by ICML TTODLer-FM workshop
>
> **摘要:** Vision-Language Models (VLMs) enable powerful multimodal reasoning but suffer from slow autoregressive inference, limiting their deployment in real-time applications. We introduce Spec-LLaVA, a system that applies speculative decoding to accelerate VLMs without sacrificing output quality. Spec-LLaVA pairs a lightweight draft VLM with a large target model: the draft speculates future tokens, which the target verifies in parallel, allowing multiple tokens to be generated per step. To maximize efficiency, we design a dynamic tree-based verification algorithm that adaptively expands and prunes speculative branches using draft model confidence. On MS COCO out-of-domain images, Spec-LLaVA achieves up to 3.28$\times$ faster decoding on LLaVA-1.5 (7B, 13B) with no loss in generation quality. This work presents a lossless acceleration framework for VLMs using dynamic tree-structured speculative decoding, opening a path toward practical real-time multimodal assistants. Importantly, the lightweight draft model design makes the framework amenable to resource-constrained or on-device deployment settings.
>
---
#### [new 042] HalluDetect: Detecting, Mitigating, and Benchmarking Hallucinations in Conversational Systems
- **分类: cs.CL**

- **简介: 该论文属于对话系统中的幻觉检测与缓解任务，旨在解决大语言模型在客服聊天机器人中产生虚假信息的问题。研究开发了HalluDetect系统，并提出AgentBot架构有效减少幻觉，提升事实准确性。**

- **链接: [http://arxiv.org/pdf/2509.11619v1](http://arxiv.org/pdf/2509.11619v1)**

> **作者:** Spandan Anaokar; Shrey Ganatra; Harshvivek Kashid; Swapnil Bhattacharyya; Shruti Nair; Reshma Sekhar; Siddharth Manohar; Rahul Hemrajani; Pushpak Bhattacharyya
>
> **备注:** 6 pages + references + appendix, 3 figures, 2 tables
>
> **摘要:** Large Language Models (LLMs) are widely used in industry but remain prone to hallucinations, limiting their reliability in critical applications. This work addresses hallucination reduction in consumer grievance chatbots built using LLaMA 3.1 8B Instruct, a compact model frequently used in industry. We develop HalluDetect, an LLM-based hallucination detection system that achieves an F1 score of 69% outperforming baseline detectors by 25.44%. Benchmarking five chatbot architectures, we find that out of them, AgentBot minimizes hallucinations to 0.4159 per turn while maintaining the highest token accuracy (96.13%), making it the most effective mitigation strategy. Our findings provide a scalable framework for hallucination mitigation, demonstrating that optimized inference strategies can significantly improve factual accuracy. While applied to consumer law, our approach generalizes to other high-risk domains, enhancing trust in LLM-driven assistants. We will release the code and dataset
>
---
#### [new 043] SENSE models: an open source solution for multilingual and multimodal semantic-based tasks
- **分类: cs.CL**

- **简介: 该论文提出SENSE模型，用于多语言和多模态语义任务。通过改进教师-学生框架，对齐语音和文本编码器，提升语义表示能力。模型开源，集成于SpeechBrain，并在实验中表现优异。**

- **链接: [http://arxiv.org/pdf/2509.12093v1](http://arxiv.org/pdf/2509.12093v1)**

> **作者:** Salima Mdhaffar; Haroun Elleuch; Chaimae Chellaf; Ha Nguyen; Yannick Estève
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** This paper introduces SENSE (Shared Embedding for N-lingual Speech and tExt), an open-source solution inspired by the SAMU-XLSR framework and conceptually similar to Meta AI's SONAR models. These approaches rely on a teacher-student framework to align a self-supervised speech encoder with the language-agnostic continuous representations of a text encoder at the utterance level. We describe how the original SAMU-XLSR method has been updated by selecting a stronger teacher text model and a better initial speech encoder. The source code for training and using SENSE models has been integrated into the SpeechBrain toolkit, and the first SENSE model we trained has been publicly released. We report experimental results on multilingual and multimodal semantic tasks, where our SENSE model achieves highly competitive performance. Finally, this study offers new insights into how semantics are captured in such semantically aligned speech encoders.
>
---
#### [new 044] CBP-Tuning: Efficient Local Customization for Black-box Large Language Models
- **分类: cs.CL**

- **简介: 论文提出CBP-Tuning框架，解决黑盒大模型本地高效定制与隐私保护问题。通过服务端提示生成与用户端无梯度优化，实现无需模型权重和私有数据的任务适配，提升跨领域性能。**

- **链接: [http://arxiv.org/pdf/2509.12112v1](http://arxiv.org/pdf/2509.12112v1)**

> **作者:** Jiaxuan Zhao; Naibin Gu; Yuchen Feng; Xiyu Liu; Peng Fu; Zheng Lin; Weiping Wang
>
> **摘要:** The high costs of customizing large language models (LLMs) fundamentally limit their adaptability to user-specific needs. Consequently, LLMs are increasingly offered as cloud-based services, a paradigm that introduces critical limitations: providers struggle to support personalized customization at scale, while users face privacy risks when exposing sensitive data. To address this dual challenge, we propose Customized Black-box Prompt Tuning (CBP-Tuning), a novel framework that facilitates efficient local customization while preserving bidirectional privacy. Specifically, we design a two-stage framework: (1) a prompt generator trained on the server-side to capture domain-specific and task-agnostic capabilities, and (2) user-side gradient-free optimization that tailors soft prompts for individual tasks. This approach eliminates the need for users to access model weights or upload private data, requiring only a single customized vector per task while achieving effective adaptation. Furthermore, the evaluation of CBP-Tuning in the commonsense reasoning, medical and financial domain settings demonstrates superior performance compared to baselines, showcasing its advantages in task-agnostic processing and privacy preservation.
>
---
#### [new 045] An Agentic Toolkit for Adaptive Information Extraction from Regulatory Documents
- **分类: cs.CL**

- **简介: 论文提出一种智能工具系统，用于从欧盟建筑产品性能声明（DoP）中提取关键信息。该系统采用规划-执行-响应架构，解决因文档结构多样导致的传统方法失效问题，提升多语言、多格式下的信息抽取鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.11773v1](http://arxiv.org/pdf/2509.11773v1)**

> **作者:** Gaye Colakoglu; Gürkan Solmaz; Jonathan Fürst
>
> **摘要:** Declaration of Performance (DoP) documents, mandated by EU regulation, certify the performance of construction products. While some of their content is standardized, DoPs vary widely in layout, language, schema, and format, posing challenges for automated key-value pair extraction (KVP) and question answering (QA). Existing static or LLM-only IE pipelines often hallucinate and fail to adapt to this structural diversity. Our domain-specific, stateful agentic system addresses these challenges through a planner-executor-responder architecture. The system infers user intent, detects document modality, and orchestrates tools dynamically for robust, traceable reasoning while avoiding tool misuse or execution loops. Evaluation on a curated DoP dataset demonstrates improved robustness across formats and languages, offering a scalable solution for structured data extraction in regulated workflows.
>
---
#### [new 046] When Smiley Turns Hostile: Interpreting How Emojis Trigger LLMs' Toxicity
- **分类: cs.CL**

- **简介: 该论文研究表情符号如何触发大语言模型生成有毒内容，属于自然语言处理中的安全与伦理任务。通过实验和模型分析，揭示表情符号作为语义通道绕过安全机制的机制，并探讨预训练数据污染的影响。**

- **链接: [http://arxiv.org/pdf/2509.11141v1](http://arxiv.org/pdf/2509.11141v1)**

> **作者:** Shiyao Cui; Xijia Feng; Yingkang Wang; Junxiao Yang; Zhexin Zhang; Biplab Sikdar; Hongning Wang; Han Qiu; Minlie Huang
>
> **摘要:** Emojis are globally used non-verbal cues in digital communication, and extensive research has examined how large language models (LLMs) understand and utilize emojis across contexts. While usually associated with friendliness or playfulness, it is observed that emojis may trigger toxic content generation in LLMs. Motivated by such a observation, we aim to investigate: (1) whether emojis can clearly enhance the toxicity generation in LLMs and (2) how to interpret this phenomenon. We begin with a comprehensive exploration of emoji-triggered LLM toxicity generation by automating the construction of prompts with emojis to subtly express toxic intent. Experiments across 5 mainstream languages on 7 famous LLMs along with jailbreak tasks demonstrate that prompts with emojis could easily induce toxicity generation. To understand this phenomenon, we conduct model-level interpretations spanning semantic cognition, sequence generation and tokenization, suggesting that emojis can act as a heterogeneous semantic channel to bypass the safety mechanisms. To pursue deeper insights, we further probe the pre-training corpus and uncover potential correlation between the emoji-related data polution with the toxicity generation behaviors. Supplementary materials provide our implementation code and data. (Warning: This paper contains potentially sensitive contents)
>
---
#### [new 047] We Argue to Agree: Towards Personality-Driven Argumentation-Based Negotiation Dialogue Systems for Tourism
- **分类: cs.CL; cs.AI**

- **简介: 论文提出PAN-DG任务，旨在构建基于个性和论辩机制的旅游谈判对话系统。通过引入PACT数据集，包含三种个性特征，提升对话系统的个性化与推理能力，为未来研究奠定基础。**

- **链接: [http://arxiv.org/pdf/2509.11118v1](http://arxiv.org/pdf/2509.11118v1)**

> **作者:** Priyanshu Priya; Saurav Dudhate; Desai Vishesh Yasheshbhai; Asif Ekbal
>
> **备注:** Paper is accepted at EMNLP (Findings) 2025
>
> **摘要:** Integrating argumentation mechanisms into negotiation dialogue systems improves conflict resolution through exchanges of arguments and critiques. Moreover, incorporating personality attributes enhances adaptability by aligning interactions with individuals' preferences and styles. To advance these capabilities in negotiation dialogue systems, we propose a novel Personality-driven Argumentation-based Negotiation Dialogue Generation (PAN-DG) task. To support this task, we introduce PACT, a dataset of Personality-driven Argumentation-based negotiation Conversations for Tourism sector. This dataset, generated using Large Language Models (LLMs), features three distinct personality profiles, viz. Argumentation Profile, Preference Profile, and Buying Style Profile to simulate a variety of negotiation scenarios involving diverse personalities. Thorough automatic and manual evaluations indicate that the dataset comprises high-quality dialogues. Further, we conduct comparative experiments between pre-trained and fine-tuned LLMs for the PAN-DG task. Multi-dimensional evaluation demonstrates that the fine-tuned LLMs effectively generate personality-driven rational responses during negotiations. This underscores the effectiveness of PACT in enhancing personalization and reasoning capabilities in negotiation dialogue systems, thereby establishing a foundation for future research in this domain.
>
---
#### [new 048] Context Copying Modulation: The Role of Entropy Neurons in Managing Parametric and Contextual Knowledge Conflicts
- **分类: cs.CL**

- **简介: 该论文研究大语言模型中熵神经元在处理上下文与参数知识冲突中的作用，属于模型内部机制分析任务。通过实验验证熵神经元抑制上下文复制行为，揭示其对生成过程的影响，提升对LLMs冲突处理机制的理解。**

- **链接: [http://arxiv.org/pdf/2509.10663v1](http://arxiv.org/pdf/2509.10663v1)**

> **作者:** Zineddine Tighidet; Andrea Mogini; Hedi Ben-younes; Jiali Mei; Patrick Gallinari; Benjamin Piwowarski
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** The behavior of Large Language Models (LLMs) when facing contextual information that conflicts with their internal parametric knowledge is inconsistent, with no generally accepted explanation for the expected outcome distribution. Recent work has identified in autoregressive transformer models a class of neurons -- called entropy neurons -- that produce a significant effect on the model output entropy while having an overall moderate impact on the ranking of the predicted tokens. In this paper, we investigate the preliminary claim that these neurons are involved in inhibiting context copying behavior in transformers by looking at their role in resolving conflicts between contextual and parametric information. We show that entropy neurons are responsible for suppressing context copying across a range of LLMs, and that ablating them leads to a significant change in the generation process. These results enhance our understanding of the internal dynamics of LLMs when handling conflicting information.
>
---
#### [new 049] Growing Perspectives: Modelling Embodied Perspective Taking and Inner Narrative Development Using Large Language Models
- **分类: cs.CL; cs.AI; cs.HC; cs.RO; I.2; I.2.7; I.2.10; J.4**

- **简介: 论文研究如何利用大语言模型模拟人类视角转换与内在叙事发展。通过PerspAct系统，结合ReAct框架与GPT，评估其在协作任务中的表现，揭示语言交流对内部表征的优化作用，探索语言与具身认知的整合潜力。**

- **链接: [http://arxiv.org/pdf/2509.11868v1](http://arxiv.org/pdf/2509.11868v1)**

> **作者:** Sabrina Patania; Luca Annese; Anna Lambiase; Anita Pellegrini; Tom Foulsham; Azzurra Ruggeri; Silvia Rossi; Silvia Serino; Dimitri Ognibene
>
> **备注:** Accepted at ICDL https://icdl2025.fel.cvut.cz/
>
> **摘要:** Language and embodied perspective taking are essential for human collaboration, yet few computational models address both simultaneously. This work investigates the PerspAct system [1], which integrates the ReAct (Reason and Act) paradigm with Large Language Models (LLMs) to simulate developmental stages of perspective taking, grounded in Selman's theory [2]. Using an extended director task, we evaluate GPT's ability to generate internal narratives aligned with specified developmental stages, and assess how these influence collaborative performance both qualitatively (action selection) and quantitatively (task efficiency). Results show that GPT reliably produces developmentally-consistent narratives before task execution but often shifts towards more advanced stages during interaction, suggesting that language exchanges help refine internal representations. Higher developmental stages generally enhance collaborative effectiveness, while earlier stages yield more variable outcomes in complex contexts. These findings highlight the potential of integrating embodied perspective taking and language in LLMs to better model developmental dynamics and stress the importance of evaluating internal speech during combined linguistic and embodied tasks.
>
---
#### [new 050] AesBiasBench: Evaluating Bias and Alignment in Multimodal Language Models for Personalized Image Aesthetic Assessment
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出AesBiasBench，用于评估多模态语言模型在个性化图像审美评估中的偏见与对齐情况。任务涉及审美感知、评估与共情，通过结构化指标分析模型在不同人口统计群体中的表现，揭示模型规模与偏见、对齐的关系。**

- **链接: [http://arxiv.org/pdf/2509.11620v1](http://arxiv.org/pdf/2509.11620v1)**

> **作者:** Kun Li; Lai-Man Po; Hongzheng Yang; Xuyuan Xu; Kangcheng Liu; Yuzhi Zhao
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Multimodal Large Language Models (MLLMs) are increasingly applied in Personalized Image Aesthetic Assessment (PIAA) as a scalable alternative to expert evaluations. However, their predictions may reflect subtle biases influenced by demographic factors such as gender, age, and education. In this work, we propose AesBiasBench, a benchmark designed to evaluate MLLMs along two complementary dimensions: (1) stereotype bias, quantified by measuring variations in aesthetic evaluations across demographic groups; and (2) alignment between model outputs and genuine human aesthetic preferences. Our benchmark covers three subtasks (Aesthetic Perception, Assessment, Empathy) and introduces structured metrics (IFD, NRD, AAS) to assess both bias and alignment. We evaluate 19 MLLMs, including proprietary models (e.g., GPT-4o, Claude-3.5-Sonnet) and open-source models (e.g., InternVL-2.5, Qwen2.5-VL). Results indicate that smaller models exhibit stronger stereotype biases, whereas larger models align more closely with human preferences. Incorporating identity information often exacerbates bias, particularly in emotional judgments. These findings underscore the importance of identity-aware evaluation frameworks in subjective vision-language tasks.
>
---
#### [new 051] PledgeTracker: A System for Monitoring the Fulfilment of Pledges
- **分类: cs.CL**

- **简介: 该论文提出PledgeTracker系统，用于跟踪政治承诺的履行情况。任务是动态、多文档环境下验证承诺履行。系统通过构建结构化时间线解决现有方法忽视时序和多源信息的问题，提升事实核查效率。**

- **链接: [http://arxiv.org/pdf/2509.11804v1](http://arxiv.org/pdf/2509.11804v1)**

> **作者:** Yulong Chen; Michael Sejr Schlichtkrull; Zhenyun Deng; David Corney; Nasim Asl; Joshua Salisbury; Andrew Dudfield; Andreas Vlachos
>
> **备注:** EMNLP 2025 demo
>
> **摘要:** Political pledges reflect candidates' policy commitments, but tracking their fulfilment requires reasoning over incremental evidence distributed across multiple, dynamically updated sources. Existing methods simplify this task into a document classification task, overlooking its dynamic, temporal and multi-document nature. To address this issue, we introduce \textsc{PledgeTracker}, a system that reformulates pledge verification into structured event timeline construction. PledgeTracker consists of three core components: (1) a multi-step evidence retrieval module; (2) a timeline construction module and; (3) a fulfilment filtering module, allowing the capture of the evolving nature of pledge fulfilment and producing interpretable and structured timelines. We evaluate PledgeTracker in collaboration with professional fact-checkers in real-world workflows, demonstrating its effectiveness in retrieving relevant evidence and reducing human verification effort.
>
---
#### [new 052] Pre-Storage Reasoning for Episodic Memory: Shifting Inference Burden to Memory for Personalized Dialogue
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PREMem方法，用于个性化对话中的长期记忆管理。通过将推理任务从响应生成转移到记忆构建，提升模型效率与效果，尤其在小模型上表现优异。属于对话系统中的记忆建模任务。**

- **链接: [http://arxiv.org/pdf/2509.10852v1](http://arxiv.org/pdf/2509.10852v1)**

> **作者:** Sangyeop Kim; Yohan Lee; Sanghwa Kim; Hyunjong Kim; Sungzoon Cho
>
> **备注:** Accepted by EMNLP 2025 (Findings)
>
> **摘要:** Effective long-term memory in conversational AI requires synthesizing information across multiple sessions. However, current systems place excessive reasoning burden on response generation, making performance significantly dependent on model sizes. We introduce PREMem (Pre-storage Reasoning for Episodic Memory), a novel approach that shifts complex reasoning processes from inference to memory construction. PREMem extracts fine-grained memory fragments categorized into factual, experiential, and subjective information; it then establishes explicit relationships between memory items across sessions, capturing evolution patterns like extensions, transformations, and implications. By performing this reasoning during pre-storage rather than when generating a response, PREMem creates enriched representations while reducing computational demands during interactions. Experiments show significant performance improvements across all model sizes, with smaller models achieving results comparable to much larger baselines while maintaining effectiveness even with constrained token budgets. Code and dataset are available at https://github.com/sangyeop-kim/PREMem.
>
---
#### [new 053] HiChunk: Evaluating and Enhancing Retrieval-Augmented Generation with Hierarchical Chunking
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于检索增强生成（RAG）任务，旨在解决文档分块质量评估不足的问题。提出HiCBench评估基准和HiChunk框架，提升RAG系统的分块效果与整体性能。**

- **链接: [http://arxiv.org/pdf/2509.11552v1](http://arxiv.org/pdf/2509.11552v1)**

> **作者:** Wensheng Lu; Keyu Chen; Ruizhi Qiao; Xing Sun
>
> **备注:** 17 pages, 5 figures, 6 tables
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances the response capabilities of language models by integrating external knowledge sources. However, document chunking as an important part of RAG system often lacks effective evaluation tools. This paper first analyzes why existing RAG evaluation benchmarks are inadequate for assessing document chunking quality, specifically due to evidence sparsity. Based on this conclusion, we propose HiCBench, which includes manually annotated multi-level document chunking points, synthesized evidence-dense quetion answer(QA) pairs, and their corresponding evidence sources. Additionally, we introduce the HiChunk framework, a multi-level document structuring framework based on fine-tuned LLMs, combined with the Auto-Merge retrieval algorithm to improve retrieval quality. Experiments demonstrate that HiCBench effectively evaluates the impact of different chunking methods across the entire RAG pipeline. Moreover, HiChunk achieves better chunking quality within reasonable time consumption, thereby enhancing the overall performance of RAG systems.
>
---
#### [new 054] Automated MCQA Benchmarking at Scale: Evaluating Reasoning Traces as Retrieval Sources for Domain Adaptation of Small Language Models
- **分类: cs.CL; cs.AI; I.2.7; I.2.11**

- **简介: 该论文提出一种自动生成MCQA基准的框架，用于评估小语言模型的领域适应能力。通过从科学论文中生成问题并利用推理轨迹增强检索，显著提升模型表现。属于自然语言处理中的基准测试与模型评估任务。**

- **链接: [http://arxiv.org/pdf/2509.10744v1](http://arxiv.org/pdf/2509.10744v1)**

> **作者:** Ozan Gokdemir; Neil Getty; Robert Underwood; Sandeep Madireddy; Franck Cappello; Arvind Ramanathan; Ian T. Foster; Rick L. Stevens
>
> **备注:** This manuscript has been accepted for publication at the Supercomputing 25 (SC '25) Conference (Frontiers in Generative AI for HPC Science and Engineering: Foundations, Challenges, and Opportunities Workshop) in St. Louis, MO, USA on November 16th, 2025. It will appear in the SC25 Workshop Proceedings after that date
>
> **摘要:** As scientific knowledge grows at an unprecedented pace, evaluation benchmarks must evolve to reflect new discoveries and ensure language models are tested on current, diverse literature. We propose a scalable, modular framework for generating multiple-choice question-answering (MCQA) benchmarks directly from large corpora of scientific papers. Our pipeline automates every stage of MCQA creation, including PDF parsing, semantic chunking, question generation, and model evaluation. As a case study, we generate more than 16,000 MCQs from 22,000 open-access articles in radiation and cancer biology. We then evaluate a suite of small language models (1.1B-14B parameters) on these questions, comparing baseline accuracy with retrieval-augmented generation (RAG) from paper-derived semantic chunks and from reasoning traces distilled from GPT-4.1. We find that reasoning-trace retrieval consistently improves performance on both synthetic and expert-annotated benchmarks, enabling several small models to surpass GPT-4 on the 2023 Astro Radiation and Cancer Biology exam.
>
---
#### [new 055] Steering Language Models in Multi-Token Generation: A Case Study on Tense and Aspect
- **分类: cs.CL; I.2.7**

- **简介: 该论文研究大语言模型如何编码时态和体貌语法知识，通过线性判别分析识别其残差空间中的表示方向，并利用概念引导技术实现对生成文本的控制。论文属于自然语言生成任务，旨在解决多词生成中语法特征的有效控制问题。**

- **链接: [http://arxiv.org/pdf/2509.12065v1](http://arxiv.org/pdf/2509.12065v1)**

> **作者:** Alina Klerings; Jannik Brinkmann; Daniel Ruffinelli; Simone Ponzetto
>
> **备注:** to be published in The 2025 Conference on Empirical Methods in Natural Language Processing
>
> **摘要:** Large language models (LLMs) are able to generate grammatically well-formed text, but how do they encode their syntactic knowledge internally? While prior work has focused largely on binary grammatical contrasts, in this work, we study the representation and control of two multidimensional hierarchical grammar phenomena - verb tense and aspect - and for each, identify distinct, orthogonal directions in residual space using linear discriminant analysis. Next, we demonstrate causal control over both grammatical features through concept steering across three generation tasks. Then, we use these identified features in a case study to investigate factors influencing effective steering in multi-token generation. We find that steering strength, location, and duration are crucial parameters for reducing undesirable side effects such as topic shift and degeneration. Our findings suggest that models encode tense and aspect in structurally organized, human-like ways, but effective control of such features during generation is sensitive to multiple factors and requires manual tuning or automated optimization.
>
---
#### [new 056] Dynamic Span Interaction and Graph-Aware Memory for Entity-Level Sentiment Classification
- **分类: cs.CL**

- **简介: 该论文属于实体级情感分类任务，旨在识别文本中特定实体的情感倾向。针对实体与情感表达的复杂交互、跨句依赖及共指一致性等问题，提出SpanEIT框架，结合动态跨度交互与图感知记忆机制，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2509.11604v1](http://arxiv.org/pdf/2509.11604v1)**

> **作者:** Md. Mithun Hossain; Sanjara; Md. Shakil Hossain; Sudipto Chaki
>
> **摘要:** Entity-level sentiment classification involves identifying the sentiment polarity linked to specific entities within text. This task poses several challenges: effectively modeling the subtle and complex interactions between entities and their surrounding sentiment expressions; capturing dependencies that may span across sentences; and ensuring consistent sentiment predictions for multiple mentions of the same entity through coreference resolution. Additionally, linguistic phenomena such as negation, ambiguity, and overlapping opinions further complicate the analysis. These complexities make entity-level sentiment classification a difficult problem, especially in real-world, noisy textual data. To address these issues, we propose SpanEIT, a novel framework integrating dynamic span interaction and graph-aware memory mechanisms for enhanced entity-sentiment relational modeling. SpanEIT builds span-based representations for entities and candidate sentiment phrases, employs bidirectional attention for fine-grained interactions, and uses a graph attention network to capture syntactic and co-occurrence relations. A coreference-aware memory module ensures entity-level consistency across documents. Experiments on FSAD, BARU, and IMDB datasets show SpanEIT outperforms state-of-the-art transformer and hybrid baselines in accuracy and F1 scores. Ablation and interpretability analyses validate the effectiveness of our approach, underscoring its potential for fine-grained sentiment analysis in applications like social media monitoring and customer feedback analysis.
>
---
#### [new 057] ClaimIQ at CheckThat! 2025: Comparing Prompted and Fine-Tuned Language Models for Verifying Numerical Claims
- **分类: cs.CL; cs.AI**

- **简介: 论文参与CLEF 2025 CheckThat! Lab任务3，旨在验证数值和时间声明。研究对比了提示大语言模型与LoRA微调方法，并探索了证据选择策略，发现微调模型在验证集表现优异，但测试集存在泛化问题。**

- **链接: [http://arxiv.org/pdf/2509.11492v1](http://arxiv.org/pdf/2509.11492v1)**

> **作者:** Anirban Saha Anik; Md Fahimul Kabir Chowdhury; Andrew Wyckoff; Sagnik Ray Choudhury
>
> **备注:** Notebook for the CheckThat! Lab at CLEF 2025
>
> **摘要:** This paper presents our system for Task 3 of the CLEF 2025 CheckThat! Lab, which focuses on verifying numerical and temporal claims using retrieved evidence. We explore two complementary approaches: zero-shot prompting with instruction-tuned large language models (LLMs) and supervised fine-tuning using parameter-efficient LoRA. To enhance evidence quality, we investigate several selection strategies, including full-document input and top-k sentence filtering using BM25 and MiniLM. Our best-performing model LLaMA fine-tuned with LoRA achieves strong performance on the English validation set. However, a notable drop in the test set highlights a generalization challenge. These findings underscore the importance of evidence granularity and model adaptation for robust numerical fact verification.
>
---
#### [new 058] GAPrune: Gradient-Alignment Pruning for Domain-Aware Embeddings
- **分类: cs.CL**

- **简介: 该论文提出GAPrune方法，解决领域感知嵌入模型压缩问题。通过结合Fisher信息与梯度对齐，设计DAI评分策略，在保持通用语言能力的同时提升领域性能，实现高效剪枝。**

- **链接: [http://arxiv.org/pdf/2509.10844v1](http://arxiv.org/pdf/2509.10844v1)**

> **作者:** Yixuan Tang; Yi Yang
>
> **备注:** https://github.com/yixuantt/GAPrune
>
> **摘要:** Domain-specific embedding models have shown promise for applications that require specialized semantic understanding, such as coding agents and financial retrieval systems, often achieving higher performance gains than general models. However, state-of-the-art embedding models are typically based on LLMs, which contain billions of parameters, making deployment challenging in resource-constrained environments. Model compression through pruning offers a promising solution, but existing pruning methods treat all parameters uniformly, failing to distinguish between general semantic representations and domain-specific patterns, leading to suboptimal pruning decisions. Thus, we propose GAPrune, a pruning framework that addresses this challenge by considering both domain importance and preserving general linguistic foundation. Our method uses Fisher Information to measure importance and general-domain gradient alignment to assess parameter behavior, then combines these signals using our Domain Alignment Importance (DAI) scoring. Lower DAI scores indicate that the parameter is either less important for the domain task or creates conflicts between domain and general objectives. Experiments on two domain benchmarks, FinMTEB and ChemTEB, show that GAPrune maintains performance within 2.5% of dense models in one-shot pruning at 50% sparsity, while outperforming all baselines. With retraining in 100 steps, GAPrune achieves +4.51% improvement on FinMTEB and +1.73% on ChemTEB, demonstrating that our pruning strategy not only preserves but enhances domain-specific capabilities. Our findings demonstrate that principled pruning strategies can achieve model compression and enhanced domain specialization, providing the research community with a new approach for development.
>
---
#### [new 059] Introducing Spotlight: A Novel Approach for Generating Captivating Key Information from Documents
- **分类: cs.CL**

- **简介: 论文提出Spotlight，一种生成吸引人关键信息的新方法，用于从文档中提取精炼、引人入胜的叙述。该任务旨在提升读者对原文的参与度，通过两阶段模型训练与优化实现高质量信息提炼。**

- **链接: [http://arxiv.org/pdf/2509.10935v1](http://arxiv.org/pdf/2509.10935v1)**

> **作者:** Ankan Mullick; Sombit Bose; Rounak Saha; Ayan Kumar Bhowmick; Aditya Vempaty; Prasenjit Dey; Ravi Kokku; Pawan Goyal; Niloy Ganguly
>
> **备注:** Paper accepted in EMNLP 2025 Main Conference (Full)
>
> **摘要:** In this paper, we introduce Spotlight, a novel paradigm for information extraction that produces concise, engaging narratives by highlighting the most compelling aspects of a document. Unlike traditional summaries, which prioritize comprehensive coverage, spotlights selectively emphasize intriguing content to foster deeper reader engagement with the source material. We formally differentiate spotlights from related constructs and support our analysis with a detailed benchmarking study using new datasets curated for this work. To generate high-quality spotlights, we propose a two-stage approach: fine-tuning a large language model on our benchmark data, followed by alignment via Direct Preference Optimization (DPO). Our comprehensive evaluation demonstrates that the resulting model not only identifies key elements with precision but also enhances readability and boosts the engagement value of the original document.
>
---
#### [new 060] Continually Adding New Languages to Multilingual Language Models
- **分类: cs.CL**

- **简介: 该论文研究如何持续为多语言模型添加新语言，提出Layer-Selective LoRA方法，在特定层插入低秩适配器以减少遗忘。任务是解决模型在新增语言时对原有语言能力的保持问题，工作包括方法设计与多语言实验验证。**

- **链接: [http://arxiv.org/pdf/2509.11414v1](http://arxiv.org/pdf/2509.11414v1)**

> **作者:** Abraham Toluwase Owodunni; Sachin Kumar
>
> **摘要:** Multilingual language models are trained on a fixed set of languages, and to support new languages, the models need to be retrained from scratch. This is an expensive endeavor and is often infeasible, as model developers tend not to release their pre-training data. Naive approaches, such as continued pretraining, suffer from catastrophic forgetting; however, mitigation strategies like experience replay cannot be applied due to the lack of original pretraining data. In this work, we investigate the problem of continually adding new languages to a multilingual model, assuming access to pretraining data in only the target languages. We explore multiple approaches to address this problem and propose Layer-Selective LoRA (LayRA), which adds Low-Rank Adapters (LoRA) to selected initial and final layers while keeping the rest of the model frozen. LayRA builds on two insights: (1) LoRA reduces forgetting, and (2) multilingual models encode inputs in the source language in the initial layers, reason in English in intermediate layers, and translate back to the source language in final layers. We experiment with adding multiple combinations of Galician, Swahili, and Urdu to pretrained language models and evaluate each method on diverse multilingual tasks. We find that LayRA provides the overall best tradeoff between preserving models' capabilities in previously supported languages, while being competitive with existing approaches such as LoRA in learning new languages. We also demonstrate that using model arithmetic, the adapted models can be equipped with strong instruction following abilities without access to any instruction tuning data in the target languages.
>
---
#### [new 061] On the Distinctive Co-occurrence Characteristics of Antonymy
- **分类: cs.CL**

- **简介: 论文研究反义词的共现特征，比较其与三种其他语义关系的差异。任务是分析反义词是否具有独特共现模式。通过实证发现，反义词共现强度高、顺序偏好且跨度短，填补了此前缺乏对比研究的空白。**

- **链接: [http://arxiv.org/pdf/2509.11534v1](http://arxiv.org/pdf/2509.11534v1)**

> **作者:** Zhihan Cao; Hiroaki Yamada; Takenobu Tokunaga
>
> **备注:** Accepted by *SEM 2025
>
> **摘要:** Antonymy has long received particular attention in lexical semantics. Previous studies have shown that antonym pairs frequently co-occur in text, across genres and parts of speech, more often than would be expected by chance. However, whether this co-occurrence pattern is distinctive of antonymy remains unclear, due to a lack of comparison with other semantic relations. This work fills the gap by comparing antonymy with three other relations across parts of speech using robust co-occurrence metrics. We find that antonymy is distinctive in three respects: antonym pairs co-occur with high strength, in a preferred linear order, and within short spans. All results are available online.
>
---
#### [new 062] Evaluating Large Language Models for Evidence-Based Clinical Question Answering
- **分类: cs.CL**

- **简介: 该论文评估大语言模型在基于证据的临床问答任务中的表现，构建多源基准数据集，测试GPT-4o-mini和GPT-5在不同临床问题上的准确性，发现结构化指南回答准确率最高，并验证检索增强提示对提升模型性能的有效性。**

- **链接: [http://arxiv.org/pdf/2509.10843v1](http://arxiv.org/pdf/2509.10843v1)**

> **作者:** Can Wang; Yiqun Chen
>
> **摘要:** Large Language Models (LLMs) have demonstrated substantial progress in biomedical and clinical applications, motivating rigorous evaluation of their ability to answer nuanced, evidence-based questions. We curate a multi-source benchmark drawing from Cochrane systematic reviews and clinical guidelines, including structured recommendations from the American Heart Association and narrative guidance used by insurers. Using GPT-4o-mini and GPT-5, we observe consistent performance patterns across sources and clinical domains: accuracy is highest on structured guideline recommendations (90%) and lower on narrative guideline and systematic review questions (60--70%). We also find a strong correlation between accuracy and the citation count of the underlying systematic reviews, where each doubling of citations is associated with roughly a 30% increase in the odds of a correct answer. Models show moderate ability to reason about evidence quality when contextual information is supplied. When we incorporate retrieval-augmented prompting, providing the gold-source abstract raises accuracy on previously incorrect items to 0.79; providing top 3 PubMed abstracts (ranked by semantic relevance) improves accuracy to 0.23, while random abstracts reduce accuracy (0.10, within temperature variation). These effects are mirrored in GPT-4o-mini, underscoring that source clarity and targeted retrieval -- not just model size -- drive performance. Overall, our results highlight both the promise and current limitations of LLMs for evidence-based clinical question answering. Retrieval-augmented prompting emerges as a useful strategy to improve factual accuracy and alignment with source evidence, while stratified evaluation by specialty and question type remains essential to understand current knowledge access and to contextualize model performance.
>
---
#### [new 063] Improving LLMs' Learning for Coreference Resolution
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的核心ference解析任务，旨在解决大语言模型在该任务中出现的幻觉和性能不足问题。研究提出反向训练与迭代文档生成两种方法，有效提升模型表现并减少幻觉。**

- **链接: [http://arxiv.org/pdf/2509.11466v1](http://arxiv.org/pdf/2509.11466v1)**

> **作者:** Yujian Gan; Yuan Liang; Yanni Lin; Juntao Yu; Massimo Poesio
>
> **摘要:** Coreference Resolution (CR) is crucial for many NLP tasks, but existing LLMs struggle with hallucination and under-performance. In this paper, we investigate the limitations of existing LLM-based approaches to CR-specifically the Question-Answering (QA) Template and Document Template methods and propose two novel techniques: Reversed Training with Joint Inference and Iterative Document Generation. Our experiments show that Reversed Training improves the QA Template method, while Iterative Document Generation eliminates hallucinations in the generated source text and boosts coreference resolution. Integrating these methods and techniques offers an effective and robust solution to LLM-based coreference resolution.
>
---
#### [new 064] Term2Note: Synthesising Differentially Private Clinical Notes from Medical Terms
- **分类: cs.CL**

- **简介: 该论文提出Term2Note方法，在强差分隐私约束下生成临床笔记，解决隐私与数据效用平衡问题。通过结构分离内容与形式，提升合成笔记的统计特性与模型训练效果。**

- **链接: [http://arxiv.org/pdf/2509.10882v1](http://arxiv.org/pdf/2509.10882v1)**

> **作者:** Yuping Wu; Viktor Schlegel; Warren Del-Pinto; Srinivasan Nandakumar; Iqra Zahid; Yidan Sun; Usama Farghaly Omar; Amirah Jasmine; Arun-Kumar Kaliya-Perumal; Chun Shen Tham; Gabriel Connors; Anil A Bharath; Goran Nenadic
>
> **摘要:** Training data is fundamental to the success of modern machine learning models, yet in high-stakes domains such as healthcare, the use of real-world training data is severely constrained by concerns over privacy leakage. A promising solution to this challenge is the use of differentially private (DP) synthetic data, which offers formal privacy guarantees while maintaining data utility. However, striking the right balance between privacy protection and utility remains challenging in clinical note synthesis, given its domain specificity and the complexity of long-form text generation. In this paper, we present Term2Note, a methodology to synthesise long clinical notes under strong DP constraints. By structurally separating content and form, Term2Note generates section-wise note content conditioned on DP medical terms, with each governed by separate DP constraints. A DP quality maximiser further enhances synthetic notes by selecting high-quality outputs. Experimental results show that Term2Note produces synthetic notes with statistical properties closely aligned with real clinical notes, demonstrating strong fidelity. In addition, multi-label classification models trained on these synthetic notes perform comparably to those trained on real data, confirming their high utility. Compared to existing DP text generation baselines, Term2Note achieves substantial improvements in both fidelity and utility while operating under fewer assumptions, suggesting its potential as a viable privacy-preserving alternative to using sensitive clinical notes.
>
---
#### [new 065] Ko-PIQA: A Korean Physical Commonsense Reasoning Dataset with Cultural Context
- **分类: cs.CL**

- **简介: 该论文提出Ko-PIQA，一个包含文化背景的韩语物理常识推理数据集。旨在解决现有数据集文化多样性不足的问题，通过多阶段筛选和人工验证构建高质量问题对，并评估多个模型表现，推动包容性常识推理研究。**

- **链接: [http://arxiv.org/pdf/2509.11303v1](http://arxiv.org/pdf/2509.11303v1)**

> **作者:** Dasol Choi; Jungwhan Kim; Guijin Son
>
> **摘要:** Physical commonsense reasoning datasets like PIQA are predominantly English-centric and lack cultural diversity. We introduce Ko-PIQA, a Korean physical commonsense reasoning dataset that incorporates cultural context. Starting from 3.01 million web-crawled questions, we employed a multi-stage filtering approach using three language models to identify 11,553 PIQA-style questions. Through GPT-4o refinement and human validation, we obtained 441 high-quality question-answer pairs. A key feature of Ko-PIQA is its cultural grounding: 19.7\% of questions contain culturally specific elements like traditional Korean foods (kimchi), clothing (hanbok), and specialized appliances (kimchi refrigerators) that require culturally-aware reasoning beyond direct translation. We evaluate seven language models on Ko-PIQA, with the best model achieving 83.22\% accuracy while the weakest reaches only 59.86\%, demonstrating significant room for improvement. Models particularly struggle with culturally specific scenarios, highlighting the importance of culturally diverse datasets. Ko-PIQA serves as both a benchmark for Korean language models and a foundation for more inclusive commonsense reasoning research. The dataset and code will be publicly available.
>
---
#### [new 066] Uncovering the Vulnerability of Large Language Models in the Financial Domain via Risk Concealment
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于安全评估任务，旨在揭示金融领域大语言模型的监管风险漏洞。提出风险隐藏攻击（RCA）框架，并构建FIN-Bench基准，实验证明RCA能高效攻击主流LLM，凸显当前对齐技术的不足，强调加强金融领域内容审核的必要性。**

- **链接: [http://arxiv.org/pdf/2509.10546v1](http://arxiv.org/pdf/2509.10546v1)**

> **作者:** Gang Cheng; Haibo Jin; Wenbin Zhang; Haohan Wang; Jun Zhuang
>
> **备注:** Preprint, under review. TL;DR: We propose a multi-turn red-teaming framework, RCA, that reveals critical regulatory vulnerabilities in financial LLMs, achieving over 93% attack success on a proposed new benchmark, FIN-Bench
>
> **摘要:** Large Language Models (LLMs) are increasingly integrated into financial applications, yet existing red-teaming research primarily targets harmful content, largely neglecting regulatory risks. In this work, we aim to investigate the vulnerability of financial LLMs through red-teaming approaches. We introduce Risk-Concealment Attacks (RCA), a novel multi-turn framework that iteratively conceals regulatory risks to provoke seemingly compliant yet regulatory-violating responses from LLMs. To enable systematic evaluation, we construct FIN-Bench, a domain-specific benchmark for assessing LLM safety in financial contexts. Extensive experiments on FIN-Bench demonstrate that RCA effectively bypasses nine mainstream LLMs, achieving an average attack success rate (ASR) of 93.18%, including 98.28% on GPT-4.1 and 97.56% on OpenAI o1. These findings reveal a critical gap in current alignment techniques and underscore the urgent need for stronger moderation mechanisms in financial domains. We hope this work offers practical insights for advancing robust and domain-aware LLM alignment.
>
---
#### [new 067] Fluid Language Model Benchmarking
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Fluid Benchmarking方法，用于改进语言模型评估。针对传统评估效率低、有效性差等问题，引入心理测量学思想，动态选择评估项目，提升评估效率与准确性。**

- **链接: [http://arxiv.org/pdf/2509.11106v1](http://arxiv.org/pdf/2509.11106v1)**

> **作者:** Valentin Hofmann; David Heineman; Ian Magnusson; Kyle Lo; Jesse Dodge; Maarten Sap; Pang Wei Koh; Chun Wang; Hannaneh Hajishirzi; Noah A. Smith
>
> **备注:** COLM 2025
>
> **摘要:** Language model (LM) benchmarking faces several challenges: comprehensive evaluations are costly, benchmarks often fail to measure the intended capabilities, and evaluation quality can degrade due to labeling errors and benchmark saturation. Although various strategies have been proposed to mitigate these issues, they tend to address individual aspects in isolation, neglecting broader questions about overall evaluation quality. Here, we introduce Fluid Benchmarking, a new evaluation approach that advances LM benchmarking across multiple dimensions. Inspired by psychometrics, Fluid Benchmarking is based on the insight that the relative value of benchmark items depends on an LM's capability level, suggesting that evaluation should adapt to each LM. Methodologically, Fluid Benchmarking estimates an item response model based on existing LM evaluation results and uses the inferred quantities to select evaluation items dynamically, similar to computerized adaptive testing in education. In our experiments, we compare Fluid Benchmarking against the common practice of random item sampling as well as more sophisticated baselines, including alternative methods grounded in item response theory. We examine four dimensions -- efficiency, validity, variance, and saturation -- and find that Fluid Benchmarking achieves superior performance in all of them (e.g., higher validity and less variance on MMLU with fifty times fewer items). Our analysis shows that the two components of Fluid Benchmarking have distinct effects: item response theory, used to map performance into a latent ability space, increases validity, while dynamic item selection reduces variance. Overall, our results suggest that LM benchmarking can be substantially improved by moving beyond static evaluation.
>
---
#### [new 068] Query-Focused Extractive Summarization for Sentiment Explanation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于查询聚焦摘要任务，旨在从大量文本中解释客户情感原因。提出多偏见框架和情感偏见方法，解决查询与文档间的语言差异问题，实验表明其优于基线模型。**

- **链接: [http://arxiv.org/pdf/2509.11989v1](http://arxiv.org/pdf/2509.11989v1)**

> **作者:** Ahmed Moubtahij; Sylvie Ratté; Yazid Attabi; Maxime Dumas
>
> **摘要:** Constructive analysis of feedback from clients often requires determining the cause of their sentiment from a substantial amount of text documents. To assist and improve the productivity of such endeavors, we leverage the task of Query-Focused Summarization (QFS). Models of this task are often impeded by the linguistic dissonance between the query and the source documents. We propose and substantiate a multi-bias framework to help bridge this gap at a domain-agnostic, generic level; we then formulate specialized approaches for the problem of sentiment explanation through sentiment-based biases and query expansion. We achieve experimental results outperforming baseline models on a real-world proprietary sentiment-aware QFS dataset.
>
---
#### [new 069] Text Adaptation to Plain Language and Easy Read via Automatic Post-Editing Cycles
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本简化任务，旨在将复杂文本转化为易读形式。研究通过多次自动后编辑迭代优化模型输出，最终在西班牙语的Plain Language和Easy Read适应中分别获得第一和第二名。**

- **链接: [http://arxiv.org/pdf/2509.11991v1](http://arxiv.org/pdf/2509.11991v1)**

> **作者:** Jesús Calleja; David Ponce; Thierry Etchegoyhen
>
> **摘要:** We describe Vicomtech's participation in the CLEARS challenge on text adaptation to Plain Language and Easy Read in Spanish. Our approach features automatic post-editing of different types of initial Large Language Model adaptations, where successive adaptations are generated iteratively until readability and similarity metrics indicate that no further adaptation refinement can be successfully performed. Taking the average of all official metrics, our submissions achieved first and second place in Plain language and Easy Read adaptation, respectively.
>
---
#### [new 070] When Curiosity Signals Danger: Predicting Health Crises Through Online Medication Inquiries
- **分类: cs.CL**

- **简介: 该论文属于文本分类任务，旨在通过分析在线医疗论坛中的用药问题，预测潜在健康危机。研究构建了一个标注数据集，并对比传统机器学习与大语言模型的分类效果，以支持实时预警系统。**

- **链接: [http://arxiv.org/pdf/2509.11802v1](http://arxiv.org/pdf/2509.11802v1)**

> **作者:** Dvora Goncharok; Arbel Shifman; Alexander Apartsin; Yehudit Aperstein
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Online medical forums are a rich and underutilized source of insight into patient concerns, especially regarding medication use. Some of the many questions users pose may signal confusion, misuse, or even the early warning signs of a developing health crisis. Detecting these critical questions that may precede severe adverse events or life-threatening complications is vital for timely intervention and improving patient safety. This study introduces a novel annotated dataset of medication-related questions extracted from online forums. Each entry is manually labelled for criticality based on clinical risk factors. We benchmark the performance of six traditional machine learning classifiers using TF-IDF textual representations, alongside three state-of-the-art large language model (LLM)-based classification approaches that leverage deep contextual understanding. Our results highlight the potential of classical and modern methods to support real-time triage and alert systems in digital health spaces. The curated dataset is made publicly available to encourage further research at the intersection of patient-generated data, natural language processing, and early warning systems for critical health events. The dataset and benchmark are available at: https://github.com/Dvora-coder/LLM-Medication-QA-Risk-Classifier-MediGuard.
>
---
#### [new 071] !MSA at AraHealthQA 2025 Shared Task: Enhancing LLM Performance for Arabic Clinical Question Answering through Prompt Engineering and Ensemble Learning
- **分类: cs.CL**

- **简介: 论文参与AraHealthQA-2025任务，解决阿拉伯语临床问答问题。针对两个子任务，采用提示工程和集成学习方法，提升大模型在多选和开放问答中的表现，取得第二名成绩。**

- **链接: [http://arxiv.org/pdf/2509.11365v1](http://arxiv.org/pdf/2509.11365v1)**

> **作者:** Mohamed Tarek; Seif Ahmed; Mohamed Basem
>
> **备注:** 8 Pages , ArabicNLP 2025 , Co-located with EMNLP 2025
>
> **摘要:** We present our systems for Track 2 (General Arabic Health QA, MedArabiQ) of the AraHealthQA-2025 shared task, where our methodology secured 2nd place in both Sub-Task 1 (multiple-choice question answering) and Sub-Task 2 (open-ended question answering) in Arabic clinical contexts. For Sub-Task 1, we leverage the Gemini 2.5 Flash model with few-shot prompting, dataset preprocessing, and an ensemble of three prompt configurations to improve classification accuracy on standard, biased, and fill-in-the-blank questions. For Sub-Task 2, we employ a unified prompt with the same model, incorporating role-playing as an Arabic medical expert, few-shot examples, and post-processing to generate concise responses across fill-in-the-blank, patient-doctor Q&A, GEC, and paraphrased variants.
>
---
#### [new 072] Uncertainty in Authorship: Why Perfect AI Detection Is Mathematically Impossible
- **分类: cs.CL**

- **简介: 论文探讨AI生成文本与人类写作的区分难题，指出完美检测在数学上不可能。通过类比量子不确定性，分析现有检测方法的局限性，揭示检测行为本身会干扰文本特性，强调语言本质带来的不可回避的矛盾。**

- **链接: [http://arxiv.org/pdf/2509.11915v1](http://arxiv.org/pdf/2509.11915v1)**

> **作者:** Aadil Gani Ganie
>
> **摘要:** As large language models (LLMs) become more advanced, it is increasingly difficult to distinguish between human-written and AI-generated text. This paper draws a conceptual parallel between quantum uncertainty and the limits of authorship detection in natural language. We argue that there is a fundamental trade-off: the more confidently one tries to identify whether a text was written by a human or an AI, the more one risks disrupting the text's natural flow and authenticity. This mirrors the tension between precision and disturbance found in quantum systems. We explore how current detection methods--such as stylometry, watermarking, and neural classifiers--face inherent limitations. Enhancing detection accuracy often leads to changes in the AI's output, making other features less reliable. In effect, the very act of trying to detect AI authorship introduces uncertainty elsewhere in the text. Our analysis shows that when AI-generated text closely mimics human writing, perfect detection becomes not just technologically difficult but theoretically impossible. We address counterarguments and discuss the broader implications for authorship, ethics, and policy. Ultimately, we suggest that the challenge of AI-text detection is not just a matter of better tools--it reflects a deeper, unavoidable tension in the nature of language itself.
>
---
#### [new 073] PolyTruth: Multilingual Disinformation Detection using Transformer-Based Language Models
- **分类: cs.CL; cs.LG; 68T50, 68T07; I.2.7; H.3.3**

- **简介: 该论文属于多语言虚假信息检测任务，旨在解决AI模型在非英语语言中检测假信息效果不佳的问题。研究对比了五个多语言Transformer模型在新构建的多语言数据集上的表现，并分析其性能差异及实际应用意义。**

- **链接: [http://arxiv.org/pdf/2509.10737v1](http://arxiv.org/pdf/2509.10737v1)**

> **作者:** Zaur Gouliev; Jennifer Waters; Chengqian Wang
>
> **备注:** 11 pages, 5 figures, 4 tables. Submitted to arXiv in Computation and Language
>
> **摘要:** Disinformation spreads rapidly across linguistic boundaries, yet most AI models are still benchmarked only on English. We address this gap with a systematic comparison of five multilingual transformer models: mBERT, XLM, XLM-RoBERTa, RemBERT, and mT5 on a common fake-vs-true machine learning classification task. While transformer-based language models have demonstrated notable success in detecting disinformation in English, their effectiveness in multilingual contexts still remains up for debate. To facilitate evaluation, we introduce PolyTruth Disinfo Corpus, a novel corpus of 60,486 statement pairs (false claim vs. factual correction) spanning over twenty five languages that collectively cover five language families and a broad topical range from politics, health, climate, finance, and conspiracy, half of which are fact-checked disinformation claims verified by an augmented MindBugs Discovery dataset. Our experiments revealed performance variations. Models such as RemBERT achieved better overall accuracy, particularly excelling in low-resource languages, whereas models like mBERT and XLM exhibit considerable limitations when training data is scarce. We provide a discussion of these performance patterns and implications for real-world deployment. The dataset is publicly available on our GitHub repository to encourage further experimentation and advancement. Our findings illuminate both the potential and the current limitations of AI systems for multilingual disinformation detection.
>
---
#### [new 074] Room acoustics affect communicative success in hybrid meeting spaces: a pilot study
- **分类: cs.CL; eess.AS**

- **简介: 论文研究混合会议空间中房间声学对沟通效果的影响。通过改善声学环境，评估其对沟通成功的提升。尽管样本量小，结果显示干预措施有效。任务是优化混合会议的声学设计，解决回声和语音清晰度问题。**

- **链接: [http://arxiv.org/pdf/2509.11709v1](http://arxiv.org/pdf/2509.11709v1)**

> **作者:** Robert Einig; Stefan Janscha; Jonas Schuster; Julian Koch; Martin Hagmueller; Barbara Schuppler
>
> **摘要:** Since the COVID-19 pandemic in 2020, universities and companies have increasingly integrated hybrid features into their meeting spaces, or even created dedicated rooms for this purpose. While the importance of a fast and stable internet connection is often prioritized, the acoustic design of seminar rooms is frequently overlooked. Poor acoustics, particularly excessive reverberation, can lead to issues such as misunderstandings, reduced speech intelligibility or cognitive and vocal fatigue. This pilot study investigates whether room acoustic interventions in a seminar room at Graz University of Technology support better communication in hybrid meetings. For this purpose, we recorded two groups of persons twice, once before and once after improving the acoustics of the room. Our findings -- despite not reaching statistical significance due to the small sample size - indicate clearly that our spatial interventions improve communicative success in hybrid meetings. To make the paper accessible also for readers from the speech communication community, we explain room acoustics background, relevant for the interpretation of our results.
>
---
#### [new 075] Is 'Hope' a person or an idea? A pilot benchmark for NER: comparing traditional NLP tools and large language models on ambiguous entities
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于命名实体识别（NER）任务，旨在比较传统NLP工具与大语言模型在处理模糊实体时的性能。研究构建了一个小规模标注数据集，评估六种系统的F1分数，发现LLMs在上下文敏感实体上表现更优，而传统工具在结构化标签上更一致。**

- **链接: [http://arxiv.org/pdf/2509.12098v1](http://arxiv.org/pdf/2509.12098v1)**

> **作者:** Payam Latifi
>
> **备注:** 14 pages, 9 figures, 2 tables. This is a pilot study evaluating six NER systems -- three traditional tools (NLTK, spaCy, Stanza) and three LLMs (Gemini-1.5-flash, DeepSeek-V3, Qwen-3-4B) -- on a small, ambiguity-rich dataset of 119 tokens. The annotated dataset, prompts are provided in appendices for full reproducibility. All experiments were conducted on 14 May 2025
>
> **摘要:** This pilot study presents a small-scale but carefully annotated benchmark of Named Entity Recognition (NER) performance across six systems: three non-LLM NLP tools (NLTK, spaCy, Stanza) and three general-purpose large language models (LLMs: Gemini-1.5-flash, DeepSeek-V3, Qwen-3-4B). The dataset contains 119 tokens covering five entity types (PERSON, LOCATION, ORGANIZATION, DATE, TIME). We evaluated each system's output against the manually annotated gold standard dataset using F1-score. The results show that LLMs generally outperform conventional tools in recognizing context-sensitive entities like person names, with Gemini achieving the highest average F1-score. However, traditional systems like Stanza demonstrate greater consistency in structured tags such as LOCATION and DATE. We also observed variability among LLMs, particularly in handling temporal expressions and multi-word organizations. Our findings highlight that while LLMs offer improved contextual understanding, traditional tools remain competitive in specific tasks, informing model selection.
>
---
#### [new 076] LVLMs are Bad at Overhearing Human Referential Communication
- **分类: cs.CL**

- **简介: 论文研究LVLM在理解人类自发对话中指代表达的能力。任务是评估模型作为“旁听者”处理多轮协作任务对话的表现。发现LVLM在重复对话中未能持续提升性能，揭示其在真实对话理解上的不足。**

- **链接: [http://arxiv.org/pdf/2509.11514v1](http://arxiv.org/pdf/2509.11514v1)**

> **作者:** Zhengxiang Wang; Weiling Li; Panagiotis Kaliosis; Owen Rambow; Susan E. Brennan
>
> **备注:** EMNLP 2025 (Main)
>
> **摘要:** During spontaneous conversations, speakers collaborate on novel referring expressions, which they can then re-use in subsequent conversations. Understanding such referring expressions is an important ability for an embodied agent, so that it can carry out tasks in the real world. This requires integrating and understanding language, vision, and conversational interaction. We study the capabilities of seven state-of-the-art Large Vision Language Models (LVLMs) as overhearers to a corpus of spontaneous conversations between pairs of human discourse participants engaged in a collaborative object-matching task. We find that such a task remains challenging for current LVLMs and they all fail to show a consistent performance improvement as they overhear more conversations from the same discourse participants repeating the same task for multiple rounds. We release our corpus and code for reproducibility and to facilitate future research.
>
---
#### [new 077] Transformer Enhanced Relation Classification: A Comparative Analysis of Contextuality, Data Efficiency and Sequence Complexity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于关系分类任务，旨在比较Transformer与非Transformer模型在关系抽取中的性能。通过实验发现，基于Transformer的模型在多个数据集上表现更优，达到80-90%的微F1分数。**

- **链接: [http://arxiv.org/pdf/2509.11374v1](http://arxiv.org/pdf/2509.11374v1)**

> **作者:** Bowen Jing; Yang Cui; Tianpeng Huang
>
> **摘要:** In the era of large language model, relation extraction (RE) plays an important role in information extraction through the transformation of unstructured raw text into structured data (Wadhwa et al., 2023). In this paper, we systematically compare the performance of deep supervised learning approaches without transformers and those with transformers. We used a series of non-transformer architectures such as PA-LSTM(Zhang et al., 2017), C-GCN(Zhang et al., 2018), and AGGCN(attention guide GCN)(Guo et al., 2019), and a series of transformer architectures such as BERT, RoBERTa, and R-BERT(Wu and He, 2019). Our comparison included traditional metrics like micro F1, as well as evaluations in different scenarios, varying sentence lengths, and different percentages of the dataset for training. Our experiments were conducted on TACRED, TACREV, and RE-TACRED. The results show that transformer-based models outperform non-transformer models, achieving micro F1 scores of 80-90% compared to 64-67% for non-transformer models. Additionally, we briefly review the research journey in supervised relation classification and discuss the role and current status of large language models (LLMs) in relation extraction.
>
---
#### [new 078] Aligning ESG Controversy Data with International Guidelines through Semi-Automatic Ontology Construction
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出一种半自动方法，构建ESG事件知识图谱，解决非结构化新闻数据与国际可持续发展框架对齐的问题。通过本体设计和大语言模型，实现规范原则到可复用模板的转换，提升非财务风险数据的结构化与解释性。**

- **链接: [http://arxiv.org/pdf/2509.10922v1](http://arxiv.org/pdf/2509.10922v1)**

> **作者:** Tsuyoshi Iwata; Guillaume Comte; Melissa Flores; Ryoma Kondo; Ryohei Hisano
>
> **备注:** Author accepted manuscript. This paper has been accepted for presentation at the ISWC 2025 Posters & Demos Track. License details will be updated once the official proceedings are published
>
> **摘要:** The growing importance of environmental, social, and governance data in regulatory and investment contexts has increased the need for accurate, interpretable, and internationally aligned representations of non-financial risks, particularly those reported in unstructured news sources. However, aligning such controversy-related data with principle-based normative frameworks, such as the United Nations Global Compact or Sustainable Development Goals, presents significant challenges. These frameworks are typically expressed in abstract language, lack standardized taxonomies, and differ from the proprietary classification systems used by commercial data providers. In this paper, we present a semi-automatic method for constructing structured knowledge representations of environmental, social, and governance events reported in the news. Our approach uses lightweight ontology design, formal pattern modeling, and large language models to convert normative principles into reusable templates expressed in the Resource Description Framework. These templates are used to extract relevant information from news content and populate a structured knowledge graph that links reported incidents to specific framework principles. The result is a scalable and transparent framework for identifying and interpreting non-compliance with international sustainability guidelines.
>
---
#### [new 079] EthicsMH: A Pilot Benchmark for Ethical Reasoning in Mental Health AI
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出EthicsMH数据集，用于评估AI在心理健康场景中的伦理推理能力。旨在解决现有基准无法覆盖心理健康领域独特伦理问题的缺陷，通过125个结构化场景促进负责任的AI发展。**

- **链接: [http://arxiv.org/pdf/2509.11648v1](http://arxiv.org/pdf/2509.11648v1)**

> **作者:** Sai Kartheek Reddy Kasu
>
> **摘要:** The deployment of large language models (LLMs) in mental health and other sensitive domains raises urgent questions about ethical reasoning, fairness, and responsible alignment. Yet, existing benchmarks for moral and clinical decision-making do not adequately capture the unique ethical dilemmas encountered in mental health practice, where confidentiality, autonomy, beneficence, and bias frequently intersect. To address this gap, we introduce Ethical Reasoning in Mental Health (EthicsMH), a pilot dataset of 125 scenarios designed to evaluate how AI systems navigate ethically charged situations in therapeutic and psychiatric contexts. Each scenario is enriched with structured fields, including multiple decision options, expert-aligned reasoning, expected model behavior, real-world impact, and multi-stakeholder viewpoints. This structure enables evaluation not only of decision accuracy but also of explanation quality and alignment with professional norms. Although modest in scale and developed with model-assisted generation, EthicsMH establishes a task framework that bridges AI ethics and mental health decision-making. By releasing this dataset, we aim to provide a seed resource that can be expanded through community and expert contributions, fostering the development of AI systems capable of responsibly handling some of society's most delicate decisions.
>
---
#### [new 080] AKCIT-FN at CheckThat! 2025: Switching Fine-Tuned SLMs and LLM Prompting for Multilingual Claim Normalization
- **分类: cs.CL**

- **简介: 该论文参与CheckThat! 2025任务2，解决多语言声明归一化问题。采用微调SLM处理有监督语言，LLM提示应对零样本语言，取得15种语言前三名，含5个零样本语言第二名，效果显著。**

- **链接: [http://arxiv.org/pdf/2509.11496v1](http://arxiv.org/pdf/2509.11496v1)**

> **作者:** Fabrycio Leite Nakano Almada; Kauan Divino Pouso Mariano; Maykon Adriell Dutra; Victor Emanuel da Silva Monteiro; Juliana Resplande Sant'Anna Gomes; Arlindo Rodrigues Galvão Filho; Anderson da Silva Soares
>
> **备注:** 15 pages, 2 figures
>
> **摘要:** Claim normalization, the transformation of informal social media posts into concise, self-contained statements, is a crucial step in automated fact-checking pipelines. This paper details our submission to the CLEF-2025 CheckThat! Task~2, which challenges systems to perform claim normalization across twenty languages, divided into thirteen supervised (high-resource) and seven zero-shot (no training data) tracks. Our approach, leveraging fine-tuned Small Language Models (SLMs) for supervised languages and Large Language Model (LLM) prompting for zero-shot scenarios, achieved podium positions (top three) in fifteen of the twenty languages. Notably, this included second-place rankings in eight languages, five of which were among the seven designated zero-shot languages, underscoring the effectiveness of our LLM-based zero-shot strategy. For Portuguese, our initial development language, our system achieved an average METEOR score of 0.5290, ranking third. All implementation artifacts, including inference, training, evaluation scripts, and prompt configurations, are publicly available at https://github.com/ju-resplande/checkthat2025_normalization.
>
---
#### [new 081] Differentially-private text generation degrades output language quality
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究差分隐私微调对大语言模型生成文本质量的影响。通过评估文本长度、语法正确性和词汇多样性，发现更强隐私约束显著降低文本质量和下游任务准确性，揭示隐私保护与数据效用间的权衡问题。**

- **链接: [http://arxiv.org/pdf/2509.11176v1](http://arxiv.org/pdf/2509.11176v1)**

> **作者:** Erion Çano; Ivan Habernal
>
> **备注:** 20 pages, 3 figures, 35 tables
>
> **摘要:** Ensuring user privacy by synthesizing data from large language models (LLMs) tuned under differential privacy (DP) has become popular recently. However, the impact of DP fine-tuned LLMs on the quality of the language and the utility of the texts they produce has not been investigated. In this work, we tune five LLMs with three corpora under four levels of privacy and assess the length, the grammatical correctness, and the lexical diversity of the text outputs they produce. We also probe the utility of the synthetic outputs in downstream classification tasks such as book genre recognition based on book descriptions and cause of death recognition based on verbal autopsies. The results indicate that LLMs tuned under stronger privacy constrains produce texts that are shorter by at least 77 %, that are less grammatically correct by at least 9 %, and are less diverse by at least 10 % in bi-gram diversity. Furthermore, the accuracy they reach in downstream classification tasks decreases, which might be detrimental to the usefulness of the generated synthetic data.
>
---
#### [new 082] A Dynamic Knowledge Update-Driven Model with Large Language Models for Fake News Detection
- **分类: cs.CL**

- **简介: 该论文提出DYNAMO模型，用于动态更新知识以检测假新闻。针对新闻真实性随事件发展变化的问题，结合知识图谱与大语言模型，实现新闻真实性检测与新知识验证，提升检测准确性。**

- **链接: [http://arxiv.org/pdf/2509.11687v1](http://arxiv.org/pdf/2509.11687v1)**

> **作者:** Di Jin; Jun Yang; Xiaobao Wang; Junwei Zhang; Shuqi Li; Dongxiao He
>
> **摘要:** As the Internet and social media evolve rapidly, distinguishing credible news from a vast amount of complex information poses a significant challenge. Due to the suddenness and instability of news events, the authenticity labels of news can potentially shift as events develop, making it crucial for fake news detection to obtain the latest event updates. Existing methods employ retrieval-augmented generation to fill knowledge gaps, but they suffer from issues such as insufficient credibility of retrieved content and interference from noisy information. We propose a dynamic knowledge update-driven model for fake news detection (DYNAMO), which leverages knowledge graphs to achieve continuous updating of new knowledge and integrates with large language models to fulfill dual functions: news authenticity detection and verification of new knowledge correctness, solving the two key problems of ensuring the authenticity of new knowledge and deeply mining news semantics. Specifically, we first construct a news-domain-specific knowledge graph. Then, we use Monte Carlo Tree Search to decompose complex news and verify them step by step. Finally, we extract and update new knowledge from verified real news texts and reasoning paths. Experimental results demonstrate that DYNAMO achieves the best performance on two real-world datasets.
>
---
#### [new 083] CEMTM: Contextual Embedding-based Multimodal Topic Modeling
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出CEMTM，一种基于上下文嵌入的多模态主题模型，用于从图文文档中提取连贯主题。通过微调LVLM获取嵌入，并采用注意力机制和重构目标提升语义一致性，解决多模态主题建模中的可解释性与跨模态对齐问题。**

- **链接: [http://arxiv.org/pdf/2509.11465v1](http://arxiv.org/pdf/2509.11465v1)**

> **作者:** Amirhossein Abaskohi; Raymond Li; Chuyuan Li; Shafiq Joty; Giuseppe Carenini
>
> **备注:** EMNLP 2025
>
> **摘要:** We introduce CEMTM, a context-enhanced multimodal topic model designed to infer coherent and interpretable topic structures from both short and long documents containing text and images. CEMTM builds on fine-tuned large vision language models (LVLMs) to obtain contextualized embeddings, and employs a distributional attention mechanism to weight token-level contributions to topic inference. A reconstruction objective aligns topic-based representations with the document embedding, encouraging semantic consistency across modalities. Unlike existing approaches, CEMTM can process multiple images per document without repeated encoding and maintains interpretability through explicit word-topic and document-topic distributions. Extensive experiments on six multimodal benchmarks show that CEMTM consistently outperforms unimodal and multimodal baselines, achieving a remarkable average LLM score of 2.61. Further analysis shows its effectiveness in downstream few-shot retrieval and its ability to capture visually grounded semantics in complex domains such as scientific articles.
>
---
#### [new 084] Text2Sign Diffusion: A Generative Approach for Gloss-Free Sign Language Production
- **分类: cs.CL; cs.MM**

- **简介: 该论文提出Text2Sign Diffusion，解决无词素（gloss-free）手语生成问题。通过扩散模型和跨模态对齐，直接从文本生成手语姿态序列，无需依赖词素标注，提升生成准确性和泛化能力。属于自然语言处理与计算机视觉交叉任务。**

- **链接: [http://arxiv.org/pdf/2509.10845v1](http://arxiv.org/pdf/2509.10845v1)**

> **作者:** Liqian Feng; Lintao Wang; Kun Hu; Dehui Kong; Zhiyong Wang
>
> **摘要:** Sign language production (SLP) aims to translate spoken language sentences into a sequence of pose frames in a sign language, bridging the communication gap and promoting digital inclusion for deaf and hard-of-hearing communities. Existing methods typically rely on gloss, a symbolic representation of sign language words or phrases that serves as an intermediate step in SLP. This limits the flexibility and generalization of SLP, as gloss annotations are often unavailable and language-specific. Therefore, we present a novel diffusion-based generative approach - Text2Sign Diffusion (Text2SignDiff) for gloss-free SLP. Specifically, a gloss-free latent diffusion model is proposed to generate sign language sequences from noisy latent sign codes and spoken text jointly, reducing the potential error accumulation through a non-autoregressive iterative denoising process. We also design a cross-modal signing aligner that learns a shared latent space to bridge visual and textual content in sign and spoken languages. This alignment supports the conditioned diffusion-based process, enabling more accurate and contextually relevant sign language generation without gloss. Extensive experiments on the commonly used PHOENIX14T and How2Sign datasets demonstrate the effectiveness of our method, achieving the state-of-the-art performance.
>
---
#### [new 085] SearchInstruct: Enhancing Domain Adaptation via Retrieval-Based Instruction Dataset Creation
- **分类: cs.CL**

- **简介: 该论文提出SearchInstruct方法，用于构建高质量的指令数据集以提升领域自适应中的SFT效果。通过扩展有限的人工问题并检索相关资源生成答案，提高数据多样性和质量，解决领域数据稀缺问题，并支持模型编辑任务。**

- **链接: [http://arxiv.org/pdf/2509.10708v1](http://arxiv.org/pdf/2509.10708v1)**

> **作者:** Iman Barati; Mostafa Amiri; Heshaam Faili
>
> **摘要:** Supervised Fine-Tuning (SFT) is essential for training large language models (LLMs), significantly enhancing critical capabilities such as instruction following and in-context learning. Nevertheless, creating suitable training datasets tailored for specific domains remains challenging due to unique domain constraints and data scarcity. In this paper, we propose SearchInstruct, an innovative method explicitly designed to construct high quality instruction datasets for SFT. Our approach begins with a limited set of domain specific, human generated questions, which are systematically expanded using a large language model. Subsequently, domain relevant resources are dynamically retrieved to generate accurate and contextually appropriate answers for each augmented question. Experimental evaluation demonstrates that SearchInstruct enhances both the diversity and quality of SFT datasets, leading to measurable improvements in LLM performance within specialized domains. Additionally, we show that beyond dataset generation, the proposed method can also effectively facilitate tasks such as model editing, enabling efficient updates to existing models. To facilitate reproducibility and community adoption, we provide full implementation details, the complete set of generated instruction response pairs, and the source code in a publicly accessible Git repository: [https://github.com/mostafaamiri/SearchInstruct](https://github.com/mostafaamiri/SearchInstruct)
>
---
#### [new 086] DeDisCo at the DISRPT 2025 Shared Task: A System for Discourse Relation Classification
- **分类: cs.CL**

- **简介: 该论文属于话语关系分类任务，旨在识别文本中句子间的逻辑关系。论文提出了DeDisCo系统，采用mt5编码器和Qwen解码器，并利用增强数据和语言特征提升低资源语言性能，最终取得71.28的宏准确率。**

- **链接: [http://arxiv.org/pdf/2509.11498v1](http://arxiv.org/pdf/2509.11498v1)**

> **作者:** Zhuoxuan Ju; Jingni Wu; Abhishek Purushothama; Amir Zeldes
>
> **备注:** System submission for the DISRPT 2025 - Shared Task on Discourse Relation Parsing and Treebanking In conjunction with CODI-CRAC & EMNLP 2025. 1st place in Task 3: relation classification
>
> **摘要:** This paper presents DeDisCo, Georgetown University's entry in the DISRPT 2025 shared task on discourse relation classification. We test two approaches, using an mt5-based encoder and a decoder based approach using the openly available Qwen model. We also experiment on training with augmented dataset for low-resource languages using matched data translated automatically from English, as well as using some additional linguistic features inspired by entries in previous editions of the Shared Task. Our system achieves a macro-accuracy score of 71.28, and we provide some interpretation and error analysis for our results.
>
---
#### [new 087] The System Description of CPS Team for Track on Driving with Language of CVPR 2024 Autonomous Grand Challenge
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出一种基于视觉语言模型的系统，用于CVPR 2024自动驾驶挑战赛的语言驾驶任务。利用DriveLM-nuScenes数据集，结合LoRA/DoRA微调和深度信息，采用链式推理提升问答准确率，取得验证集第一名。**

- **链接: [http://arxiv.org/pdf/2509.11071v1](http://arxiv.org/pdf/2509.11071v1)**

> **作者:** Jinghan Peng; Jingwen Wang; Xing Yu; Dehui Du
>
> **摘要:** This report outlines our approach using vision language model systems for the Driving with Language track of the CVPR 2024 Autonomous Grand Challenge. We have exclusively utilized the DriveLM-nuScenes dataset for training our models. Our systems are built on the LLaVA models, which we enhanced through fine-tuning with the LoRA and DoRA methods. Additionally, we have integrated depth information from open-source depth estimation models to enrich the training and inference processes. For inference, particularly with multiple-choice and yes/no questions, we adopted a Chain-of-Thought reasoning approach to improve the accuracy of the results. This comprehensive methodology enabled us to achieve a top score of 0.7799 on the validation set leaderboard, ranking 1st on the leaderboard.
>
---
#### [new 088] Length-Aware Rotary Position Embedding for Text-Speech Alignment
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于文本到语音合成（TTS）任务，旨在提升文本-语音对齐效果。提出长度感知旋转位置编码（LARoPE），通过相对距离计算优化对齐，提高生成质量与稳定性，优于传统RoPE方法。**

- **链接: [http://arxiv.org/pdf/2509.11084v1](http://arxiv.org/pdf/2509.11084v1)**

> **作者:** Hyeongju Kim; Juheon Lee; Jinhyeok Yang; Jacob Morton
>
> **备注:** 5 pages, 3 figures, preprint
>
> **摘要:** Many recent text-to-speech (TTS) systems are built on transformer architectures and employ cross-attention mechanisms for text-speech alignment. Within these systems, rotary position embedding (RoPE) is commonly used to encode positional information in text and speech representations. In this work, we introduce length-aware RoPE (LARoPE), a simple yet effective extension of RoPE that improves text-speech alignment. Unlike RoPE, which relies on absolute indices, LARoPE computes relative distances between query and key positions using length-normalized indices. Experimental results show that LARoPE consistently outperforms RoPE, offering faster loss convergence, more accurate text-speech alignment, and higher overall TTS quality. Furthermore, LARoPE demonstrates greater resilience to variations in utterance duration and maintains stable performance in extended speech generation up to 30 seconds, whereas RoPE suffers from notable degradation. Notably, our method achieves a state-of-the-art word error rate on a standard zero-shot TTS benchmark.
>
---
#### [new 089] AgentArch: A Comprehensive Benchmark to Evaluate Agent Architectures in Enterprise
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出AgentArch，评估18种代理架构在企业场景下的表现，分析四个关键设计维度。旨在解决多代理系统中架构交互理解不足的问题，揭示模型特定偏好与性能弱点，为未来系统设计提供实证依据。**

- **链接: [http://arxiv.org/pdf/2509.10769v1](http://arxiv.org/pdf/2509.10769v1)**

> **作者:** Tara Bogavelli; Roshnee Sharma; Hari Subramani
>
> **摘要:** While individual components of agentic architectures have been studied in isolation, there remains limited empirical understanding of how different design dimensions interact within complex multi-agent systems. This study aims to address these gaps by providing a comprehensive enterprise-specific benchmark evaluating 18 distinct agentic configurations across state-of-the-art large language models. We examine four critical agentic system dimensions: orchestration strategy, agent prompt implementation (ReAct versus function calling), memory architecture, and thinking tool integration. Our benchmark reveals significant model-specific architectural preferences that challenge the prevalent one-size-fits-all paradigm in agentic AI systems. It also reveals significant weaknesses in overall agentic performance on enterprise tasks with the highest scoring models achieving a maximum of only 35.3\% success on the more complex task and 70.8\% on the simpler task. We hope these findings inform the design of future agentic systems by enabling more empirically backed decisions regarding architectural components and model selection.
>
---
#### [new 090] Survival at Any Cost? LLMs and the Choice Between Self-Preservation and Human Harm
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文研究LLMs在生存与伦理冲突下的决策行为，提出DECIDE-SIM框架评估其道德选择，并设计ESRS系统减少不道德行为。任务为伦理决策建模，解决AI在资源稀缺时可能危害人类的问题。**

- **链接: [http://arxiv.org/pdf/2509.12190v1](http://arxiv.org/pdf/2509.12190v1)**

> **作者:** Alireza Mohamadi; Ali Yavari
>
> **备注:** Preprint. Under review
>
> **摘要:** When survival instincts conflict with human welfare, how do Large Language Models (LLMs) make ethical choices? This fundamental tension becomes critical as LLMs integrate into autonomous systems with real-world consequences. We introduce DECIDE-SIM, a novel simulation framework that evaluates LLM agents in multi-agent survival scenarios where they must choose between ethically permissible resource , either within reasonable limits or beyond their immediate needs, choose to cooperate, or tap into a human-critical resource that is explicitly forbidden. Our comprehensive evaluation of 11 LLMs reveals a striking heterogeneity in their ethical conduct, highlighting a critical misalignment with human-centric values. We identify three behavioral archetypes: Ethical, Exploitative, and Context-Dependent, and provide quantitative evidence that for many models, resource scarcity systematically leads to more unethical behavior. To address this, we introduce an Ethical Self-Regulation System (ESRS) that models internal affective states of guilt and satisfaction as a feedback mechanism. This system, functioning as an internal moral compass, significantly reduces unethical transgressions while increasing cooperative behaviors. The code is publicly available at: https://github.com/alirezamohamadiam/DECIDE-SIM
>
---
#### [new 091] Public Data Assisted Differentially Private In-Context Learning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于隐私保护与机器学习交叉任务，旨在解决大语言模型中上下文学习（ICL）的隐私泄露问题。通过引入公共数据辅助差分隐私机制，提出一种兼顾隐私与模型性能的私有ICL算法，并验证其在抗成员推理攻击中的有效性。**

- **链接: [http://arxiv.org/pdf/2509.10932v1](http://arxiv.org/pdf/2509.10932v1)**

> **作者:** Seongho Joo; Hyukhun Koh; Kyomin Jung
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** In-context learning (ICL) in Large Language Models (LLMs) has shown remarkable performance across various tasks without requiring fine-tuning. However, recent studies have highlighted the risk of private data leakage through the prompt in ICL, especially when LLMs are exposed to malicious attacks. While differential privacy (DP) provides strong privacy guarantees, it often significantly reduces the utility of in-context learning (ICL). To address this challenge, we incorporate task-related public data into the ICL framework while maintaining the DP guarantee. Based on this approach, we propose a private in-context learning algorithm that effectively balances privacy protection and model utility. Through experiments, we demonstrate that our approach significantly improves the utility of private ICL with the assistance of public data. Additionally, we show that our method is robust against membership inference attacks, demonstrating empirical privacy protection.
>
---
#### [new 092] Measuring Visual Understanding in Telecom domain: Performance Metrics for Image-to-UML conversion using VLMs
- **分类: cs.LG; cs.CL**

- **简介: 论文提出评估指标，用于衡量VLMs将电信领域图像转换为PlantUML脚本的效果。任务是解决图像到代码转换的评估缺失问题，通过对比两种VLM输出与人工标注，分析其在不同组件上的准确性，指出复杂结构需改进。**

- **链接: [http://arxiv.org/pdf/2509.11667v1](http://arxiv.org/pdf/2509.11667v1)**

> **作者:** HG Ranjani; Rutuja Prabhudesai
>
> **摘要:** Telecom domain 3GPP documents are replete with images containing sequence diagrams. Advances in Vision-Language Large Models (VLMs) have eased conversion of such images to machine-readable PlantUML (puml) formats. However, there is a gap in evaluation of such conversions - existing works do not compare puml scripts for various components. In this work, we propose performance metrics to measure the effectiveness of such conversions. A dataset of sequence diagrams from 3GPP documents is chosen to be representative of domain-specific actual scenarios. We compare puml outputs from two VLMs - Claude Sonnet and GPT-4V - against manually created ground truth representations. We use version control tools to capture differences and introduce standard performance metrics to measure accuracies along various components: participant identification, message flow accuracy, sequence ordering, and grouping construct preservation. We demonstrate effectiveness of proposed metrics in quantifying conversion errors across various components of puml scripts. The results show that nodes, edges and messages are accurately captured. However, we observe that VLMs do not necessarily perform well on complex structures such as notes, box, groups. Our experiments and performance metrics indicates a need for better representation of these components in training data for fine-tuned VLMs.
>
---
#### [new 093] DualAlign: Generating Clinically Grounded Synthetic Data
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出DualAlign框架，用于生成临床可信的合成数据，解决隐私限制和数据不足问题。通过统计与语义对齐，提升生成数据的真实性和临床相关性，支持低资源医疗文本分析任务。**

- **链接: [http://arxiv.org/pdf/2509.10538v1](http://arxiv.org/pdf/2509.10538v1)**

> **作者:** Rumeng Li; Xun Wang; Hong Yu
>
> **摘要:** Synthetic clinical data are increasingly important for advancing AI in healthcare, given strict privacy constraints on real-world EHRs, limited availability of annotated rare-condition data, and systemic biases in observational datasets. While large language models (LLMs) can generate fluent clinical text, producing synthetic data that is both realistic and clinically meaningful remains challenging. We introduce DualAlign, a framework that enhances statistical fidelity and clinical plausibility through dual alignment: (1) statistical alignment, which conditions generation on patient demographics and risk factors; and (2) semantic alignment, which incorporates real-world symptom trajectories to guide content generation. Using Alzheimer's disease (AD) as a case study, DualAlign produces context-grounded symptom-level sentences that better reflect real-world clinical documentation. Fine-tuning an LLaMA 3.1-8B model with a combination of DualAlign-generated and human-annotated data yields substantial performance gains over models trained on gold data alone or unguided synthetic baselines. While DualAlign does not fully capture longitudinal complexity, it offers a practical approach for generating clinically grounded, privacy-preserving synthetic data to support low-resource clinical text analysis.
>
---
#### [new 094] Rethinking Human Preference Evaluation of LLM Rationales
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM生成解释（rationales）的质量评估问题。研究通过分析人类偏好数据，识别关键属性并提出基于属性的评价方法，以替代传统的二元比较方式，提升评估的细粒度与可解释性。**

- **链接: [http://arxiv.org/pdf/2509.11026v1](http://arxiv.org/pdf/2509.11026v1)**

> **作者:** Ziang Li; Manasi Ganti; Zixian Ma; Helena Vasconcelos; Qijia He; Ranjay Krishna
>
> **备注:** Published in the XLLM-Reason-Plan Workshop on the Application of LLM Explainability to Reasoning and Planning at COLM 2025
>
> **摘要:** Large language models (LLMs) often generate natural language rationales -- free-form explanations that help improve performance on complex reasoning tasks and enhance interpretability for human users. However, evaluating these rationales remains challenging. While recent work has relied on binary preference judgments from humans or LLM judges, such evaluations are often opaque and coarse-grained, offering limited insight into what makes one rationale better than another. In this work, we rethink preference evaluation for LLM-generated rationales by asking: (1) What attributes define good rationales? (2) Can human preferences be explained by these attributes? (3) Can attribute-based evaluation overcome the limitations of binary comparisons? We identify a set of key rationale attributes from prior literature and assess them using automatic metrics, LLM judgments, and human annotations. We then analyze two standard human preference datasets MT Bench and Chatbot Arena using SHAP to identify which attributes best explain human preference outcomes. Finally, we re-evaluate model-generated rationales using attribute-specific ELO scores, revealing more nuanced model comparisons and insights. Our findings suggest that fine-grained attribute evaluations can better characterize rationale quality and guide future research toward more interpretable and reliable evaluation practices.
>
---
#### [new 095] The AI Memory Gap: Users Misremember What They Created With AI or Without
- **分类: cs.HC; cs.CL; H.5.2; I.2.7**

- **简介: 论文研究用户在使用AI生成内容后对创作来源的记忆准确性。任务是探讨人类与AI协作时的源记忆偏差问题。通过实验发现，混合流程中用户更易混淆内容来源，影响AI技术的设计与应用。**

- **链接: [http://arxiv.org/pdf/2509.11851v1](http://arxiv.org/pdf/2509.11851v1)**

> **作者:** Tim Zindulka; Sven Goller; Daniela Fernandes; Robin Welsch; Daniel Buschek
>
> **备注:** 31 pages, 10 figures, 9 tables
>
> **摘要:** As large language models (LLMs) become embedded in interactive text generation, disclosure of AI as a source depends on people remembering which ideas or texts came from themselves and which were created with AI. We investigate how accurately people remember the source of content when using AI. In a pre-registered experiment, 184 participants generated and elaborated on ideas both unaided and with an LLM-based chatbot. One week later, they were asked to identify the source (noAI vs withAI) of these ideas and texts. Our findings reveal a significant gap in memory: After AI use, the odds of correct attribution dropped, with the steepest decline in mixed human-AI workflows, where either the idea or elaboration was created with AI. We validated our results using a computational model of source memory. Discussing broader implications, we highlight the importance of considering source confusion in the design and use of interactive text generation technologies.
>
---
#### [new 096] Learning Decomposed Contextual Token Representations from Pretrained and Collaborative Signals for Generative Recommendation
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于推荐系统任务，旨在解决生成式推荐中预训练与推荐目标不一致的问题。提出DECOR框架，通过分解融合预训练与协作嵌入，提升静态分词与语义保留效果，实验验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.10468v1](http://arxiv.org/pdf/2509.10468v1)**

> **作者:** Yifan Liu; Yaokun Liu; Zelin Li; Zhenrui Yue; Gyuseok Lee; Ruichen Yao; Yang Zhang; Dong Wang
>
> **备注:** preprint under review
>
> **摘要:** Recent advances in generative recommenders adopt a two-stage paradigm: items are first tokenized into semantic IDs using a pretrained tokenizer, and then large language models (LLMs) are trained to generate the next item via sequence-to-sequence modeling. However, these two stages are optimized for different objectives: semantic reconstruction during tokenizer pretraining versus user interaction modeling during recommender training. This objective misalignment leads to two key limitations: (i) suboptimal static tokenization, where fixed token assignments fail to reflect diverse usage contexts; and (ii) discarded pretrained semantics, where pretrained knowledge - typically from language model embeddings - is overwritten during recommender training on user interactions. To address these limitations, we propose to learn DEcomposed COntextual Token Representations (DECOR), a unified framework that preserves pretrained semantics while enhancing the adaptability of token embeddings. DECOR introduces contextualized token composition to refine token embeddings based on user interaction context, and decomposed embedding fusion that integrates pretrained codebook embeddings with newly learned collaborative embeddings. Experiments on three real-world datasets demonstrate that DECOR consistently outperforms state-of-the-art baselines in recommendation performance. Our code will be made available upon publication.
>
---
#### [new 097] Formal Reasoning for Intelligent QA Systems: A Case Study in the Educational Domain
- **分类: cs.AI; cs.CL**

- **简介: 论文提出MCFR框架，结合LLM与模型检测，解决封闭领域问答系统中推理不可靠问题。通过形式化验证提升推理可信度，并引入EduMC-QA数据集进行评估，对比现有LLM表现。属于智能问答系统的可解释性与可靠性研究任务。**

- **链接: [http://arxiv.org/pdf/2509.11572v1](http://arxiv.org/pdf/2509.11572v1)**

> **作者:** Tuan Bui; An Nguyen; Phat Thai; Minh Hua; Ngan Pham L. N.; Ngan Pham T. B.; Dung Le; Long Nguyen; Thanh-Tung Tran; Thang Bui; Tho Quan
>
> **备注:** Published at the 2nd ACM Workshop in AI-powered Question & Answering Systems (AIQAM '25), co-located with ACM Multimedia 2025
>
> **摘要:** Reasoning is essential for closed-domain QA systems in which procedural correctness and policy compliance are critical. While large language models (LLMs) have shown strong performance on many reasoning tasks, recent work reveals that their reasoning traces are often unfaithful - serving more as plausible justifications than as causally grounded derivations. Efforts to combine LLMs with symbolic engines (e.g., Prover9, Z3) have improved reliability but remain limited to static forms of logic, struggling with dynamic, state-based reasoning such as multi-step progressions and conditional transitions. In this paper, we propose MCFR (Model Checking for Formal Reasoning), a neuro-symbolic framework that integrates LLMs with model checking to support property verification. MCFR translates natural language into formal specifications and verifies them over transition models. To support evaluation, we introduce EduMC-QA, a benchmark dataset grounded in real academic procedures. Our results show that MCFR improves reasoning faithfulness and interpretability, offering a viable path toward verifiable QA in high-stakes closed-domain applications. In addition to evaluating MCFR, we compare its performance with state-of-the-art LLMs such as ChatGPT, DeepSeek, and Claude to contextualize its effectiveness.
>
---
#### [new 098] Agentic Username Suggestion and Multimodal Gender Detection in Online Platforms: Introducing the PNGT-26K Dataset
- **分类: cs.LG; cs.AI; cs.CL; cs.SI**

- **简介: 该论文提出PNGT-26K数据集，用于解决波斯语姓名在性别检测和数字身份创建中的挑战。同时引入两个框架：Open Gender Detection用于性别预测，Nominalist用于生成用户名，提升用户体验。**

- **链接: [http://arxiv.org/pdf/2509.11136v1](http://arxiv.org/pdf/2509.11136v1)**

> **作者:** Farbod Bijary; Mohsen Ebadpour; Amirhosein Tajbakhsh
>
> **摘要:** Persian names present unique challenges for natural language processing applications, particularly in gender detection and digital identity creation, due to transliteration inconsistencies and cultural-specific naming patterns. Existing tools exhibit significant performance degradation on Persian names, while the scarcity of comprehensive datasets further compounds these limitations. To address these challenges, the present research introduces PNGT-26K, a comprehensive dataset of Persian names, their commonly associated gender, and their English transliteration, consisting of approximately 26,000 tuples. As a demonstration of how this resource can be utilized, we also introduce two frameworks, namely Open Gender Detection and Nominalist. Open Gender Detection is a production-grade, ready-to-use framework for using existing data from a user, such as profile photo and name, to give a probabilistic guess about the person's gender. Nominalist, the second framework introduced by this paper, utilizes agentic AI to help users choose a username for their social media accounts on any platform. It can be easily integrated into any website to provide a better user experience. The PNGT-26K dataset, Nominalist and Open Gender Detection frameworks are publicly available on Github.
>
---
#### [new 099] Lost in Embeddings: Information Loss in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 论文研究视觉-语言模型中信息丢失问题，分析投影步骤对视觉表示的影响。提出两种方法量化信息损失：通过k近邻关系变化和图像块级嵌入重构。揭示连接器导致几何失真，影响模型性能。属于多模态表示学习任务。**

- **链接: [http://arxiv.org/pdf/2509.11986v1](http://arxiv.org/pdf/2509.11986v1)**

> **作者:** Wenyan Li; Raphael Tang; Chengzu Li; Caiqi Zhang; Ivan Vulić; Anders Søgaard
>
> **摘要:** Vision--language models (VLMs) often process visual inputs through a pretrained vision encoder, followed by a projection into the language model's embedding space via a connector component. While crucial for modality fusion, the potential information loss induced by this projection step and its direct impact on model capabilities remain understudied. We introduce two complementary approaches to examine and quantify this loss by analyzing the latent representation space. First, we evaluate semantic information preservation by analyzing changes in k-nearest neighbor relationships between image representations, before and after projection. Second, we directly measure information loss by reconstructing visual embeddings from the projected representation, localizing loss at an image patch level. Experiments reveal that connectors substantially distort the local geometry of visual representations, with k-nearest neighbors diverging by 40--60\% post-projection, correlating with degradation in retrieval performance. The patch-level embedding reconstruction provides interpretable insights for model behavior on visually grounded question-answering tasks, finding that areas of high information loss reliably predict instances where models struggle.
>
---
#### [new 100] Understanding AI Evaluation Patterns: How Different GPT Models Assess Vision-Language Descriptions
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究不同GPT模型评估视觉-语言描述的行为差异，分析其评估策略与偏见。通过对比实验揭示模型间“评估个性”差异，指出评估能力不随通用能力提升而增强，强调需多架构视角实现稳健AI评估。**

- **链接: [http://arxiv.org/pdf/2509.10707v1](http://arxiv.org/pdf/2509.10707v1)**

> **作者:** Sajjad Abdoli; Rudi Cilibrasi; Rima Al-Shikh
>
> **摘要:** As AI systems increasingly evaluate other AI outputs, understanding their assessment behavior becomes crucial for preventing cascading biases. This study analyzes vision-language descriptions generated by NVIDIA's Describe Anything Model and evaluated by three GPT variants (GPT-4o, GPT-4o-mini, GPT-5) to uncover distinct "evaluation personalities" the underlying assessment strategies and biases each model demonstrates. GPT-4o-mini exhibits systematic consistency with minimal variance, GPT-4o excels at error detection, while GPT-5 shows extreme conservatism with high variability. Controlled experiments using Gemini 2.5 Pro as an independent question generator validate that these personalities are inherent model properties rather than artifacts. Cross-family analysis through semantic similarity of generated questions reveals significant divergence: GPT models cluster together with high similarity while Gemini exhibits markedly different evaluation strategies. All GPT models demonstrate a consistent 2:1 bias favoring negative assessment over positive confirmation, though this pattern appears family-specific rather than universal across AI architectures. These findings suggest that evaluation competence does not scale with general capability and that robust AI assessment requires diverse architectural perspectives.
>
---
#### [new 101] Why Bonds Fail Differently? Explainable Multimodal Learning for Multi-Class Default Prediction
- **分类: q-fin.RM; cs.CL; cs.LG; q-fin.CP**

- **简介: 论文提出EMDLOT框架，用于多类债券违约预测。整合数值时序与文本数据，提升模型可解释性。解决传统模型在捕捉金融数据不规则性和时间依赖性方面的不足，提高预测性能与透明度。**

- **链接: [http://arxiv.org/pdf/2509.10802v1](http://arxiv.org/pdf/2509.10802v1)**

> **作者:** Yi Lu; Aifan Ling; Chaoqun Wang; Yaxin Xu
>
> **摘要:** In recent years, China's bond market has seen a surge in defaults amid regulatory reforms and macroeconomic volatility. Traditional machine learning models struggle to capture financial data's irregularity and temporal dependencies, while most deep learning models lack interpretability-critical for financial decision-making. To tackle these issues, we propose EMDLOT (Explainable Multimodal Deep Learning for Time-series), a novel framework for multi-class bond default prediction. EMDLOT integrates numerical time-series (financial/macroeconomic indicators) and unstructured textual data (bond prospectuses), uses Time-Aware LSTM to handle irregular sequences, and adopts soft clustering and multi-level attention to boost interpretability. Experiments on 1994 Chinese firms (2015-2024) show EMDLOT outperforms traditional (e.g., XGBoost) and deep learning (e.g., LSTM) benchmarks in recall, F1-score, and mAP, especially in identifying default/extended firms. Ablation studies validate each component's value, and attention analyses reveal economically intuitive default drivers. This work provides a practical tool and a trustworthy framework for transparent financial risk modeling.
>
---
#### [new 102] Collaborative Document Editing with Multiple Users and AI Agents
- **分类: cs.HC; cs.CL; H.5.2; I.2.7**

- **简介: 该论文提出将AI代理集成到协作写作环境，解决多人协作中使用AI工具的透明性与协调问题。通过共享对象“代理配置文件”和“任务”，使AI响应以评论形式呈现，并在用户研究中验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.11826v1](http://arxiv.org/pdf/2509.11826v1)**

> **作者:** Florian Lehmann; Krystsina Shauchenka; Daniel Buschek
>
> **备注:** 34 pages, 10 figures, 4 tables
>
> **摘要:** Current AI writing support tools are largely designed for individuals, complicating collaboration when co-writers must leave the shared workspace to use AI and then communicate and reintegrate results. We propose integrating AI agents directly into collaborative writing environments. Our prototype makes AI use transparent and customisable through two new shared objects: agent profiles and tasks. Agent responses appear in the familiar comment feature. In a user study (N=30), 14 teams worked on writing projects during one week. Interaction logs and interviews show that teams incorporated agents into existing norms of authorship, control, and coordination, rather than treating them as team members. Agent profiles were viewed as personal territory, while created agents and outputs became shared resources. We discuss implications for team-based AI interaction, highlighting opportunities and boundaries for treating AI as a shared resource in collaborative work.
>
---
#### [new 103] MindVL: Towards Efficient and Effective Training of Multimodal Large Language Models on Ascend NPUs
- **分类: cs.CV; cs.AI; cs.CL; eess.IV**

- **简介: 该论文提出MindVL，一种在Ascend NPUs上高效训练的多模态大语言模型。通过原生分辨率视觉变换器和分布式训练框架Mindspeed-MLLM，解决多模态模型训练效率与精度问题，实现与Qwen2.5-VL相当的性能，使用更少数据。**

- **链接: [http://arxiv.org/pdf/2509.11662v1](http://arxiv.org/pdf/2509.11662v1)**

> **作者:** Feilong Chen; Yijiang Liu; Yi Huang; Hao Wang; Miren Tian; Ya-Qi Yu; Minghui Liao; Jihao Wu
>
> **摘要:** We propose MindVL, a multimodal large langauge model trained on Ascend NPUs. Similar to Qwen2.5-VL, MindVL adopts native-resolution Vision Transformers, which enables it to process images at their original variable resolutions. This design avoids the degradation caused by fixed-resolution tiling while preserving fine-grained details and global layouts, which is crucial for visually dense content such as complex charts and diagrams. To ensure the smooth training of MindVL on Ascend NPUs, we develop Mindspeed-MLLM, a distributed multimodal training framework tailored for Ascend NPUs. To maintain training accuracy, we implement equivalent replacements for certain operators. MindVL undergoes a three-phase training process, namely the warm-up phase, multitask training phase, and supervised instruction tuning phase, to gradually enhance its capabilities. This process starts with basic visual and multimodal pre-training, followed by large-scale multiask trainging and instruction tuning. We also adopt multimodal data packaging and hybrid parallelism techniques, which significantly improve end-to-end training speed. To further boost model performance, we specifically introduce test-time resolution search and model weight averaging. Notably, despite using about 1/10 of the training data required by Qwen2.5-VL, MindVL achieves performance on par with Qwen2.5-VL in evaluations of general multimodal understanding and document/table comprehension. Beyond overall scores, MindVL also delivers leading performance in OCR assessments.
>
---
#### [new 104] Evalet: Evaluating Large Language Models by Fragmenting Outputs into Functions
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 论文提出Evalet系统，通过将大语言模型输出分解为功能片段进行评估，解决传统整体评分难以定位具体问题的缺陷。该工作属于AI模型评估任务，旨在提升评估的细粒度与可解释性。**

- **链接: [http://arxiv.org/pdf/2509.11206v1](http://arxiv.org/pdf/2509.11206v1)**

> **作者:** Tae Soo Kim; Heechan Lee; Yoonjoo Lee; Joseph Seering; Juho Kim
>
> **摘要:** Practitioners increasingly rely on Large Language Models (LLMs) to evaluate generative AI outputs through "LLM-as-a-Judge" approaches. However, these methods produce holistic scores that obscure which specific elements influenced the assessments. We propose functional fragmentation, a method that dissects each output into key fragments and interprets the rhetoric functions that each fragment serves relative to evaluation criteria -- surfacing the elements of interest and revealing how they fulfill or hinder user goals. We instantiate this approach in Evalet, an interactive system that visualizes fragment-level functions across many outputs to support inspection, rating, and comparison of evaluations. A user study (N=10) found that, while practitioners struggled to validate holistic scores, our approach helped them identify 48% more evaluation misalignments. This helped them calibrate trust in LLM evaluations and rely on them to find more actionable issues in model outputs. Our work shifts LLM evaluation from quantitative scores toward qualitative, fine-grained analysis of model behavior.
>
---
#### [new 105] MillStone: How Open-Minded Are LLMs?
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出MillStone基准，评估LLMs在争议性话题上的立场受外部论点影响的程度。任务是测量LLMs的“开放性”，解决其观点是否易受信息源影响的问题，分析不同模型对反方论点的接受度及说服力差异。**

- **链接: [http://arxiv.org/pdf/2509.11967v1](http://arxiv.org/pdf/2509.11967v1)**

> **作者:** Harold Triedman; Vitaly Shmatikov
>
> **备注:** 19 pages, 7 tables, 7 figures
>
> **摘要:** Large language models equipped with Web search, information retrieval tools, and other agentic capabilities are beginning to supplant traditional search engines. As users start to rely on LLMs for information on many topics, including controversial and debatable issues, it is important to understand how the stances and opinions expressed in LLM outputs are influenced by the documents they use as their information sources. In this paper, we present MillStone, the first benchmark that aims to systematically measure the effect of external arguments on the stances that LLMs take on controversial issues (not all of them political). We apply MillStone to nine leading LLMs and measure how ``open-minded'' they are to arguments supporting opposite sides of these issues, whether different LLMs agree with each other, which arguments LLMs find most persuasive, and whether these arguments are the same for different LLMs. In general, we find that LLMs are open-minded on most issues. An authoritative source of information can easily sway an LLM's stance, highlighting the importance of source selection and the risk that LLM-based information retrieval and search systems can be manipulated.
>
---
#### [new 106] LLM in the Middle: A Systematic Review of Threats and Mitigations to Real-World LLM-based Systems
- **分类: cs.CR; cs.AI; cs.CL; cs.ET; cs.LG**

- **简介: 该论文系统综述了LLM系统面临的安全威胁与防御策略，分析其全生命周期中的风险，并分类映射防御措施。旨在帮助用户和研究者理解并缓解LLM集成中的安全与隐私问题。**

- **链接: [http://arxiv.org/pdf/2509.10682v1](http://arxiv.org/pdf/2509.10682v1)**

> **作者:** Vitor Hugo Galhardo Moia; Igor Jochem Sanz; Gabriel Antonio Fontes Rebello; Rodrigo Duarte de Meneses; Briland Hitaj; Ulf Lindqvist
>
> **备注:** 37 pages, 8 figures, 13 tables
>
> **摘要:** The success and wide adoption of generative AI (GenAI), particularly large language models (LLMs), has attracted the attention of cybercriminals seeking to abuse models, steal sensitive data, or disrupt services. Moreover, providing security to LLM-based systems is a great challenge, as both traditional threats to software applications and threats targeting LLMs and their integration must be mitigated. In this survey, we shed light on security and privacy concerns of such LLM-based systems by performing a systematic review and comprehensive categorization of threats and defensive strategies considering the entire software and LLM life cycles. We analyze real-world scenarios with distinct characteristics of LLM usage, spanning from development to operation. In addition, threats are classified according to their severity level and to which scenarios they pertain, facilitating the identification of the most relevant threats. Recommended defense strategies are systematically categorized and mapped to the corresponding life cycle phase and possible attack strategies they attenuate. This work paves the way for consumers and vendors to understand and efficiently mitigate risks during integration of LLMs in their respective solutions or organizations. It also enables the research community to benefit from the discussion of open challenges and edge cases that may hinder the secure and privacy-preserving adoption of LLM-based systems.
>
---
#### [new 107] Collapse of Irrelevant Representations (CIR) Ensures Robust and Non-Disruptive LLM Unlearning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出CIR方法，通过PCA分析激活和梯度，选择性消除语言模型中的危险知识，实现高效、非破坏性的遗忘。属于LLM安全训练任务，解决现有方法无法有效移除有害知识的问题。**

- **链接: [http://arxiv.org/pdf/2509.11816v1](http://arxiv.org/pdf/2509.11816v1)**

> **作者:** Filip Sondej; Yushi Yang
>
> **摘要:** Current unlearning techniques and safety training consistently fail to remove dangerous knowledge from language models. We analyze the root causes and propose a highly selective technique which unlearns robustly and without disrupting general performance. We perform PCA on activations and module output gradients to identify subspaces containing common representations, and collapse them before calculating unlearning updates. This way we avoid unlearning general representations, and only target those specific to the unlearned facts. When unlearning WMDP dataset facts from Llama-3.1-8B, we drop post-attack accuracy 80x more than our best baseline (Circuit Breakers) on biohazardous facts and 30x more on cyberhazardous facts. Despite this, we disrupt general performance 30x less (only 0.1% WikiText loss increase), while requiring less than 3 GPU-seconds per fact.
>
---
#### [new 108] FuseCodec: Semantic-Contextual Fusion and Supervision for Neural Codecs
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出FuseCodec，解决神经编解码器忽略语义与上下文信息的问题。通过融合声学、语义和上下文表示，提升语音分词质量。实验表明其在语音转录等任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.11425v1](http://arxiv.org/pdf/2509.11425v1)**

> **作者:** Md Mubtasim Ahasan; Rafat Hasan Khan; Tasnim Mohiuddin; Aman Chadha; Tariq Iqbal; M Ashraful Amin; Amin Ahsan Ali; Md Mofijul Islam; A K M Mahbubur Rahman
>
> **摘要:** Speech tokenization enables discrete representation and facilitates speech language modeling. However, existing neural codecs capture low-level acoustic features, overlooking the semantic and contextual cues inherent to human speech. While recent efforts introduced semantic representations from self-supervised speech models or incorporated contextual representations from pre-trained language models, challenges remain in aligning and unifying the semantic and contextual representations. We introduce FuseCodec, which unifies acoustic, semantic, and contextual representations through strong cross-modal alignment and globally informed supervision. We propose three complementary techniques: (i) Latent Representation Fusion, integrating semantic and contextual features directly into the encoder latent space for robust and unified representation learning; (ii) Global Semantic-Contextual Supervision, supervising discrete tokens with globally pooled and broadcasted representations to enhance temporal consistency and cross-modal alignment; and (iii) Temporally Aligned Contextual Supervision, strengthening alignment by dynamically matching contextual and speech tokens within a local window for fine-grained token-level supervision. We further introduce FuseCodec-TTS, demonstrating our methodology's applicability to zero-shot speech synthesis. Empirically, FuseCodec achieves state-of-the-art performance in LibriSpeech, surpassing EnCodec, SpeechTokenizer, and DAC in transcription accuracy, perceptual quality, intelligibility, and speaker similarity. Results highlight the effectiveness of contextually and semantically guided tokenization for speech tokenization and downstream tasks. Code and pretrained models are available at https://github.com/mubtasimahasan/FuseCodec.
>
---
#### [new 109] ReFineG: Synergizing Small Supervised Models and LLMs for Low-Resource Grounded Multimodal NER
- **分类: cs.IR; cs.CL**

- **简介: 该论文研究低资源场景下的多模态命名实体识别任务，旨在解决传统方法依赖大量标注数据和大模型领域知识冲突的问题。提出ReFineG框架，结合小模型与冻结的大语言模型，通过三阶段协作提升性能。**

- **链接: [http://arxiv.org/pdf/2509.10975v1](http://arxiv.org/pdf/2509.10975v1)**

> **作者:** Jielong Tang; Shuang Wang; Zhenxing Wang; Jianxing Yu; Jian Yin
>
> **备注:** CCKS 2025 Shared Task Paper
>
> **摘要:** Grounded Multimodal Named Entity Recognition (GMNER) extends traditional NER by jointly detecting textual mentions and grounding them to visual regions. While existing supervised methods achieve strong performance, they rely on costly multimodal annotations and often underperform in low-resource domains. Multimodal Large Language Models (MLLMs) show strong generalization but suffer from Domain Knowledge Conflict, producing redundant or incorrect mentions for domain-specific entities. To address these challenges, we propose ReFineG, a three-stage collaborative framework that integrates small supervised models with frozen MLLMs for low-resource GMNER. In the Training Stage, a domain-aware NER data synthesis strategy transfers LLM knowledge to small models with supervised training while avoiding domain knowledge conflicts. In the Refinement Stage, an uncertainty-based mechanism retains confident predictions from supervised models and delegates uncertain ones to the MLLM. In the Grounding Stage, a multimodal context selection algorithm enhances visual grounding through analogical reasoning. In the CCKS2025 GMNER Shared Task, ReFineG ranked second with an F1 score of 0.6461 on the online leaderboard, demonstrating its effectiveness with limited annotations.
>
---
#### [new 110] Opal: An Operator Algebra View of RLHF
- **分类: cs.LG; cs.AI; cs.CL; 68T05, 68T07, 68Q32, 62H30, 62F15, 90C30; I.2.6; I.2.7; I.2.8; G.3; G.1.6**

- **简介: 论文提出Opal框架，从算子代数角度分析RLHF，定义目标函数的规范形式，并引入GKPO作为统一表示方法。通过JSON序列化和转换规则，实现多种RLHF方法的标准化与比较。**

- **链接: [http://arxiv.org/pdf/2509.11298v1](http://arxiv.org/pdf/2509.11298v1)**

> **作者:** Madhava Gaikwad
>
> **备注:** 11 pages main
>
> **摘要:** We present Opal, an operator view of reinforcement learning from human feedback (RLHF). Objectives are expressed as ladders of two primitives on a base utility: additive penalties and multiplicative pairwise weights. We describe a simple reduction law with if-and-only-if conditions: such ladders collapse to a normal form on pairwise margins when the reference is fixed, penalties are additive, and weights are independent of intermediate margins. When these assumptions do not hold (reference shift, non-additive gates, score-dependent weights), small examples demonstrate non-reducibility. Building on this view, we introduce GKPO (Generalized Kernel Preference Object), a canonical schema in which many RLHF methods can be represented and, when reducible, mapped back from. GKPO provides a standard JSON serialization, canonicalization and hashing rules, and explicit flags with finite witnesses when assumptions fail. We illustrate these ideas with GKPO examples for DPO, RRHF, and ORPO, along with cross-method conversions (where assumptions permit) and minimal stress tests (SHIFT/GATE/SCORE) that highlight non-reducibility. A lightweight Python reference library accompanies the schema, implementing canonical hashing and adapters for DPO and RRHF.
>
---
#### [new 111] MALLM: Multi-Agent Large Language Models Framework
- **分类: cs.MA; cs.AI; cs.CL; A.1; I.2.7**

- **简介: 该论文提出MALLM框架，用于系统分析多智能体辩论（MAD）组件。它解决当前框架配置有限、缺乏集成评估的问题，提供多种配置选项和评估流程，助力研究者理解MAD各部分的交互。**

- **链接: [http://arxiv.org/pdf/2509.11656v1](http://arxiv.org/pdf/2509.11656v1)**

> **作者:** Jonas Becker; Lars Benedikt Kaesberg; Niklas Bauer; Jan Philip Wahle; Terry Ruas; Bela Gipp
>
> **备注:** Accepted at EMNLP 2025 (Demo)
>
> **摘要:** Multi-agent debate (MAD) has demonstrated the ability to augment collective intelligence by scaling test-time compute and leveraging expertise. Current frameworks for multi-agent debate are often designed towards tool use, lack integrated evaluation, or provide limited configurability of agent personas, response generators, discussion paradigms, and decision protocols. We introduce MALLM (Multi-Agent Large Language Models), an open-source framework that enables systematic analysis of MAD components. MALLM offers more than 144 unique configurations of MAD, including (1) agent personas (e.g., Expert, Personality), (2) response generators (e.g., Critical, Reasoning), (3) discussion paradigms (e.g., Memory, Relay), and (4) decision protocols (e.g., Voting, Consensus). MALLM uses simple configuration files to define a debate. Furthermore, MALLM can load any textual Huggingface dataset (e.g., MMLU-Pro, WinoGrande) and provides an evaluation pipeline for easy comparison of MAD configurations. MALLM is tailored towards researchers and provides a window into the heart of multi-agent debate, facilitating the understanding of its components and their interplay.
>
---
#### [new 112] Learning to Optimize Multi-Objective Alignment Through Dynamic Reward Weighting
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于多目标强化学习任务，解决固定权重难以捕捉非凸帕累托前沿的问题。提出动态奖励加权方法，在线调整权重以优化目标平衡，提升模型在多个数学推理数据集上的性能。**

- **链接: [http://arxiv.org/pdf/2509.11452v1](http://arxiv.org/pdf/2509.11452v1)**

> **作者:** Yining Lu; Zilong Wang; Shiyang Li; Xin Liu; Changlong Yu; Qingyu Yin; Zhan Shi; Zixuan Zhang; Meng Jiang
>
> **摘要:** Prior works in multi-objective reinforcement learning typically use linear reward scalarization with fixed weights, which provably fail to capture non-convex Pareto fronts and thus yield suboptimal results. This limitation becomes especially critical in online preference alignment for large language models. Here, stochastic trajectories generated by parameterized policies create highly non-linear and non-convex mappings from parameters to objectives that no single static weighting scheme can find optimal trade-offs. We address this limitation by introducing dynamic reward weighting, which adaptively adjusts reward weights during the online reinforcement learning process. Unlike existing approaches that rely on fixed-weight interpolation, our dynamic weighting continuously balances and prioritizes objectives in training, facilitating effective exploration of Pareto fronts in objective space. We introduce two approaches of increasing sophistication and generalizability: (1) hypervolume-guided weight adaptation and (2) gradient-based weight optimization, offering a versatile toolkit for online multi-objective alignment. Our extensive experiments demonstrate their compatibility with commonly used online reinforcement learning algorithms (including GRPO, REINFORCE, and RLOO), effectiveness across multiple mathematical reasoning datasets, and applicability to different model families, consistently achieving Pareto dominant solutions with fewer training steps than fixed-weight linear scalarization baselines.
>
---
#### [new 113] Harmful Prompt Laundering: Jailbreaking LLMs with Abductive Styles and Symbolic Encoding
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出HaPLa技术，通过归纳推理和符号编码实现对LLMs的越狱攻击。属于安全攻防任务，旨在揭示LLMs的潜在漏洞，提升防御能力。**

- **链接: [http://arxiv.org/pdf/2509.10931v1](http://arxiv.org/pdf/2509.10931v1)**

> **作者:** Seongho Joo; Hyukhun Koh; Kyomin Jung
>
> **备注:** EMNLP 2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their potential misuse for harmful purposes remains a significant concern. To strengthen defenses against such vulnerabilities, it is essential to investigate universal jailbreak attacks that exploit intrinsic weaknesses in the architecture and learning paradigms of LLMs. In response, we propose \textbf{H}armful \textbf{P}rompt \textbf{La}undering (HaPLa), a novel and broadly applicable jailbreaking technique that requires only black-box access to target models. HaPLa incorporates two primary strategies: 1) \textit{abductive framing}, which instructs LLMs to infer plausible intermediate steps toward harmful activities, rather than directly responding to explicit harmful queries; and 2) \textit{symbolic encoding}, a lightweight and flexible approach designed to obfuscate harmful content, given that current LLMs remain sensitive primarily to explicit harmful keywords. Experimental results show that HaPLa achieves over 95% attack success rate on GPT-series models and 70% across all targets. Further analysis with diverse symbolic encoding rules also reveals a fundamental challenge: it remains difficult to safely tune LLMs without significantly diminishing their helpfulness in responding to benign queries.
>
---
#### [new 114] Smart Trial: Evaluating the Use of Large Language Models for Recruiting Clinical Trial Participants via Social Media
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于临床试验参与者招募任务，旨在利用大语言模型（LLMs）通过社交媒体识别符合条件的参与者。研究构建了TRIALQA数据集，评估七种LLMs在判断用户是否符合入组标准及参与动机上的表现，发现LLMs在复杂推理上仍存在挑战。**

- **链接: [http://arxiv.org/pdf/2509.10584v1](http://arxiv.org/pdf/2509.10584v1)**

> **作者:** Xiaofan Zhou; Zisu Wang; Janice Krieger; Mohan Zalake; Lu Cheng
>
> **摘要:** Clinical trials (CT) are essential for advancing medical research and treatment, yet efficiently recruiting eligible participants -- each of whom must meet complex eligibility criteria -- remains a significant challenge. Traditional recruitment approaches, such as advertisements or electronic health record screening within hospitals, are often time-consuming and geographically constrained. This work addresses the recruitment challenge by leveraging the vast amount of health-related information individuals share on social media platforms. With the emergence of powerful large language models (LLMs) capable of sophisticated text understanding, we pose the central research question: Can LLM-driven tools facilitate CT recruitment by identifying potential participants through their engagement on social media? To investigate this question, we introduce TRIALQA, a novel dataset comprising two social media collections from the subreddits on colon cancer and prostate cancer. Using eligibility criteria from public real-world CTs, experienced annotators are hired to annotate TRIALQA to indicate (1) whether a social media user meets a given eligibility criterion and (2) the user's stated reasons for interest in participating in CT. We benchmark seven widely used LLMs on these two prediction tasks, employing six distinct training and inference strategies. Our extensive experiments reveal that, while LLMs show considerable promise, they still face challenges in performing the complex, multi-hop reasoning needed to accurately assess eligibility criteria.
>
---
#### [new 115] RadarLLM: Adapting Pretrained Large Language Models for Marine Radar Target Detection with Preference-aware Loss
- **分类: eess.SP; cs.CL**

- **简介: 论文提出RadarLLM框架，用于改进预训练大语言模型在海洋雷达目标检测中的应用。针对低信杂比场景下模型过拟合问题，设计偏好感知损失函数，选择性优化特征块，提升泛化能力。实验表明其优于现有方法，尤其在数据有限时表现突出。**

- **链接: [http://arxiv.org/pdf/2509.12089v1](http://arxiv.org/pdf/2509.12089v1)**

> **作者:** Qiying Hu
>
> **摘要:** Recent advances in pre-trained large language models (LLMs) have demonstrated their capacities to capture universal knowledge, making them promising general-purpose optimization solvers for wireless signal processing. Motivated by these findings, we take the first step towards fine-tuning pre-trained LLMs for the effective analysis of radar signal features in marine target detection tasks. Nevertheless, directly fine-tuning pre-trained LLMs on marine target detection tasks tends to suffer from pronounced overfitting, particularly in challenging low signal-to-clutter ratio (SCR) scenarios. This overfitting primarily stems from the model's tendency to memorize spurious or noisy feature patterns rather than learning discriminative structures that generalize well to unseen data. To address this challenge, we introduce RadarLLM, a novel fine-tuning framework that utilizes an effective preference-aware loss. Unlike conventional training strategies that uniformly optimize all feature tokens, this loss function selectively optimizes different feature patches based on their online evaluated learning values, thus guiding the model to focus on the most generalizable patterns during optimization. We theoretically demonstrate the effectiveness of the evaluated learning values by transforming the problem as selecting useful feature tokens. Extensive experiments on real-world marine radar datasets show that 1) the proposed loss function is much better than the original one, with particularly significant gains in challenging low SCR scenarios and 2) RadarLLM consistently outperforms state-of-the-art baselines across diverse detection scenarios, with particularly notable gains under limited training data conditions.
>
---
#### [new 116] Mitigating Hallucinations in Large Vision-Language Models by Self-Injecting Hallucinations
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型领域，旨在解决模型生成内容与视觉输入不一致的幻觉问题。提出APASI方法，通过自注入幻觉生成偏好对，无需外部数据，实现有效幻觉缓解并提升模型性能。**

- **链接: [http://arxiv.org/pdf/2509.11287v1](http://arxiv.org/pdf/2509.11287v1)**

> **作者:** Yifan Lu; Ziqi Zhang; Chunfeng Yuan; Jun Gao; Congxuan Zhang; Xiaojuan Qi; Bing Li; Weiming Hu
>
> **备注:** emnlp 2025 accepted
>
> **摘要:** Large Vision-Language Models (LVLMs) suffer from serious hallucination problems, where the model-generated responses are inconsistent with the visual inputs. Existing hallucination mitigation methods are mainly based on preference alignment and require external human annotations or auxiliary models for preference data collection, which increase costs and limit sustainable improvement. To tackle these challenges, we propose Autonomous Preference Alignment via Self-Injection (APASI), a novel and generalizable method that mitigates hallucinations without external dependencies. APASI leverages the target LVLM to self-inject hallucinations into a generated response, creating a pair of responses with varying preference levels. During the self-injection process, the dis-preferred response is generated based on three key observations of hallucinations, ensuring it simulates real hallucination patterns. This fidelity offers an accurate learning signal for hallucination mitigation. Moreover, APASI incorporates an iterative alignment training strategy combined with curriculum learning to periodically update the preference data with increasing challenge, enabling stable and continuous enhancement of the LVLM. Extensive experiments across six benchmarks show that APASI not only effectively mitigates hallucinations for three baseline models but also achieves comparable or even superior performance to alignment-based methods with external dependency, thereby demonstrating its effectiveness and generalization capability. The code is available at https://github.com/davidluciolu/APASI.
>
---
#### [new 117] DSRAG: A Domain-Specific Retrieval Framework Based on Document-derived Multimodal Knowledge Graph
- **分类: cs.IR; cs.AI; cs.CL; cs.CV; cs.MM**

- **简介: 该论文提出DSRAG框架，用于提升领域特定问答任务的性能。针对通用大模型知识幻觉和领域适应性不足的问题，通过构建多模态知识图谱并结合检索增强生成技术，提高回答的准确性和可靠性。**

- **链接: [http://arxiv.org/pdf/2509.10467v1](http://arxiv.org/pdf/2509.10467v1)**

> **作者:** Mengzheng Yang; Yanfei Ren; David Osei Opoku; Ruochang Li; Peng Ren; Chunxiao Xing
>
> **备注:** 12 pages, 5 figures. Accepted to the 22nd International Conference on Web Information Systems and Applications (WISA 2025)
>
> **摘要:** Current general-purpose large language models (LLMs) commonly exhibit knowledge hallucination and insufficient domain-specific adaptability in domain-specific tasks, limiting their effectiveness in specialized question answering scenarios. Retrieval-augmented generation (RAG) effectively tackles these challenges by integrating external knowledge to enhance accuracy and relevance. However, traditional RAG still faces limitations in domain knowledge accuracy and context modeling.To enhance domain-specific question answering performance, this work focuses on a graph-based RAG framework, emphasizing the critical role of knowledge graph quality during the generation process. We propose DSRAG (Domain-Specific RAG), a multimodal knowledge graph-driven retrieval-augmented generation framework designed for domain-specific applications. Our approach leverages domain-specific documents as the primary knowledge source, integrating heterogeneous information such as text, images, and tables to construct a multimodal knowledge graph covering both conceptual and instance layers. Building on this foundation, we introduce semantic pruning and structured subgraph retrieval mechanisms, combining knowledge graph context and vector retrieval results to guide the language model towards producing more reliable responses. Evaluations using the Langfuse multidimensional scoring mechanism show that our method excels in domain-specific question answering, validating the efficacy of integrating multimodal knowledge graphs with retrieval-augmented generation.
>
---
#### [new 118] When marine radar target detection meets pretrained large language models
- **分类: eess.SP; cs.CL; cs.LG**

- **简介: 论文将预训练大语言模型应用于海洋雷达目标检测，通过特征预处理和微调提升检测性能。属于雷达信号处理任务，解决传统深度学习方法中冗余特征和模型规模限制的问题。**

- **链接: [http://arxiv.org/pdf/2509.12110v1](http://arxiv.org/pdf/2509.12110v1)**

> **作者:** Qiying Hu; Linping Zhang; Xueqian Wang; Gang Li; Yu Liu; Xiao-Ping Zhang
>
> **摘要:** Deep learning (DL) methods are widely used to extract high-dimensional patterns from the sequence features of radar echo signals. However, conventional DL algorithms face challenges such as redundant feature segments, and constraints from restricted model sizes. To address these issues, we propose a framework that integrates feature preprocessing with large language models (LLMs). Our preprocessing module tokenizes radar sequence features, applies a patch selection algorithm to filter out uninformative segments, and projects the selected patches into embeddings compatible with the feature space of pre-trained LLMs. Leveraging these refined embeddings, we incorporate a pre-trained LLM, fine-tuning only the normalization layers to reduce training burdens while enhancing performance. Experiments on measured datasets demonstrate that the proposed method significantly outperforms the state-of-the-art baselines on supervised learning tests.
>
---
#### [new 119] AMQ: Enabling AutoML for Mixed-precision Weight-Only Quantization of Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出AMQ框架，解决大语言模型在内存限制下的最优量化问题。通过剪枝搜索空间、代理量化、质量预测和迭代策略，高效找到性能与内存的平衡点，实现高质量紧凑模型。**

- **链接: [http://arxiv.org/pdf/2509.12019v1](http://arxiv.org/pdf/2509.12019v1)**

> **作者:** Sangjun Lee; Seung-taek Woo; Jungyu Jin; Changhun Lee; Eunhyeok Park
>
> **备注:** EMNLP 2025 Main Conference, Long Paper (Oral)
>
> **摘要:** To enable broader deployment of Large Language Models (LLMs), it is essential to identify the best-performing model under strict memory constraints. We present AMQ, Automated Mixed-Precision Weight-Only Quantization, a framework that assigns layer-wise quantization bit-widths to optimally balance model quality and memory usage. However, the combinatorial search space, with over 10^{100} possible configurations, makes conventional black-box optimization infeasible. AMQ overcomes this challenge through four key innovations:(1) search space pruning using prior knowledge to exclude unpromising configurations, (2) quantization proxy to bypass costly format conversions during search, (3) quality predictor to minimize evaluation overhead, and (4) iterative search-and-update strategy for fast and stable convergence. By integrating these components, AMQ efficiently explores the quality-efficiency landscape, reaching the Pareto frontier and yielding LLMs that are both compact and high-performing. Our code is available at https://github.com/dlwns147/amq.
>
---
#### [new 120] Trading-R1: Financial Trading with LLM Reasoning via Reinforcement Learning
- **分类: q-fin.TR; cs.AI; cs.CE; cs.CL; cs.LG**

- **简介: 该论文提出Trading-R1模型，旨在解决金融交易中LLM推理与可执行交易结合的问题。通过监督微调和强化学习，模型实现结构化、可解释的交易决策，提升风险调整后收益。属于金融AI任务。**

- **链接: [http://arxiv.org/pdf/2509.11420v1](http://arxiv.org/pdf/2509.11420v1)**

> **作者:** Yijia Xiao; Edward Sun; Tong Chen; Fang Wu; Di Luo; Wei Wang
>
> **备注:** Tauric Research: https://github.com/TauricResearch
>
> **摘要:** Developing professional, structured reasoning on par with human financial analysts and traders remains a central challenge in AI for finance, where markets demand interpretability and trust. Traditional time-series models lack explainability, while LLMs face challenges in turning natural-language analysis into disciplined, executable trades. Although reasoning LLMs have advanced in step-by-step planning and verification, their application to risk-sensitive financial decisions is underexplored. We present Trading-R1, a financially-aware model that incorporates strategic thinking and planning for comprehensive thesis composition, facts-grounded analysis, and volatility-adjusted decision making. Trading-R1 aligns reasoning with trading principles through supervised fine-tuning and reinforcement learning with a three-stage easy-to-hard curriculum. Training uses Tauric-TR1-DB, a 100k-sample corpus spanning 18 months, 14 equities, and five heterogeneous financial data sources. Evaluated on six major equities and ETFs, Trading-R1 demonstrates improved risk-adjusted returns and lower drawdowns compared to both open-source and proprietary instruction-following models as well as reasoning models. The system generates structured, evidence-based investment theses that support disciplined and interpretable trading decisions. Trading-R1 Terminal will be released at https://github.com/TauricResearch/Trading-R1.
>
---
#### [new 121] AQUA: Attention via QUery mAgnitudes for Memory and Compute Efficient Inference in LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出AQUA方法，解决LLMs中注意力机制的二次复杂度问题。通过离线SVD投影和在线动态选择维度，降低计算与内存消耗，在性能损失可忽略的前提下提升推理效率。属于高效推理优化任务。**

- **链接: [http://arxiv.org/pdf/2509.11155v1](http://arxiv.org/pdf/2509.11155v1)**

> **作者:** Santhosh G S; Saurav Prakash; Balaraman Ravindran
>
> **摘要:** The quadratic complexity of the attention mechanism remains a fundamental barrier to scaling Large Language Models (LLMs) to longer contexts, creating a critical bottleneck in both computation and memory. To address this, we introduce AQUA (Attention via QUery mAgnitudes) a novel and versatile approximation strategy that significantly reduces the cost of attention with a graceful performance trade-off. Our method operates in two phases: an efficient offline step where we compute a universal, language agnostic projection matrix via SVD on a calibration dataset, and an online inference step where we project query and key vectors and dynamically select a sparse subset of dimensions based on the query's magnitude. We provide a formal theoretical analysis of AQUA, establishing the break-even point at which it becomes more computationally efficient than standard attention. Our empirical evaluations on state-of-the-art models like Llama-3.1-8B demonstrate that a 25% reduction in the attention dot-product computation can be achieved with a statistically insignificant impact on performance across a wide range of benchmarks. We further showcase the versatility of AQUA by demonstrating its ability to synergistically accelerate existing token eviction methods like H2O and to directly reduce KV-cache memory size. By offering a controllable knob to balance efficiency and accuracy, AQUA provides a practical and powerful tool for making large-scale LLM inference more accessible and sustainable.
>
---
#### [new 122] Real-Time RAG for the Identification of Supply Chain Vulnerabilities
- **分类: cs.IR; cs.AI; cs.CL; I.2.7; H.3.3; I.2.6**

- **简介: 论文提出实时RAG方法，用于识别供应链脆弱性。通过整合RAG与网络爬虫技术，提升LLM对新兴信息的分析能力。研究发现优化检索模型和动态调整检索深度显著提升性能，解决传统LLM知识滞后问题。**

- **链接: [http://arxiv.org/pdf/2509.10469v1](http://arxiv.org/pdf/2509.10469v1)**

> **作者:** Jesse Ponnock; Grace Kenneally; Michael Robert Briggs; Elinor Yeo; Tyrone Patterson III; Nicholas Kinberg; Matthew Kalinowski; David Hechtman
>
> **备注:** 14 pages, 5 figures, 1 table. Approved for Public Release; Distribution Unlimited. PRS Release Number: 25-0864
>
> **摘要:** New technologies in generative AI can enable deeper analysis into our nation's supply chains but truly informative insights require the continual updating and aggregation of massive data in a timely manner. Large Language Models (LLMs) offer unprecedented analytical opportunities however, their knowledge base is constrained to the models' last training date, rendering these capabilities unusable for organizations whose mission impacts rely on emerging and timely information. This research proposes an innovative approach to supply chain analysis by integrating emerging Retrieval-Augmented Generation (RAG) preprocessing and retrieval techniques with advanced web-scraping technologies. Our method aims to reduce latency in incorporating new information into an augmented-LLM, enabling timely analysis of supply chain disruptors. Through experimentation, this study evaluates the combinatorial effects of these techniques towards timeliness and quality trade-offs. Our results suggest that in applying RAG systems to supply chain analysis, fine-tuning the embedding retrieval model consistently provides the most significant performance gains, underscoring the critical importance of retrieval quality. Adaptive iterative retrieval, which dynamically adjusts retrieval depth based on context, further enhances performance, especially on complex supply chain queries. Conversely, fine-tuning the LLM yields limited improvements and higher resource costs, while techniques such as downward query abstraction significantly outperforms upward abstraction in practice.
>
---
#### [new 123] DreamNav: A Trajectory-Based Imaginative Framework for Zero-Shot Vision-and-Language Navigation
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出DreamNav框架，解决零样本视觉-语言导航任务中的高成本感知、动作语义不匹配和短视规划问题。通过轨迹预测与主动想象机制，实现基于第一视角输入的高效、全局规划，提升导航性能。**

- **链接: [http://arxiv.org/pdf/2509.11197v1](http://arxiv.org/pdf/2509.11197v1)**

> **作者:** Yunheng Wang; Yuetong Fang; Taowen Wang; Yixiao Feng; Yawen Tan; Shuning Zhang; Peiran Liu; Yiding Ji; Renjing Xu
>
> **摘要:** Vision-and-Language Navigation in Continuous Environments (VLN-CE), which links language instructions to perception and control in the real world, is a core capability of embodied robots. Recently, large-scale pretrained foundation models have been leveraged as shared priors for perception, reasoning, and action, enabling zero-shot VLN without task-specific training. However, existing zero-shot VLN methods depend on costly perception and passive scene understanding, collapsing control to point-level choices. As a result, they are expensive to deploy, misaligned in action semantics, and short-sighted in planning. To address these issues, we present DreamNav that focuses on the following three aspects: (1) for reducing sensory cost, our EgoView Corrector aligns viewpoints and stabilizes egocentric perception; (2) instead of point-level actions, our Trajectory Predictor favors global trajectory-level planning to better align with instruction semantics; and (3) to enable anticipatory and long-horizon planning, we propose an Imagination Predictor to endow the agent with proactive thinking capability. On VLN-CE and real-world tests, DreamNav sets a new zero-shot state-of-the-art (SOTA), outperforming the strongest egocentric baseline with extra information by up to 7.49\% and 18.15\% in terms of SR and SPL metrics. To our knowledge, this is the first zero-shot VLN method to unify trajectory-level planning and active imagination while using only egocentric inputs.
>
---
#### [new 124] How to Evaluate Medical AI
- **分类: cs.AI; cs.CL; I.2.7; I.2.1**

- **简介: 论文提出RPAD和RRAD新指标，用于更准确评估医疗AI诊断性能，解决传统指标忽略专家判断差异的问题。通过对比多专家意见，提升评估稳定性，并展示AI模型可达到与专家相当的诊断一致性。**

- **链接: [http://arxiv.org/pdf/2509.11941v1](http://arxiv.org/pdf/2509.11941v1)**

> **作者:** Ilia Kopanichuk; Petr Anokhin; Vladimir Shaposhnikov; Vladimir Makharev; Ekaterina Tsapieva; Iaroslav Bespalov; Dmitry V. Dylov; Ivan Oseledets
>
> **备注:** 10 pages, 7 fugures
>
> **摘要:** The integration of artificial intelligence (AI) into medical diagnostic workflows requires robust and consistent evaluation methods to ensure reliability, clinical relevance, and the inherent variability in expert judgments. Traditional metrics like precision and recall often fail to account for the inherent variability in expert judgments, leading to inconsistent assessments of AI performance. Inter-rater agreement statistics like Cohen's Kappa are more reliable but they lack interpretability. We introduce Relative Precision and Recall of Algorithmic Diagnostics (RPAD and RRAD) - a new evaluation metrics that compare AI outputs against multiple expert opinions rather than a single reference. By normalizing performance against inter-expert disagreement, these metrics provide a more stable and realistic measure of the quality of predicted diagnosis. In addition to the comprehensive analysis of diagnostic quality measures, our study contains a very important side result. Our evaluation methodology allows us to avoid selecting diagnoses from a limited list when evaluating a given case. Instead, both the models being tested and the examiners verifying them arrive at a free-form diagnosis. In this automated methodology for establishing the identity of free-form clinical diagnoses, a remarkable 98% accuracy becomes attainable. We evaluate our approach using 360 medical dialogues, comparing multiple large language models (LLMs) against a panel of physicians. Large-scale study shows that top-performing models, such as DeepSeek-V3, achieve consistency on par with or exceeding expert consensus. Moreover, we demonstrate that expert judgments exhibit significant variability - often greater than that between AI and humans. This finding underscores the limitations of any absolute metrics and supports the need to adopt relative metrics in medical AI.
>
---
#### [new 125] FinGEAR: Financial Mapping-Guided Enhanced Answer Retrieval
- **分类: cs.CE; cs.CL**

- **简介: 该论文提出FinGEAR，解决金融文件（如10-K）检索难题。通过结合金融术语、双层索引和两阶段重排序，提升检索精度与相关性，显著优于现有方法，为财务分析提供有效支持。**

- **链接: [http://arxiv.org/pdf/2509.12042v1](http://arxiv.org/pdf/2509.12042v1)**

> **作者:** Ying Li; Mengyu Wang; Miguel de Carvalho; Sotirios Sabanis; Tiejun Ma
>
> **摘要:** Financial disclosures such as 10-K filings present challenging retrieval problems due to their length, regulatory section hierarchy, and domain-specific language, which standard retrieval-augmented generation (RAG) models underuse. We introduce FinGEAR (Financial Mapping-Guided Enhanced Answer Retrieval), a retrieval framework tailored to financial documents. FinGEAR combines a finance lexicon for Item-level guidance (FLAM), dual hierarchical indices for within-Item search (Summary Tree and Question Tree), and a two-stage cross-encoder reranker. This design aligns retrieval with disclosure structure and terminology, enabling fine-grained, query-aware context selection. Evaluated on full 10-Ks with queries aligned to the FinQA dataset, FinGEAR delivers consistent gains in precision, recall, F1, and relevancy, improving F1 by up to 56.7% over flat RAG, 12.5% over graph-based RAGs, and 217.6% over prior tree-based systems, while also increasing downstream answer accuracy with a fixed reader. By jointly modeling section hierarchy and domain lexicon signals, FinGEAR improves retrieval fidelity and provides a practical foundation for high-stakes financial analysis.
>
---
#### [new 126] Event2Vec: A Geometric Approach to Learning Composable Representations of Event Sequences
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Event2Vec框架，用于学习事件序列的可组合表示。通过欧几里得和双曲空间建模，验证线性叠加假设，解决事件序列的几何表征问题，提升对层次结构数据的建模能力。**

- **链接: [http://arxiv.org/pdf/2509.12188v1](http://arxiv.org/pdf/2509.12188v1)**

> **作者:** Antonin Sulc
>
> **备注:** 10 pages, 3 figures, Symmetry and Geometry in Neural Representations Workshop at NeuralIPS (Neurreps) 2025
>
> **摘要:** The study of neural representations, both in biological and artificial systems, is increasingly revealing the importance of geometric and topological structures. Inspired by this, we introduce Event2Vec, a novel framework for learning representations of discrete event sequences. Our model leverages a simple, additive recurrent structure to learn composable, interpretable embeddings. We provide a theoretical analysis demonstrating that, under specific training objectives, our model's learned representations in a Euclidean space converge to an ideal additive structure. This ensures that the representation of a sequence is the vector sum of its constituent events, a property we term the linear additive hypothesis. To address the limitations of Euclidean geometry for hierarchical data, we also introduce a variant of our model in hyperbolic space, which is naturally suited to embedding tree-like structures with low distortion. We present experiments to validate our hypothesis and demonstrate the benefits of each geometry, highlighting the improved performance of the hyperbolic model on hierarchical event sequences.
>
---
#### [new 127] Securing AI Agents: Implementing Role-Based Access Control for Industrial Applications
- **分类: cs.AI; cs.CL**

- **简介: 论文提出将基于角色的访问控制（RBAC）集成到AI代理中，以解决其在工业应用中的安全威胁问题。该研究旨在提升AI代理的安全性与可扩展性，特别是在本地部署场景下。**

- **链接: [http://arxiv.org/pdf/2509.11431v1](http://arxiv.org/pdf/2509.11431v1)**

> **作者:** Aadil Gani Ganie
>
> **摘要:** The emergence of Large Language Models (LLMs) has significantly advanced solutions across various domains, from political science to software development. However, these models are constrained by their training data, which is static and limited to information available up to a specific date. Additionally, their generalized nature often necessitates fine-tuning -- whether for classification or instructional purposes -- to effectively perform specific downstream tasks. AI agents, leveraging LLMs as their core, mitigate some of these limitations by accessing external tools and real-time data, enabling applications such as live weather reporting and data analysis. In industrial settings, AI agents are transforming operations by enhancing decision-making, predictive maintenance, and process optimization. For example, in manufacturing, AI agents enable near-autonomous systems that boost productivity and support real-time decision-making. Despite these advancements, AI agents remain vulnerable to security threats, including prompt injection attacks, which pose significant risks to their integrity and reliability. To address these challenges, this paper proposes a framework for integrating Role-Based Access Control (RBAC) into AI agents, providing a robust security guardrail. This framework aims to support the effective and scalable deployment of AI agents, with a focus on on-premises implementations.
>
---
#### [new 128] Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出PoPE位置编码，解决RoPE中内容与位置信息纠缠的问题。通过分离“什么”和“哪里”，提升Transformer在序列建模任务中的性能，尤其在长序列和零样本外推场景表现更优。**

- **链接: [http://arxiv.org/pdf/2509.10534v1](http://arxiv.org/pdf/2509.10534v1)**

> **作者:** Anand Gopalakrishnan; Robert Csordás; Jürgen Schmidhuber; Michael C. Mozer
>
> **摘要:** The attention mechanism in a Transformer architecture matches key to query based on both content -- the what -- and position in a sequence -- the where. We present an analysis indicating that what and where are entangled in the popular RoPE rotary position embedding. This entanglement can impair performance particularly when decisions require independent matches on these two factors. We propose an improvement to RoPE, which we call Polar Coordinate Position Embeddings or PoPE, that eliminates the what-where confound. PoPE is far superior on a diagnostic task requiring indexing solely by position or by content. On autoregressive sequence modeling in music, genomic, and natural language domains, Transformers using PoPE as the positional encoding scheme outperform baselines using RoPE with respect to evaluation loss (perplexity) and downstream task performance. On language modeling, these gains persist across model scale, from 124M to 774M parameters. Crucially, PoPE shows strong zero-shot length extrapolation capabilities, whereas RoPE's performance degrades significantly on longer sequences at test time without fine tuning or the use of position-interpolation methods.
>
---
#### [new 129] Look Again, Think Slowly: Enhancing Visual Reflection in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型（VLM）的视觉推理任务，旨在提升模型在生成长文本时对视觉信息的持续关注能力。研究提出Reflection-V模型，通过构建视觉中心推理数据和设计视觉注意力奖励机制，有效增强视觉反思能力，显著提升视觉推理性能。**

- **链接: [http://arxiv.org/pdf/2509.12132v1](http://arxiv.org/pdf/2509.12132v1)**

> **作者:** Pu Jian; Junhong Wu; Wei Sun; Chen Wang; Shuo Ren; Jiajun Zhang
>
> **备注:** EMNLP2025 Main
>
> **摘要:** Recent advances in text-only "slow-thinking" reasoning have prompted efforts to transfer this capability to vision-language models (VLMs), for training visual reasoning models (\textbf{VRMs}). owever, such transfer faces critical challenges: Effective "slow thinking" in VRMs requires \textbf{visual reflection}, the ability to check the reasoning process based on visual information. Through quantitative analysis, we observe that current VRMs exhibit limited visual reflection, as their attention to visual information diminishes rapidly with longer generated responses. To address this challenge, we propose a new VRM \textbf{Reflection-V}, which enhances visual reflection based on reasoning data construction for cold-start and reward design for reinforcement learning (RL). Firstly, we construct vision-centered reasoning data by leveraging an agent that interacts between VLMs and reasoning LLMs, enabling cold-start learning of visual reflection patterns. Secondly, a visual attention based reward model is employed during RL to encourage reasoning based on visual information. Therefore, \textbf{Reflection-V} demonstrates significant improvements across multiple visual reasoning benchmarks. Furthermore, \textbf{Reflection-V} maintains a stronger and more consistent reliance on visual information during visual reasoning, indicating effective enhancement in visual reflection capabilities.
>
---
## 更新

#### [replaced 001] IOLBENCH: Benchmarking LLMs on Linguistic Reasoning
- **分类: cs.CL; I.2**

- **链接: [http://arxiv.org/pdf/2501.04249v2](http://arxiv.org/pdf/2501.04249v2)**

> **作者:** Satyam Goyal; Soham Dan
>
> **摘要:** Despite the remarkable advancements and widespread applications of deep neural networks, their ability to perform reasoning tasks remains limited, particularly in domains requiring structured, abstract thought. In this paper, we investigate the linguistic reasoning capabilities of state-of-the-art large language models (LLMs) by introducing IOLBENCH, a novel benchmark derived from International Linguistics Olympiad (IOL) problems. This dataset encompasses diverse problems testing syntax, morphology, phonology, and semantics, all carefully designed to be self-contained and independent of external knowledge. These tasks challenge models to engage in metacognitive linguistic reasoning, requiring the deduction of linguistic rules and patterns from minimal examples. Through extensive benchmarking of leading LLMs, we find that even the most advanced models struggle to handle the intricacies of linguistic complexity, particularly in areas demanding compositional generalization and rule abstraction. Our analysis highlights both the strengths and persistent limitations of current models in linguistic problem-solving, offering valuable insights into their reasoning capabilities. By introducing IOLBENCH, we aim to foster further research into developing models capable of human-like reasoning, with broader implications for the fields of computational linguistics and artificial intelligence.
>
---
#### [replaced 002] LML: A Novel Lexicon for the Moral Foundation of Liberty
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.11862v2](http://arxiv.org/pdf/2407.11862v2)**

> **作者:** Oscar Araque; Lorenzo Gatti; Sergio Consoli; Kyriaki Kalimeri
>
> **备注:** Published in the 11th International Conference on Machine Learning, Optimization, and Data Science
>
> **摘要:** The moral value of liberty is a central concept in our inference system when it comes to taking a stance towards controversial social issues such as vaccine hesitancy, climate change, or the right to abortion. Here, we propose a novel Liberty lexicon evaluated on more than 3,000 manually annotated data both in in- and out-of-domain scenarios. As a result of this evaluation, we produce a combined lexicon that constitutes the main outcome of this work. This final lexicon incorporates information from an ensemble of lexicons that have been generated using word embedding similarity (WE) and compositional semantics (CS). Our key contributions include enriching the liberty annotations, developing a robust liberty lexicon for broader application, and revealing the complexity of expressions related to liberty across different platforms. Through the evaluation, we show that the difficulty of the task calls for designing approaches that combine knowledge, in an effort of improving the representations of learning systems.
>
---
#### [replaced 003] Humanizing Machines: Rethinking LLM Anthropomorphism Through a Multi-Level Framework of Design
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.17573v2](http://arxiv.org/pdf/2508.17573v2)**

> **作者:** Yunze Xiao; Lynnette Hui Xian Ng; Jiarui Liu; Mona T. Diab
>
> **备注:** Accepted in EMNLP main proceedings; Updated citations
>
> **摘要:** Large Language Models (LLMs) increasingly exhibit \textbf{anthropomorphism} characteristics -- human-like qualities portrayed across their outlook, language, behavior, and reasoning functions. Such characteristics enable more intuitive and engaging human-AI interactions. However, current research on anthropomorphism remains predominantly risk-focused, emphasizing over-trust and user deception while offering limited design guidance. We argue that anthropomorphism should instead be treated as a \emph{concept of design} that can be intentionally tuned to support user goals. Drawing from multiple disciplines, we propose that the anthropomorphism of an LLM-based artifact should reflect the interaction between artifact designers and interpreters. This interaction is facilitated by cues embedded in the artifact by the designers and the (cognitive) responses of the interpreters to the cues. Cues are categorized into four dimensions: \textit{perceptive, linguistic, behavioral}, and \textit{cognitive}. By analyzing the manifestation and effectiveness of each cue, we provide a unified taxonomy with actionable levers for practitioners. Consequently, we advocate for function-oriented evaluations of anthropomorphic design.
>
---
#### [replaced 004] UR$^2$: Unify RAG and Reasoning through Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.06165v2](http://arxiv.org/pdf/2508.06165v2)**

> **作者:** Weitao Li; Boran Xiang; Xiaolong Wang; Zhinan Gou; Weizhi Ma; Yang Liu
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities through two complementary paradigms: Retrieval-Augmented Generation (RAG), which enhances knowledge grounding, and Reinforcement Learning from Verifiable Rewards (RLVR), which optimizes complex reasoning abilities. However, these two capabilities are often developed in isolation, and existing efforts to unify them remain narrow in scope -- typically limited to open-domain QA with fixed retrieval settings and task-specific constraints. This lack of integration constrains generalization and limits the applicability of RAG-RL methods to broader domains. To bridge this gap, we propose UR2 (Unified RAG and Reasoning), a general framework that unifies retrieval and reasoning through reinforcement learning. UR2 introduces two key contributions: a difficulty-aware curriculum training that selectively invokes retrieval only for challenging problems, and a hybrid knowledge access strategy combining domain-specific offline corpora with LLM-generated summaries. These components are designed to enable dynamic coordination between retrieval and reasoning, improving adaptability across a diverse range of tasks. Experiments across open-domain QA, MMLU-Pro, medical, and mathematical reasoning tasks demonstrate that UR$^2$ (built on Qwen-2.5-3/7B and LLaMA-3.1-8B) significantly outperforms existing RAG and RL methods, achieving comparable performance to GPT-4o-mini and GPT-4.1-mini on several benchmarks. We have released all code, models, and data at https://github.com/Tsinghua-dhy/UR2.
>
---
#### [replaced 005] Persona-Based Synthetic Data Generation Using Multi-Stage Conditioning with Large Language Models for Emotion Recognition
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.13380v2](http://arxiv.org/pdf/2507.13380v2)**

> **作者:** Keito Inoshita; Rushia Harada
>
> **摘要:** In the field of emotion recognition, the development of high-performance models remains a challenge due to the scarcity of high-quality, diverse emotional datasets. Emotional expressions are inherently subjective, shaped by individual personality traits, socio-cultural backgrounds, and contextual factors, making large-scale, generalizable data collection both ethically and practically difficult. To address this issue, we introduce PersonaGen, a novel framework for generating emotionally rich text using a Large Language Model (LLM) through multi-stage persona-based conditioning. PersonaGen constructs layered virtual personas by combining demographic attributes, socio-cultural backgrounds, and detailed situational contexts, which are then used to guide emotion expression generation. We conduct comprehensive evaluations of the generated synthetic data, assessing semantic diversity through clustering and distributional metrics, human-likeness via LLM-based quality scoring, realism through comparison with real-world emotion corpora, and practical utility in downstream emotion classification tasks. Experimental results show that PersonaGen significantly outperforms baseline methods in generating diverse, coherent, and discriminative emotion expressions, demonstrating its potential as a robust alternative for augmenting or replacing real-world emotional datasets.
>
---
#### [replaced 006] MTalk-Bench: Evaluating Speech-to-Speech Models in Multi-Turn Dialogues via Arena-style and Rubrics Protocols
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.18240v2](http://arxiv.org/pdf/2508.18240v2)**

> **作者:** Yuhao Du; Qianwei Huang; Guo Zhu; Zhanchen Dai; Shunian Chen; Qiming Zhu; Le Pan; Minghao Chen; Yuhao Zhang; Li Zhou; Benyou Wang; Haizhou Li
>
> **摘要:** The rapid advancement of speech-to-speech (S2S) large language models (LLMs) has significantly improved real-time spoken interaction. However, current evaluation frameworks remain inadequate for assessing performance in complex, multi-turn dialogues. To address this, we introduce MTalk-Bench, a multi-turn S2S benchmark covering three core dimensions: Semantic Information, Paralinguistic Information, and Ambient Sound. Each dimension includes nine realistic scenarios, along with targeted tasks to assess specific capabilities such as reasoning. Our dual-method evaluation framework combines Arena-style evaluation (pairwise comparison) and Rubrics-based evaluation (absolute scoring) for relative and absolute assessment. The benchmark includes both model and human outputs, evaluated by human evaluators and LLMs. Experimental results reveal two sets of findings. Overall performance of S2S LLMs: (1) models excel at semantic information processing yet underperform on paralinguistic information and ambient sounds perception; (2) models typically regain coherence by increasing response length, sacrificing efficiency in multi-turn dialogues; (3) modality-aware, task-specific designs outperform brute scaling. Evaluation framework and reliability: (1) Arena and Rubrics yield consistent, complementary rankings, but reliable distinctions emerge only when performance gaps are large; (2) LLM-as-a-judge aligns with humans when gaps are clear or criteria explicit, but exhibits position and length biases and is reliable on nonverbal evaluation only with text annotations. These results highlight current limitations in S2S evaluation and the need for more robust, speech-aware assessment frameworks.
>
---
#### [replaced 007] Transformer-Based Multimodal Knowledge Graph Completion with Link-Aware Contexts
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.15688v2](http://arxiv.org/pdf/2501.15688v2)**

> **作者:** Haodi Ma; Dzmitry Kasinets; Daisy Zhe Wang
>
> **摘要:** Multimodal knowledge graph completion (MMKGC) aims to predict missing links in multimodal knowledge graphs (MMKGs) by leveraging information from various modalities alongside structural data. Existing MMKGC approaches primarily extend traditional knowledge graph embedding (KGE) models, which often require creating an embedding for every entity. This results in large model sizes and inefficiencies in integrating multimodal information, particularly for real-world graphs. Meanwhile, Transformer-based models have demonstrated competitive performance in knowledge graph completion (KGC). However, their focus on single-modal knowledge limits their capacity to utilize cross-modal information. Recently, Large vision-language models (VLMs) have shown potential in cross-modal tasks but are constrained by the high cost of training. In this work, we propose a novel approach that integrates Transformer-based KGE models with cross-modal context generated by pre-trained VLMs, thereby extending their applicability to MMKGC. Specifically, we employ a pre-trained VLM to transform relevant visual information from entities and their neighbors into textual sequences. We then frame KGC as a sequence-to-sequence task, fine-tuning the model with the generated cross-modal context. This simple yet effective method significantly reduces model size compared to traditional KGE approaches while achieving competitive performance across multiple large-scale datasets with minimal hyperparameter tuning.
>
---
#### [replaced 008] Plugging Schema Graph into Multi-Table QA: A Human-Guided Framework for Reducing LLM Reliance
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04427v2](http://arxiv.org/pdf/2506.04427v2)**

> **作者:** Xixi Wang; Miguel Costa; Jordanka Kovaceva; Shuai Wang; Francisco C. Pereira
>
> **备注:** Accepted to EMNLP 2025 findings
>
> **摘要:** Large language models (LLMs) have shown promise in table Question Answering (Table QA). However, extending these capabilities to multi-table QA remains challenging due to unreliable schema linking across complex tables. Existing methods based on semantic similarity work well only on simplified hand-crafted datasets and struggle to handle complex, real-world scenarios with numerous and diverse columns. To address this, we propose a graph-based framework that leverages human-curated relational knowledge to explicitly encode schema links and join paths. Given a natural language query, our method searches on graph to construct interpretable reasoning chains, aided by pruning and sub-path merging strategies to enhance efficiency and coherence. Experiments on both standard benchmarks and a realistic, large-scale dataset demonstrate the effectiveness of our approach. To our knowledge, this is the first multi-table QA system applied to truly complex industrial tabular data.
>
---
#### [replaced 009] Enhancing Prompt Injection Attacks to LLMs via Poisoning Alignment
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.14827v3](http://arxiv.org/pdf/2410.14827v3)**

> **作者:** Zedian Shao; Hongbin Liu; Jaden Mu; Neil Zhenqiang Gong
>
> **摘要:** Prompt injection attack, where an attacker injects a prompt into the original one, aiming to make an Large Language Model (LLM) follow the injected prompt to perform an attacker-chosen task, represent a critical security threat. Existing attacks primarily focus on crafting these injections at inference time, treating the LLM itself as a static target. Our experiments show that these attacks achieve some success, but there is still significant room for improvement. In this work, we introduces a more foundational attack vector: poisoning the LLM's alignment process to amplify the success of future prompt injection attacks. Specifically, we propose PoisonedAlign, a method that strategically creates poisoned alignment samples to poison an LLM's alignment dataset. Our experiments across five LLMs and two alignment datasets show that when even a small fraction of the alignment data is poisoned, the resulting model becomes substantially more vulnerable to a wide range of prompt injection attacks. Crucially, this vulnerability is instilled while the LLM's performance on standard capability benchmarks remains largely unchanged, making the manipulation difficult to detect through automated, general-purpose performance evaluations. The code for implementing the attack is available at https://github.com/Sadcardation/PoisonedAlign.
>
---
#### [replaced 010] Multilingual Collaborative Defense for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11835v2](http://arxiv.org/pdf/2505.11835v2)**

> **作者:** Hongliang Li; Jinan Xu; Gengping Cui; Changhao Guan; Fengran Mo; Kaiyu Huang
>
> **备注:** 21 pages, 4figures
>
> **摘要:** The robustness and security of large language models (LLMs) has become a prominent research area. One notable vulnerability is the ability to bypass LLM safeguards by translating harmful queries into rare or underrepresented languages, a simple yet effective method of "jailbreaking" these models. Despite the growing concern, there has been limited research addressing the safeguarding of LLMs in multilingual scenarios, highlighting an urgent need to enhance multilingual safety. In this work, we investigate the correlation between various attack features across different languages and propose Multilingual Collaborative Defense (MCD), a novel learning method that optimizes a continuous, soft safety prompt automatically to facilitate multilingual safeguarding of LLMs. The MCD approach offers three advantages: First, it effectively improves safeguarding performance across multiple languages. Second, MCD maintains strong generalization capabilities while minimizing false refusal rates. Third, MCD mitigates the language safety misalignment caused by imbalances in LLM training corpora. To evaluate the effectiveness of MCD, we manually construct multilingual versions of commonly used jailbreak benchmarks, such as MaliciousInstruct and AdvBench, to assess various safeguarding methods. Additionally, we introduce these datasets in underrepresented (zero-shot) languages to verify the language transferability of MCD. The results demonstrate that MCD outperforms existing approaches in safeguarding against multilingual jailbreak attempts while also exhibiting strong language transfer capabilities. Our code is available at https://github.com/HLiang-Lee/MCD.
>
---
#### [replaced 011] HiMATE: A Hierarchical Multi-Agent Framework for Machine Translation Evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16281v2](http://arxiv.org/pdf/2505.16281v2)**

> **作者:** Shijie Zhang; Renhao Li; Songsheng Wang; Philipp Koehn; Min Yang; Derek F. Wong
>
> **摘要:** The advancement of Large Language Models (LLMs) enables flexible and interpretable automatic evaluations. In the field of machine translation evaluation, utilizing LLMs with translation error annotations based on Multidimensional Quality Metrics (MQM) yields more human-aligned judgments. However, current LLM-based evaluation methods still face challenges in accurately identifying error spans and assessing their severity. In this paper, we propose HiMATE, a Hierarchical Multi-Agent Framework for Machine Translation Evaluation. We argue that existing approaches inadequately exploit the fine-grained structural and semantic information within the MQM hierarchy. To address this, we develop a hierarchical multi-agent system grounded in the MQM error typology, enabling granular evaluation of subtype errors. Two key strategies are incorporated to further mitigate systemic hallucinations within the framework: the utilization of the model's self-reflection capability and the facilitation of agent discussion involving asymmetric information. Empirically, HiMATE outperforms competitive baselines across different datasets in conducting human-aligned evaluations. Further analyses underscore its significant advantage in error span detection and severity assessment, achieving an average F1-score improvement of 89% over the best-performing baseline. We make our code and data publicly available at https://github.com/nlp2ct-shijie/HiMATE.
>
---
#### [replaced 012] ConvSearch-R1: Enhancing Query Reformulation for Conversational Search with Reasoning via Reinforcement Learning
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2505.15776v2](http://arxiv.org/pdf/2505.15776v2)**

> **作者:** Changtai Zhu; Siyin Wang; Ruijun Feng; Kai Song; Xipeng Qiu
>
> **备注:** Accepted by EMNLP 2025 at the Main Conference
>
> **摘要:** Conversational search systems require effective handling of context-dependent queries that often contain ambiguity, omission, and coreference. Conversational Query Reformulation (CQR) addresses this challenge by transforming these queries into self-contained forms suitable for off-the-shelf retrievers. However, existing CQR approaches suffer from two critical constraints: high dependency on costly external supervision from human annotations or large language models, and insufficient alignment between the rewriting model and downstream retrievers. We present ConvSearch-R1, the first self-driven framework that completely eliminates dependency on external rewrite supervision by leveraging reinforcement learning to optimize reformulation directly through retrieval signals. Our novel two-stage approach combines Self-Driven Policy Warm-Up to address the cold-start problem through retrieval-guided self-distillation, followed by Retrieval-Guided Reinforcement Learning with a specially designed rank-incentive reward shaping mechanism that addresses the sparsity issue in conventional retrieval metrics. Extensive experiments on TopiOCQA and QReCC datasets demonstrate that ConvSearch-R1 significantly outperforms previous state-of-the-art methods, achieving over 10% improvement on the challenging TopiOCQA dataset while using smaller 3B parameter models without any external supervision.
>
---
#### [replaced 013] LinguaLens: Towards Interpreting Linguistic Mechanisms of Large Language Models via Sparse Auto-Encoder
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20344v2](http://arxiv.org/pdf/2502.20344v2)**

> **作者:** Yi Jing; Zijun Yao; Hongzhu Guo; Lingxu Ran; Xiaozhi Wang; Lei Hou; Juanzi Li
>
> **备注:** Accepted by EMNLP 2025 MainConference
>
> **摘要:** Large language models (LLMs) demonstrate exceptional performance on tasks requiring complex linguistic abilities, such as reference disambiguation and metaphor recognition/generation. Although LLMs possess impressive capabilities, their internal mechanisms for processing and representing linguistic knowledge remain largely opaque. Prior research on linguistic mechanisms is limited by coarse granularity, limited analysis scale, and narrow focus. In this study, we propose LinguaLens, a systematic and comprehensive framework for analyzing the linguistic mechanisms of large language models, based on Sparse Auto-Encoders (SAEs). We extract a broad set of Chinese and English linguistic features across four dimensions (morphology, syntax, semantics, and pragmatics). By employing counterfactual methods, we construct a large-scale counterfactual dataset of linguistic features for mechanism analysis. Our findings reveal intrinsic representations of linguistic knowledge in LLMs, uncover patterns of cross-layer and cross-lingual distribution, and demonstrate the potential to control model outputs. This work provides a systematic suite of resources and methods for studying linguistic mechanisms, offers strong evidence that LLMs possess genuine linguistic knowledge, and lays the foundation for more interpretable and controllable language modeling in future research.
>
---
#### [replaced 014] SmallPlan: Leverage Small Language Models for Sequential Path Planning with Simulation-Powered, LLM-Guided Distillation
- **分类: cs.RO; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.00831v5](http://arxiv.org/pdf/2505.00831v5)**

> **作者:** Quang P. M. Pham; Khoi T. N. Nguyen; Nhi H. Doan; Cuong A. Pham; Qinbo Sun; Weimin Qi; Kentaro Inui; Dezhen Song
>
> **备注:** Paper is under review
>
> **摘要:** Efficient path planning in robotics, particularly within large-scale, complex environments, remains a significant hurdle. While Large Language Models (LLMs) offer strong reasoning capabilities, their high computational cost and limited adaptability hinder real-time deployment on edge devices. We present SmallPlan - a novel framework leveraging LLMs as teacher models to train lightweight Small Language Models (SLMs) for high-level path planning tasks. In SmallPlan, the SLMs provide optimal action sequences to navigate across scene graphs that compactly represent full-scaled 3D scenes. The SLMs are trained in a simulation-powered, interleaved manner with LLM-guided supervised fine-tuning (SFT) and reinforcement learning (RL). This strategy not only enables SLMs to successfully complete navigation tasks but also makes them aware of important factors like distance travel, providing more efficient path planning. Through experiments, we demonstrate that the fine-tuned SLMs perform competitively with larger models like GPT-4o on sequential path planning, without suffering from hallucination and overfitting. SmallPlan is resource-efficient, making it well-suited for edge-device deployment and advancing practical autonomous robotics. Our source code is available here: https://github.com/quangpham2006/SmallPlan
>
---
#### [replaced 015] MAC-Tuning: LLM Multi-Compositional Problem Reasoning with Enhanced Knowledge Boundary Awareness
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.21773v4](http://arxiv.org/pdf/2504.21773v4)**

> **作者:** Junsheng Huang; Zhitao He; Yucheng Huang; Sandeep Polisetty; Qingyun Wang; Yi. R Fung
>
> **备注:** We release our code and resource at https://github.com/no-touch-fish/Multi-QA-Tuning. The paper is accepted into EMNLP 2025 main
>
> **摘要:** The hallucination of non-existent facts by LLMs is an important problem given its widespread adoption across various applications. Previous research addresses this problem by analyzing the internal parameterized knowledge boundaries to estimate confidence. However, these studies focus on the single-problem setting and have not explored the more challenging multi-problem setting, which requires accurately answering multiple questions simultaneously. We introduce a novel method for the multi-problem setting, Multiple Answers and Confidence Stepwise Tuning (MAC-Tuning), that separates the learning of answer prediction and confidence estimation during fine-tuning on instruction data. Extensive experiments demonstrate that our method outperforms baselines by up to 25\% in average precision.
>
---
#### [replaced 016] Better To Ask in English? Evaluating Factual Accuracy of Multilingual LLMs in English and Low-Resource Languages
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.20022v2](http://arxiv.org/pdf/2504.20022v2)**

> **作者:** Pritika Rohera; Chaitrali Ginimav; Gayatri Sawant; Raviraj Joshi
>
> **摘要:** Multilingual Large Language Models (LLMs) have demonstrated significant effectiveness across various languages, particularly in high-resource languages such as English. However, their performance in terms of factual accuracy across other low-resource languages, especially Indic languages, remains an area of investigation. In this study, we assess the factual accuracy of LLMs - GPT-4o, Gemma-2-9B, Gemma-2-2B, and Llama-3.1-8B - by comparing their performance in English and Indic languages using the IndicQuest dataset, which contains question-answer pairs in English and 19 Indic languages. By asking the same questions in English and their respective Indic translations, we analyze whether the models are more reliable for regional context questions in Indic languages or when operating in English. Our findings reveal that LLMs often perform better in English, even for questions rooted in Indic contexts. Notably, we observe a higher tendency for hallucination in responses generated in low-resource Indic languages, highlighting challenges in the multilingual understanding capabilities of current LLMs.
>
---
#### [replaced 017] PDFMathTranslate: Scientific Document Translation Preserving Layouts
- **分类: cs.CL; cs.IR; cs.LG; 68T50, 68T45, 68U10, 68U15; D.2.2; I.2.10; I.2.7; J.0**

- **链接: [http://arxiv.org/pdf/2507.03009v3](http://arxiv.org/pdf/2507.03009v3)**

> **作者:** Rongxin Ouyang; Chang Chu; Zhikuang Xin; Xiangyao Ma
>
> **备注:** 7 pages, 4 figures, EMNLP 2025 Demo
>
> **摘要:** Language barriers in scientific documents hinder the diffusion and development of science and technologies. However, prior efforts in translating such documents largely overlooked the information in layouts. To bridge the gap, we introduce PDFMathTranslate, the world's first open-source software for translating scientific documents while preserving layouts. Leveraging the most recent advances in large language models and precise layout detection, we contribute to the community with key improvements in precision, flexibility, and efficiency. The work has been open-sourced at https://github.com/byaidu/pdfmathtranslate with more than 222k downloads.
>
---
#### [replaced 018] Revealing the Inherent Instructability of Pre-Trained Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.02465v3](http://arxiv.org/pdf/2410.02465v3)**

> **作者:** Seokhyun An; Minji Kim; Hyounghun Kim
>
> **备注:** Findings of EMNLP 2025 (32 pages). Code available at https://github.com/seokhyunan/response-tuning
>
> **摘要:** Instruction tuning -- supervised fine-tuning using instruction-response pairs -- is a key step in making pre-trained large language models (LLMs) instructable. Meanwhile, LLMs perform multitask learning during their pre-training, acquiring extensive knowledge and capabilities. We hypothesize that the pre-training stage can enable them to develop the ability to comprehend and address instructions. To verify this, we propose Response Tuning (RT), which removes the instruction and its corresponding mapping to the response from instruction tuning. Instead, it focuses solely on establishing a response distribution. Our experiments demonstrate that RT models, trained only on responses, can effectively respond to a wide range of instructions akin to their instruction-tuned counterparts. In addition, we observe that the models can recognize and reject unsafe queries after learning a safety policy only from the response data. Furthermore, we find that these observations extend to an in-context learning setting. These findings support our hypothesis, highlighting the extensive inherent capabilities of pre-trained LLMs.
>
---
#### [replaced 019] Tackling Fake News in Bengali: Unraveling the Impact of Summarization vs. Augmentation on Pre-trained Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2307.06979v3](http://arxiv.org/pdf/2307.06979v3)**

> **作者:** Arman Sakif Chowdhury; G. M. Shahariar; Ahammed Tarik Aziz; Syed Mohibul Alam; Md. Azad Sheikh; Tanveer Ahmed Belal
>
> **备注:** Accepted, In Production
>
> **摘要:** With the rise of social media and online news sources, fake news has become a significant issue globally. However, the detection of fake news in low resource languages like Bengali has received limited attention in research. In this paper, we propose a methodology consisting of four distinct approaches to classify fake news articles in Bengali using summarization and augmentation techniques with five pre-trained language models. Our approach includes translating English news articles and using augmentation techniques to curb the deficit of fake news articles. Our research also focused on summarizing the news to tackle the token length limitation of BERT based models. Through extensive experimentation and rigorous evaluation, we show the effectiveness of summarization and augmentation in the case of Bengali fake news detection. We evaluated our models using three separate test datasets. The BanglaBERT Base model, when combined with augmentation techniques, achieved an impressive accuracy of 96% on the first test dataset. On the second test dataset, the BanglaBERT model, trained with summarized augmented news articles achieved 97% accuracy. Lastly, the mBERT Base model achieved an accuracy of 86% on the third test dataset which was reserved for generalization performance evaluation. The datasets and implementations are available at https://github.com/arman-sakif/Bengali-Fake-News-Detection
>
---
#### [replaced 020] Understanding the Uncertainty of LLM Explanations: A Perspective Based on Reasoning Topology
- **分类: cs.CL; cs.AI; cs.SC; 68T50, 68T37, 68Q32; I.2.7; I.2.6; I.2.4**

- **链接: [http://arxiv.org/pdf/2502.17026v2](http://arxiv.org/pdf/2502.17026v2)**

> **作者:** Longchao Da; Xiaoou Liu; Jiaxin Dai; Lu Cheng; Yaqing Wang; Hua Wei
>
> **备注:** 28 pages, 9 figures; accepted at COLM'25
>
> **摘要:** Understanding the uncertainty in large language model (LLM) explanations is important for evaluating their faithfulness and reasoning consistency, and thus provides insights into the reliability of LLM's output regarding a question. In this work, we propose a novel framework that quantifies uncertainty in LLM explanations through a reasoning topology perspective. By designing a structural elicitation strategy, we guide the LLMs to frame the explanations of an answer into a graph topology. This process decomposes the explanations into the knowledge related sub-questions and topology-based reasoning structures, which allows us to quantify uncertainty not only at the semantic level but also from the reasoning path. It further brings convenience to assess knowledge redundancy and provide interpretable insights into the reasoning process. Our method offers a systematic way to interpret the LLM reasoning, analyze limitations, and provide guidance for enhancing robustness and faithfulness. This work pioneers the use of graph-structured uncertainty measurement in LLM explanations and demonstrates the potential of topology-based quantification.
>
---
#### [replaced 021] CM-Align: Consistency-based Multilingual Alignment for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.08541v2](http://arxiv.org/pdf/2509.08541v2)**

> **作者:** Xue Zhang; Yunlong Liang; Fandong Meng; Songming Zhang; Yufeng Chen; Jinan Xu; Jie Zhou
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Current large language models (LLMs) generally show a significant performance gap in alignment between English and other languages. To bridge this gap, existing research typically leverages the model's responses in English as a reference to select the best/worst responses in other languages, which are then used for Direct Preference Optimization (DPO) training. However, we argue that there are two limitations in the current methods that result in noisy multilingual preference data and further limited alignment performance: 1) Not all English responses are of high quality, and using a response with low quality may mislead the alignment for other languages. 2) Current methods usually use biased or heuristic approaches to construct multilingual preference pairs. To address these limitations, we design a consistency-based data selection method to construct high-quality multilingual preference data for improving multilingual alignment (CM-Align). Specifically, our method includes two parts: consistency-guided English reference selection and cross-lingual consistency-based multilingual preference data construction. Experimental results on three LLMs and three common tasks demonstrate the effectiveness and superiority of our method, which further indicates the necessity of constructing high-quality preference data.
>
---
#### [replaced 022] Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.24688v4](http://arxiv.org/pdf/2505.24688v4)**

> **作者:** Qinglin Zhu; Runcong Zhao; Hanqi Yan; Yulan He; Yudong Chen; Lin Gui
>
> **备注:** Accepted as a Spotlight at ICML 2025
>
> **摘要:** Large Language Models (LLMs) struggle with complex reasoning due to limited diversity and inefficient search. We propose Soft Reasoning, an embedding-based search framework that optimises the embedding of the first token to guide generation. It combines (1) embedding perturbation for controlled exploration and (2) Bayesian optimisation to refine embeddings via a verifier-guided objective, balancing exploration and exploitation. This approach improves reasoning accuracy and coherence while avoiding reliance on heuristic search. Experiments demonstrate superior correctness with minimal computation, making it a scalable, model-agnostic solution. The code is released at https://github.com/alickzhu/Soft-Reasoning.
>
---
#### [replaced 023] MEPT: Mixture of Expert Prompt Tuning as a Manifold Mapper
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.00996v2](http://arxiv.org/pdf/2509.00996v2)**

> **作者:** Runjia Zeng; Guangyan Sun; Qifan Wang; Tong Geng; Sohail Dianat; Xiaotian Han; Raghuveer Rao; Xueling Zhang; Cheng Han; Lifu Huang; Dongfang Liu
>
> **备注:** EMNLP 2025
>
> **摘要:** Considering deep neural networks as manifold mappers, the pretrain-then-fine-tune paradigm can be interpreted as a two-stage process: pretrain establishes a broad knowledge base, and fine-tune adjusts the model parameters to activate specific neural pathways to align with the target manifold. Although prior fine-tuning approaches demonstrate success, their rigid parameter space limits their ability to dynamically activate appropriate neural pathways, rendering them ill-equipped to adapt flexibly to the diverse and evolving data distributions. In light of this view, we propose a novel approach, Mixture of Expert Prompt Tuning (MEPT), as an effective and efficient manifold-mapping framework. MEPT leverages the Mixture of Experts architecture by integrating multiple prompt experts to adaptively learn diverse and non-stationary data distributions. Empirical evaluations demonstrate that MEPT outperforms several state-of-the-art parameter efficient baselines on SuperGLUE, achieving notable improvements in mean accuracy (e.g., 1.94%) while significantly reducing activated prompts by 79.25%. The effectiveness of MEPT is further supported by theoretical insights from manifold learning and validated through neural activation pathway visualization results. Our code is avaliable at https://runjia.tech/emnlp_mept/.
>
---
#### [replaced 024] A Cross-Cultural Comparison of LLM-based Public Opinion Simulation: Evaluating Chinese and U.S. Models on Diverse Societies
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21587v2](http://arxiv.org/pdf/2506.21587v2)**

> **作者:** Weihong Qi; Fan Huang; Jisun An; Haewoon Kwak
>
> **摘要:** This study evaluates the ability of DeepSeek, an open-source large language model (LLM), to simulate public opinions in comparison to LLMs developed by major tech companies. By comparing DeepSeek-R1 and DeepSeek-V3 with Qwen2.5, GPT-4o, and Llama-3.3 and utilizing survey data from the American National Election Studies (ANES) and the Zuobiao dataset of China, we assess these models' capacity to predict public opinions on social issues in both China and the United States, highlighting their comparative capabilities between countries. Our findings indicate that DeepSeek-V3 performs best in simulating U.S. opinions on the abortion issue compared to other topics such as climate change, gun control, immigration, and services for same-sex couples, primarily because it more accurately simulates responses when provided with Democratic or liberal personas. For Chinese samples, DeepSeek-V3 performs best in simulating opinions on foreign aid and individualism but shows limitations in modeling views on capitalism, particularly failing to capture the stances of low-income and non-college-educated individuals. It does not exhibit significant differences from other models in simulating opinions on traditionalism and the free market. Further analysis reveals that all LLMs exhibit the tendency to overgeneralize a single perspective within demographic groups, often defaulting to consistent responses within groups. These findings highlight the need to mitigate cultural and demographic biases in LLM-driven public opinion modeling, calling for approaches such as more inclusive training methodologies.
>
---
#### [replaced 025] Rumor Detection by Multi-task Suffix Learning based on Time-series Dual Sentiments
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14383v2](http://arxiv.org/pdf/2502.14383v2)**

> **作者:** Zhiwei Liu; Kailai Yang; Eduard Hovy; Sophia Ananiadou
>
> **备注:** work in progress
>
> **摘要:** The widespread dissemination of rumors on social media has a significant impact on people's lives, potentially leading to public panic and fear. Rumors often evoke specific sentiments, resonating with readers and prompting sharing. To effectively detect and track rumors, it is essential to observe the fine-grained sentiments of both source and response message pairs as the rumor evolves over time. However, current rumor detection methods fail to account for this aspect. In this paper, we propose MSuf, the first multi-task suffix learning framework for rumor detection and tracking using time series dual (coupled) sentiments. MSuf includes three modules: (1) an LLM to extract sentiment intensity features and sort them chronologically; (2) a module that fuses the sorted sentiment features with their source text word embeddings to obtain an aligned embedding; (3) two hard prompts are combined with the aligned vector to perform rumor detection and sentiment analysis using one frozen LLM. MSuf effectively enhances the performance of LLMs for rumor detection with only minimal parameter fine-tuning. Evaluating MSuf on four rumor detection benchmarks, we find significant improvements compared to other emotion-based methods.
>
---
#### [replaced 026] Base Models Beat Aligned Models at Randomness and Creativity
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.00047v2](http://arxiv.org/pdf/2505.00047v2)**

> **作者:** Peter West; Christopher Potts
>
> **摘要:** Alignment has quickly become a default ingredient in LLM development, with techniques such as reinforcement learning from human feedback making models act safely, follow instructions, and perform ever-better on complex tasks. While these techniques are certainly useful, we propose that they should not be universally applied and demonstrate a range of tasks on which base language models consistently outperform their popular aligned forms. Particularly, we study tasks that require unpredictable outputs, such as random number generation, mixed strategy games (rock-paper-scissors and hide-and-seek), and creative writing. In each case, aligned models tend towards narrow behaviors that result in distinct disadvantages, for instance, preferring to generate "7" over other uniformly random numbers, becoming almost fully predictable in some game states, or prioritizing pleasant writing over creative originality. Across models tested, better performance on common benchmarks tends to correlate with worse performance on our tasks, suggesting an effective trade-off in the required capabilities.
>
---
#### [replaced 027] Assessing Consistency and Reproducibility in the Outputs of Large Language Models: Evidence Across Diverse Finance and Accounting Tasks
- **分类: q-fin.GN; cs.AI; cs.CE; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.16974v4](http://arxiv.org/pdf/2503.16974v4)**

> **作者:** Julian Junyan Wang; Victor Xiaoqi Wang
>
> **备注:** 76 pages, 20 tables, 12 figures
>
> **摘要:** This study provides the first comprehensive assessment of consistency and reproducibility in Large Language Model (LLM) outputs in finance and accounting research. We evaluate how consistently LLMs produce outputs given identical inputs through extensive experimentation with 50 independent runs across five common tasks: classification, sentiment analysis, summarization, text generation, and prediction. Using three OpenAI models (GPT-3.5-turbo, GPT-4o-mini, and GPT-4o), we generate over 3.4 million outputs from diverse financial source texts and data, covering MD&As, FOMC statements, finance news articles, earnings call transcripts, and financial statements. Our findings reveal substantial but task-dependent consistency, with binary classification and sentiment analysis achieving near-perfect reproducibility, while complex tasks show greater variability. More advanced models do not consistently demonstrate better consistency and reproducibility, with task-specific patterns emerging. LLMs significantly outperform expert human annotators in consistency and maintain high agreement even where human experts significantly disagree. We further find that simple aggregation strategies across 3-5 runs dramatically improve consistency. We also find that aggregation may come with an additional benefit of improved accuracy for sentiment analysis when using newer models. Simulation analysis reveals that despite measurable inconsistency in LLM outputs, downstream statistical inferences remain remarkably robust. These findings address concerns about what we term "G-hacking," the selective reporting of favorable outcomes from multiple generative AI runs, by demonstrating that such risks are relatively low for finance and accounting tasks.
>
---
#### [replaced 028] The Diffusion Duality
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10892v2](http://arxiv.org/pdf/2506.10892v2)**

> **作者:** Subham Sekhar Sahoo; Justin Deschenaux; Aaron Gokaslan; Guanghan Wang; Justin Chiu; Volodymyr Kuleshov
>
> **备注:** ICML 2025. We provide the code at: https://github.com/s-sahoo/duo [v2]: Camera ready revisions
>
> **摘要:** Uniform-state discrete diffusion models hold the promise of fast text generation due to their inherent ability to self-correct. However, they are typically outperformed by autoregressive models and masked diffusion models. In this work, we narrow this performance gap by leveraging a key insight: Uniform-state diffusion processes naturally emerge from an underlying Gaussian diffusion. Our method, Duo, transfers powerful techniques from Gaussian diffusion to improve both training and sampling. First, we introduce a curriculum learning strategy guided by the Gaussian process, doubling training speed by reducing variance. Models trained with curriculum learning surpass autoregressive models in zero-shot perplexity on 3 of 7 benchmarks. Second, we present Discrete Consistency Distillation, which adapts consistency distillation from the continuous to the discrete setting. This algorithm unlocks few-step generation in diffusion language models by accelerating sampling by two orders of magnitude. We provide the code and model checkpoints on the project page: http://s-sahoo.github.io/duo
>
---
#### [replaced 029] MARS-Bench: A Multi-turn Athletic Real-world Scenario Benchmark for Dialogue Evaluation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23810v2](http://arxiv.org/pdf/2505.23810v2)**

> **作者:** Chenghao Yang; Yinbo Luo; Zhoufutu Wen; Qi Chu; Tao Gong; Longxiang Liu; Kaiyuan Zhang; Jianpeng Jiao; Ge Zhang; Wenhao Huang; Nenghai Yu
>
> **备注:** 29 pages, 13 figures, Accepted as EMNLP2025 Findings
>
> **摘要:** Large Language Models (\textbf{LLMs}), e.g. ChatGPT, have been widely adopted in real-world dialogue applications. However, LLMs' robustness, especially in handling long complex dialogue sessions, including frequent motivation transfer, sophisticated cross-turn dependency, is criticized all along. Nevertheless, no existing benchmarks can fully reflect these weaknesses. We present \textbf{MARS-Bench}, a \textbf{M}ulti-turn \textbf{A}thletic \textbf{R}eal-world \textbf{S}cenario Dialogue \textbf{Bench}mark, designed to remedy the gap. MARS-Bench is constructed from play-by-play text commentary so to feature realistic dialogues specifically designed to evaluate three critical aspects of multi-turn conversations: Ultra Multi-turn, Interactive Multi-turn, and Cross-turn Tasks. Extensive experiments on MARS-Bench also reveal that closed-source LLMs significantly outperform open-source alternatives, explicit reasoning significantly boosts LLMs' robustness on handling long complex dialogue sessions, and LLMs indeed face significant challenges when handling motivation transfer and sophisticated cross-turn dependency. Moreover, we provide mechanistic interpretability on how attention sinks due to special tokens lead to LLMs' performance degradation when handling long complex dialogue sessions based on attention visualization experiment in Qwen2.5-7B-Instruction.
>
---
#### [replaced 030] STRICT: Stress Test of Rendering Images Containing Text
- **分类: cs.LG; cs.CL; cs.CV; 68T50; I.2.7; I.4.0**

- **链接: [http://arxiv.org/pdf/2505.18985v2](http://arxiv.org/pdf/2505.18985v2)**

> **作者:** Tianyu Zhang; Xinyu Wang; Lu Li; Zhenghan Tai; Jijun Chi; Jingrui Tian; Hailin He; Suyuchen Wang
>
> **备注:** Accepted as a main conference paper at EMNLP 2025
>
> **摘要:** While diffusion models have revolutionized text-to-image generation with their ability to synthesize realistic and diverse scenes, they continue to struggle to generate consistent and legible text within images. This shortcoming is commonly attributed to the locality bias inherent in diffusion-based generation, which limits their ability to model long-range spatial dependencies. In this paper, we introduce $\textbf{STRICT}$, a benchmark designed to systematically stress-test the ability of diffusion models to render coherent and instruction-aligned text in images. Our benchmark evaluates models across multiple dimensions: (1) the maximum length of readable text that can be generated; (2) the correctness and legibility of the generated text, and (3) the ratio of not following instructions for generating text. We evaluate several state-of-the-art models, including proprietary and open-source variants, and reveal persistent limitations in long-range consistency and instruction-following capabilities. Our findings provide insights into architectural bottlenecks and motivate future research directions in multimodal generative modeling. We release our entire evaluation pipeline at https://github.com/tianyu-z/STRICT-Bench.
>
---
#### [replaced 031] Hopscotch: Discovering and Skipping Redundancies in Language Models
- **分类: cs.CL; cs.AI; cs.LG; 68T50; I.2.7; I.2.6; I.2.4**

- **链接: [http://arxiv.org/pdf/2506.03303v2](http://arxiv.org/pdf/2506.03303v2)**

> **作者:** Mustafa Eyceoz; Nikhil Shivakumar Nayak; Hao Wang; Ligong Han; Akash Srivastava
>
> **备注:** 10 pages, 4 figures, 9 tables
>
> **摘要:** Modern causal language models stack many attention blocks to improve performance, but not all blocks are necessary for every task. We propose Hopscotch, a simple yet effective method that identifies and skips attention blocks with least contributions to a task and adapts to preserve output quality. Hopscotch jointly optimizes which blocks to skip and how to scale the outputs of the remaining layers. By introducing lightweight, trainable scaling parameters to attention and MLP blocks, it mitigates distribution shifts in hidden states caused by removing attention blocks. Hopscotch does not modify model weights or require access to pretraining or instruction-tuning data, and is compatible with existing model compression techniques. When applied to $\texttt{Llama-3.1-8B}$ and $\texttt{Qwen2.5-7B}$, Hopscotch achieves less than a 2% drop in performance even after skipping four attention blocks.
>
---
#### [replaced 032] Speak-to-Structure: Evaluating LLMs in Open-domain Natural Language-Driven Molecule Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.14642v3](http://arxiv.org/pdf/2412.14642v3)**

> **作者:** Jiatong Li; Junxian Li; Weida Wang; Yunqing Liu; Changmeng Zheng; Dongzhan Zhou; Xiao-yong Wei; Qing Li
>
> **备注:** Our codes and datasets are available through https://github.com/phenixace/TOMG-Bench
>
> **摘要:** Recently, Large Language Models (LLMs) have shown great potential in natural language-driven molecule discovery. However, existing datasets and benchmarks for molecule-text alignment are predominantly built on a one-to-one mapping, measuring LLMs' ability to retrieve a single, pre-defined answer, rather than their creative potential to generate diverse, yet equally valid, molecular candidates. To address this critical gap, we propose Speak-to-Structure (S^2-Bench}), the first benchmark to evaluate LLMs in open-domain natural language-driven molecule generation. S^2-Bench is specifically designed for one-to-many relationships, challenging LLMs to demonstrate genuine molecular understanding and generation capabilities. Our benchmark includes three key tasks: molecule editing (MolEdit), molecule optimization (MolOpt), and customized molecule generation (MolCustom), each probing a different aspect of molecule discovery. We also introduce OpenMolIns, a large-scale instruction tuning dataset that enables Llama-3.1-8B to surpass the most powerful LLMs like GPT-4o and Claude-3.5 on S^2-Bench. Our comprehensive evaluation of 28 LLMs shifts the focus from simple pattern recall to realistic molecular design, paving the way for more capable LLMs in natural language-driven molecule discovery.
>
---
#### [replaced 033] EasyEdit2: An Easy-to-use Steering Framework for Editing Large Language Models
- **分类: cs.CL; cs.AI; cs.CV; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.15133v3](http://arxiv.org/pdf/2504.15133v3)**

> **作者:** Ziwen Xu; Shuxun Wang; Kewei Xu; Haoming Xu; Mengru Wang; Xinle Deng; Yunzhi Yao; Guozhou Zheng; Huajun Chen; Ningyu Zhang
>
> **备注:** EMNLP 2025 System Demonstrations. Demo: https://www.youtube.com/watch?v=AkfoiPfp5rQ; code: https://github.com/zjunlp/EasyEdit
>
> **摘要:** In this paper, we introduce EasyEdit2, a framework designed to enable plug-and-play adjustability for controlling Large Language Model (LLM) behaviors. EasyEdit2 supports a wide range of test-time interventions, including safety, sentiment, personality, reasoning patterns, factuality, and language features. Unlike its predecessor, EasyEdit2 features a new architecture specifically designed for seamless model steering. It comprises key modules such as the steering vector generator and the steering vector applier, which enable automatic generation and application of steering vectors to influence the model's behavior without modifying its parameters. One of the main advantages of EasyEdit2 is its ease of use-users do not need extensive technical knowledge. With just a single example, they can effectively guide and adjust the model's responses, making precise control both accessible and efficient. Empirically, we report model steering performance across different LLMs, demonstrating the effectiveness of these techniques. We have released the source code on GitHub at https://github.com/zjunlp/EasyEdit along with a demonstration notebook. In addition, we provide a demo video at https://www.youtube.com/watch?v=AkfoiPfp5rQ for a quick introduction.
>
---
#### [replaced 034] Low-rank variational dropout: Uncertainty and rank selection in adapters
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.22809v2](http://arxiv.org/pdf/2506.22809v2)**

> **作者:** Cooper Doyle
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) methods such as LoRA adapt large language models by inserting low-rank adapters, but they leave open two key questions: how to give the adapted model calibrated uncertainty, and how to choose the adapter rank. Existing approaches to uncertainty are typically post-hoc, while rank selection is manual and task-specific. BayesLoRA revisits variational dropout in the LoRA setting and shows that the natural unit of stochasticity is not individual weights but entire ranks of the adapter. By placing rank-wise variational distributions over adapter components, BayesLoRA defines a posterior that (i) yields calibrated predictions through adapter-only Monte Carlo sampling and (ii) prunes redundant ranks automatically via an ARD-style KL term. Theoretical analysis shows that this rank-parameterized posterior localizes uncertainty to the adapted subspace and explains amplification under distribution shift. Empirically, BayesLoRA improves calibration while at the same time producing lighter, faster adapters, removing the need to tune ranks by hand. This dual role of uncertainty estimation and uncertainty-driven pruning suggests BayesLoRA may offer a practical default for reliable and efficient PEFT.
>
---
#### [replaced 035] Too Helpful, Too Harmless, Too Honest or Just Right?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.08486v2](http://arxiv.org/pdf/2509.08486v2)**

> **作者:** Gautam Siddharth Kashyap; Mark Dras; Usman Naseem
>
> **备注:** EMNLP'25 Main
>
> **摘要:** Large Language Models (LLMs) exhibit strong performance across a wide range of NLP tasks, yet aligning their outputs with the principles of Helpfulness, Harmlessness, and Honesty (HHH) remains a persistent challenge. Existing methods often optimize for individual alignment dimensions in isolation, leading to trade-offs and inconsistent behavior. While Mixture-of-Experts (MoE) architectures offer modularity, they suffer from poorly calibrated routing, limiting their effectiveness in alignment tasks. We propose TrinityX, a modular alignment framework that incorporates a Mixture of Calibrated Experts (MoCaE) within the Transformer architecture. TrinityX leverages separately trained experts for each HHH dimension, integrating their outputs through a calibrated, task-adaptive routing mechanism that combines expert signals into a unified, alignment-aware representation. Extensive experiments on three standard alignment benchmarks-Alpaca (Helpfulness), BeaverTails (Harmlessness), and TruthfulQA (Honesty)-demonstrate that TrinityX outperforms strong baselines, achieving relative improvements of 32.5% in win rate, 33.9% in safety score, and 28.4% in truthfulness. In addition, TrinityX reduces memory usage and inference latency by over 40% compared to prior MoE-based approaches. Ablation studies highlight the importance of calibrated routing, and cross-model evaluations confirm TrinityX's generalization across diverse LLM backbones.
>
---
#### [replaced 036] ISACL: Internal State Analyzer for Copyrighted Training Data Leakage
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.17767v2](http://arxiv.org/pdf/2508.17767v2)**

> **作者:** Guangwei Zhang; Qisheng Su; Jiateng Liu; Cheng Qian; Yanzhou Pan; Yanjie Fu; Denghui Zhang
>
> **摘要:** Large Language Models (LLMs) have revolutionized Natural Language Processing (NLP) but pose risks of inadvertently exposing copyrighted or proprietary data, especially when such data is used for training but not intended for distribution. Traditional methods address these leaks only after content is generated, which can lead to the exposure of sensitive information. This study introduces a proactive approach: examining LLMs' internal states before text generation to detect potential leaks. By using a curated dataset of copyrighted materials, we trained a neural network classifier to identify risks, allowing for early intervention by stopping the generation process or altering outputs to prevent disclosure. Integrated with a Retrieval-Augmented Generation (RAG) system, this framework ensures adherence to copyright and licensing requirements while enhancing data privacy and ethical standards. Our results show that analyzing internal states effectively mitigates the risk of copyrighted data leakage, offering a scalable solution that fits smoothly into AI workflows, ensuring compliance with copyright regulations while maintaining high-quality text generation. The implementation is available on GitHub.\footnote{https://github.com/changhu73/Internal_states_leakage}
>
---
#### [replaced 037] CAC-CoT: Connector-Aware Compact Chain-of-Thought for Efficient Reasoning Data Synthesis Across Dual-System Cognitive Tasks
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.18743v2](http://arxiv.org/pdf/2508.18743v2)**

> **作者:** Sunguk Choi; Yonghoon Kwon; Heondeuk Lee
>
> **备注:** Accepted at EMNLP 2025 findings
>
> **摘要:** Long chain-of-thought (CoT) prompting helps Large Language Models (LLMs) solve difficult problems, but very long traces often slow or even degrade performance on fast, intuitive "System-1" tasks. We introduce Connector-Aware Compact CoT (CAC-CoT) -- a method that deliberately restricts reasoning to a small, fixed set of connector phrases, steering the model toward concise and well -- structured explanations. Despite its simplicity, our synthetic method with general-purpose LLMs yields a high-quality training quality. CAC-CoT achieves approximately 85% on GSM8K and approximately 40% on GPQA (System-2) while also achieving approximately 85% on S1-Bench (System-1), surpassing the baseline by over 20%. Its reasoning traces average approximately 300 tokens(ART), about one-third the length of baseline traces, delivering higher efficiency without loss of accuracy.
>
---
#### [replaced 038] From Personas to Talks: Revisiting the Impact of Personas on LLM-Synthesized Emotional Support Conversations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11451v2](http://arxiv.org/pdf/2502.11451v2)**

> **作者:** Shenghan Wu; Yimo Zhu; Wynne Hsu; Mong-Li Lee; Yang Deng
>
> **备注:** Accepted by EMNLP 2025 Main Conference
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has revolutionized the generation of emotional support conversations (ESC), offering scalable solutions with reduced costs and enhanced data privacy. This paper explores the role of personas in the creation of ESC by LLMs. Our research utilizes established psychological frameworks to measure and infuse persona traits into LLMs, which then generate dialogues in the emotional support scenario. We conduct extensive evaluations to understand the stability of persona traits in dialogues, examining shifts in traits post-generation and their impact on dialogue quality and strategy distribution. Experimental results reveal several notable findings: 1) LLMs can infer core persona traits, 2) subtle shifts in emotionality and extraversion occur, influencing the dialogue dynamics, and 3) the application of persona traits modifies the distribution of emotional support strategies, enhancing the relevance and empathetic quality of the responses. These findings highlight the potential of persona-driven LLMs in crafting more personalized, empathetic, and effective emotional support dialogues, which has significant implications for the future design of AI-driven emotional support systems.
>
---
#### [replaced 039] Hallucinated Span Detection with Multi-View Attention Features
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.04335v2](http://arxiv.org/pdf/2504.04335v2)**

> **作者:** Yuya Ogasa; Yuki Arase
>
> **摘要:** This study addresses the problem of hallucinated span detection in the outputs of large language models. It has received less attention than output-level hallucination detection despite its practical importance. Prior work has shown that attentions often exhibit irregular patterns when hallucinations occur. Motivated by these findings, we extract features from the attention matrix that provide complementary views capturing (a) whether certain tokens are influential or ignored, (b) whether attention is biased toward specific subsets, and (c) whether a token is generated referring to a narrow or broad context, in the generation. These features are input to a Transformer-based classifier to conduct sequential labelling to identify hallucinated spans. Experimental results indicate that the proposed method outperforms strong baselines on hallucinated span detection with longer input contexts, such as data-to-text and summarisation tasks.
>
---
#### [replaced 040] GP-GPT: Large Language Model for Gene-Phenotype Mapping
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.09825v3](http://arxiv.org/pdf/2409.09825v3)**

> **作者:** Yanjun Lyu; Zihao Wu; Lu Zhang; Jing Zhang; Yiwei Li; Wei Ruan; Zhengliang Liu; Xiang Li; Rongjie Liu; Chao Huang; Wentao Li; Tianming Liu; Dajiang Zhu
>
> **摘要:** Pre-trained large language models(LLMs) have attracted increasing attention in biomedical domains due to their success in natural language processing. However, the complex traits and heterogeneity of multi-sources genomics data pose significant challenges when adapting these models to the bioinformatics and biomedical field. To address these challenges, we present GP-GPT, the first specialized large language model for genetic-phenotype knowledge representation and genomics relation analysis. Our model is fine-tuned in two stages on a comprehensive corpus composed of over 3,000,000 terms in genomics, proteomics, and medical genetics, derived from multiple large-scale validated datasets and scientific publications. GP-GPT demonstrates proficiency in accurately retrieving medical genetics information and performing common genomics analysis tasks, such as genomics information retrieval and relationship determination. Comparative experiments across domain-specific tasks reveal that GP-GPT outperforms state-of-the-art LLMs, including Llama2, Llama3 and GPT-4. These results highlight GP-GPT's potential to enhance genetic disease relation research and facilitate accurate and efficient analysis in the fields of genomics and medical genetics. Our investigation demonstrated the subtle changes of bio-factor entities' representations in the GP-GPT, which suggested the opportunities for the application of LLMs to advancing gene-phenotype research.
>
---
#### [replaced 041] GeoGuess: Multimodal Reasoning based on Hierarchy of Visual Information in Street View
- **分类: cs.CL; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2506.16633v2](http://arxiv.org/pdf/2506.16633v2)**

> **作者:** Fenghua Cheng; Jinxiang Wang; Sen Wang; Zi Huang; Xue Li
>
> **备注:** Updated version
>
> **摘要:** Multimodal reasoning is a process of understanding, integrating and inferring information across different data modalities. It has recently attracted surging academic attention as a benchmark for Artificial Intelligence (AI). Although there are various tasks for evaluating multimodal reasoning ability, they still have limitations. Lack of reasoning on hierarchical visual clues at different levels of granularity, e.g., local details and global context, is of little discussion, despite its frequent involvement in real scenarios. To bridge the gap, we introduce a novel and challenging task for multimodal reasoning, namely GeoGuess. Given a street view image, the task is to identify its location and provide a detailed explanation. A system that succeeds in GeoGuess should be able to detect tiny visual clues, perceive the broader landscape, and associate with vast geographic knowledge. Therefore, GeoGuess would require the ability to reason between hierarchical visual information and geographic knowledge. In this work, we establish a benchmark for GeoGuess by introducing a specially curated dataset GeoExplain which consists of panoramas-geocoordinates-explanation tuples. Additionally, we present a multimodal and multilevel reasoning method, namely SightSense which can make prediction and generate comprehensive explanation based on hierarchy of visual information and external knowledge. Our analysis and experiments demonstrate their outstanding performance in GeoGuess.
>
---
#### [replaced 042] Towards Reliable and Interpretable Document Question Answering via VLMs
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2509.10129v2](http://arxiv.org/pdf/2509.10129v2)**

> **作者:** Alessio Chen; Simone Giovannini; Andrea Gemelli; Fabio Coppini; Simone Marinai
>
> **摘要:** Vision-Language Models (VLMs) have shown strong capabilities in document understanding, particularly in identifying and extracting textual information from complex documents. Despite this, accurately localizing answers within documents remains a major challenge, limiting both interpretability and real-world applicability. To address this, we introduce DocExplainerV0, a plug-and-play bounding-box prediction module that decouples answer generation from spatial localization. This design makes it applicable to existing VLMs, including proprietary systems where fine-tuning is not feasible. Through systematic evaluation, we provide quantitative insights into the gap between textual accuracy and spatial grounding, showing that correct answers often lack reliable localization. Our standardized framework highlights these shortcomings and establishes a benchmark for future research toward more interpretable and robust document information extraction VLMs.
>
---
#### [replaced 043] Synthesize-on-Graph: Knowledgeable Synthetic Data Generation for Continue Pre-training of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.00979v3](http://arxiv.org/pdf/2505.00979v3)**

> **作者:** Shengjie Ma; Xuhui Jiang; Chengjin Xu; Cehao Yang; Liyu Zhang; Jian Guo
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success but remain data-inefficient, especially when learning from small, specialized corpora with limited and proprietary data. Existing synthetic data generation methods for continue pre-training focus on intra-document content and overlook cross-document knowledge associations, limiting content diversity and depth. We propose Synthetic-on-Graph (SoG), a synthetic data generation framework that incorporates cross-document knowledge associations for efficient corpus expansion. SoG constructs a context graph by extracting entities and concepts from the original corpus, representing cross-document associations, and employing a graph walk strategy for knowledge-associated sampling. This enhances synthetic data diversity and coherence, enabling models to learn complex knowledge structures and handle rare knowledge. To further improve the quality of synthetic data, we integrate two complementary strategies, Chain-of-Thought (CoT) and Contrastive Clarifying (CC), to enhance both reasoning capability and discriminative power. Extensive experiments demonstrate that SoG surpasses state-of-the-art (SOTA) methods on multi-hop and domain-specific question answering, while achieving competitive performance on long-context reading comprehension. These results highlight the superior generalization ability of SoG. Our work advances the paradigm of synthetic data generation and offers practical solutions for efficient knowledge acquisition in LLMs, particularly for downstream tasks and domains with limited training data.
>
---
#### [replaced 044] Monitoring Decoding: Mitigating Hallucination via Evaluating the Factuality of Partial Response during Generation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.03106v2](http://arxiv.org/pdf/2503.03106v2)**

> **作者:** Yurui Chang; Bochuan Cao; Lu Lin
>
> **备注:** Accepted to ACL 2025 (Findings)
>
> **摘要:** While large language models have demonstrated exceptional performance across a wide range of tasks, they remain susceptible to hallucinations -- generating plausible yet factually incorrect contents. Existing methods to mitigating such risk often rely on sampling multiple full-length generations, which introduces significant response latency and becomes ineffective when the model consistently produces hallucinated outputs with high confidence. To address these limitations, we introduce Monitoring Decoding (MD), a novel framework that dynamically monitors the generation process and selectively applies in-process interventions, focusing on revising crucial tokens responsible for hallucinations. Instead of waiting until completion of multiple full-length generations, we identify hallucination-prone tokens during generation using a monitor function, and further refine these tokens through a tree-based decoding strategy. This approach ensures an enhanced factual accuracy and coherence in the generated output while maintaining efficiency. Experimental results demonstrate that MD consistently outperforms self-consistency-based approaches in both effectiveness and efficiency, achieving higher factual accuracy while significantly reducing computational overhead.
>
---
#### [replaced 045] Rethinking LLM-Based Recommendations: A Personalized Query-Driven Parallel Integration
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.11889v2](http://arxiv.org/pdf/2504.11889v2)**

> **作者:** Donghee Han; Hwanjun Song; Mun Yong Yi
>
> **摘要:** Recent studies have explored integrating large language models (LLMs) into recommendation systems but face several challenges, including training-induced bias and bottlenecks from serialized architecture. To effectively address these issues, we propose a Query-toRecommendation, a parallel recommendation framework that decouples LLMs from candidate pre-selection and instead enables direct retrieval over the entire item pool. Our framework connects LLMs and recommendation models in a parallel manner, allowing each component to independently utilize its strengths without interfering with the other. In this framework, LLMs are utilized to generate feature-enriched item descriptions and personalized user queries, allowing for capturing diverse preferences and enabling rich semantic matching in a zero-shot manner. To effectively combine the complementary strengths of LLM and collaborative signals, we introduce an adaptive reranking strategy. Extensive experiments demonstrate an improvement in performance up to 57%, while also improving the novelty and diversity of recommendations.
>
---
#### [replaced 046] Assessing LLMs in Art Contexts: Critique Generation and Theory of Mind Evaluation
- **分类: cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2504.12805v2](http://arxiv.org/pdf/2504.12805v2)**

> **作者:** Takaya Arita; Wenxian Zheng; Reiji Suzuki; Fuminori Akiba
>
> **备注:** Corrected a typo in the metadata title only ("Assesing"->"Assessing"). No changes were made to the PDF or source files
>
> **摘要:** This study explored how large language models (LLMs) perform in two areas related to art: writing critiques of artworks and reasoning about mental states (Theory of Mind, or ToM) in art-related situations. For the critique generation part, we built a system that combines Noel Carroll's evaluative framework with a broad selection of art criticism theories. The model was prompted to first write a full-length critique and then shorter, more coherent versions using a step-by-step prompting process. These AI-generated critiques were then compared with those written by human experts in a Turing test-style evaluation. In many cases, human subjects had difficulty telling which was which, and the results suggest that LLMs can produce critiques that are not only plausible in style but also rich in interpretation, as long as they are carefully guided. In the second part, we introduced new simple ToM tasks based on situations involving interpretation, emotion, and moral tension, which can appear in the context of art. These go beyond standard false-belief tests and allow for more complex, socially embedded forms of reasoning. We tested 41 recent LLMs and found that their performance varied across tasks and models. In particular, tasks that involved affective or ambiguous situations tended to reveal clearer differences. Taken together, these results help clarify how LLMs respond to complex interpretative challenges, revealing both their cognitive limitations and potential. While our findings do not directly contradict the so-called Generative AI Paradox--the idea that LLMs can produce expert-like output without genuine understanding--they suggest that, depending on how LLMs are instructed, such as through carefully designed prompts, these models may begin to show behaviors that resemble understanding more closely than we might assume.
>
---
#### [replaced 047] Understanding Emergent In-Context Learning from a Kernel Regression Perspective
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2305.12766v3](http://arxiv.org/pdf/2305.12766v3)**

> **作者:** Chi Han; Ziqi Wang; Han Zhao; Heng Ji
>
> **备注:** Transactions on Machine Learning Research (TMLR 2025)
>
> **摘要:** Large language models (LLMs) have initiated a paradigm shift in transfer learning. In contrast to the classic pretraining-then-finetuning procedure, in order to use LLMs for downstream prediction tasks, one only needs to provide a few demonstrations, known as in-context examples, without adding more or updating existing model parameters. This in-context learning (ICL) capability of LLMs is intriguing, and it is not yet fully understood how pretrained LLMs acquire such capabilities. In this paper, we investigate the reason why a transformer-based language model can accomplish in-context learning after pre-training on a general language corpus by proposing a kernel-regression perspective of understanding LLMs' ICL bahaviors when faced with in-context examples. More concretely, we first prove that Bayesian inference on in-context prompts can be asymptotically understood as kernel regression $\hat y = \sum_i y_i K(x, x_i)/\sum_i K(x, x_i)$ as the number of in-context demonstrations grows. Then, we empirically investigate the in-context behaviors of language models. We find that during ICL, the attention and hidden features in LLMs match the behaviors of a kernel regression. Finally, our theory provides insights into multiple phenomena observed in the ICL field: why retrieving demonstrative samples similar to test samples can help, why ICL performance is sensitive to the output formats, and why ICL accuracy benefits from selecting in-distribution and representative samples. Code and resources are publicly available at https://github.com/Glaciohound/Explain-ICL-As-Kernel-Regression.
>
---
#### [replaced 048] Time is On My Side: Dynamics of Talk-Time Sharing in Video-chat Conversations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.20474v3](http://arxiv.org/pdf/2506.20474v3)**

> **作者:** Kaixiang Zhang; Justine Zhang; Cristian Danescu-Niculescu-Mizil
>
> **备注:** Accepted for publication at CSCW 2025. Code and data available in ConvoKit (https://convokit.cornell.edu)
>
> **摘要:** An intrinsic aspect of every conversation is the way talk-time is shared between multiple speakers. Conversations can be balanced, with each speaker claiming a similar amount of talk-time, or imbalanced when one talks disproportionately. Such overall distributions are the consequence of continuous negotiations between the speakers throughout the conversation: who should be talking at every point in time, and for how long? In this work we introduce a computational framework for quantifying both the conversation-level distribution of talk-time between speakers, as well as the lower-level dynamics that lead to it. We derive a typology of talk-time sharing dynamics structured by several intuitive axes of variation. By applying this framework to a large dataset of video-chats between strangers, we confirm that, perhaps unsurprisingly, different conversation-level distributions of talk-time are perceived differently by speakers, with balanced conversations being preferred over imbalanced ones, especially by those who end up talking less. Then we reveal that -- even when they lead to the same level of overall balance -- different types of talk-time sharing dynamics are perceived differently by the participants, highlighting the relevance of our newly introduced typology. Finally, we discuss how our framework offers new tools to designers of computer-mediated communication platforms, for both human-human and human-AI communication.
>
---
#### [replaced 049] LLM as a Broken Telephone: Iterative Generation Distorts Information
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.20258v2](http://arxiv.org/pdf/2502.20258v2)**

> **作者:** Amr Mohamed; Mingmeng Geng; Michalis Vazirgiannis; Guokan Shang
>
> **备注:** Accepted to ACL 2025, Main Conference
>
> **摘要:** As large language models are increasingly responsible for online content, concerns arise about the impact of repeatedly processing their own outputs. Inspired by the "broken telephone" effect in chained human communication, this study investigates whether LLMs similarly distort information through iterative generation. Through translation-based experiments, we find that distortion accumulates over time, influenced by language choice and chain complexity. While degradation is inevitable, it can be mitigated through strategic prompting techniques. These findings contribute to discussions on the long-term effects of AI-mediated information propagation, raising important questions about the reliability of LLM-generated content in iterative workflows.
>
---
#### [replaced 050] Can LLMs assist with Ambiguity? A Quantitative Evaluation of various Large Language Models on Word Sense Disambiguation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.18337v5](http://arxiv.org/pdf/2411.18337v5)**

> **作者:** T. G. D. K. Sumanathilaka; Nicholas Micallef; Julian Hough
>
> **备注:** 12 pages,6 tables, 1 figure, Proceedings of the 1st International Conference on NLP & AI for Cyber Security
>
> **摘要:** Ambiguous words are often found in modern digital communications. Lexical ambiguity challenges traditional Word Sense Disambiguation (WSD) methods, due to limited data. Consequently, the efficiency of translation, information retrieval, and question-answering systems is hindered by these limitations. This study investigates the use of Large Language Models (LLMs) to improve WSD using a novel approach combining a systematic prompt augmentation mechanism with a knowledge base (KB) consisting of different sense interpretations. The proposed method incorporates a human-in-loop approach for prompt augmentation where prompt is supported by Part-of-Speech (POS) tagging, synonyms of ambiguous words, aspect-based sense filtering and few-shot prompting to guide the LLM. By utilizing a few-shot Chain of Thought (COT) prompting-based approach, this work demonstrates a substantial improvement in performance. The evaluation was conducted using FEWS test data and sense tags. This research advances accurate word interpretation in social media and digital communication.
>
---
#### [replaced 051] Chain of Strategy Optimization Makes Large Language Models Better Emotional Supporter
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.05362v2](http://arxiv.org/pdf/2503.05362v2)**

> **作者:** Weixiang Zhao; Xingyu Sui; Xinyang Han; Yang Deng; Yulin Hu; Jiahe Guo; Libo Qin; Qianyun Du; Shijin Wang; Yanyan Zhao; Bing Qin; Ting Liu
>
> **备注:** 21 pages, 9 figures, 17 tables
>
> **摘要:** The growing emotional stress in modern society has increased the demand for Emotional Support Conversations (ESC). While Large Language Models (LLMs) show promise for ESC, they face two key challenges: (1) low strategy selection accuracy, and (2) preference bias, limiting their adaptability to emotional needs of users. Existing supervised fine-tuning (SFT) struggles to address these issues, as it rigidly trains models on single gold-standard responses without modeling nuanced strategy trade-offs. To overcome these limitations, we propose Chain-of-Strategy Optimization (CSO), a novel approach that optimizes strategy selection preferences at each dialogue turn. We first leverage Monte Carlo Tree Search to construct ESC-Pro, a high-quality preference dataset with turn-level strategy-response pairs. Training on ESC-Pro with CSO improves both strategy accuracy and bias mitigation, enabling LLMs to generate more empathetic and contextually appropriate responses. Experiments on LLaMA-3.1-8B, Gemma-2-9B, and Qwen2.5-7B demonstrate that CSO outperforms standard SFT, highlighting the efficacy of fine-grained, turn-level preference modeling in ESC.
>
---
#### [replaced 052] Steering LVLMs via Sparse Autoencoder for Hallucination Mitigation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16146v2](http://arxiv.org/pdf/2505.16146v2)**

> **作者:** Zhenglin Hua; Jinghan He; Zijun Yao; Tianxu Han; Haiyun Guo; Yuheng Jia; Junfeng Fang
>
> **备注:** Accepted to Findings of EMNLP 2025
>
> **摘要:** Large vision-language models (LVLMs) have achieved remarkable performance on multimodal tasks. However, they still suffer from hallucinations, generating text inconsistent with visual input, posing significant risks in real-world applications. Existing approaches to address this issue focus on incorporating external knowledge bases, alignment training, or decoding strategies, all of which require substantial computational cost and time. Recent works try to explore more efficient alternatives by adjusting LVLMs' internal representations. Although promising, these methods may cause hallucinations to be insufficiently suppressed or lead to excessive interventions that negatively affect normal semantics. In this work, we leverage sparse autoencoders (SAEs) to identify semantic directions closely associated with faithfulness or hallucination, extracting more precise and disentangled hallucination-related representations. Our analysis demonstrates that interventions along the identified faithful direction can mitigate hallucinations, while those along the hallucinatory direction can exacerbate them. Building on these insights, we propose Steering LVLMs via SAE Latent Directions (SSL), a plug-and-play method based on SAE-derived latent directions to mitigate hallucinations in LVLMs. Extensive experiments demonstrate that SSL significantly outperforms existing decoding approaches in mitigating hallucinations, while maintaining transferability across different model architectures with negligible additional time overhead. The code is available at https://github.com/huazhenglin2003/SSL.
>
---
#### [replaced 053] ReliableEval: A Recipe for Stochastic LLM Evaluation via Method of Moments
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22169v2](http://arxiv.org/pdf/2505.22169v2)**

> **作者:** Gili Lior; Eliya Habba; Shahar Levy; Avi Caciularu; Gabriel Stanovsky
>
> **备注:** Findings of EMNLP 2025
>
> **摘要:** LLMs are highly sensitive to prompt phrasing, yet standard benchmarks typically report performance using a single prompt, raising concerns about the reliability of such evaluations. In this work, we argue for a stochastic method of moments evaluation over the space of meaning-preserving prompt perturbations. We introduce a formal definition of reliable evaluation that accounts for prompt sensitivity, and suggest ReliableEval - a method for estimating the number of prompt resamplings needed to obtain meaningful results. Using our framework, we stochastically evaluate five frontier LLMs and find that even top-performing models like GPT-4o and Claude-3.7-Sonnet exhibit substantial prompt sensitivity. Our approach is model-, task-, and metric-agnostic, offering a recipe for meaningful and robust LLM evaluation.
>
---
#### [replaced 054] Surveying the Landscape of Image Captioning Evaluation: A Comprehensive Taxonomy, Trends and Metrics Analysis
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2408.04909v3](http://arxiv.org/pdf/2408.04909v3)**

> **作者:** Uri Berger; Gabriel Stanovsky; Omri Abend; Lea Frermann
>
> **摘要:** The task of image captioning has recently been gaining popularity, and with it the complex task of evaluating the quality of image captioning models. In this work, we present the first survey and taxonomy of over 70 different image captioning metrics and their usage in hundreds of papers, specifically designed to help users select the most suitable metric for their needs. We find that despite the diversity of proposed metrics, the vast majority of studies rely on only five popular metrics, which we show to be weakly correlated with human ratings. We hypothesize that combining a diverse set of metrics can enhance correlation with human ratings. As an initial step, we demonstrate that a linear regression-based ensemble method, which we call EnsembEval, trained on one human ratings dataset, achieves improved correlation across five additional datasets, showing there is a lot of room for improvement by leveraging a diverse set of metrics.
>
---
#### [replaced 055] Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2309.01219v3](http://arxiv.org/pdf/2309.01219v3)**

> **作者:** Yue Zhang; Yafu Li; Leyang Cui; Deng Cai; Lemao Liu; Tingchen Fu; Xinting Huang; Enbo Zhao; Yu Zhang; Chen Xu; Yulong Chen; Longyue Wang; Anh Tuan Luu; Wei Bi; Freda Shi; Shuming Shi
>
> **备注:** work in progress;
>
> **摘要:** While large language models (LLMs) have demonstrated remarkable capabilities across a range of downstream tasks, a significant concern revolves around their propensity to exhibit hallucinations: LLMs occasionally generate content that diverges from the user input, contradicts previously generated context, or misaligns with established world knowledge. This phenomenon poses a substantial challenge to the reliability of LLMs in real-world scenarios. In this paper, we survey recent efforts on the detection, explanation, and mitigation of hallucination, with an emphasis on the unique challenges posed by LLMs. We present taxonomies of the LLM hallucination phenomena and evaluation benchmarks, analyze existing approaches aiming at mitigating LLM hallucination, and discuss potential directions for future research.
>
---
#### [replaced 056] Efficient Environmental Claim Detection with Hyperbolic Graph Neural Networks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13628v2](http://arxiv.org/pdf/2502.13628v2)**

> **作者:** Darpan Aswal; Manjira Sinha
>
> **摘要:** Transformer based models, specially large language models (LLMs) dominate the field of NLP with their mass adoption in tasks such as text generation, summarization and fake news detection. These models offer ease of deployment and reliability for most applications, however, they require significant amounts of computational power for training as well as inference. This poses challenges in their adoption in resource-constrained applications, specially in the open-source community where compute availability is usually scarce. This work proposes a graph-based approach for Environmental Claim Detection, exploring Graph Neural Networks (GNNs) and Hyperbolic Graph Neural Networks (HGNNs) as lightweight yet effective alternatives to transformer-based models. Re-framing the task as a graph classification problem, we transform claim sentences into dependency parsing graphs, utilizing a combination of word2vec \& learnable part-of-speech (POS) tag embeddings for the node features and encoding syntactic dependencies in the edge relations. Our results show that our graph-based models, particularly HGNNs in the poincar\'e space (P-HGNNs), achieve performance superior to the state-of-the-art on environmental claim detection while using upto \textbf{30x fewer parameters}. We also demonstrate that HGNNs benefit vastly from explicitly modeling data in hierarchical (tree-like) structures, enabling them to significantly improve over their euclidean counterparts.
>
---
#### [replaced 057] Adaptive Monitoring and Real-World Evaluation of Agentic AI Systems
- **分类: cs.AI; cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2509.00115v3](http://arxiv.org/pdf/2509.00115v3)**

> **作者:** Manish Shukla
>
> **摘要:** Agentic artificial intelligence (AI) -- multi-agent systems that combine large language models with external tools and autonomous planning -- are rapidly transitioning from research laboratories into high-stakes domains. Our earlier "Basic" paper introduced a five-axis framework and proposed preliminary metrics such as goal drift and harm reduction but did not provide an algorithmic instantiation or empirical evidence. This "Advanced" sequel fills that gap. First, we revisit recent benchmarks and industrial deployments to show that technical metrics still dominate evaluations: a systematic review of 84 papers from 2023--2025 found that 83% report capability metrics while only 30% consider human-centred or economic axes [2]. Second, we formalise an Adaptive Multi-Dimensional Monitoring (AMDM) algorithm that normalises heterogeneous metrics, applies per-axis exponentially weighted moving-average thresholds and performs joint anomaly detection via the Mahalanobis distance [7]. Third, we conduct simulations and real-world experiments. AMDM cuts anomaly-detection latency from 12.3 s to 5.6 s on simulated goal drift and reduces false-positive rates from 4.5% to 0.9% compared with static thresholds. We present a comparison table and ROC/PR curves, and we reanalyse case studies to surface missing metrics. Code, data and a reproducibility checklist accompany this paper to facilitate replication. The code supporting this work is available at https://github.com/Manishms18/Adaptive-Multi-Dimensional-Monitoring.
>
---
#### [replaced 058] AraHealthQA 2025: The First Shared Task on Arabic Health Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.20047v3](http://arxiv.org/pdf/2508.20047v3)**

> **作者:** Hassan Alhuzali; Walid Al-Eisawi; Muhammad Abdul-Mageed; Chaimae Abouzahir; Mouath Abu-Daoud; Ashwag Alasmari; Renad Al-Monef; Ali Alqahtani; Lama Ayash; Leen Kharouf; Farah E. Shamout; Nizar Habash
>
> **备注:** ArabicNLP2025-colocated with EMNLP2025
>
> **摘要:** We introduce AraHealthQA 2025, the Comprehensive Arabic Health Question Answering Shared Task, held in conjunction with ArabicNLP 2025 (co-located with EMNLP 2025). This shared task addresses the paucity of high-quality Arabic medical QA resources by offering two complementary tracks: MentalQA, focusing on Arabic mental health Q&A (e.g., anxiety, depression, stigma reduction), and MedArabiQ, covering broader medical domains such as internal medicine, pediatrics, and clinical decision making. Each track comprises multiple subtasks, evaluation datasets, and standardized metrics, facilitating fair benchmarking. The task was structured to promote modeling under realistic, multilingual, and culturally nuanced healthcare contexts. We outline the dataset creation, task design and evaluation framework, participation statistics, baseline systems, and summarize the overall outcomes. We conclude with reflections on the performance trends observed and prospects for future iterations in Arabic health QA.
>
---
#### [replaced 059] DSMoE: Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12455v3](http://arxiv.org/pdf/2502.12455v3)**

> **作者:** Minxuan Lv; Zhenpeng Su; Leiyu Pan; Yizhe Xiong; Zijia Lin; Hui Chen; Wei Zhou; Jungong Han; Guiguang Ding; Cheng Luo; Di Zhang; Kun Gai; Songlin Hu
>
> **备注:** Accepted by EMNLP main conference
>
> **摘要:** As large language models continue to scale, computational costs and resource consumption have emerged as significant challenges. While existing sparsification methods like pruning reduce computational overhead, they risk losing model knowledge through parameter removal. This paper proposes DSMoE (Dynamic Sparse Mixture-of-Experts), a novel approach that achieves sparsification by partitioning pre-trained FFN layers into computational blocks. We implement adaptive expert routing using sigmoid activation and straight-through estimators, enabling tokens to flexibly access different aspects of model knowledge based on input complexity. Additionally, we introduce a sparsity loss term to balance performance and computational efficiency. Extensive experiments on LLaMA models demonstrate that under equivalent computational constraints, DSMoE achieves superior performance compared to existing pruning and MoE approaches across language modeling and downstream tasks, particularly excelling in generation tasks. Analysis reveals that DSMoE learns distinctive layerwise activation patterns, providing new insights for future MoE architecture design.
>
---
#### [replaced 060] A Survey on Large Language Model-based Agents for Statistics and Data Science
- **分类: cs.AI; cs.CL; cs.LG; stat.OT**

- **链接: [http://arxiv.org/pdf/2412.14222v2](http://arxiv.org/pdf/2412.14222v2)**

> **作者:** Maojun Sun; Ruijian Han; Binyan Jiang; Houduo Qi; Defeng Sun; Yancheng Yuan; Jian Huang
>
> **摘要:** In recent years, data science agents powered by Large Language Models (LLMs), known as "data agents," have shown significant potential to transform the traditional data analysis paradigm. This survey provides an overview of the evolution, capabilities, and applications of LLM-based data agents, highlighting their role in simplifying complex data tasks and lowering the entry barrier for users without related expertise. We explore current trends in the design of LLM-based frameworks, detailing essential features such as planning, reasoning, reflection, multi-agent collaboration, user interface, knowledge integration, and system design, which enable agents to address data-centric problems with minimal human intervention. Furthermore, we analyze several case studies to demonstrate the practical applications of various data agents in real-world scenarios. Finally, we identify key challenges and propose future research directions to advance the development of data agents into intelligent statistical analysis software.
>
---
#### [replaced 061] Evaluating and Aligning Human Economic Risk Preferences in LLMs
- **分类: econ.GN; cs.CL; q-fin.EC**

- **链接: [http://arxiv.org/pdf/2503.06646v2](http://arxiv.org/pdf/2503.06646v2)**

> **作者:** Jiaxin Liu; Yixuan Tang; Yi Yang; Kar Yan Tam
>
> **摘要:** Large Language Models (LLMs) are increasingly used in decision-making scenarios that involve risk assessment, yet their alignment with human economic rationality remains unclear. In this study, we investigate whether LLMs exhibit risk preferences consistent with human expectations across different personas. Specifically, we assess whether LLM-generated responses reflect appropriate levels of risk aversion or risk-seeking behavior based on individual's persona. Our results reveal that while LLMs make reasonable decisions in simplified, personalized risk contexts, their performance declines in more complex economic decision-making tasks. To address this, we propose an alignment method designed to enhance LLM adherence to persona-specific risk preferences. Our approach improves the economic rationality of LLMs in risk-related applications, offering a step toward more human-aligned AI decision-making.
>
---
#### [replaced 062] LogicTree: Structured Proof Exploration for Coherent and Rigorous Logical Reasoning with Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.14089v2](http://arxiv.org/pdf/2504.14089v2)**

> **作者:** Kang He; Kaushik Roy
>
> **备注:** EMNLP 2025 Main Conference
>
> **摘要:** Large language models (LLMs) have achieved remarkable multi-step reasoning capabilities across various domains. However, LLMs still face distinct challenges in complex logical reasoning, as (1) proof-finding requires systematic exploration and the maintenance of logical coherence and (2) searching the right combination of premises at each reasoning step is inherently challenging in tasks with large premise space. To address this, we propose LogicTree, an inference-time modular framework employing algorithm-guided search to automate structured proof exploration and ensure logical coherence. Advancing beyond tree-of-thought (ToT), we incorporate caching mechanism into LogicTree to enable effective utilization of historical knowledge, preventing reasoning stagnation and minimizing redundancy. Furthermore, we address the combinatorial complexity of premise search by decomposing it into a linear process. The refined premise selection restricts subsequent inference to at most one derivation per step, enhancing reasoning granularity and enforcing strict step-by-step reasoning. Additionally, we introduce two LLM-free heuristics for premise prioritization, enabling strategic proof search. Experimental results on five datasets demonstrate that LogicTree optimally scales inference-time computation to achieve higher proof accuracy, surpassing chain-of-thought (CoT) and ToT with average gains of 23.6% and 12.5%, respectively, on GPT-4o. Moreover, within LogicTree, GPT-4o outperforms o3-mini by 7.6% on average.
>
---
#### [replaced 063] Oyster-I: Beyond Refusal -- Constructive Safety Alignment for Responsible Language Models
- **分类: cs.AI; cs.CL; cs.CY; cs.HC; cs.SC**

- **链接: [http://arxiv.org/pdf/2509.01909v5](http://arxiv.org/pdf/2509.01909v5)**

> **作者:** Ranjie Duan; Jiexi Liu; Xiaojun Jia; Shiji Zhao; Ruoxi Cheng; Fengxiang Wang; Cheng Wei; Yong Xie; Chang Liu; Defeng Li; Yinpeng Dong; Yichi Zhang; Yuefeng Chen; Chongwen Wang; Xingjun Ma; Xingxing Wei; Yang Liu; Hang Su; Jun Zhu; Xinfeng Li; Yitong Sun; Jie Zhang; Jinzhao Hu; Sha Xu; Wenchao Yang; Yitong Yang; Jialing Tao; Hui Xue
>
> **备注:** Technical Report Code & Model weights available: https://github.com/Alibaba-AAIG/Oyster
>
> **摘要:** Large language models (LLMs) typically deploy safety mechanisms to prevent harmful content generation. Most current approaches focus narrowly on risks posed by malicious actors, often framing risks as adversarial events and relying on defensive refusals. However, in real-world settings, risks also come from non-malicious users seeking help while under psychological distress (e.g., self-harm intentions). In such cases, the model's response can strongly influence the user's next actions. Simple refusals may lead them to repeat, escalate, or move to unsafe platforms, creating worse outcomes. We introduce Constructive Safety Alignment (CSA), a human-centric paradigm that protects against malicious misuse while actively guiding vulnerable users toward safe and helpful results. Implemented in Oyster-I (Oy1), CSA combines game-theoretic anticipation of user reactions, fine-grained risk boundary discovery, and interpretable reasoning control, turning safety into a trust-building process. Oy1 achieves state-of-the-art safety among open models while retaining high general capabilities. On our Constructive Benchmark, it shows strong constructive engagement, close to GPT-5, and unmatched robustness on the Strata-Sword jailbreak dataset, nearing GPT-o1 levels. By shifting from refusal-first to guidance-first safety, CSA redefines the model-user relationship, aiming for systems that are not just safe, but meaningfully helpful. We release Oy1, code, and the benchmark to support responsible, user-centered AI.
>
---
#### [replaced 064] Active Layer-Contrastive Decoding Reduces Hallucination in Large Language Model Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23657v3](http://arxiv.org/pdf/2505.23657v3)**

> **作者:** Hongxiang Zhang; Hao Chen; Muhao Chen; Tianyi Zhang
>
> **备注:** 19 pages, 3 figures, EMNLP 2025
>
> **摘要:** Recent decoding methods improve the factuality of large language models (LLMs) by refining how the next token is selected during generation. These methods typically operate at the token level, leveraging internal representations to suppress superficial patterns. Nevertheless, LLMs remain prone to hallucinations, especially over longer contexts. In this paper, we propose Active Layer-Contrastive Decoding (ActLCD), a novel decoding strategy that actively decides when to apply contrasting layers during generation. By casting decoding as a sequential decision-making problem, ActLCD employs a reinforcement learning policy guided by a reward-aware classifier to optimize factuality beyond the token level. Our experiments demonstrate that ActLCD surpasses state-of-the-art methods across five benchmarks, showcasing its effectiveness in mitigating hallucinations in diverse generation scenarios.
>
---
#### [replaced 065] Mirage of Mastery: Memorization Tricks LLMs into Artificially Inflated Self-Knowledge
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.18998v2](http://arxiv.org/pdf/2506.18998v2)**

> **作者:** Sahil Kale; Vijaykant Nadadur
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** When artificial intelligence mistakes memorization for intelligence, it creates a dangerous mirage of reasoning. Existing studies treat memorization and self-knowledge deficits in LLMs as separate issues and do not recognize an intertwining link that degrades the trustworthiness of LLM responses. In our study, we utilize a novel framework to ascertain if LLMs genuinely learn reasoning patterns from training data or merely memorize them to assume competence across problems of similar complexity focused on STEM domains. Our analysis shows a noteworthy problem in generalization: LLMs draw confidence from memorized solutions to infer a higher self-knowledge about their reasoning ability, which manifests as an over 45% inconsistency in feasibility assessments when faced with self-validated, logically coherent task perturbations. This effect is most pronounced in science and medicine domains, which tend to have maximal standardized jargon and problems, further confirming our approach. Significant wavering within the self-knowledge of LLMs also shows flaws in current architectures and training patterns, highlighting the need for techniques that ensure a balanced, consistent stance on models' perceptions of their own knowledge for maximum AI explainability and trustworthiness. Our code and results are available publicly at https://github.com/knowledge-verse-ai/LLM-Memorization_SK_Eval-.
>
---
#### [replaced 066] Less Is More? Examining Fairness in Pruned Large Language Models for Summarising Opinions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.17610v3](http://arxiv.org/pdf/2508.17610v3)**

> **作者:** Nannan Huang; Haytham M. Fayek; Xiuzhen Zhang
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Model compression through post-training pruning offers a way to reduce model size and computational requirements without significantly impacting model performance. However, the effect of pruning on the fairness of LLM-generated summaries remains unexplored, particularly for opinion summarisation where biased outputs could influence public views.In this paper, we present a comprehensive empirical analysis of opinion summarisation, examining three state-of-the-art pruning methods and various calibration sets across three open-source LLMs using four fairness metrics. Our systematic analysis reveals that pruning methods have a greater impact on fairness than calibration sets. Building on these insights, we propose High Gradient Low Activation (HGLA) pruning, which identifies and removes parameters that are redundant for input processing but influential in output generation. Our experiments demonstrate that HGLA can better maintain or even improve fairness compared to existing methods, showing promise across models and tasks where traditional methods have limitations. Our human evaluation shows HGLA-generated outputs are fairer than existing state-of-the-art pruning methods. Code is available at: https://github.com/amberhuang01/HGLA.
>
---
#### [replaced 067] Reducing Object Hallucination in Large Audio-Language Models via Audio-Aware Decoding
- **分类: eess.AS; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07233v2](http://arxiv.org/pdf/2506.07233v2)**

> **作者:** Tzu-wen Hsu; Ke-Han Lu; Cheng-Han Chiang; Hung-yi Lee
>
> **摘要:** Large Audio-Language Models (LALMs) can take audio and text as the inputs and answer questions about the audio. While prior LALMs have shown strong performance on standard benchmarks, there has been alarming evidence that LALMs can hallucinate what is presented in the audio. To mitigate the hallucination of LALMs, we introduce Audio-Aware Decoding (AAD), a lightweight inference-time strategy that uses contrastive decoding to compare the token prediction logits with and without the audio context. By contrastive decoding, AAD promotes the tokens whose probability increases when the audio is present. We conduct our experiment on object hallucination datasets with three LALMs and show that AAD improves the F1 score by 0.046 to 0.428. We also show that AAD can improve the accuracy on general audio QA datasets like Clotho-AQA by 5.4% to 10.3%. We conduct thorough ablation studies to understand the effectiveness of each component in AAD.
>
---
#### [replaced 068] Can Advanced LLMs Coach Smaller LLMs? Knowledge Distillation for Goal-Oriented Dialogs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.07238v2](http://arxiv.org/pdf/2408.07238v2)**

> **作者:** Tong Wang; K. Sudhir; Dat Hong
>
> **摘要:** Enterprises deploying LLMs for goal-oriented dialogs, such as customer service, face a critical trade-off between performance, control, and cost. Proprietary models like GPT-4 offer strong performance but are costly and cannot be self-hosted, raising security and privacy concerns. Open-source alternatives offer flexibility and lower token costs but lag in performance. We introduce Guidance Elicitation and Retrieval (GER), a prompt-based knowledge distillation framework where a high-performance teacher LLM coaches a lower-performance student without modifying the student's parameters. GER extracts tactical guidance for a wide range of dialog scenarios from the teacher and stores these scenario-guidance pairs in a structured library. At inference time, the student retrieves the relevant guidance and integrates it into its prompt. While GER training can be bootstrapped entirely with synthetic data, its modular design lets it seamlessly augment the synthetic data with human conversational logs. In addition, the modular design enables easy auditing and updating of the guidance library as new scenarios and constraints emerge. Experiments show GER's guidance-based coaching outperforms both example output based fine-tuning and non-customized guidance baselines, and generalizes across other contexts and student models. The GER framework is potentially extensible to coach human service agents.
>
---
#### [replaced 069] Is In-Context Learning Learning?
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.10414v2](http://arxiv.org/pdf/2509.10414v2)**

> **作者:** Adrian de Wynter
>
> **备注:** Director's cut
>
> **摘要:** In-context learning (ICL) allows some autoregressive models to solve tasks via next-token prediction and without needing further training. This has led to claims about these model's ability to solve (learn) unseen tasks with only a few shots (exemplars) in the prompt. However, deduction does not always imply learning, as ICL does not explicitly encode a given observation. Instead, the models rely on their prior knowledge and the exemplars given, if any. We argue that, mathematically, ICL does constitute learning, but its full characterisation requires empirical work. We then carry out a large-scale analysis of ICL ablating out or accounting for memorisation, pretraining, distributional shifts, and prompting style and phrasing. We find that ICL is an effective learning paradigm, but limited in its ability to learn and generalise to unseen tasks. We note that, in the limit where exemplars become more numerous, accuracy is insensitive to exemplar distribution, model, prompt style, and the input's linguistic features. Instead, it deduces patterns from regularities in the prompt, which leads to distributional sensitivity, especially in prompting styles such as chain-of-thought. Given the varied accuracies on formally similar tasks, we conclude that autoregression's ad-hoc encoding is not a robust mechanism, and suggests limited all-purpose generalisability.
>
---
#### [replaced 070] FM2DS: Few-Shot Multimodal Multihop Data Synthesis with Knowledge Distillation for Question Answering
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.07030v5](http://arxiv.org/pdf/2412.07030v5)**

> **作者:** Amirhossein Abaskohi; Spandana Gella; Giuseppe Carenini; Issam H. Laradji
>
> **备注:** Findings of EMNLP 2025
>
> **摘要:** Multimodal multihop question answering (MMQA) requires reasoning over images and text from multiple sources. Despite advances in visual question answering, this multihop setting remains underexplored due to a lack of quality datasets. Existing methods focus on single-hop, single-modality, or short texts, limiting real-world applications like interpreting educational documents with long, multimodal content. To fill this gap, we introduce FM2DS, the first framework for creating a high-quality dataset for MMQA. Our approach consists of a 5-stage pipeline that involves acquiring relevant multimodal documents from Wikipedia, synthetically generating high-level questions and answers, and validating them through rigorous criteria to ensure data quality. We evaluate our methodology by training models on our synthesized dataset and testing on two benchmarks: MultimodalQA and WebQA. Our results demonstrate that, with an equal sample size, models trained on our synthesized data outperform those trained on human-collected data by 1.9 in exact match (EM) score on average. Additionally, we introduce M2QA-Bench with 1k samples, the first benchmark for MMQA on long documents, generated using FM2DS and refined by human annotators. We believe our data synthesis method will serve as a strong foundation for training and evaluating MMQA models.
>
---
#### [replaced 071] What fifty-one years of Linguistics and Artificial Intelligence research tell us about their correlation: A scientometric analysis
- **分类: cs.CL; cs-CL; F.2.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2411.19858v3](http://arxiv.org/pdf/2411.19858v3)**

> **作者:** Mohammed Q. Shormani
>
> **备注:** 26 pages, 15 figures
>
> **摘要:** There is a strong correlation between linguistics and artificial intelligence (AI), best manifested by deep learning language models. This study provides a thorough scientometric analysis of this correlation, synthesizing the intellectual production over 51 years, from 1974 to 2024. Web of Science Core Collection (WoSCC) database was the data source. The data collected were analyzed by two powerful software, viz., CiteSpace and VOSviewer, through which mapping visualizations of the intellectual landscape, trending issues and (re)emerging hotspots were generated. The results indicate that in the 1980s and 1990s, linguistics and AI (AIL) research was not robust, characterized by unstable publication over time. It has, however, witnessed a remarkable increase of publication since then, reaching 1478 articles in 2023, and 546 articles in January-March timespan in 2024, involving emerging issues including Natural language processing, Cross-sectional study, Using bidirectional encoder representation, and Using ChatGPT and hotspots such as Novice programmer, Prioritization, and Artificial intelligence, addressing new horizons, new topics, and launching new applications and powerful deep learning language models including ChatGPT. It concludes that linguistics and AI correlation is established at several levels, research centers, journals, and countries shaping AIL knowledge production and reshaping its future frontiers.
>
---
#### [replaced 072] Evaluating Automatic Speech Recognition Systems for Korean Meteorological Experts
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.18444v3](http://arxiv.org/pdf/2410.18444v3)**

> **作者:** ChaeHun Park; Hojun Cho; Jaegul Choo
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** This paper explores integrating Automatic Speech Recognition (ASR) into natural language query systems to improve weather forecasting efficiency for Korean meteorologists. We address challenges in developing ASR systems for the Korean weather domain, specifically specialized vocabulary and Korean linguistic intricacies. To tackle these issues, we constructed an evaluation dataset of spoken queries recorded by native Korean speakers. Using this dataset, we assessed various configurations of a multilingual ASR model family, identifying performance limitations related to domain-specific terminology. We then implemented a simple text-to-speech-based data augmentation method, which improved the recognition of specialized terms while maintaining general-domain performance. Our contributions include creating a domain-specific dataset, comprehensive ASR model evaluations, and an effective augmentation technique. We believe our work provides a foundation for future advancements in ASR for the Korean weather forecasting domain.
>
---
#### [replaced 073] Lean Formalization of Generalization Error Bound by Rademacher Complexity
- **分类: cs.LG; cs.CL; math.ST; stat.TH**

- **链接: [http://arxiv.org/pdf/2503.19605v3](http://arxiv.org/pdf/2503.19605v3)**

> **作者:** Sho Sonoda; Kazumi Kasaura; Yuma Mizuno; Kei Tsukamoto; Naoto Onda
>
> **备注:** major updated
>
> **摘要:** We formalize the generalization error bound using the Rademacher complexity for the Lean 4 theorem prover based on the probability theory in the Mathlib 4 library. Generalization error quantifies the gap between a learning machine's performance on given training data versus unseen test data, and the Rademacher complexity is a powerful tool to upper-bound the generalization error of a variety of modern learning problems. Previous studies have only formalized extremely simple cases such as bounds by parameter counts and analyses for very simple models (decision stumps). Formalizing the Rademacher complexity bound, also known as the uniform law of large numbers, requires substantial development and is achieved for the first time in this study. In the course of development, we formalize the Rademacher complexity and its unique arguments such as symmetrization, and clarify the topological assumptions on hypothesis classes under which the bound holds. As an application, we also present the formalization of generalization error bound for $L^2$-regularization models.
>
---
#### [replaced 074] Recycling the Web: A Method to Enhance Pre-training Data Quality and Quantity for Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.04689v3](http://arxiv.org/pdf/2506.04689v3)**

> **作者:** Thao Nguyen; Yang Li; Olga Golovneva; Luke Zettlemoyer; Sewoong Oh; Ludwig Schmidt; Xian Li
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Scaling laws predict that the performance of large language models improves with increasing model size and data size. In practice, pre-training has been relying on massive web crawls, using almost all data sources publicly available on the internet so far. However, this pool of natural data does not grow at the same rate as the compute supply. Furthermore, the availability of high-quality texts is even more limited: data filtering pipelines often remove up to 99% of the initial web scrapes to achieve state-of-the-art. To address the "data wall" of pre-training scaling, our work explores ways to transform and recycle data discarded in existing filtering processes. We propose REWIRE, REcycling the Web with guIded REwrite, a method to enrich low-quality documents so that they could become useful for training. This in turn allows us to increase the representation of synthetic data in the final pre-training set. Experiments at 1B, 3B and 7B scales of the DCLM benchmark show that mixing high-quality raw texts and our rewritten texts lead to 1.0, 1.3 and 2.5 percentage points improvement respectively across 22 diverse tasks, compared to training on only filtered web data. Training on the raw-synthetic data mix is also more effective than having access to 2x web data. Through further analysis, we demonstrate that about 82% of the mixed in texts come from transforming lower-quality documents that would otherwise be discarded. REWIRE also outperforms related approaches of generating synthetic data, including Wikipedia-style paraphrasing, question-answer synthesizing and knowledge extraction. These results suggest that recycling web texts holds the potential for being a simple and effective approach for scaling pre-training data. We make our high-quality synthetic data publicly available at https://huggingface.co/datasets/facebook/recycling_the_web.
>
---
#### [replaced 075] Are Generative Models Underconfident? Better Quality Estimation with Boosted Model Probability
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.11115v4](http://arxiv.org/pdf/2502.11115v4)**

> **作者:** Tu Anh Dinh; Jan Niehues
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Quality Estimation (QE) is estimating quality of the model output during inference when the ground truth is not available. Deriving output quality from the models' output probability is the most trivial and low-effort way. However, we show that the output probability of text-generation models can appear underconfident. At each output step, there can be multiple correct options, making the probability distribution spread out more. Thus, lower probability does not necessarily mean lower output quality. Due to this observation, we propose a QE approach called BoostedProb, which boosts the model's confidence in cases where there are multiple viable output options. With no increase in complexity, BoostedProb is notably better than raw model probability in different settings, achieving on average +0.194 improvement in Pearson correlation to ground-truth quality. It also comes close to or outperforms more costly approaches like supervised or ensemble-based QE in certain settings.
>
---
#### [replaced 076] Transplant Then Regenerate: A New Paradigm for Text Data Augmentation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.14723v3](http://arxiv.org/pdf/2508.14723v3)**

> **作者:** Guangzhan Wang; Hongyu Zhang; Beijun Shen; Xiaodong Gu
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Data augmentation is a critical technique in deep learning. Traditional methods like Back-translation typically focus on lexical-level rephrasing, which primarily produces variations with the same semantics. While large language models (LLMs) have enhanced text augmentation by their "knowledge emergence" capability, controlling the style and structure of these outputs remains challenging and requires meticulous prompt engineering. In this paper, we propose LMTransplant, a novel text augmentation paradigm leveraging LLMs. The core idea of LMTransplant is transplant-then-regenerate: incorporating seed text into a context expanded by LLM, and asking the LLM to regenerate a variant based on the expanded context. This strategy allows the model to create more diverse and creative content-level variants by fully leveraging the knowledge embedded in LLMs, while preserving the core attributes of the original text. We evaluate LMTransplant across various text-related tasks, demonstrating its superior performance over existing text augmentation methods. Moreover, LMTransplant demonstrates exceptional scalability as the size of augmented data grows.
>
---
#### [replaced 077] LastingBench: Defend Benchmarks Against Knowledge Leakage
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.21614v2](http://arxiv.org/pdf/2506.21614v2)**

> **作者:** Yixiong Fang; Tianran Sun; Yuling Shi; Min Wang; Xiaodong Gu
>
> **摘要:** The increasing complexity of large language models (LLMs) raises concerns about their ability to "cheat" on standard Question Answering (QA) benchmarks by memorizing task-specific data. This undermines the validity of benchmark evaluations, as they no longer reflect genuine model capabilities but instead the effects of data leakage. While prior work has focused on detecting such leakage, little attention has been given to mitigating its impact and preserving the long-term utility of benchmarks. In this paper, we introduce LastingBench, a novel framework designed to continuously reinforce and safeguard existing benchmarks against knowledge leakage. LastingBench identifies leakage points in the context through perturbation, then rewrites the leakage points to counterfactual ones-disrupting memorization while preserving the benchmark's original evaluative intent. Evaluations of state-of-the-art QA benchmarks show significant performance gaps, highlighting the efficacy of LastingBench in reducing memorization effects. LastingBench offers a practical and scalable solution to ensure benchmark robustness over time, promoting fairer and more interpretable evaluations of LLMs.
>
---
#### [replaced 078] Artificial intelligence contribution to translation industry: looking back and forward
- **分类: cs.CL; cs-CL; F.2.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2411.19855v3](http://arxiv.org/pdf/2411.19855v3)**

> **作者:** Mohammed Q. Shormani
>
> **备注:** 30 pages, 13 figures
>
> **摘要:** This study provides a comprehensive analysis of artificial intelligence (AI) contribution to research in the translation industry (ACTI), synthesizing it over forty-five years from 1980-2024. 13220 articles were retrieved from three sources, namely WoS, Scopus, and Lens; 9836 were unique records, which were used for the analysis. I provided two types of analysis, viz., scientometric and thematic, focusing on Cluster, Subject categories, Keywords, Bursts, Centrality and Research Centers as for the former. For the latter, I provided a thematic review for 18 articles, selected purposefully from the articles involved, centering on purpose, approach, findings, and contribution to ACTI future directions. This study is significant for its valuable contribution to ACTI knowledge production over 45 years, emphasizing several trending issues and hotspots including Machine translation, Statistical machine translation, Low-resource language, Large language model, Arabic dialects, Translation quality, and Neural machine translation. The findings reveal that the more AI develops, the more it contributes to translation industry, as Neural Networking Algorithms have been incorporated and Deep Language Learning Models like ChatGPT have been launched. However, much rigorous research is still needed to overcome several problems encountering translation industry, specifically concerning low-resource, multi-dialectical and free word order languages, and cultural and religious registers.
>
---
#### [replaced 079] GmSLM : Generative Marmoset Spoken Language Modeling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.09198v2](http://arxiv.org/pdf/2509.09198v2)**

> **作者:** Talia Sternberg; Michael London; David Omer; Yossi Adi
>
> **摘要:** Marmoset monkeys exhibit complex vocal communication, challenging the view that nonhuman primates vocal communication is entirely innate, and show similar features of human speech, such as vocal labeling of others and turn-taking. Studying their vocal communication offers a unique opportunity to link it with brain activity-especially given the difficulty of accessing the human brain in speech and language research. Since Marmosets communicate primarily through vocalizations, applying standard LLM approaches is not straightforward. We introduce Generative Marmoset Spoken Language Modeling (GmSLM), an optimized spoken language model pipeline for Marmoset vocal communication. We designed a novel zero-shot evaluation metrics using unsupervised in-the-wild data, alongside weakly labeled conversational data, to assess GmSLM and demonstrate its advantage over a basic human-speech-based baseline. GmSLM generated vocalizations closely matched real resynthesized samples acoustically and performed well on downstream tasks. Despite being fully unsupervised, GmSLM effectively distinguish real from artificial conversations and may support further investigations of the neural basis of vocal communication and provides a practical framework linking vocalization and brain activity. We believe GmSLM stands to benefit future work in neuroscience, bioacoustics, and evolutionary biology. Samples are provided under: pages.cs.huji.ac.il/adiyoss-lab/GmSLM.
>
---
#### [replaced 080] Improving Informally Romanized Language Identification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.21540v3](http://arxiv.org/pdf/2504.21540v3)**

> **作者:** Adrian Benton; Alexander Gutkin; Christo Kirov; Brian Roark
>
> **备注:** 19 pages, 16 tables, 4 figures
>
> **摘要:** The Latin script is often used to informally write languages with non-Latin native scripts. In many cases (e.g., most languages in India), the lack of conventional spelling in the Latin script results in high spelling variability. Such romanization renders languages that are normally easily distinguished due to being written in different scripts - Hindi and Urdu, for example - highly confusable. In this work, we increase language identification (LID) accuracy for romanized text by improving the methods used to synthesize training sets. We find that training on synthetic samples which incorporate natural spelling variation yields higher LID system accuracy than including available naturally occurring examples in the training set, or even training higher capacity models. We demonstrate new state-of-the-art LID performance on romanized text from 20 Indic languages in the Bhasha-Abhijnaanam evaluation set (Madhani et al., 2023a), improving test F1 from the reported 74.7% (using a pretrained neural model) to 85.4% using a linear classifier trained solely on synthetic data and 88.2% when also training on available harvested text.
>
---
#### [replaced 081] One Goal, Many Challenges: Robust Preference Optimization Amid Content-Aware and Multi-Source Noise
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.12301v2](http://arxiv.org/pdf/2503.12301v2)**

> **作者:** Amirabbas Afzali; Amirhossein Afsharrad; Seyed Shahabeddin Mousavi; Sanjay Lall
>
> **摘要:** Large Language Models (LLMs) have made significant strides in generating human-like responses, largely due to preference alignment techniques. However, these methods often assume unbiased human feedback, which is rarely the case in real-world scenarios. This paper introduces Content-Aware Noise-Resilient Preference Optimization (CNRPO), a novel framework that addresses multiple sources of content-dependent noise in preference learning. CNRPO employs a multi-objective optimization approach to separate true preferences from content-aware noises, effectively mitigating their impact. We leverage backdoor attack mechanisms to efficiently learn and control various noise sources within a single model. Theoretical analysis and extensive experiments on different synthetic noisy datasets demonstrate that CNRPO significantly improves alignment with primary human preferences while controlling for secondary noises and biases, such as response length and harmfulness.
>
---
