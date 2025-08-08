# 自然语言处理 cs.CL

- **最新发布 71 篇**

- **更新 63 篇**

## 最新发布

#### [new 001] RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文研究多智能体LLM系统中的上下文路由问题，提出RCR-Router框架实现动态角色感知的高效结构化记忆管理，通过轻量级评分机制优化输出整合与答案质量评估，有效降低令牌消耗并提升协作性能。**

- **链接: [http://arxiv.org/pdf/2508.04903v1](http://arxiv.org/pdf/2508.04903v1)**

> **作者:** Jun Liu; Zhenglun Kong; Changdi Yang; Fan Yang; Tianqi Li; Peiyan Dong; Joannah Nanjekye; Hao Tang; Geng Yuan; Wei Niu; Wenbin Zhang; Pu Zhao; Xue Lin; Dong Huang; Yanzhi Wang
>
> **摘要:** Multi-agent large language model (LLM) systems have shown strong potential in complex reasoning and collaborative decision-making tasks. However, most existing coordination schemes rely on static or full-context routing strategies, which lead to excessive token consumption, redundant memory exposure, and limited adaptability across interaction rounds. We introduce RCR-Router, a modular and role-aware context routing framework designed to enable efficient, adaptive collaboration in multi-agent LLMs. To our knowledge, this is the first routing approach that dynamically selects semantically relevant memory subsets for each agent based on its role and task stage, while adhering to a strict token budget. A lightweight scoring policy guides memory selection, and agent outputs are iteratively integrated into a shared memory store to facilitate progressive context refinement. To better evaluate model behavior, we further propose an Answer Quality Score metric that captures LLM-generated explanations beyond standard QA accuracy. Experiments on three multi-hop QA benchmarks -- HotPotQA, MuSiQue, and 2WikiMultihop -- demonstrate that RCR-Router reduces token usage (up to 30%) while improving or maintaining answer quality. These results highlight the importance of structured memory routing and output-aware evaluation in advancing scalable multi-agent LLM systems.
>
---
#### [new 002] Do Political Opinions Transfer Between Western Languages? An Analysis of Unaligned and Aligned Multilingual LLMs
- **分类: cs.CL; cs.CY; I.2.7; J.4**

- **简介: 该研究分析了跨语言政治观点在五种西方语言中的传递情况，验证了跨文化差异是否转化为跨语言模型的表达差异，通过对比对齐与未对齐LLM的表现，揭示了政治观点在不同语言间的统一性，解决了如何量化跨语言观点差异的问题。**

- **链接: [http://arxiv.org/pdf/2508.05553v1](http://arxiv.org/pdf/2508.05553v1)**

> **作者:** Franziska Weeber; Tanise Ceron; Sebastian Padó
>
> **摘要:** Public opinion surveys show cross-cultural differences in political opinions between socio-cultural contexts. However, there is no clear evidence whether these differences translate to cross-lingual differences in multilingual large language models (MLLMs). We analyze whether opinions transfer between languages or whether there are separate opinions for each language in MLLMs of various sizes across five Western languages. We evaluate MLLMs' opinions by prompting them to report their (dis)agreement with political statements from voting advice applications. To better understand the interaction between languages in the models, we evaluate them both before and after aligning them with more left or right views using direct preference optimization and English alignment data only. Our findings reveal that unaligned models show only very few significant cross-lingual differences in the political opinions they reflect. The political alignment shifts opinions almost uniformly across all five languages. We conclude that in Western language contexts, political opinions transfer between languages, demonstrating the challenges in achieving explicit socio-linguistic, cultural, and political alignment of MLLMs.
>
---
#### [new 003] TASE: Token Awareness and Structured Evaluation for Multilingual Language Models
- **分类: cs.CL**

- **简介: 该论文提出TASE任务框架，旨在评估LLMs在token-awareness（token感知）和structural understanding（结构理解）方面的能力，解决当前LLMs在细粒度语言理解及结构推理中的局限性。通过设计跨语言基准测试和自动生成训练数据，开发了GRPO方法并验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2508.05468v1](http://arxiv.org/pdf/2508.05468v1)**

> **作者:** Chenzhuo Zhao; Xinda Wang; Yue Huang; Junting Lu; Ziqian Liu
>
> **摘要:** While large language models (LLMs) have demonstrated remarkable performance on high-level semantic tasks, they often struggle with fine-grained, token-level understanding and structural reasoning--capabilities that are essential for applications requiring precision and control. We introduce TASE, a comprehensive benchmark designed to evaluate LLMs' ability to perceive and reason about token-level information across languages. TASE covers 10 tasks under two core categories: token awareness and structural understanding, spanning Chinese, English, and Korean, with a 35,927-instance evaluation set and a scalable synthetic data generation pipeline for training. Tasks include character counting, token alignment, syntactic structure parsing, and length constraint satisfaction. We evaluate over 30 leading commercial and open-source LLMs, including O3, Claude 4, Gemini 2.5 Pro, and DeepSeek-R1, and train a custom Qwen2.5-14B model using the GRPO training method. Results show that human performance significantly outpaces current LLMs, revealing persistent weaknesses in token-level reasoning. TASE sheds light on these limitations and provides a new diagnostic lens for future improvements in low-level language understanding and cross-lingual generalization. Our code and dataset are publicly available at https://github.com/cyzcz/Tase .
>
---
#### [new 004] Towards Robust Evaluation of Visual Activity Recognition: Resolving Verb Ambiguity with Sense Clustering
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文旨在解决视觉活动识别中的语义歧义与视角差异问题，提出通过感觉聚类框架构建多视角识别方案，并在imSitu数据集上验证其有效性，相较于传统方法更准确地评估模型表现。**

- **链接: [http://arxiv.org/pdf/2508.04945v1](http://arxiv.org/pdf/2508.04945v1)**

> **作者:** Louie Hong Yao; Nicholas Jarvis; Tianyu Jiang
>
> **备注:** 18 pages, 5 figures
>
> **摘要:** Evaluating visual activity recognition systems is challenging due to inherent ambiguities in verb semantics and image interpretation. When describing actions in images, synonymous verbs can refer to the same event (e.g., brushing vs. grooming), while different perspectives can lead to equally valid but distinct verb choices (e.g., piloting vs. operating). Standard exact-match evaluation, which relies on a single gold answer, fails to capture these ambiguities, resulting in an incomplete assessment of model performance. To address this, we propose a vision-language clustering framework that constructs verb sense clusters, providing a more robust evaluation. Our analysis of the imSitu dataset shows that each image maps to an average of 2.8 sense clusters, with each cluster representing a distinct perspective of the image. We evaluate multiple activity recognition models and compare our cluster-based evaluation with standard evaluation methods. Additionally, our human alignment analysis suggests that the cluster-based evaluation better aligns with human judgements, offering a more nuanced assessment of model performance.
>
---
#### [new 005] Evaluation of a Sign Language Avatar on Comprehensibility, User Experience \& Acceptability
- **分类: cs.CL; cs.HC**

- **简介: 该论文旨在评估调整功能对SL avatar的可理解性、用户体验与接受度的影响，研究通过Hololens 2设备测试发现调整功能未显著提升性能，需补充语音/表情元素并优化交互界面以增强用户满意度，强调默认可理解性的重要性。**

- **链接: [http://arxiv.org/pdf/2508.05358v1](http://arxiv.org/pdf/2508.05358v1)**

> **作者:** Fenya Wasserroth; Eleftherios Avramidis; Vera Czehmann; Tanja Kojic; Fabrizio Nunnari; Sebastian Möller
>
> **摘要:** This paper presents an investigation into the impact of adding adjustment features to an existing sign language (SL) avatar on a Microsoft Hololens 2 device. Through a detailed analysis of interactions of expert German Sign Language (DGS) users with both adjustable and non-adjustable avatars in a specific use case, this study identifies the key factors influencing the comprehensibility, the user experience (UX), and the acceptability of such a system. Despite user preference for adjustable settings, no significant improvements in UX or comprehensibility were observed, which remained at low levels, amid missing SL elements (mouthings and facial expressions) and implementation issues (indistinct hand shapes, lack of feedback and menu positioning). Hedonic quality was rated higher than pragmatic quality, indicating that users found the system more emotionally or aesthetically pleasing than functionally useful. Stress levels were higher for the adjustable avatar, reflecting lower performance, greater effort and more frustration. Additionally, concerns were raised about whether the Hololens adjustment gestures are intuitive and easy to familiarise oneself with. While acceptability of the concept of adjustability was generally positive, it was strongly dependent on usability and animation quality. This study highlights that personalisation alone is insufficient, and that SL avatars must be comprehensible by default. Key recommendations include enhancing mouthing and facial animation, improving interaction interfaces, and applying participatory design.
>
---
#### [new 006] Can Language Models Critique Themselves? Investigating Self-Feedback for Retrieval Augmented Generation at BioASQ 2025
- **分类: cs.CL**

- **简介: 该论文探讨了语言模型自我反馈机制在专业搜索任务中的应用效果，旨在解决用户参与度低与信息匹配偏差的问题，通过实验验证其有效性，并研究推理模型的反馈能力差异。**

- **链接: [http://arxiv.org/pdf/2508.05366v1](http://arxiv.org/pdf/2508.05366v1)**

> **作者:** Samy Ateia; Udo Kruschwitz
>
> **备注:** Version as accepted at the BioASQ Lab at CLEF 2025
>
> **摘要:** Agentic Retrieval Augmented Generation (RAG) and 'deep research' systems aim to enable autonomous search processes where Large Language Models (LLMs) iteratively refine outputs. However, applying these systems to domain-specific professional search, such as biomedical research, presents challenges, as automated systems may reduce user involvement and misalign with expert information needs. Professional search tasks often demand high levels of user expertise and transparency. The BioASQ CLEF 2025 challenge, using expert-formulated questions, can serve as a platform to study these issues. We explored the performance of current reasoning and nonreasoning LLMs like Gemini-Flash 2.0, o3-mini, o4-mini and DeepSeek-R1. A key aspect of our methodology was a self-feedback mechanism where LLMs generated, evaluated, and then refined their outputs for query expansion and for multiple answer types (yes/no, factoid, list, ideal). We investigated whether this iterative self-correction improves performance and if reasoning models are more capable of generating useful feedback. Preliminary results indicate varied performance for the self-feedback strategy across models and tasks. This work offers insights into LLM self-correction and informs future work on comparing the effectiveness of LLM-generated feedback with direct human expert input in these search systems.
>
---
#### [new 007] Parity-Aware Byte-Pair Encoding: Improving Cross-lingual Fairness in Tokenization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文旨在解决跨语言token化不均问题，提出Parity-aware BPE算法优化BPE在合并步骤中平衡局部压缩与全球偏性，从而提升跨语言token均衡性，无显著影响下游任务效果。**

- **链接: [http://arxiv.org/pdf/2508.04796v1](http://arxiv.org/pdf/2508.04796v1)**

> **作者:** Negar Foroutan; Clara Meister; Debjit Paul; Joel Niklaus; Sina Ahmadi; Antoine Bosselut; Rico Sennrich
>
> **摘要:** Tokenization is the first -- and often least scrutinized -- step of most NLP pipelines. Standard algorithms for learning tokenizers rely on frequency-based objectives, which favor languages dominant in the training data and consequently leave lower-resource languages with tokenizations that are disproportionately longer, morphologically implausible, or even riddled with <UNK> placeholders. This phenomenon ultimately amplifies computational and financial inequalities between users from different language backgrounds. To remedy this, we introduce Parity-aware Byte Pair Encoding (BPE), a variant of the widely-used BPE algorithm. At every merge step, Parity-aware BPE maximizes the compression gain of the currently worst-compressed language, trading a small amount of global compression for cross-lingual parity. We find empirically that Parity-aware BPE leads to more equitable token counts across languages, with negligible impact on global compression rate and no substantial effect on language-model performance in downstream tasks.
>
---
#### [new 008] Persistent Instability in LLM's Personality Measurements: Effects of Scale, Reasoning, and Conversation History
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在探讨大型语言模型（LLMs）在人格测量中的持久不稳定现象，通过评估框架测试25+模型（1B-671B参数）及500k响应，系统分析了尺度、推理和对话历史等因素对行为一致性的影响。研究发现其存在显著变异（SD>0.4）、推理模式变化导致20%偏移，且LLM工具与人类工具一致，揭示了行为基础而非翻译能力的限制，提出需改进人格化策略以实现行为一致性。**

- **链接: [http://arxiv.org/pdf/2508.04826v1](http://arxiv.org/pdf/2508.04826v1)**

> **作者:** Tommaso Tosato; Saskia Helbling; Yorguin-Jose Mantilla-Ramos; Mahmood Hegazy; Alberto Tosato; David John Lemay; Irina Rish; Guillaume Dumas
>
> **摘要:** Large language models require consistent behavioral patterns for safe deployment, yet their personality-like traits remain poorly understood. We present PERSIST (PERsonality Stability in Synthetic Text), a comprehensive evaluation framework testing 25+ open-source models (1B-671B parameters) across 500,000+ responses. Using traditional (BFI-44, SD3) and novel LLM-adapted personality instruments, we systematically vary question order, paraphrasing, personas, and reasoning modes. Our findings challenge fundamental deployment assumptions: (1) Even 400B+ models exhibit substantial response variability (SD > 0.4); (2) Minor prompt reordering alone shifts personality measurements by up to 20%; (3) Interventions expected to stabilize behavior, such as chain-of-thought reasoning, detailed personas instruction, inclusion of conversation history, can paradoxically increase variability; (4) LLM-adapted instruments show equal instability to human-centric versions, confirming architectural rather than translational limitations. This persistent instability across scales and mitigation strategies suggests current LLMs lack the foundations for genuine behavioral consistency. For safety-critical applications requiring predictable behavior, these findings indicate that personality-based alignment strategies may be fundamentally inadequate.
>
---
#### [new 009] Towards Assessing Medical Ethics from Knowledge to Practice
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在评估大型语言模型在医疗伦理推理中的表现，提出PrinciplismQA基准，通过多选/开放题和专家验证，系统识别模型伦理知识与实践之间的差距，揭示临床决策中利益相关原则的不足，并探索医疗域微调提升伦理能力的路径。**

- **链接: [http://arxiv.org/pdf/2508.05132v1](http://arxiv.org/pdf/2508.05132v1)**

> **作者:** Chang Hong; Minghao Wu; Qingying Xiao; Yuchi Wang; Xiang Wan; Guangjun Yu; Benyou Wang; Yan Hu
>
> **摘要:** The integration of large language models into healthcare necessitates a rigorous evaluation of their ethical reasoning, an area current benchmarks often overlook. We introduce PrinciplismQA, a comprehensive benchmark with 3,648 questions designed to systematically assess LLMs' alignment with core medical ethics. Grounded in Principlism, our benchmark features a high-quality dataset. This includes multiple-choice questions curated from authoritative textbooks and open-ended questions sourced from authoritative medical ethics case study literature, all validated by medical experts. Our experiments reveal a significant gap between models' ethical knowledge and their practical application, especially in dynamically applying ethical principles to real-world scenarios. Most LLMs struggle with dilemmas concerning Beneficence, often over-emphasizing other principles. Frontier closed-source models, driven by strong general capabilities, currently lead the benchmark. Notably, medical domain fine-tuning can enhance models' overall ethical competence, but further progress requires better alignment with medical ethical knowledge. PrinciplismQA offers a scalable framework to diagnose these specific ethical weaknesses, paving the way for more balanced and responsible medical AI.
>
---
#### [new 010] Multimodal Fact Checking with Unified Visual, Textual, and Contextual Representations
- **分类: cs.CL**

- **简介: 该论文提出了一种基于多模态的统一框架"MultiCheck"，旨在解决传统文本验证方法难以应对多模态信息（文本+图像）的挑战，通过融合文本与图像编码器和对比学习优化跨模态关系，实现了在Factify 2数据集上的显著性能提升。**

- **链接: [http://arxiv.org/pdf/2508.05097v1](http://arxiv.org/pdf/2508.05097v1)**

> **作者:** Aditya Kishore; Gaurav Kumar; Jasabanta Patro
>
> **摘要:** The growing rate of multimodal misinformation, where claims are supported by both text and images, poses significant challenges to fact-checking systems that rely primarily on textual evidence. In this work, we have proposed a unified framework for fine-grained multimodal fact verification called "MultiCheck", designed to reason over structured textual and visual signals. Our architecture combines dedicated encoders for text and images with a fusion module that captures cross-modal relationships using element-wise interactions. A classification head then predicts the veracity of a claim, supported by a contrastive learning objective that encourages semantic alignment between claim-evidence pairs in a shared latent space. We evaluate our approach on the Factify 2 dataset, achieving a weighted F1 score of 0.84, substantially outperforming the baseline. These results highlight the effectiveness of explicit multimodal reasoning and demonstrate the potential of our approach for scalable and interpretable fact-checking in complex, real-world scenarios.
>
---
#### [new 011] ASCoT: An Adaptive Self-Correction Chain-of-Thought Method for Late-Stage Fragility in LLMs
- **分类: cs.CL**

- **简介: 该论文旨在解决LLM推理过程中的晚期脆弱性问题，通过改进链式推理结构（ASCoT）应对晚期错误的影响，提出模块化框架与位置权重策略以优化自检与修正机制。**

- **链接: [http://arxiv.org/pdf/2508.05282v1](http://arxiv.org/pdf/2508.05282v1)**

> **作者:** Dongxu Zhang; Ning Yang; Jihua Zhu; Jinnan Yang; Miao Xin; Baoliang Tian
>
> **摘要:** Chain-of-Thought (CoT) prompting has significantly advanced the reasoning capabilities of Large Language Models (LLMs), yet the reliability of these reasoning chains remains a critical challenge. A widely held "cascading failure" hypothesis suggests that errors are most detrimental when they occur early in the reasoning process. This paper challenges that assumption through systematic error-injection experiments, revealing a counter-intuitive phenomenon we term "Late-Stage Fragility": errors introduced in the later stages of a CoT chain are significantly more likely to corrupt the final answer than identical errors made at the beginning. To address this specific vulnerability, we introduce the Adaptive Self-Correction Chain-of-Thought (ASCoT) method. ASCoT employs a modular pipeline in which an Adaptive Verification Manager (AVM) operates first, followed by the Multi-Perspective Self-Correction Engine (MSCE). The AVM leverages a Positional Impact Score function I(k) that assigns different weights based on the position within the reasoning chains, addressing the Late-Stage Fragility issue by identifying and prioritizing high-risk, late-stage steps. Once these critical steps are identified, the MSCE applies robust, dual-path correction specifically to the failure parts. Extensive experiments on benchmarks such as GSM8K and MATH demonstrate that ASCoT achieves outstanding accuracy, outperforming strong baselines, including standard CoT. Our work underscores the importance of diagnosing specific failure modes in LLM reasoning and advocates for a shift from uniform verification strategies to adaptive, vulnerability-aware correction mechanisms.
>
---
#### [new 012] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文旨在解决大型推理语言模型中因反射行为导致的冗余推理问题，通过动态抑制高置信度反射触发器实现高效推理，提升模型性能与实用性。**

- **链接: [http://arxiv.org/pdf/2508.05337v1](http://arxiv.org/pdf/2508.05337v1)**

> **作者:** Jiameng Huang; Baijiong Lin; Guhao Feng; Jierun Chen; Di He; Lu Hou
>
> **备注:** Technical Report
>
> **摘要:** Recent Large Reasoning Language Models (LRLMs) employ long chain-of-thought reasoning with complex reflection behaviors, typically signaled by specific trigger words (e.g., "Wait" and "Alternatively") to enhance performance. However, these reflection behaviors can lead to the overthinking problem where the generation of redundant reasoning steps that unnecessarily increase token usage, raise inference costs, and reduce practical utility. In this paper, we propose Certainty-Guided Reflection Suppression (CGRS), a novel method that mitigates overthinking in LRLMs while maintaining reasoning accuracy. CGRS operates by dynamically suppressing the model's generation of reflection triggers when it exhibits high confidence in its current response, thereby preventing redundant reflection cycles without compromising output quality. Our approach is model-agnostic, requires no retraining or architectural modifications, and can be integrated seamlessly with existing autoregressive generation pipelines. Extensive experiments across four reasoning benchmarks (i.e., AIME24, AMC23, MATH500, and GPQA-D) demonstrate CGRS's effectiveness: it reduces token usage by an average of 18.5% to 41.9% while preserving accuracy. It also achieves the optimal balance between length reduction and performance compared to state-of-the-art baselines. These results hold consistently across model architectures (e.g., DeepSeek-R1-Distill series, QwQ-32B, and Qwen3 family) and scales (4B to 32B parameters), highlighting CGRS's practical value for efficient reasoning.
>
---
#### [new 013] Decision-Making with Deliberation: Meta-reviewing as a Document-grounded Dialogue
- **分类: cs.CL**

- **简介: 该论文探讨了元综述中的决策过程，旨在通过对话代理辅助专家评审，解决了传统方法对数据稀缺性和效率提升的挑战。研究利用大语言模型生成合成数据并训练定制化对话系统，验证了其在实际场景中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.05283v1](http://arxiv.org/pdf/2508.05283v1)**

> **作者:** Sukannya Purkayastha; Nils Dycke; Anne Lauscher; Iryna Gurevych
>
> **备注:** 36 pages, 16 tables, 13 figures
>
> **摘要:** Meta-reviewing is a pivotal stage in the peer-review process, serving as the final step in determining whether a paper is recommended for acceptance. Prior research on meta-reviewing has treated this as a summarization problem over review reports. However, complementary to this perspective, meta-reviewing is a decision-making process that requires weighing reviewer arguments and placing them within a broader context. Prior research has demonstrated that decision-makers can be effectively assisted in such scenarios via dialogue agents. In line with this framing, we explore the practical challenges for realizing dialog agents that can effectively assist meta-reviewers. Concretely, we first address the issue of data scarcity for training dialogue agents by generating synthetic data using Large Language Models (LLMs) based on a self-refinement strategy to improve the relevance of these dialogues to expert domains. Our experiments demonstrate that this method produces higher-quality synthetic data and can serve as a valuable resource towards training meta-reviewing assistants. Subsequently, we utilize this data to train dialogue agents tailored for meta-reviewing and find that these agents outperform \emph{off-the-shelf} LLM-based assistants for this task. Finally, we apply our agents in real-world meta-reviewing scenarios and confirm their effectiveness in enhancing the efficiency of meta-reviewing.\footnote{Code and Data: https://github.com/UKPLab/arxiv2025-meta-review-as-dialog
>
---
#### [new 014] Attention Basin: Why Contextual Position Matters in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究了大型语言模型（LLMs）中上下文位置对性能的影响，提出注意力盆地现象（即模型优先关注开头和结尾信息），并通过引入AttnRank框架实现优化，解决了模型对上下文信息分配偏差的敏感性问题。**

- **链接: [http://arxiv.org/pdf/2508.05128v1](http://arxiv.org/pdf/2508.05128v1)**

> **作者:** Zihao Yi; Delong Zeng; Zhenqing Ling; Haohao Luo; Zhe Xu; Wei Liu; Jian Luan; Wanxia Cao; Ying Shen
>
> **摘要:** The performance of Large Language Models (LLMs) is significantly sensitive to the contextual position of information in the input. To investigate the mechanism behind this positional bias, our extensive experiments reveal a consistent phenomenon we term the attention basin: when presented with a sequence of structured items (e.g., retrieved documents or few-shot examples), models systematically assign higher attention to the items at the beginning and end of the sequence, while neglecting those in the middle. Crucially, our analysis further reveals that allocating higher attention to critical information is key to enhancing model performance. Based on these insights, we introduce Attention-Driven Reranking (AttnRank), a two-stage framework that (i) estimates a model's intrinsic positional attention preferences using a small calibration set, and (ii) reorders retrieved documents or few-shot examples to align the most salient content with these high-attention positions. AttnRank is a model-agnostic, training-free, and plug-and-play method with minimal computational overhead. Experiments on multi-hop QA and few-shot in-context learning tasks demonstrate that AttnRank achieves substantial improvements across 10 large language models of varying architectures and scales, without modifying model parameters or training procedures.
>
---
#### [new 015] Evaluation of LLMs in AMR Parsing
- **分类: cs.CL; cs.AI**

- **简介: 该论文评估了不同LLM架构在AMR解析中的性能，解决如何通过直方图微调提升模型效率的问题，发现LLaMA 3.2在结构有效性与语义表现上优于其他模型。**

- **链接: [http://arxiv.org/pdf/2508.05028v1](http://arxiv.org/pdf/2508.05028v1)**

> **作者:** Shu Han Ho
>
> **备注:** 27 pages, 32 figures
>
> **摘要:** Meaning Representation (AMR) is a semantic formalism that encodes sentence meaning as rooted, directed, acyclic graphs, where nodes represent concepts and edges denote semantic relations. Finetuning decoder only Large Language Models (LLMs) represent a promising novel straightfoward direction for AMR parsing. This paper presents a comprehensive evaluation of finetuning four distinct LLM architectures, Phi 3.5, Gemma 2, LLaMA 3.2, and DeepSeek R1 LLaMA Distilled using the LDC2020T02 Gold AMR3.0 test set. Our results have shown that straightfoward finetuning of decoder only LLMs can achieve comparable performance to complex State of the Art (SOTA) AMR parsers. Notably, LLaMA 3.2 demonstrates competitive performance against SOTA AMR parsers given a straightforward finetuning approach. We achieved SMATCH F1: 0.804 on the full LDC2020T02 test split, on par with APT + Silver (IBM) at 0.804 and approaching Graphene Smatch (MBSE) at 0.854. Across our analysis, we also observed a consistent pattern where LLaMA 3.2 leads in semantic performance while Phi 3.5 excels in structural validity.
>
---
#### [new 016] ATLANTIS at SemEval-2025 Task 3: Detecting Hallucinated Text Spans in Question Answering
- **分类: cs.CL**

- **简介: 该论文属于SemEval-2025 Task 3，旨在检测问答系统中的虚假文本段落，通过LLM结合外部上下文、few-shot提示或微调方法实现高精度识别，取得西班牙/英语/德语等多语言的优异成绩，验证了微调模型与提示工程的有效性。**

- **链接: [http://arxiv.org/pdf/2508.05179v1](http://arxiv.org/pdf/2508.05179v1)**

> **作者:** Catherine Kobus; François Lancelot; Marion-Cécile Martin; Nawal Ould Amer
>
> **摘要:** This paper presents the contributions of the ATLANTIS team to SemEval-2025 Task 3, focusing on detecting hallucinated text spans in question answering systems. Large Language Models (LLMs) have significantly advanced Natural Language Generation (NLG) but remain susceptible to hallucinations, generating incorrect or misleading content. To address this, we explored methods both with and without external context, utilizing few-shot prompting with a LLM, token-level classification or LLM fine-tuned on synthetic data. Notably, our approaches achieved top rankings in Spanish and competitive placements in English and German. This work highlights the importance of integrating relevant context to mitigate hallucinations and demonstrate the potential of fine-tuned models and prompt engineering.
>
---
#### [new 017] LLMEval-3: A Large-Scale Longitudinal Study on Robust and Fair Evaluation of Large Language Models
- **分类: cs.CL**

- **简介: LLMEval-3是动态评估LLMs的框架，解决静态基准易受数据污染和过拟合的问题，通过220k问题库和自动化流程实现90%与专家一致，结合长期研究揭示数据污染漏洞，验证动态评估方法的有效性。**

- **链接: [http://arxiv.org/pdf/2508.05452v1](http://arxiv.org/pdf/2508.05452v1)**

> **作者:** Ming Zhang; Yujiong Shen; Jingyi Deng; Yuhui Wang; Yue Zhang; Junzhe Wang; Shichun Liu; Shihan Dou; Huayu Sha; Qiyuan Peng; Changhao Jiang; Jingqi Tong; Yilong Wu; Zhihao Zhang; Mingqi Wu; Zhiheng Xi; Mingxu Chai; Tao Liang; Zhihui Fei; Zhen Wang; Mingyang Wan; Guojun Ma; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Existing evaluation of Large Language Models (LLMs) on static benchmarks is vulnerable to data contamination and leaderboard overfitting, critical issues that obscure true model capabilities. To address this, we introduce LLMEval-3, a framework for dynamic evaluation of LLMs. LLMEval-3 is built on a proprietary bank of 220k graduate-level questions, from which it dynamically samples unseen test sets for each evaluation run. Its automated pipeline ensures integrity via contamination-resistant data curation, a novel anti-cheating architecture, and a calibrated LLM-as-a-judge process achieving 90% agreement with human experts, complemented by a relative ranking system for fair comparison. An 20-month longitudinal study of nearly 50 leading models reveals a performance ceiling on knowledge memorization and exposes data contamination vulnerabilities undetectable by static benchmarks. The framework demonstrates exceptional robustness in ranking stability and consistency, providing strong empirical validation for the dynamic evaluation paradigm. LLMEval-3 offers a robust and credible methodology for assessing the true capabilities of LLMs beyond leaderboard scores, promoting the development of more trustworthy evaluation standards.
>
---
#### [new 018] Align, Don't Divide: Revisiting the LoRA Architecture in Multi-Task Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨了多任务学习（MTL）中LoRA架构的优化，旨在解决传统LoRA结构因多头/适配器冗余导致的性能下降问题。研究提出Align-LoRA，通过引入共享表示损失实现结构简化与任务对齐，验证了基于共享特征的多任务适应更为有效。**

- **链接: [http://arxiv.org/pdf/2508.05078v1](http://arxiv.org/pdf/2508.05078v1)**

> **作者:** Jinda Liu; Bo Cheng; Yi Chang; Yuan Wu
>
> **摘要:** Parameter-Efficient Fine-Tuning (PEFT) is essential for adapting Large Language Models (LLMs). In practice, LLMs are often required to handle a diverse set of tasks from multiple domains, a scenario naturally addressed by multi-task learning (MTL). Within this MTL context, a prevailing trend involves LoRA variants with multiple adapters or heads, which advocate for structural diversity to capture task-specific knowledge. Our findings present a direct challenge to this paradigm. We first show that a simplified multi-head architecture with high inter-head similarity substantially outperforms complex multi-adapter and multi-head systems. This leads us to question the multi-component paradigm itself, and we further demonstrate that a standard single-adapter LoRA, with a sufficiently increased rank, also achieves highly competitive performance. These results lead us to a new hypothesis: effective MTL generalization hinges on learning robust shared representations, not isolating task-specific features. To validate this, we propose Align-LoRA, which incorporates an explicit loss to align task representations within the shared adapter space. Experiments confirm that Align-LoRA significantly surpasses all baselines, establishing a simpler yet more effective paradigm for adapting LLMs to multiple tasks. The code is available at https://github.com/jinda-liu/Align-LoRA.
>
---
#### [new 019] A Multi-Stage Large Language Model Framework for Extracting Suicide-Related Social Determinants of Health
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在构建多阶段大语言模型框架，解决从文本中提取自杀相关社会决定因素的任务，通过对比现有模型并验证其性能提升及解释性，旨在增强SDoH因子分析的准确性和透明度。**

- **链接: [http://arxiv.org/pdf/2508.05003v1](http://arxiv.org/pdf/2508.05003v1)**

> **作者:** Song Wang; Yishu Wei; Haotian Ma; Max Lovitt; Kelly Deng; Yuan Meng; Zihan Xu; Jingze Zhang; Yunyu Xiao; Ying Ding; Xuhai Xu; Joydeep Ghosh; Yifan Peng
>
> **摘要:** Background: Understanding social determinants of health (SDoH) factors contributing to suicide incidents is crucial for early intervention and prevention. However, data-driven approaches to this goal face challenges such as long-tailed factor distributions, analyzing pivotal stressors preceding suicide incidents, and limited model explainability. Methods: We present a multi-stage large language model framework to enhance SDoH factor extraction from unstructured text. Our approach was compared to other state-of-the-art language models (i.e., pre-trained BioBERT and GPT-3.5-turbo) and reasoning models (i.e., DeepSeek-R1). We also evaluated how the model's explanations help people annotate SDoH factors more quickly and accurately. The analysis included both automated comparisons and a pilot user study. Results: We show that our proposed framework demonstrated performance boosts in the overarching task of extracting SDoH factors and in the finer-grained tasks of retrieving relevant context. Additionally, we show that fine-tuning a smaller, task-specific model achieves comparable or better performance with reduced inference costs. The multi-stage design not only enhances extraction but also provides intermediate explanations, improving model explainability. Conclusions: Our approach improves both the accuracy and transparency of extracting suicide-related SDoH from unstructured texts. These advancements have the potential to support early identification of individuals at risk and inform more effective prevention strategies.
>
---
#### [new 020] Dialogues Aspect-based Sentiment Quadruple Extraction via Structural Entropy Minimization Partitioning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话信息抽取任务，旨在从多轮对话中提取目标-观点-情感四元组，解决现有方法因依赖全局关系而引入噪声的问题。通过结构熵最小化划分对话并采用两步提取框架，实现高效准确的四元组提取。**

- **链接: [http://arxiv.org/pdf/2508.05023v1](http://arxiv.org/pdf/2508.05023v1)**

> **作者:** Kun Peng; Cong Cao; Hao Peng; Zhifeng Hao; Lei Jiang; Kongjing Gu; Yanbing Liu; Philip S. Yu
>
> **备注:** Accepted by CIKM2025
>
> **摘要:** Dialogues Aspect-based Sentiment Quadruple Extraction (DiaASQ) aims to extract all target-aspect-opinion-sentiment quadruples from a given multi-round, multi-participant dialogue. Existing methods typically learn word relations across entire dialogues, assuming a uniform distribution of sentiment elements. However, we find that dialogues often contain multiple semantically independent sub-dialogues without clear dependencies between them. Therefore, learning word relationships across the entire dialogue inevitably introduces additional noise into the extraction process. To address this, our method focuses on partitioning dialogues into semantically independent sub-dialogues. Achieving completeness while minimizing these sub-dialogues presents a significant challenge. Simply partitioning based on reply relationships is ineffective. Instead, we propose utilizing a structural entropy minimization algorithm to partition the dialogues. This approach aims to preserve relevant utterances while distinguishing irrelevant ones as much as possible. Furthermore, we introduce a two-step framework for quadruple extraction: first extracting individual sentiment elements at the utterance level, then matching quadruples at the sub-dialogue level. Extensive experiments demonstrate that our approach achieves state-of-the-art performance in DiaASQ with much lower computational costs.
>
---
#### [new 021] CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL
- **分类: cs.CL**

- **简介: 该论文提出CodeBoost框架，解决传统LLM训练依赖人工指令与代码片段不平衡问题，通过最大图连通性选课、双向预测、误差感知等技术提升模型泛化能力，验证其有效性和可扩展性。**

- **链接: [http://arxiv.org/pdf/2508.05242v1](http://arxiv.org/pdf/2508.05242v1)**

> **作者:** Sijie Wang; Quanjiang Guo; Kai Zhao; Yawei Zhang; Xin Li; Xiang Li; Siqi Li; Rui She; Shangshu Yu; Wee Peng Tay
>
> **备注:** Technical report. Project page: https://github.com/sijieaaa/CodeBoost
>
> **摘要:** Code large language models (LLMs) have become indispensable tools for building efficient and automated coding pipelines. Existing models are typically post-trained using reinforcement learning (RL) from general-purpose LLMs using "human instruction-final answer" pairs, where the instructions are usually from manual annotations. However, collecting high-quality coding instructions is both labor-intensive and difficult to scale. On the other hand, code snippets are abundantly available from various sources. This imbalance presents a major bottleneck in instruction-based post-training. We propose CodeBoost, a post-training framework that enhances code LLMs purely from code snippets, without relying on human-annotated instructions. CodeBoost introduces the following key components: (1) maximum-clique curation, which selects a representative and diverse training corpus from code; (2) bi-directional prediction, which enables the model to learn from both forward and backward prediction objectives; (3) error-aware prediction, which incorporates learning signals from both correct and incorrect outputs; (4) heterogeneous augmentation, which diversifies the training distribution to enrich code semantics; and (5) heterogeneous rewarding, which guides model learning through multiple reward types including format correctness and execution feedback from both successes and failures. Extensive experiments across several code LLMs and benchmarks verify that CodeBoost consistently improves performance, demonstrating its effectiveness as a scalable and effective training pipeline.
>
---
#### [new 022] Resource-Limited Joint Multimodal Sentiment Reasoning and Classification via Chain-of-Thought Enhancement and Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于资源受限的多模态情感推理与分类任务，旨在解决传统参数密集型模型难以满足资源约束的问题。提出"教师-助手-学生"结构的MulCoT-RD模型，通过轻量化训练提升推理效率与泛化能力，实现JMSRC任务的高效部署。**

- **链接: [http://arxiv.org/pdf/2508.05234v1](http://arxiv.org/pdf/2508.05234v1)**

> **作者:** Haonan Shangguan; Xiaocui Yang; Shi Feng; Daling Wang; Yifei Zhang; Ge Yu
>
> **摘要:** The surge in rich multimodal content on social media platforms has greatly advanced Multimodal Sentiment Analysis (MSA), with Large Language Models (LLMs) further accelerating progress in this field. Current approaches primarily leverage the knowledge and reasoning capabilities of parameter-heavy (Multimodal) LLMs for sentiment classification, overlooking autonomous multimodal sentiment reasoning generation in resource-constrained environments. Therefore, we focus on the Resource-Limited Joint Multimodal Sentiment Reasoning and Classification task, JMSRC, which simultaneously performs multimodal sentiment reasoning chain generation and sentiment classification only with a lightweight model. We propose a Multimodal Chain-of-Thought Reasoning Distillation model, MulCoT-RD, designed for JMSRC that employs a "Teacher-Assistant-Student" distillation paradigm to address deployment constraints in resource-limited environments. We first leverage a high-performance Multimodal Large Language Model (MLLM) to generate the initial reasoning dataset and train a medium-sized assistant model with a multi-task learning mechanism. A lightweight student model is jointly trained to perform efficient multimodal sentiment reasoning generation and classification. Extensive experiments on four datasets demonstrate that MulCoT-RD with only 3B parameters achieves strong performance on JMSRC, while exhibiting robust generalization and enhanced interpretability.
>
---
#### [new 023] MathSmith: Towards Extremely Hard Mathematical Reasoning by Forging Synthetic Problems with a Reinforced Policy
- **分类: cs.CL**

- **简介: 该论文提出通过合成高难度数学问题并结合强化学习优化LLM推理能力的任务，旨在解决现有训练数据不足的问题，构建可扩展的多维度挑战性问题集，提升模型长链思考能力和跨领域应用。**

- **链接: [http://arxiv.org/pdf/2508.05592v1](http://arxiv.org/pdf/2508.05592v1)**

> **作者:** Shaoxiong Zhan; Yanlin Lai; Ziyu Lu; Dahua Lin; Ziqing Yang; Fei Tang
>
> **摘要:** Large language models have achieved substantial progress in mathematical reasoning, yet their advancement is limited by the scarcity of high-quality, high-difficulty training data. Existing synthesis methods largely rely on transforming human-written templates, limiting both diversity and scalability. We propose MathSmith, a novel framework for synthesizing challenging mathematical problems to enhance LLM reasoning. Rather than modifying existing problems, MathSmith constructs new ones from scratch by randomly sampling concept-explanation pairs from PlanetMath, ensuring data independence and avoiding contamination. To increase difficulty, we design nine predefined strategies as soft constraints during rationales. We further adopts reinforcement learning to jointly optimize structural validity, reasoning complexity, and answer consistency. The length of the reasoning trace generated under autoregressive prompting is used to reflect cognitive complexity, encouraging the creation of more demanding problems aligned with long-chain-of-thought reasoning. Experiments across five benchmarks, categorized as easy & medium (GSM8K, MATH-500) and hard (AIME2024, AIME2025, OlympiadBench), show that MathSmith consistently outperforms existing baselines under both short and long CoT settings. Additionally, a weakness-focused variant generation module enables targeted improvement on specific concepts. Overall, MathSmith exhibits strong scalability, generalization, and transferability, highlighting the promise of high-difficulty synthetic data in advancing LLM reasoning capabilities.
>
---
#### [new 024] Pitch Accent Detection improves Pretrained Automatic Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文旨在改进基于半监督的自动语音识别（ASR）系统，解决传统方法在融合补充信息时性能不足的问题，通过引入联合ASR与pitch accent检测模块，显著提升了F1分数并优化了有限资源下的训练效果。**

- **链接: [http://arxiv.org/pdf/2508.04814v1](http://arxiv.org/pdf/2508.04814v1)**

> **作者:** David Sasu; Natalie Schluter
>
> **摘要:** We show the performance of Automatic Speech Recognition (ASR) systems that use semi-supervised speech representations can be boosted by a complimentary pitch accent detection module, by introducing a joint ASR and pitch accent detection model. The pitch accent detection component of our model achieves a significant improvement on the state-of-the-art for the task, closing the gap in F1-score by 41%. Additionally, the ASR performance in joint training decreases WER by 28.3% on LibriSpeech, under limited resource fine-tuning. With these results, we show the importance of extending pretrained speech models to retain or re-learn important prosodic cues such as pitch accent.
>
---
#### [new 025] I Think, Therefore I Am Under-Qualified? A Benchmark for Evaluating Linguistic Shibboleth Detection in LLM Hiring Evaluations
- **分类: cs.CL**

- **简介: 该论文旨在评估大型语言模型（LLMs）如何通过隐晦的语言陷阱（如修饰语）反映偏见，解决AI招聘中的歧视问题，构建了针对特定语言模式的基准测试。**

- **链接: [http://arxiv.org/pdf/2508.04939v1](http://arxiv.org/pdf/2508.04939v1)**

> **作者:** Julia Kharchenko; Tanya Roosta; Aman Chadha; Chirag Shah
>
> **摘要:** This paper introduces a comprehensive benchmark for evaluating how Large Language Models (LLMs) respond to linguistic shibboleths: subtle linguistic markers that can inadvertently reveal demographic attributes such as gender, social class, or regional background. Through carefully constructed interview simulations using 100 validated question-response pairs, we demonstrate how LLMs systematically penalize certain linguistic patterns, particularly hedging language, despite equivalent content quality. Our benchmark generates controlled linguistic variations that isolate specific phenomena while maintaining semantic equivalence, which enables the precise measurement of demographic bias in automated evaluation systems. We validate our approach along multiple linguistic dimensions, showing that hedged responses receive 25.6% lower ratings on average, and demonstrate the benchmark's effectiveness in identifying model-specific biases. This work establishes a foundational framework for detecting and measuring linguistic discrimination in AI systems, with broad applications to fairness in automated decision-making contexts.
>
---
#### [new 026] Pruning Large Language Models by Identifying and Preserving Functional Networks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文旨在解决大型语言模型结构化剪枝中忽略人工神经元交互与功能协同的问题，提出通过分解LLM为功能网络并保留关键节点实现高效模型压缩，实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2508.05239v1](http://arxiv.org/pdf/2508.05239v1)**

> **作者:** Yiheng Liu; Junhao Ning; Sichen Xia; Xiaohui Gao; Ning Qiang; Bao Ge; Junwei Han; Xintao Hu
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Structured pruning is one of the representative techniques for compressing large language models (LLMs) to reduce GPU memory consumption and accelerate inference speed. It offers significant practical value in improving the efficiency of LLMs in real-world applications. Current structured pruning methods typically rely on assessment of the importance of the structure units and pruning the units with less importance. Most of them overlooks the interaction and collaboration among artificial neurons that are crucial for the functionalities of LLMs, leading to a disruption in the macro functional architecture of LLMs and consequently a pruning performance degradation. Inspired by the inherent similarities between artificial neural networks and functional neural networks in the human brain, we alleviate this challenge and propose to prune LLMs by identifying and preserving functional networks within LLMs in this study. To achieve this, we treat an LLM as a digital brain and decompose the LLM into functional networks, analogous to identifying functional brain networks in neuroimaging data. Afterwards, an LLM is pruned by preserving the key neurons within these functional networks. Experimental results demonstrate that the proposed method can successfully identify and locate functional networks and key neurons in LLMs, enabling efficient model pruning. Our code is available at https://github.com/WhatAboutMyStar/LLM_ACTIVATION.
>
---
#### [new 027] The World According to LLMs: How Geographic Origin Influences LLMs' Entity Deduction Capabilities
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨地理因素对LLM实体推理能力的影响，通过20题游戏测试其表现差异，利用Geo20Q+数据集验证结果。研究发现北、南、西、东地区LLM在推理能力存在显著差异，且语言影响较小。**

- **链接: [http://arxiv.org/pdf/2508.05525v1](http://arxiv.org/pdf/2508.05525v1)**

> **作者:** Harsh Nishant Lalai; Raj Sanjay Shah; Jiaxin Pei; Sashank Varma; Yi-Chia Wang; Ali Emami
>
> **备注:** Conference on Language Modeling 2025
>
> **摘要:** Large Language Models (LLMs) have been extensively tuned to mitigate explicit biases, yet they often exhibit subtle implicit biases rooted in their pre-training data. Rather than directly probing LLMs with human-crafted questions that may trigger guardrails, we propose studying how models behave when they proactively ask questions themselves. The 20 Questions game, a multi-turn deduction task, serves as an ideal testbed for this purpose. We systematically evaluate geographic performance disparities in entity deduction using a new dataset, Geo20Q+, consisting of both notable people and culturally significant objects (e.g., foods, landmarks, animals) from diverse regions. We test popular LLMs across two gameplay configurations (canonical 20-question and unlimited turns) and in seven languages (English, Hindi, Mandarin, Japanese, French, Spanish, and Turkish). Our results reveal geographic disparities: LLMs are substantially more successful at deducing entities from the Global North than the Global South, and the Global West than the Global East. While Wikipedia pageviews and pre-training corpus frequency correlate mildly with performance, they fail to fully explain these disparities. Notably, the language in which the game is played has minimal impact on performance gaps. These findings demonstrate the value of creative, free-form evaluation frameworks for uncovering subtle biases in LLMs that remain hidden in standard prompting setups. By analyzing how models initiate and pursue reasoning goals over multiple turns, we find geographic and cultural disparities embedded in their reasoning processes. We release the dataset (Geo20Q+) and code at https://sites.google.com/view/llmbias20q/home.
>
---
#### [new 028] SONAR-LLM: Autoregressive Transformer that Thinks in Sentence Embeddings and Speaks in Tokens
- **分类: cs.CL**

- **简介: 本研究提出一种自回归Transformer模型，将句子嵌入视为思考空间，通过冻结的SONAR解码器实现token级交叉熵优化，结合LCM方法减少扩散采样并提升生成质量。**

- **链接: [http://arxiv.org/pdf/2508.05305v1](http://arxiv.org/pdf/2508.05305v1)**

> **作者:** Nikita Dragunov; Temurbek Rahmatullaev; Elizaveta Goncharova; Andrey Kuznetsov; Anton Razzhigaev
>
> **摘要:** The recently proposed Large Concept Model (LCM) generates text by predicting a sequence of sentence-level embeddings and training with either mean-squared error or diffusion objectives. We present SONAR-LLM, a decoder-only transformer that "thinks" in the same continuous SONAR embedding space, yet is supervised through token-level cross-entropy propagated via the frozen SONAR decoder. This hybrid objective retains the semantic abstraction of LCM while eliminating its diffusion sampler and restoring a likelihood-based training signal. Across model sizes from 39M to 1.3B parameters, SONAR-LLM attains competitive generation quality. We report scaling trends, ablations, benchmark results, and release the complete training code and all pretrained checkpoints to foster reproducibility and future research.
>
---
#### [new 029] OmniEAR: Benchmarking Agent Reasoning in Embodied Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在评估基于语言模型的具身代理推理能力，解决如何动态学习物理交互、工具使用及多智能体协作的问题。通过构建1500个场景的文本环境，提出OmniEAR框架，发现传统基准无法捕捉代理自主决策能力，揭示了具身推理面临的挑战。**

- **链接: [http://arxiv.org/pdf/2508.05614v1](http://arxiv.org/pdf/2508.05614v1)**

> **作者:** Zixuan Wang; Dingming Li; Hongxing Li; Shuo Chen; Yuchen Yan; Wenqi Zhang; Yongliang Shen; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **备注:** Project Page: https://zju-real.github.io/OmniEmbodied Code: https://github.com/ZJU-REAL/OmniEmbodied
>
> **摘要:** Large language models excel at abstract reasoning but their capacity for embodied agent reasoning remains largely unexplored. We present OmniEAR, a comprehensive framework for evaluating how language models reason about physical interactions, tool usage, and multi-agent coordination in embodied tasks. Unlike existing benchmarks that provide predefined tool sets or explicit collaboration directives, OmniEAR requires agents to dynamically acquire capabilities and autonomously determine coordination strategies based on task demands. Through text-based environment representation, we model continuous physical properties and complex spatial relationships across 1,500 scenarios spanning household and industrial domains. Our systematic evaluation reveals severe performance degradation when models must reason from constraints: while achieving 85-96% success with explicit instructions, performance drops to 56-85% for tool reasoning and 63-85% for implicit collaboration, with compound tasks showing over 50% failure rates. Surprisingly, complete environmental information degrades coordination performance, indicating models cannot filter task-relevant constraints. Fine-tuning improves single-agent tasks dramatically (0.6% to 76.3%) but yields minimal multi-agent gains (1.5% to 5.5%), exposing fundamental architectural limitations. These findings demonstrate that embodied reasoning poses fundamentally different challenges than current models can address, establishing OmniEAR as a rigorous benchmark for evaluating and advancing embodied AI systems. Our code and data are included in the supplementary materials and will be open-sourced upon acceptance.
>
---
#### [new 030] H-Net++: Hierarchical Dynamic Chunking for Tokenizer-Free Language Modelling in Morphologically-Rich Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在解决形态丰富语言（MRLs）中字节级语言模型的计算效率与分词依赖性问题，提出H-Net++通过动态分块学习语义分割，结合轻量化Transformer混合器、文档一致性前向概率及专项处理orthographic artifacts，实现了高效且无分词的模型设计。**

- **链接: [http://arxiv.org/pdf/2508.05628v1](http://arxiv.org/pdf/2508.05628v1)**

> **作者:** Mehrdad Zakershahrak; Samira Ghodratnama
>
> **摘要:** Byte-level language models eliminate fragile tokenizers but face computational challenges in morphologically-rich languages (MRLs), where words span many bytes. We propose H-NET++, a hierarchical dynamic-chunking model that learns linguistically-informed segmentation through end-to-end training. Key innovations include: (1) a lightweight Transformer context-mixer (1.9M parameters) for cross-chunk attention, (2) a two-level latent hyper-prior for document-level consistency, (3) specialized handling of orthographic artifacts (e.g. Persian ZWNJ), and (4) curriculum-based training with staged sequence lengths. On a 1.4B-token Persian corpus, H-NET++ achieves state-of-the-art results: 0.159 BPB reduction versus BPE-based GPT-2-fa (12% better compression), 5.4pp gain on ParsGLUE, 53% improved robustness to ZWNJ corruption, and 73.8% F1 on gold morphological boundaries. Our learned chunks align with Persian morphology without explicit supervision, demonstrating that hierarchical dynamic chunking provides an effective tokenizer-free solution for MRLs while maintaining computational efficiency.
>
---
#### [new 031] How Do LLMs Persuade? Linear Probes Can Uncover Persuasion Dynamics in Multi-Turn Conversations
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文探讨了大型语言模型（LLMs）如何通过线性探针揭示其说服行为的动态，旨在理解多轮对话中说服成功、说服策略等特征，并验证其与传统提示方法相比的优势，为复杂行为如欺骗提供新视角。**

- **链接: [http://arxiv.org/pdf/2508.05625v1](http://arxiv.org/pdf/2508.05625v1)**

> **作者:** Brandon Jaipersaud; David Krueger; Ekdeep Singh Lubana
>
> **摘要:** Large Language Models (LLMs) have started to demonstrate the ability to persuade humans, yet our understanding of how this dynamic transpires is limited. Recent work has used linear probes, lightweight tools for analyzing model representations, to study various LLM skills such as the ability to model user sentiment and political perspective. Motivated by this, we apply probes to study persuasion dynamics in natural, multi-turn conversations. We leverage insights from cognitive science to train probes on distinct aspects of persuasion: persuasion success, persuadee personality, and persuasion strategy. Despite their simplicity, we show that they capture various aspects of persuasion at both the sample and dataset levels. For instance, probes can identify the point in a conversation where the persuadee was persuaded or where persuasive success generally occurs across the entire dataset. We also show that in addition to being faster than expensive prompting-based approaches, probes can do just as well and even outperform prompting in some settings, such as when uncovering persuasion strategy. This suggests probes as a plausible avenue for studying other complex behaviours such as deception and manipulation, especially in multi-turn settings and large-scale dataset analysis where prompting-based methods would be computationally inefficient.
>
---
#### [new 032] Conformal Sets in Multiple-Choice Question Answering under Black-Box Settings with Provable Coverage Guarantees
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在解决多选题答案可靠性不足的问题，在黑盒环境下通过频率基不确定性量化方法，结合CP技术实现覆盖保障，验证采样频率优于传统概率方法，为MCQA提供可靠模型框架。**

- **链接: [http://arxiv.org/pdf/2508.05544v1](http://arxiv.org/pdf/2508.05544v1)**

> **作者:** Guang Yang; Xinyang Liu
>
> **备注:** under review
>
> **摘要:** Large Language Models (LLMs) have shown remarkable progress in multiple-choice question answering (MCQA), but their inherent unreliability, such as hallucination and overconfidence, limits their application in high-risk domains. To address this, we propose a frequency-based uncertainty quantification method under black-box settings, leveraging conformal prediction (CP) to ensure provable coverage guarantees. Our approach involves multiple independent samplings of the model's output distribution for each input, with the most frequent sample serving as a reference to calculate predictive entropy (PE). Experimental evaluations across six LLMs and four datasets (MedMCQA, MedQA, MMLU, MMLU-Pro) demonstrate that frequency-based PE outperforms logit-based PE in distinguishing between correct and incorrect predictions, as measured by AUROC. Furthermore, the method effectively controls the empirical miscoverage rate under user-specified risk levels, validating that sampling frequency can serve as a viable substitute for logit-based probabilities in black-box scenarios. This work provides a distribution-free model-agnostic framework for reliable uncertainty quantification in MCQA with guaranteed coverage, enhancing the trustworthiness of LLMs in practical applications.
>
---
#### [new 033] MyCulture: Exploring Malaysia's Diverse Culture under Low-Resource Language Constraints
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在设计一个针对低资源语言环境下的马来文化评估基准MyCulture，解决LLMs在文化偏见和评估不足的问题，通过开放问答格式减少格式偏倚并分析结构/语言偏差，验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.05429v1](http://arxiv.org/pdf/2508.05429v1)**

> **作者:** Zhong Ken Hew; Jia Xin Low; Sze Jue Yang; Chee Seng chan
>
> **摘要:** Large Language Models (LLMs) often exhibit cultural biases due to training data dominated by high-resource languages like English and Chinese. This poses challenges for accurately representing and evaluating diverse cultural contexts, particularly in low-resource language settings. To address this, we introduce MyCulture, a benchmark designed to comprehensively evaluate LLMs on Malaysian culture across six pillars: arts, attire, customs, entertainment, food, and religion presented in Bahasa Melayu. Unlike conventional benchmarks, MyCulture employs a novel open-ended multiple-choice question format without predefined options, thereby reducing guessing and mitigating format bias. We provide a theoretical justification for the effectiveness of this open-ended structure in improving both fairness and discriminative power. Furthermore, we analyze structural bias by comparing model performance on structured versus free-form outputs, and assess language bias through multilingual prompt variations. Our evaluation across a range of regional and international LLMs reveals significant disparities in cultural comprehension, highlighting the urgent need for culturally grounded and linguistically inclusive benchmarks in the development and assessment of LLMs.
>
---
#### [new 034] Learning to Reason for Factuality
- **分类: cs.CL**

- **简介: 该论文探讨了如何通过改进奖励机制提升大型语言模型在长事实任务中的事实推理能力。研究发现传统RL方法因缺乏验证机制导致奖励欺骗，提出结合事实精度、响应细节与相关性的新颖奖励函数，并利用在线RL实现高效事实推理，最终在基准测试中实现了23.1%的虚假率下降、23%的详细度提升。**

- **链接: [http://arxiv.org/pdf/2508.05618v1](http://arxiv.org/pdf/2508.05618v1)**

> **作者:** Xilun Chen; Ilia Kulikov; Vincent-Pierre Berges; Barlas Oğuz; Rulin Shao; Gargi Ghosh; Jason Weston; Wen-tau Yih
>
> **摘要:** Reasoning Large Language Models (R-LLMs) have significantly advanced complex reasoning tasks but often struggle with factuality, generating substantially more hallucinations than their non-reasoning counterparts on long-form factuality benchmarks. However, extending online Reinforcement Learning (RL), a key component in recent R-LLM advancements, to the long-form factuality setting poses several unique challenges due to the lack of reliable verification methods. Previous work has utilized automatic factuality evaluation frameworks such as FActScore to curate preference data in the offline RL setting, yet we find that directly leveraging such methods as the reward in online RL leads to reward hacking in multiple ways, such as producing less detailed or relevant responses. We propose a novel reward function that simultaneously considers the factual precision, response detail level, and answer relevance, and applies online RL to learn high quality factual reasoning. Evaluated on six long-form factuality benchmarks, our factual reasoning model achieves an average reduction of 23.1 percentage points in hallucination rate, a 23% increase in answer detail level, and no degradation in the overall response helpfulness.
>
---
#### [new 035] Rethinking Creativity Evaluation: A Critical Analysis of Existing Creativity Evaluations
- **分类: cs.CL**

- **简介: 该论文旨在批判现有创造力评价指标的局限性，通过系统比较创造性指数、困惑度等指标，揭示其跨领域不一致性，并提出需构建更通用、多维度的评估框架的任务。**

- **链接: [http://arxiv.org/pdf/2508.05470v1](http://arxiv.org/pdf/2508.05470v1)**

> **作者:** Li-Chun Lu; Miri Liu; Pin-Chun Lu; Yufei Tian; Shao-Hua Sun; Nanyun Peng
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** We systematically examine, analyze, and compare representative creativity measures--creativity index, perplexity, syntactic templates, and LLM-as-a-Judge--across diverse creative domains, including creative writing, unconventional problem-solving, and research ideation. Our analyses reveal that these metrics exhibit limited consistency, capturing different dimensions of creativity. We highlight key limitations, including the creativity index's focus on lexical diversity, perplexity's sensitivity to model confidence, and syntactic templates' inability to capture conceptual creativity. Additionally, LLM-as-a-Judge shows instability and bias. Our findings underscore the need for more robust, generalizable evaluation frameworks that better align with human judgments of creativity.
>
---
#### [new 036] Enhancing Dialogue Annotation with Speaker Characteristics Leveraging a Frozen LLM
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文探讨了通过冷冻LLM增强对话注释的方法，利用语音特征（如年龄、性别、情绪）进行补充，解决了传统LLMs在对话质量上的局限性，并实现了高效且模块化的工作方式。**

- **链接: [http://arxiv.org/pdf/2508.04795v1](http://arxiv.org/pdf/2508.04795v1)**

> **作者:** Thomas Thebaud; Yen-Ju Lu; Matthew Wiesner; Peter Viechnicki; Najim Dehak
>
> **备注:** Accepted in the 2025 IEEE Automatic Speech Recognition and Understanding Workshop
>
> **摘要:** In dialogue transcription pipelines, Large Language Models (LLMs) are frequently employed in post-processing to improve grammar, punctuation, and readability. We explore a complementary post-processing step: enriching transcribed dialogues by adding metadata tags for speaker characteristics such as age, gender, and emotion. Some of the tags are global to the entire dialogue, while some are time-variant. Our approach couples frozen audio foundation models, such as Whisper or WavLM, with a frozen LLAMA language model to infer these speaker attributes, without requiring task-specific fine-tuning of either model. Using lightweight, efficient connectors to bridge audio and language representations, we achieve competitive performance on speaker profiling tasks while preserving modularity and speed. Additionally, we demonstrate that a frozen LLAMA model can compare x-vectors directly, achieving an Equal Error Rate of 8.8% in some scenarios.
>
---
#### [new 037] The TUB Sign Language Corpus Collection
- **分类: cs.CL**

- **简介: 该论文旨在构建一个包含12种语言的平行语料库，解决跨语言对比研究问题，通过多阶段采集与处理完成，首次为8个拉丁美洲语言提供一致数据，并超过1300小时视频资料，规模达到1.3Mtoken。**

- **链接: [http://arxiv.org/pdf/2508.05374v1](http://arxiv.org/pdf/2508.05374v1)**

> **作者:** Eleftherios Avramidis; Vera Czehmann; Fabian Deckert; Lorenz Hufe; Aljoscha Lipski; Yuni Amaloa Quintero Villalobos; Tae Kwon Rhee; Mengqian Shi; Lennart Stölting; Fabrizio Nunnari; Sebastian Möller
>
> **摘要:** We present a collection of parallel corpora of 12 sign languages in video format, together with subtitles in the dominant spoken languages of the corresponding countries. The entire collection includes more than 1,300 hours in 4,381 video files, accompanied by 1,3~M subtitles containing 14~M tokens. Most notably, it includes the first consistent parallel corpora for 8 Latin American sign languages, whereas the size of the German Sign Language corpora is ten times the size of the previously available corpora. The collection was created by collecting and processing videos of multiple sign languages from various online sources, mainly broadcast material of news shows, governmental bodies and educational channels. The preparation involved several stages, including data collection, informing the content creators and seeking usage approvals, scraping, and cropping. The paper provides statistics on the collection and an overview of the methods used to collect the data.
>
---
#### [new 038] BEE-RAG: Balanced Entropy Engineering for Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于检索增强生成（RAG）任务，旨在解决长上下文对性能的影响，通过熵工程方法改进系统适应性，提出BEE-RAG框架并设计零样本策略与高效参数调优。**

- **链接: [http://arxiv.org/pdf/2508.05100v1](http://arxiv.org/pdf/2508.05100v1)**

> **作者:** Yuhao Wang; Ruiyang Ren; Yucheng Wang; Jing Liu; Wayne Xin Zhao; Hua Wu; Haifeng Wang
>
> **摘要:** With the rapid advancement of large language models (LLMs), retrieval-augmented generation (RAG) has emerged as a critical approach to supplement the inherent knowledge limitations of LLMs. However, due to the typically large volume of retrieved information, RAG tends to operate with long context lengths. From the perspective of entropy engineering, we identify unconstrained entropy growth and attention dilution due to long retrieval context as significant factors affecting RAG performance. In this paper, we propose the balanced entropy-engineered RAG (BEE-RAG) framework, which improves the adaptability of RAG systems to varying context lengths through the principle of entropy invariance. By leveraging balanced context entropy to reformulate attention dynamics, BEE-RAG separates attention sensitivity from context length, ensuring a stable entropy level. Building upon this, we introduce a zero-shot inference strategy for multi-importance estimation and a parameter-efficient adaptive fine-tuning mechanism to obtain the optimal balancing factor for different settings. Extensive experiments across multiple RAG tasks demonstrate the effectiveness of BEE-RAG.
>
---
#### [new 039] Cooper: Co-Optimizing Policy and Reward Models in Reinforcement Learning for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出了一种联合优化策略，在大型语言模型中通过动态样本生成和参考答案机制，解决了奖励模型易受攻击的问题，构建了VerifyRM模型并验证其在Qwen2.5中的性能提升。**

- **链接: [http://arxiv.org/pdf/2508.05613v1](http://arxiv.org/pdf/2508.05613v1)**

> **作者:** Haitao Hong; Yuchen Yan; Xingyu Wu; Guiyang Hou; Wenqi Zhang; Weiming Lu; Yongliang Shen; Jun Xiao
>
> **备注:** Project Page: https://zju-real.github.io/cooper Code: https://github.com/zju-real/cooper
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable performance in reasoning tasks, where reinforcement learning (RL) serves as a key algorithm for enhancing their reasoning capabilities. Currently, there are two mainstream reward paradigms: model-based rewards and rule-based rewards. However, both approaches suffer from limitations: rule-based rewards lack robustness, while model-based rewards are vulnerable to reward hacking. To address these issues, we propose Cooper(Co-optimizing Policy Model and Reward Model), a RL framework that jointly optimizes both the policy model and the reward model. Cooper leverages the high precision of rule-based rewards when identifying correct responses, and dynamically constructs and selects positive-negative sample pairs for continued training the reward model. This design enhances robustness and mitigates the risk of reward hacking. To further support Cooper, we introduce a hybrid annotation strategy that efficiently and accurately generates training data for the reward model. We also propose a reference-based reward modeling paradigm, where the reward model takes a reference answer as input. Based on this design, we train a reward model named VerifyRM, which achieves higher accuracy on VerifyBench compared to other models of the same size. We conduct reinforcement learning using both VerifyRM and Cooper. Our experiments show that Cooper not only alleviates reward hacking but also improves end-to-end RL performance, for instance, achieving a 0.54% gain in average accuracy on Qwen2.5-1.5B-Instruct. Our findings demonstrate that dynamically updating reward model is an effective way to combat reward hacking, providing a reference for better integrating reward models into RL.
>
---
#### [new 040] LAG: Logic-Augmented Generation from a Cartesian Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Logic-Augmented Generation（LAG），旨在通过将知识增强与逻辑推理结合，解决大型语言模型（LLMs）在知识密集任务中的泛化缺陷。具体工作包括：将复杂问题分解为逻辑依赖的子问题，利用前缀答案引导后续推理，实现逻辑链式推理，并设计逻辑终止机制防止错误传播，最终提升推理准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.05509v1](http://arxiv.org/pdf/2508.05509v1)**

> **作者:** Yilin Xiao; Chuang Zhou; Qinggang Zhang; Su Dong; Shengyuan Chen; Xiao Huang
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet exhibit critical limitations in knowledge-intensive tasks, often generating hallucinations when faced with questions requiring specialized expertise. While retrieval-augmented generation (RAG) mitigates this by integrating external knowledge, it struggles with complex reasoning scenarios due to its reliance on direct semantic retrieval and lack of structured logical organization. Inspired by Cartesian principles from \textit{Discours de la m\'ethode}, this paper introduces Logic-Augmented Generation (LAG), a novel paradigm that reframes knowledge augmentation through systematic question decomposition and dependency-aware reasoning. Specifically, LAG first decomposes complex questions into atomic sub-questions ordered by logical dependencies. It then resolves these sequentially, using prior answers to guide context retrieval for subsequent sub-questions, ensuring stepwise grounding in logical chain. To prevent error propagation, LAG incorporates a logical termination mechanism that halts inference upon encountering unanswerable sub-questions and reduces wasted computation on excessive reasoning. Finally, it synthesizes all sub-resolutions to generate verified responses. Experiments on four benchmark datasets demonstrate that LAG significantly enhances reasoning robustness, reduces hallucination, and aligns LLM problem-solving with human cognition, offering a principled alternative to existing RAG systems.
>
---
#### [new 041] CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation
- **分类: cs.CL**

- **简介: 该论文提出一种基于模型信心的动态插值策略（CoCoLex），用于增强法律文本生成的上下文关联性，解决了LLM生成不准确或未被约束的问题，通过直接复制模型输出与上下文分布的匹配度提升，实验表明其在长文本生成任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.05534v1](http://arxiv.org/pdf/2508.05534v1)**

> **作者:** Santosh T. Y. S. S; Youssef Tarek Elkhayat; Oana Ichim; Pranav Shetty; Dongsheng Wang; Zhiqiang Ma; Armineh Nourbakhsh; Xiaomo Liu
>
> **备注:** Accepted to ACL 2025-Main Conference
>
> **摘要:** Due to their ability to process long and complex contexts, LLMs can offer key benefits to the Legal domain, but their adoption has been hindered by their tendency to generate unfaithful, ungrounded, or hallucinatory outputs. While Retrieval-Augmented Generation offers a promising solution by grounding generations in external knowledge, it offers no guarantee that the provided context will be effectively integrated. To address this, context-aware decoding strategies have been proposed to amplify the influence of relevant context, but they usually do not explicitly enforce faithfulness to the context. In this work, we introduce Confidence-guided Copy-based Decoding for Legal Text Generation (CoCoLex)-a decoding strategy that dynamically interpolates the model produced vocabulary distribution with a distribution derived based on copying from the context. CoCoLex encourages direct copying based on the model's confidence, ensuring greater fidelity to the source. Experimental results on five legal benchmarks demonstrate that CoCoLex outperforms existing context-aware decoding methods, particularly in long-form generation tasks.
>
---
#### [new 042] Fairy$\pm i$: the First 2-bit Complex LLM with All Parameters in $\{\pm1, \pm i\}$
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出了一种基于复数根的2比特复杂LLM量化框架，解决了在极低比特约束下提升模型精度与效率的问题。通过将权重映射至四次根集合（±1, ±i），构建了信息最优的量化方案，实现乘法运算免于计算，实验表明其在PPL和下游任务中优于现有2比特方法。**

- **链接: [http://arxiv.org/pdf/2508.05571v1](http://arxiv.org/pdf/2508.05571v1)**

> **作者:** Feiyu Wang; Guoan Wang; Yihao Zhang; Shengfan Wang; Weitao Li; Bokai Huang; Shimao Chen; Zihan Jiang; Rui Xu; Tong Yang
>
> **备注:** 13 pages, 14 figures
>
> **摘要:** Quantization-Aware Training (QAT) integrates quantization into the training loop, enabling LLMs to learn robust low-bit representations, and is widely recognized as one of the most promising research directions. All current QAT research focuses on minimizing quantization error on full-precision models, where the full-precision accuracy acts as an upper bound (accuracy ceiling). No existing method has even attempted to surpass this ceiling. To break this ceiling, we propose a new paradigm: raising the ceiling (full-precision model), and then still quantizing it efficiently into 2 bits. We propose Fairy$\pm i$, the first 2-bit quantization framework for complex-valued LLMs. Specifically, our method leverages the representational advantages of the complex domain to boost full-precision accuracy. We map weights to the fourth roots of unity $\{\pm1, \pm i\}$, forming a perfectly symmetric and information-theoretically optimal 2-bit representation. Importantly, each quantized weight has either a zero real or imaginary part, enabling multiplication-free inference using only additions and element swaps. Experimental results show that Fairy$\pm i$ outperforms the ceiling of existing 2-bit quantization approaches in terms of both PPL and downstream tasks, while maintaining strict storage and compute efficiency. This work opens a new direction for building highly accurate and practical LLMs under extremely low-bit constraints.
>
---
#### [new 043] REINA: Regularized Entropy Information-Based Loss for Efficient Simultaneous Speech Translation
- **分类: cs.LG; cs.CL; eess.AS**

- **简介: 该论文旨在解决SimulST系统中翻译质量与延迟的平衡问题，提出Regularized Entropy Information-Based Loss（REINA）作为改进策略，训练非流模型并验证其有效性，取得优于现有方法的性能提升。**

- **链接: [http://arxiv.org/pdf/2508.04946v1](http://arxiv.org/pdf/2508.04946v1)**

> **作者:** Nameer Hirschkind; Joseph Liu; Mahesh Kumar Nandwana; Xiao Yu
>
> **摘要:** Simultaneous Speech Translation (SimulST) systems stream in audio while simultaneously emitting translated text or speech. Such systems face the significant challenge of balancing translation quality and latency. We introduce a strategy to optimize this tradeoff: wait for more input only if you gain information by doing so. Based on this strategy, we present Regularized Entropy INformation Adaptation (REINA), a novel loss to train an adaptive policy using an existing non-streaming translation model. We derive REINA from information theory principles and show that REINA helps push the reported Pareto frontier of the latency/quality tradeoff over prior works. Utilizing REINA, we train a SimulST model on French, Spanish and German, both from and into English. Training on only open source or synthetically generated data, we achieve state-of-the-art (SOTA) streaming results for models of comparable size. We also introduce a metric for streaming efficiency, quantitatively showing REINA improves the latency/quality trade-off by as much as 21% compared to prior approaches, normalized against non-streaming baseline BLEU scores.
>
---
#### [new 044] R-Zero: Self-Evolving Reasoning LLM from Zero Data
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出了一种基于自定义训练数据的自主进化推理LLM框架（R-Zero），解决传统方法依赖大量标注的数据瓶颈，通过两个独立模型的协同优化提升推理能力，突破了单一预训练模型的局限性。**

- **链接: [http://arxiv.org/pdf/2508.05004v1](http://arxiv.org/pdf/2508.05004v1)**

> **作者:** Chengsong Huang; Wenhao Yu; Xiaoyang Wang; Hongming Zhang; Zongxia Li; Ruosen Li; Jiaxin Huang; Haitao Mi; Dong Yu
>
> **摘要:** Self-evolving Large Language Models (LLMs) offer a scalable path toward super-intelligence by autonomously generating, refining, and learning from their own experiences. However, existing methods for training such models still rely heavily on vast human-curated tasks and labels, typically via fine-tuning or reinforcement learning, which poses a fundamental bottleneck to advancing AI systems toward capabilities beyond human intelligence. To overcome this limitation, we introduce R-Zero, a fully autonomous framework that generates its own training data from scratch. Starting from a single base LLM, R-Zero initializes two independent models with distinct roles, a Challenger and a Solver. These models are optimized separately and co-evolve through interaction: the Challenger is rewarded for proposing tasks near the edge of the Solver capability, and the Solver is rewarded for solving increasingly challenging tasks posed by the Challenger. This process yields a targeted, self-improving curriculum without any pre-existing tasks and labels. Empirically, R-Zero substantially improves reasoning capability across different backbone LLMs, e.g., boosting the Qwen3-4B-Base by +6.49 on math-reasoning benchmarks and +7.54 on general-domain reasoning benchmarks.
>
---
#### [new 045] Understanding and Mitigating Errors of LLM-Generated RTL Code
- **分类: cs.AR; cs.CL; cs.LG**

- **简介: 该论文旨在理解并缓解基于LLM生成RTL代码中的错误，解决因编程知识不足导致的错误率低的问题。通过构建领域知识库、利用RAG补充知识、设计规则检查机制及集成工具，提出改进框架以提升准确性至91.0%。**

- **链接: [http://arxiv.org/pdf/2508.05266v1](http://arxiv.org/pdf/2508.05266v1)**

> **作者:** Jiazheng Zhang; Cheng Liu; Huawei Li
>
> **备注:** 14 pages, 26 figures
>
> **摘要:** Despite the promising potential of large language model (LLM) based register-transfer-level (RTL) code generation, the overall success rate remains unsatisfactory. Errors arise from various factors, with limited understanding of specific failure causes hindering improvement. To address this, we conduct a comprehensive error analysis and manual categorization. Our findings reveal that most errors stem not from LLM reasoning limitations, but from insufficient RTL programming knowledge, poor understanding of circuit concepts, ambiguous design descriptions, or misinterpretation of complex multimodal inputs. Leveraging in-context learning, we propose targeted error correction techniques. Specifically, we construct a domain-specific knowledge base and employ retrieval-augmented generation (RAG) to supply necessary RTL knowledge. To mitigate ambiguity errors, we introduce design description rules and implement a rule-checking mechanism. For multimodal misinterpretation, we integrate external tools to convert inputs into LLM-compatible meta-formats. For remaining errors, we adopt an iterative debugging loop (simulation-error localization-correction). Integrating these techniques into an LLM-based framework significantly improves performance. We incorporate these error correction techniques into a foundational LLM-based RTL code generation framework, resulting in significantly improved performance. Experimental results show that our enhanced framework achieves 91.0\% accuracy on the VerilogEval benchmark, surpassing the baseline code generation approach by 32.7\%, demonstrating the effectiveness of our methods.
>
---
#### [new 046] Can Large Language Models Integrate Spatial Data? Empirical Insights into Reasoning Strengths and Computational Weaknesses
- **分类: cs.AI; cs.CL**

- **简介: 本研究探讨了LLMs在空间数据整合中的潜在应用，旨在解决传统规则方法效率不足的问题，通过改进方法验证其性能。**

- **链接: [http://arxiv.org/pdf/2508.05009v1](http://arxiv.org/pdf/2508.05009v1)**

> **作者:** Bin Han; Robert Wolfe; Anat Caspi; Bill Howe
>
> **摘要:** We explore the application of large language models (LLMs) to empower domain experts in integrating large, heterogeneous, and noisy urban spatial datasets. Traditional rule-based integration methods are unable to cover all edge cases, requiring manual verification and repair. Machine learning approaches require collecting and labeling of large numbers of task-specific samples. In this study, we investigate the potential of LLMs for spatial data integration. Our analysis first considers how LLMs reason about environmental spatial relationships mediated by human experience, such as between roads and sidewalks. We show that while LLMs exhibit spatial reasoning capabilities, they struggle to connect the macro-scale environment with the relevant computational geometry tasks, often producing logically incoherent responses. But when provided relevant features, thereby reducing dependence on spatial reasoning, LLMs are able to generate high-performing results. We then adapt a review-and-refine method, which proves remarkably effective in correcting erroneous initial responses while preserving accurate responses. We discuss practical implications of employing LLMs for spatial data integration in real-world contexts and outline future research directions, including post-training, multi-modal integration methods, and support for diverse data formats. Our findings position LLMs as a promising and flexible alternative to traditional rule-based heuristics, advancing the capabilities of adaptive spatial data integration.
>
---
#### [new 047] SPGISpeech 2.0: Transcribed multi-speaker financial audio for speaker-tagged transcription
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出SPGISpeech 2.0，解决跨语言语音转录任务，通过扩展数据并添加多语种标注，提升基于端到端ASR模型在财务领域的性能。**

- **链接: [http://arxiv.org/pdf/2508.05554v1](http://arxiv.org/pdf/2508.05554v1)**

> **作者:** Raymond Grossman; Taejin Park; Kunal Dhawan; Andrew Titus; Sophia Zhi; Yulia Shchadilova; Weiqing Wang; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** To be presented at Interspeech 2025
>
> **摘要:** We introduce SPGISpeech 2.0, a dataset suitable for speaker-tagged transcription in the financial domain. SPGISpeech 2.0 improves the diversity of applicable modeling tasks while maintaining the core characteristic of the original SPGISpeech dataset: audio snippets and their corresponding fully formatted text transcriptions, usable for end-to-end automatic speech recognition (ASR). SPGISpeech 2.0 consists of 3,780 additional hours of professionally transcribed earnings calls. Furthermore, the dataset contains call and speaker information for each audio snippet facilitating multi-talker ASR. We validate the utility of SPGISpeech 2.0 through improvements in speaker-tagged ASR performance of popular speech recognition models after fine-tuning on SPGISpeech 2.0. Released free for non-commercial use, we expect SPGISpeech 2.0 to foster advancements in speech recognition technologies and inspire a wide range of research applications.
>
---
#### [new 048] ConfAgents: A Conformal-Guided Multi-Agent Framework for Cost-Efficient Medical Diagnosis
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出了一种基于自适应策略的多智能体框架（Conformal-Guided），解决传统AI在医疗诊断中依赖固定策略导致的效率与战略能力不足问题。通过引入进化机制优化高阶问题解决方案，并构建EHRFlowBench基准测试，验证其显著优于现有框架，推动从工具使用者向自主任务管理者转型。**

- **链接: [http://arxiv.org/pdf/2508.04915v1](http://arxiv.org/pdf/2508.04915v1)**

> **作者:** Huiya Zhao; Yinghao Zhu; Zixiang Wang; Yasha Wang; Junyi Gao; Liantao Ma
>
> **备注:** Code: https://github.com/PKU-AICare/ConfAgents
>
> **摘要:** The efficacy of AI agents in healthcare research is hindered by their reliance on static, predefined strategies. This creates a critical limitation: agents can become better tool-users but cannot learn to become better strategic planners, a crucial skill for complex domains like healthcare. We introduce HealthFlow, a self-evolving AI agent that overcomes this limitation through a novel meta-level evolution mechanism. HealthFlow autonomously refines its own high-level problem-solving policies by distilling procedural successes and failures into a durable, strategic knowledge base. To anchor our research and facilitate reproducible evaluation, we introduce EHRFlowBench, a new benchmark featuring complex, realistic health data analysis tasks derived from peer-reviewed clinical research. Our comprehensive experiments demonstrate that HealthFlow's self-evolving approach significantly outperforms state-of-the-art agent frameworks. This work marks a necessary shift from building better tool-users to designing smarter, self-evolving task-managers, paving the way for more autonomous and effective AI for scientific discovery.
>
---
#### [new 049] Bench-2-CoP: Can We Trust Benchmarking for EU AI Compliance?
- **分类: cs.AI; cs.CL**

- **简介: 该论文旨在量化欧盟AI合规性评估中的"基准-监管缺口"，通过构建Bench-2-CoP框架解决现有评价体系对系统性风险（如控制能力缺失）的忽视问题。研究利用LLM评估194k问题，发现当前体系仅覆盖行为偏好的53.7%和28.9%，并揭示了近零覆盖率的损失场景能力，为政策优化和评估工具开发提供理论依据。**

- **链接: [http://arxiv.org/pdf/2508.05464v1](http://arxiv.org/pdf/2508.05464v1)**

> **作者:** Matteo Prandi; Vincenzo Suriani; Federico Pierucci; Marcello Galisai; Daniele Nardi; Piercosma Bisconti
>
> **摘要:** The rapid advancement of General Purpose AI (GPAI) models necessitates robust evaluation frameworks, especially with emerging regulations like the EU AI Act and its associated Code of Practice (CoP). Current AI evaluation practices depend heavily on established benchmarks, but these tools were not designed to measure the systemic risks that are the focus of the new regulatory landscape. This research addresses the urgent need to quantify this "benchmark-regulation gap." We introduce Bench-2-CoP, a novel, systematic framework that uses validated LLM-as-judge analysis to map the coverage of 194,955 questions from widely-used benchmarks against the EU AI Act's taxonomy of model capabilities and propensities. Our findings reveal a profound misalignment: the evaluation ecosystem is overwhelmingly focused on a narrow set of behavioral propensities, such as "Tendency to hallucinate" (53.7% of the corpus) and "Discriminatory bias" (28.9%), while critical functional capabilities are dangerously neglected. Crucially, capabilities central to loss-of-control scenarios, including evading human oversight, self-replication, and autonomous AI development, receive zero coverage in the entire benchmark corpus. This translates to a near-total evaluation gap for systemic risks like "Loss of Control" (0.4% coverage) and "Cyber Offence" (0.8% coverage). This study provides the first comprehensive, quantitative analysis of this gap, offering critical insights for policymakers to refine the CoP and for developers to build the next generation of evaluation tools, ultimately fostering safer and more compliant AI.
>
---
#### [new 050] Federal Reserve Communication and the COVID-19 Pandemic
- **分类: econ.GN; cs.CL; cs.IT; math.IT; q-fin.EC; stat.AP; stat.ML**

- **简介: 该论文研究美联储在疫情下的沟通策略变化，分析其应对金融稳定、市场波动等问题的反应机制，通过字典分析、情感评估和主题建模揭示新型沟通特征，并对比历史危机案例，探讨政策适应性演变。**

- **链接: [http://arxiv.org/pdf/2508.04830v1](http://arxiv.org/pdf/2508.04830v1)**

> **作者:** Jonathan Benchimol; Sophia Kazinnik; Yossi Saadon
>
> **摘要:** In this study, we examine the Federal Reserve's communication strategies during the COVID-19 pandemic, comparing them with communication during previous periods of economic stress. Using specialized dictionaries tailored to COVID-19, unconventional monetary policy (UMP), and financial stability, combined with sentiment analysis and topic modeling techniques, we identify a distinct focus in Fed communication during the pandemic on financial stability, market volatility, social welfare, and UMP, characterized by notable contextual uncertainty. Through comparative analysis, we juxtapose the Fed's communication during the COVID-19 crisis with its responses during the dot-com and global financial crises, examining content, sentiment, and timing dimensions. Our findings reveal that Fed communication and policy actions were more reactive to the COVID-19 crisis than to previous crises. Additionally, declining sentiment related to financial stability in interest rate announcements and minutes anticipated subsequent accommodative monetary policy decisions. We further document that communicating about UMP has become the "new normal" for the Fed's Federal Open Market Committee meeting minutes and Chairman's speeches since the Global Financial Crisis, reflecting an institutional adaptation in communication strategy following periods of economic distress. These findings contribute to our understanding of how central bank communication evolves during crises and how communication strategies adapt to exceptional economic circumstances.
>
---
#### [new 051] Exploring Superior Function Calls via Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文旨在通过强化学习优化函数调用任务中的策略探索与参数验证，解决传统方法在复杂动作空间和结构化推理中的不足，提出基于策略熵的创新框架，实现对86.02%准确率的突破性提升。**

- **链接: [http://arxiv.org/pdf/2508.05118v1](http://arxiv.org/pdf/2508.05118v1)**

> **作者:** Bingguang Hao; Maolin Wang; Zengzhuang Xu; Yicheng Chen; Cunyin Peng; Jinjie GU; Chenyi Zhuang
>
> **摘要:** Function calling capabilities are crucial for deploying Large Language Models in real-world applications, yet current training approaches fail to develop robust reasoning strategies. Supervised fine-tuning produces models that rely on superficial pattern matching, while standard reinforcement learning methods struggle with the complex action space of structured function calls. We present a novel reinforcement learning framework designed to enhance group relative policy optimization through strategic entropy based exploration specifically tailored for function calling tasks. Our approach addresses three critical challenges in function calling: insufficient exploration during policy learning, lack of structured reasoning in chain-of-thought generation, and inadequate verification of parameter extraction. Our two-stage data preparation pipeline ensures high-quality training samples through iterative LLM evaluation and abstract syntax tree validation. Extensive experiments on the Berkeley Function Calling Leaderboard demonstrate that this framework achieves state-of-the-art performance among open-source models with 86.02\% overall accuracy, outperforming standard GRPO by up to 6\% on complex multi-function scenarios. Notably, our method shows particularly strong improvements on code-pretrained models, suggesting that structured language generation capabilities provide an advantageous starting point for reinforcement learning in function calling tasks. We will release all the code, models and dataset to benefit the community.
>
---
#### [new 052] MELLA: Bridging Linguistic Capability and Cultural Groundedness for Low-Resource Language MLLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文旨在解决低资源语言模型在多模态与文化维度上的能力不足问题，通过构建双源数据集MELLA（结合文化相关web alt-text和语言生成描述），提升模型在不同语言环境下的泛化能力与描述质量，验证了文化与语言双重增强对低资源语言模型的有效性。**

- **链接: [http://arxiv.org/pdf/2508.05502v1](http://arxiv.org/pdf/2508.05502v1)**

> **作者:** Yufei Gao; Jiaying Fei; Nuo Chen; Ruirui Chen; Guohang Yan; Yunshi Lan; Botian Shi
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown remarkable performance in high-resource languages. However, their effectiveness diminishes significantly in the contexts of low-resource languages. Current multilingual enhancement methods are often limited to text modality or rely solely on machine translation. While such approaches help models acquire basic linguistic capabilities and produce "thin descriptions", they neglect the importance of multimodal informativeness and cultural groundedness, both of which are crucial for serving low-resource language users effectively. To bridge this gap, in this study, we identify two significant objectives for a truly effective MLLM in low-resource language settings, namely 1) linguistic capability and 2) cultural groundedness, placing special emphasis on cultural awareness. To achieve these dual objectives, we propose a dual-source strategy that guides the collection of data tailored to each goal, sourcing native web alt-text for culture and MLLM-generated captions for linguistics. As a concrete implementation, we introduce MELLA, a multimodal, multilingual dataset. Experiment results show that after fine-tuning on MELLA, there is a general performance improvement for the eight languages on various MLLM backbones, with models producing "thick descriptions". We verify that the performance gains are from both cultural knowledge enhancement and linguistic capability enhancement. Our dataset can be found at https://opendatalab.com/applyMultilingualCorpus.
>
---
#### [new 053] Cognitive Duality for Adaptive Web Agents
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文研究了Web导航中AI自主决策的双重认知机制，解决了复杂环境下的多目标行为推理问题，通过将系统1与系统2整合，构建了CogniWeb框架，实现了高效适应性学习。**

- **链接: [http://arxiv.org/pdf/2508.05081v1](http://arxiv.org/pdf/2508.05081v1)**

> **作者:** Jiarun Liu; Chunhong Zhang; Zheng Hu
>
> **摘要:** Web navigation represents a critical and challenging domain for evaluating artificial general intelligence (AGI), demanding complex decision-making within high-entropy, dynamic environments with combinatorially explosive action spaces. Current approaches to building autonomous web agents either focus on offline imitation learning or online exploration, but rarely integrate both paradigms effectively. Inspired by the dual-process theory of human cognition, we derive a principled decomposition into fast System 1 and slow System 2 cognitive processes. This decomposition provides a unifying perspective on existing web agent methodologies, bridging the gap between offline learning of intuitive reactive behaviors and online acquisition of deliberative planning capabilities. We implement this framework in CogniWeb, a modular agent architecture that adaptively toggles between fast intuitive processing and deliberate reasoning based on task complexity. Our evaluation on WebArena demonstrates that CogniWeb achieves competitive performance (43.96% success rate) while maintaining significantly higher efficiency (75% reduction in token usage).
>
---
#### [new 054] Prescriptive Agents based on Rag for Automated Maintenance (PARAM)
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; eess.SP**

- **简介: 该论文提出基于LLM的智能维护系统，解决工业设备维护效率不足的问题，通过整合振动数据分析与多智能体生成技术，实现故障诊断与结构化建议生成。**

- **链接: [http://arxiv.org/pdf/2508.04714v1](http://arxiv.org/pdf/2508.04714v1)**

> **作者:** Chitranshu Harbola; Anupam Purwar
>
> **摘要:** Industrial machinery maintenance requires timely intervention to prevent catastrophic failures and optimize operational efficiency. This paper presents an integrated Large Language Model (LLM)-based intelligent system for prescriptive maintenance that extends beyond traditional anomaly detection to provide actionable maintenance recommendations. Building upon our prior LAMP framework for numerical data analysis, we develop a comprehensive solution that combines bearing vibration frequency analysis with multi agentic generation for intelligent maintenance planning. Our approach serializes bearing vibration data (BPFO, BPFI, BSF, FTF frequencies) into natural language for LLM processing, enabling few-shot anomaly detection with high accuracy. The system classifies fault types (inner race, outer race, ball/roller, cage faults) and assesses severity levels. A multi-agentic component processes maintenance manuals using vector embeddings and semantic search, while also conducting web searches to retrieve comprehensive procedural knowledge and access up-to-date maintenance practices for more accurate and in-depth recommendations. The Gemini model then generates structured maintenance recommendations includes immediate actions, inspection checklists, corrective measures, parts requirements, and timeline specifications. Experimental validation in bearing vibration datasets demonstrates effective anomaly detection and contextually relevant maintenance guidance. The system successfully bridges the gap between condition monitoring and actionable maintenance planning, providing industrial practitioners with intelligent decision support. This work advances the application of LLMs in industrial maintenance, offering a scalable framework for prescriptive maintenance across machinery components and industrial sectors.
>
---
#### [new 055] Navigating Through Paper Flood: Advancing LLM-based Paper Evaluation through Domain-Aware Retrieval and Latent Reasoning
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出了一种基于LLM的论文评估框架PaperEval，旨在解决传统方法受限于旧领域知识和低推理能力的问题，通过领域感知检索和潜意识推理提升评估质量和准确性，并结合迭代优化策略验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.05129v1](http://arxiv.org/pdf/2508.05129v1)**

> **作者:** Wuqiang Zheng; Yiyan Xu; Xinyu Lin; Chongming Gao; Wenjie Wang; Fuli Feng
>
> **摘要:** With the rapid and continuous increase in academic publications, identifying high-quality research has become an increasingly pressing challenge. While recent methods leveraging Large Language Models (LLMs) for automated paper evaluation have shown great promise, they are often constrained by outdated domain knowledge and limited reasoning capabilities. In this work, we present PaperEval, a novel LLM-based framework for automated paper evaluation that addresses these limitations through two key components: 1) a domain-aware paper retrieval module that retrieves relevant concurrent work to support contextualized assessments of novelty and contributions, and 2) a latent reasoning mechanism that enables deep understanding of complex motivations and methodologies, along with comprehensive comparison against concurrently related work, to support more accurate and reliable evaluation. To guide the reasoning process, we introduce a progressive ranking optimization strategy that encourages the LLM to iteratively refine its predictions with an emphasis on relative comparison. Experiments on two datasets demonstrate that PaperEval consistently outperforms existing methods in both academic impact and paper quality evaluation. In addition, we deploy PaperEval in a real-world paper recommendation system for filtering high-quality papers, which has gained strong engagement on social media -- amassing over 8,000 subscribers and attracting over 10,000 views for many filtered high-quality papers -- demonstrating the practical effectiveness of PaperEval.
>
---
#### [new 056] A Novel Architecture for Symbolic Reasoning with Decision Trees and LLM Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出了一种融合符号推理与LLM能力的混合架构，旨在解决符号与神经网络耦合难题。任务为构建通用神经-符号协同推理系统，通过决策树和随机森林作为可解释的符号模块，结合LLMs进行抽象推理，优化了多智能体协作中的逻辑一致性与推理效率。**

- **链接: [http://arxiv.org/pdf/2508.05311v1](http://arxiv.org/pdf/2508.05311v1)**

> **作者:** Andrew Kiruluta
>
> **摘要:** We propose a hybrid architecture that integrates decision tree-based symbolic reasoning with the generative capabilities of large language models (LLMs) within a coordinated multi-agent framework. Unlike prior approaches that loosely couple symbolic and neural modules, our design embeds decision trees and random forests as callable oracles within a unified reasoning system. Tree-based modules enable interpretable rule inference and causal logic, while LLM agents handle abductive reasoning, generalization, and interactive planning. A central orchestrator maintains belief state consistency and mediates communication across agents and external tools, enabling reasoning over both structured and unstructured inputs. The system achieves strong performance on reasoning benchmarks. On \textit{ProofWriter}, it improves entailment consistency by +7.2\% through logic-grounded tree validation. On GSM8k, it achieves +5.3\% accuracy gains in multistep mathematical problems via symbolic augmentation. On \textit{ARC}, it boosts abstraction accuracy by +6.0\% through integration of symbolic oracles. Applications in clinical decision support and scientific discovery show how the system encodes domain rules symbolically while leveraging LLMs for contextual inference and hypothesis generation. This architecture offers a robust, interpretable, and extensible solution for general-purpose neuro-symbolic reasoning.
>
---
#### [new 057] Test-Time Reinforcement Learning for GUI Grounding via Region Consistency
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文旨在解决GUI地面任务中的高成本与低效率问题，提出基于区域一致性的Test-Time强化学习方法(GUI-RC)，通过构建投票网格提升模型精度，实现2-3%的性能提升，为自主GUI代理提供更高效的数据支持。**

- **链接: [http://arxiv.org/pdf/2508.05615v1](http://arxiv.org/pdf/2508.05615v1)**

> **作者:** Yong Du; Yuchen Yan; Fei Tang; Zhengxi Lu; Chang Zong; Weiming Lu; Shengpei Jiang; Yongliang Shen
>
> **备注:** Project Page: https://zju-real.github.io/gui-rcpo Code: https://github.com/zju-real/gui-rcpo
>
> **摘要:** Graphical User Interface (GUI) grounding, the task of mapping natural language instructions to precise screen coordinates, is fundamental to autonomous GUI agents. While existing methods achieve strong performance through extensive supervised training or reinforcement learning with labeled rewards, they remain constrained by the cost and availability of pixel-level annotations. We observe that when models generate multiple predictions for the same GUI element, the spatial overlap patterns reveal implicit confidence signals that can guide more accurate localization. Leveraging this insight, we propose GUI-RC (Region Consistency), a test-time scaling method that constructs spatial voting grids from multiple sampled predictions to identify consensus regions where models show highest agreement. Without any training, GUI-RC improves accuracy by 2-3% across various architectures on ScreenSpot benchmarks. We further introduce GUI-RCPO (Region Consistency Policy Optimization), which transforms these consistency patterns into rewards for test-time reinforcement learning. By computing how well each prediction aligns with the collective consensus, GUI-RCPO enables models to iteratively refine their outputs on unlabeled data during inference. Extensive experiments demonstrate the generality of our approach: GUI-RC boosts Qwen2.5-VL-3B-Instruct from 80.11% to 83.57% on ScreenSpot-v2, while GUI-RCPO further improves it to 85.14% through self-supervised optimization. Our approach reveals the untapped potential of test-time scaling and test-time reinforcement learning for GUI grounding, offering a promising path toward more robust and data-efficient GUI agents.
>
---
#### [new 058] JPS: Jailbreak Multimodal Large Language Models with Collaborative Visual Perturbation and Textual Steering
- **分类: cs.MM; cs.AI; cs.CL; cs.CR; I.2.7; K.4.1; K.6.5**

- **简介: 该论文旨在开发一种对抗性攻击的多模态大语言模型，解决传统方法仅关注攻击成功率（ASR）而忽略恶意意图的问题。通过协作视觉扰动与文本引导提示，结合多智能体优化策略，提出"恶意意图满足率"（MIFR）指标，并在多个模型和基准上实现突破性性能提升。**

- **链接: [http://arxiv.org/pdf/2508.05087v1](http://arxiv.org/pdf/2508.05087v1)**

> **作者:** Renmiao Chen; Shiyao Cui; Xuancheng Huang; Chengwei Pan; Victor Shea-Jay Huang; QingLin Zhang; Xuan Ouyang; Zhexin Zhang; Hongning Wang; Minlie Huang
>
> **备注:** 10 pages, 3 tables, 2 figures, to appear in the Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)
>
> **摘要:** Jailbreak attacks against multimodal large language Models (MLLMs) are a significant research focus. Current research predominantly focuses on maximizing attack success rate (ASR), often overlooking whether the generated responses actually fulfill the attacker's malicious intent. This oversight frequently leads to low-quality outputs that bypass safety filters but lack substantial harmful content. To address this gap, we propose JPS, \underline{J}ailbreak MLLMs with collaborative visual \underline{P}erturbation and textual \underline{S}teering, which achieves jailbreaks via corporation of visual image and textually steering prompt. Specifically, JPS utilizes target-guided adversarial image perturbations for effective safety bypass, complemented by "steering prompt" optimized via a multi-agent system to specifically guide LLM responses fulfilling the attackers' intent. These visual and textual components undergo iterative co-optimization for enhanced performance. To evaluate the quality of attack outcomes, we propose the Malicious Intent Fulfillment Rate (MIFR) metric, assessed using a Reasoning-LLM-based evaluator. Our experiments show JPS sets a new state-of-the-art in both ASR and MIFR across various MLLMs and benchmarks, with analyses confirming its efficacy. Codes are available at \href{https://github.com/thu-coai/JPS}{https://github.com/thu-coai/JPS}. \color{warningcolor}{Warning: This paper contains potentially sensitive contents.}
>
---
#### [new 059] Mixed-Initiative Dialog for Human-Robot Collaborative Manipulation
- **分类: cs.RO; cs.CL; cs.HC; cs.LG; cs.MA; I.2.9; I.2.7; I.2.6**

- **简介: 该论文提出了一种混合对话框架 MICoBot，解决人机协作中动态需求响应的问题，通过元计划、步骤分配和执行者决策三层机制，优化机器人与人类协作效率，验证其优于传统模型。**

- **链接: [http://arxiv.org/pdf/2508.05535v1](http://arxiv.org/pdf/2508.05535v1)**

> **作者:** Albert Yu; Chengshu Li; Luca Macesanu; Arnav Balaji; Ruchira Ray; Raymond Mooney; Roberto Martín-Martín
>
> **备注:** Project website at https://robin-lab.cs.utexas.edu/MicoBot/
>
> **摘要:** Effective robotic systems for long-horizon human-robot collaboration must adapt to a wide range of human partners, whose physical behavior, willingness to assist, and understanding of the robot's capabilities may change over time. This demands a tightly coupled communication loop that grants both agents the flexibility to propose, accept, or decline requests as they coordinate toward completing the task effectively. We apply a Mixed-Initiative dialog paradigm to Collaborative human-roBot teaming and propose MICoBot, a system that handles the common scenario where both agents, using natural language, take initiative in formulating, accepting, or rejecting proposals on who can best complete different steps of a task. To handle diverse, task-directed dialog, and find successful collaborative strategies that minimize human effort, MICoBot makes decisions at three levels: (1) a meta-planner considers human dialog to formulate and code a high-level collaboration strategy, (2) a planner optimally allocates the remaining steps to either agent based on the robot's capabilities (measured by a simulation-pretrained affordance model) and the human's estimated availability to help, and (3) an action executor decides the low-level actions to perform or words to say to the human. Our extensive evaluations in simulation and real-world -- on a physical robot with 18 unique human participants over 27 hours -- demonstrate the ability of our method to effectively collaborate with diverse human users, yielding significantly improved task success and user experience than a pure LLM baseline and other agent allocation models. See additional videos and materials at https://robin-lab.cs.utexas.edu/MicoBot/.
>
---
#### [new 060] Fine-Tuning Small Language Models (SLMs) for Autonomous Web-based Geographical Information Systems (AWebGIS)
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文探讨如何利用小语言模型（SLMs）实现自驱动Web GIS系统，旨在解决传统云基LLMs在隐私、可扩展性等方面的问题，通过三种方法（在线、离线半自动化、客户端本地化）验证其有效性，最终证明浏览器端执行模型可提升系统的可行性与性能。**

- **链接: [http://arxiv.org/pdf/2508.04846v1](http://arxiv.org/pdf/2508.04846v1)**

> **作者:** Mahdi Nazari Ashani; Ali Asghar Alesheikh; Saba Kazemi; Kimya Kheirkhah; Yasin Mohammadi; Fatemeh Rezaie; Amir Mahdi Manafi; Hedieh Zarkesh
>
> **摘要:** Autonomous web-based geographical information systems (AWebGIS) aim to perform geospatial operations from natural language input, providing intuitive, intelligent, and hands-free interaction. However, most current solutions rely on cloud-based large language models (LLMs), which require continuous internet access and raise users' privacy and scalability issues due to centralized server processing. This study compares three approaches to enabling AWebGIS: (1) a fully-automated online method using cloud-based LLMs (e.g., Cohere); (2) a semi-automated offline method using classical machine learning classifiers such as support vector machine and random forest; and (3) a fully autonomous offline (client-side) method based on a fine-tuned small language model (SLM), specifically T5-small model, executed in the client's web browser. The third approach, which leverages SLMs, achieved the highest accuracy among all methods, with an exact matching accuracy of 0.93, Levenshtein similarity of 0.99, and recall-oriented understudy for gisting evaluation ROUGE-1 and ROUGE-L scores of 0.98. Crucially, this client-side computation strategy reduces the load on backend servers by offloading processing to the user's device, eliminating the need for server-based inference. These results highlight the feasibility of browser-executable models for AWebGIS solutions.
>
---
#### [new 061] Advancing Hate Speech Detection with Transformers: Insights from the MetaHate
- **分类: cs.LG; cs.CL**

- **简介: 该论文旨在利用Transformer模型提升 hate speech 检测能力，解决大规模社交网络中仇恨言论识别的问题，并通过 MetaHate 数据集验证了 ELECTRA 模型的最高 F1 分数（0.8980）。**

- **链接: [http://arxiv.org/pdf/2508.04913v1](http://arxiv.org/pdf/2508.04913v1)**

> **作者:** Santosh Chapagain; Shah Muhammad Hamdi; Soukaina Filali Boubrahimi
>
> **备注:** Accepted to the Deviant Dynamics in Digital Spaces workshop at ASONAM 2025
>
> **摘要:** Hate speech is a widespread and harmful form of online discourse, encompassing slurs and defamatory posts that can have serious social, psychological, and sometimes physical impacts on targeted individuals and communities. As social media platforms such as X (formerly Twitter), Facebook, Instagram, Reddit, and others continue to facilitate widespread communication, they also become breeding grounds for hate speech, which has increasingly been linked to real-world hate crimes. Addressing this issue requires the development of robust automated methods to detect hate speech in diverse social media environments. Deep learning approaches, such as vanilla recurrent neural networks (RNNs), long short-term memory (LSTM), and convolutional neural networks (CNNs), have achieved good results, but are often limited by issues such as long-term dependencies and inefficient parallelization. This study represents the comprehensive exploration of transformer-based models for hate speech detection using the MetaHate dataset--a meta-collection of 36 datasets with 1.2 million social media samples. We evaluate multiple state-of-the-art transformer models, including BERT, RoBERTa, GPT-2, and ELECTRA, with fine-tuned ELECTRA achieving the highest performance (F1 score: 0.8980). We also analyze classification errors, revealing challenges with sarcasm, coded language, and label noise.
>
---
#### [new 062] A Study of the Framework and Real-World Applications of Language Embedding for 3D Scene Understanding
- **分类: cs.GR; cs.CL; cs.CV**

- **简介: 该论文研究了将语言嵌入与3D场景理解结合的方法，旨在解决计算效率低、数据标注不足等问题。通过结构化综述分析，探讨了语言引导下的3D Gaussian Splatting技术及其在文本生成、场景理解等领域的应用。**

- **链接: [http://arxiv.org/pdf/2508.05064v1](http://arxiv.org/pdf/2508.05064v1)**

> **作者:** Mahmoud Chick Zaouali; Todd Charter; Yehor Karpichev; Brandon Haworth; Homayoun Najjjaran
>
> **摘要:** Gaussian Splatting has rapidly emerged as a transformative technique for real-time 3D scene representation, offering a highly efficient and expressive alternative to Neural Radiance Fields (NeRF). Its ability to render complex scenes with high fidelity has enabled progress across domains such as scene reconstruction, robotics, and interactive content creation. More recently, the integration of Large Language Models (LLMs) and language embeddings into Gaussian Splatting pipelines has opened new possibilities for text-conditioned generation, editing, and semantic scene understanding. Despite these advances, a comprehensive overview of this emerging intersection has been lacking. This survey presents a structured review of current research efforts that combine language guidance with 3D Gaussian Splatting, detailing theoretical foundations, integration strategies, and real-world use cases. We highlight key limitations such as computational bottlenecks, generalizability, and the scarcity of semantically annotated 3D Gaussian data and outline open challenges and future directions for advancing language-guided 3D scene understanding using Gaussian Splatting.
>
---
#### [new 063] QA-Dragon: Query-Aware Dynamic RAG System for Knowledge-Intensive Visual Question Answering
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究了一个查询驱动的动态RAG系统，解决复杂视觉问答任务中单一源检索的局限性，通过引入领域路由和搜索路由实现多模态、多轮、多跳推理，显著提升VQA任务的准确性与知识覆盖度（5.06%-6.35%）。**

- **链接: [http://arxiv.org/pdf/2508.05197v1](http://arxiv.org/pdf/2508.05197v1)**

> **作者:** Zhuohang Jiang; Pangjing Wu; Xu Yuan; Wenqi Fan; Qing Li
>
> **备注:** The source code for our system is released in https://github.com/jzzzzh/QA-Dragon
>
> **摘要:** Retrieval-Augmented Generation (RAG) has been introduced to mitigate hallucinations in Multimodal Large Language Models (MLLMs) by incorporating external knowledge into the generation process, and it has become a widely adopted approach for knowledge-intensive Visual Question Answering (VQA). However, existing RAG methods typically retrieve from either text or images in isolation, limiting their ability to address complex queries that require multi-hop reasoning or up-to-date factual knowledge. To address this limitation, we propose QA-Dragon, a Query-Aware Dynamic RAG System for Knowledge-Intensive VQA. Specifically, QA-Dragon introduces a domain router to identify the query's subject domain for domain-specific reasoning, along with a search router that dynamically selects optimal retrieval strategies. By orchestrating both text and image search agents in a hybrid setup, our system supports multimodal, multi-turn, and multi-hop reasoning, enabling it to tackle complex VQA tasks effectively. We evaluate our QA-Dragon on the Meta CRAG-MM Challenge at KDD Cup 2025, where it significantly enhances the reasoning performance of base models under challenging scenarios. Our framework achieves substantial improvements in both answer accuracy and knowledge overlap scores, outperforming baselines by 5.06% on the single-source task, 6.35% on the multi-source task, and 5.03% on the multi-turn task.
>
---
#### [new 064] Iterative Learning of Computable Phenotypes for Treatment Resistant Hypertension using Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文旨在开发可解释的计算可变性（CPs）以提升治疗耐药性高血压的临床决策支持，通过大语言模型（LLM）的迭代学习优化，解决了传统模型对复杂表型建模能力不足的问题，实现了性能与训练样本的平衡。**

- **链接: [http://arxiv.org/pdf/2508.05581v1](http://arxiv.org/pdf/2508.05581v1)**

> **作者:** Guilherme Seidyo Imai Aldeia; Daniel S. Herman; William G. La Cava
>
> **备注:** To appear in PMLR, Volume 298, Machine Learning for Healthcare, 2025
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities for medical question answering and programming, but their potential for generating interpretable computable phenotypes (CPs) is under-explored. In this work, we investigate whether LLMs can generate accurate and concise CPs for six clinical phenotypes of varying complexity, which could be leveraged to enable scalable clinical decision support to improve care for patients with hypertension. In addition to evaluating zero-short performance, we propose and test a synthesize, execute, debug, instruct strategy that uses LLMs to generate and iteratively refine CPs using data-driven feedback. Our results show that LLMs, coupled with iterative learning, can generate interpretable and reasonably accurate programs that approach the performance of state-of-the-art ML methods while requiring significantly fewer training examples.
>
---
#### [new 065] FAITH: A Framework for Assessing Intrinsic Tabular Hallucinations in finance
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出了一种评估金融领域表单虚假假设的任务框架，旨在解决现有基准难以捕捉的财务数据依赖问题。研究通过构建基于掩码策略的数据集并评估LLMs的表现，验证了其在金融场景中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.05201v1](http://arxiv.org/pdf/2508.05201v1)**

> **作者:** Mengao Zhang; Jiayu Fu; Tanya Warrier; Yuwen Wang; Tianhui Tan; Ke-wei Huang
>
> **备注:** 9 pages
>
> **摘要:** Hallucination remains a critical challenge for deploying Large Language Models (LLMs) in finance. Accurate extraction and precise calculation from tabular data are essential for reliable financial analysis, since even minor numerical errors can undermine decision-making and regulatory compliance. Financial applications have unique requirements, often relying on context-dependent, numerical, and proprietary tabular data that existing hallucination benchmarks rarely capture. In this study, we develop a rigorous and scalable framework for evaluating intrinsic hallucinations in financial LLMs, conceptualized as a context-aware masked span prediction task over real-world financial documents. Our main contributions are: (1) a novel, automated dataset creation paradigm using a masking strategy; (2) a new hallucination evaluation dataset derived from S&P 500 annual reports; and (3) a comprehensive evaluation of intrinsic hallucination patterns in state-of-the-art LLMs on financial tabular data. Our work provides a robust methodology for in-house LLM evaluation and serves as a critical step toward building more trustworthy and reliable financial Generative AI systems.
>
---
#### [new 066] Speech LLMs in Low-Resource Scenarios: Data Volume Requirements and the Impact of Pretraining on High-Resource Languages
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文探讨了低资源场景下语音大语言模型的应用，旨在解决自动语音识别（ASR）中的数据不足问题，通过构建轻量级投影器与多语言预训练模型优化性能，验证了数据稀缺对模型效果的影响。**

- **链接: [http://arxiv.org/pdf/2508.05149v1](http://arxiv.org/pdf/2508.05149v1)**

> **作者:** Seraphina Fong; Marco Matassoni; Alessio Brutti
>
> **备注:** Accepted at Interspeech 2025. 5 pages, 2 figures, 3 tables
>
> **摘要:** Large language models (LLMs) have demonstrated potential in handling spoken inputs for high-resource languages, reaching state-of-the-art performance in various tasks. However, their applicability is still less explored in low-resource settings. This work investigates the use of Speech LLMs for low-resource Automatic Speech Recognition using the SLAM-ASR framework, where a trainable lightweight projector connects a speech encoder and a LLM. Firstly, we assess training data volume requirements to match Whisper-only performance, re-emphasizing the challenges of limited data. Secondly, we show that leveraging mono- or multilingual projectors pretrained on high-resource languages reduces the impact of data scarcity, especially with small training sets. Using multilingual LLMs (EuroLLM, Salamandra) with whisper-large-v3-turbo, we evaluate performance on several public benchmarks, providing insights for future research on optimizing Speech LLMs for low-resource languages and multilinguality.
>
---
#### [new 067] Posterior-GRPO: Rewarding Reasoning Processes in Code Generation
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文旨在解决代码生成中传统基于结果的奖励机制难以捕捉中间推理质量的问题。通过构建LCB-RB基准并引入OD-based奖励优化方法，提出Posterior-GRPO，将过程质量纳入RL决策，实现对成功输出的奖励定向，有效对抗奖励黑客现象，同时在数学等多任务场景上取得优异性能。**

- **链接: [http://arxiv.org/pdf/2508.05170v1](http://arxiv.org/pdf/2508.05170v1)**

> **作者:** Lishui Fan; Yu Zhang; Mouxiang Chen; Zhongxin Liu
>
> **摘要:** Reinforcement learning (RL) has significantly advanced code generation for large language models (LLMs). However, current paradigms rely on outcome-based rewards from test cases, neglecting the quality of the intermediate reasoning process. While supervising the reasoning process directly is a promising direction, it is highly susceptible to reward hacking, where the policy model learns to exploit the reasoning reward signal without improving final outcomes. To address this, we introduce a unified framework that can effectively incorporate the quality of the reasoning process during RL. First, to enable reasoning evaluation, we develop LCB-RB, a benchmark comprising preference pairs of superior and inferior reasoning processes. Second, to accurately score reasoning quality, we introduce an Optimized-Degraded based (OD-based) method for reward model training. This method generates high-quality preference pairs by systematically optimizing and degrading initial reasoning paths along curated dimensions of reasoning quality, such as factual accuracy, logical rigor, and coherence. A 7B parameter reward model with this method achieves state-of-the-art (SOTA) performance on LCB-RB and generalizes well to other benchmarks. Finally, we introduce Posterior-GRPO (P-GRPO), a novel RL method that conditions process-based rewards on task success. By selectively applying rewards to the reasoning processes of only successful outcomes, P-GRPO effectively mitigates reward hacking and aligns the model's internal reasoning with final code correctness. A 7B parameter model with P-GRPO achieves superior performance across diverse code generation tasks, outperforming outcome-only baselines by 4.5%, achieving comparable performance to GPT-4-Turbo. We further demonstrate the generalizability of our approach by extending it to mathematical tasks. Our models, dataset, and code are publicly available.
>
---
#### [new 068] Aligning LLMs on a Budget: Inference-Time Alignment with Heuristic Reward Models
- **分类: cs.LG; cs.AI; cs.CL; I.2.7; I.2.6; I.2.8**

- **简介: 该论文旨在解决LLM对齐任务在预算约束下的平衡问题，提出HIA方法通过轻量提示优化器、启发式奖励模型和两阶段过滤，在降低计算开销的同时保持对齐质量，适用于低预算场景的LLM部署。**

- **链接: [http://arxiv.org/pdf/2508.05165v1](http://arxiv.org/pdf/2508.05165v1)**

> **作者:** Mason Nakamura; Saaduddin Mahmud; Kyle H. Wray; Hamed Zamani; Shlomo Zilberstein
>
> **摘要:** Aligning LLMs with user preferences is crucial for real-world use but often requires costly fine-tuning or expensive inference, forcing trade-offs between alignment quality and computational cost. Existing inference-time methods typically ignore this balance, focusing solely on the optimized policy's performance. We propose HIA (Heuristic-Guided Inference-time Alignment), a tuning-free, black-box-compatible approach that uses a lightweight prompt optimizer, heuristic reward models, and two-stage filtering to reduce inference calls while preserving alignment quality. On real-world prompt datasets, HelpSteer and ComPRed, HIA outperforms best-of-N sampling, beam search, and greedy search baselines in multi-objective, goal-conditioned tasks under the same inference budget. We also find that HIA is effective under low-inference budgets with as little as one or two response queries, offering a practical solution for scalable, personalized LLM deployment.
>
---
#### [new 069] Making Prompts First-Class Citizens for Adaptive LLM Pipelines
- **分类: cs.DB; cs.AI; cs.CL**

- **简介: 该论文提出SPEAR系统，解决传统LLM管道中提示管理碎片化的问题，通过结构化、动态优化的prompt设计实现运行时提示控制与逻辑建模。**

- **链接: [http://arxiv.org/pdf/2508.05012v1](http://arxiv.org/pdf/2508.05012v1)**

> **作者:** Ugur Cetintemel; Shu Chen; Alexander W. Lee; Deepti Raghavan
>
> **摘要:** Modern LLM pipelines increasingly resemble data-centric systems: they retrieve external context, compose intermediate outputs, validate results, and adapt based on runtime feedback. Yet, the central element guiding this process -- the prompt -- remains a brittle, opaque string, disconnected from the surrounding dataflow. This disconnect limits reuse, optimization, and runtime control. In this paper, we describe our vision and an initial design for SPEAR, a language and runtime that fills this prompt management gap by making prompts structured, adaptive, and first-class components of the execution model. SPEAR enables (1) runtime prompt refinement -- modifying prompts dynamically in response to execution-time signals such as confidence, latency, or missing context; and (2) structured prompt management -- organizing prompt fragments into versioned views with support for introspection and logging. SPEAR defines a prompt algebra that governs how prompts are constructed and adapted within a pipeline. It supports multiple refinement modes (manual, assisted, and automatic), giving developers a balance between control and automation. By treating prompt logic as structured data, SPEAR enables optimizations such as operator fusion, prefix caching, and view reuse. Preliminary experiments quantify the behavior of different refinement modes compared to static prompts and agentic retries, as well as the impact of prompt-level optimizations such as operator fusion.
>
---
#### [new 070] Can Large Language Models Generate Effective Datasets for Emotion Recognition in Conversations?
- **分类: cs.AI; cs.CL**

- **简介: 该论文旨在通过L型语言模型生成具有多样性的情感识别 ERC 数据集，解决现有数据稀缺性与标注偏差问题，补充基准并分析标签不平衡对模型的影响，提升 ERC 模型性能。**

- **链接: [http://arxiv.org/pdf/2508.05474v1](http://arxiv.org/pdf/2508.05474v1)**

> **作者:** Burak Can Kaplan; Hugo Cesar De Castro Carneiro; Stefan Wermter
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Emotion recognition in conversations (ERC) focuses on identifying emotion shifts within interactions, representing a significant step toward advancing machine intelligence. However, ERC data remains scarce, and existing datasets face numerous challenges due to their highly biased sources and the inherent subjectivity of soft labels. Even though Large Language Models (LLMs) have demonstrated their quality in many affective tasks, they are typically expensive to train, and their application to ERC tasks--particularly in data generation--remains limited. To address these challenges, we employ a small, resource-efficient, and general-purpose LLM to synthesize ERC datasets with diverse properties, supplementing the three most widely used ERC benchmarks. We generate six novel datasets, with two tailored to enhance each benchmark. We evaluate the utility of these datasets to (1) supplement existing datasets for ERC classification, and (2) analyze the effects of label imbalance in ERC. Our experimental results indicate that ERC classifier models trained on the generated datasets exhibit strong robustness and consistently achieve statistically significant performance improvements on existing ERC benchmarks.
>
---
#### [new 071] Uni-cot: Towards Unified Chain-of-Thought Reasoning Across Text and Vision
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出 Uni-CoT 框架，解决跨模态推理（文本与图像）中因视觉状态过渡复杂性导致的方法局限问题。通过宏级高阶任务规划与微级子任务执行结合的双重推理机制及结构化训练策略，实现高效多模态推理，实验表明其在推理驱动生成与编辑任务中达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2508.05606v1](http://arxiv.org/pdf/2508.05606v1)**

> **作者:** Luozheng Qin; Jia Gong; Yuqing Sun; Tianjiao Li; Mengping Yang; Xiaomeng Yang; Chao Qu; Zhiyu Tan; Hao Li
>
> **备注:** https://sais-fuxi.github.io/projects/uni-cot/
>
> **摘要:** Chain-of-Thought (CoT) reasoning has been widely adopted to enhance Large Language Models (LLMs) by decomposing complex tasks into simpler, sequential subtasks. However, extending CoT to vision-language reasoning tasks remains challenging, as it often requires interpreting transitions of visual states to support reasoning. Existing methods often struggle with this due to limited capacity of modeling visual state transitions or incoherent visual trajectories caused by fragmented architectures. To overcome these limitations, we propose Uni-CoT, a Unified Chain-of-Thought framework that enables coherent and grounded multimodal reasoning within a single unified model. The key idea is to leverage a model capable of both image understanding and generation to reason over visual content and model evolving visual states. However, empowering a unified model to achieve that is non-trivial, given the high computational cost and the burden of training. To address this, Uni-CoT introduces a novel two-level reasoning paradigm: A Macro-Level CoT for high-level task planning and A Micro-Level CoT for subtask execution. This design significantly reduces the computational overhead. Furthermore, we introduce a structured training paradigm that combines interleaved image-text supervision for macro-level CoT with multi-task objectives for micro-level CoT. Together, these innovations allow Uni-CoT to perform scalable and coherent multi-modal reasoning. Furthermore, thanks to our design, all experiments can be efficiently completed using only 8 A100 GPUs with 80GB VRAM each. Experimental results on reasoning-driven image generation benchmark (WISE) and editing benchmarks (RISE and KRIS) indicates that Uni-CoT demonstrates SOTA performance and strong generalization, establishing Uni-CoT as a promising solution for multi-modal reasoning. Project Page and Code: https://sais-fuxi.github.io/projects/uni-cot/
>
---
## 更新

#### [replaced 001] Improving Factuality for Dialogue Response Generation via Graph-Based Knowledge Augmentation
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.12496v2](http://arxiv.org/pdf/2506.12496v2)**

> **作者:** Xiangyan Chen; Yujian Gan; Yimeng Gu; Matthew Purver
>
> **摘要:** Large Language Models (LLMs) succeed in many natural language processing tasks. However, their tendency to hallucinate - generate plausible but inconsistent or factually incorrect text - can cause significant problems in certain tasks, including response generation in dialogue. To mitigate this issue, we propose two novel graph knowledge-augmented frameworks, Dialogue Response Generation via Textualised Graphs (TG-DRG) and Graph-Aware Dialogue Response Generation (GA-DRG), which combine reasoning-guided dialogue reformulation, dialogue sense knowledge selection, and graph-enhanced response generation to improve the factuality of dialogue responses. To evaluate the factuality of generated responses, we propose a dialogue fact score that addresses the limitations of existing fact-score methods in dialogue settings, providing a more reliable assessment of factual consistency. We evaluate our methods using different baselines on the OpendialKG and HybriDialogue datasets. Our methods noticeably improve factuality compared to other graph knowledge-augmentation baselines, including the state-of-the-art G-retriever, achieving improvements of 3.47% on OpendialKG and 3.12% on HybriDialogue in terms of dialogue fact score. The code will be released on GitHub.
>
---
#### [replaced 002] A Latent-Variable Model for Intrinsic Probing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2201.08214v4](http://arxiv.org/pdf/2201.08214v4)**

> **作者:** Karolina Stańczak; Lucas Torroba Hennigen; Adina Williams; Ryan Cotterell; Isabelle Augenstein
>
> **摘要:** The success of pre-trained contextualized representations has prompted researchers to analyze them for the presence of linguistic information. Indeed, it is natural to assume that these pre-trained representations do encode some level of linguistic knowledge as they have brought about large empirical improvements on a wide variety of NLP tasks, which suggests they are learning true linguistic generalization. In this work, we focus on intrinsic probing, an analysis technique where the goal is not only to identify whether a representation encodes a linguistic attribute but also to pinpoint where this attribute is encoded. We propose a novel latent-variable formulation for constructing intrinsic probes and derive a tractable variational approximation to the log-likelihood. Our results show that our model is versatile and yields tighter mutual information estimates than two intrinsic probes previously proposed in the literature. Finally, we find empirical evidence that pre-trained representations develop a cross-lingually entangled notion of morphosyntax.
>
---
#### [replaced 003] GM-PRM: A Generative Multimodal Process Reward Model for Multimodal Mathematical Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.04088v2](http://arxiv.org/pdf/2508.04088v2)**

> **作者:** Jianghangfan Zhang; Yibo Yan; Kening Zheng; Xin Zou; Song Dai; Xuming Hu
>
> **摘要:** Multimodal Large Language Models (MLLMs) demonstrate remarkable capabilities but often struggle with complex, multi-step mathematical reasoning, where minor errors in visual perception or logical deduction can lead to complete failure. While Process Reward Models (PRMs) offer step-by-step supervision, existing multimodal PRMs are limited to being binary verifiers that can identify but not correct errors, offering little explanatory power. To address these deficiencies, we introduce the Generative Multimodal Process Reward Model (GM-PRM), a novel paradigm that transforms the PRM from a passive judge into an active reasoning collaborator. Instead of a simple scalar score, GM-PRM provides a fine-grained, interpretable analysis of each reasoning step, evaluating its step intent, visual alignment, and logical soundness. More critically, GM-PRM is trained to generate a corrected version of the first erroneous step it identifies. This unique corrective capability enables our new test-time inference strategy, Refined Best-of-N (Refined-BoN). This framework actively enhances solution quality by using the PRM's generated correction to guide the policy model toward a more promising reasoning trajectory, thereby improving the diversity and correctness of the solution pool. We demonstrate that GM-PRM achieves state-of-the-art results on multiple multimodal math benchmarks, significantly boosting policy model performance with remarkable data efficiency, requiring only a 20K-sample training dataset. Our code will be released upon acceptance.
>
---
#### [replaced 004] LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.03440v3](http://arxiv.org/pdf/2508.03440v3)**

> **作者:** Chünhung Wu; Jinliang Lu; Zixuan Ren; Gangqiang Hu; Zhi Wu; Dai Dai; Hua Wu
>
> **备注:** 11 pages, 7 figures, working in progress
>
> **摘要:** Human cognition naturally engages with abstract and fluid concepts, whereas existing reasoning models often rely on generating discrete tokens, potentially constraining their expressive capabilities. Recent advancements aim to address this limitation by enabling large language models (LLMs) to generate soft, abstract tokens, thus facilitating reasoning within a continuous concept space. This paper explores the `Soft Thinking' capabilities of various LLMs by examining the models' internal behavior using a suite of probing techniques. Contrary to the common belief that Soft Thinking enables the simultaneous exploration of diverse reasoning paths, our findings reveal that LLMs predominantly rely on the most influential component of the soft inputs during subsequent decoding steps. This reliance hinders the exploration of different reasoning paths and reduces vanilla Soft Thinking to a form of greedy decoding, obscuring the advantage of transmitting more information through Soft Tokens. To tackle this issue, we explore sampling strategies to introduce \emph{randomness}, employing methods such as Dirichlet resampling and the Gumbel-Softmax trick. Our experiments demonstrate that incorporating randomness can alleviate the limitations of vanilla approaches and unleash the potential of Soft Thinking. Notably, the Gumbel-Softmax trick provides adequate randomness with controlled smoothness, resulting in superior performance across eight reasoning benchmarks.
>
---
#### [replaced 005] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.02088v2](http://arxiv.org/pdf/2507.02088v2)**

> **作者:** Tian Lan; Xiangdong Su; Xu Liu; Ruirui Wang; Ke Chang; Jiang Li; Guanglai Gao
>
> **备注:** Accepted by ACL2025 Findings
>
> **摘要:** As large language models (LLMs) are increasingly applied to various NLP tasks, their inherent biases are gradually disclosed. Therefore, measuring biases in LLMs is crucial to mitigate its ethical risks. However, most existing bias evaluation datasets focus on English and North American culture, and their bias categories are not fully applicable to other cultures. The datasets grounded in the Chinese language and culture are scarce. More importantly, these datasets usually only support single evaluation tasks and cannot evaluate the bias from multiple aspects in LLMs. To address these issues, we present a Multi-task Chinese Bias Evaluation Benchmark (McBE) that includes 4,077 bias evaluation instances, covering 12 single bias categories, 82 subcategories and introducing 5 evaluation tasks, providing extensive category coverage, content diversity, and measuring comprehensiveness. Additionally, we evaluate several popular LLMs from different series and with parameter sizes. In general, all these LLMs demonstrated varying degrees of bias. We conduct an in-depth analysis of results, offering novel insights into bias in LLMs.
>
---
#### [replaced 006] FinCoT: Grounding Chain-of-Thought in Expert Financial Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.16123v3](http://arxiv.org/pdf/2506.16123v3)**

> **作者:** Natapong Nitarach; Warit Sirichotedumrong; Panop Pitchayarthorn; Pittawat Taveekitworachai; Potsawee Manakul; Kunat Pipatanakul
>
> **摘要:** This paper presents FinCoT, a structured chain-of-thought (CoT) prompting framework that embeds domain-specific expert financial reasoning blueprints to guide large language models' behaviors. We identify three main prompting styles in financial NLP (FinNLP): (1) standard prompting (zero-shot), (2) unstructured CoT (free-form reasoning), and (3) structured CoT (with explicitly structured reasoning steps). Prior work has mainly focused on the first two, while structured CoT remains underexplored and lacks domain expertise incorporation. Therefore, we evaluate all three prompting approaches across ten CFA-style financial domains and introduce FinCoT as the first structured finance-specific prompting approach incorporating blueprints from domain experts. FinCoT improves the accuracy of a general-purpose model, Qwen3-8B-Base, from 63.2% to 80.5%, and boosts Fin-R1 (7B), a finance-specific model, from 65.7% to 75.7%, while reducing output length by up to 8.9x and 1.16x compared to structured CoT methods, respectively. We find that FinCoT proves most effective for models lacking financial post-training. Our findings show that FinCoT does not only improve performance and reduce inference costs but also yields more interpretable and expert-aligned reasoning traces.
>
---
#### [replaced 007] From Code to Correctness: Closing the Last Mile of Code Generation with Hierarchical Debugging
- **分类: cs.CL; cs.AI; cs.PL; cs.SE**

- **链接: [http://arxiv.org/pdf/2410.01215v3](http://arxiv.org/pdf/2410.01215v3)**

> **作者:** Yuling Shi; Songsong Wang; Chengcheng Wan; Min Wang; Xiaodong Gu
>
> **备注:** Code and data available at https://github.com/YerbaPage/MGDebugger
>
> **摘要:** While large language models have made significant strides in code generation, the pass rate of the generated code is bottlenecked on subtle errors, often requiring human intervention to pass tests, especially for complex problems. Existing LLM-based debugging systems treat generated programs as monolithic units, failing to address bugs at multiple levels of granularity, from low-level syntax errors to high-level algorithmic flaws. In this paper, we introduce Multi-Granularity Debugger (MGDebugger), a hierarchical code debugger by isolating, identifying, and resolving bugs at various levels of granularity. MGDebugger decomposes problematic code into a hierarchical tree structure of subfunctions, with each level representing a particular granularity of error. During debugging, it analyzes each subfunction and iteratively resolves bugs in a bottom-up manner. To effectively test each subfunction, we propose an LLM-simulated Python executor, which traces code execution and tracks important variable states to pinpoint errors accurately. Extensive experiments demonstrate that MGDebugger outperforms existing debugging systems, achieving an 18.9% improvement in accuracy over seed generations in HumanEval and a 97.6% repair success rate in HumanEvalFix. Furthermore, MGDebugger effectively fixes bugs across different categories and difficulty levels, demonstrating its robustness and effectiveness.
>
---
#### [replaced 008] Enabling On-Device Medical AI Assistants via Input-Driven Saliency Adaptation
- **分类: cs.CL; cs.AI; cs.AR; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.11105v3](http://arxiv.org/pdf/2506.11105v3)**

> **作者:** Uttej Kallakurik; Edward Humes; Rithvik Jonna; Xiaomin Lin; Tinoosh Mohsenin
>
> **备注:** Accepted for publication in the Proceedings of IEEE BioCAS 2025
>
> **摘要:** Large Language Models (LLMs) have significant impact on the healthcare scenarios but remain prohibitively large for deployment in real-time, resource-constrained environments such as edge devices. In this work, we introduce a novel medical assistant system, optimized through our general-purpose compression framework, which tailors Large Language Models (LLMs) for deployment in specialized domains. By measuring neuron saliency on domain-specific data, our method can aggressively prune irrelevant neurons, reducing model size while preserving performance. Following pruning, we apply post-training quantization to further reduce the memory footprint, and evaluate the compressed model across medical benchmarks including MedMCQA, MedQA, and PubMedQA. We also deploy the 50\% compressed Gemma and the 67\% compressed LLaMA3 models on Jetson Orin Nano (18.7W peak) and Raspberry Pi 5 (6.3W peak), achieving real-time, energy-efficient inference under hardware constraints.
>
---
#### [replaced 009] Probabilities of Chat LLMs Are Miscalibrated but Still Predict Correctness on Multiple-Choice Q&A
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.13213v4](http://arxiv.org/pdf/2402.13213v4)**

> **作者:** Benjamin Plaut; Nguyen X. Khanh; Tu Trinh
>
> **备注:** Published in Transactions on Machine Learning Research (TMLR)
>
> **摘要:** We study 15 large language models (LLMs) fine-tuned for chat and find that their maximum softmax probabilities (MSPs) are consistently miscalibrated on multiple-choice Q&A. However, those MSPs might still encode useful uncertainty information. Specifically, we hypothesized that wrong answers would be associated with smaller MSPs compared to correct answers. Via rigorous statistical testing, we show that this hypothesis holds for models which perform well on the underlying Q&A task. We also find a strong direction correlation between Q&A accuracy and MSP correctness prediction, while finding no correlation between Q&A accuracy and calibration error. This suggests that within the current fine-tuning paradigm, we can expect correctness prediction but not calibration to improve as LLM capabilities progress. To demonstrate the utility of correctness prediction, we show that when models have the option to abstain, performance can be improved by selectively abstaining based on the MSP of the initial model response, using only a small amount of labeled data to choose the MSP threshold.
>
---
#### [replaced 010] The SMeL Test: A simple benchmark for media literacy in language models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.02074v2](http://arxiv.org/pdf/2508.02074v2)**

> **作者:** Gustaf Ahdritz; Anat Kleiman
>
> **摘要:** The internet is rife with unattributed, deliberately misleading, or otherwise untrustworthy content. Though large language models (LLMs) are often tasked with autonomous web browsing, the extent to which they have learned the simple heuristics human researchers use to navigate this noisy environment is not currently known. In this paper, we introduce the Synthetic Media Literacy Test (SMeL Test), a minimal benchmark that tests the ability of language models to actively filter out untrustworthy information in context. We benchmark a variety of commonly used instruction-tuned LLMs, including reasoning models, and find that no model consistently succeeds; while reasoning in particular is associated with higher scores, even the best API model we test hallucinates up to 70% of the time. Remarkably, larger and more capable models do not necessarily outperform their smaller counterparts. We hope our work sheds more light on this important form of hallucination and guides the development of new methods to combat it.
>
---
#### [replaced 011] Teaching LLMs How to Learn with Contextual Fine-Tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.09032v2](http://arxiv.org/pdf/2503.09032v2)**

> **作者:** Younwoo Choi; Muhammad Adil Asif; Ziwen Han; John Willes; Rahul G. Krishnan
>
> **备注:** ICLR 2025
>
> **摘要:** Prompting Large Language Models (LLMs), or providing context on the expected model of operation, is an effective way to steer the outputs of such models to satisfy human desiderata after they have been trained. But in rapidly evolving domains, there is often need to fine-tune LLMs to improve either the kind of knowledge in their memory or their abilities to perform open ended reasoning in new domains. When human's learn new concepts, we often do so by linking the new material that we are studying to concepts we have already learned before. To that end, we ask, "can prompting help us teach LLMs how to learn". In this work, we study a novel generalization of instruction tuning, called contextual fine-tuning, to fine-tune LLMs. Our method leverages instructional prompts designed to mimic human cognitive strategies in learning and problem-solving to guide the learning process during training, aiming to improve the model's interpretation and understanding of domain-specific knowledge. We empirically demonstrate that this simple yet effective modification improves the ability of LLMs to be fine-tuned rapidly on new datasets both within the medical and financial domains.
>
---
#### [replaced 012] SafeWork-R1: Coevolving Safety and Intelligence under the AI-45$^{\circ}$ Law
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.18576v3](http://arxiv.org/pdf/2507.18576v3)**

> **作者:** Shanghai AI Lab; :; Yicheng Bao; Guanxu Chen; Mingkang Chen; Yunhao Chen; Chiyu Chen; Lingjie Chen; Sirui Chen; Xinquan Chen; Jie Cheng; Yu Cheng; Dengke Deng; Yizhuo Ding; Dan Ding; Xiaoshan Ding; Yi Ding; Zhichen Dong; Lingxiao Du; Yuyu Fan; Xinshun Feng; Yanwei Fu; Yuxuan Gao; Ruijun Ge; Tianle Gu; Lujun Gui; Jiaxuan Guo; Qianxi He; Yuenan Hou; Xuhao Hu; Hong Huang; Kaichen Huang; Shiyang Huang; Yuxian Jiang; Shanzhe Lei; Jie Li; Lijun Li; Hao Li; Juncheng Li; Xiangtian Li; Yafu Li; Lingyu Li; Xueyan Li; Haotian Liang; Dongrui Liu; Qihua Liu; Zhixuan Liu; Bangwei Liu; Huacan Liu; Yuexiao Liu; Zongkai Liu; Chaochao Lu; Yudong Lu; Xiaoya Lu; Zhenghao Lu; Qitan Lv; Caoyuan Ma; Jiachen Ma; Xiaoya Ma; Zhongtian Ma; Lingyu Meng; Ziqi Miao; Yazhe Niu; Yuezhang Peng; Yuan Pu; Han Qi; Chen Qian; Xingge Qiao; Jingjing Qu; Jiashu Qu; Wanying Qu; Wenwen Qu; Xiaoye Qu; Qihan Ren; Qingnan Ren; Qingyu Ren; Jing Shao; Wenqi Shao; Shuai Shao; Dongxing Shi; Xin Song; Xinhao Song; Yan Teng; Xuan Tong; Yingchun Wang; Xuhong Wang; Shujie Wang; Xin Wang; Yige Wang; Yixu Wang; Yuanfu Wang; Futing Wang; Ruofan Wang; Wenjie Wang; Yajie Wang; Muhao Wei; Xiaoyu Wen; Fenghua Weng; Yuqi Wu; Yingtong Xiong; Xingcheng Xu; Chao Yang; Yue Yang; Yang Yao; Yulei Ye; Zhenyun Yin; Yi Yu; Bo Zhang; Qiaosheng Zhang; Jinxuan Zhang; Yexin Zhang; Yinqiang Zheng; Hefeng Zhou; Zhanhui Zhou; Pengyu Zhu; Qingzi Zhu; Yubo Zhu; Bowen Zhou
>
> **备注:** 47 pages, 18 figures, authors are listed in alphabetical order by their last names; v3 modifies minor issues
>
> **摘要:** We introduce SafeWork-R1, a cutting-edge multimodal reasoning model that demonstrates the coevolution of capabilities and safety. It is developed by our proposed SafeLadder framework, which incorporates large-scale, progressive, safety-oriented reinforcement learning post-training, supported by a suite of multi-principled verifiers. Unlike previous alignment methods such as RLHF that simply learn human preferences, SafeLadder enables SafeWork-R1 to develop intrinsic safety reasoning and self-reflection abilities, giving rise to safety `aha' moments. Notably, SafeWork-R1 achieves an average improvement of $46.54\%$ over its base model Qwen2.5-VL-72B on safety-related benchmarks without compromising general capabilities, and delivers state-of-the-art safety performance compared to leading proprietary models such as GPT-4.1 and Claude Opus 4. To further bolster its reliability, we implement two distinct inference-time intervention methods and a deliberative search mechanism, enforcing step-level verification. Finally, we further develop SafeWork-R1-InternVL3-78B, SafeWork-R1-DeepSeek-70B, and SafeWork-R1-Qwen2.5VL-7B. All resulting models demonstrate that safety and capability can co-evolve synergistically, highlighting the generalizability of our framework in building robust, reliable, and trustworthy general-purpose AI.
>
---
#### [replaced 013] DEL: Context-Aware Dynamic Exit Layer for Efficient Self-Speculative Decoding
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.05598v2](http://arxiv.org/pdf/2504.05598v2)**

> **作者:** Hossein Entezari Zarch; Lei Gao; Chaoyi Jiang; Murali Annavaram
>
> **摘要:** Speculative Decoding (SD) is a widely used approach to accelerate the inference of large language models (LLMs) without reducing generation quality. It operates by first using a compact model to draft multiple tokens efficiently, followed by parallel verification using the target LLM. This approach leads to faster inference compared to auto-regressive decoding. While there are multiple approaches to create a draft model, one promising approach is to use early-exit methods. These methods draft candidate tokens by using a subset of layers of the primary model and applying the remaining layers for verification, allowing a single model to handle both drafting and verification. While this technique reduces memory usage and computational cost, its performance relies on the choice of the exit layer for drafting and the number of tokens drafted (speculation length) in each SD round. Prior works use hyperparameter exploration to statically select these values. However, our evaluations show that these hyperparameter values are task-specific, and even within a task they are dependent on the current sequence context. We introduce DEL (Dynamic Exit Layer), a plug-and-play method that adaptively selects the exit layer and speculation length during inference. DEL dynamically tracks the token acceptance rate if the tokens are drafted at each layer of an LLM and uses that knowledge to heuristically select the optimal exit layer and speculation length. Our experiments across a broad range of models and downstream tasks show that DEL achieves overall speedups of $2.16\times$$\sim$$2.62\times$ over vanilla auto-regressive decoding and improves upon state-of-the-art SD methods, which peak at $2.43\times$, by up to $0.19\times$. The code is available at https://github.com/hoenza/DEL.
>
---
#### [replaced 014] Recent Advances in Speech Language Models: A Survey
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.03751v4](http://arxiv.org/pdf/2410.03751v4)**

> **作者:** Wenqian Cui; Dianzhi Yu; Xiaoqi Jiao; Ziqiao Meng; Guangyan Zhang; Qichao Wang; Yiwen Guo; Irwin King
>
> **备注:** The reduced version of this paper has been accepted at ACL 2025
>
> **摘要:** Large Language Models (LLMs) have recently garnered significant attention, primarily for their capabilities in text-based interactions. However, natural human interaction often relies on speech, necessitating a shift towards voice-based models. A straightforward approach to achieve this involves a pipeline of ``Automatic Speech Recognition (ASR) + LLM + Text-to-Speech (TTS)", where input speech is transcribed to text, processed by an LLM, and then converted back to speech. Despite being straightforward, this method suffers from inherent limitations, such as information loss during modality conversion, significant latency due to the complex pipeline, and error accumulation across the three stages. To address these issues, Speech Language Models (SpeechLMs) -- end-to-end models that generate speech without converting from text -- have emerged as a promising alternative. This survey paper provides the first comprehensive overview of recent methodologies for constructing SpeechLMs, detailing the key components of their architecture and the various training recipes integral to their development. Additionally, we systematically survey the various capabilities of SpeechLMs, categorize their evaluation metrics, and discuss the challenges and future research directions in this rapidly evolving field. The GitHub repository is available at https://github.com/dreamtheater123/Awesome-SpeechLM-Survey
>
---
#### [replaced 015] Can Vision Language Models Understand Mimed Actions?
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21586v2](http://arxiv.org/pdf/2506.21586v2)**

> **作者:** Hyundong Cho; Spencer Lin; Tejas Srinivasan; Michael Saxon; Deuksin Kwon; Natali T. Chavez; Jonathan May
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Nonverbal communication (NVC) plays an integral role in human language, but studying NVC in general is challenging because of its broad scope and high variance in interpretation among individuals and cultures. However, mime -- the theatrical technique of suggesting intent using only gesture, expression, and movement -- is a subset of NVC that consists of explicit and embodied actions with much lower human interpretation variance. We argue that a solid understanding of mimed actions is a crucial prerequisite for vision-language models capable of interpreting and commanding more subtle aspects of NVC. Hence, we propose Mime Identification Multimodal Evaluation (MIME), a novel video-based question answering benchmark comprising of 86 mimed actions. Constructed with motion capture data, MIME consists of variations of each action with perturbations applied to the character, background, and viewpoint for evaluating recognition robustness. We find that both open-weight and API-based vision-language models perform significantly worse than humans on MIME, motivating the need for increased research for instilling more robust understanding of human gestures.
>
---
#### [replaced 016] WhisperNER: Unified Open Named Entity and Speech Recognition
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.08107v2](http://arxiv.org/pdf/2409.08107v2)**

> **作者:** Gil Ayache; Menachem Pirchi; Aviv Navon; Aviv Shamsian; Gill Hetz; Joseph Keshet
>
> **备注:** ASRU 2025, IEEE
>
> **摘要:** Integrating named entity recognition (NER) with automatic speech recognition (ASR) can significantly enhance transcription accuracy and informativeness. In this paper, we introduce WhisperNER, a novel model that allows joint speech transcription and entity recognition. WhisperNER supports open-type NER, enabling recognition of diverse and evolving entities at inference. Building on recent advancements in open NER research, we augment a large synthetic dataset with synthetic speech samples. This allows us to train WhisperNER on a large number of examples with diverse NER tags. During training, the model is prompted with NER labels and optimized to output the transcribed utterance along with the corresponding tagged entities. To evaluate WhisperNER, we generate synthetic speech for commonly used NER benchmarks and annotate existing ASR datasets with open NER tags. Our experiments demonstrate that WhisperNER outperforms natural baselines on both out-of-domain open type NER and supervised finetuning.
>
---
#### [replaced 017] Understanding Large Language Model Behaviors through Interactive Counterfactual Generation and Analysis
- **分类: cs.CL; cs.AI; cs.HC; cs.LG; I.2.7; H.5.2**

- **链接: [http://arxiv.org/pdf/2405.00708v2](http://arxiv.org/pdf/2405.00708v2)**

> **作者:** Furui Cheng; Vilém Zouhar; Robin Shing Moon Chan; Daniel Fürst; Hendrik Strobelt; Mennatallah El-Assady
>
> **摘要:** Understanding the behavior of large language models (LLMs) is crucial for ensuring their safe and reliable use. However, existing explainable AI (XAI) methods for LLMs primarily rely on word-level explanations, which are often computationally inefficient and misaligned with human reasoning processes. Moreover, these methods often treat explanation as a one-time output, overlooking its inherently interactive and iterative nature. In this paper, we present LLM Analyzer, an interactive visualization system that addresses these limitations by enabling intuitive and efficient exploration of LLM behaviors through counterfactual analysis. Our system features a novel algorithm that generates fluent and semantically meaningful counterfactuals via targeted removal and replacement operations at user-defined levels of granularity. These counterfactuals are used to compute feature attribution scores, which are then integrated with concrete examples in a table-based visualization, supporting dynamic analysis of model behavior. A user study with LLM practitioners and interviews with experts demonstrate the system's usability and effectiveness, emphasizing the importance of involving humans in the explanation process as active participants rather than passive recipients.
>
---
#### [replaced 018] Perception-Aware Policy Optimization for Multimodal Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.06448v4](http://arxiv.org/pdf/2507.06448v4)**

> **作者:** Zhenhailong Wang; Xuehang Guo; Sofia Stoica; Haiyang Xu; Hongru Wang; Hyeonjeong Ha; Xiusi Chen; Yangyi Chen; Ming Yan; Fei Huang; Heng Ji
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has proven to be a highly effective strategy for endowing Large Language Models (LLMs) with robust multi-step reasoning abilities. However, its design and optimizations remain tailored to purely textual domains, resulting in suboptimal performance when applied to multimodal reasoning tasks. In particular, we observe that a major source of error in current multimodal reasoning lies in the perception of visual inputs. To address this bottleneck, we propose PAPO, a novel policy gradient algorithm that encourages the model to learn to perceive while learning to reason. Specifically, we introduce the Implicit Perception Loss in the form of a KL divergence term, which can be seamlessly plugged into mainstream RLVR algorithms such as GRPO and DAPO. Notably, PAPO does not rely on additional data curation, reward models, or stronger teacher models. To further enhance the training stability of PAPO, we introduce the Double Entropy Loss, which effectively regularizes the new KL objective without compromising performance. Despite its simplicity, PAPO yields significant overall improvements of 4.4%-17.5% on diverse multimodal benchmarks. The improvements are more pronounced, approaching 8.0%-19.1%, on tasks with high vision dependency. We also observe a substantial reduction of 30.5% in perception errors, indicating improved perceptual capabilities with PAPO. Overall, our work introduces a deeper integration of perception-aware supervision into core learning objectives and lays the groundwork for a new RL framework that encourages visually grounded reasoning. Code and data will be made publicly available for research purposes. Project page: https://mikewangwzhl.github.io/PAPO.
>
---
#### [replaced 019] SciReplicate-Bench: Benchmarking LLMs in Agent-driven Algorithmic Reproduction from Research Papers
- **分类: cs.CL; cs.AI; cs.MA; cs.SE**

- **链接: [http://arxiv.org/pdf/2504.00255v2](http://arxiv.org/pdf/2504.00255v2)**

> **作者:** Yanzheng Xiang; Hanqi Yan; Shuyin Ouyang; Lin Gui; Yulan He
>
> **摘要:** This study evaluates large language models (LLMs) in generating code from algorithm descriptions in recent NLP papers. The task requires two key competencies: (1) algorithm comprehension: synthesizing information from papers and academic literature to understand implementation logic, and (2) coding expertise: identifying dependencies and correctly implementing necessary APIs. To facilitate rigorous evaluation, we introduce SciReplicate-Bench, a benchmark of 100 tasks from 36 NLP papers published in 2024, featuring detailed annotations and comprehensive test cases. Building on SciReplicate-Bench, we propose Sci-Reproducer, a dual-agent framework consisting of a Paper Agent that interprets algorithmic concepts from literature and a Code Agent that retrieves dependencies from repositories and implements solutions. To assess algorithm understanding, we introduce reasoning graph accuracy, which quantifies similarity between generated and reference reasoning graphs derived from code comments and structure. For evaluating implementation quality, we employ execution accuracy, CodeBLEU, and repository dependency/API recall metrics. In our experiments, we evaluate various powerful non-reasoning and reasoning LLMs as foundational models. The best-performing LLM using \ModelName~achieves only 39% execution accuracy, highlighting the benchmark's difficulty. Our analysis identifies missing or inconsistent algorithm descriptions as key barriers to successful reproduction. We make available our benchmark and code at https://github.com/xyzCS/SciReplicate-Bench and project homepage at https://xyzcs.github.io/scireplicate.github.io/.
>
---
#### [replaced 020] Efficient Knowledge Injection in LLMs via Self-Distillation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.14964v2](http://arxiv.org/pdf/2412.14964v2)**

> **作者:** Kalle Kujanpää; Pekka Marttinen; Harri Valpola; Alexander Ilin
>
> **备注:** Preprint
>
> **摘要:** In many practical applications, large language models (LLMs) need to acquire new knowledge not present in their pre-training data. Efficiently leveraging this knowledge usually relies on supervised fine-tuning or retrieval-augmented generation (RAG). Although RAG has emerged as the industry standard for knowledge injection, fine-tuning has not yet achieved comparable success. This paper proposes utilizing prompt distillation, a self-distillation-based method previously explored primarily for style alignment and instruction tuning, to internalize new factual knowledge from free-form documents. Unlike prior methods, our approach requires neither larger teacher models nor structured knowledge formats. Across multiple LLM sizes and model families, we show that prompt distillation outperforms standard supervised fine-tuning and can even surpass RAG. We analyze the key factors contributing to prompt distillation's effectiveness and examine how it scales.
>
---
#### [replaced 021] Optimizing LLM-Based Multi-Agent System with Textual Feedback: A Case Study on Software Development
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16086v2](http://arxiv.org/pdf/2505.16086v2)**

> **作者:** Ming Shen; Raphael Shu; Anurag Pratik; James Gung; Yubin Ge; Monica Sunkara; Yi Zhang
>
> **摘要:** We have seen remarkable progress in large language models (LLMs) empowered multi-agent systems solving complex tasks necessitating cooperation among experts with diverse skills. However, optimizing LLM-based multi-agent systems remains challenging. In this work, we perform an empirical case study on group optimization of role-based multi-agent systems utilizing natural language feedback for challenging software development tasks under various evaluation dimensions. We propose a two-step agent prompts optimization pipeline: identifying underperforming agents with their failure explanations utilizing textual feedback and then optimizing system prompts of identified agents utilizing failure explanations. We then study the impact of various optimization settings on system performance with two comparison groups: online against offline optimization and individual against group optimization. For group optimization, we study two prompting strategies: one-pass and multi-pass prompting optimizations. Overall, we demonstrate the effectiveness of our optimization method for role-based multi-agent systems tackling software development tasks evaluated on diverse evaluation dimensions, and we investigate the impact of diverse optimization settings on group behaviors of the multi-agent systems to provide practical insights for future development.
>
---
#### [replaced 022] Multi-Agents Based on Large Language Models for Knowledge-based Visual Question Answering
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.18351v2](http://arxiv.org/pdf/2412.18351v2)**

> **作者:** Zhongjian Hu; Peng Yang; Bing Li; Zhenqi Wang
>
> **备注:** We would like to withdraw this submission due to ongoing internal review and coordination among the author team. Upon the supervisor's recommendation, we have decided to delay public dissemination until the manuscript undergoes further refinement and aligns with our intended academic trajectory
>
> **摘要:** Large Language Models (LLMs) have achieved impressive results in knowledge-based Visual Question Answering (VQA). However existing methods still have challenges: the inability to use external tools autonomously, and the inability to work in teams. Humans tend to know whether they need to use external tools when they encounter a new question, e.g., they tend to be able to give a direct answer to a familiar question, whereas they tend to use tools such as search engines when they encounter an unfamiliar question. In addition, humans also tend to collaborate and discuss with others to get better answers. Inspired by this, we propose the multi-agent voting framework. We design three LLM-based agents that simulate different levels of staff in a team, and assign the available tools according to the levels. Each agent provides the corresponding answer, and finally all the answers provided by the agents are voted to get the final answer. Experiments on OK-VQA and A-OKVQA show that our approach outperforms other baselines by 2.2 and 1.0, respectively.
>
---
#### [replaced 023] Learning to Diagnose Privately: DP-Powered LLMs for Radiology Report Classification
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.04450v2](http://arxiv.org/pdf/2506.04450v2)**

> **作者:** Payel Bhattacharjee; Fengwei Tian; Geoffrey D. Rubin; Joseph Y. Lo; Nirav Merchant; Heidi Hanson; John Gounley; Ravi Tandon
>
> **备注:** 18 pages, 5 figures, 2 tables
>
> **摘要:** Purpose: This study proposes a framework for fine-tuning large language models (LLMs) with differential privacy (DP) to perform multi-abnormality classification on radiology report text. By injecting calibrated noise during fine-tuning, the framework seeks to mitigate the privacy risks associated with sensitive patient data and protect against data leakage while maintaining classification performance. Materials and Methods: We used 50,232 radiology reports from the publicly available MIMIC-CXR chest radiography and CT-RATE computed tomography datasets, collected between 2011 and 2019. Fine-tuning of LLMs was conducted to classify 14 labels from MIMIC-CXR dataset, and 18 labels from CT-RATE dataset using Differentially Private Low-Rank Adaptation (DP-LoRA) in high and moderate privacy regimes (across a range of privacy budgets = {0.01, 0.1, 1.0, 10.0}). Model performance was evaluated using weighted F1 score across three model architectures: BERT-medium, BERT-small, and ALBERT-base. Statistical analyses compared model performance across different privacy levels to quantify the privacy-utility trade-off. Results: We observe a clear privacy-utility trade-off through our experiments on 2 different datasets and 3 different models. Under moderate privacy guarantees the DP fine-tuned models achieved comparable weighted F1 scores of 0.88 on MIMIC-CXR and 0.59 on CT-RATE, compared to non-private LoRA baselines of 0.90 and 0.78, respectively. Conclusion: Differentially private fine-tuning using LoRA enables effective and privacy-preserving multi-abnormality classification from radiology reports, addressing a key challenge in fine-tuning LLMs on sensitive medical data.
>
---
#### [replaced 024] Rationale-guided Prompting for Knowledge-based Visual Question Answering
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.16936v3](http://arxiv.org/pdf/2412.16936v3)**

> **作者:** Zhongjian Hu; Peng Yang; Bing Li; Fengyuan Liu
>
> **备注:** We would like to withdraw this submission due to ongoing internal review and coordination among the author team. Upon the supervisor's recommendation, we have decided to delay public dissemination until the manuscript undergoes further refinement and aligns with our intended academic trajectory
>
> **摘要:** Recently, Large Language Models (LLMs) have been used for knowledge-based Visual Question Answering (VQA). Despite the encouraging results of previous studies, prior methods prompt LLMs to predict answers directly, neglecting intermediate thought processes. We argue that prior methods do not sufficiently activate the capacities of LLMs. We propose a framework called PLRH that Prompts LLMs with Rationale Heuristics for knowledge-based VQA. The PLRH prompts LLMs with Chain of Thought (CoT) to generate rationale heuristics, i.e., intermediate thought processes, and then leverages the rationale heuristics to inspire LLMs to predict answers. Experiments show that our approach outperforms the existing baselines by more than 2.2 and 2.1 on OK-VQA and A-OKVQA, respectively.
>
---
#### [replaced 025] RLTHF: Targeted Human Feedback for LLM Alignment
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13417v3](http://arxiv.org/pdf/2502.13417v3)**

> **作者:** Yifei Xu; Tusher Chakraborty; Emre Kıcıman; Bibek Aryal; Eduardo Rodrigues; Srinagesh Sharma; Roberto Estevao; Maria Angels de Luis Balaguer; Jessica Wolk; Rafael Padilha; Leonardo Nunes; Shobana Balakrishnan; Songwu Lu; Ranveer Chandra
>
> **备注:** Presented at ICML 2025
>
> **摘要:** Fine-tuning large language models (LLMs) to align with user preferences is challenging due to the high cost of quality human annotations in Reinforcement Learning from Human Feedback (RLHF) and the generalizability limitations of AI Feedback. To address these challenges, we propose RLTHF, a human-AI hybrid framework that combines LLM-based initial alignment with selective human annotations to achieve full-human annotation alignment with minimal effort. RLTHF identifies hard-to-annotate samples mislabeled by LLMs using a reward model's reward distribution and iteratively enhances alignment by integrating strategic human corrections while leveraging LLM's correctly labeled samples. Evaluations on HH-RLHF and TL;DR datasets show that RLTHF reaches full-human annotation-level alignment with only 6-7% of the human annotation effort. Furthermore, models trained on RLTHF's curated datasets for downstream tasks outperform those trained on fully human-annotated datasets, underscoring the effectiveness of RLTHF.
>
---
#### [replaced 026] Efficient Attention Mechanisms for Large Language Models: A Survey
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.19595v2](http://arxiv.org/pdf/2507.19595v2)**

> **作者:** Yutao Sun; Zhenyu Li; Yike Zhang; Tengyu Pan; Bowen Dong; Yuyi Guo; Jianyong Wang
>
> **备注:** work in progress
>
> **摘要:** Transformer-based architectures have become the prevailing backbone of large language models. However, the quadratic time and memory complexity of self-attention remains a fundamental obstacle to efficient long-context modeling. To address this limitation, recent research has introduced two principal categories of efficient attention mechanisms. Linear attention methods achieve linear complexity through kernel approximations, recurrent formulations, or fastweight dynamics, thereby enabling scalable inference with reduced computational overhead. Sparse attention techniques, in contrast, limit attention computation to selected subsets of tokens based on fixed patterns, block-wise routing, or clustering strategies, enhancing efficiency while preserving contextual coverage. This survey provides a systematic and comprehensive overview of these developments, integrating both algorithmic innovations and hardware-level considerations. In addition, we analyze the incorporation of efficient attention into largescale pre-trained language models, including both architectures built entirely on efficient attention and hybrid designs that combine local and global components. By aligning theoretical foundations with practical deployment strategies, this work aims to serve as a foundational reference for advancing the design of scalable and efficient language models.
>
---
#### [replaced 027] PolyGuard: A Multilingual Safety Moderation Tool for 17 Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.04377v2](http://arxiv.org/pdf/2504.04377v2)**

> **作者:** Priyanshu Kumar; Devansh Jain; Akhila Yerukola; Liwei Jiang; Himanshu Beniwal; Thomas Hartvigsen; Maarten Sap
>
> **备注:** Accepted to COLM 2025 Main Conference
>
> **摘要:** Truly multilingual safety moderation efforts for Large Language Models (LLMs) have been hindered by a narrow focus on a small set of languages (e.g., English, Chinese) as well as a limited scope of safety definition, resulting in significant gaps in moderation capabilities. To bridge these gaps, we release POLYGUARD, a new state-of-the-art multilingual safety model for safeguarding LLM generations, and the corresponding training and evaluation datasets. POLYGUARD is trained on POLYGUARDMIX, the largest multilingual safety training corpus to date containing 1.91M samples across 17 languages (e.g., Chinese, Czech, English, Hindi). We also introduce POLYGUARDPROMPTS, a high quality multilingual benchmark with 29K samples for the evaluation of safety guardrails. Created by combining naturally occurring multilingual human-LLM interactions and human-verified machine translations of an English-only safety dataset (WildGuardMix; Han et al., 2024), our datasets contain prompt-output pairs with labels of prompt harmfulness, response harmfulness, and response refusal. Through extensive evaluations across multiple safety and toxicity benchmarks, we demonstrate that POLYGUARD outperforms existing state-of-the-art open-weight and commercial safety classifiers by 5.5%. Our contributions advance efforts toward safer multilingual LLMs for all global users.
>
---
#### [replaced 028] CrisisSense-LLM: Instruction Fine-Tuned Large Language Model for Multi-label Social Media Text Classification in Disaster Informatics
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.15477v3](http://arxiv.org/pdf/2406.15477v3)**

> **作者:** Kai Yin; Bo Li; Chengkai Liu; Ali Mostafavi; Xia Hu
>
> **备注:** Relevant source code and data is available: https://github.com/KaiYin97/CrsisLLM
>
> **摘要:** In the field of crisis/disaster informatics, social media is increasingly being used for improving situational awareness to inform response and relief efforts. Efficient and accurate text classification tools have been a focal area of investigation in crisis informatics. However, current methods mostly rely on single-label text classification models, which fails to capture different insights embedded in dynamic and multifaceted disaster-related social media data. This study introduces a novel approach to disaster text classification by enhancing a pre-trained Large Language Model (LLM) through instruction fine-tuning targeted for multi-label classification of disaster-related tweets. Our methodology involves creating a comprehensive instruction dataset from disaster-related tweets, which is then used to fine-tune an open-source LLM, thereby embedding it with disaster-specific knowledge. This fine-tuned model can classify multiple aspects of disaster-related information simultaneously, such as the type of event, informativeness, and involvement of human aid, significantly improving the utility of social media data for situational awareness in disasters. The results demonstrate that this approach enhances the categorization of critical information from social media posts, thereby facilitating a more effective deployment for situational awareness during emergencies. This research paves the way for more advanced, adaptable, and robust disaster management tools, leveraging the capabilities of LLMs to improve real-time situational awareness and response strategies in disaster scenarios.
>
---
#### [replaced 029] JEPA4Rec: Learning Effective Language Representations for Sequential Recommendation via Joint Embedding Predictive Architecture
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.10512v3](http://arxiv.org/pdf/2504.10512v3)**

> **作者:** Minh-Anh Nguyen; Dung D. Le
>
> **摘要:** Language representation learning has emerged as a promising approach for sequential recommendation, thanks to its ability to learn generalizable representations. However, despite its advantages, this approach still struggles with data sparsity and a limited understanding of common-sense user preferences. To address these limitations, we propose $\textbf{JEPA4Rec}$, a framework that combines $\textbf{J}$oint $\textbf{E}$mbedding $\textbf{P}$redictive $\textbf{A}$rchitecture with language modeling of item textual descriptions. JEPA4Rec captures semantically rich and transferable representations, improving recommendation performance and reducing reliance on large-scale pre-training data. Specifically, JEPA4Rec represents items as text sentences by flattening descriptive information such as $\textit{title, category}$, and other attributes. To encode these sentences, we employ a bidirectional Transformer encoder with modified embedding layers tailored for capturing item information in recommendation datasets. We apply masking to text sentences and use them to predict the representations of the unmasked sentences, helping the model learn generalizable item embeddings. To further improve recommendation performance and language understanding, we employ a two-stage training strategy incorporating self-supervised learning losses. Experiments on six real-world datasets demonstrate that JEPA4Rec consistently outperforms state-of-the-art methods, particularly in cross-domain, cross-platform, and low-resource scenarios.
>
---
#### [replaced 030] TreeDiff: AST-Guided Code Generation with Diffusion LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.01473v2](http://arxiv.org/pdf/2508.01473v2)**

> **作者:** Yiming Zeng; Jinghan Cao; Zexin Li; Yiming Chen; Tao Ren; Dawei Xiang; Xidong Wu; Shangqian Gao; Tingting Yu
>
> **摘要:** Recent advances in diffusion-based language models have opened new possibilities for controllable and bidirectional sequence generation. These models provide an alternative to traditional autoregressive approaches by framing text generation as an iterative denoising process. However, applying diffusion models to structured domains such as source code remains a significant challenge. Programming languages differ from natural language in that they follow strict syntactic and semantic rules, with hierarchical organization that must be preserved for correctness. Standard token-level corruption techniques used during training often ignore this structure, which may hinder the model's ability to learn meaningful representations of code. To address this limitation, we propose a syntax-aware diffusion framework that incorporates structural priors from Abstract Syntax Trees (ASTs) into the denoising process. Instead of masking individual tokens at random, we selectively corrupt syntactically meaningful code spans derived from AST subtrees. This enables the model to reconstruct programs in a way that respects grammatical boundaries and captures long-range dependencies. Experimental results demonstrate that syntax-aware corruption significantly improves syntactic correctness, reconstruction accuracy, and generalization to unseen code patterns. These findings highlight the potential of incorporating structural information into diffusion-based training and suggest that syntax-guided denoising is a promising direction for advancing diffusion-based language models in code generation tasks.
>
---
#### [replaced 031] Scaling Laws For Mixed Quantization
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.06722v3](http://arxiv.org/pdf/2410.06722v3)**

> **作者:** Zeyu Cao; Boyang Gu; Cheng Zhang; Pedro Gimenes; Jianqiao Lu; Jianyi Cheng; Xitong Gao; Yiren Zhao
>
> **摘要:** Post-training quantization of Large Language Models (LLMs) has proven effective in reducing the memory and computational requirements for inference. In this study, we focus on a straightforward question: When aiming for a target accuracy or perplexity with low-precision quantization, how much high-precision computation needs to be preserved, and how fine-grained this quantization would need to be as we scale LLMs to larger sizes? We first introduce two critical metrics, named the quantization ratio ($Q_r$) and quantization block size ($Q_b$). The former measures the number of parameters quantized to low-precision arithmetic normalized by the total parameter count, whereas the latter defines the number of values within a block that share a scaling factor, akin to the block size concept introduced in the FP4 format in NVIDIA's Blackwell architecture. Through extensive and carefully controlled experiments across different models and quantization methods, we propose a unified scaling law on post-training quantization (PTQ) that can predict loss degeneration for varying $Q_r$ and $Q_b$. For $Q_r$, our scaling law implies that parameter scaling and ratio scaling have a multiplicative relationship. Consequently, larger models are more amenable to a higher quantization ratio $Q_r$, thus supporting an increase in the adoption of mixed quantization for inference. Regarding $Q_b$, our findings indicate that a small block size, similar to that used in Blackwell, is not essential for large models. Employing a small $Q_b$ can instead unnecessarily complicate the design of the hardware circuit.
>
---
#### [replaced 032] R2Vul: Learning to Reason about Software Vulnerabilities with Reinforcement Learning and Structured Reasoning Distillation
- **分类: cs.SE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.04699v2](http://arxiv.org/pdf/2504.04699v2)**

> **作者:** Martin Weyssow; Chengran Yang; Junkai Chen; Ratnadira Widyasari; Ting Zhang; Huihui Huang; Huu Hung Nguyen; Yan Naing Tun; Tan Bui; Yikun Li; Ang Han Wei; Frank Liauw; Eng Lieh Ouh; Lwin Khin Shar; David Lo
>
> **摘要:** Large language models (LLMs) have shown promising performance in software vulnerability detection, yet their reasoning capabilities remain unreliable. We propose R2Vul, a method that combines reinforcement learning from AI feedback (RLAIF) and structured reasoning distillation to teach small code LLMs to detect vulnerabilities while generating security-aware explanations. Unlike prior chain-of-thought and instruction tuning approaches, R2Vul rewards well-founded over deceptively plausible vulnerability explanations through RLAIF, which results in more precise detection and high-quality reasoning generation. To support RLAIF, we construct the first multilingual preference dataset for vulnerability detection, comprising 18,000 high-quality samples in C\#, JavaScript, Java, Python, and C. We evaluate R2Vul across five programming languages and against four static analysis tools, eight state-of-the-art LLM-based baselines, and various fine-tuning approaches. Our results demonstrate that a 1.5B R2Vul model exceeds the performance of its 32B teacher model and leading commercial LLMs such as Claude-4-Opus. Furthermore, we introduce a lightweight calibration step that reduces false positive rates under varying imbalanced data distributions. Finally, through qualitative analysis, we show that both LLM and human evaluators consistently rank R2Vul model's reasoning higher than other reasoning-based baselines.
>
---
#### [replaced 033] Tell Me Who Your Students Are: GPT Can Generate Valid Multiple-Choice Questions When Students' (Mis)Understanding Is Hinted
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.05815v2](http://arxiv.org/pdf/2505.05815v2)**

> **作者:** Machi Shimmei; Masaki Uto; Yuichiroh Matsubayashi; Kentaro Inui; Aditi Mallavarapu; Noboru Matsuda
>
> **备注:** This is a pre-print version of a paper to appear in AIED2025. The camera-ready version is available at https://link.springer.com/chapter/10.1007/978-3-031-99264-3_16
>
> **摘要:** The primary goal of this study is to develop and evaluate an innovative prompting technique, AnaQuest, for generating multiple-choice questions (MCQs) using a pre-trained large language model. In AnaQuest, the choice items are sentence-level assertions about complex concepts. The technique integrates formative and summative assessments. In the formative phase, students answer open-ended questions for target concepts in free text. For summative assessment, AnaQuest analyzes these responses to generate both correct and incorrect assertions. To evaluate the validity of the generated MCQs, Item Response Theory (IRT) was applied to compare item characteristics between MCQs generated by AnaQuest, a baseline ChatGPT prompt, and human-crafted items. An empirical study found that expert instructors rated MCQs generated by both AI models to be as valid as those created by human instructors. However, IRT-based analysis revealed that AnaQuest-generated questions - particularly those with incorrect assertions (foils) - more closely resembled human-crafted items in terms of difficulty and discrimination than those produced by ChatGPT.
>
---
#### [replaced 034] Using Sentiment Analysis to Investigate Peer Feedback by Native and Non-Native English Speakers
- **分类: cs.CL; I.2.7; K.3.1**

- **链接: [http://arxiv.org/pdf/2507.22924v2](http://arxiv.org/pdf/2507.22924v2)**

> **作者:** Brittney Exline; Melanie Duffin; Brittany Harbison; Chrissa da Gomez; David Joyner
>
> **摘要:** Graduate-level CS programs in the U.S. increasingly enroll international students, with 60.2 percent of master's degrees in 2023 awarded to non-U.S. students. Many of these students take online courses, where peer feedback is used to engage students and improve pedagogy in a scalable manner. Since these courses are conducted in English, many students study in a language other than their first. This paper examines how native versus non-native English speaker status affects three metrics of peer feedback experience in online U.S.-based computing courses. Using the Twitter-roBERTa-based model, we analyze the sentiment of peer reviews written by and to a random sample of 500 students. We then relate sentiment scores and peer feedback ratings to students' language background. Results show that native English speakers rate feedback less favorably, while non-native speakers write more positively but receive less positive sentiment in return. When controlling for sex and age, significant interactions emerge, suggesting that language background plays a modest but complex role in shaping peer feedback experiences.
>
---
#### [replaced 035] MedHalu: Hallucinations in Responses to Healthcare Queries by Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.19492v2](http://arxiv.org/pdf/2409.19492v2)**

> **作者:** Vibhor Agarwal; Yiqiao Jin; Mohit Chandra; Munmun De Choudhury; Srijan Kumar; Nishanth Sastry
>
> **备注:** Accepted at ICWSM2026
>
> **摘要:** Large language models (LLMs) are starting to complement traditional information seeking mechanisms such as web search. LLM-powered chatbots like ChatGPT are gaining prominence among the general public. AI chatbots are also increasingly producing content on social media platforms. However, LLMs are also prone to hallucinations, generating plausible yet factually incorrect or fabricated information. This becomes a critical problem when laypeople start seeking information about sensitive issues such as healthcare. Existing works in LLM hallucinations in the medical domain mainly focus on testing the medical knowledge of LLMs through standardized medical exam questions which are often well-defined and clear-cut with definitive answers. However, these approaches may not fully capture how these LLMs perform during real-world interactions with patients. This work conducts a pioneering study on hallucinations in LLM-generated responses to real-world healthcare queries from patients.We introduce MedHalu, a novel medical hallucination benchmark featuring diverse health-related topics and hallucinated responses from LLMs, with detailed annotation of the hallucination types and text spans. We also propose MedHaluDetect, a comprehensive framework for evaluating LLMs' abilities to detect hallucinations. Furthermore, we study the vulnerability to medical hallucinations among three groups -- medical experts, LLMs, and laypeople. Notably, LLMs significantly underperform human experts and, in some cases, even laypeople in detecting medical hallucinations. To improve hallucination detection, we propose an expert-in-the-loop approach that integrates expert reasoning into LLM inputs, significantly improving hallucination detection for all LLMs, including a 6.3% macro-F1 improvement for GPT-4.
>
---
#### [replaced 036] DSBC : Data Science task Benchmarking with Context engineering
- **分类: cs.AI; cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2507.23336v2](http://arxiv.org/pdf/2507.23336v2)**

> **作者:** Ram Mohan Rao Kadiyala; Siddhant Gupta; Jebish Purbey; Giulio Martini; Ali Shafique; Suman Debnath; Hamza Farooq
>
> **备注:** 32 pages
>
> **摘要:** Recent advances in large language models (LLMs) have significantly impacted data science workflows, giving rise to specialized data science agents designed to automate analytical tasks. Despite rapid adoption, systematic benchmarks evaluating the efficacy and limitations of these agents remain scarce. In this paper, we introduce a comprehensive benchmark specifically crafted to reflect real-world user interactions with data science agents by observing usage of our commercial applications. We evaluate three LLMs: Claude-4.0-Sonnet, Gemini-2.5-Flash, and OpenAI-o4-Mini across three approaches: zero-shot with context engineering, multi-step with context engineering, and with SmolAgent. Our benchmark assesses performance across a diverse set of eight data science task categories, additionally exploring the sensitivity of models to common prompting issues, such as data leakage and slightly ambiguous instructions. We further investigate the influence of temperature parameters on overall and task-specific outcomes for each model and approach. Our findings reveal distinct performance disparities among the evaluated models and methodologies, highlighting critical factors that affect practical deployment. The benchmark dataset and evaluation framework introduced herein aim to provide a foundation for future research of more robust and effective data science agents.
>
---
#### [replaced 037] Verbalized Representation Learning for Interpretable Few-Shot Generalization
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.18651v3](http://arxiv.org/pdf/2411.18651v3)**

> **作者:** Cheng-Fu Yang; Da Yin; Wenbo Hu; Heng Ji; Nanyun Peng; Bolei Zhou; Kai-Wei Chang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Humans recognize objects after observing only a few examples, a remarkable capability enabled by their inherent language understanding of the real-world environment. Developing verbalized and interpretable representation can significantly improve model generalization in low-data settings. In this work, we propose Verbalized Representation Learning (VRL), a novel approach for automatically extracting human-interpretable features for object recognition using few-shot data. Our method uniquely captures inter-class differences and intra-class commonalities in the form of natural language by employing a Vision-Language Model (VLM) to identify key discriminative features between different classes and shared characteristics within the same class. These verbalized features are then mapped to numeric vectors through the VLM. The resulting feature vectors can be further utilized to train and infer with downstream classifiers. Experimental results show that, at the same model scale, VRL achieves a 24% absolute improvement over prior state-of-the-art methods while using 95% less data and a smaller mode. Furthermore, compared to human-labeled attributes, the features learned by VRL exhibit a 20% absolute gain when used for downstream classification tasks. Code is available at: https://github.com/joeyy5588/VRL/tree/main.
>
---
#### [replaced 038] CRAFT Your Dataset: Task-Specific Synthetic Dataset Generation Through Corpus Retrieval and Augmentation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.02098v2](http://arxiv.org/pdf/2409.02098v2)**

> **作者:** Ingo Ziegler; Abdullatif Köksal; Desmond Elliott; Hinrich Schütze
>
> **备注:** Accepted at TACL; Pre-MIT Press publication version. Code and dataset available at: https://github.com/ziegler-ingo/CRAFT
>
> **摘要:** Building high-quality datasets for specialized tasks is a time-consuming and resource-intensive process that often requires specialized domain knowledge. We propose Corpus Retrieval and Augmentation for Fine-Tuning (CRAFT), a method for generating synthetic datasets, given a small number of user-written few-shots that demonstrate the task to be performed. Given these examples, CRAFT uses large-scale public web-crawled corpora and similarity-based document retrieval to find other relevant human-written documents. Lastly, instruction-tuned large language models (LLMs) augment the retrieved documents into custom-formatted task samples, which then can be used for fine-tuning. We demonstrate that CRAFT can efficiently generate large-scale task-specific training datasets for four diverse tasks: biology, medicine, and commonsense question-answering (QA), as well as summarization. Our experiments show that CRAFT-based models outperform or match general LLMs on QA tasks, while exceeding models trained on human-curated summarization data by 46 preference points. CRAFT outperforms other synthetic dataset generation methods such as Self- and Evol-Instruct, and remains robust even when the quality of the initial few-shots varies.
>
---
#### [replaced 039] Explainable Recommendation with Simulated Human Feedback
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.14147v2](http://arxiv.org/pdf/2504.14147v2)**

> **作者:** Jiakai Tang; Jingsen Zhang; Zihang Tian; Xueyang Feng; Lei Wang; Xu Chen
>
> **摘要:** Recent advancements in explainable recommendation have greatly bolstered user experience by elucidating the decision-making rationale. However, the existing methods actually fail to provide effective feedback signals for potentially better or worse generated explanations due to their reliance on traditional supervised learning paradigms in sparse interaction data. To address these issues, we propose a novel human-like feedback-driven optimization framework. This framework employs a dynamic interactive optimization mechanism for achieving human-centered explainable requirements without incurring high labor costs. Specifically, we propose to utilize large language models (LLMs) as human simulators to predict human-like feedback for guiding the learning process. To enable the LLMs to deeply understand the task essence and meet user's diverse personalized requirements, we introduce a human-induced customized reward scoring method, which helps stimulate the language understanding and logical reasoning capabilities of LLMs. Furthermore, considering the potential conflicts between different perspectives of explanation quality, we introduce a principled Pareto optimization that transforms the multi-perspective quality enhancement task into a multi-objective optimization problem for improving explanation performance. At last, to achieve efficient model training, we design an off-policy optimization pipeline. By incorporating a replay buffer and addressing the data distribution biases, we can effectively improve data utilization and enhance model generality. Extensive experiments on four datasets demonstrate the superiority of our approach.
>
---
#### [replaced 040] Human Cognitive Benchmarks Reveal Foundational Visual Gaps in MLLMs
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16435v2](http://arxiv.org/pdf/2502.16435v2)**

> **作者:** Jen-Tse Huang; Dasen Dai; Jen-Yuan Huang; Youliang Yuan; Xiaoyuan Liu; Wenxuan Wang; Wenxiang Jiao; Pinjia He; Zhaopeng Tu; Haodong Duan
>
> **备注:** Update: Evaluated 20 MLLMs; Added generated test cases
>
> **摘要:** Despite significant progress on popular multimodal benchmarks, state-of-the-art Multimodal Large Language Models (MLLMs) continue to struggle with basic visual reasoning tasks that are trivially solved by humans, such as recognizing spatial relationships. To systematically investigate this gap, we introduce VisFactor, a benchmark that digitizes 20 vision-centric subtests from a well-established cognitive psychology assessment. These subtests span four core domains of human visual cognition: (1) Visualization and Spatial Processing, (2) Perceptual and Closure, (3) Memory, and (4) Reasoning. We evaluate 20 frontier MLLMs from GPT, Gemini, Claude, LLaMA, Qwen, and SEED families. The best-performing model achieves a score of only 25.19 out of 100, with consistent failures on tasks such as mental rotation, spatial relation inference, and figure-ground discrimination, regardless of model size or prompting strategy. These findings suggest that current MLLM performance gains on high-level benchmarks do not reflect human-like low-level visual cognition, challenging the assumption that large-scale pretraining naturally induces gestalt-like perceptual capabilities. The dataset and evaluation toolkit are publicly available at: https://github.com/CUHK-ARISE/VisFactor.
>
---
#### [replaced 041] When in Doubt, Cascade: Towards Building Efficient and Capable Guardrails
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.06323v2](http://arxiv.org/pdf/2407.06323v2)**

> **作者:** Manish Nagireddy; Inkit Padhi; Soumya Ghosh; Prasanna Sattigeri
>
> **摘要:** Large language models (LLMs) have convincing performance in a variety of downstream tasks. However, these systems are prone to generating undesirable outputs such as harmful and biased text. In order to remedy such generations, the development of guardrail (or detector) models has gained traction. Motivated by findings from developing a detector for social bias, we adopt the notion of a use-mention distinction - which we identified as the primary source of under-performance in the preliminary versions of our social bias detector. Armed with this information, we describe a fully extensible and reproducible synthetic data generation pipeline which leverages taxonomy-driven instructions to create targeted and labeled data. Using this pipeline, we generate over 300K unique contrastive samples and provide extensive experiments to systematically evaluate performance on a suite of open source datasets. We show that our method achieves competitive performance with a fraction of the cost in compute and offers insight into iteratively developing efficient and capable guardrail models. Warning: This paper contains examples of text which are toxic, biased, and potentially harmful.
>
---
#### [replaced 042] Large Language Models Still Exhibit Bias in Long Text
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.17519v3](http://arxiv.org/pdf/2410.17519v3)**

> **作者:** Wonje Jeung; Dongjae Jeon; Ashkan Yousefpour; Jonghyun Choi
>
> **备注:** Accepted by ACL, code and models are available at https://github.com/WonjeJeung/LTF-TEST
>
> **摘要:** Existing fairness benchmarks for large language models (LLMs) primarily focus on simple tasks, such as multiple-choice questions, overlooking biases that may arise in more complex scenarios like long-text generation. To address this gap, we introduce the Long Text Fairness Test (LTF-TEST), a framework that evaluates biases in LLMs through essay-style prompts. LTF-TEST covers 14 topics and 10 demographic axes, including gender and race, resulting in 11,948 samples. By assessing both model responses and the reasoning behind them, LTF-TEST uncovers subtle biases that are difficult to detect in simple responses. In our evaluation of five recent LLMs, including GPT-4o and LLaMa3, we identify two key patterns of bias. First, these models frequently favor certain demographic groups in their responses. Second, they show excessive sensitivity toward traditionally disadvantaged groups, often providing overly protective responses while neglecting others. To mitigate these biases, we propose FT-REGARD, a finetuning approach that pairs biased prompts with neutral responses. FT-REGARD reduces gender bias by 34.6% and improves performance by 1.4 percentage points on the BBQ benchmark, offering a promising approach to addressing biases in long-text generation tasks.
>
---
#### [replaced 043] You Cannot Feed Two Birds with One Score: the Accuracy-Naturalness Tradeoff in Translation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.24013v3](http://arxiv.org/pdf/2503.24013v3)**

> **作者:** Gergely Flamich; David Vilar; Jan-Thorsten Peter; Markus Freitag
>
> **备注:** Accepted to COLM 2025. Camera-ready version
>
> **摘要:** The goal of translation, be it by human or by machine, is, given some text in a source language, to produce text in a target language that simultaneously 1) preserves the meaning of the source text and 2) achieves natural expression in the target language. However, researchers in the machine translation community usually assess translations using a single score intended to capture semantic accuracy and the naturalness of the output simultaneously. In this paper, we build on recent advances in information theory to mathematically prove and empirically demonstrate that such single-score summaries do not and cannot give the complete picture of a system's true performance. Concretely, we prove that a tradeoff exists between accuracy and naturalness and demonstrate it by evaluating the submissions to the WMT24 shared task. Our findings help explain well-known empirical phenomena, such as the observation that optimizing translation systems for a specific accuracy metric (like BLEU) initially improves the system's naturalness, while ``overfitting'' the system to the metric can significantly degrade its naturalness. Thus, we advocate for a change in how translations are evaluated: rather than comparing systems using a single number, they should be compared on an accuracy-naturalness plane.
>
---
#### [replaced 044] Language Model Uncertainty Quantification with Attention Chain
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.19168v2](http://arxiv.org/pdf/2503.19168v2)**

> **作者:** Yinghao Li; Rushi Qiang; Lama Moukheiber; Chao Zhang
>
> **备注:** 36 pages, 7 figures, 36 tables
>
> **摘要:** Accurately quantifying a large language model's (LLM) predictive uncertainty is crucial for judging the reliability of its answers. While most existing research focuses on short, directly answerable questions with closed-form outputs (e.g., multiple-choice), involving intermediate reasoning steps in LLM responses is increasingly important. This added complexity complicates uncertainty quantification (UQ) because the probabilities assigned to answer tokens are conditioned on a vast space of preceding reasoning tokens. Direct marginalization is infeasible, and the dependency inflates probability estimates, causing overconfidence in UQ. To address this, we propose UQAC, an efficient method that narrows the reasoning space to a tractable size for marginalization. UQAC iteratively constructs an "attention chain" of tokens deemed "semantically crucial" to the final answer via a backtracking procedure. Starting from the answer tokens, it uses attention weights to identify the most influential predecessors, then iterates this process until reaching the input tokens. The resulting chain is further refined with similarity filtering and probability thresholding, which reduce the reasoning space, facilitating the approximation of the marginal answer token probabilities. We validate UQAC on multiple reasoning benchmarks with advanced open-source LLMs, demonstrating that it consistently delivers reliable UQ estimates with high computational efficiency.
>
---
#### [replaced 045] VeOmni: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo
- **分类: cs.CL; cs.AI; cs.DC**

- **链接: [http://arxiv.org/pdf/2508.02317v3](http://arxiv.org/pdf/2508.02317v3)**

> **作者:** Qianli Ma; Yaowei Zheng; Zhelun Shi; Zhongkai Zhao; Bin Jia; Ziyue Huang; Zhiqi Lin; Youjie Li; Jiacheng Yang; Yanghua Peng; Zhi Zhang; Xin Liu
>
> **摘要:** Recent advances in large language models (LLMs) have driven impressive progress in omni-modal understanding and generation. However, training omni-modal LLMs remains a significant challenge due to the heterogeneous model architectures required to process diverse modalities, necessitating sophisticated system design for efficient large-scale training. Existing frameworks typically entangle model definition with parallel logic, incurring limited scalability and substantial engineering overhead for end-to-end omni-modal training. We present VeOmni, a modular and efficient training framework to accelerate the development of omni-modal LLMs. VeOmni introduces model-centric distributed recipes that decouples communication from computation, enabling efficient 3D parallelism on omni-modal LLMs. VeOmni also features a flexible configuration interface supporting seamless integration of new modalities with minimal code change. Using VeOmni, a omni-modal mixture-of-experts (MoE) model with 30B parameters can be trained with over 2,800 tokens/sec/GPU throughput and scale to 160K context lengths via 3D parallelism on 128 GPUs, showcasing its superior efficiency and scalability for training large omni-modal LLMs.
>
---
#### [replaced 046] The Impact of Item-Writing Flaws on Difficulty and Discrimination in Item Response Theory
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2503.10533v3](http://arxiv.org/pdf/2503.10533v3)**

> **作者:** Robin Schmucker; Steven Moore
>
> **备注:** Added Acknowledgments
>
> **摘要:** High-quality test items are essential for educational assessments, particularly within Item Response Theory (IRT). Traditional validation methods rely on resource-intensive pilot testing to estimate item difficulty and discrimination. More recently, Item-Writing Flaw (IWF) rubrics emerged as a domain-general approach for evaluating test items based on textual features. This method offers a scalable, pre-deployment evaluation without requiring student data, but its predictive validity concerning empirical IRT parameters is underexplored. To address this gap, we conducted a study involving 7,126 multiple-choice questions across various STEM subjects (physical science, mathematics, and life/earth sciences). Using an automated approach, we annotated each question with a 19-criteria IWF rubric and studied relationships to data-driven IRT parameters. Our analysis revealed statistically significant links between the number of IWFs and IRT difficulty and discrimination parameters, particularly in life/earth and physical science domains. We further observed how specific IWF criteria can impact item quality more and less severely (e.g., negative wording vs. implausible distractors) and how they might make a question more or less challenging. Overall, our findings establish automated IWF analysis as a valuable supplement to traditional validation, providing an efficient method for initial item screening, particularly for flagging low-difficulty MCQs. Our findings show the need for further research on domain-general evaluation rubrics and algorithms that understand domain-specific content for robust item validation.
>
---
#### [replaced 047] Enhancing Code LLMs with Reinforcement Learning in Code Generation: A Survey
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.20367v5](http://arxiv.org/pdf/2412.20367v5)**

> **作者:** Junqiao Wang; Zeng Zhang; Yangfan He; Zihao Zhang; Xinyuan Song; Yuyang Song; Tianyu Shi; Yuchen Li; Hengyuan Xu; Kunyu Wu; Xin Yi; Zhongwei Wan; Xinhang Yuan; Zijun Wang; Kuan Lu; Menghao Huo; Tang Jingqun; Guangwu Qian; Keqin Li; Qiuwu Chen; Lewei He
>
> **摘要:** With the rapid evolution of large language models (LLM), reinforcement learning (RL) has emerged as a pivotal technique for code generation and optimization in various domains. This paper presents a systematic survey of the application of RL in code optimization and generation, highlighting its role in enhancing compiler optimization, resource allocation, and the development of frameworks and tools. Subsequent sections first delve into the intricate processes of compiler optimization, where RL algorithms are leveraged to improve efficiency and resource utilization. The discussion then progresses to the function of RL in resource allocation, emphasizing register allocation and system optimization. We also explore the burgeoning role of frameworks and tools in code generation, examining how RL can be integrated to bolster their capabilities. This survey aims to serve as a comprehensive resource for researchers and practitioners interested in harnessing the power of RL to advance code generation and optimization techniques.
>
---
#### [replaced 048] An Entity Linking Agent for Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.03865v2](http://arxiv.org/pdf/2508.03865v2)**

> **作者:** Yajie Luo; Yihong Wu; Muzhi Li; Fengran Mo; Jia Ao Sun; Xinyu Wang; Liheng Ma; Yingxue Zhang; Jian-Yun Nie
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Some Question Answering (QA) systems rely on knowledge bases (KBs) to provide accurate answers. Entity Linking (EL) plays a critical role in linking natural language mentions to KB entries. However, most existing EL methods are designed for long contexts and do not perform well on short, ambiguous user questions in QA tasks. We propose an entity linking agent for QA, based on a Large Language Model that simulates human cognitive workflows. The agent actively identifies entity mentions, retrieves candidate entities, and makes decision. To verify the effectiveness of our agent, we conduct two experiments: tool-based entity linking and QA task evaluation. The results confirm the robustness and effectiveness of our agent.
>
---
#### [replaced 049] Medal Matters: Probing LLMs' Failure Cases Through Olympic Rankings
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.06518v2](http://arxiv.org/pdf/2409.06518v2)**

> **作者:** Juhwan Choi; Seunguk Yu; JungMin Yun; YoungBin Kim
>
> **备注:** COLM 2025 ORIGen Workshop
>
> **摘要:** Large language models (LLMs) have achieved remarkable success in natural language processing tasks, yet their internal knowledge structures remain poorly understood. This study examines these structures through the lens of historical Olympic medal tallies, evaluating LLMs on two tasks: (1) retrieving medal counts for specific teams and (2) identifying rankings of each team. While state-of-the-art LLMs excel in recalling medal counts, they struggle with providing rankings, highlighting a key difference between their knowledge organization and human reasoning. These findings shed light on the limitations of LLMs' internal knowledge integration and suggest directions for improvement. To facilitate further research, we release our code, dataset, and model outputs.
>
---
#### [replaced 050] Data Processing for the OpenGPT-X Model Family
- **分类: cs.CL; H.3.1; I.2.7**

- **链接: [http://arxiv.org/pdf/2410.08800v4](http://arxiv.org/pdf/2410.08800v4)**

> **作者:** Nicolo' Brandizzi; Hammam Abdelwahab; Anirban Bhowmick; Lennard Helmer; Benny Jörg Stein; Pavel Denisov; Qasid Saleem; Michael Fromm; Mehdi Ali; Richard Rutmann; Farzad Naderi; Mohamad Saif Agy; Alexander Schwirjow; Fabian Küch; Luzian Hahn; Malte Ostendorff; Pedro Ortiz Suarez; Georg Rehm; Dennis Wegener; Nicolas Flores-Herr; Joachim Köhler; Johannes Leveling
>
> **摘要:** This paper presents a comprehensive overview of the data preparation pipeline developed for the OpenGPT-X project, a large-scale initiative aimed at creating open and high-performance multilingual large language models (LLMs). The project goal is to deliver models that cover all major European languages, with a particular focus on real-world applications within the European Union. We explain all data processing steps, starting with the data selection and requirement definition to the preparation of the final filtered data. We distinguish between curated data and web data, as each of these categories is handled by distinct pipelines, with curated data undergoing minimal filtering and web data requiring extensive filtering and deduplication. This distinction guided the development of specialized algorithmic solutions for both pipelines. In addition to describing the processing methodologies, we provide an in-depth analysis of the datasets, increasing transparency and alignment with European data regulations. Finally, we share key insights and challenges faced during the project, offering recommendations for future endeavors in large-scale multilingual data preparation for LLMs.
>
---
#### [replaced 051] IFDECORATOR: Wrapping Instruction Following Reinforcement Learning with Verifiable Rewards
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.04632v2](http://arxiv.org/pdf/2508.04632v2)**

> **作者:** Xu Guo; Tianyi Liang; Tong Jian; Xiaogui Yang; Ling-I Wu; Chenhui Li; Zhihui Lu; Qipeng Guo; Kai Chen
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) improves instruction following capabilities of large language models (LLMs), but suffers from training inefficiency due to inadequate difficulty assessment. Moreover, RLVR is prone to over-optimization, where LLMs exploit verification shortcuts without aligning to the actual intent of user instructions. We introduce Instruction Following Decorator (IFDecorator}, a framework that wraps RLVR training into a robust and sample-efficient pipeline. It consists of three components: (1) a cooperative-adversarial data flywheel that co-evolves instructions and hybrid verifications, generating progressively more challenging instruction-verification pairs; (2) IntentCheck, a bypass module enforcing intent alignment; and (3) trip wires, a diagnostic mechanism that detects reward hacking via trap instructions, which trigger and capture shortcut exploitation behaviors. Our Qwen2.5-32B-Instruct-IFDecorator achieves 87.43% accuracy on IFEval, outperforming larger proprietary models such as GPT-4o. Additionally, we demonstrate substantial improvements on FollowBench while preserving general capabilities. Our trip wires show significant reductions in reward hacking rates. We will release models, code, and data for future research.
>
---
#### [replaced 052] Hierarchical Budget Policy Optimization for Adaptive Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.15844v3](http://arxiv.org/pdf/2507.15844v3)**

> **作者:** Shangke Lyu; Linjuan Wu; Yuchen Yan; Xingyu Wu; Hao Li; Yongliang Shen; Peisheng Jiang; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **备注:** Code: https://github.com/zju-real/hbpo Project Page:https://zju-real.github.io/hbpo/
>
> **摘要:** Large reasoning models achieve remarkable performance through extensive chain-of-thought generation, yet they suffer from a critical inefficiency: applying uniformly extensive reasoning regardless of problem complexity. We present Hierarchical Budget Policy Optimization (HBPO), a reinforcement learning framework that enables models to learn problem-specific reasoning depths without sacrificing capability. Unlike existing approaches that impose rigid constraints or rely on discrete mode selection, HBPO partitions the exploration space into budget-constrained hierarchies (512-2560 tokens), each with differentiated reward structures that preserve both efficiency incentives and reasoning capabilities. This design addresses a fundamental challenge in efficient reasoning training: traditional length penalties systematically bias models away from necessary long reasoning paths, causing exploration space collapse. Through hierarchical sampling and budget-aware rewards, HBPO maintains exploration diversity while teaching models to recognize when extended deliberation is warranted. Extensive experiments demonstrate that HBPO reduces average token usage by up to 60.6% while improving accuracy by 3.14% across four reasoning benchmarks. Most notably, HBPO exhibits emergent adaptive behavior where models automatically adjust reasoning depth based on problem complexity. Our results suggest that reasoning efficiency and capability are not inherently conflicting, and can be simultaneously optimized through appropriately structured hierarchical training that preserves exploration diversity.
>
---
#### [replaced 053] DOTS: Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.03864v2](http://arxiv.org/pdf/2410.03864v2)**

> **作者:** Murong Yue; Wenlin Yao; Haitao Mi; Dian Yu; Ziyu Yao; Dong Yu
>
> **备注:** Accepted to ICLR 2025
>
> **摘要:** Enhancing the capability of large language models (LLMs) in reasoning has gained significant attention in recent years. Previous studies have demonstrated the effectiveness of various prompting strategies in aiding LLMs in reasoning (called "reasoning actions"), such as step-by-step thinking, reflecting before answering, solving with programs, and their combinations. However, these approaches often applied static, predefined reasoning actions uniformly to all questions, without considering the specific characteristics of each question or the capability of the task-solving LLM. In this paper, we propose DOTS, an approach enabling LLMs to reason dynamically via optimal reasoning trajectory search, tailored to the specific characteristics of each question and the inherent capability of the task-solving LLM. Our approach involves three key steps: i) defining atomic reasoning action modules that can be composed into various reasoning action trajectories; ii) searching for the optimal action trajectory for each training question through iterative exploration and evaluation for the specific task-solving LLM; and iii) using the collected optimal trajectories to train an LLM to plan for the reasoning trajectories of unseen questions. In particular, we propose two learning paradigms, i.e., fine-tuning an external LLM as a planner to guide the task-solving LLM, or directly fine-tuning the task-solving LLM with an internalized capability for reasoning actions planning. Our experiments across eight reasoning tasks show that our method consistently outperforms static reasoning techniques and the vanilla instruction tuning approach. Further analysis reveals that our method enables LLMs to adjust their computation based on problem complexity, allocating deeper thinking and reasoning to harder problems.
>
---
#### [replaced 054] CUPID: Evaluating Personalized and Contextualized Alignment of LLMs from Interactions
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2508.01674v2](http://arxiv.org/pdf/2508.01674v2)**

> **作者:** Tae Soo Kim; Yoonjoo Lee; Yoonah Park; Jiho Kim; Young-Ho Kim; Juho Kim
>
> **备注:** Accepted to COLM 2025. Project Website: https://cupid.kixlab.org/
>
> **摘要:** Personalization of Large Language Models (LLMs) often assumes users hold static preferences that reflect globally in all tasks. In reality, humans hold dynamic preferences that change depending on the context. As users interact with an LLM in various contexts, they naturally reveal their contextual preferences, which a model must infer and apply in future contexts to ensure alignment. To assess this, we introduce CUPID, a benchmark of 756 human-curated interaction session histories between users and LLM-based chat assistants. In each interaction session, the user provides a request in a specific context and expresses their preference through multi-turn feedback. Given a new user request and prior interaction sessions, our benchmark assesses whether LLMs can infer the preference relevant to this request and generate a response that satisfies this preference. With CUPID, we evaluated 10 open and proprietary LLMs, revealing that state-of-the-art LLMs struggle to infer preferences from multi-turn interactions and fail to discern what previous context is relevant to a new request -- under 50% precision and 65% recall. Our work highlights the need to advance LLM capabilities for more contextually personalized interactions and proposes CUPID as a resource to drive these improvements.
>
---
#### [replaced 055] Which Questions Improve Learning the Most? Utility Estimation of Questions with LM-based Simulations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17383v2](http://arxiv.org/pdf/2502.17383v2)**

> **作者:** Dong-Ho Lee; Hyundong Cho; Jonathan May; Jay Pujara
>
> **备注:** 17 pages, 5 figures, 6 tables
>
> **摘要:** Asking good questions is critical for comprehension and learning, yet evaluating and generating such questions remains a challenging problem. Prior work on inquisitive questions focuses on learner-generated, curiosity-driven queries and evaluates them using indirect metrics, such as salience or information gain, that do not directly capture a question's impact on actual learning outcomes. We introduce QUEST (Question Utility Estimation with Simulated Tests), a framework that uses language models to simulate learners and directly quantify the utility of a question - its contribution to exam performance. QUEST simulates a learner who asks questions and receives answers while studying a textbook chapter, then uses them to take an end-of-chapter exam. Through this simulation, the utility of each question is estimated by its direct effect on exam performance, rather than inferred indirectly based on the underlying content. To support this evaluation, we curate TEXTBOOK-EXAM, a benchmark that aligns textbook sections with end-of-section exam questions across five academic disciplines. Using QUEST, we filter for high-utility questions and fine-tune question generators via rejection sampling. Experiments show that questions generated by QUEST-trained models improve simulated test scores by over 20% compared to strong baselines that are fine-tuned using indirect metrics or leverage prompting methods. Furthermore, utility is only weakly correlated with salience and similarity to exam questions, suggesting that it captures unique signal that benefits downstream performance. QUEST offers a new outcome-driven paradigm for question evaluation and generation - one that moves beyond question-answer content toward measurable improvements in learning outcomes.
>
---
#### [replaced 056] Can open source large language models be used for tumor documentation in Germany? -- An evaluation on urological doctors' notes
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.12106v4](http://arxiv.org/pdf/2501.12106v4)**

> **作者:** Stefan Lenz; Arsenij Ustjanzew; Marco Jeray; Meike Ressing; Torsten Panholzer
>
> **备注:** 53 pages, 5 figures
>
> **摘要:** Tumor documentation in Germany is largely done manually, requiring reading patient records and entering data into structured databases. Large language models (LLMs) could potentially enhance this process by improving efficiency and reliability. This evaluation tests eleven different open source LLMs with sizes ranging from 1-70 billion model parameters on three basic tasks of the tumor documentation process: identifying tumor diagnoses, assigning ICD-10 codes, and extracting the date of first diagnosis. For evaluating the LLMs on these tasks, a dataset of annotated text snippets based on anonymized doctors' notes from urology was prepared. Different prompting strategies were used to investigate the effect of the number of examples in few-shot prompting and to explore the capabilities of the LLMs in general. The models Llama 3.1 8B, Mistral 7B, and Mistral NeMo 12 B performed comparably well in the tasks. Models with less extensive training data or having fewer than 7 billion parameters showed notably lower performance, while larger models did not display performance gains. Examples from a different medical domain than urology could also improve the outcome in few-shot prompting, which demonstrates the ability of LLMs to handle tasks needed for tumor documentation. Open source LLMs show a strong potential for automating tumor documentation. Models from 7-12 billion parameters could offer an optimal balance between performance and resource efficiency. With tailored fine-tuning and well-designed prompting, these models might become important tools for clinical documentation in the future. The code for the evaluation is available from https://github.com/stefan-m-lenz/UroLlmEval. We also release the dataset as a new valuable resource that addresses the shortage of authentic and easily accessible benchmarks in German-language medical NLP.
>
---
#### [replaced 057] GuARD: Effective Anomaly Detection through a Text-Rich and Graph-Informed Language Model
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.03930v2](http://arxiv.org/pdf/2412.03930v2)**

> **作者:** Yunhe Pang; Bo Chen; Fanjin Zhang; Yanghui Rao; Evgeny Kharlamov; Jie Tang
>
> **备注:** Accepted at KDD 2025
>
> **摘要:** Anomaly detection on text-rich graphs is widely prevalent in real life, such as detecting incorrectly assigned academic papers to authors and detecting bots in social networks. The remarkable capabilities of large language models (LLMs) pave a new revenue by utilizing rich-text information for effective anomaly detection. However, simply introducing rich texts into LLMs can obscure essential detection cues and introduce high fine-tuning costs. Moreover, LLMs often overlook the intrinsic structural bias of graphs which is vital for distinguishing normal from abnormal node patterns. To this end, this paper introduces GuARD, a text-rich and graph-informed language model that combines key structural features from graph-based methods with fine-grained semantic attributes extracted via small language models for effective anomaly detection on text-rich graphs. GuARD is optimized with the progressive multi-modal multi-turn instruction tuning framework in the task-guided instruction tuning regime tailed to incorporate both rich-text and structural modalities. Extensive experiments on four datasets reveal that GuARD outperforms graph-based and LLM-based anomaly detection methods, while offering up to 5$\times$ times speedup in training and 5$\times$ times speedup in inference over vanilla long-context LLMs on the large-scale WhoIsWho dataset.
>
---
#### [replaced 058] Flex-Judge: Text-Only Reasoning Unleashes Zero-Shot Multimodal Evaluators
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.18601v3](http://arxiv.org/pdf/2505.18601v3)**

> **作者:** Jongwoo Ko; Sungnyun Kim; Sungwoo Cho; Se-Young Yun
>
> **备注:** The code is available at https://github.com/jongwooko/flex-judge
>
> **摘要:** Human-generated reward signals are critical for aligning generative models with human preferences, guiding both training and inference-time evaluations. While large language models (LLMs) employed as proxy evaluators, i.e., LLM-as-a-Judge, significantly reduce the costs associated with manual annotations, they typically require extensive modality-specific training data and fail to generalize well across diverse multimodal tasks. In this paper, we propose Flex-Judge, a reasoning-guided multimodal judge model that leverages minimal textual reasoning data to robustly generalize across multiple modalities and evaluation formats. Our core intuition is that structured textual reasoning explanations inherently encode generalizable decision-making patterns, enabling an effective transfer to multimodal judgments, e.g., with images or videos. Empirical results demonstrate that Flex-Judge, despite being trained on significantly fewer text data, achieves competitive or superior performance compared to state-of-the-art commercial APIs and extensively trained multimodal evaluators. Notably, Flex-Judge presents broad impact in modalities like molecule, where comprehensive evaluation benchmarks are scarce, underscoring its practical value in resource-constrained domains. Our framework highlights reasoning-based text supervision as a powerful, cost-effective alternative to traditional annotation-intensive approaches, substantially advancing scalable multimodal model-as-a-judge.
>
---
#### [replaced 059] Basis Selection: Low-Rank Decomposition of Pretrained Large Language Models for Target Applications
- **分类: cs.LG; cs.AR; cs.CL**

- **链接: [http://arxiv.org/pdf/2405.15877v3](http://arxiv.org/pdf/2405.15877v3)**

> **作者:** Yang Li; Daniel Agyei Asante; Changsheng Zhao; Ernie Chang; Yangyang Shi; Vikas Chandra
>
> **摘要:** Large language models (LLMs) significantly enhance the performance of various applications, but they are computationally intensive and energy-demanding. This makes it challenging to deploy them on devices with limited resources, such as personal computers and mobile/wearable devices, and results in substantial inference costs in resource-rich environments like cloud servers. To extend the use of LLMs, we introduce a low-rank decomposition approach to effectively compress these models, tailored to the requirements of specific applications. We observe that LLMs pretrained on general datasets contain many redundant components not needed for particular applications. Our method focuses on identifying and removing these redundant parts, retaining only the necessary elements for the target applications. Specifically, we represent the weight matrices of LLMs as a linear combination of base components. We then prune the irrelevant bases and enhance the model with new bases beneficial for specific applications. Deep compression results on the Llama 2-7b and -13B models, conducted on target applications including mathematical reasoning and code generation, show that our method significantly reduces model size while maintaining comparable accuracy to state-of-the-art low-rank compression techniques.
>
---
#### [replaced 060] Balancing Stylization and Truth via Disentangled Representation Steering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.04530v2](http://arxiv.org/pdf/2508.04530v2)**

> **作者:** Chenglei Shen; Zhongxiang Sun; Teng Shi; Xiao Zhang; Jun Xu
>
> **摘要:** Generating stylized large language model (LLM) responses via representation editing is a promising way for fine-grained output control. However, there exists an inherent trade-off: imposing a distinctive style often degrades truthfulness. Existing representation editing methods, by naively injecting style signals, overlook this collateral impact and frequently contaminate the model's core truthfulness representations, resulting in reduced answer correctness. We term this phenomenon stylization-induced truthfulness collapse. We attribute this issue to latent coupling between style and truth directions in certain key attention heads, and propose StyliTruth, a mechanism that preserves stylization while keeping truthfulness intact. StyliTruth separates the style-relevant and truth-relevant subspaces in the model's representation space via an orthogonal deflation process. This decomposition enables independent control of style and truth in their own subspaces, minimizing interference. By designing adaptive, token-level steering vectors within each subspace, we dynamically and precisely control the generation process to maintain both stylistic fidelity and truthfulness. We validate our method on multiple styles and languages. Extensive experiments and analyses show that StyliTruth significantly reduces stylization-induced truthfulness collapse and outperforms existing inference-time intervention methods in balancing style adherence with truthfulness.
>
---
#### [replaced 061] Semantic Integrity Constraints: Declarative Guardrails for AI-Augmented Data Processing Systems
- **分类: cs.DB; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.00600v3](http://arxiv.org/pdf/2503.00600v3)**

> **作者:** Alexander W. Lee; Justin Chan; Michael Fu; Nicolas Kim; Akshay Mehta; Deepti Raghavan; Ugur Cetintemel
>
> **摘要:** AI-augmented data processing systems (DPSs) integrate large language models (LLMs) into query pipelines, allowing powerful semantic operations on structured and unstructured data. However, the reliability (a.k.a. trust) of these systems is fundamentally challenged by the potential for LLMs to produce errors, limiting their adoption in critical domains. To help address this reliability bottleneck, we introduce semantic integrity constraints (SICs) -- a declarative abstraction for specifying and enforcing correctness conditions over LLM outputs in semantic queries. SICs generalize traditional database integrity constraints to semantic settings, supporting common types of constraints, such as grounding, soundness, and exclusion, with both reactive and proactive enforcement strategies. We argue that SICs provide a foundation for building reliable and auditable AI-augmented data systems. Specifically, we present a system design for integrating SICs into query planning and runtime execution and discuss its realization in AI-augmented DPSs. To guide and evaluate our vision, we outline several design goals -- covering criteria around expressiveness, runtime semantics, integration, performance, and enterprise-scale applicability -- and discuss how our framework addresses each, along with open research challenges.
>
---
#### [replaced 062] BloomWise: Enhancing Problem-Solving capabilities of Large Language Models using Bloom's-Taxonomy-Inspired Prompts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.04094v2](http://arxiv.org/pdf/2410.04094v2)**

> **作者:** Maria-Eleni Zoumpoulidi; Georgios Paraskevopoulos; Alexandros Potamianos
>
> **备注:** 16 pages, 2 figures
>
> **摘要:** Despite the remarkable capabilities of large language models (LLMs) across a range of tasks, mathematical reasoning remains a challenging frontier. Motivated by the observation that humans learn more effectively when prompted not what to think but how to think, we introduce BloomWise, a cognitively-inspired prompting technique designed to enhance LLMs' performance on mathematical problem solving while making their solutions more explainable. BloomWise encourages LLMs to generate solutions - in the form of explanations - by progressing through a sequence of cognitive operations-from basic (e.g., remembering) to more advanced reasoning skills (e.g., evaluating) - mirroring how humans build understanding. The process iterates through these levels, halting early if a convergence criterion is met: specifically, if two or more consecutive levels yield the same answer, the solution from the earliest such level is output; otherwise, the process continues until all levels are completed. Through extensive experiments across five popular math reasoning datasets, we demonstrate the effectiveness of BloomWise. We also present comprehensive ablation studies to analyze the strengths of each component within our system.
>
---
#### [replaced 063] ArXivBench: When You Should Avoid Using ChatGPT for Academic Writing
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.10496v2](http://arxiv.org/pdf/2504.10496v2)**

> **作者:** Ning Li; Jingran Zhang; Justin Cui
>
> **摘要:** Large language models (LLMs) demonstrate strong capabilities in reasoning and question answering, yet their tendency to generate factually incorrect content remains a critical challenge. This study evaluates proprietary and open-source LLMs on generating relevant research papers with accurate arXiv links. Our evaluation reveals critical academic risks: LLMs frequently generate incorrect arXiv links or references to non-existent papers, fundamentally undermining their ability to properly attribute research contributions to the actual authors. We introduce arXivBench, a benchmark specifically designed to assess LLM performance across eight major subject categories on arXiv and five subfields within computer science, one of the most popular categories among them. Our findings show concerning accuracy variations across subjects, with Claude-3.5-Sonnet exhibiting a substantial advantage in generating both relevant and accurate responses. Notably, most LLMs perform significantly better in Artificial Intelligence than other subfields. This benchmark provides a standardized tool for evaluating LLM reliability in scientific contexts, promoting more dependable academic use in research environments. Our code and dataset are available at https://github.com/liningresearch/arXivBench and https://huggingface.co/datasets/arXivBenchLLM/arXivBench.
>
---
