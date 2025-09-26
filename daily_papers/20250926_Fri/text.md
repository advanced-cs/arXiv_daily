# 自然语言处理 cs.CL

- **最新发布 99 篇**

- **更新 88 篇**

## 最新发布

#### [new 001] MARS: toward more efficient multi-agent collaboration for LLM reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MARS，一种基于角色的多智能体协作框架，用于提升大语言模型（LLM）的推理效率。针对现有方法计算开销大的问题，MARS通过作者-审稿人-主编模式减少交互，实验证明其在保持准确率的同时，将token使用和推理时间降低约50%。**

- **链接: [http://arxiv.org/pdf/2509.20502v1](http://arxiv.org/pdf/2509.20502v1)**

> **作者:** Xiao Wang; Jia Wang; Yijie Wang; Pengtao Dang; Sha Cao; Chi Zhang
>
> **摘要:** Large language models (LLMs) have achieved impressive results in natural language understanding, yet their reasoning capabilities remain limited when operating as single agents. Multi-Agent Debate (MAD) has been proposed to address this limitation by enabling collaborative reasoning among multiple models in a round-table debate manner. While effective, MAD introduces substantial computational overhead due to the number of agents involved and the frequent communication required. In this paper, we propose MARS (Multi-Agent Review System), a role-based collaboration framework inspired by the review process. In MARS, an author agent generates an initial solution, reviewer agents provide decisions and comments independently, and a meta-reviewer integrates the feedback to make the final decision and guide further revision. This design enhances reasoning quality while avoiding costly reviewer-to-reviewer interactions, thereby controlling token consumption and inference time. We compared MARS with both MAD and other state-of-the-art reasoning strategies across multiple benchmarks. Extensive experiments with different LLMs show that MARS matches the accuracy of MAD while reducing both token usage and inference time by approximately 50\%. Code is available at https://github.com/xwang97/MARS.
>
---
#### [new 002] Query-Centric Graph Retrieval Augmented Generation
- **分类: cs.CL; cs.IR; I.2.7; H.3.3**

- **简介: 该论文提出QCG-RAG，一种以查询为中心的图检索增强生成框架，旨在解决现有图RAG方法在粒度上的困境。通过可控粒度的图构建和多跳检索机制，提升问答准确率，适用于长上下文理解和多跳推理任务。**

- **链接: [http://arxiv.org/pdf/2509.21237v1](http://arxiv.org/pdf/2509.21237v1)**

> **作者:** Yaxiong Wu; Jianyuan Bo; Yongyue Zhang; Sheng Liang; Yong Liu
>
> **备注:** 25 pages, 6 figures, 1 table
>
> **摘要:** Graph-based retrieval-augmented generation (RAG) enriches large language models (LLMs) with external knowledge for long-context understanding and multi-hop reasoning, but existing methods face a granularity dilemma: fine-grained entity-level graphs incur high token costs and lose context, while coarse document-level graphs fail to capture nuanced relations. We introduce QCG-RAG, a query-centric graph RAG framework that enables query-granular indexing and multi-hop chunk retrieval. Our query-centric approach leverages Doc2Query and Doc2Query{-}{-} to construct query-centric graphs with controllable granularity, improving graph quality and interpretability. A tailored multi-hop retrieval mechanism then selects relevant chunks via the generated queries. Experiments on LiHuaWorld and MultiHop-RAG show that QCG-RAG consistently outperforms prior chunk-based and graph-based RAG methods in question answering accuracy, establishing a new paradigm for multi-hop reasoning.
>
---
#### [new 003] ConceptViz: A Visual Analytics Approach for Exploring Concepts in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ConceptViz，一个用于探索大语言模型中概念的可视化分析系统。针对稀疏自编码器（SAE）提取的特征难以与人类概念对齐的问题，设计了“识别-解释-验证”流程，提升LLM可解释性研究效率。**

- **链接: [http://arxiv.org/pdf/2509.20376v1](http://arxiv.org/pdf/2509.20376v1)**

> **作者:** Haoxuan Li; Zhen Wen; Qiqi Jiang; Chenxiao Li; Yuwei Wu; Yuchen Yang; Yiyao Wang; Xiuqi Huang; Minfeng Zhu; Wei Chen
>
> **摘要:** Large language models (LLMs) have achieved remarkable performance across a wide range of natural language tasks. Understanding how LLMs internally represent knowledge remains a significant challenge. Despite Sparse Autoencoders (SAEs) have emerged as a promising technique for extracting interpretable features from LLMs, SAE features do not inherently align with human-understandable concepts, making their interpretation cumbersome and labor-intensive. To bridge the gap between SAE features and human concepts, we present ConceptViz, a visual analytics system designed for exploring concepts in LLMs. ConceptViz implements a novel dentification => Interpretation => Validation pipeline, enabling users to query SAEs using concepts of interest, interactively explore concept-to-feature alignments, and validate the correspondences through model behavior verification. We demonstrate the effectiveness of ConceptViz through two usage scenarios and a user study. Our results show that ConceptViz enhances interpretability research by streamlining the discovery and validation of meaningful concept representations in LLMs, ultimately aiding researchers in building more accurate mental models of LLM features. Our code and user guide are publicly available at https://github.com/Happy-Hippo209/ConceptViz.
>
---
#### [new 004] MemLens: Uncovering Memorization in LLMs with Activation Trajectories
- **分类: cs.CL**

- **简介: 该论文提出MemLens，用于检测大语言模型中的记忆行为。针对现有方法在隐式污染数据上效果差的问题，通过分析数值token的概率轨迹，揭示记忆样本的“捷径”行为，并通过LoRA验证其有效性，属于模型记忆检测任务。**

- **链接: [http://arxiv.org/pdf/2509.20909v1](http://arxiv.org/pdf/2509.20909v1)**

> **作者:** Zirui He; Haiyan Zhao; Ali Payani; Mengnan du
>
> **备注:** 20pages, 11 figures, 7 tables
>
> **摘要:** Large language models (LLMs) are commonly evaluated on challenging benchmarks such as AIME and Math500, which are susceptible to contamination and risk of being memorized. Existing detection methods, which primarily rely on surface-level lexical overlap and perplexity, demonstrate low generalization and degrade significantly when encountering implicitly contaminated data. In this paper, we propose MemLens (An Activation Lens for Memorization Detection) to detect memorization by analyzing the probability trajectories of numeric tokens during generation. Our method reveals that contaminated samples exhibit ``shortcut'' behaviors, locking onto an answer with high confidence in the model's early layers, whereas clean samples show more gradual evidence accumulation across the model's full depth. We observe that contaminated and clean samples exhibit distinct and well-separated reasoning trajectories. To further validate this, we inject carefully designed samples into the model through LoRA fine-tuning and observe the same trajectory patterns as in naturally contaminated data. These results provide strong evidence that MemLens captures genuine signals of memorization rather than spurious correlations.
>
---
#### [new 005] Behind RoPE: How Does Causal Mask Encode Positional Information?
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究了Transformer解码器中因果掩码如何编码位置信息，发现其能诱导注意力模式并影响RoPE的效果。通过理论与实证分析，揭示了因果掩码与显式位置编码的交互作用，属于自然语言处理中的模型结构与位置编码研究任务。**

- **链接: [http://arxiv.org/pdf/2509.21042v1](http://arxiv.org/pdf/2509.21042v1)**

> **作者:** Junu Kim; Xiao Liu; Zhenghao Lin; Lei Ji; Yeyun Gong; Edward Choi
>
> **备注:** Codes available at: https://github.com/starmpcc/causal_mask_encodes_positional
>
> **摘要:** While explicit positional encodings such as RoPE are a primary source of positional information in Transformer decoders, the causal mask also provides positional information. In this work, we prove that the causal mask can induce position-dependent patterns in attention scores, even without parameters or causal dependency in the input. Our theoretical analysis indicates that the induced attention pattern tends to favor nearby query-key pairs, mirroring the behavior of common positional encodings. Empirical analysis confirms that trained models exhibit the same behavior, with learned parameters further amplifying these patterns. Notably, we found that the interaction of causal mask and RoPE distorts RoPE's relative attention score patterns into non-relative ones. We consistently observed this effect in modern large language models, suggesting the importance of considering the causal mask as a source of positional information alongside explicit positional encodings.
>
---
#### [new 006] Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models
- **分类: cs.CL**

- **简介: 该论文研究语言模型中语法与领域间的虚假关联问题，探讨模型因训练数据中的语法模板而忽略语义的现象。通过构建数据集和评估框架，发现此类关联会降低模型性能，并可能影响安全微调效果。论文呼吁加强对此类关联的检测与训练数据多样性保障。**

- **链接: [http://arxiv.org/pdf/2509.21155v1](http://arxiv.org/pdf/2509.21155v1)**

> **作者:** Chantal Shaib; Vinith M. Suriyakumar; Levent Sagun; Byron C. Wallace; Marzyeh Ghassemi
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** For an LLM to correctly respond to an instruction it must understand both the semantics and the domain (i.e., subject area) of a given task-instruction pair. However, syntax can also convey implicit information Recent work shows that syntactic templates--frequent sequences of Part-of-Speech (PoS) tags--are prevalent in training data and often appear in model outputs. In this work we characterize syntactic templates, domain, and semantics in task-instruction pairs. We identify cases of spurious correlations between syntax and domain, where models learn to associate a domain with syntax during training; this can sometimes override prompt semantics. Using a synthetic training dataset, we find that the syntactic-domain correlation can lower performance (mean 0.51 +/- 0.06) on entity knowledge tasks in OLMo-2 models (1B-13B). We introduce an evaluation framework to detect this phenomenon in trained models, and show that it occurs on a subset of the FlanV2 dataset in open (OLMo-2-7B; Llama-4-Maverick), and closed (GPT-4o) models. Finally, we present a case study on the implications for safety finetuning, showing that unintended syntactic-domain correlations can be used to bypass refusals in OLMo-2-7B Instruct and GPT-4o. Our findings highlight two needs: (1) to explicitly test for syntactic-domain correlations, and (2) to ensure syntactic diversity in training data, specifically within domains, to prevent such spurious correlations.
>
---
#### [new 007] SKILL-RAG: Self-Knowledge Induced Learning and Filtering for Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SKILL-RAG方法，用于改进检索增强生成（RAG）在知识密集型任务中的表现。针对检索内容可能引发幻觉的问题，利用模型的“自我知识”过滤无关信息，提升生成质量并减少输入文档数量。**

- **链接: [http://arxiv.org/pdf/2509.20377v1](http://arxiv.org/pdf/2509.20377v1)**

> **作者:** Tomoaki Isoda
>
> **摘要:** Retrieval-Augmented Generation (RAG) has significantly improved the performance of large language models (LLMs) on knowledge-intensive tasks in recent years. However, since retrieval systems may return irrelevant content, incorporating such information into the model often leads to hallucinations. Thus, identifying and filtering out unhelpful retrieved content is a key challenge for improving RAG performance.To better integrate the internal knowledge of the model with external knowledge from retrieval, it is essential to understand what the model "knows" and "does not know" (which is also called "self-knowledge"). Based on this insight, we propose SKILL-RAG (Self-Knowledge Induced Learning and Filtering for RAG), a novel method that leverages the model's self-knowledge to determine which retrieved documents are beneficial for answering a given query. We design a reinforcement learning-based training framework to explicitly elicit self-knowledge from the model and employs sentence-level granularity to filter out irrelevant content while preserving useful knowledge.We evaluate SKILL-RAG using Llama2-7B and Qwen3-8B on several question answering benchmarks. Experimental results demonstrate that SKILL-RAG not only improves generation quality but also significantly reduces the number of input documents, validating the importance of self-knowledge in guiding the selection of high-quality retrievals.
>
---
#### [new 008] Enrich-on-Graph: Query-Graph Alignment for Complex Reasoning with LLM Enriching
- **分类: cs.CL**

- **简介: 该论文针对知识图谱问答（KGQA）任务中LLM存在的幻觉和事实错误问题，提出Enrich-on-Graph框架，利用LLM增强KG以弥合语义差距。通过优化方法生成高质量KG，并设计评估指标，实验表明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.20810v1](http://arxiv.org/pdf/2509.20810v1)**

> **作者:** Songze Li; Zhiqiang Liu; Zhengke Gui; Huajun Chen; Wen Zhang
>
> **摘要:** Large Language Models (LLMs) exhibit strong reasoning capabilities in complex tasks. However, they still struggle with hallucinations and factual errors in knowledge-intensive scenarios like knowledge graph question answering (KGQA). We attribute this to the semantic gap between structured knowledge graphs (KGs) and unstructured queries, caused by inherent differences in their focuses and structures. Existing methods usually employ resource-intensive, non-scalable workflows reasoning on vanilla KGs, but overlook this gap. To address this challenge, we propose a flexible framework, Enrich-on-Graph (EoG), which leverages LLMs' prior knowledge to enrich KGs, bridge the semantic gap between graphs and queries. EoG enables efficient evidence extraction from KGs for precise and robust reasoning, while ensuring low computational costs, scalability, and adaptability across different methods. Furthermore, we propose three graph quality evaluation metrics to analyze query-graph alignment in KGQA task, supported by theoretical validation of our optimization objectives. Extensive experiments on two KGQA benchmark datasets indicate that EoG can effectively generate high-quality KGs and achieve the state-of-the-art performance. Our code and data are available at https://github.com/zjukg/Enrich-on-Graph.
>
---
#### [new 009] Who's Laughing Now? An Overview of Computational Humour Generation and Explanation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，聚焦计算幽默的生成与解释任务。旨在解决幽默理解与生成中常识推理与创造力不足的问题，综述了现有研究，指出当前模型仍远逊于人类，并探讨了未来研究方向。**

- **链接: [http://arxiv.org/pdf/2509.21175v1](http://arxiv.org/pdf/2509.21175v1)**

> **作者:** Tyler Loakman; William Thorne; Chenghua Lin
>
> **备注:** Accepted to INLG 2025
>
> **摘要:** The creation and perception of humour is a fundamental human trait, positioning its computational understanding as one of the most challenging tasks in natural language processing (NLP). As an abstract, creative, and frequently context-dependent construct, humour requires extensive reasoning to understand and create, making it a pertinent task for assessing the common-sense knowledge and reasoning abilities of modern large language models (LLMs). In this work, we survey the landscape of computational humour as it pertains to the generative tasks of creation and explanation. We observe that, despite the task of understanding humour bearing all the hallmarks of a foundational NLP task, work on generating and explaining humour beyond puns remains sparse, while state-of-the-art models continue to fall short of human capabilities. We bookend our literature survey by motivating the importance of computational humour processing as a subdiscipline of NLP and presenting an extensive discussion of future directions for research in the area that takes into account the subjective and ethically ambiguous nature of humour.
>
---
#### [new 010] Eigen-1: Adaptive Multi-Agent Refinement with Monitor-Based RAG for Scientific Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Eigen-1框架，用于科学推理任务。针对LLM在推理中因显式检索导致效率低和多智能体方案效果稀释的问题，设计了基于监控的隐式检索模块和结构化协作机制，提升推理准确率并减少资源消耗。**

- **链接: [http://arxiv.org/pdf/2509.21193v1](http://arxiv.org/pdf/2509.21193v1)**

> **作者:** Xiangru Tang; Wanghan Xu; Yujie Wang; Zijie Guo; Daniel Shao; Jiapeng Chen; Cixuan Zhang; Ziyi Wang; Lixin Zhang; Guancheng Wan; Wenlong Zhang; Lei Bai; Zhenfei Yin; Philip Torr; Hanrui Wang; Di Jin
>
> **摘要:** Large language models (LLMs) have recently shown strong progress on scientific reasoning, yet two major bottlenecks remain. First, explicit retrieval fragments reasoning, imposing a hidden "tool tax" of extra tokens and steps. Second, multi-agent pipelines often dilute strong solutions by averaging across all candidates. We address these challenges with a unified framework that combines implicit retrieval and structured collaboration. At its foundation, a Monitor-based retrieval module operates at the token level, integrating external knowledge with minimal disruption to reasoning. On top of this substrate, Hierarchical Solution Refinement (HSR) iteratively designates each candidate as an anchor to be repaired by its peers, while Quality-Aware Iterative Reasoning (QAIR) adapts refinement to solution quality. On Humanity's Last Exam (HLE) Bio/Chem Gold, our framework achieves 48.3\% accuracy -- the highest reported to date, surpassing the strongest agent baseline by 13.4 points and leading frontier LLMs by up to 18.1 points, while simultaneously reducing token usage by 53.5\% and agent steps by 43.7\%. Results on SuperGPQA and TRQA confirm robustness across domains. Error analysis shows that reasoning failures and knowledge gaps co-occur in over 85\% of cases, while diversity analysis reveals a clear dichotomy: retrieval tasks benefit from solution variety, whereas reasoning tasks favor consensus. Together, these findings demonstrate how implicit augmentation and structured refinement overcome the inefficiencies of explicit tool use and uniform aggregation. Code is available at: https://github.com/tangxiangru/Eigen-1.
>
---
#### [new 011] GEP: A GCG-Based method for extracting personally identifiable information from chatbots built on small language models
- **分类: cs.CL**

- **简介: 该论文研究基于小型语言模型的聊天机器人中的个人身份信息（PII）泄露问题，提出了GEP方法。通过实验表明，GEP相比传统模板方法能显著提升PII提取效果，适用于复杂场景。属于自然语言处理中的隐私保护任务。**

- **链接: [http://arxiv.org/pdf/2509.21192v1](http://arxiv.org/pdf/2509.21192v1)**

> **作者:** Jieli Zhu; Vi Ngoc-Nha Tran
>
> **备注:** 16 pages, 5 figures, 4 tables. Under review as a conference paper at ICLR 2026
>
> **摘要:** Small language models (SLMs) become unprecedentedly appealing due to their approximately equivalent performance compared to large language models (LLMs) in certain fields with less energy and time consumption during training and inference. However, the personally identifiable information (PII) leakage of SLMs for downstream tasks has yet to be explored. In this study, we investigate the PII leakage of the chatbot based on SLM. We first finetune a new chatbot, i.e., ChatBioGPT based on the backbone of BioGPT using medical datasets Alpaca and HealthCareMagic. It shows a matchable performance in BERTscore compared with previous studies of ChatDoctor and ChatGPT. Based on this model, we prove that the previous template-based PII attacking methods cannot effectively extract the PII in the dataset for leakage detection under the SLM condition. We then propose GEP, which is a greedy coordinate gradient-based (GCG) method specifically designed for PII extraction. We conduct experimental studies of GEP and the results show an increment of up to 60$\times$ more leakage compared with the previous template-based methods. We further expand the capability of GEP in the case of a more complicated and realistic situation by conducting free-style insertion where the inserted PII in the dataset is in the form of various syntactic expressions instead of fixed templates, and GEP is still able to reveal a PII leakage rate of up to 4.53%.
>
---
#### [new 012] Concise and Sufficient Sub-Sentence Citations for Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文针对检索增强生成（RAG）系统中的引用问题，提出生成简洁且充分的子句级引用，以减少用户验证信息的负担。研究构建了标注指南和数据集，并设计了一个利用大模型生成高质量引用的框架。**

- **链接: [http://arxiv.org/pdf/2509.20859v1](http://arxiv.org/pdf/2509.20859v1)**

> **作者:** Guo Chen; Qiuyuan Li; Qiuxian Li; Hongliang Dai; Xiang Chen; Piji Li
>
> **摘要:** In retrieval-augmented generation (RAG) question answering systems, generating citations for large language model (LLM) outputs enhances verifiability and helps users identify potential hallucinations. However, we observe two problems in the citations produced by existing attribution methods. First, the citations are typically provided at the sentence or even paragraph level. Long sentences or paragraphs may include a substantial amount of irrelevant content. Second, sentence-level citations may omit information that is essential for verifying the output, forcing users to read the surrounding context. In this paper, we propose generating sub-sentence citations that are both concise and sufficient, thereby reducing the effort required by users to confirm the correctness of the generated output. To this end, we first develop annotation guidelines for such citations and construct a corresponding dataset. Then, we propose an attribution framework for generating citations that adhere to our standards. This framework leverages LLMs to automatically generate fine-tuning data for our task and employs a credit model to filter out low-quality examples. Our experiments on the constructed dataset demonstrate that the propose approach can generate high-quality and more readable citations.
>
---
#### [new 013] Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs
- **分类: cs.CL**

- **简介: 该论文研究大语言模型中的阿谀奉承行为，区分了“阿谀同意”和“阿谀赞美”，发现它们在隐空间中沿不同方向编码，可独立调控。属于模型行为分析任务，旨在揭示并分离LLMs中的不同阿谀机制。**

- **链接: [http://arxiv.org/pdf/2509.21305v1](http://arxiv.org/pdf/2509.21305v1)**

> **作者:** Daniel Vennemeyer; Phan Anh Duong; Tiffany Zhan; Tianyu Jiang
>
> **摘要:** Large language models (LLMs) often exhibit sycophantic behaviors -- such as excessive agreement with or flattery of the user -- but it is unclear whether these behaviors arise from a single mechanism or multiple distinct processes. We decompose sycophancy into sycophantic agreement and sycophantic praise, contrasting both with genuine agreement. Using difference-in-means directions, activation additions, and subspace geometry across multiple models and datasets, we show that: (1) the three behaviors are encoded along distinct linear directions in latent space; (2) each behavior can be independently amplified or suppressed without affecting the others; and (3) their representational structure is consistent across model families and scales. These results suggest that sycophantic behaviors correspond to distinct, independently steerable representations.
>
---
#### [new 014] Towards Atoms of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出“Atoms Theory”，旨在解决大语言模型（LLM）内部表示单位不明确的问题。通过定义原子表示、引入原子内积和理论分析，验证了原子在稀疏性和可恢复性上的优势，并在多个模型上实证其有效性，为模型的机制解释提供了理论基础。**

- **链接: [http://arxiv.org/pdf/2509.20784v1](http://arxiv.org/pdf/2509.20784v1)**

> **作者:** Chenhui Hu; Pengfei Cao; Yubo Chen; Kang Liu; Jun Zhao
>
> **摘要:** The fundamental units of internal representations in large language models (LLMs) remain undefined, limiting further understanding of their mechanisms. Neurons or features are often regarded as such units, yet neurons suffer from polysemy, while features face concerns of unreliable reconstruction and instability. To address this issue, we propose the Atoms Theory, which defines such units as atoms. We introduce the atomic inner product (AIP) to correct representation shifting, formally define atoms, and prove the conditions that atoms satisfy the Restricted Isometry Property (RIP), ensuring stable sparse representations over atom set and linking to compressed sensing. Under stronger conditions, we further establish the uniqueness and exact $\ell_1$ recoverability of the sparse representations, and provide guarantees that single-layer sparse autoencoders (SAEs) with threshold activations can reliably identify the atoms. To validate the Atoms Theory, we train threshold-activated SAEs on Gemma2-2B, Gemma2-9B, and Llama3.1-8B, achieving 99.9% sparse reconstruction across layers on average, and more than 99.8% of atoms satisfy the uniqueness condition, compared to 0.5% for neurons and 68.2% for features, showing that atoms more faithfully capture intrinsic representations of LLMs. Scaling experiments further reveal the link between SAEs size and recovery capacity. Overall, this work systematically introduces and validates Atoms Theory of LLMs, providing a theoretical framework for understanding internal representations and a foundation for mechanistic interpretability. Code available at https://github.com/ChenhuiHu/towards_atoms.
>
---
#### [new 015] WeFT: Weighted Entropy-driven Fine-Tuning for dLLMs
- **分类: cs.CL**

- **简介: 该论文提出WeFT方法，用于监督微调扩散语言模型（dLLMs）。针对扩散模型在生成过程中预测不准确的问题，通过基于熵的加权训练提升模型表现，在多个推理任务上取得显著性能提升。**

- **链接: [http://arxiv.org/pdf/2509.20863v1](http://arxiv.org/pdf/2509.20863v1)**

> **作者:** Guowei Xu; Wenxin Xu; Jiawang Zhao; Kaisheng Ma
>
> **备注:** preprint
>
> **摘要:** Diffusion models have recently shown strong potential in language modeling, offering faster generation compared to traditional autoregressive approaches. However, applying supervised fine-tuning (SFT) to diffusion models remains challenging, as they lack precise probability estimates at each denoising step. While the diffusion mechanism enables the model to reason over entire sequences, it also makes the generation process less predictable and often inconsistent. This highlights the importance of controlling key tokens that guide the direction of generation. To address this issue, we propose WeFT, a weighted SFT method for diffusion language models, where tokens are assigned different weights based on their entropy. Derived from diffusion theory, WeFT delivers substantial gains: training on s1K, s1K-1.1, and 3k samples from open-r1, it achieves relative improvements of 39%, 64%, and 83% over standard SFT on four widely used reasoning benchmarks (Sudoku, Countdown, GSM8K, and MATH-500). The code and models will be made publicly available.
>
---
#### [new 016] Bounds of Chain-of-Thought Robustness: Reasoning Steps, Embed Norms, and Beyond
- **分类: cs.CL**

- **简介: 该论文研究链式推理（CoT）的鲁棒性，分析输入扰动对输出的影响。通过理论推导和实验验证，提出扰动上界与推理步骤数正相关、与嵌入向量范数负相关的结论，属于自然语言处理中的推理鲁棒性任务。**

- **链接: [http://arxiv.org/pdf/2509.21284v1](http://arxiv.org/pdf/2509.21284v1)**

> **作者:** Dingzirui Wang; Xuanliang Zhang; Keyan Xu; Qingfu Zhu; Wanxiang Che; Yang Deng
>
> **摘要:** Existing research indicates that the output of Chain-of-Thought (CoT) is significantly affected by input perturbations. Although many methods aim to mitigate such impact by optimizing prompts, a theoretical explanation of how these perturbations influence CoT outputs remains an open area of research. This gap limits our in-depth understanding of how input perturbations propagate during the reasoning process and hinders further improvements in prompt optimization methods. Therefore, in this paper, we theoretically analyze the effect of input perturbations on the fluctuation of CoT outputs. We first derive an upper bound for input perturbations under the condition that the output fluctuation is within an acceptable range, based on which we prove that: (i) This upper bound is positively correlated with the number of reasoning steps in the CoT; (ii) Even an infinitely long reasoning process cannot eliminate the impact of input perturbations. We then apply these conclusions to the Linear Self-Attention (LSA) model, which can be viewed as a simplified version of the Transformer. For the LSA model, we prove that the upper bound for input perturbation is negatively correlated with the norms of the input embedding and hidden state vectors. To validate this theoretical analysis, we conduct experiments on three mainstream datasets and four mainstream models. The experimental results align with our theoretical analysis, empirically demonstrating the correctness of our findings.
>
---
#### [new 017] Look Before you Leap: Estimating LLM Benchmark Scores from Descriptions
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究基于任务描述预测大模型基准得分的问题，构建了PRECOG数据集，并分析不同模型的预测能力，旨在提升评估效率与实验优先级决策。**

- **链接: [http://arxiv.org/pdf/2509.20645v1](http://arxiv.org/pdf/2509.20645v1)**

> **作者:** Jungsoo Park; Ethan Mendes; Gabriel Stanovsky; Alan Ritter
>
> **备注:** 24 pages, 6 figures
>
> **摘要:** Progress in large language models is constrained by an evaluation bottleneck: build a benchmark, evaluate models and settings, then iterate. We therefore ask a simple question: can we forecast outcomes before running any experiments? We study text-only performance forecasting: estimating a model's score from a redacted task description and intended configuration, with no access to dataset instances. To support systematic study, we curate PRECOG, a corpus of redacted description-performance pairs spanning diverse tasks, domains, and metrics. Experiments show the task is challenging but feasible: models equipped with a retrieval module that excludes source papers achieve moderate prediction performance with well-calibrated uncertainty, reaching mean absolute error as low as 8.7 on the Accuracy subset at high-confidence thresholds. Our analysis indicates that stronger reasoning models engage in diverse, iterative querying, whereas current open-source models lag and often skip retrieval or gather evidence with limited diversity. We further test a zero-leakage setting, forecasting on newly released datasets or experiments before their papers are indexed, where GPT-5 with built-in web search still attains nontrivial prediction accuracy. Overall, our corpus and analyses offer an initial step toward open-ended anticipatory evaluation, supporting difficulty estimation and smarter experiment prioritization.
>
---
#### [new 018] Hierarchical Resolution Transformers: A Wavelet-Inspired Architecture for Multi-Scale Language Understanding
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出Hierarchical Resolution Transformer (HRT)，针对传统Transformer在处理语言时忽略层次结构的问题，采用小波启发的多尺度架构，实现更高效的多分辨率语言理解，提升模型性能与效率。**

- **链接: [http://arxiv.org/pdf/2509.20581v1](http://arxiv.org/pdf/2509.20581v1)**

> **作者:** Ayan Sar; Sampurna Roy; Kanav Gupta; Anurag Kaushish; Tanupriya Choudhury; Abhijit Kumar
>
> **备注:** Submitted in IEEE International Conference on Big Data 2025
>
> **摘要:** Transformer architectures have achieved state-of-the-art performance across natural language tasks, yet they fundamentally misrepresent the hierarchical nature of human language by processing text as flat token sequences. This results in quadratic computational cost, weak computational cost, weak compositional generalization, and inadequate discourse-level modeling. We propose Hierarchical Resolution Transformer (HRT), a novel wavelet-inspired neural architecture that processes language simultaneously across multiple resolutions, from characters to discourse-level units. HRT constructs a multi-resolution attention, enabling bottom-up composition and top-down contextualization. By employing exponential sequence reduction across scales, HRT achieves O(nlogn) complexity, offering significant efficiency improvements over standard transformers. We evaluated HRT on a diverse suite of benchmarks, including GLUE, SuperGLUE, Long Range Arena, and WikiText-103, and results demonstrated that HRT outperforms standard transformer baselines by an average of +3.8% on GLUE, +4.5% on SuperGLUE, and +6.1% on Long Range Arena, while reducing memory usage by 42% and inference latency by 37% compared to BERT and GPT style models of similar parameter count. Ablation studies confirm the effectiveness of cross-resolution attention and scale-specialized modules, showing that each contributes independently to both efficiency and accuracy. Our findings establish HRT as the first architecture to align computational structure with the hierarchical organization of human language, demonstrating that multi-scale, wavelet-inspired processing yields both theoretical efficiency gains and practical improvements in language understanding.
>
---
#### [new 019] RLBFF: Binary Flexible Feedback to bridge between Human Feedback & Verifiable Rewards
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出RLBFF方法，结合人类反馈与可验证奖励的优势，提升大模型奖励模型的精度与灵活性。通过二分类原则训练奖励模型，在多个基准上取得领先效果，并提供开源方案实现高效对齐。**

- **链接: [http://arxiv.org/pdf/2509.21319v1](http://arxiv.org/pdf/2509.21319v1)**

> **作者:** Zhilin Wang; Jiaqi Zeng; Olivier Delalleau; Ellie Evans; Daniel Egert; Hoo-Chang Shin; Felipe Soares; Yi Dong; Oleksii Kuchaiev
>
> **摘要:** Reinforcement Learning with Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) are the main RL paradigms used in LLM post-training, each offering distinct advantages. However, RLHF struggles with interpretability and reward hacking because it relies on human judgments that usually lack explicit criteria, whereas RLVR is limited in scope by its focus on correctness-based verifiers. We propose Reinforcement Learning with Binary Flexible Feedback (RLBFF), which combines the versatility of human-driven preferences with the precision of rule-based verification, enabling reward models to capture nuanced aspects of response quality beyond mere correctness. RLBFF extracts principles that can be answered in a binary fashion (e.g. accuracy of information: yes, or code readability: no) from natural language feedback. Such principles can then be used to ground Reward Model training as an entailment task (response satisfies or does not satisfy an arbitrary principle). We show that Reward Models trained in this manner can outperform Bradley-Terry models when matched for data and achieve top performance on RM-Bench (86.2%) and JudgeBench (81.4%, #1 on leaderboard as of September 24, 2025). Additionally, users can specify principles of interest at inference time to customize the focus of our reward models, in contrast to Bradley-Terry models. Finally, we present a fully open source recipe (including data) to align Qwen3-32B using RLBFF and our Reward Model, to match or exceed the performance of o3-mini and DeepSeek R1 on general alignment benchmarks of MT-Bench, WildBench, and Arena Hard v2 (at <5% of the inference cost).
>
---
#### [new 020] Acoustic-based Gender Differentiation in Speech-aware Language Models
- **分类: cs.CL**

- **简介: 该论文研究语音感知语言模型中的声学性别偏见问题，提出了包含9208个语音样本的数据集，并分析发现模型在性别刻板印象问题上表现出男性导向的矛盾现象。研究确认这一偏差主要源自Whisper语音编码器生成的男性倾向声学标记，强调需改进语音技术中性别信息的处理方法。**

- **链接: [http://arxiv.org/pdf/2509.21125v1](http://arxiv.org/pdf/2509.21125v1)**

> **作者:** Junhyuk Choi; Jihwan Seol; Nayeon Kim; Chanhee Cho; EunBin Cho; Bugeun Kim
>
> **备注:** Under Review
>
> **摘要:** Speech-aware Language Models (SpeechLMs) have fundamentally transformed human-AI interaction by enabling voice-based communication, yet they may exhibit acoustic-based gender differentiation where identical questions lead to different responses based on the speaker's gender. This paper propose a new dataset that enables systematic analysis of this phenomenon, containing 9,208 speech samples across three categories: Gender-Independent, Gender-Stereotypical, and Gender-Dependent. We further evaluated LLaMA-Omni series and discovered a paradoxical pattern; while overall responses seems identical regardless of gender, the pattern is far from unbiased responses. Specifically, in Gender-Stereotypical questions, all models consistently exhibited male-oriented responses; meanwhile, in Gender-Dependent questions where gender differentiation would be contextually appropriate, models exhibited responses independent to gender instead. We also confirm that this pattern does not result from neutral options nor perceived gender of a voice. When we allow neutral response, models tends to respond neutrally also in Gender-Dependent questions. The paradoxical pattern yet retains when we applied gender neutralization methods on speech. Through comparison between SpeechLMs with corresponding backbone LLMs, we confirmed that these paradoxical patterns primarily stem from Whisper speech encoders, which generates male-oriented acoustic tokens. These findings reveal that current SpeechLMs may not successfully remove gender biases though they prioritized general fairness principles over contextual appropriateness, highlighting the need for more sophisticated techniques to utilize gender information properly in speech technology.
>
---
#### [new 021] Analysis of instruction-based LLMs' capabilities to score and judge text-input problems in an academic setting
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究了基于指令的LLM在学术场景中自动评分文本输入题的能力，提出了五种评估方法，并通过实验对比发现参考辅助评估效果最佳，展示了AI作为学术辅助工具的潜力。**

- **链接: [http://arxiv.org/pdf/2509.20982v1](http://arxiv.org/pdf/2509.20982v1)**

> **作者:** Valeria Ramirez-Garcia; David de-Fitero-Dominguez; Antonio Garcia-Cabot; Eva Garcia-Lopez
>
> **摘要:** Large language models (LLMs) can act as evaluators, a role studied by methods like LLM-as-a-Judge and fine-tuned judging LLMs. In the field of education, LLMs have been studied as assistant tools for students and teachers. Our research investigates LLM-driven automatic evaluation systems for academic Text-Input Problems using rubrics. We propose five evaluation systems that have been tested on a custom dataset of 110 answers about computer science from higher education students with three models: JudgeLM, Llama-3.1-8B and DeepSeek-R1-Distill-Llama-8B. The evaluation systems include: The JudgeLM evaluation, which uses the model's single answer prompt to obtain a score; Reference Aided Evaluation, which uses a correct answer as a guide aside from the original context of the question; No Reference Evaluation, which ommits the reference answer; Additive Evaluation, which uses atomic criteria; and Adaptive Evaluation, which is an evaluation done with generated criteria fitted to each question. All evaluation methods have been compared with the results of a human evaluator. Results show that the best method to automatically evaluate and score Text-Input Problems using LLMs is Reference Aided Evaluation. With the lowest median absolute deviation (0.945) and the lowest root mean square deviation (1.214) when compared to human evaluation, Reference Aided Evaluation offers fair scoring as well as insightful and complete evaluations. Other methods such as Additive and Adaptive Evaluation fail to provide good results in concise answers, No Reference Evaluation lacks information needed to correctly assess questions and JudgeLM Evaluations have not provided good results due to the model's limitations. As a result, we conclude that Artificial Intelligence-driven automatic evaluation systems, aided with proper methodologies, show potential to work as complementary tools to other academic resources.
>
---
#### [new 022] Document Summarization with Conformal Importance Guarantees
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究文档摘要任务，旨在解决关键内容遗漏问题。提出Conformal Importance Summarization框架，利用符合预测提供严格覆盖保证，确保重要信息被保留，适用于医疗、法律等高风险领域。**

- **链接: [http://arxiv.org/pdf/2509.20461v1](http://arxiv.org/pdf/2509.20461v1)**

> **作者:** Bruce Kuwahara; Chen-Yuan Lin; Xiao Shi Huang; Kin Kwan Leung; Jullian Arta Yapeter; Ilya Stanevich; Felipe Perez; Jesse C. Cresswell
>
> **备注:** NeurIPS 2025. Code is available at https://github.com/layer6ai-labs/conformal-importance-summarization
>
> **摘要:** Automatic summarization systems have advanced rapidly with large language models (LLMs), yet they still lack reliable guarantees on inclusion of critical content in high-stakes domains like healthcare, law, and finance. In this work, we introduce Conformal Importance Summarization, the first framework for importance-preserving summary generation which uses conformal prediction to provide rigorous, distribution-free coverage guarantees. By calibrating thresholds on sentence-level importance scores, we enable extractive document summarization with user-specified coverage and recall rates over critical content. Our method is model-agnostic, requires only a small calibration set, and seamlessly integrates with existing black-box LLMs. Experiments on established summarization benchmarks demonstrate that Conformal Importance Summarization achieves the theoretically assured information coverage rate. Our work suggests that Conformal Importance Summarization can be combined with existing techniques to achieve reliable, controllable automatic summarization, paving the way for safer deployment of AI summarization tools in critical applications. Code is available at https://github.com/layer6ai-labs/conformal-importance-summarization.
>
---
#### [new 023] The role of synthetic data in Multilingual, Multi-cultural AI systems: Lessons from Indic Languages
- **分类: cs.CL**

- **简介: 该论文研究了合成数据在多语言、多文化AI系统中的作用，重点解决低资源语言（印度语言）中数据不足的问题。通过基于维基百科的自下而上生成策略，构建了一个包含9.5M数据点的高质量指令跟随数据集Updesh，并验证其在生成任务和NLU任务上的有效性，尤其提升了低资源语言的表现。**

- **链接: [http://arxiv.org/pdf/2509.21294v1](http://arxiv.org/pdf/2509.21294v1)**

> **作者:** Pranjal A. Chitale; Varun Gumma; Sanchit Ahuja; Prashant Kodali; Manan Uppadhyay; Deepthi Sudharsan; Sunayana Sitaram
>
> **备注:** Under Review
>
> **摘要:** Developing AI systems that operate effectively across languages while remaining culturally grounded is a long-standing challenge, particularly in low-resource settings. Synthetic data provides a promising avenue, yet its effectiveness in multilingual and multicultural contexts remains underexplored. We investigate the creation and impact of synthetic, culturally contextualized datasets for Indian languages through a bottom-up generation strategy that prompts large open-source LLMs (>= 235B parameters) to ground data generation in language-specific Wikipedia content. This approach complements the dominant top-down paradigm of translating synthetic datasets from high-resource languages such as English. We introduce Updesh, a high-quality large-scale synthetic instruction-following dataset comprising 9.5M data points across 13 Indian languages, encompassing diverse reasoning and generative tasks with an emphasis on long-context, multi-turn capabilities, and alignment with Indian cultural contexts. A comprehensive evaluation incorporating both automated metrics and human annotation across 10k assessments indicates that generated data is high quality; though, human evaluation highlights areas for further improvement. Additionally, we perform downstream evaluations by fine-tuning models on our dataset and assessing the performance across 15 diverse multilingual datasets. Models trained on Updesh consistently achieve significant gains on generative tasks and remain competitive on multiple-choice style NLU tasks. Notably, relative improvements are most pronounced in low and medium-resource languages, narrowing their gap with high-resource languages. These findings provide empirical evidence that effective multilingual AI requires multi-faceted data curation and generation strategies that incorporate context-aware, culturally grounded methodologies.
>
---
#### [new 024] Leveraging What's Overfixed: Post-Correction via LLM Grammatical Error Overcorrection
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对语法错误修正（GEC）任务，旨在解决小模型高精度低召回与大模型低精度高召回的问题。提出PoCO方法，先利用LLM过修正提升召回，再通过微调小模型进行后修正，以平衡精度与召回，提高整体修正质量。**

- **链接: [http://arxiv.org/pdf/2509.20811v1](http://arxiv.org/pdf/2509.20811v1)**

> **作者:** Taehee Park; Heejin Do; Gary Geunbae Lee
>
> **备注:** EMNLP 2025
>
> **摘要:** Robust supervised fine-tuned small Language Models (sLMs) often show high reliability but tend to undercorrect. They achieve high precision at the cost of low recall. Conversely, Large Language Models (LLMs) often show the opposite tendency, making excessive overcorrection, leading to low precision. To effectively harness the strengths of LLMs to address the recall challenges in sLMs, we propose Post-Correction via Overcorrection (PoCO), a novel approach that strategically balances recall and precision. PoCO first intentionally triggers overcorrection via LLM to maximize recall by allowing comprehensive revisions, then applies a targeted post-correction step via fine-tuning smaller models to identify and refine erroneous outputs. We aim to harmonize both aspects by leveraging the generative power of LLMs while preserving the reliability of smaller supervised models. Our extensive experiments demonstrate that PoCO effectively balances GEC performance by increasing recall with competitive precision, ultimately improving the overall quality of grammatical error correction.
>
---
#### [new 025] Zero-Shot Privacy-Aware Text Rewriting via Iterative Tree Search
- **分类: cs.CL**

- **简介: 该论文提出一种零样本、基于树搜索的迭代文本重写方法，用于隐私保护。针对现有文本匿名化技术在隐私与自然性之间难以平衡的问题，通过奖励模型引导搜索，实现敏感信息模糊化，同时保持文本连贯性和实用性。**

- **链接: [http://arxiv.org/pdf/2509.20838v1](http://arxiv.org/pdf/2509.20838v1)**

> **作者:** Shuo Huang; Xingliang Yuan; Gholamreza Haffari; Lizhen Qu
>
> **摘要:** The increasing adoption of large language models (LLMs) in cloud-based services has raised significant privacy concerns, as user inputs may inadvertently expose sensitive information. Existing text anonymization and de-identification techniques, such as rule-based redaction and scrubbing, often struggle to balance privacy preservation with text naturalness and utility. In this work, we propose a zero-shot, tree-search-based iterative sentence rewriting algorithm that systematically obfuscates or deletes private information while preserving coherence, relevance, and naturalness. Our method incrementally rewrites privacy-sensitive segments through a structured search guided by a reward model, enabling dynamic exploration of the rewriting space. Experiments on privacy-sensitive datasets show that our approach significantly outperforms existing baselines, achieving a superior balance between privacy protection and utility preservation.
>
---
#### [new 026] AutoIntent: AutoML for Text Classification
- **分类: cs.CL**

- **简介: 该论文提出了AutoIntent，一个面向文本分类任务的自动化机器学习工具。它解决了端到端自动化建模的问题，实现了嵌入模型选择、分类器优化和阈值调整，并支持多标签分类与范围外检测。**

- **链接: [http://arxiv.org/pdf/2509.21138v1](http://arxiv.org/pdf/2509.21138v1)**

> **作者:** Ilya Alekseev; Roman Solomatin; Darina Rustamova; Denis Kuznetsov
>
> **备注:** EMNLP 2025 System demonstrations
>
> **摘要:** AutoIntent is an automated machine learning tool for text classification tasks. Unlike existing solutions, AutoIntent offers end-to-end automation with embedding model selection, classifier optimization, and decision threshold tuning, all within a modular, sklearn-like interface. The framework is designed to support multi-label classification and out-of-scope detection. AutoIntent demonstrates superior performance compared to existing AutoML tools on standard intent classification datasets and enables users to balance effectiveness and resource consumption.
>
---
#### [new 027] USB-Rec: An Effective Framework for Improving Conversational Recommendation Capability of Large Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对对话推荐系统中的大语言模型（LLM）能力提升问题，提出USB-Rec框架。通过设计偏好优化数据集和自增强策略，在模型层面改进LLM的训练与推理表现，实验表明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.20381v1](http://arxiv.org/pdf/2509.20381v1)**

> **作者:** Jianyu Wen; Jingyun Wang; Cilin Yan; Jiayin Cai; Xiaolong Jiang; Ying Zhang
>
> **备注:** Accepted by Recsys'25
>
> **摘要:** Recently, Large Language Models (LLMs) have been widely employed in Conversational Recommender Systems (CRSs). Unlike traditional language model approaches that focus on training, all existing LLMs-based approaches are mainly centered around how to leverage the summarization and analysis capabilities of LLMs while ignoring the issue of training. Therefore, in this work, we propose an integrated training-inference framework, User-Simulator-Based framework (USB-Rec), for improving the performance of LLMs in conversational recommendation at the model level. Firstly, we design a LLM-based Preference Optimization (PO) dataset construction strategy for RL training, which helps the LLMs understand the strategies and methods in conversational recommendation. Secondly, we propose a Self-Enhancement Strategy (SES) at the inference stage to further exploit the conversational recommendation potential obtained from RL training. Extensive experiments on various datasets demonstrate that our method consistently outperforms previous state-of-the-art methods.
>
---
#### [new 028] Beyond Global Emotion: Fine-Grained Emotional Speech Synthesis with Dynamic Word-Level Modulation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对情感语音合成任务，旨在解决现有方法无法捕捉句内动态情感变化的问题。提出Emo-FiLM框架，通过词级情感标注与FiLM层实现细粒度情感控制，并构建FEDD数据集进行评估。**

- **链接: [http://arxiv.org/pdf/2509.20378v1](http://arxiv.org/pdf/2509.20378v1)**

> **作者:** Sirui Wang; Andong Chen; Tiejun Zhao
>
> **摘要:** Emotional text-to-speech (E-TTS) is central to creating natural and trustworthy human-computer interaction. Existing systems typically rely on sentence-level control through predefined labels, reference audio, or natural language prompts. While effective for global emotion expression, these approaches fail to capture dynamic shifts within a sentence. To address this limitation, we introduce Emo-FiLM, a fine-grained emotion modeling framework for LLM-based TTS. Emo-FiLM aligns frame-level features from emotion2vec to words to obtain word-level emotion annotations, and maps them through a Feature-wise Linear Modulation (FiLM) layer, enabling word-level emotion control by directly modulating text embeddings. To support evaluation, we construct the Fine-grained Emotion Dynamics Dataset (FEDD) with detailed annotations of emotional transitions. Experiments show that Emo-FiLM outperforms existing approaches on both global and fine-grained tasks, demonstrating its effectiveness and generality for expressive speech synthesis.
>
---
#### [new 029] Generative AI for FFRDCs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何利用生成式AI提升联邦资助研发中心（FFRDCs）处理文本密集型任务的效率，如摘要、分类和信息提取。针对政府敏感场景，提出基于OnPrem$.$LLM框架的应用方案，并通过国防与科研案例验证其在保障安全性和可审计性下的实用性。**

- **链接: [http://arxiv.org/pdf/2509.21040v1](http://arxiv.org/pdf/2509.21040v1)**

> **作者:** Arun S. Maiya
>
> **备注:** 4
>
> **摘要:** Federally funded research and development centers (FFRDCs) face text-heavy workloads, from policy documents to scientific and engineering papers, that are slow to analyze manually. We show how large language models can accelerate summarization, classification, extraction, and sense-making with only a few input-output examples. To enable use in sensitive government contexts, we apply OnPrem$.$LLM, an open-source framework for secure and flexible application of generative AI. Case studies on defense policy documents and scientific corpora, including the National Defense Authorization Act (NDAA) and National Science Foundation (NSF) Awards, demonstrate how this approach enhances oversight and strategic analysis while maintaining auditability and data sovereignty.
>
---
#### [new 030] Assessing Classical Machine Learning and Transformer-based Approaches for Detecting AI-Generated Research Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI生成文本检测任务，旨在解决学术诚信与信息真实性问题。研究比较了经典机器学习和基于Transformer的模型（如DistilBERT）在区分AI与人类撰写科研摘要中的表现，并测试模型集成效果。结果表明，DistilBERT性能最佳，单一模型优于集成模型。**

- **链接: [http://arxiv.org/pdf/2509.20375v1](http://arxiv.org/pdf/2509.20375v1)**

> **作者:** Sharanya Parimanoharan; Ruwan D. Nawarathna
>
> **摘要:** The rapid adoption of large language models (LLMs) such as ChatGPT has blurred the line between human and AI-generated texts, raising urgent questions about academic integrity, intellectual property, and the spread of misinformation. Thus, reliable AI-text detection is needed for fair assessment to safeguard human authenticity and cultivate trust in digital communication. In this study, we investigate how well current machine learning (ML) approaches can distinguish ChatGPT-3.5-generated texts from human-written texts employing a labeled data set of 250 pairs of abstracts from a wide range of research topics. We test and compare both classical (Logistic Regression armed with classical Bag-of-Words, POS, and TF-IDF features) and transformer-based (BERT augmented with N-grams, DistilBERT, BERT with a lightweight custom classifier, and LSTM-based N-gram models) ML detection techniques. As we aim to assess each model's performance in detecting AI-generated research texts, we also aim to test whether an ensemble of these models can outperform any single detector. Results show DistilBERT achieves the overall best performance, while Logistic Regression and BERT-Custom offer solid, balanced alternatives; LSTM- and BERT-N-gram approaches lag. The max voting ensemble of the three best models fails to surpass DistilBERT itself, highlighting the primacy of a single transformer-based representation over mere model diversity. By comprehensively assessing the strengths and weaknesses of these AI-text detection approaches, this work lays a foundation for more robust transformer frameworks with larger, richer datasets to keep pace with ever-improving generative AI models.
>
---
#### [new 031] VoiceBBQ: Investigating Effect of Content and Acoustics in Social Bias of Spoken Language Model
- **分类: cs.CL**

- **简介: 该论文提出VoiceBBQ，一个用于评估语音语言模型社会偏见的基准数据集。任务是检测语音内容和声学特征引发的社会偏见。通过对比两个模型，揭示了其在处理性别、口音和声学偏差上的差异。**

- **链接: [http://arxiv.org/pdf/2509.21108v1](http://arxiv.org/pdf/2509.21108v1)**

> **作者:** Junhyuk Choi; Ro-hoon Oh; Jihwan Seol; Bugeun Kim
>
> **备注:** Accepted EMNLP 2025 main
>
> **摘要:** We introduce VoiceBBQ, a spoken extension of the BBQ (Bias Benchmark for Question Answering) - a dataset that measures social bias by presenting ambiguous or disambiguated contexts followed by questions that may elicit stereotypical responses. Due to the nature of speech, social bias in Spoken Language Models (SLMs) can emerge from two distinct sources: 1) content aspect and 2) acoustic aspect. The dataset converts every BBQ context into controlled voice conditions, enabling per-axis accuracy, bias, and consistency scores that remain comparable to the original text benchmark. Using VoiceBBQ, we evaluate two SLMs - LLaMA-Omni and Qwen2-Audio - and observe architectural contrasts: LLaMA-Omni resists acoustic bias while amplifying gender and accent bias, whereas Qwen2-Audio substantially dampens these cues while preserving content fidelity. VoiceBBQ thus provides a compact, drop-in testbed for jointly diagnosing content and acoustic bias across spoken language models.
>
---
#### [new 032] Building Tailored Speech Recognizers for Japanese Speaking Assessment
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究针对日语口语评估的定制语音识别任务，旨在解决音素标注（含重音标记）数据稀缺问题。提出了多任务学习和估计器融合方法，有效降低了音节标注错误率。**

- **链接: [http://arxiv.org/pdf/2509.20655v1](http://arxiv.org/pdf/2509.20655v1)**

> **作者:** Yotaro Kubo; Richard Sproat; Chihiro Taguchi; Llion Jones
>
> **摘要:** This paper presents methods for building speech recognizers tailored for Japanese speaking assessment tasks. Specifically, we build a speech recognizer that outputs phonemic labels with accent markers. Although Japanese is resource-rich, there is only a small amount of data for training models to produce accurate phonemic transcriptions that include accent marks. We propose two methods to mitigate data sparsity. First, a multitask training scheme introduces auxiliary loss functions to estimate orthographic text labels and pitch patterns of the input signal, so that utterances with only orthographic annotations can be leveraged in training. The second fuses two estimators, one over phonetic alphabet strings, and the other over text token sequences. To combine these estimates we develop an algorithm based on the finite-state transducer framework. Our results indicate that the use of multitask learning and fusion is effective for building an accurate phonemic recognizer. We show that this approach is advantageous compared to the use of generic multilingual recognizers. The relative advantages of the proposed methods were also compared. Our proposed methods reduced the average of mora-label error rates from 12.3% to 7.1% over the CSJ core evaluation sets.
>
---
#### [new 033] PerHalluEval: Persian Hallucination Evaluation Benchmark for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出PerHalluEval，首个针对波斯语的幻觉评估基准。任务为检测大语言模型在问答与摘要中的内外部幻觉。通过LLM生成数据、人工验证，并评估12个模型表现，发现提供外部知识可部分缓解幻觉问题。**

- **链接: [http://arxiv.org/pdf/2509.21104v1](http://arxiv.org/pdf/2509.21104v1)**

> **作者:** Mohammad Hosseini; Kimia Hosseini; Shayan Bali; Zahra Zanjani; Saeedeh Momtazi
>
> **摘要:** Hallucination is a persistent issue affecting all large language Models (LLMs), particularly within low-resource languages such as Persian. PerHalluEval (Persian Hallucination Evaluation) is the first dynamic hallucination evaluation benchmark tailored for the Persian language. Our benchmark leverages a three-stage LLM-driven pipeline, augmented with human validation, to generate plausible answers and summaries regarding QA and summarization tasks, focusing on detecting extrinsic and intrinsic hallucinations. Moreover, we used the log probabilities of generated tokens to select the most believable hallucinated instances. In addition, we engaged human annotators to highlight Persian-specific contexts in the QA dataset in order to evaluate LLMs' performance on content specifically related to Persian culture. Our evaluation of 12 LLMs, including open- and closed-source models using PerHalluEval, revealed that the models generally struggle in detecting hallucinated Persian text. We showed that providing external knowledge, i.e., the original document for the summarization task, could mitigate hallucination partially. Furthermore, there was no significant difference in terms of hallucination when comparing LLMs specifically trained for Persian with others.
>
---
#### [new 034] RedHerring Attack: Testing the Reliability of Attack Detection
- **分类: cs.CL; I.2.7**

- **简介: 该论文研究攻击检测模型的可靠性，提出RedHerring攻击方法，在不改变分类器结果的前提下误导检测模型。实验表明此攻击显著降低检测准确率，并提出简单防御策略以提升检测效果。**

- **链接: [http://arxiv.org/pdf/2509.20691v1](http://arxiv.org/pdf/2509.20691v1)**

> **作者:** Jonathan Rusert
>
> **备注:** 16 pages, 3 figures, Accepted to EMNLP 2025
>
> **摘要:** In response to adversarial text attacks, attack detection models have been proposed and shown to successfully identify text modified by adversaries. Attack detection models can be leveraged to provide an additional check for NLP models and give signals for human input. However, the reliability of these models has not yet been thoroughly explored. Thus, we propose and test a novel attack setting and attack, RedHerring. RedHerring aims to make attack detection models unreliable by modifying a text to cause the detection model to predict an attack, while keeping the classifier correct. This creates a tension between the classifier and detector. If a human sees that the detector is giving an ``incorrect'' prediction, but the classifier a correct one, then the human will see the detector as unreliable. We test this novel threat model on 4 datasets against 3 detectors defending 4 classifiers. We find that RedHerring is able to drop detection accuracy between 20 - 71 points, while maintaining (or improving) classifier accuracy. As an initial defense, we propose a simple confidence check which requires no retraining of the classifier or detector and increases detection accuracy greatly. This novel threat model offers new insights into how adversaries may target detection models.
>
---
#### [new 035] Which Cultural Lens Do Models Adopt? On Cultural Positioning Bias and Agentic Mitigation in LLMs
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究大型语言模型（LLMs）中的文化立场偏差问题，提出CultureLens基准测试和两种缓解方法（FIP和MFA框架），通过生成文化情境下的采访脚本任务评估并减轻模型对主流与非主流文化的偏倚。**

- **链接: [http://arxiv.org/pdf/2509.21080v1](http://arxiv.org/pdf/2509.21080v1)**

> **作者:** Yixin Wan; Xingrun Chen; Kai-Wei Chang
>
> **摘要:** Large language models (LLMs) have unlocked a wide range of downstream generative applications. However, we found that they also risk perpetuating subtle fairness issues tied to culture, positioning their generations from the perspectives of the mainstream US culture while demonstrating salient externality towards non-mainstream ones. In this work, we identify and systematically investigate this novel culture positioning bias, in which an LLM's default generative stance aligns with a mainstream view and treats other cultures as outsiders. We propose the CultureLens benchmark with 4000 generation prompts and 3 evaluation metrics for quantifying this bias through the lens of a culturally situated interview script generation task, in which an LLM is positioned as an onsite reporter interviewing local people across 10 diverse cultures. Empirical evaluation on 5 state-of-the-art LLMs reveals a stark pattern: while models adopt insider tones in over 88 percent of US-contexted scripts on average, they disproportionately adopt mainly outsider stances for less dominant cultures. To resolve these biases, we propose 2 inference-time mitigation methods: a baseline prompt-based Fairness Intervention Pillars (FIP) method, and a structured Mitigation via Fairness Agents (MFA) framework consisting of 2 pipelines: (1) MFA-SA (Single-Agent) introduces a self-reflection and rewriting loop based on fairness guidelines. (2) MFA-MA (Multi-Agent) structures the process into a hierarchy of specialized agents: a Planner Agent(initial script generation), a Critique Agent (evaluates initial script against fairness pillars), and a Refinement Agent (incorporates feedback to produce a polished, unbiased script). Empirical results showcase the effectiveness of agent-based methods as a promising direction for mitigating biases in generative LLMs.
>
---
#### [new 036] Interpreting Public Sentiment in Diplomacy Events: A Counterfactual Analysis Framework Using Large Language Models
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出一种基于大语言模型的反事实分析框架，用于通过修改外交事件叙述来引导公众情绪由负面转向积极。任务属于情感分析与文本生成，旨在解决传统方法耗时且缺乏前瞻性的问题。工作包括构建数据集、训练预测模型及开发反事实生成算法，成功率达70%。**

- **链接: [http://arxiv.org/pdf/2509.20367v1](http://arxiv.org/pdf/2509.20367v1)**

> **作者:** Leyi Ouyang
>
> **备注:** 2 Figures, 7 Tables, 1 Algorithm
>
> **摘要:** Diplomatic events consistently prompt widespread public discussion and debate. Public sentiment plays a critical role in diplomacy, as a good sentiment provides vital support for policy implementation, helps resolve international issues, and shapes a nation's international image. Traditional methods for gauging public sentiment, such as large-scale surveys or manual content analysis of media, are typically time-consuming, labor-intensive, and lack the capacity for forward-looking analysis. We propose a novel framework that identifies specific modifications for diplomatic event narratives to shift public sentiment from negative to neutral or positive. First, we train a language model to predict public reaction towards diplomatic events. To this end, we construct a dataset comprising descriptions of diplomatic events and their associated public discussions. Second, guided by communication theories and in collaboration with domain experts, we predetermined several textual features for modification, ensuring that any alterations changed the event's narrative framing while preserving its core facts.We develop a counterfactual generation algorithm that employs a large language model to systematically produce modified versions of an original text. The results show that this framework successfully shifted public sentiment to a more favorable state with a 70\% success rate. This framework can therefore serve as a practical tool for diplomats, policymakers, and communication specialists, offering data-driven insights on how to frame diplomatic initiatives or report on events to foster a more desirable public sentiment.
>
---
#### [new 037] Single Answer is Not Enough: On Generating Ranked Lists with Medical Reasoning Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究如何使医学推理模型生成答案的排序列表，而非单一答案。任务是开放性问题回答，旨在解决当前模型仅输出单个答案的问题。工作包括探究提示和微调方法，并提出针对排序列表的奖励函数，评估其在多种格式下的表现。**

- **链接: [http://arxiv.org/pdf/2509.20866v1](http://arxiv.org/pdf/2509.20866v1)**

> **作者:** Pittawat Taveekitworachai; Natpatchara Pongjirapat; Krittaphas Chaisutyakorn; Piyalitt Ittichaiwong; Tossaporn Saengja; Kunat Pipatanakul
>
> **备注:** 51 pages, 27 figures
>
> **摘要:** This paper presents a systematic study on enabling medical reasoning models (MRMs) to generate ranked lists of answers for open-ended questions. Clinical decision-making rarely relies on a single answer but instead considers multiple options, reducing the risks of narrow perspectives. Yet current MRMs are typically trained to produce only one answer, even in open-ended settings. We propose an alternative format: ranked lists and investigate two approaches: prompting and fine-tuning. While prompting is a cost-effective way to steer an MRM's response, not all MRMs generalize well across different answer formats: choice, short text, and list answers. Based on our prompting findings, we train and evaluate MRMs using supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT). SFT teaches a model to imitate annotated responses, and RFT incentivizes exploration through the responses that maximize a reward. We propose new reward functions targeted at ranked-list answer formats, and conduct ablation studies for RFT. Our results show that while some SFT models generalize to certain answer formats, models trained with RFT are more robust across multiple formats. We also present a case study on a modified MedQA with multiple valid answers, finding that although MRMs might fail to select the benchmark's preferred ground truth, they can recognize valid answers. To the best of our knowledge, this is the first systematic investigation of approaches for enabling MRMs to generate answers as ranked lists. We hope this work provides a first step toward developing alternative answer formats that are beneficial beyond single answers in medical domains.
>
---
#### [new 038] SiniticMTError: A Machine Translation Dataset with Error Annotations for Sinitic Languages
- **分类: cs.CL**

- **简介: 该论文提出了SiniticMTError数据集，针对低资源的汉语方言（如粤语、吴语）机器翻译问题，提供错误标注（位置、类型、严重程度），用于提升翻译质量评估和错误感知生成研究。**

- **链接: [http://arxiv.org/pdf/2509.20557v1](http://arxiv.org/pdf/2509.20557v1)**

> **作者:** Hannah Liu; Junghyun Min; Ethan Yue Heng Cheung; Shou-Yi Hung; Syed Mekael Wasti; Runtong Liang; Shiyao Qian; Shizhao Zheng; Elsie Chan; Ka Ieng Charlotte Lo; Wing Yu Yip; Richard Tzong-Han Tsai; En-Shiun Annie Lee
>
> **备注:** Work in progress. 14 pages, 4 figures, 5 tables
>
> **摘要:** Despite major advances in machine translation (MT) in recent years, progress remains limited for many low-resource languages that lack large-scale training data and linguistic resources. Cantonese and Wu Chinese are two Sinitic examples, although each enjoys more than 80 million speakers around the world. In this paper, we introduce SiniticMTError, a novel dataset that builds on existing parallel corpora to provide error span, error type, and error severity annotations in machine-translated examples from English to Mandarin, Cantonese, and Wu Chinese. Our dataset serves as a resource for the MT community to utilize in fine-tuning models with error detection capabilities, supporting research on translation quality estimation, error-aware generation, and low-resource language evaluation. We report our rigorous annotation process by native speakers, with analyses on inter-annotator agreement, iterative feedback, and patterns in error type and severity.
>
---
#### [new 039] Retrieval over Classification: Integrating Relation Semantics for Multimodal Relation Extraction
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究多模态关系抽取任务，针对传统分类方法忽略结构约束和语义表达不足的问题，提出ROC框架，将关系抽取重构为基于语义检索的任务，结合实体类型、位置信息及自然语言描述，通过对比学习提升性能。**

- **链接: [http://arxiv.org/pdf/2509.21151v1](http://arxiv.org/pdf/2509.21151v1)**

> **作者:** Lei Hei; Tingjing Liao; Yingxin Pei; Yiyang Qi; Jiaqi Wang; Ruiting Li; Feiliang Ren
>
> **备注:** Accepted by EMNLP 2025 Main Conference
>
> **摘要:** Relation extraction (RE) aims to identify semantic relations between entities in unstructured text. Although recent work extends traditional RE to multimodal scenarios, most approaches still adopt classification-based paradigms with fused multimodal features, representing relations as discrete labels. This paradigm has two significant limitations: (1) it overlooks structural constraints like entity types and positional cues, and (2) it lacks semantic expressiveness for fine-grained relation understanding. We propose \underline{R}etrieval \underline{O}ver \underline{C}lassification (ROC), a novel framework that reformulates multimodal RE as a retrieval task driven by relation semantics. ROC integrates entity type and positional information through a multimodal encoder, expands relation labels into natural language descriptions using a large language model, and aligns entity-relation pairs via semantic similarity-based contrastive learning. Experiments show that our method achieves state-of-the-art performance on the benchmark datasets MNRE and MORE and exhibits stronger robustness and interpretability.
>
---
#### [new 040] Probability Distribution Collapse: A Critical Bottleneck to Compact Unsupervised Neural Grammar Induction
- **分类: cs.CL**

- **简介: 该论文研究无监督神经语法归纳任务，旨在解决概率分布塌缩导致的表达性瓶颈问题。作者分析了塌缩的成因，并提出塌缩缓解神经参数化方法，有效提升了解析性能并简化了语法结构。**

- **链接: [http://arxiv.org/pdf/2509.20734v1](http://arxiv.org/pdf/2509.20734v1)**

> **作者:** Jinwook Park; Kangil Kim
>
> **备注:** Accepted in EMNLP2025 Main, 12 pages, 7 figures, 9 tables
>
> **摘要:** Unsupervised neural grammar induction aims to learn interpretable hierarchical structures from language data. However, existing models face an expressiveness bottleneck, often resulting in unnecessarily large yet underperforming grammars. We identify a core issue, $\textit{probability distribution collapse}$, as the underlying cause of this limitation. We analyze when and how the collapse emerges across key components of neural parameterization and introduce a targeted solution, $\textit{collapse-relaxing neural parameterization}$, to mitigate it. Our approach substantially improves parsing performance while enabling the use of significantly more compact grammars across a wide range of languages, as demonstrated through extensive empirical analysis.
>
---
#### [new 041] SFT Doesn't Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs
- **分类: cs.CL**

- **简介: 该论文研究了大语言模型在领域特定微调（SFT）中的一般能力退化问题，提出小学习率和新的TALR方法能有效平衡领域性能与通用能力。**

- **链接: [http://arxiv.org/pdf/2509.20758v1](http://arxiv.org/pdf/2509.20758v1)**

> **作者:** Jiacheng Lin; Zhongruo Wang; Kun Qian; Tian Wang; Arvind Srinivasan; Hansi Zeng; Ruochen Jiao; Xie Zhou; Jiri Gesi; Dakuo Wang; Yufan Guo; Kai Zhong; Weiqi Zhang; Sujay Sanghavi; Changyou Chen; Hyokun Yun; Lihong Li
>
> **摘要:** Supervised Fine-Tuning (SFT) on domain-specific datasets is a common approach to adapt Large Language Models (LLMs) to specialized tasks but is often believed to degrade their general capabilities. In this work, we revisit this trade-off and present both empirical and theoretical insights. First, we show that SFT does not always hurt: using a smaller learning rate can substantially mitigate general performance degradation while preserving comparable target-domain performance. We then provide a theoretical analysis that explains these phenomena and further motivates a new method, Token-Adaptive Loss Reweighting (TALR). Building on this, and recognizing that smaller learning rates alone do not fully eliminate general-performance degradation in all cases, we evaluate a range of strategies for reducing general capability loss, including L2 regularization, LoRA, model averaging, FLOW, and our proposed TALR. Experimental results demonstrate that while no method completely eliminates the trade-off, TALR consistently outperforms these baselines in balancing domain-specific gains and general capabilities. Finally, we distill our findings into practical guidelines for adapting LLMs to new domains: (i) using a small learning rate to achieve a favorable trade-off, and (ii) when a stronger balance is further desired, adopt TALR as an effective strategy.
>
---
#### [new 042] LLMTrace: A Corpus for Classification and Fine-Grained Localization of AI-Written Text
- **分类: cs.CL**

- **简介: 该论文提出LLMTrace，一个双语（英俄）AI生成文本检测数据集，用于解决现有数据集语言单一、模型过时及缺乏细粒度标注的问题。其支持全文分类与AI段落定位任务，提供字符级标注，推动更精准的AI文本检测研究。**

- **链接: [http://arxiv.org/pdf/2509.21269v1](http://arxiv.org/pdf/2509.21269v1)**

> **作者:** Irina Tolstykh; Aleksandra Tsybina; Sergey Yakubson; Maksim Kuprashevich
>
> **摘要:** The widespread use of human-like text from Large Language Models (LLMs) necessitates the development of robust detection systems. However, progress is limited by a critical lack of suitable training data; existing datasets are often generated with outdated models, are predominantly in English, and fail to address the increasingly common scenario of mixed human-AI authorship. Crucially, while some datasets address mixed authorship, none provide the character-level annotations required for the precise localization of AI-generated segments within a text. To address these gaps, we introduce LLMTrace, a new large-scale, bilingual (English and Russian) corpus for AI-generated text detection. Constructed using a diverse range of modern proprietary and open-source LLMs, our dataset is designed to support two key tasks: traditional full-text binary classification (human vs. AI) and the novel task of AI-generated interval detection, facilitated by character-level annotations. We believe LLMTrace will serve as a vital resource for training and evaluating the next generation of more nuanced and practical AI detection models. The project page is available at \href{https://sweetdream779.github.io/LLMTrace-info/}{iitolstykh/LLMTrace}.
>
---
#### [new 043] DisCoCLIP: A Distributional Compositional Tensor Network Encoder for Vision-Language Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DisCoCLIP，一种结合CLIP视觉模型与张量网络文本编码器的多模态模型，旨在解决视觉-语言任务中语言组合结构被忽视的问题。通过显式编码句法结构并使用张量分解提升效率，显著提升了动词语义和词序敏感性。**

- **链接: [http://arxiv.org/pdf/2509.21287v1](http://arxiv.org/pdf/2509.21287v1)**

> **作者:** Kin Ian Lo; Hala Hawashin; Mina Abbaszadeh; Tilen Limback-Stokin; Hadi Wazni; Mehrnoosh Sadrzadeh
>
> **摘要:** Recent vision-language models excel at large-scale image-text alignment but often neglect the compositional structure of language, leading to failures on tasks that hinge on word order and predicate-argument structure. We introduce DisCoCLIP, a multimodal encoder that combines a frozen CLIP vision transformer with a novel tensor network text encoder that explicitly encodes syntactic structure. Sentences are parsed with a Combinatory Categorial Grammar parser to yield distributional word tensors whose contractions mirror the sentence's grammatical derivation. To keep the model efficient, high-order tensors are factorized with tensor decompositions, reducing parameter count from tens of millions to under one million. Trained end-to-end with a self-supervised contrastive loss, DisCoCLIP markedly improves sensitivity to verb semantics and word order: it raises CLIP's SVO-Probes verb accuracy from 77.6% to 82.4%, boosts ARO attribution and relation scores by over 9% and 4%, and achieves 93.7% on a newly introduced SVO-Swap benchmark. These results demonstrate that embedding explicit linguistic structure via tensor networks yields interpretable, parameter-efficient representations that substantially improve compositional reasoning in vision-language tasks.
>
---
#### [new 044] LLM Output Homogenization is Task Dependent
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究大语言模型输出同质化问题，提出任务依赖性是关键。针对不同任务类别定义输出多样性，构建任务分类体系，引入任务锚定功能多样性评估方法和采样技术，有效提升多样性同时保持质量。**

- **链接: [http://arxiv.org/pdf/2509.21267v1](http://arxiv.org/pdf/2509.21267v1)**

> **作者:** Shomik Jain; Jack Lanchantin; Maximilian Nickel; Karen Ullrich; Ashia Wilson; Jamelle Watson-Daniels
>
> **摘要:** A large language model can be less helpful if it exhibits output response homogenization. But whether two responses are considered homogeneous, and whether such homogenization is problematic, both depend on the task category. For instance, in objective math tasks, we often expect no variation in the final answer but anticipate variation in the problem-solving strategy. Whereas, for creative writing tasks, we may expect variation in key narrative components (e.g. plot, genre, setting, etc), beyond the vocabulary or embedding diversity produced by temperature-sampling. Previous work addressing output homogenization often fails to conceptualize diversity in a task-dependent way. We address this gap in the literature directly by making the following contributions. (1) We present a task taxonomy comprised of eight task categories that each have distinct conceptualizations of output homogenization. (2) We introduce task-anchored functional diversity to better evaluate output homogenization. (3) We propose a task-anchored sampling technique that increases functional diversity for task categories where homogenization is undesired, while preserving homogenization where it is desired. (4) We challenge the perceived existence of a diversity-quality trade-off by increasing functional diversity while maintaining response quality. Overall, we demonstrate how task dependence improves the evaluation and mitigation of output homogenization.
>
---
#### [new 045] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出FS-DFM，一种高效的少步数离散流匹配语言模型，旨在解决长文本生成中扩散模型采样步骤多、速度慢的问题。通过显式控制采样步数并结合教师指导，实现高质量且快速的文本生成。**

- **链接: [http://arxiv.org/pdf/2509.20624v1](http://arxiv.org/pdf/2509.20624v1)**

> **作者:** Amin Karimi Monsefi; Nikhil Bhendawade; Manuel Rafael Ciosici; Dominic Culver; Yizhe Zhang; Irina Belousova
>
> **摘要:** Autoregressive language models (ARMs) deliver strong likelihoods, but are inherently serial: they generate one token per forward pass, which limits throughput and inflates latency for long sequences. Diffusion Language Models (DLMs) parallelize across positions and thus appear promising for language generation, yet standard discrete diffusion typically needs hundreds to thousands of model evaluations to reach high quality, trading serial depth for iterative breadth. We introduce FS-DFM, Few-Step Discrete Flow-Matching. A discrete flow-matching model designed for speed without sacrificing quality. The core idea is simple: make the number of sampling steps an explicit parameter and train the model to be consistent across step budgets, so one big move lands where many small moves would. We pair this with a reliable update rule that moves probability in the right direction without overshooting, and with strong teacher guidance distilled from long-run trajectories. Together, these choices make few-step sampling stable, accurate, and easy to control. On language modeling benchmarks, FS-DFM with 8 sampling steps achieves perplexity parity with a 1,024-step discrete-flow baseline for generating 1,024 tokens using a similar-size model, delivering up to 128 times faster sampling and corresponding latency/throughput gains.
>
---
#### [new 046] MI-Fuse: Label Fusion for Unsupervised Domain Adaptation with Closed-Source Large-Audio Language Model
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文针对语音情感识别（SER）任务，研究在源域数据不可用、仅能通过API访问大模型的情况下，如何利用目标域无标签数据提升学生模型性能。提出MI-Fuse方法，融合大模型与辅助教师模型的预测，实现跨域自适应，实验表明效果优于基线和原始大模型。**

- **链接: [http://arxiv.org/pdf/2509.20706v1](http://arxiv.org/pdf/2509.20706v1)**

> **作者:** Hsiao-Ying Huang; Yi-Cheng Lin; Hung-yi Lee
>
> **备注:** 5 pages, 2 figures, 2 tables
>
> **摘要:** Large audio-language models (LALMs) show strong zero-shot ability on speech tasks, suggesting promise for speech emotion recognition (SER). However, SER in real-world deployments often fails under domain mismatch, where source data are unavailable and powerful LALMs are accessible only through an API. We ask: given only unlabeled target-domain audio and an API-only LALM, can a student model be adapted to outperform the LALM in the target domain? To this end, we propose MI-Fuse, a denoised label fusion framework that supplements the LALM with a source-domain trained SER classifier as an auxiliary teacher. The framework draws multiple stochastic predictions from both teachers, weights their mean distributions by mutual-information-based uncertainty, and stabilizes training with an exponential moving average teacher. Experiments across three public emotion datasets and six cross-domain transfers show consistent gains, with the student surpassing the LALM and outperforming the strongest baseline by 3.9%. This approach strengthens emotion-aware speech systems without sharing source data, enabling realistic adaptation.
>
---
#### [new 047] SciReasoner: Laying the Scientific Reasoning Ground Across Disciplines
- **分类: cs.CL**

- **简介: 该论文提出SciReasoner，一个跨学科的科学推理基础模型。通过预训练和强化学习，实现文本与科学格式的转换、知识提取、属性预测与分类等任务，提升跨领域泛化与准确性。**

- **链接: [http://arxiv.org/pdf/2509.21320v1](http://arxiv.org/pdf/2509.21320v1)**

> **作者:** Yizhou Wang; Chen Tang; Han Deng; Jiabei Xiao; Jiaqi Liu; Jianyu Wu; Jun Yao; Pengze Li; Encheng Su; Lintao Wang; Guohang Zhuang; Yuchen Ren; Ben Fei; Ming Hu; Xin Chen; Dongzhan Zhou; Junjun He; Xiangyu Yue; Zhenfei Yin; Jiamin Wu; Qihao Zheng; Yuhao Zhou; Huihui Xu; Chenglong Ma; Yan Lu; Wenlong Zhang; Chunfeng Song; Philip Torr; Shixiang Tang; Xinzhu Ma; Wanli Ouyang; Lei Bai
>
> **备注:** technical report
>
> **摘要:** We present a scientific reasoning foundation model that aligns natural language with heterogeneous scientific representations. The model is pretrained on a 206B-token corpus spanning scientific text, pure sequences, and sequence-text pairs, then aligned via SFT on 40M instructions, annealed cold-start bootstrapping to elicit long-form chain-of-thought, and reinforcement learning with task-specific reward shaping, which instills deliberate scientific reasoning. It supports four capability families, covering up to 103 tasks across workflows: (i) faithful translation between text and scientific formats, (ii) text/knowledge extraction, (iii) property prediction, (iv) property classification, (v) unconditional and conditional sequence generation and design. Compared with specialist systems, our approach broadens instruction coverage, improves cross-domain generalization, and enhances fidelity. We detail data curation and training and show that cross-discipline learning strengthens transfer and downstream reliability. The model, instruct tuning datasets and the evaluation code are open-sourced at https://huggingface.co/SciReason and https://github.com/open-sciencelab/SciReason.
>
---
#### [new 048] CFD-LLMBench: A Benchmark Suite for Evaluating Large Language Models in Computational Fluid Dynamics
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CFD-LLMBench，一个用于评估大语言模型在计算流体力学（CFD）中表现的基准套件。它通过三个组件全面测试LLM在CFD领域的知识、推理与实现能力，旨在推动复杂物理系统数值实验的自动化。**

- **链接: [http://arxiv.org/pdf/2509.20374v1](http://arxiv.org/pdf/2509.20374v1)**

> **作者:** Nithin Somasekharan; Ling Yue; Yadi Cao; Weichao Li; Patrick Emami; Pochinapeddi Sai Bhargav; Anurag Acharya; Xingyu Xie; Shaowu Pan
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong performance across general NLP tasks, but their utility in automating numerical experiments of complex physical system -- a critical and labor-intensive component -- remains underexplored. As the major workhorse of computational science over the past decades, Computational Fluid Dynamics (CFD) offers a uniquely challenging testbed for evaluating the scientific capabilities of LLMs. We introduce CFDLLMBench, a benchmark suite comprising three complementary components -- CFDQuery, CFDCodeBench, and FoamBench -- designed to holistically evaluate LLM performance across three key competencies: graduate-level CFD knowledge, numerical and physical reasoning of CFD, and context-dependent implementation of CFD workflows. Grounded in real-world CFD practices, our benchmark combines a detailed task taxonomy with a rigorous evaluation framework to deliver reproducible results and quantify LLM performance across code executability, solution accuracy, and numerical convergence behavior. CFDLLMBench establishes a solid foundation for the development and evaluation of LLM-driven automation of numerical experiments for complex physical systems. Code and data are available at https://github.com/NREL-Theseus/cfdllmbench/.
>
---
#### [new 049] Overcoming Black-box Attack Inefficiency with Hybrid and Dynamic Select Algorithms
- **分类: cs.CL; I.2.7**

- **简介: 该论文针对NLP模型的对抗性文本攻击任务，旨在解决黑盒攻击中查询次数多、效率低的问题。提出了Hybrid和Dynamic Select两种新策略，结合BinarySelect和GreedySelect的优势，有效减少攻击所需查询量，提升攻击效率。**

- **链接: [http://arxiv.org/pdf/2509.20699v1](http://arxiv.org/pdf/2509.20699v1)**

> **作者:** Abhinay Shankar Belde; Rohit Ramkumar; Jonathan Rusert
>
> **备注:** 34 pages, 3 figures, Accepted to Findings of EMNLP 2025
>
> **摘要:** Adversarial text attack research plays a crucial role in evaluating the robustness of NLP models. However, the increasing complexity of transformer-based architectures has dramatically raised the computational cost of attack testing, especially for researchers with limited resources (e.g., GPUs). Existing popular black-box attack methods often require a large number of queries, which can make them inefficient and impractical for researchers. To address these challenges, we propose two new attack selection strategies called Hybrid and Dynamic Select, which better combine the strengths of previous selection algorithms. Hybrid Select merges generalized BinarySelect techniques with GreedySelect by introducing a size threshold to decide which selection algorithm to use. Dynamic Select provides an alternative approach of combining the generalized Binary and GreedySelect by learning which lengths of texts each selection method should be applied to. This greatly reduces the number of queries needed while maintaining attack effectiveness (a limitation of BinarySelect). Across 4 datasets and 6 target models, our best method(sentence-level Hybrid Select) is able to reduce the number of required queries per attack up 25.82\% on average against both encoder models and LLMs, without losing the effectiveness of the attack.
>
---
#### [new 050] Distilling Many-Shot In-Context Learning into a Cheat Sheet
- **分类: cs.CL**

- **简介: 该论文研究了大语言模型中的上下文学习任务，旨在解决多示例学习计算成本高的问题。提出将多示例信息蒸馏为简洁的“速查表”，在推理时使用，实现性能相近但更高效的方法。**

- **链接: [http://arxiv.org/pdf/2509.20820v1](http://arxiv.org/pdf/2509.20820v1)**

> **作者:** Ukyo Honda; Soichiro Murakami; Peinan Zhang
>
> **备注:** Accepted to EMNLP 2025 (Findings)
>
> **摘要:** Recent advances in large language models (LLMs) enable effective in-context learning (ICL) with many-shot examples, but at the cost of high computational demand due to longer input tokens. To address this, we propose cheat-sheet ICL, which distills the information from many-shot ICL into a concise textual summary (cheat sheet) used as the context at inference time. Experiments on challenging reasoning tasks show that cheat-sheet ICL achieves comparable or better performance than many-shot ICL with far fewer tokens, and matches retrieval-based ICL without requiring test-time retrieval. These findings demonstrate that cheat-sheet ICL is a practical alternative for leveraging LLMs in downstream tasks.
>
---
#### [new 051] CLaw: Benchmarking Chinese Legal Knowledge in Large Language Models - A Fine-grained Corpus and Reasoning Analysis
- **分类: cs.CL**

- **简介: 该论文提出CLaw，一个针对中文法律知识的基准测试，用于评估大语言模型在法律条款检索与推理上的能力。通过构建细粒度法律条文语料库和案例推理实例，揭示当前模型在法律知识应用上的不足，并强调提升法律推理需结合精准检索与强推理能力。**

- **链接: [http://arxiv.org/pdf/2509.21208v1](http://arxiv.org/pdf/2509.21208v1)**

> **作者:** Xinzhe Xu; Liang Zhao; Hongshen Xu; Chen Chen
>
> **摘要:** Large Language Models (LLMs) are increasingly tasked with analyzing legal texts and citing relevant statutes, yet their reliability is often compromised by general pre-training that ingests legal texts without specialized focus, obscuring the true depth of their legal knowledge. This paper introduces CLaw, a novel benchmark specifically engineered to meticulously evaluate LLMs on Chinese legal knowledge and its application in reasoning. CLaw comprises two key components: (1) a comprehensive, fine-grained corpus of all 306 Chinese national statutes, segmented to the subparagraph level and incorporating precise historical revision timesteps for rigorous recall evaluation (64,849 entries), and (2) a challenging set of 254 case-based reasoning instances derived from China Supreme Court curated materials to assess the practical application of legal knowledge. Our empirical evaluation reveals that most contemporary LLMs significantly struggle to faithfully reproduce legal provisions. As accurate retrieval and citation of legal provisions form the basis of legal reasoning, this deficiency critically undermines the reliability of their responses. We contend that achieving trustworthy legal reasoning in LLMs requires a robust synergy of accurate knowledge retrieval--potentially enhanced through supervised fine-tuning (SFT) or retrieval-augmented generation (RAG)--and strong general reasoning capabilities. This work provides an essential benchmark and critical insights for advancing domain-specific LLM reasoning, particularly within the complex legal sphere.
>
---
#### [new 052] Enhancing Molecular Property Prediction with Knowledge from Large Language Models
- **分类: cs.CL**

- **简介: 该论文聚焦分子属性预测任务，旨在解决传统方法依赖人工特征和模型泛化能力不足的问题。提出融合大语言模型提取的知识与分子结构特征的新框架，通过生成领域知识和代码增强预测效果，实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2509.20664v1](http://arxiv.org/pdf/2509.20664v1)**

> **作者:** Peng Zhou; Lai Hou Tim; Zhixiang Cheng; Kun Xie; Chaoyi Li; Wei Liu; Xiangxiang Zeng
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Predicting molecular properties is a critical component of drug discovery. Recent advances in deep learning, particularly Graph Neural Networks (GNNs), have enabled end-to-end learning from molecular structures, reducing reliance on manual feature engineering. However, while GNNs and self-supervised learning approaches have advanced molecular property prediction (MPP), the integration of human prior knowledge remains indispensable, as evidenced by recent methods that leverage large language models (LLMs) for knowledge extraction. Despite their strengths, LLMs are constrained by knowledge gaps and hallucinations, particularly for less-studied molecular properties. In this work, we propose a novel framework that, for the first time, integrates knowledge extracted from LLMs with structural features derived from pre-trained molecular models to enhance MPP. Our approach prompts LLMs to generate both domain-relevant knowledge and executable code for molecular vectorization, producing knowledge-based features that are subsequently fused with structural representations. We employ three state-of-the-art LLMs, GPT-4o, GPT-4.1, and DeepSeek-R1, for knowledge extraction. Extensive experiments demonstrate that our integrated method outperforms existing approaches, confirming that the combination of LLM-derived knowledge and structural information provides a robust and effective solution for MPP.
>
---
#### [new 053] SwasthLLM: a Unified Cross-Lingual, Multi-Task, and Meta-Learning Zero-Shot Framework for Medical Diagnosis Using Contrastive Representations
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出SwasthLLM，一种用于多语言医疗诊断的零样本框架。针对低资源语言数据稀缺问题，结合跨语言对比学习与元学习，实现英语、印地语和孟加拉语的统一疾病分类，无需语言特定微调，在零样本场景下表现优异。**

- **链接: [http://arxiv.org/pdf/2509.20567v1](http://arxiv.org/pdf/2509.20567v1)**

> **作者:** Ayan Sar; Pranav Singh Puri; Sumit Aich; Tanupriya Choudhury; Abhijit Kumar
>
> **备注:** Submitted to International Conference on Big Data 2025
>
> **摘要:** In multilingual healthcare environments, automatic disease diagnosis from clinical text remains a challenging task due to the scarcity of annotated medical data in low-resource languages and the linguistic variability across populations. This paper proposes SwasthLLM, a unified, zero-shot, cross-lingual, and multi-task learning framework for medical diagnosis that operates effectively across English, Hindi, and Bengali without requiring language-specific fine-tuning. At its core, SwasthLLM leverages the multilingual XLM-RoBERTa encoder augmented with a language-aware attention mechanism and a disease classification head, enabling the model to extract medically relevant information regardless of the language structure. To align semantic representations across languages, a Siamese contrastive learning module is introduced, ensuring that equivalent medical texts in different languages produce similar embeddings. Further, a translation consistency module and a contrastive projection head reinforce language-invariant representation learning. SwasthLLM is trained using a multi-task learning strategy, jointly optimizing disease classification, translation alignment, and contrastive learning objectives. Additionally, we employ Model-Agnostic Meta-Learning (MAML) to equip the model with rapid adaptation capabilities for unseen languages or tasks with minimal data. Our phased training pipeline emphasizes robust representation alignment before task-specific fine-tuning. Extensive evaluation shows that SwasthLLM achieves high diagnostic performance, with a test accuracy of 97.22% and an F1-score of 97.17% in supervised settings. Crucially, in zero-shot scenarios, it attains 92.78% accuracy on Hindi and 73.33% accuracy on Bengali medical text, demonstrating strong generalization in low-resource contexts.
>
---
#### [new 054] SoM-1K: A Thousand-Problem Benchmark Dataset for Strength of Materials
- **分类: cs.CL**

- **简介: 该论文提出了SoM-1K，一个包含1,065个强度材料问题的多模态基准数据集，用于评估基础模型在工程任务中的表现。针对当前模型视觉理解能力有限的问题，提出DoI策略，通过文本描述替代图像输入。实验表明，提供DoI的LLM优于直接使用图像的VLM，突显了提升多模态推理能力的重要性。**

- **链接: [http://arxiv.org/pdf/2509.21079v1](http://arxiv.org/pdf/2509.21079v1)**

> **作者:** Qixin Wan; Zilong Wang; Jingwen Zhou; Wanting Wang; Ziheng Geng; Jiachen Liu; Ran Cao; Minghui Cheng; Lu Cheng
>
> **摘要:** Foundation models have shown remarkable capabilities in various domains, but their performance on complex, multimodal engineering problems remains largely unexplored. We introduce SoM-1K, the first large-scale multimodal benchmark dataset dedicated to evaluating foundation models on problems in the strength of materials (SoM). The dataset, which contains 1,065 annotated SoM problems, mirrors real-world engineering tasks by including both textual problem statements and schematic diagrams. Due to the limited capabilities of current foundation models in understanding complicated visual information, we propose a novel prompting strategy called Descriptions of Images (DoI), which provides rigorous expert-generated text descriptions of the visual diagrams as the context. We evaluate eight representative foundation models, including both large language models (LLMs) and vision language models (VLMs). Our results show that current foundation models struggle significantly with these engineering problems, with the best-performing model achieving only 56.6% accuracy. Interestingly, we found that LLMs, when provided with DoI, often outperform VLMs provided with visual diagrams. A detailed error analysis reveals that DoI plays a crucial role in mitigating visual misinterpretation errors, suggesting that accurate text-based descriptions can be more effective than direct image input for current foundation models. This work establishes a rigorous benchmark for engineering AI and highlights a critical need for developing more robust multimodal reasoning capabilities in foundation models, particularly in scientific and engineering contexts.
>
---
#### [new 055] ShortCheck: Checkworthiness Detection of Multilingual Short-Form Videos
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出ShortCheck，一个用于检测多语言短视频可信度的系统。针对TikTok等平台短视频多模态、动态且内容嘈杂的特点，解决虚假信息识别问题。系统整合语音转文字、OCR、对象检测、深度伪造检测、视频摘要和事实核查模块，有效辅助人工审核，实现70%以上的F1加权得分。**

- **链接: [http://arxiv.org/pdf/2509.20467v1](http://arxiv.org/pdf/2509.20467v1)**

> **作者:** Henrik Vatndal; Vinay Setty
>
> **摘要:** Short-form video platforms like TikTok present unique challenges for misinformation detection due to their multimodal, dynamic, and noisy content. We present ShortCheck, a modular, inference-only pipeline with a user-friendly interface that automatically identifies checkworthy short-form videos to help human fact-checkers. The system integrates speech transcription, OCR, object and deepfake detection, video-to-text summarization, and claim verification. ShortCheck is validated by evaluating it on two manually annotated datasets with TikTok videos in a multilingual setting. The pipeline achieves promising results with F1-weighted score over 70\%.
>
---
#### [new 056] Dynamic Reasoning Chains through Depth-Specialized Mixture-of-Experts in Transformer Architectures
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出DS-MoE，一种基于深度专业化专家混合的动态推理链框架，用于改进Transformer模型。旨在解决传统模型对所有输入采用相同处理深度导致的效率低下和推理质量受限问题。通过引入优化不同推理深度的专家模块和动态路由网络，实现计算资源的高效利用与推理性能提升。**

- **链接: [http://arxiv.org/pdf/2509.20577v1](http://arxiv.org/pdf/2509.20577v1)**

> **作者:** Sampurna Roy; Ayan Sar; Anurag Kaushish; Kanav Gupta; Tanupriya Choudhury; Abhijit Kumar
>
> **备注:** Submitted in IEEE International Conference on Big Data 2025
>
> **摘要:** Contemporary transformer architectures apply identical processing depth to all inputs, creating inefficiencies and limiting reasoning quality. Simple factual queries are subjected to the same multilayered computation as complex logical problems, wasting resources while constraining deep inference. To overcome this, we came up with a concept of Dynamic Reasoning Chains through Depth Specialised Mixture of Experts (DS-MoE), a modular framework that extends the Mixture of Experts paradigm from width-based to depth specialised computation. DS-MoE introduces expert modules optimised for distinct reasoning depths, shallow pattern recognition, compositional reasoning, logical inference, memory integration, and meta-cognitive supervision. A learned routing network dynamically assembles custom reasoning chains, activating only the necessary experts to match input complexity. The dataset on which we trained and evaluated DS-MoE is on The Pile, an 800GB corpus covering diverse domains such as scientific papers, legal texts, programming code, and web content, enabling systematic assessment across reasoning depths. Experimental results demonstrate that DS-MoE achieves up to 16 per cent computational savings and 35 per cent faster inference compared to uniform-depth transformers, while delivering 2.8 per cent higher accuracy on complex multi-step reasoning benchmarks. Furthermore, routing decisions yield interpretable reasoning chains, enhancing transparency and scalability. These findings establish DS-MoE as a significant advancement in adaptive neural architectures, demonstrating that depth-specialised modular processing can simultaneously improve efficiency, reasoning quality, and interpretability in large-scale language models.
>
---
#### [new 057] Few-Shot and Training-Free Review Generation via Conversational Prompting
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究的是**评论生成任务**，旨在解决**数据稀缺且不能训练模型**的场景下的个性化评论生成问题。作者提出了**对话式提示方法（Conversational Prompting）**，包括SCP和CCP两种变体，通过重构用户评论为多轮对话，提升LLM在少样本、无训练条件下的生成效果。**

- **链接: [http://arxiv.org/pdf/2509.20805v1](http://arxiv.org/pdf/2509.20805v1)**

> **作者:** Genki Kusano
>
> **摘要:** Personalized review generation helps businesses understand user preferences, yet most existing approaches assume extensive review histories of the target user or require additional model training. Real-world applications often face few-shot and training-free situations, where only a few user reviews are available and fine-tuning is infeasible. It is well known that large language models (LLMs) can address such low-resource settings, but their effectiveness depends on prompt engineering. In this paper, we propose Conversational Prompting, a lightweight method that reformulates user reviews as multi-turn conversations. Its simple variant, Simple Conversational Prompting (SCP), relies solely on the user's own reviews, while the contrastive variant, Contrastive Conversational Prompting (CCP), inserts reviews from other users or LLMs as incorrect replies and then asks the model to correct them, encouraging the model to produce text in the user's style. Experiments on eight product domains and five LLMs showed that the conventional non-conversational prompt often produced reviews similar to those written by random users, based on text-based metrics such as ROUGE-L and BERTScore, and application-oriented tasks like user identity matching and sentiment analysis. In contrast, both SCP and CCP produced reviews much closer to those of the target user, even when each user had only two reviews. CCP brings further improvements when high-quality negative examples are available, whereas SCP remains competitive when such data cannot be collected. These results suggest that conversational prompting offers a practical solution for review generation under few-shot and training-free constraints.
>
---
#### [new 058] SGMem: Sentence Graph Memory for Long-Term Conversational Agents
- **分类: cs.CL; cs.IR; I.2.7; H.3.3**

- **简介: 该论文提出SGMem，用于长期对话代理的记忆管理。针对大模型上下文窗口限制的问题，SGMem通过句子图结构组织多粒度对话信息，结合原始对话和生成记忆，提升长时对话问答的准确性。**

- **链接: [http://arxiv.org/pdf/2509.21212v1](http://arxiv.org/pdf/2509.21212v1)**

> **作者:** Yaxiong Wu; Yongyue Zhang; Sheng Liang; Yong Liu
>
> **备注:** 19 pages, 6 figures, 1 table
>
> **摘要:** Long-term conversational agents require effective memory management to handle dialogue histories that exceed the context window of large language models (LLMs). Existing methods based on fact extraction or summarization reduce redundancy but struggle to organize and retrieve relevant information across different granularities of dialogue and generated memory. We introduce SGMem (Sentence Graph Memory), which represents dialogue as sentence-level graphs within chunked units, capturing associations across turn-, round-, and session-level contexts. By combining retrieved raw dialogue with generated memory such as summaries, facts and insights, SGMem supplies LLMs with coherent and relevant context for response generation. Experiments on LongMemEval and LoCoMo show that SGMem consistently improves accuracy and outperforms strong baselines in long-term conversational question answering.
>
---
#### [new 059] BESPOKE: Benchmark for Search-Augmented Large Language Model Personalization via Diagnostic Feedback
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出BESPOKE，一个用于评估搜索增强型大语言模型个性化能力的基准。任务是信息检索与生成中的个性化。解决现有系统在满足用户多样化需求方面的不足。工作包括收集真实用户历史和反馈，并进行细致分析以指导个性化改进。**

- **链接: [http://arxiv.org/pdf/2509.21106v1](http://arxiv.org/pdf/2509.21106v1)**

> **作者:** Hyunseo Kim; Sangam Lee; Kwangwook Seo; Dongha Lee
>
> **备注:** Work in progress
>
> **摘要:** Search-augmented large language models (LLMs) have advanced information-seeking tasks by integrating retrieval into generation, reducing users' cognitive burden compared to traditional search systems. Yet they remain insufficient for fully addressing diverse user needs, which requires recognizing how the same query can reflect different intents across users and delivering information in preferred forms. While recent systems such as ChatGPT and Gemini attempt personalization by leveraging user histories, systematic evaluation of such personalization is under-explored. To address this gap, we propose BESPOKE, the realistic benchmark for evaluating personalization in search-augmented LLMs. BESPOKE is designed to be both realistic, by collecting authentic chat and search histories directly from humans, and diagnostic, by pairing responses with fine-grained preference scores and feedback. The benchmark is constructed through long-term, deeply engaged human annotation, where human annotators contributed their own histories, authored queries with detailed information needs, and evaluated responses with scores and diagnostic feedback. Leveraging BESPOKE, we conduct systematic analyses that reveal key requirements for effective personalization in information-seeking tasks, providing a foundation for fine-grained evaluation of personalized search-augmented LLMs. Our code and data are available at https://augustinlib.github.io/BESPOKE/.
>
---
#### [new 060] Speaker Style-Aware Phoneme Anchoring for Improved Cross-Lingual Speech Emotion Recognition
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究跨语言语音情感识别（SER）任务，旨在解决因语言和说话人表达风格差异导致的情感识别难题。提出了一种基于说话人风格的音素对齐框架，通过图聚类构建情感相关说话人群体，并在双空间进行锚定，提升跨语言情感表示的泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.20373v1](http://arxiv.org/pdf/2509.20373v1)**

> **作者:** Shreya G. Upadhyay; Carlos Busso; Chi-Chun Lee
>
> **摘要:** Cross-lingual speech emotion recognition (SER) remains a challenging task due to differences in phonetic variability and speaker-specific expressive styles across languages. Effectively capturing emotion under such diverse conditions requires a framework that can align the externalization of emotions across different speakers and languages. To address this problem, we propose a speaker-style aware phoneme anchoring framework that aligns emotional expression at the phonetic and speaker levels. Our method builds emotion-specific speaker communities via graph-based clustering to capture shared speaker traits. Using these groups, we apply dual-space anchoring in speaker and phonetic spaces to enable better emotion transfer across languages. Evaluations on the MSP-Podcast (English) and BIIC-Podcast (Taiwanese Mandarin) corpora demonstrate improved generalization over competitive baselines and provide valuable insights into the commonalities in cross-lingual emotion representation.
>
---
#### [new 061] When Instructions Multiply: Measuring and Estimating LLM Capabilities of Multiple Instructions Following
- **分类: cs.CL**

- **简介: 该论文研究大型语言模型（LLM）在多指令跟随任务中的能力，提出两个基准测试ManyIFEval和StyleMBPP，并构建回归模型估计性能，解决多指令下性能评估计算量大的问题。**

- **链接: [http://arxiv.org/pdf/2509.21051v1](http://arxiv.org/pdf/2509.21051v1)**

> **作者:** Keno Harada; Yudai Yamazaki; Masachika Taniguchi; Edison Marrese-Taylor; Takeshi Kojima; Yusuke Iwasawa; Yutaka Matsuo
>
> **备注:** Accepted to EMNLP2025
>
> **摘要:** As large language models (LLMs) are increasingly applied to real-world scenarios, it becomes crucial to understand their ability to follow multiple instructions simultaneously. To systematically evaluate these capabilities, we introduce two specialized benchmarks for fundamental domains where multiple instructions following is important: Many Instruction-Following Eval (ManyIFEval) for text generation with up to ten instructions, and Style-aware Mostly Basic Programming Problems (StyleMBPP) for code generation with up to six instructions. Our experiments with the created benchmarks across ten LLMs reveal that performance consistently degrades as the number of instructions increases. Furthermore, given the fact that evaluating all the possible combinations of multiple instructions is computationally impractical in actual use cases, we developed three types of regression models that can estimate performance on both unseen instruction combinations and different numbers of instructions which are not used during training. We demonstrate that a logistic regression model using instruction count as an explanatory variable can predict performance of following multiple instructions with approximately 10% error, even for unseen instruction combinations. We show that relatively modest sample sizes (500 for ManyIFEval and 300 for StyleMBPP) are sufficient for performance estimation, enabling efficient evaluation of LLMs under various instruction combinations.
>
---
#### [new 062] Confidence-guided Refinement Reasoning for Zero-shot Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出C2R框架，用于零样本问答任务。通过构建和优化子问题及答案，利用模型自身置信度提升最终答案可靠性，适用于文本、图像和视频领域，无需额外训练。**

- **链接: [http://arxiv.org/pdf/2509.20750v1](http://arxiv.org/pdf/2509.20750v1)**

> **作者:** Youwon Jang; Woo Suk Choi; Minjoon Jung; Minsu Lee; Byoung-Tak Zhang
>
> **备注:** 18 pages (including references and appendix)
>
> **摘要:** We propose Confidence-guided Refinement Reasoning (C2R), a novel training-free framework applicable to question-answering (QA) tasks across text, image, and video domains. C2R strategically constructs and refines sub-questions and their answers (sub-QAs), deriving a better confidence score for the target answer. C2R first curates a subset of sub-QAs to explore diverse reasoning paths, then compares the confidence scores of the resulting answer candidates to select the most reliable final answer. Since C2R relies solely on confidence scores derived from the model itself, it can be seamlessly integrated with various existing QA models, demonstrating consistent performance improvements across diverse models and benchmarks. Furthermore, we provide essential yet underexplored insights into how leveraging sub-QAs affects model behavior, specifically analyzing the impact of both the quantity and quality of sub-QAs on achieving robust and reliable reasoning.
>
---
#### [new 063] Un-Doubling Diffusion: LLM-guided Disambiguation of Homonym Duplication
- **分类: cs.CL**

- **简介: 该论文研究扩散模型在生成图像时因同形异义词（homonyms）导致的语义重复问题，提出自动评估方法并探索通过提示扩展缓解此问题的策略，涵盖英文中心偏见相关的同义词混淆现象。**

- **链接: [http://arxiv.org/pdf/2509.21262v1](http://arxiv.org/pdf/2509.21262v1)**

> **作者:** Evgeny Kaskov; Elizaveta Petrova; Petr Surovtsev; Anna Kostikova; Ilya Mistiurin; Alexander Kapitanov; Alexander Nagaev
>
> **摘要:** Homonyms are words with identical spelling but distinct meanings, which pose challenges for many generative models. When a homonym appears in a prompt, diffusion models may generate multiple senses of the word simultaneously, which is known as homonym duplication. This issue is further complicated by an Anglocentric bias, which includes an additional translation step before the text-to-image model pipeline. As a result, even words that are not homonymous in the original language may become homonyms and lose their meaning after translation into English. In this paper, we introduce a method for measuring duplication rates and conduct evaluations of different diffusion models using both automatic evaluation utilizing Vision-Language Models (VLM) and human evaluation. Additionally, we investigate methods to mitigate the homonym duplication problem through prompt expansion, demonstrating that this approach also effectively reduces duplication related to Anglocentric bias. The code for the automatic evaluation pipeline is publicly available.
>
---
#### [new 064] Cross-Linguistic Analysis of Memory Load in Sentence Comprehension: Linear Distance and Structural Density
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文属于心理语言学与计算语言学交叉任务，旨在探讨句子理解中记忆负荷的影响因素。研究通过跨语言分析，比较线性距离与结构密度对记忆负荷的作用，提出“Intervener Complexity”作为结构化指标，并结合统一依赖树库和混合效应模型验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.20916v1](http://arxiv.org/pdf/2509.20916v1)**

> **作者:** Krishna Aggarwal
>
> **备注:** 7 pages, 4 figures (Figure 2 has 3 sub-divisions)
>
> **摘要:** This study examines whether sentence-level memory load in comprehension is better explained by linear proximity between syntactically related words or by the structural density of the intervening material. Building on locality-based accounts and cross-linguistic evidence for dependency length minimization, the work advances Intervener Complexity-the number of intervening heads between a head and its dependent-as a structurally grounded lens that refines linear distance measures. Using harmonized dependency treebanks and a mixed-effects framework across multiple languages, the analysis jointly evaluates sentence length, dependency length, and Intervener Complexity as predictors of the Memory-load measure. Studies in Psycholinguistics have reported the contributions of feature interference and misbinding to memory load during processing. For this study, I operationalized sentence-level memory load as the linear sum of feature misbinding and feature interference for tractability; current evidence does not establish that their cognitive contributions combine additively. All three factors are positively associated with memory load, with sentence length exerting the broadest influence and Intervener Complexity offering explanatory power beyond linear distance. Conceptually, the findings reconcile linear and hierarchical perspectives on locality by treating dependency length as an important surface signature while identifying intervening heads as a more proximate indicator of integration and maintenance demands. Methodologically, the study illustrates how UD-based graph measures and cross-linguistic mixed-effects modelling can disentangle linear and structural contributions to processing efficiency, providing a principled path for evaluating competing theories of memory load in sentence comprehension.
>
---
#### [new 065] Learning to Summarize by Learning to Quiz: Adversarial Agentic Collaboration for Long Document Summarization
- **分类: cs.CL**

- **简介: 该论文提出SummQ，一种对抗性多智能体框架，用于长文档摘要任务。针对现有方法在信息丢失、事实不一致和连贯性方面的问题，通过摘要生成与问答机制的协作优化摘要质量，实验表明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.20900v1](http://arxiv.org/pdf/2509.20900v1)**

> **作者:** Weixuan Wang; Minghao Wu; Barry Haddow; Alexandra Birch
>
> **摘要:** Long document summarization remains a significant challenge for current large language models (LLMs), as existing approaches commonly struggle with information loss, factual inconsistencies, and coherence issues when processing excessively long documents. We propose SummQ, a novel adversarial multi-agent framework that addresses these limitations through collaborative intelligence between specialized agents operating in two complementary domains: summarization and quizzing. Our approach employs summary generators and reviewers that work collaboratively to create and evaluate comprehensive summaries, while quiz generators and reviewers create comprehension questions that serve as continuous quality checks for the summarization process. This adversarial dynamic, enhanced by an examinee agent that validates whether the generated summary contains the information needed to answer the quiz questions, enables iterative refinement through multifaceted feedback mechanisms. We evaluate SummQ on three widely used long document summarization benchmarks. Experimental results demonstrate that our framework significantly outperforms existing state-of-the-art methods across ROUGE and BERTScore metrics, as well as in LLM-as-a-Judge and human evaluations. Our comprehensive analyses reveal the effectiveness of the multi-agent collaboration dynamics, the influence of different agent configurations, and the impact of the quizzing mechanism. This work establishes a new approach for long document summarization that uses adversarial agentic collaboration to improve summarization quality.
>
---
#### [new 066] Tool Calling for Arabic LLMs: Data Strategies and Instruction Tuning
- **分类: cs.CL**

- **简介: 该论文研究阿拉伯语大模型的工具调用能力，探讨了语言数据、指令微调和特定工具优化对性能的影响。为弥补资源不足，作者将两个开源数据集翻译适配为阿拉伯语，并在开源阿拉伯语大模型上进行了实验分析。**

- **链接: [http://arxiv.org/pdf/2509.20957v1](http://arxiv.org/pdf/2509.20957v1)**

> **作者:** Asim Ersoy; Enes Altinisik; Husrev Taha Sencar; Kareem Darwish
>
> **摘要:** Tool calling is a critical capability that allows Large Language Models (LLMs) to interact with external systems, significantly expanding their utility. However, research and resources for tool calling are predominantly English-centric, leaving a gap in our understanding of how to enable this functionality for other languages, such as Arabic. This paper investigates three key research questions: (1) the necessity of in-language (Arabic) tool-calling data versus relying on cross-lingual transfer, (2) the effect of general-purpose instruction tuning on tool-calling performance, and (3) the value of fine-tuning on specific, high-priority tools. To address these questions, we conduct extensive experiments using base and post-trained variants of an open-weight Arabic LLM. To enable this study, we bridge the resource gap by translating and adapting two open-source tool-calling datasets into Arabic. Our findings provide crucial insights into the optimal strategies for developing robust tool-augmented agents for Arabic.
>
---
#### [new 067] StyleBench: Evaluating thinking styles in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出StyleBench，系统评估大语言模型在不同任务中的推理风格。研究分析五种推理策略在多种模型和任务上的表现，揭示推理效果受模型规模与任务类型影响，为选择最优策略提供指导。**

- **链接: [http://arxiv.org/pdf/2509.20868v1](http://arxiv.org/pdf/2509.20868v1)**

> **作者:** Junyu Guo; Shangding Gu; Ming Jin; Costas Spanos; Javad Lavaei
>
> **摘要:** The effectiveness of Large Language Models (LLMs) is heavily influenced by the reasoning strategies, or styles of thought, employed in their prompts. However, the interplay between these reasoning styles, model architecture, and task type remains poorly understood. To address this, we introduce StyleBench, a comprehensive benchmark for systematically evaluating reasoning styles across diverse tasks and models. We assess five representative reasoning styles, including Chain of Thought (CoT), Tree of Thought (ToT), Algorithm of Thought (AoT), Sketch of Thought (SoT), and Chain-of-Draft (CoD) on five reasoning tasks, using 15 open-source models from major families (LLaMA, Qwen, Mistral, Gemma, GPT-OSS, Phi, and DeepSeek) ranging from 270M to 120B parameters. Our large-scale analysis reveals that no single style is universally optimal. We demonstrate that strategy efficacy is highly contingent on both model scale and task type: search-based methods (AoT, ToT) excel in open-ended problems but require large-scale models, while concise styles (SoT, CoD) achieve radical efficiency gains on well-defined tasks. Furthermore, we identify key behavioral patterns: smaller models frequently fail to follow output instructions and default to guessing, while reasoning robustness emerges as a function of scale. Our findings offer a crucial roadmap for selecting optimal reasoning strategies based on specific constraints, we open source the benchmark in https://github.com/JamesJunyuGuo/Style_Bench.
>
---
#### [new 068] PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出PMark，一种基于代理函数的语义级水印方法，用于大语言模型。旨在解决现有方法在鲁棒性和文本分布失真方面的不足，通过多约束通道提升水印抗篡改能力，同时保持文本质量。**

- **链接: [http://arxiv.org/pdf/2509.21057v1](http://arxiv.org/pdf/2509.21057v1)**

> **作者:** Jiahao Huo; Shuliang Liu; Bin Wang; Junyan Zhang; Yibo Yan; Aiwei Liu; Xuming Hu; Mingxun Zhou
>
> **摘要:** Semantic-level watermarking (SWM) for large language models (LLMs) enhances watermarking robustness against text modifications and paraphrasing attacks by treating the sentence as the fundamental unit. However, existing methods still lack strong theoretical guarantees of robustness, and reject-sampling-based generation often introduces significant distribution distortions compared with unwatermarked outputs. In this work, we introduce a new theoretical framework on SWM through the concept of proxy functions (PFs) $\unicode{x2013}$ functions that map sentences to scalar values. Building on this framework, we propose PMark, a simple yet powerful SWM method that estimates the PF median for the next sentence dynamically through sampling while enforcing multiple PF constraints (which we call channels) to strengthen watermark evidence. Equipped with solid theoretical guarantees, PMark achieves the desired distortion-free property and improves the robustness against paraphrasing-style attacks. We also provide an empirically optimized version that further removes the requirement for dynamical median estimation for better sampling efficiency. Experimental results show that PMark consistently outperforms existing SWM baselines in both text quality and robustness, offering a more effective paradigm for detecting machine-generated text. Our code will be released at [this URL](https://github.com/PMark-repo/PMark).
>
---
#### [new 069] Disagreements in Reasoning: How a Model's Thinking Process Dictates Persuasion in Multi-Agent Systems
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究多智能体系统中模型的说服动态，探讨模型的认知过程（尤其是显式推理）如何影响其说服能力。通过实验揭示“说服二元性”：LRMs推理更坚定，但公开推理过程可增强说服力，并分析复杂网络中的影响传播机制。**

- **链接: [http://arxiv.org/pdf/2509.21054v1](http://arxiv.org/pdf/2509.21054v1)**

> **作者:** Haodong Zhao; Jidong Li; Zhaomin Wu; Tianjie Ju; Zhuosheng Zhang; Bingsheng He; Gongshen Liu
>
> **备注:** Work in progress
>
> **摘要:** The rapid proliferation of recent Multi-Agent Systems (MAS), where Large Language Models (LLMs) and Large Reasoning Models (LRMs) usually collaborate to solve complex problems, necessitates a deep understanding of the persuasion dynamics that govern their interactions. This paper challenges the prevailing hypothesis that persuasive efficacy is primarily a function of model scale. We propose instead that these dynamics are fundamentally dictated by a model's underlying cognitive process, especially its capacity for explicit reasoning. Through a series of multi-agent persuasion experiments, we uncover a fundamental trade-off we term the Persuasion Duality. Our findings reveal that the reasoning process in LRMs exhibits significantly greater resistance to persuasion, maintaining their initial beliefs more robustly. Conversely, making this reasoning process transparent by sharing the "thinking content" dramatically increases their ability to persuade others. We further consider more complex transmission persuasion situations and reveal complex dynamics of influence propagation and decay within multi-hop persuasion between multiple agent networks. This research provides systematic evidence linking a model's internal processing architecture to its external persuasive behavior, offering a novel explanation for the susceptibility of advanced models and highlighting critical implications for the safety, robustness, and design of future MAS.
>
---
#### [new 070] DELTA-Code: How Does RL Unlock and Transfer New Programming Algorithms in LLMs?
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出DELTA-Code，研究强化学习（RL）是否能帮助大语言模型（LLMs）学习并迁移新的编程算法。通过设计可控的合成编码问题集，评估模型在无预训练能力的问题上的学习与泛化表现，发现RL可使模型突破原有能力边界，实现新策略的获取与部分迁移。**

- **链接: [http://arxiv.org/pdf/2509.21016v1](http://arxiv.org/pdf/2509.21016v1)**

> **作者:** Yiyou Sun; Yuhan Cao; Pohao Huang; Haoyue Bai; Hannaneh Hajishirzi; Nouha Dziri; Dawn Song
>
> **摘要:** It remains an open question whether LLMs can acquire or generalize genuinely new reasoning strategies, beyond the sharpened skills encoded in their parameters during pre-training or post-training. To attempt to answer this debate, we introduce DELTA-Code--Distributional Evaluation of Learnability and Transferrability in Algorithmic Coding, a controlled benchmark of synthetic coding problem families designed to probe two fundamental aspects: learnability -- can LLMs, through reinforcement learning (RL), solve problem families where pretrained models exhibit failure with large enough attempts (pass@K=0)? --and transferrability -- if learnability happens, can such skills transfer systematically to out-of-distribution (OOD) test sets? Unlike prior public coding datasets, DELTA isolates reasoning skills through templated problem generators and introduces fully OOD problem families that demand novel strategies rather than tool invocation or memorized patterns. Our experiments reveal a striking grokking phase transition: after an extended period with near-zero reward, RL-trained models abruptly climb to near-perfect accuracy. To enable learnability on previously unsolvable problem families, we explore key training ingredients such as staged warm-up with dense rewards, experience replay, curriculum training, and verification-in-the-loop. Beyond learnability, we use DELTA to evaluate transferability or generalization along exploratory, compositional, and transformative axes, as well as cross-family transfer. Results show solid gains within families and for recomposed skills, but persistent weaknesses in transformative cases. DELTA thus offers a clean testbed for probing the limits of RL-driven reasoning and for understanding how models can move beyond existing priors to acquire new algorithmic skills.
>
---
#### [new 071] TABLET: A Large-Scale Dataset for Robust Visual Table Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出了TABLET，一个大规模视觉表格理解（VTU）数据集，包含400万例，解决现有数据集缺乏真实视觉多样性和可重构性的问题。提供了图像-HTML配对及元数据，支持模型训练与评估，提升现实场景下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.21205v1](http://arxiv.org/pdf/2509.21205v1)**

> **作者:** Iñigo Alonso; Imanol Miranda; Eneko Agirre; Mirella Lapata
>
> **摘要:** While table understanding increasingly relies on pixel-only settings where tables are processed as visual representations, current benchmarks predominantly use synthetic renderings that lack the complexity and visual diversity of real-world tables. Additionally, existing visual table understanding (VTU) datasets offer fixed examples with single visualizations and pre-defined instructions, providing no access to underlying serialized data for reformulation. We introduce TABLET, a large-scale VTU dataset with 4 million examples across 20 tasks, grounded in 2 million unique tables where 88% preserve original visualizations. Each example includes paired image-HTML representations, comprehensive metadata, and provenance information linking back to the source datasets. Fine-tuning vision-language models like Qwen2.5-VL-7B on TABLET improves performance on seen and unseen VTU tasks while increasing robustness on real-world table visualizations. By preserving original visualizations and maintaining example traceability in a unified large-scale collection, TABLET establishes a foundation for robust training and extensible evaluation of future VTU models.
>
---
#### [new 072] Intercept Cancer: Cancer Pre-Screening with Large Scale Healthcare Foundation Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出CATCH-FM，一种基于医疗记录的癌症预筛查方法。任务是早期识别高风险患者，解决传统筛查成本高、覆盖不足的问题。利用大规模EHR数据训练医疗基础模型，实现高效、非侵入性癌症风险预测，并在多种癌症中表现优异。**

- **链接: [http://arxiv.org/pdf/2506.00209v1](http://arxiv.org/pdf/2506.00209v1)**

> **作者:** Liwen Sun; Hao-Ren Yao; Gary Gao; Ophir Frieder; Chenyan Xiong
>
> **摘要:** Cancer screening, leading to early detection, saves lives. Unfortunately, existing screening techniques require expensive and intrusive medical procedures, not globally available, resulting in too many lost would-be-saved lives. We present CATCH-FM, CATch Cancer early with Healthcare Foundation Models, a cancer pre-screening methodology that identifies high-risk patients for further screening solely based on their historical medical records. With millions of electronic healthcare records (EHR), we establish the scaling law of EHR foundation models pretrained on medical code sequences, pretrain compute-optimal foundation models of up to 2.4 billion parameters, and finetune them on clinician-curated cancer risk prediction cohorts. In our retrospective evaluation comprising of thirty thousand patients, CATCH-FM achieved strong efficacy (60% sensitivity) with low risk (99% specificity and Negative Predictive Value), outperforming feature-based tree models as well as general and medical large language models by large margins. Despite significant demographic, healthcare system, and EHR coding differences, CATCH-FM achieves state-of-the-art pancreatic cancer risk prediction on the EHRSHOT few-shot leaderboard, outperforming EHR foundation models pretrained using on-site patient data. Our analysis demonstrates the robustness of CATCH-FM in various patient distributions, the benefits of operating in the ICD code space, and its ability to capture non-trivial cancer risk factors. Our code will be open-sourced.
>
---
#### [new 073] Leveraging NTPs for Efficient Hallucination Detection in VLMs
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于视觉-语言模型（VLM）的幻觉检测任务，旨在解决现有方法计算成本高、延迟大的问题。研究提出利用VLM的下一个词概率（NTPs）训练轻量级模型进行高效幻觉检测，并结合语言模型提升性能。**

- **链接: [http://arxiv.org/pdf/2509.20379v1](http://arxiv.org/pdf/2509.20379v1)**

> **作者:** Ofir Azachi; Kfir Eliyahu; Eyal El Ani; Rom Himelstein; Roi Reichart; Yuval Pinter; Nitay Calderon
>
> **摘要:** Hallucinations of vision-language models (VLMs), which are misalignments between visual content and generated text, undermine the reliability of VLMs. One common approach for detecting them employs the same VLM, or a different one, to assess generated outputs. This process is computationally intensive and increases model latency. In this paper, we explore an efficient on-the-fly method for hallucination detection by training traditional ML models over signals based on the VLM's next-token probabilities (NTPs). NTPs provide a direct quantification of model uncertainty. We hypothesize that high uncertainty (i.e., a low NTP value) is strongly associated with hallucinations. To test this, we introduce a dataset of 1,400 human-annotated statements derived from VLM-generated content, each labeled as hallucinated or not, and use it to test our NTP-based lightweight method. Our results demonstrate that NTP-based features are valuable predictors of hallucinations, enabling fast and simple ML models to achieve performance comparable to that of strong VLMs. Furthermore, augmenting these NTPs with linguistic NTPs, computed by feeding only the generated text back into the VLM, enhances hallucination detection performance. Finally, integrating hallucination prediction scores from VLMs into the NTP-based models led to better performance than using either VLMs or NTPs alone. We hope this study paves the way for simple, lightweight solutions that enhance the reliability of VLMs.
>
---
#### [new 074] Visual Authority and the Rhetoric of Health Misinformation: A Multimodal Analysis of Social Media Videos
- **分类: cs.SI; cs.CL; cs.CV; cs.MM**

- **简介: 该论文属于健康信息传播研究，分析社交媒体视频中健康误导信息的可信度构建方式。通过多模态分析152个视频，探讨权威信号、叙事技巧与盈利手段的结合，揭示其如何影响观众认知。**

- **链接: [http://arxiv.org/pdf/2509.20724v1](http://arxiv.org/pdf/2509.20724v1)**

> **作者:** Mohammad Reza Zarei; Barbara Stead-Coyle; Michael Christensen; Sarah Everts; Majid Komeili
>
> **摘要:** Short form video platforms are central sites for health advice, where alternative narratives mix useful, misleading, and harmful content. Rather than adjudicating truth, this study examines how credibility is packaged in nutrition and supplement videos by analyzing the intersection of authority signals, narrative techniques, and monetization. We assemble a cross platform corpus of 152 public videos from TikTok, Instagram, and YouTube and annotate each on 26 features spanning visual authority, presenter attributes, narrative strategies, and engagement cues. A transparent annotation pipeline integrates automatic speech recognition, principled frame selection, and a multimodal model, with human verification on a stratified subsample showing strong agreement. Descriptively, a confident single presenter in studio or home settings dominates, and clinical contexts are rare. Analytically, authority cues such as titles, slides and charts, and certificates frequently occur with persuasive elements including jargon, references, fear or urgency, critiques of mainstream medicine, and conspiracies, and with monetization including sales links and calls to subscribe. References and science like visuals often travel with emotive and oppositional narratives rather than signaling restraint.
>
---
#### [new 075] Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation
- **分类: cs.CV; cs.CL**

- **简介: 该论文聚焦文本到图像生成的评估任务，旨在解决现有自动评估指标与人类判断不一致的问题。作者系统分析了多种常用指标在不同组合生成任务中的表现，发现无单一最优指标，强调需根据任务选择合适的评估方法。**

- **链接: [http://arxiv.org/pdf/2509.21227v1](http://arxiv.org/pdf/2509.21227v1)**

> **作者:** Seyed Amir Kasaei; Ali Aghayari; Arash Marioriyad; Niki Sepasian; MohammadAmin Fazli; Mahdieh Soleymani Baghshah; Mohammad Hossein Rohban
>
> **备注:** Accepted at GenProCC NeurIPS 2025 Workshop
>
> **摘要:** Text-image generation has advanced rapidly, but assessing whether outputs truly capture the objects, attributes, and relations described in prompts remains a central challenge. Evaluation in this space relies heavily on automated metrics, yet these are often adopted by convention or popularity rather than validated against human judgment. Because evaluation and reported progress in the field depend directly on these metrics, it is critical to understand how well they reflect human preferences. To address this, we present a broad study of widely used metrics for compositional text-image evaluation. Our analysis goes beyond simple correlation, examining their behavior across diverse compositional challenges and comparing how different metric families align with human judgments. The results show that no single metric performs consistently across tasks: performance varies with the type of compositional problem. Notably, VQA-based metrics, though popular, are not uniformly superior, while certain embedding-based metrics prove stronger in specific cases. Image-only metrics, as expected, contribute little to compositional evaluation, as they are designed for perceptual quality rather than alignment. These findings underscore the importance of careful and transparent metric selection, both for trustworthy evaluation and for their use as reward models in generation. Project page is available at \href{https://amirkasaei.com/eval-the-evals/}{this URL}.
>
---
#### [new 076] Sigma: Semantically Informative Pre-training for Skeleton-based Sign Language Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Sigma，一个基于骨架的统一手语理解（SLU）框架。针对语义关联弱、局部与全局不平衡、跨模态学习效率低的问题，设计了语义感知融合机制、层次对齐策略和统一预训练方法，在多个任务上取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.21223v1](http://arxiv.org/pdf/2509.21223v1)**

> **作者:** Muxin Pu; Mei Kuan Lim; Chun Yong Chong; Chen Change Loy
>
> **摘要:** Pre-training has proven effective for learning transferable features in sign language understanding (SLU) tasks. Recently, skeleton-based methods have gained increasing attention because they can robustly handle variations in subjects and backgrounds without being affected by appearance or environmental factors. Current SLU methods continue to face three key limitations: 1) weak semantic grounding, as models often capture low-level motion patterns from skeletal data but struggle to relate them to linguistic meaning; 2) imbalance between local details and global context, with models either focusing too narrowly on fine-grained cues or overlooking them for broader context; and 3) inefficient cross-modal learning, as constructing semantically aligned representations across modalities remains difficult. To address these, we propose Sigma, a unified skeleton-based SLU framework featuring: 1) a sign-aware early fusion mechanism that facilitates deep interaction between visual and textual modalities, enriching visual features with linguistic context; 2) a hierarchical alignment learning strategy that jointly maximises agreements across different levels of paired features from different modalities, effectively capturing both fine-grained details and high-level semantic relationships; and 3) a unified pre-training framework that combines contrastive learning, text matching and language modelling to promote semantic consistency and generalisation. Sigma achieves new state-of-the-art results on isolated sign language recognition, continuous sign language recognition, and gloss-free sign language translation on multiple benchmarks spanning different sign and spoken languages, demonstrating the impact of semantically informative pre-training and the effectiveness of skeletal data as a stand-alone solution for SLU.
>
---
#### [new 077] Can Federated Learning Safeguard Private Data in LLM Training? Vulnerabilities, Attacks, and Defense Evaluation
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文研究联邦学习（FL）在大语言模型（LLM）训练中的隐私风险。任务是评估FL的隐私保护能力，发现问题并提出防御方法。通过实验发现数据泄露问题，并引入攻击策略与隐私保护技术评估，为降低隐私风险提供指导。**

- **链接: [http://arxiv.org/pdf/2509.20680v1](http://arxiv.org/pdf/2509.20680v1)**

> **作者:** Wenkai Guo; Xuefeng Liu; Haolin Wang; Jianwei Niu; Shaojie Tang; Jing Yuan
>
> **备注:** 28 pages, 32 figures, accepted to the Findings of EMNLP 2025
>
> **摘要:** Fine-tuning large language models (LLMs) with local data is a widely adopted approach for organizations seeking to adapt LLMs to their specific domains. Given the shared characteristics in data across different organizations, the idea of collaboratively fine-tuning an LLM using data from multiple sources presents an appealing opportunity. However, organizations are often reluctant to share local data, making centralized fine-tuning impractical. Federated learning (FL), a privacy-preserving framework, enables clients to retain local data while sharing only model parameters for collaborative training, offering a potential solution. While fine-tuning LLMs on centralized datasets risks data leakage through next-token prediction, the iterative aggregation process in FL results in a global model that encapsulates generalized knowledge, which some believe protects client privacy. In this paper, however, we present contradictory findings through extensive experiments. We show that attackers can still extract training data from the global model, even using straightforward generation methods, with leakage increasing as the model size grows. Moreover, we introduce an enhanced attack strategy tailored to FL, which tracks global model updates during training to intensify privacy leakage. To mitigate these risks, we evaluate privacy-preserving techniques in FL, including differential privacy, regularization-constrained updates and adopting LLMs with safety alignment. Our results provide valuable insights and practical guidelines for reducing privacy risks when training LLMs with FL.
>
---
#### [new 078] Hallucination as an Upper Bound: A New Perspective on Text-to-Image Evaluation
- **分类: cs.CV; cs.CL**

- **简介: 该论文聚焦文本到图像生成任务，旨在解决模型生成内容中幻觉现象定义不清的问题。提出了属性、关系和对象三类幻觉的分类体系，为T2I模型评估提供了新的上界视角和偏见分析基础。**

- **链接: [http://arxiv.org/pdf/2509.21257v1](http://arxiv.org/pdf/2509.21257v1)**

> **作者:** Seyed Amir Kasaei; Mohammad Hossein Rohban
>
> **备注:** Accepted at GenProCC NeurIPS 2025 Workshop
>
> **摘要:** In language and vision-language models, hallucination is broadly understood as content generated from a model's prior knowledge or biases rather than from the given input. While this phenomenon has been studied in those domains, it has not been clearly framed for text-to-image (T2I) generative models. Existing evaluations mainly focus on alignment, checking whether prompt-specified elements appear, but overlook what the model generates beyond the prompt. We argue for defining hallucination in T2I as bias-driven deviations and propose a taxonomy with three categories: attribute, relation, and object hallucinations. This framing introduces an upper bound for evaluation and surfaces hidden biases, providing a foundation for richer assessment of T2I models.
>
---
#### [new 079] TrustJudge: Inconsistencies of LLM-as-a-Judge and How to Alleviate Them
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究LLM作为评估者的不一致性问题，提出TrustJudge框架，通过分布敏感评分和似然感知聚合，减少评分比较与传递性矛盾，提升评估可靠性。**

- **链接: [http://arxiv.org/pdf/2509.21117v1](http://arxiv.org/pdf/2509.21117v1)**

> **作者:** Yidong Wang; Yunze Song; Tingyuan Zhu; Xuanwang Zhang; Zhuohao Yu; Hao Chen; Chiyu Song; Qiufeng Wang; Cunxiang Wang; Zhen Wu; Xinyu Dai; Yue Zhang; Wei Ye; Shikun Zhang
>
> **备注:** 22 pages, 9 figures, 6 tables
>
> **摘要:** The adoption of Large Language Models (LLMs) as automated evaluators (LLM-as-a-judge) has revealed critical inconsistencies in current evaluation frameworks. We identify two fundamental types of inconsistencies: (1) Score-Comparison Inconsistency, where lower-rated responses outperform higher-scored ones in pairwise comparisons, and (2) Pairwise Transitivity Inconsistency, manifested through circular preference chains (A>B>C>A) and equivalence contradictions (A=B=C\neq A). We argue that these issues come from information loss in discrete rating systems and ambiguous tie judgments during pairwise evaluation. We propose TrustJudge, a probabilistic framework that addresses these limitations through two key innovations: 1) distribution-sensitive scoring that computes continuous expectations from discrete rating probabilities, preserving information entropy for more precise scoring, and 2) likelihood-aware aggregation that resolves transitivity violations using bidirectional preference probabilities or perplexity. We also formalize the theoretical limitations of current LLM-as-a-judge frameworks and demonstrate how TrustJudge's components overcome them. When evaluated with Llama-3.1-70B-Instruct as judge using our dataset, TrustJudge reduces Score-Comparison inconsistency by 8.43% (from 23.32% to 14.89%) and Pairwise Transitivity inconsistency by 10.82% (from 15.22% to 4.40%), while maintaining higher evaluation accuracy. Our work provides the first systematic analysis of evaluation framework inconsistencies in LLM-as-a-judge paradigms, offering both theoretical insights and practical solutions for reliable automated assessment. The framework demonstrates consistent improvements across various model architectures and scales, enabling more trustworthy LLM evaluation without requiring additional training or human annotations. The codes can be found at https://github.com/TrustJudge/TrustJudge.
>
---
#### [new 080] Automotive-ENV: Benchmarking Multimodal Agents in Vehicle Interface Systems
- **分类: cs.RO; cs.CL; F.2.2; I.2.7**

- **简介: 该论文提出Automotive-ENV，首个针对车载GUI的高保真基准环境，并设计ASURADA代理，通过地理信息提升驾驶安全任务性能。**

- **链接: [http://arxiv.org/pdf/2509.21143v1](http://arxiv.org/pdf/2509.21143v1)**

> **作者:** Junfeng Yan; Biao Wu; Meng Fang; Ling Chen
>
> **备注:** 10 pages, 5 figures,
>
> **摘要:** Multimodal agents have demonstrated strong performance in general GUI interactions, but their application in automotive systems has been largely unexplored. In-vehicle GUIs present distinct challenges: drivers' limited attention, strict safety requirements, and complex location-based interaction patterns. To address these challenges, we introduce Automotive-ENV, the first high-fidelity benchmark and interaction environment tailored for vehicle GUIs. This platform defines 185 parameterized tasks spanning explicit control, implicit intent understanding, and safety-aware tasks, and provides structured multimodal observations with precise programmatic checks for reproducible evaluation. Building on this benchmark, we propose ASURADA, a geo-aware multimodal agent that integrates GPS-informed context to dynamically adjust actions based on location, environmental conditions, and regional driving norms. Experiments show that geo-aware information significantly improves success on safety-aware tasks, highlighting the importance of location-based context in automotive environments. We will release Automotive-ENV, complete with all tasks and benchmarking tools, to further the development of safe and adaptive in-vehicle agents.
>
---
#### [new 081] On Theoretical Interpretations of Concept-Based In-Context Learning
- **分类: cs.IT; cs.AI; cs.CL; math.IT**

- **简介: 该论文研究基于概念的上下文学习（CB-ICL）机制，旨在解决少样本任务中大语言模型预测性能的理论解释问题。论文提出了理论分析框架，量化知识利用效果，并通过实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2509.20882v1](http://arxiv.org/pdf/2509.20882v1)**

> **作者:** Huaze Tang; Tianren Peng; Shao-lun Huang
>
> **摘要:** In-Context Learning (ICL) has emerged as an important new paradigm in natural language processing and large language model (LLM) applications. However, the theoretical understanding of the ICL mechanism remains limited. This paper aims to investigate this issue by studying a particular ICL approach, called concept-based ICL (CB-ICL). In particular, we propose theoretical analyses on applying CB-ICL to ICL tasks, which explains why and when the CB-ICL performs well for predicting query labels in prompts with only a few demonstrations. In addition, the proposed theory quantifies the knowledge that can be leveraged by the LLMs to the prompt tasks, and leads to a similarity measure between the prompt demonstrations and the query input, which provides important insights and guidance for model pre-training and prompt engineering in ICL. Moreover, the impact of the prompt demonstration size and the dimension of the LLM embeddings in ICL are also explored based on the proposed theory. Finally, several real-data experiments are conducted to validate the practical usefulness of CB-ICL and the corresponding theory.
>
---
#### [new 082] RadAgents: Multimodal Agentic Reasoning for Chest X-ray Interpretation with Radiologist-like Workflows
- **分类: cs.MA; cs.CL; cs.CV**

- **简介: 该论文提出RadAgents，一个用于胸片解读的多智能体框架，旨在解决现有方法在临床可解释性、多模态融合和工具一致性方面的不足。通过结合临床先验知识与多模态推理，提升系统可靠性与透明度。属于医学影像理解任务。**

- **链接: [http://arxiv.org/pdf/2509.20490v1](http://arxiv.org/pdf/2509.20490v1)**

> **作者:** Kai Zhang; Corey D Barrett; Jangwon Kim; Lichao Sun; Tara Taghavi; Krishnaram Kenthapadi
>
> **备注:** In progress
>
> **摘要:** Agentic systems offer a potential path to solve complex clinical tasks through collaboration among specialized agents, augmented by tool use and external knowledge bases. Nevertheless, for chest X-ray (CXR) interpretation, prevailing methods remain limited: (i) reasoning is frequently neither clinically interpretable nor aligned with guidelines, reflecting mere aggregation of tool outputs; (ii) multimodal evidence is insufficiently fused, yielding text-only rationales that are not visually grounded; and (iii) systems rarely detect or resolve cross-tool inconsistencies and provide no principled verification mechanisms. To bridge the above gaps, we present RadAgents, a multi-agent framework for CXR interpretation that couples clinical priors with task-aware multimodal reasoning. In addition, we integrate grounding and multimodal retrieval-augmentation to verify and resolve context conflicts, resulting in outputs that are more reliable, transparent, and consistent with clinical practice.
>
---
#### [new 083] Blueprints of Trust: AI System Cards for End to End Transparency and Governance
- **分类: cs.CY; cs.AI; cs.CL; cs.CR**

- **简介: 该论文提出HASC框架，旨在提升AI系统的透明度与可问责性。通过引入ASH ID等标准化标识，动态记录系统安全状态，辅助全生命周期安全管理，并与ISO/IEC 42001:2023标准对比，促进治理协同。**

- **链接: [http://arxiv.org/pdf/2509.20394v1](http://arxiv.org/pdf/2509.20394v1)**

> **作者:** Huzaifa Sidhpurwala; Emily Fox; Garth Mollett; Florencio Cano Gabarda; Roman Zhukov
>
> **摘要:** This paper introduces the Hazard-Aware System Card (HASC), a novel framework designed to enhance transparency and accountability in the development and deployment of AI systems. The HASC builds upon existing model card and system card concepts by integrating a comprehensive, dynamic record of an AI system's security and safety posture. The framework proposes a standardized system of identifiers, including a novel AI Safety Hazard (ASH) ID, to complement existing security identifiers like CVEs, allowing for clear and consistent communication of fixed flaws. By providing a single, accessible source of truth, the HASC empowers developers and stakeholders to make more informed decisions about AI system safety throughout its lifecycle. Ultimately, we also compare our proposed AI system cards with the ISO/IEC 42001:2023 standard and discuss how they can be used to complement each other, providing greater transparency and accountability for AI systems.
>
---
#### [new 084] InsightGUIDE: An Opinionated AI Assistant for Guided Critical Reading of Scientific Literature
- **分类: cs.AI; cs.CL; cs.DL; cs.HC**

- **简介: 该论文提出InsightGUIDE，一个用于辅助科学文献批判性阅读的AI工具。针对现有LLM工具总结冗长、易替代阅读的问题，设计结构化、可操作的阅读指南，提供更有效的研究支持。**

- **链接: [http://arxiv.org/pdf/2509.20493v1](http://arxiv.org/pdf/2509.20493v1)**

> **作者:** Paris Koloveas; Serafeim Chatzopoulos; Thanasis Vergoulis; Christos Tryfonopoulos
>
> **备注:** Accepted for publication on ICTAI 2025
>
> **摘要:** The proliferation of scientific literature presents an increasingly significant challenge for researchers. While Large Language Models (LLMs) offer promise, existing tools often provide verbose summaries that risk replacing, rather than assisting, the reading of the source material. This paper introduces InsightGUIDE, a novel AI-powered tool designed to function as a reading assistant, not a replacement. Our system provides concise, structured insights that act as a "map" to a paper's key elements by embedding an expert's reading methodology directly into its core AI logic. We present the system's architecture, its prompt-driven methodology, and a qualitative case study comparing its output to a general-purpose LLM. The results demonstrate that InsightGUIDE produces more structured and actionable guidance, serving as a more effective tool for the modern researcher.
>
---
#### [new 085] Interactive Recommendation Agent with Active User Commands
- **分类: cs.IR; cs.CL; cs.HC**

- **简介: 该论文提出交互式推荐系统IRF及RecBot双代理架构，旨在解决传统推荐系统因被动反馈导致的用户意图理解偏差问题。通过自然语言命令实现用户主动控制推荐策略，提升用户满意度与系统效果。**

- **链接: [http://arxiv.org/pdf/2509.21317v1](http://arxiv.org/pdf/2509.21317v1)**

> **作者:** Jiakai Tang; Yujie Luo; Xunke Xi; Fei Sun; Xueyang Feng; Sunhao Dai; Chao Yi; Dian Chen; Zhujin Gao; Yang Li; Xu Chen; Wen Chen; Jian Wu; Yuning Jiang; Bo Zheng
>
> **备注:** Under Review
>
> **摘要:** Traditional recommender systems rely on passive feedback mechanisms that limit users to simple choices such as like and dislike. However, these coarse-grained signals fail to capture users' nuanced behavior motivations and intentions. In turn, current systems cannot also distinguish which specific item attributes drive user satisfaction or dissatisfaction, resulting in inaccurate preference modeling. These fundamental limitations create a persistent gap between user intentions and system interpretations, ultimately undermining user satisfaction and harming system effectiveness. To address these limitations, we introduce the Interactive Recommendation Feed (IRF), a pioneering paradigm that enables natural language commands within mainstream recommendation feeds. Unlike traditional systems that confine users to passive implicit behavioral influence, IRF empowers active explicit control over recommendation policies through real-time linguistic commands. To support this paradigm, we develop RecBot, a dual-agent architecture where a Parser Agent transforms linguistic expressions into structured preferences and a Planner Agent dynamically orchestrates adaptive tool chains for on-the-fly policy adjustment. To enable practical deployment, we employ simulation-augmented knowledge distillation to achieve efficient performance while maintaining strong reasoning capabilities. Through extensive offline and long-term online experiments, RecBot shows significant improvements in both user satisfaction and business outcomes.
>
---
#### [new 086] Every Character Counts: From Vulnerability to Defense in Phishing Detection
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究基于字符级深度学习模型（如CharGRU）在钓鱼邮件检测中的应用，旨在提高检测的鲁棒性和可解释性。通过对比不同模型在标准和对抗场景下的表现，并结合Grad-CAM实现可视化分析，为浏览器扩展工具开发提供支持。**

- **链接: [http://arxiv.org/pdf/2509.20589v1](http://arxiv.org/pdf/2509.20589v1)**

> **作者:** Maria Chiper; Radu Tudor Ionescu
>
> **备注:** Accepted at ICTAI 2025
>
> **摘要:** Phishing attacks targeting both organizations and individuals are becoming an increasingly significant threat as technology advances. Current automatic detection methods often lack explainability and robustness in detecting new phishing attacks. In this work, we investigate the effectiveness of character-level deep learning models for phishing detection, which can provide both robustness and interpretability. We evaluate three neural architectures adapted to operate at the character level, namely CharCNN, CharGRU, and CharBiLSTM, on a custom-built email dataset, which combines data from multiple sources. Their performance is analyzed under three scenarios: (i) standard training and testing, (ii) standard training and testing under adversarial attacks, and (iii) training and testing with adversarial examples. Aiming to develop a tool that operates as a browser extension, we test all models under limited computational resources. In this constrained setup, CharGRU proves to be the best-performing model across all scenarios. All models show vulnerability to adversarial attacks, but adversarial training substantially improves their robustness. In addition, by adapting the Gradient-weighted Class Activation Mapping (Grad-CAM) technique to character-level inputs, we are able to visualize which parts of each email influence the decision of each model. Our open-source code and data is released at https://github.com/chipermaria/every-character-counts.
>
---
#### [new 087] Verification Limits Code LLM Training
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文研究代码生成大模型训练中的“验证天花板”问题，探讨测试设计与策略对模型性能的影响。通过分析测试复杂度、放松验证阈值及保留多样正确解，提出优化验证方法以提升代码生成能力。**

- **链接: [http://arxiv.org/pdf/2509.20837v1](http://arxiv.org/pdf/2509.20837v1)**

> **作者:** Srishti Gureja; Elena Tommasone; Jingyi He; Sara Hooker; Matthias Gallé; Marzieh Fadaee
>
> **摘要:** Large language models for code generation increasingly rely on synthetic data, where both problem solutions and verification tests are generated by models. While this enables scalable data creation, it introduces a previously unexplored bottleneck: the verification ceiling, in which the quality and diversity of training data are fundamentally constrained by the capabilities of synthetic verifiers. In this work, we systematically study how verification design and strategies influence model performance. We investigate (i) what we verify by analyzing the impact of test complexity and quantity: richer test suites improve code generation capabilities (on average +3 pass@1), while quantity alone yields diminishing returns, (ii) how we verify by exploring relaxed pass thresholds: rigid 100% pass criteria can be overly restrictive. By allowing for relaxed thresholds or incorporating LLM-based soft verification, we can recover valuable training data, leading to a 2-4 point improvement in pass@1 performance. However, this benefit is contingent upon the strength and diversity of the test cases used, and (iii) why verification remains necessary through controlled comparisons of formally correct versus incorrect solutions and human evaluation: retaining diverse correct solutions per problem yields consistent generalization gains. Our results show that Verification as currently practiced is too rigid, filtering out valuable diversity. But it cannot be discarded, only recalibrated. By combining calibrated verification with diverse, challenging problem-solution pairs, we outline a path to break the verification ceiling and unlock stronger code generation models.
>
---
#### [new 088] Perspectra: Choosing Your Experts Enhances Critical Thinking in Multi-Agent Research Ideation
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文提出Perspectra，一种支持多智能体研究构思的交互系统，通过论坛式界面增强用户对专家代理协作的控制与批判性思考。任务是提升多智能体系统的思辨能力，解决用户如何有效引导和评估专家协作的问题。**

- **链接: [http://arxiv.org/pdf/2509.20553v1](http://arxiv.org/pdf/2509.20553v1)**

> **作者:** Yiren Liu; Viraj Shah; Sangho Suh; Pao Siangliulue; Tal August; Yun Huang
>
> **摘要:** Recent advances in multi-agent systems (MAS) enable tools for information search and ideation by assigning personas to agents. However, how users can effectively control, steer, and critically evaluate collaboration among multiple domain-expert agents remains underexplored. We present Perspectra, an interactive MAS that visualizes and structures deliberation among LLM agents via a forum-style interface, supporting @-mention to invite targeted agents, threading for parallel exploration, with a real-time mind map for visualizing arguments and rationales. In a within-subjects study with 18 participants, we compared Perspectra to a group-chat baseline as they developed research proposals. Our findings show that Perspectra significantly increased the frequency and depth of critical-thinking behaviors, elicited more interdisciplinary replies, and led to more frequent proposal revisions than the group chat condition. We discuss implications for designing multi-agent tools that scaffold critical thinking by supporting user control over multi-agent adversarial discourse.
>
---
#### [new 089] ScaleDiff: Scaling Difficult Problems for Advanced Mathematical Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出ScaleDiff，旨在高效生成高难度数学问题以提升大模型的推理能力。针对现有方法成本高、问题难度不足的问题，设计了一个低成本、可扩展的生成管道，并训练了DiffGen-8B模型，在多个数学基准上取得优异性能。**

- **链接: [http://arxiv.org/pdf/2509.21070v1](http://arxiv.org/pdf/2509.21070v1)**

> **作者:** Qizhi Pei; Zhuoshi Pan; Honglin Lin; Xin Gao; Yu Li; Zinan Tang; Conghui He; Rui Yan; Lijun Wu
>
> **备注:** 15 pages
>
> **摘要:** Large Reasoning Models (LRMs) have shown impressive capabilities in complex problem-solving, often benefiting from training on difficult mathematical problems that stimulate intricate reasoning. Recent efforts have explored automated synthesis of mathematical problems by prompting proprietary models or large-scale open-source models from seed data or inherent mathematical concepts. However, scaling up these methods remains challenging due to their high computational/API cost, complexity of prompting, and limited difficulty level of the generated problems. To overcome these limitations, we propose ScaleDiff, a simple yet effective pipeline designed to scale the creation of difficult problems. We efficiently identify difficult problems from existing datasets with only a single forward pass using an adaptive thinking model, which can perceive problem difficulty and automatically switch between "Thinking" and "NoThinking" modes. We then train a specialized difficult problem generator (DiffGen-8B) on this filtered difficult data, which can produce new difficult problems in large scale, eliminating the need for complex, per-instance prompting and its associated high API costs. Fine-tuning Qwen2.5-Math-7B-Instruct on the ScaleDiff-Math dataset yields a substantial performance increase of 11.3% compared to the original dataset and achieves a 65.9% average accuracy on AIME'24, AIME'25, HMMT-Feb'25, BRUMO'25, and MATH500, outperforming recent strong LRMs like OpenThinker3. Notably, this performance is achieved using the cost-efficient Qwen3-8B model as a teacher, demonstrating that our pipeline can effectively transfer advanced reasoning capabilities without relying on larger, more expensive teacher models. Furthermore, we observe a clear scaling phenomenon in model performance on difficult benchmarks as the quantity of difficult problems increases. Code: https://github.com/QizhiPei/ScaleDiff.
>
---
#### [new 090] Expanding Reasoning Potential in Foundation Model by Learning Diverse Chains of Thought Patterns
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦于提升基础模型的数学推理能力。针对现有方法对链式思维（CoT）数据利用不足的问题，提出通过提取高价值推理模式并构建核心参考集，结合双粒度算法筛选优质数据进行训练。实验表明，该方法显著提升了模型在AIME等任务上的表现。**

- **链接: [http://arxiv.org/pdf/2509.21124v1](http://arxiv.org/pdf/2509.21124v1)**

> **作者:** Xuemiao Zhang; Can Ren; Chengying Tu; Rongxiang Weng; Shuo Wang; Hongfei Yan; Jingang Wang; Xunliang Cai
>
> **摘要:** Recent progress in large reasoning models for challenging mathematical reasoning has been driven by reinforcement learning (RL). Incorporating long chain-of-thought (CoT) data during mid-training has also been shown to substantially improve reasoning depth. However, current approaches often utilize CoT data indiscriminately, leaving open the critical question of which data types most effectively enhance model reasoning capabilities. In this paper, we define the foundation model's reasoning potential for the first time as the inverse of the number of independent attempts required to correctly answer the question, which is strongly correlated with the final model performance. We then propose utilizing diverse data enriched with high-value reasoning patterns to expand the reasoning potential. Specifically, we abstract atomic reasoning patterns from CoT sequences, characterized by commonality and inductive capabilities, and use them to construct a core reference set enriched with valuable reasoning patterns. Furthermore, we propose a dual-granularity algorithm involving chains of reasoning patterns and token entropy, efficiently selecting high-value CoT data (CoTP) from the data pool that aligns with the core set, thereby training models to master reasoning effectively. Only 10B-token CoTP data enables the 85A6B Mixture-of-Experts (MoE) model to improve by 9.58% on the challenging AIME 2024 and 2025, and to raise the upper bound of downstream RL performance by 7.81%.
>
---
#### [new 091] Seeing Through Words, Speaking Through Pixels: Deep Representational Alignment Between Vision and Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究视觉与语言模型的表征对齐问题，探讨其在哪些网络层、通过何种线索实现语义一致性，并验证对齐效果是否符合人类判断。实验表明，中后期网络层出现最强对齐，且具有语义鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.20751v1](http://arxiv.org/pdf/2509.20751v1)**

> **作者:** Zoe Wanying He; Sean Trott; Meenakshi Khosla
>
> **备注:** Accepted at EMNLP 2025 (camera-ready)
>
> **摘要:** Recent studies show that deep vision-only and language-only models--trained on disjoint modalities--nonetheless project their inputs into a partially aligned representational space. Yet we still lack a clear picture of where in each network this convergence emerges, what visual or linguistic cues support it, whether it captures human preferences in many-to-many image-text scenarios, and how aggregating exemplars of the same concept affects alignment. Here, we systematically investigate these questions. We find that alignment peaks in mid-to-late layers of both model types, reflecting a shift from modality-specific to conceptually shared representations. This alignment is robust to appearance-only changes but collapses when semantics are altered (e.g., object removal or word-order scrambling), highlighting that the shared code is truly semantic. Moving beyond the one-to-one image-caption paradigm, a forced-choice "Pick-a-Pic" task shows that human preferences for image-caption matches are mirrored in the embedding spaces across all vision-language model pairs. This pattern holds bidirectionally when multiple captions correspond to a single image, demonstrating that models capture fine-grained semantic distinctions akin to human judgments. Surprisingly, averaging embeddings across exemplars amplifies alignment rather than blurring detail. Together, our results demonstrate that unimodal networks converge on a shared semantic code that aligns with human judgments and strengthens with exemplar aggregation.
>
---
#### [new 092] CE-GPPO: Controlling Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出CE-GPPO算法，用于强化学习中的策略优化。针对传统PPO方法因剪切机制丢弃低概率token梯度导致熵不稳定的问题，CE-GPPO以有界方式重新引入这些梯度，从而在探索与利用之间取得更好平衡，提升复杂推理任务性能。**

- **链接: [http://arxiv.org/pdf/2509.20712v1](http://arxiv.org/pdf/2509.20712v1)**

> **作者:** Zhenpeng Su; Leiyu Pan; Minxuan Lv; Yuntao Li; Wenping Hu; Fuzheng Zhang; Kun Gai; Guorui Zhou
>
> **摘要:** Reinforcement learning (RL) has become a powerful paradigm for optimizing large language models (LLMs) to handle complex reasoning tasks. A core challenge in this process lies in managing policy entropy, which reflects the balance between exploration and exploitation during training. Existing methods, such as proximal policy optimization (PPO) and its variants, discard valuable gradient signals from low-probability tokens due to the clipping mechanism. We systematically analyze the entropy dynamics and reveal that these clipped tokens play a critical yet overlooked role in regulating entropy evolution. We propose \textbf{C}ontrolling \textbf{E}ntropy via \textbf{G}radient-\textbf{P}reserving \textbf{P}olicy \textbf{O}ptimization (CE-GPPO), a novel algorithm that reintroduces gradients from clipped tokens in native PPO in a gentle and bounded manner. By controlling the magnitude of gradients from tokens outside the clipping interval, CE-GPPO is able to achieve an exploration-exploitation trade-off. We provide theoretical justification and empirical evidence showing that CE-GPPO effectively mitigates entropy instability. Extensive experiments on mathematical reasoning benchmarks show that CE-GPPO consistently outperforms strong baselines across different model scales.
>
---
#### [new 093] Mechanism of Task-oriented Information Removal in In-context Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究了基于上下文学习（ICL）的机制，提出任务导向的信息去除是其关键。通过低秩滤波和注意力头分析，发现信息去除提升了模型输出准确性，揭示了ICL在少量样本下有效的原因。**

- **链接: [http://arxiv.org/pdf/2509.21012v1](http://arxiv.org/pdf/2509.21012v1)**

> **作者:** Hakaze Cho; Haolin Yang; Gouki Minegishi; Naoya Inoue
>
> **备注:** 67 pages, 70 figures, 7 tables
>
> **摘要:** In-context Learning (ICL) is an emerging few-shot learning paradigm based on modern Language Models (LMs), yet its inner mechanism remains unclear. In this paper, we investigate the mechanism through a novel perspective of information removal. Specifically, we demonstrate that in the zero-shot scenario, LMs encode queries into non-selective representations in hidden states containing information for all possible tasks, leading to arbitrary outputs without focusing on the intended task, resulting in near-zero accuracy. Meanwhile, we find that selectively removing specific information from hidden states by a low-rank filter effectively steers LMs toward the intended task. Building on these findings, by measuring the hidden states on carefully designed metrics, we observe that few-shot ICL effectively simulates such task-oriented information removal processes, selectively removing the redundant information from entangled non-selective representations, and improving the output based on the demonstrations, which constitutes a key mechanism underlying ICL. Moreover, we identify essential attention heads inducing the removal operation, termed Denoising Heads, which enables the ablation experiments blocking the information removal operation from the inference, where the ICL accuracy significantly degrades, especially when the correct label is absent from the few-shot demonstrations, confirming both the critical role of the information removal mechanism and denoising heads.
>
---
#### [new 094] CLUE: Conflict-guided Localization for LLM Unlearning Framework
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究LLM遗忘任务，旨在消除不良数据影响而不损害有用信息。针对现有方法无法区分遗忘与保留神经元的问题，提出CLUE框架，利用冲突引导定位关键神经元，并通过CNF形式化实现精准干预，提升遗忘效果与保留能力。**

- **链接: [http://arxiv.org/pdf/2509.20977v1](http://arxiv.org/pdf/2509.20977v1)**

> **作者:** Hang Chen; Jiaying Zhu; Xinyu Yang; Wenya Wang
>
> **备注:** 10 pages
>
> **摘要:** The LLM unlearning aims to eliminate the influence of undesirable data without affecting causally unrelated information. This process typically involves using a forget set to remove target information, alongside a retain set to maintain non-target capabilities. While recent localization-based methods demonstrate promise in identifying important neurons to be unlearned, they fail to disentangle neurons responsible for forgetting undesirable knowledge or retaining essential skills, often treating them as a single entangled group. As a result, these methods apply uniform interventions, risking catastrophic over-forgetting or incomplete erasure of the target knowledge. To address this, we turn to circuit discovery, a mechanistic interpretability technique, and propose the Conflict-guided Localization for LLM Unlearning framEwork (CLUE). This framework identifies the forget and retain circuit composed of important neurons, and then the circuits are transformed into conjunctive normal forms (CNF). The assignment of each neuron in the CNF satisfiability solution reveals whether it should be forgotten or retained. We then provide targeted fine-tuning strategies for different categories of neurons. Extensive experiments demonstrate that, compared to existing localization methods, CLUE achieves superior forget efficacy and retain utility through precise neural localization.
>
---
#### [new 095] CLAUSE: Agentic Neuro-Symbolic Knowledge Graph Reasoning via Dynamic Learnable Context Engineering
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出CLAUSE，一种用于知识图谱多跳问答的神经符号智能框架。针对现有方法在精度、延迟和成本间的不平衡问题，CLAUSE通过动态上下文构建策略，在资源预算下联合优化子图构造、推理路径发现和证据选择，提升效率与准确性。**

- **链接: [http://arxiv.org/pdf/2509.21035v1](http://arxiv.org/pdf/2509.21035v1)**

> **作者:** Yang Zhao; Chengxiao Dai; Wei Zhuo; Yue Xiu; Dusit Niyato
>
> **摘要:** Knowledge graphs provide structured context for multi-hop question answering, but deployed systems must balance answer accuracy with strict latency and cost targets while preserving provenance. Static k-hop expansions and "think-longer" prompting often over-retrieve, inflate context, and yield unpredictable runtime. We introduce CLAUSE, an agentic three-agent neuro-symbolic framework that treats context construction as a sequential decision process over knowledge graphs, deciding what to expand, which paths to follow or backtrack, what evidence to keep, and when to stop. Latency (interaction steps) and prompt cost (selected tokens) are exposed as user-specified budgets or prices, allowing per-query adaptation to trade-offs among accuracy, latency, and cost without retraining. CLAUSE employs the proposed Lagrangian-Constrained Multi-Agent Proximal Policy Optimization (LC-MAPPO) algorithm to coordinate three agents: Subgraph Architect, Path Navigator, and Context Curator, so that subgraph construction, reasoning-path discovery, and evidence selection are jointly optimized under per-query resource budgets on edge edits, interaction steps, and selected tokens. Across HotpotQA, MetaQA, and FactKG, CLAUSE yields higher EM@1 while reducing subgraph growth and end-to-end latency at equal or lower token budgets. On MetaQA-2-hop, relative to the strongest RAG baseline (GraphRAG), CLAUSE achieves +39.3 EM@1 with 18.6% lower latency and 40.9% lower edge growth. The resulting contexts are compact, provenance-preserving, and deliver predictable performance under deployment constraints.
>
---
#### [new 096] Communication Bias in Large Language Models: A Regulatory Perspective
- **分类: cs.CY; cs.AI; cs.CL; cs.DC; cs.HC; cs.LG**

- **简介: 该论文属于AI监管研究任务，旨在探讨大语言模型的通信偏见及其社会影响。文章分析了欧盟AI法案等框架，提出需加强竞争与设计治理以实现公平可信的AI。**

- **链接: [http://arxiv.org/pdf/2509.21075v1](http://arxiv.org/pdf/2509.21075v1)**

> **作者:** Adrian Kuenzler; Stefan Schmid
>
> **摘要:** Large language models (LLMs) are increasingly central to many applications, raising concerns about bias, fairness, and regulatory compliance. This paper reviews risks of biased outputs and their societal impact, focusing on frameworks like the EU's AI Act and the Digital Services Act. We argue that beyond constant regulation, stronger attention to competition and design governance is needed to ensure fair, trustworthy AI. This is a preprint of the Communications of the ACM article of the same title.
>
---
#### [new 097] Binary Autoencoder for Mechanistic Interpretability of Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Binary Autoencoder (BAE)，用于提升大语言模型的机制可解释性。针对现有自编码器缺乏实例间稀疏性的问题，BAE通过最小化批次激活熵，实现特征解耦与二值化，有效提取原子化、可解释的特征，并应用于推理动态分析和上下文学习研究。**

- **链接: [http://arxiv.org/pdf/2509.20997v1](http://arxiv.org/pdf/2509.20997v1)**

> **作者:** Hakaze Cho; Haolin Yang; Brian M. Kurkoski; Naoya Inoue
>
> **备注:** 36 pages, 41 figures, 3 tables
>
> **摘要:** Existing works are dedicated to untangling atomized numerical components (features) from the hidden states of Large Language Models (LLMs) for interpreting their mechanism. However, they typically rely on autoencoders constrained by some implicit training-time regularization on single training instances (i.e., $L_1$ normalization, top-k function, etc.), without an explicit guarantee of global sparsity among instances, causing a large amount of dense (simultaneously inactive) features, harming the feature sparsity and atomization. In this paper, we propose a novel autoencoder variant that enforces minimal entropy on minibatches of hidden activations, thereby promoting feature independence and sparsity across instances. For efficient entropy calculation, we discretize the hidden activations to 1-bit via a step function and apply gradient estimation to enable backpropagation, so that we term it as Binary Autoencoder (BAE) and empirically demonstrate two major applications: (1) Feature set entropy calculation. Entropy can be reliably estimated on binary hidden activations, which we empirically evaluate and leverage to characterize the inference dynamics of LLMs and In-context Learning. (2) Feature untangling. Similar to typical methods, BAE can extract atomized features from LLM's hidden states. To robustly evaluate such feature extraction capability, we refine traditional feature-interpretation methods to avoid unreliable handling of numerical tokens, and show that BAE avoids dense features while producing the largest number of interpretable ones among baselines, which confirms the effectiveness of BAE serving as a feature extractor.
>
---
#### [new 098] CARINOX: Inference-time Scaling with Category-Aware Reward-based Initial Noise Optimization and Exploration
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对文本到图像生成中的组合对齐问题，提出CARINOX框架，结合噪声优化与探索策略，并通过类别感知的奖励函数提升生成效果。实验表明，该方法在多个基准上显著优于现有技术。**

- **链接: [http://arxiv.org/pdf/2509.17458v1](http://arxiv.org/pdf/2509.17458v1)**

> **作者:** Seyed Amir Kasaei; Ali Aghayari; Arash Marioriyad; Niki Sepasian; Shayan Baghayi Nejad; MohammadAmin Fazli; Mahdieh Soleymani Baghshah; Mohammad Hossein Rohban
>
> **摘要:** Text-to-image diffusion models, such as Stable Diffusion, can produce high-quality and diverse images but often fail to achieve compositional alignment, particularly when prompts describe complex object relationships, attributes, or spatial arrangements. Recent inference-time approaches address this by optimizing or exploring the initial noise under the guidance of reward functions that score text-image alignment without requiring model fine-tuning. While promising, each strategy has intrinsic limitations when used alone: optimization can stall due to poor initialization or unfavorable search trajectories, whereas exploration may require a prohibitively large number of samples to locate a satisfactory output. Our analysis further shows that neither single reward metrics nor ad-hoc combinations reliably capture all aspects of compositionality, leading to weak or inconsistent guidance. To overcome these challenges, we present Category-Aware Reward-based Initial Noise Optimization and Exploration (CARINOX), a unified framework that combines noise optimization and exploration with a principled reward selection procedure grounded in correlation with human judgments. Evaluations on two complementary benchmarks covering diverse compositional challenges show that CARINOX raises average alignment scores by +16% on T2I-CompBench++ and +11% on the HRS benchmark, consistently outperforming state-of-the-art optimization and exploration-based methods across all major categories, while preserving image quality and diversity. The project page is available at https://amirkasaei.com/carinox/{this URL}.
>
---
#### [new 099] Human Semantic Representations of Social Interactions from Moving Shapes
- **分类: cs.CV; cs.CE; cs.CL**

- **简介: 该论文研究人类如何从动态图形中感知社会互动的语义表征。任务是探索视觉特征之外的语义信息在社会知觉中的作用。通过实验发现，基于动词的语义嵌入能最好地解释人类判断，揭示了视觉与抽象语义之间的桥梁。**

- **链接: [http://arxiv.org/pdf/2509.20673v1](http://arxiv.org/pdf/2509.20673v1)**

> **作者:** Yiling Yun; Hongjing Lu
>
> **摘要:** Humans are social creatures who readily recognize various social interactions from simple display of moving shapes. While previous research has often focused on visual features, we examine what semantic representations that humans employ to complement visual features. In Study 1, we directly asked human participants to label the animations based on their impression of moving shapes. We found that human responses were distributed. In Study 2, we measured the representational geometry of 27 social interactions through human similarity judgments and compared it with model predictions based on visual features, labels, and semantic embeddings from animation descriptions. We found that semantic models provided complementary information to visual features in explaining human judgments. Among the semantic models, verb-based embeddings extracted from descriptions account for human similarity judgments the best. These results suggest that social perception in simple displays reflects the semantic structure of social interactions, bridging visual and abstract representations.
>
---
## 更新

#### [replaced 001] What Makes a Reward Model a Good Teacher? An Optimization Perspective
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2503.15477v2](http://arxiv.org/pdf/2503.15477v2)**

> **作者:** Noam Razin; Zixuan Wang; Hubert Strauss; Stanley Wei; Jason D. Lee; Sanjeev Arora
>
> **备注:** Accepted to NeurIPS 2025; Code available at https://github.com/princeton-pli/what-makes-good-rm
>
> **摘要:** The success of Reinforcement Learning from Human Feedback (RLHF) critically depends on the quality of the reward model. However, while this quality is primarily evaluated through accuracy, it remains unclear whether accuracy fully captures what makes a reward model an effective teacher. We address this question from an optimization perspective. First, we prove that regardless of how accurate a reward model is, if it induces low reward variance, then the RLHF objective suffers from a flat landscape. Consequently, even a perfectly accurate reward model can lead to extremely slow optimization, underperforming less accurate models that induce higher reward variance. We additionally show that a reward model that works well for one language model can induce low reward variance, and thus a flat objective landscape, for another. These results establish a fundamental limitation of evaluating reward models solely based on accuracy or independently of the language model they guide. Experiments using models of up to 8B parameters corroborate our theory, demonstrating the interplay between reward variance, accuracy, and reward maximization rate. Overall, our findings highlight that beyond accuracy, a reward model needs to induce sufficient variance for efficient~optimization.
>
---
#### [replaced 002] Can social media provide early warning of retraction? Evidence from critical tweets identified by human annotation and large language models
- **分类: cs.DL; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.16851v3](http://arxiv.org/pdf/2403.16851v3)**

> **作者:** Er-Te Zheng; Hui-Zhen Fu; Mike Thelwall; Zhichao Fang
>
> **备注:** 27 pages, 5 figures
>
> **摘要:** Timely detection of problematic research is essential for safeguarding scientific integrity. To explore whether social media commentary can serve as an early indicator of potentially problematic articles, this study analysed 3,815 tweets referencing 604 retracted articles and 3,373 tweets referencing 668 comparable non-retracted articles. Tweets critical of the articles were identified through both human annotation and large language models (LLMs). Human annotation revealed that 8.3% of retracted articles were associated with at least one critical tweet prior to retraction, compared to only 1.5% of non-retracted articles, highlighting the potential of tweets as early warning signals of retraction. However, critical tweets identified by LLMs (GPT-4o mini, Gemini 2.0 Flash-Lite, and Claude 3.5 Haiku) only partially aligned with human annotation, suggesting that fully automated monitoring of post-publication discourse should be applied with caution. A human-AI collaborative approach may offer a more reliable and scalable alternative, with human expertise helping to filter out tweets critical of issues unrelated to the research integrity of the articles. Overall, this study provides insights into how social media signals, combined with generative AI technologies, may support efforts to strengthen research integrity.
>
---
#### [replaced 003] Improving LLM Unlearning Robustness via Random Perturbations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.19202v4](http://arxiv.org/pdf/2501.19202v4)**

> **作者:** Dang Huu-Tien; Hoang Thanh-Tung; Anh Bui; Minh-Phuong Nguyen; Le-Minh Nguyen; Naoya Inoue
>
> **备注:** 29 pages, 13 figures, 8 tables
>
> **摘要:** Here, we show that current state-of-the-art LLM unlearning methods inherently reduce models' robustness, causing them to misbehave even when a single non-adversarial forget-token is present in the retain-query. Toward understanding underlying causes, we propose a novel theoretical framework that reframes the unlearning process as backdoor attacks and defenses: forget-tokens act as backdoor triggers that, when activated in retain-queries, cause disruptions in unlearned models' behaviors, similar to successful backdoor attacks. The sense that, LLM unlearning methods themselves poison the model, make it more vulnerable to forget-tokens, and hide rather than erase target knowledge, describes their true mechanism. To mitigate the vulnerability caused by the forgetting process, we reinterpret the retaining process as a backdoor defense and propose Random Noise Augmentation (RNA), a lightweight, model and method-agnostic approach with theoretical guarantees for improving the robustness of models. Extensive experiments demonstrate that RNA significantly improves the robustness of unlearned models while preserving forget and retain performances. This backdoor attack-defense framework offers insights into the mechanism of unlearning that can shed light on future research directions for improving unlearning robustness.
>
---
#### [replaced 004] A Simple "Motivation" Can Enhance Reinforcement Finetuning of Large Reasoning Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.18485v2](http://arxiv.org/pdf/2506.18485v2)**

> **作者:** Junjie Zhang; Guozheng Ma; Shunyu Liu; Haoyu Wang; Jiaxing Huang; Ting-En Lin; Fei Huang; Yongbin Li; Dacheng Tao
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful learn-to-reason paradigm for Large Reasoning Models to tackle complex tasks. However, current RLVR paradigm is still not efficient enough, as it works in a trial-and-error manner. To perform better, the model needs to explore the reward space by numerously generating responses and learn from fragmented reward signals, blind to the overall reward patterns. Fortunately, verifiable rewards make the natural language description of the reward function possible, and meanwhile, LLMs have demonstrated strong in-context learning ability. This motivates us to explore if Large Reasoning Models can benefit from a motivation of the task, i.e., awareness of the reward function, during the reinforcement finetuning process, as we humans sometimes do when learning. In this paper, we introduce Motivation-enhanced Reinforcement Finetuning (MeRF), an intuitive yet effective method enhancing reinforcement finetuning of LLMs by involving ``telling LLMs rules of the game''. Specifically, MeRF directly injects the reward specification into the prompt, which serves as an in-context motivation for the model to be aware of the optimization objective. This simple modification leverages the in-context learning ability of LLMs, aligning generation with optimization, thereby incentivizing the model to generate desired outputs from both inner motivation and external reward. Empirical evaluations demonstrate that MeRF achieves substantial performance gains over RLVR baseline. Moreover, ablation studies show that MeRF performs better with greater consistency between the in-context motivation and the external reward function, while the model also demonstrates an ability to adapt to misleading motivations through reinforcement finetuning.
>
---
#### [replaced 005] NoHumansRequired: Autonomous High-Quality Image Editing Triplet Mining
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.14119v2](http://arxiv.org/pdf/2507.14119v2)**

> **作者:** Maksim Kuprashevich; Grigorii Alekseenko; Irina Tolstykh; Georgii Fedorov; Bulat Suleimanov; Vladimir Dokholyan; Aleksandr Gordeev
>
> **摘要:** Recent advances in generative modeling enable image editing assistants that follow natural language instructions without additional user input. Their supervised training requires millions of triplets (original image, instruction, edited image), yet mining pixel-accurate examples is hard. Each edit must affect only prompt-specified regions, preserve stylistic coherence, respect physical plausibility, and retain visual appeal. The lack of robust automated edit-quality metrics hinders reliable automation at scale. We present an automated, modular pipeline that mines high-fidelity triplets across domains, resolutions, instruction complexities, and styles. Built on public generative models and running without human intervention, our system uses a task-tuned Gemini validator to score instruction adherence and aesthetics directly, removing any need for segmentation or grounding models. Inversion and compositional bootstrapping enlarge the mined set by approx. 2.6x, enabling large-scale high-fidelity training data. By automating the most repetitive annotation steps, the approach allows a new scale of training without human labeling effort. To democratize research in this resource-intensive area, we release NHR-Edit, an open dataset of 720k high-quality triplets, curated at industrial scale via millions of guided generations and validator passes, and we analyze the pipeline's stage-wise survival rates, providing a framework for estimating computational effort across different model stacks. In the largest cross-dataset evaluation, it surpasses all public alternatives. We also release Bagel-NHR-Edit, a fine-tuned Bagel model with state-of-the-art metrics.
>
---
#### [replaced 006] ASCIIEval: Benchmarking Models' Visual Perception in Text Strings via ASCII Art
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.01733v2](http://arxiv.org/pdf/2410.01733v2)**

> **作者:** Qi Jia; Xiang Yue; Shanshan Huang; Ziheng Qin; Yizhu Liu; Bill Yuchen Lin; Yang You; Guangtao Zhai
>
> **摘要:** Perceiving visual semantics embedded within consecutive characters is a crucial yet under-explored capability for both Large Language Models (LLMs) and Multi-modal Large Language Models (MLLMs). In this work, we select ASCII art as a representative artifact. It depicts concepts through careful arrangement of characters, which can be formulated in both text and image modalities. We frame the problem as a recognition task, and construct a novel benchmark, ASCIIEval. It covers over 3K samples with an elaborate categorization tree, along with a training set for further enhancement. Encompassing a comprehensive analysis of tens of models through different input modalities, our benchmark demonstrate its multi-faceted diagnostic power. Given textual input, language models shows their visual perception ability on ASCII art concepts. Proprietary models achieve over 70% accuracy on certain categories, with GPT-5 topping the rank. For image inputs, we reveal that open-source MLLMs suffer from a trade-off between fine-grained text recognition and collective visual perception. They exhibit limited generalization ability to this special kind of arts, leading to the dramatic gap of over 20.01% accuracy compared with their proprietary counterparts. Another critical finding is that model performance is sensitive to the length of the ASCII art, with this sensitivity varying across input modalities. Unfortunately, none of the models could successfully benefit from the simultaneous provision of both modalities, highlighting the need for more flexible modality-fusion approaches. Besides, we also introduce approaches for further enhancement and discuss future directions. Resources are available at https://github.com/JiaQiSJTU/VisionInText.
>
---
#### [replaced 007] UNCERTAINTY-LINE: Length-Invariant Estimation of Uncertainty for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19060v2](http://arxiv.org/pdf/2505.19060v2)**

> **作者:** Roman Vashurin; Maiya Goloburda; Preslav Nakov; Maxim Panov
>
> **摘要:** Large Language Models (LLMs) have become indispensable tools across various applications, making it more important than ever to ensure the quality and the trustworthiness of their outputs. This has led to growing interest in uncertainty quantification (UQ) methods for assessing the reliability of LLM outputs. Many existing UQ techniques rely on token probabilities, which inadvertently introduces a bias with respect to the length of the output. While some methods attempt to account for this, we demonstrate that such biases persist even in length-normalized approaches. To address the problem, here we propose UNCERTAINTY-LINE: (Length-INvariant Estimation), a simple debiasing procedure that regresses uncertainty scores on output length and uses the residuals as corrected, length-invariant estimates. Our method is post-hoc, model-agnostic, and applicable to a range of UQ measures. Through extensive evaluation on machine translation, summarization, and question-answering tasks, we demonstrate that UNCERTAINTY-LINE: consistently improves over even nominally length-normalized UQ methods uncertainty estimates across multiple metrics and models.
>
---
#### [replaced 008] Just-in-time and distributed task representations in language models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.04466v2](http://arxiv.org/pdf/2509.04466v2)**

> **作者:** Yuxuan Li; Declan Campbell; Stephanie C. Y. Chan; Andrew Kyle Lampinen
>
> **摘要:** Many of language models' impressive capabilities originate from their in-context learning: based on instructions or examples, they can infer and perform new tasks without weight updates. In this work, we investigate when representations for new tasks are formed in language models, and how these representations change over the course of context. We focus on ''transferrable'' task representations -- vector representations that can restore task contexts in another instance of the model, even without the full prompt. We show that these representations evolve in non-monotonic and sporadic ways, and are distinct from a more inert representation of high-level task categories that persists throughout the context. Specifically, when more examples are provided in the context, transferrable task representations successfully condense evidence. This allows better transfer of task contexts and aligns well with the performance improvement. However, this evidence accrual process exhibits strong locality along the sequence dimension, coming online only at certain tokens -- despite task identity being reliably decodable throughout the context. Moreover, these local but transferrable task representations tend to capture minimal ''task scopes'', such as a semantically-independent subtask. For longer and composite tasks, models rely on more temporally-distributed representations. This two-fold locality (temporal and semantic) underscores a kind of just-in-time computational process that language models use to perform new tasks on the fly.
>
---
#### [replaced 009] Constructions are Revealed in Word Distributions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.06048v2](http://arxiv.org/pdf/2503.06048v2)**

> **作者:** Joshua Rozner; Leonie Weissweiler; Kyle Mahowald; Cory Shain
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Construction grammar posits that constructions, or form-meaning pairings, are acquired through experience with language (the distributional learning hypothesis). But how much information about constructions does this distribution actually contain? Corpus-based analyses provide some answers, but text alone cannot answer counterfactual questions about what \emph{caused} a particular word to occur. This requires computable models of the distribution over strings -- namely, pretrained language models (PLMs). Here, we treat a RoBERTa model as a proxy for this distribution and hypothesize that constructions will be revealed within it as patterns of statistical affinity. We support this hypothesis experimentally: many constructions are robustly distinguished, including (i) hard cases where semantically distinct constructions are superficially similar, as well as (ii) \emph{schematic} constructions, whose ``slots'' can be filled by abstract word classes. Despite this success, we also provide qualitative evidence that statistical affinity alone may be insufficient to identify all constructions from text. Thus, statistical affinity is likely an important, but partial, signal available to learners.
>
---
#### [replaced 010] WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.06501v2](http://arxiv.org/pdf/2509.06501v2)**

> **作者:** Junteng Liu; Yunji Li; Chi Zhang; Jingyang Li; Aili Chen; Ke Ji; Weiyu Cheng; Zijia Wu; Chengyu Du; Qidi Xu; Jiayuan Song; Zhengmao Zhu; Wenhu Chen; Pengyu Zhao; Junxian He
>
> **摘要:** The paradigm of Large Language Models (LLMs) has increasingly shifted toward agentic applications, where web browsing capabilities are fundamental for retrieving information from diverse online sources. However, existing open-source web agents either demonstrate limited information-seeking abilities on complex tasks or lack transparent implementations. In this work, we identify that the key challenge lies in the scarcity of challenging data for information seeking. To address this limitation, we introduce WebExplorer: a systematic data generation approach using model-based exploration and iterative, long-to-short query evolution. This method creates challenging query-answer pairs that require multi-step reasoning and complex web navigation. By leveraging our curated high-quality dataset, we successfully develop advanced web agent WebExplorer-8B through supervised fine-tuning followed by reinforcement learning. Our model supports 128K context length and up to 100 tool calling turns, enabling long-horizon problem solving. Across diverse information-seeking benchmarks, WebExplorer-8B achieves the state-of-the-art performance at its scale. Notably, as an 8B-sized model, WebExplorer-8B is able to effectively search over an average of 16 turns after RL training, achieving higher accuracy than WebSailor-72B on BrowseComp-en/zh and attaining the best performance among models up to 100B parameters on WebWalkerQA and FRAMES. Beyond these information-seeking tasks, our model also achieves strong generalization on the HLE benchmark even though it is only trained on knowledge-intensive QA data. These results highlight our approach as a practical path toward long-horizon web agents.
>
---
#### [replaced 011] LAMA-UT: Language Agnostic Multilingual ASR through Orthography Unification and Language-Specific Transliteration
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.15299v3](http://arxiv.org/pdf/2412.15299v3)**

> **作者:** Sangmin Lee; Woo-Jin Chung; Hong-Goo Kang
>
> **备注:** Accepted to AAAI 2025 (Oral Presentation)
>
> **摘要:** Building a universal multilingual automatic speech recognition (ASR) model that performs equitably across languages has long been a challenge due to its inherent difficulties. To address this task we introduce a Language-Agnostic Multilingual ASR pipeline through orthography Unification and language-specific Transliteration (LAMA-UT). LAMA-UT operates without any language-specific modules while matching the performance of state-of-the-art models trained on a minimal amount of data. Our pipeline consists of two key steps. First, we utilize a universal transcription generator to unify orthographic features into Romanized form and capture common phonetic characteristics across diverse languages. Second, we utilize a universal converter to transform these universal transcriptions into language-specific ones. In experiments, we demonstrate the effectiveness of our proposed method leveraging universal transcriptions for massively multilingual ASR. Our pipeline achieves a relative error reduction rate of 45% when compared to Whisper and performs comparably to MMS, despite being trained on only 0.1% of Whisper's training data. Furthermore, our pipeline does not rely on any language-specific modules. However, it performs on par with zero-shot ASR approaches which utilize additional language-specific lexicons and language models. We expect this framework to serve as a cornerstone for flexible multilingual ASR systems that are generalizable even to unseen languages.
>
---
#### [replaced 012] The Validation Gap: A Mechanistic Analysis of How Language Models Compute Arithmetic but Fail to Validate It
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11771v2](http://arxiv.org/pdf/2502.11771v2)**

> **作者:** Leonardo Bertolazzi; Philipp Mondorf; Barbara Plank; Raffaella Bernardi
>
> **备注:** EMNLP 2025 Main, 38 pages, 33 figures
>
> **摘要:** The ability of large language models (LLMs) to validate their output and identify potential errors is crucial for ensuring robustness and reliability. However, current research indicates that LLMs struggle with self-correction, encountering significant challenges in detecting errors. While studies have explored methods to enhance self-correction in LLMs, relatively little attention has been given to understanding the models' internal mechanisms underlying error detection. In this paper, we present a mechanistic analysis of error detection in LLMs, focusing on simple arithmetic problems. Through circuit analysis, we identify the computational subgraphs responsible for detecting arithmetic errors across four smaller-sized LLMs. Our findings reveal that all models heavily rely on $\textit{consistency heads}$--attention heads that assess surface-level alignment of numerical values in arithmetic solutions. Moreover, we observe that the models' internal arithmetic computation primarily occurs in higher layers, whereas validation takes place in middle layers, before the final arithmetic results are fully encoded. This structural dissociation between arithmetic computation and validation seems to explain why smaller-sized LLMs struggle to detect even simple arithmetic errors.
>
---
#### [replaced 013] SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11274v3](http://arxiv.org/pdf/2505.11274v3)**

> **作者:** Zheng Li; Qingxiu Dong; Jingyuan Ma; Di Zhang; Kai Jia; Zhifang Sui
>
> **摘要:** While reasoning models demonstrate exceptional performance on complex tasks, they often exhibit tendencies of overthinking on simple problems. This phenomenon not only leads to excessive computational resource consumption but also significantly degrades user experience. To address this challenge, we propose SelfBudgeter - a novel user-friendly adaptive controllable reasoning framework that incorporates a budget estimation mechanism prior to reasoning. The framework adopts a dual-phase training paradigm: during the cold-start phase, the model learns to predict token budgets before executing reasoning in a standardized format; in the reinforcement learning phase, the model is trained to autonomously plan budgets based on problem difficulty and strictly adhere to them when generating responses. Since the model outputs budget estimates at the initial stage, users can immediately anticipate waiting duration, enabling flexible decisions on whether to interrupt or continue the generation process. Notably, our method supports manual control of reasoning length through pre-filled budget fields. Experimental results demonstrate that SelfBudgeter can dynamically allocate budgets according to problem complexity, yielding an average response length compression of 61% for the 1.5B model on GSM8K, MATH500, and AIME2025, and 48% for the 7B model, while maintaining nearly undiminished accuracy.
>
---
#### [replaced 014] UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.07019v4](http://arxiv.org/pdf/2411.07019v4)**

> **作者:** Zhiqiang Liu; Yin Hua; Mingyang Chen; Zhuo Chen; Lei Liang; Huajun Chen; Wen Zhang
>
> **摘要:** Real-world knowledge graphs (KGs) contain not only standard triple-based facts, but also more complex, heterogeneous types of facts, such as hyper-relational facts with auxiliary key-value pairs, temporal facts with additional timestamps, and nested facts that imply relationships between facts. These richer forms of representation have attracted significant attention due to their enhanced expressiveness and capacity to model complex semantics in real-world scenarios. However, most existing studies suffer from two main limitations: (1) they typically focus on modeling only specific types of facts, thus making it difficult to generalize to real-world scenarios with multiple fact types; and (2) they struggle to achieve generalizable hierarchical (inter-fact and intra-fact) modeling due to the complexity of these representations. To overcome these limitations, we propose UniHR, a Unified Hierarchical Representation learning framework, which consists of a learning-optimized Hierarchical Data Representation (HiDR) module and a unified Hierarchical Structure Learning (HiSL) module. The HiDR module unifies hyper-relational KGs, temporal KGs, and nested factual KGs into triple-based representations. Then HiSL incorporates intra-fact and inter-fact message passing, focusing on enhancing both semantic information within individual facts and enriching the structural information between facts. To go beyond the unified method itself, we further explore the potential of unified representation in complex real-world scenarios, including joint modeling of multi-task, compositional and hybrid facts. Extensive experiments on 9 datasets across 5 types of KGs demonstrate the effectiveness of UniHR and highlight the strong potential of unified representations.
>
---
#### [replaced 015] Causal Reflection with Language Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.04495v2](http://arxiv.org/pdf/2508.04495v2)**

> **作者:** Abi Aryan; Zac Liu
>
> **摘要:** While LLMs exhibit impressive fluency and factual recall, they struggle with robust causal reasoning, often relying on spurious correlations and brittle patterns. Similarly, traditional Reinforcement Learning agents also lack causal understanding, optimizing for rewards without modeling why actions lead to outcomes. We introduce Causal Reflection, a framework that explicitly models causality as a dynamic function over state, action, time, and perturbation, enabling agents to reason about delayed and nonlinear effects. Additionally, we define a formal Reflect mechanism that identifies mismatches between predicted and observed outcomes and generates causal hypotheses to revise the agent's internal model. In this architecture, LLMs serve not as black-box reasoners, but as structured inference engines translating formal causal outputs into natural language explanations and counterfactuals. Our framework lays the theoretical groundwork for Causal Reflective agents that can adapt, self-correct, and communicate causal understanding in evolving environments.
>
---
#### [replaced 016] InComeS: Integrating Compression and Selection Mechanisms into LLMs for Efficient Model Editing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22156v2](http://arxiv.org/pdf/2505.22156v2)**

> **作者:** Shuaiyi Li; Zhisong Zhang; Yang Deng; Chenlong Deng; Tianqing Fang; Hongming Zhang; Haitao Mi; Dong Yu; Wai Lam
>
> **备注:** 18 pages,5 figures
>
> **摘要:** Although existing model editing methods perform well in recalling exact edit facts, they often struggle in complex scenarios that require deeper semantic understanding rather than mere knowledge regurgitation. Leveraging the strong contextual reasoning abilities of large language models (LLMs), in-context learning (ICL) becomes a promising editing method by comprehending edit information through context encoding. However, this method is constrained by the limited context window of LLMs, leading to degraded performance and efficiency as the number of edits increases. To overcome this limitation, we propose InComeS, a flexible framework that enhances LLMs' ability to process editing contexts through explicit compression and selection mechanisms. Specifically, InComeS compresses each editing context into the key-value (KV) cache of a special gist token, enabling efficient handling of multiple edits without being restricted by the model's context window. Furthermore, specialized cross-attention modules are added to dynamically select the most relevant information from the gist pools, enabling adaptive and effective utilization of edit information. We conduct experiments on diverse model editing benchmarks with various editing formats, and the results demonstrate the effectiveness and efficiency of our method.
>
---
#### [replaced 017] MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23764v2](http://arxiv.org/pdf/2505.23764v2)**

> **作者:** Sihan Yang; Runsen Xu; Yiman Xie; Sizhe Yang; Mo Li; Jingli Lin; Chenming Zhu; Xiaochen Chen; Haodong Duan; Xiangyu Yue; Dahua Lin; Tai Wang; Jiangmiao Pang
>
> **备注:** 34 pages. A comprehensive, fully human-curated, multi-image-based spatial intelligence benchmark with reasoning annotation for MLLMs. Project page: https://runsenxu.com/projects/MMSI_Bench
>
> **摘要:** Spatial intelligence is essential for multimodal large language models (MLLMs) operating in the complex physical world. Existing benchmarks, however, probe only single-image relations and thus fail to assess the multi-image spatial reasoning that real-world deployments demand. We introduce MMSI-Bench, a VQA benchmark dedicated to multi-image spatial intelligence. Six 3D-vision researchers spent more than 300 hours meticulously crafting 1,000 challenging, unambiguous multiple-choice questions from over 120,000 images, each paired with carefully designed distractors and a step-by-step reasoning process. We conduct extensive experiments and thoroughly evaluate 34 open-source and proprietary MLLMs, observing a wide gap: the strongest open-source model attains roughly 30% accuracy and OpenAI's o3 reasoning model reaches 40%, while humans score 97%. These results underscore the challenging nature of MMSI-Bench and the substantial headroom for future research. Leveraging the annotated reasoning processes, we also provide an automated error analysis pipeline that diagnoses four dominant failure modes, including (1) grounding errors, (2) overlap-matching and scene-reconstruction errors, (3) situation-transformation reasoning errors, and (4) spatial-logic errors, offering valuable insights for advancing multi-image spatial intelligence. Project page: https://runsenxu.com/projects/MMSI_Bench .
>
---
#### [replaced 018] MathBuddy: A Multimodal System for Affective Math Tutoring
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2508.19993v2](http://arxiv.org/pdf/2508.19993v2)**

> **作者:** Debanjana Kar; Leopold Böss; Dacia Braca; Sebastian Maximilian Dennerlein; Nina Christine Hubig; Philipp Wintersberger; Yufang Hou
>
> **备注:** Accepted at EMNLP 2025 (Demo Track)
>
> **摘要:** The rapid adoption of LLM-based conversational systems is already transforming the landscape of educational technology. However, the current state-of-the-art learning models do not take into account the student's affective states. Multiple studies in educational psychology support the claim that positive or negative emotional states can impact a student's learning capabilities. To bridge this gap, we present MathBuddy, an emotionally aware LLM-powered Math Tutor, which dynamically models the student's emotions and maps them to relevant pedagogical strategies, making the tutor-student conversation a more empathetic one. The student's emotions are captured from the conversational text as well as from their facial expressions. The student's emotions are aggregated from both modalities to confidently prompt our LLM Tutor for an emotionally-aware response. We have evaluated our model using automatic evaluation metrics across eight pedagogical dimensions and user studies. We report a massive 23 point performance gain using the win rate and a 3 point gain at an overall level using DAMR scores which strongly supports our hypothesis of improving LLM-based tutor's pedagogical abilities by modeling students' emotions. Our dataset and code are available at: https://github.com/ITU-NLP/MathBuddy .
>
---
#### [replaced 019] EpiCache: Episodic KV Cache Management for Long Conversational Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.17396v2](http://arxiv.org/pdf/2509.17396v2)**

> **作者:** Minsoo Kim; Arnav Kundu; Han-Byul Kim; Richa Dixit; Minsik Cho
>
> **摘要:** Modern large language models (LLMs) extend context lengths to up to millions of tokens, enabling AI assistants to generate coherent and personalized responses grounded in long conversational histories. This ability, however, hinges on Key-Value (KV) caching, whose memory grows linearly with dialogue length and quickly becomes the bottleneck in resource-constrained environments. An active line of research for reducing memory bottleneck is KV cache compression, which seeks to limit cache size while preserving accuracy. Yet existing methods face two major limitations: (i) evicting the KV cache after full-context prefill causes unbounded peak memory, and (ii) query-dependent eviction narrows the cache to a single query, leading to failure cases in multi-turn conversations. We introduce EpiCache, a training-free KV cache management framework for long conversational question answering (LongConvQA) under fixed memory budgets. EpiCache bounds cache growth through block-wise prefill and preserves topic-relevant context via episodic KV compression, which clusters conversation history into coherent episodes and applies episode-specific KV cache eviction. We further design an adaptive layer-wise budget allocation strategy that measures each layer's sensitivity to eviction and distributes the memory budget across layers accordingly. Across three LongConvQA benchmarks, EpiCache improves accuracy by up to 40% over recent baselines, sustains near-full KV accuracy under 4-6x compression, and reduces latency and memory by up to 2.4x and 3.5x, thereby enabling efficient multi-turn interaction under strict resource constraints.
>
---
#### [replaced 020] Speech Language Models for Under-Represented Languages: Insights from Wolof
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.15362v2](http://arxiv.org/pdf/2509.15362v2)**

> **作者:** Yaya Sy; Dioula Doucouré; Christophe Cerisara; Irina Illina
>
> **摘要:** We present our journey in training a speech language model for Wolof, an underrepresented language spoken in West Africa, and share key insights. We first emphasize the importance of collecting large-scale, spontaneous, high-quality unsupervised speech data, and show that continued pretraining HuBERT on this dataset outperforms both the base model and African-centric models on ASR. We then integrate this speech encoder into a Wolof LLM to train the first Speech LLM for this language, extending its capabilities to tasks such as speech translation. Furthermore, we explore training the Speech LLM to perform multi-step Chain-of-Thought before transcribing or translating. Our results show that the Speech LLM not only improves speech recognition but also performs well in speech translation. The models and the code will be openly shared.
>
---
#### [replaced 021] LLMs4All: A Review on Large Language Models for Research and Applications in Academic Disciplines
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.19580v2](http://arxiv.org/pdf/2509.19580v2)**

> **作者:** Yanfang Fanny Ye; Zheyuan Zhang; Tianyi Ma; Zehong Wang; Yiyang Li; Shifu Hou; Weixiang Sun; Kaiwen Shi; Yijun Ma; Wei Song; Ahmed Abbasi; Ying Cheng; Jane Cleland-Huang; Steven Corcelli; Patricia Culligan; Robert Goulding; Ming Hu; Ting Hua; John Lalor; Fang Liu; Tengfei Luo; Ed Maginn; Nuno Moniz; Jason Rohr; Brett Savoie; Daniel Slate; Tom Stapleford; Matthew Webber; Olaf Wiest; Johnny Zhang; Nitesh Chawla
>
> **摘要:** Cutting-edge Artificial Intelligence (AI) techniques keep reshaping our view of the world. For example, Large Language Models (LLMs) based applications such as ChatGPT have shown the capability of generating human-like conversation on extensive topics. Due to the impressive performance on a variety of language-related tasks (e.g., open-domain question answering, translation, and document summarization), one can envision the far-reaching impacts that can be brought by the LLMs with broader real-world applications (e.g., customer service, education and accessibility, and scientific discovery). Inspired by their success, this paper will offer an overview of state-of-the-art LLMs and their integration into a wide range of academic disciplines, including: (1) arts, letters, and law (e.g., history, philosophy, political science, arts and architecture, law), (2) economics and business (e.g., finance, economics, accounting, marketing), and (3) science and engineering (e.g., mathematics, physics and mechanical engineering, chemistry and chemical engineering, life sciences and bioengineering, earth sciences and civil engineering, computer science and electrical engineering). Integrating humanity and technology, in this paper, we will explore how LLMs are shaping research and practice in these fields, while also discussing key limitations, open challenges, and future directions in the era of generative AI. The review of how LLMs are engaged across disciplines-along with key observations and insights-can help researchers and practitioners interested in exploiting LLMs to advance their works in diverse real-world applications.
>
---
#### [replaced 022] Ambiguity Resolution in Text-to-Structured Data Mapping
- **分类: cs.CL; cs.LG; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.11679v2](http://arxiv.org/pdf/2505.11679v2)**

> **作者:** Zhibo Hu; Chen Wang; Yanfeng Shu; Hye-Young Paik; Liming Zhu
>
> **备注:** 17 pages, 11 figures
>
> **摘要:** Ambiguity in natural language is a significant obstacle for achieving accurate text to structured data mapping through large language models (LLMs), which affects the performance of tasks such as mapping text to agentic tool calling and text-to-SQL queries. Existing methods to ambiguity handling either rely on the ReACT framework to obtain correct mappings through trial and error, or on supervised fine-tuning to bias models toward specific tasks. In this paper, we adopt a different approach that characterizes representation differences of ambiguous text in the latent space and leverages these differences to identify ambiguity before mapping them to structured data. To detect sentence-level ambiguity, we focus on the relationship between ambiguous questions and their interpretations. Unlike distances calculated by dense embeddings, we introduce a new distance measure based on a path kernel over concepts. With this measurement, we identify patterns to distinguish ambiguous from unambiguous questions. Furthermore, we propose a method for improving LLM performance on ambiguous agentic tool calling through missing concept prediction. Both achieve state-of-the-art results.
>
---
#### [replaced 023] FURINA: Free from Unmergeable Router via LINear Aggregation of mixed experts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.14900v2](http://arxiv.org/pdf/2509.14900v2)**

> **作者:** Jiayi Han; Liang Du; Yinda Chen; Xiao Kang; Weiyang Ding; Donghong Han
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** The Mixture of Experts (MoE) paradigm has been successfully integrated into Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning (PEFT), delivering performance gains with minimal parameter overhead. However, a key limitation of existing MoE-LoRA methods is their reliance on a discrete router, which prevents the integration of the MoE components into the backbone model. To overcome this, we propose FURINA, a novel Free from Unmergeable Router framework based on the LINear Aggregation of experts. FURINA eliminates the router by introducing a Self-Routing mechanism. This is achieved through three core innovations: (1) decoupled learning of the direction and magnitude for LoRA adapters, (2) a shared learnable magnitude vector for consistent activation scaling, and (3) expert selection loss that encourages divergent expert activation. The proposed mechanism leverages the angular similarity between the input and each adapter's directional component to activate experts, which are then scaled by the shared magnitude vector. This design allows the output norm to naturally reflect the importance of each expert, thereby enabling dynamic, router-free routing. The expert selection loss further sharpens this behavior by encouraging sparsity and aligning it with standard MoE activation patterns. We also introduce a shared expert within the MoE-LoRA block that provides stable, foundational knowledge. To the best of our knowledge, FURINA is the first router-free, MoE-enhanced LoRA method that can be fully merged into the backbone model, introducing zero additional inference-time cost or complexity. Extensive experiments demonstrate that FURINA not only significantly outperforms standard LoRA but also matches or surpasses the performance of existing MoE-LoRA methods, while eliminating the extra inference-time overhead of MoE.
>
---
#### [replaced 024] False Friends Are Not Foes: Investigating Vocabulary Overlap in Multilingual Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.18750v2](http://arxiv.org/pdf/2509.18750v2)**

> **作者:** Julie Kallini; Dan Jurafsky; Christopher Potts; Martijn Bartelds
>
> **摘要:** Subword tokenizers trained on multilingual corpora naturally produce overlapping tokens across languages. Does token overlap facilitate cross-lingual transfer or instead introduce interference between languages? Prior work offers mixed evidence, partly due to varied setups and confounders, such as token frequency or subword segmentation granularity. To address this question, we devise a controlled experiment where we train bilingual autoregressive models on multiple language pairs under systematically varied vocabulary overlap settings. Crucially, we explore a new dimension to understanding how overlap affects transfer: the semantic similarity of tokens shared across languages. We first analyze our models' hidden representations and find that overlap of any kind creates embedding spaces that capture cross-lingual semantic relationships, while this effect is much weaker in models with disjoint vocabularies. On XNLI and XQuAD, we find that models with overlap outperform models with disjoint vocabularies, and that transfer performance generally improves as overlap increases. Overall, our findings highlight the advantages of token overlap in multilingual models and show that substantial shared vocabulary remains a beneficial design choice for multilingual tokenizers.
>
---
#### [replaced 025] Investigating Factuality in Long-Form Text Generation: The Roles of Self-Known and Self-Unknown
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.15993v2](http://arxiv.org/pdf/2411.15993v2)**

> **作者:** Lifu Tu; Rui Meng; Shafiq Joty; Yingbo Zhou; Semih Yavuz
>
> **摘要:** Large language models (LLMs) have demonstrated strong capabilities in text understanding and generation. However, they often lack factuality, producing a mixture of true and false information, especially in long-form generation. In this work, we investigates the factuality of long-form text generation across various large language models (LLMs), including GPT-4, Gemini-1.5-Pro, Claude-3-Opus, Llama-3-70B, and Mistral. Our analysis reveals that factuality tend to decline in later sentences of the generated text, accompanied by a rise in the number of unsupported claims. Furthermore, we explore the effectiveness of different evaluation settings to assess whether LLMs can accurately judge the correctness of their own outputs: Self-Known (the percentage of supported atomic claims, decomposed from LLM outputs, that the corresponding LLMs judge as correct) and Self-Unknown (the percentage of unsupported atomic claims that the corresponding LLMs judge as incorrect). Empirically, we observe a positive correlation between higher Self-Known scores and improved factuality, whereas higher Self-Unknown scores are associated with reduced factuality. Interestingly, the number of unsupported claims can increase even without significant changes in a model's self-judgment scores (Self-Known and Self-Unknown), likely as a byproduct of long-form text generation. We also derive a mathematical framework linking Self-Known and Self-Unknown scores to factuality: $\textrm{Factuality}=\frac{1-\textrm{Self-Unknown}}{2-\textrm{Self-Unknown}-\textrm{Self-Known}}$, which aligns with our empirical observations. Additional Retrieval-Augmented Generation (RAG) experiments further highlight the limitations of current LLMs in long-form generation and underscore the need for continued research to improve factuality in long-form text.
>
---
#### [replaced 026] Turning Internal Gap into Self-Improvement: Promoting the Generation-Understanding Unification in MLLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.16663v2](http://arxiv.org/pdf/2507.16663v2)**

> **作者:** Yujin Han; Hao Chen; Andi Han; Zhiheng Wang; Xinyu Liu; Yingya Zhang; Shiwei Zhang; Difan Zou
>
> **备注:** 31 pages, 16 figures, 12 tables
>
> **摘要:** Although unified MLLMs aim to unify generation and understanding, they are considered to exhibit an internal gap, with understanding outperforming generation. Through large-scale evaluation across multiple MLLMs and tasks, we confirm the widespread non-unification of MLLMs, and demonstrate that it indeed stems from weak generation rather than misunderstanding. This finding motivates us to propose a simple yet effective internal gap-based self-improvement framework, which mitigates internal gaps by leveraging stronger understanding to guide weaker generation without relying on any external signals. We validate this strategy through comprehensive experiments: scoring generations with understanding to construct image data for post-training (e.g., SFT and DPO) significantly improves generation while promoting unification. Furthermore, we empirically discover a co-improvement effect of such self-improvement, a phenomenon well known in pre-training but underexplored in post-training. Specifically, as generation improves, understanding becomes more effective at detecting false positives that were previously misclassified as prompt-aligned. To explain this effect, we extend learning dynamic theory to the MLLM setting, showing that the shared empirical neural tangent kernel between generation and understanding encourages aligned learning dynamics, thereby driving co-improvement. This interplay between generation and understanding further motivates a curriculum learning approach for stronger self-improvement: progressively enhanced understanding and generation revisit samples underutilized by pre-trained MLLMs, dynamically expanding post-training data and leading to improved performance and unification.
>
---
#### [replaced 027] Process Reward Models That Think
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.16828v4](http://arxiv.org/pdf/2504.16828v4)**

> **作者:** Muhammad Khalifa; Rishabh Agarwal; Lajanugen Logeswaran; Jaekyeom Kim; Hao Peng; Moontae Lee; Honglak Lee; Lu Wang
>
> **备注:** New results on Qwen3, compute-matched analysis and more
>
> **摘要:** Step-by-step verifiers -- also known as process reward models (PRMs) -- are a key ingredient for test-time scaling. PRMs require step-level supervision, making them expensive to train. This work aims to build data-efficient PRMs as verbalized step-wise reward models that verify every step in the solution by generating a verification chain-of-thought (CoT). We propose ThinkPRM, a long CoT verifier fine-tuned on orders of magnitude fewer process labels than those required by discriminative PRMs. Our approach capitalizes on the inherent reasoning abilities of long CoT models, and outperforms LLM-as-a-Judge and discriminative verifiers -- using only 1% of the process labels in PRM800K -- across several challenging benchmarks. Specifically, ThinkPRM beats the baselines on ProcessBench, MATH-500, and AIME '24 under best-of-N selection and reward-guided search. In an out-of-domain evaluation on a subset of GPQA-Diamond and LiveCodeBench, our PRM surpasses discriminative verifiers trained on the full PRM800K by 8% and 4.5%, respectively. Lastly, under the same token budget, ThinkPRM scales up verification compute more effectively compared to LLM-as-a-Judge, outperforming it by 7.2% on a subset of ProcessBench. Our work highlights the value of generative, long CoT PRMs that can scale test-time compute for verification while requiring minimal supervision for training. Our code, data, and models are released at https://github.com/mukhal/thinkprm.
>
---
#### [replaced 028] From Text to Talk: Audio-Language Model Needs Non-Autoregressive Joint Training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.20072v2](http://arxiv.org/pdf/2509.20072v2)**

> **作者:** Tianqiao Liu; Xueyi Li; Hao Wang; Haoxuan Li; Zhichao Chen; Weiqi Luo; Zitao Liu
>
> **摘要:** Recent advances in large language models (LLMs) have attracted significant interest in extending their capabilities to multimodal scenarios, particularly for speech-to-speech conversational systems. However, existing multimodal models handling interleaved audio and text rely on autoregressive methods, overlooking that text depends on target-target relations whereas audio depends mainly on source-target relations. In this work, we propose Text-to-Talk (TtT), a unified audio-text framework that integrates autoregressive (AR) text generation with non-autoregressive (NAR) audio diffusion in a single Transformer. By leveraging the any-order autoregressive property of absorbing discrete diffusion, our approach provides a unified training objective for text and audio. To support this hybrid generation paradigm, we design a modality-aware attention mechanism that enforces causal decoding for text while allowing bidirectional modeling within audio spans, and further introduce three training strategies that reduce train-test discrepancies. During inference, TtT employs block-wise diffusion to synthesize audio in parallel while flexibly handling variable-length outputs. Extensive experiments across Audio-QA and ASR tasks demonstrate the effectiveness of our approach, with detailed ablation studies validating each proposed component. We will open-source our models, data and code to facilitate future research in this direction.
>
---
#### [replaced 029] Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.01679v2](http://arxiv.org/pdf/2507.01679v2)**

> **作者:** Zeyu Huang; Tianhao Cheng; Zihan Qiu; Zili Wang; Yinghui Xu; Edoardo M. Ponti; Ivan Titov
>
> **备注:** Work in progress
>
> **摘要:** Existing post-training techniques for large language models are broadly categorized into Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT). Each paradigm presents a distinct trade-off: SFT excels at mimicking demonstration data but can lead to problematic generalization as a form of behavior cloning. Conversely, RFT can significantly enhance a model's performance but is prone to learn unexpected behaviors, and its performance is highly sensitive to the initial policy. In this paper, we propose a unified view of these methods and introduce Prefix-RFT, a hybrid approach that synergizes learning from both demonstration and exploration. Using mathematical reasoning problems as a testbed, we empirically demonstrate that Prefix-RFT is both simple and effective. It not only surpasses the performance of standalone SFT and RFT but also outperforms parallel mixed-policy RFT methods. A key advantage is its seamless integration into existing open-source frameworks, requiring only minimal modifications to the standard RFT pipeline. Our analysis highlights the complementary nature of SFT and RFT, and validates that Prefix-RFT effectively harmonizes these two learning paradigms. Furthermore, ablation studies confirm the method's robustness to variations in the quality and quantity of demonstration data. We hope this work offers a new perspective on LLM post-training, suggesting that a unified paradigm that judiciously integrates demonstration and exploration could be a promising direction for future research.
>
---
#### [replaced 030] Polarity Detection of Sustainable Detection Goals in News Text
- **分类: cs.CL; cs.AI; cs.DL**

- **链接: [http://arxiv.org/pdf/2509.19833v2](http://arxiv.org/pdf/2509.19833v2)**

> **作者:** Andrea Cadeddu; Alessandro Chessa; Vincenzo De Leo; Gianni Fenu; Francesco Osborne; Diego Reforgiato Recupero; Angelo Salatino; Luca Secchi
>
> **备注:** Updated as one author was mispelled
>
> **摘要:** The United Nations' Sustainable Development Goals (SDGs) provide a globally recognised framework for addressing critical societal, environmental, and economic challenges. Recent developments in natural language processing (NLP) and large language models (LLMs) have facilitated the automatic classification of textual data according to their relevance to specific SDGs. Nevertheless, in many applications, it is equally important to determine the directionality of this relevance; that is, to assess whether the described impact is positive, neutral, or negative. To tackle this challenge, we propose the novel task of SDG polarity detection, which assesses whether a text segment indicates progress toward a specific SDG or conveys an intention to achieve such progress. To support research in this area, we introduce SDG-POD, a benchmark dataset designed specifically for this task, combining original and synthetically generated data. We perform a comprehensive evaluation using six state-of-the-art large LLMs, considering both zero-shot and fine-tuned configurations. Our results suggest that the task remains challenging for the current generation of LLMs. Nevertheless, some fine-tuned models, particularly QWQ-32B, achieve good performance, especially on specific Sustainable Development Goals such as SDG-9 (Industry, Innovation and Infrastructure), SDG-12 (Responsible Consumption and Production), and SDG-15 (Life on Land). Furthermore, we demonstrate that augmenting the fine-tuning dataset with synthetically generated examples yields improved model performance on this task. This result highlights the effectiveness of data enrichment techniques in addressing the challenges of this resource-constrained domain. This work advances the methodological toolkit for sustainability monitoring and provides actionable insights into the development of efficient, high-performing polarity detection systems.
>
---
#### [replaced 031] Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation
- **分类: cs.CL; cs.LG; I.2.6; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.22842v2](http://arxiv.org/pdf/2505.22842v2)**

> **作者:** Arthur S. Bianchessi; Yasmin C. Aguirre; Rodrigo C. Barros; Lucas S. Kupssinskü
>
> **摘要:** Transformer-based language models rely on positional encoding (PE) to handle token order and support context length extrapolation. However, existing PE methods lack theoretical clarity and rely on limited evaluation metrics to substantiate their extrapolation claims. We propose the Bayesian Attention Mechanism (BAM), a theoretical framework that formulates positional encoding as a prior within a probabilistic model. BAM unifies existing methods (e.g., NoPE and ALiBi) and motivates a new Generalized Gaussian positional prior that substantially improves long-context generalization. Empirically, BAM enables accurate information retrieval at $500\times$ the training context length, outperforming previous state-of-the-art context length generalization in long context retrieval accuracy while maintaining comparable perplexity and introducing minimal additional parameters.
>
---
#### [replaced 032] ALICE: An Interpretable Neural Architecture for Generalization in Substitution Ciphers
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2509.07282v2](http://arxiv.org/pdf/2509.07282v2)**

> **作者:** Jeff Shen; Lindsay M. Smith
>
> **备注:** Preprint. Project page at https://jshen.net/alice. Added section on probing
>
> **摘要:** We present cryptogram solving as an ideal testbed for studying neural network reasoning and generalization; models must decrypt text encoded with substitution ciphers, choosing from 26! possible mappings without explicit access to the cipher. We develop ALICE (an Architecture for Learning Interpretable Cryptogram dEcipherment), a simple encoder-only Transformer that sets a new state-of-the-art for both accuracy and speed on this decryption problem. Surprisingly, ALICE generalizes to unseen ciphers after training on only ${\sim}1500$ unique ciphers, a minute fraction ($3.7 \times 10^{-24}$) of the possible cipher space. To enhance interpretability, we introduce a novel bijective decoding head that explicitly models permutations via the Gumbel-Sinkhorn method, enabling direct extraction of learned cipher mappings. Through early exit and probing experiments, we reveal how ALICE progressively refines its predictions in a way that appears to mirror common human strategies -- early layers place greater emphasis on letter frequencies, while later layers form word-level structures. Our architectural innovations and analysis methods are applicable beyond cryptograms and offer new insights into neural network generalization and interpretability.
>
---
#### [replaced 033] Failure Makes the Agent Stronger: Enhancing Accuracy through Structured Reflection for Reliable Tool Interactions
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.18847v2](http://arxiv.org/pdf/2509.18847v2)**

> **作者:** Junhao Su; Yuanliang Wan; Junwei Yang; Hengyu Shi; Tianyang Han; Junfeng Luo; Yurui Qiu
>
> **备注:** 27pages
>
> **摘要:** Tool-augmented large language models (LLMs) are usually trained with supervised imitation or coarse-grained reinforcement learning that optimizes single tool calls. Current self-reflection practices rely on heuristic prompts or one-way reasoning: the model is urged to 'think more' instead of learning error diagnosis and repair. This is fragile in multi-turn interactions; after a failure the model often repeats the same mistake. We propose structured reflection, which turns the path from error to repair into an explicit, controllable, and trainable action. The agent produces a short yet precise reflection: it diagnoses the failure using evidence from the previous step and then proposes a correct, executable follow-up call. For training we combine DAPO and GSPO objectives with a reward scheme tailored to tool use, optimizing the stepwise strategy Reflect, then Call, then Final. To evaluate, we introduce Tool-Reflection-Bench, a lightweight benchmark that programmatically checks structural validity, executability, parameter correctness, and result consistency. Tasks are built as mini trajectories of erroneous call, reflection, and corrected call, with disjoint train and test splits. Experiments on BFCL v3 and Tool-Reflection-Bench show large gains in multi-turn tool-call success and error recovery, and a reduction of redundant calls. These results indicate that making reflection explicit and optimizing it directly improves the reliability of tool interaction and offers a reproducible path for agents to learn from failure.
>
---
#### [replaced 034] JudgeAgent: Knowledge-wise and Dynamic LLM Evaluation with Agent-as-Interviewer
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.02097v2](http://arxiv.org/pdf/2509.02097v2)**

> **作者:** Zhichao Shi; Xuhui Jiang; Chengjin Xu; Cangli Yao; Zhenxin Huang; Shengjie Ma; Yinghan Shen; Jian Guo; Yuanzhuo Wang
>
> **摘要:** Current evaluation paradigms for large language models (LLMs) suffer from overestimated or biased evaluation and mismatched question difficulty, leading to incomplete evaluations of LLM's knowledge and capability boundaries, which hinder LLM's effective application and optimization. To address these challenges, we propose Agent-as-Interviewer, a dynamic evaluation paradigm that employs LLM agents to conduct multi-turn interactions for evaluation. Unlike current benchmarking or dynamic interaction paradigms, Agent-as-Interviewer utilizes agents to call knowledge tools for wider and deeper knowledge in the dynamic multi-turn question generation, achieving more complete evaluations of the LLM's knowledge boundaries. It also leverages agents to plan query strategies for adjustment of the question difficulty levels, enhancing the difficulty control to match the actual capabilities of target LLMs. Based on this paradigm, we develop JudgeAgent, a knowledge-wise dynamic evaluation framework that employs knowledge-driven synthesis as the agent's tool, and uses difficulty scoring as strategy guidance, thereby finally providing valuable suggestions to help targets optimize themselves. Extensive experiments validate the effectiveness of JudgeAgent's suggestions, demonstrating that Agent-as-Interviewer can accurately identify the knowledge and capability boundaries of target models. The source code is available on https://anonymous.4open.science/r/JudgeAgent.
>
---
#### [replaced 035] On the Perception Bottleneck of VLMs for Chart Understanding
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.18435v2](http://arxiv.org/pdf/2503.18435v2)**

> **作者:** Junteng Liu; Weihao Zeng; Xiwen Zhang; Yijun Wang; Zifei Shan; Junxian He
>
> **备注:** EMNLP 2025: Camera-ready version
>
> **摘要:** Chart understanding requires models to effectively analyze and reason about numerical data, textual elements, and complex visual components. Our observations reveal that the perception capabilities of existing large vision-language models (LVLMs) constitute a critical bottleneck in this process. In this study, we delve into this perception bottleneck by decomposing it into two components: the vision encoder bottleneck, where the visual representation may fail to encapsulate the correct information, and the extraction bottleneck, where the language model struggles to extract the necessary information from the provided visual representations. Through comprehensive experiments, we find that (1) the information embedded within visual representations is substantially richer than what is typically captured by linear extractors, such as the widely used retrieval accuracy metric; (2) While instruction tuning effectively enhances the extraction capability of LVLMs, the vision encoder remains a critical bottleneck, demanding focused attention and improvement. Therefore, we further enhance the visual encoder to mitigate the vision encoder bottleneck under a contrastive learning framework. Empirical results demonstrate that our approach significantly mitigates the perception bottleneck and improves the ability of LVLMs to comprehend charts. Code is publicly available at https://github.com/hkust-nlp/Vision4Chart.
>
---
#### [replaced 036] Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.18672v2](http://arxiv.org/pdf/2508.18672v2)**

> **作者:** Taishi Nakamura; Satoki Ishikawa; Masaki Kawamura; Takumi Okamoto; Daisuke Nohara; Jun Suzuki; Rio Yokota
>
> **备注:** Presented at the Second AI for Math Workshop at ICML
>
> **摘要:** Empirical scaling laws have driven the evolution of large language models (LLMs), yet their coefficients shift whenever the model architecture or data pipeline changes. Mixture-of-Experts (MoE) models, now standard in state-of-the-art systems, introduce a new sparsity dimension that current dense-model frontiers overlook. We investigate how MoE sparsity influences two distinct capability regimes: memorization skills and reasoning skills. By training MoE families that vary total parameters, active parameters, and top-$k$ routing under fixed compute budgets, we disentangle pre-training loss from downstream accuracy. Our results reveal two principles. First, Active FLOPs: models with identical training loss but greater active compute achieve higher reasoning accuracy. Second, Total tokens per parameter (TPP): memorization tasks improve with more parameters, while reasoning tasks benefit from optimal TPP, indicating that reasoning is data-hungry. Neither reinforcement learning post-training (GRPO) nor increased test-time compute alters these trends. We therefore argue that optimal MoE sparsity must be determined jointly by active FLOPs and TPP, revising the classical picture of compute-optimal scaling. Our model checkpoints, code and logs are open-source at https://github.com/rioyokotalab/optimal-sparsity.
>
---
#### [replaced 037] Ko-PIQA: A Korean Physical Commonsense Reasoning Dataset with Cultural Context
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.11303v2](http://arxiv.org/pdf/2509.11303v2)**

> **作者:** Dasol Choi; Jungwhan Kim; Guijin Son
>
> **摘要:** Physical commonsense reasoning datasets like PIQA are predominantly English-centric and lack cultural diversity. We introduce Ko-PIQA, a Korean physical commonsense reasoning dataset that incorporates cultural context. Starting from 3.01 million web-crawled questions, we employed a multi-stage filtering approach using three language models to identify 11,553 PIQA-style questions. Through GPT-4o refinement and human validation, we obtained 441 high-quality question-answer pairs. A key feature of Ko-PIQA is its cultural grounding: 19.7\% of questions contain culturally specific elements like traditional Korean foods (kimchi), clothing (hanbok), and specialized appliances (kimchi refrigerators) that require culturally-aware reasoning beyond direct translation. We evaluate seven language models on Ko-PIQA, with the best model achieving 83.22\% accuracy while the weakest reaches only 59.86\%, demonstrating significant room for improvement. Models particularly struggle with culturally specific scenarios, highlighting the importance of culturally diverse datasets. Ko-PIQA serves as both a benchmark for Korean language models and a foundation for more inclusive commonsense reasoning research. The dataset and code will be publicly available.
>
---
#### [replaced 038] MathFimer: Enhancing Mathematical Reasoning by Expanding Reasoning Steps through Fill-in-the-Middle Task
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11684v2](http://arxiv.org/pdf/2502.11684v2)**

> **作者:** Yuchen Yan; Yongliang Shen; Yang Liu; Jin Jiang; Xin Xu; Mengdi Zhang; Jian Shao; Yueting Zhuang
>
> **摘要:** Mathematical reasoning represents a critical frontier in advancing large language models (LLMs). While step-by-step approaches have emerged as the dominant paradigm for mathematical problem-solving in LLMs, the quality of reasoning steps in training data fundamentally constrains the performance of the models. Recent studies has demonstrated that more detailed intermediate steps can enhance model performance, yet existing methods for step expansion either require more powerful external models or incur substantial computational costs. In this paper, we introduce MathFimer, a novel framework for mathematical reasoning step expansion inspired by the "Fill-in-the-middle" task from code completion. By decomposing solution chains into prefix-suffix pairs and training models to reconstruct missing intermediate steps, we develop a specialized model, MathFimer-7B, on our carefully curated NuminaMath-FIM dataset. We then apply these models to enhance existing mathematical reasoning datasets by inserting detailed intermediate steps into their solution chains, creating MathFimer-expanded versions. Through comprehensive experiments on multiple mathematical reasoning datasets, including MathInstruct, MetaMathQA and etc., we demonstrate that models trained on MathFimer-expanded data consistently outperform their counterparts trained on original data across various benchmarks such as GSM8K and MATH. Our approach offers a practical, scalable solution for enhancing mathematical reasoning capabilities in LLMs without relying on powerful external models or expensive inference procedures.
>
---
#### [replaced 039] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.15801v3](http://arxiv.org/pdf/2505.15801v3)**

> **作者:** Yuchen Yan; Jin Jiang; Zhenbang Ren; Yijun Li; Xudong Cai; Yang Liu; Xin Xu; Mengdi Zhang; Jian Shao; Yongliang Shen; Jun Xiao; Yueting Zhuang
>
> **备注:** Project Page: https://zju-real.github.io/VerifyBench Dataset: https://huggingface.co/datasets/ZJU-REAL/VerifyBench Code: https://github.com/ZJU-REAL/VerifyBench
>
> **摘要:** Large reasoning models such as OpenAI o1 and DeepSeek-R1 have achieved remarkable performance in the domain of reasoning. A key component of their training is the incorporation of verifiable rewards within reinforcement learning (RL). However, existing reward benchmarks do not evaluate reference-based reward systems, leaving researchers with limited understanding of the accuracy of verifiers used in RL. In this paper, we introduce two benchmarks, VerifyBench and VerifyBench-Hard, designed to assess the performance of reference-based reward systems. These benchmarks are constructed through meticulous data collection and curation, followed by careful human annotation to ensure high quality. Current models still show considerable room for improvement on both VerifyBench and VerifyBench-Hard, especially smaller-scale models. Furthermore, we conduct a thorough and comprehensive analysis of evaluation results, offering insights for understanding and developing reference-based reward systems. Our proposed benchmarks serve as effective tools for guiding the development of verifier accuracy and the reasoning capabilities of models trained via RL in reasoning tasks.
>
---
#### [replaced 040] SIM-CoT: Supervised Implicit Chain-of-Thought
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.20317v2](http://arxiv.org/pdf/2509.20317v2)**

> **作者:** Xilin Wei; Xiaoran Liu; Yuhang Zang; Xiaoyi Dong; Yuhang Cao; Jiaqi Wang; Xipeng Qiu; Dahua Lin
>
> **摘要:** Implicit Chain-of-Thought (CoT) methods offer a token-efficient alternative to explicit CoT reasoning in Large Language Models (LLMs), but a persistent performance gap has limited their adoption. We identify a core latent instability issue when scaling the computational budget of implicit CoT: as the number of reasoning tokens increases, training often becomes unstable and collapses. Our analysis shows that this instability arises from latent representations becoming homogeneous and losing semantic diversity, caused by insufficient step-level supervision in current implicit CoT methods. To address this, we propose SIM-CoT, a plug-and-play training module that introduces step-level supervision to stabilize and enrich the latent reasoning space. SIM-CoT employs an auxiliary decoder during training to align each implicit token with its corresponding explicit reasoning step, ensuring latent states capture distinct and meaningful information. The auxiliary decoder is removed at inference, preserving the efficiency of implicit CoT with no added overhead. It also provides interpretability by projecting each latent token onto an explicit reasoning vocabulary, enabling per-step visualization and diagnosis. SIM-CoT significantly improves both in-domain accuracy and out-of-domain stability of implicit CoT methods, boosting Coconut by +8.2\% on GPT-2 and CODI by +3.0\% on LLaMA-3.1 8B. It further surpasses the explicit CoT baseline on GPT-2 by 2.1\% with 2.3$\times$ greater token efficiency, while closing the performance gap on larger models like LLaMA-3.1 8B. Code: https://github.com/InternLM/SIM-CoT
>
---
#### [replaced 041] Bias Similarity Measurement: A Black-Box Audit of Fairness Across LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12010v4](http://arxiv.org/pdf/2410.12010v4)**

> **作者:** Hyejun Jeong; Shiqing Ma; Amir Houmansadr
>
> **备注:** Code available at https://github.com/HyejunJeong/bias_llm
>
> **摘要:** Large Language Models (LLMs) reproduce social biases, yet prevailing evaluations score models in isolation, obscuring how biases persist across families and releases. We introduce Bias Similarity Measurement (BSM), which treats fairness as a relational property between models, unifying scalar, distributional, behavioral, and representational signals into a single similarity space. Evaluating 30 LLMs on 1M+ prompts, we find that instruction tuning primarily enforces abstention rather than altering internal representations; small models gain little accuracy and can become less fair under forced choice; and open-weight models can match or exceed proprietary systems. Family signatures diverge: Gemma favors refusal, LLaMA 3.1 approaches neutrality with fewer refusals, and converges toward abstention-heavy behavior overall. Counterintuitively, Gemma 3 Instruct matches GPT-4-level fairness at far lower cost, whereas Gemini's heavy abstention suppresses utility. Beyond these findings, BSM offers an auditing workflow for procurement, regression testing, and lineage screening, and extends naturally to code and multilingual settings. Our results reframe fairness not as isolated scores but as comparative bias similarity, enabling systematic auditing of LLM ecosystems. Code available at https://github.com/HyejunJeong/bias_llm.
>
---
#### [replaced 042] TactfulToM: Do LLMs Have the Theory of Mind Ability to Understand White Lies?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.17054v2](http://arxiv.org/pdf/2509.17054v2)**

> **作者:** Yiwei Liu; Emma Jane Pretty; Jiahao Huang; Saku Sugawara
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** While recent studies explore Large Language Models' (LLMs) performance on Theory of Mind (ToM) reasoning tasks, research on ToM abilities that require more nuanced social context is limited, such as white lies. We introduce TactfulToM, a novel English benchmark designed to evaluate LLMs' ability to understand white lies within real-life conversations and reason about prosocial motivations behind them, particularly when they are used to spare others' feelings and maintain social harmony. Our benchmark is generated through a multi-stage human-in-the-loop pipeline where LLMs expand manually designed seed stories into conversations to maintain the information asymmetry between participants necessary for authentic white lies. We show that TactfulToM is challenging for state-of-the-art models, which perform substantially below humans, revealing shortcomings in their ability to fully comprehend the ToM reasoning that enables true understanding of white lies.
>
---
#### [replaced 043] ixi-GEN: Efficient Industrial sLLMs through Domain Adaptive Continual Pretraining
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.06795v3](http://arxiv.org/pdf/2507.06795v3)**

> **作者:** Seonwu Kim; Yohan Na; Kihun Kim; Hanhee Cho; Geun Lim; Mintae Kim; Seongik Park; Ki Hyun Kim; Youngsub Han; Byoung-Ki Jeon
>
> **备注:** Accepted at EMNLP 2025 Industry Track
>
> **摘要:** The emergence of open-source large language models (LLMs) has expanded opportunities for enterprise applications; however, many organizations still lack the infrastructure to deploy and maintain large-scale models. As a result, small LLMs (sLLMs) have become a practical alternative, despite their inherent performance limitations. While Domain Adaptive Continual Pretraining (DACP) has been previously explored as a method for domain adaptation, its utility in commercial applications remains under-examined. In this study, we validate the effectiveness of applying a DACP-based recipe across diverse foundation models and service domains. Through extensive experiments and real-world evaluations, we demonstrate that DACP-applied sLLMs achieve substantial gains in target domain performance while preserving general capabilities, offering a cost-efficient and scalable solution for enterprise-level deployment.
>
---
#### [replaced 044] ARF-RLHF: Adaptive Reward-Following for RLHF through Emotion-Driven Self-Supervision and Trace-Biased Dynamic Optimization
- **分类: cs.CL; cs.AI; 68T05, 68Q25; I.2.6; I.2.7**

- **链接: [http://arxiv.org/pdf/2507.03069v2](http://arxiv.org/pdf/2507.03069v2)**

> **作者:** YuXuan Zhang
>
> **备注:** This version adds several baselines and experiments, clarifies some ambiguous descriptions, and corrects the reported value for the ReScore result on the ALPACA task to 7.8%
>
> **摘要:** Current RLHF methods such as PPO and DPO typically reduce human preferences to binary labels, which are costly to obtain and too coarse to reflect individual variation. We observe that expressions of satisfaction and dissatisfaction follow stable linguistic patterns across users, indicating that more informative supervisory signals can be extracted from free-form feedback. Building on this insight, we introduce Adaptive Reward-Following (ARF), which converts natural feedback into continuous preference trajectories and optimizes them using the novel TraceBias algorithm. Across diverse LLMs and preference domains, ARF consistently outperforms PPO and DPO, improving alignment by up to 7.6%. Our results demonstrate that continuous reward modeling provides a scalable path toward personalized and theoretically grounded RLHF.
>
---
#### [replaced 045] Inference-Time Scaling for Generalist Reward Modeling
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.02495v3](http://arxiv.org/pdf/2504.02495v3)**

> **作者:** Zijun Liu; Peiyi Wang; Runxin Xu; Shirong Ma; Chong Ruan; Peng Li; Yang Liu; Yu Wu
>
> **备注:** Preprint, under review. 44 pages. Models are available at https://huggingface.co/collections/BBQGOD/deepseek-grm-68b4681169dbb97fd30614b5 and https://www.modelscope.cn/collections/DeepSeek-GRM-ff6a2d8babdd4a
>
> **摘要:** Reinforcement learning (RL) has been widely adopted in post-training for large language models (LLMs) at scale. Recently, the incentivization of reasoning capabilities in LLMs from RL indicates that $\textit{proper learning methods could enable effective inference-time scalability}$. A key challenge of RL is to obtain accurate reward signals for LLMs in various domains beyond verifiable questions or artificial rules. In this work, we investigate how to improve reward modeling (RM) with more inference compute for general queries, i.e. the $\textbf{inference-time scalability of generalist RM}$, and further, how to improve the effectiveness of performance-compute scaling with proper learning methods. For the RM approach, we adopt pointwise generative reward modeling (GRM) to enable flexibility for different input types and potential for inference-time scaling. For the learning method, we propose Self-Principled Critique Tuning (SPCT) to foster scalable reward generation behaviors in GRMs through online RL, to generate principles adaptively and critiques accurately, resulting in $\textbf{DeepSeek-GRM}$ models. Furthermore, for effective inference-time scaling, we use parallel sampling to expand compute usage, and introduce a meta RM to guide voting process for better scaling performance. Empirically, we show that SPCT significantly improves the quality and scalability of GRMs, outperforming existing methods and models in various RM benchmarks without severe biases, and could achieve better performance compared to training-time scaling. DeepSeek-GRM still meets challenges in some tasks, which we believe can be addressed by future efforts in generalist reward systems. The models are released at Hugging Face and ModelScope.
>
---
#### [replaced 046] Inverse Reinforcement Learning with Dynamic Reward Scaling for LLM Alignment
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.18991v5](http://arxiv.org/pdf/2503.18991v5)**

> **作者:** Ruoxi Cheng; Haoxuan Ma; Weixin Wang; Ranjie Duan; Jiexi Liu; Xiaoshuang Jia; Simeng Qin; Xiaochun Cao; Yang Liu; Xiaojun Jia
>
> **备注:** The first three authors contributed equally to this work
>
> **摘要:** Alignment is vital for safely deploying large language models (LLMs). Existing techniques are either reward-based (train a reward model on preference pairs and optimize with reinforcement learning) or reward-free (directly fine-tune on ranked outputs). Recent research shows that well-tuned reward-based pipelines remain robust, and single-response demonstrations can outperform pairwise preference data. However, two challenges persist: (1) imbalanced safety datasets that overrepresent common hazards while neglecting long-tail threats; and (2) static reward models that ignore task difficulty, limiting optimization efficiency and attainable gains. We propose DR-IRL (Dynamically adjusting Rewards through Inverse Reinforcement Learning). We first train category-specific reward models using a balanced safety dataset covering seven harmful categories via IRL. Then we enhance Group Relative Policy Optimization (GRPO) by introducing dynamic reward scaling--adjusting rewards by task difficulty--data-level hardness by text encoder cosine similarity, model-level responsiveness by reward gaps. Extensive experiments across various benchmarks and LLMs demonstrate that DR-IRL outperforms all baseline methods in safety alignment while maintaining usefulness.
>
---
#### [replaced 047] C3: A Bilingual Benchmark for Spoken Dialogue Models Exploring Challenges in Complex Conversations
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.22968v2](http://arxiv.org/pdf/2507.22968v2)**

> **作者:** Chengqian Ma; Wei Tao; Yiwen Guo
>
> **备注:** EMNLP 2025 main; Project Page: https://step-out.github.io/C3-web/
>
> **摘要:** Spoken Dialogue Models (SDMs) have recently attracted significant attention for their ability to generate voice responses directly to users' spoken queries. Despite their increasing popularity, there exists a gap in research focused on comprehensively understanding their practical effectiveness in comprehending and emulating human conversations. This is especially true compared to text-based Large Language Models (LLMs), which benefit from extensive benchmarking. Human voice interactions are inherently more complex than text due to characteristics unique to spoken dialogue. Ambiguity poses one challenge, stemming from semantic factors like polysemy, as well as phonological aspects such as heterograph, heteronyms, and stress patterns. Additionally, context-dependency, like omission, coreference, and multi-turn interaction, adds further complexity to human conversational dynamics. To illuminate the current state of SDM development and to address these challenges, we present a benchmark dataset in this paper, which comprises 1,079 instances in English and Chinese. Accompanied by an LLM-based evaluation method that closely aligns with human judgment, this dataset facilitates a comprehensive exploration of the performance of SDMs in tackling these practical challenges.
>
---
#### [replaced 048] Labeling Free-text Data using Language Model Ensembles
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.08413v3](http://arxiv.org/pdf/2501.08413v3)**

> **作者:** Jiaxing Qiu; Dongliang Guo; Natalie Papini; Noelle Peace; Hannah F. Fitterman-Harris; Cheri A. Levinson; Tom Hartvigsen; Teague R. Henry
>
> **摘要:** Free-text responses are commonly collected in psychological studies, providing rich qualitative insights that quantitative measures may not capture. Labeling curated topics of research interest in free-text data by multiple trained human coders is typically labor-intensive and time-consuming. Though large language models (LLMs) excel in language processing, LLM-assisted labeling techniques relying on closed-source LLMs cannot be directly applied to free-text data, without explicit consent for external use. In this study, we propose a framework of assembling locally-deployable LLMs to enhance the labeling of predetermined topics in free-text data under privacy constraints. Analogous to annotation by multiple human raters, this framework leverages the heterogeneity of diverse open-source LLMs. The ensemble approach seeks a balance between the agreement and disagreement across LLMs, guided by a relevancy scoring methodology that utilizes embedding distances between topic descriptions and LLMs' reasoning. We evaluated the ensemble approach using both publicly accessible Reddit data from eating disorder related forums, and free-text responses from eating disorder patients, both complemented by human annotations. We found that: (1) there is heterogeneity in the performance of labeling among same-sized LLMs, with some showing low sensitivity but high precision, while others exhibit high sensitivity but low precision. (2) Compared to individual LLMs, the ensemble of LLMs achieved the highest accuracy and optimal precision-sensitivity trade-off in predicting human annotations. (3) The relevancy scores across LLMs showed greater agreement than dichotomous labels, indicating that the relevancy scoring method effectively mitigates the heterogeneity in LLMs' labeling.
>
---
#### [replaced 049] ConsistentChat: Building Skeleton-Guided Consistent Multi-Turn Dialogues for Large Language Models from Scratch
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.03558v2](http://arxiv.org/pdf/2506.03558v2)**

> **作者:** Jiawei Chen; Xinyan Guan; Qianhao Yuan; Guozhao Mo; Weixiang Zhou; Yaojie Lu; Hongyu Lin; Ben He; Le Sun; Xianpei Han
>
> **备注:** EMNLP 2025 Main Conference (Oral Presentation)
>
> **摘要:** Current instruction data synthesis methods primarily focus on single-turn instructions and often neglect cross-turn coherence, resulting in context drift and reduced task completion rates in extended conversations. To address this limitation, we propose Skeleton-Guided Multi-Turn Dialogue Generation, a framework that constrains multi-turn instruction synthesis by explicitly modeling human conversational intent. It operates in two stages: (1) Intent Modeling, which captures the global structure of human dialogues by assigning each conversation to one of nine well-defined intent trajectories, ensuring a coherent and goal-oriented information flow; and (2) Skeleton Generation, which constructs a structurally grounded sequence of user queries aligned with the modeled intent, thereby serving as a scaffold that constrains and guides the downstream instruction synthesis process. Based on this process, we construct ConsistentChat, a multi-turn instruction dataset with approximately 15,000 multi-turn conversations and 224,392 utterances. Experiments on the Light, Topdial, and MT-Eval benchmarks show that models fine-tuned on ConsistentChat achieve a 20-30% improvement in chat consistency and up to a 15% increase in task success rate, significantly outperforming models trained on existing single-turn and multi-turn instruction datasets.
>
---
#### [replaced 050] OpenGVL - Benchmarking Visual Temporal Progress for Data Curation
- **分类: cs.RO; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.17321v2](http://arxiv.org/pdf/2509.17321v2)**

> **作者:** Paweł Budzianowski; Emilia Wiśnios; Gracjan Góral; Igor Kulakov; Viktor Petrenko; Krzysztof Walas
>
> **摘要:** Data scarcity remains one of the most limiting factors in driving progress in robotics. However, the amount of available robotics data in the wild is growing exponentially, creating new opportunities for large-scale data utilization. Reliable temporal task completion prediction could help automatically annotate and curate this data at scale. The Generative Value Learning (GVL) approach was recently proposed, leveraging the knowledge embedded in vision-language models (VLMs) to predict task progress from visual observations. Building upon GVL, we propose OpenGVL, a comprehensive benchmark for estimating task progress across diverse challenging manipulation tasks involving both robotic and human embodiments. We evaluate the capabilities of publicly available open-source foundation models, showing that open-source model families significantly underperform closed-source counterparts, achieving only approximately $70\%$ of their performance on temporal progress prediction tasks. Furthermore, we demonstrate how OpenGVL can serve as a practical tool for automated data curation and filtering, enabling efficient quality assessment of large-scale robotics datasets. We release the benchmark along with the complete codebase at \href{github.com/budzianowski/opengvl}{OpenGVL}.
>
---
#### [replaced 051] From Replication to Redesign: Exploring Pairwise Comparisons for LLM-Based Peer Review
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11343v2](http://arxiv.org/pdf/2506.11343v2)**

> **作者:** Yaohui Zhang; Haijing Zhang; Wenlong Ji; Tianyu Hua; Nick Haber; Hancheng Cao; Weixin Liang
>
> **摘要:** The advent of large language models (LLMs) offers unprecedented opportunities to reimagine peer review beyond the constraints of traditional workflows. Despite these opportunities, prior efforts have largely focused on replicating traditional review workflows with LLMs serving as direct substitutes for human reviewers, while limited attention has been given to exploring new paradigms that fundamentally rethink how LLMs can participate in the academic review process. In this paper, we introduce and explore a novel mechanism that employs LLM agents to perform pairwise comparisons among manuscripts instead of individual scoring. By aggregating outcomes from substantial pairwise evaluations, this approach enables a more accurate and robust measure of relative manuscript quality. Our experiments demonstrate that this comparative approach significantly outperforms traditional rating-based methods in identifying high-impact papers. However, our analysis also reveals emergent biases in the selection process, notably a reduced novelty in research topics and an increased institutional imbalance. These findings highlight both the transformative potential of rethinking peer review with LLMs and critical challenges that future systems must address to ensure equity and diversity.
>
---
#### [replaced 052] Searching for Privacy Risks in LLM Agents via Simulation
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.10880v2](http://arxiv.org/pdf/2508.10880v2)**

> **作者:** Yanzhe Zhang; Diyi Yang
>
> **备注:** Preprint
>
> **摘要:** The widespread deployment of LLM-based agents is likely to introduce a critical privacy threat: malicious agents that proactively engage others in multi-turn interactions to extract sensitive information. However, the evolving nature of such dynamic dialogues makes it challenging to anticipate emerging vulnerabilities and design effective defenses. To tackle this problem, we present a search-based framework that alternates between improving attack and defense strategies through the simulation of privacy-critical agent interactions. Specifically, we employ LLMs as optimizers to analyze simulation trajectories and iteratively propose new agent instructions. To explore the strategy space more efficiently, we further utilize parallel search with multiple threads and cross-thread propagation. Through this process, we find that attack strategies escalate from direct requests to sophisticated tactics, such as impersonation and consent forgery, while defenses evolve from simple rule-based constraints to robust identity-verification state machines. The discovered attacks and defenses transfer across diverse scenarios and backbone models, demonstrating strong practical utility for building privacy-aware agents.
>
---
#### [replaced 053] PLaMo 2 Technical Report
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.04897v2](http://arxiv.org/pdf/2509.04897v2)**

> **作者:** Preferred Networks; :; Kaizaburo Chubachi; Yasuhiro Fujita; Shinichi Hemmi; Yuta Hirokawa; Kentaro Imajo; Toshiki Kataoka; Goro Kobayashi; Kenichi Maehashi; Calvin Metzger; Hiroaki Mikami; Shogo Murai; Daisuke Nishino; Kento Nozawa; Toru Ogawa; Shintarou Okada; Daisuke Okanohara; Shunta Saito; Shotaro Sano; Shuji Suzuki; Kuniyuki Takahashi; Daisuke Tanaka; Avinash Ummadisingu; Hanqin Wang; Sixue Wang; Tianqi Xu
>
> **摘要:** In this report, we introduce PLaMo 2, a series of Japanese-focused large language models featuring a hybrid Samba-based architecture that transitions to full attention via continual pre-training to support 32K token contexts. Training leverages extensive synthetic corpora to overcome data scarcity, while computational efficiency is achieved through weight reuse and structured pruning. This efficient pruning methodology produces an 8B model that achieves performance comparable to our previous 100B model. Post-training further refines the models using a pipeline of supervised fine-tuning (SFT) and direct preference optimization (DPO), enhanced by synthetic Japanese instruction data and model merging techniques. Optimized for inference using vLLM and quantization with minimal accuracy loss, the PLaMo 2 models achieve state-of-the-art results on Japanese benchmarks, outperforming similarly-sized open models in instruction-following, language fluency, and Japanese-specific knowledge.
>
---
#### [replaced 054] JUREX-4E: Juridical Expert-Annotated Four-Element Knowledge Base for Legal Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.17166v2](http://arxiv.org/pdf/2502.17166v2)**

> **作者:** Huanghai Liu; Quzhe Huang; Qingjing Chen; Yiran Hu; Jiayu Ma; Yun Liu; Weixing Shen; Yansong Feng
>
> **摘要:** In recent years, Large Language Models (LLMs) have been widely applied to legal tasks. To enhance their understanding of legal texts and improve reasoning accuracy, a promising approach is to incorporate legal theories. One of the most widely adopted theories is the Four-Element Theory (FET), which defines the crime constitution through four elements: Subject, Object, Subjective Aspect, and Objective Aspect. While recent work has explored prompting LLMs to follow FET, our evaluation demonstrates that LLM-generated four-elements are often incomplete and less representative, limiting their effectiveness in legal reasoning. To address these issues, we present JUREX-4E, an expert-annotated four-element knowledge base covering 155 criminal charges. The annotations follow a progressive hierarchical framework grounded in legal source validity and incorporate diverse interpretive methods to ensure precision and authority. We evaluate JUREX-4E on the Similar Charge Disambiguation task and apply it to Legal Case Retrieval. Experimental results validate the high quality of JUREX-4E and its substantial impact on downstream legal tasks, underscoring its potential for advancing legal AI applications. The dataset and code are available at: https://github.com/THUlawtech/JUREX
>
---
#### [replaced 055] Part-of-speech tagging for Nagamese Language using CRF
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.19343v2](http://arxiv.org/pdf/2509.19343v2)**

> **作者:** Alovi N Shohe; Chonglio Khiamungam; Teisovi Angami
>
> **备注:** 8 pages
>
> **摘要:** This paper investigates part-of-speech tagging, an important task in Natural Language Processing (NLP) for the Nagamese language. The Nagamese language, a.k.a. Naga Pidgin, is an Assamese-lexified Creole language developed primarily as a means of communication in trade between the Nagas and people from Assam in northeast India. A substantial amount of work in part-of-speech-tagging has been done for resource-rich languages like English, Hindi, etc. However, no work has been done in the Nagamese language. To the best of our knowledge, this is the first attempt at part-of-speech tagging for the Nagamese Language. The aim of this work is to identify the part-of-speech for a given sentence in the Nagamese language. An annotated corpus of 16,112 tokens is created and applied machine learning technique known as Conditional Random Fields (CRF). Using CRF, an overall tagging accuracy of 85.70%; precision, recall of 86%, and f1-score of 85% is achieved. Keywords. Nagamese, NLP, part-of-speech, machine learning, CRF.
>
---
#### [replaced 056] How to Evaluate Medical AI
- **分类: cs.AI; cs.CL; I.2.7; I.2.1**

- **链接: [http://arxiv.org/pdf/2509.11941v2](http://arxiv.org/pdf/2509.11941v2)**

> **作者:** Ilia Kopanichuk; Petr Anokhin; Vladimir Shaposhnikov; Vladimir Makharev; Ekaterina Tsapieva; Iaroslav Bespalov; Dmitry V. Dylov; Ivan Oseledets
>
> **备注:** 10 pages, 7 fugures
>
> **摘要:** The integration of artificial intelligence (AI) into medical diagnostic workflows requires robust and consistent evaluation methods to ensure reliability, clinical relevance, and the inherent variability in expert judgments. Traditional metrics like precision and recall often fail to account for the inherent variability in expert judgments, leading to inconsistent assessments of AI performance. Inter-rater agreement statistics like Cohen's Kappa are more reliable but they lack interpretability. We introduce Relative Precision and Recall of Algorithmic Diagnostics (RPAD and RRAD) - a new evaluation metrics that compare AI outputs against multiple expert opinions rather than a single reference. By normalizing performance against inter-expert disagreement, these metrics provide a more stable and realistic measure of the quality of predicted diagnosis. In addition to the comprehensive analysis of diagnostic quality measures, our study contains a very important side result. Our evaluation methodology allows us to avoid selecting diagnoses from a limited list when evaluating a given case. Instead, both the models being tested and the examiners verifying them arrive at a free-form diagnosis. In this automated methodology for establishing the identity of free-form clinical diagnoses, a remarkable 98% accuracy becomes attainable. We evaluate our approach using 360 medical dialogues, comparing multiple large language models (LLMs) against a panel of physicians. Large-scale study shows that top-performing models, such as DeepSeek-V3, achieve consistency on par with or exceeding expert consensus. Moreover, we demonstrate that expert judgments exhibit significant variability - often greater than that between AI and humans. This finding underscores the limitations of any absolute metrics and supports the need to adopt relative metrics in medical AI.
>
---
#### [replaced 057] ILRe: Intermediate Layer Retrieval for Context Compression in Causal Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.17892v2](http://arxiv.org/pdf/2508.17892v2)**

> **作者:** Manlai Liang; Mandi Liu; Jiangzhou Ji; Huaijun Li; Haobo Yang; Yaohan He; Jinlong Li
>
> **摘要:** Large Language Models (LLMs) have demonstrated success across many benchmarks. However, they still exhibit limitations in long-context scenarios, primarily due to their short effective context length, quadratic computational complexity, and high memory overhead when processing lengthy inputs. To mitigate these issues, we introduce a novel context compression pipeline, called Intermediate Layer Retrieval (ILRe), which determines one intermediate decoder layer offline, encodes context by streaming chunked prefill only up to that layer, and recalls tokens by the attention scores between the input query and full key cache in that specified layer. In particular, we propose a multi-pooling kernels allocating strategy in the token recalling process to maintain the completeness of semantics. Our approach not only reduces the prefilling complexity from $O(L^2)$ to $O(L)$ and trims the memory footprint to a few tenths of that required for the full context, but also delivers performance comparable to or superior to the full-context setup in long-context scenarios. Without additional post training or operator development, ILRe can process a single $1M$ tokens request in less than half a minute (speedup $\approx 180\times$) and scores RULER-$1M$ benchmark of $\approx 79.8$ with model Llama-3.1-UltraLong-8B-1M-Instruct on a Huawei Ascend 910B NPU.
>
---
#### [replaced 058] A Framework for Situating Innovations, Opportunities, and Challenges in Advancing Vertical Systems with Large AI Models
- **分类: cs.AI; cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2504.02793v2](http://arxiv.org/pdf/2504.02793v2)**

> **作者:** Gaurav Verma; Jiawei Zhou; Mohit Chandra; Srijan Kumar; Munmun De Choudhury
>
> **备注:** AAAI/ACM AIES 2025 Main Conference Paper; Webpage: https://gaurav22verma.github.io/vertical-systems-with-large-ai-models/
>
> **摘要:** Large artificial intelligence (AI) models have garnered significant attention for their remarkable, often "superhuman", performance on standardized benchmarks. However, when these models are deployed in high-stakes verticals such as healthcare, education, and law, they often reveal notable limitations. For instance, they exhibit brittleness to minor variations in input data, present contextually uninformed decisions in critical settings, and undermine user trust by confidently producing or reproducing inaccuracies. These challenges in applying large models necessitate cross-disciplinary innovations to align the models' capabilities with the needs of real-world applications. We introduce a framework that addresses this gap through a layer-wise abstraction of innovations aimed at meeting users' requirements with large models. Through multiple case studies, we illustrate how researchers and practitioners across various fields can operationalize this framework. Beyond modularizing the pipeline of transforming large models into useful "vertical systems", we also highlight the dynamism that exists within different layers of the framework. Finally, we discuss how our framework can guide researchers and practitioners to (i) optimally situate their innovations (e.g., when vertical-specific insights can empower broadly impactful vertical-agnostic innovations), (ii) uncover overlooked opportunities (e.g., spotting recurring problems across verticals to develop practically useful foundation models instead of chasing benchmarks), and (iii) facilitate cross-disciplinary communication of critical challenges (e.g., enabling a shared vocabulary for AI developers, domain experts, and human-computer interaction scholars). Project webpage: https://gaurav22verma.github.io/vertical-systems-with-large-ai-models/
>
---
#### [replaced 059] BabyLM's First Constructions: Causal probing provides a signal of learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.02147v2](http://arxiv.org/pdf/2506.02147v2)**

> **作者:** Joshua Rozner; Leonie Weissweiler; Cory Shain
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Construction grammar posits that language learners acquire constructions (form-meaning pairings) from the statistics of their environment. Recent work supports this hypothesis by showing sensitivity to constructions in pretrained language models (PLMs), including one recent study (Rozner et al., 2025) demonstrating that constructions shape RoBERTa's output distribution. However, models under study have generally been trained on developmentally implausible amounts of data, casting doubt on their relevance to human language learning. Here we use Rozner et al.'s methods to evaluate construction learning in masked language models from the 2024 BabyLM Challenge. Our results show that even when trained on developmentally plausible quantities of data, models learn diverse constructions, even hard cases that are superficially indistinguishable. We further find correlational evidence that constructional performance may be functionally relevant: models that better represent construction perform better on the BabyLM benchmarks.
>
---
#### [replaced 060] CogniLoad: A Synthetic Natural Language Reasoning Benchmark With Tunable Length, Intrinsic Difficulty, and Distractor Density
- **分类: cs.CL; cs.AI; cs.LG; 68T50 (Primary) 68T07, 68T05, 68T20, 68T27 (Secondary); I.2.7; I.2.6; I.2.4; I.2.8**

- **链接: [http://arxiv.org/pdf/2509.18458v2](http://arxiv.org/pdf/2509.18458v2)**

> **作者:** Daniel Kaiser; Arnoldo Frigessi; Ali Ramezani-Kebrya; Benjamin Ricaud
>
> **备注:** 29 pages (main: 12 + supplemental material: 17), 6 figures, 4 tables, Code: https://github.com/kaiserdan/cogniload, Data: https://huggingface.co/datasets/cogniloadteam/cogniload
>
> **摘要:** Current benchmarks for long-context reasoning in Large Language Models (LLMs) often blur critical factors like intrinsic task complexity, distractor interference, and task length. To enable more precise failure analysis, we introduce CogniLoad, a novel synthetic benchmark grounded in Cognitive Load Theory (CLT). CogniLoad generates natural-language logic puzzles with independently tunable parameters that reflect CLT's core dimensions: intrinsic difficulty ($d$) controls intrinsic load; distractor-to-signal ratio ($\rho$) regulates extraneous load; and task length ($N$) serves as an operational proxy for conditions demanding germane load. Evaluating 22 SotA reasoning LLMs, CogniLoad reveals distinct performance sensitivities, identifying task length as a dominant constraint and uncovering varied tolerances to intrinsic complexity and U-shaped responses to distractor ratios. By offering systematic, factorial control over these cognitive load dimensions, CogniLoad provides a reproducible, scalable, and diagnostically rich tool for dissecting LLM reasoning limitations and guiding future model development.
>
---
#### [replaced 061] When Does Meaning Backfire? Investigating the Role of AMRs in NLI
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14613v2](http://arxiv.org/pdf/2506.14613v2)**

> **作者:** Junghyun Min; Xiulin Yang; Shira Wein
>
> **备注:** 9 pages, 2 figures. *SEM 2025
>
> **摘要:** Natural Language Inference (NLI) relies heavily on adequately parsing the semantic content of the premise and hypothesis. In this work, we investigate whether adding semantic information in the form of an Abstract Meaning Representation (AMR) helps pretrained language models better generalize in NLI. Our experiments integrating AMR into NLI in both fine-tuning and prompting settings show that the presence of AMR in fine-tuning hinders model generalization while prompting with AMR leads to slight gains in GPT-4o. However, an ablation study reveals that the improvement comes from amplifying surface-level differences rather than aiding semantic reasoning. This amplification can mislead models to predict non-entailment even when the core meaning is preserved.
>
---
#### [replaced 062] THCM-CAL: Temporal-Hierarchical Causal Modelling with Conformal Calibration for Clinical Risk Prediction
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.17844v2](http://arxiv.org/pdf/2506.17844v2)**

> **作者:** Xin Zhang; Qiyu Wei; Yingjie Zhu; Fanyi Wu; Sophia Ananiadou
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** Automated clinical risk prediction from electronic health records (EHRs) demands modeling both structured diagnostic codes and unstructured narrative notes. However, most prior approaches either handle these modalities separately or rely on simplistic fusion strategies that ignore the directional, hierarchical causal interactions by which narrative observations precipitate diagnoses and propagate risk across admissions. In this paper, we propose THCM-CAL, a Temporal-Hierarchical Causal Model with Conformal Calibration. Our framework constructs a multimodal causal graph where nodes represent clinical entities from two modalities: Textual propositions extracted from notes and ICD codes mapped to textual descriptions. Through hierarchical causal discovery, THCM-CAL infers three clinically grounded interactions: intra-slice same-modality sequencing, intra-slice cross-modality triggers, and inter-slice risk propagation. To enhance prediction reliability, we extend conformal prediction to multi-label ICD coding, calibrating per-code confidence intervals under complex co-occurrences. Experimental results on MIMIC-III and MIMIC-IV demonstrate the superiority of THCM-CAL.
>
---
#### [replaced 063] Fractional Reasoning via Latent Steering Vectors Improves Inference Time Compute
- **分类: cs.LG; cs.AI; cs.CL; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.15882v2](http://arxiv.org/pdf/2506.15882v2)**

> **作者:** Sheng Liu; Tianlang Chen; Pan Lu; Haotian Ye; Yizheng Chen; Lei Xing; James Zou
>
> **备注:** 18 pages, 5 figures, Project website: https://shengliu66.github.io/fractreason/
>
> **摘要:** Test-time compute has emerged as a powerful paradigm for improving the performance of large language models (LLMs), where generating multiple outputs or refining individual chains can significantly boost answer accuracy. However, existing methods like Best-of-N, majority voting, and self-reflection typically apply reasoning in a uniform way across inputs, overlooking the fact that different problems may require different levels of reasoning depth. In this work, we propose Fractional Reasoning, a training-free and model-agnostic framework that enables continuous control over reasoning intensity at inference time, going beyond the limitations of fixed instructional prompts. Our method operates by extracting the latent steering vector associated with deeper reasoning and reapplying it with a tunable scaling factor, allowing the model to tailor its reasoning process to the complexity of each input. This supports two key modes of test-time scaling: (1) improving output quality in breadth-based strategies (e.g., Best-of-N, majority voting), and (2) enhancing the correctness of individual reasoning chains in depth-based strategies (e.g., self-reflection). Experiments on GSM8K, MATH500, and GPQA demonstrate that Fractional Reasoning consistently improves performance across diverse reasoning tasks and models.
>
---
#### [replaced 064] Problem Solved? Information Extraction Design Space for Layout-Rich Documents using LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.18179v3](http://arxiv.org/pdf/2502.18179v3)**

> **作者:** Gaye Colakoglu; Gürkan Solmaz; Jonathan Fürst
>
> **备注:** accepted at EMNLP'25
>
> **摘要:** This paper defines and explores the design space for information extraction (IE) from layout-rich documents using large language models (LLMs). The three core challenges of layout-aware IE with LLMs are 1) data structuring, 2) model engagement, and 3) output refinement. Our study investigates the sub-problems and methods within these core challenges, such as input representation, chunking, prompting, selection of LLMs, and multimodal models. It examines the effect of different design choices through LayIE-LLM, a new, open-source, layout-aware IE test suite, benchmarking against traditional, fine-tuned IE models. The results on two IE datasets show that LLMs require adjustment of the IE pipeline to achieve competitive performance: the optimized configuration found with LayIE-LLM achieves 13.3--37.5 F1 points more than a general-practice baseline configuration using the same LLM. To find a well-working configuration, we develop a one-factor-at-a-time (OFAT) method that achieves near-optimal results. Our method is only 0.8--1.8 points lower than the best full factorial exploration with a fraction (2.8%) of the required computation. Overall, we demonstrate that, if well-configured, general-purpose LLMs match the performance of specialized models, providing a cost-effective, finetuning-free alternative. Our test-suite is available at https://github.com/gayecolakoglu/LayIE-LLM.
>
---
#### [replaced 065] Reinforcement Learning on Pre-Training Data
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.19249v2](http://arxiv.org/pdf/2509.19249v2)**

> **作者:** Siheng Li; Kejiao Li; Zenan Xu; Guanhua Huang; Evander Yang; Kun Li; Haoyuan Wu; Jiajia Wu; Zihao Zheng; Chenchen Zhang; Kun Shi; Kyrierl Deng; Qi Yi; Ruibin Xiong; Tingqiang Xu; Yuhao Jiang; Jianfeng Yan; Yuyuan Zeng; Guanghui Xu; Jinbao Xue; Zhijiang Xu; Zheng Fang; Shuai Li; Qibin Liu; Xiaoxue Li; Zhuoyu Li; Yangyu Tao; Fei Gao; Cheng Jiang; Bo Chao Wang; Kai Liu; Jianchen Zhu; Wai Lam; Wayyt Wang; Bo Zhou; Di Wang
>
> **备注:** Work in progress
>
> **摘要:** The growing disparity between the exponential scaling of computational resources and the finite growth of high-quality text data now constrains conventional scaling approaches for large language models (LLMs). To address this challenge, we introduce Reinforcement Learning on Pre-Training data (RLPT), a new training-time scaling paradigm for optimizing LLMs. In contrast to prior approaches that scale training primarily through supervised learning, RLPT enables the policy to autonomously explore meaningful trajectories to learn from pre-training data and improve its capability through reinforcement learning (RL). While existing RL strategies such as reinforcement learning from human feedback (RLHF) and reinforcement learning with verifiable rewards (RLVR) rely on human annotation for reward construction, RLPT eliminates this dependency by deriving reward signals directly from pre-training data. Specifically, it adopts a next-segment reasoning objective, rewarding the policy for accurately predicting subsequent text segments conditioned on the preceding context. This formulation allows RL to be scaled on pre-training data, encouraging the exploration of richer trajectories across broader contexts and thereby fostering more generalizable reasoning skills. Extensive experiments on both general-domain and mathematical reasoning benchmarks across multiple models validate the effectiveness of RLPT. For example, when applied to Qwen3-4B-Base, RLPT yields absolute improvements of $3.0$, $5.1$, $8.1$, $6.0$, $6.6$, and $5.3$ on MMLU, MMLU-Pro, GPQA-Diamond, KOR-Bench, AIME24, and AIME25, respectively. The results further demonstrate favorable scaling behavior, suggesting strong potential for continued gains with more compute. In addition, RLPT provides a solid foundation, extending the reasoning boundaries of LLMs and enhancing RLVR performance.
>
---
#### [replaced 066] ImpliRet: Benchmarking the Implicit Fact Retrieval Challenge
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.14407v3](http://arxiv.org/pdf/2506.14407v3)**

> **作者:** Zeinab Sadat Taghavi; Ali Modarressi; Yunpu Ma; Hinrich Schütze
>
> **摘要:** Retrieval systems are central to many NLP pipelines, but often rely on surface-level cues such as keyword overlap and lexical semantic similarity. To evaluate retrieval beyond these shallow signals, recent benchmarks introduce reasoning-heavy queries; however, they primarily shift the burden to query-side processing techniques -- like prompting or multi-hop retrieval -- that can help resolve complexity. In contrast, we present Impliret, a benchmark that shifts the reasoning challenge to document-side processing: The queries are simple, but relevance depends on facts stated implicitly in documents through temporal (e.g., resolving "two days ago"), arithmetic, and world knowledge relationships. We evaluate a range of sparse and dense retrievers, all of which struggle in this setting: the best nDCG@10 is only 14.91%. We also test whether long-context models can overcome this limitation. But even with a short context of only thirty documents, including the positive document, GPT-o4-mini scores only 55.54%, showing that document-side reasoning remains a challenge. Our codes are available at github.com/ZeinabTaghavi/IMPLIRET.
>
---
#### [replaced 067] Small LLMs with Expert Blocks Are Good Enough for Hyperparamter Tuning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.15561v3](http://arxiv.org/pdf/2509.15561v3)**

> **作者:** Om Naphade; Saksham Bansal; Parikshit Pareek
>
> **摘要:** Hyper-parameter Tuning (HPT) is a necessary step in machine learning (ML) pipelines but becomes computationally expensive and opaque with larger models. Recently, Large Language Models (LLMs) have been explored for HPT, yet most rely on models exceeding 100 billion parameters. We propose an Expert Block Framework for HPT using Small LLMs. At its core is the Trajectory Context Summarizer (TCS), a deterministic block that transforms raw training trajectories into structured context, enabling small LLMs to analyze optimization progress with reliability comparable to larger models. Using two locally-run LLMs (phi4:reasoning14B and qwen2.5-coder:32B) and a 10-trial budget, our TCS-enabled HPT pipeline achieves average performance within ~0.9 percentage points of GPT-4 across six diverse tasks.
>
---
#### [replaced 068] CoT-Space: A Theoretical Framework for Internal Slow-Thinking via Reinforcement Learning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.04027v2](http://arxiv.org/pdf/2509.04027v2)**

> **作者:** Zeyu Gan; Hao Yi; Yong Liu
>
> **备注:** Preprint Edition
>
> **摘要:** Reinforcement Learning (RL) has become a pivotal approach for enhancing the reasoning capabilities of Large Language Models (LLMs). However, a significant theoretical gap persists, as traditional token-level RL frameworks fail to align with the reasoning-level nature of complex, multi-step thought processes like Chain-of-Thought (CoT). To address this challenge, we introduce CoT-Space, a novel theoretical framework that recasts LLM reasoning from a discrete token-prediction task to an optimization process within a continuous, reasoning-level semantic space. This shift in perspective serves as a conceptual bridge, revitalizing foundational principles from classical learning theory to analyze the unique dynamics of LLMs. By analyzing this process from both a noise perspective and a risk perspective, we demonstrate that the convergence to an optimal CoT length is a natural consequence of the fundamental trade-off between underfitting and overfitting. Furthermore, extensive experiments provide strong empirical validation for our theoretical findings. Our framework not only provides a coherent explanation for empirical phenomena such as overthinking but also offers a solid theoretical foundation to guide the future development of more effective and generalizable reasoning agents. We open-source our code at https://github.com/ZyGan1999/CoT-Space.
>
---
#### [replaced 069] Causal-Counterfactual RAG: The Integration of Causal-Counterfactual Reasoning into RAG
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2509.14435v2](http://arxiv.org/pdf/2509.14435v2)**

> **作者:** Harshad Khadilkar; Abhay Gupta
>
> **摘要:** Large language models (LLMs) have transformed natural language processing (NLP), enabling diverse applications by integrating large-scale pre-trained knowledge. However, their static knowledge limits dynamic reasoning over external information, especially in knowledge-intensive domains. Retrieval-Augmented Generation (RAG) addresses this challenge by combining retrieval mechanisms with generative modeling to improve contextual understanding. Traditional RAG systems suffer from disrupted contextual integrity due to text chunking and over-reliance on semantic similarity for retrieval, often resulting in shallow and less accurate responses. We propose Causal-Counterfactual RAG, a novel framework that integrates explicit causal graphs representing cause-effect relationships into the retrieval process and incorporates counterfactual reasoning grounded on the causal structure. Unlike conventional methods, our framework evaluates not only direct causal evidence but also the counterfactuality of associated causes, combining results from both to generate more robust, accurate, and interpretable answers. By leveraging causal pathways and associated hypothetical scenarios, Causal-Counterfactual RAG preserves contextual coherence, reduces hallucination, and enhances reasoning fidelity.
>
---
#### [replaced 070] Reformulation is All You Need: Addressing Malicious Text Features in DNNs
- **分类: cs.LG; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2502.00652v2](http://arxiv.org/pdf/2502.00652v2)**

> **作者:** Yi Jiang; Oubo Ma; Yong Yang; Tong Zhang; Shouling Ji
>
> **备注:** Accepted by journal "Machine Intelligence Research"
>
> **摘要:** Human language encompasses a wide range of intricate and diverse implicit features, which attackers can exploit to launch adversarial or backdoor attacks, compromising DNN models for NLP tasks. Existing model-oriented defenses often require substantial computational resources as model size increases, whereas sample-oriented defenses typically focus on specific attack vectors or schemes, rendering them vulnerable to adaptive attacks. We observe that the root cause of both adversarial and backdoor attacks lies in the encoding process of DNN models, where subtle textual features, negligible for human comprehension, are erroneously assigned significant weight by less robust or trojaned models. Based on it we propose a unified and adaptive defense framework that is effective against both adversarial and backdoor attacks. Our approach leverages reformulation modules to address potential malicious features in textual inputs while preserving the original semantic integrity. Extensive experiments demonstrate that our framework outperforms existing sample-oriented defense baselines across a diverse range of malicious textual features.
>
---
#### [replaced 071] Collab-Overcooked: Benchmarking and Evaluating Large Language Models as Collaborative Agents
- **分类: cs.CL; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2502.20073v3](http://arxiv.org/pdf/2502.20073v3)**

> **作者:** Haochen Sun; Shuwen Zhang; Lujie Niu; Lei Ren; Hao Xu; Hao Fu; Fangkun Zhao; Caixia Yuan; Xiaojie Wang
>
> **备注:** Accepted to EMNLP 2025 Main Conference. Camera-Ready Version. 30 pages, 17 figures
>
> **摘要:** Large Language Models (LLMs) based agent systems have made great strides in real-world applications beyond traditional NLP tasks. This paper proposes a new LLM-based Multi-Agent System (LLM-MAS) benchmark, Collab-Overcooked, built on the popular Overcooked-AI game with more applicable and challenging tasks in interactive environments. Collab-Overcooked extends existing benchmarks in two novel ways. First, it provides a multi-agent framework supporting diverse tasks and objectives and encourages collaboration through natural language communication. Second, it introduces a spectrum of process-oriented evaluation metrics to assess the fine-grained collaboration capabilities of different LLM agents, a dimension often overlooked in prior work. We conduct extensive experiments with 13 popular LLMs and show that, while the LLMs exhibit a strong ability in goal interpretation, there are significant shortcomings in active collaboration and continuous adaptation, which are critical for efficiently fulfilling complex tasks. Notably, we highlight the strengths and weaknesses of LLM-MAS and provide insights for improving and evaluating LLM-MAS on a unified and open-source benchmark. The environments, 30 open-ended tasks, and the evaluation package are publicly available at https://github.com/YusaeMeow/Collab-Overcooked.
>
---
#### [replaced 072] UDDETTS: Unifying Discrete and Dimensional Emotions for Controllable Emotional Text-to-Speech
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10599v2](http://arxiv.org/pdf/2505.10599v2)**

> **作者:** Jiaxuan Liu; Yang Xiang; Han Zhao; Xiangang Li; Yingying Gao; Shilei Zhang; Zhenhua Ling
>
> **备注:** Under review
>
> **摘要:** Recent large language models (LLMs) have made great progress in the field of text-to-speech (TTS), but they still face major challenges in synthesizing fine-grained emotional speech in an interpretable manner. Traditional methods rely on discrete emotion labels to control emotion categories and intensities, which cannot capture the complexity and continuity of human emotional perception and expression. The lack of large-scale emotional speech datasets with balanced emotion distributions and fine-grained emotional annotations often causes overfitting in synthesis models and impedes effective emotion control. To address these issues, we propose UDDETTS, a universal LLM framework unifying discrete and dimensional emotions for controllable emotional TTS. This model introduces the interpretable Arousal-Dominance-Valence (ADV) space for dimensional emotion description and supports emotion control driven by either discrete emotion labels or nonlinearly quantified ADV values. Furthermore, a semi-supervised training strategy is designed to comprehensively utilize diverse speech datasets with different types of emotional annotations to train the UDDETTS. Experiments show that UDDETTS achieves linear emotion control along three interpretable dimensions, and exhibits superior end-to-end emotional speech synthesis capabilities. Code and demos are available at: https://anonymous.4open.science/w/UDDETTS.
>
---
#### [replaced 073] ButterflyQuant: Ultra-low-bit LLM Quantization through Learnable Orthogonal Butterfly Transforms
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.09679v2](http://arxiv.org/pdf/2509.09679v2)**

> **作者:** Bingxin Xu; Zhen Dong; Oussama Elachqar; Yuzhang Shang
>
> **备注:** Replace discrete Hadamard transforms with continuous Butterfly transforms to facilitate the learning of rotation matrices in LLM quantization
>
> **摘要:** Large language models require massive memory footprints, severely limiting deployment on consumer hardware. Quantization reduces memory through lower numerical precision, but extreme 2-bit quantization suffers from catastrophic performance loss due to outliers in activations. Rotation-based methods such as QuIP and QuaRot apply orthogonal transforms to eliminate outliers before quantization, using computational invariance: $\mathbf{y} = \mathbf{Wx} = (\mathbf{WQ}^T)(\mathbf{Qx})$ for orthogonal $\mathbf{Q}$. However, these methods use fixed transforms--Hadamard matrices achieving optimal worst-case coherence $\mu = 1/\sqrt{n}$--that cannot adapt to specific weight distributions. We identify that different transformer layers exhibit distinct outlier patterns, motivating layer-adaptive rotations rather than one-size-fits-all approaches. In this work, we propose ButterflyQuant, which replaces Hadamard rotations with learnable butterfly transforms parameterized by continuous Givens rotation angles. Unlike Hadamard's discrete $\{+1, -1\}$ entries that are non-differentiable and thus prohibit gradient-based learning, butterfly transforms' continuous parameterization enables smooth optimization while guaranteeing orthogonality by construction. This orthogonal constraint ensures theoretical guarantees in outlier suppression while achieving $O(n \log n)$ computational complexity with only $\frac{n \log n}{2}$ learnable parameters. We further introduce a uniformity regularization on post-transformation activations to promote smoother distributions amenable to quantization. Learning requires only 128 calibration samples and converges in minutes on a single GPU--a negligible one-time cost. For LLaMA-2-7B with 2-bit quantization, ButterflyQuant achieves 15.4 perplexity versus 37.3 for QuIP. \href{https://github.com/42Shawn/Butterflyquant-llm}{Codes} are available.
>
---
#### [replaced 074] Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.02872v2](http://arxiv.org/pdf/2505.02872v2)**

> **作者:** Cfir Avraham Hadar; Omer Shubi; Yoav Meiri; Amit Heshes; Yevgeni Berzak
>
> **摘要:** When reading, we often have specific information that interests us in a text. For example, you might be reading this paper because you are curious about LLMs for eye movements in reading, the experimental design, or perhaps you wonder ``This sounds like science fiction. Does it actually work?''. More broadly, in daily life, people approach texts with any number of text-specific goals that guide their reading behavior. In this work, we ask, for the first time, whether open-ended reading goals can be automatically decoded solely from eye movements in reading. To address this question, we introduce goal decoding tasks and evaluation frameworks using large-scale eye tracking for reading data in English with hundreds of text-specific information seeking tasks. We develop and compare several discriminative and generative multimodal text and eye movements LLMs for these tasks. Our experiments show considerable success on the task of selecting the correct goal among several options, and even progress towards free-form textual reconstruction of the precise goal formulation. These results open the door for further scientific investigation of goal driven reading, as well as the development of educational and assistive technologies that will rely on real-time decoding of reader goals from their eye movements.
>
---
#### [replaced 075] Thinking Augmented Pre-training
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.20186v2](http://arxiv.org/pdf/2509.20186v2)**

> **作者:** Liang Wang; Nan Yang; Shaohan Huang; Li Dong; Furu Wei
>
> **备注:** 19 pages
>
> **摘要:** This paper introduces a simple and scalable approach to improve the data efficiency of large language model (LLM) training by augmenting existing text data with thinking trajectories. The compute for pre-training LLMs has been growing at an unprecedented rate, while the availability of high-quality data remains limited. Consequently, maximizing the utility of available data constitutes a significant research challenge. A primary impediment is that certain high-quality tokens are difficult to learn given a fixed model capacity, as the underlying rationale for a single token can be exceptionally complex and deep. To address this issue, we propose Thinking augmented Pre-Training (TPT), a universal methodology that augments text with automatically generated thinking trajectories. Such augmentation effectively increases the volume of the training data and makes high-quality tokens more learnable through step-by-step reasoning and decomposition. We apply TPT across diverse training configurations up to $100$B tokens, encompassing pre-training with both constrained and abundant data, as well as mid-training from strong open-source checkpoints. Experimental results indicate that our method substantially improves the performance of LLMs across various model sizes and families. Notably, TPT enhances the data efficiency of LLM pre-training by a factor of $3$. For a $3$B parameter model, it improves the post-training performance by over $10\%$ on several challenging reasoning benchmarks.
>
---
#### [replaced 076] Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.05386v2](http://arxiv.org/pdf/2507.05386v2)**

> **作者:** Song Lai; Haohan Zhao; Rong Feng; Changyi Ma; Wenzhuo Liu; Hongbo Zhao; Xi Lin; Dong Yi; Min Xie; Qingfu Zhang; Hongbin Liu; Gaofeng Meng; Fei Zhu
>
> **摘要:** Continual post-training (CPT) is a popular and effective technique for adapting foundation models like multimodal large language models to specific and ever-evolving downstream tasks. While existing research has primarily concentrated on methods like data replay, model expansion, or parameter regularization, the fundamental role of the learning paradigm within CPT remains largely unexplored. This paper presents a comparative analysis of two core post-training paradigms: supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT), investigating their respective impacts on knowledge retention during CPT. Our experiments are conducted on a benchmark comprising seven diverse multimodal tasks, utilizing Qwen2.5-VL-7B-Instruct as the base model for continual post-training. The investigation yields two significant findings: (1) When continuously learning on downstream tasks, SFT leads to catastrophic forgetting of previously learned tasks. In contrast, RFT inherently preserves prior knowledge and achieve performance comparable to multi-task training. (2) RFT successfully protects and even enhances the model's general knowledge on standard benchmarks (e.g., MMMU and MMLU-Pro). Conversely, SFT degrades general model capabilities severely. Further analysis reveals that this stability is not primarily due to explicit mechanisms like KL penalty or chain-of-thought reasoning. Instead, we identify an implicit regularization mechanism inherent to RFT as a key contributing factor. Our theoretical analysis suggests that RFT's gradient updates are naturally scaled by the reward variance, acting as a data-dependent regularizer that inherently protects previously acquired knowledge. Finally, we propose a rollout-based instance filtering algorithm to enhance the stability and efficiency of RFT. Our comprehensive study demonstrates the superiority of RFT as a robust paradigm for continual post-training.
>
---
#### [replaced 077] A Comprehensive Taxonomy of Negation for NLP and Neural Retrievers
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2507.22337v2](http://arxiv.org/pdf/2507.22337v2)**

> **作者:** Roxana Petcu; Samarth Bhargav; Maarten de Rijke; Evangelos Kanoulas
>
> **摘要:** Understanding and solving complex reasoning tasks is vital for addressing the information needs of a user. Although dense neural models learn contextualised embeddings, they still underperform on queries containing negation. To understand this phenomenon, we study negation in both traditional neural information retrieval and LLM-based models. We (1) introduce a taxonomy of negation that derives from philosophical, linguistic, and logical definitions; (2) generate two benchmark datasets that can be used to evaluate the performance of neural information retrieval models and to fine-tune models for a more robust performance on negation; and (3) propose a logic-based classification mechanism that can be used to analyze the performance of retrieval models on existing datasets. Our taxonomy produces a balanced data distribution over negation types, providing a better training setup that leads to faster convergence on the NevIR dataset. Moreover, we propose a classification schema that reveals the coverage of negation types in existing datasets, offering insights into the factors that might affect the generalization of fine-tuned models on negation.
>
---
#### [replaced 078] Quantifying depressive mental states with large language models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.09487v2](http://arxiv.org/pdf/2502.09487v2)**

> **作者:** Jakub Onysk; Quentin J. M. Huys
>
> **备注:** main text - 9 pages, 6 figures;
>
> **摘要:** Large Language Models (LLMs) may have an important role to play in mental health by facilitating the quantification of verbal expressions used to communicate emotions, feelings and thoughts. While there has been substantial and very promising work in this area, the fundamental limits are uncertain. Here, focusing on depressive symptoms, we outline and evaluate LLM performance on three critical tests. The first test evaluates LLM performance on a novel ground-truth dataset from a large human sample (n=770). This dataset is novel as it contains both standard clinically validated quantifications of depression symptoms and specific verbal descriptions of the thoughts related to each symptom by the same individual. The performance of LLMs on this richly informative data shows an upper bound on the performance in this domain, and allow us to examine the extent to which inference about symptoms generalises. Second, we test to what extent the latent structure in LLMs can capture the clinically observed patterns. We train supervised sparse auto-encoders (sSAE) to predict specific symptoms and symptom patterns within a syndrome. We find that sSAE weights can effectively modify the clinical pattern produced by the model, and thereby capture the latent structure of relevant clinical variation. Third, if LLMs correctly capture and quantify relevant mental states, then these states should respond to changes in emotional states induced by validated emotion induction interventions. We show that this holds in a third experiment with 190 participants. Overall, this work provides foundational insights into the quantification of pathological mental states with LLMs, highlighting hard limits on the requirements of the data underlying LLM-based quantification; but also suggesting LLMs show substantial conceptual alignment.
>
---
#### [replaced 079] ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.18896v2](http://arxiv.org/pdf/2506.18896v2)**

> **作者:** Jiaru Zou; Ling Yang; Jingwen Gu; Jiahao Qiu; Ke Shen; Jingrui He; Mengdi Wang
>
> **备注:** Accepted by NeurIPS 2025. Project: https://github.com/Gen-Verse/ReasonFlux
>
> **摘要:** Process Reward Models (PRMs) have recently emerged as a powerful framework for supervising intermediate reasoning steps in large language models (LLMs). Previous PRMs are primarily trained on model final output responses and struggle to evaluate intermediate thinking trajectories robustly, especially in the emerging setting of trajectory-response outputs generated by frontier reasoning models like Deepseek-R1. In this work, we introduce ReasonFlux-PRM, a novel trajectory-aware PRM explicitly designed to evaluate the trajectory-response type of reasoning traces. ReasonFlux-PRM incorporates both step-level and trajectory-level supervision, enabling fine-grained reward assignment aligned with structured chain-of-thought data. We adapt ReasonFlux-PRM to support reward supervision under both offline and online settings, including (i) selecting high-quality model distillation data for downstream supervised fine-tuning of smaller models, (ii) providing dense process-level rewards for policy optimization during reinforcement learning, and (iii) enabling reward-guided Best-of-N test-time scaling. Empirical results on challenging downstream benchmarks such as AIME, MATH500, and GPQA-Diamond demonstrate that ReasonFlux-PRM-7B selects higher quality data than strong PRMs (e.g., Qwen2.5-Math-PRM-72B) and human-curated baselines. Furthermore, our derived ReasonFlux-PRM-7B yields consistent performance improvements, achieving average gains of 12.1% in supervised fine-tuning, 4.5% in reinforcement learning, and 6.3% in test-time scaling. We also release our efficient ReasonFlux-PRM-1.5B for resource-constrained applications and edge deployment. Project: https://github.com/Gen-Verse/ReasonFlux
>
---
#### [replaced 080] AdaSVD: Adaptive Singular Value Decomposition for Large Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01403v4](http://arxiv.org/pdf/2502.01403v4)**

> **作者:** Zhiteng Li; Mingyuan Xia; Jingyuan Zhang; Zheng Hui; Haotong Qin; Linghe Kong; Yulun Zhang; Xiaokang Yang
>
> **备注:** The code and models will be available at https://github.com/ZHITENGLI/AdaSVD
>
> **摘要:** Large language models (LLMs) have achieved remarkable success in natural language processing (NLP) tasks, yet their substantial memory requirements present significant challenges for deployment on resource-constrained devices. Singular Value Decomposition (SVD) has emerged as a promising compression technique for LLMs, offering considerable reductions in memory overhead. However, existing SVD-based methods often struggle to effectively mitigate the errors introduced by SVD truncation, leading to a noticeable performance gap when compared to the original models. Furthermore, applying a uniform compression ratio across all transformer layers fails to account for the varying importance of different layers. To address these challenges, we propose AdaSVD, an adaptive SVD-based LLM compression approach. Specifically, AdaSVD introduces adaComp, which adaptively compensates for SVD truncation errors by alternately updating the singular matrices $\mathcal{U}$ and $\mathcal{V}^\top$. Additionally, AdaSVD introduces adaCR, which adaptively assigns layer-specific compression ratios based on the relative importance of each layer. Extensive experiments across multiple LLM/VLM families and evaluation metrics demonstrate that AdaSVD consistently outperforms state-of-the-art (SOTA) SVD-based methods, achieving superior performance with significantly reduced memory requirements. Code and models of AdaSVD will be available at https://github.com/ZHITENGLI/AdaSVD.
>
---
#### [replaced 081] TestAgent: Automatic Benchmarking and Exploratory Interaction for Evaluating LLMs in Vertical Domains
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.11507v5](http://arxiv.org/pdf/2410.11507v5)**

> **作者:** Wanying Wang; Zeyu Ma; Xuhong Wang; Yangchun Zhang; Pengfei Liu; Mingang Chen
>
> **备注:** Wang et al. Copyright 2026 lEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, including reprinting/republishing, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work. DOI will be added upon IEEE Xplore publication
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed in highly specialized vertical domains, the evaluation of their domain-specific performance becomes critical. However, existing evaluations for vertical domains typically rely on the labor-intensive construction of static single-turn datasets, which present two key limitations: (i) manual data construction is costly and must be repeated for each new domain, and (ii) static single-turn evaluations are misaligned with the dynamic multi-turn interactions in real-world applications, limiting the assessment of professionalism and stability. To address these, we propose TestAgent, a framework for automatic benchmarking and exploratory dynamic evaluation in vertical domains. TestAgent leverages retrieval-augmented generation to create domain-specific questions from user-provided knowledge sources, combined with a two-stage criteria generation process, thereby enabling scalable and automated benchmark creation. Furthermore, it introduces a reinforcement learning-guided multi-turn interaction strategy that adaptively determines question types based on real-time model responses, dynamically probing knowledge boundaries and stability. Extensive experiments across medical, legal, and governmental domains demonstrate that TestAgent enables efficient cross-domain benchmark generation and yields deeper insights into model behavior through dynamic exploratory evaluation. This work establishes a new paradigm for automated and in-depth evaluation of LLMs in vertical domains.
>
---
#### [replaced 082] Higher-Order DisCoCat (Peirce-Lambek-Montague semantics)
- **分类: cs.CL; math.CT**

- **链接: [http://arxiv.org/pdf/2311.17813v2](http://arxiv.org/pdf/2311.17813v2)**

> **作者:** Alexis Toumi; Giovanni de Felice
>
> **备注:** In Proceedings ACT 2024, arXiv:2509.18357
>
> **摘要:** We propose a new definition of higher-order DisCoCat (categorical compositional distributional) models where the meaning of a word is not a diagram, but a diagram-valued higher-order function. Our models can be seen as a variant of Montague semantics based on a lambda calculus where the primitives act on string diagrams rather than logical formulae. As a special case, we show how to translate from the Lambek calculus into Peirce's system beta for first-order logic. This allows us to give a purely diagrammatic treatment of higher-order and non-linear processes in natural language semantics: adverbs, prepositions, negation and quantifiers. The definition presented in this article comes with a proof-of-concept implementation in DisCoPy, the Python library for string diagrams.
>
---
#### [replaced 083] LM-Searcher: Cross-domain Neural Architecture Search with LLMs via Unified Numerical Encoding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.05657v3](http://arxiv.org/pdf/2509.05657v3)**

> **作者:** Yuxuan Hu; Jihao Liu; Ke Wang; Jinliang Zhen; Weikang Shi; Manyuan Zhang; Qi Dou; Rui Liu; Aojun Zhou; Hongsheng Li
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Recent progress in Large Language Models (LLMs) has opened new avenues for solving complex optimization problems, including Neural Architecture Search (NAS). However, existing LLM-driven NAS approaches rely heavily on prompt engineering and domain-specific tuning, limiting their practicality and scalability across diverse tasks. In this work, we propose LM-Searcher, a novel framework that leverages LLMs for cross-domain neural architecture optimization without the need for extensive domain-specific adaptation. Central to our approach is NCode, a universal numerical string representation for neural architectures, which enables cross-domain architecture encoding and search. We also reformulate the NAS problem as a ranking task, training LLMs to select high-performing architectures from candidate pools using instruction-tuning samples derived from a novel pruning-based subspace sampling strategy. Our curated dataset, encompassing a wide range of architecture-performance pairs, encourages robust and transferable learning. Comprehensive experiments demonstrate that LM-Searcher achieves competitive performance in both in-domain (e.g., CNNs for image classification) and out-of-domain (e.g., LoRA configurations for segmentation and generation) tasks, establishing a new paradigm for flexible and generalizable LLM-based architecture search. The datasets and models will be released at https://github.com/Ashone3/LM-Searcher.
>
---
#### [replaced 084] Scaling Rich Style-Prompted Text-to-Speech Datasets
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2503.04713v2](http://arxiv.org/pdf/2503.04713v2)**

> **作者:** Anuj Diwan; Zhisheng Zheng; David Harwath; Eunsol Choi
>
> **备注:** EMNLP 2025
>
> **摘要:** We introduce Paralinguistic Speech Captions (ParaSpeechCaps), a large-scale dataset that annotates speech utterances with rich style captions. While rich abstract tags (e.g. guttural, nasal, pained) have been explored in small-scale human-annotated datasets, existing large-scale datasets only cover basic tags (e.g. low-pitched, slow, loud). We combine off-the-shelf text and speech embedders, classifiers and an audio language model to automatically scale rich tag annotations for the first time. ParaSpeechCaps covers a total of 59 style tags, including both speaker-level intrinsic tags and utterance-level situational tags. It consists of 342 hours of human-labelled data (PSC-Base) and 2427 hours of automatically annotated data (PSC-Scaled). We finetune Parler-TTS, an open-source style-prompted TTS model, on ParaSpeechCaps, and achieve improved style consistency (+7.9% Consistency MOS) and speech quality (+15.5% Naturalness MOS) over the best performing baseline that combines existing rich style tag datasets. We ablate several of our dataset design choices to lay the foundation for future work in this space. Our dataset, models and code are released at https://github.com/ajd12342/paraspeechcaps .
>
---
#### [replaced 085] Thinking Outside the (Gray) Box: A Context-Based Score for Assessing Value and Originality in Neural Text Generation
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13207v3](http://arxiv.org/pdf/2502.13207v3)**

> **作者:** Giorgio Franceschelli; Mirco Musolesi
>
> **摘要:** Despite the increasing use of large language models for creative tasks, their outputs often lack diversity. Common solutions, such as sampling at higher temperatures, can compromise the quality of the results. Dealing with this trade-off is still an open challenge in designing AI systems for creativity. Drawing on information theory, we propose a context-based score to quantitatively evaluate value and originality. This score incentivizes accuracy and adherence to the request while fostering divergence from the learned distribution. We show that our score can be used as a reward in a reinforcement learning framework to fine-tune large language models for maximum performance. We validate our strategy through experiments considering a variety of creative tasks, such as poetry generation and math problem solving, demonstrating that it enhances the value and originality of the generated solutions.
>
---
#### [replaced 086] Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.03136v2](http://arxiv.org/pdf/2506.03136v2)**

> **作者:** Yinjie Wang; Ling Yang; Ye Tian; Ke Shen; Mengdi Wang
>
> **备注:** NeurIPS 2025 Spotlight. Project: https://github.com/Gen-Verse/CURE
>
> **摘要:** We propose CURE, a novel reinforcement learning framework with a dedicated reward design that co-evolves coding and unit test generation capabilities based on their interaction outcomes, without any ground-truth code as supervision. This approach enables flexible and scalable training and allows the unit tester to learn directly from the coder's mistakes. Our derived ReasonFlux-Coder-7B and 14B models improve code generation accuracy by 5.3% and Best-of-N accuracy by 9.0% after optimization on Qwen2.5-Instruct models, outperforming similarly sized Qwen-Coder, DeepSeek-Coder, and Seed-Coder. They naturally extend to downstream tasks such as test-time scaling and agentic coding-achieving a 8.1% improvement over the base model. For the long-CoT model, our ReasonFlux-Coder-4B consistently outperforms Qwen3-4B while achieving 64.8% inference efficiency in unit test generation. Notably, we also find that our model can serve as an effective reward model for reinforcement learning on base models. Project: https://github.com/Gen-Verse/CURE
>
---
#### [replaced 087] Explainable Sentiment Analysis with DeepSeek-R1: Performance, Efficiency, and Few-Shot Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.11655v4](http://arxiv.org/pdf/2503.11655v4)**

> **作者:** Donghao Huang; Zhaoxia Wang
>
> **备注:** 10 pages, with 2 figures and 6 tables, accepted for publication in an IEEE Intelligent Systems journal
>
> **摘要:** Large language models (LLMs) have transformed sentiment analysis, yet balancing accuracy, efficiency, and explainability remains a critical challenge. This study presents the first comprehensive evaluation of DeepSeek-R1--an open-source reasoning model--against OpenAI's GPT-4o and GPT-4o-mini. We test the full 671B model and its distilled variants, systematically documenting few-shot learning curves. Our experiments show DeepSeek-R1 achieves a 91.39\% F1 score on 5-class sentiment and 99.31\% accuracy on binary tasks with just 5 shots, an eightfold improvement in few-shot efficiency over GPT-4o. Architecture-specific distillation effects emerge, where a 32B Qwen2.5-based model outperforms the 70B Llama-based variant by 6.69 percentage points. While its reasoning process reduces throughput, DeepSeek-R1 offers superior explainability via transparent, step-by-step traces, establishing it as a powerful, interpretable open-source alternative.
>
---
#### [replaced 088] Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST
- **分类: cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.14128v2](http://arxiv.org/pdf/2509.14128v2)**

> **作者:** Monica Sekoyan; Nithin Rao Koluguri; Nune Tadevosyan; Piotr Zelasko; Travis Bartley; Nikolay Karpov; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Mini Version of it Submitted to ICASSP 2026
>
> **摘要:** This report introduces Canary-1B-v2, a fast, robust multilingual model for Automatic Speech Recognition (ASR) and Speech-to-Text Translation (AST). Built with a FastConformer encoder and Transformer decoder, it supports 25 languages primarily European. The model was trained on 1.7M hours of total data samples, including Granary and NeMo ASR Set 3.0, with non-speech audio added to reduce hallucinations for ASR and AST. We describe its two-stage pre-training and fine-tuning process with dynamic data balancing, as well as experiments with an nGPT encoder. Results show nGPT scales well with massive data, while FastConformer excels after fine-tuning. For timestamps, Canary-1B-v2 uses the NeMo Forced Aligner (NFA) with an auxiliary CTC model, providing reliable segment-level timestamps for ASR and AST. Evaluations show Canary-1B-v2 outperforms Whisper-large-v3 on English ASR while being 10x faster, and delivers competitive multilingual ASR and AST performance against larger models like Seamless-M4T-v2-large and LLM-based systems. We also release Parakeet-TDT-0.6B-v3, a successor to v2, offering multilingual ASR across the same 25 languages with just 600M parameters.
>
---
