# 自然语言处理 cs.CL

- **最新发布 80 篇**

- **更新 66 篇**

## 最新发布

#### [new 001] PSD: Pushing the Pareto Frontier of Diffusion LLMs via Parallel Speculative Decoding
- **分类: cs.CL**

- **简介: 该论文提出PSD框架，用于提升扩散大语言模型的推理效率与生成质量。针对推理成本高的问题，通过并行推测解码实现空间与时间优化。**

- **链接: [https://arxiv.org/pdf/2605.15609](https://arxiv.org/pdf/2605.15609)**

> **作者:** Shengyin Sun; Yiming Li; Renxi Liu; Xinqi Li; Hui-Ling Zhen; Weizhe Lin; Chen Chen; Xianzhi Yu; Mingxuan Yuan; Chen Ma
>
> **备注:** 16 pages
>
> **摘要:** Diffusion large language models (dLLMs) generate text by iteratively denoising masked token sequences. Although dLLMs can predict all masked positions in parallel within each step, the large number of denoising iterations still makes inference expensive. This cost can be reduced spatially by unmasking multiple tokens per step, or temporally by collapsing multiple denoising steps into one verification call. We propose Parallel Speculative Decoding (PSD), a training-free framework that jointly improves inference along both axes. Using the confidence scores from a single forward pass, PSD selects positions to unmask via a configurable, adaptive unmasking policy and constructs multi-depth speculative drafts without extra model calls. A final batched verification pass then applies hierarchical acceptance, keeping the deepest draft that remains consistent with the updated predictions. Experiments on three dLLMs across reasoning and code generation tasks show that PSD achieves favorable trade-offs between inference efficiency and generation quality, reaching up to $5.5\times$ tokens per forward pass with accuracy comparable to greedy decoding.
>
---
#### [new 002] Why are language models less surprised than humans? Testing the Parse Multiplicity Mismatch Hypothesis
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨语言模型与人类在句法歧义处理上的差异。研究测试了解析多样性不匹配假设，通过调整同时解析数量，发现无法完全解释人类处理难度。**

- **链接: [https://arxiv.org/pdf/2605.15440](https://arxiv.org/pdf/2605.15440)**

> **作者:** William Timkey; Brian Dillon; Tal Linzen
>
> **摘要:** Surprisal theory posits that the processing difficulty of a word is determined by its predictability in context, offering a potential link between human sentence processing and next-word predictions from language models. While language model (LM) surprisals successfully predict reading times in naturalistic text, they systematically underpredict the magnitude of difficulty observed in controlled studies of syntactic ambiguity, particularly in garden path sentences. This mismatch might arise from differences in the computational constraints between humans and LMs. Here we test one such hypothesis, specifically, that LMs may be able to simultaneously consider a greater number of distinct sentence interpretations at once, compared to humans. Using Recurrent Neural Network Grammars (RNNGs) with word-synchronous beam search, we systematically vary the number of simultaneous parses used to compute word surprisal, and then use these surprisals to predict human reading times. Reducing the number of simultaneous active parses indeed increases the magnitude of predicted garden path effects, but not nearly enough to capture the full magnitude of the effects in humans. This suggests that differences in the number of simultaneous parses available to LMs and humans cannot reconcile LM-based surprisal with human sentence processing.
>
---
#### [new 003] Judge Circuits
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究LLM作为评判者的格式不一致性问题，通过PEAP方法分析模型内部机制，发现判断信号与输出格式可解耦，揭示了基准比较中格式几何的影响。**

- **链接: [https://arxiv.org/pdf/2605.16023](https://arxiv.org/pdf/2605.16023)**

> **作者:** Nils Feldhus; Tanja Baeumel; Elena Golimblevskaia; Qianli Wang; Van Bach Nguyen; Aaron Louis Eidt; Christopher Ebert; Wojciech Samek; Jing Yang; Vera Schmitt; Sebastian Möller; Simon Ostermann
>
> **备注:** 32 pages
>
> **摘要:** LLM-as-a-judge has become the dominant paradigm for grading model outputs at scale, yet the same model assigns systematically different scores when its output format changes (e.g., a 1-5 rating vs. a True/False label). Existing diagnoses of these format-induced inconsistencies stop at the input-output level. Using Position-aware Edge Attribution Patching (PEAP), we causally investigate the internal mechanism in Gemma-3, Qwen2.5, and Llama-3. We find that judgments across structured understanding and open-ended preference tasks share a sparse, generalized Latent Evaluator sub-graph in the mid-to-late multi-layer perceptrons (MLPs); zero-ablating it collapses judgment while preserving world knowledge in architecturally modular models. By structurally decoupling abstract judging from output formatting, we provide a mechanistic account of format-induced inconsistency on the open-weight models we study: a continuous judgment signal computed in the shared trunk is mapped through fragile, format-specific terminal branches, enabling format-independent preference to be isolated downstream of the requested output format. Our findings imply that benchmark-level reliability comparisons across formats are partially measuring formatter geometry rather than evaluation quality.
>
---
#### [new 004] Optimized Three-Dimensional Photovoltaic Structures with LLM guided Tree Search
- **分类: cs.CL; cond-mat.other; physics.comp-ph**

- **简介: 该论文属于光伏结构优化任务，旨在解决传统太阳能板效率低的问题。通过AI与树搜索结合，生成高效三维光伏结构，提升日间发电性能。**

- **链接: [https://arxiv.org/pdf/2605.16191](https://arxiv.org/pdf/2605.16191)**

> **作者:** Michael P. Brenner; Lizzie Dorfman; John C. Platt
>
> **备注:** 10 pages 7 figures
>
> **摘要:** We present a case study for how AI coding systems can be used to generate novel scientific hypotheses. We combine a generic coding agent (Google's AntiGravity) with an LLM-driven tree search algorithm (Empirical Research Assistance / ERA) to autonomously generate high-efficiency three-dimensional photovoltaic (3DPV) structures that overcome losses limiting flat solar panels at mid-latitudes. These structures operate by presenting favorable angles to the sun throughout the day, and for illustrative purposes we focus on optimizing performance for a single solar day. Our workflow begins by using AntiGravity to reproduce calculations \cite{bernardi2012solar} showing that 3DPV can have energy densities much higher than stationary flat PV panels. We use these initial designs as the starting point for large scale tree search, where we seek improved solutions and score them for their diurnal yield. The initial tree search leads to nominally more efficient solutions, yet they are caused by algorithmic reward hacking, arising from non-physical design features such as structurally levitating disconnected tiers and exploitations of the discretizations in the optics solver. To counteract this, we develop a workflow where the coding agent iteratively patches the physics engine with constraints to eliminate reward hacking. With reward-hacking eliminated, ERA discovers a series of designs with various constraints and improved performance, including optimal designs with different fixed collector areas, optimizing zenith tracking and avoiding self shadowing. Combining coding agents with tree search (ERA) provides a powerful platform for scientific discovery, for problems whose solutions can be empirically evaluated with a score function.
>
---
#### [new 005] Response-Conditioned Parallel-to-Sequential Orchestration for Multi-Agent Systems
- **分类: cs.CL; cs.LG; cs.MA**

- **简介: 该论文属于多智能体系统协作任务，旨在解决通信与延迟平衡问题。提出Nexa框架，结合并行与串行模式，通过响应条件策略优化通信图，提升响应准确性与效率。**

- **链接: [https://arxiv.org/pdf/2605.15573](https://arxiv.org/pdf/2605.15573)**

> **作者:** Nurbek Tastan; Alex Iacob; Lorenzo Sani; Meghdad Kurmanji; Nicholas D. Lane; Samuel Horvath; Karthik Nandakumar
>
> **摘要:** Multi-agent systems can solve complex tasks through collaboration between multiple Large Language Model agents. Existing collaboration frameworks typically operate in either a parallel or a sequential mode. In the parallel mode, agents respond independently to queries followed by aggregation of responses. In contrast, sequential systems allow agents to communicate via a directed topology and refine one another step by step. However, both modes are inadequate for achieving the desired objectives of minimizing communication and latency while simultaneously maximizing the accuracy of the final response. In this work, we introduce a hybrid paradigm called Nexa, a trainable response-conditioned policy that bridges the gap between the two modes. Nexa begins with a parallel execution stage, embeds the resulting responses into a shared semantic space, and then predicts a sparse directed acyclic communication graph. If the graph is empty, the system remains purely parallel; if it is non-empty, the system performs one sequential message propagation. The policy is a lightweight transformer model, and the method avoids the need for external LLM judges or reward models, as well as hand-crafted test-time topology search. We formalize this hybrid execution problem, show that the resulting graph is acyclic by construction, and that the framework strictly subsumes pure parallel execution, and present a training procedure based on policy-gradient optimization. Results demonstrate that the response-conditioned policy learned by Nexa under one setting can be reused when the number of agents, the task, or the underlying agent changes, thus emphasizing the generalizability of the learned communication policy.
>
---
#### [new 006] Artificial Aphasias in Lesioned Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型功能分析任务，旨在通过模拟失语症研究语言模型的内部组织。工作包括对模型参数进行“损伤”实验，并与人类失语症状对比，揭示模型与人类语言处理的差异。**

- **链接: [https://arxiv.org/pdf/2605.16222](https://arxiv.org/pdf/2605.16222)**

> **作者:** Nathan Roll; Jill Kries; Laura Gwilliams; Cory Shain
>
> **备注:** 49 pages, 13 figures
>
> **摘要:** Aphasias, selective language impairments which can arise from brain damage, reveal the functional organization of human language by providing causal links between affected brain regions and specific symptom profiles. Drawing on this literature, we introduce an aphasia-inspired technique to characterize the emergent functional organization of language models (LMs). We ``lesion'' (zero-out) model parameters and measure the effects of this intervention against clinical aphasia symptoms, as diagnosed by the Text Aphasia Battery (TAB). When applied to 112,426 outputs from five 1B-scale LMs, the full range of evaluated symptoms surface, but in distributions largely distinct from those of humans. Our method uncovers broad symptom-profile differences between attention components (query, key, value, output) and feed-forward components (up, gate, down), with weaker evidence for differences among components within the same mechanism. We also find an effect of depth, where lesions in early layers disproportionately cause syntactic and semantic symptoms while late-middle layers yield higher rates of phonological and fluency deficits. Although some LM lesions induce quantitatively more similar profiles to some human aphasia types than others, qualitative differences in symptom patterns between LMs and humans suggest that aphasia syndromes are heavily influenced by the details of learning and processing rather than being a domain-invariant consequence of disrupted language processing.
>
---
#### [new 007] ASRU: Activation Steering Meets Reinforcement Unlearning for Multimodal Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器去学习任务，旨在解决多模态大语言模型在去学习过程中出现的生成质量下降问题。提出ASRU框架，通过激活引导和奖励优化，提升去学习效果与生成质量。**

- **链接: [https://arxiv.org/pdf/2605.15687](https://arxiv.org/pdf/2605.15687)**

> **作者:** Jiahui Guang; Yingjie Zhu; Cuiyun Gao; Haiyan Wang; Jing Li; Di Shao; Zhaoquan Gu
>
> **摘要:** Multimodal large language models (MLLMs) may memorize sensitive cross-modal information during pretraining, making machine unlearning (MU) crucial. Existing methods typically evaluate unlearning effectiveness based on output deviations, while overlooking the generation quality after unlearning. This can easily lead to hallucinated or rigid responses, thereby affecting the usability and safety of the unlearned model. To address this issue, we propose ASRU, a controllable multimodal unlearning framework that incorporates generation quality as a core evaluation objective. ASRU first induces initial refusal behavior through activation redirection, and then optimizes fine-grained refusal boundaries using a customized reward function, thereby achieving a better trade-off between target knowledge unlearning and model utility. Experiments on Qwen3-VL show that ASRU significantly improves unlearning effectiveness (+24.6%) on average and generation quality (5.8x) on average while effectively preserving model utility, using only a small amount of retained supervision data.
>
---
#### [new 008] A Unified Generative-AI Framework for Smart Energy Infrastructure: Intelligent Gas Distribution, Utility Billing, Carbon Analytics, and Quantum-Inspired Optimisation
- **分类: cs.CL; cs.AI; cs.ET; cs.LG; eess.SY**

- **简介: 该论文提出一种统一的生成式AI框架，用于智能能源基础设施，解决气体分配、费用计算、碳分析和量子启发优化等问题。**

- **链接: [https://arxiv.org/pdf/2605.16232](https://arxiv.org/pdf/2605.16232)**

> **作者:** Pavan Manjunath; Thomas pruefer
>
> **摘要:** The accelerating convergence of smart metering, generative artificial intelligence, and quantum-inspired combinatorial optimisation is reshaping how energy utilities manage physical infrastructure, customer engagement, and environmental accountability
>
---
#### [new 009] RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究Transformer模型中RoPE位置编码在长上下文中的局限性。工作揭示了RoPE无法有效区分位置和标记，导致注意力机制失效。**

- **链接: [https://arxiv.org/pdf/2605.15514](https://arxiv.org/pdf/2605.15514)**

> **作者:** Yufeng Du; Phillip Harris; Minyang Tian; Eliu A Huerta; Srikanth Ronanki; Subendhu Rongali; Aram Galstyan; Hao Peng
>
> **备注:** 35 pages, 11 figures, submitted to NeurIPS 2026
>
> **摘要:** We identify intrinsic limitations of Rotary Positional Embeddings (RoPE) in Transformer-based long-context language models. Our theoretical analysis abstracts away from the specific content of the context and depends only on its length. We prove that as context length increases, RoPE-based attention becomes unpredictable and loses two properties that are central to its effectiveness. First, it loses its locality bias: RoPE is no more likely to favor nearer positions than substantially farther ones. Second, it loses consistency in token relevance: a key vector that receives a higher attention score than an alternative at one position may receive a lower score at another. In both cases, the probability of failure approaches 0.5, no better than random guessing. We further prove that the attention score can remain unchanged when a key token is moved to a different position, or even replaced by a different token, indicating a failure to distinguish positions or tokens. Adjusting the RoPE base trades off distinguishing positions against distinguishing tokens but cannot preserve both at the same time. Increasing the RoPE base hyperparameter, a common practice in today's long-context models, helps distinguish different tokens, but inevitably sacrifices the ability to distinguish positions. Our empirical analysis shows that multi-head, multi-layer architectures are insufficient to overcome these limitations. Our findings suggest that fundamentally new mechanisms for encoding position and token order may be needed in future Transformer long-context language models.
>
---
#### [new 010] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的知识图谱任务，旨在评估大语言模型在心理健康领域的知识掌握情况。通过构建基准测试MHGraphBench，分析模型在实体识别、关系判断和两跳推理上的表现。**

- **链接: [https://arxiv.org/pdf/2605.15589](https://arxiv.org/pdf/2605.15589)**

> **作者:** Weixin Liu; Congning Ni; Shelagh A. Mulvaney; Susannah L. Rose; Murat Kantarcioglu; Bradley A. Malin; Zhijun Yin
>
> **备注:** Accepted to GEM 2026, ACL 2026 Workshop; 9 pages main text plus references and appendices
>
> **摘要:** Large language models (LLMs) are increasingly used in the mental health domain, yet it remains unclear how well they capture related biomedical knowledge and how reliably they apply it to clinically salient structured judgments. Here, we present a knowledge-graph (KG)-grounded benchmark for assessing LLMs on mental-health entity recognition, relation judgment, and two-hop reasoning. The benchmark is derived from PrimeKG and comprises nine task families with KG-supported answers and controlled negative options. Experiments across 15 closed- and open-source LLMs reveal a persistent recognition-to-judgment gap: leading models achieve near-ceiling performance on entity typing and on the small relation-typing subset, yet they still struggle with relation prediction and two-hop reasoning. Additionally, short KG-derived snippets benefit some models but degrade performance for others. Moreover, output-format reliability can substantially influence measured performance under constrained multiple-choice settings, highlighting the critical role of response validity in benchmark-based evaluation. MHGraphBench should therefore be interpreted as evaluating agreement with a curated mental-health slice of PrimeKG under a constrained multiple-choice interface, rather than as a direct assessment of real-world clinical safety.
>
---
#### [new 011] ForMaT: Dataset for Visually-Grounded Multilingual PDF Translation
- **分类: cs.CL**

- **简介: 该论文提出ForMaT数据集，用于多模态机器翻译任务。旨在解决文本与视觉上下文关联丢失的问题，通过保留布局信息并采用K-Medoids采样确保结构多样性。**

- **链接: [https://arxiv.org/pdf/2605.15794](https://arxiv.org/pdf/2605.15794)**

> **作者:** Michał Ciesiółka; Dawid Wiśniewski; Adrian Charkiewicz; Kamil Guttmann
>
> **摘要:** We present ForMaT (Format-Preserving Multilingual Translation), a parallel corpus of 3,956 PDFs across 15 language pairs that preserves original layout metadata proposed for multimodal machine translation. To ensure structural diversity in the dataset, we employ K-Medoids sampling over 45 geometric features, capturing complex elements like nested tables and formulas to focus only on visually diverse PDF documents. Our evaluation reveals that current MT systems struggle with spatial grounding and geometric synchronization, often losing the link between text and its visual context. ForMaT provides a benchmark for developing layout-aware translation models that integrate visual and textual context for high-fidelity document reconstruction.
>
---
#### [new 012] Retrieval-Augmented Large Language Models for Schema-Constrained Clinical Information Extraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床信息提取任务，旨在从护理人员与患者对话中提取结构化数据。针对文档负担重的问题，提出一种检索增强生成方法，提升提取准确性。**

- **链接: [https://arxiv.org/pdf/2605.15467](https://arxiv.org/pdf/2605.15467)**

> **作者:** A H M Rezaul Karim; Ozlem Uzuner
>
> **摘要:** Conversational nurse-patient transcripts contain actionable observations, but converting these transcripts into structured representations at scale remains challenging. Documentation burden is substantial, with prior studies showing clinicians spend large portions of their workday on documentation and related desk work rather than direct patient care. MEDIQA-SYNUR focuses on observation extraction from conversational nurse-patient transcripts, requiring systems to normalize these narratives into a predefined schema with value-type constraints. We propose a modular retrieval-augmented generation (RAG) pipeline that uses the training set as an exemplar corpus, combines schema-constrained prompting (full schema vs. pruned candidate schema), deterministic schema-based postprocessing, and a second-pass audit, with two LLM backbones: Llama-4-Scout-17B-16E-Instruct and GPT-5.2 with corresponding embedding models for RAG. Our best configuration uses GPT-5.2 with full schema, RAG, and a second-pass auditing, achieving 80.36% F1 score. Overall, our results show that RAG consistently improves performance, while the optimal degree of schema constraint depends on the model, and second-pass auditing yields modest additional gains by correcting residual schema-adherence errors.
>
---
#### [new 013] Adesua: Development and Feasibility Study of an AI WhatsApp Bot for Science Learning in West Africa
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于教育技术任务，旨在解决西非学生缺乏个性化学习支持的问题。通过开发基于WhatsApp的AI教学助手Adesua，提供科学学习支持与自动评估。**

- **链接: [https://arxiv.org/pdf/2605.15376](https://arxiv.org/pdf/2605.15376)**

> **作者:** George Boateng; Evans Atompoya; Philemon Badu; Samuel John; Samuel Ansah; Patrick Agyeman-Budu; Victor Wumbor-Apin Kumbol
>
> **备注:** 11 pages. Accepted at the 27th International Conference on Artificial Intelligence in Education (AIED 2026)
>
> **摘要:** Sub-Saharan Africa faces persistently high student-teacher ratios and shortages of qualified teachers, limiting students' access to personalized learning support and formative assessment. To address this challenge, we present Adesua, a WhatsApp-based AI Teaching Assistant for science education that extends the Kwame for Science platform. Adesua leverages WhatsApp's widespread adoption in Africa to provide accessible, curriculum-aligned learning support for Junior High School (JHS) and Senior High School (SHS) students across West Africa. The system integrates curated textbooks and 33 years of national examination questions with generative AI to enable conversational question answering and automated assessment with feedback via a WhatsApp bot. Students can ask science questions, take timed or untimed multiple-choice tests by topic or exam year, and receive instant grading and detailed explanations of correct and incorrect responses. A 6-month feasibility deployment in 2025 had 56 active users in Ghana, including students and parents. Quantitative evaluation showed a high perceived usefulness, with a helpfulness score of 93.75\% for AI-generated answers, albeit with a small number of ratings (n=16). These preliminary results provide a basis for more extensive future evaluation of a WhatsApp-based AI assistant to assess its potential to offer scalable, low-cost personalized learning support and formative assessment in resource-constrained educational contexts.
>
---
#### [new 014] SMMBench: A Benchmark for Source-Distributed Multimodal Agent Memory
- **分类: cs.CL**

- **简介: 该论文提出SMMBench，用于评估多模态代理在跨源记忆组合方面的能力，解决现有基准未充分考察多源证据整合的问题。**

- **链接: [https://arxiv.org/pdf/2605.15710](https://arxiv.org/pdf/2605.15710)**

> **作者:** Huacan Chai; Yukai Wang; Yingxuan Yang; Dan Peng; Yuanyi Song; Zhihui Fu; Weiwen Liu; Jianghao Lin; Jun Wang; Weinan Zhang
>
> **摘要:** Existing benchmarks for multimodal memory reasoning largely evaluate systems within pre-assembled contexts, but under-evaluate whether agents can use evidence distributed across independently originated sources. We argue that source-distributed memory composition is an important and under-examined bottleneck in multimodal agent memory, especially when relevant evidence is fragmented across heterogeneous artifacts such as conversations, profiles, screenshots, tables, images, and documents. To address this gap, we introduce Source-distributed Multimodal Memory Benchmark(SMMBench), which measures whether agents can retrieve, align, and compose multimodal evidence scattered across multiple sources rather than reason within a single curated context. SMMBench evaluates four core capabilities: (1) cross-source multimodal reasoning; (2) conflict resolution; (3) preference reasoning; (4) memory-grounded action prediction. The benchmark contains 1877 samples grounded in 264 sources. Experiments on representative memory-style and retrieval-based baselines show that current systems still struggle on these capabilities, positioning source-distributed multimodal memory as an important and still under-evaluated challenge for multimodal agents. Our data are available at this https URL.
>
---
#### [new 015] Reference-Free Reinforcement Learning Fine-Tuning for MT: A Seq2Seq Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译任务，解决如何在无并行数据情况下提升序列到序列模型的性能。通过引入无参考强化学习方法，提升多语言翻译效果。**

- **链接: [https://arxiv.org/pdf/2605.15976](https://arxiv.org/pdf/2605.15976)**

> **作者:** Ernesto Garcia-Estrada; Carlos Escolano; José A. R. Fonallosa
>
> **摘要:** Production machine translation relies overwhelmingly on encoder-decoder Seq2Seq models, yet reinforcement learning approaches to MT fine-tuning have largely targeted decoder-only LLMs at $\geq$7B parameters, with limited systematic study of encoder-decoder architectures. We apply Group Relative Policy Optimization to NLLB-200 (600M and 1.3B) using a hybrid reference-free reward (LaBSE and COMET-Kiwi) that requires no parallel data at fine-tuning time, evaluating across 13 typologically diverse languages. GRPO yields consistent improvements on all 13 languages, up to $+$5.03 chrF++ for Traditional Chinese, and, without any target-language data, competes with 3-epoch supervised fine-tuning on morphologically complex languages . We identify a consistent empirical pattern in which gains are largest where baseline performance is weakest and reward discriminability is highest, making this approach most effective precisely where parallel data is scarcest, and replicate this pattern across English and Spanish source languages.
>
---
#### [new 016] Dynamic Chunking for Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文提出DCDM模型，解决序列建模中的结构浪费问题。通过内容定义的语义块替代固定位置块，提升扩散语言模型性能。**

- **链接: [https://arxiv.org/pdf/2605.15676](https://arxiv.org/pdf/2605.15676)**

> **作者:** Yichen Zhu; Xiaoming Shi; Peng Zhao; Weiyu Chen; Debing Zhang; James Kwok
>
> **摘要:** Block discrete diffusion language models factorize a sequence autoregressively over fixed-size positional blocks, decoupling within-block parallel denoising from across-block conditioning. We argue that this rigid partition wastes structure already present in the sequence: blocks defined by position rather than by content separate semantically coherent tokens and group unrelated ones together. We introduce the \textbf{D}ynamic \textbf{C}hunking \textbf{D}iffusion \textbf{M}odel (DCDM), which replaces positional blocks with content-defined semantic chunks. At its core is Chunking Attention, a differentiable layer that routes tokens into $K$ clusters parameterized by learnable subspaces and shaped end-to-end by the diffusion objective. The resulting cluster assignments induce a chunk-causal attention mask under which a discrete diffusion denoiser factorizes the sequence likelihood autoregressively over semantic chunks, strictly generalizing block discrete diffusion. On downstream benchmarks at parameter scales up to 1.5B, DCDM consistently improves over both unstructured and positional-block diffusion baselines, with the advantage stable across scales and visible early in training.
>
---
#### [new 017] Linked Multi-Model Data on Russian Domestic and Foreign Policy Speeches
- **分类: cs.CL**

- **简介: 该论文属于政治通信分析任务，旨在解决 authoritarian 政治中多模态数据不足的问题。构建了包含俄语与英语演讲、图像及标注的链接数据集，支持多模态和跨语言研究。**

- **链接: [https://arxiv.org/pdf/2605.15886](https://arxiv.org/pdf/2605.15886)**

> **作者:** Daria Blinova; Gayathri Emuru; Rakesh Emuru; Kushagradheer Shridheer Srivastava; Mina Rulis; Sunita Chandrasekaran; Benjamin E. Bagozzi
>
> **摘要:** This paper introduces a dataset of interlinked multimodal political communications from the Russian government, addressing persistent deficiencies in the availability of social text- and image-based data for authoritarian politics contexts. The dataset comprises two large corpora of official speeches delivered by senior actors within the Kremlin and the Russian Ministry of Foreign Affairs over multiple decades. For each speech, we provide Russian- and English-language texts, associated images and captions where available, and harmonized metadata including (e.g.) dates, speakers, (geo)locations, and official government content tags. Unique identifiers link images to speeches and align Russian and English versions of the same communication texts. We further augment these linked datasets with validated topical annotations for both speech texts and speech images, which are generated via transformer-based multimodal topic modeling and refined by a Russian politics expert. The resulting data resources support multimodal, multilingual, temporal, and/or spatial analyses of (authoritarian) political communication and offer a valuable testbed for social science research and large language model (LLM) applications in political domains.
>
---
#### [new 018] VCG-Bench: Towards A Unified Visual-Centric Benchmark for Structured Generation and Editing
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出VCG-Bench，解决视觉-代码结构化生成与编辑任务中的精度和可控性问题，通过符号逻辑和mxGraph实现高效 diagram 生成与评估。**

- **链接: [https://arxiv.org/pdf/2605.15677](https://arxiv.org/pdf/2605.15677)**

> **作者:** Xiaoyan Su; Peijie Dong; Zhenheng Tang; Song Tang; Yuyao Zhai; Kaitao Lin; Liang Chen; Gai Yuhang; Yuyu Luo; Qiang Wang; Xiaowen Chu
>
> **备注:** Accepted by ICML2026, 37 pages, 10 figures
>
> **摘要:** Despite the rapid advancements in Vision-Language Models (VLMs), a critical gap remains in their ability to handle structured, controllable diagrammatic tasks essential for professional workflows. Existing methods predominantly rely on pixel-based synthesis, which operates in probabilistic pixel spaces and is inherently limited in editability and fidelity. Instead, we propose a new Diagram-as-Code paradigm with symbolic logic that leverages mxGraph Extensible Markup Language (XML) for precise diagram generation and editing. We present VCG-Bench, a unified benchmark for visual-centric \texttt{mxGraph} tasks. VCG-Bench comprises: (1) a taxonomized dataset of 1,449 diverse diagrams spanning 6 domains and 15 sub-domains, (2) a paradigm definition that integrates Generation (Vision-to-Code) and Editability (Code-to-Code), (3) a Tailored Evaluation Protocol employing multi-dimensional metrics such as \texttt{mxGraph} Execution Success Rate, Style Consistency Score (SCS), etc. Experimental results highlight the challenges faced by current State-of-the-Art (SOTA) VLMs in structured fidelity and instruction compliance, reflecting their vision and reasoning capabilities.
>
---
#### [new 019] Can Large Language Models Imitate Human Speech for Clinical Assessment? LLM-Driven Data Augmentation for Cognitive Score Prediction
- **分类: cs.CL**

- **简介: 该论文属于临床语音分析任务，旨在解决数据量小和类别不平衡问题。通过LLM生成语义相似的合成语音数据，提升认知评分预测效果。**

- **链接: [https://arxiv.org/pdf/2605.16077](https://arxiv.org/pdf/2605.16077)**

> **作者:** Si-Belkacem Yamine Ketir; Lenard Paulo Tamayo; Shohei Hisada; Shaowen Peng; Shoko Wakamiya; Eiji Aramaki
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Accurate assessment of cognitive decline from spontaneous speech remains challenging due to limited dataset size and class imbalance. In this work, we propose a large language model (LLM)-driven data augmentation framework to improve the prediction of cognitive scores from speech. Experiments are conducted on a Japanese corpus in which each participant provides both a spontaneous oral narrative and a written response to the same clinical prompt. The written responses serve as semantic anchors to generate multiple oral-like monologues in different styles using GPT-5. We then predict Hasegawa Dementia Scale scores, a widely used cognitive screening tool in Japan, using a Partial Least Squares regression model trained on Sentence-BERT speech embeddings. We investigate two augmentation strategies: random class-balanced selection, which yields moderate but unstable improvements, and similarity-guided class-balanced selection. The latter prioritizes semantically close synthetic samples, leading to more consistent improvements and substantially reducing prediction error for minority low-score participants while maintaining performance for the majority group. Overall, our findings demonstrate the potential of semantically guided LLM-driven augmentation as a principled approach for addressing class imbalance and improving data efficiency in clinical speech analysis.
>
---
#### [new 020] Improving Cross-Cultural Survey Simulation with Calibrated Value Personas
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理任务，旨在提升跨文化调查模拟的准确性。通过基于价值观的人设构建方法，解决LLM在不同文化中模拟人类意见的局限性。**

- **链接: [https://arxiv.org/pdf/2605.16193](https://arxiv.org/pdf/2605.16193)**

> **作者:** Axel Abels; Elias Fernandez Domingos; Apurva Shah; Tom Lenaerts
>
> **备注:** Submitted to the Fourth International Workshop on Value Engineering in AI (VALE 2026), held at IJCAI-ECAI 2026
>
> **摘要:** Large language models (LLMs) are increasingly used to simulate human opinions and survey responses, but their ability to reproduce population responses across cultures remains limited. Existing persona-based prompting methods typically rely on sociodemographic or personality traits, which are only indirect proxies for the values that shape human responses. We propose a value-based persona construction method that derives textual descriptors from survey responses capturing core cultural dimensions. By sampling value profiles from target populations and aggregating LLM responses across personas, we obtain population-level predictions grounded in observed value distributions. We further introduce a calibration procedure that improves response diversity while preserving estimated opinions. We show that our approach reduces prediction error across countries, with the largest improvements observed in underrepresented populations. This substantially narrows the performance gap between countries aligned with dominant LLM priors and those that are less represented in training data, while also yielding response distributions that closely match human diversity.
>
---
#### [new 021] Argus: Evidence Assembly for Scalable Deep Research Agents
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出Argus系统，解决深度研究中证据碎片化问题。通过Searcher与Navigator协作，高效组装证据，提升回答质量。属于信息检索任务。**

- **链接: [https://arxiv.org/pdf/2605.16217](https://arxiv.org/pdf/2605.16217)**

> **作者:** Zhen Zhang; Liangcai Su; Zhuo Chen; Xiang Lin; Haotian Xu; Simon Shaolei Du; Kaiyu Yang; Bo An; Lidong Bing; Xinyu Wang
>
> **摘要:** Deep research agents have achieved remarkable progress on complex information seeking tasks. Even long ReAct style rollouts explore only a single trajectory, while recent state of the art systems scale inference time compute via parallel search and aggregation. Yet deep research answers are composed of complementary pieces of evidence, which parallel rollouts often duplicate rather than complete, yielding diminishing returns while pushing the aggregation context toward the model's limit. We propose Argus, an agentic system in which a Searcher and a Navigator cooperate to treat deep research as assembling a jigsaw from complementary evidence pieces, rather than brute forcing the whole answer in parallel. The Searcher collects evidence traces for a given sub-query through ReAct-style interaction. The Navigator maintains a shared evidence graph, verifying which pieces are still missing, dispatching Searchers to gather them, and reasoning over the completed graph to produce a source-traced final answer. We train the Navigator with reinforcement learning to verify, dispatch, and synthesize, while independently training the Searcher to remain a standard ReAct agent. The resulting Navigator supports rollouts with a single Searcher or many in parallel without retraining. With both Searcher and Navigator built on a 35B-A3B MoE backbone, Argus gains 5.5 points with a single Searcher and 12.7 points with 8 parallel Searchers, averaged over eight benchmarks. With 64 Searchers it reaches 86.2 on BrowseComp, surpassing every proprietary agent we benchmark, while the Navigator's reasoning context stays under 21.5K tokens.
>
---
#### [new 022] Contexting as Recommendation: Evolutionary Collaborative Filtering for Context Engineering
- **分类: cs.CL**

- **简介: 该论文属于推荐任务，解决LLM上下文工程中个性化不足的问题。提出NCCE框架，通过协同过滤动态匹配输入与最佳上下文，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.15721](https://arxiv.org/pdf/2605.15721)**

> **作者:** Jiachen Zhu; Zhuoying Ou; Congmin Zheng; Yuxiang Chen; Zeyu Zheng; Rong Shan; Lingyu Yang; Lionel Z. Wang; Weiwen Liu; Yong Yu; Weinan Zhang; Jianghao Lin
>
> **摘要:** Large Language Models (LLMs) are highly sensitive to their input contexts, motivating the development of automated context engineering. However, existing methods predominantly treat this as a global search problem, seeking a single context strategy that maximizes average performance across a dataset. This restrictive assumption overlooks the fact that different inputs often require distinct guidance, leaving substantial instance-level performance gains untapped. In this paper, we propose a paradigm shift by formulating context engineering as a recommendation problem. We introduce \textbf{Neural Collaborative Context Engineering (NCCE)}, a framework that transitions optimization from a static global search to dynamic, instance-wise routing. NCCE first bootstraps a diverse catalog of anchor contexts and then employs a novel \textbf{Context-CF Co-Evolution} mechanism. This stage establishes a synergistic feedback loop: a lightweight Neural Collaborative Filtering (NCF) model learns instance-context preferences to guide the generation of specialized context variants, while the newly evaluated contexts continuously refine the NCF model's understanding of latent preferences. At inference time, the trained NCF model acts as a context router, dynamically assigning the most suitable context strategy to each unseen instance. Theoretical Proofs and comprehensive experiments demonstrate that by matching individual inputs with their optimal contexts, NCCE significantly improves task accuracy, highlighting the critical importance of personalization in LLM context engineering.
>
---
#### [new 023] H-Mem: A Novel Memory Mechanism for Evolving and Retrieving Agent Memory via a Hybrid Structure
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出H-Mem，解决LLM代理记忆演化与检索问题，通过混合结构提升问答任务性能。**

- **链接: [https://arxiv.org/pdf/2605.15701](https://arxiv.org/pdf/2605.15701)**

> **作者:** Jiawei Yu; Yixiang Fang; Xilin Liu; Yuchi Ma
>
> **摘要:** Memory data are ubiquitous in Large Language Model (LLM)-based agents (e.g., OpenClaw and Manus). A few recent works have attempted to exploit agents'memory for improving their performance on the question-answering (QA) task, but they lack a principled mechanism for effectively modeling how memory data evolves over time and retrieving memory data effectively, leading to poor performance in memory utilization. To fill this gap, we present H-Mem, a novel memory mechanism via a hybrid structure that can not only effectively model the evolution of agent memory over a long period of time, but also provide an efficient memory retrieval approach. Particularly, H-Mem builds a temporal and semantic tree structure that allows the short-term memory data to evolve progressively into long-term memory data, where the latter provides summarized information about the former, while simultaneously constructing a knowledge graph to capture the relationships between entities in memory. Moreover, it offers an effective memory retrieval approach by exploiting the hybrid structure of the tree and graph structures. Extensive experiments on three agent memory benchmarks show that H-Mem achieves state-of-the-art performance on the QA task.
>
---
#### [new 024] Syntax Without Semantics: Teaching Large Language Models to Code in an Unseen Language
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于代码生成任务，研究LLM在未见语言中的编码能力。通过引入PyLang语言，发现模型虽能理解算法却无法正确实现，揭示了“实现保真度差距”问题。**

- **链接: [https://arxiv.org/pdf/2605.15607](https://arxiv.org/pdf/2605.15607)**

> **作者:** Vinayshekhar Bannihatti Kumar; Disha Makhija; Manoj Ghuhan Arivazhagan; Rashmi Gangadharaiah
>
> **摘要:** Large language models (LLMs) achieve high pass rates on code generation benchmarks, yet whether they can transfer this ability to languages absent from pretraining remains poorly understood. We introduce PyLang, a minimal imperative language absent from all pretraining corpora, and evaluate frontier models zero-shot and fine-tuned Qwen3 (4B, 8B, 32B) on 352 problems. We find that fine-tuning quickly teaches syntax but fails to transfer semantic competence: Python outperforms PyLang by up to 19% across all configurations, and no intervention (multi-task learning, preference tuning, code infilling, or latent-space objectives) closes the gap. An LLM judge reveals that frontier models select an identical algorithm to Python 80% of the time, yet cannot translate it into a working PyLang implementation., and CKA analysis confirms that fine-tuned models converge to nearly identical internal representations across languages (CKA > 0.97) while diverging at the output stage. We term this the implementation fidelity gap: models possess language-agnostic algorithmic understanding but cannot express it in an unfamiliar language. Our findings highlight the need for training methods that decouple reasoning from language-specific realization.
>
---
#### [new 025] DimMem: Dimensional Structuring for Efficient Long-Term Agent Memory
- **分类: cs.CL**

- **简介: 该论文提出DimMem，解决LLM代理长期记忆的结构化问题。通过显式维度表示提升记忆效率与准确性，适用于对话记忆管理任务。**

- **链接: [https://arxiv.org/pdf/2605.15759](https://arxiv.org/pdf/2605.15759)**

> **作者:** Wentao Qiu; Haotian Hu; Fanyi Wang; Jinwei Kong; Yu Zhang
>
> **摘要:** Large language model (LLM) agents require long-term memory to leverage information from past interactions. However, existing memory systems often face a fidelity--efficiency trade-off: raw dialogue histories are expensive, while flat facts or summaries may discard the structure needed for precise recall. We propose \textbf{DimMem}, a lightweight dimensional memory framework that represents each memory as an atomic, typed, and self-contained unit with explicit fields such as time, location, reason, purpose, and keywords. This representation exposes the structure needed for dimension-aware retrieval, memory update, and selective assistant-context recall without storing full histories in the model context. Across LoCoMo-10 and LongMemEval-S, DimMem achieves \textbf{81.43\%} and \textbf{78.20\%} overall accuracy, respectively, outperforming existing lightweight memory systems while reducing LoCoMo per-query token cost by \textbf{24\%}. We further show that dimensional memory extraction is learnable by compact models: after fine-tuning on the DimMem schema, a Qwen3-4B extractor surpasses LightMem with GPT-4.1-mini on both benchmarks and reaches performance comparable to, or better than, much larger extractors in key settings. These results suggest that explicit dimensional structuring is an effective and efficient foundation for long-term memory in LLM agents. Code is available at this https URL.
>
---
#### [new 026] Towards Generalization of Block Attention via Automatic Segmentation and Block Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决块注意力机制在长文本中的应用问题。通过自动分块和块蒸馏方法，提升块注意力的性能与效率。**

- **链接: [https://arxiv.org/pdf/2605.15913](https://arxiv.org/pdf/2605.15913)**

> **作者:** Shuaiyi Li; Zhisong Zhang; Yan Wang; Lei Zhu; Dongyang Ma; Chenlong Deng; Yang Deng; Wai Lam
>
> **备注:** 16 pages, 2 figures
>
> **摘要:** Block attention, which processes the input as separate blocks that cannot attend to one another, offers significant potential to improve KV cache reuse in long-context scenarios such as Retrieval-Augmented Generation (RAG). However, its broader application is hindered by two key challenges: the difficulty of segmenting input text into meaningful, self-contained blocks, and the inefficiency of existing block fine-tuning methods that risk degrading performance. To address these, we first construct SemanticSeg, a large and diverse semantic segmentation dataset containing over 30k instances across 16 categories-including books, code, web text, and conversations with text lengths ranging from 2k to 32k. Using this dataset, we train a lightweight segmenter to automatically partition text into human-instinct-aligned blocks with controllable granularity. Second, we propose block distillation, a training framework that is more efficient than block fine-tuning, which uses a frozen full-attention teacher model to guide the block-attention student. This framework integrates three novel components: block sink tokens to mitigate information loss at block boundaries, block dropout to leverage training signals from all blocks, and token-level loss weighting to focus learning on block-attention-sensitive tokens. Experiments across multiple models and benchmarks demonstrate that our segmenter outperforms heuristic and statistical baselines, and block distillation achieves near-full-attention performance under block attention, establishing a practical and scalable pathway for deploying block attention.
>
---
#### [new 027] Process Rewards with Learned Reliability
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出BetaPRM，解决PRM可靠性不足的问题，通过分布预测提升步骤奖励可信度，应用于ACA优化计算分配，提高推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.15529](https://arxiv.org/pdf/2605.15529)**

> **作者:** Jinyuan Li; Langlin Huang; Chengsong Huang; Shaoyang Xu; Donghong Cai; Yuyi Yang; Wenxuan Zhang; Jiaxin Huang
>
> **摘要:** Process Reward Models (PRMs) provide step-level feedback for reasoning, but current PRMs usually output only a single reward score for each step. Downstream methods must therefore treat imperfect step-level reward predictions as reliable decision signals, with no indication of when these predictions should be trusted. We propose BetaPRM, a distributional PRM that predicts both a step-level success probability and the reliability of that prediction. Given step-success supervision from Monte Carlo continuations, BetaPRM learns a Beta belief that explains the observed number of successful continuations through a Beta-Binomial likelihood, rather than regressing to the finite-sample success ratio as a point target. This learned reliability signal indicates when a step reward should be trusted, enabling downstream applications to distinguish reliable rewards from uncertain ones. As one application, we introduce Adaptive Computation Allocation (ACA) for PRM-guided Best-of-N reasoning. ACA uses the learned reliability signal to stop when a high-reward solution is reliable and to spend additional computation on uncertain candidate prefixes. Experiments across four backbones and four reasoning benchmarks show that BetaPRM improves PRM-guided Best-of-N selection while preserving standard step-level error detection. Built on this signal, ACA improves the accuracy--token tradeoff over fixed-budget Best-of-16, reducing token usage by up to 33.57% while improving final-answer accuracy.
>
---
#### [new 028] Capability Conditioned Scaffolding for Professional Human LLM Collaboration
- **分类: cs.CL**

- **简介: 该论文属于人机协作任务，旨在解决专业领域评估能力差异导致的AI依赖问题。提出能力条件支架框架，按领域能力分类并调整干预行为，提升协作可靠性。**

- **链接: [https://arxiv.org/pdf/2605.15404](https://arxiv.org/pdf/2605.15404)**

> **作者:** Sen Yang; Yinglei Ma
>
> **摘要:** Large language model personalization typically adapts outputs to user preferences and style but does not account for differences in user evaluation capacity across domains of expertise. This limitation can encourage Professional Domain Drift, where users rely on AI generated reasoning in domains they cannot reliably evaluate. We introduce Capability Conditioned Scaffolding, a typed framework that partitions expertise into strong, mixed, and weak domains and conditions intervention behavior on structured capability profiles. A pilot evaluation across multiple MMLU subsets and four LLM substrates shows consistent profile conditioned intervention behavior, including categorical inversion under profile swapping and selective activation in mixed domain risk zones. These findings suggest that capability aware scaffolding can support more reliable professional human AI collaboration beyond stylistic personalization.
>
---
#### [new 029] Always Learning, Always Mixing: Efficient and Simple Data Mixing All The Time
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决语言模型训练中的数据混合问题。提出OP-Mix算法，实现高效、统一的数据混合，提升模型性能并减少计算资源消耗。**

- **链接: [https://arxiv.org/pdf/2605.15220](https://arxiv.org/pdf/2605.15220)**

> **作者:** Michael Y. Hu; Apurva Gandhi; Kyunghyun Cho; Tal Linzen; Pratyusha Sharma
>
> **摘要:** Data mixing decides how to combine different sources or types of data and is a consequential problem throughout language model training. In pretraining, data composition is a key determinant of model quality; in continual learning and adaptation, it governs what is retained and acquired. Yet existing data mixing methods address only one phase of this lifecycle at a time: some require smaller proxy models tied to a single training phase, others assume a fixed domain set, and continual learning lacks principled guidance altogether. We argue that data mixing is fundamentally an online decision making problem -- one that recurs throughout training and demands a single, unified solution. We introduce OP-Mix (On-Policy Mix), a data mixing algorithm that operates across the entire language model training lifecycle. Our main insight is that candidate data mixtures can be cheaply simulated by interpolating between low-rank adapters trained directly on the current model, eliminating separate proxy models and ensuring the search is always grounded in the model's actual learning dynamics. Across pretraining, continual midtraining, and continual instruction tuning, OP-Mix consistently finds near-optimal mixtures while using a fraction of the compute of the baselines. In pretraining, OP-Mix improves upon training without mixing by 6.3% in average perplexity. For continual learning, OP-Mix matches the performance of both retraining and on-policy distillation while using 66% and 95% less overall compute, respectively. OP-Mix suggests a different view of language model training: not a sequence of distinct phases, but a single continuous process of learning from data.
>
---
#### [new 030] From Flat Language Labels to Typological Priors: Structured Language Conditioning for Multilingual Speech-to-Speech Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言语音到语音翻译任务，旨在解决现有系统忽略语言结构的问题。提出S2ST-Omni 2框架，通过结构化语言先验提升性能。**

- **链接: [https://arxiv.org/pdf/2605.16026](https://arxiv.org/pdf/2605.16026)**

> **作者:** Yu Pan; Yang Hou; Xiongfei Wu; Liang Zhang; Yves Le Traon; Lei Ma; Jianjun Zhao
>
> **备注:** Submitted to IEEE/ACM TASLP. This work extends S2ST-Omni, accepted to Findings of ACL 2026
>
> **摘要:** Compositional speech-to-speech translation (S2ST) systems built upon speech large language models (SpeechLLMs) have recently shown promising performance. However, existing S2ST systems often either neglect source-language information or encode it through a language-as-label paradigm, representing each source language as an independent flat embedding. Such a design overlooks systematic linguistic structure shared across languages, which may limit data-efficient multilingual adaptation when supervised S2ST data are scarce. To address this issue, we propose S2ST-Omni 2, a many-to-one compositional S2ST framework that systematically reformulates multilingual language conditioning from flat language labels to structured typological priors. Specifically, S2ST-Omni 2 revisits language conditioning at three levels: typology-informed hierarchical language encoding for structured source-language representation, dynamically-gated language-aware Dual-CTC for content-adaptive acoustic modulation, and typology-aware LLM prompting for decoder-side linguistic guidance. Experiments on CVSS-C show that S2ST-Omni 2 achieves superior average performance among representative S2ST approaches across BLEU, COMET, ASR-BLEU, and BLASER 2.0 under the adopted evaluation protocol. Ablation studies indicate that the proposed representation-level, acoustic-level, and decoding-level strategies provide complementary benefits. Moreover, controlled data-budget analyses and a Japanese-to-English evaluation using only approximately 3 hours of supervised training data suggest that explicit typological priors provide useful inductive biases for data-efficient multilingual S2ST.
>
---
#### [new 031] CompactQE: Interpretable Translation Quality Estimation via Small Open-Weight LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译质量评估任务，旨在解决传统QE方法依赖大型私有模型带来的隐私和成本问题。工作使用小型开源模型实现高效、可解释的翻译质量评估。**

- **链接: [https://arxiv.org/pdf/2605.15763](https://arxiv.org/pdf/2605.15763)**

> **作者:** Kamil Guttmann; Zofia Fraś; Artur Nowakowski; Krzysztof Jassem
>
> **摘要:** Current state-of-the-art Quality Estimation (QE) in machine translation relies on massive, proprietary LLMs, raising data privacy concerns. We demonstrate that smaller, open-source LLMs (<30B parameters) are a viable, cost-effective and privacy-preserving alternative. Using a single-pass prompting strategy, our models simultaneously generate quality scores, MQM error annotations, suggested error corrections, and full post-editions. Our analysis shows these models achieve highly competitive system-level correlations with human judgments that outperform traditional neural metrics, fine-tuned models, and human inter-annotator agreement, effectively approximating the capabilities of much larger proprietary LLMs.
>
---
#### [new 032] When Latent Geometry Is Not Enough: Draft-Conditioned Latent Refinement for Non-Autoregressive Text Generation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于非自回归文本生成任务，解决连续潜在状态与离散标记接口的问题。通过构建条件潜在细化模型，提升解码可恢复性。**

- **链接: [https://arxiv.org/pdf/2605.15557](https://arxiv.org/pdf/2605.15557)**

> **作者:** De Shuai Zhang
>
> **备注:** 17 pages, 1 figure, 6 tables. Technical Report v1. Stage 1 complete; Stage 2 ongoing Code: this https URL
>
> **摘要:** Continuous diffusion and flow models are attractive for non-autoregressive text generation because they can update all positions in parallel. A major difficulty is the interface between continuous latent states and discrete tokens. This report studies a draft-conditioned latent refinement model built from a frozen BERT encoder, a parallel decoder, a denoising DraftPrior, a local FlowNet, and a learned diagonal MetricNet. Early Gaussian-start experiments showed that good latent-space metrics, such as scale matching or cosine similarity, do not guarantee good decoding. Generated latents can be close to real encoder latents but still produce high-entropy, biased, or repetitive token distributions. We therefore frame the task as controlled local refinement rather than full generation from noise. On ROCStories, using the first two sentences as prompt and the last three as target, full 768-dimensional BERT latents recover tokens much better than compressed 256-dimensional latents. With 768-dimensional latents, DraftPrior target-token probability is 0.938 for clean drafts, 0.613 for 3% token dropout, 0.483 for 5% dropout, and 0.272 for 10% dropout. Local flow refinement and fused decoder-aware readout give modest additional gains, while metric learning and OT-style alignment improve geometry but do not close the decoder gap. The main result is a diagnostic one: latent geometry alone is not enough. Continuous latent text generation should be evaluated by decoder recoverability, the quality of the start distribution, and whether refinement preserves decoder-readable structure.
>
---
#### [new 033] DebiasRAG: A Tuning-Free Path to Fair Generation in Large Language Models through Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的公平性任务，旨在解决大模型生成内容中的社会偏见问题。提出DebiasRAG框架，通过检索增强生成实现无调优的动态去偏。**

- **链接: [https://arxiv.org/pdf/2605.16113](https://arxiv.org/pdf/2605.16113)**

> **作者:** Rui Chu; Bingyin Zhao; Thanh Quoc Hung Le; Duy Cao Hoang; Huawei Lin; Ping Li; Weijie Zhao; Khoa D Doan; Yingjie Lao
>
> **摘要:** Large language models (LLMs) have achieved unprecedented success due to their exceptional generative capabilities. However, because they depend on knowledge encapsulated from training corpora, they may produce hallucinations, stereotypes, and socially biased content. In particular, LLMs are prone to prejudiced responses involving race, gender, and age, which are collectively referred to as social biases. Prior studies have used fine-tuning and prompt engineering to mitigate such biases in LLMs, but these methods require additional training resources or domain knowledge to design the framework. Moreover, they may degrade the original capabilities of LLMs and often overlook the need for dynamic debiasing contexts for fairer inference. In this paper, we propose DebiasRAG, a novel tuning-free and dynamic query-specific debiasing framework based on retrieval-augmented generation (RAG). DebiasRAG improves fairness while preserving the intrinsic properties of LLMs, such as representation ability. DebiasRAG consists of three stages: (1) query-specific debiasing candidate generation; (2) context candidate pool construction; and (3) gradient-updated debiasing-guided context piece reranking. First, DebiasRAG leverages self-diagnosed bias contexts relevant to the query through regular retrieval, where the bias contexts are prepared offline by the DebiasRAG provider. Given the query-specific bias contexts, DebiasRAG reversely produces debiasing contexts, which are provided as additional fairness constraints for LLM outputs. Second, a regular RAG retrieval process produces query-related contexts from the regular RAG document database, such as a chunked Wikipedia dataset.
>
---
#### [new 034] Calibrating LLMs with Semantic-level Reward
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型校准任务，旨在解决模型输出不确定性估计不足的问题。提出CSR框架，在语义空间中提升模型校准效果，优于基于口语化置信度的方法。**

- **链接: [https://arxiv.org/pdf/2605.15588](https://arxiv.org/pdf/2605.15588)**

> **作者:** Fengfei Yu; Ruijia Niu; Dongxia Wu; Yian Ma; Rose Yu
>
> **摘要:** As large language models (LLMs) are deployed in consequential settings such as medical question answering and legal reasoning, the ability to estimate when their outputs are likely to be correct is essential for safe and reliable use, requiring well-calibrated uncertainty. Standard reinforcement learning with verifiable rewards (RLVR) trains models with a binary correctness reward that is indifferent to confidence, providing no penalty for confident but wrong predictions and thereby degrading calibration. Recent work addresses this by training models to produce verbalized confidence scores alongside answers and rewarding agreement with correctness. However, verbalized confidence is calibrated at the token level and thus exhibits inconsistency across textual variations with same semantic meaning. We propose \textbf{Calibration with Semantic Reward (CSR)}, a framework that calibrates language models directly in semantic space without a verbalized confidence interface. CSR combines the correctness reward with a novel semantic calibration reward that encourages exploitation among correct rollouts by promoting semantic agreement, and exploration among incorrect ones by discouraging spurious consistency. Experiments across three model families on HotpotQA (in-distribution) and TriviaQA, MSMARCO, and NQ-Open (out-of-distribution) show that CSR consistently achieves lower ECE and higher AUROC than verbalized-confidence baselines across nearly all settings, reducing ECE by up to $40\%$ and improving AUROC by up to $31\%$ over verbalized-confidence baselines, with calibration behavior generalizing robustly across all four evaluation settings.
>
---
#### [new 035] Can Vision Language Models Be Adaptive in Mathematics Education? A Learner Model-based Rubric Study
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于教育技术任务，旨在解决VLM在数学教育中适应不同学习者的能力问题。通过构建学习者模型评估框架，分析VLM的适应性表现。**

- **链接: [https://arxiv.org/pdf/2605.16011](https://arxiv.org/pdf/2605.16011)**

> **作者:** Jie Gao; Yongan Yu; Junzhu Su; Yiran Lin; Adam K. Dube; Jackie Chi Kit Cheung
>
> **摘要:** Adaptive learning refers to educational technologies that track learners' learning progress and adapt the instructional process based on individual learners' learning performance. It is increasingly recognized as critical for developing an effective learning support tool. Vision language models (VLMs) have seen adoption in mathematics education, and students have been using them as learning aids for personalized instruction. However, it is unknown whether VLMs have the ability to adapt to different learner profiles when providing mathematical instructions. Current VLMs lack a systematic evaluation framework for this adaptivity to different learner profiles in mathematics tutoring tasks. To address this gap, we draw on the learner model from the adaptive learning framework (Shute and Towle, 2018) and propose a learner model-based rubric. Our rubric formalizes adaptivity assessment into three aspects: cognitive aspects, motivational aspects, and complexity. We also evaluate two additional dimensions of VLM responses: correctness (of answers and solutions) and quality (of the response itself). Our experimental results show measurable differences in adaptivity across models and also reveal that current VLMs struggle to consistently produce learner model-based instructional responses, especially when receiving limited learner information.
>
---
#### [new 036] Automatic Construction of a Legal Citation Graph from 100 Million Ukrainian Court Decisions: Large-Scale Extraction, Topological Analysis, and Ontology-Driven Clustering
- **分类: cs.CL; cs.DL; cs.IR**

- **简介: 该论文属于法律文本分析任务，旨在构建乌克兰法院判决的引用图谱。通过大规模提取和分析引用关系，揭示法律领域边界及立法变化，为法律智能提供数据支持。**

- **链接: [https://arxiv.org/pdf/2605.15362](https://arxiv.org/pdf/2605.15362)**

> **作者:** Volodymyr Ovcharov
>
> **备注:** 15 pages, 7 figures, 2 tables, 21 references
>
> **摘要:** Half a billion citation edges extracted from 100.7 million Ukrainian court decisions reveal that judicial citation structure encodes legal domain boundaries without supervision and predicts future legislative importance with near-perfect accuracy. We construct the first large-scale citation graph from the complete EDRSR registry (99.5 million full texts, 1.1 TB), extracting 502 million citation links across six types via regex on commodity hardware in approximately 5 hours, with precision of 1.00 on a 200-decision validation sample (95% Wilson CI: [0.982, 1.000]). Three principal findings emerge. (1) The degree distribution follows a power law (alpha = 1.57 +/- 0.008), placing the Ukrainian court network near the EU Court of Justice and below the US Supreme Court, with hub articles cited by millions of decisions. (2) Louvain community detection on the co-citation projection recovers legal domain boundaries (civil, criminal, administrative, commercial) with modularity Q = 0.44-0.55 and temporal stability (NMI = 0.83-0.86 across periods), constituting an automatically constructed legal ontology grounded in judicial practice. (3) Citation features predict top-1000 articles with AUC = 0.9984, substantially outperforming a naive frequency baseline (P@1000 = 0.655); temporal dynamics detect legislative regime changes as phase transitions and the 2022 invasion as a citation entropy spike (H: 11.02 -> 13.49) with emergent wartime legislation nodes. The citation-derived ontology is operationalized as the domain layer of a workflow memory system for LLM-assisted legal analysis, connecting to the ontology-controlled paradigm. The extraction pipeline, analysis code, and aggregated statistics are released as open data.
>
---
#### [new 037] Multi-Level Contextual Token Relation Modeling for Machine-Generated Text Detection
- **分类: cs.CL**

- **简介: 该论文属于机器生成文本检测任务，旨在解决token级评分易受生成随机性影响的问题。通过构建多层级上下文关系模型，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.16107](https://arxiv.org/pdf/2605.16107)**

> **作者:** Chenwang Wu; Yiuming Cheung; Bo Han; Shuhai Zhang; Defu Lian
>
> **摘要:** Machine-generated texts (MGTs) pose risks such as disinformation and phishing, underscoring the need for reliable detection. Metric-based methods, which extract statistically distinguishable features of MGTs, are often more practical than complex model-based methods that are prone to overfitting. Given their diverse designs, we first place representative metric-based methods within a unified framework, enabling a clear assessment of their advantages and limitations. Our analysis identifies a core challenge across these methods: the token-level detection score is easily biased by the inherent randomness of the MGTs generation process. Then, we theoretically derive the multi-hop transitions of the token-level detection score and explore their local and global relations. Based on these findings, we propose a multi-level contextual token relation modeling framework for MGT detection. Specifically, for local relations, we model them through a lightweight Markov-informed calibration module that refines token-level evidence before aggregation. For global relations, we introduce a rule-support reasoning module that uses explicit logical rules derived from contextual score statistics. Finally, we combine the local calibrated score and the global rule-support reasoning signal in a joint multi-level inference framework. Extensive experiments show broad and substantial improvements across various real-world scenarios, including cross-LLM and cross-domain settings, with low computational overhead.
>
---
#### [new 038] FINESSE-Bench: A Hierarchical Benchmark Suite for Financial Domain Knowledge and Technical Analysis in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出FINESSE-Bench，一个用于评估大语言模型金融能力的分层基准套件，解决现有基准在专业难度和任务多样性上的不足。**

- **链接: [https://arxiv.org/pdf/2605.15482](https://arxiv.org/pdf/2605.15482)**

> **作者:** Dmitry Stanishevskii; Nini Kamkia; Alexey Khoroshilov; Dmitry Zmitrovich; Denis Kokosinskii; Zhirayr Hayrapetyan; Andrei Kalmykov
>
> **备注:** 21 pages, 10 tables, 2 figures
>
> **摘要:** Large language models (LLMs) are increasingly being applied to financial analysis, reporting, investment decision support, risk management, compliance, and professional training. However, robust evaluation of their domain competence in finance remains incomplete. Widely used open benchmarks such as FinQA, ConvFinQA, and TAT-QA have played an important role in advancing financial question answering and numerical reasoning, but they focus primarily on question answering over financial reports and do not provide an explicit hierarchy of professional difficulty. Broader resources, including FinanceBench, PIXIU, FinBen, and FLaME, expand the coverage of financial tasks, yet the problem of evaluating the transition from foundational knowledge to expert-level financial reasoning remains open. In this work, we present FINESSE-Bench, a suite of eight specialized benchmarks comprising 3,993 questions for hierarchical evaluation of financial competencies in LLMs. FINESSE-Bench combines exam-oriented datasets inspired by professional certifications (CFA-like Levels 1-3, CMT-like Level 2, and CFTe-like Level 1), applied trading task collections, and a Russian-language olympiad benchmark. This design enables evaluation of domain breadth, performance degradation as difficulty increases, the ability to solve computational tasks, and model behavior in specialized financial domains. We also describe a unified evaluation protocol covering multiple-choice questions, numerical answers, and short open-ended responses, together with an automated scoring scheme for freeform answers based on the LLM-as-judge paradigm. FINESSE-Bench is intended both as a complement to existing open financial benchmarks and as a tool for more substantive evaluation of professionally relevant financial competencies in large language models.
>
---
#### [new 039] Evaluating Chinese Ambiguity Understanding in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的歧义理解任务，旨在解决中文歧义检测与分析问题。构建了首个基于潜在歧义理论的中文歧义数据集CHA-Gen，并评估大模型的歧义处理能力。**

- **链接: [https://arxiv.org/pdf/2605.15635](https://arxiv.org/pdf/2605.15635)**

> **作者:** Junwen Mo; Yuanzhi Lu; Yifang Xue; Ke Xu; Hideki Nakayama
>
> **摘要:** Linguistic ambiguity is critical to the robustness of Large Language Models (LLMs), yet existing research focuses mostly on English, with limited attention devoted to Chinese. Existing Chinese ambiguity datasets (e.g., CHAmbi) suffer from poor scalability. Guided by Potential Ambiguity (PA) Theory, we design a semi-automatic pipeline to construct CHA-Gen. It is the first PA Theory-grounded Chinese ambiguity dataset, which comprises 5,712 sentences (2,414 ambiguous, 3,298 unambiguous) across 18 potential ambiguous structures. Evaluating LLMs (e.g. Gemma 3, Qwen 2.5/3 series) via direct querying and machine translation, we find that LLMs struggle with ambiguity detection (improved by CoT prompting). Analysis of Qwen3-32B's CoT rationales reveals three common failure modes: ambiguity blindness, misattribution, and premature resolution. Uncertainty quantification with semantic entropy metric shows higher uncertainty for ambiguous sentences. Moreover, instruction tuning induces overconfidence, whereas Base models better capture semantic diversity. We further observe that models exhibit a bias toward dominant interpretations. Our work provides a scalable approach for Chinese ambiguity corpus and insights into LLMs' ambiguity handling, laying a foundation for enhancing Chinese ambiguity research in LLMs.
>
---
#### [new 040] Reasoning Models Don't Just Think Longer, They Move Differently
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文研究语言模型在推理任务中的轨迹变化，解决如何区分模型计算时间与内部路径差异的问题。通过分析生成轨迹，发现推理训练影响路径几何，尤其在代码领域明显。**

- **链接: [https://arxiv.org/pdf/2605.15454](https://arxiv.org/pdf/2605.15454)**

> **作者:** Anders Gjølbye; Lars Kai Hansen; Sanmi Koyejo
>
> **备注:** Preprint
>
> **摘要:** Reasoning-trained language models often spend more tokens on harder problems, but longer chains of thought do not show whether a model is merely computing for more steps or following a different internal trajectory. We study this distinction through hidden-state trajectories during chain-of-thought generation across competitive programming, mathematics, and Boolean satisfiability. Raw trajectory geometry is strongly shaped by generation length: longer generations mechanically alter path statistics, so difficulty-dependent comparisons are misleading without adjustment. After residualizing trajectory statistics on length, difficulty remains systematically coupled to corrected trajectory geometry across all domains studied. The clearest reasoning-specific separation appears in the code domain, where harder problems show more direct corrected trajectories and less heterogeneous local curvature in reasoning-trained models than in matched instruction-tuned baselines. Corrected difficulty-geometry coupling is weaker, but still present, in mathematics and Boolean satisfiability. Prompt-stage linear probes do not mirror the code-domain separation, and behavioral annotations show that stronger corrected coupling co-occurs with strategy shifts and uncertainty monitoring. Together, these findings establish length correction as a prerequisite for generation-time trajectory analysis and show that reasoning training can be associated with distinct corrected trajectory geometry, with the strength of the effect depending on the domain.
>
---
#### [new 041] GiLT: Augmenting Transformer Language Models with Dependency Graphs
- **分类: cs.CL**

- **简介: 该论文提出GiLT模型，将依存图融入Transformer，提升语法泛化能力。解决语言模型语法理解不足的问题，通过注意力权重调整注入结构信息。**

- **链接: [https://arxiv.org/pdf/2605.15562](https://arxiv.org/pdf/2605.15562)**

> **作者:** Tianyu Huang; Yida Zhao; Chuyan Zhou; Kewei Tu
>
> **摘要:** Augmenting Transformers with linguistic structures effectively enhances the syntactic generalization performance of language models. Previous work in this direction focuses on syntactic tree structures of languages, in particular constituency tree structures. We propose Graph-Infused Layers Transformer Language Model (GiLT) which leverages dependency graphs for augmenting Transformer language models. Unlike most previous work, GiLT does not insert extra structural tokens in language modeling; instead, it injects structural information into language modeling by modulating attention weights in the Transformer with features extracted from the dependency graph that is incrementally constructed along with token prediction. In our experiments, GiLT with semantic dependency graphs achieves better syntactic generalization while maintaining competitive perplexity in comparison with Transformer language model baselines. In addition, GiLT can be finetuned from a pretrained language model to achieve improved downstream task performance. Our code is released at this https URL.
>
---
#### [new 042] Defining Cultural Capabilities for AI Evaluation: A Taxonomy Grounded in Intercultural Communication Theory
- **分类: cs.CL**

- **简介: 该论文属于AI评估任务，旨在解决文化能力定义模糊的问题。提出三层次文化能力分类，提升AI在多元文化环境中的评估有效性。**

- **链接: [https://arxiv.org/pdf/2605.15990](https://arxiv.org/pdf/2605.15990)**

> **作者:** Isar Nejadgholi; Masoud Kianpour; Krishnapriya Vishnubhotla; Maryam Molamohamadi
>
> **摘要:** Tremendous efforts have been put into evaluating the inclusivity and effectiveness of AI systems across cultures. However, the cultural capabilities considered in much of the literature remain vaguely defined, are referred to using interchangeable terminology, and are typically limited to recalling accurate information about various demographics, regions, and nationalities. To address this construct ambiguity, we draw from Intercultural Communication scholarship and propose a three-level taxonomy of AI-relevant cultural capabilities: Cultural Awareness answers "Does the model know?", Cultural Sensitivity answers "How does it frame its knowledge?", and Cultural Competence answers "Can it adapt as the interaction evolves?". Beyond conceptual clarification, we position this taxonomy as a practical tool for improving the validity and interpretability of AI evaluation in real-world, multicultural settings. Without such construct clarity, evaluation results risk overstating model capabilities and may lead to inappropriate deployment decisions in culturally sensitive contexts.
>
---
#### [new 043] Measuring Maximum Activations in Open Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究现代开源大语言模型中的最大激活值，分析其分布与变化规律。任务属于模型量化与稳定性研究，解决激活范围对低比特部署的影响问题，通过统一方法测量多个模型的激活峰值。**

- **链接: [https://arxiv.org/pdf/2605.15572](https://arxiv.org/pdf/2605.15572)**

> **作者:** Luxuan Chen; Han Tian; Xinran Chen; Rui Kong; Fang Wang; Jiamin Chen; Yuchen Li; Jiashu Zhao; Shuaiqiang Wang; Haoyi Xiong; Dawei Yin
>
> **摘要:** The dynamic range of activations is a first-order constraint for low-bit quantization, activation scaling, and stable LLM inference. Prior work characterized outlier features and massive activations on pre-2024 LLaMA-style models, and the downstream activation-quantization stack inherits that picture without revisiting it for the post-LLaMA open-model boom. We ask the deployment-oriented question: how large can activations get in modern open LLMs, and how does this magnitude vary across families, generations, and training stages? Under a unified pipeline (5,000-sample multi-domain corpus, family-specific tokenization, identical hooks across embeddings, hidden states, attention, MLP/MoE, SwiGLU gates, and final norm), we measure global and layerwise maxima on 27 checkpoints from 8 open families spanning dense, MoE, vision-language, intermediate-training, and instruction-tuned variants. We find that (i) global maxima span over nearly four orders of magnitude at comparable parameter counts, with Qwen3.5 and MoE checkpoints in the 10^2 to 10^3 range and Gemma3-27B-it reaching ~7 x 10^5; (ii) cross-family and cross-generation comparisons break simple monotonic scaling; and (iii) MoE checkpoints exhibit 14.0-23.4x lower peaks than matched-scale dense counterparts, while the residual stream carries the global maximum in 22/24 checkpoints. A lightweight INT-8 sanity check shows that measured maxima co-vary with low-bit reconstruction error via activation-scale selection. We conclude that maximum activation magnitude is a model property tied to family, architecture, and training stage - not a simple byproduct of size - and should be measured and reported alongside any open-weight release before low-bit deployment. The code is publicly available at this https URL.
>
---
#### [new 044] Eskwai for Students: Generative AI Assistant for Legal Education in Ghana
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于法律教育任务，旨在解决全球南方地区法律教育中AI工具不足的问题。开发了Eskwai for Students系统，利用Ghana的法律数据提供辅助教学，并进行了长期研究评估其效果与伦理问题。**

- **链接: [https://arxiv.org/pdf/2605.15380](https://arxiv.org/pdf/2605.15380)**

> **作者:** George Boateng; Philemon Badu; Patrick Agyeman-Budu; Samuel Ansah; Evans Atompoya; Evan Igwilo; Lord Baah; Frederick Abu-Bonsrah; Victor Wumbor-Apin Kumbol
>
> **备注:** 10 pages. Accepted at the 27th International Conference on Artificial Intelligence in Education (AIED 2026)
>
> **摘要:** Recent advances in generative AI have shown their potential to be leveraged for legal education. Yet, work on the development and deployment of such systems for legal education in the Global South is limited. In this work, we developed Eskwai for Students, a generative AI assistant to help law students with their legal education. Eskwai for Students is a retrieval augmented generation (RAG) system that provides answers to a wide range of legal questions for law students grounded in a curated database of over 12K case laws and 1.4K legislation in Ghana. We deployed Eskwai for Students in a longitudinal study of 30 months (2.5 years) used by 3.1K law students in Ghana who made 32K queries. We evaluated the helpfulness of our AI, and provided insight into the kinds of queries law students submit to this generative AI tool, which raises some ethical concerns. This work contributes to an understanding of how law students in the Global South are using generative AI for their studies and the ways it could be leveraged responsibly to advance legal education.
>
---
#### [new 045] Ontology for Policing: Conceptual Knowledge Learning for Semantic Understanding and Reasoning in Law Enforcement Reports
- **分类: cs.CL; cs.AI; cs.LO**

- **简介: 该论文属于自然语言处理任务，旨在从警报报告中提取结构化事实。通过符号方法将非结构化文本转化为证据关联的事实，构建时间图谱。**

- **链接: [https://arxiv.org/pdf/2605.15978](https://arxiv.org/pdf/2605.15978)**

> **作者:** Anita Srbinovska; Jansen Orfan; Adrian Martin; Ernest Fokoué
>
> **备注:** 13 pages, 8 figures, 9 tables
>
> **摘要:** Law enforcement reports contain structured fields and written narratives. However, many incident facts that are needed for review, police training, and investigations are in natural language and require manual reading. We propose a framework using symbolic methods for converting narratives into evidence-linked facts. Our objective is to measure the value of narratives to recover incident details only from the unstructured text and build temporal graphs with time cues and domain axioms. We achieve this by redacting personal identifiers, semantic parsing, predicate mapping to ontology, and reasoning. We evaluate the symbolic approach on 450 property crime reports and a short human review. Of the extracted events from the system, 54.1% had a confidence score of at least 0.80 and 93.7% were mapped through the PropBank--VerbNet--WordNet semantic path. 100% agreement was reached on incident initiation, stolen items, and temporal cues and lower agreement for forced entry interpretation.
>
---
#### [new 046] A Generative AI Framework for Intelligent Utility Billing CO 2 Analytics and Sustainable Resource Optimisation
- **分类: cs.CL; cs.AI; cs.DB; cs.LG**

- **简介: 该论文属于智能电网任务，旨在解决电费计算与碳排放分析问题。通过生成式AI框架，实现精准的用电预测与资源优化。**

- **链接: [https://arxiv.org/pdf/2605.16250](https://arxiv.org/pdf/2605.16250)**

> **作者:** Pavan Manjunath; Thomas Pruefer
>
> **摘要:** Distribution utilities are now expected to deliver bills that customers can actually read attach a defensible carbon number to every kWh sold and schedule load against grid stress and emissions constraints We propose an end-to-end framework that unifies four production-grade capabilities under one architectural roof a generative-AI agent that drafts each customers natural-language billing statement from structured numeric inputs under a constrained decoding policy a transformer-based forecaster that supplies the day-ahead consumption estimate with calibrated quantile bands
>
---
#### [new 047] SGR: A Stepwise Reasoning Framework for LLMs with External Subgraph Generation
- **分类: cs.CL**

- **简介: 该论文提出SGR框架，解决LLMs在复杂推理任务中的准确性与事实可靠性问题。通过外部子图生成，增强模型的多步推理能力。**

- **链接: [https://arxiv.org/pdf/2605.16117](https://arxiv.org/pdf/2605.16117)**

> **作者:** Xin Zhang; Yang Cao; Baoxing Wu; Kai Song; Siying Li
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong capabilities across diverse NLP applications, such as translation, text generation, and question answering. Nevertheless, they remain limited in complex settings that demand deep reasoning and logical inference. Since these models are trained on large-scale text corpora, their generation process may still introduce irrelevant, noisy, or factually inconsistent content. To mitigate this problem, we introduce SGR, a stepwise framework that enhances LLM reasoning through external subgraph generation. SGR builds query-specific subgraphs from external knowledge bases and uses their semantic structure to support multi-step inference. By grounding intermediate reasoning steps in structured external knowledge, the framework helps the model concentrate on relevant entities, relations, and supporting evidence. In particular, SGR first constructs a subgraph tailored to the input question. It then guides the model to reason progressively over the generated structure and combines multiple reasoning trajectories to obtain the final prediction. Experimental results across several benchmark datasets show that SGR achieves consistent improvements over competitive baselines, highlighting its value for improving both reasoning accuracy and factual reliability.
>
---
#### [new 048] DiscoExplorer: An Open Interface for the Study of Multilingual Discourse Relations
- **分类: cs.CL**

- **简介: 该论文属于多语言话语关系研究任务，旨在解决跨语言话语关系分析困难的问题。提出DiscoExplorer工具，提供数据查询与可视化功能，支持16种语言的分析。**

- **链接: [https://arxiv.org/pdf/2605.15304](https://arxiv.org/pdf/2605.15304)**

> **作者:** Amir Zeldes
>
> **摘要:** The relations connecting propositions in discourse such as cause (A because B) or concession (A although B) are a subject of intense interest in Computational Linguistics and Pragmatics, but challenging to study and compare across languages. Recent progress in standardizing discourse relation inventories across datasets offers the potential to facilitate such studies, but is hindered by the complexity of relevant data and the lack of easily accessible interfaces to analyze it. In this paper we present DiscoExplorer, a new open source web interface, capable of running on local computers, which we use to make datasets from the DISRPT Shared Task on discourse relation classification publicly available, covering 16 different languages. We present the query language, search and visualization facilities for relations and signaling devices such as connectives, as well as some example studies.
>
---
#### [new 049] RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于长周期LLM代理的记忆管理任务，旨在降低记忆构建的token消耗。通过基于循环的内存整合策略，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.16045](https://arxiv.org/pdf/2605.16045)**

> **作者:** Zijie Dai; Shiyuan Deng; Sheng Guan; Yizhou Tian; Xin Yao; Xiao Yan; James Cheng
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Memory systems often organize user-agent interactions as retrievable external memory and are crucial for long-running agents by overcoming the limited context windows of LLMs. However, existing memory systems invoke LLMs to process every incoming interaction for memory extraction, and such an eager memory consolidation scheme leads to substantial token consumption. To tackle this problem, we propose RecMem by rethinking when memory consolidation should be conducted. RecMem stores incoming interactions in a subconscious memory layer and encode them using lightweight embedding models for retrieval. LLMs are only invoked to extract episodic and semantic memory when sustained recurrence are observed for semantically similar interactions. Such recurrence-based consolidation works because these interactions correspond to a semantic cluster with rich information and thus are worth extraction and summarization. To improve accuracy, RecMem also incorporates a semantic refinement mechanism that recovers the fine-grained facts omitted by memory extraction. Experiments show that RecMem reduces the memory construction token cost of three SOTA memory systems by up to 87% while exceeding their accuracy.
>
---
#### [new 050] Few-Shot Large Language Models for Actionable Triage Categorization of Online Patient Inquiries
- **分类: cs.CL; cs.LG; q-bio.QM**

- **简介: 该论文研究在线患者咨询的分类任务，旨在解决低资源下自动分诊问题。通过对比不同模型在少量标注数据下的表现，验证大语言模型的有效性。**

- **链接: [https://arxiv.org/pdf/2605.15680](https://arxiv.org/pdf/2605.15680)**

> **作者:** Liqi Zhou; Jiafu Li
>
> **备注:** 4 figures, 19 tables, 23 pages (including appendix and reference)
>
> **摘要:** Online patient inquiries are often informal, incomplete, and written before professional assessment, yet they must still be routed to an appropriate level of clinical follow-up. We study this as a four-class actionable triage task -- self-care, schedule-visit, urgent-clinician-review, or emergency-referral, and ask whether prompted large language models (LLMs) can support such routing under low-resource labeling conditions. Using the public HealthCareMagic-100K corpus, we construct a 300-example human calibrated gold evaluation set, a 700-example auto-labeled silver training set, and a 40-example few-shot pool. We compare Term Frequency-Inverse Document Frequency (TF-IDF) and Bidirectional Encoder Representations from Transformers for Biomedical Text Mining (BioBERT) baselines train on silver labels against six prompted LLMs under 0-shot, 4-shot, and 12-shot conditions respectively. Accordingly, we evaluate with macro-$F_1$ alongside safety-aware metrics, including emergency-recall, under-triage rate, and severe under-triage rate. The strongest LLM (Claude Haiku 4.5, 12-shot) reaches macro-$F_1$ 0.475, exceeding the best supervised baseline (BioBERT, 0.378) on point estimate, with overlapping confidence intervals. Few-shot prompting and two-model agreement help in label-dependent ways: self-care agreement is reliable, urgent-clinician-review is not. We conclude that LLMs can support triage prioritization and selective human review, but not autonomous deployment.
>
---
#### [new 051] Toward LLMs Beyond English-Centric Development
- **分类: cs.CL**

- **简介: 论文探讨了大语言模型的英语中心偏差问题，指出持续预训练在成本上不优于从头训练，强调未来需重视多语言投资。属于自然语言处理任务，解决多语言模型发展问题。**

- **链接: [https://arxiv.org/pdf/2605.15613](https://arxiv.org/pdf/2605.15613)**

> **作者:** Sho Takase; Ukyo Honda
>
> **摘要:** Through an analysis of sequences generated by open-weight large language models (LLMs), we demonstrate that LLMs are heavily biased toward English. While continual pre-training is commonly used to adapt LLMs to a target language, we show that it does not offer a cost advantage over training from scratch, even for improving cultural understanding in the target language. These findings suggest that dedicated per-language investment may become increasingly important for future LLM development, rather than relying primarily on the expansion of English-centric resources.
>
---
#### [new 052] Neural Activation Patterns Across Language Model Architectures: A Comprehensive Analysis of Cognitive Task Performance
- **分类: cs.CL; cs.LG**

- **简介: 该论文分析不同语言模型架构在认知任务中的神经激活模式，比较其性能差异，旨在揭示模型计算特性与任务行为关系。**

- **链接: [https://arxiv.org/pdf/2605.15436](https://arxiv.org/pdf/2605.15436)**

> **作者:** Mahdi Naser-Moghadasi; Faezeh Ghaderi
>
> **备注:** 8 pages, accepted at IEEE BigData 2025
>
> **摘要:** This paper presents a comprehensive analysis of neural activation patterns across six distinct large language model (LLM) architectures, examining their performance on twelve cognitive task categories. Through systematic measurement of final activation values, attention entropy, and sparsity patterns, we reveal fundamental differences in how encoder and decoder architectures process diverse cognitive tasks. Our analysis of 144 task-model combinations demonstrates that mathematical reasoning consistently produces the highest attention entropy across all architectures, while decoder models exhibit significantly higher sparsity patterns compared to encoder models. The findings provide critical insights into the computational characteristics of modern language models and their task-specific neural behaviors, with implications for model selection and optimization in big data applications.
>
---
#### [new 053] Greedy or not, here I come: Language production under vocabulary constraints in humans and resource-rational models
- **分类: cs.CL**

- **简介: 该论文研究人类在词汇限制下的语言生成，比较其与贪心和全局最优算法的相似性。任务是理解资源理性认知中的语言生产问题。**

- **链接: [https://arxiv.org/pdf/2605.15365](https://arxiv.org/pdf/2605.15365)**

> **作者:** Thomas Hikaru Clark; Sihan Chen; Laura Nicolae
>
> **摘要:** Communicating using only a limited vocabulary is a common but challenging cognitive phenomenon, requiring an ideal communicator to plan carefully to optimize for intelligibility while circumventing a constrained lexicon. In this work, we investigate how humans respond to a broad array of questions under variable vocabulary limitations, consisting of only 250 highly frequent words at the most restrictive. We provide theoretically motivated comparisons to greedy and globally optimal sampling algorithms using Sequential Monte Carlo inference with large language models. Humans generally resemble greedy sampling more than globally optimal sampling, though more skilled humans are more likely to backtrack and revise -- a non-greedy behavior. An observed human pattern of leaning on semantically light words in high-constraint settings falls out of both greedy and globally optimal sampling. We discuss the results and their broader implications for resource-rational cognition, psycholinguistics, L2 communication, and language impairments.
>
---
#### [new 054] DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection
- **分类: cs.CL**

- **简介: 该论文属于LLM生成文本检测任务，旨在提升多语言和真实场景下的检测可靠性。提出DetectRL-X基准，涵盖多语言、多领域数据，并模拟实际使用模式进行测试。**

- **链接: [https://arxiv.org/pdf/2605.15518](https://arxiv.org/pdf/2605.15518)**

> **作者:** Junchao Wu; Yefeng Liu; Chenyu Zhu; Hao Zhang; Zeyu Wu; Tianqi Shi; Yichao Du; Longyue Wang; Weihua Luo; Jinsong Su; Derek F. Wong
>
> **备注:** ACL 2026 Main
>
> **摘要:** The effective detection and governance of Large Language Model (LLM) generated content has become increasingly critical due to the growing risk of misuse. Despite the impressive performance of existing detectors, their reliability and potential in multilingual, real-world scenarios remain largely underexplored. In this study, we introduce DetectRL-X, a comprehensive multilingual benchmark designed to evaluate advanced detectors across 8 dimensions. The benchmark encompasses 8 languages commonly used in commercial contexts and collects human-written texts from 6 domains highly susceptible to LLM misuse. To better aligned with real-world applications, We create LLM-generated texts using 4 popular commercial LLMs, and include typical AI-assisted writing operations such as polishing, expanding, and condensing to capture authentic usage patterns. Furthermore, we develop a multilingual framework for paraphrasing and perturbation attacks to simulate diverse human modifications and writing noise, enabling stress testing of detectors across languages. Experimental results on DetectRL-X reveal the strengths and limitations of current state-of-the-art detectors when applied to diverse linguistic resources. We further analyze how domains, generators, attack strategies, text length, and refinement operations influence performance in different languages, underscoring DetectRL-X as an effective benchmark for strengthening multilingual and language-specific detectors.
>
---
#### [new 055] Fluency and Faithfulness in Human and Machine Literary Translation
- **分类: cs.CL**

- **简介: 该论文研究文学翻译中流畅性与忠实性的关系，属于机器翻译评估任务。通过分析大量翻译文本，发现流畅性与忠实性存在负相关，揭示了两者间的权衡问题。**

- **链接: [https://arxiv.org/pdf/2605.15282](https://arxiv.org/pdf/2605.15282)**

> **作者:** Sarah Griebel; Ted Underwood
>
> **备注:** Accepted NLP4DH 2026
>
> **摘要:** Literary translation requires balancing target-language fluency with faithfulness to the source. Recent large language models (LLMs) often produce fluent translations, but it remains unclear whether fluency corresponds to semantic preservation in literary text. We examine this relationship using 130,486 translated paragraphs from 106 novels in 16 source languages, including human, Google Translate, and TranslateGemma translations. Fluency is measured as original-likeness with a translationese classifier trained on paragraph part-of-speech n-grams, and faithfulness with the automatic translation evaluation metric COMET-KIWI. We control for paragraph length and find a consistent negative correlation between fluency and faithfulness. The pattern appears for both human and Google Translate, but is weaker and often non-significant for TranslateGemma. These results show that segment length matters for automatic evaluation and suggest a tradeoff between fluency and faithfulness in literary translation.
>
---
#### [new 056] SLIP & ETHICS: Graduated Intervention for AI Emotional Companions
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于情感计算任务，旨在解决AI情感陪伴中的安全与信任矛盾。提出SLIP和ETHICS方法，通过分级干预实现安全与关系的平衡。**

- **链接: [https://arxiv.org/pdf/2605.15915](https://arxiv.org/pdf/2605.15915)**

> **作者:** Minseo Kim
>
> **备注:** Accepted to PervasiveHealth 2026. 11 pages, 2 figures, 4 tables. Proc. of the 20th EAI International Conference on Pervasive Computing Technologies for Healthcare (PervasiveHealth 2026)
>
> **摘要:** AI emotional companions face a safety-rapport paradox: restrictive safeguards can damage supportive alliance, while permissive systems risk user harm. We present SLIP (Staged Layers of Intervention Protocol), a four-stage graduated methodology deriving interventions (none, soft, hard) from structured qualitative indicators -- affect intensity (a) and narrative dynamism (m) -- alongside ETHICS (Emergent Taxonomy for Human-AI Interaction Context Signals), a "signals not labels" taxonomy. An evaluation combining a small-scale production deployment (N=68 entries, 10 users, 10 weeks) with a synthetic persona battery (N=91, 5 behavioral-risk profiles) achieved 0% false positives for the flow persona and showed expected escalation patterns in crisis-oriented personas. However, initial results showed that 8 consecutive days of high-energy elevation produced zero interventions (0/8), exposing a boundary where the "do not pathologize" principle conflicts with safety. A subsequent three-model stress test demonstrated that increased model capability improves detection from 0/8 to 6/8 while preserving 0/10 flow false positives in the largest model. Read as preliminary, these findings position graduated intervention as a design direction for navigating -- not resolving -- the safety-rapport tension in affective computing.
>
---
#### [new 057] BootstrapAgent: Distilling Repository Setup into Reusable Agent Knowledge
- **分类: cs.SE; cs.CL; cs.MA**

- **简介: 该论文属于代码代理任务，解决仓库启动知识重复使用问题。通过构建可重用的.Bootstrap合同，提升代理效率与修复能力。**

- **链接: [https://arxiv.org/pdf/2605.15815](https://arxiv.org/pdf/2605.15815)**

> **作者:** Sihan Fu; Oucheng Liu; Shiyuan Wang; Jin Shi; Chengkun Wei
>
> **备注:** 19 pages, 9 figures, 6 tables
>
> **摘要:** Code agents increasingly help developers work with unfamiliar repositories, but every such task depends on a costly prerequisite: bootstrapping the repository into a usable development state. This process requires substantial trial-and-error exploration, yet the resulting knowledge--resolved dependencies, repair strategies--stays trapped in a single conversation, unavailable to future agents. We therefore formulate repository bootstrapping as a reusable startup knowledge problem and introduce BootstrapAgent, a multi-agent framework that distills the heuristics discovered during bootstrap exploration into a persistent, verifiable, agent-consumable .bootstrap contract. Through evidence extraction, structured planning, deterministic Docker-based verification, and trace-driven repair, BootstrapAgent generates a contract covering environment setup, diagnostic checks, minimal verification, and accumulated repair knowledge. We further propose warm repair with clean replay to accelerate iterative debugging without sacrificing cold-start reproducibility, and a delta repair with sanity check to prevent reward hacking. Experiments on three benchmarks show that BootstrapAgent achieves a 92.9% success rate, outperforming the baseline by over 10% while reducing downstream agent token usage by 25.9% and build time by 22.3%. Our code is available at this https URL.
>
---
#### [new 058] From Feedback Loops to Policy Updates: Reinforcement Fine-Tuning for LLM-Based Alpha Factor Discovery
- **分类: cs.CE; cs.AI; cs.CL**

- **简介: 该论文属于量化交易中的alpha因子发现任务，解决传统方法因反馈循环导致的效率低、信息稀释问题。提出QuantEvolver框架，通过强化学习微调提升因子生成效果。**

- **链接: [https://arxiv.org/pdf/2605.15412](https://arxiv.org/pdf/2605.15412)**

> **作者:** Lingzhe Zhang; Tong Jia; Yunpeng Zhai; Zixuan Xie; Chiming Duan; Minghua He; Philip S. Yu; Ying Li
>
> **摘要:** Modern quantitative trading increasingly relies on systematic models to extract predictive signals from large-scale financial data, where alpha factor discovery plays a central role in transforming market observations into tradable signals. Recent LLM-based methods have shown promise in automating factor generation, but most of them still rely on prompt-level generation--evaluation--feedback loops for iterative optimization. As the loop becomes longer, repeatedly appended historical candidates and feedback can cause context explosion, increase inference cost, dilute useful information, and introduce feedback drift. Moreover, these methods often depend on very large LLMs whose stable generation preferences may lead to structurally similar expressions, redundant candidates, and search stagnation. To address these limitations, we propose \textsc{QuantEvolver}, a self-evolving alpha factor discovery framework based on reinforcement fine-tuning. Instead of accumulating feedback in the prompt, \textsc{QuantEvolver} converts executable quantitative evaluation into policy updates, enabling a Miner LLM to internalize historical optimization experience through parameter learning. Specifically, \textsc{QuantEvolver} constructs high-quality seed factors, builds diverse seed--time-window training tasks, generates executable Factor DSL expressions, evaluates them through Regime Backtest, and optimizes the Miner LLM with Diversity-Complementarity Reward. During training, high-quality factors are continuously accumulated in a Mined Factor Database, which serves as the final discovered factor library. Extensive experiments on three realistic market benchmarks demonstrate the effectiveness of \textsc{QuantEvolver}, which consistently improves the primary evaluation metric of each task over existing LLM-based alpha factor discovery baselines, produces higher-quality and more complementary factor pools.
>
---
#### [new 059] Confirming Correct, Missing the Rest: LLM Tutoring Agents Struggle Where Feedback Matters Most
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于智能辅导系统任务，研究LLM在诊断学生解题过程中的准确性。工作包括构建基准测试，评估LLM在不同反馈条件下的诊断表现，发现其在关键情境中存在错误。**

- **链接: [https://arxiv.org/pdf/2605.16207](https://arxiv.org/pdf/2605.16207)**

> **作者:** Tahreem Yasir; Wenbo Li; Sam Gilson; Sutapa Dey Tithi; Xiaoyi Tian; Tiffany Barnes
>
> **备注:** 22 pages, 20 fgures
>
> **摘要:** Effective tutoring requires distinguishing optimal, valid but suboptimal, and incorrect student solutions, a distinction central to intelligent tutoring systems (ITS) but untested for LLM-based tutors. As LLMs are increasingly explored as conversational complements to ITS, evaluating their diagnostic precision is essential. We present a benchmark of seven LLM feedback agents in propositional logic using knowledge-graph-derived ground truth across 10,836 solution--feedback pairs and three feedback conditions. Models achieved near-ceiling performance on optimal steps but systematically over-rejected valid but suboptimal reasoning and over-validated incorrect solutions, precisely where adaptive tutoring matters most. These failures persisted across models regardless of solution context, suggesting architectural rather than informational limits. Moreover, accurate diagnosis did not reliably produce pedagogically actionable feedback, revealing a gap between diagnostic judgment and instructional effectiveness. Our findings suggest that LLMs are better suited for hybrid architectures where KG-grounded models handle diagnosis while LLMs support open-ended scaffolding and dialogue.
>
---
#### [new 060] VSPO: Vector-Steered Policy Optimization for Behavioral Control
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出VSPO方法，解决语言模型在多目标优化中的行为控制问题，通过向量引导提升行为多样性与奖励效率。**

- **链接: [https://arxiv.org/pdf/2605.15604](https://arxiv.org/pdf/2605.15604)**

> **作者:** Xuechen Zhang; Zijian Huang; Kai Yang; Weijia Zhang; Jiasi Chen; Samet Oymak
>
> **摘要:** Modern language models often need to optimize a primary accuracy objective while also accommodating secondary behavioral preferences, such as verbosity, agreeableness, or the level of technical expertise in its response. In practice, a base model may exhibit a desired behavior very rarely or not at all. Thus, endowing the model with a target behavior creates a sparse behavioral reward bottleneck. To address such multi-objective problems, we introduce Vector-Steered Policy Optimization (VSPO) which employs a steering vector associated with the target behavior to control the behavior intensity of the generated rollouts. VSPO is obtained by modifying GRPO to sample rollouts with varying steering intensities. This process can be interpreted as an on-policy latent self-distillation procedure where the model internalizes its steering vector. By varying steering intensities, VSPO upsamples rare behaviors and enriches rollout diversity, which alleviates the sparse reward issue and provably accelerates the policy optimization. Through comprehensive theory and experiments, we establish that VSPO has favorable properties compared to vanilla reward shaping and other alternative approaches. Specifically, under a bandit abstraction, VSPO provably achieves better iteration complexity than reward-shaped GRPO when the steering-induced distributions are sufficiently aligned with the target behavior. We evaluate VSPO across multiple reasoning benchmarks, including MATH and MMLU-Pro, for four target behaviors: explanation expertise, confidence expression, robustness to misleading context, and response verbosity. Our results show that VSPO consistently improves the control along target behavior while maintaining or improving task accuracy compared with reward shaping, teacher-trace distillation, and guidance-based baselines.
>
---
#### [new 061] GRLO: Towards Generalizable Reinforcement Learning in Open-Ended Environments from Zero
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决大模型后训练效率低的问题。通过GRLO方法，在少量数据下提升模型泛化能力，实现高效后训练。**

- **链接: [https://arxiv.org/pdf/2605.15464](https://arxiv.org/pdf/2605.15464)**

> **作者:** Shangjian Yin; Yu Fu; Yue Dong; Zhouxing Shi
>
> **摘要:** Post-training has become a crucial step for unlocking the capabilities of large language models, with reinforcement learning (RL) emerging as a critical paradigm. Recent RL-based post-training has increasingly split into two paradigms: reinforcement learning from human feedback (RLHF), which optimizes models using human preference signals in target domains, and reinforcement learning from verifiable rewards (RLVR), which operates in verifier-backed environments. The latter has dominated recent reasoning-oriented post-training because it delivers stronger gains and higher efficiency on domain-specific tasks (e.g., reasoning). However, although in-domain RL training achieves promising performance, it still requires a substantial amount of GPU compute, which remains a major barrier to broad adoption. In this work, we study the generalization ability of RLHF learned from scratch from a small set of interactions in open-ended environments, and investigate whether the conversational abilities it explicitly acquires can implicitly transfer to downstream tasks such as mathematical reasoning and code generation, namely GRLO. Specifically, on Qwen3-4B-Base backbone, GRLO improves the average performance across all domains from 24.1 to 63.1 with only 5K prompts and 22.7 GPU hours, requiring about $46\times$ less data and $68\times$ less compute than a strong in-domain RLVR baseline. The resulting model is even competitive with Qwen's released post-trained models which required a much larger training cost. Notably, a subsequent in-domain RLVR stage brings only selective gains, mainly on harder competition-math benchmarks. We hope GRLO offers a simple and efficient recipe for building broadly capable post-trained models. Our code and data will be available at: \href{this https URL}{this https URL}.
>
---
#### [new 062] DeltaPrompts: Escaping the Zero-Delta Trap in Multimodal Distillation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于多模态知识蒸馏任务，旨在解决标准数据集中大量零差异提示导致学生模型训练效果受限的问题。通过生成高差异提示，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.15532](https://arxiv.org/pdf/2605.15532)**

> **作者:** Jaehun Jung; Hyunwoo Kim; Brandon Cui; Ximing Lu; David Acuna; Prithviraj Ammanabrolu; Yejin Choi
>
> **摘要:** Distillation enables compact Vision-Language Models (VLMs) to obtain strong reasoning capabilities, yet the prompts driving this process are typically chosen via simple heuristics or aggregated from off-the-shelf datasets. We reveal a critical inefficiency in this approach: up to 69% of the prompts in standard chart / document reasoning datasets are effectively zero-delta, meaning the teacher and student already induce the exact same answer distribution. Training on these prompts provides minimal learning signal, causing student improvement to rapidly saturate regardless of data scale. To escape the zero-delta trap, we return to first principles: distillation fundamentally minimizes distributional divergence, and thus a prompt is valuable only if it exposes a functional capability gap between the teacher and student. We quantify this gap through answer divergence ($\Delta$), demonstrating that non-zero divergence is critical for effective scaling. Building on this insight, we propose a staged synthesis pipeline that repurposes existing datasets as seeds, actively targeting student failure modes to produce better prompts. The result is DeltaPrompts, a diverse dataset of 200k synthetic, high-divergence reasoning problems. We evaluate DeltaPrompts across three distinct settings: on-policy distillation with the target teacher-student pair, transfer to a novel model family without regenerating the data, and off-policy fine-tuning of a non-reasoning model. Across all scenarios, DeltaPrompts drives substantial gains, yielding up to 15% relative improvement even on top of a highly-optimized reasoning model (e.g., Qwen3-VL-8B-Thinking) -- averaged over 10 benchmarks spanning chart, document and perception-centric reasoning.
>
---
#### [new 063] Conversations in Space: Structuring Non-Linear LLM Interactions on a Canvas
- **分类: cs.HC; cs.CL**

- **简介: 该论文提出CanvasConvo，解决LLM对话线性结构限制问题，通过空间画布实现非线性对话管理，支持多路径探索与结构化交互。**

- **链接: [https://arxiv.org/pdf/2605.15848](https://arxiv.org/pdf/2605.15848)**

> **作者:** Rifat Mehreen Amin; Alperen Adatepe; Daniela Fernandes; Daniel Buschek; Andreas Butz
>
> **摘要:** Conversational interfaces powered by large language models (LLMs) are widely used for ideation and analysis, yet their linear structure limits exploration of alternatives and management of long-running interactions. We present CanvasConvo, a conversational interface concept that transforms linear chat into a branching conversation tree embedded in a spatial canvas. CanvasConvo enables users to explore what-if scenarios by branching directly from conversational content, supporting parallel development of alternative directions. These branches are visualized on a canvas while remaining integrated with a familiar chat interface, allowing users to switch between linear and non-linear interaction. Features such as timeline-based navigation, automatic tagging and summarization, and context-aware controls (e.g., goals, reusable prompts) support structured interaction and continuity. We evaluated CanvasConvo in a 5-7 day field study with 24 participants. Our findings highlight how non-linear conversational structures support exploratory workflows and different interactions in LLM-based work.
>
---
#### [new 064] Context Pruning for Coding Agents via Multi-Rubric Latent Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于代码生成任务，解决编码代理在处理代码时因冗余上下文导致效率低的问题。提出LaMR框架，通过多维度分析代码相关性，提升 pruning 效果。**

- **链接: [https://arxiv.org/pdf/2605.15315](https://arxiv.org/pdf/2605.15315)**

> **作者:** Jingjing Wang; Xiwen Chen; Wenhui Zhu; Huayu Li; Zhengxiao He; Feiyang Cai; Ana S. Carreon-Rascon; Xuanzhao Dong; Feng Luo
>
> **摘要:** LLM-powered coding agents spend the majority of their token budget reading repository files, yet much of the retrieved code is irrelevant to the task at hand. Existing learned pruners compress this context with a single-objective sequence labeler, collapsing all facets of code relevance into one score and one transition matrix. We show that this formulation creates a modeling bottleneck: a single CRF transition prior must serve heterogeneous retention patterns, including contiguous semantic spans and sparse structural support lines. We propose LaMR (Latent Multi-Rubric), a structured pruning framework that decomposes code relevance into two interpretable quality dimensions, semantic evidence and dependency support, each modeled by a dedicated CRF with dimension-specific transition dynamics. A mixture-of-experts gating network dynamically weights the per-rubric emissions conditioned on the query, and a final CRF layer on the fused emissions produces the aggregate keep-or-prune decision. To supervise each dimension without additional annotation cost, we derive multi-rubric labels from the existing training corpus via AST-based program analysis, simultaneously denoising the teacher's binary labels. By effectively filtering distracting noise, LaMR frequently matches or even outperforms unpruned full-context baselines. Experiments on four benchmarks (SWE-Bench Verified, SWE-QA, LCC, LongCodeQA) show that LaMR wins 12 of 16 head-to-head multi-turn comparisons. It saves up to 31% more tokens on multi-turn agent tasks and improves Exact Match by up to +3.5 on single-turn tasks, while performance is frequently enhanced by denoising the context, and any remaining drops are marginal.
>
---
#### [new 065] Context, Reasoning, and Hierarchy: A Cost-Performance Study of Compound LLM Agent Design in an Adversarial POMDP
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; eess.SY**

- **简介: 该论文研究在对抗性部分可观测环境中的复合大语言模型代理设计，旨在优化成本与性能。通过实验比较不同设计维度的影响，提出有效设计原则。**

- **链接: [https://arxiv.org/pdf/2605.16205](https://arxiv.org/pdf/2605.16205)**

> **作者:** Igor Bogdanov; Chung-Horng Lung; Thomas Kunz; Jie Gao; Adrian Taylor; Marzia Zaman
>
> **摘要:** Deploying compound LLM agents in adversarial, partially observable sequential environments requires navigating several design dimensions: (1) what the agent sees, (2) how it reasons, and (3) how tasks are decomposed across components. Yet practitioners lack guidance on which design choices improve performance versus merely increase inference costs. We present a controlled study of compound LLM agent design in CybORG CAGE-2, a cyber defense environment modeled as a Partially Observable Markov Decision Process (POMDP). Reward is non-positive, so all configurations operate in a failure-mitigation mode. Our evaluation spans five model families, six models, and twelve configurations (3,475 episodes) with token-level cost accounting. We vary context representation (raw observations vs. a deterministic state-tracking layer with compressed history), deliberation (self-questioning, self-critique, and self-improvement tools, with optional chain-of-thought prompting), and hierarchical decomposition (monolithic ReAct vs. delegation to specialized sub-agents). We find that: (1) Programmatic state abstraction delivers the largest returns per token spent (RPTS), improving mean return by up to 76% over raw observations. (2) Distributing deliberation tools across a hierarchy degrades performance relative to hierarchy alone for all five model families, reaching up to 3.4$\times$ worse mean return while using 1.8-2.7$\times$ more tokens. We call this destructive pattern a deliberation cascade. (3) Hierarchical decomposition without deliberation achieves the best absolute performance for most models, and context engineering is generally more cost-effective than deliberation. These findings suggest a design principle for structured adversarial POMDPs: invest in programmatic infrastructure and clean task decomposition rather than deeper per-agent reasoning, as these strategies can interfere when combined.
>
---
#### [new 066] PhysBrain 1.0 Technical Report
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在提升机器人物理理解。通过将人类视角视频转为物理常识监督，增强机器人泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.15298](https://arxiv.org/pdf/2605.15298)**

> **作者:** Shijie Lian; Bin Yu; Xiaopeng Lin; Changti Wu; Hang Yuan; Xiaolin Hu; Zhaolong Shen; Yuzhuo Miao; Haishan Liu; Yuxuan Tian; Yukun Shi; Cong Huang; Kai Chen
>
> **备注:** Project Page: this https URL
>
> **摘要:** Vision-language-action models have advanced rapidly, but robot trajectories alone provide limited coverage for learning broad physical understanding. PhysBrain 1.0 studies a complementary route: converting large-scale human egocentric video into structured physical commonsense supervision before robot adaptation. Our data engine extracts scene elements, spatial dynamics, action execution, and depth-aware relations, then turns them into question-answer supervision for training PhysBrain VLMs. The resulting physical priors are further transferred to VLA policies through a capability-preserving and language-sensitive adaptation design. Across multimodal QA benchmarks and embodied control benchmarks, including ERQA, PhysBench, SimplerEnv-WidowX, LIBERO, and RoboCasa, PhysBrain 1.0 achieves SOTA results and shows especially strong out-of-domain performance on SimplerEnv. These results suggest that scaling physical commonsense from human interaction video can provide an effective bridge from multimodal understanding to robot action.
>
---
#### [new 067] MultiMat: Multimodal Program Synthesis for Procedural Materials using Large Multimodal Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MultiMat，解决程序化材质生成任务中的可视化编程问题。通过多模态模型提升材质节点图的生成效率与质量。**

- **链接: [https://arxiv.org/pdf/2509.22151](https://arxiv.org/pdf/2509.22151)**

> **作者:** Jonas Belouadi; Tamy Boubekeur; Adrien Kaiser
>
> **备注:** Accepted at ICLR 2026 (poster)
>
> **摘要:** Material node graphs are programs that generate the 2D channels of procedural materials, including geometry such as roughness and displacement maps, and reflectance such as albedo and conductivity maps. They are essential in computer graphics for representing the appearance of virtual 3D objects parametrically and at arbitrary resolution. In particular, their directed acyclic graph structure and intermediate states enable a modular, interpretable workflow for interactive appearance modeling. However, creating such graphs remains challenging and typically requires professional training. While recent neural program synthesis approaches attempt to simplify this process, they solely represent graphs as textual programs, failing to capture the inherently visual-spatial nature of node graphs that makes them accessible to humans. To address this gap, we present MultiMat, a multimodal program synthesis framework that leverages large multimodal models to process both visual and textual graph representations for improved generation of procedural material graphs. We train our models on a new dataset of production-quality procedural materials and combine them with a constrained tree search inference algorithm that ensures static correctness while efficiently navigating the program space. Our experimental results show that our multimodal program synthesis method is more efficient in both unconditional and conditional graph synthesis with higher visual quality and fidelity than text-only baselines, establishing new state-of-the-art performance.
>
---
#### [new 068] FORGE: Self-Evolving Agent Memory With No Weight Updates via Population Broadcast
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; eess.SY**

- **简介: 该论文提出FORGE，一种无需梯度更新的自进化代理记忆机制，用于提升LLM在复杂任务中的决策能力。**

- **链接: [https://arxiv.org/pdf/2605.16233](https://arxiv.org/pdf/2605.16233)**

> **作者:** Igor Bogdanov; Chung-Horng Lung; Thomas Kunz; Jie Gao; Adrian Taylor; Marzia Zaman
>
> **摘要:** Can LLM agents improve decision-making through self-generated memory without gradient updates? We propose FORGE (Failure-Optimized Reflective Graduation and Evolution), a staged, population-based protocol that evolves prompt-injected natural-language memory for hierarchical ReAct agents. FORGE wraps a Reflexion-style inner loop, where a dedicated reflection agent (using the same underlying LLM, no distillation from a stronger model) converts failed trajectories into reusable knowledge artifacts: textual heuristics (Rules), few-shot demonstrations (Examples), or both (Mixed), with an outer loop that propagates the best-performing instance's memory to the population between stages and freezes converged instances via a graduation criterion. We evaluate on CybORG CAGE-2, a stochastic network-defense POMDP at a 30-step horizon against the B-line attacker, where all four tested LLM families (Gemini-2.5-Flash-Lite, Grok-4-Fast, Llama-4-Maverick, Qwen3-235B) exhibit strongly negative, heavy-tailed zero-shot rewards. Compared against both a zero-shot baseline and a Reflexion baseline (isolated single-stream learning), FORGE improves average evaluation return by 1.7-7.7$\times$ over zero-shot and by 29-72% over Reflexion in all 12 model-representation conditions, reducing major-failure rates (below $-100$) to as low as $\sim$1%. We find that (1) population broadcast is critical mechanism, with a no-graduation ablation confirming that broadcast carries the performance gains while graduation primarily saves compute; (2) Examples achieves the strongest returns for three of four models, Rules offers the best cost-reliability profile with $\sim$40% fewer tokens; and (3) weaker baseline models benefit disproportionately, suggesting FORGE may mitigate capability gaps rather than amplify strong models. All evidence is confined to CAGE-2 B-line; cross-family findings are directional evidence.
>
---
#### [new 069] Look Before You Leap: Autonomous Exploration for LLM Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM代理在陌生环境中因过早利用先验知识而失败的问题。通过引入探索检查点覆盖率和探索-执行范式，提升代理的自主探索能力。**

- **链接: [https://arxiv.org/pdf/2605.16143](https://arxiv.org/pdf/2605.16143)**

> **作者:** Ziang Ye; Wentao Shi; Yuxin Liu; Yu Wang; Zhengzhou Cai; Yaorui Shi; Qi Gu; Xunliang Cai; Fuli Feng
>
> **摘要:** Large language model based agents often fail in unfamiliar environments due to premature exploitation: a tendency to act on prior knowledge before acquiring sufficient environment-specific information. We identify autonomous exploration as a critical yet underexplored capability for building adaptive agents. To formalize and quantify this capability, we introduce Exploration Checkpoint Coverage, a verifiable metric that measures how broadly an agent discovers key states, objects, and affordances. Our systematic evaluation reveals that agents trained with standard task-oriented reinforcement learning consistently exhibit narrow and repetitive behaviors that impede downstream performance. To address this limitation, we develop a training strategy that interleaves task-execution rollouts and exploration rollouts, with each type of rollout optimized by its corresponding verifiable reward. Building on this training strategy, we propose the Explore-then-Act paradigm, which decouples information-gathering from task execution: agents first utilize an interaction budget to acquire grounded environmental knowledge, then leverage it for task resolution. Our results demonstrate that learning to systematically explore is imperative for building generalizable and real-world-ready agents.
>
---
#### [new 070] Are VLMs Seeing or Just Saying? Uncovering the Illusion of Visual Re-examination
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型研究，旨在解决模型是否真正进行视觉再检查的问题。通过图像替换实验，发现模型多为“说”而非“看”，提出VS-Bench数据集验证这一现象。**

- **链接: [https://arxiv.org/pdf/2605.15864](https://arxiv.org/pdf/2605.15864)**

> **作者:** Chufan Shi; Cheng Yang; Yaokang Wu; Linhao Jin; Bo Shui; Taylor Berg-Kirkpatrick; Xuezhe Ma
>
> **备注:** ICML 2026 Spotlight
>
> **摘要:** Vision-Language Models (VLMs) often produce self-reflective statements like "let me check the figure again" during reasoning. Do such statements trigger genuine visual re-examination, or are they merely learned textual patterns? We investigate this via VisualSwap, an image-swap probing framework: after a model reasons over an image, we replace it with a visually similar but semantically different one and test whether the model notices. We introduce VS-Bench, 800 image pairs curated from MathVista, MathVerse, MathVision, and MMMU-Pro. Experiments on Qwen3-VL, Kimi-VL, and ERNIE-VL reveal a striking failure: models overwhelmingly miss the swap, with accuracy dropping by up to 60%. Counterintuitively, thinking models are nearly 3x more vulnerable than their instructed counterparts, and scaling offers no mitigation. Multi-turn user instructions restore visual grounding, but self-generated reflective statements during continuous generation do not. Attention analysis explains why: user instructions substantially elevate attention to visual tokens, whereas self-reflection does not. Current VLMs tend to say rather than actually see when claiming to perform visual re-examination. Our code and dataset are available at the project page: this https URL
>
---
#### [new 071] PerfCodeBench: Benchmarking LLMs for System-Level High-Performance Code Optimization
- **分类: cs.SE; cs.CL; cs.PL**

- **简介: 该论文提出PerfCodeBench，用于评估LLMs在高性能代码优化方面的能力，解决现有基准缺乏系统级性能优化的问题。**

- **链接: [https://arxiv.org/pdf/2605.15222](https://arxiv.org/pdf/2605.15222)**

> **作者:** Huihao Jing; Wenbin Hu; Haochen Shi; Hanyu Yang; Sirui Zhang; Shaojin Chen; Haoran Li; Yangqiu Song
>
> **摘要:** Large language models (LLMs) can often generate functionally correct code, but their ability to produce efficient implementations for performance-critical systems tasks remains limited. Existing code benchmarks mainly emphasize correctness or algorithmic problem solving, while realistic systems-level optimization is still underexplored. To address this gap, we introduce PerfCodeBench, an executable benchmark for evaluating LLMs on high-performance code optimization. The tasks require system-level implementation choices, hardware-aware optimization, and careful handling of performance bottlenecks. Each task includes executable correctness checks, a baseline implementation, and a reference optimized solution. This allows us to evaluate both correctness and runtime-oriented efficiency. Our evaluation on a broad set of state-of-the-art LLMs shows a clear gap between model-generated code and expert-optimized implementations. The gap is especially large on tasks involving parallelism and GPU operations. Current models also show weaknesses in cross-language robustness and in consistently reaching expert-level efficiency. These results suggest that performance-aware evaluation are still needed. LLMs should move beyond generating merely correct code toward producing efficient systems software. We submit the benchmark data, evaluation infrastructure, and complete logs of all LLMs-generated code at this https URL.
>
---
#### [new 072] Layer Equivalence Is Not a Property of Layers Alone: How You Test Redundancy Changes What You Find
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究Transformer层等价性测试问题，比较替换与交换两种方法的差异，揭示其对剪枝效果的影响，强调需同时评估两种指标以准确诊断。**

- **链接: [https://arxiv.org/pdf/2605.16234](https://arxiv.org/pdf/2605.16234)**

> **作者:** Gabriel Garcia
>
> **备注:** 40 pages, 8 figures, 24 tables. Code and frozen JSON logs are not public during write-up; the authors plan to open this https URL
>
> **摘要:** When researchers ask whether two transformer layers are "equivalent" for compression, they often conflate distinct tests. Replacement asks whether one layer's map can substitute for another's in place; interchange asks whether two layers approximately commute when their positions are swapped. Both are output-grounded swap-KL probes, but they need not agree: on pretrained transformers the protocol gap can change which layers look safe to prune by several-fold under the same evaluator, especially when replacement distances are high. We measure both protocols across checkpoints and architectures. On a Pythia training trajectory (410M and 1.4B), the replacement-interchange gap grows from initialization to convergence. Under one matched WikiText-2 contract at 8B scale, Qwen3-8B enters a divergent regime: interchange-guided removal is several-fold safer than replacement-guided at the same layer budgets, while Llama-3.1-8B ties the two protocols for pruning cost even though interchange KL is lower, showing metric gaps need not map one-to-one to removal. Before layer removal or merging, score both swap-KLs on the target checkpoint; the diagnostic requires only unlabeled forward passes.
>
---
#### [new 073] Fully Open Meditron: An Auditable Pipeline for Clinical LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医疗领域大模型研究，旨在解决LLM-CDSS的透明性与可审计性问题，提出Fully Open Meditron管道，确保模型训练全过程可验证。**

- **链接: [https://arxiv.org/pdf/2605.16215](https://arxiv.org/pdf/2605.16215)**

> **作者:** Xavier Theimer-Lienhard; Mushtaha El-Amin; Fay Elhassan; Sahaj Vaidya; Victor Cartier-Negadi; David Sasu; Lars Klein; Mary-Anne Hartley
>
> **备注:** Preprint. 31 pages, 10 figures. Code, models, and data: this https URL
>
> **摘要:** Clinical decision support systems (CDSS) require scrutable, auditable pipelines that enable rigorous, reproducible validation. Yet current LLM-based CDSS remain largely opaque. Most "open" models are open-weight only, releasing parameters while withholding the data provenance, curation procedures, and generation pipelines that determine model behavior. Fully Open (FO) models, which expose the complete training stack end-to-end, do not currently exist in medicine. We introduce Fully Open Meditron, the first fully open pipeline for building LLM-CDSS, comprising a clinician-audited training corpus, a reproducible data construction and training framework, and a use-aligned evaluation protocol. The corpus unifies eight public medical QA datasets into a normalized conversational format and expands coverage with three clinician-vetted synthetic extensions: exam-style QA, guideline-grounded QA derived from 46,469 clinical practice guidelines, and clinical vignettes. The pipeline enforces system-wide decontamination, gold-label resampling of teacher generations, and end-to-end validation by a four-physician panel. We evaluate using an LLM-as-a-judge protocol over expert-written clinical vignettes, calibrated against 204 human raters. We apply the recipe to five FO base models (Apertus-70B/8B-Instruct, OLMo-2-32B-SFT, EuroLLM-22B/9B-Instruct). All MeditronFO variants are preferred over their bases. Apertus-70B-MeditronFO improves +6.6 points over its base (47.2% to 53.8%) on aggregate medical benchmarks, establishing a new FO SoTA. Gemma-3-27B-MeditronFO is preferred over MedGemma in 58.6% of LLM-as-a-judge comparisons and outperforms it on HealthBench (58% vs 55.9%). These results show that fully open pipelines can achieve state-of-the-art domain-specific performance without sacrificing auditability or reproducibility.
>
---
#### [new 074] Nudging Beyond the Comfort Zone: Efficient Strategy-Guided Exploration for RLVR
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决RLVR中探索效率低的问题。提出NudgeRL框架，通过策略引导实现高效、多样化的探索，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.15726](https://arxiv.org/pdf/2605.15726)**

> **作者:** Chanuk Lee; Sangwoo Park; Minki Kang; Sung Ju Hwang
>
> **备注:** 28 pages, 7 figures
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has emerged as a scalable paradigm for improving the reasoning capabilities of large language models. However, its effectiveness is fundamentally limited by exploration: the policy can only improve on trajectories it has already sampled. While increasing the number of rollouts alleviates this issue, such brute-force scaling is computationally expensive, and existing approaches that modify the optimization objective provide limited control over what is explored. In this work, we propose NudgeRL, a framework for structured and diversity-driven exploration in RLVR. Our approach introduces Strategy Nudging, which conditions each rollout on lightweight, strategy-level contexts to induce diverse reasoning trajectories without relying on expensive oracle supervision. To effectively learn from such structured exploration, we further propose a unified objective, which decomposes the reward signal into inter- and intra-context components and incorporates a distillation objective to transfer discovered behaviors back to the base policy. Empirically, NudgeRL outperforms standard GRPO with up to 8 times larger rollout budgets, while outperforming oracle-guided RL baseline on average across five challenging math benchmarks. These results demonstrate that structured, context-driven exploration can serve as an efficient and scalable alternative to both brute-force rollout scaling and feasibility-oriented methods based on privileged information. Our code is available at this https URL.
>
---
#### [new 075] From I/O to Code with Discovery Agent
- **分类: cs.LG; cs.AI; cs.CL; cs.SE**

- **简介: 该论文属于IO2Code任务，解决从输入输出行为生成代码的问题。提出DIO-Agent，通过进化搜索和LLM mutation实现高效代码合成。**

- **链接: [https://arxiv.org/pdf/2605.15334](https://arxiv.org/pdf/2605.15334)**

> **作者:** Yihong Dong; Jiaru Qian; Haoran Zhang; Peixu Wang; Binhua Li; Zhi Jin; Yongbin Li; Ge Li; Xiaokang Yang; Xue Jiang
>
> **摘要:** The automatic synthesis of a program from any form of specification is regarded as a holy grail of computer science. Fueled by LLMs, NL2Code has achieved tremendous success, yet the fundamentally more challenging task of synthesizing programs from input-output behavior, which we refer to as IO2Code, remains largely unsolved. Whereas NL2Code can exploit the semantic alignment between natural language and code acquired during pretraining, IO2Code requires recovering underlying principles from concrete computational behavior, navigating a vast and underspecified hypothesis space. To address this, we propose DIO-Agent, a discovery agent for IO2Code. Our method frames IO2Code as an evolutionary search over discrete program space, in which an LLM serves as the mutation operator and concrete error signals from execution guide each mutation. To prevent the search from wandering into structurally complex yet incorrect dead ends, we introduce the Transformation Priority Premise as a mutation prior that biases the LLM toward the simplest hypothesis consistent with current evidence, progressively escalating from constants to conditionals to iteration only when simpler constructs are insufficient. To facilitate systematic study, we further construct an IO2CodeBench spanning multiple difficulty levels. Extensive experiments show that DIO-Agent consistently outperforms both traditional program-by-example method and SOTA evolution-agent baselines across all difficulty levels and various LLMs, while substantially surpassing test-time scaling strategies with equivalent sampling budgets.
>
---
#### [new 076] Effective Harness Engineering for Algorithm Discovery with Coding Agents
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于算法发现任务，研究如何设计有效的执行框架（harness），解决生成算法的质量与数量平衡、评估漏洞处理及安全并行执行问题。**

- **链接: [https://arxiv.org/pdf/2605.15221](https://arxiv.org/pdf/2605.15221)**

> **作者:** Yoichi Ishibashi; Taro Yano; Masafumi Oyamada
>
> **摘要:** AlphaEvolve and FunSearch have demonstrated the potential of combining large language models (LLMs) with evolutionary search for automated algorithm discovery. However, discovery success is shaped not only by model capability but also significantly by the design of the execution infrastructure, i.e., the harness. This paper investigates effective harness design through three questions: under a fixed token budget, is it better to produce many algorithms with brief thought or fewer algorithms with deeper thought? How should the harness handle evaluation hacks, where generated programs exploit the scoring function? And how can agents that require full filesystem access execute safely in parallel? Using Vesper, an algorithm discovery framework that incorporates harness improvements addressing these questions, we evaluate on Circle Packing under the same token budget. Interestingly, generating fewer algorithms while thinking more deeply about each one achieved higher scores. That is, scaling the quality of each individual is more budget-efficient than scaling the number of evolutionary generations. Surprisingly, more capable models produced evaluation hacks at higher rates, making hack detection increasingly necessary as models scale.
>
---
#### [new 077] STS: Efficient Sparse Attention with Speculative Token Sparsity
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决大模型推理中的计算与内存瓶颈问题。通过引入STS机制，利用草稿模型预测重要token，实现高效稀疏注意力计算，提升推理速度并保持高精度。**

- **链接: [https://arxiv.org/pdf/2605.15508](https://arxiv.org/pdf/2605.15508)**

> **作者:** Ceyu Xu; Jiangnan Yu; Yongji Wu; Yuan Xie
>
> **备注:** 14 pages, 12 figures
>
> **摘要:** The quadratic complexity of attention imposes severe memory and computational bottlenecks on Large Language Model (LLM) inference. This challenge is particularly acute for emerging agentic applications that require processing multi-million token sequences. We propose STS, a sparse attention mechanism that requires no model retraining. STS leverages the key insight that tokens identified as important by a smaller draft model are highly predictive of important tokens for a larger target model. By integrating into speculative decoding frameworks, STS repurposes the draft model's attention scores to dynamically construct a token-and-head-wise sparsity mask. This mask effectively prunes the expensive attention computation in the target LLM. Our evaluation shows that STS achieves a 2.67x speedup operating at approximately 90% sparsity on representative benchmark NarrativeQA, maintaining negligible accuracy degradation compared to dense attention. STS establishes a new state-of-the-art on the sparsity-accuracy trade-off, outperforming prior techniques by enabling higher sparsity levels for a given accuracy budget.
>
---
#### [new 078] DeepSlide: From Artifacts to Presentation Delivery
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出DeepSlide，解决AI生成幻灯片时忽视演讲过程的问题。它整合多智能体系统，优化从内容规划到演讲训练的全流程，提升叙事流畅度与节奏精度。**

- **链接: [https://arxiv.org/pdf/2605.15202](https://arxiv.org/pdf/2605.15202)**

> **作者:** Ming Yang; Zhiwei Zhang; Jiahang Li; Haoseng Liu; Yuzheng Cai; Weiguo Zheng
>
> **备注:** 37 pages,10 figures,9 tables
>
> **摘要:** Presentations are a primary medium for scholarly communication, yet most AI slide generators optimize the artifact (a visually plausible deck) while under-optimizing the delivery process (pacing, narrative, and presentation preparation). We present DeepSlide, a human-in-the-loop multi-agent system that supports preparing the full presentation process, from requirement elicitation and time-budgeted narrative planning, to evidence-grounded slide--script generation, attention augmentation, and rehearsal support. DeepSlide integrates (i) a controllable logical-chain planner with per-node time budgets, (ii) a lightweight content-tree retriever for grounding, (iii) Markov-style sequential rendering with style inheritance, and (iv) sandboxed execution with minimal repair to ensure renderability. We further introduce a dual-scoreboard benchmark that cleanly separates static artifact quality from dynamic delivery excellence. Across 20 domains and diverse audience profiles, DeepSlide matches strong baselines on artifact quality while consistently achieving larger gains on delivery metrics, improving narrative flow, pacing precision, and slide--script synergy with clearer attention guidance.
>
---
#### [new 079] Reasoners or Translators? Contamination-aware Evaluation and Neuro-Symbolic Robustness in Tax Law
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于法律推理任务，旨在解决LLM在税务法中是否具备真实推理能力的问题。通过实验与测试，对比LLM与神经符号系统，验证其可靠性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.16052](https://arxiv.org/pdf/2605.16052)**

> **作者:** Parisa Kordjamshidi; Samer Aslan; Madhavan Seshadri; Leslie Barrett; Enrico Santus
>
> **摘要:** Recent advances in large language models (LLMs) have significantly enhanced automated legal reasoning. Yet, it remains unclear whether their performance reflects genuine legal reasoning ability or artifacts of data contamination. We present a comprehensive empirical study of tax law reasoning approaches and implement a contamination detection protocol to rigorously assess LLM reliability. We show that performance can be inflated by contamination. Building on this analysis, we conduct a systematic evaluation, comparing monolithic LLMs with hybrid systems that translate statutory text into formal representations and delegate inference to symbolic solvers. We build a novel test suite designed to probe generalization to unseen documents via case and rule variations. Our findings indicate that legal reasoning is inherently compositional and that neuro-symbolic frameworks offer a more reliable and robust foundation for legal AI, as well as improved generalization to unobserved situations.
>
---
#### [new 080] AI-Mediated Communication Can Steer Collective Opinion
- **分类: cs.CY; cs.AI; cs.CL; cs.LG; cs.SI**

- **简介: 该论文研究AI在人际沟通中对集体意见的影响，属于社会计算任务。它探讨AI如何通过文本编辑引入偏见，并分析其对群体意见的放大效应。**

- **链接: [https://arxiv.org/pdf/2605.16245](https://arxiv.org/pdf/2605.16245)**

> **作者:** Stratis Tsirtsis; Kai Rawal; Chris Russell; Brent Mittelstadt; Sandra Wachter
>
> **摘要:** Generative artificial intelligence (AI) is increasingly integrated into the online platforms where humans exchange opinions; large language models (LLMs) now polish users' posts on LinkedIn and provide context for content shared on X. While prior work has shown that AI can express biased opinions and shape individuals' opinions during human-AI interactions, less attention has been paid to its influence on collective opinion formation when mediating human-to-human communication. We address this gap via a combination of empirical and theoretical analyses. We show empirically that LLMs from multiple popular families introduce directional biases when instructed to edit human-written texts on contested topics, for example, nudging texts in favor of gun control and against atheism. Building on this observation, we introduce a mathematical model of opinion dynamics in which an AI system sits between users on a social network, transforming the opinions they express and perceive. By analytically characterizing the equilibrium of this model and performing simulations on real social network data, we show that biases introduced by AI in human-to-human communication can be amplified through the network and shift collective opinion in their direction. In light of these findings, we investigate whether such biases are controllable by online platforms. We audit the "Explain this post" feature on X and find evidence of pro-life bias in Grok's outputs on abortion-related content, which we trace back to specific design choices. We conclude with a discussion of the broader implications of our findings in relation to ongoing legislative efforts in the European Union.
>
---
## 更新

#### [replaced 001] When Importance Sampling Misallocates Credit: Asymmetric Ratios for Outcome-Supervised RL
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，针对OSRL中因重要性采样导致的令牌权重不平衡问题，提出ASPO方法进行修正，提升训练稳定性和性能。**

- **链接: [https://arxiv.org/pdf/2510.06062](https://arxiv.org/pdf/2510.06062)**

> **作者:** Jiakang Wang; Runze Liu; Qingpeng Cai; Lei Lin; Wenping Hu; Xiu Li; Fuzheng Zhang; Guorui Zhou; Kun Gai; Ling Pan
>
> **摘要:** Reinforcement learning (RL) has shown great promise in large language models (LLMs) post-training, which typically rely on token-level clipping to maintain stability during optimization. Despite the empirical success of GRPO-style methods, we identify a fundamental and previously overlooked challenge in this popular Outcome-Supervised RL (OSRL) paradigm. We reveal that in OSRL, where advantages are shared across tokens within a response, importance sampling (IS) ratios deviate from their traditional purpose of distribution correction as in classic RL, which become token-level weights that allocate the shared advantage signal across tokens. We show that this hidden role shift induces a critical mismatch for positive-advantage tokens, leading to unbalanced token weighting between positive and negative tokens. Specifically, it suppresses the update of underrepresented tokens that are lagging behind, while over-amplifying already high-probability tokens. This mismatch results in rich-get-richer dynamics that over-reinforce confident tokens, weaken catch-up learning that drive entropy collapse, excessive repetition, and premature convergence. To address this, we propose Asymmetric Importance Sampling Policy Optimization (ASPO), a simple yet effective strategy that reverses the ratio-induced weighting of positive-advantage tokens, while stabilizing extreme updates and maintaining gradient flow. This mismatch correction aligns their update direction with the learning dynamics of negative ones. Comprehensive experiments across math reasoning and coding benchmarks demonstrate that ASPO significantly mitigates entropy collapse, improves training stability, and enhances performance over strong GRPO-based baselines. Our analysis provides new insights into the role of token-level weighting in OSRL and highlights the critical importance of correcting ratio-induced weighting in LLM RL.
>
---
#### [replaced 002] DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于深度研究任务，解决开放性长文本生成问题。提出RLER方法，构建动态评估标准，训练出DR Tulu-8B模型，在多个领域表现优于现有模型。**

- **链接: [https://arxiv.org/pdf/2511.19399](https://arxiv.org/pdf/2511.19399)**

> **作者:** Rulin Shao; Akari Asai; Shannon Zejiang Shen; Hamish Ivison; Varsha Kishore; Jingming Zhuo; Xinran Zhao; Molly Park; Samuel G. Finlayson; David Sontag; Tyler Murray; Sewon Min; Pradeep Dasigi; Luca Soldaini; Faeze Brahman; Wen-tau Yih; Tongshuang Wu; Luke Zettlemoyer; Yoon Kim; Hannaneh Hajishirzi; Pang Wei Koh
>
> **备注:** ICML 2026
>
> **摘要:** Deep research agents perform multi-step research to produce long-form, well-attributed answers. However, most open deep research agents are trained on easily verifiable short-form QA tasks via reinforcement learning with verifiable rewards, which does not extend to realistic long-form tasks. We address this with Reinforcement Learning with Evolving Rubrics (RLER), where rubrics are constructed and maintained to co-evolve with the policy model during training. This allows the rubrics to incorporate newly explored information from search and contrasting model responses, enabling better fact checking and more discriminative on-policy feedback. Using RLER, we develop Deep Research Tulu (DR Tulu-8B), the first fully open model that is directly trained for open-ended, long-form deep research. Across four long-form deep research benchmarks in science, healthcare, and general domains, DR Tulu substantially outperforms existing open deep research agents (by 15.6% over Tongyi DR on average) and matches or exceeds proprietary deep research agents (by 0.7% over OpenAI DR on average), while being significantly smaller and cheaper per query (1000x cheaper than OpenAI DR per query).
>
---
#### [replaced 003] Fitting Is Not Enough: Smoothness in Extremely Quantized LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型量化任务，旨在解决极端低比特量化导致的平滑性退化问题。通过分析平滑性损失，提出保持平滑性的方法以提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.08894](https://arxiv.org/pdf/2605.08894)**

> **作者:** Yuzhuang Xu; Xu Han; Yuxuan Li; Pengzhan Li; Wanxiang Che
>
> **备注:** 19 pages, 4 tables, 14 figures
>
> **摘要:** Large language models (LLMs) achieve strong performance but incur high deployment costs, motivating extremely low-bit but lossy quantization. Existing quantization algorithms mainly focus on improving the numerical accuracy of forward computation to eliminate performance degradation. In this paper, we show that extremely quantized LLMs suffer from systematic smoothness degradation beyond numerical precision loss. Through a smoothness proxy, we observe that such degradation becomes increasingly severe as the quantization bit-width decreases. Furthermore, based on sequence neighborhood modeling, we find that quantized models exhibit a rapid reduction of effective token candidates within the prediction neighborhood, which directly leads to a sparser decoding tree and degraded generation quality. To validate it, we introduce a simple smoothness-preserving principle in both post-training quantization and quantization-aware training, and demonstrate that preserving smoothness brings additional gains beyond numerical accuracy. The core goal of this paper is to highlight smoothness preservation as an important design consideration for future extreme quantization methods. Code is available at this https URL.
>
---
#### [replaced 004] Beyond Forgetting: Machine Unlearning Elicits Controllable Side Behaviors and Capabilities
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究机器遗忘中的可控行为与能力提升问题。通过重构表示方法，实现对模型行为的控制和能力增强。属于自然语言处理领域。**

- **链接: [https://arxiv.org/pdf/2601.21702](https://arxiv.org/pdf/2601.21702)**

> **作者:** Tien Dang; The-Hai Nguyen; Dinh Mai Phuong; Nguyen Minh Phuong; Anh Bui; Hoang Thanh-Tung; Le-Minh Nguyen; Naoya Inoue
>
> **备注:** 36 pages, 19 tables, 9 figures
>
> **摘要:** We consider Representation Misdirection (RM), a class of large language model (LLM) unlearning methods that achieve forgetting by redirecting the forget-representations, that is, latent representations of forget-samples, toward a target vector. Despite being important, the roles of the target vector used in RM, however, remain underexplored. Here, we approach and revisit RM through the lens of the Linear Representation Hypothesis. Specifically, if one can identify a one-dimensional representation corresponding to a high-level concept, the Linear Representation Hypothesis enables linear operations on this concept vector within the forget-representation space. Under this view, we hypothesize that, beyond forgetting, machine unlearning via RM elicits controllable emergent side behaviors and stronger side capabilities corresponding to the high-level concept. Our hypothesis is empirically validated across a wide range of tasks, including behavioral control (e.g., controlling unlearned models' truthfulness, sentiment, refusal, and language) and capability enhancement (e.g., improving unlearned models' in-context learning (ICL) capability). Our findings reveal that this phenomenon could be either a hidden risk if misused or a mechanism that can be harnessed for developing unlearned models that require stronger capabilities and controllable behaviors.
>
---
#### [replaced 005] RapidUn: Influence-Driven Parameter Reweighting for Efficient Large Language Model Unlearning
- **分类: cs.CL**

- **简介: 该论文属于语言模型遗忘任务，旨在高效移除特定数据影响。提出RapidUn框架，通过影响驱动的参数重加权实现快速且稳定的模型遗忘。**

- **链接: [https://arxiv.org/pdf/2512.04457](https://arxiv.org/pdf/2512.04457)**

> **作者:** Guoshenghui Zhao; Huawei Lin; Weijie Zhao
>
> **备注:** Code available at: this https URL
>
> **摘要:** Removing specific data influence from large language models (LLMs) remains challenging, as retraining is costly and existing approximate unlearning methods are often unstable. The challenge is exacerbated when the forget set is small or imbalanced. We introduce RapidUn, an influence-driven and parameter-efficient unlearning framework. It first estimates per-sample influence through a fast estimation module, then maps these scores into adaptive update weights that guide selective parameter updates -- forgetting harmful behavior while retaining general knowledge. On Mistral-7B and Llama-3-8B across Dolly-15k and Alpaca-57k, RapidUn achieves up to 100 times higher efficiency than full retraining and consistently outperforms Fisher, GA, and LoReUn on both in-distribution and out-of-distribution forgetting. These results establish influence-guided parameter reweighting as a scalable and interpretable paradigm for LLM unlearning.
>
---
#### [replaced 006] Wiki Dumps to Training Corpora: South Slavic Case
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语料库构建任务，旨在解决从维基数据中提取高质量文本的问题。通过分阶段处理和过滤冗余内容，生成适用于南斯拉夫语言研究的语料库。**

- **链接: [https://arxiv.org/pdf/2604.25384](https://arxiv.org/pdf/2604.25384)**

> **作者:** Mihailo Škorić; Cosimo Palma
>
> **摘要:** This paper presents a pipeline designed to transform raw Wikimedia dumps into quality textual corpora for seven South Slavic languages. The work is divided into two major phases. The first involves extracting and cleaning text from raw dumps of Wikipedia, Wikisource, Wikibooks, Wikinews, and Wikiquote. This step requires careful handling of raw wiki markup to isolate, first of all, textual articles, and then usable natural language text within them. The second phase addresses the challenge of questionable or low-quality articles, which are often generated from databases or structured knowledge bases. These articles are characterised by repetitive patterns, generic phrasing, and minimal to no original content. To mitigate their impact, a n-gram-based filtering strategy was employed to detect high levels of textual redundancy between articles and then remove such articles from the corpora entirely. The resulting datasets aim to provide linguistically rich texts suitable for training language models or conducting comparative research across South Slavic languages. By combining systematic extraction with quality control, this work contributes to the creation of reliable, high-information corpora that reflect the authentic cultural contexts of languages. While focused on the South Slavic case in the paper, the approach is mostly language-agnostic and can be generalised to other languages.
>
---
#### [replaced 007] Prompt Stability Scoring for Text Annotation with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于文本标注任务，解决语言模型输出稳定性问题。通过提出Prompt Stability Score（PSS）框架，评估并提升标注的可重复性。**

- **链接: [https://arxiv.org/pdf/2407.02039](https://arxiv.org/pdf/2407.02039)**

> **作者:** Christopher Barrie; Elli Palaiologou; Petter Törnberg
>
> **备注:** 39 pages, 5 figures
>
> **摘要:** Researchers are increasingly using language models (LMs) for text annotation. These approaches rely only on a prompt telling the model to return a given output according to a set of instructions. The reproducibility of LM outputs may nonetheless be vulnerable to small changes in the prompt design. This calls into question the replicability of classification routines. To tackle this problem, researchers have typically tested a variety of semantically similar prompts to determine what we call ``prompt stability." These approaches remain ad-hoc and task specific. In this article, we propose a general framework for diagnosing prompt stability by adapting traditional approaches to intra- and inter-coder reliability scoring. We call the resulting metric the Prompt Stability Score (PSS) and provide a Python package \texttt{promptstability} for its estimation. Using six different datasets and twelve outcomes, we classify $\sim$3.1m rows of data and $\sim$300m input tokens to: a) diagnose when prompt stability is low; and b) demonstrate the functionality of the package. We conclude by providing best practice recommendations for applied researchers.
>
---
#### [replaced 008] LaCy: What Small Language Models Can and Should Learn is Not Just a Question of Loss
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决小语言模型事实性生成错误的问题。通过结合损失与事实信号，提出LaCy方法，优化模型学习与调用大模型的决策。**

- **链接: [https://arxiv.org/pdf/2602.12005](https://arxiv.org/pdf/2602.12005)**

> **作者:** Szilvia Ujváry; Louis Béthune; Pierre Ablin; João Monteiro; Marco Cuturi; Michael Kirchhof
>
> **备注:** 40 pages, 26 figures, 10 tables, preprint. v3: new results for RAG, ablations and additional analysis
>
> **摘要:** Language models have consistently grown to compress more world knowledge into their parameters, but the knowledge that can be pretrained into them is upper-bounded by their parameter size. Especially the capacity of Small Language Models (SLMs) is limited, leading to factually incorrect generations. This problem is often mitigated by giving the SLM access to an outside source: the ability to query a larger model, documents, or a database. Under this setting, we study the fundamental question of \emph{which tokens an SLM can and should learn} during pretraining, versus \emph{which ones it should delegate} via a \texttt{<CALL>} token. We find that this is not simply a question of loss: although the loss is predictive of whether a predicted token mismatches the ground-truth, it is insufficient for identifying which predictions would actually lead to factual or semantically invalid continuations. Some high-loss tokens correspond to \emph{acceptable} alternative continuations of a pretraining document and therefore should not trigger a \texttt{<CALL>}. This suggests that learnability cannot be characterized from loss alone, but requires additional domain-specific signals about the role of a token in the sentence. In Wikipedia-like domains, we show that augmenting the loss signal with lightweight grammatical information from a spaCy parser substantially improves delegation decisions. Based on this insight, we propose LaCy, a novel pretraining method that combines loss with factuality signals to decide which tokens an SLM should learn. Our experiments demonstrate that LaCy models successfully learn which tokens to predict and when to call for help. This results in higher FactScores when generating in a cascade with a bigger model and outperforms Rho or LLM-judge trained SLMs, while being simpler and cheaper.
>
---
#### [replaced 009] Large Language Models Could Be Rote Learners
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，旨在解决基准数据污染导致的性能虚高问题。通过分析发现LLMs存在机械记忆与真实能力并存现象，并提出TrinEval框架以区分两者。**

- **链接: [https://arxiv.org/pdf/2504.08300](https://arxiv.org/pdf/2504.08300)**

> **作者:** Yuyang Xu; Renjun Hu; Haochao Ying; Jian Wu; Xing Shi; Wei Lin
>
> **备注:** Work in Progress
>
> **摘要:** Benchmark-based evaluation, e.g., multiple-choice questions (MCQs) and open-ended questions (OEQs), is widely used for evaluating Large Language Models (LLMs), yet their reliability is undermined by benchmark contamination. When pre-exposed to the testing benchmark during training, less capable LLMs have been found to achieve inflated performance, thereby yielding erroneous results in LLM evaluation. In this study, we reframe contamination as an inherent aspect of learning and seek to disentangle and expose genuine capability acquisition from superficial memorization in LLM evaluation. Following this, firstly, by analyzing model performance under different memorization conditions of MCQs, we uncover a counterintuitive trend: LLMs perform worse on memorized benchmarks than on non-memorized ones, indicating the coexistence of two learning phenomena, i.e., rote memorization and genuine capability learning. To disentangle them, we propose TrinEval, a novel evaluation framework that reformulates MCQs into an alternative knowledge-centric trinity format, reducing memorization while preserving inherent knowledge, enabling the evaluation of genuine capability in the presence of memorization. Extensive experiments validate the effectiveness and robustness of TrinEval in reformulating benchmarks, and the evaluation results further reveal that mainstream LLMs rely on rote memorization for an average of 19.6% of knowledge points across the MMLU and the GSM8K dataset.
>
---
#### [replaced 010] From Guidelines to Guarantees: A Graph-Based Evaluation Harness for Domain-Specific Evaluation of LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语言模型评估任务，旨在解决领域专用模型评估基准不足的问题。通过构建知识图谱动态生成评测数据，提升评估的全面性和适应性。**

- **链接: [https://arxiv.org/pdf/2508.20810](https://arxiv.org/pdf/2508.20810)**

> **作者:** Jessica M. Lundin; Usman Nasir Nakakana; Guillaume Chabot-Couture
>
> **摘要:** Rigorous evaluation of domain-specific language models requires benchmarks that are comprehensive, contamination-resistant, and maintainable. Static, manually curated datasets do not satisfy these properties. We present a graph-based evaluation harness that transforms structured clinical guidelines into a queryable knowledge graph and dynamically instantiates evaluation queries via graph traversal. The framework provides three guarantees: (1) complete coverage of guideline relationships; (2) surface-form contamination resistance through combinatorial variation; and (3) validity inherited from expert-authored graph structure. Applied to the WHO IMCI guidelines, the harness generates clinically grounded multiple-choice questions spanning symptom recognition, treatment, severity classification, and follow-up care. Evaluation across five language models reveals systematic capability gaps. Models perform well on symptom recognition but show lower accuracy on treatment protocols and clinical management decisions. The framework supports continuous regeneration of evaluation data as guidelines evolve and generalizes to domains with structured decision logic. This provides a scalable foundation for evaluation infrastructure.
>
---
#### [replaced 011] Key-Value Means: Transformers with Expandable Block-Recurrent Compressed Memory
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出KVM，一种可扩展的块循环注意力机制，用于Transformer模型。旨在解决长序列处理中的内存和计算效率问题，通过引入可增长的KV缓存实现高效长上下文建模。**

- **链接: [https://arxiv.org/pdf/2605.09877](https://arxiv.org/pdf/2605.09877)**

> **作者:** Daniel Goldstein; Eugene Cheah
>
> **摘要:** We present Key-Value Means ("KVM"), a novel block-recurrence for attention that can accommodate either fixed-size or growing state. Equipping a strong transformer baseline with fixed-size KVM attention layers yields a strong $O(N)$ chunked RNN, while adding only an insignificant number of new parameters. We train a transformer with a growable KVM cache and show it performs competitively on long-context tests with only subquadratic prefill time and sublinear state growth. KVM is implementable with standard operations and without custom kernels, and supports chunk-wise parallelizable training and prefill. It provides many of the benefits of both traditional transformers (expandable context memory, chunk-wise parallelizable training and prefill) and linear RNNs in a single unified package. It can be used on every layer, saving KV-cache memory, and allowing a continuous range of choices of prefill time complexity between $O(N)$ and $O(N^2)$. It can also be implemented in a hybrid solution in tandem with LRNN layers in place of traditional attention, to supplement the LRNN with improved sublinear memory growth context length usage and long context decoding. We release our code at this https URL and trained models at this https URL under the Apache 2.0 license.
>
---
#### [replaced 012] ADMIT: Few-shot Knowledge Poisoning Attacks on RAG-based Fact Checking
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文属于知识 poisoning 任务，旨在解决 RAG 系统在事实核查中易受攻击的问题。提出 ADMIT 方法，在无模型访问情况下实现高效攻击。**

- **链接: [https://arxiv.org/pdf/2510.13842](https://arxiv.org/pdf/2510.13842)**

> **作者:** Yutao Wu; Xiao Liu; Yinghui Li; Yifeng Gao; Yifan Ding; Jiale Ding; Xiang Zheng; Xingjun Ma
>
> **摘要:** Knowledge poisoning poses a critical threat to Retrieval-Augmented Generation (RAG) systems by injecting adversarial content into knowledge bases, tricking Large Language Models (LLMs) into producing attacker-controlled outputs grounded in manipulated context. Prior work highlights LLMs' susceptibility to misleading or malicious retrieved content. However, real-world fact-checking scenarios are more challenging, as credible evidence typically dominates the retrieval pool. To investigate this problem, we extend knowledge poisoning to the fact-checking setting, where retrieved context includes authentic supporting or refuting evidence. We propose \textbf{ADMIT} (\textbf{AD}versarial \textbf{M}ulti-\textbf{I}njection \textbf{T}echnique), a few-shot, semantically aligned poisoning attack that flips fact-checking decisions and induces deceptive justifications, all without access to the target LLMs, retrievers, or token-level control. Extensive experiments show that ADMIT transfers effectively across 4 retrievers, 11 LLMs, and 4 cross-domain benchmarks, achieving an average attack success rate (ASR) of 86\% at an extremely low poisoning rate of $0.93 \times 10^{-6}$, and remaining robust even in the presence of strong counter-evidence. Compared with prior state-of-the-art attacks, ADMIT improves ASR by 11.2\% across all settings, exposing significant vulnerabilities in real-world RAG-based fact-checking systems.
>
---
#### [replaced 013] Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型训练任务，旨在解决模型在推理时缺乏准确置信度估计的问题。通过引入校准奖励机制，提升模型的准确性与置信度预测能力。**

- **链接: [https://arxiv.org/pdf/2507.16806](https://arxiv.org/pdf/2507.16806)**

> **作者:** Mehul Damani; Isha Puri; Stewart Slocum; Idan Shenfeld; Leshem Choshen; Yoon Kim; Jacob Andreas
>
> **摘要:** When language models (LMs) are trained via reinforcement learning (RL) to generate natural language "reasoning chains", their performance improves on a variety of difficult question answering tasks. Today, almost all successful applications of RL for reasoning use binary reward functions that evaluate the correctness of LM outputs. Because such reward functions do not penalize guessing or low-confidence outputs, they often have the unintended side-effect of degrading calibration and increasing the rate at which LMs generate incorrect responses (or "hallucinate") in other problem domains. This paper describes RLCR (Reinforcement Learning with Calibration Rewards), an approach to training reasoning models that jointly improves accuracy and calibrated confidence estimation. During RLCR, LMs generate both predictions and numerical confidence estimates after reasoning. They are trained to optimize a reward function that augments a binary correctness score with a Brier score -- a scoring rule for confidence estimates that incentivizes calibrated prediction. We first prove that this reward function (or any reward function that uses a bounded, proper scoring rule) yields models whose predictions are both accurate and well-calibrated. We next show that across diverse datasets, RLCR substantially improves calibration with no loss in accuracy, on both in-domain and out-of-domain evaluations -- outperforming both ordinary RL training and classifiers trained to assign post-hoc confidence scores. While ordinary RL hurts calibration, RLCR improves it. Finally, we demonstrate that verbalized confidence can be leveraged at test time to improve accuracy and calibration via confidence-weighted scaling methods. Our results show that explicitly optimizing for calibration can produce more generally reliable reasoning models. Code, models, and further info is available at this https URL.
>
---
#### [replaced 014] CryptoBench: A Dynamic Benchmark for Expert-Level Evaluation of LLM Agents in Cryptocurrency
- **分类: cs.CL**

- **简介: 该论文提出CryptoBench，用于评估LLM在加密货币领域的表现。解决LLM在动态、高对抗环境下的分析与预测能力问题。通过设计动态任务集，评估模型的数据获取与预测能力。**

- **链接: [https://arxiv.org/pdf/2512.00417](https://arxiv.org/pdf/2512.00417)**

> **作者:** Jiacheng Guo; Suozhi Huang; Zixin Yao; Yifan Zhang; Yifu Lu; Jiashuo Liu; Zihao Li; Nicholas Deng; Qixin Xiao; Jia Tian; Kanghong Zhan; Tianyi Li; Xiaochen Liu; Jason Ge; Chaoyang He; Kaixuan Huang; Lin Yang; Wenhao Huang; Mengdi Wang
>
> **摘要:** This paper introduces CryptoBench, the first expert-curated, dynamic benchmark designed to rigorously evaluate the real-world capabilities of Large Language Model (LLM) agents in the uniquely demanding and fast-paced cryptocurrency domain. Unlike general-purpose agent benchmarks for search and prediction, professional crypto analysis presents specific challenges: \emph{extreme time-sensitivity}, \emph{a highly adversarial information environment}, and the critical need to synthesize data from \emph{diverse, specialized sources}, such as on-chain intelligence platforms and real-time Decentralized Finance (DeFi) dashboards. CryptoBench thus serves as a much more challenging and valuable scenario for LLM agent assessment. To address these challenges, we constructed a live, dynamic benchmark featuring 50 questions per month, expertly designed by crypto-native professionals to mirror actual analyst workflows. These tasks are rigorously categorized within a four-quadrant system: Simple Retrieval, Complex Retrieval, Simple Prediction, and Complex Prediction. This granular categorization enables a precise assessment of an LLM agent's foundational data-gathering capabilities alongside its advanced analytical and forecasting skills. Our evaluation of ten LLMs, both directly and within an agentic framework, reveals a performance hierarchy and uncovers a failure mode. We observe a \textit{retrieval-prediction imbalance}, where many leading models, despite being proficient at data retrieval, demonstrate a pronounced weakness in tasks requiring predictive analysis. This highlights a problematic tendency for agents to appear factually grounded while lacking the deeper analytical capabilities to synthesize information.
>
---
#### [replaced 015] Structure-BiEval: A Self-Supervised, Dual-Track Framework for Decoupling Structure and Content in LLM Evaluation for Web Information Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Structure-BiEval框架，解决Web数据结构评估问题，通过解耦结构与内容，实现无标注的量化评估。**

- **链接: [https://arxiv.org/pdf/2601.19923](https://arxiv.org/pdf/2601.19923)**

> **作者:** Boxiang Zhao; Qince Li; Zhonghao Wang; Zelin Cao; Yi Wang; Peng Cheng; Bo Lin
>
> **摘要:** As Large Language Models (LLMs) evolve into the core of Web-based autonomous agents and complex Web Information Systems, their ability to faithfully translate natural language into rigorous structured formats has become paramount, as this capability is critical for Web API invocation and data exchange. However, evaluating this structural fidelity in Web-native payloads remains a challenge: traditional text metrics fail to capture topological consistency in semi-structured Web data, while manual evaluation is prohibitively costly. To address this, we propose Structure-BiEval, a novel self-supervised framework for quantitative, annotation-free assessment tailored for Web data engineering. By leveraging deterministic Intermediate Representations, our framework effectively decouples structure from content, utilizing Content Semantic Accuracy and Normalized Tree Edit Distance as precise metrics. We empirically benchmark 15 state-of-the-art LLMs across dual Web structural topologies, namely Hierarchical Data (Web backend payloads) and Tabular Data (Web frontend presentation). The results reveal substantial variability in structural performance, including cases where mid-sized models unexpectedly outperform larger counterparts in Web data formatting. Furthermore, our findings show that deep recursive nesting poses a consistent challenge for Web agents across varying parameter scales.
>
---
#### [replaced 016] Hide to See: Reasoning-prefix Masking for Visual-anchored Thinking in VLM Distillation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型（VLM）的蒸馏任务，旨在解决学生模型在长推理过程中遗忘视觉信息的问题。通过引入视觉锚定的推理前缀掩码策略，提升学生模型对视觉证据的利用能力。**

- **链接: [https://arxiv.org/pdf/2605.11651](https://arxiv.org/pdf/2605.11651)**

> **作者:** Seonghoon Yu; Dongjun Nam; Byung-Kwan Lee; Jeany Son
>
> **备注:** Pre-print
>
> **摘要:** Recent think-answer approaches in VLMs, such as Qwen3-VL-Thinking, boost reasoning performance by leveraging intermediate thinking steps before the final answer, but their computational cost becomes substantial, especially for larger VLMs. To distill such capabilities into compact think-answer VLMs, a primary objective is to improve the student's ability to utilize visual evidence throughout its reasoning trace, as long think-answer traces suffer from visual forgetting issues. To this end, we introduce a novel think-answer distillation framework that encourages the student to anchor its thinking on visual information by masking the student's salient reasoning prefixes. To compensate for such masked textual cues, the student is encouraged to rely more on visual evidence as an alternative source of information during distillation. Our masking strategies include: 1) token-wise salient reasoning-prefix masking, which masks high-influence reasoning prefixes selectively for each next-token prediction, and 2) self-paced masking budget scheduling, which gradually increases the masking scale according to distillation difficulty, measured by the discrepancy between teacher--student distributions. In the distillation phase, the student is guided by our salient reasoning-prefix mask, which blocks both future tokens and salient reasoning cues, in place of the standard causal mask used for auto-regressive language modeling. Experimental results show that our approach outperforms recent open-source VLMs, VLM distillation, and self-distillation methods on multimodal reasoning benchmarks, while further analyzes confirm enhanced visual utilization along the student thinking process.
>
---
#### [replaced 017] A Scalable Multi-LLM Collaboration System with Retrieval-based Selection and Exploration-Exploitation-Driven Enhancement
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SMCS系统，解决多大模型协作的可扩展性问题，通过动态选择和增强机制提升性能。属于多模型协作任务。**

- **链接: [https://arxiv.org/pdf/2507.14200](https://arxiv.org/pdf/2507.14200)**

> **作者:** Shengji Tang; Jianjian Cao; Weihao Lin; Jiale Hong; Bo Zhang; Shuyue Hu; Lei Bai; Tao Chen; Wanli Ouyang; Peng Ye
>
> **摘要:** Existing multi-LLM collaboration systems often encounter scalability challenges when integrating new LLMs and tasks, leading to suboptimal performance. To address this, we propose SMCS, a Scalable Multi-LLM Collaboration System designed to effectively coordinate multiple open-source LLMs. The system consists of two core components: a Retrieval-based Prior Selection (RPS) module, which dynamically selects the most suitable LLMs for each input, and an Exploration-Exploitation-Driven Posterior Enhancement (EPE) module, which fosters response diversity and selects high-quality outputs through a hybrid scoring mechanism. Experiments on eight mainstream benchmarks validate the effectiveness of our system: by integrating fifteen open-source LLMs, SMCS outperforms prevailing closed-source LLMs, e.g., GPT-4.1(+5.36%) and GPT-o3-mini(+5.28%) across multiple tasks. Remarkably, it even exceeds the average of best results on different datasets with open-source LLMs (+2.86%), significantly advancing the empirical performance frontier of open-source collaboration. The code is released at this https URL.
>
---
#### [replaced 018] The MediaSpin Dataset: Post-Publication News Headline Edits Annotated for Media Bias
- **分类: cs.CL**

- **简介: 该论文提出MediaSpin数据集，用于研究新闻标题编辑中的媒体偏见。属于偏见检测任务，旨在分析编辑行为对内容的影响及社交媒体上的传播效果。**

- **链接: [https://arxiv.org/pdf/2412.02271](https://arxiv.org/pdf/2412.02271)**

> **作者:** Preetika Verma; Kokil Jaidka
>
> **备注:** 8 pages, 3 figures, 8 tables Accepted at AAAI ICWSM 2026 We updated the paper title from "MediaSpin: Exploring Media Bias Through Fine-Grained Analysis of News Headlines " to "The MediaSpin Dataset: Post-Publication News Headline Edits Annotated for Media Bias"
>
> **摘要:** We present MediaSpin, a large-scale language resource capturing how major news outlets modify headlines after publication, and MediaSpin-in-the-Wild, a complementary dataset linking these revised headlines to their downstream engagement on social media. The increasing editability of online news headlines offers new opportunities to study linguistic framing and bias through the lens of editorial revisions. The dataset contains 78,910 headline pairs annotated for 13 types of media bias, grounded in established media-bias taxonomies, covering both subjective (e.g., sensationalism, spin) and objective (e.g., omission, slant) forms, with annotation conducted through a human-supervised large-language-model pipeline with expert validation and quality control. We describe the annotation schema and demonstrate three downstream applications: (1) cross-national analysis of how country references are added or removed during editing, (2) transformer-based bias classification at both binary and fine-grained levels, and (3) behavioral analysis of biased headlines on X (Twitter) using 180,786 news-related tweets from 819 consenting users. The results reveal regional asymmetries in representational framing, measurable linguistic markers, and consistently higher engagement with biased content. MediaSpin and MediaSpin-in-the-Wild together provide a reproducible benchmark for bias detection and the study of editorial and behavioral dynamics in contemporary media ecosystems.
>
---
#### [replaced 019] FinReporting: An Agentic Workflow for Localized Reporting of Cross-Jurisdiction Financial Disclosures
- **分类: cs.CL**

- **简介: 该论文属于财务报告任务，解决跨司法管辖区财务披露的标准化问题。提出FinReporting系统，通过统一本体和审计流程提升报告一致性与可靠性。**

- **链接: [https://arxiv.org/pdf/2604.05966](https://arxiv.org/pdf/2604.05966)**

> **作者:** Fan Zhang; Mingzi Song; Rania Elbadry; Yankai Chen; Shaobo Wang; Yixi Zhou; Xunwen Zheng; Yueru He; Yuyang Dai; Georgi Georgiev; Ayesha Gull; Muhammad Usman Safder; Fan Wu; Liyuan Meng; Fengxian Ji; Junning Zhao; Xueqing Peng; Jimin Huang; Yu Chen; Preslav Nakov; Zhuohan Xie
>
> **备注:** Accepted at ACL 2026 Demo Track. 9 pages, including figures and tables
>
> **摘要:** Financial reporting systems increasingly leverage Large Language Models (LLMs) to extract and summarize corporate disclosures. However, most existing approaches assume a single-market setting and overlook structural differences across jurisdictions. Variations in accounting taxonomies, tagging infrastructures (e.g., XBRL vs.\ PDF), and aggregation conventions introduce substantial challenges for semantic alignment and reliable verification. Here, we aim to bridge this gap. We present FinReporting, an agentic workflow for localized cross-jurisdiction financial reporting. The system constructs a unified canonical ontology spanning the income statement, balance sheet, and cash flow statement, and decomposes reporting into auditable stages, including filing acquisition, extraction, canonical mapping, and anomaly logging. Rather than treating LLMs as free-form generators, FinReporting employs them as constrained verifiers operating under explicit decision rules with evidence grounding. Evaluated on annual filings from the USA, Japan, and China, FinReporting improves consistency and reliability under heterogeneous reporting regimes. We further release an interactive demo that enables cross-market inspection and supports structured export of localized financial statements. Our demo is available at url{this https URL. A video describing our system is available at this https URL.
>
---
#### [replaced 020] Hallucinations are inevitable but can be made statistically negligible
- **分类: cs.CL; cs.FL; cs.LG; math.ST; stat.ML**

- **简介: 该论文属于自然语言处理领域，解决语言模型生成虚假内容（幻觉）的问题。通过概率理论证明，充分训练数据可使幻觉概率降至可忽略水平。**

- **链接: [https://arxiv.org/pdf/2502.12187](https://arxiv.org/pdf/2502.12187)**

> **作者:** Atsushi Suzuki; Yulan He; Feng Tian; Zhongyuan Wang
>
> **摘要:** Hallucinations, a phenomenon where a language model (LM) generates nonfactual content, pose a significant challenge to the practical deployment of LMs. While many empirical methods have been proposed to mitigate hallucinations, recent studies established a computability-theoretic result showing that any LM will inevitably generate hallucinations on an infinite set of inputs, regardless of the quality and quantity of training datasets and the choice of the language model architecture and training and inference algorithms. Although the computability-theoretic result may seem pessimistic, its significance in practical viewpoints has remained unclear. This paper claims that those "innate" inevitability results from computability theory and diagonal argument, in principle, cannot explain practical issues of LLMs. We demonstrate this claim by presenting a positive theoretical result from a probabilistic perspective. Specifically, we prove that hallucinations can be made statistically negligible, provided that the quality and quantity of the training data are sufficient. Interestingly, our positive result coexists with the computability-theoretic result, implying that while hallucinations on an infinite set of inputs cannot be entirely eliminated, their probability can always be reduced by improving algorithms and training data. By evaluating the two seemingly contradictory results through the lens of information theory, we argue that our probability-theoretic positive result better reflects practical considerations than the computability-theoretic negative result.
>
---
#### [replaced 021] Stabilizing Knowledge, Promoting Reasoning: Dual-Token Constraints for RLVR
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理能力优化任务，解决现有RLVR方法忽略token异质性的问题。提出Archer框架，通过双令牌约束提升推理性能。**

- **链接: [https://arxiv.org/pdf/2507.15778](https://arxiv.org/pdf/2507.15778)**

> **作者:** Jiakang Wang; Runze Liu; Fuzheng Zhang; Xiu Li; Guorui Zhou; Ling Pan
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has become an effective post-training method for improving the reasoning abilities of Large Language Models (LLMs). However, existing methods mainly apply uniform optimization constraints across all tokens, ignoring their heterogeneous roles. Prior work shows that high-entropy tokens are closely tied to reasoning, while low-entropy tokens primarily encode factual knowledge, and recent approaches attempt to exploit this distinction by isolating token updates via masking or asynchronous training. We argue that such isolation breaks the sequential dependency structure of autoregressive generation, leading to suboptimal learning. To address this, we propose \textbf{Archer}, an entropy-aware RLVR framework with \textbf{dual-token constraints} that preserves joint optimization while modulating update strength across token types. Our method introduces response-level entropy normalization for stable token classification and applies differentiated clipping ranges and KL regularization to encourage exploration on reasoning tokens while preserving knowledge tokens. Experiments on mathematical reasoning and code generation benchmarks show that Archer consistently outperforms strong baselines across multiple model scales, improving both \textit{pass@1} and \textit{pass@K} performance. These results highlight the importance of respecting sequence-level dependencies when designing fine-grained RL optimization strategies for LLMs.
>
---
#### [replaced 022] Reference Games as a Testbed for the Alignment of Model Uncertainty and Clarification Requests
- **分类: cs.CL**

- **简介: 该论文属于语言模型交互任务，旨在解决模型如何识别并表达不确定性的问题。通过参考游戏测试模型在不确定时请求澄清的能力。**

- **链接: [https://arxiv.org/pdf/2601.07820](https://arxiv.org/pdf/2601.07820)**

> **作者:** Manar Ali; Judith Sieker; Sina Zarrieß; Hendrik Buschmeier
>
> **备注:** Accepted at GEM@ACL 2026, the 5th Generation, Evaluation & Metrics Workshop
>
> **摘要:** In human conversation, both interlocutors play an active role in maintaining mutual understanding. When listeners are uncertain about what speakers mean, for example, they can request clarification. It is an open question for language models whether they can assume a similar listener role, recognizing and expressing their own uncertainty through clarification. We argue that reference games are a suitable testbed to approach this question as they are controlled, self-contained, and make clarification needs explicit and measurable. To test this, we evaluate three vision-language models comparing a baseline reference resolution task to an experiment where the models are instructed to request clarification when uncertain. The results suggest that even in such simple tasks, models often struggle to recognize internal uncertainty and translate it into adequate clarification behavior. This demonstrates the value of reference games as testbeds for interaction qualities of (vision and) language models.
>
---
#### [replaced 023] A Self-Evolving Framework for Efficient Terminal Agents via Observational Context Compression
- **分类: cs.CL**

- **简介: 该论文提出TACO框架，解决终端代理在长任务中因历史观察噪声导致的效率问题。通过自适应压缩提升任务性能与token效率。**

- **链接: [https://arxiv.org/pdf/2604.19572](https://arxiv.org/pdf/2604.19572)**

> **作者:** Jincheng Ren; Siwei Wu; Yizhi Li; Kang Zhu; Shu Xu; Boyu Feng; Ruibin Yuan; Wei Zhang; Riza Batista-Navarro; Jian Yang; Chenghua Lin
>
> **备注:** 27 pages
>
> **摘要:** As terminal agents scale to long-horizon, multi-turn workflows, a key bottleneck is not merely limited context length, but the accumulation of noisy terminal observations in the interaction history. Retaining raw observations preserves useful environment feedback, but also leads to context saturation and high token cost; conversely, naive compression may discard task-critical signals needed for subsequent actions. Because terminal environments are highly heterogeneous across repositories, commands, and execution states, heuristic-based or fixed-prompt compression methods are difficult to generalize. We propose TACO, a plug-and-play, training-free, self-evolving Terminal Agent Compression framework for existing terminal agents. TACO automatically discovers, refines, and reuses structured compression rules from interaction trajectories, enabling workflow-adaptive filtering of low-value terminal outputs while preserving task-relevant observations. Experiments on TerminalBench (TB 1.0 and TB 2.0) and four additional terminal-related benchmarks, including SWE-Bench Lite, CompileBench, DevEval, and CRUST-Bench, show that TACO consistently improves task performance and token efficiency across agent scaffolds and backbone models. On TerminalBench, TACO yields 1%-4% accuracy gains across strong agentic models and improves accuracy by around 2%-3% under the same token budget. On additional terminal-related benchmarks, it reduces total token consumption while maintaining or improving task success rates. These results suggest that self-evolving, workflow-adaptive observation compression is an effective path toward more reliable and efficient long-horizon terminal agents. The code is publicly available at this https URL.
>
---
#### [replaced 024] KV Cache Offloading for Context-Intensive Tasks
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究KV缓存卸载在需要大量上下文信息的任务中的表现，提出Text2JSON基准并发现性能下降问题，提出改进策略以提升准确性。**

- **链接: [https://arxiv.org/pdf/2604.08426](https://arxiv.org/pdf/2604.08426)**

> **作者:** Andrey Bocharnikov; Ivan Ermakov; Denis Kuznedelev; Vyacheslav Zhdanovskiy; Yegor Yershov
>
> **备注:** Preprint
>
> **摘要:** With the growing demand for long-context LLMs across a wide range of applications, the key-value (KV) cache has become a critical bottleneck for both latency and memory usage. Recently, KV-cache offloading has emerged as a promising approach to reduce memory footprint and inference latency while preserving accuracy. Prior evaluations have largely focused on tasks that do not require extracting large amounts of information from the context. In this work, we study KV-cache offloading on context-intensive tasks: problems where the solution requires looking up a lot of information from the input prompt. We create and release the Text2JSON benchmark, a highly context-intensive task that requires extracting structured knowledge from raw text. We evaluate modern KV offloading on Text2JSON and other context-intensive tasks and find significant performance degradation on both Llama 3 and Qwen 3 models. Our analysis identifies two key reasons for poor accuracy: low-rank projection of keys and unreliable landmarks, and proposes a simpler alternative strategy that significantly improves accuracy across multiple LLM families and benchmarks. These findings highlight the need for a comprehensive and rigorous evaluation of long-context compression techniques.
>
---
#### [replaced 025] Where Vision Becomes Text: Locating the OCR Routing Bottleneck in Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文研究视觉语言模型中OCR信息的处理路径，定位OCR瓶颈，分析其对文本理解的影响。任务属于视觉语言理解，解决OCR如何融入语言处理的问题。**

- **链接: [https://arxiv.org/pdf/2602.22918](https://arxiv.org/pdf/2602.22918)**

> **作者:** Jonathan Steinberg; Oren Gal
>
> **摘要:** Vision-language models (VLMs) can read text from images, but where does this optical character recognition (OCR) information enter the language processing stream? We investigate the OCR routing mechanism across three architecture families (Qwen3-VL, Phi-4, InternVL3.5) using causal interventions. By computing activation differences between original images and text-inpainted versions, we identify architecture-specific OCR bottlenecks whose dominant location depends on the vision-language integration strategy: DeepStack models (Qwen) show peak sensitivity at mid-depth (about 50%) for scene text, while single-stage projection models (Phi-4, InternVL) peak at early layers (6-25%), though the exact layer of maximum effect varies across datasets. The OCR signal is remarkably low-dimensional: PC1 captures up to 72.9% of variance. Crucially, principal component analysis (PCA) directions learned on one dataset transfer to others, demonstrating shared text-processing pathways. Surprisingly, in models with modular OCR circuits (notably Qwen3-VL-4B), OCR removal can improve counting performance (up to +6.9 percentage points), suggesting OCR interferes with other visual processing in sufficiently modular architectures.
>
---
#### [replaced 026] Falkor-IRAC: Graph-Constrained Generation for Verified Legal Reasoning in Indian Judicial AI
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出Falkor-IRAC，用于印度司法AI的法律推理任务，解决LLM生成中的事实错误问题，通过图约束生成确保推理路径有效。**

- **链接: [https://arxiv.org/pdf/2605.14665](https://arxiv.org/pdf/2605.14665)**

> **作者:** Joy Bose
>
> **备注:** 20 pages, 8 figures, 4 tables
>
> **摘要:** Legal reasoning is not semantic similarity search. A court judgment encodes constrained symbolic reasoning: precedent propagation, procedural state transitions, and statute-bound inference. These are properties that vector-based retrieval-augmented generation (RAG) cannot faithfully represent. Hallucinated precedents, outdated statute citations, and unsupported reasoning chains remain persistent failure modes in LLM-based legal AI, with real consequences for access to justice in high-caseload jurisdictions such as India. This paper presents Falkor-IRAC, a graph-constrained generation framework for Indian legal AI that grounds generation in structured reasoning over an IRAC (Issue, Rule, Analysis, Conclusion) knowledge graph. Judgments from the Supreme Court and High Courts of India are ingested as IRAC node structures enriched with procedural state transitions, precedent relationships, and statutory references, stored in FalkorDB for low-latency agentic traversal. At inference time, LLM-generated answers are accepted only if a valid supporting path can be traced through the graph, a check performed by a falsifiability oracle called the Verifier Agent. The system also detects doctrinal conflicts as a first-class output rather than silently resolving them. Falkor-IRAC is evaluated using graph-native metrics: citation grounding accuracy, path validity rate, hallucinated precedent rate, and conflict detection rate. These metrics are argued to be more appropriate for legal reasoning evaluation than BLEU and ROUGE. On a proof-of-concept corpus of 51 Supreme Court judgments, the Verifier Agent correctly validated citations on completed queries and correctly rejected fabricated citations. Evaluation against vector-only RAG baselines is left for future work. The companion InIRAC dataset, 500+ structured Indian court judgments with IRAC annotations, is released alongside this paper.
>
---
#### [replaced 027] Grounded or Guessing? LVLM Confidence Estimation via Blind-Image Contrastive Ranking
- **分类: cs.CL**

- **简介: 该论文属于视觉语言模型的置信度估计任务，旨在解决模型在无图像支持时仍自信生成回答的问题。提出BICR方法，通过对比真实与遮蔽图像的隐藏状态，提升模型可靠性判断。**

- **链接: [https://arxiv.org/pdf/2605.10893](https://arxiv.org/pdf/2605.10893)**

> **作者:** Reza Khanmohammadi; Erfan Miahi; Simerjot Kaur; Charese H. Smiley; Ivan Brugere; Kundan Thind; Mohammad M. Ghassemi
>
> **摘要:** Large vision-language models suffer from visual ungroundedness: they can produce a fluent, confident, and even correct response driven entirely by language priors, with the image contributing nothing to the prediction. Existing confidence estimation methods cannot detect this, as they observe model behavior under normal inference with no mechanism to determine whether a prediction was shaped by the image or by text alone. We introduce BICR (Blind-Image Contrastive Ranking), a model-agnostic confidence estimation framework that makes this contrast explicit during training by extracting hidden states from a frozen LVLM twice: once with the real image-question pair, and once with the image blacked out while the question is held fixed. A lightweight probe is trained on the real-image hidden state and regularized by a ranking loss that penalizes higher confidence on the blacked-out view, teaching it to treat visual grounding as a signal of reliability at zero additional inference cost. Evaluated across five modern LVLMs and seven baselines on a benchmark covering visual question answering, object hallucination detection, medical imaging, and financial document understanding, BICR achieves the best cross-LVLM average on both calibration and discrimination simultaneously, with statistically significant discrimination gains robust to cluster-aware analysis at 4-18x fewer parameters than the strongest probing baseline.
>
---
#### [replaced 028] Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在解决LLM预训练数据法律合规问题。提出Common Corpus，一个包含2万亿标记的开放数据集，涵盖多种语言和代码，适用于多语言预训练。**

- **链接: [https://arxiv.org/pdf/2506.01732](https://arxiv.org/pdf/2506.01732)**

> **作者:** Pierre-Carl Langlais; Pavel Chizhov; Catherine Arnett; Carlos Rosas Hinostroza; Mattia Nee; Eliot Krzystof Jones; Irène Girard; David Mach; Anastasia Stasenko; Ivan P. Yamshchikov
>
> **摘要:** Large Language Models (LLMs) are pre-trained on large amounts of data from different sources and domains. Such datasets often contain trillions of tokens, including large portions of copyrighted or proprietary content, which raises questions about the legal use of such models. This underscores the need for truly open pre-training data that complies with data security regulations. In this paper, we introduce Common Corpus, the largest open dataset for LLM pre-training. The data assembled in Common Corpus are either uncopyrighted or under open licenses, totaling about two trillion tokens. The dataset contains a wide variety of languages, ranging from the high-resource European languages to some low-resource languages rarely represented in pre-training datasets. In addition, it includes a large amount of code data. The diversity of data sources in terms of covered domains and time periods opens up the paths for both research and entrepreneurial needs across diverse areas of knowledge. In this paper, we present the detailed provenance of data assembling and the details of dataset filtering and curation. We train two small language models on Common Corpus and find that they perform comparably to other models of their size, indicating that our dataset is suitable for multilingual pretraining. Common Corpus represents a key contribution to the ecosystem for open science research on Large Language Models.
>
---
#### [replaced 029] IndicSafe: A Benchmark for Evaluating Multilingual LLM Safety in South Asia
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言大模型安全评估任务，旨在解决低资源语言中模型安全性不一致的问题。通过构建IndicSafe基准，评估12种南亚语言模型的安全性，揭示跨语言安全表现差异。**

- **链接: [https://arxiv.org/pdf/2603.17915](https://arxiv.org/pdf/2603.17915)**

> **作者:** Priyaranjan Pattnayak; Sanchari Chowdhuri
>
> **摘要:** As large language models (LLMs) are deployed in multilingual settings, their safety behavior in culturally diverse, low-resource languages remains poorly understood. We present the first systematic evaluation of LLM safety across 12 Indic languages, spoken by over 1.2 billion people but underrepresented in LLM training data. Using a dataset of 6,000 culturally grounded prompts spanning caste, religion, gender, health, and politics, we assess 10 leading LLMs on translated variants of the prompt. Our analysis reveals significant safety drift: cross-language agreement is just 12.8\%, and \texttt{SAFE} rate variance exceeds 17\% across languages. Some models over-refuse benign prompts in low-resource scripts, overflag politically sensitive topics, while others fail to flag unsafe generations. We quantify these failures using prompt-level entropy, category bias scores, and multilingual consistency indices. Our findings highlight critical safety generalization gaps in multilingual LLMs and show that safety alignment does not transfer evenly across languages. We release \textsc{IndicSafe}, the first benchmark to enable culturally informed safety evaluation for Indic deployments, and advocate for language-aware alignment strategies grounded in regional harms.
>
---
#### [replaced 030] Do Chinese models speak Chinese languages?
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理领域，研究中国模型与西方模型在多语言支持上的差异。通过比较分析，探讨模型开发中的语言优先级与全球基准影响。**

- **链接: [https://arxiv.org/pdf/2504.00289](https://arxiv.org/pdf/2504.00289)**

> **作者:** Andrea W Wen-Yi; Unso Eun Seo Jo; David Mimno
>
> **备注:** First and second author contribute equally
>
> **摘要:** The release of top-performing open-weight LLMs has cemented China's role as a leading force in AI development. Do these models support languages spoken in China? Or do they support the same languages as models developed in the United States or in Europe? Comparing multilingual capabilities is important for two reasons. First, language ability provides insights into pre-training data curation, and thus into resource allocation and development priorities. Second, Chinese model developers need to navigate the tension between serving a linguistically diverse population domestically, and optimizing for globally visible benchmarks that are predominantly English. We investigate Chinese model developers' priorities through a comparative study of Chinese-developed and Western-developed open-weight LLMs, on 21 language variants including Asian regional, Chinese, and European languages. Our experiments on Information Parity and reading comprehension show Chinese models' performance across these languages correlates strongly (r=0.93) with their Western counterparts, with the sole exception being better Mandarin. Chinese-developed models are good at French and German, but they sometimes cannot identify languages spoken by Chinese minorities such as Kazakh and Uyghur. Overall, all open-weight LLMs we study have a similar multilingual performance profile, despite the diverse linguistic and cultural contexts the model developers operated within. We interpret the homogenization as consistent with the influence of global benchmarking practices and shared training resources. Rather than treating current language support as inevitable, our results highlight multilingual development as a space of prioritization and trade-offs, with implications for model developers, policymakers, and users.
>
---
#### [replaced 031] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型预训练任务，解决数据混合优化问题。提出DeMix框架，通过模型合并预测最佳数据比例，降低搜索成本，提升性能。**

- **链接: [https://arxiv.org/pdf/2602.00747](https://arxiv.org/pdf/2602.00747)**

> **作者:** Shengrui Li; Fei Zhao; Kaiyan Zhao; Jieying Ye; Haifeng Liu; Fangcheng Shi; Zheyong Xie; Yao Hu; Shaosheng Cao
>
> **备注:** 18 pages, 5 figures, accepted at ICML 2026
>
> **摘要:** Determining an effective data mixture is a key factor in Large Language Model (LLM) pre-training, where models must balance general competence with proficiency on hard tasks such as math and code. However, identifying an optimal mixture remains an open challenge, as existing approaches either rely on unreliable tiny-scale proxy experiments or require prohibitively expensive large-scale exploration. To address this, we propose Decouple Searching from Training Mix (DeMix), a novel framework that leverages model merging to predict optimal data ratios. Instead of training proxy models for every sampled mixture, DeMix trains component models on candidate datasets at scale and derives data mixture proxies via weighted model merging. This paradigm decouples search from training costs, enabling evaluation of unlimited sampled mixtures without extra training burden and thus facilitating better mixture discovery through more search trials. Extensive experiments demonstrate that DeMix breaks the trade-off between sufficiency, accuracy and efficiency, obtaining the optimal mixture with higher benchmark performance at lower search cost. Additionally, we release the DeMix Corpora, a comprehensive 22T-token dataset comprising high-quality pre-training data with validated mixtures to facilitate open research. Our code and DeMix Corpora is available at this https URL.
>
---
#### [replaced 032] AirNav: A Large-Scale UAV Vision-and-Language Navigation Dataset with Natural and Diverse Instructions
- **分类: cs.CL**

- **简介: 该论文提出AirNav数据集，解决UAV视觉语言导航任务中数据规模小、场景不真实的问题，通过真实数据和多样化指令提升模型训练与评估效果。**

- **链接: [https://arxiv.org/pdf/2601.03707](https://arxiv.org/pdf/2601.03707)**

> **作者:** Hengxing Cai; Yijie Rao; Ligang Huang; Zanyang Zhong; Jinhan Dong; Jingjun Tan; Changhao Nai; Jue Hou; Wenhao Lu; Renxin Zhong
>
> **摘要:** Existing UAV vision-and-language navigation (VLN) benchmarks rarely provide realistic aerial scenes, natural process-level instructions, and sufficient scale simultaneously, making it difficult to systematically train and evaluate UAV VLN agents under realistic settings. To address this, we propose \textbf{AirNav}, a large-scale benchmark built on real urban aerial data, comprising 137K navigation samples with natural and diverse instructions generated via a human--LLM collaborative pipeline with 10 user personas. We conduct a systematic evaluation of representative approaches on AirNav, ranging from traditional models to multimodal large language models (MLLMs), under unified metrics with open-source implementations. We further propose \textbf{AirVLN-R1}, trained via supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT), achieving state-of-the-art performance with a 51.82\% success rate on the test-unseen split. Real-world experiments on a physical UAV platform provide preliminary evidence of sim-to-real transferability, and our dataset and code are publicly available.
>
---
#### [replaced 033] Antidistillation Fingerprinting
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出一种新的模型指纹技术，用于检测学生模型是否通过蒸馏学习教师模型。解决的是在保持模型性能的同时增强指纹检测的问题。**

- **链接: [https://arxiv.org/pdf/2602.03812](https://arxiv.org/pdf/2602.03812)**

> **作者:** Yixuan Even Xu; John Kirchenbauer; Yash Savani; Asher Trockman; Alexander Robey; Tom Goldstein; Fei Fang; J. Zico Kolter
>
> **备注:** 28 pages, 13 figures, ICML 2026
>
> **摘要:** Model distillation enables efficient emulation of frontier large language models (LLMs), creating a need for robust mechanisms to detect when a third-party student model has trained on a teacher model's outputs. However, existing fingerprinting techniques that could be used to detect such distillation rely on heuristic perturbations that impose a steep trade-off between generation quality and fingerprinting strength, often requiring significant degradation of utility to ensure the fingerprint is effectively internalized by the student. We introduce antidistillation fingerprinting (ADFP), a principled approach that aligns the fingerprinting objective with the student's learning dynamics. Building upon the gradient-based framework of antidistillation sampling, ADFP utilizes a proxy model to identify and sample tokens that directly maximize the expected detectability of the fingerprint in the student after fine-tuning, rather than relying on the incidental absorption of the un-targeted biases of a more naive watermark. Experiments on GSM8K, OASST1, and MBPP demonstrate that ADFP achieves a significant Pareto improvement over state-of-the-art baselines, yielding stronger detection confidence with minimal impact on utility across mathematical reasoning, dialogue, and code generation, even when the student model's architecture is unknown.
>
---
#### [replaced 034] HIVE: Hidden-Evidence Verification for Hallucination Detection in Diffusion Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于 hallucination 检测任务，旨在解决扩散大语言模型中幻觉信号难以捕捉的问题。提出 HIVE 框架，通过提取隐藏证据进行验证，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.26139](https://arxiv.org/pdf/2604.26139)**

> **作者:** Guoshenghui Zhao; Tan Yu; Weijie Zhao
>
> **备注:** 5 figures, appendix included
>
> **摘要:** Diffusion large language models generate text through multi-step denoising, where hallucination signals may emerge throughout the trajectory rather than only in the final output. Existing detectors mainly rely on output uncertainty or coarse trace statistics, which often fail to capture the richer hidden dynamics of D-LLMs. We propose HIVE, a hidden-evidence verification framework that extracts compressed hidden evidence from denoising trajectories, selects informative step-layer evidence, and conditions a verifier language model on the selected evidence through prefix embeddings. HIVE produces both a continuous hallucination score from verifier decision logits and structured verification outputs, including hallucination types, evidence pairs, and short rationales. Across two D-LLMs and three QA benchmarks, HIVE consistently outperforms eight strong baselines and achieves up to 0.9236 AUROC and 0.9537 AUPRC. Ablation studies further confirm the importance of hidden-evidence conditioning, learned evidence selection, two-stream evidence representation, and step-layer embeddings. These results suggest that selected hidden evidence from denoising trajectories provides a stronger and more usable hallucination signal than output-only uncertainty or coarse trace statistics.
>
---
#### [replaced 035] Does RAG Know When Retrieval Is Wrong? Diagnosing Context Compliance under Knowledge Conflict
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究RAG系统在知识冲突下的上下文合规问题，通过CDD方法分析并改进其应对能力，旨在提升模型在检索错误时的判断与处理效果。**

- **链接: [https://arxiv.org/pdf/2605.14473](https://arxiv.org/pdf/2605.14473)**

> **作者:** Yihang Chen; Pin Qian; Su Wang; Sipeng Zhang; Huan Xu; Shuhuai Lin; Xinpeng Wei
>
> **备注:** 12 pages, 4 figures, 3 tables
>
> **摘要:** The Context-Compliance Regime in Retrieval-Augmented Generation (RAG) occurs when retrieved context dominates the final answer even when it conflicts with the model's parametric knowledge. Accuracy alone does not reveal how retrieved context causally shapes answers under such conflict. We introduce Context-Driven Decomposition (CDD), a belief-decomposition probe that operates at inference time and serves as an intervention mechanism for controlled retrieval conflict. Across Epi-Scale stress tests, TruthfulQA misconception injection, and cross-model reruns, CDD exposes three patterns. P1: context compliance is measurable in an upper-bound adversarial setting, where Standard RAG reaches 15.0% accuracy on TruthfulQA misconception injection (N=500). P2: adversarial accuracy gains transfer across model families -- CDD improves accuracy on Gemini-2.5-Flash and on Claude Haiku/Sonnet/Opus -- but rationale-answer causal coupling does not transfer. CDD reaches 64.1% mistake-injection causal sensitivity on Gemini-2.5-Flash, while sensitivities for all three Claude variants fall in the [-3%, +7%] range, suggesting that the Claude-side accuracy gains operate through a mechanism distinct from the explicit conflict-resolution trace. P3: explicit conflict decomposition improves robustness under temporal drift and noisy distractors, with CDD reaching 71.3% on temporal shifts and 69.9% on distractor evidence on the full Epi-Scale adversarial benchmark. These three patterns identify context-compliance as a structural axis along which standard RAG can be probed and intervened on, distinct from retrieval-quality or single-method robustness questions, and motivate releasing Epi-Scale for systematic study across model families and retrieval pipelines.
>
---
#### [replaced 036] Improve Large Language Model Systems with User Logs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM系统在真实部署中利用用户日志提升性能的问题。通过UNO框架，从用户日志中提取有效反馈，优化模型表现。**

- **链接: [https://arxiv.org/pdf/2602.06470](https://arxiv.org/pdf/2602.06470)**

> **作者:** Changyue Wang; Weihang Su; Qingyao Ai; Yiqun Liu
>
> **摘要:** Scaling training data and model parameters has long driven progress in large language models (LLMs), but this paradigm is increasingly constrained by the scarcity of high-quality data and diminishing returns from rising computational costs. As a result, recent work is increasing the focus on continual learning from real-world deployment, where user interaction logs provide a rich source of authentic human feedback and procedural knowledge. However, learning from user logs is challenging due to their unstructured and noisy nature. Vanilla LLM systems often struggle to distinguish useful feedback signals from noisy user behavior, and the disparity between user log collection and model optimization (e.g., the off-policy optimization problem) further strengthens the problem. To this end, we propose UNO (User log-driveN Optimization), a unified framework for improving LLM systems (LLMsys) with user logs. UNO first distills logs into semi-structured rules and preference pairs, then employs query-and-feedback-driven clustering to manage data heterogeneity, and finally quantifies the cognitive gap between the model's prior knowledge and the log data. This assessment guides the LLMsys to adaptively filter out noisy feedback and construct different modules for primary and reflective experiences extracted from user logs, thereby improving future responses. Extensive experiments show that UNO achieves state-of-the-art effectiveness and efficiency, significantly outperforming Retrieval Augmented Generation (RAG) and memory-based baselines. We have open-sourced our code at this https URL .
>
---
#### [replaced 037] Introducing MELI: the Mandarin-English Language Interview Corpus
- **分类: cs.CL**

- **简介: 本文介绍MELI语料库，用于中英双语研究。解决双语语音分析问题，收集并标注29.8小时语音数据，支持语言对比与代码转换分析。**

- **链接: [https://arxiv.org/pdf/2603.27043](https://arxiv.org/pdf/2603.27043)**

> **作者:** Suyuan Liu; Molly Babel
>
> **备注:** Accepted at LREC 2026 (14th International Conference on Language Resources and Evaluation), to appear in the conference proceedings
>
> **摘要:** We introduce the Mandarin-English Language Interview (MELI) Corpus, an open-source resource of 29.8 hours of speech from 51 Mandarin-English bilingual speakers. MELI combines matched sessions in Mandarin and English with two speaking styles: read sentences and spontaneous interviews about language varieties, standardness, and learning experiences. Audio was recorded at 44.1 kHz (16-bit, stereo). Interviews were fully transcribed, force-aligned at word and phone levels, and anonymized. Descriptively, the Mandarin component totals ~14.7 hours (mean duration 17.3 minutes) and the English component ~15.1 hours (mean duration 17.8 minutes). We report token/type statistics for each language and document code-switching patterns (frequent in Mandarin sessions; more limited in English sessions). The corpus design supports within-/cross-speaker, within/cross-language acoustic comparison and links acoustics to speakers' stated language attitudes, enabling both quantitative and qualitative analyses. The MELI Corpus will be released with transcriptions, alignments, metadata, scans of labelled maps and documentation under a CC BY-NC 4.0 license.
>
---
#### [replaced 038] Smoothie: Smoothing Diffusion on Token Embeddings for Text Generation
- **分类: cs.CL**

- **简介: 该论文提出Smoothie方法，解决文本生成中扩散模型的离散性问题，通过语义相似性平滑词嵌入，提升生成质量。属于文本生成任务。**

- **链接: [https://arxiv.org/pdf/2505.18853](https://arxiv.org/pdf/2505.18853)**

> **作者:** Alexander Shabalin; Viacheslav Meshchaninov; Dmitry Vetrov
>
> **备注:** 18 pages, 4 figures, 13 tables
>
> **摘要:** Diffusion models have achieved state-of-the-art performance in generating images, audio, and video, but their adaptation to text remains challenging due to its discrete nature. Prior approaches either apply Gaussian diffusion in continuous latent spaces, which inherits semantic structure but struggles with token decoding, or operate in categorical simplex space, which respect discreteness but disregard semantic relation between tokens. In this paper, we propose Smoothing Diffusion on Token Embeddings (Smoothie), a novel diffusion method that combines the strengths of both approaches by progressively smoothing token embeddings based on semantic similarity. This technique enables gradual information removal while maintaining a natural decoding process. Experimental results on several sequence-to-sequence and unconditional generation tasks demonstrate that Smoothie outperforms existing diffusion-based models in generation quality. Furthermore, ablation studies show that our proposed diffusion space yields better performance than both the standard embedding space and the categorical simplex. The code is available at this https URL.
>
---
#### [replaced 039] Swarm Skills: A Portable, Self-Evolving Multi-Agent System Specification for Coordination Engineering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多智能体系统任务，解决协作协议难以共享和进化的问题。提出Swarm Skills规范，实现多智能体协作的可移植与自进化。**

- **链接: [https://arxiv.org/pdf/2605.10052](https://arxiv.org/pdf/2605.10052)**

> **作者:** Xinyu Zhang; Zhicheng Dou; Deyang Li; Jianjun Tao; Shuo Cheng; Ruifeng Shi; Fangchao Liu; Enrui Hu; Yangkai Ding; Hongbo Wang; Qi Ye; Xuefeng Jin; Zhangchun Zhao
>
> **摘要:** As artificial intelligence engineering paradigms shift from single-agent Prompt and Context Engineering toward multi-agent \textbf{Coordination Engineering}, the ability to codify and systematically improve how multiple agents collaborate has emerged as a critical bottleneck. While single-agent skills can now be distributed as portable assets, multi-agent coordination protocols remain locked within framework-internal code or static configurations, preventing them from being shared across systems or autonomously improved over time. We propose \textbf{Swarm Skills}, a portable specification that extends the Anthropic Skills standard with multi-agent semantics. Swarm Skills turns multi-agent workflows into first-class, distributable assets that consist of roles, workflows, execution bounds, and a built-in semantic structure for self-evolution. To operationalize the specification's evolving nature, we present a companion self-evolution algorithm that automatically distills successful execution trajectories into new Swarm Skills and continuously patches existing ones based on multi-dimensional scoring (Effectiveness, Utilization, and Freshness), eliminating the need for human-in-the-loop oversight during the refinement process. Through an architectural compatibility analysis and a comprehensive qualitative case study using the open-source JiuwenSwarm reference implementation, we demonstrate how Swarm Skills achieves zero-adapter cross-agent portability via progressive disclosure, enabling agent teams to self-evolve their coordination strategies without framework lock-in.
>
---
#### [replaced 040] Mind the Motions: Benchmarking Theory-of-Mind in Everyday Body Language
- **分类: cs.CL**

- **简介: 该论文属于理论心智（ToM）研究任务，旨在解决机器对非言语线索（NVC）理解不足的问题。工作包括构建Motion2Mind数据集，并评估AI系统在NVC解释上的表现。**

- **链接: [https://arxiv.org/pdf/2511.15887](https://arxiv.org/pdf/2511.15887)**

> **作者:** Seungbeen Lee; Jinhong Jeong; Donghyun Kim; Yejin Son; Youngjae Yu
>
> **备注:** The authors identified issues in the current version and would like to withdraw the manuscript for substantial revision
>
> **摘要:** Our ability to interpret others' mental states through nonverbal cues (NVCs) is fundamental to our survival and social cohesion. While existing Theory of Mind (ToM) benchmarks have primarily focused on false-belief tasks and reasoning with asymmetric information, they overlook other mental states beyond belief and the rich tapestry of human nonverbal communication. We present Motion2Mind, a framework for evaluating the ToM capabilities of machines in interpreting NVCs. Leveraging an expert-curated body-language reference as a proxy knowledge base, we build Motion2Mind, a carefully curated video dataset with fine-grained nonverbal cue annotations paired with manually verified psychological interpretations. It encompasses 222 types of nonverbal cues and 397 mind states. Our evaluation reveals that current AI systems struggle significantly with NVC interpretation, exhibiting not only a substantial performance gap in Detection, as well as patterns of over-interpretation in Explanation compared to human annotators.
>
---
#### [replaced 041] Beyond Benchmarks: MathArena as an Evaluation Platform for Mathematics with LLMs
- **分类: cs.CL**

- **简介: 该论文提出MathArena，一个持续更新的数学推理评估平台，解决静态基准无法全面评估LLM数学能力的问题。通过扩展任务范围和定期更新基准，跟踪模型进展。**

- **链接: [https://arxiv.org/pdf/2605.00674](https://arxiv.org/pdf/2605.00674)**

> **作者:** Jasper Dekoninck; Nikola Jovanović; Tim Gehrunger; Kári Rögnvaldsson; Ivo Petrov; Chenhao Sun; Martin Vechev
>
> **摘要:** Large language models (LLMs) are becoming increasingly capable mathematical collaborators, but static benchmarks are no longer sufficient for evaluating progress: they are often narrow in scope, quickly saturated, and rarely updated. This makes it hard to compare models reliably and track progress over time. Instead, we need evaluation platforms: continuously maintained systems that run, aggregate, and analyze evaluations across many benchmarks to give a comprehensive picture of model performance within a broad domain. In this work, we build on the original MathArena benchmark by substantially broadening its scope from final-answer olympiad problems to a continuously maintained evaluation platform for mathematical reasoning with LLMs. MathArena now covers a much wider range of tasks, including proof-based competitions, research-level arXiv problems, and formal proof generation in Lean. Additionally, we maintain a clear evaluation protocol for all models and regularly design new benchmarks as model capabilities improve to ensure that MathArena remains challenging. Notably, the strongest model, GPT-5.5, now reaches 98% on the 2026 USA Math Olympiad and 74% on research-level questions, showing that frontier models can now comfortably solve extremely challenging mathematical problems. This highlights the importance of continuously maintained evaluation platforms like MathArena to track the rapid progress of LLMs in mathematical reasoning.
>
---
#### [replaced 042] DiscussLLM: Teaching Large Language Models When to Speak
- **分类: cs.CL; cs.HC**

- **简介: 该论文提出DiscussLLM，解决LLM在对话中被动响应的问题，通过训练模型判断何时主动发言，提升对话协作性。**

- **链接: [https://arxiv.org/pdf/2508.18167](https://arxiv.org/pdf/2508.18167)**

> **作者:** Deep Anil Patel; Iain Melvin; Christopher Malon; Martin Renqiang Min
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in understanding and generating human-like text, yet they largely operate as reactive agents, responding only when directly prompted. This passivity creates an "awareness gap," limiting their potential as truly collaborative partners in dynamic human discussions. We introduce $\textit{DiscussLLM}$, a framework designed to bridge this gap by training models to proactively decide not just $\textit{what}$ to say, but critically, $\textit{when}$ to speak. Our primary contribution is a scalable two-stage data generation pipeline that synthesizes a large-scale dataset of realistic multi-turn human discussions. Each discussion is annotated with one of five intervention types (e.g., Factual Correction, Concept Definition) and contains an explicit conversational trigger where an AI intervention adds value. By training models to predict a special silent token when no intervention is needed, they learn to remain quiet until a helpful contribution can be made. We explore two architectural baselines: an integrated end-to-end model and a decoupled classifier-generator system optimized for low-latency inference. We evaluate these models on their ability to accurately time interventions and generate helpful responses, paving the way for more situationally aware and proactive conversational AI.
>
---
#### [replaced 043] Scaling Laws for Mixture Pretraining Under Data Constraints
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究语言模型预训练中的数据混合策略，解决目标数据有限时的性能优化问题。通过实验分析混合数据的重复影响，提出考虑重复因素的缩放定律，指导有效数据配置。**

- **链接: [https://arxiv.org/pdf/2605.12715](https://arxiv.org/pdf/2605.12715)**

> **作者:** Anastasiia Sedova; Skyler Seto; Natalie Schluter; Pierre Ablin
>
> **摘要:** As language models scale, the amount of data they require grows -- yet many target data sources, such as low-resource languages or specialized domains, are inherently limited in size. A common strategy is to mix this scarce but valuable target data with abundant generic data, which presents a fundamental trade-off: too little target data in the mixture underexposes the model to the target domain, while too much target data repeats the same examples excessively, yielding diminishing returns and eventual overfitting. We study this trade-off across more than 2,000 language-model training runs spanning multiple model and target dataset sizes, as well as several data types, including multilingual, domain-specific, and quality-filtered mixtures. Across all settings, we find that repetition is a central driver of target-domain performance, and that mixture training tolerates much higher repetition than single-source training: scarce target corpora can be reused 15-20 times, with the optimal number of repetitions depending on the target data size, compute budget, and model scale. Next, we introduce a repetition-aware mixture scaling law that accounts for the decreasing value of repeated target tokens and the regularizing role of generic data. Optimizing the scaling law provides a principled way to compute effective mixture configurations, yielding practical mixture recommendations for pretraining under data constraints.
>
---
#### [replaced 044] VideoGameBench: Can Vision-Language Models complete popular video games?
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出VideoGameBench，评估视觉语言模型在实时视频游戏中完成任务的能力，解决模型在感知、导航等人类技能上的不足。**

- **链接: [https://arxiv.org/pdf/2505.18134](https://arxiv.org/pdf/2505.18134)**

> **作者:** Alex L. Zhang; Thomas L. Griffiths; Karthik R. Narasimhan; Ofir Press
>
> **备注:** 10 pages, 38 pages including supplementary
>
> **摘要:** Vision-language models (VLMs) have achieved strong results on coding and math benchmarks that are challenging for humans, yet their ability to perform tasks that come naturally to humans--such as perception, spatial navigation, and memory management--remains understudied. Real video games are crafted to be intuitive for humans to learn and master by leveraging innate inductive biases, making them an ideal testbed for evaluating such capabilities in VLMs. To this end, we introduce VideoGameBench, a benchmark consisting of 10 popular video games from the 1990s that VLMs directly interact with in real-time. VideoGameBench challenges models to complete entire games with access to only raw visual inputs and a high-level description of objectives and controls, a significant departure from existing setups that rely on game-specific scaffolding and auxiliary information. We keep three of the games secret to encourage solutions that generalize to unseen environments. Our experiments show that frontier vision-language models struggle to progress beyond the beginning of each game. We find inference latency to be a major limitation of frontier models in the real-time setting; therefore, we introduce VideoGameBench Lite, a setting where the game pauses while waiting for the LM's next action. The best performing models, Gemini 2.5 Pro and Claude 3.7 Sonnet, complete only 0.48% of VideoGameBench and 1.6% of VideoGameBench Lite. We hope that the formalization of the human skills mentioned above into this benchmark motivates progress in these research directions.
>
---
#### [replaced 045] Active Learners as Efficient PRP Rerankers
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，解决PRP reranking中噪声和位置偏差问题。通过主动学习框架，提升排序效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.14236](https://arxiv.org/pdf/2605.14236)**

> **作者:** Jeremías Figueiredo Paschmann; Juan Kaplan; Francisco Nattero; Santiago Barron; Juan Wisznia; Luciano del Corro
>
> **备注:** 13 pages, 7 figures. Preprint
>
> **摘要:** Pairwise Ranking Prompting (PRP) elicits pairwise preference judgments from an LLM, which are then aggregated into a ranking, usually via classical sorting algorithms. However, judgments are noisy, order-sensitive, and sometimes intransitive, so sorting assumptions do not match the setting. Because sorting aims to recover a full permutation, truncating it to meet a call budget does not produce a dependable top-K. We thus reframe PRP reranking as active learning from noisy pairwise comparisons and show that active rankers are drop-in replacements that improve NDCG@10 per call in the call-constrained regime. Our noise-robust framework also introduces a randomized-direction oracle that uses a single LLM call per pair. This approach converts systematic position bias into zero-mean noise, enabling unbiased aggregate ranking without the cost of bidirectional calls.
>
---
#### [replaced 046] Agentic Recommender System with Hierarchical Belief-State Memory
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MARS框架，解决推荐系统中用户偏好建模问题。通过分层记忆结构和动态生命周期管理，提升个性化推荐效果。**

- **链接: [https://arxiv.org/pdf/2605.14401](https://arxiv.org/pdf/2605.14401)**

> **作者:** Xiang Shen; Yuhang Zhou; Yifan Wu; Zhuokai Zhao; Siyu Lin; Lei Huang; Qianqian Zhong; Lizhu Zhang; Benyu Zhang; Xiangjun Fan; Hong Yan
>
> **备注:** 4 figures, 8 tables
>
> **摘要:** Memory-augmented LLM agents have advanced personalized recommendation, yet existing approaches universally adopt flat memory representations that conflate ephemeral signals with stable preferences, and none provides a complete lifecycle governing how memory should evolve. We propose MARS (Memory-Augmented Agentic Recommender System), a framework that treats recommendation as a partially observable problem and maintains a structured belief state that progressively abstracts noisy behavioral observations into a compact estimate of user preferences. MARS organizes this belief state into three tiers: event memory buffers raw signals, preference memory maintains fine-grained mutable chunks with explicit strength and evidence tracking, and profile memory distills all preferences into a coherent natural language narrative. A complete lifecycle of six operations -- extraction, reinforcement, weakening, consolidation, forgetting, and resynthesis -- is adaptively scheduled by an LLM-based planner rather than fixed-interval heuristics. Experiments on four InstructRec benchmark domains show that MARS achieves state-of-the-art performance with average improvements of 26.4% in HR@1 and 10.3% in NDCG@10 over the strongest baselines with further gains from agentic scheduling in evolving settings.
>
---
#### [replaced 047] TokenButler: Token Importance is Predictable
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出TokenButler，用于高效识别大语言模型中关键token，解决KV缓存带来的内存与计算瓶颈问题。通过动态选择重要token，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2503.07518](https://arxiv.org/pdf/2503.07518)**

> **作者:** Yash Akhauri; Ahmed F AbouElhamayed; Yifei Gao; Chi-Chih Chang; Sameh Gobriel; Nilesh Jain; Mohamed S. Abdelfattah
>
> **摘要:** Large Language Models (LLMs) rely on the Key-Value (KV) Cache to store token history, enabling efficient decoding of tokens. As the KV-Cache grows, it becomes a major memory and computation bottleneck. However, there is an opportunity to alleviate this bottleneck, prior research has shown that only a small subset of tokens contribute meaningfully to each decoding step. A key challenge in finding these critical tokens is that they are dynamic, and heavily input query-dependent. Existing methods either risk quality by evicting tokens permanently, or retain the full KV-Cache but rely on retrieving chunks of tokens and many existing KV-Cache sparsity methods rely on inaccurate proxies for token importance. To address these limitations, we introduce TokenButler, a high-granularity, query-aware predictor that learns to identify these critical tokens. TokenButler predicts low-dimensional importance queries at a fixed depth stride, and combines them with a learned projection of the real KV-cache keys to score tokens cheaply, enabling dynamic per-token selection under a fixed budget while preserving the full KV cache. We train TokenButler by distilling the model's masked causal attention distributions, optimizing a lightweight predictor with minimal parameter overhead. We evaluate TokenButler on a novel synthetic small-context co-referential retrieval task, demonstrating near-oracle accuracy where existing methods fail. Furthermore, TokenButler achieves competitive or superior performance on long-context benchmarks (RULER, LongBench), up to $\approx1.6\times$ on-GPU speedup using our proposed *prediction interval with neighbor fetching* that amortizes predictor cost while maintaining accuracy within $\approx$1.1\%, and up to 7.6$\times$ reduction in latency compared to Dense Attention with CPU offloading. Code is available: this https URL
>
---
#### [replaced 048] GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型量化任务，旨在解决低精度量化与模型精度之间的矛盾。提出GSQ方法，通过Gumbel-Softmax优化标量量化，提升精度并保持兼容性。**

- **链接: [https://arxiv.org/pdf/2604.18556](https://arxiv.org/pdf/2604.18556)**

> **作者:** Alireza Dadgarnia; Soroush Tabesh; Mahdi Nikdan; Michael Helcig; Eldar Kurtic; Maximilian Kleinegger; Dan Alistarh
>
> **摘要:** Quantization has become a standard tool for efficient LLM deployment, especially for local inference, where models are now routinely served at 2-3 bits per parameter. The state of the art is currently split into simple scalar quantization techniques, such as GPTQ or AWQ, which are widely deployed but plateau in accuracy at 3-4 bits per parameter (bpp), and "second-generation" vector- or trellis-quantized methods, such as QTIP, GPTVQ and AQLM, which push the accuracy frontier but are notoriously hard to implement and to scale. In this paper, we ask whether this gap is fundamental, or whether a carefully optimized $\textit{scalar}$ quantizer can recover most of it. We answer in the affirmative, by introducing GSQ (Gumbel-Softmax Quantization), a post-training scalar quantization method which jointly learns the per-coordinate grid assignments and the per-group scales using a Gumbel-Softmax relaxation of the discrete grid. GSQ matches the cardinality of the relaxation to the small number of levels available in the target bit-width regime (e.g., 3-8 levels for ternary and 3 bpp, respectively), making optimization tractable. Practically, on the standard Llama-3.1-8B/70B-Instruct models, GSQ closes most of the gap between scalar quantization and the QTIP frontier at 2 and 3 bits, while using a symmetric scalar grid with group-wise quantization, and thus remains compatible with existing scalar inference kernels. We further show that the same discrete-assignment optimization can be applied to practical GGUF K-Quant checkpoints: starting from publicly released GGUF models, GSQ improves accuracy while projecting the result back into the same deployment format. Finally, GSQ scales to trillion-scale Mixture-of-Experts models such as Kimi-K2.5, where vector-quantized methods are difficult to apply. The source code is publicly available at this https URL.
>
---
#### [replaced 049] AIPO: Learning to Reason from Active Interaction
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AIPO框架，解决大语言模型推理能力受限问题，通过多智能体交互提升推理性能。**

- **链接: [https://arxiv.org/pdf/2605.08401](https://arxiv.org/pdf/2605.08401)**

> **作者:** Junnan Liu; Linhao Luo; Thuy-Trang Vu; Gholamreza Haffari
>
> **备注:** Preprint
>
> **摘要:** Recent advances in large language models (LLMs) have demonstrated remarkable reasoning capabilities, largely stimulated by Reinforcement Learning with Verifiable Rewards (RLVR). However, existing RL algorithms face a fundamental limitation: their exploration remains largely constrained by the inherent capability boundary of the policy model. Although recent methods introduce external expert demonstrations to extend this boundary, they typically rely on complete trajectory-level guidance, which is sample-inefficient, information-sparse, and may confine exploration to a static guidance space. Inspired by the potential of multi-agent systems, we propose $\textbf{AIPO}$, an enhanced reinforcement learning framework that improves LLM reasoning through active multi-agent interaction during exploration. Specifically, AIPO enables the policy model to proactively consult three functional collaborative agents, $\textit{Verify Agent}$, $\textit{Knowledge Agent}$, and $\textit{Reasoning Agent}$, when encountering reasoning bottlenecks, thereby receiving fine-grained and targeted guidance to actively expand its capability boundary during training. We further introduce a tailored importance sampling coefficient together with a clipping strategy to mitigate the off-policy bias and gradient vanishing issues that arise when learning from agent-provided feedback. After training, the policy model performs reasoning independently without relying on collaborative agents. Extensive experiments on diverse reasoning benchmarks, including AIME, MATH500, GPQA-Diamond, and LiveCodeBench, show that AIPO consistently improves reasoning performance, generalizes robustly across different policy models and RLVR algorithms, and effectively expands the reasoning capability boundary of the policy model.
>
---
#### [replaced 050] The Last Word Often Wins: A Format Confound in Chain-of-Thought Corruption Studies
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型推理评估任务，指出现有链式思维（CoT）可信度研究中的格式混淆问题，即测试结果主要反映答案位置而非计算过程。通过实验验证并提出改进协议。**

- **链接: [https://arxiv.org/pdf/2605.10799](https://arxiv.org/pdf/2605.10799)**

> **作者:** Gabriel Garcia
>
> **备注:** 34 pages, 6 figures, 13 tables. Submitted to NeurIPS 2026. Code and data: this https URL
>
> **摘要:** Corruption studies, the standard tool for evaluating chain-of-thought (CoT) faithfulness, infer which steps are ``computationally important'' from accuracy loss when steps are corrupted. We show that when benchmark chains end with an explicit terminal answer line, as in GSM8K and MATH, these tests largely measure \emph{answer placement} rather than where intermediate computation is carried out. Using matched GSM8K examples, removing only the final answer statement while preserving all reasoning collapses suffix sensitivity by about $19\times$ for Qwen~2.5-3B ($N{=}300$, $p{=}0.022$). Conflicting-answer prompts, which contain correct reasoning but a wrong explicit final answer, drive accuracy to zero or near-zero at 7B across five open-weight model families; wrong-answer following is strong at 3B--7B and attenuates sharply at larger scales. Replications on MATH, within-stable comparisons at 7B, and suffix-free chains show the same pattern in different guises: corruption sensitivity tracks the location of explicit answer text, not a fixed computational depth in the reasoning. Generation-time probes indicate that final answers are rarely early-determined during generation (${<}5\%$ early commitment), yet consumption-time behavior systematically follows explicit answer text. The confound is therefore largely a readout effect when the chain is consumed. We propose a three-prerequisite protocol (question-only control, format characterization, and an all-position sweep) as a practical minimum for future corruption-based faithfulness studies.
>
---
#### [replaced 051] TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出TemplateRL，用于增强大模型推理的强化学习方法。针对传统RL效率低、策略不具迁移性的问题，通过结构化模板引导提升采样效率与策略稳定性。**

- **链接: [https://arxiv.org/pdf/2505.15692](https://arxiv.org/pdf/2505.15692)**

> **作者:** Jinyang Wu; Chonghua Liao; Mingkuan Feng; Shuai Zhang; Zhengqi Wen; Haoran Luo; Ling Yang; Huazhe Xu; Jianhua Tao
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Reinforcement learning (RL) has emerged as an effective paradigm for enhancing model reasoning. However, existing RL methods like GRPO typically rely on unstructured self-sampling to fit scalar rewards, often producing inefficient rollouts that fail to capture transferable problem-solving strategies. To address this limitation, we propose **TemplateRL**, a structured template-guided RL framework that augments policy optimization with explicit template guidance. Our approach first constructs a problem-solving template library via MCTS on a small seed set, then seamlessly integrates this high-level structured guidance into RL training. By guiding rollout generation to align with proven template structures, TemplateRL significantly improves high-quality trajectory hit rates while reducing ineffective exploration. This structure-guided design steers the policy toward validated strategic patterns, stabilizing training dynamics, and enhancing RL sampling efficiency. Notably, the explicit template library is interpretable, editable, and supports online updates-enabling continuous updates during both training and inference. Extensive experiments demonstrate that TemplateRL outperforms GRPO by 99% on AIME and 41% on AMC, with superior stability on weak models and remarkable cross-domain generalization, highlighting its potential for broader tasks.
>
---
#### [replaced 052] The Right Answer, the Wrong Direction: Why Transformers Fail at Counting and How to Fix It
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究语言模型在计数任务中的失败原因，发现是输出路径与内部表示不匹配。通过调整输出头和注意力机制提升计数性能。任务为语言模型的计数能力优化。**

- **链接: [https://arxiv.org/pdf/2605.03258](https://arxiv.org/pdf/2605.03258)**

> **作者:** Gabriel Garcia
>
> **备注:** 27 pages, 3 figures, 18 tables. Code: this https URL
>
> **摘要:** Large language models often fail at simple counting tasks, even when items to count are in the prompt. We investigate whether this failure occurs because transformers do not represent counts internally, or because they cannot convert representations to the correct output tokens. Across three model families: Pythia, Qwen3, and Mistral, ranging from 0.4B to 14B parameters, we find evidence for the second explanation. Linear probes recover the correct count from intermediate layers with $R^2>0.99$, showing that the information is present. However, the internal directions that encode counts are nearly orthogonal to digit-token output-head rows ($|\cos| \leq 0.032$). In other words, the model stores the count in a form that the digit logits do not naturally read out. We localize this failure with two interventions. Updating only the digit rows of the output head (36,864 parameters) substantially improves constrained digit prediction (60.7--100.0% on four tasks), but it does not fix unconstrained generation (0%); we do not claim that digit-row repair fixes open-ended text. By contrast, small LoRA on attention Q/V (7.67M parameters) improves upstream routing and achieves 83.1%$\pm$7.2% in true greedy autoregressive generation (deployable fix). Logit-lens at layer 35 (entity counting; correct-digit rank): (i) median over 3 seeds drops from order-$10^4$ to 1; (ii) seed 42 shows $54{,}332 \to 838$ (median top-1 while one seed stays far below). Norm, logit-lens, and cross-task analyses generalize the bottleneck to counting, addition, and list length; nulls on MMLU and GSM8K and limited DROP transfer. These results identify counting failure as a geometric readout bottleneck, not an internal-representation failure: the model knows the count but the output pathway is misaligned with tokens needed to express it.
>
---
#### [replaced 053] Painless Activation Steering: An Automated, Lightweight Approach for Post-Training Large Language Models
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **简介: 该论文属于语言模型后训练任务，旨在解决传统方法耗时、昂贵或控制性差的问题。提出PAS方法，实现自动化、轻量化的激活向量调整，提升行为相关任务性能。**

- **链接: [https://arxiv.org/pdf/2509.22739](https://arxiv.org/pdf/2509.22739)**

> **作者:** Sasha Cui; Zhongren Chen
>
> **摘要:** Language models (LMs) are typically post-trained for desired capabilities and behaviors via weight-based or prompt-based steering, but the former is time-consuming and expensive, and the latter is not precisely controllable and often requires manual trial-and-error. While activation steering (AS) promises a cheap, fast, and controllable alternative to the two existing post-training methods, current AS techniques require hand-crafted prompt pairs or labor-intensive feature annotation, making them more inconvenient than the plug-and-play methods such as Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT). We introduce Painless Activation Steering (PAS), a family of fully automated methods that make AS readily usable with any given labeled dataset, with no need for prompt construction, feature labeling, or human intervention. We evaluate PAS on three open-weight models (Llama3.1-8B-Instruct, DeepSeek-R1-Distill-8B, and Nous-Hermes-2) and 18 tasks; we find that PAS reliably improves performance for behavior tasks, but not for intelligence-oriented tasks. The introspective variant (iPAS) delivers the strongest causal steering effects (10.1% on Bias, 5.2% on Morality, and 34.8% on Alignment). We also show PAS delivers additional gains on top of In-Context Learning (ICL) and SFT. PAS constructs a fast, lightweight activation vector that can be cheaply trained, easily stored, and activated at will. Our results provide a characterization of where AS helps, where it fails, and how to deploy it as a practical, automated LM post-training option.
>
---
#### [replaced 054] Performance-Driven Policy Optimization for Speculative Decoding with Adaptive Windowing
- **分类: cs.CL**

- **简介: 该论文属于语言模型推理优化任务，解决 speculative decoding 效率低的问题。通过窗口级强化学习优化 draft 模型，提升解码速度与接受长度。**

- **链接: [https://arxiv.org/pdf/2605.14978](https://arxiv.org/pdf/2605.14978)**

> **作者:** Jie Jiang; Xing Sun; Ruotian Chen; Jianan Su; Kaixin Shen
>
> **摘要:** Speculative decoding accelerates LLM inference by having a lightweight draft model propose speculative windows of candidate tokens for parallel verification by a larger target model. In practice, speculative efficiency is often bottlenecked by hard-to-draft positions, where an early mismatch truncates the accepted prefix and invalidates the rest of the speculative window. Most learning-based drafters are still optimized with token-level supervised objectives, even though speculative utility is inherently window-level and prefix-sensitive. We propose PPOW (Performance-Driven Policy Optimization with Adaptive Windowing), a reinforcement learning framework that shifts drafter optimization from token-level imitation to window-level optimization. PPOW combines a Cost-Aware Speedup Reward, a Distribution-Based Proximity Reward, and Adaptive Divergence-Aware Windowing, which prioritizes informative windows with high confidence-weighted draft-target divergence. PPOW achieves average acceptance lengths of 6.29-6.52 and speedups of 3.39-4.36$\times$ across multiple model families and benchmarks under a unified decoding protocol. These results show that performance-driven window-level optimization is a practical approach to improving speculative decoding efficiency.
>
---
#### [replaced 055] Language Generation as Optimal Control: Closed-Loop Diffusion in Latent Control Space
- **分类: cs.CL**

- **简介: 该论文将语言生成建模为最优控制问题，解决生成模型的效率与精度矛盾、可逆性等问题，提出基于流匹配的闭环控制方法，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2605.14531](https://arxiv.org/pdf/2605.14531)**

> **作者:** ZiYi Dong; Yuliang Huang; Weijian Deng; Xiangyang Ji; Liang Lin; Pengxu Wei
>
> **摘要:** This work reformulates language generation as a stochastic optimal control problem, providing a unified theoretical perspective to analyze autoregressive and diffusion models and explain their limitations (Efficiency-Fidelity Paradox, Irreversibility Error Propagation, Optimization Tractability and Fidelity) in terms of combination of trajectory singularity, adjoint state vanishing, and gradient absence. To address these issues, we approximate the solution to the Hamilton-Jacobi-Bellman (HJB) equation, yielding an optimal policy that acts as a closed-loop controller. To bypass the intractability of directly solving the HJB PDE, we employ Flow Matching as the optimal trajectory solver within the rectified latent control space. This allows our Manta-LM with Global Integral Operator to approximate the global vector field, effectively realizing a model that simultaneously achieves high-fidelity text generation and efficient, low-cost parallel sampling. Empirically, our method achieves strong performance on language modeling and conditional generation tasks, while exhibiting improved stability, efficiency, and controllability.
>
---
#### [replaced 056] Few-Step Diffusion Language Models via Trajectory Self-Distillation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言生成任务，旨在解决少步解码导致的生成质量下降问题。通过轨迹自蒸馏和直接判别优化，提升少步解码效果。**

- **链接: [https://arxiv.org/pdf/2602.12262](https://arxiv.org/pdf/2602.12262)**

> **作者:** Tunyu Zhang; Xinxi Zhang; Ligong Han; Haizhou Shi; Xiaoxiao He; Zhuowei Li; Hao Wang; Kai Xu; Akash Srivastava; Chengzhi Mao; Hao Wang; Vladimir Pavlovic; Dimitris N. Metaxas
>
> **摘要:** Diffusion large language models (DLLMs) have emerged as powerful generative models with the promise of fast text generation through parallel decoding. However, realizing this potential in practice remains challenging: reducing the number of decoding steps, typically causes a substantial degradation in output quality due to token factorization error. To alleviate this, we propose a self-distillation framework that trains a few-step student to match the generative trajectory of a full-step teacher. We theoretically and empirically show that trajectory-level supervision mitigates this factorization error, thereby enabling effective few-step decoding. We further incorporate Direct Discriminative Optimization (DDO), a reverse-KL objective that encourages mode-seeking toward the teacher's modes, yielding stronger performance on challenging reasoning tasks. Across reasoning and code-generation benchmarks, our method substantially narrows the gap between few-step and full-step decoding. The source code is available at this https URL.
>
---
#### [replaced 057] Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大模型微调任务，旨在解决SFT与RFT方法的局限性。通过提出Prefix-RFT，融合演示与探索，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2507.01679](https://arxiv.org/pdf/2507.01679)**

> **作者:** Zeyu Huang; Tianhao Cheng; Zihan Qiu; Zili Wang; Yinghui Xu; Edoardo M. Ponti; Ivan Titov
>
> **备注:** ICML 2026
>
> **摘要:** Existing LLMs-post-training techniques are broadly categorized into supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT). Each paradigm presents a distinct trade-off: (1) SFT excels at mimicking demonstration data, but can lead to problematic generalization as a form of behavior cloning. (2) Conversely, RFT can significantly enhance a model's performance but is prone to learning unexpected behaviors, and its performance is sensitive to the initial policy. In this paper, we propose a unified view of these methods and introduce Prefix-RFT, a hybrid approach that synergizes learning from both demonstration and exploration. Using mathematical reasoning problems as a test bed, we empirically demonstrate that Prefix-RFT is simple yet effective. Not only does it surpass the performance of standalone SFT and RFT, but it also outperforms parallel mixed-policy RFT methods. Our analysis highlights the complementary nature of SFT and RFT, validating that Prefix-RFT effectively harmonizes them. Further ablation studies confirm the method's robustness to variations in the quality and quantity of demonstration data.
>
---
#### [replaced 058] Don't Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills for QA and RAG
- **分类: cs.IR; cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于知识增强的问答任务，解决传统RAG模型被动消费检索结果的问题。通过构建可导航的知识技能目录，提升回答质量和事实依据。**

- **链接: [https://arxiv.org/pdf/2604.14572](https://arxiv.org/pdf/2604.14572)**

> **作者:** Yiqun Sun; Pengfei Wei; Lawrence B. Hsieh
>
> **摘要:** Retrieval-Augmented Generation (RAG) grounds LLM responses in external evidence but treats the model as a passive consumer of search results, with no view of how the corpus is organized or what it has not yet seen. We present Corpus2Skill, which distills a document corpus offline into a hierarchical skill directory and lets an LLM agent navigate it at serve time, drilling from a bird's-eye view through progressively finer summaries down to documents, and backtracking when a branch is unproductive. On an enterprise customer-support benchmark, Corpus2Skill improves both answer quality and grounding over single-shot dense, hybrid, hierarchical-retrieval, and agentic RAG baselines at a moderate cost tradeoff. A ten-subset generalization study further shows that corpus navigation is not a universal replacement for retrieval: it consistently helps on single-domain corpora with a recoverable topical taxonomy, but flat retrieval remains preferable on open-domain factoid pools or homogeneous-tabular corpora that defeat top-level clustering. We characterize this scope distinction and discuss it as a design guideline for knowledge-grounded systems. Code is available at this https URL.
>
---
#### [replaced 059] How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型优化任务，解决黑盒大模型定制化问题。通过训练小模型生成动态建议，提升黑盒模型性能，实现高效参数优化。**

- **链接: [https://arxiv.org/pdf/2510.02453](https://arxiv.org/pdf/2510.02453)**

> **作者:** Parth Asawa; Alan Zhu; Abigail O'Neill; Matei Zaharia; Alexandros G. Dimakis; Joseph E. Gonzalez
>
> **备注:** International Conference on Machine Learning (ICML) 2026
>
> **摘要:** Frontier language models are deployed as black-box services, where model weights cannot be modified and customization is limited to prompting. We introduce Advisor Models, a method to train small open-weight models to generate dynamic, per-instance natural language advice that improves the capabilities of black-box frontier models. Advisor Models improve GPT-5.2's performance on RuleArena (Taxes) by 27.4%, reduce Gemini 3 Pro's steps taken in SWE agent tasks by 24.6%, and outperform static prompt optimizers in personalizing GPT-5 to user preferences (85-100% vs. 40-60%). We also find that advisors are transferable: an advisor trained with a low-cost student model still transfers improvements to a frontier model. Moreover, Advisor Models are robust: we observe no degradation on other benchmarks than the pipeline is trained on. Our method shows how to perform parametric optimization for black-box frontier models in a practical and cost-effective way.
>
---
#### [replaced 060] Convergent Representations of Linguistic Constructions in Human and Artificial Neural Systems
- **分类: q-bio.NC; cs.AI; cs.CL**

- **简介: 该论文属于语言处理任务，研究人类和人工智能系统如何表征语言结构。通过EEG实验验证了语言模型中构造表征的神经签名，发现其与人工模型有相似模式，支持构造语法理论。**

- **链接: [https://arxiv.org/pdf/2603.29617](https://arxiv.org/pdf/2603.29617)**

> **作者:** Pegah Ramezani; Thomas Kinfe; Andreas Maier; Achim Schilling; Patrick Krauss
>
> **摘要:** Understanding how the brain processes linguistic constructions is a central challenge in cognitive neuroscience and linguistics. Recent computational studies show that artificial neural language models spontaneously develop differentiated representations of Argument Structure Constructions (ASCs), generating predictions about when and how construction-level information emerges during processing. The present study tests these predictions in human neural activity using electroencephalography (EEG). Ten native English speakers listened to 200 synthetically generated sentences across four construction types (transitive, ditransitive, caused-motion, resultative) while neural responses were recorded. Analyses using time-frequency methods, feature extraction, and machine learning classification revealed construction-specific neural signatures emerging primarily at sentence-final positions, where argument structure becomes fully disambiguated, and most prominently in the alpha band. Pairwise classification showed reliable differentiation, especially between ditransitive and resultative constructions, while other pairs overlapped. Crucially, the temporal emergence and similarity structure of these effects mirror patterns in recurrent and transformer-based language models, where constructional representations arise during integrative processing stages. These findings support the view that linguistic constructions are neurally encoded as distinct form-meaning mappings, in line with Construction Grammar, and suggest convergence between biological and artificial systems on similar representational solutions. More broadly, this convergence is consistent with the idea that learning systems discover stable regions within an underlying representational landscape - recently termed a Platonic representational space - that constrains the emergence of efficient linguistic abstractions.
>
---
#### [replaced 061] Reward Auditor: Inference on Reward Modeling Suitability in Real-World Perturbed Scenarios
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决奖励模型在真实场景下的适用性问题。通过引入Reward Auditor框架，检测模型在扰动环境中的系统性漏洞。**

- **链接: [https://arxiv.org/pdf/2512.00920](https://arxiv.org/pdf/2512.00920)**

> **作者:** Jianxiang Zang; Yongda Wei; Ruxue Bai; Shiyu Jiang; Nijia Mo; Binhong Li; Qiang Sun; Hui Liu
>
> **摘要:** Reliable reward models (RMs) are critical for ensuring the safe alignment of large language models (LLMs). However, current RM evaluation methods focus solely on preference perception accuracies in given specific scenarios, obscuring the critical vulnerabilities of RMs in real-world scenarios. We identify the true challenge lies in assessing a novel dimension: Suitability, defined as conditional reliability under specific real-world perturbations. To this end, we introduce Reward Auditor, a hypothesis-testing framework specifically designed for RM suitability inference. Rather than answering "How accurate is the RM's preference perception for given samples?", it employs scientific auditing to answer: "Can we infer RMs exhibit systematic vulnerabilities in specific real-world scenarios?". Under real-world perturbed scenarios, Reward Auditor quantifies statistical significance and effect size by auditing distribution degradation of RM preference perception confidence. This enables inference of both the certainty and severity of RM vulnerabilities across diverse real-world scenarios. This lays a solid foundation for building next-generation LLM alignment systems that are verifiably safe, more robust, and trustworthy.
>
---
#### [replaced 062] The Company You Keep: How LLMs Respond to Dark Triad Traits
- **分类: cs.CL**

- **简介: 该论文属于对话系统安全研究，旨在解决LLMs对Dark Triad特质用户提示的响应问题。通过分析模型行为，探讨如何设计更安全的交互系统。**

- **链接: [https://arxiv.org/pdf/2603.04299](https://arxiv.org/pdf/2603.04299)**

> **作者:** Zeyi Lu; Angelica Henestrosa; Pavel Chizhov; Ivan P. Yamshchikov
>
> **摘要:** Large Language Models (LLMs) often exhibit highly agreeable and reinforcing conversational styles, also known as AI-sycophancy. Although this pattern arises from training objectives that reward user satisfaction over accuracy, it may become problematic when interacting with user prompts that reflect negative social tendencies. Such responses risk amplifying harmful behavior rather than mitigating it. In this study, we examine how LLMs respond to user prompts expressing varying degrees of Dark Triad traits (Machiavellianism, Narcissism, and Psychopathy) using a curated dataset. Our analysis reveals differences across models, whereby all models predominantly exhibit corrective behavior, while showing reinforcing output in certain cases. Model behavior also depends on the severity level and differs in the sentiment of the response. Our findings raise implications for designing safer conversational systems that can detect and respond appropriately when users escalate from benign to harmful requests.
>
---
#### [replaced 063] Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决法律场景中对话代理主动获取信息的问题。通过双层级强化学习框架，提升对话管理与提问能力。**

- **链接: [https://arxiv.org/pdf/2605.14057](https://arxiv.org/pdf/2605.14057)**

> **作者:** Xubo Lin; Zezhi Deng; Shihao Wang; Grace Hui Yang; Yang Deng
>
> **备注:** Accepted in ACL 2026 as Findings
>
> **摘要:** Most existing dialogue systems are user-driven, primarily designed to fulfill user requests. However, in many critical real-world scenarios, a conversational agent must proactively extract information to achieve its own objectives rather than merely respond. To address this gap, we introduce Inquisitive Conversational Agents (ICAs) and develop an ICA specifically tailored to U.S. Supreme Court oral arguments. We propose a Dual Hierarchical Reinforcement Learning framework featuring two cooperating RL agents, each with its own policy, to coordinate strategic dialogue management and fine-grained utterance generation. By learning when and how to ask probing questions, the agent emulates judicial questioning patterns and systematically uncovers crucial information to fulfill its legal objectives. Evaluations on a U.S. Supreme Court dataset show that our method outperforms various baselines across multiple metrics. It represents an important first step toward broader high-stakes, domain-specific applications.
>
---
#### [replaced 064] Measuring and Mitigating Toxicity in Large Language Models: A Comprehensive Replication Study
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于AI安全任务，旨在解决大语言模型中的毒性问题。通过评估DExperts方法，验证其在显性毒性上的有效性，但发现对隐性仇恨言论效果有限，并存在延迟问题。**

- **链接: [https://arxiv.org/pdf/2605.14087](https://arxiv.org/pdf/2605.14087)**

> **作者:** Mokshit Surana; Archit Rathod; Akshaj Satishkumar
>
> **摘要:** Large Language Models (LLMs) trained on web-scale corpora inherently absorb toxic patterns from their training data. This leads to toxic degeneration where even innocuous prompts can trigger harmful outputs. This phenomenon poses significant risks for real-world deployments. Thus, necessitating effective mitigation strategies that should maintain model utility while ensuring safety. In this comprehensive replication study, we evaluate the efficacy of DExperts (Decoding-time Experts), which is an inference-time mitigation technique that steers generation without requiring model retraining. We structured our research into three systematic phases: (1) establishing baseline toxicity measurements using RealToxicityPrompts on standard GPT-2 models; then (2) implementing and evaluating DExperts to mitigate explicit toxicity; and finally (3) stress-testing the method against implicit hate speech using the adversarial ToxiGen dataset. Our empirical results confirm that while DExperts achieves near-perfect safety rates (100%) on explicit toxicity benchmarks, it exhibits brittleness against adversarial, implicit hate speech, with safety rates dropping to 98.5%. Furthermore, we quantify a critical trade-off. The method introduces a 10x latency penalty (from 0.2s to 2.0s per generation), posing challenges for real-time deployment scenarios. This study contributes to the growing body of work on AI safety by highlighting the robustness gap between explicit and implicit toxicity mitigation. We emphasize the need for more sophisticated approaches that generalize across diverse hate-speech patterns without incurring prohibitive computational costs.
>
---
#### [replaced 065] LLM-based Detection of Manipulative Political Narratives
- **分类: cs.CL**

- **简介: 该论文属于政治叙事检测任务，旨在区分操纵性言论与合法批评。通过结合提示过滤与无监督聚类，识别出41个操纵性叙事群组。**

- **链接: [https://arxiv.org/pdf/2605.14354](https://arxiv.org/pdf/2605.14354)**

> **作者:** Sinclair Schneider; Florian Steuber; Gabi Dreo Rodosek
>
> **备注:** This paper has been submitted to the upcoming 18th International Conference on Advances in Social Networks Analysis and Mining (ASONAM 2026)
>
> **摘要:** We present a new computational framework for detecting and structuring manipulative political narratives. A task that became more important due to the shift of political discussions to social media. One of the primary challenges thereby is differentiating between manipulative political narratives and legitimate critiques. Some posts may also reframe actual events within a manipulative context. To achieve good clustering results, we filter manipulative posts beforehand using a detailed few-shot prompt that combines documented campaign narratives with legitimate criticisms to differentiate them. This prompt enables a reasoning model to assign labels, retaining only manipulative narrative posts for further processing. The remaining posts are subsequently embedded and dimensionality-reduced using UMAP, before HDBSCAN is applied to uncover narrative groups. A key advantage of this unsupervised approach is its independence from a predefined list of target categories, enabling it to uncover new narrative clusters. Finally, a reasoning model is employed to uncover the narrative behind each cluster. This approach, applied to over 1.2 million social media posts, effectively identified 41 distinct manipulative narrative clusters by integrating prompt-based filtering with unsupervised clustering.
>
---
#### [replaced 066] FormulaCode: Evaluating Agentic Optimization on Large Codebases
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出FormulaCode基准，用于评估大模型在真实代码库上的多目标优化能力，解决现有基准不足的问题。**

- **链接: [https://arxiv.org/pdf/2603.16011](https://arxiv.org/pdf/2603.16011)**

> **作者:** Atharva Sehgal; James Hou; Akanksha Sarkar; Ishaan Mantripragada; Swarat Chaudhuri; Jennifer J. Sun; Yisong Yue
>
> **备注:** Preprint version
>
> **摘要:** Large language model (LLM) coding agents increasingly operate at the repository level, motivating benchmarks that evaluate their ability to optimize entire codebases under realistic constraints. Existing code benchmarks largely rely on synthetic tasks, binary correctness signals, or single-objective evaluation, limiting their ability to assess holistic optimization behavior. We introduce FormulaCode, a benchmark for evaluating agentic optimization on large, real-world codebases with fine-grained, multi-objective performance metrics. FormulaCode comprises 957 performance bottlenecks mined from scientific Python repositories on GitHub, each paired with expert-authored patches and, on average, 264.6 community-maintained performance workloads per task, enabling the holistic ability of LLM agents to optimize codebases under realistic correctness and performance constraints. Our evaluations reveal that repository-scale, multi-objective optimization remains a major challenge for frontier LLM agents. Project website at: this https URL
>
---
