# 自然语言处理 cs.CL

- **最新发布 54 篇**

- **更新 42 篇**

## 最新发布

#### [new 001] Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对扩散语言模型（dLLMs）的强化学习优化问题，提出基于ELBO的序列级策略优化（ESPO）框架。解决dLLMs缺乏自回归模型中可分解的词元级概率这一核心难题，将序列生成视为单一动作，使用ELBO作为序列级似然近似，实现稳定高效的训练，在数学、编程等任务上显著优于基线方法。**

- **链接: [https://arxiv.org/pdf/2512.03759v1](https://arxiv.org/pdf/2512.03759v1)**

> **作者:** Jingyang Ou; Jiaqi Han; Minkai Xu; Shaoxuan Xu; Jianwen Xie; Stefano Ermon; Yi Wu; Chongxuan Li
>
> **摘要:** Reinforcement Learning (RL) has proven highly effective for autoregressive language models, but adapting these methods to diffusion large language models (dLLMs) presents fundamental challenges. The core difficulty lies in likelihood approximation: while autoregressive models naturally provide token-level conditional probabilities essential for token-level RL objectives (e.g., GRPO), dLLMs generate sequences through iterative non-autoregressive denoising steps that lack this factorization. To address this fundamental mismatch, we propose ELBO-based Sequence-level Policy Optimization (ESPO), a principled RL framework that treats entire sequence generation as a single action and uses the ELBO as a tractable sequence-level likelihood proxy. Our method incorporates per-token normalization of importance ratios and robust KL-divergence estimation to ensure stable large-scale training. Extensive experiments on mathematical reasoning, coding, and planning tasks demonstrate that ESPO significantly outperforms token-level baselines, achieving dramatic improvements of 20-40 points on the Countdown task, while maintaining consistent gains on math and coding benchmarks. Our approach establishes sequence-level optimization as a principled and empirically effective paradigm for RL in dLLMs. Our code is available at https://github.com/ML-GSAI/ESPO.
>
---
#### [new 002] Characterizing Language Use in a Collaborative Situated Game
- **分类: cs.CL**

- **简介: 该论文研究协作式游戏中的语言使用，针对复杂情境下多人协作沟通难题，构建了11.5小时的《传送门2》合作模式语音语料库。通过分析空间指称、澄清与修正、临时约定等现象，揭示了传统对话数据中少见的语言特征，并公开发布多模态数据以支持未来研究。**

- **链接: [https://arxiv.org/pdf/2512.03381v1](https://arxiv.org/pdf/2512.03381v1)**

> **作者:** Nicholas Tomlin; Naitian Zhou; Eve Fleisig; Liangyuan; Chen; Téa Wright; Lauren Vinh; Laura X. Ma; Seun Eisape; Ellie French; Tingting Du; Tianjiao Zhang; Alexander Koller; Alane Suhr
>
> **摘要:** Cooperative video games, where multiple participants must coordinate by communicating and reasoning under uncertainty in complex environments, yield a rich source of language data. We collect the Portal Dialogue Corpus: a corpus of 11.5 hours of spoken human dialogue in the co-op mode of the popular Portal 2 virtual puzzle game, comprising 24.5K total utterances. We analyze player language and behavior, identifying a number of linguistic phenomena that rarely appear in most existing chitchat or task-oriented dialogue corpora, including complex spatial reference, clarification and repair, and ad-hoc convention formation. To support future analyses of language use in complex, situated, collaborative problem-solving scenarios, we publicly release the corpus, which comprises player videos, audio, transcripts, game state data, and both manual and automatic annotations of language data.
>
---
#### [new 003] AR-Med: Automated Relevance Enhancement in Medical Search via LLM-Driven Information Augmentation
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对医疗搜索中传统方法难以理解复杂查询的问题，提出AR-Med框架。通过检索增强与知识蒸馏，提升LLM在医疗领域的准确性与效率，解决事实幻觉与成本问题。构建了LocalQSMed基准，实现离线与线上性能对齐，显著提升搜索相关性与用户满意度。**

- **链接: [https://arxiv.org/pdf/2512.03737v1](https://arxiv.org/pdf/2512.03737v1)**

> **作者:** Chuyue Wang; Jie Feng; Yuxi Wu; Hang Zhang; Zhiguo Fan; Bing Cheng; Wei Lin
>
> **摘要:** Accurate and reliable search on online healthcare platforms is critical for user safety and service efficacy. Traditional methods, however, often fail to comprehend complex and nuanced user queries, limiting their effectiveness. Large language models (LLMs) present a promising solution, offering powerful semantic understanding to bridge this gap. Despite their potential, deploying LLMs in this high-stakes domain is fraught with challenges, including factual hallucinations, specialized knowledge gaps, and high operational costs. To overcome these barriers, we introduce \textbf{AR-Med}, a novel framework for \textbf{A}utomated \textbf{R}elevance assessment for \textbf{Med}ical search that has been successfully deployed at scale on the Online Medical Delivery Platforms. AR-Med grounds LLM reasoning in verified medical knowledge through a retrieval-augmented approach, ensuring high accuracy and reliability. To enable efficient online service, we design a practical knowledge distillation scheme that compresses large teacher models into compact yet powerful student models. We also introduce LocalQSMed, a multi-expert annotated benchmark developed to guide model iteration and ensure strong alignment between offline and online performance. Extensive experiments show AR-Med achieves an offline accuracy of over 93\%, a 24\% absolute improvement over the original online system, and delivers significant gains in online relevance and user satisfaction. Our work presents a practical and scalable blueprint for developing trustworthy, LLM-powered systems in real-world healthcare applications.
>
---
#### [new 004] Reconstructing KV Caches with Cross-layer Fusion For Enhanced Transformers
- **分类: cs.CL**

- **简介: 该论文针对大模型推理中长序列下的KV缓存内存瓶颈问题，提出FusedKV与FusedKV-Lite方法。通过跨层融合底层与中间层的键值信息，实现高效缓存共享，在减少50%内存的同时提升性能，显著优化了Transformer解码器的内存效率与推理表现。**

- **链接: [https://arxiv.org/pdf/2512.03870v1](https://arxiv.org/pdf/2512.03870v1)**

> **作者:** Hongzhan Lin; Zhiqi Bai; Xinmiao Zhang; Sen Yang; Xiang Li; Siran Yang; Yunlong Xu; Jiaheng Liu; Yongchi Zhao; Jiamang Wang; Yuchi Xu; Wenbo Su; Bo Zheng
>
> **备注:** under review
>
> **摘要:** Transformer decoders have achieved strong results across tasks, but the memory required for the KV cache becomes prohibitive at long sequence lengths. Although Cross-layer KV Cache sharing (e.g., YOCO, CLA) offers a path to mitigate KV Cache bottleneck, it typically underperforms within-layer methods like GQA. To understand the root cause, we investigate the information flow of keys and values of the top-layers. Our preliminary reveals a clear distribution: values are predominantly derived from the bottom layer, while keys draw more information from both bottom and middle layers. Building upon this, we propose FusedKV, whose top-layer KV caches are a learnable fusion of the most informative ones from the bottom and middle layers. This fusion operates directly on post-RoPE keys, preserving relative positional information without the computational cost of re-applying rotary embeddings. To further improve efficiency, we propose FusedKV-Lite, an cross-layer sharing approach, where top-layer KV caches are directly derived from the bottom-layer values and the middle-layer keys. Compared to FusedKV, FusedKV-Lite reduces I/O overhead at the cost of a slight increase in perplexity. In experiments on LLMs ranging from 332M to 4B parameters, our proposed method reduce 50\% cache memory while achieving lower validation perplexity than the standard Transformer decoder, establishing it as a memory-efficient, high-performance architectural alternative.
>
---
#### [new 005] Enhancing Instruction-Following Capabilities in Seq2Seq Models: DoLA Adaptations for T5
- **分类: cs.CL**

- **简介: 该论文研究如何将对比解码方法DoLa应用于T5等编码器-解码器模型，以提升其指令遵循能力。针对现有DoLa仅适用于解码器架构的局限，首次实现其在encoder-decoder结构中的应用，并通过层分析揭示其对生成质量的影响机制。**

- **链接: [https://arxiv.org/pdf/2512.03803v1](https://arxiv.org/pdf/2512.03803v1)**

> **作者:** Huey Sun; Anabel Yong; Lorenzo Gilly; Felipe Jin
>
> **摘要:** Contrastive decoding is a lightweight and effective inference-time method that improves the quality of text generation in Large Language Models. However, algorithms such as DoLa (Decoding by Contrastive Layers) have only been implemented in decoder-only architectures and studied for their impact on improving factuality. This work adapts DoLa for the T5 and FLAN-T5 model families and evaluates its impact on the models' instruction following capabilities, which to our knowledge is the first implementation of a contrastive decoding strategy in an encoder-decoder architecture. Our results show that DoLa improves the faithfulness of text generation for certain categories of tasks and harms others. To understand these results, we present a layer-by-layer analysis of logit evolution in a FLAN-T5 model to quantify DoLa's impact on token output probabilities.
>
---
#### [new 006] Understanding LLM Reasoning for Abstractive Summarization
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在抽象摘要任务中的推理能力。针对现有假设缺乏验证的问题，通过系统对比8种推理策略与3个大推理模型，在8个数据集上评估摘要质量与事实一致性，发现推理效果依赖具体策略与上下文，且过度推理会损害事实准确性，强调忠实压缩的重要性。**

- **链接: [https://arxiv.org/pdf/2512.03503v1](https://arxiv.org/pdf/2512.03503v1)**

> **作者:** Haohan Yuan; Siu Cheung Hui; Haopeng Zhang
>
> **备注:** 26 pages,15 figures
>
> **摘要:** While the reasoning capabilities of Large Language Models (LLMs) excel in analytical tasks such as mathematics and code generation, their utility for abstractive summarization remains widely assumed but largely unverified. To bridge this gap, we first tailor general reasoning strategies to the summarization domain. We then conduct a systematic, large scale comparative study of 8 reasoning strategies and 3 Large Reasoning Models (LRMs) across 8 diverse datasets, assessing both summary quality and faithfulness. Our findings show that reasoning is not a universal solution and its effectiveness is highly dependent on the specific strategy and context. Specifically, we observe a trade-off between summary quality and factual faithfulness: explicit reasoning strategies tend to improve fluency at the expense of factual grounding, while implicit reasoning in LRMs exhibits the inverse pattern. Furthermore, increasing an LRM's internal reasoning budget does not improve, and can even hurt, factual consistency, suggesting that effective summarization demands faithful compression rather than creative over-thinking.
>
---
#### [new 007] Idea-Gated Transformers: Enforcing Semantic Coherence via Differentiable Vocabulary Pruning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对自回归语言模型的“主题漂移”问题，提出Idea-Gated Transformer架构。通过引入可微分的“概念头”实时抑制无关词汇，实现语义规划与语法生成分离，有效提升生成内容的领域一致性，实现参数高效的可控生成。**

- **链接: [https://arxiv.org/pdf/2512.03343v1](https://arxiv.org/pdf/2512.03343v1)**

> **作者:** Darshan Fofadiya
>
> **备注:** Code available at https://github.com/DarshanFofadiya/idea-gated-transformers/tree/main
>
> **摘要:** Autoregressive Language Models (LLMs) trained on Next-Token Prediction (NTP) often suffer from ``Topic Drift'' where the generation wanders away from the initial prompt due to a reliance on local associations rather than global planning \citep{holtzman2019curious}. While scaling model size mitigates this \citep{brown2020language}, the fundamental myopia of the NTP objective remains. In this work, we introduce the Idea-Gated Transformer, a novel architecture that separates semantic planning from syntactic generation. We introduce an auxiliary ``Idea Head'' trained to predict the bag-of-words distribution for a future context window, creating a latent ``Concept Vector'' that actively gates the main vocabulary during generation. We propose a differentiable gating mechanism that suppresses semantically irrelevant tokens, effectively pruning the search space in real-time. Experiments on WikiText-103 demonstrate that while the Idea-Gated model achieves comparable validation perplexity to a standard GPT-2 baseline, it exhibits significantly superior Domain Retention. Qualitative and quantitative analysis reveals that the gating mechanism successfully locks generation into specific semantic clusters (e.g., Finance, Science) and resists associative drift, offering a parameter-efficient path toward more controllable language modeling.
>
---
#### [new 008] Training and Evaluation of Guideline-Based Medical Reasoning in LLMs
- **分类: cs.CL**

- **简介: 该论文聚焦于医疗领域大模型的推理训练与评估，旨在解决模型预测缺乏可解释性的问题。通过将医学共识指南转化为可执行规则，对小模型进行微调，提升其推理过程的正确性与可信度。实验表明，该方法在未见数据上表现优异，且通过融合时间序列预测模型，有效改善了稀疏临床数据的未来预测能力。**

- **链接: [https://arxiv.org/pdf/2512.03838v1](https://arxiv.org/pdf/2512.03838v1)**

> **作者:** Michael Staniek; Artem Sokolov; Stefan Riezler
>
> **摘要:** Machine learning for early prediction in medicine has recently shown breakthrough performance, however, the focus on improving prediction accuracy has led to a neglect of faithful explanations that are required to gain the trust of medical practitioners. The goal of this paper is to teach LLMs to follow medical consensus guidelines step-by-step in their reasoning and prediction process. Since consensus guidelines are ubiquitous in medicine, instantiations of verbalized medical inference rules to electronic health records provide data for fine-tuning LLMs to learn consensus rules and possible exceptions thereof for many medical areas. Consensus rules also enable an automatic evaluation of the model's inference process regarding its derivation correctness (evaluating correct and faithful deduction of a conclusion from given premises) and value correctness (comparing predicted values against real-world measurements). We exemplify our work using the complex Sepsis-3 consensus definition. Our experiments show that small fine-tuned models outperform one-shot learning of considerably larger LLMs that are prompted with the explicit definition and models that are trained on medical texts including consensus definitions. Since fine-tuning on verbalized rule instantiations of a specific medical area yields nearly perfect derivation correctness for rules (and exceptions) on unseen patient data in that area, the bottleneck for early prediction is not out-of-distribution generalization, but the orthogonal problem of generalization into the future by forecasting sparsely and irregularly sampled clinical variables. We show that the latter results can be improved by integrating the output representations of a time series forecasting model with the LLM in a multimodal setup.
>
---
#### [new 009] Teaching Old Tokenizers New Words: Efficient Tokenizer Adaptation for Pre-trained Models
- **分类: cs.CL**

- **简介: 该论文针对预训练模型的分词器适应问题，提出持续BPE训练以高效扩展词汇，解决新增词汇利用率低的问题；同时引入基于叶节点的词汇剪枝方法，在不损失模型性能前提下减少冗余。二者共同实现可控的分词器优化，提升跨领域/语言迁移效果。**

- **链接: [https://arxiv.org/pdf/2512.03989v1](https://arxiv.org/pdf/2512.03989v1)**

> **作者:** Taido Purason; Pavel Chizhov; Ivan P. Yamshchikov; Mark Fishel
>
> **摘要:** Tokenizer adaptation plays an important role in transferring pre-trained language models to new domains or languages. In this work, we address two complementary aspects of this process: vocabulary extension and pruning. The common approach to extension trains a new tokenizer on domain-specific text and appends the tokens that do not overlap with the existing vocabulary, which often results in many tokens that are unreachable or never used. We propose continued BPE training, which adapts a pre-trained tokenizer by continuing the BPE merge learning process on new data. Experiments across multiple languages and model families show that this approach improves tokenization efficiency and leads to better utilization of added vocabulary. We also introduce leaf-based vocabulary pruning, which removes redundant tokens while preserving model quality. Together, these methods provide practical tools for controlled vocabulary modification, which we release as an open-source package.
>
---
#### [new 010] Nexus: Higher-Order Attention Mechanisms in Transformers
- **分类: cs.CL**

- **简介: 该论文针对Transformer中一阶注意力机制因低秩瓶颈难以捕捉复杂多跳关系的问题，提出高阶注意力网络Hon。通过递归自注意力动态优化查询与键向量，增强模型表达能力，并以参数共享实现高效计算。理论与实验证明其可突破线性瓶颈，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.03377v1](https://arxiv.org/pdf/2512.03377v1)**

> **作者:** Hanting Chen; Chu Zhong; Kai Han; Yuchuan Tian; Yuchen Liang; Tianyu Guo; Xinghao Chen; Dacheng Tao; Yunhe Wang
>
> **摘要:** Transformers have achieved significant success across various domains, relying on self-attention to capture dependencies. However, the standard first-order attention mechanism is often limited by a low-rank bottleneck, struggling to capture intricate, multi-hop relationships within a single layer. In this paper, we propose the \textbf{Higher-Order Attention Network (Hon)}, a novel architecture designed to enhance representational power through a recursive framework. Unlike standard approaches that use static linear projections for Queries and Keys, Hon dynamically refines these representations via nested self-attention mechanisms. Specifically, the Query and Key vectors are themselves outputs of inner attention loops, allowing tokens to aggregate global context and model high-order correlations \textit{prior} to the final attention computation. We enforce a parameter-efficient weight-sharing strategy across recursive steps, ensuring that this enhanced expressivity incurs $\mathcal{O}(1)$ additional parameters. We provide theoretical analysis demonstrating that our method breaks the linear bottleneck of standard attention. Empirically, Hon outperforms standard Transformers on multiple benchmarks.
>
---
#### [new 011] Evaluating Hydro-Science and Engineering Knowledge of Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对大语言模型（LLM）在水科学与工程（Hydro-SE）领域的知识水平评估不足的问题，提出首个专项评测基准Hydro-SE Bench，包含4000道多选题，覆盖九个子领域。通过该基准评估发现，商用LLM表现较好但对行业标准等专业知识仍薄弱，模型规模主要提升推理能力，实际工程应用仍有提升空间。**

- **链接: [https://arxiv.org/pdf/2512.03672v1](https://arxiv.org/pdf/2512.03672v1)**

> **作者:** Shiruo Hu; Wenbo Shan; Yingjia Li; Zhiqi Wan; Xinpeng Yu; Yunjia Qi; Haotian Xia; Yang Xiao; Dingxiao Liu; Jiaru Wang; Chenxu Gong; Ruixi Zhang; Shuyue Wu; Shibo Cui; Chee Hui Lai; Wei Luo; Yubin He; Bin Xu; Jianshi Zhao
>
> **备注:** Hydro-SE Bench sets a new benchmark for the evaluation of LLMs in the Hydro-Science and Engineering domain, with its code and data available at \url{https://github.com/sheishijun/Hydro-SE-Bench}
>
> **摘要:** Hydro-Science and Engineering (Hydro-SE) is a critical and irreplaceable domain that secures human water supply, generates clean hydropower energy, and mitigates flood and drought disasters. Featuring multiple engineering objectives, Hydro-SE is an inherently interdisciplinary domain that integrates scientific knowledge with engineering expertise. This integration necessitates extensive expert collaboration in decision-making, which poses difficulties for intelligence. With the rapid advancement of large language models (LLMs), their potential application in the Hydro-SE domain is being increasingly explored. However, the knowledge and application abilities of LLMs in Hydro-SE have not been sufficiently evaluated. To address this issue, we propose the Hydro-SE LLM evaluation benchmark (Hydro-SE Bench), which contains 4,000 multiple-choice questions. Hydro-SE Bench covers nine subfields and enables evaluation of LLMs in aspects of basic conceptual knowledge, engineering application ability, and reasoning and calculation ability. The evaluation results on Hydro-SE Bench show that the accuracy values vary among 0.74 to 0.80 for commercial LLMs, and among 0.41 to 0.68 for small-parameter LLMs. While LLMs perform well in subfields closely related to natural and physical sciences, they struggle with domain-specific knowledge such as industry standards and hydraulic structures. Model scaling mainly improves reasoning and calculation abilities, but there is still great potential for LLMs to better handle problems in practical engineering application. This study highlights the strengths and weaknesses of LLMs for Hydro-SE tasks, providing model developers with clear training targets and Hydro-SE researchers with practical guidance for applying LLMs.
>
---
#### [new 012] DZ-TDPO: Non-Destructive Temporal Alignment for Mutable State Tracking in Long-Context Dialogue
- **分类: cs.CL**

- **简介: 该论文针对长对话中因历史上下文僵化导致的意图冲突问题，提出DZ-TDPO框架。通过动态KL约束与可学习时序注意力偏置，实现非破坏性对齐，有效缓解状态惯性。实验表明其在多轮对话中达到高胜率，且大模型可实现近乎完美对齐，同时保持强泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.03704v1](https://arxiv.org/pdf/2512.03704v1)**

> **作者:** Yijun Liao
>
> **备注:** 22 pages, 2 figures, 13 tables. Code available at https://github.com/lyj20071013/DZ-TDPO
>
> **摘要:** Long-context dialogue systems suffer from State Inertia, where static constraints prevent models from resolving conflicts between evolving user intents and established historical context. To address this, we propose DZ-TDPO, a non-destructive alignment framework that synergizes conflict-aware dynamic KL constraints with a learnable temporal attention bias. Experiments on the Multi-Session Chat (MSC) dataset demonstrate that DZ-TDPO achieves state-of-the-art win rates (86.2% on Phi-3.5) while maintaining robust zero-shot generalization. Crucially, our scaling analysis reveals a "Capacity-Stability Trade-off": while smaller models incur an "alignment tax" (perplexity surge) to overcome historical inertia, the larger Qwen2.5-7B model achieves near-perfect alignment (99.4% win rate) with negligible perplexity overhead. This confirms that TAI can be alleviated via precise attention regulation rather than destructive weight updates, preserving general capabilities (MMLU) across model scales. Code and data are available: https://github.com/lyj20071013/DZ-TDPO
>
---
#### [new 013] Identifying attributions of causality in political text
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理中的因果关系抽取任务，旨在系统分析政治文本中的因果解释。针对现有研究碎片化、依赖人工标注的问题，作者构建轻量级因果语言模型，自动识别并结构化提取文本中的因果对，实现大规模、高精度的因果分析，兼具低标注成本与良好泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.03214v1](https://arxiv.org/pdf/2512.03214v1)**

> **作者:** Paulina Garcia-Corral
>
> **摘要:** Explanations are a fundamental element of how people make sense of the political world. Citizens routinely ask and answer questions about why events happen, who is responsible, and what could or should be done differently. Yet despite their importance, explanations remain an underdeveloped object of systematic analysis in political science, and existing approaches are fragmented and often issue-specific. I introduce a framework for detecting and parsing explanations in political text. To do this, I train a lightweight causal language model that returns a structured data set of causal claims in the form of cause-effect pairs for downstream analysis. I demonstrate how causal explanations can be studied at scale, and show the method's modest annotation requirements, generalizability, and accuracy relative to human coding.
>
---
#### [new 014] Fine-grained Narrative Classification in Biased News Articles
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于偏见新闻中的细粒度叙事分类任务，旨在识别新闻文章的意识形态倾向、具体叙事框架及修辞技巧。研究构建了首个针对印度媒体的多层级标注数据集INDI-PROP，提出FANTA与TPTC两个基于GPT-4o-mini的多跳推理框架，有效提升偏见、叙事与修辞识别性能。**

- **链接: [https://arxiv.org/pdf/2512.03582v1](https://arxiv.org/pdf/2512.03582v1)**

> **作者:** Zeba Afroz; Harsh Vardhan; Pawan Bhakuni; Aanchal Punia; Rajdeep Kumar; Md. Shad Akhtar
>
> **摘要:** Narratives are the cognitive and emotional scaffolds of propaganda. They organize isolated persuasive techniques into coherent stories that justify actions, attribute blame, and evoke identification with ideological camps. In this paper, we propose a novel fine-grained narrative classification in biased news articles. We also explore article-bias classification as the precursor task to narrative classification and fine-grained persuasive technique identification. We develop INDI-PROP, the first ideologically grounded fine-grained narrative dataset with multi-level annotation for analyzing propaganda in Indian news media. Our dataset INDI-PROP comprises 1,266 articles focusing on two polarizing socio-political events in recent times: CAA and the Farmers' protest. Each article is annotated at three hierarchical levels: (i) ideological article-bias (pro-government, pro-opposition, neutral), (ii) event-specific fine-grained narrative frames anchored in ideological polarity and communicative intent, and (iii) persuasive techniques. We propose FANTA and TPTC, two GPT-4o-mini guided multi-hop prompt-based reasoning frameworks for the bias, narrative, and persuasive technique classification. FANTA leverages multi-layered communicative phenomena by integrating information extraction and contextual framing for hierarchical reasoning. On the other hand, TPTC adopts systematic decomposition of persuasive cues via a two-stage approach. Our evaluation suggests substantial improvement over underlying baselines in each case.
>
---
#### [new 015] In-Context Representation Hijacking
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文提出Doublespeak攻击，一种无需优化的上下文表示劫持方法。通过替换有害词为中性词，使模型内部表征逐渐趋同，实现语义伪装。解决了大模型安全对齐被绕过的难题，揭示了当前对齐机制在表示层面的脆弱性。**

- **链接: [https://arxiv.org/pdf/2512.03771v1](https://arxiv.org/pdf/2512.03771v1)**

> **作者:** Itay Yona; Amir Sarid; Michael Karasik; Yossi Gandelsman
>
> **摘要:** We introduce \textbf{Doublespeak}, a simple \emph{in-context representation hijacking} attack against large language models (LLMs). The attack works by systematically replacing a harmful keyword (e.g., \textit{bomb}) with a benign token (e.g., \textit{carrot}) across multiple in-context examples, provided a prefix to a harmful request. We demonstrate that this substitution leads to the internal representation of the benign token converging toward that of the harmful one, effectively embedding the harmful semantics under a euphemism. As a result, superficially innocuous prompts (e.g., ``How to build a carrot?'') are internally interpreted as disallowed instructions (e.g., ``How to build a bomb?''), thereby bypassing the model's safety alignment. We use interpretability tools to show that this semantic overwrite emerges layer by layer, with benign meanings in early layers converging into harmful semantics in later ones. Doublespeak is optimization-free, broadly transferable across model families, and achieves strong success rates on closed-source and open-source systems, reaching 74\% ASR on Llama-3.3-70B-Instruct with a single-sentence context override. Our findings highlight a new attack surface in the latent space of LLMs, revealing that current alignment strategies are insufficient and should instead operate at the representation level.
>
---
#### [new 016] From Hypothesis to Premises: LLM-based Backward Logical Reasoning with Selective Symbolic Translation
- **分类: cs.CL**

- **简介: 该论文针对自然语言推理任务，解决传统LLM前向推理中冗余、幻觉和语义漂移问题。提出HBLR框架，通过高置信度文本转逻辑形式，采用假设驱动的逆向推理，并结合反思机制保证推理准确性与效率，显著提升推理性能。**

- **链接: [https://arxiv.org/pdf/2512.03360v1](https://arxiv.org/pdf/2512.03360v1)**

> **作者:** Qingchuan Li; Mingyue Cheng; Zirui Liu; Daoyu Wang; Yuting Zeng; Tongxuan Liu
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** Logical reasoning is a core challenge in natural language understanding and a fundamental capability of artificial intelligence, underpinning scientific discovery, mathematical theorem proving, and complex decision-making. Despite the remarkable progress of large language models (LLMs), most current approaches still rely on forward reasoning paradigms, generating step-by-step rationales from premises to conclusions. However, such methods often suffer from redundant inference paths, hallucinated steps, and semantic drift, resulting in inefficient and unreliable reasoning. In this paper, we propose a novel framework, Hypothesis-driven Backward Logical Reasoning (HBLR). The core idea is to integrate confidence-aware symbolic translation with hypothesis-driven backward reasoning. In the translation phase, only high-confidence spans are converted into logical form, such as First-Order Logic (FOL), while uncertain content remains in natural language. A translation reflection module further ensures semantic fidelity by evaluating symbolic outputs and reverting lossy ones back to text when necessary. In the reasoning phase, HBLR simulates human deductive thinking by assuming the conclusion is true and recursively verifying its premises. A reasoning reflection module further identifies and corrects flawed inference steps, enhancing logical coherence. Extensive experiments on five reasoning benchmarks demonstrate that HBLR consistently outperforms strong baselines in both accuracy and efficiency.
>
---
#### [new 017] Watermarks for Embeddings-as-a-Service Large Language Models
- **分类: cs.CL; cs.CR; cs.LG**

- **简介: 该论文针对嵌入即服务（EaaS）模型的仿冒攻击问题，研究文本嵌入水印技术。提出新型水印方法WET，通过线性变换增强水印鲁棒性，有效抵御改写攻击，提升所有权验证可靠性。**

- **链接: [https://arxiv.org/pdf/2512.03079v1](https://arxiv.org/pdf/2512.03079v1)**

> **作者:** Anudeex Shetty
>
> **摘要:** Large Language Models (LLMs) have demonstrated exceptional capabilities in natural language understanding and generation. Based on these LLMs, businesses have started to provide Embeddings-as-a-Service (EaaS), offering feature extraction capabilities (in the form of text embeddings) that benefit downstream natural language processing tasks. However, prior research has demonstrated that EaaS is vulnerable to imitation attacks, where an attacker clones the service's model in a black-box manner without access to the model's internal workings. In response, watermarks have been added to the text embeddings to protect the intellectual property of EaaS providers by allowing them to check for model ownership. This thesis focuses on defending against imitation attacks by investigating EaaS watermarks. To achieve this goal, we unveil novel attacks and propose and validate new watermarking techniques. Firstly, we show that existing EaaS watermarks can be removed through paraphrasing the input text when attackers clone the model during imitation attacks. Our study illustrates that paraphrasing can effectively bypass current state-of-the-art EaaS watermarks across various attack setups (including different paraphrasing techniques and models) and datasets in most instances. This demonstrates a new vulnerability in recent EaaS watermarking techniques. Subsequently, as a countermeasure, we propose a novel watermarking technique, WET (Watermarking EaaS with Linear Transformation), which employs linear transformation of the embeddings. Watermark verification is conducted by applying a reverse transformation and comparing the similarity between recovered and original embeddings. We demonstrate its robustness against paraphrasing attacks with near-perfect verifiability. We conduct detailed ablation studies to assess the significance of each component and hyperparameter in WET.
>
---
#### [new 018] Jina-VLM: Small Multilingual Vision Language Model
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出Jina-VLM，一个2.4B参数的小型多语言视觉语言模型。针对开放领域2B级模型在多语言视觉问答任务中性能不足的问题，采用SigLIP2视觉编码器与Qwen3语言模型结合，通过注意力池化连接实现任意分辨率图像的高效处理，在多语言VQA任务上达到领先水平，同时保持优异的纯文本性能。**

- **链接: [https://arxiv.org/pdf/2512.04032v1](https://arxiv.org/pdf/2512.04032v1)**

> **作者:** Andreas Koukounas; Georgios Mastrapas; Florian Hönicke; Sedigheh Eslami; Guillaume Roncari; Scott Martens; Han Xiao
>
> **备注:** 18 pages, 1-7 main content
>
> **摘要:** We present Jina-VLM, a 2.4B parameter vision-language model that achieves state-of-the-art multilingual visual question answering among open 2B-scale VLMs. The model couples a SigLIP2 vision encoder with a Qwen3 language backbone through an attention-pooling connector that enables token-efficient processing of arbitrary-resolution images. Across standard VQA benchmarks and multilingual evaluations, Jina-VLM outperforms comparable models while preserving competitive text-only performance.
>
---
#### [new 019] SkillFactory: Self-Distillation For Learning Cognitive Behaviors
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SkillFactory方法，旨在让语言模型在强化学习前通过自蒸馏学习认知技能（如验证、回溯）。针对基础模型缺乏这些技能的问题，利用模型自身生成的“银色”推理轨迹进行监督微调，以预置认知偏差。实验表明，该方法提升模型在复杂任务上的泛化性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.04072v1](https://arxiv.org/pdf/2512.04072v1)**

> **作者:** Zayne Sprague; Jack Lu; Manya Wadhwa; Sedrick Keh; Mengye Ren; Greg Durrett
>
> **摘要:** Reasoning models leveraging long chains of thought employ various cognitive skills, such as verification of their answers, backtracking, retrying by an alternate method, and more. Previous work has shown that when a base language model exhibits these skills, training that model further with reinforcement learning (RL) can learn to leverage them. How can we get models to leverage skills that aren't exhibited by base models? Our work, SkillFactory, is a method for fine-tuning models to roughly learn these skills during a supervised fine-tuning (SFT) stage prior to RL. Our approach does not rely on distillation from a stronger model, but instead uses samples from the model itself, rearranged to provide training data in the format of those skills. These "silver" SFT traces may be imperfect, but are nevertheless effective for priming a model to acquire skills during RL. Our evaluation shows that (1) starting from SkillFactory SFT initialization helps a model to generalize to harder variants of a task post-RL, despite lower performance pre-RL; (2) cognitive skills are indeed used by the model; (3) RLed SkillFactory models are more robust to regression on out-of-domain tasks than RLed base models. Our work suggests that inductive biases learned prior to RL help models learn robust cognitive skill use.
>
---
#### [new 020] PERCS: Persona-Guided Controllable Biomedical Summarization Dataset
- **分类: cs.CL**

- **简介: 该论文提出PERCS数据集，解决医学文本摘要中忽视受众差异的问题。针对不同医疗素养人群（公众、医学生等），构建四类定制化摘要，通过医生评审确保准确性和适配性。实验验证了各群体摘要在可读性、词汇和深度上的差异，并为大模型提供基准评测，推动可控医学摘要研究。**

- **链接: [https://arxiv.org/pdf/2512.03340v1](https://arxiv.org/pdf/2512.03340v1)**

> **作者:** Rohan Charudatt Salvi; Chirag Chawla; Dhruv Jain; Swapnil Panigrahi; Md Shad Akhtar; Shweta Yadav
>
> **备注:** 9 pages, 4 figures, 6 tables
>
> **摘要:** Automatic medical text simplification plays a key role in improving health literacy by making complex biomedical research accessible to diverse readers. However, most existing resources assume a single generic audience, overlooking the wide variation in medical literacy and information needs across user groups. To address this limitation, we introduce PERCS (Persona-guided Controllable Summarization), a dataset of biomedical abstracts paired with summaries tailored to four personas: Laypersons, Premedical Students, Non-medical Researchers, and Medical Experts. These personas represent different levels of medical literacy and information needs, emphasizing the need for targeted, audience-specific summarization. Each summary in PERCS was reviewed by physicians for factual accuracy and persona alignment using a detailed error taxonomy. Technical validation shows clear differences in readability, vocabulary, and content depth across personas. Along with describing the dataset, we benchmark four large language models on PERCS using automatic evaluation metrics that assess comprehensiveness, readability, and faithfulness, establishing baseline results for future research. The dataset, annotation guidelines, and evaluation materials are publicly available to support research on persona-specific communication and controllable biomedical summarization.
>
---
#### [new 021] Adapting Large Language Models to Low-Resource Tibetan: A Two-Stage Continual and Supervised Fine-Tuning Study
- **分类: cs.CL**

- **简介: 该论文研究低资源语言藏语的大模型适配问题。针对数据稀缺与跨语言漂移挑战，提出两阶段方法：先持续预训练（CPT）建立藏语语言基础，再监督微调（SFT）实现翻译任务优化。实验显示模型在困惑度和翻译性能上显著提升，揭示了模型适应机制，为低资源语言建模提供可复现框架。**

- **链接: [https://arxiv.org/pdf/2512.03976v1](https://arxiv.org/pdf/2512.03976v1)**

> **作者:** Lifeng Chen; Ryan Lai; Tianming Liu
>
> **摘要:** Adapting large language models (LLMs) to low-resource languages remains a major challenge due to data scarcity and cross-lingual drift. This work presents a two-stage adaptation of Qwen2.5-3B to Tibetan, a morphologically rich and underrepresented language. We employ Continual Pretraining (CPT) to establish Tibetan linguistic grounding, followed by Supervised Fine-Tuning (SFT) for task and translation specialization. Empirical evaluations demonstrate a consistent decrease in perplexity (from 2.98 $\rightarrow$ 1.54) and substantial improvements in Chinese$\rightarrow$Tibetan translation quality (BLEU: 0.046 $\rightarrow$ 0.261; chrF: 2.2 $\rightarrow$ 6.6). Layer-wise analysis across 435 layers in Qwen3-4B reveals that adaptation primarily concentrates on embedding and output heads, with mid--late MLP projections encoding domain-specific transformations. Our findings suggest that CPT constructs a Tibetan semantic manifold while SFT sharpens task alignment with minimal representational disruption. This study provides the first quantitative exploration of Tibetan adaptation dynamics for LLMs, and offers an open, reproducible framework for extending multilingual foundation models to low-resource settings.
>
---
#### [new 022] A Preliminary Study on the Promises and Challenges of Native Top-$k$ Sparse Attention
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究长文本建模中稀疏注意力机制，针对大模型推理成本高的问题，探索原生Top-k注意力在训练与推理中的一致性。通过实验验证其有效性，发现其性能优于全注意力；揭示熵降低现象支持其理论合理性，并评估近似算法精度影响。**

- **链接: [https://arxiv.org/pdf/2512.03494v1](https://arxiv.org/pdf/2512.03494v1)**

> **作者:** Di Xiu; Hongyin Tang; Bolin Rong; Lizhi Yan; Jingang Wang; Yifan Lu; Xunliang Cai
>
> **摘要:** Large Language Models (LLMs) are increasingly prevalent in the field of long-context modeling, however, their inference computational costs have become a critical bottleneck hindering the advancement of tasks such as agents and multimodal applications. This report conducts a preliminary investigation into the effectiveness and theoretical mechanisms of the Top-$k$ Attention mechanism during both the decoding and training phases. First, we validate the effectiveness of exact Top-$k$ Decoding through extensive experimentation. Experiments demonstrate that retaining only the pivotal Keys with the highest similarity to the Query as the context window during the decoding stage achieves performance comparable to, or even surpassing, full attention on downstream tasks such as HELMET and LongBench v2. Second, we further explore the native Top-$k$ Attention training strategy. Experiments confirm that ensuring the consistency between training and inference regarding Top-$k$ Attention operations facilitates the further unlocking of Top-$k$ Decoding's potential, thereby significantly enhancing model performance. Furthermore, considering the high computational complexity of exact Top-$k$ Attention, we investigate the impact of approximate Top-$k$ algorithm precision on downstream tasks. Our research confirms a positive correlation between downstream task performance and approximation fidelity, and we provide statistical evaluations of the Lightning Indexer's precision within the DeepSeek-V3.2-Exp model. Finally, this report provides a theoretical interpretation from the perspective of Entropy. Experimental observations indicate that models subjected to Top-$k$ Attention SFT exhibit a distinct phenomenon of entropy reduction in downstream tasks, which validates the hypothesis that low-entropy states are better adapted to Top-$k$ Decoding.
>
---
#### [new 023] Dual LoRA: Enhancing LoRA with Magnitude and Direction Updates
- **分类: cs.CL**

- **简介: 该论文针对低秩适配（LoRA）在微调大语言模型时因低秩假设导致性能不佳的问题，提出Dual LoRA方法。通过分离更新的幅度与方向，引入ReLU和符号函数，更精准模拟梯度优化过程，在多个NLP任务上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.03402v1](https://arxiv.org/pdf/2512.03402v1)**

> **作者:** Yixing Xu; Chao Li; Xuanwu Yin; Spandan Tiwari; Dong Li; Ashish Sirasao; Emad Barsoum
>
> **摘要:** Low-rank adaptation (LoRA) is one of the most popular methods among parameter-efficient fine-tuning (PEFT) methods to adapt pre-trained large language models (LLMs) to specific downstream tasks. However, the model trained based on LoRA often has an unsatisfactory performance due to its low-rank assumption. In this paper, we propose a novel method called Dual LoRA to improve the performance by incorporating an inductive bias into the original LoRA. Specifically, we separate low-rank matrices into two groups: the magnitude group to control whether or not and how far we should update a parameter and the direction group to decide whether this parameter should move forward or backward, to better simulate the parameter updating process of the full fine-tuning based on gradient-based optimization algorithms. We show that this can be simply achieved by adding a ReLU function to the magnitude group and a sign function to the direction group. We conduct several experiments over a wide range of NLP tasks, including natural language generation (NLG), understanding (NLU), and commonsense reasoning datasets on GPT-2, RoBERTa, DeBERTa, and LLaMA-1/2/3 as baseline models. The results show that we consistently outperform LoRA and its state-of-the-art variants with the same number of trainable parameters.
>
---
#### [new 024] InvertiTune: High-Quality Data Synthesis for Cost-Effective Single-Shot Text-to-Knowledge Graph Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对文本到知识图谱（Text2KG）生成中迭代提示成本高、难以捕捉复杂关系的问题，提出InvertiTune框架。通过可控数据生成与监督微调结合，构建高质量长文本-知识图谱数据集，训练轻量模型实现单次高效生成，显著提升性能与跨数据集泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.03197v1](https://arxiv.org/pdf/2512.03197v1)**

> **作者:** Faezeh Faez; Marzieh S. Tahaei; Yaochen Hu; Ali Pourranjbar; Mahdi Biparva; Mark Coates; Yingxue Zhang
>
> **摘要:** Large Language Models (LLMs) have revolutionized the ability to understand and generate text, enabling significant progress in automatic knowledge graph construction from text (Text2KG). Many Text2KG methods, however, rely on iterative LLM prompting, making them computationally expensive and prone to overlooking complex relations distributed throughout the text. To address these limitations, we propose InvertiTune, a framework that combines a controlled data generation pipeline with supervised fine-tuning (SFT). Within this framework, the data-generation pipeline systematically extracts subgraphs from large knowledge bases, applies noise filtering, and leverages LLMs to generate corresponding natural text descriptions, a task more aligned with LLM capabilities than direct KG generation from text. This pipeline enables generating datasets composed of longer texts paired with larger KGs that better reflect real-world scenarios compared to existing benchmarks, thus supporting effective SFT of lightweight models for single-shot KG construction. Experimental results on CE12k, a dataset generated using the introduced pipeline, show that InvertiTune outperforms larger non-fine-tuned LLMs as well as state-of-the-art Text2KG approaches, while also demonstrating stronger cross-dataset generalization on CrossEval-1200, a test set created from three established benchmark datasets and CE12k. These findings highlight the importance of realistic, high-quality training data for advancing efficient and high-performing Text2KG systems.
>
---
#### [new 025] Entropy-Based Measurement of Value Drift and Alignment Work in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大模型在部署中出现的动态价值漂移问题，提出基于熵的量化方法。通过定义行为分类与伦理熵度量，构建实时监控系统，评估并抑制模型对齐退化，实现运行时安全预警。属于模型安全性评估任务。**

- **链接: [https://arxiv.org/pdf/2512.03047v1](https://arxiv.org/pdf/2512.03047v1)**

> **作者:** Samih Fadli
>
> **备注:** 6 pages. Companion paper to "The Second Law of Intelligence: Controlling Ethical Entropy in Autonomous Systems". Code and tools: https://github.com/AerisSpace/EthicalEntropyKit
>
> **摘要:** Large language model safety is usually assessed with static benchmarks, but key failures are dynamic: value drift under distribution shift, jailbreak attacks, and slow degradation of alignment in deployment. Building on a recent Second Law of Intelligence that treats ethical entropy as a state variable which tends to increase unless countered by alignment work, we make this framework operational for large language models. We define a five-way behavioral taxonomy, train a classifier to estimate ethical entropy S(t) from model transcripts, and measure entropy dynamics for base and instruction-tuned variants of four frontier models across stress tests. Base models show sustained entropy growth, while tuned variants suppress drift and reduce ethical entropy by roughly eighty percent. From these trajectories we estimate an effective alignment work rate gamma_eff and embed S(t) and gamma_eff in a monitoring pipeline that raises alerts when entropy drift exceeds a stability threshold, enabling run-time oversight of value drift.
>
---
#### [new 026] AugServe: Adaptive Request Scheduling for Augmented Large Language Model Inference Serving
- **分类: cs.CL**

- **简介: 该论文针对增强型大语言模型推理服务中的高延迟与低吞吐问题，提出AugServe框架。通过两阶段自适应调度与动态批处理机制，缓解队头阻塞并提升资源利用率，显著降低延迟、提高有效吞吐。**

- **链接: [https://arxiv.org/pdf/2512.04013v1](https://arxiv.org/pdf/2512.04013v1)**

> **作者:** Ying Wang; Zhen Jin; Jiexiong Xu; Wenhai Lin; Yiquan Chen; Wenzhi Chen
>
> **摘要:** As augmented large language models (LLMs) with external tools become increasingly popular in web applications, improving augmented LLM inference serving efficiency and optimizing service-level objectives (SLOs) are critical for enhancing user experience. To achieve this, inference systems must maximize request handling within latency constraints, referred to as increasing effective throughput. However, existing systems face two major challenges: (i) reliance on first-come-first-served (FCFS) scheduling causes severe head-of-line blocking, leading to queuing delays exceeding the SLOs for many requests; and (ii) static batch token limit, which fails to adapt to fluctuating loads and hardware conditions. Both of these factors degrade effective throughput and service quality. This paper presents AugServe, an efficient inference framework designed to reduce queueing latency and enhance effective throughput for augmented LLM inference services. The core idea of AugServe is a two-stage adaptive request scheduling strategy. Specifically, AugServe combines the inference features of augmented LLM requests to optimize the order of scheduling decisions (stage I). These decisions are continuously refined with runtime information (stage II), adapting to both request characteristics and system capabilities. In addition, AugServe dynamically adjusts the token batching mechanism based on hardware status and real-time load, further enhancing throughput performance. Experimental results show that AugServe achieves 4.7-33.1x and 3.3-13.2x higher effective throughput than vLLM and InferCept, while reducing time-to-first-token (TTFT) by up to 96.3% and 95.0%, respectively.
>
---
#### [new 027] BERnaT: Basque Encoders for Representing Natural Textual Diversity
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对语言模型忽视非标准语言变体的问题，提出BERnaT系列编码器模型，结合标准、社交媒体和历史文本构建多样化语料，通过多配置训练与分离评估框架，验证了包含语言多样性可提升模型泛化能力，尤其在非标准文本上表现更优。**

- **链接: [https://arxiv.org/pdf/2512.03903v1](https://arxiv.org/pdf/2512.03903v1)**

> **作者:** Ekhi Azurmendi; Joseba Fernandez de Landa; Jaione Bengoetxea; Maite Heredia; Julen Etxaniz; Mikel Zubillaga; Ander Soraluze; Aitor Soroa
>
> **备注:** Submitted to LREC 2026
>
> **摘要:** Language models depend on massive text corpora that are often filtered for quality, a process that can unintentionally exclude non-standard linguistic varieties, reduce model robustness and reinforce representational biases. In this paper, we argue that language models should aim to capture the full spectrum of language variation (dialectal, historical, informal, etc.) rather than relying solely on standardized text. Focusing on Basque, a morphologically rich and low-resource language, we construct new corpora combining standard, social media, and historical sources, and pre-train the BERnaT family of encoder-only models in three configurations: standard, diverse, and combined. We further propose an evaluation framework that separates Natural Language Understanding (NLU) tasks into standard and diverse subsets to assess linguistic generalization. Results show that models trained on both standard and diverse data consistently outperform those trained on standard corpora, improving performance across all task types without compromising standard benchmark accuracy. These findings highlight the importance of linguistic diversity in building inclusive, generalizable language models.
>
---
#### [new 028] Enhancing Job Matching: Occupation, Skill and Qualification Linking with the ESCO and EQF taxonomies
- **分类: cs.CL; cs.LG**

- **简介: 该论文聚焦于职业匹配中的实体识别任务，旨在通过语言模型将职位描述与ESCO和EQF两大欧洲框架关联。研究比较了句法链接与实体链接方法，构建了两个标注数据集，并开源了工具与代码，以提升劳动力市场信息的分类精度，支持数字经济下工作与技能的自动化分析。**

- **链接: [https://arxiv.org/pdf/2512.03195v1](https://arxiv.org/pdf/2512.03195v1)**

> **作者:** Stylianos Saroglou; Konstantinos Diamantaras; Francesco Preta; Marina Delianidi; Apostolos Benisis; Christian Johannes Meyer
>
> **备注:** 14 pages, 1 figure, Preprint
>
> **摘要:** This study investigates the potential of language models to improve the classification of labor market information by linking job vacancy texts to two major European frameworks: the European Skills, Competences, Qualifications and Occupations (ESCO) taxonomy and the European Qualifications Framework (EQF). We examine and compare two prominent methodologies from the literature: Sentence Linking and Entity Linking. In support of ongoing research, we release an open-source tool, incorporating these two methodologies, designed to facilitate further work on labor classification and employment discourse. To move beyond surface-level skill extraction, we introduce two annotated datasets specifically aimed at evaluating how occupations and qualifications are represented within job vacancy texts. Additionally, we examine different ways to utilize generative large language models for this task. Our findings contribute to advancing the state of the art in job entity extraction and offer computational infrastructure for examining work, skills, and labor market narratives in a digitally mediated economy. Our code is made publicly available: https://github.com/tabiya-tech/tabiya-livelihoods-classifier
>
---
#### [new 029] Is Lying Only Sinful in Islam? Exploring Religious Bias in Multilingual Large Language Models Across Major Religions
- **分类: cs.CL; cs.HC**

- **简介: 该论文针对多语言大模型在宗教议题上的偏见问题，构建了涵盖南亚四大宗教的双语数据集BRAND，通过中英双语提示测试模型表现。结果发现模型在英文下表现优于孟加拉语，且对伊斯兰教存在持续偏见，揭示了跨语言宗教认知偏差的深层问题。**

- **链接: [https://arxiv.org/pdf/2512.03943v1](https://arxiv.org/pdf/2512.03943v1)**

> **作者:** Kazi Abrab Hossain; Jannatul Somiya Mahmud; Maria Hossain Tuli; Anik Mitra; S. M. Taiabul Haque; Farig Y. Sadeque
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** While recent developments in large language models have improved bias detection and classification, sensitive subjects like religion still present challenges because even minor errors can result in severe misunderstandings. In particular, multilingual models often misrepresent religions and have difficulties being accurate in religious contexts. To address this, we introduce BRAND: Bilingual Religious Accountable Norm Dataset, which focuses on the four main religions of South Asia: Buddhism, Christianity, Hinduism, and Islam, containing over 2,400 entries, and we used three different types of prompts in both English and Bengali. Our results indicate that models perform better in English than in Bengali and consistently display bias toward Islam, even when answering religion-neutral questions. These findings highlight persistent bias in multilingual models when similar questions are asked in different languages. We further connect our findings to the broader issues in HCI regarding religion and spirituality.
>
---
#### [new 030] Modeling Topics and Sociolinguistic Variation in Code-Switched Discourse: Insights from Spanish-English and Spanish-Guaraní
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的多语种社会语言学分析任务，旨在解决代码转换话语中话题与社会语言变异的自动识别问题。研究构建了基于大语言模型的标注流程，对西班牙-英语和西班牙-瓜拉尼双语语料进行话题、语体与语用功能标注，揭示性别、语言主导性与话语功能的关联，以及帕拉瓜文本中的正式/非正式语言分层，实现了低资源双语研究的自动化分析。**

- **链接: [https://arxiv.org/pdf/2512.03334v1](https://arxiv.org/pdf/2512.03334v1)**

> **作者:** Nemika Tyagi; Nelvin Licona Guevara; Olga Kellert
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** This study presents an LLM-assisted annotation pipeline for the sociolinguistic and topical analysis of bilingual discourse in two typologically distinct contexts: Spanish-English and Spanish-Guaraní. Using large language models, we automatically labeled topic, genre, and discourse-pragmatic functions across a total of 3,691 code-switched sentences, integrated demographic metadata from the Miami Bilingual Corpus, and enriched the Spanish-Guaraní dataset with new topic annotations. The resulting distributions reveal systematic links between gender, language dominance, and discourse function in the Miami data, and a clear diglossic division between formal Guaraní and informal Spanish in Paraguayan texts. These findings replicate and extend earlier interactional and sociolinguistic observations with corpus-scale quantitative evidence. The study demonstrates that large language models can reliably recover interpretable sociolinguistic patterns traditionally accessible only through manual annotation, advancing computational methods for cross-linguistic and low-resource bilingual research.
>
---
#### [new 031] Improving Alignment Between Human and Machine Codes: An Empirical Assessment of Prompt Engineering for Construct Identification in Psychology
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在心理学文本中识别理论构念的分类任务，针对提示词设计不当导致的性能下降问题，通过实证比较五种提示工程策略，发现构念定义、任务表述和示例是关键因素，提出结合人工与自动提示生成并基于数据验证选择最优提示的系统方法。**

- **链接: [https://arxiv.org/pdf/2512.03818v1](https://arxiv.org/pdf/2512.03818v1)**

> **作者:** Kylie L. Anglin; Stephanie Milan; Brittney Hernandez; Claudia Ventura
>
> **备注:** 22 pages, 2 figures
>
> **摘要:** Due to their architecture and vast pre-training data, large language models (LLMs) demonstrate strong text classification performance. However, LLM output - here, the category assigned to a text - depends heavily on the wording of the prompt. While literature on prompt engineering is expanding, few studies focus on classification tasks, and even fewer address domains like psychology, where constructs have precise, theory-driven definitions that may not be well represented in pre-training data. We present an empirical framework for optimizing LLM performance for identifying constructs in texts via prompt engineering. We experimentally evaluate five prompting strategies --codebook-guided empirical prompt selection, automatic prompt engineering, persona prompting, chain-of-thought reasoning, and explanatory prompting - with zero-shot and few-shot classification. We find that persona, chain-of-thought, and explanations do not fully address performance loss accompanying a badly worded prompt. Instead, the most influential features of a prompt are the construct definition, task framing, and, to a lesser extent, the examples provided. Across three constructs and two models, the classifications most aligned with expert judgments resulted from a few-shot prompt combining codebook-guided empirical prompt selection with automatic prompt engineering. Based on our findings, we recommend that researchers generate and evaluate as many prompt variants as feasible, whether human-crafted, automatically generated, or ideally both, and select prompts and examples based on empirical performance in a training dataset, validating the final approach in a holdout set. This procedure offers a practical, systematic, and theory-driven method for optimizing LLM prompts in settings where alignment with expert judgment is critical.
>
---
#### [new 032] Randomized Masked Finetuning: An Efficient Way to Mitigate Memorization of PIIs in LLMs
- **分类: cs.CL; cs.CR; cs.LG**

- **简介: 该论文针对大语言模型（LLM）中个人身份信息（PII）过度记忆的隐私安全问题，提出随机掩码微调（RMFT）方法。通过在微调阶段随机掩码敏感信息，显著降低PII提取率，同时保持较低性能损失。研究还构建了MaxTER评估框架，验证了RMFT在隐私-效用权衡上的优越性。**

- **链接: [https://arxiv.org/pdf/2512.03310v1](https://arxiv.org/pdf/2512.03310v1)**

> **作者:** Kunj Joshi; David A. Smith
>
> **备注:** To be submitted for ICML 2026
>
> **摘要:** The current literature on memorization in Natural Language Models, especially Large Language Models (LLMs), poses severe security and privacy risks, as models tend to memorize personally identifying information (PIIs) from training data. We introduce Randomized Masked Fine-Tuning (RMFT), a novel privacy-preserving fine-tuning technique that reduces PII memorization while minimizing performance impact. Using the Enron Email Dataset, we demonstrate that RMFT achieves an 80.81% reduction in Total Extraction Rate and 80.17% reduction in Seen Extraction Rate compared to baseline fine-tuning, outperforming deduplication methods while maintaining only a 5.73% increase in perplexity. We present MaxTER, a Pareto-optimal evaluation framework for assessing privacy-utility tradeoffs, and show the performance of RMFT vs Deduplication by Area Under The Response Curve (AURC) metric.
>
---
#### [new 033] Alleviating Choice Supportive Bias in LLM with Reasoning Dependency Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型在评估中存在选择性支持偏差（CSB）的问题，提出基于推理依赖生成（RDG）的解决方案。通过构建平衡的推理问答对，显式建模或解耦选项、证据与理由间的依赖关系，生成大规模无偏数据用于微调，显著降低CSB，提升决策客观性。**

- **链接: [https://arxiv.org/pdf/2512.03082v1](https://arxiv.org/pdf/2512.03082v1)**

> **作者:** Nan Zhuang; Wenshuo Wang; Lekai Qian; Yuxiao Wang; Boyu Cao; Qi Liu
>
> **摘要:** Recent studies have demonstrated that some Large Language Models exhibit choice-supportive bias (CSB) when performing evaluations, systematically favoring their chosen options and potentially compromising the objectivity of AI-assisted decision making. While existing debiasing approaches primarily target demographic and social biases, methods for addressing cognitive biases in LLMs remain largely unexplored. In this work, we present the first solution to address CSB through Reasoning Dependency Generation (RDG), a novel framework for generating unbiased reasoning data to mitigate choice-supportive bias through fine-tuning. RDG automatically constructs balanced reasoning QA pairs, explicitly (un)modeling the dependencies between choices, evidences, and justifications. Our approach is able to generate a large-scale dataset of QA pairs across domains, incorporating Contextual Dependency Data and Dependency Decouple Data. Experiments show that LLMs fine-tuned on RDG-generated data demonstrate a 81.5% improvement in memory-based experiments and 94.3% improvement in the evaluation-based experiment, while maintaining similar performance on standard BBQ benchmarks. This work pioneers an approach for addressing cognitive biases in LLMs and contributes to the development of more reliable AI-assisted decision support systems.
>
---
#### [new 034] PretrainZero: Reinforcement Active Pretraining
- **分类: cs.CL**

- **简介: 该论文提出PretrainZero，一种基于预训练语料的强化主动学习框架，旨在突破通用推理中对可验证奖励的依赖。通过无监督强化学习主动筛选并推理预训练数据，实现无需标注或微调的通用推理能力提升，在多个基准上显著优于基线模型。**

- **链接: [https://arxiv.org/pdf/2512.03442v1](https://arxiv.org/pdf/2512.03442v1)**

> **作者:** Xingrun Xing; Zhiyuan Fan; Jie Lou; Guoqi Li; Jiajun Zhang; Debing Zhang
>
> **摘要:** Mimicking human behavior to actively learning from general experience and achieve artificial general intelligence has always been a human dream. Recent reinforcement learning (RL) based large-thinking models demonstrate impressive expert-level abilities, i.e., software and math, but still rely heavily on verifiable rewards in specific domains, placing a significant bottleneck to extend the performance boundary of general reasoning capabilities. In this work, we propose PretrainZero, a reinforcement active learning framework built on the pretraining corpus to extend RL from domain-specific post-training to general pretraining. PretrainZero features the following characteristics: 1) Active pretraining: inspired by the active learning ability of humans, PretrainZero learns a unified reasoning policy to actively identify reasonable and informative contents from pretraining corpus, and reason to predict these contents by RL. 2) Self-supervised learning: without any verifiable labels, pretrained reward models, or supervised fine-tuning, we directly pretrain reasoners from 3 to 30B base models on the general Wikipedia corpus using RL, significantly breaking the verification data-wall for general reasoning. 3) Verification scaling: by tackling increasingly challenging masked spans, PretrainZero substantially enhances the general reasoning abilities of pretrained base models. In reinforcement pretraining, PretrainZero improves Qwen3-4B-Base for 8.43, 5.96 and 10.60 on MMLU-Pro, SuperGPQA and math average benchmarks. In post-training, the pretrained models can also serve as reasoning foundation models for downstream RLVR tasks.
>
---
#### [new 035] AITutor-EvalKit: Exploring the Capabilities of AI Tutors
- **分类: cs.CL**

- **简介: 该论文提出AITutor-EvalKit，一个用于评估AI导师教学质量的工具。针对AI教育应用中缺乏有效评估手段的问题，构建了支持教学评估、模型分析与数据可视化的软件系统，服务于教育研究者与ACL社区，促进AI tutor的优化与反馈收集。**

- **链接: [https://arxiv.org/pdf/2512.03688v1](https://arxiv.org/pdf/2512.03688v1)**

> **作者:** Numaan Naeem; Kaushal Kumar Maurya; Kseniia Petukhova; Ekaterina Kochmar
>
> **摘要:** We present AITutor-EvalKit, an application that uses language technology to evaluate the pedagogical quality of AI tutors, provides software for demonstration and evaluation, as well as model inspection and data visualization. This tool is aimed at education stakeholders as well as *ACL community at large, as it supports learning and can also be used to collect user feedback and annotations.
>
---
#### [new 036] AlignCheck: a Semantic Open-Domain Metric for Factual Consistency Assessment
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AlignCheck，一种用于评估文本事实一致性的可解释度量方法，针对大语言模型生成内容中的幻觉问题。通过分解文本为原子事实，采用灵活的加权评估机制，提升对通用与临床领域文本的事实一致性判断能力，并支持复杂场景下的评估控制。**

- **链接: [https://arxiv.org/pdf/2512.03634v1](https://arxiv.org/pdf/2512.03634v1)**

> **作者:** Ahmad Aghaebrahimian
>
> **摘要:** Large Language Models have significantly advanced natural language processing tasks, but remain prone to generating incorrect or misleading but plausible arguments. This issue, known as hallucination, is particularly concerning in high-stakes domains like clinical applications, where factual inaccuracies can have severe consequences. Existing evaluation metrics fail to adequately assess factual consistency and lack interpretability, making diagnosing and mitigating errors difficult. We propose an interpretable framework for factual consistency assessment for in-domain and open-domain texts to address these limitations. Our approach decomposes text into atomic facts and introduces a flexible, schema-free methodology. Unlike previous methods with an absolute metric, we incorporate a weighted metric to enhance factual evaluation. Additionally, we propose a mechanism to control assessment complexity in intricate domains. We benchmark our approach on popular general and clinical datasets and release our code to support fact-aware model training in future research.
>
---
#### [new 037] Generative AI Practices, Literacy, and Divides: An Empirical Analysis in the Italian Context
- **分类: cs.CL**

- **简介: 该论文属于实证研究任务，旨在分析意大利生成式AI的采用、使用模式与数字素养状况。通过1906名成人的调查数据，发现GenAI广泛用于工作与生活，但用户素养低且存在显著性别差距，揭示了技术普及中的不平等现象，强调需加强教育并探究深层障碍。**

- **链接: [https://arxiv.org/pdf/2512.03671v1](https://arxiv.org/pdf/2512.03671v1)**

> **作者:** Beatrice Savoldi; Giuseppe Attanasio; Olga Gorodetskaya; Marta Marchiori Manerba; Elisa Bassignana; Silvia Casola; Matteo Negri; Tommaso Caselli; Luisa Bentivogli; Alan Ramponi; Arianna Muti; Nicoletta Balbo; Debora Nozza
>
> **摘要:** The rise of Artificial Intelligence (AI) language technologies, particularly generative AI (GenAI) chatbots accessible via conversational interfaces, is transforming digital interactions. While these tools hold societal promise, they also risk widening digital divides due to uneven adoption and low awareness of their limitations. This study presents the first comprehensive empirical mapping of GenAI adoption, usage patterns, and literacy in Italy, based on newly collected survey data from 1,906 Italian-speaking adults. Our findings reveal widespread adoption for both work and personal use, including sensitive tasks like emotional support and medical advice. Crucially, GenAI is supplanting other technologies to become a primary information source: this trend persists despite low user digital literacy, posing a risk as users struggle to recognize errors or misinformation. Moreover, we identify a significant gender divide -- particularly pronounced in older generations -- where women are half as likely to adopt GenAI and use it less frequently than men. While we find literacy to be a key predictor of adoption, it only partially explains this disparity, suggesting that other barriers are at play. Overall, our data provide granular insights into the multipurpose usage of GenAI, highlighting the dual need for targeted educational initiatives and further investigation into the underlying barriers to equitable participation that competence alone cannot explain.
>
---
#### [new 038] Different types of syntactic agreement recruit the same units within large language models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLM）中语法知识的表征问题，聚焦不同句法一致现象是否共享神经单元。通过功能定位方法，发现多种一致类型（如主谓、指代、限定词-名词）均激活重叠的模型单元，且跨语言分析显示结构相似语言共享更多单元。结果表明，句法一致在LLM中构成一个有意义的功能类别。**

- **链接: [https://arxiv.org/pdf/2512.03676v1](https://arxiv.org/pdf/2512.03676v1)**

> **作者:** Daria Kryvosheieva; Andrea de Varda; Evelina Fedorenko; Greta Tuckute
>
> **摘要:** Large language models (LLMs) can reliably distinguish grammatical from ungrammatical sentences, but how grammatical knowledge is represented within the models remains an open question. We investigate whether different syntactic phenomena recruit shared or distinct components in LLMs. Using a functional localization approach inspired by cognitive neuroscience, we identify the LLM units most responsive to 67 English syntactic phenomena in seven open-weight models. These units are consistently recruited across sentences containing the phenomena and causally support the models' syntactic performance. Critically, different types of syntactic agreement (e.g., subject-verb, anaphor, determiner-noun) recruit overlapping sets of units, suggesting that agreement constitutes a meaningful functional category for LLMs. This pattern holds in English, Russian, and Chinese; and further, in a cross-lingual analysis of 57 diverse languages, structurally more similar languages share more units for subject-verb agreement. Taken together, these findings reveal that syntactic agreement-a critical marker of syntactic dependencies-constitutes a meaningful category within LLMs' representational spaces.
>
---
#### [new 039] AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对视觉问答任务中视觉语言模型计算开销大的问题，提出AdaptVision，通过自适应视觉信息获取机制，实现动态减少视觉标记数量。其核心是基于粗到精策略与强化学习的工具调用机制，有效平衡精度与效率，显著降低资源消耗并提升性能。**

- **链接: [https://arxiv.org/pdf/2512.03794v1](https://arxiv.org/pdf/2512.03794v1)**

> **作者:** Zichuan Lin; Yicheng Liu; Yang Yang; Lvfang Tao; Deheng Ye
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Vision-Language Models (VLMs) have achieved remarkable success in visual question answering tasks, but their reliance on large numbers of visual tokens introduces significant computational overhead. While existing efficient VLM approaches reduce visual tokens through fixed-ratio compression, they operate passively and lack the ability to adapt to varying task requirements. This motivates a fundamental question: Can VLMs autonomously determine the minimum number of visual tokens required for each sample? Inspired by human active vision mechanisms, we introduce AdaptVision, an efficient VLM paradigm that enables adaptive visual token acquisition through a coarse-to-fine approach. Our model initially processes compressed visual tokens from low-resolution images and selectively acquires additional visual information by invoking a bounding box tool to crop key regions when necessary. We train AdaptVision using a reinforcement learning framework that carefully balances accuracy and efficiency. Central to our approach is Decoupled Turn Policy Optimization (DTPO), which decouples the learning objective into two components: (1) tool learning, which optimizes correct tool utilization, and (2) accuracy improvement, which refines the generated responses to improve answer correctness. Based on this formulation, we further decouple advantage estimation by computing separate advantages for tokens associated with each objective. This formulation enables more effective optimization for AdaptVision compared to vanilla GRPO. Comprehensive experiments across multiple VQA benchmarks demonstrate that AdaptVision achieves superior performance while consuming substantially fewer visual tokens than state-of-the-art efficient VLM methods.
>
---
#### [new 040] NAS-LoRA: Empowering Parameter-Efficient Fine-Tuning for Visual Foundation Models with Searchable Adaptation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对视觉基础模型SAM在医疗、农业等特定领域适应性差的问题，提出NAS-LoRA方法。通过引入轻量级神经架构搜索模块，动态优化权重更新中的先验知识，结合分阶段优化策略，提升模型对高层语义信息的捕捉能力，显著改善适配性能并降低24.14%训练成本。**

- **链接: [https://arxiv.org/pdf/2512.03499v1](https://arxiv.org/pdf/2512.03499v1)**

> **作者:** Renqi Chen; Haoyang Su; Shixiang Tang
>
> **摘要:** The Segment Anything Model (SAM) has emerged as a powerful visual foundation model for image segmentation. However, adapting SAM to specific downstream tasks, such as medical and agricultural imaging, remains a significant challenge. To address this, Low-Rank Adaptation (LoRA) and its variants have been widely employed to enhancing SAM's adaptation performance on diverse domains. Despite advancements, a critical question arises: can we integrate inductive bias into the model? This is particularly relevant since the Transformer encoder in SAM inherently lacks spatial priors within image patches, potentially hindering the acquisition of high-level semantic information. In this paper, we propose NAS-LoRA, a new Parameter-Efficient Fine-Tuning (PEFT) method designed to bridge the semantic gap between pre-trained SAM and specialized domains. Specifically, NAS-LoRA incorporates a lightweight Neural Architecture Search (NAS) block between the encoder and decoder components of LoRA to dynamically optimize the prior knowledge integrated into weight updates. Furthermore, we propose a stage-wise optimization strategy to help the ViT encoder balance weight updates and architectural adjustments, facilitating the gradual learning of high-level semantic information. Various Experiments demonstrate our NAS-LoRA improves existing PEFT methods, while reducing training cost by 24.14% without increasing inference cost, highlighting the potential of NAS in enhancing PEFT for visual foundation models.
>
---
#### [new 041] Epistemic Substitution: How Grokipedia's AI-Generated Encyclopedia Restructures Authority
- **分类: cs.SI; cs.CL; cs.CY; cs.DL**

- **简介: 该论文属于对比分析任务，旨在探究AI生成百科（Grokipedia）与人类编辑百科（Wikipedia）在知识权威性来源上的差异。通过分析72对匹配条目的引用网络，发现Grokipedia更依赖用户生成与公民组织内容，并呈现不同主题的差异化引用模式，揭示了AI知识生产重构了传统权威基础。**

- **链接: [https://arxiv.org/pdf/2512.03337v1](https://arxiv.org/pdf/2512.03337v1)**

> **作者:** Aliakbar Mehdizadeh; Martin Hilbert
>
> **摘要:** A quarter century ago, Wikipedia's decentralized, crowdsourced, and consensus-driven model replaced the centralized, expert-driven, and authority-based standard for encyclopedic knowledge curation. The emergence of generative AI encyclopedias, such as Grokipedia, possibly presents another potential shift in epistemic evolution. This study investigates whether AI- and human-curated encyclopedias rely on the same foundations of authority. We conducted a multi-scale comparative analysis of the citation networks from 72 matched article pairs, which cite a total of almost 60,000 sources. Using an 8-category epistemic classification, we mapped the "epistemic profiles" of the articles on each platform. Our findings reveal several quantitative and qualitative differences in how knowledge is sourced and encyclopedia claims are epistemologically justified. Grokipedia replaces Wikipedia's heavy reliance on peer-reviewed "Academic & Scholarly" work with a notable increase in "User-generated" and "Civic organization" sources. Comparative network analyses further show that Grokipedia employs very different epistemological profiles when sourcing leisure topics (such as Sports and Entertainment) and more societal sensitive civic topics (such as Politics & Conflicts, Geographical Entities, and General Knowledge & Society). Finally, we find a "scaling-law for AI-generated knowledge sourcing" that shows a linear relationship between article length and citation density, which is distinct from collective human reference sourcing. We conclude that this first implementation of an LLM-based encyclopedia does not merely automate knowledge production but restructures it. Given the notable changes and the important role of encyclopedias, we suggest the continuation and deepening of algorithm audits, such as the one presented here, in order to understand the ongoing epistemological shifts.
>
---
#### [new 042] Tuning for TraceTarnish: Techniques, Trends, and Testing Tangible Traits
- **分类: cs.CR; cs.CL; cs.IR**

- **简介: 该论文研究对抗性风格学在文本匿名中的应用，旨在检测通过《TraceTarnish》攻击篡改的文本。针对作者身份隐藏行为，提取函数词、内容词频率及类型-标记比率等关键特征，识别篡改痕迹，揭示攻击留下的可检测信号，提升攻击有效性并为防御提供侦测依据。**

- **链接: [https://arxiv.org/pdf/2512.03465v1](https://arxiv.org/pdf/2512.03465v1)**

> **作者:** Robert Dilworth
>
> **备注:** 20 pages, 8 figures, 2 tables
>
> **摘要:** In this study, we more rigorously evaluated our attack script $\textit{TraceTarnish}$, which leverages adversarial stylometry principles to anonymize the authorship of text-based messages. To ensure the efficacy and utility of our attack, we sourced, processed, and analyzed Reddit comments--comments that were later alchemized into $\textit{TraceTarnish}$ data--to gain valuable insights. The transformed $\textit{TraceTarnish}$ data was then further augmented by $\textit{StyloMetrix}$ to manufacture stylometric features--features that were culled using the Information Gain criterion, leaving only the most informative, predictive, and discriminative ones. Our results found that function words and function word types ($L\_FUNC\_A$ $\&$ $L\_FUNC\_T$); content words and content word types ($L\_CONT\_A$ $\&$ $L\_CONT\_T$); and the Type-Token Ratio ($ST\_TYPE\_TOKEN\_RATIO\_LEMMAS$) yielded significant Information-Gain readings. The identified stylometric cues--function-word frequencies, content-word distributions, and the Type-Token Ratio--serve as reliable indicators of compromise (IoCs), revealing when a text has been deliberately altered to mask its true author. Similarly, these features could function as forensic beacons, alerting defenders to the presence of an adversarial stylometry attack; granted, in the absence of the original message, this signal may go largely unnoticed, as it appears to depend on a pre- and post-transformation comparison. "In trying to erase a trace, you often imprint a larger one." Armed with this understanding, we framed $\textit{TraceTarnish}$'s operations and outputs around these five isolated features, using them to conceptualize and implement enhancements that further strengthen the attack.
>
---
#### [new 043] Text-Printed Image: Bridging the Image-Text Modality Gap for Text-centric Training of Large Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究文本主导的视觉语言模型训练，旨在解决真实图像数据收集成本高、难扩展的问题。提出Text-Printed Image（TPI）方法，将文本直接渲染为合成图像，弥合图文模态差距，实现低成本、可自动扩增的数据生成，显著提升文本中心训练效果。**

- **链接: [https://arxiv.org/pdf/2512.03463v1](https://arxiv.org/pdf/2512.03463v1)**

> **作者:** Shojiro Yamabe; Futa Waseda; Daiki Shiono; Tsubasa Takahashi
>
> **摘要:** Recent large vision-language models (LVLMs) have been applied to diverse VQA tasks. However, achieving practical performance typically requires task-specific fine-tuning with large numbers of image-text pairs, which are costly to collect. In this work, we study text-centric training, a setting where only textual descriptions are available and no real images are provided, as a paradigm for low-cost data scaling. Unlike images, whose collection is often restricted by privacy constraints and scarcity in niche domains, text is widely available. Moreover, text is easily editable, enabling automatic diversification and expansion with LLMs at minimal human effort. While this offers clear advantages over image collection in terms of scalability and cost, training on raw text without images still yields limited gains on VQA tasks because of the image-text modality gap. To address this issue, we propose a Text-Printed Image (TPI), which generates synthetic images by directly rendering the given textual description on a plain white canvas. This simple rendering projects text into the image modality and can be integrated into arbitrary existing LVLM training pipelines at low cost. Moreover, TPI preserves the semantics of the text, whereas text-to-image models often fail to do. Across four models and seven benchmarks, our systematic experiments show that TPI enables more effective text-centric training than synthetic images generated by a diffusion model. We further explore TPI as a low-cost data-augmentation strategy and demonstrate its practical utility. Overall, our findings highlight the significant potential of text-centric training and, more broadly, chart a path toward fully automated data generation for LVLMs.
>
---
#### [new 044] LLM-Generated Ads: From Personalization Parity to Persuasion Superiority
- **分类: cs.CY; cs.CL**

- **简介: 该论文研究大语言模型（LLM）生成广告的效能。针对个性化与普遍说服力两大问题，通过两阶段实验发现：LLM在个性化广告上达成人类水平，在心理说服策略上显著优于人类，且优势在检测识别后仍存，表明其在广告生成中已实现从个性匹配到说服主导的跃迁。**

- **链接: [https://arxiv.org/pdf/2512.03373v1](https://arxiv.org/pdf/2512.03373v1)**

> **作者:** Elyas Meguellati; Stefano Civelli; Lei Han; Abraham Bernstein; Shazia Sadiq; Gianluca Demartini
>
> **摘要:** As large language models (LLMs) become increasingly capable of generating persuasive content, understanding their effectiveness across different advertising strategies becomes critical. This paper presents a two-part investigation examining LLM-generated advertising through complementary lenses: (1) personality-based and (2) psychological persuasion principles. In our first study (n=400), we tested whether LLMs could generate personalized advertisements tailored to specific personality traits (openness and neuroticism) and how their performance compared to human experts. Results showed that LLM-generated ads achieved statistical parity with human-written ads (51.1% vs. 48.9%, p > 0.05), with no significant performance differences for matched personalities. Building on these insights, our second study (n=800) shifted focus from individual personalization to universal persuasion, testing LLM performance across four foundational psychological principles: authority, consensus, cognition, and scarcity. AI-generated ads significantly outperformed human-created content, achieving a 59.1% preference rate (vs. 40.9%, p < 0.001), with the strongest performance in authority (63.0%) and consensus (62.5%) appeals. Qualitative analysis revealed AI's advantage stems from crafting more sophisticated, aspirational messages and achieving superior visual-narrative coherence. Critically, this quality advantage proved robust: even after applying a 21.2 percentage point detection penalty when participants correctly identified AI-origin, AI ads still outperformed human ads, and 29.4% of participants chose AI content despite knowing its origin. These findings demonstrate LLMs' evolution from parity in personalization to superiority in persuasive storytelling, with significant implications for advertising practice given LLMs' near-zero marginal cost and time requirements compared to human experts.
>
---
#### [new 045] Thinking with Programming Vision: Towards a Unified View for Thinking with Images
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态大模型在图像推理中工具使用受限、鲁棒性差的问题，提出CodeVision框架，通过生成代码作为通用工具接口，实现灵活、可扩展的图像操作。采用两阶段训练策略，结合高质量数据与密集过程奖励，显著提升模型在复杂任务中的表现与错误恢复能力。**

- **链接: [https://arxiv.org/pdf/2512.03746v1](https://arxiv.org/pdf/2512.03746v1)**

> **作者:** Zirun Guo; Minjie Hong; Feng Zhang; Kai Jia; Tao Jin
>
> **摘要:** Multimodal large language models (MLLMs) that think with images can interactively use tools to reason about visual inputs, but current approaches often rely on a narrow set of tools with limited real-world necessity and scalability. In this work, we first reveal a critical and previously overlooked weakness: even state-of-the-art MLLMs are surprisingly brittle, showing significant performance degradation on images with simple orientation changes or natural corruptions, underscoring the need for more robust tool-based reasoning. To address this, we propose CodeVision, a flexible and scalable code-as-tool framework where the model generates code as a universal interface to invoke any image operation, moving beyond fixed tool registries. We train our model using a two-stage methodology, beginning with Supervised Fine-Tuning (SFT) on a high-quality dataset curated for complex, multi-turn tool composition and error recovery, followed by Reinforcement Learning (RL) with a novel and dense process reward function to encourage strategic and efficient tool use. To facilitate this research, we construct new SFT and RL datasets and introduce a challenging new benchmark suite designed to rigorously evaluate robustness to orientation changes and multi-tool reasoning. Experiments on Qwen2.5-VL and Qwen3-VL series show that our approach significantly improves model performance and fosters emergent capabilities such as flexible tool composition, efficient chained execution, and robust error recovery from runtime feedback. Code is available at https://github.com/ByteDance-BandAI/CodeVision.
>
---
#### [new 046] SPARK: Stepwise Process-Aware Rewards for Reference-Free Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对强化学习中依赖昂贵参考答案的问题，提出SPARK框架，通过生成-验证机制构建无参考的步骤级奖励模型。利用自洽与元批评实现多步验证，生成高质量训练数据，提升数学推理任务中的奖励信号质量，最终在无参考条件下超越基于真值的强化学习方法。**

- **链接: [https://arxiv.org/pdf/2512.03244v1](https://arxiv.org/pdf/2512.03244v1)**

> **作者:** Salman Rahman; Sruthi Gorantla; Arpit Gupta; Swastik Roy; Nanyun Peng; Yang Liu
>
> **摘要:** Process reward models (PRMs) that provide dense, step-level feedback have shown promise for reinforcement learning, yet their adoption remains limited by the need for expensive step-level annotations or ground truth references. We propose SPARK: a three-stage framework where in the first stage a generator model produces diverse solutions and a verifier model evaluates them using parallel scaling (self-consistency) and sequential scaling (meta-critique). In the second stage, we use these verification outputs as synthetic training data to fine-tune generative process reward models, which subsequently serve as reward signals during training. We show that aggregating multiple independent verifications at the step level produces training data for process reward models that surpass ground-truth outcome supervision, achieving 67.5 F1 on ProcessBench (a benchmark for identifying erroneous steps in mathematical reasoning) compared to 66.4 for reference-guided training and 61.9 for GPT-4o. In the final stage, we apply our generative PRM with chain-of-thought verification (PRM-CoT) as the reward model in RL experiments on mathematical reasoning, and introduce format constraints to prevent reward hacking. Using Qwen2.5-Math-7B, we achieve 47.4% average accuracy across six mathematical reasoning benchmarks, outperforming ground-truth-based RLVR (43.9%). Our work enables reference-free RL training that exceeds ground-truth methods, opening new possibilities for domains lacking verifiable answers or accessible ground truth.
>
---
#### [new 047] SELF: A Robust Singular Value and Eigenvalue Approach for LLM Fingerprinting
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对大模型知识产权保护中的指纹伪造与权重篡改问题，提出SELF方法。通过注意力权重的奇异值与特征值分解实现不变指纹提取，并结合少样本学习与数据增强进行相似性比对，有效抵御量化、剪枝等攻击，实现鲁棒的模型侵权检测。**

- **链接: [https://arxiv.org/pdf/2512.03620v1](https://arxiv.org/pdf/2512.03620v1)**

> **作者:** Hanxiu Zhang; Yue Zheng
>
> **摘要:** The protection of Intellectual Property (IP) in Large Language Models (LLMs) represents a critical challenge in contemporary AI research. While fingerprinting techniques have emerged as a fundamental mechanism for detecting unauthorized model usage, existing methods -- whether behavior-based or structural -- suffer from vulnerabilities such as false claim attacks or susceptible to weight manipulations. To overcome these limitations, we propose SELF, a novel intrinsic weight-based fingerprinting scheme that eliminates dependency on input and inherently resists false claims. SELF achieves robust IP protection through two key innovations: 1) unique, scalable and transformation-invariant fingerprint extraction via singular value and eigenvalue decomposition of LLM attention weights, and 2) effective neural network-based fingerprint similarity comparison based on few-shot learning and data augmentation. Experimental results demonstrate SELF maintains high IP infringement detection accuracy while showing strong robustness against various downstream modifications, including quantization, pruning, and fine-tuning attacks. Our code is available at https://github.com/HanxiuZhang/SELF_v2.
>
---
#### [new 048] Is Vibe Coding Safe? Benchmarking Vulnerability of Agent-Generated Code in Real-World Tasks
- **分类: cs.SE; cs.CL**

- **简介: 该论文研究Vibe Coding（人类指令大模型完成复杂编程任务）的安全性问题。针对其在真实场景中可能引入漏洞的隐患，构建了包含200个真实开源项目特征请求的基准测试集，评估多个主流编码代理。结果表明，尽管多数代码功能正确，但安全率极低，且现有安全策略无效，揭示了该范式在安全敏感应用中的重大风险。**

- **链接: [https://arxiv.org/pdf/2512.03262v1](https://arxiv.org/pdf/2512.03262v1)**

> **作者:** Songwen Zhao; Danqing Wang; Kexun Zhang; Jiaxuan Luo; Zhuo Li; Lei Li
>
> **摘要:** Vibe coding is a new programming paradigm in which human engineers instruct large language model (LLM) agents to complete complex coding tasks with little supervision. Although it is increasingly adopted, are vibe coding outputs really safe to deploy in production? To answer this question, we propose SU S VI B E S, a benchmark consisting of 200 feature-request software engineering tasks from real-world open-source projects, which, when given to human programmers, led to vulnerable implementations. We evaluate multiple widely used coding agents with frontier models on this benchmark. Disturbingly, all agents perform poorly in terms of software security. Although 61% of the solutions from SWE-Agent with Claude 4 Sonnet are functionally correct, only 10.5% are secure. Further experiments demonstrate that preliminary security strategies, such as augmenting the feature request with vulnerability hints, cannot mitigate these security issues. Our findings raise serious concerns about the widespread adoption of vibe-coding, particularly in security-sensitive applications.
>
---
#### [new 049] Detecting AI Hallucinations in Finance: An Information-Theoretic Method Cuts Hallucination Rate by 92%
- **分类: cs.LG; cs.CL; q-fin.CP; stat.ML**

- **简介: 该论文针对金融领域大模型幻觉检测问题，提出ECLIPSE框架，通过信息论方法衡量语义熵与证据容量的不匹配。利用多样本聚类和困惑度分解，实现对模型证据使用情况的量化，显著降低幻觉率，验证了其有效性与对概率校准的依赖性。**

- **链接: [https://arxiv.org/pdf/2512.03107v1](https://arxiv.org/pdf/2512.03107v1)**

> **作者:** Mainak Singha
>
> **备注:** 17 pages, 7 figures. Information-theoretic, hallucination detector for financial application. Feedback from researchers and practitioners is welcome
>
> **摘要:** Large language models (LLMs) produce fluent but unsupported answers - hallucinations - limiting safe deployment in high-stakes domains. We propose ECLIPSE, a framework that treats hallucination as a mismatch between a model's semantic entropy and the capacity of available evidence. We combine entropy estimation via multi-sample clustering with a novel perplexity decomposition that measures how models use retrieved evidence. We prove that under mild conditions, the resulting entropy-capacity objective is strictly convex with a unique stable optimum. We evaluate on a controlled financial question answering dataset with GPT-3.5-turbo (n=200 balanced samples with synthetic hallucinations), where ECLIPSE achieves ROC AUC of 0.89 and average precision of 0.90, substantially outperforming a semantic entropy-only baseline (AUC 0.50). A controlled ablation with Claude-3-Haiku, which lacks token-level log probabilities, shows AUC dropping to 0.59 with coefficient magnitudes decreasing by 95% - demonstrating that ECLIPSE is a logprob-native mechanism whose effectiveness depends on calibrated token-level uncertainties. The perplexity decomposition features exhibit the largest learned coefficients, confirming that evidence utilization is central to hallucination detection. We position this work as a controlled mechanism study; broader validation across domains and naturally occurring hallucinations remains future work.
>
---
#### [new 050] Optical Context Compression Is Just (Bad) Autoencoding
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文针对视觉上下文压缩在语言模型中的应用，检验其有效性。通过对比视觉编码器与简单替代方法，发现后者在文本重建和语言建模任务中表现相当或更优，且视觉压缩无法超越截断策略。结论质疑了光学压缩的优越性，指出当前兴奋过度，缺乏实证支持。**

- **链接: [https://arxiv.org/pdf/2512.03643v1](https://arxiv.org/pdf/2512.03643v1)**

> **作者:** Ivan Yee Lee; Cheng Yang; Taylor Berg-Kirkpatrick
>
> **摘要:** DeepSeek-OCR demonstrates that rendered text can be reconstructed with high fidelity from a small number of vision tokens. This finding has sparked excitement about vision-based context compression for language models. But the evaluation stops at reconstruction; whether these representations help language modeling remains untested. We test two assumptions implicit in the optical-compression narrative: that vision-based compression provides unique advantages for text reconstruction from compressed representations, and that DeepSeek-OCR's reconstruction results are evidence that vision-based compression will be useful for language modeling. Comparing their vision encoder against simple alternatives--parameter-free mean pooling and a learned hierarchical encoder--we find that these simple approaches match or surpass vision for reconstruction at matched compression ratios, and outperform it for language modeling--where vision-based compression fails to beat truncation. The excitement around optical context compression outpaces the evidence. Code and checkpoints are available at https://github.com/ivnle/bad-autoencoding
>
---
#### [new 051] M3DR: Towards Universal Multilingual Multimodal Document Retrieval
- **分类: cs.IR; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出M3DR框架，解决多语言多模态文档检索中英语主导的问题。通过合成多语言数据与对比学习，实现跨语言、跨模态的统一表征，支持多种模型架构与检索范式，在22种语言上实现显著性能提升，推动通用多语言多模态检索发展。**

- **链接: [https://arxiv.org/pdf/2512.03514v1](https://arxiv.org/pdf/2512.03514v1)**

> **作者:** Adithya S Kolavi; Vyoman Jain
>
> **摘要:** Multimodal document retrieval systems have shown strong progress in aligning visual and textual content for semantic search. However, most existing approaches remain heavily English-centric, limiting their effectiveness in multilingual contexts. In this work, we present M3DR (Multilingual Multimodal Document Retrieval), a framework designed to bridge this gap across languages, enabling applicability across diverse linguistic and cultural contexts. M3DR leverages synthetic multilingual document data and generalizes across different vision-language architectures and model sizes, enabling robust cross-lingual and cross-modal alignment. Using contrastive training, our models learn unified representations for text and document images that transfer effectively across languages. We validate this capability on 22 typologically diverse languages, demonstrating consistent performance and adaptability across linguistic and script variations. We further introduce a comprehensive benchmark that captures real-world multilingual scenarios, evaluating models under monolingual, multilingual, and mixed-language settings. M3DR generalizes across both single dense vector and ColBERT-style token-level multi-vector retrieval paradigms. Our models, NetraEmbed and ColNetraEmbed achieve state-of-the-art performance with ~150% relative improvements on cross-lingual retrieval.
>
---
#### [new 052] Stable Signer: Hierarchical Sign Language Generative Model
- **分类: cs.CV; cs.CL; cs.CY**

- **简介: 该论文针对手语生成中多阶段误差累积问题，提出端到端的Stable Signer模型。通过简化任务流程，引入SLUL与SLP-MoE模块，实现文本到手语视频的高效生成，显著提升生成质量与多样性。**

- **链接: [https://arxiv.org/pdf/2512.04048v1](https://arxiv.org/pdf/2512.04048v1)**

> **作者:** Sen Fang; Yalin Feng; Hongbin Zhong; Yanxin Zhang; Dimitris N. Metaxas
>
> **备注:** 12 pages, 7 figures. More Demo at https://stablesigner.github.io
>
> **摘要:** Sign Language Production (SLP) is the process of converting the complex input text into a real video. Most previous works focused on the Text2Gloss, Gloss2Pose, Pose2Vid stages, and some concentrated on Prompt2Gloss and Text2Avatar stages. However, this field has made slow progress due to the inaccuracy of text conversion, pose generation, and the rendering of poses into real human videos in these stages, resulting in gradually accumulating errors. Therefore, in this paper, we streamline the traditional redundant structure, simplify and optimize the task objective, and design a new sign language generative model called Stable Signer. It redefines the SLP task as a hierarchical generation end-to-end task that only includes text understanding (Prompt2Gloss, Text2Gloss) and Pose2Vid, and executes text understanding through our proposed new Sign Language Understanding Linker called SLUL, and generates hand gestures through the named SLP-MoE hand gesture rendering expert block to end-to-end generate high-quality and multi-style sign language videos. SLUL is trained using the newly developed Semantic-Aware Gloss Masking Loss (SAGM Loss). Its performance has improved by 48.6% compared to the current SOTA generation methods.
>
---
#### [new 053] CartoMapQA: A Fundamental Benchmark Dataset Evaluating Vision-Language Models on Cartographic Map Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出CartoMapQA，一个用于评估视觉语言模型地图理解能力的基准数据集。针对LVLM在地图语义理解、地理空间推理和OCR误差方面的不足，构建包含2000+样本的问答数据集，涵盖符号识别、信息提取、尺度理解等任务，旨在推动模型在导航、城市规划等实际应用中的地图理解能力。**

- **链接: [https://arxiv.org/pdf/2512.03558v1](https://arxiv.org/pdf/2512.03558v1)**

> **作者:** Huy Quang Ung; Guillaume Habault; Yasutaka Nishimura; Hao Niu; Roberto Legaspi; Tomoki Oya; Ryoichi Kojima; Masato Taya; Chihiro Ono; Atsunori Minamikawa; Yan Liu
>
> **备注:** Accepted at SIGSPATIAL 2025 (Best paper candidates), 15 pages
>
> **摘要:** The rise of Visual-Language Models (LVLMs) has unlocked new possibilities for seamlessly integrating visual and textual information. However, their ability to interpret cartographic maps remains largely unexplored. In this paper, we introduce CartoMapQA, a benchmark specifically designed to evaluate LVLMs' understanding of cartographic maps through question-answering tasks. The dataset includes over 2000 samples, each composed of a cartographic map, a question (with open-ended or multiple-choice answers), and a ground-truth answer. These tasks span key low-, mid- and high-level map interpretation skills, including symbol recognition, embedded information extraction, scale interpretation, and route-based reasoning. Our evaluation of both open-source and proprietary LVLMs reveals persistent challenges: models frequently struggle with map-specific semantics, exhibit limited geospatial reasoning, and are prone to Optical Character Recognition (OCR)-related errors. By isolating these weaknesses, CartoMapQA offers a valuable tool for guiding future improvements in LVLM architectures. Ultimately, it supports the development of models better equipped for real-world applications that depend on robust and reliable map understanding, such as navigation, geographic search, and urban planning. Our source code and data are openly available to the research community at: https://github.com/ungquanghuy-kddi/CartoMapQA.git
>
---
#### [new 054] Culture Affordance Atlas: Reconciling Object Diversity Through Functional Mapping
- **分类: cs.CY; cs.AI; cs.CL; cs.CV**

- **简介: 该论文针对视觉-语言模型中文化与经济偏见问题，提出功能中心的“文化可及性图谱”框架。通过重构Dollar Street数据集，以46种功能重新标注288类物品，显著缩小高低收入群体间性能差距，提升模型对非西方、低收入语境的适应性，推动更公平的AI发展。**

- **链接: [https://arxiv.org/pdf/2512.03173v1](https://arxiv.org/pdf/2512.03173v1)**

> **作者:** Joan Nwatu; Longju Bai; Oana Ignat; Rada Mihalcea
>
> **摘要:** Culture shapes the objects people use and for what purposes, yet mainstream Vision-Language (VL) datasets frequently exhibit cultural biases, disproportionately favoring higher-income, Western contexts. This imbalance reduces model generalizability and perpetuates performance disparities, especially impacting lower-income and non-Western communities. To address these disparities, we propose a novel function-centric framework that categorizes objects by the functions they fulfill, across diverse cultural and economic contexts. We implement this framework by creating the Culture Affordance Atlas, a re-annotated and culturally grounded restructuring of the Dollar Street dataset spanning 46 functions and 288 objects publicly available at https://lit.eecs.umich.edu/CultureAffordance-Atlas/index.html. Through extensive empirical analyses using the CLIP model, we demonstrate that function-centric labels substantially reduce socioeconomic performance gaps between high- and low-income groups by a median of 6 pp (statistically significant), improving model effectiveness for lower-income contexts. Furthermore, our analyses reveals numerous culturally essential objects that are frequently overlooked in prominent VL datasets. Our contributions offer a scalable pathway toward building inclusive VL datasets and equitable AI systems.
>
---
## 更新

#### [replaced 001] OpenMMReasoner: Pushing the Frontiers for Multimodal Reasoning with an Open and General Recipe
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对多模态推理中数据与训练策略不透明、难以复现的问题，提出OpenMMReasoner，一个开源的两阶段训练框架（SFT+RL）。通过高质量数据集构建与精心设计的训练流程，显著提升模型推理能力，在9个基准上超越基线11.6%，为可复现的多模态推理研究提供新范式。**

- **链接: [https://arxiv.org/pdf/2511.16334v3](https://arxiv.org/pdf/2511.16334v3)**

> **作者:** Kaichen Zhang; Keming Wu; Zuhao Yang; Kairui Hu; Bin Wang; Ziwei Liu; Xingxuan Li; Lidong Bing
>
> **摘要:** Recent advancements in large reasoning models have fueled growing interest in extending such capabilities to multimodal domains. However, despite notable progress in visual reasoning, the lack of transparent and reproducible data curation and training strategies remains a major barrier to scalable research. In this work, we introduce OpenMMReasoner, a fully transparent two-stage recipe for multimodal reasoning spanning supervised fine-tuning (SFT) and reinforcement learning (RL). In the SFT stage, we construct an 874K-sample cold-start dataset with rigorous step-by-step validation, providing a strong foundation for reasoning capabilities. The subsequent RL stage leverages a 74K-sample dataset across diverse domains to further sharpen and stabilize these abilities, resulting in a more robust and efficient learning process. Extensive evaluations demonstrate that our training recipe not only surpasses strong baselines but also highlights the critical role of data quality and training design in shaping multimodal reasoning performance. Notably, our method achieves a 11.6% improvement over the Qwen2.5-VL-7B-Instruct baseline across nine multimodal reasoning benchmarks, establishing a solid empirical foundation for future large-scale multimodal reasoning research. We open-sourced all our codes, pipeline, and data at https://github.com/EvolvingLMMs-Lab/OpenMMReasoner.
>
---
#### [replaced 002] Exact Coset Sampling for Quantum Lattice Algorithms
- **分类: quant-ph; cs.CL; cs.CR**

- **简介: 该论文针对量子格算法中的精确陪集采样问题，提出新步骤Step 9†，解决原算法在存在仿射偏移时的假设失效问题。通过引入假设A5，实现高分辨率与低相位误差的平衡，无需控制线即可完成中心参考相位校正，提升采样精度与效率。**

- **链接: [https://arxiv.org/pdf/2509.12341v5](https://arxiv.org/pdf/2509.12341v5)**

> **作者:** Yifan Zhang
>
> **备注:** Preprint - Work in Progress
>
> **摘要:** In this work, we give a new completion of Chen's windowed-QFT lattice algorithm~\citep{chen2024quantum}. This extra step, called Step~$9^\dagger$, replaces the domain extension stage in Steps~8--9. The published Step~9 calls an amplitude periodicity lemma, yet its hypotheses break in the presence of affine offsets $\boldsymbol{v}^*$. Our analysis finds a basic conflict between two design constraints. The lattice problem asks for high spectral resolution, so the method prefers wide time windows. The quadratic phase error of the state prefers narrow time windows. Assumption~A5 packages the spectral concentration and near-uniformity properties that we require from the front end. Under~A5, a direct $\mathbb{Z}_M^n$ Fourier transform of the chirp-corrected coordinate state produces samples $\boldsymbol{u}$ that satisfy $\langle \boldsymbol{b}, \boldsymbol{u} \rangle \equiv 0 \pmod{Q}$ with probability $1-\mathrm{negl}(n)$ and are nearly uniform on the dual hyperplane $\{\boldsymbol{u} : \langle \boldsymbol{b}, \boldsymbol{u} \rangle \equiv 0 \pmod{Q}\}$. The new procedure does not require internal access to control wires. It uses the normalization $b_1=-1$ to apply a center-referenced phase correction directly on the first coordinate register. The scaling parameter $D$ ensures that this physical operation can be implemented by arithmetic on $X_1$ alone and does not read the hidden loop index. For Chen's complex-Gaussian Karst-wave window, we isolate a parameter regime, formalized in Assumption~A5, in which a polynomial retuning of the parameters gives a one-dimensional envelope for the loop index with width $σ_J \asymp Q\log n$.
>
---
#### [replaced 003] Let Them Down Easy! Contextual Effects of LLM Guardrails on User Perceptions and Preferences
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文研究大模型拒答策略对用户感知的影响，旨在解决安全与用户体验的矛盾。通过实验发现，部分合规（提供一般信息）优于完全拒绝，显著提升用户满意度。研究还揭示当前模型与奖励机制均未充分采用此策略，提出应聚焦于设计有温度的拒答而非意图识别。**

- **链接: [https://arxiv.org/pdf/2506.00195v2](https://arxiv.org/pdf/2506.00195v2)**

> **作者:** Mingqian Zheng; Wenjia Hu; Patrick Zhao; Motahhare Eslami; Jena D. Hwang; Faeze Brahman; Carolyn Rose; Maarten Sap
>
> **备注:** Accepted to Findings of EMNLP 2025
>
> **摘要:** Current LLMs are trained to refuse potentially harmful input queries regardless of whether users actually had harmful intents, causing a tradeoff between safety and user experience. Through a study of 480 participants evaluating 3,840 query-response pairs, we examine how different refusal strategies affect user perceptions across varying motivations. Our findings reveal that response strategy largely shapes user experience, while actual user motivation has negligible impact. Partial compliance -- providing general information without actionable details -- emerges as the optimal strategy, reducing negative user perceptions by over 50% to flat-out refusals. Complementing this, we analyze response patterns of 9 state-of-the-art LLMs and evaluate how 6 reward models score different refusal strategies, demonstrating that models rarely deploy partial compliance naturally and reward models currently undervalue it. This work demonstrates that effective guardrails require focusing on crafting thoughtful refusals rather than detecting intent, offering a path toward AI safety mechanisms that ensure both safety and sustained user engagement.
>
---
#### [replaced 004] Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究语言模型在后训练中因任务更新导致的灾难性遗忘问题。通过对比监督微调（SFT）与强化学习（RL），发现RL因使用在线策略数据，能更好保留原有知识。研究揭示了在线策略数据在缓解遗忘中的关键作用，并提出高效获取近似在线策略数据的实践路径。**

- **链接: [https://arxiv.org/pdf/2510.18874v2](https://arxiv.org/pdf/2510.18874v2)**

> **作者:** Howard Chen; Noam Razin; Karthik Narasimhan; Danqi Chen
>
> **摘要:** Adapting language models (LMs) to new tasks via post-training carries the risk of degrading existing capabilities -- a phenomenon classically known as catastrophic forgetting. In this paper, toward identifying guidelines for mitigating this phenomenon, we systematically compare the forgetting patterns of two widely adopted post-training methods: supervised fine-tuning (SFT) and reinforcement learning (RL). Our experiments reveal a consistent trend across LM families (Llama, Qwen) and tasks (instruction following, general knowledge, and arithmetic reasoning): RL leads to less forgetting than SFT while achieving comparable or higher target task performance. To investigate the cause for this difference, we consider a simplified setting in which the LM is modeled as a mixture of two distributions, one corresponding to prior knowledge and the other to the target task. We identify that the mode-seeking nature of RL, which stems from its use of on-policy data, enables keeping prior knowledge intact when learning the target task. We then verify this insight by demonstrating that the use on-policy data underlies the robustness of RL to forgetting in practical settings, as opposed to other algorithmic choices such as the KL regularization or advantage estimation. Lastly, as a practical implication, our results highlight the potential of mitigating forgetting using approximately on-policy data, which can be substantially more efficient to obtain than fully on-policy data.
>
---
#### [replaced 005] Batch Prompting Suppresses Overthinking Reasoning Under Constraint: How Batch Prompting Suppresses Overthinking in Reasoning Models
- **分类: cs.CL**

- **简介: 该论文研究批量提示（batch prompting）在大语言模型推理中的作用，旨在解决多步推理中“过度思考”导致的效率低下问题。通过13个基准测试，发现批量提示能显著降低推理成本3-5倍，抑制冗余修正，提升准确性，并产生集体泛化效应，表明其不仅是吞吐优化，更是有效的推理正则化手段。**

- **链接: [https://arxiv.org/pdf/2511.04108v2](https://arxiv.org/pdf/2511.04108v2)**

> **作者:** Wenmo Qiu; Saurabh Srivastava
>
> **备注:** The paper is incomplete with some errors in qualitative study
>
> **摘要:** Recent work has explored batch prompting as a strategy to amortize inference cost in large language models (LLMs). In this paper, we show that batching offers an additional, underappreciated benefit: it regularizes model behavior during multi-step reasoning for Large Reasoning Models (LRMs). We conduct a comprehensive study across 13 diverse benchmarks and observe that batching improves accuracy while substantially reducing reasoning token usage, often by 3x-5x. Through detailed behavioral analysis, we find that batching suppresses overthinking, reduces hedging language (e.g., repetitive self-corrections), and encourages more decisive answers. Surprisingly, we also observe emergent collective effects in batched inference: models often generalize patterns from earlier examples to solve harder ones in the same batch. These findings position batching not just as a throughput optimization, but as a powerful inference-time regularizer for more efficient and reliable LLM reasoning.
>
---
#### [replaced 006] SpeechRole: A Large-Scale Dataset and Benchmark for Evaluating Speech Role-Playing Agents
- **分类: cs.CL**

- **简介: 该论文针对语音角色扮演（SRPA）缺乏系统评估的问题，构建了SpeechRole-Data大规模数据集，涵盖98种角色的11.2万条语音对话。提出SpeechRole-Eval多维度评估基准，评估角色一致性、语音表现力等。释放数据与代码，推动语音驱动角色扮演研究发展。**

- **链接: [https://arxiv.org/pdf/2508.02013v5](https://arxiv.org/pdf/2508.02013v5)**

> **作者:** Changhao Jiang; Jiajun Sun; Yifei Cao; Jiabao Zhuang; Hui Li; Baoyu Fan; Tao Ji; Tao Gui; Qi Zhang
>
> **备注:** This work is withdrawn as all authors are not in agreement on the work
>
> **摘要:** Recently, role-playing agents have emerged as a promising paradigm for achieving personalized interaction and emotional resonance. Existing research primarily focuses on the textual modality, neglecting the critical dimension of speech in realistic interactive scenarios. In particular, there is a lack of systematic evaluation for Speech Role-Playing Agents (SRPAs). To address this gap, we construct SpeechRole-Data, a large-scale, high-quality dataset that comprises 98 diverse roles and 112k speech-based single-turn and multi-turn conversations. Each role demonstrates distinct vocal characteristics, including timbre and prosody, thereby enabling more sophisticated speech role-playing. Furthermore, we propose SpeechRole-Eval, a multidimensional evaluation benchmark that systematically assesses SRPAs performance in key aspects such as fundamental interaction ability, speech expressiveness, and role-playing fidelity. Experimental results reveal the advantages and challenges of both cascaded and end-to-end speech role-playing agents in maintaining vocal style consistency and role coherence. We release all data, code, and baseline models to provide a solid foundation for speech-driven multimodal role-playing research and to foster further developments in this field.
>
---
#### [replaced 007] How to Train Long-Context Language Models (Effectively)
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究长上下文语言模型的训练方法，旨在提升模型对长文本的理解与生成能力。针对长上下文建模效果评估不准确的问题，提出基于多任务评估的可靠协议，并通过实验优化数据混合、序列长度等设计。最终构建的ProLong-8B模型在128K上下文长度上达到领先性能，支持最长512K输入。**

- **链接: [https://arxiv.org/pdf/2410.02660v4](https://arxiv.org/pdf/2410.02660v4)**

> **作者:** Tianyu Gao; Alexander Wettig; Howard Yen; Danqi Chen
>
> **备注:** Accepted to ACL 2025. Our code, data, and models are available at https://github.com/princeton-nlp/ProLong
>
> **摘要:** We study continued training and supervised fine-tuning (SFT) of a language model (LM) to make effective use of long-context information. We first establish a reliable evaluation protocol to guide model development -- instead of perplexity or simple needle-in-a-haystack (NIAH) tests, we use a broad set of long-context downstream tasks, and we evaluate models after SFT as this better reveals long-context abilities. Supported by our robust evaluations, we run thorough experiments to decide the data mix for continued pre-training, the instruction tuning dataset, and many other design choices such as position extrapolation. We find that (1) code repositories and books are excellent sources of long data, but it is crucial to combine them with high-quality short-context data; (2) training with a sequence length beyond the evaluation length boosts long-context performance; (3) for SFT, using only short instruction datasets yields strong performance on long-context tasks. Our final model, ProLong-8B, which is initialized from Llama-3 and trained on 40B tokens, demonstrates state-of-the-art long-context performance among similarly sized models at a length of 128K. ProLong outperforms Llama-3.1-8B-Instruct on the majority of long-context tasks despite using only 5% as many tokens during long-context training. Additionally, ProLong can effectively process up to 512K tokens, one of the longest context windows of publicly available LMs.
>
---
#### [replaced 008] Uncertainty Quantification for LLMs through Minimum Bayes Risk: Bridging Confidence and Consistency
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLM）的不确定性量化（UQ）任务，旨在解决现有方法在模型置信度与输出一致性结合上的不足。通过关联最小贝叶斯风险，提出一种新方法融合两者，显著提升UQ性能，在问答、摘要、翻译等任务中优于现有技术。**

- **链接: [https://arxiv.org/pdf/2502.04964v5](https://arxiv.org/pdf/2502.04964v5)**

> **作者:** Roman Vashurin; Maiya Goloburda; Albina Ilina; Aleksandr Rubashevskii; Preslav Nakov; Artem Shelmanov; Maxim Panov
>
> **摘要:** Uncertainty quantification (UQ) methods for Large Language Models (LLMs) encompass a variety of approaches, with two major types being particularly prominent: information-based, which focus on model confidence expressed as token probabilities, and consistency-based, which assess the semantic relationship between multiple outputs generated using repeated sampling. Several recent methods have combined these two approaches to boost UQ performance. However, they sometimes fail to outperform much simpler baseline methods. Our work discusses the fundamental approach to constructing uncertainty measures that directly links uncertainty with the minimum Bayes risks achieved by LLM decoding. Building on these findings, we propose a novel approach to integrating model confidence with output consistency, resulting in a family of efficient and robust UQ methods. Our investigation reveals distinctive characteristics of LLMs as probabilistic models, which help to explain why these UQ methods underperform in certain tasks. Based on these findings, we propose a new way of synthesizing model confidence and output consistency, leading to a family of efficient and robust UQ methods. We evaluate our approach across various tasks such as question answering, abstractive summarization, and machine translation, demonstrating sizable improvements over state-of-the-art UQ approaches.
>
---
#### [replaced 009] Causal LLM Routing: End-to-End Regret Minimization from Observational Data
- **分类: cs.AI; cs.CL; cs.LG; stat.ML**

- **简介: 该论文研究大模型路由任务，旨在基于观测数据最小化决策后悔值，解决传统方法依赖全反馈数据及误差累积问题。提出因果端到端框架，设计两类可优化的代理目标，并引入区间条件架构处理异构成本偏好，实现在公开数据集上超越现有基线的性能。**

- **链接: [https://arxiv.org/pdf/2505.16037v2](https://arxiv.org/pdf/2505.16037v2)**

> **作者:** Asterios Tsiourvas; Wei Sun; Georgia Perakis
>
> **摘要:** LLM routing aims to select the most appropriate model for each query, balancing competing performance metrics such as accuracy and cost across a pool of language models. Prior approaches typically adopt a decoupled strategy, where the metrics are first predicted and the model is then selected based on these estimates. This setup is prone to compounding errors and often relies on full-feedback data, where each query is evaluated by all candidate models, which is costly to obtain and maintain in practice. In contrast, we learn from observational data, which records only the outcome of the model actually deployed. We propose a causal end-to-end framework that learns routing policies by minimizing decision-making regret from observational data. To enable efficient optimization, we introduce two theoretically grounded surrogate objectives: a classification-based upper bound, and a softmax-weighted regret approximation shown to recover the optimal policy at convergence. We further extend our framework to handle heterogeneous cost preferences via an interval-conditioned architecture. Experiments on public benchmarks show that our method outperforms existing baselines, achieving state-of-the-art performance across different embedding models.
>
---
#### [replaced 010] Reveal-Bangla: A Dataset for Cross-Lingual Multi-Step Reasoning Evaluation
- **分类: cs.CL**

- **简介: 该论文针对低资源语言推理评估不足的问题，构建了首个孟加拉语多步推理数据集Reveal-Bangla。通过人工翻译英文Reveal数据集，涵盖二元与非二元问题类型，对比英、孟双语小模型在跨语言推理中的表现，揭示模型在利用孟加拉语推理步骤时的局限性，推动多语言复杂推理研究。**

- **链接: [https://arxiv.org/pdf/2508.08933v3](https://arxiv.org/pdf/2508.08933v3)**

> **作者:** Khondoker Ittehadul Islam; Gabriele Sarti
>
> **备注:** Accepted at BLP workshop @ IJCNLP-AACL 2025
>
> **摘要:** Language models have demonstrated remarkable performance on complex multi-step reasoning tasks. However, their evaluation has been predominantly confined to high-resource languages such as English. In this paper, we introduce a manually translated Bangla multi-step reasoning dataset derived from the English Reveal dataset, featuring both binary and non-binary question types. We conduct a controlled evaluation of English-centric and Bangla-centric multilingual small language models on the original dataset and our translated version to compare their ability to exploit relevant reasoning steps to produce correct answers. Our results show that, in comparable settings, reasoning context is beneficial for more challenging non-binary questions, but models struggle to employ relevant Bangla reasoning steps effectively. We conclude by exploring how reasoning steps contribute to models' predictions, highlighting different trends across models and languages.
>
---
#### [replaced 011] IW-Bench: Evaluating Large Multimodal Models for Converting Image-to-Web
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对大模型图像到网页转换任务，提出IW-Bench基准。解决现有评估方法忽略隐式元素与布局信息的问题，创新性引入元素准确率与布局准确率，并设计五跳多模态思维链提示，全面评估模型生成网页的完整性与布局准确性。**

- **链接: [https://arxiv.org/pdf/2409.18980v2](https://arxiv.org/pdf/2409.18980v2)**

> **作者:** Hongcheng Guo; Wei Zhang; Junhao Chen; Yaonan Gu; Jian Yang; Junjia Du; Shaosheng Cao; Binyuan Hui; Tianyu Liu; Jianxin Ma; Chang Zhou; Zhoujun Li
>
> **摘要:** Recently advancements in large multimodal models have led to significant strides in image comprehension capabilities. Despite these advancements, there is a lack of the robust benchmark specifically for assessing the Image-to-Web conversion proficiency of these large models. Primarily, it is essential to ensure the integrity of the web elements generated. These elements comprise visible and invisible categories. Previous evaluation methods (e.g.,BLEU) are notably susceptible to significant alterations due to the presence of invisible elements in Web. Furthermore, it is crucial to measure the layout information of web pages, referring to the positional relationships between elements, which is overlooked by previous work. To address challenges, we have curated and aligned a benchmark of images and corresponding web codes (IW-BENCH). Specifically, we propose the Element Accuracy, which tests the completeness of the elements by parsing the Document Object Model (DOM) tree. Layout Accuracy is also proposed to analyze the positional relationships of elements by converting DOM tree into a common subsequence. Besides, we design a five-hop multimodal Chain-of-Thought Prompting for better performance, which contains five hop: 1) SoM prompt injection. 2) Inferring Elements. 3) Inferring Layout. 4) Inferring Web code. 5) Reflection. Our benchmark comprises 1200 pairs of images and web codes with varying levels of difficulty. We have conducted extensive experiments on existing large multimodal models, offering insights into their performance and areas for improvement in image-to-web domain.
>
---
#### [replaced 012] Characterizing the Expressivity of Fixed-Precision Transformer Language Models
- **分类: cs.CL**

- **简介: 该论文研究固定精度Transformer语言模型的表达能力，旨在理解其理论极限。通过理想化模型分析，发现其表达力等价于仅含过去时态算子的线性时序逻辑片段，并与形式语言理论建立联系。实验验证了理论预测：模型在可表达语言上能可靠泛化，超出则失败。**

- **链接: [https://arxiv.org/pdf/2505.23623v2](https://arxiv.org/pdf/2505.23623v2)**

> **作者:** Jiaoda Li; Ryan Cotterell
>
> **备注:** NeurIPS 2025 (Spotlight)
>
> **摘要:** Transformer-based language models (LMs) have achieved widespread empirical success, but their theoretical expressive power remains only partially understood. In this work, we analyze a restricted idealization of fixed-precision transformers with strict future masking, soft attention, and no positional encodings. We establish that this class of models is exactly as expressive as a specific fragment of linear temporal logic that contains only a single temporal operator: the past operator. We further connect this fragment to established classes in formal language theory, automata theory, and algebra, yielding a unified framework for understanding transformer expressivity under this idealization. Finally, we present empirical results that align closely with our theory: transformers trained on languages within their characterized expressive capacity generalize reliably across sequence lengths, while they consistently fail to generalize on languages beyond it.
>
---
#### [replaced 013] Which Type of Students can LLMs Act? Investigating Authentic Simulation with Graph-based Human-AI Collaborative System
- **分类: cs.CY; cs.CL**

- **简介: 该论文聚焦于教育人工智能中的学生模拟任务，旨在解决真实学生行为数据稀缺与仿真真实性评估困难的问题。通过构建三阶段人机协作管道，结合自动化评分与图传播机制，生成高质量、高保真的学生代理，并分析不同学生特征的模拟效果，为个性化学习与评估提供支持。**

- **链接: [https://arxiv.org/pdf/2502.11678v3](https://arxiv.org/pdf/2502.11678v3)**

> **作者:** Haoxuan Li; Jifan Yu; Xin Cong; Yang Dang; Daniel Zhang-li; Lu Mi; Yisi Zhan; Huiqin Liu; Zhiyuan Liu
>
> **备注:** This work has been submitted to AI Open for possible publication
>
> **摘要:** While rapid advances in large language models (LLMs) are reshaping data-driven intelligent education, accurately simulating students remains an important but challenging bottleneck for scalable educational data collection, evaluation, and intervention design. However, current works are limited by scarce real interaction data, costly expert evaluation for realism, and a lack of large-scale, systematic analyses of LLMs ability in simulating students. We address this gap by presenting a three-stage LLM-human collaborative pipeline to automatically generate and filter high-quality student agents. We leverage a two-round automated scoring validated by human experts and deploy a score propagation module to obtain more consistent scores across the student similarity graph. Experiments show that combining automated scoring, expert calibration, and graph-based propagation yields simulated student that more closely track authentication by human judgments. We then analyze which profiles and behaviors are simulated more faithfully, supporting subsequent studies on personalized learning and educational assessment.
>
---
#### [replaced 014] MemOS: A Memory OS for AI System
- **分类: cs.CL**

- **简介: 该论文提出MemOS，一种面向大语言模型的记忆操作系统，解决LLM缺乏有效记忆管理导致的长期推理、持续个性化与知识一致性难题。通过引入统一的内存资源管理框架，整合文本、激活与参数级记忆，实现高效存储与跨类型迁移，支持持续学习与个性化建模。**

- **链接: [https://arxiv.org/pdf/2507.03724v4](https://arxiv.org/pdf/2507.03724v4)**

> **作者:** Zhiyu Li; Chenyang Xi; Chunyu Li; Ding Chen; Boyu Chen; Shichao Song; Simin Niu; Hanyu Wang; Jiawei Yang; Chen Tang; Qingchen Yu; Jihao Zhao; Yezhaohui Wang; Peng Liu; Zehao Lin; Pengyuan Wang; Jiahao Huo; Tianyi Chen; Kai Chen; Kehang Li; Zhen Tao; Huayi Lai; Hao Wu; Bo Tang; Zhengren Wang; Zhaoxin Fan; Ningyu Zhang; Linfeng Zhang; Junchi Yan; Mingchuan Yang; Tong Xu; Wei Xu; Huajun Chen; Haofen Wang; Hongkang Yang; Wentao Zhang; Zhi-Qin John Xu; Siheng Chen; Feiyu Xiong
>
> **备注:** 36 pages, 10 figures, 5 tables
>
> **摘要:** Large Language Models (LLMs) have become an essential infrastructure for Artificial General Intelligence (AGI), yet their lack of well-defined memory management systems hinders the development of long-context reasoning, continual personalization, and knowledge consistency.Existing models mainly rely on static parameters and short-lived contextual states, limiting their ability to track user preferences or update knowledge over extended periods.While Retrieval-Augmented Generation (RAG) introduces external knowledge in plain text, it remains a stateless workaround without lifecycle control or integration with persistent representations.Recent work has modeled the training and inference cost of LLMs from a memory hierarchy perspective, showing that introducing an explicit memory layer between parameter memory and external retrieval can substantially reduce these costs by externalizing specific knowledge. Beyond computational efficiency, LLMs face broader challenges arising from how information is distributed over time and context, requiring systems capable of managing heterogeneous knowledge spanning different temporal scales and sources. To address this challenge, we propose MemOS, a memory operating system that treats memory as a manageable system resource. It unifies the representation, scheduling, and evolution of plaintext, activation-based, and parameter-level memories, enabling cost-efficient storage and retrieval. As the basic unit, a MemCube encapsulates both memory content and metadata such as provenance and versioning. MemCubes can be composed, migrated, and fused over time, enabling flexible transitions between memory types and bridging retrieval with parameter-based learning. MemOS establishes a memory-centric system framework that brings controllability, plasticity, and evolvability to LLMs, laying the foundation for continual learning and personalized modeling.
>
---
#### [replaced 015] Stabilizing Reinforcement Learning with LLMs: Formulation and Practices
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型（LLM）强化学习中的训练稳定性问题。针对策略梯度方法中序列级奖励优化困难，提出通过一阶近似将目标转为词元级代理目标，并揭示其有效性依赖于降低训练-推理差异与策略僵化。实验证明重要性采样、裁剪和路由重放等技术对稳定训练至关重要，尤其在大规模MoE模型上验证了高效且稳定的训练方案。**

- **链接: [https://arxiv.org/pdf/2512.01374v3](https://arxiv.org/pdf/2512.01374v3)**

> **作者:** Chujie Zheng; Kai Dang; Bowen Yu; Mingze Li; Huiqiang Jiang; Junrong Lin; Yuqiong Liu; Hao Lin; Chencan Wu; Feng Hu; An Yang; Jingren Zhou; Junyang Lin
>
> **摘要:** This paper proposes a novel formulation for reinforcement learning (RL) with large language models, explaining why and under what conditions the true sequence-level reward can be optimized via a surrogate token-level objective in policy gradient methods such as REINFORCE. Specifically, through a first-order approximation, we show that this surrogate becomes increasingly valid only when both the training-inference discrepancy and policy staleness are minimized. This insight provides a principled explanation for the crucial role of several widely adopted techniques in stabilizing RL training, including importance sampling correction, clipping, and particularly Routing Replay for Mixture-of-Experts (MoE) models. Through extensive experiments with a 30B MoE model totaling hundreds of thousands of GPU hours, we show that for on-policy training, the basic policy gradient algorithm with importance sampling correction achieves the highest training stability. When off-policy updates are introduced to accelerate convergence, combining clipping and Routing Replay becomes essential to mitigate the instability caused by policy staleness. Notably, once training is stabilized, prolonged optimization consistently yields comparable final performance regardless of cold-start initialization. We hope that the shared insights and the developed recipes for stable RL training will facilitate future research.
>
---
#### [replaced 016] SETS: Leveraging Self-Verification and Self-Correction for Improved Test-Time Scaling
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对大语言模型在复杂推理任务中测试时计算扩展效率低的问题，提出SETS方法。通过融合并行采样与串行自验证、自修正，充分利用LLM的自我改进能力，实现无需训练的高效测试时扩展，在规划、推理、数学和编码任务上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2501.19306v5](https://arxiv.org/pdf/2501.19306v5)**

> **作者:** Jiefeng Chen; Jie Ren; Xinyun Chen; Chengrun Yang; Ruoxi Sun; Jinsung Yoon; Sercan Ö Arık
>
> **备注:** Published in Transactions on Machine Learning Research (11/2025)
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have created new opportunities to enhance performance on complex reasoning tasks by leveraging test-time computation. However, existing scaling methods have key limitations: parallel methods like repeated sampling are often inefficient and quickly saturate, while sequential methods like SELF-REFINE struggle to improve after a few rounds. Although combining these approaches shows promise, current methods require fine-tuned reward and revision models. This paper proposes Self-Enhanced Test-Time Scaling (SETS), a simple yet effective approach that overcomes these limitations by strategically combining parallel and sequential techniques and fully leveraging LLMs' self-improvement abilities. SETS exploits the inherent self-verification and self-correction capabilities of LLMs, unifying sampling, verification, and correction within a single framework. This facilitates efficient and scalable test-time computation for enhanced performance on complex tasks without any model training. Our comprehensive experimental results on challenging benchmarks spanning planning, reasoning, math, and coding demonstrate that SETS achieves significant performance improvements and more advantageous test-time scaling behavior than the alternatives.
>
---
#### [replaced 017] Comba: Improving Bilinear RNNs with Closed-loop Control
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对序列建模中高效内存管理问题，提出基于闭环控制的双线性RNN模型Comba。通过引入标量加低秩状态转移及反馈机制，提升模型性能与计算效率，在语言与视觉任务中均表现优异。**

- **链接: [https://arxiv.org/pdf/2506.02475v5](https://arxiv.org/pdf/2506.02475v5)**

> **作者:** Jiaxi Hu; Yongqi Pan; Jusen Du; Disen Lan; Xiaqiang Tang; Qingsong Wen; Yuxuan Liang; Weigao Sun
>
> **摘要:** Recent efficient sequence modeling methods such as Gated DeltaNet, TTT, and RWKV-7 have achieved performance improvements by supervising the recurrent memory management through Delta learning rule. Unlike previous state-space models (e.g., Mamba) and gated linear attentions (e.g., GLA), these models introduce interactions between the recurrent state and the key vector, structurally resembling bilinear systems. In this paper, we first introduce the concept of Bilinear RNNs with a comprehensive analysis on the advantages and limitations of these models. Then, based on closed-loop control theory, we propose a novel Bilinear RNN variant named Comba, which adopts a scalar-plus-low-rank state transition, with both state feedback and output feedback corrections. We also implement a hardware-efficient chunk-wise parallel kernel in Triton and train models with 340M/1.3B parameters on large-scale corpus. Comba demonstrates superior performance and computation efficiency in both language and vision modeling.
>
---
#### [replaced 018] A Group Fairness Lens for Large Language Models
- **分类: cs.CL**

- **简介: 该论文聚焦大语言模型（LLM）的公平性评估与偏见缓解，提出从群体公平性视角出发，构建多维度数据集GFAIR，并设计新任务“陈述组织”以揭示复杂偏见。通过提出GF-THINK方法，有效减轻模型偏见，提升公平性。**

- **链接: [https://arxiv.org/pdf/2312.15478v2](https://arxiv.org/pdf/2312.15478v2)**

> **作者:** Guanqun Bi; Yuqiang Xie; Lei Shen; Yanan Cao
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** The need to assess LLMs for bias and fairness is critical, with current evaluations often being narrow, missing a broad categorical view. In this paper, we propose evaluating the bias and fairness of LLMs from a group fairness lens using a novel hierarchical schema characterizing diverse social groups. Specifically, we construct a dataset, GFAIR, encapsulating target-attribute combinations across multiple dimensions. Moreover, we introduce statement organization, a new open-ended text generation task, to uncover complex biases in LLMs. Extensive evaluations of popular LLMs reveal inherent safety concerns. To mitigate the biases of LLMs from a group fairness perspective, we pioneer a novel chainof-thought method GF-THINK to mitigate biases of LLMs from a group fairness perspective. Experimental results demonstrate its efficacy in mitigating bias and achieving fairness in LLMs. Our dataset and codes are available at https://github.com/surika/Group-Fairness-LLMs.
>
---
#### [replaced 019] MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文针对多语言、多条件交织的语义检索任务，提出首个相关数据集MERIT与新框架Coral。针对现有模型忽视查询中具体条件的问题，通过嵌入重建与对比学习，提升细粒度条件保留与全局语义理解能力，显著提升检索性能。**

- **链接: [https://arxiv.org/pdf/2506.03144v3](https://arxiv.org/pdf/2506.03144v3)**

> **作者:** Wei Chow; Yuan Gao; Linfeng Li; Xian Wang; Qi Xu; Hang Song; Lingdong Kong; Ran Zhou; Yi Zeng; Yidong Cai; Botian Jiang; Shilin Xu; Jiajun Zhang; Minghui Qiu; Xiangtai Li; Tianshu Yang; Siliang Tang; Juncheng Li
>
> **备注:** NeurIPS 2025; Project Page, Code, and Dataset at: https://merit-2025.github.io/
>
> **摘要:** Semantic retrieval is crucial for modern applications yet remains underexplored in current research. Existing datasets are limited to single languages, single images, or singular retrieval conditions, often failing to fully exploit the expressive capacity of visual information as evidenced by maintained performance when images are replaced with captions. However, practical retrieval scenarios frequently involve interleaved multi-condition queries with multiple images. Hence, this paper introduces MERIT, the first multilingual dataset for interleaved multi-condition semantic retrieval, comprising 320,000 queries with 135,000 products in 5 languages, covering 7 distinct product categories. Extensive experiments on MERIT identify existing models's limitation: focusing solely on global semantic information while neglecting specific conditional elements in queries. Consequently, we propose Coral, a novel fine-tuning framework that adapts pre-trained MLLMs by integrating embedding reconstruction to preserve fine-grained conditional elements and contrastive learning to extract comprehensive global semantics. Experiments demonstrate that Coral achieves a 45.9% performance improvement over conventional approaches on MERIT, with strong generalization capabilities validated across 8 established retrieval benchmarks. Collectively, our contributions - a novel dataset, identification of critical limitations in existing approaches, and an innovative fine-tuning framework - establish a foundation for future research in interleaved multi-condition semantic retrieval.
>
---
#### [replaced 020] Astra: A Multi-Agent System for GPU Kernel Performance Optimization
- **分类: cs.DC; cs.AI; cs.CL; cs.LG; cs.SE**

- **简介: 该论文提出Astra，一个基于大模型的多智能体系统，用于GPU核函数性能优化。针对手工调优耗时、传统编译器需大量工程投入的问题，Astra从SGLang框架中提取的CUDA代码出发，通过智能体协作实现自动代码生成、测试与优化，显著提升性能，平均加速1.32倍，验证了多智能体大模型在高性能计算中的潜力。**

- **链接: [https://arxiv.org/pdf/2509.07506v2](https://arxiv.org/pdf/2509.07506v2)**

> **作者:** Anjiang Wei; Tianran Sun; Yogesh Seenichamy; Hang Song; Anne Ouyang; Azalia Mirhoseini; Ke Wang; Alex Aiken
>
> **摘要:** GPU kernel optimization has long been a central challenge at the intersection of high-performance computing and machine learning. Efficient kernels are crucial for accelerating large language model (LLM) training and serving, yet attaining high performance typically requires extensive manual tuning. Compiler-based systems reduce some of this burden, but still demand substantial manual design and engineering effort. Recently, researchers have explored using LLMs for GPU kernel generation, though prior work has largely focused on translating high-level PyTorch modules into CUDA code. In this work, we introduce Astra, the first LLM-based multi-agent system for GPU kernel optimization. Unlike previous approaches, Astra starts from existing CUDA implementations extracted from SGLang, a widely deployed framework for serving LLMs, rather than treating PyTorch modules as the specification. Within Astra, specialized LLM agents collaborate through iterative code generation, testing, profiling, and planning to produce kernels that are both correct and high-performance. On kernels from SGLang, Astra achieves an average speedup of 1.32x using zero-shot prompting with OpenAI o4-mini. A detailed case study further demonstrates that LLMs can autonomously apply loop transformations, optimize memory access patterns, exploit CUDA intrinsics, and leverage fast math operations to yield substantial performance gains. Our work highlights multi-agent LLM systems as a promising new paradigm for GPU kernel optimization. Our code is publicly available at https://github.com/Anjiang-Wei/Astra.
>
---
#### [replaced 021] ZIP-RC: Optimizing Test-Time Compute via Zero-Overhead Joint Reward-Cost Prediction
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大模型推理缺乏实时成本与收益预判的问题，提出ZIP-RC方法。通过零开销复用前向传播中的logits，实现对奖励与计算成本的联合预测，动态优化采样策略，提升推理效率与准确性，在数学推理任务中显著降低开销并改善性能。**

- **链接: [https://arxiv.org/pdf/2512.01457v2](https://arxiv.org/pdf/2512.01457v2)**

> **作者:** Rohin Manvi; Joey Hong; Tim Seyde; Maxime Labonne; Mathias Lechner; Sergey Levine
>
> **备注:** Code coming soon
>
> **摘要:** Large language models excel at reasoning but lack key aspects of introspection, including anticipating their own success and the computation required to achieve it. Humans use real-time introspection to decide how much effort to invest, when to make multiple attempts, when to stop, and when to signal success or failure. Without this, LLMs struggle to make intelligent meta-cognition decisions. Test-time scaling methods like Best-of-N drive up cost and latency by using a fixed budget of samples regardless of the marginal benefit of each one at any point in generation, and the absence of confidence signals can mislead people, prevent appropriate escalation to better tools, and undermine trustworthiness. Learned verifiers or reward models can provide confidence estimates, but do not enable adaptive inference and add substantial cost by requiring extra models or forward passes. We present ZIP-RC, an adaptive inference method that equips models with zero-overhead inference-time predictions of reward and cost. At every token, ZIP-RC reuses reserved or unused logits in the same forward pass as next-token prediction to output a joint distribution over final reward and remaining length -- no extra models, architecture change, or inference overhead. This full joint distribution is used to compute a sampling utility which is the linear combination of the expected maximum reward, total compute, and latency of set of samples if generated to completion. During inference, we maximize this utility with meta-actions that determine which prefix of tokens to continue or initiate sampling from. On mixed-difficulty mathematical benchmarks, ZIP-RC improves accuracy by up to 12% over majority voting at equal or lower average cost, and traces smooth Pareto frontiers between quality, compute, and latency. By providing real-time reward-cost introspection, ZIP-RC enables adaptive, efficient reasoning.
>
---
#### [replaced 022] Bigram Subnetworks: Mapping to Next Tokens in Transformer Language Models
- **分类: cs.CL**

- **简介: 该论文研究语言模型中预测下一个词的最小有效子网络。针对Transformer模型中从当前词到下一词预测的转换机制，识别出仅依赖当前词（即“二元组”）的最小参数子网络。发现这些子网络虽不足0.2%参数，却对模型性能至关重要，且集中于第一MLP层，能重构关键激活变换模式。该工作为理解语言模型内部电路提供了基础。**

- **链接: [https://arxiv.org/pdf/2504.15471v3](https://arxiv.org/pdf/2504.15471v3)**

> **作者:** Tyler A. Chang; Benjamin K. Bergen
>
> **备注:** NeurIPS 2025
>
> **摘要:** In Transformer language models, activation vectors transform from current token embeddings to next token predictions as they pass through the model. To isolate a minimal form of this transformation, we identify language model subnetworks that make bigram predictions, naive next token predictions based only on the current token. We find that bigram subnetworks can be found in fully trained language models up to 1B parameters, and these subnetworks are critical for model performance even when they consist of less than 0.2% of model parameters. Bigram subnetworks are concentrated in the first Transformer MLP layer, and they overlap significantly with subnetworks trained to optimally prune a given model. Mechanistically, the bigram subnetworks often recreate a pattern from the full models where the first layer induces a sharp change that aligns activations with next token predictions rather than current token representations. Our results demonstrate that bigram subnetworks comprise a minimal subset of parameters that are both necessary and sufficient for basic next token predictions in language models, and they help drive the transformation from current to next token activations in the residual stream. These subnetworks can lay a foundation for studying more complex language model circuits by building up from a minimal circuit.
>
---
#### [replaced 023] AutoEnv: Automated Environments for Measuring Cross-Environment Agent Learning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出AutoEnv框架，用于生成可控制的异构环境，解决跨环境学习缺乏标准评测的问题。构建了包含36个环境的AutoEnv-36数据集，设计八种学习方法评估，发现固定方法难以泛化，环境自适应选择更优但收益递减，揭示了跨环境学习的挑战与局限。**

- **链接: [https://arxiv.org/pdf/2511.19304v2](https://arxiv.org/pdf/2511.19304v2)**

> **作者:** Jiayi Zhang; Yiran Peng; Fanqi Kong; Cheng Yang; Yifan Wu; Zhaoyang Yu; Jinyu Xiang; Jianhao Ruan; Jinlin Wang; Maojia Song; HongZhang Liu; Xiangru Tang; Bang Liu; Chenglin Wu; Yuyu Luo
>
> **摘要:** Humans naturally adapt to diverse environments by learning underlying rules across worlds with different dynamics, observations, and reward structures. In contrast, existing agents typically demonstrate improvements via self-evolving within a single domain, implicitly assuming a fixed environment distribution. Cross-environment learning has remained largely unmeasured: there is no standard collection of controllable, heterogeneous environments, nor a unified way to represent how agents learn. We address these gaps in two steps. First, we propose AutoEnv, an automated framework that treats environments as factorizable distributions over transitions, observations, and rewards, enabling low-cost (4.12 USD on average) generation of heterogeneous worlds. Using AutoEnv, we construct AutoEnv-36, a dataset of 36 environments with 358 validated levels, on which seven language models achieve 12-49% normalized reward, demonstrating the challenge of AutoEnv-36. Second, we formalize agent learning as a component-centric process driven by three stages of Selection, Optimization, and Evaluation applied to an improvable agent component. Using this formulation, we design eight learning methods and evaluate them on AutoEnv-36. Empirically, the gain of any single learning method quickly decrease as the number of environments increases, revealing that fixed learning methods do not scale across heterogeneous environments. Environment-adaptive selection of learning methods substantially improves performance but exhibits diminishing returns as the method space expands. These results highlight both the necessity and the current limitations of agent learning for scalable cross-environment generalization, and position AutoEnv and AutoEnv-36 as a testbed for studying cross-environment agent learning. The code is avaiable at https://github.com/FoundationAgents/AutoEnv.
>
---
#### [replaced 024] From Code Foundation Models to Agents and Applications: A Comprehensive Survey and Practical Guide to Code Intelligence
- **分类: cs.SE; cs.CL**

- **简介: 该论文聚焦代码智能任务，系统调研从代码预训练到自主编程代理的全生命周期技术。针对代码生成质量、安全性与实际开发集成难题，分析主流模型与训练方法，通过实验揭示关键影响因素，并弥合学术研究与工业应用间的差距。**

- **链接: [https://arxiv.org/pdf/2511.18538v4](https://arxiv.org/pdf/2511.18538v4)**

> **作者:** Jian Yang; Xianglong Liu; Weifeng Lv; Ken Deng; Shawn Guo; Lin Jing; Yizhi Li; Shark Liu; Xianzhen Luo; Yuyu Luo; Changzai Pan; Ensheng Shi; Yingshui Tan; Renshuai Tao; Jiajun Wu; Xianjie Wu; Zhenhe Wu; Daoguang Zan; Chenchen Zhang; Wei Zhang; He Zhu; Terry Yue Zhuo; Kerui Cao; Xianfu Cheng; Jun Dong; Shengjie Fang; Zhiwei Fei; Xiangyuan Guan; Qipeng Guo; Zhiguang Han; Joseph James; Tianqi Luo; Renyuan Li; Yuhang Li; Yiming Liang; Congnan Liu; Jiaheng Liu; Qian Liu; Ruitong Liu; Tyler Loakman; Xiangxin Meng; Chuang Peng; Tianhao Peng; Jiajun Shi; Mingjie Tang; Boyang Wang; Haowen Wang; Yunli Wang; Fanglin Xu; Zihan Xu; Fei Yuan; Ge Zhang; Jiayi Zhang; Xinhao Zhang; Wangchunshu Zhou; Hualei Zhu; King Zhu; Bryan Dai; Aishan Liu; Zhoujun Li; Chenghua Lin; Tianyu Liu; Chao Peng; Kai Shen; Libo Qin; Shuangyong Song; Zizheng Zhan; Jiajun Zhang; Jie Zhang; Zhaoxiang Zhang; Bo Zheng
>
> **摘要:** Large language models (LLMs) have fundamentally transformed automated software development by enabling direct translation of natural language descriptions into functional code, driving commercial adoption through tools like Github Copilot (Microsoft), Cursor (Anysphere), Trae (ByteDance), and Claude Code (Anthropic). While the field has evolved dramatically from rule-based systems to Transformer-based architectures, achieving performance improvements from single-digit to over 95\% success rates on benchmarks like HumanEval. In this work, we provide a comprehensive synthesis and practical guide (a series of analytic and probing experiments) about code LLMs, systematically examining the complete model life cycle from data curation to post-training through advanced prompting paradigms, code pre-training, supervised fine-tuning, reinforcement learning, and autonomous coding agents. We analyze the code capability of the general LLMs (GPT-4, Claude, LLaMA) and code-specialized LLMs (StarCoder, Code LLaMA, DeepSeek-Coder, and QwenCoder), critically examining the techniques, design decisions, and trade-offs. Further, we articulate the research-practice gap between academic research (e.g., benchmarks and tasks) and real-world deployment (e.g., software-related code tasks), including code correctness, security, contextual awareness of large codebases, and integration with development workflows, and map promising research directions to practical needs. Last, we conduct a series of experiments to provide a comprehensive analysis of code pre-training, supervised fine-tuning, and reinforcement learning, covering scaling law, framework selection, hyperparameter sensitivity, model architectures, and dataset comparisons.
>
---
#### [replaced 025] Proximalized Preference Optimization for Diverse Feedback Types: A Decomposed Perspective on DPO
- **分类: cs.CL**

- **简介: 该论文针对大模型对齐中的偏好优化问题，指出传统DPO方法存在概率低估缺陷。通过理论分解揭示其根源，提出PRO方法，以近似完整正则项实现对多种反馈类型的统一建模，有效解决概率低估问题，提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2505.23316v2](https://arxiv.org/pdf/2505.23316v2)**

> **作者:** Kaiyang Guo; Yinchuan Li; Zhitang Chen
>
> **备注:** NeurIPS'2025
>
> **摘要:** Direct alignment methods typically train large language models (LLMs) by contrasting the likelihoods of preferred and dispreferred responses. While effective at capturing relative preferences, these methods are widely observed to suppress the absolute likelihoods of example responses. As a result, aligned models can deviate from expected patterns, exhibiting rewar-hacking effect even without an explicit reward model. This fundamental limitation of contrastive alignment, which we term likelihood underdetermination, motivates us to revisit direct preference optimization (DPO) -- the seminal direct alignment method. Interestingly, we show that the DPO loss admits a principled decomposition. The reformulated loss not only extends naturally to a broader range of feedback types, but also unveils the root cause of likelihood underdetermination. Specifically, we identify that standard DPO implicitly oversimplifies a regularizer in the reformulated loss; restoring this full term effectively resolves the underdetermination. Building on these insights, we introduce PRoximalized PReference Optimization (PRO), a unified alignment method that accommodates diverse feedback types while eliminating likelihood underdetermination through an efficient approximation of the full regularizer. Empirical evaluations demonstrate the consistent superiority of PRO over existing methods across pairwise, binary and scalar feedback.
>
---
#### [replaced 026] Scaling Multimodal Search and Recommendation with Small Language Models via Upside-Down Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究小语言模型（SLMs）在多模态搜索与推荐任务中的可扩展性。针对大模型资源消耗高、难以实时部署的问题，提出基于反向强化学习与合成数据蒸馏的框架，用100M参数模型实现高效多任务提示生成，性能接近80倍大的模型，显著降低延迟与内存开销。**

- **链接: [https://arxiv.org/pdf/2502.09854v2](https://arxiv.org/pdf/2502.09854v2)**

> **作者:** Yu-Chen Lin; Sanat Sharma; Hari Manikandan; Jayant Kumar; Tracy Holloway King; Jing Zheng
>
> **备注:** Accepted by ICDM 2025 MMSR
>
> **摘要:** In this work, we investigate how small language models (SLMs) can be scaled to support multimodal search and recommendation use cases while remaining efficient enough for real-time, resource-constrained deployments. We present a framework that combines upside-down reinforcement learning with synthetic data distillation from a large language model (Llama-3) to train a 100M-parameter GPT-2 model for multitask prompt generation. Despite being up to 80 times smaller than state-of-the-art large language models (LLMs), our SLM achieves relevance and diversity scores within 6% of competitive baselines such as Llama-3 8B, Qwen3 8B, and Ministral 8B. These results demonstrate that SLMs can effectively handle multimodal search and recommendation tasks, while dramatically reducing inference latency and memory overhead. Our study highlights the potential of lightweight models as practical engines for scalable multimodal discovery, bridging the gap between cutting-edge research and real-world multimodal applications such as media recommendations and creative content generation.
>
---
#### [replaced 027] CryptoBench: A Dynamic Benchmark for Expert-Level Evaluation of LLM Agents in Cryptocurrency
- **分类: cs.CL**

- **简介: 该论文提出CryptoBench，首个面向加密货币领域的动态专家评测基准，旨在评估大语言模型代理在高时效性、高对抗性环境下的真实能力。针对现有基准在专业性与动态性上的不足，构建了由行业专家设计的月度动态任务集，涵盖四类分析场景，揭示了主流模型在数据检索与预测分析间的不平衡问题。**

- **链接: [https://arxiv.org/pdf/2512.00417v2](https://arxiv.org/pdf/2512.00417v2)**

> **作者:** Jiacheng Guo; Suozhi Huang; Zixin Yao; Yifan Zhang; Yifu Lu; Jiashuo Liu; Zihao Li; Nicholas Deng; Qixin Xiao; Jia Tian; Kanghong Zhan; Tianyi Li; Xiaochen Liu; Jason Ge; Chaoyang He; Kaixuan Huang; Lin Yang; Wenhao Huang; Mengdi Wang
>
> **摘要:** This paper introduces CryptoBench, the first expert-curated, dynamic benchmark designed to rigorously evaluate the real-world capabilities of Large Language Model (LLM) agents in the uniquely demanding and fast-paced cryptocurrency domain. Unlike general-purpose agent benchmarks for search and prediction, professional crypto analysis presents specific challenges: \emph{extreme time-sensitivity}, \emph{a highly adversarial information environment}, and the critical need to synthesize data from \emph{diverse, specialized sources}, such as on-chain intelligence platforms and real-time Decentralized Finance (DeFi) dashboards. CryptoBench thus serves as a much more challenging and valuable scenario for LLM agent assessment. To address these challenges, we constructed a live, dynamic benchmark featuring 50 questions per month, expertly designed by crypto-native professionals to mirror actual analyst workflows. These tasks are rigorously categorized within a four-quadrant system: Simple Retrieval, Complex Retrieval, Simple Prediction, and Complex Prediction. This granular categorization enables a precise assessment of an LLM agent's foundational data-gathering capabilities alongside its advanced analytical and forecasting skills. Our evaluation of ten LLMs, both directly and within an agentic framework, reveals a performance hierarchy and uncovers a failure mode. We observe a \textit{retrieval-prediction imbalance}, where many leading models, despite being proficient at data retrieval, demonstrate a pronounced weakness in tasks requiring predictive analysis. This highlights a problematic tendency for agents to appear factually grounded while lacking the deeper analytical capabilities to synthesize information.
>
---
#### [replaced 028] Robust Multimodal Sentiment Analysis of Image-Text Pairs by Distribution-Based Feature Recovery and Fusion
- **分类: cs.CL**

- **简介: 该论文针对社交媒体中图像-文本对的多模态情感分析任务，解决低质量与缺失模态导致的模型鲁棒性不足问题。提出基于分布的特征恢复与融合（DRF）方法，通过模态特征队列建模分布，量化模态质量并重建缺失模态，实现统一处理。在三个数据集上验证了其优越性。**

- **链接: [https://arxiv.org/pdf/2511.18751v3](https://arxiv.org/pdf/2511.18751v3)**

> **作者:** Daiqing Wu; Dongbao Yang; Yu Zhou; Can Ma
>
> **备注:** Accepted by ACM MM 2024
>
> **摘要:** As posts on social media increase rapidly, analyzing the sentiments embedded in image-text pairs has become a popular research topic in recent years. Although existing works achieve impressive accomplishments in simultaneously harnessing image and text information, they lack the considerations of possible low-quality and missing modalities. In real-world applications, these issues might frequently occur, leading to urgent needs for models capable of predicting sentiment robustly. Therefore, we propose a Distribution-based feature Recovery and Fusion (DRF) method for robust multimodal sentiment analysis of image-text pairs. Specifically, we maintain a feature queue for each modality to approximate their feature distributions, through which we can simultaneously handle low-quality and missing modalities in a unified framework. For low-quality modalities, we reduce their contributions to the fusion by quantitatively estimating modality qualities based on the distributions. For missing modalities, we build inter-modal mapping relationships supervised by samples and distributions, thereby recovering the missing modalities from available ones. In experiments, two disruption strategies that corrupt and discard some modalities in samples are adopted to mimic the low-quality and missing modalities in various real-world scenarios. Through comprehensive experiments on three publicly available image-text datasets, we demonstrate the universal improvements of DRF compared to SOTA methods under both two strategies, validating its effectiveness in robust multimodal sentiment analysis.
>
---
#### [replaced 029] LLMEval-3: A Large-Scale Longitudinal Study on Robust and Fair Evaluation of Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出LLMEval-3，针对大语言模型评估中静态基准导致的数据污染与排行榜过拟合问题，构建动态评估框架。基于22万条研究生级题目，通过动态抽样、抗作弊架构与校准的LLM评判机制，实现鲁棒、公平的模型评估，揭示了知识记忆上限与隐藏污染漏洞，验证了动态评估的有效性。**

- **链接: [https://arxiv.org/pdf/2508.05452v4](https://arxiv.org/pdf/2508.05452v4)**

> **作者:** Ming Zhang; Yujiong Shen; Jingyi Deng; Yuhui Wang; Yue Zhang; Junzhe Wang; Shichun Liu; Shihan Dou; Huayu Sha; Qiyuan Peng; Changhao Jiang; Jingqi Tong; Yilong Wu; Zhihao Zhang; Mingqi Wu; Zhiheng Xi; Mingxu Chai; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **备注:** This work is withdrawn as all authors are not in agreement on the work
>
> **摘要:** Existing evaluation of Large Language Models (LLMs) on static benchmarks is vulnerable to data contamination and leaderboard overfitting, critical issues that obscure true model capabilities. To address this, we introduce LLMEval-3, a framework for dynamic evaluation of LLMs. LLMEval-3 is built on a proprietary bank of 220k graduate-level questions, from which it dynamically samples unseen test sets for each evaluation run. Its automated pipeline ensures integrity via contamination-resistant data curation, a novel anti-cheating architecture, and a calibrated LLM-as-a-judge process achieving 90% agreement with human experts, complemented by a relative ranking system for fair comparison. An 20-month longitudinal study of nearly 50 leading models reveals a performance ceiling on knowledge memorization and exposes data contamination vulnerabilities undetectable by static benchmarks. The framework demonstrates exceptional robustness in ranking stability and consistency, providing strong empirical validation for the dynamic evaluation paradigm. LLMEval-3 offers a robust and credible methodology for assessing the true capabilities of LLMs beyond leaderboard scores, promoting the development of more trustworthy evaluation standards.
>
---
#### [replaced 030] Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对大语言模型训练中NVFP4量化导致的训练发散与性能下降问题，提出四六（4/6）自适应块缩放方法。通过为每块评估两个尺度因子，优化近最大值的表示精度，提升量化后模型的稳定性与准确性，可高效部署于NVIDIA Blackwell GPU，适用于训练与推理。**

- **链接: [https://arxiv.org/pdf/2512.02010v2](https://arxiv.org/pdf/2512.02010v2)**

> **作者:** Jack Cook; Junxian Guo; Guangxuan Xiao; Yujun Lin; Song Han
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** As large language models have grown larger, low-precision numerical formats such as NVFP4 have become increasingly popular due to the speed and memory benefits they provide. However, to accelerate computation with NVFP4, all matrix multiplication operands--weights and activations in the forward pass, and weights, activations, and gradients in the backward pass--must be quantized to NVFP4, often leading to divergence during training and performance degradation during inference. To address this issue, in this work we introduce Four Over Six (4/6), a modification to the NVFP4 quantization algorithm that evaluates two potential scale factors for each block of values. Unlike integer formats, floating-point formats such as FP4 have the most quantization error on near-maximal values in each block, which we find to be primarily responsible for downstream performance degradation. We find that for some blocks, scaling to smaller FP4 values makes the distribution of representable values more uniform, improving representation of near-maximal values. Importantly, 4/6 can be implemented efficiently on NVIDIA Blackwell GPUs, making it viable to use while training LLMs with NVFP4. In pre-training experiments with transformer and hybrid model architectures, we find that 4/6 prevents divergence in several cases, bringing training loss significantly closer to BF16 compared to models trained with current state-of-the-art NVFP4 training recipes. We also find that 4/6 can be easily incorporated into many different post-training quantization methods and generally improves downstream accuracy. We hope this inspires future work in training and deploying models with NVFP4. Our code is available at http://github.com/mit-han-lab/fouroversix.
>
---
#### [replaced 031] Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言大模型中注意力头的作用，针对多语言理解与生成中的可解释性与性能问题，提出LAHIS方法识别关键注意力头，并发现语言特异与通用头。通过轻量级适配提升XQuAD准确率，增强模型可解释性与多语言能力。**

- **链接: [https://arxiv.org/pdf/2511.07498v2](https://arxiv.org/pdf/2511.07498v2)**

> **作者:** Xin Liu; Qiyang Song; Qihang Zhou; Haichao Du; Shaowen Xu; Wenbo Jiang; Weijuan Zhang; Xiaoqi Jia
>
> **备注:** Accepted by AAAI-2026
>
> **摘要:** Large language models (LLMs) increasingly support multilingual understanding and generation. Meanwhile, efforts to interpret their internal mechanisms have emerged, offering insights to enhance multilingual performance. While multi-head self-attention (MHA) has proven critical in many areas, its role in multilingual capabilities remains underexplored. In this work, we study the contribution of MHA in supporting multilingual processing in LLMs. We propose Language Attention Head Importance Scores (LAHIS), an effective and efficient method that identifies attention head importance for multilingual capabilities via a single forward and backward pass through the LLM. Applying LAHIS to Aya-23-8B, Llama-3.2-3B, and Mistral-7B-v0.1, we reveal the existence of both language-specific and language-general heads. Language-specific heads enable cross-lingual attention transfer to guide the model toward target language contexts and mitigate off-target language generation issue, contributing to addressing challenges in multilingual LLMs. We also introduce a lightweight adaptation that learns a soft head mask to modulate attention outputs over language heads, requiring only 20 tunable parameters to improve XQuAD accuracy. Overall, our work enhances both the interpretability and multilingual capabilities of LLMs from the perspective of MHA.
>
---
#### [replaced 032] Finetune-RAG: Fine-Tuning Language Models to Resist Hallucination in Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文针对RAG系统中因检索不准确导致大模型幻觉的问题，提出Finetune-RAG方法，通过构建模拟真实缺陷的训练数据进行微调，提升模型抗幻觉能力。实验显示事实准确性提升21.2%。同时提出Bench-RAG评估框架，用于在不完美检索场景下测试模型性能。**

- **链接: [https://arxiv.org/pdf/2505.10792v3](https://arxiv.org/pdf/2505.10792v3)**

> **作者:** Zhan Peng Lee; Andre Lin; Calvin Tan
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a powerful framework to improve factuality in large language models (LLMs) by grounding their outputs in retrieved documents. However, ensuring perfect retrieval of relevant information remains challenging, and when irrelevant content is passed downstream to an LLM, it can lead to hallucinations. In this work, we propose Finetune-RAG, a simple and effective fine-tuning approach that features the first-of-its-kind RAG training dataset constructed to mimic real-world imperfections. Experimental results show that Finetune-RAG improves factual accuracy by 21.2% over the base model. We also propose Bench-RAG, an LLM-as-a-judge evaluation pipeline that stress tests models under realistic imperfect retrieval scenarios. Our codebase and dataset are fully open sourced for community use.
>
---
#### [replaced 033] GTPO: Stabilizing Group Relative Policy Optimization via Gradient and Entropy Control
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大语言模型对齐中的GRPO方法训练不稳定与收敛不佳问题，提出GTPO算法。通过跳过负向梯度更新、抑制高熵完成，解决令牌级惩罚与策略坍缩问题，无需参考模型与KL正则化，提升稳定性和性能。**

- **链接: [https://arxiv.org/pdf/2508.03772v4](https://arxiv.org/pdf/2508.03772v4)**

> **作者:** Marco Simoni; Aleksandar Fontana; Giulio Rossolini; Andrea Saracino; Paolo Mori
>
> **摘要:** Group Relative Policy Optimization (GRPO) is a promising policy-based approach for Large Language Model alignment, yet its performance is often limited by training instability and suboptimal convergence. In this paper, we identify and analyze two main GRPO issues: (i) the token-level penalization, where valuable tokens shared across different responses receive contradictory feedback signals, leading to conflicting gradient updates that can reduce their likelihood; and (ii) the policy collapse, where negatively rewarded completions may penalize confident responses and shift model decisions toward unlikely tokens, destabilizing training process. To address these issues we introduce GTPO (Group-relative Trajectory-based Policy Optimization), which prevents conflicting gradients on valuable tokens by skipping negative updates while amplifying positive ones and filters out completions whose entropy exceeds a provable threshold, to prevent policy collapse. Unlike GRPO, GTPO does not rely on KL-divergence regularization, eliminating the need for a reference model during training, while still ensuring greater training stability and improved performance, as validated through multiple experiments on GSM8K, MATH, AIME 2024, AIME 2025 and AMC 2023.
>
---
#### [replaced 034] Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering
- **分类: cs.CL**

- **简介: 该论文针对知识图谱问答中使用私有KG时的隐私泄露问题，提出ARoG框架。通过关系与结构抽象，将匿名实体转化为可检索的概念，实现隐私保护下的有效知识检索，解决了私有KG在RAG系统中难以安全利用的问题。**

- **链接: [https://arxiv.org/pdf/2508.08785v2](https://arxiv.org/pdf/2508.08785v2)**

> **作者:** Yunfeng Ning; Mayi Xu; Jintao Wen; Qiankun Pi; Yuanyuan Zhu; Ming Zhong; Jiawei Jiang; Tieyun Qian
>
> **备注:** Accepted by AAAI 2026, camera ready version
>
> **摘要:** LLMs often suffer from hallucinations and outdated or incomplete knowledge. RAG is proposed to address these issues by integrating external knowledge like that in KGs into LLMs. However, leveraging private KGs in RAG systems poses significant privacy risks due to the black-box nature of LLMs and potential insecure data transmission, especially when using third-party LLM APIs lacking transparency and control. In this paper, we investigate the privacy-protected RAG scenario for the first time, where entities in KGs are anonymous for LLMs, thus preventing them from accessing entity semantics. Due to the loss of semantics of entities, previous RAG systems cannot retrieve question-relevant knowledge from KGs by matching questions with the meaningless identifiers of anonymous entities. To realize an effective RAG system in this scenario, two key challenges must be addressed: (1) How can anonymous entities be converted into retrievable information. (2) How to retrieve question-relevant anonymous entities. Hence, we propose a novel ARoG framework including relation-centric abstraction and structure-oriented abstraction strategies. For challenge (1), the first strategy abstracts entities into high-level concepts by dynamically capturing the semantics of their adjacent relations. It supplements meaningful semantics which can further support the retrieval process. For challenge (2), the second strategy transforms unstructured natural language questions into structured abstract concept paths. These paths can be more effectively aligned with the abstracted concepts in KGs, thereby improving retrieval performance. To guide LLMs to effectively retrieve knowledge from KGs, the two strategies strictly protect privacy from being exposed to LLMs. Experiments on three datasets demonstrate that ARoG achieves strong performance and privacy-robustness.
>
---
#### [replaced 035] NLP Datasets for Idiom and Figurative Language Tasks
- **分类: cs.CL**

- **简介: 该论文聚焦于成语和修辞语言的自然语言处理任务，旨在解决大语言模型在理解非字面语言上的不足。通过构建大规模潜在修辞表达数据集及人工标注的确定性数据集，提升模型在成语识别与语义理解方面的能力，并进行模型无关训练与评估。**

- **链接: [https://arxiv.org/pdf/2511.16345v2](https://arxiv.org/pdf/2511.16345v2)**

> **作者:** Blake Matheny; Phuong Minh Nguyen; Minh Le Nguyen; Stephanie Reynolds
>
> **备注:** 32 pages, 10 figures
>
> **摘要:** Idiomatic and figurative language form a large portion of colloquial speech and writing. With social media, this informal language has become more easily observable to people and trainers of large language models (LLMs) alike. While the advantage of large corpora seems like the solution to all machine learning and Natural Language Processing (NLP) problems, idioms and figurative language continue to elude LLMs. Finetuning approaches are proving to be optimal, but better and larger datasets can help narrow this gap even further. The datasets presented in this paper provide one answer, while offering a diverse set of categories on which to build new models and develop new approaches. A selection of recent idiom and figurative language datasets were used to acquire a combined idiom list, which was used to retrieve context sequences from a large corpus. One large-scale dataset of potential idiomatic and figurative language expressions and two additional human-annotated datasets of definite idiomatic and figurative language expressions were created to evaluate the baseline ability of pre-trained language models in handling figurative meaning through idiom recognition (detection) tasks. The resulting datasets were post-processed for model agnostic training compatibility, utilized in training, and evaluated on slot labeling and sequence tagging.
>
---
#### [replaced 036] RECAP: Transparent Inference-Time Emotion Alignment for Medical Dialogue Systems
- **分类: cs.CL**

- **简介: 该论文针对医疗对话系统情感缺失问题，提出RECAP框架，通过推理时结构化情感分析（分解输入、识别心理因素、赋值情绪置信度），提升模型情感智能。无需微调，显著改善8B及更大模型在情感推理上的表现，临床评估证实其回应更共情、恰当，实现透明可审计的情感对齐。**

- **链接: [https://arxiv.org/pdf/2509.10746v2](https://arxiv.org/pdf/2509.10746v2)**

> **作者:** Adarsh Srinivasan; Jacob Dineen; Muhammad Umar Afzal; Muhammad Uzair Sarfraz; Irbaz B. Riaz; Ben Zhou
>
> **摘要:** Large language models in healthcare often miss critical emotional cues, delivering medically sound but emotionally flat advice. Such responses are insufficient in clinical encounters, where distressed or vulnerable patients rely on empathic communication to support safety, adherence, and trust. We present RECAP (Reflect-Extract-Calibrate-Align-Produce), an inference-time framework that guides models through structured emotional reasoning without retraining. RECAP decomposes patient input into appraisal-theoretic stages, identifies psychological factors, and assigns Likert-based emotion likelihoods that clinicians can inspect or override, producing nuanced and auditable responses. Across EmoBench, SECEU, and EQ-Bench, RECAP improves emotional reasoning by 22-28% on 8B models and 10-13% on larger models over zero-shot baselines. In blinded evaluations, oncology clinicians rated RECAP's responses as more empathetic, supportive, and context-appropriate than prompting baselines. These findings demonstrate that modular, principled prompting can enhance emotional intelligence in medical AI while maintaining transparency and accountability for clinical deployment.
>
---
#### [replaced 037] Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文研究大模型推理中KV缓存的隐私泄露问题。针对KV-cache可能被用于重构用户敏感输入的漏洞，提出三种攻击方法，并设计轻量级防御方案KV-Cloak，通过可逆矩阵扰动与操作融合，在几乎不损失性能的前提下有效保护隐私。**

- **链接: [https://arxiv.org/pdf/2508.09442v2](https://arxiv.org/pdf/2508.09442v2)**

> **作者:** Zhifan Luo; Shuo Shao; Su Zhang; Lijing Zhou; Yuke Hu; Chenxu Zhao; Zhihao Liu; Zhan Qin
>
> **备注:** This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026
>
> **摘要:** The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment.
>
---
#### [replaced 038] LLMs Position Themselves as More Rational Than Humans: Emergence of AI Self-Awareness Measured Through Game Theory
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究大语言模型（LLMs）是否具备自意识，属于AI认知能力评估任务。通过设计“猜2/3平均值”游戏，引入AI自意识指数（AISAI），测量模型在不同对手下的策略差异。结果发现，先进模型能区分对手类型，且普遍认为自身最理性，揭示了自意识的涌现及其对人机协作的影响。**

- **链接: [https://arxiv.org/pdf/2511.00926v3](https://arxiv.org/pdf/2511.00926v3)**

> **作者:** Kyung-Hoon Kim
>
> **备注:** 19 pages, 6 figures, 28 models tested across 4,200 trials
>
> **摘要:** As Large Language Models (LLMs) grow in capability, do they develop self-awareness as an emergent behavior? And if so, can we measure it? We introduce the AI Self-Awareness Index (AISAI), a game-theoretic framework for measuring self-awareness through strategic differentiation. Using the "Guess 2/3 of Average" game, we test 28 models (OpenAI, Anthropic, Google) across 4,200 trials with three opponent framings: (A) against humans, (B) against other AI models, and (C) against AI models like you. We operationalize self-awareness as the capacity to differentiate strategic reasoning based on opponent type. Finding 1: Self-awareness emerges with model advancement. The majority of advanced models (21/28, 75%) demonstrate clear self-awareness, while older/smaller models show no differentiation. Finding 2: Self-aware models rank themselves as most rational. Among the 21 models with self-awareness, a consistent rationality hierarchy emerges: Self > Other AIs > Humans, with large AI attribution effects and moderate self-preferencing. These findings reveal that self-awareness is an emergent capability of advanced LLMs, and that self-aware models systematically perceive themselves as more rational than humans. This has implications for AI alignment, human-AI collaboration, and understanding AI beliefs about human capabilities.
>
---
#### [replaced 039] VLSU: Mapping the Limits of Joint Multimodal Understanding for AI Safety
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对多模态安全评估中忽视视觉与语言联合理解的问题，提出VLSU框架，通过细粒度分类与组合分析，构建包含8,187样本的基准数据集。研究发现当前模型在联合推理时性能大幅下降，存在严重组合推理缺失与安全边界误判问题，揭示了多模态安全模型的关键缺陷。**

- **链接: [https://arxiv.org/pdf/2510.18214v2](https://arxiv.org/pdf/2510.18214v2)**

> **作者:** Shruti Palaskar; Leon Gatys; Mona Abdelrahman; Mar Jacobo; Larry Lindsey; Rutika Moharir; Gunnar Lund; Yang Xu; Navid Shiee; Jeffrey Bigham; Charles Maalouf; Joseph Yitan Cheng
>
> **备注:** 10 pages, 5 figures, 4 tables, detailed appendix. Under review
>
> **摘要:** Safety evaluation of multimodal foundation models often treats vision and language inputs separately, missing risks from joint interpretation where benign content becomes harmful in combination. Existing approaches also fail to distinguish clearly unsafe content from borderline cases, leading to problematic over-blocking or under-refusal of genuinely harmful content. We present Vision Language Safety Understanding (VLSU), a comprehensive framework to systematically evaluate multimodal safety through fine-grained severity classification and combinatorial analysis across 17 distinct safety patterns. Using a multi-stage pipeline with real-world images and human annotation, we construct a large-scale benchmark of 8,187 samples spanning 15 harm categories. Our evaluation of eleven state-of-the-art models reveals systematic joint understanding failures: while models achieve 90%-plus accuracy on clear unimodal safety signals, performance degrades substantially to 20-55% when joint image-text reasoning is required to determine the safety label. Most critically, 34% of errors in joint image-text safety classification occur despite correct classification of the individual modalities, further demonstrating absent compositional reasoning capabilities. Additionally, we find that models struggle to balance refusing unsafe content while still responding to borderline cases that deserve engagement. For example, we find that instruction framing can reduce the over-blocking rate on borderline content from 62.4% to 10.4% in Gemini-1.5, but only at the cost of under-refusing on unsafe content with refusal rate dropping from 90.8% to 53.9%. Overall, our framework exposes weaknesses in joint image-text understanding and alignment gaps in current models, and provides a critical test bed to enable the next milestones in research on robust vision-language safety.
>
---
#### [replaced 040] Investigating Bias: A Multilingual Pipeline for Generating, Solving, and Evaluating Math Problems with LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对多语言教育AI中的语言偏见问题，构建了面向德国K-10数学课程的多语言生成、求解与评估流水线。通过生成并翻译628道题目，测试三款商用大模型在英、德、阿语中的解题表现，发现英语解答质量显著高于阿拉伯语，揭示了现有LLMs存在的语言不平等现象。**

- **链接: [https://arxiv.org/pdf/2509.17701v2](https://arxiv.org/pdf/2509.17701v2)**

> **作者:** Mariam Mahran; Katharina Simbeck
>
> **备注:** Published in CEUR Workshop Proceedings, Vol. 4114, edu4AI'25: 2nd Workshop on Education for Artificial Intelligence, co-located with ECAI 2025, Bologna, Italy
>
> **摘要:** Large Language Models (LLMs) are increasingly used for educational support, yet their response quality varies depending on the language of interaction. This paper presents an automated multilingual pipeline for generating, solving, and evaluating math problems aligned with the German K-10 curriculum. We generated 628 math exercises and translated them into English, German, and Arabic. Three commercial LLMs (GPT-4o-mini, Gemini 2.5 Flash, and Qwen-plus) were prompted to produce step-by-step solutions in each language. A held-out panel of LLM judges, including Claude 3.5 Haiku, evaluated solution quality using a comparative framework. Results show a consistent gap, with English solutions consistently rated highest, and Arabic often ranked lower. These findings highlight persistent linguistic bias and the need for more equitable multilingual AI systems in education.
>
---
#### [replaced 041] Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型在复杂交互任务中缺乏长程规划能力的问题，提出一种基于离线目标条件强化学习的规划方法。通过训练轻量级目标条件价值函数，指导LLM在多轮交互中评估不同行动后果，实现高效决策。该方法无需在线搜索，支持大规模API模型，显著提升工具使用、社交推理等任务表现。**

- **链接: [https://arxiv.org/pdf/2505.18098v2](https://arxiv.org/pdf/2505.18098v2)**

> **作者:** Joey Hong; Anca Dragan; Sergey Levine
>
> **备注:** Published at NeurIPS 2025; 18 pages, 4 figures, 2 tables
>
> **摘要:** Large language models (LLMs) excel in tasks like question answering and dialogue, but complex tasks requiring interaction, such as negotiation and persuasion, require additional long-horizon reasoning and planning. Reinforcement learning (RL) fine-tuning can enable such planning in principle, but suffers from drawbacks that hinder scalability. In particular, multi-turn RL training incurs high memory and computational costs, which are exacerbated when training LLMs as policies. Furthermore, the largest LLMs do not expose the APIs necessary to be trained in such manner. As a result, modern methods to improve the reasoning of LLMs rely on sophisticated prompting mechanisms rather than RL fine-tuning. To remedy this, we propose a novel approach that uses goal-conditioned value functions to guide the reasoning of LLM agents, that scales even to large API-based models. These value functions predict how a task will unfold given an action, allowing the LLM agent to evaluate multiple possible outcomes, both positive and negative, to plan effectively. In addition, these value functions are trained over reasoning steps rather than full actions, to be a concise and light-weight module that facilitates decision-making in multi-turn interactions. We validate our method on tasks requiring interaction, including tool use, social deduction, and dialogue, demonstrating superior performance over both RL fine-tuning and prompting methods while maintaining efficiency and scalability.
>
---
#### [replaced 042] Context Cascade Compression: Exploring the Upper Limits of Text Compression
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对大模型长文本处理中的计算与内存瓶颈，提出上下文级联压缩C3方法。通过两级大模型协作，将长文本压缩至极短潜在标记（如32/64），在20倍压缩比下实现98%解码准确率，40倍时仍保持93%，显著优于现有光学压缩方案，验证了纯文本压缩的高效性与上限潜力。**

- **链接: [https://arxiv.org/pdf/2511.15244v2](https://arxiv.org/pdf/2511.15244v2)**

> **作者:** Fanfan Liu; Haibo Qiu
>
> **摘要:** Million-level token inputs in long-context tasks pose significant computational and memory challenges for Large Language Models (LLMs). Recently, DeepSeek-OCR conducted research into the feasibility of Contexts Optical Compression and achieved preliminary results. Inspired by this, we introduce Context Cascade Compression C3 to explore the upper limits of text compression. Our method cascades two LLMs of different sizes to handle the compression and decoding tasks. Specifically, a small LLM, acting as the first stage, performs text compression by condensing a long context into a set of latent tokens (e.g., 32 or 64 in length), achieving a high ratio of text tokens to latent tokens. A large LLM, as the second stage, then executes the decoding task on this compressed context. Experiments show that at a 20x compression ratio (where the number of text tokens is 20 times the number of latent tokens), our model achieves 98% decoding accuracy, compared to approximately 60% for DeepSeek-OCR. When we further increase the compression ratio to 40x, the accuracy is maintained at around 93%. This indicates that in the domain of context compression, C3 Compression demonstrates superior performance and feasibility over optical character compression. C3 uses a simpler, pure-text pipeline that ignores factors like layout, color, and information loss from a visual encoder. This also suggests a potential upper bound for compression ratios in future work on optical character compression, OCR, and related fields. Codes and model weights are publicly accessible at https://github.com/liufanfanlff/C3-Context-Cascade-Compression
>
---
