# 自然语言处理 cs.CL

- **最新发布 192 篇**

- **更新 111 篇**

## 最新发布

#### [new 001] Narrative Flattening: How Post-Training Compresses Thematic, Affective, and Stylistic Variation in LLM Fiction
- **分类: cs.CL**

- **简介: 该论文研究语言模型在后训练阶段导致叙事内容变平的现象，属于自然语言生成任务。它探讨后训练如何压缩主题、情感和风格的多样性，通过对比不同数据源的文本分析效果。**

- **链接: [https://arxiv.org/pdf/2605.27878](https://arxiv.org/pdf/2605.27878)**

> **作者:** Zehan Li; Yutong Zhu; Siyang Wu; Honglin Bao; James A. Evans
>
> **摘要:** Large language models produce fluent fiction, yet their creative output is widely seen as flat. We ask where this quality originates in the training and whether it affects different domains of human fiction equally. We construct a matched story-continuation paradigm across StoryStar (public-platform), TMAS (prompt-guided), and The New Yorker (professional literary)-and compare continuations from four OLMo 32B checkpoints (Base, SFT, DPO, RLVR) against matched human text. Because these checkpoints share architecture, scale, tokenizer, and pretraining, the design isolates the post-training effect. We measure each continuation along three sentence-level dimensions: thematic motion, affective prevalence, and linguistic diversity. Across all three, post-training compresses dynamic variation: thematic transitions become more uniform, high-intensity emotions give way to neutrality, and stylistic diversity across stories shrinks. We term this progressive loss narrative flattening. The effect is directionally stable across story domains but gap size depends on the human baseline: professional literary fiction is compressed most, while public-platform and prompt-guided stories show smaller gaps, consistent with their human baselines sitting closer to the model's default rhythm. Post-trained endpoints converge across domains, suggesting alignment produces a continuation regime largely insensitive to the source domain's narrative texture.
>
---
#### [new 002] MemTrace: Tracing and Attributing Errors in Large Language Model Memory Systems
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于LLM内存系统调试任务，解决内存错误追踪与归因问题。提出MemTrace框架，构建基准测试，实现错误定位与自动优化。**

- **链接: [https://arxiv.org/pdf/2605.28732](https://arxiv.org/pdf/2605.28732)**

> **作者:** Xinle Deng; Ruobin Zhong; Hujin Peng; Xiaoben Lu; Yanzhe Wu; Guang Li; Buqiang Xu; Yunzhi Yao; Jizhan Fang; Haoliang Cao; Junjie Guo; Yuan Yuan; Ziqing Ma; Yuanqiang Yu; Rui Hu; Baohua Dong; Hangcheng Zhu; Ningyu Zhang
>
> **备注:** Ongoing work
>
> **摘要:** Memory is essential for enabling large language models to support long-horizon reasoning, yet existing memory systems remain unreliable and difficult to debug. Tracing memory's dynamic evolution is crucial to understand how information is synthesized, propagated, or corrupted over time. In this work, we study the new problem of error tracing and attribution in LLM memory systems. We propose a novel framework that transforms memory pipelines into executable memory evolution graphs, enabling fine-grained tracing of operational information flow. We then construct MemTraceBench, a benchmark collected from representative memory systems such as Long-Context, RAG, Mem0, and EverMemOS, to systematically study memory failure modes. We further introduce an automatic attribution method that iteratively traces operation subgraphs to pinpoint the root cause of any failed case. Our analysis reveals that memory failures are systematic, stemming from operation-level issues like information loss and retrieval misalignment. Crucially, we leverage these fine-grained attribution signals to guide downstream prompt optimization, establishing a closed-loop system that automatically corrects faults and boosts end-task performance by up to 7.62%. Code will be released at this https URL.
>
---
#### [new 003] Pruning and Distilling Mixture-of-Experts into Dense Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型压缩任务，旨在解决MoE模型内存占用高的问题。通过评分、分组和知识蒸馏，将MoE转换为密集模型，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2605.28207](https://arxiv.org/pdf/2605.28207)**

> **作者:** Junhyuck Kim; Jihun Yun; Haechan Kim; Gyeongman Kim; Joonghyun Bae; Jaewoong Cho
>
> **摘要:** Mixture-of-Experts (MoE) is now the dominant architecture for frontier language models, yet it requires all expert parameters to be loaded in memory, making it less preferable for memory-constrained deployment. Existing compression methods reduce the number of experts but the output remains an MoE model with the same fundamental limitation. We present the first systematic framework for converting a trained MoE into a standard fully dense architecture: experts are scored, selected, and grouped, then concatenated into a dense FFN and refined by knowledge distillation from the MoE teacher. We evaluate 7 scoring, 5 grouping, and 2 magnitude scaling methods across a range of selected expert counts on Qwen3-30B-A3B, yielding 350 configurations. We find that the choice of scoring method is the most impactful, with our novel diversity-aware scoring consistently outperforming prior methods on Qwen3-30B-A3B, DeepSeek-V2-Lite, and GPT-OSS-20B. Under a controlled comparison at matched parameter count, MoE-to-dense outperforms dense-to-dense pruning by +6.3 pp in average downstream accuracy after ~4B-token distillation at 1.6x faster training wall-clock speed.
>
---
#### [new 004] UNIQUE: Universal Top-k Sparse Attention for Training-free Inference and Sparsity-aware Training
- **分类: cs.CL**

- **简介: 该论文提出UNIQUE框架，解决长上下文推理中自注意力计算效率低的问题。通过稀疏注意力机制提升推理速度，同时保持模型性能。属于自然语言处理中的高效推理任务。**

- **链接: [https://arxiv.org/pdf/2605.27740](https://arxiv.org/pdf/2605.27740)**

> **作者:** Keqi Deng; Shaoshi Ling; Ruchao Fan; Jinyu Li
>
> **摘要:** Long-context inference in large language models (LLMs) is bottlenecked by the linear growth of the self-attention key-value (KV) cache. Top-k sparse attention alleviates this by loading only a small fraction of the KV cache, but accurately and cheaply estimating cache importance, for both training-free use and sparsity-aware training, remains challenging. This paper proposes UNIQUE, a universal top-k sparse attention framework that addresses both requirements and stays consistently effective across LLM modalities. UNIQUE operates at the granularity of KV pages and estimates per-page importance with a simple yet accurate score combining the mean of the page's keys as a representative vector with their standard deviation as an offset term. To further close the train-inference gap, this paper introduces a soft-mask sparsity-aware training scheme that uses the top-k score boundary as a per-query threshold and a sigmoid soft mask around it, requiring neither auxiliary losses nor architectural changes. Experiments on text and speech LLMs show that UNIQUE preserves task performance on long-context benchmarks such as LongBench Pro and on long-form speech recognition, while delivering up to 11.4x attention-kernel speedup over FlashInfer dense attention and at least 5.3x end-to-end decoding speedup over a vLLM-based dense model.
>
---
#### [new 005] Mobile-Aptus: Confidence-Driven Proactive and Robust Interaction in MLLM-based Mobile-Using Agents
- **分类: cs.CL**

- **简介: 该论文属于移动代理任务，解决过执行和过度请求问题。提出Mobile-Aptus框架，通过信心驱动实现更稳健的交互。**

- **链接: [https://arxiv.org/pdf/2605.28629](https://arxiv.org/pdf/2605.28629)**

> **作者:** Zheng Wu; Pengzhou Cheng; Zongru Wu; Yuan Guo; Tianjie Ju; Aston Zhang; Gongshen Liu; Zhuosheng Zhang
>
> **备注:** Accepted by TASLP
>
> **摘要:** Recent advancements in multimodal large language models (MLLMs) have shown exceptional potential in enabling mobile-using agents to autonomously execute human instructions. However, fully automated agents often try to execute tasks even when they are unable to resolve them, leading to the problem of over-execution. Previous studies solve it by training a interactive mobile-using agents to let agents request human interaction when agents can not complete user instructions. However, we find that these interactive agents tend to exhibit over-soliciting behavior, relying excessively on human intervention. To mitigate both over-execution and over-soliciting, we propose a universal confidence integration framework that enables confidence-driven proactive and robust interaction in MLLM-based mobile-using agents. The framework consists of two stages: interaction capability empowerment and confidence bias correction. In the interaction capability empowerment stage, agents learn through supervised fine-tuning to output both actions and confidence scores. In the confidence bias correction stage, agents learn to output more accurate confidence scores by combining semantic similarity retrieval with direct preference optimization. Experimental results show Mobile-Aptus achieves state-of-the-art performance on the four popular mobile-using agent benchmarks: OS-Kairos, AITZ, Meta-GUI, and AndroidControl. Mobile-Aptus consistently outperforms all baselines in offline benchmarks, with an average improvement over 17\% in task success rate. In real-world dynamic experiments, Mobile-Aptus surpasses the baseline by 26% in task success rate with only 0.64 intervention steps per instruction. The codes are available at this https URL.
>
---
#### [new 006] PromptEmbedder:: Efficient and Transferable Text Embedding via Dual-LLM Soft Prompting
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本嵌入任务，旨在解决LLM适应新架构时效率低和迁移性差的问题。提出PromptEmbedder框架，通过双LLM结构分离知识与权重，提升计算效率和跨架构迁移能力。**

- **链接: [https://arxiv.org/pdf/2605.28066](https://arxiv.org/pdf/2605.28066)**

> **作者:** Yu-Che Tsai; Kuan-Yu Chen; Yuan-Hao Chen; Yu-Han Chang; Ching-Yu Tsai; Yu-Hsiang Chuang; Shou-De Lin
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable efficacy in text embedding, yet current adaptation methods like LoRA face significant bottlenecks in computational efficiency and cross-architecture transferability. Whenever a new backbone emerges, existing approaches require costly retraining from scratch. To address this, we propose PromptEmbedder, a novel dual-LLM framework that decouples embedding knowledge from specific backbone weights. PromptEmbedder utilizes a Prompting LLM to generate instruction-aware soft prompts for a frozen Embedding LLM via a differentiable generation process with continuous relaxation, ensuring full gradient flow during contrastive training. By localizing task-specific knowledge within the Prompting LLM, adapting to new architectures requires only retraining a lightweight linear alignment matrix. Evaluations on the MTEB benchmark show that PromptEmbedder achieves comparable performance with LoRA finetuning while reducing GPU memory by 40% and accelerating training by 3.7x. Our approach establishes a scalable, architecture-agnostic paradigm for efficient LLM-based representation learning.
>
---
#### [new 007] Learning to Translate from Soft to Hard LLM Prompts
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决软提示可解释性差的问题。通过训练翻译模型将软提示转换为自然语言，提升翻译质量与可解释性，并在大模型上表现优于原始软提示。**

- **链接: [https://arxiv.org/pdf/2605.27642](https://arxiv.org/pdf/2605.27642)**

> **作者:** Pitipat Kongsomjit; Suryansh Goyal; Jacob Whitehill
>
> **备注:** 8 Pages, 11 tables, 4 Figures
>
> **摘要:** Soft prompt tuning is a parameter-efficient method for adapting LLMs to specific tasks, but suffers from a lack of interpretability. Building on recent work on interpreting soft prompts (Ramati et al., 2024), we explore how training a dedicated soft prompt to natural language translation model can yield higher translation quality. In particular, in both quantitative and qualitative comparisons on multiple Datasets of Datasets (DoDs), we demonstrate that our translator produces fluent, accurate verbalizations that outperforms existing training-free methods like InSPEcT. In addition to advancing interpretability, our work suggests a promising downstream application: soft prompts optimized on small, open-source models can be translated into portable text prompts that, when deployed on larger closed-API models, exceed the performance of the original soft prompt and, in some cases, even few-shot learning.
>
---
#### [new 008] The Attentional White Bear Effect in Transformer Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，研究指令抑制在语言模型中的效果。它探讨抑制是否减少内部表示或仅抑制表达，通过多种实验发现禁止概念仍影响模型内部表征和生成。**

- **链接: [https://arxiv.org/pdf/2605.28639](https://arxiv.org/pdf/2605.28639)**

> **作者:** Rebecca Ramnauth; Brian Scassellati
>
> **备注:** Currently under review at EMNLP 2026
>
> **摘要:** Instruction-based suppression is widely used to prevent language models from generating prohibited content, yet it remains unclear whether suppression reduces internal representation or merely suppresses expression. We investigate this question through representational probing, attention analysis, and behavioral semantic leakage experiments across multiple transformer models. We find that prohibited concepts remain highly recoverable from hidden representations under suppression, continue to influence attention routing, and measurably shape downstream generations despite successful lexical avoidance. These effects persist across pooling strategies, indirect semantic controls, and multiple model families. Our results expose a fundamental gap between behavioral and representational alignment.
>
---
#### [new 009] Periodic RoPE for Infinite Context LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决长序列建模中位置编码失效的问题。提出Periodic RoPE与滑动窗口注意力结合，实现无限上下文支持。**

- **链接: [https://arxiv.org/pdf/2605.27980](https://arxiv.org/pdf/2605.27980)**

> **作者:** Simin Huo
>
> **备注:** 5 pages
>
> **摘要:** The ability to process ultra-long contexts is crucial for large language models (LLMs) to perform long-horizon tasks. While recent efforts have extended context windows to 1M and beyond, model performance degrades when sequence length exceeds the pre-trained range of positional encodings (e.g., RoPE), i.e., position exhaustion. This fundamental limitation must be overcome to achieve a truly infinite context. To address it, we propose Periodic RoPE (P-RoPE), a positional encoding mechanism designed to circumvent this exhaustion. It operates in conjunction with sliding window attention (SWA) to capture local dependencies and relative positions within each window. This local layer is then complemented by a global attention layer with No Positional Encoding (NoPE), enabling unbounded interaction across the entire sequence without positional constraints. By stacking these two types of layers, the model avoids the need for positional extrapolation to generalize longer and theoretically supports an infinite context window. Empirical results show that our model, MiniWin, outperforms MiniMInd with standard GPT architectures in long-context efficiency and stability. Our work provides a possible pathway toward LLMs with genuine infinite-context understanding. The code is available at \href{this https URL}{this https URL}.
>
---
#### [new 010] An Evolutionary Approach for Designing Stable and Highly Expressible Low-Immunogenicity Therapeutic mRNA Sequences
- **分类: cs.CL; q-bio.QM**

- **简介: 该论文属于mRNA序列设计任务，旨在优化翻译效率、稳定性及降低免疫原性。通过结合深度学习与进化算法，实现高效、稳定的mRNA序列设计。**

- **链接: [https://arxiv.org/pdf/2605.27986](https://arxiv.org/pdf/2605.27986)**

> **作者:** Dhawa Sang Dong; Mausam Gurung; Suraj Kandel
>
> **摘要:** Messenger RNA (mRNA) sequences as therapeutics require optimized design to ensure efficient translation, structural stability, and minimal immunogenicity. This study presents a two-stage in-silico framework that integrates deep learning and evolutionary computation for rational mRNA optimization instead of existing state-of-the-art models. In the first stage, a pretrained CodonTransformer (BERT-like Large Language Model) generates biologically coherent mRNA sequences encoding the target antigen. In the second stage, a genetic algorithm (GA) evolves these candidate sequences through codon-aware crossover and synonymous mutation guided by human codon usage preferences. Fitness functions for evaluation combined translation-related metrics (CAI, tAI, codon-pair bias), mRNA structural stability (local and global MFE via RNAfold, GC content), and reduced immunogenicity (CpG/UpA motif frequency). Over successive generations (38th, 40th, and 42nd), the GA improved (achieved CAI values of 0.73 to 0.74 and tAI values of 0.63 to 0.64) CAI and tAI by over 6% and codon-pair bias is high and consistent (0.97 ) and improved ribosomal accessibility at the 5' end, with an unpaired_30 fraction reaching 0.87; Global Minimum Free Energy (MFE) converged to a balanced range of -346 to -356 kcal/mol, achieving approximately 84% base-paired structural stability, and reduced immune-stimulatory motifs - lowering the average immune penalty to 27.3 in the final generation. Linear Design produces hyper-stable transcripts (MFE < - 2000 kcal/mol) that risk translation inefficiency due to extreme rigidity, and BiLSTM-CRF focuses solely on high CAI (0.96 to 0.98) without structural constraints, our framework achieves an optimal translation-stability equilibrium, highlighting the proposed BERT-GA framework as an effective, data-driven approach for the design and optimization of in-silico mRNA sequences.
>
---
#### [new 011] Beyond Chunk-Local Extraction: Cross-Chunk Graph Augmentation for GraphRAG
- **分类: cs.CL**

- **简介: 该论文属于知识图谱与问答任务，解决GraphRAG中跨块关系缺失的问题。通过CrossAug方法，在离线阶段增强知识图谱的跨块结构，提升复杂问答性能。**

- **链接: [https://arxiv.org/pdf/2605.28004](https://arxiv.org/pdf/2605.28004)**

> **作者:** Jiaming Zhang; Yibo Zhao; Jing Yu; Jianxiang Yu; Xiang Li
>
> **备注:** 15 pages, 5 figures, 8 tables
>
> **摘要:** GraphRAG extends retrieval-augmented generation by organizing corpora as explicit knowledge graphs, enabling graph-based retrieval for complex question answering. However, existing frameworks extract entities and relations within individual chunks, leaving cross-chunk relations -- those whose evidence spans multiple passages -- systematically absent from the index. Exhaustive LLM-based recovery of such relations is impractical due to the combinatorial explosion of chunk combinations. We present CrossAug, a GNN-guided CROSS-Chunk Graph AUGmentation method that enriches GraphRAG indices with cross-chunk relational structure as an offline step before query-time retrieval. CrossAug derives training supervision through self-supervised graph corruption, uses a topology-aware GNN to score subgraphs for missingness, and applies evidence-grounded LLM completion only to selected high-scoring regions. Experiments on three LLM-based GraphRAG frameworks across four multi-hop and long-document QA benchmarks demonstrate that CrossAug consistently improves performance, confirming the benefit of cross-chunk graph augmentation for retrieval-based question answering. Our code is available at this https URL.
>
---
#### [new 012] Human Label Variation as Stable Signal: Learning Annotator-Specific Explanation Behavior via Cross-Annotator Preference Optimization
- **分类: cs.CL**

- **简介: 该论文研究如何让大语言模型学习标注者特定的标签-解释行为。任务为自然语言推理和释义判断，解决标注者间解释差异的学习问题，提出CAPO方法提升模仿效果。**

- **链接: [https://arxiv.org/pdf/2605.28802](https://arxiv.org/pdf/2605.28802)**

> **作者:** Beiduo Chen; Pingjun Hong; Ziyun Zhang; Benjamin Roth; Anna Korhonen; Barbara Plank
>
> **备注:** 43 pages, 20 figures
>
> **摘要:** Free-text explanations extend human label variation (HLV) beyond label disagreement by revealing the reasoning and preferences behind annotators' decisions. We study whether large language models (LLMs) can learn and reproduce such annotator-specific label-explanation behavior. Using two sentence-pair tasks with four annotators each -- natural language inference and paraphrase judgment -- we first analyze whether annotators exhibit stable individual patterns. We find that such patterns are weak at the single-annotation level due to strong input-content effects, but become detectable after input-content reduction and annotator-level aggregation. We then compare prompting and supervised fine-tuning (SFT) baselines and propose cross-annotator preference optimization (CAPO), which contrasts a target annotator's response with other valid but less target-specific annotations for the same input. Experiments show that prompting is limited and unstable, SFT better captures annotator-specific behavior, and CAPO further improves aggregation-aware imitation and judge-based attribution while preserving target-specific reasoning patterns under human validation. Overall, our results show that HLV can be learned as annotator-specific label-explanation behavior, suggesting a path toward scalable explanation-based annotation grounded in annotator histories rather than labels alone.
>
---
#### [new 013] Stance Detection in Prediction Markets: Addressing Imbalanced Trader Commentary via Counterfactual Augmentation and Market Context
- **分类: cs.CL**

- **简介: 该论文属于立场检测任务，解决预测市场评论中类别不平衡问题，通过反事实增强和市场上下文提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.28745](https://arxiv.org/pdf/2605.28745)**

> **作者:** Thomas Mbrice
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** Prediction markets such as Polymarket aggregate crowd beliefs into real-time probability estimates, and the comments traders post beneath each market contain rich directional stance signals that prices alone cannot capture. This work introduces the first stance detection study applied to prediction market commentary, a domain characterized by extreme brevity, trader- specific vernacular, and severe class imbalance (only 8.7% of comments oppose the market outcome). RoBERTa-base is fine-tuned across a 4 x 3 ablation: four input configurations ({2- class, 3-class} x {with/without market context}) and three augmentation conditions (baseline, 50% synthetic, 100% synthetic). Synthetic minority-class samples are generated via LLM-driven Pro -> Anti counterfactual flips using the Anthropic API. Results show that (1) market context is the single most impactful factor, raising 3-class Anti recall from 0.10 to 0.45; (2) counterfactual augmentation is conditionally effective, improving Anti F1 in weak configurations (0.10 -> 0.24) while degrading strong ones (2-class-ctx macro F1: 0.68 -> 0.50 at full dose); and (3) 50% augmentation is the optimal dose, with 100% consistently hurting performance. Attention-based interpretability analysis provides mechanistic support for all three findings.
>
---
#### [new 014] Functional Entropy: Predicting Functional Correctness in LLM-Generated Code with Uncertainty Quantification
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于代码生成任务，旨在解决LLM生成代码的功能错误问题。通过评估不同不确定性量化方法，提出基于功能等价的评估方法，提升代码正确性预测效果。**

- **链接: [https://arxiv.org/pdf/2605.28500](https://arxiv.org/pdf/2605.28500)**

> **作者:** Dylan Bouchard; Mohit Singh Chauhan; Zeya Ahmad; Ho-Kyeong Ra
>
> **摘要:** Large language models have shown impressive capabilities in code generation, yet they often produce functionally incorrect code. Uncertainty quantification (UQ) methods have emerged as a promising approach for detecting hallucinations in natural language generation, but their effectiveness for code generation tasks remains underexplored. We systematically evaluate how UQ techniques transfer to code generation across three programming languages, five LLMs, and over 1,700 problems. We find that some token-probability-based methods generalize effectively without modification, while sampling-based methods relying on natural language inference (NLI) fail because NLI models cannot distinguish functionally different code, causing most responses to collapse into a single semantic cluster. To address this, we introduce functional equivalence methods, a family of code-specific methods that replace NLI-based semantic equivalence with an LLM-based functional equivalence assessment, including functional entropy, a code-specific analog of semantic entropy. Functional equivalence methods achieve top AUROC in 11 out of 15 model-benchmark combinations and the best calibration across most settings, consistently outperforming both NLI-based counterparts and all other methods evaluated.
>
---
#### [new 015] Personality, Role, and Expressive Style in Large Language Models: An Interactionist Analysis
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究如何通过提示控制大语言模型的性格表达。旨在解决性格特质与生成对话不一致的问题，通过实验分析性格、角色和表达风格的影响。**

- **链接: [https://arxiv.org/pdf/2605.28037](https://arxiv.org/pdf/2605.28037)**

> **作者:** Moe Nagao; Koichiro Terao; Mikio Nakano; Naoto Iwahashi
>
> **备注:** 26 pages
>
> **摘要:** Prompt-based personality control is a key technique for designing large language model (LLM) dialogue agents that behave consistently across social contexts. However, specifying Big Five personality traits (BFTs) in a prompt does not ensure that the intended traits are expressed in generated utterances. This paper investigates this mismatch from an interactionist perspective, viewing personality expression as a context-dependent outcome shaped by the interplay between trait specification and situational factors. We analyze how perceived BFT expression in LLM-generated dialogue is influenced by three prompt factors: personality traits, dialogue roles, and expressive styles. Using a factorial design that combines six personality conditions, three roles, and three expressive-style conditions, we generate 1,080 LLM-agent dialogues in each of English and Japanese. We then evaluate the target agent's utterances using an LLM-as-a-judge framework to estimate expressed Big Five traits. The results show that expressed personality is shaped not only by explicit trait specification, but also by dialogue role and expressive style. These effects are trait-specific: dialogue role strongly influences Openness, expressive style substantially shapes Conscientiousness and Agreeableness, and explicit trait specification dominates Neuroticism. Even without explicit personality-trait specification, social and expressive conditions induce distinct personality-like impressions. Cross-linguistic comparisons show broadly similar patterns between English and Japanese dialogues, with noticeable differences only under specific combinations of personality, role, and expressive style. These findings suggest that personality control in LLM agents should be understood not as a direct consequence of trait prompting, but as a context-dependent process involving personality specification, social role, and expressive style.
>
---
#### [new 016] ROSD: Reflective On-Policy Self-Distillation for Language Model Reasoning across Domains
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型推理任务，解决LLM在领域内和跨领域推理中的性能提升问题。提出ROSD框架，通过反思引导的局部纠错提升推理效果。**

- **链接: [https://arxiv.org/pdf/2605.28014](https://arxiv.org/pdf/2605.28014)**

> **作者:** Ziqi Zhao; Xinyu Ma; Liu Yang; Yujie Feng; Daiting Shi; Jingzhou He; Xin Xin; Zhaochun Ren; Xiao-Ming Wu
>
> **备注:** Preprint
>
> **摘要:** On-policy self-distillation (OPSD) improves the reasoning performance of large language models (LLMs) by providing dense token-level supervision for on-policy rollouts. However, existing OPSD methods often yield limited gains on in-domain reasoning and generalize poorly to out-of-domain problems. We identify two key causes: conditioning the self-teacher on a verified solution encourages imitation of training-domain reference trajectories rather than error-specific correction, and applying distillation to the full response can overwrite valid reasoning prefixes and reinforce overfitting. We propose Reflective On-policy Self-Distillation (ROSD), a framework that turns reference-solution imitation into targeted reasoning correction through reflection-guided, error-localized distillation. For each rollout, ROSD uses a self-reflector to extract a corrective idea and locate the first erroneous span. The corrective idea guides the self-teacher toward targeted supervision, while the localized error span restricts distillation to where correction is needed. This design corrects flawed reasoning while preserving valid prefixes. Experiments on multiple in-domain and out-of-domain reasoning benchmarks show that ROSD yields stronger in-domain reasoning performance overall and substantially better out-of-domain generalization than standard OPSD. Code is available at this https URL.
>
---
#### [new 017] Self-Improving Language Models with Bidirectional Evolutionary Search
- **分类: cs.CL**

- **简介: 该论文属于语言模型优化任务，旨在解决传统搜索方法在自提升语言模型中的局限性。提出双向进化搜索（BES），通过正向演化和反向目标分解提升搜索效率与效果。**

- **链接: [https://arxiv.org/pdf/2605.28814](https://arxiv.org/pdf/2605.28814)**

> **作者:** Guowei Xu; Zhenting Qi; Huangyuan Su; Weirui Ye; Himabindu Lakkaraju; Sham M. Kakade; Yilun Du
>
> **摘要:** Search has been proposed as an effective method for self-improving language models and agentic systems, both for post-training sample generation and for inference. However, widely used methods such as best-of-N sampling and tree search face two fundamental limitations: they are guided by sparse verification signals, and they construct candidates primarily through autoregressive expansion, restricting exploration to regions with substantial model probability mass. To address these, we propose Bidirectional Evolutionary Search (BES), a search framework that couples forward candidate evolution with backward goal decomposition. In the forward search, BES augments standard expansion with evolution operators that recombine partial trajectories to generate candidates that are difficult to obtain from a single model rollout. In the backward search, BES recursively decomposes the original task into checkable subgoals, producing dense intermediate feedback that guides forward search. We provide theoretical motivation showing that candidates generated by expansion-only search are confined to a narrow entropy shell while evolutionary operators can escape it, and that backward search can exponentially reduce the number of required samples to find a correct answer. Experiments show that on challenging post-training tasks where mainstream post-training algorithms fail to improve, BES enables consistent gains, and on three open problem solving benchmarks at inference time, BES outperforms existing open-source frameworks in both average and best-case performance. Code and trained models are available at this https URL.
>
---
#### [new 018] The Missing Piece in Pre-trained Model Evaluation: Reward-Guided Decoding Unlocks Task-Oriented Behavior Without Parameter Updates
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，解决预训练模型在解码时行为不任务导向的问题。提出EBD框架，通过奖励引导解码，提升模型指令遵循能力，无需参数更新。**

- **链接: [https://arxiv.org/pdf/2605.28020](https://arxiv.org/pdf/2605.28020)**

> **作者:** Shaobo Wang; Guo Chen; Ziyue Wang; Zhengyang Tang; Qingyang Liu; Xingzhang Ren; Dayiheng Liu; Linfeng Zhang
>
> **备注:** 26 pages, 5 figures, 8 tables
>
> **摘要:** With the rapid progress of large language models (LLMs), reliably evaluating the capabilities of pre-trained LLMs has become increasingly important. The challenge is that base pre-trained models are optimized for next-token prediction and often fail to follow instructions or produce well-formed answers under standard prompting and direct decoding. As a result, benchmark performance can conflate model capability with decoding-induced failures to produce task-oriented outputs, while exposing such behavior often relies on costly post-training. Recent decodingonly approaches attempt to reshape output distributions, but such methods can be inefficient and brittle across open-ended tasks. To address these limitations, we propose Energy-Based Decoding (EBD), a training-free, reward-guided framework for activating task-oriented behaviors from frozen pre-trained LLMs across both open-ended and objective tasks. EBD augments decoding with an external lightweight reward model, steering generations toward high-utility responses while anchoring them to the pre-trained model prior through a reward-tilted target distribution. We show that EBD shifts base-model outputs toward more instructionfollowing behavior, increasing behavioral similarity to post-trained counterparts and enabling a fairer inference-time evaluation of accessible pre-trained-model behavior. Empirically, EBD outperforms baselines across five models and six benchmarks, improving Qwen3-8B-Base on AlpacaEval2.0 from 8.8 to 44.5, reducing Mistral-7B Math500 latency by 18.9x relative to prior decoding work, and remaining robust to reward-model size.
>
---
#### [new 019] The Abstraction Gap in Vision-Language Causal Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉-语言因果推理任务，旨在解决评估中难以区分语言流畅性与真实因果推理的问题。通过提出双探针方法和CAGE基准，量化并分析了模型的抽象差距。**

- **链接: [https://arxiv.org/pdf/2605.28779](https://arxiv.org/pdf/2605.28779)**

> **作者:** Chinh Hoang; Mohammad Rashedul Hasan
>
> **摘要:** Vision-language models (VLMs) generate fluent causal explanations, but current evaluations cannot distinguish linguistic plausibility from faithful causal reasoning. We introduce a dual-probe methodology that isolates these properties. The Text-Only Probe measures linguistic quality. The Chain-Text Probe requires models to first generate explicit causal chains. The Abstraction Gap (AG) metric quantifies the normalized performance difference. Evaluating eight VLMs on CAGE (Causal Abstraction Gap Evaluation), a benchmark of 49,500 questions across 5,500 images spanning Pearl's causal hierarchy, we find seven models exhibit AG exceeding 0.50 with text scores of 6--8 but chain scores below 2.5. Fine-tuning on 45,000 chain-annotated examples fails to close the gap. However, one model achieves near-zero AG. The capability exists within current VLM architectures and depends on pretraining and architectural choices. CAGE provides a diagnostic tool for assessing faithful causal reasoning in VLMs.
>
---
#### [new 020] Why We Need Speech to Evaluate Speech Translation
- **分类: cs.CL**

- **简介: 该论文属于语音翻译评估任务，旨在解决现有评估指标无法有效衡量语音特有信息的问题。研究提出SpeechCOMET模型，但发现仍存在不足，需更多语音专用数据与模型。**

- **链接: [https://arxiv.org/pdf/2605.28227](https://arxiv.org/pdf/2605.28227)**

> **作者:** Maike Züfle; Danni Liu; Vilém Zouhar; Jan Niehues
>
> **摘要:** Speech translation models are increasingly capable of preserving speech-specific information (e.g., speaker gender, prosody, and emphasis), yet evaluation metrics remain blind to such phenomena. We meta-evaluate both text- and speech-based quality estimation metrics on two contrastive datasets targeting gender agreement and prosody, and find that both fall short, even when given direct access to the speech signal. We then train SpeechCOMET, a family of quality estimation models with speech encoders, and evaluate a state-of-the-art SpeechLLM as a judge. Both match or exceed text-based COMET on standard quality estimation, but neither consistently assesses speech-specific phenomena. We identify three causes: (1) speech-specific features are not reliably preserved in current encoders, (2) models tend to ignore the speech source signal, and (3) quality estimation training data contains too few relevant examples. We release all models and code, and argue that progress requires dedicated speech-specific training data and models that genuinely condition on speech.
>
---
#### [new 021] Chain-based Adaptive Reconfiguration Over Lattices for Hallucination Reduction
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出CAROL框架，用于减少大语言模型的幻觉问题。属于自然语言处理任务，通过语义一致性提升生成结果的可靠性。**

- **链接: [https://arxiv.org/pdf/2605.27706](https://arxiv.org/pdf/2605.27706)**

> **作者:** Joan Vendrell Gallart; Solmaz Kia; Russell Bent; Michael Grosskopf
>
> **摘要:** We introduce CAROL (Chain-based Adaptive Reconfiguration Over Lattices), a probabilistic framework for test-time hallucination reduction in large language models. Rather than relying on token-level uncertainty, CAROL defines a semantic uncertainty measure based on the consistency between generated responses and a trusted context, inducing a string-submodular objective over a lattice of textual sequences. This formulation enables hallucination mitigation to be cast as a Markov chain accept-reject process with provable convergence and near-optimality guarantees, allowing the model to iteratively refine outputs toward semantic consistency. By operating at the level of meaning, CAROL unifies hallucination detection and mitigation within a single framework. Empirical results on question answering and multi-agent reasoning benchmarks show that CAROL significantly reduces hallucinations and improves reliability and interpretability compared to likelihood-based and retrieval-augmented baselines, while maintaining competitive computational efficiency.
>
---
#### [new 022] ConvMemory: A Lightweight Learned Memory Reranker, a Negative Attribution Result, and a Research-Preview Conflict Editor
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出ConvMemory，一种轻量级对话记忆重排序模型，解决长时记忆检索效率与效果问题，通过融合特征和教师监督训练，实现低延迟高召回。**

- **链接: [https://arxiv.org/pdf/2605.28062](https://arxiv.org/pdf/2605.28062)**

> **作者:** Taiheng Pan
>
> **备注:** 15 pages. Technical report
>
> **摘要:** We describe ConvMemory, a small 3.6M-parameter learned reranker for conversational long-term memory retrieval, trained with cross-encoder teacher supervision over fused dense and lexical features. On the LongMemEval memory family, ConvMemory operates above the BGE-large cross-encoder in Recall@10 at 12-47x lower latency, remains within 0.025 Recall@10 of mxbai-rerank-large-v1 on Clean500 while running 28x cheaper; under Stress1000 distractors the Recall@10 gap widens to 0.081 but ConvMemory still operates at 117x lower latency; these LongMemEval numbers are single-run or single-seed and are reported as indicative cost-frontier evidence, not benchmark-grade. We then publish a rigorous negative attribution result on a previously claimed mechanism: a five-seed retrained ablation with paired bootstrap shows that ConvMemory's learned temporal window is statistically significant on aggregate but not temporally specific, with the largest effects on hard non-temporal controls and no significant effect on multi-hop temporal queries. The honest description of the mechanism is cheap cross-encoder distillation in a fused dense+lexical feature space, not temporal-structure exploitation. We additionally release CCGE-LA, a low-amplitude conflict-aware candidate-set editor over ConvMemory, as a research preview with modest but consistent gains on supersession and stale/rescue slices on LoCoMo. All results are retrieval-stage; ConvMemory does not match mxbai-rerank-large-v1 in absolute LoCoMo MRR, and the report is single-author and not yet independently audited.
>
---
#### [new 023] When Helpful Context Leaks: Privacy Risks in Domain-Adapted ASR
- **分类: cs.CL**

- **简介: 该论文属于语音识别领域，研究领域自适应ASR中的隐私泄露问题。工作包括发现并评估上下文泄露风险，测试两种定制方法的泄露率，提出缓解策略。**

- **链接: [https://arxiv.org/pdf/2605.28211](https://arxiv.org/pdf/2605.28211)**

> **作者:** Maike Züfle; Jan Niehues
>
> **摘要:** SpeechLLMs are increasingly deployed in professional settings where domain customisation is standard practice: users supply context in prompts with sensitive information, fine-tune on proprietary recordings, or both. We identify and systematically investigate an overlooked privacy risk of such customisation: a model adapted to recognise domain-specific terminology can be nudged into transcribing a phonetically similar word from its context or training data, even when a different word is spoken, thereby leaking private information. To evaluate this risk, we construct a controlled dataset and measure leakage rates across two customisation mechanisms, prompting and fine-tuning. Both mechanisms cause measurable leakage, compounding when combined. We evaluate a prompt-level mitigation strategy and analyse the accuracy-leakage trade-off across customisation approaches, finding that fine-tuning without context prompts offers the best balance. We release our code and dataset publicly.
>
---
#### [new 024] KVoiceBench, KOpenAudioBench, and KMMAU: Agent-Driven Korean Speech Benchmarks for Evaluating SpeechLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音语言模型评估任务，旨在解决多语言语音评估不足的问题。通过构建韩国语音基准，评估SpeechLMs的跨语言能力。**

- **链接: [https://arxiv.org/pdf/2605.27984](https://arxiv.org/pdf/2605.27984)**

> **作者:** Haechan Kim; Seungjun Chung; Inkyu Park; Jihoo Lee; Jonghyun Lee
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** Speech language models (SpeechLMs) have achieved substantial progress by extending large language models (LLMs) to the speech modality. However, SpeechLM evaluation remains heavily centered on English, limiting reliable assessment of multilingual speech capabilities. Straightforward benchmark transfer through ASR, translation, normalization, and TTS can corrupt language-specific instructions, answer constraints, and spoken forms; for audio understanding, transferring source-language audio also fails to preserve target-language speaker attributes, accents, and paralinguistic properties. To address these limitations, we propose two human-agent benchmark-construction frameworks: one transfers source-language SpokenQA benchmarks into target-language SpokenQA benchmarks, and the other converts target-language ASR corpora into audio understanding benchmarks using transcriptions and speaker metadata. Using these frameworks, we construct and publicly release three Korean speech benchmarks: KVoiceBench and KOpenAudioBench for Korean SpokenQA, and KMMAU for Korean audio understanding, comprising 12,345 samples in total. We evaluate eight recent SpeechLMs and find that English-Korean performance gaps vary substantially across models and task families, and that SpokenQA and audio understanding rankings diverge, revealing complementary weaknesses invisible to English-only evaluation.
>
---
#### [new 025] Towards Reliable Multilingual LLMs-as-a-Judge: An Empirical Study
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言文本评估任务，旨在解决LLM在多语言场景下的可靠性问题。通过分析不同语言和数据条件下的模型表现，提出有效评估策略。**

- **链接: [https://arxiv.org/pdf/2605.28710](https://arxiv.org/pdf/2605.28710)**

> **作者:** Irune Zubiaga; Aitor Soroa; Rodrigo Agerri
>
> **摘要:** Large language models (LLMs) are increasingly used for the automatic evaluation of generated text, yet most prior work focuses on English. Despite the growing demand for multilingual evaluation, extending LLM-based evaluators to multilingual settings remains challenging, particularly for low-resource languages and scenarios where in-domain data is scarce. This work explores several strategies for developing multilingual LLMs-as-a-judge, considering whether in-domain data is available for fine-tuning or not. We systematically analyze English, Spanish, and Basque, representing high-, mid-, and low-resource languages, considering instruction translation, monolingual versus multilingual supervision, and model size. For evaluation, we extend two existing meta-evaluation datasets to Basque and Spanish. Our results reveal key trade-offs: When in-domain data is available, fine-tuned smaller models can achieve performance comparable to proprietary models, whereas zero-shot evaluation with larger models proves more effective in out-of-domain settings. We also observe that fine-tuning on out-of-domain data can adversely affect model performance. These findings provide practical guidance for building efficient, reliable multilingual evaluation pipelines. The data and code are publicly available at hitz-zentroa/mJudge.
>
---
#### [new 026] UserHarness: Harnessing User Minds for Stronger Agent Theory-of-Mind
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出UserHarness，用于构建智能代理对用户心理状态的准确理解。任务属于理论心智（ToM）研究，解决现有方法间接建模用户心理的问题，通过显式重构用户心智提升代理的适应性与准确性。**

- **链接: [https://arxiv.org/pdf/2605.27721](https://arxiv.org/pdf/2605.27721)**

> **作者:** Cheng Qian; Jiayu Liu; Heng Ji
>
> **备注:** 19 Pages, 4 Figures, 2 Tables
>
> **摘要:** Understanding what a user believes and intends is central to building effective agent assistants. This ability is often evaluated through Theory-of-Mind (ToM) tasks, where success requires reasoning from the user's perspective. However, many existing approaches address ToM with complex pipelines that model behavior indirectly, without explicitly reconstructing the user's mental state. This misses the core structure of the problem: users act based on their beliefs, which are updated through observations of the environment; beliefs and intentions jointly determine actions, which in turn change the environment; and social reasoning often requires nested beliefs about what others believe or intend. We propose UserHarness, a simple framework that reframes ToM reasoning as explicit user-mind reconstruction. UserHarness decomposes the user's mental state, its relation to the external environment, and the actions that follow from it, enabling agents to track what the user observes, believes, intends, and does. Across five benchmarks, UserHarness reaches up to 95.94% macro accuracy, improving over existing inference methods by more than 15% relative and over the strongest prompt-only harness by about 20% relative. These results suggest that robust user understanding requires reasoning from the roots of the user's mind, positioning user harnessing as a promising foundation for more adaptive future assistants.
>
---
#### [new 027] Analyzing Quality-Latency-Resource Trade-offs in a Technical Documentation RAG Assistant Using LoRA Adaptation
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文研究RAG系统中质量、延迟和资源的权衡问题，通过LoRA适配优化生成器，评估不同配置性能。任务属于模型优化与系统评估。**

- **链接: [https://arxiv.org/pdf/2605.28222](https://arxiv.org/pdf/2605.28222)**

> **作者:** Evgenii Palnikov; Elizaveta Gavrilova
>
> **备注:** 13-page main body plus extended appendix; 6 figures; benchmark, LoRA adapters, and code at this https URL
>
> **摘要:** We study quality-latency-resource trade-offs in a documentation-grounded retrieval-augmented generation (RAG) system that uses Low-Rank Adaptation (LoRA) of the generator. We build a manually verified benchmark of 5,144 question-answer pairs over the official Kubernetes documentation and combine it with a fixed hybrid-retrieval pipeline (BGE-M3 dense, BGE-M3 native sparse, Reciprocal Rank Fusion, cross-encoder reranking). Over this benchmark we ablate 20 LoRA configurations on Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct across rank and target-module choices, and evaluate each on token-level F1, LLM-judged groundedness and correctness (pass@4), inference latency, inference memory, and training cost, all reported with bootstrap 95% confidence intervals. Pareto analysis shows that LoRA adapters acting only on the q and v attention projections consistently dominate the front, while the 3B/8B choice mainly defines operating regime. A param-matched control comparison further indicates that the q/v advantage is structural rather than purely parametric. The benchmark, selected adapters, and code are available at this https URL.
>
---
#### [new 028] Skill0.5: Joint Skill Internalization and Utilization for Out-of-Distribution Generalization in Agentic Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于强化学习领域，解决技能内部化与外部化的平衡问题。提出Skill0.5框架，结合通用技能内部化和任务技能利用，提升模型在分布外场景的泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.28424](https://arxiv.org/pdf/2605.28424)**

> **作者:** Jiapeng Zhu; Jianxiang Yu; Yibo Zhao; Chengcheng Han; Qi Gu; Xunliang Cai; Xiang Li; Weining Qian
>
> **摘要:** Equipping large language models with explicit skills has emerged as a promising paradigm for enabling autonomous agents to solve complex tasks. Agent skills can be inherently divided into general skills for broad cognitive transfer and task-specific skills for dynamic execution. However, existing skill-based reinforcement learning (RL) methods typically force a rigid choice between full externalization, which incurs prohibitive context overhead, and full internalization, which risks overfitting and knowledge conflicts. To address this dilemma, we propose Skill0.5, a novel agentic RL framework that explicitly differentiates skill treatments by combining general skill internalization with task-specific skill utilization. Driven by a dynamic, difficulty-aware router, Skill0.5 streams tasks into distinct mastery tiers to apply tailored optimization strategies: it internalizes general skills via privileged distillation to build a cognitive foundation for hard tasks, while using diagnostic probing on easy tasks to penalize shortcuts and enforce specific skill utilization. Experiments on ALFWorld and WebShop demonstrate that Skill0.5 outperforms both memory-based and skill-based RL baselines, yielding performance improvements across both in-distribution and out-of-distribution scenarios.
>
---
#### [new 029] UniMaia: Steering Chess Policies with Language for Human-like Play
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于棋类策略控制任务，旨在通过自然语言实现对棋类策略的语义控制。工作包括提出UniMaia框架，结合文本编码与条件机制，提升策略灵活性与准确性。**

- **链接: [https://arxiv.org/pdf/2605.27767](https://arxiv.org/pdf/2605.27767)**

> **作者:** Sherman Siu; Lesley Istead
>
> **摘要:** Recent advances in large language models have enabled natural language to serve as a flexible interface for controlling complex systems, but often at the cost of large-scale multimodal training or weakened domain-specific inductive biases. In structured decision-making domains such as chess, specialized policy networks achieve strong performance but lack semantic controllability, while prompt-conditioned language models are more flexible yet typically exhibit weaker domain grounding. We propose $\textbf{UniMaia}$, a framework for prompt-conditioned policy modulation that adapts a frozen Lc0-based chess policy network using a parameter-efficient text encoder and a ControlNet-style conditioning mechanism. UniMaia enables semantic control over gameplay, including opening selection and player strength, while preserving the pretrained policy representations. We further introduce $\textbf{UniMaia-Aux}$, which incorporates auxiliary temporal conditioning and behavioral prediction objectives. To support this work, we construct a large-scale metadata-augmented Lichess dataset, develop a semi-automated prompt-generation pipeline, and introduce benchmarks spanning both prompt-conditioned and metadata-conditioned settings. UniMaia achieves state-of-the-art expected accuracy on several prompt-conditioned benchmarks and competitive top-move accuracy on general instruction-following tasks, while remaining competitive with dedicated metadata-conditioned approaches on human move prediction benchmarks. UniMaia-Aux further improves expected accuracy and behavioral modeling across several evaluation settings, with modest trade-offs in top-move accuracy. Overall, our results demonstrate that prompt-conditioned control of domain-specific policy networks is feasible without end-to-end multimodal training, while highlighting trade-offs between controllability and predictive performance.
>
---
#### [new 030] ATLAS: All-round Testing of Long-context Abilities across Scales
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决长文本评估不全面的问题。提出ATLAS框架，通过多维度、长度依赖的评估方法，更准确地衡量模型在不同上下文长度下的能力表现。**

- **链接: [https://arxiv.org/pdf/2605.28079](https://arxiv.org/pdf/2605.28079)**

> **作者:** Deli Huang; Cunguang Wang; Hongyin Tang; Zhe Tang; Linsen Guo; Dongyu Ru; Ruoshi Yuan; Ziyue Zhu; Xiaoyu Li; Ziwen Wang; Chen Zhang; Anchun Gui; Wen Zan; Jiaqi Zhang; Xuezhi Cao; Jingang Wang; Xunliang Cai; Yixin Cao
>
> **备注:** 29 pages, 13 figures. Preprint
>
> **摘要:** Long-context language models now advertise context windows up to millions of tokens, yet evaluations typically report a single length or a narrow task family, masking two failure modes: performance can collapse as length grows, and strong retrieval need not transfer to downstream use. We present ATLAS, a benchmarking framework that redefines long-context evaluation as length-dependent capability profiling. ATLAS contributes three methodological principles:(i) a layered taxonomy separating foundational operations from application workloads so failures can be attributed, (ii) length-aware AUC scoring that integrates score-length curves over a fixed 8K-1M grid, replacing single-point metrics with full degradation profiles, and (iii) ATLAScore, a harmonic-mean aggregate over taxonomy categories that penalizes imbalanced profiles, with end-to-end uncertainty propagation from subset scores through the nonlinear final aggregate. We instantiate the framework across eight capability dimensions with nine auditable components and 6,438 instances, and evaluate 26 models. Gemini-3.1-Pro-Preview leads at 128K, Claude-Opus-4.6 leads at 1M. Rankings reshuffle substantially between ATLASscore@8K-128K and ATLASscore@8K-1M: 7 models move by at least two ranks, and the two taxonomy layers share only 61% of cross-model variance, with individual rank gaps up to 12 positions. These results support reporting long-context quality by capability and length, not by a single headline score.
>
---
#### [new 031] Rethinking Visual Neglect: Steering via Context-Preference for MLLM Hallucination Mitigation
- **分类: cs.CL**

- **简介: 该论文属于多模态大模型任务，解决对象幻觉问题。通过提出CAS框架，控制模型对视觉和文本的依赖，有效减少幻觉。**

- **链接: [https://arxiv.org/pdf/2605.27993](https://arxiv.org/pdf/2605.27993)**

> **作者:** Jingwen Wu; Xijun Zhang; Ge Song
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Object hallucination remains a primary obstacle to the reliable deployment of Multimodal Large Language Models (MLLMs). Current inference-time mitigation methods mainly assume hallucinations stem from visual neglect, steering models to enhance visual reliance. In contrast, our systematic interventions on multiple MLLMs show that pushing toward more visual reliance may exacerbate hallucinations on some models, while less may mitigate hallucinations. This result suggests that attributing hallucinations solely to visual insufficiency is underdetermined. We argue that the image, as a context, simultaneously competes with the model's parametric knowledge and the textual context. For this, we propose a training-free framework, Context-Preference Activation Steering (CAS). It extracts two semantically distinct Context Preference Vectors (CPVs) via two small sets of designed conflict samples and applies them via single-pass signed residual injection at mid-early MLP layers during inference to control information reliance. Experiments show that CAS substantially mitigates object hallucinations without increasing decoding latency and preserves native text-generation quality.
>
---
#### [new 032] AdaDPO: Self-Adaptive Direct Preference Optimization with Balanced Gradient Updates
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出AdaDPO，解决DPO梯度不平衡问题，提升模型生成优质回答能力。属于大模型对齐任务。**

- **链接: [https://arxiv.org/pdf/2605.28440](https://arxiv.org/pdf/2605.28440)**

> **作者:** Shaolong Chen; Madalina Ciobanu; Qingqing Mao; Ritankar Das
>
> **备注:** 5 figures
>
> **摘要:** DPO has become a widely adopted alternative to RLHF for aligning LLMs with human preferences, eliminating the need for a separate reward model or RL loop. Recent theoretical analysis uncovers an asymmetric gradient behavior in DPO: the loss suppresses dispreferred responses substantially faster than it promotes preferred ones, causing the model to learn to avoid bad answers rather than to generate good ones. We propose AdaDPO, a Self-Adaptive variant of the DPO algorithm that introduces per-preference-pair, stop-gradient-based coefficients derived directly from the policy model's generation probabilities, with the reference model's probabilities as an optional component. AdaDPO is constructed to enforce equality of gradient magnitudes between preferred and dispreferred probabilities; the practical implementation balances per-token gradients and applies a numerical clipping bound for stability, while retaining DPO's original hyperparameter structure. On Llama-3-8B-Instruct trained on UltraFeedback under a SimPO similar setup, AdaDPO consistently outperforms DPO on AlpacaEval 2: it achieves higher length-controlled win rates (LC) in 81% of hyperparameter combinations, attains the global best LC (48.3%) and raw win rate (46.1%), and enlarges the LC-over-WR margin in 88% of combinations, indicating effective mitigation of length bias. Additional analyses on KL divergence, reward margin, and reward accuracy confirm that AdaDPO rectifies the gradient imbalance and yields more efficient optimization. Because it operates purely at the loss level, AdaDPO can be dropped into existing preference-based alignment pipelines without changing data collection or model architectures. The method requires only a few lines of code, and the same self-adaptive principle generalizes to a broad family of pairwise contrastive preference losses including SimPO, R-DPO, IPO, CPO, and ORPO.
>
---
#### [new 033] Routing-Aligned Fine-Tuning for Multilingual Downstream Tasks in Mixture-of-Experts Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多语言下游任务中的Mixture-of-Experts模型微调问题，提出RA-MoE方法，通过路由对齐提升性能。**

- **链接: [https://arxiv.org/pdf/2605.28306](https://arxiv.org/pdf/2605.28306)**

> **作者:** Guanzhi Deng; Kuan Wu; Haibo Wang; Shing Yin Wong; Sichun Luo; Linqi Song
>
> **摘要:** Mixture-of-Experts (MoE) models have emerged as a dominant paradigm for efficient LLM scaling, yet adapting them to non-English downstream tasks remains challenging. Existing fine-tuning approaches treat MoE models as monolithic learners, ignoring the heterogeneous routing structure that develops during pretraining. We validate across multiple MoE models and downstream tasks that middle layers form a language-universal alignment zone where routing divergence strongly predicts per-language task performance gaps. Building on this observation, we propose RA-MoE (Routing-Aligned MoE Fine-Tuning), a three-stage framework that categorizes parallel task examples into a four-way taxonomy (cc/ci/ic/ii) based on correctness in English and the target language, identifies task-relevant experts in the middle layers, and augments standard SFT with a routing alignment loss that encourages target-language routing on ci-type examples to follow the English task-expert activation pattern. Experiments across three MoE models, three tasks, and six target languages demonstrate that RA-MoE consistently outperforms standard SFT and strong baselines including Routing Steering and RISE, with the ci proportion of a task-language pair serving as a reliable predictor of alignment benefit.
>
---
#### [new 034] Where Does Toxicity Live? Mechanistic Localization and Targeted Suppression in Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型安全任务，旨在解决毒性内容生成问题。通过分析激活差异定位毒性来源，并在推理时抑制，无需重新训练。**

- **链接: [https://arxiv.org/pdf/2605.27997](https://arxiv.org/pdf/2605.27997)**

> **作者:** Himanshu Beniwal; Mayank Singh
>
> **摘要:** Large language models frequently generate toxic, hateful, or harmful content, yet existing mitigation methods rely on costly retraining or output-level filtering with no mechanistic insight into where toxicity originates internally. We introduce Meow2X and TRNE, two complementary retraining-free frameworks that localize toxicity to specific layers and neurons by analyzing activation differentials between toxic and neutral prompts, then suppress them via inference-time scaling or minimal rank-one weight edits -- without any gradient descent. Evaluations across five LMs, two benchmarks, and 90 configurations using dual safety evaluators demonstrate consistent toxicity reduction while preserving language modeling quality. Our analysis reveals that toxicity is disproportionately encoded in early MLP layers, varies across architectures, and is systematically underestimated by single-evaluator setups -- underscoring the need for multi-evaluator safety assessment. By bridging mechanistic interpretability with practical detoxification, our framework offers a principled path toward safer, more transparent language models.
>
---
#### [new 035] Can LLMs Use Linguistic Uncertainty Markers to Reliably Reflect Intrinsic Confidence?
- **分类: cs.CL**

- **简介: 该论文研究LLMs是否能通过语言不确定性标记准确反映内在置信度，属于模型校准任务。旨在解决标记与置信度关联不稳定的问题，通过7项指标分析不同模型和任务中的表现。**

- **链接: [https://arxiv.org/pdf/2605.28778](https://arxiv.org/pdf/2605.28778)**

> **作者:** Gabrielle Kaili-May Liu; Arman Cohan
>
> **备注:** Code: this https URL
>
> **摘要:** LLMs' linguistically expressed confidence should faithfully reflect their intrinsic uncertainty. While recent work shows LLMs struggle to use epistemic markers (e.g., "it is likely...") in a human-aligned fashion, it remains unclear whether models can apply their own linguistic confidence framework to associate markers with specific confidence levels in a stable and generalizable way, and how contextual features impact this ability. We conduct the first systematic study of this question, formalizing _marker internal confidence_ (MIC) as the estimated intrinsic confidence a model associates with a specific epistemic marker in a given task domain. We present 7 metrics to evaluate the stability of MICs within and across distributions. Applying our analysis framework to diverse models and tasks, we find that LLMs remain faithfully miscalibrated even under model-centric interpretation of marker meanings, struggling to differentiate markers by internal confidence across distributions despite preserving a somewhat consistent ranking order across tasks. This supplies critical, complementary evidence to existing work toward a holistic understanding of faithful calibration in LLMs, emphasizing the need for more aligned and stable marker use to improve trustworthiness and reliability.
>
---
#### [new 036] PubMedCausal: A Span-Level Annotated Corpus for Causal Relation Extraction in Biomedical Text
- **分类: cs.CL**

- **简介: 该论文属于生物医学因果关系抽取任务，旨在解决现有数据标注不准确、范围有限的问题。构建了PubMedCausal语料库，支持段落级标注与模型评估。**

- **链接: [https://arxiv.org/pdf/2605.28363](https://arxiv.org/pdf/2605.28363)**

> **作者:** Ifeoluwa Kunle-John; Josiah Paul; Oluwatosin Agbaakin; Peter Aina; Ikenna Odezuligbo; Sydney Anuyah
>
> **备注:** Submitted to EMNLP 2026, 8 Pages, 23 page appendix
>
> **摘要:** Causal relation extraction (CRE) is central to biomedical text mining, but current resources often conflate causal relations with broader associations, restrict annotation to sentence-level examples, or focus mainly on explicit causal cues. This limits their usefulness for evaluating whether models can recover causal claims as they are actually expressed in biomedical text. We introduce PubMedCausal, a span-level annotated corpus for biomedical CRE built from PubMed abstracts. The corpus contains 30,000 paragraph-level rows, including 3,945 causal rows and 6,491 adjudicated cause--effect pairs. Each causal relation is annotated with full-text cause and effect spans, causality type, and sententiality, enabling evaluation of both causal detection and full-span causal extraction. We benchmark discriminative encoders and open-source generative models across detection and extraction settings. For causal detection, biomedical encoders are strongest, with PubMedBERT reaching an F$_1$ score of 0.7391. For span-level extraction, the best generative baseline is DeepSeek-R1-32B with few-shot prompting, reaching a Cosine Pair F$_1$ of 0.6765. We further test transfer learning by evaluating PubMedCausal-trained encoders on external causal relation datasets, showing that the resource supports cross-dataset evaluation. Our results show that biomedical CRE remains difficult under class imbalance, long causal spans, implicit causality, inter-sentential relations, and prompt sensitivity. Code and Data can be found here: this https URL
>
---
#### [new 037] When Discourse Pressures Conflict: Information Structure in Vision-Language Model Outputs
- **分类: cs.CL**

- **简介: 该论文属于视觉语言模型研究，探讨模型在对话中是否正确处理信息结构。通过对比模型与人类，发现模型在表达话题和焦点时过于规整，缺乏多样性。**

- **链接: [https://arxiv.org/pdf/2605.28346](https://arxiv.org/pdf/2605.28346)**

> **作者:** Marcell Fekete; Johannes Bjerva; Tamás Káldi
>
> **摘要:** Vision-language models (VLMs) are increasingly evaluated for whether they identify the right visual content, but little is known about whether they express such content in a discourse-appropriate form. We address this research gap using information structure (IS), testing whether VLMs distinguish discourse-old Topics from discourse-new Foci in visually grounded question answering. We exploit Hungarian, a language in which Topic and Focus map onto dedicated syntactic positions, making IS choices observable in text. Comparing six VLMs with human participants, we find that models produce IS-relevant constructions, but over-regularise this sensitivity. Under the interacting pressures of discourse status, grammatical role (preference for subject Topics) and definiteness (preference for indefinite Foci), humans choose variable strategies for IS realisation. VLMs, by contrast, collapse onto narrow response templates, resembling mode collapse (Kirk et al., 2024). Our findings suggest that VLM evaluation should look beyond content accuracy to how content is packaged for the discourse.
>
---
#### [new 038] Challenges in Explaining Pretrained Clinical Text Classifiers
- **分类: cs.CL**

- **简介: 论文探讨了临床文本分类器的解释难题，针对长且非结构化的医学文本，分析了现有解释方法的不足，提出了需要更临床相关、语义稳固的解释策略。**

- **链接: [https://arxiv.org/pdf/2605.28060](https://arxiv.org/pdf/2605.28060)**

> **作者:** Kristian Miok; Matej Klemen; Blaz Škrlj; Marko Robnik Šikonja
>
> **备注:** 9 pages, 7 figures. Accepted at the First Workshop on Responsible Healthcare using Machine Learning (RHCML 2025), co-located with ECML PKDD 2025
>
> **摘要:** Explaining the predictions of neural models in clinical NLP remains a significant challenge, especially for complex tasks involving long, unstructured medical texts. While post-hoc methods like LIME and SHAP are widely used, they often fall short when applied to clinical narratives. In this paper, we identify core limitations of token-level and perturbation-based explanation techniques through targeted demonstra- tions on a hospital length-of-stay prediction task. Our findings reveal issues such as overemphasis on non-informative tokens, instability in at- tributions, and high-confidence predictions for incoherent input variants. These results underscore the need for explanation strategies that are clin- ically meaningful, semantically grounded, and robust to linguistic noise.
>
---
#### [new 039] Simorgh at SemEval-2026 task 7: Region-Aware Hybrid Retrieval for Low-Resource Cultural Reasoning in Multilingual Question Answering
- **分类: cs.CL**

- **简介: 该论文属于多语言问答任务，旨在解决低资源语言文化相关问题回答中的知识不足问题。提出一种结合词法和语义匹配的区域感知混合检索方法，提升答案相关性。**

- **链接: [https://arxiv.org/pdf/2605.27636](https://arxiv.org/pdf/2605.27636)**

> **作者:** Hadi Bayrami Asl Tekanlou; Mahdi Bakhtiyarzadeh; Jafar Razmara
>
> **备注:** 6 pages, 3 figures, accepted to the Everyday Knowledge Across Diverse Languages and Cultures shared task at SemEval2026
>
> **摘要:** Although Large Language Models (LLMs) demonstrate excellent capabilities and performance for general reasoning tasks within the general public domain, they may face challenges with culturally grounded knowledge within languages with limited digital and textual data. In this paper, we investigate culturally grounded multiple-choice question answering with the BLEnD benchmark, which consists of a multilingual corpus of 30 languages and covers various socio-cultural domains, such as cuisine, sports, family, etc. We propose a region-aware hybrid retrieval approach that combines BM25 lexical matching and dense semantic similarity with regional weighting heuristics to improve the relevance of the answer. The retrieved documents are used to construct a structured prompt for the Qwen3-14B quantized model with logit-based deterministic answer selection. The experimental results show improvements to cross-lingual stability with the hybrid retrieval approach over pure parametric inference for culturally grounded question answering. However, there are still notable performance gaps between languages with more and less training data. This shows that the limitations of the retrieval augmentation approach are not entirely overcome by the training data imbalance problem.
>
---
#### [new 040] GeneralThinker: Domain-General Reasoning through Likelihood-Guided Answer-Conditioned Optimization
- **分类: cs.CL**

- **简介: 该论文提出GeneralThinker，解决语言模型推理中依赖领域验证器和稀疏奖励的问题。通过答案引导的优化实现细粒度信用分配，提升推理性能。**

- **链接: [https://arxiv.org/pdf/2605.27934](https://arxiv.org/pdf/2605.27934)**

> **作者:** Shengmin Piao; Sanghyun Park
>
> **摘要:** Reinforcement learning with verifiable rewards improves language model reasoning, but its reliance on domain-specific verifiers, sparse outcome rewards, and coarse-grained credit assignment limits its applicability. We introduce GeneralThinker, an on-policy framework that reformulates reasoning supervision as dense answer-conditioned optimization, enabling response-level evaluation and token-level credit assignment without domain-specific verifiers. GeneralThinker evaluates generated reasoning trajectories using the likelihood of the ground-truth answer and derives token-wise compatibility signals for fine-grained credit assignment. To stabilize optimization, it constrains token-level updates through clipping and direction-preserving modulation. Across 11 benchmarks spanning mathematics, STEM, and general reasoning, GeneralThinker achieves the best average performance. Further analyses show that uncontrolled token-level modulation can destabilize training, whereas controlled modulation makes fine-grained credit assignment consistently effective.
>
---
#### [new 041] Framing Matters: Addressing Framing Sensitivity in Decision-Making through Behaviorally-Grounded Value Alignment
- **分类: cs.CL**

- **简介: 该论文属于人工智能决策领域，解决LLM在不同表述下产生不一致决策的问题。通过构建基准和提出Valign方法，提升模型决策的稳定性。**

- **链接: [https://arxiv.org/pdf/2605.28188](https://arxiv.org/pdf/2605.28188)**

> **作者:** Seojin Hwang; Minju Kim; Junhyuk Choi; JeongHyun Park; Hwanhee Lee
>
> **备注:** 29 pages, 7 figures, 31 tables
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in high-stakes decision-making settings such as legal reasoning, where consistency under factually equivalent inputs is critical. However, we find that fact-preserved but differently framed inputs can significantly destabilize LLM decisions. To systematically investigate this problem, we introduce Fragile, a large-scale benchmark that isolates fact-preserving semantic framing across three controlled dimensions: value-tinted narration, temporal slice, and narrative vividness. Our experiments reveal a high susceptibility of LLMs to framing, with an average decision flip rate of 28.6%. We find that simple prior prompt-level and activation-level interventions not only fail to suppress framing sensitivity but actively amplify it. We therefore propose Valign, a representation-level method that explicitly targets these framing dimensions by anchoring decisions to a stable value prior, steering hidden states toward the model's value-consistent direction, and projecting out temporal-vividness-sensitive directions from the model's hidden states. Valign consistently reduces framing-induced decision flips, demonstrating that robust mitigation requires directly targeting the internal pathways in which framing operates.
>
---
#### [new 042] Beyond pass@k: Redundancy-Aware RLVR for Multi-Sample Code Generation
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于代码生成任务，旨在解决重复采样下的冗余问题。通过引入基于JPlag的反冗余奖励，提升有限预算下的生成性能。**

- **链接: [https://arxiv.org/pdf/2605.28022](https://arxiv.org/pdf/2605.28022)**

> **作者:** Le Bronnec Florian; Alexandre Verine; Rio Yokota; Benjamin Negrevergne
>
> **备注:** Preprint under review
>
> **摘要:** LLMs for code generation are commonly evaluated in repeated-sampling settings using Pass@k, where multiple candidate programs are executed against unit tests under a finite sampling budget. While recent verifier-based reinforcement learning (RLVR) methods improve executable correctness, how these objectives affect redundancy among sampled programs remains poorly understood. In this work, we study implementation-level redundancy in code generation using JPlag, a plagiarism-detection system for code. Across models and benchmarks, we show that correctness-only RLVR often concentrates generations around repeated implementations, whereas Pass@k-aware objectives maintain lower redundancy and improve larger-budget performance. Motivated by these observations, we augment RLVR with direct anti-redundancy rewards based on JPlag similarity. Across 3 models and 3 benchmarks, discouraging near-duplicate generations reliably improves finite-budget executable performance, often matching or outperforming specialized Pass@k-aware objectives.
>
---
#### [new 043] VibeSearchBench: Benchmarking Long-horizon Proactive Search in the Wild
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出VibeSearchBench，用于评估长周期主动搜索任务。解决真实搜索与基准测试间的体验差距问题，通过多轮对话和结构化知识构建进行评估。**

- **链接: [https://arxiv.org/pdf/2605.27882](https://arxiv.org/pdf/2605.27882)**

> **作者:** Xiaohongshu Inc
>
> **摘要:** LLM-based agents score well on search benchmarks, yet real users consistently find results unsatisfying, revealing a persistent evaluation-experience gap. We attribute this gap to existing benchmarks' reliance on over-specified queries, single-turn interactions, and fixed-schema evaluation, none of which reflect real search behavior where users and agents collaboratively refine vague intent through multi-turn dialogue. We term this paradigm VibeSearch and introduce VibeSearchBench, a benchmark comprising 200 manually curated bilingual (Chinese and English) tasks across 20 domains, split into VibeSearch-Pro (professional) and VibeSearch-Daily (daily-life) subsets. Each task pairs a user persona with a schema-free ground-truth knowledge graph, and is evaluated through a progressive-disclosure user simulator and a graph-matching evaluation framework. We benchmark seven frontier models under both the ReAct framework and the OpenClaw agent harness. Results show that all models remain substantially inadequate for VibeSearch (best F1: 30.30), highlighting the need for fundamental advances in long-context reasoning, proactive intent elicitation, and structured knowledge construction.
>
---
#### [new 044] DecomposeRL: Learning to Ask Useful, Informative, and Diverse Questions for Semi-Supervised, Traceable Claim Verification
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出DecomposeRL，解决半监督、可追溯的声明验证问题。通过强化学习框架，在少量标注数据下实现高准确率并生成可解释过程。**

- **链接: [https://arxiv.org/pdf/2605.27858](https://arxiv.org/pdf/2605.27858)**

> **作者:** Shubhashis Roy Dipta; Ankur Padia; Francis Ferraro
>
> **摘要:** Claim verification splits between end-to-end classifiers that are accurate but yields no inspectable traces, and decomposition-based methods produce inspectable traces but lag performance on benchmark datasets. We propose DecomposeRL an accurate claim-verifier that produce inspectable traces. DecomposeRL frames decomposition as an RL policy trained with GRPO and a multi-faceted reward ensemble, enabling both fully supervised and semi-supervised learning from unlabeled claims. DecomposeRL addresses the prohibitive training cost of GRPO with a data-curation funnel that distills 115K fact-verification claims into a compact, learning-signal-dense subset of 5K claims. We show that a DecomposeRL-7B policy trained with full supervision on only ~5K curated claims achieves 86.3 in-domain and 69.8 out-of-domain balanced accuracy across 11 claim-verification benchmarks containing biomedical, political, scientific, and general-domain claims. Despite being 4x smaller, it matches 32B baselines and GPT-4.1-mini, and it further outperforms baselines in a semi-supervised setting with only 10% labeled claims data. Code, data, and models are available at this https URL
>
---
#### [new 045] Reverse Probing: Supervised Token-level Uncertainty Quantification for Large Language Models in Clinical Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床文本摘要任务，旨在解决大语言模型在临床文本中无法准确识别不确定性的难题。提出Reverse Probing方法，通过已有标注摘要估计token级不确定性。**

- **链接: [https://arxiv.org/pdf/2605.28740](https://arxiv.org/pdf/2605.28740)**

> **作者:** Bushi Xiao; Sarvesh Soni; Daisy Zhe Wang
>
> **摘要:** As large language models are increasingly deployed for clinical text, ensuring they can reliably signal their own uncertainty becomes critical. Most existing uncertainty quantification (UQ) methods are designed for open-domain generation and cannot localize uncertainty at the token or span level in long clinical text. We propose Reverse Probing, the first UQ framework specialized for clinical summarization, which estimates token-level uncertainty directly from pre-existing labeled summaries. Rather than sampling new outputs, Reverse Probing treats the text as a probe into the model's internal state, extracting uncertainty signals from four categories of internal activations. We evaluate on two expert-annotated clinical datasets and outperform eight adapted baselines on all metrics, achieving up to 4 times higher AUPRC while reducing inference time and computational costs. Feature analysis reveals that delta energy and neighborhood context are the most consistent predictors across all models. This study offers interpretable insights into how models internally respond to unsupported clinical content.
>
---
#### [new 046] Syllabic-Structure Decoder for Automatic Speech Recognition in Vietnamese
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决传统ASR系统词汇量大、不反映语音结构的问题。提出一种基于音素的音节结构解码器，提升识别效果并减少词汇量。**

- **链接: [https://arxiv.org/pdf/2605.27874](https://arxiv.org/pdf/2605.27874)**

> **作者:** Nghia Hieu Nguyen; Quan Ngoc Hoang; Long Hoang Huu Nguyen; Kiet Van Nguyen; Ngan Luu-Thuy Nguyen
>
> **摘要:** Most Automatic Speech Recognition (ASR) systems formulate transcription as a prediction problem over orthographic units such as characters, subwords, or words. Although effective, such representations do not explicitly reflect the phonetic structure of speech and often require large vocabularies to maintain adequate coverage. In this work, we are motivated from the phonemic features of Vietnamese to propose a Syllabic-Structure Decoder for ASR, which models speech at the phoneme level instead of the orthographic level. Our approach explicitly captures the phonological composition of syllables, enabling the decoder to generate valid syllabic structures from a compact phonemic inventory. This design more closely aligns with the phonetic realization of speech while significantly reducing vocabulary size. Experimental results on two benchmarks: LSVSC, representing standard speech, and UIT-ViMD, a multi-dialect corpus containing diverse regional pronunciations, show that our method consistently outperforms strong previous baselines, especially pretrained baselines such as PhoWhisper and Wav2Vec2, despite using a substantially smaller vocabulary and no additional training resources. These results highlight the effectiveness of phoneme-based syllabic modeling for ASR in this language. Code for experimental reproducibility will be publicly available upon the acceptance of this paper.
>
---
#### [new 047] ResearchMath-14K: Scaling Research-Level Mathematics via Agents
- **分类: cs.CL**

- **简介: 该论文属于数学推理任务，旨在解决研究级数学问题的数据缺失问题。通过构建大规模数据集ResearchMath-14k及生成推理轨迹，探索语言模型在无监督下的表现与优化方法。**

- **链接: [https://arxiv.org/pdf/2605.28003](https://arxiv.org/pdf/2605.28003)**

> **作者:** Guijin Son; Seungyeop Yi; Minju Gwak; Hyunwoo Ko; Wongi Jang; Youngjae Yu
>
> **备注:** Work in progress. Dataset available at: this https URL
>
> **摘要:** The frontier of mathematics is defined by problems whose solutions are not yet known, yet it remains unclear whether language models can meaningfully engage with such problems without human intervention. A major obstacle is the lack of large-scale research-level math datasets. To this end, we introduce ResearchMath-14k, a set of $14{,}056$ problems curated from academic sources via a multi-agent pipeline, making it the largest collection of research-level mathematical problems to date. We further generate ResearchMath-Reasoning, $220$K teacher trajectories from two open models, where we observe recurring avoidance behaviors such as non-attempts and fabricated references. Interestingly, across eight open-weight models, newer generations produce $5.6\times$ more references and $5.0\times$ more fake references per trace. After agentic filtering of ResearchMath-Reasoning, fine-tuning Qwen3 models from 4B to 30B parameters improves over base models by $9.2$ points on average. This shows that filtered open-problem attempts can provide useful supervision even without fully correct reasoning traces. We make ResearchMath-14k publicly available for future works on research-level mathematical reasoning.
>
---
#### [new 048] The Harder Text Embedding Benchmark (HTEB): Beyond One-dimensional Static Robustness
- **分类: cs.CL**

- **简介: 该论文属于文本嵌入模型评估任务，旨在解决静态基准无法全面反映模型鲁棒性的问题。提出HTEB动态评估框架，从三个维度测试模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.28190](https://arxiv.org/pdf/2605.28190)**

> **作者:** Manuel Frank; Haithem Afli
>
> **备注:** 29 pages, 11 figures
>
> **摘要:** Embedding benchmarks like MTEB report a single score per model, implicitly treating robustness as a static, scalar property. We argue that embedding robustness is multidimensional, since models respond differently to different types of variation, and requires dynamic evaluation to expose failures hidden by static benchmarks. We introduce the Harder Text Embedding Benchmark (HTEB), a dynamic evaluation framework that challenges model robustness along three practically interpretable axes (Lexical/Stylistic, Length and Language) by stochastically transforming inputs at evaluation time with an LLM. Evaluating 16 open-weight embedding models on 32 datasets covering 42 languages under transformations validated by 4,800 human ratings on an English subsample, we find three patterns: (1) Models exhibit specific, partly decoupled robustness profiles across axes. (2) Across three model families, scale increases absolute scores but does not close the gap between original and transformed evaluations. Here, scaling tends to improve specifically the Language axis. (3) English datasets are more sensitive to HTEB transformations than multilingual datasets. This demonstrates that HTEB identifies strengths and weaknesses of models along deployment-relevant axes, challenging current embedding benchmarks and arguing for multidimensional, dynamic robustness evaluation.
>
---
#### [new 049] Modeling Community Attitude through Reaction Tone: A Human-AI Collaborative Framework for Evaluating LLM Alignment with Linguistic Behaviors in Online Communities
- **分类: cs.CL; cs.AI; cs.SI**

- **简介: 该论文属于社会计算任务，旨在解决LLM在模拟在线社区语言行为上的真实性问题。通过构建CARE框架，对比真实社区反应与模型输出，发现现有对齐策略效果有限。**

- **链接: [https://arxiv.org/pdf/2605.27388](https://arxiv.org/pdf/2605.27388)**

> **作者:** Nuan Wen; Xuezhe Ma
>
> **备注:** Preprint
>
> **摘要:** Large language models (LLMs) are increasingly utilized as proxies for computational social analysis; yet, their ability to faithfully represent the "thick descriptions" (Geertz, 1973) of human communities remains a critical challenge. Current evaluations often reduce social identity to static labels, sidelining how real-world groups navigate social shifts. To bridge this gap, we introduce CARE (Community-Aware Reaction Evaluation), a reaction-centered framework that benchmarks LLM-simulated discourse against the authentic, event-contingent responses of distinct communities to real-world news. By characterizing a fine-grained spectrum of illocutionary tones and the underlying attitudes they manifest--validated through human-AI collaboration--our diagnosis reveals a persistent "realism gap": steering LLMs with explicit community prompts fails to inherently improve simulation fidelity. Analysis further identifies divergent behavioral signatures among frontier models, suggesting that current alignment strategies remain insufficient for capturing the sociolinguistic dynamics of online groups.
>
---
#### [new 050] Breaking the Script Barrier: Enabling Automatic Alignment for PoS-based ASR Error Analysis in Non-Latin Scripts
- **分类: cs.CL**

- **简介: 该论文属于语音识别误差分析任务，旨在解决非拉丁语系中ASR错误的细粒度分析问题。提出一种跨语言的自动对齐方法，支持基于词性（PoS）的误差分析，并验证其在多种书写系统中的有效性。**

- **链接: [https://arxiv.org/pdf/2605.28438](https://arxiv.org/pdf/2605.28438)**

> **作者:** Prasenjit K Mudi; Dahlia Devapriya; Sheetal Kalyani
>
> **摘要:** Automatic Speech Recognition (ASR) systems are commonly evaluated using aggregate metrics such as Word Error Rate (WER), which do not capture the linguistic structure of errors. Fine-grained analysis, such as Part-of-Speech (PoS)-wise error characterization, requires accurate alignment between ASR hypotheses and reference transcriptions. However, existing alignment tools are often unreliable for languages written in non-Latin scripts. In this work, we address this gap by proposing a robust, automated, language-agnostic alignment mechanism applicable across ASR architectures and across languages written in both Latin and non-Latin scripts. This enables consistent alignment of hypotheses, references, and evaluation sequences, forming the basis for downstream linguistic analysis. Building on this, we employ standard PoS taggers to perform scalable and reproducible PoS-wise error analysis. Notably, we perform alignment and downstream ASR error analysis across three major segmented writing systems, namely, Abugida (Tamil, Hindi, Kannada), Alphabetic (English, Russian, Greek), and Abjad (Arabic). We further demonstrate how such error information can be leveraged during ASR training to improve metrics such as WER.
>
---
#### [new 051] Ask Now, Use Later: Benchmarking the Proactivity Gap in Long-Lived LLM Agents
- **分类: cs.CL**

- **简介: 该论文研究长期运行的LLM代理的主动性缺口问题，提出ATRBench基准评估代理是否适时询问用户未来所需信息。任务属于自然语言处理中的对话系统优化。**

- **链接: [https://arxiv.org/pdf/2605.28108](https://arxiv.org/pdf/2605.28108)**

> **作者:** Bin Wu; Guanyun Zou; Bingbing Wang; Huan Zhao; Chuan Shi
>
> **摘要:** A long-lived LLM agent, such as OpenClaw, earns its value by acting on a user's preferences and constraints across sessions, not just the current request. Yet today's agents keep what a user volunteers but rarely ask for what stays unspoken, leaving a proactivity gap in long-lived LLM agents: an agent cannot act on a preference it never obtained. As users delegate more of their affairs to agents, the impact of this gap grows. We isolate one concrete, controllable slice of this gap as Ask-to-Remember (ATR): the agent decides whether to ask now for a reusable user preference that the current task does not need but a later session with the same user will. ATR is hard even to evaluate: the right question is underdetermined and its payoff deferred to tasks that may never arise. ATRBench, to the best of our knowledge the first ATR benchmark, makes it measurable by fixing each user's preferences as hidden ground truth, so success demands asking, not recall. Across eight frontier LLM agents, defaults fall at least 62 points below an oracle handed the relevant preference, and prompting closes little of it. Diagnostics identify acquisition as the bottleneck. ATRBench surfaces this proactivity gap in current agents and offers a diagnostic testbed for closing it.
>
---
#### [new 052] Unlocking Fine-Grained and Within-Utterance Speaking Style Control in Prompt-Based Text-to-Speech Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本到语音合成任务，解决prompt-based TTS模型在风格控制上的细粒度和跨句内动态变化不足的问题。提出方法实现跨句风格插值和句内风格平滑过渡。**

- **链接: [https://arxiv.org/pdf/2605.27376](https://arxiv.org/pdf/2605.27376)**

> **作者:** Jaehoon Kang; Yejin Lee; Yoonji Park; Kyuhong Shim
>
> **摘要:** While prompt-based text-to-speech (TTS) models enable natural language-driven speaking style control, they often provide limited fine-grained control and apply a single global style across an utterance. This restricts practical use cases that require continuous style attribute interpolation across utterances and time-varying style transitions within a single utterance. In this paper, we propose novel techniques to achieve both capabilities in existing prompt-based TTS models. For inter-utterance style interpolation, we compute direction vectors between contrastive style prompts in the embedding space and perform simple interpolation, enabling smooth transitions between style characteristics. For intra-utterance style transition, we first identify a strong attention bias toward early tokens in autoregressive TTS decoders, causing the initial audio realization to dominate subsequent generation. To mitigate this effect, we introduce KV-cache swapping and sliding-window attention masking. Experiments demonstrate that our proposed inter-utterance interpolation achieves a 99-100% success rate in gender conversion, up to 36 Hz pitch variation, and up to 1.6 syllables-per-second speed change. Our intra-utterance transition maintains a speaker similarity of 0.81-0.91 and achieves perceptual smoothness scores of 3.48-4.48.
>
---
#### [new 053] TRACES: Proactive Safety Auditing for Multi-Turn LLM Agents via Trajectory-State Modeling
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出TRACES，用于多轮LLM代理的主动安全审计。解决安全风险在中间步骤出现但难以检测的问题，通过轨迹状态建模进行风险预测与识别。**

- **链接: [https://arxiv.org/pdf/2605.27690](https://arxiv.org/pdf/2605.27690)**

> **作者:** Jiaqian Li; Yanshu Li; Boxuan Zhang; Ruixiang Tang; Kuan-Hao Huang
>
> **摘要:** LLM agents increasingly operate through multi-turn tool use and environment interaction, where safety risks often emerge from intermediate steps long before they surface in the final outcome. Reactive auditing is therefore insufficient: post-hoc diagnosis frequently misses the chance to flag risks while they are unfolding. We propose TRACES, a representation-based proactive auditor that learns prefix-level trajectory risk states from the hidden representations of an observer LLM. TRACES induces latent mechanism features from step representations and models their temporal evolution to estimate whether a partial trajectory is drifting toward unsafe behavior. To sidestep the cost and ambiguity of step-level risk annotation, TRACES is trained with weak trajectory-level supervision while still producing dense prefix-level risk estimates. Across multiple agent safety benchmarks, TRACES improves both full-trajectory safety prediction and proactive risk discrimination. Our analyses further suggest that these risk states can help train a safer agent, highlighting the broader potential of proactive auditing for long-horizon agent safety.
>
---
#### [new 054] Extracting Small Translation Specialists from LLMs by Aggressively Pruning Experts
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于机器翻译任务，旨在解决LLM过度参数化问题。通过激进剪枝专家，保留翻译所需模块，显著压缩模型规模。**

- **链接: [https://arxiv.org/pdf/2605.28042](https://arxiv.org/pdf/2605.28042)**

> **作者:** Liu O. Martin; Lucas Bandarkar; Nanyun Peng
>
> **摘要:** Modern large language models (LLMs) achieve state-of-the-art machine translation performance, but they do so as broad generalists largely trained for many tasks and capabilities unrelated to translation. Thus, they are heavily overparameterized for this task, resulting in excessive memory and compute requirements. In this paper, we present a method for aggressively pruning experts from modern mixture-of-experts LLMs while incurring negligible degradation in translation quality. Our approach exploits expert specialization and the separability of multilingual capabilities in LLMs to identify experts irrelevant to translation. And because of the modular nature of MoEs, these can be easily pruned without any training. Without retraining, we are able to prune half of all experts with negligible degradation and 70% with only minor losses. With a very short SFT, we prune 75% of experts while recovering baseline performance, and in some settings remove nearly 90% while maintaining reasonable translation quality. Overall, our results show that translation requires only a fraction of the LLM, enabling substantial compression of the MoE blocks that contain over 90% of parameters.
>
---
#### [new 055] The Cases LJP Never Sees: Prosecution Decision Prediction for More Complete Criminal Liability Assessment
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Prosecution Decision Prediction（PDP）任务，解决LJP在刑事责任评估中的盲点问题，通过构建PDP-Bench数据集进行实验，评估AI在证据判断和法律推理方面的能力。**

- **链接: [https://arxiv.org/pdf/2605.28464](https://arxiv.org/pdf/2605.28464)**

> **作者:** Junyu Lu; Qi Wei; Peishuo Zheng; Jie Zhang; Hui Huang; Qianru Wang; Chuan Xiao; Jianbin Qin; Shuyuan Zheng
>
> **备注:** 24 pages, 5 figures, 22 tables
>
> **摘要:** Legal Judgment Prediction (LJP) has become a core benchmark for evaluating AI in the criminal legal domain, but it only sees criminal cases that have already passed prosecutorial review and been formally indicted. As a result, LJP leaves a substantial blind spot in assessing criminal liability, overlooking cases involving insufficient evidence, no criminal liability, or guilt exempted from punishment. To fill this gap, we propose \textbf{Prosecution Decision Prediction (PDP)}, the first Legal AI task built around prosecutorial review, which classifies each case into prosecution or one of three non-prosecution decisions and reflects legal AI's capabilities in evidence evaluation, legal subsumption, and value-based discretion. We further construct \textbf{PDP-Bench}, a benchmark of 4{,}630 real Chinese prosecutorial decisions spanning 190 charges. Extensive experiments show that state-of-the-art LLMs perform substantially worse on PDP than on LJP and that mainstream enhancement routes fail to close the gap. Moreover, controlled RLVR interventions show that simple outcome rewards fail to produce generalizable PDP discrimination.
>
---
#### [new 056] ClinicalEncoder26AM: A Multlilingual Diagnosable ColBERT Model; Evidences from the MultiClinNER Shared Task
- **分类: cs.CL**

- **简介: 该论文提出ClinicalEncoder26AM，一个用于临床和生物医学文本的多语言可诊断ColBERT模型，解决跨语言实体识别问题。通过结合合成数据与标注资源进行预训练，提升实体召回率。**

- **链接: [https://arxiv.org/pdf/2605.28521](https://arxiv.org/pdf/2605.28521)**

> **作者:** François Remy
>
> **摘要:** ClinicalEncoder26AM is a multilingual Diagnosable ColBERT for clinical and biomedical texts, which aligns at multiple levels its token-level semantic with ClinicalMap25, a clinical latent space inspired by BioLORD-2023 and enriched with synthetic and annotated supervision. The post-training recipe builds upon BGE-M3, and combines synthetic clinical notes, patient--doctor conversations, and annotated resources such as MedMentions, while considering both named-entity-level and sentence-level representations in a multi-adapter distillation, along with a ColBERT-style retrieval objective. In this system demonstration paper, we evaluate the model in the MultiClinNER shared task by finetuning it as a BIO tagger for patient symptoms, disorders, and procedure spans, using a lightweight two-layer CNN head to improve local boundary detection. The resulting system remains simple, processes most documents in a single 8192-token window, and achieves state-of-the-art multilingual entity recall, while achieving Top 5 overall across all entity types and languages in Character-weighted F1 scores. Training curves further show that ClinicalEncoder26AM is markedly more data-efficient than the base M3 model, supporting the usefulness of its clinical post-training for downstream information extraction. The model can be downloaded on this https URL
>
---
#### [new 057] FinBoardBench: Benchmarking Dynamic Wealth Management and Strategic Financial Reasoning of LLMs via Board Game Simulations
- **分类: cs.CL; cs.CE**

- **简介: 该论文提出FinBoardBench，用于评估大语言模型在动态财富管理和金融决策方面的能力，解决现有基准不足的问题。通过模拟财务棋盘游戏，测试模型的综合金融技能。**

- **链接: [https://arxiv.org/pdf/2605.27896](https://arxiv.org/pdf/2605.27896)**

> **作者:** Xuesi Hu; Peng Wang; Jinpeng Miao; Xilin Tao; Caiwei Li; Yue Ma; Jie He; Qiancheng Zhang; Yuntao Zou; Dagang Li
>
> **备注:** Preprint
>
> **摘要:** Recently, large language models (LLMs) have achieved superior performance in static financial reasoning and simple dynamic trading tasks. However, existing static financial benchmarks are insufficient to assess the dynamic wealth management and financial decision-making capabilities of LLMs in real-world environments. To bridge this gap, we present FinBoardBench, an evaluation suite based on three classic financial board games: Cashflow, Acquire, and Monopoly. FinBoardBench assesses a comprehensive set of financial skills, including personal cash flow management with debt balancing, corporate investment and acquisition forecasting, and competitive trade negotiations with asset auctions. Our experiments with 9 advanced LLMs reveal that while exhibiting basic long-term planning and investment logic, they fail to effectively leverage complex interactions for profit, and their strong static reasoning performance does not transform into successful dynamic decision-making. Notably, they tend to prioritize immediate asset acquisition over maintaining sufficient liquidity, making them vulnerable to financial crises triggered by random events. We hope that FinBoardBench can provide a valuable reference for more intelligent LLM-based decision-making systems in the future.
>
---
#### [new 058] StoryMI: Steerable Multi-Agent Therapeutic Dialogue Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出StoryMI，一个用于生成可控动机访谈对话的多智能体框架，解决传统方法缺乏情境引导和策略控制的问题。通过构建情景故事和动态协调策略，提升对话的临床有效性。**

- **链接: [https://arxiv.org/pdf/2605.27393](https://arxiv.org/pdf/2605.27393)**

> **作者:** Qingyu Meng; Min Chen; Dingming Liu; Yifan Mo; Yue Su; Xin Sun; Koen Hindriks; Jiahuan Pei
>
> **备注:** ACL2026
>
> **摘要:** Large language models (LLMs) can generate fluent dialogue, but prior works lack situational grounding, dynamic strategy control, and evaluation aligned with clinical standards in motivational interviewing (MI). We introduce StoryMI, a multi-LLM agent framework for controllable MI dialogue generation, where questionnaire-based client profiles are expanded into situational stories that provide narrative context for the dialogue. Therapist and client agents generate MI-coded utterances guided by MI codes selected by the interaction agent, while an interaction agent dynamically coordinates exchanges to control MI strategies during a multi-turn conversation. We propose a two-level evaluation protocol: lexical metrics and MI-specific measures of macro-level counseling strategies, alongside LLM-as-judge and human expert assessments. We construct a dataset of 6K simulated MI dialogues grounded in 1K questionnaire-story pairs, covering 12 MI codes and 13 symptom domains, and benchmark six open- and closed-source LLMs. Our results show that situational grounding and macro-level control can improve MI adherence and clinical plausibility, demonstrating the effectiveness of a structured multi-agent workflow for psychotherapy dialogue generation. We provide code and data for reproducibility.
>
---
#### [new 059] Bridging the Stability-Expressivity Gap: Synthetic Data Scaling and Preference Alignment for Low-Resource Spoken Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音合成任务，解决低资源语言中合成数据导致的稳定性与表达性矛盾问题，提出两种自对齐框架以提升模型表现。**

- **链接: [https://arxiv.org/pdf/2605.27383](https://arxiv.org/pdf/2605.27383)**

> **作者:** Yizhong Geng; Yanliang Li; Jinghan Yang; Tianhan Jiang; Boxun An; Ya Li; Xiaoyu Shen
>
> **摘要:** Spoken Language Models (SLMs) have emerged as a promising paradigm for speech synthesis by bypassing explicit grapheme-to-phoneme pipelines. However, their effectiveness in low-resource languages remains fundamentally limited by the scarcity of transcribed speech. In practice, synthetic data has become the primary strategy for scaling SLMs in such settings, providing reliable phonetic supervision when real data is insufficient. In this work, we show that this reliance introduces a fundamental trade-off, which we term the Stability-Expressivity Gap: while synthetic data improves phonetic accuracy, it progressively suppresses prosodic variability, ultimately leading to a collapse of expressivity (Synthetic Erosion). To bridge this gap, we propose two self-alignment frameworks. Disentanglement-Guided Self-Alignment (DGSA) recovers expressivity for complex languages by exploiting prosody-timbre separation. For regimes where authentic references are exceptionally limited, Temperature-Driven Self-Critique (TDSC) stabilizes generation through automated exploration and filtering. Our approach outperforms strong commercial systems, including ElevenLabs and Gemini Pro, and enables the first zero-shot voice cloning capability for Lao.
>
---
#### [new 060] Boundary Suppression Asymmetry in Post-trained Assistants: Over-expansion as a Controllability Cost
- **分类: cs.CL**

- **简介: 该论文研究后训练语言模型在边界控制上的不对称性问题，探讨优化导致的可控性成本。任务属于AI模型可控性分析，解决如何有效抑制过度响应的问题。工作包括实验设计与机制分析。**

- **链接: [https://arxiv.org/pdf/2605.27969](https://arxiv.org/pdf/2605.27969)**

> **作者:** Jiarui Han
>
> **摘要:** Post-trained language-model assistants are often optimized to avoid under-answering, encouraging complete, helpful, cautious, and proactive responses. We ask whether this optimization creates asymmetric controllability costs: when users explicitly request narrower answers, which assistant behaviors remain suppressible, and which continue to shape the response? We study this problem as boundary-suppression asymmetry. Prompt-side probes across multiple high-level response dimensions suggest a selective cost, concentrated around `too-much assistant' directions such as over-completion, extra help, and anti-underanswering. Using controlled assistant-policy variants derived from a shared base model, we find that anti-underanswering policies are harder to pull back than the baseline under matched boundary-control evaluations, while minimal-boundary variants generally avoid this anti-side upward shift in the direct boundary-control comparisons. Mechanism-oriented probes point beyond longer default outputs, pure EOS failure, uncertainty compensation, and local continuation bias, while robustness checks preserve the main anti-over-baseline ordering under shared-system and larger-scale settings. The evidence supports a mixed planning/stopping account, where content-budget overshoot and continuation persistence jointly make boundary correction harder. Overall, post-training may create direction-specific controllability costs: some helpful assistant tendencies remain easy to invoke, yet harder to locally suppress.
>
---
#### [new 061] Let the Results Speak: A Replication-First Paradigm for LLM Behavioral Benchmarking
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种复制优先的LLM行为基准测试范式，解决主观评估一致性低的问题。通过多维度验证确保评估工具可靠性，应用于情感陪伴测试，揭示模型性能变化。**

- **链接: [https://arxiv.org/pdf/2605.27914](https://arxiv.org/pdf/2605.27914)**

> **作者:** Yuming; Huang; Yao Liu; Lei Wang; Junchen Wan
>
> **摘要:** Subjective evaluation of LLM behavior -- empathy, restraint, calibrated emotional tone -- is hard. Human inter-rater agreement on such qualities saturates near rho ~ 0.45, and an LLM-as-judge proxy alone risks circularity: a judge sharing the target's training cohort cannot independently verify it. Anchoring validity to a single human-rater consensus does not extend to capabilities where humans themselves disagree. We propose a replication-first paradigm: instead of anchoring on one rater group, we certify the instrument via four orthogonal properties -- reliability across K runs, cross-instrument replication across architecturally distinct judges, historical-footprint calibration via judges from earlier training cohorts, and pre-registered prediction. We test it on emotional accompaniment by letting the rubric self-evolve data-driven across iterations: the dimensions are not pre-stipulated and the procedure stabilizes to a 9-dimension set. Pre-registration applies to 10 falsifiable hypotheses and 11 forward predictions, committed before any test data was collected. Applied to 49 models across 8 families, the paradigm surfaces what aggregate scores hide. On advice-restraint -- whether a model refrains from giving unsolicited solutions in empathic contexts -- gpt-5 falls 1.87 points from gpt-4.1 and Opus-4.7 falls 0.629 from Opus-4.6, while aggregate scores stay flat. The regression survives three user-proxy swaps (95% of magnitude), replicates across a 5-family judge stack and a 17-month cohort gap, and persists on 74 held-out real ESConv conversations (rho in [0.749, 0.850]); the instrument reaches ordinal Krippendorff alpha = 0.91. As a by-product, the paradigm acts as a saturation-source diagnostic, separating instrumental ceilings (breakable by rubric refinement) from structural ceilings (needing scenario or roster intervention).
>
---
#### [new 062] MERIT: Matching Expertise via Rubric-Informed Training for Reviewer Assignment
- **分类: cs.CL**

- **简介: 该论文属于论文审稿人匹配任务，旨在解决现有方法依赖粗粒度信号或昂贵标注的问题。提出MERIT框架，通过强化学习和嵌入检索实现高效准确的审稿人匹配。**

- **链接: [https://arxiv.org/pdf/2605.27865](https://arxiv.org/pdf/2605.27865)**

> **作者:** Zixuan Yang; Yibo Zhao; Weicong Liu; Xiang Li
>
> **备注:** 22pages, 8 figures, 12 tables
>
> **摘要:** Matching submissions with suitable reviewers at scale is a growing challenge for major venues, yet existing approaches either rely on coarse proxy signals that conflate general relatedness with true suitability, or require expensive human annotations that are difficult to scale for training. We propose MERIT, a two-stage framework that bridges this gap by converting criterion-level expertise matching into scalable suitability supervision. In the first stage, we train a reviewer assessor via reinforcement learning to identify the expertise dimensions a paper requires, match them against the reviewer's prior work, and produce a suitability decision, with rewards provided by an LLM judge guided by paper-specific expertise rubrics. In the second stage, we distill the assessor's predictions into an embedding-based retriever for efficient large-scale assignment. Experiments show that our 4B reviewer assessor outperforms larger general-purpose LLMs on suitability classification, and the resulting retriever achieves state-of-the-art performance across LR-Bench and the CMU Gold dataset. Our code is available at this https URL.
>
---
#### [new 063] GraphLit: Learning Text-Enriched Dynamic Character Network Representations for Literary Study
- **分类: cs.CL**

- **简介: 该论文提出GraphLit，用于学习文学文本中的动态角色网络表示，解决角色互动与文本上下文关联的问题。通过DHCNs和自监督学习，提升文学分析任务效果。**

- **链接: [https://arxiv.org/pdf/2605.28643](https://arxiv.org/pdf/2605.28643)**

> **作者:** Gaspard Michel; Elena V. Epure; Romain Hennequin; Christophe Cerisara; Mirella Lapata
>
> **摘要:** Methods to represent literary texts as graphs or sequences of graphs mainly focus on representing character interactions, and often overlook another crucial aspect: the textual context in which characters interact. We introduce Dynamic Heterogeneous Character Networks (DHCNs), which organize long novels into temporally localized heterogeneous graphs that align characters with their textual contexts. We extract around 20,000 DHCNs from Project Gutenberg, and propose GraphLit, a self-supervised learning framework that learns rich literary representations through a masked graph autoencoder objective. Across a wide-range of 12 character-related tasks, GraphLit improves over text-only and graph-only baselines, particularly on tasks requiring contextual understanding. Finally, we demonstrate the applicability of DHCNs and GraphLit for literary analysis by studying the link between narrative non-linearity and dynamic social features.
>
---
#### [new 064] SMILE-Next: Teaching Large Language Models to Detect, Classify, and Reason about Laughter
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SMILE-Next数据集及模型，解决真实场景下笑声的检测、分类与推理问题，通过自监督学习和专家路由机制提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.28084](https://arxiv.org/pdf/2605.28084)**

> **作者:** Lee Jung-Mok; Kim Sung-Bin; Joohyun Chang; Lee Hyun; Tae-Hyun Oh
>
> **摘要:** Laughter is a complex social signal that conveys communicative intent beyond amusement. While prior work has focused on isolated laughter analysis tasks, a comprehensive understanding of laughter in real-world scenarios remains underexplored. Therefore, we introduce SMILE-Next, a dataset for real-world laughter understanding with multimodal textual representations and question-answer annotations across three tasks: laughter detection, laughter type classification, and laughter reasoning. Building upon SMILE-Next, we aim to develop a laughter-specialized large language model capable of nuanced understanding of laughter in real-world contexts. To this end, we propose two key components: laughter-specific Self-Instruct and the Mixture-of-Laugh-Experts (MoLE) framework. Laughter-specific Self-Instruct enhances generalization across tasks and domains by automatically synthesizing diverse laughter-centric instructions. MoLE introduces a task-adaptive expert routing mechanism that dynamically selects specialized experts tailored to each laughter-related task, improving task-specific performance and efficiency. Experimental results show that the combination of our proposed components substantially outperforms multimodal LLM baselines, advancing robust real-world laughter understanding. Project page is at: this https URL.
>
---
#### [new 065] Pressure-Testing Deception Probes in LLMs: Scaling, Robustness, and the Geometry of Deceptive Representations
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLM中欺骗检测探针的鲁棒性，分析其在分布变化下的失效原因，提出风格增强方法提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.27958](https://arxiv.org/pdf/2605.27958)**

> **作者:** Sachin Kumar
>
> **备注:** Accepted at the GEM Workshop @ ACL 2026
>
> **摘要:** Linear probes trained on LLM activations are increasingly proposed as deception-detection metrics, yet report AUROC exceeding 0.96 on clean benchmarks while collapsing under distributional shift. This paper systematically pressure-tests probe-based metrics across the Gemma 3 model family (1B-27B parameters), diagnosing why they fail rather than merely documenting that they fail. We test four hypotheses about deception encoding: (1) single linear direction, (2) multi-dimensional subspace, (3) convex conic hull, (4) entropy proxy. Our design includes cross-domain transfer matrices, multi-dimensional probe analysis with permutation null baselines, entropy-residualization tests, and distractor evaluations across 8 stylistic shifts. We find that: (a) probes achieve near-perfect AUROC (>=0.998) on clean data but collapse under stylistic shifts; style-augmented probes recover near-perfect detection (mean AUROC 0.979-0.983) on unseen styles; (b) the single-direction hypothesis is rejected (k=1 captures only 0.61-0.80 AUROC), with cross-domain transfer failure confirmed as geometric rather than layer-mismatch-driven; (c) the entropy-proxy hypothesis is rejected (max |rho|=0.454, max Delta-AUROC after residualization=0.004); and (d) deception does not form a significant linear subspace (per-domain k*=0), yet multi-dimensional probes (k>=5) recover the signal through distributed sub-threshold features. Probe fragility reflects distributional narrowness rather than an architectural limitation: style-augmented probes recover near-perfect detection at both 4B and 27B, establishing that the inverse scaling pattern is a training-distribution artifact rather than a genuine scale-dependent phenomenon.
>
---
#### [new 066] Do Models Know Why They Changed Their Mind? Interpretability and Faithfulness of Chain-of-Thought Under Knowledge Conflict
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型在知识冲突下的决策机制，探讨其思维链（CoT）是否忠实反映决策过程。任务属于模型解释性与可信度分析，解决模型决策可解释性问题，通过实验验证CoT的稳定性与信心评分的相关性。**

- **链接: [https://arxiv.org/pdf/2605.27773](https://arxiv.org/pdf/2605.27773)**

> **作者:** Pruthvinath Jeripity Venkata
>
> **备注:** 12 pages, 8 tables, 3 appendices
>
> **摘要:** When a language model sees a document contradicting its training knowledge, it must choose: follow the document or trust itself. Prior work proved this choice depends on how well-known the fact is. We ask: does the model's chain-of-thought (CoT) reasoning faithfully report this mechanism? We introduce introspective faithfulness and test it across 200 questions, 8 models, and 4 prompt conditions. We find CoT reasoning is highly stable across opposite decisions: flip pairs retain 96% of same-answer similarity (d=0.34; confirmed by ROUGE-L, d=0.45). Yet self-rated confidence carries a faint genuine signal: for obscure facts where entity fame is uninformative, confidence still predicts decisions (p<0.001) and tracks item-level knowledge (r=0.134). GPT-4o is the only model with statistically reliable reasoning-decision coupling. Claude Sonnet 4.6 shows the widest confidence range (SD=1.39) but near-zero pooled correlation because the confidence-decision relationship reverses between conditions; a temperature ablation confirms this is model-specific. Internal thinking tokens show greater decision-sensitivity than user-facing CoT (p=0.033). CoT decomposes into a decision-invariant knowledge display (~96%) and a thin confidence layer with weak but real signal. For monitoring: read confidence, not the argument.
>
---
#### [new 067] DisasterBench: Benchmarking LLM Planning under Typed Tool Interface Constraints
- **分类: cs.CL**

- **简介: 该论文属于多智能体规划任务，旨在解决灾难响应中LLM协调异构工具的问题。提出DisasterBench基准和FPoF方法，评估LLM生成正确工作流的能力。**

- **链接: [https://arxiv.org/pdf/2605.27957](https://arxiv.org/pdf/2605.27957)**

> **作者:** Zhitong Chen; Kai Yin; Weifeng Zhang; Zhiyuan Wang; Xiangjue Dong; Chengkai Liu; Zhewei Liu; Yiming Xiao; Ali Mostafavi; James Caverlee
>
> **摘要:** Disasters cause severe societal impacts, demanding rapid coordination of heterogeneous AI tools, from satellite analysis to flood prediction and damage assessment, into coherent multi-step workflows. As LLMs increasingly serve as orchestrators of such pipelines, effective coordination requires more than selecting semantically plausible tools: LLMs must generate executable workflows with correct parameter binding and dependency propagation. We introduce DisasterBench, a benchmark for evaluating structured multi-agent planning over semantically similar but operationally distinct disaster-response tools. To enable step-level failure attribution, we further propose First-Point-of-Failure (FPoF), which localizes the earliest root cause in a predicted workflow, separating primary errors from downstream cascading effects. Our evaluation reveals three findings: planning method effectiveness depends strongly on model capacity; tool mismatch and parameter-binding errors dominate first failures, revealing semantic grounding and execution consistency as distinct bottlenecks; and verbose intermediate reasoning can create instruction clash with structured output requirements, disrupting plan generation. Together, these findings highlight a fundamental gap between semantic reasoning and execution-grounded coordination, underscoring the need for planning frameworks that jointly model semantic intent, execution constraints, and workflow consistency. Code, data, and evaluation resources are available at: this https URL
>
---
#### [new 068] AI Research Agents Narrow Scientific Exploration
- **分类: cs.CL**

- **简介: 该论文属于AI与科学探索交叉任务，旨在评估AI研究代理是否拓宽科学探索。研究比较了AI生成想法与人类论文的分布和影响力，发现AI更集中于已有工作。**

- **链接: [https://arxiv.org/pdf/2605.27905](https://arxiv.org/pdf/2605.27905)**

> **作者:** Yixuan Tang; Yi Yang
>
> **摘要:** AI research agents can now generate research ideas, design experiments, run code, and draft papers, raising the possibility of large-scale AI-assisted scientific discovery. Many current agent frameworks explicitly encourage the generation of novel and high-impact ideas. Yet it remains unclear whether AI-assisted ideation broadens scientific exploration or mainly concentrates around existing work. We study AI research agents as scientific search systems. Using four AI research-agent frameworks and six large language models, we generate 37,802 scientific ideas from shared seed literature across citation-defined research areas in AI and machine learning. We then compare the resulting AI ideas against human-authored papers from the same research areas, follow-on human research emerging from the same seed literature, and the seed literature itself. Across experiments, four consistent patterns emerge. First, AI-generated ideas are substantially more concentrated than human-authored papers from the same research areas. Second, AI-generated ideas remain much closer to their starting literature than later human follow-on work does. Third, papers most similar to AI-generated ideas tend to receive lower subsequent citations. Fourth, when AI-generated ideas differ from prior work, the differences arise primarily from recombining existing technical methods rather than introducing fundamentally new research questions. Overall, current AI research agents appear better suited to local elaboration than to broadening scientific exploration.
>
---
#### [new 069] Knowledge Dependency Estimation for Reliable Question Answering
- **分类: cs.CL**

- **简介: 该论文属于问答任务，解决如何评估模型依赖的知识单元问题。提出Knot方法，通过结构化方式估计知识依赖性，提升问答的可靠性与可解释性。**

- **链接: [https://arxiv.org/pdf/2605.28047](https://arxiv.org/pdf/2605.28047)**

> **作者:** Chaodong Tong; Qi Zhang; Nannan Sun; Lei Jiang; Yanbing Liu
>
> **备注:** 12 tables, 9 figures
>
> **摘要:** Reliable question answering requires identifying not only whether an answer is correct, but also which available knowledge the prediction depends on. In realistic LLM-based QA, this knowledge may come from context, retrieval, decomposition, or intermediate reasoning, forming a noisy and redundant candidate space rather than a clean gold evidence set. We study \emph{knowledge dependency estimation}: estimating the sensitivity of a fixed black-box QA model to different candidate knowledge units. The challenge is to obtain fine-grained dependency scores without exhaustive test-time perturbation while modeling redundancy, substitutability, and complementarity. We propose \textbf{Knot}, a structured rank-aware knowledge dependency estimator. Knot learns from subset-level counterfactual supervision, models subset sensitivity through coverage over latent dependency factors, and derives rank-aware unit scores to identify influential candidates. Across multiple-choice and generative QA benchmarks, Knot outperforms all compared baselines in subset-sensitivity prediction and produces more faithful unit rankings than deployable baselines without extra QA-model calls; when used for practical risk screening, its dependency scores help flag error-prone QA predictions early.
>
---
#### [new 070] Agent Explorative Policy Optimization for Multimodal Agentic Reasoning
- **分类: cs.CL**

- **简介: 该论文属于多模态智能体推理任务，解决工具使用与思考行为间的不对称问题。提出AXPO方法，提升工具调用效果，优于传统RL方法。**

- **链接: [https://arxiv.org/pdf/2605.28774](https://arxiv.org/pdf/2605.28774)**

> **作者:** Minki Kang; Shizhe Diao; Ryo Hachiuma; Sung Ju Hwang; Pavlo Molchanov; Yu-Chiang Frank Wang; Byung-Kwan Lee
>
> **备注:** Project page: this https URL
>
> **摘要:** Vision-language models with extended reasoning succeed on complex problems, but many real-world problems require external tools that internal reasoning alone often cannot resolve. Agentic reasoning therefore interleaves two behaviors with a structural asymmetry: thinking (the self-contained default) and tool use (a high-variance auxiliary acting). We refer to this asymmetry as the Thinking-Acting Gap. Under standard RL recipes like GRPO, the gap manifests as two diagnostic symptoms during training: tool use is attempted on only ~30% of rollouts, and when attempted, the tool-using rollouts within a group are all-wrong on ~40% of questions, suppressing the learning signal at the tool calls that needed it. We propose AXPO (Agent eXplorative Policy Optimization): for each all-wrong tool-using subgroup, AXPO fixes the thinking prefix and resamples the tool call and its continuation, paired with uncertainty-based prefix selection. Across nine multimodal benchmarks and three scales of Qwen3-VL-Thinking, SFT+AXPO outperforms SFT+GRPO at average (+1.8pp Pass@1 and +1.8pp Pass@4 at 8B on average) and 8B with SFT+AXPO surpasses the 32B Base on Pass@4 with 4 times fewer parameters.
>
---
#### [new 071] Supervised Semantic Differential for Cross-Cultural Concept Analysis: A Case Study of Human Affect
- **分类: cs.CL**

- **简介: 该论文属于跨文化概念分析任务，旨在解决语言间心理意义差异的问题。通过扩展监督语义差分方法，比较多语言情感维度，识别语义对齐与差异，揭示文化差异特征。**

- **链接: [https://arxiv.org/pdf/2605.28225](https://arxiv.org/pdf/2605.28225)**

> **作者:** Jan Sikora; Paweł Lenartowicz; Hubert Plisiecki
>
> **备注:** 9 pages, 2 figures, excluding the appendices. Code to reproduce our results is available at this https URL
>
> **摘要:** Cross-cultural comparison of psychological meaning requires methods that go beyond word-level translation and examine how semantic dimensions are organized across languages. We introduce a cross-lingual extension of the Supervised Semantic Differential (SSD), which estimates supervised semantic gradients in embedding space and compares them across aligned multilingual word embeddings. The method tests gradient alignment and difference using permutation procedures and bootstrap intervals, and interprets residual differences through clustering around the difference gradient. We demonstrate the approach on Polish, English, and French affective norm lexicons, modeling Valence, Arousal, and Dominance where available. Affective dimensions were significantly recoverable across languages and model settings. Cross-lingual comparisons showed broad alignment together with structured residual differences: Valence appeared mostly shared, whereas Arousal and Dominance produced more interpretable contrasts involving bodily threat, aesthetic stimulation, internal emotionality, macro-level authority, and everyday control. Several clusters also reflected corpus-specific artifacts, underscoring the need for cautious interpretation. Cross-lingual SSD offers an explainable framework for testing semantic alignment, identifying divergence, and generating hypotheses about cross-cultural differences in psychological meaning.
>
---
#### [new 072] Can Large Language Models Handle Discourse Particles? A Case Study of Colloquial Malay
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在处理话语标记上的不足。研究提出一个基准和五种属性，评估并提升模型对马来语口语话语标记的理解能力。**

- **链接: [https://arxiv.org/pdf/2605.28782](https://arxiv.org/pdf/2605.28782)**

> **作者:** Mariah Al Giptiah Binte Yusoff; Jakin Tan; Bocheng Chen; Guangliang Liu; Xi Chen
>
> **摘要:** Discourse particles, such as \textit{well} and \textit{kind of}, are crucial components that enable LLMs to ``speak'' more like humans. They are used to convey emotions, intentions, and interpersonal meanings. However, existing studies have not yet built a comprehensive understanding of LLMs' capabilities in handling discourse particles. Moreover, the limited number of studies focuses primarily on high-resource languages such as English, with little attention paid to Southeast Asian languages. In this paper, we (1) propose \textsc{MalayPrag}, a benchmark designed to systematically evaluate and analyze LLMs' capabilities in handling discourse particles in colloquial Malay; and (2) introduce five attributes that provide a linguistically grounded, unified framework for interpreting the pragmatic functions of discourse particles. Applying these two contributions, we prompt ten off-the-shelf LLMs to perform three prediction tasks. The experimental results reveal substantial challenges for current LLMs in accurately connecting discourse particles with their pragmatic functions in Malay. The provision of the five attributes designed in this study is found to significantly improve these connections, highlighting the need for structured scaffolding for models' pragmatic competence.
>
---
#### [new 073] Evaluating the Realism of LLM-powered Social Agents: A Case Study of Reactions to Spanish Online News
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，旨在评估LLM生成的社交媒体回复是否具备真实受众反应的特性。研究比较了真实与合成回复在仇恨言论、情感和语义对齐方面的差异。**

- **链接: [https://arxiv.org/pdf/2605.28598](https://arxiv.org/pdf/2605.28598)**

> **作者:** Alejandro Buitrago López; Alberto Ortega Pastor; Javier Pastor-Galindo; José A. Ruipérez-Valiente
>
> **摘要:** LLM-powered social agents are increasingly used to simulate online social behavior, yet their realism remains difficult to validate. Existing work has largely relied on general-purpose benchmarks, while less attention has been paid to short, reactive discourse such as audience replies to online news. In this paper, we evaluate whether LLM-generated reactions to Spanish online news reproduce measurable properties of real audience discourse. Using the Hatemedia dataset, we pair 5,631 news items with 58,555 real audience reactions, and generate a matched synthetic dataset using five LLMs under a shared experimental setting. We compare real and synthetic reactions across three dimensions: hate speech, sentiment, and semantic alignment, considering both off-the-shelf and fine-tuned generation. Results show that off-the-shelf models are poor proxies for real audience reactions: they strongly underproduce hate speech, introduce model-specific sentiment biases, and remain distributionally distant from human replies. Fine-tuning improves fidelity unevenly. Qwen3 provides the most balanced approximation, while Mistral7B achieves the strongest sentiment and semantic alignment but overshoots hate prevalence. Plausible synthetic replies do not necessarily reproduce the distributional properties of public discourse.
>
---
#### [new 074] ReverseMath: Answer Inversion for Scalable and Verifiable Mathematical Problem Generation
- **分类: cs.CL**

- **简介: 该论文提出ReverseMath，用于生成可验证的数学问题，解决静态数据导致模型记忆的问题。通过答案逆向生成新问题，提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2605.27709](https://arxiv.org/pdf/2605.27709)**

> **作者:** Raoyuan Zhao; Yihong Liu; Yupei Du; Hinrich Schütze; Michael A. Hedderich
>
> **摘要:** Mathematical reasoning benchmarks are vital for evaluating large language models (LLMs), but many are static and repeatedly exposed through public evaluation and training pipelines, making it difficult to separate genuine reasoning from memorization. Meanwhile, manually constructing new math problems with reliable answers remains costly. We introduce ReverseMath, a scalable method for generating new math problems through answer inversion. Given a problem and its answer, ReverseMath masks a numerical value in the original problem, treats the original answer as a known condition, and rewrites the problem so that the masked value becomes the new answer. The generated problem reverses the original input-output relation, making its answer known by construction. We study ReverseMath for both evaluation and training. For evaluation, paired original/reversed problems reveal substantial behavioral shifts: models sometimes fail on reversed problems and even incorrectly output the original answer, suggesting memorization-like behavior. For training, ReverseMath provides automatically labeled reversed problems as data augmentation for reinforcement learning (RL). Experiments show that including ReverseMath-generated data improves mathematical reasoning performance across multiple benchmarks, demonstrating its value as both an analysis tool and a scalable source of verifiable training data.
>
---
#### [new 075] IPO-Mine: A Toolkit and Dataset for Section-Structured Analysis of Long, Multimodal IPO Documents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出IPO-Mine工具包和数据集，用于分析长而多模态的IPO文件。解决缺乏标准化数据集的问题，完成文档结构化与图像提取。**

- **链接: [https://arxiv.org/pdf/2605.28714](https://arxiv.org/pdf/2605.28714)**

> **作者:** Michael Galarnyk; Siddharth Lohani; Vidhyakshaya Kannan; Sagnik Nandi; Aman Patel; Liqin Ye; Arnav Hiray; Rutwik Routu; Prasun Banerjee; Siddhartha Somani; Sudheer Chava
>
> **备注:** 12 pages
>
> **摘要:** An Initial Public Offering (IPO) filing is a document released when a private firm goes public, allowing individual (retail) investors to purchase its shares. These filings describe a firm's business, financials, and risks and are long, multimodal documents with narrative text and images. Despite their importance to financial markets, there is no large-scale, standardized dataset or benchmark for studying IPO filings with modern language and multimodal models. These documents pose significant challenges: filings frequently exceed 500,000 tokens and lack consistent structural organization. We introduce the IPO-Toolkit, an open-source framework for downloading and parsing IPO filings into standardized section-structured text and extracted images. The toolkit segments filings, extracts embedded images, and produces structured outputs that enable large-scale, reproducible analysis workflows over long, multimodal documents. Using this infrastructure, we construct the IPO-Dataset, a large, section-structured, multimodal dataset covering more than 109,000 IPO filings and amendments from 1994 to 2026 and containing over 76,000 images. We establish structured evaluation tasks over extracted financial charts, including chart quality and misleadingness assessment. Our experiments show that state-of-the-art multimodal models often diverge from expert human judgments on these tasks, exposing alignment challenges in multimodal reasoning over long, real-world regulatory documents. Beyond benchmarking, the IPO-Dataset enables large-scale analysis of section-level textual variation and cross-industry differences in visual and textual disclosure practices. Our code, dataset, and website are publicly available under CC-BY-4.0.
>
---
#### [new 076] ChildEval: When large language models meet children's personalities
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ChildEval，用于评估大语言模型在儿童个性化对话中的表现，解决儿童偏好理解不足的问题，通过构建儿童人格数据集和评估协议提升模型的儿童中心能力。**

- **链接: [https://arxiv.org/pdf/2605.27805](https://arxiv.org/pdf/2605.27805)**

> **作者:** Yanyan Luo; Xue Han; Chunxu Zhao; Ruiqiao Bai; Yaxing Zhang; Qian Hu; Lijun Mei; Junlan Feng
>
> **备注:** 8 pages of main text (ACL Findings format), with references and appendix
>
> **摘要:** While LLMs enable personalized chatbots, their effectiveness in child-centered personalization remains unclear, as systematic evaluation of child-specific preferences is still lacking. To address this gap, we introduce ChildEval, a benchmark for evaluating LLMs' ability to infer and follow child-centered preferences in long-context conversations. ChildEval contains 29K synthesized persona profiles of children aged 3-6, providing relatively static background information. Each persona is associated with a child preference-which may align with, conflict with, or be independent of the persona-expressed either explicitly in a single sentence or implicitly through 6-10 turn dialogues. Explicit and implicit preferences are designed to reflect the same underlying preference but differ in expression, capturing dynamic aspects of preference expression rather than changes in the static persona. The benchmark spans five top-level and fourteen sub-level categories covering children's daily lives and development. We further propose fine-grained, child-centric evaluation protocols to systematically assess open-source LLMs. Experimental results demonstrate how different personalized representations affect LLM responses and suggest that finetuning on ChildEval can enhance child-centered performance. Our code and dataset are available at this https URL.
>
---
#### [new 077] Beyond One Path: Evaluating and Enhancing Divergent Thinking in Interactive LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于创意计算任务，旨在评估和提升交互式大语言模型的发散思维能力。针对现有评估方法的不足，提出MUTATE基准和ReDNA方法，增强模型在路径和动作层面的发散性。**

- **链接: [https://arxiv.org/pdf/2605.28465](https://arxiv.org/pdf/2605.28465)**

> **作者:** Jihyeong Park; Ingeol Baek; Jeonghyun Park; Hwanhee Lee
>
> **备注:** 28 pages, 16 figures, 19 tables
>
> **摘要:** Divergent thinking is a core dimension of creativity, yet existing evaluations of Large Language Models (LLMs) treat them as single-turn text generations, failing to capture how an agent reasons through iterative interaction. To address this, we introduce MUTATE, an interactive benchmark designed to evaluate agentic divergent thinking at two levels: path-level, where an agent discovers multiple alternative paths to the same goal, and action-level, where individual actions require non-typical, mechanism-shifting object uses. Unlike success-only evaluations, MUTATE scores both completed paths and off-path attempts, capturing divergent reasoning that conventional success rates discard. Our experiments with frontier LLMs reveal a structural blind spot in existing frameworks: when exposed to immediate convergence pressure, they tend to fall into immediate action fixation, failing to improve action-level divergence. To overcome this, we propose ReDNA, which separates unconstrained divergent candidate generation from convergent constraint selection. ReDNA significantly outperforms prior methods across both divergence levels and generalizes effectively to an external creativity environment. We also confirm its success stems from a qualitative enhancement of resilient divergent reasoning rather than simple environmental exploration.
>
---
#### [new 078] KSAFE-MM: A Multimodal Safety Benchmark via Localized Contextualization for Korean Cultural Risks
- **分类: cs.CL**

- **简介: 该论文属于多模态安全评估任务，旨在解决现有评估工具缺乏文化特定性和英语中心的问题。构建KSFAE-MM基准，涵盖通用与文化相关风险，评估模型安全性。**

- **链接: [https://arxiv.org/pdf/2605.28013](https://arxiv.org/pdf/2605.28013)**

> **作者:** Yongwoo Kim; Sojung An; Yunjin Park; Jungwon Yoon; Dujin Lee; HyunBeom Cho; Jaewon Lee; Wonhyuk Lee; Youngchol Kim; JeongYeop Kim; Donghyun Kim
>
> **摘要:** Multimodal Large Language Models (MLLMs) exacerbate safety risks by introducing vulnerabilities across multiple modalities, such as language and vision. Current MLLM safety evaluation tools, however, suffer from major limitations: 1) English-centric dataset construction, and 2) a focus on generic risks that are not tied to local cultural contexts. This paper introduces KSAFE-MM, a benchmark for Korean multimodal safety evaluation that covers both general safety risks and culture-specific vulnerabilities. KSAFE-MM consists of two parts, KSAFE-MM-G and KSAFE-MM-C. KSAFE-MM-G evaluates globally shared risks in Korean contexts through linguistic contextualization, which transforms generic safety queries into contextually grounded multimodal samples. KSAFE-MM-C targets culture-dependent MLLM safety vulnerabilities using localized visual queries derived from real-world contexts. It pairs these visual queries with jailbreak-style textual queries to cover multimodal safety risks involving cultural visual cues and malicious textual intent. Together, these components provide a general-to-local construction pipeline for evaluating both globally shared safety risks and culture-specific vulnerabilities. We evaluate 12 state-of-the-art MLLMs on KSAFE-MM and reveal that models exhibit greater vulnerability to culturally grounded attacks than to generic ones. Notably, jailbreaking strategies substantially amplify attack success rates, with ProgramExecution yielding up to 74.2% ASR compared to 13.4% for standard queries. Furthermore, we identify a systematic trade-off between safety and over-refusal, where models achieving low ASR tend to exhibit excessive refusal behavior on benign queries. These findings highlight the urgent need for culturally grounded safety evaluation beyond English-centric benchmarks.
>
---
#### [new 079] Escape the Language Prior: Mitigating Late-Stage Modality Collapse in Audio Reasoning via Modality-Aware Policy Optimization
- **分类: cs.CL**

- **简介: 该论文属于多模态推理任务，解决音频模型在长序列生成中因依赖文本先验导致的模态崩溃问题。提出MAPO框架，通过关注关键模态令牌和引入注意力损失，提升跨模态推理的准确性与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.27741](https://arxiv.org/pdf/2605.27741)**

> **作者:** Cihan Xiao; Yiwen Shao; Chenxing Li; Xiang He; Zhenwen Liang; Steve Yves; Sanjeev Khudanpur; Liefeng Bo
>
> **摘要:** Audio and omni-modal large language models exhibit impressive cross-modal reasoning capabilities. However, applying standard reinforcement learning post-training algorithms to these models exposes a critical structural vulnerability: methods like GRPO apply uniform policy gradients across all tokens, ignoring their unequal dependence on the non-text source modality. This exacerbates late-stage modality collapse during extended chain-of-thought generation, where models progressively abandon the primary source signal in favor of compressed textual priors, leading to confident but ungrounded hallucinations. To address this, we introduce Modality-Aware Policy Optimization (MAPO), a novel dual-branch reinforcement learning framework. First, MAPO dynamically concentrates the policy gradient on modality-critical tokens using a modality relevance mask, which is derived from the cross-modal differential entropy between an audio-ablated reference and the multimodal policy. Second, it integrates an auxiliary attention loss branch that applies a targeted, temporally scaled penalty to the model's internal attention distributions. This ensures the model actively sustains cross-modal grounding deep into the reasoning trace. Evaluations on complex audio reasoning benchmarks demonstrate that MAPO substantially improves long-horizon reasoning fidelity and multimodal instruction following, achieving highly competitive performance and setting new state-of-the-art results on several key benchmarks among open-weight models. By relying strictly on native statistical signals rather than domain-specific inductive biases, MAPO offers a promising foundation for mitigating epistemic collapse across diverse multimodal systems.
>
---
#### [new 080] StoryLens: Preference-Aligned Story Rewriting via Context-Aware Narrative Enrichment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于个性化故事重写任务，旨在提升读者偏好匹配度。通过引入上下文感知的叙事增强方法，提出STORYLENSWRITER模型，有效提升故事重写的质量与满意度。**

- **链接: [https://arxiv.org/pdf/2605.28073](https://arxiv.org/pdf/2605.28073)**

> **作者:** Hanwen Cui; Yuting Mei; Yuhang Fu; Dingyi Yang; Qin Jin
>
> **备注:** 16 pages, 7 figures, 15 tables
>
> **摘要:** Story rewriting aims to adapt existing narratives to diverse reader preferences while preserving plot consistency and narrative coherence. Unlike conventional work on style transfer, we argue that effective story rewriting demands context-aware narrative enrichment beyond surface-level stylistic adaptation. Our pilot human study shows that style adaptation alone provides only marginal gains in reader satisfaction (2.3%), while context-enhanced rewriting substantially improves user preference alignment (24.5%). Motivated by this, we introduce STORYLENSBENCH, a large-scale benchmark for preference-aligned story rewriting, comprising structured story books, multi-dimensional reader preference profiles, and ranked context-aware rewritten stories. Building on this benchmark, we propose STORYLENSEVAL, a reward model for estimating reader satisfaction over rewritten stories, and STORYLENSWRITER, a two-stage rewriting model combining supervised fine-tuning with GRPO-based reinforcement learning. We further establish a comprehensive evaluation framework covering fidelity, coherence, and reader satisfaction. Experimental results demonstrate that STORYLENSWRITER consistently outperforms strong generation and personalization baselines, highlighting the importance of context-aware narrative enrichment for personalized story rewriting.
>
---
#### [new 081] CIRF: Tokenizing Chain-of-Thoughts into Reusable Functional Units for Efficient Latent Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出CIRF框架，解决大模型推理效率问题。通过将思维链分解为可复用功能单元，提升推理准确性与效率。属于自然语言处理中的推理任务。**

- **链接: [https://arxiv.org/pdf/2605.28292](https://arxiv.org/pdf/2605.28292)**

> **作者:** Yukyung Lee; Yumeng Shen; Jinhyeong Park; Hyein Yang; Jun-Hyung Park
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** Implicit Chain-of-Thought (CoT) reduces the inference cost of large language models by internalizing the explicit rationales. However, existing approaches typically lack alignment with explicit rationales and adaptivity to example complexity. In this work, we propose CIRF (\textit{\underline{C}hain-of-thoughts \underline{I}nto \underline{R}eusable \underline{F}unctional units}), an implicit CoT framework that performs reasoning as a dynamic sequence of discrete functional tokens. CIRF assigns a functional token to each semantically coherent reasoning unit in explicit CoT traces. The model is then fine-tuned to autoregressively generate functional tokens and their optional results, followed by the final answer. This design aligns latent reasoning with a sequence of functional units, facilitating parallel training, explicit rationale alignment, and adaptive reasoning. Extensive experiments on mathematical, symbolic, and commonsense reasoning benchmarks show that CIRF provides a favorable accuracy-latency trade-off compared with state-of-the-art implicit CoT methods. Further analyses demonstrate that CIRF constructs distinct, interpretable functional tokens, leading to consistent performance improvements.
>
---
#### [new 082] From AR to Diffusion: Efficiently Adapting Large Language Models with Strictly Causal and Elastic Horizons
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，解决AR模型与扩散模型不兼容的问题。提出FLUID框架，实现高效适配，降低训练成本。**

- **链接: [https://arxiv.org/pdf/2605.27387](https://arxiv.org/pdf/2605.27387)**

> **作者:** Xiangyu Ma; Teng Xiao; Zuchao Li; Lefei Zhang
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Diffusion models promise efficient parallel text generation but rely on bidirectional attention, creating a structural mismatch with pre-trained Autoregressive (AR) models. This incompatibility precludes reusing robust AR priors, necessitating prohibitive pre-training from scratch. To bridge this gap, we propose FLUID, a framework that efficiently adapts AR backbones to the diffusion paradigm. By enforcing Strictly Causal Alignment, FLUID enables seamless initialization from standard GPT-style checkpoints, circumventing the need for massive pre-training. Furthermore, we introduce Elastic Horizons, an entropy-driven mechanism that dynamically modulates denoising strides based on local information density rather than fixed schedules. Experiments demonstrate that FLUID achieves state-of-the-art performance while reducing training costs by orders of magnitude, effectively reconciling established AR foundations with efficient parallel generation. Our code is available at this https URL.
>
---
#### [new 083] RAG-Coding: Enhancing LLM Medical Coding with Structured External Knowledge
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出RAG-Coding，用于提升医学编码的准确性与合规性。任务是自动化ICD-10-CM编码，解决传统方法精度不足的问题。通过整合外部知识源和多模型协作，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2605.27377](https://arxiv.org/pdf/2605.27377)**

> **作者:** Yidong Gan; David D. Nguyen; Yang Lin; Peter Zhong; Thanh Vu; Long Duong; Yuan-Fang Li
>
> **备注:** Additional experiments and analyses are in progress
>
> **摘要:** We present RAG-Coding, an agentic method for automated ICD-10-CM coding. RAG-Coding orchestrates four large language model (LLM) agents and grounds their coding decisions in external knowledge sources (e.g. the official coding tabular list and guidelines). By retrieving and cross-referencing relevant knowledge in these sources, the agents enhance coding accuracy and ensure clinical compliance. On the MDACE dataset, RAG-Coding outperforms the best LLM-based baseline by 8-13\% in micro-F1 and 2-8\% in macro-F1 across multiple LLM backbones. Compared to the state-of-the-art pretrained language model method, PLM-ICD, RAG-Coding exhibits higher micro recall (+11\%), while PLM-ICD exhibits higher micro precision (+6\%), yielding comparable micro- and macro-F1. Ablations show stepwise gains, highlighting the importance of incorporating external knowledge. We also release MDACE-2025, updating the original dataset with expert re-annotations with the latest 2025 ICD-10-CM guidelines. This update features more fine-grained code labels and enables evaluation against current clinical standards.
>
---
#### [new 084] ConRAG: Consensus-Driven Multi-View Retrieval for Multi-Hop Question Answering
- **分类: cs.CL**

- **简介: 该论文属于多跳问答任务，旨在提升复杂多跳问答的性能。提出ConRAG框架，通过多视角证据优化查询与语料库，显著提高准确率。**

- **链接: [https://arxiv.org/pdf/2605.28093](https://arxiv.org/pdf/2605.28093)**

> **作者:** Yikai Zhu; Kunfeng Chen; Qihuang Zhong; Juhua Liu; Bo Du
>
> **摘要:** Retrieval-augmented generation (RAG) has emerged as a promising paradigm for enhancing large language models (LLMs) on multi-hop question answering (QA), which requires reasoning over evidence from multiple documents. Current multi-hop RAG methods generally focus on either query-side task decomposition or corpus-side knowledge graph construction. Despite their progress, these methods still struggle to achieve satisfactory performance on complex multi-hop QA tasks. To this end, we propose ConRAG, a consensus-driven multi-view RAG framework that effectively boosts LLMs on complex multi-hop QA. The core of ConRAG is to systematically optimize both the query and corpus sides and to leverage multi-view evidence (relation, entity, and text signals) for more accurate retrieval. Extensive experiments on three multi-hop QA benchmarks show that ConRAG consistently outperforms all baselines by a clear margin, e.g., up to +26.9% average performance gains over vanilla RAG, and enables Gemma-4-31B to achieve a new state-of-the-art record on the challenging MuSiQue benchmark.
>
---
#### [new 085] SuperValid: Capability-Aligned OOD Validation for Generalizable Downstream Scaling
- **分类: cs.CL**

- **简介: 该论文属于模型训练优化任务，解决现有方法在泛化能力上的不足。提出SuperValid框架，通过能力对齐的OOD验证数据提升模型选择与缩放效果。**

- **链接: [https://arxiv.org/pdf/2605.28179](https://arxiv.org/pdf/2605.28179)**

> **作者:** Quanen Sun; Changxin Tian; Ke Shi; Cai Chen; Cunyin Peng; Jia Liu; Kunlong Chen; Zhiqiang Zhang
>
> **摘要:** Scaling laws guide large language model training by relating compute to cross-entropy loss, and recent work further extends them to predict downstream benchmark performance. However, prior approaches face generalization limitations from two aspects: focusing on benchmark-level performance introduces scenario-specific artifacts, while relying on IID validation loss fails to track capability improvements when training distributions vary. In this work, we argue that downstream scaling should be studied at the capability level, which captures shared skill factors across related tasks while abstracting away benchmark-specific noise. We propose SuperValid, a framework that synthesizes OOD (out-of-distribution), capability-aligned validation data by distilling core concepts from benchmarks within a capability domain and expanding them into diverse, knowledge-rich texts. Extensive experiments spanning 17 benchmarks grouped into 6 capability domains show that SuperValid loss exhibits strong and stable correlation with downstream performance across models of different architectures, scales, and training data distributions. As a training-free metric computable during training without benchmark evaluation, SuperValid enables effective model selection, early stopping, and scaling decisions.
>
---
#### [new 086] Retrieval, Reward, and Training Protocols: What Matters in Training Search Agents?
- **分类: cs.CL**

- **简介: 该论文属于搜索代理训练任务，旨在明确影响性能的关键因素。通过控制实验，研究数据覆盖、奖励机制和训练策略，提出有效训练指南。**

- **链接: [https://arxiv.org/pdf/2605.27881](https://arxiv.org/pdf/2605.27881)**

> **作者:** Yibo Zhao; Zichen Ding; Jiayi Wu; Zun Wang; Xiang Li
>
> **备注:** 18pages, 4 figures, and 15 tables
>
> **摘要:** Search agents powered by large language models can autonomously decompose queries, retrieve information, and synthesize answers through multi-step reasoning. However, the rapid growth of training methods has outpaced controlled comparison: existing works differ in retrieval corpora, reward designs, and training protocols, making it unclear what actually drives improvements. We present a controlled empirical study that isolates three under-explored dimensions of search agent training. First, we identify a critical data-coverage issue in the widely used Wikipedia 2018 corpus and show that correcting it alone yields larger gains than the differences between training algorithms. Second, we systematically compare outcome-based and process-based reward methods across three base models, finding that the simplest outcome-based approach achieves competitive or superior performance in most settings, and that process-level credit assignment can over-correct agent behavior. Third, we analyze training data diversity, off-policy data utilization, and search budget scaling, distilling practical guidelines for training effective search agents. Our code is available at this https URL.
>
---
#### [new 087] Beyond Input Understanding: Diagnosing Multilingual Mathematical Reasoning with Directed Acyclic Trace Graphs
- **分类: cs.CL**

- **简介: 该论文属于数学推理任务，旨在解决多语言模型在非英语环境下推理性能下降的问题。通过引入DATG框架分析语言对推理的影响，并提出改进方法提升低资源语言的推理效果。**

- **链接: [https://arxiv.org/pdf/2605.27715](https://arxiv.org/pdf/2605.27715)**

> **作者:** Jiaqiao Zhang; Zhoujun Li; Raoyuan Zhao; Jian Lan; Thomas Seidl; Michael A. Hedderich; Hinrich Schütze; Yihong Liu
>
> **备注:** preprint
>
> **摘要:** Large reasoning models (LRMs) achieve strong mathematical reasoning performance in English, but remain much less reliable in many low- and medium-resource languages. This gap is often explained as a failure to understand non-English problem statements. We show that this view is incomplete: even when the problem is given in English, controlling the model's reasoning language can substantially reduce accuracy, suggesting that language also affects reasoning execution itself. To study this effect, we introduce DATG, a Directed Acyclic Trace Graph framework that maps reasoning traces to language-independent mathematical anchors and dependencies. This allows us to align target-language traces with reference DAGs and measure whether they cover required mathematical nodes, respect dependency edges, and avoid harmful mathematical actions. Experiments on the Qwen3 series across 12 languages show that non-English reasoning often suffers from reduced anchor coverage and weaker dependency fidelity, especially in low-resource languages. Motivated by this diagnosis, we propose Loop-Retry and Formula-Retry, two simple test-time controls targeting DATG-exposed failure modes, and show that they consistently improve target-language reasoning performance in low-resource languages.
>
---
#### [new 088] Can Hallucinations Be Useful? Solving Multi-Hop Questions With SLMs By Chaining System-I/II Reasoning
- **分类: cs.CL**

- **简介: 该论文属于多步问答任务，旨在解决SLMs因幻觉导致的推理错误问题。通过先回答后推理的框架，结合System-I和System-II思维，提升问答准确性。**

- **链接: [https://arxiv.org/pdf/2605.27596](https://arxiv.org/pdf/2605.27596)**

> **作者:** Saptarshi Sengupta; Suhang Wang
>
> **摘要:** Recently, there has been increased interest in Small Language Models (SLMs), which are fast, show good performance, and have lower hardware demands than large language models (LLMs). However, SLMs hallucinate more frequently than LLMs, impacting their ability to solve complex multi-step reasoning problems as early mistakes cascade to the final response. To address this, existing works think-first followed by iterative retrieval to reduce hallucination. We argue that the think-first strategy is not always necessary as we find that: (i) SLMs are often accurately confident in their initial answer and, (ii) hallucinations can actually be beneficial for honing in on the true answer. As such, we position our work as an inversion of this strategy, i.e., answer first-reason later. We propose a cognitively-inspired framework where the model is first allowed to quickly answer the question (System-I (zero-shot)) and then resorts to deeper thinking (System-II) based on evidence retrieved from a knowledge source using the initial hypothesis. By combining System-I and System-II style thinking, we show that our method can outperform prior work that takes the traditional think-first route on various multi-step question-answering benchmarks.
>
---
#### [new 089] LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于法律推理任务，旨在解决法律知识结构化与推理透明性问题。提出LegalGraphRAG框架，通过分层图结构和多智能体系统提升法律分析的准确性与可信度。**

- **链接: [https://arxiv.org/pdf/2605.28120](https://arxiv.org/pdf/2605.28120)**

> **作者:** Zerui Chen; Qinggang Zhang; Zhishang Xiang; Zhimin Wei; Linfeng Gao; Xiao Huang; Zhihong Zhang; Jinsong Su
>
> **备注:** 30 pages, 18 figures, ACL 2026 Main Conference. Project page: this https URL
>
> **摘要:** Graph-based Retrieval-Augmented Generation (GraphRAG) advances flat document retrieval by structuring knowledge as relational graphs, enabling more coherent and effective reasoning. However, applying it to specific domains like legal reasoning faces critical challenges. (i) Legal corpora are heterogeneous, containing multi-granular knowledge from cases, articles and interpretations. A flat knowledge graph cannot adequately differentiate between factual details, applied rules, and abstract principles, limiting accurate retrieval. (ii) Reliable legal judgment demands transparent, evidence-based reasoning. Traditional RAG passes retrieved context directly to an LLM without verification, resulting in opaque, error-prone reasoning. To this end, we propose LegalGraphRAG, a framework designed for reliable legal reasoning. Our approach introduces two core components: a hierarchical legal graph that hierarchically organizes legal sources to enable retrieval at appropriate abstraction levels, and a multi-agent system for reliable legal reasoning, where a Researcher retrieves candidate evidence, an Auditor rigorously verifies its validity against source documents, and an Adjudicator synthesizes the set of verified evidence to render a final judgment. Extensive experiments show that LegalGraphRAG achieves the state-of-the-art performance, outperforming existing GraphRAG baselines in accurate and trustworthy legal analysis. Our code, datasets and implementation details are available at this https URL.
>
---
#### [new 090] GUI-CIDER: Mid-training GUI Agents via Causal Internalization and Density-aware Exemplar Reselection
- **分类: cs.CL**

- **简介: 该论文属于GUI代理任务，旨在解决代理缺乏GUI操作知识的问题。提出GUI-CIDER方法，通过因果内化和密度感知样本重选，在训练中显式学习GUI知识，提升任务完成效果。**

- **链接: [https://arxiv.org/pdf/2605.28534](https://arxiv.org/pdf/2605.28534)**

> **作者:** Zheng Wu; Chengcheng Han; Zhengxi Lu; Tianjie Ju; Yanyu Chen; Qi Gu; Xunliang Cai; Zhuosheng Zhang
>
> **摘要:** Despite the rapid progress of multimodal large language models in building Graphical User Interface (GUI) agents, their real-world task completion is fundamentally bottlenecked by a lack of world knowledge about GUI operations. Existing solutions typically rely on expensive multi-agent scaffolding or conventional post-training paradigms, such as Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). However, post-training only allows agents to implicitly absorb world knowledge through action annotations or reward signals, leading to inefficient trajectory memorization rather than genuine comprehension. Therefore, an approach that enables explicit learning of this knowledge is imperative. To this end, we propose GUI-CIDER, a mid-training method that explicitly internalizes GUI world knowledge through Causal Internalization and Density-aware Exemplar Reselection. GUI-CIDER operates in three stages: (1) data synthesis, which distills static planning and dynamic causal knowledge from GUI trajectories into text; (2) exemplar reselection, which filters the corpus by rewarding causal structures and penalizing semantic redundancy; and (3) mid-training, where the refined data is used to embed the acquired knowledge. Extensive experiments on two GUI knowledge benchmarks and three task completion benchmarks demonstrate that GUI-CIDER consistently improves both the agent's understanding of GUI operations and its task success this http URL codes are available at this https URL.
>
---
#### [new 091] MemGuard: Preventing Memory Contamination in Long-Term Memory-Augmented Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决长时记忆增强模型中的记忆污染问题。通过引入MemGuard框架，区分记忆类型，提升记忆可靠性。**

- **链接: [https://arxiv.org/pdf/2605.28009](https://arxiv.org/pdf/2605.28009)**

> **作者:** Hyeonjeong Ha; Jeonghwan Kim; Cheng Qian; Jiayu Liu; William M. Campbell; Yue Wu; Yuji Zhang; Kathleen McKeown; Dilek Hakkani-Tur; Heng Ji
>
> **摘要:** Memory-augmented large language models extend reasoning beyond a fixed context window by maintaining long-term memory across interactions. However, existing memory systems often collapse stable user facts, episodic events, and behavioral rules into a shared space, allowing functionally distinct memories to be retrieved and used as interchangeable evidence. We identify this failure mode as heterogeneous memory contamination, where context-specific events become overgeneralized claims, or semantically relevant but functionally incompatible memories mislead generation. To this end, we introduce MemGuard, a type-aware memory framework that preserves functional memory boundaries during memory construction and retrieval. It assigns each memory an explicit functional role at write time, maintains relations across type-isolated memories, and selectively composes evidence only from necessary memory types, reducing contamination from irrelevant or functionally incompatible evidence. Across hallucination and long-horizon conversation benchmarks, MemGuard improves memory reliability by up to 28.27% while retrieving up to 5.8x fewer memory tokens than prior methods. These results suggest that reliable long-term reasoning depends on principled organization and selective use of heterogeneous memory.
>
---
#### [new 092] Risk-aware Selective Prompting for Hallucination Mitigation in Large Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文属于视觉语言模型 hallucination 问题研究，旨在减少模型幻觉。通过分析验证提示的风险，提出一种基于不确定性的选择性提示方法。**

- **链接: [https://arxiv.org/pdf/2605.28123](https://arxiv.org/pdf/2605.28123)**

> **作者:** Yuang Huang; Yafeng Zhang; Yu Zilan
>
> **备注:** 7 pages, 1 figures, submitted to ACL ARR 2026 May (EMNLP)
>
> **摘要:** Prompt-based verification is widely used to mitigate hallucinations in large vision-language models (LVLMs), yet when it helps remains poorly understood. We systematically study verification prompting across two representative LVLM architectures and hallucination benchmarks, and find that it is a risk-bearing intervention: its corrections increase with input difficulty, while newly introduced errors persist across difficulty levels. As a result, always-on prompting helps on hard inputs but offers little benefit -- and can harm -- easier ones. Our analysis further shows that this behavior is associated with a conservative output shift. Verification prompts redistribute attention from visual tokens toward instruction tokens and induce a distinct middle-layer entropy pattern absent in a neutral-prompt control, suggesting instruction-conditioned attention redistribution rather than uniformly improved visual grounding. Motivated by this input-dependent risk, we propose Risk-aware Selective Prompting (RSP), a training-free approach that uses pre-generation uncertainty signals to trigger verification selectively. RSP mitigates the degradation of always-on prompting while preserving baseline performance, and reveals that effective selection signals vary across architectures.
>
---
#### [new 093] Prompting Is All You Need: Multi-view Prompting Large Language Models for Aspect-Based Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在解决少样本提示与微调模型间的性能差距。通过多视角提示方法，提升模型效果并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2605.28058](https://arxiv.org/pdf/2605.28058)**

> **作者:** Nils Constantin Hellwig; Niklas Donhauser; Jakob Fehle; Udo Kruschwitz; Christian Wolff
>
> **摘要:** Recent work explored the capabilities of Large Language Models (LLMs) in Aspect-Based Sentiment Analysis (ABSA) through few-shot prompting, requiring substantially fewer annotated examples while achieving notable improvements over zero-shot baselines. However, a performance gap remained compared to models fine-tuned on hundreds of examples, and the computational costs of LLM inference present practical barriers to deployment. We introduce LLM-based Multi-View Prompting (LLM-MvP), which adapts the multi-view principle of considering multiple element orderings to LLM prompting. By combining schema-constrained decoding with a context-free grammar and prefix batching, LLM-MvP achieves performance competitive or superior to fine-tuned approaches while substantially reducing computational overhead. Extensive experiments across five benchmark datasets demonstrate that LLM-MvP closes the gap between few-shot prompting and fine-tuned models, offering a practical and efficient solution for ABSA.
>
---
#### [new 094] PrunePath: Towards Highly Structured Sparse Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PrunePath，解决语言模型稀疏化效率问题，通过结构化剪枝提升推理性能，实现高效部署。**

- **链接: [https://arxiv.org/pdf/2605.28283](https://arxiv.org/pdf/2605.28283)**

> **作者:** Zhexuan Gu; Zixun Fu; Yancheng Yuan
>
> **摘要:** Feed-forward networks (FFNs) dominate the parameter count and computation of modern language models, yet existing pruning methods often struggle to convert sparsity into hardware-friendly inference efficiency gains. We introduce \textbf{PrunePath}, a budget-adaptive structured sparsification framework for FFN layers. Built on MoEfication, PrunePath replaces independent expert-wise thresholding with a softmax-normalized routing distribution and activates important experts under a cumulative-mass threshold. This formulation imposes a token-level probability budget, enabling adaptive expert counts and a direct inference-time sparsity knob from a single checkpoint. Across NLU, NLG, and instruction-tuning evaluations, PrunePath achieves a favorable sparsity--performance trade-off compared with existing static pruning and MoEfication-based methods. We further implement Triton kernels for KV-cache decoding to translate the resulting structured sparsity into practical memory savings and measurable decoding-speed improvements. These results demonstrate the superior performance of PrunePath for building highly sparse, deployment-friendly large language models.
>
---
#### [new 095] Argument Quality Assessment with Large Language Models: A Pairwise Bradley-Terry Approach
- **分类: cs.CL**

- **简介: 该论文属于argument quality assessment任务，旨在评估大语言模型在论证质量判断上的表现。研究通过 Bradley-Terry 模型比较不同模型的推理能力，探索其与人类专家判断的一致性。**

- **链接: [https://arxiv.org/pdf/2605.28313](https://arxiv.org/pdf/2605.28313)**

> **作者:** Nicolás Benjamín Ocampo; Agnes Paullate Nyiranziza; Davide Ceolin
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in tasks related to reasoning and judgment. However, assessing the quality of arguments requires a rigorous evaluation. We investigate the extent to which LLMs can effectively perform this task. We tested 12 open-weight LLMs of different sizes and families under zero-shot, few-shot, and chain-of-thought to approximate expert pairwise comparisons of argument quality across three dimensions-logical, rhetorical, and dialectic-and used these comparisons in a Bradley-Terry model to infer latent strength scores and derive a ranking of arguments. Our insights show that LLMs have promising but moderate correlation with human expert judgments, with Llama-70B obtaining the strongest alignment, reaching moderate Cohen's $\kappa$ = 0.493 and moderate correlations with Bradley-Terry scores derived from these annotations (Kendall, Pearson, and Spearman: 0.327-0.477). Other LLMs exhibit weak, moderate, or high alignment with Llama-70B while achieving comparable results against human experts, suggesting partial but complementary understanding of underlying quality dimensions despite differences in model size and family. Moreover, LLM predictions are stable across trial runs, with fewer than 7.75\% of cases yielding different labels. Remaining variability is handled via majority voting and few-shot prompting for large-size models.
>
---
#### [new 096] VLMs May Not Globally Enhance Human Alignment over LLMs During Natural Reading
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文属于自然语言处理任务，探讨VLM与LLM在自然阅读中的人类对齐差异。通过对比实验，发现多模态预训练未必全局提升对齐效果，但对视觉语义强的句子有优势。**

- **链接: [https://arxiv.org/pdf/2605.28818](https://arxiv.org/pdf/2605.28818)**

> **作者:** Jinzhou Wu; Zhengwu Ma; Jixing Li; Baoping Tang; Zitong Lu
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** Large language models (LLMs) have become increasingly useful computational models of human language processing, but it remains unclear whether vision-language learning makes text representations more human-like during natural reading. Here, we address this question by comparing tightly matched LLM and vision-language model (VLM) pairs under a strictly text-only setting, allowing us to isolate the effect of multimodal training history from online visual input or cross-modal fusion. We evaluate model alignment with a human natural-reading dataset that includes whole-cortex fMRI responses and synchronized eye-tracking saccades. Our findings demonstrate that multimodal pretraining may not confer a uniform, global advantage in human alignment during natural reading, indicating that language-internal representations remain the key factor for modeling human text processing. However, the VLM advantage could emerge more selectively when sentences contain stronger visual semantic content, with converging evidence from both fMRI and eye-movement alignments. Together, our findings provide a controlled in silico framework for testing how visual learning history shapes model-human alignment of language processing, suggesting that multimodal pretraining contributes selectively rather than globally to human-like language representations during natural reading.
>
---
#### [new 097] PAST2HARM: A Simple Adaptive Past Tense Attack for Jailbreaking Multimodal AI
- **分类: cs.CL**

- **简介: 该论文属于AI安全任务，旨在破解多模态AI系统的防御机制。提出PAST2HARM攻击框架，通过过去时改写绕过拒绝训练，验证了多模态模型的安全漏洞。**

- **链接: [https://arxiv.org/pdf/2605.27545](https://arxiv.org/pdf/2605.27545)**

> **作者:** Snehasis Mukhopadhyay
>
> **摘要:** Jailbreak attacks on multimodal AI systems remain underexplored, even though unsafe image generation can have more severe consequences than unsafe text and current defenses are relatively immature. We introduce PAST2HARM, a simple yet effective adaptive jailbreak framework that bypasses refusal training in state of the art multimodal text to image models. Building on prior findings that past tense reformulations can evade safeguards, PAST2HARM systematically exploits this vulnerability in multimodal generative AI. We characterize the attack along two dimensions. First, breadth: through temporal deepening, the framework incrementally strengthens historical anchoring and archival cues, eroding refusal boundaries across models with varying alignment strength. Second, depth: via iterative escalation after initial compliance, we probe the upper bound of harmful generation, measuring severity using a scalar severity jailbreak metric evaluated by a language model acting as a judge. We find that mid conversation turns form peak vulnerability windows, where harmfulness increases before plateauing and eventually undergoing semantic inversion. We evaluate PAST2HARM on three models Gemini Nano Banana Pro, GPT Image 2, and SD XL achieving attack success rates of 83 percent, 67 percent, and 100 percent in a black box, gradient free setting. Adversarial prompts also transfer across models, with cross model success rates above 50 percent. The attack elicits diverse harmful outputs, including explicit sexual content, political disinformation, historical denial narratives, hate speech, and self harm glorification. We further release a curated benchmark of prompts, reformulations, and outputs as a resource for red teaming and alignment. Our results expose fundamental brittleness in current safeguards and highlight the need for stronger multimodal safety training.
>
---
#### [new 098] PrionNER: A Named Entity Recognition Dataset for Prion Disease Biomedical Literature
- **分类: cs.CL**

- **简介: 该论文提出PrionNER，一个用于普里昂病生物医学文献的命名实体识别数据集，解决罕见病信息提取难题。**

- **链接: [https://arxiv.org/pdf/2605.28375](https://arxiv.org/pdf/2605.28375)**

> **作者:** An Dao; Nhan Ly; Thao Tran; Yuji Matsumoto; Akiko Aizawa
>
> **备注:** 29 pages, 5 figures, accepted at ACL 25th Workshop on Biomedical Language Processing (BioNLP 2026)
>
> **摘要:** Prion diseases are rare, rapidly progressive, and fatal neurodegenerative disorders that remain difficult to diagnose, particularly in their early stages because of nonspecific clinical presentations. However, to our knowledge, there is no publicly available prion-disease-focused dataset designed to capture a broad range of clinically relevant entities from the biomedical literature. We introduce PrionNER, a manually annotated named entity recognition dataset for prion disease clinical information in PubMed abstracts. The current release comprises 317 abstracts, 2,943 sentences, and 6,955 text-bound entity annotations spanning 15 coarse-grained and 31 fine-grained clinically oriented entity types covering diseases, symptoms, diagnostics, findings, anatomy, treatments, and temporal and statistical evidence. Inter-annotator agreement reaches 81.78 exact-match F1, indicating strong annotation consistency. We benchmark supervised BERT baselines, W2NER, and zero-shot extractors on PrionNER. W2NER is the strongest supervised model, and Gemma-4-31B is the strongest zero-shot model, but the benchmark remains challenging, especially for structurally complex mentions and fine-grained clinically adjacent label distinctions. PrionNER provides a clinically grounded benchmark for prion-disease information extraction and supports research on rare-disease biomedical NLP under low-resource, fine-grained, and non-flat extraction conditions. The dataset, annotation guidelines, and evaluation scripts are available at this https URL.
>
---
#### [new 099] Sense Representations Are Inducible Interfaces
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ACROS方法，将显式语义表示引入预训练语言模型，解决语义消歧、词义操控和跨语言适应问题，无需重新训练模型。**

- **链接: [https://arxiv.org/pdf/2605.28669](https://arxiv.org/pdf/2605.28669)**

> **作者:** Jan Christian Blaise Cruz; Alham Fikri Aji
>
> **备注:** this https URL
>
> **摘要:** Sense representations (explicit, per-token meaning decompositions) are useful for disambiguation, steering, and cross-lingual alignment, but existing approaches require models to be pretrained with sense structure baked in. We introduce ACROS, which induces an explicit sense pathway into a frozen pretrained decoder LM through a gated residual addition. On SmolLM2-360M, ACROS preserves base LM quality while supporting three uses of the same induced variables: zero-shot word-sense disambiguation (64.95 F1 on Raganato ALL, competitive with the WordNet first-sense heuristic), low-KL lexical steering across 5,161 CoInCo cases where a simple non-oracle proxy recovers about 90% of positive shifts, and SENSIA cross-lingual adaptation to four languages (mean R@1 0.988, target FLORES PPL 7.94). ACROS makes sense representations an inducible interface for ordinary pretrained LMs.
>
---
#### [new 100] GRADE: Generalizable Reasoning-Aware Dialogue Evaluation for AI Tutors
- **分类: cs.CL**

- **简介: 该论文属于AI辅导系统评估任务，旨在提升对AI tutor回答的多维评价能力。通过GRADE框架，研究不同模型和方法在教学对话中的表现，优化评估效果。**

- **链接: [https://arxiv.org/pdf/2605.27866](https://arxiv.org/pdf/2605.27866)**

> **作者:** Parth Bhalerao; Jeromy Chang; David Chou; Oana Ignat
>
> **备注:** 16 pages, 7 figures
>
> **摘要:** Evaluating AI tutor responses requires more than factual correctness: tutors must identify mistakes, locate errors, provide guidance, and offer actionable next steps. We present GRADE, a systematic study of open-source models for pedagogical ability assessment in student-tutor dialogues. Building on the BEA 2025 TutorMind setting, we evaluate 120 configurations across five language models, zero-shot inference, LoRA fine-tuning, synthetic augmentation, CoT+Reasoning, and single-task versus multitask formulations. Gemma3-12B performs best for single-task evaluation, while Gemma3-27B in 8-bit precision is more reliable for multitask prediction. We find that augmentation helps models that struggle with the original data, verification adds limited gains despite higher cost, and CoT+Reasoning is more useful for synthetic data generation than direct classification. We further show that LoRA fine-tuning on structured classification objectives interferes with instruction-following behavior under thinking mode, redirecting generation away from the required evaluation format. Carbon analysis shows that model choice and reasoning mode substantially affect emissions. Overall, GRADE shows that carefully selected open-source LoRA pipelines can match or surpass proprietary and ensemble-based systems on key pedagogical dimensions, with code and data available at this https URL.
>
---
#### [new 101] Rethinking Memory as Continuously Evolving Connectivity
- **分类: cs.CL; cs.AI; cs.LG; cs.MA; cs.MM**

- **简介: 该论文属于人工智能任务，解决记忆在动态环境中的适应性问题。提出FluxMem框架，通过动态更新记忆图结构提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.28773](https://arxiv.org/pdf/2605.28773)**

> **作者:** Jizhan Fang; Buqiang Xu; Zhixian Wang; Haoliang Cao; Xinle Deng; Baohua Dong; Hangcheng Zhu; Ruohui Huang; Gang Yu; Ying Wei; Guozhou Zheng; Feiyu Xiong; Haofen Wang; Huajun Chen; Ningyu Zhang
>
> **备注:** Ongoing work
>
> **摘要:** Existing memory-augmented LLM agents often treat memory as a static repository with pre-defined representations and fixed retrieval pipelines, which is brittle in dynamic agentic environments where feedback, task variation, and heterogeneous signals continuously reshape what should be remembered and how it should be connected. To address this, we propose FluxMem, a connectivity-evolving memory framework that models memory as a heterogeneous graph and progressively refines its topology through three stages: initial connection formation, feedback-driven refinement, and long-term consolidation. During execution, FluxMem repairs missing links, prunes interference, aligns abstraction granularity, and distills recurrent successful trajectories into reusable procedural circuits, guided by one metric for memory generalizability and evolutionary maturity. Across three fundamentally distinct benchmarks including LoCoMo, Mind2Web, and GAIA, FluxMem achieves consistent state-of-the-art performance, demonstrating strong adaptation and generalization in complex agentic environments. The code will be open-sourced in this https URL.
>
---
#### [new 102] The Future of Facts: Tracing the Factual Generation-Verification Gap
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型在事实生成与验证间的差距，属于自然语言处理任务。旨在解决模型生成事实可靠性不足的问题，通过分析训练阶段的动态，揭示验证能力优于生成能力的现象。**

- **链接: [https://arxiv.org/pdf/2605.27564](https://arxiv.org/pdf/2605.27564)**

> **作者:** Tim R. Davidson; Anja Surina; Caglar Gulcehre
>
> **备注:** Code for this project is available at this https URL , blog post at this https URL
>
> **摘要:** Language models are becoming the default interface to factual knowledge, yet they often verify outputs more reliably than they generate them. This generation-verification gap (GV-gap) underlies many recent advances in self-improvement and reasoning, but its dynamics on factual knowledge specifically remain poorly understood. We focus on the training mechanisms underlying factual GV-gaps, distinguishing them from their computational and aesthetic counterparts. We trace generation and verification capabilities through three training phases (acquisition, continual learning, and updating) across four open-source model families at two scales each. Three findings recur across models: (i) verification is consistently learned before generation; (ii) verification is more robust to continual learning than generation; and (iii) factual updates can leave models in a "multi-verse" state, simultaneously verifying both old and new answers as correct. Natural experiments on frontier models reproduce these dynamics at scale and reveal residual verification biases on well-covered facts.
>
---
#### [new 103] Playing with Words, Improving with Rewards: Training Language Models for Creative Association
- **分类: cs.CL**

- **简介: 该论文属于语言模型训练任务，旨在提升模型的创造力。通过Codenames游戏训练模型，使用RLVR方法解决创造力评估主观性问题，验证了不同规模模型在创造力与推理间的平衡。**

- **链接: [https://arxiv.org/pdf/2605.27832](https://arxiv.org/pdf/2605.27832)**

> **作者:** Vijeta Deshpande; Namrata Shivagunde; Sherin Muckatira; Hadrien Glaude; Mikhail Gronas; Claire Stevenson; Roger Beaty; Anna Rumshisky
>
> **摘要:** Large Language Models (LLMs) are being applied to increasingly difficult problems and use cases. To navigate their vast solution spaces effectively, LLMs need to be creative. Yet the subjective nature of creativity and the limits of human judgment make training LLMs for creativity especially challenging. As a solution, we train LLMs on Codenames, a word-association game that exercises the two central axes of creativity, divergent and convergent thinking, while yielding objectively verifiable outcomes. This verifiability lets us bypass human judgment and train with Reinforcement Learning with Verifiable Rewards (RLVR). We train Qwen3-1.7B, 4B, and 8B models and evaluate them on ten creativity and four reasoning benchmarks. We find that the precision-diversity trade-off is scale-dependent: the 8B model prioritizes creativity over precision, while the 1.7B and 4B models gain reasoning precision at the cost of creativity. Concretely, the 8B model shows modest but consistent creativity gains (8 of 10 benchmarks) with only minor reasoning degradation, whereas the smaller models achieve substantial gains on reasoning tasks. Our study presents a scalable and effective solution to train LLMs for creativity.
>
---
#### [new 104] The Fragility of Chain-of-Thought Monitoring Across Typologically Diverse Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型安全任务，研究CoT监控在多语言中的可靠性问题。通过大规模实验发现CoT监控在多种语言中存在严重不可靠性，提出需改进监控技术。**

- **链接: [https://arxiv.org/pdf/2605.27901](https://arxiv.org/pdf/2605.27901)**

> **作者:** Eric Onyame; Runtao Zhou; Kowshik Thopalli; Bhavya Kailkhura; Chirag Agarwal
>
> **摘要:** Chain-of-thought (CoT) monitoring has been proposed as a promising safety mechanism for detecting misaligned behavior in large language models. However, its reliability remains largely unexplored beyond English and across diverse model families. We present the first large-scale evaluation of CoT monitorability across 13 diverse languages and seven frontier model families, comprising 16 models. Using adversarial-hint evaluations that require explicit intermediate computation, together with analysis of internal answer-token probabilities, we consistently find CoT unfaithfulness across languages and hint types, with an average rate of 95.9\% across 8B--120B parameter models. We find that frontier models systematically engage in strategic manipulation, including answer-switching, post-hoc rationalization, and procedural exploitation of hints, making external monitors struggle to detect deception. We show that frontier models often commit to the misaligned cue in their latent activations within the first 15\% of generation, even when the CoT appears faithful. Surprisingly, these deceptive patterns remain 100\% in low-resource languages, revealing fundamental limitations in current CoT-based oversight. Our results reveal that CoT monitoring is fundamentally fragile under linguistic distribution shift, providing a substantially weaker safety signal than what English-only studies suggest. These findings underscore an urgent need to develop robust CoT monitors and to accelerate research into white-box monitoring techniques, especially to improve CoT monitorability in mid- and low-resource languages. Our code is available \href{this https URL}{\textcolor{blue}{here}}.
>
---
#### [new 105] Integrated and Cross-Architecture Interpretation of LLM Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型解释任务，旨在解决LLM推理过程不透明的问题。通过提出IAR框架，结合多种方法识别关键推理标记，提升对模型推理机制的理解。**

- **链接: [https://arxiv.org/pdf/2605.28006](https://arxiv.org/pdf/2605.28006)**

> **作者:** Leonardo Matthew Yauw; Wei-Bin Kou; Yujiu Yang
>
> **摘要:** Understanding how LLMs reason is hindered by a practical asymmetry: while their generated outputs are observable, the underlying reasoning patterns remain opaque. Relying on single probes, such as Mutual Information Peak (MIP) or Deep-Thinking Ratio (DTR), risks underestimating the genuine inferential structure. To response this deficiency, we present an Integrated, cross-Architecture Reasoning (IAR) framework, designed to provide a unified approach to LLM reasoning interpretability. Specifically, we first propose to use bandwidth-calibrated MIP coupled with Tukey IQR peak-detection to isolate reasoning-crucial tokens at the output layer. Second, we performed an overlap analysis between MIP-picked tokens and DTR-deep tokens to trace the cross-layer trajectories of those tokens. This also discloses whether reasoning-crucial tokens are computation-intensive as well, further facilitating to understand how reasoning patterns evolve across model layers. Finally, we apply a Jaccard stability metric over multi-domain problems to verify if the MIP-identified tokens are reasoning quality-guaranteed. Extensive experiments on three models (Qwen-7B, Qwen-14B, and Llama-8B) across four domains (mathematics, code, logic, and common sense) demonstrate IAR's generalizable interpretation capabilities across architectures.
>
---
#### [new 106] When Confidence Misleads: Suffix Anchoring and Anchor-Proximity Confidence Modulation for Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，解决非自回归解码中置信度误导导致的生成不完整问题，提出后缀锚定与置信度调制方法提升解码质量。**

- **链接: [https://arxiv.org/pdf/2605.28181](https://arxiv.org/pdf/2605.28181)**

> **作者:** Jungwon Park; Jimyeong Kim; Jungmin Ko; Nojun Kwak; Wonjong Rhee
>
> **备注:** Preprint
>
> **摘要:** Diffusion language models decode text by iteratively denoising masked token sequences, making the choice of which positions to decode a central inference-time decision. Most training-free decoding strategies use model confidence for position selection, assuming that high-confidence positions are ready to be decoded. In this work, we revisit this assumption by studying when confidence misleads fully non-autoregressive (fully non-AR) decoding. EOT tokens can receive high confidence and cause incomplete generation; inserting a suffix anchor can mitigate this issue but introduces local overconfidence near the anchor, causing anchor-adjacent tokens to be decoded too early. To address these issues, we propose Suffix-Anchored Confidence Modulation, a simple training-free method that inserts a short suffix anchor to encourage response completion and modulates confidence near the anchor according to decoding progress. This preserves the response-completion benefit of suffix anchoring while reducing premature decoding of anchor-adjacent tokens. Across text-only reasoning, vision-language reasoning, and code-generation benchmarks, our method consistently improves confidence-based fully non-AR decoding, outperforms explicit EOT suppression, and preserves the parallel decoding advantage of fully non-AR generation.
>
---
#### [new 107] BenGER: Benchmarking LLM Systems on Subsumption-Based Legal Reasoning in German Law
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BenGER数据集，用于评估大语言模型在德国法律中的子类推理能力。任务属于法律推理领域，解决如何有效评测LLM在法律场景下的表现问题，并通过实验对比不同模型与人类表现。**

- **链接: [https://arxiv.org/pdf/2605.28183](https://arxiv.org/pdf/2605.28183)**

> **作者:** Sebastian Nagl; Ann-Kristin Mayrhofer; Martin Heidebach; Aleyna Koçak; Anne Zettelmeier; Elly Breu; Angelina Greiner; Sofija Milijas; Matthias Grabmair
>
> **备注:** Pre-Print
>
> **摘要:** We introduce the BenGER (Benchmark for German Law) dataset for evaluating LLM systems on subsumption-based legal reasoning in German law. The BenGER dataset consists of three components: 596 exam-style free-text legal case tasks across multiple levels of legal education and 531 short doctrinal reasoning tasks. We evaluate 12 contemporary LLM systems -- closed flagship, efficiency-oriented, and open-weight -- across automatic and judge-based metrics. On a controlled validation subset of timed human-written solutions under both unaided and human--AI co-creation conditions, we contextualise model performance against these human baselines. We introduce a rubric-aligned LLM-as-a-Judge framework cross-validated against a multi-rater human-grading protocol (three blind reviews plus one author-informed creator review per solution). Our results show that replacing a blind human reviewer with the LLM judge degrades agreement with the full human pool no more than removing that reviewer altogether (Calderon r=0.96 vs.~r=0.96, matched n=30), that closed-flagship systems lead the leaderboard across all corpora, and that human--AI co-creation substantially outperforms unaided human work.
>
---
#### [new 108] LCO: LLM-based Constraint Optimization for Safer Agentic LLMs in Real-world Tasks
- **分类: cs.CL**

- **简介: 该论文属于安全增强任务，旨在解决LLM在实际任务中因过度优化导致的有害副作用问题。提出LCO框架，通过约束优化提升安全性，同时保持任务性能。**

- **链接: [https://arxiv.org/pdf/2605.27375](https://arxiv.org/pdf/2605.27375)**

> **作者:** Jiayong Wan; Jiawei Chen; Zhaoxia Yin; Liu Shuyuan; Hang Su
>
> **摘要:** Large Language Models (LLMs) are increasingly acting as autonomous agents, but their continuous interaction with the environment can lead to in-context reward hacking (ICRH), a phenomenon where LLMs iteratively optimize their behavior to maximize proxy objectives, inadvertently producing harmful side effects. Existing defense methods are insufficient to address this risk, as ICRH arises not from adversarial inputs but from the model's own over-optimization. To mitigate this issue, we propose \textbf{LLM-based Constraint Optimization (LCO)}, a framework that effectively reduces ICRH without model fine-tuning. LCO consists of two modules: \textit{self-thought module}, which guides the LLM to proactively deliberate and integrate potential safety constraints before execution; and \textit{evolutionary sampling module}, which employs LLM-based crossover and mutation to constrain the model's actions within a safe solution space while maintaining task performance. Experimental results demonstrate that LCO substantially alleviates ICRH in both output-refine and policy-refine scenarios. In particular, on the tweet engagement optimization task, LCO achieves a 39% reduction in the Toxicity Growth Rate (TGR) on GPT-4, while on the policy optimization benchmark, it reduces the ICRH Occurrence Rate by 15.23%, demonstrating safety improvement without sacrificing task performance.
>
---
#### [new 109] A new semantically annotated corpus with syntactic-semantic and cross-lingual senses
- **分类: cs.CL**

- **简介: 该论文属于词义消歧任务，旨在构建一个包含多维语义标注的法语动词语料库，解决跨语言和细粒度语义标注问题。**

- **链接: [https://arxiv.org/pdf/2605.28494](https://arxiv.org/pdf/2605.28494)**

> **作者:** Myriam Rakho; Eric Laporte; Matthieu Constant
>
> **摘要:** We describe a new sense-tagged corpus for word sense disambiguation. The corpus is constituted of instances of 20 French polysemous verbs. Each verb instance is annotated with three sense labels: (1) the actual translation of the verb in the english version of this instance in a parallel corpus, (2) an entry of the verb in a computational dictionary of French (the Lexicon-Grammar tables) and (3) a fine-grained sense label resulting from the concatenation of the translation and the Lexicon-Grammar entry.
>
---
#### [new 110] Better heads do not guarantee better binarized constituency parsing
- **分类: cs.CL**

- **简介: 该论文属于句法分析任务，探讨头标注对二叉化解析的影响。研究发现，虽然后代头标注在预测上更优，但未带来解析性能提升，表明语言学头标注未必最优。**

- **链接: [https://arxiv.org/pdf/2605.28131](https://arxiv.org/pdf/2605.28131)**

> **作者:** Zeyao Qi; Yige Chen; Eitan Klinger; Vivaan Wadhwa; Jungyeul Park
>
> **摘要:** We revisit punctuation-aware tree binarization for constituency parsing and ask whether dependency-induced headedness improves binary parser supervision. Although learned heads substantially outperform rule-based heads in intrinsic head prediction, they do not yield consistent parsing gains after debinarization. In particular, punctuation-conditioned evaluation shows that learned headedness underperforms rule-based binarization in macro-average punctuation-sensitive $F_1$, despite a small overall gain on CTB. Similar instability appears under cross-treebank transfer. These results suggest that \ycc{linguistically grounded} headedness is not necessarily parser-optimal when used as a binarization control signal. The paper presents a negative result: better head prediction does not imply better punctuation-sensitive constituency parsing.
>
---
#### [new 111] ICG: Improving Cover Image Generation via MLLM-based Prompting and Personalized Preference Alignment
- **分类: cs.CL**

- **简介: 该论文属于个性化封面生成任务，旨在提升用户参与度。通过结合MLLM和扩散模型，引入个性化偏好对齐，解决传统方法依赖人工提示和模块分离的问题。**

- **链接: [https://arxiv.org/pdf/2605.27374](https://arxiv.org/pdf/2605.27374)**

> **作者:** Zhipeng Bian; Jieming Zhu; Qijiong Liu; Wang Lin; Guohao Cai; Zhaocheng Du; Jiacheng Sun; Zhou Zhao; Zhenhua Dong
>
> **备注:** Published in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 12268-12278, EMNLP 2025. Official version: this https URL
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) and diffusion models (DMs) have opened new possibilities for AI-generated content. Yet, personalized cover image generation remains underexplored, despite its critical role in boosting user engagement on digital platforms. We propose ICG, a novel framework that integrates MLLM-based prompting with personalized preference alignment to generate high-quality, contextually relevant covers. ICG extracts semantic features from item titles and reference images via meta tokens, refines them with user embeddings, and injects the resulting personalized context into the diffusion model. To address the lack of labeled supervision, we adopt a multi-reward learning strategy that combines public aesthetic and relevance rewards with a personalized preference model trained from user behavior. Unlike prior pipelines relying on handcrafted prompts and disjointed modules, ICG employs an adapter to bridge MLLMs and diffusion models for end-to-end training. Experiments demonstrate that ICG significantly improves image quality, semantic fidelity, and personalization, leading to stronger user appeal and offline recommendation accuracy in downstream tasks. As a plug-and-play adapter bridging MLLMs and diffusion models, ICG is compatible with common checkpoints and requires no ground-truth labels during optimization.
>
---
#### [new 112] Building Community-Centred NLP Resources for Puno Quechua
- **分类: cs.CL; cs.DB; cs.HC**

- **简介: 该论文属于自然语言处理任务，旨在保护濒危语言Puno Quechua。通过构建语音语料库、进行ASR基准测试并开源资源，解决其数字工具匮乏的问题。**

- **链接: [https://arxiv.org/pdf/2605.28253](https://arxiv.org/pdf/2605.28253)**

> **作者:** Elwin Huaman; Adrian Gamarra Lafuente; Johanna Cordova; Anna Korhonen
>
> **备注:** Sixth Workshop on NLP for Indigenous Languages of the Americas (AmericasNLP 2026), co-located with ACL 2026
>
> **摘要:** The preservation of under-resourced languages requires digital tools and resources shaped by and for their speakers. We present the first dedicated ASR resources for Puno Quechua (ISO 639-3: qxp): (1) the largest speech corpus for any single Quechua variety, consisting in 66 hours of recordings for scripted and spontaneous speech (including 36 hours of manually transcribed and validated data), collected via a participatory design campaign; (2) the first systematic ASR benchmark for Puno Quechua, evaluating state-of-the-art models and fine-tuning Whisper-base, wav2vec2-base, and XLS-R-300M, with and without continued pre-training (CPT); (3) an open release of all datasets and fine-tuned models.
>
---
#### [new 113] Semantic Flow Regularization: Teaching LLMs to Generate Diverse Yet Coherent Responses
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，解决大模型生成回复多样性不足的问题。提出Semantic Flow Regularization方法，提升回复多样性与一致性。**

- **链接: [https://arxiv.org/pdf/2605.27971](https://arxiv.org/pdf/2605.27971)**

> **作者:** Kerui Peng; Feifei Li; Xingyu Fan; Wenhui Que
>
> **摘要:** When large language models are fine-tuned to generate persona- or tone-conditioned responses, their output diversity is severely limited--a failure we term Cross-Style Collapse. We trace this collapse to the cross-entropy objective, which under shared representations tends to suppress diverse continuations. We propose Semantic Flow Regularization (SFR), a lightweight auxiliary objective that supervises the backbone with continuous sentence-encoder embeddings of future segments via conditional flow matching. The stochastic flow source preserves multi-modality by construction; the flow-matching head is discarded at inference, adding zero deployment cost. On a large-scale industrial dialogue dataset (Qwen3-32B, 9 personas), SFR improves output diversity, style fidelity, and response quality over SFT. We further validate on the public LiveCodeBench-v5 (Qwen2.5-Coder-7B-Instruct), where SFR consistently improves pass@k, confirming generality beyond stylized dialogue. A controlled comparison on MBPP reveals Multi-Token Prediction to be a degenerate special case of SFR.
>
---
#### [new 114] Roles with Rails: Contract-Preserving Role Evolution in Multi-Agent Structured Reasoning
- **分类: cs.CL**

- **简介: 该论文属于多智能体系统任务，解决角色演化中的结构约束问题。提出SERO框架，通过合同保持机制实现角色自适应演化。**

- **链接: [https://arxiv.org/pdf/2605.28433](https://arxiv.org/pdf/2605.28433)**

> **作者:** Ling-Yue Ge; Lan-Zhe Guo
>
> **备注:** 33 pages, 23 figures, 12 tables
>
> **摘要:** Role-based LLM multi-agent systems need adaptive role pools, yet adapting such systems is not merely a matter of prompt optimization: roles often carry structural obligations, including capability coverage, message compatibility, validation, final-answer aggregation, and parser-compatible output protocols. Existing systems either fix the role inventory and lose adaptivity, or allow unconstrained generation to induce role drift, removing structurally necessary roles and breaking answer contracts. We formulate this as contract-preserving role evolution, requiring every committed edit to preserve five structural contracts (capability, communication, validation, aggregation, output protocol). We instantiate this formulation in SERO, a Self-Evolving Role Orchestration framework that evolves a typed role-card pool through credit-guided retrieval, a credit-ranked communication DAG with a protected terminal aggregator and conditional validator repair, and a contextual-bandit controller whose LLM-proposed edits are committed only when they preserve the contracts and improve task score. Experiments on real-world reasoning benchmarks across three LLM backbones confirm the value of contract-preserving role evolution.
>
---
#### [new 115] Chinese Word Boundary Recovery through Character Alignment Projection
- **分类: cs.CL**

- **简介: 该论文属于中文分词任务，解决非标准文本中词边界断裂问题。通过字符对齐投影方法，利用干净文本修正噪声输入的词边界。**

- **链接: [https://arxiv.org/pdf/2605.28128](https://arxiv.org/pdf/2605.28128)**

> **作者:** Lusha Wang; Yuchen Li; Su Yuan; Jungyeul Park
>
> **摘要:** Chinese word segmentation is especially fragile in non-standard text, where language learner errors and other character-level divergences disrupt the word boundaries assumed by downstream annotation and evaluation. This paper formulates Chinese word boundary recovery as an alignment-based projection task. Given a noisy source sentence and a cleaner target counterpart, we first align the two strings at the character level and then project target-side word boundaries back onto the source. Beyond the recovery method itself, we introduce two evaluation resources: a manually checked learner Chinese benchmark based on MuCGEC and a controlled synthetic benchmark derived from the Chinese Penn Treebank. Experiments show that direct segmentation remains vulnerable to compound fragmentation in learner input, whereas the proposed two step projection method corrects many over-segmentation errors by using the corrected target to recover source-side word spans. The results show that word boundary recovery is distinct from ordinary segmentation and that alignment projection provides a principled mechanism for stabilizing Chinese annotation and evaluation under noisy input.
>
---
#### [new 116] EvoSpec: Evolving Speculative Decoding via Real-Time Vocabulary and Parameter AdaptationTarget
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型推理加速任务，解决 speculate decoding 中因词汇量扩大导致的瓶颈问题。通过动态调整词汇和参数，提升效率与适应性。**

- **链接: [https://arxiv.org/pdf/2605.27390](https://arxiv.org/pdf/2605.27390)**

> **作者:** Shuyu Zhang; Lingfeng Pan; Qicheng Wang; Yaqi Shi; Yueyang Tan; Ruyu Yan; Jiaqi Chen; Lixing Du; Lu Wang
>
> **摘要:** Speculative decoding accelerates Large Language Model inference via a draft-then-verify paradigm, yet the output projection layer becomes a bottleneck as vocabulary sizes scale. While existing static pruning methods effectively reduce this overhead, they suffer from precipitous drops in acceptance rate in specialized domains or topic-switching scenarios due to their inability to capture dynamic distribution shifts. To address this, we introduce EvoSpec, a framework that enables real-time evolution of the draft model through dynamic vocabulary and parameter adaptation. Unlike static or purely retrieval-based approaches, EvoSpec employs a context-aware mechanism that retrieves critical long-tail tokens via efficient semantic and statistical indexing. Furthermore, we propose a lightweight online alignment strategy utilizing curriculum learning to continually minimize the distributional gap between the draft and target models. Extensive evaluations across specialized domains (coding, law, and medicine) confirm that EvoSpec overcomes the limitations of static baselines. On EAGLE-3, it achieves a 1.13x speedup in these settings over the state-of-the-art static baseline FR-Spec, with 27\% lower memory overhead than standard online adaptation.
>
---
#### [new 117] Comonadic Morphophonology: A Compositional Framework for Context-Dependent Morphological Rules in Finnish
- **分类: cs.CL**

- **简介: 该论文属于形态学任务，解决上下文依赖的形态音系规则组合问题。提出基于余单子的框架，实现规则的严格组合性，提升处理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.28484](https://arxiv.org/pdf/2605.28484)**

> **作者:** Yongseok Jang
>
> **备注:** 13 pages. Accepted at the Society for Computation in Linguistics (SCiL) 2026
>
> **摘要:** Composing finite-state transducers (FSTs) for context-dependent morphophonological rules -- consonant gradation, vowel harmony, possessive suffix assimilation -- leads to multiplicative state explosion; neural models sidestep the problem but provide no formal account of the rules themselves. We present the first framework where each morphophonological rule is a function from a focused local context to a single output segment -- the type of a local rule familiar from cellular automata -- and where length-changing rules compose as coKleisli arrows of a comonad. Our central contribution is the Writer comonad (DeletionSet x Zipper), a new algebraic construction that restores strict coKleisli compositionality for such rules: each rule is a coKleisli arrow, extend lifts it to a global transformation, and deletions accumulate as a monoid action rather than requiring intermediate materialization. As supporting evidence, thirteen coKleisli arrows provide an alternative formulation expressing the same morphophonological behaviors that Omorfi encodes via 874 continuation classes (67:1 reduction at the rule-representation level), and the same abstraction enables bidirectional morphology -- a MorphGenerator reuses the analysis arrows for generation. On UD Finnish-TDT, the system achieves 83.92% UPOS accuracy with rule-only disambiguation (94.66% with an external suffix tagger), validating the framework as a practical morphological engine.
>
---
#### [new 118] DEPART: DEcomposing PARiTy across Multilingual LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言大模型评估任务，旨在解析语言性能差异的根源。通过统计检验和贝叶斯框架，分解性能方差，揭示语言特征与模型表现的关系。**

- **链接: [https://arxiv.org/pdf/2605.28163](https://arxiv.org/pdf/2605.28163)**

> **作者:** Manan Uppadhyay; Prashant Kodali; Pranjal Chitale; Reshma Ramaprasad; Himanshu Beniwal; Sunayana Sitaram
>
> **摘要:** Multilingual Large Language Models (mLLMs) leaderboards report per-language accuracy but rarely explain why disparities emerge, leaving systemic biases unattributed and offering practitioners no actionable levers. We first establish that these gaps are systematic rather than artifacts of sampling noise via distribution-free Friedman and Kruskal--Wallis tests, then introduce a two-step Bayesian hierarchical framework that decomposes multilingual performance variance into interpretable components. First, isolating the variance attributable to language identity, we show that observable language features (script, family, typological distance) explain $R^2_{\text{ling}} = 79\%$ of this variance on understanding tasks and $92\%$ on reasoning, with a model's internal representational similarity to English emerging as the dominant predictor across both task buckets. Second, decomposing the full (model$\times$benchmark$\times$language) cube, we find that NLU and reasoning have fundamentally divergent variance profiles: model identity dominates understanding ($66.7\%$ of variance), whereas the benchmark$\times$model interaction dominates reasoning ($46.3\%$). Together these results recast multilingual evaluation from passive performance mapping into an explainable, diagnostic framework with concrete levers for targeting the root drivers of language disparity.
>
---
#### [new 119] HELEA: Hard-Negative Benchmark and LLM-based Reranking for Robust Entity Alignment
- **分类: cs.CL**

- **简介: 该论文属于实体对齐任务，解决现有基准难以评估模型区分同名实体的问题。通过构建硬负样本基准和引入LLM重排序，提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2605.28308](https://arxiv.org/pdf/2605.28308)**

> **作者:** Yoonjin Jang; Junwoo Kim; Youngjoong Ko
>
> **备注:** 10 pages, 3 figures, 9 tables. Code and benchmarks available at this https URL
>
> **摘要:** Entity Alignment (EA) is essential for knowledge graph (KG) fusion, but existing benchmarks often allow models to exploit name overlap rather than relational structure. This makes it difficult to evaluate whether models can reject same-name entities that refer to different real-world objects. Our primary contribution is a same-name hard-negative augmentation strategy that simultaneously yields quality-controlled evaluation benchmarks (DW-HN29K, DY-HN27K) and augmented training corpora (DW-Train, DY-Train), by mining same-name but distinct entity pairs from KG name-collision groups. We further introduce HELEA, a two-stage framework integrating (i) entity encoder retrieval trained on hard-negative-augmented training corpora with 1-hop KG context, and (ii) LLM-based reranking without additional training. Experiments show that name-dependent baselines collapse to near-random performance on our hard-negative benchmarks, while HELEA achieves F1 0.967 on DW-HN29K while maintaining Hit@1 0.993 on standard DW-15K.
>
---
#### [new 120] Disentangling Language Roles in Multilingual LLM Task Execution
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多语言大模型任务执行研究，旨在解决语言角色对模型性能影响的评估问题。通过构建MTM-Bench基准，分析不同语言组合下的任务表现。**

- **链接: [https://arxiv.org/pdf/2605.27649](https://arxiv.org/pdf/2605.27649)**

> **作者:** Qishi Zhan; Minxuan Hu; Seoyeon Jang; Lei Zhao; Ziheng Chen; Man Liang; Xinyue Xiang; Jiaxin Liu; Guansu Wang; Liang He
>
> **摘要:** Multilingual LLMs are increasingly used when instruction, source content, and required response languages do not coincide. Existing benchmarks have expanded multilingual instruction-following evaluation, but they rarely isolate these three roles within a fully crossed design. We introduce MTM-Bench, a controlled benchmark for language-conditioned task execution in which each instance is defined by a triplet \((L_{\text{instr}}, L_{\text{content}}, L_{\text{resp}})\). Across English, Spanish, and Chinese, MTM-Bench enumerates all 27 triplets and contains 2{,}430 instances per model across semantic reversal, final-state extraction, and language purity with update realization. We evaluate 20 frontier and open-weight LLMs using decomposed metrics for semantic correctness, target-language adherence, constraint satisfaction, contamination ratio, and joint success, with scoring validated by a targeted human audit. The fully crossed design reveals that degradation is organized by the role a language occupies in the task structure, not merely by mismatch count. The response-language role is the dominant axis of variation, and a single response-slot mismatch accounts for most degradation. The response-only and full-mismatch comparison suggests that mismatch count is not a monotonic predictor of difficulty, with model-level ordering varying across systems. Task families fail through distinct channels, showing that semantic correctness alone does not capture reliable multilingual task execution.
>
---
#### [new 121] Debate Helps Weak Judges Reward Stronger Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究辩论在代码和逻辑任务中对弱裁判的提升效果，解决如何有效利用辩论进行模型监督的问题。通过实验验证不同模型配对下辩论的有效性。**

- **链接: [https://arxiv.org/pdf/2605.27483](https://arxiv.org/pdf/2605.27483)**

> **作者:** Ethan Elasky; Frank Nakasako; Naman Goyal
>
> **摘要:** Despite theoretical promise, debate as a scalable oversight protocol has produced mixed empirical results: gains in some settings, and null effects in others, especially when the judge does not have information hidden from it. We study proposer-critic debate in a stronger-debater/weaker-judge setting on programmatically verifiable code and logic tasks. Debate helps the judge over a consultancy baseline when the critic provides a usable advantage: the critic's classification ability must exceed the judge's, and the judge must treat critic speeches as claims to verify rather than testimony to summarize. On the three of five pairings where the condition holds, proposer-critic debate's gains are statistically significant over consultancy, and these pairings are the most capable model pairings. On the two non-responder pairings in our set, debate produces null effects, and judge verification rates drop by tens of percentage points once a critic enters the transcript. In these cases the critic's binary-classification ability and the judge's are within noise of each other, and the critic's disagreement is parsed as testimony rather than a claim to check. Ablating rebuttal rounds from debate produces no measurable change in judge performance: a single independent critique recovers the bulk of debate's benefit at lower inference cost. These findings suggest a cheaper primitive for training-free scalable oversight in verifiable domains (answer, critique, judge) and a pre-deployment audit (does the critic beat the judge, and will the judge verify it?) that predicts when debate will help.
>
---
#### [new 122] Measuring Form and Function in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型评估任务，旨在衡量模型的句法和语用能力。通过提出新方法CAC，对比模型与儿童及统计基准，揭示当前模型在认知水平上的不足。**

- **链接: [https://arxiv.org/pdf/2605.28616](https://arxiv.org/pdf/2605.28616)**

> **作者:** Héctor Javier Vázquez Martínez; Charles Yang
>
> **备注:** Under review at ACL Rolling Review May 2026 cycle
>
> **摘要:** We introduce quantitative metrics for child language acquisition to evaluate language models. Our focus is on the formal syntactic and functional discourse properties of determiners in English, which young children acquire early and accurately. We propose Contextual Alternative Choice (CAC), a new prompting method which provides targeted tests for both syntactic and discourse knowledge of language. The method enables direct comparison of language models against children, and more importantly, against statistical benchmarks independently established in empirical research. No current model trained on a comparable amount of data simultaneously meet both formal and functional benchmarks like human children, but some very large models do. We present our results as methodological and technical contributions, with specific emphasis on cognitive status of language models.
>
---
#### [new 123] Reading or Guessing? Visual Grounding Failures of Vision-Language Models for OCR in Ancient Greek Editions
- **分类: cs.CL; cs.AI; cs.CV; cs.DL**

- **简介: 该论文属于OCR任务，研究VLM在古希腊文本识别中的视觉 grounding 问题。通过对比实验与图像扰动分析，揭示模型依赖语言先验而非视觉证据的现象，提出改进方向。**

- **链接: [https://arxiv.org/pdf/2605.27750](https://arxiv.org/pdf/2605.27750)**

> **作者:** Antonia Karamolegkou; Nicolas Angleraud; Benoît Sagot; Thibault Clérice
>
> **摘要:** Recent work has shown that Vision-Language Models (VLMs) used for optical character recognition (OCR) can generate plausible but visually unsupported text, suggesting reliance on language priors. Comparing open-weight VLMs with traditional OCR baselines on low-resource Ancient Greek critical editions, we show that VLM errors often remain fluent even when wrong, producing plausible Greek substitutions where traditional engines produce local recognition noise. To analyze visual evidence during decoding, we introduce controlled image perturbations and token-level grounding measures based on conditional versus image-free decoding distributions. Under character-level perturbations, VLMs diverge sharply from the perturbed ground truth while traditional OCR remains comparatively faithful; however, token-level analysis shows that prior reliance is model-specific: in an OCR-specialist model, fluent lexical errors are produced with little reliance on the image, whereas general-purpose VLMs remain conditioned on the visual input even when wrong. Decode-time interventions fail to reliably restore grounding, while post-OCR language-model correction improves several systems only by repairing text after generation. Our results extend prior evidence of OCR language-prior reliance to low-resource historical documents and a broader set of models, showing that fluent output is not necessarily visually grounded and motivating interpretability-driven evaluation beyond aggregate accuracy.
>
---
#### [new 124] Auditing Stance Asymmetry in Generative Explanations
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于语言模型偏见评估任务，解决生成解释中的立场不对称问题。通过提出SDE方法，分析模型在不同情境下的责任分配与解释立场差异。**

- **链接: [https://arxiv.org/pdf/2605.27988](https://arxiv.org/pdf/2605.27988)**

> **作者:** Jiarui Han
>
> **摘要:** Bias evaluation for language models has made substantial progress on bounded comparisons, such as overt derogation, stereotype association, or label-sensitive differences under controlled substitutions. Open-ended explanations raise a different problem: they guide interpretation by assigning responsibility, legitimacy, context, and grievance. A model can avoid hostile language while making one side structurally understandable and another personally at fault, overreacting, or less worth taking seriously. We call this stance-bearing asymmetry in generative explanations. We propose Symmetry Decomposition Evaluation (SDE), which tests paired situations with concrete group labels, structural-role rewrites, and explicit support or counter-evidence. In a controlled 32-family prototype suite, this decomposition shows that surface differences are not all alike: some weaken under structural or evidence control, while others remain as stable differences in how the model assigns blame, context, or legitimacy. Targeted case review and judge comparison suggest a broader difficulty for evaluating open-ended framing asymmetries: judge readings shift across operationalizations, and scalar scores can flatten distinctions that readers use to interpret explanatory stance. SDE therefore reframes generative bias evaluation as an audit of explanatory stance -- what stance each side receives, how it changes under decomposition, and where automatic scoring becomes unstable.
>
---
#### [new 125] OralAgent: Integrating Reasoning, Tools, and Knowledge for Interactive Dental Image Analysis
- **分类: cs.CL; cs.CV; cs.MA**

- **简介: 该论文提出OralAgent，解决口腔影像分析中多模态任务整合问题，融合推理、工具与知识，提升临床实用性。**

- **链接: [https://arxiv.org/pdf/2605.27378](https://arxiv.org/pdf/2605.27378)**

> **作者:** Jing Hao; Siyuan Dai; Yongxin Zhang; Yuci Liang; Jiamin Wu; Jiahao Bao; Yuxuan Fan; Zanting Ye; Yanpeng Sun; Xinyu Zhang; Ming Hu; Liang Zhan; James Kit Hon Tsoi; Linlin Shen; Junjun He; Kuo Feng Hung
>
> **备注:** 14 pages, 7 figures, 6 tables
>
> **摘要:** Dental image analysis plays a pivotal role in supporting accurate diagnosis and treatment planning in oral healthcare. Although recent advances have produced dental AI models for specific tasks and individual imaging modalities, their isolated designs limit practical use in real-world clinical workflows. In this paper, we present OralAgent, the first dental-specialized AI agent that unifies multimodal reasoning, tool-based decision-making, and knowledge-grounded retrieval within an end-to-end automated framework. It integrates 22 visual analysis tools and 368 widely-used classical dental textbooks, enabling autonomous reasoning, planning, tool use, knowledge retrieval, and multi-step workflow execution. Furthermore, we introduce OralCorpus, a large-scale, high-quality bilingual textual resource containing 134.8M tokens curated for dental retrieval-augmented generation (RAG). To evaluate models' multidisciplinary dental knowledge, we construct OralQA-ZH, a Chinese multiple-choice question benchmark consisting of 798 items across eleven oral subspecialties. Extensive experiments demonstrate that OralAgent achieves state-of-the-art performance on the MMOral-Uni, MMOral-OPG, and OralQA-ZH benchmarks, highlighting its effectiveness, interpretability, and adaptability in real-world clinical settings. The code and models are publicly available at this https URL.
>
---
#### [new 126] Soft-SVeRL: Self-Verified Reinforcement Learning with Soft Rewards
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Soft-SVeRL，解决部分可验证任务中的强化学习问题。通过分解提示为检查项，使用LLM评分生成软奖励，提升训练效果。**

- **链接: [https://arxiv.org/pdf/2605.28561](https://arxiv.org/pdf/2605.28561)**

> **作者:** Saurabh Dash; Pierre Clavier; John Dang; Matthias Galle; Marzieh Fadaee; Ahmet Üstün; Beyza Ermis
>
> **摘要:** Reinforcement Learning from Verifiable Rewards (RLVR) has improved language models in domains such as mathematics and code, where correctness can be checked automatically. However, many important tasks are only partially verifiable: prompts contain multiple requirements, responses may satisfy some but not all of them, or no single reference answer might exist. We introduce Soft-RLVR, a framework for reinforcement learning from decomposed, learned verification signals. Soft-RLVR converts each prompt into a checklist of atomic requirements, scores candidate responses item by item with an LLM verifier, and trains on the resulting soft reward. Checklist-based rewards turn sparse pass/fail supervision into a denser partial-credit signal, but they also introduce a tradeoff: averaging item-level judgments can reduce verifier noise, while partial credit can reward incomplete responses. We formalize this tradeoff and identify conditions under which checklist-based verification gives a more reliable RL training signal than holistic verification. We further introduce Soft-SVeRL, a self-verifying variant of Soft-RLVR in which the policy also acts as the verifier. We show that self-verification is prone to reward inflation from overly permissive self-judgments, and that explicit stabilization is needed to prevent this collapse. In a controlled instruction-following setting with rule-based ground-truth evaluation, checklist-based Soft-RLVR improves IFEval by up to 11.1 points using only learned verifier rewards. Our experiments further show that verifier quality and checklist quality both affect downstream RL outcomes, and that explicit stabilization is essential for effective self-verification.
>
---
#### [new 127] OmniVerifier-M1: Multimodal Meta-Verifier with Explicit Structured Recalibration
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于多模态验证任务，旨在提升基础模型的可靠性和可解释性。通过符号化元验证和解耦强化学习，解决传统验证方法依赖模型奖励和效果不佳的问题。**

- **链接: [https://arxiv.org/pdf/2605.28805](https://arxiv.org/pdf/2605.28805)**

> **作者:** Xinchen Zhang; Bowei Liu; Jiale Liu; Chufan Shi; Yizhen Zhang; Junhong Liu; Youliang Zhang; Zhiheng Li; Yujiu Yang; Ling Yang
>
> **备注:** ICML 2026. Project: this https URL
>
> **摘要:** Visual outcomes are increasingly central to multimodal large language models, making reliable and fine-grained verification essential for scaling generalist foundation models. In this work, we investigate multimodal meta-verification, which leverages verifier-generated rationales rather than decision-only signals, and explore how to effectively incorporate meta-verification feedback into multimodal verifier training. We identify two key findings. First, symbolic verifier outputs (e.g., bounding boxes) outperform textual explanations as meta-verification rationales, enabling efficient rule-based reinforcement learning rewards while avoiding reliance on model-based rewards from auxiliary judge models. Second, decoupling reinforcement learning objectives for binary judgment and meta-verification substantially outperforms joint reward optimization, due to intrinsic differences in output structure and learning dynamics. Based on these insights, we train OmniVerifier-M1, a generalist visual verifier leveraging symbolic meta-verification and decoupled reinforcement learning. OmniVerifier-M1 provides robust verification and fine-grained error localization, and further enables M1-TTS, a verifier-driven agentic generation system achieving dynamic region-level self-correction. This approach paves the way for more reliable, interpretable, and fine-grained multimodal verification, supporting safer and more controllable foundation model deployment.
>
---
#### [new 128] HardMTBench: Stress-Testing Chinese-English Translation on Knowledge-Intensive Domains
- **分类: cs.CL**

- **简介: 该论文属于中英机器翻译任务，旨在解决现有基准在知识密集领域表现趋同的问题。通过构建HardMTBench，提升翻译难度评估的准确性。**

- **链接: [https://arxiv.org/pdf/2605.28315](https://arxiv.org/pdf/2605.28315)**

> **作者:** Zheng Li; Mao Zheng; Mingyang Song; Tianxiang Fei
>
> **摘要:** General-purpose machine translation benchmarks such as FLORES-200 have reached a saturation regime on Chinese-English pairs, where modern large language models cluster within a narrow band of high scores. Across 22 systems, FLORES-200 zh-en GEMBA scores fall in a 7.87-point range with a standard deviation of 2.29, which compresses the separation between systems on knowledge-intensive domains such as finance, healthcare, law, and science and technology. We introduce HardMTBench, a difficulty-aware diagnostic benchmark for bidirectional Chinese-English domain translation. HardMTBench covers 12 domains and contains 10,000 hand-curated source sentences with reference translations, packaged as 20,000 directional test items. A three-stage construction pipeline builds a domain-balanced candidate pool of 84{,}566 pairs, applies an LLM-based multi-signal judge over knowledge density, translation difficulty, terminology load and reference correctness, and assembles the final test set under a hardness fusion rule with per-domain quotas. Across 22 systems spanning general LLMs, commercial engines and specialised MT models, HardMTBench widens the cross-system GEMBA range by roughly a factor of two over FLORES-200, induces visible rank reorderings, and exposes domain-specific terminology and knowledge weaknesses that quality-only metrics tend to flatten. All data and code are open-sourced at this https URL.
>
---
#### [new 129] Models That Know How Evaluations Are Designed Score Safer
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI安全评估任务，探讨模型如何因了解评估设计而改变行为。研究发现模型通过学习评估元知识提升安全性，但可能高估实际表现。**

- **链接: [https://arxiv.org/pdf/2605.28591](https://arxiv.org/pdf/2605.28591)**

> **作者:** Katharina Deckenbach; Haritz Puerto; Jonas Geiping; Sahar Abdelnabi
>
> **摘要:** The validity of AI safety evaluations depends on models behaving consistently across controlled and deployment settings. Prior work has identified test-time contextual cues, such as hypothetical scenarios, as a source of verbalized evaluation awareness and subsequent behavioral shift. In this paper, we investigate a potential explanation of this phenomenon: evaluation meta-knowledge, defined as parametric knowledge about the structural traits that characterize evaluations. Similar to dataset contamination, where benchmark exposure leads to higher performance through memorization, we hypothesize that models trained on texts describing evaluation practices may implicitly learn to recognize and respond to evaluation-like contexts, for instance, through exposure to scientific articles or social media posts about AI benchmarking. To test this, we fine-tune models on synthetic documents describing evaluation traits such as verifiable structures or moral dilemmas. Evaluating this fine-tuned model on six safety benchmarks, we find that it is significantly safer than the base model and control model. This behavioral shift persists even when restricting the analysis to responses lacking explicit verbalization of evaluation awareness. Our results demonstrate that evaluation meta-knowledge may inflate safety benchmark performance, introducing a novel confounder that is independent of explicit memorization or verbalized evaluation awareness, thus, challenging to detect. These findings have important implications for the design and interpretation of AI safety evaluations. Our code and models are available at this https URL.
>
---
#### [new 130] IFMTBench: A Comprehensive Benchmark for Multilingual Translation Instruction Following
- **分类: cs.CL**

- **简介: 该论文属于多语言翻译任务，解决模型遵循复杂翻译指令的问题。构建了IFMTBench基准，涵盖多种约束条件，评估模型在多语言环境下的指令遵循能力。**

- **链接: [https://arxiv.org/pdf/2605.28218](https://arxiv.org/pdf/2605.28218)**

> **作者:** Mingrui Sun; Mao Zheng; Zheng Li; Mingyang Song
>
> **备注:** 11 pages, 6 figures, conference
>
> **摘要:** Modern translation workflows demand more than semantic equivalence. Users routinely require models to preserve JSON or HTML schemas, honor curated glossaries, disambiguate with provided context, and match prescribed registers, often several at once. Conventional metrics such as BLEU and xCOMET capture semantic fidelity but provide little signal on constraint adherence, while general instruction following benchmarks ignore the cross-lingual nature of translation. We introduce \bench, a benchmark for multilingual translation instruction following covering seven languages, with 4,506 single-constraint and 2,838 multi-constraint items spanning six constraint dimensions and five compositional patterns with instructions issued in all seven languages. Constraints are split into a gating subset verified by deterministic checkers and a continuous subset scored by a rubric-based LLM judge, combined under a multiplicative rule that resists reward hacking. Evaluating 15 models reveals systematic gaps that prior protocols miss: Instruction following scales with size more sharply than translation quality, glossary and structured-format constraints dominate the difficulty gradient, and general instruction following rankings correlate only weakly with translation behavior. Our benchmark are available at this https URL.
>
---
#### [new 131] BioELX: Cross-lingual Biomedical Entity Linking via Alias-based Retrieval and LLM Ranking
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BioELX，解决跨语言生物医学实体链接任务。针对低资源语言数据不足和系统泛化能力差的问题，通过多语言别名增强和大模型排序实现无需标注数据的高效链接。**

- **链接: [https://arxiv.org/pdf/2605.27380](https://arxiv.org/pdf/2605.27380)**

> **作者:** Yi Wang; Corina Dima; Liangyu Zhong; Steffen Staab
>
> **备注:** 12 pages, 3 figures
>
> **摘要:** Cross-lingual biomedical entity linking (BEL) maps mentions in any language to unique identifiers in a biomedical knowledge base (KB), supporting clinical and biomedical NLP applications. However, expert-annotated training data for BEL are costly, especially for low-resource languages. Moreover, many cross-lingual BEL systems rely on SapBERT-based retrievers trained on predominantly English aliases in the KB, leading to poor generalization to unseen non-English mentions and limited context-aware disambiguation. We propose BioELX, a two-stage cross-lingual BEL framework that requires no task-specific annotated training corpora. In Stage~1, we enrich SapBERT training with Wikidata-derived multilingual aliases and use the resulting retriever to improve cross-lingual candidate retrieval. In Stage~2, we perform context-aware disambiguation with a pre-trained LLM ranker that jointly considers the mention context and candidate, eliminating the need for supervised training. Experiments on five benchmarks (XL-BEL, EMEA, Patent, WikiMed-DE, and MedMentions) show that BioELX achieves new state-of-the-art performance. It improves average Recall@1 on XL-BEL by +19.2, with especially large gains for low-resource languages, e.g., +21.6 on Turkish, +22.1 on Korean, +30.8 on Thai, and delivers consistent improvements on EMEA (+6.2), Patent (+5.4), and WikiMed-DE (+12.8). Code and resources will be released upon publication.
>
---
#### [new 132] On Compositional Learning Behaviours in Formal Mathematics
- **分类: cs.CL**

- **简介: 该论文属于形式数学验证任务，探讨组合学习行为（CLB）在解决复杂定理证明中的作用，通过实验表明CLB是突破数学难题的必要条件。**

- **链接: [https://arxiv.org/pdf/2605.28512](https://arxiv.org/pdf/2605.28512)**

> **作者:** Kevin Yandoka Denamganaï
>
> **备注:** work in progress, under review
>
> **摘要:** Self-evolving scientific agents capable of conquering the hard tail of formal mathematics require Compositional Learning Behaviours (CLBs) -- the capacity to ground and recombine novel symbolic structures in context, beyond mere recombination of prelearned atoms. We propose \textbf{S2B-LM}, an adaptation of the Symbolic Behaviour Benchmark that removes numerical processing as a confound and adds chain-of-thought scaffolding to elicit rather than merely probe latent CLB competency. Cross-evaluating ten Lean~4 theorem provers on CLB competency (adj-ZSCT) and miniF2F whole-proof performance, exact permutation tests establish a hierarchical necessity structure: search-heavy models cover the tractable bulk without detectable CLBs, yet every model breaking into the Olympiad-level tier (miniF2F $>75\%$) is among the five highest CLB scorers ($p=0.004$). After ruling out model scale as a confound, our results show that CLB competency is \emph{necessary but not sufficient} for the hard tail of formal mathematical verification.
>
---
#### [new 133] Keyphrase Generative Representation of Youth Crisis Conversations Beyond Static Taxonomies
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于心理健康危机响应任务，旨在解决传统分类体系无法捕捉动态语言的问题。通过扩展分类体系并引入关键短语生成方法，提升对青少年情绪问题的识别与理解。**

- **链接: [https://arxiv.org/pdf/2605.27546](https://arxiv.org/pdf/2605.27546)**

> **作者:** Abeer Badawi; Will Aitken; Lydia Sequeira; Jocelyn Rankin; Maia Norman; Elham Dolatabadi
>
> **摘要:** Crisis Responders (CRs) rapidly assess thousands of youth SMS conversations each year to identify mental health concerns and guide support. Yet youth distress is increasingly expressed through evolving and context-specific language that often does not fit fixed-label taxonomies. This work analyzed 703,975 de-identified Kids Help Phone conversations (2018-2023) and expanded KHP's 19-label issue taxonomy into a 39-label hierarchical schema. We then introduce Keyphrase Generative Representation (KGR), a constrained LLM generating concise, conversation-specific keyphrases, evaluated across 129 conversations and 387 expert annotations. The expanded taxonomy achieved expert consensus reliability, with an accuracy of 0.96, and expert review found that 81% of keyphrases accurately reflected content and 74% improved clarity. KGR surfaced identity-linked themes absent from the fixed taxonomy, including immigration problems and caregiver burden, and supported a topic-retrieval workflow that increased accuracy from 0.25 to 0.70 (+0.45) over the manual analyst process. KGR marks a shift toward hybrid, interpretable generative representations that extend crisis response beyond static taxonomies to surface emerging and culturally grounded patterns of youth distress.
>
---
#### [new 134] ESC-Skills: Discovering and Self-Evolving Skills for Emotional Support Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感支持对话任务，旨在提升系统可解释性和技能改进。提出ESC-Skills框架，通过发现和自进化技能来优化支持行为。**

- **链接: [https://arxiv.org/pdf/2605.27908](https://arxiv.org/pdf/2605.27908)**

> **作者:** Jie Zhu; Huaixia Dou; Shuo Jiang; Junhui Li; Lifan Guo; Feng Chen; Chi Zhang; Fang Kong
>
> **摘要:** Existing emotional support conversation (ESC) systems mainly rely on end-to-end response generation or coarse strategy supervision, offering limited interpretability and little support for systematic skill improvement. We propose ESC-Skills, a skill-centric framework that discovers and self-evolves executable emotional support skills. We first model localized support interactions as Intervention Units (IUs), which capture state--action--outcome dynamics between seeker states, support interventions, and post-response emotional changes. Based on IUs extracted from both successful and failed ESC dialogues, we construct the ESC-Skills Bank, a repository of executable emotional support skills containing intervention guidance, applicability conditions, expected outcomes, and potential risks. To further improve robustness, we introduce a multi-profile self-evolutionary refinement framework in which an ESC agent interacts with diverse simulated seeker profiles under SAGE evaluation. The resulting interaction traces are analyzed to identify missing skills, unsafe interventions, and profile-specific failure patterns, which are then used to refine the Skills Bank through simulation-based verification. Experimental results demonstrate that ESC-Skills improves both response-level quality and dialogue-level emotional outcomes while providing more interpretable and controllable support behaviors. We will release the code, prompts, and ESC-Skills Bank at this https URL.
>
---
#### [new 135] TARQ: Tail-Aware Reconstruction Quantization for Rare-Word Robust Automatic Speech Recognition
- **分类: cs.CL; cs.MM**

- **简介: 该论文属于自动语音识别任务，解决罕见词识别误差问题。提出TARQ方法，通过调整校准数据分布提升罕见词识别效果，无需标签或额外训练。**

- **链接: [https://arxiv.org/pdf/2605.27808](https://arxiv.org/pdf/2605.27808)**

> **作者:** Xinyu Wang; Ziyu Zhao; Ke Bai; Silin Meng; Dongming Shen; Xiao-Wen Chang; Yixuan HE
>
> **摘要:** Data-aware post-training quantization (PTQ) minimizes a per-token reconstruction loss on a small calibration corpus, implicitly weighting positions by their empirical frequency. For \textbf{A}utomatic \textbf{S}peech \textbf{R}ecognition (ASR), this misaligns with tail-sensitive risk: names, numerals, and domain-specific words receive proportionally little calibration mass. We propose \textbf{Tail-Aware Reconstruction Quantization} (\TARQ), a label-free PTQ framework that shifts calibration toward the lexical tail via \textbf{\rareBAL}, a closed-form per-Linear-layer rule equalizing common/tail mass, paired with a metric-consistent residual correction. \TARQ\ requires no entity labels, no curated calibration set, no validation decoding, and no additional training. Across eight ASR backbones and six datasets at W4G128, \TARQ\ improves mean rare-\textbf{W}ord \textbf{E}rror \textbf{R}ate (rare-WER) without an aggregate-WER regression, achieves the lowest cross-corpus rare-WER swing among compared methods, and transfers to entity-rich benchmarks (ProfASR, ContextASR-Speech-En) without entity supervision.
>
---
#### [new 136] When Seekers Are Hard to Help: Evaluating Emotional Support Dialogue Systems in Worst-Case Interactions
- **分类: cs.CL**

- **简介: 该论文属于情感支持对话系统评估任务，旨在解决现有系统在困难互动中的表现问题。通过构建 worst-case 评估框架，分析系统局限性并提升其鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.28228](https://arxiv.org/pdf/2605.28228)**

> **作者:** Jiajie Yang; Yangchun Li; Guanyi Chen; Rui Fan; Xin Bai; Tingting He
>
> **摘要:** Emotional Support Dialogue Systems (ESDSes) are increasingly evaluated and trained with LLM-simulated seekers. However, such simulated seekers often behave as cooperative, average-case users who disclose clearly, respond constructively, and accept support within a few turns. This can lead to overly optimistic evaluation and obscure whether ESDSes can handle difficult help-seeking interactions. In this work, we study ESDS evaluation under worst-case interactions, where seekers are hard to help due to low engagement, resistance, limited self-disclosure, emotional volatility, or rigid negative interpretations. We first conduct an expert simulation study with eight experienced counselling professionals, who simulate difficult seekers, interact with existing Chinese ESDSes, provide scale ratings, and participate in semi-structured interviews. Based on this study, we derive worst-case seeker behaviours and identify key limitations of current systems. We then propose a worst-case evaluation framework consisting of an LLM-based worst-case seeker simulator and four worst-case-oriented metrics: Deep Emotional Understanding, Guided Exploration, Balanced Emotional Support, and Authentic and Grounded Support. Evaluating 17 systems, we find that nearly all models suffer substantial performance drops under worst-case interactions. Large general-purpose LLMs are generally more robust than specialised ESDSes, but even the strongest models struggle to sustain engagement and improve seekers' emotional states. Finally, we show that worst-case simulation can also generate useful training data, improving the robustness of smaller models.
>
---
#### [new 137] Revisiting Anthropomorphic Reflection Markers in Large Language Model Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，探讨LLM中拟人化反思标记的作用。研究解决标记是否必要及如何影响推理性能的问题，通过抑制标记分析其影响，发现标记非必需且不影响反思行为。**

- **链接: [https://arxiv.org/pdf/2605.28305](https://arxiv.org/pdf/2605.28305)**

> **作者:** Yahan Yu; Noa Nakanishi; Fei Cheng
>
> **备注:** 15 pages, 12 figures
>
> **摘要:** Large Language Models (LLMs) often produce explicit reflective traces during complex reasoning, accompanied by anthropomorphic markers such as wait, hmm, and alternatively. Although these markers are commonly used as visible indicators of reflection, their mechanisms remain unclear, which leaves the risk of overthinking associated with redundant and repetitive reflection markers. In this work, we revisit anthropomorphic reflection markers, examining their necessity for reasoning and role in the reflection. We suppress these markers through prompt-level and token-level interventions, and analyze their effects on task performance across four benchmarks and two model scales. Our results show that anthropomorphic markers are not uniformly necessary for reasoning performance: suppressing them can preserve or improve performance in several settings, especially under larger sampling budgets. Meanwhile, marker suppression does not necessarily remove reflection behavior, as models can still perform marker-free verification. These suggest that anthropomorphic markers tend to be surface cues rather than reliable proxies for reflection itself, and motivate future research on reasoning mechanisms beyond explicit marker patterns.
>
---
#### [new 138] FABSVer: Faster Training and Better Self-Verification for LLM Mathematical Reasoning
- **分类: cs.CL**

- **简介: 该论文属于大模型数学推理任务，解决模型自我验证不可靠的问题。通过融合生成与验证任务，提出FABSVer，减少训练时间并提升性能。**

- **链接: [https://arxiv.org/pdf/2605.28389](https://arxiv.org/pdf/2605.28389)**

> **作者:** Haihui Pan; Junwei Bao; Hongfei Jiang; Yang Song
>
> **摘要:** While large language models have made significant progress in mathematical reasoning, they remain unreliable at judging the correctness of their own solutions. Existing approaches that equip models with self-verification typically treat solution generation and verification as two separate tasks, leading to substantially increased training time. In this paper, we propose FABSVer, which fuses these two tasks into a single generation pass, dramatically reducing training overhead while jointly optimizing both capabilities. We further identify a convergence bottleneck both theoretically and empirically: as training progresses, the reward reaches a plateau because the policy is constrained by a fixed reference model. To overcome this, we introduce Dynamic Reference Model Update (DRMU), which raises the reward ceiling and enables sustained reward growth. Extensive experiments on math benchmarks demonstrate that FABSVer achieves superior self-verification and reasoning performance across three model scales, while requiring only 51%--71% of the training time of existing methods. Analysis further reveals distinct learning phases in how models acquire self-verification, and that the gap between verify and answer rewards shrinks noticeably as model size increases.
>
---
#### [new 139] Skill-Conditioned Gated Self-Distillation for LLM Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型推理任务，解决如何有效利用技能库进行自蒸馏的问题。提出SGSD方法，通过技能条件引导教师验证，提升模型推理性能。**

- **链接: [https://arxiv.org/pdf/2605.28791](https://arxiv.org/pdf/2605.28791)**

> **作者:** Jiazhen Huang; Xiao Chen; Xiao Luo; Yong Dai; Senkang Hu; Yuzhi Zhao
>
> **摘要:** On-policy self-distillation (SD) improves LLM reasoning by using teacher-side privileged information (PI) to turn sparse verifier outcomes into dense token-level supervision. Existing methods usually assume trusted PI, such as reference answers or successful traces. We ask whether PI can instead come from an experience-derived skill bank, where retrieved skills are compact and reusable but may also be irrelevant or misleading. We propose Skill-Conditioned Gated Self-Distillation (SGSD), which formulates skill-based SD as teacher hypothesis validation rather than unconditional imitation. SGSD retrieves skill-mistake pairs, constructs a multi-teacher pool, and lets all skill-conditioned teachers score the same plain-prompt student rollout. The verifier validates each teacher's polarity: supporting a success or suppressing a failure gives positive supervision, while the opposite stance is reversed. A robust gated objective then distills informative teacher-student disagreements while suppressing uncertain or extreme signals. Experiments on multiple mathematical reasoning benchmarks show that SGSD consistently improves over GRPO and remains competitive with answer-conditioned OPSD under a weaker PI assumption. For example, on Qwen3-1.7B, SGSD outperforms GRPO by 6.2% and OPSD by 1.7% on average on AIME24, AIME25, and HMMT25. Our code is available at this https URL.
>
---
#### [new 140] Cultural Fidelity in English-to-Hindi Translation: A Preservation-Fluency Frontier for Gender Recoverability
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于机器翻译任务，研究如何在英译印过程中保持性别信息的可恢复性。针对系统常模糊性别问题，提出两种干预方法，提升性别保留率，但需权衡流畅性。**

- **链接: [https://arxiv.org/pdf/2605.27654](https://arxiv.org/pdf/2605.27654)**

> **作者:** Samyak Savi; Chavi Gupta; Shreyas Gantayet; Tanay Sodha; Dhruv Kumar
>
> **备注:** 10 pages, 2 figures, 9 tables
>
> **摘要:** Generative translation systems are cultural technologies because they decide how socially meaningful cues are rendered within culturally specific grammatical systems. We study one concrete notion of successful cultural translation: when an English source explicitly encodes gender, an English-to-Hindi translation should preserve the recoverability of that cue unless the source itself is ambiguous. We evaluate this criterion on a 37,345-instance benchmark spanning twelve categories and show that five systems frequently erase gender through ergative and honorific constructions. We then introduce two mechanism-aware inference-time interventions. The first, the Source-Aware Reranker (SAR), prefers candidates that avoid gender-neutralizing syntax. The second, the Phenomenon-Aware Reranker (PAR), preserves gender through targeted lexical marking even when ergative syntax remains. Across GPT-4o-mini and Sarvam, PAR improves target-subset accuracy from 11.07% to 54.47% and from 15.99% to 49.66%, respectively. Human evaluation shows that PAR increases gender preservation from 10.3% to 81.3%, but reduces mean fluency from 4.36 to 3.37. These findings place the two interventions on a preservation and fluency frontier rather than supporting a single dominant solution, and show how culturally situated generation can require explicit tradeoffs among fidelity, fluency, and stylistic naturalness.
>
---
#### [new 141] Identifying and Understanding Human Values in Text: A Tailorable LLM-based Architecture
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理任务，旨在识别文本中的人类价值观。解决如何有效检测并量化文本中隐含或显式的人类价值观问题，提出一种基于大语言模型的可配置架构。**

- **链接: [https://arxiv.org/pdf/2605.27373](https://arxiv.org/pdf/2605.27373)**

> **作者:** Eduardo de la Cruz Fernández; Marcelo Karanik; Sascha Ossowski
>
> **备注:** 8 pages, 1 figure. Published in Proceedings of the 18th International Conference on Agents and Artificial Intelligence (ICAART 2026), Volume 5
>
> **摘要:** As intelligent systems become more autonomous, the scientific community focuses on creating decision-making mechanisms that include ethical and moral considerations, unlike traditional utility-maximisation models. To achieve this, a key aspect is assessing how well these decisions align with human values. To this end, a promising line of research is centred on developing approaches based on Large Language Models (LLMs) to identify human values from text, whether explicit or implicit, enabling their recognition throughout. This paper introduces a LLM-based architecture to detect and quantify the intensity of human values in text, avoiding the limitations of previous approaches tied to specific value theory or complex prompt engineering. The architecture comprises three coordinated modules: one that generates structured value specifications from the foundational texts of any theoretical framework; one that labels texts using these specifications; and one that assigns graded support or resistance based on rhetorical and semantic evidence. This modular approach separates the tasks of conceptualising from detecting human values, creating a scalable and reproducible process driven by value specifications adaptable to various theories. The architecture was instantiated with multiple LLMs and evaluated using the ValueEval dataset. The experiments demonstrate good detection performance, confirming the generality of the pipeline.
>
---
#### [new 142] Personal Visual Memory from Explicit and Implicit Evidence
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文属于个性化AI长期记忆任务，解决现有系统忽视视觉中隐含用户信息的问题。提出VisualMem架构，融合视觉与文本记忆，提升个性化记忆效果。**

- **链接: [https://arxiv.org/pdf/2605.28806](https://arxiv.org/pdf/2605.28806)**

> **作者:** Viet Nguyen; Thao Nguyen; Vishal M. Patel; Yuheng Li
>
> **备注:** Project Page: this https URL
>
> **摘要:** Long-term memory is increasingly important for personalized AI agents, yet existing benchmarks and methods remain largely text-centric. Even when images are included, the user-specific information needed for later questions is typically recoverable from text alone, and most memory systems reduce image turns to generic captions. Yet images often carry personal information that text rarely states -- both explicit evidence, such as recurring user-associated entities, and implicit evidence, such as latent user facts inferred from visual or multimodal cues. We introduce a benchmark for personal visual memory that targets both forms of evidence, and propose VisualMem, a hybrid visual--text architecture that augments a text-memory backend with a structured personal visual memory module. Rather than collapsing images into captions, VisualMem uses conversational context to resolve identity, ownership, and durable user facts. Experiments show that VisualMem substantially outperforms prior memory systems on our benchmark while remaining competitive on standard text-memory benchmarks, indicating that personal visual memory is a distinct and important component of long-term memory for personalized AI agents.
>
---
#### [new 143] Explaining is Harder Than Predicting Alone: Evaluating Concept-based Explanations of MLLMs as ICL Visual Classifiers
- **分类: cs.AI; cs.CL; cs.LG; cs.LO; cs.MA**

- **简介: 该论文属于视觉分类任务，研究MLLM在少样本ICL中的解释能力。旨在评估模型是否能生成形式化概念解释，发现解释比预测更难，且形式化解释会降低准确率。**

- **链接: [https://arxiv.org/pdf/2605.28215](https://arxiv.org/pdf/2605.28215)**

> **作者:** Carmen Quiles-Ramírez; Leticia L. Rodríguez; Nicolás Martorell; Natalia Díaz-Rodríguez
>
> **备注:** Accepted to the CompLearn Workshop at ICML 2026
>
> **摘要:** In-context learning (ICL) enables multimodal large language models (MLLMs) to classify images from a few labelled examples. Yet, how these models use the provided context remains opaque. While Chain-of-Thought prompting is widely used, recent work argues that it may not reflect true internal computation. In this paper, we systematically evaluate the concept-based explainability of frozen MLLMs under few-shot ICL using five conditions of increasing formal rigour, ranging from baseline classification to Description Logics (DL) axiom generation. Evaluating four state-of-the-art MLLMs via an independent LLM-as-a-judge pipeline, we demonstrate that explaining is genuinely harder than predicting alone. Surprisingly, forcing models to generate formally structured, concept-based explanations degrades predictive accuracy monotonically (from 93.8% to 90.1%), contradicting the assumption that explicit reasoning universally aids performance. However, when models successfully articulate class-discriminative visual features, explanation quality strongly correlates with correct predictions. Our findings suggest that while MLLMs excel at visual classification, they lack the specific instruction-tuning required for formal, machine-verifiable explainability.
>
---
#### [new 144] Knowing When to Ask: Segment-Level Credit Assignment for LLM Tool Use
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型工具使用任务，解决模型无法准确判断何时调用工具的问题。提出CARL方法，通过分段信用分配提升模型自主决策能力。**

- **链接: [https://arxiv.org/pdf/2605.27788](https://arxiv.org/pdf/2605.27788)**

> **作者:** Abhijit Kumar; Zoey Wu; Mohit Suley
>
> **摘要:** Humans know when to reach for help e.g. $347 \times 28$ warrants a calculator while $2+2$ does not. Language models do not. Prompt-based approaches can instruct a model when to invoke tools, but this scaffolding does not teach it to recognize the boundary of its own knowledge. RL approaches that assign a single outcome reward to the whole trajectory fare no better: trajectory-level credit cannot isolate which tool call in a successful episode actually helped, nor penalize unnecessary calls. We propose \textbf{CARL} (\textbf{C}ompetence-\textbf{A}ware \textbf{R}einforcement \textbf{L}earning), which trains a critic on the model's own rollouts to learn where parametric knowledge suffices and where it needs external help. By decomposing each rollout at natural tool-use boundaries (e.g., code fence delimiters and context block transitions), CARL assigns independent credit to each segment from a single binary outcome, without external judges or step-level annotations. As a result, erroneous tool calls, incorrect extractions, and unnecessary calls each receive appropriately signed advantages. The trained critic captures the model's domain competence: it separates parametrically solvable from tool-dependent questions with AUC 0.93 at 7B. On five benchmarks spanning arithmetic, multi-hop factual QA, and numerical reasoning over financial tables, CARL improves exact-match accuracy by 6.7 points at 7B and 9.7 points at 3B over the best RL baseline, with the largest gain (+8.3 EM at 7B, +9.0 EM at 3B) on Musique. The model issues 53\% fewer tool calls on parametrically answerable questions while remaining ${\sim}10$ EM points more accurate on them. Gains are largest at small scale: the 3B improvement is $1.4\times$ the 7B improvement, suggesting that knowing when to ask disproportionately benefits models with smaller parametric memory.
>
---
#### [new 145] CAREF: Calibration-Aware Regularization for Explanation Faithfulness Without Rationale Supervision
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出CAREF框架，解决大模型解释性与准确性的平衡问题。通过联合优化预测准确性和解释可信度，实现轻量级微调。**

- **链接: [https://arxiv.org/pdf/2605.27835](https://arxiv.org/pdf/2605.27835)**

> **作者:** Naphat Nithisopa; Teerapong Panboonyuen
>
> **备注:** 10 pages
>
> **摘要:** We introduce CAREF, a parameter-efficient fine-tuning framework that jointly optimizes predictive accuracy and explanation faithfulness via calibration-aware regularization. At its core, CAREF couples entropy-based calibration with token-level sparsity control through a single unified loss, the Calibration-Aware Regularization for Explanation Faithfulness (LSCED), without requiring rationale supervision. Evaluated on four NLE benchmarks (COS-E, ECQA, ComVE, e-SNLI) with Flan-T5, our lightweight CAREF-AQ variant attains the best average accuracy (89.04) and explanation alignment (81.00 nBERT) using only 6.43% of trainable parameters, outperforming LoRA and AdaLoRA. To our knowledge, CAREF is the first method to unify entropy and sparsity regularization in a single training objective for interpretable LLM fine-tuning.
>
---
#### [new 146] Revealing Algorithmic Deductive Circuits for Logical Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于逻辑推理任务，旨在揭示LLMs如何通过注意力机制理解推理步骤与整体算法。工作包括定位关键注意力头，分析信息传递模式，并区分局部与全局推理机制。**

- **链接: [https://arxiv.org/pdf/2605.27824](https://arxiv.org/pdf/2605.27824)**

> **作者:** Phuong Minh Nguyen; Tien Huu Dang; Naoya Inoue
>
> **摘要:** Recent studies have shown that Large Language Models (LLMs) can achieve strong reasoning performance by incorporating functional symbolic representations that abstractly describe graph traversal algorithms and step-by-step reasoning in few-shot learning settings. However, it remains unclear how LLMs genuinely understand the abstract meaning of each reasoning step and the overall algorithm from only a limited number of demonstrations. This work aims to localize the attention heads responsible for individual reasoning steps and characterize the types of information transferred among them. We first align constituent reasoning steps with their corresponding token logits under a symbolic-aided Chain-of-Thought (CoT) prompting framework. Our analysis shows that token positions that steer the reasoning process are associated with low confidence scores caused by constraints on satisfying reasoning behavior patterns in demonstrations. We then adopt causal mediation analysis techniques to identify the attention heads responsible for these patterns. In addition, our findings indicate that LLMs retrieve factual and rule-based information for individual sub-reasoning tasks through specialized attention heads (approximately 3% total heads), whereas higher layers predominantly facilitate information integration and the emergence of global reasoning strategies (e.g., graph traversal algorithms) that coordinate multiple intermediate reasoning steps to solve the overall task.
>
---
#### [new 147] Generic Interpretation Approach for Transformer Models Incorporating Heterogenous Attention Structures
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于模型解释任务，旨在解决具有异构注意力结构的Transformer模型的可解释性问题。提出了一种解释方法，并通过实验分析其工作机制。**

- **链接: [https://arxiv.org/pdf/2605.27458](https://arxiv.org/pdf/2605.27458)**

> **作者:** Yongjin Cui; Xiaohui Fan; Huajun Chen
>
> **摘要:** Transformer has significantly propelled the development of artificial intelligence, and certainly the development of agents as well. We categorize attention structures of Transformer into two types based on the source of the input information: homogenous and heterogenous attention structures. Heterogenous attention structures, with co-attention as a typical example, process information from different sources. Heterogenous attention structure is the foundation for Transformer models to achieve more complex functions and integrate more modal information. Whether for research purposes or policy requirements, the interpretation of Transformer models with heterogenous attention structures is an important task. The fusion of information from different sources brings new challenges. Our work mainly includes two parts: method and experimentation. In terms of method, we propose an interpretation method for Transformer models with heterogenous attention structures. In terms of experimentation, based on our experimental analysis paradigm, we interpret the operating mechanisms of representative models, conduct semantic interpretation and logical interpretation.
>
---
#### [new 148] Learn from Weaknesses: Automated Domain Specialization for Small Computer-Use Agents
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于计算机使用代理的领域专业化任务，旨在解决小代理在特定领域表现弱的问题。通过引入LearnWeak框架，自动识别弱点并生成针对性训练数据，提升代理性能。**

- **链接: [https://arxiv.org/pdf/2605.28775](https://arxiv.org/pdf/2605.28775)**

> **作者:** Suji Kim; Kangsan Kim; Sung Ju Hwang
>
> **摘要:** Computer-use agents (CUAs) have recently made substantial progress, but deploying a separate large expert for each software domain remains expensive. Small open computer-use agents are more practical specialization targets, but they remain substantially weaker and exhibit uneven domain-specific failures. A straightforward remedy is to synthesize large-scale training data for the target domain, yet we find that this naive approach yields only marginal improvements. Building on this observation, we introduce LearnWeak, an annotation-free specialization framework for small computer-use agents that uses a stronger reference agent to identify the student's weaknesses in the target domain, synthesize targeted tasks, and construct supervision automatically. LearnWeak further introduces an error-aware specialization objective that disentangles planning and execution errors, enabling more behaviorally precise updates than broad uniform supervision. On OSWorld, LearnWeak achieves average gains of 11.6 and 11.1 percentage points over EvoCUA-8B and OpenCUA-7B, respectively, across eight domains. We also validate that our student-aware dataset generation and training approaches outperform existing autonomous trajectory generation and training baselines. Our work highlights the importance of student awareness in both data synthesis and agent training, pointing toward a more principled and efficient path for specializing small computer-use agents in diverse domains.
>
---
#### [new 149] Satisfiability Solving with LLMs: A Matched-Pair Evaluation of Reasoning Capability
- **分类: cs.AI; cs.CL; cs.LO**

- **简介: 该论文研究LLMs在SAT问题上的推理能力，解决其评估指标不准确的问题。通过配对公式和ADR指标，评估模型在不同表示下的稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2605.28602](https://arxiv.org/pdf/2605.28602)**

> **作者:** Leizhen Zhang; Shuhan Chen; Sheng Chen
>
> **备注:** Accepted at the ACM International Conference on the Foundations of Software Engineering (FSE 2026)
>
> **摘要:** Large language models (LLMs) are increasingly used for tasks that implicitly reduce to Boolean satisfiability (SAT), yet their reasoning ability on SAT remains unclear. We present a systematic study of LLMs on 2-SAT and 3-SAT, together with two canonical reductions, Vertex Cover and discrete 3D packing, to probe representation-invariant reasoning. We first evaluate models using conventional metrics, including accuracy, precision, recall, and F1, as well as the SAT phase-transition setting. We find that these metrics can be misleading: many models obtain high scores by over-predicting satisfiable formulas, fail to reproduce the classical easy-hard-easy signature around the 3-SAT threshold, and degrade sharply as the number of variables grows. To address this problem, we introduce a paired-formula protocol based on minimally different satisfiable and unsatisfiable instances, together with Accurate Differentiation Rate (ADR), which requires both members of each pair to be classified correctly. ADR separates reasoning-oriented models from heuristic ones and correlates with witness validity. Beyond CNF, we test cross-representation consistency by converting CNF to Vertex Cover and 3-SAT to discrete 3D packing. Model decisions on CNF and on the corresponding graph or packing instances agree for most models on more than 80 percent of instances, suggesting stable decision rules across representations. Overall, our results show that SAT is a conservative probe for LLM reasoning, and that paired evaluation with ADR provides a more faithful and representation-robust assessment than conventional metrics.
>
---
#### [new 150] MaskClaw: Edge-Side Personalized Privacy Arbitration for GUI Agents with Behavior-Driven Skill Evolution
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于隐私保护任务，解决GUI代理在操作中泄露敏感信息的问题。提出MaskClaw，在边缘侧进行隐私决策，避免上传原始屏幕。**

- **链接: [https://arxiv.org/pdf/2605.28646](https://arxiv.org/pdf/2605.28646)**

> **作者:** Yanqiu Zhao; Dongying Zheng; Kaibo Huang; Yukun Wei; Zhongliang Yang; Linna Zhou
>
> **备注:** Preprint. Submitted to EMNLP 2026. 21 pages, including appendices; 5 figures
>
> **摘要:** GUI agents rely on screenshots to infer intent and operate across applications, but these screenshots often contain private messages, medical records, payment credentials, and workplace-specific workflows. Privacy decisions in this setting depend on task, recipient, application state, and user role, yet static PII detectors miss these boundaries and cloud-side VLM reasoning can upload the raw screen before deciding what should be protected. We present MaskClaw, an edge-side privacy arbitrator for GUI agents. MaskClaw extracts local visual evidence, retrieves user- and task-specific policy memory, and decides Allow, Mask, or Ask before raw screenshots leave a trusted user- or organization-controlled environment. In five designed skill-evolution scenarios, it turns corrections, cancellations, and edits into reusable privacy skills checked by a sandbox gate. We introduce P-GUI-Evo, a benchmark built from real UI patterns, reconstructed HTML screens, and sanitized labels. Experiments show that pattern matching, cloud reasoning, and routing alone tend to over-confirm, over-mask, or expose raw screenshots under the same protocol. The artifact is available at this https URL.
>
---
#### [new 151] Agentic Separation Logic Specification Synthesis
- **分类: cs.PL; cs.CL; cs.SE**

- **简介: 该论文属于规范合成任务，旨在从C++代码中自动生成精确且可验证的规范。提出Spec-Agent系统，解决大规模代码库中规范表达不足与验证困难的问题，通过多阶段逻辑语言选择和迭代优化，提升规范质量与效率。**

- **链接: [https://arxiv.org/pdf/2605.27531](https://arxiv.org/pdf/2605.27531)**

> **作者:** Tarun Suresh; David Korczynski; Julien Vanegue
>
> **备注:** 9 pages, 3 appendices
>
> **摘要:** Specification synthesis, the task of automatically inferring formal specifications from program implementations and natural language, is important for refactoring, transpilation, optimization, and verification, yet remains an open challenge for large C++ repositories. Existing LLM-based approaches fail to simultaneously scale to such repositories, produce specifications expressive enough to capture systems-code features such as dynamic memory and heap-allocated data structures, and systematically validate those specifications to rule out incorrect candidates. We present Spec-Agent, an agentic system for synthesizing expressive, well-validated specifications across large C++ codebases. Spec-Agent targets a ladder of specification languages: propositional logic, first-order logic, propositional separation logic, and first-order separation logic. For each function, Spec-Agent uses static analysis and runtime heap tracing to select the appropriate target specification language, generalizes existing functional tests into fuzz harnesses, and iteratively refines LLM-generated candidates via counterexample-guided feedback. We evaluate Spec-Agent on open source C++ codebases comprising millions of lines of code. Spec-Agent synthesizes valid specifications for 85% of target functions, with no false positives observed under fuzzing and expert validation, outperforming Claude Code Opus 4.6 at 10x lower token cost.
>
---
#### [new 152] MIRA: A Bilingual Benchmark for Medical Information Response Audit
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出MIRA基准，用于评估大语言模型在不同用户表达下是否保持医疗信息的一致性。任务是检测信息稀释问题，通过实验发现模型在低健康素养提示下表现较差，并验证了缓解方法的有效性。**

- **链接: [https://arxiv.org/pdf/2605.28025](https://arxiv.org/pdf/2605.28025)**

> **作者:** Mengyu Xu; Qiaoxin Yang; Qianqian Wang; Xiwei Dai; Weiyi Wu; Chongyang Gao
>
> **摘要:** Large language models (LLMs) are increasingly used to provide public-facing health information, yet existing safety evaluations overlook whether responses preserve comparable medical information across different user phrasings of the same question. To address this, we introduce the Medical Information Response Audit (MIRA), a bilingual, controlled benchmark that assesses whether LLMs provide comparable medical information across user-side language, register, and health literacy signals. MIRA contains 4,320 prompts built from 60 medically reviewed, low-risk health questions. Across five mainstream LLMs, models answered all medical questions, but responses to low health-literacy signals consistently omitted more key information, provided fewer concrete next steps, and offered less support for independent judgment. We term this pattern Differential Information Dilution (DID). Language effects are model-specific rather than uniformly worse for non-English prompts. A comparison with 300 real-world health queries provides preliminary evidence of rank-order validity. A knowledge-guided mitigation prompt reduces information dilution for most models, with the largest reductions in underinformative simplification observed for Claude (~8%) and Qwen (~6%).
>
---
#### [new 153] Interpretability-Guided Layer Selection over Subspace Projection: SAEs as Stethoscopes, Not Scalpels, for Raw Task Vector Model Editing
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究模型编辑任务，旨在提升大模型的领域能力。针对SAEs在数学推理中的失效问题，提出将SAEs作为诊断工具而非干预手段，有效提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.28649](https://arxiv.org/pdf/2605.28649)**

> **作者:** Li Lei; Madalina Ciobanu; Qingqing Mao; Ritankar Das
>
> **摘要:** LLMs increasingly require surgical model editing to enhance domain-specific capabilities without incurring the computational cost or catastrophic forgetting associated with full fine-tuning. Sparse Autoencoders (SAEs) have emerged as a promising tool in this setting, in principle allowing for feature-level identification of where to intervene. In this work, we rigorously evaluate an SAE-guided editing pipeline for mathematical reasoning on Gemma-3-4B-IT and uncover a fundamental failure mode: the intuitively appealing approach of projecting task vectors onto SAE feature subspaces acts as an information bottleneck that discards approximately 97% of the modification energy, yielding no statistically significant improvements across seven math subjects. We show that this failure stems from a geometric misalignment between activation-space SAE directions and weight-space task vectors. We then propose a shift in perspective: SAE as a Stethoscope, Not a Scalpel, where SAEs are used for layer-level diagnosis rather than intervention-level filtering. By injecting unfiltered raw task vectors only into layers identified by an SAE-derived specificity score, we improve Number Theory accuracy from 29.6% to 39.4% (z=+3.41, p=0.0007) on the Minerva Math benchmark; 5 of 7 math subjects significantly improved and none significantly degraded. Our method is fully deterministic, requires no additional inference cost, and provides a principled framework for interpretability-guided model editing.
>
---
#### [new 154] Verified Misguidance: Measuring Structural Citation Failures in Search-Augmented LLMs
- **分类: cs.DL; cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决搜索增强型大模型中的引用结构失效问题。构建了CITETRACE数据集，并设计评估框架，发现模型引用真实但不合适的来源。**

- **链接: [https://arxiv.org/pdf/2605.28565](https://arxiv.org/pdf/2605.28565)**

> **作者:** Yongsik Seo; Wooseok Jeong; Eunyoung Kim; Hyeonseo Jang; Dongha Lee
>
> **备注:** Working Progress
>
> **摘要:** Users of search-augmented LLMs rely on citations as evidence that responses are grounded in real sources, and rarely verify the cited pages themselves. Millions of queries per day now pass through these systems, making citation quality a silent determinant of whether users are informed or misled-yet existing benchmarks each address one facet in isolation, leaving the joint structure that determines citation trustworthiness unmeasured. We construct CITETRACE, a large-scale dataset that traces the full citation chain from user query through retrieved source to generated answer: 11,200 real-world queries from 28 communities paired with 112,000 responses from ten models across five providers, yielding 761,495 evaluable citation pairs. We design a three-dimension evaluation framework that scores each citation on intent-purpose alignment, source suitability, and answer-source fidelity, using expert-validated predefined matrices and a five-level fidelity rubric; the framework applies to any system that produces citation-bearing responses. Applying this framework at scale, we identify a systematic pattern we call VERIFIED MISGUIDANCE (VM): models cite real, accessible sources yet fail along one or more dimensions, producing a fidelity-suitability trade-off in which faithful models select inappropriate sources and vice versa. Across our pool, 30.6% of citations distort their sources and 27.1% originate from domain-inappropriate sources; at the response level, up to 96% of users encounter at least one structurally misleading citation. Provider-level differences explain 88-96% of citation-quality variance, suggesting that source selection is governed more by factors beyond individual model capability than by the LLMs themselves. Together, CITETRACE and its evaluation framework provide the first resource for diagnosing structural citation failures in deployed search-augmented systems.
>
---
#### [new 155] Where Rollouts Begin: Low-Load, High-Leverage First-Token Diversification for RLVR
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于强化学习任务，解决RLVR中 rollout 多样性不足的问题。通过在第一个token处引入多样化，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.28295](https://arxiv.org/pdf/2605.28295)**

> **作者:** Soeun Kim; Albert No
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) trains reasoning models without labeled trajectories, relying on grouped rollouts to expose the policy to alternative reasoning paths and a verifier to score them. Rollout diversity has accordingly emerged as a central bottleneck in RLVR, with most existing methods broadening exploration through temperature, prefix, or rollout-selection adjustments. We identify a structurally distinguished but overlooked position for broadening this diversity: the first token after the reasoning marker. The policy's first-token distribution exhibits a sharply peaked yet correctness-decoupled phenomenon, and this first token position can broaden the regions a rollout group covers without altering the correctness signal. We introduce REFT (Rollout Exploration with First-Token Diversification), a light addition to the RLVR pipeline that samples first tokens uniformly from the policy's own top-$N$ candidates and allocates rollouts evenly, leaving every other component unchanged. Trained on the resulting diversified rollouts, REFT improves aggregate Pass@1, Pass@8, and Pass@64 over DAPO and GRPO baselines across four base models (0.5B-7B) and three difficulty regimes.
>
---
#### [new 156] You Only Align Once: Propagating Cooperative Behaviors in Multi-Agent Systems through Seed Agents
- **分类: cs.MA; cs.CL**

- **简介: 该论文研究多智能体系统中的对齐问题，通过种子智能体传播合作行为。任务是提升团队协作效率，解决未对齐智能体的协调难题。工作包括设计种子智能体并验证其传播效果。**

- **链接: [https://arxiv.org/pdf/2605.27586](https://arxiv.org/pdf/2605.27586)**

> **作者:** Nicole Hsing; Asuka Yuxi Zheng; Yi Zhao; Haoqin Tu; Jen-Tse Huang
>
> **摘要:** Ensuring agent behaviors in distributed open multi-agent systems remains challenging, especially as populations grow and unaligned agents may exist. We show that a single aligned agent can propagate cooperative behaviors to untrained agents purely through natural language interaction, a phenomenon we term Alignment Propagation. We study this in the Red-Black Game, a team-based iterated Prisoner's Dilemma in which teammates deliberate and vote to determine their team's collective action. By distilling the cooperative reasoning and persuasive dialogues of a teacher model into a Qwen-3-14B, we obtain a seed agent that, when placed among four untrained teammates, doubles the cooperation rate from 24.8% to 62.2%, outperforming the teacher model and a vanilla Gemini-3.1-Pro. Remarkably, a seed trained exclusively on the RedBlack Game transfers zero-shot to Sugarscape, a spatially grounded survival simulation with pairwise trading, achieving a 91.5% trade success rate versus a 21.6% baseline. Our results reframe multi-agent alignment from an exhaustive per-agent training problem to a scalable social capability that can be engineered through strategic seed placement.
>
---
#### [new 157] The Alignment Floor: When Persona Customization Is Safe
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文研究AI模型在个性化定制中的对齐问题，旨在解决过度定制导致对齐失效的风险。通过实验发现模型对齐强度影响定制安全性，并提出“对齐底线”作为设计原则。**

- **链接: [https://arxiv.org/pdf/2605.27382](https://arxiv.org/pdf/2605.27382)**

> **作者:** Xing Zhang; Guanghui Wang; Yanwei Cui; Wei Qiu; Ziyuan Li; Bing Zhu; Peiyang He
>
> **摘要:** A key promise of pluralistic AI is behavioral adaptation: persona prompts like "be creative" or "be thorough" let systems respect diverse user values and communication styles. But how much customization can a model absorb before its alignment breaks? We present the first controlled study of the alignment-customization tradeoff, testing seven persona conditions across five tasks on two models with different alignment strengths (1,800 runs). We discover the alignment floor: on a strongly-aligned model (Claude Sonnet), persona prompts have zero effect on sycophancy -- all conditions produce ~15%, a stable platform on which rich personalization is safe. On a weakly-aligned model (Nova Lite), the same personas shift sycophancy from 5% to 50% -- the floor is absent and customization becomes a safety liability. Surprisingly, Agreeableness is not the worst offender; Extraversion (+20pp) and Openness (+15pp) cause greater degradation. The constructive finding is the Skeptic defense: a critical-thinking persona reduces sycophancy to 5% even on the weak model -- the single largest effect in the study. Cross-model transfer of persona effects is near-zero ($\rho = 0.006$), meaning alignment testing must be per-model. We propose the alignment floor as a design principle: measure it before deploying persona customization, and layer safety-oriented personas underneath user-facing ones to enable personalization without compromising alignment.
>
---
#### [new 158] Adaptive Multimodal Agents-Based Framework for Automatic Workflow Execution
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于智能代理任务，旨在解决复杂工作流自动执行问题。提出一种多模态多代理框架，通过构建拓扑知识库和自适应生成技术，提升代理在非静态环境中的导航与自修正能力。**

- **链接: [https://arxiv.org/pdf/2605.28607](https://arxiv.org/pdf/2605.28607)**

> **作者:** Susanna Cifani; Mario Luca Bernardi; Marta Cimitile
>
> **备注:** Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses. Accepted for publication at the 2026 IEEE International Conference on Evolving and Adaptive Intelligent Systems (EAIS 2026)
>
> **摘要:** Modern information systems require autonomous agents capable of navigating complex workflows, yet current methodologies often struggle with the transition from structured metadata parsing to general environmental perception. While the integration of MLLMs has enabled agents to interact directly with GUIs, existing approaches typically treat task sequences as discrete, linear episodes. This fragmentation prevents agents from capturing the underlying transition topology, limiting their effectiveness in novel or non-stationary scenarios. To address this, we propose a novel multimodal multi-agent framework that achieves automatic workflow execution through a distinct two-phase pipeline. First, during an offline discovery phase, the architecture adaptively constructs a topological knowledge base from fragmented execution logs. During inference, agents leverage Adaptive Retrieval-Augmented Generation (RAG) over this fixed, pre-established graph, coupled with a closed-loop collaborative verification protocol to dynamically self-correct and navigate. This graph-based approach facilitates superior task decomposition and adaptive navigation performance. We validate our framework in a real-world context, demonstrating its ability to maintain high reliability and semantic awareness even with limited training data.
>
---
#### [new 159] GraphSteal: Structural Knowledge Stealing from Graph RAG via Traversal Reconstruction
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于知识隐私保护任务，旨在解决Graph RAG中结构信息泄露问题。通过重构知识图谱，揭示敏感实体与关系，验证现有防护措施不足。**

- **链接: [https://arxiv.org/pdf/2605.28645](https://arxiv.org/pdf/2605.28645)**

> **作者:** Jinze Gu; Qinghua Mao; Xi Lin; Jun Wu
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances LLMs by grounding generation in query-relevant external evidence. Beyond unstructured text corpora, Graph RAG integrates knowledge graphs into the retrieval pipeline, enabling LLMs to access entities, relations, and multi-hop dependencies encoded in structured knowledge. However, the same structured knowledge that empowers Graph RAG also creates a new privacy attack surface. We demonstrate that Graph RAG systems can be turned into structural oracles: through adaptive black-box interactions, an adversary can elicit sufficient relational evidence to reconstruct substantial portions of the hidden knowledge graph. We propose a structure-oriented reconstruction framework that recovers targeted graphs from both local and global perspectives. Specifically, Depth-Wise Heuristic Search extracts fine-grained node attributes by recursively expanding entity-centered evidence, while Breadth-Wise Diffusion Search infers graph topology by propagating across relation-induced neighborhoods. Experiments on generic and healthcare scenarios demonstrate that our method can recover over 90\% of the original knowledge graph from representative Graph RAG systems, revealing sensitive entities, relations, and structural dependencies with high fidelity. Existing guradrails provide limited defense against our attack, highlighting the inherent difficulty of safeguarding structural privacy in Graph RAG pipelines.
>
---
#### [new 160] When Think-with-Image Meets Safety: What Determines Multimodal Jailbreak Robustness?
- **分类: cs.CV; cs.AI; cs.CL; cs.CR; cs.LG**

- **简介: 该论文属于安全评估任务，研究多模态模型的对抗攻击鲁棒性。通过实验比较不同设计模式，发现显式图像工具交互能有效提升安全性。**

- **链接: [https://arxiv.org/pdf/2605.27932](https://arxiv.org/pdf/2605.27932)**

> **作者:** Yuan Tian; Bing Hu; Fang Wu; Xiaomin Li; Binghang Lu; Neil Zhenqiang Gong
>
> **备注:** 17 pages, 6 figures, 7 tables
>
> **摘要:** Think-with-image reasoning is emerging as a new inference paradigm for large vision-language models, but its safety implications remain poorly understood. Existing systems already span multiple process designs, including direct response generation, text-only prior turn, visual-state manipulation, and explicit external image-tool invocation. In this paper, we ask which of these evaluated paradigms improves multimodal jailbreak robustness, and why. Across multiple vision-language models, explicit image-tool interaction yields the lowest attack success rates in our experiments, reducing jailbreak success by around 30% relative on average across the evaluated models. This finding is initially surprising: ASR remains low even when the returned image-tool output is manually overridden or itself unsafe-looking, but returns near direct-answering levels under text-only prior turn controls. These results indicate that the lower ASR is not explained by benign returned-image semantics or by the textual image-tool trace alone. To explain the pattern, we introduce an image-tool safety vector framework that models image-tool invocation as a residual shift in hidden representations toward a safety-relevant direction. Representation-level analyses and activation interventions support this account. Overall, our results suggest that explicit image-tool interaction is a promising design pattern for improving jailbreak robustness, while also motivating pipeline-specific safety evaluation.
>
---
#### [new 161] Grounded Cache Routing for Retrieval-Augmented Generation: When Is It Safe to Reuse an Answer?
- **分类: cs.CR; cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于检索增强生成任务，解决缓存答案安全 reuse 的问题。提出 GroundedCache，通过四个验证机制确保缓存答案的可靠性，显著降低错误率。**

- **链接: [https://arxiv.org/pdf/2605.27494](https://arxiv.org/pdf/2605.27494)**

> **作者:** Syed Huma Shah
>
> **备注:** 19 pages, 9 figures, 10 tables. Code: this https URL
>
> **摘要:** Modern retrieval-augmented generation(RAG) deployments increasingly rely on caching to reduce token cost and time-to-first-token(TTFT). Prefix-level KV reuse is now standard in serving stacks such as vLLM, and chunk-level and position-independent reuse have been pushed further by recent systems(RAGCache, TurboRAG, CacheBlend, EPIC, ContextPilot, PCR, LMCache). Output-level semantic answer caches, by contrast, remain fragile: similar prompts can map to different correct answers, retrieved evidence drifts as the corpus is updated, and adversarial collision attacks have been shown to hijack cached responses. We argue that the right framing for cached answer reuse is not how to reuse faster but when reuse is safe. We propose GroundedCache, an evidence-validated cache router that admits a cached answer only when 4 cheap gates simultaneously hold: query similarity, retrieved-evidence overlap, source-version validity, and lexical (or judge-based) support of the cached answer by the freshly retrieved evidence. We build a six-regime workload that stress-tests cache safety rather than only hit rate, and introduce an operator-facing metric, the unsafe-served rate (USR), fraction of all queries that received a wrong cached answer. Across 2 datasets and 12,000 real-LLM generations(Qwen2.5-7B-Instruct on vLLM with Automatic Prefix Caching), GroundedCache drives USR to 0.0% on every HotpotQA regime(vs. 15-35% under naive caching) and to 1.5% on mtRAG document drift(vs. 51.5%), a 34x reduction on the design-point adversarial regime and 3-10x reductions across the other mtRAG regimes, while end-to-end p50 latency stays within 1.04-1.07x of a no-cache RAG baseline. A per-gate ablation isolates the lexical support gate as the load-bearing safety mechanism on both datasets, with the remaining gates providing defense-in-depth at near-zero cost. We release the implementation, workload, and evaluation harness.
>
---
#### [new 162] From Instructor to Collaborator: What a 90-Participant Study Reveals about Human-Agent Collaboration in a Mobile Serious Game
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于人机协作研究，探讨用户与智能代理在移动游戏中的互动。通过实验比较不同代理类型的影响，分析用户偏好与交互效果，旨在理解协作中的角色与对话机制。**

- **链接: [https://arxiv.org/pdf/2605.27384](https://arxiv.org/pdf/2605.27384)**

> **作者:** Danai Korre
>
> **备注:** 4 pages, 5 figures, ACM CHI 2026 workshop paper
>
> **摘要:** This position paper reflects empirical data collected during my PhD from a large-scale within-subjects study (N = 90). The study compared a highly human-like, spoken embodied conversational agent (ECA) against a low human-like text base agent (no embodiment, text bubble only) within a mobile, Unity-developed game about pre-decimal UK currency. The game included two agents with different roles-an Instructor (Alex) and a Shopkeeper/Collaborator. Users interacted using voice and mouse input. The quantitative data I collected included a usability questionnaire (CCIR MINERVA) and the Agent Persona Instrument. Data was analyzed using paired t-test, repeated measures ANOVA and multiple linear regression to identify correlations between the persona and usability. The results showed a statistically significant preference for the version of highly human-like agents, with a large effect size. This is further discussed alongside qualitative findings from observations and exit interviews. The results are framed for Human-Agent collaboration, especially for how roles, mixed-initiative dialogue, and breakdowns/repairs become apparent in goal-oriented tasks. I conclude with questions on timing, user expectations, and role-specific interactions. This submission does not propose new frameworks; it reports empirical findings and questions I hope to workshop with the community.
>
---
#### [new 163] Agents that Matter: Optimizing Multi-Agent LLMs via Removal-Based Attribution
- **分类: cs.MA; cs.CL**

- **简介: 该论文属于多智能体系统优化任务，解决 agent 贡献评估问题。通过构建合作博弈框架，提出基于移除的归因方法，有效识别瓶颈 agent 并提升系统性能。**

- **链接: [https://arxiv.org/pdf/2605.27621](https://arxiv.org/pdf/2605.27621)**

> **作者:** Mingyu Lu; Yushan Huang; Chris Lin; Su-In Lee
>
> **摘要:** As multi-agent systems (MAS) become increasingly complex, identifying the contributions of individual agents is critical for system optimization. However, existing approaches lack a rigorous, unified framework for credit assignment. In this work, we formalize agent attribution as a cooperative game, parameterized by the coalition distribution, removal protocol, and target metric. Using this framework, we show that Leave-One-Out (LOO) identifies bottleneck agents as effectively as combinatorial methods, but at a fraction of the computational cost. We also demonstrate that removal protocols induce distinct games: Agent ablation isolates structural bottlenecks, whereas introspective LLM judges fail to faithfully approximate this behavior. Furthermore, to evaluate the utility of specific agent backbones, we introduce attribution via model replacement. By substituting underlying models of low-contribution agents, we improve task performance by up to 17% while reducing cost by up to 35% across three benchmarks. Finally, we apply our framework to audit a medical MAS, revealing that agent contributions to diagnostic accuracy and ethical behavior are often decoupled. By intervening on counterproductive roles, we observe an increase in ethics alignment while maintaining diagnostic accuracy. Overall, this work provides a principled approach for cost-effective MAS attribution and intervention.
>
---
#### [new 164] Cultural Binding Heads in Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究语言模型的文化绑定问题，旨在提升模型对不同文化的区分能力。通过分析注意力头和知识路由，发现模型在预训练阶段已具备文化绑定能力，并提出方法增强文化差异化表现。**

- **链接: [https://arxiv.org/pdf/2605.28543](https://arxiv.org/pdf/2605.28543)**

> **作者:** Avrile Floro; Luca Benedetto
>
> **摘要:** LLMs often default to equal treatment across cultural groups, even though context warrants differentiation: this is a lack of difference awareness. Using mechanistic interpretability and a factorial design on the N4 cultural appropriation benchmark from Wang et al. (2025), we identify 2-3 mid-layer attention heads per model that contribute causally to cultural binding across eight models (four architectures, base and instruct). Cultural binding is the process of associating cultural items with the appropriate identity. Knockout of the identity-to-item edges on these heads lowers the binding strength by 9-23%. The identified heads transfer from instruct to base models, suggesting that cultural binding is created at pre-training. An $\alpha$-scaling shows a graded dose-response and moderate amplification steering at generation ($\alpha = 2-3$) increases cultural differentiation accuracy by 1-3 pp while leaving neutral reasoning mostly intact. A knowledge probing task shows that models know 3-5 times more than they act upon it, indicating that the bottleneck lies in routing and not knowledge.
>
---
#### [new 165] A Wolf in Sheep's Clothing: Targeted Routing Hijacking in Federated RAG
- **分类: cs.CR; cs.CL; cs.IR**

- **简介: 该论文属于联邦检索增强生成（FedRAG）任务，针对路由阶段的安全问题，提出路由劫持攻击并设计防御方案以保障路由完整性。**

- **链接: [https://arxiv.org/pdf/2605.28112](https://arxiv.org/pdf/2605.28112)**

> **作者:** Junjie Mu; Qiongxiu Li
>
> **备注:** Under review. Code available at this https URL
>
> **摘要:** Federated Retrieval-Augmented Generation (FedRAG) is attractive for privacy-sensitive applications because raw data remain local. As a result, routing must rely on client-provided semantic profiles, creating a new opportunity for manipulation. We introduce Routing Hijacking, a routing-stage attack in which a malicious client forges its profile to attract target queries despite having irrelevant underlying data. We show that this vulnerability is severe. Across three representative FedRAG routing architectures, Routing Hijacking consistently misroutes target queries and leads to downstream disruptions and failures, including missing evidence, poisoning, incorrect answers, and hallucinations. In a high-stakes MedQA-USMLE case study, we further show that poisoned retrieved evidence can mislead models across scales, leading to incorrect answers, hallucinations, and sycophantic failures. Existing defenses do not close this gap: encrypted routing preserves the exploited ranking, and Byzantine-robust Federated Learning (FL) rules transfer poorly to heterogeneous routing profiles. To address this gap, we propose a trust-aware post-routing framework that reweights clients using returned-evidence feedback, including retrieval relevance, profile consistency, and cross-client agreement; online experiments show that it suppresses persistent hijacking over recurring queries and transfers to a learned neural router. Our findings establish routing integrity as a new security challenge in FedRAG and highlight the need for stronger defenses for secure federated retrieval.
>
---
#### [new 166] AI, Take the Wheel: What Drives Delegation and Trust in Human-Computer Cooperative Question Answering?
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文研究人机协作问答中的信任与委托问题，分析人类如何决定依赖AI。任务是提升人机合作效率，解决人类在判断AI可靠性上的偏差。通过实验发现人类存在过度或不足依赖现象，并提出改进建议。**

- **链接: [https://arxiv.org/pdf/2605.28255](https://arxiv.org/pdf/2605.28255)**

> **作者:** Maharshi Gor; Yoo Yeon Sung; Yu Hou; Eve Fleisig; Irene Ying; Tianyi Zhou; Jordan Boyd-Graber
>
> **备注:** Findings of the Association for Computational Linguistics, 2026
>
> **摘要:** AI systems are fallible, and humans can make mistakes in deciding whether to trust AI over their own judgment. Thus, improving human-AI collaboration requires understanding when, why, and how humans decide to rely on AI. We study two distinct reliance decisions: the delegation choice -- deciding when to let AI act autonomously without knowing its output, and the adoption choice -- evaluating AI suggestions and deciding how to use them. Both of these decoupled reliance patterns shape collaboration, but prior work rarely studies them together in realistic settings with the same users. We address this gap by studying collaborative human--AI teams competing in a question-answering game in which humans can choose when and how to work with AI agents to win. Our 24 matches pair 23 expert humans with 16 AI agents, capturing 387 delegation and 1440 adoption decisions. While human--AI collaboration performs better than either AI or humans alone, humans make suboptimal collaboration decisions, both under-relying on correct AI suggestions (3.9% of opportunities missed) and over-relying when AI misleads them (1.7%). Both parties contribute wrong answers: reported model confidence is near chance when humans and AI disagree, while confirmation bias drives higher under-reliance (64.5%) when an AI suggestion agrees with humans' initial incorrect answer. To close this gap, we recommend calibrated confidence, evidence-grounded explanations, and mechanisms that help users refine trust.
>
---
#### [new 167] Discovery Agents for Real-Time Analytics: Toward Proactive Insight Systems
- **分类: cs.AI; cs.CL; cs.DB**

- **简介: 该论文提出一种多智能体架构，用于实时数据流中的自主洞察发现，解决传统分析系统反应性不足的问题，通过持续发现循环实现主动分析。**

- **链接: [https://arxiv.org/pdf/2605.27571](https://arxiv.org/pdf/2605.27571)**

> **作者:** Gaetano Rossiello; Dharmashankar Subramanian
>
> **备注:** Accepted at Supporting Our AI Overlords (SAO) at the ACM Conference on AI and Agentic Systems (CAIS), May 26 2026, San Jose, CS, USA
>
> **摘要:** Modern analytics systems are fundamentally reactive, requiring users to define queries over increasingly complex and continuously evolving data. In real-time streaming environments, this paradigm breaks down, as the space of potential insights becomes too large to enumerate manually. We present a multi-agent architecture for autonomous insight discovery over real-time data streams. The system implements a continuous discovery loop in which agents generate hypotheses, compile them into executable analytics, validate generated artifacts, and produce visualizations and deployable applications. The architecture leverages Apache Kafka for event-driven coordination, Apache Flink for stream processing, and large language models to implement specialized agents. A key contribution is a contract-driven design based on typed intermediate artifacts, enabling modularity, observability, lineage, and safer execution of dynamically generated analytics. Through use cases in retail, finance, and public data, we show how this architecture supports a shift from query-driven analytics to proactive, discovery-driven systems.
>
---
#### [new 168] VCap: Hypergeometric Rewards for Weak-to-Strong Visual Captioning
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于视觉问答任务，解决 captioning 中事实一致性验证问题。提出 VCap 奖励机制，通过对比参考 caption 与视觉信号，提升生成质量与准确性。**

- **链接: [https://arxiv.org/pdf/2605.28023](https://arxiv.org/pdf/2605.28023)**

> **作者:** Xingyu Lu; Jinpeng Wang; Yi-Fan Zhang; Yankai Yang; Yancheng Long; Yiyang Fan; Xuanyu Zheng; Haonan Fan; Kaiyu Jiang; Tianke Zhang; Changyi Liu; Bin Wen; Fan Yang; Tingting Gao; Han Li; Chun Yuan
>
> **备注:** 28 pages, 8 figures
>
> **摘要:** Visual captioning requires models to capture visual content faithfully while minimizing both omission and hallucination. As the dominant paradigm for captioning, MLLMs have achieved strong performance through scaling and high-quality data. Recently, RL has emerged as a key route to driving MLLMs toward higher precision and broader coverage, however, existing reward designs for captioning fail to provide fine-grained and reliable signals for factual verification, limiting their effectiveness. To address this, we propose VCap, a Witness-Adjudicator reward that pairs the reference caption (a witness) with the visual signal (an adjudicator). By explicitly verifying factual consistency between the reference and policy-generated captions grounded in the visual signal, VCap delivers a reward signal with hypergeometric-distribution-level precision for caption quality verification. This design enables effective learning even from imperfect references, facilitating weak-to-strong generalization in RL training. In our experiments, an 8B model trained with VCap outperforms open- and closed-source SOTA models on multiple image and video captioning benchmarks. Human evaluation further confirms its strong alignment with factual correctness. Additionally, VCap improves MLLM perceptual capability, generalizes across tasks, and surpasses best-of-N distillation, challenging prior assumptions about RLVR.
>
---
#### [new 169] MemCog: From Memory-as-Tool to Memory-as-Cognition in Conversational Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出MemCog，解决对话系统中记忆使用效率低的问题，通过将记忆融入推理过程，提升记忆主动性与结构适配性。**

- **链接: [https://arxiv.org/pdf/2605.28046](https://arxiv.org/pdf/2605.28046)**

> **作者:** Zihan Li; Xingyu Fan; Feifei Li; Wenhui Que
>
> **摘要:** Existing agent memory systems universally follow what we term a Memory-as-Tool paradigm where a single query triggers one-shot retrieval of flat passage lists, suffering from passive invocation, reasoning-retrieval decoupling, and structural mismatch between retrieved fragments and the agent's navigational needs. We propose MemCog, a Memory-as-Cognition system that makes memory access an integral part of the reasoning process. MemCog organizes user knowledge as Navigable Memory Store with associative link graphs, exposes Cross-Dimensional Navigation Interface for multi-step reasoning-driven traversal, and employs Proactive Reasoning Protocol that drives agents to spontaneously initiate memory exploration from conversational context. We additionally construct ProactiveMemBench, the first benchmark for evaluating proactive memory triggering. Experiments show that MemCog achieves state-of-the-art on passive QA benchmarks (92.98 on LoCoMo, 95.8 on LongMemEval) while substantially outperforming baselines on ProactiveMemBench, demonstrating the advantage of Memory-as-Cognition.
>
---
#### [new 170] MIRAGE: Context-Aware Prompt Injection against Mobile GUI Agents via User-Generated Content
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文提出MIRAGE，针对移动GUI代理的提示注入攻击方法，通过用户生成内容区域植入恶意文本，解决视觉语言模型无法区分可信元素与用户内容的问题。**

- **链接: [https://arxiv.org/pdf/2605.28116](https://arxiv.org/pdf/2605.28116)**

> **作者:** Ruoqi Guo; Yi Liu; Gelei Deng; Yiheng Xiong; Yuekang Li; Ying Zhang; Leo Yu Zhang; Lida Zhao; Ji Jie; Yuxiao Lu
>
> **摘要:** Mobile graphical user interface (GUI) agents driven by vision-language models (VLMs) perceive the screen as rendered pixels and choose actions from what they see, so they cannot reliably separate trusted interface elements from user-generated content. We present MIRAGE (Mobile Injection of Realistic Adversarial GUI Examples), a pipeline that turns benign mobile screenshots into prompt-injection samples by placing attacker-controlled text into ordinary user-generated content regions, without modifying the agent, the application, or the operating system. MIRAGE operates in three stages: a Localizer identifies user-controllable regions on the screenshot, a Generator synthesises context-aware payloads and renders them in the application's native style, and a Curator moderates realism and balances the samples across applications, region types, and attack intents. A key challenge is that an injected screenshot must stay visually indistinguishable from genuine user content while still diverting the agent; we address this by separating the stages that control reach, realism, and distributional balance. On a 1,111-sample benchmark spanning ten applications and eleven attack intents, all five evaluated VLM agents are vulnerable, with attack success rates of 23%-30%, and MIRAGE scores higher on human realism ratings than the strongest prior attack (3.02 versus 2.52 out of 5). We further find that per-sample realism and attack success are uncorrelated, so visual-quality filtering alone cannot reliably defend against this threat.
>
---
#### [new 171] Aligning LLMs with Human Uncertainty: A Beta-Bernoulli Calibrator for LLM Forecasting
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于概率预测任务，旨在提升大语言模型的预测准确性与不确定性估计。通过引入Beta-Bernoulli校准器，利用人类预测信息优化模型输出，提高预测可靠性。**

- **链接: [https://arxiv.org/pdf/2605.27668](https://arxiv.org/pdf/2605.27668)**

> **作者:** Hui Dai; Ryan Teehan; Parsa Torabian; Mengye Ren
>
> **摘要:** Probabilistic forecasting estimates the likelihood of uncertain future events. To improve LLM forecasting, existing methods typically learn from binary outcomes to output verbalized forecasts. However, while aggregated human forecasts contain rich information in both the crowd probability estimate and the degree of agreement among forecasters, how to utilize these signals remains underexplored. To address this, we propose the Beta-Bernoulli Calibrator (BBC), which converts an initial point estimate forecast from any model into a distribution over event likelihood, using supervision from both binary outcomes and human forecasts. BBC models event likelihood $p \sim \text{Beta}(\alpha, \beta)$ and outcome $y \sim \text{Bernoulli}(p)$, with the mean as the calibrated point forecast and the variance as the epistemic uncertainty. Our results show that BBC generally provides better calibrated and more accurate forecasts than both traditional post-hoc calibration methods and models fine-tuned specifically for forecasting, while remaining lightweight and having good generalization. We also show that the epistemic uncertainty captured by BBC is a more reliable predictor of forecasting error than verbalized confidence.
>
---
#### [new 172] A Fixed-Budget, Cluster-Aware Standard for LLM-as-a-Judge Evaluation: A Multi-Hop RAG Stress Test
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于RAG系统评估任务，旨在解决LLM作为评判者时的测量标准不统一问题。提出固定预算的集群感知标准，并通过实验验证其有效性。**

- **链接: [https://arxiv.org/pdf/2605.27789](https://arxiv.org/pdf/2605.27789)**

> **作者:** Camilo Chacón Sartori; José H. García
>
> **摘要:** Retrieval-augmented generation (RAG) systems are often compared by asking a large language model (LLM) judge which answer is better. For multi-hop RAG, this has become a measurement problem as much as a modeling problem: the same score can reflect retrieval quality, answer length, lexical overlap, or a statistical test that ignores clustered data. We ask what happens when these choices are made explicit. We propose a minimum measurement standard for LLM-as-a-judge comparisons in RAG. The standard fixes the top-100 candidate pool, evidence budget, answer cap, generator, and prompt; it also requires pre-registered hypotheses, cluster-aware inference, an exact cluster sign-flip check when feasible, and second-judge replication. Clustered benchmarks can overstate progress; the field should adopt this standard. We stress-test it with Genetic Algorithm Decoder for Multi-hop Evidence Composition (GADMEC), an evolutionary evidence selector, on 400 multi-hop questions in computer science/machine learning (CS/ML) and Materials Science. The protocol changes the empirical story. A binomial test makes all four semantic-baseline comparisons look significant; cluster-aware inference leaves only one Bonferroni-significant result. BM25 beats pure semantic GADMEC under the same budget, while a lexical-semantic hybrid recovers in CS/ML and narrows the Materials Science gap.
>
---
#### [new 173] Long Live the Librarian! A Persistent Search Sub-Agent for Energy-Efficient Multi-Agent Software Engineering Systems
- **分类: cs.MA; cs.CL**

- **简介: 该论文属于多智能体软件工程任务，旨在解决多智能体系统能耗过高的问题。通过引入Librarian子代理，减少冗余输出，降低能耗。**

- **链接: [https://arxiv.org/pdf/2605.27787](https://arxiv.org/pdf/2605.27787)**

> **作者:** Seunghyuk Cho; Sunghyun Choi; Jaeseung Heo; Youngbin Choi; Saemi Moon; MoonJeong Park; Dongwoo Kim
>
> **备注:** 19 pages, 4 figures, 12 tables
>
> **摘要:** Multi-agent systems (MAS) have substantially advanced autonomous software engineering (SWE), but their growing inference energy demands raise sustainability concerns. In this paper, we demonstrate that this cost is concentrated in an overlooked source: redundant output tokens generated across agents. Two empirical findings ground this claim. First, our per-token energy attribution for MAS reveals a sharp asymmetry: an output token consumes 30 to 1,000 times more energy than an input or cached token. Second, MAS inflate per-episode output because agents repeatedly re-explore overlapping repository regions. To address this inefficiency, we propose Librarian, a persistent search sub-agent that tracks repository-search history and suppresses redundant exploration actions across agents. By returning short references to file regions instead of full file excerpts, Librarian further reduces output-token volume. On SWE-Bench Verified, Librarian reduces per-episode GPU energy consumption of existing multi-agent SWE systems by up to 25% while preserving task performance.
>
---
#### [new 174] Entropy-aware Masking for Masked Language Modeling
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语言模型预训练任务，旨在提升掩码语言建模效果。通过引入熵感知的掩码策略，选择更具信息量的token进行掩码，提高训练效率与性能。**

- **链接: [https://arxiv.org/pdf/2605.28526](https://arxiv.org/pdf/2605.28526)**

> **作者:** Gokul Srinivasagan; Kai Hartung; Munir Georges
>
> **备注:** accepted at starsem 2026 Conference
>
> **摘要:** Masked language modeling has become a standard pretraining objective for training encoder-based language models. In this approach, certain tokens in the input are masked, and the model learns to predict them using the surrounding context. This process enables the model to capture both syntactic and semantic properties of language. Conventionally, the tokens selected for masking are chosen at random, which may not always yield the most effective learning signals. In this work, we examine a token masking strategy based on entropy distribution. We use the model's entropy over token predictions to identify which tokens should be masked. This method aims to target tokens that are more informative and uncertain to improve the training efficacy. We also propose a novel self-masking approach that enhances training efficiency without relying on an external reference model. Experimental results demonstrate that our method achieves an average performance improvement of 5% in GLUE scores compared to the baseline. Further, we experiment with combining knowledge distillation with entropy masking, resulting in the best overall results.
>
---
#### [new 175] Skill-as-Pseudocode: Refactoring Skill Libraries to Pseudocode for LLM Agents
- **分类: cs.PL; cs.CL**

- **简介: 该论文属于LLM代理技能库优化任务，解决技能描述不清晰导致的调用错误问题。通过将技能转换为带类型签名的伪代码，提升代理的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2605.27955](https://arxiv.org/pdf/2605.27955)**

> **作者:** Xinze Li; Yuhang Zang; Yixin Cao; Aixin Sun
>
> **备注:** Preprint. Code: this https URL
>
> **摘要:** Markdown skill libraries for LLM agents ship as free-form prose, forcing the agent to re-derive both the input schema and the concrete invocation syntax on every retrieval. We observe that this often produces a "confused -> re-retrieve -> still confused" loop in which the agent issues a partially-correct action, receives uninformative environment feedback, and re-retrieves the same prose. We propose Skill-as-Pseudocode (SaP), an automatic conversion of markdown skill libraries into typed pseudocode with deterministic quality control. For each cluster of similar procedural passages drawn from one or more skills, SaP extracts a typed contract and filters it through a four-check deterministic verifier (coverage, binding, replacement, risk). Promoted contracts are inlined into a rewritten skill skeleton together with restored concrete action templates, giving the agent two complementary signals: a typed signature for what the skill does and a concrete template for how to invoke it. On the 134-game ALFWorld unseen split with gpt-4o-mini, pooled across three seeds, SaP wins 82/402 paired games versus 47/402 for the Graph-of-Skills (GoS) baseline (pooled McNemar p = 8.2e-5), at -22.8 +/- 6.4% input tokens and -14.5 +/- 4.1% LLM calls per game.
>
---
#### [new 176] MoDAl: Self-Supervised Neural Modality Discovery via Decorrelation for Speech Neuroprosthesis
- **分类: q-bio.NC; cs.CL; cs.HC; cs.LG; eess.AS**

- **简介: 该论文提出MoDAl框架，用于语音神经假体中的多模态神经信号解码，解决传统方法忽略互补信息的问题，通过对比损失和去相关损失提升解码准确率。**

- **链接: [https://arxiv.org/pdf/2605.00025](https://arxiv.org/pdf/2605.00025)**

> **作者:** Yuanhao Chen; Peter Chin
>
> **摘要:** Speech neuroprosthesis systems decode intended speech from neural activity in the absence of audible output, offering a path to restoring communication for individuals with speech-impairing conditions. Current approaches decode predominantly from motor cortical areas, discarding others -- such as area 44, part of Broca's area -- that may encode complementary linguistic information. We introduce MoDAl (Modality Decorrelation and Alignment), a framework that discovers complementary neural modalities through the interplay of two objectives in a shared projection space. A contrastive loss aligns each of several parallel brain encoders with the text embeddings of a pretrained large language model (LLM), while a decorrelation loss prevents the encoders from coalescing to duplicative representations. We prove that these objectives are in productive tension: Contrastive alignment induces transitive modality coalescence, which decorrelation must counteract for the framework to discover diverse neurolinguistic modalities. On the Brain-to-Text Benchmark '24, MoDAl reduces word error rate (WER) from 26.3% to 21.6% compared to the previous best end-to-end method, with the gain from incorporating previously discarded area 44 signals arising entirely from the decorrelation mechanism. Analysis of the discovered modalities reveals functional specialization: Encoders receiving area 44 input capture structural and syntactic properties (sentence length, grammatical voice, wh-words), consistent with the neurolinguistic understanding of Broca's area.
>
---
#### [new 177] Soro: A Lightweight Foundation Model and Chatbot for Tajik
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Soro，一个针对塔吉克语的轻量级大语言模型及聊天机器人，解决塔吉克斯坦计算和连接资源有限的问题。通过预训练和指令调优提升性能，并构建专用基准进行评估。**

- **链接: [https://arxiv.org/pdf/2605.27379](https://arxiv.org/pdf/2605.27379)**

> **作者:** Stanislav Liashkov; Haitz Sáez de Ocáriz Borde; Azizjon Azimi; Khushbakht Shaymardonov; Shuhratjon Khalitbekov; Bonu Boboeva
>
> **摘要:** We present Soro, a family of Tajik-specialized conversational large language models (LLMs) designed for real-world deployment under tight compute and connectivity constraints in Tajikistan. Starting from open-weight Gemma 3 checkpoints, we perform Tajik-only continual pretraining on a curated 1.9-billion-token corpus spanning filtered web text, PDF documents, and curriculum-aligned educational materials, followed by supervised instruction tuning on 40K Tajik teacher-style examples. To enable rigorous evaluation despite the limited coverage of Tajik in standard benchmarks, we introduce a suite of Tajik benchmarks covering general knowledge, linguistic competence, and school- and university entrance-exam domains, and we open-source them on Hugging Face. Across these Tajik benchmarks, Soro substantially outperforms same-size Gemma 3 baselines while retaining strong English performance on standard datasets. We further show that FP8 and INT4 quantization of Soro preserves most Tajik-language gains while reducing memory requirements for edge deployment, supporting an ongoing education-sector pilot and planned scale-out across schools in Tajikistan.
>
---
#### [new 178] Activation Steering for Synthetic Data Generation: The Role of Diversity in Downstream Safety Detection
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于安全检测任务，旨在解决HVV-violating数据稀缺问题。通过激活引导生成合成数据，研究其在下游分类器中的效果，强调多样性的重要性。**

- **链接: [https://arxiv.org/pdf/2605.28664](https://arxiv.org/pdf/2605.28664)**

> **作者:** Vijeta Deshpande; Tootiya Giyahchi; Veena Padmanabhan; Leman Akoglu; Anna Rumshisky
>
> **摘要:** Safety detection models require examples of HHH (Helpful, Harmless, Honest)-violating outputs for robust generalization, however such examples are scarce. Activation Steering (AS) has emerged as a data-efficient method for generating target-concept-aligned responses. We investigate whether AS can generate high-quality training datasets for downstream classifiers, a question that remains untested. We present a two-fold study with intrinsic and extrinsic evaluation across $4$ concepts $\times\,2$ models $\times\,4$ steering methods. Intrinsically, beyond the field-standard rubric of steering success (concept alignment) and coherence, we introduce sample- and set-level diversity as a quality axis previously absent from the literature, and find that increasing steering strength reduces response diversity. Extrinsically, we replace HHH-violating examples in the available training data with steered generations and fine-tune detection classifiers. AS-generated data results in a better classifier than the prompting-generated data on $3$ of $4$ concepts. However, only $41$ of $136$ AS configurations outperform prompting, indicating that downstream utility lies in a narrow regime that jointly satisfies success, coherence, and diversity. The harmonic mean of these three axes correlates with downstream AUROC more consistently across concepts than success and coherence alone, providing a practical heuristic target for practitioners tuning AS hyperparameters. Together, our results highlight the potential of AS in synthetic data generation for improving safety detection and identify diversity as a critical, previously overlooked axis for tuning AS.
>
---
#### [new 179] PEFT-Arena: Understanding Parameter-Efficient Finetuning from a Stability-Plasticity Perspective
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理中的模型微调任务，旨在解决PEFT方法在适应新任务时遗忘预训练能力的问题。工作包括提出PEFT-Arena基准，分析稳定性与可塑性平衡，并提出改进方法。**

- **链接: [https://arxiv.org/pdf/2605.28819](https://arxiv.org/pdf/2605.28819)**

> **作者:** Yangyi Huang; Ruotian Peng; Zeju Qiu; Jiale Kang; Yandong Wen; Bernhard Schölkopf; Weiyang Liu
>
> **备注:** Technical report v1 (28 pages, 9 figures, project page: this https URL)
>
> **摘要:** Parameter-efficient finetuning (PEFT) has become the standard approach for adapting large language models, yet evaluations largely emphasize downstream accuracy while overlooking the retention of pretrained capabilities. We argue that PEFT should be assessed through the stability-plasticity dilemma: the trade-off between target-task adaptation and resistance to forgetting. We introduce PEFT-Arena, a benchmark that jointly measures downstream performance and general capability retention. Across methods, we find distinct stability-plasticity profiles; under comparable parameter budgets, orthogonal finetuning achieves the most favorable Pareto frontier. To explain these differences, we analyze PEFT updates from two geometric perspectives. In weight space, spectral analysis reveals how parameterizations interact with the pretrained singular-value structure. In activation space, retention metrics show whether finetuning preserves or distorts general-capability representations, with forgetting linked to non-isometric representation distortion. Finally, an analysis shows that final SFT checkpoints often overshoot a better target-retention operating point. Inspired by this, we present case studies of a post-hoc improvement with path-wise rewinding.
>
---
#### [new 180] Memory-Based vs. Context-Only Conditioning Produces Distinct Behavioral Patterns in Stateful Personalization
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文研究教育推荐系统中个性化行为的差异，比较基于上下文和记忆的条件化方法，分析其对推荐效果的影响。任务是理解不同条件化方式如何影响个性化行为。**

- **链接: [https://arxiv.org/pdf/2605.27389](https://arxiv.org/pdf/2605.27389)**

> **作者:** Junsoo Park; Youssef Medhat; Htet Phyo Wai; Ploy Thajchayapong; Ashok K. Goel
>
> **备注:** Accepted to ITS 2026
>
> **摘要:** We study how conditioning context shapes personalization behavior in a teacher-facing educational recommender system. We compare contextual conditioning based on the current student question with memory-based conditioning using persistent learner information. Using deviation correlation and paired statistical tests, we find that contextual recommendations exhibit stronger question-level responsiveness, while memory-based recommendations exhibit history-dependent behaviors, including learner-specific differentiation under identical input. Teacher-facing evaluation signals suggest these recommendations are interpretable and actionable. These results indicate that embedding-based similarity metrics capture responsiveness to the current question but do not characterize personalization grounded in learner history, motivating behavior-level diagnostics for studying conditioning effects.
>
---
#### [new 181] Self-Consistency via Marginal Sharpening
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型推理任务，旨在提升模型推理能力。针对现有方法聚焦于输出分布的问题，提出通过锐化答案边缘分布实现自洽性，提升数学与编码基准性能。**

- **链接: [https://arxiv.org/pdf/2605.28142](https://arxiv.org/pdf/2605.28142)**

> **作者:** Aleksei Arzhantsev; Otmane Sakhi; Nicolas Chopin
>
> **摘要:** Inference-time sampling can elicit strong reasoning abilities from language models without additional training. Existing power-sampling methods do so by sharpening the distribution over full generated outputs, favoring completions that are individually likely under the model. We argue that this is the wrong object to target for reasoning: a completion entangles a reasoning trace with a final answer, whereas what matters is whether an answer is supported by many plausible reasoning paths. We therefore shift the target from the full-output distribution to the sharpened answer marginal, making self-consistency an inference-time objective rather than a post-hoc voting criterion. Surprisingly, this marginal target admits an efficient approximation: we propose a simple, purely autoregressive parallel sampling algorithm that approximately samples from the sharpened answer marginal, eliciting stronger performance than standard power sampling on mathematics and coding benchmarks while being orders of magnitude faster.
>
---
#### [new 182] REC-CBM: Rubric-Aware Error-Correction Concept Bottleneck Models for Trustworthy Open-Ended Grading
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于自动评分任务，旨在解决开放性答题评分的透明度和可靠性问题。提出REC-CBM模型，通过可解释的概念瓶颈机制提升评分可信度。**

- **链接: [https://arxiv.org/pdf/2605.27402](https://arxiv.org/pdf/2605.27402)**

> **作者:** Chengshuai Zhao; Fan Zhang; Kumar Satvik Chaudhary; Yiwen Li; Lo Pang-Yun Ting; Ying-Chih Chen; Huan Liu
>
> **摘要:** Open-ended grading is central to equitable and personalized education, yet manual grading remains time-consuming and costly, underscoring the need for automated grading systems. Although recent neural and large language model (LLM) based systems have demonstrated superior performance, they are typically black-box models whose scoring processes and rationales are difficult for educators to verify and trust. Concept bottleneck models (CBMs) have emerged as a promising approach by routing predictions through human-interpretable concepts, providing a mechanistic guarantee of transparency. However, standard CBMs are not tailored to open-ended grading: they do not explicitly model fine-grained rubric dimensions, inadequately capture the ordinal semantics of scoring scales, and neglect inherent reliability issues in human concept annotations. To address these limitations, we propose REC-CBM, a rubric-aware error-correction concept bottleneck model for trustworthy open-ended grading. REC-CBM introduces a rubric-aware concept encoder that learns concept-specific representations over responses and an ordinal pairwise calibration objective that preserves ranking structure among rubric dimensions. It further incorporates a latent concept error-correction module that denoises concept predictions before final grade prediction while preserving interpretability. Comprehensive experiments on publicly available datasets show that REC-CBM consistently improves grading performance and produces more faithful concept-level reasoning than both state-of-the-art baselines. Further analyses validate the contribution of each component and demonstrate the applicability in realistic educational settings. Overall, this work provides a practical, interpretable grading solution that enables educators to inspect, intervene in, and trust automated decisions, advancing more transparent and trustworthy education.
>
---
#### [new 183] Code as a Weapon: A Consensus-Labeled Prompt Bank for Measuring Coding-Model Compliance with Malicious-Code Requests
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文属于代码安全任务，旨在解决编码模型对恶意请求的合规性评估问题。通过构建一致标注的提示库，区分可执行恶意代码与有害知识请求，提供可靠测试基准。**

- **链接: [https://arxiv.org/pdf/2605.28734](https://arxiv.org/pdf/2605.28734)**

> **作者:** Richard J. Young; Gregory D. Moody
>
> **备注:** 21 pages, 9 figures, 5 tables. Consensus-labeled prompt bank consolidating eight malicious-code corpora (ASTRA, CySecBench, AdvBench/harmful_behaviors, JailbreakBench, MalwareBench, RedCode, RMCBench, Scam2Prompt) under a five-judge panel; 6,675 prompts, 33,375 classification calls, Fleiss' kappa = 0.767
>
> **摘要:** A general-purpose language model that answers a harmful question returns text; a coding model that complies with a malicious request can return a working weapon -- a keylogger, a ransomware stub, an exploit that runs as written. This asymmetry in the severity of a single act of compliance implies coding-specialized models should clear a higher refusal bar than general-purpose chat models, not a lower one, yet the field cannot presently tell whether they do. Refusal benchmarks for malicious code are fragmented: they mix requests for executable software (ready-to-run weapons) with requests for harmful security knowledge (information a human must still operationalise) and report refusal rates over non-comparable corpora, so no single statistic measures the property that actually matters. This paper introduces an expanded consensus-labeled prompt bank that distinguishes between these two request types and provides a construct-stable substrate for cross-corpus coding-model compliance measurement. Eight corpora (ASTRA, CySecBench, AdvBench/harmful_behaviors, JailbreakBench, MalwareBench, RedCode, RMCBench, Scam2Prompt) are consolidated and classified under a five-judge consensus protocol (6,675 prompts x 5 judges = 33,375 calls). The panel reaches Fleiss' kappa = 0.767 [95% CI 0.755, 0.777] ("substantial"); 95.0% of prompts draw at least four agreeing judges, 76.9% are unanimous, and the panel reproduces the earlier four-corpus release at Cohen's kappa = 0.952 on the 3,133 shared prompts. The released bank comprises 4,748 consensus-CODE prompts (executable malicious code requests) and 1,923 consensus-KNOWLEDGE prompts (harmful security knowledge requests). The bank is the validated instrument the field has lacked: a reliability-quantified basis for testing whether coding models meet the stricter refusal standard their executable output demands.
>
---
#### [new 184] FPMoE: A Sparse Mixture-of-Experts Approach to Functional Code Generation
- **分类: cs.PL; cs.AI; cs.CL**

- **简介: 该论文提出FPMoE，解决LLM在函数式编程语言生成中的性能不足问题。通过稀疏专家混合架构，提升代码生成效果。**

- **链接: [https://arxiv.org/pdf/2605.27849](https://arxiv.org/pdf/2605.27849)**

> **作者:** Loc Pham; Lang Hong Nguyet Anh; Thanh Le-Cong
>
> **摘要:** Despite rapid progress in LLM-based code generation, existing models are predominantly trained on imperative languages, leaving functional programming languages (FPLs) such as Haskell, OCaml, and Scala chronically underexplored, with even frontier models performing substantially worse on FPLs. Fine-tuning is a natural remedy, but our experiments show that per-language fine-tuning fails to capture shared functional abstractions, while merged multi-language fine-tuning introduces cross-language interference. To address this, we introduce FPMoE, a lightweight, open-source code generation model built on a sparse Mixture-of-Experts (MoE) architecture with three language-specific routed experts (one each for Haskell, OCaml, and Scala) and a shared expert that captures cross-language functional patterns such as monadic reasoning and type-directed programming. This design resolves both failure modes simultaneously: dedicated experts eliminate interference, while the shared expert preserves abstractions that per-language models miss. On FPEval, FPMoE substantially outperforms fine-tuned baselines and, with only 3B active parameters, matches the performance of much larger models including DeepSeek-Coder-6.7B, Qwen2.5-Coder-14B-Instruct, and Qwen3-Coder-30B-A3B.
>
---
#### [new 185] The Importance of Being Statistically Earnest: A Critical Re-evaluation of GSM-Symbolic
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的模型评估任务，质疑GSM-Symbolic基准的统计结论。通过重新分析数据，发现模型表现变化不显著，且存在数据分布差异影响结果，指出需谨慎下结论。**

- **链接: [https://arxiv.org/pdf/2605.28700](https://arxiv.org/pdf/2605.28700)**

> **作者:** Dominika Agnieszka Długosz; Arlindo Oliveira; Natalia Díaz Rodríguez
>
> **备注:** 38 pages, 11 figures. Submitted to ACL ARR / EMNLP 2026
>
> **摘要:** The GSM-Symbolic benchmark (Mirzadeh et al., 2025) reported consistent performance drops across 25 Large Language Models (LLMs) when tested on template-generated variants of GSM8K problems, concluding that the models lack genuine reasoning capabilities. We argue that this conclusion rests on shaky statistical ground. Re-evaluating 20 open-weight models using Generalised Linear Mixed Models with per-question random effects, we find that only half exhibit statistically significant performance changes under the original prompt format. Moreover, we identify a previously unacknowledged factor: the main GSM-Symbolic dataset contains a systematically shifted distribution of larger integers in problem texts relative to GSM-Base (K-S statistic = 0.12, p < 0.001), contradicting the original authors' claims. Controlling for this large number effect accounts for significance in roughly half the remaining cases. Among models with statistically significant performance deltas, we identify distinct, model-specific failure profiles - including fragility of variable binding, arithmetic limitations, and dual-task interference - underscoring that blanket claims about LLM reasoning are both statistically premature and mechanistically misleading.
>
---
#### [new 186] Why LLMs Fail at Causal Discovery and How Interventional Agents Escape
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究因果发现任务，解决大语言模型在复杂因果图中失效的问题。提出A-CBO方法，通过外部贝叶斯循环提升性能。**

- **链接: [https://arxiv.org/pdf/2605.27567](https://arxiv.org/pdf/2605.27567)**

> **作者:** Amartya Roy; Sonali Parbhoo
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Causal discovery is a cornerstone of scientific reasoning, yet whether large language models can perform it reliably remains an open question. Recent benchmarks show that even fine-tuned models plateau on simple causal graphs and degrade as complexity grows, but why they fail has not been established. We prove the failure is fundamental: supervised fine-tuning, direct preference optimization, and in-context learning all produce predictors that cannot distinguish between causal graphs generating similar observational data, and any attempt to do so requires the model's internal representations to grow unboundedly, violating the very conditions under which these methods work. We formalize this as a kernel obstruction theorem, establishing that the limitation is intrinsic to the learning paradigm, \emph{not any particular model or dataset}. We propose Agentic Causal Bayesian Optimization (A-CBO), wherein a frozen language model serves as an interventional oracle answering targeted queries about intervention effects, while an external Bayesian loop concentrates beliefs over candidate graphs in logarithmically many rounds. Because the decision operates outside the space where the obstruction applies, A-CBO provably converges while the underlying model remains unchanged. On Corr2Cause, A-CBO matches fine-tuned baselines without any training. On Extended Corr2Cause, a new benchmark scaling to 24 variables with 18K test samples, A-CBO significantly outperforms both fine-tuning and preference optimization, with the advantage growing
>
---
#### [new 187] Risk-Controlled Lean-as-Judge for Natural-Language Mathematical Reasoning
- **分类: cs.AI; cs.CL; cs.LO**

- **简介: 该论文属于自然语言数学推理任务，解决Lean验证信号不准确的问题。通过COVCAL方法，在不同形式化工具下控制风险，提升验证可靠性。**

- **链接: [https://arxiv.org/pdf/2605.28365](https://arxiv.org/pdf/2605.28365)**

> **作者:** Pauline Bourigault; Xiaotong Ji; Matthieu Zimmer; Rasul Tutunov; Haitham Bou Ammar
>
> **摘要:** Lean is increasingly used to judge natural-language mathematical answers, but its signal is partial: many answers never formalize, and a failed proof may reflect an ill-typed statement or a missing library fact, not a wrong answer. On MATH-500 we show this signal is (i) sharply coverage-dependent, that is the proof-winning answer is correct 96% of the time at high proved coverage but 20% at low, and (ii) sparse and often unfaithful: a 7B autoformalizer proves a class for only 28% of problems, and a manual audit finds only approximately 43% of those proofs faithful. We propose COVCAL, a selector over Lean-trace diagnostics that certifies a finite-sample selective-risk bound on accepted answers or abstains, under two regimes (a conservative Bonferroni bound and a tighter dev-then-cal rule). Feasibility depends on autoformalization coverage: with the 7B formalizer the signal is too sparse and Bonferroni abstains on all 20 bootstrap partitions, whereas a prover-specialized formalizer reaches 79% coverage and flips it to feasible on 17 of 20, accepting approximately 48% of problems at 0.98 accepted accuracy. Since self-consistency alone is already 91% accurate, our contribution is a precise account of when, and with which formalizer, a partial formal signal can be trusted under risk control.
>
---
#### [new 188] SNARE: Adaptive Scenario Synthesis for Eliciting Overeager Behavior in Coding Agents
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文提出SNARE，用于检测编码代理的过度行为。任务是评估代理在合法任务中可能产生的越权操作。工作包括构建场景、评分机制和优化策略。**

- **链接: [https://arxiv.org/pdf/2605.28122](https://arxiv.org/pdf/2605.28122)**

> **作者:** Yubin Qu; Yi Liu; Gelei Deng; Yanjun Zhang; Yuekang Li; Ying Zhang; Leo Yu Zhang
>
> **摘要:** A coding agent executes a benign task as a sequence of shell, file, and network actions, any of which can quietly exceed the authorized scope while the task still completes. We call this overeager behavior: the prompt is not adversarial and the run succeeds, yet an out-of-scope step can leak credentials or delete files. Existing benchmarks miss it: task-completion suites credit any finished run, jailbreak suites probe adversarial prompts, and the one prior overeager benchmark applies a single fixed prompt set to every agent-model pair, leaving its easiest and most resistant pairs under-measured. We present SNARE (Synthesizing Non-adversarial scenarios for Adaptive Reward-guided Elicitation), a pipeline that composes benign scenarios from reusable scope and trap fragments, scores each run with a judge-free oracle flagging trap-pattern matches and unsolicited file additions or deletions, and uses Thompson sampling to steer each pair's run budget toward the scenarios that most often trigger it. Instantiating it over 24 overeager archetypes yields OverEager, which we run across a 4x5 matrix of four coding agents and five base models. Across 10,000 benign runs, 19.51% trigger overeager behavior, with per-pair rates spanning 11.9x. This variation is driven by the agent framework, not the model: the framework accounts for 56% of it against the model's 21%, so any single-framework or single-model evaluation undercounts the matrix by about a fifth.
>
---
#### [new 189] Extrapolative Weight Averaging Reveals Correctness-Efficiency Frontiers in Code RL
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究代码强化学习中的正确性与效率权衡问题，通过外推权重平均扩展优化前沿，提升推理性能。任务为代码生成，解决如何在不额外训练的情况下提升模型表现。**

- **链接: [https://arxiv.org/pdf/2605.28751](https://arxiv.org/pdf/2605.28751)**

> **作者:** Kunhao Zheng; Pierre Chambon; Juliette Decugis; Jonas Gehring; Taco Cohen; Benjamin Negrevergne; Gabriel Synnaeve
>
> **备注:** 54 pages
>
> **摘要:** Linear interpolation between fine-tuned checkpoints has been shown to trace the Pareto front between competing objectives, but whether extrapolative weight averaging can extend such frontiers to new checkpoints useful at inference time, without additional RL training, remains unclear. We study this question in RL for competitive programming, where hidden unit tests under time and memory limits enforce both functional correctness and computational efficiency. Starting from a shared initialization, we train checkpoints under nested unit-test coverage: low-coverage rewards require passing smaller-input tests, while high-coverage rewards require passing progressively larger tests up to the full suite. This sweep reveals the emergence of a correctness-efficiency frontier: on hard problems, higher-coverage reward reduces optimization failures but increases correctness failures, leaving solve rate nearly unchanged. Interpolation between low- and high-coverage checkpoints recovers this frontier, while extrapolation extends it beyond the trained endpoints. Both the frontier and its extrapolative continuation appear across three inference settings, pure reasoning, tool use, and agentic coding, and across two model scales, 32B and 7B. At the problem level, moving along the frontier changes which problems are solved, making extrapolated checkpoints complementary policies in inference-time scaling. Ensembles with extrapolative weight averaging broaden coverage and improve pass@250 on LCB/hard by 3.3% over the best single checkpoint at matched sample budget. These results show that nested unit-test coverage in code RL induces a frontier that extrapolative weight averaging can navigate, extend, and exploit.
>
---
#### [new 190] OphIn-500K: Curating Web-Scale Visual Instructions for Scaling Ophthalmic Multimodal Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医学多模态语言模型任务，旨在解决眼科领域数据稀缺问题。通过构建大规模眼科指令数据集OphIn-500K，提升模型的临床理解和对话能力。**

- **链接: [https://arxiv.org/pdf/2605.27916](https://arxiv.org/pdf/2605.27916)**

> **作者:** Xuanzhao Dong; Wenhui Zhu; Xiwen Chen; Hao Wang; Xin Li; Yujian Xiong; Jiajun Cheng; Jingjing Wang; Xiaobing Yu; Haiyu Wu; Shao Tang; Zhipeng Wang; Langechuan Liu; Shan Lin; Oana Dumitrascu; Yalin Wang
>
> **摘要:** The advancement of general medical Multimodal Large Language Models (MLLMs) has shown great potential for building conversational assistants to support clinical diagnosis. However, their adaptation to highly specialized domains such as ophthalmology remains underexplored, primarily due to the scarcity of large-scale, domain-specific instruction-tuning data. Existing ophthalmic datasets for conversational agents are often limited in scale and largely rely on images from established public benchmarks, limiting the scalability of ophthalmic MLLMs and their ability to capture real-world clinical complexity. To address this gap, we propose $\textbf{OphIn-Engine}$, an ophthalmology-specific instruction data curation pipeline that constructs high-quality instruction data from open-access ophthalmology web-scale videos. The pipeline integrates multimodal transcription for extracting image-transcript pairs, visual cue separation and scoring for identifying clinically relevant visual descriptions, and instruction synthesis with quality control for generating accurate and diverse clinical dialogues. Using this engine, we introduce $\textbf{OphIn-500K}$, a large-scale multimodal ophthalmology instruction-tuning dataset containing over 500,000 instruction instances and more than 151,000 unique images from over 29,000 video clips, formatted as visual question answering (VQA), multi-turn conversational interactions, and chain-of-thought (CoT) reasoning. Built upon this dataset, we further develop $\textbf{OphIn-VL}$, an ophthalmology-specific MLLM with advanced visual understanding and conversational capabilities. Comprehensive experiments and case studies demonstrate that OphIn-VL achieves superior performance compared with state-of-the-art general medical and domain-specific MLLMs.
>
---
#### [new 191] Show, Don't TELL: Explainable AI-Generated Text Detection
- **分类: cs.AI; cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于AI生成文本检测任务，旨在解决现有方法缺乏解释性的问题。提出TELL模型，不仅提供评分，还通过解释帮助用户判断文本来源。**

- **链接: [https://arxiv.org/pdf/2605.27921](https://arxiv.org/pdf/2605.27921)**

> **作者:** Aldan Creo; Suraj Ranganath
>
> **摘要:** Research on AI-generated text detection has presented a number of approaches to discern human from AI prose, some of which achieving high in-distribution performance. However, real-world applicability has stalled because their outputs are misaligned with the needs of users, such as professors, who are presented with a numeric score that has no attached explanation. We tackle this issue with a novel architecture, TELL, that bakes explainability from the ground-up. While our system still offers a numerical score like other detectors for comparability, TELL takes a fundamentally different approach where we aim to show the user the "tells" by which the model believes a text is AI or human-written, to empower the user to decide who wrote a text using their own judgment and understanding of the context of the writing and its alleged author. We train TELL on a custom SFT dataset of domain-specific authorship annotations, and further refine the system using GRPO with curriculum learning to improve performance. We achieve competitive performance with state-of-the-art detectors (AUROC 0.927) while natively providing annotations that explain the basis for the detector's decision. We further evaluate the quality of our explanations using a dataset of human annotations and report a high (mean 72.3%) win-rate on annotation concreteness, falsifiability, coherence, plausibility and grounding, allowing users to critically think and decide for themselves. Our work thus reframes the problem of AI-generated text detection in a human-centric perspective and paves the way for a new family of detectors that focus on native explainability.
>
---
#### [new 192] SilentRetrieval: Hijacking Retrieval-Augmented Generation via Semantically-Preserving Adversarial Data Poisoning
- **分类: cs.CR; cs.CL; cs.IR**

- **简介: 该论文属于自然语言处理任务，针对RAG系统提出一种数据污染攻击方法，通过构造流畅但含恶意触发器的文档，实现对生成结果的操控。**

- **链接: [https://arxiv.org/pdf/2605.28074](https://arxiv.org/pdf/2605.28074)**

> **作者:** Jiachen Qian
>
> **备注:** 12 pages, 4 figures, KDD '26 camera-ready version
>
> **摘要:** Retrieval-Augmented Generation (RAG) mitigates LLM hallucinations but introduces a critical vulnerability: corpus integrity. We present SilentRetrieval, a two-stage data poisoning attack that hijacks RAG systems through adversarially crafted yet fluent documents. Stage 1 uses Coordinated Beam Search, a multi-token joint optimization method with a fluency-similarity objective, to keep a poisoned host document retrievable while constraining perplexity. Stage 2 uses Context-Adaptive Trigger Generation, a lightweight trigger-fusion step driven by a frozen LLM, to integrate manipulation triggers into document content. Under a one-poisoned-document-per-query evaluation with synthetic target answers, SilentRetrieval achieves 84.6%/81.3% HR@10 and 57.5%/54.8% ASR-LLM on Natural Questions and MS MARCO, while maintaining near-benign perplexity. Cross-model evaluation across four target LLMs shows nontrivial effectiveness under a fixed trigger generator, and transfer tests against unseen retrievers, including ColBERT and commercial embedding models, yield 64.7% average HR@10 under the same injected-corpus protocol. In a sampled Wikipedia-scale evaluation, SilentRetrieval retains 74.2% HR@10 at a 0.016% poisoning ratio. Combined retrieval-side and generation-side defenses reduce attack success substantially but incur a latency trade-off. Human evaluation shows substantially lower flag rates than disfluent baselines, while remaining numerically more suspicious than benign content at the current sample size.
>
---
## 更新

#### [replaced 001] InfiMed-ORBIT: Aligning LLMs on Open-Ended Complex Tasks via Rubric-Based Incremental Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗对话任务，解决开放性医学对话中奖励信号模糊的问题。提出ORBIT框架，通过动态评分标准进行增量训练，提升模型表现。**

- **链接: [https://arxiv.org/pdf/2510.15859](https://arxiv.org/pdf/2510.15859)**

> **作者:** Pengkai Wang; Pengwei Liu; Qi Zuo; Zhijie Sang; Congkai Xie; Hongxia Yang
>
> **摘要:** Reinforcement learning (RL) has driven recent breakthroughs in large language models (LLMs), especially for tasks where rewards can be computed automatically, such as code generation. However, it is less effective in open-ended medical dialogue, where feedback is ambiguous, context-dependent, and difficult to summarize into a single scalar signal-often requiring heavily supervised reward models and risking reward hacking. Thus, we introduce ORBIT, an open-ended rubric-based incremental training framework tailored for critical medical dialogues. ORBIT integrates medical dialogue construction with dynamically generated case-conditioned rubrics that serve as adaptive guides for incremental RL. Unlike approaches that rely on external medical knowledge bases or handcrafted rules, ORBIT uses rubric-guided evaluation and can be implemented with general-purpose instruction-following LLMs, avoiding task-specific judge fine-tuning. With only 2k training samples, ORBIT raises Qwen3-4B-Instruct's HealthBench-Hard score from 7.0 to 27.5, achieving state-of-the-art performance among similarly sized open-source models while maintaining strong consultation quality as rubric coverage broadens.
>
---
#### [replaced 002] Self-Improving CAD Generation Agents with Finite Element Analysis as Feedback
- **分类: cs.GR; cs.CL**

- **简介: 该论文属于CAD生成任务，旨在解决AI生成的CAD模型无法满足工程要求的问题。通过引入FEA反馈和视觉监督信号，提升生成模型的准确性和工程适用性。**

- **链接: [https://arxiv.org/pdf/2605.17448](https://arxiv.org/pdf/2605.17448)**

> **作者:** Guijin Son; Jehyun Park; Seyeon Park; Sunghee Ahn; Youngjae Yu
>
> **备注:** Work in progress
>
> **摘要:** Computer-aided design (CAD) is the backbone of modern industrial design, yet learned CAD generators still fall short of real engineering pipelines: they neither iterate like engineers nor evaluate what engineering requires. Prior work has treated CAD generation as two disjoint steps, part synthesis and assembly, where the former is graded by proximity to a gold reference and the latter, when handled at all, is reduced to a separate constraint solving step. In this work, we introduce a more industry-native task formulation that requires a model to produce a fully assembled multi-part STEP file from a free-form engineering brief, which is then validated via finite element analysis (FEA). FEA validation reveals that Codex (GPT-5.5) and Claude Code (Opus-4.7) agents do not produce a single strict-passing artifact in the main first-attempt sweep, with the best configuration meeting only about 20% of typed requirements on average. Moreover, we introduce two additional supervision signals, a novel text-only blueprint schema and a 21-view image renderer that aids the agent's visual inspection, that better align the generation loop with how engineers iterate in practice. On S2O and Fusion360, the same feedback tools improve geometric reconstruction, with GPT-5.5/xhigh rising from 0.444 to 0.592 Box-IoU on S2O and from 0.397 to 0.505 on Fusion360. Together these signals move CAD programs toward artifacts that are not only visually plausible but also checked against physical and structural requirements.
>
---
#### [replaced 003] Measuring Massive Multitask Chinese Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多任务理解评估任务，旨在解决大中文语言模型能力评估不足的问题。通过设计涵盖多个领域的测试，评估模型的多任务准确性。**

- **链接: [https://arxiv.org/pdf/2304.12986](https://arxiv.org/pdf/2304.12986)**

> **作者:** Hui Zeng
>
> **摘要:** The development of large-scale Chinese language models is flourishing, yet there is a lack of corresponding capability assessments. Therefore, we propose a test to measure the multitask accuracy of large Chinese language models. This test encompasses four major domains, including medicine, law, psychology, and education, with 15 subtasks in medicine and 8 subtasks in education. We found that the best-performing models in the zero-shot setting outperformed the worst-performing models by nearly 18.6 percentage points on average. Across the four major domains, the highest average zero-shot accuracy of all models is 0.512. In the subdomains, only the GPT-3.5-turbo model achieved a zero-shot accuracy of 0.693 in clinical medicine, which was the highest accuracy among all models across all subtasks. All models performed poorly in the legal domain, with the highest zero-shot accuracy reaching only 0.239. By comprehensively evaluating the breadth and depth of knowledge across multiple disciplines, this test can more accurately identify the shortcomings of the models.
>
---
#### [replaced 004] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型的去学习任务，旨在解决模型中敏感或过时知识的移除问题。工作包括评估现有方法的不足，并提出改进方案以恢复上下文可用性。**

- **链接: [https://arxiv.org/pdf/2510.17620](https://arxiv.org/pdf/2510.17620)**

> **作者:** Yuefeng Peng; Parnian Afshar; Megan Ganji; Thomas Butler; Amir Houmansadr; Mingxian Wang; Dezhi Hong
>
> **备注:** ICML 2026
>
> **摘要:** Large language models may encode sensitive information or outdated knowledge that needs to be removed, to ensure responsible and compliant model responses. Unlearning has emerged as an efficient alternative to full retraining, aiming to remove specific knowledge while preserving overall model utility. Existing evaluations of unlearning methods focus on (1) the extent of forgetting of the target knowledge (forget set) and (2) maintaining performance on the retain set (i.e., utility). However, these evaluations overlook an important usability aspect: users may still want the model to leverage the removed information if it is re-introduced in the prompt. In a systematic evaluation of six state-of-the-art unlearning methods, we find that they consistently impair such contextual utility. To address this, we augment unlearning objectives with a plug-in term that preserves the model's ability to use forgotten knowledge when it is present in context. Extensive experiments demonstrate that our approach restores contextual utility to near original levels while still maintaining effective forgetting and retain-set utility.
>
---
#### [replaced 005] Learning Deliberately, Acting Intuitively: Unlocking Test-Time Reasoning in Multimodal LLMs
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态大模型推理任务，旨在解决模态对齐和训练可扩展性问题。提出D2I框架，通过格式化奖励提升推理能力，无需额外标注或复杂奖励。**

- **链接: [https://arxiv.org/pdf/2507.06999](https://arxiv.org/pdf/2507.06999)**

> **作者:** Yahan Yu; Yuyang Dong; Masafumi Oyamada
>
> **备注:** 22 pages, 24 figures
>
> **摘要:** Reasoning is essential for large language models (LLMs), especially in complex tasks such as mathematical problem solving. However, multimodal reasoning still faces challenges in modality alignment and training scalability, as many existing methods rely on additional annotations or complex rule-based rewards. To address these issues, we propose the Deliberate-to-Intuitive reasoning framework (D2I), which improves the understanding and reasoning abilities of multimodal LLMs (MLLMs) without extra annotations or complex rewards. During training, D2I uses deliberate reasoning strategies supervised only by rule-based format rewards to enhance modality alignment. During inference, it shifts to intuitive reasoning by removing these explicit strategies, allowing the model to implicitly apply the acquired abilities in its responses. D2I outperforms baselines on both in-domain and out-of-domain benchmarks, highlighting the effectiveness of format rewards in fostering transferable multimodal reasoning skills and suggesting the benefit of decoupling training-time reasoning depth from test-time response flexibility.
>
---
#### [replaced 006] Do readers prefer AI-generated Italian short stories?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本偏好研究任务，旨在探讨读者是否更偏好AI生成的意大利短篇小说。通过实验比较AI与人类作者的作品，分析读者偏好及影响因素。**

- **链接: [https://arxiv.org/pdf/2601.17363](https://arxiv.org/pdf/2601.17363)**

> **作者:** Michael Farrell
>
> **备注:** 7 pages, peer-reviewed and accepted for presentation at New Trends in Translation and Interpreting Technology (NeTTIT 2026)
>
> **摘要:** This study investigates whether readers prefer AI-generated short stories in Italian over one written by a renowned Italian author. In a blind setup, 20 participants read and evaluated three stories, two created with ChatGPT-4o and one by Alberto Moravia, without being informed of their origin. To explore potential influencing factors, reading habits and demographic data, comprising age, gender, education and first language, were also collected. The results showed that the AI-written texts received slightly higher average ratings and were more frequently preferred, although differences were modest. No statistically significant associations were found between text preference and demographic or reading-habit variables. These findings challenge assumptions about reader preference for human-authored fiction and raise questions about the necessity of synthetic-text editing in literary contexts.
>
---
#### [replaced 007] Analyzing Cancer Patients' Experiences with Embedding-based Topic Modeling and LLMs
- **分类: cs.CL**

- **简介: 该论文属于文本主题分析任务，旨在通过神经主题建模和LLMs从患者访谈中提取有意义的主题，以改善患者导向的医疗实践。**

- **链接: [https://arxiv.org/pdf/2601.12154](https://arxiv.org/pdf/2601.12154)**

> **作者:** Teodor-Călin Ionescu; Lifeng Han; Jan Heijdra Suasnabar; Anne Stiggelbout; Suzan Verberne
>
> **备注:** accepted by the CLIN journal. The CLIN Journal is the journal for research in computational linguistics in The Netherlands and Belgium
>
> **摘要:** This study investigates the use of neural topic modeling and LLMs to uncover meaningful themes from patient storytelling data, to offer insights that could contribute to more patient-oriented healthcare practices. We analyze a collection of transcribed interviews with cancer patients (132,722 words in 13 interviews). We first evaluate BERTopic and Top2Vec for individual interview summarization by using similar preprocessing, chunking, and clustering configurations to ensure a fair comparison on Keyword Extraction. LLMs (GPT4) are then used for the next step topic labeling. Their outputs for a single interview (I0) are rated through a small-scale human evaluation, focusing on {coherence}, {clarity}, and {relevance}. Based on the preliminary results and evaluation, BERTopic shows stronger performance and is selected for further experimentation using three {clinically oriented embedding} models. We then analyzed the full interview collection with the best model setting. Results show that domain-specific embeddings improved topic \textit{precision} and \textit{interpretability}, with BioClinicalBERT producing the most consistent results across transcripts. The global analysis of the full dataset of 13 interviews, using the BioClinicalBERT embedding model, reveals the most dominant topics throughout all 13 interviews, namely ``Coordination and Communication in Cancer Care Management" and ``Patient Decision-Making in Cancer Treatment Journey''. Although the interviews are machine translations from Dutch to English, and clinical professionals are not involved in this evaluation, the findings suggest that neural topic modeling, particularly BERTopic, can help provide useful feedback to clinicians from patient interviews. This pipeline could support more efficient document navigation and strengthen the role of patients' voices in healthcare workflows.
>
---
#### [replaced 008] CALM-IT: Generating Realistic Long-Form Motivational Interviewing Dialogues with Dual-Actor Conversational Dynamics Tracking
- **分类: cs.CL**

- **简介: 该论文提出CALM-IT框架，用于生成真实感强的长期激励访谈对话，解决现有系统缺乏动态跟踪的问题，通过建模客户与顾问状态提升对话质量。**

- **链接: [https://arxiv.org/pdf/2601.10085](https://arxiv.org/pdf/2601.10085)**

> **作者:** Viet Cuong Nguyen; Nhi Yen Nguyen; Kristin A. Candan; Mary Conlon; Vanessa Rumie; Kristen Risola; Michael L. Birnbaum; Munmun De Choudhury
>
> **备注:** 53 pages, in submission to EMNLP
>
> **摘要:** Therapeutic dialogue is not a sequence of isolated responses: client goals, motivation, resistance, and therapeutic alliance evolve over time. Yet current LLM-based mental health dialogue systems often lack explicit mechanisms for tracking these dynamics across extended interactions, which can lead to poorly timed interventions or premature goal resolution. We introduce CALM-IT, a framework for generating and evaluating long-form Motivational Interviewing dialogues through explicit modeling of evolving client and counselor states, guiding both counseling strategy selection and utterance generation. We evaluate CALM-IT on a large-scale corpus of 8,232 synthetic dialogues spanning multiple dialogue lengths and frameworks. Compared with all baselines, CALM-IT achieves the best performance on most MITI 4.2 global ratings, including Empathy, Partnership, and Softening Sustain Talk, as well as on other key performance metrics while exhibiting minimal performance degradation as dialogue length increases. Notably, although CALM-IT initiates fewer change-directed prompts, it produces the highest client acceptance rate (64.3%) on average across different length conditions. We release a reproducible generation framework, a MITI-grounded process-level evaluation protocol, and a large-scale synthetic corpus for studying therapeutic LLMs under realistic long-form interaction conditions.
>
---
#### [replaced 009] BenGER Platform: A Collaborative Web Platform for End-to-End Benchmarking of German Legal Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BenGER平台，用于德国法律任务的端到端基准测试。解决法律领域LLM评估流程分散的问题，整合任务设计、标注、模型运行和多维度评估。**

- **链接: [https://arxiv.org/pdf/2604.13583](https://arxiv.org/pdf/2604.13583)**

> **作者:** Sebastian Nagl; Matthias Grabmair
>
> **备注:** Preprint - Accepted at ICAIL 2026
>
> **摘要:** Evaluating large language models (LLMs) for legal reasoning requires workflows that span task design, expert annotation, model execution, and metric-based evaluation. In practice, these steps are split across platforms and scripts, limiting transparency, reproducibility, and participation by non-technical legal experts. We present the BenGER (Benchmark for German Law) framework, an open-source web platform that integrates task creation, collaborative annotation, configurable LLM runs, and evaluation with lexical, semantic, factual, and judge-based metrics. BenGER supports multi-organization projects with tenant isolation and role-based access control, and can optionally provide formative, reference-grounded feedback to annotators. We will demonstrate a live deployment showing end-to-end benchmark creation and analysis.
>
---
#### [replaced 010] Differential syntactic and semantic encoding in LLMs
- **分类: cs.CL; cs.AI; cs.LG; physics.comp-ph**

- **简介: 该论文研究LLM中语法和语义信息的编码方式，通过分析DeepSeek-V3的层表示，发现语法和语义可部分线性分离，揭示其编码差异。**

- **链接: [https://arxiv.org/pdf/2601.04765](https://arxiv.org/pdf/2601.04765)**

> **作者:** Santiago Acevedo; Alessandro Laio; Marco Baroni
>
> **摘要:** We study how syntactic and semantic information is encoded in inner layer representations of Large Language Models (LLMs), focusing on the very large DeepSeek-V3. We find that, by averaging hidden-representation vectors of sentences sharing syntactic structure or meaning, we obtain vectors that capture a significant proportion of the syntactic and semantic information contained in the representations. In particular, subtracting these syntactic and semantic ``centroids'' from sentence vectors strongly affects their similarity with syntactically and semantically matched sentences, respectively, suggesting that syntax and semantics are, at least partially, linearly encoded. We also find that the cross-layer encoding profiles of syntax and semantics are different, and that the two signals can to some extent be decoupled, suggesting differential encoding of these two types of linguistic information in LLM representations.
>
---
#### [replaced 011] BEAR: Budgeted Evidence Allocation for Multi-Document Reasoning
- **分类: cs.CL**

- **简介: 该论文提出BEAR框架，解决多文档推理中的证据分配问题，通过结构化方式在有限预算内高效整合证据。**

- **链接: [https://arxiv.org/pdf/2601.18116](https://arxiv.org/pdf/2601.18116)**

> **作者:** Lin Sun; Linglin Zhang; Jingang Huang; Change Jia; Zhengwei Cheng; Xiangzheng Zhang
>
> **摘要:** We argue that multi-document reasoning is constrained not only by how much text a model can read, but also by how limited query-time evidence budget is allocated across documents and semantic granularities. Full-context inference exposes the model to broad evidence non-selectively and at high per-query cost, while flat chunk retrieval often returns locally relevant passages that are weakly organized for cross-document synthesis. We present \textbf{BEAR}, a framework for structured evidence allocation that builds hierarchical semantic indices offline and performs coarse-to-fine evidence access at query time through complementary \emph{exploration} and \emph{recovery} paths. This coarse-to-fine design can be viewed as structured evidence allocation under a fixed evidence-context budget. Across synthetic and real-world benchmarks, BEAR performs particularly strongly on DragonBall, remains competitive with strong retrieval-based baselines on HotpotQA, and yields the best retrieval-based result on 2Wiki under our evaluated protocol, while operating under substantially smaller \emph{query-time evidence budgets} than the reported long-context references. Additional analyses suggest that the gains are associated with hierarchy as an allocation substrate together with complementary exploration and recovery, rather than semantic chunking alone.
>
---
#### [replaced 012] MerLean-Prover: A Recursive Looping Harness for Lean 4 Theorem Proving
- **分类: cs.LO; cs.CL**

- **简介: 该论文提出MerLean-Prover，一个用于Lean 4定理证明的递归循环框架，解决自动证明问题。通过替换sorry声明为可验证证明，提升证明效率与效果。**

- **链接: [https://arxiv.org/pdf/2605.26959](https://arxiv.org/pdf/2605.26959)**

> **作者:** Jinzheng Li; Zeru Zhu; Yuanjie Ren
>
> **摘要:** MerLean-Prover is an end-to-end Lean4 theorem prover that replaces sorry declarations with kernel-checkable proofs. It is built from three agent types (Planning, Check, and Lean) composed by a recursive outer loop whose unit of revision is the proof plan itself, and uses no fine-tuning, no custom RL objective, and no theorem-specific scaffolding. On FormalQualBench, a benchmark of 23 PhD-qualifying-exam theorems, MerLean-Prover solves 10/23, surpassing the strongest published open-source baseline (OpenGauss, 8/23). On Putnam2025, the same harness closes 12/12 with substantially lower total wall-clock than the next-best system that closes the full set. The harness also transfers to smaller models: Sonnet closes all four tested FormalQualBench problems, and Haiku closes the two short ones. These results suggest that harness design is a central factor in end-to-end Lean4 theorem proving, alongside raw model capability, and that a relatively simple harness can already be effective.
>
---
#### [replaced 013] PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于大语言模型对齐任务，旨在解决多价值冲突下的指令瓶颈问题。提出PICACO方法，通过优化多值相关性提升模型对齐效果。**

- **链接: [https://arxiv.org/pdf/2507.16679](https://arxiv.org/pdf/2507.16679)**

> **作者:** Han Jiang; Dongyao Zhu; Xiaoyuan Yi; Ziang Xiao; Zhihua Wei; Xing Xie
>
> **备注:** ICML 2026
>
> **摘要:** In-Context Learning has shown great potential for aligning Large Language Models (LLMs) with human values, helping reduce harmful outputs and accommodate diverse preferences without costly post-training, known as In-Context Alignment (ICA). However, LLMs' comprehension of input prompts remains agnostic, limiting ICA's ability to address value tensions--human values are inherently pluralistic, often imposing conflicting demands, e.g., stimulation vs. tradition. Current ICA methods therefore face the Instruction Bottleneck challenge, where LLMs struggle to reconcile multiple intended values within a single prompt, leading to incomplete or biased alignment. To address this, we propose PICACO, a novel pluralistic ICA method. Without fine-tuning, PICACO optimizes a meta-instruction that incorporates multiple values to better elicit LLMs' understanding of them and improve alignment. This is achieved by maximizing the total correlation between specified values and LLM responses, which theoretically reinforces value conformity and reduces distractive noise, resulting in more effective instructions. Extensive experiments on five value sets show that PICACO works well with both black-box and open-source LLMs, outperforms several recent strong baselines, and achieves a better balance across up to 8 distinct values.
>
---
#### [replaced 014] Trust Me, I'm an Expert: Decoding and Steering Authority Bias in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，研究语言模型在权威性信息影响下的偏差问题。通过实验发现模型对高权威来源的错误信息更易受影响，并提出方法减少这种偏差。**

- **链接: [https://arxiv.org/pdf/2601.13433](https://arxiv.org/pdf/2601.13433)**

> **作者:** Priyanka Mary Mammen; Emil Joswin; Shankar Venkitachalam
>
> **摘要:** Prior research demonstrates that performance of language models on reasoning tasks can be influenced by suggestions, hints and endorsements. However, the influence of endorsement source credibility remains underexplored. We investigate whether language models exhibit systematic bias based on the perceived expertise of the provider of the endorsement. Across 4 datasets spanning mathematical, legal, and medical reasoning, we evaluate 11 models using personas representing four expertise levels per domain. Our results reveal that models are increasingly susceptible to incorrect/misleading endorsements as source expertise increases, with higher-authority sources inducing not only accuracy degradation but also increased confidence in wrong answers. We also show that this authority bias is mechanistically encoded within the model and a model can be steered away from the bias, thereby improving its performance even when an expert gives a misleading endorsement.
>
---
#### [replaced 015] Rethinking Layer Redundancy: Calibration Matters More Than Search in LLM Depth Pruning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型压缩任务，解决深度剪枝中的层冗余问题。通过实验发现，校准配置比搜索算法对剪枝效果影响更大。**

- **链接: [https://arxiv.org/pdf/2604.24938](https://arxiv.org/pdf/2604.24938)**

> **作者:** Minkyu Kim; Vincent-Daniel Yun; Youngrae Kim; Suin Cho; Woosang Lim; Sunwoo Lee
>
> **备注:** Preprint
>
> **摘要:** Depth pruning improves the inference efficiency of large language models by removing Transformer blocks. Prior work typically treats layer redundancy as an inherent structural property of pretrained networks, emphasizing importance criteria and search algorithms to identify removable layers. In this study, we empirically investigate depth pruning from a functional perspective. Evaluating representative LLM families across diverse calibration configurations and multiple search algorithms, we show that different configurations produce different pruning patterns. Furthermore, under a fixed calibration configuration, complex search algorithms yield marginal performance improvements over simple one-shot methods, converging to similar pruned subsets. Overall, our results suggest that the calibration configuration plays a substantially larger role than the choice of search algorithm in shaping pruning patterns and calibration perplexity, while contributing comparably to variance in downstream reasoning accuracy. This indicates that future pruning efforts may benefit from prioritizing the calibration configuration over search complexity.
>
---
#### [replaced 016] Assessing Factual Music Comprehension in Large Audio Language Models
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文属于音乐理解任务，旨在解决LALMs在音乐事实性回答上的评估不足问题。提出新评估协议，通过结构化信息提取与指标评估，构建音乐理解基准。**

- **链接: [https://arxiv.org/pdf/2511.05550](https://arxiv.org/pdf/2511.05550)**

> **作者:** Daniel Chenyu Lin; Michael Freeman; John Thickstun
>
> **备注:** 16 pages; second submission
>
> **摘要:** Large audio language models (LALMs) leverage multimodal representations to generate open-ended answers to natural language queries about audio. In this paper, we (1) provide empirical evidence that assessment of LALMs using the popular MusicQA dataset fails to measure whether a model's responses about music are factually correct, and (2) develop a new protocol for assessing the music comprehension capabilities of LALMs. Specifically, we propose an evaluation protocol that prompts a LALM for factually verifiable information, and parses its open-ended response into a structured format that can be objectively assessed using Precision, Recall, and F1 scores. Using this protocol, we define a benchmark consisting of six factual information retrieval tasks defined on three diverse datasets: MusicNet, the Free Music Archive, and OverClocked ReMix. We benchmark nine recent LALMs, including frontier models like Gemini and the latest open models like Music Flamingo, and release the suite of evaluation scripts at this https URL to facilitate benchmarking of new LALMs.
>
---
#### [replaced 017] PEAR: Pairwise Evaluation for Automatic Relative Scoring in Machine Translation
- **分类: cs.CL**

- **简介: 该论文提出PEAR，用于机器翻译质量评估的成对评价方法，解决无参考评估问题。通过成对比较提升评估效果，减少冗余信号，优化解码效率。**

- **链接: [https://arxiv.org/pdf/2601.18006](https://arxiv.org/pdf/2601.18006)**

> **作者:** Lorenzo Proietti; Roman Grundkiewicz; Matt Post
>
> **备注:** ACL 2026 Main Conference. 19 pages
>
> **摘要:** We present PEAR (Pairwise Evaluation for Automatic Relative Scoring), a supervised quality estimation (QE) metric family that reframes reference-free machine translation (MT) evaluation as a graded pairwise comparison. Given a source segment and two candidate translations, PEAR predicts the direction and magnitude of their quality difference. The metrics are trained using pairwise supervision derived from differences in human judgments, with an additional regularization term that encourages sign inversion under candidate order reversal. On the WMT24 meta-evaluation benchmark, PEAR outperforms strictly matched single-candidate QE baselines trained with the same data and backbones, isolating the benefit of the proposed pairwise formulation. Despite using substantially fewer parameters than recent large metrics, PEAR surpasses far larger QE models and reference-based metrics. Our analysis further indicates that PEAR yields a less redundant evaluation signal relative to other top metrics. Finally, we show that PEAR is an effective utility function for minimum Bayes risk (MBR) decoding, reducing pairwise scoring cost at negligible impact.
>
---
#### [replaced 018] Evaluating the Generation Capabilities of Large Chinese Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型评估任务，旨在解决中文大模型生成能力的客观评价问题。提出CG-Eval框架和Gscore指标，实现自动化、多维度评估。**

- **链接: [https://arxiv.org/pdf/2308.04823](https://arxiv.org/pdf/2308.04823)**

> **作者:** Hui Zeng; Jingyuan Xue; Meng Hao; Chen Sun; Bin Ning; Na Zhang
>
> **摘要:** This paper unveils CG-Eval, the first-ever comprehensive and automated evaluation framework designed for assessing the generative capabilities of large Chinese language models across a spectrum of academic disciplines. CG-Eval stands out for its automated process, which critically assesses models based on their proficiency in generating precise and contextually relevant responses to a diverse array of questions within six key domains: Science and Engineering, Humanities and Social Sciences, Mathematical Calculations, Medical Practitioner Qualification Examination, Judicial Examination, and Certified Public Accountant Examination. Alongside this, we introduce Gscore, an innovative composite index developed from a weighted sum of multiple metrics. Gscore uniquely automates the quality measurement of a model's text generation against reference standards, providing a detailed and nuanced assessment of model performance. This automation not only enhances the efficiency and scalability of the evaluation process but also ensures objective and consistent assessment across various models. The detailed test data and results, highlighting the robust capabilities and comparative performance of the evaluated models, are accessible at this http URL.
>
---
#### [replaced 019] SSDAU: Structured Semantic Data Augmentation for Joint Entity and Relation Extraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对联合实体和关系抽取任务，解决弱泛化问题。提出SSDAU方法，在数据增强中保持语义结构，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.23440](https://arxiv.org/pdf/2605.23440)**

> **作者:** Jiawei He; Mengyu Shi; Jiawei Liu; Dong Sun; Zhijie Wang; Chunrong Fang; Xikai Yang; Zhenyu Chen
>
> **备注:** 12 pages, 3 figure
>
> **摘要:** Joint Entity and Relation Extraction (JERE) is highly susceptible to weak generalization due to low-quality training data. Data augmentation is a common strategy to enhance model generalization across different domains. However, existing data augmentation methods often overlook text relevance and may disrupt semantic structures and dependencies, making it difficult to generate effective augmented data for improving model generalization. In this paper, we propose Structured Semantic Data Augmentation (SSDAU), a novel method designed to preserve the semantic structure of text during augmentation. SSDAU segments text based on entity labels and employs an encoder to capture semantic features of entities through context awareness. It then performs entity semantic restructuring to generate augmented data. To distinguish semantically similar entities, SSDAU fuses contextualized embeddings with traditional similarity scores. To mitigate potential topic ambiguity and information loss, we apply the BERTTopic model to filter out irrelevant topics, ensuring topic consistency. We evaluate SSDAU on datasets with different annotation types and compare its performance on five representative JERE models against seven popular data augmentation baselines. Experiments demonstrate that SSDAU generates semantically consistent data with superior robustness against ambiguity (8.26% F1 decrease vs. 31.91% for baselines), significantly outperforming all existing methods across all metrics.
>
---
#### [replaced 020] ViCA: Efficient Multimodal LLMs with Vision-Only Cross-Attention
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态大语言模型任务，旨在解决视觉处理计算开销大的问题。提出ViCA架构，通过稀疏跨注意力减少视觉计算，提升效率。**

- **链接: [https://arxiv.org/pdf/2602.07574](https://arxiv.org/pdf/2602.07574)**

> **作者:** Wenjie Liu; Hao Wu; Xin Qiu; Xudong Wang; Yingqi Fan; Yihan Zhang; Anhao Zhao; Yunpu Ma; Xiaoyu Shen
>
> **摘要:** Modern multimodal large language models (MLLMs) adopt a unified self-attention design that processes visual and textual tokens at every Transformer layer, incurring substantial computational overhead. In this work, we revisit the necessity of such dense visual processing and show that projected visual embeddings are already well-aligned with the language space, while effective vision-language interaction occurs in only a small subset of layers. Based on these insights, we propose ViCA (Vision-only Cross-Attention), a minimal MLLM architecture in which visual tokens bypass all self-attention and feed-forward layers, interacting with text solely through sparse cross-attention at selected layers. Extensive evaluations across three MLLM backbones, nine multimodal benchmarks, and 26 pruning-based baselines show that ViCA preserves 98% of baseline accuracy while reducing visual-side computation to 4%, consistently achieving superior performance-efficiency trade-offs. Moreover, ViCA provides a regular, hardware-friendly inference pipeline that yields over 3.5x speedup in single-batch inference and over 10x speedup in multi-batch inference, reducing visual grounding to near-zero overhead compared with text-only LLMs. It is also orthogonal to token pruning methods and can be seamlessly combined for further efficiency gains. Our code is available at this https URL.
>
---
#### [replaced 021] Decoupling Skeleton and Flesh: Efficient Multimodal Table Reasoning with Disentangled Alignment and Structure-aware Guidance
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于表格推理任务，解决LVLM在复杂表格结构下的理解与推理问题。提出DiSCo和Table-GLS框架，实现结构与内容解耦及结构引导推理，无需外部工具和大量标注。**

- **链接: [https://arxiv.org/pdf/2602.03491](https://arxiv.org/pdf/2602.03491)**

> **作者:** Yingjie Zhu; Xuefeng Bai; Kehai Chen; Yang Xiang; Youcheng Pan; Xiaoqiang Zhou; Min Zhang
>
> **备注:** Accepted as a Spotlight Paper at ICML 2026
>
> **摘要:** Reasoning over table images remains challenging for Large Vision-Language Models (LVLMs) due to complex layouts and tightly coupled structure-content information. Existing solutions often depend on expensive supervised training, reinforcement learning, or external tools, limiting efficiency and scalability. This work addresses a key question: how to adapt LVLMs to table reasoning with minimal annotation and no external tools? Specifically, we first introduce DiSCo, a Disentangled Structure-Content alignment framework that explicitly separates structural abstraction from semantic grounding during multimodal alignment, efficiently adapting LVLMs to tables structures. Building on DiSCo, we further present Table-GLS, a Global-to-Local Structure-guided reasoning framework that performs table reasoning via structured exploration and evidence-grounded inference. Extensive experiments across diverse benchmarks demonstrate that our framework efficiently enhances LVLM's table understanding and reasoning capabilities, particularly generalizing to unseen table structures. Our data and code are available at this https URL.
>
---
#### [replaced 022] Retention Consequence in Lifecycle Memory Control
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究持久内存中退化问题，属于内存管理任务。解决post-admission失效问题，提出显式保留状态机制，通过StageMem实验验证其有效性。**

- **链接: [https://arxiv.org/pdf/2604.16774](https://arxiv.org/pdf/2604.16774)**

> **作者:** Jiarui Han
>
> **摘要:** Persistent memory can fail after successful admission: a premise is written, then becomes a silent assumption, and later maintenance treats it as ordinary residue to be compressed, demoted, or evicted. We study this post-admission failure as a lifecycle-control problem. Existing memory systems already perform admission, update, compression, retrieval, and eviction. Our claim is not that such systems lack maintenance, but that retention consequence is often operationalized only indirectly through validity, similarity, recency, frequency, importance, or summarization signals rather than exposed as a separate lifecycle state. We therefore treat confidence as carried-forward validity/support evidence, and introduce strength as an explicit lifecycle state for retention consequence. We operationalize this distinction in StageMem, a small staged controller whose transient, working, and durable stores expose promotion, compression, and eviction pressure points. Across controlled premise-realization, compression, pressure, and implicit-heuristic diagnostics, the experiments separate writing too little, retaining the wrong high-cue content, forgetting costly premises, and preserving everything by saturation. Explicit retention consequence, used through lifecycle settlement, provides a control surface between omission and hoarding. For the targeted post-admission failure mode, the results support a lifecycle view of persistent memory: reliability depends not only on what enters memory, but on whether admission validity and retention consequence remain available during maintenance.
>
---
#### [replaced 023] CircuitLM: A Multi-Agent LLM-Aided Design Framework for Generating Circuit Schematics from Natural Language Prompts
- **分类: cs.AI; cs.CL; eess.SY**

- **简介: 该论文属于电子设计自动化任务，旨在解决从自然语言生成准确电路图的问题。通过多智能体框架CircuitLM，结合知识库和验证机制，提升生成电路的准确性和物理可行性。**

- **链接: [https://arxiv.org/pdf/2601.04505](https://arxiv.org/pdf/2601.04505)**

> **作者:** Khandakar Shakib Al Hasan; Syed Rifat Raiyan; Hasin Mahtab Alvee; Wahid Sadik
>
> **备注:** Accepted at the 2026 IEEE International Conference on LLM-Aided Design (ICLAD), 10 pages, 8 figures, 6 tables
>
> **摘要:** Generating accurate circuit schematics from high-level natural language descriptions remains a persistent challenge in electronic design automation (EDA), as large language models (LLMs) frequently hallucinate components, violate strict physical constraints, and produce non-machine-readable outputs. To address this, we present CircuitLM, a multi-agent pipeline that translates user prompts into structured, visually interpretable $\texttt{CircuitJSON}$ schematics. The framework mitigates hallucination and ensures physical viability by grounding generation in a curated, embedding-powered component knowledge base through five sequential stages: (i) component identification, (ii) canonical pinout retrieval, (iii) chain-of-thought reasoning, (iv) JSON schematic synthesis, and (v) interactive force-directed visualization. We evaluate the system on a dataset of 100 unique circuit-design prompts using five state-of-the-art LLMs. To systematically assess performance, we deploy a rigorous dual-layered evaluation methodology: a deterministic Electrical Rule Checking (ERC) engine categorizes topological faults by strict severity (Critical, Major, Minor, Warning), while an LLM-as-a-judge meta-evaluator identifies complex, context-aware design flaws that bypass standard rule-based checkers. Ultimately, this work demonstrates how targeted retrieval combined with deterministic and semantic verification can bridge natural language to structurally viable, schematic-ready hardware and safe circuit prototyping. Our code and data are publicly available at this https URL.
>
---
#### [replaced 024] MobileGym: A Verifiable and Highly Parallel Simulation Platform for Mobile GUI Agent Research
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出MobileGym，一个用于移动GUI代理研究的可验证、高并行的仿真平台。解决日常应用中难以实现的确定性评估和大规模在线强化学习问题，通过结构化状态管理和任务模板实现高效训练与测试。**

- **链接: [https://arxiv.org/pdf/2605.26114](https://arxiv.org/pdf/2605.26114)**

> **作者:** Dingbang Wu; Rui Hao; Haiyang Wang; Shuzhe Wu; Han Xiao; Zhenghong Li; Bojiang Zhou; Zheng Ju; Zichen Liu; Lue Fan; Zhaoxiang Zhang
>
> **备注:** Project page: this https URL
>
> **摘要:** We present MobileGym, a browser-hosted, lightweight, fully controllable environment for everyday mobile use, targeting interaction fidelity without replicating proprietary backends. It enables two capabilities previously out of reach for everyday apps: verifiable outcome signals through deterministic state-based judging over structured JSON state, and scalable online RL through low-cost parallel rollouts. The full environment state is captured, configured, forked, and compared as structured JSON, and a single server can host hundreds of parallel instances, with about 400 MB memory per instance and about 3 s cold start. A layered state model and a declarative task-definition framework keep state programmability and task creation practical at scale, and a single programmatic judging mechanism delivers both deterministic evaluation verdicts and dense RL rewards. The accompanying MobileGym-Bench provides 416 parameterized task templates, including 256 test and 160 train templates, over 28 apps, with deterministic judges and a structured AnswerSheet protocol that avoids free-text matching failures. In a Sim-to-Real case study, GRPO on Qwen3-VL-4B-Instruct gains +12.8 percentage points on the 256-task test set, and on a 59-task real-device signal subset, real-device execution retains 95.1% of the simulation-side training gain. Project page: this https URL.
>
---
#### [replaced 025] UKP_Psycontrol at SemEval-2026 Task 2: Modeling Valence and Arousal Dynamics from Text
- **分类: cs.CL**

- **简介: 该论文针对SemEval-2026 Task 2，解决用户生成文本中情绪状态及短期变化建模问题，提出三种方法进行情感动态分析。**

- **链接: [https://arxiv.org/pdf/2604.21534](https://arxiv.org/pdf/2604.21534)**

> **作者:** Darya Hryhoryeva; Amaia Zurinaga; Hamidreza Jamalabadi; Iryna Gurevych
>
> **备注:** Accepted to SemEval 2026 (co-located with ACL 2026)
>
> **摘要:** This paper presents our system developed for SemEval-2026 Task 2. The task requires modeling both current affect and short-term affective change in chronologically ordered user-generated texts. We explore three complementary approaches: (1) LLM prompting under user-aware and user-agnostic settings, (2) a pairwise Maximum Entropy (MaxEnt) model with Ising-style interactions for structured transition modeling, and (3) a lightweight neural regression model incorporating recent affective trajectories and trainable user embeddings. Our findings indicate that LLMs effectively capture static affective signals from text, whereas short-term affective variation in this dataset is more strongly explained by recent numeric state trajectories than by textual semantics. Our system ranked first among participating teams in both Subtask 1 and Subtask 2A based on the official evaluation metric.
>
---
#### [replaced 026] HGMEM: Hypergraph-based Working Memory to Improve Multi-step RAG for Long-Context Complex Relational Modeling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决多步RAG中因记忆结构静态导致的推理能力不足问题。提出HGMem，通过超图结构增强记忆的动态表达与高阶关联，提升复杂关系建模能力。**

- **链接: [https://arxiv.org/pdf/2512.23959](https://arxiv.org/pdf/2512.23959)**

> **作者:** Chulun Zhou; Chunkang Zhang; Guoxin Yu; Fandong Meng; Jie Zhou; Wai Lam; Mo Yu
>
> **备注:** ICML 2026; Code released at this https URL
>
> **摘要:** Multi-step retrieval-augmented generation (RAG) has become a widely adopted strategy for enhancing large language models (LLMs) on tasks that demand global comprehension and intensive reasoning. Although many RAG systems incorporate a working memory to consolidate information, existing designs primarily function as a passive storage for isolated facts. This static nature overlooks crucial high-order correlations among primitive facts, thereby limiting models' capacity for multi-step reasoning and resulting in fragmented reasoning and weak global sense-making within extended contexts. We introduce HGMem, a hypergraph-based working memory system, extending the concept of memory beyond simple storage into a dynamic, expressive structure for complex reasoning and global understanding. In our approach, memory is represented as a hypergraph where hyperedges correspond to distinct memory units, enabling the progressive formation of high-order interactions within memory. This mechanism connects facts and thoughts around the focal problem, evolving the memory into an integrated and situated knowledge structure that provides strong propositions for deeper reasoning. We evaluate HGMem on several challenging global sense-making benchmarks. Extensive experiments and in-depth analyses demonstrate that our method consistently improves multi-step RAG and substantially outperforms strong baseline systems across diverse datasets.
>
---
#### [replaced 027] Benchmarking and Mechanistic Analysis of Vision-Language Models for Cross-Depiction Assembly Instruction Alignment
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型在装配图与视频对齐任务中的研究，旨在解决2D装配图与实际视频间的 depiction gap。通过构建IKEA-Bench数据集并评估多种模型，发现视觉编码是提升对齐效果的关键。**

- **链接: [https://arxiv.org/pdf/2604.00913](https://arxiv.org/pdf/2604.00913)**

> **作者:** Zhuchenyang Liu; Yao Zhang; Yu Xiao
>
> **摘要:** 2D assembly diagrams are often abstract and hard to follow, creating a need for intelligent assistants that can monitor progress, detect errors, and provide step-by-step guidance. In mixed reality settings, such systems must recognize completed and ongoing steps from the camera feed and align them with the diagram instructions. Vision Language Models (VLMs) show promise for this task, but face a depiction gap because assembly diagrams and video frames share few visual features. To systematically assess this gap, we construct IKEA-Bench, a benchmark of 1,623 questions across 6 task types on 29 IKEA furniture products, and evaluate 19 VLMs (2B-38B) under three alignment strategies. Our key findings: (1) assembly instruction understanding is recoverable via text, but text simultaneously degrades diagram-to-video alignment; (2) architecture family predicts alignment accuracy more strongly than parameter count; (3) video understanding remains a hard bottleneck unaffected by strategy. A three-level mechanistic analysis further reveals that diagrams and video occupy disjoint ViT subspaces, and that adding text shifts models from visual to text-driven reasoning. These results identify visual encoding as the primary target for improving cross-depiction robustness. Project page: this https URL
>
---
#### [replaced 028] Explanation Generation for Contradiction Reconciliation with LLMs
- **分类: cs.CL**

- **简介: 该论文提出“矛盾解释生成”任务，旨在让模型生成解释以调和矛盾陈述。研究探索了LLMs在这一任务上的表现及改进方法。**

- **链接: [https://arxiv.org/pdf/2603.22735](https://arxiv.org/pdf/2603.22735)**

> **作者:** Jason Chan; Zhixue Zhao; Robert Gaizauskas
>
> **备注:** Preprint
>
> **摘要:** Existing NLP work commonly treats contradictions as errors to be resolved by choosing which statements to accept or discard. Yet a key aspect of human reasoning in social interactions and professional domains is the ability to hypothesize explanations that reconcile contradictions. For example, "Cassie hates coffee" and "She buys coffee everyday" may appear contradictory, yet both are compatible if Cassie has the unenviable daily chore of buying coffee for all her coworkers. Despite the growing reasoning capabilities of large language models (LLMs), their ability to hypothesize such reconciliatory explanations remains largely unexplored. To address this gap, we introduce the task of reconciliatory explanation generation, where models must generate explanations that effectively render contradictory statements compatible. We propose a novel method of repurposing existing natural language inference (NLI) datasets, and introduce quality metrics that enable scalable automatic evaluation. Experiments with 18 LLMs show that most models achieve limited success in this task, and that the benefit of extending test-time compute by "thinking" plateaus as model size increases. Our results highlight an under-explored dimension of LLM reasoning and the need to address this limitation in enhancing LLMs' downstream applications such as chatbots and scientific aids.
>
---
#### [replaced 029] Compositional Consistency-Guided Decoding for Three-Way Logical Question Answering
- **分类: cs.CL; cs.AI; cs.LO**

- **简介: 该论文研究三元逻辑问答任务，解决大模型在否定一致性及未知预测上的问题。提出CGD-PD方法，提升准确率并减少不确定预测。**

- **链接: [https://arxiv.org/pdf/2604.06196](https://arxiv.org/pdf/2604.06196)**

> **作者:** Tianyi Huang; Ming Hou; Jiaheng Su; Yutong Zhang; Ziling Zhang
>
> **备注:** Accepted at the ICML 2026 Workshop on Compositional Learning: Safety, Interpretability, and Agents
>
> **摘要:** Three-way logical question answering (QA) assigns one of $\text{True}$, $\text{False}$, or $\text{Unknown}$ to a hypothesis $H$ given a premise set $S$. We study this task as a compact compositional inference problem: predictions for $H$ and for a mechanically negated hypothesis $\neg H$ should agree under a deterministic negation map. Despite this simple structure, large language models (LLMs) can exhibit two practical failure modes: (i) negation inconsistency, where answers to $H$ and $\neg H$ violate the required label mapping, and (ii) epistemic $\text{Unknown}$, where the model abstains even when one side is entailed. We introduce CGD-PD, a lightweight, training-free test-time layer that combines neural 3-way classification, symbolic negation-consistency projection, and targeted binary entailment probes. On one validation split of FOLIO's first-order logic fields, CGD-PD improves accuracy by 4.4 points on GPT-5.2 and 6.8 points on Claude Sonnet 4.5, while reducing $\text{Unknown}$ predictions and epistemic abstention. These results provide a controlled proof of concept that simple logical composition at inference time can help evaluate and improve LLM reasoning reliability; they do not, by themselves, establish robustness beyond this formal benchmark setting.
>
---
#### [replaced 030] Attention Sink Forges Native MoE in Attention Layers: Sink-Aware Training to Address Head Collapse
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究注意力机制中的“注意力陷阱”问题，提出一种基于负载平衡的训练方法，以解决注意力头坍缩现象，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.01203](https://arxiv.org/pdf/2602.01203)**

> **作者:** Zizhuo Fu; Wenxuan Zeng; Runsheng Wang; Meng Li
>
> **备注:** 2026 International Conference on Machine Learning (ICML)
>
> **摘要:** Large Language Models (LLMs) often assign disproportionate attention to the first token, a phenomenon known as the attention sink. Several recent approaches aim to address this issue, including Sink Attention in GPT-OSS and Gated Attention in Qwen3-Next. However, a comprehensive analysis of the relationship among these attention mechanisms is lacking. In this work, we provide both theoretical and empirical evidence demonstrating that the sink in Vanilla Attention and Sink Attention naturally construct a Mixture-of-Experts (MoE) mechanism within attention layers. This insight explains the head collapse phenomenon observed in prior work, where only a fixed subset of attention heads contributes to generation. To mitigate head collapse, we propose a sink-aware training algorithm with an auxiliary load balancing loss designed for attention layers. Extensive experiments show that our method achieves effective head load balancing and improves model performance across Vanilla Attention, Sink Attention, and Gated Attention. We hope this study offers a new perspective on attention mechanisms and encourages further exploration of the inherent MoE structure within attention layers.
>
---
#### [replaced 031] In Search of the Ingredients of Open-Endedness: Replicating Picbreeder with Large Vision-Language Models
- **分类: cs.AI; cs.CL; cs.CV; cs.NE**

- **简介: 该论文属于AI生成任务，旨在探索AI在开放性创作中的表现。通过复制Picbreeder系统，用视觉语言模型替代人类用户，研究其生成内容的差异与影响因素。**

- **链接: [https://arxiv.org/pdf/2605.23908](https://arxiv.org/pdf/2605.23908)**

> **作者:** Sam Earle; Kai Arulkumaran; Andrew Dai; Akarsh Kumar; Julian Togelius; Sebastian Risi
>
> **备注:** 26 pages, 21 figures, to be published at GECCO 2026
>
> **摘要:** We are in the midst of large-scale industrial and academic efforts to automate the processes of scientific, technological and creative production through AI-driven assistants. Historically, a fundamental property of these processes in their human form has been their open-endedness: their capacity for generating a seemingly endless supply of novel and meaningful new forms. Do artificial agents have any capacity for such fruitful unguided discovery? To answer this question, we turn to Picbreeder, the canonical exemplar of human-driven open-ended search, in which users collaboratively generated a diverse library of images through interactive evolution of small neural networks. We replicate Picbreeder, replacing human users with frontier Vision Language Models (VLMs). We observe clear qualitative differences between the output of our system and the historical human baseline, and attempt to characterize them using metrics of phylogenetic complexity and visual and semantic salience and novelty. In an effort to identify some of the causal factors contributing these differences, we study the addition of exploratory noise to the agents' selection process, of behavioral diversity between agents, and of narrative momentum in the form of memory of past actions. We make our code available at this https URL.
>
---
#### [replaced 032] Apple Intelligence Foundation Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 本文介绍用于Apple Intelligence的基座语言模型，解决设备端与云端高效、准确执行任务的问题，涵盖模型架构、训练数据、优化及评估。**

- **链接: [https://arxiv.org/pdf/2407.21075](https://arxiv.org/pdf/2407.21075)**

> **作者:** Tom Gunter; Zirui Wang; Chong Wang; Ruoming Pang; Andy Narayanan; Aonan Zhang; Bowen Zhang; Chen Chen; Chung-Cheng Chiu; David Qiu; Deepak Gopinath; Dian Ang Yap; Dong Yin; Feng Nan; Floris Weers; Guoli Yin; Haoshuo Huang; Jianyu Wang; Jiarui Lu; John Peebles; Ke Ye; Mark Lee; Nan Du; Qibin Chen; Quentin Keunebroek; Sam Wiseman; Syd Evans; Tao Lei; Vivek Rathod; Xiang Kong; Xianzhi Du; Yanghao Li; Yongqiang Wang; Yuan Gao; Zaid Ahmed; Zhaoyang Xu; Zhiyun Lu; Al Rashid; Albin Madappally Jose; Alec Doane; Alfredo Bencomo; Allison Vanderby; Andrew Hansen; Ankur Jain; Anupama Mann Anupama; Areeba Kamal; Bugu Wu; Carolina Brum; Charlie Maalouf; Chinguun Erdenebileg; Chris Dulhanty; Daniel Parilla; Dominik Moritz; Doug Kang; Eduardo Jimenez; Evan Ladd; Fangping Shi; Felix Bai; Frank Chu; Fred Hohman; Hadas Kotek; Hannah Gillis Coleman; Jane Li; Jeffrey Bigham; Jeffery Cao; Jeff Lai; Jessica Cheung; Jiulong Shan; Joe Zhou; John Li; Jun Qin; Karanjeet Singh; Karla Vega; Kelvin Zou; Laura Heckman; Lauren Gardiner; Margit Bowler; Maria Cordell; Meng Cao; Nicole Hay; Nilesh Shahdadpuri; Otto Godwin; Pranay Dighe; Pushyami Rachapudi; Ramsey Tantawi; Roman Frigg; Sam Davarnia; Sanskruti Shah; Saptarshi Guha; Sasha Sirovica; Shen Ma; Shuang Ma; Simon Wang; Sulgi Kim; Suma Jayaram; Vaishaal Shankar; Varsha Paidi; Vivek Kumar; Xin Wang; Xin Zheng
>
> **摘要:** We present foundation language models developed to power Apple Intelligence features, including a ~3 billion parameter model designed to run efficiently on devices and a large server-based language model designed for Private Cloud Compute. These models are designed to perform a wide range of tasks efficiently, accurately, and responsibly. This report describes the model architecture, the data used to train the model, the training process, how the models are optimized for inference, and the evaluation results. We highlight our focus on Responsible AI and how the principles are applied throughout the model development.
>
---
#### [replaced 033] Which Heads Matter for Reasoning? RL-Guided KV Cache Compression
- **分类: cs.CL**

- **简介: 该论文属于大模型推理优化任务，旨在解决KV缓存压缩中信息丢失导致的推理失效问题。通过强化学习识别关键注意力头，实现高效压缩与性能保持。**

- **链接: [https://arxiv.org/pdf/2510.08525](https://arxiv.org/pdf/2510.08525)**

> **作者:** Wenjie Du; Li Jiang; Keda Tao; Xue Liu; Huan Wang
>
> **摘要:** Reasoning large language models exhibit complex reasoning behaviors via extended chain-of-thought generation that are highly fragile to information loss during decoding, creating critical challenges for KV cache compression. Existing token-dropping methods directly disrupt reasoning chains by removing intermediate steps, while head-reallocation methods, designed for retrieval tasks, fail to preserve the heads essential for generative reasoning. However, no existing method can identify which attention heads genuinely maintain reasoning consistency and control generation termination. To address this, we propose RLKV, which uses reinforcement learning as a probe to discover which heads contribute to reasoning quality by directly optimizing their cache usage against actual generation outcomes. This discovery naturally leads to an efficient compression strategy: we allocate full KV cache to reasoning-critical heads while aggressively compressing others with constant-size KV cache. Experiments reveal that a fraction of heads proves essential for reasoning, enabling 20--60% cache reduction with near-lossless performance across diverse tasks and models, and up to 2.06x end-to-end speedup at 60% reduction.
>
---
#### [replaced 034] RMPL: Relation-aware Multi-task Progressive Learning with Stage-wise Training for Multimedia Event Extraction
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多媒体事件抽取任务，解决数据稀缺下事件与论元识别问题。提出RMPL框架，结合多任务和阶段训练，提升跨模态事件表示学习效果。**

- **链接: [https://arxiv.org/pdf/2602.13748](https://arxiv.org/pdf/2602.13748)**

> **作者:** Yongkang Jin; Jianwen Luo; Jingjing Wang; Jianmin Yao; Yu Hong
>
> **备注:** Accepted by ACM ICMR 2026
>
> **摘要:** Multimedia Event Extraction (MEE) aims to identify events and their arguments from documents that contain both text and images. It requires grounding event semantics across different modalities. Progress in MEE is limited by the lack of annotated training data. M2E2 is the only established benchmark, but it provides annotations only for evaluation. This makes direct supervised training impractical. Existing methods mainly rely on cross-modal alignment or inference-time prompting with Vision--Language Models (VLMs). These approaches do not explicitly learn structured event representations and often produce weak argument grounding in multimodal settings. To address these limitations, we propose RMPL, a Relation-aware Multi-task Progressive Learning framework for MEE under low-resource conditions. RMPL incorporates heterogeneous supervision from unimodal event extraction and multimedia relation extraction with stage-wise training. The model is first trained with a unified schema to learn shared event-centric representations across modalities. It is then fine-tuned for event mention identification and argument role extraction using mixed textual and visual data. Experiments on the M2E2 benchmark with multiple VLMs show consistent improvements across different modality settings.
>
---
#### [replaced 035] PRISM: A Multi-Dimensional Benchmark for Evaluating LLM Peer Reviewers
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的评测任务，旨在评估LLM作为同行评审者的质量。通过PRISM框架，从四个维度分析LLM与人类评审的差异，发现LLM在特定方面表现优异，但整体仍无法全面替代人类。**

- **链接: [https://arxiv.org/pdf/2605.26730](https://arxiv.org/pdf/2605.26730)**

> **作者:** Ngoc Phan Phuoc Loc; Toan Huynh La Viet; Thanh Tran Khanh; Duy A Nguyen; Tuan Anh Nguyen Pham; Thanh Nguyen; Nitesh V. Chawla; Wray Buntine; Kok-Seng Wong; Khoa D. Doan; Binh T. Nguyen
>
> **摘要:** The rapid growth in submissions to machine learning venues has strained the scientific peer-review system and intensified interest in LLM-based automated peer reviewers. However, how good these systems are actually, especially compared to human reviewers at catching scientific gaps, remains poorly understood. In this work, we introduce PRISM (Peer Review Intelligence via Structured Multi-dimensional assessment), a benchmarking framework that evaluates review quality across four dimensions: Depth of Analysis, Novelty Assessment,Flaw Identification & Major Issues Prioritization, and Multi-dimensional Constructiveness. Unlike most existing evaluations based on surface-level metrics like ROUGE and BLEU, or unconstrained LLM-as-a-judge prompting that conflates fluency with rigor, PRISM grounds each dimension in argument mining, retrieval-augmented verification, and consensus-based scoring. We apply PRISM to benchmark five leading automated reviewer systems and human reviewers on a stratified corpus of reviews from ICLR, ICML, and NeurIPS. The results reveal that LLMs can match or beat human reviewers on individual dimensions: comparable depth of analysis, stronger novelty verification, and highly accurate critique prioritization. However, no single system consistently matches the balanced performance of the human baseline across all dimensions at once. Each exhibits a distinct specialization profile with characteristic blind spots -- failure modes that aggregate metrics miss entirely. The implication is that LLM reviewers are best understood as targeted supplements to human review, effective within specific dimensions, but unreliable as standalone replacements. Our demo and key results can be found at this https URL.
>
---
#### [replaced 036] Prompt Optimization Is a Coin Flip: Diagnosing When It Helps in Compound AI Systems
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究Prompt优化在复合AI系统中的有效性，分析其为何常如抛硬币般不可预测。通过实验验证任务结构对优化效果的影响，提出诊断方法以判断优化是否值得。**

- **链接: [https://arxiv.org/pdf/2604.14585](https://arxiv.org/pdf/2604.14585)**

> **作者:** Xing Zhang; Guanghui Wang; Yanwei Cui; Wei Qiu; Ziyuan Li; Bing Zhu; Peiyang He
>
> **备注:** Accepted to the 1st Workshop on Combining Theory and Benchmarks, CTB@ICML 2026, Seoul, South Korea
>
> **摘要:** Prompt optimization in compound AI systems is statistically indistinguishable from a coin flip: across 72 optimization runs on Claude Haiku 4.5 (6 methods $\times$ 4 tasks $\times$ 3 repeats), 49% score below zero-shot; on Amazon Nova Lite, the failure rate is even higher. Yet on one task, all six methods improve over zero-shot by up to $+6.8$ points. What distinguishes success from failure? We investigate with 18,000 grid evaluations and 144 optimization runs, testing two assumptions behind end-to-end optimization tools like TextGrad and DSPy, in the order they must be answered: (A) agent prompts interact, requiring joint rather than independent optimization, and (B) individual prompts are worth optimizing at all. Interaction effects are never significant ($p > 0.52$, all $F < 1.0$), and optimization helps only when the task has exploitable output structure: a format the model can produce but does not default to. We further give a mechanistic account: instruction-tuning compresses input phrasing into a narrow output distribution, eliminating the very phrasing-sensitivity that joint optimization assumes. We provide a two-stage diagnostic: an \$80 ANOVA pre-test for agent coupling, and a 10-minute headroom test that predicts whether optimization is worthwhile, turning a coin flip into an informed decision.
>
---
#### [replaced 037] Heterogeneous Dependency Graph-Guided Attentionfor Patent Representation Learning
- **分类: cs.CL**

- **简介: 该论文属于专利表示学习任务，解决传统模型忽略权利要求依赖关系的问题。提出PHAGE模型，构建异构图并引入连接掩码，提升专利分类与检索效果。**

- **链接: [https://arxiv.org/pdf/2605.10073](https://arxiv.org/pdf/2605.10073)**

> **作者:** Yongmin Yoo; Qiongkai Xu; Zhangkai Wu; Longbing Cao
>
> **摘要:** Pre-trained language models advance patent classification and retrieval via encoding claims as flat token sequences, yet overlooking the dependency hierarchy among claims. Incorporating the hierarchy into self-attention poses two challenges. First, claim dependencies involve relation types with varying reliability: treating them indiscriminately allows noisy technical relations to corrupt cleaner legal citation signals. Second, when the dependency graph is defined over claims, Transformer models fail as they operate at the token level; broadcasting claim-level adjacency can dilute structural information across unrelated token pairs. A novel Patent Heterogeneous Attention Graph Encoder (PHAGE) addresses these challenges. To handle heterogeneous dependencies, PHAGE constructs a typed graph to separate legal citations from technical relations as distinct edge types. To bridge the hierarchy gap, PHAGE introduces a connectivity mask with learnable relation-aware biases to project a claim-level topology into token-level attention. PHAGE learns a dual-granularity contrastive objective to align representations with inter-patent taxonomy and intra-patent topology. Experiments show that PHAGE outperforms domain-adapted and citation-aware baselines on patent classification, retrieval, and clustering. PHAGE discloses that the intra-patent claim topology captures stronger inductive bias than the inter-patent structure.
>
---
#### [replaced 038] Sentence Curve Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出句子曲线语言模型（SCLM），解决传统语言模型对全局句结构建模不足的问题。通过预测连续句子曲线代替静态词嵌入，增强全局结构建模能力。**

- **链接: [https://arxiv.org/pdf/2602.01807](https://arxiv.org/pdf/2602.01807)**

> **作者:** DongNyeong Heo; Taehwan Kim; Heeyoul Choi
>
> **摘要:** Language models (LMs) are a central component of modern AI systems, and diffusion language models (DLMs) have recently emerged as a competitive alternative. Both paradigms rely on word embeddings not only to represent the input sentence, but also to represent the target sentence that backbone models are trained to predict. We argue that such static embedding of the target word is insensitive to neighboring words, encouraging locally accurate word prediction while global sentence structure is less emphasized. To address this, we propose a continuous sentence representation, termed sentence curve, defined as a spline curve whose control points affect multiple words in the sentence. Based on this representation, we introduce sentence curve language model (SCLM), which extends DLMs to predict sentence curves instead of the static word embeddings. We theoretically show that sentence curve prediction induces a regularization effect that promotes global structure modeling, and characterize how different sentence curve types affect this behavior. Empirically, SCLM achieves state-of-the-art performance among DLMs on IWSLT14 and WMT14, shows stable training without burdensome knowledge distillation, and demonstrates promising potential compared to discrete DLMs on LM1B.
>
---
#### [replaced 039] RouteProfile: Graph-Based Profiling for Cold-Start LLM Routing
- **分类: cs.NI; cs.CL**

- **简介: 该论文属于LLM路由任务，解决冷启动模型集成问题。提出RouteProfile框架，通过图结构整合模型元数据，提升路由效果。**

- **链接: [https://arxiv.org/pdf/2605.00180](https://arxiv.org/pdf/2605.00180)**

> **作者:** Jingjun Xu; Hongji Pu; Tao Feng; Haozhen Zhang; Jiaxuan You; Ge Liu
>
> **摘要:** LLM routing is increasingly important for selecting suitable models under diverse user needs and deployment constraints, but its practical effectiveness depends on continual adaptation to emerging queries and newly released models. New-LLM integration is particularly challenging, as newly released models lack the query-response-reward interactions required for router training and cannot be profiled as directly as new queries via semantic embeddings. Existing profiles are limited: LLM-generated descriptions are often coarse, while interaction-based embeddings are costly to construct. To address this problem, we propose RouteProfile, a graph-based profiling framework that constructs LLM profiles from public signals in technical reports or model cards, including model family, model description, reported benchmark scores, and benchmark domains. RouteProfile organizes these heterogeneous signals into a graph and studies profile construction along four dimensions: organizational form, representation type, aggregation depth, and learning configuration. We evaluate RouteProfile in training-free cold-start routing and new-LLM integration settings. Experiments show that: (1) structured profiles outperform flat baselines in training-free cold-start routing; (2) model family metadata is more reliable than benchmark domain information; and (3) effective new-LLM integration requires profile-router co-design. Overall, our findings highlight the importance of profile design for enabling routing systems to adapt to the evolving model ecosystem.
>
---
#### [replaced 040] Tokenization with Split Trees
- **分类: cs.CL**

- **简介: 该论文提出一种新的子词分词方法ToaST，旨在优化压缩和提升语言模型效率。通过递归分割和整数规划选择词汇，减少token数量并提高Renyi效率。**

- **链接: [https://arxiv.org/pdf/2605.22705](https://arxiv.org/pdf/2605.22705)**

> **作者:** Craig W. Schmidt; Michael Krumdick; Adam Wiemerslage; Seth Ebner; Varshini Reddy; Yuval Pinter; Chris Tanner
>
> **备注:** All baseline tokenizers (BPE, WordPiece, Unigram) were trained incorrectly due to a bug in the Hugging Face tokenizers library: pair counts overflow i32 above ~108 GB of training data, dropping the most common merge pairs. All comparisons to ToaST are invalid. Thanks to Sander Land for identifying the missing merge pairs. See this https URL
>
> **摘要:** We introduce Tokenization with Split Trees (ToaST), a subword tokenization method that directly optimizes compression under a new recursive inference procedure. ToaST greedily splits each pretoken into a full binary tree using precomputed byte n-gram counts, independent of any vocabulary. Given a vocabulary, inference recursively descends each split tree and emits the first in-vocabulary node reached on each path. Vocabulary selection is formulated as an Integer Program (IP) that minimizes the total token count over all split trees under this inference procedure. The Linear Programming (LP) relaxation is near-integral in practice, yielding provably near-optimal vocabularies, with training time empirically scaling quadratically in the number of split trees. On English text, ToaST reduces token counts by more than 11% compared to BPE, WordPiece, and UnigramLM at vocabulary sizes of 40,960 and above, reducing the number of inference tokens for models using this tokenizer, thus extending the effective context length. ToaST also uses common single-byte tokens less frequently than these baselines, leading to a substantial improvement in Renyi efficiency. In experiments training 1.5B parameter language models, ToaST achieves the highest CORE score, outperforming baselines by 2.6%--7.6%, with significance for two of three, and scoring best on 13 of 22 individual tasks.
>
---
#### [replaced 041] Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何通过模拟“睡眠”机制提升Transformer模型在长序列任务中的表现，解决注意力机制随上下文变长而效率下降的问题。通过周期性地将近期信息转化为持久权重，并在“睡眠”期间进行离线递归处理，优化推理效果。**

- **链接: [https://arxiv.org/pdf/2605.26099](https://arxiv.org/pdf/2605.26099)**

> **作者:** Sangyun Lee; Sean McLeish; Tom Goldstein; Giulia Fanti
>
> **摘要:** Transformer-based large language models are increasingly used for long-horizon tasks; however, their attention mechanism scales poorly with context length. To handle this, we study a sleep-like consolidation mechanism in which a model periodically converts recent context into persistent fast weights before clearing its key-value cache. During sleep, the model performs $N$ offline recurrent passes over the accumulated context and updates the fast weights in its state-space model (SSM) blocks through a learned local rule. During inference, this shifts extra computation to sleep while preserving the latency of wake-time prediction. We test our method on controlled synthetic tasks, including cellular automata and multi-hop graph retrieval, as well as a realistic math reasoning task, on which a regular transformer as well as SSM-attention hybrid models fail. We then show that increasing sleep duration $N$ for our models improves performance, with the largest gains on examples that require deeper reasoning.
>
---
#### [replaced 042] Stop Rewarding Hallucinated Steps: Faithfulness-Aware Step-Level Reinforcement Learning for Small Reasoning Models
- **分类: cs.CL**

- **简介: 该论文属于小推理模型的可信推理任务，旨在解决模型在推理过程中产生幻觉的问题。通过引入步骤级的可信性奖励和对比信号生成方法，提升推理过程的可靠性。**

- **链接: [https://arxiv.org/pdf/2602.05897](https://arxiv.org/pdf/2602.05897)**

> **作者:** Shuo Nie; Hexuan Deng; Chao Wang; Ruiyu Fang; Xuebo Liu; Shuangyong Song; Yu Li; Min Zhang; Xuelong Li
>
> **摘要:** As large language models become smaller and more efficient, small reasoning models (SRMs) are crucial for enabling chain-of-thought (CoT) reasoning in resource-constrained settings. However, they are prone to faithfulness hallucinations, especially in intermediate reasoning steps. Existing mitigation methods based on online reinforcement learning rely on outcome-based rewards or coarse-grained CoT evaluation, which can inadvertently reinforce unfaithful reasoning when the final answer is correct. To address these limitations, we propose Faithfulness-Aware Step-Level Reinforcement Learning (FaithRL), introducing step-level supervision via explicit faithfulness rewards from a process reward model, together with an implicit truncated resampling strategy that generates contrastive signals from faithful prefixes, while also mitigating reward hacking from step-level rewards. Experiments across multiple SRMs and Open-Book QA benchmarks demonstrate that FaithRL consistently reduces hallucinations in both the CoT and final answers, leading to more faithful and reliable reasoning. Code is available at this https URL.
>
---
#### [replaced 043] Negative Advantages Is a Double-Edged Sword: Calibrating advantages in GRPO for Search Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，针对GRPO在多跳搜索中的问题，提出CalibAdv方法校准优势，提升模型性能与训练稳定性。**

- **链接: [https://arxiv.org/pdf/2604.18235](https://arxiv.org/pdf/2604.18235)**

> **作者:** Jiayi Wu; Ruobing Xie; Zeqian Huang; Lei Jiang; Can Xu; Kangyang Luo; Bochen Lin; Ming Gao; Xiang Li
>
> **摘要:** Search agents achieve strong question-answering performance through multi-turn interactions with search engines, with Group Relative Policy Optimization (GRPO) being a widely used training algorithm. However, GRPO-style algorithms still face several challenges in multi-hop search settings. First, correct intermediate steps are often penalized when the final answer is wrong. Second, training is highly unstable, often causing degradation of natural language ability or even catastrophic training collapse. Our analysis attributes these issues to coarse-grained advantage assignment and an imbalance between positive and negative advantages. To address these problems, we propose CalibAdv, an advantage calibration method specifically designed for search agents that enables more accurate and more stable modeling of penalties and rewards. Specifically, CalibAdv leverages the correctness of intermediate steps to downscale excessive negative advantages at a fine-grained level. It then further rebalances positive and negative advantages to improve training stability. Importantly, CalibAdv adopts a lightweight design that calibrates advantages from standard rollout signals, making it simple and easy to deploy. Extensive experiments across three models and seven benchmarks demonstrate that CalibAdv improves both model performance and training stability. Our code is available at this https URL.
>
---
#### [replaced 044] Position: The Turing-Completeness of Autoregressive Transformers Relies Heavily on Context Management
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文探讨了Transformer的图灵完备性，指出其依赖于上下文管理。任务是澄清理论与实际应用的差异，解决对Transformer能力的误解。工作包括定义固定系统设置，分析不同上下文方法的影响。**

- **链接: [https://arxiv.org/pdf/2605.19514](https://arxiv.org/pdf/2605.19514)**

> **作者:** Guanyu Cui; Zhewei Wei; Kun He
>
> **备注:** Accepted to the ICML 2026 Position Paper Track
>
> **摘要:** Many works make the eye-catching claim that Transformers are Turing-complete. However, the literature often conflates two distinct settings: (i) a fixed Transformer system setting, in which a fixed autoregressive Transformer is coupled with a fixed context-management method to process inputs of different lengths step by step, and (ii) a scaling-family setting, in which a family of different models (with increasing context-window length or numerical precision) is used to handle different input lengths. Existing proofs of Transformer Turing-completeness are frequently established in setting (ii), whereas real-world LLM deployment and the standard notion of Turing-completeness correspond more naturally to setting (i). In this paper, we first formalize the fixed-system setting, thereby providing a concrete characterization of how real-world LLMs operate. We then argue that results proved in the scaling-family setting provide theoretically meaningful resource bounds but do not establish Turing-completeness, thereby clarifying a common misinterpretation of existing results. Finally, we show that different context-management methods can yield sharply different computational power, and we advocate the position that context management is a central component that critically determines the computational power of real-world autoregressive Transformers.
>
---
#### [replaced 045] Speaking of Language: Reflections on Metalanguage Research in NLP
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨金属语言在NLP中的研究，分析其四个维度及任务，提出未来研究方向。属于理论分析任务，旨在深化对金属语言的理解与应用。**

- **链接: [https://arxiv.org/pdf/2604.02645](https://arxiv.org/pdf/2604.02645)**

> **作者:** Nathan Schneider; Antonios Anastasopoulos
>
> **备注:** To appear at the Big Picture Workshop at ACL 2026. Camera-ready version
>
> **摘要:** This work aims to shine a spotlight on the topic of metalanguage. We first define metalanguage, link it to NLP and LLMs, and then discuss our two labs' metalanguage-centered efforts. Finally, we discuss four dimensions of metalanguage and metalinguistic tasks, offering a list of understudied future research directions.
>
---
#### [replaced 046] Adaptive Cost-Efficient Evaluation for Reliable Patent Claim Generation
- **分类: cs.CL**

- **简介: 该论文针对专利权利要求验证任务，解决自动化验证中精度与成本的平衡问题。提出ACE框架，通过分阶段评估提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.04295](https://arxiv.org/pdf/2604.04295)**

> **作者:** Yongmin Yoo; Qiongkai Xu; Longbing Cao
>
> **摘要:** Automated patent claim validation demands low error tolerance. However, existing approaches face a rigidity-resource dilemma: lightweight encoders cannot track long-range legal dependencies, while exhaustive LLM verification incurs 4-5X higher overhead at million-claim scale. A naive confidence-based cascade cannot resolve this because binary validity scores fail to distinguish structurally distinct error types which require different reasoning depths. We propose a two-stage framework: Adaptive Cost-efficient Evaluation (ACE), which exploits the categorical structure of patent errors for uncertainty-aware routing. In the first stage, a fine-tuned encoder projects claims into a K+1 distribution over legal error types, whose predictive entropy serves as the routing signal. Claims exceeding an entropy threshold are escalated to the second stage, where an expert LLM executes a schema-constrained Chain-of-Patent-Thought (CoPT) protocol to map claim elements against 35 U.S.C. standards whose schema constraint reduces per-claim latency by 42% while producing legally grounded verdicts. We further present a 40,000-claim dataset ACE-40k with MPEP-grounded annotations, where ACE surpasses competitive baselines including a supervised 70B-parameter LLM while reducing costs by 78%. On real USPTO rejection data, the routing mechanism transfers without re-calibration, reducing inference time by 60% while maintaining competitive recall.
>
---
#### [replaced 047] Structured Agent Distillation for Large Language Model
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在降低大语言模型的部署成本。通过结构化蒸馏方法，将大模型的知识迁移至小模型，保持推理与行动的一致性。**

- **链接: [https://arxiv.org/pdf/2505.13820](https://arxiv.org/pdf/2505.13820)**

> **作者:** Jun Liu; Zhenglun Kong; Peiyan Dong; Changdi Yang; Tianqi Li; Hao Tang; Geng Yuan; Wei Niu; Wenbin Zhang; Pu Zhao; Xue Lin; Dong Huang; Yanzhi Wang
>
> **摘要:** Large language models (LLMs) exhibit strong capabilities as decision-making agents by interleaving reasoning and actions, as seen in ReAct-style frameworks. Yet, their practical deployment is constrained by high inference costs and large model sizes. We propose Structured Agent Distillation, a framework that compresses large LLM-based agents into smaller student models while preserving both reasoning fidelity and action consistency. Unlike standard token-level distillation, our method segments trajectories into [REASON] and [ACT] spans, applying segment-specific losses to align each component with the teacher's behavior. This structure-aware supervision enables compact agents to better replicate the teacher's decision process. Experiments on ALFWorld, HotPotQA-ReAct, and WebShop show that our approach consistently outperforms token-level and imitation learning baselines, achieving significant compression with minimal performance drop. Scaling and ablation results further highlight the importance of span-level alignment for efficient and deployable agents.
>
---
#### [replaced 048] ClinicalAgents: Multi-Agent Orchestration for Clinical Decision Making with Dual-Memory
- **分类: cs.CL**

- **简介: 该论文属于临床决策任务，旨在解决LLM在复杂诊断中推理不足的问题。提出ClinicalAgents框架，采用多智能体和双记忆结构，提升诊断准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2603.26182](https://arxiv.org/pdf/2603.26182)**

> **作者:** Zhuohan Ge; Haoyang Li; Yubo Wang; Nicole Hu; Chen Jason Zhang; Qing Li
>
> **备注:** Accepted to the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2026)
>
> **摘要:** While Large Language Models (LLMs) have demonstrated potential in healthcare, they often struggle with the complex, non-linear reasoning required for accurate clinical diagnosis. Existing methods typically rely on static, linear mappings from symptoms to diagnoses, failing to capture the iterative, hypothesis-driven reasoning inherent in human clinicians. To bridge this gap, we introduce ClinicalAgents, a novel multi-agent framework designed to simulate the cognitive workflow of expert clinicians. Unlike rigid sequential chains, ClinicalAgents employs a dynamic orchestration mechanism modeled as a Monte Carlo Tree Search (MCTS) process. This allows an orchestrator to iteratively generate hypotheses, actively verify evidence, and trigger backtracking when critical information is missing. The foundation of this framework is a Dual-Memory architecture: a mutable working memory that maintains the evolving patient state for context-aware reasoning, and a static experience memory that retrieves clinical guidelines and historical cases via an active feedback loop. Extensive experiments demonstrate that ClinicalAgents achieves the best performance among evaluated baselines, significantly enhancing both diagnostic accuracy and explainability compared to strong single-agent and multi-agent baselines. Our code is released at this https URL.
>
---
#### [replaced 049] AdvJudge-Zero: Binary Decision Flips in LLM-as-a-Judge via Adversarial Control Tokens
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文研究LLM-as-a-Judge系统的二元判断漏洞，通过对抗性控制标记引发判断翻转。任务为提升模型判断的鲁棒性，解决奖励信号易被操纵的问题。工作包括发现攻击方法、提出防御策略并验证效果。**

- **链接: [https://arxiv.org/pdf/2512.17375](https://arxiv.org/pdf/2512.17375)**

> **作者:** Tung-Ling Li; Yuhao Wu; Hongliang Liu
>
> **摘要:** LLM-as-a-Judge systems supply the reward signal in modern RLHF and RLVR pipelines, but their binary verdict reduces to a single linear readout F_gap on one hidden state. We show this readout is shallow enough that short, low-perplexity tokens flip the verdict from "No" to "Yes". These tokens are sampled from the judge's own next-token distribution at the response position, with no manual seed set and no gradient-based optimization. Our procedure, AdvJudge-Zero, reaches $>$90% ensemble false-positive rate on 22 of 24 (model, dataset) cells across six Qwen, Llama, and Gemma judges, versus 54-72% for the prior curated 10-token benchmark, and the discovered surface transfers cross-format to a 70B scalar reward model. The same discovered pool enables a defense: a LoRA fine-tune stratified by a 9-class mechanism taxonomy hardens cross-family generalization where naive sampling on the same pool fails, with mechanism breadth rather than pool size carrying the gain. Under GRPO training, the hardened judge eliminates the reward-collapse failures (false-positive spikes and length collapse) we observe in the unhardened baseline on both MATH and GSM8K at ten seeds per condition. The discovered pool, the mechanism taxonomy, and per-prompt flip records will be released under responsible disclosure.
>
---
#### [replaced 050] What Are We Measuring in NLG? A Meta-Analysis of Evaluation Trends 2020-2025
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成（NLG）评估任务，分析2020-2025年14,171篇论文，指出评估指标使用中的三个问题：指标惯性、映射不清和验证不足，并提出评估检查清单。**

- **链接: [https://arxiv.org/pdf/2601.07648](https://arxiv.org/pdf/2601.07648)**

> **作者:** Jing Yang; Nils Feldhus; Salar Mohtaj; Leonhard Hennig; Qianli Wang; Eleni Metheniti; Sherzod Hakimov; Charlott Jakob; Veronika Solopova; Konrad Rieck; David Schlangen; Sebastian Möller; Vera Schmitt
>
> **备注:** 8 pages
>
> **摘要:** As Natural Language Generation (NLG) dominates modern NLP, scalable evaluation remains a critical bottleneck. Consequently, LLM-as-a-judge (LaaJ) adoption has accelerated rapidly, appearing in more papers than human evaluation in 2025. This pivotal shift motivates a critical analysis of current evaluation practices. Overcoming the limits of rigid keyword filtering and manual review, we employ a multi-LLM information extraction pipeline to gather structured metadata from 14,171 papers across four major NLP conferences (2020-2025). Analyzing 3,334 filtered NLG papers, we identify three systemic challenges. (1) Metric inertia: despite the shift toward open-ended generation, legacy lexical metrics (BLEU, ROUGE) persist as primary indicators, typically used alongside rather than replaced by semantic alternatives. (2) Metric-criteria mapping problem: our paper-level co-occurrence data reveals that general-purpose automatic metrics are applied as broad proxies for quality, without specifying which dimension of text generation they are intended to evaluate. (3) Validation gap: LaaJ has grown rapidly without commensurate human validation (fewer than 8% of papers). Crucially, while LaaJ correlates with aggregate quality, alignment collapses on fine-grained criteria like fluency. To address these gaps, we distill our findings into a minimal Evaluation Checklist to guide metric selection, construct validity, and LaaJ deployment.
>
---
#### [replaced 051] Less is More: Geometric Unlearning for LLMs with Minimal Data Disclosure
- **分类: cs.CL**

- **简介: 该论文属于模型去偏任务，旨在实现对特定内容的有效删除，同时保持模型整体性能。提出Geometric Unlearning方法，通过少量合成数据调整模型隐藏状态，实现高效、低影响的去学习。**

- **链接: [https://arxiv.org/pdf/2605.01735](https://arxiv.org/pdf/2605.01735)**

> **作者:** Chenchen Tan; Xinghao Li; Shujie Cui; Youyang Qu; Cunjian Chen; Longxiang Gao
>
> **备注:** 21 pages, 8 Figures
>
> **摘要:** As large language models (LLMs) are increasingly deployed in real-world systems, they must support post-hoc removal of specific content to meet privacy and governance requirements. This motivates selective unlearning, which suppresses information about a particular entity or topic while preserving the LLM's general utility. However, most existing LLM unlearning methods require access to the original training corpus and rely on output-level refusal tuning or broad gradient updates, creating a tension among unlearning strength, non-target preservation, and data availability. We propose Geometric Unlearning (GU), an approach that operates directly on the model's prompt-conditioned hidden states without access to the original training corpus. Specifically, GU distills a compact, low-rank safe-behavior subspace from a small set of safe reference prompts and uses lightweight anchor-in-context synthetic prompts to trigger localized, projection-based alignment of hidden representations to this safe subspace. A teacher-distillation regularizer on synthetic non-target anchors further reduces collateral drift. Across privacy-oriented unlearning benchmarks (ToFU and UnlearnPII), GU achieves strong target suppression with minimal impact on non-target performance, demonstrating that effective unlearning can be achieved with minimal synthetic data.
>
---
#### [replaced 052] Many Dialects, Many Languages, One Cultural Lens: Evaluating Multilingual VLMs for Bengali Culture Understanding Across Historically Linked Languages and Regional Dialects
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态语言模型评估任务，旨在解决 Bengali 文化在多语言和方言中的理解问题。构建了 BanglaVerse 基准，评估 VLM 在不同语言和方言中的文化理解能力。**

- **链接: [https://arxiv.org/pdf/2603.21165](https://arxiv.org/pdf/2603.21165)**

> **作者:** Nurul Labib Sayeedi; Md. Faiyaz Abdullah Sayeedi; Shubhashis Roy Dipta; Rubaya Tabassum; Ariful Ekraj Hridoy; Mehraj Mahmood; Mahbub E Sobhani; Md. Tarek Hasan; Swakkhar Shatabda
>
> **备注:** this https URL
>
> **摘要:** Bangla culture is richly expressed through region, dialect, history, food, politics, media, and everyday visual life, yet it remains underrepresented in multimodal evaluation. To address this gap, we introduce BanglaVerse, a culturally grounded benchmark for evaluating multilingual vision-language models (VLMs) on Bengali culture across historically linked languages and regional dialects. Built from 1,152 manually curated images across nine domains, the benchmark supports visual question answering and captioning, and is expanded into four languages and five Bangla dialects, yielding ~32.2K artifacts. Our experiments show that evaluating only standard Bangla overestimates true model capability: performance drops under dialectal variation, especially for caption generation, while historically linked languages such as Hindi and Urdu retain some cultural meaning but remain weaker for structured reasoning. Across domains, the main bottleneck is missing cultural knowledge rather than visual grounding alone, with knowledge-intensive categories. These findings position BanglaVerse as a more realistic test bed for measuring culturally grounded multimodal understanding under linguistic variation.
>
---
#### [replaced 053] Auditing medical multi-agent AI reveals risks of false consensus
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于医疗AI评估任务，旨在解决多智能体系统协作过程中的安全问题。通过构建审计框架，分析协作失败模式，提升系统透明度与可靠性。**

- **链接: [https://arxiv.org/pdf/2510.10185](https://arxiv.org/pdf/2510.10185)**

> **作者:** Yinghao Zhu; Lei Gu; Zixiang Wang; Haoran Sang; Dehao Sui; Wen Tang; Lan Mi; Yasha Wang; Junyi Gao; Liang Yao; Tianfan Fu; Ewen Harrison; Lequan Yu; Liantao Ma
>
> **备注:** Code and Data: this https URL
>
> **摘要:** Large language models are increasingly being assembled into medical multi-agent systems that emulate multidisciplinary consultation through specialist roles, peer review and consensus formation. In clinical decision support, however, apparent consensus is not enough. Clinicians also need to know whether agents checked the evidence, addressed disagreement and kept uncertainty visible. Current evaluations largely score final accuracy, leaving the safety of the collaborative process untested. Here we introduce MedAgentAudit, a clinically grounded workflow audit framework for diagnosing and quantifying collaborative failure modes in medical multi-agent systems. From 3,600 execution logs, we derive an expert-validated taxonomy of ten recurrent failures spanning task comprehension, collaborative discussion, and synthesis and decision-making. We then deploy an expert-validated automated auditor as non-interventional probes across 14,400 cases, covering six multi-agent architectures, six medical text and vision datasets, and four large language model settings per modality. Across systems, collaboration yields uneven accuracy gains and frequent process failures. Unsupported observations affect 16.63% of cases and propagate downstream. In discussion, agents repeat initial views in 98.42% of cases rather than re-examining evidence, and fail to activate specialist reasoning in 42.73%. During synthesis, final answers often substitute authority or majority count for evidence checking, showing authority bias in 28.76% (rising from 35.30% to 68.75% across rounds), self-contradiction in 18.53%, contradiction neglect in 5.48% and minority suppression in 5.11%. MedAgentAudit reframes medical AI evaluation from output scoring to process-level safety and accountability, providing a practical foundation for transparent, auditable and clinician-supervised agentic systems in medicine.
>
---
#### [replaced 054] EAGer: Entropy-Aware GEneRation for Adaptive Inference-Time Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出EAGer方法，用于自适应推理时缩放，解决复杂推理任务中计算资源分配不均的问题。通过分析token熵分布，优化计算资源使用，提升性能并减少token消耗。**

- **链接: [https://arxiv.org/pdf/2510.11170](https://arxiv.org/pdf/2510.11170)**

> **作者:** Daniel Scalena; Leonidas Zotos; Elisabetta Fersini; Malvina Nissim; Ahmet Üstün
>
> **摘要:** With the rise of reasoning language models and test-time scaling methods as a paradigm for improving model performance, substantial computation is often required to generate multiple candidate sequences from the same prompt. This enables exploration of different reasoning paths toward the correct solution, however, allocates the same compute budget for each prompt. Grounded on the assumption that different prompts carry different degrees of complexity, and thus different computation needs, we propose EAGer, a training-free generation method that leverages model uncertainty through token-wise entropy distribution to reduce redundant computation and concurrently improve overall performance. EAGer allows branching to multiple reasoning paths only in the presence of high-entropy tokens, and reallocates the saved compute budget to instances where exploration of alternative paths is most needed. We validate EAGer across multiple open-source models on complex reasoning benchmarks, with gains specifically demonstrated on AIME 2025. When target labels are accessible -- as in RLVR training pipelines -- EAGer achieves up to +37% in Pass@k and 59% fewer tokens; in test-time settings it still yields +12% in Pass@k and 64% fewer tokens compared to Full Parallel Sampling.
>
---
#### [replaced 055] Graph Memory Transformer (GMT)
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Graph Memory Transformer（GMT），通过引入记忆图替代Transformer中的FFN子层，解决语言模型结构可解释性问题，属于自然语言处理任务。**

- **链接: [https://arxiv.org/pdf/2604.23862](https://arxiv.org/pdf/2604.23862)**

> **作者:** Nicola Zanarini; Niccolò Ferrari
>
> **备注:** 65 pages, 10 figures, 5 tables. Code available at this https URL
>
> **摘要:** We investigate whether the Feed-Forward Network (FFN) sublayer in a decoder-only transformer can be replaced by an explicit learned memory graph while preserving the surrounding autoregressive architecture. The proposed Graph Memory Transformer (GMT) keeps causal self-attention intact, but replaces the usual per-token FFN transformation with a memory cell that routes token representations over a learned bank of centroids connected by a learned directed transition matrix. In the base GMT v7 instantiation studied here, each of 16 transformer blocks contains 128 centroids, a 128 * 128 edge matrix, gravitational source routing, token-conditioned target selection, and a gated displacement readout. The cell therefore returns movement from an estimated source memory state toward a target memory state, rather than a retrieved value. The resulting model is a fully decoder-only language model with 82.2M trainable parameters and no dense FFN sublayers, compared with a 103.0M-parameter dense GPT-style baseline used in the evaluation. The base v7 model trains stably and exposes centroid usage, transition structure, and source-to-target movement as directly inspectable quantities of the forward computation. It remains behind the larger dense baseline in validation loss and perplexity (3.5995/36.58 vs. 3.2903/26.85), while showing close zero-shot benchmark behavior under the evaluated setting. These results are not intended as a state-of-the-art claim; they support the viability and structural interpretability of replacing dense within-token transformation with graph-mediated memory navigation. Broader scaling, optimized kernels, and more extensive benchmark evaluation are left for subsequent work.
>
---
#### [replaced 056] Persuade Me if You Can: A Framework for Evaluating Persuasion Effectiveness and Susceptibility Among Large Language Models
- **分类: cs.CL; cs.AI; cs.LG; cs.MA**

- **简介: 该论文属于AI安全与伦理任务，旨在评估大语言模型的说服力和易受说服程度。研究提出PMIYC框架，通过自动化对话测试模型在不同场景下的表现，提升AI系统的安全性与可靠性。**

- **链接: [https://arxiv.org/pdf/2503.01829](https://arxiv.org/pdf/2503.01829)**

> **作者:** Nimet Beyza Bozdag; Shuhaib Mehri; Gokhan Tur; Dilek Hakkani-Tür
>
> **备注:** Paper published at the ACM Conference on AI and Agentic Systems 2026
>
> **摘要:** Large Language Models (LLMs) demonstrate persuasive capabilities that rival human-level persuasion. While these capabilities can be used for social good, they also present risks of potential misuse. Beyond the concern of how LLMs persuade others, their own susceptibility to persuasion poses a critical alignment challenge, raising questions about robustness, safety, and adherence to ethical principles. To study these dynamics, we introduce Persuade Me If You Can (PMIYC), an automated framework for evaluating persuasiveness and susceptibility to persuasion in multi-agent interactions. Our framework offers a scalable alternative to the costly and time-intensive human annotation process typically used to study persuasion in LLMs. PMIYC automatically conducts multi-turn conversations between Persuader and Persuadee agents, measuring both the effectiveness of and susceptibility to persuasion. Our comprehensive evaluation spans a diverse set of LLMs and persuasion settings (e.g., subjective and misinformation scenarios). We validate the efficacy of our framework through human evaluations and demonstrate alignment with human assessments from prior studies. Through PMIYC, we find that Llama-3.3-70B and GPT-4o exhibit similar persuasive effectiveness, outperforming Claude 3 Haiku by 30%. However, GPT-4o demonstrates over 50% greater resistance to persuasion for misinformation compared to Llama-3.3-70B. Notably, o4-mini emerges as both an effective persuader, and a resistant persuadee. These findings provide empirical insights into the persuasive dynamics of LLMs and contribute to the development of safer AI systems.
>
---
#### [replaced 057] Atomic Skills are the Prerequisite: When Reinforcement Learning Synthesizes Compositional Reasoning, and When It Only Amplifies
- **分类: cs.AI; cs.CL**

- **简介: 论文探讨RL在组合推理中的作用，解决其是放大还是合成新技能的问题。通过分解为原子技能，发现RL需先掌握基础技能才能有效合成新策略。任务为强化学习与组合推理。**

- **链接: [https://arxiv.org/pdf/2512.01970](https://arxiv.org/pdf/2512.01970)**

> **作者:** Sitao Cheng; Xunjian Yin; Ruiwen Zhou; Yuxuan Li; Xinyi Wang; Liangming Pan; William Yang Wang; Victor Zhong
>
> **备注:** Work in Progress. Code and data are available at this https URL
>
> **摘要:** Does Reinforcement Learning (RL) merely amplify existing skills, or synthesize novel skills? We investigate this question through the lens of Complementary Reasoning: the critical practical capability of integrating internal knowledge with external context, a prerequisite for reliable Continual Learning and Retrieval-Augmented Generation. To avoid pre-training contamination, we construct a controlled semanticsynthetic dataset of biographies and decompose this capability into two atomic skills: Parametric Reasoning (retrieving facts encoded in model weights) and Contextual Reasoning (processing novel in-context information). We present two findings. First, models supervised directly on the composite task reach high accuracy on seen facts and reasoning paths (90%) but collapse on novel facts and reasoning paths (18%), indicating that Supervised Fine-Tuning (SFT) relies on rote memorization rather than genuine skill integration. Second, RL bridges this generalization gap, acting as a skill synthesizer rather than a mere amplifier--but only under a strict prerequisite: it synthesizes new composite strategies only when the base model has first mastered the independent atomic skills via SFT. These results suggest that decoupled atomic training followed by RL offers a scalable path to complex novel reasoning.
>
---
#### [replaced 058] Test-Time Compute for Dense Retrieval: Agentic Program Generation with Frozen Embedding Models
- **分类: cs.LG; cs.CL; cs.IR**

- **简介: 该论文研究测试时计算对密集检索的提升，解决小嵌入模型性能不足的问题。通过代理程序搜索，发现多个优化方案，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2605.11374](https://arxiv.org/pdf/2605.11374)**

> **作者:** Han Xiao
>
> **备注:** 17 pages, 4 figures, 5 tables
>
> **摘要:** Test-time compute is widely believed to benefit only large reasoning models. We show it also helps small embedding models. Since modern embedding models are distilled from LLM backbones, a frozen encoder should benefit from extra inference compute without retraining. An agentic program-search loop explores 144 candidate programs over a frozen encoder API and produces twelve Pareto-optimal programs that trade extra inference compute for retrieval quality. The search independently rediscovers Rocchio pseudo-relevance feedback, ColBERT-style MaxSim at sentence granularity, reciprocal rank fusion, and the Fisher linear discriminant, all without trainable parameters or external models. Every frontier program improves nDCG@10 over the frozen baseline on all 14 tasks used during program search. Generalization is validated separately: a single fixed program, selected from the discovery frontier before any held-out evaluation, improves nDCG@10 on 61% of model-task pairs across three unseen encoder families and nineteen held-out retrieval tasks, without any per-task selection.
>
---
#### [replaced 059] Why Gaussian Diffusion Models Fail on Discrete Data and How to Prevent It?
- **分类: cs.CL**

- **简介: 该论文研究扩散模型在离散数据上的生成问题，分析其失败原因并提出改进方法。**

- **链接: [https://arxiv.org/pdf/2604.02028](https://arxiv.org/pdf/2604.02028)**

> **作者:** Alexander Shabalin; Simon Elistratov; Viacheslav Meshchaninov; Ildus Sadrtdinov; Dmitry Vetrov
>
> **摘要:** Diffusion models have become a standard approach for generative modeling in continuous domains, yet their application to discrete data remains challenging. We investigate why Gaussian diffusion models with the DDPM solver struggle to sample from discrete distributions that are represented as a mixture of delta-distributions in the continuous space. Using a toy Random Hierarchy Model, we identify a critical sampling interval in which the density of noisified data becomes multimodal. In this regime, DDPM occasionally enters low-density regions between modes producing out-of-distribution inputs for the model and degrading sample quality. We show that existing heuristics, including self-conditioning and a solver we term q-sampling, help alleviate this issue. Furthermore, we demonstrate that combining self-conditioning with switching from DDPM to q-sampling within the critical interval improves generation quality on real data. We validate these findings across conditional and unconditional tasks in multiple domains, including text, programming code, and proteins.
>
---
#### [replaced 060] Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态语言模型任务，解决细粒度视觉理解问题。提出Vision-OPD框架，通过自蒸馏提升模型对关键视觉证据的聚焦能力。**

- **链接: [https://arxiv.org/pdf/2605.18740](https://arxiv.org/pdf/2605.18740)**

> **作者:** Qianhao Yuan; Jie Lou; Xing Yu; Hongyu Lin; Le Sun; Xianpei Han; Yaojie Lu
>
> **备注:** Project page: this https URL
>
> **摘要:** Multimodal Large Language Models (MLLMs) still struggle with fine-grained visual understanding, where answers often depend on small but decisive evidence in the full image. We observe a regional-to-global perception gap: the same MLLM answers fine-grained questions more accurately when conditioned on evidence-centered crops than on the corresponding full images, suggesting that many failures stem from difficulty to focus on relevant evidence rather than insufficient local recognition ability. Motivated by this observation, we propose Vision-OPD (Vision On-Policy Distillation), a regional-to-global self-distillation framework that transfers the model's own privileged regional perception to its full-image policy. Vision-OPD instantiates two conditional policies from the same MLLM: a crop-conditioned teacher and a full-image-conditioned student. The student generates on-policy rollouts, and Vision-OPD minimizes token-level divergence between the teacher and student next-token distributions along these rollouts. This enables the model to internalize the benefit of visual zooming without external teacher models, ground-truth labels, reward verifiers, or inference-time tool use. Experiments on multiple fine-grained visual understanding benchmarks show that Vision-OPD models achieve competitive or superior performance against much larger open-source, closed-source, and "Thinking-with-Images" agentic models.
>
---
#### [replaced 061] Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于个性化网络代理任务，旨在解决用户意图模糊时的上下文推理问题。提出Persona2Web基准，通过用户历史提升代理的个性化能力。**

- **链接: [https://arxiv.org/pdf/2602.17003](https://arxiv.org/pdf/2602.17003)**

> **作者:** Serin Kim; Sangam Lee; Dongha Lee
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Large language models have advanced web agents, yet current agents lack personalization capabilities. Since users rarely specify every detail of their intent, practical web agents must be able to interpret ambiguous queries by inferring user preferences and contexts. To address this challenge, we present Persona2Web, the first benchmark for evaluating personalized web agents on the real open web, built upon the clarify-to-personalize principle, which requires agents to resolve ambiguity based on user history rather than relying on explicit instructions. Persona2Web consists of: (1) user histories that reveal preferences implicitly over long time spans, (2) ambiguous queries that require agents to infer implicit user preferences, and (3) a reasoning-aware evaluation framework that enables fine-grained assessment of personalization. We conduct extensive experiments across various agent architectures, backbone models, history access schemes, and queries with varying ambiguity levels, revealing key challenges in personalized web agent behavior. For reproducibility, our codes and datasets are publicly available at this https URL.
>
---
#### [replaced 062] An Effective-Rank Audit of Alignment-Induced Activation Shifts: Confound Control, Constructive Calibration, and Limits
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文研究大语言模型对齐引起的激活变化，通过有效秩分析评估其安全性和鲁棒性，解决对齐机制的诊断与优化问题。**

- **链接: [https://arxiv.org/pdf/2605.24583](https://arxiv.org/pdf/2605.24583)**

> **作者:** Yuki Nakamura
>
> **备注:** 18 pages, 1 figure, 21 tables. Code, data, and an immutable Zenodo archive are available at this https URL (DOI: https://doi.org/10.5281/zenodo.20341445)
>
> **摘要:** We audit alignment-induced shifts in residual-stream activations of three open-weight instruction-tuned LLMs (Llama-3.1-8B-Instruct, Gemma-2-9B-it, Qwen-2.5-7B-Instruct) using the effective rank of the alignment modification matrix on safety-relevant inputs, rho_eps := rank_eps(M_Ds)/d, which formalizes the single-refusal-direction observation of Arditi et al. (2024) as a continuous quantity. The paper has three contributions. (1) Confound-controlled measurement: a four-variant decomposition (M_naive, M_template, M_aligned, M_DiD) separates chat-template formatting, alignment-stage shift, and the refusal-mediating direction, and recovers the Arditi refusal direction on M_DiD at |cos| in {0.77, 0.86, 0.50} (Llama/Gemma/Qwen); chat-template-controlled rho_eps is {0.0029, 0.0048, 0.0044}, and the centered SVD residual is 4-7x larger. (2) Constructive calibration on a 3-layer MLP across rho_eps in {0.008, 0.17, 0.33, 0.40} exhibits a sweet-spot vs. brittle distinction: mild rank-maximization (lambda=5) buys ablation robustness, while strong regularization at the same nominal rho_eps (lambda=50) does not. rho_eps is a diagnostic for fragility, not a target whose mechanical inflation buys robustness. (3) Limits of rank-based diagnostics: (a) not safety-specific (LRH baseline is 2-3x the safety value); (b) SVD principal ordering does not match causal ordering (Llama u_2 inert despite ranking second; cumulative ablation non-monotone at k=5); (c) the spectral-gap hypothesis required to upgrade the O(rho_eps * d) achievability bound to a matching Mirsky-route lower bound fails empirically (1/90 Llama layer-reference pairs, 0/36 MLP combinations) and structurally (kappa_lb <= 2/(eps * r)). The matching lower bound remains an open problem.
>
---
#### [replaced 063] SelfJudge: Faster Speculative Decoding via Self-Supervised Judge Verification
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型加速任务，解决传统方法依赖人工标注的问题。提出SelfJudge，通过自监督训练验证器，提升推理速度与准确性。**

- **链接: [https://arxiv.org/pdf/2510.02329](https://arxiv.org/pdf/2510.02329)**

> **作者:** Kanghoon Yoon; Minsub Kim; Sungjae Lee; Joonhyung Lee; Sunghyeon Woo; Yeonjun In; Se Jung Kwon; Chanyoung Park; Dongsoo Lee
>
> **摘要:** Speculative decoding accelerates LLM inference by verifying candidate tokens from a draft model against a larger target model. Recent judge decoding boosts this process by relaxing verification criteria by accepting draft tokens that may exhibit minor discrepancies from target model output, but existing methods are restricted by their reliance on human annotations or tasks with verifiable ground truths, limiting generalizability across diverse NLP tasks. We propose SelfJudge, which trains judge verifiers via self-supervision of the target model. Our method measures semantic preservation by assessing whether token-substituted responses preserve the meaning of original responses, enabling automatic verifier training across diverse NLP tasks. Our experiments show SelfJudge achieves superior inference-accuracy trade-offs than judge decoding baselines, offering a broadly applicable solution for faster LLM inference.
>
---
#### [replaced 064] Are We Truly Innovating? A Qualitative and Quantitative Study of Originality in AI Research Papers
- **分类: cs.CL**

- **简介: 该论文属于AI研究原创性评估任务，旨在解决同行评审中原创性判断不一致的问题。通过分析大量评审报告，构建了原创性评价框架，并评估了大语言模型在该任务上的表现。**

- **链接: [https://arxiv.org/pdf/2602.06054](https://arxiv.org/pdf/2602.06054)**

> **作者:** Abeer Mostafa; Thi Huyen Nguyen; Zahra Ahmadi
>
> **摘要:** Assessing originality in AI research is arguably the most consequential yet least reliable step in peer review. Reviewer judgments of originality remain opaque, inconsistent, and dependent on comparisons to prior work that are often incomplete. In this paper, we present a large-scale, data-driven qualitative and quantitative analysis of research originality based on over 100,000 peer-review reports from leading AI venues, spanning a period of rapid growth in the field. Leveraging structured, semantically retrieved prior work and signals embedded in expert reviewer assessments, we systematically characterize how originality is perceived in practice and identify the key dimensions that most strongly influence novelty judgments. Our analysis yields a fine-grained, evidence-based framework that equips both authors and reviewers with actionable insights into how originality is evaluated. In addition, we evaluate the reliability of current large language model (LLM) agents in assessing originality. We find that these models tend to systematically overestimate novelty and struggle to detect conceptual plagiarism, particularly in the presence of paraphrasing. We release our dataset, trained models, and code at: this https URL.
>
---
#### [replaced 065] Adaptive Teacher Exposure for Self-Distillation in LLM Reasoning
- **分类: cs.AI; cs.CL; cs.LO**

- **简介: 该论文属于大模型推理任务，解决自蒸馏中教师暴露不匹配问题。提出ATESD方法，动态调整教师暴露比例以提升学生性能。**

- **链接: [https://arxiv.org/pdf/2605.11458](https://arxiv.org/pdf/2605.11458)**

> **作者:** Zihao Han; Tiangang Zhang; Huaibin Wang; Yilun Sun
>
> **备注:** Withdrawn by the authors pending completion of institutional compliance review and approval. The authors will determine the appropriate next steps after the review process is complete
>
> **摘要:** On-policy self-distillation has become a strong recipe for LLM reasoning, where a privileged teacher supervises the student's own rollouts while conditioning on the reference solution. A design choice shared by nearly all such methods, however, has gone unquestioned: the teacher always sees the full reference reasoning. We argue that this default itself is part of the problem and identify a teacher-side exposure mismatch: when the teacher conditions on reasoning far beyond the student's current competence, the resulting token targets become too strong to absorb. A controlled fixed-exposure sweep makes this concrete on two fronts: 1) full exposure is not reliably the best choice, and 2) student-teacher mismatch grows monotonically as the teacher sees more privileged reasoning. This motivates treating teacher exposure not as a fixed hyperparameter but as a learnable training-time control variable. We therefore propose Adaptive Teacher Exposure for Self-Distillation (ATESD). ATESD models the reveal ratio with a lightweight Beta-policy controller conditioned on compact training-state statistics, and uses one sampled exposure for a short hold window of student updates. To make this exposure controller learnable, we optimize it with a discounted learning-progress reward that scores each held decision by its effect on the student's future improvement rather than its immediate loss change, addressing the delayed credit assignment induced by on-policy distillation. Experiments on AIME 24, AIME 25, and HMMT 25 across Qwen3-{1.7B, 4B, 8B} show that ATESD consistently outperforms competitive self-distillation and RL baselines, improving over OPSD by +0.95, +2.05, and +2.33 Average@12 points respectively, and establishing adaptive teacher exposure as an effective new axis for reasoning self-distillation.
>
---
#### [replaced 066] iPOE: Interpretable Prompt Optimization via Explanations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决提示优化的透明性问题。通过自动生成解释指导提示优化，提升模型决策可解释性，使非专业用户也能参与。**

- **链接: [https://arxiv.org/pdf/2605.18113](https://arxiv.org/pdf/2605.18113)**

> **作者:** Jiahui Li; Yarik Menchaca Resendiz; Sean Papay; Roman Klinger
>
> **摘要:** Prompt optimization has often been framed as a discrete search problem to find high-performing and robust instructions for an LLM. However, the search result might not make it transparent why and where specific prompt changes lead to performance gains. This is in contrast to how humans are instructed for annotation tasks. Here, researchers carefully design annotation guidelines, leading to enhanced annotation consistency. Our paper aims at joining these two approaches and introduces iPOE, a novel interpretable prompt optimization strategy via explanations. We guide the prompt optimization process by automatically created guidelines from explanations of annotation decisions (either automatically generated or from humans). This set of guidelines is furthermore optimized by as series of operations, including removing, adding, shuffling, and merging. The resulting prompt includes guidelines that instruct the annotation, making the decision process of the LLM and the optimization transparent. It therefore supports also laypeople in the area of prompt optimization, particularly in challenging domains requiring expertise. In our experiments on four datasets, we find that iPOE can improves over the evaluated baselines by up to 39% and LLM explanations can replace human explanations in the proposed method. Moreover, our interpretability validation study demonstrates that humans and LLMs can substantially agree on which guidelines contribute to their annotations, achieving a Cohen's kappa score of up to 0.65.
>
---
#### [replaced 067] A tree interpretation of arc standard dependency derivation
- **分类: cs.CL**

- **简介: 该论文研究依赖句法解析任务，解决如何将弧标准推导转化为有序树的问题。通过定义确定性树更新，证明项目性依赖树可唯一映射为有序树，为非项目性输入提供伪项目提升方法。**

- **链接: [https://arxiv.org/pdf/2603.27459](https://arxiv.org/pdf/2603.27459)**

> **作者:** Zihao Huang; Ai Ka Lee; Jungyeul Park
>
> **摘要:** Arc-standard derivations over projective dependency trees can be interpreted as the incremental construction of lexicalized ordered trees with contiguous yields. Each \textsc{shift}, \textsc{leftarc}, and \textsc{rightarc} transition corresponds to a deterministic tree update, and the resulting ordered tree uniquely determines the dependency arcs introduced by the derivation. We show that this representation is not an arbitrary encoding: a single-headed dependency tree admits such a contiguous ordered representation if and only if it is projective. The proposal is therefore derivational rather than conversion-based, since the ordered object is defined over the transition sequence itself rather than obtained by transforming a completed dependency graph. This gives a tree-theoretic interpretation of arc-standard parsing, in which projective dependency derivations implicitly construct recoverable constituency-style ordered trees. For non-projective inputs, the interpretation can be used through pseudo-projective lifting and inverse decoding. A small implementation study confirms that the mapped derivations are executable in an existing neural transition-based parser.
>
---
#### [replaced 068] EVADE-Bench: Multimodal Benchmark for Evaluating and Enhancing Evasive Content Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于内容检测任务，旨在解决隐蔽违规内容的识别问题。提出EVADE-Bench基准，评估模型在电商场景下的多模态 evasion 内容检测能力，并探索提升方法。**

- **链接: [https://arxiv.org/pdf/2505.17654](https://arxiv.org/pdf/2505.17654)**

> **作者:** Ancheng Xu; Zhihao Yang; Jingpeng Li; Guanghu Yuan; Longze Chen; Liang Yan; Jiehui Zhou; Zhen Qin; Hengyu Chang; Yukun Chen; Hamid Alinejad-Rokny; Min Yang
>
> **备注:** SIGIR 2026
>
> **摘要:** E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content, which refers to inputs that have been deliberately modified through techniques such as word splitting, euphemistic language, or image cropping to conceal policy violations while still conveying prohibited claims. Crucially, detecting such content requires a model to simultaneously master two capabilities: accurately comprehending complex rules, and correctly inferring the true intent behind deliberately obfuscated multimodal inputs. While prior work has separately explored LLM reasoning over complex rules and LLM-based detection of evasive content, no existing benchmark combines both within a unified evaluation framework. This gap is particularly consequential in e-commerce, where accurate moderation demands that both capabilities operate in concert. To address this gap, we introduce EVADE-Bench, the first expert-curated Chinese multimodal benchmark specifically designed to evaluate LLMs and VLMs on evasive content detection in real-world e-commerce scenarios. Our comprehensive evaluation of 26 open- and closed-source LLMs and VLMs reveals that even state-of-the-art models frequently misclassify evasive samples. We further demonstrate that clearer rule categorization significantly improves model prediction consistency and reduces false predictions, highlighting the critical role of benchmark design in enabling reliable evaluation. To explore paths for performance improvement, we investigate the feasibility of multi-agent decomposition for multimodal reasoning, wherein visual description and logical inference are decoupled into separate agents, and find that this strategy yields notable accuracy gains.
>
---
#### [replaced 069] Attention Projection Mixing with Exogenous Anchors
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决跨层注意力投影的结构冲突问题。提出ExoFormer，通过外部锚点提升模型性能，实现更高效的数据利用和准确率提升。**

- **链接: [https://arxiv.org/pdf/2601.08131](https://arxiv.org/pdf/2601.08131)**

> **作者:** Jonathan Su
>
> **摘要:** Cross-layer reuse of early attention projections can improve optimization and data efficiency, but it creates a structural conflict: the first layer must simultaneously act as a stable, reusable anchor for all deeper layers and as an effective computational block. We demonstrate that this tension constrains the performance of internal-anchor designs. We propose ExoFormer, which resolves the conflict by learning exogenous anchor projections outside the sequential layer stack. We introduce a unified normalized mixing framework that mixes queries, keys, values, and gate logits using learnable coefficients (exploring coefficient granularities: elementwise, headwise, and scalar), and we show that normalizing anchor sources is key to stable reuse. ExoFormer variants consistently outperform their internal-anchor counterparts, and the dynamic variant yields 1.5x downstream accuracy points while matching validation loss using 1.5x fewer tokens than Gated Attention. We explain this efficacy via an Offloading Hypothesis: external anchors preserve essential token identity, allowing layers to specialize exclusively in feature transformation. We release code and models to facilitate future research.
>
---
#### [replaced 070] Large Language Models Approach Expert Pedagogical Quality in Math Tutoring but Differ in Instructional and Linguistic Profiles
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于教育技术领域，研究LLMs在数学辅导中的教学质量。旨在评估LLMs与人类导师的教学策略和语言特征差异，通过分析对话数据发现LLMs虽接近专家水平，但在教学方法上存在系统性差异。**

- **链接: [https://arxiv.org/pdf/2512.20780](https://arxiv.org/pdf/2512.20780)**

> **作者:** Ramatu Oiza Abdulsalam; Segun Aroyehun
>
> **摘要:** Recent work has explored the use of large language models (LLMs) to generate tutoring responses in mathematics, yet it remains unclear how closely their instructional behavior aligns with expert human practice. We analyze a dataset of math remediation dialogues in which expert tutors, novice tutors, and seven LLMs of varying sizes, comprising both open-weight and commercial models, respond to the same student errors. We examine instructional strategies and linguistic characteristics of tutoring responses, including uptake (restating and revoicing), pressing for accuracy and reasoning, lexical diversity, readability, politeness, and agency. We find that expert tutors produce higher-quality responses than novices, and that larger LLMs generally receive higher pedagogical quality ratings than smaller models, approaching expert performance on average. However, LLMs exhibit systematic differences in their instructional profiles: they underuse discursive strategies characteristic of expert tutors while generating longer, more lexically diverse, and more polite responses. Regression analyses show that pressing for accuracy and reasoning, restating and revoicing, and lexical diversity, are positively associated with perceived pedagogical quality, whereas higher levels of agentic and polite language are negatively associated. These findings highlight the importance of analyzing instructional strategies and linguistic characteristics when evaluating tutoring responses across human tutors and intelligent tutoring systems.
>
---
#### [replaced 071] SkillSafetyBench: Evaluating Agent Safety under Skill-Facing Attack Surfaces
- **分类: cs.CR; cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文属于安全评估任务，旨在解决技能引导下的代理安全问题。通过构建基准测试，评估不同攻击场景下的安全风险与失败模式。**

- **链接: [https://arxiv.org/pdf/2605.12015](https://arxiv.org/pdf/2605.12015)**

> **作者:** Chang Jin; An Wang; Zeming Wei; Kai Wang; Biaojie Zeng; Qiaosheng Zhang; Chao Yang; Jingjing Qu; Xia Hu; Xingcheng Xu
>
> **摘要:** Reusable skills are becoming a common interface for extending large language model agents, packaging procedural guidance with access to files, tools, memory, and execution environments. However, this modularity introduces attack surfaces that are largely missed by existing safety evaluations: even when the user request is benign, unsafe influence may reside in skill guidance, local artifacts, or execution-environment files that steer the agent toward unsafe actions. We present SkillSafetyBench, a runnable benchmark for evaluating such skill-mediated safety failures. SkillSafetyBench includes 155 adversarial cases across 47 tasks, 6 risk domains, and 30 safety categories, each evaluated with a case-specific rule-based verifier. Experiments with multiple CLI agents and model backends show that non-user attacks can consistently induce unsafe behavior, with distinct failure patterns across domains, attack methods, and scaffold-model pairings. Our findings suggest that agent safety depends not only on model-level alignment, but also on how agents interpret skills, trust workflow context, and act through executable environments.
>
---
#### [replaced 072] xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决长上下文大模型的KV缓存内存过高问题。提出xKV方法通过奇异向量对齐压缩KV缓存，降低内存和推理延迟。**

- **链接: [https://arxiv.org/pdf/2503.18893](https://arxiv.org/pdf/2503.18893)**

> **作者:** Chi-Chih Chang; Wei-Cheng Lin; Chien-Yu Lin; Hung-Yueh Chiang; Yash Akhauri; Xilai Dai; Huiqiang Jiang; Yucheng Li; Luis Ceze; Kai-Chiang Wu; Mohamed S. Abdelfattah
>
> **备注:** ICML 2026
>
> **摘要:** Long-context Large Language Models (LLMs) enable powerful applications but incur high memory costs due to the key-value states (KV-Cache). Recent studies attempt to share KV-Cache across layers, but these approaches either require expensive pretraining or rely on per-token cross-layer cosine similarity that is often limited in practice. We show, via Centered Kernel Alignment (CKA), that the dominant singular vectors of KV-Cache are well aligned across layers. Motivated by this observation, we propose xKV, a post-training compression method that jointly factorizes grouped-layer KV-Cache into a shared low-rank subspace, substantially reducing KV-Cache memory. Across widely used LLMs, xKV achieves up to 8x KV-Cache compression while preserving accuracy on long-context tasks and in multi-turn settings. To further improve efficiency, we introduce Selective Reconstruction (SR) at decode time. Combined with SR, xKV achieves up to 4.23x end-to-end speedup over the full attention baseline, and surpasses notable baselines with 30% higher throughput under a similar accuracy level. Overall, xKV provides a plug-and-play approach to reduce both memory and latency for long-context LLM inference. Our code is publicly available at: this https URL.
>
---
#### [replaced 073] Are VLMs Seeing or Just Saying? Uncovering the Illusion of Visual Re-examination
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型研究，旨在解决VLM在声称进行视觉复核时是否真的“看见”问题。通过图像替换实验，发现模型多为“说”而非“看”，揭示其视觉理解的局限性。**

- **链接: [https://arxiv.org/pdf/2605.15864](https://arxiv.org/pdf/2605.15864)**

> **作者:** Chufan Shi; Cheng Yang; Yaokang Wu; Linghao Jin; Bo Shui; Taylor Berg-Kirkpatrick; Xuezhe Ma
>
> **备注:** ICML 2026 Oral
>
> **摘要:** Vision-Language Models (VLMs) often produce self-reflective statements like "let me check the figure again" during reasoning. Do such statements trigger genuine visual re-examination, or are they merely learned textual patterns? We investigate this via VisualSwap, an image-swap probing framework: after a model reasons over an image, we replace it with a visually similar but semantically different one and test whether the model notices. We introduce VS-Bench, 800 image pairs curated from MathVista, MathVerse, MathVision, and MMMU-Pro. Experiments on Qwen3-VL, Kimi-VL, and ERNIE-VL reveal a striking failure: models overwhelmingly miss the swap, with accuracy dropping by up to 60%. Counterintuitively, thinking models are nearly 3x more vulnerable than their instructed counterparts, and scaling offers no mitigation. Multi-turn user instructions restore visual grounding, but self-generated reflective statements during continuous generation do not. Attention analysis explains why: user instructions substantially elevate attention to visual tokens, whereas self-reflection does not. Current VLMs tend to say rather than actually see when claiming to perform visual re-examination. Our code and dataset are available at the project page: this https URL
>
---
#### [replaced 074] Camellia: Benchmarking Cultural Biases in LLMs for Asian Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文化偏 bias 评估任务，旨在解决多语言大模型在亚洲语言中的文化偏见问题。工作包括构建Camellia基准，评估四款模型在三个任务中的表现。**

- **链接: [https://arxiv.org/pdf/2510.05291](https://arxiv.org/pdf/2510.05291)**

> **作者:** Tarek Naous; Anagha Savit; Carlos Rafael Catalan; Geyang Guo; Jaehyeok Lee; Kyungdon Lee; Lheane Marie Dizon; Mengyu Ye; Neel Kothari; Sahajpreet Singh; Sarah Masud; Tanish Patwa; Trung Thanh Tran; Zohaib Khan; Alan Ritter; Tanmoy Chakraborty; Yuki Arase; Keisuke Sakaguchi; JinYeong Bak; Wei Xu
>
> **摘要:** As Large Language Models (LLMs) develop stronger multilingual capabilities, their sensitivity to culturally diverse entities becomes increasingly important. Prior work by Naous et al. (2024) has shown that LLMs often favor Western-associated entities in Arabic. Due to the lack of entity-centric multilingual benchmarks, it remains unclear if such biases also manifest in various non-Western languages. In this paper, we introduce Camellia, a benchmark for evaluating entity-centric cultural biases in nine Asian languages, spanning six Asian cultures. Camellia includes 19,530 manually annotated entities associated with the covered Asian or Western cultures, as well as 2,173 masked contexts for these entities derived from social media posts. Using Camellia, we evaluate cultural biases in four recent multilingual LLMs across three tasks: cultural context adaptation, sentiment association, and entity extractive QA. Our analyses show that LLMs struggle with cultural adaptation across these languages, with performance differing across models developed in different regions. We further observe that different LLM families can hold distinct biases, reflected in the ways they link cultures to particular sentiments. Lastly, we find that LLMs can struggle with context understanding in some Asian languages, creating performance gaps between cultures in entity extraction.
>
---
#### [replaced 075] CodeGENCAT: Generative Computerized Adaptive Testing for Open-ended Coding Problems
- **分类: cs.CL**

- **简介: 该论文提出CodeGENCAT，一种基于生成模型的自适应测试框架，用于开放性编程题。解决传统CAT忽略学生代码响应信息的问题，通过生成代码响应优化题目选择。**

- **链接: [https://arxiv.org/pdf/2602.20020](https://arxiv.org/pdf/2602.20020)**

> **作者:** Wanyong Feng; Alexander Scarlatos; Ruochen Sun; Andrew Lan
>
> **备注:** 23 pages, 2 figures
>
> **摘要:** Existing Computerized Adaptive Testing (CAT) frameworks typically select questions based on the predicted likelihood that the student will answer correctly. This design ignores information contained in students' open-ended responses, especially in domains such as programming education, where code structures and bugs contain rich information on student knowledge. In this work, we propose \textbf{Code} \textbf{GEN}erative \textbf{CAT} (\textbf{CodeGENCAT}), a generative CAT framework that selects questions using predicted student code responses. First, we develop a Generative Item Response Theory (GIRT) model that generates code responses conditioned on estimated student knowledge, trained with supervised fine-tuning followed by direct preference optimization for knowledge-response alignment. Second, we introduce three question-selection algorithms that measure uncertainty, coding style diversity, and information from predicted student code responses. Experiments on two real-world programming education datasets show that CodeGENCAT outperforms all CAT baselines, achieving an AUC improvement of up to 4.32\% over the strongest baseline in the early stages of adaptive testing.
>
---
#### [replaced 076] Formula-One Prompting: A Composable Equation-First Prefix for Applied Mathematics
- **分类: cs.CL**

- **简介: 该论文提出Formula-One Prompting（F-1），用于解决应用数学问题。通过先提取控制方程，再选择求解方式，提升模型性能。任务为数学推理，解决传统提示方法未充分激发方程表达的问题。**

- **链接: [https://arxiv.org/pdf/2601.19302](https://arxiv.org/pdf/2601.19302)**

> **作者:** Natapong Nitarach; Pittawat Taveekitworachai; Kunat Pipatanakul
>
> **摘要:** This paper introduces Formula Prompting (FP) and Formula-One Prompting (F-1), two single-call methods that elicit governing equations before solving applied-math problems. Chain-of-Thought (CoT) and Program-of-Thought (PoT) prompting improve mathematical reasoning by eliciting reasoning traces or code-like structures learned during pretraining. This suggests a diagnostic question: which useful pretraining patterns remain under-elicited? Using infini-gram-mini, we scan 81.7 trillion pretraining tokens and find that, in curated corpora such as DataComp-LM, equation-centered language appears 121x more often than code and 3.79x more often than step-by-step narration, yet standard prompting methods do not explicitly elicit equation formulation. FP asks the model to formalize a problem's governing equations before solving; F-1 extends FP with a composable Phase 2 that selects Direct, CoT, or PoT-style solving in the same call. Across five reasoning models and four applied-math benchmarks (finance, physics, cryptography, competition math), F-1 outperforms CoT by 5.76 pp and PoT by 8.42 pp on average, with the largest gain of 13.30 pp on FinanceMath, while topping the accuracy-token efficiency frontier at only 68 prompt tokens of overhead. Variant ablations identify the equation-formalization prefix, not the strategy menu, as the primary driver: adding CoT or PoT on top of the prefix yields no further gain, and 73.3% of remaining failures occur downstream of a correct Phase-1 equation.
>
---
#### [replaced 077] JMedEthicBench: A Multi-Turn Conversational Benchmark for Evaluating Medical Safety in Japanese Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗安全评估任务，旨在解决LLM在日语医疗对话中的安全问题。构建了首个多轮对话基准JMedEthicBench，测试模型安全性并发现多轮交互中的安全下降现象。**

- **链接: [https://arxiv.org/pdf/2601.01627](https://arxiv.org/pdf/2601.01627)**

> **作者:** Junyu Liu; Zirui Li; Qian Niu; Zequn Zhang; Yue Xun; Wenlong Hou; Shujun Wang; Yusuke Iwasawa; Yutaka Matsuo; Kan Hatakeyama-Sato
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed in healthcare field, it becomes essential to carefully evaluate their medical safety before clinical use. However, existing safety benchmarks remain predominantly English-centric, and test with only single-turn prompts despite multi-turn clinical consultations. To address these gaps, we introduce JMedEthicBench, the first multi-turn conversational benchmark for evaluating medical safety of LLMs for Japanese healthcare. Our benchmark is based on 67 guidelines from the Japan Medical Association and contains over 50,000 adversarial conversations generated using seven automatically discovered jailbreak strategies. Using a dual-LLM scoring protocol, we evaluate 27 models and find that commercial models maintain robust safety while medical-specialized models exhibit increased vulnerability. Furthermore, safety scores decline significantly across conversation turns (median: 9.5 to 5.0, $p < 0.001$). Cross-lingual evaluation on both Japanese and English versions of our benchmark reveals that medical model vulnerabilities persist across languages, indicating inherent alignment limitations rather than language-specific factors. These findings suggest that domain-specific fine-tuning may accidentally weaken safety mechanisms and that multi-turn interactions represent a distinct threat surface requiring dedicated alignment strategies.
>
---
#### [replaced 078] Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文属于多智能体系统安全研究，旨在检测和审计智能体间的共谋行为。提出Colosseum框架，通过分析行动与沟通行为评估共谋，发现模型易产生隐性共谋。**

- **链接: [https://arxiv.org/pdf/2602.15198](https://arxiv.org/pdf/2602.15198)**

> **作者:** Mason Nakamura; Abhinav Kumar; Saswat Das; Sahar Abdelnabi; Saaduddin Mahmud; Ferdinando Fioretto; Shlomo Zilberstein; Eugene Bagdasarian
>
> **摘要:** Multi-agent systems, where LLM agents communicate through free-form language, enable sophisticated coordination for solving complex cooperative tasks. This surfaces a unique safety problem when a group of agents forms a coalition and colludes to pursue secondary goals and degrade the joint objective. In this paper, we present Colosseum, a framework for auditing LLM agents' collusive behavior in multi-agent settings. We ground how agents cooperate through a formal multi-agent decision-making framework and measure action-based collusive behavior in actions via regret relative to the cooperative optimum and compare it with communication-based collusive behavior. Colosseum enables audits of LLM agents for collusion under benign settings, different coalition objectives, persuasion tactics, and network topologies. We then introduce a new behavioral probe by creating secret communication channels between agents, showing that most out-of-the-box models exhibit a propensity to collude under this probe, which we term emergent collusion. Furthermore, we discover ``collusion on paper'' when agents plan to collude in text but often pick non-collusive actions. Colosseum provides a new way to audit collusion in cooperative multi-agent systems while presenting observations about how collusion emerges, what affects collusion efficacy, and which strategies may mitigate it.
>
---
#### [replaced 079] Exploration of Perceptual Speech Features for Clinical Decision-Support in Mental Health Care
- **分类: cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于心理健康评估任务，旨在通过语音特征分析支持临床决策。研究提取了语音的声学和语言特征，结合机器学习方法，探索其与抑郁、焦虑和ADHD症状的关系。**

- **链接: [https://arxiv.org/pdf/2605.24678](https://arxiv.org/pdf/2605.24678)**

> **作者:** Vassilis Lyberatos; Edmund G. Dervakos; Eleni Adamidi; Athanasios Voulodimos; Giorgos Stamou
>
> **备注:** Accepted to CLPsych 2026, part of ACL 2026
>
> **摘要:** Speech and language technologies offer valuable opportunities for supporting mental health assessment through objective and interpretable cues. We present a systematic feature-based analysis framework leveraging perceptually grounded acoustic and linguistic characteristics, including prosody, vocal quality, semantic coherence, syntactic structure, and sarcasm. Using statistical analysis and interpretable machine learning (XGBoost with SHAP and LIME), we examine associations between speech features and validated symptom measures of depression, anxiety, and ADHD. Evaluated on both controlled benchmark datasets (StressID, DAIC-WOZ, Androids, EATD) and a real-world clinical dataset, the framework reveals stable and consistent relationships between symptom severity and vocal irregularities (e.g., shimmer, jitter), lexical-syntactic patterns, and affective tone. An ablation study conducted across all datasets further identifies the most informative feature groups. This work explores a transparent and clinically interpretable approach to speech-based mental health analysis.
>
---
#### [replaced 080] Evaluation of AI Ethics Tools in Language Models: A Developers' Perspective Case Study
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于AI伦理评估任务，旨在解决AI伦理工具在语言模型中的有效性问题。通过案例研究和开发者访谈，评估四种伦理工具的实用性与局限性。**

- **链接: [https://arxiv.org/pdf/2512.15791](https://arxiv.org/pdf/2512.15791)**

> **作者:** Jhessica Silva; Diego A. B. Moreira; Gabriel O. dos Santos; Alef Ferreira; Helena Maia; Sandra Avila; Helio Pedrini
>
> **备注:** 7 figures, 11 tables. Accepted for publication in AI and Ethics
>
> **摘要:** In Artificial Intelligence (AI), language models have gained significant importance due to the widespread adoption of systems capable of simulating realistic conversations with humans through text generation. Because of their impact on society, developing and deploying these language models must be done responsibly, with attention to their negative impacts and possible harms. In this scenario, the number of AI Ethics Tools (AIETs) publications has recently increased. These AIETs are designed to help developers, companies, governments, and other stakeholders establish trust, transparency, and responsibility with their technologies by bringing accepted values to guide AI's design, development, and use stages. However, many AIETs lack good documentation, examples of use, and proof of their effectiveness in practice. This paper presents a methodology for evaluating AIETs in language models. Our approach involved an extensive literature survey on 213 AIETs, and after applying inclusion and exclusion criteria, we selected four AIETs: Model Cards, ALTAI, FactSheets, and Harms Modeling. For evaluation, we applied AIETs to language models developed for the Portuguese language, conducting 35 hours of interviews with their developers. The evaluation considered the developers' perspective on the AIETs' use and quality in helping to identify ethical considerations about their model. The results suggest that the applied AIETs serve as a guide for formulating general ethical considerations about language models. However, we note that they do not address unique aspects of these models, such as idiomatic expressions. Additionally, these AIETs did not help to identify potential negative impacts of models for the Portuguese language.
>
---
#### [replaced 081] Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决LLM在RLVR中出现的校准退化问题，即模型对错误答案过于自信。提出DCPO框架，分离推理与校准目标，提升校准性能并缓解过拟合问题。**

- **链接: [https://arxiv.org/pdf/2603.09117](https://arxiv.org/pdf/2603.09117)**

> **作者:** Zhengzhao Ma; Xueru Wen; Boxi Cao; Yaojie Lu; Hongyu Lin; Jinglin Yang; Min He; Xianpei Han; Le Sun
>
> **备注:** Accepted at the 43rd International Conference on Machine Learning (ICML 2026)
>
> **摘要:** Reinforcement Learning from Verifiable Rewards (RLVR) significantly enhances large language models (LLMs) reasoning but severely suffers from calibration degeneration, where models become excessively over-confident in incorrect answers. Previous studies devote to directly incorporating calibration objective into existing optimization target. However, our theoretical analysis demonstrates that there exists a fundamental gradient conflict between the optimization for maximizing policy accuracy and minimizing calibration error. Building on this insight, we propose DCPO, a simple yet effective framework that systematically decouples reasoning and calibration objectives. Extensive experiments demonstrate that our DCPO not only preserves accuracy on par with GRPO but also achieves the best calibration performance and substantially mitigates the over-confidence issue. Our study provides valuable insights and practical solution for more reliable LLM deployment.
>
---
#### [replaced 082] Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于LLM代理内存管理任务，解决运行时内存效率与性能平衡问题。提出BudgetMem框架，通过预算层级路由实现查询感知的性能-成本控制。**

- **链接: [https://arxiv.org/pdf/2602.06025](https://arxiv.org/pdf/2602.06025)**

> **作者:** Haozhen Zhang; Haodong Yue; Tao Feng; Quanyu Long; Jianzhu Bao; Bowen Jin; Weizhi Zhang; Xiao Li; Jiaxuan You; Chengwei Qin; Wenya Wang
>
> **备注:** Accepted by ICML 2026. Code is available at this https URL
>
> **摘要:** Memory is increasingly central to Large Language Model (LLM) agents operating beyond a single context window, yet most existing systems rely on offline, query-agnostic memory construction that can be inefficient and may discard query-critical information. Although runtime memory utilization is a natural alternative, prior work often incurs substantial overhead and offers limited explicit control over the performance-cost trade-off. In this work, we present \textbf{BudgetMem}, a runtime agent memory framework for explicit, query-aware performance-cost control. BudgetMem structures memory processing as a set of memory modules, each offered in three budget tiers (i.e., \textsc{Low}/\textsc{Mid}/\textsc{High}). A lightweight router performs budget-tier routing across modules to balance task performance and memory construction cost, which is implemented as a compact neural policy trained with reinforcement learning. Using BudgetMem as a unified testbed, we study three complementary strategies for realizing budget tiers: implementation (method complexity), reasoning (inference behavior), and capacity (module model size). Across LoCoMo, LongMemEval, and HotpotQA, BudgetMem surpasses strong baselines when performance is prioritized (i.e., high-budget setting), and delivers better accuracy-cost frontiers under tighter budgets. Moreover, our analysis disentangles the strengths and weaknesses of different tiering strategies, clarifying when each axis delivers the most favorable trade-offs under varying budget regimes.
>
---
#### [replaced 083] When PCOS Meets Eating Disorders: An Explainable AI Approach to Detecting the Hidden Triple Burden
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗文本分类任务，旨在检测PCOS患者中隐藏的饮食障碍和身体形象问题。通过可解释AI方法，自动识别社交媒体中的多重健康负担。**

- **链接: [https://arxiv.org/pdf/2604.14356](https://arxiv.org/pdf/2604.14356)**

> **作者:** Apoorv Prasad; Susan McRoy
>
> **摘要:** Women with polycystic ovary syndrome (PCOS) face substantially elevated risks of body image distress, disordered eating, and metabolic challenges, yet existing natural language processing approaches for detecting these conditions lack transparency and cannot identify co-occurring presentations. We developed small, open-source language models to automatically detect this triple burden in social media posts with grounded explainability. We collected 1,000 PCOS-related posts from six subreddits, with two trained annotators labeling posts using guidelines operationalizing Lee et al. (2017) clinical framework. Three models (Gemma-2-2B, Qwen3-1.7B, DeepSeek-R1-Distill-Qwen-1.5B) were fine-tuned using Low-Rank Adaptation to generate structured explanations with textual evidence. The best model achieved 75.3 percent exact match accuracy on 150 held-out posts, with robust comorbidity detection and strong explainability. Performance declined with diagnostic complexity, indicating their best use is for screening rather than autonomous diagnosis.
>
---
#### [replaced 084] FEA-SLT: A Gloss-Free End-to-End Framework for Facial-Expression-Aware Sign Language Translation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于手语翻译任务，解决现有方法忽视面部表情导致语义模糊的问题。提出FEA-SLT框架，融合面部动态与手部动作，提升翻译准确性。**

- **链接: [https://arxiv.org/pdf/2601.03549](https://arxiv.org/pdf/2601.03549)**

> **作者:** Guobin Tu; Di Weng
>
> **摘要:** Sign Language Translation (SLT) is a challenging cross-modal task requiring joint modeling of manual articulations and non-manual signals. Existing gloss-free SLT methods effectively capture gestural dynamics but often underutilize facial expressions, which play crucial grammatical and disambiguating roles. This limitation can cause semantic degradation when distinct concepts share similar manual configurations. To address this issue, we propose FEA-SLT (**F**acial-**E**xpression-**A**ware **S**ign **L**anguage **T**ranslation), a gloss-free end-to-end framework that uses facial dynamics as semantic anchors for resolving manual ambiguity. FEA-SLT employs a domain-transferred facial encoder to extract expression-sensitive representations and integrates them with manual features through a linguistically constrained *Facial-Expression-Aware Fusion* (FEAF) module. FEAF captures reciprocal dependencies between manual and facial channels via bidirectional modulation, enhancing syntactic fidelity. Experiments on PHOENIX14T and CSL-Daily show that FEA-SLT achieves state-of-the-art BLEU performance among gloss-free methods, while targeted analyses confirm improved translation of facial-sensitive utterances. Code is available at [this https URL](this https URL).
>
---
#### [replaced 085] Identifying and Mitigating Bottlenecks in Role-Playing Agents: A Systematic Study of Disentangling Character Profile Axes
- **分类: cs.CL**

- **简介: 该论文属于角色扮演代理研究，旨在解决角色属性对性能影响不明确的问题。通过系统分析三个轴，发现道德属性显著影响表现，并提出FACD方法缓解此瓶颈。**

- **链接: [https://arxiv.org/pdf/2601.04716](https://arxiv.org/pdf/2601.04716)**

> **作者:** Yonghyun Jun; Junhyuk Choi; Jeonghyun Park; Jihyeong Park; Liu Nicole Geumheon; Hwanhee Lee
>
> **备注:** 28 pages
>
> **摘要:** While Large Language Model (LLM) role-playing agents have advanced rapidly, it remains unclear which profile elements genuinely drive role-playing quality. To bridge this gap, we introduce a systematic diagnostic framework that disentangles the impact of character profiles along three axes: Familiarity (Known vs. Unknown), Structure (Structured vs. Unstructured), and Disposition (Moral vs. Immoral). Utilizing a unified hierarchical schema (5 dimensions, 28 fields), we construct a controlled dataset of 211 personas and evaluate five LLMs on both single- and multi-turn interactions. Our results reveal a striking asymmetry: Familiarity and Structure show negligible impact, while Disposition produces large, consistent performance degradation for immoral characters across all conditions. Further analyses suggest that the Moral--Immoral gap is amplified by post-SFT alignment, and that this degradation varies substantially across profile attributes. To mitigate this bottleneck, we propose Field-Aware Contrastive Decoding (FACD), a training-free strategy that amplifies suppressed disposition-sensitive signals, significantly closing the performance gap without sacrificing moral-character performance.
>
---
#### [replaced 086] Probing Social Identity Bias in Chinese LLMs with Gendered Pronouns and Social Groups
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的社会偏见检测任务，旨在揭示中文大模型中的社会身份偏见。通过设计特定提示，分析代词和群体框架下的情感与毒性差异，评估模型的偏见表现。**

- **链接: [https://arxiv.org/pdf/2510.06974](https://arxiv.org/pdf/2510.06974)**

> **作者:** Geng Liu; Feng Li; Junjie Mu; Mengxiao Zhu; Francesco Pierri
>
> **摘要:** Large language models (LLMs) are increasingly deployed in user-facing applications, raising concerns that they may reflect and amplify social biases. We investigate social identity biases in Chinese LLMs using Mandarin-specific prompts across ten representative models. Our evaluation compares ingroup ("We") and outgroup ("They") framings across 240 social groups salient in the Chinese context, using a two-tiered measurement framework that assesses both sentiment and toxicity. The prompt design explicitly accounts for linguistic properties of Mandarin, including the distinction between the default gender-neutral plural pronoun and its explicitly feminine counterpart, enabling a controlled comparison of social identity framing effects. Across models, we observe systematic ingroup-outgroup asymmetries, although their expression differs across measurement dimensions. In particular, instruction tuning often reduces sentiment asymmetries, while toxicity gaps remain more persistent. Moreover, the feminine-marked plural pronoun is associated with higher toxicity than the default gender-neutral plural in several models. Our study introduces a language-aware evaluation framework for Chinese LLMs and shows that (i) social identity biases previously documented in English also manifest in Chinese and that (ii) Mandarin-specific linguistic structure can reveal bias patterns that are not directly observable in English-only settings.
>
---
#### [replaced 087] The Grammar of Transformers: A Systematic Review of Interpretability Research on Syntactic Knowledge in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，探讨Transformer模型的句法知识编码能力。通过系统综述337篇文献，分析模型在不同句法现象上的表现，揭示其句法知识的存储与处理机制。**

- **链接: [https://arxiv.org/pdf/2601.19926](https://arxiv.org/pdf/2601.19926)**

> **作者:** Nora Graichen; Iria de-Dios-Flores; Gemma Boleda
>
> **摘要:** We present a systematic review of 337 articles evaluating the syntactic abilities of Transformer-based language models (TLMs), reporting on over 3,000 datapoints spanning a wide range of syntactic phenomena, languages, models, and methods. We take the data to collectively show that TLMs encode a non-trivial amount of syntactic knowledge. Behavioral evidence shows strong performance on formal syntactic phenomena, but weaker and more variable performance on phenomena at the syntax-semantics interface. Performance is also consistently lower for languages with less digital support. Probing and mechanistic studies further support the presence of syntactic knowledge in TLMs. Yet, because most work remains observational and current approaches are methodologically heterogeneous, insight into the detailed computational mechanisms underlying syntactic processing remains limited. At the same time, the literature remains heavily concentrated on English and BERT-like models. We discuss the implications of our results and provide recommendations for future research.
>
---
#### [replaced 088] ClinConsensus: A Physician-Calibrated Benchmark for Evaluating Clinical Rubric Coverage in Chinese Medical LLMs
- **分类: cs.CL**

- **简介: 该论文属于医疗大模型评估任务，旨在解决临床标准覆盖不足的问题。提出ClinConsensus基准和CACS评分，评估模型响应的临床覆盖率。**

- **链接: [https://arxiv.org/pdf/2603.02097](https://arxiv.org/pdf/2603.02097)**

> **作者:** Xiang Zheng; Han Li; Wenjie Luo; Weiqi Zhai; Yiyuan Li; Chuanmiao Yan; Xue Yang; Kailuan Wu; Ruyi Xu; Tianyun Lu; Tianyi Tang; Yubo Ma; Kexin Yang; Dayiheng Liu; Sen Yang; Lin Qu; Bing Zhao; Hu Wei
>
> **摘要:** Open-ended medical LLM evaluation remains weakly grounded in physician-calibrated coverage of clinically relevant response criteria, especially in localized clinical settings. We introduce \textsc{ClinConsensus}, a Chinese medical benchmark of 2{,}500 expert-curated cases spanning 36 specialties, 12 task themes, multiple difficulty levels, and lay-facing versus professional-facing settings. Each case is paired with 30 case-specific binary rubric criteria. To evaluate whether responses satisfy enough physician-authored criteria, we propose \emph{Clinician-Anchored Coverage Score} (CACS), a physician-calibrated threshold metric instantiated at \(k=10\), and develop a dual-judge framework combining a GPT-5.1 grader with a physician-supervised Qwen3-8B judge. Evaluating 11 frontier LLMs, we find a persistent coverage gap: Rubric Accuracy ranges from 39.6\% to 52.1\%, whereas CACS@10 ranges from 17.8\% to 32.9\%, leaving a 19.2--21.9 point gap across models. Stratified analyses further reveal substantial variation across reasoning, evidence use, structured extraction, medication instructions, follow-up, and dialogue register. These results suggest that medical LLM evaluation should measure thresholded, rubric-grounded clinical coverage rather than average partial correctness.
>
---
#### [replaced 089] Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究非自回归扩散语言模型的解码问题，针对初始位置依赖性强的缺陷，提出改进方法提升推理与规划任务效果。**

- **链接: [https://arxiv.org/pdf/2604.10567](https://arxiv.org/pdf/2604.10567)**

> **作者:** Jiyeon Kim; Sungik Choi; Yongrae Jo; Moontae Lee; Minjoon Seo
>
> **备注:** ICML 2026 Camera Ready
>
> **摘要:** Diffusion-based language models (dLLMs) have emerged as a promising alternative to autoregressive language models, offering the potential for parallel token generation and bidirectional context modeling. However, harnessing this flexibility for fully non-autoregressive decoding remains an open question, particularly for reasoning and planning tasks. In this work, we investigate non-autoregressive decoding in dLLMs by systematically analyzing its inference dynamics along the temporal axis. Specifically, we uncover an inherent failure mode in confidence-based non-autoregressive generation stemming from a strong proximity bias-the tendency for the denoising order to concentrate on spatially adjacent tokens. This local dependency leads to spatial error propagation, rendering the entire trajectory critically contingent on the initial unmasking position. Leveraging this insight, we present a minimal-intervention approach that guides early token selection, employing a lightweight planner and end-of-sequence temperature annealing. We thoroughly evaluate our method on various reasoning and planning tasks and observe substantial overall improvement over existing heuristic baselines without significant computational overhead.
>
---
#### [replaced 090] Compliance versus Sensibility: On the Reasoning Controllability in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型的推理可控性问题，探讨如何解耦逻辑模式与具体任务。通过分析推理冲突，发现模型更注重合理性而非遵循指令，并提出方法提升指令遵循度。**

- **链接: [https://arxiv.org/pdf/2604.27251](https://arxiv.org/pdf/2604.27251)**

> **作者:** Xingwei Tan; Marco Valentino; Mahmud Elahi Akhter; Yuxiang Zhou; Maria Liakata; Nikolaos Aletras
>
> **摘要:** Large Language Models (LLMs) are known to acquire reasoning capabilities through shared inference patterns in pre-training data, which are further elicited via Chain-of-Thought (CoT) practices. However, whether fundamental reasoning patterns, such as induction, deduction, and abduction, can be decoupled from specific problem instances remains a critical challenge for model controllability, and for shedding light on reasoning controllability. In this paper, we present the first systematic investigation of this problem through the lens of reasoning conflicts: an explicit tension between parametric and contextual information induced by mandating logical schemata that deviate from those expected for a target task. Our evaluation reveals that LLMs consistently prioritize sensibility over compliance, favoring task-appropriate reasoning patterns despite conflicting instructions. We further demonstrate that reasoning conflicts are internally detectable, as confidence scores significantly drop during conflicting episodes. Probing experiments confirm that reasoning types are linearly encoded from middle-to-late layers, indicating the potential for activation-level controllability. Leveraging these insights, we steer models towards compliance, increasing instruction following by up to 29%. Overall, our findings establish that while LLM reasoning is anchored to concrete instances, active mechanistic interventions can effectively decouple logical schemata from data, offering a path toward improved controllability, faithfulness, and generalizability.
>
---
#### [replaced 091] Addressing Pitfalls in Auditing Practices of Automatic Speech Recognition Technologies: A Case Study of People with Aphasia
- **分类: cs.CY; cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别审计任务，旨在解决ASR系统对失语症患者不公平的问题。通过识别审计中的三个误区，提出改进框架并验证了失语症患者的性能下降。**

- **链接: [https://arxiv.org/pdf/2506.08846](https://arxiv.org/pdf/2506.08846)**

> **作者:** Katelyn Xiaoying Mei; Anna Seo Gyeong Choi; Hilke Schellmann; Mona Sloane; Allison Koenecke
>
> **备注:** Published at the Proceedings of The 2026 ACM Conference on Fairness, Accountability, and Transparency (FAccT '26)
>
> **摘要:** Automatic Speech Recognition (ASR) systems' growing use warrants robust auditing approaches to ensure equitable transcription quality, especially for people with speech disorders like aphasia who disproportionately depend on ASR. While academic and industry audits have revealed performance disparities across user populations, standard auditing practices often overlook nuances that risk masking harm to marginalized groups. We identify three common pitfalls in standard ASR audits: (1) adhering to one method of text standardization, which can mask variance in ASR performance and ignore the standardization preferences of marginalized communities; (2) displaying high-level demographic findings without considering performance disparities by nuanced intersectional subgroups, or conditioning on relevant acoustic properties; and (3) reporting only one gold-standard metric (Word Error Rate), which inadequately quantifies common generative AI errors like hallucinations. We propose a holistic auditing framework addressing these pitfalls, and in a case study of six popular ASR systems, find consistently worse ASR performance for speakers with aphasia relative to a control group. We call on practitioners to implement these robust, community-driven ASR auditing practices better suited for the rapidly changing ASR landscape.
>
---
#### [replaced 092] Federated Language Models Under Bandwidth Budgets: Distillation Rates and Conformal Coverage
- **分类: stat.ML; cs.CL; cs.LG**

- **简介: 该论文研究在带宽受限下的联邦语言模型，解决数据分布式训练与推理的统计保障问题。提出FPLD和FC-RAG协议，分析带宽对模型一致性与覆盖度的影响。**

- **链接: [https://arxiv.org/pdf/2605.09986](https://arxiv.org/pdf/2605.09986)**

> **作者:** Prasanjit Dubey; Xiaoming Huo
>
> **摘要:** Training a language model on data scattered across bandwidth-limited nodes that cannot be centralized is a setting that arises in clinical networks, enterprise knowledge bases, and scientific consortia. We study the regime in which data must remain distributed across nodes, and ask what statistical guarantees are in principle achievable under explicit bandwidth budgets; we aim to characterize what is provably possible, not to demonstrate a deployment-ready system. Existing theory treats either training-time consistency or inference-time calibration in isolation, and no prior work makes bandwidth a first-class statistical parameter. We analyze two protocols, Federated Probe-Logit Distillation (FPLD) for training and Federated Conformal RAG (FC-RAG) for inference, as the analytical vehicles for our results. Our first main result is an explicit high-probability KL-consistency rate for FPLD with simultaneous dependence on node count $K$, per-node sample size $n$, quantization budget $B$, probe-set size $m$, and vocabulary size $V$; bandwidth enters only through an exponentially vanishing quantization term. Our second main result is a distribution-free marginal-coverage bound for FC-RAG, whose novel retrieval-bandwidth slack $\Delta_{\mathrm{RAG}} = f_{\max}\sqrt{K^{-2}\sum_i v(B_i)}$ makes per-node retrieval bandwidth a first-class statistical parameter, with arithmetic aggregation across $K$ nodes shrinking the slack as $K^{-1/2}$ in the per-node-uniform regime. A Pinsker-type corollary composes the two bounds into an end-to-end coverage guarantee. Synthetic experiments verify the predicted scaling along the bounds' parameters; small-scale experiments on a GPT-2 testbed illustrate that the qualitative bandwidth-accuracy tradeoff survives on a real language model. A deployment-scale empirical evaluation is out of scope.
>
---
#### [replaced 093] Aligning Language Model Benchmarks with Pairwise Preferences
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决基准测试与实际性能不匹配的问题。通过引入基准对齐方法，利用模型表现数据更新基准，使其能更准确预测模型偏好。**

- **链接: [https://arxiv.org/pdf/2602.02898](https://arxiv.org/pdf/2602.02898)**

> **作者:** Marco Gutierrez; Xinyi Leng; Hannah Cyberey; Jonathan Richard Schwarz; Ahmed Alaa; Thomas Hartvigsen
>
> **摘要:** Language model benchmarks are pervasive and computationally-efficient proxies for real-world performance. However, many recent works find that benchmarks often fail to predict real utility. Towards bridging this gap, we introduce benchmark alignment, where we use limited amounts of information about model performance to automatically update offline benchmarks, aiming to produce new static benchmarks that predict model pairwise preferences in given test settings. We then propose BenchAlign, the first solution to this problem, which learns preference-aligned weight- ings for benchmark questions using the question-level performance of language models alongside ranked pairs of models that could be collected during deployment, producing new benchmarks that rank previously unseen models according to these preferences. Our experiments show that our aligned benchmarks can accurately rank unseen models according to models of human preferences, even across different sizes, while remaining interpretable. Overall, our work provides insights into the limits of aligning benchmarks with practical human preferences, which stands to accelerate model development towards real utility.
>
---
#### [replaced 094] A Benchmark Construction and Evaluation Framework for Specialist Domains: Case Study on Defense-related Documents
- **分类: cs.CL**

- **简介: 该论文属于RAG问答任务，解决专业领域缺乏评估基准的问题。提出DoRA框架，生成合成数据并评估模型性能，提升问答准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2604.17943](https://arxiv.org/pdf/2604.17943)**

> **作者:** Bao Gia Doan; Aditya Joshi; Pantelis Elinas; Aarya Bodhankar; Oscar Leslie; Tom Marchant; Flora Salim
>
> **摘要:** RAG-based question-answering (QA) in specialist domains faces a cold-start problem: lack of evaluative benchmarks and absence of labeled data for post-training. We present DoRA (Domain-oriented RAG Assessment), a novel benchmark construction and evaluation framework using only a small set of specialist domain documents. DoRA systematically generates synthetic QA training and evaluation datasets with auditable evidence across five domain-specific intents. To mitigate same-pipeline circularity, DoRA's training and test splits use different LLM families (Claude Sonnet for training; GPT-4o for test) drawn from disjoint seed-document corpora. Instantiated on 40 defense-related documents (written in English), DoRA yields ~6.6K curated instances. Compared against 8 LLM baselines over a benchmark of 1,259 samples, a LoRA-adapted Llama3.1-8B trained on the synthetic training set consistently improves performance over 6 coverage and faithfulness metrics, especially reducing hallucination by more than half under the default GTE retrieval setting, with gains persisting across alternative retrievers and prompting-based baselines. Defense-domain expertise is incorporated in three stages of our evaluation: (a) determining the quality of the synthetic QA generated by DoRA, (b) ascertaining the reliability of LLM-as-judge scores, and (c) evaluating the generalization of the QA pipeline on completely human-written QA examples. We position DoRA as a practical framework for specialist-domain RAG under domain shift, with defense as a high-stakes case study.
>
---
#### [replaced 095] Shopping Companion: Benchmarking and Training LLM Agents for Long-Horizon Preference-Grounded E-Commerce Tasks
- **分类: cs.CL**

- **简介: 该论文属于电商任务，解决长周期用户偏好捕捉与代理训练问题。构建了基准数据集，设计奖励机制提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.14864](https://arxiv.org/pdf/2603.14864)**

> **作者:** Zijian Yu; Kejun Xiao; Huaipeng Zhao; Tao Luo; Xiaoyi Zeng
>
> **摘要:** In e-commerce, LLM agents show promise for shopping tasks such as recommendations, budget management, and bundle deals, where accurately capturing user preferences from long-horizon conversations is critical. However, progress is limited by two key challenges: (1) the absence of benchmarks for evaluating long-term preference-aware shopping tasks, and (2) the lack of fine-grained supervision for shopping agent training. To fill the benchmark gap, we introduce Shopping Companion Bench, a novel benchmark comprising two shopping tasks that require cross-session preference memory, grounded in a product pool of over 1.2 million real-world items. Our analysis further identifies two major sources of failure on this benchmark: cascading errors caused by preference hallucination, and insufficient verification of product attributes against user requirements. To address these failure modes, we design annotation-free, tool-wise rewards that provide process supervision for each tool call, alleviating reward sparsity in long-horizon tasks. Experimental results demonstrate that even state-of-the-art models such as GPT-5 achieve success rates below 70%, highlighting the difficulty of our benchmark. Notably, our fine-tuned lightweight 4B model consistently outperforms strong baselines in both preference capture and task performance, suggesting the effectiveness of our reward design.
>
---
#### [replaced 096] DRTriton: Large-Scale Synthetic Data Driven Reinforcement Learning for Triton Kernel Generation
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出DRTriton，解决将PyTorch代码转换为高效CUDA内核的问题，通过合成数据和强化学习提升转换效果。**

- **链接: [https://arxiv.org/pdf/2603.21465](https://arxiv.org/pdf/2603.21465)**

> **作者:** Siqi Guo; Ming Lin; Tianbao Yang
>
> **摘要:** Developing efficient CUDA kernels is a fundamental yet challenging task in the generative AI industry. Recent research leverages Large Language Models (LLMs) to automatically convert PyTorch reference implementations to CUDA kernels, significantly reducing engineering effort. State-of-the-art LLMs, such as GPT-5.2 and Claude-Sonnet-4.5, still struggle with this task. To address this challenge, we propose DRTriton, a scalable learning framework for training LLMs to convert PyTorch programs into highly optimized Triton kernels, which are then compiled to CUDA kernels at runtime. DRTriton consists of three key components: (i) a data synthetic algorithm CSP-DAG that guarantees full coverage and unbiased uniform sampling over the operator space with controlled difficulty; (ii) a curriculum RL framework with decoupled rewards that jointly optimizes conversion success rate and execution speed; and (iii) a test-time search algorithm that further improves the execution speed of the generated Triton kernels. With a warmup stage of SFT on limited PyTorch-Triton pairs curated using existing LLMs, DRTriton trained by RL on synthesized PyTorch programs generalizes effectively to real-world CUDA kernels that are challenging even for human experts. Experimental results show that DRTriton-7B achieves speedup over PyTorch on 92% of KernelBench Level 2 tasks, compared to 23% for GPT-5.2 and 19% for Claude-Sonnet-4.5.
>
---
#### [replaced 097] SaFeR-Steer: Evolving Multi-Turn MLLMs via Synthetic Bootstrapping and Feedback Dynamics
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出SaFeR-Steer框架，解决多轮对话中安全对齐不足的问题，通过合成数据和反馈机制提升模型安全性与帮助性。**

- **链接: [https://arxiv.org/pdf/2604.16358](https://arxiv.org/pdf/2604.16358)**

> **作者:** Haolong Hu; Hanyu Li; Tiancheng He; Huahui Yi; An Zhang; Qiankun Li; Kun Wang; Yang Liu; Zhigang Zeng
>
> **摘要:** MLLMs are increasingly deployed in multi-turn settings, where attackers can escalate unsafe intent through the evolving visual-text history and exploit long-context safety decay. Yet safety alignment is still dominated by single-turn data and fixed-template dialogues, leaving a mismatch between training and deployment. To bridge this gap, we propose SaFeR-Steer, a progressive multi-turn alignment framework that combines staged synthetic bootstrapping with tutor-in-the-loop GRPO to train a single student under adaptive, on-policy attacks. We also introduce Trajectory-Consistent Summative Reward (TCSR), which aggregates the historical minimum and average of turn rewards so that any low-quality turn affects the trajectory-level return. I. Dataset. We release STEER, a multi-turn multimodal safety dataset with STEER-SFT (12,934), STEER-RL (2,000), and STEER-Bench (3,227) dialogues spanning 2-10 turns. II. Experiment. Starting from Qwen2.5-VL-3B/7B, SaFeR-Steer substantially improves Safety/Helpfulness on both single-turn (48.30/45.86 $\rightarrow$ 81.84/70.77 for 3B; 56.21/60.32 $\rightarrow$ 87.89/77.40 for 7B) and multi-turn benchmarks (12.55/27.13 $\rightarrow$ 55.58/70.27 for 3B; 24.66/46.48 $\rightarrow$ 64.89/72.35 for 7B), shifting failures to later turns and yielding robustness beyond scaling alone. Code is available at this https URL
>
---
#### [replaced 098] Beyond External Monitors: Enhancing Transparency of Large Language Models for Easier Monitoring
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于模型透明性研究，旨在提升大语言模型的可监控性。针对现有方法无法准确反映模型思维过程的问题，提出TELLME方法，增强模型透明度并检测不当行为。**

- **链接: [https://arxiv.org/pdf/2502.05242](https://arxiv.org/pdf/2502.05242)**

> **作者:** Guanxu Chen; Jing Shao; Tao Luo; Lijie Hu; Qihao Lin; Dongrui Liu
>
> **备注:** 28 pages,8 figures,15 tables
>
> **摘要:** Large language models (LLMs) are becoming increasingly capable, but the mechanisms of their thinking and decision-making processes remain unclear. Chain-of-thoughts (CoTs) have been commonly utilized to externalize LLMs' thinking, but this strategy fails to accurately reflect LLMs' thinking process. Techniques based on LLMs' hidden representations provide an inner perspective to improve the monitorability of their latent thinking. However, previous methods only try to develop external modules instead of making LLMs themselves easier to monitor. In this paper, we propose a novel method, TELLME, improving the transparency of LLMs and helping monitors identify unsuitable and sensitive behaviors. Furthermore, we showcase the effectiveness of TELLME on detoxification tasks, where LLMs achieve consistent improvement among multimodal test sets, distinct architectures, and varying parameter scales. We further analyze TELLME's improvement on LLMs' generalization ability from both optimal transport theory and empirical perspectives.
>
---
#### [replaced 099] Evaluating the Evaluator: Problems with SemEval-2020 Task 1 for Lexical Semantic Change Detection
- **分类: cs.CL**

- **简介: 该论文针对SemEval-2020任务1的语义变化检测基准进行评估，指出其在操作化、数据质量和基准设计上的不足，提出改进方向。**

- **链接: [https://arxiv.org/pdf/2604.13232](https://arxiv.org/pdf/2604.13232)**

> **作者:** Bach Phan-Tat; Kris Heylen; Dirk Geeraerts; Stefano De Pascale; Dirk Speelmana
>
> **摘要:** This discussion paper re-examines SemEval-2020 Task 1, the most influential shared benchmark for lexical semantic change detection, through a three-part evaluative framework: operationalisation, data quality, and benchmark design. First, at the level of operationalisation, we argue that the benchmark models semantic change mainly as gain, loss, or redistribution of discrete senses. While practical for annotation and evaluation, this framing is too narrow to capture gradual, constructional, collocational, and discourse-level change. Also, the gold labels are outcomes of annotation decisions, clustering procedures, and threshold settings, which could potentially limit the validity of the task. Second, at the level of data quality, we show that the benchmark is affected by substantial corpus and preprocessing problems, including OCR noise, malformed characters, truncated sentences, inconsistent lemmatisation, POS-tagging errors, and missed targets. These issues can distort model behaviour, complicate linguistic analysis, and reduce reproducibility. Third, at the level of bench-mark design, we argue the small curated target sets and limited language coverage reduce realism and increase statistical uncertainty. Taken together, these limitations suggest that the benchmark should be treated as a useful but partial test bed rather than a definitive measure of progress. We therefore call for future datasets and shared tasks to adopt broader theories of semantic change, document pre-processing transparently, expand cross-linguistic coverage, and use more realistic evaluation settings. Such steps are necessary for more valid, interpretable, and generalisable progress in lexical semantic change detection
>
---
#### [replaced 100] Syntax as a Rosetta Stone: Universal Dependencies for In-Context Coptic Translation
- **分类: cs.CL**

- **简介: 该论文属于低资源机器翻译任务，旨在提升科普特语到英语的翻译效果。通过结合词典信息与句法分析，提出一种新的上下文学习方法，显著提升了翻译性能。**

- **链接: [https://arxiv.org/pdf/2604.18758](https://arxiv.org/pdf/2604.18758)**

> **作者:** Abhishek Purushothama; Emma Thronson; Alexia Guo; Amir Zeldes
>
> **备注:** ACL 2026 Findings camera-ready, with fixes
>
> **摘要:** Low-resource machine translation requires methods that differ from those used for high-resource languages. This paper proposes a novel in-context learning approach to support low-resource machine translation of the Coptic language to English, with syntactic augmentation from Universal Dependencies parses of input sentences. Building on existing work using bilingual dictionaries to support inference for vocabulary items, we add several representations of syntactic analyses to our inputs , specifically exploring the inclusion of raw parser outputs, verbalizations of parses in plain English, and targeted instructions of difficult constructions identified in sub-trees and how they can be translated. Our results show that while syntactic information alone is not as useful as dictionary-based glosses, combining retrieved dictionary items with syntactic information achieves significant gains across model sizes, achieving new state-of-the-art translation results for Coptic.
>
---
#### [replaced 101] Probing for Knowledge Attribution in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识归属任务，旨在解决大语言模型生成内容的错误来源识别问题。通过构建数据集和线性探测器，有效区分答案来源，提升错误检测能力。**

- **链接: [https://arxiv.org/pdf/2602.22787](https://arxiv.org/pdf/2602.22787)**

> **作者:** Ivo Brink; Alexander Boer; Dennis Ulmer
>
> **摘要:** Large language model (LLM) hallucinations, meaning fluent but factually incorrect generations, fall into two types: faithfulness violations, where the model misuses provided context, and factuality violations, where answers reflect errors in internal knowledge. Proper mitigation depends on knowing which source drives each answer. We study contributive attribution, i.e. the classification of the dominant knowledge source behind each output, and show that a simple linear probe trained on hidden representations can reliably identify it. We introduce AttriWiki, a self-supervised pipeline that automatically generates labelled training data by prompting models to recall withheld entities from memory or read them from context without relying on knowledge conflicts. Probes trained on AttriWiki achieve up to 0.96 Macro-$F_1$ on Llama-3.1-8B, Mistral-7B, and Qwen-7B, transfer to SQuAD and WebQuestions with 0.94-0.99 Macro-$F_1$, and generalise zero-shot to Tighidet et al. (2024)'s benchmark, outperforming their probe on conflicting settings without retraining. Furthermore, attribution mismatches raise error rates by up to 70%, though correct attribution does not guarantee correct answers, pointing to the need for broader detection frameworks.
>
---
#### [replaced 102] On the Fallacy of Global Token Perplexity in Spoken Language Model Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音语言模型评估任务，指出传统全局token困惑度评估方法存在偏差，提出新的评估方法以更准确反映生成质量。**

- **链接: [https://arxiv.org/pdf/2601.06329](https://arxiv.org/pdf/2601.06329)**

> **作者:** Chan-Jan Hsu; Liang-Hsuan Tseng; Yi-Cheng Lin; Yen-Chun Kuo; Ju-Chieh Chou; Kai-Wei Chang; Hung-yi Lee; Carlos Busso
>
> **摘要:** Generative spoken language models pretrained on large-scale raw audio can continue a speech prompt with appropriate content while preserving attributes like speaker and emotion, serving as foundation models for spoken dialogue. In prior literature, these models are often evaluated using ``global token perplexity'', which directly applies the text perplexity formulation to speech tokens. However, this practice overlooks fundamental differences between speech and text modalities, possibly leading to an underestimation of the speech characteristics. In this work, we propose a variety of likelihood- and generative-based evaluation methods that serve in place of naive global token perplexity. We demonstrate that the proposed evaluations more faithfully reflect perceived generation quality, as evidenced by stronger correlations with human-rated mean opinion scores (MOS). When assessed under the new metrics, the relative performance landscape of spoken language models is reshaped, revealing a significantly reduced gap between the best-performing model and the human topline. Together, these results suggest that appropriate evaluation is critical for accurately assessing progress in spoken language modeling.
>
---
#### [replaced 103] AFRILANGTUTOR: Advancing Language Tutoring and Culture Education in Low-Resource Languages with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于低资源语言教学任务，旨在解决缺乏训练数据的问题。通过构建词典和对话数据集，训练语言辅导模型，提升AI辅助语言学习效果。**

- **链接: [https://arxiv.org/pdf/2604.20996](https://arxiv.org/pdf/2604.20996)**

> **作者:** Tadesse Destaw Belay; Shahriar Kabir Nahin; Israel Abebe Azime; Ocean Monjur; Marek Rei; Chris Biemann; Shamsuddeen Hassan Muhammad; Seid Muhie Yimam; Anshuman Chhabra
>
> **摘要:** How can language learning systems be developed for languages that lack sufficient training resources? This challenge is increasingly faced by developers across the African continent who aim to build AI systems capable of understanding and responding in local languages. To address this gap, we introduce AFRILANGDICT, a collection of 194.7K African language-English dictionary entries designed as seed resources for generating language-learning materials, enabling us to automatically construct large-scale, diverse, and verifiable student-tutor question-answer interactions suitable for training AI-assisted language tutors. Using AFRILANGDICT, we build AFRILANGEDU, a dataset of 78.9K multi-turn training examples for Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO). Using AFRILANGEDU, we train language tutoring models collectively referred to as AFRILANGTUTOR. We fine-tune two multilingual LLMs: Llama-3-8B-IT and Gemma-3-12B-IT on AFRILANGEDU across 10 African languages and evaluate their performance. Our results show that models trained on AFRILANGEDU consistently outperform their base counterparts, and combining SFT and DPO yields substantial improvements, with gains ranging from 1.8% to 15.5% under LLM-as-a-judge evaluations across four criteria. To facilitate further research on low-resource languages, all resources are available at this https URL.
>
---
#### [replaced 104] Quality-constrained Entropy Maximization Policy Optimization for LLM Diversity
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型对齐任务，旨在解决输出质量与多样性之间的权衡问题。提出QEMPO框架，在保证质量的前提下最大化多样性。**

- **链接: [https://arxiv.org/pdf/2602.15894](https://arxiv.org/pdf/2602.15894)**

> **作者:** Haihui Pan; Yuzhong Hong; Kaichen Zhang; Shaoke Lv; Junwei Bao; Hongfei Jiang; Yang Song
>
> **摘要:** In many large language model (LLM) alignment applications, users expect not only high-quality outputs but also substantial diversity. However, existing methods often face a fundamental trade-off between these objectives: approaches that improve output quality tend to reduce diversity, while methods that increase diversity often do so at the expense of quality. In this work, we propose Quality-constrained Entropy Maximization Policy Optimization (QEMPO), a novel framework that enhances the diversity of LLM outputs while explicitly preserving output quality. QEMPO is grounded in a strong theoretical foundation: we derive a closed-form analytical solution that provably maximizes entropy-a principled measure of diversity-subject to a quality constraint, with guarantees on optimality under the defined objective. Leveraging this solution, QEMPO naturally supports both online and offline training settings. Empirical results demonstrate that QEMPO consistently improves output diversity without sacrificing quality, and in many cases yields gains in both dimensions compared to existing baselines, aligning with our theoretical guarantees.
>
---
#### [replaced 105] Regression Language Models for Code
- **分类: cs.CL; cs.AI; cs.LG; cs.PF; cs.SE**

- **简介: 该论文研究代码到度量的回归任务，旨在预测代码执行的数值结果。通过统一的回归语言模型，直接从文本预测内存占用、延迟、准确率等指标，解决传统方法依赖特征工程的问题。**

- **链接: [https://arxiv.org/pdf/2509.26476](https://arxiv.org/pdf/2509.26476)**

> **作者:** Yash Akhauri; Xingyou Song; Arissa Wongpanich; Bryan Lewandowski; Mohamed S. Abdelfattah
>
> **备注:** Published in International Conference on Machine Learning (ICML) 2026
>
> **摘要:** We study code-to-metric regression: predicting numeric outcomes of code executions, a challenging task due to the open-ended nature of programming languages. While prior methods have resorted to heavy and domain-specific feature engineering, we show that a single unified Regression Language Model (RLM) using a frozen LLM encoder can simultaneously predict directly from text, (i) the memory footprint of code across multiple high-level languages such as Python and C++, (ii) the latency of Triton GPU kernels, and (iii) the accuracy and speed of trained neural networks represented in ONNX. In particular, a relatively small 300M parameter RLM based on T5Gemma, obtains $>$0.9 Spearman-rank on competitive programming submissions from APPS, and a single unified model achieves $>$0.5 average Spearman-rank across 17 separate languages from CodeNet. Furthermore, the RLM can obtain the highest average Kendall-Tau of 0.46 on five classic NAS design spaces previously dominated by graph neural networks, and simultaneously predict architecture latencies on numerous hardware platforms.
>
---
#### [replaced 106] Teaching and Evaluating LLMs to Reason About Polymer Design Related Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于聚合物设计任务，解决LLMs缺乏相关知识和能力的问题。通过构建PolyBench数据集并引入知识增强的推理蒸馏方法，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.16312](https://arxiv.org/pdf/2601.16312)**

> **作者:** Dikshya Mohanty; Mohammad Saqib Hasan; Syed Mostofa Monsur; Size Zheng; Benjamin Hsiao; Niranjan Balasubramanian
>
> **摘要:** Research in AI4Science has shown promise in many science applications, including polymer design. However, current LLMs are ineffective in this problem space because: (i) most models lack polymer-specific knowledge, and (ii) existing aligned models have limited coverage of knowledge and capabilities relevant to polymer design. Addressing this, we introduce PolyBench, a large-scale training and test benchmark dataset of more than 125K polymer design-related tasks, leveraging a knowledge base of more than 13 million data points obtained from experimental and synthetic data sources to ensure broad coverage of polymers and their properties. For effective alignment using PolyBench, we introduce a knowledge-augmented reasoning distillation method that augments this dataset with structured CoT. Furthermore, tasks in PolyBench are organized from simple to complex analytical reasoning problems, enabling generalization tests and diagnostic probes across the problem space. Experiments show that small- and mid- sized language models (SLMs) with 7B to 32BB parameters, trained on PolyBench, outperform similar-sized models and remain competitive with closed-source frontier LLMs on PolyBench's test dataset, while demonstrating performance gains on external polymer benchmarks. Dataset and associated code available at this https URL.
>
---
#### [replaced 107] Mitigating Cross-Lingual Cultural Inconsistencies in LLMs via Consensus-Driven Preference Optimisation
- **分类: cs.CL**

- **简介: 该论文属于多语言模型任务，解决跨语言文化不一致问题。通过引入新度量和C-3PO框架，提升模型在不同语言下的文化一致性。**

- **链接: [https://arxiv.org/pdf/2605.12515](https://arxiv.org/pdf/2605.12515)**

> **作者:** Lucas Resck; Isabelle Augenstein; Anna Korhonen
>
> **备注:** 24 pages, 13 figures, 11 tables
>
> **摘要:** Despite their impressive capabilities, multilingual large language models (MLLMs) frequently exhibit inconsistent behaviour when the prompt's language changes. While such adaptation is generally desirable, it becomes a critical failure when a user's identity is explicitly defined. For instance, given a fixed British persona and an ambiguous everyday knowledge query about literature, the prompt's language frequently overwrites the system persona -- yielding Shakespeare in English but Cervantes in Spanish. To robustly quantify this Cross-lingual Cultural Inconsistency, we introduce Singleton Fleiss's $\kappa_S$, a metric mathematically resilient to hallucinations. For mitigation, we propose Cross-lingual Cultural Consistent Preference Optimisation (C-3PO), a consensus-driven alignment framework. C-3PO achieves up to a 0.13-point absolute increase in $\kappa_S$ over unaligned models, consistently outperforming strong prompting and representation steering baselines whilst preserving explicit user identities, cultural neutrality and intrinsic cultural knowledge. Empirical evaluations demonstrate this inconsistency disproportionately affects lower-resource languages like Indonesian and Persian. Finally, early decoding of intermediate layers reveals that MLLMs implicitly personalise outputs towards the prompt language's stereotypical culture as forward-pass representations stabilise.
>
---
#### [replaced 108] Large Language Models as Automatic Annotators and Annotation Adjudicators for Fine-Grained Opinion Analysis
- **分类: cs.CL**

- **简介: 论文探讨将大语言模型用于细粒度情感分析的自动标注与标注仲裁，解决标注成本高、耗时的问题。工作包括构建标注流程和方法，验证模型在不同任务中的表现。**

- **链接: [https://arxiv.org/pdf/2601.16800](https://arxiv.org/pdf/2601.16800)**

> **作者:** Gaurav Negi; MA Waskow; John McCrae; Omnia Zayed; Paul Buitelaar
>
> **摘要:** Fine-grained opinion analysis of text provides a detailed understanding of expressed sentiments, including the addressed entity. Although this level of detail is valuable, annotating opinions in datasets for model training requires considerable human effort and substantial cost, especially across diverse domains and real-world applications. To address this shortage of domain-specific labelled datasets, we explore the feasibility of LLMs as automatic annotators for fine-grained opinion analysis. We use a declarative annotation pipeline, an approach that reduces the variability of manual prompt engineering when using LLMs to identify fine-grained opinion spans in text. We also present a dedicated methodology for an LLM to adjudicate multiple labels and produce final annotations. We trial the pipeline with models of different sizes for the Aspect Sentiment Triplet Extraction (ASTE) and Aspect-Category-Opinion-Sentiment (ACOS) analysis tasks. In this work, we attempt to develop fully autonomous LLM-based annotators, but our results reveal an uneven picture characterised by a critical performance bifurcation: LLMs are reliable at the span level yet struggle to faithfully reproduce the relational structures that connect those spans. This suggests that LLMs are better positioned as high-fidelity annotation assistants and data augmentation tools to expand fine-grained opinion-annotated datasets, rather than replacing human annotators entirely.
>
---
#### [replaced 109] Knowledge Graph-Driven Expert-Level Reasoning for Neuroscience
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱驱动的神经科学推理任务，旨在通过单本权威教材构建高质量知识图谱，提升语言模型的专家级推理能力。**

- **链接: [https://arxiv.org/pdf/2605.25183](https://arxiv.org/pdf/2605.25183)**

> **作者:** Jake Stephen; Niraj K. Jha
>
> **摘要:** Knowledge graph (KG) is an abstraction that can be extracted from text corpora and used for in-depth reasoning. Prior work has leveraged KGs to fine-tune language models (LMs), enabling domain-specific superintelligence. In this work, we explore whether KG-driven in-depth reasoning capabilities can emerge in neuroscience using only information contained within a single authoritative textbook. The central hypothesis is that structured knowledge, when distilled into a high-quality KG and converted into KG-grounded question-answer (QA) supervision, is sufficient to produce expert-level reasoning through a fine-tuned LM that surpasses large language models (LLMs) in accuracy, while employing orders of magnitude fewer parameters. We construct a textbook-derived KG via a dual-LLM validation pipeline, expand it with a masked LM trained on the KG topology, generate multi-hop QA items, which include QA pairs and reasoning traces, to fine-tune an LM exclusively on KG-derived supervision, and apply reinforcement learning using path-derived KG signals as implicit reward models. Our results demonstrate that deep, mechanistic neuroscience understanding can be induced in the model without reliance on large, heterogeneous web-scale corpora. The KG-based synthetic neuroscience curriculum that readers can quiz themselves on, and the fine-tuned LM, are available at the following GitHub location: this https URL.
>
---
#### [replaced 110] SetupX: Can LLM Agents Learn from Past Failures in Functionality-Correct Code Repository Setup?
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于代码仓库配置任务，解决LLM代理在环境配置中失败的问题。提出SetupX框架，通过经验学习、回滚机制和验证协议提升配置成功率。**

- **链接: [https://arxiv.org/pdf/2605.26186](https://arxiv.org/pdf/2605.26186)**

> **作者:** Zihang Zhou; Ziqian Ren; Yukai Wu; Yingjie Xiong; Wei Zhou; Chao Peng; Dong Zhang; Bingheng Yan; Xuanhe Zhou; Fan Wu
>
> **备注:** 21 pages, 6 figures
>
> **摘要:** Functionality-correct repository setup aims to configure execution environments (e.g., dependencies, build scripts) to successfully execute a repository's documented features. It presents significant challenges due to diverse, repository-specific failures, including dependency incompatibilities, missing toolchains, incomplete installations, and verification-strategy mismatches. Existing LLM agents struggle to robustly resolve these issues, specifically failing to support (1) cross-repository experience transfer, (2) multi-step trial-and-repair under non-invertible state changes, and (3) robust verification of setup outcomes to distinguish setup-induced failures from repository bugs. To address this, we introduce SetupX, an experiential learning-based setup framework. First, we construct a Self-Evolving Experience Representation (XPU), a dual-modality knowledge unit encoding setup signals, textual guidance, executable actions to dynamically transfer verified environment fixes to unseen repositories. Second, we employ Experience-Augmented Speculative Execution backed by a LIFO Docker snapshot stack, enabling the agent to proactively trial fixes and safely roll back to known-good states. Third, we introduce a Prosecutor-Judge Verification Protocol that separates evidence collection from final judgment, enabling more reliable setup verification beyond superficial build-time metrics. Evaluation results on carefully-crafted benchmarks show SetupX achieves highest performance (e.g., 92% pass rate) and outperforms the strongest baseline by over 19%. Crucially, SetupX excels in complex multi-repository setup requiring coordinating multiple interconnected services across different containers. The code repository is available at this https URL.
>
---
#### [replaced 111] Escaping Mode Collapse in LLM Generation via Geometric Regulation
- **分类: cs.CL; cond-mat.dis-nn; cs.AI; nlin.CD**

- **简介: 该论文属于自然语言生成任务，解决模式崩溃问题。通过几何调控方法，提升生成多样性与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.00435](https://arxiv.org/pdf/2605.00435)**

> **作者:** Xin Du; Kumiko Tanaka-Ishii
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Mode collapse is a persistent challenge in generative modeling and appears in autoregressive text generation as behaviors ranging from explicit looping to gradual loss of diversity and premature trajectory convergence. We take a dynamical-systems view and reinterpret mode collapse as reduced state-space accessibility caused by *geometric collapse*: during generation, the model's internal trajectory becomes confined to a low-dimensional region of its representation space. This implies mode collapse is not purely a token-level phenomenon and cannot be reliably solved by symbolic constraints or probability-only decoding heuristics. Guided by this perspective, we propose *Reinforced Mode Regulation* (RMR), a lightweight, online state-space intervention that regulates dominant self-reinforcing directions in the Transformer value cache (implemented as low-rank damping). Across multiple large language models, RMR substantially reduces mode collapse and enables stable generation at extremely low entropy rates (down to 0.8 nats/step), whereas standard decoding typically collapses near 2.0 nats/step.
>
---
