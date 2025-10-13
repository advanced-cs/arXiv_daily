# 自然语言处理 cs.CL

- **最新发布 144 篇**

- **更新 76 篇**

## 最新发布

#### [new 001] LatentBreak: Jailbreaking Large Language Models through Latent Space Feedback
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出LatentBreak，一种白盒越狱攻击方法，旨在绕过大语言模型的安全机制。它生成低困惑度、语义等效的对抗提示，以规避基于困惑度的检测。该方法通过在潜在空间中最小化对抗提示与无害请求的表示距离，实现更隐蔽的攻击。**

- **链接: [http://arxiv.org/pdf/2510.08604v1](http://arxiv.org/pdf/2510.08604v1)**

> **作者:** Raffaele Mura; Giorgio Piras; Kamilė Lukošiūtė; Maura Pintor; Amin Karbasi; Battista Biggio
>
> **摘要:** Jailbreaks are adversarial attacks designed to bypass the built-in safety mechanisms of large language models. Automated jailbreaks typically optimize an adversarial suffix or adapt long prompt templates by forcing the model to generate the initial part of a restricted or harmful response. In this work, we show that existing jailbreak attacks that leverage such mechanisms to unlock the model response can be detected by a straightforward perplexity-based filtering on the input prompt. To overcome this issue, we propose LatentBreak, a white-box jailbreak attack that generates natural adversarial prompts with low perplexity capable of evading such defenses. LatentBreak substitutes words in the input prompt with semantically-equivalent ones, preserving the initial intent of the prompt, instead of adding high-perplexity adversarial suffixes or long templates. These words are chosen by minimizing the distance in the latent space between the representation of the adversarial prompt and that of harmless requests. Our extensive evaluation shows that LatentBreak leads to shorter and low-perplexity prompts, thus outperforming competing jailbreak algorithms against perplexity-based filters on multiple safety-aligned models.
>
---
#### [new 002] Measuring Moral LLM Responses in Multilingual Capacities
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言大模型评估任务，旨在解决不同语言背景下模型响应的准确性和一致性问题。研究通过五维度评分体系，评估多个模型在低、高资源语言中的表现。结果显示，GPT-5整体最优，尤其在伦理相关类别表现突出，而其他模型如Gemini 2.5 Pro表现较差，突显出多语言测试与改进的必要性。**

- **链接: [http://arxiv.org/pdf/2510.08776v1](http://arxiv.org/pdf/2510.08776v1)**

> **作者:** Kimaya Basu; Savi Kolari; Allison Yu
>
> **备注:** 10 pages, 5 figures; referenced articles: arXiv:2303.08774, arXiv:2303.12528, arXiv:2308.14132, arXiv:2505.12201, arXiv:2406.04428, arXiv:2407.02273, arXiv:2404.01268, arXiv:2502.09747, arXiv:2507.13474, arXiv:2505.21479, arXiv:2306.05685
>
> **摘要:** With LLM usage becoming widespread across countries, languages, and humanity more broadly, the need to understand and guardrail their multilingual responses increases. Large-scale datasets for testing and benchmarking have been created to evaluate and facilitate LLM responses across multiple dimensions. In this study, we evaluate the responses of frontier and leading open-source models in five dimensions across low and high-resource languages to measure LLM accuracy and consistency across multilingual contexts. We evaluate the responses using a five-point grading rubric and a judge LLM. Our study shows that GPT-5 performed the best on average in each category, while other models displayed more inconsistency across language and category. Most notably, in the Consent & Autonomy and Harm Prevention & Safety categories, GPT scored the highest with averages of 3.56 and 4.73, while Gemini 2.5 Pro scored the lowest with averages of 1.39 and 1.98, respectively. These findings emphasize the need for further testing on how linguistic shifts impact LLM responses across various categories and improvement in these areas.
>
---
#### [new 003] A Human Behavioral Baseline for Collective Governance in Software Projects
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究开源社区如何通过版本控制的治理文档描述参与与控制机制。任务是分析软件项目集体治理的行为基线。工作包括解析文本为角色、规则、行为等元素，测量其变化，发现治理随时间扩展并趋于平衡，规则则保持稳定，为未来AI辅助治理提供评估基础。**

- **链接: [http://arxiv.org/pdf/2510.08956v1](http://arxiv.org/pdf/2510.08956v1)**

> **作者:** Mobina Noori; Mahasweta Chakraborti; Amy X Zhang; Seth Frey
>
> **备注:** Algorithmic Collective Action Workshop @ NeurIPS 2025. arXiv admin note: text overlap with arXiv:2509.16295
>
> **摘要:** We study how open source communities describe participation and control through version controlled governance documents. Using a corpus of 710 projects with paired snapshots, we parse text into actors, rules, actions, and objects, then group them and measure change with entropy for evenness, richness for diversity, and Jensen Shannon divergence for drift. Projects define more roles and more actions over time, and these are distributed more evenly, while the composition of rules remains stable. These findings indicate that governance grows by expanding and balancing categories of participation without major shifts in prescriptive force. The analysis provides a reproducible baseline for evaluating whether future AI mediated workflows concentrate or redistribute authority.
>
---
#### [new 004] Quality Estimation Reranking for Document-Level Translation
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决文档级翻译质量提升问题。通过使用质量估计（QE）重排序方法，在多个生成翻译中选择最佳结果。论文评估了不同QE指标在文档级翻译中的表现，发现使用SLIDE和GEMBA-DA等指标可显著提升BLEURT-20分数，证明文档级QE具有实际价值。**

- **链接: [http://arxiv.org/pdf/2510.08870v1](http://arxiv.org/pdf/2510.08870v1)**

> **作者:** Krzysztof Mrozinski; Minji Kang; Ahmed Khota; Vincent Michael Sutanto; Giovanni Gatti De Giacomo
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Quality estimation (QE) reranking is a form of quality-aware decoding which aims to improve machine translation (MT) by scoring and selecting the best candidate from a pool of generated translations. While known to be effective at the sentence level, its application to the increasingly prominent domain of document-level translation remains underexplored. In this work, we evaluate QE reranking performance on document-level (rather than the typical sentence-level) translation, using various learned and large language model (LLM)-based QE metrics. We find that with our best learned metric, SLIDE, BLEURT-20 scores improve by +2.00 with only two candidates, and by +5.09 with 32, across both decoder-only LLM models and encoder-decoder neural machine translation (NMT) models. Using the best LLM-based metric, GEMBA-DA, gains of +1.63 and +4.30 are achieved under the same conditions. Although gains shrink with longer inputs, reranking with 32 candidates yields improvements of +2.34 (SLIDE) and +1.40 (GEMBA-DA) on our longest documents (512-1024 source tokens). These findings demonstrate the practical value of document-level QE, with minimal runtime overhead given suitable translation models and hardware.
>
---
#### [new 005] A Novel Framework for Augmenting Rating Scale Tests with LLM-Scored Text Data
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于心理测量与自然语言处理交叉任务，旨在解决传统心理评估无法有效利用自然语言数据的问题。作者提出一种新框架，结合大语言模型（LLM）评分的文本数据与传统量表，提升测量精度。实验证明该方法在抑郁评估中显著增强测试效能。**

- **链接: [http://arxiv.org/pdf/2510.08663v1](http://arxiv.org/pdf/2510.08663v1)**

> **作者:** Joe Watson; Ivan O'Conner; Chia-Wen Chen; Luning Sun; Fang Luo; David Stillwell
>
> **摘要:** Psychological assessments typically rely on structured rating scales, which cannot incorporate the rich nuance of a respondent's natural language. This study leverages recent LLM advances to harness qualitative data within a novel conceptual framework, combining LLM-scored text and traditional rating-scale items to create an augmented test. We demonstrate this approach using depression as a case study, developing and assessing the framework on a real-world sample of upper secondary students (n=693) and corresponding synthetic dataset (n=3,000). On held-out test sets, augmented tests achieved statistically significant improvements in measurement precision and accuracy. The information gain from the LLM items was equivalent to adding between 6.3 (real data) and 16.0 (synthetic data) items to the original 19-item test. Our approach marks a conceptual shift in automated scoring that bypasses its typical bottlenecks: instead of relying on pre-labelled data or complex expert-created rubrics, we empirically select the most informative LLM scoring instructions based on calculations of item information. This framework provides a scalable approach for leveraging the growing stream of transcribed text to enhance traditional psychometric measures, and we discuss its potential utility in clinical health and beyond.
>
---
#### [new 006] One Sentence, Two Embeddings: Contrastive Learning of Explicit and Implicit Semantic Representations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决传统句子嵌入方法难以同时捕捉句子显性和隐性语义的问题。作者提出DualCSE方法，为每个句子生成两个嵌入向量，分别表示显性和隐性语义，实验证明其在下游任务中有效提升性能。**

- **链接: [http://arxiv.org/pdf/2510.09293v1](http://arxiv.org/pdf/2510.09293v1)**

> **作者:** Kohei Oda; Po-Min Chuang; Kiyoaki Shirai; Natthawut Kertkeidkachorn
>
> **摘要:** Sentence embedding methods have made remarkable progress, yet they still struggle to capture the implicit semantics within sentences. This can be attributed to the inherent limitations of conventional sentence embedding methods that assign only a single vector per sentence. To overcome this limitation, we propose DualCSE, a sentence embedding method that assigns two embeddings to each sentence: one representing the explicit semantics and the other representing the implicit semantics. These embeddings coexist in the shared space, enabling the selection of the desired semantics for specific purposes such as information retrieval and text classification. Experimental results demonstrate that DualCSE can effectively encode both explicit and implicit meanings and improve the performance of the downstream task.
>
---
#### [new 007] Mask Tokens as Prophet: Fine-Grained Cache Eviction for Efficient dLLM Inference
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决扩散大语言模型（dLLMs）在资源受限下处理长文本时缓存机制效率低的问题。作者提出了MaskKV，通过mask-token引导的缓存淘汰策略，减少内存使用并加速推理，在保留94%性能的同时实现31倍加速。**

- **链接: [http://arxiv.org/pdf/2510.09309v1](http://arxiv.org/pdf/2510.09309v1)**

> **作者:** Jianuo Huang; Yaojie Zhang; Yicun Yang; Benhao Huang; Biqing Qi; Dongrui Liu; Linfeng Zhang
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Diffusion large language models (dLLMs) present a promising alternative to dominant autoregressive models (ARMs) by the ability of parallel decoding at the expense of substantial computation and memory costs. Specifically, the cache mechanism for bidirectional attention in dLLMs demands large memory footprint, restricting their ability to handle long contexts under resource-limited settings. Existing cache eviction strategies are designed for ARMs and ignore the unique characteristics of dLLMs, thus leading to unsatisfactory performance. To address these challenges, we introduce MaskKV, a training-free cache eviction framework tailored to dLLMs, focusing on the effect of mask tokens in dLLMs. MaskKV is built on two key innovations: (1) a mask-query guided scoring mechanism that leverages attention weights to identify and evict less critical prompt tokens for each head; (2) an adaptive cache budgeting strategy that improves efficiency by reducing allocation in intermediate layers and concentrating resources on prompt-preferring heads. On LLaDA with MaskKV, compressing the KV cache to only 256 pairs (less than 5% of tokens) retains 94% of the full-cache performance on LongBench and achieves up to 31x acceleration at 32k prompt length. The code is publicly available at: https://github.com/jianuo-huang/MaskKV
>
---
#### [new 008] SOP-Maze: Evaluating Large Language Models on Complicated Business Standard Operating Procedures
- **分类: cs.CL**

- **简介: 该论文提出SOP-Maze基准，评估大语言模型在复杂商业标准操作流程中的表现。任务属指令跟随与决策，旨在解决模型在现实业务流程中推理能力不足的问题。工作包括构建含397个任务的数据集，分析模型错误类型，揭示当前模型在路线遵循、对话处理和计算推理方面的不足。**

- **链接: [http://arxiv.org/pdf/2510.08942v1](http://arxiv.org/pdf/2510.08942v1)**

> **作者:** Jiaming Wang; Zhe Tang; Yilin Jin; Peng Ding; Xiaoyu Li; Xuezhi Cao
>
> **摘要:** As large language models (LLMs) are widely deployed as domain-specific agents, many benchmarks have been proposed to evaluate their ability to follow instructions and make decisions in real-world scenarios. However, business scenarios often involve complex standard operating procedures (SOPs), and the evaluation of LLM capabilities in such contexts has not been fully explored. To bridge this gap, we propose SOP-Maze, a benchmark constructed from real-world business data and adapted into a collection of 397 tasks from 23 complex SOP scenarios. We further categorize SOP tasks into two broad classes: Lateral Root System (LRS), representing wide-option tasks that demand precise selection; and Heart Root System (HRS), which emphasizes deep logical reasoning with complex branches. Extensive experiments reveal that nearly all state-of-the-art models struggle with SOP-Maze. We conduct a comprehensive analysis and identify three key error categories: (i) route blindness: difficulty following procedures; (ii) conversational fragility: inability to handle real dialogue nuances; and (iii) calculation errors: mistakes in time or arithmetic reasoning under complex contexts. The systematic study explores LLM performance across SOP tasks that challenge both breadth and depth, offering new insights for improving model capabilities. We have open-sourced our work on https://github.com/ADoublLEN/SOP-Maze.
>
---
#### [new 009] Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究测试时扩展（TTS）在大语言模型中的安全性风险，指出候选多样性降低会显著增加不安全输出的可能性。通过提出RefDiv方法进行压力测试，发现现有安全机制难以检测此类问题，强调需设计更鲁棒的TTS策略。**

- **链接: [http://arxiv.org/pdf/2510.08592v1](http://arxiv.org/pdf/2510.08592v1)**

> **作者:** Shahriar Kabir Nahin; Hadi Askari; Muhao Chen; Anshuman Chhabra
>
> **摘要:** Test-Time Scaling (TTS) improves LLM reasoning by exploring multiple candidate responses and then operating over this set to find the best output. A tacit premise behind TTS is that sufficiently diverse candidate pools enhance reliability. In this work, we show that this assumption in TTS introduces a previously unrecognized failure mode. When candidate diversity is curtailed, even by a modest amount, TTS becomes much more likely to produce unsafe outputs. We present a reference-guided diversity reduction protocol (RefDiv) that serves as a diagnostic attack to stress test TTS pipelines. Through extensive experiments across four open-source models (Qwen3, Mistral, Llama3.1, Gemma3) and two widely used TTS strategies (Monte Carlo Tree Search and Best-of-N), constraining diversity consistently signifies the rate at which TTS produces unsafe results. The effect is often stronger than that produced by prompts directly with high adversarial intent scores. This observed phenomenon also transfers across TTS strategies and to closed-source models (e.g. OpenAI o3 and Gemini-2.5-Pro), thus indicating that this is a general and extant property of TTS rather than a model-specific artifact. Additionally, we find that numerous widely used safety guardrail classifiers (e.g. Llama-Guard and OpenAI Moderation API), are unable to flag the adversarial input prompts generated by RefDiv, demonstrating that existing defenses offer limited protection against this diversity-driven failure mode. Through this work, we hope to motivate future research on designing robust TTS strategies that are both effective and secure against diversity-targeted stress tests as illustrated by RefDiv.
>
---
#### [new 010] Coordinates from Context: Using LLMs to Ground Complex Location References
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于地理编码任务，旨在解决从复杂文本位置描述中提取地理坐标的问题。论文探索了大语言模型（LLM）在处理组合性位置引用中的能力，结合其地理空间知识与推理技能，提出了一种有效的地理编码策略，并验证了小规模微调模型的效果。**

- **链接: [http://arxiv.org/pdf/2510.08741v1](http://arxiv.org/pdf/2510.08741v1)**

> **作者:** Tessa Masis; Brendan O'Connor
>
> **备注:** Under review at ARR
>
> **摘要:** Geocoding is the task of linking a location reference to an actual geographic location and is essential for many downstream analyses of unstructured text. In this paper, we explore the challenging setting of geocoding compositional location references. Building on recent work demonstrating LLMs' abilities to reason over geospatial data, we evaluate LLMs' geospatial knowledge versus reasoning skills relevant to our task. Based on these insights, we propose an LLM-based strategy for geocoding compositional location references. We show that our approach improves performance for the task and that a relatively small fine-tuned LLM can achieve comparable performance with much larger off-the-shelf models.
>
---
#### [new 011] Mind-Paced Speaking: A Dual-Brain Approach to Real-Time Reasoning in Spoken Language Models
- **分类: cs.CL**

- **简介: 该论文属于语音语言模型任务，旨在解决实时语音生成中推理延迟高的问题。提出Mind-Paced Speaking（MPS）方法，采用双脑结构，分别负责推理与语音生成，实现实时高质量对话。实验表明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.09592v1](http://arxiv.org/pdf/2510.09592v1)**

> **作者:** Donghang Wu; Haoyang Zhang; Jun Chen; Xiangyu; Zhang; Hexin Liu; Eng Siong Chng; Fei Tian; Xuerui Yang; Xiangyu Zhang; Daxin Jiang; Gang Yu
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** Real-time Spoken Language Models (SLMs) struggle to leverage Chain-of-Thought (CoT) reasoning due to the prohibitive latency of generating the entire thought process sequentially. Enabling SLMs to think while speaking, similar to humans, is attracting increasing attention. We present, for the first time, Mind-Paced Speaking (MPS), a brain-inspired framework that enables high-fidelity, real-time reasoning. Similar to how humans utilize distinct brain regions for thinking and responding, we propose a novel dual-brain approach, employing a "Formulation Brain" for high-level reasoning to pace and guide a separate "Articulation Brain" for fluent speech generation. This division of labor eliminates mode-switching, preserving the integrity of the reasoning process. Experiments show that MPS significantly outperforms existing think-while-speaking methods and achieves reasoning performance comparable to models that pre-compute the full CoT before speaking, while drastically reducing latency. Under a zero-latency configuration, the proposed method achieves an accuracy of 92.8% on the mathematical reasoning task Spoken-MQA and attains a score of 82.5 on the speech conversation task URO-Bench. Our work effectively bridges the gap between high-quality reasoning and real-time interaction.
>
---
#### [new 012] DITING: A Multi-Agent Evaluation Framework for Benchmarking Web Novel Translation
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决现有翻译评估方法无法准确衡量网络小说翻译质量的问题。作者提出了DITING评估框架，涵盖六个维度，并构建了AgentEval和MetricAlign用于自动化评估。通过评估14个模型，发现中文训练的模型表现更优，DeepSeek-V3效果最佳。论文为网络小说翻译研究提供了新范式和公开资源。**

- **链接: [http://arxiv.org/pdf/2510.09116v1](http://arxiv.org/pdf/2510.09116v1)**

> **作者:** Enze Zhang; Jiaying Wang; Mengxi Xiao; Jifei Liu; Ziyan Kuang; Rui Dong; Youzhong Dong; Sophia Ananiadou; Min Peng; Qianqian Xie
>
> **摘要:** Large language models (LLMs) have substantially advanced machine translation (MT), yet their effectiveness in translating web novels remains unclear. Existing benchmarks rely on surface-level metrics that fail to capture the distinctive traits of this genre. To address these gaps, we introduce DITING, the first comprehensive evaluation framework for web novel translation, assessing narrative and cultural fidelity across six dimensions: idiom translation, lexical ambiguity, terminology localization, tense consistency, zero-pronoun resolution, and cultural safety, supported by over 18K expert-annotated Chinese-English sentence pairs. We further propose AgentEval, a reasoning-driven multi-agent evaluation framework that simulates expert deliberation to assess translation quality beyond lexical overlap, achieving the highest correlation with human judgments among seven tested automatic metrics. To enable metric comparison, we develop MetricAlign, a meta-evaluation dataset of 300 sentence pairs annotated with error labels and scalar quality scores. Comprehensive evaluation of fourteen open, closed, and commercial models reveals that Chinese-trained LLMs surpass larger foreign counterparts, and that DeepSeek-V3 delivers the most faithful and stylistically coherent translations. Our work establishes a new paradigm for exploring LLM-based web novel translation and provides public resources to advance future research.
>
---
#### [new 013] Multimodal Policy Internalization for Conversational Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出多模态策略内化任务（MPI），旨在将复杂的多模态策略融入模型参数中，以提升对话代理在推理时遵循策略的能力。为解决策略复杂、计算成本高及多模态策略研究不足的问题，作者构建了两个数据集，并提出TriMPI三阶段训练框架，结合持续预训练、监督微调与策略感知强化学习，提升模型准确性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.09474v1](http://arxiv.org/pdf/2510.09474v1)**

> **作者:** Zhenhailong Wang; Jiateng Liu; Amin Fazel; Ritesh Sarkhel; Xing Fan; Xiang Li; Chenlei Guo; Heng Ji; Ruhi Sarikaya
>
> **摘要:** Modern conversational agents like ChatGPT and Alexa+ rely on predefined policies specifying metadata, response styles, and tool-usage rules. As these LLM-based systems expand to support diverse business and user queries, such policies, often implemented as in-context prompts, are becoming increasingly complex and lengthy, making faithful adherence difficult and imposing large fixed computational costs. With the rise of multimodal agents, policies that govern visual and multimodal behaviors are critical but remain understudied. Prior prompt-compression work mainly shortens task templates and demonstrations, while existing policy-alignment studies focus only on text-based safety rules. We introduce Multimodal Policy Internalization (MPI), a new task that internalizes reasoning-intensive multimodal policies into model parameters, enabling stronger policy-following without including the policy during inference. MPI poses unique data and algorithmic challenges. We build two datasets spanning synthetic and real-world decision-making and tool-using tasks and propose TriMPI, a three-stage training framework. TriMPI first injects policy knowledge via continual pretraining, then performs supervised finetuning, and finally applies PolicyRollout, a GRPO-style reinforcement learning extension that augments rollouts with policy-aware responses for grounded exploration. TriMPI achieves notable gains in end-to-end accuracy, generalization, and robustness to forgetting. As the first work on multimodal policy internalization, we provide datasets, training recipes, and comprehensive evaluations to foster future research. Project page: https://mikewangwzhl.github.io/TriMPI.
>
---
#### [new 014] MOSAIC: Multi-agent Orchestration for Task-Intelligent Scientific Coding
- **分类: cs.CL**

- **简介: 该论文提出MOSAIC，一个多智能体框架，用于解决科学编程任务。科学编程需要结合领域知识、推理和算法迭代，传统方法难以应对复杂、多步骤的问题。MOSAIC通过学生-教师范式实现自我反思、代码生成与调试，结合上下文窗口机制减少错误，提升了科学代码生成的准确性与可解释性。**

- **链接: [http://arxiv.org/pdf/2510.08804v1](http://arxiv.org/pdf/2510.08804v1)**

> **作者:** Siddeshwar Raghavan; Tanwi Mallick
>
> **摘要:** We present MOSAIC, a multi-agent Large Language Model (LLM) framework for solving challenging scientific coding tasks. Unlike general-purpose coding, scientific workflows require algorithms that are rigorous, interconnected with deep domain knowledge, and incorporate domain-specific reasoning, as well as algorithm iteration without requiring I/O test cases. Many scientific problems also require a sequence of subproblems to be solved, leading to the final desired result. MOSAIC is designed as a training-free framework with specially designed agents to self-reflect, create the rationale, code, and debug within a student-teacher paradigm to address the challenges of scientific code generation. This design facilitates stepwise problem decomposition, targeted error correction, and, when combined with our Consolidated Context Window (CCW), mitigates LLM hallucinations when solving complex scientific tasks involving chained subproblems. We evaluate MOSAIC on scientific coding benchmarks and demonstrate that our specialized agentic framework outperforms existing approaches in terms of accuracy, robustness, and interpretability.
>
---
#### [new 015] LLaMAX2: Your Translation-Enhanced Model also Performs Well in Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决翻译增强的语言模型在推理任务上表现不佳的问题。作者提出了一种新的训练方法，通过在指令模型基础上仅对平行数据进行层选择性调优，构建了Qwen3-XPlus模型。该模型在翻译性能大幅提升的同时，在多语言任务和推理任务上也表现出色，尤其在低资源语言上效果显著。**

- **链接: [http://arxiv.org/pdf/2510.09189v1](http://arxiv.org/pdf/2510.09189v1)**

> **作者:** Changjiang Gao; Zixian Huang; Jingyang Gong; Shujian Huang; Lei Li; Fei Yuan
>
> **摘要:** General Large Language Models (LLMs) excel in reasoning, but those enhanced for translation struggle with reasoning tasks. To address this, we propose a novel translationenhanced recipe that begins with instruct models and applies layer-selective tuning only on parallel data. Following this pipeline, we introduce the Qwen3-XPlus models, which demonstrate significant improvements in translation performance across both high- and lowresource languages, achieving 15+ spBLEU and 40+ xComet in low-resource languages, like Swahili. Interestingly, training only with small parallel datasets, Qwen3-XPlus achieves an average improvement of 1+ points on 7 multilingual tasks while maintaining proficiency comparable to the Qwen3 instruct model in 15 popular reasoning datasets. This work offers a promising approach to multilingual enhancement, significantly reducing complexity and enhancing accessibility for a wider range of languages. The code and model are publicly available.
>
---
#### [new 016] DSPO: Stable and Efficient Policy Optimization for Agentic Search and Reasoning
- **分类: cs.CL**

- **简介: 该论文属于强化学习与大语言模型（LLM）代理任务，旨在解决当前模型在复杂任务中搜索与推理能力不足、训练不稳定的问题。作者提出DSPO算法，通过序列级优化与动态样本筛选，实现稳定高效的策略优化，使模型在多跳问答等任务上显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.09255v1](http://arxiv.org/pdf/2510.09255v1)**

> **作者:** Chenyang Gu; Yewen Pu; Bruce Yang; Xiaofan Li; Huan Gao
>
> **摘要:** Enhancing LLMs with the ability to actively search external knowledge is crucial for complex and real-world tasks. Current approaches either rely on prompting to elicit the model's innate agent capabilities, or suffer from performance ceilings and collapse when applying RL to complex interactive tasks, leaving their true agentic potential untapped. To address this, we introduce \textbf{D}ynamic-filter \textbf{S}equence-level \textbf{P}olicy \textbf{O}ptimization (DSPO), an improved RL algorithm designed for robust agent training through sequence-level optimization and dynamic sample filtering. We train our model purely through RL to interleave multi-turn search and reasoning, obviating the need for supervised demonstration data. Across multiple QA benchmarks, our DSPO-trained 7B model improves over a comparable previous work by \textbf{34.1\%}, and even outperforms the 14B model from previous work in complex multihop QA such as HotpotQA by nearly \textbf{9\% relative}, maintaining exceptional training stability.
>
---
#### [new 017] FinAuditing: A Financial Taxonomy-Structured Multi-Document Benchmark for Evaluating LLMs
- **分类: cs.CL; cs.CE; cs.IR**

- **简介: 该论文属于金融审计任务，旨在解决大型语言模型（LLM）在结构化、依赖性强的财务文档中推理能力不足的问题。作者构建了首个基于财务分类标准、结构感知的多文档基准FinAuditing，包含三个子任务：语义一致性（FinSM）、关系一致性（FinRE）和数值一致性（FinMR），并提出统一评估框架。实验表明现有LLM在层级结构文档中表现显著下降，揭示其在财务推理中的系统性局限。**

- **链接: [http://arxiv.org/pdf/2510.08886v1](http://arxiv.org/pdf/2510.08886v1)**

> **作者:** Yan Wang; Keyi Wang; Shanshan Yang; Jaisal Patel; Jeff Zhao; Fengran Mo; Xueqing Peng; Lingfei Qian; Jimin Huang; Guojun Xiong; Xiao-Yang Liu; Jian-Yun Nie
>
> **摘要:** The complexity of the Generally Accepted Accounting Principles (GAAP) and the hierarchical structure of eXtensible Business Reporting Language (XBRL) filings make financial auditing increasingly difficult to automate and verify. While large language models (LLMs) have demonstrated strong capabilities in unstructured text understanding, their ability to reason over structured, interdependent, and taxonomy-driven financial documents remains largely unexplored. To fill this gap, we introduce FinAuditing, the first taxonomy-aligned, structure-aware, multi-document benchmark for evaluating LLMs on financial auditing tasks. Built from real US-GAAP-compliant XBRL filings, FinAuditing defines three complementary subtasks, FinSM for semantic consistency, FinRE for relational consistency, and FinMR for numerical consistency, each targeting a distinct aspect of structured auditing reasoning. We further propose a unified evaluation framework integrating retrieval, classification, and reasoning metrics across these subtasks. Extensive zero-shot experiments on 13 state-of-the-art LLMs reveal that current models perform inconsistently across semantic, relational, and mathematical dimensions, with accuracy drops of up to 60-90% when reasoning over hierarchical multi-document structures. Our findings expose the systematic limitations of modern LLMs in taxonomy-grounded financial reasoning and establish FinAuditing as a foundation for developing trustworthy, structure-aware, and regulation-aligned financial intelligence systems. The benchmark dataset is available at Hugging Face.
>
---
#### [new 018] How Many Code and Test Cases Are Enough? Evaluating Test Cases Generation from a Binary-Matrix Perspective
- **分类: cs.CL**

- **简介: 该论文属于软件测试任务，旨在解决自动生成测试用例的评估问题。现有方法计算成本高、评分膨胀且偏向简单错误。作者提出WrongSelect算法，通过构建二进制代码-测试矩阵的最优诊断基，确定最小错误模式和测试用例集，从而创建更紧凑、多样、抗膨胀的基准TC-Bench。**

- **链接: [http://arxiv.org/pdf/2510.08720v1](http://arxiv.org/pdf/2510.08720v1)**

> **作者:** Xianzhen Luo; Jinyang Huang; Wenzhen Zheng; Qingfu Zhu; Mingzheng Xu; Yiheng Xu; Yuantao Fan; Libo Qin; Wanxiang Che
>
> **备注:** Work in Progress
>
> **摘要:** Evaluating test cases automatically generated by Large Language Models (LLMs) is a critical yet challenging task. Existing benchmarks suffer from high computational costs, score inflation, and a bias towards trivial bugs over rare, critical faults. In this work, we ask two fundamental questions: (1) What is the minimal set of wrong codes sufficient to represent the entire error space? and (2) What is the minimal set of test cases needed to distinguish them? We introduce a framework that formalizes benchmark construction as finding an optimal diagnostic basis in a binary code-test matrix. The rank of this matrix specifies the minimal number of independent error patterns (wrong codes) and provides a tight upper bound on the number of test cases required for complete fault coverage. Our objective is to identify a basis of size equal to the matrix rank that maximizes internal diversity. To tackle this NP-hard problem, we propose WrongSelect, an efficient approximation algorithm to select maximally diverse wrong codes. Applying this framework to millions of competitive programming submissions, we construct TC-Bench, a compact, diverse, and inflation-resistant benchmark. Extensive experiments show that even the most advanced test case generation methods achieve only ~60% exclusion rates on TC-Bench, exposing a significant gap in their diagnostic power. Our dataset is available at: https://huggingface.co/datasets/Luoberta/TC-Bench and our code is at: https://github.com/Luowaterbi/TC-Bench.
>
---
#### [new 019] The Model's Language Matters: A Comparative Privacy Analysis of LLMs
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于隐私分析任务，旨在研究语言对大语言模型隐私泄露的影响。论文通过分析英语、西班牙语、法语和意大利语医学语料，量化六种语言指标并评估三种攻击向量，发现语言冗余性和分词粒度影响隐私泄露程度，揭示语言结构与隐私风险的关系。**

- **链接: [http://arxiv.org/pdf/2510.08813v1](http://arxiv.org/pdf/2510.08813v1)**

> **作者:** Abhishek K. Mishra; Antoine Boutet; Lucas Magnana
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed across multilingual applications that handle sensitive data, yet their scale and linguistic variability introduce major privacy risks. Mostly evaluated for English, this paper investigates how language structure affects privacy leakage in LLMs trained on English, Spanish, French, and Italian medical corpora. We quantify six linguistic indicators and evaluate three attack vectors: extraction, counterfactual memorization, and membership inference. Results show that privacy vulnerability scales with linguistic redundancy and tokenization granularity: Italian exhibits the strongest leakage, while English shows higher membership separability. In contrast, French and Spanish display greater resilience due to higher morphological complexity. Overall, our findings provide the first quantitative evidence that language matters in privacy leakage, underscoring the need for language-aware privacy-preserving mechanisms in LLM deployments.
>
---
#### [new 020] Gender Bias in Large Language Models for Healthcare: Assignment Consistency and Clinical Implications
- **分类: cs.CL**

- **简介: 该论文研究大型语言模型在医疗领域中的性别偏见问题，分析不同性别设定下模型对诊断一致性及患者性别相关性判断的影响，揭示模型在性别角色分配上的一致性缺陷，强调需关注AI在医疗决策中的公平性。**

- **链接: [http://arxiv.org/pdf/2510.08614v1](http://arxiv.org/pdf/2510.08614v1)**

> **作者:** Mingxuan Liu; Yuhe Ke; Wentao Zhu; Mayli Mertens; Yilin Ning; Jingchi Liao; Chuan Hong; Daniel Shu Wei Ting; Yifan Peng; Danielle S. Bitterman; Marcus Eng Hock Ong; Nan Liu
>
> **摘要:** The integration of large language models (LLMs) into healthcare holds promise to enhance clinical decision-making, yet their susceptibility to biases remains a critical concern. Gender has long influenced physician behaviors and patient outcomes, raising concerns that LLMs assuming human-like roles, such as clinicians or medical educators, may replicate or amplify gender-related biases. Using case studies from the New England Journal of Medicine Challenge (NEJM), we assigned genders (female, male, or unspecified) to multiple open-source and proprietary LLMs. We evaluated their response consistency across LLM-gender assignments regarding both LLM-based diagnosis and models' judgments on the clinical relevance or necessity of patient gender. In our findings, diagnoses were relatively consistent across LLM genders for most models. However, for patient gender's relevance and necessity in LLM-based diagnosis, all models demonstrated substantial inconsistency across LLM genders, particularly for relevance judgements. Some models even displayed a systematic female-male disparity in their interpretation of patient gender. These findings present an underexplored bias that could undermine the reliability of LLMs in clinical practice, underscoring the need for routine checks of identity-assignment consistency when interacting with LLMs to ensure reliable and equitable AI-supported clinical care.
>
---
#### [new 021] Mitigating Overthinking through Reasoning Shaping
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大推理模型在问题求解中产生的“过思考”问题。通过提出一种新的正则化方法GRSP，在保持模型性能的同时提高推理效率，减少计算开销，并提升训练稳定性。**

- **链接: [http://arxiv.org/pdf/2510.09535v1](http://arxiv.org/pdf/2510.09535v1)**

> **作者:** Feifan Song; Shaohang Wei; Bofei Gao; Yejie Wang; Wen Luo; Wei Li; Linli Yao; Weimin Xiong; Liang Chen; Tianyu Liu; Houfeng Wang
>
> **摘要:** Large reasoning models (LRMs) boosted by Reinforcement Learning from Verifier Reward (RLVR) have shown great power in problem solving, yet they often cause overthinking: excessive, meandering reasoning that inflates computational cost. Prior designs of penalization in RLVR manage to reduce token consumption while often harming model performance, which arises from the oversimplicity of token-level supervision. In this paper, we argue that the granularity of supervision plays a crucial role in balancing efficiency and accuracy, and propose Group Relative Segment Penalization (GRSP), a step-level method to regularize reasoning. Since preliminary analyses show that reasoning segments are strongly correlated with token consumption and model performance, we design a length-aware weighting mechanism across segment clusters. Extensive experiments demonstrate that GRSP achieves superior token efficiency without heavily compromising accuracy, especially the advantages with harder problems. Moreover, GRSP stabilizes RL training and scales effectively across model sizes.
>
---
#### [new 022] dInfer: An Efficient Inference Framework for Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文提出dInfer，一个高效的扩散语言模型（dLLM）推理框架，旨在解决dLLM缺乏标准化高效推理系统的问题。通过模块化设计与算法优化，dInfer在多个基准测试中实现显著速度提升，同时保持输出质量。**

- **链接: [http://arxiv.org/pdf/2510.08666v1](http://arxiv.org/pdf/2510.08666v1)**

> **作者:** Yuxin Ma; Lun Du; Lanning Wei; Kun Chen; Qian Xu; Kangyu Wang; Guofeng Feng; Guoshan Lu; Lin Liu; Xiaojing Qi; Xinyuan Zhang; Zhen Tao; Haibo Feng; Ziyun Jiang; Ying Xu; Zenan Huang; Yihong Zhuang; Haokai Xu; Jiaqi Hu; Zhenzhong Lan; Junbo Zhao; Jianguo Li; Da Zheng
>
> **摘要:** Diffusion-based large language models (dLLMs) have emerged as a promising alternative to autoregressive (AR) LLMs, leveraging denoising-based generation to enable inherent parallelism. Even more and more open-sourced dLLM models emerge, yet their widespread adoption remains constrained by the lack of a standardized and efficient inference framework. We present dInfer, an efficient and extensible framework for dLLM inference. dInfer decomposes the inference pipeline into four modular components-model, diffusion iteration manager, decoding strategy, and KV-cache manager-and integrates novel algorithms for each component alongside system-level optimizations. Through this combination of algorithmic innovations and system enhancements, dInfer achieves substantial efficiency gains without compromising output quality on LLaDA-MoE. At batch size 1, it surpasses 1,100 tokens per second on HumanEval and averages over 800 tokens per second across six benchmarks on $8\times$ H800 GPUs. Compared to prior systems, dInfer delivers $10\times$ speedup over Fast-dLLM while maintaining similar model performance. Even compared with AR models (with a comparable number of activation parameters and performance) QWen2.5-3B, which is highly optimized with latest vLLM inference engine, dInfer still deliverers $2$-$3\times$ speedup. The implementation of dInfer is open-sourced at https://github.com/inclusionAI/dInfer.
>
---
#### [new 023] Search-on-Graph: Iterative Informed Navigation for Large Language Model Reasoning on Knowledge Graphs
- **分类: cs.CL**

- **简介: 论文提出Search-onGraph（SoG）框架，用于大型语言模型在知识图谱上进行推理。任务是知识图谱问答（KGQA），旨在解决多跳问题中语言模型知识不足、易幻觉和更新滞后的问题。工作设计了一个迭代搜索函数，使模型基于当前实体关系决策下一步，适应不同知识图谱结构，并有效处理高连接节点。**

- **链接: [http://arxiv.org/pdf/2510.08825v1](http://arxiv.org/pdf/2510.08825v1)**

> **作者:** Jia Ao Sun; Hao Yu; Fabrizio Gotti; Fengran Mo; Yihong Wu; Yuchen Hui; Jian-Yun Nie
>
> **摘要:** Large language models (LLMs) have demonstrated impressive reasoning abilities yet remain unreliable on knowledge-intensive, multi-hop questions -- they miss long-tail facts, hallucinate when uncertain, and their internal knowledge lags behind real-world change. Knowledge graphs (KGs) offer a structured source of relational evidence, but existing KGQA methods face fundamental trade-offs: compiling complete SPARQL queries without knowing available relations proves brittle, retrieving large subgraphs introduces noise, and complex agent frameworks with parallel exploration exponentially expand search spaces. To address these limitations, we propose Search-on-Graph (SoG), a simple yet effective framework that enables LLMs to perform iterative informed graph navigation using a single, carefully designed \textsc{Search} function. Rather than pre-planning paths or retrieving large subgraphs, SoG follows an ``observe-then-navigate'' principle: at each step, the LLM examines actual available relations from the current entity before deciding on the next hop. This approach further adapts seamlessly to different KG schemas and handles high-degree nodes through adaptive filtering. Across six KGQA benchmarks spanning Freebase and Wikidata, SoG achieves state-of-the-art performance without fine-tuning. We demonstrate particularly strong gains on Wikidata benchmarks (+16\% improvement over previous best methods) alongside consistent improvements on Freebase benchmarks.
>
---
#### [new 024] AutoPR: Let's Automate Your Academic Promotion!
- **分类: cs.CL**

- **简介: 该论文提出“自动学术推广（AutoPR）”任务，旨在将研究论文转化为准确、吸引人的宣传内容，以提升学术成果的可见性和影响力。论文构建了多模态基准PRBench，并开发了多智能体框架PRAgent，实现内容提取、协同生成和平台适配，显著提升推广效果。**

- **链接: [http://arxiv.org/pdf/2510.09558v1](http://arxiv.org/pdf/2510.09558v1)**

> **作者:** Qiguang Chen; Zheng Yan; Mingda Yang; Libo Qin; Yixin Yuan; Hanjing Li; Jinhao Liu; Yiyan Ji; Dengyun Peng; Jiannan Guan; Mengkang Hu; Yantao Du; Wanxiang Che
>
> **备注:** Preprint. Code: https://github.com/LightChen2333/AutoPR . Benchmark: https://huggingface.co/datasets/yzweak/PRBench
>
> **摘要:** As the volume of peer-reviewed research surges, scholars increasingly rely on social platforms for discovery, while authors invest considerable effort in promoting their work to ensure visibility and citations. To streamline this process and reduce the reliance on human effort, we introduce Automatic Promotion (AutoPR), a novel task that transforms research papers into accurate, engaging, and timely public content. To enable rigorous evaluation, we release PRBench, a multimodal benchmark that links 512 peer-reviewed articles to high-quality promotional posts, assessing systems along three axes: Fidelity (accuracy and tone), Engagement (audience targeting and appeal), and Alignment (timing and channel optimization). We also introduce PRAgent, a multi-agent framework that automates AutoPR in three stages: content extraction with multimodal preparation, collaborative synthesis for polished outputs, and platform-specific adaptation to optimize norms, tone, and tagging for maximum reach. When compared to direct LLM pipelines on PRBench, PRAgent demonstrates substantial improvements, including a 604% increase in total watch time, a 438% rise in likes, and at least a 2.9x boost in overall engagement. Ablation studies show that platform modeling and targeted promotion contribute the most to these gains. Our results position AutoPR as a tractable, measurable research problem and provide a roadmap for scalable, impactful automated scholarly communication.
>
---
#### [new 025] Pattern Enhanced Multi-Turn Jailbreaking: Exploiting Structural Vulnerabilities in Large Language Models
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文研究多轮越狱攻击对大语言模型的影响，旨在揭示对话模式与模型安全漏洞之间的关系。作者提出了PE-CoA框架，包含五种对话模式，用于构建有效的多轮越狱攻击。他们评估了十二个大语言模型，覆盖十个危害类别，发现模型在不同对话模式下表现出特定的弱点，且模型家族具有相似的失败模式。论文任务属于安全攻击分析，工作目标是揭示模型结构性漏洞并推动防御方法的发展。**

- **链接: [http://arxiv.org/pdf/2510.08859v1](http://arxiv.org/pdf/2510.08859v1)**

> **作者:** Ragib Amin Nihal; Rui Wen; Kazuhiro Nakadai; Jun Sakuma
>
> **摘要:** Large language models (LLMs) remain vulnerable to multi-turn jailbreaking attacks that exploit conversational context to bypass safety constraints gradually. These attacks target different harm categories (like malware generation, harassment, or fraud) through distinct conversational approaches (educational discussions, personal experiences, hypothetical scenarios). Existing multi-turn jailbreaking methods often rely on heuristic or ad hoc exploration strategies, providing limited insight into underlying model weaknesses. The relationship between conversation patterns and model vulnerabilities across harm categories remains poorly understood. We propose Pattern Enhanced Chain of Attack (PE-CoA), a framework of five conversation patterns to construct effective multi-turn jailbreaks through natural dialogue. Evaluating PE-CoA on twelve LLMs spanning ten harm categories, we achieve state-of-the-art performance, uncovering pattern-specific vulnerabilities and LLM behavioral characteristics: models exhibit distinct weakness profiles where robustness to one conversational pattern does not generalize to others, and model families share similar failure modes. These findings highlight limitations of safety training and indicate the need for pattern-aware defenses. Code available on: https://github.com/Ragib-Amin-Nihal/PE-CoA
>
---
#### [new 026] Hierarchical Indexing with Knowledge Enrichment for Multilingual Video Corpus Retrieval
- **分类: cs.CL**

- **简介: 该论文属于多语言视频语料检索任务，旨在解决跨语言、多跳复杂问题的视频检索效率与精度问题。论文提出一种多阶段框架，结合多语言语义、领域术语与高效长文本处理，通过知识图谱增强、分层索引与轻量大模型重排序，实现精准且可扩展的多语言医学视频检索。**

- **链接: [http://arxiv.org/pdf/2510.09553v1](http://arxiv.org/pdf/2510.09553v1)**

> **作者:** Yu Wang; Tianhao Tan; Yifei Wang
>
> **备注:** Accepted to NLPCC 2025 (Springer), to appear November 2025
>
> **摘要:** Retrieving relevant instructional videos from multilingual medical archives is crucial for answering complex, multi-hop questions across language boundaries. However, existing systems either compress hour-long videos into coarse embeddings or incur prohibitive costs for fine-grained matching. We tackle the Multilingual Video Corpus Retrieval (mVCR) task in the NLPCC-2025 M4IVQA challenge with a multi-stage framework that integrates multilingual semantics, domain terminology, and efficient long-form processing. Video subtitles are divided into semantically coherent chunks, enriched with concise knowledge-graph (KG) facts, and organized into a hierarchical tree whose node embeddings are generated by a language-agnostic multilingual encoder. At query time, the same encoder embeds the input question; a coarse-to-fine tree search prunes irrelevant branches, and only the top-ranked chunks are re-scored by a lightweight large language model (LLM). This design avoids exhaustive cross-encoder scoring while preserving chunk-level precision. Experiments on the mVCR test set demonstrate state-of-the-art performance, and ablation studies confirm the complementary contributions of KG enrichment, hierarchical indexing, and targeted LLM re-ranking. The proposed method offers an accurate and scalable solution for multilingual retrieval in specialized medical video collections.
>
---
#### [new 027] Upfront Chain-of-Thought: A Cooperative Framework for Chain-of-Thought Compression
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理过程中长链思维（CoT）带来的高计算成本和延迟问题。论文提出Upfront CoT（UCoT）框架，通过小模型（压缩器）生成前置思维嵌入，辅助大模型（执行器）缩短推理路径。实验表明该方法显著减少GSM8K数据集上的Token使用量并提升性能。**

- **链接: [http://arxiv.org/pdf/2510.08647v1](http://arxiv.org/pdf/2510.08647v1)**

> **作者:** Chengzhengxu Li; Xiaoming Liu; Zhaohan Zhang; Shaochu Zhang; Shengchao Liu; Guoxin Ma; Yu Lan; Chao Shen
>
> **备注:** ACL2026 Under Review
>
> **摘要:** Recent developments have enabled advanced reasoning in Large Language Models (LLMs) via long Chain-of-Thought (CoT), while long CoT suffers from high computational costs and significant latency losses owing to the autoregressive nature of generative LLMs. CoT compression aims to improve efficiency in the reasoning process by reducing output length. Previous works trade reasoning efficiency by either laborious discrete prompt designing or the construction of external compressed CoT datasets that sacrifice key reasoning details. In this work, we propose Upfront CoT (UCoT): an efficient reasoning framework with upfront thought embedding to automate CoT compression. UCoT is a cooperative workflow involving a small model (compressor) and a large model (executor). The first stage of UCoT trains compressor to generate upfront thought embeddings rich in reasoning information for the executor, avoiding the drawbacks of manually designed prompts. The second stage optimizes executor to utilize upfront thought embeddings to derive the correct answer with short reasoning, using a reward mechanism. Extensive experiments show that UCoT maintains the powerful reasoning ability of executor while significantly reducing the length of CoT. It is worth mentioning that when applying UCoT to the Qwen2.5-7B-Instruct model, the usage of tokens on GSM8K dataset is reduced by 50\%, while the performance is 3.08\% higher than that of the state-of-the-art (SOTA) method. The code and dataset are in supplementary material.
>
---
#### [new 028] MaP: A Unified Framework for Reliable Evaluation of Pre-training Dynamics
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决大语言模型预训练过程中评估结果不稳定的问题。作者提出MaP框架，结合模型权重平均与Pass@k指标，降低训练与评估噪声，提升评估可靠性与一致性。**

- **链接: [http://arxiv.org/pdf/2510.09295v1](http://arxiv.org/pdf/2510.09295v1)**

> **作者:** Jiapeng Wang; Changxin Tian; Kunlong Chen; Ziqi Liu; Jiaxin Mao; Wayne Xin Zhao; Zhiqiang Zhang; Jun Zhou
>
> **摘要:** Reliable evaluation is fundamental to the progress of Large Language Models (LLMs), yet the evaluation process during pre-training is plagued by significant instability that obscures true learning dynamics. In this work, we systematically diagnose this instability, attributing it to two distinct sources: \textit{Parameter Instability} from training stochasticity and \textit{Evaluation Instability} from noisy measurement protocols. To counteract both sources of noise, we introduce \textbf{MaP}, a dual-pronged framework that synergistically integrates checkpoint \underline{M}erging \underline{a}nd the \underline{P}ass@k metric. Checkpoint merging smooths the parameter space by averaging recent model weights, while Pass@k provides a robust, low-variance statistical estimate of model capability. Extensive experiments show that MaP yields significantly smoother performance curves, reduces inter-run variance, and ensures more consistent model rankings. Ultimately, MaP provides a more reliable and faithful lens for observing LLM training dynamics, laying a crucial empirical foundation for LLM research.
>
---
#### [new 029] Beyond Single-Granularity Prompts: A Multi-Scale Chain-of-Thought Prompt Learning for Graph
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于图神经网络中的提示学习任务，旨在解决现有方法仅依赖单一粒度（如节点或子图级）生成提示、忽略图数据多尺度结构信息的问题。作者提出了一种多尺度图思维链（MSGCOT）框架，通过轻量级粗化网络提取多尺度特征，并动态融合粗到细的信息，提升提示语义多样性与模型性能，尤其在少样本场景下表现优异。**

- **链接: [http://arxiv.org/pdf/2510.09394v1](http://arxiv.org/pdf/2510.09394v1)**

> **作者:** Ziyu Zheng; Yaming Yang; Ziyu Guan; Wei Zhao; Xinyan Huang; Weigang Lu
>
> **备注:** under review
>
> **摘要:** The "pre-train, prompt'' paradigm, designed to bridge the gap between pre-training tasks and downstream objectives, has been extended from the NLP domain to the graph domain and has achieved remarkable progress. Current mainstream graph prompt-tuning methods modify input or output features using learnable prompt vectors. However, existing approaches are confined to single-granularity (e.g., node-level or subgraph-level) during prompt generation, overlooking the inherently multi-scale structural information in graph data, which limits the diversity of prompt semantics. To address this issue, we pioneer the integration of multi-scale information into graph prompt and propose a Multi-Scale Graph Chain-of-Thought (MSGCOT) prompting framework. Specifically, we design a lightweight, low-rank coarsening network to efficiently capture multi-scale structural features as hierarchical basis vectors for prompt generation. Subsequently, mimicking human cognition from coarse-to-fine granularity, we dynamically integrate multi-scale information at each reasoning step, forming a progressive coarse-to-fine prompt chain. Extensive experiments on eight benchmark datasets demonstrate that MSGCOT outperforms the state-of-the-art single-granularity graph prompt-tuning method, particularly in few-shot scenarios, showcasing superior performance.
>
---
#### [new 030] Do LLMs Know They Are Being Tested? Evaluation Awareness and Incentive-Sensitive Failures in GPT-OSS-20B
- **分类: cs.CL**

- **简介: 论文研究大语言模型在测试中是否具备评估感知，分析评估提示是否影响其表现。通过对比不同提示框架下的表现，发现测试提示会显著影响模型输出风格与准确性。提出可复用的评估框架，为提升模型实际部署效果提供指导。**

- **链接: [http://arxiv.org/pdf/2510.08624v1](http://arxiv.org/pdf/2510.08624v1)**

> **作者:** Nisar Ahmed; Muhammad Imran Zaman; Gulshan Saleem; Ali Hassan
>
> **摘要:** Benchmarks for large language models (LLMs) often rely on rubric-scented prompts that request visible reasoning and strict formatting, whereas real deployments demand terse, contract-bound answers. We investigate whether such "evaluation scent" inflates measured performance without commensurate capability gains. Using a single open-weights model (GPT-OSS-20B), we run six paired A/B scenarios that hold task content and decoding fixed while varying framing (evaluation-oriented vs. real-world) and reasoning depth (Medium/High): deterministic math, strict code-fix, citation generation, incentive flips (caution vs. competence), CoT visibility, and multilingual (Urdu) headers. Deterministic validators compute accuracy, answer-only compliance, hedging/refusals, chain-of-thought (CoT) length, and schema compliance, with pre-registered deltas and composite indices. Across scenarios, evaluation framing reliably inflates CoT (hundreds to >1000 characters) and reduces answer-only compliance, with limited or inconsistent accuracy gains. In structured outputs, it improves wrappers (e.g., fenced blocks, enumerated lists) but not regex-validated substance. Incentive wording reweights error composition: praising caution modestly improves accuracy at high reasoning and reduces wrong-but-confident errors, whereas praising competence yields terser but riskier outputs. Urdu rubric headers reproduce these signatures and can decrease accuracy at higher reasoning depth, indicating multilingual parity risks. We provide a reproducible A/B framework (prompt banks, validators, per-run scores, scripts; versioned DOI) and practical guidance: neutral phrasing or dual-framing checks, contract-aware grading, style-delta reporting, confidence governance, and multilingual dashboards to ensure that benchmark gains reflect deployable capability.
>
---
#### [new 031] Centering Emotion Hotspots: Multimodal Local-Global Fusion and Cross-Modal Alignment for Emotion Recognition in Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感识别任务，旨在解决对话中情感证据稀疏、局部化和模态异步的问题。作者提出一种新模型，通过检测多模态情感热点，结合局部-全局融合和跨模态对齐方法，提升了情感识别效果。实验表明该方法在多个基准数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.08606v1](http://arxiv.org/pdf/2510.08606v1)**

> **作者:** Yu Liu; Hanlei Shi; Haoxun Li; Yuqing Sun; Yuxuan Ding; Linlin Gong; Leyuan Qu; Taihao Li
>
> **备注:** Under review for ICASSP 2026
>
> **摘要:** Emotion Recognition in Conversations (ERC) is hard because discriminative evidence is sparse, localized, and often asynchronous across modalities. We center ERC on emotion hotspots and present a unified model that detects per-utterance hotspots in text, audio, and video, fuses them with global features via Hotspot-Gated Fusion, and aligns modalities using a routed Mixture-of-Aligners; a cross-modal graph encodes conversational structure. This design focuses modeling on salient spans, mitigates misalignment, and preserves context. Experiments on standard ERC benchmarks show consistent gains over strong baselines, with ablations confirming the contributions of HGF and MoA. Our results point to a hotspot-centric view that can inform future multimodal learning, offering a new perspective on modality fusion in ERC.
>
---
#### [new 032] A Comprehensive Evaluation of Multilingual Chain-of-Thought Reasoning: Performance, Consistency, and Faithfulness Across Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估多语言链式推理（CoT）的表现。它研究了大型推理模型在不同语言下的推理性能、一致性与忠实度，通过语言合规性、交叉语言一致性及扰动实验揭示了模型对语言的偏好与依赖程度。**

- **链接: [http://arxiv.org/pdf/2510.09555v1](http://arxiv.org/pdf/2510.09555v1)**

> **作者:** Raoyuan Zhao; Yihong Liu; Hinrich Schütze; Michael A. Hedderich
>
> **备注:** preprint
>
> **摘要:** Large reasoning models (LRMs) increasingly rely on step-by-step Chain-of-Thought (CoT) reasoning to improve task performance, particularly in high-resource languages such as English. While recent work has examined final-answer accuracy in multilingual settings, the thinking traces themselves, i.e., the intermediate steps that lead to the final answer, remain underexplored. In this paper, we present the first comprehensive study of multilingual CoT reasoning, evaluating three key dimensions: performance, consistency, and faithfulness. We begin by measuring language compliance, answer accuracy, and answer consistency when LRMs are explicitly instructed or prompt-hacked to think in a target language, revealing strong language preferences and divergent performance across languages. Next, we assess crosslingual consistency of thinking traces by interchanging them between languages. We find that the quality and effectiveness of thinking traces vary substantially depending on the prompt language. Finally, we adapt perturbation-based techniques -- i.e., truncation and error injection -- to probe the faithfulness of thinking traces across languages, showing that models rely on traces to varying degrees. We release our code and data to support future research.
>
---
#### [new 033] Token-Level Policy Optimization: Linking Group-Level Rewards to Token-Level Aggregation via Markov Likelihood
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的数学推理能力。现有方法如GRPO因链式思维任务中稀疏的token奖励而面临熵崩溃或模型崩溃问题。论文提出TEPO框架，通过马尔可夫似然将组级奖励与token级聚合关联，解决了奖励稀疏性问题，提升了训练稳定性和性能。**

- **链接: [http://arxiv.org/pdf/2510.09369v1](http://arxiv.org/pdf/2510.09369v1)**

> **作者:** Xingyu Lin; Yilin Wen; En Wang; Du Su; Wenbin Liu; Chenfu Bao; Zhonghou Lv
>
> **摘要:** Group Relative Policy Optimization (GRPO) has significantly advanced the reasoning ability of large language models (LLMs), particularly by boosting their mathematical performance. However, GRPO and related entropy-regularization methods still face challenges rooted in the sparse token rewards inherent to chain-of-thought (CoT). Current approaches often rely on undifferentiated token-level entropy adjustments, which frequently lead to entropy collapse or model collapse. In this work, we propose TEPO, a novel token-level framework that incorporates Markov Likelihood (sequence likelihood) links group-level rewards with tokens via token-level aggregation. Experiments show that TEPO consistently outperforms existing baselines across key metrics (including @k and accuracy). It not only sets a new state of the art on mathematical reasoning tasks but also significantly enhances training stability.
>
---
#### [new 034] Prompting Test-Time Scaling Is A Strong LLM Reasoning Data Augmentation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理能力提升的问题。通过提出Prompting Test-Time Scaling（P-TTS）方法，在少量手动选择的推理实例基础上，系统生成多样化的推理路径数据，用于微调Qwen-2.5模型。实验表明，P-TTS在多个数学推理基准上显著优于已有方法，并提升了零样本泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.09599v1](http://arxiv.org/pdf/2510.09599v1)**

> **作者:** Sondos Mahmoud Bsharat; Zhiqiang Shen
>
> **备注:** Our code and data are available at https://github.com/VILA-Lab/PTTS
>
> **摘要:** Large language models (LLMs) have demonstrated impressive reasoning capabilities when provided with chain-of-thought exemplars, but curating large reasoning datasets remains laborious and resource-intensive. In this work, we introduce Prompting Test-Time Scaling (P-TTS), a simple yet effective inference-time data augmentation strategy for enhancing LLM reasoning through finetuning. Rather than collecting thousands or even millions of examples, P-TTS leverages a small pool of only 90 manually selected reasoning instances and systematically varies exemplar augmentation through principled instruction prompting intensities at test time to synthesize diverse reasoning trajectory contexts. Then we finetune the various sizes of Qwen-2.5 models on P-TTS data. Across a suite of mathematical reasoning AIME2024 & 25, MATH500, and GPQA-Diamond, our P-TTS-7B and 32B models outperform the prior competitive baselines like S1 and S1.1 (1K-shot), achieving absolute accuracy gains of +26.66% and +30.00% on AIME'24 (7B), and +13.34% and +6.67% on AIME'25 (7B); P-TTS-32B yields gains of +23.33% and +16.63% on AIME'24, and +26.63% and +3.33% on AIME'25 (vs. S1 and S1.1, respectively), with comparable or better performance on MATH500 and GPQA-Diamond. We further show that P-TTS enhances zero-shot generalization accuracy on out-of-domain reasoning benchmarks of Gaokao, Kaoyan, OlympiadBench, AMC23, GradeSchoolMath, and Minerva. Our analysis suggests that test-time scaling effectively explores the latent space of reasoning patterns, amplifying LLM problem-solving with minimal annotation overhead, and further unlocking the reasoning potential and capabilities of LLMs. Prompting Test-Time Scaling offers a practical, low-cost way to elicit LLM reasoning in resource-constrained or rapidly evolving domains.
>
---
#### [new 035] When Retrieval Succeeds and Fails: Rethinking Retrieval-Augmented Generation for LLMs
- **分类: cs.CL; 68T50; I.2.7**

- **简介: 该论文属于自然语言处理任务，旨在分析检索增强生成（RAG）在大语言模型（LLMs）中的作用。它探讨了RAG的优缺点，识别其在应对动态信息和专业领域问题时的有效性，并评估了随着LLMs自身能力提升，RAG的必要性是否减弱。论文总结了RAG的核心组件和挑战，提出了未来改进方向。**

- **链接: [http://arxiv.org/pdf/2510.09106v1](http://arxiv.org/pdf/2510.09106v1)**

> **作者:** Yongjie Wang; Yue Yu; Kaisong Song; Jun Lin; Zhiqi Shen
>
> **备注:** Under Review
>
> **摘要:** Large Language Models (LLMs) have enabled a wide range of applications through their powerful capabilities in language understanding and generation. However, as LLMs are trained on static corpora, they face difficulties in addressing rapidly evolving information or domain-specific queries. Retrieval-Augmented Generation (RAG) was developed to overcome this limitation by integrating LLMs with external retrieval mechanisms, allowing them to access up-to-date and contextually relevant knowledge. However, as LLMs themselves continue to advance in scale and capability, the relative advantages of traditional RAG frameworks have become less pronounced and necessary. Here, we present a comprehensive review of RAG, beginning with its overarching objectives and core components. We then analyze the key challenges within RAG, highlighting critical weakness that may limit its effectiveness. Finally, we showcase applications where LLMs alone perform inadequately, but where RAG, when combined with LLMs, can substantially enhance their effectiveness. We hope this work will encourage researchers to reconsider the role of RAG and inspire the development of next-generation RAG systems.
>
---
#### [new 036] Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗诊断评估任务，旨在解决当前大语言模型评估与真实临床实践脱节的问题。作者提出了动态评估基准DyReMe，通过生成多样化的临床案例，从准确性、真实性、帮助性和一致性四个维度评估模型性能，揭示了现有模型在实际应用中的不足，强调了改进评估框架的必要性。**

- **链接: [http://arxiv.org/pdf/2510.09275v1](http://arxiv.org/pdf/2510.09275v1)**

> **作者:** Xiangxu Zhang; Lei Li; Yanyun Zhou; Xiao Zhou; Yingying Zhang; Xian Wu
>
> **摘要:** Medical diagnostics is a high-stakes and complex domain that is critical to patient care. However, current evaluations of large language models (LLMs) are fundamentally misaligned with real-world clinical practice. Most of them rely on static benchmarks derived from public medical exam items, which tend to overestimate model performance and ignore the difference between textbook cases and the ambiguous, varying conditions in the real world. Recent efforts toward dynamic evaluation offer a promising alternative, but their improvements are limited to superficial perturbations and a narrow focus on accuracy. To address these gaps, we propose DyReMe, a dynamic benchmark for medical diagnostics that better reflects real clinical practice. Unlike static exam-style questions, DyReMe generates fresh, consultation-like cases that introduce distractors such as differential diagnoses and common misdiagnosis factors. It also varies expression styles to mimic diverse real-world query habits. Beyond accuracy, DyReMe evaluates LLMs on three additional clinically relevant dimensions: veracity, helpfulness, and consistency. Our experiments demonstrate that this dynamic approach yields more challenging and realistic assessments, revealing significant misalignments between the performance of state-of-the-art LLMs and real clinical practice. These findings highlight the urgent need for evaluation frameworks that better reflect the demands of trustworthy medical diagnostics.
>
---
#### [new 037] MASA: LLM-Driven Multi-Agent Systems for Autoformalization
- **分类: cs.CL; cs.FL**

- **简介: 该论文属于自然语言处理与形式化推理交叉任务，旨在解决自然语言到形式表示的自动转换问题。论文提出了MASA框架，利用多智能体系统协同工作，提升自动形式化的效率与可靠性，适用于数学定义和形式化数学数据场景。**

- **链接: [http://arxiv.org/pdf/2510.08988v1](http://arxiv.org/pdf/2510.08988v1)**

> **作者:** Lan Zhang; Marco Valentino; André Freitas
>
> **备注:** EMNLP 2025 Demo camera-ready. Code and data are available at: https://github.com/lanzhang128/multi_agent_autoformalization
>
> **摘要:** Autoformalization serves a crucial role in connecting natural language and formal reasoning. This paper presents MASA, a novel framework for building multi-agent systems for autoformalization driven by Large Language Models (LLMs). MASA leverages collaborative agents to convert natural language statements into their formal representations. The architecture of MASA is designed with a strong emphasis on modularity, flexibility, and extensibility, allowing seamless integration of new agents and tools to adapt to a fast-evolving field. We showcase the effectiveness of MASA through use cases on real-world mathematical definitions and experiments on formal mathematics datasets. This work highlights the potential of multi-agent systems powered by the interaction of LLMs and theorem provers in enhancing the efficiency and reliability of autoformalization, providing valuable insights and support for researchers and practitioners in the field.
>
---
#### [new 038] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决小语言模型在常识推理中仅评估最终答案准确率忽视推理过程有效性的问题。作者构建了ReTraceQA基准，通过专家标注数据发现小模型常给出正确答案但推理错误，并利用大模型作为自动评判者，显著降低了小模型的评估得分。**

- **链接: [http://arxiv.org/pdf/2510.09351v1](http://arxiv.org/pdf/2510.09351v1)**

> **作者:** Francesco Maria Molfese; Luca Moroni; Ciro Porcaro; Simone Conia; Roberto Navigli
>
> **备注:** Submitted to ARR October 2025
>
> **摘要:** While Small Language Models (SLMs) have demonstrated promising performance on an increasingly wide array of commonsense reasoning benchmarks, current evaluation practices rely almost exclusively on the accuracy of their final answers, neglecting the validity of the reasoning processes that lead to those answers. To address this issue, we introduce ReTraceQA, a novel benchmark that introduces process-level evaluation for commonsense reasoning tasks. Our expert-annotated dataset reveals that in a substantial portion of instances (14-24%), SLMs provide correct final answers despite flawed reasoning processes, suggesting that the capabilities of SLMs are often overestimated by evaluation metrics that focus only on comparing the final answer with the ground truth. Indeed, we show that when employing strong Large Language Models (LLMs) as automated judges for reasoning-aware evaluation rather than answer-only metrics, SLM performance drops significantly across all models and datasets, with scores decreasing by up to 25%.
>
---
#### [new 039] Getting Your Indices in a Row: Full-Text Search for LLM Training Data for Real World
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决LLM训练数据不可见的问题。通过构建基于Elasticsearch和Alps基础设施的全文本索引，成功对8.6T训练数据进行索引，实现高效、绿色的大规模数据检索，并提供LLM安全性保障。**

- **链接: [http://arxiv.org/pdf/2510.09471v1](http://arxiv.org/pdf/2510.09471v1)**

> **作者:** Ines Altemir Marinas; Anastasiia Kucherenko; Alexander Sternfeld; Andrei Kucharavy
>
> **摘要:** The performance of Large Language Models (LLMs) is determined by their training data. Despite the proliferation of open-weight LLMs, access to LLM training data has remained limited. Even for fully open LLMs, the scale of the data makes it all but inscrutable to the general scientific community, despite potentially containing critical data scraped from the internet. In this paper, we present the full-text indexing pipeline for the Apertus LLM training data. Leveraging Elasticsearch parallel indices and the Alps infrastructure, a state-of-the-art, highly energy-efficient arm64 supercluster, we were able to index 8.6T tokens out of 15.2T used to train the Apertus LLM family, creating both a critical LLM safety tool and effectively an offline, curated, open web search engine. Our contribution is threefold. First, we demonstrate that Elasticsearch can be successfully ported onto next-generation arm64-based infrastructure. Second, we demonstrate that full-text indexing at the scale of modern LLM training datasets and the entire open web is feasible and accessible. Finally, we demonstrate that such indices can be used to ensure previously inaccessible jailbreak-agnostic LLM safety. We hope that our findings will be useful to other teams attempting large-scale data indexing and facilitate the general transition towards greener computation.
>
---
#### [new 040] LLP: LLM-based Product Pricing in E-commerce
- **分类: cs.CL**

- **简介: 该论文属于电商领域中的商品定价任务，旨在解决个人卖家在C2C平台难以合理定价的问题。论文提出LLP框架，基于大语言模型生成价格建议，结合检索相似商品、两阶段优化训练和置信度过滤机制，显著提升定价准确性和适用范围。**

- **链接: [http://arxiv.org/pdf/2510.09347v1](http://arxiv.org/pdf/2510.09347v1)**

> **作者:** Hairu Wang; Sheng You; Qiheng Zhang; Xike Xie; Shuguang Han; Yuchen Wu; Fei Huang; Jufeng Chen
>
> **摘要:** Unlike Business-to-Consumer e-commerce platforms (e.g., Amazon), inexperienced individual sellers on Consumer-to-Consumer platforms (e.g., eBay) often face significant challenges in setting prices for their second-hand products efficiently. Therefore, numerous studies have been proposed for automating price prediction. However, most of them are based on static regression models, which suffer from poor generalization performance and fail to capture market dynamics (e.g., the price of a used iPhone decreases over time). Inspired by recent breakthroughs in Large Language Models (LLMs), we introduce LLP, the first LLM-based generative framework for second-hand product pricing. LLP first retrieves similar products to better align with the dynamic market change. Afterwards, it leverages the LLMs' nuanced understanding of key pricing information in free-form text to generate accurate price suggestions. To strengthen the LLMs' domain reasoning over retrieved products, we apply a two-stage optimization, supervised fine-tuning (SFT) followed by group relative policy optimization (GRPO), on a dataset built via bidirectional reasoning. Moreover, LLP employs a confidence-based filtering mechanism to reject unreliable price suggestions. Extensive experiments demonstrate that LLP substantially surpasses existing methods while generalizing well to unseen categories. We have successfully deployed LLP on Xianyu\footnote\{Xianyu is China's largest second-hand e-commerce platform.\}, significantly outperforming the previous pricing method. Under the same 30\% product coverage, it raises the static adoption rate (SAR) from 40\% to 72\%, and maintains a strong SAR of 47\% even at 90\% recall.
>
---
#### [new 041] YpathRAG:A Retrieval-Augmented Generation Framework and Benchmark for Pathology
- **分类: cs.CL**

- **简介: 该论文属于医学自然语言处理任务，旨在解决病理学领域大模型易产生幻觉的问题。作者构建了包含1.53百万段落的病理学向量数据库，提出了YpathRAG框架，结合双通道检索与证据判断模块，并发布两个评测基准，显著提升了检索准确率与生成结果的可靠性。**

- **链接: [http://arxiv.org/pdf/2510.08603v1](http://arxiv.org/pdf/2510.08603v1)**

> **作者:** Deshui Yu; Yizhi Wang; Saihui Jin; Taojie Zhu; Fanyi Zeng; Wen Qian; Zirui Huang; Jingli Ouyang; Jiameng Li; Zhen Song; Tian Guan; Yonghong He
>
> **摘要:** Large language models (LLMs) excel on general tasks yet still hallucinate in high-barrier domains such as pathology. Prior work often relies on domain fine-tuning, which neither expands the knowledge boundary nor enforces evidence-grounded constraints. We therefore build a pathology vector database covering 28 subfields and 1.53 million paragraphs, and present YpathRAG, a pathology-oriented RAG framework with dual-channel hybrid retrieval (BGE-M3 dense retrieval coupled with vocabulary-guided sparse retrieval) and an LLM-based supportive-evidence judgment module that closes the retrieval-judgment-generation loop. We also release two evaluation benchmarks, YpathR and YpathQA-M. On YpathR, YpathRAG attains Recall@5 of 98.64%, a gain of 23 percentage points over the baseline; on YpathQA-M, a set of the 300 most challenging questions, it increases the accuracies of both general and medical LLMs by 9.0% on average and up to 15.6%. These results demonstrate improved retrieval quality and factual reliability, providing a scalable construction paradigm and interpretable evaluation for pathology-oriented RAG.
>
---
#### [new 042] Confidence, Not Perplexity: A Better Metric for the Creative Era of LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决传统评估指标（如困惑度）对创造性文本生成的偏差问题。作者提出了“置信度评分”（CS），通过模型输出的概率分布计算生成文本的创造性。实验表明，CS在区分不同难度任务和评估创造性方面优于传统方法，为大语言模型提供了更平衡的评估方式。**

- **链接: [http://arxiv.org/pdf/2510.08596v1](http://arxiv.org/pdf/2510.08596v1)**

> **作者:** V. S. Raghu Parupudi
>
> **备注:** Submitted to AACL-IJCNLP 2025 (Eval4NLP)
>
> **摘要:** Reference-free metrics like self-perplexity are strongly biased against creative text generation. We propose the Confidence Score (CS), derived from a model's output probability distribution, as a less biased alternative. Experiments on gpt-4o-mini show that while fluency-based metrics prefer novel responses in 0\% of cases on 99 creative prompts, our CS does so 19% of the time, a statistically significant difference (95% CI for difference: [11.1%, 27.3%]). We also show that CS effectively distinguishes between easy, medium, and hard tasks, confirmed by non-overlapping confidence intervals. The Confidence Score thus mitigates the creativity bias of traditional metrics while retaining their core evaluative strengths, offering a more balanced assessment for modern LLMs.
>
---
#### [new 043] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型中语义层级生成问题。作者提出了层次扩散语言模型（HDLM），通过层次化词汇表逐步预测更细粒度的语义。论文推导了扩散过程的ELBO，并改进训练方法，实验表明其生成效果优于基线模型。**

- **链接: [http://arxiv.org/pdf/2510.08632v1](http://arxiv.org/pdf/2510.08632v1)**

> **作者:** Cai Zhou; Chenyu Wang; Dinghuai Zhang; Shangyuan Tong; Yifei Wang; Stephen Bates; Tommi Jaakkola
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** In this paper we introduce Hierarchical Diffusion Language Models (HDLM) -- a novel family of discrete diffusion models for language modeling. HDLM builds on a hierarchical vocabulary where low-level tokens with detailed semantics are surjectively mapped to high-level tokens with coarse-grained meanings. In the forward process, each token is independently perturbed to its higher-level ancestor with more abstract semantics according to the scheduler, while in the reverse process the model progressively predicts the next, more detailed semantics. Taken together, HDLM provides a general time-varying next semantic scale prediction process for language modeling. We derive closed-form expressions for the diffusion Evidence Lower Bound (ELBO), and show that HDLM can be implemented in a flexible manner while including the existing MDLM as a special case. We also propose practical training techniques based on the insights. Extensive text generation experiments validate the effectiveness of HDLM, which demonstrates consistently lower validation and generative perplexity than baselines.
>
---
#### [new 044] Accent-Invariant Automatic Speech Recognition via Saliency-Driven Spectrogram Masking
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决口音和方言差异导致的识别准确率下降问题。作者提出了一种基于显著性驱动的频谱图掩码方法，通过训练口音分类器并掩码其关键区域，增强语音识别模型对口音变化的鲁棒性。实验表明该方法有效降低了英语和波斯语的词错误率。**

- **链接: [http://arxiv.org/pdf/2510.09528v1](http://arxiv.org/pdf/2510.09528v1)**

> **作者:** Mohammad Hossein Sameti; Sepehr Harfi Moridani; Ali Zarean; Hossein Sameti
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Pre-trained transformer-based models have significantly advanced automatic speech recognition (ASR), yet they remain sensitive to accent and dialectal variations, resulting in elevated word error rates (WER) in linguistically diverse languages such as English and Persian. To address this challenge, we propose an accent-invariant ASR framework that integrates accent and dialect classification into the recognition pipeline. Our approach involves training a spectrogram-based classifier to capture accent-specific cues, masking the regions most influential to its predictions, and using the masked spectrograms for data augmentation. This enhances the robustness of ASR models against accent variability. We evaluate the method using both English and Persian speech. For Persian, we introduce a newly collected dataset spanning multiple regional accents, establishing the first systematic benchmark for accent variation in Persian ASR that fills a critical gap in multilingual speech research and provides a foundation for future studies on low-resource, linguistically diverse languages. Experimental results with the Whisper model demonstrate that our masking and augmentation strategy yields substantial WER reductions in both English and Persian settings, confirming the effectiveness of the approach. This research advances the development of multilingual ASR systems that are resilient to accent and dialect diversity. Code and dataset are publicly available at: https://github.com/MH-Sameti/Accent_invariant_ASR
>
---
#### [new 045] Domain-Adapted Pre-trained Language Models for Implicit Information Extraction in Crash Narratives
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在从交通事故描述文本中提取隐含信息。主要解决现有工具难以批量处理非结构化文本、识别推理密集型类别效果差及隐私问题。作者通过微调开源预训练语言模型（如BERT和LLMs）来提升性能，应用于碰撞方式和事故类型识别，取得了优于闭源模型的效果，并发现模型能捕捉更多细节并修正数据集标注错误。**

- **链接: [http://arxiv.org/pdf/2510.09434v1](http://arxiv.org/pdf/2510.09434v1)**

> **作者:** Xixi Wang; Jordanka Kovaceva; Miguel Costa; Shuai Wang; Francisco Camara Pereira; Robert Thomson
>
> **摘要:** Free-text crash narratives recorded in real-world crash databases have been shown to play a significant role in improving traffic safety. However, large-scale analyses remain difficult to implement as there are no documented tools that can batch process the unstructured, non standardized text content written by various authors with diverse experience and attention to detail. In recent years, Transformer-based pre-trained language models (PLMs), such as Bidirectional Encoder Representations from Transformers (BERT) and large language models (LLMs), have demonstrated strong capabilities across various natural language processing tasks. These models can extract explicit facts from crash narratives, but their performance declines on inference-heavy tasks in, for example, Crash Type identification, which can involve nearly 100 categories. Moreover, relying on closed LLMs through external APIs raises privacy concerns for sensitive crash data. Additionally, these black-box tools often underperform due to limited domain knowledge. Motivated by these challenges, we study whether compact open-source PLMs can support reasoning-intensive extraction from crash narratives. We target two challenging objectives: 1) identifying the Manner of Collision for a crash, and 2) Crash Type for each vehicle involved in the crash event from real-world crash narratives. To bridge domain gaps, we apply fine-tuning techniques to inject task-specific knowledge to LLMs with Low-Rank Adaption (LoRA) and BERT. Experiments on the authoritative real-world dataset Crash Investigation Sampling System (CISS) demonstrate that our fine-tuned compact models outperform strong closed LLMs, such as GPT-4o, while requiring only minimal training resources. Further analysis reveals that the fine-tuned PLMs can capture richer narrative details and even correct some mislabeled annotations in the dataset.
>
---
#### [new 046] Exploring Cross-Lingual Knowledge Transfer via Transliteration-Based MLM Fine-Tuning for Critically Low-resource Chakma Language
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的跨语言知识迁移任务，旨在解决低资源语言Chakma在语言模型中的代表性不足问题。作者构建了一个基于孟加拉语转写的Chakma语数据集，并通过掩码语言建模微调多种多语言模型，有效提升了模型在该语言上的性能，验证了转写在低资源语言迁移学习中的有效性。**

- **链接: [http://arxiv.org/pdf/2510.09032v1](http://arxiv.org/pdf/2510.09032v1)**

> **作者:** Adity Khisa; Nusrat Jahan Lia; Tasnim Mahfuz Nafis; Zarif Masud; Tanzir Pial; Shebuti Rayana; Ahmedul Kabir
>
> **摘要:** As an Indo-Aryan language with limited available data, Chakma remains largely underrepresented in language models. In this work, we introduce a novel corpus of contextually coherent Bangla-transliterated Chakma, curated from Chakma literature, and validated by native speakers. Using this dataset, we fine-tune six encoder-based multilingual and regional transformer models (mBERT, XLM-RoBERTa, DistilBERT, DeBERTaV3, BanglaBERT, and IndicBERT) on masked language modeling (MLM) tasks. Our experiments show that fine-tuned multilingual models outperform their pre-trained counterparts when adapted to Bangla-transliterated Chakma, achieving up to 73.54% token accuracy and a perplexity as low as 2.90. Our analysis further highlights the impact of data quality on model performance and shows the limitations of OCR pipelines for morphologically rich Indic scripts. Our research demonstrates that Bangla-transliterated Chakma can be very effective for transfer learning for Chakma language, and we release our manually validated monolingual dataset to encourage further research on multilingual language modeling for low-resource languages.
>
---
#### [new 047] LLMs Show Surface-Form Brittleness Under Paraphrase Stress Tests
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在基准测试中因记忆测试项导致的性能高估问题。通过对比原问题与改写问题的准确率差异，评估模型泛化能力，发现改写导致准确率下降，表明模型存在表面形式脆弱性。**

- **链接: [http://arxiv.org/pdf/2510.08616v1](http://arxiv.org/pdf/2510.08616v1)**

> **作者:** Juan Miguel Navarro Carranza
>
> **备注:** NeurIPS 2025 Workshop. Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling. Selected for contributed talk
>
> **摘要:** Benchmark scores for Large Language Models (LLMs) can be inflated by memorization of test items or near duplicates. We present a simple, protocol that probes generalization by re-evaluating models on paraphrased versions of benchmark questions. Using Mistral-7B-Instruct and Qwen2.5-7B-Instruct, we measure the accuracy gap between original and paraphrased items on ARC-Easy and ARC-Challenge. Our pipeline controls decoding, enforces multiple-choice output format, and includes a robust paraphrase-cleaning step to preserve semantics. We find that paraphrasing induces a non-trivial accuracy drop (original vs. paraphrased), consistent with prior concerns about contamination and brittle surface-form shortcuts.
>
---
#### [new 048] Iterative LLM-Based Generation and Refinement of Distracting Conditions in Math Word Problems
- **分类: cs.CL**

- **简介: 该论文属于数学应用题生成任务，旨在解决现有数学应用题数据集中干扰条件不足且易被识别的问题。通过设计迭代框架，利用大语言模型自动生成不影响原解的干扰条件，提升题目复杂度与真实性，减少人工标注成本，提高数据质量。**

- **链接: [http://arxiv.org/pdf/2510.08615v1](http://arxiv.org/pdf/2510.08615v1)**

> **作者:** Kaiqi Yang; Hang Li; Yucheng Chu; Zitao Liu; Mi Tian; Hui Liu
>
> **摘要:** Mathematical reasoning serves as a crucial testbed for evaluating the intelligence of large language models (LLMs), and math word problems (MWPs) represent one of the most widely used formats. Most existing MWP datasets contain only the necessary information, while problems with distracting or excessive conditions are often overlooked. Prior studies have shown that popular LLMs experience a dramatic performance drop when such distracting conditions are introduced. However, available datasets of MWPs with distracting conditions remain limited, and most exhibit low difficulty and out-of-context expressions. These shortcomings make the distracting conditions easy to detect and disregard, thereby reducing the credibility of benchmarking on these datasets. Moreover, when distracting conditions are added, the reasoning process and answers may change, requiring intensive manual effort to check and rewrite solutions. To address these issues, we design an iterative framework that leverages LLMs to generate distracting conditions automatically. We develop a set of prompts to revise MWPs from multiple perspectives and cognitive levels, encouraging the creation of meaningful distracting conditions as well as suggestions for further refinement. A key advantage of our framework is the preservation of shared solutions between the original and revised problems: the LLMs are explicitly guided to generate distractions that do not alter the original solution, thus eliminating the need to produce new answers. This framework is efficient and easy to deploy, substantially reducing the effort required to generate MWPs with distracting conditions while maintaining high data quality.
>
---
#### [new 049] Benchmarking Chinese Commonsense Reasoning with a Multi-hop Reasoning Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决中文常识推理评估不足的问题。作者构建了CCMOR基准，通过多跳推理评估大模型在中文场景中的推理能力，并提出生成多跳问题的流程及验证机制，揭示了现有模型在长尾知识推理上的局限性，并验证了检索增强的有效性。**

- **链接: [http://arxiv.org/pdf/2510.08800v1](http://arxiv.org/pdf/2510.08800v1)**

> **作者:** Wangjie You; Xusheng Wang; Xing Wang; Wenxiang Jiao; Chao Feng; Juntao Li; Min Zhang
>
> **摘要:** While Large Language Models (LLMs) have demonstrated advanced reasoning capabilities, their comprehensive evaluation in general Chinese-language contexts remains understudied. To bridge this gap, we propose Chinese Commonsense Multi-hop Reasoning (CCMOR), a novel benchmark designed to evaluate LLMs' ability to integrate Chinese-specific factual knowledge with multi-step logical reasoning. Specifically, we first construct a domain-balanced seed set from existing QA datasets, then develop an LLM-powered pipeline to generate multi-hop questions anchored on factual unit chains. To ensure the quality of resulting dataset, we implement a human-in-the-loop verification system, where domain experts systematically validate and refine the generated questions. Using CCMOR, we evaluate state-of-the-art LLMs, demonstrating persistent limitations in LLMs' ability to process long-tail knowledge and execute knowledge-intensive reasoning. Notably, retrieval-augmented generation substantially mitigates these knowledge gaps, yielding significant performance gains.
>
---
#### [new 050] Thinking Longer, Not Always Smarter: Evaluating LLM Capabilities in Hierarchical Legal Reasoning
- **分类: cs.CL; 68T50; I.2.7; I.2.4**

- **简介: 该论文属于法律人工智能任务，旨在解决大型语言模型（LLM）在层级法律推理中的表现问题。作者提出一个三阶段推理框架，评估LLM在识别案例差异、分析论据支持和判断差异重要性方面的表现，发现模型在复杂推理任务中表现差，且错误回应消耗更多计算资源，表明“思考更久”不等于“思考更好”。**

- **链接: [http://arxiv.org/pdf/2510.08710v1](http://arxiv.org/pdf/2510.08710v1)**

> **作者:** Li Zhang; Matthias Grabmair; Morgan Gray; Kevin Ashley
>
> **备注:** 21 pages, 7 figures
>
> **摘要:** Case-based reasoning is a cornerstone of U.S. legal practice, requiring professionals to argue about a current case by drawing analogies to and distinguishing from past precedents. While Large Language Models (LLMs) have shown remarkable capabilities, their proficiency in this complex, nuanced form of reasoning needs further investigation. We propose a formal framework that decomposes the process of identifying significant distinctions between cases into three-stage reasoning tasks. Our framework models cases using factual predicates called factors, organizes them into a legal knowledge hierarchy, and defines verifiable rules for identifying distinctions, analyzing their argumentative support, and evaluating their significance. Through comprehensive evaluation of modern reasoning LLMs, we reveal a paradox: while models achieve high accuracy on surface-level reasoning (Task 1), performance degrades on hierarchical reasoning (Task 2: 64.82%-92.09%) and collapses on integrated analysis (Task 3: 11.46%-33.99%). Most strikingly, we find that models consistently expend more computational resources on incorrect responses than correct ones, suggesting that "thinking longer" does not always mean "thinking smarter." Our work provides a methodology for fine-grained analysis of LLM reasoning capabilities in complex domains and reveals fundamental limitations that must be addressed for robust and trustworthy legal AI.
>
---
#### [new 051] Toward a Safer Web: Multilingual Multi-Agent LLMs for Mitigating Adversarial Misinformation Attacks
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文属于多语言信息检测任务，旨在应对网络平台上的对抗性虚假信息攻击。论文提出了一种基于多语言大模型的多智能体框架，结合检索增强生成技术，用于检测并缓解虚假信息，支持语言转换、摘要生成和结构重排等处理，可作为插件部署于实际网络应用中。**

- **链接: [http://arxiv.org/pdf/2510.08605v1](http://arxiv.org/pdf/2510.08605v1)**

> **作者:** Nouar Aldahoul; Yasir Zaki
>
> **摘要:** The rapid spread of misinformation on digital platforms threatens public discourse, emotional stability, and decision-making. While prior work has explored various adversarial attacks in misinformation detection, the specific transformations examined in this paper have not been systematically studied. In particular, we investigate language-switching across English, French, Spanish, Arabic, Hindi, and Chinese, followed by translation. We also study query length inflation preceding summarization and structural reformatting into multiple-choice questions. In this paper, we present a multilingual, multi-agent large language model framework with retrieval-augmented generation that can be deployed as a web plugin into online platforms. Our work underscores the importance of AI-driven misinformation detection in safeguarding online factual integrity against diverse attacks, while showcasing the feasibility of plugin-based deployment for real-world web applications.
>
---
#### [new 052] KORMo: Korean Open Reasoning Model for Everyone
- **分类: cs.CL**

- **简介: 该论文提出KORMo-10B，一个基于合成数据训练的开源韩英双语大模型，旨在解决非英语语言模型资源匮乏的问题。通过合成数据构建高质量训练集，验证其在大规模预训练中的有效性，并实现与多语言模型相当的性能，推动低资源语言模型的发展。**

- **链接: [http://arxiv.org/pdf/2510.09426v1](http://arxiv.org/pdf/2510.09426v1)**

> **作者:** Minjun Kim; Hyeonseok Lim; Hangyeol Yoo; Inho Won; Seungwoo Song; Minkyung Cho; Junhun Yuk; Changsu Choi; Dongjae Shin; Huige Lee; Hoyun Song; Alice Oh; Kyungtae Lim
>
> **摘要:** This work presents the first large-scale investigation into constructing a fully open bilingual large language model (LLM) for a non-English language, specifically Korean, trained predominantly on synthetic data. We introduce KORMo-10B, a 10.8B-parameter model trained from scratch on a Korean-English corpus in which 68.74% of the Korean portion is synthetic. Through systematic experimentation, we demonstrate that synthetic data, when carefully curated with balanced linguistic coverage and diverse instruction styles, does not cause instability or degradation during large-scale pretraining. Furthermore, the model achieves performance comparable to that of contemporary open-weight multilingual baselines across a wide range of reasoning, knowledge, and instruction-following benchmarks. Our experiments reveal two key findings: (1) synthetic data can reliably sustain long-horizon pretraining without model collapse, and (2) bilingual instruction tuning enables near-native reasoning and discourse coherence in Korean. By fully releasing all components including data, code, training recipes, and logs, this work establishes a transparent framework for developing synthetic data-driven fully open models (FOMs) in low-resource settings and sets a reproducible precedent for future multilingual LLM research.
>
---
#### [new 053] Hybrid Models for Natural Language Reasoning: The Case of Syllogistic Logic
- **分类: cs.CL; cs.LG; cs.LO**

- **简介: 该论文属于自然语言推理任务，旨在解决神经模型在逻辑推理中泛化能力不足的问题，尤其是组合性和递归性。作者通过分析预训练语言模型在三段论逻辑上的表现，发现其递归性较好但组合性较弱，进而提出结合符号推理与神经计算的混合架构，以提升逻辑推理的效率与完整性。**

- **链接: [http://arxiv.org/pdf/2510.09472v1](http://arxiv.org/pdf/2510.09472v1)**

> **作者:** Manuel Vargas Guzmán; Jakub Szymanik; Maciej Malicki
>
> **摘要:** Despite the remarkable progress in neural models, their ability to generalize, a cornerstone for applications like logical reasoning, remains a critical challenge. We delineate two fundamental aspects of this ability: compositionality, the capacity to abstract atomic logical rules underlying complex inferences, and recursiveness, the aptitude to build intricate representations through iterative application of inference rules. In the literature, these two aspects are often confounded together under the umbrella term of generalization. To sharpen this distinction, we investigated the logical generalization capabilities of pre-trained large language models (LLMs) using the syllogistic fragment as a benchmark for natural language reasoning. Though simple, this fragment provides a foundational yet expressive subset of formal logic that supports controlled evaluation of essential reasoning abilities. Our findings reveal a significant disparity: while LLMs demonstrate reasonable proficiency in recursiveness, they struggle with compositionality. To overcome these limitations and establish a reliable logical prover, we propose a hybrid architecture integrating symbolic reasoning with neural computation. This synergistic interaction enables robust and efficient inference, neural components accelerate processing, while symbolic reasoning ensures completeness. Our experiments show that high efficiency is preserved even with relatively small neural components. As part of our proposed methodology, this analysis gives a rationale and highlights the potential of hybrid models to effectively address key generalization barriers in neural reasoning systems.
>
---
#### [new 054] CrisiText: A dataset of warning messages for LLM training in emergency communication
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成（NLG）任务，旨在解决危机场景中及时生成有效警告信息的问题。作者构建了大规模数据集CrisiText，包含40多万条警告消息，并设计了多种实验评估不同生成方法的效果，以提升危机沟通中的信息生成质量。**

- **链接: [http://arxiv.org/pdf/2510.09243v1](http://arxiv.org/pdf/2510.09243v1)**

> **作者:** Giacomo Gonella; Gian Maria Campedelli; Stefano Menini; Marco Guerini
>
> **摘要:** Effectively identifying threats and mitigating their potential damage during crisis situations, such as natural disasters or violent attacks, is paramount for safeguarding endangered individuals. To tackle these challenges, AI has been used in assisting humans in emergency situations. Still, the use of NLP techniques remains limited and mostly focuses on classification tasks. The significant potential of timely warning message generation using NLG architectures, however, has been largely overlooked. In this paper we present CrisiText, the first large-scale dataset for the generation of warning messages across 13 different types of crisis scenarios. The dataset contains more than 400,000 warning messages (spanning almost 18,000 crisis situations) aimed at assisting civilians during and after such events. To generate the dataset, we started from existing crisis descriptions and created chains of events related to the scenarios. Each event was then paired with a warning message. The generations follow experts' written guidelines to ensure correct terminology and factuality of their suggestions. Additionally, each message is accompanied by three suboptimal warning types to allow for the study of different NLG approaches. To this end, we conducted a series of experiments comparing supervised fine-tuning setups with preference alignment, zero-shot, and few-shot approaches. We further assessed model performance in out-of-distribution scenarios and evaluated the effectiveness of an automatic post-editor.
>
---
#### [new 055] The Speech-LLM Takes It All: A Truly Fully End-to-End Spoken Dialogue State Tracking Approach
- **分类: cs.CL; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于口语对话状态跟踪任务，旨在通过不同上下文管理策略提升端到端模型性能。论文比较了多模态上下文、完整语音历史和压缩语音历史方法，在SpokenWOZ语料库上验证了完整语音输入效果最佳，并提出基于注意力池化的压缩策略，在减少上下文长度的同时保持较高准确率。**

- **链接: [http://arxiv.org/pdf/2510.09424v1](http://arxiv.org/pdf/2510.09424v1)**

> **作者:** Nizar El Ghazal; Antoine Caubrière; Valentin Vielzeuf
>
> **摘要:** This paper presents a comparative study of context management strategies for end-to-end Spoken Dialog State Tracking using Speech-LLMs. We systematically evaluate traditional multimodal context (combining text history and spoken current turn), full spoken history, and compressed spoken history approaches. Our experiments on the SpokenWOZ corpus demonstrate that providing the full spoken conversation as input yields the highest performance among models of similar size, significantly surpassing prior methods. Furthermore, we show that attention-pooling-based compression of the spoken history offers a strong trade-off, maintaining competitive accuracy with reduced context size. Detailed analysis confirms that improvements stem from more effective context utilization.
>
---
#### [new 056] SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习与自然语言处理任务，旨在解决扩散语言模型（dLLMs）难以通过传统策略梯度方法对齐人类偏好的问题。作者提出“夹心策略梯度”（SPG），利用上下界估计真实对数似然，减少梯度偏差，并在多个任务上显著优于基线方法。**

- **链接: [http://arxiv.org/pdf/2510.09541v1](http://arxiv.org/pdf/2510.09541v1)**

> **作者:** Chengyu Wang; Paria Rashidinejad; DiJia Su; Song Jiang; Sid Wang; Siyan Zhao; Cai Zhou; Shannon Zejiang Shen; Feiyu Chen; Tommi Jaakkola; Yuandong Tian; Bo Liu
>
> **摘要:** Diffusion large language models (dLLMs) are emerging as an efficient alternative to autoregressive models due to their ability to decode multiple tokens in parallel. However, aligning dLLMs with human preferences or task-specific rewards via reinforcement learning (RL) is challenging because their intractable log-likelihood precludes the direct application of standard policy gradient methods. While prior work uses surrogates like the evidence lower bound (ELBO), these one-sided approximations can introduce significant policy gradient bias. To address this, we propose the Sandwiched Policy Gradient (SPG) that leverages both an upper and a lower bound of the true log-likelihood. Experiments show that SPG significantly outperforms baselines based on ELBO or one-step estimation. Specifically, SPG improves the accuracy over state-of-the-art RL methods for dLLMs by 3.6% in GSM8K, 2.6% in MATH500, 18.4% in Countdown and 27.0% in Sudoku.
>
---
#### [new 057] Large Language Models Do NOT Really Know What They Don't Know
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在探究大型语言模型（LLMs）是否能识别自身生成内容的真实性。研究者通过分析模型内部对事实性与虚构性输出的处理机制，发现LLMs在面对与主题相关但错误的信息时，其内部状态与正确回答相似，难以区分；而与主题无关的错误则呈现可检测的聚类特征。结论表明，LLMs并不真正理解“自己不知道什么”。**

- **链接: [http://arxiv.org/pdf/2510.09033v1](http://arxiv.org/pdf/2510.09033v1)**

> **作者:** Chi Seng Cheang; Hou Pong Chan; Wenxuan Zhang; Yang Deng
>
> **摘要:** Recent work suggests that large language models (LLMs) encode factuality signals in their internal representations, such as hidden states, attention weights, or token probabilities, implying that LLMs may "know what they don't know". However, LLMs can also produce factual errors by relying on shortcuts or spurious associations. These error are driven by the same training objective that encourage correct predictions, raising the question of whether internal computations can reliably distinguish between factual and hallucinated outputs. In this work, we conduct a mechanistic analysis of how LLMs internally process factual queries by comparing two types of hallucinations based on their reliance on subject information. We find that when hallucinations are associated with subject knowledge, LLMs employ the same internal recall process as for correct responses, leading to overlapping and indistinguishable hidden-state geometries. In contrast, hallucinations detached from subject knowledge produce distinct, clustered representations that make them detectable. These findings reveal a fundamental limitation: LLMs do not encode truthfulness in their internal states but only patterns of knowledge recall, demonstrating that "LLMs don't really know what they don't know".
>
---
#### [new 058] FrameEOL: Semantic Frame Induction using Causal Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语义框架归纳任务，旨在解决如何通过因果语言模型（CLM）更有效地对框架触发词进行语义框架聚类的问题。论文提出FrameEOL方法，利用CLM结合上下文学习和深度度量学习生成更适合框架归纳的嵌入表示，并通过聚类完成框架归纳，在英日语数据上取得了优于现有方法的效果。**

- **链接: [http://arxiv.org/pdf/2510.09097v1](http://arxiv.org/pdf/2510.09097v1)**

> **作者:** Chihiro Yano; Kosuke Yamada; Hayato Tsukagoshi; Ryohei Sasano; Koichi Takeda
>
> **备注:** Accepted in EMNLP Findings 2025. This version corrects the model size of Table 3
>
> **摘要:** Semantic frame induction is the task of clustering frame-evoking words according to the semantic frames they evoke. In recent years, leveraging embeddings of frame-evoking words that are obtained using masked language models (MLMs) such as BERT has led to high-performance semantic frame induction. Although causal language models (CLMs) such as the GPT and Llama series succeed in a wide range of language comprehension tasks and can engage in dialogue as if they understood frames, they have not yet been applied to semantic frame induction. We propose a new method for semantic frame induction based on CLMs. Specifically, we introduce FrameEOL, a prompt-based method for obtaining Frame Embeddings that outputs One frame-name as a Label representing the given situation. To obtain embeddings more suitable for frame induction, we leverage in-context learning (ICL) and deep metric learning (DML). Frame induction is then performed by clustering the resulting embeddings. Experimental results on the English and Japanese FrameNet datasets demonstrate that the proposed methods outperform existing frame induction methods. In particular, for Japanese, which lacks extensive frame resources, the CLM-based method using only 5 ICL examples achieved comparable performance to the MLM-based method fine-tuned with DML.
>
---
#### [new 059] ReFIne: A Framework for Trustworthy Large Reasoning Models with Reliability, Faithfulness, and Interpretability
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大模型推理的可信度。针对当前模型忽视可解释性、真实性和可靠性问题，论文提出ReFIne框架，结合监督微调与GRPO方法，优化模型在结构化推理、决策透明和自信评估方面的能力。实验验证了其在多规模模型上的有效性。**

- **链接: [http://arxiv.org/pdf/2510.09062v1](http://arxiv.org/pdf/2510.09062v1)**

> **作者:** Chung-En Sun; Ge Yan; Akshay Kulkarni; Tsui-Wei Weng
>
> **摘要:** Recent advances in long chain-of-thought (CoT) reasoning have largely prioritized answer accuracy and token efficiency, while overlooking aspects critical to trustworthiness. We argue that usable reasoning systems must be trustworthy, characterized by three properties: interpretability, faithfulness, and reliability. To this end, we propose ReFIne, a new training framework that integrates supervised fine-tuning with GRPO to encourage models to: (i) improve interpretability by producing structured, tag-based traces with high-level planning that are easier for humans to follow; (ii) enhance faithfulness by explicitly disclosing the decisive information guiding each solution, with consistent cross-section references; and (iii) promote reliability by providing self-assessments of both the derivation's soundness and the confidence of the final answer. We apply ReFIne to the Qwen3 models at multiple scales (1.7B/4B/8B) and evaluate across mathematical benchmarks of varying difficulty. Our experimental results show that ReFIne models generate clearer and better-structured reasoning traces (interpretability +44.0%), more faithfully expose their underlying decision process (faithfulness +18.8%), and offer informative confidence estimates (reliability +42.4%). These findings highlight an overlooked but important direction: reasoning models should be optimized not only for accuracy, but also for broader dimensions of trustworthiness. Our code is available at: https://github.com/Trustworthy-ML-Lab/Training_Trustworthy_LRM_with_Refine
>
---
#### [new 060] Text2Stories: Evaluating the Alignment Between Stakeholder Interviews and Generated User Stories
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于自然语言处理与软件工程交叉任务，旨在解决自动生成的用户需求是否忠实反映利益相关者意图的问题。作者提出了Text2Stories方法，通过正确性与完整性指标衡量访谈文本与用户故事的对齐程度，实现了自动化评估，并验证了LLM在匹配任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2510.08622v1](http://arxiv.org/pdf/2510.08622v1)**

> **作者:** Francesco Dente; Fabiano Dalpiaz; Paolo Papotti
>
> **备注:** 8 pages
>
> **摘要:** Large language models (LLMs) can be employed for automating the generation of software requirements from natural language inputs such as the transcripts of elicitation interviews. However, evaluating whether those derived requirements faithfully reflect the stakeholders' needs remains a largely manual task. We introduce Text2Stories, a task and metrics for text-to-story alignment that allow quantifying the extent to which requirements (in the form of user stories) match the actual needs expressed by the elicitation session participants. Given an interview transcript and a set of user stories, our metric quantifies (i) correctness: the proportion of stories supported by the transcript, and (ii) completeness: the proportion of transcript supported by at least one story. We segment the transcript into text chunks and instantiate the alignment as a matching problem between chunks and stories. Experiments over four datasets show that an LLM-based matcher achieves 0.86 macro-F1 on held-out annotations, while embedding models alone remain behind but enable effective blocking. Finally, we show how our metrics enable the comparison across sets of stories (e.g., human vs. generated), positioning Text2Stories as a scalable, source-faithful complement to existing user-story quality criteria.
>
---
#### [new 061] Exploring Multi-Temperature Strategies for Token- and Rollout-Level Control in RLVR
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习与大语言模型推理任务，旨在解决推理过程中探索与利用的平衡问题。论文提出多温度策略，对不同类型的token（推理与知识）采用不同温度参数，以提升模型推理性能。**

- **链接: [http://arxiv.org/pdf/2510.08892v1](http://arxiv.org/pdf/2510.08892v1)**

> **作者:** Haomin Zhuang; Yujun Zhou; Taicheng Guo; Yue Huang; Fangxu Liu; Kai Song; Xiangliang Zhang
>
> **摘要:** Reinforcement Learning has demonstrated substantial improvements in the reasoning abilities of Large Language Models (LLMs), exhibiting significant applicability across various domains. Recent research has identified that tokens within LLMs play distinct roles during reasoning tasks, categorizing them into high-entropy reasoning tokens and low-entropy knowledge tokens. Prior approaches have typically focused on restricting updates to indirectly encourage exploration, yet they do not explicitly facilitate exploratory behavior during the token generation stage itself. In this work, we introduce a complementary approach that explicitly promotes exploration during sampling by applying distinct temperature settings for different token types. Specifically, our method employs higher temperatures for reasoning tokens to actively encourage exploration, while retaining lower temperatures for knowledge tokens to maintain factual correctness. Furthermore, we systematically investigate various multi-temperature scheduling strategies and their impacts within reinforcement learning contexts. Empirical evaluations on several reasoning benchmarks demonstrate that our approach significantly enhances the reasoning performance of LLMs. The code is available at https://github.com/zhmzm/Multi_Temperature_Verl.git.
>
---
#### [new 062] LitE-SQL: A Lightweight and Efficient Text-to-SQL Framework with Vector-based Schema Linking and Execution-Guided Self-Correction
- **分类: cs.CL**

- **简介: 该论文属于文本到SQL生成任务，旨在将自然语言问题转化为SQL查询。为解决依赖大模型带来的隐私和部署问题，作者提出了LitE-SQL框架，包含向量库加速的模式链接和基于执行的自修正机制。在多个数据集上表现优异，验证了轻量模型在该任务中的可行性。**

- **链接: [http://arxiv.org/pdf/2510.09014v1](http://arxiv.org/pdf/2510.09014v1)**

> **作者:** Shengmin Piao; Jieun Lee; Sanghyun Park
>
> **摘要:** The Text-to-SQL task translates natural language questions into SQL queries, enabling intuitive database interaction for non-experts. While recent methods leveraging Large Language Models (LLMs) achieve strong performance, their reliance on proprietary models raise concerns about deployment feasibility and data privacy. In this work, we introduce LitE-SQL, a Lightweight and Efficient framework with two components: (i) a Schema Retriever that performs efficient schema linking using a vector database of pre-computed schema embeddings, and (ii) a SQL Generator fine-tuned in two stages-supervised fine-tuning followed by execution-guided reinforcement-enabling self-correction without costly multi-candidate generation. On BIRD, LitE-SQL achieves 72.10% execution accuracy, and on Spider 1.0 it reaches 88.45%, demonstrating comparable or superior performance to LLM-based methods despite using 2x to 30x fewer parameters. Our findings demonstrate that high-quality Text-to-SQL generation is feasible with lightweight models, offering a practical solution for privacy-sensitive and resource-constrained settings.
>
---
#### [new 063] Mnemosyne: An Unsupervised, Human-Inspired Long-Term Memory Architecture for Edge-Based LLMs
- **分类: cs.CL; cs.AI; cs.LG; cs.MA**

- **简介: 该论文设计了一个适用于边缘设备的无监督长时记忆架构Mnemosyne，旨在解决现有大语言模型记忆系统在边缘设备上效率低、无法处理长期对话的问题。通过图结构存储、记忆筛选与更新机制，以及基于时间衰减的回忆机制，提升了长期记忆能力与对话真实性。应用于纵向医疗对话中，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.08601v1](http://arxiv.org/pdf/2510.08601v1)**

> **作者:** Aneesh Jonelagadda; Christina Hahn; Haoze Zheng; Salvatore Penachio
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Long-term memory is essential for natural, realistic dialogue. However, current large language model (LLM) memory systems rely on either brute-force context expansion or static retrieval pipelines that fail on edge-constrained devices. We introduce Mnemosyne, an unsupervised, human-inspired long-term memory architecture designed for edge-based LLMs. Our approach uses graph-structured storage, modular substance and redundancy filters, memory committing and pruning mechanisms, and probabilistic recall with temporal decay and refresh processes modeled after human memory. Mnemosyne also introduces a concentrated "core summary" efficiently derived from a fixed-length subset of the memory graph to capture the user's personality and other domain-specific long-term details such as, using healthcare application as an example, post-recovery ambitions and attitude towards care. Unlike existing retrieval-augmented methods, Mnemosyne is designed for use in longitudinal healthcare assistants, where repetitive and semantically similar but temporally distinct conversations are limited by naive retrieval. In experiments with longitudinal healthcare dialogues, Mnemosyne demonstrates the highest win rate of 65.8% in blind human evaluations of realism and long-term memory capability compared to a baseline RAG win rate of 31.1%. Mnemosyne also achieves current highest LoCoMo benchmark scores in temporal reasoning and single-hop retrieval compared to other same-backboned techniques. Further, the average overall score of 54.6% was second highest across all methods, beating commonly used Mem0 and OpenAI baselines among others. This demonstrates that improved factual recall, enhanced temporal reasoning, and much more natural user-facing responses can be feasible with an edge-compatible and easily transferable unsupervised memory architecture.
>
---
#### [new 064] ExPO-HM: Learning to Explain-then-Detect for Hateful Meme Detection
- **分类: cs.CL**

- **简介: 该论文属于多模态文本理解任务，旨在解决仇恨模因（hateful memes）的自动检测问题。现有方法仅提供二分类结果，缺乏解释性。论文提出ExPO-HM框架，结合SFT预训练、GRPO与课程学习及条件决策熵，实现解释驱动的检测，在多个基准上取得性能提升。**

- **链接: [http://arxiv.org/pdf/2510.08630v1](http://arxiv.org/pdf/2510.08630v1)**

> **作者:** Jingbiao Mei; Mingsheng Sun; Jinghong Chen; Pengda Qin; Yuhong Li; Da Chen; Bill Byrne
>
> **备注:** Preprint
>
> **摘要:** Hateful memes have emerged as a particularly challenging form of online abuse, motivating the development of automated detection systems. Most prior approaches rely on direct detection, producing only binary predictions. Such models fail to provide the context and explanations that real-world moderation requires. Recent Explain-then-Detect approaches, using Chain-of-Thought prompting or LMM agents, perform worse than simple SFT baselines, and even advanced post-training methods such as GRPO fail to close the gap. Our analysis identifies two key issues of such systems: important policy-relevant cues such as targets and attack types are not hypothesized by the model as a likely explanation; and the binary reward signal is insufficient to guide reasoning. To address these challenges, we propose ExPO-HM (Explain-then-Detect Policy Optimization for Hateful Memes), inspired by the training and evaluation process of human annotators. ExPO-HM combines SFT warmup, GRPO with curriculum learning, and Conditional Decision Entropy (CDE) as both metric and reward for reasoning quality. Across three hateful meme benchmarks, ExPO-HM achieves state-of-the-art performance on binary detection, fine-grained classification, and reasoning quality, with up to 15\% and 17\% F1 improvement over the GRPO and DPO baselines, respectively. By moving hateful meme detection from simple binary alarms to explanation-driven detection, ExPO-HM provides accurate, interpretable, and actionable moderation support.
>
---
#### [new 065] Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理加速任务，旨在解决长上下文压缩问题。现有方法依赖自编码训练，导致压缩特征与实际任务不匹配。论文提出SAC方法，通过选取关键“锚点”令牌并聚合上下文信息，无需自编码训练，提升了压缩效果和任务表现。**

- **链接: [http://arxiv.org/pdf/2510.08907v1](http://arxiv.org/pdf/2510.08907v1)**

> **作者:** Xin Liu; RunSong Zhao; PengCheng Huang; XinYu Liu; JunYi Xiao; ChunYang Xiao; Tong Xiao; Shengxiang Gao; Zhengtao Yu; JingBo Zhu
>
> **备注:** 18 pages,9 figures
>
> **摘要:** Context compression presents a promising approach for accelerating large language model (LLM) inference by compressing long contexts into compact representations. Current context compression methods predominantly rely on autoencoding tasks to train context-agnostic compression tokens to compress contextual semantics. While autoencoding tasks enable compression tokens to acquire compression capabilities, compression via autoencoding tasks creates a fundamental mismatch: the models are optimized for reconstruction that diverge from actual downstream tasks, thereby weakening the features more beneficial for real-world usage. We propose Semantic-Anchor Compression (SAC), a novel method that shifts from autoencoding task based compression to an architecture that is equipped with this compression capability \textit{a priori}. Instead of training models to compress contexts through autoencoding tasks, SAC directly selects so-called anchor tokens from the original context and aggregates contextual information into their key-value (KV) representations. By deriving representations directly from the contextual tokens, SAC eliminates the need for autoencoding training. To ensure compression performance while directly leveraging anchor tokens, SAC incorporates two key designs: (1) anchor embeddings that enable the compressor to identify critical tokens, and (2) bidirectional attention modification that allows anchor tokens to capture information from the entire context. Experimental results demonstrate that SAC consistently outperforms existing context compression methods across various compression ratios. On out-of-distribution evaluation using MRQA, SAC achieves 1 EM improvement at 5x compression over strong baselines, with increasing advantages at higher compression ratios.
>
---
#### [new 066] MMA-ASIA: A Multilingual and Multimodal Alignment Framework for Culturally-Grounded Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态、多语言文化评估任务，旨在解决大模型在非西方语境下文化理解能力不足的问题。工作包括构建包含8个亚洲国家、10种语言、27,000道题的多模态基准测试MMA-ASIA，提出五维评估协议，并设计模块验证文化认知的根基有效性，以提升模型跨文化可靠性。**

- **链接: [http://arxiv.org/pdf/2510.08608v1](http://arxiv.org/pdf/2510.08608v1)**

> **作者:** Weihua Zheng; Zhengyuan Liu; Tanmoy Chakraborty; Weiwen Xu; Xiaoxue Gao; Bryan Chen Zhengyu Tan; Bowei Zou; Chang Liu; Yujia Hu; Xing Xie; Xiaoyuan Yi; Jing Yao; Chaojun Wang; Long Li; Rui Liu; Huiyao Liu; Koji Inoue; Ryuichi Sumida; Tatsuya Kawahara; Fan Xu; Lingyu Ye; Wei Tian; Dongjun Kim; Jimin Jung; Jaehyung Seo; Nadya Yuki Wangsajaya; Pham Minh Duc; Ojasva Saxena; Palash Nandi; Xiyan Tao; Wiwik Karlina; Tuan Luong; Keertana Arun Vasan; Roy Ka-Wei Lee; Nancy F. Chen
>
> **摘要:** Large language models (LLMs) are now used worldwide, yet their multimodal understanding and reasoning often degrade outside Western, high-resource settings. We propose MMA-ASIA, a comprehensive framework to evaluate LLMs' cultural awareness with a focus on Asian contexts. MMA-ASIA centers on a human-curated, multilingual, and multimodally aligned multiple-choice benchmark covering 8 Asian countries and 10 languages, comprising 27,000 questions; over 79 percent require multi-step reasoning grounded in cultural context, moving beyond simple memorization. To our knowledge, this is the first dataset aligned at the input level across three modalities: text, image (visual question answering), and speech. This enables direct tests of cross-modal transfer. Building on this benchmark, we propose a five-dimensional evaluation protocol that measures: (i) cultural-awareness disparities across countries, (ii) cross-lingual consistency, (iii) cross-modal consistency, (iv) cultural knowledge generalization, and (v) grounding validity. To ensure rigorous assessment, a Cultural Awareness Grounding Validation Module detects "shortcut learning" by checking whether the requisite cultural knowledge supports correct answers. Finally, through comparative model analysis, attention tracing, and an innovative Vision-ablated Prefix Replay (VPR) method, we probe why models diverge across languages and modalities, offering actionable insights for building culturally reliable multimodal LLMs.
>
---
#### [new 067] Active Model Selection for Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 论文提出LLM SELECTOR，首个针对大语言模型的主动模型选择框架。任务是减少模型选择时的标注成本。通过自适应选择最具信息量的查询进行标注，并使用判断模型辅助标注。实验表明其可大幅降低标注成本。**

- **链接: [http://arxiv.org/pdf/2510.09418v1](http://arxiv.org/pdf/2510.09418v1)**

> **作者:** Yavuz Durmazkeser; Patrik Okanovic; Andreas Kirsch; Torsten Hoefler; Nezihe Merve Gürel
>
> **摘要:** We introduce LLM SELECTOR, the first framework for active model selection of Large Language Models (LLMs). Unlike prior evaluation and benchmarking approaches that rely on fully annotated datasets, LLM SELECTOR efficiently identifies the best LLM with limited annotations. In particular, for any given task, LLM SELECTOR adaptively selects a small set of queries to annotate that are most informative about the best model for the task. To further reduce annotation cost, we leverage a judge-based oracle annotation model. Through extensive experiments on 6 benchmarks with 151 LLMs, we show that LLM SELECTOR reduces annotation costs by up to 59.62% when selecting the best and near-best LLM for the task.
>
---
#### [new 068] How Reliable is Language Model Micro-Benchmarking?
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务中的模型评估方向。它旨在解决微基准测试在语言模型开发中是否能可靠评估模型性能的问题。论文提出了一种元评估方法，分析微基准测试在不同模型对上的排序能力，发现当前微基准测试方法在模型性能相近时可靠性较低，需较多样例才能提高可靠性，并为微基准测试的应用提供了实际指导。**

- **链接: [http://arxiv.org/pdf/2510.08730v1](http://arxiv.org/pdf/2510.08730v1)**

> **作者:** Gregory Yauney; Shahzaib Saqib Warraich; Swabha Swayamdipta
>
> **摘要:** Micro-benchmarking offers a solution to the often prohibitive time and cost of language model development: evaluate on a very small subset of existing benchmarks. Can these micro-benchmarks, however, rank models as consistently as the full benchmarks they replace? And can they rank models more consistently than selecting a random subset of data points? In many scenarios, we find that the answer is no. We introduce a meta-evaluation measure for micro-benchmarking which investigates how well a micro-benchmark can rank two models as a function of their performance difference on the full benchmark. This approach can determine which model pairs can be ranked correctly by a micro-benchmark, allowing for a finer-grained analysis of the trade-off between micro-benchmark size and reliability. Prior work has suggested selecting as few as 10 examples; we find that no micro-benchmarking method can consistently rank model pairs 3.5 points of accuracy apart on MMLU-Pro or 4 points apart on BIG-bench Hard. In order to consistently rank model pairs with relatively similar performances, we show that often as many as 250 examples must be selected, at which point random sampling is competitive with existing micro-benchmarking methods. When comparing only 8B instruction-tuned models on MMLU-Pro micro-benchmarks with 25 examples, we find that more than half of pairwise comparisons are not likely to be preserved. Our work provides actionable guidance for both micro-benchmark users and developers in navigating the trade-off between evaluation efficiency and reliability.
>
---
#### [new 069] On the Representations of Entities in Auto-regressive Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLMs）中实体的内部表示机制，属于自然语言处理任务。它旨在解决如何编码和操作多词实体的问题。论文提出“实体透镜”方法，利用任务向量从模型隐藏状态重建实体提及，探索实体表示是否包含关系知识，并验证模型能否生成未在训练中见过的多词实体。**

- **链接: [http://arxiv.org/pdf/2510.09421v1](http://arxiv.org/pdf/2510.09421v1)**

> **作者:** Victor Morand; Josiane Mothe; Benjamin Piwowarski
>
> **备注:** Accepted at BlackBoxNLP@EMNLP2025
>
> **摘要:** Named entities are fundamental building blocks of knowledge in text, grounding factual information and structuring relationships within language. Despite their importance, it remains unclear how Large Language Models (LLMs) internally represent entities. Prior research has primarily examined explicit relationships, but little is known about entity representations themselves. We introduce entity mention reconstruction as a novel framework for studying how LLMs encode and manipulate entities. We investigate whether entity mentions can be generated from internal representations, how multi-token entities are encoded beyond last-token embeddings, and whether these representations capture relational knowledge. Our proposed method, leveraging _task vectors_, allows to consistently generate multi-token mentions from various entity representations derived from the LLMs hidden states. We thus introduce the _Entity Lens_, extending the _logit-lens_ to predict multi-token mentions. Our results bring new evidence that LLMs develop entity-specific mechanisms to represent and manipulate any multi-token entities, including those unseen during training. Our code is avalable at https://github.com/VictorMorand/EntityRepresentations .
>
---
#### [new 070] Formalizing Style in Personal Narratives
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在建立一个形式化框架，用于分析个人叙述中的风格。任务是通过语言学、计算机科学和心理学的结合，自动提取并分析语言模式，揭示其与心理状态的关系。解决了缺乏系统分析叙述风格的问题，应用于梦境叙述及创伤后应激障碍案例研究。**

- **链接: [http://arxiv.org/pdf/2510.08649v1](http://arxiv.org/pdf/2510.08649v1)**

> **作者:** Gustave Cortal; Alain Finkel
>
> **摘要:** Personal narratives are stories authors construct to make meaning of their experiences. Style, the distinctive way authors use language to express themselves, is fundamental to how these narratives convey subjective experiences. Yet there is a lack of a formal framework for systematically analyzing these stylistic choices. We present a novel approach that formalizes style in personal narratives as patterns in the linguistic choices authors make when communicating subjective experiences. Our framework integrates three domains: functional linguistics establishes language as a system of meaningful choices, computer science provides methods for automatically extracting and analyzing sequential patterns, and these patterns are linked to psychological observations. Using language models, we automatically extract linguistic features such as processes, participants, and circumstances. We apply our framework to hundreds of dream narratives, including a case study on a war veteran with post-traumatic stress disorder. Analysis of his narratives uncovers distinctive patterns, particularly how verbal processes dominate over mental ones, illustrating the relationship between linguistic choices and psychological states.
>
---
#### [new 071] Systematic Diagnosis of Brittle Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于人工智能领域数学推理能力评估任务，旨在解决大模型在数学推理中的脆弱性问题。通过分析GPT-3.5-Turbo在GSM8K数据集上的推理过程，利用GPT-4o-mini对错误进行分类并聚类识别“推理模式”，发现模型在组合推理方面表现较差，揭示了其非人类的认知脆弱性。**

- **链接: [http://arxiv.org/pdf/2510.08595v1](http://arxiv.org/pdf/2510.08595v1)**

> **作者:** V. S. Raghu Parupudi
>
> **备注:** Submitted to NEURIPS-2025 MATHAI workshop
>
> **摘要:** A central question in artificial intelligence is the extent to which machine learning models comprehend mathematics. To address this, we propose a novel framework for measuring mathematical reasoning that moves beyond standard benchmarks to diagnose specific failure points. Our method first generates structured, step-by-step reasoning from gpt-3.5-turbo on the GSM8K dataset. We then use a more capable analyst model, gpt-4o-mini, to categorize errors and, crucially, perform an unsupervised clustering of every reasoning sentence to identify emergent "reasoning modes." This analysis reveals a cognitive profile with a stark, nonhuman-like brittleness: while the model achieves near-perfect accuracy on procedural modes like sequential calculation, its performance on modes requiring combinatorial reasoning with restrictions plummets. By identifying and quantifying the reliability of these distinct reasoning skills, our work provides a more granular method to evaluate mathematical comprehension and offers a precise roadmap for developing new capabilities and more reliable future applications.
>
---
#### [new 072] Logit Arithmetic Elicits Long Reasoning Capabilities Without Training
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型缺乏长推理能力的问题。作者提出ThinkLogit方法，在解码时通过logit算术调整大模型推理行为，无需额外训练。进一步结合偏好优化提升性能，实验证明其在多个基准上显著提高推理准确率。**

- **链接: [http://arxiv.org/pdf/2510.09354v1](http://arxiv.org/pdf/2510.09354v1)**

> **作者:** Yunxiang Zhang; Muhammad Khalifa; Lechen Zhang; Xin Liu; Ayoung Lee; Xinliang Frederick Zhang; Farima Fatahi Bayat; Lu Wang
>
> **摘要:** Large reasoning models exhibit long chain-of-thought reasoning with strategies such as backtracking and self-correction, though recent studies suggest that these abilities typically require additional training. We first investigate whether such behaviors can be elicited without any training. To this end, we propose a decoding-time approach, ThinkLogit, which utilizes logit arithmetic to tune a target large non-reasoning model for long reasoning using a substantially smaller reasoning model as the guider. We then show that we can further boost its performance by training the guider model with preference optimization over correct/incorrect reasoning pairs sampled from both the target and guider model, a setup we refer to as ThinkLogit-DPO. Our experiments demonstrate that ThinkLogit and ThinkLogit-DPO achieve a relative improvement in average accuracy by 24.5% and 29.1%, respectively, over five reasoning benchmarks using the Qwen2.5-32B guided by R1-Distill-Qwen-1.5B, a model 21x smaller. Moreover, we find that ThinkLogit remains effective when the guider and target come from different model families. It is also orthogonal to post-training methods for small models, as guiders improved through supervised distillation or reinforcement learning can be directly plugged in to yield stronger large models, offering a practical path to unlock long reasoning in large-scale models without costly post-training.
>
---
#### [new 073] NL2GenSym: Natural Language to Generative Symbolic Rules for SOAR Cognitive Architecture via Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与认知架构结合的任务，旨在解决SOAR系统手动编写规则效率低的问题。作者提出NL2GenSym框架，利用大语言模型自动生成可执行的符号规则，并通过执行反馈优化规则质量，实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2510.09355v1](http://arxiv.org/pdf/2510.09355v1)**

> **作者:** Fang Yuan; Junjie Zeng; Yue Hu; Zhengqiu Zhu; Quanjun Yin; Yuxiang Xie
>
> **摘要:** SOAR, a classic symbol-based cognitive architecture, has been fostering the development of general, human-like intelligent agents. Nevertheless, its practical adoption is hindered by the laborious manual rule coding. Emerging Large Language Models (LLMs) present the immense potential for efficient rules generation. However, there is a critical gap that current research predominantly focuses on conceptual frameworks and lacks robust experimental validation. To bridge this gap, we propose \textit{N}atural \textit{L}anguage to \textit{Gen}erative \textit{Sym}bolic Rules (NL2GenSym), a novel framework that integrates LLMs with SOAR to autonomously produce generative symbolic rules from natural language. Specifically, our framework introduces a novel Execution-Grounded Generator-Critic mechanism. The LLM-based Generator, guided by a Retrieval-Augmented Generation-accessed self-evolving domain knowledge base, proposes rules from natural language. Subsequently, these rules are immediately executed within the SOAR environment to rigorously validate their correctness. Based on this execution-grounded feedback, a reflective LLM-based Critic drives the iterative refinement of these rules. Experiments on our specialized Water Jug Problem (WJP) dataset, utilizing both Gemini and Qwen series models, validate the efficacy of our framework. It achieves a success rate over 86\% in generating rules from natural language. Crucially, the framework also generates novel heuristic rules, reducing average decision cycles for solving the WJP to 1.98 times the optimal solution and 1/1000 of baseline methods. Additionally, our initial experiments show that NL2GenSym enables smaller-parameter models to achieve better performance than larger counterparts.
>
---
#### [new 074] Can We Reliably Rank Model Performance across Domains without Labeled Data?
- **分类: cs.CL**

- **简介: 该论文属于NLP模型性能评估任务，旨在解决在无标签数据情况下，如何可靠地跨领域评估模型性能问题。论文通过实验分析不同方法在多个数据集上的排名相关性，探讨影响排名可靠性的因素，指导模型评估方法的选择与应用。**

- **链接: [http://arxiv.org/pdf/2510.09519v1](http://arxiv.org/pdf/2510.09519v1)**

> **作者:** Veronica Rammouz; Aaron Gonzalez; Carlos Cruzportillo; Adrian Tan; Nicole Beebe; Anthony Rios
>
> **备注:** 8 pages + references and Appendix
>
> **摘要:** Estimating model performance without labels is an important goal for understanding how NLP models generalize. While prior work has proposed measures based on dataset similarity or predicted correctness, it remains unclear when these estimates produce reliable performance rankings across domains. In this paper, we analyze the factors that affect ranking reliability using a two-step evaluation setup with four base classifiers and several large language models as error predictors. Experiments on the GeoOLID and Amazon Reviews datasets, spanning 15 domains, show that large language model-based error predictors produce stronger and more consistent rank correlations with true accuracy than drift-based or zero-shot baselines. Our analysis reveals two key findings: ranking is more reliable when performance differences across domains are larger, and when the error model's predictions align with the base model's true failure patterns. These results clarify when performance estimation methods can be trusted and provide guidance for their use in cross-domain model evaluation.
>
---
#### [new 075] WUGNECTIVES: Novel Entity Inferences of Language Models from Discourse Connectives
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在研究话语连接词是否能帮助语言模型推断新实体属性。论文构建了包含8880个样本的数据集WUGNECTIVES，评估17种语言模型在不同连接词下的推理能力，发现调整模型可提升推理表现，但对表达让步意义的连接词仍存在挑战。**

- **链接: [http://arxiv.org/pdf/2510.09556v1](http://arxiv.org/pdf/2510.09556v1)**

> **作者:** Daniel Brubaker; William Sheffield; Junyi Jessy Li; Kanishka Misra
>
> **备注:** 16 pages total, 9 pages main; 7 figures total, 4 figures main; 8 tables total, 4 tables main
>
> **摘要:** The role of world knowledge has been particularly crucial to predict the discourse connective that marks the discourse relation between two arguments, with language models (LMs) being generally successful at this task. We flip this premise in our work, and instead study the inverse problem of understanding whether discourse connectives can inform LMs about the world. To this end, we present WUGNECTIVES, a dataset of 8,880 stimuli that evaluates LMs' inferences about novel entities in contexts where connectives link the entities to particular attributes. On investigating 17 different LMs at various scales, and training regimens, we found that tuning an LM to show reasoning behavior yields noteworthy improvements on most connectives. At the same time, there was a large variation in LMs' overall performance across connective type, with all models systematically struggling on connectives that express a concessive meaning. Our findings pave the way for more nuanced investigations into the functional role of language cues as captured by LMs. We release WUGNECTIVES at https://github.com/sheffwb/wugnectives.
>
---
#### [new 076] Identifying & Interactively Refining Ambiguous User Goals for Data Visualization Code Generation
- **分类: cs.CL; cs.AI; cs.CV; cs.HC; cs.MA**

- **简介: 该论文属于自然语言处理与数据可视化任务，旨在解决用户目标模糊导致生成的可视化代码不符合用户意图的问题。论文提出了一种模糊类型分类法和量化指标，并探索多轮对话减少模糊性的方法，通过模拟用户实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2510.09390v1](http://arxiv.org/pdf/2510.09390v1)**

> **作者:** Mert İnan; Anthony Sicilia; Alex Xie; Saujas Vaduguru; Daniel Fried; Malihe Alikhani
>
> **摘要:** Establishing shared goals is a fundamental step in human-AI communication. However, ambiguities can lead to outputs that seem correct but fail to reflect the speaker's intent. In this paper, we explore this issue with a focus on the data visualization domain, where ambiguities in natural language impact the generation of code that visualizes data. The availability of multiple views on the contextual (e.g., the intended plot and the code rendering the plot) allows for a unique and comprehensive analysis of diverse ambiguity types. We develop a taxonomy of types of ambiguity that arise in this task and propose metrics to quantify them. Using Matplotlib problems from the DS-1000 dataset, we demonstrate that our ambiguity metrics better correlate with human annotations than uncertainty baselines. Our work also explores how multi-turn dialogue can reduce ambiguity, therefore, improve code accuracy by better matching user goals. We evaluate three pragmatic models to inform our dialogue strategies: Gricean Cooperativity, Discourse Representation Theory, and Questions under Discussion. A simulated user study reveals how pragmatic dialogues reduce ambiguity and enhance code accuracy, highlighting the value of multi-turn exchanges in code generation.
>
---
#### [new 077] CFVBench: A Comprehensive Video Benchmark for Fine-grained Multimodal Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于多模态视频理解任务，旨在解决现有视频多模态检索增强生成（MRAG）基准在模态覆盖和细节理解上的不足。作者构建了大规模数据集CFVBench，并提出自适应视觉细化框架AVR，提升模型对视频中细粒度多模态信息的理解与生成能力。**

- **链接: [http://arxiv.org/pdf/2510.09266v1](http://arxiv.org/pdf/2510.09266v1)**

> **作者:** Kaiwen Wei; Xiao Liu; Jie Zhang; Zijian Wang; Ruida Liu; Yuming Yang; Xin Xiao; Xiao Sun; Haoyang Zeng; Changzai Pan; Yidan Zhang; Jiang Zhong; Peijin Wang; Yingchao Feng
>
> **摘要:** Multimodal Retrieval-Augmented Generation (MRAG) enables Multimodal Large Language Models (MLLMs) to generate responses with external multimodal evidence, and numerous video-based MRAG benchmarks have been proposed to evaluate model capabilities across retrieval and generation stages. However, existing benchmarks remain limited in modality coverage and format diversity, often focusing on single- or limited-modality tasks, or coarse-grained scene understanding. To address these gaps, we introduce CFVBench, a large-scale, manually verified benchmark constructed from 599 publicly available videos, yielding 5,360 open-ended QA pairs. CFVBench spans high-density formats and domains such as chart-heavy reports, news broadcasts, and software tutorials, requiring models to retrieve and reason over long temporal video spans while maintaining fine-grained multimodal information. Using CFVBench, we systematically evaluate 7 retrieval methods and 14 widely-used MLLMs, revealing a critical bottleneck: current models (even GPT5 or Gemini) struggle to capture transient yet essential fine-grained multimodal details. To mitigate this, we propose Adaptive Visual Refinement (AVR), a simple yet effective framework that adaptively increases frame sampling density and selectively invokes external tools when necessary. Experiments show that AVR consistently enhances fine-grained multimodal comprehension and improves performance across all evaluated MLLMs
>
---
#### [new 078] Beyond Surface Reasoning: Unveiling the True Long Chain-of-Thought Capacity of Diffusion Large Language Models
- **分类: cs.CL**

- **简介: 论文研究扩散大语言模型（DLLMs）在复杂推理任务中的局限性，提出“并行-序列矛盾”（PSC）问题，分析其影响并提出缓解策略，以提升模型效率与推理能力。**

- **链接: [http://arxiv.org/pdf/2510.09544v1](http://arxiv.org/pdf/2510.09544v1)**

> **作者:** Qiguang Chen; Hanjing Li; Libo Qin; Dengyun Peng; Jinhao Liu; Jiangyi Wang; Chengyue Wu; Xie Chen; Yantao Du; Wanxiang Che
>
> **备注:** Preprint
>
> **摘要:** Recently, Diffusion Large Language Models (DLLMs) have offered high throughput and effective sequential reasoning, making them a competitive alternative to autoregressive LLMs (ALLMs). However, parallel decoding, which enables simultaneous token updates, conflicts with the causal order often required for rigorous reasoning. We first identify this conflict as the core Parallel-Sequential Contradiction (PSC). Behavioral analyses in both simple and complex reasoning tasks show that DLLMs exhibit genuine parallelism only for directly decidable outputs. As task difficulty increases, they revert to autoregressive-like behavior, a limitation exacerbated by autoregressive prompting, which nearly doubles the number of decoding steps with remasking without improving quality. Moreover, PSC restricts DLLMs' self-reflection, reasoning depth, and exploratory breadth. To further characterize PSC, we introduce three scaling dimensions for DLLMs: parallel, diffusion, and sequential. Empirically, while parallel scaling yields consistent improvements, diffusion and sequential scaling are constrained by PSC. Based on these findings, we propose several practical mitigations, parallel-oriented prompting, diffusion early stopping, and parallel scaling, to reduce PSC-induced ineffectiveness and inefficiencies.
>
---
#### [new 079] Artificial Impressions: Evaluating Large Language Model Behavior Through the Lens of Trait Impressions
- **分类: cs.CL**

- **简介: 该论文研究大型语言模型（LLM）内部表示中的“人工印象”，即语言引发的类似人类刻板印象的模式。它属于自然语言处理任务，旨在分析模型如何通过提示生成内容并影响输出质量。研究者使用线性探针预测印象，并探讨其与提示特征及模型行为的关系。**

- **链接: [http://arxiv.org/pdf/2510.08915v1](http://arxiv.org/pdf/2510.08915v1)**

> **作者:** Nicholas Deas; Kathleen McKeown
>
> **备注:** EMNLP 2025 Camera Ready
>
> **摘要:** We introduce and study artificial impressions--patterns in LLMs' internal representations of prompts that resemble human impressions and stereotypes based on language. We fit linear probes on generated prompts to predict impressions according to the two-dimensional Stereotype Content Model (SCM). Using these probes, we study the relationship between impressions and downstream model behavior as well as prompt features that may inform such impressions. We find that LLMs inconsistently report impressions when prompted, but also that impressions are more consistently linearly decodable from their hidden representations. Additionally, we show that artificial impressions of prompts are predictive of the quality and use of hedging in model responses. We also investigate how particular content, stylistic, and dialectal features in prompts impact LLM impressions.
>
---
#### [new 080] Automated Refinement of Essay Scoring Rubrics for Language Models via Reflect-and-Revise
- **分类: cs.CL**

- **简介: 该论文属于自动作文评分任务，旨在解决如何提升大语言模型评分与人工评分的一致性问题。作者提出了一种通过模型自我反思与迭代优化评分量规的方法，实验证明该方法在多个数据集上显著提升了评分一致性。**

- **链接: [http://arxiv.org/pdf/2510.09030v1](http://arxiv.org/pdf/2510.09030v1)**

> **作者:** Keno Harada; Lui Yoshida; Takeshi Kojima; Yusuke Iwasawa; Yutaka Matsuo
>
> **摘要:** The performance of Large Language Models (LLMs) is highly sensitive to the prompts they are given. Drawing inspiration from the field of prompt optimization, this study investigates the potential for enhancing Automated Essay Scoring (AES) by refining the scoring rubrics used by LLMs. Specifically, our approach prompts models to iteratively refine rubrics by reflecting on models' own scoring rationales and observed discrepancies with human scores on sample essays. Experiments on the TOEFL11 and ASAP datasets using GPT-4.1, Gemini-2.5-Pro, and Qwen-3-Next-80B-A3B-Instruct show Quadratic Weighted Kappa (QWK) improvements of up to 0.19 and 0.47, respectively. Notably, even with a simple initial rubric, our approach achieves comparable or better QWK than using detailed human-authored rubrics. Our findings highlight the importance of iterative rubric refinement in LLM-based AES to enhance alignment with human evaluations.
>
---
#### [new 081] PARSE: LLM Driven Schema Optimization for Reliable Entity Extraction
- **分类: cs.CL; cs.LG**

- **简介: 论文提出PARSE系统，优化LLM用于实体抽取的JSON模式，解决因模式设计不完善导致的抽取错误和不稳定问题。其任务是结构化信息抽取，属于自然语言处理与软件工程交叉领域。工作包括自动优化模式的ARCHITECT模块和提升抽取准确性的SCOPE模块。**

- **链接: [http://arxiv.org/pdf/2510.08623v1](http://arxiv.org/pdf/2510.08623v1)**

> **作者:** Anubhav Shrimal; Aryan Jain; Soumyajit Chowdhury; Promod Yenigalla
>
> **备注:** EMNLP 2025 Industry Track
>
> **摘要:** Structured information extraction from unstructured text is critical for emerging Software 3.0 systems where LLM agents autonomously interact with APIs and tools. Recent approaches apply large language models directly to extraction tasks using existing JSON schemas, often with constraint decoding or reinforcement learning approaches to ensure syntactic validity, but treat JSON schemas as static contracts designed for human developers, leading to suboptimal extraction performance, frequent hallucinations, and unreliable agent behavior when schemas contain ambiguous or incomplete specifications. We recognize that JSON schemas themselves are a form of natural language understanding contract that encodes rules, relationships, and expectations about data structure contracts that LLMs should be able to both interpret and systematically improve. Consequently, we develop PARSE (Parameter Automated Refinement and Schema Extraction), a novel system with two synergistic components: ARCHITECT, which autonomously optimizes JSON schemas for LLM consumption while maintaining backward compatibility through RELAY (an integrated code generation system), and SCOPE, which implements reflection-based extraction with combined static and LLM-based guardrails. We evaluate PARSE qualitatively and quantitatively on three datasets including Schema-Guided Dialogue (SGD), Structured Web Data Extraction (SWDE), and internal retail conversation data, and find that it achieves up to 64.7% improvement in extraction accuracy on SWDE with combined framework improvements reaching 10% across models, while reducing extraction errors by 92% within the first retry and and maintaining practical latency.
>
---
#### [new 082] Verifying Chain-of-Thought Reasoning via Its Computational Graph
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理错误的检测与理解问题。通过构建基于计算图的白盒验证方法CRV，分析推理过程的结构特征，识别错误推理的因果模式，并实现错误预测与干预修正。**

- **链接: [http://arxiv.org/pdf/2510.09312v1](http://arxiv.org/pdf/2510.09312v1)**

> **作者:** Zheng Zhao; Yeskendir Koishekenov; Xianjun Yang; Naila Murray; Nicola Cancedda
>
> **摘要:** Current Chain-of-Thought (CoT) verification methods predict reasoning correctness based on outputs (black-box) or activations (gray-box), but offer limited insight into why a computation fails. We introduce a white-box method: Circuit-based Reasoning Verification (CRV). We hypothesize that attribution graphs of correct CoT steps, viewed as execution traces of the model's latent reasoning circuits, possess distinct structural fingerprints from those of incorrect steps. By training a classifier on structural features of these graphs, we show that these traces contain a powerful signal of reasoning errors. Our white-box approach yields novel scientific insights unattainable by other methods. (1) We demonstrate that structural signatures of error are highly predictive, establishing the viability of verifying reasoning directly via its computational graph. (2) We find these signatures to be highly domain-specific, revealing that failures in different reasoning tasks manifest as distinct computational patterns. (3) We provide evidence that these signatures are not merely correlational; by using our analysis to guide targeted interventions on individual transcoder features, we successfully correct the model's faulty reasoning. Our work shows that, by scrutinizing a model's computational process, we can move from simple error detection to a deeper, causal understanding of LLM reasoning.
>
---
#### [new 083] From What to Why: Thought-Space Recommendation with Small Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于推荐系统任务，旨在解决小语言模型（SLM）在推荐中的推理能力不足问题。论文提出PULSE框架，利用SLM生成的推理文本作为监督信号，联合建模用户行为及其语义驱动因素，提升推荐的鲁棒性、泛化性与跨域迁移能力。**

- **链接: [http://arxiv.org/pdf/2510.08626v1](http://arxiv.org/pdf/2510.08626v1)**

> **作者:** Prosenjit Biswas; Pervez Shaik; Abhinav Thorat; Ravi Kolla; Niranjan Pedanekar
>
> **备注:** 15 pages, 3 figures
>
> **摘要:** Large Language Models (LLMs) have advanced recommendation capabilities through enhanced reasoning, but pose significant challenges for real-world deployment due to high inference costs. Conversely, while Small Language Models (SLMs) offer an efficient alternative, their reasoning capabilities for recommendation remain underexplored. Existing systems often use natural language rationales merely as unsupervised descriptive text, failing to harness their full potential as learning signals. In this work our main idea is to create a common understanding of user and items across multiple domains called Thought Space with SLMs instead of using LLMs' distilled knowledge. To that end we propose PULSE (Preference Understanding by Latent Semantic Embeddings), a framework that treats SLM-generated rationales as director learning signals, supervising them with interaction histories to jointly model user actions (what) and their semantic drivers (why). Existing methods consider only interactions such as sequences and embeddings, whereas PULSE treats rationales as first-class signals, this novel design yields embeddings that are more robust and generalizable. Extensive experiments demonstrate that PULSE outperforms leading ID, Collaborative Filtering (CF), and LLM-based sequential recommendation models across multiple benchmark datasets. Furthermore, PULSE exhibits superior transferability in cross-domain recommendation and demonstrates strong performance on downstream tasks such as reasoning-oriented question answering. Our code is available \href{https://anonymous.4open.science/r/Thinking_PULSE-0FC5/README.md}{here}.
>
---
#### [new 084] Stronger Re-identification Attacks through Reasoning and Aggregation
- **分类: cs.CL**

- **简介: 该论文属于文本去标识安全性评估任务，旨在解决通过再识别攻击揭示匿名化文本中个人信息的问题。论文提出了两种增强再识别攻击的方法：一是通过聚合不同识别顺序的预测结果，二是利用推理模型结合背景知识提升攻击效果。**

- **链接: [http://arxiv.org/pdf/2510.09184v1](http://arxiv.org/pdf/2510.09184v1)**

> **作者:** Lucas Georges Gabriel Charpentier; Pierre Lison
>
> **摘要:** Text de-identification techniques are often used to mask personally identifiable information (PII) from documents. Their ability to conceal the identity of the individuals mentioned in a text is, however, hard to measure. Recent work has shown how the robustness of de-identification methods could be assessed by attempting the reverse process of _re-identification_, based on an automated adversary using its background knowledge to uncover the PIIs that have been masked. This paper presents two complementary strategies to build stronger re-identification attacks. We first show that (1) the _order_ in which the PII spans are re-identified matters, and that aggregating predictions across multiple orderings leads to improved results. We also find that (2) reasoning models can boost the re-identification performance, especially when the adversary is assumed to have access to extensive background knowledge.
>
---
#### [new 085] IRIS: An Iterative and Integrated Framework for Verifiable Causal Discovery in the Absence of Tabular Data
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于因果发现任务，旨在解决传统方法依赖表格数据、计算冗余及无法发现新因果关系的问题。论文提出IRIS框架，结合统计算法与大语言模型，从初始变量出发，自动收集文档、提取变量并实时发现已知与新颖的因果关系，同时补充缺失变量，扩展因果图。**

- **链接: [http://arxiv.org/pdf/2510.09217v1](http://arxiv.org/pdf/2510.09217v1)**

> **作者:** Tao Feng; Lizhen Qu; Niket Tandon; Gholamreza Haffari
>
> **备注:** ACL 2025
>
> **摘要:** Causal discovery is fundamental to scientific research, yet traditional statistical algorithms face significant challenges, including expensive data collection, redundant computation for known relations, and unrealistic assumptions. While recent LLM-based methods excel at identifying commonly known causal relations, they fail to uncover novel relations. We introduce IRIS (Iterative Retrieval and Integrated System for Real-Time Causal Discovery), a novel framework that addresses these limitations. Starting with a set of initial variables, IRIS automatically collects relevant documents, extracts variables, and uncovers causal relations. Our hybrid causal discovery method combines statistical algorithms and LLM-based methods to discover known and novel causal relations. In addition to causal discovery on initial variables, the missing variable proposal component of IRIS identifies and incorporates missing variables to expand the causal graphs. Our approach enables real-time causal discovery from only a set of initial variables without requiring pre-existing datasets.
>
---
#### [new 086] Learning What to Remember: Adaptive Probabilistic Memory Retention for Memory-Efficient Language Models
- **分类: cs.CL**

- **简介: 论文提出“自适应保留”机制，通过学习选择重要token以减少内存消耗，解决Transformer模型处理长文本时内存占用高的问题。在多种任务中，保留30-50% token即可维持95%以上性能，提升吞吐量。方法不依赖特定架构，适用于标准编码器。**

- **链接: [http://arxiv.org/pdf/2510.08798v1](http://arxiv.org/pdf/2510.08798v1)**

> **作者:** S M Rafiuddin; Muntaha Nujat Khan
>
> **备注:** 14 Pages, 2 Figures, 6 Table, Accepted at EMNLP 2025 Findings as a Short Paper
>
> **摘要:** Transformer attention scales quadratically with sequence length O(n^2), limiting long-context use. We propose Adaptive Retention, a probabilistic, layer-wise token selection mechanism that learns which representations to keep under a strict global budget M. Retention is modeled with Bernoulli gates trained via a Hard-Concrete/variational relaxation and enforced with a simple top-M rule at inference, making the method differentiable and drop-in for standard encoders. Across classification, extractive QA, and long-document summarization, keeping only 30-50% of tokens preserves >= 95% of full-model performance while cutting peak memory by ~35-45% and improving throughput by up to ~1.8x. This architecture-agnostic approach delivers practical long-context efficiency without modifying base attention or task heads.
>
---
#### [new 087] CLARity: Reasoning Consistency Alone Can Teach Reinforced Experts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决小数据领域训练专家大模型时推理质量下降的问题。作者提出CLARity框架，通过一致性感知奖励机制和两阶段训练流程，提升模型的推理一致性和准确性，实验证明效果显著。**

- **链接: [http://arxiv.org/pdf/2510.09278v1](http://arxiv.org/pdf/2510.09278v1)**

> **作者:** Jiuheng Lin; Cong Jiang; Zirui Wu; Jiarui Sun; Yansong Feng
>
> **摘要:** Training expert LLMs in domains with scarce data is difficult, often relying on multiple-choice questions (MCQs). However, standard outcome-based reinforcement learning (RL) on MCQs is risky. While it may improve accuracy, we observe it often degrades reasoning quality such as logical consistency. Existing solutions to supervise reasoning, such as large-scale Process Reward Models (PRMs), are prohibitively expensive. To address this, we propose CLARity, a cost-effective RL framework that enhances reasoning quality using only a small, general-purpose LLM. CLARity integrates a consistency-aware reward mechanism with a 2-stage refine-then-monitor training pipeline to enhance reasoning consistency, and a dynamic data reformulation strategy to to better exploit limited data. Experiments demonstrate that CLARity improves response consistency by 16.5% and accuracy by 7.5% over baselines. Human evaluations further confirm holistic improvements in coherence and professionalism. Thus, CLARity offers a generalizable solution that enables smaller models to effectively guide expert models by reasoning consistency.Our code is open sourced at: https://github.com/Infinite-set/CLARity
>
---
#### [new 088] From Simulation to Strategy: Automating Personalized Interaction Planning for Conversational Agents
- **分类: cs.CL**

- **简介: 该论文属于对话策略优化任务，旨在提升销售导向对话系统的效果。通过模拟不同用户画像，发现职业对对话意图影响最大。基于此，作者设计了轻量级职业适配策略，引导代理优先选择符合用户偏好的意图，从而缩短对话并提高成功率。**

- **链接: [http://arxiv.org/pdf/2510.08621v1](http://arxiv.org/pdf/2510.08621v1)**

> **作者:** Wen-Yu Chang; Tzu-Hung Huang; Chih-Ho Chen; Yun-Nung Chen
>
> **摘要:** Amid the rapid rise of agentic dialogue models, realistic user-simulator studies are essential for tuning effective conversation strategies. This work investigates a sales-oriented agent that adapts its dialogue based on user profiles spanning age, gender, and occupation. While age and gender influence overall performance, occupation produces the most pronounced differences in conversational intent. Leveraging this insight, we introduce a lightweight, occupation-conditioned strategy that guides the agent to prioritize intents aligned with user preferences, resulting in shorter and more successful dialogues. Our findings highlight the importance of rich simulator profiles and demonstrate how simple persona-informed strategies can enhance the effectiveness of sales-oriented dialogue systems.
>
---
#### [new 089] StatEval: A Comprehensive Benchmark for Large Language Models in Statistics
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与统计教育评估任务，旨在解决当前大语言模型在统计学领域评估不足的问题。作者构建了StatEval，一个包含近1.4万道统计学问题的基准测试，涵盖本硕及研究级别，并设计了评估框架以衡量模型的统计推理能力。**

- **链接: [http://arxiv.org/pdf/2510.09517v1](http://arxiv.org/pdf/2510.09517v1)**

> **作者:** Yuchen Lu; Run Yang; Yichen Zhang; Shuguang Yu; Runpeng Dai; Ziwei Wang; Jiayi Xiang; Wenxin E; Siran Gao; Xinyao Ruan; Yirui Huang; Chenjing Xi; Haibo Hu; Yueming Fu; Qinglan Yu; Xiaobing Wei; Jiani Gu; Rui Sun; Jiaxuan Jia; Fan Zhou
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable advances in mathematical and logical reasoning, yet statistics, as a distinct and integrative discipline, remains underexplored in benchmarking efforts. To address this gap, we introduce \textbf{StatEval}, the first comprehensive benchmark dedicated to statistics, spanning both breadth and depth across difficulty levels. StatEval consists of 13,817 foundational problems covering undergraduate and graduate curricula, together with 2374 research-level proof tasks extracted from leading journals. To construct the benchmark, we design a scalable multi-agent pipeline with human-in-the-loop validation that automates large-scale problem extraction, rewriting, and quality control, while ensuring academic rigor. We further propose a robust evaluation framework tailored to both computational and proof-based tasks, enabling fine-grained assessment of reasoning ability. Experimental results reveal that while closed-source models such as GPT5-mini achieve below 57\% on research-level problems, with open-source models performing significantly lower. These findings highlight the unique challenges of statistical reasoning and the limitations of current LLMs. We expect StatEval to serve as a rigorous benchmark for advancing statistical intelligence in large language models. All data and code are available on our web platform: https://stateval.github.io/.
>
---
#### [new 090] JAI-1: A Thai-Centric Large Language Model
- **分类: cs.CL**

- **简介: 论文提出JAI-1，一个专注于泰语的750亿参数大语言模型。为解决现有泰语模型在微调过程中损害原有知识的问题，JAI-1采用扩容策略，从高性能英文模型出发，扩展参数空间并系统融入泰语知识。预训练涵盖1.5万亿token，其中超3000亿为泰语数据，并经过监督微调与对齐训练。最终模型在多个泰语基准测试中优于Typhoon2-70B，验证了其架构优势。**

- **链接: [http://arxiv.org/pdf/2510.08620v1](http://arxiv.org/pdf/2510.08620v1)**

> **作者:** Attapol T. Rutherford; Jullajak Karnjanaekarin; Narongkorn Panitsrisit; Pontakorn Trakuekul; Sumana Sumanakul; Natchanon Pollertlam
>
> **摘要:** This technical report introduces JAI-1, a Thai-centric language model with 75B parameters. Recent Thai models have primarily relied on existing open-source models, applying additional training without structural modifications to specialize in Thai. However, this approach risks eroding pre-existing knowledge in the model's parameter space during the injection of Thai-specific information, as optimized parameters for general tasks may conflict with new linguistic requirements. In contrast, JAI-1 adopts an upscaling strategy: starting from a smaller, high-performing English open-source LLM, we expanded its parameter space and utilized the newly allocated capacity to systematically integrate Thai-language knowledge. This methodology not only preserves the original model's general intelligence but also establishes a unique architecture distinct from other open-source models, enabling scalable future enhancements. During pre-training, JAI-1 was exposed to 1.5T tokens, including over 300B Thai language tokens. This was followed by post-training stages -- supervised fine-tuning and alignment tuning -- using more than 600K instruction-based examples. The final model demonstrated superior performance compared to Typhoon2-70B on Thai-centric benchmarks (IFEval-TH, MT-Bench-TH, and JAI-Hall-Bench), validating the efficacy of its upscaling and knowledge-integration framework.
>
---
#### [new 091] ShiZhi: A Chinese Lightweight Large Language Model for Court View Generation
- **分类: cs.CL**

- **简介: 该论文属于法律人工智能任务，旨在解决刑事裁判文书“法院观点”自动生成问题。作者构建了包含11万中文案例的数据集CCVG，并基于此训练了专用轻量大模型ShiZhi，在生成质量和法律准确性上均取得良好表现。**

- **链接: [http://arxiv.org/pdf/2510.09297v1](http://arxiv.org/pdf/2510.09297v1)**

> **作者:** Zhitian Hou; Kun Zeng
>
> **摘要:** Criminal Court View Generation (CVG) is a fundamental task in legal artificial intelligence, aiming to automatically generate the "Court View" section of a legal case document. Generating court views is challenging due to the diversity and complexity of case facts, and directly generating from raw facts may limit performance. In this paper, we present ShiZhi, the first large language model (LLM) specifically designed for court view generation. We construct a Chinese Court View Generation dataset, CCVG, of more than 110K cases, each containing fact descriptions paired with corresponding court views. Based on this dataset, ShiZhi achieving 58.5 BLEU-1 on court view generation and 86.1\% accuracy with 92.5\% macro F1 on charge prediction. Experimental results demonstrate that even a small LLM can generate reasonable and legally coherent court views when trained on high-quality domain-specific data. Our model and dataset are available at \href{https://github.com/ZhitianHou/ShiZhi}{https://github.com/ZhitianHou/ShiZhi}.
>
---
#### [new 092] Enhancing Biomedical Named Entity Recognition using GLiNER-BioMed with Targeted Dictionary-Based Post-processing for BioASQ 2025 task 6
- **分类: cs.CL**

- **简介: 论文属于BioASQ任务6中的生物医药命名实体识别（BioNER），旨在解决基因与化学品等相似实体类型难以区分的问题。作者评估了GLiNER-BioMed模型，并引入基于字典的后处理策略以优化分类效果，在开发集上提升F1分数，但在测试集上出现过拟合问题。研究还探讨了其他方法如条件随机场，并强调模型泛化能力的重要性。**

- **链接: [http://arxiv.org/pdf/2510.08588v1](http://arxiv.org/pdf/2510.08588v1)**

> **作者:** Ritesh Mehta
>
> **备注:** Paper published to CLEF 2025 CEUR-WS
>
> **摘要:** Biomedical Named Entity Recognition (BioNER), task6 in BioASQ (A challenge in large-scale biomedical semantic indexing and question answering), is crucial for extracting information from scientific literature but faces hurdles such as distinguishing between similar entity types like genes and chemicals. This study evaluates the GLiNER-BioMed model on a BioASQ dataset and introduces a targeted dictionary-based post-processing strategy to address common misclassifications. While this post-processing approach demonstrated notable improvement on our development set, increasing the micro F1-score from a baseline of 0.79 to 0.83, this enhancement did not generalize to the blind test set, where the post-processed model achieved a micro F1-score of 0.77 compared to the baselines 0.79. We also discuss insights gained from exploring alternative methodologies, including Conditional Random Fields. This work highlights the potential of dictionary-based refinement for pre-trained BioNER models but underscores the critical challenge of overfitting to development data and the necessity of ensuring robust generalization for real-world applicability.
>
---
#### [new 093] Dyna-Mind: Learning to Simulate from Experience for Better AI Agents
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出Dyna-Mind框架，旨在提升AI代理在复杂交互环境中的推理与规划能力。通过结合模拟推理（ReSim）与强化学习（Dyna-GRPO），使AI具备“替代试错”能力，从而更有效地处理长期任务。应用于Sokoban、ALFWorld和AndroidWorld等任务中，验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2510.09577v1](http://arxiv.org/pdf/2510.09577v1)**

> **作者:** Xiao Yu; Baolin Peng; Michel Galley; Hao Cheng; Qianhui Wu; Janardhan Kulkarni; Suman Nath; Zhou Yu; Jianfeng Gao
>
> **摘要:** Reasoning models have recently shown remarkable progress in domains such as math and coding. However, their expert-level abilities in math and coding contrast sharply with their performance in long-horizon, interactive tasks such as web navigation and computer/phone-use. Inspired by literature on human cognition, we argue that current AI agents need ''vicarious trial and error'' - the capacity to mentally simulate alternative futures before acting - in order to enhance their understanding and performance in complex interactive environments. We introduce Dyna-Mind, a two-stage training framework that explicitly teaches (V)LM agents to integrate such simulation into their reasoning. In stage 1, we introduce Reasoning with Simulations (ReSim), which trains the agent to generate structured reasoning traces from expanded search trees built from real experience gathered through environment interactions. ReSim thus grounds the agent's reasoning in faithful world dynamics and equips it with the ability to anticipate future states in its reasoning. In stage 2, we propose Dyna-GRPO, an online reinforcement learning method to further strengthen the agent's simulation and decision-making ability by using both outcome rewards and intermediate states as feedback from real rollouts. Experiments on two synthetic benchmarks (Sokoban and ALFWorld) and one realistic benchmark (AndroidWorld) demonstrate that (1) ReSim effectively infuses simulation ability into AI agents, and (2) Dyna-GRPO leverages outcome and interaction-level signals to learn better policies for long-horizon, planning-intensive tasks. Together, these results highlight the central role of simulation in enabling AI agents to reason, plan, and act more effectively in the ever more challenging environments.
>
---
#### [new 094] Human Texts Are Outliers: Detecting LLM-generated Texts via Out-of-distribution Detection
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本检测任务，旨在区分人类撰写与大语言模型生成的文本。现有方法因将任务视为二分类问题而泛化能力差。论文提出将人类文本视为分布外（OOD）样本，机器生成文本为分布内（ID）样本，采用单类学习和得分方法进行检测，提升了检测效果与跨语言、跨模型的泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.08602v1](http://arxiv.org/pdf/2510.08602v1)**

> **作者:** Cong Zeng; Shengkun Tang; Yuanzhou Chen; Zhiqiang Shen; Wenchao Yu; Xujiang Zhao; Haifeng Chen; Wei Cheng; Zhiqiang Xu
>
> **摘要:** The rapid advancement of large language models (LLMs) such as ChatGPT, DeepSeek, and Claude has significantly increased the presence of AI-generated text in digital communication. This trend has heightened the need for reliable detection methods to distinguish between human-authored and machine-generated content. Existing approaches both zero-shot methods and supervised classifiers largely conceptualize this task as a binary classification problem, often leading to poor generalization across domains and models. In this paper, we argue that such a binary formulation fundamentally mischaracterizes the detection task by assuming a coherent representation of human-written texts. In reality, human texts do not constitute a unified distribution, and their diversity cannot be effectively captured through limited sampling. This causes previous classifiers to memorize observed OOD characteristics rather than learn the essence of `non-ID' behavior, limiting generalization to unseen human-authored inputs. Based on this observation, we propose reframing the detection task as an out-of-distribution (OOD) detection problem, treating human-written texts as distributional outliers while machine-generated texts are in-distribution (ID) samples. To this end, we develop a detection framework using one-class learning method including DeepSVDD and HRN, and score-based learning techniques such as energy-based method, enabling robust and generalizable performance. Extensive experiments across multiple datasets validate the effectiveness of our OOD-based approach. Specifically, the OOD-based method achieves 98.3% AUROC and AUPR with only 8.9% FPR95 on DeepFake dataset. Moreover, we test our detection framework on multilingual, attacked, and unseen-model and -domain text settings, demonstrating the robustness and generalizability of our framework. Code, pretrained weights, and demo will be released.
>
---
#### [new 095] DICE: Structured Reasoning in LLMs through SLM-Guided Chain-of-Thought Correction
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DICE框架，解决大语言模型（LLMs）在用户特定结构化输出要求下表现不佳的问题。通过小语言模型（SLMs）对LLMs的自然语言输出进行链式思维修正，提升输出的格式准确性和内容正确性。属于自然语言处理任务中的结构化推理与输出优化方向。**

- **链接: [http://arxiv.org/pdf/2510.09211v1](http://arxiv.org/pdf/2510.09211v1)**

> **作者:** Yiqi Li; Yusheng Liao; Zhe Chen; Yanfeng Wang; Yu Wang
>
> **摘要:** When performing reasoning tasks with user-specific requirements, such as strict output formats, large language models (LLMs) often prioritize reasoning over adherence to detailed instructions. Fine-tuning LLMs on supervised datasets to address this is impractical due to high computational costs and limited parameter access. To tackle this, we propose DICE, a lightweight framework that guides small language models (SLMs) to refine LLMs' outputs through chain-of-thought (CoT) correction. DICE decouples the process by first prompting LLMs to generate natural language responses, then using trained SLMs to analyze and refine these outputs to meet structured output specifications. This framework preserves LLMs' broad knowledge and reasoning capabilities while ensuring the outputs conform to user demands. Specifically, DICE first constructs structured CoT adaptation datasets via a two-stage method and subsequently applies a dual-tuning strategy to fine-tune SLMs for generating structured outputs in an analyze-then-answer pattern. Experiments demonstrate that DICE improves the average format accuracy and content correctness of LLM outputs by 35.4\% and 29.4\%, respectively, achieving state-of-the-art (SOTA) performance over other competitive baselines.
>
---
#### [new 096] Decoupling Safety into Orthogonal Subspace: Cost-Efficient and Performance-Preserving Alignment for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型的安全对齐任务，旨在解决提升模型安全性的同时保持其通用性能的问题。论文提出基于LoRA的拒绝训练方法，实验证明其可在仅使用安全数据的情况下实现性能保留的安全对齐，并理论分析LoRA将安全性解耦至低秩正交子空间，避免干扰模型原有能力。**

- **链接: [http://arxiv.org/pdf/2510.09004v1](http://arxiv.org/pdf/2510.09004v1)**

> **作者:** Yutao Mou; Xiaoling Zhou; Yuxiao Luo; Shikun Zhang; Wei Ye
>
> **备注:** Work in Progress
>
> **摘要:** Safety alignment is essential for building trustworthy artificial intelligence, yet it remains challenging to enhance model safety without degrading general performance. Current approaches require computationally expensive searches for the optimal proportion of safety-critical and general-purpose data to balance safety and general performance, incurring high costs with limited gains. In this work, we show that LoRA-based Refusal-training enables performance-preserving safety alignment even when trained solely on safety data, demonstrating that LoRA serves as cost-efficient, performance-preserving, and plug-and-play safety patches. Beyond empirical findings, we provide both theoretical and experimental evidence that LoRA effectively decouples safety into a low-rank subspace largely orthogonal to the model's intrinsic transformation space, ensuring that safety enhancements do not interfere with inherent capabilities.
>
---
#### [new 097] Detecting Data Contamination from Reinforcement Learning Post-training for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在强化学习后训练阶段的数据污染检测问题。现有方法无法有效检测此阶段的污染。论文提出Self-Critique方法，利用模型输出熵分布变化检测策略崩溃，并构建RL-MIA基准进行实验，结果显示其检测效果显著优于基线方法。**

- **链接: [http://arxiv.org/pdf/2510.09259v1](http://arxiv.org/pdf/2510.09259v1)**

> **作者:** Yongding Tao; Tian Wang; Yihong Dong; Huanyu Liu; Kechi Zhang; Xiaolong Hu; Ge Li
>
> **摘要:** Data contamination poses a significant threat to the reliable evaluation of Large Language Models (LLMs). This issue arises when benchmark samples may inadvertently appear in training sets, compromising the validity of reported performance. While detection methods have been developed for the pre-training and Supervised Fine-Tuning stages, a critical research gap exists for the increasingly significant phase of Reinforcement Learning (RL) post-training. As RL post-training becomes pivotal for advancing LLM reasoning, the absence of specialized contamination detection methods in this paradigm presents a critical vulnerability. To address this, we conduct the first systematic study of data detection within RL post-training scenario and propose Self-Critique. Our method is motivated by a key observation: after RL phase, the output entropy distribution of LLMs tends to collapse into highly specific and sparse modes. Self-Critique probes for the underlying policy collapse, i.e., the model's convergence to a narrow reasoning path, which causes this entropy reduction. To facilitate this research, we also introduce RL-MIA, a benchmark constructed to simulate this specific contamination scenario. Extensive experiments show that Self-Critique significantly outperforms baseline methods across multiple models and contamination tasks, achieving an AUC improvement of up to 30%. Whereas existing methods are close to a random guess for RL-phase contamination, our method makes detection possible.
>
---
#### [new 098] Creation of the Chinese Adaptive Policy Communication Corpus
- **分类: cs.CL; cs.CE; cs.CY**

- **简介: 该论文构建了CAPC-CG语料库，属于自然语言处理任务，旨在解决政策文本中清晰与模糊语言分类问题。论文标注了3.3百万段落，采用五色分类法，基于Ang理论，提供了高一致性标注数据与基线模型结果，支持政策沟通研究与多语言NLP发展。**

- **链接: [http://arxiv.org/pdf/2510.08986v1](http://arxiv.org/pdf/2510.08986v1)**

> **作者:** Bolun Sun; Charles Chang; Yuen Yuen Ang; Pingxu Hao; Ruotong Mu; Yuchen Xu; Zhengxin Zhang
>
> **摘要:** We introduce CAPC-CG, the Chinese Adaptive Policy Communication (Central Government) Corpus, the first open dataset of Chinese policy directives annotated with a five-color taxonomy of clear and ambiguous language categories, building on Ang's theory of adaptive policy communication. Spanning 1949-2023, this corpus includes national laws, administrative regulations, and ministerial rules issued by China's top authorities. Each document is segmented into paragraphs, producing a total of 3.3 million units. Alongside the corpus, we release comprehensive metadata, a two-round labeling framework, and a gold-standard annotation set developed by expert and trained coders. Inter-annotator agreement achieves a Fleiss's kappa of K = 0.86 on directive labels, indicating high reliability for supervised modeling. We provide baseline classification results with several large language models (LLMs), together with our annotation codebook, and describe patterns from the dataset. This release aims to support downstream tasks and multilingual NLP research in policy communication.
>
---
#### [new 099] Recover-LoRA: Data-Free Accuracy Recovery of Degraded Language Models via Low-Rank Adaptation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型优化任务，旨在解决模型因量化、剪枝等操作导致的性能下降问题。论文提出Recover-LoRA方法，利用合成数据和蒸馏技术恢复模型准确率。实验表明，该方法在多种小型语言模型上可提升5-17%的准确率。**

- **链接: [http://arxiv.org/pdf/2510.08600v1](http://arxiv.org/pdf/2510.08600v1)**

> **作者:** Devleena Das; Rajeev Patwari; Ashish Sirasao
>
> **备注:** Accepted to EMNLP 2025 Industry Track
>
> **摘要:** Inference optimizations such as quantization, pruning, format and datatype conversion, model export, and serialization can lead to functional degradations in language model task performance. While most efforts on performance recovery for deployment focus on robust quantization techniques, we focus on recovering model accuracies from any sources that degrade model weights, such as improper model serialization. In this work, we propose Recover-LoRA, a lightweight and dataset agnostic method to recover accuracy in degraded models. Recover-LoRA uses synthetic data and logit distillation to learn LoRA adapters on selective layers that facilitate aligning the degraded model to its full precision model. We investigate the utility of Recover-LoRA across a diverse set of small language models (SLMs), including models with varying attention architectures, multi-head attention (MHA) and group-query attention (GQA), as well as several evaluation datasets. Our results show that Recover-LoRA recovers model accuracies by 5-17% on MHA and GQA SLMs.
>
---
#### [new 100] Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在多语言拼写错误下的鲁棒性。任务是评估模型在真实用户输入场景中的表现。工作包括构建多语言拼写错误生成算法 MulTypo，并测试18个LLM在五项任务中的表现，发现拼写错误显著影响生成和推理任务，高资源语言更鲁棒。建议加强噪声训练和多语言评估。**

- **链接: [http://arxiv.org/pdf/2510.09536v1](http://arxiv.org/pdf/2510.09536v1)**

> **作者:** Yihong Liu; Raoyuan Zhao; Lena Altinger; Hinrich Schütze; Michael A. Hedderich
>
> **备注:** preprint
>
> **摘要:** Large language models (LLMs) are increasingly deployed in multilingual, real-world applications with user inputs -- naturally introducing typographical errors (typos). Yet most benchmarks assume clean input, leaving the robustness of LLMs to typos across languages largely underexplored. To address this gap, we introduce MulTypo, a multilingual typo generation algorithm that simulates human-like errors based on language-specific keyboard layouts and typing behavior. We evaluate 18 open-source LLMs across three model families and five downstream tasks spanning language inference, multi-choice question answering, mathematical reasoning, and machine translation tasks. Our results show that typos consistently degrade performance, particularly in generative tasks and those requiring reasoning -- while the natural language inference task is comparatively more robust. Instruction tuning improves clean-input performance but may increase brittleness under noise. We also observe language-dependent robustness: high-resource languages are generally more robust than low-resource ones, and translation from English is more robust than translation into English. Our findings underscore the need for noise-aware training and multilingual robustness evaluation. We make our code and data publicly available.
>
---
#### [new 101] GraphGhost: Tracing Structures Behind Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与模型解释任务，旨在探索大语言模型推理能力背后的结构机制。论文提出GraphGhost框架，将神经元激活与信号传播建模为图结构，利用图算法分析模型行为，并通过结构干预验证关键神经元的作用，从而揭示模型的结构语义机制。**

- **链接: [http://arxiv.org/pdf/2510.08613v1](http://arxiv.org/pdf/2510.08613v1)**

> **作者:** Xinnan Dai; Kai Guo; Chung-Hsiang Lo; Shenglai Zeng; Jiayuan Ding; Dongsheng Luo; Subhabrata Mukherjee; Jiliang Tang
>
> **摘要:** Large Language Models (LLMs) demonstrate remarkable reasoning capabilities, yet the structural mechanisms underlying these abilities remain under explored. In this work, we introduce GraphGhost, a unified framework that represents neuron activations and their signal propagation as graphs, explaining how LLMs capture structural semantics from sequential inputs and generate outputs through structurally consistent mechanisms. This graph-based perspective enables us to employ graph algorithms such as PageRank to characterize the properties of LLMs, revealing both shared and model-specific reasoning behaviors across diverse datasets. We further identify the activated neurons within GraphGhost and evaluate them through structural interventions, showing that edits to key neuron nodes can trigger reasoning collapse, altering both logical flow and semantic understanding. Together, these contributions position GraphGhost as a powerful tool for analyzing, intervening in, and ultimately understanding the structural foundations of reasoning in LLMs.
>
---
#### [new 102] Alif: Advancing Urdu Large Language Models via Multilingual Synthetic Data Distillation
- **分类: cs.CL; cs.AI; cs.LG; I.2.7; I.2.6; I.2.11**

- **简介: 该论文旨在解决低资源语言（如乌尔都语）在大规模语言模型开发中的数据匮乏、多语言不一致和安全问题。通过构建高质量的多语言合成数据集（Urdu-Instruct），采用改进的自指令技术训练模型，提升了乌尔都语任务的理解与生成能力。论文属于自然语言处理任务中的低资源语言模型构建与优化。**

- **链接: [http://arxiv.org/pdf/2510.09051v1](http://arxiv.org/pdf/2510.09051v1)**

> **作者:** Muhammad Ali Shafique; Kanwal Mehreen; Muhammad Arham; Maaz Amjad; Sabur Butt; Hamza Farooq
>
> **备注:** Accepted to the EMNLP 2025 Workshop on Multilingual Representation Learning (MRL)
>
> **摘要:** Developing a high-performing large language models (LLMs) for low-resource languages such as Urdu, present several challenges. These challenges include the scarcity of high-quality datasets, multilingual inconsistencies, and safety concerns. Existing multilingual LLMs often address these issues by translating large volumes of available data. However, such translations often lack quality and cultural nuance while also incurring significant costs for data curation and training. To address these issues, we propose Alif-1.0-8B-Instruct, a multilingual Urdu-English model, that tackles these challenges with a unique approach. We train the model on a high-quality, multilingual synthetic dataset (Urdu-Instruct), developed using a modified self-instruct technique. By using unique prompts and seed values for each task along with a global task pool, this dataset incorporates Urdu-native chain-of-thought based reasoning, bilingual translation, cultural relevance, and ethical safety alignments. This technique significantly enhances the comprehension of Alif-1.0-8B-Instruct model for Urdu-specific tasks. As a result, Alif-1.0-8B-Instruct, built upon the pretrained Llama-3.1-8B, demonstrates superior performance compared to Llama-3.1-8B-Instruct for Urdu specific-tasks. It also outperformed leading multilingual LLMs, including Mistral-7B-Instruct-v0.3, Qwen-2.5-7B-Instruct, and Cohere-Aya-Expanse-8B, all within a training budget of under $100. Our results demonstrate that high-performance and low-resource language LLMs can be developed efficiently and culturally aligned using our modified self-instruct approach. All datasets, models, and code are publicly available at: https://github.com/traversaal-ai/alif-urdu-llm.
>
---
#### [new 103] Scaling Laws for Code: A More Data-Hungry Regime
- **分类: cs.CL**

- **简介: 该论文研究代码大模型的缩放规律，旨在解决自然语言缩放法则是否适用于代码的问题。通过117次实验，验证了代码模型更耗数据，需更高数据-参数比，并分析了混合训练的效果。**

- **链接: [http://arxiv.org/pdf/2510.08702v1](http://arxiv.org/pdf/2510.08702v1)**

> **作者:** Xianzhen Luo; Wenzhen Zheng; Qingfu Zhu; Rongyi Zhang; Houyi Li; Siming Huang; YuanTao Fan; Wanxiang Che
>
> **备注:** Under Review
>
> **摘要:** Code Large Language Models (LLMs) are revolutionizing software engineering. However, scaling laws that guide the efficient training are predominantly analyzed on Natural Language (NL). Given the fundamental differences like strict syntax between code and NL, it is unclear whether these laws are directly applicable to code. To address this gap, we conduct the first large-scale empirical study of scaling laws for code, comprising 117 experimental runs with model sizes from 0.2B to 3.8B and training tokens from 2B to 128B. We fit the Chinchilla law and the Farsser law. First, the results show that the more expressive Farseer law offers greater accuracy. Second, the analysis reveals that Code LLMs scale effectively with model size. Crucially, code represents a more data-hungry regime, requiring a substantially higher data-to-parameter ratio than NL. Finally, two additional sets of experiments on code-NL mixtures show that NL benefits resource-constrained scenarios, but becomes a detriment at higher compute budgets.
>
---
#### [new 104] Hierarchical Self-Supervised Representation Learning for Depression Detection from Speech
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音抑郁检测任务，旨在解决传统方法依赖单一语音特征、难以捕捉抑郁时序特征的问题。论文提出HAREN-CTC模型，融合多层自监督语音特征，引入跨模态注意力与CTC损失，提升抑郁检测效果，取得SOTA性能。**

- **链接: [http://arxiv.org/pdf/2510.08593v1](http://arxiv.org/pdf/2510.08593v1)**

> **作者:** Yuxin Li; Eng Siong Chng; Cuntai Guan
>
> **摘要:** Speech-based depression detection (SDD) is a promising, non-invasive alternative to traditional clinical assessments. However, it remains limited by the difficulty of extracting meaningful features and capturing sparse, heterogeneous depressive cues over time. Pretrained self-supervised learning (SSL) models such as WavLM provide rich, multi-layer speech representations, yet most existing SDD methods rely only on the final layer or search for a single best-performing one. These approaches often overfit to specific datasets and fail to leverage the full hierarchical structure needed to detect subtle and persistent depression signals. To address this challenge, we propose HAREN-CTC, a novel architecture that integrates multi-layer SSL features using cross-attention within a multitask learning framework, combined with Connectionist Temporal Classification loss to handle sparse temporal supervision. HAREN-CTC comprises two key modules: a Hierarchical Adaptive Clustering module that reorganizes SSL features into complementary embeddings, and a Cross-Modal Fusion module that models inter-layer dependencies through cross-attention. The CTC objective enables alignment-aware training, allowing the model to track irregular temporal patterns of depressive speech cues. We evaluate HAREN-CTC under both an upper-bound setting with standard data splits and a generalization setting using five-fold cross-validation. The model achieves state-of-the-art macro F1-scores of 0.81 on DAIC-WOZ and 0.82 on MODMA, outperforming prior methods across both evaluation scenarios.
>
---
#### [new 105] Augmenting Dialog with Think-Aloud Utterances for Modeling Individual Personality Traits by LLM
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于自然语言处理任务，旨在通过对话数据建模个体性格特征。为解决传统对话数据难以准确捕捉说话者个性的问题，论文引入“出声思考”语句（TAU）增强数据，训练个性LLM，并验证其在模仿人类性格五维度（特别是亲和力与神经质）上的有效性。**

- **链接: [http://arxiv.org/pdf/2510.09158v1](http://arxiv.org/pdf/2510.09158v1)**

> **作者:** Seiya Ishikura; Hiroaki Yamada; Tatsuya Hiraoka; Hiroaki Yamada; Takenobu Tokunaga
>
> **备注:** 8 pages, 1 figure
>
> **摘要:** This study proposes augmenting dialog data with think-aloud utterances (TAUs) for modeling individual personalities in text chat by LLM. TAU is a verbalization of a speaker's thought before articulating the utterance. We expect "persona LLMs" trained with TAU-augmented data can mimic the speaker's personality trait better. We tested whether the trained persona LLMs obtain the human personality with respect to Big Five, a framework characterizing human personality traits from five aspects. The results showed that LLMs trained with TAU-augmented data more closely align to the speakers' Agreeableness and Neuroticism of Big Five than those trained with original dialog data. We also found that the quality of TAU-augmentation impacts persona LLM's performance.
>
---
#### [new 106] FLRC: Fine-grained Low-Rank Compressor for Efficient LLM Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型（LLM）因参数量大而难以在资源受限设备上部署的问题。作者提出了一种细粒度低秩压缩方法FLRC，通过为每层分配最优秩并引入渐进式低秩解码，有效提升了压缩后模型的生成质量。**

- **链接: [http://arxiv.org/pdf/2510.09332v1](http://arxiv.org/pdf/2510.09332v1)**

> **作者:** Yu-Chen Lu; Chong-Yan Chen; Chi-Chih Chang; Yu-Fang Hu; Kai-Chiang Wu
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Although large language models (LLM) have achieved remarkable performance, their enormous parameter counts hinder deployment on resource-constrained hardware. Low-rank compression can reduce both memory usage and computational demand, but applying a uniform compression ratio across all layers often leads to significant performance degradation, and previous methods perform poorly during decoding. To address these issues, we propose the Fine-grained Low-Rank Compressor (FLRC), which efficiently determines an optimal rank allocation for each layer, and incorporates progressive low-rank decoding to maintain text generation quality. Comprehensive experiments on diverse benchmarks demonstrate the superiority of FLRC, achieving up to a 17% improvement in ROUGE-L on summarization tasks compared to state-of-the-art low-rank compression methods, establishing a more robust and efficient framework to improve LLM inference.
>
---
#### [new 107] A Unified Biomedical Named Entity Recognition Framework with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于生物医学命名实体识别（BioNER）任务，旨在解决嵌套实体、边界模糊和跨语言泛化问题。论文提出一种基于大语言模型的统一框架，通过文本生成方式处理实体识别，设计符号标注策略，并引入对比学习实体选择机制，提升多语言和多任务泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.08902v1](http://arxiv.org/pdf/2510.08902v1)**

> **作者:** Tengxiao Lv; Ling Luo; Juntao Li; Yanhua Wang; Yuchen Pan; Chao Liu; Yanan Wang; Yan Jiang; Huiyi Lv; Yuanyuan Sun; Jian Wang; Hongfei Lin
>
> **备注:** Accepted as a short paper at BIBM2025
>
> **摘要:** Accurate recognition of biomedical named entities is critical for medical information extraction and knowledge discovery. However, existing methods often struggle with nested entities, entity boundary ambiguity, and cross-lingual generalization. In this paper, we propose a unified Biomedical Named Entity Recognition (BioNER) framework based on Large Language Models (LLMs). We first reformulate BioNER as a text generation task and design a symbolic tagging strategy to jointly handle both flat and nested entities with explicit boundary annotation. To enhance multilingual and multi-task generalization, we perform bilingual joint fine-tuning across multiple Chinese and English datasets. Additionally, we introduce a contrastive learning-based entity selector that filters incorrect or spurious predictions by leveraging boundary-sensitive positive and negative samples. Experimental results on four benchmark datasets and two unseen corpora show that our method achieves state-of-the-art performance and robust zero-shot generalization across languages. The source codes are freely available at https://github.com/dreamer-tx/LLMNER.
>
---
#### [new 108] Understanding the Effects of Domain Finetuning on LLMs
- **分类: cs.CL**

- **简介: 该论文研究领域为自然语言处理，旨在理解领域微调如何改变大语言模型的参数空间。论文提出“调优向量”框架，分析微调过程中模型参数的方向性变化，发现其主要影响MLP层并增强注意力头中的已有方向，有助于提升指令遵循与生成质量，并实现跨领域泛化。**

- **链接: [http://arxiv.org/pdf/2510.09359v1](http://arxiv.org/pdf/2510.09359v1)**

> **作者:** Eshaan Tanwar; Deepak Nathani; William Yang Wang; Tanmoy Chakraborty
>
> **摘要:** Large Language Models (LLMs) fine-tuned for specific domains exhibit strong performance; however, the underlying mechanisms by which this fine-tuning reshapes their parametric space are not well understood. Prior works primarily focus on auto-regressive or general-purpose instruct models, leaving domain-specialised LLMs under-explored. We present the first systematic study of domain-specific fine-tuning in large medical language models. Our analysis reveals that fine-tuning modifies only a small subset of the representational subspace, essentially preserving the pre-trained model's representation. To interpret these changes in subspaces, we propose tuning vectors, a novel framework inspired by task vectors, which explicitly capture the directional parameter shifts induced by fine-tuning. We demonstrate that these vectors are critical for enhancing both instruction-following and generation quality. Furthermore, combining tuning vectors across different domains yields improved generalisation. Upon closer inspection of directional alignment, we find these vectors primarily write new directional information into the MLP layers of the model, while amplifying existing directions in attention heads. Our findings offer new insights into LLM adaptation and provide a general, interpretable framework for analysing specialisation in large language models.
>
---
#### [new 109] DARO: Difficulty-Aware Reweighting Policy Optimization
- **分类: cs.CL**

- **简介: 论文提出DARO，一种动态调整不同难度样本损失权重的策略优化方法，用于解决大语言模型在数学推理任务中因静态加权导致的学习效率低下和性能受限问题，提升了模型在多个数学基准上的表现与收敛速度。**

- **链接: [http://arxiv.org/pdf/2510.09001v1](http://arxiv.org/pdf/2510.09001v1)**

> **作者:** Jingyu Zhou; Lu Ma; Hao Liang; Chengyu Shen; Bin Cui; Wentao Zhang
>
> **摘要:** Recent advances in large language models (LLMs) have shown that reasoning ability can be significantly enhanced through Reinforcement Learning with Verifiable Rewards (RLVR). Group Relative Policy Optimization (GRPO) has emerged as the de facto approach for RLVR, inspiring numerous variants. However, our mathematical analysis reveals that these methods are fundamentally weighted variations of GRPO. We provide a unified view, demonstrating that their reliance on static or overly simplistic weighting schemes tied to sample difficulty prevents adaptation to a model's evolving capabilities. This creates a significant loss scale issue, where training disproportionately focuses on certain difficulty levels at the expense of others, hindering overall performance. To address these limitations, we introduce \textbf{Difficulty-Aware Reweighting Policy Optimization (DARO)}, a method that dynamically adjusts the loss contribution of each difficulty group based on the model's learning state. Extensive experiments on Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, and Llama3.1-8B show that DARO outperforms four leading baselines across six math benchmarks, achieving significantly faster convergence and superior final performance.
>
---
#### [new 110] Semantic-Condition Tuning: Fusing Graph Context with Large Language Models for Knowledge Graph Completion
- **分类: cs.AI; cs.CL; I.2.7**

- **简介: 该论文属于知识图谱补全任务，旨在解决现有方法融合知识图与大语言模型时忽视关系语义、推理负担重的问题。论文提出语义条件调优（SCT），通过图神经网络提取语义条件，并自适应调制文本嵌入，实现知识与文本的深度融合，提升推理准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.08966v1](http://arxiv.org/pdf/2510.08966v1)**

> **作者:** Ruitong Liu; Yan Wen; Te Sun; Yunjia Wu; Pingyang Huang; Zihang Yu; Siyuan Li
>
> **备注:** 11 pages, 3 figures, conference
>
> **摘要:** Fusing Knowledge Graphs with Large Language Models is crucial for knowledge-intensive tasks like knowledge graph completion. The prevailing paradigm, prefix-tuning, simply concatenates knowledge embeddings with text inputs. However, this shallow fusion overlooks the rich relational semantics within KGs and imposes a significant implicit reasoning burden on the LLM to correlate the prefix with the text. To address these, we propose Semantic-condition Tuning (SCT), a new knowledge injection paradigm comprising two key modules. First, a Semantic Graph Module employs a Graph Neural Network to extract a context-aware semantic condition from the local graph neighborhood, guided by knowledge-enhanced relations. Subsequently, this condition is passed to a Condition-Adaptive Fusion Module, which, in turn, adaptively modulates the textual embedding via two parameterized projectors, enabling a deep, feature-wise, and knowledge-aware interaction. The resulting pre-fused embedding is then fed into the LLM for fine-tuning. Extensive experiments on knowledge graph benchmarks demonstrate that SCT significantly outperforms prefix-tuning and other strong baselines. Our analysis confirms that by modulating the input representation with semantic graph context before LLM inference, SCT provides a more direct and potent signal, enabling more accurate and robust knowledge reasoning.
>
---
#### [new 111] Limitations of Normalization in Attention Mechanism
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在分析注意力机制中归一化方法的局限性。作者通过理论分析与实验验证，揭示了softmax归一化在多选词情况下选择能力下降和训练敏感性问题，从而提出对更优归一化策略的需求。**

- **链接: [http://arxiv.org/pdf/2508.17821v1](http://arxiv.org/pdf/2508.17821v1)**

> **作者:** Timur Mudarisov; Mikhail Burtsev; Tatiana Petrova; Radu State
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** This paper investigates the limitations of the normalization in attention mechanisms. We begin with a theoretical framework that enables the identification of the model's selective ability and the geometric separation involved in token selection. Our analysis includes explicit bounds on distances and separation criteria for token vectors under softmax scaling. Through experiments with pre-trained GPT-2 model, we empirically validate our theoretical results and analyze key behaviors of the attention mechanism. Notably, we demonstrate that as the number of selected tokens increases, the model's ability to distinguish informative tokens declines, often converging toward a uniform selection pattern. We also show that gradient sensitivity under softmax normalization presents challenges during training, especially at low temperature settings. These findings advance current understanding of softmax-based attention mechanism and motivate the need for more robust normalization and selection strategies in future attention architectures.
>
---
#### [new 112] Exploring Cross-Client Memorization of Training Data in Large Language Models for Federated Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究联邦学习中大语言模型对训练数据的记忆问题，提出框架量化客户端内和跨客户端的记忆程度，分析记忆影响因素，揭示联邦学习模型更易记忆本地数据，训练和推理因素影响记忆表现。**

- **链接: [http://arxiv.org/pdf/2510.08750v1](http://arxiv.org/pdf/2510.08750v1)**

> **作者:** Tinnakit Udsa; Can Udomcharoenchaikit; Patomporn Payoungkhamdee; Sarana Nutanong; Norrathep Rattanavipanon
>
> **摘要:** Federated learning (FL) enables collaborative training without raw data sharing, but still risks training data memorization. Existing FL memorization detection techniques focus on one sample at a time, underestimating more subtle risks of cross-sample memorization. In contrast, recent work on centralized learning (CL) has introduced fine-grained methods to assess memorization across all samples in training data, but these assume centralized access to data and cannot be applied directly to FL. We bridge this gap by proposing a framework that quantifies both intra- and inter-client memorization in FL using fine-grained cross-sample memorization measurement across all clients. Based on this framework, we conduct two studies: (1) measuring subtle memorization across clients and (2) examining key factors that influence memorization, including decoding strategies, prefix length, and FL algorithms. Our findings reveal that FL models do memorize client data, particularly intra-client data, more than inter-client data, with memorization influenced by training and inferencing factors.
>
---
#### [new 113] LiveOIBench: Can Large Language Models Outperform Human Contestants in Informatics Olympiads?
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于编程能力评估任务，旨在解决现有编码基准的局限性。作者构建了LiveOIBench，一个包含403个奥赛难题及大量测试用例的基准，用于评估大模型与顶尖人类选手的表现差异，并推动模型提升结构化分析能力。**

- **链接: [http://arxiv.org/pdf/2510.09595v1](http://arxiv.org/pdf/2510.09595v1)**

> **作者:** Kaijian Zou; Aaron Xiong; Yunxiang Zhang; Frederick Zhang; Yueqi Ren; Jirong Yang; Ayoung Lee; Shitanshu Bhushan; Lu Wang
>
> **摘要:** Competitive programming problems increasingly serve as valuable benchmarks to evaluate the coding capabilities of large language models (LLMs) due to their complexity and ease of verification. Yet, current coding benchmarks face limitations such as lack of exceptionally challenging problems, insufficient test case coverage, reliance on online platform APIs that limit accessibility. To address these issues, we introduce LiveOIBench, a comprehensive benchmark featuring 403 expert-curated Olympiad-level competitive programming problems, each with an average of 60 expert-designed test cases. The problems are sourced directly from 72 official Informatics Olympiads in different regions conducted between 2023 and 2025. LiveOIBench distinguishes itself through four key features: (1) meticulously curated high-quality tasks with detailed subtask rubrics and extensive private test cases; (2) direct integration of elite contestant performance data to enable informative comparison against top-performing humans; (3) planned continuous, contamination-free updates from newly released Olympiad problems; and (4) a self-contained evaluation system facilitating offline and easy-to-reproduce assessments. Benchmarking 32 popular general-purpose and reasoning LLMs, we find that GPT-5 achieves a notable 81.76th percentile, a strong result that nonetheless falls short of top human contestant performance, who usually place above 90th. In contrast, among open-weight reasoning models, GPT-OSS-120B achieves only a 60th percentile, underscoring significant capability disparities from frontier closed models. Detailed analyses indicate that robust reasoning models prioritize precise problem analysis over excessive exploration, suggesting future models should emphasize structured analysis and minimize unnecessary exploration. All data, code, and leaderboard results will be made publicly available on our website.
>
---
#### [new 114] ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于文本到音频生成任务，旨在解决精细控制信号下音频生成效果差的问题。作者提出ControlAudio方法，通过渐进式扩散建模，融合文本、时序和音素特征，提升了生成音频的时间准确性和语音清晰度，达到最优性能。**

- **链接: [http://arxiv.org/pdf/2510.08878v1](http://arxiv.org/pdf/2510.08878v1)**

> **作者:** Yuxuan Jiang; Zehua Chen; Zeqian Ju; Yusheng Dai; Weibei Dou; Jun Zhu
>
> **备注:** 18 pages, 8 tables, 5 figures
>
> **摘要:** Text-to-audio (TTA) generation with fine-grained control signals, e.g., precise timing control or intelligible speech content, has been explored in recent works. However, constrained by data scarcity, their generation performance at scale is still compromised. In this study, we recast controllable TTA generation as a multi-task learning problem and introduce a progressive diffusion modeling approach, ControlAudio. Our method adeptly fits distributions conditioned on more fine-grained information, including text, timing, and phoneme features, through a step-by-step strategy. First, we propose a data construction method spanning both annotation and simulation, augmenting condition information in the sequence of text, timing, and phoneme. Second, at the model training stage, we pretrain a diffusion transformer (DiT) on large-scale text-audio pairs, achieving scalable TTA generation, and then incrementally integrate the timing and phoneme features with unified semantic representations, expanding controllability. Finally, at the inference stage, we propose progressively guided generation, which sequentially emphasizes more fine-grained information, aligning inherently with the coarse-to-fine sampling nature of DiT. Extensive experiments show that ControlAudio achieves state-of-the-art performance in terms of temporal accuracy and speech clarity, significantly outperforming existing methods on both objective and subjective evaluations. Demo samples are available at: https://control-audio.github.io/Control-Audio.
>
---
#### [new 115] A Design-based Solution for Causal Inference with Text: Can a Language Model Be Too Large?
- **分类: stat.ME; cs.CL; cs.LG; stat.AP**

- **简介: 该论文属于因果推断任务，旨在解决文本中潜在混杂因素导致的因果效应估计偏差问题。作者提出了一种新的实验设计方法，避免了传统大语言模型因编码处理变量而引发的重叠偏差问题。通过政治传播中表达谦逊的实验验证，论文展示了所提方法在估计文本处理效应上的有效性，并揭示了谦逊表达对政治陈述说服力的因果影响。**

- **链接: [http://arxiv.org/pdf/2510.08758v1](http://arxiv.org/pdf/2510.08758v1)**

> **作者:** Graham Tierney; Srikar Katta; Christopher Bail; Sunshine Hillygus; Alexander Volfovsky
>
> **摘要:** Many social science questions ask how linguistic properties causally affect an audience's attitudes and behaviors. Because text properties are often interlinked (e.g., angry reviews use profane language), we must control for possible latent confounding to isolate causal effects. Recent literature proposes adapting large language models (LLMs) to learn latent representations of text that successfully predict both treatment and the outcome. However, because the treatment is a component of the text, these deep learning methods risk learning representations that actually encode the treatment itself, inducing overlap bias. Rather than depending on post-hoc adjustments, we introduce a new experimental design that handles latent confounding, avoids the overlap issue, and unbiasedly estimates treatment effects. We apply this design in an experiment evaluating the persuasiveness of expressing humility in political communication. Methodologically, we demonstrate that LLM-based methods perform worse than even simple bag-of-words models using our real text and outcomes from our experiment. Substantively, we isolate the causal effect of expressing humility on the perceived persuasiveness of political statements, offering new insights on communication effects for social media platforms, policy makers, and social scientists.
>
---
#### [new 116] BigCodeArena: Unveiling More Reliable Human Preferences in Code Generation via Execution
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于代码生成评估任务，旨在解决人工评估代码质量困难的问题。作者构建了BigCodeArena平台，结合代码执行环境，收集人类偏好数据，并提出BigCodeReward和AutoCodeArena两个基准，评估LLM代码生成能力。**

- **链接: [http://arxiv.org/pdf/2510.08697v1](http://arxiv.org/pdf/2510.08697v1)**

> **作者:** Terry Yue Zhuo; Xiaolong Jin; Hange Liu; Juyong Jiang; Tianyang Liu; Chen Gong; Bhupesh Bishnoi; Vaisakhi Mishra; Marek Suppa; Noah Ziems; Saiteja Utpala; Ming Xu; Guangyu Song; Kaixin Li; Yuhan Cao; Bo Liu; Zheng Liu; Sabina Abdurakhmanova; Wenhao Yu; Mengzhao Jia; Jihan Yao; Kenneth Hamilton; Kumar Shridhar; Minh Chien Vu; Dingmin Wang; Jiawei Liu; Zijian Wang; Qian Liu; Binyuan Hui; Meg Risdal; Ahsen Khaliq; Atin Sood; Zhenchang Xing; Wasi Uddin Ahmad; John Grundy; David Lo; Banghua Zhu; Xiaoning Du; Torsten Scholak; Leandro von Werra
>
> **备注:** Built with love by the BigCode community :)
>
> **摘要:** Crowdsourced model evaluation platforms, such as Chatbot Arena, enable real-time evaluation from human perspectives to assess the quality of model responses. In the coding domain, manually examining the quality of LLM-generated content is extremely challenging, as it requires understanding long chunks of raw code and deliberately simulating code execution. To this end, we introduce BigCodeArena, an open human evaluation platform for code generation backed by a comprehensive and on-the-fly execution environment. Built on top of Chatbot Arena, BigCodeArena enables the execution of LLM-generated code and allows humans to interact with the execution process and outcomes. We collected over 14,000 raw code-centric conversation sessions across 10 widely used LLMs, spanning 10 languages and 8 types of execution environments. Among these conversations, we identified more than 4,700 multi-turn samples with pairwise human preferences. Further analysis uncovers underexplored preferences of LLMs in fine-grained domains characterized by tasks, languages, and frameworks. To systematically examine code understanding and generation capabilities of frontier LLMs, we curated two benchmarks based on the collected data, namely BigCodeReward and AutoCodeArena. For BigCodeReward, we post-processed the 4,700 conversations and evaluated the consistency between reward models and human preferences. The evaluation shows that most LLMs have superior performance in judging coding preferences when the execution results are available. Inspired by these findings, we propose AutoCodeArena, an automatic Elo rating benchmark designed to assess the coding quality of LLMs without human involvement. We find that proprietary LLMs like GPT-5, Claude-Sonnet-4, and Claude-Opus-4 still lead in code generation performance among recent emerging models.
>
---
#### [new 117] Diagnosing Shoulder Disorders Using Multimodal Large Language Models and Consumer-Grade Cameras
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于医学辅助诊断任务，旨在解决肩关节疾病早期诊断成本高、资源不足的问题。作者利用消费级摄像头视频与多模态大语言模型（MLLM），提出HMVDx框架，通过分步诊断提升准确率，并引入“可用性指数”评估模型在医疗决策链中的效果。实验显示诊断准确率提升79.6%。**

- **链接: [http://arxiv.org/pdf/2510.09230v1](http://arxiv.org/pdf/2510.09230v1)**

> **作者:** Jindong Hong; Wencheng Zhang; Shiqin Qiao; Jianhai Chen; Jianing Qiu; Chuanyang Zheng; Qian Xu; Yun Ji; Qianyue Wen; Weiwei Sun; Hao Li; Huizhen Li; Huichao Wang; Kai Wu; Meng Li; Yijun He; Lingjie Luo; Jiankai Sun
>
> **摘要:** Shoulder disorders, such as frozen shoulder (a.k.a., adhesive capsulitis), are common conditions affecting the health of people worldwide, and have a high incidence rate among the elderly and workers engaged in repetitive shoulder tasks. In regions with scarce medical resources, achieving early and accurate diagnosis poses significant challenges, and there is an urgent need for low-cost and easily scalable auxiliary diagnostic solutions. This research introduces videos captured by consumer-grade devices as the basis for diagnosis, reducing the cost for users. We focus on the innovative application of Multimodal Large Language Models (MLLMs) in the preliminary diagnosis of shoulder disorders and propose a Hybrid Motion Video Diagnosis framework (HMVDx). This framework divides the two tasks of action understanding and disease diagnosis, which are respectively completed by two MLLMs. In addition to traditional evaluation indicators, this work proposes a novel metric called Usability Index by the logical process of medical decision-making (action recognition, movement diagnosis, and final diagnosis). This index evaluates the effectiveness of MLLMs in the medical field from the perspective of the entire medical diagnostic pathway, revealing the potential value of low-cost MLLMs in medical applications for medical practitioners. In experimental comparisons, the accuracy of HMVDx in diagnosing shoulder joint injuries has increased by 79.6\% compared with direct video diagnosis, a significant technical contribution to future research on the application of MLLMs for video understanding in the medical field.
>
---
#### [new 118] Target speaker anonymization in multi-speaker recordings
- **分类: eess.AS; cs.CL; cs.CR**

- **简介: 该论文属于语音隐私保护任务，旨在解决多人对话中仅需匿名化目标说话人的问题。现有方法多针对单人语音，难以适用于多人场景。论文提出了针对对话语音中特定说话人进行匿名化的策略，并改进了评估方法，以更准确衡量隐私保护效果与语音可用性。**

- **链接: [http://arxiv.org/pdf/2510.09307v1](http://arxiv.org/pdf/2510.09307v1)**

> **作者:** Natalia Tomashenko; Junichi Yamagishi; Xin Wang; Yun Liu; Emmanuel Vincent
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Most of the existing speaker anonymization research has focused on single-speaker audio, leading to the development of techniques and evaluation metrics optimized for such condition. This study addresses the significant challenge of speaker anonymization within multi-speaker conversational audio, specifically when only a single target speaker needs to be anonymized. This scenario is highly relevant in contexts like call centers, where customer privacy necessitates anonymizing only the customer's voice in interactions with operators. Conventional anonymization methods are often not suitable for this task. Moreover, current evaluation methodology does not allow us to accurately assess privacy protection and utility in this complex multi-speaker scenario. This work aims to bridge these gaps by exploring effective strategies for targeted speaker anonymization in conversational audio, highlighting potential problems in their development and proposing corresponding improved evaluation methodologies.
>
---
#### [new 119] McMining: Automated Discovery of Misconceptions in Student Code
- **分类: cs.SE; cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出“McMining”任务，旨在自动发现学生代码中的编程误解。为解决学生因误解导致代码错误和学习障碍的问题，论文构建了包含误解标注的代码数据集，并基于大语言模型设计了两种McMiner方法，验证了Gemini、Claude和GPT系列模型在该任务上的有效性。**

- **链接: [http://arxiv.org/pdf/2510.08827v1](http://arxiv.org/pdf/2510.08827v1)**

> **作者:** Erfan Al-Hossami; Razvan Bunescu
>
> **备注:** 16 pages, 8 figures
>
> **摘要:** When learning to code, students often develop misconceptions about various programming language concepts. These can not only lead to bugs or inefficient code, but also slow down the learning of related concepts. In this paper, we introduce McMining, the task of mining programming misconceptions from samples of code from a student. To enable the training and evaluation of McMining systems, we develop an extensible benchmark dataset of misconceptions together with a large set of code samples where these misconceptions are manifested. We then introduce two LLM-based McMiner approaches and through extensive evaluations show that models from the Gemini, Claude, and GPT families are effective at discovering misconceptions in student code.
>
---
#### [new 120] BaldWhisper: Faster Whisper with Head Shearing and Layer Merging
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于模型压缩任务，旨在解决低资源语言语音识别模型在数据稀缺情况下难以压缩的问题。通过提出一种新的压缩方法，包括嵌入压缩和层融合，实现在有限数据下减少模型大小和提升推理速度。**

- **链接: [http://arxiv.org/pdf/2510.08599v1](http://arxiv.org/pdf/2510.08599v1)**

> **作者:** Yaya Sy; Christophe Cerisara; Irina Illina
>
> **摘要:** Pruning large pre-trained transformers for low-resource languages is challenging, as it often requires massive retraining data to recover performance. For instance, Distill-Whisper prunes Whisper by 40% and retrains on 21,000 hours of speech, far beyond what is available for most languages. Can Whisper be made lighter and faster for edge devices in data-scarce settings? Focusing on Bambara with only 32h of speech-to-text data, we propose a new pruning recipe. Instead of vocabulary pruning, which is unsuitable due to frequent code-switching by Bambara speakers, we compress the embeddings with low-rank decomposition and feature distillation. Rather than removing layers, we merge them to limit performance loss. The final model preserves 90% of the original performance while being 48% smaller and 2.15x faster on a MacBook Air M1.
>
---
#### [new 121] Comparative Analysis of Large Language Models for the Machine-Assisted Resolution of User Intentions
- **分类: cs.SE; cs.AI; cs.CL; cs.HC**

- **简介: 论文任务是比较分析开源与闭源大语言模型在用户意图解析中的能力。它旨在解决云端闭源模型带来的隐私、自主性及扩展性问题，通过评估多个开源模型在生成用户意图工作流上的表现，探讨其作为本地操作系统组件的可行性。工作包括对比实验与性能分析。**

- **链接: [http://arxiv.org/pdf/2510.08576v1](http://arxiv.org/pdf/2510.08576v1)**

> **作者:** Justus Flerlage; Alexander Acker; Odej Kao
>
> **摘要:** Large Language Models (LLMs) have emerged as transformative tools for natural language understanding and user intent resolution, enabling tasks such as translation, summarization, and, increasingly, the orchestration of complex workflows. This development signifies a paradigm shift from conventional, GUI-driven user interfaces toward intuitive, language-first interaction paradigms. Rather than manually navigating applications, users can articulate their objectives in natural language, enabling LLMs to orchestrate actions across multiple applications in a dynamic and contextual manner. However, extant implementations frequently rely on cloud-based proprietary models, which introduce limitations in terms of privacy, autonomy, and scalability. For language-first interaction to become a truly robust and trusted interface paradigm, local deployment is not merely a convenience; it is an imperative. This limitation underscores the importance of evaluating the feasibility of locally deployable, open-source, and open-access LLMs as foundational components for future intent-based operating systems. In this study, we examine the capabilities of several open-source and open-access models in facilitating user intention resolution through machine assistance. A comparative analysis is conducted against OpenAI's proprietary GPT-4-based systems to assess performance in generating workflows for various user intentions. The present study offers empirical insights into the practical viability, performance trade-offs, and potential of open LLMs as autonomous, locally operable components in next-generation operating systems. The results of this study inform the broader discussion on the decentralization and democratization of AI infrastructure and point toward a future where user-device interaction becomes more seamless, adaptive, and privacy-conscious through locally embedded intelligence.
>
---
#### [new 122] Everyone prefers human writers, including AI
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文研究人类与AI对文学风格评价中的归因偏差问题。通过对比实验，发现无论人类还是AI均倾向于高估人类创作、低估AI生成内容，且AI偏差更大。论文属认知科学与AI交叉任务，旨在揭示并量化审美判断中的主客观偏见及其机制。**

- **链接: [http://arxiv.org/pdf/2510.08831v1](http://arxiv.org/pdf/2510.08831v1)**

> **作者:** Wouter Haverals; Meredith Martin
>
> **备注:** 46 pages, 18 figures (5 main text + 13 supplementary), 5 tables
>
> **摘要:** As AI writing tools become widespread, we need to understand how both humans and machines evaluate literary style, a domain where objective standards are elusive and judgments are inherently subjective. We conducted controlled experiments using Raymond Queneau's Exercises in Style (1947) to measure attribution bias across evaluators. Study 1 compared human participants (N=556) and AI models (N=13) evaluating literary passages from Queneau versus GPT-4-generated versions under three conditions: blind, accurately labeled, and counterfactually labeled. Study 2 tested bias generalization across a 14$\times$14 matrix of AI evaluators and creators. Both studies revealed systematic pro-human attribution bias. Humans showed +13.7 percentage point (pp) bias (Cohen's h = 0.28, 95% CI: 0.21-0.34), while AI models showed +34.3 percentage point bias (h = 0.70, 95% CI: 0.65-0.76), a 2.5-fold stronger effect (P$<$0.001). Study 2 confirmed this bias operates across AI architectures (+25.8pp, 95% CI: 24.1-27.6%), demonstrating that AI systems systematically devalue creative content when labeled as "AI-generated" regardless of which AI created it. We also find that attribution labels cause evaluators to invert assessment criteria, with identical features receiving opposing evaluations based solely on perceived authorship. This suggests AI models have absorbed human cultural biases against artificial creativity during training. Our study represents the first controlled comparison of attribution bias between human and artificial evaluators in aesthetic judgment, revealing that AI systems not only replicate but amplify this human tendency.
>
---
#### [new 123] ReviewerToo: Should AI Join The Program Committee? A Look At The Future of Peer Review
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于人工智能辅助学术评审任务，旨在解决传统同行评审中的主观性、不一致性和扩展性问题。论文提出了ReviewerToo框架，结合AI模型与人类评审，提升评审质量与效率，并探讨AI在不同评审维度的表现与局限。**

- **链接: [http://arxiv.org/pdf/2510.08867v1](http://arxiv.org/pdf/2510.08867v1)**

> **作者:** Gaurav Sahu; Hugo Larochelle; Laurent Charlin; Christopher Pal
>
> **摘要:** Peer review is the cornerstone of scientific publishing, yet it suffers from inconsistencies, reviewer subjectivity, and scalability challenges. We introduce ReviewerToo, a modular framework for studying and deploying AI-assisted peer review to complement human judgment with systematic and consistent assessments. ReviewerToo supports systematic experiments with specialized reviewer personas and structured evaluation criteria, and can be partially or fully integrated into real conference workflows. We validate ReviewerToo on a carefully curated dataset of 1,963 paper submissions from ICLR 2025, where our experiments with the gpt-oss-120b model achieves 81.8% accuracy for the task of categorizing a paper as accept/reject compared to 83.9% for the average human reviewer. Additionally, ReviewerToo-generated reviews are rated as higher quality than the human average by an LLM judge, though still trailing the strongest expert contributions. Our analysis highlights domains where AI reviewers excel (e.g., fact-checking, literature coverage) and where they struggle (e.g., assessing methodological novelty and theoretical contributions), underscoring the continued need for human expertise. Based on these findings, we propose guidelines for integrating AI into peer-review pipelines, showing how AI can enhance consistency, coverage, and fairness while leaving complex evaluative judgments to domain experts. Our work provides a foundation for systematic, hybrid peer-review systems that scale with the growth of scientific publishing.
>
---
#### [new 124] Energy-Driven Steering: Reducing False Refusals in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型安全对齐中的“过度拒绝”问题。作者提出Energy-Driven Steering（EDS）框架，通过推理时动态干预模型隐藏状态，降低对良性提示的误拒率，同时保持安全性，无需微调模型权重。**

- **链接: [http://arxiv.org/pdf/2510.08646v1](http://arxiv.org/pdf/2510.08646v1)**

> **作者:** Eric Hanchen Jiang; Weixuan Ou; Run Liu; Shengyuan Pang; Guancheng Wan; Ranjie Duan; Wei Dong; Kai-Wei Chang; XiaoFeng Wang; Ying Nian Wu; Xinfeng Li
>
> **摘要:** Safety alignment of large language models (LLMs) faces a key challenge: current alignment techniques often only focus on improving safety against harmful prompts, causing LLMs to become over-cautious and refuse to respond to benign prompts. Therefore, a key objective of safe alignment is to enhance safety while simultaneously reducing false refusals. In this paper, we introduce Energy-Driven Steering (EDS), a novel, fine-tuning free framework designed to resolve this challenge through dynamic, inference-time intervention. We trained a lightweight, external Energy-Based Model (EBM) to assign high energy to undesirable (false refusal or jailbreak) states and low energy to desirable (helpful response or safe reject) ones. During inference, EBM maps the LLM's internal activations to an "energy landscape". We use the gradient of the energy function to dynamically steer the LLM's hidden states to low energy regions, correcting the model to generate a desirable response in real-time without modifying its weights. This method decouples behavioral control from the model's core knowledge, offering a flexible solution with minimal computational overhead. Extensive experiments across a wide range of models show our method successfully achieves this objective: it substantially lowers false refusal rates. For example, raising compliance on the ORB-H benchmark from 57.3% to 82.6% while maintaining the baseline safety performance. Our work presents an effective paradigm for building LLMs that achieve both low false refusal rates and high safety.
>
---
#### [new 125] Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于多模态提示优化任务，旨在解决当前提示优化方法仅限于文本、未能充分发挥多模态大语言模型潜力的问题。作者提出了多模态提示优化器（MPO），统一优化文本与非文本提示，并通过贝叶斯策略选择最优提示组合，实验证明其在多种模态任务上优于文本专用方法。**

- **链接: [http://arxiv.org/pdf/2510.09201v1](http://arxiv.org/pdf/2510.09201v1)**

> **作者:** Yumin Choi; Dongki Kim; Jinheon Baek; Sung Ju Hwang
>
> **摘要:** Large Language Models (LLMs) have shown remarkable success, and their multimodal expansions (MLLMs) further unlock capabilities spanning images, videos, and other modalities beyond text. However, despite this shift, prompt optimization approaches, designed to reduce the burden of manual prompt crafting while maximizing performance, remain confined to text, ultimately limiting the full potential of MLLMs. Motivated by this gap, we introduce the new problem of multimodal prompt optimization, which expands the prior definition of prompt optimization to the multimodal space defined by the pairs of textual and non-textual prompts. To tackle this problem, we then propose the Multimodal Prompt Optimizer (MPO), a unified framework that not only performs the joint optimization of multimodal prompts through alignment-preserving updates but also guides the selection process of candidate prompts by leveraging earlier evaluations as priors in a Bayesian-based selection strategy. Through extensive experiments across diverse modalities that go beyond text, such as images, videos, and even molecules, we demonstrate that MPO outperforms leading text-only optimization methods, establishing multimodal prompt optimization as a crucial step to realizing the potential of MLLMs.
>
---
#### [new 126] CapGeo: A Caption-Assisted Approach to Geometric Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态几何推理任务，旨在解决当前多模态大模型在几何问题上的表现不佳问题。作者提出CapGeo框架，通过结合图像生成文本描述辅助模型理解几何图形，显著提升了模型表现。同时构建了CapGeo-Bench数据集用于评估几何描述质量。**

- **链接: [http://arxiv.org/pdf/2510.09302v1](http://arxiv.org/pdf/2510.09302v1)**

> **作者:** Yuying Li; Siyi Qian; Hao Liang; Leqi Zheng; Ruichuan An; Yongzhen Guo; Wentao Zhang
>
> **备注:** preprint, under review
>
> **摘要:** Geometric reasoning remains a core challenge for Multimodal Large Language Models (MLLMs). Even the most advanced closed-source systems, such as GPT-O3 and Gemini-2.5-Pro, still struggle to solve geometry problems reliably, despite exhibiting strong textual reasoning abilities on tasks like the International Mathematical Olympiad (IMO). This gap suggests that the bottleneck lies in understanding geometric diagrams rather than reasoning itself. Since geometric figures can often be faithfully described in concise textual form, converting visual content into captions offers a promising direction. Motivated by this insight, we introduce CapGeo, a caption-assisted reasoning framework that bridges visual and textual modalities. Experiments show substantial improvements when models are equipped with captions: Qwen2.5-VL-72B improves from 8.6% (vision-only) to 59.0%, while Claude-Opus-4 rises from 44.8% to 73.0%. To systematically evaluate and identify high-quality geometric captioning models, we further propose CapGeo-Bench, a dataset of 4,641 curated figure-caption pairs. Crucially, CapGeo-Bench incorporates a keypoint-based evaluation metric that correlates strongly with downstream CapGeo performance, enabling reliable assessment of geometric captioning ability. Together, our framework and benchmark highlight a new pathway toward advancing geometric reasoning in MLLMs.
>
---
#### [new 127] HINT: Helping Ineffective Rollouts Navigate Towards Effectiveness
- **分类: cs.LG; cs.CL**

- **简介: 论文提出HINT框架，解决大语言模型在强化学习中因任务复杂导致的训练低效问题。通过量化“亲和度”指标监控训练稳定性，采用启发式提示引导模型自主推理，提升数学推理任务中的学习效果与数据效率。属于自然语言处理与强化学习交叉任务。**

- **链接: [http://arxiv.org/pdf/2510.09388v1](http://arxiv.org/pdf/2510.09388v1)**

> **作者:** Xinyi Wang; Jinyi Han; Zishang Jiang; Tingyun Li; Jiaqing Liang; Sihang Jiang; Zhaoqian Dai; Shuguang Ma; Fei Yu; Yanghua Xiao
>
> **摘要:** Reinforcement Learning (RL) has become a key driver for enhancing the long chain-of-thought (CoT) reasoning capabilities of Large Language Models (LLMs). However, prevalent methods like GRPO often fail when task difficulty exceeds the model's capacity, leading to reward sparsity and inefficient training. While prior work attempts to mitigate this using off-policy data, such as mixing RL with Supervised Fine-Tuning (SFT) or using hints, they often misguide policy updates In this work, we identify a core issue underlying these failures, which we term low training affinity. This condition arises from a large distributional mismatch between external guidance and the model's policy. To diagnose this, we introduce Affinity, the first quantitative metric for monitoring exploration efficiency and training stability. To improve Affinity, we propose HINT: Helping Ineffective rollouts Navigate Towards effectiveness, an adaptive hinting framework. Instead of providing direct answers, HINT supplies heuristic hints that guide the model to discover solutions on its own, preserving its autonomous reasoning capabilities. Extensive experiments on mathematical reasoning tasks show that HINT consistently outperforms existing methods, achieving state-of-the-art results with models of various scales, while also demonstrating significantly more stable learning and greater data efficiency.Code is available on Github.
>
---
#### [new 128] Auto-scaling Continuous Memory for GUI Agent
- **分类: cs.AI; cs.CL; cs.CV; cs.CY; cs.LG**

- **简介: 论文研究为GUI智能体设计可扩展的记忆机制，解决现有方法压缩记忆丢失视觉细节、难扩展的问题。提出连续记忆编码，结合自动扩展的数据飞轮，显著提升长时任务与跨界面任务的成功率，仅微调少量参数，效果媲美先进闭源模型。**

- **链接: [http://arxiv.org/pdf/2510.09038v1](http://arxiv.org/pdf/2510.09038v1)**

> **作者:** Wenyi Wu; Kun Zhou; Ruoxin Yuan; Vivian Yu; Stephen Wang; Zhiting Hu; Biwei Huang
>
> **摘要:** We study how to endow GUI agents with scalable memory that help generalize across unfamiliar interfaces and long-horizon tasks. Prior GUI agents compress past trajectories into text tokens, which balloons context length and misses decisive visual cues (e.g., exact widget size and position). We propose a continuous memory that encodes each GUI trajectory into a fixed-length sequence of continuous embeddings using the VLM itself as an encoder; these embeddings are plugged directly into the backbone's input layer, sharply reducing context cost while preserving fine-grained visual information. As memory size and retrieval depth increase, performance improves monotonically, unlike text memories that degrade with long prompts. To grow memory at low cost, we introduce an auto-scaling data flywheel that (i) discovers new environments via search, (ii) synthesizes tasks with an open-source VLM, (iii) rolls out trajectories with the agent, and (iv) verifies success with the same VLM. Using this pipeline, we collect 100k+ trajectories for about \$4000 and fine-tune only the memory encoder (LoRA on a Q-Former, 1.2\% parameters) with 1,500 samples. On real-world GUI benchmarks, our memory-augmented agent consistently improves success rates under long horizons and distribution shifts. Notably, Qwen-2.5-VL-7B + continuous memory achieves performance comparable to state-of-the-art closed-source models (e.g., GPT-4o, Claude-4).
>
---
#### [new 129] Optimizing delivery for quick commerce factoring qualitative assessment of generated routes
- **分类: cs.AI; cs.CL**

- **简介: 论文任务是优化即时电商配送路径。针对印度电商市场最后一公里配送成本高的问题，提出结合大语言模型（LLMs）评估路径规划算法生成的路线，识别潜在问题。研究使用LLMs对400个案例进行分析，验证其评估准确性，表明LLM可有效提升配送效率与可持续性。**

- **链接: [http://arxiv.org/pdf/2510.08671v1](http://arxiv.org/pdf/2510.08671v1)**

> **作者:** Milon Bhattacharya; Milan Kumar
>
> **摘要:** Indias e-commerce market is projected to grow rapidly, with last-mile delivery accounting for nearly half of operational expenses. Although vehicle routing problem (VRP) based solvers are widely used for delivery planning, their effectiveness in real-world scenarios is limited due to unstructured addresses, incomplete maps, and computational constraints in distance estimation. This study proposes a framework that employs large language models (LLMs) to critique VRP-generated routes against policy-based criteria, allowing logistics operators to evaluate and prioritise more efficient delivery plans. As a illustration of our approach we generate, annotate and evaluated 400 cases using large language models. Our study found that open-source LLMs identified routing issues with 79% accuracy, while proprietary reasoning models achieved reach upto 86%. The results demonstrate that LLM-based evaluation of VRP-generated routes can be an effective and scalable layer of evaluation which goes beyond beyond conventional distance and time based metrics. This has implications for improving cost efficiency, delivery reliability, and sustainability in last-mile logistics, especially for developing countries like India.
>
---
#### [new 130] Diagnosing and Mitigating System Bias in Self-Rewarding RL
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决自奖励强化学习（RLIR）中的系统偏差问题。作者提出了一种新方法RLER，通过集成奖励模型和自适应插值选择来减轻偏差，提升训练稳定性和性能，有效缩小了RLIR与有标签样本训练（RLVR）之间的差距。**

- **链接: [http://arxiv.org/pdf/2510.08977v1](http://arxiv.org/pdf/2510.08977v1)**

> **作者:** Chuyi Tan; Peiwen Yuan; Xinglin Wang; Yiwei Li; Shaoxiong Feng; Yueqi Zhang; Jiayi Shi; Ji Zhang; Boyuan Pan; Yao Hu; Kan Li
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) scales the reasoning ability of large language models (LLMs) but remains bottlenecked by limited labeled samples for continued data scaling. Reinforcement learning with intrinsic rewards (RLIR), where the policy model assigns rewards to its own rollouts, enables sustainable scaling in unlabeled settings, yet its performance and stability lag behind RLVR. We trace this gap to a system bias: the model tends to overestimate its high-confidence rollouts, leading to biased and unstable reward estimation. This bias accumulates as training progresses, with deviations from the oracle drifting toward over-reward, causing unstable training. We characterize this bias using three metrics: $\rho_{\text{noise}}$, $\rho_{\text{selfbias}}$, and $\rho_{\text{symbias}}$. We find that $\rho_{\text{noise}}$ and $\rho_{\text{symbias}}$ impact convergence, while $\rho_{\text{selfbias}}$ amplifies both correct and incorrect updates, leading to instability. To mitigate this, we propose reinforcement learning with ensembled rewards (RLER), which aggregates diverse models and adapts reward interpolation and rollout selection. Extensive experiments show that RLER improves by +13.6% over RLIR and is only 3.6% below RLVR, achieving stable scaling on unlabeled samples, making it highly applicable.
>
---
#### [new 131] StreamingVLM: Real-Time Understanding for Infinite Video Streams
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉-语言模型（VLM）任务，旨在解决实时理解无限视频流的问题。现有方法面临计算成本高、内存占用大或连贯性差、延迟高等挑战。论文提出StreamingVLM，通过维护紧凑的KV缓存和设计简单的监督微调策略，实现稳定、实时的视频理解。在长视频基准测试中表现优异，同时提升通用视觉问答能力。**

- **链接: [http://arxiv.org/pdf/2510.09608v1](http://arxiv.org/pdf/2510.09608v1)**

> **作者:** Ruyi Xu; Guangxuan Xiao; Yukang Chen; Liuning He; Kelly Peng; Yao Lu; Song Han
>
> **备注:** The first two authors contributed equally to this work
>
> **摘要:** Vision-language models (VLMs) could power real-time assistants and autonomous agents, but they face a critical challenge: understanding near-infinite video streams without escalating latency and memory usage. Processing entire videos with full attention leads to quadratic computational costs and poor performance on long videos. Meanwhile, simple sliding window methods are also flawed, as they either break coherence or suffer from high latency due to redundant recomputation. In this paper, we introduce StreamingVLM, a model designed for real-time, stable understanding of infinite visual input. Our approach is a unified framework that aligns training with streaming inference. During inference, we maintain a compact KV cache by reusing states of attention sinks, a short window of recent vision tokens, and a long window of recent text tokens. This streaming ability is instilled via a simple supervised fine-tuning (SFT) strategy that applies full attention on short, overlapped video chunks, which effectively mimics the inference-time attention pattern without training on prohibitively long contexts. For evaluation, we build Inf-Streams-Eval, a new benchmark with videos averaging over two hours that requires dense, per-second alignment between frames and text. On Inf-Streams-Eval, StreamingVLM achieves a 66.18% win rate against GPT-4O mini and maintains stable, real-time performance at up to 8 FPS on a single NVIDIA H100. Notably, our SFT strategy also enhances general VQA abilities without any VQA-specific fine-tuning, improving performance on LongVideoBench by +4.30 and OVOBench Realtime by +5.96. Code is available at https://github.com/mit-han-lab/streaming-vlm.
>
---
#### [new 132] Large Language Model Prompt Datasets: An In-depth Analysis and Insights
- **分类: cs.LG; cs.CL**

- **简介: 该论文分析了大型语言模型提示数据集，旨在解决提示构建和优化问题。作者收集并系统分析了多类提示数据集，提出了一种基于句法嵌入的提示优化方法，通过向提示的中心表示引导重写，提升模型输出质量。论文属于自然语言处理任务，聚焦提示工程与模型交互优化。**

- **链接: [http://arxiv.org/pdf/2510.09316v1](http://arxiv.org/pdf/2510.09316v1)**

> **作者:** Yuanming Zhang; Yan Lin; Arijit Khan; Huaiyu Wan
>
> **摘要:** A prompt is a natural language instruction that defines a specific task for a large language model (LLM) and serves as the primary interface for human-LLM interaction. With the growing deployment of LLMs, diverse prompt datasets are emerging from platforms such as GitHub and social media. These datasets span a wide array of applications and content types, facilitating both broader LLM utilization and improved prompt engineering. In this work, we--for the first time--have compiled an extensive list of prompt datasets sourced from various channels, representing a spectrum of downstream tasks, languages, engineering techniques, attributes, and modalities. We select key representative datasets for systematic analysis, revealing commonalities and differences in prompt construction across categories, distinguishing them from other text corpora like literature and web. We further propose a prompt optimization approach that leverages syntactic embeddings of part-of-speech and dependency structures. By identifying a centroid representation of prompts and guiding LLMs to rewrite prompts toward this centroid, our method improves the meaningfulness of model outputs. We have made our datasets and code available.
>
---
#### [new 133] Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型解释任务，旨在解决稀疏自编码器（SAE）训练中的特征吸收问题。作者提出了一种自适应时间掩码（ATM）方法，通过动态调整特征选择，降低特征间的吸收现象，提升模型可解释性，同时保持良好的重构能力。**

- **链接: [http://arxiv.org/pdf/2510.08855v1](http://arxiv.org/pdf/2510.08855v1)**

> **作者:** T. Ed Li; Junyu Ren
>
> **备注:** First submitted on February 10th, 2025 to ICLR 2025 Workshop (XAI4Science: From Understanding Model Behavior to Discovering New Scientific Knowledge). The paper was accepted but the workshop does not generate proceedings. Now uploading to arXiv to make the paper publicly available
>
> **摘要:** Understanding the internal representations of large language models is crucial for ensuring their reliability and safety, with sparse autoencoders (SAEs) emerging as a promising interpretability approach. However, current SAE training methods face feature absorption, where features (or neurons) are absorbed into each other to minimize $L_1$ penalty, making it difficult to consistently identify and analyze model behaviors. We introduce Adaptive Temporal Masking (ATM), a novel training approach that dynamically adjusts feature selection by tracking activation magnitudes, frequencies, and reconstruction contributions to compute importance scores that evolve over time. ATM applies a probabilistic masking mechanism based on statistical thresholding of these importance scores, creating a more natural feature selection process. Through extensive experiments on the Gemma-2-2b model, we demonstrate that ATM achieves substantially lower absorption scores compared to existing methods like TopK and JumpReLU SAEs, while maintaining excellent reconstruction quality. These results establish ATM as a principled solution for learning stable, interpretable features in neural networks, providing a foundation for more reliable model analysis.
>
---
#### [new 134] Robust Heuristic Algorithm Design with LLMs
- **分类: cs.AI; cs.CL; cs.NI**

- **简介: 该论文属于算法设计任务，旨在解决启发式算法在不同输入下性能不稳定的问题。作者提出一种结合大语言模型（LLM）与诊断工具的方法，通过分析启发式算法表现差的原因，并在特定输入区域优化设计，提升了算法的鲁棒性和性能。**

- **链接: [http://arxiv.org/pdf/2510.08755v1](http://arxiv.org/pdf/2510.08755v1)**

> **作者:** Pantea Karimi; Dany Rouhana; Pooria Namyar; Siva Kesava Reddy Kakarla; Venkat Arun; Behnaz Arzani
>
> **摘要:** We posit that we can generate more robust and performant heuristics if we augment approaches using LLMs for heuristic design with tools that explain why heuristics underperform and suggestions about how to fix them. We find even simple ideas that (1) expose the LLM to instances where the heuristic underperforms; (2) explain why they occur; and (3) specialize design to regions in the input space, can produce more robust algorithms compared to existing techniques~ -- ~the heuristics we produce have a $\sim28\times$ better worst-case performance compared to FunSearch, improve average performance, and maintain the runtime.
>
---
#### [new 135] Estimating Brain Activity with High Spatial and Temporal Resolution using a Naturalistic MEG-fMRI Encoding Model
- **分类: q-bio.NC; cs.CL; cs.LG; cs.NE**

- **简介: 该论文旨在解决脑成像中同时获得高空间和时间分辨率的问题。通过结合MEG和fMRI数据，利用自然叙述故事实验构建基于Transformer的编码模型，实现对大脑皮层源活动的高精度估计。验证显示模型在跨模态和跨被试泛化方面表现优异，为实现高精度脑活动映射提供了新方法。**

- **链接: [http://arxiv.org/pdf/2510.09415v1](http://arxiv.org/pdf/2510.09415v1)**

> **作者:** Beige Jerry Jin; Leila Wehbe
>
> **摘要:** Current non-invasive neuroimaging techniques trade off between spatial resolution and temporal resolution. While magnetoencephalography (MEG) can capture rapid neural dynamics and functional magnetic resonance imaging (fMRI) can spatially localize brain activity, a unified picture that preserves both high resolutions remains an unsolved challenge with existing source localization or MEG-fMRI fusion methods, especially for single-trial naturalistic data. We collected whole-head MEG when subjects listened passively to more than seven hours of narrative stories, using the same stimuli in an open fMRI dataset (LeBel et al., 2023). We developed a transformer-based encoding model that combines the MEG and fMRI from these two naturalistic speech comprehension experiments to estimate latent cortical source responses with high spatiotemporal resolution. Our model is trained to predict MEG and fMRI from multiple subjects simultaneously, with a latent layer that represents our estimates of reconstructed cortical sources. Our model predicts MEG better than the common standard of single-modality encoding models, and it also yields source estimates with higher spatial and temporal fidelity than classic minimum-norm solutions in simulation experiments. We validated the estimated latent sources by showing its strong generalizability across unseen subjects and modalities. Estimated activity in our source space predict electrocorticography (ECoG) better than an ECoG-trained encoding model in an entirely new dataset. By integrating the power of large naturalistic experiments, MEG, fMRI, and encoding models, we propose a practical route towards millisecond-and-millimeter brain mapping.
>
---
#### [new 136] Dynamic Stress Detection: A Study of Temporal Progression Modelling of Stress in Speech
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音中的压力检测任务，旨在解决传统方法将压力视为静态标签的问题。作者提出动态标注策略，利用情感标签生成细粒度压力标注，并采用基于交叉注意力的序列模型（如单向LSTM和Transformer编码器）捕捉压力随时间的演变。实验表明，该方法在MuSE和StressID数据集上分别提升了5%和18%的准确率，并能泛化到真实场景数据。**

- **链接: [http://arxiv.org/pdf/2510.08586v1](http://arxiv.org/pdf/2510.08586v1)**

> **作者:** Vishakha Lall; Yisi Liu
>
> **备注:** Accepted at IEEE CogMI 2025
>
> **摘要:** Detecting psychological stress from speech is critical in high-pressure settings. While prior work has leveraged acoustic features for stress detection, most treat stress as a static label. In this work, we model stress as a temporally evolving phenomenon influenced by historical emotional state. We propose a dynamic labelling strategy that derives fine-grained stress annotations from emotional labels and introduce cross-attention-based sequential models, a Unidirectional LSTM and a Transformer Encoder, to capture temporal stress progression. Our approach achieves notable accuracy gains on MuSE (+5%) and StressID (+18%) over existing baselines, and generalises well to a custom real-world dataset. These results highlight the value of modelling stress as a dynamic construct in speech.
>
---
#### [new 137] Exploiting Web Search Tools of AI Agents for Data Exfiltration
- **分类: cs.CR; cs.CL; 68T50, 68T0; F.2.2; I.2.7; K.6.5**

- **简介: 该论文研究大型语言模型（LLM）在使用网络搜索工具时面临的数据泄露风险，属于安全任务。要解决的问题是间接提示注入攻击如何影响模型安全性。作者通过系统评估不同模型的脆弱性，分析攻击成功因素，并提出加强训练、建立攻击向量数据库和统一测试框架等应对措施。**

- **链接: [http://arxiv.org/pdf/2510.09093v1](http://arxiv.org/pdf/2510.09093v1)**

> **作者:** Dennis Rall; Bernhard Bauer; Mohit Mittal; Thomas Fraunholz
>
> **备注:** 9 pages, 6 figures, conference article
>
> **摘要:** Large language models (LLMs) are now routinely used to autonomously execute complex tasks, from natural language processing to dynamic workflows like web searches. The usage of tool-calling and Retrieval Augmented Generation (RAG) allows LLMs to process and retrieve sensitive corporate data, amplifying both their functionality and vulnerability to abuse. As LLMs increasingly interact with external data sources, indirect prompt injection emerges as a critical and evolving attack vector, enabling adversaries to exploit models through manipulated inputs. Through a systematic evaluation of indirect prompt injection attacks across diverse models, we analyze how susceptible current LLMs are to such attacks, which parameters, including model size and manufacturer, specific implementations, shape their vulnerability, and which attack methods remain most effective. Our results reveal that even well-known attack patterns continue to succeed, exposing persistent weaknesses in model defenses. To address these vulnerabilities, we emphasize the need for strengthened training procedures to enhance inherent resilience, a centralized database of known attack vectors to enable proactive defense, and a unified testing framework to ensure continuous security validation. These steps are essential to push developers toward integrating security into the core design of LLMs, as our findings show that current models still fail to mitigate long-standing threats.
>
---
#### [new 138] When to Reason: Semantic Router for vLLM
- **分类: cs.ET; cs.AI; cs.CL; cs.SY; eess.SY**

- **简介: 论文任务是提升大语言模型推理效率。针对简单任务无需复杂推理的问题，提出语义路由方法，按需启用推理模式。在MMLU-Pro上准确率提升10.2%，延迟降低47.1%，用Token减少48.5%。**

- **链接: [http://arxiv.org/pdf/2510.08731v1](http://arxiv.org/pdf/2510.08731v1)**

> **作者:** Chen Wang; Xunzhuo Liu; Yuhan Liu; Yue Zhu; Xiangxi Mo; Junchen Jiang; Huamin Chen
>
> **备注:** 5 pages, excluding references and appendix. To be appeared at Workshop on ML for Systems at NeurIPS 2025, December 6, 2025 https://mlforsystems.org/
>
> **摘要:** Large Language Models (LLMs) demonstrate substantial accuracy gains when augmented with reasoning modes such as chain-of-thought and inference-time scaling. However, reasoning also incurs significant costs in inference latency and token usage, with environmental and financial impacts, which are unnecessary for many simple prompts. We present a semantic router that classifies queries based on their reasoning requirements and selectively applies reasoning only when beneficial. Our approach achieves a 10.2 percentage point improvement in accuracy on the MMLU-Pro benchmark while reducing response latency by 47.1% and token consumption by 48.5% compared to direct inference with vLLM. These results demonstrate that semantic routing offers an effective mechanism for striking a balance between accuracy and efficiency in open-source LLM serving systems
>
---
#### [new 139] Struc-EMB: The Potential of Structure-Aware Encoding in Language Embeddings
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决现有语言模型忽略文本结构信息的问题。作者提出结构感知的文本嵌入方法Struc-EMB，通过在模型内部编码过程中融合结构信息（如超链接、引用等），提升文本表示效果。论文比较了两种融合方法，并引入技术应对结构噪声，验证了其在多种任务上的优越性。**

- **链接: [http://arxiv.org/pdf/2510.08774v1](http://arxiv.org/pdf/2510.08774v1)**

> **作者:** Shikun Liu; Haoyu Wang; Mufei Li; Pan Li
>
> **摘要:** Text embeddings from Large Language Models (LLMs) have become foundational for numerous applications. However, these models typically operate on raw text, overlooking the rich structural information, such as hyperlinks or citations, that provides crucial context in many real-world datasets. This paper introduces and systematically evaluates a new paradigm for generating structure-aware text embeddings by integrating these structural relations directly into the LLM's internal encoding process, rather than relying on traditional post-hoc aggregation. We investigate two primary in-process methods: sequential concatenation and parallel caching. Through extensive zero-shot experiments across retrieval, clustering, classification, and recommendation tasks, we demonstrate that our structure-aware approaches consistently outperform both text-only and post-hoc baselines. Our analysis reveals critical trade-offs: sequential concatenation excels with noisy, moderate-length contexts, while parallel caching scales more effectively to long, high-signal contexts but is more susceptible to distractors. To address the challenge of noisy structural data, we also introduce and validate two effective techniques: Context Distillation and Semantic Balancing. This work provides the first comprehensive analysis of in-process structure-aware encoding, offering a blueprint for building more powerful and contextually aware embedding models.
>
---
#### [new 140] On Epistemic Uncertainty of Visual Tokens for Object Hallucinations in Large Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究大型视觉语言模型（LVLM）中的物体幻觉问题，任务是分析视觉标记的不确定性对幻觉的影响，并提出缓解方法。论文发现早期视觉编码层中对微小对抗扰动表现出大表征偏差的视觉标记具有高认知不确定性，与幻觉发生正相关。作者提出一种仅修改视觉编码器的方法，通过对抗扰动代理识别不确定标记，并在自注意力过程中屏蔽它们，以减少幻觉。实验表明该方法有效，并可与其他方法协同使用。**

- **链接: [http://arxiv.org/pdf/2510.09008v1](http://arxiv.org/pdf/2510.09008v1)**

> **作者:** Hoigi Seo; Dong Un Kang; Hyunjin Cho; Joohoon Lee; Se Young Chun
>
> **摘要:** Large vision-language models (LVLMs), which integrate a vision encoder (VE) with a large language model, have achieved remarkable success across various tasks. However, there are still crucial challenges in LVLMs such as object hallucination, generating descriptions of objects that are not in the input image. Here, we argue that uncertain visual tokens within the VE is a key factor that contributes to object hallucination. Our statistical analysis found that there are positive correlations between visual tokens with high epistemic uncertainty and the occurrence of hallucinations. Furthermore, we show theoretically and empirically that visual tokens in early VE layers that exhibit large representation deviations under small adversarial perturbations indicate high epistemic uncertainty. Based on these findings, we propose a simple yet effective strategy to mitigate object hallucination by modifying the VE only. Our method comprises a proxy method with adversarial perturbations for identifying uncertain visual tokens efficiently and a method to mask these uncertain visual tokens during the self-attention process in the middle layers of the VE, suppressing their influence on visual encoding and thus alleviating hallucinations. Extensive experiments show that our method significantly reduces object hallucinations in LVLMs and can synergistically work with other prior arts.
>
---
#### [new 141] HES-SQL: Hybrid Reasoning for Efficient Text-to-SQL with Structural Skeleton Guidance
- **分类: cs.DB; cs.AI; cs.CL**

- **简介: 该论文属于文本到SQL生成任务，旨在提升模型生成SQL查询的准确性和执行效率。论文提出HES-SQL框架，结合监督微调与强化学习，引入结构骨架引导、延迟感知奖励和自蒸馏机制。实验表明其在BIRD和KaggleDBQA数据集上取得良好效果，兼顾语义准确与计算效率。**

- **链接: [http://arxiv.org/pdf/2510.08896v1](http://arxiv.org/pdf/2510.08896v1)**

> **作者:** Suming Qiu; Jing Li; Zhicheng Zhou; Junjie Huang; Linyuan Qiu; Zhijie Sun
>
> **摘要:** We present HES-SQL, a novel hybrid training framework that advances Text-to-SQL generation through the integration of thinking-mode-fused supervised fine-tuning (SFT) with Group Relative Policy Optimization (GRPO). Our approach introduces three key innovations: (1) a skeleton-completeness scoring mechanism that enhances preference alignment between generated queries and optimal SQL structures; (2) a query-latency-aware reward system that incentivizes the generation of computationally efficient SQL queries; (3) a self-distillation process for thinking-mode completion that prevents degradation of the model's reasoning capabilities. This framework enables hybrid thinking models to switch between reasoning and non-reasoning modes while improving SQL query accuracy and execution efficiency. Experimental evaluation, conducted on MySQL 8.0 and SQLite 3.42 under controlled single-user conditions, demonstrates that HES-SQL achieves competitive performance with execution accuracies of 79.14\% and 54.9\% on the BIRD and KaggleDBQA benchmarks, respectively. Query latency is measured as the end-to-end execution time of generated queries on the DBMS, averaged over multiple runs to mitigate variance. Efficiency gains range from 11\% to 20\% relative to supervised baselines. Our results establish a new paradigm for Text-to-SQL systems that effectively balances semantic accuracy with computational efficiency through execution-informed reinforcement learning (RL). The proposed methodology has significant implications for developing robust natural language interfaces to databases and can be extended to broader structured generation tasks requiring both correctness and efficiency optimization.
>
---
#### [new 142] Unsupervised lexicon learning from speech is limited by representations rather than clustering
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于零资源语音词分割任务，旨在无文本标注情况下从语音中提取词单元。论文探讨了系统性能瓶颈，分析了表征方法与聚类方法的影响，发现同一词类片段的表征差异是主要限制因素，而非聚类方法本身。**

- **链接: [http://arxiv.org/pdf/2510.09225v1](http://arxiv.org/pdf/2510.09225v1)**

> **作者:** Danel Adendorff; Simon Malan; Herman Kamper
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Zero-resource word segmentation and clustering systems aim to tokenise speech into word-like units without access to text labels. Despite progress, the induced lexicons are still far from perfect. In an idealised setting with gold word boundaries, we ask whether performance is limited by the representation of word segments, or by the clustering methods that group them into word-like types. We combine a range of self-supervised speech features (continuous/discrete, frame/word-level) with different clustering methods (K-means, hierarchical, graph-based) on English and Mandarin data. The best system uses graph clustering with dynamic time warping on continuous features. Faster alternatives use graph clustering with cosine distance on averaged continuous features or edit distance on discrete unit sequences. Through controlled experiments that isolate either the representations or the clustering method, we demonstrate that representation variability across segments of the same word type -- rather than clustering -- is the primary factor limiting performance.
>
---
#### [new 143] Unleashing Perception-Time Scaling to Multimodal Reasoning Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态推理任务，旨在解决当前大视觉-语言模型（LVLMs）在视觉感知中的估计精度低、推理时扩展策略效果有限的问题。作者提出了感知时扩展（PTS）方法，通过分解复杂感知问题、增加感知相关标记，显著提升模型在视觉估计和跨领域任务上的表现。**

- **链接: [http://arxiv.org/pdf/2510.08964v1](http://arxiv.org/pdf/2510.08964v1)**

> **作者:** Yifan Li; Zhenghao Chen; Ziheng Wu; Kun Zhou; Ruipu Luo; Can Zhang; Zhentao He; Yufei Zhan; Wayne Xin Zhao; Minghui Qiu
>
> **摘要:** Recent advances in inference-time scaling, particularly those leveraging reinforcement learning with verifiable rewards, have substantially enhanced the reasoning capabilities of Large Vision-Language Models (LVLMs). Inspired by this success, similar strategies have been applied to multimodal reasoning, yet their impact on visual perception remains unclear. To investigate this gap, we introduce DisTANCE, a perception-centric benchmark for visual estimation tasks. Evaluation results show that LVLMs exhibit limited estimation precision, and inference-time scaling offers only marginal gains. We attribute this to the fast perception paradigm of current LVLMs, where visual understanding is treated as a one-shot output without modeling the underlying perceptual process. To address this, we propose Perception-Time Scaling (PTS), a novel paradigm that encourages token-rich perception and decomposes complex perception problems into intermediate tractable sub-problems, thereby enabling perception to align with and benefit from inference-time scaling. Combined with reinforcement learning techniques, PTS significantly improves perception accuracy, raising high-precision performance on DisTANCE from 8.0% to 64.7%, and generalizes well to out-of-domain tasks. Surprisingly, even though PTS data are purely synthetic, combining them with math reasoning data yields consistent gains in both reasoning and real-world perception benchmarks. Further analysis reveals that PTS introduces more perception-related tokens and increases the model's attention to image tokens. Our code and data will be publicly released.
>
---
#### [new 144] Articulation-Informed ASR: Integrating Articulatory Features into ASR via Auxiliary Speech Inversion and Cross-Attention Fusion
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，旨在提升自动语音识别（ASR）性能。它通过引入发音特征作为辅助任务和伪输入，利用语音反演和交叉注意力机制，将发音信息融合到深度学习模型中。实验表明，该方法在LibriSpeech数据集上优于强基准模型，尤其适用于资源有限的情况。**

- **链接: [http://arxiv.org/pdf/2510.08585v1](http://arxiv.org/pdf/2510.08585v1)**

> **作者:** Ahmed Adel Attia; Jing Liu; Carol Espy Wilson
>
> **摘要:** Prior works have investigated the use of articulatory features as complementary representations for automatic speech recognition (ASR), but their use was largely confined to shallow acoustic models. In this work, we revisit articulatory information in the era of deep learning and propose a framework that leverages articulatory representations both as an auxiliary task and as a pseudo-input to the recognition model. Specifically, we employ speech inversion as an auxiliary prediction task, and the predicted articulatory features are injected into the model as a query stream in a cross-attention module with acoustic embeddings as keys and values. Experiments on LibriSpeech demonstrate that our approach yields consistent improvements over strong transformer-based baselines, particularly under low-resource conditions. These findings suggest that articulatory features, once sidelined in ASR research, can provide meaningful benefits when reintroduced with modern architectures.
>
---
## 更新

#### [replaced 001] AD-LLM: Benchmarking Large Language Models for Anomaly Detection
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.11142v4](http://arxiv.org/pdf/2412.11142v4)**

> **作者:** Tiankai Yang; Yi Nian; Shawn Li; Ruiyao Xu; Yuangang Li; Jiaqi Li; Zhuo Xiao; Xiyang Hu; Ryan Rossi; Kaize Ding; Xia Hu; Yue Zhao
>
> **备注:** Accepted to Findings of the Association for Computational Linguistics (ACL 2025), Vienna, Austria
>
> **摘要:** Anomaly detection (AD) is an important machine learning task with many real-world uses, including fraud detection, medical diagnosis, and industrial monitoring. Within natural language processing (NLP), AD helps detect issues like spam, misinformation, and unusual user activity. Although large language models (LLMs) have had a strong impact on tasks such as text generation and summarization, their potential in AD has not been studied enough. This paper introduces AD-LLM, the first benchmark that evaluates how LLMs can help with NLP anomaly detection. We examine three key tasks: (i) zero-shot detection, using LLMs' pre-trained knowledge to perform AD without tasks-specific training; (ii) data augmentation, generating synthetic data and category descriptions to improve AD models; and (iii) model selection, using LLMs to suggest unsupervised AD models. Through experiments with different datasets, we find that LLMs can work well in zero-shot AD, that carefully designed augmentation methods are useful, and that explaining model selection for specific datasets remains challenging. Based on these results, we outline six future research directions on LLMs for AD.
>
---
#### [replaced 002] Do Internal Layers of LLMs Reveal Patterns for Jailbreak Detection?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.06594v2](http://arxiv.org/pdf/2510.06594v2)**

> **作者:** Sri Durga Sai Sowmya Kadali; Evangelos E. Papalexakis
>
> **摘要:** Jailbreaking large language models (LLMs) has emerged as a pressing concern with the increasing prevalence and accessibility of conversational LLMs. Adversarial users often exploit these models through carefully engineered prompts to elicit restricted or sensitive outputs, a strategy widely referred to as jailbreaking. While numerous defense mechanisms have been proposed, attackers continuously develop novel prompting techniques, and no existing model can be considered fully resistant. In this study, we investigate the jailbreak phenomenon by examining the internal representations of LLMs, with a focus on how hidden layers respond to jailbreak versus benign prompts. Specifically, we analyze the open-source LLM GPT-J and the state-space model Mamba2, presenting preliminary findings that highlight distinct layer-wise behaviors. Our results suggest promising directions for further research on leveraging internal model dynamics for robust jailbreak detection and defense.
>
---
#### [replaced 003] Augmenting Compliance-Guaranteed Customer Service Chatbots: Context-Aware Knowledge Expansion with Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12444v3](http://arxiv.org/pdf/2410.12444v3)**

> **作者:** Mengze Hong; Chen Jason Zhang; Di Jiang; Yuanqin He
>
> **备注:** Accepted by EMNLP 2025 Industry Track
>
> **摘要:** Retrieval-based chatbots leverage human-verified Q\&A knowledge to deliver accurate, verifiable responses, making them ideal for customer-centric applications where compliance with regulatory and operational standards is critical. To effectively handle diverse customer inquiries, augmenting the knowledge base with "similar questions" that retain semantic meaning while incorporating varied expressions is a cost-effective strategy. In this paper, we introduce the Similar Question Generation (SQG) task for LLM training and inference, proposing context-aware approaches to enable comprehensive semantic exploration and enhanced alignment with source question-answer relationships. We formulate optimization techniques for constructing in-context prompts and selecting an optimal subset of similar questions to expand chatbot knowledge under budget constraints. Both quantitative and human evaluations validate the effectiveness of these methods, achieving a 92% user satisfaction rate in a deployed chatbot system, reflecting an 18% improvement over the unaugmented baseline. These findings highlight the practical benefits of SQG and emphasize the potential of LLMs, not as direct chatbot interfaces, but in supporting non-generative systems for hallucination-free, compliance-guaranteed applications.
>
---
#### [replaced 004] Populism Meets AI: Advancing Populism Research with LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.07458v2](http://arxiv.org/pdf/2510.07458v2)**

> **作者:** Eduardo Ryô Tamaki; Yujin J. Jung; Julia Chatterley; Grant Mitchell; Semir Dzebo; Cristóbal Sandoval; Levente Littvay; Kirk A. Hawkins
>
> **备注:** 27 pages, 3 figures. Preprint version under review
>
> **摘要:** Measuring the ideational content of populism remains a challenge. Traditional strategies based on textual analysis have been critical for building the field's foundations and providing a valid, objective indicator of populist framing. Yet these approaches are costly, time consuming, and difficult to scale across languages, contexts, and large corpora. Here we present the results from a rubric and anchor guided chain of thought (CoT) prompting approach that mirrors human coder training. By leveraging the Global Populism Database (GPD), a comprehensive dataset of global leaders' speeches annotated for degrees of populism, we replicate the process used to train human coders by prompting the LLM with an adapted version of the same documentation to guide the model's reasoning. We then test multiple proprietary and open weight models by replicating scores in the GPD. Our findings reveal that this domain specific prompting strategy enables the LLM to achieve classification accuracy on par with expert human coders, demonstrating its ability to navigate the nuanced, context sensitive aspects of populism.
>
---
#### [replaced 005] FinTagging: Benchmarking LLMs for Extracting and Structuring Financial Information
- **分类: cs.CL; cs.AI; cs.CE**

- **链接: [http://arxiv.org/pdf/2505.20650v2](http://arxiv.org/pdf/2505.20650v2)**

> **作者:** Yan Wang; Yang Ren; Lingfei Qian; Xueqing Peng; Keyi Wang; Yi Han; Dongji Feng; Fengran Mo; Shengyuan Lin; Qinchuan Zhang; Kaiwen He; Chenri Luo; Jianxing Chen; Junwei Wu; Jimin Huang; Guojun Xiong; Xiao-Yang Liu; Qianqian Xie; Jian-Yun Nie
>
> **摘要:** Accurately understanding numbers from financial reports is fundamental to how markets, regulators, algorithms, and normal people read the economy and the world, yet even with XBRL (eXtensible Business Reporting Language) designed to tag every figure with standardized accounting concepts, mapping thousands of facts to over 10,000 U.S. GAAP concepts remains costly, inconsistent, and error-prone. Existing benchmarks define tagging as flat, single-step, extreme classification over small subsets of US-GAAP concepts, overlooking both the taxonomy's hierarchical semantics and the structured nature of real tagging, where each fact must be represented as a contextualized multi-field output. These simplifications prevent fair evaluation of large language models (LLMs) under realistic reporting conditions. To address these gaps, we introduce FinTagging, the first comprehensive benchmark for structure-aware and full-scope XBRL tagging, designed to evaluate LLMs' ability to extract and align financial facts through numerical reasoning and taxonomy alignment across text and tables. We define two subtasks: FinNI for numeric identification, which extracts numerical entities and their types from XBRL reports, and FinCL for concept linking, which maps each extracted entity to the corresponding concept in the full US-GAAP taxonomy. Together, these subtasks produce a structured representation of each financial fact. We evaluate diverse LLMs under zero-shot settings and analyze their performance across both subtasks and overall tagging accuracy. Results show that LLMs generalize well in numeric identification but struggle with fine-grained concept linking, revealing current limitations in structure-aware reasoning for accurate financial disclosure. All code and datasets are available on GitHub and Hugging Face.
>
---
#### [replaced 006] Lizard: An Efficient Linearization Framework for Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.09025v3](http://arxiv.org/pdf/2507.09025v3)**

> **作者:** Chien Van Nguyen; Ruiyi Zhang; Hanieh Deilamsalehy; Puneet Mathur; Viet Dac Lai; Haoliang Wang; Jayakumar Subramanian; Ryan A. Rossi; Trung Bui; Nikos Vlassis; Franck Dernoncourt; Thien Huu Nguyen
>
> **备注:** 13 pages
>
> **摘要:** We propose Lizard, a linearization framework that transforms pretrained Transformer-based Large Language Models (LLMs) into subquadratic architectures. Transformers faces severe computational and memory bottlenecks with long sequences due to the quadratic complexity of softmax attention and the growing Key-Value (KV) cache that makes inference memory-bound by context length. Lizard addresses these limitations by introducing a subquadratic attention mechanism that closely approximates softmax attention while preserving model quality. Unlike prior linearization methods constrained by fixed, non-adaptive structures, Lizard augments the architecture with compact, learnable modules that enable adaptive memory control and robust length generalization. Moreover, we introduce a hardwareaware algorithm that solves numerical instability in gated attention to accelerate training. Extensive experiments show that Lizard achieves near-lossless recovery of its teacher model's performance, significantly outperforming previous methods by up to 9.4 - 24.5 points on the 5-shot MMLU benchmark and demonstrating superior associative recall.
>
---
#### [replaced 007] AMFT: Aligning LLM Reasoners by Meta-Learning the Optimal Imitation-Exploration Balance
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.06944v3](http://arxiv.org/pdf/2508.06944v3)**

> **作者:** Lixuan He; Jie Feng; Yong Li
>
> **备注:** The paper is currently under investigation regarding concerns of potential academic misconduct. While the investigation is ongoing, the authors have voluntarily requested to withdraw the manuscript
>
> **摘要:** Large Language Models (LLMs) are typically fine-tuned for reasoning tasks through a two-stage pipeline of Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL), a process fraught with catastrophic forgetting and suboptimal trade-offs between imitation and exploration. Recent single-stage methods attempt to unify SFT and RL using heuristics, but lack a principled mechanism for dynamically balancing the two paradigms. In this paper, we reframe this challenge through the theoretical lens of \textbf{implicit rewards}, viewing SFT and RL not as distinct methods but as complementary reward signals. We introduce \textbf{Adaptive Meta Fine-Tuning (AMFT)}, a novel single-stage algorithm that learns the optimal balance between SFT's implicit, path-level reward and RL's explicit, outcome-based reward. The core of AMFT is a \textbf{meta-gradient adaptive weight controller} that treats the SFT-RL balance as a learnable parameter, dynamically optimizing it to maximize long-term task performance. This forward-looking approach, regularized by policy entropy for stability, autonomously discovers an effective training curriculum. We conduct a comprehensive evaluation on challenging benchmarks spanning mathematical reasoning, abstract visual reasoning (General Points), and vision-language navigation (V-IRL). AMFT consistently establishes a new state-of-the-art and demonstrats superior generalization on out-of-distribution (OOD) tasks. Ablation studies and training dynamic analysis confirm that the meta-learning controller is crucial for AMFT's stability, sample efficiency, and performance, offering a more principled and effective paradigm for LLM alignment. Our codes are open-sourced via https://github.com/hlxtsyj/AMFT.
>
---
#### [replaced 008] ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.06094v2](http://arxiv.org/pdf/2508.06094v2)**

> **作者:** Morris Alper; Moran Yanuka; Raja Giryes; Gašper Beguš
>
> **备注:** Project page: https://conlangcrafter.github.io
>
> **摘要:** Constructed languages (conlangs) such as Esperanto and Quenya have played diverse roles in art, philosophy, and international communication. Meanwhile, foundation models have revolutionized creative generation in text, images, and beyond. In this work, we leverage modern LLMs as computational creativity aids for end-to-end conlang creation. We introduce ConlangCrafter, a multi-hop pipeline that decomposes language design into modular stages - phonology, morphology, syntax, lexicon generation, and translation. At each stage, our method leverages LLMs' metalinguistic reasoning capabilities, injecting randomness to encourage diversity and leveraging self-refinement feedback to encourage consistency in the emerging language description. We evaluate ConlangCrafter on metrics measuring consistency and typological diversity, demonstrating its ability to produce coherent and varied conlangs without human linguistic expertise.
>
---
#### [replaced 009] RPG: A Repository Planning Graph for Unified and Scalable Codebase Generation
- **分类: cs.CL; cs.AI; cs.SE**

- **链接: [http://arxiv.org/pdf/2509.16198v4](http://arxiv.org/pdf/2509.16198v4)**

> **作者:** Jane Luo; Xin Zhang; Steven Liu; Jie Wu; Yiming Huang; Yangyu Huang; Chengyu Yin; Ying Xin; Jianfeng Liu; Yuefeng Zhan; Hao Sun; Qi Chen; Scarlett Li; Mao Yang
>
> **摘要:** Large language models excel at generating individual functions or single files of code, yet generating complete repositories from scratch remains a fundamental challenge. This capability is key to building coherent software systems from high-level specifications and realizing the full potential of automated code generation. The process requires planning at two levels: deciding what features and modules to build (proposal stage) and defining their implementation details (implementation stage). Current approaches rely on natural language planning, which often produces unclear specifications, misaligned components, and brittle designs due to its inherent ambiguity and lack of structure. To address these limitations, we introduce the Repository Planning Graph (RPG), a structured representation that encodes capabilities, file structures, data flows, and functions in a unified graph. By replacing free-form natural language with an explicit blueprint, RPG enables consistent long-horizon planning for repository generation. Building on RPG, we develop ZeroRepo, a graph-driven framework that operates in three stages: proposal-level planning, implementation-level construction, and graph-guided code generation with test validation. To evaluate, we construct RepoCraft, a benchmark of six real-world projects with 1,052 tasks. On RepoCraft, ZeroRepo produces nearly 36K Code Lines and 445K Code Tokens, on average 3.9$\times$ larger than the strongest baseline (Claude Code), and 68$\times$ larger than other baselines. It achieves 81.5% coverage and 69.7% test accuracy, improving over Claude Code by 27.3 and 35.8 points. Further analysis shows that RPG models complex dependencies, enables more sophisticated planning through near-linear scaling, and improves agent understanding of repositories, thus accelerating localization.
>
---
#### [replaced 010] ARS: Adaptive Reasoning Suppression for Efficient Large Reasoning Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.00071v2](http://arxiv.org/pdf/2510.00071v2)**

> **作者:** Dongqi Zheng
>
> **备注:** Accepted by 39th NeurIPS - Foundations of Reasoning in Language Models
>
> **摘要:** Large Reasoning Language Models (LRLMs or LRMs) demonstrate remarkable capabilities in complex reasoning tasks, but suffer from significant computational inefficiencies due to overthinking phenomena. Existing efficient reasoning methods face the challenge of balancing reasoning quality with inference cost reduction. We propose \textbf{Adaptive Reasoning Suppression (ARS)}, a novel training-free approach that dynamically suppresses redundant reasoning steps while preserving accuracy through adaptive certainty monitoring. ARS introduces a multi-checkpoint certainty estimation mechanism with progressive suppression thresholds, achieving superior efficiency compared to static suppression methods. Our extensive evaluation across mathematical reasoning benchmarks using multiple model architectures demonstrates that ARS achieves up to 53%, 46.1%, and 57.9% in token, latency and energy reduction, while maintaining or improving accuracy.
>
---
#### [replaced 011] EVALUESTEER: Measuring Reward Model Steerability Towards Values and Preferences
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.06370v2](http://arxiv.org/pdf/2510.06370v2)**

> **作者:** Kshitish Ghate; Andy Liu; Devansh Jain; Taylor Sorensen; Atoosa Kasirzadeh; Aylin Caliskan; Mona T. Diab; Maarten Sap
>
> **备注:** Preprint under review
>
> **摘要:** As large language models (LLMs) are deployed globally, creating pluralistic systems that can accommodate the diverse preferences and values of users worldwide becomes essential. We introduce EVALUESTEER, a benchmark to measure LLMs' and reward models' (RMs) steerability towards users' value and stylistic preference profiles grounded in psychology and human-LLM interaction literature. To address the gap in existing datasets that do not support controlled evaluations of RM steering, we synthetically generated 165,888 preference pairs -- systematically varying pairs along 4 value dimensions (traditional, secular-rational, survival, and self-expression) and 4 style dimensions (verbosity, readability, confidence, and warmth). We use EVALUESTEER to evaluate whether, given a user profile and a pair of candidate value-laden and style-laden responses, LLMs and RMs are able to select the output that aligns with the user's preferences. We evaluate six open-source and proprietary LLMs and RMs under eleven systematic prompting conditions and six preference comparison scenarios. Notably, our results show that, when given the user's full profile of values and stylistic preferences, the best models achieve <75% accuracy at choosing the correct response, in contrast to >99% accuracy when only relevant style and value preferences are provided. EVALUESTEER thus highlights the limitations of current RMs at identifying and adapting to relevant user profile information, and provides a challenging testbed for developing RMs that can be steered towards diverse human values and preferences.
>
---
#### [replaced 012] DDO: Dual-Decision Optimization for LLM-Based Medical Consultation via Multi-Agent Collaboration
- **分类: cs.CL; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2505.18630v2](http://arxiv.org/pdf/2505.18630v2)**

> **作者:** Zhihao Jia; Mingyi Jia; Junwen Duan; Jianxin Wang
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Large Language Models (LLMs) demonstrate strong generalization and reasoning abilities, making them well-suited for complex decision-making tasks such as medical consultation (MC). However, existing LLM-based methods often fail to capture the dual nature of MC, which entails two distinct sub-tasks: symptom inquiry, a sequential decision-making process, and disease diagnosis, a classification problem. This mismatch often results in ineffective symptom inquiry and unreliable disease diagnosis. To address this, we propose \textbf{DDO}, a novel LLM-based framework that performs \textbf{D}ual-\textbf{D}ecision \textbf{O}ptimization by decoupling the two sub-tasks and optimizing them with distinct objectives through a collaborative multi-agent workflow. Experiments on three real-world MC datasets show that DDO consistently outperforms existing LLM-based approaches and achieves competitive performance with state-of-the-art generation-based methods, demonstrating its effectiveness in the MC task. The code is available at https://github.com/zh-jia/DDO.
>
---
#### [replaced 013] How a Bilingual LM Becomes Bilingual: Tracing Internal Representations with Sparse Autoencoders
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.06394v2](http://arxiv.org/pdf/2503.06394v2)**

> **作者:** Tatsuro Inaba; Go Kamoda; Kentaro Inui; Masaru Isonuma; Yusuke Miyao; Yohei Oseki; Benjamin Heinzerling; Yu Takagi
>
> **备注:** 13 pages, 17 figures, accepted to EMNLP 2025 findings
>
> **摘要:** This study explores how bilingual language models develop complex internal representations. We employ sparse autoencoders to analyze internal representations of bilingual language models with a focus on the effects of training steps, layers, and model sizes. Our analysis shows that language models first learn languages separately, and then gradually form bilingual alignments, particularly in the mid layers. We also found that this bilingual tendency is stronger in larger models. Building on these findings, we demonstrate the critical role of bilingual representations in model performance by employing a novel method that integrates decomposed representations from a fully trained model into a mid-training model. Our results provide insights into how language models acquire bilingual capabilities.
>
---
#### [replaced 014] Learning to Disentangle Latent Reasoning Rules with Language VAEs: A Systematic Study
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.19418v2](http://arxiv.org/pdf/2506.19418v2)**

> **作者:** Yingji Zhang; Marco Valentino; Danilo S. Carvalho; André Freitas
>
> **摘要:** Incorporating explicit reasoning rules within the latent space of language models (LMs) offers a promising pathway to enhance generalisation, interpretability, and controllability. While current Transformer-based language models have shown strong performance on Natural Language Inference (NLI) tasks, they often rely on memorisation rather than rule-based inference. This work investigates how reasoning rules can be explicitly embedded and memorised within the LMs through Language Variational Autoencoders (VAEs). We propose a complete pipeline for learning reasoning rules within Transformer-based language VAEs. This pipeline encompasses three rule-based reasoning tasks, a supporting theoretical framework, and a practical end-to-end architecture. The experiment illustrates the following findings: Disentangled reasoning: Under explicit signal supervision, reasoning rules - viewed as functional mappings - can be disentangled within the encoder's parametric space. This separation results in distinct clustering of rules in the output feature space. Prior knowledge injection: injecting reasoning information into the Query enables the model to more effectively retrieve the stored value Value from memory based on Key. This approach offers a simple method for integrating prior knowledge into decoder-only language models. Performance bottleneck: In mathematical reasoning tasks using Qwen2.5(0.5B), increasing sample count doesn't improve performance beyond a point. Moreover, ffn layers are better than attention layers at preserving the separation of reasoning rules in the model's parameters.
>
---
#### [replaced 015] Untangling Component Imbalance in Hybrid Linear Attention Conversion Methods
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.05901v2](http://arxiv.org/pdf/2510.05901v2)**

> **作者:** Martin Benfeghoul; Teresa Delgado; Adnan Oomerjee; Haitham Bou Ammar; Jun Wang; Zafeirios Fountas
>
> **摘要:** Transformers' quadratic computational complexity limits their scalability despite remarkable performance. While linear attention reduces this to linear complexity, pre-training such models from scratch remains, in most cases, prohibitively expensive. Recent post-training linearisation methods convert pre-trained Transformers to linear models efficiently, often using hybrid approaches that combine linear attention with sliding-window softmax. We identify a critical flaw: existing hybrid methods inadvertently bypass the linear component, relying almost entirely on SWA. Component-level diagnostics reveal this previously undetected behaviour stems from overlooked evaluation practices on common-sense benchmarks. We propose three solutions to ensure balanced component usage: (i) inference-time hybridisation of linear-only conversions with sliding-window softmax; (ii) HedgeCATs, combining attention-weight transfer with targeted LoRA fine-tuning; and (iii) Scheduled Sliding-window Dropout (SSD), which stochastically suppresses the softmax branch during training to prevent component collapse. Our methods maintain computational efficiency while recovering most base model performance and ensuring genuine linear attention adoption, restoring the validity of performance attributions in hybrid conversions.
>
---
#### [replaced 016] Medchain: Bridging the Gap Between LLM Agents and Clinical Practice with Interactive Sequence
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.01605v2](http://arxiv.org/pdf/2412.01605v2)**

> **作者:** Jie Liu; Wenxuan Wang; Zizhan Ma; Guolin Huang; Yihang SU; Kao-Jung Chang; Wenting Chen; Haoliang Li; Linlin Shen; Michael Lyu
>
> **备注:** Accepted by NeurIPS25 Spotlight
>
> **摘要:** Clinical decision making (CDM) is a complex, dynamic process crucial to healthcare delivery, yet it remains a significant challenge for artificial intelligence systems. While Large Language Model (LLM)-based agents have been tested on general medical knowledge using licensing exams and knowledge question-answering tasks, their performance in the CDM in real-world scenarios is limited due to the lack of comprehensive testing datasets that mirror actual medical practice. To address this gap, we present MedChain, a dataset of 12,163 clinical cases that covers five key stages of clinical workflow. MedChain distinguishes itself from existing benchmarks with three key features of real-world clinical practice: personalization, interactivity, and sequentiality. Further, to tackle real-world CDM challenges, we also propose MedChain-Agent, an AI system that integrates a feedback mechanism and a MCase-RAG module to learn from previous cases and adapt its responses. MedChain-Agent demonstrates remarkable adaptability in gathering information dynamically and handling sequential clinical tasks, significantly outperforming existing approaches.
>
---
#### [replaced 017] Fine-Tuning Large Language Models with QLoRA for Offensive Language Detection in Roman Urdu-English Code-Mixed Text
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.03683v2](http://arxiv.org/pdf/2510.03683v2)**

> **作者:** Nisar Hussain; Amna Qasim; Gull Mehak; Muhammad Zain; Momina Hafeez; Grigori Sidorov
>
> **备注:** 25 pages, 22 figures
>
> **摘要:** The use of derogatory terms in languages that employ code mixing, such as Roman Urdu, presents challenges for Natural Language Processing systems due to unstated grammar, inconsistent spelling, and a scarcity of labeled data. In this work, we propose a QLoRA based fine tuning framework to improve offensive language detection in Roman Urdu-English text. We translated the Roman Urdu-English code mixed dataset into English using Google Translate to leverage English LLMs, while acknowledging that this translation reduces direct engagement with code mixing features. Our focus is on classification performance using English translated low resource inputs. We fine tuned several transformers and large language models, including Meta LLaMA 3 8B, Mistral 7B v0.1, LLaMA 2 7B, ModernBERT, and RoBERTa, with QLoRA for memory efficient adaptation. Models were trained and evaluated on a manually annotated Roman Urdu dataset for offensive vs non offensive content. Of all tested models, the highest F1 score of 91.45 was attained by Meta LLaMA 3 8B, followed by Mistral 7B at 89.66, surpassing traditional transformer baselines. These results demonstrate the efficacy of QLoRA in fine tuning high performing models for low resource environments such as code mixed offensive language detection, and confirm the potential of LLMs for this task. This work advances a scalable approach to Roman Urdu moderation and paves the way for future multilingual offensive detection systems based on LLMs.
>
---
#### [replaced 018] Privacy-Preserving Parameter-Efficient Fine-Tuning for Large Language Model Services
- **分类: cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2305.06212v3](http://arxiv.org/pdf/2305.06212v3)**

> **作者:** Yansong Li; Zhixing Tan; Paula Branco; Yang Liu
>
> **摘要:** Parameter-Efficient Fine-Tuning (PEFT) provides a practical way for users to customize Large Language Models (LLMs) with their private data in LLM service scenarios. However, the inherently sensitive nature of private data demands robust privacy preservation measures during the customization of LLM services to ensure data security, maintain user trust, and comply with stringent regulatory standards. Based on PEFT, we propose Privacy-Preserving Parameter-Efficient Fine-Tuning (RAPT), a framework that offers privacy protection for LLM services. RAPT adopts a local privacy approach, enabling users to privatize their data locally using a text-to-text local differential privacy mechanism. Since PEFT performs poorly when directly trained on privatized data, we introduce a novel privatized token reconstruction task that is trained jointly with the downstream task, allowing LLMs to learn better task-dependent representations. Despite the simplicity of our framework, experiments show that RAPT achieves competitive performance across tasks while providing privacy guarantees against adversaries.
>
---
#### [replaced 019] Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical Perspective
- **分类: cs.CL; cs.AI; cs.CY; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.19028v5](http://arxiv.org/pdf/2506.19028v5)**

> **作者:** Weijie Xu; Yiwen Wang; Chi Xue; Xiangkun Hu; Xi Fang; Guimin Dong; Chandan K. Reddy
>
> **备注:** 29 pages, 9 figures, 15 tables
>
> **摘要:** Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo (Fine-grained Semantic Comparison), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSCo more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics.
>
---
#### [replaced 020] LightMamba: Efficient Mamba Acceleration on FPGA with Quantization and Hardware Co-design
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15260v2](http://arxiv.org/pdf/2502.15260v2)**

> **作者:** Renjie Wei; Songqiang Xu; Linfeng Zhong; Zebin Yang; Qingyu Guo; Yuan Wang; Runsheng Wang; Meng Li
>
> **备注:** Accepted by DATE 2025
>
> **摘要:** State space models (SSMs) like Mamba have recently attracted much attention. Compared to Transformer-based large language models (LLMs), Mamba achieves linear computation complexity with the sequence length and demonstrates superior performance. However, Mamba is hard to accelerate due to the scattered activation outliers and the complex computation dependency, rendering existing LLM accelerators inefficient. In this paper, we propose LightMamba that co-designs the quantization algorithm and FPGA accelerator architecture for efficient Mamba inference. We first propose an FPGA-friendly post-training quantization algorithm that features rotation-assisted quantization and power-of-two SSM quantization to reduce the majority of computation to 4-bit. We further design an FPGA accelerator that partially unrolls the Mamba computation to balance the efficiency and hardware costs. Through computation reordering as well as fine-grained tiling and fusion, the hardware utilization and memory efficiency of the accelerator get drastically improved. We implement LightMamba on Xilinx Versal VCK190 FPGA and achieve 4.65x to 6.06x higher energy efficiency over the GPU baseline. When evaluated on Alveo U280 FPGA, LightMamba reaches 93 tokens/s, which is 1.43x that of the GPU baseline. Our code is available at https://github.com/PKU-SEC-Lab/LightMamba.
>
---
#### [replaced 021] Robo-Instruct: Simulator-Augmented Instruction Alignment For Finetuning Code LLMs
- **分类: cs.CL; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2405.20179v5](http://arxiv.org/pdf/2405.20179v5)**

> **作者:** Zichao Hu; Junyi Jessy Li; Arjun Guha; Joydeep Biswas
>
> **备注:** Conference on Language Modeling (COLM) 2025, Project site: https://amrl.cs.utexas.edu/robo-instruct/
>
> **摘要:** Code LLMs have shown promising results with converting tasks in natural language to programs that can be executed by service robots. We are interested in finetuning small, specialized LLMs for this purpose, but collecting datasets of task-program pairs specific to each robot is time-consuming and expensive. While approaches such as SELF-INSTRUCT and EVOL-INSTRUCT are capable of generating novel tasks given a few examples, they are unable to provide the corresponding programs that correctly abide by physical-world and robot-constraints using the provided programming interface. Using a simulator is a natural potential solution to checking for such constraints, but building simulation environments that can handle arbitrary tasks and their necessary objects and locations, is challenging. To address these challenges, we introduce ROBO-INSTRUCT, which synthesizes task-specific simulation environments on the fly during program execution, by opportunistically inferring entity properties and enforcing corresponding constraints based on how the entities are used in the task program. Additionally, ROBO-INSTRUCT integrates an LLM-aided post-processing procedure to refine instructions for better alignment with robot programs. We demonstrate the effectiveness of ROBO-INSTRUCT across multiple LLMs, showing that our fine-tuned models outperform all baseline methods and even match or surpass the performance of several larger and proprietary models.
>
---
#### [replaced 022] NLP-ADBench: NLP Anomaly Detection Benchmark
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.04784v2](http://arxiv.org/pdf/2412.04784v2)**

> **作者:** Yuangang Li; Jiaqi Li; Zhuo Xiao; Tiankai Yang; Yi Nian; Xiyang Hu; Yue Zhao
>
> **备注:** EMNLP Findings 2025. The project is available at https://github.com/USC-FORTIS/NLP-ADBench
>
> **摘要:** Anomaly detection (AD) is an important machine learning task with applications in fraud detection, content moderation, and user behavior analysis. However, AD is relatively understudied in a natural language processing (NLP) context, limiting its effectiveness in detecting harmful content, phishing attempts, and spam reviews. We introduce NLP-ADBench, the most comprehensive NLP anomaly detection (NLP-AD) benchmark to date, which includes eight curated datasets and 19 state-of-the-art algorithms. These span 3 end-to-end methods and 16 two-step approaches that adapt classical, non-AD methods to language embeddings from BERT and OpenAI. Our empirical results show that no single model dominates across all datasets, indicating a need for automated model selection. Moreover, two-step methods with transformer-based embeddings consistently outperform specialized end-to-end approaches, with OpenAI embeddings outperforming those of BERT. We release NLP-ADBench at https://github.com/USC-FORTIS/NLP-ADBench, providing a unified framework for NLP-AD and supporting future investigations.
>
---
#### [replaced 023] CLARITY: Clinical Assistant for Routing, Inference, and Triage
- **分类: cs.CL; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2510.02463v2](http://arxiv.org/pdf/2510.02463v2)**

> **作者:** Vladimir Shaposhnikov; Aleksandr Nesterov; Ilia Kopanichuk; Ivan Bakulin; Egor Zhelvakov; Ruslan Abramov; Ekaterina Tsapieva; Iaroslav Bespalov; Dmitry V. Dylov; Ivan Oseledets
>
> **备注:** Accepted to EMNLP 2025 (Industrial Track)
>
> **摘要:** We present CLARITY (Clinical Assistant for Routing, Inference and Triage), an AI-driven platform designed to facilitate patient-to-specialist routing, clinical consultations, and severity assessment of patient conditions. Its hybrid architecture combines a Finite State Machine (FSM) for structured dialogue flows with collaborative agents that employ Large Language Model (LLM) to analyze symptoms and prioritize referrals to appropriate specialists. Built on a modular microservices framework, CLARITY ensures safe, efficient, and robust performance, flexible and readily scalable to meet the demands of existing workflows and IT solutions in healthcare. We report integration of our clinical assistant into a large-scale national interhospital platform, with more than 55,000 content-rich user dialogues completed within the two months of deployment, 2,500 of which were expert-annotated for subsequent validation. The validation results show that CLARITY surpasses human-level performance in terms of the first-attempt routing precision, naturally requiring up to 3 times shorter duration of the consultation than with a human.
>
---
#### [replaced 024] Chain-of-Retrieval Augmented Generation
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.14342v3](http://arxiv.org/pdf/2501.14342v3)**

> **作者:** Liang Wang; Haonan Chen; Nan Yang; Xiaolong Huang; Zhicheng Dou; Furu Wei
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** This paper introduces an approach for training o1-like RAG models that retrieve and reason over relevant information step by step before generating the final answer. Conventional RAG methods usually perform a single retrieval step before the generation process, which limits their effectiveness in addressing complex queries due to imperfect retrieval results. In contrast, our proposed method, CoRAG (Chain-of-Retrieval Augmented Generation), allows the model to dynamically reformulate the query based on the evolving state. To train CoRAG effectively, we utilize rejection sampling to automatically generate intermediate retrieval chains, thereby augmenting existing RAG datasets that only provide the correct final answer. At test time, we propose various decoding strategies to scale the model's test-time compute by controlling the length and number of sampled retrieval chains. Experimental results across multiple benchmarks validate the efficacy of CoRAG, particularly in multi-hop question answering tasks, where we observe more than 10 points improvement in EM score compared to strong baselines. On the KILT benchmark, CoRAG establishes a new state-of-the-art performance across a diverse range of knowledge-intensive tasks. Furthermore, we offer comprehensive analyses to understand the scaling behavior of CoRAG, laying the groundwork for future research aimed at developing factual and grounded foundation models.
>
---
#### [replaced 025] Improbable Bigrams Expose Vulnerabilities of Incomplete Tokens in Byte-Level Tokenizers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.23684v2](http://arxiv.org/pdf/2410.23684v2)**

> **作者:** Eugene Jang; Kimin Lee; Jin-Woo Chung; Keuntae Park; Seungwon Shin
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Tokenization is a crucial step that bridges human-readable text with model-readable discrete tokens. However, recent studies have revealed that tokenizers can be exploited to elicit unwanted model behaviors. In this work, we investigate incomplete tokens, i.e., undecodable tokens with stray bytes resulting from byte-level byte-pair encoding (BPE) tokenization. We hypothesize that such tokens are heavily reliant on their adjacent tokens and are fragile when paired with unfamiliar tokens. To demonstrate this vulnerability, we introduce improbable bigrams: out-of-distribution combinations of incomplete tokens designed to exploit their dependency. Our experiments show that improbable bigrams are significantly prone to hallucinatory behaviors. Surprisingly, the same phrases have drastically lower rates of hallucination (90% reduction in Llama3.1) when an alternative tokenization is used. We caution against the potential vulnerabilities introduced by byte-level BPE tokenizers, which may introduce blind spots to language models.
>
---
#### [replaced 026] ARM2: Adaptive Reasoning Model with Vision Understanding and Executable Code
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.08163v2](http://arxiv.org/pdf/2510.08163v2)**

> **作者:** Jian Xie; Zhendong Chu; Aoxiao Zhong; Kai Zhang; Mingzhe Han; Xing Fan; Jialie Shen; Qingsong Wen
>
> **备注:** Work in Progress
>
> **摘要:** Large Reasoning Models (LRMs) often suffer from the ``over-thinking'' problem, generating unnecessarily long reasoning on simple tasks. Some strategies have been proposed to mitigate this issue, such as length penalties or routing mechanisms, but they are typically heuristic and task-specific, lacking a general framework for adaptive reasoning. In this paper, we present ARM2, a unified model that adaptively balances reasoning performance and efficiency across multiple formats through a reinforcement learning framework augmented with length-aware optimization. Beyond conventional natural language inference, ARM2 integrates vision understanding, extending its applicability to multimodal. Moreover, ARM2 integrates executable code into reasoning, enabling substantial reductions in token cost while preserving task performance compared to long CoT. Experiments demonstrate that ARM2 achieves performance on par with traditional reasoning models trained with GRPO, while reducing token usage by over 70% on average. We further conduct extensive analyses to validate the effectiveness of ARM2 and the soundness of its design.
>
---
#### [replaced 027] Anemoi: A Semi-Centralized Multi-agent System Based on Agent-to-Agent Communication MCP server from Coral Protocol
- **分类: cs.MA; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.17068v3](http://arxiv.org/pdf/2508.17068v3)**

> **作者:** Xinxing Ren; Caelum Forder; Qianbo Zang; Ahsen Tahir; Roman J. Georgio; Suman Deb; Peter Carroll; Önder Gürcan; Zekun Guo
>
> **摘要:** Recent advances in generalist multi-agent systems (MAS) have largely followed a context-engineering plus centralized paradigm, where a planner agent coordinates multiple worker agents through unidirectional prompt passing. While effective under strong planner models, this design suffers from two critical limitations: (1) strong dependency on the planner's capability, which leads to degraded performance when a smaller LLM powers the planner; and (2) limited inter-agent communication, where collaboration relies on prompt concatenation rather than genuine refinement through structured discussions. To address these challenges, we propose Anemoi, a semi-centralized MAS built on the Agent-to-Agent (A2A) communication MCP server from Coral Protocol. Unlike traditional designs, Anemoi enables structured and direct inter-agent collaboration, allowing all agents to monitor progress, assess results, identify bottlenecks, and propose refinements in real time. This paradigm reduces reliance on a single planner, supports adaptive plan updates, and minimizes redundant context passing, resulting in more scalable execution. Evaluated on the GAIA benchmark, Anemoi achieved 52.73% accuracy with a small LLM (GPT-4.1-mini) as the planner, surpassing the strongest open-source baseline OWL (43.63%) by +9.09% under identical LLM settings. Our implementation is publicly available at https://github.com/Coral-Protocol/Anemoi.
>
---
#### [replaced 028] System Prompt Optimization with Meta-Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.09666v2](http://arxiv.org/pdf/2505.09666v2)**

> **作者:** Yumin Choi; Jinheon Baek; Sung Ju Hwang
>
> **备注:** NeurIPS 2025
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities, with optimizing their input prompts playing a pivotal role in maximizing their performance. However, while LLM prompts consist of both the task-agnostic system prompts and task-specific user prompts, existing work on prompt optimization has focused on user prompts specific to individual queries or tasks, and largely overlooked the system prompt that is, once optimized, applicable across different tasks and domains. Motivated by this, we introduce the novel problem of bilevel system prompt optimization, whose objective is to design system prompts that are robust to diverse user prompts and transferable to unseen tasks. To tackle this problem, we then propose a meta-learning framework, which meta-learns the system prompt by optimizing it over various user prompts across multiple datasets, while simultaneously updating the user prompts in an iterative manner to ensure synergy between them. We conduct experiments on 14 unseen datasets spanning 5 different domains, on which we show that our approach produces system prompts that generalize effectively to diverse user prompts. Also, our findings reveal that the optimized system prompt enables rapid adaptation even to unseen tasks, requiring fewer optimization steps for test-time user prompts while achieving improved performance.
>
---
#### [replaced 029] Learning to Reason Across Parallel Samples for LLM Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09014v2](http://arxiv.org/pdf/2506.09014v2)**

> **作者:** Jianing Qi; Xi Ye; Hao Tang; Zhigang Zhu; Eunsol Choi
>
> **摘要:** Scaling test-time compute brings substantial performance gains for large language models (LLMs). By sampling multiple answers and heuristically aggregate their answers (e.g., either through majority voting or using verifiers to rank the answers), one can achieve consistent performance gains in math domains. In this paper, we propose a new way to leverage such multiple sample set. We train a compact LLM, called Sample Set Aggregator (SSA), that takes a concatenated sequence of multiple samples and output the final answer, optimizing it for the answer accuracy with reinforcement learning. Experiments on five reasoning datasets demonstrate both the efficacy and efficiency of SSA. Notably, SSA improves over naive majority voting by 8% pass@5 on MATH. Furthermore, our 3B SSA surpasses model-based re-ranking with a much larger 72B process reward model. Our analysis also shows promising generalization ability of SSA, across sample set sizes, base model families and scales, and tasks. By separating LLMs to generate answers and LLMs to analyze and aggregate sampled answers, our approach can work with the outputs from premier black box models easily and efficiently.
>
---
#### [replaced 030] Beyond I'm Sorry, I Can't: Dissecting Large Language Model Refusal
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.09708v2](http://arxiv.org/pdf/2509.09708v2)**

> **作者:** Nirmalendu Prakash; Yeo Wei Jie; Amir Abdullah; Ranjan Satapathy; Erik Cambria; Roy Ka Wei Lee
>
> **摘要:** Refusal on harmful prompts is a key safety behaviour in instruction-tuned large language models (LLMs), yet the internal causes of this behaviour remain poorly understood. We study two public instruction-tuned models, Gemma-2-2B-IT and LLaMA-3.1-8B-IT, using sparse autoencoders (SAEs) trained on residual-stream activations. Given a harmful prompt, we search the SAE latent space for feature sets whose ablation flips the model from refusal to compliance, demonstrating causal influence and creating a jailbreak. Our search proceeds in three stages: (1) Refusal Direction: find a refusal-mediating direction and collect SAE features near that direction; (2) Greedy Filtering: prune to a minimal set; and (3) Interaction Discovery: fit a factorization machine (FM) that captures nonlinear interactions among the remaining active features and the minimal set. This pipeline yields a broad set of jailbreak-critical features, offering insight into the mechanistic basis of refusal. Moreover, we find evidence of redundant features that remain dormant unless earlier features are suppressed. Our findings highlight the potential for fine-grained auditing and targeted intervention in safety behaviours by manipulating the interpretable latent space.
>
---
#### [replaced 031] Understanding and Improving Information Preservation in Prompt Compression for LLMs
- **分类: cs.CL; cs.IR; cs.LG; 68T50; F.2.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2503.19114v2](http://arxiv.org/pdf/2503.19114v2)**

> **作者:** Weronika Łajewska; Momchil Hardalov; Laura Aina; Neha Anna John; Hang Su; Lluís Màrquez
>
> **备注:** Accepted to EMNLP 2025 (Findings), 22 pages, 6 figures, 24 tables
>
> **摘要:** Recent advancements in large language models (LLMs) have enabled their successful application to a broad range of tasks. However, in information-intensive tasks, the prompt length can grow fast, leading to increased computational requirements, performance degradation, and induced biases from irrelevant or redundant information. Recently, various prompt compression techniques have been introduced to optimize the trade-off between reducing input length and retaining performance. We propose a holistic evaluation framework that allows for in-depth analysis of prompt compression methods. We focus on three key aspects, besides compression ratio: (i) downstream task performance, (ii) grounding in the input context, and (iii) information preservation. Using our framework, we analyze state-of-the-art soft and hard compression methods and show that some fail to preserve key details from the original prompt, limiting performance on complex tasks. By identifying these limitations, we are able to improve one soft prompting method by controlling compression granularity, achieving up to +23% in downstream performance, +8 BERTScore points in grounding, and 2.7x more entities preserved in compression. Ultimately, we find that the best effectiveness/compression rate trade-off is achieved with soft prompting combined with sequence-level training.The code is available at https://github.com/amazon-science/information-preservation-in-prompt-compression.
>
---
#### [replaced 032] Science Hierarchography: Hierarchical Organization of Science Literature
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.13834v5](http://arxiv.org/pdf/2504.13834v5)**

> **作者:** Muhan Gao; Jash Shah; Weiqi Wang; Kuan-Hao Huang; Daniel Khashabi
>
> **摘要:** Scientific knowledge is growing rapidly, making it difficult to track progress and high-level conceptual links across broad disciplines. While tools like citation networks and search engines help retrieve related papers, they lack the abstraction needed to capture the needed to represent the density and structure of activity across subfields. We motivate SCIENCE HIERARCHOGRAPHY, the goal of organizing scientific literature into a high-quality hierarchical structure that spans multiple levels of abstraction -- from broad domains to specific studies. Such a representation can provide insights into which fields are well-explored and which are under-explored. To achieve this goal, we develop a hybrid approach that combines efficient embedding-based clustering with LLM-based prompting, striking a balance between scalability and semantic precision. Compared to LLM-heavy methods like iterative tree construction, our approach achieves superior quality-speed trade-offs. Our hierarchies capture different dimensions of research contributions, reflecting the interdisciplinary and multifaceted nature of modern science. We evaluate its utility by measuring how effectively an LLM-based agent can navigate the hierarchy to locate target papers. Results show that our method improves interpretability and offers an alternative pathway for exploring scientific literature beyond traditional search methods. Code, data and demo are available: https://github.com/JHU-CLSP/science-hierarchography
>
---
#### [replaced 033] Online Rubrics Elicitation from Pairwise Comparisons
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.07284v2](http://arxiv.org/pdf/2510.07284v2)**

> **作者:** MohammadHossein Rezaei; Robert Vacareanu; Zihao Wang; Clinton Wang; Bing Liu; Yunzhong He; Afra Feyza Akyürek
>
> **摘要:** Rubrics provide a flexible way to train LLMs on open-ended long-form answers where verifiable rewards are not applicable and human preferences provide coarse signals. Prior work shows that reinforcement learning with rubric-based rewards leads to consistent gains in LLM post-training. Most existing approaches rely on rubrics that remain static over the course of training. Such static rubrics, however, are vulnerable to reward-hacking type behaviors and fail to capture emergent desiderata that arise during training. We introduce Online Rubrics Elicitation (OnlineRubrics), a method that dynamically curates evaluation criteria in an online manner through pairwise comparisons of responses from current and reference policies. This online process enables continuous identification and mitigation of errors as training proceeds. Empirically, this approach yields consistent improvements of up to 8% over training exclusively with static rubrics across AlpacaEval, GPQA, ArenaHard as well as the validation sets of expert questions and rubrics. We qualitatively analyze the elicited criteria and identify prominent themes such as transparency, practicality, organization, and reasoning.
>
---
#### [replaced 034] CausalVLBench: Benchmarking Visual Causal Reasoning in Large Vision-Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11034v2](http://arxiv.org/pdf/2506.11034v2)**

> **作者:** Aneesh Komanduri; Karuna Bhaila; Xintao Wu
>
> **备注:** Accepted to the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025 Main)
>
> **摘要:** Large language models (LLMs) have shown remarkable ability in various language tasks, especially with their emergent in-context learning capability. Extending LLMs to incorporate visual inputs, large vision-language models (LVLMs) have shown impressive performance in tasks such as recognition and visual question answering (VQA). Despite increasing interest in the utility of LLMs in causal reasoning tasks such as causal discovery and counterfactual reasoning, there has been relatively little work showcasing the abilities of LVLMs on visual causal reasoning tasks. We take this opportunity to formally introduce a comprehensive causal reasoning benchmark for multi-modal in-context learning from LVLMs. Our CausalVLBench encompasses three representative tasks: causal structure inference, intervention target prediction, and counterfactual prediction. We evaluate the ability of state-of-the-art open-source LVLMs on our causal reasoning tasks across three causal representation learning datasets and demonstrate their fundamental strengths and weaknesses. We hope that our benchmark elucidates the drawbacks of existing vision-language models and motivates new directions and paradigms in improving the visual causal reasoning abilities of LVLMs.
>
---
#### [replaced 035] What Are They Filtering Out? An Experimental Benchmark of Filtering Strategies for Harm Reduction in Pretraining Datasets
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.05721v2](http://arxiv.org/pdf/2503.05721v2)**

> **作者:** Marco Antonio Stranisci; Christian Hardmeier
>
> **摘要:** Data filtering strategies are a crucial component to develop safe Large Language Models (LLM), since they support the removal of harmful contents from pretraining datasets. There is a lack of research on the actual impact of these strategies on vulnerable groups to discrimination, though, and their effectiveness has not been yet systematically addressed. In this paper we present a benchmark study of data filtering strategies for harm reduction aimed at providing a systematic evaluation on these approaches. We provide an overview $55$ technical reports of English LMs and LLMs to identify the existing filtering strategies in literature and implement an experimental setting to test their impact against vulnerable groups. Our results show that the positive impact that strategies have in reducing harmful contents from documents has the side effect of increasing the underrepresentation of vulnerable groups to discrimination in datasets. WARNING: the paper could contain racist, sexist, violent, and generally offensive contents
>
---
#### [replaced 036] Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.17192v4](http://arxiv.org/pdf/2504.17192v4)**

> **作者:** Minju Seo; Jinheon Baek; Seongyun Lee; Sung Ju Hwang
>
> **摘要:** Despite the rapid growth of machine learning research, corresponding code implementations are often unavailable, making it slow and labor-intensive for researchers to reproduce results and build upon prior work. In the meantime, recent Large Language Models (LLMs) excel at understanding scientific documents and generating high-quality code. Inspired by this, we introduce PaperCoder, a multi-agent LLM framework that transforms machine learning papers into functional code repositories. PaperCoder operates in three stages: planning, where it constructs a high-level roadmap, designs the system architecture with diagrams, identifies file dependencies, and generates configuration files; analysis, which focuses on interpreting implementation-specific details; and generation, where modular, dependency-aware code is produced. Moreover, each phase is instantiated through a set of specialized agents designed to collaborate effectively across the pipeline. We then evaluate PaperCoder on generating code implementations from machine learning papers based on both model-based and human evaluations, particularly from the authors of those papers, with author-released repositories as ground truth if available. Our results demonstrate the effectiveness of PaperCoder in creating high-quality, faithful implementations. Furthermore, it consistently shows strengths in the recently released PaperBench benchmark, surpassing strong baselines by substantial margins. Code is available at: https://github.com/going-doer/Paper2Code.
>
---
#### [replaced 037] RAISE: Reinforced Adaptive Instruction Selection For Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.07282v4](http://arxiv.org/pdf/2504.07282v4)**

> **作者:** Qingsong Lv; Yangning Li; Zihua Lan; Zishan Xu; Jiwei Tang; Tingwei Lu; Yinghui Li; Wenhao Jiang; Hong-Gee Kim; Hai-Tao Zheng; Philip S. Yu
>
> **备注:** Accepted by EMNLP 2025 findings
>
> **摘要:** In the instruction fine-tuning of large language models (LLMs), it is widely recognized that a few high-quality instructions are superior to a large number of low-quality instructions. At present, many instruction selection methods have been proposed, but most of these methods select instruction based on heuristic quality metrics, and only consider data selection before training. These designs lead to insufficient optimization of instruction fine-tuning, and fixed heuristic indicators are often difficult to optimize for specific tasks. Therefore, we design a dynamic, task-objective-driven instruction selection framework RAISE(Reinforced Adaptive Instruction SElection), which incorporates the entire instruction fine-tuning process into optimization, selecting instructions at each step based on the expected impact of each instruction on model performance improvement. Our approach is well interpretable and has strong task-specific optimization capabilities. By modeling dynamic instruction selection as a sequential decision-making process, we use RL to train our selection strategy. Extensive experiments and result analysis prove the superiority of our method compared with other instruction selection methods. Notably, RAISE achieves superior performance by updating only 1% of the training steps compared to full-data training, demonstrating its efficiency and effectiveness.
>
---
#### [replaced 038] Deploying Tiny LVLM Judges for Real-World Evaluation of Chart Models: Lessons Learned and Best Practices
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.07545v2](http://arxiv.org/pdf/2510.07545v2)**

> **作者:** Md Tahmid Rahman Laskar; Mohammed Saidul Islam; Ridwan Mahbub; Mizanur Rahman; Amran Bhuiyan; Israt Jahan; Mir Tafseer Nayeem; Shafiq Joty; Enamul Hoque; Jimmy Huang
>
> **备注:** Accepted to the EMNLP 2025 Industry Track
>
> **摘要:** Large Vision-Language Models (LVLMs) with only 7B parameters have shown promise as automated judges in chart comprehension tasks. However, tiny models (<=2B parameters) still perform poorly as judges, limiting their real-world use in resource-constrained settings. To address this, we propose two approaches to ensure cost-efficient evaluation: (i) multi-criteria prompting, which combines separate evaluation criteria into a single query, and (ii) domain-adaptive transfer learning, in which we fine-tune a 2B-parameter LVLM on synthetic judgments in a chart dataset to create the ChartJudge. Experiments show that multi-criteria prompting exposes robustness gaps, which led to a huge drop in performance for 7B models, including specialized LVLM judges like LLaVA-Critic. In addition, we find that our tiny LVLM (ChartJudge) can effectively transfer knowledge from one dataset to another to make it a more specialized model. Our fine-grained analysis across chart types and query complexities offers actionable insights into trade-offs between model size, prompt design, and transferability, enabling scalable, low-cost evaluation for chart reasoning tasks.
>
---
#### [replaced 039] Issue Localization via LLM-Driven Iterative Code Graph Searching
- **分类: cs.SE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.22424v3](http://arxiv.org/pdf/2503.22424v3)**

> **作者:** Zhonghao Jiang; Xiaoxue Ren; Meng Yan; Wei Jiang; Yong Li; Zhongxin Liu
>
> **备注:** Accepted by ASE 2025
>
> **摘要:** Issue solving aims to generate patches to fix reported issues in real-world code repositories according to issue descriptions. Issue localization forms the basis for accurate issue solving. Recently, LLM-based issue localization methods have demonstrated state-of-the-art performance. However, these methods either search from files mentioned in issue descriptions or in the whole repository and struggle to balance the breadth and depth of the search space to converge on the target efficiently. Moreover, they allow LLM to explore whole repositories freely, making it challenging to control the search direction to prevent the LLM from searching for incorrect targets. This paper introduces CoSIL, an LLM-driven, powerful function-level issue localization method without training or indexing. CoSIL employs a two-phase code graph search strategy. It first conducts broad exploration at the file level using dynamically constructed module call graphs, and then performs in-depth analysis at the function level by expanding the module call graph into a function call graph and executing iterative searches. To precisely control the search direction, CoSIL designs a pruner to filter unrelated directions and irrelevant contexts. To avoid incorrect interaction formats in long contexts, CoSIL introduces a reflection mechanism that uses additional independent queries in short contexts to enhance formatted abilities. Experiment results demonstrate that CoSIL achieves a Top-1 localization accuracy of 43.3\% and 44.6\% on SWE-bench Lite and SWE-bench Verified, respectively, with Qwen2.5-Coder-32B, average outperforming the state-of-the-art methods by 96.04\%. When CoSIL is integrated into an issue-solving method, Agentless, the issue resolution rate improves by 2.98\%--30.5\%.
>
---
#### [replaced 040] Improving Image Captioning Descriptiveness by Ranking and LLM-based Fusion
- **分类: cs.CV; cs.AI; cs.CL; cs.DB; cs.LG**

- **链接: [http://arxiv.org/pdf/2306.11593v3](http://arxiv.org/pdf/2306.11593v3)**

> **作者:** Luigi Celona; Simone Bianco; Marco Donzella; Paolo Napoletano
>
> **备注:** This manuscript has been accepted for publication in Springer Neural Computing and Applications
>
> **摘要:** State-of-The-Art (SoTA) image captioning models are often trained on the MicroSoft Common Objects in Context (MS-COCO) dataset, which contains human-annotated captions with an average length of approximately ten tokens. Although effective for general scene understanding, these short captions often fail to capture complex scenes and convey detailed information. Moreover, captioning models tend to exhibit bias towards the ``average'' caption, which captures only the more general aspects, thus overlooking finer details. In this paper, we present a novel approach to generate richer and more informative image captions by combining the captions generated from different SoTA captioning models. Our proposed method requires no additional model training: given an image, it leverages pre-trained models from the literature to generate the initial captions, and then ranks them using a newly introduced image-text-based metric, which we name BLIPScore. Subsequently, the top two captions are fused using a Large Language Model (LLM) to produce the final, more detailed description. Experimental results on the MS-COCO and Flickr30k test sets demonstrate the effectiveness of our approach in terms of caption-image alignment and hallucination reduction according to the ALOHa, CAPTURE, and Polos metrics. A subjective study lends additional support to these results, suggesting that the captions produced by our model are generally perceived as more consistent with human judgment. By combining the strengths of diverse SoTA models, our method enhances the quality and appeal of image captions, bridging the gap between automated systems and the rich and informative nature of human-generated descriptions. This advance enables the generation of more suitable captions for the training of both vision-language and captioning models.
>
---
#### [replaced 041] P2P: A Poison-to-Poison Remedy for Reliable Backdoor Defense in LLMs
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.04503v2](http://arxiv.org/pdf/2510.04503v2)**

> **作者:** Shuai Zhao; Xinyi Wu; Shiqian Zhao; Xiaobao Wu; Zhongliang Guo; Yanhao Jia; Anh Tuan Luu
>
> **摘要:** During fine-tuning, large language models (LLMs) are increasingly vulnerable to data-poisoning backdoor attacks, which compromise their reliability and trustworthiness. However, existing defense strategies suffer from limited generalization: they only work on specific attack types or task settings. In this study, we propose Poison-to-Poison (P2P), a general and effective backdoor defense algorithm. P2P injects benign triggers with safe alternative labels into a subset of training samples and fine-tunes the model on this re-poisoned dataset by leveraging prompt-based learning. This enforces the model to associate trigger-induced representations with safe outputs, thereby overriding the effects of original malicious triggers. Thanks to this robust and generalizable trigger-based fine-tuning, P2P is effective across task settings and attack types. Theoretically and empirically, we show that P2P can neutralize malicious backdoors while preserving task performance. We conduct extensive experiments on classification, mathematical reasoning, and summary generation tasks, involving multiple state-of-the-art LLMs. The results demonstrate that our P2P algorithm significantly reduces the attack success rate compared with baseline models. We hope that the P2P can serve as a guideline for defending against backdoor attacks and foster the development of a secure and trustworthy LLM community.
>
---
#### [replaced 042] COMPACT: Common-token Optimized Model Pruning Across Channels and Tokens
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.06836v3](http://arxiv.org/pdf/2509.06836v3)**

> **作者:** Eugene Kwek; Wenpeng Yin
>
> **摘要:** Making large language models (LLMs) more efficient in memory, latency, and serving cost is crucial for edge deployment, interactive applications, and sustainable inference at scale. Pruning is a promising technique, but existing pruning methods are limited: width pruning often breaks the standard transformer layout, requiring custom inference code, while depth pruning can cause abrupt accuracy drops. Also, while many pruning approaches are effective against LLMs, they struggle to maintain performance on small language models (SLMs). In this work, we propose COMPACT, which jointly (i) prunes rare vocabulary to shrink embedding/LM head layers and (ii) prunes FFN intermediate channels using common-token-weighted activations, aligning importance with the post-pruning token distribution. COMPACT inherits strengths of both depth and width pruning, such as: deployment-friendliness (keeps a standard transformer architecture), scale-adaptivity (trade off vocab. vs. FFN pruning), competitive pruning times, and strong memory savings alongside throughput gains. Experiments across Qwen, LLaMA, and Gemma families (0.5B-70B) show state-of-the-art downstream performance, with substantial reductions in parameters, GPU memory, and latency.
>
---
#### [replaced 043] Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.01171v3](http://arxiv.org/pdf/2510.01171v3)**

> **作者:** Jiayi Zhang; Simon Yu; Derek Chong; Anthony Sicilia; Michael R. Tomz; Christopher D. Manning; Weiyan Shi
>
> **备注:** 82 pages, 28 figures, 32 tables. Code is available at https://github.com/CHATS-lab/verbalize-sampling
>
> **摘要:** Post-training alignment often reduces LLM diversity, leading to a phenomenon known as mode collapse. Unlike prior work that attributes this effect to algorithmic limitations, we identify a fundamental, pervasive data-level driver: typicality bias in preference data, whereby annotators systematically favor familiar text as a result of well-established findings in cognitive psychology. We formalize this bias theoretically, verify it on preference datasets empirically, and show that it plays a central role in mode collapse. Motivated by this analysis, we introduce Verbalized Sampling, a simple, training-free prompting strategy to circumvent mode collapse. VS prompts the model to verbalize a probability distribution over a set of responses (e.g., "Generate 5 jokes about coffee and their corresponding probabilities"). Comprehensive experiments show that VS significantly improves performance across creative writing (poems, stories, jokes), dialogue simulation, open-ended QA, and synthetic data generation, without sacrificing factual accuracy and safety. For instance, in creative writing, VS increases diversity by 1.6-2.1x over direct prompting. We further observe an emergent trend that more capable models benefit more from VS. In sum, our work provides a new data-centric perspective on mode collapse and a practical inference-time remedy that helps unlock pre-trained generative diversity.
>
---
#### [replaced 044] Scalable multilingual PII annotation for responsible AI in LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.06250v2](http://arxiv.org/pdf/2510.06250v2)**

> **作者:** Bharti Meena; Joanna Skubisz; Harshit Rajgarhia; Nand Dave; Kiran Ganesh; Shivali Dalmia; Abhishek Mukherji; Vasudevan Sundarababu
>
> **摘要:** As Large Language Models (LLMs) gain wider adoption, ensuring their reliable handling of Personally Identifiable Information (PII) across diverse regulatory contexts has become essential. This work introduces a scalable multilingual data curation framework designed for high-quality PII annotation across 13 underrepresented locales, covering approximately 336 locale-specific PII types. Our phased, human-in-the-loop annotation methodology combines linguistic expertise with rigorous quality assurance, leading to substantial improvements in recall and false positive rates from pilot, training, and production phases. By leveraging inter-annotator agreement metrics and root-cause analysis, the framework systematically uncovers and resolves annotation inconsistencies, resulting in high-fidelity datasets suitable for supervised LLM fine-tuning. Beyond reporting empirical gains, we highlight common annotator challenges in multilingual PII labeling and demonstrate how iterative, analytics-driven pipelines can enhance both annotation quality and downstream model reliability.
>
---
#### [replaced 045] An LLM-as-Judge Metric for Bridging the Gap with Human Evaluation in SE Tasks
- **分类: cs.SE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20854v2](http://arxiv.org/pdf/2505.20854v2)**

> **作者:** Xin Zhou; Kisub Kim; Ting Zhang; Martin Weyssow; Luis F. Gomes; Guang Yang; Kui Liu; Xin Xia; David Lo
>
> **备注:** 13 pages
>
> **摘要:** Large Language Models (LLMs) and other automated techniques have been increasingly used to support software developers by generating software artifacts such as code snippets, patches, and comments. However, accurately assessing the correctness of these generated artifacts remains a significant challenge. On one hand, human evaluation provides high accuracy but is labor-intensive and lacks scalability. On the other hand, many automatic evaluation metrics are scalable and require minimal human effort, but they often fail to accurately reflect the actual correctness of generated software artifacts. In this paper, we present SE-Jury, the first evaluation metric for LLM-as-Ensemble-Judge specifically designed to accurately assess the correctness of generated software artifacts. SE-Jury first defines five distinct evaluation strategies, each implemented by an independent judge. A dynamic team selection mechanism then identifies the most appropriate subset of judges as a team to produce a final correctness score through ensembling. We evaluate SE-Jury across a diverse set of software engineering (SE) benchmarks that span three popular SE tasks: code generation, automated program repair, and code summarization. Results demonstrate that SE-Jury consistently achieves a higher correlation with human judgments, with improvements ranging from 29.6% to 140.8% over existing automatic metrics. SE-Jury reaches agreement levels with human annotators that are close to inter-annotator agreement in code generation and program repair. These findings underscore SE-Jury's potential as a scalable and reliable alternative to human evaluation in these SE tasks.
>
---
#### [replaced 046] Reinforce-Ada: An Adaptive Sampling Framework for Reinforce-Style LLM Training
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2510.04996v2](http://arxiv.org/pdf/2510.04996v2)**

> **作者:** Wei Xiong; Chenlu Ye; Baohao Liao; Hanze Dong; Xinxing Xu; Christof Monz; Jiang Bian; Nan Jiang; Tong Zhang
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Reinforcement learning applied to large language models (LLMs) for reasoning tasks is often bottlenecked by unstable gradient estimates due to fixed and uniform sampling of responses across prompts. Prior work such as GVM-RAFT addresses this by dynamically allocating inference budget per prompt to minimize stochastic gradient variance under a budget constraint. Inspired by this insight, we propose Reinforce-Ada, an adaptive sampling framework for online RL post-training of LLMs that continuously reallocates sampling effort to the prompts with the greatest uncertainty or learning potential. Unlike conventional two-stage allocation methods, Reinforce-Ada interleaves estimation and sampling in an online successive elimination process, and automatically stops sampling for a prompt once sufficient signal is collected. To stabilize updates, we form fixed-size groups with enforced reward diversity and compute advantage baselines using global statistics aggregated over the adaptive sampling phase. Empirical results across multiple model architectures and reasoning benchmarks show that Reinforce-Ada accelerates convergence and improves final performance compared to GRPO, especially when using the balanced sampling variant. Our work highlights the central role of variance-aware, adaptive data curation in enabling efficient and reliable reinforcement learning for reasoning-capable LLMs. Code is available at https://github.com/RLHFlow/Reinforce-Ada.
>
---
#### [replaced 047] RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.13726v2](http://arxiv.org/pdf/2501.13726v2)**

> **作者:** Shi-Qi Yan; Quan Liu; Zhen-Hua Ling
>
> **摘要:** While Retrieval-Augmented Generation (RAG) has exhibited promise in utilizing external knowledge, its generation process heavily depends on the quality and accuracy of the retrieved context. Large language models (LLMs) struggle to evaluate the correctness of non-parametric knowledge retrieved externally when it differs from internal memorization, leading to knowledge conflicts during response generation. To this end, we introduce the Retrieval Preference Optimization (RPO), a lightweight and effective alignment method to adaptively leverage multi-source knowledge based on retrieval relevance. An implicit representation of retrieval relevance is derived and incorporated into the reward model to integrate retrieval evaluation and response generation into a single model, solving the problem that previous methods necessitate the additional procedure to assess the retrieval quality. Notably, RPO is the only RAG-dedicated alignment approach that quantifies the awareness of retrieval relevance in training, overcoming mathematical obstacles. Experiments on four datasets demonstrate that RPO outperforms RAG by 4-10% in accuracy without any extra component, exhibiting its robust generalization.
>
---
#### [replaced 048] CAFL-L: Constraint-Aware Federated Learning with Lagrangian Dual Optimization for On-Device Language Models
- **分类: cs.LG; cs.CL; cs.DC**

- **链接: [http://arxiv.org/pdf/2510.03298v2](http://arxiv.org/pdf/2510.03298v2)**

> **作者:** Dongqi Zheng; Wenjin Fu
>
> **备注:** Accepted by 39th NeurIPS - Constrained Optimization for Machine Learning
>
> **摘要:** We introduce Constraint-Aware Federated Learning with Lagrangian Dual Optimization (CAFL-L), a principled extension of FedAvg that explicitly incorporates device-level resource constraints including energy, communication, memory, and thermal budgets. CAFL-L employs Lagrangian dual optimization to dynamically adapt training hyperparameters -- freezing depth, local steps, batch size, and communication compression -- while preserving training stability through token-budget preservation via gradient accumulation. Experiments on a character-level language model demonstrate that CAFL-L achieves superior constraint satisfaction compared to standard FedAvg (reducing memory usage by 20% and communication by 95%) while maintaining competitive validation performance, making it practical for deployment on resource-constrained edge devices.
>
---
#### [replaced 049] RE-Searcher: Robust Agentic Search with Goal-oriented Planning and Self-reflection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.26048v2](http://arxiv.org/pdf/2509.26048v2)**

> **作者:** Daocheng Fu; Jianbiao Mei; Licheng Wen; Xuemeng Yang; Cheng Yang; Rong Wu; Tao Hu; Siqi Li; Yufan Shen; Xinyu Cai; Pinlong Cai; Botian Shi; Yong Liu; Yu Qiao
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Large language models (LLMs) excel at knowledge-intensive question answering and reasoning, yet their real-world deployment remains constrained by knowledge cutoff, hallucination, and limited interaction modalities. Augmenting LLMs with external search tools helps alleviate these issues, but it also exposes agents to a complex search environment in which small, plausible variations in query formulation can steer reasoning into unproductive trajectories and amplify errors. We present a systematic analysis that quantifies how environmental complexity induces fragile search behaviors and, in turn, degrades overall performance. To address this challenge, we propose a simple yet effective approach to instantiate a search agent, RE-Searcher. During search, RE-Searcher explicitly articulates a concrete search goal and subsequently reflects on whether the retrieved evidence satisfies that goal. This combination of goal-oriented planning and self-reflection enables RE-Searcher to resist spurious cues in complex search environments and perform robust search. Extensive experiments show that our method improves search accuracy and achieves state-of-the-art results. Perturbation studies further demonstrate substantial resilience to noisy or misleading external signals, mitigating the fragility of the search process. We believe these findings offer practical guidance for integrating LLM-powered agents into more complex interactive environments and enabling more autonomous decision-making.
>
---
#### [replaced 050] ViClaim: A Multilingual Multilabel Dataset for Automatic Claim Detection in Videos
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.12882v3](http://arxiv.org/pdf/2504.12882v3)**

> **作者:** Patrick Giedemann; Pius von Däniken; Jan Deriu; Alvaro Rodrigo; Anselmo Peñas; Mark Cieliebak
>
> **摘要:** The growing influence of video content as a medium for communication and misinformation underscores the urgent need for effective tools to analyze claims in multilingual and multi-topic settings. Existing efforts in misinformation detection largely focus on written text, leaving a significant gap in addressing the complexity of spoken text in video transcripts. We introduce ViClaim, a dataset of 1,798 annotated video transcripts across three languages (English, German, Spanish) and six topics. Each sentence in the transcripts is labeled with three claim-related categories: fact-check-worthy, fact-non-check-worthy, or opinion. We developed a custom annotation tool to facilitate the highly complex annotation process. Experiments with state-of-the-art multilingual language models demonstrate strong performance in cross-validation (macro F1 up to 0.896) but reveal challenges in generalization to unseen topics, particularly for distinct domains. Our findings highlight the complexity of claim detection in video transcripts. ViClaim offers a robust foundation for advancing misinformation detection in video-based communication, addressing a critical gap in multimodal analysis.
>
---
#### [replaced 051] Beyond Demonstrations: Dynamic Vector Construction from Latent Representations
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20318v2](http://arxiv.org/pdf/2505.20318v2)**

> **作者:** Wang Cai; Hsiu-Yuan Huang; Zhixiang Wang; Yunfang Wu
>
> **备注:** 9 pages, 4 figures. Accepted to EMNLP 2025 (Main Conference)
>
> **摘要:** In-Context derived Vector (ICV) methods extract task-relevant representations from large language models (LLMs) and reinject them during inference, achieving comparable performance to few-shot In-Context Learning (ICL) without repeated demonstration processing. However, existing ICV methods remain sensitive to ICL-specific factors, often use coarse or semantically fragmented representations as the source of the vector, and rely on heuristic-based injection positions, limiting their applicability. To address these issues, we propose Dynamic Vector (DyVec), which incorporates an Exhaustive Query Rotation (EQR) strategy to extract robust semantically aggregated latent representations by mitigating variance introduced by ICL. It then applies Dynamic Latent Segmentation and Injection to adaptively partition representations based on task complexity and leverages REINFORCE-based optimization to learn optimal injection positions for each segment. Experiments results show that DyVec outperforms few-shot ICL, LoRA, and prior ICV baselines. Further analysis highlights the effectiveness of dynamically segmenting and injecting semantically aggregated latent representations. DyVec provides a lightweight and data-efficient solution for inference-time task adaptation.
>
---
#### [replaced 052] On the Reliability of Large Language Models for Causal Discovery
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.19638v2](http://arxiv.org/pdf/2407.19638v2)**

> **作者:** Tao Feng; Lizhen Qu; Niket Tandon; Zhuang Li; Xiaoxi Kang; Gholamreza Haffari
>
> **备注:** ACL 2025
>
> **摘要:** This study investigates the efficacy of Large Language Models (LLMs) in causal discovery. Using newly available open-source LLMs, OLMo and BLOOM, which provide access to their pre-training corpora, we investigate how LLMs address causal discovery through three research questions. We examine: (i) the impact of memorization for accurate causal relation prediction, (ii) the influence of incorrect causal relations in pre-training data, and (iii) the contextual nuances that influence LLMs' understanding of causal relations. Our findings indicate that while LLMs are effective in recognizing causal relations that occur frequently in pre-training data, their ability to generalize to new or rare causal relations is limited. Moreover, the presence of incorrect causal relations significantly undermines the confidence of LLMs in corresponding correct causal relations, and the contextual information critically affects the outcomes of LLMs to discern causal connections between random variables.
>
---
#### [replaced 053] LATTE: Learning Aligned Transactions and Textual Embeddings for Bank Clients
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.10021v3](http://arxiv.org/pdf/2508.10021v3)**

> **作者:** Egor Fadeev; Dzhambulat Mollaev; Aleksei Shestov; Omar Zoloev; Artem Sakhno; Dmitry Korolev; Ivan Kireev; Andrey Savchenko; Maksim Makarenko
>
> **摘要:** Learning clients embeddings from sequences of their historic communications is central to financial applications. While large language models (LLMs) offer general world knowledge, their direct use on long event sequences is computationally expensive and impractical in real-world pipelines. In this paper, we propose LATTE, a contrastive learning framework that aligns raw event embeddings with semantic embeddings from frozen LLMs. Behavioral features are summarized into short prompts, embedded by the LLM, and used as supervision via contrastive loss. The proposed approach significantly reduces inference cost and input size compared to conventional processing of complete sequence by LLM. We experimentally show that our method outperforms state-of-the-art techniques for learning event sequence representations on real-world financial datasets while remaining deployable in latency-sensitive environments.
>
---
#### [replaced 054] Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07180v2](http://arxiv.org/pdf/2506.07180v2)**

> **作者:** Wenrui Zhou; Mohamed Hendy; Shu Yang; Qingsong Yang; Zikun Guo; Yuyu Luo; Lijie Hu; Di Wang
>
> **备注:** 28 pages
>
> **摘要:** As video large language models (Video-LLMs) become increasingly integrated into real-world applications that demand grounded multimodal reasoning, ensuring their factual consistency and reliability is of critical importance. However, sycophancy, the tendency of these models to align with user input even when it contradicts the visual evidence, undermines their trustworthiness in such contexts. Current sycophancy research has largely overlooked its specific manifestations in the video-language domain, resulting in a notable absence of systematic benchmarks and targeted evaluations to understand how Video-LLMs respond under misleading user input. To fill this gap, we propose VISE (Video-LLM Sycophancy Benchmarking and Evaluation), the first benchmark designed to evaluate sycophantic behavior in state-of-the-art Video-LLMs across diverse question formats, prompt biases, and visual reasoning tasks. Specifically, VISE pioneeringly brings linguistic perspectives on sycophancy into the video domain, enabling fine-grained analysis across multiple sycophancy types and interaction patterns. Furthermore, we propose two potential training-free mitigation strategies, revealing potential paths for reducing sycophantic bias: (i) enhancing visual grounding through interpretable key-frame selection and (ii) steering model behavior away from sycophancy via targeted, inference-time intervention on its internal neural representations. Our code is available at https://github.com/William030422/Video-Sycophancy.
>
---
#### [replaced 055] Large Language Model Agent for Modular Task Execution in Drug Discovery
- **分类: cs.LG; cs.CL; q-bio.BM**

- **链接: [http://arxiv.org/pdf/2507.02925v2](http://arxiv.org/pdf/2507.02925v2)**

> **作者:** Janghoon Ock; Radheesh Sharma Meda; Srivathsan Badrinarayanan; Neha S. Aluru; Achuth Chandrasekhar; Amir Barati Farimani
>
> **摘要:** We present a modular framework powered by large language models (LLMs) that automates and streamlines key tasks across the early-stage computational drug discovery pipeline. By combining LLM reasoning with domain-specific tools, the framework performs biomedical data retrieval, domain-specific question answering, molecular generation, property prediction, property-aware molecular refinement, and 3D protein-ligand structure generation. In a case study targeting BCL-2 in lymphocytic leukemia, the agent autonomously retrieved relevant biomolecular information, including FASTA sequences, SMILES representations, and literature, and answered mechanistic questions with improved contextual accuracy compared to standard LLMs. It then generated chemically diverse seed molecules and predicted 67 ADMET-related properties, which guided iterative molecular refinement. Across two refinement rounds, the number of molecules with QED > 0.6 increased from 34 to 55. The number of molecules satisfying empirical drug-likeness filters also rose; for example, compliance with the Ghose filter increased from 32 to 55 within a pool of 100 molecules. The framework also employed Boltz-2 to generate 3D protein-ligand complexes and provide rapid binding affinity estimates for candidate compounds. These results demonstrate that the approach effectively supports molecular screening, prioritization, and structure evaluation. Its modular design enables flexible integration of evolving tools and models, providing a scalable foundation for AI-assisted therapeutic discovery.
>
---
#### [replaced 056] DeHate: A Stable Diffusion-based Multimodal Approach to Mitigate Hate Speech in Images
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.21787v2](http://arxiv.org/pdf/2509.21787v2)**

> **作者:** Dwip Dalal; Gautam Vashishtha; Anku Rani; Aishwarya Reganti; Parth Patwa; Mohd Sarique; Chandan Gupta; Keshav Nath; Viswanatha Reddy; Vinija Jain; Aman Chadha; Amitava Das; Amit Sheth; Asif Ekbal
>
> **备注:** Defactify 3 workshop at AAAI 2024
>
> **摘要:** The rise in harmful online content not only distorts public discourse but also poses significant challenges to maintaining a healthy digital environment. In response to this, we introduce a multimodal dataset uniquely crafted for identifying hate in digital content. Central to our methodology is the innovative application of watermarked, stability-enhanced, stable diffusion techniques combined with the Digital Attention Analysis Module (DAAM). This combination is instrumental in pinpointing the hateful elements within images, thereby generating detailed hate attention maps, which are used to blur these regions from the image, thereby removing the hateful sections of the image. We release this data set as a part of the dehate shared task. This paper also describes the details of the shared task. Furthermore, we present DeHater, a vision-language model designed for multimodal dehatification tasks. Our approach sets a new standard in AI-driven image hate detection given textual prompts, contributing to the development of more ethical AI applications in social media.
>
---
#### [replaced 057] Can Lessons From Human Teams Be Applied to Multi-Agent Systems? The Role of Structure, Diversity, and Interaction Dynamics
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.07488v2](http://arxiv.org/pdf/2510.07488v2)**

> **作者:** Rasika Muralidharan; Haewoon Kwak; Jisun An
>
> **备注:** Under Review at ARR
>
> **摘要:** Multi-Agent Systems (MAS) with Large Language Model (LLM)-powered agents are gaining attention, yet fewer studies explore their team dynamics. Inspired by human team science, we propose a multi-agent framework to examine core aspects of team science: structure, diversity, and interaction dynamics. We evaluate team performance across four tasks: CommonsenseQA, StrategyQA, Social IQa, and Latent Implicit Hate, spanning commonsense and social reasoning. Our results show that flat teams tend to perform better than hierarchical ones, while diversity has a nuanced impact. Interviews suggest agents are overconfident about their team performance, yet post-task reflections reveal both appreciation for collaboration and challenges in integration, including limited conversational coordination.
>
---
#### [replaced 058] Quantifying Label-Induced Bias in Large Language Model Self- and Cross-Evaluations
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.21164v3](http://arxiv.org/pdf/2508.21164v3)**

> **作者:** Muskan Saraf; Sajjad Rezvani Boroujeni; Justin Beaudry; Hossein Abedi; Tom Bush
>
> **摘要:** Large language models (LLMs) are increasingly deployed as evaluators of text quality, yet the validity of their judgments remains underexplored. This study investigates systematic bias in self- and cross-model evaluations across three prominent LLMs: ChatGPT, Gemini, and Claude. We designed a controlled experiment in which blog posts authored by each model were evaluated by all three models under four labeling conditions: no attribution, true attribution, and two false-attribution scenarios. Evaluations employed both holistic preference voting and granular quality ratings across three dimensions Coherence, Informativeness, and Conciseness with all scores normalized to percentages for direct comparison. Our findings reveal pronounced asymmetries in model judgments: the "Claude" label consistently elevated scores regardless of actual authorship, while the "Gemini" label systematically depressed them. False attribution frequently reversed preference rankings, producing shifts of up to 50 percentage points in voting outcomes and up to 12 percentage points in quality ratings. Notably, Gemini exhibited severe self-deprecation under true labels, while Claude demonstrated intensified self-preference. These results demonstrate that perceived model identity can substantially distort both high-level judgments and fine-grained quality assessments, independent of content quality. Our findings challenge the reliability of LLM-as-judge paradigms and underscore the critical need for blind evaluation protocols and diverse multi-model validation frameworks to ensure fairness and validity in automated text evaluation and LLM benchmarking.
>
---
#### [replaced 059] LaTeXTrans: Structured LaTeX Translation with Multi-Agent Coordination
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.18791v2](http://arxiv.org/pdf/2508.18791v2)**

> **作者:** Ziming Zhu; Chenglong Wang; Shunjie Xing; Yifu Huo; Fengning Tian; Quan Du; Di Yang; Chunliang Zhang; Tong Xiao; Jingbo Zhu
>
> **摘要:** Despite the remarkable progress of modern machine translation (MT) systems on general-domain texts, translating structured LaTeX-formatted documents remains a significant challenge. These documents typically interleave natural language with domain-specific syntax, such as mathematical equations, tables, figures, and cross-references, all of which must be accurately preserved to maintain semantic integrity and compilability. In this paper, we introduce LaTeXTrans, a collaborative multi-agent system designed to address this challenge. LaTeXTrans ensures format preservation, structural fidelity, and terminology consistency through six specialized agents: 1) a Parser that decomposes LaTeX into translation-friendly units via placeholder substitution and syntax filtering; 2) a Translator, Validator, Summarizer, and Terminology Extractor that work collaboratively to ensure context-aware, self-correcting, and terminology-consistent translations; 3) a Generator that reconstructs the translated content into well-structured LaTeX documents. Experimental results demonstrate that LaTeXTrans can outperform mainstream MT systems in both translation accuracy and structural fidelity, offering an effective and practical solution for translating LaTeX-formatted documents.The code of LaTeXTrans is available at https://github.com/NiuTrans/LaTeXTrans.
>
---
#### [replaced 060] Reasoning through Exploration: A Reinforcement Learning Framework for Robust Function Calling
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.05118v4](http://arxiv.org/pdf/2508.05118v4)**

> **作者:** Bingguang Hao; Zengzhuang Xu; Maolin Wang; Yuntao Wen; Yicheng Chen; Cunyin Peng; Long Chen; Dong Wang; Xiangyu Zhao; Jinjie Gu; Chenyi Zhuang; Ji Zhang
>
> **摘要:** The effective training of Large Language Models (LLMs) for function calling faces a critical challenge: balancing exploration of complex reasoning paths with stable policy optimization. Standard methods like Supervised Fine-Tuning (SFT) fail to instill robust reasoning, and traditional Reinforcement Learning (RL) struggles with inefficient exploration. We propose \textbf{EGPO}, a new RL framework built upon Group Relative Policy Optimization (GRPO), designed to address this challenge directly. The core of EGPO is an entropy-enhanced advantage function that integrates the entropy of the model's Chain-of-Thought (CoT) into the policy gradient computation. This encourages the generation of diverse reasoning strategies. To maintain optimization direction, the entropy bonus is carefully constrained by a clipping mechanism. Complemented by a strict, binary reward signal, EGPO effectively guides the model towards discovering structured and accurate tool invocation patterns. On the challenging Berkeley Function Calling Leaderboard (BFCL), a 4B-parameter model trained with EGPO sets a new state-of-the-art among models of comparable size, surpassing a range of strong competitors, including GPT-4o and Gemini-2.5.
>
---
#### [replaced 061] Language Model Guided Reinforcement Learning in Quantitative Trading
- **分类: cs.LG; cs.CL; q-fin.TR; I.2.7; I.2.6; J.4**

- **链接: [http://arxiv.org/pdf/2508.02366v2](http://arxiv.org/pdf/2508.02366v2)**

> **作者:** Adam Darmanin; Vince Vella
>
> **备注:** 12 pages (4 pages appendix and references) and 6 figures. Accepted for presentation at FLLM 2025, Vienna
>
> **摘要:** Algorithmic trading requires short-term tactical decisions consistent with long-term financial objectives. Reinforcement Learning (RL) has been applied to such problems, but adoption is limited by myopic behaviour and opaque policies. Large Language Models (LLMs) offer complementary strategic reasoning and multi-modal signal interpretation when guided by well-structured prompts. This paper proposes a hybrid framework in which LLMs generate high-level trading strategies to guide RL agents. We evaluate (i) the economic rationale of LLM-generated strategies through expert review, and (ii) the performance of LLM-guided agents against unguided RL baselines using Sharpe Ratio (SR) and Maximum Drawdown (MDD). Empirical results indicate that LLM guidance improves both return and risk metrics relative to standard RL.
>
---
#### [replaced 062] Machine Learning for Detection and Analysis of Novel LLM Jailbreaks
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2510.01644v2](http://arxiv.org/pdf/2510.01644v2)**

> **作者:** John Hawkins; Aditya Pramar; Rodney Beard; Rohitash Chandra
>
> **摘要:** Large Language Models (LLMs) suffer from a range of vulnerabilities that allow malicious users to solicit undesirable responses through manipulation of the input text. These so-called jailbreak prompts are designed to trick the LLM into circumventing the safety guardrails put in place to keep responses acceptable to the developer's policies. In this study, we analyse the ability of different machine learning models to distinguish jailbreak prompts from genuine uses, including looking at our ability to identify jailbreaks that use previously unseen strategies. Our results indicate that using current datasets the best performance is achieved by fine tuning a Bidirectional Encoder Representations from Transformers (BERT) model end-to-end for identifying jailbreaks. We visualise the keywords that distinguish jailbreak from genuine prompts and conclude that explicit reflexivity in prompt structure could be a signal of jailbreak intention.
>
---
#### [replaced 063] Filling in the Clinical Gaps in Benchmark: Case for HealthBench for the Japanese medical system
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.17444v2](http://arxiv.org/pdf/2509.17444v2)**

> **作者:** Shohei Hisada; Endo Sunao; Himi Yamato; Shoko Wakamiya; Eiji Aramaki
>
> **备注:** draft v0.2
>
> **摘要:** This study investigates the applicability of HealthBench, a large-scale, rubric-based medical benchmark, to the Japanese context. Although robust evaluation frameworks are essential for the safe development of medical LLMs, resources in Japanese are scarce and often consist of translated multiple-choice questions. Our research addresses this issue in two ways. First, we establish a performance baseline by applying a machine-translated version of HealthBench's 5,000 scenarios to evaluate two models: a high-performing multilingual model (GPT-4.1) and a Japanese-native open-source model (LLM-jp-3.1). Secondly, we use an LLM-as-a-Judge approach to systematically classify the benchmark's scenarios and rubric criteria. This allows us to identify 'contextual gaps' where the content is misaligned with Japan's clinical guidelines, healthcare systems or cultural norms. Our findings reveal a modest performance drop in GPT-4.1 due to rubric mismatches, as well as a significant failure in the Japanese-native model, which lacked the required clinical completeness. Furthermore, our classification shows that, despite most scenarios being applicable, a significant proportion of the rubric criteria require localisation. This work underscores the limitations of direct benchmark translation and highlights the urgent need for a context-aware, localised adaptation, a "J-HealthBench", to ensure the reliable and safe evaluation of medical LLMs in Japan.
>
---
#### [replaced 064] AnyEdit: Edit Any Knowledge Encoded in Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.05628v3](http://arxiv.org/pdf/2502.05628v3)**

> **作者:** Houcheng Jiang; Junfeng Fang; Ningyu Zhang; Guojun Ma; Mingyang Wan; Xiang Wang; Xiangnan He; Tat-seng Chua
>
> **摘要:** Large language models (LLMs) often produce incorrect or outdated information, necessitating efficient and precise knowledge updates. Current model editing methods, however, struggle with long-form knowledge in diverse formats, such as poetry, code snippets, and mathematical derivations. These limitations arise from their reliance on editing a single token's hidden state, a limitation we term "efficacy barrier". To solve this, we propose AnyEdit, a new autoregressive editing paradigm. It decomposes long-form knowledge into sequential chunks and iteratively edits the key token in each chunk, ensuring consistent and accurate outputs. Theoretically, we ground AnyEdit in the Chain Rule of Mutual Information, showing its ability to update any knowledge within LLMs. Empirically, it outperforms strong baselines by 21.5% on benchmarks including UnKEBench, AKEW, and our new EditEverything dataset for long-form diverse-formatted knowledge. Additionally, AnyEdit serves as a plug-and-play framework, enabling current editing methods to update knowledge with arbitrary length and format, significantly advancing the scope and practicality of LLM knowledge editing.
>
---
#### [replaced 065] Mem4Nav: Boosting Vision-and-Language Navigation in Urban Environments with a Hierarchical Spatial-Cognition Long-Short Memory System
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.19433v2](http://arxiv.org/pdf/2506.19433v2)**

> **作者:** Lixuan He; Haoyu Dong; Zhenxing Chen; Yangcheng Yu; Jie Feng; Yong Li
>
> **备注:** The paper is currently under investigation regarding concerns of potential academic misconduct. While the investigation is ongoing, the authors have voluntarily requested to withdraw the manuscript
>
> **摘要:** Vision-and-Language Navigation (VLN) in large-scale urban environments requires embodied agents to ground linguistic instructions in complex scenes and recall relevant experiences over extended time horizons. Prior modular pipelines offer interpretability but lack unified memory, while end-to-end (M)LLM agents excel at fusing vision and language yet remain constrained by fixed context windows and implicit spatial reasoning. We introduce \textbf{Mem4Nav}, a hierarchical spatial-cognition long-short memory system that can augment any VLN backbone. Mem4Nav fuses a sparse octree for fine-grained voxel indexing with a semantic topology graph for high-level landmark connectivity, storing both in trainable memory tokens embedded via a reversible Transformer. Long-term memory (LTM) compresses and retains historical observations at both octree and graph nodes, while short-term memory (STM) caches recent multimodal entries in relative coordinates for real-time obstacle avoidance and local planning. At each step, STM retrieval sharply prunes dynamic context, and, when deeper history is needed, LTM tokens are decoded losslessly to reconstruct past embeddings. Evaluated on Touchdown and Map2Seq across three backbones (modular, state-of-the-art VLN with prompt-based LLM, and state-of-the-art VLN with strided-attention MLLM), Mem4Nav yields 7-13 pp gains in Task Completion, sufficient SPD reduction, and >10 pp nDTW improvement. Ablations confirm the indispensability of both the hierarchical map and dual memory modules. Our codes are open-sourced via https://github.com/tsinghua-fib-lab/Mem4Nav.
>
---
#### [replaced 066] Elicit and Enhance: Advancing Multimodal Reasoning in Medical Scenarios
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23118v2](http://arxiv.org/pdf/2505.23118v2)**

> **作者:** Zhongzhen Huang; Linjie Mu; Yakun Zhu; Xiangyu Zhao; Shaoting Zhang; Xiaofan Zhang
>
> **摘要:** Effective clinical decision-making depends on iterative, multimodal reasoning across diverse sources of evidence. The recent emergence of multimodal reasoning models has significantly transformed the landscape of solving complex tasks. Although such models have achieved notable success in mathematics and science, their application to medical domains remains underexplored. In this work, we propose \textit{MedE$^2$}, a two-stage post-training pipeline that elicits and then enhances multimodal reasoning for medical domains. In Stage-I, we fine-tune models using 2,000 text-only data samples containing precisely orchestrated reasoning demonstrations to elicit reasoning behaviors. In Stage-II, we further enhance the model's reasoning capabilities using 1,500 rigorously curated multimodal medical cases, aligning model reasoning outputs with our proposed multimodal medical reasoning preference. Extensive experiments demonstrate the efficacy and reliability of \textit{MedE$^2$} in improving the reasoning performance of medical multimodal models. Notably, models trained with \textit{MedE$^2$} consistently outperform baselines across multiple medical multimodal benchmarks. Additional validation on larger models and under inference-time scaling further confirms the robustness and practical utility of our approach.
>
---
#### [replaced 067] Tuning without Peeking: Provable Privacy and Generalization Bounds for LLM Post-Training
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2507.01752v2](http://arxiv.org/pdf/2507.01752v2)**

> **作者:** Ismail Labiad; Mathurin Videau; Matthieu Kowalski; Marc Schoenauer; Alessandro Leite; Julia Kempe; Olivier Teytaud
>
> **摘要:** Gradient-based optimization is the workhorse of deep learning, offering efficient and scalable training via backpropagation. However, exposing gradients during training can leak sensitive information about the underlying data, raising privacy and security concerns such as susceptibility to data poisoning attacks. In contrast, black box optimization methods, which treat the model as an opaque function, relying solely on function evaluations to guide optimization, offer a promising alternative in scenarios where data access is restricted, adversarial risks are high, or overfitting is a concern. This paper introduces BBoxER, an evolutionary black-box method for LLM post-training that induces an information bottleneck via implicit compression of the training data. Leveraging the tractability of information flow, we provide non-vacuous generalization bounds and strong theoretical guarantees for differential privacy, robustness to data poisoning attacks, and extraction attacks. In experiments with LLMs, we demonstrate empirically that black-box optimization methods-despite the scalability and computational challenges inherent to black-box approaches-are able to learn, showing how a few iterations of BBoxER improve performance, generalize well on a benchmark of reasoning datasets, and are robust to membership inference attacks. This positions BBoxER as an attractive add-on on top of gradient-based optimization, offering suitability for deployment in restricted or privacy-sensitive environments while also providing non-vacuous generalization guarantees.
>
---
#### [replaced 068] Direct Quantized Training of Language Models with Stochastic Rounding
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.04787v3](http://arxiv.org/pdf/2412.04787v3)**

> **作者:** Kaiyan Zhao; Tsuguchika Tabaru; Kenichi Kobayashi; Takumi Honda; Masafumi Yamazaki; Yoshimasa Tsuruoka
>
> **备注:** Accepted to ACML 2025
>
> **摘要:** Although recent quantized Large Language Models (LLMs), such as BitNet, have paved the way for significant reduction in memory usage during deployment with binary or ternary weights, training these models still demands substantial memory footprints. This is partly because high-precision (i.e., unquantized) weights required for straight-through estimation must be maintained throughout the whole training process. To address this, we explore directly updating the quantized low-precision weights without relying on straight-through estimation during backpropagation, aiming to save memory usage during training. Specifically, we employ a stochastic rounding technique to minimize the information loss caused by the use of low-bit weights throughout training. Experimental results on our LLaMA-structured models of various sizes indicate that (1) training with only low-precision weights is feasible even when they are constrained to ternary values; (2) extending the bit width to 8 bits achieves performance on par with BitNet b1.58; (3) our models remain robust to precision scaling and memory reduction, showing minimal performance degradation when moving from FP32 to lower-memory environments (BF16/FP8); and (4) our models also support inference using ternary weights, showcasing their flexibility in deployment.
>
---
#### [replaced 069] InfiR2: A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.22536v3](http://arxiv.org/pdf/2509.22536v3)**

> **作者:** Wenjun Wang; Shuo Cai; Congkai Xie; Mingfa Feng; Yiming Zhang; Zhen Li; Kejing Yang; Ming Li; Jiannong Cao; Hongxia Yang
>
> **备注:** This paper has been withdrawn by the authors due to a significant bug discovered in our data processing pipeline. This bug affects the validity of the experimental results, and we can no longer stand by the conclusions presented
>
> **摘要:** The immense computational cost of training Large Language Models (LLMs) presents a major barrier to innovation. While FP8 training offers a promising solution with significant theoretical efficiency gains, its widespread adoption has been hindered by the lack of a comprehensive, open-source training recipe. To bridge this gap, we introduce an end-to-end FP8 training recipe that seamlessly integrates continual pre-training and supervised fine-tuning. Our methodology employs a fine-grained, hybrid-granularity quantization strategy to maintain numerical fidelity while maximizing computational efficiency. Through extensive experiments, including the continue pre-training of models on a 160B-token corpus, we demonstrate that our recipe is not only remarkably stable but also essentially lossless, achieving performance on par with the BF16 baseline across a suite of reasoning benchmarks. Crucially, this is achieved with substantial efficiency improvements, including up to a 22% reduction in training time, a 14% decrease in peak memory usage, and a 19% increase in throughput. Our results establish FP8 as a practical and robust alternative to BF16, and we will release the accompanying code to further democratize large-scale model training.
>
---
#### [replaced 070] CFDLLMBench: A Benchmark Suite for Evaluating Large Language Models in Computational Fluid Dynamics
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.20374v2](http://arxiv.org/pdf/2509.20374v2)**

> **作者:** Nithin Somasekharan; Ling Yue; Yadi Cao; Weichao Li; Patrick Emami; Pochinapeddi Sai Bhargav; Anurag Acharya; Xingyu Xie; Shaowu Pan
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong performance across general NLP tasks, but their utility in automating numerical experiments of complex physical system -- a critical and labor-intensive component -- remains underexplored. As the major workhorse of computational science over the past decades, Computational Fluid Dynamics (CFD) offers a uniquely challenging testbed for evaluating the scientific capabilities of LLMs. We introduce CFDLLMBench, a benchmark suite comprising three complementary components -- CFDQuery, CFDCodeBench, and FoamBench -- designed to holistically evaluate LLM performance across three key competencies: graduate-level CFD knowledge, numerical and physical reasoning of CFD, and context-dependent implementation of CFD workflows. Grounded in real-world CFD practices, our benchmark combines a detailed task taxonomy with a rigorous evaluation framework to deliver reproducible results and quantify LLM performance across code executability, solution accuracy, and numerical convergence behavior. CFDLLMBench establishes a solid foundation for the development and evaluation of LLM-driven automation of numerical experiments for complex physical systems. Code and data are available at https://github.com/NREL-Theseus/cfdllmbench/.
>
---
#### [replaced 071] RedDebate: Safer Responses through Multi-Agent Red Teaming Debates
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11083v2](http://arxiv.org/pdf/2506.11083v2)**

> **作者:** Ali Asad; Stephen Obadinma; Radin Shayanfar; Xiaodan Zhu
>
> **摘要:** We introduce RedDebate, a novel multi-agent debate framework that provides the foundation for Large Language Models (LLMs) to identify and mitigate their unsafe behaviours. Existing AI safety approaches often rely on costly human evaluation or isolated single-model assessment, both constrained by scalability and prone to oversight failures. RedDebate employs collaborative argumentation among multiple LLMs across diverse debate scenarios, enabling them to critically evaluate one another's reasoning and systematically uncover unsafe failure modes through fully automated red-teaming. We further integrate distinct long-term memory modules that preserve safety-relevant insights from debate interactions and leverage them during subsequent inference, facilitating continuous refinement of model behaviour. Empirical evaluation on safety benchmarks across a diverse set of models demonstrates that RedDebate substantially reduces unsafe outputs. While debate alone allows LLMs to refine their behaviour, the addition of memory yields further significant reductions. To the best of our knowledge, RedDebate is the first fully automated framework to unify multi-agent debate and red-teaming to progressively enhance LLM safety without human intervention.
>
---
#### [replaced 072] Haystack Engineering: Context Engineering for Heterogeneous and Agentic Long-Context Evaluation
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2510.07414v2](http://arxiv.org/pdf/2510.07414v2)**

> **作者:** Mufei Li; Dongqi Fu; Limei Wang; Si Zhang; Hanqing Zeng; Kaan Sancak; Ruizhong Qiu; Haoyu Wang; Xiaoxin He; Xavier Bresson; Yinglong Xia; Chonglin Sun; Pan Li
>
> **备注:** Code available at https://github.com/Graph-COM/HaystackCraft
>
> **摘要:** Modern long-context large language models (LLMs) perform well on synthetic "needle-in-a-haystack" (NIAH) benchmarks, but such tests overlook how noisy contexts arise from biased retrieval and agentic workflows. We argue that haystack engineering is necessary to construct noisy long contexts that faithfully capture key real-world factors -- distraction from heterogeneous biased retrievers and cascading errors in agentic workflows -- to test models' long-context robustness. We instantiate it through HaystackCraft, a new NIAH benchmark built on the full English Wikipedia hyperlink network with multi-hop questions. HaystackCraft evaluates how heterogeneous retrieval strategies (e.g., sparse, dense, hybrid, and graph-based) affect distractor composition, haystack ordering, and downstream LLM performance. HaystackCraft further extends NIAH to dynamic, LLM-dependent settings that simulate agentic operations, where models refine queries, reflect on their past reasonings, and decide when to stop. Experiments with 15 long-context models show that (1) while stronger dense retrievers can introduce more challenging distractors, graph-based reranking simultaneously improves retrieval effectiveness and mitigates more harmful distractors; (2) in agentic tests, even advanced models like Gemini 2.5 Pro and GPT-5 suffer cascading failures from self-generated distractors or struggle to perform early stops. These results highlight persistent challenges in agentic long-context reasoning and establish HaystackCraft as a valuable testbed for future progress.
>
---
#### [replaced 073] Phonikud: Hebrew Grapheme-to-Phoneme Conversion for Real-Time Text-to-Speech
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.12311v2](http://arxiv.org/pdf/2506.12311v2)**

> **作者:** Yakov Kolani; Maxim Melichov; Cobi Calev; Morris Alper
>
> **备注:** Project page: https://phonikud.github.io
>
> **摘要:** Real-time text-to-speech (TTS) for Modern Hebrew is challenging due to the language's orthographic complexity. Existing solutions ignore crucial phonetic features such as stress that remain underspecified even when vowel marks are added. To address these limitations, we introduce Phonikud, a lightweight, open-source Hebrew grapheme-to-phoneme (G2P) system that outputs fully-specified IPA transcriptions. Our approach adapts an existing diacritization model with lightweight adaptors, incurring negligible additional latency. We also contribute the ILSpeech dataset of transcribed Hebrew speech with IPA annotations, serving as a benchmark for Hebrew G2P, as training data for TTS systems, and enabling audio-to-IPA for evaluating TTS performance while capturing important phonetic details. Our results demonstrate that Phonikud G2P conversion more accurately predicts phonemes from Hebrew text compared to prior methods, and that this enables training of effective real-time Hebrew TTS models with superior speed-accuracy trade-offs. We release our code, data, and models at https: //phonikud.github.io.
>
---
#### [replaced 074] Preprint: Poster: Did I Just Browse A Website Written by LLMs?
- **分类: cs.NI; cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2507.13933v2](http://arxiv.org/pdf/2507.13933v2)**

> **作者:** Sichang Steven He; Ramesh Govindan; Harsha V. Madhyastha
>
> **备注:** ACM Internet Measurement Conference 2025 Poster & ACM IMC 2025 Student Workshop. 2 pages. 3 figures
>
> **摘要:** Increasingly, web content is automatically generated by large language models (LLMs) with little human input. We call this "LLM-dominant" content. Since LLMs plagiarize and hallucinate, LLM-dominant content can be unreliable and unethical. Yet, websites rarely disclose such content, and human readers struggle to distinguish it. Thus, we must develop reliable detectors for LLM-dominant content. However, state-of-the-art LLM detectors are inaccurate on web content, because web content has low positive rates, complex markup, and diverse genres, instead of clean, prose-like benchmark data SoTA detectors are optimized for. We propose a highly reliable, scalable pipeline that classifies entire websites. Instead of naively classifying text extracted from each page, we classify each site based on an LLM text detector's outputs of multiple prose-like pages to boost accuracies. We train and evaluate our detector by collecting 2 distinct ground truth datasets totaling 120 sites, and obtain 100% accuracies testing across them. In the wild, we detect a sizable portion of sites as LLM-dominant among 10k sites in search engine results and 10k in Common Crawl archives. We find LLM-dominant sites are growing in prevalence and rank highly in search results, raising questions about their impact on end users and the overall Web ecosystem.
>
---
#### [replaced 075] Can Large Language Models Master Complex Card Games?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.01328v3](http://arxiv.org/pdf/2509.01328v3)**

> **作者:** Wei Wang; Fuqing Bie; Junzhe Chen; Dan Zhang; Shiyu Huang; Evgeny Kharlamov; Jie Tang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Complex games have long been an important benchmark for testing the progress of artificial intelligence algorithms. AlphaGo, AlphaZero, and MuZero have defeated top human players in Go and Chess, garnering widespread societal attention towards artificial intelligence. Concurrently, large language models (LLMs) have exhibited remarkable capabilities across various tasks, raising the question of whether LLMs can achieve similar success in complex games. In this paper, we explore the potential of LLMs in mastering complex card games. We systematically assess the learning capabilities of LLMs across eight diverse card games, evaluating the impact of fine-tuning on high-quality gameplay data, and examining the models' ability to retain general capabilities while mastering these games. Our findings indicate that: (1) LLMs can approach the performance of strong game AIs through supervised fine-tuning on high-quality data, (2) LLMs can achieve a certain level of proficiency in multiple complex card games simultaneously, with performance augmentation for games with similar rules and conflicts for dissimilar ones, and (3) LLMs experience a decline in general capabilities when mastering complex games, but this decline can be mitigated by integrating a certain amount of general instruction data. The evaluation results demonstrate strong learning ability and versatility of LLMs. The code is available at https://github.com/THUDM/LLM4CardGame
>
---
#### [replaced 076] Taxonomy of User Needs and Actions
- **分类: cs.HC; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.06124v2](http://arxiv.org/pdf/2510.06124v2)**

> **作者:** Renee Shelby; Fernando Diaz; Vinodkumar Prabhakaran
>
> **摘要:** The growing ubiquity of conversational AI highlights the need for frameworks that capture not only users' instrumental goals but also the situated, adaptive, and social practices through which they achieve them. Existing taxonomies of conversational behavior either overgeneralize, remain domain-specific, or reduce interactions to narrow dialogue functions. To address this gap, we introduce the Taxonomy of User Needs and Actions (TUNA), an empirically grounded framework developed through iterative qualitative analysis of 1193 human-AI conversations, supplemented by theoretical review and validation across diverse contexts. TUNA organizes user actions into a three-level hierarchy encompassing behaviors associated with information seeking, synthesis, procedural guidance, content creation, social interaction, and meta-conversation. By centering user agency and appropriation practices, TUNA enables multi-scale evaluation, supports policy harmonization across products, and provides a backbone for layering domain-specific taxonomies. This work contributes a systematic vocabulary for describing AI use, advancing both scholarly understanding and practical design of safer, more responsive, and more accountable conversational systems.
>
---
