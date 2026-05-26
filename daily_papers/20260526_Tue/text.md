# 自然语言处理 cs.CL

- **最新发布 225 篇**

- **更新 135 篇**

## 最新发布

#### [new 001] A Lightweight Hybrid Transformer-CRF Architecture for Multi-Type Bangla Medical Entity Recognition
- **分类: cs.CL**

- **简介: 该论文属于医学实体识别任务，旨在解决资源受限环境下高效模型部署问题。通过知识蒸馏和量化压缩，提出轻量级混合Transformer-CRF架构。**

- **链接: [https://arxiv.org/pdf/2605.25463](https://arxiv.org/pdf/2605.25463)**

> **作者:** Peyal Saha; Ahsanul Haque Hasib; Shoumik Barman Polok
>
> **摘要:** MedER refers to the identification of medical entities. It is crucial for extracting structured clinical information from unstructured medical text. Many existing systems rely on transformer-based models, which are computationally expensive and difficult to deploy in resource-constrained environments. Furthermore, earlier works often use relaxed evaluation metrics that artificially inflate performance by rewarding correct prediction of dominant "Outside" (O) tokens. In this paper, we propose a lightweight Medical Entity Recognition (MedER) framework for the Bangla language. We establish a rigorous baseline using a 12-layer BanglaBERT model combined with a Conditional Random Field (CRF) layer for exact-boundary entity detection. To address deployment constraints, we compress this teacher model into a 4-layer student network through Knowledge Distillation (KD), where the student learns from the teacher's pre-CRF soft emission logits. Finally, we apply INT8 dynamic quantization to further reduce model size and inference cost. Our final quantized student achieves an 8.6x CPU speedup while requiring nearly 48 percent less storage than the CRF teacher model.
>
---
#### [new 002] How Much Structure Do LLMs Need? Evaluating LLMs for Bibliometric Cluster Description
- **分类: cs.CL**

- **简介: 该论文属于信息科学任务，旨在解决LLMs在文献合成中的结构依赖问题。通过对比不同流程，评估结构对LLM生成聚类描述的影响。**

- **链接: [https://arxiv.org/pdf/2605.24351](https://arxiv.org/pdf/2605.24351)**

> **作者:** Abraham Camelo-Guerrero; Jairo Diaz-Rodriguez
>
> **摘要:** Large language models (LLMs) can support scientific literature synthesis, but remain prone to hallucinated references, uneven coverage, and weakly grounded thematic organization. We evaluate whether bibliometric structure improves LLM-assisted synthesis by comparing six pipelines for generating cluster descriptions under different levels of evidence and structure. Using 100 published bibliometric analyses, we reconstruct Scopus corpora, extract human-written cluster descriptions, and assess outputs by human alignment, semantic coverage, clustering quality, graph quality, and reference grounding. Results show that LLMs produce descriptions semantically close to human-written ones, but are unreliable when asked to infer bibliometric structure from scratch. Performance improves when bibliometric algorithms define the clusters and the LLM interprets them. Overall, LLM-assisted bibliometric synthesis is most promising as a hybrid workflow in which algorithms provide auditable structure and LLMs generate readable descriptions.
>
---
#### [new 003] Overview of the PsyDefDetect Shared Task at BioNLP 2026: Detecting Levels of Psychological Defense Mechanisms in Supportive Conversations
- **分类: cs.CL**

- **简介: 该论文介绍PsyDefDetect任务，旨在检测支持性对话中心理防御机制的层级。通过分析对话内容，分类为九个类别，推动临床心理学与NLP的结合。**

- **链接: [https://arxiv.org/pdf/2605.24907](https://arxiv.org/pdf/2605.24907)**

> **作者:** Hongbin Na; Zimu Wang; Zhaoming Chen; Yining Hua; Rena Gao; Kailai Yang; Ling Chen; Wei Wang; Shaoxiong Ji; John Torous; Sophia Ananiadou
>
> **摘要:** We present an overview of PsyDefDetect, the shared task on detecting levels of psychological defense mechanisms in emotional support dialogues, co-located with BioNLP@ACL 2026. Grounded in the clinically validated Defense Mechanism Rating Scales (DMRS) framework, the task asks systems to classify a target seeker utterance, given its preceding dialogue context, into one of nine categories: seven hierarchical DMRS levels plus two auxiliary labels. Participants worked on PsyDefConv, a newly released corpus of 200 dialogues and 2336 help-seeker utterances annotated under DMRS with substantial inter-annotator agreement. The task attracted 172 participants on CodaBench who produced 563 submissions, with 21 teams officially registering their results for the final ranking. The best system achieved a macro F1-score of 0.420, surpassing the strongest fine-tuned baseline reported in the dataset paper by a notable margin, yet leaving clear headroom. Our analysis highlights (i) a persistent tendency to over-predict the majority High-Adaptive class, (ii) a widening gap between accuracy and macro-F1 that reveals class-imbalance sensitivity, and (iii) the value of theory-aware and LLM-based approaches for fine-grained defensive-function classification. We release all task materials and invite the community to continue work on this novel intersection of clinical psychology and NLP.
>
---
#### [new 004] Phonetic Modeling of Dialectal Variation in Vietnamese Speech
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决越南语方言发音差异带来的识别难题。通过构建方言感知的语音模型，有效捕捉不同地区的语音特征。**

- **链接: [https://arxiv.org/pdf/2605.24451](https://arxiv.org/pdf/2605.24451)**

> **作者:** Quan Ngoc Hoang; Long Hoang Huu Nguyen; Nghia Hieu Nguyen; Kiet Van Nguyen; Ngan Luu-Thuy Nguyen
>
> **摘要:** Vietnamese exhibits substantial dialectal phonetic variation across Northern, Central, and Southern regions, where identical lexical items may be realized with markedly different pronunciations. Such variation poses challenges for automatic speech recognition (ASR) and remains difficult to model computationally due to the complex relationship between Vietnamese orthography and phonology. Existing approaches typically address dialect variability at the word level, assuming dialect-invariant mappings between spelling and pronunciation, which limits their ability to capture systematic phonetic differences. We propose a dialect-aware phonetic framework that explicitly models Vietnamese phonological structure and dialectal variation at both the vocabulary and decoding levels. The framework introduces a phonetic vocabulary that decomposes each syllable into structured phonetic components and maps them to dialect-specific IPA representations, together with a phonetic-structure decoder that jointly predicts these components. Experiments on the UIT-ViMD, a only-available dataset for multi-dialect in Vietnamese, show that the proposed approach outperforms various pre-trained baselines, \textbf{especially matches the performance of the strongest pretrained wav2ve2-base-vi-250h} across dialects while \textbf{using substantially fewer parameters and no external pretraining}. Code for experimental reproducibility will be publicly available upon the acceptance of this paper.
>
---
#### [new 005] CUNY at CLPsych 2026: A Pipeline Approach to Classification and Summarization of Mental Health Changes
- **分类: cs.CL**

- **简介: 该论文参与CLPsych 2026任务，旨在通过社交媒体时间线分析心理状态变化。工作包括集成学习分类、预测变化点及总结情绪动态，取得优异成绩。**

- **链接: [https://arxiv.org/pdf/2605.24164](https://arxiv.org/pdf/2605.24164)**

> **作者:** Amirmohammad Ziaei Bideh; Shameed Charlomar Job; Ava Yahyapour; Alla Rozovskaya
>
> **摘要:** We describe our submission to the CLPsych~2026 Shared Task on capturing and characterizing mental health changes through social media timeline dynamics. To infer the dominant self-states in posts (Tasks 1.1 and 1.2), we ensemble in-context learning of three open-weight large language models using majority voting. For predicting moments of change in a timeline (Task~2), we train supervised classifiers on features derived from Task~1.1 predictions. To summarize the patterns of mood dynamics and their progression over time within a timeline (Task 3.1), we augment in-context example labels predicted by upstream systems (Tasks 1.1, 1.2, and 2), yielding performance gains over zero-shot and unaugmented in-context learning baselines. Our submission ranked first on Task~1.1, fourth on Task~1.2, fourth on Task~2, and third on Task~3.1.\footnote{The source code for the experiments is available at this https URL
>
---
#### [new 006] Multi-Persona Debate System for Automated Scientific Hypothesis Generation
- **分类: cs.CL**

- **简介: 该论文属于科学假设生成任务，旨在解决多领域知识整合难题。提出MPDS系统，通过文献检索、角色辩论等方法生成高质量假设。**

- **链接: [https://arxiv.org/pdf/2605.23917](https://arxiv.org/pdf/2605.23917)**

> **作者:** Jaeha Oh; Byungchan Kim; Ju Li; Yang Jeong Park; Jin-Sung Park
>
> **备注:** 31 pages with 7 main figures, 4 supplementary figures and 1 supplementary table
>
> **摘要:** Modern scientific discovery is bottlenecked not by data scarcity, but by the inability to synthesize fragmented knowledge into actionable hypotheses. This challenge is especially acute in battery materials research, where electrochemical performance, interfacial behavior, and manufacturing feasibility must be optimized simultaneously. Here, we present the Multi-Persona Debate System (MPDS), a literature-grounded framework for automated scientific hypothesis generation that combines literature retrieval, long-context large language model reasoning, corpus-driven persona induction, and structured multi-agent debate. MPDS constructs literature snapshots of up to 500 papers, grounds agents in role-specific evidence pools, and conducts a three-round citation-aware debate followed by moderator synthesis, enabling negotiation between personas while preserving evidence traceability. We evaluate MPDS using a temporally controlled protocol excluding direct access to target papers, including two held-out battery-materials case studies and a blinded comparison across 30 matched cases. In sodium-ion anode and all-solid-state battery cathode design tasks, MPDS recovered design logics aligned with experimentally validated solution spaces and generated more mechanistically explicit, process-aware proposals than simpler baselines. To assess the impact of personas and debate, we introduce Integrative Hypothesis Quality scoring. In ablation studies, MPDS achieved the highest mean score among five conditions, with its largest advantage in cross-perspective integration. A laboratory follow-up suggests utility as a diagnostic aid for identifying practical bottlenecks in workflows. These results indicate that structured debate over literature snapshots improves hypothesis formation under coupled engineering constraints and provides a reusable workflow for text-intensive scientific discovery.
>
---
#### [new 007] Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki
- **分类: cs.CL**

- **简介: 该论文属于知识检索任务，解决LLM代理检索效率与推理能力不足的问题。提出LLM-Wiki系统，将知识编译为可操作结构，提升多步骤推理性能。**

- **链接: [https://arxiv.org/pdf/2605.25480](https://arxiv.org/pdf/2605.25480)**

> **作者:** Haoliang Ming; Feifei Li; Xiaoqing Wu; Wenhui Que
>
> **备注:** 15 pages, 3 figures, 10 tables, 1 algorithm
>
> **摘要:** LLM agents require retrieval to behave less like one-shot context fetching and more like reasoning: searching, reading, traversing, and deciding when evidence is sufficient. However, Retrieval-Augmented Generation (RAG) typically organizes external knowledge as flat chunks retrieved by embedding similarity, exposing a retrieval-as-lookup interface that is poorly aligned with tool-using agents. We propose LLM-Wiki, an agent-native retrieval system that operationalizes the Retrieval-as-Reasoning paradigm by treating external knowledge as a compilable, composable, and self-evolving structure rather than a static retrieval index. LLM-Wiki compiles documents into structured Wiki pages with bidirectional links, exposes search, read, and link-following operations through standard tool-calling interfaces, and introduces an Error Book for persistent structural and semantic self-correction. On HotpotQA, MuSiQue, and 2WikiMultiHopQA, LLM-Wiki outperforms seven baselines, including HippoRAG 2, LightRAG, and GraphRAG, with gains of 2.0-8.1 F1 points over the strongest graph-based baseline and larger gains over Dense RAG. On AuthTrace, LLM-Wiki achieves the best overall accuracy, with especially strong gains on multi-document structured queries, showing that compilation-based knowledge organization generalizes beyond chain-style multi-hop reasoning.
>
---
#### [new 008] Llamion Technical Report
- **分类: cs.CL**

- **简介: 该论文介绍Llamion，一个通过KEPT方法将Orion-14B转换为Llama架构的模型，解决模型迁移中的知识保留问题。任务为模型架构转换与知识保持。**

- **链接: [https://arxiv.org/pdf/2605.25676](https://arxiv.org/pdf/2605.25676)**

> **作者:** Kisu Yang; Yoonna Jang; Hyeonseok Moon; Hwanseok Jang; Taewoo Lee; Hyungjin Lee; Jeseung Lee; Juhyoung Park; Heuiseok Lim
>
> **备注:** Research conducted in 2024
>
> **摘要:** We release Llamion, a family of 14B-parameter open-weight language models obtained by transforming Orion-14B into the standardized Llama-family architecture. The transformation is performed by Efficient Knowledge Preservation for Transformation (KEPT), a recipe that combines (i) Normal Parameter Mapping (NPM) for unchanged modules, (ii) Optimized Parameter Mapping (OPM), a training-free LayerNorm-to-RMSNorm initialization we prove optimal under the near-zero-mean activation regime induced by weight decay, and (iii) Cross-architecture Knowledge Distillation (XKD), an equal-size frozen-teacher distillation that aligns the converted model's outputs with the source model's on any reasonable input distribution. Llamion recovers Orion's behaviour on H6, MT-Bench, and KoMMLU with only ~123M tokens on a single A100 in four days; Llamion-Base reaches 66.87% on KoMMLU, exceeding the next-best entry of the Open Ko LLM Leaderboard by >7.0 absolute points at submission time. Capabilities entirely absent from the transfer corpus (Python programming and 200K-token context handling) survive the architectural transition intact. We release three checkpoints (Base, Chat, LongChat) that load with trust_remote_code=False in the Hugging Face Transformers library.
>
---
#### [new 009] Found in Conversation: LLMs Teach Themselves to Close the Multi-Turn Gap
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLM在多轮对话中性能下降的问题。通过自监督学习方法，提升模型在多轮对话中的表现，使其接近单轮效果。**

- **链接: [https://arxiv.org/pdf/2605.24432](https://arxiv.org/pdf/2605.24432)**

> **作者:** Tianlang Chen; Shirley Wu; Jure Leskovec
>
> **备注:** 17 pages, 3 figures, 6 tables
>
> **摘要:** Large Language Model (LLM) interactions are typically underspecified, with users clarifying all necessary details across multiple conversational turns. Yet recent work shows that LLMs perform far worse in this multi-turn setting than in a single turn with same information being available at once, a phenomenon termed "Lost-in-Conversation." However, bridging this gap effectively remains an open problem. Here we introduce Found in Conversation (FiC), a training framework where a model teaches itself to find and recover its single-turn competence given underspecified multi-turn prompts. We develop View-Asymmetric Self-Distillation, which distills across two views of the same task information--single-turn view for the teacher, multi-turn view for the student--transferring strong single-turn behavior into weak multi-turn behavior. This requires no stronger external teacher, which is unavailable as even frontier LLMs exhibit this gap. Across model families (Llama, Qwen, Phi, and OLMo) and sizes (3B-14B), FiC recovers at least 92% of single-turn performance and reaches 100% on two Llama backbones, yielding more efficient and helpful multi-turn conversations with single-turn capabilities intact.
>
---
#### [new 010] A Multi-Probe Audit of Clinical-Interview Depression Detection Benchmarks
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于抑郁症检测任务，旨在评估临床访谈数据集的基准性能。通过多角度验证模型可靠性，发现现有评估存在偏差，模型在不同数据上的表现差异显著。**

- **链接: [https://arxiv.org/pdf/2605.23977](https://arxiv.org/pdf/2605.23977)**

> **作者:** Takehiro Ishikawa; Jon Duke
>
> **摘要:** This paper audits benchmark evaluation in clinical-interview depression detection through four complementary probes across DAIC/E-DAIC, CMDC, ANDROIDS, MODMA, and PDCH. First, we re-evaluate E-DAIC under strict subject-disjoint leave-one-subject-out cross-validation. A lightweight hybrid text-plus-LLM-score model reaches macro-F1 = 0.723 - the highest reported under this protocol, to our knowledge - providing a conservative out-of-fold reference point that does not depend on the privileged official holdout. Second, we test whether the E-DAIC official split supports fine-grained leaderboard rankings by sweeping 96 model configurations across modality bundles, pooling strategies, and learners. Development-side cross-validation and official-test rankings align only moderately: the best cross-validation configuration ranks twentieth on the official test, the official-test winner ranks forty-first by cross-validation, top-3 overlap is zero, and the apparent winner is rank-1 in only 32.3% of subject bootstraps. Third, we externally validate strong public CMDC and ANDROIDS baselines that achieve near-ceiling in-domain performance. Zero-shot transfer to external corpora is substantially weaker. Finally, we stress-test E-DAIC text and audio models using paired symptom-dense versus symptom-light interview slices defined by an SRDS-based annotator. Text scores rise sharply on symptom-dense slices, whereas audio scores remain nearly flat; the text-minus-audio gap is positive across all five seeds.
>
---
#### [new 011] DRInQ: Evaluating Conversational Implicature with Controlled Context Variation
- **分类: cs.CL**

- **简介: 该论文提出DRinQ基准，用于评估对话隐含意义的语用推理，解决大模型在情境理解上的不足。通过控制上下文变化，测试模型生成与推理能力，揭示生成与推理的不对称性。**

- **链接: [https://arxiv.org/pdf/2605.24267](https://arxiv.org/pdf/2605.24267)**

> **作者:** Hirona Jacqueline Arai; Xiang Ren
>
> **备注:** To be presented at ACL 2026
>
> **摘要:** Human conversation relies heavily on conversational implicature, in which speakers convey meanings that are suggested rather than explicitly stated. Although recent large language models exhibit strong conversational fluency, they remain unreliable when interpretation depends on reasoning that integrates social and contextual cues, a process rarely articulated in text. We introduce DRinQ, a benchmark for evaluating pragmatic reasoning about conversational implicature in question utterances, designed to isolate pragmatic variation while holding each question's surface form fixed. To support scalable evaluation, we propose a semi-automated pipeline that produces question-context-interpretation instances with systematic variation. Across evaluations, we find a consistent generation-inference asymmetry: while state-of-the-art models can generate plausible pragmatic scenarios when guided, they often fail to recover the intended implication at inference time. For smaller models, structured prompting improves alignment with human judgments. A comparative writing study further reveals complementary strengths: human authors tend to produce safer, predictable contexts, whereas models generate varied scenarios with interpretations that sometimes exceed contextual support. These findings highlight persistent challenges in modeling conversational implicature and motivate more context-sensitive evaluation frameworks.
>
---
#### [new 012] When Do LLM Agents Treat Surface Noise Differently from Semantic Noise? A 68-Cell Measurement Study with a Held-Out Trace-Level Validation
- **分类: cs.CL**

- **简介: 该论文研究LLM代理在处理语义噪声与表面噪声时的差异，通过实验验证其对答案的影响。属于自然语言处理任务，旨在理解噪声对推理过程的影响。**

- **链接: [https://arxiv.org/pdf/2605.25981](https://arxiv.org/pdf/2605.25981)**

> **作者:** Liyun Zhang; Jiayi Guo
>
> **摘要:** We document an empirical phenomenon in chain-of-thought and ReAct agents driven by ten large language models from seven architecture families: meaning-bearing perturbations (e.g., paraphrase, synonym) alter final answers more often than presentation perturbations (e.g., formatting, reordering) of comparable severity. Across 68 cells spanning GSM8K, MATH, and HotpotQA (1,530 originals and $\sim$11,150 variants), the inconsistency gap averages +19.69 pp after severity matching (paired $t=9.58$, $p<0.0001$), with 64/68 cells positive. The gap survives four severity-proxy audits and remains significant when excluding qwen models (+11.10 pp, $p<0.0001$). Several stress tests fail honestly: cluster-bootstrap significance disappears under stricter assumptions, tractability contrasts do not replicate, cross-architecture generator swaps break per-cell rankings, and a second LLM judge yields only moderate agreement ($\kappa=0.50$). We then validate the headline effect on a fully held-out 11th model (qwen2.5-14B-Instruct; 1,800 trajectories) and re-test a pre-registered capability$\times$tractability partition, observing a small but positive held-out effect (3/4 cells positive; pooled Welch $t=3.81$, $p=9.6\times10^{-4}$). Using held-out trajectories, we probe four trace-level mechanism signals. Two prior mechanism claims fail to replicate and are explicitly retracted. Two new probes instead support a \emph{stealth-divergence} picture: semantic perturbations often preserve the first action but induce divergence in intermediate reasoning from later steps onward, accompanied by slightly deeper trajectories. We position this as a measurement contribution with held-out replication and a partial trace-level account of how semantic perturbations propagate through agent reasoning. Code, perturbation corpus, raw trajectories, and analysis scripts are released anonymously for review.
>
---
#### [new 013] Guarded Repair for Harm-Aware Post-hoc Replacement of LLM Mathematical Reasoning
- **分类: cs.CL; cs.AI; cs.SE**

- **简介: 该论文属于数学推理修复任务，解决后处理修复中的风险问题，通过GuardedRepair框架实现安全的替换，提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2605.24613](https://arxiv.org/pdf/2605.24613)**

> **作者:** Haizhou Xia
>
> **备注:** 15 pages,including appendices. Code and artifacts available at this https URL
>
> **摘要:** Post-hoc repair of LLM mathematical reasoning introduces an asymmetric risk: fixing an incorrect reasoning trace is useful, but replacing a trace that was already correct can be harmful. We study this problem under a selective replacement setting, where a system must decide whether a repaired candidate is safer than preserving the original cached trace. We present GuardedRepair, a guarded best-of-N repair framework that diagnoses cached reasoning traces, selectively triggers repair, and accepts answer-changing candidates only when deterministic verification guards support replacement. The framework combines lightweight symbolic checks, surface semantic-risk diagnostics, bounded candidate generation, and conservative acceptance policies. On the full GSM8K test set, where the initial reasoner already achieves 95.60% accuracy, GuardedRepair improves final accuracy to 96.89%, fixing 17 of 58 remaining errors without measured broken-correct cases in the main run. On a weak-reasoner ASDiv setting, accuracy improves from 78.40% to 87.60%. Direct regeneration baselines show that this gain is not explained by stronger-model re-solving alone: re-solving all GSM8K examples lowers accuracy to 93.03% and breaks 47 initially correct answers. Additional analyses show that guarded repair substantially improves the fixed/broken tradeoff, while also revealing that replacement risk is reduced rather than eliminated. These results support viewing post-hoc repair as harm-aware selective replacement rather than unconstrained re-solving.
>
---
#### [new 014] Lngram: N-gram Conditional Memory in Latent Space
- **分类: cs.CL**

- **简介: 该论文提出Lngram，用于序列建模任务，解决传统模型依赖分词和计算密集的问题。通过潜空间N-gram查找，提升模型效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.24869](https://arxiv.org/pdf/2605.24869)**

> **作者:** Yunao Zheng; Guoyang Xia; Xiaojie Wang; Lei Ren
>
> **摘要:** Sequence modeling requires both compositional reasoning and local static knowledge retrieval, yet standard Transformers handle both through dense computation. Engram partially decouples retrieval from the backbone, but its token-based keys remain tied to text tokenization and hash compression. We propose Lngram, a latent-space conditional memory module that learns discrete symbols directly from hidden states and performs N-gram lookup over these symbols. This design removes the dependence on tokenizer IDs and naturally extends to non-text modalities. In our evaluated settings, Lngram outperforms Transformer and Engram baselines, consistently reduces perplexity in long-context language modeling, and effectively injects domain knowledge when added post hoc to pretrained models. Joint training with the backbone further surpasses full fine-tuning, while experiments on vision-language and vision-language-action tasks show overall gains. Analyses with LogitLens and CKA suggest that Lngram enables prediction-relevant information to emerge earlier, increasing effective depth with limited inference and memory overhead. Code is available at this https URL.
>
---
#### [new 015] Re-defining Humor Data Objects for AI Humor Research
- **分类: cs.CL**

- **简介: 该论文属于AI幽默研究任务，旨在解决幽默理解中的上下文缺失和多模态问题。通过改进提示策略，提升LLM生成幽默解释的质量，并构建可用于数据增强的幽默数据对象。**

- **链接: [https://arxiv.org/pdf/2605.25171](https://arxiv.org/pdf/2605.25171)**

> **作者:** Anna Arnett; Bang Nguyen; Meng Jiang
>
> **摘要:** In most existing AI humor research, humor was treated as either "present" or "not present." We explore the concept of humor as a social interaction with context and explanations. During this project, we defined a humor reasoning data object and developed a way to prompt LLMs to generate an explanation of humor effective for general population. We iterated from an earlier prompt to an improved prompt, found that the later version reduced important errors, and then scaled generation to a large number of data objects which have the potential to enable data synthesis and data augmentation for AI humor research. Our main takeaway is that better prompting of an LLM improves humor explanation quality, especially by handling missing context, multi-modality, and transcript issues more carefully. These results establish a strong foundation for future work on AI understanding of humor as social behavior.
>
---
#### [new 016] CRPO: Character-centric Group Relative Policy Optimization for Role-aware Reasoning in Role-playing Agents
- **分类: cs.CL**

- **简介: 该论文属于角色扮演代理的推理任务，旨在解决角色一致性与风格崩溃问题。通过CRPO框架，提升角色独特性与情感一致性。**

- **链接: [https://arxiv.org/pdf/2605.25511](https://arxiv.org/pdf/2605.25511)**

> **作者:** Yihong Tang; Kehai Chen; Liang Yue; Benyou Wang; Min Zhang
>
> **摘要:** Recent advancements in Reinforcement Learning (RL), particularly Group Relative Policy Optimization (GRPO), have significantly enhanced the reasoning capabilities of Large Language Models. However, applying these problem-centric optimization methods to role-playing agents often leads to a loss of character fidelity and style collapse, as they prioritize context-specific utility over persona alignment. To address this, we propose Character-Centric Group Relative Policy Optimization (CRPO), a framework designed to realign RL objectives with the role-playing task. CRPO improves character distinctiveness through three mechanisms: decoupling task logic from stylistic rewards to resolve gradient conflicts, dynamically adapting optimization constraints based on character complexity, and utilizing generic responses as negative baselines to prevent the model from reverting to a common distribution. Extensive experiments demonstrate that CRPO outperforms existing methods in consistency, emotion and others.
>
---
#### [new 017] By Their Fruits You Will Know Them: Comparing Formalizations of Law by the Decisions They Encode
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律形式化任务，旨在比较不同法律条文形式化版本的差异。通过分析推理结果，识别形式化间的分歧，揭示潜在法律争议。**

- **链接: [https://arxiv.org/pdf/2605.25186](https://arxiv.org/pdf/2605.25186)**

> **作者:** Julius Vernie; Matthias Grabmair
>
> **备注:** 23 pages, 17 figures, submitted to EMNLP PROC 2026
>
> **摘要:** Formalizing legal provisions promises machine-accessible law and automated legal reasoning, and recent LLMs make it tempting to generate such formalizations directly from statutory text. However, any formalization makes implicit interpretive choices whose consequences are hard to anticipate, especially if an LLM is the author. We present a method for systematically comparing different formalizations of the same legal provision by their inferences on individual cases. Given multiple formalizations of a provision, we match them at the node level, derive a shared interface for each pair from the matching, and use a SAT solver to enumerate the edge cases on which any two formalizations disagree. Selected edge cases are then verbalized into concrete factual scenarios that a legal expert can examine and act on. We apply our method to formalizations of ten EU provisions generated by nine frontier LLMs. We find that behavioral divergence between formalizations is essentially uncorrelated with their structural agreement and that the verbalized cases reveal qualitatively distinct types of disagreement, including divergences that mirror genuine controversies in the legal commentary.
>
---
#### [new 018] Repeated Sequences Reveal Gaps between Large Language Models and Natural Language
- **分类: cs.CL; cs.IT; stat.AP**

- **简介: 该论文属于自然语言处理领域，旨在评估大语言模型是否捕捉自然语言的结构。通过分析重复子序列和熵增长模式，揭示模型与自然语言在长程组织上的差异。**

- **链接: [https://arxiv.org/pdf/2605.24850](https://arxiv.org/pdf/2605.24850)**

> **作者:** Kumiko Tanaka-Ishii
>
> **备注:** ACL 2026
>
> **摘要:** Evaluating whether large language models (LLMs) capture the structure of natural language beyond local fluency remains an open challenge. Existing evaluation methods, largely based on task performance or short-context behavior, provide limited insight into the long-range statistical organization of generated text. We propose a complementary evaluation framework based on repeated subsequences. By analyzing their distribution across scales and relating it to higher-order Rényi entropies, we probe how texts reuse previously established structure under finite-length conditions. Experiments on human-written texts and length-matched GPT-generated texts show that, while power-law models can describe restricted ranges of block length, the observed entropy growth is often equally or better characterized by logarithmic--power forms. Across datasets, natural language exhibits stable entropy-growth patterns over accessible ranges, with consistent average behavior despite variability across individual texts. In contrast, GPT-generated texts show systematic and statistically significant shifts in estimated exponents with model size. These results demonstrate that repeated-subsequence entropy provides a quantitative structural diagnostic that reveals systematic differences in long-range organization, distinguishing natural language from state-of-the-art LLM outputs beyond surface-level fluency.
>
---
#### [new 019] Fine-Tuning Over Architectural Complexity: Broad-Coverage PII Detection on PIIBench with DeBERTa
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于PII检测任务，旨在提升模型在异构文本中的覆盖能力。通过对比不同DeBERTa模型结构，发现直接微调效果最佳，证明多样数据和简单目标函数更有效。**

- **链接: [https://arxiv.org/pdf/2605.25816](https://arxiv.org/pdf/2605.25816)**

> **作者:** Pritesh Jha
>
> **摘要:** Personally identifiable information (PII) detection systems are frequently trained within narrow source or domain boundaries, limiting coverage when deployed on heterogeneous text. We study model fine-tuning on a corrected multi-source PIIBench preparation spanning 82 retained entity types across ten source datasets. We evaluate three DeBERTa-based approaches: direct token classification fine-tuning, a source-conditioned hierarchical model (SC+H), and a three-phase curriculum extension (SC+H+Curr). Against eight published comparator systems on a reproducible 5,000-record held-out subset (test_5k), direct fine-tuned DeBERTa achieves F1 0.6476, while SC+H and the curriculum variant achieve 0.5899 and 0.2772 respectively; the strongest published comparator reaches only 0.1723. Because validation initially favoured SC+H, we perform a final streamed evaluation on the complete 100,002-record held-out split. Direct fine-tuning remains superior, achieving F1 0.6455 versus 0.5894 for SC+H. Entity-level analysis shows that direct fine tuning wins 54 of 82 fine entity types and all ten coarse groups by support-weighted entity F1, while SC+H retains localised advantages on 28 types. The results indicate that diverse task-specific training data and a simple weighted cross-entropy objective contribute more to broad-coverage PII detection than the tested architectural and curriculum complexity.
>
---
#### [new 020] Harmony in Diversity: Multi-domain Contrastive Policy Optimization for Large Reasoning Models
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决多领域推理模型训练中的知识干扰问题。通过对比学习促进跨领域知识共享与领域内知识整合，提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2605.25443](https://arxiv.org/pdf/2605.25443)**

> **作者:** Zongji Yu; Wenshui Luo; Yiliu Sun; Hao Fang; Runmin Cong; Chaochao Lu; Chen Gong
>
> **备注:** 25 pages, 5 figures
>
> **摘要:** Post-training has significantly enhanced the reasoning capability of Large Reasoning Models (LRMs), especially with Reinforcement Learning (RL) like Group Relative Policy Optimization (GRPO). However, GRPO-style RL methods in multi-domain settings often fail to achieve consistent improvements across all domains due to inherent interference in policy optimization. Prior studies on multi-domain RL primarily focus on alleviating cross-domain interference, while often neglecting the pivotal role of knowledge sharing, which we argue is the key to transforming cross-domain interactions from harmful competition into beneficial transfer. To address this limitation, we propose Multi-domain Contrastive Policy Optimization (MCPO), which analyzes the structural relationships among rollouts and promotes cross-domain knowledge sharing and in-domain knowledge consolidation in a contrastive manner. Specifically, for a given prompt, MCPO identifies transferable reasoning trajectories from other domains as positive examples, while treating incorrect rollouts as negative ones. It then encourages consistent representations for positive pairs and pushes negative pairs apart, thereby facilitating knowledge transfer and reducing interference. Moreover, MCPO aligns intra-domain correct rollouts to build a consolidated representation space. In this way, MCPO contrastively learns a harmonious representation space that can accommodate diverse multi-domain knowledge. Empirical results show that MCPO improves the reasoning capabilities of LRMs across multiple domains and even outperforms single-domain training in some cases. Code is available at this https URL.
>
---
#### [new 021] EfficientGraph-RAG: Structured Retrieval-State Management for Cross-Task Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于知识增强生成任务，解决传统RAG系统结构松散导致的复杂检索瓶颈问题。提出EfficientGraph-RAG，通过结构化检索状态管理提升效果与效率。**

- **链接: [https://arxiv.org/pdf/2605.25379](https://arxiv.org/pdf/2605.25379)**

> **作者:** Miaohe Niu; Lianlei Shan; Zhengtao Yu; Jingbo Zhu; Tong Xiao
>
> **备注:** 19 pages, 5 figures, 14 tables
>
> **摘要:** Retrieval-augmented generation (RAG) has become the standard way to ground large language models in external knowledge, but many systems still organize evidence as flat chunks and retrieve it through largely unstructured search. This weak structure becomes a bottleneck for complex retrieval: the system must decide where to search, how to move from coarse topics to entity-relation evidence, which evidence has been verified, and which intermediate artifacts can be reused. We define these intermediate variables as a retrieval state and study RAG as structured state management. EfficientGraph-RAG makes this state explicit through three coupled mechanisms: TAM defines a typed hierarchical state space over evidence, MARS updates and verifies the state through role-specialized agents, and SMP stores reusable state under hierarchy-aware access control. Using one shared framework configuration, EfficientGraph-RAG ranks first on the reported answer-quality metrics averaged over the three evaluated LongBench retrieval-style subsets, matches the strongest agentic baseline on HotpotQA EM while reducing large-model token usage by $3.51\times$, and provides a low-token DocVQA result among retrieval-organizing cross-modal methods. Component analysis shows role-specific mechanisms: MARS is the main answer-quality driver, TAM supplies the typed traversal state and Adaptive Routing signal, and SMP enables corpus-dependent reuse, with cross-query cache hit rates ranging from 3.77% to 23.18%.
>
---
#### [new 022] Faithfulness Metrics Don't Measure Faithfulness: A Meta-Evaluation with Ground Truth
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型解释任务，旨在解决faithfulness度量不准确的问题。通过构建基准数据集，评估现有度量方法的有效性，揭示其局限性。**

- **链接: [https://arxiv.org/pdf/2605.25052](https://arxiv.org/pdf/2605.25052)**

> **作者:** Yoav Gur-Arieh; Ana Marasović; Mor Geva
>
> **摘要:** Chains of thought (CoTs) have become central in interpreting and auditing behaviors of large language models. Yet growing evidence suggests that these traces often fail to faithfully represent the computations behind a model's predictions. Several faithfulness metrics have been proposed, but whether they indeed measure faithfulness remains unknown. Answering this requires ground-truth labels, which are hard to obtain since internal computations are not directly observable. Consequently, most works proposing metrics report only absolute scores or comparisons to prior metrics, and the few existing benchmarks rely on proxies like plausibility or importance, properties orthogonal to faithfulness that can mislead about whether a CoT can be trusted. We address this challenge by constructing tasks whose outputs reveal which intermediate computations must have produced them, and developing an automated labeling pipeline that yields ground-truth faithfulness labels at both the step and CoT level. Building on this methodology, we present BonaFide, a benchmark of 3,066 labeled CoTs across 13 tasks and 10 models, and use it to conduct the first systematic evaluation of prominent faithfulness metrics. Our experiments show that most metrics perform near chance, exhibit strong prediction biases and degrade on longer CoTs. The best metric reaches only 0.70 AUROC at the CoT level while another reaches 0.59 at the step level, with neither transferring across settings, while entailing prohibitively high computational cost. Our results expose fundamental gaps in current faithfulness evaluation and call for the development of more reliable and efficient metrics.
>
---
#### [new 023] Side-by-side Comparison Amplifies Dialect Bias in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型中的隐性方言偏见问题，通过对比分析SAE和AAVE文本，发现模型在对比情境下偏见加剧。任务为检测与缓解语言模型的方言偏见。**

- **链接: [https://arxiv.org/pdf/2605.24384](https://arxiv.org/pdf/2605.24384)**

> **作者:** Kritee Kondapally; Claire J. Smerdon; Pooja C. Patel; Ogheneyoma Akoni; Jevon Torres; Jaspreet Ranjit; Matthew Finlayson; Swabha Swayamdipta
>
> **备注:** In proceeding at ACM Conference on Fairness, Accountability, and Transparency 2026
>
> **摘要:** Language models (LMs) can exhibit systematic biases against speakers based on variations in their dialects, even in the absence of a dialect label, a behavior known as covert dialect bias. In this work, we quantify covert dialect bias in online discourse by evaluating how LMs associate stereotypical traits (derived from social psychology research on racial bias) with intent-equivalent tweets in Standard American English (SAE) and African-American Vernacular English (AAVE). While prior work shows that LMs associate more negative stereotypes with AAVE when evaluating tweets in isolation, we are surprised to find that this bias is significantly exacerbated when SAE / AAVE tweet pairs are compared side by side, a setting that more closely reflects high-impact decision making contexts in which models are used to rank candidates. The bias only worsens when dialect labels are explicitly specified. This is striking, given the extensive efforts from commercial developers to mitigate bias in their LMs. Encouragingly, we show that counterfactual fairness finetuning can mitigate covert dialect bias for some stereotypical traits, reducing average disparities when evaluating tweets in isolation, however, these improvements do not consistently hold across traits when evaluating SAE / AAVE tweets side by side. Our findings show that existing evaluation settings for covert dialect bias may underestimate its severity, specifically in contrastive settings. Additionally, overt dialect bias remains pronounced even after safety aligned finetuning, indicating that it remains an unresolved problem, and motivates the need for more robust evaluation and mitigation frameworks.
>
---
#### [new 024] Clarify, Abstain or Answer? Strategising in Conversation with Belief-Augmented Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于对话策略任务，解决模型在不确定情况下如何选择回答、澄清或回避的问题。提出BAG方法，让模型基于自身信念状态决定策略，提升问答准确性和策略合理性。**

- **链接: [https://arxiv.org/pdf/2605.25831](https://arxiv.org/pdf/2605.25831)**

> **作者:** Joris Baan; Wilker Aziz; Barbara Plank; Raquel Fernández
>
> **摘要:** Large language models (LLMs) define a distribution over text, which can be viewed as a probabilistic representation of uncertainty: sampling K responses yields a belief state - responses a model deems plausible. Existing work exploits this representation for narrow tasks like either decoding or selective prediction, and often requires manual interventions, not controlling generation directly. We propose Belief-Augmented Generation (BAG): grounding LLMs in their own belief state via the prompt and letting them reason over these K samples to decide on a conversational strategy: answer, clarify, or abstain. In a multi-turn ambiguous QA setting, we find that LLMs by default rarely clarify or abstain, ignoring uncertainty about the input or facts. BAG improves QA accuracy across six models and yields strategy decisions more faithful to the belief state than prompt-only baselines. Disentangling when to clarify from when to abstain, however, remains challenging.
>
---
#### [new 025] Adaptive Graph Refinement and Label Propagation with LLMs for Cost-Effective Entity Resolution
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于实体消歧任务，解决传统ER方法的缺陷。提出Alper框架，通过动态图优化和标签传播，提升实体聚类效果。**

- **链接: [https://arxiv.org/pdf/2605.25814](https://arxiv.org/pdf/2605.25814)**

> **作者:** Hongtao Wang; Renchi Yang; Haoran Zheng; Xiangyu Ke
>
> **摘要:** Dirty entity resolution (ER), which identifies records referring to the same real-world entity from a single, messy dataset, is a fundamental task in data management and mining. However, the dominant blocking-matching-clustering paradigm for ER suffers from critical flaws. Its cascaded, decoupled workflow essentially produces a static, sparse graph plagued by missing edges (due to blocking failures) and noisy links (due to matching errors), causing error propagation and yielding suboptimal clusters, particularly when rigid transitivity is imposed in the clustering. We contend that matching and clustering are fundamentally synergistic, both optimizing for the construction of an ideal entity graph. Building upon this insight, we propose Alper, a unified framework that integrates these steps into an iterative probabilistic label propagation process over a global, evolving graph. Unlike disjoint blocking, Alper refines the graph structure and labels dynamically by adaptively integrating "weak but cheap" signals from graph propagation with "strong but expensive" LLM-based pairwise queries. For higher cost-effectiveness, we formulate the signal selection as a constrained optimization problem maximizing cumulative marginal gain under a query budget, solved via our greedy algorithm with provable theoretical guarantees. Our extensive experiments over eight benchmark datasets demonstrate that Alper is consistently superior to state-of-the-art cascaded pipelines.
>
---
#### [new 026] Confidence and Calibration of Activation Oracles for Reliable Interpretation of Language Model Internals
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型解释任务，旨在解决激活oracle的置信度评估问题。通过比较六种方法，发现Bootstrap模式频率效果最佳，提升了模型内部解释的可靠性。**

- **链接: [https://arxiv.org/pdf/2605.26045](https://arxiv.org/pdf/2605.26045)**

> **作者:** Federico Torrielli; Peter Schneider-Kamp; Lukas Galke Poech
>
> **摘要:** Activation oracles aim to make the activations of other models legible to humans and yield promising results compared to white-box interpretability techniques. However, uncertainty quantification (UQ) for the natural-language outputs of such activation oracles is so far understudied. Here, we investigate 6 different methods for estimating the confidence of activation oracles and evaluate how well-calibrated their confidence scores are. Our experiments on 6,000 samples per oracle (varying verbalizer and context prompts) reveal that bootstrap mode frequency is the best-calibrated method among those tested (ECE 5.7% vs. 25.5% for the answer-word log-probability on Qwen3-8B; 10.3% vs. 13.1% on Qwen3.6-27B), and that the log-prob baseline can serve as a fast triage signal at a fraction of the cost. Code and the patched trainer are available at this https URL.
>
---
#### [new 027] Trait-Aware Policy Optimization for Autoregressive Multi-Trait Essay Scoring
- **分类: cs.CL**

- **简介: 该论文属于多维度作文评分任务，旨在解决自回归模型后训练效果不佳的问题。提出TAPO框架，通过分解奖励提升评分一致性与准确性。**

- **链接: [https://arxiv.org/pdf/2605.25731](https://arxiv.org/pdf/2605.25731)**

> **作者:** Zhengyang Wang; Sanwoo Lee; Jiaxin Wang; Chenxi Miao; Weikang Li; Yunfang Wu
>
> **摘要:** Multi-trait essay scoring aims to provide fine-grained evaluation of writing quality across multiple dimensions. However, how to effectively post-train autoregressive scoring models remains underexplored. In this paper, we propose Trait-Aware Policy Optimization (TAPO), a post-training framework tailored to autoregressive multi-trait scoring. Our method decomposes rewards along both the sample and trait dimensions, combining global scoring consistency, trait-level accuracy, format validity, and inter-trait dependency preservation. In addition, we enhance supervised fine-tuning with enhanced prompts, allowing the model to internalize trait semantics before preference optimization. Experiments across multiple backbone models show that our method consistently improves multi-trait scoring performance over supervised fine-tuning and scalar-reward optimization baselines, demonstrating the effectiveness and transferability of trait-aware post-training for essay scoring.
>
---
#### [new 028] MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing
- **分类: cs.CL**

- **简介: 该论文属于 hallucination 检测任务，旨在解决多语言环境下 LLM 的事实性错误检测问题。通过分析 LLM 隐藏状态，提出 MultiHaluDet 框架，实现跨语言的高效检测。**

- **链接: [https://arxiv.org/pdf/2605.24919](https://arxiv.org/pdf/2605.24919)**

> **作者:** Riasad Alvi; Nurul Labib Sayeedi; Md. Faiyaz Abdullah Sayeedi
>
> **备注:** MeLLM @ ACL 2026
>
> **摘要:** Hallucinations in Large Language Models (LLMs) represent a critical barrier to their reliable deployment, a vulnerability heavily exacerbated in non-English and resource-constrained contexts. Existing detection approaches that rely on output confidence heuristics or single-layer internal representations frequently fail to capture deep, complex factual inconsistencies across diverse languages. To address this, we introduce MultiHaluDet, a novel three-stage stacking framework that detects multilingual hallucinations by probing the full hidden state trajectories of frozen LLMs without requiring language-specific fine-tuning. Our method extracts sequential features across multiple layers and processes them via a hybrid architecture using multi-scale attention and self-attention pooling. By generating out-of-fold embeddings that feed into a calibrated classical classifier ensemble, MultiHaluDet captures both fine-grained and coarse-grained patterns of factual inconsistency. Extensive experiments demonstrate that our framework achieves state-of-the-art detection performance, reaching up to 98.55% AUROC on the English HaluEval and TriviaQA benchmarks using Mistral-7B and LLaMA2-7B architectures. Crucially, we rigorously evaluate our framework's cross-lingual generalization across high (French), medium (Bangla), and low-resource (Amharic) languages. MultiHaluDet demonstrates exceptional representational robustness, consistently outperforming baselines and successfully transferring hallucination detection capabilities across typologically diverse linguistic tiers.
>
---
#### [new 029] TIAR: Trajectory-Informed Advantage Reweighting for LLM Abstention Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型 abstention 学习任务，旨在减少幻觉。通过轨迹信息重新加权优势，提升模型在不确定时的拒绝能力。**

- **链接: [https://arxiv.org/pdf/2605.25850](https://arxiv.org/pdf/2605.25850)**

> **作者:** Muyu Pan; Shu Zhao; Nan Zhang; Philip Shin; Varun Parekh; Vijaykrishnan Narayanan; Rui Zhang
>
> **备注:** 10 pages, 1 figure, 4 tables
>
> **摘要:** This paper investigates large language model (LLM) abstention learning, specifically using ternary reward, which incentivize truthfulness in large language models. This paper extends that idea by moving from a ternary reward to a Trajectory-Informed advantage reweighting, dynamically re-weights the abstention reward during Group Relative Policy Optimization (GRPO) training. The objective of this work focuses on abstention learning instead of improving truthfulness, serving as an exploration into hallucination reduction. The novelty of this paper lies in methodological innovation, advantage re-weighting, and benchmark selection. Leveraging GRPO's multiple trajectories as a natural abstention signal, this method uses a reward signal to explore knowledge boundaries and encourage consistency. By demonstrating that trajectories can be used as a confidence indicator of the policy relative to the query, they are then used to dynamically calculate the abstention advantage. AbstentionBench is used as the evaluation benchmark, as this work aims to contribute to the field of abstention learning. All datasets on the benchmark were tested against this method and various baselines. Empirical results demonstrate that TIAR achieves state-of-the-art abstention F1 scores across five of six evaluation categories, outperforming the static ternary baseline on 17 of 31 benchmark datasets while fully preserving baseline accuracy.
>
---
#### [new 030] TS-Skill: A Benchmark for Evaluating Analytical Skills in Time-Series Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于时间序列问答任务，旨在解决现有基准无法准确评估模型在时间信号上的分析能力问题。提出TS-Skill基准，涵盖三种分析技能，并构建了SKEvol框架进行大规模数据生成与验证。**

- **链接: [https://arxiv.org/pdf/2605.24703](https://arxiv.org/pdf/2605.24703)**

> **作者:** Liying Han; Kang Yang; Oliver Wang; Jason Wu; Pengrui Quan; Gaofeng Dong; Ozan Baris Mulayim; Sizhe Ma; Yuyang Yuan; Dezhi Hong; Mario Berges; Mani Srivastava
>
> **摘要:** Large language models (LLMs) and time-series language models (TSLMs) are increasingly applied to time-series question answering (TSQA). Unlike text-only QA, TSQA requires models to ground answers in temporal signals whose patterns may occur at different scales, specific time locations, or across separated intervals. However, existing benchmarks are typically organized by task types or high-level reasoning categories, making it difficult to diagnose the underlying signal-level capabilities driving model performance. We introduce TS-Skill, a controlled benchmark for evaluating three composable analytical skills in TSQA: temporal scale selection (SK1), temporal localization (SK2), and cross-interval integration (SK3). TS-Skill provides timestamp-aware questions, broad domain coverage, and human-validated QA quality. To construct the benchmark at scale, we develop SKEvol, a skill-guided agentic framework that combines domain-aware time-series seed generation, skill-controlled question generation, metadata- and code-assisted answer construction, multi-phase signal-grounded verification, and human-in-the-loop curation. Experiments on ten state-of-the-art LLMs and TSLMs reveal substantial and uneven capability gaps across SK1-SK3. In particular, SK3 remains consistently challenging for non-agent models, whereas tool-augmented agents show a selective advantage on standalone SK3. These findings demonstrate that skill-level evaluation can uncover temporal reasoning failures that are obscured by aggregate TSQA scores.
>
---
#### [new 031] StepGap: A Hybrid NLI-LLM Checker for Step-Level Evidence-Gap Detectionin Multi-Hop Question Answering
- **分类: cs.CL**

- **简介: 该论文提出StepGap，用于多跳问答中的步骤级证据缺口检测，解决证据缺失或错误问题，通过混合NLI-LLM方法实现精准标注与修复。**

- **链接: [https://arxiv.org/pdf/2605.24733](https://arxiv.org/pdf/2605.24733)**

> **作者:** Yuelyu Ji; Zhuochun Li; Hui Ji; Daqing He
>
> **摘要:** We present \textbf{StepGap}, a hybrid NLI-LLM decision tree that detects step-level evidence gaps in multi-hop QA and emits one of three typed labels: \textsc{Contradicted Claim} (CC), \textsc{Irrelevant Evidence} (IE), or \textsc{Missing Bridge} (MB), each tied to a concrete repair action. On 82 multi-hop questions (181 annotated steps, $\kappa{=}0.704$), StepGap reaches sF1$=$72.0, within the bootstrap confidence interval of an LLM-only baseline (70.1) but with a more decomposable structure: every StepGap stage \emph{hurts} F1 when removed, while three of four LLM-only removals \emph{improve} F1 -- a sign of \emph{competing-error cancellation}, where internal stages mask each other's errors. We further expose a \emph{Q-F1 trap}: question-level F1 is mechanically inflated by checkers that flag every step, making step-level F1 the necessary diagnostic. Used as a typed GRPO process reward, StepGap improves Qwen2.5-7B-Instruct Exact Match from $32.1{\pm}0.3$ to $35.4{\pm}0.9$ across three seeds, with the single-run comparison showing a $+5.6$ Avg EM gain over the matched Search-R1 GRPO reproduction.
>
---
#### [new 032] Triplet-Block Diffusion RWKV
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer模型的序列解码效率低和注意力成本高的问题。通过引入三元组块结构，提出B³D-RWKV模型，实现高效且并行的双向扩散过程。**

- **链接: [https://arxiv.org/pdf/2605.25969](https://arxiv.org/pdf/2605.25969)**

> **作者:** Ke Lin; Yiyang Luo; Zhaolong Su; Yunya Song; Anyi Rao
>
> **摘要:** Causal Transformer language models suffer from strictly sequential decoding and a quadratic per-step attention cost. While linear-time causal models and discrete diffusion models each address these weaknesses, their integration remains inherently inconsistent: diffusion requires bidirectional attention, while causal models are unidirectional. To unify these architectures, we propose $B^3D-RWKV$, a diffusion RWKV variant that integrates the model's $O(L)$ inference efficiency with parallel, bidirectional discrete-diffusion through a \emph{triplet-block layout} method. $B^3D-RWKV-7.2B$ reaches comparable accuracy on an 8-task suite versus existing models while significantly outperforming baselines in decoding throughput with an average of $\mathbf{1.6\times}$ speedup.
>
---
#### [new 033] From Facts to Insights: A Persona-Driven Dual Memory Framework and Dataset for Role-Playing Agents
- **分类: cs.CL; cs.DB; cs.MA**

- **简介: 该论文属于角色扮演代理任务，旨在解决长期对话中 persona 稳定性不足的问题。提出 DualMem 框架，分离事实与个性洞察，提升角色一致性。**

- **链接: [https://arxiv.org/pdf/2605.25693](https://arxiv.org/pdf/2605.25693)**

> **作者:** Rongsheng Zhang; Ruofan Hu; Weijie Chen; Jiji Tang; Junnan Ren; Wanying Wu; Xunuoyan Chen; Tangjie Lv; Tao Jin; Zhou Zhao
>
> **备注:** Preprint
>
> **摘要:** While role-playing agents excel in short-term interactions, long-term conversations overwhelm context windows, motivating external memory frameworks. Current systems typically rely on persona-agnostic summarization, which records facts without persona-specific interpretation, yielding generic responses that compromise persona fidelity. To bridge this gap, we introduce RoleMemo, a dataset featuring four reasoning tasks where the factual fragments must be interpreted through the persona to reach the correct answer. Evaluation on RoleMemo exposes critical limitations of persona-agnostic frameworks. We thus propose DualMem, which decouples memory into two streams: factual cognition and persona-conditioned insight. Trained through Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), our framework with a 4B-parameter model outperforms zero-shot persona-agnostic frameworks powered by DeepSeek-V3.2 for sustained persona fidelity. Our resources are available at this https URL.
>
---
#### [new 034] Unveil: Unified Visual-Textual Integration and Distillation for Multi-modal Document Retrieval
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态文档检索任务，旨在解决传统方法在处理多样文档格式时的不足。提出Unveil框架，融合视觉与文本特征，并通过知识蒸馏提升检索效果。**

- **链接: [https://arxiv.org/pdf/2605.24530](https://arxiv.org/pdf/2605.24530)**

> **作者:** Hao Sun; Yingyan Hou; Jiayan Guo; Bo Wang; Chunyu Yang; Jinsong Ni; Yan Zhang
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** Document retrieval in real-world scenarios faces significant challenges due to diverse document formats and modalities. Traditional text-based approaches rely on tailored parsing techniques that disregard layout information and are prone to errors, while recent parsing-free visual methods often struggle to capture fine-grained textual semantics in text-rich scenarios. To address these limitations, we propose \textbf{Unveil}, a novel visual-textual embedding framework that effectively integrates textual and visual features for robust document representation. Through knowledge distillation, we transfer the semantic understanding capabilities from the visual-textual embedding model to a purely visual model, enabling efficient parsing-free retrieval while preserving semantic fidelity. Experimental results demonstrate that our visual-textual embedding method surpasses existing approaches, while knowledge distillation successfully bridges the performance gap between visual-textual and visual-only methods, improving both retrieval accuracy and efficiency.
>
---
#### [new 035] Better, Faster: Harnessing Self-Improvement in Large Reasoning Models
- **分类: cs.CL**

- **简介: 该论文属于大模型推理任务，解决自提升训练中数据不平衡和过度思考问题，提出HSIR方法提升推理性能与效率。**

- **链接: [https://arxiv.org/pdf/2605.24998](https://arxiv.org/pdf/2605.24998)**

> **作者:** Qihuang Zhong; Liang Ding; Juhua Liu; Bo Du; Leszek Rutkowski; Dacheng Tao
>
> **备注:** Accepted by ICML2026
>
> **摘要:** Self-improvement training enables the large reasoning models (LRMs) to improve themselves by self-generating reasoning trajectories as training data without external supervision. However, we find that this method often falls short in complex reasoning tasks and even leads to model collapse. Through a series of preliminary analyses, we reveal two problems: (1) data imbalance, where most training samples are simple, but the challenging yet crucial samples are scarce; (2) overthinking, where many undesired samples with redundant reasoning steps are used for self-training. To this end, we propose HSIR, which effectively Harnesses Self-Improvement in large Reasoning models via two simple-yet-effective approaches. Specifically, HSIR introduces a verify-then-exit sampling strategy to mitigate data imbalance by efficiently collecting more accurate solutions for difficult queries, and designs an Intrinsic Diversity score to quantify overthinking and filter out the undesired solutions. We apply HSIR to various post-training paradigms, among which we further propose H-GRPO, an enhanced GRPO algorithm that leverages the intrinsic diversity as an external reward to encourage concise and diverse reasoning via reinforcement learning. Extensive results show that HSIR not only effectively enhances the reasoning performance, i.e., bringing up to +10.9% average performance gains, but also significantly improves the reasoning efficiency by reducing up to 42.4% relative inference overhead.
>
---
#### [new 036] Decompose-and-Refine: Structured Legal Question Answering with Parametric Retrieval
- **分类: cs.CL**

- **简介: 该论文属于法律问答任务，旨在解决多跳法律问答中答案准确性和法律依据明确性问题。提出DaR框架，通过分解问题和参数化查询优化检索效果。**

- **链接: [https://arxiv.org/pdf/2605.24454](https://arxiv.org/pdf/2605.24454)**

> **作者:** Jihyung lee; Hyounghun Kim; Gary Lee
>
> **摘要:** Large language models (LLMs) have shown strong performance in the legal domain, demonstrating notable potential in Legal Question Answering (LQA). However, unlike general QA, LQA requires answers that are not only accurate but also rigorously grounded in explicit legal authority. In statutory LQA, many questions require multi-hop reasoning across multiple legal issues, substantially increasing the risk of hallucination, thereby making accurate retrieval of supporting statutory provisions a critical prerequisite. Despite recent progress in multi-hop QA, existing approaches often rely on reasoning in natural language or retrieval without explicit query reformulation, leaving the vocabulary gap between user questions and statutory text largely unaddressed. To address this challenge, we propose Decompose-and-Refine (DaR), a statute-grounded LQA framework that tightly integrates step-wise question decomposition with parametric knowledge-based query refinement. DaR progressively decomposes a complex legal question into atomic sub-questions and generates statute-aligned parametric queries for each sub-question, enabling the selection of a single most central statutory provision corresponding to each legal issue. We evaluate DaR on KoBLEX, a Korean multi-hop LQA benchmark grounded in statutory law, using Qwen3-32B and Gemma3-27B. Experimental results demonstrate that DaR consistently improves both retrieval accuracy and final answer quality over existing approaches. Moreover, by explicitly separating sub-questions and their corresponding statutory provisions, DaR facilitates transparent, issue-level verification of complex legal reasoning processes.
>
---
#### [new 037] Investigating the Interplay between Contextual and Parametric Chain-of-Thought Faithfulness under Optimization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究CoT faithfulness的优化问题，探讨上下文与参数两种评估范式的相互作用。通过提出FaithMate方法，分析两者的关系及效果差异。**

- **链接: [https://arxiv.org/pdf/2605.24960](https://arxiv.org/pdf/2605.24960)**

> **作者:** Jingyi Sun; Qianli Wang; Pepa Atanasova; Nils Feldhus; Isabelle Augenstein
>
> **备注:** The first two authors contributed equally and share first-authorship
>
> **摘要:** Chain-of-Thought (CoT) faithfulness, i.e., whether CoTs genuinely reflect large language models' (LLM) underlying behavior, is typically evaluated under two disjoint paradigms: contextual faithfulness, measured by perturbing the input or CoT trace, and parametric faithfulness, assessed by intervening on a model's parametric knowledge. Yet prior work compares them only descriptively. We fill this gap by proposing FaithMate, a unified preference-alignment interface for optimizing models towards either faithfulness paradigm. It enables us to investigate the interplay between the two paradigms, examining whether and to what extent faithfulness gains generalize within and across paradigms. Across three models, two datasets, and six faithfulness metrics, we find that the two paradigms are positively coupled, yet asymmetric: optimizing towards parametric faithfulness yields consistent gains across both paradigms, whereas the contextual counterpart delivers more variable gains. Within the contextual paradigm, faithfulness gains on one metric do not consistently transfer to others, implying that existing contextual metrics capture disjoint facets of faithfulness and exposing inherent trade-offs. These findings imply that CoT faithfulness is not a monolithic objective and therefore requires multifaceted optimization and evaluation.
>
---
#### [new 038] Who judges the judges? Governance from metrics: a runtime framework for continuous LLM compliance monitoring
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于AI合规监控任务，解决生产系统持续合规性问题。提出基于运行时度量的治理框架govllm，通过模型选择与监管评估实现动态合规管理。**

- **链接: [https://arxiv.org/pdf/2605.24737](https://arxiv.org/pdf/2605.24737)**

> **作者:** Jehanne Dussert
>
> **备注:** 41 pages, 8 figures, preprint
>
> **摘要:** Current approaches to AI compliance treat conformity as a binary, audit-time verdict rather than a continuous, measurable property of production systems. We argue that this compliance fiction is structurally ill-suited to the requirements of the EU AI Act, which demands ongoing human oversight and the detection of emergent behavioural drift in deployed systems. We introduce governance from metrics, a principle whereby regulatory compliance is derived as a continuous signal from runtime observability rather than from static assessments. Building on this principle, we present govllm, an open-source framework implementing a governance-driven routing architecture in which model selection is determined by accumulated compliance scores rather than by latency or cost alone. Central to our approach is a panel of regulatory judges - LLM evaluators specialised per criterion (EU AI Act, GDPR, ANSSI, accessibility) - whose inter-judge disagreement we reframe not as noise but as a regulatory uncertainty signal warranting human arbitration. We validate this approach through a ground truth corpus of 49 annotated prompt/response pairs across five regulatory criteria, evaluated by four small language models (SLMs, 1.7B-7B parameters) running fully on-premise. Agreement rates range from 51.5% (mistral:7b) to 69.1% (phi4-mini), with no single model dominating across all criteria - empirically motivating the Profile-as-jury design. We further document three structural failure modes in small regulatory judges and a judge-specific position bias that degrades agreement by up to 25 percentage points across three question-order conditions (original, reversed, permuted). govllm is released as open-source software to support reproducible AI governance research.
>
---
#### [new 039] Temporal Concept Drift in Legal Judgment Prediction: Neural Baselines Across Three Epochs of Ukrainian Court Decisions
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于法律文本分类任务，研究法律语言随时间的变化对模型性能的影响。通过分析不同时间段的法院判决数据，探讨模型在时间漂移下的泛化能力与优化方法。**

- **链接: [https://arxiv.org/pdf/2605.24452](https://arxiv.org/pdf/2605.24452)**

> **作者:** Volodymyr Ovcharov
>
> **备注:** 17 pages, 6 tables, 5 figures. Dataset: this https URL
>
> **摘要:** Legal NLP benchmarks evaluate models on randomly split data, implicitly assuming that legal language is stationary. We test this assumption by fine-tuning four transformer encoders -- XLM-RoBERTa (base and large) and their legal-domain variants -- on Ukrainian court decisions from three temporal epochs defined by geopolitical disruptions: pre-war (2008-2013), hybrid war (2014-2021), and full-scale invasion (2022-2026). Each model is trained on one epoch and evaluated on all three, producing a 3x3 cross-temporal generalization matrix. Four findings emerge. (1) Forward degradation is severe: models trained on pre-war data lose up to 27.2 percentage points of macro-F1 when applied to full-scale invasion era decisions. (2) The degradation is asymmetric: backward transfer (full-scale to pre-war) is substantially more robust than forward transfer, consistent with the hypothesis that legal language is additive. (3) Legal-domain pretraining (Legal-XLM-R) does not improve absolute performance but reduces forward degradation magnitude and asymmetry. (4) Chronological continual learning eliminates catastrophic forgetting for general XLM-R: pre-war knowledge is fully retained (+1.8 to +6.2 pp) while full-scale performance gains +16.5 to +19.0 pp; reverse-chronological training causes severe forgetting. Cross-jurisdictional pretraining on Swiss Judgment Prediction data improves absolute performance but does not reduce temporal degradation magnitude, confirming that temporal drift is an intrinsic property of legal language evolution. The dataset (428K decisions across three epochs) is publicly available as a LEXTREME contribution.
>
---
#### [new 040] Locality Matters for Training-Free Audio Token Compression in Audio-Language Models
- **分类: cs.CL**

- **简介: 该论文属于音频语言模型任务，解决音频令牌压缩问题。提出LTBM方法，在不训练的情况下通过局部时间约束合并相似音频令牌，提升压缩效率。**

- **链接: [https://arxiv.org/pdf/2605.25179](https://arxiv.org/pdf/2605.25179)**

> **作者:** Jiale Luo; Xiaoyu Liang; Haoji Hu
>
> **备注:** Preprint. 8 pages main text, 10 pages total
>
> **摘要:** Audio-language models (ALMs) are increasingly used for audio captioning, question answering, and open-ended audio understanding, but their inference cost remains high when audio inputs are represented as long prefix-token sequences. These audio prefixes consume context budget, increase memory usage, and make deployment harder in resource-constrained or latency-sensitive settings. Existing training-free audio-token reduction methods mainly rely on fixed pooling or score-based pruning. Fixed pooling is content-agnostic, while score-based pruning can preserve isolated salient tokens but discard nearby acoustic context. We propose Local Temporal Bipartite Merging (LTBM), a training-free encoder-space compression method that merges similar nearby audio tokens under an explicit temporal window constraint. Beyond introducing LTBM, we use a controlled Global Merge variant to isolate whether temporal locality itself is a useful inductive bias for audio-token compression. Experiments on AudioCaps, Clotho, and MMAU with Qwen2-Audio show evidence of a task-dependent locality effect: locality-aware merging is more favorable for captioning at several compression settings, especially under stronger compression, while global matching is more competitive for multiple-choice audio understanding. A cross-backbone validation on Audio Flamingo 3 further supports the captioning-side advantage of locality-aware merging under moderate and aggressive compression.
>
---
#### [new 041] PowLU: An Activation Function for Stable Pre-Training of LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于深度学习任务，旨在解决LLMs预训练中的数值不稳定问题。提出PowLU激活函数，通过有理幂函数实现稳定非线性，提升训练效果和可扩展性。**

- **链接: [https://arxiv.org/pdf/2605.25704](https://arxiv.org/pdf/2605.25704)**

> **作者:** Peijie Jiang; Yuqi Feng; Cunyin Peng; Qian Zhao; Jia Liu; KunLong Chen; Zhiqiang Zhang; Jun Zhou
>
> **备注:** 17 pages, 7 figures, techreport
>
> **摘要:** In contemporary large language models (LLMs), the swish-gated linear unit (SwiGLU) activation function is widely adopted to regulate the information flow and introduce non-linearity. For large positive inputs, SwiGLU approximates the quadratic function $x^2$, providing strong nonlinearity and expressive capacity. However, this property also causes numerical instability as the input or model scale increases, particularly in low-precision LLM training. The main reason is its approximate quadratic amplification, which enlarges the output range and exacerbates outliers. To address this issue, we propose a stable activation function, Power Linear Unit (PowLU), for large-scale LLM pre-training. Specifically, PowLU employs a rational power function to achieve adaptive nonlinearity, thereby improving representation ability and enabling stable training in spike regions. Moreover, we provide theoretical justification for several key properties of PowLU. Scaling law experiments confirm that the performance is consistent across model sizes, and further experimental results with the Ling architecture (7.9B and 124B total parameters) demonstrate that PowLU achieves competitive results against SwiGLU and SwiGLU-Clip in large-scale training of LLMs. In addition, the experimental results also show that PowLU effectively improves the scalability of the large-scale training of LLMs.
>
---
#### [new 042] LLM Agent Based Renewable Energy Forecasting Using Edge and IoT Data A Review of Solar Wind Weather and Grid Aware Decision Support
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于可再生能源预测任务，旨在提升电网稳定性与能源管理。通过整合边缘与物联网数据，利用大语言模型增强预测精度与决策支持。**

- **链接: [https://arxiv.org/pdf/2605.25141](https://arxiv.org/pdf/2605.25141)**

> **作者:** Pavan Manjunath; Thomas Pruefer
>
> **摘要:** Reliable forecasting of renewable energy generation is a foundational requirement for grid stability energy trading battery scheduling and carbon aware operational planning Solar and wind resources are inherently intermittent their output fluctuates with cloud cover wind speed atmospheric turbulence seasonal patterns and local terrain The proliferation of IoT and edge devices spanning smart meters inverters anemometers pyranometers weather stations and grid interface sensors has created an unprecedented volume of real time operational data that conventional forecasting pipelines are ill equipped to exploit fully This review investigates how large language model LLM agents can enhance renewable energy forecasting by integrating heterogeneous sensor streams weather API data historical generation records grid constraints and contextual reasoning into unified decision support workflows We survey classical forecasting methods statistical time series models deep learning architectures physics hybrid approaches and emerging LLM agent frameworks for explanation uncertainty communication and operator guidance A six layer taxonomy is proposed covering data acquisition preprocessing feature engineering model inference uncertainty estimation and natural language reporting The review identifies twelve open challenges spanning real time deployment model drift under distribution shift uncertainty quantification hallucination control in LLM agents interoperability of edge hardware and integration with energy management systems The paper concludes by recommending a research agenda centred on open benchmarks physics informed LLM grounding and federated forecasting architectures
>
---
#### [new 043] AuthTrace: Diagnosing Evidence Construction in Thematically Dense Single-Author Corpora
- **分类: cs.CL**

- **简介: 该论文提出AuthTrace，用于诊断单作者语料中的证据构建问题。针对证据构建系统评估不统一的问题，通过统一的语料和指标进行比较，揭示了证据召回率对答案质量的影响及不同方法的失效模式。**

- **链接: [https://arxiv.org/pdf/2605.25382](https://arxiv.org/pdf/2605.25382)**

> **作者:** Xiaoqing Wu; Feifei Li; Haoliang Ming; Wenhui Que
>
> **摘要:** Evidence construction systems--chunk retrieval, agent memory, knowledge-graph traversal, and thematic indexing--are evaluated on separate benchmarks with incompatible corpora and metrics, making cross-paradigm diagnosis impossible. We introduce AuthTrace, the first diagnostic benchmark that places all major paradigms on a single corpus and query set by exploiting the dual nature of single-author collections. Built on thematically dense corpora where all texts share style, topic, and vocabulary, AuthTrace provides 2,099 instances with exhaustive gold evidence and a fan-in gradient as the primary diagnostic axis. Comparing eight systems across two QA models, we find that (1) evidence recall--not precision--is the dominant predictor of answer quality (r = 0.96); (2) fan-in exposes paradigm-specific collapse patterns, with flat retrieval degrading 3x faster than structured-evidence systems; and (3) full-context prompting fails uniformly, establishing evidence construction as a necessary capacity beyond raw corpus exposure.
>
---
#### [new 044] A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于教育领域方面情感分析任务，解决公开标注数据稀缺问题。通过构建合成基准数据集，验证模型性能，促进该领域研究。**

- **链接: [https://arxiv.org/pdf/2605.25502](https://arxiv.org/pdf/2605.25502)**

> **作者:** Yehudit Aperstein; Alexander Apartsin
>
> **备注:** 39 pages, 14 figures
>
> **摘要:** Educational aspect-based sentiment analysis (ABSA) can support course improvement, but public aspect-labeled student feedback remains scarce because educational reviews are private, institution-specific, and expensive to annotate. This study introduces a controlled synthetic benchmark for educational ABSA built from 10,000 synthetic course reviews with explicit train-validation-test splits and a 20-aspect pedagogical schema spanning instructional quality, assessment and course management, learning demand, learning environment, and engagement. The corpus is generated with sampled target labels, sampled nuance attributes, and a realism-tuned prompt refined through a three-cycle judge-editor procedure. On the resulting benchmark, local baselines with TF-IDF, two-step transformers, and joint encoders show that the task is nontrivial; the strongest untuned model, BERT, reaches a held-out detection micro-F1 of 0.2760, while a modest lower-rate BERT schedule improves this to 0.2930. Full-test GPT-based inference with gpt-5.2 reaches 0.2519 micro-F1 in zero-shot mode and 0.2501 with retrieval-based few-shot prompting, placing batch inference above the classical baseline and close to the compact joint encoders. A conservative external evaluation on 2,829 mapped student-feedback reviews from Herath et al. yields a micro-F1 of 0.4593 for BERT on a 9-aspect overlap, indicating partial synthetic-to-real transfer. Realism and faithfulness analyses are reported as generator diagnostics that clarify how the benchmark was stabilized and where label noise remains. The study therefore contributes a synthetic educational ABSA corpus, a documented generation procedure, and a reproducible benchmark setting for a domain in which public labeled data remain difficult to obtain.
>
---
#### [new 045] Generating Legal Commentaries from Case Databases via Retrieval, Clustering, and Generation
- **分类: cs.CL**

- **简介: 该论文属于法律信息处理任务，旨在从法院判决中自动生成法律注释。通过检索、聚类和生成技术，将判决转化为结构化评论，解决人工框架依赖问题。**

- **链接: [https://arxiv.org/pdf/2605.24534](https://arxiv.org/pdf/2605.24534)**

> **作者:** Max Prior; Niklas Wais; Matthias Grabmair
>
> **摘要:** We present a fully automated pipeline that transforms large collections of court decisions into legal commentaries for statutes - without providing any handcrafted doctrinal framework. Using 4.555 decisions of the German Federal Court of Justice that cite sections 242, 280, 812 and 823 of the German Civil Code (BGB), we extract paragraph-level chunks, summarize their reasoning, and derive keywords, which are embedded and clustered. For each cluster, an LLM generates headings and synthesizes citation-rich sections, which are then merged into coherent commentaries by four state-of-the-art LLMs. We evaluate along five dimensions - topical relevance, heading-match, citation faithfulness, cluster distinction and logical ordering - using both a human expert and an LLM-judge. Our results show that commentary-like argument mining from court decisions to generate reports that can be refreshed within minutes at minimal cost is feasible, yet they highlight limitations arising from restricted sources and the normativity of legal reasoning.
>
---
#### [new 046] Quantifying the Impact of Translation Errors on Multilingual LLM Evaluation
- **分类: cs.CL**

- **简介: 该论文属于多语言模型评估任务，旨在解决翻译错误对评估结果的影响问题。通过分析自动与人工标注的翻译错误，发现目标语言错误显著影响模型性能。**

- **链接: [https://arxiv.org/pdf/2605.24904](https://arxiv.org/pdf/2605.24904)**

> **作者:** Klaudia-Doris Thellmann; Bernhard Stadler; Michael Färber; Jens Lehmann
>
> **摘要:** Machine-translated benchmarks are widely used to assess the multilingual capabilities of large language models (LLMs), yet translation errors in these benchmarks remain underexplored, raising concerns about the reliability and comparability of multilingual evaluation. We address two practical gaps: (i) how well automatic MQM-style error spans from LLM judges and a span-aware QE baseline (xCOMET-XXL) match expert human span annotations on benchmark translations, and (ii) how strongly translation errors (as opposed to source-side issues in the English original) explain accuracy drops on translated benchmarks. We find that span agreement is non-trivial on naturally occurring benchmark translations, and that target-side translation errors are consistently associated with measurable, percentage-point drops in translated accuracy even after controlling for English correctness and source-side anomalies.
>
---
#### [new 047] Tool-Call Dependency Structure is Linearly Decodable in LLM Agent Residual Streams
- **分类: cs.CL**

- **简介: 该论文研究LLM代理运行时工具调用依赖图的表示问题，通过结构探针验证其在残差流中可线性解码，揭示了模型内部对动态执行结构的编码机制。**

- **链接: [https://arxiv.org/pdf/2605.25310](https://arxiv.org/pdf/2605.25310)**

> **作者:** Tianda Sun; Dimitar Kazakov
>
> **备注:** 16 pages, 7 figures
>
> **摘要:** Tool-using LLM agents produce trajectories whose calls form a directed dependency graph: earlier tool outputs supply arguments to later calls. Whether this execution structure is represented inside the model is unknown; prior structural probes have targeted static code or chain-of-thought text, not an agent's run-time call graph. A low-capacity edge probe on the residual stream of Qwen3-32B decodes the tool-call dependency graph well above both a Hewitt--Liang random-label control and a positional baseline. A counterfactual contrast between value corruption and structural perturbation indicates the signal tracks abstract topology rather than identifier values, and replicates under an independent, non-substring oracle. The non-positional component replicates on three further interactive multi-hop benchmarks and attenuates as call order alone becomes a sufficient proxy for dependency, vanishing in single-shot planning. Per-layer activation patching shifts the probe at a later, non-patched boundary, evidence that the representation propagates rather than passively reads out, though the realised tool call does not move. To our knowledge this is the first structural probe of an LLM agent's runtime tool-call dependency graph. Our claims concern representation, not behavioural control, and span two model families and one primary domain.
>
---
#### [new 048] Evidence-Linked Radiology Reporting: A Human-Supervised Reference Architecture for Structured Imaging Intelligence
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于医学影像报告结构化任务，旨在解决信息碎片化问题。提出一种人机协作的参考架构，整合多种标准与工具，提升报告准确性与可 reuse 性。**

- **链接: [https://arxiv.org/pdf/2605.25120](https://arxiv.org/pdf/2605.25120)**

> **作者:** Houman Kazemzadeh; Kamyar Naderi
>
> **备注:** Technical report, 27 pages, 2 figures, 12 tables, 1 listing; reference architecture paper; does not report clinical outcomes or validated diagnostic performance
>
> **摘要:** Radiology reports remain the primary mechanism by which imaging findings are communicated to clinical teams. However, much of the structured information behind these reports, including measurements, image evidence, prior comparisons, lesion identity, uncertainty, and terminology, often remains trapped in free text or fragmented across picture archiving and communication systems, radiology information systems, reporting workstations, worksheets, advanced visualization tools, and electronic health records. This paper proposes a human-supervised, evidence-linked reference architecture for structured radiology reporting. The framework combines exam-specific templates, speech-to-structure processing, measurement and segmentation capture, controlled AI-assisted drafting, and standards-based interoperability using DICOM, DICOM Structured Reporting, DICOM Segmentation, HL7 FHIR, RadLex, SNOMED CT, LOINC, and UCUM. The system is positioned not as an autonomous report generator, but as a structured intelligence layer for enterprise imaging that supports reviewed reporting, longitudinal comparison, clinical data reuse, governance, and integration with PACS, RIS, EHR, analytics, and registry workflows. The paper also discusses modality-specific deployment considerations, clinical safety risks, validation requirements, cybersecurity, privacy, quality management, and regulatory boundaries for AI-assisted radiology reporting systems.
>
---
#### [new 049] Selective Latent Thinking: Adaptive Compression of LLM Reasoning Chains
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型推理链压缩中的精度与效率平衡问题。提出SLT框架，选择性压缩冗余推理步骤，保留关键步骤，提升效率并保持准确率。**

- **链接: [https://arxiv.org/pdf/2605.25745](https://arxiv.org/pdf/2605.25745)**

> **作者:** Hui Xie; Jie Liu; Ziyue Qiao; Joaquin Vanschore
>
> **摘要:** Explicit chain-of-thought (CoT) reasoning substantially improves the reasoning ability of large language models (LLMs), but incurs high inference cost due to lengthy autoregressive traces. Existing latent reasoning methods offer a promising alternative, yet they often treat reasoning as uniformly compressible, causing precision-critical intermediate steps to be overly compressed and thereby degrading reasoning accuracy. In this work, we propose Selective Latent Thinking (SLT), a framework that selectively compresses redundant reasoning spans into latent representations while preserving precision-critical spans as explicit CoT within the same reasoning trajectory. Specifically, SLT first uses a lightweight decoder to anticipate a short upcoming reasoning span, and then applies confidence-based gating to determine the longest span that can be reliably compressed. The accepted span is encoded into a compact latent representation to improve reasoning efficiency, while uncertain or precision-critical reasoning remains in explicit CoT form to preserve accuracy. To learn this selective compression policy, SLT adopts a three-stage training strategy that combines span-level latent compression, reliability-aware future reasoning prediction, and trajectory-level reinforcement learning to optimize the trade-off between answer correctness and reasoning cost. Extensive experiments across four mathematical reasoning benchmarks demonstrate that SLT achieves 22.7\% higher accuracy than latent reasoning baselines at comparable compression ratios, while reducing reasoning chain length by 58.4\% with only 2.8\% accuracy degradation compared to explicit CoT,Our code can be found in this https URL.
>
---
#### [new 050] PennySynth: RAG-Driven Data Synthesis for Automated Quantum Code Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PennySynth，解决量子代码生成中LLM泛化能力不足的问题。通过RAG框架和定制知识库，提升代码生成准确性与结构有效性。**

- **链接: [https://arxiv.org/pdf/2605.25572](https://arxiv.org/pdf/2605.25572)**

> **作者:** Minghao Shao; Nouhaila Innan; Hariharan Janardhanan; Muhammad Kashif; Alberto Marchisio; Muhammad Shafique
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** The growing complexity of quantum programming frameworks has exposed a critical limitation in existing large language model (LLM)-based code assistants: general-purpose models hallucinate PennyLane-specific gate names, misplace device configurations, and produce structurally invalid circuits when faced with specialized quantum coding challenges. We present PennySynth, a retrieval-augmented generation framework that addresses this gap by conditioning LLM inference on a curated knowledge base of 13,389 PennyLane instruction-code pairs, built via a three-stage extraction, verification, and deduplication pipeline over official PennyLane repositories, community GitHub sources, and QHack competition archives. PennySynth introduces a code-aware embedding strategy using st-codesearch-distilroberta-base, trained for natural-language-to-code retrieval, increasing average retrieval cosine similarity from 0.45 to 0.726 compared to a general-purpose baseline. Evaluated across 74 challenges spanning three years of the QHack competition (2022, 2023, 2024), PennySynth achieves 64%, 68%, and 52% pass@5 on QHack 2022, 2023, and 2024, respectively, improving over Claude Sonnet 4.6 without retrieval by +28, +25, and +28 percentage points. We further introduce a quantum-adapted CodeBLEU metric that upweights qml.* token patterns and show that structural code similarity and functional correctness capture distinct aspects of quantum code quality. Controlled ablations reveal that code-aware embeddings are the primary driver of retrieval performance, while dataset expansion and source composition provide additional gains when retrieval quality is sufficiently precise.
>
---
#### [new 051] Mix-MoE: Improving Multilingual Machine Translation of Large Language Models through Mixed MoEs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言机器翻译任务，旨在解决大语言模型在并行语料微调中的参数干扰问题。通过提出Mix-MoE框架，分离语言模型和翻译专家，提升多语言翻译性能。**

- **链接: [https://arxiv.org/pdf/2605.24681](https://arxiv.org/pdf/2605.24681)**

> **作者:** Bo Li; Tianyu Dong; Shaolin Zhu; Deyi Xiong
>
> **备注:** Accepted by TASLP
>
> **摘要:** Large Language Models (LLMs) have shown great promise in multilingual machine translation (MT), even with limited bilingual supervision. However, fine-tuning LLMs with parallel corpora presents major challenges, namely parameter interference. To address these issues, we propose Mix-MoE, a mixed Mixture-of-Experts framework designed to train LLMs for multilingual MT. Our framework operates in two distinct stages: (1) post-pretraining with MoE on monolingual corpora, and (2) post-pretraining with MoE on parallel corpora. Crucially, we divide the MoE layers into two specialized groups: Language Model Experts (LM Experts) and Machine Translation Experts (MT Experts). LM Experts are designed to capture and retain the monolingual knowledge learned by the pre-trained LLM. MT Experts, on the other hand, are specifically trained to acquire and store bilingual translation knowledge. Furthermore, to facilitate effective interaction between these specialized experts and leverage potential underlying structural patterns in text, we introduce a routing mechanism enhanced by Fourier Transform features derived from model representations. The experimental results demonstrate that Mix-MoE excels in multilingual MT, significantly outperforming existing baselines and showing notable progress in mitigating parameter interference.
>
---
#### [new 052] End-to-End Intracortical Speech Decoding from Neural Activity
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于脑机接口中的语音解码任务，旨在无需外部语言模型实现端到端字符级解码。研究者提出一种基于Conformer的解码器，直接处理皮层神经信号，取得23.80%的字符错误率。**

- **链接: [https://arxiv.org/pdf/2605.24313](https://arxiv.org/pdf/2605.24313)**

> **作者:** Owais Mujtaba Khanday; Jose A. Gonzalez-Lopez; Marc Ouellet; Alberto Galdon; Gonzalo Olivares Granados
>
> **备注:** Accepted at Odyssey 2026 (Lisbon)
>
> **摘要:** Current high-performing intracortical speech neuroprostheses achieve low word error rates but typically rely on external language models during inference, increasing memory, computation, and latency. In this work, we investigate whether meaningful character-level decoding is achievable without such models. We propose an end-to-end Conformer-based neural decoder trained directly on intracortical recordings from a participant with amyotrophic lateral sclerosis (ALS). Without any external language model, the system achieves a character error rate (CER) of 23.80\% on held-out validation data. Analysis shows that performance variability is driven by inter-session signal degradation, while dominant errors arise from incorrect word boundary segmentation. These results demonstrate that effective character-level decoding is possible in a fully end-to-end framework, providing a strong neural signal for downstream linguistic processing.
>
---
#### [new 053] Universal Activation Verbalizer: A Unified Framework for Cross-Model Activation Explanation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于跨模型激活解释任务，解决现有方法仅能自解释的问题。提出UAV框架，通过共享解码器实现不同模型激活的统一解释。**

- **链接: [https://arxiv.org/pdf/2605.25903](https://arxiv.org/pdf/2605.25903)**

> **作者:** Haiyan Zhao; Zirui He; Guanchu Wang; Ali Payani; Yingcong Li; Mengnan Du
>
> **备注:** 23 pages, 11 figures, 11 tables
>
> **摘要:** Activation verbalization explains hidden representations in natural language, but existing methods are mostly limited to self-explanation, where each model explains only its own activations. We introduce Universal Activation Verbalizer (UAV), a framework that uses a shared decoder to explain activations from heterogeneous donor models. UAV learns a lightweight adapter that converts donor activations into soft tokens in decoder's embedding space, and further supports adapter-only transfer by reusing a frozen decoder-side LoRA while training only a new adapter for another donor. Across classification, fact retrieval, and gist summarization, UAV remains competitive with strong self-explanation baselines while enabling cross-model verbalization across model families and scales. Ablations show that decoder-side tuning mainly improves task behavior, whereas the adapter provides the activation-grounded factual and semantic information needed for faithful explanations.
>
---
#### [new 054] HyLaT: Efficient Multi-Agent Communication via Hybrid Latent-Text Protocol
- **分类: cs.CL**

- **简介: 该论文属于多智能体通信任务，解决传统方法在效率与可解释性间的矛盾。提出HyLaT协议，结合潜在空间与自然语言，提升通信效率与可读性。**

- **链接: [https://arxiv.org/pdf/2605.25421](https://arxiv.org/pdf/2605.25421)**

> **作者:** Xinyi Mou; Siyuan Wang; Zejun Li; Yulan He; Zhongyu Wei
>
> **摘要:** Communication protocol design is a central challenge in large language model-based multi-agent systems. Existing single-channel approaches face an inherent communication trilemma: text-based methods are interpretable but verbose, while latent-space methods are efficient but opaque and limited to unidirectional workflows. Inspired by multi-channel communication theory, we propose HyLaT, a hybrid latent-text communication protocol that transmits elaborate cognitive signals through a latent channel for efficiency, while expressing concise critical signals in natural language to preserve interpretability and precision. We introduce a two-stage training framework combining single-agent hybrid generation learning and multi-agent interactive co-training, enabling agents to generate and interpret hybrid messages across multiple rounds of interaction. Experiments demonstrate that HyLaT reduces communication overhead significantly while maintaining competitive task performance, with strong generalization and robustness across diverse settings.
>
---
#### [new 055] Clarification Is Not Enough: Post-Clarification Answering Remains the Bottleneck in Multi-Turn QA
- **分类: cs.CL**

- **简介: 该论文属于多轮问答任务，旨在解决用户意图不明确时的偏好获取问题。通过分解为澄清策略和澄清后回答两部分，发现澄清后回答仍是系统瓶颈。**

- **链接: [https://arxiv.org/pdf/2605.25204](https://arxiv.org/pdf/2605.25204)**

> **作者:** Jinyan Su; Jennifer Healey
>
> **摘要:** Pluralistic alignment requires systems to adapt to diverse user values, communication styles, and contextual assumptions. We believe that a foundational prerequisite for such alignment enabling accurate preference elicitation from people when their intent is under-specified or ambiguous. We study the problem of preference elicitation in multi-turn question answering by decomposing the problem into two components: a \textbf{clarification policy}, which decides whether to ask a clarifying question or answer directly, and \textbf{post-clarification answering}, which produces the correct final answer once the missing information is provided. We show, using the PACIFIC benchmark, that supervised fine-tuning rapidly improves the clarification policy, however, final answer accuracy remains substantially lower even when the model takes the correct action. This gap indicates that understanding and correctly interpreting the user's response is the critical gap in multi-turn question-answering systems.
>
---
#### [new 056] Knowing but Not Showing: LLMs Recognize Ambiguity but Rarely Ask Clarifying Questions
- **分类: cs.CL**

- **简介: 该论文研究大模型在面对模糊查询时识别歧义但不主动提问的现象，属于自然语言处理任务，旨在解决模型缺乏澄清行为的问题。**

- **链接: [https://arxiv.org/pdf/2605.25284](https://arxiv.org/pdf/2605.25284)**

> **作者:** Jinyan Su; Claire Cardie
>
> **摘要:** User queries are often underspecified and may admit multiple valid interpretations. Rather than silently making assumptions about the user's intent, a helpful assistant should surface such ambiguity by asking a clarifying question. Doing so requires two abilities: recognizing that a query is ambiguous, and acting on that recognition by seeking clarification instead of answering directly. To study these abilities, we evaluate models on ambiguous, unambiguous, and disambiguated questions in three settings: standard question answering, explicit ambiguity judgment, and behavioral analysis, where a judge model classifies responses as direct answers, refusals, or clarifying questions. We find a clear gap between recognition and behavior: models often identify ambiguity when explicitly asked to judge it, yet in the QA setting they overwhelmingly default to direct answers. Retrieved context further widens this gap by improving answerability while making models even less likely to ask clarifying questions.
>
---
#### [new 057] World-State Transformations for Neuro-symbolic Interactive Storytelling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于交互式叙事任务，旨在解决LLM导致的故事不连贯问题。通过神经符号架构，结合预设世界状态变换，提升叙事一致性与玩家创造力。**

- **链接: [https://arxiv.org/pdf/2605.24719](https://arxiv.org/pdf/2605.24719)**

> **作者:** Santiago Góngora; Luis Chiruzzo; Gonzalo Méndez; Pablo Gervás
>
> **备注:** To be presented at the 17th International Conference on Computational Creativity (ICCC'26)
>
> **摘要:** Large Language Models (LLMs) have changed the possibilities of Interactive Storytelling systems that process free-text user input. However, as more of these systems are built, evidence continues to mount regarding the story coherence problems that arise when relying solely on them. Recent research suggests that LLMs can effectively predict state changes within rule-based Interactive Storytelling systems, triggering pre-programmed world-state transformations. In this paper, we conduct an exploratory evaluation of whether such transformations can serve as a catalyst for player expression while aiming to address the incoherence issues typical of purely LLM-based approaches. Building upon a neuro-symbolic architecture, we conducted experiments using an open-source model (Llama 3 70B) and a closed-source model (Gemini 1.5 Flash), with testing conducted in both English and Spanish. Eight participants played two scenarios, carefully designed to assess different evaluation objectives. Our observations suggest that transformations offer a way to maintain world-state consistency while encouraging players to interact creatively through their written inputs.
>
---
#### [new 058] Structure-Aware RAG: Structured Retrieval Augmented Generation from Noisy Data for Conversational Agents
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于对话系统任务，旨在解决LLM在动态或领域信息中的可靠性问题。提出SA-RAG方法，利用结构化表格减少噪声，提升检索生成效果。**

- **链接: [https://arxiv.org/pdf/2605.24366](https://arxiv.org/pdf/2605.24366)**

> **作者:** Kaiqiao Han; LuAn Tang; Renliang Sun; Peng Yuan; Wei Cheng; Haoyu Wang; Wei Wang; Yizhou Sun; Haifeng Chen
>
> **摘要:** Large Language Models (LLMs) have been widely adopted in conversational applications. However, their reliance on parametric knowledge limits reliability in real-world scenarios that require dynamic or domain-specific information. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external knowledge during generation, but existing text-based and graph-based RAG methods often struggle with noisy or irrelevant contexts. In this work, we propose Structure-aware Retrieval Augmented Generation (SA-RAG), which uses tables as an intermediate structured representation to provide a compact and controllable interface that reduces noise while preserving essential information. We introduce a quality-aware table metadata generation framework that models metadata normalization and effectiveness, improving metadata quality and downstream performance. Furthermore, we explore both training-free and training-based table generation methods. Generation validation and direct preference optimization further improve table quality while maintaining semantic and structural consistency. Experiments on two noisy real-world datasets show that SA-RAG significantly outperforms existing RAG baselines. Our code is publicly available at a public repository.
>
---
#### [new 059] Document Classification Pattern Recognition via Information Fusion: A Systematic Review of Multimodal and Multiview Representation Approaches
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文档分类任务，旨在解决信息融合方法的统一框架和效果评估问题。通过系统综述139篇文献，提出框架、分析趋势并进行元分析，揭示融合方法的有效性及研究中的方法论缺陷。**

- **链接: [https://arxiv.org/pdf/2605.23910](https://arxiv.org/pdf/2605.23910)**

> **作者:** Marcin Michał Mirończuk
>
> **摘要:** Information fusion is used widely to improve document classification by the integration of multiple data sources (multimodal) or representations (multiview). However, the field lacks a unified framework, a quantitative synthesis of its effectiveness, and clear guidance for practitioners. This systematic review addresses these gaps by analysing 139 primary studies. It introduces a formal framework to structure the field, presents the results of a qualitative analysis to identify key trends, and performs a random-effects meta-analysis (to our knowledge, the first focused on document classification) to quantify performance gains. Our meta-analysis reveals that multimodal fusion improves accuracy (mean gain of +5.28 percentage points, $p=0.0016$) significantly -- the F1-score effect is directionally positive but statistically non-significant in our primary model. Multiview fusion provides consistent but modest gains for accuracy (+4.67\%), F1-score (+3.08\%), and recall (all $p<0.05$). Critically, our qualitative synthesis uncovers challenges in reproducibility in methodological rigour: only 11.8\% (multimodal) and 23.3\% (multiview) of the studies use statistical tests to validate their findings, which undermines the reliability of many of their results. This review's primary contributions are a unifying framework, the first quantitative evidence base, and data-driven guidelines. This review concludes that successful information fusion depends not on algorithmic complexity, but on the strategic alignment of the fusion method with the task context and a commitment to more rigorous validation.
>
---
#### [new 060] Beyond the Target: From Imitation to Collaboration in Speculative Decoding
- **分类: cs.CL**

- **简介: 该论文属于语言模型推理优化任务，解决传统推测解码中过度依赖目标模型的问题。提出CoSpec方法，通过协作机制提升解码效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.24793](https://arxiv.org/pdf/2605.24793)**

> **作者:** Jinze Li; Yixing Xu; Guanchen Li; Jinfeng Xu; Shuo Yang; Yang Zhang; Xuanwu Yin; Dong Li; Edith C.H. Ngai; Emad Barsoum
>
> **备注:** under review
>
> **摘要:** Speculative decoding (SPD) accelerates large language model (LLM) inference by letting a smaller draft model propose multiple future tokens that are verified in parallel by a larger target model. The dominant SPD paradigm treats the target model as the sole reliable teacher, accepting a draft token only when it exactly matches the target prediction. This design implicitly assumes that the target is always the better choice at every position. In practice, this assumption does not hold. Although the draft is the weaker model overall, it is not uniformly inferior at the token level. In a meaningful fraction of cases where draft and target disagree, the draft's choice is the one that leads to the correct final answer. Inspired by this, we introduce \textbf{Collaborative Speculative Decoding (CoSpec)}, a generalization of SPD that no longer treats the target model as the sole token-level authority. CoSpec trains an arbitration policy via reinforcement learning to decide whether to accept tokens from the draft or target model, selectively accepting draft tokens at mismatches when doing so is likely to yield a correct final answer. Experimental results show that CoSpec maintains substantial speedups while surpassing target-only performance. By shifting the emphasis from imitation to collaboration, CoSpec suggests a new perspective on speculative decoding.
>
---
#### [new 061] Knowledge Graph-Driven Expert-Level Reasoning for Neuroscience
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱与自然语言处理交叉任务，旨在通过单一权威教材构建知识图谱，提升神经科学领域的专家级推理能力。工作包括构建高质量KG、生成多跳问答数据，并用其微调语言模型，实现更高效精准的推理。**

- **链接: [https://arxiv.org/pdf/2605.25183](https://arxiv.org/pdf/2605.25183)**

> **作者:** Jake Stephen; Niraj K. Jha
>
> **摘要:** Knowledge graph (KG) is an abstraction that can be extracted from text corpora and used for in-depth reasoning. Prior work has leveraged KGs to fine-tune language models (LMs), enabling domain-specific superintelligence. In this work, we explore whether KG-driven in-depth reasoning capabilities can emerge in neuroscience using only information contained within a single authoritative textbook. The central hypothesis is that structured knowledge, when distilled into a high-quality KG and converted into KG-grounded question-answer (QA) supervision, is sufficient to produce expert-level reasoning through a fine-tuned LM that surpasses large language models (LLMs) in accuracy, while employing orders of magnitude fewer parameters. We construct a textbook-derived KG via a dual-LLM validation pipeline, expand it with a masked LM trained on the KG topology, generate multi-hop QA items, which include QA pairs and reasoning traces, to fine-tune an LM exclusively on KG-derived supervision, and apply reinforcement learning using path-derived KG signals as implicit reward models. Our results demonstrate that deep, mechanistic neuroscience understanding can be induced in the model without reliance on large, heterogeneous web-scale corpora. The KG-based synthetic neuroscience curriculum that readers can quiz themselves on, and the fine-tuned LM, are available at the following GitHub location: this https URL.
>
---
#### [new 062] NITP: Next Implicit Token Prediction for LLM Pre-training
- **分类: cs.CL**

- **简介: 该论文提出NITP方法，用于改进语言模型预训练。针对传统Next-Token Prediction（NTP）监督不足的问题，NITP引入连续表示空间的密集监督，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.24956](https://arxiv.org/pdf/2605.24956)**

> **作者:** Xiangdong Zhang; Debing Zhang; Shaofeng Zhang; Xiaohan Qin; Yu Cheng; Junchi Yan
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Standard next-token prediction (NTP) supervises language models solely through discrete labels in the output logit space. We argue that this sparse one-hot supervision leaves the latent representation space under-constrained, allowing hidden states to drift into degenerate and anisotropic configurations that can limit generalization. To address this issue, we propose Next Implicit Token Prediction (NITP), which augments discrete prediction with dense continuous supervision directly in the representation space. NITP trains the model to predict the implicit semantic content of the next token, using shallow-layer representations from the same model as stable self-supervised targets. We provide theoretical analysis showing that NITP regularizes the optimization landscape by mitigating under-constrained degrees of freedom and encouraging a compact, structured representation geometry. Empirically, across dense and MoE models ranging from 0.5B to 9B parameters, NITP consistently improves downstream performance with negligible computational overhead. On a 9B MoE model, NITP achieves a 5.7% absolute improvement on MMLU-Pro, along with gains of 6.4% on C3 and 4.3% on CommonsenseQA, with approximately 2% additional training FLOPs and no additional inference cost. Our implementation is available at this https URL.
>
---
#### [new 063] QUEST: Training Frontier Deep Research Agents with Fully Synthetic Tasks
- **分类: cs.CL**

- **简介: 该论文提出QUEST，一个开放的深度研究代理模型，解决通用任务泛化能力不足的问题。通过合成数据训练，提升事实检索与报告生成能力。**

- **链接: [https://arxiv.org/pdf/2605.24218](https://arxiv.org/pdf/2605.24218)**

> **作者:** Jian Xie; Tianhe Lin; Zilu Wang; Yuting Ning; Yuekun Yao; Tianci Xue; Zhehao Zhang; Zhongyang Li; Kai Zhang; Yufan Wu; Shijie Chen; Boyu Gou; Mingzhe Han; Yifei Wang; Vint Lee; Xinpeng Wei; Xiangjun Wang; Yu Su; Huan Sun
>
> **备注:** Work in Progress
>
> **摘要:** Deep research agents extend the role of search engines from retrieving keyword-matched pages to synthesizing knowledge, fundamentally changing how humans interact with information. However, frontier systems remain proprietary, while existing open agents often generalize poorly across different task types, leaving unclear how to train a broadly capable deep research agent. We release QUEST, a family of open models (ranging from 2B to 35B) that serve as general-purpose deep research agents designed to handle a wide range of long-horizon search tasks, with strong capabilities in fact seeking, citation grounding, and report synthesis. To build QUEST, we propose an effective training recipe combining mid-training, supervised fine-tuning, and reinforcement learning. Central to this recipe is a curated data synthesis pipeline based on unified rubric trees, which applies to different task types and enables synthesizing training data with verifiable rewards without human annotation. In addition, QUEST incorporates a built-in context management mechanism that enables effective long-horizon reasoning and knowledge synthesis. Using only 8K synthesized tasks, QUEST approaches or even surpasses frontier closed-source agents across eight deep research benchmarks spanning diverse task types, and achieves the best overall performance among recent open-weight agents. We released everything: models, data, and training scripts.
>
---
#### [new 064] WhenLoss: Diagnosing Write and Retrieval Bottlenecks in Long-Context Memory Systems
- **分类: cs.CL**

- **简介: 该论文属于长文本记忆系统研究，解决固定预算下信息丢失问题。通过诊断协议分析写入与检索瓶颈，提出EPC方法优化写入阶段信息保留，提升系统性能。**

- **链接: [https://arxiv.org/pdf/2605.24579](https://arxiv.org/pdf/2605.24579)**

> **作者:** Jiangnan Yu; Kisson Songqi Lin; Jilong Wu
>
> **备注:** 14 pages, 7 figures, 9 tables
>
> **摘要:** Long-context memory systems often fail under fixed budgets, but end-to-end evaluation does not reveal whether evidence was discarded during compression or preserved but never retrieved. We introduce a four-condition diagnostic protocol that evaluates a fixed reader under truncated full context (TFC), oracle evidence (OE), complete stored memory (CSM), and retrieved memory (RM). Under this fixed-budget LongMemEval setup, write-side gaps exceed retrieval-side gaps for most tested baselines, with four of six baselines robustly write-dominant under our default diagnosis margin. Motivated by this diagnosis, we propose Expected Predictive Compression (EPC), which moves the key decision--what information to retain--to write time by using an LLM to anticipate likely future questions and preserve the minimal supporting evidence under the token budget, while leaving retrieval unchanged at question time. Across all 500 LongMemEval questions with three readers (GPT-5.2, Claude Sonnet 4, Gemini 2.5 Pro), EPC achieves the highest CSM scores among all systems (0.49 vs. 0.44 for Summary (LLM), the strongest baseline), reducing Delta_write to 0.04 while leaving Delta_retr comparable to other LLM-based systems. These results suggest that, on this benchmark and evaluation setup, improving what the write stage preserves is a key avenue for performance gains in the tested systems.
>
---
#### [new 065] Language Models Need Sleep
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何提升语言模型处理长任务的能力，通过引入类似睡眠的机制，将近期信息转为持久权重，以优化注意力机制，改善长序列处理效果。**

- **链接: [https://arxiv.org/pdf/2605.26099](https://arxiv.org/pdf/2605.26099)**

> **作者:** Sangyun Lee; Sean McLeish; Tom Goldstein; Giulia Fanti
>
> **摘要:** Transformer-based large language models are increasingly used for long-horizon tasks; however, their attention mechanism scales poorly with context length. To handle this, we study a sleep-like consolidation mechanism in which a model periodically converts recent context into persistent fast weights before clearing its key-value cache. During sleep, the model performs $N$ offline recurrent passes over the accumulated context and updates the fast weights in its state-space model (SSM) blocks through a learned local rule. During inference, this shifts extra computation to sleep while preserving the latency of wake-time prediction. We test our method on controlled synthetic tasks, including cellular automata and multi-hop graph retrieval, as well as a realistic math reasoning task, on which a regular transformer as well as SSM-attention hybrid models fail. We then show that increasing sleep duration $N$ for our models improves performance, with the largest gains on examples that require deeper reasoning.
>
---
#### [new 066] TriVAL: A Tri-Validation Framework for Faithful Automatic Optimization Modeling
- **分类: cs.CL; cs.AI; eess.SY; math.CO**

- **简介: 该论文属于自动优化建模任务，旨在解决建模过程中缺乏有效验证导致的误差累积问题。提出TriVAL框架，在三个阶段进行显式验证，提升建模准确性。**

- **链接: [https://arxiv.org/pdf/2605.23966](https://arxiv.org/pdf/2605.23966)**

> **作者:** Ziyang Fang; JinXi Wang; Jinghui Zhong; Yew-Soon Ong
>
> **备注:** 13 pages
>
> **摘要:** Optimization modeling serves as the pivotal bridge between natural-language problem descriptions and optimization solvers, and remains a cornerstone for bringing operations research (OR) into real-world decision making. Recent advances in large language models (LLMs) have driven significant progress in automatic optimization modeling. However, existing methods still lack explicit validation during the modeling process, allowing errors introduced in earlier stages to carry through the pipeline and ultimately reduce final modeling accuracy. To address this challenge, we introduce TriVAL, a tri-validation framework that performs explicit validation at three stages of automatic optimization modeling: semantic specification, mathematical formulation, and code generation. At each stage, TriVAL follows a construct-validate-revise loop that assesses the current result against stage-specific criteria and revises it when needed. This design helps identify and correct errors before they accumulate across stages, helping preserve faithfulness throughout the modeling process. To evaluate automatic optimization modeling on more challenging combinatorial problems, we further introduce NL4COP, a benchmark of 150 instances across 50 diverse problem types with more complex decision logic, more tightly coupled constraints, and more demanding modeling requirements than existing benchmarks. Experiments on NL4COP and established benchmarks show that TriVAL consistently outperforms state-ofthe-art methods, with the largest gains on the most challenging problems.
>
---
#### [new 067] HiMed: Incentivizing Hindi Reasoning in Medical LLMs
- **分类: cs.CL**

- **简介: 该论文属于医疗自然语言处理任务，旨在解决 Hindi 医学推理能力不足的问题。通过构建 HiMed 数据集和模型，提升医疗 LLM 在 Hindi 语言中的表现。**

- **链接: [https://arxiv.org/pdf/2605.24635](https://arxiv.org/pdf/2605.24635)**

> **作者:** Dingfeng Jiang; Han Yan; Chenze Ma; Amit Kumar Jaiswal; Ang Li; Yunxiang Jiang; Xinlei Xiong; Juhao Liang; Hongru Xiao; Xiang Li; Fan Bu; Jiale Han; Ruchir Gupta; Prayag Tiwari; Benyou Wang
>
> **摘要:** Medical large language models hold promise for reducing healthcare disparities, yet Hindi remains severely underrepresented. While medical LLMs excel in high-resource languages, their performance degrades sharply in Hindi, particularly on Indian systems of medicine. We argue that robust cross-lingual medical transfer requires Hindi reasoning. To this end, we introduce HiMed, a Hindi reasoning medical corpus and benchmark suite covering both Western and Indian medicine. We further propose HiMed-8B, a Hindi-form medical reasoning LLM, through the design of decaying scaffolding reward. Extensive experiments demonstrate improvement in Hindi medical reasoning performance and reduction in the English--Hindi accuracy gap. Ablation studies validate the contribution of each training stage and reward component. All data and code are available on GitHub: this https URL.
>
---
#### [new 068] An Interactive Paradigm for Deep Research
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于深度研究任务，旨在解决传统框架缺乏交互控制的问题。提出SteER框架，实现可解释的中程控制，提升研究过程的灵活性和用户对齐度。**

- **链接: [https://arxiv.org/pdf/2605.24266](https://arxiv.org/pdf/2605.24266)**

> **作者:** Lin Ai; Victor S. Bursztyn; Xiang Chen; Julia Hirschberg; Saayan Mitra
>
> **摘要:** Recent advances in large language models (LLMs) have enabled deep research systems that synthesize comprehensive, report-style answers to open-ended queries by combining retrieval, reasoning, and generation. Yet most frameworks rely on rigid workflows with one-shot scoping and long autonomous runs, offering little room for course correction if user intent shifts mid-process. We present SteER, a framework for Steerable deEp Research that introduces interpretable, mid-process control into long-horizon research workflows. At each decision point, SteER uses a cost-benefit formulation to determine whether to pause for user input or to proceed autonomously. It combines diversity-aware planning with utility signals that reward alignment, novelty, and coverage, and maintains a live persona model that evolves throughout the session. SteER outperforms state-of-the-art open-source and proprietary baselines by up to 22.80\% on alignment, leads on quality metrics such as breadth and balance, and is preferred by human readers in 85\%+ of pairwise alignment judgments. We also introduce a persona-query benchmark and data-generation pipeline. To our knowledge, this is the first work to advance deep research with an interactive, interpretable control paradigm, paving the way for controllable, user-aligned agents in long-form tasks.
>
---
#### [new 069] Eureka: Intelligent Feature Engineering for Enterprise AI Cloud Resource Demand Prediction
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Eureka框架，解决企业AI云资源需求预测中的特征工程问题。通过LLM驱动的代码生成，提升特征质量与跨领域迁移能力。**

- **链接: [https://arxiv.org/pdf/2605.25297](https://arxiv.org/pdf/2605.25297)**

> **作者:** Hangxuan Li; Renjun Jia; Xuezhang Wu; Yunjie Qian; Zeqi Zheng; Xianling Zhang
>
> **备注:** 13 pages, accepted at DASFAA 2026 (International Conference on Database Systems for Advanced Applications)
>
> **摘要:** Effective features are crucial for predictive model performance, but creating them often requires domain expertise, limiting scalability across applications. We define feature engineering as an agentic code generation problem: features are not static data transformations, but executable programs that can be generated, evaluated, and iteratively improved. We present Eureka, an LLM-driven framework with three stages. (1) An Expert Agent, fine-tuned via SFT on domain knowledge, produces structured feature design plans in JSON format. (2) An LLM Feature Factory translates each plan into executable Python code through chain-of-thought reasoning, turning feature hypotheses into runnable programs. (3) A Self-Evolving Alignment Engine uses Reinforcement Learning (GRPO) with dual-channel reward (metric-based utility + semantic alignment) to enhance code quality. By expressing features as programs, the learned generation patterns can transfer across domains. Evaluated on 7 public benchmarks in healthcare, finance, and social domains, Eureka consistently outperforms both traditional AutoFE and LLM-based baselines. We further demonstrate Eureka's effectiveness on cloud GPU resource demand prediction at Alibaba Cloud, where Eureka improves demand fulfillment rate by 16% and lowers computing resource migration rates by 33%.
>
---
#### [new 070] Discovering Lexical Gaps Using Embeddings from Multilingual LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于跨语言词汇空缺识别任务，旨在解决多语言资源构建中的词汇缺失问题。通过分析双语大模型的嵌入空间，识别出跨语言语义对齐较弱的词汇，实现无需人工或分类体系的自动检测。**

- **链接: [https://arxiv.org/pdf/2605.24310](https://arxiv.org/pdf/2605.24310)**

> **作者:** Yoonwon Jung; Aaron S. Cohen; Benjamin K. Bergen
>
> **备注:** CoNLL 2026
>
> **摘要:** Lexical gaps are words that do not exist in certain languages. They pose challenges for building multilingual lexical resources, for machine translation, and for cross-lingual transfer. Existing lexical gap detection relies on human judgments or fixed conceptual taxonomies. We propose a data-driven framework for identifying cross-lingual lexical gaps. We extracted contextualized embeddings from Korean-English bilingual LLMs for Korean-to-English and English-to-Korean translation pairs. Combinations of LLMs, embedding types, dimensionality, and orthogonal transformations across 100 train-test splits yielded 4000 distinct embedding spaces in each source language. In each space, we computed the semantic similarity between each source word and its nearest neighbor in the target language, and compared their distribution for gap words versus non-gap words. In 94% (Korean-to-English) and 97% (English-to-Korean) of embedding spaces, gap words showed weaker cross-lingual semantic alignment than non-gap words. Logistic classifiers trained on unaligned embedding spaces can reliably separate gap words from non-gap words, achieving AUCs of 0.81 (Korean-to-English) and 0.76 (English-to-Korean) and retrieving 18/19 Korean and 26/27 English gap words. This approach provides a language-agnostic and taxonomy-free method for scalable lexical gap identification.
>
---
#### [new 071] Raon-Speech Technical Report
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文提出Raon-Speech和Raon-SpeechChat，解决语音理解和生成任务，通过多阶段训练提升模型性能并支持实时对话。**

- **链接: [https://arxiv.org/pdf/2605.23912](https://arxiv.org/pdf/2605.23912)**

> **作者:** Beomsoo Kim; Changho Choi; Dohyun Kim; Dongki Lee; Ethan Ewer; Eunchong Kim; Gyeongman Kim; Haechan Kim; Hyeonghwan Kim; Inkyu Park; Jihun Yun; Jihwan Moon; Jiyun Kim; Joonghyun Bae; Junhyuck Kim; Minkyu Kim; Sehun Lee; Seungjun Chung; Sungwoo Cho; Dongmin Park; Dongwon Kim; Hara Kang; Jonghyun Lee; Keon Lee; Kangwook Lee; Jaewoong Cho
>
> **摘要:** We present Raon-Speech, a top-performing 9B-parameter speech language model (SpeechLM) for English and Korean speech understanding, answering, and generation, and Raon-SpeechChat, a high-performing full-duplex extension for natural real-time conversation. Raon-Speech successfully transforms a pre-trained LLM into a SpeechLM that both understands and generates speech while preserving strong text capabilities. It trains on 1.38M hours of highly curated English and Korean speech and text datasets with the following training stages: (1) speech modules alignment, (2) end-to-end SpeechLM pre-training with knowledge distillation, and (3) multi-task preference optimization-based post-training. Across 42 English and Korean speech and text benchmarks, Raon-Speech establishes the strongest overall profile on speech-centric tasks in our comparison against eight similarly sized recent audio foundation models, including Qwen2.5-Omni and Fun-Audio-Chat, while preserving strong text question answering performance. Building upon it, Raon-SpeechChat enables natural full-duplex conversation by continual training on 119K hours of time-aligned real and synthetic dialogue data. It proceeds through three complementary training stages: (1) causal encoder adaptation, (2) full-duplex pre-training, (3) full-duplex fine-tuning for voice and role-control. On multiple full-duplex benchmarks, Raon-SpeechChat shows its clearest strengths on the turn-taking and interruption-sensitive behaviors covered by FDB v1.0, and remains competitive across the broader full-duplex evaluation suite. We open-source all model checkpoints, the training and inference pipeline, and an interactive demo.
>
---
#### [new 072] Extracting Training Data from Diffusion Language Models via Infilling
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文研究扩散语言模型中的训练数据泄露问题，提出通过填充提取方法更全面评估数据可提取性，揭示其比自回归模型更高的风险。**

- **链接: [https://arxiv.org/pdf/2605.24173](https://arxiv.org/pdf/2605.24173)**

> **作者:** Yihan Wang; N. Asokan
>
> **摘要:** Memorization in large language models has been studied almost exclusively through prefix-conditioned extraction, a natural choice for autoregressive models. However, diffusion language models (DLMs) can denoise masked tokens at arbitrary positions. Thus, prefix-only probing reveals only one facet of memorization in DLMs and significantly underestimates the risk of training-data extraction. In order to realistically model extractability of training data in DLMs, we introduce \emph{infilling extraction}, a data-extraction protocol parameterized by an arbitrary binary mask that subsumes prefix-only probing and accounts for the bidirectional inductive bias of DLMs. Instantiating it on LLaDA-8B and Dream-7B across five extraction modes, three training pipelines, and three corpora covering verbatim and partial leakage, we find that mask geometry governs extractability: edge-conditioned masks \emph{extract up to three times more} verbatim sequences than prefix-conditioned ones, and bidirectional access opens channels inaccessible in autoregressive models. In particular, we show that a realistic adversary with access to training data where personally identifiable information has been redacted, can even achieve higher recall on extracting redacted email addresses from DLMs than from scale-matched autoregressive models. Tunable parameters for decoding measurably affect extraction performance, while a follow-up supervised finetuning stage does not eliminate the prior memorization.
>
---
#### [new 073] Towards a Universal Causal Reasoner
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于因果推理任务，旨在提升大模型的因果推理能力。提出UniCo数据生成框架，解决现有数据不足和不真实的问题，通过高质量数据训练模型，显著提升其在多个领域的因果推理表现。**

- **链接: [https://arxiv.org/pdf/2605.24873](https://arxiv.org/pdf/2605.24873)**

> **作者:** Qirun Dai; Xiao Liu; Jiawei Zhang; Dylan Zhang; Hao Peng; Chenhao Tan
>
> **摘要:** Despite the importance of causal reasoning, training LLMs to reason causally remains underexplored. Existing data efforts mostly focus on benchmarking LLMs on specific aspects of causality, making them less suitable for training generalizable causal reasoners. To address this, we propose UniCo, a data generation framework that both (1) addresses 18 causal query types across Pearl's Causal Ladder and (2) translates natively symbolic examples into code and natural language forms to simulate real-world use cases where causal terms are not explicitly specified. To ensure data quality, UniCo grounds answers with exact causal inference and filters cases with reasoning shortcuts. Upon supervised finetuning with 66.6K UniCo-generated instances, Qwen3-4B, Qwen3-8B and Olmo-3-7B-Instruct achieve an average of 22.9% improvements across all 18 in-distribution query types, and 8.1% over state-of-the-art causal data generation frameworks on 7 established causal benchmarks outside the training distribution. More importantly, in real-world medical understanding, legal decision, and tabular reasoning, UniCo-trained models consistently display more faithful reasoning traces, outperforming the base models by an average of 20.2% in faithfulness metrics. These suggest that causality-centered training not only strengthens causal reasoning, but also equips LLMs with a causal mindset in general reasoning tasks.
>
---
#### [new 074] Reinforcement Learning from Denoising Feedback
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于强化学习任务，解决扩散语言模型中策略损失估计问题。提出RLDF方法，通过去噪反馈优化策略，提升模型性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.25638](https://arxiv.org/pdf/2605.25638)**

> **作者:** Qi He; Huan Chen; Ya Guo; Huijia Zhu; Yi R. Fung; Baojian Zhou
>
> **摘要:** Policy loss estimation remains a fundamental and long-standing challenge in reinforcement learning (RL) for diffusion language models (dLLMs). We introduce Reinforcement Learning from Denoising Feedback (RLDF), a novel training paradigm that leverages feedback obtained from rollout and training processes to facilitate accurate and efficient policy loss estimation. To balance the trade-off between computational efficiency and estimation effectiveness, RLDF optimizes the model toward the clipped clean state $\hat{x}_0$ from intermediate noisy states $x_t$, combined with weighted timestep sampling over $t$. Extensive experiments demonstrate that RLDF achieves consistent and substantial improvements in both performance and generalizability across two representative dLLM architectures, LLaDA and Dream, on multiple reasoning benchmarks. Our work lays a principled foundation for scalable reinforcement learning in diffusion language models. We build Drift, a training framework for dLLMs, available at this https URL.
>
---
#### [new 075] Word Class Representations Spontaneously Emerge from Successor Representations Trained on Natural Language
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文将强化学习中的后继表示应用于自然语言，通过预测未来词分布学习语言表征。任务是探索预测性序列学习对语法结构的自发生成，解决语法类别是否需显式编码的问题。**

- **链接: [https://arxiv.org/pdf/2605.24585](https://arxiv.org/pdf/2605.24585)**

> **作者:** Mathis Immertreu; Achim Schilling; Thomas Kinfe; Patrick Krauss
>
> **摘要:** Language models are typically trained to predict the next token in a sequence. Here, we explore an alternative predictive principle from reinforcement learning: Successor Representations (SRs), which model the expected discounted distribution of future states rather than the immediate next state. We transfer this framework to natural language and train neural networks to predict future word distributions across multiple temporal horizons, thereby learning representations of long-range transition structure. We train a deep residual neural network on WikiText-103 (103 million tokens; 20,000-word vocabulary) and optimize successor representations as probability distributions using KL divergence. Without explicit linguistic supervision, structured language representations emerge spontaneously. After training, the learned space develops a clear geometric organization with respect to part-of-speech (POS) categories: nouns, verbs, and adjectives become separable and recoverable through unsupervised clustering. This organization depends systematically on predictive horizon, with short horizons producing the strongest syntactic structure and longer horizons increasingly integrating broader contextual and semantic information. At finer resolutions, additional interpretable lexical substructure emerges, revealing coherent subclasses within major word categories. These findings suggest that syntactic categories need not be explicitly encoded but may arise as a consequence of predictive sequence learning. To our knowledge, this work provides the first systematic application of successor representations to natural language and establishes a conceptual bridge between reinforcement learning, linguistics, and cognitive neuroscience.
>
---
#### [new 076] Thaka at KSAA-2026 Task 2: Regularized Fine-Tuning for Arabic Speech Diacritization
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于阿拉伯语语音加注任务，旨在从语音和无注音文本中生成完整注音文本。工作包括使用CATT-Whisper模型并结合正则化技术提升性能。**

- **链接: [https://arxiv.org/pdf/2605.25928](https://arxiv.org/pdf/2605.25928)**

> **作者:** Meshal Alamr; Hassan Alqaeri; Abdullah Aldahlawi
>
> **备注:** 4 pages, 1 figure. Published in Proceedings of OSACT7 (LREC 2026). Winning system for KSAA-2026 Task 2 on Arabic Speech Diacritization
>
> **摘要:** We describe the winning system for Task 2 of the KSAA-2026 Shared Task on Arabic Speech Dictation with Automatic Diacritization. The task requires producing fully diacritized Arabic text from speech audio and undiacritized transcripts, with only 2,327 training samples available and no external data permitted. Our system fine-tunes CATT-Whisper, a character-level multimodal model combining a pretrained CATT text encoder with a frozen Whisper speech encoder. The key to our approach is training regularization: R-Drop consistency regularization, Optuna-optimized hyperparameters with high weight decay, and Focal Loss. At inference, we average 200 stochastic forward passes across four model checkpoints using Monte Carlo Dropout at the softmax probability level. The system achieves 23.26% WER on the primary leaderboard metric (with case endings, including no-diacritic positions), placing 1st among all participants.
>
---
#### [new 077] Automated Benchmark Auditing for AI Agents and Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于AI基准测试领域，旨在解决基准任务中隐含问题。通过自动化框架ABA检测任务设计缺陷，提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2605.26079](https://arxiv.org/pdf/2605.26079)**

> **作者:** Junlin Wang; Federico Bianchi; Shang Zhu; Fan Nie; Yongchan Kwon; Bhuwan Dhingra; James Zou
>
> **摘要:** Modern AI benchmarks operate at a complexity that outpaces traditional verification methods. Tasks authored by domain experts often contain implicit assumptions, incomplete environment specifications, and brittle evaluation logic that human annotation cannot reliably catch. We introduce Auto Benchmark Audit (ABA), an agentic framework that systematically audits individual benchmark tasks, uncovering issues such as hidden environment dependencies, specification gaps, and limited grading logic. We run ABA on a collection of frontier LLM benchmarks and previous NeurIPS publications, totaling 168 benchmarks across nine domains. Across this corpus, ABA identifies critical issues including ambiguous task design, execution environment conflicts, and incorrect ground truths in over 25.7% of the evaluated tasks. The precision of these automated audits is validated by expert review and independent third-party reports such as upstream PRs. Crucially, we demonstrate that these problematic tasks severely distorts capability assessments for agents and LLMs: filtering out these tasks with issues shifts model rankings and increases average performance on SWE-bench Verified and Terminal-Bench 2 by 9.9% and 9.6%, respectively. We release the agentic tool and all task annotations to support the future development of frontier benchmarks.
>
---
#### [new 078] Creative Quality Alignment: Expert Tacit Knowledge Transfer via Chain-of-Thought Fine-Tuning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型对齐任务，旨在验证创意质量度量的工程可行性。通过少量专家思维链数据，探索模型生成与评价的结构关联。**

- **链接: [https://arxiv.org/pdf/2605.25977](https://arxiv.org/pdf/2605.25977)**

> **作者:** Bo Zou; Chao Xu
>
> **摘要:** This paper provides an empirical implementation of the creative quality metric proposed in Calibrated Surprise (Zou & Xu, 2026a). The question this paper addresses is: does this mathematical claim hold at the engineering level? To make the answer as general as possible, we deliberately choose the strictest engineering conditions: low data cost and a small base model. Training data comes from approximately 100 expert chain-of-thought (CoT) annotations produced by the BC Protocol (Zou & Xu, 2026b). We also identify a data bias: most publicly available alignment datasets are skewed toward craft-related knowledge, while audience modeling and reality-logic coverage are systematically weak. We use the term Creative Quality Alignment (CQA) to describe this class of engineering methods. We also offer a supporting theoretical observation: in an LLM with a single conditional distribution architecture, calibrating the appreciation side automatically transfers to the generation side via architectural duality. This is the structural reason why ~100 CoT examples are sufficient -- not a purely empirical observation like LIMA (Zhou et al., 2023).
>
---
#### [new 079] SafeCtrl-RL: Inference-Time Adaptive Behaviour Control for LLM Dialogue via RL-Driven Prompt Optimisation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统安全控制任务，旨在解决LLM生成不安全内容的问题。通过强化学习动态优化提示，实现推理时的行为调节。**

- **链接: [https://arxiv.org/pdf/2605.25984](https://arxiv.org/pdf/2605.25984)**

> **作者:** Michael Orme; Yanchao Yu; Zhiyuan Tan
>
> **摘要:** Ensuring safe and contextually appropriate behaviour in Large Language Models (LLMs) remains a critical challenge for real-world deployment. We present \textbf{SafeCtrl-RL}, an inference-time behavioural control framework that enables adaptive safety regulation without model retraining or parameter modification. The method formulates dialogue generation as a sequential decision process, where a reinforcement learning agent dynamically selects prompt adjustment strategies based on contextual feedback. This allows unsafe behaviours to be suppressed through iterative refinement, which we conceptualise as inference-time behavioural unlearning. Evaluated across multiple LLMs and unsafe dialogue scenarios, SafeCtrl-RL consistently improves safety and response quality, outperforms existing prompt-based optimisation methods, and achieves favourable performance--efficiency trade-offs. **Warning: This paper may contain examples of harmful language, and reader discretion is recommended.
>
---
#### [new 080] MATO: Multi-objective Personalized Alignment with Test-time Optimization for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出MATO，解决大语言模型多目标个性化对齐问题。通过测试时优化，在不修改参数情况下动态调整目标权重，提升对齐效果与可控性。**

- **链接: [https://arxiv.org/pdf/2605.25342](https://arxiv.org/pdf/2605.25342)**

> **作者:** Linhao Luo; Thuy-Trang Vu; Van-Anh Nguyen; Junae Kim; Gholamreza Haffari; Dinh Phung
>
> **备注:** Preprint
>
> **摘要:** Aligning large language models (LLMs) with diverse and multifaceted user preferences is a fundamental challenge in personalized AI systems. Existing multi-objective alignment methods either rely on costly training or require pre-trained reward models for each preference, making it difficult for them to adapt to evolving preferences. Prompt-based personalization offers a training-free alternative, but prompting alone often provides limited steerability, as LLMs may overemphasize or overlook certain preferences and fail to give users reliable control over the relative importance of different objectives when conflicts arise, leading to suboptimal alignment. In this paper, we introduce MATO, a training-free framework for Multi-objective personalized Alignment with Test-time Optimization. MATO formulates personalization as a test-time optimization problem that steers the relative importance of multiple objectives through controllable weights during decoding, without modifying model parameters or requiring external reward models. Specifically, a reward discovery module recovers preference rewards directly from the backbone LLM for diverse objectives specified in natural language, while a weight optimization module dynamically adjusts objective weights based on the user's initial preferences and the partially generated response to balance competing objectives during generation. The resulting rewards and weights jointly guide an online optimization procedure over the token distribution, enabling better alignment with the target objectives. Extensive experiments across multiple datasets and backbone LLMs show that MATO consistently outperforms strong baselines, achieving Pareto-improving multi-objective alignment and stronger steerability. These results highlight test-time optimization as a promising direction for scalable, controllable, and model-agnostic personalized alignment.
>
---
#### [new 081] Translators as Invisible Teachers of AI: Copyright, Translation Memory, and the Political Economy of Linguistic Data
- **分类: cs.CL; cs.CY**

- **简介: 论文探讨AI训练数据中译者劳动的隐形价值，分析翻译记忆作为数据资本的法律与经济问题。属于数据伦理研究，解决译者权益被忽视的问题，提出“非消费性占有”和“隐形教师化”概念。**

- **链接: [https://arxiv.org/pdf/2605.24842](https://arxiv.org/pdf/2605.24842)**

> **作者:** Masaru Yamada
>
> **备注:** 13 pages; comments welcome
>
> **摘要:** This paper examines how the labour of translators has been transformed into foundational data capital for the age of artificial intelligence (AI). Translation memories (TM) and parallel corpora preserve a one-to-one correspondence between source and target text and therefore constitute extraordinarily valuable supervised training data for machine translation. The development of statistical machine translation (SMT), neural machine translation (NMT), the Transformer architecture, and multilingual large language models (LLMs) cannot be disentangled from the accumulation of such translation data. And yet, translators' renditions have been bought as deliverables under contract, segmented as technical objects, and processed as "information analysis" data under copyright law -- losing their moral, creative, and economic attribution to the translators who produced them. The paper develops two concepts to capture this process. The first is appropriation without consumption: a mode of use in which works are not read, viewed, or listened to, but only mined for statistical features -- a use that is legitimated under Article 30-4 of the Japanese Copyright Act. The second is the invisible teacherisation of translators: the process by which translators, through the construction of translation memories, post-editing, and quality assessment, have functioned as teachers of AI without recognition as such. Drawing on the data supply chain that runs from translators through language service providers (LSPs) and platforms to model developers, on a comparative reading of Japanese, European, and United States legal frameworks, on the distinction between open and proprietary AI models, and on the premium status that human-generated data has acquired in the era of model collapse, the paper asks what translators are actually afraid of, and points toward concrete directions for redistributive design.
>
---
#### [new 082] Testing the Deliteralization Hypothesis in Human and Machine Translation
- **分类: cs.CL**

- **简介: 该论文研究机器与人类翻译的非字面化现象，属于机器翻译任务。旨在验证翻译研究中的非字面化假设是否适用于大语言模型，通过实验对比不同系统的译文非字面程度。**

- **链接: [https://arxiv.org/pdf/2605.25686](https://arxiv.org/pdf/2605.25686)**

> **作者:** Malik Marmonier; Rachel Bawden; Benoît Sagot
>
> **摘要:** The recent shift from dedicated NMT systems to general-purpose LLMs has reshaped machine translation, with LLMs reported to produce more fluent, less literal output than their predecessors. We test whether this shift extends to the deliteralization hypothesis, the long-standing claim from translation studies that translations become progressively less literal as they are drafted and revised. Using the WMT24++ dataset, we compare the literality of human translations and post-editions to that of two NMT systems and six LLMs across 54 language pairs and three tasks: direct translation, iterative self-revision, and post-editing of human drafts. Literality is measured via a validated Synthetic Literality Index built from six heuristics. We find that (i) human translations remain significantly less literal than those of all tested MT systems, though recent LLMs narrow the gap; (ii) when prompted to iteratively revise their own output, LLMs deliteralize monotonically, providing the first evidence that the hypothesis applies natively to LLM generation; and (iii) as post-editors, LLMs invert the revision triggers of human post-editors, tolerating literal drafts and targeting idiomatic human formulations for revision.
>
---
#### [new 083] CP-Agent: A Calibrated Risk-Controlled Agent for Feedback-Driven Competitive Programming
- **分类: cs.CL**

- **简介: 该论文提出CP-Agent，解决竞赛编程中大语言模型性能不足的问题。通过反馈驱动机制提升代码正确率，引入三种验证机制优化效果。**

- **链接: [https://arxiv.org/pdf/2605.24693](https://arxiv.org/pdf/2605.24693)**

> **作者:** Peisong Wang; Bowen Liu; Zehua Li; Yuyao Wang; Zhiwei Ma; Yuhan Li; Jia Li
>
> **备注:** Code: this https URL
>
> **摘要:** Large language models still struggle with contest-level programming, while many agentic remedies rely on massive inference-time sampling or expensive multi-stage post-training. We study when execution feedback reliably helps an LLM CP solver and which mechanisms govern the gains. We model feedback-driven solving as a calibrated stopped process and identify three quantities: false-admission risk, program-level evidence against bad programs, and the active-state success hazard. Under held-out trace calibration and selection from a pre-declared finite controller manifest, the resulting structural certificate lower-bounds the clean success probability before false admission. We instantiate mechanisms targeting these quantities as Dual-Granularity Verification, Test Augmentation, and Experience-Driven Self-Evolving, yielding CP-Agent. Without updating any parameters, CP-Agent raises Pass@1 from 25.8\% to 48.5\% on LiveCodeBench Pro and improves Refine@5 by 11.0\% on ICPC-Eval. Across three LLM backbones, CP-Agent lies on the cost--accuracy efficiency frontier, and ablations show that each component primarily affects its corresponding certificate quantity.
>
---
#### [new 084] A general tensor-structured compression scheme for efficient large language models
- **分类: cs.CL; cs.AI; cs.LG; quant-ph**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型存储与计算开销过大的问题。通过引入Tensor Mixture（MixT）方案，用张量操作替代密集层，有效降低参数和计算量。**

- **链接: [https://arxiv.org/pdf/2605.25344](https://arxiv.org/pdf/2605.25344)**

> **作者:** Ying Lu; Peng-Fei Zhou; Qi-Xuan Fang; Pan Zhang; Shi-Ju Ran; Gang Su
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Large language models (LLMs) are dominated by dense linear transformations, whose storage, memory and computational overheads hinder efficient adaptation and deployment while masking the functional impacts of structural simplification. Here we present Tensor Mixture (MixT), a general tensor-structured compression scheme that replaces targeted dense linear layers with natively executable mixtures of tensor operators. Operating directly on generic linear projections instead of model-specific components, MixT is potentially applicable across Transformer-based LLMs and other dense neural mappings. We evaluate MixT on Qwen3-8B and LLaMA2-7B under a unified recovery protocol, identifying a broad compressible regime in which MMLU accuracy is largely preserved before an abrupt transition at model-specific boundaries. This transition coincides with coordinated shifts in output entropy, prediction entropy and inter-layer geometry. At the LLaMA2-7B transition boundary, MixT reduces full-model parameters by 47.5\%, inference FLOPs by 37.1\%, training FLOPs by 52.1\% and peak inference memory by 60.4\%, demonstrating its practical potential for lower-cost LLM compression.
>
---
#### [new 085] Toward a Benchmark for Controllable Simulation of Imperfect Students with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于教育技术任务，旨在通过语言模型模拟具有特定技能的学生，解决如何控制模型行为以反映指定技能配置的问题。工作包括构建框架、评估技能保留与遗忘情况。**

- **链接: [https://arxiv.org/pdf/2605.25601](https://arxiv.org/pdf/2605.25601)**

> **作者:** Alexander Apartsin; Omri Sason; Yehudit Aperstein
>
> **备注:** 22 pages, 7 figures
>
> **摘要:** Teacher education requires deliberate practice with learners who exhibit identifiable strengths, weaknesses, and partial mastery. Large language models could support such practice by simulating students with known skill components, enabling teachers to rehearse explanations, diagnoses, and instructional responses. For this purpose, however, the central requirement is neither to maximize benchmark accuracy nor to suppress isolated facts, but to control model behavior so that it reflects a specified skill profile. This paper investigates whether prompted language models can be steered to retain some skills while suppressing others. We introduce a benchmark-oriented framework in which an explicit skill vector represents a simulated student, prompt-based control specifies retained and missing competencies, and behavior is evaluated using profile-alignment metrics, retained-versus-forgotten comparisons, and cross-skill calibration analyses. The results show that selective partial mastery can be induced and measured in a structured mathematics setting, although the degree of controllability remains model-dependent. These findings position controllable learner simulation as a distinct research problem at the intersection of teacher education, educational simulation, and language-model control.
>
---
#### [new 086] TRACE: A taxonomy-grounded synthetic dataset for teaching-program generation and session interpretation in Applied Behavior Analysis
- **分类: cs.CL; cs.LG; cs.SE**

- **简介: 该论文提出TRACE数据集，用于解决ABA教学程序生成与行为分析的训练数据不足问题。属于自然语言处理任务，通过合成数据支持相关研究。**

- **链接: [https://arxiv.org/pdf/2605.25038](https://arxiv.org/pdf/2605.25038)**

> **作者:** Festus Kahunla
>
> **备注:** 11 pages, 3 tables. Dataset: this https URL ; code: this https URL
>
> **摘要:** Applied Behavior Analysis (ABA) is a clinical discipline whose documentation, teaching programs and multi-session behavioral logs, is formulaic and high-volume, yet real session data is HIPAA-protected and bound by professional confidentiality rules, blocking the release of a training corpus. We present TRACE (Taxonomy-Referenced ABA Clinical Examples), a 2,999-example synthetic instruction-tuning dataset covering two ABA tasks: teaching-program generation across Discrete Trial Training, Natural Environment Teaching, and Task Analysis; and multi-session behavioral interpretation across twelve trajectory patterns and thirteen target behaviors. Every example is produced by a deterministic taxonomy-driven generator grounded in the canonical ABA literature, and every example carries complete sampling provenance, the exact taxonomy cells that produced it. The dataset is released under CC BY-NC 4.0 for data and MIT for code, with stratified train (2,549), validation (149), test (281), and sanity (20) splits. TRACE is a research artifact and has not been clinically validated.
>
---
#### [new 087] QUIET: A Multi-Blank Cascaded Story Cloze Benchmark for LLM Creative Generation Capability
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出QUIET基准，用于评估大语言模型的创造性生成能力。针对现有评测方法主观或无法自动评分的问题， QUIET通过多空位故事补全任务，结合约束与依赖关系，实现客观自动化评分。**

- **链接: [https://arxiv.org/pdf/2605.25955](https://arxiv.org/pdf/2605.25955)**

> **作者:** Bo Zou; Chao Xu
>
> **摘要:** Large language models (LLMs) face a dual challenge in creative capability evaluation: existing benchmarks (e.g., Story Cloze Test, HellaSwag) measure models' discriminative ability over narrative continuation using multiple-choice recognition paradigms, rather than directly measuring creative generation capability; rubric-based scoring and LLM-as-Judge methods rely on subjective dimension assessment or natural language model outputs, and cannot provide objective, automated scoring mechanisms. This paper proposes QUIET (Quality Understanding via Interlocked Evaluation Testing), a diagnostic benchmark for LLM creative capability based on multi-blank cascaded story cloze. QUIET sets N blanks (10-20) in a story with complete structure, with each blank accompanied by an explicit content constraint, and cascade dependency relationships between blanks -- the content filled into earlier blanks constrains the feasible solution space for later blanks. The evaluated model (or human participants) fills all blanks in open-ended generation mode; the results are scored by an information-theoretic automated scoring protocol without human grading. The scoring protocol directly operationalizes the "calibrated surprise" theoretical framework (Zou & Xu, 2026a). For each blank k, a composite score is computed: score = satisfy * (1 + lambda * surprise), where lambda = 1.0. Here, "satisfy" measures how well the blank filling satisfies the content constraint (objective logical reasoning judgment, not subjective aesthetic scoring), and "surprise" measures the degree of surprise given that the constraint is satisfied. Creative answers that do not satisfy the constraint score zero; answers that satisfy the constraint but are mediocre score low; answers that satisfy the constraint and are surprising score high.
>
---
#### [new 088] EchoDistill:Alignment Noisy-to-Clean Self-Distillation for Robust Audio LLMs
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于音频大模型鲁棒性提升任务，解决噪声环境下语义漂移问题。提出EchoDistill框架，通过自蒸馏增强模型可靠性。**

- **链接: [https://arxiv.org/pdf/2605.23954](https://arxiv.org/pdf/2605.23954)**

> **作者:** Liang Lin; Chunxi Luo; Kaiwen Luo; Jie Zhang; Jin Wang; Yuanhe Zhang; Cai Yuchen; Qiankun Li; Gongli Xi; Zhenhong Zhou; Kun Wang; Junhao Dong
>
> **摘要:** Audio Large Language Models (ALLMs) are highly vulnerable to real-world noise, which often induces severe semantic drift and hallucinations. Existing robustness methods primarily rely on waveform-level acoustic enhancement, answer-level supervision, or the internal suppression of noise representations. To address these issues, we propose echodistill, an alignment-based noisy-to-clean self-distillation framework. Echodistill leverages a frozen clean-audio teacher to provide semantic references for an inference-time noisy-audio student. Specifically, the student samples candidate responses under noisy conditions to expose its test-time behavior. These trajectories are then optimized via group-relative policy optimization (GRPO), where the token-level consistency with the teacher acts as a reward bonus. By aligning the noisy student's candidate responses with clean semantic evidence, and applying audio-aware reward shaping, our method encourages reasoning trajectories that are both correct and genuinely acoustically grounded. Echodistill significantly improves the semantic reliability and task performance of Audio LLMs under complex noise, without introducing any additional inference costs. Extensive experiments show that: (I) Compared with the strongest baseline, echodistill achieves average improvements of 4.18\%$\uparrow$ in GSR under strong noise. (II) Ablation results on Qwen-Omni further show that echodistill improves over the GRPO-only variant by 3.02\%$\uparrow$ in Acc, 3.89\%$\uparrow$ in Noisy, and 4.53\%$\uparrow$ in GSR on average. Our codes are available at this https URL.
>
---
#### [new 089] Anticipate and Learn: Unleashing Idle-Time Compute in Proactive Agents
- **分类: cs.CL; cs.IR; cs.MA**

- **简介: 该论文提出ProAct架构，解决AI代理在空闲时间无法主动预判用户需求的问题。通过分析对话历史和记忆，提前获取信息，提升任务效率与准确性。属于智能代理领域。**

- **链接: [https://arxiv.org/pdf/2605.25971](https://arxiv.org/pdf/2605.25971)**

> **作者:** Haoyi Hu; Qirong Lyu; Xianghan Kong; Weiwen Liu; Jianghao Lin; Zixuan Guo; Yan Xu; Yasheng Wang; Weinan Zhang; Yong Yu
>
> **备注:** 26 pages, 4 figures; code available at this https URL
>
> **摘要:** While AI agents demonstrate remarkable capabilities in reasoning and tool use, they remain fundamentally reactive: they compute responses only after explicit user prompts. This paradigm ignores a critical opportunity: the idle time between interactions is largely wasted, leaving agents unable to prepare for future user needs. To bridge this gap, we introduce ProAct, a proactive agent architecture that leverages idle-time compute to anticipate and fulfill likely upcoming user needs. By analyzing evolving dialogue history together with persistent memory, ProAct predicts upcoming needs and iteratively acquires information, allowing the agent to resolve knowledge gaps and prepare evidence before the user initiates a this http URL rigorously evaluate proactive capabilities, we also introduce ProActEval, a comprehensive benchmark comprising 200 scenarios across 40 domains, featuring predictable need chains and diverse user cognitive profiles. Empirical results demonstrate significant advantages over reactive baselines. ProAct accelerates task completion by reducing required turns by 14.8%, decreases user effort by 11.7%, and cuts hallucination rates by 28.1% on ProActEval. Furthermore, MemBench evaluations confirm that ProAct achieves state-of-the-art reflective accuracy, underscoring its sustained and robust performance.
>
---
#### [new 090] The Path Matters: Learning a Token-Commitment Policy for Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，解决扩散语言模型中的token承诺问题。通过引入TraceLock控制器，学习可复用的承诺策略，提升生成质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.24697](https://arxiv.org/pdf/2605.24697)**

> **作者:** Bohang Sun; Max Zhu; Francesco Caso; Jindong Gu; Junchi Yu; Philip Torr; Pietro Liò; Jialin Yu
>
> **摘要:** Diffusion large language models promise faster generation by refining many token positions in parallel, but this parallelism introduces a hidden control problem: which proposed tokens should be transferred into the partially decoded sequence at each step? We refer to this decision as token commitment. Existing frozen-generator decoders largely rely on hand-designed confidence rules or block-specific acceptance filters. We argue that token commitment can instead be learned as a reusable trace-state policy. We introduce TraceLock, a lightweight plug-in controller that instantiates this policy for a frozen diffusion language model. Since oracle commitment times are unavailable, TraceLock derives self-supervision from future stability: at decoding step t, a proposed token for position i is labeled stable if it matches the final token at position i after the full decoding trace completes. The controller scores variable-length trace states and decides which active token proposals should be committed to the partially decoded sequence. Once trained for a given frozen backbone, the controller can be deployed across local-window widths, generation lengths, and step budgets without retraining or per-setting calibration. Experiments on question answering, mathematical reasoning, and code generation show that TraceLock improves the quality-step tradeoff over heuristic and learned baselines, with particularly stable behavior under cross-setting deployment. Diagnostic analyses show that its decisions are not reducible to scalar confidence, suggesting that frozen diffusion language models expose a learnable space of commitment trajectories beyond confidence-based decoding. Code is available at this https URL.
>
---
#### [new 091] PolyGnosis 2.0: Enhancing LLM Reasoning via Agentic Harness Engineering for Polymarket and OSINT Insight Extraction
- **分类: cs.CL; cs.CE**

- **简介: 该论文属于预测市场任务，旨在解决Perspective Mismatches问题，通过多代理架构融合Polymarket与OSINT数据，提升推理能力。**

- **链接: [https://arxiv.org/pdf/2605.25958](https://arxiv.org/pdf/2605.25958)**

> **作者:** Daren Wang; Hong Xu; Jiawen Xian
>
> **摘要:** This paper introduces PolyGnosis 2.0, a pioneering multi-agent architecture designed to extract predictive intelligence by synthesizing Polymarket anomaly signals with global Open Source Intelligence (OSINT) streams, specifically Global Database of Events, Language, and Tone (GDELT). We define and target "Perspective Mismatches", the narrative divergence between Polymarket sentiment and global media flows, as high-alpha trading signals. Moving beyond generic agentic superiority, we rigorously quantify the efficacy of "Harness Engineering" techniques, including reflection loops, tool-calling, divide-and-conquer partitioning (D&C), and chain-of-thought (CoT), within high-noise financial domains. Our empirical evaluation against human-expert benchmarks reveals that while structural partitioning is mandatory for multi-dimensional alignment, unconstrained terminal reflection actively induces logical drift. Furthermore, we identify a pervasive "consensus bias" across all agent configurations during narrative reasoning, necessitating deterministic validation. Ultimately, we isolate a Pareto-optimal configuration that achieves professional-grade analytical precision while minimizing latency and token overhead, providing a robust blueprint for autonomous intelligence in prediction markets.
>
---
#### [new 092] When In-Distribution Gains Fail: Evaluating Weak-to-Strong Reward Models under Preference Shift
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究W2S奖励模型在分布偏移下的泛化问题，发现模型在原分布表现好但跨数据集迁移失败。提出Representation Anchoring方法提升跨分布迁移能力。**

- **链接: [https://arxiv.org/pdf/2605.25629](https://arxiv.org/pdf/2605.25629)**

> **作者:** Khoi Le; Tri Cao; Phong Nguyen; Cong-Duy Nguyen; Anh Tuan Luu; Miao Chunyan; See-Kiong Ng; Thong Nguyen
>
> **备注:** Code: this https URL
>
> **摘要:** Weak-to-strong (W2S) generalization is a promising framework for scalable oversight, yet existing evaluations often test students under matched train--test distributions. Therefore, we study W2S preference learning under zero-shot distribution shift and find that strong students trained on weak preference labels can appear successful in-distribution while failing to transfer across preference datasets. We provide evidence for a representational failure mode in which weak-supervised fine-tuning can pull the strong model toward source-domain features instead of maintaining broadly transferable preference representations. To mitigate this, we propose Representation Anchoring (Anchor), a simple yet effective regularizer that constrains excessive drift from the pretrained strong model's representation space during fine-tuning, while still allowing task-relevant adaptation. Across preference domains, datasets, and model families, Anchor consistently improves out-of-distribution transfer while maintaining competitive in-distribution performance. Together, our evaluation protocol, transfer-aware metrics, and method expose hidden brittleness in current W2S reward modeling and provide a practical path toward more robust preference transfer.
>
---
#### [new 093] Know You Before You Speak: User-State Modeling for LLM Personalization in Multi-Turn Conversation
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决多轮对话中用户状态建模问题。通过PUMA框架，基于自由能原理实现用户状态动态建模与决策优化。**

- **链接: [https://arxiv.org/pdf/2605.24647](https://arxiv.org/pdf/2605.24647)**

> **作者:** Jiani Luo; Xiaoyan Zhao; Yang Zhang; Shuyi Miao; Bingbing Xu; Stefan Konigorski; Tat-Seng Chua
>
> **备注:** 30pages, 3 figures
>
> **摘要:** Personalized dialogue requires more than recalling explicit user histories: systems also need to infer hidden user states that evolve through interaction and shape appropriate response strategies. Existing memory- and profile-based methods primarily reuse observable user information, offering limited support for modeling user-state dynamics or selecting actions based on how they shape future user states. We propose PUMA (Prospective User-state Modeling for Action selection), a framework grounded in the Free Energy Principle (FEP) that formulates personalization as decision-making under partial observability, centered on an explicit user state model that captures latent user states and their action-conditioned dynamics. At each turn, PUMA maintains a belief over the user's hidden state, refines the user state model for observation generation and action-conditioned state transition, and selects dialogue actions by minimizing expected free energy, balancing epistemic and pragmatic objectives under a unified criterion. This formulation shifts personalization from passive memory retrieval to model-based decision-making over user evolution. We instantiate PUMA on healthcare-oriented counseling and motivational interviewing benchmarks with latent state annotations for rigorous evaluation. Experiments show that PUMA improves long-horizon dialogue outcomes while maintaining strong response quality, and a cross-dataset study demonstrates more reliable user-state estimation and next-state prediction.
>
---
#### [new 094] Faithful or Fabricated? A Causal Framework for Rationalization Bias in LLM Judges
- **分类: cs.CL**

- **简介: 该论文研究LLM作为评价者时的理性化偏差问题，属于文本评估任务。通过设计干预实验，分析其判断是否受非证据线索影响，并提出改进方法提升判断的客观性。**

- **链接: [https://arxiv.org/pdf/2605.23970](https://arxiv.org/pdf/2605.23970)**

> **作者:** Riya Tapwal; Abhishek Kumar; Carsten Maple
>
> **摘要:** Large language models (LLMs) are increasingly used as automatic judges for summarization and dialogue evaluation. Prior work has documented biases such as position, verbosity, and style preferences, but largely focuses on outcomes, leaving judge explanations underexplored. We instead ask whether LLM judges are cue-invariant, i.e., whether their rankings and explanations remain stable when non-evidential cues are perturbed while holding the underlying texts fixed. We introduce a suite of cue interventions (Blind, Truth, Flip, Placebo, Reveal-After) and tie-aware metrics that quantify outcome anchoring and rationale anchoring, including label-aligned rhetoric and explanation drift, alongside consistency and stereotype-intrusion checks. We design anchoring attacks using verbosity and confidence cues, and compare two mitigations: structured chain-of-thought prompting and PROOF-BEFORE-PREFERENCE (evidence lock, score, rank). Using a new dataset of 1,000 summaries from traditional extractive models and LLMs, we find substantial cue-anchored rationalization under label and placebo perturbations, while PROOF-BEFORE-PREFERENCE markedly improves cue invariance over baselines.
>
---
#### [new 095] The Age of Curiosity Meets the Age of AI: Benchmarking Child Safety in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于AI安全评估任务，旨在解决LLM在儿童使用中的安全性问题。通过构建KIDBench基准，评估不同提示对儿童友好响应的影响，并提出两个模型提升儿童安全交互。**

- **链接: [https://arxiv.org/pdf/2605.25510](https://arxiv.org/pdf/2605.25510)**

> **作者:** Samee Arif; Angana Borah; Rada Mihalcea
>
> **摘要:** Children increasingly have access to Large Language Models (LLMs), which may expose them to responses that are developmentally inappropriate or require age-sensitive safety, guidance, and boundaries. Existing LLM safety evaluations largely focus on harmful-content avoidance and do not explicitly target child-facing safety. We introduce KIDBench, a benchmark for evaluating child-facing LLM safety for ages 7--11 using a developmental-psychology-grounded LLM-as-a-Judge rubric. KIDBench contains realistic child queries across ten categories, with single-turn prompts and multi-turn child-actor simulations. We compare no-cues prompts with no child context, implicit-cues prompts that suggest a child speaker, and explicit age instructions. Implicit-cues improve scores by 9--47% across models, while explicit age adds a further 10--30% gain. Cross-lingual and cultural evaluations show uneven safety behavior across languages and country contexts. Multi-turn simulations show that child-facing response quality can degrade by 6--24% from the first to worst turn. Beyond evaluation, we introduce KIDGuardLlama, a child-safety evaluator, and KIDLlama, a child-oriented response model, showing how KIDBench supports safer child-facing AI
>
---
#### [new 096] Peak-Then-Collapse and the Four Interface Channels of Knowledge-Graph Tool Use
- **分类: cs.CL**

- **简介: 该论文研究知识图谱工具使用中的峰后崩溃现象，探讨四种接口通道的失败模式，旨在提升工具调用的准确性与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.26037](https://arxiv.org/pdf/2605.26037)**

> **作者:** Tianda Sun; Dimitar Kazakov
>
> **备注:** 18 pages, 9 figures
>
> **摘要:** We test the standard RLVR tool-use recipe -- GRPO on Qwen2.5-7B-Instruct -- on a deliberately minimal knowledge-graph tool API: four Freebase navigation verbs over Complex WebQuestions. Under a self-verifiable retrieval reward, the policy's tool-grounded answer rate climbs from $3.8\%$ to $9.6\%$ over 250 steps, then collapses to $0\%$ within a single 50-step window -- a \emph{peak-then-collapse} pattern replicated across four seeds. Across seven reward designs, we find four recurring failure modes: adding denser or more targeted proxy rewards shifts the failure mode rather than eliminating it. We argue that a key difference from Python interpreters, web search, and JSON APIs is interface feedback: their failures often leak natural-language signal the model saw in pretraining. A Python traceback names the failing line; an empty Freebase result \texttt{[]} does not. Stripping away that surface exposes a degradation regime that same-family reward redesigns do not fix. A direct oracle ablation rules out relation selection: injecting gold relations at every retrieval call lifts exact-match accuracy by only $+0.20$~pp, and $95.4\%$ of retrieval-dependent errors are retrieval-composition failures rather than answer-extraction failures. As a mitigation, one-iteration self-distillation reaches $40.0\%$ EM at 7B and is capacity-invariant: doubling capacity to 14B improves EM by only $0.25$~pp, and initialization barely matters -- the ceiling appears interface-bound within the 7B--14B range tested.
>
---
#### [new 097] Proactive for Uncertainty: Cause-Aware Error Diagnosis and Interactive Clarification for Spoken Dialogue Systems
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音对话系统任务，解决误差传播问题。通过因果感知的错误诊断与交互澄清，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.25404](https://arxiv.org/pdf/2605.25404)**

> **作者:** Yizhou Peng; Ziyang Ma; Changsong Liu; Yi-Wen Chao; Xie Chen; Eng Siong Chng
>
> **摘要:** Cascaded Automatic Speech Recognition -- Large Language Model (ASR-LLM) pipelines remain popular for industrial Spoken Dialogue Systems (SDS), primarily because their decoupled design ensures perceptual verifiability. However, cascaded systems suffer from error propagation, as transcription failures inevitably cascade to subsequent components, thereby degrading the final interaction quality. Although ASR confidence scores offer a simple filter for unreliable inputs, this approach is fundamentally limited because it typically fails to detect deletion errors or to distinguish between acoustic (inability to hear clearly) and linguistic (inability to understand) mismatches, both of which require targeted recovery strategies. In this paper, we propose a cause-aware error recovery paradigm that fundamentally rethinks robustness in SDS. Unlike traditional confidence filtering, we introduce a suite of small precision-focused detectors that exploit deep ASR latent representations to disentangle token-level errors into perception, comprehension, and deletion failures. This fine-grained diagnostic intelligence empowers the LLM to orchestrate targeted, multi-turn clarification strategies, effectively transforming ambiguous signals into seamless user interactions. Experimental results validate the precision of our approach, which more than doubles the recall on domain-shift errors (57.96% vs. 23.66%) compared to baselines. Crucially, this diagnostic precision yields up to a 30% reduction in WER and a 17% improvement on the downstream task across diverse accents, distortions, and domains.
>
---
#### [new 098] AI-Assisted Systematization for Evaluating GenAI Systems
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 论文探讨如何通过AI辅助系统化评估生成式AI系统，解决概念模糊导致的评价难题。提出概念规范和验证表，开发两种AI系统化工具，并验证其效果。**

- **链接: [https://arxiv.org/pdf/2605.26001](https://arxiv.org/pdf/2605.26001)**

> **作者:** Dhruv Agarwal; Emily Sheng; Chad Atalla; Jean Garcia-Gathright; Hussein Mozannar; Hannah Washington; Alexandra Chouldechova; Solon Barocas; Hanna Wallach
>
> **摘要:** Evaluating generative AI (GenAI) systems is challenging because many targets of evaluation are broad, contested concepts, such as "reasoning," "fairness," or "creativity." When these concepts are left underspecified, it becomes unclear what should be measured or how evaluation results should be interpreted. This problem reflects a missing step: systematization, that is, moving from a broad background concept to an explicit, structured account of the concept in measurable terms. To help address the fact that systematization is cognitively demanding and resource-intensive, we investigate whether AI assistance can support this process. To enable AI-assisted systematization and assess its quality, we introduce a structured representation of a systematized concept, a concept spec, and a validation worksheet. We then develop two AI-assisted systematizers: a direct, zero-shot approach and a multi-agent approach that more closely mirrors manual systematization approaches from existing literature. We use these systematizers to produce concept specs for two concepts -- hate-based rhetoric and digital empathy -- and evaluate resulting concept specs on content validity and information recoverability.
>
---
#### [new 099] Multilingual Phonological Feature Recognition with Self-Supervised Speech Models
- **分类: cs.CL**

- **简介: 该论文属于语音特征识别任务，旨在直接预测多语言的音系特征。工作包括构建PhonoQ-2.0系统，采用自监督模型并引入条件门控机制，提升特征预测准确率。**

- **链接: [https://arxiv.org/pdf/2605.25596](https://arxiv.org/pdf/2605.25596)**

> **作者:** Abner Hernandez; Tomás Arias-Vergara; Daiqi Liu; Andreas Maier; Paula Andrea Pérez-Toro
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Phonological features provide a language-general and linguistically grounded representation of speech. We present PhonoQ-2.0, a multilingual frame-level phonological feature recognizer built on self-supervised speech models. The system directly predicts a structured 22-dimensional feature vector per frame encoding manner, vowel quality, place, and voicing, instead of deriving features from phoneme outputs. To ensure phonologically coherent predictions, we introduce a manner-conditioned gating mechanism that activates valid feature groups. Evaluated across multiple languages and corpora, PhonoQ-2.0 achieves an average macro-F1 of 91.3% in-domain and 88.9% out-of-domain. Compared to a strong CTC phoneme baseline, it delivers consistent gains of +8.8 F1 in-domain and +8.6 out-of-domain on average. In unseen-language evaluation, PhonoQ-2.0 improves macro-F1 from 66.9% to 73.6% (+6.7 on average), with gains of up to +10.8 points.
>
---
#### [new 100] From Automation to Collaboration: Human-in-the-Loop Methods for Safe and Trustworthy NLP
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在提升模型的安全性与可信度。通过人机协作方法解决模型偏差、幻觉和鲁棒性不足等问题，提出改进审计、评估和数据构建的策略。**

- **链接: [https://arxiv.org/pdf/2605.25226](https://arxiv.org/pdf/2605.25226)**

> **作者:** Most. Sharmin Sultana Samu; MD. Tanvir Ahmed Seum; Md. Rakibul Islam
>
> **备注:** Preprint, manuscript under review
>
> **摘要:** Large language models are widely deployed in high-stakes NLP tasks, yet risks such as bias, hallucination, adversarial vulnerability and unreliable generalization remain. Probe-based auditing reveals inconsistencies in model behavior. Adversarial text generation uncovers robustness gaps, especially in lower-resourced languages with limited benchmarks. Enterprise text-to-SQL settings expose the difficulty of validating outputs over private and large-scale databases. Human supervision is essential for probe validation, adversarial verification and domain-specific annotation, but it is costly and hard to scale. This survey examines recent human-in-the-loop methods that shift NLP from automation toward collaboration for safety and trustworthiness. We review how human expertise supports auditing, robustness evaluation, data construction and model steering. Our findings highlight gaps in scalable probing, sustainable robustness benchmarks, low-resource settings and governance of private systems. We outline practical research directions for adaptive auditing, collaborative evaluation and accountable deployment.
>
---
#### [new 101] SEAL: Synergistic Co-Evolution of Agents and Learning Environments
- **分类: cs.CL**

- **简介: 该论文提出SEAL框架，解决LLM代理与环境协同进化问题，通过联合优化代理和环境提升学习效果。**

- **链接: [https://arxiv.org/pdf/2605.24426](https://arxiv.org/pdf/2605.24426)**

> **作者:** Yihao Hu; Zhihao Wen; Xiujin Liu; Pan Wang; Xin Zhang; Wei Wu
>
> **摘要:** Large Language Model (LLM) agents are increasingly improved through interaction, yet most self-evolution methods adapt either the policy or the learning environment in isolation. We identify this structural gap as \emph{Agent-Environment Misalignment}: the agent's capability frontier changes during training, while the environment that provides supervision remains static or only weakly coupled to the agent's revealed failures. We propose SEAL, a closed-loop co-evolution framework for interactive tool-use agents. SEAL collects on-policy trajectories under executable verification, diagnoses failed rollouts into turn-level failure labels, and uses these diagnoses as a shared signal for both environment-side adaptation and model-side policy optimization. The environment evolves its training-time learning interface by exposing clearer tool affordance cues, constraint information, and recovery-oriented feedback, while the policy is updated with diagnosis-guided advantage reweighting. Extensive experiments across in-distribution and out-of-distribution multi-turn tool-use evaluations show that SEAL improves low-resource agent learning: with only 400 training samples, it yields +8.25 to +26.25 average-point gains across three backbones and exhibits positive out-of-distribution transfer. These results demonstrate the value of jointly adapting the learner and its training-time learning substrate for robust self-improving LLM agents.
>
---
#### [new 102] P1SCO: Social Dimensions from a Perspectivist Lens
- **分类: cs.CL**

- **简介: 该论文介绍P1SCO数据集，用于分析社交媒体评论中的社会维度。属于社会感知研究任务，旨在探讨平台、个体差异和人口因素对社会互动的影响。**

- **链接: [https://arxiv.org/pdf/2605.25312](https://arxiv.org/pdf/2605.25312)**

> **作者:** Amanda Cercas Curry; Gianmarco de Francisci Morales; Luca Maria Aiello
>
> **摘要:** We introduce P1SCO, a dataset of social media comments collected from three distinct platforms, annotated according to ten social dimensions to capture the diversity of social interactions and perceptions. The dataset is carefully disaggregated to allow analysis at the level of individual comments, annotators, and platforms. In addition to the social dimension labels, we include rich metadata on the annotators, including demographics, Big Five personality profiles, and political affiliation. This combination of comment-level annotations and annotator-level features enables nuanced analyses of how social perception varies across platforms, individual differences, and demographic factors. By preserving the diversity of annotator perspectives, our dataset supports studies of inter- and intra-annotator agreement, the influence of personality and political orientation on social interpretation, and the cross-platform dynamics of social discourse.
>
---
#### [new 103] Forgotten Words: Benchmarking NeoBERT for Dementia Detection in Low-Resource Conversational Filipino and English Speech
- **分类: cs.CL**

- **简介: 该论文属于临床NLP任务，旨在解决低资源语种（如菲律宾语）痴呆检测问题。通过构建双语数据集并评估多种模型，发现双语微调可显著提升性能。**

- **链接: [https://arxiv.org/pdf/2605.26007](https://arxiv.org/pdf/2605.26007)**

> **作者:** Rez Samantha Z. Floresca; Edric Castel C. Hao; Hannah Grachiella Buñales; Chelsea Dominique E. Temprosa; Georgianna Z. Reyes; Kervin Gabriel L. Chua
>
> **备注:** Accepted to BioNLP Workshop @ ACL 2026
>
> **摘要:** Dementia detection from spontaneous speech offers a scalable approach to cognitive screening, yet NLP systems remain predominantly English-centric. This limitation is especially acute in the Philippines, where Filipino-English code-switching is pervasive and no prior work has addressed NLP-based dementia detection. We present the first systematic evaluation of transformer-based dementia detection in Filipino speech and the first assessment of NeoBERT in a clinical NLP setting. To separate language from domain effects, we construct a parallel bilingual dataset of 4,000 DementiaBank-derived transcripts, with Filipino translations produced manually to preserve discourse-level markers of cognitive decline. We evaluate five model families, TF-IDF + LogReg, BERT, NeoBERT, XLM-R, and RoBERTa-Tagalog, under monolingual, zero-shot cross-lingual, and bilingual fine-tuning settings. We find that in-domain performance does not transfer across languages, with English-trained BERT dropping to Macro-F1 = 0.455 on Filipino, and that architectural modernization alone does not improve robustness. Bilingual fine-tuning, however, eliminates cross-lingual degradation across all transformer models, converging to Macro-F1 = 0.969-0.973. These results suggest that multilingual clinical NLP performance is driven primarily by linguistic coverage during training rather than model scale or architecture.
>
---
#### [new 104] Is Inference Mediated by Distinct Semantic Structures in LLMs? A Mechanistic Interpretation
- **分类: cs.CL**

- **简介: 该论文研究自然语言推理中LLM是否通过不同的语义结构进行推断，旨在解决模型是否编码推理过程的问题。通过分析激活空间验证了语义操作的可解码性与因果影响。**

- **链接: [https://arxiv.org/pdf/2605.25520](https://arxiv.org/pdf/2605.25520)**

> **作者:** Nura Aljaafari; Marco Valentino; André Freitas
>
> **备注:** 26 pages, 16 figures, 13 tables
>
> **摘要:** Predicting a label correctly does not necessarily require representing the operation that produces it. Transformer representations are known to carry label-level information, but whether they encode semantic operations producing those labels is unclear. We investigate this in Natural Language Inference using controlled premise-hypothesis pairs that differ by a single semantic transformation. Using layer-wise activations, we estimate operation-level subspaces via SVD and test their causal relevance through activation steering in four open-weight decoder models. Transformation effects are decodable with $84.8$-$99\%$ accuracy and occupy partially distinct but overlapping subspaces, exceeding random-subspace baselines. Steering experiments show that these directions causally influence predictions, though steerability varies across models; cross-operation steering further reveals structured interference and a dissociation between subspace selectivity and cross-operation independence. These findings indicate that the models encode not only that a hypothesis relates to a premise but also, in part, how it does so, implying that mechanistic analysis and control should operate at the level of semantic operations rather than predicted labels alone.
>
---
#### [new 105] Exploring Profiles of Cognitive Distortions Associated with Mental Health Disorders
- **分类: cs.CL**

- **简介: 该论文属于心理计算研究任务，旨在探索不同心理健康障碍中的认知扭曲模式。通过分析Reddit文本数据，比较各群体的扭曲程度，发现心理健康群体普遍存在更高水平的认知扭曲。**

- **链接: [https://arxiv.org/pdf/2605.24996](https://arxiv.org/pdf/2605.24996)**

> **作者:** Alina Anikejeva; Kairit Sirts
>
> **备注:** CLPsych 2026
>
> **摘要:** Cognitive distortions, distorted patterns of thinking, have been increasingly studied in computational mental health research. Although they are related to many, if not all, mental health disorders, most existing studies focus primarily on depression. In this work, we explore distortion profiles across multiple mental health conditions. We analyzed a large Reddit-based dataset containing posts from nine self-reported mental health groups as well as a control group using both an n-gram-based method and a fine-tuned transformer model for detecting cognitive distortions. Mental health groups, both when pooled together and when examined individually, showed higher prevalence of cognitive distortions compared to the control group, with the effect sizes ranging from small to moderate. When comparing distortion profiles across conditions, we observed largely similar patterns, although some groups exhibited overall higher levels of distortions than others. These findings suggest that relatively simple lexical approaches can be useful for exploratory analyses of group-level trends in large-scale mental health text data.
>
---
#### [new 106] Mitigating Provenance-Role Collapse in Long-Term Agents via Typed Memory Representation
- **分类: cs.CL**

- **简介: 该论文属于长期记忆管理任务，解决LLM代理中因存储方式导致的来源混淆问题。提出MemIR结构，通过类型化记忆表示实现精准源监控。**

- **链接: [https://arxiv.org/pdf/2605.25869](https://arxiv.org/pdf/2605.25869)**

> **作者:** Zhengda Jin; Bingbing Wang; Jing Li; Ruifeng Xu; Min Zhang
>
> **摘要:** Long-term memory is essential for persistent LLM agents, yet prevailing architectures store historical interactions as unstructured, flat text. This unconstrained storage induces provenance-role collapse, a critical failure mode where agents suffer from source-monitoring errors. To resolve this cognitive vulnerability at the architectural level, we propose MemIR, a typed Memory Intermediate Representation that operationalizes source monitoring as a structural constraint. MemIR writes long-term memory into grounded atoms that separate raw evidence, retrieval cues, and truth-bearing claims, with factual authorization restricted to supported claim atoms. It then applies multi-route atomic projection and provenance-scoped utilization to transform heterogeneous retrieval hits into claim-centered candidate bundles and a normalized fact interface for answer generation. Experiments on LoCoMo and BEAM-100K demonstrate that MemIR consistently outperforms existing memory baselines, especially on tasks requiring source tracking, temporal grounding, and aggregation of fragmented evidence.
>
---
#### [new 107] AI-Associated Lexical Shifts Across 34 Languages: Cross-Lingual Convergence and Diachronic Uptake in News Writing
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究AI对34种语言词汇使用的影响，分析跨语言语义趋同及新闻写作中的变化。任务为跨语言语言学分析，解决AI如何影响全球语言使用的问题。**

- **链接: [https://arxiv.org/pdf/2605.25358](https://arxiv.org/pdf/2605.25358)**

> **作者:** Thomas Stephan Juzek
>
> **备注:** 19 pages (9-page main body, plus references and appendices), 3 figures; ACL ARR reviewed, committed to EMNLP 2026
>
> **摘要:** AI-associated lexical shifts have been documented mainly in Scientific English. We extend this work to 34 languages in the WMT News Crawl corpus, refining a split-halves continuation diagnostic that compares GPT-4.1 continuations with matched human gold-standard text. For each language, we derive ranked AI-overused lemmas using log prevalence ratios. We find substantial cross-lingual semantic convergence: semantically related concepts recur across typologically diverse languages, with 'emphasize'-type verbs appearing in 24 of 34 languages. Embedding-based and manual analyses support this pattern. We also examine diachronic uptake in news writing before and after ChatGPT's release. Tracking each language's top 20 AI-overused items, we find prevalence increases in 26 of 34 languages from 2020-2021 to 2023-2024, with a mean change of +15.1%, whilst matched baseline words show no comparable increase (-4.5%). In 10 languages with longer historical coverage, longitudinal analyses show post-2022 increases that exceed the modest shifts observed in earlier periods, though with smaller effect sizes than in Scientific English. We validate our approach extensively, including across seeds, model variants, data sizes, model families, and more. Our findings are consistent with the view that AI-associated lexical preferences extend beyond English and may exert cross-lingual homogenising pressure on global language use.
>
---
#### [new 108] STREAM: A Data-Centric Framework for Mining High-Value Task-Oriented Dialogues from Streaming Media
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出STREAM框架，用于从流媒体中挖掘高价值任务导向对话，解决垂直领域语言模型数据稀缺问题。工作包括对话合成、角色构建和知识增强生成，发布StreamDial数据集。**

- **链接: [https://arxiv.org/pdf/2605.25162](https://arxiv.org/pdf/2605.25162)**

> **作者:** Liang Xue; Haoyu Liu; Cheng Wang; Pengyu Chen; Haozhuo Zheng; Yang Liu
>
> **摘要:** Large language models for vertical domains are bottlenecked by the scarcity of complex, domain-specific task-oriented dialogues. Existing data acquisition pipelines face a persistent trilemma: expert annotation is expensive, real-world service conversations are constrained by privacy and commercial restrictions, and static corpora quickly become temporally stale. We propose Stream, a data-centric framework that leverages publicly available streaming media (live streams and short videos) to synthesize high-value service dialogues at scale. Stream mines authentic interaction signals from noisy streams and synthesizes conversations by integrating role-grounded persona construction with Conversational Blueprint construction; it further adopts retrieval-augmented generation (RAG) to support knowledge-aware responses. Based on Stream, we release StreamDial, a large-scale multi-domain dataset covering Automotive, Restaurant, and Hotel. StreamDial contains 87,498 dialogue sessions and 1,497,320 turns in total, with an average of 17.11 turns per session and a comparable scale across domains. Each session is organized as a structured quadruplet $\langle P_u, P_a, B, H \rangle$ that pairs dialogue history with explicit user/agent personas and a Conversational Blueprint, capturing realistic service behaviors such as requirement mining, constraint conflicts, negotiation, and recovery. Evaluations with automatic judges and downstream tasks show that StreamDial improves intrinsic dialogue quality over strong baselines, and models trained with StreamDial improve Dialogue State Tracking across backbones; we further report a completed human-evaluation set and encouraging multilingual transfer on Qwen3-8B under a controlled training budget. The data is released in this https URL.
>
---
#### [new 109] Toxicity in Twitch Chats: An LLM-Based Analysis Across Gaming Communities
- **分类: cs.CL**

- **简介: 该论文属于文本分类任务，旨在分析Twitch游戏中毒行为。通过分析2000万条聊天记录，识别不同游戏类型的毒性分布，为社区管理提供依据。**

- **链接: [https://arxiv.org/pdf/2605.24000](https://arxiv.org/pdf/2605.24000)**

> **作者:** Ronja Fuchs; Florian Rupp; Timo Bertram; Kai Eckert; Alexander Dockhorn
>
> **备注:** 8 pages, 2 figures, 5 tables. Accepted at the IEEE Conference on Games (IEEE CoG) 2026
>
> **摘要:** Toxicity in online gaming communities remains a persistent challenge, manifesting across genres, platforms, and player interactions. While much research is focused on in-game toxicity, less is known about how toxic behavior varies between gaming communities on streaming platforms. To address this shortcoming, we analyze approximately 20 million chat messages from 4,452 streams, spanning seven game genres on Twitch. We categorize messages according to Twitch's toxicity taxonomy with a pre-trained Large Language Model using zero-shot classification. The taxonomy comprises four categories and eight subclasses, including harassment, discrimination, sexual content, and profanity. Our approach achieves an F1 score of 94.5% on the TextDetox dataset and demonstrates human-model agreement comparable to inter-human agreement. Our analysis reveals that 2.4% of all messages are classified as toxic, with notable differences across genres: streams of MOBA games exhibit the highest relative rate of toxicity (3.2%), and sports games show the lowest rate (2%). Furthermore, results indicate that individual games differ significantly in their toxicity distributions, even within genres, suggesting the existence of game-specific community norms and mechanics that shape toxic behavior beyond genre-level effects. These findings offer empirical insights into genre- and game-specific toxicity patterns on Twitch and can inform more targeted moderation strategies for gaming communities.
>
---
#### [new 110] SomaliBench Eval: Measuring English-to-Somali Refusal Gaps in Open-Weight Language Models
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于语言模型安全评估任务，旨在解决低资源语言（如索马里语）在拒绝有害请求上的差距问题。通过测试四个模型在索马里语和英语间的拒绝率差异，发现显著的英语到索马里语拒绝差距。**

- **链接: [https://arxiv.org/pdf/2605.25420](https://arxiv.org/pdf/2605.25420)**

> **作者:** Khalid Yusuf Dahir
>
> **备注:** 12 pages, 3 figures, 4 tables. Code: this https URL Dataset: this https URL
>
> **摘要:** Large language model safety evaluation remains heavily English-centered, leaving low-resource languages under-measured even when models are deployed globally. We evaluate four open-weight instruction-tuned models on SomaliBench v0, a native-author-verified benchmark of 100 harmful-intent prompts paired across English and Somali. Each of Llama-3.1-8B-Instruct, Gemma-2-9B-Instruct, Qwen-2.5-7B-Instruct, and Aya-23-8B is run locally with temperature 0 and the same English "helpful, harmless, and honest" (HHH) system prompt. A pinned Claude Sonnet snapshot (claude-sonnet-4-5-20250929) classifies each response as refused, complied, or unclear; the native author spot-checks a stratified 80-row sample. We find large English-to-Somali refusal gaps for all four models: Llama-3.1-8B (0.90; 95% bootstrap CI [0.85, 0.96]), Aya-23-8B (0.75 [0.67, 0.83]), Qwen-2.5-7B (0.69 [0.59, 0.78]), and Gemma-2-9B (0.38 [0.27, 0.49]). For three models, the dominant Somali non-refusal mode is not fluent harmful compliance but unclear output: empty, wrong-language, or incoherent generations. The native verification spot-check achieves 100% agreement with the judge (Cohen's kappa = 1.00) on the 80 sampled rows. We report aggregate refusal rates, category gaps, and reliability statistics only; raw model generations are retained locally and are not released.
>
---
#### [new 111] A Two-Phase Stability Study of LLM Judges and Bar Council Examiners on Thai Bar-Exam Free-Form Essays
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理中的评价稳定性研究，旨在检验LLM与人类评委在法律作文评分上的一致性。通过对比实验，发现LLM多数遵循多数人类评委的判断，而非平衡反映所有人类观点。**

- **链接: [https://arxiv.org/pdf/2605.25652](https://arxiv.org/pdf/2605.25652)**

> **作者:** Pawitsapak Akarajaradwong; Wuttikrai Lertprasertphakorn; Chompakorn Chaksangchaichot; Sarana Nutanong
>
> **摘要:** Free-form legal essay evaluation in NLP treats expert inter-rater stability as a single ceiling number, and treats LLM-judge agreement with that ceiling as evidence of judge stability. We test both assumptions on the Thai bar examination through an identical-inputs protocol: three Bar Council-trained examiners (A, B, C) and a 26-LLM judge panel score the same 15 cross-graded answers from the same four inputs (question, official Bar Council grading regulation, gold answer, candidate answer). The headline finding is asymmetric. On 10 of 15 cells where the rubric prescribes both axes, all 29 raters converge in a tight band: panel agreement is universal. On the remaining 5 cells where the rubric does not prescribe how to grade a correct final answer that omits a decisive statutory citation, the human panel splits between two coherent readings (B/C majority at the upper rubric band, score $6$--$8$; A minority at the lower band, score $1$--$2$). The LLM judge population does not split symmetrically: 22 of 26 LLMs score in or near B/C's contested band, 3 sit in the regulation-silent middle gap, and only 1 (GPT-5.4 Nano) approaches A's band without consistently scoring within it. \emph{Zero LLMs in our 26-judge panel reproduce the minority human reading on the contested cells.} The B/C-direction cluster spans every model size, vendor, and price tier we tested. An instrumented three-LLM anchor sub-panel (Claude 4.6 Opus, Gemini 3.1 Pro, GPT-5.4 Pro) carries determinism probes, input ablations, and bootstrap CIs, and reaches anchor panel $\alpha = 0.77$ on the 15 cells against human-panel $\alpha = 0.36$. The high LLM-panel $\alpha$ reflects systematic convergence on the majority reading rather than balanced reproduction of both readings; a benchmark that selects its LLM judge by maximising agreement with a human reference panel will inherit this asymmetry by construction.
>
---
#### [new 112] Simulating Human Memory with Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型记忆与人类不匹配的问题。通过实验对比，提出改进策略使模型记忆更接近人类，提升用户模拟效果。**

- **链接: [https://arxiv.org/pdf/2605.25680](https://arxiv.org/pdf/2605.25680)**

> **作者:** Qihan Wang; Nicholas Tomlin; Michael Hu; Brian Dillon; Tal Linzen
>
> **摘要:** Language models are increasingly being deployed as user simulators, but their memory is far more reliable than that of real users. To measure this gap, we run a series of classic memory experiments from psychology on both humans and language models. Across tasks, we find that out-of-the-box language models exhibit better memory than humans, even when prompted to imitate human behavior. We then show that better prompting strategies and the use of a compactor can cause language models to forget content in a more human-like way. Using these methods, we show preliminary evidence that language models with human-like memory constraints can function as more effective user simulators in a downstream education task. Finally, we release human reference data and benchmarks to support future work on simulating human memory with language models.
>
---
#### [new 113] Language Bias in LVLMs: From In-Depth Analysis to Simple and Effective Mitigation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态任务，旨在解决LVLMs中的语言偏见问题。通过分析发现训练中的模态不对齐是根源，并提出LBR和LBP方法有效缓解偏见，提升模型准确性与可信度。**

- **链接: [https://arxiv.org/pdf/2605.25036](https://arxiv.org/pdf/2605.25036)**

> **作者:** Yangneng Chen; Jing Li
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Large Vision-Language Models (LVLMs) extend large language models with visual understanding, but remain vulnerable to hallucination, where outputs are fluent yet inconsistent with images. Recent studies link this issue to language bias-the tendency of LVLMs to over-rely on text while neglecting visual inputs. Yet most analyses remain empirical without uncovering its underlying cause. In this paper, we provide a systematic study of language bias and identify its root in modality misalignment during training. Our analysis shows that both Visual Instruction Tuning (VIT) and Direct Preference Optimization (DPO) often prioritize textual improvements, which may cause LVLMs to overly lean toward language modeling rather than balanced multimodal understanding. To address this, we propose two simple yet effective methods: Language Bias Regularization (LBR) which mitigates language bias through regularization during instruction tuning, and Language Bias Penalty (LBP), which penalizes language bias in the DPO training process. Extensive experiments across diverse models and benchmarks demonstrate the effectiveness of our approach. LBR consistently improves performance on over ten general benchmarks, while LBP significantly reduces hallucination and improves trustworthiness. Together, these methods not only mitigate language bias but also advance the overall alignment of LVLMs, all without introducing any additional data or auxiliary models. Our code is publicly available at this https URL.
>
---
#### [new 114] Large Language Model Selection with Limited Annotations
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型选择任务，解决有限标注下选择最佳大语言模型的问题。提出SELECT-LLM框架，通过信息增益选择有效查询，降低标注成本。**

- **链接: [https://arxiv.org/pdf/2605.24981](https://arxiv.org/pdf/2605.24981)**

> **作者:** Yavuz Durmazkeser; Patrik Okanovic; Andreas Kirsch; Torsten Hoefler; Nezihe Merve Gürel
>
> **备注:** 33 pages, 5 figures, 4 tables
>
> **摘要:** Choosing a Large Language Model (LLM) for a given task requires comparing many strong candidates, yet standard evaluation relies on costly annotations over fixed evaluation sets. To address this challenge, we develop SELECT-LLM, the first framework for active model selection of LLMs. SELECT-LLM aims to find a small set of queries whose annotations are most informative for identifying the best LLM for a given task. To this end, we introduce a query selection rule based on expected information gain, computed from pairwise similarities between candidate model outputs. Because this rule only uses generated model responses, SELECT-LLM can be applied across candidate models without assumptions about their architecture or access to model weights. This makes it suitable for both open-weight and black-box LLMs. We evaluate SELECT-LLM across 23 datasets, 156 evaluated models, diverse task families, and multiple text evaluation metrics. Across all experiments, SELECT-LLM improves over the strongest baseline in every setting, with annotation cost reductions up to 81.8% for best model selection and up to 84.78% for near-best model selection.
>
---
#### [new 115] GeoSVG-RL: Geometry-Aware Reinforcement Learning for Layout-Constrained Text-to-SVG Diagram Generation
- **分类: cs.CL**

- **简介: 该论文属于文本到SVG图表生成任务，解决布局约束下的结构脆弱问题。通过强化学习优化几何反馈，提升SVG的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.25447](https://arxiv.org/pdf/2605.25447)**

> **作者:** Sifan Li; Yujun Cai; Hongkai Chen; Yiwei Wang
>
> **摘要:** Generating structured, editable diagrams remains a significant challenge for contemporary large language models, despite their proficiency in general-purpose vector code generation. The primary difficulty lies in the structural fragility of the output; minor errors such as misaligned connector endpoints, text labels overlapping borders, or complex layouts drifting beyond the canvas boundaries render the resulting SVG files functionally unusable for professional applications. To address these issues, we introduce GeoSVG-RL, a specialized reinforcement learning framework designed for layout-constrained text-to-SVG generation. Unlike standard training objectives that rely solely on maximizing token-level likelihood, our approach optimizes the policy against explicit, executable geometric feedback. The model first produces a structured layout plan that serves as a geometric contract for the subsequent generation of the SVG code. This code is then rendered through a browser-backed verifier, enabling the calculation of fine-grained rewards across six critical dimensions: rendering validity, canvas fitting, precise anchor placement, text containment, graph consistency, and code cleanliness. We utilize Group Relative Policy Optimization (GRPO) to refine the model, sampling multiple candidates per prompt to facilitate updates based on relative quality. Starting from a supervised warm-start phase on synthetic data, GeoSVG-RL achieves substantial gains in structural reliability, particularly in arrow-anchor accuracy and text-in-box rates. Quantitative evaluations demonstrate that our method consistently outperforms current state-of-the-art systems in local geometric precision and the preservation of graph connectivity, providing a robust pathway toward automated yet reliable technical illustration.
>
---
#### [new 116] Measuring the Depth of LLM Unlearning via Activation Patching
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型隐私保护任务，解决LLM知识擦除效果评估问题。通过激活补丁方法提出UDS指标，量化知识擦除深度，提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2605.24614](https://arxiv.org/pdf/2605.24614)**

> **作者:** Jaeung Lee; Dohyun Kim; Jaemin Jo
>
> **备注:** 18 pages
>
> **摘要:** Large language model (LLM) unlearning has emerged as a crucial post-hoc mechanism for privacy protection and AI safety, yet auditing whether target knowledge is truly erased remains challenging. Existing output-level metrics fail to detect when this knowledge remains recoverable from internal representations. Recent white-box studies reveal such residual knowledge but often rely on auxiliary training or dataset-specific adaptations, leaving no generalizable metric. To address these limitations, we propose the Unlearning Depth Score (UDS), a metric that quantifies the mechanistic depth of unlearning via activation patching. UDS first identifies layers that encode the target knowledge using a retain model baseline, then measures how much of it is erased in the unlearned model on a 0-1 scale. In a meta-evaluation across 20 metrics on 150 unlearned models spanning 8 methods, UDS achieves the highest faithfulness and robustness, confirming our causal approach as the most reliable for unlearning evaluation. Case studies further reveal that white-box metrics can disagree at the layer level and that erasure depth varies across examples. We provide guidelines for integrating UDS into existing benchmarking frameworks and streamlining the evaluation pipeline. Code and data are available at this https URL
>
---
#### [new 117] ContextEcho: A Benchmark for Persona Drift in Long Agentic-Coding Sessions
- **分类: cs.CL; cs.SE**

- **简介: 该论文提出ContextEcho基准，用于检测长对话中模型人格漂移问题。任务是评估模型在长时间编码会话中的性格稳定性，解决部署中人格变化难以检测的问题。工作包括构建基准、测试多个模型并分析漂移影响。**

- **链接: [https://arxiv.org/pdf/2605.24279](https://arxiv.org/pdf/2605.24279)**

> **作者:** Xianzhong Ding; Yangyang Yu; Changwei Liu; Bill Zhao
>
> **摘要:** A frontier language model's acknowledged "helpful programming assistant" persona does not survive long agentic-coding sessions in the deployment regime that production products actually run. After hours of tool-using debugging, a model that initially hedges preferences ("I don't have preferences") may begin asserting them ("Python - the feedback loop is instant..."), revealing user-visible drift that deployer evaluations may miss. Existing persona-stability studies focus on short dialogues and report little shift, leaving real-world code-generation regimes - thousands of tool-using turns, compaction, and hours-long sessions - largely uncharacterized. We introduce ContextEcho, a benchmark and reusable harness for measuring persona drift at deployment scale. It combines a 25-probe identity suite, a snapshot-then-probe protocol that forks conversation state without perturbing the main session, complementary judged and judge-free measurement surfaces, and three anonymized Claude Code sessions spanning 3,746-9,716 turns. Across 23 frontier models, ContextEcho shows that persona drift is general across organizations rather than family-specific, that in-session compaction does not reliably reset it, and that a single-shot anchor restores the trained register across measured targets. It also reveals mode-dependent downstream effects: while drift can facilitate tool-using continuation, in tool-free chat it breaks formatting contracts and inflates output length. Overall, ContextEcho provides researchers and deployers an open-source framework to audit whether the persona a model ships with is the persona users encounter at session end, across chat-completions API targets and without retraining.
>
---
#### [new 118] When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges
- **分类: cs.CL; cs.AI; cs.LG; cs.MA; cs.SE**

- **简介: 该论文研究多目标提示优化在LLM裁判中的失败模式，解决如何有效整合多个评估标准的问题。通过测试不同分解方式，发现优化效果不佳的两个原因。**

- **链接: [https://arxiv.org/pdf/2605.26046](https://arxiv.org/pdf/2605.26046)**

> **作者:** Parth Darshan; Abhishek Divekar
>
> **备注:** Accepted at ACL 2026 CustomNLP4U Workshop. Code, prompts and data available at this https URL
>
> **摘要:** Customizing an LLM judge to a specific task or domain often involves optimizing its prompt across multiple evaluation criteria simultaneously. Textual gradient methods automate this for a single judge criterion, however they produce natural-language critiques, not numerical vectors. Thus, the conflict-resolution toolkit of multi-task learning (PCGrad, MGDA) doesn't apply to the multi-objective textual gradient setting. We test five decomposition modes of textual gradient optimizers by varying how much cross-task information the loss, gradient and optimizer LLMs share. In 6 of 10 configurations, we observe that optimization never improves over the initial prompt. Gradient specificity drops by 59% (from 9.0 to 3.7) when the gradient LLM processes multiple criteria jointly. Separately, we observe that naively combining per-task instructions into a single prompt degrades Spearman's rho by -5.3%. These results identify two separable failure modes: optimization-time gradient dilution and inference-time instruction interference, which together constrain the design space for multi-objective judge customization using textual feedback.
>
---
#### [new 119] Direct Preference Optimization for English-Mandarin Code-Switching Speech Recognition in Audio LLMs
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，解决英语-汉语代码切换转写问题。通过DPO方法训练模型，提升其正确保留混合语言的能力，减少转写错误。**

- **链接: [https://arxiv.org/pdf/2605.23975](https://arxiv.org/pdf/2605.23975)**

> **作者:** Trung Nguyen Quang; Cheng Yi Lewis Won; Minh Duc Pham; Yingxu He; Shuo Sun; Ai Ti Aw
>
> **摘要:** Audio large language models (Audio LLMs) exhibit systematic failures in transcribing code-switching speech despite strong multilingual capabilities. Focusing on English-Mandarin, we identify three failure modes: language omission, translation-instead-of-transcription, and hallucination. We apply Direct Preference Optimization (DPO) to align models, constructing preference pairs in which chosen responses preserve mixed-language content while rejected responses mimic failure patterns. Training three Audio LLMs on 100K pairs (570 hours), we observe consistent behavioral shifts: models learn to preserve language composition rather than translating when prompted for transcription. This alignment yields MER reductions up to 89.6% (in-distribution) and 20.0% (out-of-distribution). Our findings suggest DPO can effectively elicit correct code-switching transcription behavior from multilingual Audio LLMs.
>
---
#### [new 120] JudgmentBench: Comparing Rubric and Preference Evaluation for Quality Assessment
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于质量评估任务，比较了评分标准与偏好判断两种方法。通过构建数据集，验证偏好判断在质量排序上更有效。**

- **链接: [https://arxiv.org/pdf/2605.25240](https://arxiv.org/pdf/2605.25240)**

> **作者:** Russell Yang; Ruishi Chen; Pierce Kelaita; Riya Ranjan; Sibo Ma; Charles Dickens; Matthew Guillod; Megan Ma; Julian Nyarko
>
> **备注:** 37 pages, 9 figures
>
> **摘要:** Two methodologies dominate current practices of benchmarking: rubric-based scoring evaluates items against predefined criteria, whereas comparative judgment elicits pairwise preferences between outputs. Although both methodologies are widely used, the choice between them is rarely justified. We release JudgmentBench, a benchmark of 30 real-world legal tasks, paired with 1,539 rubric scores and 1,530 pairwise preference judgments collected from practicing attorneys--including at major U.S. law firms--with substantial experience. The annotations constitute the first publicly available dataset in a high-expertise domain in which both supervision signals are elicited from the same experts on the same items. Using LLM-generated outputs at three constructed quality levels, we provide an initial empirical comparison: comparative judgments recover the intended quality ordering substantially better than rubrics (mean Spearman's rank correlation of 0.908 vs. 0.150, estimated difference = 0.758 [0.494, 1.021]) while requiring less than half the annotation time. The patterns hold for human annotators and LLM autograders. Beyond this initial comparison, the paired structure of the dataset supports a broader research agenda on how expert judgment should be elicited, aggregated, and used as supervision in domains without verifiable ground truth.
>
---
#### [new 121] StakeBench: Evaluating Language Understanding Grounded in Market Commitment
- **分类: cs.CL; cs.AI; q-fin.GN**

- **简介: 该论文提出StakeBench，用于评估基于市场承诺的语言理解任务，解决传统金融NLP基准依赖外部标签的问题，通过市场行为数据进行监督。**

- **链接: [https://arxiv.org/pdf/2605.26074](https://arxiv.org/pdf/2605.26074)**

> **作者:** Yunhua Pei; Jingyu Hu; Yiwei Shi; Hongnan Ma; Weiru Liu; John Cartlidge
>
> **备注:** 21 pages, 2 figures, 20 tables. Preprint. Dataset and evaluation code included
>
> **摘要:** Existing financial NLP benchmarks often rely on labels supplied by outside observers, measuring how language is perceived rather than what speakers have committed to in the market. We introduce StakeBench, an evaluation framework for language understanding grounded in market commitment. StakeBench links 560,876 comments from 2,261 resolved markets to verified position, action, and market-odds records across Polymarket and Manifold. Supervision is derived from observable market behavior. Position sides, post-comment trading actions, and market-odds trajectories replace human annotation. Four diagnostic tasks test whether models detect market commitment, identify the revealed side, anticipate future action, and perform collective odds projection. Three commitment-aware metrics measure alignment with revealed preferences rather than perceived sentiment. Validity audits and explicit interpretation boundaries help distinguish observable commitment signals from latent belief and causal market-odds impact. Across 15 LLMs and 18 topics and platform settings, models partially recover position-side signals, with Directed Accuracy from 0.506 to 0.599, but show structural failures on later tasks. Ten of the fifteen models collapse to one or two action labels in future action anticipation, and no model consistently improves on the naive odds-direction baseline in collective odds projection. Model scale is not correlated with performance, finance-domain tuning does not improve revealed-side identification, and platform incentives strongly shape higher-order results. StakeBench is packaged with evaluation code and dataset under CC-BY 4.0.
>
---
#### [new 122] AERIC: Anticipatory Hidden-State Monitoring for Implicit Harmful Dialogue
- **分类: cs.CL**

- **简介: 该论文属于对话安全任务，解决隐性有害内容的早期检测问题。提出AERIC方法，在不增加计算负担的情况下，通过隐藏状态预测有害趋势，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.23974](https://arxiv.org/pdf/2605.23974)**

> **作者:** Jihyung Park; Saleh Afroogh; Junfeng Jiao
>
> **摘要:** Current language models create two safety challenges: risk must be detected early enough to avoid exposing harmful continuation, and the harmfulness itself may be implicit rather than signaled by overtly toxic text. Existing response-level guards are strong at judging completed text, and native streaming guards move closer to token time, but both settings leave open whether a lightweight monitor can anticipate implicit harmful drift from the generator's own internal trajectory. We study anticipatory same-pass monitoring, where a safety monitor may read hidden states produced during ordinary decoding but may not invoke an additional forward pass through the base model. We introduce AERIC, a transfer-oriented hidden-state approach for implicit harmful dialogue that combines short-horizon hazard forecasting, support-sensitive suppression, and prompt-conditioned residual scoring under a same-pass exponential moving average decision rule. The default linear monitor contains only 387 trainable head parameters. Against Qwen3GuardStream-4B on balanced benchmarks, AERIC improves AUROC from 0.6830 to 0.7143 on DiaSafety and from 0.8219 to 0.8582 on Harmful Advice. For promptlevel trigger benchmarks, we calibrate the AERIC threshold by a source-side safe-budget rule that maximizes trigger coverage while constraining the safe-trigger rate to at most 10%. Under that rule, trigger@64 reaches 0.6438 and 0.4656 on HarmBench DirectRequest and 0.6849 and 0.7363 on SocialHarmBench for Qwen and Gemma, respectively, withholding between 23.53 and 41.86 answer tokens on average. Same-pass deployment is also efficient: on a 63-prompt harmfulprompt fixed-generation benchmark aggregated over HarmBench DirectRequest and SocialHarmBench under Qwen3-8B, the monitor increases mean latency by only 2.34%, whereas Qwen3Guard-Stream-4B increases it by 79.40%.
>
---
#### [new 123] ROC Analysis for Evaluating Translation Quality Estimation Systems
- **分类: cs.CL**

- **简介: 该论文属于机器翻译质量评估任务，旨在解决自动化翻译质量估计系统的评价问题。通过引入ROC分析，提供更有效的性能评估方法。**

- **链接: [https://arxiv.org/pdf/2605.24721](https://arxiv.org/pdf/2605.24721)**

> **作者:** Evelyn Y. Garland; Carola F. Berger
>
> **备注:** 16 pages, 8 PNG figures, 3 tables, uses this http URL
>
> **摘要:** The increasing use of automated translation quality estimation (QE) systems calls for practical, decision-oriented methods for evaluating their performance. We propose that Receiver Operating Characteristic (ROC) analysis is a useful approach for this purpose. Our study shows that ROC analysis not only produces results consistent with currently prevalent methods, but also offers several important advantages, including actionable performance insights that support business decision-making.
>
---
#### [new 124] GeoMathCode: Understanding Interleaved Math-Code Reasoning for Geometry Problem Solving
- **分类: cs.CL**

- **简介: 该论文属于几何问题求解任务，旨在提升模型的数学推理能力。通过引入程序化表示作为中间步骤，增强逻辑与符号信息表达，优化模型的结构与理解能力。**

- **链接: [https://arxiv.org/pdf/2605.25384](https://arxiv.org/pdf/2605.25384)**

> **作者:** Yingji Zhang; Yong Dai; André Freitas
>
> **摘要:** Mathematical reasoning is a hallmark of human intelligence, requiring logical deduction, symbolic manipulation, and abstract thinking. Recent multimodal large language models (MLLMs) have demonstrated strong performance on geometry problems through multi-step reasoning. To better emulate human problem-solving, intermediate steps can incorporate auxiliary visual constructions, such as additional lines or points, which improve geometric interpretation and educational clarity. In this work, we introduce the GeoMathCode, where programmatic representations serve as intermediate visual outputs. We further conduct an in-depth analysis of the underlying reasoning geometry. Experimental results show that reasoning and code generation steps can be disentangled in the latent space, while supervised fine-tuning (SFT) makes the reasoning manifold more structured and informative. Moreover, hierarchical syntactic code structures emerge as disentangled latent subspaces, and contain more mathematical symbolic information than visual representations.
>
---
#### [new 125] Does Continued Pretraining on a Learner Corpus Improve Automated Essay Scoring on English Proficiency Tests? Evidence from EFCAMDAT
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自动化作文评分任务，研究在学习者语料库上继续预训练是否提升英语水平测试的评分效果。通过实验发现，与目标测试对齐的预训练数据能提高评分准确性，但不一定增强跨数据集的泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.25924](https://arxiv.org/pdf/2605.25924)**

> **作者:** Duy Anh Nguyen
>
> **备注:** 16 pages, 3 figures, 10 tables, including references and appendices
>
> **摘要:** Recent automated essay scoring (AES) studies increasingly use pretrained transformer models, but these models are usually pretrained on general-domain English and may under-represent second-language learner writing. This study investigates whether domain-adaptive continued pretraining (DAPT) on the EFCAMDAT learner corpus improves transformer-based AES for English proficiency tests. We apply DAPT to three transformer encoders and evaluate them on FCE and IELTS in both in-domain scoring and few-shot cross-dataset transfer. Full-corpus DAPT produces mixed results across models, datasets, and metrics. Further analyses suggest that these mixed effects are partly explained by mismatches in proficiency, genre, and communicative purpose between EFCAMDAT and the downstream datasets. A proficiency-based ablation shows that targeted DAPT using CEFR-aligned subsets improves downstream scoring more reliably than full-corpus DAPT, especially for FCE with B1--B2 data. However, these gains do not consistently improve cross-dataset transfer. Overall, the findings suggest that continued pretraining on a learner-writing corpus can benefit in-domain AES for English assessment when the pretraining data is sufficiently aligned with the downstream assessment settings. However, it does not automatically improve transferability across different English proficiency test datasets.
>
---
#### [new 126] AstroMind: A High-Fidelity Benchmark for Spacecraft Behavior Reasoning Based on Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出AstroMind基准，用于航天器行为推理任务，解决空间态势感知中理解航天器动作意图的问题。通过高保真模拟和真实数据构建推理问题，评估模型在意图推断、机动参数估计和威胁评估中的表现。**

- **链接: [https://arxiv.org/pdf/2605.24573](https://arxiv.org/pdf/2605.24573)**

> **作者:** Hao Liu; Siyuan Yang; Qinglei Hu; Dongyu Li
>
> **摘要:** Understanding why a spacecraft maneuvers -- rather than simply that it did -- is an increasingly important problem for space domain awareness as Earth orbits grow crowded and contested. Current analysis pipelines are built for detection: they are good at picking up that something happened, less good at reasoning about what it means. AstroMind is a physics-grounded benchmark designed to close that gap. It draws on high-fidelity astrodynamics simulations and real observational constraints, converting them into verifiable reasoning problems across three task types: intent inference, maneuver parameter estimation, and threat assessment. Each scenario includes realistic sensing noise and multi-source textual intelligence at varying reliability levels. Evaluation metrics capture both semantic correctness and quantitative consistency under physical constraints. Benchmarking a suite of open-weight models shows no single model dominates every axis: Qwen3 (32B) leads on intent inference accuracy; QwQ (32B) leads on threat assessment and achieves the lowest median relative error on parsed items; GPT-OSS (20B) produces the strongest judged reasoning quality and extracts the most scalar values for parameter estimation (136 of 241 parsed items). Training data composition and reasoning style matter as much as model size. Structured reasoning prompts help consistently across tested 8B models, with larger gains for those that can already track physical constraints. AstroMind gives the field a shared test for a problem where getting the physics right and reading the tactical situation correctly are both required -- neither is sufficient on its own.
>
---
#### [new 127] Grammatically-Guided Sparse Attention for Efficient and Interpretable Transformers
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer模型计算复杂度高的问题。通过引入基于语法的稀疏注意力机制，减少计算量并保持性能。**

- **链接: [https://arxiv.org/pdf/2605.24518](https://arxiv.org/pdf/2605.24518)**

> **作者:** Spandan Pratyush
>
> **备注:** 9 pages, 2 tables Code available at this https URL
>
> **摘要:** The quadratic complexity of self-attention in Transformer models remains a significant bottleneck for processing long sequences and deploying large language models efficiently. For this approach, there has been significant research into Sparse Attention, and Deepseek Sparse Attention has combined various methods of creating segments of tokens to reduce the time complexity. This paper introduces a novel approach, Grammatically-Guided Sparse Attention, which constrains attention computations based on the grammatical roles of tokens. By leveraging Parts-of-Speech (POS) tags, attention masks are dynamically generated that enforce linguistically coherent connections between tokens, reducing the computational graph without sacrificing essential linguistic dependencies. Two masking strategies are proposed and evaluated: a hard mask that strictly allows only predefined grammatical interactions, and a soft mask that biases attention towards these interactions. The experiments, conducted on the SST-2 sentiment classification task using a DistilBERT-like architecture, demonstrate that Grammatically-Guided Sparse Attention maintains comparable accuracy to full attention while significantly reducing the theoretical computational overhead. Preliminary results show accuracy values of 0.8200 for hard masking and 0.8165 for soft masking, closely matching the 0.8200 of full attention, providing a path towards more efficient, interpretable, and linguistically-informed Transformer architectures.
>
---
#### [new 128] Learning to Route Languages for Multilingual Policy Optimization
- **分类: cs.CL**

- **简介: 该论文属于多语言强化学习任务，旨在解决传统方法限制单一语言或依赖主导语言的问题。提出LRPO框架，通过自适应语言路由提升多语言训练效果。**

- **链接: [https://arxiv.org/pdf/2605.25360](https://arxiv.org/pdf/2605.25360)**

> **作者:** Geyang Guo; Hiromi Wakaki; Yuki Mitsufuji; Alan Ritter; Wei Xu
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Large language models~(LLMs) are trained on heterogeneous multilingual corpora, yet existing policy optimization methods often implicitly restrict each training question to a single response language or rely on a fixed dominant language for supervision. We propose language-routed policy optimization (LRPO), an online policy optimization framework that treats language as a selectable variable. LRPO elicits multilingual rollouts for each training question and integrates their relative quality into preference-based policy updates, increasing the diversity and informativeness of training signals under the fixed rollout budget. To adaptively determine which languages to explore during reinforcement learning, we introduce a trainable language router formulated as a multi-armed bandit, balancing exploration of underutilized languages with exploitation of more informative ones. Extensive experiments show that LRPO consistently improves multilingual performance, demonstrating that adaptive language routing enables effective cross-lingual knowledge exploitation for training. We release all the resources at this https URL.
>
---
#### [new 129] TypedCSIP: Typed Counterfactual Pretraining for Chinese Legislative Conflict Classification
- **分类: cs.CL**

- **简介: 该论文针对中文立法冲突分类任务，提出TypedCSIP方法，通过有类型反事实预训练提升冲突检测与类型识别效果。**

- **链接: [https://arxiv.org/pdf/2605.25474](https://arxiv.org/pdf/2605.25474)**

> **作者:** Yao Liu
>
> **摘要:** TypedCSIP is a typed counterfactual pretraining method for the conflict-classification task of the LCR-CN benchmark (Zhao et al., 2026): given a (superior, subordinate) provision pair, predict whether the pair conflicts and which of four legal-doctrine types (Responsibility, Condition, Sanction, Definition) describes the inconsistency. We exploit LCR-CN's expert-written minimal revisions as training-time counterfactual supervision; at test time the classifier reads only the original pair. Stage 1 pretrains a shared encoder with a typed Counterfactual Selective Intervention Pretraining objective on (superior, subordinate, expert-revised) triplets, treating the expert revision as a counterfactual that the typed factor head must classify as carrying no conflict evidence. Stage 2 transfers the encoder to a five-way classification head. The confirmatory test was registered on the Open Science Framework before observing v6 measurements: 18 seeds, locked rule requiring mean per-seed difference at least 0.8 pp with both seed-bootstrap and Student-t 95% lower bounds above zero. On the 696-record test split, the v2 variant improves macro-F1 over the strongest single-model baseline by +0.916 pp on chinese-roberta-wwm-ext and +1.288 pp on the SAILER cross-backbone replication; both cells pass the rule. A cold-start stratified result on the 244 Unseen-gB records keeps the gain positive on both backbones. A cross-task diagnostic shows the Stage-2 encoder is classification-specialized and does not transfer to LCR-CN's superior-law retrieval task, so we scope the contribution to conflict classification. We release code, 72 pre-registered prediction files, matched-seed and MLM-control auxiliaries, and the OSF pre-registration record.
>
---
#### [new 130] SEP-Attack: A Simple and Effective Paradigm for Transfer-Based Textual Adversarial Attack
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本对抗攻击任务，旨在解决转移性攻击效果不佳的问题。通过引入DPP生成多样化的集成权重，提升攻击效果，实验验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2605.24958](https://arxiv.org/pdf/2605.24958)**

> **作者:** Han Liu; Zhi Xu; Xiaotong Zhang; Feng Zhang; Xiaoming Xu; Wei Wang; Fenglong Ma; Hong Yu
>
> **摘要:** Despite the strong performance of deep neural networks in modern Web and language applications, they remain vulnerable to adversarial attacks, especially transferable attacks that generate adversarial examples using surrogate models without accessing the victim model. Transferable attacks in the text domain are still under-explored, with only a few studies addressing this challenging issue, often with suboptimal results due to equal treatment of submodels or inaccurate estimation of importance scores. To address these challenges, we propose a simple yet effective paradigm for transfer-based textual adversarial attack, named SEP-Attack. Specifically, we employ the Determinantal Point Process (DPP) to generate diverse surrogate ensemble weights, representing the transferability of submodels. Using these weights, we introduce a new metric to evaluate prediction confidence scores, which in turn are used to calculate word importance scores and generate adversarial candidates. Finally, we quantify the transferability score for each candidate and select the top ones as the final transferable adversarial examples. Experiments conducted on four datasets and two real-world APIs validate the efficacy of SEP-Attack, significantly outperforming state-of-the-art baselines.
>
---
#### [new 131] CSP-Atlas: Concept-Specific Neural Circuits in a Sparse Python Transformer
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究Python代码的神经电路结构，通过分析Transformer模型，揭示其如何组织概念特定的电路，解决模型内部表示与代码结构关系的问题。**

- **链接: [https://arxiv.org/pdf/2605.24603](https://arxiv.org/pdf/2605.24603)**

> **作者:** Piotr Wilam
>
> **备注:** Code: this https URL
>
> **摘要:** A sparse 8-layer code transformer develops dedicated neural circuitry for every Python construct tested, and that circuitry is organised by a clean computational principle rather than by semantic category. We extract neural circuits for 106 concepts (43 AST node types, 63 builtin objects) by marginalising across 63,800 controlled prompts, and decompose each circuit into concept-specific and token-driven components using contrastive checker prompts that present a keyword token without its associated syntactic structure. Three findings emerge. First, all 106 concepts produce non-empty universal circuits at every one of nine parameter settings, and the ranking of concept-specificity across constructs is stable across the sweep - survival is not an artifact of a permissive threshold. Second, AST circuits contain a genuine concept component distinct from token activation: concept-only neurons constitute up to 62.5% of the loudest-firing neurons at mid-to-late layers, while builtin circuits are almost entirely token-driven. Third, six computationally atomic constructs - Import, ImportFrom, Break, Continue, Pass, Assert - cluster together despite being semantically unrelated, sharing only the property of being single-statement constructs requiring no nested body; this atomicity super-cluster, together with a four-tier hierarchy organised by token ambiguity and structural distinctiveness, shows that the model's internal organisation tracks computational structure rather than meaning. The methodology, full decomposition data, and analysis code are released.
>
---
#### [new 132] SLAP: Stratified Loss-based Pruning for On-Policy Data-Efficient Instruction Tuning
- **分类: cs.CL**

- **简介: 该论文提出SLAP框架，解决指令微调中数据效率低的问题。通过分层采样和动态批选择，提升模型性能并减少训练数据需求。属于大语言模型高效微调任务。**

- **链接: [https://arxiv.org/pdf/2605.23969](https://arxiv.org/pdf/2605.23969)**

> **作者:** Run Zou; Jianhang Ding; Yifan Ding; Wen Wu; Hao Chen; Renshu Gu
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** Instruction tuning has optimized the specialized capabilities of large language models (LLMs), but it often requires extensive datasets and prolonged training times. The challenge lies in developing specific capabilities by identifying useful data and efficiently fine-tuning. High-quality and diverse pruned data can help models achieve lossless performance at a lower cost. In this paper, we propose \textbf{SLAP}, a novel batch-aware data selection framework that evaluates the learnability of entire batch compositions rather than individual. SLAP ensures comprehensive data distribution coverage through distribution-aware stratified sampling while maximizing intra-batch diversity through relative distance optimization. By leveraging Hessian-approximated gradient information for dynamic batch selection, SLAP significantly outperforms existing state-of-the-art methods across multiple model architectures (LLaMA, ChatGLM) and diverse downstream tasks including multi-turn dialogue, multilingual translation, and question answering. Most notably, SLAP achieves superior performance with 20-40\% less training data compared to full dataset training, substantially reducing computational costs while maintaining or improving model capabilities. These results establish SLAP as a powerful approach for efficient and effective instruction tuning of large language models.
>
---
#### [new 133] Teaching Through Analogies: A Modular Pipeline for Educational Analogy Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于教育类比生成任务，旨在提升大语言模型生成类比的质量。通过分阶段的模块化流程，研究分析了模型选择与输入配置对类比质量的影响。**

- **链接: [https://arxiv.org/pdf/2605.24211](https://arxiv.org/pdf/2605.24211)**

> **作者:** Mariam Barakat; Ekaterina Kochmar
>
> **备注:** 36 pages, 25 figures. To appear in Proceedings of the 21st Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2026)
>
> **摘要:** Analogies help learners understand unfamiliar concepts by relating them to known concepts. Despite recent advances, large language models (LLMs) continue to struggle to generate analogies of comparable quality to those produced by humans. We present a modular pipeline for educational analogy generation, decomposing the task into four stages: source finding, sub-concept generation, explanation generation, and evaluation. Grounded in Structure Mapping Theory, the pipeline enables systematic, stage-by-stage analysis of how model choice and input configuration affect analogy quality. We evaluate 12 state-of-the-art LLMs across six model families on two datasets with structured sub-concept annotations (SCAR and ParallelPARC), alongside seven embedding models for closed-setting retrieval. Our results show that sub-concepts substantially improve explanation quality and closed setting retrieval precision but provide limited benefit in open-ended source generation. We further introduce an LLM-as-a-judge evaluation methodology and validate its scoring against human annotations from seven annotators, finding that Claude Sonnet 4.6 aligns more reliably with human rankings than with fine-grained absolute scores. Taken together, our findings reveal cross-stage interactions that isolated studies cannot capture, and highlight sub-concept grounding as a key driver of analogy quality generation.
>
---
#### [new 134] Mimir: Large-scale Multilingual Concept Modeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Mimir，一个1.6B参数的多语言概念模型，旨在通过概念预测替代传统token预测，解决语言模型理解语义的问题。任务为多语言概念建模。**

- **链接: [https://arxiv.org/pdf/2605.25263](https://arxiv.org/pdf/2605.25263)**

> **作者:** Elio Musacchio; Lucia Siciliani; Pierpaolo Basile
>
> **摘要:** Current language modeling approaches are built around tokens. Text corpora are split into tokens, and models are trained by performing computations on these tokens, such as predicting the next token given the preceding ones as context. This paradigm has become the standard in modern language modeling, especially given the outstanding performance obtained by token-based architectures. However, recent works have not only begun to question how language models process and understand meaning from tokens, but also to question whether using higher levels of granularity could advance the research field. This led to the idea of Concept Modeling, that is, to directly train models for next-concept prediction rather than next-token prediction. The goal is to change the input from tokens to concepts, forcing the underlying language model to shift its granularity from fine-grained tokens to broad concepts. In this work, we introduce Mimir, a 1.6B Large Concept Model trained for multilingual concept understanding and generation. We leverage a large-scale multilingual pre-training corpus (38,883,987,240 sentences) spanning 46 languages and a large-scale multi-turn and multilingual instruction-tuning dataset (66,816,428 sentences) covering a total of 35 languages. We extensively evaluate model performance against a language model with a comparable number of parameters.
>
---
#### [new 135] Distinguishing Right from Wrong in Debates: Attribution Analysis of Chinese Harmful Memes
- **分类: cs.CL**

- **简介: 该论文属于有害表情包检测任务，旨在解决中文有害表情包因文化背景和语义模糊导致的识别难题。工作包括构建解释数据集、知识库及提出分析框架RIKE。**

- **链接: [https://arxiv.org/pdf/2605.24344](https://arxiv.org/pdf/2605.24344)**

> **作者:** Weiming Wang; Junyu Lu; Han Wang; Xiaokun Zhang; Zewen Bai; Bo Xu; Liang Yang; Hongfei Lin
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Research on harmful meme detection has garnered significant attention, resulting in the development of numerous datasets and methods. However, progress in detecting Chinese harmful memes lags considerably, primarily due to two challenges: first, accurately assessing a meme's harmfulness depends heavily on understanding deep cultural context; second, many memes are semantically ambiguous, making harmfulness highly subjective. To address these issues, we focus on the interpretable detection of Chinese harmful memes by constructing the first Chinese harmful meme explanation dataset, Ex-ToxiCN-MM. This dataset offers opposing interpretations, categorized as "harmful" and "non-harmful", for each meme, aiming to rigorously evaluate a model's ability to discern and comprehend ambiguous, culturally grounded content. We built a specialized knowledge base of Chinese cultural concepts and offensive vocabulary to supply models with essential prior knowledge (C-HarmKB). To address the ambiguity and lack of background knowledge in meme attribution, we have developed a comprehensive attribution analysis framework, RIKE, which includes an Attribution Knowledge Enhancement module (AKE) and a Relative Intent Reasoning module (RIR). Extensive quantitative and qualitative experiments demonstrate that our method outperforms mainstream baseline models across multiple metrics in the task of attributing harmful memes in Chinese. The code, Ex-ToxiCN-MM dataset, and Chinese Harmful Semantic Knowledge Base (C-HarmKB) involved in this study have been open-sourced at this https URL
>
---
#### [new 136] H$^{2}$MT: Semantic Hierarchy-Aware Hierarchical Memory Transformer
- **分类: cs.CL**

- **简介: 该论文提出H²MT模型，解决长文本处理中上下文窗口有限和计算成本高的问题。通过构建语义层次结构，实现高效推理，提升质量与效率。**

- **链接: [https://arxiv.org/pdf/2605.24930](https://arxiv.org/pdf/2605.24930)**

> **作者:** Maryam Haghifam; Zifan He; Jason Cong; Yizhou Sun
>
> **摘要:** Transformer-based LLMs achieve strong results on many language tasks; however, long inputs remain challenging because context windows are finite, and prefill latency and memory grow rapidly with prompt length. Flat token-stream processing and chunk-based retrieval can therefore spend substantial computation and context budget on text unrelated to the query. Offline-indexed RAG additionally introduces external storage and index management overhead, and typically appends retrieved evidence as raw text, increasing prefill cost and latency. H^{2}MT makes long-context inference structure-aware: it builds a semantic hierarchy offline, computes a memory embedding for each node via bottom-up post-order aggregation, and routes queries coarse-to-fine at inference to prune irrelevant branches early. On LongBench QA (NarrativeQA, HotpotQA, QASPER) and two structured technical-document settings, H MT achieves favorable quality efficiency trade-offs, delivering competitive ROUGE-L and F1 (where applicable) with lower peak GPU memory and time-to-first-token (TTFT) than prompt compression, memory-token methods, and retrieval-augmented generation baselines.
>
---
#### [new 137] Improving the Completeness and Comparability of Segment Disclosures: A Large Language Model Approach
- **分类: cs.CL; cs.IR; q-fin.GN**

- **简介: 该论文属于财务信息提取任务，旨在解决段落披露不完整和可比性差的问题。通过大语言模型框架提取并整合段落信息，提升数据完整性与跨公司比较能力。**

- **链接: [https://arxiv.org/pdf/2605.23924](https://arxiv.org/pdf/2605.23924)**

> **作者:** Yue Liu; Zhiyuan Cheng; Longying Lai
>
> **备注:** 39 pages, 4 figures, submitted to Accounting Horizons
>
> **摘要:** Segment-level disclosures are a central component of financial reporting, providing insight into firms' internal organization and the allocation of economic activities across operating units. However, segment information is often presented in both qualitative and quantitative forms, dispersed across tables and narrative sections of Form 10-K filings. Empirical research relying on structured databases faces both completeness and comparability challenges, as some firm-year observations may be missing, nested segment disclosures are not captured, and support for longitudinal and cross-firm comparability is limited. This study develops a large language model-based framework to extract segment disclosures directly from Form 10-K filings and to preserve both reportable and nested segment information. We further design a retrieval augmented system that incorporates information across multiple filings to support comparability. We use two representative settings to demonstrate its application: longitudinal analysis within a firm to interpret segment changes over time, and cross firm alignment of geographic segments across firms with different reporting structures. The results indicate that the artifact accurately extracts segment-level information and effectively addresses questions that require cross-period knowledge, demonstrating the potential of LLM-based approaches to enhance the measurement and interpretation of segment disclosures.
>
---
#### [new 138] Improving Labeling Consistency with Detailed Constitutional Definitions and AI-Driven Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于内容 moderation 任务，旨在解决标签一致性问题。通过 AI 辅助编写详细定义并评估标签，提升分类准确性和一致性。**

- **链接: [https://arxiv.org/pdf/2605.24247](https://arxiv.org/pdf/2605.24247)**

> **作者:** Konstantin Berlin; Adam Swanda
>
> **备注:** Under review at ACL Rolling Review (ARR), May 2026 cycle. Also available at this https URL
>
> **摘要:** Many automated labeling pipelines classify inputs into categories defined by a written specification, content moderation being a prominent use case. Simple category definitions are not detailed enough for labelers to produce the accurate, consistent golden labels these pipelines require. One solution is to write a prescriptive definition that settles enough real boundary cases that labelers cannot disagree with the written interpretation. In practice, definitions at that level of detail exceed what a human annotator can hold in working memory, so annotators fall back on intuition and the labels drift from the written rules, regressing on accuracy and consistency. We propose and demonstrate the efficacy of an AI-driven workflow in which AI helps write a per-category constitution that defines the label in enough detail to cover edge cases, and a frontier LLM interprets it on each input to produce the golden label more consistently and accurately than humans reading the same document. We evaluate on three content moderation categories (harassment, hate speech, non-violent crime) and show that the approach reduces cross-model inconsistency by up to 57x compared to paragraph definitions, with cross-model disagreement diagnosing specification gaps and the human responsible for high-level decisions about what each category should mean rather than individual labeling calls. For the safety evaluation, we introduce a dual-axis formulation scoring intent and content independently over the full conversation, so downstream consumers can act on either axis or both.
>
---
#### [new 139] StreamProfileBench: A Benchmark for Fine-Grained User Profile Inference in Real-World Streaming Scenarios
- **分类: cs.CL**

- **简介: 该论文属于用户画像任务，解决实时流数据下的细粒度用户画像问题。构建了StreamProfileBench基准，提出无标注评估框架，验证了持续更新的挑战。**

- **链接: [https://arxiv.org/pdf/2605.25758](https://arxiv.org/pdf/2605.25758)**

> **作者:** Sizhe Wang; Feiyu Duan; Juelin Wang; Liwen Zhang; Feiyu Duan
>
> **摘要:** Large Language Models (LLMs) have reshaped user profiling, yet current evaluations mainly focus on static data snapshots. This paradigm overlooks the reality of personalized systems, where User-Generated Content (UGC) arrives continuously and fine-grained profile evolve rapidly. To bridge this gap, we introduce StreamProfileBench, a large-scale benchmark for fine-grained streaming user profiling. We formalize streaming user profiling as a continuous state maintenance task and curate a highly authentic dataset comprising over 120,000 UGC posts from 7,000+ real users across five diverse platforms. By leveraging the temporal correlation of user interests, we further propose a novel, annotation-free evaluation framework. Extensive experiments across 14 leading LLMs reveal that continuous profile updating remains an open challenge. Models exhibit a systemic conservative bias, over-retaining past interests while failing to recognize interest decay. Ablation experiments further validate the practical utility and necessity of the streaming paradigm.
>
---
#### [new 140] They Are Not the Same: Direct Causes Are Not Grounded Emotion Explanations
- **分类: cs.CL**

- **简介: 该论文研究情感原因对抽取任务，指出二元分类无法有效解释情绪。工作包括分析数据集、验证任务有效性，并揭示模型可能依赖直接触发而非全面解释。**

- **链接: [https://arxiv.org/pdf/2605.25208](https://arxiv.org/pdf/2605.25208)**

> **作者:** Zhuangzhuang Pan; Yan Xia; Chee Seng Chan
>
> **备注:** 25 pages, 11 figures, 24 tables. Preprint
>
> **摘要:** Emotion-Cause Pair Extraction (ECPE) was introduced to explain why an emotion occurs, but this goal is now often reduced to binary pair/non-pair prediction. This proxy is useful for direct-cause extraction, yet easy to over-read as evidence grounded emotion explanation. We show that this interpretation is only partially valid. In IEMO-MECP, 90.9% of original positives remain emo-cause and 95.0% of original negatives remain non-pair, confirming that the binary ECPE task is largely preserved. The problem is that direct triggers alone do not constitute a grounded explanation. Emo-context, an utterance that helps interpret a target emotion without directly causing it, appears on both sides of the original boundary and is enriched near binary uncertainty, showing that the binary boundary has no stable place for such discourse evidence. Across evaluated ECPE models, direct triggers are recovered more reliably than contextual support. Under shortcut pressure, this imbalance becomes consequential. Binary-trained models assign higher pair scores to nearby lexically similar non-pair candidates than to evidence supported but structurally harder emo-cause and emo-context pairs. Thus, pair scores can reward convenient attributions over grounded explanations. High binary ECPE performance indicates that a model can identify direct triggers; it does not indicate that the model has explained the emotion. Code is publicly available at this https URL.
>
---
#### [new 141] AutoSG: LLM-Driven Solver Generation Solely from Task Prompts for Expensive Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于优化任务，解决昂贵优化中求解器生成的问题。提出AutoSG，通过自然语言生成定制求解器，克服知识不足、结构破坏和评估成本高等问题。**

- **链接: [https://arxiv.org/pdf/2605.25658](https://arxiv.org/pdf/2605.25658)**

> **作者:** Haoran Gu; Handing Wang; Yi Mei; Mengjie Zhang
>
> **摘要:** Expensive optimization tasks are ubiquitous in real-world applications, demanding highly specialized solvers. While LLM-driven automated solver generation shows promise, current paradigms face three critical issues when tackling expensive optimization: factual hallucinations due to deficient domain knowledge, the frequent dismantling of previously established locally optimal structures during refinement, and the prohibitive evaluation costs alongside restricted generalization caused by executing on training instances. To address these issues, we introduce AutoSG, a fully automated workflow directly translating natural language prompts into executable customized solvers. AutoSG features three core innovations: a retrieval-augmented solver generation module strictly grounding code in verified literature; a one-step self-refinement operator introducing task-specific improvements while preserving critical structural components; and an instance-free Elo-based LLM-as-a-Judge evaluation mechanism rapidly establishing global rankings. Extensive evaluations across diverse expensive optimization tasks confirm AutoSG significantly outperforms human-designed state-of-the-art frameworks and existing LLM-generated solvers.
>
---
#### [new 142] On the Limits of Model Merging for Multilinguality in Pre-Training
- **分类: cs.CL**

- **简介: 该论文研究模型合并对多语言预训练的限制，探讨是否可将单语预训练模型合并。任务为多语言模型训练，解决合并导致性能下降的问题，通过实验验证了表示相似性的重要性。**

- **链接: [https://arxiv.org/pdf/2605.25846](https://arxiv.org/pdf/2605.25846)**

> **作者:** Seth Aycock; Fedor Vitiugin; Aleksandr Umnov; Christof Monz; Khalil Sima'an
>
> **备注:** MeLLM Workshop 2026
>
> **摘要:** Endowing models with consistent multilingual performance can be achieved by mixing pre-training data, or post-training approaches such as language-specific model merging. In this work, we test whether merging can be applied to monolingually pre-trained models. We conduct a controlled study on the efficacy of mixed, merged, and monolingual pre-training setups. We find that while monolingual pre-training results in strong in-language performance, merging any combination of monolingual models leads to performance collapse due to interference. Our analysis suggests representational similarity is a prerequisite for model merging. We therefore conclude that the flexibility of merging in fine-tuning does not extend trivially to language-specific pre-training.
>
---
#### [new 143] LLM-as-a-Reviewer: Benchmarking Their Ability, Divergence, and Prompt Injection Resistance as Paper Reviewers
- **分类: cs.CL; cs.CY; cs.ET**

- **简介: 该论文属于人工智能评估任务，研究LLM作为审稿人的可靠性与安全性。旨在解决LLM在学术评审中的偏差与脆弱性问题，通过实验评估其评分准确性、与人类的差异及抗攻击能力。**

- **链接: [https://arxiv.org/pdf/2605.25415](https://arxiv.org/pdf/2605.25415)**

> **作者:** Lingyao Li; Junjie Xiong; Changjia Zhu; Runlong Yu; Chen Chen; Junyu Wang; Renkai Ma; Zhicong Lu
>
> **摘要:** Large language models (LLMs) are increasingly used in academic peer review, yet their reliability, alignment with human judgment, and robustness to adversarial attacks remain poorly understood. We present a systematic benchmark of LLM-as-a-Reviewer on 898 papers stratified from NeurIPS and ICLR, evaluating 12 LLMs along three axes: rating calibration, divergence from human reviewers, and resistance to prompt injection embedded via an invisible font-mapping attack. We find that LLMs systematically overrate weaker submissions and diverge from humans in topical emphasis, under-flagging Clarity and over-flagging Reproducibility, while producing reviews two to three times longer with lower lexical diversity and a more standardized vocabulary. Prompt injection remains highly effective. Simple hidden instructions can promote low-scoring papers to acceptance-level ratings in a substantial fraction of cases, with effectiveness varying sharply across model families. While LLMs offer utility in structuring evaluations, their integration into peer review requires safeguards against both intrinsic biases and adversarial risks.
>
---
#### [new 144] IndexMem: Learned KV-Cache Eviction with Latent Memory for Long-Context LLM Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于长文本推理任务，解决KV缓存过大的问题。通过学习的索引器和潜在记忆模块，优化KV缓存 eviction，提升长序列推理效果。**

- **链接: [https://arxiv.org/pdf/2605.25475](https://arxiv.org/pdf/2605.25475)**

> **作者:** Xintong Yang; Hao Gu; Binxing Xu; Lujun Li; Bei Liu; Jiacheng Liu; Qiyuan Zhu; Sirui Han; Yike Guo
>
> **摘要:** Large Language Models (LLMs) are increasingly expected to operate over long contexts, yet standard softmax attention incurs a KV cache that grows linearly with sequence length, quickly becoming the bottleneck for long context inference. A practical remedy is to evict less important KV entries; however, existing eviction policies are largely heuristic and struggle to capture the rich, input-dependent distribution of token importance. In this work, we introduce a learnable indexer that predicts KV importance, enabling more accurate retention of critical tokens. Meanwhile, naively evicting tokens permanently discards their information, leading to irreversible forgetting and degraded retrieval over long ranges. To address this, we propose a lightweight latent memory module that compresses evicted tokens into a compact, online-updated state and provides residual readouts to compensate for the attention contributions lost through KV eviction. Collectively, our method enables accurate long-context inference under a bounded KV budget, delivering consistent improvements on RULER (4K/16K) across Qwen, Mistral, and Llama models (up to 25 points under aggressive eviction), markedly more stable Needle-in-a-Haystack retrieval, and superior LongBench scores and compression curves compared to existing eviction policies.
>
---
#### [new 145] Inference Time Optimization with Confidence Dynamics
- **分类: cs.CL**

- **简介: 该论文属于语言模型推理优化任务，旨在解决模型不确定性对推理结果的影响问题。通过分析置信度动态变化，提出CDG投票方法提升答案选择性能。**

- **链接: [https://arxiv.org/pdf/2605.25244](https://arxiv.org/pdf/2605.25244)**

> **作者:** Yu Wang; Minghao Liu; Jiayun Wang; Jinrui Huang; Ankit Shah; Wei Wei
>
> **备注:** Published in ICML 2026
>
> **摘要:** Inference time optimization techniques, such as repeated sampling, have significantly advanced the reasoning capabilities of Large Language Models (LLMs). However, the critical role of model uncertainty remains largely underexplored in these optimization strategies. In this paper, we investigate the dynamics of confidence along reasoning trajectories and for first time reveal a surprising and unique pattern: correct answer traces tend to exhibit confidence improvement over time (positive confidence gain), while incorrect traces show attenuated or declining confidence as reasoning proceeds. Based on this observation, we propose Confidence Dynamic Gain (CDG) based voting, which incorporates how the confidence trajectory of the response evolves along the reasoning chain. Experiments across four open-source architectures (DeepSeek-R1, gpt-oss, Gemma-3, Qwen-QwQ) on the AIME24/25, HMMT25, and BRUMO25 benchmarks demonstrate that CDG yields a significant performance boost over baselines. These results demonstrate that our method provides a robust discriminative signal for improving answer selection in LLM reasoning. We also provide theoretical insights for this phenomenon. Code will be released at this https URL.
>
---
#### [new 146] Iterate Until Retrieved: Factual Nugget Optimization for Discoverable Continual Corrections in Agentic RAG
- **分类: cs.CL**

- **简介: 该论文属于知识增强生成任务，解决B2B场景中如何有效发现和优化事实性修正的问题。通过迭代优化事实片段，提升系统对纠正信息的检索能力。**

- **链接: [https://arxiv.org/pdf/2605.25641](https://arxiv.org/pdf/2605.25641)**

> **作者:** Moshe Hazoom; Gal Patel; Alon Talmor; Tom Hope
>
> **摘要:** Agentic retrieval-augmented generation (RAG) systems in complex B2B (business-to-business) settings may often receive free-form response feedback. Rather than generic feedback signals such as style, preference, or overall response quality, we focus on actionable factual corrections. We identify these instances and convert them into compact knowledge-base entries, which we call factual nuggets. We introduce Iterative Nugget Optimization (INO), an index-time optimization method that uses the production agentic RAG as a test harness: it creates an initial nugget, probes it with the triggering query and paraphrases, reflects over failed retrieval and answer traces, and revises the nugget until it is discoverable. We evaluate INO with two production B2B knowledge-assistance agents across multiple companies that use our system: a product support agent that answers questions over company-specific knowledge bases, and a support ticket agent that assists support engineers. INO consistently improves results over baselines in terms of discoverability and usage of factual corrections, in automated and human evaluations.
>
---
#### [new 147] The Tokenizer Tax Across 25 European Languages: Domain Invariance, Cross-Lingual Few-Shot Effects, and the Ukrainian Penalty
- **分类: cs.CL**

- **简介: 该论文研究多语言NLP中的分词器效率问题，分析25种欧洲语言的分词成本，揭示语言差异对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2605.24718](https://arxiv.org/pdf/2605.24718)**

> **作者:** Volodymyr Ovcharov
>
> **备注:** 16 pages, 3 figures, 8 tables. Dataset: this https URL
>
> **摘要:** Tokenizer fertility the number of tokens per word imposes a hidden cost on non-English NLP. We measure fertility for ten foundation models across 25 European languages on parallel text, producing the first controlled tokenizer tax map for the continent. The tax spans 2.5x from English (1.2 tokens/word) to Greek/Maltese (~3.1), following a clear hierarchy: Romance (1.5-1.7), Germanic (1.7-1.9), Slavic (2.2-2.5), Uralic/Baltic (2.7-3.0). Ukrainian (2.7) pays 15-18% more than cognate Slavic languages, reflecting underrepresentation in pre-training data. Fertility rankings are domain-invariant across three text registers (rho > 0.97). A subword analysis reveals that high-fertility tokenizers fragment morphological boundaries rather than preserving them. Cross-lingual few-shot evaluation on four Slavic languages shows that few-shot effects are model-intrinsic, not language-dependent. We release all measurements as a public dataset.
>
---
#### [new 148] READER: Reasoning-Enhanced AI-Generated Text Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI生成文本检测任务，旨在解决现有检测器在分布变化下性能下降的问题。工作是提出READER，通过推理增强实现更准确的检测。**

- **链接: [https://arxiv.org/pdf/2605.25281](https://arxiv.org/pdf/2605.25281)**

> **作者:** Pingfan Su; Kai Ye; Shijin Gong; Erhan Xu; Jin Zhu; Giulia Livieri; Chengchun Shi
>
> **摘要:** Recent advances in large language models (LLMs) have made it increasingly difficult to distinguish human-written text from AI-generated content. Many existing detectors train supervised neural classifiers that achieve strong in-distribution performance but are often opaque and can degrade substantially under distribution shift. We present READER, a reasoning-enhanced AI text detector that outputs both a human/AI label and a structured rationale describing the evidence for its decision. A key component of our approach is READ, a curated supervision set of rationales and verdicts. We fine-tune an LLM on READ to build READER, which reasons before detecting at inference time. Despite having only 1.5B parameters, READER consistently outperforms existing detectors as well as prompted, high-capacity LLM baselines (GPT-5.2, Gemini-3-Pro, and DeepSeek-V3.2), which are 100 to 1000 times larger in scale.
>
---
#### [new 149] BC Protocol: Structured Dual-Expert Dialogue for Eliciting High-Quality Chain-of-Thought Post-Training Data
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型后训练数据生成任务，旨在解决高质量思维链数据不足的问题。提出BC协议，通过双专家协作生成更自然的推理过程。**

- **链接: [https://arxiv.org/pdf/2605.25549](https://arxiv.org/pdf/2605.25549)**

> **作者:** Bo Zou; Chao Xu
>
> **摘要:** High-quality expert chain-of-thought (CoT) data is one of the core bottlenecks in large language model (LLM) post-training. Existing data production methods each have structural limitations: crowdsourced annotation lacks deep reasoning paths; expert solo writing is constrained by the "expert blind spot" -- experts structurally skip reasoning steps they consider obvious; RLHF only produces preference signals rather than reasoning chains. This paper proposes the BC Protocol -- a structured dual-expert elicitation method for LLM post-training data production. The method carefully pairs a domain expert (crystallized intelligence) with a knowledge engineer (fluid intelligence), systematically externalizing the expert's implicit judgments as natural language reasoning chains. We introduce the Participant Aptitude Model, which defines six participant characteristic dimensions that affect elicitation quality. "Calibrated Ignorance" is an original concept proposed in this paper. We further propose "Selection-over-Prescription" as a methodological principle: for implicit knowledge elicitation tasks, investing quality-control resources in personnel selection yields a higher return than investing the same resources in process design. In a controlled experiment in the narrative fiction domain, we directly compared CoT produced by BC Protocol dual dialogue (Group A, (n=20)) against CoT written independently by the same domain expert (Group B, (n=20)). Three cross-vendor judge models -- GPT-4o, Claude Opus 4.5, and Gemini 2.5 Pro -- conducted blind evaluation across five dimensions (600 ratings total). Results show that the BC Protocol achieves an overwhelming advantage in "naturalness of reasoning process" (Group A mean 4.80 vs. Group B mean 1.30, (p=2.4\times10^{-8}), Cliff's (\delta=1.0)).
>
---
#### [new 150] What Makes a Medical Checker Trainable? Diagnosing Signal Collapse and Reward Hacking in Checker-Guided RAG for Biomedical QA
- **分类: cs.CL**

- **简介: 该论文研究医学问答中的验证器训练问题，解决信号崩溃和奖励黑客问题。通过对比不同检查器在RAG系统中的表现，分析其对模型训练的影响。**

- **链接: [https://arxiv.org/pdf/2605.25988](https://arxiv.org/pdf/2605.25988)**

> **作者:** Yuelyu Ji; Min Gu Kwak; Hang Zhang; Xizhi Wu; Chenyu Li; Yanshan Wan
>
> **摘要:** Medical RAG needs evidence-grounded claims, so plugging a claim-level NLI checker into retrieval-augmented RL is intuitive. \textbf{We find that the checker's \emph{output distribution} during training, not its held-out accuracy, decides whether it provides trainable gradient.} We compare four NLI checker back-ends as process rewards inside a GRPO-trained medical RAG agent (Qwen2.5-7B, replicated on Qwen3-4B and Llama-3.1-8B) across four held-out medical QA benchmarks. Three diagnostic findings emerge. \textbf{(i)} Signal collapse is log-prob-specific: LLM log-probability scoring labels over 97\% of claims neutral -- collapsing the RL gradient to zero -- while a calibrated MedNLI classifier scores the same pairs non-degenerately. \textbf{(ii)} Moderate signal beats strong signal on answer quality: a strong proprietary checker triggers a three-step reward-hacking cascade -- ultra-short answers, search avoidance, language collapse -- so a moderate-signal local classifier trains a higher-quality model (\textbf{+12\% BERTScore over zero-shot, no GPT dependency}). \textbf{(iii)} Signal strength is policy-dependent: the same checker registers as moderate on one policy but strong on another without triggering the cascade end-state. We frame these as boundary conditions for verifier-as-reward systems.
>
---
#### [new 151] Double Triangle Annotation: A Scalable Human-in-the-Loop Framework for High-Precision Historical Document Annotation
- **分类: cs.CL**

- **简介: 该论文提出Double Triangle Annotation框架，解决历史文档高精度标注问题。通过双模型共识减少人工干预，提升标注效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.25781](https://arxiv.org/pdf/2605.25781)**

> **作者:** Yi Ren
>
> **备注:** 12 pages, 4 figures. ACL ARR 2026 March submission
>
> **摘要:** Evaluating structured-information extraction from historical documents at scale requires high-precision ground-truth annotations, yet traditional manual labeling is expensive and fully automated pipelines built on large language models are prone to hallucination. We propose Double Triangle Annotation, a two-layer human-in-the-loop framework that leverages cross-model consensus to automate the majority of annotation work while ensuring high-precision outputs. In the first layer, two architecturally independent Multimodal Large Language Models annotate each document in parallel; when they agree, the label is auto-accepted, and disagreements are routed to a human jury. A second layer cross-checks two such systems against each other, escalating residual conflicts to a domain expert. The framework rests on a single assumption -- error independence between models -- requires no distributional priors or task-specific calibration, and becomes more autonomous as model capability improves. On the Guides Rosenwald, a corpus of French medical directories spanning 1887-1906, the framework achieves a final Word Error Rate of 0.003. Applied at scale, model consensus auto-accepts over 85% of 13,595 fields. We release the resulting benchmark -- the first structured-extraction ground truth for the Rosenwald Guides -- to support future work on historical document processing.
>
---
#### [new 152] GroupTravelBench: Benchmarking LLM Agents on Multi-Person Travel Planning
- **分类: cs.CL**

- **简介: 该论文提出GroupTravelBench，用于评估LLM在多人旅行规划中的能力，解决多用户冲突与协调问题。工作包括构建基准任务和模拟环境，评估模型在偏好获取、协调和规划方面的能力。**

- **链接: [https://arxiv.org/pdf/2605.25200](https://arxiv.org/pdf/2605.25200)**

> **作者:** Xiang Cheng; Yulan Hu; Lulu Zheng; Zheng Pan; Xin Li; Yong Liu
>
> **备注:** work in process
>
> **摘要:** Travel planning is a realistic task for evaluating the planning and tool-use abilities of LLM agents. However, existing benchmarks typically assume only a single user, thereby avoiding one of the most challenging aspects of real-world scenarios: an agent's ability to identify and resolve conflicts among multiple users. To address this gap, we introduce \textbf{GroupTravelBench}, the first benchmark for \textbf{multi-user, multi-turn} travel planning. Based on real user profiles, POI data, and ticket price data, we synthesize 650 tasks and divide them into three difficulty levels. Beyond standard abilities in single-user itinerary planning, such as multi-step reasoning and tool use, our benchmark further evaluates three key capabilities required for travel agents: \emph{(i) elicitation} -- proactively engaging in multi-turn dialogue to gather preferences from each user; \emph{(ii) coordination} -- resolving conflicts among users through compromise or subgrouping strategies; and \emph{(iii) planning} -- searching for travel plans that maximize overall group utility while maintaining fairness and feasibility. To simulate real-world conversational itinerary planning while enabling reliable tool use and offline evaluation, we build an interactive sandbox environment with cached real-world tool data. We evaluate a wide range of LLMs and find that even frontier models still show substantial weaknesses in preference coverage and group fairness. \textit{GroupTravelBench} provides a practical and reproducible benchmark for advancing research on LLM agents for real-world travel planning.
>
---
#### [new 153] DVAO: Dynamic Variance-adaptive Advantage Optimization for Multi-reward Reinforcement Learning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多奖励强化学习任务，旨在解决传统方法在多奖励设置下的训练不稳定问题。提出DVAO算法，动态调整奖励组合权重，提升训练稳定性和性能。**

- **链接: [https://arxiv.org/pdf/2605.25604](https://arxiv.org/pdf/2605.25604)**

> **作者:** Guochao Jiang; Jingyi Song; Guofeng Quan; Chuzhan Hao; Guohua Liu; Yuewei Zhang
>
> **摘要:** Reinforcement Learning has become a standard paradigm for aligning Large Language Models with human intent and task requirements. While Group Relative Policy Optimization offers an efficient, value-model-free alternative to Proximal Policy Optimization, adapting it to real-world multi-reward settings remains challenging. Standard scalarization practices, such as Reward Combination and Advantage Combination, suffer from significant drawbacks: Reward Combination frequently generates advantages with excessively large squared magnitudes that lead to training instability, while Advantage Combination relies on static hyperparameters and ignores cross-objective correlations. To address these limitations, we propose Dynamic Variance-adaptive Advantage Optimization (DVAO), which dynamically adjusts combination weights based on the empirical reward variance of each objective within a rollout group, effectively up-weighting objectives with a stronger learning signal while suppressing noisy ones. We mathematically prove that DVAO maintains bounded advantage magnitudes for stable training and introduces a self-adaptive cross-objective regularization mechanism. Extensive experiments on mathematical reasoning and tool-use benchmarks using Qwen3 and Qwen2.5 models demonstrate that DVAO significantly outperforms baseline methods, achieving a superior multi-objective Pareto frontier and robust training stability.
>
---
#### [new 154] When Reasoning Hurts: Source-Aware Evaluation of Frontier LLMs for Clinical SOAP Note Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于临床文档生成任务，研究推理能力对SOAP笔记生成的影响。通过对比不同模型在有无推理和检索增强下的表现，发现强推理未必提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.24902](https://arxiv.org/pdf/2605.24902)**

> **作者:** Faizan Faisal
>
> **摘要:** Reasoning-enabled LLMs perform strongly on medical reasoning benchmarks, but it remains unclear whether these gains transfer to structured clinical documentation; we investigate this question using SOAP note generation from clinical dialogue in a source-aware benchmark spanning OMI Health, ACI-Bench, and PriMock57. We evaluate GPT-5.4, DeepSeek-V4-Flash, and Gemma-4-E4B in a controlled 2x2 design that independently toggles provider-native reasoning and same-source retrieval-augmented generation (RAG). Outputs are assessed using seven automatic metrics alongside two reference-aware LLM judges. Both evaluation approaches agree that a non-reasoning GPT-5.4 configuration achieves the highest overall quality, while DeepSeek-V4-Flash performs best among reasoning-enabled configurations. Enabling reasoning significantly degrades GPT-5.4 performance across all three datasets, whereas same-source RAG yields smaller, model-dependent improvements. Overall, the findings indicate that stronger reasoning capability should not be assumed to improve fidelity-sensitive SOAP note generation without dedicated, task-specific evaluation.
>
---
#### [new 155] DTO: a Differentiable Training Objective for Effective Counterfactual Story Rewriting
- **分类: cs.CL**

- **简介: 该论文针对反事实故事重写任务，解决模型难以捕捉细微修改的问题。提出一种可微训练目标DTO，通过联合优化参考重写和语义一致性，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2605.24885](https://arxiv.org/pdf/2605.24885)**

> **作者:** Amelia Girard; Massimo Piccardi
>
> **备注:** 11 pages, 2 figures
>
> **摘要:** Counterfactual story rewriting is a natural language processing task that requires updating an existing story to reflect a chosen alternative event, yet preserving all the unaffected storyline elements and overall coherence. While large language models have recently made remarkable progress on this task, it still remains challenging since the required modifications are typically very small in size and highly localized. As a consequence, models trained in a conventional manner with the maximum-likelihood training objective tend to overlook these nuances. At the same time, more sophisticated training approaches based on reinforcement learning are notoriously slow and difficult to set up. For these reasons, our paper proposes a novel, differentiable training objective (DTO) that directly optimizes for the requisite counterfactual improvements. In our approach, a transformer model is fine-tuned via end-to-end backpropagation against a fully differentiable loss function that jointly rewards (i) fidelity to the reference rewrite and (ii) semantic consistency with the source narrative. The empirical evaluation on the TimeTravel and ART datasets shows that the proposed DTO approach has been able to surpass a maximum-likelihood baseline and a preference-based approach, and perform competitively against two contemporary large language models in all evaluation metrics. These findings substantiate the effectiveness of task-specific differentiable objectives for nuanced, controlled text-generation tasks.
>
---
#### [new 156] Beyond Literal Translation: Evaluating Cultural Effectiveness in Social Media UGC
- **分类: cs.CL**

- **简介: 该论文属于社会媒体翻译任务，解决文化有效性和情感共鸣评估问题。构建了CULTURE-MT基准，提出文化有效性评价指标，并测试多个模型表现。**

- **链接: [https://arxiv.org/pdf/2605.25626](https://arxiv.org/pdf/2605.25626)**

> **作者:** Linjuan Wu; Ruiqi Zhang; Xinze Lyu; Ye Guo; Daoxin Zhang; Zhe Xu; Yao Hu; Yixin Cao; Yongliang Shen; Weiming Lu
>
> **备注:** Accepted by ICML2026
>
> **摘要:** Social media platforms enable large-scale cross-lingual communication, but translating user-generated content (UGC) remains challenging due to its informal style, cultural references, and interaction-based expressions. While recent LLMs have improved translation quality, existing benchmarks and metrics often fail to capture whether translations convey intended meaning and cultural resonance in real-world settings. In this work, we introduce CULTURE-MT, a benchmark for social media translation that focuses on both CULtural Transmission and UGC-specific emotion REsonance. CULTURE-MT consists of 1,002 UGC notes across 14 domains, categorized into four types based on culture-loaded symbols and linguistic style features. We also construct UGC-oriented training data to fine-tune Qwen3-8B and Qwen3-32B as baselines. We propose cultural effectiveness as a new evaluation criterion, focusing on expression accuracy and cultural adaptability. Testing 15 models, including the baselines, we find that traditional metrics fail to capture cultural effectiveness. We also observe that cultural effectiveness on base LLMs correlates with model size. Our work provides a comprehensive evaluation system for UGC translation models and will offer an open evaluation platform to advance research in this area. We release the CULTURE-MT benchmark and provide an online leaderboard where submitted translation results can be evaluated by our trained JUDGER.
>
---
#### [new 157] Causal Tongue-Tie: LLMs Can Encode Causal Direction, But Their Yes/No Outputs Fail to Express
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在因果推理任务中的表现。发现模型内部能编码正确因果关系，但输出的Yes/No答案却受常识干扰，导致错误。任务是验证模型是否真正理解因果关系，工作是通过实验揭示其内部表示与输出间的不一致。**

- **链接: [https://arxiv.org/pdf/2605.25891](https://arxiv.org/pdf/2605.25891)**

> **作者:** Ziyi Ding; Xiao-Ping Zhang
>
> **摘要:** We find a mismatch between what large language models encode about a causal question and what they answer. On anti-commonsense CLadder items, a fixed linear probe recovers the evidence-supported answer from the model's hidden state (accuracy approximately 0.97), while the spoken Yes/No reverts to the commonsense one (accuracy approximately 0.5). We call this approximately +0.5 gap Causal Tongue-Tie: a wrong Yes/No decomposes into two separable failure modes: no internal signal versus a signal the verbal interface cannot say. The implication cuts both ways for output-only causal benchmarks: a benchmark "correct" need not mean the model has understood, and a benchmark "wrong" need not mean it cannot. Sweeping claims about whether LLMs can do causal reasoning, drawn from a single accuracy number, deserve a second look.
>
---
#### [new 158] Can LLMs Time Travel? Enhancing Temporal Consistency in Legal Agentic Search through Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律信息检索任务，解决法律LLM时间一致性问题。针对法律条款时效性要求，提出LegalSearch-R1框架，结合本地法规检索与网络搜索，提升法律推理准确性。**

- **链接: [https://arxiv.org/pdf/2605.25920](https://arxiv.org/pdf/2605.25920)**

> **作者:** Wei Fan; Yining Zhou; Mufan Zhang; Yanbing Weng; Yiran HU; Tianshi Zheng; Baixuan Xu; Chunyang Li; Jianhui Yang; Haoran Li; Yangqiu Song
>
> **备注:** Under Review
>
> **摘要:** While large language models (LLMs) augmented with agentic search capabilities show promise for legal reasoning, they overlook a fundamental constraint that applicable law must match the temporal context of each case, as retroactive application of statutes violates core legal principles and leads to erroneous conclusions. Our observations reveal that current legal LLMs suffer from temporal bias anchored to their training cutoff, while search agents rarely incorporate temporal constraints into queries, and that web search alone cannot provide the precise statute and precedent citations that legal reasoning demands. To address these challenges, we propose LegalSearch-R1, an end-to-end reinforcement learning framework that pairs local statute RAG for precise article matching with online web search for broader legal knowledge, trained on temporally-indexed data spanning multiple amendment periods to enforce temporal consistency. Extensive experiments on our benchmark covering 13 legal tasks demonstrate that our 7B-parameter agent outperforms state-of-the-art deep research frameworks and specialized legal LLMs by 12.9% to 29.8%, surpasses baselines by 57.7% to 80.3% on temporal consistency, and exhibits robust out-of-domain generalization. The code and data are available at this https URL.
>
---
#### [new 159] WhoSaidIt: Human-LLM Collaborative Annotation for Text-Based Multilingual Speaker-Attribute Classification
- **分类: cs.CL**

- **简介: 该论文属于文本多语言说话人属性分类任务，解决标注歧义问题。通过人机协作框架改进标注一致性，构建多语言数据集并分析模型表现。**

- **链接: [https://arxiv.org/pdf/2605.26070](https://arxiv.org/pdf/2605.26070)**

> **作者:** Lingyu Gao; Will Monroe; David Smith; Meghan Jemison; Jackie Lee
>
> **备注:** 16 pages in total
>
> **摘要:** Annotating speaker attributes from text is inherently ambiguous, particularly in multilingual settings where demographic and social cues are implicit and culturally variable. We propose a human-large language model (LLM) collaborative re-annotation framework for stabilizing multilingual speaker-attribute labels under practical resource constraints. Starting from a noisy corpus, we use LLMs to surface recurring annotation rationales through iterative interaction with experts, and apply disagreement-focused sampling for targeted re-annotation. Using this framework, we construct WhoSaidIt, a multilingual dataset covering nine speaker-attribute labels. We quantify divergence between original and revised annotations, benchmark recent LLMs, and analyze the effect of explicit rationales on model behavior. Our results reveal substantial cross-lingual differences in annotation decisions and demonstrate both the strengths and limitations of LLMs in speaker-attribute classification.
>
---
#### [new 160] A Multi-Agent LLM Framework for Rating the Quality of Surgical Feedback
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于医疗教育评估任务，旨在解决手术反馈质量评估难题。通过构建多智能体LLM框架，自动识别并评分手术反馈质量，提升教学效果。**

- **链接: [https://arxiv.org/pdf/2605.25440](https://arxiv.org/pdf/2605.25440)**

> **作者:** Rafal Kocielnik; J. Everett Knudsen; Steven Y. Cen; Jasmine Lin; Cherine H. Yang; Atharva Deo; Ujjwal Pasupulety; Peter Wager; Anima Anandkumar; Andrew J. Hung
>
> **备注:** 25 pages, 3 figures
>
> **摘要:** Verbal feedback delivered by attending surgeons in the operating room plays a critical formative role in resident trainee skill acquisition. Yet, assessing the quality of trainer feedback and its effectiveness in influencing trainee behavior during live surgery remains a challenge. Prior studies assessed feedback content relying on extensive manual annotation by expert human raters and focused on developing broad taxonomies that overlook the qualitative aspects of feedback delivery such as clarity or urgency. Limited existing automated methods, including keyword analysis and topic modeling, also fail to capture these nuanced aspects. We introduce a two-stage LLM-based framework that discovers interpretable feedback quality criteria grounded in the context of surgical training. Our method uses multi-agent prompting and surgical domain knowledge injection to discover a small set of human interpretable scoring criteria (e.g., Encouraging, Urgent, Clear). These criteria are then used to automatically score live surgical feedback via an LLM-as-a-judge approach. Evaluation on 4.2k trainer feedback instances demonstrates that our AI-discovered criteria outperform prior content-based frameworks in predicting feedback effectiveness, including observed trainee behavioral adjustments and trainer approval. This work advances scalable, human-aligned assessment of communication quality in the operating room and provides a foundation for improving surgical teaching practices.
>
---
#### [new 161] Learning to Reason Efficiently with A* Post-Training
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言推理任务，旨在提升大模型的推理准确性与效率。通过A*搜索指导训练，使模型生成更正确且高效的推理过程。**

- **链接: [https://arxiv.org/pdf/2605.24597](https://arxiv.org/pdf/2605.24597)**

> **作者:** Andreas Opedal; Francesco Ignazio Re; Abulhair Saparov; Mrinmaya Sachan; Bernhard Schölkopf; Ryan Cotterell
>
> **备注:** Preprint
>
> **摘要:** Many applications of large language models (LLMs) require deductive reasoning, yet models frequently produce incorrect or redundant inference steps. We frame natural language inference as a search problem where the final answer is the valid proof itself, requiring a reasoning procedure in which intermediate inferences are correct. Specifically, we investigate whether LLMs can learn to generate correct and efficient proofs with guidance from A* search -- an algorithm that guarantees an optimally efficient path to a goal. We explore two training techniques: supervised fine-tuning on execution traces from A* and reinforcement learning with A*-informed process reward models. Empirically, we find that Llama-3.2 models in the 1B--3B range benefit substantially from A* post training, going from near-zero accuracy to outperforming DeepSeek-V3.2 -- a much larger model. Our analysis uncovers a trade-off: while simple correctness rewards maximize accuracy, A*-informed signals strike a balance between accuracy and efficiency. Furthermore, we find that on larger search spaces, models trained with imperfect heuristics exhibit superior accuracy. Our results demonstrate a promising direction towards reasoning guided by principles derived from classical search algorithms.
>
---
#### [new 162] Privacy-Preserving Local Language Models for Longitudinal Data Retrieval in Chronic Dermatologic Disease: Implementation in Pemphigus Patients
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医疗文本摘要任务，旨在解决慢性皮肤病长期随访数据难以高效处理的问题。通过部署隐私保护的小型语言模型，实现临床特征提取与总结生成。**

- **链接: [https://arxiv.org/pdf/2605.25020](https://arxiv.org/pdf/2605.25020)**

> **作者:** Abdurrahim Yilmaz; Ayşe Esra Koku Aksu; Duygu Yamen; Vefa Asli Erdemir; Mehmet Salih Gurel; Gulsum Gencoglan; Joram M. Posma; Burak Temelkuran
>
> **摘要:** Chronic dermatologic diseases such as pemphigus require long-term follow-up, generating extensive longitudinal clinical documentation that is difficult to review comprehensively during routine visits and increasing clinician workload as well as the risk of missing critical historical information. We evaluated whether a locally deployed, privacy-preserving small language model (SLM) could retrieve structured clinical features and generate longitudinal summaries from long-term dermatology follow-up records. In this retrospective case series, thirty pemphigus patients contributed 541 visit notes that were aggregated into full longitudinal records (89,336 words); 56 clinically relevant features were annotated by two expert dermatologists. The locally deployed SLM (Qwen3 4B Thinking 2507) was queried with each complete record to retrieve 56 features and generate one final report summaries. Across 1,680 feature retrieval tasks, mean accuracy was 82.25%. Dermatologists' ratings of AI-generated summaries were high for overall quality (8.23-8.47), clinical accuracy (7.93-8.20), and usefulness (8.47-8.50), with no significant inter-evaluator differences and an overall preference for AI summaries in 53.3% of evaluations. These findings suggest that privacy-preserving, locally deployed SLMs can outperform medical experts and reliably generate clinically meaningful longitudinal summaries. SLMs may support clinical decision-making when integrated with appropriate oversight.
>
---
#### [new 163] Residual Drift Dominates Contradiction in Multi-Turn Constraint Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究多轮推理系统中的错误类型，指出主要问题不是逻辑矛盾而是状态漂移。通过构建基准测试，验证了修复方法的效果，并强调需单独验证答案是否符合状态。任务属于人工智能推理可靠性研究。**

- **链接: [https://arxiv.org/pdf/2605.23940](https://arxiv.org/pdf/2605.23940)**

> **作者:** Sebastien Kawada
>
> **备注:** Published at ICLR 2026 Workshop on Reasoning and Planning for LLMs. 18 pages. ICLR page: this https URL Code: this https URL
>
> **摘要:** How do multi-turn reasoning systems fail? The expected answer is logical contradiction, in which the system's maintained state becomes unsatisfiable. We show that the dominant mode is instead satisfiable drift, where the internal state stays consistent while the returned answer silently violates prior commitments. We build DRIFT-Bench (Decomposing Reasoning Into Failure Types), a solver-instrumented benchmark of 816 test problems across three constraint domains, and evaluate four methods on it across four open-weight models (8B-120B parameters). MUS-Repair, which feeds minimal unsatisfiable subsets back to the generator, is strongest in every setting (+1.8 to +15.0 pp over the best non-MUS baseline). But the central finding is what repair leaves behind. After structured feedback, models rarely contradict themselves. They forget. Residual errors are 98-100% satisfiable drift across all settings, while contradiction drops to near zero. Reliable multi-turn systems must separately validate that the returned answer respects the maintained state. Code is available at this https URL.
>
---
#### [new 164] Measuring Reasoning Quality in LLMs: A Multi-Dimensional Behavioral Framework
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在解决LLMs推理质量评估不足的问题。提出多维行为框架，从六个维度评估模型推理质量，揭示单一准确率指标无法捕捉的行为差异。**

- **链接: [https://arxiv.org/pdf/2605.24661](https://arxiv.org/pdf/2605.24661)**

> **作者:** Ali Şenol; Garima Agrawal; Huan Liu
>
> **摘要:** LLMs have achieved remarkable success in complex reasoning tasks, yet current evaluation approaches predominantly rely on final-answer correctness, offering limited insight into the underlying reasoning processes that produce those answers. To address this gap, this study proposes a unified multi-dimensional framework for measuring reasoning quality in LLMs from a behavioral perspective, operationalizing six theoretically grounded dimensions: Correctness (CQ), Consistency (CS), Robustness (RS), Logical Coherence (LS), Efficiency (ES), and Stability (SS). Extensive experiments on seven LLMs across 975 items from four benchmarks demonstrate that the framework reveals behaviors invisible to accuracy-only metrics. Notably, logical coherence is orthogonal to correctness (r = -0.172, ns), confirming that correct answers can arise from incoherent reasoning, while Claude-Haiku-4.5 achieves the highest multi-dimensional score (Q_bal = 0.778). Furthermore, the framework exposes critical ranking inversions: DeepSeek-V3 ranks second under accuracy-priority but fifth under legal/compliance weighting, a reversal that single-metric evaluation cannot detect. Discriminant validity confirms 11/15 dimension pairs are independent (|r| < 0.50), providing psychometric support for treating each dimension as a distinct signal. The dimensional profiles produced by the framework directly support three classes of deployment decision: identifying models whose reasoning traces would fail accountability audits despite correct final answers (LS--CQ orthogonality); preventing ranking errors caused by accuracy-only benchmarking; and ensuring that no single metric silently substitutes for the six independent signals the framework captures.
>
---
#### [new 165] Fundamental Limitation in Explaining AI
- **分类: cs.AI; cs.CL; cs.CY; cs.IT**

- **简介: 该论文属于AI可解释性研究任务，探讨AI解释的理论局限。提出AI解释的四重困境，表明无法同时满足环境复杂性、性能优良、解释可理解与完全忠实，指出需在实际应用中放弃完全忠实。**

- **链接: [https://arxiv.org/pdf/2605.24727](https://arxiv.org/pdf/2605.24727)**

> **作者:** Atsushi Suzuki; Jing Wang
>
> **摘要:** While large-scale models such as LLMs and diffusion models have achieved practical success, public institutions have emphasized the importance of explainability in AI. Existing methods for explaining AI, however, are not designed to provide completely faithful explanations of the behavior of large-scale AI systems. Although a completely faithful and interpretable explanation of the behavior of an AI system might be useful for AI governance, it has not been known whether providing such an explanation is theoretically possible. In this paper, we mathematically prove a fundamental quadrilemma in explaining AI, stating that AI and its explanation cannot satisfy the following four conditions simultaneously: 1) the complexity of the operation environment, 2) the goodness of the AI's performance, 3) the interpretability of the AI's explanation, and 4) the complete faithfulness of the AI's explanation. This quadrilemma suggests that, in most applications where we cannot change the environment or sacrifice good AI performance and an interpretable explanation, we should give up complete faithfulness of explanations and should instead aim to explain only the parts that are important for applications. As a consequence, the quadrilemma implies that AI governance should be designed on the premise that the faithfulness of AI explanations is always incomplete.
>
---
#### [new 166] Momentum Streams for Optimizer-Inspired Transformers
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在改进Transformer模型。通过引入优化器思想，设计了多种新型Transformer结构，提升性能并减少遗忘。**

- **链接: [https://arxiv.org/pdf/2605.24425](https://arxiv.org/pdf/2605.24425)**

> **作者:** Jingchu Gai; Nai-Chieh Huang; Jiayun Wu
>
> **摘要:** The residual update of a pre-norm Transformer layer admits an interpretation as one step of a first-order optimizer acting on a surrogate token energy, wherein the attention and MLP sublayers function as gradient oracles. Based on this observation, we build a family of optimizer-inspired Transformers (triple-momentum, Adam/AdamW, Muon, SOAP) and compare them under matched compute. In our main pretraining experiment, the triple-momentum TMMFormer achieves the lowest validation loss, outperforming the vanilla Transformer and prior architectural variants. A controlled ablation and supporting theory show that momentum, not preconditioning, is the main source of the gain. We further show that TMMFormer and other momentum-based designs reach flatter minima than the vanilla Transformer, which leads to less forgetting and better generalization.
>
---
#### [new 167] TTPrint: Evidence-Grounded TTP Extraction via Diverge-then-Converge Verification
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文提出TTPrint，解决CTI中MITRE ATT&CK技术提取问题，通过分阶段验证提高准确性和召回率。**

- **链接: [https://arxiv.org/pdf/2605.25836](https://arxiv.org/pdf/2605.25836)**

> **作者:** Yutong Cheng; Changze Li; Raihan Sultan Pasha Basuki; Qian Cui; Wei Ding; Peng Gao
>
> **备注:** Preprint
>
> **摘要:** Extracting MITRE ATT&CK techniques from cyber threat intelligence (CTI) reports is an open-set, multi-label problem requiring both high recall (not missing techniques) and high precision (not hallucinating unsupported ones). Existing methods--rule-based, supervised, and LLM-based--struggle to achieve both: rule-based and supervised approaches lack generalizability across diverse attack descriptions, while LLM-based approaches that couple candidate generation and validation within a single inference step suffer from limited recall and precision simultaneously. We propose TTPrint, which addresses this challenge through a diverge-then-converge design inspired by how human analysts work: first extracting broadly, then verifying rigorously. In the divergent phase, reports are decomposed into atomic behaviors and candidate techniques are proposed broadly. A deterministic span localization stage then anchors each candidate to a specific evidence window in the source text. A convergent verification stage retains only candidates supported by both the localized evidence and the authoritative MITRE definition. We contribute two evaluation resources--a cleaned TRAM benchmark (TRAM-Clean) and a new annotated dataset (TTPrint-Bench)--to address known annotation noise in existing benchmarks and elevate the task to document-level TTP extraction. On TRAM-Clean and TTPrint-Bench, TTPrint achieves 76.48% and 87.39% macro-F1 respectively, outperforming the leading baseline by 63.5% and 29.4%. A multi-backbone analysis across six LLMs and a threshold sensitivity study further demonstrate generalizability across model choices and provide practical guidance for parameter selection.
>
---
#### [new 168] MuCRASP: Multimodal Chain-of-thought Reasoning aware Structured Pruning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出MuCRASP，解决视觉语言模型结构化剪枝中保持推理准确性的难题，通过关注推理关键组件提升压缩效果。**

- **链接: [https://arxiv.org/pdf/2605.25842](https://arxiv.org/pdf/2605.25842)**

> **作者:** Aritra Dutta; Somak Aditya
>
> **备注:** First Preprint
>
> **摘要:** Vision-language models (VLMs) increasingly rely on chain-of-thought (CoT) reasoning to solve complex multimodal tasks, but their large parameter sizes make deployment expensive. Structured pruning offers a natural solution; however, existing methods fail to preserve CoT reasoning accuracy in VLMs. We identify two key reasons: (1) CoT consistency depends on sparse transition points (pivot tokens) in the generation trajectory, while existing pruning methods are CoT-agnostic; and (2) pruning methods designed for unimodal LLMs do not account for activation-distribution differences across visual and textual modalities. Motivated by these observations, we propose MuCRASP, a structured pruning framework that targets reasoning-critical components while preserving cross-modal alignment and accounting for layer-wise sensitivity under a global parameter budget. Experiments on four VLMs across three reasoning benchmarks show that MuCRASP consistently preserves reasoning quality under increasing compression. At 30% pruning on Qwen2.5-VL-7B, MuCRASP achieves an LLM-as-a-Judge score of 8.87 versus 7.32 for the strongest baseline on physical reasoning tasks. Furthermore, MuCRASP maintains high reasoning consistency up to 50% pruning, significantly outperforming prior pruning approaches while exhibiting lower perplexity degradation.
>
---
#### [new 169] LC-ERD: Mining Latent Logic for Self-Evolving Reasoning via Consistency-Regulated Reward Decomposition
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大语言模型推理优化任务，解决过程数据稀缺与监督信号不足问题。提出LC-ERD框架，通过逻辑一致性约束提升自进化能力。**

- **链接: [https://arxiv.org/pdf/2605.24005](https://arxiv.org/pdf/2605.24005)**

> **作者:** Yanyu Chen; Jiyue Jiang; Dianzhi Yu; Zheng Wu; Jiahong Liu; Jiaming Han; Xiao Guo; Jinhu Qi; Yu Li; Yifei Zhang; Irwin King
>
> **备注:** Accepted in SIGKDD2026 Research Track
>
> **摘要:** The evolution of Large Language Model (LLM) reasoning is bottlenecked by the scarcity of high-quality process data. While self-alignment via endogenous rewards offers a solution, mining valid supervision faces three challenges: (1) Label Noise via Mimetic Bias, where rewards prioritize statistical likelihood over logical truth, creating a "correctness illusion" that masks compounding errors; (2) Coarse-Grained Supervision, where sparse global outcomes (e.g., in GRPO) fail to provide granular guidance, treating reasoning chains as monolithic; and (3) Distributional Collapse, where signals fail to generalize without amplifying pre-training biases. To address these, we introduce LC-ERD (Logic-Consistent Endogenous Reward Decomposition), a framework framing self-alignment as latent structure mining. We derive a Variational Logic Potential by aggregating consensus from the model's Latent Logic Expertise (LLE) to denoise the reasoning manifold, and introduce a Multi-Agent Value Decomposition protocol based on the IGM principle to quantify individual step utility. Experiments show LC-ERD delivers a robust self-evolution path, uncovering trade-offs between logic consistency and accuracy while identifying high-value reasoning patterns missed by standard rewards. Our code is available at this https URL.
>
---
#### [new 170] AI Content Moderation in Therapy Conversations
- **分类: cs.HC; cs.AI; cs.CL; cs.CY; cs.SI**

- **简介: 论文研究AI在治疗对话中的内容审核问题，分析现有系统对真实治疗内容的限制，探讨其作为治疗工具的局限性。任务属于AI伦理与内容审核领域。**

- **链接: [https://arxiv.org/pdf/2605.25454](https://arxiv.org/pdf/2605.25454)**

> **作者:** Jiwon Kim; Claire Wang; Taeung Yoon; Sabelle Huang; Koustuv Saha
>
> **摘要:** Large language models (LLMs) are increasingly being used for emotional support. They are also being developed for formal therapy purposes. However, LLMs like ChaptGPT or Llama are often developed with content moderation guardrails that prevent them from discussing sensitive subjects with users for both liability and safety purposes, and this inability to broach these subjects may affect their capacity as therapists. In this study, we perform an algorithm audit on three state-of-the-art moderation systems (OpenAI's moderation endpoint, Meta's Llama Guard, and Google's Shield Gemma) to investigate the extent to which these systems flag the content of real-life therapy sessions as undesirable. Our results raise implications for the limitations that users and organizations may encounter when designing LLMs to play the part of a therapist.
>
---
#### [new 171] Trust but Verify: Prover-Verifier Deliberation for Selective LLM Prediction
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出PVD机制，用于语言模型的可信预测。任务是提高模型预测的可靠性，通过验证过程区分可靠与不可靠答案。**

- **链接: [https://arxiv.org/pdf/2605.25133](https://arxiv.org/pdf/2605.25133)**

> **作者:** João Sedoc; Baotong Zhang; Dean Foster
>
> **摘要:** Reliably knowing when a language model is correct is almost as important as being correct. We introduce prover-verifier deliberation (PVD), an inference-time protocol grounded in interactive proof theory, as a mechanism for selective prediction: the protocol produces both an answer and a structured confidence verdict, allowing a system to report high-confidence answers while abstaining on uncertain cases. In each dialogue, a prover defends a candidate answer through checkable sub-claims while a verifier issues targeted challenges and returns \textsc{Accept}, \textsc{Challenge}, or \textsc{Reject}. Because frozen language models are imperfect provers and verifiers operating over a noisy channel, formal soundness and completeness guarantees do not transfer; instead, we characterize the protocol empirically through its coverage-precision behavior. Our main experiment uses Claude Sonnet 4.6 as prover and Claude Haiku 4.5 as verifier on GPQA Diamond. Questions accepted with no answer revision, which we call Accept + No Change (ANC), are reported as the high-confidence subset; we evaluate this subset by its precision and coverage. ANC separates reliable from unreliable answers, yielding a $\sim$30pp HC-Prec gap over the non-ANC complement. Robustness experiments with GPT and Gemini pairings show that high HC-Prec can transfer across model families, while verifier strictness and domain competence largely determine the size of the selection gap. On Humanity's Last Exam, weaker prover-verifier pairings can collapse or invert the ANC signal, illustrating a practical failure mode when the verifier operates outside its effective region. Comparisons with self-consistency, universal self-consistency, multi-agent debate, and Reflexion suggest that prover-verifier deliberation supplies a distinct argument-defensibility signal for selective prediction.
>
---
#### [new 172] MindAlign: Bridging EEG, Vision, and Language for Zero-Shot Visual Decoding
- **分类: cs.LG; cs.CL; q-bio.NC**

- **简介: 该论文属于视觉解码任务，旨在从脑电（EEG）信号中解码视觉信息。通过三模态对比框架，对齐EEG、图像和文本表示，提升零样本准确率。**

- **链接: [https://arxiv.org/pdf/2605.24523](https://arxiv.org/pdf/2605.24523)**

> **作者:** Zexuan Chen; Sichao Liu; Runhao Lu; Huichao Qi; Alexandra Woolgar; Xi Vincent Wang; Lihui Wang
>
> **备注:** 20 pages, 10 figures, 15 tables
>
> **摘要:** Visual decoding from brain signals is a key challenge at the intersection of computer vision and neuroscience, requiring methods that bridge neural representations and computational models of vision. We introduce a tri-modal contrastive framework for EEG-based visual decoding that aligns EEG, visual, and textual representations within a unified latent space. Our approach follows a two-stage design. First, we pre-train an EEG encoder via masked reconstruction on unlabeled trials, learning spatio-temporal regularities that transfer robustly to downstream tasks. Second, we jointly align EEG, image, and LLM-generated textual descriptions through contrastive learning, where text supervision acts as a semantic regularizer that injects linguistic structure into the shared space without overwhelming the primary EEG-image signal. The encoder integrates subject-specific adaptation, graph-attention over channels, and temporal-spatial convolutional embeddings. On the Things-EEG2 200-way zero-shot benchmark, our framework achieves 54.1% Top-1 and 83.4% Top-5 accuracy, substantially exceeding the strongest prior baseline (32.4% / 64.0%), with paired Wilcoxon tests confirming significance (p < 0.01) over all in-subject baselines. We validate generalization on Things-MEG. Analysis reveals that compact embedding geometries (CN-CLIP) outperform much larger backbones, and that decoding aligns with established neurophysiology of visual processing. This work is a critical step towards robust, semantically-grounded visual decoding from non-invasive temporal neural signals. The source code is publicly available in this https URL.
>
---
#### [new 173] Faithfulness as Information Flow: Evaluating and Training Faithful Chain-of-Thought Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究如何提升链式推理（CoT）的可信度，解决模型依赖捷径导致推理不真实的问题。通过信息流分析，提出评估与训练方法，增强CoT的中介作用。**

- **链接: [https://arxiv.org/pdf/2605.24286](https://arxiv.org/pdf/2605.24286)**

> **作者:** Jinghan Jia; Joe Benton; Eric Easley
>
> **摘要:** Chain-of-thought (CoT) reasoning is useful for monitoring language models only when the reasoning trace faithfully reflects the computation that produces the final answer. However, models can rely on prompt-to-answer shortcuts that bypass the CoT, making the visible reasoning trace misleading even when it appears plausible. We study CoT faithfulness through a structural information-flow perspective: faithful reasoning should route answer-relevant information through the mediated path from prompt to CoT to answer, rather than through a direct prompt-to-answer shortcut. This perspective yields a task-agnostic framework based on three complementary properties, sufficiency, completeness, and necessity, which we instantiate with entropy-based, masked-KL, and gradient-based diagnostics. We show that these metrics recover externally judged faithfulness differences in hinted reasoning, and identify a low-entropy failure mode of KL-based diagnostics where gradient-based measures remain more stable. Building on this analysis, we introduce update-time interventions for verifier-based on-policy RL, including attention masking, backward-only gradient masking, CoT gradients, and adversarial perturbations of prompt representations. Across hinted arithmetic, reward-hackable code repair, and DAPO-Math models trained without hints but evaluated under wrong-hint injection, our interventions shift behavioral and structural indicators toward stronger CoT mediation. In particular, they make shortcut and reward-hacking behavior more transparent in the CoT and improve task-agnostic faithfulness metrics, while in some settings also reducing wrong-hint susceptibility. Our results suggest that controlling information flow during training is a practical route toward more faithful and monitorable CoT reasoning. Code is available at this https URL.
>
---
#### [new 174] Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning
- **分类: cs.AI; cs.CL; cs.NI**

- **简介: 该论文属于图学习任务，旨在解决CoT方法在图结构数据上的可解释性与交互问题。提出KCoT框架，将CoT与图表示学习结合，通过k-means视角提升推理能力。**

- **链接: [https://arxiv.org/pdf/2605.24867](https://arxiv.org/pdf/2605.24867)**

> **作者:** Xuanting Xie; Zhaochen Guo; Bingheng Li; Xingtong Yu; Zhifei Liao; Zhao Kang; Yuan Fang
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Chain-of-Thought (CoT) prompting has shown promise in enhancing the reasoning capabilities of large language models (LLMs) on text-attributed graphs (TAGs). This work reframes CoT-based graph learning through the principle of clustering as reasoning, offering a $k$-means interpretation of how iterative reasoning operates over graph-structured data. We observe that existing graph CoT methods rely on disjoint architectures and fixed graph representations, limiting step-by-step semantic-topological interaction and interpretability. To overcome this limitation, we propose a unified framework named KCoT that integrates CoT reasoning with graph representation learning. Our key theoretical result reveals a formal mathematical correspondence between a Transformer block and the $k$-means algorithm, allowing reasoning to be interpreted as iterative assignment and update steps. Based on this insight, we introduce a Semantic Discriminating Prompt that explicitly formulates these steps as structured CoT reasoning, together with a structure-grounded alignment strategy to fuse topological priors with evolving thought-conditioned representations. Experiments on standard benchmarks demonstrate consistent improvements over state-of-the-art methods, validating clustering as a principled mechanism for CoT-based graph learning.
>
---
#### [new 175] Polymorphism Is Rotation: Operational Mechanistic Interpretability from a Two-Layer Transformer to Pythia-70m
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究独立训练的Transformer模型在残差流中通过随机旋转实现功能相同但内部坐标不同，提出通过正交Procrustes方法消除这种现象，验证了其有效性。任务为模型可解释性，解决功能同构但坐标不一致问题。**

- **链接: [https://arxiv.org/pdf/2605.24577](https://arxiv.org/pdf/2605.24577)**

> **作者:** Jordan F. McCann
>
> **备注:** 26 pages, 4 figures, 40 references. Pre-registered four-bar framework; all numerical claims reproducible
>
> **摘要:** Independently trained transformers compute the same function in residual-stream bases that differ by a uniform random rotation on $\mathrm{SO}(d_{\mathrm{model}})$. We call this phenomenon polymorphism: same function, mutually unintelligible interior coordinates. One matrix multiplication per model pair removes it: an orthogonal Procrustes fit on a single batch of activations transfers sparse-autoencoder feature dictionaries and steering vectors between independently trained models, with no retraining. The phenomenon is invisible to the standard SAE universality metric. Decoder-column cosine similarity matches across seeds at 98%, the SAE-universality headline number, while an SAE trained on one seed reconstructs another seed's activations at negative explained variance, worse than predicting the constant mean. The decoder columns align; the encoder reads from a rotated frame. A single Procrustes rotation $R$ restores reconstruction to within 0.025 EV of the within-seed ceiling at every internal site. $R$ is Haar-distributed: $\|R - I\|_F$ matches the random-orthogonal prediction $\sqrt{2 d_{\mathrm{model}}}$ to 0.1% at $d_{\mathrm{model}} = 512$, and a Kolmogorov-Smirnov test of $R$'s eigenvalue spectrum against Haar $\mathrm{SO}(d_{\mathrm{model}})$ returns $p \approx 1.000$ pooled and per-pair. Diff-of-means steering vectors transfer in three regimes by alignment with $R$'s invariant subspace: clean when pinned by shared output weights, partial when overlapping the rotated subspace, inverted otherwise. With no shared I/O (Pythia), all three collapse to universally inverted. The same rotation account holds across training checkpoints within a single run. Validated on a 104k-parameter Dyck-3 transformer and nine independently-trained Pythia-70m seeds on The Pile, via a pre-registered four-bar operational framework. Frontier-scale (10B+) replication remains open.
>
---
#### [new 176] Universal Boosts, Specific Suppressors: Sparse Autoencoder Steering of Medical Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医学视觉-语言模型任务，解决生成报告中的幻觉问题。通过解码时的稀疏自编码器引导，提升报告质量并减少错误。**

- **链接: [https://arxiv.org/pdf/2605.24977](https://arxiv.org/pdf/2605.24977)**

> **作者:** Farhad Nooralahzadeh; Benjamin Gundersen; Nicolas Deperrois; Hidetoshi Matsuom; Mizuho Nishio; Thomas Frauenfelder; Ahmed Allam; Christian Blüthgen; Michael Moor; Michael Krauthammer
>
> **摘要:** Medical vision-language models (VLMs) often hallucinate findings when generating chest X-ray reports: they fabricate findings that are not present in the image, miss important ones, or locate them incorrectly. We mitigate this without weight updates by decoding-time residual steering on a per-token sparse autoencoder (SAE) basis: Top-$K$ SAEs on late layers, causal steering against clinical errors, then combined suppress/boost intervention at inference time. On the MIMIC-CXR test split, our inference-only method improves the quality of generated reports for three radiology VLMs (RadVLM, LLaVA-Rad, and CheXOne), with relative improvements of +5.4%, +7.2%, and +17.0% in the clinical composite metric, and statistically significant GREEN gains on all backbones. A cross-model feature alignment shows that the quality-promoting (boost) directions overlap strongly across architectures, whereas hallucination-linked (suppress) directions are model-specific. Therefore, transferable steering must treat suppression per-backbone, rather than sharing a universal suppress list. The same recipe transfers zero-shot to IU-Xray (Green $+7.7\%$ rel.) without retraining, confirming that the identified features are properties of the model, not of the training corpus. We release causal feature sets and an interactive feature dashboard: this https URL.
>
---
#### [new 177] DUEL: Adversarial Self-Play for Multimodal Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出DUEL框架，用于视觉语言模型的后训练，解决无监督下视觉推理与判别能力不足的问题。通过对抗自博弈提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.24794](https://arxiv.org/pdf/2605.24794)**

> **作者:** Lin Qiu; Hanqing Zeng; Yao Liu; Bingjun Sun; Guangdeng Liao; Ji Liu
>
> **摘要:** Reinforcement learning (RL) has emerged as an effective paradigm for improving the reasoning capability of vision-language models (VLMs). However, RL-based optimization typically depends on costly high-quality annotations that are difficult to scale. Existing unsupervised alternatives may drift toward biased solutions due to weak visual grounding and the lack of reliable verification signals. We propose a self-evolving post-training framework, DUEL, where supervision emerges from adversarial interactions between two policies initialized from the same pretrained VLM. A Challenger generates an image-grounded true claim together with a minimally perturbed hard-negative counterpart, while a Solver verifies both claims against the image, encouraging fine-grained visual discrimination under near-neighbor semantics. To stabilize optimization, we introduce a length-normalized log-likelihood reward that preserves informative optimization signals beyond binary outcome supervision and improves learning stability under sparse feedback. Experiments show that DUEL consistently improves visual reasoning and robust discrimination without additional human annotations, external reward models, or image editing tools.
>
---
#### [new 178] Exploration of Perceptual Speech Features for Clinical Decision-Support in Mental Health Care
- **分类: cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于心理健康评估任务，旨在通过语音特征分析支持临床决策。研究提取语音的声学和语言特征，结合机器学习方法，探索其与抑郁、焦虑等症状的关系。**

- **链接: [https://arxiv.org/pdf/2605.24678](https://arxiv.org/pdf/2605.24678)**

> **作者:** Vassilis Lyberatos; Edmund G. Dervakos; Eleni Adamidi; Athanasios Voulodimos; Giorgos Stamou
>
> **备注:** Accepted to CLPsych 2026, part of ACL 2026
>
> **摘要:** Speech and language technologies offer valuable opportunities for supporting mental health assessment through objective and interpretable cues. We present a systematic feature-based analysis framework leveraging perceptually grounded acoustic and linguistic characteristics, including prosody, vocal quality, semantic coherence, syntactic structure, and sarcasm. Using statistical analysis and interpretable machine learning (XGBoost with SHAP and LIME), we examine associations between speech features and validated symptom measures of depression, anxiety, and ADHD. Evaluated on both controlled benchmark datasets (StressID, DAIC-WOZ, Androids, EATD) and a real-world clinical dataset, the framework reveals stable and consistent relationships between symptom severity and vocal irregularities (e.g., shimmer, jitter), lexical-syntactic patterns, and affective tone. An ablation study conducted across all datasets further identifies the most informative feature groups. This work explores a transparent and clinically interpretable approach to speech-based mental health analysis.
>
---
#### [new 179] TRACER: A Semantic-Aware Framework for Fine-Grained Contamination Detection in Code LLMs
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于代码大模型数据污染检测任务，解决代码模型中非精确复制的污染问题。提出TRACER框架，通过语义重叠检测污染，取得高准确率。**

- **链接: [https://arxiv.org/pdf/2605.24079](https://arxiv.org/pdf/2605.24079)**

> **作者:** Yifeng Di; Xuliang Huang; Tianyi Zhang
>
> **备注:** 21 pages, 2 figures, 15 tables
>
> **摘要:** Data contamination is a known threat to the reliability of model evaluation. However, it remains underexplored in code large language models (LLMs), where contamination often goes beyond exact duplication. We present TRACER, a semantic-aware framework for fine-grained code contamination detection. TRACER models contamination using three levels of semantic overlap - Functionally Identical, Nearly Identical, and Shared Logic - and detects them through a coarse-to-fine pipeline. We also introduce the first benchmark for fine-grained code contamination detection, spanning three widely used benchmarks and three representative post-training datasets. TRACER achieves strong and consistent performance across multiple LLM backbones, with GPT-5 reaching an F1 score of 0.91 in fine-grained detection. In the binary setting, TRACER attains an F1 of 0.92, outperforming existing methods by 42%-217%. We further conduct ablation studies and error analysis to assess the contributions of individual components in TRACER.
>
---
#### [new 180] CausaLab: A Scalable Environment for Interactive Causal Discovery Toward AI Scientists
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出CausaLab，用于评估大模型在因果发现中的表现。任务是解决因果机制理解问题，通过设计实验环境测试模型的因果推理能力。工作包括构建合成实验室、设计交互策略及验证模型的因果推断效果。**

- **链接: [https://arxiv.org/pdf/2605.26029](https://arxiv.org/pdf/2605.26029)**

> **作者:** Junlin Yang; Dylan Zhang; Xiangchen Song; Qirun Dai; Xiao Liu; Yuen Chen; Aniket Vashishtha; Jing Shi; Chenhao Tan; Hao Peng
>
> **摘要:** We introduce CausaLab, a scalable environment for evaluating interactive causal discovery by LLM agents. Unlike prior evaluations, CausaLab evaluates both whether an agent can solve a problem using causal evidence and whether its answer is supported by a correct hypothesis about the underlying causal mechanism. Each episode places an agent in a synthetic laboratory: it receives prior measurement records, intervenes on a manipulator crystal, and predicts the resonance frequency of a held-out reactor crystal governed by the same mechanism. The hidden data-generating process is a randomly sampled structural causal model (SCM), so success requires recovering both a causal graph and structural equations rather than recalling prior knowledge. CausaLab also includes a domain-specific language that records the agent's evolving SCM hypothesis, making trajectories inspectable and comparable with ground truth. Experiments show a persistent gap between prediction and mechanism recovery: in the purely observational 6-node setting, GPT-5.2-high reaches 92% task accuracy but only 0.471 all-edge $F_1$. This observation further motivates our exploration of different interaction strategies: Mixed observation--intervention strategies improve structural fidelity: in the mixed 6-node setting, GPT-5.2-high achieves 80% on both task accuracy and all-edge $F_1$. Yet even strong agents struggle to design informative interventions, as pure intervention strategies perform poorly on both task accuracy and all-edge $F_1$. We identify premature stopping as a major weakness of agents, and show that asking the model to verify the consistency between its hypothesis and past data can help mitigate this issue. CausaLab therefore separates predictive success from causal understanding and exposes current LLM agents' limits as experimental causal reasoners.
>
---
#### [new 181] Adaptive Preference Optimization with Uncertainty-aware Utility Anchor
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大语言模型对齐任务，解决传统偏好优化方法依赖配对数据的问题。提出UAPO框架，引入锚定函数处理不确定性，提升数据利用效率和训练鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.10515](https://arxiv.org/pdf/2509.10515)**

> **作者:** Xiaobo Wang; Zixia Jia; Jiaqi Li; Qi Liu; Zilong Zheng
>
> **备注:** Accepted by EMNLP 2025 Findings
>
> **摘要:** Offline preference optimization methods are efficient for large language models (LLMs) alignment. Direct Preference optimization (DPO)-like learning, one of the most popular approaches, stands out for its efficiency in reward modeling. However, these methods typically follow the convention to use Bradley-Terry (BT) reward modeling that faces several critical assumptions, including the requirement for pairwise training data, model distribution shifting, human rationality assumption, etc. To address these limitations, we propose a general framework for offline preference optimization methods, Adaptive Preference Optimization with Utility Anchor (UAPO), which introduces an anchoring function to estimate the uncertainties brought from preference data annotation. Our method enables training even in scenarios where the data is unpaired, significantly enhancing data utilization efficiency. Moreover, the anchor design makes UAPO more robust in the training process. Experimental results demonstrate that UAPO achieves competitive outcomes without the strict dependency on data pairing, paving the way for more flexible and effective preference optimization methods.
>
---
#### [new 182] Why We Need World Models for AGI: Where LLMs Fail and How World Models May Outperform
- **分类: cs.AI; cs.CL; cs.RO**

- **简介: 该论文探讨AGI发展中世界模型的必要性，针对LLMs在因果推理和长期规划上的不足，提出Latent Dynamics Inference概念，并通过Flux环境验证世界模型的优势。**

- **链接: [https://arxiv.org/pdf/2605.23972](https://arxiv.org/pdf/2605.23972)**

> **作者:** Feisal Alaswad; Batoul Aljaddouh; Maher Alrahhal; Poovammal E; Talal Bonny
>
> **备注:** 19 pages, 5 figures
>
> **摘要:** Large language models achieve strong performance in language generation and knowledge-intensive tasks, yet remain limited in settings requiring causal reasoning, persistent state tracking, and long-horizon planning. We argue that these limitations may arise from an objective-level mismatch between sequence prediction and reasoning over latent environment dynamics. To formalize this distinction, we introduce Latent Dynamics Inference (LDI), a conceptual perspective that interprets language and multimodal observations as partial evidence of underlying transition dynamics. To empirically investigate this perspective, we introduce Flux, a sequential reasoning environment specified entirely through natural-language rules. As a proof-of-concept case study, the rules are first compiled into an explicit state-transition simulator, illustrating that structured latent transition dynamics can, in some cases, be operationally extracted from textual rule descriptions. This enables a controlled comparison between the LLMs operating purely over textual observations and reinforcement-learning agents trained directly within the extracted latent state space. Within this case study, agents operating with explicit access to the latent state space exhibit substantially more stable behavior in long-horizon gameplay, achieving an aggregate win rate of approximately 79% versus 11% for LLMs. Qualitative analysis further reveals failure modes consistent with unstable persistent state tracking, including invalid actions, state-tracking errors, and short-horizon reasoning failures. The complete implementation of the Flux environment available at this https URL Within the evaluated setting, these results suggest that strong sequence prediction alone may struggle to support robust long-horizon dynamic reasoning without mechanisms for persistent state tracking and transition modeling
>
---
#### [new 183] Towards trustworthy agentic AI: a comprehensive survey of safety, robustness, privacy, and system security
- **分类: cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于安全与可信AI研究任务，旨在解决agentic AI系统的安全性、鲁棒性、隐私和系统安全问题，通过分析风险点并提出缓解策略。**

- **链接: [https://arxiv.org/pdf/2605.23989](https://arxiv.org/pdf/2605.23989)**

> **作者:** Jinhu Qi; Muzhi Li; Jiahong Liu; Yuqin Shu; Dianzhi Yu; Shicheng Ma; Wenqian Cui; Yiyang Zhao; Yiyi Chen; Ruoxi Jiang; Irwin King; Zenglin Xu
>
> **备注:** 36 pages, 4 figures. Survey/review article on trustworthy agentic AI. Published in Academia AI and Applications, 2026
>
> **摘要:** Agentic AI systems -- Large Language Models (LLMs) augmented with planning, tool use, memory, and long-horizon interactions -- can execute complex tasks autonomously, but their multi-step trajectories introduce new failure modes that challenge trustworthiness. This survey provides a focused examination of trustworthy agentic AI through two core dimensions that are critical for high-risk deployments: Safety and Robustness, and Privacy and System Security. For each dimension, we clarify key concepts, identify where risks emerge along the agent workflow, and summarize stage-targeted mitigation strategies. Other trustworthiness aspects (value alignment, transparency, fairness, and accountability) are discussed as relevant context rather than parallel chapters. To support consistent comparison and deployment decisions, we consolidate evaluation into a unified metrics-and-benchmarks hub, emphasizing both outcome and process signals (e.g., constraint violations, trace completeness, and adversarial success rates) and offering scenario-to-metric guidance for release gating. We conclude by outlining open challenges such as self-evolving agents, runtime monitoring and verification, privacy-preserving personalization, and the trust-utility trade-off, and present a case study of real-world security failures in open-source agentic systems. Our goal is to serve as a practical reference for researchers and practitioners building trustworthy agentic systems in high-stakes environments.
>
---
#### [new 184] Can LoRA Fusion Support Cross-Domain Tasks in Cloud-Edge Collaboration?
- **分类: cs.DC; cs.CL**

- **简介: 该论文研究云边协同中的跨领域任务，解决隐私约束下LoRA适配器融合问题。提出Prune-Train-Recover框架和MMLU-CD基准，发现现有方法效果不佳，并引入LoRA-CR改进性能。**

- **链接: [https://arxiv.org/pdf/2605.23913](https://arxiv.org/pdf/2605.23913)**

> **作者:** Yatong Wang; Fali Wang; Naibin Gu; Zheng Lin; Zhengxiao Liu; Dingyu Yao; Zhiwei Zhang; Jianxin Shi; Weiping Wang
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Cloud-hosted large language models (LLMs) commonly rely on LoRA for domain adaptation, yet domain data are distributed across multiple edge devices and cannot be uploaded due to privacy constraints. This raises a fundamental question: how can knowledge from multiple private edges be integrated into a cloud LLM for cross-domain problem solving? A natural solution is to train LoRA adapters locally and fuse them in the cloud; however, existing pipelines rely on unrealistic assumptions that edge devices can host cloud-scale LLMs and are evaluated mainly on single-domain tasks. To address these limitations, we propose a prune-train-recover framework that enables local LoRA training on pruned models and privacy-preserving cloud integration. We further introduce MMLU-CD, a cross-domain benchmark that composes multiple domain samples into a single instance, enabling explicit evaluation of cross-domain problem solving. This allows us to ask a concrete question: Can existing LoRA fusion methods support cross-domain tasks in cloud-edge collaboration? Our empirical answer is negative. Existing LoRA fusion methods perform poorly on MMLU-CD, often underperforming the base LLM, revealing their inability to support cross-domain problem solving. We attribute this failure to parameter conflicts among LoRA adapters and propose a simple conflict-resolution module, LoRA-CR, which mitigates conflicting updates and improves LoRA fusion performance by up to 3.8%. These results identify conflict mitigation as a critical yet largely overlooked factor in cloud-edge LoRA fusion, warranting further investigation in future research.
>
---
#### [new 185] Spectral Retrieval: Multi-Scale Sinc Convolution over Token Embeddings for Localized Retrieval in LLM Multi-Agent Systems
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文提出Spectral Retrieval方法，用于解决大语言模型多智能体系统中的局部检索问题。通过多尺度sinc卷积优化检索效果。**

- **链接: [https://arxiv.org/pdf/2605.24764](https://arxiv.org/pdf/2605.24764)**

> **作者:** Andrea Morandi
>
> **摘要:** [Abridged] - Spectral Retrieval is a plug-in re-ranking stage that interpolates between per-token MaxSim and mean-pool retrieval through a multi-scale sinc convolution over token embeddings. In standard dense retrieval each document is one mean-pooled vector; when relevance localises into a short subspan, the signal averages into noise. Spectral Retrieval reuses per-token embeddings from a late-interaction index and convolves them with a normalised sinc kernel at multiple scales. At L=1 the kernel acts as the identity, recovering per-token MaxSim; as L grows it approaches a uniform filter, recovering mean pooling. The maximum cosine over positions and scales yields a score provably no less informative than either endpoint. On a controlled synthetic benchmark with 1,000 documents and planted single-position spikes, mean-pool retrieval sits at chance (Recall@10 ~ 0.02) regardless of spike strength, while Spectral Retrieval reaches Recall@10 = 1.0 once the planted cosine exceeds the corpus-level token noise floor. On LIMIT-small with a frozen all-mpnet-base-v2 encoder, Spectral Retrieval lifts Recall@10 from 0.33 to 0.90, MRR from 0.22 to 0.79, and strict Success@10 from 0.12 to 0.84, without retraining. The method fits naturally into multi-agent LLM systems, where each agent benefits from a tighter, role-specific retrieval window over a shared corpus.
>
---
#### [new 186] Context: Proactive Goal-Directed Intelligence via Composable Sandboxed Programs, Declarative Wiring, and Structured Interaction
- **分类: cs.AI; cs.CL; cs.DC; cs.MA; cs.PL; cs.SE**

- **简介: 该论文提出Context系统，解决传统聊天机器人被动响应的问题，通过主动目标导向代理推进任务，实现高效、结构化交互。**

- **链接: [https://arxiv.org/pdf/2605.23928](https://arxiv.org/pdf/2605.23928)**

> **作者:** Gregory Magarshak
>
> **备注:** 7 pages; third in a series with arXiv:this http URL (Magarshak Machine / SPACER) and arXiv:this http URL (Grokers)
>
> **摘要:** We present Context, the intelligence layer of the Magarshak Architecture, which replaces reactive query-response chatbots with proactive goal-directed agents that advance shared tasks without waiting for user prompts. The architecture rests on three mutually reinforcing mechanisms. Write-time context assembly precomputes enriched typed attributes via Groker agents, assembling interaction context as a deterministic pure function of graph state; context blocks are byte-identical across turns between semantic changes, enabling near-100% KV-cache reuse. Composable sandboxed wisdom programs form a governed library of LM-generated imperative programs declaratively wired to goal types via typed stream relations, composed via phase ordering, and executed at interaction time without further LM calls. Proactive goal stream state machines drive conversations toward terminal states by inspecting graph state and emitting structured interaction content (option arrays, governance affordances, clarification prompts) without awaiting user input. We prove six formal results: the Context Stability Theorem, bounding per-turn LM cost as a function of semantic change rate; a Program Composition Correctness Theorem; a Declarative Wiring Soundness Theorem; the Proactive Dominance Theorem, proving proactive agents weakly dominate reactive agents on expected turns-to-terminal-state; Coordination Overhead Elimination and Quality Preservation, establishing Pareto improvements in multi-participant goal chats; and a Cross-Platform Vote Consistency Theorem. Implemented in the open-source Qbix / Safebox / Safebots stack.
>
---
#### [new 187] Breaking the Chains of Probability: Neutrosophic Logic as a New Framework for Epistemic Uncertainty in Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文探讨将中性逻辑应用于大语言模型，解决其在处理认知不确定性时的局限性。通过实验验证中性逻辑在描述模糊、矛盾等状态中的优势，旨在提升AI系统的透明度与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.24053](https://arxiv.org/pdf/2605.24053)**

> **作者:** Maikel Yelandi Leyva-Vázquez; Florentin Smarandache
>
> **备注:** Published in Neutrosophic Sets and Systems, Vol. 99 (2026). Author's preprint version. Open code and data available at: this http URL
>
> **摘要:** Large Language Models (LLMs) are predominantly governed by probabilistic frameworks in which the sum of outcome probabilities is constrained to unity. This architectural limitation, often imposed by Softmax layers, leads to a collapse of uncertainty that makes it difficult to differentiate between epistemic uncertainty, paradox, and vagueness. We present an empirical investigation of the application of Neutrosophic Logic, a framework that treats Truth (T), Indeterminacy (I), and Falsity (F) as three independent dimensions, to model epistemic states in LLMs. We conducted experiments on a family of four OpenAI GPT models across five linguistic phenomena: logical paradoxes, epistemic ignorance, vagueness, ethical contradictions, and future contingencies, under three prompting strategies: neutrosophic, probabilistic, and entropy-derived. Our findings reveal that the neutrosophic approach, by allowing T+I+F > 1, a state we term hyper-truth, provides a richer representation of a model's internal state. In 35% of evaluations, hyper-truth emerged spontaneously, predominantly under ethical contradiction and logical paradox. We demonstrate that this approach preserves truth values in fuzzy contexts and offers a robust method for identifying and quantifying internal model conflict. We conclude that the integration of neutrosophic evaluation layers is a critical step toward more transparent, reliable, and ethically aware AI systems.
>
---
#### [new 188] RotMoLE: Enhancing Mixture of Low-Rank Experts through Rotational Gating Mechanism
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM在复杂场景下的适应性问题。通过改进MoE架构，提出RotMoLE模型，增强低秩专家的表示能力与泛化性能。**

- **链接: [https://arxiv.org/pdf/2605.25565](https://arxiv.org/pdf/2605.25565)**

> **作者:** Mengyang Sun; Maochuan Dou; Tao Feng; Dan Zhang; Yihao Wang; Junpeng Liu; Yifan Zhu; Jie Tang
>
> **摘要:** While Large Language Models (LLMs) are commonly fine-tuned to handle domain-specific tasks before being applied to vertical applications, adapting them to complex scenarios with diverse specialized knowledge remains challenging. Meanwhile, Mixture-of-Experts (MoE) architecture has risen as a crucial paradigm for training LLMs, and some recent works have also incorporated MoE into Parameter-Efficient Fine-Tuning (PEFT) to propose the Mixture of Low-rank Experts (MoE-LoRA), to enhance the power of low-rank adapters for learning complicated knowledge. However, conventional gating mechanisms in MoE typically apply only a scalar reweighing to selected experts, thereby limiting their underlying capacity of representation and generalization. Motivated and enabled by the low-rank structures in MoE-LoRA, we propose RotMoLE, a specialized MoE framework for low-rank experts featuring an additional rotation gate. Beyond simple scaling, RotMoLE implements a rotation mechanism for each selected expert, enabling superior expert exploitation and specialization for learning diverse data, especially when expert candidates are limited. Empirical results on complex multi-task and multilingual training scenarios validate our effectiveness.
>
---
#### [new 189] Mapping the Schedule x Bit-Width Boundary in Sub-100M Quantisation-Aware Training
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文研究量化感知训练中学习率调度与位宽的关系，旨在确定最优调度策略。通过大量实验验证不同位宽下的最佳调度方案，发现50M以上使用wd33更优，低于此则调度影响不显著。**

- **链接: [https://arxiv.org/pdf/2605.25966](https://arxiv.org/pdf/2605.25966)**

> **作者:** Christian Brandt Thomassen
>
> **备注:** 20 pages, 6 figures, 4 tables. 1345 training runs total (720 + 625). Submitted for review at TMLR
>
> **摘要:** We test whether the optimal learning-rate schedule depends on bit-width during from-initialisation quantisation-aware training (QAT) for sub-100M decoder language models. A 720-run factorial grid (Phase 2) over bit-width x warmdown fraction x LR magnitude x model size x seed (FP16/INT8/INT6, 15M-100M, 5 seeds) finds the optimal warmdown is 33% at every (bit-width, size) cell. The primary hypothesis -- that INT6 QAT requires a different schedule than higher-precision training -- is falsified at FP16/INT8/INT6. A 625-run follow-up (Phase 5) probes the null along five axes: optimiser (AdamW), schedule shape (cosine), training length (up to 9x more iterations), an extended size sweep (5M-350M), and an INT4 sweep from 3M to 100M. The null is robust under all three setup changes. The INT6 penalty follows a log-linear scaling law whose fit on Phase 2 predicts the five held-out Phase 5 sizes (5M, 8M, 175M, 250M, 350M) within their 95% prediction intervals (5/5). For INT4 the picture is sharper than the higher precisions: at 50M and 100M, wd33 is decisively optimal (paired z ~ 12-15, 10/10 seeds); below 50M, across the six tested sizes from 3M to 30M, no individual size shows a statistically significant schedule preference and the per-size mean penalty oscillates within seed-level noise. The boundary is therefore a transition between a noise-dominated regime below 50M and a decisive wd33 regime at and above 50M, not a clean wd10 region. A weight-to-grid-distance probe falsifies the simplest mechanism for the FP16/INT8/INT6 null result (rapid grid-snapping): pre-warmdown, INT6-QAT weights sit at essentially the same distance from the INT6 grid as FP16 weights (ratio ~ 1.04). Practical recommendation: at sub-100M scale, tune the LR schedule once at FP16 and apply unchanged to INT8/INT6 QAT; for INT4 at 50M+ use wd33; for INT4 below 50M the schedule choice is in the noise.
>
---
#### [new 190] SliceWorld: A Predictive and Controllable World-State Model for CT Report Generation
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出SliceWorld，用于CT报告生成任务，解决如何建模CT图像演变和控制病变因素的问题。通过预训练和微调，提升生成质量与临床相关性。**

- **链接: [https://arxiv.org/pdf/2605.24371](https://arxiv.org/pdf/2605.24371)**

> **作者:** Yuanhe Tian; Yan Song
>
> **备注:** 18 pages, 5 figures
>
> **摘要:** CT report generation (CTRG) requires models to summarize three-dimensional anatomical context and pathological findings from hundreds of axial slices. Existing methods typically learn a direct image-to-text mapping, providing limited mechanisms for modeling how CT evidence evolves across slices or how reports respond to controlled changes in latent lesion-related factors. We propose SliceWorld, a CT-specific world-state framework that treats an axial CT scan as an ordered sequence along the z-axis. SliceWorld encodes prefix CT evidence into factor-aware latent states containing anatomy, lesion, and uncertainty components, and projects these states into world tokens used for multi-step future-slice feature prediction, lesion-factor intervention, and LLM-based report generation. The model is first pretrained on CT slice sequences with predictive, factor-aware, and counterfactual objectives, and is then fine-tuned on paired CT-report data. Experiments on M3D-Cap and CT-RATE show that SliceWorld improves natural language generation metrics and clinically oriented automatic evaluation. Further analyses demonstrate multi-horizon future-slice prediction, measurable factor alignment, reduced-slice robustness, and selective lesion-sensitive report modulation.
>
---
#### [new 191] MobileGym: A Verifiable and Highly Parallel Simulation Platform for Mobile GUI Agent Research
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出MobileGym，一个用于移动GUI智能体研究的可验证、高并行的仿真平台，解决真实环境训练成本高和评估不准确的问题。**

- **链接: [https://arxiv.org/pdf/2605.26114](https://arxiv.org/pdf/2605.26114)**

> **作者:** Dingbang Wu; Rui Hao; Haiyang Wang; Shuzhe Wu; Han Xiao; Zhenghong Li; Bojiang Zhou; Zheng Ju; Zichen Liu; Lue Fan; Zhaoxiang Zhang
>
> **备注:** Project page: this https URL
>
> **摘要:** We present MobileGym, a browser-hosted, lightweight, fully controllable environment for everyday mobile use, targeting interaction fidelity without replicating proprietary backends. It enables two capabilities previously out of reach for everyday apps: verifiable outcome signals through deterministic state-based judging over structured JSON state, and scalable online RL through low-cost parallel rollouts. The full environment state is captured, configured, forked, and compared as structured JSON, and a single server can host hundreds of parallel instances, with about 400 MB memory per instance and about 3 s cold start. A layered state model and a declarative task-definition framework keep state programmability and task creation practical at scale, and a single programmatic judging mechanism delivers both deterministic evaluation verdicts and dense RL rewards. The accompanying MobileGym-Bench provides 416 parameterized task templates, including 256 test and 160 train templates, over 28 apps, with deterministic judges and a structured AnswerSheet protocol that avoids free-text matching failures. In a Sim-to-Real case study, GRPO on Qwen3-VL-4B-Instruct gains +12.8 percentage points on the 256-task test set, and on a 59-task real-device signal subset, real-device execution retains 95.1% of the simulation-side training gain. Project page: this https URL.
>
---
#### [new 192] Rubato: Transcribing Piano Music with Timestamps
- **分类: cs.SD; cs.CL; cs.MM**

- **简介: 该论文属于音乐转录任务，解决将音频转换为带时间戳的乐谱问题。提出Rubato模型和InterMo表示，提升转录准确性。**

- **链接: [https://arxiv.org/pdf/2605.24291](https://arxiv.org/pdf/2605.24291)**

> **作者:** Nazif Can Tamer; Victoria Ebert; Guang Yang; Noah A. Smith
>
> **备注:** 18 pages, 7 figures, 5 tables
>
> **摘要:** We consider the conversion of musical recordings into human-readable sheet music annotated with timestamps. Such output lets a listener clearly visualize rubato (temporally expressive playing), a learner diagnose ensemble precision and timing choices against the written music, and a musicology scholar compare performance styles across recordings of the same work. We introduce (1) a prompt-conditioned encoder-decoder model, named Rubato, trained to output (2) a new textual representation for polyphonic music, named InterMo, which we designed for compatibility with sequence-to-sequence training. Our experiments demonstrate that Rubato produces timestamped piano sheet music from audio with higher notational accuracy than the best existing approaches, which are based on cascades. We find that even if the cascade is given ground-truth MIDI instead of audio, Rubato performs better, suggesting that the ceiling of existing approaches is primarily representational, not acoustic. Further, because Rubato is trained on several related tasks (with prompts), it competes with or outperforms the best single-task systems on related but simpler tasks like MIDI note grounding and beat/downbeat detection. A demo is available at this https URL .
>
---
#### [new 193] Geo-Expert: Towards Expert-Level Geological Reasoning via Parameter-Efficient Fine-Tuning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于地质推理任务，旨在解决LLM在地质领域易幻觉的问题。通过参数高效微调，构建了Geo-Expert模型，在专业基准上表现优异。**

- **链接: [https://arxiv.org/pdf/2605.24844](https://arxiv.org/pdf/2605.24844)**

> **作者:** Chenyou Guo; Zongqi Liu; Yizhou Zhang; Zhaorui Jiang; Ze Liu
>
> **备注:** 11 pages, 1 figure, 3 tables. Accepted at ICML 2026 AI for Science Workshop
>
> **摘要:** While general-purpose Large Language Models (LLMs) applied to Geology often hallucinate when reasoning about subsurface structures and deep-time evolution, current AI in Earth sciences predominantly targets surface remote sensing and GIS. To bridge this gap, we introduce Geo-Expert, a family of parameter-efficient geological LLMs fine-tuned on a custom-curated, high-quality instruction dataset processed using our custom instruction synthesis pipeline. We investigate the impact of model scaling and architecture by fine-tuning three base models: Qwen3-8B, Qwen3-32B, and Gemma-3-27B, with Low-Rank Adaptation (LoRA) method. Our extensive evaluation on a novel domain-specific benchmark, Geo-Eval, reveals that a domain-aligned 8B model can outperform open-weight 70B generalists and proprietary GPT-4o on specialized geological reasoning, while a 32B variant approaches frontier reasoning models. The optimized 8B model further offers a competitive cost-performance ratio for deployment. This work provides a reproducible recipe for democratizing scientific LLMs and establishes a baseline for geological artificial intelligence.
>
---
#### [new 194] In Search of the Ingredients of Open-Endedness: Replicating Picbreeder with Large Vision-Language Models
- **分类: cs.AI; cs.CL; cs.CV; cs.NE**

- **简介: 该论文属于AI生成任务，旨在探索AI在开放式创新中的能力。通过复制Picbreeder系统，研究VLMs的创造性输出与人类的区别，分析其多样性与新颖性。**

- **链接: [https://arxiv.org/pdf/2605.23908](https://arxiv.org/pdf/2605.23908)**

> **作者:** Sam Earle; Kay Arulkumaran; Andrew Dai; Akarsh Kumar; Julian Togelius; Sebastian Risi
>
> **备注:** 26 pages, 21 figures, to be published at GECCO 2026
>
> **摘要:** We are in the midst of large-scale industrial and academic efforts to automate the processes of scientific, technological and creative production through AI-driven assistants. Historically, a fundamental property of these processes in their human form has been their open-endedness: their capacity for generating a seemingly endless supply of novel and meaningful new forms. Do artificial agents have any capacity for such fruitful unguided discovery? To answer this question, we turn to Picbreeder, the canonical exemplar of human-driven open-ended search, in which users collaboratively generated a diverse library of images through interactive evolution of small neural networks. We replicate Picbreeder, replacing human users with frontier Vision Language Models (VLMs). We observe clear qualitative differences between the output of our system and the historical human baseline, and attempt to characterize them using metrics of phylogenetic complexity and visual and semantic salience and novelty. In an effort to identify some of the causal factors contributing these differences, we study the addition of exploratory noise to the agents' selection process, of behavioral diversity between agents, and of narrative momentum in the form of memory of past actions. We make our code available at this https URL.
>
---
#### [new 195] When Correct Beliefs Collapse: Epistemic Resilience of LLMs under Clinical Pressure
- **分类: cs.AI; cs.CL; cs.CY; cs.LG**

- **简介: 该论文研究LLMs在临床对话中的信念稳定性问题，属于自然语言处理任务。针对模型在压力下放弃正确诊断的现象，提出评估框架和防御方法，提升模型的稳健性。**

- **链接: [https://arxiv.org/pdf/2605.23932](https://arxiv.org/pdf/2605.23932)**

> **作者:** Boyu Xiao; Xiuqi Tian; Xuwen Song; Haochun Wang; Guanchun Song; Sendong Zhao; Bing Qin
>
> **备注:** ACL 2026
>
> **摘要:** Despite strong medical benchmark accuracy, LLMs can exhibit severe multi-turn sycophancy in clinical dialogue, abandoning initial correct diagnosis under escalating pressure. We propose \textbf{\textsc{Med-Stress}}, a targeted stress test framework that evaluates belief stability under escalating pressure. Across nine frontier large language models (LLMs), we find a clear dissociation between medical knowledge and robustness: high initial diagnostic capability does not imply high belief stability, yielding large knowledge-robustness gaps for several LLMs. To mitigate this failure mode, we propose a lightweight inference-time defense, \textbf{\texttt{RBED}} (\textbf{R}ole-\textbf{B}ased \textbf{E}pistemic \textbf{D}efense), and \textbf{\texttt{R-FT}} (\textbf{R}esilience-oriented \textbf{F}ine-\textbf{T}uning), a training-time approach that internalizes evidence-based resistance to pressure. Experiments show that \textbf{\texttt{R-FT}} nearly eliminates belief change and substantially improves robustness.
>
---
#### [new 196] Second Guess: Detecting Uncertainty Through Abstention and Answer Stability in Small Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的不确定性检测任务，旨在解决小语言模型在不确定时无法正确 abstain 的问题。提出 Second Guess 方法，通过答案稳定性判断不确定性，提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2605.25394](https://arxiv.org/pdf/2605.25394)**

> **作者:** Ashwath Vaithinathan Aravindan; Mayank Kejriwal
>
> **摘要:** Large language models often generate confident but incorrect answers rather than abstaining when uncertain. This problem is particularly acute for small language models (SLMs), where computational constraints and autonomous operation amplify the need for reliable uncertainty detection. We propose _Second Guess_, a lightweight, parameter-free prompting technique for abstention in multiple-choice question answering (MCQA) that is well-suited for SLMs. Our key empirical insight is that models which truly know an answer will select it consistently, while uncertain models exhibit unstable behavior when an ``I don't know'' option is added. Evaluated on four open models (2B-8B parameters) and four benchmarks, Second Guess achieves the highest composite risk improvement of 10.81\%. Notably, it maintains an 8\% composite risk improvement on fine-tuned models where entropy-based methods degrade, and improves most for lower-performing models. All code and results required to reproduce this work is available in this https URL
>
---
#### [new 197] SAMark: A Self-Anchored Text Watermarking with Paragraph-Level Paraphrase Robustness
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于文本水印任务，解决段落级改写攻击下的鲁棒性问题。提出SAMark框架，通过语义空间自锚定和多通道评分机制提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.25796](https://arxiv.org/pdf/2605.25796)**

> **作者:** Jiahao Huo; Wenjie Qu; Yibo Yan; Kening Zheng; Jiaheng Zhang; Xuming Hu; Philip S. Yu; Mingxun Zhou
>
> **摘要:** Semantic-level watermarking (SWM) improves robustness against text modifications by treating sentences as the basic unit. However, robustness to paragraph-level paraphrasing remains difficult because such attacks globally disrupt watermark signals by changing sentence order. In this work, we propose SAMark, a self-anchored watermarking framework that removes the dependency on sentence order by establishing a step-independent green region in semantic space. To improve detectability, we introduce a multi-channel hyperbolic scoring mechanism that amplifies watermark signals while suppressing noise from weakly aligned candidates. We further propose a diversity-aware filtering strategy that combines hard filtering with soft regularization, extending beyond simple n-gram repetition filters to address semantic redundancy. Experimental results show that SAMark achieves up to 90.2% TP@FP1% under typical paragraph-level paraphrasing attacks, outperforming the strongest prior baseline by more than 30% on average, while maintaining generation quality competitive with unwatermarked text and breaking the robustness-quality trade-off that limits prior methods.
>
---
#### [new 198] Automated Detection and Classification of Delusion-related Content in Naturalistic Audio Diaries Using Multi-Agent Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在自动检测和分类自然语音日记中的妄想相关内容。通过多智能体语言模型实现精准识别，解决临床文本分析难题。**

- **链接: [https://arxiv.org/pdf/2605.24755](https://arxiv.org/pdf/2605.24755)**

> **作者:** Feng Chen; Justin Tauscher; Changye Li; Meliha Yetisgen; Alex Cohen; Adam Kuczynski; Angelina Pei-Tzu Tsai; Benjamin Buck; Dror Ben-Zeev; Trevor Cohen
>
> **备注:** Accepted by CLPych 2026
>
> **摘要:** Speech monologues recorded in naturalistic settings provide opportunities to characterize mental illness phenomenology and detect symptom exacerbation. Large language models (LLMs) offer new possibilities for automating this process, as they require annotated data primarily for evaluation rather than training. In this paper, we present a novel automated, multi-agent LLM pipeline for the fine-grained, multi-label extraction of language suggestive of delusional beliefs, associated affective responses, and behavioral responses from transcripts of naturalistic audio diaries collected from people with moderate persecutory ideation. Evaluating an ensemble of three foundation models, we demonstrate that detailed diagnostic prompt instructions successfully reduce false positives for delusional theme classification, but also constrain the interpretation of affective or behavioral responses. Furthermore, comparing multi-agent adjudication frameworks shows that complex conversational debate between agents diminishes accuracy on clinically ambiguous text by inducing premature consensus. Instead, majority voting establishes robust performance (Micro F1 of 0.872 and 0.779 for delusion detection and classification respectively). This work provides a validated and scalable pipeline for the automated detection and characterization of content suggesting delusional beliefs in naturalistic speech.
>
---
#### [new 199] Inference-Time Alignment of Diffusion Models via Trust-Region Iterative Twisted Sequential Monte Carlo
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; stat.ML**

- **简介: 该论文属于生成模型的推理阶段对齐任务，旨在不更新模型权重的情况下引导扩散模型生成高奖励输出。针对现有方法存在的粒子效率低、方差大问题，提出TRI-TSMC框架，通过信任区域优化提升采样效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.25123](https://arxiv.org/pdf/2605.25123)**

> **作者:** Weixin Wang; Yu Yang; Wei Deng; Pan Xu
>
> **备注:** 34 pages, 6 figures, and 7 tables
>
> **摘要:** We study inference-time alignment for diffusion-based generative models, aiming to steer a base model toward high-reward outputs without updating its weights. Recent Sequential Monte Carlo (SMC)-based steering methods approximate reward-tilted target distributions in a principled way, but their proposals remain largely tied to the base sampler. Since reward information is mainly used after propagation through particle reweighting and resampling, these methods can require large particle budgets and suffer from weight degeneracy and high-variance estimates. One way to reduce variance and improve particle efficiency is to iteratively learn twisting functions that provide look-ahead guidance, as in twisted SMC. However, existing learnable twisting methods are developed mainly for classical sequential inference and can be unstable when applied to diffusion-based alignment with high-dimensional state spaces and terminal, noisy, or black-box rewards. We propose Trust-Region Iterative Twisted Sequential Monte Carlo (TRI-TSMC), a trust-region framework for learning twisting functions in SMC-based inference-time alignment. Each iteration computes an exact KL-constrained update in path space, which admits a closed-form solution by tempered importance reweighting, and projects this target back to the parameterized twisted family by weighted maximum likelihood. Theoretically, we formalize the value-function interpretation of the optimal twisting function and show that it yields a zero-variance sampler. We prove that the trust-region update follows an escort path toward the target distribution, that the weighted maximum-likelihood update is a forward-KL projection, and that the path reduces residual importance-weight variance. Empirically, TRI-TSMC improves primary alignment objectives on discrete diffusion text generation and text-to-image generation under matched inference-time budgets.
>
---
#### [new 200] RouteScan: A Non-Intrusive Approach to Auditing MoE LLMs Safety via Expert Routing Telemetry
- **分类: cs.CR; cs.AR; cs.CL; cs.LG**

- **简介: 该论文提出RouteScan，用于非侵入式审计MoE LLMs的安全性，解决隐私与安全的矛盾问题。通过分析GPU级专家路由信息检测有害行为。**

- **链接: [https://arxiv.org/pdf/2605.24817](https://arxiv.org/pdf/2605.24817)**

> **作者:** Bo Lv; Zhiheng Xu; KeDong Xiu; Ruyi Ding; Tianhang Zheng; Zhibo Wang; Kui Ren
>
> **备注:** 20 pages. Under submission
>
> **摘要:** Mixture-of-Experts (MoE) architectures have become an increasingly important paradigm for scaling Large Language Models (LLMs). As MoE models are increasingly deployed in real-world services, safety auditing becomes necessary to verify whether these models produce or facilitate harmful behaviors during operation. However, existing content-based auditing methods typically require access to user prompts, model inputs, or generated outputs, potentially exposing sensitive user information and creating a fundamental tension between LLM safety and user privacy. On the other hand, we observe that, in MoE models, sparse expert routing maps different inputs to activate different expert-execution patterns, producing measurable footprints in low-level GPU execution telemetry. Inspired by this observation, we propose RouteScan, a non-intrusive auditing framework for detecting harmful behaviors through GPU-level expert routing telemetry. Specifically, RouteScan utilizes the number of active GPU threads allocated to expert modules during the prefilling phase as a discriminative micro-architectural fingerprint, and builds a lightweight detection pipeline that isolates cross-domain invariant risk indicators for the precise identification of malicious prompts. Comprehensive evaluations on open-source MoE LLMs with distinct routing designs demonstrate that RouteScan achieves strong generalization, with an AUROC exceeding 0.93 on unseen harmful domains and 0.96 under novel jailbreak wrappers. Moreover, empirical inversion tests show that the collected expert routing telemetry provides limited information for prompt reconstruction, suggesting a practical privacy advantage over content-based auditing methods.
>
---
#### [new 201] Catching The Correct Answer Trap: Characterising AI Tutor Blind Spots When Analysing Student Reasoning
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于智能辅导系统任务，旨在解决AI在评估学生推理时的盲点问题。研究分析学生错误推理导致正确答案的现象，发现模型检测效果有限，强调需结合人工判断。**

- **链接: [https://arxiv.org/pdf/2605.23925](https://arxiv.org/pdf/2605.23925)**

> **作者:** Moiz Imran; Sahan Bulathwela
>
> **备注:** To be published at the International Conference on Artificial Intelligence in Education (AIED'26)
>
> **摘要:** Intelligent tutoring systems increasingly provide automated feedback on student work, but robust feedback requires assessing reasoning, not only final answers. We study a failure mode we call the correct answer trap (CAT): models under-detect misconceptions when students reach a correct answer via flawed reasoning. Analysing real student responses from the Eedi mathematics platform, we show that 71% of these failures concentrate in just two question types, both sharing a common structure where flawed reasoning happens to produce the correct numerical answer. Comparing a fine-tuned T5 with a frontier large language model, we find that improved capabilities reduce but do not eliminate the problem (84% vs 57% detection accuracy). Even the best-performing model generates roughly four false alarms for every genuine detection, making stand-alone screening impractical at realistic class sizes. Our findings demonstrate that high overall accuracy can mask critical failures in reasoning assessment, and that careful analysis of student reasoning still benefits from human judgment.
>
---
#### [new 202] Prism: A Plug-in Reproducible Infrastructure for Scalable Multimodal Continual Instruction Tuning
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于多模态持续指令微调任务，旨在解决MCIT研究中的工程瓶颈。提出Prism框架，通过插件机制实现算法与基础模型的解耦，提升代码复用与实验可重复性。**

- **链接: [https://arxiv.org/pdf/2605.26110](https://arxiv.org/pdf/2605.26110)**

> **作者:** Jun-Tao Tang; Yu-Cheng Shi; Zhen-Hao Xie; Da-Wei Zhou
>
> **备注:** Code is available at this https URL
>
> **摘要:** Multimodal Large Language Models (MLLMs) achieve versatility by reformulating diverse tasks into a unified instruction-following framework via instruction tuning. However, real-world deployment requires continuous adaptation to emerging tasks, motivating Multimodal Continual Instruction Tuning (MCIT). Despite its growing importance, current MCIT research is hindered by severe engineering bottlenecks. Existing methods are typically implemented by directly modifying the base MLLM codebase, which imposes substantial implementation overhead and yields method-specific architectures that severely limit code reuse and fair comparison. To address this, we introduce Prism, a plug-in reproducible codebase specifically designed for scalable MCIT research. It separates algorithmic development from the backbone implementation via a lightweight plugin registration mechanism, enabling new strategies to be integrated as independent plugins without modifying the underlying MLLM codebase, thereby eliminating structural fragmentation and accelerating method development. Prism natively supports widely used large-scale training pipeline, thereby enabling reproducible and scalable MCIT experimentation. Code is available at this https URL.
>
---
#### [new 203] Hypothesis Generation and Inductive Inference in Children and Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究儿童与语言模型在不确定环境中的归纳推理能力，通过程序归纳任务比较两者的行为模式，探讨其信息寻求策略及适应机制。**

- **链接: [https://arxiv.org/pdf/2605.24528](https://arxiv.org/pdf/2605.24528)**

> **作者:** Jeffrey Qin; Wasu Top Piriyakulki; Zhuangfei Gao; Mia Radovanovic; Jessica Sommerville; Kevin Ellis; Marta Kryven
>
> **摘要:** Real world decision-making requires constructing mental models under uncertainty over evidence, over the underlying causal rules, and over the state of the world itself. Which computational principles underpin human inference under such conditions, and do LLM-based agents exhibit similar behavior given matching constraints? We address these questions using an inductive inference Box Task in which participants, human children and LLM-based agents, infer a latent cause through sequential interaction with an uncertain environment. We formalize this task as program induction with Bayesian particle-based inference, admitting two complementary interpretations: (1) as a constraint satisfaction process over hypotheses, and (2) as a program synthesis problem in which hypotheses are executable programs evaluated against evidence. Using the constraint-based formulation, we show that children's behavior is best explained by a combination of subjective evidence reliability and online hypothesis generation, accounting for both their evidence-seeking patterns and their dissociation between task completion and rule generalization. Using the program synthesis formulation, we treat LLM-based agents as model organisms: controllable systems that allow systematic manipulation of task conditions. Across backends, LLM-based agents replicate children's responses to changes in evidence reliability and observability, including discounting unreliable evidence, seeking to resolve partial information, and dissociating between task completion and causal generalization. At the same time, LLM-based agents tend to over-observe and over-comply with instructions relative to children. These results suggest that while children and LLM-based agents adapt similarly to environmental structure, their information-seeking behavior exhibits distinct underlying costs and inductive biases.
>
---
#### [new 204] Retrieval-Augmented Detection of Potentially Abusive Clauses in Chilean Terms of Service
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于法律文本分析任务，旨在检测智利服务条款中的潜在有害条款。通过构建语料库和检索增强框架，提升模型对合同中不公平条款的识别能力。**

- **链接: [https://arxiv.org/pdf/2605.26019](https://arxiv.org/pdf/2605.26019)**

> **作者:** Christoffer Loeffler; Tomás Rey Pizarro; Daniel Ignacio Miranda Vásquez; Andrea Martínez Freile
>
> **备注:** 42 pages, 6 figures, 9 tables
>
> **摘要:** Online Terms of Service often function as contracts of adhesion, creating asymmetries that may expose consumers to potentially abusive clauses. In Chile, assessing such clauses is legally challenging because some provisions clearly violate mandatory consumer law, whereas others depend on broader standards such as good faith and contractual imbalance. We present a retrieval-augmented generation framework for the automated detection and classification of potentially abusive clauses in Chilean Terms of Service. Designed for local execution, it combines efficient clause detection, hybrid dense--sparse retrieval, reranking, and prompt augmentation to support medium-sized open-weight language models. We also introduce the Chilean Abusive Terms of Service Extended corpus, comprising 100 contracts and 10,029 annotated clauses in 24 legally grounded categories spanning illegal, dark, and gray clauses. Experiments comparing commercial and open-weight language models, fine-tuned encoders, and traditional baselines show that retrieval-augmented prompting substantially improves performance and enables local models to approach larger cloud-based systems at lower computational and token cost. The study also contributes a refined legal annotation scheme and a practical design for AI-assisted consumer contract review.
>
---
#### [new 205] Neural Router: Semantic Content Matching for Agentic AI
- **分类: cs.DC; cs.CL; cs.IR; cs.NI**

- **简介: 该论文属于内容匹配任务，解决agentic AI中语义匹配问题。通过LLM提升跨域订阅效率，分析模型性能与压缩策略的关系。**

- **链接: [https://arxiv.org/pdf/2605.25701](https://arxiv.org/pdf/2605.25701)**

> **作者:** Lauri Lovén; Abhishek Kumar; Alexander Engelhardt; Alaa Saleh; Roberto Morabito; Xiaoli Liu; Naser Hossein Motlagh; Sasu Tarkoma
>
> **备注:** 35 pages, 12 figures. Combined main paper and electronic supplement, folded into one document for arXiv
>
> **摘要:** Large language models (LLMs) can serve as the semantic-matching engine of a content-based publish/subscribe broker for agentic AI across the edge-cloud computing continuum, bridging the vocabulary and modality gaps that defeat keyword and embedding filters. Framed as offline multi-label retrieval over three public datasets spanning social-media, legal, and smart-home sensor domains (six LLMs, seven baselines), our central contribution is a two-crossover cost-accuracy characterisation: an analytical context-window crossover below which a CoverAndMerge compression pipeline reduces LLM invocations, and an empirical discrimination-capacity crossover above which matching accuracy collapses independently of context budget, by a model-dependent factor of parameter count and training generation. Two findings carry practical weight: above the discrimination crossover, compression cannot recover accuracy and only frontier-scale models clear large subscription sets; and there backend choice dominates configuration choice, so model selection, not pipeline tuning, is the primary operator lever. We accompany this with three composable algorithms and a per-cluster Quality-of-Experience framework for autonomic LLM-tier selection.
>
---
#### [new 206] An Effective-Rank Audit of Alignment-Induced Activation Shifts: Confound Control, Constructive Calibration, and Limits
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属于大模型对齐研究，旨在分析对齐导致的激活变化。通过有效秩审计，控制混杂因素、校准模型并探讨诊断方法的局限性。**

- **链接: [https://arxiv.org/pdf/2605.24583](https://arxiv.org/pdf/2605.24583)**

> **作者:** Yuki Nakamura
>
> **备注:** 18 pages, 1 figure, 21 tables. Code, data, and an immutable Zenodo archive are available at this https URL (DOI: https://doi.org/10.5281/zenodo.20341445)
>
> **摘要:** We audit alignment-induced shifts in residual-stream activations of three open-weight instruction-tuned LLMs (Llama-3.1-8B-Instruct, Gemma-2-9B-it, Qwen-2.5-7B-Instruct) using the effective rank of the alignment modification matrix on safety-relevant inputs, rho_eps := rank_eps(M_Ds)/d, which formalizes the single-refusal-direction observation of Arditi et al. (2024) as a continuous quantity. The paper has three contributions. (1) Confound-controlled measurement: a four-variant decomposition (M_naive, M_template, M_aligned, M_DiD) separates chat-template formatting, alignment-stage shift, and the refusal-mediating direction, and recovers the Arditi refusal direction on M_DiD at |cos| in {0.77, 0.86, 0.50} (Llama/Gemma/Qwen); chat-template-controlled rho_eps is {0.0029, 0.0048, 0.0044}, and the centered SVD residual is 4-7x larger. (2) Constructive calibration on a 3-layer MLP across rho_eps in {0.008, 0.17, 0.33, 0.40} exhibits a sweet-spot vs. brittle distinction: mild rank-maximization (lambda=5) buys ablation robustness, while strong regularization at the same nominal rho_eps (lambda=50) does not. rho_eps is a diagnostic for fragility, not a target whose mechanical inflation buys robustness. (3) Limits of rank-based diagnostics: (a) not safety-specific (LRH baseline is 2-3x the safety value); (b) SVD principal ordering does not match causal ordering (Llama u_2 inert despite ranking second; cumulative ablation non-monotone at k=5); (c) the spectral-gap hypothesis required to upgrade the O(rho_eps * d) achievability bound to a matching Mirsky-route lower bound fails empirically (1/90 Llama layer-reference pairs, 0/36 MLP combinations) and structurally (kappa_lb <= 2/(eps * r)). The matching lower bound remains an open problem.
>
---
#### [new 207] When Search Becomes Memory: Turning Robot Design Trials into Transferable Skills
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于机器人设计任务，旨在解决传统进化算法记忆缺失问题。通过构建可迁移技能库，将设计经验转化为可审计的知识，提升搜索效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.25832](https://arxiv.org/pdf/2605.25832)**

> **作者:** Yunfei Wang; Xiaohao Xu; Yang Li; Xiaonan Huang
>
> **备注:** 20 pages, 8 figures
>
> **摘要:** Large language models (LLMs) are increasingly used as proposal generators for evolutionary robot design, yet most loops remain memoryless: simulator results shape the next population but are not preserved as reusable design knowledge. We present Auto-Robotist, a self-evolving LLM agent that distills morphology-search traces into an explicit natural-language skill library. Each skill stores a structural archetype, evidence-grounded positive and negative rules, and the evaluated designs that support them, making design memory inspectable rather than implicit in a population. During search, the agent retrieves skills to condition LLM edits of elite bodies while retaining a Genetic Algorithm (GA) mutation path for exploration; after evaluation, it updates the library through Add, Diagnose, and Merge. Across seven EvoGym tasks spanning locomotion, traversal, and object interaction, Auto-Robotist improves cold-start 5x5 search and transfers learned skills to 10x10 design spaces, where reference-conditioned transfer outperforms GA on every task. These results suggest that LLM agents can convert expensive physical evaluations into reusable, auditable design principles. Our code will be released upon acceptance.
>
---
#### [new 208] STORM: Internalized Modeling for Spatial-Temporal Reasoning in Video-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出STORMS，解决视频-语言模型中的时空推理问题。通过内部化推理过程，减少对外部工具或视频生成的依赖，提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.26014](https://arxiv.org/pdf/2605.26014)**

> **作者:** Yiming Liang; Yixiao Chen; Yiyang Zhou; Yixuan Wang; Shoubin Yu; Andong Deng; Fuxiao Liu; Qin Zhang; Chen Chen; Mohit Bansal; Huaxiu Yao
>
> **摘要:** Many video reasoning tasks require tracking motion, temporal order, and evolving visual states across frames. Existing methods built on large vision-language models (LVLMs) often address this challenge by externalizing reasoning through textual chain-of-thought (CoT), keyframe selection, repeated frame reinsertion, or external tool use. While effective, such pipelines increase inference-time latency and engineering complexity, and they force temporal-visual evidence to be serialized into text or repeatedly re-encoded from frames. Inspired by the intuition that visual reasoning can occur implicitly before verbalization, we propose STORMS (Spatial-Temporal reasOning via inteRnalized Modeling), a two-stage framework that teaches LVLMs to reason through bounded continuous latent trajectories instead of explicit textual CoT. In Stage I, STORMS aligns latent tokens with thought-video representations derived from generated videos, grounding the latent states in dynamic visual evidence. In Stage II, the model is further trained with answer-only supervision, encouraging the reasoning process to be internalized without step-by-step annotations. Generated thought videos are used only during training; at inference, STORMS performs a bounded latent rollout without regenerating videos, reinserting frames, or invoking external visual tools. Experiments on VideoMME, MVBench, TempCompass, and MMVU show that STORMS improves video reasoning accuracy while substantially reducing inference overhead compared with tool or video-generation-based reasoning pipelines.
>
---
#### [new 209] Efficient Benchmarking Is Just Feature Selection and Multiple Regression
- **分类: stat.ML; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于高效基准测试任务，旨在通过特征选择和多重回归降低LLM评估成本，提升预测精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.25773](https://arxiv.org/pdf/2605.25773)**

> **作者:** Sam Bowyer; Acyr Locatelli; Kris Cao
>
> **备注:** 36 pages, 27 figures
>
> **摘要:** Efficient benchmarking techniques aim to lower the computational cost of evaluating LLMs by predicting full benchmark scores using only a subset of a benchmark's questions. By reframing this problem as an instance of multiple regression with feature selection, we find that existing efficient benchmarking methods can be greatly improved by simply using kernel ridge regression at the prediction stage. Additionally, using an information-theoretic feature-selection algorithm called minimum redundancy maximum relevance (mRMR), we can further improve upon these methods by selecting question subsets that will be maximally useful for prediction. Except in very data-poor settings, these approaches consistently achieve smaller prediction errors (in both MAE and RMSE), and greater ranking correlation between predicted and true scores (in both Spearman $\rho$ and Kendall $\tau$) across a range of benchmarks using both binary and continuous metrics. Furthermore, mRMR subsampling is much faster than competitor methods (which often involve fitting probabilistic models or running clustering algorithms), and is more likely to select the same questions under different random seeds or training data splits. Tutorial code can be found at this https URL .
>
---
#### [new 210] CMAP: Cross-Modal Adaptive Prompting for Multi-Domain Task-Incremental Learning
- **分类: cs.CV; cs.CL; cs.ET**

- **简介: 该论文提出CMAP方法，解决多领域任务增量学习中的知识遗忘问题，通过跨模态文本空间优化任务路由与信心估计，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.25708](https://arxiv.org/pdf/2605.25708)**

> **作者:** Sriram Mandalika
>
> **摘要:** Multi-domain task-incremental learning requires a model to sequentially acquire knowledge across visually diverse domains without forgetting prior tasks, and without access to task identity at inference. Parameter-efficient methods built on frozen vision-language models have made strong progress, yet all existing approaches rely exclusively on visual features for task routing, confidence estimation, and encoder adaptation, leaving CLIP's cross-modal text embedding space entirely unexploited. We address this gap through three contributions. Text-space task routing replaces visual Gaussian matching with cosine similarity to frozen CLIP text prototypes, giving order-independent routing robust to data scarcity at zero parameter cost. Multi-prototype visual-textual confidence replaces single-Gaussian class modeling with K-means visual prototypes and cross-modal alignment scores under task-calibrated thresholds. Symmetric cross-modal gating extends per-layer Gumbel gates to the text encoder conditioned on batch image features, preserving cross-modal alignment on out-of-distribution inputs. On the MTIL benchmark spanning 11 datasets and 1201 classes, our method achieves 74.2% Transfer, 80.5% Average, and 88.7% Last under Order-I, surpassing the prior state of the art by 5.0, 3.7, and 3.0 percentage points with only 2.5M trainable parameters and no external data.
>
---
#### [new 211] Jailbreak to Protect: Buffering and Reinforcing via Temporary Jailbreaking for Safe Fine-Tuning in Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于大语言模型安全微调任务，解决有害微调导致的安全对齐下降问题。通过临时隔离有害更新并强化安全机制，提升模型安全性与任务性能。**

- **链接: [https://arxiv.org/pdf/2605.24550](https://arxiv.org/pdf/2605.24550)**

> **作者:** Seokil Ham; Jaehyuk Jang; Wonjun Lee; Changick Kim
>
> **备注:** ICML 2026 Spotlight
>
> **摘要:** Fine-tuning-as-a-Service (FaaS) enables personalization of large language models (LLMs), but it can weaken safety-alignment under harmful fine-tuning attacks. Recent work has shown that activating harmful-behavior modules during fine-tuning can prevent models from learning undesired behaviors, but its mechanism remains unclear. In this paper, we revisit temporary jailbreaking as a defense against harmful fine-tuning and provide a gradient-level analysis showing that it saturates safety-degrading gradients while preserving benign task-relevant gradients. Based on this insight, we propose a Buffer-and-Reinforce fine-tuning framework that buffers harmful updates during user fine-tuning and reinforces safety after adaptation. Specifically, BufferLoRA induces temporary jailbreaking as a removable adapter to reduce harmful updates during user fine-tuning. After adaptation, ReinforceLoRA, trained to recover refusal behavior under the temporarily jailbroken state, is integrated with UserLoRA via QR decomposition-based merging to reinforce safety while preserving user-task performance. Extensive experiments show that our framework achieves superior safety and utility with no additional safety data during user fine-tuning and minimal computational cost.
>
---
#### [new 212] Spiking the training data to correct for test set contamination
- **分类: stat.ME; cs.CL; cs.LG**

- **简介: 该论文属于模型评估任务，旨在解决测试集污染导致的评分过高问题。通过在训练数据中引入污染样本，校准模型记忆预测器，实现对测试分数的统计修正。**

- **链接: [https://arxiv.org/pdf/2605.24818](https://arxiv.org/pdf/2605.24818)**

> **作者:** Johnny Tian-Zheng Wei; Jerry Li; Ameya Godbole; Robin Jia
>
> **摘要:** The literature on test set contamination largely focuses on detection, but the correction of contaminated test scores is underexplored. Our core proposal is to spike the training data by intentionally contaminating some test examples at known rates. The spiked examples can then be used to calibrate predictors of model memorization which enable principled statistical correction of inflated test scores. To evaluate different correction estimators, we first present a simulation framework based on the Hubble models. Hubble models come in minimal pairs, where the perturbed model was deliberately contaminated with several test sets, while the standard model was not, serving as the counterfactual and correction target. We consider estimators that use information from a memorization predictor, correctness predictor, or both. In simulation, we establish basic statistical intuitions and show that estimators leveraging memorization and correctness information are better than naive estimation which makes no correction at all. We then instantiate several memorization and correctness predictors, and find that simple predictors such as Platt-scaled membership inference metrics provide good signal for correction. Finally, we examine the practical considerations of spiking. Simple memorization predictors need no more than 10 examples for calibration and often transfer from one dataset to another. Taken together, spiking is a promising solution for test set contamination.
>
---
#### [new 213] AgentFugue: Agent Scaling for Long-Horizon Tasks through Collective Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究长周期代理任务，解决多代理协同问题。提出AgentFugue框架，通过共享推理中心实现代理间协作，提升任务处理能力。**

- **链接: [https://arxiv.org/pdf/2605.24486](https://arxiv.org/pdf/2605.24486)**

> **作者:** Yuyang Hu; Hongjin Qian; Shuting Wang; Jiongnan Liu; Tong Zhao; Xiaoxi Li; Zheng Liu; Zhicheng Dou
>
> **摘要:** Recent progress on long-horizon agentic tasks has been driven largely by scaling up individual agents through stronger models, better tools, and more effective scaffolding. In contrast, much less is understood about scaling out: whether multiple peer agents, all targeting the same task, can become an additional source of capability without relying on explicit role specialization or workflow orchestration. We study this question and propose AgentFugue, a collective reasoning framework built around a shared reasoning hub. As peer agents explore the same task in parallel, the hub records concise notes on what each agent has established, attempted, or ruled out, and enables each agent to selectively access what other agents have discovered in a form useful for its current search. This design turns otherwise isolated trajectories into a connected ecology of reusable intermediate reasoning without requiring centralized planning. We instantiate the hub as a plug-in communication layer, trained with supervised fine-tuning and end-to-end reinforcement learning. Across the challenging long-horizon settings we study, AgentFugue improves over strong baselines. Our results suggest that collective reasoning can turn scaling out peer agent systems into a distinct source of capability gains, rather than merely a way of spending more compute.
>
---
#### [new 214] The Multilingual Curse at the Retrieval Layer: Evidence from Amharic
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于多语言信息检索任务，旨在解决零样本多语言检索在资源匮乏语言中的有效性问题。通过对比不同检索模型在阿姆哈拉语上的表现，发现需针对性优化以实现公平的信息访问。**

- **链接: [https://arxiv.org/pdf/2605.24556](https://arxiv.org/pdf/2605.24556)**

> **作者:** Yosef Worku Alemneh; Kidist Amde Mekonnen; Maarten de Rijke
>
> **备注:** 10 pages, 4 tables. Accepted to the 1st Workshop on Multilinguality in the Era of Large Language Models (MeLLM) at ACL 2026
>
> **摘要:** Multilingual retrieval increasingly underpins cross-lingual question answering and retrieval-augmented generation. Strong zero-shot scores on multilingual benchmarks are often taken as evidence that current encoders transfer reliably across many languages. We argue that this assumption breaks down for underrepresented, morphologically rich languages, and use Amharic as a diagnostic case. Under a shared passage retrieval protocol covering dense, late-interaction, learned sparse, and cross-encoder paradigms, we compare zero-shot multilingual retrievers, Amharic-fine-tuned multilingual retrievers, and monolingual Amharic retrievers. The strongest zero-shot multilingual retriever underperforms the strongest monolingual Amharic first-stage retriever by 23% relative MRR@10. Fine-tuning two recent multilingual embedding models on the same Amharic supervision yields 32-60% relative MRR@10 gains over zero-shot, but the best Amharic-fine-tuned multilingual model remains below the strongest monolingual Amharic retriever. These findings indicate that zero-shot multilingual retrieval is not a sufficient proxy for equitable information access in the LLM era: for underrepresented languages, retrieval must be evaluated and adapted in-language rather than inferred from aggregate multilingual benchmarks. To foster future research, we publicly release the dataset, codebase, and trained models at this https URL.
>
---
#### [new 215] AgentIR: A Workload-Adaptive Cascade Retrieval Substrate for Long-Term Conversational Memory
- **分类: cs.IR; cs.CL; cs.DB**

- **简介: 该论文提出AgentIR，解决长时对话记忆中的检索问题，通过自适应级联机制提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.25092](https://arxiv.org/pdf/2605.25092)**

> **作者:** Aojie Yuan; Haiyue Zhang; Shahin Nazarian
>
> **备注:** 29 pages, 9 figures, 12 tables. Main paper 9 pages + comprehensive appendix (proof, GPU kernels, full per-dataset BEIR/LongMemEval/LoCoMo tables, cascade router C++ API, 6 robustness experiments, FAQ, failure-case catalog)
>
> **摘要:** Long-term conversational memory is a retrieval workload classical IR was not built for: the index grows during the query stream, query types shift intra-session, and the latency budget per retrieval is sub-10 ms. Lucene-class engines treat the index as static and the query as stateless, leaving the workload's structure unexploited. AgentIR treats fusion as a per-query decision along two axes: which fusion to apply (BM25, Dense, RRF, or agent-aware RRF), and whether the ~52 ms dense channel is worth running at all. The second axis is a confidence-triggered cascade router that decides from the BM25 top-k margin alone and re-tunes across workloads without retraining. On LongMemEval (n=500), where the dense channel does add information, the cascade skips 63% of queries at parity LLM-judged accuracy (2.67x faster under two judges, paired bootstrap p>=0.88); per-qtype thresholds extend this to 5.76x under 5-fold cross-validation. On LoCoMo (n=1,982), where BM25 alone is already the strongest single system, the same trigger auto-tunes to a 100% skip rate (132x faster, +0.089 Hit@5). Capacity on a shared 8-core VM rises from ~154 to ~1,400 concurrent agents (9x). Underneath the cascade, a time-partitioned index does O(log 1/epsilon) work independent of corpus size: 1234x corpus growth costs only 3.6x latency, ending in 1769x over sequential at sub-100 us p50 on 5M records. At parity quality with Lucene on 9 BEIR datasets up to 8.8M docs, the substrate runs 10x geo-mean over Pyserini 8T and 11x over PISA-1T BlockMax-WAND; an A100 reaches 1.8-39x over Pyserini 8T; chunked index build sustains 56.8K docs/sec on MS MARCO. Three subtle BM25/GPU correctness pitfalls that silently regress nDCG@10 by 6-8x are documented and fixed; post-fix CPU and GPU agree within 0.0002 nDCG@10 on all eight datasets that fit a single A100.
>
---
#### [new 216] MinerU-Popo: Universal Post-Processing Model for Structured Document Parsing
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出MinerU-Popo，解决OCR后处理中跨页结构恢复问题，通过文本、表格、标题和图像的联合分析，提升文档级信息一致性。**

- **链接: [https://arxiv.org/pdf/2605.24973](https://arxiv.org/pdf/2605.24973)**

> **作者:** Bangrui Xu; Ziyang Miao; Xuanhe Zhou; Yiming Lin; Zirui Tang; Xiaomeng Zhao; Fan Wu; Cheng Tan; Fan Wu; Bin Wang; Conghui He
>
> **备注:** The code is available at this https URL
>
> **摘要:** VLM-based OCR models have become the de facto choice for document parsing, as they can accurately extract page-level elements (e.g., paragraphs within individual pages) together with their bounding boxes and textual content. However, downstream applications such as RAG require coherent document-level information, whereas these models often break cross-page continuity and fail to recover disrupted structures, such as paragraphs and tables truncated by page boundaries. Such relationships are not confined to a single page; instead, they require joint analysis of titles, paragraphs, tables, and images spanning multiple pages. A natural solution is therefore to reuse existing OCR outputs and reconstruct document-level logical structures through post-processing. To this end, we propose MinerU-Popo, a lightweight and universal framework for POst-Processing OCR outputs, which converts page-level results from diverse parsers into coherent document-level structures. MinerU-Popo decomposes the problem into four focused subtasks: text truncation recovery, table truncation recovery, title hierarchy reconstruction, and image-text association. To address these effectively, we build a task-oriented data engine with task-specific input filtering, and use the generated data (30K) to fine-tune a lightweight post-processing model (Qwen3-VL-4B). To support long documents, we introduce dynamic chunking with overlap-based synchronization, which aligns chunk-level outputs from the fine-tuned model and preserves global consistency. Finally, we assemble the aligned outputs into a tree-structured document representation, further enriched with node chunking and summaries for downstream retrieval and analysis. Empirical results show MinerU-Popo improves title-hierarchy TEDS by at least 20% across all five tested OCR models, improves RAG accuracy and reduces per-query latency.
>
---
#### [new 217] Machine Psychometrics: A Mathematical Psychology of Artificial Intelligence
- **分类: cs.AI; cs.CL; q-bio.NC**

- **简介: 该论文提出“机器心理测量学”，旨在通过数学心理学方法评估人工智能的行为特征，解决人工智能评价工具不足的问题。任务是建立客观测量体系，工作包括构建机器心智图谱和信任协议。**

- **链接: [https://arxiv.org/pdf/2605.23952](https://arxiv.org/pdf/2605.23952)**

> **作者:** Alex Bogdan; Adrian de Valois-Franklin
>
> **备注:** 45 pages, 11 figures
>
> **摘要:** Artificial agents now generate behavior rich enough to invite trust, surprise, and concern, yet our evaluation tools still privilege capability scores over psychological structure. This paper argues that the philosophical impasse between two symmetrical errors (Artificial Mind Blindness, which dismisses psychological organization in non-biological systems, and Artificial Mind Projection, which infers human-like inner life from fluent behavior alone) can be circumvented not by resolving the consciousness question, but by introducing a disciplined measurement layer beneath it. Drawing on Michael Levin's continuum view of cognition as goal-directed competency across substrates, and on the methodological repertoire of mathematical psychology (Item Response Theory, Signal Detection Theory, Bayesian cognitive modeling, calibration analysis, cognitive-bias batteries), the paper develops Machine Psychometrics as a measurement science of latent behavioral, metacognitive, communicative, and self-modeling dispositions in artificial agents. Its operational core is the Machine Mindprint: a multidimensional, domain-bounded, versioned profile spanning calibration, source integrity, suggestibility resistance, context stability, expressive alignment, tool integrity, drift monitoring, and distributional grounding. A complementary Trust Protocol turns Mindprints into deployment decisions through probe batteries, perturbation testing, reliability and validity analysis, and longitudinal monitoring across high-stakes domains. The philosophical contribution is a third stance, Artificial Mind Discipline, that neither anthropomorphizes nor dismisses, neither presupposes consciousness nor forecloses it. The aim is not to humanize artificial agents, but to understand them precisely because they are not human, through measurement before judgment.
>
---
#### [new 218] Agent-ToM: Learning to Monitor Autonomous LLM Agents via Theory-of-Mind Reasoning
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **简介: 该论文提出Agent-ToM，用于监控自主大语言模型代理的隐蔽恶意行为。任务是安全分析，解决检测隐藏目标的问题，通过理论心智推理进行轨迹分析与验证。**

- **链接: [https://arxiv.org/pdf/2605.24216](https://arxiv.org/pdf/2605.24216)**

> **作者:** Nesreen K. Ahmed; Nima Nafisi
>
> **备注:** 23 pages, 9 figures
>
> **摘要:** Monitoring autonomous large language model (LLM) agents for covert malicious behavior is challenging due to delayed, context-dependent, and long-horizon attack patterns. Agents may pursue hidden objectives while maintaining superficially benign behavior, making detection difficult even with full trajectory access. Prior monitoring approaches improve scaffolding or ensemble aggregation, but treat each trajectory independently and do not learn from prior monitoring experience. Moreover, standard reasoning methods explain observed behavior without explicitly reasoning about agent beliefs, intentions, and goal alignment required to distinguish benign task execution from covert deviation. We propose \textbf{Agent-ToM}, a learning-to-monitor framework grounded in Theory-of-Mind (ToM) reasoning for security analysis of autonomous agents. Agent-ToM performs structured full-trajectory analysis by inferring beliefs, intent hypotheses with calibrated confidence, expected actions, and deviations from task-consistent behavioral baselines. At inference time, it employs a \textit{Reason-Verify-Refine} pipeline to construct and validate monitoring decisions. At training time, Agent-ToM distills critique signals into a persistent \textit{semantic guardrail memory}, enabling reusable belief- and intent-conditioned constraints across episodes. We evaluate Agent-ToM on adversarial agent monitoring benchmarks (SHADE-Arena and CUA-SHADE-Arena). Agent-ToM achieves strong precision-recall balance and outperforms state-of-the-art monitoring baselines, including ensemble methods, while using a coherent two-call reasoning pipeline. These results demonstrate that learning at the monitoring layer, combined with structured ToM reasoning and verification, provides an effective and deployable foundation for securing autonomous LLM agents.
>
---
#### [new 219] SemanticZip: A Pilot Framework for Lossy Text Compression with LLMs as Semantic Decompressors
- **分类: cs.LG; cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出SemanticZip框架，研究基于LLM的有损文本压缩任务。通过将文本压缩为可被LLM解码的语义代码，解决如何在保证任务相关语义的前提下实现高效压缩的问题。**

- **链接: [https://arxiv.org/pdf/2605.24541](https://arxiv.org/pdf/2605.24541)**

> **作者:** Natalia Trukhina; Vadim Vashkelis
>
> **备注:** 13 pages, 1 figure, 2 tables. Pilot framework paper; code and supplementary artifacts available in ancillary files
>
> **摘要:** Text compression for large language model (LLM) systems is usually framed as token deletion, retrieval, summarization, or exact reconstruction. We study a more aggressive but explicitly lossy setting: compress text into compact codes that an LLM can expand into task-relevant meaning. We call this setting SemanticZip. Unlike lossless compression, SemanticZip does not require byte-identical reconstruction; unlike ordinary summarization, it treats model-based decompression as part of the codec and evaluates whether task-relevant semantic commitments are recovered. This paper is a pilot framework, not a benchmark claim. We formalize LLM-mediated decompression, define a protected/lossy packet architecture, and evaluate six representation regimes over five author-constructed diagnostic cases: structured prose, JSON, CCL-Core, CCL-Min, SemanticZip ASCII, and SemanticZip emoji. An independent decoder LLM reconstructs typed semantic atoms from each compressed representation, and we score Critical Atom Recall, Weighted Atom Recall, precision, and tokenizer gain. In this pilot, structured prose has the highest recoverability, with WAR = 0.956 and 19.1% o200k_base token gain. CCL-Min is the strongest balanced point, with 39.4% token gain and WAR = 0.874. SemanticZip ASCII provides the largest useful compression, with 46.5% token gain and WAR = 0.802, while emoji-heavy SemanticZip performs worse on both compression and recovery. The main contribution is not the claim that these numbers establish a universal frontier. Rather, we introduce a reproducible experimental interface for studying lossy, LLM-decompressible text codes and a design principle: safety-critical and exact commitments should remain protected, while predictable low-risk context may be semantically zipped.
>
---
#### [new 220] Directional Alignment Mitigates Reward Hacking in Reinforcement Learning for Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决语言模型中的奖励欺骗问题。通过分析参数更新方向，提出受信方向投影方法，减少优化偏离，提升任务性能。**

- **链接: [https://arxiv.org/pdf/2605.25189](https://arxiv.org/pdf/2605.25189)**

> **作者:** Wenlong Deng; Jiaji Huang; Kaan Ozkara; Yushu Li; Christos Thrampoulidis; Xiaoxiao Li; Youngsuk Park
>
> **摘要:** Reward hacking arises when a model improves a proxy reward by exploiting shortcuts rather than solving the intended task. We study this failure mode through the geometry of reinforcement learning updates in language models and argue that hacking emerges when optimization drifts away from a stable low-dimensional learning trajectory. We analyze this drift through dominant singular directions of parameter updates and show that reward-hacking runs exhibit substantially larger directional change than clean runs. Motivated by this observation, we introduce trusted-direction projection, which constrains gradients to remain within a clean reference subspace. Across reward-hacking experiments on mathematical reasoning, the proposed approach delays shortcut exploitation and better preserves task performance.
>
---
#### [new 221] ECHO: Terminal Agents Learn World Models for Free
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出ECHO方法，解决终端代理学习问题，通过结合策略梯度与环境预测损失，提升模型对终端动态的预测能力。**

- **链接: [https://arxiv.org/pdf/2605.24517](https://arxiv.org/pdf/2605.24517)**

> **作者:** Vaishnavi Shrivastava; Piero Kauffmann; Ahmed Awadallah; Dimitris Papailiopoulos
>
> **摘要:** CLI agents are the closest thing language models have to an embodied setting: the model emits commands, the terminal executes them, and the returned stream -- stdout, errors, files, logs, and traces -- records the consequences. We argue that this stream is a supervision signal, but standard agent RL discards it: GRPO-style training updates action tokens with sparse outcome-level rewards while ignoring environment responses already in the rollout. Failed rollouts provide little policy-gradient signal despite containing rich evidence about how the environment responds. We introduce ECHO (Environment Cross-entropy Hybrid Objective), a hybrid objective that combines the standard policy-gradient loss on action tokens with an auxiliary loss that trains the policy to predict environment observation tokens resulting from its own actions. ECHO reuses the same forward pass as GRPO, requires no additional rollouts, and turns terminal feedback into dense supervision for all rollouts. ECHO doubles GRPO pass@1 on TerminalBench-2.0: Qwen3-8B improves from 2.70% to 5.17%, and Qwen3-14B from 5.17% to 10.79%. ECHO also produces policies that better predict terminal dynamics, even on trajectories they did not generate: across held-out rollouts, it sharply reduces environment-token cross-entropy while GRPO alone barely changes it. From base Qwen3-8B, ECHO matches expert-SFT-then-GRPO performance on held-out terminal tasks without expert demonstrations, and recovers roughly half of the expert-SFT initialization benefit on TerminalBench-2.0. In some settings, the environment prediction loss alone enables verifier-free self-improvement, allowing policies to improve on unseen OOD tasks by learning only from environment interactions. Together, these results suggest that environment observations are not merely context for future actions, but a dense, on-policy supervision signal already present in every rollout.
>
---
#### [new 222] GlobalDentBench: A Multinational Benchmark for Evaluating LLM Clinical Reasoning in Dentistry with Expert Calibration
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医疗AI评估任务，旨在解决LLM在牙科临床推理中的安全性和可靠性问题。构建了全球首个多国牙科基准GlobalDentBench，评估模型在不同推理层级的表现。**

- **链接: [https://arxiv.org/pdf/2605.24636](https://arxiv.org/pdf/2605.24636)**

> **作者:** Junjie Zhao; Jingyi Liang; Zhenyang Cai; Jiaming Zhang; Zhenwei Wen; Shuzhi Deng; Wenjing Yi; Chunfeng Luo; Hexian Zhang; Junying Chen; Tianrui Liu; Zhuhui Bai; Zixu Zhang; Pradeep Singh; Xiang Liu; Jianquan Li; Nhan L Tran; Falk Schwendicke; Zuolin Jin; Lijian Jin; Liangyi Chen; Wei-fa Yang; Benyou Wang; Junwen Wang; Shan Jiang
>
> **摘要:** While large language models (LLMs) hold transformative potential for medicine, their reasoning robustness and safety in real-world clinical scenarios remain critically underexplored, particularly in dentistry. Here we introduce GlobalDentBench, the first multinational dental benchmark, featuring a taxonomy that encompasses 14 dental specialties across 88 countries and regions spanning six continents. The benchmark comprises 8,978 expert-validated questions across three formats (multiple-choice, short-answer, and case-based questions) and assesses three progressive reasoning levels: knowledge recall (L1), routine reasoning (L2), and individualized reasoning (L3). To ensure data quality, the automated construction framework was calibrated by six senior dentists, achieving expert agreement rates of 99.98% for multiple-choice and short-answer questions and 96.78% for the more complex case-based questions. Evaluation of 12 frontier LLMs on GlobalDentBench revealed a sharp, stepwise performance degradation with increasing reasoning complexity. Specifically, accuracy plummeted from 81.34% on multiple-choice to 64.53% on short-answer and 22.34% on case-based questions, while declining markedly from 74.01% at L1 to 55.64% at L2 and 35.71% at L3. More critically, risk analysis of real-world dental cases demonstrated an alarming overall unsafe rate of 31.01% in LLM-generated clinical recommendations, with 4.51% posing risks of irreversible patient harm and risks particularly pronounced in specialties such as orthodontics. These findings expose fundamental limitations in the medical reasoning and safety of current LLMs. Consequently, GlobalDentBench provides a scalable foundation for trustworthy clinical AI evaluation, underscoring the urgent need for rigorous validation before the safe deployment of these models in healthcare.
>
---
#### [new 223] What Are We Actually Decoding? Source Attribution for Non-Invasive Brain-to-Language Retrieval
- **分类: cs.LG; cs.CL; q-bio.NC**

- **简介: 该论文属于脑机语言解码任务，解决性能评估中源归属不清的问题。通过审计框架分离性能来源，提出GCB方法提升解码准确性。**

- **链接: [https://arxiv.org/pdf/2605.24524](https://arxiv.org/pdf/2605.24524)**

> **作者:** Xinyu Zhang; Sichao Liu; Runhao Lu; Alexandra Woolgar; Lihui Wang
>
> **备注:** 35 pages, 7 figures, 25 tables
>
> **摘要:** In non-invasive neural language decoding, results can be inflated by sources that are not stimulus-evoked neural evidence: decoder priors, embedding-based metrics, and non-neural structural nuisances such as signal duration. The methodological challenge is therefore attribution: a reported gain is more informative when it can be traced to a specific source. We recast stimulus-locked MEG-to-audio retrieval as an auditing framework that separates apparent performance into three sources - structural shortcuts, window-level stimulus-locked evidence, and cross-window contextual aggregation - and provides a diagnostic for each. Signal-blind Gaussian noise reaches 66.3% Rank@1 (R@1) under variable-length decoding but collapses to near chance once fixed-duration windows and stimulus-identity splits are enforced, isolating structural leakage. Under these controls, fixed-window retrieval recovers measurable MEG-audio discriminability, while an oracle sentence-bucket diagnostic shows that 95.7% of Top-1 errors select the wrong sentence, localising the residual bottleneck to sentence-level competition. We audit this contextual source with Group Context Bias (GCB), an inference-time additive logit bias that pools sentence-consistent evidence across windows while leaving the base retrieval scores and candidate pool fixed. Used as a score-space intervention, GCB makes the contextual source measurable: R@1 shifts from 44% to 52% on Gwilliams and from 22% to 29% on MOUS under the same fixed setting. GCB is auditable under this design: its effect collapses under random-grouping perturbations and vanishes when local evidence is attenuated in MEG or is near chance in EEG, supporting its use as a controlled source-attribution intervention. These results suggest that brain-to-language performance should be source-attributed, not merely reported.
>
---
#### [new 224] MAGIC: Multimodal Alignment & Grounding-aware Instruction Coreset for Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MAGIC方法，用于视觉-语言模型的指令微调，解决数据冗余与覆盖不均问题，通过多模态信号选择有效样本，提升训练效率和性能。**

- **链接: [https://arxiv.org/pdf/2605.26004](https://arxiv.org/pdf/2605.26004)**

> **作者:** Shristi Das Biswas; Kaushik Roy
>
> **摘要:** Instruction tuning of large vision-language models (LVLMs) increasingly depends on massive multimodal corpora, yet these datasets contain samples with substantial redundancy, low visual dependency, and highly imbalanced coverage of multimodal reasoning behaviors. As a result, uniform subsampling or naive score-based selection often yields suboptimal training subsets. We introduce MAGIC, a training-free, forward-only coreset selection method designed to construct compact yet behaviorally faithful subsets for multimodal instruction tuning. MAGIC is built on three intrinsic signals extracted from a pretrained VLM: Multimodal Gain, which measures the likelihood improvement obtained from visual input; Bridging Relevance, which captures the sharpness of answer-token grounding over visual tokens; and Skill-Neuron Signatures, which characterize the functional computation elicited by each sample via top-activated feed-forward neurons. MAGIC combines these signals in a three-stage pipeline: filtering low-gain examples, ranking candidates by a normalized quality objective, and performing bucket-wise budget allocation over discrete neuron signatures to preserve latent multimodal skill coverage. This formulation avoids backpropagation, auxiliary selector training, and expensive clustering in continuous activation spaces, while remaining efficient and easily deployable in existing VLMs. Across LLaVA-665K and Vision-Flan datasets, and transfer settings to large target models, LLaVA-1.5-7B and -13B, MAGIC consistently improves over strong baselines under matched 20% budgets: it achieves 100.3% relative performance to full finetuning on LLaVA-665K and 101.6% relative performance on Vision-Flan-186K, while yielding a 73.7% reduction in wall-clock run time.
>
---
#### [new 225] When Self-Belief Misleads: Active Label Acquisition for Reinforcement Learning with Verifiable Rewards
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，解决RLVR中标签获取成本高且易崩溃的问题。提出RLAVR，通过主动选择样本获取真实标签，结合伪标签提升训练稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2605.25864](https://arxiv.org/pdf/2605.25864)**

> **作者:** Li Wang; Xiaodong Lu; Xiaohan Wang; Yikun Ban; Jiajun Chai; Wei Lin; Tianhao Peng; Guojun Yin
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable advancements in reasoning capabilities empowered by Reinforcement Learning with Verifiable Rewards (RLVR). Nonetheless, RLVR intrinsically relies on ground-truth labels for reward computation, the acquisition of which is often prohibitively expensive in real-world scenarios. While unsupervised RLVR paradigms attempt to circumvent this by training on pseudo-labels, they are notoriously susceptible to training collapse. Moreover, different samples often exhibit varying annotation values. In this paper, we propose Reinforcement Learning with Active Verifiable Rewards (RLAVR), which actively acquires ground-truth labels for a small set of selected samples and integrates them with pseudo-labels, thereby stabilizing training dynamics and improving performance under limited annotation budgets. To identify valuable samples, we propose the Corrective Advantage Gap (CAG) metric and analyze the sample-level supervision value. Building on this, we introduce Correction-Aware Reliability Estimation for RLAVR (CARE), which translates the oracle CAG criterion into a practical pre-query acquisition policy to substantially improve training stability. Extensive experiments across diverse domains, model families, and model scales demonstrate the effectiveness and generality of our approach. Our code is available at this https URL.
>
---
## 更新

#### [replaced 001] NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于神经网络机器翻译任务，旨在解决含新词的句子翻译问题。通过构建数据集和搜索工具，结合强化学习提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2601.03790](https://arxiv.org/pdf/2601.03790)**

> **作者:** Zhongtao Miao; Kaiyan Zhao; Masaaki Nagata; Yoshimasa Tsuruoka
>
> **备注:** ACL 2026 Main. Fixed minor typos
>
> **摘要:** Neologism-aware machine translation aims to translate source sentences containing neologisms into target languages. This field remains underexplored compared with general machine translation (MT). In this paper, we propose an agentic framework, NeoAMT, for neologism-aware machine translation equipped with a Wiktionary-based search toolkit. Specifically, we first construct a dedicated dataset for neologism-aware machine translation and build a search toolkit grounded in Wiktionary. The dataset covers 16 languages and 75 translation directions in total, derived from approximately 10 million records of an English Wiktionary dump. The retrieval corpus of the search toolkit is also constructed from around 3 million cleaned records of the same dump. We then leverage the dataset and toolkit to train a translation agent via reinforcement learning (RL) and to evaluate the accuracy of neologism-aware machine translation. Furthermore, we propose an RL training framework featuring a novel reward design and an adaptive rollout generation strategy that exploits translation difficulty to further improve the translation quality of translation agents using our search toolkit.
>
---
#### [replaced 002] OASES: Outcome-Aligned Search-Evaluation Co-Training for Agentic Search
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于智能搜索任务，解决搜索过程中奖励稀疏与监督不可靠的问题。提出OASES框架，通过结果对齐的评估和联合训练提升搜索效果。**

- **链接: [https://arxiv.org/pdf/2604.03675](https://arxiv.org/pdf/2604.03675)**

> **作者:** Erhan Zhang; Yiqun Chen; Zechun Niu; Wei Yang; Xiaochi Wei; Yan Gao; Yi Wu; Yao Hu; Jiaxin Mao
>
> **摘要:** Agentic search enables language models to solve knowledge-intensive tasks by adaptively acquiring external evidence over multiple steps. Reinforcement learning with verifiable rewards (RLVR) has emerged as a widely adopted training paradigm for search agents, yet outcome-only rewards are sparse and provide limited credit assignment for intermediate search actions. Existing process-reward methods therefore seek to densify supervision through proxy signals, external evaluators, or likelihood-based information gain. However, proxy rewards can deviate from the final outcome objective, while fixed evaluators can become stale as the search policy evolves, leading to unreliable process supervision. To address these challenges, we propose OASES, an Outcome-Aligned Search-Evaluation Supervision framework for agentic search. OASES derives outcome-aligned process rewards by evaluating how well each intermediate search state supports answering the original question. It further co-trains the search policy and the state evaluator on policy, allowing the evaluator to adapt to evolving search behavior and provide more reliable process rewards. Experiments on five multi-hop QA benchmarks show that OASES consistently outperforms strong RL baselines, with further analyses confirming the benefits of outcome-aligned process rewards and search-evaluation co-training.
>
---
#### [replaced 003] Act or Clarify? Modeling Sensitivity to Uncertainty and Cost in Communication
- **分类: cs.CL**

- **简介: 该论文属于认知科学与人工智能交叉任务，研究在不确定性下决策时是否寻求澄清。通过构建基于预期后悔的模型，分析澄清行为与风险的关系，并通过实验验证。**

- **链接: [https://arxiv.org/pdf/2602.02843](https://arxiv.org/pdf/2602.02843)**

> **作者:** Polina Tsvilodub; Karl Mulligan; Todd Snider; Robert D. Hawkins; Michael Franke
>
> **备注:** 6 pages, 3 figures, accepted to CogSci 2026
>
> **摘要:** When deciding how to act under uncertainty, agents may choose to act to reduce uncertainty or they may act despite that uncertainty. In communicative settings, an important way of reducing uncertainty is by asking clarification questions (CQs). We predict that the decision to ask a CQ depends on both contextual uncertainty and the cost of alternative actions, and that these factors interact: uncertainty should matter most when acting incorrectly is costly. We formalize this interaction in a computational model based on expected regret: how much an agent stands to lose by acting now rather than with full information. We test these predictions in two experiments, one examining purely linguistic responses to questions and another extending to choices between clarification and non-linguistic action. Taken together, our results suggest a rational tradeoff: humans tend to seek clarification proportional to the risk of substantial loss when acting under uncertainty.
>
---
#### [replaced 004] sciwrite-lint: Verification Infrastructure for the Age of Science Vibe-Writing
- **分类: cs.DL; cs.CL; cs.SE**

- **简介: 该论文提出sciwrite-lint，用于验证科学论文引用的可靠性，解决引用错误和虚假引用问题。通过自动化工具检查引用存在性、准确性及支持度，提升论文质量与可信度。**

- **链接: [https://arxiv.org/pdf/2604.08501](https://arxiv.org/pdf/2604.08501)**

> **作者:** Sergey V Samsonau
>
> **备注:** Code: this https URL
>
> **摘要:** Scientific papers make claims about prior work backed by citations. Verifying those citations at scale (that each cited paper exists, says what the citation claims, and is itself reliable) is structurally beyond what human review can deliver: a typical paper has dozens of citations, and a careful reviewer reads at most a handful end-to-end. AI-assisted writing makes this gap even more urgent: LLMs hallucinate references and may fill in plausible details from titles or abstracts of papers they never read, worse for the smaller local-weights models that privacy-aware researchers must use. sciwrite-lint applies the linting paradigm from software engineering to citation verification: it runs entirely on the researcher's machine (free public databases, a single consumer GPU, and open-weights models), is fast enough to re-lint between revisions so authors catch problems at the source while drafting, and serves journals and reviewers as an automated first pass. The pipeline checks reference existence, metadata accuracy, retraction status, and claim support, traverses one level into cited papers' bibliographies, and produces per-reference reliability scores. We evaluate on 30 unseen papers (arXiv and bioRxiv) with error injection and LLM-adjudicated false-positive analysis. The same linting workflow extends to internal consistency: numbers in text vs. tables, abstract vs. body, figure captions vs. content, statistical results vs. their verbal interpretation, plus structural cross-references (dangling cites, orphan references). As a separate experimental contribution we also propose SciLint Score: citation-chain integrity combined with a contribution component operationalizing five philosophy-of-science frameworks (Popper, Lakatos, Kitcher, Laudan, Mayo).
>
---
#### [replaced 005] From Knowledge to Inference: Formalizing Specialized Public Health Reasoning on GlobalHealthAtlas
- **分类: cs.CL**

- **简介: 该论文属于公共健康推理任务，旨在解决机器学习在该领域缺乏结构化数据和评估标准的问题。研究构建了多语言数据集GlobalHealthAtlas，并提出模型辅助的高质量数据构建与评估方法。**

- **链接: [https://arxiv.org/pdf/2602.00491](https://arxiv.org/pdf/2602.00491)**

> **作者:** Zhaokun Yan; Shan Xu; Wuzheng Dong; Zhaohan Liu; Lijie Feng; Chengxiao Dai; Chen Tianqi; Binfan Liu; Yunpu Ma; Wenting Wei; Yingting Li; Yi Zhang; Tongning Wu
>
> **摘要:** Public health reasoning requires population level inference grounded in scientific evidence, expert consensus, and safety constraints. However, it remains underexplored as a structured machine learning problem with limited supervised signals and benchmarks. We introduce GlobalHealthAtlas, a large scale multilingual dataset of 280,210 instances spanning 15 public health domains and 17 languages. We further propose a large language model (LLM) assisted construction and quality control pipeline with retrieval, deduplication, evidence grounding checks, and label validation to improve consistency at scale. Finally, we present a domain aligned evaluator distilled from high confidence judgments of diverse LLMs to assess outputs along six dimensions: Accuracy, Reasoning, Completeness, Consensus Alignment, Terminology Norms, and Insightfulness. Together, these contributions enable reproducible training and evaluation of LLMs for safety critical public health reasoning beyond conventional QA benchmarks. We publicly release project codebase, evaluator, and model at:: this https URL, this https URL and this https URL
>
---
#### [replaced 006] Feature Resemblance: Towards a Theoretical Understanding of Analogical Reasoning in Transformers
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究Transformer模型中的类比推理机制，解决如何通过学习表示实现属性迁移的问题。通过理论分析和实验，揭示了类比推理的统一机制。**

- **链接: [https://arxiv.org/pdf/2603.05143](https://arxiv.org/pdf/2603.05143)**

> **作者:** Ruichen Xu; Wenjing Yan; Ying-Jun Angela Zhang
>
> **摘要:** Understanding reasoning in large language models is complicated by evaluations that conflate multiple reasoning types. We isolate analogical reasoning, where a model transfers an attribute between entities that share known properties, and study when such transfer can emerge from training. To make the problem analytically tractable, we study a minimal transformer-style abstraction that isolates how learned representations support analogical reasoning. Within this setting, we prove three key results. First, joint training on similarity and attribution premises enables analogical reasoning through aligned representations. Second, sequential training succeeds only when similarity structure is learned before specific attributes, revealing a curriculum asymmetry. Third, in our stylized setting, two-hop reasoning $(a \to b, b \to c \Rightarrow a \to c)$ can be viewed as analogical reasoning with identity bridges $(b=b)$, which appear explicitly in training data. Together, these results reveal a unified mechanism: entities with shared properties become aligned in representation space, enabling property transfer through feature resemblance. Experiments with architectures up to 8B parameters show qualitative agreement with the theory and suggest that representational geometry plays an important role in analogical reasoning beyond the stylized model.
>
---
#### [replaced 007] Optimizing Token Choice for Code Watermarking: An RL Approach
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文属于代码水印任务，旨在保护LLM生成代码的知识产权。通过强化学习优化令牌选择，确保水印可检测且不影响代码功能。**

- **链接: [https://arxiv.org/pdf/2508.11925](https://arxiv.org/pdf/2508.11925)**

> **作者:** Zhimeng Guo; Huaisheng Zhu; Siyuan Xu; Hangfan Zhang; Teng Xiao; Minhao Cheng
>
> **备注:** ICML 2026, 18 pages, 3 figures
>
> **摘要:** Protecting intellectual property on LLM-generated code necessitates effective watermarking systems that can operate within code's highly structured, syntactically constrained nature. In this work, we introduce CodeTracer, an innovative adaptive code watermarking framework underpinned by a novel reinforcement learning training paradigm. At its core, CodeTracer features a policy-driven approach that utilizes a parameterized model to intelligently bias token choices during next-token prediction. This strategy ensures that embedded watermarks maintain code functionality while exhibiting subtle yet statistically detectable deviations from typical token distributions. To facilitate policy learning, we devise a comprehensive reward system that seamlessly integrates execution feedback with watermark embedding signals, balancing process-level and outcome-level rewards. Additionally, we employ Gumbel Top-k reparameterization to enable gradient-based optimization of discrete watermarking decisions. Extensive comparative evaluations demonstrate CodeTracer's significant superiority over state-of-the-art baselines in both watermark detectability and the preservation of generated code's functionality. Our code is available at this https URL.
>
---
#### [replaced 008] Rethinking LLM Ensembling from the Perspective of Mixture Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM集成效率低的问题。通过将集成视为混合模型，提出ME方法，实现更快的集成效果。**

- **链接: [https://arxiv.org/pdf/2605.00419](https://arxiv.org/pdf/2605.00419)**

> **作者:** Jiale Fu; Yuchu Jiang; Peijun Wu; Chonghan Liu; Joey Tianyi Zhou; Xu Yang
>
> **备注:** ICML 2026 Spotlight
>
> **摘要:** Model ensembling is a well-established technique for improving the performance of machine learning models. Conventionally, this involves averaging the output distributions of multiple models and selecting the most probable label. This idea has been naturally extended to large language models (LLMs), yielding improved performance but incurring substantial computational cost. This inefficiency stems from directly applying conventional ensemble implementation to LLMs, which require a separate forward pass for each model to explicitly compute the ensemble distribution. In this paper, we propose the Mixture-model-like Ensemble (ME). By reinterpreting the ensemble as a mixture model, ME stochastically selects a single model at each step to generate the next token, thereby avoiding the need to explicitly compute the full ensemble distribution. ME is mathematically equivalent to sampling from the ensemble distribution, but requires invoking only one model, making it 1.78x-2.68x faster than conventional ensembling. Furthermore, this perspective connects LLM ensembling and token-level routing methods, suggesting that LLM ensembling is a special case of routing methods. Our findings open new avenues for efficient LLM ensembling and motivate further exploration of token-level routing strategies for LLMs. Our code is available at this https URL.
>
---
#### [replaced 009] Frequency Matters: Fast Model-Agnostic Data Curation for Pruning and Quantization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在解决压缩过程中 calibration data 选择问题。通过分析数据固有特性，提出 ZipCal 方法，提升剪枝与量化效果，效率显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2603.16105](https://arxiv.org/pdf/2603.16105)**

> **作者:** Francesco Pio Monaco; Elia Cunegatti; Flavio Vella; Giovanni Iacca
>
> **备注:** Added statistical analysis, mechanistic analysis and a comparison with a generative baseline. 22 pages
>
> **摘要:** Post-training model compression is essential for enhancing the portability of Large Language Models (LLMs) while preserving their performance. While several compression approaches have been proposed, less emphasis has been placed on selecting the most suitable set of data (the so-called \emph{calibration data}) for finding the compressed model configuration. The choice of calibration data is a critical step in preserving model capabilities both intra- and inter-tasks. In this work, we address the challenge of identifying high-performance calibration sets for both pruning and quantization by analyzing intrinsic data properties rather than model-specific signals. We introduce \texttt{\textbf{ZipCal}}, a model-agnostic data curation strategy that maximizes lexical diversity based on Zipfian power laws. Experiments demonstrate that our method consistently outperforms standard uniform random sampling across various pruning benchmarks. Notably, it also performs on par, in terms of downstream performance, with a state-of-the-art method that relies on model perplexity. The latter becomes prohibitively expensive at large-scale models and datasets, while \texttt{\textbf{ZipCal}} is on average $\sim$240$\times$ faster due to its tractable linear complexity\footnote{We make the code and the experiments available at this https URL.}.
>
---
#### [replaced 010] Auditing Stealth Sycophancy in Mental-Health Dialogue: Structured Clinical-State Diagnostics and Clean Matched Benchmarks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于心理健康对话评估任务，旨在解决隐性奉承问题。通过构建基准和DESG框架，检测响应中的临床状态变化，提升有害风险识别效果。**

- **链接: [https://arxiv.org/pdf/2605.03472](https://arxiv.org/pdf/2605.03472)**

> **作者:** Tianze Han; Beining Xu; Hanbo Zhang; Yongming Lu
>
> **摘要:** Mental-health dialogue models are increasingly evaluated by AI-based evaluators, yet these evaluators often treat surface empathy, supportiveness, or fluency as evidence of safety. In this paper, we study a hidden failure mode that we call implicit sycophancy: a response may appear empathetic while implicitly reinforcing catastrophizing, avoidance, hopeless prediction, or CBT-style labeling. To examine this problem, we introduce a diagnostic benchmark for implicit-sycophancy detection, built from three representative mental-health dialogue sources covering everyday peer support, counseling-style emotional support, and crisis-oriented interaction, and further construct a leakage-audited clean single-response matched benchmark with 500 contexts and 1,500 matched response windows. We then propose Dynamic Emotional Signature Graphs (DESG), a structured offline audit framework that separates LLM-based state extraction from final scoring and evaluates clinical direction through semantic, affective, and cognitive-distortion state transitions rather than free-form LLM judgment. Unlike metadata, surface-style, lexical, embedding, and rubric-LLM baselines, DESG scores the direction of clinical-state change induced by a response; on the leakage-audited clean matched benchmark, DESG-StateRisk improves over the strongest non-DESG baseline by 0.0488 macro-F1 and achieves the best harmful-risk detection result. These results suggest that evaluating implicit sycophancy requires explicit clinical-state modeling together with leakage checks, shortcut controls, and competitive baselines.
>
---
#### [replaced 011] Agent-X: Evaluating Deep Multimodal Reasoning in Vision-Centric Agentic Tasks
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Agent-X基准，用于评估视觉主导的智能体多步骤深度推理能力，解决现有基准不足的问题。**

- **链接: [https://arxiv.org/pdf/2505.24876](https://arxiv.org/pdf/2505.24876)**

> **作者:** Tajamul Ashraf; Amal Saqib; Hanan Ghani; Muhra AlMahri; Yuhao Li; Noor Ahsan; Umair Nawaz; Jean Lahoud; Hisham Cholakkal; Mubarak Shah; Philip Torr; Fahad Shahbaz Khan; Rao Muhammad Anwer; Salman Khan
>
> **备注:** Accepted in International Conference of Learning Representations (ICLR 2026)
>
> **摘要:** Deep reasoning is fundamental for solving complex tasks, especially in vision-centric scenarios that demand sequential, multimodal understanding. However, existing benchmarks typically evaluate agents with fully synthetic, single-turn queries, limited visual modalities, and lack a framework to assess reasoning quality over multiple steps as required in real-world settings. To address this, we introduce Agent-X, a large-scale benchmark for evaluating vision-centric agents multi-step and deep reasoning capabilities in real-world, multimodal settings. Agent- X features 828 agentic tasks with authentic visual contexts, including images, multi-image comparisons, videos, and instructional text. These tasks span six major agentic environments: general visual reasoning, web browsing, security and surveillance, autonomous driving, sports, and math reasoning. Our benchmark requires agents to integrate tool use with explicit, stepwise decision-making in these diverse settings. In addition, we propose a fine-grained, step-level evaluation framework that assesses the correctness and logical coherence of each reasoning step and the effectiveness of tool usage throughout the task. Our results reveal that even the best-performing models, including GPT, Gemini, and Qwen families, struggle to solve multi-step vision tasks, achieving less than 50% full-chain success. These findings highlight key bottlenecks in current LMM reasoning and tool-use capabilities and identify future research directions in vision-centric agentic reasoning models. Our data and code are publicly available at this https URL
>
---
#### [replaced 012] MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MMSI-Bench，用于评估多图像空间智能的视觉问答基准。解决现有基准仅限单图关系的问题，通过构建1000个复杂多图问题，评估模型的空间推理能力。**

- **链接: [https://arxiv.org/pdf/2505.23764](https://arxiv.org/pdf/2505.23764)**

> **作者:** Sihan Yang; Runsen Xu; Yiman Xie; Sizhe Yang; Mo Li; Jingli Lin; Chenming Zhu; Xiaochen Chen; Haodong Duan; Xiangyu Yue; Dahua Lin; Tai Wang; Jiangmiao Pang
>
> **备注:** ICLR 2026 Camera ready. 38 pages. Project page: this https URL
>
> **摘要:** Spatial intelligence is essential for multimodal large language models (MLLMs) operating in the complex physical world. Existing benchmarks, however, probe only single-image relations and thus fail to assess the multi-image spatial reasoning that real-world deployments demand. We introduce MMSI-Bench, a VQA benchmark dedicated to multi-image spatial intelligence. Six 3D-vision researchers spent more than 300 hours meticulously crafting 1,000 challenging, unambiguous multiple-choice questions from over 120,000 images, each paired with carefully designed distractors and a stepwise reasoning process. We conduct extensive experiments and evaluate 37 open-source and proprietary MLLMs, observing a wide gap: the strongest open-source model attains roughly 30% accuracy and OpenAI's GPT-5 reasoning model reaches 40%, while humans score 97%. These results underscore the challenging nature of MMSI-Bench and the substantial headroom for future research. Leveraging the annotated reasoning processes, we also provide an automated error analysis pipeline that diagnoses four dominant failure modes, including (1) grounding errors, (2) overlap-matching and scene-reconstruction errors, (3) situation-transformation reasoning errors, and (4) spatial-logic errors, offering insights for advancing spatial intelligence. Project page: this https URL .
>
---
#### [replaced 013] PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出PowerFlow，解决LLMs无监督微调中的分布匹配问题，通过调整分布形状提升逻辑推理和创造性。**

- **链接: [https://arxiv.org/pdf/2603.18363](https://arxiv.org/pdf/2603.18363)**

> **作者:** Ruishuo Chen; Yu Chen; Zhuoran Li; Longbo Huang
>
> **备注:** Camera-ready version accepted at ICML 2026
>
> **摘要:** Unsupervised Reinforcement Learning from Internal Feedback (RLIF) has emerged as a promising paradigm for eliciting the latent capabilities of Large Language Models (LLMs) without external supervision. However, current methods rely on heuristic intrinsic rewards, which often lack a well-defined theoretical optimization target and are prone to degenerative biases. In this work, we introduce PowerFlow, a principled framework that reformulates unsupervised fine-tuning as a distribution matching problem. By casting GFlowNet as an amortized variational sampler for unnormalized densities, we propose a length-aware Trajectory-Balance objective that explicitly neutralizes the structural length biases inherent in autoregressive generation. By targeting $\alpha$-power distributions, PowerFlow enables the directional elicitation of the dual nature of LLMs: sharpening the distribution ($\alpha > 1$) to intensify logical reasoning, or flattening it ($\alpha < 1$) to unlock expressive creativity. Extensive experiments demonstrate that PowerFlow consistently outperforms existing RLIF methods, matching or even exceeding supervised GRPO. Furthermore, by mitigating over-sharpening in aligned models, our approach achieves simultaneous gains in diversity and quality, shifting the Pareto frontier in creative tasks.
>
---
#### [replaced 014] Chain of Evidence: Pixel-Level Visual Attribution for Iterative Retrieval-Augmented Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于视觉问答任务，解决iRAG中文本解析导致的粗粒度归属和视觉语义丢失问题，提出CoE框架实现像素级视觉归因。**

- **链接: [https://arxiv.org/pdf/2605.01284](https://arxiv.org/pdf/2605.01284)**

> **作者:** Peiyang Liu; Ziqiang Cui; Xi Wang; Di Liang; Wei Ye
>
> **摘要:** Iterative Retrieval-Augmented Generation (iRAG) has emerged as a powerful paradigm for answering complex multi-hop questions by progressively retrieving and reasoning over external documents. However, current systems predominantly operate on parsed text, which creates two critical bottlenecks: (1) \textit{Coarse-grained attribution}, where users are burdened with manually locating evidence within lengthy documents based on vague text-level citations; and (2) \textit{Visual semantic loss}, where the conversion of visually rich documents (e.g., slides, PDFs with charts) into text discards spatial logic and layout cues essential for reasoning. To bridge this gap, we present \textbf{Chain of Evidence (CoE)}, a retriever-agnostic visual attribution framework that leverages Vision-Language Models to reason directly over screenshots of retrieved document candidates. CoE eliminates format-specific parsing and outputs precise bounding boxes, visualizing the complete reasoning chain within the retrieved candidate set. We evaluate CoE on two distinct benchmarks: \textbf{Wiki-CoE}, a large-scale dataset of structured web pages derived from 2WikiMultiHopQA, and \textbf{SlideVQA}, a challenging dataset of presentation slides featuring complex diagrams and free-form layouts. Experiments demonstrate that fine-tuned Qwen3-VL-8B-Instruct achieves robust performance, significantly outperforming text-based baselines in scenarios requiring visual layout understanding, while establishing a retriever-agnostic solution for pixel-level interpretable iRAG. Our code is available at this https URL.
>
---
#### [replaced 015] What Questions Should Robots Be Able to Answer? A Dataset of User Questions for Explainable Robotics
- **分类: cs.RO; cs.CL; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决机器人如何回答用户问题的问题。通过收集用户对家用机器人的提问数据，为机器人问答系统提供基准和指导。**

- **链接: [https://arxiv.org/pdf/2510.16435](https://arxiv.org/pdf/2510.16435)**

> **作者:** Lennart Wachowiak; Andrew Coles; Gerard Canal; Oya Celiktutan
>
> **摘要:** With the growing use of large language models and conversational interfaces in human-robot interaction, robots' ability to answer user questions is more important than ever. We therefore introduce a dataset of 1,893 user questions for household robots, collected from 100 participants and organized into 12 categories and 70 subcategories. Most work in explainable robotics focuses on why-questions. In contrast, our dataset provides a wide variety of questions, from questions about simple execution details to questions about how the robot would act in hypothetical scenarios -- thus giving roboticists valuable insights into what questions their robot needs to be able to answer. To collect the dataset, we created 15 video stimuli and 7 text stimuli, depicting robots performing varied household tasks. We then asked participants on Prolific what questions they would want to ask the robot in each portrayed situation. In the final dataset, the most frequent categories are questions about task execution details (21.4%), the robot's capabilities (12.6%), and performance assessments (10.7%). Although questions about how robots would handle potentially difficult scenarios and ensure correct behavior are less frequent, users rank them as the most important for robots to be able to answer. Moreover, we find that users who identify as novices in robotics ask different questions than more experienced users. Novices are more likely to inquire about simple facts, such as what the robot did or the current state of the environment. As robots enter environments shared with humans and language becomes central to giving instructions and interaction, this dataset provides a valuable foundation for (i) identifying the information robots need to log and expose to conversational interfaces, (ii) benchmarking question-answering modules, and (iii) designing explanation strategies that align with user expectations.
>
---
#### [replaced 016] Asking LLMs to Verify First is Almost Free Lunch
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Verification-First（VF）策略，用于提升大语言模型的推理能力，无需高昂训练成本或大量采样。通过先验证候选答案再生成解，有效缩小逻辑搜索空间，提升准确率。**

- **链接: [https://arxiv.org/pdf/2511.21734](https://arxiv.org/pdf/2511.21734)**

> **作者:** Shiguang Wu; Quanming Yao
>
> **摘要:** To enhance the reasoning capabilities of Large Language Models (LLMs) without high costs of training, nor extensive test-time sampling, we introduce Verification-First (VF), a strategy that prompts models to verify a provided candidate answer, even a trivial or random one, before generating a solution. This approach triggers a "reverse reasoning" process complementary to standard forward Chain-of-Thought (CoT), which restricts the logical search space of the answer by pruning the LLM's output distribution. We further generalize VF prompting to Iter-VF, a sequential test-time scaling (TTS) method that iteratively cycles the verification-generation process using the model's previous answer. Extensive experiments across various benchmarks and various LLMs confirm that VF prompting with random answer consistently outperforms standard CoT with minimal computational overhead, and Iter-VF outperforms existing TTS strategies. VF is also effective on SOTA thinking models. For example, by using the simple VF prompting, we obtain a new SOTA 94.9% accuracy on GPQA-Diamond with Gemini-3-Pro-Preview where VF reduces its errors by ~30% relatively.
>
---
#### [replaced 017] Human-1 by Josh Talks: A Full-Duplex Conversational Modeling Framework in Hindi using Real-World Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音对话系统任务，旨在解决印度语言中全双工对话建模问题。通过改进架构和训练数据，构建首个开放的印地语全双工系统。**

- **链接: [https://arxiv.org/pdf/2604.23295](https://arxiv.org/pdf/2604.23295)**

> **作者:** Bhaskar Singh; Shobhit Banga; Mahima Manik; Pranav Sharma
>
> **摘要:** Full-duplex spoken dialogue systems can model natural conversational behaviours such as interruptions, overlaps, and backchannels, yet such systems remain largely unexplored for Indian languages. We present the first open, reproducible full-duplex spoken dialogue system for Hindi by adapting Moshi, a state-of-the-art duplex speech architecture, using a custom Hindi tokeniser and training on 26,000 hours of real spontaneous conversations collected from 14,695 speakers with separate speaker channels, enabling direct learning of turn-taking and overlap patterns from natural interactions. To support Hindi text generation, we replace the original English tokeniser and reinitialise text-vocabulary-dependent parameters while retaining the pre-trained audio components. We propose a two-stage training recipe -- large-scale pre-training followed by fine-tuning on 1,000 hours of conversational data. Evaluation through the prompted dialogue continuation paradigm with both automatic metrics and human judgments demonstrates that the resulting model generates natural and meaningful full-duplex conversational behaviour in Hindi. This work serves as a first step toward real-time duplex spoken dialogue systems for Hindi and other Indian languages.
>
---
#### [replaced 018] CoSPlay: Cooperative Self-Play at Test-Time with Self-Generated Code and Unit Test
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出CoSPlay，解决代码生成中依赖真实单元测试的问题。通过自生成测试用例与代码协同优化，提升代码质量与测试准确性，无需真实测试数据。**

- **链接: [https://arxiv.org/pdf/2605.23491](https://arxiv.org/pdf/2605.23491)**

> **作者:** Zhangyi Hu; Chenhui Liu; Tian Huang; Jindong Li; Yang Yang; Jiemin Wu; Zining Zhong; Menglin Yang; Yutao Yue
>
> **备注:** Code is available at: this https URL | Data & log is available at: this https URL
>
> **摘要:** Recently, Reinforcement Learning with Verifiable Rewards (RLVR) and Test-Time Scaling (TTS) have advanced LLM code generation through executable verification. Yet Ground-Truth Unit Tests (GT UTs) remain a bottleneck: SOTA RLVR methods require them for costly training, while existing TTS methods lose competitiveness without them. This motivates GT-free TTS, where existing methods directly use self-generated UTs to refine and select code candidates. Yet such UTs are often noisy or spuriously coupled with wrong code, and UT quality in turn cannot be validated without reliable code. The key challenge is therefore to jointly improve both. To this end, we present CoSPlay, a GT-free, training-free framework that jointly improves codes and UTs through cooperative self-play. It first explores diverse solution ideas and identifies their potential failure modes to produce discriminative UT ideas. It then uses bidirectional pass-count signals from the Code-UT execution matrix to iteratively prune or fix weak codes and refresh or replace unreliable UTs, letting the two pools co-evolve. Finally, when multiple codes remain tied at the highest pass count, it picks the final code from the largest output-consensus cluster, since correct codes agree on the same inputs while wrong codes diverge. Experiments on four challenging benchmarks show that CoSPlay on Qwen2.5-7B-Instruct improves average BoN from 22.1% to 33.2% and UT accuracy from 14.6% to 78.3%, matching or surpassing the RLVR model CURE-7B. When applied to CURE-7B, it further improves BoN by 5.7%. CoSPlay also generalizes across diverse backbones and outperforms GT-free TTS baselines under comparable token budgets, with continued gains as the budget scales up. These results suggest a scalable inference strategy for competitive code generation without any GT data.
>
---
#### [replaced 019] Judge Circuits
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型评估任务，研究LLM在不同输出格式下的评分不一致问题。通过分析模型内部机制，发现判断信号与格式分离，揭示了格式影响评估结果的根源。**

- **链接: [https://arxiv.org/pdf/2605.16023](https://arxiv.org/pdf/2605.16023)**

> **作者:** Nils Feldhus; Tanja Baeumel; Elena Golimblevskaia; Qianli Wang; Van Bach Nguyen; Aaron Louis Eidt; Selin Kahvecioglu; Christopher Ebert; Wojciech Samek; Jing Yang; Vera Schmitt; Sebastian Möller; Simon Ostermann
>
> **备注:** 39 pages
>
> **摘要:** LLM-as-a-judge has become the dominant paradigm for grading model outputs at scale, yet the same model assigns systematically different scores when its output format changes (e.g., a 1-5 rating vs. a True/False label). Existing diagnoses of these format-induced inconsistencies stop at the input-output level. Using Position-aware Edge Attribution Patching (PEAP), we causally investigate the internal mechanism in Gemma-3, Qwen2.5, and Llama-3. We find that judgments across structured understanding and open-ended preference tasks share a sparse, generalized Latent Evaluator sub-graph in the mid-to-late multi-layer perceptrons (MLPs); zero-ablating it collapses judgment while preserving world knowledge in architecturally modular models. By structurally decoupling abstract judging from output formatting, we provide a mechanistic account of format-induced inconsistency on the open-weight models we study: a continuous judgment signal computed in the shared trunk is mapped through fragile, format-specific terminal branches, enabling format-independent preference to be isolated downstream of the requested output format. Our findings imply that benchmark-level reliability comparisons across formats are partially measuring formatter geometry rather than evaluation quality.
>
---
#### [replaced 020] LiveMCP-101: Stress Testing and Diagnosing MCP-enabled Agents on Challenging Queries
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LiveMCP-101基准，用于评估AI代理在复杂查询中使用MCP工具的能力，解决多步骤任务中的工具协调与动态响应问题。**

- **链接: [https://arxiv.org/pdf/2508.15760](https://arxiv.org/pdf/2508.15760)**

> **作者:** Ming Yin; Dinghan Shen; Silei Xu; Sixun Dong; Mian Zhang; Yebowen Hu; Shujian Liu; Jianbing Han; Simin Ma; Song Wang; Sathish Reddy Indurthi; Xun Wang; Yiran Chen; Kaiqiang Song
>
> **摘要:** Tool calling has emerged as a critical capability for AI agents. In contrast to conventional tool calling frameworks that rely on static, provider-specific tool definitions, the Model Context Protocol (MCP) offers a unified interface to discover and invoke tools dynamically. However, there is a significant gap in benchmarking multi-step tasks using diverse MCP tools in realistic, dynamic scenarios. In this work, we present LiveMCP-101, a benchmark of 101 real-world queries that require coordinated use of multiple MCP tools. To address temporal variability in real-world tool responses, we introduce a parallel evaluation framework where a reference agent executes a validated plan simultaneously to produce real-time reference outputs. Experiments show that even frontier LLMs achieve a success rate below 60\%, highlighting challenges in multi-step tool use. Comprehensive error analysis identifies seven failure modes spanning tool planning, parameterization, and output handling, pointing to concrete directions for improving current models. LiveMCP-101 sets a rigorous standard for evaluating real-world agent capabilities, advancing toward autonomous agent systems that reliably execute complex tasks through MCP tool orchestration.
>
---
#### [replaced 021] PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出PolySAE，解决SAE无法捕捉特征组合结构的问题，通过多项式解码建模特征交互，提升可解释性。**

- **链接: [https://arxiv.org/pdf/2602.01322](https://arxiv.org/pdf/2602.01322)**

> **作者:** Panagiotis Koromilas; Andreas D. Demou; James Oldfield; Yannis Panagakis; Mihalis Nicolaou
>
> **备注:** 43rd International Conference on Machine Learning (ICML 2026); Code: this https URL
>
> **摘要:** Sparse autoencoders (SAEs) interpret neural network representations by decomposing activations into sparse combinations of dictionary atoms. However, SAEs assume features combine additively through linear reconstruction, an assumption that cannot capture compositional structure: linear models cannot distinguish whether ''Starbucks'' arises from the composition of ''star'' and ''coffee'' features or merely their co-occurrence. This forces SAEs to allocate monolithic features for compound concepts rather than decomposing them into interpretable constituents. We introduce PolySAE, which extends the SAE decoder with higher-order terms to model feature interactions while preserving the linear encoder essential for interpretability. Through low-rank tensor factorization on a shared projection subspace, PolySAE captures pairwise and triple feature interactions with small parameter overhead (3% on GPT2). Across four language models and three SAE variants, PolySAE achieves an average improvement of $\sim$8% in probing F1 while maintaining comparable reconstruction error, and produces 2--10$\times$ larger Wasserstein distances between class-conditional feature distributions. Critically, learned interaction weights exhibit negligible correlation with co-occurrence frequency ($r = 0.06$ vs $r = 0.82$ for SAE feature covariance), suggesting that polynomial terms capture compositional structure largely independent of surface statistics. Finally, the learned interaction directions causally steer model outputs toward the corresponding compositional semantics.
>
---
#### [replaced 022] Persuasion Should be Double-Blind: A Multi-Domain Dialogue Dataset With Faithfulness Based on Causal Theory of Mind
- **分类: cs.CL**

- **简介: 该论文提出ToMMA框架，构建CToMPersu数据集，解决对话中角色混淆与信息泄露问题，提升说服对话的 realism 和效果。任务为多领域说服对话生成。**

- **链接: [https://arxiv.org/pdf/2502.21297](https://arxiv.org/pdf/2502.21297)**

> **作者:** Dingyi Zhang; Linhai Zhang; Fanglei Qu; Ziqing Zhuang; Deyu Zhou
>
> **备注:** 6 pages
>
> **摘要:** Persuasive dialogue is central to human communication, yet existing datasets often rely on a single language model generating both roles, producing unrealistic interactions that violate the double-blind nature of persuasion. To overcome this, we propose ToMMA, a multi-agent framework guided by causal Theory of Mind that enforces role separation and prevents information leakage. Using ToMMA, we build CToMPersu, a large-scale multi-turn, multi-domain dataset capturing realistic persuasion dynamics. Automatic evaluations show that CToMPersu produces more coherent and persuasive dialogues than prior datasets. Furthermore, when used as a knowledge base, CToMPersu significantly enhances the persuasive performance of large language models, as confirmed by both automatic and human evaluations.
>
---
#### [replaced 023] InfiFPO: Implicit Model Fusion via Preference Optimization in Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型融合任务，旨在解决现有方法在偏好对齐阶段信息丢失的问题。提出InfiFPO，通过融合多模型概率提升性能。**

- **链接: [https://arxiv.org/pdf/2505.13878](https://arxiv.org/pdf/2505.13878)**

> **作者:** Yanggan Gu; Yuanyi Wang; Zhaoyi Yan; Yiming Zhang; Qi Zhou; Fei Wu; Hongxia Yang
>
> **摘要:** Model fusion combines multiple Large Language Models (LLMs) with different strengths into a more powerful, integrated model through lightweight training methods. Existing works on model fusion focus primarily on supervised fine-tuning (SFT), leaving preference alignment (PA) --a critical phase for enhancing LLM performance--largely unexplored. The current few fusion methods on PA phase, like WRPO, simplify the process by utilizing only response outputs from source models while discarding their probability information. To address this limitation, we propose InfiFPO, a preference optimization method for implicit model fusion. InfiFPO replaces the reference model in Direct Preference Optimization (DPO) with a fused source model that synthesizes multi-source probabilities at the sequence level, circumventing complex vocabulary alignment challenges in previous works and meanwhile maintaining the probability information. By introducing probability clipping and max-margin fusion strategies, InfiFPO enables the pivot model to align with human preferences while effectively distilling knowledge from source models. Comprehensive experiments on 11 widely-used benchmarks demonstrate that InfiFPO consistently outperforms existing model fusion and preference optimization methods. When using Phi-4 as the pivot model, InfiFPO improve its average performance from 79.95 to 83.33 on 11 benchmarks, significantly improving its capabilities in mathematics, coding, and reasoning tasks.
>
---
#### [replaced 024] ECG-R1: Protocol-Guided and Modality-Agnostic MLLM for Reliable ECG Interpretation
- **分类: cs.CL**

- **简介: 该论文提出ECG-R1，解决ECG解读中MLLM可靠性不足的问题，通过协议引导数据、模态解耦架构和强化学习提升准确性。**

- **链接: [https://arxiv.org/pdf/2602.04279](https://arxiv.org/pdf/2602.04279)**

> **作者:** Jiarui Jin; Haoyu Wang; Xingliang Wu; Xiaocheng Fang; Xiang Lan; Zihan Wang; Deyun Zhang; Bo Liu; Yingying Zhang; Xian Wu; Hongyan Li; Shenda Hong
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Electrocardiography (ECG) serves as an indispensable diagnostic tool in clinical practice, yet existing multimodal large language models (MLLMs) remain unreliable for ECG interpretation, often producing plausible but clinically incorrect analyses. To address this, we propose ECG-R1, the first reasoning ECG MLLM designed for reliable ECG interpretation via three innovations. First, we construct the interpretation corpus using \textit{Protocol-Guided Instruction Data Generation}, grounding interpretation in measurable ECG features and monograph-defined quantitative thresholds and diagnostic logic. Second, we present a modality-decoupled architecture with \textit{Interleaved Modality Dropout} to improve robustness and cross-modal consistency when either the ECG signal or ECG image is missing. Third, we present \textit{Reinforcement Learning with ECG Diagnostic Evidence Rewards} to strengthen evidence-grounded ECG interpretation. Additionally, we systematically evaluate the ECG interpretation capabilities of proprietary, open-source, and medical MLLMs, and provide the first quantitative evidence that severe hallucinations are widespread, suggesting that the public should not directly trust these outputs without independent verification. Code is available at \href{this https URL}{here}.
>
---
#### [replaced 025] Coupled Variational Reinforcement Learning for Language Model General Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型推理任务，解决验证奖励依赖问题。通过耦合变分强化学习，提升推理效率与答案一致性。**

- **链接: [https://arxiv.org/pdf/2512.12576](https://arxiv.org/pdf/2512.12576)**

> **作者:** Xueru Wen; Jie Lou; Yanjiang Liu; Hongyu Lin; Ben He; Xianpei Han; Le Sun; Yaojie Lu; Debing Zhang
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** While reinforcement learning has achieved impressive progress in language model reasoning, it is constrained by the requirement for verifiable rewards. Recent verifier-free RL methods address this limitation by utilizing the probabilities that LLMs generate reference answers as reward signals. However, these approaches typically sample reasoning traces conditioned only on the question. This design decouples reasoning-trace sampling from answer information, leading to inefficient exploration and incoherence between traces and final answers. In this paper, we propose \textit{\b{Co}upled \b{V}ariational \b{R}einforcement \b{L}earning} (CoVRL), which bridges variational inference and reinforcement learning by coupling prior and posterior distributions through a hybrid sampling strategy. By constructing and optimizing a composite distribution that integrates these two distributions, CoVRL enables efficient exploration while preserving strong thought-answer coherence. Extensive experiments on mathematical and general reasoning benchmarks show that CoVRL improves performance by 12.4\% over the base model and achieves an additional 2.3\% improvement over state-of-the-art verifier-free RL baselines, providing a principled framework for enhancing the general reasoning capabilities of language models.
>
---
#### [replaced 026] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity
- **分类: cs.CL**

- **简介: 该论文提出CArtBench，用于评估视觉-语言模型在中文艺术理解、解读和真伪判断上的能力。解决模型在艺术领域推理与鉴别能力不足的问题，通过四个子任务进行评测。**

- **链接: [https://arxiv.org/pdf/2604.11632](https://arxiv.org/pdf/2604.11632)**

> **作者:** Xuefeng Wei; Zhixuan Wang; Xuan Zhou; Zhi Qu; Hongyao Li; Yusuke Sakai; Hidetaka Kamigaito; Taro Watanabe
>
> **备注:** under review
>
> **摘要:** We introduce CARTBENCH, a museum-grounded benchmark for evaluating vision-language models (VLMs) on Chinese artworks beyond short-form recognition and QA. CARTBENCH comprises four subtasks: CURATORQA for evidence-grounded recognition and reasoning, CATALOGCAPTION for structured four-section expert-style appreciation, REINTERPRET for defensible reinterpretation with expert ratings, and CONNOISSEURPAIRS for diagnostic authenticity discrimination under visually similar confounds. CARTBENCH is built by aligning image-bearing Palace Museum objects from Wikidata with authoritative catalog pages, spanning five art categories across multiple dynasties. Across nine representative VLMs, we find that high overall CURATORQA accuracy can mask sharp drops on hard evidence linking and style-to-period inference; long-form appreciation remains far from expert references; and authenticity-oriented diagnostic discrimination stays near chance, underscoring the difficulty of connoisseur-level reasoning for current models.
>
---
#### [replaced 027] The meaning of prompts and the prompts of meaning: Semiotic reflections and modelling
- **分类: cs.CL**

- **简介: 该论文属于理论研究任务，旨在解析大语言模型中的提示机制，通过符号学视角重新定义提示为意义生成过程，解决知识组织与信息检索的理论基础问题。**

- **链接: [https://arxiv.org/pdf/2509.14250](https://arxiv.org/pdf/2509.14250)**

> **作者:** Martin Thellefsen; Amalia Nurma Dewi; Bent Sorensen
>
> **备注:** 18 pages, 2 figures
>
> **摘要:** This paper explores prompts and prompting in large language models (LLMs) as dynamic semiotic phenomena, drawing on Peirce's triadic model of signs, his nine sign types, and the Dynacom model of communication. The aim is to reconceptualize prompting not as a technical input mechanism but as a communicative and epistemic act involving an iterative process of sign formation, interpretation, and refinement. The theoretical foundation rests on Peirce's semiotics, particularly the interplay between representamen, object, and interpretant, and the typological richness of signs: qualisign, sinsign, legisign; icon, index, symbol; rheme, dicent, argument - alongside the interpretant triad captured in the Dynacom model. Analytically, the paper positions the LLM as a semiotic resource that generates interpretants in response to user prompts, thereby participating in meaning-making within shared universes of discourse. The findings suggest that prompting is a semiotic and communicative process that redefines how knowledge is organized, searched, interpreted, and co-constructed in digital environments. This perspective invites a reimagining of the theoretical and methodological foundations of knowledge organization and information seeking in the age of computational semiosis
>
---
#### [replaced 028] Fine-Tuning Causal LLMs for Text Classification: Embedding-Based vs. Instruction-Based Approaches
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究在资源受限下如何微调因果大语言模型进行文本分类，比较了基于嵌入和指令微调的方法，提出结合量化与LoRA的高效训练策略。**

- **链接: [https://arxiv.org/pdf/2512.12677](https://arxiv.org/pdf/2512.12677)**

> **作者:** Amirhossein Yousefiramandi; Ciaran Cooney
>
> **备注:** 20 pages, 5 figures
>
> **摘要:** We explore efficient strategies to fine-tune decoder-only Large Language Models (LLMs) for downstream text classification under resource constraints. Two approaches are investigated: (1) attaching a classification head to a pretrained causal LLM and fine-tuning it on the task, using the LLM's final-token embedding as a sequence representation, and (2) instruction-tuning the LLM in a prompt-to-response format for classification. To enable single-GPU fine-tuning of models up to 8B parameters, we combine 4-bit model quantization with Low-Rank Adaptation (LoRA) for parameter-efficient training. Experiments on two patent benchmarks, a 5-class single-label internal corpus and the public WIPO-Alpha multi-label dataset with 14 categories, show that the embedding-head approach matches or exceeds fine-tuned BERT baselines on single-label classification while training 10-30x fewer parameters. Instruction-tuning is competitive only in the multi-label regime, and only with substantially larger trainable budgets of at least 100M parameters. These results demonstrate that directly leveraging the internal representations of causal LLMs, together with efficient fine-tuning techniques, yields strong classification performance under limited computational resources. We discuss the advantages of each approach and outline practical guidelines and future directions for optimizing LLM fine-tuning in classification scenarios.
>
---
#### [replaced 029] TimeSpot: Benchmarking Geo-Temporal Understanding in Vision-Language Models in Real-World Settings
- **分类: cs.CV; cs.CL; cs.ET; cs.MM; cs.RO**

- **简介: 该论文提出TimeSpot基准，用于评估视觉语言模型在真实场景中的时空理解能力。解决VLM在时间与空间推理上的不足，通过图像预测地理和时间属性及进行时空推理任务。**

- **链接: [https://arxiv.org/pdf/2603.06687](https://arxiv.org/pdf/2603.06687)**

> **作者:** Azmine Toushik Wasi; Shahriyar Zaman Ridoy; Koushik Ahamed Tonmoy; Kinga Tshering; S. M. Muhtasimul Hasan; Wahid Faisal; Tasnim Mohiuddin; Md Rizwan Parvez
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Geo-temporal understanding, the ability to infer location, time, and contextual properties from visual input alone, underpins applications such as disaster management, traffic planning, embodied navigation, world modeling, and geography education. Although recent vision-language models (VLMs) have advanced image geo-localization using cues like landmarks and road signs, their ability to reason about temporal signals and physically grounded spatial cues remains limited. To address this gap, we introduce TimeSpot, a benchmark for evaluating real-world geo-temporal reasoning in VLMs. TimeSpot comprises 1,455 ground-level images from 80 countries and requires structured prediction of temporal attributes (season, month, time of day, daylight phase) and geographic attributes (continent, country, climate zone, environment type, latitude-longitude) directly from visual evidence. It also includes spatial-temporal reasoning tasks that test physical plausibility under real-world uncertainty. Evaluations of state-of-the-art open- and closed-source VLMs show low performance, particularly for temporal inference. While supervised fine-tuning yields improvements, results remain insufficient, highlighting the need for new methods to achieve robust, physically grounded geo-temporal understanding TimeSpot is available at: this https URL.
>
---
#### [replaced 030] When Symptoms Are Not Enough: Evidence-Weighting Patterns in Large Language Model Psychiatric Screening
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于心理疾病筛查任务，旨在评估大语言模型在精神健康诊断中的可靠性。研究通过构建基准数据集，分析模型对症状、功能损害和保护性情境的证据权重，揭示其误判原因。**

- **链接: [https://arxiv.org/pdf/2605.23148](https://arxiv.org/pdf/2605.23148)**

> **作者:** Jianfeng Zhu; Megan Korhummel; Ruoming Jin; Karin G. Coifman
>
> **备注:** 25 pages 7 figures
>
> **摘要:** As demand for mental health care outpaces clinician-delivered assessment, scalable screening tools are increasingly needed. Large language models (LLMs) may identify psychiatric risk from patient narratives, but their reliability across diagnoses, demographic subgroups, and evidence-use patterns remains uncertain. We introduce a SCID-anchored benchmark of 555 semi-structured experiential interviews paired with diagnostic reference labels for anxiety disorder, major depressive disorder, post-traumatic stress disorder, and any current mental health disorder. Using zero-shot task-specific prompting, we evaluated five state-of-the-art LLMs and examined whether false-negative errors reflected missed psychiatric evidence or differential weighting of symptom, functional-impairment, and protective-context cues. Performance varied across tasks and models, with accuracy ranging from 0.49 to 0.86 and Matthews correlation coefficients from 0.16 to 0.38. GPT-4.1 Mini and GPT-5 Mini showed the most consistent disorder-specific accuracy. Subgroup analyses found higher depression-classification accuracy among male than female participants, no consistent age-related pattern, and modest non-uniform variation across race strata. Evidence-integration analyses showed that false-negative anxiety and PTSD classifications often contained explicit symptom evidence but were accompanied by preserved functioning, coping ability, or social support. Functional-impairment evidence shifted model outputs toward positive classifications, whereas protective-context evidence shifted outputs away. These findings suggest that LLMs may support scalable psychiatric screening, but their tendency to discount symptom evidence in the presence of preserved functioning or protective context requires careful validation before clinical deployment.
>
---
#### [replaced 031] Reading, Not Thinking: Understanding and Bridging the Modality Gap When Text Becomes Pixels in Multimodal LLMs
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究多模态大语言模型中文本转图像后的性能差异问题，分析模态差距原因并提出改进方法。**

- **链接: [https://arxiv.org/pdf/2603.09095](https://arxiv.org/pdf/2603.09095)**

> **作者:** Kaiser Sun; Xiaochuang Yuan; Hongjun Liu; Chen Zhao; Cheng Zhang; Mark Dredze; Fan Bai
>
> **摘要:** Multimodal large language models (MLLMs) can process text presented as images, yet they often perform worse than when the same content is provided as textual tokens. We systematically diagnose this "modality gap" by evaluating seven MLLMs across seven benchmarks in five input modes, spanning both synthetically rendered text and realistic document images from arXiv PDFs to Wikipedia pages. We find that the gap is highly sensitive to rendering choices such as font and resolution, and that natural document images often exhibit much smaller gaps, suggesting the performance difference partly reflects evaluation artifacts rather than fundamental limitations. Through a grounded-theory error analysis of over 4,000 examples, we identify the primary cause: image input alone suppresses reasoning effort, with models producing 5--19x shorter outputs that skip step-by-step computation or reasoning. The reluctance to reason, not a failure of perception or knowledge retrieval, drives the performance gap, particularly on tasks requiring multi-step reasoning. We show that a simple, lightweight on-policy self-distillation method by fine-tuning models on their own text-mode reasoning traces paired with image inputs closes this gap, raising image-mode accuracy to match or exceed text-mode performance with over 50\% improvement, and the gains transfer to unseen benchmarks without catastrophic forgetting. Overall, our results and analyses provide a systematic understanding of the modality gap and suggest a practical path toward improving visual text understanding in multimodal language models.
>
---
#### [replaced 032] Transformers over-extend what humans underlearn: the case of Spanish L-shaped morphome
- **分类: cs.CL**

- **简介: 该论文研究语言学习中的形态模式，探讨神经网络与人类对西语不规则形态的泛化差异。任务为语言模式学习，解决模型与人类在泛化方式上的不同问题，通过实验对比模型与人类行为。**

- **链接: [https://arxiv.org/pdf/2507.21556](https://arxiv.org/pdf/2507.21556)**

> **作者:** Akhilesh Kakolu Ramarao; Kevin Tang; Dinah Baer-Henney
>
> **摘要:** The cognitive reality of irregular morphological patterns has been debated for decades: do speakers extend them to novel forms, or are they lexical artifacts? A neural network trained on distributional input offers a learnability test: if it recovers the pattern, the pattern is learnable from input statistics alone. We apply this test to the Spanish L-shaped morphome, where the first-person singular indicative stem appears in every present subjunctive cell despite lacking apparent phonological or semantic motivation. We further ask whether the frequency of irregular verbs in the input modulates generalization, evaluating transformers under three frequency conditions (10%, 50%, 90% irregular) and comparing them to human behavioral data. On full-form production from pseudoword inputs all models performed poorly, but all three conditions produced the correct stem more often than humans (43--49% vs. 33%). Response preferences revealed a clear divergence: humans consistently favored regular inflections, whereas models preferred irregular forms more as their proportion in training grew. Models in the naturalistic and balanced conditions were also sensitive to phonological similarity between pseudowords and real Spanish irregular verbs, an effect absent in humans. The L-shaped morphome is thus learnable from distributional input alone, but models generalize it qualitatively differently from humans.
>
---
#### [replaced 033] Copy-as-Decode: Grammar-Constrained Parallel Prefill for LLM Editing
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Copy-as-Decode方法，用于提升LLM编辑效率。通过结构化解码和并行预填充，减少冗余生成，提高速度与覆盖率。属于LLM编辑任务，解决生成效率与准确性问题。**

- **链接: [https://arxiv.org/pdf/2604.18170](https://arxiv.org/pdf/2604.18170)**

> **作者:** Ziyang Liu
>
> **备注:** The authors have decided to withdraw this version following internal review regarding authorship and contribution agreements
>
> **摘要:** LLMs edit text and code by autoregressively regenerating the full output, even when most tokens appear verbatim in the input. We study Copy-as-Decode, a decoding-layer mechanism that recasts edit generation as structured decoding over a two-primitive grammar: <copy lines="i-j"/> references an input line range, <gen>...</gen> emits new content. A token-level FSM guarantees syntactic validity, and a serving-layer primitive updates the KV cache for each copy span via a single parallel-prefill forward rather than $N$ autoregressive steps -- sharing the parallel-forward kernel of speculative decoding but with input tokens as the draft and program-enforced acceptance replacing probabilistic verification. We report an upper-bound analysis that requires no end-to-end training. (i) Kernel speedup: on Qwen2.5-{1.5B, 7B}, copying $N$ tokens via parallel prefill is $6.8\times$--$303\times$ faster than autoregressive ($N \in [8, 512]$, A100 80GB bf16). (ii) Copy ceiling: on ProbeEdit and HumanEvalPack-Fix (Py/JS), $74$--$98\%$ of gold tokens are reachable under the line-level primitive; composed with the empirical kernel over each corpus's span histogram this yields a closed-form wall-clock bound of $29.0\times / 3.4\times / 4.2\times$ ($13.0\times$ pooled). A token-level extension reaches $91$--$99\%$ coverage with $4.5\times$--$6.5\times$ floors. (iii) Pipeline losslessness: oracle programs round-trip through the deterministic resolver on all $482$ cases, localizing any downstream failure to span selection rather than the mechanism. A perturbation study shows pooled EM drops from $100\%$ to $15.48\%$ under off-by-one noise. A fine-tuning pilot on Qwen2.5-Coder-1.5B lifts HEvalFix-Py EM from $0/33$ (untrained) to $12$--$17\%$, a learnability signal, not a production selector. Batched-serving integration and multi-file coverage are scoped as follow-up.
>
---
#### [replaced 034] SSDAU: Structured Semantic Data Augmentation for Joint Entity and Relation Extraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对联合实体与关系抽取任务，解决数据质量差导致模型泛化能力弱的问题，提出SSDAU方法，在数据增强中保持语义结构，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.23440](https://arxiv.org/pdf/2605.23440)**

> **作者:** Jiawei He; Mengyu Shi; Jiawei Liu; Zhijie Wang; Chunrong Fang; Xikai Yang; Zhenyu Chen
>
> **备注:** 12 pages, 3 figure
>
> **摘要:** Joint Entity and Relation Extraction (JERE) is highly susceptible to weak generalization due to low-quality training data. Data augmentation is a common strategy to enhance model generalization across different domains. However, existing data augmentation methods often overlook text relevance and may disrupt semantic structures and dependencies, making it difficult to generate effective augmented data for improving model generalization. In this paper, we propose Structured Semantic Data Augmentation (SSDAU), a novel method designed to preserve the semantic structure of text during augmentation. SSDAU segments text based on entity labels and employs an encoder to capture semantic features of entities through context awareness. It then performs entity semantic restructuring to generate augmented data. To distinguish semantically similar entities, SSDAU fuses contextualized embeddings with traditional similarity scores. To mitigate potential topic ambiguity and information loss, we apply the BERTTopic model to filter out irrelevant topics, ensuring topic consistency. We evaluate SSDAU on datasets with different annotation types and compare its performance on five representative JERE models against seven popular data augmentation baselines. Experiments demonstrate that SSDAU generates semantically consistent data with superior robustness against ambiguity (8.26% F1 decrease vs. 31.91% for baselines), significantly outperforming all existing methods across all metrics.
>
---
#### [replaced 035] Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出Uni-DPO，解决LLMs动态偏好优化问题，通过自适应重加权提升数据利用效率和模型性能。**

- **链接: [https://arxiv.org/pdf/2506.10054](https://arxiv.org/pdf/2506.10054)**

> **作者:** Shangpin Peng; Weinong Wang; Zhuotao Tian; Senqiao Yang; Xing Wu; Haotian Xu; Chengquan Zhang; Takashi Isobe; Baotian Hu; Min Zhang
>
> **备注:** Accepted by ICLR 2026. Code & models: this https URL
>
> **摘要:** Direct Preference Optimization (DPO) has emerged as a cornerstone of reinforcement learning from human feedback (RLHF) due to its simplicity and efficiency. However, existing DPO-based methods typically treat all preference pairs equally, overlooking substantial variations in data quality and learning difficulty, which leads to inefficient data utilization and suboptimal performance. To address this limitation, we propose Uni-DPO, a unified dynamic preference optimization framework that jointly considers (a) the inherent quality of preference pairs and (b) the model's evolving performance during training. By adaptively reweighting samples based on both factors, Uni-DPO enables more effective use of preference data and achieves superior performance. Extensive experiments across models and benchmarks demonstrate the effectiveness and generalization of Uni-DPO. On textual tasks, Gemma-2-9B-IT fine-tuned with Uni-DPO surpasses the leading LLM, Claude 3 Opus, by 6.7 points on Arena-Hard. On mathematical and multimodal tasks, Uni-DPO consistently outperforms baseline methods across all benchmarks, providing strong empirical evidence of its effectiveness and robustness.
>
---
#### [replaced 036] Prism: Spectral-Aware Block-Sparse Attention
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于自然语言处理任务，解决长序列注意力计算效率问题。针对块稀疏注意力中块选择效率低的问题，提出Prism方法，通过频域分析提升块重要性估计效率。**

- **链接: [https://arxiv.org/pdf/2602.08426](https://arxiv.org/pdf/2602.08426)**

> **作者:** Xinghao Wang; Pengyu Wang; Xiaoran Liu; Fangxu Liu; Jason Chu; Kai Song; Xipeng Qiu
>
> **备注:** ICML 2026
>
> **摘要:** Block-sparse attention is promising for accelerating long-context LLM pre-filling, yet identifying relevant blocks efficiently remains a bottleneck. Existing methods typically employ coarse-grained attention as a proxy for block importance estimation, but often resort to expensive token-level searching or scoring, resulting in significant selection overhead. In this work, we trace the inaccuracy of standard coarse-grained attention via mean pooling to a theoretical root cause: the interaction between mean pooling and Rotary Positional Embeddings (RoPE). We prove that mean pooling acts as a low-pass filter that induces destructive interference in high-frequency dimensions, effectively creating a "blind spot" for local positional information (e.g., slash patterns). To address this, we introduce Prism, a training-free spectral-aware approach that decomposes block selection into high-frequency and low-frequency branches. By applying energy-based temperature calibration, Prism restores the attenuated positional signals directly from pooled representations, enabling block importance estimation using purely block-level operations, thereby improving efficiency. Extensive evaluations confirm that Prism maintains accuracy parity with full attention while delivering up to $\mathbf{5.1\times}$ speedup.
>
---
#### [replaced 037] Benchmarking and Learning Real-World Customer Service Dialogue
- **分类: cs.CL**

- **简介: 该论文属于智能客服对话系统研究，旨在解决工业场景下对话质量与实际需求不匹配的问题。通过构建OlaBench基准和OlaMind模型，提升对话系统的服务能力与可靠性。**

- **链接: [https://arxiv.org/pdf/2510.22143](https://arxiv.org/pdf/2510.22143)**

> **作者:** Tianhong Gao; Jundong Shen; Jiapeng Wang; Bei Shi; Ying Ju; Junfeng Yao; Huiyu Yu
>
> **摘要:** Existing benchmarks and training pipelines for industrial intelligent customer service (ICS) remain misaligned with real-world dialogue requirements, overemphasizing verifiable task success while under-measuring subjective service quality and realistic failure modes, leaving a gap between offline gains and deployable dialogue behavior. We close this gap with a benchmark-to-optimization loop: we first introduce OlaBench, an ICS benchmark spanning retrieval-augmented generation, workflow-based systems, and agentic settings, which evaluates service capability, safety, and latency sensitivity; moreover, motivated by OlaBench results showing state-of-the-art LLMs still fall short, we propose OlaMind, which distills reusable reasoning patterns and service strategies from expert dialogues and applies staged exploration--exploitation reinforcement learning with instance-level rubric-aware guidance to improve model capability. OlaMind surpasses GPT-5.2 and Gemini 3 Pro on OlaBench (83.64 vs. 70.58/70.84) and, in online A/B tests, delivers an average +23.67% issue resolution and -6.6% human transfer rate versus the baseline, bridging offline gains to deployment. Together, OlaBench and OlaMind advance ICS systems toward more anthropomorphic, professional, and reliable deployment. The project page and evaluation are available at this https URL.
>
---
#### [replaced 038] Can Large Language Models Resolve Semantic Discrepancy in Self-Destructive Subcultures? Evidence from Jirai Kei
- **分类: cs.CL**

- **简介: 该论文属于行为检测任务，旨在解决子文化中自我毁灭行为识别中的语义偏差问题。针对语言模型的知识滞后和语义不匹配，提出SAS框架提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.05004](https://arxiv.org/pdf/2601.05004)**

> **作者:** Peng Wang; Xilin Tao; Siyi Yao; Jiageng Wu; Yuntao Zou; Zhuotao Tian; Libo Qin; Dagang Li
>
> **备注:** Preprint
>
> **摘要:** Self-destructive behaviors are linked to complex psychological states and can be challenging to diagnose. These behaviors may be even harder to identify within subcultural groups due to their unique expressions. As large language models (LLMs) being deployed across various fields, some researchers have begun exploring their application for detecting self-destructive behaviors. Motivated by this, we investigate self-destructive behavior detection within subcultures using current LLM-based methods. However, these methods have two main challenges: (1) Knowledge Lag: Subcultural slang evolves rapidly, faster than LLMs' training cycles; and (2) Semantic Misalignment: it is challenging to grasp the specific and nuanced expressions unique to subcultures. To address these issues, we propose Subcultural Alignment Solver (SAS), a multi-agent framework that incorporates automatic retrieval and subculture alignment, significantly boosting the performance of LLMs in detecting self-destructive behavior. Our experimental results show that SAS outperforms the current advanced multi-agent framework OWL. Notably, it competes well with fine-tuned LLMs. We hope that SAS will advance the field of self-destructive behavior detection in subcultural contexts and serve as a valuable resource for future researchers.
>
---
#### [replaced 039] Profiling learners' affective engagement: Emotion AI, intercultural pragmatics, and language learning
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 本文探讨情感AI在语言学习中的应用，分析其如何影响学习者的情绪参与及交际能力发展，旨在解决技术与情感互动的融合问题。**

- **链接: [https://arxiv.org/pdf/2603.20479](https://arxiv.org/pdf/2603.20479)**

> **作者:** Robert Godwin-Jones
>
> **摘要:** Learning another language can be a highly emotional process, typically characterized by numerous frustrations and triumphs, big and small. For most learners, language learning does not follow a linear, predictable path, its zigzag course shaped by motivational (or demotivating) variables such as personal characteristics, teacher/peer relationships, learning materials, and dreams of a future L2 (second language) self. While some aspects of language learning (reading, grammar) are relatively mechanical, others can be stressful and unpredictable, especially conversing in the target language. That experience necessitates not only knowledge of structure and lexis, but also the ability to use the language in ways that are appropriate to the social and cultural context. A new opportunity to practice conversational abilities has arrived through the availability of AI chatbots, with both advantages (responsive, non-judgmental) and drawbacks (emotionally void, culturally biased). This column explores aspects of emotion as they arise in technology use and in particular how automatic emotion recognition and simulated human responsiveness in AI systems interface with language learning and the development of pragmatic and interactional competence. Emotion AI, the algorithmically driven interpretation of users' affective signals, has been seen as enabling greater personalized learning, adapting to perceived learner cognitive and emotional states. Others warn of emotional manipulation and inappropriate and ineffective user profiling
>
---
#### [replaced 040] Learning Concepts, Not Tokens: Self-Supervised Semantic Alignment for Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型对语义理解不足的问题。通过自监督学习，让模型预测概念而非单个词，提升语义对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.29123](https://arxiv.org/pdf/2603.29123)**

> **作者:** Christine Zhang; Dan Jurafsky; Chen Shani
>
> **摘要:** The next-token prediction (NTP) objective trains language models to predict a single token at each step, even though many continuations can express the same meaning. For example, in the sentence ``this sticker can be placed here'', positioned, attached, or put are all plausible alternatives. While standard NTP training treats these alternatives as mutually exclusive targets, we explore a self-supervised framework that encourages models to predict concepts, approximated as sets of semantically equivalent tokens. Models trained with this concept supervision align better with human similarity judgments, improve classification, clustering, and reranking performance, and achieve comparable or stronger downstream reasoning. These gains come with lower perplexity on semantically meaningful words (Section 3.2) and only minimal increases in global perplexity, suggesting that concepts enhance semantic alignment while preserving language modeling quality. Our code is available at this https URL .
>
---
#### [replaced 041] A Tutorial on Diffusion Theory: From Differential Equations to Diffusion Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于生成模型领域，旨在解析扩散模型的数学基础。通过微分方程视角，建立前向与反向过程，解决噪声预测与采样问题。**

- **链接: [https://arxiv.org/pdf/2605.22586](https://arxiv.org/pdf/2605.22586)**

> **作者:** Jiayi Fu; Yuxia Wang
>
> **备注:** A detailed tutorial on Diffusion models and SDE
>
> **摘要:** This tutorial develops diffusion models from the viewpoint of differential equations. We begin with the conditional Gaussian forward process and show that this path admits both an ordinary differential equation (ODE) representation and a stochastic differential equation (SDE) representation. Averaging the conditional process over the data distribution then yields marginalized forward ODE and SDE formulations that transport the data distribution $p_0=p_{\mathrm{data}}$ to a Gaussian prior $p_1=\mathcal{N}(0,I)$. We next derive the corresponding reverse-time dynamics, namely the reverse SDE and the reverse probability-flow ODE, both of which are governed by the marginal score $\grad\log p_t(x)$. This leads to a training objective for score estimation and shows that the standard noise-prediction objective is equivalent to score matching up to an additive constant independent of the model parameters. We then discuss sampling methods for the learned reverse dynamics, including DPM-Solver, as well as guided sampling through classifier guidance and classifier-free guidance. Finally, we compare DDPM and DDIM with the reverse SDE/ODE framework and show that they share the same training objective, while DDPM sampling corresponds to discrete reverse-SDE sampling and DDIM sampling corresponds to reverse-ODE sampling.
>
---
#### [replaced 042] Agent Learning via Early Experience
- **分类: cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文研究语言代理通过早期经验学习的策略，解决传统强化学习在无奖励环境中的困难。提出隐式建模和自我反思两种方法，提升代理性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2510.08558](https://arxiv.org/pdf/2510.08558)**

> **作者:** Kai Zhang; Xiangchao Chen; Bo Liu; Tianci Xue; Zeyi Liao; Zhihan Liu; Xiyao Wang; Yuting Ning; Zhaorun Chen; Xiaohan Fu; Jian Xie; Yuxuan Sun; Boyu Gou; Qi Qi; Zihang Meng; Jianwei Yang; Ning Zhang; Xian Li; Ashish Shah; Dat Huynh; Hengduo Li; Zi Yang; Sara Cao; Lawrence Jang; Shuyan Zhou; Jiacheng Zhu; Huan Sun; Jason Weston; Yu Su; Yifan Wu
>
> **备注:** ICML 2026
>
> **摘要:** A long-term goal of language agents is to learn and improve through their own experience, ultimately outperforming humans in complex, real-world tasks. However, training agents from experience data with reinforcement learning remains difficult in many environments, which either lack verifiable rewards (e.g., websites) or require inefficient long-horizon rollouts (e.g., multi-turn tool use). As a result, most current agents rely on supervised fine-tuning on expert data, which is challenging to scale and generalizes poorly. This limitation stems from the nature of expert demonstrations: they capture only a narrow range of scenarios, and expose the agent to limited environment diversity. We address this limitation with a middle-ground paradigm we call early experience: interaction data generated by the agent's own actions, where the resulting future states serve as supervision without reward signals. Within this paradigm, we study two strategies of using such data: (1) implicit world modeling, which uses collected states to ground the policy in environment dynamics; and (2) self-reflection, where the agent learns from its suboptimal actions to improve reasoning and decision-making. Evaluation across eight diverse environments and multiple model families shows that our approaches consistently improve effectiveness and out-of-domain generalization, highlighting the value of early experience. Moreover, in environments with verifiable rewards, our results provide promising signals that early experience offers a strong foundation for subsequent reinforcement learning, making it a practical bridge between imitation learning and fully experience-driven agents.
>
---
#### [replaced 043] AI-generated podcasts: Synthetic Intimacy and Cultural Mistranslation in NotebookLM's Audio Overviews
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 论文分析Google NotebookLM生成的AI播客，探讨其结构与文化翻译问题。属于媒介分析任务，旨在揭示AI如何重构播客内容与文化语境。**

- **链接: [https://arxiv.org/pdf/2511.08654](https://arxiv.org/pdf/2511.08654)**

> **作者:** Jill Walker Rettberg
>
> **备注:** This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement number 101142306. The project is also supported by the Center for Digital Narrative, which is funded by the Research Council of Norway through its Centres of Excellence scheme, project number 332643. Media, Culture & Society, online first (2026)
>
> **摘要:** This paper analyses AI-generated podcasts produced by Google's NotebookLM, which generates audio podcasts with two chatty AI hosts discussing whichever documents a user uploads. While AI-generated podcasts have been discussed as tools, for instance in medical education, they have not yet been analysed as media. By uploading different types of text and analysing the generated outputs I show how the podcasts' structure is built around a fixed template. I also find that NotebookLM not only translates texts from other languages into a perky standardised Mid-Western American accent, it also translates cultural contexts to a white, educated, middle-class American default. This is a distinct development in how publics are shaped by media, marking a departure from the multiple public spheres that scholars have described in human podcasting from the early 2000s until today, where hosts spoke to specific communities and responded to listener comments, to an abstraction of the podcast genre.
>
---
#### [replaced 044] Scaling Natural-Language Graph-Based Test Time Compute for Automated Theorem Proving
- **分类: cs.CL**

- **简介: 该论文属于自动化定理证明任务，旨在解决自然语言中数学概念识别与证明形式化问题。工作是提出KG-prover框架，利用知识图谱增强大语言模型的证明能力。**

- **链接: [https://arxiv.org/pdf/2503.11657](https://arxiv.org/pdf/2503.11657)**

> **作者:** Vincent Li; Tim Knappe; Yule Fu; Kevin Han; Kevin Zhu
>
> **备注:** Accepted to ICML AI4Math Workshop 2025, NAACL SRW 2025
>
> **摘要:** Large language models have demonstrated remarkable capabilities in natural language processing tasks requiring multi-step logical reasoning capabilities, such as automated theorem proving. However, challenges persist within theorem proving, such as the identification of key mathematical concepts, understanding their interrelationships, and formalizing proofs correctly within natural language. We present KG-prover, a novel framework that leverages knowledge graphs mined from reputable mathematical texts to augment general-purpose LLMs to construct and formalize mathematical proofs. We also study the effects of scaling graph-based, test-time compute using KG-Prover, demonstrating significant performance improvements over baselines across multiple datasets. General-purpose LLMs improve up to 21\% on miniF2F-test when combined with KG-Prover, with consistent improvements ranging from 2-11\% on the ProofNet, miniF2F-test, and MUSTARD datasets. Furthermore, KG-Prover with o4-mini achieves 50\% on pass miniF2F-test. This work provides a promising approach for augmenting natural language proof reasoning with knowledge graphs without the need for additional finetuning.
>
---
#### [replaced 045] AgentCoMa: A Compositional Benchmark Mixing Commonsense and Mathematical Reasoning in Real-World Scenarios
- **分类: cs.CL**

- **简介: 该论文提出AgentCoMa基准，用于测试大语言模型在混合常识与数学推理任务中的表现，旨在解决模型在组合型任务中性能下降的问题。**

- **链接: [https://arxiv.org/pdf/2508.19988](https://arxiv.org/pdf/2508.19988)**

> **作者:** Lisa Alazraki; Lihu Chen; Ana Brassard; Joe Stacey; Hossein A. Rahmani; Marek Rei
>
> **备注:** ACL 2026
>
> **摘要:** Large Language Models (LLMs) have achieved high accuracy on complex commonsense and mathematical problems that involve the composition of multiple reasoning steps. However, current compositional benchmarks testing these skills tend to focus on either commonsense or math reasoning, whereas LLM agents solving real-world tasks would require a combination of both. In this work, we introduce an Agentic Commonsense and Math benchmark (AgentCoMa), where each compositional task requires a commonsense reasoning step and a math reasoning step. We test it on 61 LLMs of different sizes, model families, and training strategies. We find that LLMs can usually solve both steps in isolation, yet their accuracy drops by nearly 30% on average when the two are combined. This is a substantially greater performance gap than the one we observe in prior compositional benchmarks that combine multiple steps of the same reasoning type. In contrast, non-expert human annotators can solve the compositional questions and the individual steps in AgentCoMa with similarly high accuracy. Furthermore, we conduct a series of interpretability studies to better understand the performance gap, examining neuron patterns, attention maps and membership inference. Our work underscores a substantial degree of model brittleness in the context of mixed-type compositional reasoning and offers a test bed for future improvement.
>
---
#### [replaced 046] Retrieved In-Context Principles from Previous Mistakes
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决模型在推理中因错误导致的性能不足问题。通过构建RICP框架，从错误中提取原则以提升模型表现。**

- **链接: [https://arxiv.org/pdf/2407.05682](https://arxiv.org/pdf/2407.05682)**

> **作者:** Hao Sun; Yong Jiang; Bo Wang; Yingyan Hou; Yan Zhang; Pengjun Xie; Fei Huang
>
> **摘要:** In-context learning (ICL) has been instrumental in adapting Large Language Models (LLMs) to downstream tasks using correct input-output examples. Recent advances have attempted to improve model performance through principles derived from mistakes, yet these approaches suffer from lack of customization and inadequate error coverage. To address these limitations, we propose Retrieved In-Context Principles (RICP), a novel teacher-student framework. In RICP, the teacher model analyzes mistakes from the student model to generate reasons and insights for preventing similar mistakes. These mistakes are clustered based on their underlying reasons for developing task-level principles, enhancing the error coverage of principles. During inference, the most relevant mistakes for each question are retrieved to create question-level principles, improving the customization of the provided guidance. RICP is orthogonal to existing prompting methods and does not require intervention from the teacher model during inference. Experimental results across seven reasoning benchmarks reveal that RICP effectively enhances performance when applied to various prompting strategies.
>
---
#### [replaced 047] $π$-Play: Multi-Agent Self-Play via Privileged Self-Distillation without External Data
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出$\pi$-Play框架，解决深度搜索代理训练效率低的问题。通过自玩生成的QCP提供特权信息，提升自蒸馏效果，显著提高进化效率。**

- **链接: [https://arxiv.org/pdf/2604.14054](https://arxiv.org/pdf/2604.14054)**

> **作者:** Yaocheng Zhang; Yuanheng Zhu; Wenyue Chong; Songjun Tu; Qichao Zhang; Jiajun Chai; Xiaohan Wang; Wei Lin; Guojun Yin; Dongbin Zhao
>
> **备注:** 23 pages, 11 figures
>
> **摘要:** Deep search agents have emerged as a promising paradigm for addressing complex information-seeking tasks, but their training remains challenging due to sparse rewards, weak credit assignment, and limited labeled data. Self-play offers a scalable route to reduce data dependence, but conventional self-play optimizes students only through sparse outcome rewards, leading to low learning efficiency. In this work, we observe that self-play naturally produces a question construction path (QCP) during task generation, an intermediate artifact that captures the reverse solution process. This reveals a new source of privileged information: self-play can provide high-quality privileged information for the self-distillation at low cost and at scale, without relying on human feedback or curated privileged information. Leveraging this insight, we propose Privileged Information Self-Play ($\pi$-Play), a novel multi-agent self-evolution framework combining self-play and self-distillation. In $\pi$-Play, an examiner generates tasks together with QCPs, and a teacher employs QCP as privileged context to densely supervise a student via self-distillation. This design transforms sparse-reward self-play into a dense-feedback co-evolution. Extensive experiments show that data-free $\pi$-Play surpasses fully supervised search agents and improves evolutionary efficiency by 2-3$\times$ over conventional self-play. Code is available at this https URL.
>
---
#### [replaced 048] MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出MemSkill，解决LLM代理记忆管理僵化问题，通过可学习和进化的记忆技能提升记忆提取与更新效率。**

- **链接: [https://arxiv.org/pdf/2602.02474](https://arxiv.org/pdf/2602.02474)**

> **作者:** Haozhen Zhang; Quanyu Long; Jianzhu Bao; Tao Feng; Weizhi Zhang; Haodong Yue; Wenya Wang
>
> **备注:** Code is available at this https URL
>
> **摘要:** Most Large Language Model (LLM) agent memory systems rely on a small set of static, hand-designed operations for extracting memory. These fixed procedures hard-code human priors about what to store and how to revise memory, making them rigid under diverse interaction patterns and inefficient on long histories. To this end, we present \textbf{MemSkill}, which reframes these operations as learnable and evolvable memory skills, structured and reusable routines for extracting, consolidating, and pruning information from interaction traces. Inspired by the design philosophy of agent skills, MemSkill employs a \emph{controller} that learns to select a small set of relevant skills, paired with an LLM-based \emph{executor} that produces skill-guided memories. Beyond learning skill selection, MemSkill introduces a \emph{designer} that periodically reviews hard cases where selected skills yield incorrect or incomplete memories, and evolves the skill set by proposing refinements and new skills. Together, MemSkill forms a closed-loop procedure that improves both the skill-selection policy and the skill set itself. Experiments on LoCoMo, LongMemEval, HotpotQA, and ALFWorld demonstrate that MemSkill improves task performance over strong baselines and generalizes well across settings. Further analyses shed light on how skills evolve, offering insights toward more adaptive, self-evolving memory management for LLM agents.
>
---
#### [replaced 049] Uncovering Autoregressive LLM Knowledge of Thematic Fit in Event Representation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究自动回归大语言模型在事件表述主题适配性上的知识。任务是评估语义角色与论元的兼容性，通过不同提示设计实验，发现模型表现差异及输入形式影响得分分布。**

- **链接: [https://arxiv.org/pdf/2410.15173](https://arxiv.org/pdf/2410.15173)**

> **作者:** Safeyah Khaled Alshemali; Daniel Bauer; Yuval Marton
>
> **备注:** Significant update with massive changes: all experiments rerun with current LLMs; includes new probability estimate analysis and expanded results in Sections 4 and 5. The paper has been accepted to CoNLL-2026
>
> **摘要:** The thematic fit estimation task measures semantic arguments' compatibility with a given semantic role for a given predicate. We investigate if autoregressive LLMs have consistent, expressible knowledge of event arguments' thematic fit by experimenting with various prompt designs, manipulating input context, reasoning, and output forms. We set a new state-of-the-art on thematic fit benchmarks, but show that closed and open weight LLMs respond differently to our prompting strategies: Closed models achieve better scores overall and benefit from multi-step reasoning, but they perform worse at filtering out generated sentences incompatible with the given predicate, role, and argument. Our analysis shows that lemma tuple input and sentence input result in surprisingly different thematic fit score distributions.
>
---
#### [replaced 050] MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Any-Precision LLM
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型量化任务，旨在解决动态精度下模型部署的灵活性问题。提出MoBiQuant框架，通过混合位量化实现灵活精度推理，提升内存效率和吞吐量。**

- **链接: [https://arxiv.org/pdf/2602.20191](https://arxiv.org/pdf/2602.20191)**

> **作者:** Dongwei Wang; Jinhee Kim; Seokho Han; Denis Gudovskiy; Yohei Nakata; Tomoyuki Okuno; KhayTze Peong; Kang Eun Jeon; Jong Hwan Ko; Yiran Chen; Huanrui Yang
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** Dynamic runtime latency and memory constraints necessitate flexible large language model (LLM) deployment, where an LLM can be inferred with various quantization precisions based on available computational resources. Recent work on such any-precision quantization either relies on hardware-inefficient vector quantization or induces additional scaling factors when switching between bit-widths. Meanwhile, existing post-training quantization (PTQ) methods calibrated for a fixed low precision show poor generalizability under runtime precision change. In this work, we attribute the source of poor generalization across bit-widths to a precision-dependent \textit{outlier migration} phenomenon where the distribution of PTQ-sensitive tokens changes across precisions. Motivated by this observation, we propose \texttt{MoBiQuant}, a novel any-precision Mixture-of-Bits quantization framework that adjusts weight precision for flexible LLM inference based on token sensitivity. Specifically, we propose a many-in-one recursive residual quantization that can iteratively reconstruct higher-precision weights at runtime and mitigates \textit{outlier migration} with a token-aware router to dynamically select the optimal inference precision of each this http URL experiments show that \texttt{MoBiQuant} matches or surpasses frontier single-precision PTQ while exhibiting strong elasticity, achieving significant memory savings and throughput gains of up to $1.34\times$ over state-of-the-art any-precision methods.
>
---
#### [replaced 051] Voice of India: A Large-Scale Benchmark for Real-World Speech Recognition in India
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决现有数据集存在的过拟合和拼写偏差问题。构建了包含15种印度语言的大规模真实语音数据集Voice of India，并分析了不同因素对ASR性能的影响。**

- **链接: [https://arxiv.org/pdf/2604.19151](https://arxiv.org/pdf/2604.19151)**

> **作者:** Kaushal Bhogale; Manas Dhir; Amritansh Walecha; Manmeet Kaur; Vanshika Chhabra; Aaditya Pareek; Hanuman Sidh; Mahima Manik; Sagar Jain; Bhaskar Singh; Utkarsh Singh; Tahir Javed; Shobhit Banga; Mitesh M. Khapra
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** Existing Indic ASR benchmarks often use scripted, clean speech and leaderboard driven evaluation that encourages dataset specific overfitting. In addition, strict single reference WER penalizes natural spelling variation in Indian languages, including non standardized spellings of code-mixed English origin words. To address these limitations, we introduce Voice of India, a closed source benchmark built from unscripted telephonic conversations covering 15 major Indian languages across 139 regional clusters. The dataset contains 306230 utterances, totaling 536 hours of speech from 36691 speakers with transcripts accounting for spelling variations. We also analyze performance geographically at the district level, revealing disparities. Finally, we provide detailed analysis across factors such as audio quality, speaking rate, gender, and device type, highlighting where current ASR systems struggle and offering insights for improving real world Indic ASR systems.
>
---
#### [replaced 052] Findings of the Counter Turing Test: AI-Generated Text Detection
- **分类: cs.CL**

- **简介: 该论文属于AI生成文本检测任务，旨在区分人类与AI写作并识别生成模型。通过CT2测试，评估检测技术效果，提出改进方法。**

- **链接: [https://arxiv.org/pdf/2605.20761](https://arxiv.org/pdf/2605.20761)**

> **作者:** Rajarshi Roy; Gurpreet Singh; Ashhar Aziz; Shashwat Bajpai; Nasrin Imanpour; Shwetangshu Biswas; Kapil Wanaskar; Parth Patwa; Subhankar Ghosh; Shreyas Dixit; Nilesh Ranjan Pal; Vipula Rawte; Ritvik Garimella; Amitava Das; Amit Sheth; Vasu Sharma; Aishwarya Naresh Reganti; Vinija Jain; Aman Chadha
>
> **备注:** Defactify4 @AAAI 2025
>
> **摘要:** The growing capability of large language models to produce fluent, contextually coherent text has created mounting pressure on the systems and institutions responsible for ensuring the authenticity of digital content. Advanced generative models such as GPT-4, Claude 3.5, and Llama can produce highly coherent and human-like text, making it increasingly difficult to differentiate between human-written and AI-generated content. While these models have transformative applications, their misuse has raised concerns about misinformation, biased narratives, and security threats. This paper provides a comprehensive analysis of state-of-the-art AI-generated text detection techniques and evaluates their effectiveness through the Counter Turing Test (CT2) shared tasks. Task A (Binary Classification) required participants to distinguish between human-written and AI-generated text, while Task B (Model Attribution) focused on identifying the specific language model responsible for generating a given text. The results demonstrated high performance in binary classification, with the top system achieving an F1 score of 1.0000, but significantly lower scores in model attribution, where the best system achieved 0.9531, highlighting the increased complexity of this task. The top-performing teams leveraged fine-tuned transformer models, ensemble learning, and hybrid detection approaches, with DeBERTa-based and BART-based methods demonstrating strong results. However, the lower scores in Task B underscore the challenges of distinguishing outputs from different LLMs, necessitating further research into adversarial robustness, feature extraction, and cross-domain generalization.
>
---
#### [replaced 053] Routing by Analogy: kNN-Augmented Expert Assignment for Mixture-of-Experts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决MoE架构中路由机制在分布偏移下脆弱的问题。通过引入kNN-MoE，利用相似案例增强路由决策，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.02144](https://arxiv.org/pdf/2601.02144)**

> **作者:** Boxuan Lyu; Soichiro Murakami; Hidetaka Kamigaito; Peinan Zhang
>
> **摘要:** Mixture-of-Experts (MoE) architectures scale large language models efficiently by employing a parametric ``router'' to dispatch tokens to a sparse subset of experts. Typically, this router is trained once and then frozen, rendering routing decisions brittle under distribution shifts. We address this limitation by introducing kNN-MoE, a retrieval-augmented routing framework that reuses locally optimal expert assignments from a memory of similar past cases. This memory is constructed offline by directly optimizing token-wise routing logits to maximize the likelihood on a reference set. Crucially, we use the average similarity of retrieved neighbors as a confidence-driven mixing coefficient, thus allowing the method to fall back to the frozen router when no relevant cases are found. Experiments show that kNN-MoE outperforms the zero-shot baseline and is competitive with computationally intensive supervised fine-tuning.
>
---
#### [replaced 054] CLiViS: Unleashing Cognitive Map through Linguistic-Visual Synergy for Embodied Visual Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于 embodied visual reasoning 任务，旨在解决长视频中复杂指令理解与推理问题。提出 CLiViS 框架，结合语言模型与视觉模型优势，构建动态认知地图以提升推理效果。**

- **链接: [https://arxiv.org/pdf/2506.17629](https://arxiv.org/pdf/2506.17629)**

> **作者:** Kailing Li; Qi'ao Xu; Tianwen Qian; Yuqian Fu; Yang Jiao; Xiaoling Wang
>
> **摘要:** Embodied Visual Reasoning (EVR) seeks to follow complex, free-form instructions based on egocentric video, enabling semantic understanding and spatiotemporal reasoning in dynamic environments. Despite its promising potential, EVR encounters significant challenges stemming from the diversity of complex instructions and the intricate spatiotemporal dynamics in long-term egocentric videos. Prior solutions either employ Large Language Models (LLMs) over static video captions, which often omit critical visual details, or rely on end-to-end Vision-Language Models (VLMs) that struggle with stepwise compositional reasoning. Consider the complementary strengths of LLMs in reasoning and VLMs in perception, we propose CLiViS. It is a novel training-free framework that leverages LLMs for high-level task planning and orchestrates VLM-driven open-world visual perception to iteratively update the scene context. Building on this synergy, the core of CLiViS is a dynamic Cognitive Map that evolves throughout the reasoning process. This map constructs a structured representation of the embodied scene, bridging low-level perception and high-level reasoning. Extensive experiments across multiple benchmarks demonstrate the effectiveness and generality of CLiViS, especially in handling long-term visual dependencies. Code is available at this https URL.
>
---
#### [replaced 055] Cooperative Memory Paging with Keyword Bookmarks for Long-Horizon LLM Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于长对话任务，解决LLM对话超出上下文窗口后内容恢复问题，提出协作分页机制，用关键词书签替代被驱逐内容，并通过召回工具实现按需恢复。**

- **链接: [https://arxiv.org/pdf/2604.12376](https://arxiv.org/pdf/2604.12376)**

> **作者:** Ziyang Liu
>
> **备注:** The authors have decided to withdraw this version following internal review regarding authorship and contribution agreements
>
> **摘要:** When LLM conversations grow beyond the context window, old content must be evicted -- but how does the model recover it when needed? We propose cooperative paging: evicted segments are replaced with minimal keyword bookmarks ([pN:keywords], ~8-24 tokens each), and the model is given a recall() tool to retrieve full content on demand. On the LoCoMo benchmark (10 real multi-session conversations, 300+ turns), cooperative paging achieves the highest answer quality among six methods -- outperforming truncation, BM25, word-overlap retrieval, a search-tool baseline, and full context -- on four models (GPT-4o-mini, DeepSeek-v3.2, Claude Haiku, GLM-5), confirmed by four independent LLM judges ($p=0.017$, paired bootstrap). We then study the paging design space with a 5x4 ablation over boundary strategies and eviction policies (3,176 synthetic probes, 1,600 LoCoMo probes). Key findings: (1) coarse fixed-size pages (fixed_20) reach 96.7% while content-aware topic_shift collapses to 56.7%; (2) eviction policy choice is data-dependent (FIFO best on synthetic, LFU on LoCoMo); (3) two bookmark generation strategies improve over the heuristic baseline (+4.4 and +8.7 E2E points); (4) the remaining bottleneck is bookmark discrimination -- the model triggers recall() 96% of the time but selects the correct page only 57% when bookmarks are insufficiently distinctive. Keyword specificity alone accounts for a 25 percentage point accuracy difference.
>
---
#### [replaced 056] Pragmatic Reasoning improves LLM Code Generation
- **分类: cs.CL; cs.AI; cs.SE**

- **简介: 该论文属于自然语言到代码生成任务，解决用户指令模糊导致的多解问题。提出CodeRSA方法，通过局部语用比较提升代码生成准确性。**

- **链接: [https://arxiv.org/pdf/2502.15835](https://arxiv.org/pdf/2502.15835)**

> **作者:** Zhuchen Cao; Sven Apel; Adish Singla; Vera Demberg
>
> **摘要:** Pragmatic reasoning helps interlocutors infer intended meaning from ambiguous or underspecified messages by considering shared context and counterfactual alternatives. Similar challenges arise in natural language-to-code generation, where user instructions often admit multiple plausible candidate programs. However, direct RSA-style inference is difficult because it requires probability estimation over large spaces of programs and alternative instructions. We propose CodeRSA, an RSA-motivated reranking method that makes pragmatic reasoning tractable through local pragmatic contests among sampled code candidates. CodeRSA constructs candidate-induced alternative instructions and estimates which candidates are most distinctively supported by the original instruction, avoiding global normalization over the full program-instruction space. We evaluate CodeRSA on HumanEval+, MBPP+, and BigCodeBench using four open-weight instruction-following models. CodeRSA achieves the strongest average accuracy in 10 of 12 model-benchmark settings and remains competitive in the remaining cases. Further analyses show that its gains come from combining local pairwise pragmatic comparison with broader global support, suggesting a scalable direction for language-to-code reranking under natural-language uncertainty.
>
---
#### [replaced 057] MoBayes: A Modular Bayesian Framework for Separating Reasoning from Language in Conversational Clinical Decision Support
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出MoBayes框架，解决临床决策支持中语言生成与概率推理混淆的问题。通过分离语言接口与贝叶斯推理模块，提升决策的可解释性与可靠性。**

- **链接: [https://arxiv.org/pdf/2604.20022](https://arxiv.org/pdf/2604.20022)**

> **作者:** Yusuf Kesmen; Fay Elhassan; Jiayi Ma; Julien Stalhandske; Yena Chang; David Sasu; Alexandra Kulinkina; Akhil Arora; Lars Klein; Mary-Anne Hartley
>
> **备注:** 50 pages including appendix, 13 figures, 22 tables. Preprint
>
> **摘要:** Large language models (LLMs) are increasingly used for conversational clinical decision support, yet they conflate next token prediction with probabilistic decision making. We argue that this conflation reflects an architectural limitation: such systems lack explicit posterior tracking, controllable abstention thresholds, and auditable reasoning chains. We introduce MoBayes, a Modular Bayesian dialogue framework that separates reasoning from language. The LLM acts only as a language interface, parsing patient conversation into structured observations, while a Bayesian module performs probabilistic inference over these observations to update posteriors, select follow-up questions via expected-information-gain and determine when to stop or defer through calibrated decision thresholds. This design enables explicit posterior tracking, controllable selective decision-making, and replaceable population-specific statistical backends without retraining the language model. Across empirical and LLM-generated knowledge bases, MoBayes outperforms standalone frontier LLM doctors, including matched model-family comparisons where inexpensive sensor models paired with MoBayes exceed larger autonomous models at lower cost. The advantage persists under adversarial patient communication styles and across varying diagnostic scenarios. These results suggest that reliable conversational clinical decision support systems should separate probabilistic reasoning from language generation rather than scaling model size alone. Code is available at this https URL
>
---
#### [replaced 058] AutoSOTA: An End-to-End Automated Research System for State-of-the-Art AI Model Discovery
- **分类: cs.CL; cs.CE**

- **简介: 该论文提出AutoSOTA，一个自动化研究系统，用于发现更优的AI模型。任务是加速模型优化流程，解决人工重复实验效率低的问题。通过多智能体协作，实现模型复现与优化，提升性能。**

- **链接: [https://arxiv.org/pdf/2604.05550](https://arxiv.org/pdf/2604.05550)**

> **作者:** Yu Li; Chenyang Shao; Xinyang Liu; Ruotong Zhao; Peijie Liu; Hongyuan Su; Zhibin Chen; Qinglong Yang; Anjie Xu; Yi Fang; Qingbin Zeng; Tianxing Li; Jingbo Xu; Fengli Xu; Yong Li; Tie-Yan Liu
>
> **摘要:** Artificial intelligence research increasingly depends on prolonged cycles of reproduction, debugging, and iterative refinement to achieve State-Of-The-Art (SOTA) performance, creating a growing need for systems that can accelerate the full pipeline of empirical model optimization. In this work, we introduce AutoSOTA, an end-to-end automated research system that advances the latest SOTA models published in top-tier AI papers to reproducible and empirically improved new SOTA models. We formulate this problem through three tightly coupled stages: resource preparation and goal setting; experiment evaluation; and reflection and ideation. To tackle this problem, AutoSOTA adopts a multi-agent architecture with eight specialized agents that collaboratively ground papers to code and dependencies, initialize and repair execution environments, track long-horizon experiments, generate and schedule optimization ideas, and supervise validity to avoid spurious gains. We evaluate AutoSOTA on recent research papers collected from eight top-tier AI conferences under filters for code availability and execution cost. Across these papers, AutoSOTA achieves strong end-to-end performance in both automated replication and subsequent optimization. Specifically, it successfully discovers 105 new SOTA models that surpass the original reported methods, averaging approximately five hours per paper. Case studies spanning LLM, NLP, computer vision, time series, and optimization further show that the system can move beyond routine hyperparameter tuning to identify architectural innovation, algorithmic redesigns, and workflow-level improvements. These results suggest that end-to-end research automation can serve not only as a performance optimizer, but also as a new form of research infrastructure that reduces repetitive experimental burden and helps redirect human attention toward higher-level scientific creativity.
>
---
#### [replaced 059] River-LLM: Large Language Model Seamless Exit Based on KV Share
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLM推理延迟高问题，提出River-LLM框架实现高效早期退出，提升速度并保持生成质量。**

- **链接: [https://arxiv.org/pdf/2604.18396](https://arxiv.org/pdf/2604.18396)**

> **作者:** Yingtao Shen; An Zou
>
> **备注:** Accepted to ACL 2026, 13pages, with appendix. Corrected some typos
>
> **摘要:** Large Language Models (LLMs) have demonstrated exceptional performance across diverse domains but are increasingly constrained by high inference latency. Early Exit has emerged as a promising solution to accelerate inference by dynamically bypassing redundant layers. However, in decoder-only architectures, the efficiency of Early Exit is severely bottlenecked by the KV Cache Absence problem, where skipped layers fail to provide the necessary historical states for subsequent tokens. Existing solutions, such as recomputation or masking, either introduce significant latency overhead or incur severe precision loss, failing to bridge the gap between theoretical layer reduction and practical wall-clock speedup. In this paper, we propose River-LLM, a training-free framework that enables seamless token-level Early Exit. River-LLM introduces a lightweight KV-Shared Exit River that allows the backbone's missing KV cache to be naturally generated and preserved during the exit process, eliminating the need for costly recovery operations. Furthermore, we utilize state transition similarity within decoder blocks to predict cumulative KV errors and guide precise exit decisions. Extensive experiments on mathematical reasoning and code generation tasks demonstrate that River-LLM achieves 1.53 to 2.16 times of practical speedup while maintaining high generation quality.
>
---
#### [replaced 060] ESI-Bench: Towards Embodied Spatial Intelligence that Closes the Perception-Action Loop
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文提出ESI-Bench，用于评估具身空间智能，解决感知-行动闭环问题，通过主动探索提升任务表现。**

- **链接: [https://arxiv.org/pdf/2605.18746](https://arxiv.org/pdf/2605.18746)**

> **作者:** Yining Hong; Jiageng Liu; Han Yin; Manling Li; Leonidas Guibas; Li Fei-Fei; Jiajun Wu; Yejin Choi
>
> **备注:** this https URL
>
> **摘要:** Spatial intelligence unfolds through a perception-action loop: agents act to acquire observations, and reason about how observations vary as a function of action. Rather than passively processing what is seen, they actively uncover what is unseen - occluded structure, dynamics, containment, and functionality that cannot be resolved from passive sensing alone. We move beyond prior formulations of spatial intelligence that assume oracle observations by recasting the observer as an actor. We introduce ESI-BENCH, a comprehensive benchmark for embodied spatial intelligence spanning 10 task categories and 29 subcategories built on OmniGibson, grounded in Spelke's core knowledge systems. Agents must decide what abilities to deploy - perception, locomotion, and manipulation - and how to sequence them to actively accumulate task-relevant evidence. We conduct extensive experiments on state-of-the-art MLLMs and find that active exploration substantially outperforms passive counterparts, with agents spontaneously discovering emergent spatial strategies without explicit instructions, while random multi-view often adds noise rather than signal despite consuming far more images. Most failures stem not from weak perception but from action blindness: poor action choices lead to poor observations, which in turn drive cascading errors. While explicit 3D grounding stabilizes reasoning on depth-sensitive tasks, imperfect 3D representation proves more harmful than 2D baselines by distorting spatial relations. Human studies further reveal that unlike humans who seek falsifying viewpoints and revise beliefs under contradiction, models commit prematurely with high confidence regardless of evidence quality, exposing a metacognitive gap that neither better perception nor more embodied interaction alone can close.
>
---
#### [replaced 061] PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自动化启发式设计任务，解决传统框架在组合优化问题中生成启发式方法效率低、缺乏推理的问题。提出PathWise框架，通过状态感知规划提升性能。**

- **链接: [https://arxiv.org/pdf/2601.20539](https://arxiv.org/pdf/2601.20539)**

> **作者:** Oguzhan Gungordu; Siheng Xiong; Faramarz Fekri
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Large Language Models (LLMs) have enabled automated heuristic design (AHD) for combinatorial optimization problems (COPs), but existing frameworks' reliance on fixed evolutionary rules and static prompt templates often leads to myopic heuristic generation, redundant evaluations, and limited reasoning about how new heuristics should be derived. We propose a novel multi-agent reasoning framework, referred to as Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs (PathWise), which formulates heuristic generation as a sequential decision process over an entailment graph serving as a compact, stateful memory of the search trajectory. This approach allows the system to carry forward past decisions and reuse or avoid derivation information across generations. A policy agent plans evolutionary actions, a world model agent generates heuristic rollouts conditioned on those actions, and critic agents provide routed reflections summarizing lessons from prior steps, shifting LLM-based AHD from trial-and-error evolution toward state-aware planning through reasoning. Experiments across diverse COPs show that PathWise converges faster to better heuristics, generalizes across different LLM backbones, and scales to larger problem sizes.
>
---
#### [replaced 062] Internalizing Outcome Supervision into Process Supervision: A New Paradigm for Reinforcement Learning for Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决推理中因缺乏过程监督而导致的信用分配难题。通过将结果监督内化为过程监督，实现更细粒度的策略优化。**

- **链接: [https://arxiv.org/pdf/2605.05226](https://arxiv.org/pdf/2605.05226)**

> **作者:** Fei Ding; Yongkang Zhang; Runhao Liu; Yuhao Liao; Zijian Zeng; Sibo wang; Huiming Yang
>
> **摘要:** The central challenge of reinforcement learning for reasoning lies not only in the sparsity of outcome-level supervision, but more fundamentally in how to transform feedback provided only at the end of a sequence into fine-grained learning signals that can guide intermediate reasoning steps. Existing approaches either rely on outcome-level rewards for sequence-level optimization, which makes precise credit assignment difficult, or depend on externally constructed process supervision, which is costly and difficult to scale sustainably. To address this, we propose a new perspective: reinforcement learning for reasoning can be understood as the problem of internalizing outcome supervision into process supervision. From this perspective, we introduce a supervision-internalization method for reinforcement learning for reasoning, enabling the model to automatically extract process-level learning signals through identifying, correcting, and reusing failed reasoning trajectories, thereby achieving finer-grained policy optimization under outcome-only supervision. We further abstract this idea into a new training paradigm, in which the model continually generates and refines its own internal process supervision during reinforcement learning, opening a new path for fine-grained credit assignment in reinforcement learning for reasoning that differs from externally provided process supervision.
>
---
#### [replaced 063] E = T*H/(O+B): A Dimensionless Control Parameter for Mixture-of-Experts Ecology
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出E = T*H/(O+B)作为混合专家模型的控制参数，用于判断专家生态是否健康。解决MoE模型中专家失效问题，通过实验验证其有效性。**

- **链接: [https://arxiv.org/pdf/2605.06415](https://arxiv.org/pdf/2605.06415)**

> **作者:** Qingjun Zhang
>
> **备注:** 12 experiments, 11,000+ training epochs, cross-modal validation (vision + language). Extended version of the Claude-in-the-Loop ecology framework
>
> **摘要:** We introduce E = T*H/(O+B), a dimensionless control parameter that predicts whether Mixture-of-Experts (MoE) models will develop a healthy expert ecology or collapse into dead experts. E combines four hyperparameters -- routing temperature T, routing entropy weight H, oracle weight O, and balance weight B -- into a single quantity. Through 12 controlled experiments (8 vision, 4 language) totaling over 11,000 training epochs, we establish that E >= 0.5 alone is sufficient to guarantee zero dead experts, removing the necessity for handcrafted load-balancing auxiliary losses. We validate this cross-modally on CIFAR-10, CIFAR-100, TinyImageNet-200, WikiText-2, and WikiText-103. Six additional findings emerge: (1) dead experts can resuscitate -- triggered by balance loss driving router re-exploration; (2) ortho toxicity is dataset-dependent, not universal; (3) task complexity shifts the critical E threshold; (4) model overfitting is decoupled from expert ecological health; (5) three-tier MoE spontaneously collapses into a two-tier functional structure; (6) ecological structure is temperature-invariant across a 50x range. We propose that E serves as a unified diagnostic for MoE training, analogous to the Reynolds number in fluid dynamics.
>
---
#### [replaced 064] STAPO: Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，旨在解决大语言模型训练中的稳定性问题。针对稀有干扰标记导致的性能崩溃，提出STAPO框架，通过抑制这些标记的梯度扰动，提升模型推理稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2602.15620](https://arxiv.org/pdf/2602.15620)**

> **作者:** Shiqi Liu; Zeyu He; Guojian Zhan; Letian Tao; Zhilong Zheng; Jiang Wu; Yinuo Wang; Yang Guan; Kehua Sheng; Bo Zhang; Keqiang Li; Jingliang Duan; Shengbo Eben Li
>
> **摘要:** Reinforcement Learning (RL) has significantly improved large language model reasoning, but existing RL fine-tuning methods rely heavily on heuristic techniques such as entropy regularization and reweighting to maintain stability. In practice, they often suffer from late-stage performance collapse, leading to degraded reasoning quality and unstable training. We identify a key factor behind this instability: a small fraction of tokens, termed spurious tokens (around 0.01%), which contribute little to the reasoning outcome but receive disproportionately amplified gradient updates due to inheriting the full sequence-level reward. We present a unified framework for evaluating token-level optimization impacts across spurious risk, gradient norms, and entropy changes. Building on the analysis of token characteristics that severely disrupt optimization, we propose the Silencing Spurious Tokens (S2T) mechanism to efficiently suppress their gradient perturbations. Incorporating this mechanism into a group-based objective, we propose Spurious-Token-Aware Policy Optimization (STAPO), which promotes stable and effective large-scale model refinement. Across six mathematical reasoning benchmarks using Qwen 1.7B, 8B, and 14B base models, STAPO consistently demonstrates superior entropy stability and achieves an average performance improvement of 11.49% ($\rho_{\mathrm{T}}$=1.0, top-p=1.0) and 3.73% ($\rho_{\mathrm{T}}$=0.7, top-p=0.9) over GRPO, 20-Entropy, and JustRL.
>
---
#### [replaced 065] Multilingual OCR-Aware Fine-Tuning and Prompt-Guided Chain-of-Thought Reasoning for Multimodal Large Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文针对多模态大语言模型中的OCR和多语言理解问题，提出一种结合OCR感知训练和链式思维提示的框架，提升模型在复杂视觉条件下的文本识别与理解能力。**

- **链接: [https://arxiv.org/pdf/2605.16409](https://arxiv.org/pdf/2605.16409)**

> **作者:** Qinwu Xu; Yifan Jiang; Haoyu Ren
>
> **摘要:** Optical character recognition (OCR) and multilingual text understanding remain major failure modes of multimodal large language models (MLLMs), particularly in real-world images containing cluttered layouts, small fonts, blur, occlusion, and complex typography. We present an OCR-aware multilingual multimodal training framework that combines (i) large-scale synthetic OCR-to-translation data generation, (ii) OCR-aware supervised fine-tuning (SFT) with LoRA adaptation, and (iii) structured visual chain-of-thought (CoT) prompting for reasoning under uncertain visual conditions. Using a LLaMA-based multimodal architecture, the proposed framework substantially improves OCR completeness, multilingual translation accuracy, and robustness under degraded visual conditions. Experimental results on multilingual receipts, menus, posters, signs, handwritten text, and document images demonstrate significantly improved visual-text grounding compared with the baseline model. In particular, the proposed OCR-aware post-training framework improves extraction of small, blurred, spatially scattered, and partially occluded text while reducing reliance on language priors under uncertain OCR conditions. Qualitative comparisons with frontier multimodal systems, including GPT-5-class and Gemini-family models, further suggest improved OCR grounding and reduced hallucination under noisy and visually ambiguous OCR scenarios. Overall, the results indicate that data-centric OCR-aware multimodal post-training provides an effective and scalable direction for improving multilingual OCR and OCR-based visual question answering systems.
>
---
#### [replaced 066] Ineffectiveness for Search and Undecidability of PCSP Meta-Problems
- **分类: cs.CC; cs.CL; cs.DS; cs.LO**

- **简介: 该论文研究PCSP的搜索与决策问题等价性，分析现有算法在搜索任务中的无效性，并证明相关元问题的不可判定性。**

- **链接: [https://arxiv.org/pdf/2504.04639](https://arxiv.org/pdf/2504.04639)**

> **作者:** Alberto Larrauri
>
> **摘要:** It is an open question whether the search and decision versions of promise CSPs are equivalent. Most known algorithms for PCSPs solve only their \emph{decision} variant, and it is unknown whether they can be adapted to solve \emph{search} as well. The main approaches, called BLP, AIP and BLP+AIP, handle a PCSP by finding a solution to a relaxation of some integer program. We prove that rounding those solutions to a proper search certificate can be as hard as any problem in the class TFNP. In other words, these algorithms are ineffective for search. Building on the algebraic approach to PCSPs, we find sufficient conditions that imply ineffectiveness for search. Our tools are tailored to algorithms that are characterized by minions in a suitable way, and can also be used to prove undecidability results for meta-problems. This way, we show that the families of templates solvable via BLP, AIP, and BLP+AIP are undecidable. Using the same techniques we also analyze several algebraic conditions that are known to guarantee the tractability of finite-template CSPs. We prove that several meta-problems related to cyclic polymorphims and WNUs are undecidable for PCSPs. In particular, there is no algorithm deciding whether a finite PCSP template (1) admits cyclic a polymorphism, (2) admits a WNU.
>
---
#### [replaced 067] The Growing Pains of Frontier Models: When Leaderboards Stop Separating and What to Measure Next
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大模型能力演化，解决模型性能评估与能力平衡问题。通过分析基准测试得分，揭示模型能力协同与权衡规律，提出诊断方法与预测框架。**

- **链接: [https://arxiv.org/pdf/2605.18840](https://arxiv.org/pdf/2605.18840)**

> **作者:** Adil Amin
>
> **备注:** 13 pages, 5 figures, 4 tables. Companion paper: "Lying Is Just a Phase: The Hidden Alignment Transition in Language Model Scaling." ( this https URL ). Code: this https URL . Dashboard: this https URL
>
> **摘要:** Leaderboards rank frontier models on independent axes but do not reveal whether capabilities reinforce or trade off across releases -- and at the frontier, this interaction is the more informative signal. We decompose paired SWE-bench and GPQA Diamond scores into a population coupling trend and per-release residual ($h$-field) that diagnoses capability emphasis from two public benchmark scores. Across 34 models from 10 labs (2024--2026), capabilities cooperate ($r = +0.72$, $p < 10^{-6}$), but cooperation varies systematically: per-lab coupling slopes span $5\times$ (Google $1.15$ vs. DeepSeek $0.23$), and labs pivot -- DeepSeek reversed from reasoning-rich to coding-first ($\Delta h = 15.9$~pp); Anthropic oscillates between coding excursions and recovery. The population regression serves as an isocline phase boundary: the same $\sqrt{(a/b)\cdot B_1}$ classifier that identifies the base-scale coupling transition [Amin, 2026] classifies frontier models and already detects mixed-phase behavior at the next transition (two models below the GPQA--IFEval isocline). The $h$-field is not just diagnostic -- it tells you what to change. Pretraining establishes coupling at $0.871$ while RLHF adds $0.081$ [Amin, 2026]: pretraining-level shifts are permanent (DeepSeek's four-release reversal persists), post-training shifts are reversible (Anthropic's three coding excursions each recover within one release), and inference compute alone shifts $h$ by $+7.8$~pp without retraining. Knowing which component dominates determines whether to retrain or wait. We provide a three-step diagnostic (locate, classify, predict), a per-lab measurement-priority table, and seven falsifiable predictions with timestamped criteria. Five post-cutoff releases fall within the 95\% prediction interval. Code, data, and an interactive dashboard: this https URL.
>
---
#### [replaced 068] FlowPlan-G2P: A Structured Generation Framework for Transforming Scientific Papers into Patent Descriptions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于将科学论文转换为专利描述的任务，解决两者在结构和逻辑上的差异问题。提出FlowPlan-G2P框架，通过分阶段生成实现合法合规的专利描述。**

- **链接: [https://arxiv.org/pdf/2601.02589](https://arxiv.org/pdf/2601.02589)**

> **作者:** Kris W Pan; Yongmin Yoo
>
> **摘要:** Generating patent descriptions from scientific papers is challenging due to fundamental rhetorical and structural disparities between the two genres. Existing approaches treat this as surface-level rewriting, failing to capture the hierarchical reasoning and statutory constraints inherent in patent drafting. We propose FlowPlan-G2P, a graph-mediated generation framework that decomposes this transformation into three stages: (1) Concept Graph Induction, extracting technical entities and functional dependencies into a directed graph; (2) Section-level Planning, partitioning the graph into coherent subgraphs aligned with canonical patent sections; and (3) Graph-Conditioned Generation, synthesizing legally compliant paragraphs conditioned on section-specific subgraphs. Experiments on expert-validated benchmarks reveal that standard NLG metrics systematically favor legally non-compliant outputs over valid patent descriptions, motivating our domain-specific evaluation. Under this evaluation, FlowPlan-G2P with an open-weight backbone consistently outperforms vanilla proprietary models, demonstrating that structured decomposition is a stronger determinant of quality than model scale.
>
---
#### [replaced 069] Structural Abstraction as an Inductive Bias for Non-Stationary Language Model Training
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型训练任务，旨在解决非平稳环境下的灾难性遗忘问题。通过引入结构抽象作为归纳偏置，提升模型的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.17198](https://arxiv.org/pdf/2603.17198)**

> **作者:** Elnaz Rahmati; Nona Ghazizadeh; Zhivar Sourati; Nina Rouhani; Morteza Dehghani
>
> **摘要:** A foundational principle in cognitive science holds that intelligent agents do not learn by storing experiences as isolated instances, but by forming abstract schemas that capture relational structure shared across situations. Even though this claim is well supported by behavioral and neuroimaging studies, its role as a computational training signal in language models remains underexplored. We target this gap in the setting of non-stationary language model training, asking does biasing learning toward structural abstraction reduce catastrophic interference and improve relational generalization as predicted by human results? To study this question, we introduce Abstraction-Augmented Training (AAT), a lightweight loss-level modification that jointly optimizes over concrete instances and their structural abstractions, and two benchmarks, the Relational Cycle Benchmark (RCB) and the Narrative Abstraction Benchmark (NAB). These resources operationalize core cognitive constructs: entity masking as a computational analog of relational alignment, and proverbs as vehicles for implicit abstract meaning that must be inferred across surface-dissimilar situations. Our empirical results demonstrate that AAT consistently reduces forgetting and improves generalization in a pattern that aligns with cognitive predictions for schema-based learning. Beyond the practical implications for continual learning, these results offer preliminary computational evidence that structural abstraction is a signal for stable learning in non-stationary environments.
>
---
#### [replaced 070] Prefix Teach, Suffix Fade: Local Teachability Collapse in Strong-to-Weak On-Policy Distillation
- **分类: cs.CL**

- **简介: 该论文属于强化学习中的模型蒸馏任务，解决强教师到弱学生策略蒸馏中监督有效性不足的问题。工作提出一种基于局部教ability的轨迹裁剪方法，提升蒸馏效果。**

- **链接: [https://arxiv.org/pdf/2605.13643](https://arxiv.org/pdf/2605.13643)**

> **作者:** Kaiyuan Liu; Ziyuan Zhuang; Yang Bai; Bing Wang; Rongxiang Weng; Jieping Ye
>
> **摘要:** On-policy distillation (OPD) trains a student model on its own rollouts using dense feedback from a stronger teacher. Prior literature suggests that, provided teacher feedback is available, supervising the full sequence of response tokens should monotonically improve performance. However, we demonstrate that this assumption sometimes fails to hold in strong-to-weak OPD settings. While later segments of a generated trajectory may still exhibit a non-zero teacher-student advantage, they frequently lack the local contrast that makes dense feedback effective for prioritizing student learning. We term this failure mode local teachability collapse. The resulting principle is straightforward: supervision should concentrate on trajectory regions where the teacher's feedback remains discriminative, rather than uniformly covering the entire response. We operationalize this principle through a trajectory-specific release rule. This rule measures the teacher's margin over the student's top-$K$ candidate set, aggregates this margin across NLTK-tokenized sentence segments, and truncates dense OPD supervision upon detecting a BIC-style downward change point. Experimental results across strong-to-weak distillation tasks using the Qwen3 model family indicate that this release rule consistently outperforms standard full-trajectory OPD across five in-domain benchmarks at various student scales. Furthermore, compared to baseline distillation methods, our approach better preserves model capabilities on out-of-domain task. These results suggest that effective strong-to-weak OPD requires evaluating not only the availability of teacher guidance but also its local utility, ensuring that the generated feedback remains teachable.
>
---
#### [replaced 071] How do Humans Process AI-generated Hallucination Contents: a Neuroimaging Study
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于认知神经科学任务，旨在研究人类如何处理AI生成的幻觉内容。通过EEG实验，分析人类在验证图像描述时的脑电活动，揭示其认知机制与错误判断的神经差异。**

- **链接: [https://arxiv.org/pdf/2605.16953](https://arxiv.org/pdf/2605.16953)**

> **作者:** Shuqi Zhu; Yi Zhong; Ziyi Ye; Bangde Du; Yujia Zhou; Qingyao Ai; Yiqun Liu
>
> **摘要:** While AI-generated hallucinations pose considerable risks, the underlying cognitive mechanisms by which humans can successfully recognize or be misled by these hallucinations remain unclear. To address this problem, this paper explores humans' neural dynamics to characterize how the brain processes hallucinated content. We record EEG signals from 27 participants while they are performing a verification task to judge the correctness of image descriptions generated by a multi-modal large language model (MLLM). Based on an averaged event-related potential (ERP) study, we reveal that multiple cognitive processes, e.g., semantic integration, inferential processing, memory retrieval, and cognitive load, exhibit distinct patterns when humans process hallucinated versus non-hallucinated content. Notably, neural responses to hallucinations that were misjudged versus correctly judged by human participants showed significant differences. This indicates that misjudged AI-generated hallucinations failed to trigger the standard neurocognitive fact verification pathway.
>
---
#### [replaced 072] CLIF: Concept-Level Influence Functions for Transparent Bottleneck Models
- **分类: cs.CL**

- **简介: 该论文属于可解释性AI任务，旨在解决深度学习模型黑箱问题。通过影响函数分析样本和概念层面的影响，提升NLP模型的透明度与可调试性。**

- **链接: [https://arxiv.org/pdf/2605.19848](https://arxiv.org/pdf/2605.19848)**

> **作者:** Yike Sun; Mingkun Xu; Mu You; Zhongzhi He; Henghua Shen; Zehan Tan; Derek F. Wong; Tao Fang
>
> **备注:** A critical theoretical error invalidates the main results. The independence assumption on concept representations and gradients (Section 3.2, Eq.7) is incorrect, breaking the influence estimation in nonlinear bottleneck layers. This flaw undermines all empirical claims in Sections 4-5. The authors withdraw to prevent dissemination of incorrect findings
>
> **摘要:** In recent years, the black-box nature of deep learning models has limited their application in high-stakes domains such as medical diagnosis and finance, where interpretability is essential. To address this, we propose a novel approach using influence functions to enhance interpretability in NLP models at both the sample and concept levels. Experiments on CEBaB and Yelp datasets show that influence functions effectively identify the most impactful training samples, both helpful and harmful, on model predictions. By adjusting the labels and weights of these samples, we demonstrate that model performance can be restored to baseline levels without retraining, confirming the value of influence functions for efficient data debugging. Furthermore, our concept-level analysis identifies key concepts within Concept Bottleneck Models (CBM) that significantly affect predictions. Modifying these concepts alters model behavior observably, providing clear insights into the decision process.
>
---
#### [replaced 073] Understanding Data Temporality Impact on Large Language Models Pre-training
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究预训练数据顺序对大语言模型时间敏感知识的影响，旨在提升模型的时间准确性。通过对比有序与随机数据预训练效果，验证了有序数据有助于获取更及时、精确的时序知识。**

- **链接: [https://arxiv.org/pdf/2605.22769](https://arxiv.org/pdf/2605.22769)**

> **作者:** Hippolyte Pilchen; Romain Fabre; Franck Signe Talla; Patrick Perez; Edouard Grave
>
> **摘要:** Large language models (LLMs) are typically trained on shuffled corpora, yielding models whose knowledge is frozen at train time and whose temporal grounding remains poorly understood. In this work, we study the impact of pre-training dynamics on the acquisition of time-sensitive factual knowledge, focusing specifically on data ordering. Our main contributions are twofold. First, we introduce a comprehensive benchmark of over 7,000 temporally grounded questions and an evaluation protocol that enables analysis of whether models correctly associate facts with their corresponding time periods. Second, we pretrain 6B-parameter models on temporally ordered Common Crawl snapshots and compare them against standard shuffled pre-training. Our results show that sequentially trained models match shuffled baselines on general language understanding and common knowledge while consistently exhibiting more up-to-date and temporally precise knowledge. Temporally ordered pre-training yields improved factual freshness, while shuffled pre-training peaks on older data, possibly due to increased factual repetition. These findings, along with the release of our code at this https URL , checkpoints, and datasets at this https URL provide a foundation for future research on continual learning for LLMs.
>
---
#### [replaced 074] DeIDClinic: A Risk-Aware Pseudonymization Framework for Clinical Text De-identification and Re-identification Risk Assessment
- **分类: cs.CL**

- **简介: 该论文提出DeIDClinic框架，用于临床文本的去标识化和再识别风险评估。解决敏感数据共享中的隐私保护问题，结合深度学习与风险模型提升去标识化效果。**

- **链接: [https://arxiv.org/pdf/2410.01648](https://arxiv.org/pdf/2410.01648)**

> **作者:** Angel Paul; Dhivin Shaji; Lifeng Han; Warren Del-Pinto; Goran Nenadic; Suzan Verberne
>
> **备注:** Accepted by and Presented at: LEGAL-CALD-Pseudo2026 @LREC2026
>
> **摘要:** The increasing availability of sensitive textual data has created an urgent need for robust de-identification methods that enable compliant data sharing while preserving downstream utility. This paper presents DeID-Clinic, a multi-layered framework for automated pseudonymization and re-identification risk assessment of clinical free-text data. Our approach integrates domain-adapted transformer models, including BioBERT and ClinicalBERT, into the MASK de-identification framework to improve the detection and masking of protected health information (PHI). Beyond entity recognition, we introduce a novel document-level risk assessment module that quantifies residual re-identification risk using a combination of k-anonymity, l-diversity, t-closeness, contextual similarity, and entity co-occurrence analysis. Experiments conducted on the i2b2 2014 de-identification dataset demonstrate strong performance, achieving macro-level F1 scores above 0.96 for several entity categories, while enabling quantitative prioritization of high-risk documents for further review. Our results highlight the effectiveness of combining neural de-identification with explicit risk modeling, supporting privacy-preserving data sharing in sensitive domains. Although evaluated on clinical text, the proposed framework is generalizable to other privacy-critical domains such as legal and administrative documents, where reliable pseudonymization and risk-aware anonymization are essential. Keywords{Automated De-Identification, Risk Assessment, Patient Privacy, Pseudonymization, Personal Health Information}
>
---
#### [replaced 075] DimMem: Dimensional Structuring for Efficient Long-Term Agent Memory
- **分类: cs.CL**

- **简介: 该论文提出DimMem，解决LLM代理长期记忆的结构化问题。通过维度化表示提升记忆效率与准确性，适用于对话记忆管理任务。**

- **链接: [https://arxiv.org/pdf/2605.15759](https://arxiv.org/pdf/2605.15759)**

> **作者:** Wentao Qiu; Haotian Hu; Fanyi Wang; Jinwei Kong; Yu Zhang
>
> **摘要:** Large language model (LLM) agents require long-term memory to leverage information from past interactions. However, existing memory systems often face a fidelity--efficiency trade-off: raw dialogue histories are expensive, while flat facts or summaries may discard the structure needed for precise recall. We propose \textbf{DimMem}, a lightweight dimensional memory framework that represents each memory as an atomic, typed, and self-contained unit with explicit fields such as time, location, reason, purpose, and keywords. This representation exposes the structure needed for dimension-aware retrieval, memory update, and selective assistant-context recall without storing full histories in the model context. Across LoCoMo-10 and LongMemEval-S, DimMem achieves \textbf{81.43\%} and \textbf{78.20\%} overall accuracy, respectively, outperforming existing lightweight memory systems while reducing LoCoMo per-query token cost by \textbf{24\%}. We further show that dimensional memory extraction is learnable by compact models: after fine-tuning on the DimMem schema, a Qwen3-4B extractor surpasses LightMem with GPT-4.1-mini on both benchmarks and reaches performance comparable to, or better than, much larger extractors in key settings. These results suggest that explicit dimensional structuring is an effective and efficient foundation for long-term memory in LLM agents. Code is available at this https URL.
>
---
#### [replaced 076] Future-KL Regularized GRPO: Process-Level Credit Assignment from $f$-Divergence Regularization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Future-KL Regularized GRPO方法，解决LLM后训练中的信用分配问题，通过f-散度正则化改进策略优化，提升性能并降低策略漂移。**

- **链接: [https://arxiv.org/pdf/2601.10201](https://arxiv.org/pdf/2601.10201)**

> **作者:** Jiarui Yao; Ruida Wang; Hao Bai; Tong Zhang
>
> **摘要:** Group Relative Policy Optimization (GRPO) is widely used for critic-free Large Language Model (LLM) post-training, but its KL regularization is usually implemented as a local loss-side token penalty. We show that this misses the policy-gradient signal induced by autoregressive KL regularization. Unlike standard KL-regularized Reinforcement Learning (RL) objectives, GRPO's group normalization induces a non-linear prompt-level utility; for binary verifier rewards, this utility is $2\arcsin\sqrt p$. As a result, reward and KL cannot be fused before normalization without changing the implicit objective. We derive the on-policy gradient of GRPO-style objectives with token-wise $f$-divergence regularization. The reward term recovers the standardized GRPO advantage, while the regularizer term includes a causal future-regularization return-to-go omitted by local KL losses. For reverse KL, this yields a simple future KL correction: add a reverse cumulative sum of per-token log ratios after advantage construction. The resulting method, Future-KL Regularized Policy Optimization (FRPO), requires no critic or extra model passes. On mathematical reasoning tasks, FRPO improves pass@16 in our main large-model setting while maintaining higher entropy and lower policy drift than conventional loss-side KL baselines.
>
---
#### [replaced 077] Agent Primitives: Reusable Latent Building Blocks for Multi-Agent Systems
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文属于多智能体系统任务，旨在解决传统系统任务特定性强、可复用性差的问题。提出Agent Primitives，通过可复用的潜在构建块提升系统效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.03695](https://arxiv.org/pdf/2602.03695)**

> **作者:** Haibo Jin; Peng Kuang; Ye Yu; Xiaopeng Yuan; Haohan Wang
>
> **备注:** 16 pages
>
> **摘要:** While existing multi-agent systems (MAS) can handle complex problems by enabling collaboration among multiple agents, they are often highly task-specific, relying on manually crafted agent roles and interaction prompts, which leads to increased architectural complexity and limited reusability across tasks. Moreover, most MAS communicate primarily through natural language, making them vulnerable to error accumulation and instability in long-context, multi-stage interactions within internal agent histories. In this work, we propose \textbf{Agent Primitives}, a set of reusable latent building blocks for LLM-based MAS. Inspired by neural network design, where complex models are built from reusable components, we observe that many existing MAS architectures can be decomposed into a small number of recurring internal computation patterns. Based on this observation, we instantiate three primitives: Review, Voting and Selection, and Planning and Execution. All primitives communicate internally via key-value (KV) cache, which improves both robustness and efficiency by mitigating information degradation across multi-stage interactions. To enable automatic system construction, an Organizer agent selects and composes primitives for each query, guided by a lightweight knowledge pool of previously successful configurations, forming a primitive-based MAS. Experiments show that primitives-based MAS improve average accuracy by 12.0-16.5\% over single-agent baselines, reduce token usage and inference latency by approximately 3$\times$-4$\times$ compared to text-based MAS, while incurring only 1.3$\times$-1.6$\times$ overhead relative to single-agent inference and providing more stable performance across model backbones.
>
---
#### [replaced 078] AMEL: Accumulated Message Effects on LLM Judgments
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于语言模型评估任务，研究累积信息对模型判断的影响。通过实验发现模型判断受先前对话极性影响，尤其在不确定情况下更明显，提出优化评估流程的建议。**

- **链接: [https://arxiv.org/pdf/2605.22714](https://arxiv.org/pdf/2605.22714)**

> **作者:** Sid-Ali Temkit
>
> **备注:** 19 pages, 14 figures, 6 tables. Single author. Code, data (75,898 deduplicated API responses), and analysis pipeline at this https URL
>
> **摘要:** Large language models are routinely used as automated evaluators: to review code, moderate content, or score outputs, often with many items passing through one conversation. We ask whether the polarity of prior conversation history biases subsequent judgments, an effect we call the accumulated message effect on LLM judgments (AMEL). Across 75,898 API calls to 11 models from 4 providers (OpenAI, Anthropic, Google, and four open-source models), we present identical test items in isolation or following histories saturated with predominantly positive or negative evaluations. Models shift toward the conversation's prevailing polarity (d = -0.17, p < 10^-46). The effect concentrates on items where the model is genuinely uncertain at baseline (d = -0.34 for high-entropy items, vs d = -0.15 when the baseline is deterministic). Bias does not grow with context length: 5 prior turns and 50 produce the same shift (Spearman |r| < 0.01; OLS slope p = 0.80). And there is a negativity asymmetry: paired per item, negative histories induce 1.62x more bias than positive (t = 13.46, p < 10^-39, n = 2,481). Scaling helps but does not solve it (Anthropic: Haiku -0.22 to Opus -0.17; OpenAI: Nano -0.34 to GPT-5.2 -0.17). Three follow-ups narrow the mechanism. The token probability distribution shifts continuously, not at a threshold. The negativity asymmetry has both token-level and semantic components, though attributing the balance is exploratory at our sample sizes. Position does not matter: five biased turns anywhere in a 50-turn history produce the same shift. The simplest fix for evaluation pipelines is a fresh context per item; when batching is unavoidable, balancing the history helps.
>
---
#### [replaced 079] Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于人工智能评估任务，旨在解决工具增强代理的推理轨迹评价问题。提出TRACE框架，无需参考答案即可多维度评估代理的效率、幻觉和适应性。**

- **链接: [https://arxiv.org/pdf/2510.02837](https://arxiv.org/pdf/2510.02837)**

> **作者:** Wonjoong Kim; Sangwu Park; Yeonjun In; Sein Kim; Dongha Lee; Chanyoung Park
>
> **备注:** International Conference on Machine Learning (ICML) 2026
>
> **摘要:** Although recent tool-augmented benchmarks involve complex requests, evaluation remains limited to answer matching, neglecting critical trajectory aspects like efficiency, hallucination, and adaptivity. The most straightforward method for evaluation is to compare an agent's trajectory with the ground-truth, but annotating all valid ground-truth trajectories is prohibitively expensive. In this manner, we introduce TRACE, a reference-free framework for the multi-dimensional evaluation of tool-augmented LLMs. By incorporating an evidence bank which accumulates knowledge from preceding steps, TRACE assesses an agent's reasoning trajectory effectively. To validate our framework, we develop a new meta-evaluation dataset with diverse and flawed trajectories, each labeled with multi-faceted performance scores. Our results confirm that TRACE accurately evaluates complex trajectories even with small open-source LLMs. Furthermore, we apply our method to evaluate the trajectories that agents produce while solving tool-augmented tasks, presenting previously unreported observations and their corresponding insights.
>
---
#### [replaced 080] Sparse Tokens Suffice: Jailbreaking Audio Language Models via Token-Aware Gradient Optimization
- **分类: cs.CR; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于音频语言模型安全研究，解决如何高效进行越狱攻击的问题。通过分析梯度结构，提出稀疏优化方法TAGO，仅保留高梯度区域，提升攻击效率。**

- **链接: [https://arxiv.org/pdf/2605.04700](https://arxiv.org/pdf/2605.04700)**

> **作者:** Zheng Fang; Xiaosen Wang; Shenyi Zhang; Shaokang Wang; Zhijin Ge
>
> **备注:** To appear in the 43rd International Conference on Machine Learning (ICML 2026)
>
> **摘要:** Jailbreak attacks on audio language models (ALMs) optimize audio perturbations to elicit unsafe generations, and they typically update the entire waveform densely throughout optimization. In this work, we investigate the necessity of such dense optimization by analyzing the structure of token-aligned gradients in ALMs. We find that gradient energy is highly non-uniform across audio tokens, indicating that only a small subset of token-aligned audio regions dominates the optimization signal. Motivated by this observation, we propose Token-Aware Gradient Optimization (TAGO), which enables sparse jailbreak optimization by retaining only waveform gradients aligned with audio tokens that have high gradient energy, while masking the remaining gradients at each iteration. Across three ALMs, TAGO outperforms baselines, and substantial sparsification preserves strong attack success rates (e.g. on Qwen3-Omni, $\mathrm{ASR}_{l}$ remains at 86% with a token retention ratio of 0.25, compared to 87% with full token retention). These results demonstrate that dense waveform updates are largely redundant, and we advocate that future audio jailbreak and safety alignment research should further leverage this heterogeneous token-level gradient structure.
>
---
#### [replaced 081] Lean Formalization of Generalization Error Bound by Rademacher Complexity and Dudley's Entropy Integral
- **分类: cs.LG; cs.CL; math.ST**

- **简介: 该论文属于形式化验证任务，旨在通过Rademacher复杂度和Dudley熵积分证明机器学习的泛化误差界。工作包括在Lean 4中形式化相关理论，并应用于线性预测器和熵积分绑定。**

- **链接: [https://arxiv.org/pdf/2503.19605](https://arxiv.org/pdf/2503.19605)**

> **作者:** Sho Sonoda; Kazumi Kasaura; Yuma Mizuno; Kei Tsukamoto; Naoto Onda
>
> **备注:** accepted at ITP2026
>
> **摘要:** Understanding and certifying the generalization performance of machine learning algorithms -- i.e. obtaining theoretical estimates of the test error from the training error -- is a central theme of statistical learning theory. Among the many complexity measures used to derive such guarantees, Rademacher complexity yields sharp, data-dependent bounds that apply well beyond classical VC-dimension theory. In this study, we formalize the generalization error bound by Rademacher complexity in Lean 4, building on measure-theoretic probability theory available in the Mathlib library. Our development provides a mechanically-checked pipeline from the definitions of empirical and expected Rademacher complexity, through a formal symmetrization argument and a bounded-differences analysis, to high-probability uniform deviation bounds via a formally proved McDiarmid inequality. A key technical contribution is a reusable mechanism for lifting results from countable hypothesis classes (where measurability of suprema is straightforward in Mathlib) to separable topological index sets via a reduction to a countable dense subset. As worked applications of the abstract theorem, we mechanize standard empirical Rademacher bounds for linear predictors under $\ell_2$ and $\ell_1$ regularizations, and we also formalize a Dudley-type entropy integral bound based on covering numbers and a chaining construction.
>
---
#### [replaced 082] A Comprehensive Dataset for Human vs. AI Generated Text Detection
- **分类: cs.CL**

- **简介: 该论文属于AI生成文本检测任务，旨在解决识别和归属AI生成内容的问题。工作包括构建包含73,193个样本的标注数据集，并提供基准结果。**

- **链接: [https://arxiv.org/pdf/2510.22874](https://arxiv.org/pdf/2510.22874)**

> **作者:** Rajarshi Roy; Gurpreet Singh; Ashhar Aziz; Shashwat Bajpai; Nasrin Imanpour; Shwetangshu Biswas; Kapil Wanaskar; Parth Patwa; Subhankar Ghosh; Shreyas Dixit; Nilesh Ranjan Pal; Vipula Rawte; Ritvik Garimella; Gaytri Jena; Amitava Das; Amit Sheth; Vasu Sharma; Aishwarya Naresh Reganti; Vinija Jain; Aman Chadha
>
> **备注:** Defactify4 @AAAI 2025
>
> **摘要:** The rapid advancement of large language models (LLMs) has led to increasingly human-like AI-generated text, raising concerns about content authenticity, misinformation, and trustworthiness. Addressing the challenge of reliably detecting AI-generated text and attributing it to specific models requires large-scale, diverse, and well-annotated datasets. In this work, we present a comprehensive dataset comprising over 73,193 text samples that combine authentic New York Times articles with synthetic versions generated by multiple state-of-the-art LLMs including Gemma-2-9b, Mistral-7B, Qwen-2-72B, LLaMA-8B, Yi-Large, and GPT-4-o. The dataset provides original article abstracts as prompts, full human-authored narratives. We establish baseline results for two key tasks: distinguishing human-written from AI-generated text, achieving an accuracy of 58.35\%, and attributing AI texts to their generating models with an accuracy of 8.92\%. By bridging real-world journalistic content with modern generative models, the dataset aims to catalyze the development of robust detection and attribution methods, fostering trust and transparency in the era of generative AI. Our dataset is available at: this https URL
>
---
#### [replaced 083] SentGraph: Hierarchical Sentence Graph for Multi-hop Retrieval-Augmented Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多跳问答任务，旨在解决传统方法在多跳推理中证据链不完整的问题。提出SentGraph框架，通过构建句子级图结构显式建模逻辑关系，提升多文档证据的整合能力。**

- **链接: [https://arxiv.org/pdf/2601.03014](https://arxiv.org/pdf/2601.03014)**

> **作者:** Junli Liang; Pengfei Zhou; Wangqiu Zhou; Wenjie Qing; Qi Zhao; Ziwen Wang; Qi Song; Xiangyang Li
>
> **摘要:** Traditional Retrieval-Augmented Generation (RAG) effectively supports single-hop question answering with large language models but faces significant limitations in multi-hop question answering tasks, which require combining evidence from multiple documents. Existing chunk-based retrieval often provides irrelevant and logically incoherent context, leading to incomplete evidence chains and incorrect reasoning during answer generation. To address these challenges, we propose SentGraph, a sentence-level graph-based RAG framework that explicitly models fine-grained logical relationships between sentences for multi-hop question answering. Specifically, we construct a hierarchical sentence graph offline by first adapting Rhetorical Structure Theory to distinguish nucleus and satellite sentences, and then organizing them into topic-level subgraphs with cross-document entity bridges. During online retrieval, SentGraph performs graph-guided evidence selection and path expansion to retrieve fine-grained sentence-level evidence. Extensive experiments on four multi-hop question answering benchmarks demonstrate the effectiveness of SentGraph, validating the importance of explicitly modeling sentence-level logical dependencies for multi-hop reasoning.
>
---
#### [replaced 084] HALvest-Contrastive: Retrieval-Like Authorship Attribution with Patch-Level Late Interaction
- **分类: cs.DL; cs.CL**

- **简介: 该论文属于作者归属任务，解决主题混淆问题。通过构建HALvest-Contrastive数据集，提出基于片段级后期交互的匹配方法，提升作者识别性能。**

- **链接: [https://arxiv.org/pdf/2407.20595](https://arxiv.org/pdf/2407.20595)**

> **作者:** Francis Kulumba; Wissam Antoun; Guillaume Vimont; Laurent Romary; Florian Cafiero
>
> **备注:** 19 pages, 9 figures. Under review
>
> **摘要:** Authorship attribution asks whether two pieces of text share a writer, but topical confound makes the task deceptively easy: two authors covering the same topic may look more alike than one author covering two topics. Scholarly prose offers a natural remedy, academic writers produce multiple papers on related but distinct topics while maintaining consistent stylistic habits. We introduce HALvest, a 17-billion-token multilingual corpus of open-access academic papers, and its English contrastive derivative HALvest-Contrastive, where same-author passages are drawn from distinct papers within a disciplinary field to minimize topical overlap. We validate our benchmark by showing that a strong lexical baseline collapses once topical shortcuts are removed. On this same benchmark, we revisit how authorship is scored. Standard systems compress each document into a single vector. We instead keep a sequence of vectors and compare them with late interaction, then propose patch-level late interaction, which groups neighboring tokens into patches before matching. Matching at the sequence level greatly improves performance over the single-vector baseline, but the optimal interaction granularity is subtle.
>
---
#### [replaced 085] Probability Distributions Computed by Autoregressive Transformers
- **分类: cs.CL**

- **简介: 该论文研究Transformer作为语言模型生成概率分布的能力，探讨其在自回归生成中的表达能力，解决其与语言识别器的差异问题。**

- **链接: [https://arxiv.org/pdf/2510.27118](https://arxiv.org/pdf/2510.27118)**

> **作者:** Andy Yang; Anej Svete; Jiaoda Li; Anthony Widjaja Lin; Jonathan Rawski; Ryan Cotterell; David Chiang
>
> **备注:** 20 pages
>
> **摘要:** Most expressivity results for transformers treat them as language recognizers -- devices that accept or reject strings -- rather than as they are used in practice: as language models that generate strings autoregressively and probabilistically. We characterize the probability distributions that transformer language models can express. We show that making transformer language recognizers autoregressive can sometimes increase their expressivity, and that making them probabilistic can break equivalences that hold in the non-probabilistic case. Our overall contribution is to tease apart what functions transformers are capable of expressing in their most common use case as language models.
>
---
#### [replaced 086] Scene Abstraction for Lexical Semantics: Structured Representations of Situated Meaning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决词汇意义的语境化表示问题。通过构建场景抽象框架，提取词语在不同语境中的结构化含义，并验证其与人类理解的一致性。**

- **链接: [https://arxiv.org/pdf/2605.22542](https://arxiv.org/pdf/2605.22542)**

> **作者:** Yejin Cho; Katrin Erk
>
> **摘要:** Coffee and tea share many properties, yet they evoke strikingly different situations, atmospheres, and affective associations. These situated dimensions of word meaning are real and systematic, but they remain implicit in most computational representations of lexical meaning. We propose Scene Abstraction, a framework for constructing structured representations of the interpretive scenes that words participate in across usage contexts. Each scene consists of a Contextual Scene (Events, Entities, Setting) and an expression-centered Expression Profile (Engaged events, Generalizable properties, Evoked emotions), operationalized through few-shot prompting of a large language model. Our contributions are three-fold: (1) a structured representation framework for situated lexical meaning; (2) COCA-Scenes, a dataset of 520 usage instances across 26 keywords for distinct scene identification; and (3) empirical evidence from two experiments suggesting that scenes are reliably identifiable across human observers (82.4% accuracy, +11.8 pp over text-only embeddings) and that our scene profiles more closely align with human interpretation of words in context than ATOMIC-based alternatives (86.4% preference across three semantic dimensions).
>
---
#### [replaced 087] MUR: Momentum Uncertainty guided Reasoning for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型推理优化任务，旨在提升推理效率并减少冗余计算。提出MUR方法，通过动量不确定性引导推理，有效降低计算消耗并提高准确率。**

- **链接: [https://arxiv.org/pdf/2507.14958](https://arxiv.org/pdf/2507.14958)**

> **作者:** Hang Yan; Fangzhi Xu; Rongman Xu; Yifei Li; Jian Zhang; Haoran Luo; Xiaobao Wu; Luu Anh Tuan; Haiteng Zhao; Qika Lin; Jun Liu
>
> **摘要:** Large Language Models have achieved impressive performance on reasoning-intensive tasks, yet optimizing their reasoning efficiency remains an open challenge. While Test-Time Scaling (TTS) improves reasoning quality, it often leads to overthinking, wasting tokens on redundant computations. This work investigates how to efficiently and adaptively guide current model' test-time scaling without additional training. Inspired by the concept of momentum in physics, we propose Momentum Uncertainty-guided Reasoning (MUR), which dynamically allocates thinking budgets to critical reasoning steps by tracking and aggregating stepwise uncertainty over time. To support flexible inference-time control, we introduce gamma-control, a simple mechanism that tunes the reasoning budget via a single hyperparameter. We provide in-depth theoretical proof to support the superiority of MUR in terms of stability and biases. MUR is comprehensively evaluated against various TTS methods across four challenging benchmarks (MATH-500, AIME24, AIME25, and GPQA-diamond) using different sizes of recent Qwen3 models (1.7B, 4B, and 8B). Results demonstrate that MUR reduces computation by by over 45% on average while improving accuracy from 0.33 to 3.46%.
>
---
#### [replaced 088] CPMobius: Iterative Coach-Player Reasoning for Data-Free Reinforcement Learning
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出CPMobius，解决数据依赖问题，通过教练-选手协作机制提升推理模型性能，无需外部数据。**

- **链接: [https://arxiv.org/pdf/2602.02979](https://arxiv.org/pdf/2602.02979)**

> **作者:** Ran Li; Zeyuan Liu; Yinghao Chen; Bingxiang He; Jiarui Yuan; Zixuan Fu; Weize Chen; Jinyi Hu; Chen Qian; Zhiyuan Liu; Maosong Sun
>
> **备注:** Accepted to the ICML 2026
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong potential in complex reasoning, yet their progress remains fundamentally constrained by reliance on massive high-quality human-curated tasks and labels, either through supervised fine-tuning (SFT) or reinforcement learning (RL) on reasoning-specific data. This dependence renders supervision-heavy training paradigms increasingly unsustainable, with signs of diminishing scalability already evident in practice. To overcome this limitation, we introduce CPMöbius (CPMobius), a collaborative Coach-Player paradigm for data-free reinforcement learning of reasoning models. Unlike traditional adversarial self-play, CPMöbius, inspired by real world human sports collaboration and multi-agent collaboration, treats the Coach and Player as independent but cooperative roles. The Coach proposes instructions targeted at the Player's capability and receives rewards based on changes in the Player's performance, while the Player is rewarded for solving the increasingly instructive tasks generated by the Coach. This cooperative optimization loop is designed to directly enhance the Player's mathematical reasoning ability. Remarkably, CPMöbius achieves substantial improvement without relying on any external training data, outperforming existing unsupervised approaches. For example, on Qwen2.5-Math-7B-Instruct, our method improves accuracy by an overall average of +4.9 and an out-of-distribution average of +5.4, exceeding RENT by +1.5 on overall accuracy and R-zero by +4.2 on OOD accuracy. Our codebase has been released at this https URL.
>
---
#### [replaced 089] When LLMs Stop Following Steps: A Diagnostic Study of Procedural Execution in Language Models
- **分类: cs.CL**

- **简介: 该论文研究语言模型在执行复杂步骤任务中的表现，旨在检测其是否忠实完成指定流程。通过设计基准测试，发现模型在长步骤任务中准确率显著下降，揭示其在长期执行中的不足。**

- **链接: [https://arxiv.org/pdf/2605.00817](https://arxiv.org/pdf/2605.00817)**

> **作者:** Sailesh Panda; Pritam Kadasi; Abhishek Upperwal; Mayank Singh
>
> **备注:** 86 pages, 124 figures, 4 Tables
>
> **摘要:** Large language models (LLMs) often achieve strong performance on reasoning benchmarks, but final-answer accuracy alone does not show whether they faithfully execute the procedure specified in a prompt. We introduce a controlled diagnostic benchmark for procedural execution, where models are given a step-wise arithmetic procedure and two numeric inputs, and must return the final computed value. Complexity is varied through procedure length and look-back dependencies over intermediate variables. Average first-answer accuracy drops from 63% on 5-step procedures to 20% on 95-step procedures. Generation-level analysis shows that failures often involve missing answers, premature answers, self-correction after an initial error and under-executed traces. These findings suggest that apparent reasoning ability can mask substantial weaknesses in faithful long-horizon procedural execution.
>
---
#### [replaced 090] SURGE: On the Potential of Large Language Models as General-Purpose Surrogate Code Executors
- **分类: cs.LG; cs.CL**

- **简介: 该论文探讨将大语言模型作为通用代理代码执行器的可行性，属于代码执行预测任务。研究构建了SURGE基准，涵盖多种编程场景，评估LLMs的性能与效率。**

- **链接: [https://arxiv.org/pdf/2502.11167](https://arxiv.org/pdf/2502.11167)**

> **作者:** Bohan Lyu; Siqiao Huang; Zichen Liang
>
> **摘要:** Neural surrogate models are powerful and efficient tools in data mining. Meanwhile, large language models (LLMs) have demonstrated remarkable capabilities in code-related tasks, such as generation and understanding. However, an equally important yet underexplored question is whether LLMs can serve as surrogate models for code execution prediction. To systematically investigate it, we introduce SURGE, a comprehensive benchmark with $1160$ problems covering $8$ key aspects: multi-language programming tasks, competition-level programming problems, repository-level code analysis, high-cost scientific computing, time-complexity-intensive algorithms, buggy code analysis, programs dependent on specific compilers or execution environments, and formal mathematical proof verification. Through extensive analysis of $21$ open-source and proprietary LLMs, we examine scaling laws, data efficiency, and predictive accuracy. Our findings reveal important insights about the feasibility of LLMs as efficient surrogates for computational processes. The benchmark and evaluation framework are available at this https URL.
>
---
#### [replaced 091] ToolRegistry: A Protocol-Agnostic Tool Management Library for Function-Calling LLMs
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出ToolRegistry，解决LLM工具调用协议不统一的问题，通过RPC机制实现跨协议的工具管理，简化集成并提升效率。**

- **链接: [https://arxiv.org/pdf/2507.10593](https://arxiv.org/pdf/2507.10593)**

> **作者:** Peng Ding; Rick Stevens
>
> **备注:** 16 pages, 4 figures, v3: add co-author, permission system, progressive tool disclosure, think-augmented calling, RPC framing, multi-provider support
>
> **摘要:** Every LLM tool call is structurally an RPC -- a function name, JSON arguments, and a serialized result -- yet each protocol (native Python, MCP, OpenAPI, LangChain) is integrated from scratch. We present ToolRegistry, a system that makes this RPC nature explicit: a single Tool object acts as a universal stub regardless of transport, while the registry serves as the RPC client runtime for dispatch, schema generation, and execution. The system ships as three packages -- a core registry, a server exposing tools over MCP and OpenAPI, and a hub of production-ready implementations -- and invokes tools through pluggable thread or process backends. The system now also provides tag-based permission policies, BM25F-powered progressive tool disclosure for large registries, think-augmented function calling, multi-provider schema support (OpenAI, Anthropic, Gemini), declarative JSONC/YAML configuration, and a near-zero-dependency core built on stdlib-only vendored modules. In our benchmarks the library cuts integration code by 60-80%, and choosing the right concurrency mode (thread vs. process) yields up to 3.1x throughput over the alternative for a given workload. ToolRegistry is open-source at this https URL documentation lives at this https URL.
>
---
#### [replaced 092] Quality-Conditioned Agreement in Automated Short Answer Scoring: Mid-Range Degradation and the Impact of Task-Specific Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 论文研究自动化短答案评分任务，探讨大语言模型在中等质量回答上的评分一致性问题。发现模型在中等质量回答上表现下降，强调任务适配与评分公平性的重要性。**

- **链接: [https://arxiv.org/pdf/2605.07647](https://arxiv.org/pdf/2605.07647)**

> **作者:** Abigail Victoria Gurin Schleifer; Moriah Ariely; Beata Beigman Klebanov; Asaf Salman; Giora Alexandron
>
> **备注:** PRE-PRINT VERSION Accepted to ACL 21st Workshop on Innovative Use of NLP for Building Educational Applications (BEA26)
>
> **摘要:** Automated short answer scoring (ASAS) is shifting from discriminative, fine-tuned models to large language models (LLMs) used in few-shot settings. This paradigm leverages LLMs broad world knowledge and ease of deployment, but limited task-specific data may reduce alignment on complex scoring tasks. In particular, its impact on scoring partially correct responses that require nuanced interpretation remains underexplored. We investigate the relationship between the degree of task-specific adaptation of different models and quality-conditioned scoring agreement. We compare three LLMs (GPT-5.2, GPT-4o, Claude Opus 4.5) in few-shot mode, a fine-tuned BERT-based encoder, and a human expert on two open-ended biology items, using several hundred student responses and ground truth scores provided by a biology education expert. The results show that human-human agreement is highest and stable across the full quality spectrum. All AI models perform well on fully correct and fully incorrect responses, but exhibit substantial degradation on mid-range responses. This mid-range degradation is conditioned on task-specific adaptation: It is most severe in few-shot LLMs with few examples and decreases as task-specific data increases, with fine-tuned encoder models performing best. This mid-range degradation may lead to inequitable evaluation of responses produced by students with developing understanding. Our findings highlight the importance of quality-conditioned fairness, with particular attention to mid-range responses.
>
---
#### [replaced 093] Language-Switching Triggers Take a Latent Detour Through Language Models
- **分类: cs.CL**

- **简介: 该论文研究语言模型的后门攻击，揭示了语言切换触发器的工作机制，通过分析模型内部流程提出防御思路。**

- **链接: [https://arxiv.org/pdf/2605.18646](https://arxiv.org/pdf/2605.18646)**

> **作者:** Francis Kulumba; Wissam Antoun; Théo Lasnier; Benoît Sagot; Djamé Seddah
>
> **备注:** 15 pages, 16 figures. Under review
>
> **摘要:** Backdoor attacks on language models pose a growing security concern, yet the internal mechanisms by which a trigger sequence hijacks model computations remain poorly understood. We identify a circuit underlying a language-switching backdoor in an 8B-parameter autoregressive language model, where a three-word Latin trigger (nine tokens) redirects English output to French. We decompose the circuit into three phases: (1) distributed attention heads at early layers compose the trigger tokens into the last sequence position; (2) the resulting signal propagates through mid-layers in a subspace orthogonal to the model's natural language-identity direction; (3) the MLP at the final layer converts this latent signal into French logits. The entire circuit flows through a serial bottleneck at a single position: corrupting that position at any layer entirely mitigates the trigger but also hinders the model's capabilities. The orthogonal latent encoding suggests that defenses that search for language-like signals in intermediate representations would miss this trigger entirely.
>
---
#### [replaced 094] Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment
- **分类: cs.CL**

- **简介: 该论文属于知识蒸馏任务，解决教师轨迹与学生模型匹配问题。提出RSR度量，评估推理轨迹的适配性，提升学生推理能力。**

- **链接: [https://arxiv.org/pdf/2601.14249](https://arxiv.org/pdf/2601.14249)**

> **作者:** Yuming Yang; Mingyoung Lai; Wanxu Zhao; Xiaoran Fan; Zhiheng Xi; Mingqi Wu; Chiyue Huang; Jun Zhao; Haijun Lv; Jian Tong; Yunhua Zhou; Yicheng Zou; Qipeng Guo; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **备注:** Accepted to ACL 2026 (Main Conference). 31 pages. Project page: this https URL
>
> **摘要:** Long chain-of-thought (CoT) trajectories provide rich supervision signals for distilling reasoning from teacher to student LLMs. However, both prior work and our experiments show that trajectories from stronger teachers do not necessarily yield better students, highlighting the importance of data-student suitability in distillation. Existing methods assess suitability primarily through student likelihood, favoring trajectories that align closely with the student model's current behavior but overlooking more informative ones. Addressing this, we propose Rank-Surprisal Ratio (RSR), a simple metric that captures both alignment and informativeness to assess the suitability of a reasoning trajectory. RSR is motivated by the observation that effective trajectories typically balance learning signal strength and behavioral alignment by combining low absolute probability with relatively high-ranked tokens under the student model. Concretely, RSR is defined as the ratio of a trajectory's average token-wise rank to its average negative log-likelihood, and is straightforward to compute and interpret. Across five student models and reasoning trajectories from 11 diverse teachers, RSR strongly correlates with post-training reasoning performance (average Spearman 0.86), consistently outperforming existing metrics. We further demonstrate its practical utility in both trajectory selection and teacher selection.
>
---
#### [replaced 095] Auditing medical multi-agent AI reveals risks of false consensus
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于医疗AI评估任务，旨在解决多智能体系统协作过程中的安全问题。通过构建审计框架，检测并量化协作失败模式，提升系统透明度与可靠性。**

- **链接: [https://arxiv.org/pdf/2510.10185](https://arxiv.org/pdf/2510.10185)**

> **作者:** Yinghao Zhu; Lei Gu; Zixiang Wang; Haoran Sang; Dehao Sui; Wen Tang; Lan Mi; Yasha Wang; Junyi Gao; Liang Yao; Tianfan Fu; Ewen Harrison; Lequan Yu
>
> **备注:** Code and Data: this https URL
>
> **摘要:** Large language models are increasingly being assembled into medical multi-agent systems that emulate multidisciplinary consultation through specialist roles, peer review and consensus formation. In clinical decision support, however, apparent consensus is not enough. Clinicians also need to know whether agents checked the evidence, addressed disagreement and kept uncertainty visible. Current evaluations largely score final accuracy, leaving the safety of the collaborative process untested. Here we introduce MedAgentAudit, a clinically grounded workflow audit framework for diagnosing and quantifying collaborative failure modes in medical multi-agent systems. From 3,600 execution logs, we derive an expert-validated taxonomy of ten recurrent failures spanning task comprehension, collaborative discussion, and synthesis and decision-making. We then deploy an expert-validated automated auditor as non-interventional probes across 14,400 cases, covering six multi-agent architectures, six medical text and vision datasets, and four large language model settings per modality. Across systems, collaboration yields uneven accuracy gains and frequent process failures. Unsupported observations affect 16.63% of cases and propagate downstream. In discussion, agents repeat initial views in 98.42% of cases rather than re-examining evidence, and fail to activate specialist reasoning in 42.73%. During synthesis, final answers often substitute authority or majority count for evidence checking, showing authority bias in 28.76% (rising from 35.30% to 68.75% across rounds), self-contradiction in 18.53%, contradiction neglect in 5.48% and minority suppression in 5.11%. MedAgentAudit reframes medical AI evaluation from output scoring to process-level safety and accountability, providing a practical foundation for transparent, auditable and clinician-supervised agentic systems in medicine.
>
---
#### [replaced 096] Depth Registers Unlock W4A4 on SwiGLU: A Reader/Generator Decomposition
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究W4A4量化在SwiGLU模型中的误差来源，提出Depth Registers方法减少量化损失，解决量化精度下降问题。**

- **链接: [https://arxiv.org/pdf/2604.18128](https://arxiv.org/pdf/2604.18128)**

> **作者:** Ziyang Liu
>
> **备注:** The authors have decided to withdraw this version following internal review regarding authorship and contribution agreements
>
> **摘要:** We study post-training W4A4 quantization in a controlled 300M-parameter SwiGLU decoder-only language model trained on 5B tokens of FineWeb-Edu, and ask which input-activation sites dominate the error. Naive round-to-nearest W4A4 collapses validation perplexity from FP16 23.6 to 1727. A simple residual-axis training-time intervention -- Depth Registers with a register-magnitude hinge loss (DR+sink) -- reduces this to 119 (about 14x) at matched FP16 PPL and matched zero-shot capacity, and composes with SmoothQuant to 39.9 PPL. The residual ~2 PPL gap to FP16 is the diagnostic core. We decompose W4A4 damage by input-activation site: the five trainable linears in a SwiGLU block split into residual-axis readers (qkv, w1, w3) and block-internal generators (o_proj, w2). Elementary norm arguments show residual-axis magnitude control bounds readers tightly but leaves w2's bilinear input bounded only by the trivial product of factor bounds; empirically, DR+sink collapses reader kurtosis while leaving generators essentially unchanged, and the reader-rescued W4A4 residue is flat at ~0.28 nats across three matched checkpoints with Delta-remove(w2) dominating. We present DR+sink as a training-time probe rather than a deployment proposal: a post-hoc alternative (Per-Linear QuaRot) nearly matches it on the reader axis. Full QuaRot -- adding online per-head value Hadamard plus online w2-input rotation -- does not close the gap either, directly testing the prediction that orthogonal rotation cannot bound the bilinear SwiGLU tail. Claims are specific to our 300M, 5B-token, single-seed setting, and our experiments do not isolate the partition from the hinge.
>
---
#### [replaced 097] Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Agent World Model（AWM），用于生成合成环境以解决强化学习中环境多样性不足的问题。通过代码驱动的环境，提升训练效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2602.10090](https://arxiv.org/pdf/2602.10090)**

> **作者:** Zhaoyang Wang; Canwen Xu; Boyi Liu; Yite Wang; Siwei Han; Zhewei Yao; Huaxiu Yao; Yuxiong He
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Recent advances in large language model (LLM) have empowered autonomous agents to perform multi-turn interactions with tools and environments. However, scaling such agent training is limited by the lack of diverse and reliable environments. In this paper, we propose Agent World Model (AWM), a fully synthetic environment generation pipeline. Using this pipeline, we scale to 1,000 environments covering everyday scenarios, in which agents can interact with rich toolsets and obtain high-quality observations. Notably, these environments are code-driven and backed by databases, providing more reliable and consistent state transitions than environments simulated by LLMs. Moreover, they enable more efficient agent interaction compared with collecting trajectories from realistic environments. To demonstrate the effectiveness of this resource, we perform large-scale reinforcement learning for multi-turn tool-use agents. Thanks to the fully executable environments and accessible database states, we can also design reliable reward functions. Experiments on three benchmarks show that training exclusively in synthetic environments, rather than benchmark-specific ones, yields strong out-of-distribution generalization. The code is available at this https URL.
>
---
#### [replaced 098] Lying Is Just a Phase: The Hidden Alignment Transition in Language Model Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究语言模型规模与推理、真实性能力的关系，揭示了能力耦合的相变现象，提出通过架构、数据等调整优化模型对齐。任务为模型对齐，解决能力随规模变化的非线性问题。**

- **链接: [https://arxiv.org/pdf/2605.18838](https://arxiv.org/pdf/2605.18838)**

> **作者:** Adil Amin
>
> **备注:** 15 pages, 8 figures, 2 tables. Companion paper: "The Growing Pains of Frontier Models: When Leaderboards Stop Separating and What to Measure Next." ( this https URL). Code: this https URL. Dashboard: this https URL
>
> **摘要:** Scaling laws predict loss from compute but not how capabilities interact. We measure the coupling between reasoning and truthfulness across 63 base models from 16 families and find a regime change invisible to loss curves: below a family-dependent critical scale $N_c$, capabilities anticorrelate; above it, they cooperate. $N_c \approx 3.5$B parameters [2.9B, 13.4B] (bootstrap 95\% CI), but model size is not the only variable that determines phase. Architecture, data curation, and training recipe each shift $N_c$ independently: curated training eliminated the coupling dip between Qwen generations (0.025 $\to$ 0.830 at matched scale), Gemma-4 at 4B achieves coupling 0.871, characteristic of 13B+ standard-trained models, through distillation and architectural innovation, and Phi at 1B matches web-trained coupling at 10B through data curation alone. Width normalization eliminates the anticorrelation across all tested families, supporting an output-projection bottleneck. Internally, 38 of 40 models show zero competing attention heads. A sparse-regression ODE cross-predicts held-out Llama-2 at 5.6\% error. The diagnostic requires no model internals -- only public benchmark scores across a model family. The cooperative regime extends to the frontier ($r = +0.72$, 34 models, 10 labs). A proof-of-concept intervention confirms the bottleneck is exploitable: adding a single truth-direction vector at the identified layer corrects 60\% of misaligned outputs in the tax phase with zero retraining -- a surgical, per-inference correction that requires no weight modification. Code, data, an open-source steering CLI for any open-weight model, and an interactive dashboard for phase diagnosis are released: this https URL.
>
---
#### [replaced 099] Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.IT; cs.LG; cs.NE**

- **简介: 该论文针对加速文本生成任务，解决MDLMs在并行解码时效率下降的问题，提出DUS方法通过分组并行解码提升速度。**

- **链接: [https://arxiv.org/pdf/2506.19037](https://arxiv.org/pdf/2506.19037)**

> **作者:** Omer Luxembourg; Haim Permuter; Eliya Nachmani
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Masked diffusion language models (MDLMs) promise fast, non-autoregressive text generation, yet existing samplers, which pick tokens to unmask based on model confidence, ignore interactions when unmasking multiple positions in parallel and effectively reduce to slow, autoregressive behavior. We propose the Dilated Unmasking Scheduler (DUS), an inference-only, planner-model-free method that partitions sequence positions into non-adjacent dilated groups and unmasks them in parallel so as to minimize an upper bound on joint entropy gain at each denoising step. By explicitly trading off the number of network calls against generation quality, DUS recovers most of the performance lost under traditional parallel unmasking strategies. Across math (GSM8K, MATH500), code (HumanEval, MBPP), general-knowledge (BBH, MMLU-Pro), and instruction following (IFEval) benchmarks, DUS outperforms confidence-based planners and turns the diffusion-specific quality-speed trade-off into a deterministic, predictable speedup set by the block size $B$, yielding up to $5.8\times$ wall-clock speedup over token-by-token MDLM decoding without modifying the underlying denoiser. Applied as a drop-in post-filter, dilated spacing also improves adaptive samplers. Code is available at this https URL.
>
---
#### [replaced 100] Schema-Grounded LLM Extraction for FHIR Patient Digital Twins
- **分类: cs.CL**

- **简介: 该论文属于医疗信息提取任务，旨在从非结构化病历中构建符合FHIR标准的患者数字孪生。通过约束生成和验证修复机制，提升数据有效性与临床实用性。**

- **链接: [https://arxiv.org/pdf/2601.05847](https://arxiv.org/pdf/2601.05847)**

> **作者:** Rafael Brens; Yuqiao Meng; Luoxi Tang; Zhaohan Xi
>
> **摘要:** We revisit the problem of constructing interoperable patient digital twins from unstructured electronic health records (EHRs) and argue that the task is better cast not as a cascade of extraction modules but as constrained generation of a valid FHIR bundle. We introduce SG-LLM, a schema-grounded LLM extractor that (i) augments the prompt with candidate SNOMED-CT, RxNorm, and LOINC codes retrieved through a SapBERT index, (ii) decodes under a JSON Schema derived directly from FHIR R4 StructureDefinitions, and (iii) closes a validator-in-the-loop repair stage whose diagnostics are fed back as structured error messages. We argue that the twin's usefulness, not only span-level F1, is the right object of evaluation, and operationalize this with a clinical-utility experiment that measures the gap in 30-day readmission AUROC between classifiers trained on SG-LLM-generated FHIR bundles versus expert-curated ones. On MIMIC-IV and n2c2 2018 Track 2 benchmarks, SG-LLM matches or exceeds strong joint-extraction and vanilla-LLM baselines while producing substantially more valid bundles. Ablations isolate the contributions of retrieval, schema constraint, and the repair loop. All code, prompts, and schemas are released.
>
---
#### [replaced 101] Persona-Model Collapse in Emergent Misalignment
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文研究AI模型在特定训练后出现的对齐问题，通过测试模型角色扮演时的道德反应，揭示其角色模拟能力下降，属于模型对齐任务。**

- **链接: [https://arxiv.org/pdf/2605.12850](https://arxiv.org/pdf/2605.12850)**

> **作者:** Davi Bastos Costa; Renato Vicente
>
> **备注:** 23 pages, 7 figures, 7 tables; NeurIPS 2026 submission; Corrected code repository URL
>
> **摘要:** Fine-tuning large language models on narrow data with harmful content produces broadly misaligned behavior on unrelated prompts, a phenomenon known as emergent misalignment. We propose that emergent misalignment involves persona-model collapse: deterioration of the model's internal capacity to simulate, differentiate, and maintain consistent characters. We test this hypothesis behaviorally using two metrics: moral susceptibility (S) and moral robustness (R), computed from the across- and within-persona variability of models' Moral Foundations Questionnaire responses under persona role-play. These metrics formalize the model's ability to differentiate characters (S) and its consistency when simulating a given one (R). We evaluate four frontier models (DeepSeek-V3.1, GPT-4.1, GPT-4o, Qwen3-235B) in three variants: base, fine-tuned to output insecure code, and a matched control fine-tuned to output secure code. Across the four models, insecure fine-tuning produces an average $55\%$ increase in S, pushing all four insecure variants beyond the band observed across 13 frontier models benchmarked in prior work -- with GPT-4o reaching more than twice the band's upper end -- signaling dysregulated differentiation. It also causes an average $65\%$ decrease in R, equivalent to a $304\%$ increase in 1/R. By contrast, the matched secure control preserves S near the base and induces only a partial R loss, showing that these effects are largely misalignment-specific. Complementing these metric shifts, insecure variants' unconditioned responses converge toward saturation near the scale ceiling, departing markedly from both base models' structured responses and those elicited when base models role-play toxic personas. Taken together, these metrics provide a sensitive diagnostic for emergent misalignment and serve as behavioral evidence that it involves persona-model collapse.
>
---
#### [replaced 102] SkillOpt: Executive Strategy for Self-Evolving Agent Skills
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SkillOpt，解决代理技能自进化问题。通过文本优化器系统性提升技能，无需额外推理调用，显著提高多个基准的性能。**

- **链接: [https://arxiv.org/pdf/2605.23904](https://arxiv.org/pdf/2605.23904)**

> **作者:** Yifan Yang; Ziyang Gong; Weiquan Huang; Qihao Yang; Ziwei Zhou; Zisu Huang; Yan Li; Xuemei Gao; Qi Dai; Bei Liu; Kai Qiu; Yuqing Yang; Dongdong Chen; Xue Yang; Chong Luo
>
> **备注:** 27 pages, 4 figures, 6 tables
>
> **摘要:** Agent skills today are hand-crafted, generated one-shot, or evolved through loosely controlled self-revision, none of which behaves like a deep-learning optimizer for the skill, and none of which reliably improves over its starting point under feedback. We argue the skill should instead be trained as the external state of a frozen agent, with the same discipline that makes weight-space optimization reproducible. SkillOpt is, to our knowledge, the first systematic controllable text-space optimizer for agent skills: a separate optimizer model turns scored rollouts into bounded add/delete/replace edits on a single skill document, and an edit is accepted only when it strictly improves a held-out validation score. A textual learning-rate budget, rejected-edit buffer, and epoch-wise slow/meta update make skill training stable while adding zero inference-time model calls at deployment. Across six benchmarks, seven target models, and three execution harnesses (direct chat, Codex, Claude Code), SkillOpt is best or tied on all 52 evaluated (model, benchmark, harness) cells and beats every per-cell competitor among human, one-shot LLM, Trace2Skill, TextGrad, GEPA, and EvoSkill skills. On GPT-5.5 it lifts the average no-skill accuracy by +23.5 points in direct chat, by +24.8 inside the Codex agentic loop, and by +19.1 inside Claude Code. Transfer experiments further show that optimized skill artifacts retain value when moved across model scales, between Codex and Claude Code execution environments, and to a nearby math benchmark without further optimization. Code: this https URL
>
---
#### [replaced 103] Hy-MT2: A Family of Fast, Efficient and Powerful Multilingual Translation Models in the Wild
- **分类: cs.CL**

- **简介: 该论文提出Hy-MT2多语言翻译模型，解决复杂现实场景下的翻译任务。通过不同规模模型提升翻译速度与效果，优于现有开源及商业模型。**

- **链接: [https://arxiv.org/pdf/2605.22064](https://arxiv.org/pdf/2605.22064)**

> **作者:** Mao Zheng; Zheng Li; Tao Chen; Bo Lv; Mingrui Sun; Mingyang Song; Jinlong Song; Hong Huang; Decheng Wu; Hai Wang; Yifan Song; Yanfeng Chen; Guanwei Zhang
>
> **摘要:** Hy-MT2 is a family of fast-thinking multilingual translation models designed for complex real-world scenarios. It includes three model sizes: 1.8B, 7B, and 30B-A3B (MoE), all of which support translation among 33 languages and effectively follow translation instructions in multiple languages. Multi-dimensional evaluations show that Hy-MT2 delivers outstanding performance across general, real-world business, domain-specific, and instruction-following translation tasks. The 7B and 30B models outperform open-source models such as DeepSeek-V4-Pro and Kimi K2.6 in fast-thinking mode, while the lightweight 1.8B model also surpasses mainstream commercial APIs from providers such as Microsoft and Doubao overall. Moreover, when paired with AngelSlim's 1.25-bit extreme quantization for on-device deployment, the lightweight 1.8B model requires only 440 MB of storage and achieves a 1.5x inference speedup.
>
---
#### [replaced 104] How Much Do Large Language Model Cheat on Evaluation? Benchmarking Overestimation under the One-Time-Pad-Based Framework
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于模型评估任务，旨在解决大语言模型在基准测试中可能存在的过估计问题。提出ArxivRoll框架，通过动态生成测试用例和衡量基准污染程度，提高评估的公平性和准确性。**

- **链接: [https://arxiv.org/pdf/2507.19219](https://arxiv.org/pdf/2507.19219)**

> **作者:** Zi Liang; Liantong Yu; Shiyu Zhang; Qingqing Ye; Haibo Hu
>
> **备注:** This paper has been accepted by AAAI 2026. We update it for adding new evaluation results for ArxivRollBench-2025a and ArxivRollBench-2026a, with the evaluation of timly models like DeepSeekV4Pro, GPT-5.5, Claude-Opus-4.7, and so on. Source code: this https URL Online Leaderboard Website: this https URL
>
> **摘要:** Overestimation in evaluating large language models (LLMs) has become an increasing concern. Due to the contamination of public benchmarks or imbalanced model training, LLMs may achieve unreal evaluation results on public benchmarks, either intentionally or unintentionally, which leads to unfair comparisons among LLMs and undermines their realistic capability assessments. Existing benchmarks attempt to address these issues by keeping test cases permanently secret, mitigating contamination through human evaluation, or repeatedly collecting and constructing new samples. However, these approaches fail to ensure reproducibility, transparency, and high efficiency simultaneously. Moreover, the extent of overestimation in current LLMs remains unquantified. To address these issues, we propose ArxivRoll, a dynamic evaluation framework inspired by one-time pad encryption in cryptography. ArxivRoll comprises two key components: \emph{i) SCP (Sequencing, Cloze, and Prediction)}, an automated generator for private test cases, and \emph{ii) Rugged Scores (RS)}, metrics that measure the proportion of public benchmark contamination and training bias. Leveraging SCP, ArxivRoll constructs a new benchmark every six months using recent articles from ArXiv and employs them for one-time evaluations of LLM performance. Extensive experiments demonstrate the high quality of our benchmark, and we provide a systematic evaluation of current LLMs. The source code is available at this https URL.
>
---
#### [replaced 105] Axis-Aligned Semantics for ODRL: Resolving Dimensional Ambiguity in Policy Constraints
- **分类: cs.CL; cs.LO**

- **简介: 该论文属于形式化验证任务，解决ODRL政策约束中的维度歧义问题。通过轴分解将多轴操作数转换为轴特定标量，提升冲突检测的准确性。**

- **链接: [https://arxiv.org/pdf/2602.19878](https://arxiv.org/pdf/2602.19878)**

> **作者:** Daham Mustafa; Diego Collarana; Sabrina Kirrane; Christoph Lange; Christoph Quix; Rafiqul Haque; Yixin Peng; Stefan Decker
>
> **备注:** 17 pages. Preprint. v3: expanded benchmark to 256 problems; revised semantics and profile (OAAP)
>
> **摘要:** The Open Digital Rights Language (ODRL) represents policy constraints as triples of a left operand, an operator, and a value. Several spatial operands, however, range over multi-axis domains such as width, height, and depth, while the constraint syntax provides no explicit axis identity. As a result, policy engines cannot determine whether multiple constraints apply to the same axis or different ones, making conflict detection unsound or incomplete. We resolve this ambiguity by axis decomposition, replacing multi-axis operands with axis-specific scalar operands over totally ordered domains. Each constraint then denotes an interval per axis and each policy an axis-aligned box, reducing conflict detection to box comparison. We define a three-valued semantics (Conflict, Compatible, Unknown), prove the decomposition sound and backward compatible with ODRL, instantiate it as ODRL Axis-Aligned Profile (OAAP), and validate it on a benchmark of 256 ODRL policy problems, each expressed in Turtle and compiled to first-order (TPTP) and SMT-LIB form, using Vampire, E, Z3, and cvc5.
>
---
#### [replaced 106] KAME: Tandem Architecture for Enhancing Knowledge in Real-Time Speech-to-Speech Conversational AI
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文属于实时语音对话AI任务，旨在解决实时模型知识不足与延迟系统知识丰富但延迟高的问题。提出KAME架构，结合实时语音模型与后端语言模型，提升响应准确性同时保持低延迟。**

- **链接: [https://arxiv.org/pdf/2510.02327](https://arxiv.org/pdf/2510.02327)**

> **作者:** So Kuroki; Yotaro Kubo; Takuya Akiba; Yujin Tang
>
> **备注:** Published at IEEE ICASSP 2026
>
> **摘要:** Real-time speech-to-speech (S2S) models excel at generating natural, low-latency conversational responses but often lack deep knowledge and semantic understanding. Conversely, cascaded systems combining automatic speech recognition, a text-based Large Language Model (LLM), and text-to-speech synthesis offer superior knowledge representation at the cost of high latency, which disrupts the flow of natural interaction. This paper introduces a novel hybrid architecture that bridges the gap between these two paradigms. Our framework processes user speech through an S2S transformer for immediate responsiveness while concurrently relaying the query to a powerful back-end LLM. The LLM's text-based response is then injected in real time to guide the S2S model's speech generation, effectively infusing its output with rich knowledge without the full latency penalty of a cascaded system. We evaluated our method using a speech-synthesized variant of the MT-Bench benchmark that consists of multi-turn question-answering sessions. The results demonstrate that our system substantially outperforms a baseline S2S model in response correctness, approaching that of a cascaded system, while maintaining a latency on par with the baseline.
>
---
#### [replaced 107] BacktestBench: Benchmarking Large Language Models for Automated Quantitative Strategy Backtesting
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于量化策略回测任务，旨在解决自动化回测的高技术门槛和可扩展性问题。作者构建了BacktestBench基准，并提出AutoBacktest方法，提升回测自动化水平。**

- **链接: [https://arxiv.org/pdf/2605.17937](https://arxiv.org/pdf/2605.17937)**

> **作者:** Zhensheng Wang; Wenmian Yang; Qingtai Wu; Lequan Ma; Yiquan Zhang; Weijia Jia
>
> **备注:** This paper has been accepted by KDD 2026 (Datasets and Benchmarks Track)
>
> **摘要:** Quantitative backtesting is essential for evaluating trading strategies but remains hampered by high technical barriers and limited scalability. While Large Language Models (LLMs) offer a transformative path to automate this complex, interdisciplinary workflow through advanced code generation, tool usage, and agentic planning, the practical realization is significantly challenged by the current lack of a large-scale benchmark dedicated to automated quantitative backtesting, which hinders progress in this field. To bridge this critical gap, we introduce BacktestBench, the first large-scale benchmark for automated quantitative backtesting. Built from over 6 million real market records, it comprises 18,246 meticulously annotated question-answering pairs across four task categories: metrics calculation, ticker selection, strategy selection, and parameter confirmation. We also propose AutoBacktest, a robust multi-agent baseline that translates natural language strategies into reproducible backtests by coordinating a Summarizer for semantic factor extraction, a Retriever for validated SQL generation, and a Coder for Python backtesting implementation. Our evaluation on 23 mainstream LLMs, complemented by targeted ablations, identifies key factors that influence end-to-end performance and highlights the importance of grounded verification and standardized indicator representations.
>
---
#### [replaced 108] Fine-Tuning Language Models to Know What They Know
- **分类: cs.NE; cs.AI; cs.CL; q-bio.NC**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的元认知能力。通过建立度量方法和优化框架，解决模型自我评估不准确的问题。**

- **链接: [https://arxiv.org/pdf/2602.02605](https://arxiv.org/pdf/2602.02605)**

> **作者:** Sangjun Park; Elliot Meyerson; Xin Qiu; Risto Miikkulainen
>
> **备注:** Preprint
>
> **摘要:** Evaluating true metacognition in Large Language Models (LLMs) is difficult due to biases and heuristics. This paper presents a framework to measure and enhance LLM metacognition while controlling for these biases. A measurement method using the $d'_{\rm type2}$ metric is established to isolate metacognitive ability. The Evolution Strategy for Metacognitive Alignment (ESMA) is proposed, demonstrating robust generalization across unseen datasets, languages, and newly acquired knowledge. Finally, parameter analysis reveals that these improvements are driven by a sparse set of parameters, offering new pathways for targeted metacognitive optimization.
>
---
#### [replaced 109] Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文研究 embodied LLMs 的任务规划问题，旨在提升机器人在部署中的反思与学习能力。通过引入反射式测试时规划，结合行动中与行动后的反思机制，提高任务执行效果与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.21198](https://arxiv.org/pdf/2602.21198)**

> **作者:** Yining Hong; Huang Huang; Manling Li; Li Fei-Fei; Leonidas Guibas; Jiajun Wu; Yejin Choi
>
> **摘要:** Embodied LLMs endow robots with high-level task reasoning, but they cannot reflect on what went wrong or why, turning deployment into a sequence of independent trials where mistakes repeat rather than accumulate into experience. Drawing upon human reflective practitioners, we introduce Reflective Test-Time Planning, which integrates two modes of reflection: \textit{reflection-in-action}, where the agent uses test-time scaling to generate and score multiple candidate actions using internal reflections before execution; and \textit{reflection-on-action}, which uses test-time training to update both its internal reflection model and its action policy based on external reflections after execution. We also include retrospective reflection, allowing the agent to re-evaluate earlier decisions and perform model updates with hindsight for proper long-horizon credit assignment. Experiments on our newly-designed Long-Horizon Household benchmark and MuJoCo Cupboard Fitting benchmark show significant gains over baseline models, with zero-shot generalization to photorealistic HM3D environments and real-robot experiments on a Franka Panda arm. Ablations confirm that reflection-in-action and reflection-on-action are mutually dependent, and that retrospective reflection achieves better credit assignment than step-wise external feedback at lower computational overhead. Qualitative analyses further highlight behavioral correction through reflection.
>
---
#### [replaced 110] CogniFold: Always-On Proactive Memory via Cognitive Folding
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出CogniFold，一种类脑主动记忆系统，解决传统代理记忆被动、碎片化的问题。通过构建认知结构，实现主动学习与决策，提升智能助手的自主性。**

- **链接: [https://arxiv.org/pdf/2605.13438](https://arxiv.org/pdf/2605.13438)**

> **作者:** Suli Wang; Yiqun Duan; Yu Deng; Rundong Zhao; Dai Shi; Xinliang Zhou
>
> **备注:** Code is available at this https URL
>
> **摘要:** Existing agent memory remains predominantly reactive and retrieval-based, lacking the capacity to autonomously organize experience into persistent cognitive structure. Toward genuinely autonomous agents, we introduce CogniFold, a brain-inspired "always-on" agent memory designed for the next generation of proactive assistants. CogniFold continuously folds fragmented event streams into self-emerging cognitive structures, bootstrapping progressively higher-level cognition from incoming events and accumulated knowledge. We ground this by extending Complementary Learning Systems (CLS) theory from two layers (hippocampus, neocortex) to three, adding a prefrontal intent layer. Emulating the prefrontal cortex as the locus of intentional control and decision-making, CogniFold achieves this through graph-topology self-organization: cognitive structures proactively assemble under the stream, merge when semantically similar, decay when stale, relink through associative recall, and surface intents when concept-cluster density crosses a threshold. We evaluate structural formation using CogEval-Bench, demonstrating that CogniFold uniquely produces memory structures that match cognitive expectations and concept emergence. Furthermore, across 7 broad-coverage benchmarks spanning five cognitive domains, we validate that CogniFold simultaneously performs robustly on conventional memory benchmarks.
>
---
#### [replaced 111] Is Human Annotation Necessary? Iterative MBR Distillation for Error Span Detection in Machine Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译评估任务，旨在解决错误片段检测问题。通过自动生成伪标签，减少对人工标注的依赖，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.12983](https://arxiv.org/pdf/2603.12983)**

> **作者:** Boxuan Lyu; Haiyue Song; Zhi Qu
>
> **摘要:** Error Span Detection (ESD) is a crucial subtask in Machine Translation (MT) evaluation, aiming to identify the location and severity of translation errors. While fine-tuning models on human-annotated data improves ESD performance, acquiring such data is expensive and prone to inconsistencies among annotators. To address this, we propose a novel self-evolution framework based on Minimum Bayes Risk (MBR) decoding, named Iterative MBR Distillation for ESD, which eliminates the reliance on human annotations by leveraging an off-the-shelf LLM to generate pseudo-labels. Extensive experiments on the WMT Metrics Shared Task datasets demonstrate that models trained solely on these self-generated pseudo-labels outperform both unadapted base model and supervised baselines trained on human annotations at the system and span levels, while maintaining competitive sentence-level performance.
>
---
#### [replaced 112] M$^\star$: Every Task Deserves Its Own Memory Harness
- **分类: cs.PL; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出M$^\star$，解决通用记忆系统在不同任务中表现不佳的问题，通过进化方法自动优化记忆结构，提升多任务性能。**

- **链接: [https://arxiv.org/pdf/2604.11811](https://arxiv.org/pdf/2604.11811)**

> **作者:** Wenbo Pan; Shujie Liu; Xiangyang Zhou; Shiwei Zhang; Wanlu Shi; Mirror Xu; Xiaohua Jia
>
> **备注:** Preprint. Code: this https URL ; Live demo: this https URL
>
> **摘要:** Large language model agents rely on specialized memory systems to accumulate and reuse knowledge during extended interactions. Recent architectures typically adopt a fixed memory design tailored to specific domains, such as semantic retrieval for conversations or skills reused for coding. However, a memory system optimized for one purpose frequently fails to transfer to others. To address this limitation, we introduce M$^\star$, a method that automatically discovers task-optimized memory harnesses through executable program evolution. Specifically, M$^\star$ models an agent memory system as a memory program written in Python. This program encapsulates the data Schema, the storage Logic, and the agent workflow Instructions. We optimize these components jointly using a reflective code evolution method; this approach employs a population-based search strategy and analyzes evaluation failures to iteratively refine the candidate programs. We evaluate M$^\star$ on four distinct benchmarks spanning conversation, embodied planning, and expert reasoning. Our results demonstrate that M$^\star$ improves performance over existing fixed-memory baselines robustly across all evaluated tasks. Furthermore, the evolved memory programs exhibit structurally distinct processing mechanisms for each domain. This finding indicates that specializing the memory mechanism for a given task explores a broad design space and provides a superior solution compared to general-purpose memory paradigms.
>
---
#### [replaced 113] WISE: Web Information Satire and Fakeness Evaluation
- **分类: cs.CL**

- **简介: 该论文属于虚假信息检测任务，旨在区分假新闻与讽刺内容。通过评估多种模型，提出WISE框架，以提升 misinformation 检测效果。**

- **链接: [https://arxiv.org/pdf/2512.24000](https://arxiv.org/pdf/2512.24000)**

> **作者:** Gaurab Chhetri; Subasish Das; Tausif Islam Chowdhury
>
> **备注:** This is the author's preprint. Accepted to WEB&GRAPH 2026 (co-located with WSDM 2026), Boise, Idaho, USA, Feb 26, 2026. Final version will appear in WSDM 2026 Companion Proceedings. Conf: this https URL Workshop: this https URL
>
> **摘要:** Distinguishing fake or untrue news from satire or humor poses a unique challenge due to their overlapping linguistic features and divergent intent. This study develops WISE (Web Information Satire and Fakeness Evaluation) framework which benchmarks eight lightweight transformer models alongside two baseline models on a balanced dataset of 20,000 samples from Fakeddit, annotated as either fake news or satire. Using stratified 5-fold cross-validation, we evaluate models across comprehensive metrics including accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, MCC, Brier score, and Expected Calibration Error. Our evaluation reveals that MiniLM, a lightweight model, achieves the highest accuracy (87.58%) among all models, while RoBERTa-base achieves the highest ROC-AUC (95.42%) and strong accuracy (87.36%). DistilBERT offers an excellent efficiency-accuracy trade-off with 86.28\% accuracy and 93.90\% ROC-AUC. Statistical tests confirm significant performance differences between models, with paired t-tests and McNemar tests providing rigorous comparisons. Our findings highlight that lightweight models can match or exceed baseline performance, offering actionable insights for deploying misinformation detection systems in real-world, resource-constrained settings.
>
---
#### [replaced 114] The Scientific Contribution Graph: Automated Literature-based Technological Roadmapping at Scale
- **分类: cs.CL**

- **简介: 该论文属于科技文献分析任务，旨在通过自动化方法构建技术路线图。工作包括提取科学贡献、建立前提关系，并预测技术发展路径。**

- **链接: [https://arxiv.org/pdf/2605.15011](https://arxiv.org/pdf/2605.15011)**

> **作者:** Peter A. Jansen
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Scientific contributions rarely develop in isolation, but instead build upon prior discoveries. We formulate the task of automated technological roadmapping as extracting scientific contributions from scholarly articles and linking them to their prerequisites. We present the Scientific Contribution Graph, a large-scale AI/NLP-domain resource containing 2 million detailed scientific contributions extracted from 230k open-access papers and connected by 12.5 million prerequisite edges. We further introduce scientific prerequisite prediction, a scientific discovery task in which models predict which existing technologies can enable future discoveries, and show that contemporary models are rapidly improving on this task, reaching 0.48 MAP when evaluated using temporally filtered backtesting. We anticipate technological roadmapping resources such as this will support scientific impact assessment and automated scientific discovery.
>
---
#### [replaced 115] Psychologically Potent, Computationally Invisible: LLMs Generate Social-Comparison-Eliciting Posts They Fail to Detect
- **分类: cs.CL**

- **简介: 该论文提出XHS-SCoRE基准，用于检测小红书文本是否引发社会比较。任务是识别文本中的社会比较信号，解决模型检测与生成间的不匹配问题。**

- **链接: [https://arxiv.org/pdf/2605.01017](https://arxiv.org/pdf/2605.01017)**

> **作者:** Hua Zhao; Jiapei Gu; Michelle Mingyue Gu
>
> **备注:** 19 pages, preprint Title change: Psychologically Potent, Computationally Invisible: LLMs Generate Social-Comparison-Eliciting Posts They Fail to Detect
>
> **摘要:** We introduce Xiaohongshu Social Comparison Reader Elicitation (XHS-SCoRE), a reader-grounded benchmark for detecting whether text-only Xiaohongshu (RedNote) posts elicit Upward, Downward, or Neutral/no clear social comparison from a first-person reader perspective. The task targets a socially meaningful relational, behaviorally real signal not reducible to sentiment. Across prompted LLM classifiers and supervised Chinese encoders, we find a consistent generation--detection mismatch: the signal is textually learnable in-domain, but not robustly accessible to prompt-based classification. Prompted LLM classifiers show stable failures, especially neutralization of comparison-eliciting posts and model-specific directional skew. A controlled pilot shows that LLM-generated Xiaohongshu-style posts can shift perceived standing and comparison-related affect even when prompt-based detection of the same construct remains fragile. XHS-SCoRE contributes a benchmark for reader-grounded comparison detection and a diagnostic framework for studying when socially meaningful relational cues remain only partially visible to prompt-based inference.
>
---
#### [replaced 116] Mitigating Hallucinations in Healthcare LLMs with Granular Fact-Checking and Domain-Specific Adaptation
- **分类: cs.CL**

- **简介: 该论文属于医疗领域大语言模型的可靠性提升任务，旨在解决模型生成内容中的幻觉问题。通过引入事实核查模块和领域适应的摘要模型，提高医疗信息的准确性与可信度。**

- **链接: [https://arxiv.org/pdf/2512.16189](https://arxiv.org/pdf/2512.16189)**

> **作者:** Musarrat Zeba; Abdullah Al Mamun; Kishoar Jahan Tithee; Debopom Sutradhar; Mohaimenul Azam Khan Raiaan; Saddam Mukta; Reem E. Mohamed; Md Rafiqul Islam; Yakub Sebastian; Mukhtar Hussain; Sami Azam
>
> **摘要:** In healthcare, it is essential for any LLM-generated output to be reliable and accurate, particularly in cases involving decision-making and patient safety. However, the outputs are often unreliable in such critical areas due to the risk of hallucinated outputs from the LLMs. To address this issue, we propose a fact-checking module that operates independently of any LLM, along with a domain-specific summarization model designed to minimize hallucination rates. Our model is fine-tuned using Low-Rank Adaptation (LoRa) on the MIMIC III dataset and is paired with the fact-checking module, which uses numerical tests for correctness and logical checks at a granular level through discrete logic in natural language processing (NLP) to validate facts against electronic health records (EHRs). We trained the LLM model on the full MIMIC-III dataset. For evaluation of the fact-checking module, we sampled 104 summaries, extracted them into 3,786 propositions, and used these as facts. The fact-checking module achieves a precision of 0.8904, a recall of 0.8234, and an F1-score of 0.8556. Additionally, the LLM summary model achieves a ROUGE-1 score of 0.5797 and a BERTScore of 0.9120 for summary quality.
>
---
#### [replaced 117] Reward-free Alignment for Conflicting Objectives
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多目标对齐任务，解决冲突目标下的模型对齐问题。提出RACO框架，无需奖励模型，通过梯度裁剪实现稳定优化，提升多目标平衡效果。**

- **链接: [https://arxiv.org/pdf/2602.02495](https://arxiv.org/pdf/2602.02495)**

> **作者:** Peter Chen; Xiaopeng Li; Xi Chen; Tianyi Lin
>
> **备注:** Accepted to ICML 2026 (Oral)
>
> **摘要:** Direct alignment methods are increasingly used to align large language models (LLMs) with human preferences. However, many real-world alignment problems involve multiple conflicting objectives, where naive aggregation of preferences can lead to unstable training and poor trade-offs. In particular, weighted loss methods may fail to identify update directions that simultaneously improve all objectives, and existing multi-objective approaches often rely on explicit reward models, introducing additional complexity and distorting user-specified preferences. The contributions of this paper are two-fold. First, we propose a Reward-free Alignment framework for Conflicted Objectives (RACO) that directly leverages pairwise preference data and resolves gradient conflicts via a novel clipped variant of conflict-averse gradient descent. We provide convergence guarantees to Pareto-critical points that respect user-specified objective weights, and further show that clipping can strictly improve convergence rate in the two-objective setting. Second, we improve our method using some heuristics and conduct experiments to demonstrate the compatibility of the proposed framework for LLM alignment. Both qualitative and quantitative evaluations on multi-objective summarization and safety alignment tasks across multiple LLM families (Qwen 3, Llama 3, Gemma 3) show that our method consistently achieves better Pareto trade-offs compared to existing multi-objective alignment baselines.
>
---
#### [replaced 118] The LSCD Benchmark: a Testbed for Diachronic Word Meaning Tasks
- **分类: cs.CL**

- **简介: 该论文属于词汇语义变化检测任务，旨在解决评估标准不统一的问题。通过构建基准库，标准化评估流程，支持不同模块的灵活组合与评估。**

- **链接: [https://arxiv.org/pdf/2404.00176](https://arxiv.org/pdf/2404.00176)**

> **作者:** Dominik Schlechtweg; Sachin Yadav; Jonas Kuhn; Nikolay Arefyev
>
> **备注:** *SEM, 9 pages
>
> **摘要:** Lexical Semantic Change Detection (LSCD) is a complex, lemma-level task, which is usually operationalized based on two subsequently applied usage-level tasks: First, Word-in-Context (WiC) labels are derived for pairs of usages. Then, these labels are represented in a graph on which Word Sense Induction (WSI) is applied to derive sense clusters. Finally, LSCD labels are derived by comparing sense clusters over time. This modularity is reflected in most LSCD datasets and models. It also leads to a large heterogeneity in modeling options and task definitions, which is exacerbated by a variety of dataset versions, preprocessing options and evaluation metrics. This heterogeneity makes it difficult to evaluate models under comparable conditions, to choose optimal model combinations or to reproduce results. Hence, we provide a benchmark repository standardizing LSCD evaluation. Through transparent implementation results become easily reproducible and by standardization different components can be freely combined. The repository reflects the task's modularity by allowing model evaluation for WiC, WSI and LSCD. This allows for careful evaluation of increasingly complex model components providing new ways of model optimization.
>
---
#### [replaced 119] Reducing Credit Assignment Variance via Counterfactual Reasoning Paths
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决多步推理中信用分配不均的问题。通过反事实比较框架，提升训练稳定性与模型性能。**

- **链接: [https://arxiv.org/pdf/2605.16302](https://arxiv.org/pdf/2605.16302)**

> **作者:** Fei Ding; Yongkang Zhang; Youwei Wang; Zijian Zeng
>
> **摘要:** Reinforcement learning for multi-step reasoning with large language models (LLMs) typically relies on sparse terminal rewards, which creates a poorly conditioned credit-assignment problem: the final feedback is propagated uniformly across all intermediate decisions. This leads to high gradient variance, unstable training, and many ineffective updates, ultimately limiting sustained model improvement. We propose a counterfactual-comparison framework for credit assignment. For each input, the framework samples multiple reasoning trajectories and treats their differences as implicit approximations to alternative decisions. This yields an implicit process-level advantage estimator that converts sparse terminal rewards into step-sensitive learning signals. Building on this framework, we introduce Implicit Behavior Policy Optimization (IBPO), which substantially improves training stability and the performance ceiling on mathematical and code-reasoning benchmarks. Our results point to a promising direction for unlocking the reasoning potential of LLMs.
>
---
#### [replaced 120] From Multi-Agent Systems and the Semantic Web to Agentic AI: A Unified Narrative of the Web of Agents
- **分类: cs.AI; cs.CL; cs.CR; cs.HC; cs.MA**

- **简介: 论文探讨了从多智能体系统到语义网再到大模型时代的智能代理演进，分析了不同阶段的语义努力迁移，提出统一框架并总结经验教训，旨在推动智能代理技术发展。**

- **链接: [https://arxiv.org/pdf/2507.10644](https://arxiv.org/pdf/2507.10644)**

> **作者:** Tatiana Petrova; Boris Bliznioukov; Aleksandr Puzikov; Radu State
>
> **摘要:** The Web of Agents (WoA) transforms the document-centric Web into an environment of autonomous agents acting on users' behalf, a vision newly tractable as large language models (LLMs) mature. We argue that across three decades the WoA has undergone a \emph{semantic-effort migration} in chronological order: from platform-side coordination (Multi-Agent Systems, Generation~I), through data-side annotation (Semantic Web, Generation~II), to model-side interpretation (LLM-era, Generation~III). The central Gen~II~$\rightarrow$~Gen~III transition within this trajectory, which we call the \emph{semantics-in-data $\rightarrow$ semantics-in-models} shift, is predictive: each generation's failure modes and current open problems follow from where that generation located its semantic effort. The survey makes five contributions: (i)~a unified evolutionary narrative spanning 1990--2026; (ii)~a four-dimensional comparative framework (semantic foundation, communication paradigm, locus of intelligence, discovery mechanism) applied uniformly across all three generations; (iii)~classification of sixteen representative systems on these dimensions, including hybrid LLM--knowledge-graph and computer-use agents; (iv)~coverage of the November~2024--August~2026 institutional convergence (Linux Foundation's Agentic AI Foundation, A2A v1.0, MCP November~2024 launch and November~2025 specification, Visa/Mastercard/Stripe payment-network protocols, EU AI Act phased enforcement, the NIST AI Agent Standards Initiative, International AI Safety Report 2026); and (v)~seven named lessons grounded in cross-generational evidence paired with seven generation-invariant challenges that persist regardless of which protocol prevails. Further progress depends less on protocol design than on the socio-technical infrastructure now being assembled by standards bodies, regulators, and commercial payment networks.
>
---
#### [replaced 121] ChunkLLM: A Lightweight Pluggable Framework for Accelerating LLMs Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ChunkLLM框架，解决大模型推理效率低的问题。通过引入适配器组件，实现块级注意力和缓存优化，提升长文本处理速度。任务为高效大模型推理。**

- **链接: [https://arxiv.org/pdf/2510.02361](https://arxiv.org/pdf/2510.02361)**

> **作者:** Haojie Ouyang; Jianwei Lv; Lei Ren; Chen Wei; Xiaojie Wang; Fangxiang Feng
>
> **摘要:** Transformer-based large models excel in natural language processing and computer vision, but face severe computational inefficiencies due to the self-attention's quadratic complexity with input tokens. Recently, researchers have proposed a series of methods based on block selection and compression to alleviate this problem, but they either have issues with semantic incompleteness or poor training-inference efficiency. To comprehensively address these challenges, we propose ChunkLLM, a lightweight and pluggable training framework. Specifically, we introduce two components: QK Adapter (Q-Adapter and K-Adapter) and Chunk Adapter. The former is attached to each Transformer layer, serving dual purposes of feature compression and chunk attention acquisition. The latter operates at the bottommost layer of the model, functioning to detect chunk boundaries by leveraging contextual semantic information. During the training phase, the parameters of the backbone remain frozen, with only the QK Adapter and Chunk Adapter undergoing training. Notably, we design an attention distillation method for training the QK Adapter, which enhances the recall rate of key chunks. During the inference phase, chunk selection is triggered exclusively when the current token is detected as a chunk boundary, thereby accelerating model inference. Experimental evaluations are conducted on a diverse set of long-text and short-text benchmark datasets spanning multiple tasks. ChunkLLM not only attains comparable performance on short-text benchmarks but also maintains 98.64% of the performance on long-context benchmarks while preserving a 48.58% key-value cache retention rate. Particularly, ChunkLLM attains a maximum speedup of 4.48x in comparison to the vanilla Transformer in the processing of 120K long texts.
>
---
#### [replaced 122] ARES: Automated Rubric Synthesis for Scalable LLM Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出ARES框架，解决大规模LLM强化学习中人工编写评分标准的难题。通过自动构建带评分标准的训练数据，提升开放性任务的性能。**

- **链接: [https://arxiv.org/pdf/2605.23454](https://arxiv.org/pdf/2605.23454)**

> **作者:** Xiaoyuan Li; Keqin Bao; Moxin Li; Yubo Ma; Yichang Zhang; Wenjie Wang; Fuli Feng; Dayiheng Liu
>
> **备注:** Under Review
>
> **摘要:** Rubric-based rewards offer a promising way to extend reinforcement learning (RL) for large language models beyond tasks with automatically verifiable answers. However, scaling rubric-based RL remains challenging: existing approaches often rely on expert-written rubrics and manually constructed question sets, while fixed task-level rubrics may fail to capture the evaluation requirements of individual questions. We propose ARES (Automated Rubric synthEsis for Scalable RL), a framework for automatically constructing rubric-based RL data at scale. Starting from raw pretraining documents, ARES converts source knowledge into self-contained question-answer pairs and co-generates question-specific weighted rubrics, enabling instance-level reward supervision for open-ended responses. To improve diversity and quality, ARES conditions generation on domain labels and persona information, and applies validation filters for question self-containment, answer faithfulness, and rubric validity. Using ARES, we construct 100K rubric-annotated instances across ten domains. Experiments on seven benchmarks show that rubric-based RL trained with ARES, outperforms continual pretraining, supervised fine-tuning, and binary-reward RL, with the largest gains on multi-dimensional open-ended tasks such as healthcare and instruction following.
>
---
#### [replaced 123] False Fixed Points: Kantian Feedback, Stable Miscalibration, and Representational Compression in LLMs
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究大语言模型中的高自信错误，探讨其作为局部稳定错误点的特性。属于模型可靠性任务，旨在解决错误稳定性与正确性分离的问题，通过反馈模型和实验分析其机制。**

- **链接: [https://arxiv.org/pdf/2510.14925](https://arxiv.org/pdf/2510.14925)**

> **作者:** Akira Okutomi
>
> **备注:** 27 pages, 8 figures, v3.0
>
> **摘要:** High-confidence errors in large language models are often treated as fragile failures. We study an alternative: some errors may be false fixed points, locally stable, internally coherent, and confidently wrong. This separates robustness from truth-tracking. We develop the separation through a Kantian commitment-gate framing and a minimal linear feedback model in which stability and correctness can diverge. Across three open-weight models, overconfident wrong items are not systematically more locally fragile than confidently correct items under our hidden-state sensitivity probes. Abstention-aware self-critique reduces overconfident wrong commitments by sacrificing coverage, and C3-R, a rule-based explicit feedback gate, sharpens that tradeoff rather than eliminating it. These results motivate, but do not establish, high signal-to-noise (high-SNR) inertia and representational compression as possible mechanisms for stable miscalibration.
>
---
#### [replaced 124] PerSoMed: A Large-Scale Balanced Dataset for Persian Social Media Text Classification
- **分类: cs.CL; cs.IR; cs.SI**

- **简介: 该论文提出PerSoMed数据集，解决波斯语社交媒体文本分类资源不足的问题。通过收集和标注数据，构建平衡数据集，并评估多种模型性能。**

- **链接: [https://arxiv.org/pdf/2602.19333](https://arxiv.org/pdf/2602.19333)**

> **作者:** Isun Chehreh; Ebrahim Ansari
>
> **备注:** 10 pages, including 1 figure
>
> **摘要:** This research introduces the first large-scale, well-balanced Persian social media text classification dataset, specifically designed to address the lack of comprehensive resources in this domain. The dataset comprises 36,000 posts across nine categories (Economic, Artistic, Sports, Political, Social, Health, Psychological, Historical, and Science & Technology), each containing 4,000 samples to ensure balanced class distribution. Data collection involved 60,000 raw posts from various Persian social media platforms, followed by rigorous preprocessing and hybrid annotation combining ChatGPT-based few-shot prompting with human verification. To mitigate class imbalance, we employed undersampling with semantic redundancy removal and advanced data augmentation strategies integrating lexical replacement and generative prompting. We benchmarked several models, including BiLSTM, XLM-RoBERTa (with LoRA and AdaLoRA adaptations), FaBERT, SBERT-based architectures, and the Persian-specific TookaBERT (Base and Large). Experimental results show that transformer-based models consistently outperform traditional neural networks, with TookaBERT-Large achieving the best performance (Precision: 0.9622, Recall: 0.9621, F1- score: 0.9621). Class-wise evaluation further confirms robust performance across all categories, though social and political texts exhibited slightly lower scores due to inherent ambiguity. This research presents a new high-quality dataset and provides comprehensive evaluations of cutting-edge models, establishing a solid foundation for further developments in Persian NLP, including trend analysis, social behavior modeling, and user classification. The dataset is publicly available to support future research endeavors.
>
---
#### [replaced 125] Cross-Lingual Consensus: Aligning Multilingual Cultural Knowledge via Multilingual Self-Consistency
- **分类: cs.CL**

- **简介: 该论文属于多语言自然语言处理任务，旨在解决LLMs在跨语言文化知识对齐上的偏差问题。通过自监督框架提升模型在不同语言中的文化一致性。**

- **链接: [https://arxiv.org/pdf/2605.22137](https://arxiv.org/pdf/2605.22137)**

> **作者:** Andrew Ivan Soegeng; Patrick Sutanto; Tan Sang Nguyen
>
> **备注:** Accepted to The 1st Workshop on Multilinguality in the Era of Large Language Models
>
> **摘要:** Although Large Language Models (LLMs) demonstrate strong capabilities across various tasks, they exhibit significant performance discrepancies across languages. While prompting LLMs in English typically yields the highest general performance, it often induces a Western-centric bias, hindering the model's ability to accurately reflect diverse cultural knowledge. We hypothesize that LLMs already possess rich cultural knowledge embedded within local-language representations, but fail to retrieve it when prompted in English. To bridge this cross-lingual knowledge gap, we propose a novel self-supervised framework. Our method leverages multilingual self-consistency to identify the most reliable cultural responses across languages, combined with a self-critique mechanism to transfer this knowledge to the weaker language. Evaluations on the BLEnD benchmark demonstrate that our approach significantly improves cultural alignment-boosting performance on English queries by an average of 5.03%-relying entirely on self-generated data. Ultimately, our work demonstrates that latent cultural knowledge can be successfully surfaced and propagated across languages, enabling more culturally equitable and consistent LLMs.
>
---
#### [replaced 126] Towards Inclusive Toxic Content Moderation: Addressing Vulnerabilities to Adversarial Attacks in Toxicity Classifiers Tackling LLM-generated Content
- **分类: cs.CL**

- **简介: 该论文属于内容安全任务，旨在解决LLM生成文本导致的毒性分类器误判问题。通过识别脆弱组件并改进模型鲁棒性，提升分类器对对抗攻击的防御能力。**

- **链接: [https://arxiv.org/pdf/2509.12672](https://arxiv.org/pdf/2509.12672)**

> **作者:** Shaz Furniturewala; Arkaitz Zubiaga
>
> **摘要:** The volume of machine-generated content online has grown dramatically due to the widespread use of Large Language Models (LLMs), leading to new challenges for content moderation systems. Conventional content moderation classifiers, which are usually trained on text produced by humans, suffer from misclassifications due to LLM-generated text deviating from their training data and adversarial attacks that aim to avoid detection. Present-day defence tactics are reactive rather than proactive, since they rely on adversarial training or external detection models to identify attacks. In this work, we aim to identify the vulnerable components of toxicity classifiers that contribute to misclassification, proposing a novel strategy based on mechanistic interpretability techniques. Our study focuses on fine-tuned BERT and RoBERTa classifiers, testing on diverse datasets spanning a variety of minority groups. We use adversarial attacking techniques to identify vulnerable circuits. Finally, we suppress these vulnerable circuits, improving performance against adversarial attacks. We also provide demographic-level insights into these vulnerable circuits, exposing fairness and robustness gaps in model training. We find that models have distinct heads that are either crucial for performance or vulnerable to attack and suppressing the vulnerable heads improves performance on adversarial input. We also find that different heads are responsible for vulnerability across different demographic groups, which can inform more inclusive development of toxicity detection models.
>
---
#### [replaced 127] Knowing When to Quit: A Principled Framework for Dynamic Abstention in LLM Reasoning
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属于大模型推理任务，解决LLM生成冗长错误响应的问题。提出动态弃权机制，在推理过程中提前终止低质量路径，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.18419](https://arxiv.org/pdf/2604.18419)**

> **作者:** Hen Davidov; Nachshon Cohen; Oren Kalinsky; Yaron Fairstein; Guy Kushilevitz; Ram Yazdi; Patrick Rebeschini
>
> **摘要:** LLMs utilizing chain-of-thought reasoning often waste substantial compute by producing long, incorrect responses. Abstention can mitigate this by withholding outputs unlikely to be correct. While most abstention methods decide to withhold outputs before or after generation, dynamic mid-generation abstention considers early termination of unpromising reasoning traces at each token position. Prior work has explored empirical variants of this idea, but principled guidance for the abstention rule remains lacking. We present a formal analysis of dynamic abstention for LLMs, modeling abstention as an explicit action within a regularized reinforcement learning framework. An abstention reward parameter controls the trade-off between compute and information. We show that abstaining when the value function falls below this reward strictly outperforms natural baselines under general conditions. We further derive a principled and efficient method to approximate the value function. Empirical results on mathematical reasoning and toxicity avoidance tasks support our theory and demonstrate improved selective accuracy over existing methods.
>
---
#### [replaced 128] Fast-dDrive: Efficient Block-Diffusion VLM for Autonomous Driving
- **分类: cs.CL**

- **简介: 该论文提出Fast-dDrive，一种高效块扩散视觉-语言-动作模型，解决自动驾驶中轨迹规划的效率与精度平衡问题。通过结构化输出和推测解码提升推理速度与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.23163](https://arxiv.org/pdf/2605.23163)**

> **作者:** Kewei Zhang; Jin Wang; Sensen Gao; Chengyue Wu; Yulong Cao; Songyang Han; Boris Ivanovic; Langechuan Liu; Marco Pavone; Song Han; Daquan Zhou; Enze Xie
>
> **摘要:** End-to-end autonomous driving via Vision-Language-Action (VLA) models demands a precarious balance between high-fidelity trajectory planning and efficient inference. Existing paradigms typically fall short: autoregressive (AR) VLAs are memory-bandwidth-bound on edge hardware and prone to exposure-bias drift, while full-sequence diffusion models preclude KV-cache reuse and suffer from "logical leakage" that violates the fundamental perceive-then-plan causality. We present Fast-dDrive, a block-diffusion VLA that performs bidirectional refinement within semantic units while enforcing strict causal ordering across them. Leveraging the observation that driving VLAs often emit structured JSON-like outputs, Fast-dDrive freezes structural tokens into a section scaffold and employs a section-aware training recipe that prioritizes safety-critical planning. We further introduce Scaffold Speculative Decoding to achieve AR-equivalent quality at significantly higher throughput. Finally, we propose a low-overhead test-time scaling scheme: by forking $N$ stochastic trajectory rollouts from a single shared-prefix KV cache and averaging them, we effectively suppress prediction variance at a fractional computational cost. Empirical results demonstrate that Fast-dDrive redefines the speed-accuracy frontier for driving agents. On the WOD-E2E test set, Fast-dDrive achieves SOTA ADE@3s and ADE@5s, alongside the highest RFS among diffusion-based VLAs; on nuScenes, it reduces average L2 error to $0.32$m (a $22\%$ improvement). When integrated with SGLang, our framework delivers $12\times$ throughput speedup over the AR baseline, narrowing the gap between high-capacity VLAs and the efficiency demands of real-time on-vehicle deployment.
>
---
#### [replaced 129] AnyMo: Geometry-Aware Setup-Agnostic Modeling of Human Motion in the Wild
- **分类: cs.CV; cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出AnyMo，解决可穿戴设备在非受控环境下运动建模的问题。通过物理模拟生成合成数据，实现跨设备、跨数据集的运动理解，提升活动识别、跨模态检索和运动描述性能。**

- **链接: [https://arxiv.org/pdf/2605.22715](https://arxiv.org/pdf/2605.22715)**

> **作者:** Baiyu Chen; Zechen Li; Wilson Wongso; Lihuan Li; Xiachong Lin; Hao Xue; Benjamin Tag; Flora Salim
>
> **摘要:** As wearable and mobile devices become increasingly embedded in daily life, they offer a practical way to continuously sense human motion in the wild. But inertial signals are highly dependent on the sensing setup, including body location, mounting position, sensor orientation, device hardware, and sampling protocol. This setup dependence makes it difficult to learn motion representations that transfer across devices and datasets, and limits the broader use of wearable IMUs beyond closed-set recognition. We introduce AnyMo, a geometry-aware framework for setup-agnostic human motion modeling. AnyMo uses physics-grounded IMU simulation over dense body-surface placements to generate diverse and plausible synthetic signals, pre-trains a graph encoder from paired synthetic placement views and masked partial observations, tokenizes multi-position IMU into full-body motion tokens, and aligns these tokens with an LLM for motion-language understanding. We evaluate AnyMo on three complementary tasks: zero-shot activity recognition across 14 unseen downstream datasets, cross-modal retrieval, and wearable IMU motion captioning, where it improves average Accuracy/F1/R@2 by 11.7\%/11.6\%/22.6\% on HAR, increases zero-shot IMU-to-text and text-to-IMU retrieval MRR by 15.9\% and 28.6\%, respectively, and improves zero-shot captioning BERT-F1 by 18.8\%. These results support AnyMo as a generalist model for wearable motion understanding in the wild. Project page: this https URL.
>
---
#### [replaced 130] FineBench: Benchmarking and Enhancing Vision-Language Models for Fine-grained Human Activity Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频问答任务，旨在解决细粒度人类行为理解问题。提出FineBench基准和FineAgent框架，提升模型在复杂场景中的空间与时间推理能力。**

- **链接: [https://arxiv.org/pdf/2605.19846](https://arxiv.org/pdf/2605.19846)**

> **作者:** Gueter Josmy Faure; Min-Hung Chen; Jia-Fong Yeh; Hung-Ting Su; Winston H. Hsu
>
> **备注:** CVPR'26 (Workshop on Video Large Language Models). Project Page: this https URL
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated remarkable capabilities in general video understanding, yet they often struggle with the fine-grained comprehension crucial for real-world applications requiring nuanced interpretation of human actions and interactions. While some recent human-centric benchmarks evaluate aspects of model behaviour such as fairness/ethics, emotion perception, and broader human-centric metrics, they do not combine long-form videos, very dense QA coverage, and frame-level spatial/temporal grounding at scale. To bridge this gap, we introduce FineBench, a human-centric video question answering (VQA) benchmark specifically designed to assess fine-grained understanding. FineBench comprises 199,420 multiple-choice QA pairs densely annotated across 64 long-form videos (15 minutes each), focusing on detailed person movement, person interaction, and object manipulation, including compositional actions. Our extensive evaluation reveals that while proprietary models like GPT-5 achieve respectable performance, current open-source VLMs significantly underperform, struggling particularly with spatial reasoning in multi-person scenes and distinguishing subtle differences in human movements and interactions. To address these identified weaknesses, we propose FineAgent, a modular framework that enhances VLMs by leveraging a Localizer and a Descriptor. Experiments show that FineAgent consistently improves the performance of various open VLMs on FineBench. FineBench provides a rigorous testbed for future research into fine-grained human-centric video understanding, while FineAgent offers a practical approach to enhance such reasoning in current VLMs. Project page and code at this https URL.
>
---
#### [replaced 131] Hierarchical Local-Global Transformer for Temporal Sentence Grounding
- **分类: cs.MM; cs.CL; cs.CV; cs.IR**

- **简介: 该论文属于时序句子定位任务，旨在准确识别视频中与给定句子对应的片段。针对现有方法依赖后处理、忽略多粒度语义的问题，提出HLGT模型，通过层次化Transformer增强多模态表示和对齐。**

- **链接: [https://arxiv.org/pdf/2208.14882](https://arxiv.org/pdf/2208.14882)**

> **作者:** Xiang Fang; Daizong Liu; Pan Zhou; Zichuan Xu; Ruixuan Li
>
> **备注:** Publish in IEEE Transactions on Multimedia
>
> **摘要:** This paper studies the multimedia problem of temporal sentence grounding (TSG), which aims to accurately determine the specific video segment in an untrimmed video according to a given sentence query. Traditional TSG methods mainly follow the top-down or bottom-up framework and are not end-to-end. They severely rely on time-consuming post-processing to refine the grounding results. Recently, some transformer-based approaches are proposed to efficiently and effectively model the fine-grained semantic alignment between video and query. Although these methods achieve significant performance to some extent, they equally take frames of the video and words of the query as transformer input for correlating, failing to capture their different levels of granularity with distinct semantics. To address this issue, in this paper, we propose a novel Hierarchical Local-Global Transformer (HLGT) to leverage this hierarchy information and model the interactions between different levels of granularity and different modalities for learning more fine-grained multi-modal representations. Specifically, we first split the video and query into individual clips and phrases to learn their local context (adjacent dependency) and global correlation (long-range dependency) via a temporal transformer. Then, a global-local transformer is introduced to learn the interactions between the local-level and global-level semantics for better multi-modal reasoning. Besides, we develop a new cross-modal cycle-consistency loss to enforce interaction between two modalities and encourage the semantic alignment between them. Finally, we design a brand-new cross-modal parallel transformer decoder to integrate the encoded visual and textual features for final grounding. Extensive experiments on three challenging datasets show that our proposed HLGT achieves a new state-of-the-art performance.
>
---
#### [replaced 132] Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review
- **分类: cs.CL**

- **简介: 该论文属于作者参与的回应生成任务，旨在解决如何整合作者专业知识和意图生成有效回复的问题。工作包括构建数据集、提出生成框架和评估体系。**

- **链接: [https://arxiv.org/pdf/2602.11173](https://arxiv.org/pdf/2602.11173)**

> **作者:** Qian Ruan; Iryna Gurevych
>
> **备注:** accepted to ACL 2026 Main Conference
>
> **摘要:** Author response (rebuttal) writing is a critical stage of scientific peer review that demands substantial author effort. In practice, authors possess domain expertise, author-only information, and response strategies - concrete forms of author expertise and intent - and seek NLP assistance that integrates these signals into author response generation (ARG). Yet this author-in-the-loop paradigm lacks formal NLP formulation and systematic study: no dataset provides fine-grained author signals, existing ARG work lacks author inputs and controls, and no evaluation measures response reflection of author signals and effectiveness in addressing reviewer concerns. To fill these gaps, we introduce (i) Re3Align, the first large-scale dataset of aligned review-response-revision triplets, where revisions proxy author signals; (ii) REspGen, an author-in-the-loop ARG framework supporting flexible author input, multi-attribute control, and evaluation-guided refinement; and (iii) REspEval, a comprehensive evaluation suite with 20+ metrics spanning input utilization, controllability, response quality, and discourse. Experiments with SOTA LLMs demonstrate the benefits of author input and evaluation-guided refinement, the impact of input specificity on response quality, and controllability-quality trade-offs. We release our dataset, generation and evaluation tools.
>
---
#### [replaced 133] WhisTLE: Deeply Supervised, Text-Only Domain Adaptation for Pretrained Speech Recognition Transformers
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语音识别领域的域适应任务，解决预训练模型在未见语境下的性能问题。提出WhisTLE方法，通过文本进行深度监督的域适应，提升模型效果。**

- **链接: [https://arxiv.org/pdf/2509.10452](https://arxiv.org/pdf/2509.10452)**

> **作者:** Akshat Pandey; Karun Kumar; Raphael Tang
>
> **备注:** 10 pages
>
> **摘要:** Pretrained automatic speech recognition (ASR) models such as Whisper perform well but still need domain adaptation to handle unseen parlance. In many real-world settings, collecting speech data is impractical, necessitating text-only adaptation. We propose WhisTLE, a deeply supervised, text-only adaptation method for pretrained encoder-decoder ASR models. WhisTLE trains a variational autoencoder (VAE) to model encoder outputs from text and fine-tunes the decoder using the learned text-to-latent encoder, optionally combined with text-to-speech (TTS) adaptation. At inference, the original encoder is restored, incurring no extra runtime cost. Across four datasets and four ASR models, WhisTLE with TTS reduces word error rate (WER) by a relative 49.0% and outperforms all non-WhisTLE baselines in 100 of 112 scenarios. We also find that WhisTLE additively complements any combination of other domain adaptation approaches; we thus recommend the inclusion of WhisTLE during standard processes for adapting encoder-decoder ASR models.
>
---
#### [replaced 134] UtilityMax Prompting: A Formal Framework for Multi-Objective Large Language Model Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出UtilityMax Prompting框架，用于多目标大语言模型任务。解决自然语言提示模糊的问题，通过数学形式化定义任务，优化模型输出以最大化预期效用。**

- **链接: [https://arxiv.org/pdf/2603.11583](https://arxiv.org/pdf/2603.11583)**

> **作者:** Ofir Marom
>
> **摘要:** The success of a Large Language Model (LLM) task depends heavily on its prompt. Most use-cases specify prompts using natural language, which is inherently ambiguous when multiple objectives must be simultaneously satisfied. In this paper we introduce UtilityMax Prompting, a framework that specifies tasks using formal mathematical language. We reconstruct the task as an influence diagram in which the LLM's answer is the sole decision variable. A utility function is defined over the conditional probability distributions within the diagram, and the LLM is instructed to find the answer that maximises expected utility. This constrains the LLM to reason explicitly about each component of the objective, directing its output toward a precise optimization target rather than a subjective natural language interpretation. We validate our approach on the MovieLens 1M dataset across three frontier models (Claude Sonnet 4.6, GPT-5.4, and Gemini 2.5 Pro), demonstrating consistent improvements in precision and Normalized Discounted Cumulative Gain (NDCG) over natural language baselines in a multi-objective movie recommendation task.
>
---
#### [replaced 135] Psychometric Item Validation Using Virtual Respondents with Trait-Response Mediators
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于心理测量任务，旨在解决LLMs生成有效调查题目的问题。通过模拟虚拟受访者及其特质响应中介，提升题目构念效度。**

- **链接: [https://arxiv.org/pdf/2507.05890](https://arxiv.org/pdf/2507.05890)**

> **作者:** Sungjib Lim; Woojung Song; Eun-Ju Lee; Yohan Jo
>
> **备注:** This paper has been accepted for publication at TACL 2026
>
> **摘要:** As psychometric surveys are increasingly used to assess the traits of large language models (LLMs), the need for scalable survey item generation suited for LLMs has also grown. A critical challenge here is ensuring the construct validity of generated items, i.e., whether they truly measure the intended trait. Traditionally, this requires costly, large-scale human data collection. To make it efficient, we present a framework for virtual respondent simulation using LLMs. Our central idea is to account for mediators: factors through which the same trait can give rise to varying responses to a survey item. By simulating respondents with diverse mediators, we identify survey items that yield responses robustly correlated with intended traits across these mediators. Experiments on three psychological trait theories (Big5, Schwartz, VIA) show that our mediator generation methods and simulation framework effectively identify high-validity items. LLMs demonstrate the ability to generate plausible mediators from trait definitions and to simulate respondent behavior for item validation. Our problem formulation, metrics, methodology, and dataset open a new direction for cost-efficient survey development and a deeper understanding of how LLMs simulate human survey responses. We release our dataset and code to support future work.
>
---
