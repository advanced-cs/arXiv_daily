# 音频 cs.SD;  eess.AS

- **最新发布 21 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] The Differentiable Auditory Loop (DAL): An ML Framework for Hyper-Personalized Hearing Aids
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于听力辅助技术任务，旨在解决传统助听器在复杂环境中的性能不足问题。提出DAL框架，结合CARFAC和SEANet模型，实现个性化信号处理优化。**

- **链接: [https://arxiv.org/pdf/2606.04103](https://arxiv.org/pdf/2606.04103)**

> **作者:** Alejandro Ballesta Rosen; Jason Mikiel-Hunter; Julian Maclaren; Jack Collins; Richard F. Lyon; Simon Carlile
>
> **摘要:** Conventional hearing aids rely on fixed, frequency-dependent amplification and compression to manage reduced sensitivity, which often fails to provide sufficient listening support in complex environments, such as situations with multiple speakers (the ``cocktail party'' problem). To more comprehensively address the underlying encoding dysfunctions of hearing loss, we introduce the Differentiable Auditory Loop (DAL), a new open-source framework for personalized hearing aid design and fitting. Our first implementation of DAL incorporates CARFAC, a differentiable model of human cochlear function, which we ported to JAX, to optimize a deep neural network to match impaired auditory neural activity patterns with a normal-hearing reference. To build a hearing aid with the fine-grained spectro-temporal signal processing required, we adopt SEANet, a waveform-to-waveform fully convolutional UNet generator. We fine-tune the network by comparing the outputs of a CARFAC model fitted to normal hearing with that of a CARFAC model fitted to match each subject's individual hearing impairment. The comparison is done using loss functions derived from the respective CARFAC neural activity pattern (NAP) outputs and stabilized auditory images (SAIs), the latter providing a 2D representation that captures phase-insensitive temporal structure in the auditory nerve output. Through gradient descent, the SEANet model learns to both denoise the input and compensate for the hearing loss modelled by the impaired CARFAC model. Across neural-representation and signal-fidelity metrics, the DAL-optimized SEANet model outperformed the tested master hearing aid (MHA) baselines. The DAL framework provides a practical path toward model-based, machine-learning-driven personalization of hearing aid signal processing. Next steps include hardware deployment to enable real-world clinical testing.
>
---
#### [new 002] Masked Wavelet Scattering Transform Neural Field for Sound Field Reconstruction
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于声场重建任务，旨在解决稀疏观测下的重建问题。通过引入小波散射变换和神经场，结合掩码策略提升重建效果。**

- **链接: [https://arxiv.org/pdf/2606.04370](https://arxiv.org/pdf/2606.04370)**

> **作者:** Xinmeng Luan; Samuel A. Verburg; Efren Fernandez-Grande; Gary Scavone
>
> **备注:** 5 pages, 2 figures, conference
>
> **摘要:** In this paper, we propose a reconstruction framework that leverages the Wavelet Scattering Transform (WST) as a multi-scale feature extractor to impose statistical priors under sparse observation conditions. The reconstruction problem is formulated as an optimization task and solved using a neural field, with the WST incorporated into the training loss function. As a proof of concept, we validate the proposed method on HRTF upsampling. A masking strategy is applied to the WST coefficients, resulting in a two-phase procedure. The first phase learns a binary mask from a small multi-subject dataset, while the second phase applies the learned mask to the WST coefficients of an individual HRTF to preserve informative statistical structures during reconstruction. Validation against baseline methods, which also serve as an ablation study of the different components of the framework, demonstrates the effectiveness of the proposed approach.
>
---
#### [new 003] Read What You Hear: Reference-Free Hypotheses Evaluation with Acoustic Discrepancy
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决无参考评估问题。提出READ方法，通过声学差异评估ASR假设，无需额外训练即可提升识别效果。**

- **链接: [https://arxiv.org/pdf/2606.04680](https://arxiv.org/pdf/2606.04680)**

> **作者:** Zhihan Li; Hankun Wang; Yiwei Guo; Bohan Li; Xie Chen; Kai Yu
>
> **备注:** Submitted to Interspeech 2026. 6 pages, 4 figures
>
> **摘要:** Automatic speech recognition systems commonly rely on reference transcriptions for evaluation, while reference-free approaches often depend on internal confidence estimation or auxiliary language models. We propose READ (Reference-free Hypothesis Evaluation with Acoustic Discrepancy), a novel metric that evaluates ASR hypotheses directly from the speech signal. READ emphasizes the acoustic grounding of hypotheses. It uses a pretrained auto-regressive TTS model to compute the conditional likelihood of speech tokens given a text hypothesis, to measure fine-grained acoustic discrepancy between speech and text. Without additional training, READ can be applied for hypothesis refinement. Experiments show that READ correlates with specific recognition errors and improves ASR outputs, achieving up to 20\% relative error rate reduction, with particularly strong gains under noisy conditions.
>
---
#### [new 004] Beyond Text Following: Repairable Arbitration Reversals in Audio-Language Models
- **分类: cs.SD; cs.CL**

- **简介: 该论文研究音频-语言模型在音频与文本冲突时的决策机制，旨在解决模型偏好文本而非音频的问题。通过实验发现音频信息被编码但被文本覆盖，并提出GACL方法提升模型对音频的依赖。**

- **链接: [https://arxiv.org/pdf/2606.05161](https://arxiv.org/pdf/2606.05161)**

> **作者:** Yichen Gao; Yiqun Zhang; Zijing Wang; Yujia Li; Heng Guo; Xi Wu; Xiaocui Yang; Shi Feng; Yifei Zhang; Daling Wang
>
> **摘要:** Audio-language models (ALMs) often follow text that conflicts with audio, even when the audio evidence is clear. This raises a basic question: is the audio-supported answer unavailable, or is it represented but overridden by the conflicting text? We examine this question using a same-audio counterfactual that keeps the audio fixed, removes only the conflicting text, and measures the resulting shift in model preference. Across five ALMs and four conflict tasks, 64.1% of conflict samples show a sign flip: the same-audio branch prefers the audio-supported answer, whereas the joint branch prefers the text-supported answer. This pattern suggests that the relevant audio evidence is encoded but loses in arbitration. Activation patching further localizes the reversal to answer-position computation, and patching effects closely track output candidate-score differences (Spearman rho=0.93). Using this diagnostic, we propose Gated Audio Counterfactual Logit Correction (GACL), a training-free decoding rule that interpolates between joint and same-audio scores. Under a strict 5 pp faithfulness-drop budget, GACL improves nAUC by 17.8 points over the best contrastive baseline and transfers without retuning to vision-text arbitration (up to +40.5 pp).
>
---
#### [new 005] Representation Matters in Randomized Smoothing for Audio Classification
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于音频分类任务，研究随机平滑的表示影响。解决直接平滑在音频中定义不明确的问题，通过分析不同表示下的鲁棒性，提出应明确报告任务特定的认证对象和扰动模型。**

- **链接: [https://arxiv.org/pdf/2606.04210](https://arxiv.org/pdf/2606.04210)**

> **作者:** Jong-Ik Park; Shreyas Chaudhari; José M. F. Moura; Carlee Joe-Wong
>
> **摘要:** Randomized smoothing (RS) certifies robustness in the vector space where Gaussian noise is added. In audio classification, this space is often not uniquely defined as standard pipelines normalize, range-control, and transform waveforms into log-mel or other spectral features. We show that direct RS is therefore under-specified unless the certified object and preprocessing policy are explicit. On two audio benchmarks, keyword spotting and environmental-sound classification, we study waveform, feature-space, and post-processed smoothing. Our diagnostics show why representation-aware reporting is necessary: at the same smoothing level $\sigma=0.0025$, the two datasets share the same median raw radius $.007996$, but different waveform energies yield different SNR-equivalent scales ($83.98$ vs. $90.97$ dB); log-mel smoothing gives higher positive-radius certified accuracy on environmental sounds ($68.42\%$ vs. $65.53\%$), certifying more examples with nonzero radius but over features rather than waveforms; and clipping or peak normalization changes the effective perturbation norm by roughly $230$--$351\times$. We therefore recommend that audio RS studies choose and report the task-specific certified object and perturbation model, including the perturbation location, gain policy, raw radius, and any post-noise geometry changes.
>
---
#### [new 006] FoeGlass: Simple In-Context Learning Is Enough for Red Teaming Audio Deepfake Detectors
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决检测模型的漏洞问题。提出FoeGlass方法，通过大模型的上下文学习能力自动生成欺骗样本，提升检测器鲁棒性。**

- **链接: [https://arxiv.org/pdf/2606.05101](https://arxiv.org/pdf/2606.05101)**

> **作者:** Sepehr Dehdashtian; Jacob H Seidman; Vishnu N Boddeti; Gaurav Bharaj
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Audio deepfake detection (ADD) models are critical for countering the malicious use of text-to-speech (TTS) models. Evaluating and strengthening ADD models requires developing datasets that span the space of generated audio and highlight high-error regions. Existing dataset development strategies face two challenges: (i) manual collection, and (ii) inefficient discovery of blind spots in the ADD models. To address these challenges, we propose FoeGlass, the first black-box automated red-teaming method for ADDs, which effectively discovers ADD failure modes in the space of generated audio underexplored by state-of-the-art deepfake benchmarks. FoeGlass uses the in-context learning capabilities of an LLM to explore the input space of a TTS model, generating audio samples that fool the target ADD using only black-box access to all components. By using a carefully designed context based on diversity measurements, FoeGlass mitigates the common problem of mode collapse in automated red-teaming systems. Empirical evaluations on several open-source ADD and TTS models demonstrate that data generated from FoeGlass substantially improves the false negative rates over unconditional sampling baselines and recent spoofing datasets by up to 94%, while requiring no manual supervision. Furthermore, we show that the attacks generated by FoeGlass are transferable across different target ADDs, demonstrating its broad applicability and ease of use for the automated red teaming of ADD systems. Finally, fine-tuning ADD models on FoeGlass-generated samples notably enhances the robustness of the detectors (up 41%).
>
---
#### [new 007] Audio Interaction Model
- **分类: cs.SD; cs.AI; cs.CL; cs.MM; eess.AS**

- **简介: 该论文提出Audio-Interaction模型，解决传统音频模型任务单一、无法实时交互的问题。通过统一的在线音频语言模型，实现多任务实时音频处理与响应。**

- **链接: [https://arxiv.org/pdf/2606.05121](https://arxiv.org/pdf/2606.05121)**

> **作者:** Zhifei Xie; Zihang Liu; Ze An; Xiaobin Hu; Yue Liao; Ziyang Ma; Dongchao Yang; Mingbao Lin; Deheng Ye; Shuicheng Yan; Chunyan Miao
>
> **备注:** Next generation of LALMs, work in progress
>
> **摘要:** Audio is an inherently interactive modality, yet today's Large Audio Language Models (LALMs) are offline, and streaming audio models each handle only a single task such as streaming ASR or voice chatting. It is time to unify them into one online LALM: a model that, through an always-on perceive-decide-respond loop, listens to sound, environment, and instructions in real time and reacts on the fly. We formalize this regime as the Audio Interaction Model, and realize it with Audio-Interaction, a unified streaming model that retains offline task execution while adding online general audio instruction following, from dialogue to full voice chatting, deciding when to respond from the semantics of the stream. To enable this, we propose SoundFlow, a framework that instantiates the perceive-decide-respond loop end to end, from data to training to deployment, through streaming-native data construction, comprehension-aware training, and asynchronous low-latency inference for stable real-time interaction. We further construct StreamAudio-2M, a 2.6M-item streaming corpus spanning 7 fundamental abilities and 28 sub-tasks, and Proactive-Sound-Bench for evaluating proactive audio intervention. Across 8 benchmarks, Audio-Interaction preserves competitive performance on mainstream audio tasks while unlocking capabilities inaccessible to offline LALMs, including real-time ASR, streaming audio instruction following, and proactive help.
>
---
#### [new 008] Flow-HOA: Generative Joint Optimization for Ambisonics Encoding via Flow Matching
- **分类: cs.SD**

- **简介: 论文提出Flow-HOA，解决稀疏麦克风阵列的HOA编码问题，通过生成模型联合优化时域、频域和空间保真度，生成高效FIR滤波器。**

- **链接: [https://arxiv.org/pdf/2606.04570](https://arxiv.org/pdf/2606.04570)**

> **作者:** Yuhuan You; Yufan Qian; Tianshu Qu; Bin Wang; Xueyang Lv
>
> **备注:** Accepted for presentation at AES Europe 2026 Convention (AES 160th Convention), Copenhagen, Denmark, May 28-30, 2026
>
> **摘要:** Higher-Order Ambisonics (HOA) encoding from sparse, irregular microphone arrays remains a critical challenge for consumer spatial audio capture in immersive communication and XR. We propose Flow-HOA, a generative framework that jointly optimizes a multi-dimensional objective encompassing time-domain, spectral, and spatial fidelity while producing a deployable, time-invariant bank of Finite Impulse Response (FIR) encoding filters. Using conditional flow matching, the model learns to map a simple prior distribution to the target distribution of FIR filter coefficients. Training is guided by a composite loss that balances time-domain waveform fidelity, multi-resolution spectral consistency, sub-band energy preservation, and spatial directivity constraints. Objective evaluations on synthetically simulated data demonstrate improved performance over strong model-based baselines in both signal fidelity and spatial accuracy metrics. Subjective listening tests on real microphone array recordings further confirm that Flow-HOA yields higher overall sound quality with reduced artifacts, demonstrating generalization from synthetic training data to real-world capture conditions.
>
---
#### [new 009] SHB-AE: Spherical harmonic beamforming based Ambisonics encoding and upscaling method for smartphone microphone array
- **分类: cs.SD**

- **简介: 该论文属于空间音频编码任务，旨在解决智能手机麦克风阵列在HOA录制中的限制。通过设计基于球面谐波的波束成形器，实现Ambisonics编码与升频。**

- **链接: [https://arxiv.org/pdf/2606.04584](https://arxiv.org/pdf/2606.04584)**

> **作者:** Yuhuan You; Yufan Qian; Tianshu Qu; Bin Wang; Xueyang Lv
>
> **备注:** Accepted for presentation at AES Europe 2025 Convention (AES 158th Convention), Warsaw, Poland, May 22-24, 2025
>
> **摘要:** With the rapid development of virtual reality (VR) and augmented reality (AR), spatial audio recording and reproduction have gained increasing research interest. Higher Order Ambisonics (HOA) stands out for its adaptability to various playback devices and its ability to integrate head orientation. However, current HOA recordings often rely on bulky spherical microphone arrays (SMA), and portable devices like smartphones are limited by array configuration and number of microphones. We propose SHB-AE, a spherical harmonic beamforming based method for Ambisonics encoding using a smartphone microphone array (SPMA). By designing beamformers for each order of spherical harmonic functions based on the array manifold, the method enables Ambisonics encoding and up-scaling. Validation on a real SPMA and its simulated free-field counterpart in noisy and reverberant conditions showed that the method successfully encodes and up-scales Ambisonics up to the fourth order with just four irregularly arranged microphones.
>
---
#### [new 010] UAT: Unified Audio-Text Diffusion for Audio Generation, Editing, and Captioning
- **分类: eess.AS**

- **简介: 该论文提出UAT，解决音频生成、编辑与字幕生成的统一问题。通过结合音频扩散与文本扩散，实现双向建模，提升合成与语义预测的平衡。**

- **链接: [https://arxiv.org/pdf/2606.04939](https://arxiv.org/pdf/2606.04939)**

> **作者:** Hui Wang; Yifan Yang; Zeyue Tian; Yuhang Jia; Jinghua Zhao; Long Zhou; Bing Han; Cheng Liu; Jiaming Zhou; Geng Tu; Yong Qin
>
> **摘要:** Audio generation and audio-to-text understanding remain largely separate, with diffusion models dominating high-fidelity synthesis and autoregressive (AR) language models driving captioning and semantic prediction. Existing unified approaches typically rely on either heterogeneous modules or AR-centric modeling, which can hinder joint optimization and limit acoustic fidelity. We present UAT, to our knowledge, the first diffusion-centric framework that supports unified audio generation, editing, and captioning. UAT couples continuous latent diffusion for audio with masked discrete diffusion for text, enabling bidirectional audio-text modeling within a shared dual-stream backbone. Experiments show that UAT preserves strong audio generation and editing capabilities while achieving competitive captioning performance, demonstrating a favorable balance between acoustic synthesis and semantic prediction. Demo samples are available at this https URL.
>
---
#### [new 011] Drift-Augmented Scoring: Text-Derived Noise Robustness for Zero-Shot Audio-Language Classification
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于零样本音频分类任务，解决噪声环境下模型性能下降问题。提出DAS方法，通过文本预测噪声方向调整得分，提升分类准确率和mAP。**

- **链接: [https://arxiv.org/pdf/2606.04844](https://arxiv.org/pdf/2606.04844)**

> **作者:** Tu Vo; Sheir Zaheer; Chan Y. Park
>
> **摘要:** Contrastive audio-language models such as CLAP enable zero-shot audio classification: a sound is labelled by matching its embedding to text prompt embeddings, with no labelled audio. This matching breaks down under acoustic noise, where accuracy and mAP fall by 12-30 percentage points at 0 dB SNR on standard benchmarks. We propose Drift Augmented Scoring (DAS), a small per-class bonus added to the cosine score. The bonus rewards a class when the noisy audio embedding drifts in the direction that the class's noise-conditioned text prompts predict. It is derived from text alone, computed once and cached, and adds a single inner product per class at inference, with no gradients and no test-time batch. On a LAION CLAP backbone, we compare DAS against the four variants of Acevedo et al.'s concurrent method on UrbanSound8K and the full FSD50K eval set, mixing each clip with urban acoustic scene noise across a range of SNRs. DAS improves the metric on every test condition: by +2.60 to +5.75 accuracy points on UrbanSound8K and +1.50 to +1.74 mAP points on FSD50K.
>
---
#### [new 012] Gauss Circle Lattices with Geometric Convolutions for Synthesizing High Dimensional Image-Source Room Impulse Responses
- **分类: cs.SD; eess.AS; math.CO**

- **简介: 该论文属于声学模拟任务，解决高维房间脉冲响应（RIR）计算效率问题。通过将图像源模型转化为高斯圆问题，提出更高效的计算方法。**

- **链接: [https://arxiv.org/pdf/2606.04358](https://arxiv.org/pdf/2606.04358)**

> **作者:** Yuancheng Luo
>
> **备注:** Accepted for publication at the 29th International Conference on Digital Audio Effects 2026
>
> **摘要:** The image-source model (ISM) is a widely adopted method for efficiently simulating acoustic room impulse responses (RIRs) under specular reflection assumptions. Acoustic paths between source and receiver are traced to lattice points computed from successive reflections over bounding planes of the room. Rectangular rooms bound the total number of image-sources to be polynomial in the RIR's duration or distance $k$ equivalent, with degree equal the number of room dimensions $N$. Direct ISM simulations are therefore compute upper-bound by $O \left ( k^N \right )$, and consider only cases of $N \leq 3$ for tractability and real-world applications. This work proposes an alternative computational method that lowers the asymptotic compute bound to $O \left ( N k^2 \log k \right )$ for integer coordinates and room dimensions via reducing ISM lattice point counting to the classic Gauss circle problem (GCP). We extend the lattice counting model to frequency-dependent and reflection weighted image-sources in higher dimensions, relating solutions between successive dimensions via the convolution operator. Two constructions for realizing RIRs are presented, along with time-frequency controls, error and run-time analysis, and RIR statistics.
>
---
#### [new 013] A Second-Order Cepstral Signature of Contact-Vibration Sounds Reproduced by Laptop Loudspeakers: A Synthetic Case Study
- **分类: cs.SD; cs.MM; math.SP**

- **简介: 该论文研究手机振动声音在笔记本电脑扬声器播放时的听觉差异，分析其双阶倒谱特征。属于音频信号处理任务，旨在解释振动声音在传播过程中的特性变化。**

- **链接: [https://arxiv.org/pdf/2606.04475](https://arxiv.org/pdf/2606.04475)**

> **作者:** Jim Salsman
>
> **备注:** 11 pages, 4 tables, 5 figures, 8 references
>
> **摘要:** A mobile phone vibrating on a hard surface often sounds qualitatively unlike ordinary audiovisual recordings when reproduced through laptop loudspeakers. We propose that part of this perceptual distinctiveness can be described as a nested periodicity: a first-order cepstral structure reflecting the vibration period and its multiples, and a second-order cepstral structure reflecting repeated spacing within the first-order cepstrum. Treating the perceptual effect as real and using a deliberately transparent synthetic signal chain, we model six stages: mechanical generation, surface and air propagation, microphone capture, encoding and decoding, laptop-speaker playback, and re-recording or post-processing. The synthetic analysis shows that the first-order cepstral periodicity is preserved across the chain, whereas a cleaner bimodal or quasi-bimodal second-order cepstral signature is most evident at the mechanical source and at laptop-speaker playback. The result supports, but does not prove, the hypothesis that laptop reproduction can re-emphasize a latent contact-vibration periodicity that is less cleanly expressed in intermediate recorded and encoded forms. We frame second-order cepstral bimodality as an exploratory descriptor of contact-vibration playback rather than as a completed perceptual metric. Required validation includes recordings of real devices, controlled playback transfer functions, perceptual judgments, and comparisons against ordinary speech, music, and environmental recordings.
>
---
#### [new 014] Channel-Oriented Design for EEG-to-Music Reconstruction
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于EEG-to-music重建任务，解决弱信号易受噪声干扰的问题。提出通道导向设计，包括通道级分词、多视角自蒸馏和结构化通道增强，提升信号稳定性与音乐语义对齐。**

- **链接: [https://arxiv.org/pdf/2606.04040](https://arxiv.org/pdf/2606.04040)**

> **作者:** Jiaxin Qing; Junwei Lu; Lexin Li
>
> **摘要:** Brain-computer interfaces aim to decode naturalistic stimuli from neural signals, yet most progress to date has focused on vision and language. In this article, we study a more challenging but far less explored setting, EEG-to-music reconstruction, where signals are weak, distributed, and highly susceptible to noise and channel variability. Our central finding is that early channel mixing destroys weak but discriminative EEG signals. To address this, we propose a channel-oriented design with three key components. Specifically, channel-wise tokenization treats each electrode as an explicit token to retain spatially localized neural evidence, channel-wise multi-view self-distillation enforces consistency across temporal crops and random channel subsets to learn robust and distributed representations, and channel-wise data augmentation introduces structured channel dropout to improve invariance to noise, artifacts, and missing electrodes. Together, these components preserve weak yet informative signals across channels and enable stable alignment to a semantic music representation space. We integrate this channel-oriented design within an encoding-alignment-decoding pipeline for EEG-to-music reconstruction. Theoretically, we characterize when preserving channel-level structure leads to improved alignment. Empirically, we compare with a range of state-of-the-art baselines and demonstrate consistent and significant performance gains.
>
---
#### [new 015] CleanCodec: Efficient and Robust Speech Tokenization via Perceptually Guided Encoding
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出CleanCodec，解决语音编码中信息冗余与效率不足的问题，通过感知引导编码，提升语音重建质量与token效率。**

- **链接: [https://arxiv.org/pdf/2606.04418](https://arxiv.org/pdf/2606.04418)**

> **作者:** Eugene Kwek; Feng Liu; Rui Zhang; Wenpeng Yin
>
> **摘要:** Neural audio codecs are a key component of speech processing pipelines, compressing audio into discrete tokens for downstream modeling. However, existing codecs struggle to balance reconstruction quality with token efficiency, often encoding perceptually irrelevant information such as background noise and recording artifacts at the expense of linguistically and acoustically meaningful content. We reframe audio tokenization as a selective information bottleneck problem and propose CleanCodec, a denoising audio codec which learns to encode only perceptually important features and discard imperceptible information. At just 12.5 tokens per second, CleanCodec achieves state-of-the-art tokenization efficiency, substantially outperforming existing codecs in speaker similarity and speech intelligibility. Evaluations on downstream text-to-speech and voice conversion tasks further demonstrate improved performance and up to 17x faster inference, highlighting significant efficiency gains.
>
---
#### [new 016] Differentiable Articulatory Copy-Synthesis of Biphonic Singing
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于语音合成任务，旨在解决双声唱法sygyt的音色复制难题。通过改进的波导模型和参数化方法，提升对特定谐波的控制能力，显著降低频谱距离。**

- **链接: [https://arxiv.org/pdf/2606.04943](https://arxiv.org/pdf/2606.04943)**

> **作者:** Mateo Cámara; María Pilar Daza-Llin; Fernando Marcos-Macías; José Luis Blanco
>
> **备注:** Accepted to DAFx 2026
>
> **摘要:** Sygyt is a Tuvan style of biphonic singing in which a low vocal drone is sustained while a high harmonic is selectively amplified in the 1--3\,kHz region. Copy-synthesizing this effect remains challenging for articulatory models, since it requires fine control of narrowly focused resonances that standard low-dimensional tract parameterizations cannot easily reproduce. We address this problem with a differentiable Kelly--Lochbaum waveguide augmented with a sublingual second source, cubic B-spline tract parameterization, and spatially varying learnable damping, optimized end-to-end by gradient descent from audio. On 20 segments from two independent sygyt datasets (5 singers, 10 pitches), the proposed model reduces log-spectral distance by 30--38\% relative to an articulatory baseline, with the largest gains concentrated in the overtone region. Cepstral-envelope analysis further shows more accurate recovery of the merged formant structure characteristic of sygyt production. The model also outperforms a DDSP harmonic-plus-noise baseline with direct per-harmonic spectral control, suggesting that explicit acoustic structure is a useful inductive bias for overtone-singing copy-synthesis.
>
---
#### [new 017] Feasibility of Time-Domain DNN-Based Speech Enhancement on Embedded FPGA for Hearing Aid
- **分类: cs.SD; cs.AR; eess.AS**

- **简介: 该论文研究嵌入式FPGA上基于时域DNN的语音增强可行性，解决听力设备的延迟与功耗问题，通过优化模型精度和架构实现低延迟语音处理。**

- **链接: [https://arxiv.org/pdf/2606.04221](https://arxiv.org/pdf/2606.04221)**

> **作者:** Feyisayo Olalere; Umut Altin; Kiki van der Heijden; Marcel van Gerven
>
> **备注:** 13 pages
>
> **摘要:** Hearing aids impose strict latency and power constraints that current DNN-based speech enhancement systems struggle to meet on embedded hardware. We characterize this gap by deploying both speech separation and denoising using the lightweight SuDoRM-RF++ architecture on the AMD-Xilinx Kria KV260, evaluated at FP32 and 16-bit fixed-point precision for each task. Across these configurations, first-sample latency tracks with on-chip parameter caching rather than arithmetic throughput, identifying data movement as the primary bottleneck. Precision reduction halves the model memory footprint without compromising objective speech quality. The fixed-point denoising accelerator achieves a first-sample latency of 9.7~ms, meeting the 10~ms clinical threshold, while speech separation reaches 16.0~ms. These measurements establish concrete resource requirements for embedded DNN-based speech enhancement and quantify the remaining gap to hearing aid deployment.
>
---
#### [new 018] SURF: Separation via Unsupervised Remixing Flow
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于单通道声源分离任务，解决无监督环境下源信号分离问题。提出SURF方法，通过流匹配和自监督技术，直接从混合信号中学习分离模型，取得新突破。**

- **链接: [https://arxiv.org/pdf/2606.04921](https://arxiv.org/pdf/2606.04921)**

> **作者:** Henry Li; Robin Scheibler; Efthymios Tzinis; Matt Shannon; Arnaud Doucet; John R. Hershey
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** The goal of single-channel source separation is to reconstruct $K$ sources given their mixture. In supervised settings where vast amounts of clean source data are available, this challenging, ill-posed problem has been addressed successfully by generative diffusion and flow-based prior models. However, access to such clean source samples is often limited, and even when available, supervised models are vulnerable to domain shifts. To bridge this gap, we present Separation via Unsupervised Remixing Flow (SURF), an unsupervised flow matching approach for source separation that learns directly from observed mixtures. This method relies on a novel combination of state-of-the-art supervised flow matching and regression-based self-supervised techniques. At a high level, starting from a teacher model, we utilize a "remixing" step to bootstrap the learning of a student flow model from the teacher's estimates. We provide insights into the objectives optimized by this approach and draw a novel connection to the Wake-Sleep algorithm. Empirical evaluations on image and audio benchmarks demonstrate that SURF establishes a new state-of-the-art, significantly outperforming existing unsupervised methods. See our demo page for examples. this https URL
>
---
#### [new 019] DetectZoo: A Unified Toolkit for AI-Generated Content Detection Across Text, Audio, and Image Modalities
- **分类: cs.MM; cs.AI; cs.CL; cs.CV; cs.LG; cs.SD**

- **简介: 该论文提出DetectZoo，一个统一的AI生成内容检测工具包，解决多模态内容检测的标准化问题，整合数据、模型和评估流程，便于研究与比较。**

- **链接: [https://arxiv.org/pdf/2606.04205](https://arxiv.org/pdf/2606.04205)**

> **作者:** Sajad Ebrahimi; Nima Jamali; Bardia Shirsalimian; Kelly McConvey; Wentao Zhang; Jalehsadat Mahdavimoghaddam; Maksym Taranukhin; Maura Grossman; Vered Shwartz; Yuntian Deng; Ebrahim Bagheri
>
> **摘要:** The growing popularity and capacity of generative models have eroded the distinction between human and machine-generated content, motivating a growing body of work on detection across text, images, and audio. Most available detectors are either commercial software or, if open-source, come with incompatible codebases with bespoke preprocessing, evaluation protocols, and evaluation metrics, which make their adoption, fair comparison, and reproduction quite difficult. To address this critical gap, we introduce DetectZoo, a first-of-its-kind, extensible toolkit designed to provide a unified interface for AI-generated content detection across text, audio, and image modalities. DetectZoo standardizes the complete empirical pipeline, from data ingestion and preprocessing to model assessment, offering researchers a cohesive framework to benchmark state-of-the-art detectors systematically. By integrating diverse public datasets and baseline detection algorithms under a single, unified API, our toolkit facilitates rigorous and reproducible evaluation. DetectZoo provides reference implementations of 61 detectors, native loaders for 22 benchmark datasets, and a standardized evaluation pipeline that reports multiple metrics through a common interface. Each detector is self-contained yet accessible through the same interface, automatically caches pretrained weights, and reproduces the original published results. DetectZoo lowers the barrier to entry for multi-modal AI forensics, enabling researchers to identify performance gaps across domains and accelerating the development of robust, generalizable detection techniques. The open-source repository and comprehensive documentation are publicly available at this https URL, and the package can be installed via pip install detectzoo.
>
---
#### [new 020] Entity Binding Failures in Speech LLM Reasoning: Diagnosis and Chain-of-Thought Intervention
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究语音大语言模型在逻辑推理中的实体绑定失败问题，提出EA-CoT方法提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2606.04474](https://arxiv.org/pdf/2606.04474)**

> **作者:** Ming-Hao Hsu; Xiaohai Tian; Jun Zhang; Zhizheng Wu
>
> **摘要:** Speech Large Language Models (SLLMs) underperform their text counterparts on complex reasoning. We reveal that this modality gap is not a uniform cognitive deficit. Evaluating three diverse SLLMs, we show speech-to-text (S2T) matches or exceeds text-to-text (T2T) on spatial, syntactic, and factual tasks. However, on logical tasks requiring entity tracking, S2T accuracy collapses to chance. We diagnose this localized degradation as an entity binding failure: continuous speech features cause models to lose precise entity-property associations during implicit reasoning. To resolve this, we propose Entity-Aware Chain-of-Thought (EA-CoT), forcing SLLMs to explicitly enumerate entities and bind them to claims before reasoning. Strikingly, EA-CoT bridges the gap, even when spoken names are misrecognized, yielding up to a 24.4% absolute accuracy improvement. Ablations confirm these gains stem entirely from explicit semantic binding, reframing the gap as a resolvable bottleneck.
>
---
#### [new 021] Multilingual Long-Form Speech Instruction Following: KIT's Submission to IWSLT 2026
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于多语言长文本语音指令跟随任务，解决模型过拟合已知任务的问题。工作包括数据增强、标签生成和跨语言翻译，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2606.04730](https://arxiv.org/pdf/2606.04730)**

> **作者:** Enes Yavuz Ugan; Maike Züfle; Yuka Ko; Supriti Sinhamahapatra; Fabian Retkowski; Seymanur Akti; Jan Niehues; Alexander Waibel
>
> **备注:** 9 pages main paper, IWSLT 2026 Instruction Following track
>
> **摘要:** With the advent of Large Language Models, single-task and token-based multi-task models have evolved into instruction-based systems that infer task and target language implicitly from natural language prompts. This trend is reflected in IWSLT's Instruction Following Track, which this year introduced new tasks including an unknown surprise task, posing a genuine challenge against overfitting to known tasks. We present KIT's submission to the Long and Short Instruction Following tracks in the unconstrained setting. Our approach combines a general data augmentation pipeline that converts short-form corpora into long-form training data through segment concatenation, LLM-based label generation, and cross-lingual translation, yielding over 1M instances across six tasks and four languages. We further show that likelihood-based re-ranking, while highly effective for ASR, systematically degrades semantic tasks by spuriously selecting candidates generated from segmented audio processing rather than holistic long-form inference, a failure mode resolved by combining likelihood with Minimum Bayes Risk decoding.
>
---
## 更新

#### [replaced 001] Neural Directional Filtering with Configurable Directivity Pattern at Inference
- **分类: eess.AS**

- **简介: 该论文属于音频空间滤波任务，旨在解决用户自定义方向性模式的实时滤波问题。通过引入FiLM架构，实现灵活的方向性控制与高精度模式逼近。**

- **链接: [https://arxiv.org/pdf/2510.20253](https://arxiv.org/pdf/2510.20253)**

> **作者:** Weilong Huang; Srikanth Raj Chetupalli; Emanuël A. P. Habets
>
> **备注:** Final camera-ready version of EUSIPCO 2026
>
> **摘要:** Spatial filtering with a desired directivity pattern is advantageous for many audio applications. In this work, we propose neural directional filtering with user-defined directivity patterns (UNDF), which enables spatial filtering based on directivity patterns that users can define during inference. To achieve this, we propose a DNN architecture that integrates feature-wise linear modulation (FiLM), allowing user-defined patterns to serve as conditioning inputs. Through analysis, we demonstrate that the FiLM-based architecture enables the UNDF to generalize to unseen user-defined patterns during interference with higher directivities, scaling variations, and different steering directions. Furthermore, we progressively refine training strategies to enhance pattern approximation and enable UNDF to approximate irregular shapes. Lastly, experimental comparisons show that UNDF outperforms conventional methods.
>
---
#### [replaced 002] VGGSounder: Audio-Visual Evaluations for Foundation Models
- **分类: cs.MM; cs.AI; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于多模态评估任务，旨在解决现有数据集的标注不全、类别重叠等问题。作者提出VGGSounder，改进标注并设计新指标以更准确评估音频-视觉基础模型。**

- **链接: [https://arxiv.org/pdf/2508.08237](https://arxiv.org/pdf/2508.08237)**

> **作者:** Daniil Zverev; Thaddäus Wiedemer; Ameya Prabhu; Matthias Bethge; Wieland Brendel; A. Sophia Koepke
>
> **备注:** Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** The emergence of audio-visual foundation models underscores the importance of reliably assessing their multi-modal understanding. The VGGSound dataset is commonly used as a benchmark for evaluation audio-visual classification. However, our analysis identifies several limitations of VGGSound, including incomplete labelling, partially overlapping classes, and misaligned modalities. These lead to distorted evaluations of auditory and visual capabilities. To address these limitations, we introduce VGGSounder, a comprehensively re-annotated, multi-label test set that extends VGGSound and is specifically designed to evaluate audio-visual foundation models. VGGSounder features detailed modality annotations, enabling precise analyses of modality-specific performance. Furthermore, we reveal model limitations by analysing performance degradation when adding another input modality with our new modality confusion metric.
>
---
#### [replaced 003] SpeechEditBench: A Bilingual Multi-Attribute Benchmark for Instruction-Guided Speech Editing
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出SpeechEditBench，一个用于评估指令引导语音编辑能力的多属性基准。解决现有评估碎片化问题，通过原子与组合任务及评估协议，分析模型表现，推动语音大模型发展。**

- **链接: [https://arxiv.org/pdf/2606.01804](https://arxiv.org/pdf/2606.01804)**

> **作者:** Hanlin Zhang; Daxin Tan; Dehua Tao; Xiao Chen; Haochen Tan; Linqi Song
>
> **摘要:** Instruction-guided speech editing requires a model to modify specified speech attributes while preserving unrelated characteristics. Despite rapid progress in Speech Large Language Models (Speech LLMs), systematic evaluation of this capability remains challenging, as existing benchmarks are fragmented across isolated editing tasks. To bridge this gap, we introduce SpeechEditBench, a bilingual multi-attribute benchmark for instruction-guided speech editing. SpeechEditBench encompasses seven atomic editing tasks, as well as compositional editing tasks that integrate multiple operations within a single instruction. We propose an anchor-based evaluation protocol that separately assesses the edit success of target attributes and the preservation of untargeted attributes, leading to three metrics: target success, preservation success, and joint success. Using this benchmark, we evaluate mainstream Speech LLMs and specialized speech editing systems. The results reveal three key findings: (1) no single model performs well across all editing dimensions; (2) closed-source Speech LLMs generally outperform open-source models; (3) compositional editing remains highly challenging, with even the most advanced models struggling to achieve high joint success. SpeechEditBench provides a rigorous diagnostic framework to identify bottlenecks in Speech LLMs, thereby facilitating the development of next-generation Speech LLMs with more robust and precise instruction-guided editing capabilities. Data and code are avaialble at this https URL .
>
---
#### [replaced 004] AUDDT: A Unified Benchmark Toolkit for Audio and Speech Deepfake Detectors
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决现有基准数据集有限、检测器泛化能力不足的问题。工作包括构建统一的评估工具AUDDT，支持多场景检测分析。**

- **链接: [https://arxiv.org/pdf/2509.21597](https://arxiv.org/pdf/2509.21597)**

> **作者:** Yi Zhu; Heitor R. Guimarães; Arthur Pimentel; Tiago Falk
>
> **摘要:** With the prevalence of artificial intelligence (AI)-generated content, such as audio deepfakes, a large body of recent work has focused on developing deepfake detection techniques. However, existing benchmarks employ a narrow set of datasets, leaving detector generalization to real-world conditions uncertain. In this paper, we systematically review 31 existing audio deepfake datasets and present an open-source benchmarking toolkit called AUDDT (this https URL). The goal of this toolkit is to automate the evaluation of pretrained detectors across a wide range of speech and non-speech audio datasets, giving users direct feedback on the advantages and shortcomings of their deepfake detectors under diverse manipulation types and recording conditions. We start by showcasing the usage of the developed toolkit, the composition of our benchmark, and the breakdown of different deepfake subgroups. Next, we highlight how AUDDT differs from existing benchmarking efforts by enabling large-scale, diverse evaluation across modern spoofing methods and richer attribute-level analysis through comprehensive metadata annotation. Using a widely adopted pretrained deepfake detector, we present in- and out-of-domain detection results, revealing notable performance variability across different conditions and audio manipulation types. Lastly, we also analyze the limitations of these existing datasets and their gaps relative to practical deployment scenarios.
>
---
#### [replaced 005] A Study of the Scale Invariant Signal to Distortion Ratio in Speech Separation with Noisy References
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文研究语音分离任务中使用SI-SDR评估和训练时，噪声参考信号的影响。提出增强参考信号的方法，以减少分离结果中的噪声。**

- **链接: [https://arxiv.org/pdf/2508.14623](https://arxiv.org/pdf/2508.14623)**

> **作者:** Simon Dahl Jepsen; Mads Græsbøll Christensen; Jesper Rindom Jensen
>
> **备注:** Accepted for IEEE ASRU 2025, Workshop on Automatic Speech Recognition and Understanding. Copyright (c) 2025 IEEE. 8 pages, 6 figures, 2 tables
>
> **摘要:** This paper examines the implications of using the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) as both evaluation and training objective in supervised speech separation, when the training references contain noise, as is the case with the de facto benchmark WSJ0-2Mix. A derivation of the SI-SDR with noisy references reveals that noise limits the achievable SI-SDR, or leads to undesired noise in the separated outputs. To address this, a method is proposed to enhance references and augment the mixtures with WHAM!, aiming to train models that avoid learning noisy references. Two models trained on these enhanced datasets are evaluated with the non-intrusive NISQA.v2 metric. Results show reduced noise in separated speech but suggest that processing references may introduce artefacts, limiting overall quality gains. Negative correlation is found between SI-SDR and perceived noisiness across models on the WSJ0-2Mix and Libri2Mix test sets, underlining the conclusion from the derivation.
>
---
#### [replaced 006] SpeakerCard-1M: An Evidence-Grounded Speaker Card Corpus for In-the-Wild Speaker Verification
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出SpeakerCard-1M，用于开放域说话人验证，解决现有数据缺乏说话人级监督的问题。构建了多模态数据集，并设计了跨模态评估协议。**

- **链接: [https://arxiv.org/pdf/2606.03283](https://arxiv.org/pdf/2606.03283)**

> **作者:** Junyi Peng; Oldřich Plchot; Xiao Song; Dading Chong; Lichun Fan; Hang Su; Themos Stafylakis; Junjie Li; Kong Aik Lee; Shuai Wang; Jian Luan; Jan Černocký
>
> **备注:** Corpus and protocols at this https URL
>
> **摘要:** Modern speaker verification (SV) systems rely on speaker embeddings that are effective but difficult to interpret or query in natural language. Most existing speech-text corpora target controllable synthesis or utterance-level captioning, and provide limited speaker-level supervision for in-the-wild speaker recognition. This paper introduces SpeakerCard-1M, a bilingual speaker-centric resource for evidence-grounded SV, derived from VoxCeleb1/2 and CN-Celeb1/2, where the "-1M" suffix refers to the 1.78M utterance-level captions contained in the release. We adopt a tool-first, LLM-last approach: ten acoustic probes produce field-level evidence, the evidence is aggregated into speaker profiles under a schema that separates relatively stable traits from utterance-level states, and bilingual Speaker Cards are rendered by a constrained LLM that sees only the structured fields. The release includes 56.7K Speaker Card records over 10.2K speakers, 1.78M utterance-level captions, and speaker-ID-disjoint hard-negative triplets. We further define two SV-oriented cross-modal protocols, bidirectional Speaker-Text Retrieval (T2S-R / S2T-R) and Attribute-Conditioned Verification (AC-Verify), and compare a dual-encoder baseline against recent audio language models under a zero-shot forced-choice setting. Joint audio-text training increases VoxCeleb1-O EER by 0.31% absolute over the audio-only baseline. Under a style-symmetric LLM-generated counterfactual protocol, eight recent audio language models (7B-30B+ parameters, both open- and closed-source) score 49-77% on pitch-level AC-Verify under two-way forced choice, compared with 88.66% reached by our dual encoder.
>
---
#### [replaced 007] Analysis-Driven Procedural Generation of an Engine Sound Dataset with Embedded Control Annotations
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音频生成任务，旨在解决发动机声音数据获取困难的问题。通过分析驱动的方法生成带有精确控制标注的发动机声音数据集，以支持声学建模与合成研究。**

- **链接: [https://arxiv.org/pdf/2603.07584](https://arxiv.org/pdf/2603.07584)**

> **作者:** Robin Doerfler; Lonce Wyse
>
> **备注:** To appear in the Proceedings of the 34th European Signal Processing Conference (EUSIPCO 2026)
>
> **摘要:** Computational engine sound modeling is central to the automotive audio industry, particularly for active sound design applications and virtual prototyping. Emerging data-driven engine sound synthesis methods require large volumes of standardized, clean audio recordings with precisely time-aligned operating-state annotations: data that is difficult to obtain due to high costs, specialized measurement equipment requirements, and inevitable noise contamination. We present an analysis-driven framework for generating engine audio with sample-accurate control annotations. The method extracts harmonic structures from real recordings through pitch-adaptive spectral analysis, which then drive an extended parametric harmonic-plus-noise synthesizer. With this framework, we augment 5-10 min of source audio per engine 15-30x via diverse control trajectories and parametric variation, producing the Procedural Engine Sounds Dataset (19.0 h, 5,935 files): a set of engine audio signals with sample-accurate RPM and torque annotations spanning a wide range of operating conditions, signal complexities, and harmonic profiles. Comparison against real recordings validates that the synthesized data preserves characteristic harmonic structures, and a baseline differentiable synthesis network trained on the dataset confirms its suitability for data-driven engine sound modeling. The dataset is released publicly to support research on engine timbre analysis, control parameter estimation, and neural generative synthesis.
>
---
#### [replaced 008] Physics-Informed Neural Engine Sound Modeling with Differentiable Pulse-Train Synthesis
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频合成任务，旨在更准确地建模发动机声音。通过直接模拟脉冲形状和时间结构，提出PTR模型，解决传统方法在谐波重建上的不足。**

- **链接: [https://arxiv.org/pdf/2603.09391](https://arxiv.org/pdf/2603.09391)**

> **作者:** Robin Doerfler; Lonce Wyse
>
> **备注:** Revised version; to appear in the Proceedings of the 34th European Signal Processing Conference (EUSIPCO 2026)
>
> **摘要:** Engine sounds originate from sequential exhaust pressure pulses rather than sustained harmonic oscillations. While neural synthesis methods typically aim to approximate the resulting spectral characteristics, we propose directly modeling the underlying pulse shapes and temporal structure. We present the Pulse-Train-Resonator (PTR) model, a differentiable synthesis architecture that generates engine audio as parameterized pulse trains aligned to engine firing patterns and propagates them through recursive Karplus-Strong resonators simulating exhaust acoustics. The architecture integrates physics-informed inductive biases including harmonic decay, thermodynamic pitch modulation, valve-dynamics envelopes, exhaust system resonances and derived engine operating modes such as throttle operation and Deceleration Fuel Cutoff (DFCO). Validated on three diverse engine types totaling 7.5 hours of audio, PTR achieves a 21% improvement in harmonic reconstruction and a 5.7% reduction in total loss over a harmonic-plus-noise baseline model, while providing interpretable parameters corresponding to physical phenomena. Complete code, model weights, and audio examples are openly available.
>
---
#### [replaced 009] Extracting accent features in spoken Brazilian Portuguese without sociolinguistic labels
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音识别中的方言分类任务，旨在解决缺乏可靠社会语言标签的问题。通过仅使用声学标签提取特征，提升区域口音识别效果。**

- **链接: [https://arxiv.org/pdf/2605.30457](https://arxiv.org/pdf/2605.30457)**

> **作者:** Pedro H. L. Leite; Pedro Benevenuto Valadares; Luiz W. P. Biscainho
>
> **备注:** This work was submitted to the XLIV Brazilian Symposium on Telecommunications and Signal Processing (SBrT 2026)
>
> **摘要:** Regional accent classification in Brazilian Portuguese (pt-BR) suffers from the need for reliable labeling. While large self-supervised learning (SSL) speech models are powerful, their training pipelines dilute sociophonetic information, since accent labels are generally not reliable or are not used in training objectives. This work introduces a novel workflow for feature extraction using only acoustic labels. By isolating explicit regional accent landmarks and using a phoneme-based forced aligner (ZIPA), our targeted feature set captures dialectal variance more effectively than utterance embeddings, demonstrating that localized features can outperform general-purpose architectures on accent-related tasks using minimal and objective data labels.
>
---
