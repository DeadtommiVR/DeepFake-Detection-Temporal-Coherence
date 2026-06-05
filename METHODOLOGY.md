# Methodology

*Deepfake detection via optical-flow residuals and a CNN-LSTM temporal model. Companion document to the [README](README.md) — §1 sets out the problem framing and stakes, §2 reviews the related literature.*

---

## 1. Problem Framing

### 1.1 Overview

The premise of this work is that synthetically generated video and physically captured video differ in a measurable way that is **not artefact-based but structural**. Generated video is produced by a *deterministic procedure* operating on a latent representation. Frame-to-frame coherence is achieved by interpolation in latent space, by attention mechanisms bridging frames, or by explicit temporal-consistency losses. These procedures carry detectable structural patterns between attended states, inherently distinct from the high-entropy noise signal of genuine capture environments. Even when the visible output is convincing, the residuals between frames are not independent in the way real sensor noise is. **The hypothesis is that this structural difference is measurable.**

**Conceptual definition of deepfake.** A subset of synthetic media or substantially manipulated video media produced by generative models, in which a subject's face, voice, or actions are fabricated or transferred from another source. The content might be created from scratch, or pre-existing content may have been manipulated. Deepfakes are often created with the intention to deceive, defame, or misrepresent someone or something.

> Sources:
> - The Alan Turing Institute — [What are deepfakes and how can we detect them?](https://www.turing.ac.uk/blog/what-are-deepfakes-and-how-can-we-detect-them)
> - UK Government — [Deepfake detection technology](https://www.gov.uk/government/publications/deepfake-detection-technology/deepfake-detection-technology)

**Operational definition (study-specific).** Used here in the broad sense: any AI-generated or AI-altered footage, whether produced by full synthesis, partial manipulation, frame interpolation, face-swap, or other procedural or technical permutation. *Not* restricted to the narrow sense of face-swap GAN outputs.

**Definition difficulties.** The category becomes harder to delimit at its edges. Mixed-media workflows, in which AI-generated elements are composited into otherwise real footage, or where real footage is selectively altered frame-by-frame, sit on a spectrum rather than in a binary. Human-in-the-loop pipelines, in which a generated output is finished by a colourist, editor, or VFX artist, further blur the boundary: the resulting footage is neither purely synthetic nor purely captured. For the purposes of this study, any footage whose temporal structure has been substantively produced or modified by a generative process is treated as within scope, on the understanding that the boundary is *gradient, not categorical*.

### 1.2 Project Development and Lineage

This hypothesis developed from an earlier project developing realistic synthetic motion in procedural animation rather than detecting it.

Having worked professionally as a filmmaker and editor, I began earlier module work investigating machine learning techniques for 3D character facial animation, specifically the problem of unnatural expression transfer in facial rigging interpolation. That work directly observed that conventional interpolation between expression states generates unnatural linear conversions. PCA was used as a dimensionality-reduction tool over facial landmark coordinates, to identify low-dimensional structure in motion patterns.

A subsequent module piece applied a similar framework to deepfake detection directly, using KNN over PCA-reduced facial landmark and temporal features. The present project is the synthesis and development of those two threads: taking the detection question from the second, the motion-and-PCA framework from the first, and grounding both in the underlying theory that *deterministic synthesis necessarily leaves a statistical signature in the residual structure of the generated signal*.

Addressing deepfake detection from this angle reframes the question. The relevant question is not *"what artefact does this generator produce"* — rather the question is *"what statistical signature does any deterministic construction process necessarily leave behind."* If that signature exists and can be characterised, it is in principle **generator-agnostic**.

### 1.3 Motion Fields vs. Artefacts

Traditional methods for deepfake detection focus on individual frames and learn spatial inconsistencies such as parallax, eye reflections, skin texture, and more frequency-domain artefacts. These methods report strong numbers on clean benchmarks but degrade sharply on in-the-wild content and on novel generation methods.

Motion is a complementary and arguably stronger signal for the determinism-signature hypothesis specifically. The physical grammar of how a real camera records a dynamic environment — such as lens characteristics, rolling shutter, micro-jitter, focus breathing, perspective, essentially the independent noise signature of every frame — is hard for a generative model to reproduce without leaving formulaic traces. Procedural and synthetic frame transformation tends to produce motion fields that are *smoother* than physically captured ones, with cross-frame dependencies that real sensor noise does not have.

**Optical flow** is the per-pixel motion field between two consecutive video frames, representing how each region of the image has shifted in space and time. It is computed by estimating, for every pixel in frame *n*, where the same content has moved to in frame *n+1*. **Optical flow residuals** are the difference between the predicted next frame and the actual next frame, after motion compensation. Once isolated, they are the part of the inter-frame variation that is *not* explainable by physical motion — and therefore the place the determinism signature should be most visible.

### 1.4 Limits of Detection

The evolving landscape yields entirely new attack vectors in all domains. It's in these cases that the adversary is sophisticated, well-resourced, and highly motivated to defeat detection. The pipeline-versus-finished-output distinction becomes acute where a state actor or organised group producing deepfakes has the time and skill to layer analogue artefacts, regrade, blend with real footage, simulate plausible compression histories, even film the synthetic output off a screen with a real camera or using virtual production. Despite regulatory attempts and asset-manifest specifications, metadata fields can be easily manipulated or removed entirely by bad actors using file or metadata editing tools, and watermark embedding or cryptographic binding can be bypassed using autoencoders.

From my personal experience in professionally creating media, I know that the rapidly evolving processes to create deepfakes will likely undermine tangible detection in the wild in many cases. Detection is one intervention in a much bigger and complex problem. The deeper issue is structural, concerning how decision structures are engineered and how trust is established and maintained within them. The real answer to cracking the deepfake riddle is redesigning these architectures so that authenticity can be established *at the source* rather than reconstructed after the fact. Perhaps the real question is how we can orchestrate an epistemic anatomisation of information ecosystems and redesign monolithic societal structures entirely. Engineering trust provenance is the new frontier. Perhaps it's the most lucrative future industry.

> "The technology intervenes especially because of developments in communications , and in general the handling of information. wherever you look within industry, within government, absolutely all human affairs are so richly interconected- the rise in the degree of interaction is on the exponential curve- that all sorts of knitting happens across vertical structures. the vertical monoliths on the blackboard cannot handle this sort of thing. insofar that they are aware of the new problems, the vertical monolith,s try to respond by the establishment of properly regulated inter monolith committees. we must note that these will not meet the need very well, if the need is indeed for the demolition and rebuilding of monoliths themselves"
>
> — **Stafford Beer**, *Platform for Change*, 1975, p.284

### 1.5 Stakes and Provenance

The first deepfakes were created using Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs). However, the use of these technologies have evolved and the availability of new tools powered by Generative AI (GenAI), which allow users to create wholly new content that is significantly more convincing. These tools have also made it easier to create deepfakes, such that anybody with modest technical skills can do so.

Detection research is often benchmarked against *entertainment-tier* media: face swaps in viral clips, celebrity impersonation, essentially low-effort manipulation made with cheap accessible software. These cases are somewhat simple, and they are largely solved with decent models that can identify them well. However, higher-stake synthetic media is often harder to detect. In the research paper titled *Deepfakes and Cheapfakes: The Manipulation of Audiovisual Evidence*, Britt Paris explains how "both can be used to influence the politics of evidence: how evidence changes and is changed by its existence in cultural, social, and political structures" ([Data & Society, 2019](https://datasociety.net/wp-content/uploads/2019/09/DS_Deepfakes_Cheap_FakesFinal-1-1.pdf)).

High-stakes synthetic media spans several distinct domains:

- **Political** — fabricated speeches, manufactured statements, or synthetic footage of public figures, particularly around elections or geopolitical flashpoints.
- **Legal and evidentiary** — synthetic content presented as evidence in court, or genuine evidence undermined by plausible-deniability claims of synthesis.
- **Intimate-image abuse** — non-consensual sexual content generated to harass, extort, or harm individuals, disproportionately affecting women and minors.
- **Cybersecurity** — voice and video synthesis used to amplify existing attack vectors (impersonation, social engineering, fraud) and to enable new ones.
- **Human rights documentation** — synthetic footage used to fabricate atrocities, or used to discredit genuine documentation of them.

On that last point, as Sam Gregory observed in *MIT Technology Review*:

> "this fear that real evidence can plausibly be dismissed as fake gives another weapon to the powerful — to say 'it's a deepfake' about anything that people who are out of power are trying to use to show corruption, to show human rights abuses."

#### Example cases

- [GNET — Deepfake doomsday: AI amplifying apocalyptic Islamist propaganda](https://gnet-research.org/2023/08/29/deepfake-doomsday-the-role-of-artificial-intelligence-in-amplifying-apocalyptic-islamist-propaganda/)
- [Microsoft On the Issues — Digital threats and cyberattacks from East Asia](https://blogs.microsoft.com/on-the-issues/2023/09/07/digital-threats-cyberattacks-east-asia-china-north-korea/)
- [Bloomberg — Trolls in Slovakian election tap AI deepfakes to spread disinfo](https://www.bloomberg.com/news/articles/2023-09-29/trolls-in-slovakian-election-tap-ai-deepfakes-to-spread-disinfo)
- [WIRED — Yahoo Boys and real-time deepfake scams](https://www.wired.com/story/yahoo-boys-real-time-deepfake-scams/)
- [BBC News](https://www.bbc.co.uk/news/uk-66130785)
- [SAGE Journals — academic article](https://journals.sagepub.com/doi/full/10.1177/0964663920947791)
- [Ofcom — Open data and research](https://www.ofcom.org.uk/about-ofcom/our-research/opendata)
- [CapX — The deepfake general election](https://capx.co/the-deepfake-general-election/)
- [ABC News — UK election: pro-Russian Facebook pages coordinating](https://www.abc.net.au/news/2024-06-29/uk-election-pro-russian-facebook-pages-coordinating/104038246)
- [Women in Journalism — Condemning deepfake attacks on Cathy Newman](https://www.womeninjournalism.org/threats-all/uk-women-press-freedom-condemns-deepfake-attacks-on-cathy-newman-as-part-of-a-growing-trend-against-journalists)
- [National Crime Agency — Urgent warning about sextortion](https://www.nationalcrimeagency.gov.uk/news/nca-issues-urgent-warning-about-sextortion)
- [Fenimore Harper — Deepfake political advertising](https://www.fenimoreharper.com/research/deepfake-political-advertising)
- [404 Media — AI-generated non-consensual imagery of ordinary people](https://www.404media.co/irl-fakes-where-people-pay-for-ai-generated-porn-of-normal-people/)

#### Surveys and analysis

- [arXiv — Deepfake survey (2407.05529)](https://arxiv.org/pdf/2407.05529v1)
- [Ofcom — Open data and research](https://www.ofcom.org.uk/about-ofcom/our-research/opendata)
- [Channel 4 News — British celebrities targeted by deepfake abuse imagery](https://www.channel4.com/news/exclusive-hundreds-of-british-celebrities-victims-of-deepfake-porn)
- [The Verge — Sensity report on Telegram deepfake bot](https://www.theverge.com/2020/10/20/21519322/deepfake-fake-nudes-telegram-bot-deepnude-sensity-report)
- [Home Security Heroes — State of Deepfakes report](https://www.homesecurityheroes.com/state-of-deepfakes/#appendix)

#### Deepfake applications

- [My Image My Choice — Take action](https://myimagemychoice.org/take-action/)
- [Business Insider — Apple removes apps promoting non-consensual image generation](https://markets.businessinsider.com/news/stocks/apple-removed-ai-apps-from-its-appstore-which-promoted-creating-non-consensual-nude-images-1033295650)

---

## 2. Related Work

Deepfake video detection has, broadly, organised itself around two questions: *what* signal betrays a manipulation, and *where* in the data that signal can be found. The literature this project sits within can be read along the axis from methods that look for spatial artefacts within individual frames, through methods that look for temporal inconsistency across frames, to the ensemble and generative approaches that currently top the benchmarks. This project belongs to the temporal-coherence branch, and the sections below place it relative to the work that defines that branch and the work that surrounds it.

### 2.1 Frame-Based Detection

The dominant paradigm treats deepfake detection as a per-frame image-classification problem. Early work in this vein looked for visible spatial artefacts and inconsistencies in regions that generation pipelines reconstruct poorly. Methods of this kind detect the residue left when a synthesised face is composited onto a target, including resolution mismatches at face-warping boundaries and the blending edges between a manipulated region and its surroundings.

The strength of frame-based methods is also their weakness. They report strong numbers on clean benchmarks, but they depend on artefacts that improving generators are steadily removing, and they tend to degrade on content that differs from their training distribution. As generative models have advanced (from GANs and VAEs to diffusion-based synthesis) the per-frame visual trace they leave has diminished, and detectors built on those traces inherit a shrinking margin. This is the paradigm this project deliberately moved away from: the signal frame-based detection depends on is the one most directly eroded by both better generators and lossy distribution.

### 2.2 Motion-Based and Temporal Detection

The closer literature to this project treats a video as a *sequence* rather than a batch of frames, and looks for inconsistency in how the image changes over time. The intuition is that motion produced by a generative model need not obey the physical consistency that real camera motion does, even when each individual frame is plausible.

Two strands matter here. The first is the **optical-flow lineage** this project descends from directly. Amerini et al. proposed exploiting discrepancies in optical-flow fields between real and manipulated faces, feeding flow estimates into a CNN classifier and establishing optical flow as a usable detection substrate. Nassif et al. extended this line with an improved optical-flow estimation method, comparing CNN backbones (finding the VGG family strongest for flow-derived inputs) and examining the effect of training hardware; notably, their cross-dataset results show accuracy collapsing toward chance when a model trained on one dataset is evaluated on another — which is the generalisation problem in concrete form. The CNN-LSTM architecture used in the present project is a descendant of this thinking, combined with the recurrent-temporal approach of Güera and Delp, who used a CNN to extract per-frame features and an RNN to classify the resulting sequence.

The second strand is the more recent move toward **transformer-based temporal modelling**. *GC-ConsFlow* is the closest prior work to this project: it combines optical-flow residuals with global context in a deep framework, targeting exactly the motion-residual signal this project set out to exploit. Spatial-temporal video transformers such as *ISTVT* model intra-frame and inter-frame relationships jointly, with attention to interpretability. A methodological note also belongs here: the forward/backward frame-differencing mechanism used to sharpen short-term temporal sensitivity (adopted in this project's handling of inter-frame motion) has a clear analogue in dynamic facial-expression recognition, where Liu et al.'s *Differential Temporal Transformer* applies bidirectional frame differencing ahead of a transformer encoder to capture instantaneous variation. That work targets expression recognition rather than deepfake detection, so it is cited here as a *source of method, not as detection literature*; the relevance is the differencing technique, not the task.

### 2.3 Ensemble and Generative Approaches

A third group of methods represents the current state of the art on standard benchmarks, and is included here to mark where the field's performance ceiling sits — and why this project does not attempt to compete with it directly. *GenConViT* combines convolutional and transformer feature extractors (ConvNeXt and Swin) with autoencoder and variational-autoencoder branches, learning from both visual artefacts and latent-distribution cues, and reports high accuracy across DFDC, FaceForensics++, DeepfakeTIMIT, and Celeb-DF (v2). *DFCON* ensembles several strong backbones (MaxViT, CoAtNet, EVA-02) under a supervised-contrastive objective, explicitly targeting *in-the-wild* generalisation via the DFWild-Cup setting — the same real-world-robustness concern that motivates this project's emphasis on compression and post-production.

These approaches are methodologically distinct from the present work: they pursue accuracy through *scale and architectural combination*, where this project pursues a *single interpretable signal* (motion residuals) on a deliberately modest compute budget. They are the right point of comparison for *what is achievable* on benchmarks, and the wrong model for *what this project was trying to learn*.

### 2.4 Cross-Cutting Observations

Two observations recur across every group above, whether stated or implied.

The first is the **lab-to-deployment gap**: reported accuracy on clean, high-bitrate benchmark video systematically overstates real-world performance, because the high-frequency temporal residuals that detection relies on are precisely what platform re-encoding smooths away, amongst other encoding processes.

The second is **dataset familiarity**: models consistently perform best on the dataset they were trained on, and cross-dataset evaluation — where it is reported at all — tends to reveal sharp drops.

Both observations point the same direction, and both are taken up in this project's own limitations rather than treated as solved.

### 2.5 Position of This Work

Within this map, the present project sits in the motion-based branch (§2.2), closest in substrate to *GC-ConsFlow* — both rest on optical-flow residuals as the locus where a generative model's failure to preserve physical motion coherence should be measurable. It is distinct in two respects. **Architecturally**, it pairs a CNN spatial backbone with an LSTM temporal model rather than an end-to-end transformer, a decoupled design chosen for tractability under limited compute. **Conceptually**, it foregrounds the two cross-cutting problems of §2.4 — compression and post-production — not as future work bolted on at the end, but as the reason the chosen signal is fragile in exactly the high-stakes settings where detection would matter most. The contribution is therefore less a claim to state-of-the-art accuracy than an *honest characterisation of one temporal signal*: what it detects, and the conditions under which it stops working.

---

## References

1. I. Amerini, L. Galteri, R. Caldelli, and A. Del Bimbo, "Deepfake Video Detection through Optical Flow based CNN," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV) Workshops*, Seoul, Korea, 2019.
2. R. Caldelli, L. Galteri, I. Amerini, and A. Del Bimbo, "Optical Flow based CNN for detection of unlearnt deepfake manipulations," *Pattern Recognition Letters*, vol. 146, pp. 31–37, 2021, doi: 10.1016/j.patrec.2021.03.005.
3. A. B. Nassif, Q. Nasir, M. A. Talib, and O. M. Gouda, "Improved Optical Flow Estimation Method for Deepfake Videos," *Sensors*, vol. 22, no. 7, p. 2500, 2022, doi: 10.3390/s22072500.
4. D. Güera and E. J. Delp, "Deepfake Video Detection Using Recurrent Neural Networks," in *Proc. 15th IEEE Int. Conf. Advanced Video and Signal Based Surveillance (AVSS)*, Auckland, New Zealand, 2018, pp. 1–6, doi: 10.1109/AVSS.2018.8639163.
5. "GC-ConsFlow: Leveraging Optical Flow Residuals and Global Context for Robust Deepfake Detection," arXiv:2501.13435, 2025.
6. C. Zhao, C. Wang, G. Hu, H. Chen, C. Liu, and J. Tang, "ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection," *IEEE Trans. Information Forensics and Security*, vol. 18, pp. 1335–1348, 2023, doi: 10.1109/TIFS.2023.3239223.
7. D. Wodajo, S. Atnafu, and Z. Akhtar, "Deepfake Video Detection Using Generative Convolutional Vision Transformer," arXiv:2307.07036, 2023.
8. M. S. H. Shanto et al. (Team Straw Hats), "DFCON: Attention-Driven Supervised Contrastive Learning for Robust Deepfake Detection," IEEE SPS Signal Processing Cup 2025 (DFWild-Cup), arXiv:2501.16704, 2025.
9. W. Liu, L. Li, C. Yan, Y. Zhang, X. Cheng, X. Zhao, and M. Liu, "Dynamic Facial Expression Recognition of Learners via Adaptive Global Attention and Differential Temporal Transformer," *CAAI Transactions on Intelligence Technology*, vol. 11, no. 2, pp. 514–528, 2026, doi: 10.1049/cit2.70115.
10. L. Li, J. Bao, T. Zhang, H. Yang, D. Chen, F. Wen, and B. Guo, "Face X-Ray for More General Face Forgery Detection," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020.
