# Data Card: Deepfake Detection Dataset

## Dataset Overview

This dataset is a processed subset of the **Deep Fake Detection (DFD)** dataset, prepared for a deepfake detection project focused on **temporal motion structure** rather than static frame-level artefacts.

The dataset supports experiments using optical-flow-derived motion features and a CNN-LSTM temporal model. Each sample is treated as a video sequence rather than an isolated image, allowing the model to learn from frame-to-frame behaviour.

The dataset is not original. It is a project-specific processed version of an existing public deepfake detection dataset.

---

## Purpose and Scope

The dataset was prepared to develop and evaluate deepfake video detection methods based on **motion inconsistency**, **temporal coherence**, and **optical flow features**.

The central assumption is that generated or manipulated videos may contain motion patterns that differ from physically captured video. These differences may appear in the residual structure between frames, especially after optical flow is used to estimate inter-frame motion.

The dataset was therefore organised to support a temporal detection pipeline rather than a purely spatial classifier.

---

## Source Dataset

The data was sourced from the **Deep Fake Detection (DFD)** dataset and then processed for this project.

Processing included:

* extracting frame sequences from real and fake videos
* standardising sequence length
* preparing the data for optical flow computation
* organising the processed data into train, test, and holdout sets

The preprocessing, extraction, and organisation were carried out as part of this independent research project.

---

## Dataset Composition

Each instance represents a video sequence made up of **305 consecutive frames**.

The dataset contains two classes:

* **Real sequences**: genuine recorded human video
* **Fake sequences**: deepfake-generated or manipulated video

The dataset was constructed as a balanced set of real and fake sequences where possible. Each sequence is intended to provide enough temporal information for motion-based analysis.

The dataset is a subset of a larger source dataset and was selected according to processing feasibility, availability, and relevance to motion-based detection.

---

## Labels

The dataset uses binary labels:

* `0` = real video
* `1` = fake video

These labels are used for supervised classification.

---

## Data Structure

Each instance consists of:

* a sequence of 305 extracted frames
* a corresponding class label
* optical-flow feature maps computed dynamically during training

Optical flow was not stored permanently as precomputed arrays. Instead, it was calculated during the model pipeline to reduce the already substantial storage burden.

---

## Train, Test, and Holdout Splits

The dataset was split into:

* **Training set**: 80%
* **Test set**: 20%
* **Holdout set**: 1 real sequence and 1 fake sequence for final validation

The holdout set was used as a small final check after training and testing.

---

## Preprocessing Pipeline

The preprocessing pipeline included:

1. **Frame extraction**
   Videos were split into individual image frames.

2. **Sequence standardisation**
   Each video sequence was trimmed or padded to 305 frames.

3. **Normalisation**
   Frames were normalised for model compatibility and to reduce variation caused by inconsistent raw video formatting.

4. **Optical flow preparation**
   Optical flow was computed dynamically during training to capture frame-to-frame motion patterns.

5. **Dataset splitting**
   Processed sequences were divided into training, testing, and holdout sets.

This preprocessing stage was one of the most practically difficult parts of the project. Extracting frames from video, standardising sequence lengths, normalising frame data,


