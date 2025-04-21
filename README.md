## Overview

This repository contains a detailed analysis of corpora belonging to different genres using different techniques such as Bag-of-Words (BoW), Naïve Bayes probability modeling, and topic modeling via Latent Dirichlet Allocation (LDA).

### Dataset

The dataset is composed of chapters extracted from romance and murder mystery novels sourced from [Project Gutenberg](https://www.gutenberg.org/). Each chapter is treated as an individual document and organized under two categories: **Romance** and **Crime**.

### Project Objectives

The main goals of this assignment were to:

- Convert raw text data into BoW representations (both standard and binary).
- Calculate log-likelihood ratios (LLR) using a Naïve Bayes-style estimation.
- Perform topic modeling using LDA.
- Experiment with the impact of text normalization and BoW variations on output quality.

### File Structure

- `text_data/` – Contains chapter-wise split text files categorized into romance and crime.
- `normalize_text_module.py` – Custom module for preprocessing and text normalization (configurable).
- `count_bow.py` – Script to generate standard BoW representations and compute LLR.
- `binary.py` – Alternate script to evaluate binary BoW representation and compare effects.
- `lda_vis.html` – LDA visualization generated using default BoW settings.
- `lda_vis_bin.html` – LDA visualization based on binary BoW representation.
- `Report.pdf` – Fully documented analysis, methodology, visualizations, and insights.
- `README.md` – Repository guide and structure (this file).

### Text Normalization Module

This project includes a `normalize_text_module.py` module that performs standard preprocessing tasks such as tokenization, lower-casing, stemming, etc. This module was adapted from my other repository [NormalizeText](https://github.com/scythemenace/NormalizeText) by refactoring the original script into a reusable module and enabling it to process arbitrary files rather than relying on a hardcoded input.

### Results & Analysis

A detailed explanation of the methodology, comparison between preprocessing strategies, log-likelihood word analysis, and topic modeling results can be found in the `Report.pdf`.
