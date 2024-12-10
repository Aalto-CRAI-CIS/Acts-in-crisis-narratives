# Acts-in-crisis-narratives

The purpose of this project is to analyze narratives and conversations occurring during crises.
More specifically, this project investigates how computational analysis of acts in crisis news comments can aid more in-depth analyses of public perspectives on crises, and support crisis response and mitigation.

To address this, we selected the YouTube channels of a set of news agencies, collected user comments to their videos, and analyzed those comments using few-shot learning. The comments were classified into categories based on communicative acts described in CMC and conversation analytic research, using multilabel classification (e.g. Herring et al., 2005; Clark and Schaefer, 1989; Paakki et al., 2021; Stivers, 2013). See Paakki and Ghorbanpour (2025) below for more details. 

## Act annotation scheme

The acts used in the annotation of our data and training of models are based on CMC and conversation analytic literature (e.g. Clark and Schaefer, 1989; Herring et al., 2005; Stivers, 2013). See Paakki and Ghorbanpour (2024) below for more details. Acts used include:  `["question", "challenge", "request", "statement", "appreciation", "denial", "apology", "acceptance"]`

Annotation guidelines: _(forthcoming)_

## Analysis

An example of the time series analysis conducted in this work is provided as a notebook in this repository.
- `Identify_significant_peaks_in_acts_NDTV_Phase2_multilabel_example.ipynb`:
This is a jupyter notebook file for which the input is a .csv file containing weekly frequencies of different acts in YouTube comments to crisis videos. The notebook code will draw a time series analysis on the weekly data using differences from group means to measure frequencies of acts.

## Models

If you wish to utilize our act classifier models, please see: https://huggingface.co/CrisisNarratives

## Citation

If you use any of the resources described or provided here, please cite:

Paakki, H. & Ghorbanpour, F. 2025: _(forthcoming)_ Computational Analysis of Communicative Acts for Understanding Crisis News Comment Discourses. Social Networks Analysis and Mining, 16th International Conference, ASONAM 2024, Rende, Italy, September 2–5, 2024, Proceedings.
