# MUSE: Music Recommender System with Shuffle Play Recommendation Enhancement

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://uobevents.eventsair.com/cikm2023//" alt="Conference">
        <img src="https://img.shields.io/badge/CIKM'23-green" /></a>
</p>

The official source code for "[MUSE: Music Recommender System with Shuffle Play Recommendation Enhancement]()", accepted at CIKM 2023.


## Overview

Recommender systems have become indispensable in music streaming services, enhancing user experiences by personalizing playlists and facilitating the serendipitous discovery of new music. However, the existing recommender systems overlook the unique challenges inherent in the music domain, specifically shuffle play, which provides subsequent tracks in a random sequence. Based on our observation that the shuffle play sessions hinder the overall training process of music recommender systems mainly due to the high unique transition rates of shuffle play sessions, we propose a Music Recommender System with Shuffle Play Recommendation Enhancement (**MUSE**). **MUSE** employs the self-supervised learning framework that maximizes the agreement between the original session and the augmented session, which is augmented by our novel session augmentation method, called transition-based augmentation. To further facilitate the alignment of the representations between the two views, we devise two fine-grained matching strategies, i.e., item- and similarity-based matching strategies. Through rigorous experiments conducted across diverse environments, we demonstrate **MUSE**’s efficacy over 12 baseline models on a large-scale Music Streaming Sessions Dataset (MSSD) from Spotify.

<p float="middle">
  <img src="https://github.com/yunhak0/MUSE/assets/40286691/5eaf7132-9df0-4ff8-820c-88fc646c1f62" width="51.5%" /><img src="https://github.com/yunhak0/MUSE/assets/40286691/a6296eca-cd3c-4741-b069-7dbaac535bee" width="40%" /> 
</p>


## Requirements

* `python`: 3.9.17
* `pytorch`: 1.11.0
* `numpy`: 1.25.2
* `pandas`: 1.5.3
* `scipy`: 1.11.1
* `pyyaml`: 6.0
* `tqdm`: 4.65.0
* `pyarrow`: 11.0.0 or `fastparquet`: 2023.4.0 (for preprocessing)


## Data Preprocessing

Data is available in the [original challenge](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge) with [published paper](https://arxiv.org/pdf/1901.09851.pdf)[1].
For the data preprocessing, please refer in './data/mssd-org/README.md' file.

:hourglass: You can find the preprocessed data [here](https://drive.google.com/drive/folders/1D6OTdSsgRcVvTn-WD98FfiJpPNd6mGtm?usp=drive_link).


## How to run

> python main.py

The arguments and its description is in './utils/argument.py' file.


## Reference

[1] B. Brost, R. Mehrotra, and T. Jehan, The Music Streaming Sessions Dataset (2019), Proceedings of the 2019 Web Conference


## Code Collaborator

@[SukwonYun](https://github.com/SukwonYun), @[Sein-Kim](https://github.com/SukwonYun)
