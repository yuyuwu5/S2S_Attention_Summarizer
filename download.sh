#!/usr/bin/env bash

word2index="https://www.dropbox.com/s/tjicxdiwqtwlssn/word2index.json?dl=1"
index2word="https://www.dropbox.com/s/jnal9b1chvia4se/index2word.json?dl=1"
embedding="https://www.dropbox.com/s/pff2ho3rzzshhh3/wordEmbedding.npy?dl=1"
extractive="https://www.dropbox.com/s/a1owcq8cwarg4cj/Tag.model?dl=1"
s2s="https://www.dropbox.com/s/awz31ebzc4ngzyy/S2S.model?dl=1"
attention="https://www.dropbox.com/s/5d7mqjd07vtdttk/attention.model?dl=1"

wget "${word2index}" -O ./data/word2index.json
wget "${index2word}" -O ./data/index2word.json
wget "${embedding}" -O ./data/wordEmbedding.npy
wget "${extractive}" -O ./Tag/Tag.model
wget "${s2s}" -O ./S2S/S2S.model
wget "${attention}" -O ./Attention/attention.model
