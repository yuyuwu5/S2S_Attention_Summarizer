# ADL HW1 Summarization
			 
## Preprocessing
* 將glove word vector轉成gensim可以讀的形式
  python -m gensim.scripts.glove2word2vec --input  glove.840B.300d.txt --output glove.840B.300d.w2vformat.txt
* Run python buildVabEmbedding.py 
  會在指定data資料夾生成index2word.json, word2index.json, wordEmbedding.npy
## Extractive: Sequence Tagging
* 進入Tag資料夾
* Run python buildTagDataset.py 會在data資料夾生成trainTag.pkl, validTag.pkl, testTag.pkl
* Run python train.py 開始訓練
* Run python Tag_prediction.py --test_data_path <file> --output_path <file> 開始預測
* Run python draw_tag.py畫report 的histogram
## Abstractive: sequence to  sequence model
* 進入S2S資料夾
* Run python buildS2SDataset.py 會在data資料夾生成trainS2S.pkl, validS2S.pkl, testS2S.pkl
* Run python train.py開始訓練
* Run python S2S_prediction.py --test_data_path <file> --output_path <file> 開始預測
## Abstractive: sequence to  sequence with attention
* 進入Attention資料夾
* Run python buildS2SDataset.py 會在data資料夾生成trainS2S.pkl, validS2S.pkl, testS2S.pkl
* Run python train.py開始訓練
* Run python S2S_attention_prediction.py --test_data_path <file> --output_path <file> 開始預測
* 進入draw資料夾，Run python draw_attention.py可畫出attention weight圖
