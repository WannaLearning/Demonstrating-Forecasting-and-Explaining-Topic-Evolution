***A Framework for Demonstrating, Forecasting, and Explaining Topic Evolution by Analyzing Geometrical Motion of Topic Embeddings***

This repository contains the code to reproduce the reults of our paper:  
1. Extract_cs_data.py -> collect and preprocess the data from the MAG dataset. (The MAG dataset can be downloaded from https://www.aminer.cn/oag-2-1)
2. vec_by_bert.py -> generate the topic embeddings via BERT model; A decoder is trained to decode a embeddings to fos.
3. vec_by_mpnet.py -> generate the topic embeddings via MPNET model; A decoder is trained to decode a embeddings to fos. (The MPNET model can be downloaded from https://www.sbert.net)
4. Semantic_Movement_Verify.py -> Section 3.2	Demonstrating the motion of topic embeddings via SVM.
5. Semantic_Movement_Predict.py -> Section 3.3 Forecasting motion via vector regression models.
6. Semantic_Movement_Explain.py -> Section 3.4Explaining forecast motion via a text generation model.
