# DeepTradeRiskEsti_TXF
Trading Strategy Using Keras/Tensorflow LSTM and Custom Loss Function

使用LSTM來做台指期預測，自定義損失函數

損失函數概念參考自
https://github.com/happynoom/DeepTrade

但因為原始的概念是用在操作股票，reLU激勵之後只會有正數，如果用在台指期上就變成只會做多了
因此在keras 的activations 下面自己新增了一個 relu_inverse 函數，截斷x為正的情形，變成只輸出負數
(請參考__keras_modifications__) ,修改位置位於

模型訓練： start_train.bat (可修改裡面參數，0 為model0，只輸出做多訊號，1 為 model1，只輸出做空訊號 )


dataset download:
https://drive.google.com/open?id=0B9JaGZQ0H8ksNWNDQ1pkSzhhc1E

下載完後放在 datasetLoader 目錄下
