# -*- coding: utf-8 -*-

import os,sys
from rawdata import RawData, read_sample_data
from dataset import DataSet
from datasetLoader import chart
import numpy
import pandas as pd

days_for_test_perc = 700/6465
days_for_validate_perc = 0.133

def generateDataSetTXF(rootDir , input_wind_size=30 , toDataSet=False ):
    
    selector = ["ROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME",
        "TXFTWSE_DIFF_ROCP","EXFFXF_DIFF_ROCP","UDV_ROCP","BAV_ROCP","TBA_ROCP",
        "TXFTWSE_RATIO_ROCP","EXFFXF_RATIO_ROCP","UDV_RATIO_ROCP","BAV_RATIO_ROCP","TBA_RATIO_ROCP"
        ]
    
    dataset_dir = rootDir + '\\datasetLoader\\datasetTXF\\'
    data_txf = dataset_dir + 'TXF1Min.txt'
    data_twse = dataset_dir + 'TWSE1Min.txt'
    
    data_exf = dataset_dir + 'EXF1Min.txt'
    data_fxf = dataset_dir + 'FXF1Min.txt'
    
    data_tx_dv = dataset_dir + 'TXF1_DV1Min.txt'
    data_tx_uv = dataset_dir + 'TXF1_UV1Min.txt'
    
    data_tx_av = dataset_dir + 'TXF1_AV1Min.txt'
    data_tx_bv = dataset_dir + 'TXF1_BV1Min.txt'
    
    data_tx_ta = dataset_dir + 'TXF1_TA1Min.txt'
    data_tx_tb = dataset_dir + 'TXF1_TB1Min.txt'
    
    
    df_txf_raw = pd.read_csv(data_txf,index_col=0, parse_dates=[['Date', 'Time']]) 
    df_txf_raw['TotalVolume'] = df_txf_raw['TotalVolume'].astype('float64')
    df_twse_raw  = pd.read_csv(data_twse,index_col=0,parse_dates=[['Date', 'Time']]) 
    df_twse_raw.rename(columns={'Close': 'TWSE'}, inplace=True) 
    
    df_txfTwse_diff = pd.merge(pd.DataFrame(df_txf_raw['Close']),
                      pd.DataFrame(df_twse_raw['TWSE']),
                      left_index=True,  
                      right_index=True, 
                      ) 
    df_txfTwse_diff['txfTwseRatio'] = df_txfTwse_diff.Close / df_txfTwse_diff.TWSE 
    df_txfTwse_diff['txfTwseDiff'] = df_txfTwse_diff.Close - df_txfTwse_diff.TWSE  
    df_txfTwse_ratio = pd.DataFrame(df_txfTwse_diff['txfTwseRatio']) 
    df_txfTwse_diff = pd.DataFrame(df_txfTwse_diff['txfTwseDiff']) 
    
    
    df_exf_raw = pd.read_csv(data_exf,index_col=0,parse_dates=[['Date', 'Time']]) 
    df_fxf_raw  = pd.read_csv(data_fxf,index_col=0,parse_dates=[['Date', 'Time']]) 
    df_fxf_raw.rename(columns={'Close': 'FXF'}, inplace=True) 
    df_exfFxf_diff = pd.merge(pd.DataFrame(df_exf_raw['Close']),
                               pd.DataFrame(df_fxf_raw['FXF']),
                               left_index=True,  
                               right_index=True, 
                               )
    df_exfFxf_diff['exfFxfRatio'] = (df_exfFxf_diff.Close * 4) / (df_exfFxf_diff.FXF * 1)
    df_exfFxf_diff['exfFxfDiff'] = (df_exfFxf_diff.Close * 4) - (df_exfFxf_diff.FXF * 1) 
    df_exfFxf_ratio = pd.DataFrame(df_exfFxf_diff['exfFxfRatio']) 
    df_exfFxf_diff = pd.DataFrame(df_exfFxf_diff['exfFxfDiff']) 
    
    
    df_uv_raw = pd.read_csv(data_tx_uv,index_col=0,parse_dates=[['Date', 'Time']]) 
    df_dv_raw = pd.read_csv(data_tx_dv,index_col=0,parse_dates=[['Date', 'Time']]) 
    df_dv_raw .rename(columns={'Close': 'DV'}, inplace=True) 
    df_uvdv_diff = pd.merge(pd.DataFrame(df_uv_raw['Close']),
                              pd.DataFrame(df_dv_raw['DV']),
                              left_index=True,  
                              right_index=True, 
                              )
    df_uvdv_diff['uvdv_ratio'] = df_uvdv_diff.Close / df_uvdv_diff.DV  
    df_uvdv_diff['uvdv'] = df_uvdv_diff.Close - df_uvdv_diff.DV  
    df_uvdv_ratio = pd.DataFrame(df_uvdv_diff['uvdv_ratio']) 
    df_uvdv_diff = pd.DataFrame(df_uvdv_diff['uvdv']) 
    
    
    
    df_bv_raw = pd.read_csv(data_tx_bv,index_col=0,parse_dates=[['Date', 'Time']]) 
    df_av_raw = pd.read_csv(data_tx_av,index_col=0,parse_dates=[['Date', 'Time']]) 
    df_av_raw.rename(columns={'Close': 'AV'}, inplace=True) 
    df_bvav_diff = pd.merge(pd.DataFrame(df_bv_raw['Close']),
                              pd.DataFrame(df_av_raw['AV']),
                              left_index=True,  
                              right_index=True, 
                              )
    df_bvav_diff['bvavRatio'] = df_bvav_diff.Close / df_bvav_diff.AV 
    df_bvav_diff['bvav'] = df_bvav_diff.Close - df_bvav_diff.AV
    df_bvav_ratio = pd.DataFrame(df_bvav_diff['bvavRatio']) 
    df_bvav_diff = pd.DataFrame(df_bvav_diff['bvav']) 
    
    
    
    df_tb_raw = pd.read_csv(data_tx_tb,index_col=0,parse_dates=[['Date', 'Time']]) 
    df_ta_raw = pd.read_csv(data_tx_ta,index_col=0,parse_dates=[['Date', 'Time']]) 
    df_ta_raw.rename(columns={'Close': 'TA'}, inplace=True) 
    df_tbta_diff = pd.merge(pd.DataFrame(df_tb_raw['Close']),
                              pd.DataFrame(df_ta_raw['TA']),
                              left_index=True,  
                              right_index=True, 
                              )
    df_tbta_diff['tbtaRatio'] = df_tbta_diff.Close / df_tbta_diff.TA 
    df_tbta_diff['tbta'] = df_tbta_diff.Close - df_tbta_diff.TA 
    df_tbta_ratio  = pd.DataFrame(df_tbta_diff['tbtaRatio']) 
    df_tbta_diff = pd.DataFrame(df_tbta_diff['tbta']) 
    
    
    
    maxStartDate = df_txf_raw.index.min()
    startDates = [
        df_txf_raw.index.min(),
        df_txfTwse_diff.index.min(),
        df_exfFxf_diff.index.min(),
        df_uvdv_diff.index.min(),
        df_bvav_diff.index.min(),
        df_tbta_diff.index.min()   
    ]
    for _dt_ in startDates:  
        if(_dt_ > maxStartDate ):
            maxStartDate = _dt_
    
    minEndDate = df_txf_raw.index.max()
    endDates = [
        df_txf_raw.index.max(),
        df_txfTwse_diff.index.max(),
        df_exfFxf_diff.index.max(),
        df_uvdv_diff.index.max(),
        df_bvav_diff.index.max(),
        df_tbta_diff.index.max()   
    ]    
    for _dt_ in endDates:  
        if(_dt_ < minEndDate):
            minEndDate = _dt_
    
    
    df_txf_raw = df_txf_raw.join(df_txfTwse_diff, how='outer', rsuffix='_1')  
    df_txf_raw = df_txf_raw.join(df_exfFxf_diff, how='outer', rsuffix='_2') 
    df_txf_raw = df_txf_raw.join(df_uvdv_diff, how='outer', rsuffix='_3')
    df_txf_raw = df_txf_raw.join(df_bvav_diff, how='outer', rsuffix='_4')
    df_txf_raw = df_txf_raw.join(df_tbta_diff, how='outer', rsuffix='_5')
    
    df_txf_raw = df_txf_raw.join(df_txfTwse_ratio, how='outer', rsuffix='_6')
    df_txf_raw = df_txf_raw.join(df_exfFxf_ratio, how='outer', rsuffix='_7') 
    df_txf_raw = df_txf_raw.join(df_uvdv_ratio, how='outer', rsuffix='_8')
    df_txf_raw = df_txf_raw.join(df_bvav_ratio, how='outer', rsuffix='_9')
    df_txf_raw = df_txf_raw.join(df_tbta_ratio, how='outer', rsuffix='_10')
    
    
    df_txf_raw = df_txf_raw[maxStartDate : minEndDate] 
    
    df_txf_raw = df_txf_raw.dropna()
    moving_features, moving_labels, featureSize = chart.extract_feature(
                                                        raw_data = df_txf_raw,
                                                        selector=selector,
                                                        window=input_wind_size,
                                                        with_label=True,
                                                        flatten=True
                                                    )
    
    print("feature extraction done, start writing to file...")
    numDays_for_test = int(days_for_test_perc * moving_features.shape[0])
    numDays_for_validation = int(days_for_validate_perc * moving_features.shape[0])
    
    validationData = moving_features[:numDays_for_validation]
    validationData_1 = numpy.reshape(validationData , [-1, input_wind_size, featureSize ])
    validationLabel =  numpy.asarray(moving_labels[:numDays_for_validation]) 
    
    trainData = moving_features[numDays_for_validation: -numDays_for_test]
    trainData_1 = numpy.reshape(trainData, [-1, input_wind_size, featureSize ]) 
    trainLabel = numpy.asarray(moving_labels[numDays_for_validation: -numDays_for_test])
    
    testData = moving_features[-numDays_for_test:] 
    testData_1 = numpy.reshape(testData , [-1, input_wind_size, featureSize])
    testLabel = numpy.asarray(moving_labels[-numDays_for_test:])
    
    if(toDataSet):
        train_set = DataSet(trainData_1, trainLabel)
        validation_set = DataSet(validationData_1, validationLabel)
        test_set = DataSet(testData_1, testLabel)
        return train_set ,validation_set, test_set , featureSize
    else:
        return trainData_1,trainLabel,validationData_1,validationLabel,testData_1,testLabel,featureSize


def writeFeature(trainingData,trainingLabel,testData,testLabel):
    for i in range(0, train_end_test_begin):
        for item in moving_features[i]:
            fp.write("%s\t" % item)
        fp.write("\n")
    for i in range(0, train_end_test_begin):
        print(moving_labels[i])
        lp.write("%s\n" % moving_labels[i])
    # test set
    for i in range(train_end_test_begin, moving_features.shape[0]):
        for item in moving_features[i]:
            fpt.write("%s\t" % item)
        fpt.write("\n")
    for i in range(train_end_test_begin, moving_features.shape[0]):
        lpt.write("%s\n" % moving_labels[i])

    fp.close()
    lp.close()
    fpt.close()
    lpt.close()    

if __name__ == '__main__':
    trainingData,trainingLabel,testData,testLabel = generateDataSetTXF()
    