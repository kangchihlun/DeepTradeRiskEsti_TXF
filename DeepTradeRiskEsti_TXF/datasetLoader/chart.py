# coding:UTF-8
import numpy 
import talib
import math
from sklearn.preprocessing import scale
from sklearn.preprocessing import minmax_scale


class ChartFeature(object):
    def __init__(self, selector):
        self.selector = selector
        self.supported = {
            "ROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", 
            "VMA", "PRICE_VOLUME",
            "TXFTWSE_DIFF_ROCP","EXFFXF_DIFF_ROCP","UDV_ROCP","BAV_ROCP","TBA_ROCP"
            "TXFTWSE_RATIO_ROCP","EXFFXF_RATIO_ROCP","UDV_RATIO_ROCP","BAV_RATIO_ROCP","TBA_RATIO_ROCP"
            }
        self.feature = []

    def moving_extract(self, window=30, close_prices=None,
                       TXFTWSE_DIFF=None, EXFFXF_DIFF=None, UDV=None,BAV=None,TBA=None,
                       TXFTWSE_RATIO=None, EXFFXF_RATIO=None, UDV_RATIO=None,BAV_RATIO=None,TBA_RATIO=None,
                       volumes=None, with_label=True, flatten=True):
        self._window_ = window
        self.extract(
                        close_prices=close_prices, 
                        TXFTWSE_DIFF=TXFTWSE_DIFF,
                        EXFFXF_DIFF=EXFFXF_DIFF,
                        UDV=UDV,
                        BAV=BAV,
                        TBA=TBA,
                        TXFTWSE_RATIO=TXFTWSE_RATIO,
                        EXFFXF_RATIO=EXFFXF_RATIO,
                        UDV_RATIO=UDV_RATIO,
                        BAV_RATIO=BAV_RATIO,
                        TBA_RATIO=TBA_RATIO,
                        volumes=volumes
                    )
        feature_arr = numpy.asarray(self.feature)
        p = 0
        rows = feature_arr.shape[0]
        print("feature dimension: %s" % rows) 
        
        moving_features = []
        moving_labels = []
        dataLen = feature_arr.shape[1]
        while p + window < dataLen : 
            x = feature_arr[:, p:p + window] 
            
            tgt_idx = min( [p + window * 2 , dataLen-1]) 
            fRngArr = close_prices[p + window : tgt_idx]
            if(len(fRngArr)):
                fRngMax = max(fRngArr)
                fRngMin = min(fRngArr)
                curClose_idx = max([ p + window -1 , 0])
                curClose = close_prices[curClose_idx]
                if(curClose > 0.01):
                    difUp = fRngMax - curClose
                    difDw = fRngMin - curClose
                    tgtFutureClose = fRngMax if abs(difUp)>abs(difDw) else fRngMin
                    p_change = (tgtFutureClose - curClose) / curClose * 100.0 
                    p_change = max(-1, min(p_change , 1))
                    
                    
                    y = p_change
                    if flatten:
                        x = x.flatten("F")
                    moving_features.append(numpy.nan_to_num(x))
                    moving_labels.append(y)
            
            p += 1
        
        return numpy.asarray(moving_features), moving_labels , rows  

    def extract(self, close_prices=None,
                TXFTWSE_DIFF=None,EXFFXF_DIFF=None,UDV=None,BAV=None,TBA=None,
                TXFTWSE_RATIO=None,EXFFXF_RATIO=None,UDV_RATIO=None,BAV_RATIO=None,TBA_RATIO=None,
                volumes=None):
        self.feature = []
        for feature_type in self.selector: 
            if feature_type in self.supported:
                print("extracting feature : %s" % feature_type)
                self.extract_by_type(feature_type, close_prices,
                                     TXFTWSE_DIFF,EXFFXF_DIFF,UDV,BAV,TBA,
                                     TXFTWSE_RATIO,EXFFXF_RATIO,UDV_RATIO,BAV_RATIO,TBA_RATIO,
                                     volumes)
            else:
                print("feature type not supported: %s" % feature_type)
        return self.feature
    
    def normalise_windows(window_data):
        normalised_data = []
        for window in window_data:
            normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
            normalised_data.append(normalised_window)
        return normalised_data    
    
    def extract_by_type(self, feature_type, close_prices=None,
                        TXFTWSE_DIFF=None,EXFFXF_DIFF=None,UDV=None,BAV=None,TBA=None,
                        TXFTWSE_RATIO=None,EXFFXF_RATIO=None,UDV_RATIO=None,BAV_RATIO=None,TBA_RATIO=None,
                        volumes=None):
        if feature_type == 'ROCP':
            rocp = numpy.nan_to_num(talib.ROCP(close_prices, timeperiod=1))
            rocp_n = minmax_scale(rocp,feature_range=(-1, 1))
            self.feature.append(rocp) # for ad in rocp : print(ad)
            
        if feature_type == 'OROCP':
            orocp = talib.ROCP(open_prices, timeperiod=1)
            self.feature.append(orocp) 
        if feature_type == 'HROCP':
            hrocp = talib.ROCP(high_prices, timeperiod=1)
            self.feature.append(hrocp)
        if feature_type == 'LROCP':
            lrocp = talib.ROCP(low_prices, timeperiod=1)
            self.feature.append(lrocp)
        if feature_type == 'MACD':
            macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            macd = numpy.nan_to_num(macd) 
            norm_macd = minmax_scale(macd ,feature_range=(-1, 1))
            self.feature.append(norm_macd)
            
            signal = numpy.nan_to_num(signal)
            norm_signal = minmax_scale(signal ,feature_range=(-1, 1))
            self.feature.append(norm_signal)
            
            hist = numpy.nan_to_num(hist)
            norm_hist = minmax_scale(hist ,feature_range=(-1, 1))
            self.feature.append(norm_hist)

        if feature_type == 'RSI':
            rsi6 = talib.RSI(close_prices, timeperiod=6)
            rsi12 = talib.RSI(close_prices, timeperiod=12)
            rsi24 = talib.RSI(close_prices, timeperiod=24)
            rsi6rocp = talib.ROCP(rsi6 + 100., timeperiod=1)
            rsi12rocp = talib.ROCP(rsi12 + 100., timeperiod=1)
            rsi24rocp = talib.ROCP(rsi24 + 100., timeperiod=1)
            
            rsi6 = numpy.nan_to_num(rsi6)
            rsi6 = rsi6 / 100.0 - 0.5
            norm_rsi6 = minmax_scale(rsi6 ,feature_range=(-1, 1))
            self.feature.append(norm_rsi6)
            
            rsi12 = numpy.nan_to_num(rsi12)
            rsi12 = rsi12 / 100.0 - 0.5
            norm_rsi12 = minmax_scale(rsi12 ,feature_range=(-1, 1))
            self.feature.append(norm_rsi12)
            
            rsi24 = numpy.nan_to_num(rsi24)
            rsi24 = rsi24 / 100.0 - 0.5
            norm_rsi24 = minmax_scale(rsi24 ,feature_range=(-1, 1))
            self.feature.append(norm_rsi24)
            
        if feature_type == 'VROCP':
            norm_volumes = minmax_scale(volumes ,feature_range=(-1, 1))
            self.feature.append(norm_volumes)
            
        if feature_type == 'BOLL':
            upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=self._window_ , nbdevup=2, nbdevdn=2, matype=0)
            upperband = (upperband - close_prices) / close_prices
            upperband = numpy.nan_to_num(upperband)
            norm_upperband = minmax_scale(upperband ,feature_range=(-1, 1))
            self.feature.append(norm_upperband)
            
            middleband = (middleband - close_prices) / close_prices
            middleband = numpy.nan_to_num(middleband)
            norm_middleband = minmax_scale(middleband ,feature_range=(-1, 1))
            self.feature.append(norm_middleband)
            
            lowerband = (lowerband - close_prices) / close_prices
            lowerband = numpy.nan_to_num(lowerband)
            norm_lowerband = minmax_scale(lowerband,feature_range=(-1, 1))
            self.feature.append(norm_lowerband)
            
        if feature_type == 'MA':
            
            ma5 = talib.MA(close_prices, timeperiod=5)
            ma5_clo = (ma5 - close_prices) / close_prices
            ma5_clo = numpy.nan_to_num(ma5_clo)
            norm_ma5_clo = minmax_scale(ma5_clo,feature_range=(-1, 1))
            self.feature.append(norm_ma5_clo)
            
            ma10 = talib.MA(close_prices, timeperiod=10)
            ma10_clo = (ma10 - close_prices) / close_prices
            ma10_clo = numpy.nan_to_num(ma10_clo)
            norm_ma10_clo = minmax_scale(ma10_clo,feature_range=(-1, 1))
            self.feature.append(norm_ma10_clo)
            
            ma20 = talib.MA(close_prices, timeperiod=20)
            ma20_clo = (ma20 - close_prices) / close_prices
            ma20_clo = numpy.nan_to_num(ma20_clo)
            norm_ma20_clo = minmax_scale(ma20_clo,feature_range=(-1, 1))
            self.feature.append(norm_ma20_clo)
            
            ma30 = talib.MA(close_prices, timeperiod=30)
            ma30_clo = (ma30 - close_prices) / close_prices
            ma30_clo = numpy.nan_to_num(ma30_clo)
            norm_ma30_clo = minmax_scale(ma30_clo,feature_range=(-1, 1))
            self.feature.append(norm_ma30_clo)
            
            ma60 = talib.MA(close_prices, timeperiod=60)
            ma60_clo = (ma60 - close_prices) / close_prices
            ma60_clo = numpy.nan_to_num(ma60_clo)
            norm_ma60_clo = minmax_scale(ma60_clo,feature_range=(-1, 1))
            self.feature.append(norm_ma60_clo)
            
        if feature_type == 'VMA':
            ma5 = talib.MA(volumes, timeperiod=5) 
            ma5_clo = ((ma5 - volumes) / (volumes + 1))
            ma5_clo = numpy.nan_to_num(ma5_clo)
            norm_ma5_clo = minmax_scale(ma5_clo,feature_range=(-1, 1))
            self.feature.append(norm_ma5_clo)
            
            
            ma10 = talib.MA(volumes, timeperiod=10)
            ma10_clo = ((ma5 - volumes) / (volumes + 1))
            ma10_clo = numpy.nan_to_num(ma10_clo)
            norm_ma10_clo = minmax_scale(ma10_clo ,feature_range=(-1, 1))
            self.feature.append(norm_ma10_clo)
            
            
            ma20 = talib.MA(volumes, timeperiod=20)
            ma20_clo = ((ma5 - volumes) / (volumes + 1))
            ma20_clo = numpy.nan_to_num(ma20_clo)
            norm_ma20_clo = minmax_scale(ma20_clo ,feature_range=(-1, 1))
            self.feature.append(norm_ma20_clo)            
            
            
        if feature_type == 'PRICE_VOLUME':
            rocp = talib.ROCP(close_prices, timeperiod=1)
            norm_volumes = (volumes - numpy.mean(volumes)) / math.sqrt(numpy.var(volumes))
            vrocp = talib.ROCP(norm_volumes + numpy.max(norm_volumes) - numpy.min(norm_volumes), timeperiod=1)
            pv = rocp * vrocp * 100
            pv = numpy.nan_to_num(pv)
            norm_pv = minmax_scale(pv ,feature_range=(-1, 1))
            self.feature.append(norm_pv)
            
        if feature_type == 'TXFTWSE_DIFF_ROCP':
            norm_volumes = minmax_scale(TXFTWSE_DIFF ,feature_range=(-1, 1))
            self.feature.append(norm_volumes) 
            
        if feature_type == 'TXFTWSE_RATIO_ROCP':
            norm_volumesr = minmax_scale(TXFTWSE_RATIO ,feature_range=(-1, 1))
            self.feature.append(norm_volumesr) # for ad in norm_volumesr :print(ad)
            
        if feature_type == 'EXFFXF_DIFF_ROCP':
            norm_volumes = minmax_scale(EXFFXF_DIFF,feature_range=(-1, 1))
            self.feature.append(norm_volumes) 
            
        if feature_type == 'EXFFXF_RATIO_ROCP':
            norm_volumes = minmax_scale(EXFFXF_RATIO ,feature_range=(-1, 1))
            self.feature.append(norm_volumes) # for ad in norm_volumes:print(ad)
            
        if feature_type == 'UDV_ROCP':
            UDV = numpy.nan_to_num(UDV)
            norm_volumes = minmax_scale(UDV,feature_range=(-1, 1))
            self.feature.append(norm_volumes) # for ad in norm_volumes:print(ad)
            
        if feature_type == 'UDV_RATIO_ROCP':
            norm_volumes = minmax_scale(UDV_RATIO ,feature_range=(-1, 1))
            self.feature.append(norm_volumes)
        
            ma5 = talib.MA(UDV_RATIO, timeperiod=5)
            ma5_clo = (ma5 - UDV_RATIO) / UDV_RATIO
            ma5_clo = numpy.nan_to_num(ma5_clo)
            norm_ma5_clo = minmax_scale(ma5_clo ,feature_range=(-1, 1))
            self.feature.append(norm_ma5_clo)# for ad in norm_ma5_clo:print(ad)
        
            ma10 = talib.MA(UDV_RATIO, timeperiod=10)
            ma10_clo = (ma10 - UDV_RATIO) / UDV_RATIO
            ma10_clo = numpy.nan_to_num(ma10_clo)
            norm_ma10_clo = minmax_scale(ma10_clo ,feature_range=(-1, 1))
            self.feature.append(norm_ma10_clo)# for ad in norm_ma10_clo:print(ad)
        
            ma20 = talib.MA(UDV_RATIO, timeperiod=20)
            ma20_clo = (ma20 - UDV_RATIO) / UDV_RATIO
            ma20_clo = numpy.nan_to_num(ma20_clo)
            norm_ma20_clo = minmax_scale(ma20_clo ,feature_range=(-1, 1))# for ad in norm_ma20_clo: print(ad)
            self.feature.append(norm_ma20_clo)

        if feature_type == 'BAV_ROCP':
            BAV = numpy.nan_to_num(BAV)
            norm_volumes = minmax_scale(BAV,feature_range=(-1, 1))
            self.feature.append(norm_volumes)
                    
        if feature_type == 'BAV_RATIO_ROCP':
            BAV_RATIO = numpy.nan_to_num(BAV_RATIO)
            norm_volumes = minmax_scale(BAV_RATIO ,feature_range=(-1, 1))
            self.feature.append(norm_volumes)
            
            ma5 = talib.MA(BAV_RATIO, timeperiod=5)
            ma5_clo = (ma5 - BAV_RATIO) / BAV_RATIO
            ma5_clo = numpy.nan_to_num(ma5_clo)
            norm_ma5_clo = minmax_scale(ma5_clo,feature_range=(-1, 1))
            self.feature.append(norm_ma5_clo)
        
            ma10 = talib.MA(BAV_RATIO, timeperiod=10)
            ma10_clo = (ma10 - BAV_RATIO) / BAV_RATIO
            ma10_clo = numpy.nan_to_num(ma10_clo)
            norm_ma10_clo = minmax_scale(ma10_clo,feature_range=(-1, 1))
            self.feature.append(norm_ma10_clo)
        
            ma20 = talib.MA(BAV_RATIO, timeperiod=20)
            ma20_clo = (ma20 - BAV_RATIO) / BAV_RATIO
            ma20_clo = numpy.nan_to_num(ma20_clo)
            norm_ma20_clo = minmax_scale(ma20_clo,feature_range=(-1, 1))
            self.feature.append(norm_ma20_clo)
            
        if feature_type == 'TBA_ROCP':
            norm_volumes = minmax_scale(TBA,feature_range=(-1, 1))
            self.feature.append(norm_volumes)
            
        if feature_type == 'TBA_RATIO_ROCP':
            norm_volumes = minmax_scale(TBA_RATIO,feature_range=(-1, 1))
            self.feature.append(norm_volumes)        
        
            ma5 = talib.MA(TBA_RATIO , timeperiod=5)
            ma5_clo = (ma5 - TBA_RATIO ) / TBA_RATIO 
            ma5_clo = numpy.nan_to_num(ma5_clo)
            norm_ma5_clo = minmax_scale(ma5_clo ,feature_range=(-1, 1))# for ad in norm_ma5_clo: print(ad)
            self.feature.append(norm_ma5_clo)
        
            ma10 = talib.MA(TBA_RATIO , timeperiod=10)
            ma10_clo = (ma10 - TBA_RATIO ) / TBA_RATIO 
            ma10_clo = numpy.nan_to_num(ma10_clo)
            norm_ma10_clo = minmax_scale(ma10_clo ,feature_range=(-1, 1))# for ad in norm_ma10_clo: print(ad)
            self.feature.append(norm_ma10_clo)
        
            ma20 = talib.MA(TBA_RATIO , timeperiod=20)
            ma20_clo = (ma20 - TBA_RATIO ) / TBA_RATIO 
            ma20_clo = numpy.nan_to_num(ma20_clo)
            norm_ma20_clo = minmax_scale(ma20_clo ,feature_range=(-1, 1))# for ad in norm_ma10_clo: print(ad)
            self.feature.append(norm_ma20_clo)        
            

def extract_feature(raw_data,selector, window=30, with_label=True, flatten=True):
    chart_feature = ChartFeature(selector)
    closes = raw_data.Close.values # len(closes)
    volumes = raw_data.TotalVolume.values # len(volumes)
    txftwse_dif = raw_data.txfTwseDiff.values # len(txftwse_dif)
    exffxf_diff = raw_data.exfFxfDiff.values # len(exffxf_diff)
    udv=raw_data.uvdv.values # len(udv)
    bav=raw_data.bvav.values # len(bav)
    tba=raw_data.tbta.values # len(tba)
    
    txftwse_ratio = raw_data.txfTwseRatio.values # len(txftwse_dif)
    exffxf_ratio = raw_data.exfFxfRatio.values # len(exffxf_diff)
    udv_ratio=raw_data.uvdv_ratio.values # len(udv)
    bav_ratio=raw_data.bvavRatio.values # len(bav)
    tba_ratio=raw_data.tbtaRatio.values # len(tba)    
    
    if with_label:
        moving_features,moving_labels,numRows = chart_feature.moving_extract(
                                                                        window=window,
                                                                        close_prices=closes,
                                                                        TXFTWSE_DIFF=txftwse_dif,
                                                                        EXFFXF_DIFF=exffxf_diff,
                                                                        UDV=udv,
                                                                        BAV=bav,
                                                                        TBA=tba,
                                                                        TXFTWSE_RATIO=txftwse_ratio,
                                                                        EXFFXF_RATIO=exffxf_ratio,
                                                                        UDV_RATIO=udv_ratio,
                                                                        BAV_RATIO=bav_ratio,
                                                                        TBA_RATIO=tba_ratio,                                                                        
                                                                        volumes=volumes,
                                                                        with_label=with_label,
                                                                        flatten=flatten
                                                                     )
        return moving_features, moving_labels,numRows 




