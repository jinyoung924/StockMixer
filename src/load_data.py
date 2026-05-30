import numpy as np
import os
from tqdm import tqdm

# 원시 csv에서 .pkl을 만들때 사용하는 코드 - 지금은 있는 .pkl 데이터 사용한다.

'''
이 함수를 거쳐 만들어지는 배열 4개

1. eod_data — 입력 피처. 형태는 (종목수 1026, 거래일수 1245, 피처수 5)
    0 : 5일 이동평균 (정규화)
    1 : 10일 이동평균 (정규화)
    2 : 20일 이동평균 (정규화)
    3 : 30일 이동평균 (정규화)
    4 : 당일 정규화 종가 (close)
2. gt_data — 정답. 각 종목·각 날짜의 실제 수익률(다음 날 종가 변화율)
3. price_data — 기준가(예측 시점의 종가)
4. mask_data — 유효 여부. 결측이면 0, 정상이면 1 : 직사각형 매트릭스에서 의미있는 값이면 1로 되어있다. 
'''

def load_EOD_data(data_path, market_name, tickers, steps=1):
    eod_data = []
    masks = []
    ground_truth = []
    base_price = []
    for index, ticker in enumerate(tqdm(tickers)):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ':
            # remove the last day since lots of missing data
            single_EOD = single_EOD[:-1, :]
        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0],
                                 single_EOD.shape[1] - 1], dtype=np.float32)
            masks = np.ones([len(tickers), single_EOD.shape[0]],
                            dtype=np.float32)
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]],
                                    dtype=np.float32)
            base_price = np.zeros([len(tickers), single_EOD.shape[0]],
                                  dtype=np.float32)
        for row in range(single_EOD.shape[0]):
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                masks[index][row] = 0.0
            elif row > steps - 1 and abs(single_EOD[row - steps][-1] + 1234) \
                    > 1e-8:
                ground_truth[index][row] = \
                    (single_EOD[row][-1] - single_EOD[row - steps][-1]) / \
                    single_EOD[row - steps][-1]
            for col in range(single_EOD.shape[1]):
                if abs(single_EOD[row][col] + 1234) < 1e-8:
                    single_EOD[row][col] = 1.1
        eod_data[index, :, :] = single_EOD[:, 1:]
        base_price[index, :] = single_EOD[:, -1]
    return eod_data, masks, ground_truth, base_price


def load_graph_relation_data(relation_file, lap=False):
    relation_encoding = np.load(relation_file)
    print('relation encoding shape:', relation_encoding.shape)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    ajacent = np.where(mask_flags, np.zeros(rel_shape, dtype=float),
                       np.ones(rel_shape, dtype=float))
    degree = np.sum(ajacent, axis=0)
    for i in range(len(degree)):
        degree[i] = 1.0 / degree[i]
    np.sqrt(degree, degree)
    deg_neg_half_power = np.diag(degree)
    if lap:
        return np.identity(ajacent.shape[0], dtype=float) - np.dot(
            np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)
    else:
        return np.dot(np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)


def load_relation_data(relation_file):
    relation_encoding = np.load(relation_file)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))
    return relation_encoding, mask


def build_SFM_data(data_path, market_name, tickers):
    eod_data = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0]],
                                dtype=np.float32)

        for row in range(single_EOD.shape[0]):
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                if row < 3:
                    for i in range(row + 1, single_EOD.shape[0]):
                        if abs(single_EOD[i][-1] + 1234) > 1e-8:
                            eod_data[index][row] = single_EOD[i][-1]
                            break
                else:
                    eod_data[index][row] = np.sum(
                        eod_data[index, row - 3:row]) / 3
            else:
                eod_data[index][row] = single_EOD[row][-1]
    np.save(market_name + '_sfm_data', eod_data)