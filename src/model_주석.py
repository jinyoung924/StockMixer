"""
================================================================================
 StockMixer (AAAI 2024) - model.py 상세 주석본
================================================================================

[전체 구조 한눈에 보기]
논문 StockMixer는 "indicator mixing → time mixing → stock mixing" 순서로
주가 데이터의 세 가지 상관관계를 모델링하는 순수 MLP 기반 아키텍처다.

  입력  : X ∈ (N, T, F)   N=종목 수(1026), T=lookback(16), F=지표 수(5)
  출력  : p ∈ (N, 1)      각 종목의 다음날 종가 예측 → 이걸로 수익률(return ratio) 계산

코드 안의 클래스는 크게 두 갈래로 나뉜다.
  (1) 실제 StockMixer가 사용하는 핵심 모듈
        - TriU            : 인과적(causal) time mixing의 핵심. 상삼각 마스크 역할
        - MixerBlock      : 표준 MLP mixing 블록 (indicator/channel mixing에 사용)
        - Mixer2dTriU     : indicator mixing + (TriU)time mixing 묶음
        - MultTime2dMixer : 원본 스케일 + 다운샘플 스케일을 합치는 multi-scale mixer
        - NoGraphMixer    : stock mixing (stock→market→stock 병목 구조)
        - StockMixer      : 위 모듈들을 조립한 최종 모델
  (2) 논문/실험 과정에서 쓰였지만 최종 forward 경로에는 안 들어가는 보조 모듈
        - Mixer2d, TimeMixerBlock, MultiScaleTimeMixer
      (표준 MLP mixing이나 다른 다중스케일 구현을 비교/실험하기 위한 잔재로 보면 됨)

아래에서 각 부분을 위→아래 순으로 따라가며 설명한다.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 전역 활성화 함수. 논문에서는 GELU/ReLU/HardSwish를 비교했고,
# 여기서는 GELU를 기본으로 두고 MixerBlock 등에서 공통으로 가져다 쓴다.
acv = nn.GELU()


# ==============================================================================
# 손실 함수: 회귀(MSE) 손실 + 순위(ranking) 손실
# ==============================================================================
def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    """
    논문 식 (9)의 L = L_MSE + alpha * (pairwise ranking loss) 를 구현.

    인자(shape는 한 거래일에 대한 전 종목 배치 기준):
      prediction   : (N, 1)  모델이 예측한 '다음날 종가'
      ground_truth : (N, 1)  실제 1일 수익률(return ratio)  ← 정답
      base_price   : (N, 1)  현재(마지막) 거래일 종가. 예측가를 수익률로 환산할 때 분모
      mask         : (N, 1)  유효 종목 표시(1=유효, 0=결측/상장폐지 등)
      batch_size   : N (종목 수). all_one 벡터 길이로 사용
      alpha        : ranking loss 가중치 (train.py에서 0.1)
    """
    device = prediction.device

    # (N, 1) 모양의 전부 1 벡터. 뒤에서 "브로드캐스트용 행렬"을 만들 때 쓴다.
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)

    # 예측 종가를 1일 수익률로 환산: (예측가 - 기준가) / 기준가
    #   return_ratio = (prediction - base_price) / base_price   → (N, 1)
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)

    # ---- (1) 회귀 손실: 예측 수익률과 실제 수익률의 MSE (mask로 유효 종목만 반영) ----
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)

    # ---- (2) 순위 손실: 종목 간 '상대적 순서'를 맞추도록 유도하는 pairwise hinge loss ----
    # 예측 수익률의 모든 쌍 차이 행렬을 만든다.
    #   return_ratio @ all_one.t()  → (N,N), 원소[i,j] = return_ratio[i] (행마다 상수)
    #   all_one @ return_ratio.t()  → (N,N), 원소[i,j] = return_ratio[j] (열마다 상수)
    #   따라서 pre_pw_dif[i,j] = return_ratio[i] - return_ratio[j]
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),
        all_one @ return_ratio.t()
    )

    # 정답 수익률의 쌍 차이 행렬 (부호가 예측쪽과 반대로 정의됨에 주의)
    #   gt_pw_dif[i,j] = ground_truth[j] - ground_truth[i]
    gt_pw_dif = torch.sub(
        all_one @ ground_truth.t(),
        ground_truth @ all_one.t()
    )

    # 두 종목 모두 유효할 때만 1이 되는 쌍 마스크: mask_pw[i,j] = mask[i]*mask[j]
    mask_pw = mask @ mask.t()

    # pre_pw_dif * gt_pw_dif = (r_i - r_j) * (gt_j - gt_i) = -(r_i - r_j)(gt_i - gt_j)
    # 예측 순서와 정답 순서가 '일치'하면 음수 → relu로 0이 되어 벌점 없음.
    # '불일치'하면 양수 → 그대로 벌점으로 부과. 즉 순위가 어긋난 쌍만 페널티.
    rank_loss = torch.mean(
        F.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )

    # 최종 손실 = 회귀 + alpha * 순위
    loss = reg_loss + alpha * rank_loss
    # return_ratio도 함께 반환 → 평가(evaluate) 단계에서 예측 수익률로 사용
    return loss, reg_loss, rank_loss, return_ratio


# ==============================================================================
# MixerBlock: 가장 기본적인 MLP mixing 블록 (Linear → 활성화 → Linear)
# ==============================================================================
class MixerBlock(nn.Module):
    """
    표준 MLP-Mixer의 한 블록. 어떤 한 축(mlp_dim)에 대해
    'Linear → 활성화 → (dropout) → Linear → (dropout)'를 적용한다.
    StockMixer에서는 channel(=indicator) 축을 섞는 channel mixing에 사용된다.
      mlp_dim    : 입력/출력 차원 (예: 채널 수 F)
      hidden_dim : 은닉 차원 (논문 관례상 보통 mlp_dim과 같게 둠)
    """
    def __init__(self, mlp_dim, hidden_dim, dropout=0.0):
        super(MixerBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)  # 차원 확장/투영
        self.LN = acv                                   # 활성화(GELU). 이름은 LN이지만 실제로는 활성화 함수
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)   # 원래 차원으로 복원

    def forward(self, x):
        x = self.dense_1(x)
        x = self.LN(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        x = self.dense_2(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        return x


# ==============================================================================
# Mixer2d: 표준 2D MLP-Mixer 블록  *** 최종 StockMixer에서는 미사용(비교용) ***
# ==============================================================================
class Mixer2d(nn.Module):
    """
    표준 MLP-Mixer 스타일의 2D 블록.
    time 축을 일반 Linear(MixerBlock)로 '전부 동등하게' 섞는다.
    → 논문이 지적한 한계(시간의 인과성 무시)를 가진 '표준' 버전.
    StockMixer는 이 대신 시간축을 TriU로 처리하는 Mixer2dTriU를 쓴다.
    여기 정의된 Mixer2d는 표준 mixing과의 비교 실험용 잔재로 보면 된다.
    """
    def __init__(self, time_steps, channels):
        super(Mixer2d, self).__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = MixerBlock(time_steps, time_steps)     # 시간축을 일반 Linear로 섞음(비인과적)
        self.channelMixer = MixerBlock(channels, channels)      # 채널(지표)축 섞음

    def forward(self, inputs):
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)      # (B, T, C) → (B, C, T) : 시간축을 마지막으로
        x = self.timeMixer(x)       # 시간 mixing
        x = x.permute(0, 2, 1)      # 원위치 (B, T, C)

        x = self.LN_2(x + inputs)   # 잔차 연결 후 정규화
        y = self.channelMixer(x)    # 채널 mixing
        return x + y                # 잔차 연결


# ==============================================================================
# TriU: 상삼각(Upper-Triangular) 마스킹을 구현하는 인과적 time mixing의 핵심
# ==============================================================================
class TriU(nn.Module):
    """
    논문 Figure 2의 핵심 아이디어. time mixing에서 "각 시점 t는 자기 자신과
    그 이전 시점만 참조"하도록 만드는 모듈(미래 정보 누설 차단 = 인과성 보장).

    구현 트릭:
      시점 i(0부터 시작)의 출력을 만들 때 입력의 0..i 시점만 받는
      Linear(i+1 → 1)을 따로 둔다. 즉 시점마다 입력 길이가 다른 작은 선형층을
      time_step개 만들어 두고, 각 출력 시점이 '그 시점까지의 입력'만 보게 한다.
      이렇게 시점별로 가변 길이를 쓰는 것이 상삼각 가중치 행렬과 동일한 효과.

    입력 shape: (B, C, T)  ※ 마지막 축 T가 시간. (호출 측에서 permute로 맞춰줌)
      inputs[:, :, 0:i+1] = 0..i번째 시점까지의 부분 시퀀스
    """
    def __init__(self, time_step):
        super(TriU, self).__init__()
        self.time_step = time_step
        # 시점 i용 선형층: 입력 길이 (i+1) → 출력 1.  i=0..T-1
        self.triU = nn.ParameterList(
            [
                nn.Linear(i + 1, 1)
                for i in range(time_step)
            ]
        )

    def forward(self, inputs):
        # 첫 시점(0): 입력의 0번째 시점 1개만 사용 → (B, C, 1)
        x = self.triU[0](inputs[:, :, 0].unsqueeze(-1))
        # 이후 시점 i: 0..i 시점을 입력으로 받아 1개 출력 → 누적 concat
        for i in range(1, self.time_step):
            x = torch.cat([x, self.triU[i](inputs[:, :, 0:i + 1])], dim=-1)
        # 결과 shape: (B, C, T)  (시간축 길이는 유지되지만 각 출력은 인과적으로 계산됨)
        return x


# ==============================================================================
# TimeMixerBlock: TriU 두 개를 쌓은 time mixing 블록 *** 최종 모델 미사용(보조) ***
# ==============================================================================
class TimeMixerBlock(nn.Module):
    """
    인과적 time mixing을 'TriU → 활성화 → TriU' 형태로 구성한 블록.
    개념적으로 MixerBlock의 시간축 인과 버전. 최종 StockMixer forward 경로에는
    들어가지 않으며, 다중스케일 mixer(MultiScaleTimeMixer)에서 쓰이는 부품/실험용.
    """
    def __init__(self, time_step):
        super(TimeMixerBlock, self).__init__()
        self.time_step = time_step
        self.dense_1 = TriU(time_step)
        self.LN = acv
        self.dense_2 = TriU(time_step)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.LN(x)
        x = self.dense_2(x)
        return x


# ==============================================================================
# MultiScaleTimeMixer: conv로 시퀀스를 여러 스케일로 줄여 mixing
#                      *** 최종 StockMixer는 이 클래스 대신 직접 conv를 씀(보조) ***
# ==============================================================================
class MultiScaleTimeMixer(nn.Module):
    """
    multi-scale time mixing을 한 모듈로 묶어 본 버전.
    scale i마다 Conv1d(kernel=stride=2**i)로 시퀀스를 1/2^i로 다운샘플한 뒤
    TriU로 인과적 mixing을 하고, 각 스케일 결과를 concat한다.
    단, scale 0(전체 길이)에서는 conv 대신 LayerNorm + TriU를 쓰도록 덮어씀.

    최종 StockMixer는 이 통합 클래스를 쓰지 않고, forward 안에서 conv를 직접 호출해
    '원본 스케일'과 '1/2 스케일' 두 갈래만 만든다. 그래도 multi-scale 아이디어 자체를
    이해하는 데 도움이 되므로 남겨둔 비교/실험용 구현으로 보면 된다.
    """
    def __init__(self, time_step, channel, scale_count=1):
        super(MultiScaleTimeMixer, self).__init__()
        self.time_step = time_step
        self.scale_count = scale_count
        # 스케일별 처리 블록 리스트
        self.mix_layer = nn.ParameterList([nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel,
                      kernel_size=2 ** i, stride=2 ** i),          # 1/2^i 다운샘플
            TriU(int(time_step / 2 ** i)),
            nn.Hardswish(),
            TriU(int(time_step / 2 ** i))
        ) for i in range(scale_count)])
        # scale 0은 다운샘플 없이 원본 길이 그대로 처리하도록 교체
        self.mix_layer[0] = nn.Sequential(
            nn.LayerNorm([time_step, channel]),
            TriU(int(time_step)),
            nn.Hardswish(),
            TriU(int(time_step))
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)            # (B, T, C) → (B, C, T)
        y = self.mix_layer[0](x)          # 원본 스케일 결과
        for i in range(1, self.scale_count):
            y = torch.cat((y, self.mix_layer[i](x)), dim=-1)  # 다른 스케일 결과를 시간축에 concat
        return y


# ==============================================================================
# Mixer2dTriU: StockMixer가 실제로 쓰는 '인코더 한 블록'
#              = indicator(channel) mixing + 인과적 time mixing(TriU)
# ==============================================================================
class Mixer2dTriU(nn.Module):
    """
    Mixer2d의 인과 버전. 시간축을 TriU로 처리한다는 점만 다르다.
    동작:
      1) LayerNorm
      2) 시간축으로 옮겨 TriU로 인과적 time mixing
      3) 잔차 + LayerNorm
      4) channel(지표) mixing
      5) 잔차
    입력/출력 shape: (B, time_steps, channels) 동일.
    """
    def __init__(self, time_steps, channels):
        super(Mixer2dTriU, self).__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = TriU(time_steps)                  # 인과적 시간 mixing
        self.channelMixer = MixerBlock(channels, channels) # 지표(채널) mixing = indicator mixing

    def forward(self, inputs):
        x = self.LN_1(inputs)       # (B, T, C)
        x = x.permute(0, 2, 1)      # (B, C, T) : TriU가 마지막 축을 시간으로 보므로 transpose
        x = self.timeMixer(x)       # 인과적 time mixing
        x = x.permute(0, 2, 1)      # (B, T, C) 복귀

        x = self.LN_2(x + inputs)   # 잔차 연결 후 정규화
        y = self.channelMixer(x)    # 채널(지표) mixing
        return x + y                # 잔차 연결 → (B, T, C)


# ==============================================================================
# MultTime2dMixer: multi-scale을 합치는 mixer (원본 스케일 + 다운샘플 스케일)
# ==============================================================================
class MultTime2dMixer(nn.Module):
    """
    두 개의 시간 스케일을 각각 Mixer2dTriU로 인코딩한 뒤, '원본 입력까지' 셋을
    시간축(dim=1)으로 이어 붙인다.

      mix_layer       : 원본 길이 T(=16) 처리
      scale_mix_layer : 다운샘플 길이 scale_dim(=8) 처리

    forward(inputs, y):
      inputs : (B, T, C)        원본 시퀀스
      y      : (B, scale_dim, C) conv로 1/2 다운샘플된 시퀀스 (StockMixer에서 만들어 전달)

    출력 시간축 길이 = T(원본 입력) + T(mix 결과) + scale_dim(다운샘플 mix 결과)
                     = 16 + 16 + 8 = 40  (= time_steps*2 + scale_dim)
    """
    def __init__(self, time_step, channel, scale_dim=8):
        super(MultTime2dMixer, self).__init__()
        self.mix_layer = Mixer2dTriU(time_step, channel)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channel)

    def forward(self, inputs, y):
        y = self.scale_mix_layer(y)   # 다운샘플 스케일 인코딩 (B, 8, C)
        x = self.mix_layer(inputs)    # 원본 스케일 인코딩      (B, 16, C)
        # 원본 입력 + 원본 인코딩 + 다운샘플 인코딩을 시간축으로 concat → (B, 40, C)
        return torch.cat([inputs, x, y], dim=1)


# ==============================================================================
# NoGraphMixer: stock mixing (그래프 없이 stock→market→stock 병목으로 종목 관계 학습)
# ==============================================================================
class NoGraphMixer(nn.Module):
    """
    논문 식 (8)의 stock mixing. 종목끼리 직접(N×N) 섞는 대신,
    'N개 종목 → m개 시장(market) 상태로 압축 → 다시 N개 종목으로 복원'하는
    병목(bottleneck) 구조를 둔다. (하이퍼그래프/시장상태 학습과 유사)
    → 과적합을 줄이고 더 견고한 종목 상관관계를 모델링.

      stocks     : 종목 수 N
      hidden_dim : 시장 차원 m (NASDAQ에서 20)

    입력 shape: (N, d)   d = 종목별 시간/지표가 정리된 특징 길이 (StockMixer에서 40)
    """
    def __init__(self, stocks, hidden_dim=20):
        super(NoGraphMixer, self).__init__()
        self.dense1 = nn.Linear(stocks, hidden_dim)     # N → m  (stock-to-market: 종목들을 시장상태로 집약)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, stocks)     # m → N  (market-to-stock: 시장상태를 다시 각 종목으로)
        self.layer_norm_stock = nn.LayerNorm(stocks)

    def forward(self, inputs):
        x = inputs
        x = x.permute(1, 0)             # (N, d) → (d, N) : 마지막 축을 '종목'으로
        x = self.layer_norm_stock(x)    # 종목 축에 대해 정규화
        x = self.dense1(x)              # N → m : 종목들을 m개 시장상태로 압축
        x = self.activation(x)
        x = self.dense2(x)              # m → N : 시장상태가 각 종목에 끼치는 영향 복원
        x = x.permute(1, 0)             # (d, N) → (N, d) 복귀
        return x


# ==============================================================================
# StockMixer: 최종 모델 — 위 모듈들을 조립
# ==============================================================================
class StockMixer(nn.Module):
    """
    전체 파이프라인:
      입력 (N, T, F)
        │  conv1d로 시간축 1/2 다운샘플 → (N, T/2, F)  (multi-scale의 두 번째 스케일)
        ▼
      MultTime2dMixer(원본 + 다운샘플) → (N, 40, F)   [indicator&time mixing 인코더]
        │  channel_fc로 지표(F) 차원 압축 → (N, 40)
        ▼
      두 갈래로 분기:
        - y 갈래: time_fc로 시간(40)→1  ............ 종목 '자기 자신'의 표현
        - z 갈래: stock_mixer(시장영향) 후 time_fc_로 40→1 ... '시장 영향' 표현
        ▼
      최종 예측 = y + z  → (N, 1)   (각 종목의 다음날 종가 예측)

    생성자 인자(train.py에서 전달):
      stocks=1026, time_steps=16, channels=5(=F), market=20(=m), scale=3(미사용 인자)
    """
    def __init__(self, stocks, time_steps, channels, market, scale):
        super(StockMixer, self).__init__()
        scale_dim = 8  # 다운샘플 스케일의 시간 길이 (= time_steps/2)

        # indicator&time mixing 인코더 (원본 16 + 다운샘플 8 스케일을 합쳐 시간축 40 생성)
        self.mixer = MultTime2dMixer(time_steps, channels, scale_dim=scale_dim)

        # 지표(채널) 축을 1로 줄이는 선형층: (N, 40, F) → (N, 40)
        self.channel_fc = nn.Linear(channels, 1)

        # 시간축(40)을 1로 줄이는 선형층 (자기 표현 갈래용)
        self.time_fc = nn.Linear(time_steps * 2 + scale_dim, 1)

        # 시간축 1/2 다운샘플용 conv: 시간축에 kernel=2, stride=2 → 길이 절반
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels,
                              kernel_size=2, stride=2)

        # stock mixing 모듈 (N 종목 ↔ m=market 병목)
        self.stock_mixer = NoGraphMixer(stocks, market)

        # 시장영향 갈래의 시간축(40)→1 선형층 (자기 갈래와 별도 파라미터)
        self.time_fc_ = nn.Linear(time_steps * 2 + scale_dim, 1)

    def forward(self, inputs):
        # inputs: (N, T, F) = (1026, 16, 5)

        # --- 1) 다운샘플 스케일 만들기 ---
        x = inputs.permute(0, 2, 1)   # (N, F, T) : conv1d는 (batch, channel, length) 형태 요구
        x = self.conv(x)              # 시간축 16 → 8 다운샘플 → (N, F, 8)
        x = x.permute(0, 2, 1)        # (N, 8, F) : 다시 (종목, 시간, 지표) 배열로

        # --- 2) indicator & time mixing (multi-scale 인코딩) ---
        y = self.mixer(inputs, x)     # (N, 40, F)  [원본+원본mix+다운샘플mix를 시간축으로 concat]

        # --- 3) 지표(채널) 차원 축소 ---
        y = self.channel_fc(y).squeeze(-1)  # (N, 40, 1) → (N, 40)

        # --- 4) 두 갈래로 분기 ---
        z = self.stock_mixer(y)       # 시장 영향 표현 (N, 40) : 종목 간 관계 반영
        y = self.time_fc(y)           # 자기 표현을 시간축 압축 (N, 1)
        z = self.time_fc_(z)          # 시장 영향 표현을 시간축 압축 (N, 1)

        # --- 5) 합쳐서 최종 예측가 ---
        return y + z                  # (N, 1) : 각 종목의 다음날 예측 종가