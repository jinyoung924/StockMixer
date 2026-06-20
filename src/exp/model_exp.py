"""
원본 StockMixer 의 mixing 부분은 그대로 재사용하고, 결합부(combine head)만 교체하는 래퍼.

원본 model.py 의 forward 와 동일하게:
  conv -> MultTime2dMixer -> channel_fc -> (종목 본연 y, 시장 영향 z) 생성
까지 진행한 뒤, 마지막 'y+z' 부분만 combine_mode 로 교체한다.
combine_mode='add' 이면 원본 StockMixer 와 동일한 동작(고정 합산)이다.

indicator/time/stock mixing 은 시장 불변 연산이므로 건드리지 않는다 (결합부만 동적화).
arm 간 차이는 (1) 입력 채널 수 fea_num(5 baseline / 13 add·concat·gating), (2) 결합부 head 뿐.
"""
import os
import sys
import torch.nn as nn

# src/exp/ 에서 부모 src 의 원본 model.py(검증된 mixing 블록)를 import 하기 위한 경로 추가
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from model import MultTime2dMixer, NoGraphMixer   # 원본 검증 블록 그대로 재사용
from final_layers import build_head, M_DIM


class StockMixerExp(nn.Module):
    def __init__(self, stocks, time_steps, channels, market, scale,
                 combine_mode="add", beta=2.0, hidden=None, market_dim=M_DIM):
        super().__init__()
        # scale_dim = conv(k=2,s=2) 출력 시간축 길이 = T//2.
        # T=16(lookback) → scale_dim=8 → feat_dim h = time_steps*2 + scale_dim = 40 (부록 H=40 일치).
        scale_dim = time_steps // 2
        self.combine_mode = combine_mode

        # ---- 원본과 동일한 인코더 ----
        self.mixer = MultTime2dMixer(time_steps, channels, scale_dim=scale_dim)
        self.channel_fc = nn.Linear(channels, 1)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels,
                              kernel_size=2, stride=2)
        self.stock_mixer = NoGraphMixer(stocks, market)   # market = NoGraphMixer hidden(=20), m_τ 무관

        # ---- 교체 대상: 결합부 ----
        self.feat_dim = time_steps * 2 + scale_dim                 # = 40
        self.head = build_head(combine_mode, self.feat_dim,
                               beta=beta, hidden=hidden, market_dim=market_dim)

    def forward(self, inputs, m_tau=None):
        # m_tau: 그날 외부 시장레짐 (m_dim,) 또는 (N, m_dim). add 헤드는 무시.
        x = inputs.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        y = self.mixer(inputs, x)
        y = self.channel_fc(y).squeeze(-1)   # (N, feat_dim) : 종목 본연 표현
        z = self.stock_mixer(y)              # (N, feat_dim) : 시장 영향 표현
        return self.head(y, z, m_tau)        # (N, 1)

    def describe(self):
        n_params = sum(p.numel() for p in self.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        return {
            "combine_mode": self.combine_mode,
            "total_params": n_params,
            "head_params": head_params,   # 용량 통제: concat_mlp vs gate 비교 시 반드시 로깅
        }
