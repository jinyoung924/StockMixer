"""
원본 StockMixer 의 mixing 부분은 그대로 재사용하고, 최종 head 만 교체 가능한 래퍼.

원본 model.py 의 forward 와 동일하게:
  conv -> MultTime2dMixer -> channel_fc -> (개별표현 y, 시장표현 z) 생성
까지 진행한 뒤, 마지막 'y+z' 부분만 pluggable head 로 대체한다.
head='original' 이면 원본과 수치적으로 동일한 동작을 한다.
"""
import os
import sys
import torch.nn as nn

# src/exp/ 에서 부모 src 의 원본 model.py 를 import 하기 위한 경로 추가
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# 원본 model.py 의 검증된 블록들을 그대로 가져와 재사용
from model import MultTime2dMixer, NoGraphMixer
from final_layers import build_head


class StockMixerExp(nn.Module):
    def __init__(self, stocks, time_steps, channels, market, scale,
                 variant="original", version=1):
        super().__init__()
        scale_dim = 8
        self.variant = variant
        self.version = version

        # ---- 원본과 동일한 인코더 부분 ----
        self.mixer = MultTime2dMixer(time_steps, channels, scale_dim=scale_dim)
        self.channel_fc = nn.Linear(channels, 1)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels,
                              kernel_size=2, stride=2)
        self.stock_mixer = NoGraphMixer(stocks, market)

        # ---- 교체 대상: 최종 head ----
        self.feat_dim = time_steps * 2 + scale_dim   # 원본 time_fc 입력차원과 동일
        self.head = build_head(variant, version, self.feat_dim)

    def forward(self, inputs):
        x = inputs.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        y = self.mixer(inputs, x)
        y = self.channel_fc(y).squeeze(-1)   # (stocks, feat_dim) : 개별 표현
        z = self.stock_mixer(y)              # (stocks, feat_dim) : 시장 표현
        return self.head(y, z)               # (stocks, 1)

    def describe(self):
        n_params = sum(p.numel() for p in self.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        return {
            "variant": self.variant,
            "version": self.version,
            "total_params": n_params,
            "head_params": head_params,
        }