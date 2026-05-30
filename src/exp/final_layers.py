"""
최종 레이어(head) 교체 실험용 모듈.

StockMixer 의 마지막 단계는 두 표현을 합쳐 종목별 예측 1개를 내는 부분이다.
  y : 개별 종목의 시계열 표현      (stocks, feat_dim)
  z : 시장/종목 mixing 을 거친 표현 (stocks, feat_dim)
원본은 Linear(y) + Linear(z) 로 단순히 더한다.

여기서는 이 'head' 를 갈아끼울 수 있도록 분리하고,
(variant 이름, version 번호) → head 클래스 로 매핑하는 레지스트리를 둔다.
새 아이디어가 생기면 새 클래스를 만들고 레지스트리에 version 을 추가하면 된다.

모든 head 는 forward(y, z) -> (stocks, 1) 형태를 지킨다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# original : 원본 구현 그대로 (y, z 를 각각 선형사상 후 합)
# ----------------------------------------------------------------------
class OriginalHead(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.time_fc = nn.Linear(feat_dim, 1)
        self.time_fc_ = nn.Linear(feat_dim, 1)

    def forward(self, y, z):
        return self.time_fc(y) + self.time_fc_(z)


# ----------------------------------------------------------------------
# FiLM : 시장 표현 z 가 개별 표현 y 를 feature-wise 로 변조(modulate)
#        gamma(z) * y + beta(z) 후 1차원으로 사상
#   - v1 : 단일 선형층으로 gamma/beta 생성
#   - v2 : 작은 MLP 로 gamma/beta 생성 (표현력 ↑) — 직접 실험용 예시
# ----------------------------------------------------------------------
class FiLMHeadV1(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.to_gamma = nn.Linear(feat_dim, feat_dim)
        self.to_beta = nn.Linear(feat_dim, feat_dim)
        self.proj = nn.Linear(feat_dim, 1)

    def forward(self, y, z):
        gamma = self.to_gamma(z)
        beta = self.to_beta(z)
        modulated = gamma * y + beta
        return self.proj(modulated)


class FiLMHeadV2(nn.Module):
    def __init__(self, feat_dim, hidden=None):
        super().__init__()
        hidden = hidden or feat_dim
        self.gen = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat_dim * 2),  # gamma, beta 한 번에
        )
        self.proj = nn.Linear(feat_dim, 1)

    def forward(self, y, z):
        gamma, beta = self.gen(z).chunk(2, dim=-1)
        modulated = gamma * y + beta
        return self.proj(modulated)


# ----------------------------------------------------------------------
# market gating : 시장 표현 z 로부터 게이트를 만들어 시장 신호의 반영 정도를 조절
#   - v1 : 스칼라 게이트 (종목별 1개)   pred = fc(y) + g * fc_(z)
#   - v2 : 벡터 게이트 (feature별)      먼저 z 를 게이트로 y 에 적용 후 합
# ----------------------------------------------------------------------
class MarketGatingHeadV1(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.time_fc = nn.Linear(feat_dim, 1)
        self.time_fc_ = nn.Linear(feat_dim, 1)
        self.gate = nn.Linear(feat_dim, 1)

    def forward(self, y, z):
        g = torch.sigmoid(self.gate(z))     # (stocks, 1)  0~1
        return self.time_fc(y) + g * self.time_fc_(z)


class MarketGatingHeadV2(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.gate = nn.Linear(feat_dim, feat_dim)   # feature별 게이트
        self.proj = nn.Linear(feat_dim, 1)

    def forward(self, y, z):
        g = torch.sigmoid(self.gate(z))     # (stocks, feat_dim)
        fused = y * (1 - g) + z * g         # 게이트로 y/z 를 혼합
        return self.proj(fused)


# ----------------------------------------------------------------------
# 레지스트리 : (variant, version) -> head 클래스
#   새 실험을 추가할 때 여기에 한 줄만 등록하면 됨.
# ----------------------------------------------------------------------
FINAL_LAYER_REGISTRY = {
    ("original", 1): OriginalHead,
    ("film", 1): FiLMHeadV1,
    ("film", 2): FiLMHeadV2,
    ("market_gating", 1): MarketGatingHeadV1,
    ("market_gating", 2): MarketGatingHeadV2,
}


def build_head(variant, version, feat_dim):
    """variant 이름과 version 으로 head 인스턴스를 생성."""
    key = (variant, version)
    if key not in FINAL_LAYER_REGISTRY:
        raise ValueError(
            f"등록되지 않은 조합: variant='{variant}', version={version}\n"
            f"사용 가능: {sorted(FINAL_LAYER_REGISTRY.keys())}"
        )
    return FINAL_LAYER_REGISTRY[key](feat_dim)


def list_variants():
    """등록된 (variant, version) 목록."""
    return sorted(FINAL_LAYER_REGISTRY.keys())