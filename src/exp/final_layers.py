"""
결합부(combine) 메커니즘 4종 — add → gate → film → attn 의 '표현력 사다리'.

StockMixer 마지막 단계는 두 표현을 합쳐 종목별 예측 1개를 낸다.
  y : 종목 본연 표현        (stocks, h)
  z : 시장 영향 표현        (stocks, h)
원본은 time_fc(y) + time_fc_(z) 로 '고정 가중치' 합산한다.

여기서는 같은 날짜의 시장 상태 m_tau 로 결합 비율을 '동적'으로 조절하는
메커니즘들을 단계적으로 복잡하게 쌓아, 표현력이 높을수록 정말 좋아지는지
(아니면 데이터 부족으로 과적합하는지) 검증한다.

공통 규약 (공정 비교를 위해 모든 모드가 동일):
  - 입력은 (y, z) 로 한정 (+ 내부 합성 m_tau, mask 는 유효 종목 표시용).
  - 출력은 (stocks, 1).
  - y/z 를 스칼라로 사상하는 time_fc / time_fc_ 는 모든 모드에 공통.
combine_mode ∈ {'add','gate','film','attn'} 로 전환한다.
"""
import math
import torch
import torch.nn as nn


def make_market(y, mask=None):
    """종목축(dim=0) 통계로 그날의 시장 상태 m_tau=(2h,) 합성 (방식 B1).

    mask(유효 종목=1)가 주어지면 무효 종목이 평균을 오염시키지 않도록
    masked mean/std 로 계산한다 (NASDAQ 결측 대응). SP500 은 mask 전부 1.
    """
    if mask is not None:
        m = mask.view(-1, 1)                       # (N,1)
        denom = m.sum().clamp(min=1.0)
        mean = (y * m).sum(0) / denom
        std = torch.sqrt(((y - mean) ** 2 * m).sum(0) / denom + 1e-8)
    else:
        mean, std = y.mean(0), y.std(0)
    return torch.cat([mean, std], dim=0)           # (2h,)


# ----------------------------------------------------------------------
# add : 고정 가중치 단순 합산 (원본 baseline, 표현력 최저)
# ----------------------------------------------------------------------
class AddHead(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.time_fc = nn.Linear(feat_dim, 1)
        self.time_fc_ = nn.Linear(feat_dim, 1)

    def forward(self, y, z, mask=None):
        return self.time_fc(y) + self.time_fc_(z)


# ----------------------------------------------------------------------
# gate : m_tau 로 차원별 게이트를 만들어 y, z 에 곱해 결합 (MASTER식)
#        a = h * softmax(W(m_tau)/beta) — beta 작을수록 게이팅 강함
# ----------------------------------------------------------------------
class GateHead(nn.Module):
    def __init__(self, feat_dim, beta=5.0):
        super().__init__()
        self.h = feat_dim
        self.beta = beta
        self.W_gy = nn.Linear(2 * feat_dim, feat_dim)   # m_tau → y 게이트
        self.W_gz = nn.Linear(2 * feat_dim, feat_dim)   # m_tau → z 게이트
        self.time_fc = nn.Linear(feat_dim, 1)
        self.time_fc_ = nn.Linear(feat_dim, 1)

    def forward(self, y, z, mask=None):
        m = make_market(y, mask)
        a_y = self.h * torch.softmax(self.W_gy(m) / self.beta, dim=-1)   # (h,)
        a_z = self.h * torch.softmax(self.W_gz(m) / self.beta, dim=-1)   # (h,)
        return self.time_fc(a_y * y) + self.time_fc_(a_z * z)


# ----------------------------------------------------------------------
# film : m_tau 가 scale(gamma) + shift 로 y, z 를 변조 후 결합 (gate 상위)
#        gate 와 달리 '기준선(shift)'까지 시장이 조절 → 표현력 ↑
# ----------------------------------------------------------------------
class FiLMHead(nn.Module):
    def __init__(self, feat_dim, beta=5.0):
        super().__init__()
        self.W_fy = nn.Linear(2 * feat_dim, 2 * feat_dim)   # m_tau → (gamma_y, shift_y)
        self.W_fz = nn.Linear(2 * feat_dim, 2 * feat_dim)   # m_tau → (gamma_z, shift_z)
        self.time_fc = nn.Linear(feat_dim, 1)
        self.time_fc_ = nn.Linear(feat_dim, 1)

    def forward(self, y, z, mask=None):
        m = make_market(y, mask)
        gamma_y, shift_y = self.W_fy(m).chunk(2, dim=-1)
        gamma_z, shift_z = self.W_fz(m).chunk(2, dim=-1)
        y_mod = gamma_y * y + shift_y
        z_mod = gamma_z * z + shift_z
        return self.time_fc(y_mod) + self.time_fc_(z_mod)


# ----------------------------------------------------------------------
# attn : y, z 를 토큰 2개로 보고 m_tau 를 query 로 한 어텐션 가중합 (표현력 최고)
#        토큰이 2개뿐이라 과적합 위험은 제한적이되 가장 무겁다.
#        ※ 종목간/시간 어텐션으로 확장 금지 — 입력은 (y,z,m_tau)로 한정.
# ----------------------------------------------------------------------
class AttnHead(nn.Module):
    def __init__(self, feat_dim, d_attn=16):
        super().__init__()
        self.d_attn = d_attn
        self.W_q = nn.Linear(2 * feat_dim, d_attn)   # m_tau → query
        self.W_k = nn.Linear(feat_dim, d_attn)       # 토큰(y,z) → key
        self.time_fc = nn.Linear(feat_dim, 1)
        self.time_fc_ = nn.Linear(feat_dim, 1)

    def forward(self, y, z, mask=None):
        m = make_market(y, mask)
        sy = self.time_fc(y)             # (N,1)  토큰 y 의 값
        sz = self.time_fc_(z)            # (N,1)  토큰 z 의 값
        q = self.W_q(m)                  # (d_attn,)
        k_y = self.W_k(y.mean(0))        # (d_attn,)  토큰 키 (시장 평균)
        k_z = self.W_k(z.mean(0))
        scores = torch.stack([q @ k_y, q @ k_z]) / math.sqrt(self.d_attn)
        w = torch.softmax(scores, dim=-1)   # (2,) y/z 가중치
        return w[0] * sy + w[1] * sz


# ----------------------------------------------------------------------
# 레지스트리 : combine_mode -> head 클래스
# ----------------------------------------------------------------------
COMBINE_REGISTRY = {
    "add": AddHead,
    "gate": GateHead,
    "film": FiLMHead,
    "attn": AttnHead,
}


def build_head(combine_mode, feat_dim, beta=5.0, d_attn=16):
    """combine_mode 로 결합부 head 인스턴스를 생성."""
    if combine_mode not in COMBINE_REGISTRY:
        raise ValueError(
            f"등록되지 않은 combine_mode='{combine_mode}'. "
            f"사용 가능: {sorted(COMBINE_REGISTRY)}"
        )
    cls = COMBINE_REGISTRY[combine_mode]
    if combine_mode in ("gate", "film"):
        return cls(feat_dim, beta=beta)
    if combine_mode == "attn":
        return cls(feat_dim, d_attn=d_attn)
    return cls(feat_dim)


def list_modes():
    """등록된 combine_mode 목록."""
    return sorted(COMBINE_REGISTRY)
