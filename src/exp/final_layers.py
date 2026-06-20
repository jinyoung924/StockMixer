"""
결합부(combine head) — '같은 정보(m_τ)를 어떤 구조로 주입하느냐'만 변수로 둔 통제 비교.

StockMixer 마지막 단계는 두 표현을 합쳐 종목별 예측 1개를 낸다.
  y : 종목 본연 표현 (temporal/channel mix)   (N, h)   h = time_steps*2 + scale_dim = 40
  z : 시장 영향 표현 (cross-stock mix)          (N, h)
원본은  time_fc(y) + time_fc_(z)  로 '시장 레짐 무시' 고정 합산한다.

본 실험은 CSI300 위에서 외부 시장레짐 m_τ ∈ R^63 (MASTER csi_market_information)을
readout 직전 40차원 y,z 에 동일 주입하되, 주입 '구조'만 다르게 해서
'덧셈적 concat vs 곱셈적 gating'이 횡단면 랭킹을 바꾸는지 검증한다.

  add        : m_τ 미사용 (baseline / add arm 공용 헤드, 정보 미반영 기준점)
  concat_lin : m_τ 를 선형 concat → Linear([y;m]) = W_y·y + W_m·m + b
               W_m·m+b 는 그날 전 종목 공통 상수 → 순위 불변 (D1: 선형 주입의 한계 실증)
  concat_mlp : m_τ 를 비선형 concat (GELU) → y×m 상호작용 허용 (강한 베이스라인)
  gate       : m_τ 로 종목별 y/z 반영비를 곱셈적으로 차등 스케일 (주 모델, MASTER식)

공통 규약: 입력 (y, z, m_tau) → 출력 (N, 1).
  - m_tau 는 '그날 하나의 벡터' (m_dim,) 또는 전 종목 동일 broadcast (N, m_dim).
  - baseline/add(=AddHead)는 m_tau 를 무시한다.
  - concat/gate 는 m_dim=63 외부 벡터를 받는다 (market_dim 으로 파라미터화).
"""
import torch
import torch.nn as nn

M_DIM = 63   # 시장레짐 m_τ 차원 (MASTER data/csi_market_information.csv, 일자별 63열)


def as_market_vec(m_tau):
    """외부 m_τ 를 그날 단일 벡터 (m_dim,) 로 환원.

    m_tau 는 (m_dim,) 또는 (N, m_dim)(종목 공통이라 모든 행 동일)일 수 있다.
    head 수식이 (m_dim,) 단일 벡터를 기대하므로 종목축이 있으면 평균으로 환원한다.
    """
    return m_tau.mean(0) if m_tau.dim() > 1 else m_tau


# ----------------------------------------------------------------------
# add : 고정 가중치 단순 합산 (원본 baseline). baseline arm(5피처)·add arm(13피처) 공용.
# ----------------------------------------------------------------------
class AddHead(nn.Module):
    def __init__(self, feat_dim, market_dim=None):
        super().__init__()
        self.time_fc = nn.Linear(feat_dim, 1)
        self.time_fc_ = nn.Linear(feat_dim, 1)

    def forward(self, y, z, m_tau=None):
        return self.time_fc(y) + self.time_fc_(z)


# ----------------------------------------------------------------------
# concat_lin : m_τ 를 변환 없이 y/z 에 붙여 선형 readout (약한 베이스라인, D1 진단용)
#   m_τ 항이 그날 전 종목 공통 상수라 횡단면 순위에 원리적으로 기여 못 함을 실증한다.
# ----------------------------------------------------------------------
class ConcatLinHead(nn.Module):
    def __init__(self, feat_dim, market_dim=M_DIM):
        super().__init__()
        self.time_fc = nn.Linear(feat_dim + market_dim, 1)    # [y ; m_τ] -> 1
        self.time_fc_ = nn.Linear(feat_dim + market_dim, 1)   # [z ; m_τ] -> 1

    def forward(self, y, z, m_tau):
        m = as_market_vec(m_tau)
        m_bc = m.unsqueeze(0).expand(y.shape[0], -1)          # (N, m_dim) 전 종목 동일
        return self.time_fc(torch.cat([y, m_bc], -1)) \
             + self.time_fc_(torch.cat([z, m_bc], -1))


# ----------------------------------------------------------------------
# concat_mlp : m_τ 를 concat 하되 GELU 비선형으로 y×m_τ 상호작용 허용 (강한 베이스라인)
#   concat 경로로도 m_τ 가 순위에 영향 가능 → "약한 상대만 이겼다" 반박을 막는 대조군.
# ----------------------------------------------------------------------
class ConcatMLPHead(nn.Module):
    def __init__(self, feat_dim, hidden=None, market_dim=M_DIM):
        super().__init__()
        hidden = hidden or feat_dim
        din = feat_dim + market_dim
        self.mlp_y = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.mlp_z = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, y, z, m_tau):
        m = as_market_vec(m_tau)
        m_bc = m.unsqueeze(0).expand(y.shape[0], -1)
        return self.mlp_y(torch.cat([y, m_bc], -1)) + self.mlp_z(torch.cat([z, m_bc], -1))


# ----------------------------------------------------------------------
# gate : m_τ 로 차원별 게이트를 만들어 y, z 에 곱해 결합 (주 모델, 곱셈적 주입)
#   a = h * softmax(W(m_τ)/β),  y/z 경쟁적 배분. β 작을수록 게이팅 강함.
#   a_y 는 종목 무관이지만 a_y * y[n] 은 종목마다 달라짐 → 횡단면 순위를 실제로 바꿈.
# ----------------------------------------------------------------------
class GateHead(nn.Module):
    def __init__(self, feat_dim, beta=2.0, market_dim=M_DIM):
        super().__init__()
        self.h = feat_dim
        self.beta = beta
        # 부록 스텁의 gate = Linear(m_dim, 2*h) → softmax(view(2,h),dim=0) 와 동치 구성:
        # y/z 게이트를 각각 둬 경쟁(softmax)을 채널별로 둔다.
        self.W_gy = nn.Linear(market_dim, feat_dim)   # m_τ → y 게이트 로짓
        self.W_gz = nn.Linear(market_dim, feat_dim)   # m_τ → z 게이트 로짓
        self.time_fc = nn.Linear(feat_dim, 1)
        self.time_fc_ = nn.Linear(feat_dim, 1)

    def forward(self, y, z, m_tau):
        m = as_market_vec(m_tau)
        # 채널별 y vs z 경쟁 배분: stack 후 softmax(dim=0) (부록의 view(2,h) 와 동일 의미)
        g = torch.stack([self.W_gy(m), self.W_gz(m)], dim=0) / self.beta   # (2, h)
        a = torch.softmax(g, dim=0)                                        # (2, h)
        a_y, a_z = self.h * a[0], self.h * a[1]                            # (h,) 각
        return self.time_fc(a_y * y) + self.time_fc_(a_z * z)


# ----------------------------------------------------------------------
# 레지스트리 : combine_mode -> head 클래스
# ----------------------------------------------------------------------
COMBINE_REGISTRY = {
    "add": AddHead,                 # baseline / add arm (m_τ 미사용)
    "concat_lin": ConcatLinHead,    # 덧셈적 concat — 약한 베이스라인 (D1)
    "concat_mlp": ConcatMLPHead,    # 덧셈적 concat + 비선형 — 강한 베이스라인
    "gate": GateHead,               # 곱셈적 게이팅 — 주 모델
}


def build_head(combine_mode, feat_dim, beta=2.0, hidden=None, market_dim=M_DIM):
    """combine_mode 로 결합부 head 인스턴스를 생성.

    market_dim : m_τ 차원(외부 63). add 는 m_τ 미사용이라 무시된다.
    """
    if combine_mode not in COMBINE_REGISTRY:
        raise ValueError(
            f"등록되지 않은 combine_mode='{combine_mode}'. 사용 가능: {sorted(COMBINE_REGISTRY)}"
        )
    cls = COMBINE_REGISTRY[combine_mode]
    if combine_mode == "add":
        return cls(feat_dim, market_dim=market_dim)
    if combine_mode == "gate":
        return cls(feat_dim, beta=beta, market_dim=market_dim)
    if combine_mode == "concat_mlp":
        return cls(feat_dim, hidden=hidden, market_dim=market_dim)
    return cls(feat_dim, market_dim=market_dim)   # concat_lin


def list_modes():
    return sorted(COMBINE_REGISTRY)
