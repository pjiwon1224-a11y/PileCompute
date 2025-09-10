# -*- coding: utf-8 -*-
"""
pile_module.py

말뚝 계산 파이프라인을 한 파일로 모아 둔 모듈.
- df2 -> 변위요약(summary) -> 말뚝별 작용력 테이블 -> 머리력 블록 -> Calc(깊이별 결과)
- 지진/일반 자동 분기, 최대 모멘트 지점 마킹 포함
- alpha(Col) 계산, 좌표/중립축 계산, 요약표(summarize_Mmax)까지 포함

필요 라이브러리: numpy, pandas
"""

from __future__ import annotations

import logging
import re
from typing import Sequence, Optional, Union, Dict, List, Set

import numpy as np
import pandas as pd












# -----------------------------------------------------------------------------
# Logger (선택)
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger("pile")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)


# =============================================================================
# 0) 기초 계산 유틸
# =============================================================================
def D_solver(init_Beta: float, K_H0: float, D1: float, E: float, I: float,
             tol: float = 1e-6, max_iter: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    상시용(평상시) 1/B 수렴 계산
    """
    B_value: List[float] = []
    KH_value: List[float] = []
    Beta = float(init_Beta)

    for _ in range(max_iter):
        B_H = np.sqrt(D1 * Beta)
        K_H = K_H0 * (B_H / 0.3) ** -0.75
        Beta_new = 1.0 / ((K_H * D1 / (4.0 * E * I)) ** 0.25)

        B_value.append(Beta)
        KH_value.append(K_H)

        if len(B_value) > 1:
            rel_err = abs((B_value[-1] - B_value[-2]) / B_value[-1])
            if rel_err < tol:
                break
        Beta = Beta_new

    return np.array(B_value), np.array(KH_value)


def D_solver2(init_Beta: float, K_H01: float, D1: float, E: float, I: float,
              tol: float = 1e-6, max_iter: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    지진용 1/B 수렴 계산
    """
    B_value: List[float] = []
    KH_value: List[float] = []
    Beta = float(init_Beta)

    for _ in range(max_iter):
        B_H = np.sqrt(D1 * Beta)
        K_H = K_H01 * (B_H / 0.3) ** -0.75
        Beta_new = 1.0 / ((K_H * D1 / (4.0 * E * I)) ** 0.25)

        B_value.append(Beta)
        KH_value.append(K_H)

        if len(B_value) > 1:
            rel_err = abs((B_value[-1] - B_value[-2]) / B_value[-1])
            if rel_err < tol:
                break
        Beta = Beta_new

    return np.array(B_value), np.array(KH_value)


def coord_general(coords: Sequence[float],
                  counts: Optional[Sequence[Union[int, float]]] = None,
                  x0: float = 0.0,
                  labels: Optional[Sequence[str]] = None,
                  return_df: bool = False,
                  return_group_signed: bool = False,
                  as_row: bool = True):
    """
    범용 중립축 계산기.

    반환 (기본):
      NE, coords_arr, distances, signed, groups
      [option] df
      [option] group_signed (NE - coords, 1xM or (M,))
    """
    coords = np.asarray(coords, dtype=float).ravel()
    if coords.size == 0:
        raise ValueError("coords는 최소 1개 이상이어야 합니다.")

    # counts
    if counts is None:
        counts_arr = np.ones(coords.size, dtype=int)
    else:
        counts_arr = np.asarray(counts)
        if counts_arr.size == 1:
            val = int(np.asarray(counts_arr).reshape(-1)[0])
            counts_arr = np.full(coords.size, val, dtype=int)
        else:
            if counts_arr.size != coords.size:
                raise ValueError("counts 길이는 coords와 같아야 합니다.")
            counts_arr = counts_arr.astype(int, copy=False)

    if np.any(counts_arr < 0):
        raise ValueError("counts에는 음수가 들어갈 수 없습니다.")

    N = int(counts_arr.sum())
    if N == 0:
        raise ValueError("총 말뚝 본수 N은 0보다 커야 합니다.")

    # labels
    if labels is None:
        labels = [f"G{i+1}" for i in range(coords.size)]
    else:
        if len(labels) != coords.size:
            raise ValueError("labels 길이는 coords와 같아야 합니다.")

    # 원점 보정
    coords0 = coords - float(x0)

    # 중립축(가중평균)
    NE = float((coords0 * counts_arr).sum() / N)

    # 전체 전개
    coords_arr = np.repeat(coords0, counts_arr)
    groups = np.repeat(np.arange(coords.size, dtype=int), counts_arr)

    signed = NE - coords_arr
    distances = np.abs(signed)

    group_signed = None
    if return_group_signed:
        group_signed = NE - coords0
        if as_row:
            group_signed = group_signed.reshape(1, -1)

    if return_df:
        df = pd.DataFrame({
            "group_idx": groups,
            "group_label": [labels[g] for g in groups],
            "coord": coords_arr,
            "signed": signed,
            "distance": distances
        })
        if return_group_signed:
            return NE, coords_arr, distances, signed, groups, df, group_signed
        return NE, coords_arr, distances, signed, groups, df

    if return_group_signed:
        return NE, coords_arr, distances, signed, groups, group_signed
    return NE, coords_arr, distances, signed, groups


def Col(kind: str, *, Lp: float, D2: float) -> float:
    """
    말뚝 종류별 α = a*(Lp/D2) + b
    - 공백/하이픈/밑줄/공법 접미어 제거, 흔한 표기(쏘일/소일) 허용
    """
    key = str(kind).strip()
    key = key.replace(" ", "").replace("-", "").replace("_", "")
    key = key.replace("쏘일", "소일")
    key = key.replace("말뚝공법", "말뚝")
    key = key.replace("공법", "")

    COEF = {
        '타격공법':       (0.014,  0.72),
        '타격':          (0.014,  0.72),
        '바이브로해머공법': (0.014, -0.014),
        '바이브로해머':    (0.014, -0.014),
        '현장타설말뚝':    (0.031, -0.15),
        '현장타설':       (0.031, -0.15),
        '내부굴착말뚝':    (0.010,  0.36),
        '내부굴착':       (0.010,  0.36),
        '프리보링말뚝':    (0.13,   0.53),
        '프리보링':       (0.13,   0.53),
        '내부굴착강관말뚝': (0.013,  0.53),
        '강관말뚝':       (0.013,  0.53),
        '소일시멘트말뚝':   (0.04,   0.15),
        '소일시멘트':      (0.04,   0.15),
    }
    ALIAS = [
        ("타격", "타격"),
        ("바이브", "바이브로해머"),
        ("현장타설", "현장타설"),
        ("프리보링", "프리보링"),
        ("강관", "강관말뚝"),
        ("내부굴착강관", "강관말뚝"),
        ("내부굴착", "내부굴착"),
        ("소일시멘트", "소일시멘트"),
    ]

    if key not in COEF:
        for kw, name in ALIAS:
            if kw in key:
                a, b = COEF[name]
                return a * (Lp / D2) + b
        alpha = float(input("기타말뚝: α 값을 직접 입력하세요: "))
        return alpha

    a, b = COEF[key]
    return a * (Lp / D2) + b


def Calc(H: float, Mt: float, dx: float, Lp: float, B: float = 0.0, E: float = 0.0, I: float = 0.0, *,
         as_df: bool = True,
         ndigits: int = 3,
         include_Mmax_row: bool = True,
         tol: float = 1e-9):
    """
    깊이별 모멘트/전단력/변위 계산.
    - Lp는 반드시 인자로 전달(전역 의존 제거).
    - 이론 최대 모멘트 위치 L_m 포함 & '최대 모멘트 발생 지점' 마킹.
    - 사용 불가하면 z=0 제외 격자 |M| 최대인 행 마킹.
    """
    dz = Lp / 15.0
    h0 = Mt / H if H not in (0, 0.0) else np.nan

    # z>0 격자
    z = np.linspace(dz, Lp, 15) if dz > 0 else np.array([Lp], dtype=float)

    if B == 0:
        eBz = np.ones_like(z); cBz = np.ones_like(z); sBz = np.zeros_like(z)
    else:
        eBz = np.exp(-B * z)
        cBz = np.cos(B * z)
        sBz = np.sin(B * z)

    if B == 0 or E == 0 or I == 0 or H == 0:
        y = np.zeros_like(z)
        M = np.zeros_like(z)
        S = np.zeros_like(z)
        L_m = np.nan
    else:
        y = H/(2*E*I*B**3) * eBz * ((1 + B*h0)*cBz - B*h0*sBz) * 1000.0
        M = -H/B * eBz * ((1 + B*h0)*sBz + B*h0*cBz)
        S = -H    * eBz * (cBz - (1 + 2*B*h0)*sBz)
        L_m = (1.0/B) * np.arctan(1.0/(1.0 + 2.0*B*h0))

    # z=0 삽입
    z = np.insert(z, 0, 0.0)
    y = np.insert(y, 0, dx * 1000.0)
    M = np.insert(M, 0, -Mt)
    S = np.insert(S, 0, -H)

    if not as_df:
        if np.isfinite(L_m) and 0 <= L_m <= Lp and B not in (0, 0.0) and H not in (0, 0.0):
            e = np.exp(-B*L_m); c = np.cos(B*L_m); s = np.sin(B*L_m)
            M_m = -H/B * e * ((1 + B*h0)*s + B*h0*c)
        else:
            M_m = np.nan
        return z, M, S, y, L_m, M_m

    df = pd.DataFrame({
        '깊이(m)':      np.asarray(z, float),
        '모멘트(kN.m)': np.asarray(M, float),
        '전단력(kN)':   np.asarray(S, float),
        '변위(mm)':     np.asarray(y, float),
        '비 고':        [''] * len(z)
    })

    # Lm 행 포함 & 마킹
    marked = False
    close_atol = max(tol, 1e-6)

    if include_Mmax_row and np.isfinite(L_m) and (0 <= L_m <= Lp) and (B not in (0, 0.0)) and (H not in (0, 0.0)):
        e = np.exp(-B*L_m); c = np.cos(B*L_m); s = np.sin(B*L_m)
        y_l = H/(2*E*I*B**3) * e * ((1 + B*h0)*c - B*h0*s) * 1000.0
        M_l = -H/B * e * ((1 + B*h0)*s + B*h0*c)
        S_l = -H    * e * (c - (1 + 2*B*h0)*s)

        if not np.any(np.isclose(df['깊이(m)'].to_numpy(), L_m, atol=close_atol)):
            df = pd.concat([
                df,
                pd.DataFrame({
                    '깊이(m)':      [float(L_m)],
                    '모멘트(kN.m)': [float(M_l)],
                    '전단력(kN)':   [float(S_l)],
                    '변위(mm)':     [float(y_l)],
                    '비 고':        ['']
                })
            ], ignore_index=True).sort_values('깊이(m)').reset_index(drop=True)

        zvals = df['깊이(m)'].to_numpy()
        dist  = np.abs(zvals - float(L_m))
        if dist.size > 0:
            dist[0] = np.inf  # z=0 제외
        idx = int(np.argmin(dist))

        df.loc[:, '비 고'] = ''
        df.iloc[idx, df.columns.get_loc('비 고')] = '최대 모멘트 발생 지점'
        marked = True

    if not marked and len(df) > 1:
        seg = df['모멘트(kN.m)'].to_numpy()[1:]  # z=0 제외
        idx = 1 + int(np.nanargmax(np.abs(seg)))
        df.loc[:, '비 고'] = ''
        df.iloc[idx, df.columns.get_loc('비 고')] = '최대 모멘트 발생 지점'

    df = df.round({'깊이(m)': ndigits, '모멘트(kN.m)': ndigits,
                   '전단력(kN)': ndigits, '변위(mm)': ndigits})
    return df


def summarize_Mmax(results_dict: Dict[str, pd.DataFrame],
                   ndigits: int = 3,
                   prefer_theory: bool = False) -> pd.DataFrame:
    """
    콤보별 최대 모멘트 요약표
    """
    rows: List[Dict[str, Union[str, float]]] = []
    for combo, df in results_dict.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        idx = df['모멘트(kN.m)'].abs().idxmax()
        y_max = float(df.at[idx, '변위(mm)'])
        M_max = float(df.at[idx, '모멘트(kN.m)'])
        z_at  = float(df.at[idx, '깊이(m)'])
        note  = str(df.at[idx, '비 고'])

        if prefer_theory and 'theory' in note:
            m = re.search(r"theory\s+([-\d.]+)\s*@\s*([-\d.]+)", note)
            if m:
                try:
                    M_max = float(m.group(1))
                    z_at  = float(m.group(2))
                except Exception:
                    pass

        rows.append({'콤보': combo,
                     'z_max(m)': round(z_at, ndigits),
                     'y_max(mm)': round(y_max, ndigits),
                     'M_max(kN.m)': round(M_max, ndigits),
                     '비 고': note})
    out = pd.DataFrame(rows).sort_values(
        '콤보',
        key=lambda s: s.astype(str).str.extract(r'(\d+)').astype(float).fillna(1e9)[0]
    )
    return out.reset_index(drop=True)

def summarize_Ymax(results_dict: dict[str, pd.DataFrame], ndigits: int = 3) -> pd.DataFrame:
    """
    Calc() 결과 dict -> 콤보별 최대 변위(|y|) 지점 요약.
    - z=0 포함
    return columns: ['콤보','z_at(m)','y_max(mm)','M_at(kN.m)']
    """
    rows = []
    for combo, df in (results_dict or {}).items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        y = pd.to_numeric(df['변위(mm)'], errors='coerce').to_numpy()
        M = pd.to_numeric(df['모멘트(kN.m)'], errors='coerce').to_numpy()
        z = pd.to_numeric(df['깊이(m)'], errors='coerce').to_numpy()
        if y.size == 0:
            continue
        idx = int(np.nanargmax(np.abs(y)))  # 전체에서 최대값 탐색
        rows.append({
            '콤보': combo,
            'z_at(m)': round(float(z[idx]), ndigits),
            'y_max(mm)': round(float(y[idx]), ndigits),
            'M_at(kN.m)': round(float(M[idx]), ndigits),
        })
    return pd.DataFrame(rows).sort_values(
        '콤보',
        key=lambda s: s.astype(str).str.extract(r'(\d+)').astype(float).fillna(1e9)[0]
    ).reset_index(drop=True)

def springs_summary_from_bundle(bundle: dict) -> dict[str, pd.DataFrame]:
    """
    bundle['groups']에서 K1..K4/Kv 값을 요약 테이블로 만든다.
    return: {'고정': df, '힌지': df or None}
    """
    def to_df(name, ks_norm, ks_eq):
        if not ks_norm:
            return None
        rows = []
        for key in ['Kv','K1','K2','K3','K4']:
            rows.append({
                '항목': key,
                '상시': float(ks_norm.get(key, np.nan)),
                '지진': float((ks_eq or {}).get(key, np.nan))
            })
        return pd.DataFrame(rows, columns=['항목','상시','지진'])

    groups = bundle.get('groups', {})
    out = {}
    if '고정' in groups:
        out['고정'] = to_df('고정', groups['고정']['Ks']['normal'], groups['고정']['Ks']['eq'])
    if '힌지' in groups and groups['힌지'].get('all_out') is not None:
        out['힌지'] = to_df('힌지', groups['힌지']['Ks']['normal'], groups['힌지']['Ks']['eq'])
    return out




# =============================================================================
# 1) 파이프라인 유틸 (df2 -> summary)
# =============================================================================
def _combo_key(s: str) -> int:
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else 10 ** 9


def _norm_group(sr: pd.Series) -> pd.Series:
    return sr.astype(str).str.replace(r"\s+", "", regex=True)


def calc_disp_from_LC(df2: pd.DataFrame, IK_Mat, IK_Mat_e,
                      swap_first_two: bool = True,
                      round_ndigits: Optional[int] = 15):
    """
    df2: ['구 분','조합','Vo(kN)','Ho(kN)','Mo(kN.m)','비 고']
    IK_Mat / IK_Mat_e: 3x3 matrix (numpy array-like) - '지진'인 경우 IK_Mat_e 적용
    return: df_out, summary (MultiIndex columns: {'사용하중','계수하중'} × ['δx(m)','δy(m)','α(rad)'])
    """
    out = df2.copy()
    for c in ["Vo(kN)", "Ho(kN)", "Mo(kN.m)"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    R_cols = ["Ho(kN)", "Vo(kN)", "Mo(kN.m)"] if swap_first_two else ["Vo(kN)", "Ho(kN)", "Mo(kN.m)"]

    mask_e = out["비 고"].astype(str).str.contains("지진", na=False)

    IK   = np.asarray(IK_Mat,   dtype=float)
    IK_e = np.asarray(IK_Mat_e, dtype=float)
    if IK.shape != (3, 3) or IK_e.shape != (3, 3):
        raise ValueError("IK_Mat/IK_Mat_e는 (3,3)이어야 합니다.")

    R = out[R_cols].to_numpy()  # (N,3)
    N = len(out)
    res = np.full((N, 3), np.nan, float)

    idx_e = np.where(mask_e.values)[0]
    idx_n = np.where(~mask_e.values)[0]
    if idx_n.size: res[idx_n, :] = R[idx_n, :].dot(IK.T)
    if idx_e.size: res[idx_e, :] = R[idx_e, :].dot(IK_e.T)

    out[["δx(m)", "δy(m)", "α(rad)"]] = res

    grp = _norm_group(out["구 분"])
    use  = out.loc[grp.eq("사용하중"), ["조합", "δx(m)", "δy(m)", "α(rad)"]].set_index("조합")
    fact = out.loc[grp.eq("계수하중"), ["조합", "δx(m)", "δy(m)", "α(rad)"]].set_index("조합")

    use  = use.sort_index(key=lambda s: s.map(_combo_key))
    fact = fact.sort_index(key=lambda s: s.map(_combo_key))

    summary = pd.concat({"사용하중": use, "계수하중": fact}, axis=1)
    if round_ndigits is not None:
        summary = summary.round(round_ndigits)
    return out, summary


# =============================================================================
# 2) summary -> 좌표별 작용력 테이블
# =============================================================================
def per_pile_table_from_summary(summary: pd.DataFrame,
                                df_out: pd.DataFrame,
                                Co: np.ndarray,
                                Ks_normal: Dict[str, float],
                                Ks_eq: Dict[str, float],
                                group_name: str = "사용하중",
                                theta_by_pos: Optional[np.ndarray] = None,
                                round_map: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    반환: ['구 분','No','δx(i)','δy(i)','α(rad)','Pn(i)','Ph(i)','Mt(i)'] (콤보별 1..M행)
    """
    need = ['δx(m)', 'δy(m)', 'α(rad)']
    use = summary[group_name][need].copy()
    use = use.sort_index(key=lambda s: s.map(_combo_key))

    grp = _norm_group(df_out["구 분"])
    tag = (df_out.loc[grp.eq(group_name), ['조합', '비 고']]
                 .groupby('조합')['비 고']
                 .apply(lambda s: s.astype(str).str.contains("지진").any()))
    is_eq = tag.to_dict()

    Co = np.asarray(Co, dtype=float).ravel()
    M = Co.size
    if M == 0:
        raise ValueError("Co가 비었습니다.")

    if theta_by_pos is None:
        theta_by_pos = np.zeros(M, float)
    theta_by_pos = np.asarray(theta_by_pos, dtype=float).ravel()
    if theta_by_pos.size != M:
        raise ValueError("theta_by_pos 길이 불일치")

    cos = np.cos(theta_by_pos); sin = np.sin(theta_by_pos)

    rows: List[Dict[str, Union[str, int, float]]] = []
    for combo in use.index:
        dx = float(use.at[combo, 'δx(m)'])
        dy = float(use.at[combo, 'δy(m)'])
        a  = float(use.at[combo, 'α(rad)'])
        Ks = Ks_eq if is_eq.get(combo, False) else Ks_normal
        Kv, K1, K2, K3, K4 = Ks['Kv'], Ks['K1'], Ks['K2'], Ks['K3'], Ks['K4']

        dya  = dy + a * Co                # (M,)
        dx_i = dx * cos - dya * sin
        dy_i = dx * sin + dya * cos
        Pn_i = Kv * dy_i
        Ph_i = K1 * dx_i - K2 * a
        Mt_i = -K3 * dx_i + K4 * a

        for j in range(M):
            rows.append({
                '구 분': combo if j == 0 else '',
                'No': j + 1,
                'δx(i)': dx_i[j],
                'δy(i)': dy_i[j],
                'α(rad)': a,
                'Pn(i)': Pn_i[j],
                'Ph(i)': Ph_i[j],
                'Mt(i)': Mt_i[j],
            })

    out = pd.DataFrame(rows)
    if round_map is None:
        round_map = {'δx(i)': 7, 'δy(i)': 7, 'α(rad)': 7, 'Pn(i)': 3, 'Ph(i)': 3, 'Mt(i)': 3}
    for c, nd in round_map.items():
        out[c] = pd.to_numeric(out[c], errors='coerce').round(nd)
    return out


# =============================================================================
# 3) 머리력 정리 블록
# =============================================================================
def build_headforce_block(per_pile: pd.DataFrame,
                          Co: np.ndarray,
                          counts: np.ndarray,
                          theta_by_pos: np.ndarray | None = None,
                          round_dist: int = 3, round_ang: int = 0, round_force: int = 3,
                          force_zero_total_moment: bool | None = None,   # ← 추가
                          zero_tol: float = 1e-8                          # ← 추가
                          ) -> pd.DataFrame:
    """
    per_pile: per_pile_table_from_summary() 출력
    반환: 콤보별 블록(각 콤보 M행 + '계' 1행)

    force_zero_total_moment:
        - None(기본) : 자동 감지(모든 Mt(i)≈0이면 힌지로 간주하고 '계' 모멘트 0 처리)
        - True       : 무조건 '계' 모멘트 0 처리 (힌지 강제)
        - False      : 기존 로직 유지
    """
    need = ['구 분','No','Pn(i)','Ph(i)','Mt(i)']
    if any(c not in per_pile.columns for c in need):
        raise ValueError("per_pile 컬럼 누락")

    Co     = np.asarray(Co, float).ravel()
    counts = np.asarray(counts, int).ravel()
    M = Co.size
    if M == 0 or M != counts.size:
        raise ValueError("Co/counts 길이 불일치")

    if theta_by_pos is None:
        theta_by_pos = np.zeros(M, float)
    else:
        theta_by_pos = np.asarray(theta_by_pos, float).ravel()
        if theta_by_pos.size != M:
            raise ValueError("theta_by_pos 길이 불일치")

    df = per_pile.copy()
    df['콤보'] = df['구 분'].replace('', np.nan).ffill()
    df['No']   = pd.to_numeric(df['No'], errors='coerce').astype(int)
    df = df.sort_values(['콤보','No']).reset_index(drop=True)

    pos2x  = {i+1: Co[i] for i in range(M)}
    pos2th = {i+1: theta_by_pos[i] for i in range(M)}
    pos2N  = {i+1: counts[i] for i in range(M)}

    df['수평거리'] = df['No'].map(pos2x).astype(float)
    df['각도']     = np.degrees(df['No'].map(pos2th).astype(float))
    df['본수']     = df['No'].map(pos2N).astype(int)

    th = df['No'].map(pos2th).astype(float).to_numpy()
    cos_t, sin_t = np.cos(th), np.sin(th)
    Pn = pd.to_numeric(df['Pn(i)'], errors='coerce').to_numpy(float)
    Ph = pd.to_numeric(df['Ph(i)'], errors='coerce').to_numpy(float)
    Mt = pd.to_numeric(df['Mt(i)'], errors='coerce').to_numpy(float)

    V_i = Pn * cos_t - Ph * sin_t
    H_i = Pn * sin_t + Ph * cos_t

    body = pd.DataFrame({
        '콤보':     df['콤보'],
        'No':        df['No'],
        '수평거리':   df['수평거리'],
        '각도':      df['각도'],
        '축력':      V_i,
        '수평력':     H_i,
        '모멘트':     Mt,
        '본수':      df['본수'],
    })

    blocks = []
    for combo, part in body.groupby('콤보', sort=False):
        part = part.sort_values('No').reset_index(drop=True)
        Nvec = part['본수'].to_numpy(float)
        Vvec = part['축력'].to_numpy(float)
        Hvec = part['수평력'].to_numpy(float)
        Mvec = part['모멘트'].to_numpy(float)
        xvec = part['수평거리'].to_numpy(float)

        # 기본 합계
        Vt = float((Vvec * Nvec).sum())
        Ht = float((Hvec * Nvec).sum())
        Mt_total = float(((Vvec * xvec + Mvec) * Nvec).sum())

        # --- 힌지 판단/강제 0 처리 ---
        if force_zero_total_moment is None:
            # 자동: 모든 개별 Mt(i)가 거의 0이면 힌지로 보고 합계를 0 처리
            hinge_like = (np.nanmax(np.abs(Mvec)) if Mvec.size else 0.0) < zero_tol
            if hinge_like:
                Mt_total = 0.0
        elif force_zero_total_moment:
            Mt_total = 0.0

        # 아주 작은 수는 0으로 스냅
        if abs(Mt_total) < max(zero_tol, 10**(-round_force)):
            Mt_total = 0.0

        show = part[['No','수평거리','각도','축력','수평력','모멘트','본수']].copy()
        show.insert(0, '구 분', '')
        show.loc[0, '구 분'] = combo

        total = pd.DataFrame([{
            '구 분':'', 'No':'계', '수평거리':'', '각도':'',
            '축력':Vt, '수평력':Ht, '모멘트':Mt_total, '본수':int(Nvec.sum())
        }])
        blocks.append(pd.concat([show, total], ignore_index=True))

    out = pd.concat(blocks, ignore_index=True)
    out['수평거리'] = pd.to_numeric(out['수평거리'], errors='coerce').round(round_dist).astype(object)
    out['각도']     = pd.to_numeric(out['각도'], errors='coerce').round(round_ang).astype(object)
    for c in ['축력','수평력','모멘트']:
        out[c] = pd.to_numeric(out[c], errors='coerce').round(round_force)
    return out


# =============================================================================
# 4) Calc 실행 래퍼(지진 자동 판정 포함)
# =============================================================================
def _extract_params_for_calc(table_print: pd.DataFrame,
                             table_perpile: pd.DataFrame) -> pd.DataFrame:
    tp = table_print.copy()
    tp['구 분'] = tp['구 분'].replace('', np.nan).ffill()
    tp['_No'] = pd.to_numeric(tp['No'], errors='coerce')
    tp1 = (tp.loc[tp['_No'].eq(1), ['구 분', '수평력', '모멘트']]
             .groupby('구 분', sort=False).first()
             .rename(columns={'수평력': 'H', '모멘트': 'Mt'}))

    pp = table_perpile.copy()
    pp['구 분'] = pp['구 분'].replace('', np.nan).ffill()
    pp['_No'] = pd.to_numeric(pp['No'], errors='coerce')
    dx1 = (pp.loc[pp['_No'].eq(1), ['구 분', 'δx(i)']]
             .groupby('구 분', sort=False)['δx(i)'].first()
             .rename('dx'))

    params = tp1.join(dx1, how='inner')
    for c in ['H', 'Mt', 'dx']:
        params[c] = pd.to_numeric(params[c], errors='coerce')
    return params


def _detect_eq_combos_from_dfout(df_out: pd.DataFrame, group_name: str) -> Set[str]:
    grp = _norm_group(df_out['구 분'])
    sub = df_out.loc[grp.eq(group_name), ['조합', '비 고']].copy()
    is_eq = sub['비 고'].astype(str).str.contains("지진", na=False)
    tag = sub.assign(is_eq=is_eq).groupby('조합')['is_eq'].any()
    return set(tag[tag].index.tolist())


def run_Calc_per_combo(table_print: pd.DataFrame,
                       table_perpile: pd.DataFrame,
                       *,
                       B: float, E: float, I: float,
                       B_eq: Optional[float] = None, E_eq: Optional[float] = None, I_eq: Optional[float] = None,
                       eq_combos: Optional[Union[List[str], Set[str]]] = None,
                       df_out: Optional[pd.DataFrame] = None, group_name: Optional[str] = None,
                       as_df: bool = True, ndigits: int = 3, Lp: Optional[float] = None):
    """
    콤보별(No=1) H/Mt/dx 추출 후 Calc() 실행.
    반환: (results_dict, params_df(with is_eq & 사용 B/E/I))
    """
    if Lp is None:
        raise ValueError("run_Calc_per_combo(): Lp를 지정하세요.")

    params = _extract_params_for_calc(table_print, table_perpile)

    if eq_combos is not None:
        eq_set = set(eq_combos)
    elif (df_out is not None) and (group_name is not None):
        eq_set = _detect_eq_combos_from_dfout(df_out, group_name)
    else:
        eq_set = set()

    B_eq = B if B_eq is None else B_eq
    E_eq = E if E_eq is None else E_eq
    I_eq = I if I_eq is None else I_eq

    params = params.copy()
    params['is_eq'] = False
    params['B_used'] = np.nan
    params['E_used'] = np.nan
    params['I_used'] = np.nan

    results: Dict[str, pd.DataFrame] = {}
    for combo, row in params.iterrows():
        H, Mt, dx = float(row['H']), float(row['Mt']), float(row['dx'])
        use_eq = combo in eq_set
        Bb, Ee, Ii = (B_eq, E_eq, I_eq) if use_eq else (B, E, I)
        params.at[combo, 'is_eq'] = use_eq
        params.at[combo, 'B_used'] = Bb
        params.at[combo, 'E_used'] = Ee
        params.at[combo, 'I_used'] = Ii

        results[combo] = Calc(H=H, Mt=Mt, dx=dx, Lp=Lp, B=Bb, E=Ee, I=Ii,
                              as_df=as_df, ndigits=ndigits)
    return results, params


# =============================================================================
# 5) 풀 파이프라인
# =============================================================================
def build_all_tables(df2: pd.DataFrame,
                     IK_Mat, IK_Mat_e,
                     Ks_normal: Dict[str, float], Ks_eq: Dict[str, float],
                     Co: np.ndarray, counts: np.ndarray,
                     theta_by_pos: Optional[np.ndarray] = None,
                     *,
                     B: float, E: float, I: float,
                     B_eq: Optional[float] = None, E_eq: Optional[float] = None, I_eq: Optional[float] = None,
                     Lp: Optional[float] = None) -> Dict[str, object]:
    """
    입력: df2, IK행렬, 스프링계수(일반/지진), 좌표 Co, 본수 counts, (옵션) 각도
    출력 dict 키:
      'df_out','summary',
      'table_use','table_fact',
      'table_use_print','table_fact_print',
      'use_results','use_params',
      'fact_results','fact_params'
    """
    if Lp is None:
        raise ValueError("build_all_tables(): Lp가 필요합니다.")

    # 1) 변위 요약
    df_out, summary = calc_disp_from_LC(df2, IK_Mat, IK_Mat_e)

    # 2) 좌표별 작용력
    table_use  = per_pile_table_from_summary(summary, df_out, Co, Ks_normal, Ks_eq, '사용하중', theta_by_pos)
    table_fact = per_pile_table_from_summary(summary, df_out, Co, Ks_normal, Ks_eq, '계수하중', theta_by_pos)

    # 3) 머리력 정리표
    table_use_print  = build_headforce_block(table_use,  Co, counts, theta_by_pos)
    table_fact_print = build_headforce_block(table_fact, Co, counts, theta_by_pos)

    # 4) Calc
    use_results,  use_params  = run_Calc_per_combo(
        table_use_print,  table_use,  B=B, E=E, I=I,
        B_eq=B_eq, E_eq=E_eq, I_eq=I_eq,
        df_out=df_out, group_name='사용하중', Lp=Lp
    )
    fact_results, fact_params = run_Calc_per_combo(
        table_fact_print, table_fact, B=B, E=E, I=I,
        B_eq=B_eq, E_eq=E_eq, I_eq=I_eq,
        df_out=df_out, group_name='계수하중', Lp=Lp
    )

    return {
        'df_out': df_out, 'summary': summary,
        'table_use': table_use, 'table_fact': table_fact,
        'table_use_print': table_use_print, 'table_fact_print': table_fact_print,
        'use_results': use_results, 'use_params': use_params,
        'fact_results': fact_results, 'fact_params': fact_params,
    }


# -----------------------------------------------------------------------------
# export
# -----------------------------------------------------------------------------
__all__ = [
    "LOGGER",
    "D_solver", "D_solver2",
    "coord_general", "Col",
    "Calc", "summarize_Mmax",
    "calc_disp_from_LC", "per_pile_table_from_summary", "build_headforce_block",
    "run_Calc_per_combo", "build_all_tables",
]
