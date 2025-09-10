import sys
API_ROOT = r"C:\Users\USER\Desktop\Workspace\Pile\Api"  # 본인 경로, 타 사용자가 사용시, 경로 복사로 경로를 설정해주십시오.
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)
import numpy as np
import function as fn
import pandas as pd
from export_to_excel_sheet import export_bundle_to_single_sheet
from export_to_excel_sheet import export_bundle_to_template_auto
# 이제 ‘폴더 전체 export’를 한 번에
from function import *

# -----------------------------
# 작은 유틸: β <-> Kh 변환
# -----------------------------
def _beta_from_Kh(Kh: float, D1: float, E: float, I: float) -> float:
    """Kh -> B(β)  ;  1/B = ((Kh*D1)/(4EI))**0.25  =>  B = ((Kh*D1)/(4EI))**0.25"""
    return ((Kh * D1) / (4.0 * E * I)) ** 0.25

def _Kh_from_beta(B: float, D1: float, E: float, I: float) -> float:
    """B(β) -> Kh  ;  Kh = (4EI/D1) * (1/B**4)"""
    if B == 0:
        return np.inf
    return (4.0 * E * I / D1) * (1.0 / (B ** 4))


# ---------------------------------------------------------
# 메인: 그룹 좌표/개수 입력 → 전체 계산(고정/힌지) + 오버라이드
# ---------------------------------------------------------
def compute_pile_data_groups(
    xlsx_path: str,
    *,
    pile_name: str,
    alpha: float,
    alpha_eq: float,
    group_xs: list[float],
    counts:   list[int],
    x0: float = 0.0,
    include_hinge: bool = True,
    # === 오버라이드 옵션 ===
    a_override:  float | None = None,   # a 수동
    Kv_override: float | None = None,   # Kv 수동(있으면 a는 메타용)
    Eo_override: float | None = None,   # Eo 수동
    Kh_override: float | None = None,   # 상시 Kh 수동
    Kh_eq_override: float | None = None,# 지진 Kh_e 수동(없으면 2*Kh_override)
    B_override: float | None = None,    # 상시 B(β) 수동
    B_eq_override: float | None = None, # 지진 B_e 수동
):
    """
    그룹 좌표/본수를 받아 coord_general로 전개하고,
    (머리 고정/힌지) 해석, summary/per-pile/머리력/Calc 및 반복 수렴 이력까지 반환.

    오버라이드 우선순위:
      - Kv: Kv_override > (a_override or Col) * Ap*E/Lp
      - Eo: Eo_override > N_head*700
      - 상시:   B_override > Kh_override > 반복해(D_solver)
      - 지진:   B_eq_override > Kh_eq_override > (2*Kh_override) > 반복해(D_solver2)
    """
    # ---------------- 1) 입력/엑셀 ----------------
    group_xs = np.asarray(group_xs, dtype=float).ravel()
    counts   = np.asarray(counts,   dtype=int).ravel()
    if group_xs.size == 0:
        raise ValueError("group_xs 가 비었습니다.")
    if group_xs.size != counts.size:
        raise ValueError("group_xs와 counts 길이가 다릅니다.")

    df1 = extract_df1_design_only(xlsx_path)
    df2 = extract_df2_loads_exact(xlsx_path)

    # ---------------- 2) coord_general (전개/보정) ----------------
    NE, coords_arr, distances, signed, groups, detail_df, group_signed = coord_general(
        group_xs, counts=counts, x0=x0, return_df=True, return_group_signed=True, as_row=True
    )
    layout_df = pd.DataFrame({"PILE_ID": np.arange(1, coords_arr.size+1), "X": coords_arr})
    Co_ord = np.asarray(group_signed, dtype=float).ravel()  # (M,)

    # ---------------- 3) df1 → 기하/재료 ----------------
    s   = df1.set_index("name")["value"]
    D   = float(s.at["직경(mm)"])
    t   = float(s.at["두께(mm)"])
    c   = float(s.at["부식두께(mm)"])
    Lp  = float(s.at["길이(m)"])
    E   = float(s.at["E(MPa)"]) * 1000.0
    D2  = (D - 2*c)/1000.0
    D1  = (D - 2*t)/1000.0
    I   = (D2**4 - D1**4) * np.pi / 64.0
    Ap  = np.pi/4.0 * (D2**2 - D1**2)
    N_head = float(s.at["N_head"])

    # ---------------- 4) 공법계수 a, Kv ----------------
    a_auto = Col(pile_name, Lp=Lp, D2=D2)
    a_used = float(a_override) if (a_override is not None) else float(a_auto)
    if Kv_override is not None:
        Kv_used = float(Kv_override)
    else:
        Kv_used = a_used * Ap * E / Lp
    a  = float(a_used)
    Kv = float(Kv_used)

    # ---------------- 5) β, KH 결정(오버라이드 우선) ----------------
    # Eo
    Eo_auto = N_head * 700.0
    Eo      = float(Eo_override) if (Eo_override is not None) else float(Eo_auto)

    # 상시
    B_used  = None
    Kh_used = None
    if B_override is not None:
        B_used  = float(B_override)
        Kh_used = _Kh_from_beta(B_used, D1=D2, E=E, I=I)
        B1 = np.array([1.0 / B_used]); KH1 = np.array([Kh_used])
    elif Kh_override is not None:
        Kh_used = float(Kh_override)
        B_used  = _beta_from_Kh(Kh_used, D1=D2, E=E, I=I)
        B1 = np.array([1.0 / B_used]); KH1 = np.array([Kh_used])
    else:
        K_H0     = (1/0.3) * alpha * Eo
        B1, KH1  = D_solver(D2, K_H0, D2, E, I)
        B_used   = 1.0 / B1[-1]
        Kh_used  = KH1[-1]

    # 지진
    B_eq_used  = None
    Kh_eq_used = None
    if B_eq_override is not None:
        B_eq_used  = float(B_eq_override)
        Kh_eq_used = _Kh_from_beta(B_eq_used, D1=D2, E=E, I=I)
        B2 = np.array([1.0 / B_eq_used]); KH2 = np.array([Kh_eq_used])
    elif Kh_eq_override is not None:
        Kh_eq_used = float(Kh_eq_override)
        B_eq_used  = _beta_from_Kh(Kh_eq_used, D1=D2, E=E, I=I)
        B2 = np.array([1.0 / B_eq_used]); KH2 = np.array([Kh_eq_used])
    elif Kh_override is not None:
        Kh_eq_used = 2.0 * float(Kh_override)
        B_eq_used  = _beta_from_Kh(Kh_eq_used, D1=D2, E=E, I=I)
        B2 = np.array([1.0 / B_eq_used]); KH2 = np.array([Kh_eq_used])
    else:
        K_H01     = (1/0.3) * alpha_eq * Eo
        B2, KH2   = D_solver2(D2, K_H01, D2, E, I)
        B_eq_used = 1.0 / B2[-1]
        Kh_eq_used= KH2[-1]

    # 반복 이력 프레임
    df_attribution = pd.concat([
        pd.Series(B1,  name='상시 1/β(m)'),
        pd.Series(KH1, name='상시 KH(kN/㎡)'),
        pd.Series(B2,  name='지진 1/β(m)'),
        pd.Series(KH2, name='지진 KH(kN/㎡)'),
    ], axis=1)

    # 최종 사용 β/Kh
    B   = float(B_used)
    B_e = float(B_eq_used)
    Kh  = float(Kh_used)
    Kh_e= float(Kh_eq_used)

    BL  = B   * Lp
    BeL = B_e * Lp

    # 말뚝 특성치 표(공통)
    df_spring = pd.DataFrame({
        '':   ['Ap','a','Kv','I'],
        '값': [Ap, a,  Kv,  I],
        '단위':['m²','-','kN/m','m⁴'],
    })

    # ---------------- 6) 스프링/조합행렬 (머리 고정) ----------------
    theta_p = np.zeros(coords_arr.size)
    cos_p = np.cos(theta_p); sin_p = np.sin(theta_p)
    x1 = signed; x2 = x1**2

    # 고정 머리 스프링 (상시/지진)
    K1_fx = 4*E*I*B**3;   K2_fx = 2*E*I*B**2;   K3_fx = 2*E*I*B**2;   K4_fx = 2*E*I*B
    K1_fx_e = 4*E*I*B_e**3; K2_fx_e = 2*E*I*B_e**2; K3_fx_e = 2*E*I*B_e**2; K4_fx_e = 2*E*I*B_e

    # 조합행렬(상시)
    Axx = K1_fx*cos_p**2 + Kv*sin_p**2
    Axy = (Kv-K1_fx)*sin_p*cos_p
    Axa = (Kv-K1_fx)*x1*sin_p*cos_p - K2_fx*cos_p
    Ayy = Kv*cos_p**2 + K1_fx*sin_p**2
    Aya = (Kv*cos_p**2 + K1_fx*sin_p**2)*x1 + K2_fx*sin_p
    Aaa = (Kv*cos_p**2 + K1_fx*sin_p**2)*x2 + (K2_fx+K3_fx)*x1*sin_p + K4_fx
    K_Mat_fix  = np.array([[Axx.sum(), Axy.sum(), Axa.sum()],
                           [Axy.sum(), Ayy.sum(), Aya.sum()],
                           [Axa.sum(), Aya.sum(), Aaa.sum()]])
    IK_Mat_fix = np.linalg.inv(K_Mat_fix)

    # 조합행렬(지진)
    Axx_e = K1_fx_e*cos_p**2 + Kv*sin_p**2
    Axy_e = (Kv-K1_fx_e)*sin_p*cos_p
    Axa_e = (Kv-K1_fx_e)*x1*sin_p*cos_p - K2_fx_e*cos_p
    Ayy_e = Kv*cos_p**2 + K1_fx_e*sin_p**2
    Aya_e = (Kv*cos_p**2 + K1_fx_e*sin_p**2)*x1 + K2_fx_e*sin_p
    Aaa_e = (Kv*cos_p**2 + K1_fx_e*sin_p**2)*x2 + (K2_fx_e+K3_fx_e)*x1*sin_p + K4_fx_e
    K_Mat_fix_e  = np.array([[Axx_e.sum(), Axy_e.sum(), Axa_e.sum()],
                             [Axy_e.sum(), Ayy_e.sum(), Aya_e.sum()],
                             [Axa_e.sum(), Aya_e.sum(), Aaa_e.sum()]])
    IK_Mat_fix_e = np.linalg.inv(K_Mat_fix_e)

    Ks_fix    = {'Kv':Kv, 'K1':K1_fx,   'K2':K2_fx,   'K3':K3_fx,   'K4':K4_fx}
    Ks_fix_eq = {'Kv':Kv, 'K1':K1_fx_e, 'K2':K2_fx_e, 'K3':K3_fx_e, 'K4':K4_fx_e}

    # ---------------- 7) 스프링/조합행렬 (머리 힌지: 옵션) ----------------
    head_hinge_payload = None
    if include_hinge:
        K1_hg = 2*E*I*B**3; K2_hg = 0.0; K3_hg = 0.0; K4_hg = 0.0
        Axx2 = K1_hg*cos_p**2 + Kv*sin_p**2
        Axy2 = (Kv-K1_hg)*sin_p*cos_p
        Axa2 = (Kv-K1_hg)*x1*sin_p*cos_p - K2_hg*cos_p
        Ayy2 = Kv*cos_p**2 + K1_hg*sin_p**2
        Aya2 = (Kv*cos_p**2 + K1_hg*sin_p**2)*x1 + K2_hg*sin_p
        Aaa2 = (Kv*cos_p**2 + K1_hg*sin_p**2)*x2 + (K2_hg+K3_hg)*x1*sin_p + K4_hg
        K_Mat_hinge  = np.array([[Axx2.sum(), Axy2.sum(), Axa2.sum()],
                                 [Axy2.sum(), Ayy2.sum(), Aya2.sum()],
                                 [Axa2.sum(), Aya2.sum(), Aaa2.sum()]])
        IK_Mat_hinge = np.linalg.inv(K_Mat_hinge)

        K1_hg_e = 2*E*I*B_e**3; K2_hg_e = 0.0; K3_hg_e = 0.0; K4_hg_e = 0.0
        Axx_e2 = K1_hg_e*cos_p**2 + Kv*sin_p**2
        Axy_e2 = (Kv-K1_hg_e)*sin_p*cos_p
        Axa_e2 = (Kv-K1_hg_e)*x1*sin_p*cos_p - K2_hg_e*cos_p
        Ayy_e2 = Kv*cos_p**2 + K1_hg_e*sin_p**2
        Aya_e2 = (Kv*cos_p**2 + K1_hg_e*sin_p**2)*x1 + K2_hg_e*sin_p
        Aaa_e2 = (Kv*cos_p**2 + K1_hg_e*sin_p**2)*x2 + (K2_hg_e+K3_hg_e)*x1*sin_p + K4_hg_e
        K_Mat_hinge_e  = np.array([[Axx_e2.sum(), Axy_e2.sum(), Axa_e2.sum()],
                                   [Axy_e2.sum(), Ayy_e2.sum(), Aya_e2.sum()],
                                   [Axa_e2.sum(), Aya_e2.sum(), Aaa_e2.sum()]])
        IK_Mat_hinge_e = np.linalg.inv(K_Mat_hinge_e)

        Ks_hinge    = {'Kv':Kv, 'K1':K1_hg,   'K2':K2_hg,   'K3':K3_hg,   'K4':K4_hg}
        Ks_hinge_eq = {'Kv':Kv, 'K1':K1_hg_e, 'K2':K2_hg_e, 'K3':K3_hg_e, 'K4':K4_hg_e}
        head_hinge_payload = {
            "Ks": {"normal": Ks_hinge, "eq": Ks_hinge_eq},
            "K_mats": {"K": K_Mat_hinge, "IK": IK_Mat_hinge,
                       "Ke": K_Mat_hinge_e, "IKe": IK_Mat_hinge_e}
        }

    # ---------------- 8) 전체 테이블 빌드(머리 고정/힌지) ----------------
    all_out_fix = build_all_tables(
        df2, IK_Mat_fix, IK_Mat_fix_e,
        Ks_fix, Ks_fix_eq,
        Co=Co_ord, counts=counts,
        theta_by_pos=None,
        B=B, E=E, I=I,
        B_eq=B_e, E_eq=E, I_eq=I,
        Lp=Lp
    )
    all_out_hinge = None
    if include_hinge:
        all_out_hinge = build_all_tables(
            df2, IK_Mat_hinge, IK_Mat_hinge_e,
            Ks_hinge, Ks_hinge_eq,
            Co=Co_ord, counts=counts,
            theta_by_pos=None,
            B=B, E=E, I=I,
            B_eq=B_e, E_eq=E, I_eq=I,
            Lp=Lp
        )

    # ---------------- 9) 반환 ----------------
    meta = {
        "XLSX_PATH": xlsx_path,
        "pile_name": pile_name,
        "alpha": alpha, "alpha_eq": alpha_eq,
        "Lp": Lp, "E": E, "I": I,
        "D1": D1, "D2": D2, "Ap": Ap,
        "a": a, "Kv": Kv,
        "a_auto": a_auto, "a_override": a_override,
        "Kv_override": Kv_override,
        "Eo": Eo, "Eo_auto": Eo_auto, "Eo_override": Eo_override,
        "Kh": Kh, "Kh_e": Kh_e,
        "Kh_override": Kh_override, "Kh_eq_override": Kh_eq_override,
        "B": B, "B_e": B_e, "B_override": B_override, "B_eq_override": B_eq_override,
        "BL": BL, "BeL": BeL,
        "is_semi_infinite_fix":  bool(BL  >= 3.0),
        "is_semi_infinite_eq":   bool(BeL >= 3.0),
        "group_xs": group_xs.tolist(),
        "counts": counts.tolist(),
        "NE": NE,
        "x0": x0,
    }
    frames = {
        "df1": df1, "df2": df2,
        "layout_df": layout_df,
        "detail_df": detail_df,
        "spring_common": df_spring,
        "df_attribution": df_attribution
    }

    groups = {
        "고정": {
            "all_out": all_out_fix,
            "Ks": {"normal": Ks_fix, "eq": Ks_fix_eq},
            "K_mats": {"K": K_Mat_fix, "IK": IK_Mat_fix, "Ke": K_Mat_fix_e, "IKe": IK_Mat_fix_e},
        }
    }
    if include_hinge and all_out_hinge is not None:
        groups["힌지"] = {
            "all_out": all_out_hinge,
            "Ks": {"normal": Ks_hinge, "eq": Ks_hinge_eq},
            "K_mats": {"K": K_Mat_hinge, "IK": IK_Mat_hinge, "Ke": K_Mat_hinge_e, "IKe": IK_Mat_hinge_e},
        }

    # 반복 이력 마지막 B 추출(가드)
    try:
        B_last  = float(1.0 / df_attribution['상시 1/β(m)'].dropna().iloc[-1])
    except Exception:
        B_last = B
    try:
        Be_last = float(1.0 / df_attribution['지진 1/β(m)'].dropna().iloc[-1])
    except Exception:
        Be_last = B_e

    return {
        "meta": meta,
        "frames": frames,
        "head_fix": {
            "Ks": {"normal": Ks_fix, "eq": Ks_fix_eq},
            "K_mats": {"K": K_Mat_fix, "IK": IK_Mat_fix, "Ke": K_Mat_fix_e, "IKe": IK_Mat_fix_e},
            "all_out": all_out_fix
        },
        "head_hinge": (
            {
                "Ks": {"normal": Ks_hinge, "eq": Ks_hinge_eq},
                "K_mats": {"K": K_Mat_hinge, "IK": IK_Mat_hinge, "Ke": K_Mat_hinge_e, "IKe": IK_Mat_hinge_e},
                "all_out": all_out_hinge
            } if include_hinge else None
        ),
        "groups": groups,
        "iterations": {
            "df_attribution": df_attribution,
            "B_last":  B_last,
            "Be_last": Be_last,
        }
    }


# ---------------------------------------------------------
# CSV 문자열 입력 헬퍼 (UI에서 편하게)
# ---------------------------------------------------------
def compute_pile_data_from_strings(
    xlsx_path: str,
    *,
    pile_name: str,
    alpha: float,
    alpha_eq: float,
    counts_csv: str,     # "5,5,5"
    xs_csv: str,         # "0.65,1.95,4.55"
    x0: float = 0.0,
    include_hinge: bool = True,
    # 오버라이드(선택)
    a_override: float | None = None,
    Kv_override: float | None = None,
    Eo_override: float | None = None,
    Kh_override: float | None = None,
    Kh_eq_override: float | None = None,
    B_override: float | None = None,
    B_eq_override: float | None = None,
):
    counts = [int(s.strip()) for s in counts_csv.split(",") if s.strip()]
    group_xs = [float(s.strip()) for s in xs_csv.split(",") if s.strip()]
    return compute_pile_data_groups(
        xlsx_path=xlsx_path,
        pile_name=pile_name,
        alpha=alpha, alpha_eq=alpha_eq,
        group_xs=group_xs, counts=counts,
        x0=x0, include_hinge=include_hinge,
        a_override=a_override, Kv_override=Kv_override, Eo_override=Eo_override,
        Kh_override=Kh_override, Kh_eq_override=Kh_eq_override,
        B_override=B_override, B_eq_override=B_eq_override
    )


#입력
bundle = compute_pile_data_groups(
    xlsx_path= r"C:\Users\USER\Desktop\말뚝 계산 Workbook_Lite_Version.xlsx",
    pile_name="내부굴착강관말뚝",
    alpha=1.0,
    alpha_eq=2.0,
    group_xs=[0.65,1.95,4.55],
    counts=[5,5,5],
    include_hinge=True,
   # === 오버라이드 옵션 === 아래 None을 지우고 사용자가 직접 입력 시 아래 값으로 연산이 진행됨
    a_override= None,   # a 수동, 공법에 따라 시방서와 다른 값을 사용시
    Kv_override = None,   # Kv 수동(있으면 a는 메타용), 쏘일 시멘트에 의해 다른 식을 써야 할때
    Eo_override = None,   # Eo 수동, 700*N에 의해 다른 값을 써야 할때,
    Kh_override = None,   # 상시 Kh 수동, 지반 값이 정해져 있을 경우
    Kh_eq_override = None,# 지진 Kh_e 수동(없으면 2*Kh_override)
    B_override = None,    # 상시 B(β) 수동, 실험식이 존재하여 다르게 계산해야 할 경우
    B_eq_override = None, # 지진 B_e 수동, 실험식이 존재하여 다르게 계산해야 할 경우
)
#출력
out = export_bundle_to_new_sheet_as_tables(
    bundle,
    template_path=r"C:\Users\USER\Desktop\Workspace\Pile\Api\엑셀 파일 저장함\말뚝 계산 Workbook.xlsx",
    save_path=r"C:\Users\USER\Desktop\Workspace\Pile\Api\엑셀 파일 저장함\말뚝 계산out.xlsx",
   sheet_name="계산결과(자동-표)",
    include_calc=True,
    include_fix=True,
    include_hinge=True,
    add_summaries=True,               # ✅ 스프링정수, 최대변위 요약 추가
    sanitize=True,                    # ✅ 손상 경고 줄이기
    keep_values_for_formulas=False    # 필요시 True
)
print("저장:", out)




