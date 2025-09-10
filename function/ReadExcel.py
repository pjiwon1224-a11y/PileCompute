from .pilelib.common import np, pd, LOGGER   # ← 이렇게 'pd'를 실제로 가져오세요
import re

    
#Read the Excell
def _cell_str(x):
    if x is None: return ""
    if isinstance(x, float) and pd.isna(x): return ""
    return str(x)

def _norm_key(s: str) -> str:
    """헤더 매칭용 정규화: 공백/특수문자/단위 제거, 소문자."""
    s = _cell_str(s).strip().lower()
    s = s.replace("\u00a0","").replace(" ","").replace("(","").replace(")","").replace(".","").replace(",","").replace("_","")
    s = s.replace("k n","kn").replace("k.n","kn")
    return s

def last_numeric_in_cells(cells) -> float | None:
    """셀 묶음에서 파싱 가능한 '마지막' 숫자 반환 (문자열 내 숫자 추출 지원)."""
    vals = []
    for v in cells:
        if isinstance(v, (int, float)) and not pd.isna(v):
            vals.append(float(v))
        elif isinstance(v, str):
            for c in re.findall(r"[-+]?\d+(?:[.,]\d+)?", v.replace(",", "")):
                try: vals.append(float(c))
                except: pass
    return vals[-1] if vals else None

# ---------- df1: 설계조건(요구 항목만) ----------
def _find_row_index(df: pd.DataFrame, must_contain: list[str]) -> int | None:
    low = df.applymap(lambda x: _cell_str(x).strip().lower())
    for i in range(len(low)):
        if all(tok in " ".join(low.iloc[i].tolist()) for tok in must_contain):
            return i
    return None

def _find_dim_header(df: pd.DataFrame):
    """직경/두께/부식두께/길이 헤더행과 각 컬럼 인덱스 맵"""
    low = df.applymap(_cell_str)
    header_idx, col_map = None, {}
    for i in range(len(low)):
        row = " ".join(low.iloc[i].astype(str).tolist())
        if ("직경" in row) and ("두께" in row) and (("부식" in row) or ("부식두께" in row)) and ("길이" in row):
            header_idx = i
            keys = {"직경":"diameter_mm", "두께":"thickness_mm", "부식":"corrosion_thickness_mm", "길이":"length_m"}
            for j, val in enumerate(low.iloc[i].tolist()):
                sval = _cell_str(val)
                for k, outkey in keys.items():
                    if k in sval and outkey not in col_map:
                        col_map[outkey] = j
            break
    return header_idx, col_map

def extract_df1_design_only(xlsx_path: str) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
    best = None
    for sh in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sh, header=None, dtype=object)
        if df.empty: 
            continue

        # 1) 가로형 제원 (헤더 다음 행)
        vals = {}
        hdr_idx, col_map = _find_dim_header(df)
        if hdr_idx is not None and col_map and hdr_idx + 1 < len(df):
            val_row = df.iloc[hdr_idx + 1]
            for key, j in col_map.items():
                vals[key] = last_numeric_in_cells([val_row.iloc[j]])

        # 2) 개별 키워드 행 스캔
        idx_E  = _find_row_index(df, ["탄성계수"])
        idx_I  = _find_row_index(df, ["단면2차모멘트"])
        idx_Nh = _find_row_index(df, ["머리", "평균", "n"])
        idx_Nt = _find_row_index(df, ["선단", "평균", "n"])
        idx_Df = _find_row_index(df, ["근입깊이"])
        idx_h  = _find_row_index(df, ["돌출된", "길이"])

        if idx_E  is not None: vals["E_MPa"]  = last_numeric_in_cells(df.iloc[idx_E].tolist())
        if idx_I  is not None: vals["I_m4"]   = last_numeric_in_cells(df.iloc[idx_I].tolist())
        if idx_Nh is not None: vals["N_head"] = last_numeric_in_cells(df.iloc[idx_Nh].tolist())
        if idx_Nt is not None: vals["N_tip"]  = last_numeric_in_cells(df.iloc[idx_Nt].tolist())
        if idx_Df is not None: vals["Df_m"]   = last_numeric_in_cells(df.iloc[idx_Df].tolist())
        if idx_h  is not None: vals["h_m"]    = last_numeric_in_cells(df.iloc[idx_h].tolist())

        if sum(v is not None for v in vals.values()) >= 5:
            best = vals
            break

    if not best:
        raise RuntimeError("설계조건(df1) 영역을 찾지 못했습니다.")

    ordered = [
        ("diameter_mm", "직경(mm)"),
        ("thickness_mm", "두께(mm)"),
        ("corrosion_thickness_mm", "부식두께(mm)"),
        ("length_m", "길이(m)"),
        ("E_MPa", "E(MPa)"),
        ("I_m4", "I(m^4)"),
        ("N_head", "N_head"),
        ("N_tip", "N_tip"),
        ("Df_m", "Df(m)"),
        ("h_m", "h(m)"),
    ]
    rows = [{"name": label, "value": float(best[key])}
            for key, label in ordered if best.get(key) is not None]
    return pd.DataFrame(rows, columns=["name","value"]).reset_index(drop=True)

# ---------- df2: 하중조합(사용하중/계수하중 + COMBO n만) ----------
_combo_pat = re.compile(r"(?i)^\s*combo\s*(\d+)\s*$")

def _find_combo_in_row(row_vals) -> str | None:
    for v in row_vals:
        s = _cell_str(v)
        m = _combo_pat.match(s)
        if m: return f"COMBO {int(m.group(1))}"
    return None

def _to_float_series(col: pd.Series) -> pd.Series:
    return pd.to_numeric(
        col.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )

def extract_df2_loads_exact(xlsx_path: str) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
    best_df, best_score = None, -1

    for sh in xl.sheet_names:
        raw = pd.read_excel(xl, sheet_name=sh, header=None, dtype=object)
        if raw.empty: 
            continue

        # 헤더 탐색: '구분'+Vo/Ho/Mo(+비고)
        for i in range(len(raw)):
            keys = [_norm_key(x) for x in raw.iloc[i].tolist()]
            col_group = col_vo = col_ho = col_mo = col_rem = None
            for j, k in enumerate(keys):
                if ("구분" in k) or (k == "구분"): col_group = j if col_group is None else col_group
                if "vo" in k: col_vo = j
                if "ho" in k: col_ho = j
                if ("mo" in k) and ("mz" not in k): col_mo = j
                if ("비고" in k) or ("remark" in k): col_rem = j
            if not all(c is not None for c in [col_group, col_vo, col_ho, col_mo]):
                continue

            body = raw.iloc[i+1:].copy()
            if body.empty: 
                continue

            # 조합은 'COMBO n'을 행 전체에서 직접 탐지
            combo_ser = body.apply(lambda r: _find_combo_in_row(r.tolist()), axis=1)

            df = pd.DataFrame({
                "구 분":   body.iloc[:, col_group],
                "조합":    combo_ser,
                "Vo(kN)":  body.iloc[:, col_vo],
                "Ho(kN)":  body.iloc[:, col_ho],
                "Mo(kN.m)":body.iloc[:, col_mo],
                "비 고":   body.iloc[:, col_rem] if col_rem is not None else pd.Series([np.nan]*len(body)),
            })

            # 병합셀 ffill + 숫자화
            df["구 분"] = df["구 분"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan}).ffill()
            for c in ["Vo(kN)","Ho(kN)","Mo(kN.m)"]:
                df[c] = _to_float_series(df[c])

            # 핵심 필터:
            # A) 조합은 COMBO n
            mask_combo = df["조합"].notna()
            # B) 구 분은 '사용하중' 또는 '계수하중'
            grp_norm = df["구 분"].astype(str).str.replace(r"\s+", "", regex=True)
            mask_group = grp_norm.isin({"사용하중","계수하중"})
            # C) 수치열 중 하나라도 값 존재
            mask_numeric = df[["Vo(kN)","Ho(kN)","Mo(kN.m)"]].notna().any(axis=1)

            df = df[mask_combo & mask_group & mask_numeric].reset_index(drop=True)
            if df.empty: 
                continue

            score = len(df)
            if score > best_score:
                best_df, best_score = df, score

    if best_df is None or best_df.empty:
        raise RuntimeError("하중조합(df2) 표를 찾지 못했습니다.")

    # 보기용 반올림(원 값 유지 원하면 주석)
    best_df["Vo(kN)"] = best_df["Vo(kN)"].round(3)
    best_df["Ho(kN)"] = best_df["Ho(kN)"].round(3)
    best_df["Mo(kN.m)"] = best_df["Mo(kN.m)"].round(3)

    return best_df[["구 분","조합","Vo(kN)","Ho(kN)","Mo(kN.m)","비 고"]]

