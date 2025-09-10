from __future__ import annotations
from .pilelib.common import np, pd, LOGGER
import re

def coord_general(coords: Sequence[float],
                  counts: Optional[Sequence[Union[int, float]]] = None,
                  x0: float = 0.0,
                  labels: Optional[Sequence[str]] = None,
                  return_df: bool = False,
                  return_group_signed: bool = False,   # ← 추가: NE - coords 반환 여부
                  as_row: bool = True                   # ← 추가: 1×M 행 벡터 형태로 반환
                 ):
    """
    범용 중립축 계산기.

    인자
    ----
    coords : sequence of float
        그룹별 좌표값 (예: [x1, x2, x3, ...])
    counts : sequence of int, optional
        각 그룹에 속한 말뚝의 개수. None이면 각 그룹 1개로 간주.
    x0 : float, optional
        원점 오프셋(입력 coords에서 뺄 값). 기본 0.0.
    labels : sequence of str, optional
        각 그룹 라벨.
    return_df : bool, optional
        True면 (NE, coords_arr, distances, signed, groups, df, [group_signed]) 반환.
        False면 (NE, coords_arr, distances, signed, groups, [group_signed]) 반환.
    return_group_signed : bool, optional
        True면 그룹 좌표에 대한 NE - coords 도 함께 반환.
    as_row : bool, optional
        True면 group_signed 를 1×M 행 벡터로 반환(예: (1,3)). False면 1D (M,) 반환.

    반환 (기본)
    ----
    NE : float                          # 가중평균 중립축
    coords_arr : np.ndarray, shape (N,) # 모든 말뚝 좌표(반복 포함)
    distances : np.ndarray, shape (N,)  # |NE - coords_arr|
    signed : np.ndarray, shape (N,)     # NE - coords_arr  (부호 포함)
    groups : np.ndarray, shape (N,)     # 그룹 인덱스(0-based)
    [옵션] df : pandas.DataFrame
    [옵션] group_signed : np.ndarray, shape (1,M) 또는 (M,)
        그룹 좌표에 대한 NE - coords (요청사항)

    """
    coords = np.asarray(coords, dtype=float).ravel()
    if coords.size == 0:
        raise ValueError("coords는 최소한 하나 이상의 좌표를 가져야 합니다.")

    # counts 처리
    if counts is None:
        counts_arr = np.ones(coords.size, dtype=int)
    else:
        counts_arr = np.asarray(counts)
        if counts_arr.size == 1 and coords.size > 1:
            counts_arr = np.full(coords.size, int(counts_arr), dtype=int)
        else:
            if counts_arr.size != coords.size:
                raise ValueError("counts의 길이는 coords의 길이와 같아야 합니다.")
            counts_arr = counts_arr.astype(int)

    if np.any(counts_arr < 0):
        raise ValueError("counts에는 음수가 들어갈 수 없습니다.")
    N = int(counts_arr.sum())
    if N == 0:
        raise ValueError("총 말뚝 갯수(sum(counts))는 0보다 커야 합니다.")

    # labels 처리
    if labels is None:
        labels = [f"G{i+1}" for i in range(coords.size)]
    else:
        if len(labels) != coords.size:
            raise ValueError("labels의 길이는 coords의 길이와 같아야 합니다.")

    # 원점 보정
    coords0 = coords - float(x0)

    # 중립축(가중평균)
    NE = float((coords0 * counts_arr).sum() / N)

    # 전체 말뚝 좌표 전개
    coords_arr = np.repeat(coords0, counts_arr)
    groups = np.repeat(np.arange(coords.size, dtype=int), counts_arr)

    # per-pile signed/abs
    signed = NE - coords_arr
    distances = np.abs(signed)

    # 요청한 그룹 단위 signed: NE - coords
    group_signed = None
    if return_group_signed:
        group_signed = NE - coords0
        if as_row:
            group_signed = group_signed.reshape(1, -1)  # 1×M

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
        else:
            return NE, coords_arr, distances, signed, groups, df

    if return_group_signed:
        return NE, coords_arr, distances, signed, groups, group_signed
    else:
        return NE, coords_arr, distances, signed, groups





def _parse_tuple_like(s: str, typ=float):
    """'5,5,5' / '(5, 5, 5)' / ' 5  ,  5 ,5 ' 모두 허용"""
    s = s.strip().strip("()[]")
    parts = [p for p in re.split(r"[,\s]+", s) if p != ""]
    return [typ(p) for p in parts]

def input_pile_layout_xonly(x0: float = 0.0, as_row: bool = True):
    """
    1) 사용자 입력 → counts, xs 파싱
    2) coord_general(coords=xs, counts=counts, x0=...) 호출
    3) 결과(NE, group_signed)와 per-pile/detail DataFrame 반환
    """
    counts_str = input("각 x축 말뚝 개수 (예: 5,5,5) = ").strip()
    xs_str     = input("각 x좌표 (예: 0.65,1.95,4.55) = ").strip()

    counts = _parse_tuple_like(counts_str, typ=float)  # int형도 float로 들어올 수 있어 안전 처리
    counts = [int(round(c)) for c in counts]
    xs     = _parse_tuple_like(xs_str, typ=float)

    if len(counts) != len(xs):
        raise ValueError(f"개수({len(counts)})와 좌표({len(xs)}) 길이가 다릅니다.")

    labels = [f"G{i+1}" for i in range(len(xs))]

    # --- coord_general 호출 (여기서 실제 계산) ---
    # coord_general이 현재 셀에 정의되어 있다고 가정.
    # 모듈에 있을 경우: from coord import coord_general 로 대체 가능.
    NE, coords_arr, distances, signed, groups, df_detail, group_signed = coord_general(
        coords=xs,
        counts=counts,
        x0=x0,
        labels=labels,
        return_df=True,
        return_group_signed=True,
        as_row=as_row
    )

    # PILE_ID 부여 및 레이아웃 DF 생성
    df_detail = df_detail.reset_index(drop=True)
    df_detail.insert(0, "PILE_ID", np.arange(1, len(df_detail) + 1))
    layout_df = df_detail[["PILE_ID", "coord"]].rename(columns={"coord": "X"})

    # 호출 결과 패키징
    result = {
        "NE": NE,
        "counts": counts,
        "coords": xs,
        "labels": labels,
        "group_signed": group_signed,     # NE - coords (shape: (1,M) if as_row=True)
        "layout_df": layout_df,           # (PILE_ID, X)
        "detail_df": df_detail            # (PILE_ID, group_idx, group_label, X, signed, distance)
    }
    return result