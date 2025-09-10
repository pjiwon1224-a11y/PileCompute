# function/__init__.py
from __future__ import annotations
import importlib, pkgutil, inspect

# 이 패키지(=function) 아래의 모든 모듈을 스캔해서
# 함수/클래스/상수 등을 패키지 네임스페이스로 올려둔다.
__all__: list[str] = []

for m in pkgutil.iter_modules(__path__):
    name = m.name
    if name.startswith("_"):        # _로 시작하는 파일은 건너뜀
        continue
    mod = importlib.import_module(f".{name}", __name__)

    for attr, obj in inspect.getmembers(mod):
        # _ 로 시작하는 내부심볼은 제외
        if attr.startswith("_"):
            continue
        # 내보낼 대상을 골라서 올려준다(함수/클래스/상수 등)
        if inspect.isfunction(obj) or inspect.isclass(obj) or not inspect.ismodule(obj):
            globals()[attr] = obj
            __all__.append(attr)

# 편의: 전체 모듈 다시 읽어오는 함수(노트북에서 개발 중 쓸 때)
def _reload_all():
    """개발 중 변경된 function.* 모듈을 전부 리로드"""
    for m in list(pkgutil.iter_modules(__path__)):
        importlib.reload(importlib.import_module(f".{m.name}", __name__))
