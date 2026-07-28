# typings/tilelang/__init__.pyi

class Fragment:
    """Represents a memory fragment in TileLang."""

    def __init__(self) -> None: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> str: ...
    def __add__(self, other: Fragment) -> Fragment: ...
    def __mul__(self, other: Fragment | int) -> Fragment: ...
    # 可根据实际 API 扩展更多方法

class T:
    """TileLang tensor/fragment factory."""

    @staticmethod
    def alloc_fragment(size: int | tuple[int, ...], name: str = "", dtype: str = "float32", layout: str = "dense") -> Fragment: ...

    # 如果还有其他方法，也可以加上，例如：
    # @staticmethod
    # def zeros(shape: Tuple[int, ...]) -> Fragment: ...

# 其他可能的顶层函数或类（按需添加）
