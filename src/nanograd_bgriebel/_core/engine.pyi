class Value:
    @property
    def data(self) -> float: ...
    @data.setter
    def data(self, new_data: float): ...
    @property
    def grad(self) -> float: ...
    def __init__(self, data: float) -> None: ...
    def __add__(self, other: Value | float) -> Value: ...
    def __mul__(self, other: Value | float) -> Value: ...
    def __pow__(self, other: float) -> Value: ...
    def __neg__(self) -> Value: ...
    def __sub__(self, other: Value | float) -> Value: ...
    def __rsub__(self, other: Value | float): ...
    def __radd__(self, other: Value | float) -> Value: ...
    def __rmul__(self, other: Value | float) -> Value: ...
    def __truediv__(self, other: Value | float) -> Value: ...
    def __rtruediv__(self, other: Value | float) -> Value: ...
    def relu(self) -> Value: ...
    def backwards(self): ...
