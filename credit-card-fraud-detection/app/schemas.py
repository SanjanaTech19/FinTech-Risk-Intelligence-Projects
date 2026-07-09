from pydantic import BaseModel
from typing import List

class TransactionPayload(BaseModel):
    Time: float
    Amount: float
    V_features: List[float]