"""
Shared application state initialised once inside create_app().

Other modules import these names directly:

    from .extensions import ml_predictor, USE_ML, CHATBOT_DATA

create_app() mutates these module-level variables during startup so every
module that imported them gets the live reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ml_model import RiskPredictor

ml_predictor: "RiskPredictor | None" = None
USE_ML: bool = False
CHATBOT_DATA: dict = {}
