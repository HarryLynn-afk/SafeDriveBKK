"""
Shared application state initialised once inside create_app().

Other modules import these names directly:

    from .extensions import ml_predictor, USE_ML, gmaps_client, CHATBOT_DATA

create_app() mutates these module-level variables during startup so every
module that imported them gets the live reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import googlemaps
    from .ml_model import RiskPredictor

ml_predictor: "RiskPredictor | None" = None
USE_ML: bool = False
gmaps_client: "googlemaps.Client | None" = None
CHATBOT_DATA: dict = {}
