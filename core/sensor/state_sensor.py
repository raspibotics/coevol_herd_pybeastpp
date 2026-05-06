import numpy as np

from core.world.world_object import WorldObject
from core.sensor.base import Sensor, MatchFunction, EvaluateFunction, ScaleFunction
from core.sensor.function.evaluate import EvaluateFunction
from core.utils import Vec2

# custom sensor, distinct from self sensor
# used to allow agent variables to be toggled as true false (mapped to 1 or 0 )
# by adding them as sensor can then be used by brain as a "state"
# Allows for implementation of recurrent elements without major structural overhauls

class StateSensor(Sensor):
    def initialise(self) -> None:
        if self.owner:
            self.state = 0.0
        super().__init__()

    # seperate commmand to update state, not linked to other sensors directly, must be manually implemented by user
    def updateState(self, newState) -> None:
        self.state = newState
    
    # Do not update or pass autonomously
    def update(self) -> None:
        pass

    def interact(self, other: WorldObject) -> None:
        pass
    
    # Pnly returns state, acts as steady singmal without updateState configured.
    def output(self) -> float:
        return self.state



