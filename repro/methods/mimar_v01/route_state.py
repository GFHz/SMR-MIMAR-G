"""Recoverable route and item feedback memory."""
from dataclasses import dataclass, field

ROUTE_PENALTY_DECAY = 0.8
ROUTE_REJECTION_INCREMENT = 0.5
ITEM_COOLDOWN_STEPS = 2


@dataclass
class FeedbackMemory:
    route_penalties: dict = field(default_factory=dict)
    item_cooldowns: dict = field(default_factory=dict)

    def advance(self, route=None, rejected_item_id=None):
        self.route_penalties = {key: value * ROUTE_PENALTY_DECAY
                                for key, value in self.route_penalties.items()
                                if value * ROUTE_PENALTY_DECAY > 1e-12}
        self.item_cooldowns = {key: value - 1 for key, value in self.item_cooldowns.items()
                               if value - 1 > 0}
        if route is not None:
            self.route_penalties[tuple(route)] = self.route_penalties.get(tuple(route), 0.0) + ROUTE_REJECTION_INCREMENT
        if rejected_item_id is not None:
            self.item_cooldowns[int(rejected_item_id)] = ITEM_COOLDOWN_STEPS

    def penalty(self, route):
        return self.route_penalties.get(tuple(route), 0.0)

    def cooling_down(self, item_id):
        return self.item_cooldowns.get(int(item_id), 0) > 0
