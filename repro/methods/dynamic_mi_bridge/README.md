# Dynamic MI-Bridge v0.1

This experimental module is isolated from frozen Static MI-Bridge v1. It keeps an exact Last-20 accepted-interaction window. At every step it recomputes genre frequencies, Need-based bridge scores, the frozen deterministic bridge ranking, direct-first feasibility, fallback, and catalog candidates. A caller supplies a one-item generation callback.

Every valid non-target item selected from the current bridge candidates is treated as accepted offline, appended on the right, and replaces the oldest window item. A target ends planning without entering the window. Invalid output, unavailable bridges, or eight accepted intermediates stop planning. There is no feedback simulation, time decay, IoR reward, competitor scoring, or evaluator change.
