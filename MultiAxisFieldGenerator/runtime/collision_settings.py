"""Pure-Python collision activation policy shared by CLI and runtime."""
def collision_enabled(base_weight=10000., epochs=100, schedule=None):
    stages = []
    for entry in schedule or []:
        start, weight = entry.split(':') if isinstance(entry, str) else entry
        stages.append((int(start), float(weight)))
    weight = float(base_weight)
    if stages and stages[0][0] <= 1:
        weight = stages[0][1]
    return weight > 0 or any(1 < start <= max(1, int(epochs)) and value > 0 for start, value in stages)
