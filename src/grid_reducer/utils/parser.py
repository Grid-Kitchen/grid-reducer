def get_number_of_phases_from_bus(bus: str) -> int:
    if "." not in bus:
        return 3
    bus_splits = bus.split(".")[1:]
    if "0" in bus_splits:
        bus_splits.remove("0")
    return len(bus_splits)
