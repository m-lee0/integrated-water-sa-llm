def overrides(config: dict, param: str, object: str, value: float):
    if '-to-' in object:
        # Update arc
        if object not in config["overrides"]["arcs"]:
            config["overrides"]["arcs"][object] = {
                "name": object,
                "type_": "arc",
            }
        config["overrides"]["arcs"][object][param] = value
    elif param == 'percolation_coefficient':
        # Update node surfaces
        if object not in config["overrides"]["nodes"]:
            config["overrides"]["nodes"][object] = {
                "name": object,
                "type_": "Land",
            }
        if "surfaces" not in config["overrides"]["nodes"][object]:
            config["overrides"]["nodes"][object]["surfaces"] = {}
        for s in config["nodes"][object]["surfaces"]:
            if s not in config["overrides"]["nodes"][object]["surfaces"]:
                config["overrides"]["nodes"][object]["surfaces"][s] = {}
            if config["nodes"][object]["surfaces"][s]["surface"] != "Impervious":
                config["overrides"]["nodes"][object]["surfaces"][s]["percolation_coefficient"] = value
    elif param == 'surface_coefficient':
        # Update node surfaces
        if object not in config["overrides"]["nodes"]:
            config["overrides"]["nodes"][object] = {
                "name": object,
                "type_": "Land",
            }
        if "surfaces" not in config["overrides"]["nodes"][object]:
            config["overrides"]["nodes"][object]["surfaces"] = {}
        for s in config["nodes"][object]["surfaces"]:
            if s not in config["overrides"]["nodes"][object]["surfaces"]:
                config["overrides"]["nodes"][object]["surfaces"][s] = {}
            if config["nodes"][object]["surfaces"][s]["surface"] != "Impervious":
                config["overrides"]["nodes"][object]["surfaces"][s]["surface_coefficient"] = value
    else:
        obj_type = config["nodes"][object]["type_"]
        if object not in config["overrides"]["nodes"]:
            config["overrides"]["nodes"][object] = {
                "name": object,
                "type_": obj_type,
            }
        config["overrides"]["nodes"][object][param] = value

    return config