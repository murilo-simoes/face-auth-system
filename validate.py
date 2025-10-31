def validateToxin(toxin_data: dict) -> bool:
    for k in toxin_data:
        if not toxin_data[k]:
            return False

    return True
