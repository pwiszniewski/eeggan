import configparser
import os


def read_config():
    computer_name = os.environ.get("COMPUTERNAME")
    print(computer_name)
    if computer_name == "DESKTOP-SAI3PJD":
        config_path = 'H:\\Documents\\1_dokt\\other_programs\\eeggan\\settings_laptop.ini'
    elif computer_name == 'DESKTOP-CM8AHTB':
        config_path = 'C:\\Users\przem\\OneDrive - Politechnika Warszawska\Dokumenty\\1_dokt\\other_programs_PC\\eeggan\\settings_PC.ini'
    else:
        raise Exception(f"Computer name not found {computer_name}")

    config = configparser.ConfigParser()
    config.read(config_path)
    return config
