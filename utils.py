import configparser
import os


def read_config():
    computer_name = os.environ.get("COMPUTERNAME")
    print(computer_name)
    if computer_name == "DESKTOP-SAI3PJD":
        config_path = 'H:\\Documents\\1_dokt\\other_programs\\eeggan\\settings_laptop.ini'
    else:
        raise Exception("Computer name not found")

    config = configparser.ConfigParser()
    config.read(config_path)
    return config
