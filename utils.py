import configparser
import os
import datetime


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


def convert_to_base60(x):
    if x == 0:
        return "Γ"
    digits = []
    while x > 0:
        digits.append(x % 60)
        x //= 60
    digits.reverse()
    signs = "ΓΔΘΞΦΨΩΠABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return "".join(signs[digit] for digit in digits)

def get_experiment_prefix():
    now = datetime.datetime.now()
    current_time_string = f"{now.year % 10:01d}" \
                        f"{now.month:02d}" \
                        f"{now.day:02d}_" \
                        f"{now.hour:02d}" \
                        f"{now.minute:02d}" \
                        f"{convert_to_base60(now.second)}"
    return current_time_string