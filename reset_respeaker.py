import subprocess


def find_repspeaker_lite_id():
    # Run lsusb and get the output
    output = subprocess.check_output(['lsusb'], text=True)

    # Search for lines containing 'ReSpeaker Lite'
    for line in output.split('\n'):
        if "ReSpeaker Lite" in line:
            parts = line.split()
            # Format is: Bus xxx Device xxx: ID xxxx:xxxx Description
            bus = parts[1]
            device = parts[3].replace(':', '')
            device_id = None
            for i, part in enumerate(parts):
                if part == 'ID' and i+1 < len(parts):
                    device_id = parts[i+1]
            return bus, device, device_id
    return None, None, None

def run_usbreset(device_id):
    subprocess.run(['sudo', 'usbreset', device_id], check=True)

def reset_respeaker_lite():
    bus, device, device_id = find_repspeaker_lite_id()
    if device_id:
        print(f"ReSpeaker Lite found with ID: {device_id} on bus {bus} device {device}")
        run_usbreset(device_id)
    else:
        print("ReSpeaker Lite not found.")

if __name__ == "__main__":
    reset_respeaker_lite()
