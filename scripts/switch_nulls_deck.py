"""Select an existing preset on the calibrated Chinese Nulls deck page."""
import argparse
from io import BytesIO
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
from cr_rl.client.runtime import capture_screen, ensure_transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('slot', type=int, choices=(1, 2, 3))
    parser.add_argument('--adb', default='G:/sdk/platform-tools/adb.exe')
    parser.add_argument('--serial', default='127.0.0.1:16416')
    args = parser.parse_args()
    ensure_transport(args.serial,adb=args.adb)
    command = [args.adb, '-s', args.serial]
    root = Path(__file__).resolve().parents[1]
    template = np.asarray(Image.open(root/'configs/nulls_ui_900x1600/deck_tab.png').convert('RGB'), dtype=np.int16)
    centers = (338, 448, 558)

    def selected():
        frame = Image.open(BytesIO(capture_screen(args.adb,args.serial))).convert('RGB')
        if frame.size != (900,1600) or np.abs(np.asarray(frame.crop((175,135,355,210)), dtype=np.int16)-template).mean() >= 12:
            raise SystemExit('Open the calibrated Chinese deck page first; no tap sent.')
        active = []
        for i, x in enumerate(centers, 1):
            patch = np.asarray(frame.crop((x-27,250,x-17,290)), dtype=np.int16)
            yellow = (patch[...,0] > 180) & (patch[...,1] > 115) & (patch[...,2] < 150)
            if yellow.mean() > .65:
                active.append(i)
        return active[0] if len(active) == 1 else None

    old = selected()
    if old is None:
        raise SystemExit('Cannot verify the highlighted preset; no tap sent.')
    if old != args.slot:
        subprocess.run(command+['shell','input','tap',str(centers[args.slot-1]),'276'], check=True, timeout=5)
        for _ in range(8):
            time.sleep(.25)
            if selected() == args.slot:
                break
        else:
            raise SystemExit('Preset did not become highlighted; inspect the deck page.')
    print(f'Confirmed highlighted deck preset: {args.slot}')


if __name__ == '__main__':
    main()
