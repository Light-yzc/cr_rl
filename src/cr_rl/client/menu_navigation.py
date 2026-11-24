"""Opt-in navigation for the calibrated Chinese Nulls lobby and result screen."""
import io
from pathlib import Path
import subprocess
import time
from functools import lru_cache

import numpy as np
from .runtime import capture_screen


def guarded_tap(adb, serial, screen, evidence=None, *, timeout=12, tap=True, failure_dir=None):
    from PIL import Image

    template_name, box, point = {
        'battle': ('battle',(325,1185,580,1310),(450,1250)),
        'result': ('result',(345,1413,553,1494),(450,1450)),
        'open_decks': ('battle',(325,1185,580,1310),(223,1515)),
        'leave_decks': ('deck_tab',(175,135,355,210),(450,1515)),
    }[screen]
    template = Path(__file__).resolve().parents[3]/'configs/nulls_ui_900x1600'/f'{template_name}.png'
    ref = np.asarray(Image.open(template).convert('RGB'), dtype=np.int16)
    cmd = [adb, '-s', serial]
    end = time.monotonic() + timeout
    png = None
    best_error = float('inf')
    while time.monotonic() < end:
        png = capture_screen(adb,serial)
        img = Image.open(io.BytesIO(png)).convert('RGB')
        if img.size != (900,1600):
            raise ValueError('Menu automation requires the calibrated 900x1600 UI')
        crop = np.asarray(img.crop(box), dtype=np.int16)
        error = float(np.abs(ref-crop).mean())
        best_error = min(best_error,error)
        if error < 13:
            if evidence is not None:
                Path(evidence).write_bytes(png)
            if tap:
                subprocess.run(cmd+['shell','input','tap',*map(str,point)],check=True,timeout=5)
            return
        time.sleep(.45)
    if failure_dir is not None and png is not None:
        folder=Path(failure_dir);folder.mkdir(parents=True,exist_ok=True)
        (folder/f'navigation_failed_{screen}_{time.time_ns()}.png').write_bytes(png)
    raise RuntimeError(f'Could not verify the {screen} button; no tap sent (best pixel error {best_error:.2f})')


# These are measured landmarks from this exact 900x1600 private client.
# Android BACK opens the quit dialog here; it is deliberately never a fallback.
RECOVERY_SCREENS = {
    'trophy_road': ([('trophy_confirm',(344,1492,559,1592)),
                     ('trophy_progress',(12,1260,33,1430))], (450,1540)),
    'mode_picker': ([('mode_title',(330,333,570,397)),
                     ('mode_close',(365,189,535,289))], (450,235)),
    'exit_dialog': ([('exit_header',(135,508,750,562)),
                     ('exit_cancel',(177,944,389,1039))], (280,992)),
    'profile': ([('profile_header',(340,60,560,210)),
                 ('profile_close',(800,133,856,191))], (830,160)),
    'decks': ([('deck_tab',(175,135,355,210))], (450,1515)),
    'result': ([('result',(345,1413,553,1494))], (450,1450)),
    'lobby': ([('battle',(325,1185,580,1310))], None),
}


@lru_cache(maxsize=16)
def _reference(name):
    from PIL import Image
    path=Path(__file__).resolve().parents[3]/'configs/nulls_ui_900x1600'/(name+'.png')
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.int16)


def classify_menu(png):
    from PIL import Image
    image=Image.open(io.BytesIO(png)).convert('RGB')
    if image.size!=(900,1600):
        raise ValueError('Menu automation requires the calibrated 900x1600 UI')
    matches=[]
    for name,(landmarks,point) in RECOVERY_SCREENS.items():
        errors=[float(np.abs(np.asarray(image.crop(box),dtype=np.int16)-_reference(ref)).mean())
                for ref,box in landmarks]
        if all(error<13 for error in errors):
            matches.append((name,max(errors)))
    # Multiple matches mean ambiguous evidence, not permission to guess.
    return matches[0][0] if len(matches)==1 else 'unknown'


def ensure_lobby(adb, serial, *, timeout=30, failure_dir=None, battle_probe=None,
                 allow_result=False, emit=None):
    """Bounded scene recovery, two observations per action, verified postcondition.

    battle_probe returns True/False/None for active/inactive/unknown. A live or
    uncertain memory state blocks all recovery clicks. Unknown screens stop
    safely instead of attempting blind back presses or arbitrary taps.
    """
    deadline=time.monotonic()+timeout
    previous=None;stable=0;actions=0;png=None
    while time.monotonic()<deadline:
        png=capture_screen(adb,serial)
        screen=classify_menu(png)
        stable=stable+1 if screen==previous else 1
        previous=screen
        if stable<2:
            time.sleep(.3);continue
        if screen=='lobby':
            return
        recoverable=screen in RECOVERY_SCREENS and (screen!='result' or allow_result)
        if recoverable and actions<6:
            if battle_probe is not None and battle_probe() is not False:
                raise RuntimeError('Menu recovery blocked by active or uncertain battle memory')
            point=RECOVERY_SCREENS[screen][1]
            if point is not None:
                if failure_dir is not None:
                    folder=Path(failure_dir);folder.mkdir(parents=True,exist_ok=True)
                    (folder/f'navigation_recover_{screen}_{time.time_ns()}.png').write_bytes(png)
                subprocess.run([adb,'-s',serial,'shell','input','tap',*map(str,point)],
                               check=True,timeout=5)
                if emit is not None:emit('menu_recovery',screen=screen,point=point)
                actions+=1;stable=0;previous=None
                time.sleep(.65);continue
        time.sleep(.4)
    if failure_dir is not None and png is not None:
        folder=Path(failure_dir);folder.mkdir(parents=True,exist_ok=True)
        (folder/f'navigation_failed_lobby_{time.time_ns()}.png').write_bytes(png)
    raise RuntimeError(f'Could not verify the battle button after menu recovery; last screen={previous}; no blind tap sent')
