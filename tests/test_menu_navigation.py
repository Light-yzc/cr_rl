from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from cr_rl.client import menu_navigation as nav


def png(image):
    data=BytesIO();image.save(data,format='PNG');return data.getvalue()


def test_verified_button_taps_once_and_probe_does_not_tap(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    frame=Image.new('RGB',(900,1600))
    frame.paste(Image.open(root/'configs/nulls_ui_900x1600/battle.png'),(325,1185))
    calls=[]
    monkeypatch.setattr(nav.subprocess,'check_output',lambda *a,**k:png(frame))
    monkeypatch.setattr(nav.subprocess,'run',lambda *a,**k:calls.append(a[0]))
    nav.guarded_tap('adb','test','battle',tap=False)
    assert not calls
    nav.guarded_tap('adb','test','battle')
    assert calls==[['adb','-s','test','shell','input','tap','450','1250']]


def test_unknown_page_never_taps_and_preserves_evidence(monkeypatch,tmp_path):
    frame=Image.new('RGB',(900,1600),'black')
    clock=iter([0.,0.,2.])
    monkeypatch.setattr(nav.time,'monotonic',lambda:next(clock))
    monkeypatch.setattr(nav.time,'sleep',lambda _:None)
    monkeypatch.setattr(nav.subprocess,'check_output',lambda *a,**k:png(frame))
    monkeypatch.setattr(nav.subprocess,'run',lambda *a,**k:pytest.fail('blind tap'))
    with pytest.raises(RuntimeError,match='best pixel error'):
        nav.guarded_tap('adb','test','battle',timeout=1,failure_dir=tmp_path)
    assert len(list(tmp_path.glob('*.png')))==1


def screen(name):
    image=Image.new('RGB',(900,1600))
    for reference,box in nav.RECOVERY_SCREENS[name][0]:
        image.paste(Image.fromarray(nav._reference(reference).astype('uint8')),box[:2])
    return png(image)


@pytest.mark.parametrize('name', ['trophy_road','mode_picker','exit_dialog','profile','lobby','decks','result'])
def test_recovery_scene_landmarks_are_unambiguous(name):
    assert nav.classify_menu(screen(name))==name


def test_trophy_road_recovery_uses_confirm_and_verifies_lobby(monkeypatch):
    shots=iter([screen('trophy_road')]*2+[screen('lobby')]*2)
    monkeypatch.setattr(nav,'capture_screen',lambda *a:next(shots))
    monkeypatch.setattr(nav.time,'sleep',lambda _:None)
    taps=[]
    monkeypatch.setattr(nav.subprocess,'run',lambda cmd,**k:taps.append(cmd))
    nav.ensure_lobby('adb','test',battle_probe=lambda:False)
    assert taps==[['adb','-s','test','shell','input','tap','450','1540']]


def test_recovery_never_clicks_over_active_battle(monkeypatch):
    monkeypatch.setattr(nav,'capture_screen',lambda *a:screen('trophy_road'))
    monkeypatch.setattr(nav.time,'sleep',lambda _:None)
    monkeypatch.setattr(nav.subprocess,'run',lambda *a,**k:pytest.fail('active battle tap'))
    with pytest.raises(RuntimeError,match='active or uncertain'):
        nav.ensure_lobby('adb','test',battle_probe=lambda:True)
