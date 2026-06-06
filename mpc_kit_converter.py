#!/usr/bin/env python3
"""
Generate MPC 3 .xpj project files from expansion XPM kits.

HOW IT WORKS
  Each expansion kit is an .xpm file (XML) that lists instruments by number.
  This script reads the XPM, builds a proper .xpj project (gzip JSON) with
  explicit pad-to-WAV assignments and mute groups, and copies each kit's WAVs
  into a KitName_[ProjectData]/ hidden folder beside the .xpj file.

  The _[ProjectData] folders are invisible in the MPC device browser, so the
  expansion folder appears flat: open golddust/ and see all kit names directly.

MODES

  Single XPM kit:
    python3 mpc_kit_converter.py --xpm "/Library/.../golddust/Kit.xpm" --output ~/Desktop/kits

  One entire expansion:
    python3 mpc_kit_converter.py --expansion "/Library/.../golddust" --output ~/Desktop/kits

  Expansion by short name:
    python3 mpc_kit_converter.py --expansion-name golddust --output ~/Desktop/kits

  All expansions at once:
    python3 mpc_kit_converter.py --all-expansions --output ~/Desktop/kits

  List all expansions without generating:
    python3 mpc_kit_converter.py --list-expansions

  Legacy — folder that already has A01/A02 named WAVs (.mpcsample output):
    python3 mpc_kit_converter.py "/path/to/Kit Folder" [--bpm 128]

OPTIONS
  --output DIR            Where to create expansion folders (required for XPM modes)
  --template FILE         Path to a .xpj file from the device to use as template
                          (default: template/device_template.xpj next to this script)
  --content-dir DIR       MPC Content directory (default: /Library/Application Support/Akai/MPC/Content)
  --bpm N                 Override BPM (auto-detected from filename by default)
  --bpm-fallback N        Default BPM when auto-detection fails (default: 120)
  --bars N                Sequence length in bars (default: 2)
  --filter PATTERN        Only convert kits whose name contains PATTERN (case-insensitive)
  --skip PATTERN          Skip kits whose name contains PATTERN (case-insensitive)
  --overwrite             Replace existing output
  --dry-run               Preview without writing anything
  --no-copy-wavs          Only write .xpj, skip copying WAVs
  --quiet                 Suppress per-kit output lines, show only summaries
  --workers N             Number of parallel workers for kit generation (default: 1)

DISK SPACE NOTE
  Copying all WAVs from all 26 expansions duplicates ~29 GB of audio.
  Use --no-copy-wavs if the expansions are already on your device and MPC
  can see them at their original paths.
"""

import concurrent.futures
import gzip
import json
import os
import platform
import re
import shutil
import struct
import sys
import threading
import argparse
import xml.etree.ElementTree as ET

__version__ = "1.0.0"

MPCSAMPLE_HEADER = b"ACVS\n3.8.0.25\nSerialisableAC50ExportData\njson\nOSX\n"
XPJ_HEADER       = b"ACVS\n1.3.0.12\nSerialisableProjectData\njson\nLinux\n"
BANKS = list("ABCDEFGH")

_OS = platform.system()
if _OS == "Windows":
    DEFAULT_CONTENT_DIR = r"C:\ProgramData\Akai\MPC\Content"
elif _OS == "Darwin":
    DEFAULT_CONTENT_DIR = "/Library/Application Support/Akai/MPC/Content"
else:
    DEFAULT_CONTENT_DIR = "/usr/share/Akai/MPC/Content"

_print_lock = threading.Lock()


def _log(*args, quiet=False, **kwargs):
    if not quiet:
        with _print_lock:
            print(*args, **kwargs)


# ── SXQ (Akai demo sequence) parsing ─────────────────────────────────────────

def _read_var_len(data: bytes, pos: int) -> tuple[int, int]:
    val = 0
    while True:
        b = data[pos]; pos += 1
        val = (val << 7) | (b & 0x7F)
        if not (b & 0x80):
            return val, pos


def parse_sxq(sxq_path: str) -> dict | None:
    """Parse an Akai .sxq file (MIDI format with proprietary meta events).

    Returns {"bpm": float, "ppqn": int, "notes": [(tick, midi_note, vel), ...]}
    or None if the file is missing or unreadable.

    Timing: MIDI delta-time ticks accumulated into absolute tick positions.
    Note events are stored as 0xFF 0x7F meta events with 20-byte Akai payload
    where payload[2]==0x11 marks a drum hit; payload[4]=note, payload[5]=vel.
    """
    try:
        with open(sxq_path, "rb") as f:
            data = f.read()
    except OSError:
        return None

    if len(data) < 14 or data[:4] != b"MThd":
        return None

    ntracks = struct.unpack_from(">H", data, 10)[0]
    ppqn    = struct.unpack_from(">H", data, 12)[0]
    if ppqn == 0:
        return None

    bpm   = None
    notes = []
    pos   = 14

    for _ in range(ntracks):
        if len(data) < pos + 8 or data[pos:pos+4] != b"MTrk":
            break
        tlen = struct.unpack_from(">I", data, pos+4)[0]
        td   = data[pos+8 : pos+8+tlen]
        pos += 8 + tlen

        tp = 0; tick = 0
        while tp < len(td):
            delta, tp = _read_var_len(td, tp)
            tick += delta
            if tp >= len(td):
                break
            status = td[tp]; tp += 1

            if status == 0xFF:
                if tp >= len(td): break
                mtype = td[tp]; tp += 1
                mlen, tp = _read_var_len(td, tp)
                mdata = td[tp:tp+mlen]; tp += mlen
                if mtype == 0x51 and len(mdata) >= 3:
                    uspqn = struct.unpack_from(">I", b"\x00" + mdata[:3])[0]
                    if uspqn:
                        bpm = round(60_000_000 / uspqn, 1)
                elif mtype == 0x2F:
                    break
                elif (mtype == 0x7F and len(mdata) == 20
                      and mdata[0] == 0x47 and mdata[1] == 0x61
                      and mdata[2] == 0x11):
                    notes.append((tick, mdata[4], mdata[5]))
            elif status & 0xF0 in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                tp += 2
            elif status & 0xF0 in (0xC0, 0xD0):
                tp += 1
            elif status in (0xF0, 0xF7):
                slen, tp = _read_var_len(td, tp)
                tp += slen

    return {"bpm": bpm, "ppqn": ppqn, "notes": sorted(notes)} if notes else None


def _build_clip_event_list(notes: list, assignments: list, ppqn: int) -> dict:
    """Translate SXQ note tuples into an XPJ clip eventList dict.

    Filters to only pads that actually have a WAV assigned in the kit.
    MPC drum MIDI mapping: instrument slot i (0-based) → MIDI note 36+i.
    """
    valid_notes = {36 + i for i, a in enumerate(assignments) if a is not None}

    scale = 960 / ppqn if ppqn != 960 else 1
    hits  = sorted(
        (round(t * scale), n, v)
        for t, n, v in notes
        if n in valid_notes
    )

    if not hits:
        return _empty_event_list()

    unique = sorted({n for _, n, _ in hits})

    events = [
        {
            "version": 2, "time": 0, "type": 1, "channel": 0,
            "selected": False, "muted": False, "invented": True,
            "automation": {"note": n, "value": 0.0, "parameter": 131},
        }
        for n in unique
    ]
    for tick, note, vel in hits:
        events.append({
            "version": 2, "time": tick, "type": 3, "channel": 0,
            "selected": False, "muted": False, "invented": False,
            "note": {
                "version": 1,
                "note": note,
                "velocity": round(vel / 127.0, 6),
                "length": 120,
                "probability": 100,
                "ratchet": 1,
                "articulation": 197,
                **{f"modifierValue{i}":      (0.5 if i in (0, 1, 5) else (1.0 if i == 6 else 0.0)) for i in range(16)},
                **{f"modifierActiveState{i}": False                                                  for i in range(16)},
                "EnumCerealisationWrapper(selectedModifierType)": "Tuning (coarse)",
            },
        })

    return {
        "length": 9223372036854775807,
        "events": events,
        "version": 2,
        "quantisation": {"version": 1, "pulses": 0, "swing": 0.0, "strength": 1.0},
        "numFilterTypes": 30,
    }


def _empty_event_list() -> dict:
    return {
        "length": 9223372036854775807,
        "events": [],
        "version": 2,
        "quantisation": {"version": 1, "pulses": 0, "swing": 0.0, "strength": 1.0},
        "numFilterTypes": 30,
    }


# ── WAV frame count (reads header only, no audio data loaded) ────────────────

def _get_wav_frames(path: str) -> int:
    """Return total frame count from a WAV file header, or 0 on any error."""
    try:
        with open(path, "rb") as f:
            header = f.read(44)
        if len(header) < 44 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return 0
        bytes_per_frame = 0
        f2 = open(path, "rb")
        f2.read(12)
        while True:
            chunk_hdr = f2.read(8)
            if len(chunk_hdr) < 8:
                break
            cid, csize = chunk_hdr[:4], struct.unpack_from("<I", chunk_hdr, 4)[0]
            if cid == b"fmt ":
                fmt = f2.read(min(csize, 16))
                channels        = struct.unpack_from("<H", fmt, 2)[0]
                bits_per_samp   = struct.unpack_from("<H", fmt, 14)[0]
                bytes_per_frame = channels * (bits_per_samp // 8)
            elif cid == b"data":
                f2.close()
                return csize // bytes_per_frame if bytes_per_frame else 0
            else:
                f2.seek(csize, 1)
        f2.close()
    except Exception:
        pass
    return 0


# ── XPJ template (loaded once from a real device .xpj file) ──────────────────

_XPJ_TEMPLATE = None


def _ensure_template(template_path=None):
    global _XPJ_TEMPLATE
    if _XPJ_TEMPLATE is not None:
        return _XPJ_TEMPLATE

    if template_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(script_dir, "template", "device_template.xpj")

    if not os.path.exists(template_path):
        sys.exit(
            f"ERROR: XPJ template not found at {template_path!r}\n"
            f"Use --template to point to a .xpj file exported from your MPC device."
        )

    raw = gzip.decompress(open(template_path, "rb").read())
    pos = 0
    for _ in range(5):
        pos = raw.index(b"\n", pos) + 1
    _XPJ_TEMPLATE = json.loads(raw[pos:])["data"]
    return _XPJ_TEMPLATE


# ── XPJ per-instrument / layer builders ──────────────────────────────────────

def _make_sample_entry(wav_file: str, bpm: float) -> dict:
    stem = os.path.splitext(wav_file)[0]
    return {
        "version":  1,
        "name":     stem,
        "path":     wav_file,
        "loadImpl": 0,
        "metadata": {"tempo": bpm, "rootNote": 60, "tune": 0.0, "key": "C Major"},
    }


def _make_layer(wav_file: str, num_frames: int = 0) -> dict:
    stem = os.path.splitext(wav_file)[0]
    return {
        "active":       True,
        "volume":       {"gainCoefficient": 1.0, "controlValue": 1.0, "law": 1},
        "pan":          0.5,
        "pitch":        0.0,
        "coarseTune":   0,
        "fineTune":     0,
        "velocityStart": 0,
        "velocityEnd":  127,
        "sampleStart":  0,
        "sampleEnd":    num_frames,
        "loop":         False,
        "loopStart":    0,
        "loopEnd":      num_frames,
        "loopCrossfadeLength": 0,
        "loopFineTune": 0,
        "mute":         False,
        "rootNote":     0,
        "keyTrackEnable": False,
        "sampleName":   stem,
        "sampleFile":   wav_file,
        "sliceIndex":   129,
        "direction":    0,
        "offset":       0,
        "sliceInfo": {
            "Start": 0, "End": num_frames, "LoopStart": 0, "LoopMode": 0,
            "PulsePosition": 0, "LoopCrossfadeLength": 0, "LoopCrossfadeType": 0,
            "TailLength": 0.0, "TailLoopPosition": 0.5,
        },
        "version":      11,
        "pitchRandom":  0.0,
        "VolumeRandom": 0.0,
        "PanRandom":    0.0,
        "OffsetRandom": 0.0,
        "layerLoopModeOverridesSliceLoopMode": True,
        "loopMode":     0,
    }


# ── XPJ byte generation ───────────────────────────────────────────────────────

def build_xpj_bytes(kit_name: str, bpm: float, bars: int,
                    assignments: list, template_path=None,
                    sxq_data: dict | None = None) -> bytes:
    """
    assignments: 128-element list; each slot is either None (empty pad) or
                 {"dest_name": "file.wav", "mute_group": 0, "num_frames": N}
    sxq_data: optional parsed SXQ dict from parse_sxq(); populates the sequence
              with the kit's demo pattern. When None, an empty sequence is used.
    """
    tmpl = _ensure_template(template_path)
    tmpl_track  = tmpl["tracks"][0]
    tmpl_drum   = tmpl_track["program"]["drum"]
    tmpl_empty  = tmpl_drum["instruments"][16]   # empty-pad template (shared ref, read-only)
    tmpl_active = tmpl_drum["instruments"][0]    # active-pad template (shared ref, read-only)
    old_name    = tmpl_track["name"]
    pulses      = 3840 * bars

    samples = [
        _make_sample_entry(a["dest_name"], bpm)
        for a in assignments if a is not None
    ]

    instruments = []
    for slot in assignments:
        if slot is not None:
            instruments.append({
                **tmpl_active,
                "whichMuteGroup": slot["mute_group"],
                "layersv":        [_make_layer(slot["dest_name"], slot.get("num_frames", 0))],
            })
        else:
            instruments.append(tmpl_empty)

    drum    = {**tmpl_drum, "instruments": instruments}
    program = {**tmpl_track["program"], "name": kit_name, "drum": drum}
    track0  = {**tmpl_track, "name": kit_name, "samples": samples, "program": program}

    # Build event list: from SXQ demo data if available, otherwise empty.
    if sxq_data:
        event_list = _build_clip_event_list(
            sxq_data["notes"], assignments, sxq_data["ppqn"]
        )
    else:
        event_list = _empty_event_list()

    def _remap_clip(clip):
        """Remap the kit clip to the new name/key; update pulse lengths; clear stale events."""
        if clip["key"] != old_name:
            # Non-kit clips (output routings etc.) — update only pulse lengths.
            return {**clip, "value": {
                **clip["value"],
                "endPulses":      pulses,
                "loopEndPulses":  pulses,
            }}
        return {**clip, "key": kit_name, "value": {
            **clip["value"],
            "name":           kit_name,
            "endPulses":      pulses,
            "loopEndPulses":  pulses,
            "eventList":      event_list,
        }}

    # Use the template's key-0 sequence as the single clean base sequence.
    base_seq = next(
        (e for e in tmpl["sequences"] if e["key"] == 0),
        tmpl["sequences"][-1],
    )
    sv = base_seq["value"]
    single_sequence = [{
        "key": 0,
        "value": {
            **sv,
            "bpm":           bpm,
            "lengthBars":    bars,
            "lengthPulses":  pulses,
            "loopEndBar":    bars,
            "loopEndPulses": pulses,
            "trackClipMaps": [
                [_remap_clip(clip) for clip in track_clips]
                for track_clips in sv.get("trackClipMaps", [])
            ],
        },
    }]

    data = {
        **tmpl,
        "masterTempo":        bpm,
        "masterTempoEnabled": True,
        "info":               {**tmpl.get("info", {}), "title": kit_name},
        "samples":            samples,
        "tracks":             [track0] + tmpl["tracks"][1:],
        "sequences":          single_sequence,
    }

    payload = XPJ_HEADER + json.dumps({"data": data}, separators=(",", ":")).encode("utf-8")
    return gzip.compress(payload, compresslevel=9)


# ── Legacy .mpcsample byte generation ────────────────────────────────────────

def build_mpcsample_bytes(bpm: float, bars: int,
                          mute_groups: list[int] | None = None) -> bytes:
    data = {
        "data": {
            "sequences": [{"key": 0, "value": {
                "bpm": bpm, "lengthBars": bars, "lengthPulses": 3840 * bars,
                "tempoEnable": True,
                "timeSignatureList": {
                    "timeSignatures": [{"beatsPerBar": 4, "beatLength": 960, "barStart": 0}]
                },
                "eventList": {
                    "length": 9223372036854775807, "events": [], "version": 2,
                    "quantisation": {"version": 1, "pulses": 0, "swing": 0.0, "strength": 1.0},
                    "numFilterTypes": 30,
                },
            }}],
            "muteGroups":        mute_groups if mute_groups is not None else [0] * 128,
            "simultPlayTargets": [0] * 128,
        }
    }
    payload = MPCSAMPLE_HEADER + json.dumps(data, indent=2).encode("utf-8")
    return gzip.compress(payload, compresslevel=9)


# ── XPM parsing ───────────────────────────────────────────────────────────────

def extract_bpm(kit_name: str, fallback: float = 120.0) -> float:
    for m in re.findall(r"\b(\d{2,3})\b", kit_name):
        bpm = int(m)
        if 60 <= bpm <= 220:
            return float(bpm)
    return fallback


def pad_label(instrument_number: int) -> str:
    idx  = instrument_number - 1
    bank = BANKS[idx // 16] if idx // 16 < len(BANKS) else "?"
    return f"{bank}{(idx % 16) + 1:02d}"


def parse_xpm(xpm_path: str) -> dict:
    """
    Returns:
      {
        "instruments": [{"index": 0, "pad": "A01", "primary": "SampleName", "layers": [...]}, ...],
        "mute_groups": [0, 0, 1, ...]   # 128 entries
      }
    """
    try:
        tree = ET.parse(xpm_path)
    except ET.ParseError as e:
        print(f"  WARN  XML parse error in {os.path.basename(xpm_path)}: {e}")
        return {"instruments": [], "mute_groups": [0] * 128}

    prog = tree.getroot().find("Program")
    if prog is None:
        return {"instruments": [], "mute_groups": [0] * 128}

    instruments = []
    mute_groups = [0] * 128

    for inst in prog.findall("Instruments/Instrument"):
        num = int(inst.get("number", 0))
        if num == 0 or num > 128:
            continue

        mg = int((inst.findtext("MuteGroup") or "0").strip() or "0")
        mute_groups[num - 1] = mg

        layers = [
            (layer.findtext("SampleName") or "").strip()
            for layer in inst.findall("Layers/Layer")
        ]
        layers = [s for s in layers if s]
        if layers:
            instruments.append({
                "index":   num - 1,
                "pad":     pad_label(num),
                "primary": layers[0],
                "layers":  layers,
            })

    return {"instruments": instruments, "mute_groups": mute_groups}


def clean_expansion_name(raw_name: str) -> str:
    for prefix in ("com.akaipro.mpc.expansion.", "com.akaipro.mpc.",
                   "com.native.", "com.akaipro.", "com."):
        if raw_name.lower().startswith(prefix):
            return raw_name[len(prefix):]
    return raw_name


def build_wav_index(directory: str) -> dict[str, str]:
    return {
        os.path.splitext(f)[0].strip().lower(): f
        for f in os.listdir(directory)
        if f.lower().endswith(".wav")
    }


# ── Expansion discovery ───────────────────────────────────────────────────────

def _get_content_dir(explicit: str | None) -> str:
    return os.path.abspath(explicit) if explicit else DEFAULT_CONTENT_DIR


def _scan_expansions(content_dir: str) -> list[str]:
    """Return sorted list of absolute expansion directory paths."""
    if not os.path.isdir(content_dir):
        sys.exit(f"ERROR: Content directory not found: {content_dir!r}")
    return sorted(
        os.path.join(content_dir, d)
        for d in os.listdir(content_dir)
        if os.path.isdir(os.path.join(content_dir, d)) and not d.startswith(".")
    )


def list_expansions(content_dir: str) -> None:
    """Print a table of expansion names and kit counts."""
    expansions = _scan_expansions(content_dir)
    if not expansions:
        print("No expansions found.")
        return

    rows = []
    for path in expansions:
        name  = clean_expansion_name(os.path.basename(path))
        count = sum(1 for f in os.listdir(path) if f.lower().endswith(".xpm"))
        rows.append((name, count))

    col = max(len(r[0]) for r in rows) + 2
    print(f"\n{'Expansion':<{col}} {'Kits':>5}")
    print("─" * (col + 7))
    for name, count in rows:
        print(f"{name:<{col}} {count:>5}")
    print("─" * (col + 7))
    print(f"{'TOTAL':<{col}} {sum(r[1] for r in rows):>5}  ({len(rows)} expansions)\n")


def find_expansion_by_name(name: str, content_dir: str) -> str:
    """Return path to the expansion matching name (case-insensitive). Exits on no match."""
    for path in _scan_expansions(content_dir):
        if clean_expansion_name(os.path.basename(path)).lower() == name.lower():
            return path
    sys.exit(
        f"ERROR: Expansion {name!r} not found in {content_dir}\n"
        f"Run --list-expansions to see available expansions."
    )


# ── Kit generation (XPM → .xpj + _[ProjectData]/) ───────────────────────────

def generate_from_xpm(xpm_path: str, expansion_dir: str, output_dir: str,
                      bpm_override: float | None, bars: int,
                      overwrite: bool, dry_run: bool, copy_wavs: bool,
                      template_path=None, quiet: bool = False,
                      bpm_fallback: float = 120.0) -> bool:
    kit_name = os.path.splitext(os.path.basename(xpm_path))[0]
    bpm      = bpm_override if bpm_override is not None else extract_bpm(kit_name, bpm_fallback)

    out_xpj    = os.path.join(output_dir, f"{kit_name}.xpj")
    out_folder = os.path.join(output_dir, f"{kit_name}_[ProjectData]")

    if os.path.exists(out_xpj) and not overwrite:
        _log(f"  SKIP  {kit_name!r} (exists — use --overwrite)", quiet=quiet)
        return False

    xpm_data    = parse_xpm(xpm_path)
    instruments = xpm_data["instruments"]
    mute_groups = xpm_data["mute_groups"]

    if not instruments:
        _log(f"  SKIP  {kit_name!r} (no instruments in XPM)", quiet=quiet)
        return False

    wav_index = build_wav_index(expansion_dir)

    assignments = [None] * 128
    missing = []
    for inst in instruments:
        idx      = inst["index"]
        wav_file = wav_index.get(inst["primary"].lower())
        if wav_file:
            src = os.path.join(expansion_dir, wav_file)
            assignments[idx] = {
                "dest_name":  wav_file,
                "src":        src,
                "mute_group": mute_groups[idx],
                "num_frames": _get_wav_frames(src),
            }
        else:
            missing.append(inst["primary"])

    # Look for a companion .sxq demo sequence alongside the .xpm file.
    sxq_path = os.path.splitext(xpm_path)[0] + ".sxq"
    sxq_data  = parse_sxq(sxq_path)
    if sxq_data and sxq_data.get("bpm") and bpm_override is None:
        bpm = sxq_data["bpm"]

    active = [a for a in assignments if a is not None]
    multi  = sum(1 for i in instruments if len(i["layers"]) > 1
                 and wav_index.get(i["primary"].lower()))

    status = f"BPM={bpm:.0f}  pads={len(active)}"
    if sxq_data:
        valid_hits = sum(
            1 for _, n, _ in sxq_data["notes"]
            if assignments[n - 36] is not None
        ) if sxq_data["notes"] else 0
        status += f"  demo={valid_hits}hits"
    if missing:
        status += f"  missing={len(missing)}"
    if multi:
        status += f"  multi-layer={multi} (layer 1 used)"

    lines = [f"  {'DRY' if dry_run else 'OK ':3s}  {kit_name}  [{status}]"]
    for m in missing:
        lines.append(f"        WARN WAV not found: {m!r}")
    _log("\n".join(lines), quiet=quiet)

    if dry_run:
        return True

    os.makedirs(out_folder, exist_ok=True)

    if copy_wavs:
        for a in assignments:
            if a is not None:
                dest = os.path.join(out_folder, a["dest_name"])
                if not os.path.exists(dest) or overwrite:
                    shutil.copy2(a["src"], dest)

    with open(out_xpj, "wb") as f:
        f.write(build_xpj_bytes(kit_name, bpm, bars, assignments, template_path, sxq_data))

    return True


def process_expansion(expansion_dir: str, output_dir: str,
                      bpm_override: float | None, bars: int,
                      overwrite: bool, dry_run: bool, copy_wavs: bool,
                      template_path=None, quiet: bool = False,
                      filter_pat: str | None = None, skip_pat: str | None = None,
                      workers: int = 1, bpm_fallback: float = 120.0) -> tuple[int, int]:
    xpms = sorted(f for f in os.listdir(expansion_dir) if f.lower().endswith(".xpm"))
    if not xpms:
        return 0, 0

    # Apply name filters before processing
    if filter_pat:
        xpms = [x for x in xpms if filter_pat.lower() in os.path.splitext(x)[0].lower()]
    if skip_pat:
        xpms = [x for x in xpms if skip_pat.lower() not in os.path.splitext(x)[0].lower()]

    if not xpms:
        return 0, 0

    exp_name       = clean_expansion_name(os.path.basename(expansion_dir))
    exp_output_dir = os.path.join(output_dir, exp_name)

    _log(f"\n{'─'*70}", quiet=quiet)
    _log(f"  Expansion: {exp_name}  ({len(xpms)} kits)", quiet=quiet)
    _log(f"{'─'*70}", quiet=quiet)

    # Pre-load template before spawning threads
    _ensure_template(template_path)

    def _run_one(xpm_file):
        return generate_from_xpm(
            os.path.join(expansion_dir, xpm_file),
            expansion_dir, exp_output_dir,
            bpm_override, bars, overwrite, dry_run, copy_wavs,
            template_path, quiet, bpm_fallback,
        )

    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            ok = sum(pool.map(_run_one, xpms))
    else:
        ok = sum(_run_one(xpm) for xpm in xpms)

    return ok, len(xpms)


# ── Legacy mode: folder with pre-named A01/A02 WAVs (.mpcsample) ─────────────

def infer_pad_prefix(filename: str) -> str | None:
    name = filename.upper()
    for bank in BANKS:
        for pad in range(1, 17):
            if name.startswith(f"{bank}{pad:02d}"):
                return f"{bank}{pad:02d}"
    return None


def generate_for_folder(folder: str, bpm: float, bars: int,
                        overwrite: bool, dry_run: bool) -> bool:
    folder   = os.path.abspath(folder)
    kit_name = os.path.basename(folder)
    out_path = os.path.join(folder, f"{kit_name}.mpcsample")

    if os.path.exists(out_path) and not overwrite:
        print(f"  SKIP  {kit_name!r} (exists — use --overwrite)")
        return False

    wavs = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
    if not wavs:
        print(f"  SKIP  {kit_name!r} (no WAV files)")
        return False

    named   = [f for f in wavs if infer_pad_prefix(f)]
    unnamed = [f for f in wavs if not infer_pad_prefix(f)]
    print(f"\n  Kit : {kit_name}  BPM={bpm}")
    print(f"  WAVs: {len(wavs)} total  ({len(named)} prefixed, {len(unnamed)} without prefix)")
    if unnamed:
        print("  NOTE: Files without A01/A02 prefix won't map to MPC pads correctly.")

    if dry_run:
        print(f"  DRY   Would write: {out_path}")
        return True

    with open(out_path, "wb") as f:
        f.write(build_mpcsample_bytes(bpm, bars))
    print(f"  OK    {out_path}")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate MPC 3 .xpj projects from expansion XPM kits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--xpm", metavar="FILE",
                       help="Single .xpm kit file")
    modes.add_argument("--expansion", metavar="DIR",
                       help="One expansion directory (all XPM kits inside)")
    modes.add_argument("--expansion-name", metavar="NAME",
                       help="Expansion short name, e.g. golddust (searches Content dir)")
    modes.add_argument("--all-expansions", metavar="DIR", nargs="?", const=True,
                       help="All expansions under Content dir")
    modes.add_argument("--list-expansions", action="store_true",
                       help="List expansion names and kit counts, then exit")

    p.add_argument("folder", nargs="?",
                   help="Legacy: folder with A01/A02-prefixed WAVs")
    p.add_argument("--content-dir", metavar="DIR",
                   help=f"MPC Content directory (default: {DEFAULT_CONTENT_DIR})")
    p.add_argument("--output",       metavar="DIR",
                   help="Where to create expansion folders (XPM modes)")
    p.add_argument("--template",     metavar="FILE",
                   help="Path to a .xpj file from your MPC device (template)")
    p.add_argument("--bpm",          type=float, default=None, metavar="N",
                   help="Override BPM for every kit")
    p.add_argument("--bpm-fallback", type=float, default=120.0, metavar="N",
                   help="Default BPM when auto-detection fails (default: 120)")
    p.add_argument("--bars",         type=int,   default=2,
                   help="Sequence length in bars (default: 2)")
    p.add_argument("--filter",       metavar="PATTERN",
                   help="Only convert kits whose name contains PATTERN (case-insensitive)")
    p.add_argument("--skip",         metavar="PATTERN",
                   help="Skip kits whose name contains PATTERN (case-insensitive)")
    p.add_argument("--overwrite",    action="store_true",
                   help="Replace existing output files")
    p.add_argument("--dry-run",      action="store_true",
                   help="Preview without writing anything")
    p.add_argument("--no-copy-wavs", action="store_true",
                   help="Write .xpj files only, skip copying WAVs")
    p.add_argument("--quiet",        action="store_true",
                   help="Suppress per-kit output lines, show only summaries")
    p.add_argument("--workers",      type=int, default=1, metavar="N",
                   help="Parallel workers for kit generation (default: 1)")
    p.add_argument("--batch",        action="store_true",
                   help="Legacy: treat 'folder' as parent, process each subfolder")

    args = p.parse_args()

    if args.dry_run:
        print("[DRY RUN — nothing will be written]\n")

    copy_wavs     = not args.no_copy_wavs
    template_path = args.template
    content_dir   = _get_content_dir(args.content_dir)

    # Shared kwargs forwarded to process_expansion
    exp_kwargs = dict(
        bpm_override=args.bpm,
        bars=args.bars,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        copy_wavs=copy_wavs,
        template_path=template_path,
        quiet=args.quiet,
        filter_pat=args.filter,
        skip_pat=args.skip,
        workers=args.workers,
        bpm_fallback=args.bpm_fallback,
    )

    def require_output():
        if not args.output:
            p.error("--output DIR is required for XPM modes")
        return os.path.abspath(args.output)

    # ── --list-expansions ─────────────────────────────────────────────────────
    if args.list_expansions:
        list_expansions(content_dir)
        return

    # ── --xpm ─────────────────────────────────────────────────────────────────
    if args.xpm:
        xpm        = os.path.abspath(args.xpm)
        output_dir = require_output()
        os.makedirs(output_dir, exist_ok=True)
        generate_from_xpm(
            xpm, os.path.dirname(xpm), output_dir,
            args.bpm, args.bars, args.overwrite, args.dry_run,
            copy_wavs, template_path, args.quiet, args.bpm_fallback,
        )
        return

    # ── --expansion ───────────────────────────────────────────────────────────
    if args.expansion:
        output_dir = require_output()
        os.makedirs(output_dir, exist_ok=True)
        ok, total = process_expansion(os.path.abspath(args.expansion), output_dir,
                                      **exp_kwargs)
        print(f"\nDone: {ok}/{total} kits.")
        return

    # ── --expansion-name ──────────────────────────────────────────────────────
    if args.expansion_name:
        exp_dir    = find_expansion_by_name(args.expansion_name, content_dir)
        output_dir = require_output()
        os.makedirs(output_dir, exist_ok=True)
        ok, total  = process_expansion(exp_dir, output_dir, **exp_kwargs)
        print(f"\nDone: {ok}/{total} kits.")
        return

    # ── --all-expansions ──────────────────────────────────────────────────────
    if args.all_expansions is not None:
        # --all-expansions DIR (legacy positional on the flag) takes precedence
        if isinstance(args.all_expansions, str):
            content_dir = os.path.abspath(args.all_expansions)

        output_dir = require_output()
        os.makedirs(output_dir, exist_ok=True)
        expansions = _scan_expansions(content_dir)

        total_ok = total_kits = 0
        for exp in expansions:
            ok, total = process_expansion(exp, output_dir, **exp_kwargs)
            total_ok   += ok
            total_kits += total

        print(f"\n{'═'*70}")
        print(f"ALL DONE: {total_ok}/{total_kits} kits across {len(expansions)} expansions.")
        return

    # ── Legacy folder mode ────────────────────────────────────────────────────
    if args.folder:
        bpm    = args.bpm or 120.0
        folder = os.path.abspath(args.folder)

        if args.batch:
            subfolders = sorted(
                os.path.join(folder, d)
                for d in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, d)) and not d.startswith(".")
            )
            if not subfolders:
                print(f"No subfolders in: {folder}")
                sys.exit(1)
            print(f"Batch: {len(subfolders)} folders in {folder}")
            ok = sum(generate_for_folder(f, bpm, args.bars, args.overwrite, args.dry_run)
                     for f in subfolders)
            print(f"\nDone: {ok}/{len(subfolders)} kits.")
        else:
            generate_for_folder(folder, bpm, args.bars, args.overwrite, args.dry_run)
        return

    p.print_help()


if __name__ == "__main__":
    main()
