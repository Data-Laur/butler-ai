#!/usr/bin/env python3
"""Run the table-setting pipeline driven by VOICE (live microphone or audio file).

Usage:
    # 1. Using a recorded audio file or voice note:
    python scripts/run_voice_pipeline.py --audio stage1_voice/samples/official_command.m4a

    # 2. Live microphone recording (speaks into laptop mic):
    python scripts/run_voice_pipeline.py --mic --seconds 6

    # 3. Watch in real-time 3D MuJoCo desktop viewer:
    python scripts/run_voice_pipeline.py --audio stage1_voice/samples/official_command.m4a --view

    # 4. Record high-res video of the execution:
    python scripts/run_voice_pipeline.py --audio stage1_voice/samples/official_command.m4a --record outputs/voice_run.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import speech_recognition as sr
from common import EXAMPLE_COMMAND
from common.pipeline import run_once
from stage1_voice.voice import parse_command, parse_text, transcribe_audio


def record_live_microphone(seconds: int = 5) -> str:
    """Record live audio from the laptop microphone and transcribe it."""
    r = sr.Recognizer()
    print("\n" + "=" * 60)
    print("[MIC] MICROPHONE ACTIVE")
    print(f"   Please speak your table-setting command now (listening for {seconds}s)...")
    print("   Example: 'Open the top drawer, pick up the plate, and pour water into the mug.'")
    print("=" * 60 + "\n", flush=True)

    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.8)
            print(">>> Recording now... Speak!", flush=True)
            audio_data = r.record(source, duration=seconds)
            print(">>> Done recording. Transcribing...", flush=True)

        transcript = r.recognize_google(audio_data)
        print(f"\n[ASR Transcribed]: \"{transcript}\"\n", flush=True)
        return transcript.strip()
    except sr.UnknownValueError:
        print("[ASR Warning]: Could not clearly understand speech; falling back to example command.")
        return EXAMPLE_COMMAND
    except Exception as exc:
        print(f"[ASR Error]: {exc}; falling back to example command.")
        return EXAMPLE_COMMAND


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the table-setting pipeline driven by Voice.")
    parser.add_argument("--audio", type=Path, default=None, help="Path to an audio file (.wav, .m4a, .mp3).")
    parser.add_argument("--mic", action="store_true", help="Record live from your laptop microphone.")
    parser.add_argument("--seconds", type=int, default=6, help="Microphone recording duration in seconds.")
    parser.add_argument("--command", type=str, default=None, help="Direct text command override.")
    parser.add_argument("--seed", type=int, default=0, help="Randomization seed for reset_scene.")
    parser.add_argument("--view", action="store_true", help="Launch interactive 3D viewer during execution.")
    parser.add_argument("--record", type=Path, default=None, help="Save MP4/GIF video of the run.")
    args = parser.parse_args()

    print("=" * 72)
    print("VOICE-DRIVEN ROBOTIC PIPELINE")
    print("Audio -> Stage 1 Voice -> Stage 2 Perception -> Stage 3 Policy -> Stage 4 MuJoCo -> Stage 6 Verify")
    print("=" * 72)

    # 1. Acquire voice command
    command_text = args.command
    if command_text is None:
        if args.mic:
            command_text = record_live_microphone(seconds=args.seconds)
        elif args.audio and args.audio.is_file():
            print(f"Transcribing audio file: {args.audio}...")
            command_text = transcribe_audio(str(args.audio))
            print(f"[Audio Transcript]: \"{command_text}\"")
        else:
            default_audio = ROOT / "stage1_voice" / "samples" / "official_command.wav"
            if default_audio.is_file():
                print(f"No input provided; using sample audio: {default_audio.name}...")
                command_text = transcribe_audio(str(default_audio))
                print(f"[Audio Transcript]: \"{command_text}\"")
            else:
                command_text = EXAMPLE_COMMAND

    # 2. Execution mode
    executor_factory = None

    if args.view:
        import mujoco.viewer
        from scripts.visualize_run import ViewerSyncTrajectoryExecutor

        def make_viewer_executor(model, data, contact_audit=None):
            # Launch passive viewer if not already open
            if not hasattr(make_viewer_executor, "_viewer") or make_viewer_executor._viewer is None:
                make_viewer_executor._viewer = mujoco.viewer.launch_passive(model, data)
            return ViewerSyncTrajectoryExecutor(model, data, make_viewer_executor._viewer, contact_audit)

        executor_factory = make_viewer_executor

    elif args.record:
        from scripts.record_evaluation import RecordingTrajectoryExecutor, _make_camera, WIDTH, HEIGHT, FPS
        from stage4_bimanual.sim import MuJoCoSim
        import mujoco

        args.record.parent.mkdir(parents=True, exist_ok=True)
        rec_cls = RecordingTrajectoryExecutor
        rec_cls.renderer = None
        rec_cls.camera = _make_camera()
        rec_cls.frames = []

        def make_rec_executor(model, data, contact_audit=None):
            if rec_cls.renderer is None:
                rec_cls.renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
            return rec_cls(model, data, contact_audit=contact_audit)

        executor_factory = make_rec_executor

    # 3. Run pipeline end-to-end
    t0 = time.time()
    result = run_once(command_text, seed=args.seed, executor_factory=executor_factory)
    elapsed = time.time() - t0

    for line in result.log:
        print(line)

    if args.record and executor_factory and hasattr(RecordingTrajectoryExecutor, "frames") and RecordingTrajectoryExecutor.frames:
        print(f"\nSaving execution recording to {args.record}...")
        try:
            import imageio.v2 as imageio
            import numpy as np
            with imageio.get_writer(str(args.record), fps=30, codec="libx264", quality=8) as w:
                for f in RecordingTrajectoryExecutor.frames:
                    w.append_data(np.asarray(f))
            print(f"Video saved successfully! ({len(RecordingTrajectoryExecutor.frames)} frames)")
        except Exception as exc:
            print(f"Could not encode MP4 ({exc}); saving GIF...")
            gif_path = args.record.with_suffix(".gif")
            RecordingTrajectoryExecutor.frames[0].save(
                gif_path, save_all=True, append_images=RecordingTrajectoryExecutor.frames[1:],
                duration=int(1000 / 30), loop=0
            )
            print(f"GIF saved to {gif_path}")

    print("\n" + "=" * 72)
    if result.success:
        print(f"RESULT: SUCCESS (Completed in {elapsed:.1f}s)")
        print("Table successfully set by bimanual robot from voice instruction!")
        print("=" * 72)
        return 0
    else:
        print(f"RESULT: FAIL (after {result.attempts} attempts)")
        print("=" * 72)
        return 1


if __name__ == "__main__":
    sys.exit(main())
