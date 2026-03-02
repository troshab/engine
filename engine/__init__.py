from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["BpmEngine", "BpmResult", "TempoCnnEngine", "__version__"]

import argparse
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("engine")

_BPM_MIN = 30


@dataclass
class BpmResult:
    bpm: float
    confidence: float


class BpmEngine:
    _predictor = None
    _init_lock = threading.Lock()

    def __init__(self, device: str | None = None) -> None:
        self._device = device

    def _get_predictor(self):
        if BpmEngine._predictor is None:
            with BpmEngine._init_lock:
                if BpmEngine._predictor is None:
                    from deeprhythm import DeepRhythmPredictor

                    device = self._device
                    if device is None:
                        try:
                            import torch

                            device = "cuda" if torch.cuda.is_available() else "cpu"
                        except ImportError:
                            device = "cpu"
                    BpmEngine._predictor = DeepRhythmPredictor(device=device)
        return BpmEngine._predictor

    def _predict(
        self,
        clips,
        *,
        all_clips: bool = True,
        weighted_mean: bool = True,
        window: int = 2,
    ) -> BpmResult:
        import torch

        predictor = self._get_predictor()
        from deeprhythm.audio_proc.hcqm import compute_hcqm

        if not all_clips:
            clips = clips[:1]

        input_batch = compute_hcqm(
            clips.to(device=predictor.device), *predictor.specs
        ).permute(0, 3, 1, 2)

        predictor.model.eval()
        with torch.no_grad():
            outputs = predictor.model(input_batch.to(device=predictor.device))
            probs = torch.softmax(outputs, dim=1)
            mean_probs = probs.mean(dim=0).cpu().numpy()

        peak_idx = int(np.argmax(mean_probs))
        confidence = float(mean_probs[peak_idx])

        if weighted_mean:
            lo = max(0, peak_idx - window)
            hi = min(len(mean_probs), peak_idx + window + 1)
            window_probs = mean_probs[lo:hi]
            window_bpms = np.arange(lo, hi) + _BPM_MIN
            prob_sum = window_probs.sum()
            bpm = (
                float(np.dot(window_bpms, window_probs) / prob_sum)
                if prob_sum > 0
                else float(peak_idx + _BPM_MIN)
            )
        else:
            bpm = float(peak_idx + _BPM_MIN)

        return BpmResult(bpm=bpm, confidence=confidence)

    def analyze(
        self,
        filepath: str | Path,
        *,
        all_clips: bool = False,
        weighted_mean: bool = False,
        window: int = 2,
    ) -> BpmResult:
        from deeprhythm.utils import load_and_split_audio

        clips = load_and_split_audio(str(filepath), sr=22050)
        if clips is None:
            return BpmResult(bpm=0.0, confidence=0.0)
        return self._predict(
            clips, all_clips=all_clips, weighted_mean=weighted_mean, window=window
        )


_TC_SR = 11025
_TC_N_FFT = 1024
_TC_HOP_LENGTH = 512
_TC_N_MELS = 40
_TC_FMIN = 20.0
_TC_FMAX = 5000.0
_TC_PATCH_FRAMES = 256
_TC_PATCH_HOP = 128


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    freqs = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_bins))
    for i in range(n_mels):
        lo, mid, hi = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if mid > lo:
            fb[i, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fb[i, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return fb


def _melspectrogram(y, sr, n_fft, hop_length, n_mels, fmin, fmax, power):
    y = np.ascontiguousarray(y, dtype=np.float64)
    n_samples = len(y)
    if n_samples < n_fft:
        y = np.pad(y, (0, n_fft - n_samples))
        n_samples = n_fft
    n_frames = 1 + (n_samples - n_fft) // hop_length
    window = np.hanning(n_fft)
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_frames, n_fft),
        strides=(y.strides[0] * hop_length, y.strides[0]),
    ).copy()
    S = np.abs(np.fft.rfft(frames * window, axis=1)).T
    if power != 1.0:
        S = S ** power
    fb = _mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    return fb @ S


def _resample(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y
    n_samples = int(len(y) * target_sr / orig_sr)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float32)
    indices = np.linspace(0, len(y) - 1, n_samples)
    return np.interp(indices, np.arange(len(y)), y).astype(np.float32)


class TempoCnnEngine:
    _session = None
    _essentia_extractor = None
    _init_lock = threading.Lock()

    def __init__(
        self,
        *,
        model_path: str | None = None,
        backend: str | None = None,
        device: str | None = None,
        mel_bands: int = _TC_N_MELS,
        mel_fmin: float = _TC_FMIN,
        mel_fmax: float = _TC_FMAX,
        sample_rate: int = _TC_SR,
        hop_length: int = _TC_HOP_LENGTH,
        patch_frames: int = _TC_PATCH_FRAMES,
        patch_hop: int = _TC_PATCH_HOP,
        power: float = 1.0,
        normalize: bool = True,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._mel_bands = mel_bands
        self._mel_fmin = mel_fmin
        self._mel_fmax = mel_fmax
        self._sr = sample_rate
        self._hop_length = hop_length
        self._patch_frames = patch_frames
        self._patch_hop = patch_hop
        self._power = power
        self._normalize = normalize

        if backend:
            self._backend = backend
        elif model_path and Path(model_path).suffix.lower() == ".onnx":
            self._backend = "onnx"
        else:
            self._backend = "essentia"

    @staticmethod
    def is_available() -> bool:
        try:
            import essentia
            return True
        except ImportError:
            pass
        try:
            import onnxruntime
            return True
        except ImportError:
            pass
        return False

    def warmup(self) -> None:
        if self._backend == "onnx":
            self._get_onnx_session()
        else:
            self._get_essentia_extractor()

    def _get_essentia_extractor(self):
        if TempoCnnEngine._essentia_extractor is None:
            with TempoCnnEngine._init_lock:
                if TempoCnnEngine._essentia_extractor is None:
                    import essentia.standard as es

                    kw = {}
                    if self._model_path:
                        kw["graphFilename"] = str(self._model_path)
                    TempoCnnEngine._essentia_extractor = es.TempoCNN(**kw)
        return TempoCnnEngine._essentia_extractor

    def _get_onnx_session(self):
        if TempoCnnEngine._session is None:
            with TempoCnnEngine._init_lock:
                if TempoCnnEngine._session is None:
                    import onnxruntime as ort

                    if not self._model_path or not Path(self._model_path).exists():
                        raise FileNotFoundError(
                            f"model not found: {self._model_path}"
                        )
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    TempoCnnEngine._session = ort.InferenceSession(
                        str(self._model_path), providers=providers
                    )
        return TempoCnnEngine._session

    def analyze(
        self,
        filepath: str | Path,
        *,
        weighted_mean: bool = False,
        window: int = 2,
        confidence_window: int = 2,
    ) -> BpmResult:
        if self._backend == "onnx":
            return self._analyze_onnx(
                filepath,
                weighted_mean=weighted_mean,
                window=window,
                confidence_window=confidence_window,
            )
        return self._analyze_essentia(
            filepath,
            weighted_mean=weighted_mean,
            window=window,
            confidence_window=confidence_window,
        )

    def _analyze_essentia(
        self, filepath, *, weighted_mean, window, confidence_window
    ) -> BpmResult:
        import essentia.standard as es

        audio = es.MonoLoader(
            filename=str(filepath), sampleRate=float(self._sr)
        )()
        global_tempo, local_tempo, local_probs = self._get_essentia_extractor()(
            audio
        )

        bpm = float(global_tempo)
        confidence = 0.0

        if local_probs is not None and len(local_probs) > 0:
            avg_probs = np.array(local_probs)
            if avg_probs.ndim > 1:
                avg_probs = avg_probs.mean(axis=0)

            peak_idx = int(np.argmax(avg_probs))

            if weighted_mean and len(avg_probs) > 1:
                lo = max(0, peak_idx - window)
                hi = min(len(avg_probs), peak_idx + window + 1)
                wp = avg_probs[lo:hi]
                wb = np.arange(lo, hi) + _BPM_MIN
                s = float(wp.sum())
                if s > 0:
                    bpm = float(np.dot(wb, wp) / s)
            elif not weighted_mean:
                bpm = float(peak_idx + _BPM_MIN)

            clo = max(0, peak_idx - confidence_window)
            chi = min(len(avg_probs), peak_idx + confidence_window + 1)
            confidence = float(min(1.0, np.sum(avg_probs[clo:chi])))

        return BpmResult(
            bpm=bpm, confidence=round(max(0.0, min(1.0, confidence)), 4)
        )

    def _analyze_onnx(
        self, filepath, *, weighted_mean, window, confidence_window
    ) -> BpmResult:
        import soundfile as sf

        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
        y = data.mean(axis=1)
        y = _resample(y, sr, self._sr)

        mel = _melspectrogram(
            y,
            self._sr,
            _TC_N_FFT,
            self._hop_length,
            self._mel_bands,
            self._mel_fmin,
            self._mel_fmax,
            self._power,
        )
        mel = np.log(np.clip(mel, a_min=1e-7, a_max=None))

        n_frames = mel.shape[1]
        if n_frames < self._patch_frames:
            pad = np.zeros(
                (self._mel_bands, self._patch_frames - n_frames), dtype=mel.dtype
            )
            mel = np.concatenate([mel, pad], axis=1)
            n_frames = self._patch_frames

        patches = []
        for start in range(0, n_frames - self._patch_frames + 1, self._patch_hop):
            patch = mel[:, start : start + self._patch_frames].copy()
            if self._normalize:
                std = patch.std()
                if std > 1e-6:
                    patch = (patch - patch.mean()) / std
                else:
                    patch = patch - patch.mean()
            patches.append(patch)

        if not patches:
            return BpmResult(bpm=0.0, confidence=0.0)

        batch = np.stack(patches).astype(np.float32)[:, :, :, np.newaxis]

        session = self._get_onnx_session()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        preds = session.run([output_name], {input_name: batch})[0]

        avg_probs = preds.mean(axis=0)
        peak_idx = int(np.argmax(avg_probs))

        if weighted_mean:
            lo = max(0, peak_idx - window)
            hi = min(len(avg_probs), peak_idx + window + 1)
            wp = avg_probs[lo:hi]
            wb = np.arange(lo, hi) + _BPM_MIN
            s = float(wp.sum())
            bpm = (
                float(np.dot(wb, wp) / s) if s > 0 else float(peak_idx + _BPM_MIN)
            )
        else:
            bpm = float(peak_idx + _BPM_MIN)

        clo = max(0, peak_idx - confidence_window)
        chi = min(len(avg_probs), peak_idx + confidence_window + 1)
        confidence = float(min(1.0, np.sum(avg_probs[clo:chi])))

        return BpmResult(
            bpm=bpm, confidence=round(max(0.0, min(1.0, confidence)), 4)
        )


def _setup_logging(level_name: str = "info", verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else getattr(logging, level_name.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], default=None,
        help="compute device (default: auto-detect)",
    )
    parser.add_argument(
        "--model", metavar="PATH", default=None,
        help="path to custom model weights",
    )
    parser.add_argument(
        "--log-level", choices=["debug", "info", "warning", "error"],
        default="info", help="logging level (default: info)",
    )


def _add_engine_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("analysis parameters")
    bpm_mode = g.add_mutually_exclusive_group()
    bpm_mode.add_argument(
        "--weighted-mean", action="store_true", default=False,
        help="compute weighted-mean BPM around peak",
    )
    bpm_mode.add_argument(
        "--argmax", action="store_true", default=False,
        help="use argmax integer BPM (default)",
    )
    g.add_argument(
        "--window", type=int, default=2,
        help="BPM window half-width for weighted mean (default: 2)",
    )
    g.add_argument(
        "--min-confidence", type=float, default=0.0, metavar="FLOAT",
        help="minimum confidence threshold (default: 0.0)",
    )

    dr = parser.add_argument_group("deeprhythm options")
    clip_mode = dr.add_mutually_exclusive_group()
    clip_mode.add_argument(
        "--all-clips", action="store_true", default=False,
        help="use all audio clips for analysis",
    )
    clip_mode.add_argument(
        "--first-clip-only", action="store_true", default=False,
        help="use only the first clip (default)",
    )
    dr.add_argument(
        "--clip-duration", type=float, default=8.0, metavar="SEC",
        help="clip duration in seconds (default: 8.0)",
    )
    dr.add_argument(
        "--clip-overlap", type=float, default=0.0, metavar="FLOAT",
        help="clip overlap ratio 0.0-1.0 (default: 0.0)",
    )

    tc = parser.add_argument_group("tempocnn options")
    tc.add_argument(
        "--tempocnn-backend", choices=["essentia", "onnx"], default=None,
        help="backend (default: auto-detect from --model)",
    )
    tc.add_argument(
        "--mel-bands", type=int, default=40,
        help="mel filterbank bands (default: 40)",
    )
    tc.add_argument(
        "--mel-fmin", type=float, default=20.0,
        help="mel minimum frequency Hz (default: 20.0)",
    )
    tc.add_argument(
        "--mel-fmax", type=float, default=5000.0,
        help="mel maximum frequency Hz (default: 5000.0)",
    )
    tc.add_argument(
        "--sample-rate", type=int, default=11025,
        help="audio sample rate Hz (default: 11025)",
    )
    tc.add_argument(
        "--hop-length", type=int, default=512,
        help="STFT hop length (default: 512)",
    )
    tc.add_argument(
        "--patch-frames", type=int, default=256,
        help="spectrogram patch width in frames (default: 256)",
    )
    tc.add_argument(
        "--patch-hop", type=int, default=128,
        help="patch sliding window hop (default: 128)",
    )
    tc.add_argument(
        "--confidence-window", type=int, default=2,
        help="BPM bins for confidence summation (default: 2)",
    )
    tc.add_argument(
        "--power", type=float, default=1.0,
        help="mel spectrogram power (default: 1.0)",
    )
    tc.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=True,
        help="Z-normalize each spectrogram patch",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engine",
        description="CNN BPM analysis engine",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="verbose logging to stderr",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="analyze audio files (one-shot)")
    p_analyze.add_argument("files", nargs="+", metavar="FILE", help="audio files")
    p_analyze.add_argument(
        "--engine", choices=["deeprhythm", "tempocnn"], default="deeprhythm",
        help="engine to use (default: deeprhythm)",
    )
    p_analyze.add_argument(
        "--output-format", "--format", dest="output_format",
        choices=["jsonl", "text", "csv"], default="jsonl",
        help="output format (default: jsonl)",
    )
    p_analyze.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help="parallel file workers (default: 1)",
    )
    p_analyze.add_argument(
        "--timeout", type=float, default=300, metavar="SEC",
        help="per-file timeout in seconds (default: 300)",
    )
    _add_common_args(p_analyze)
    _add_engine_args(p_analyze)

    p_serve = sub.add_parser("serve", help="stdin/stdout JSONL server")
    p_serve.add_argument(
        "--batch-size", type=int, default=1, metavar="N",
        help="request buffer size (default: 1)",
    )
    p_serve.add_argument(
        "--warmup", action=argparse.BooleanOptionalAction, default=False,
        help="preload models on startup",
    )
    p_serve.add_argument(
        "--cache-audio", action=argparse.BooleanOptionalAction, default=False,
        help="cache decoded audio in memory",
    )
    _add_common_args(p_serve)
    _add_engine_args(p_serve)

    return parser


def _make_tempocnn(args):
    return TempoCnnEngine(
        model_path=args.model,
        backend=args.tempocnn_backend,
        device=args.device,
        mel_bands=args.mel_bands,
        mel_fmin=args.mel_fmin,
        mel_fmax=args.mel_fmax,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        patch_frames=args.patch_frames,
        patch_hop=args.patch_hop,
        power=args.power,
        normalize=args.normalize,
    )


def _format_result(fmt: str, filepath: str, result, engine_name: str) -> str:
    if fmt == "csv":
        return f"{filepath},{engine_name},{result.bpm:.2f},{result.confidence:.4f}"
    elif fmt == "text":
        return f"{filepath}: {result.bpm:.2f} BPM (confidence: {result.confidence:.4f})"
    return json.dumps({
        "file": filepath,
        "engine": engine_name,
        "bpm": round(result.bpm, 2),
        "confidence": round(result.confidence, 4),
    })


def _cmd_analyze(args: argparse.Namespace) -> None:
    fmt = args.output_format

    if fmt == "csv":
        print("file,engine,bpm,confidence", flush=True)

    if args.engine == "tempocnn":
        eng = _make_tempocnn(args)
        for filepath in args.files:
            try:
                result = eng.analyze(
                    filepath,
                    weighted_mean=args.weighted_mean,
                    window=args.window,
                    confidence_window=args.confidence_window,
                )
                print(
                    _format_result(fmt, filepath, result, "tempocnn"),
                    flush=True,
                )
            except Exception as e:
                if fmt == "jsonl":
                    print(
                        json.dumps({"file": filepath, "engine": "tempocnn", "error": str(e)}),
                        flush=True,
                    )
                else:
                    print(f"{filepath}: ERROR - {e}", file=sys.stderr, flush=True)
    else:
        eng = BpmEngine(device=args.device)
        for filepath in args.files:
            try:
                result = eng.analyze(
                    filepath,
                    all_clips=args.all_clips,
                    weighted_mean=args.weighted_mean,
                    window=args.window,
                )
                print(
                    _format_result(fmt, filepath, result, "deeprhythm"),
                    flush=True,
                )
            except Exception as e:
                if fmt == "jsonl":
                    print(
                        json.dumps({"file": filepath, "engine": "deeprhythm", "error": str(e)}),
                        flush=True,
                    )
                else:
                    print(f"{filepath}: ERROR - {e}", file=sys.stderr, flush=True)


def _cmd_serve(args: argparse.Namespace) -> None:
    jsonl_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    sys.stdout = sys.stderr

    dr_engine = BpmEngine(device=args.device)

    tc_engine = None
    if TempoCnnEngine.is_available():
        try:
            tc_engine = _make_tempocnn(args)
        except Exception as e:
            logger.warning("tempocnn init failed: %s", e)

    engines = ["deeprhythm"]
    if tc_engine is not None:
        engines.append("tempocnn")

    def _send(obj: dict) -> None:
        jsonl_out.write(json.dumps(obj) + "\n")
        jsonl_out.flush()

    if args.warmup:
        logger.info("serve: warming up engines")
        dr_engine._get_predictor()
        if tc_engine:
            try:
                tc_engine.warmup()
            except Exception as e:
                logger.warning("tempocnn warmup failed: %s", e)
                tc_engine = None
                engines = [x for x in engines if x != "tempocnn"]

    logger.info("serve: waiting for commands on stdin")

    stdin_bin = sys.stdin.buffer
    while True:
        raw_line = stdin_bin.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _send({"error": f"invalid JSON: {e}"})
            continue

        cmd = req.get("cmd", "")

        if cmd == "ping":
            dr_engine._get_predictor()
            if tc_engine:
                try:
                    tc_engine.warmup()
                except Exception as e:
                    logger.warning("tempocnn warmup failed: %s", e)
                    tc_engine = None
                    engines = [x for x in engines if x != "tempocnn"]
            _send({"status": "ready", "engines": engines})

        elif cmd == "shutdown":
            _send({"status": "bye"})
            break

        elif cmd == "load_model":
            size = req.get("size", 0)
            if size <= 0:
                _send({"error": "invalid size"})
                continue
            model_bytes = b""
            while len(model_bytes) < size:
                chunk = stdin_bin.read(size - len(model_bytes))
                if not chunk:
                    break
                model_bytes += chunk
            if len(model_bytes) != size:
                _send({"error": f"expected {size} bytes, got {len(model_bytes)}"})
                continue
            try:
                import onnxruntime as ort
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                TempoCnnEngine._session = ort.InferenceSession(model_bytes, providers=providers)
                if tc_engine is None:
                    tc_engine = _make_tempocnn(args)
                tc_engine._backend = "onnx"
                if "tempocnn" not in engines:
                    engines.append("tempocnn")
                _send({"status": "model_loaded"})
            except Exception as e:
                _send({"error": f"model load failed: {e}"})

        elif cmd == "analyze":
            filepath = req.get("file", "")
            if not filepath:
                _send({"error": "missing 'file' field"})
                continue

            engine_name = req.get("engine", "deeprhythm")

            try:
                if engine_name == "tempocnn":
                    if tc_engine is None:
                        _send({
                            "file": filepath,
                            "engine": "tempocnn",
                            "error": "tempocnn not available",
                        })
                        continue
                    result = tc_engine.analyze(
                        filepath,
                        weighted_mean=args.weighted_mean,
                        window=args.window,
                        confidence_window=args.confidence_window,
                    )
                else:
                    result = dr_engine.analyze(
                        filepath,
                        all_clips=args.all_clips,
                        weighted_mean=args.weighted_mean,
                        window=args.window,
                    )

                _send({
                    "file": filepath,
                    "engine": engine_name,
                    "bpm": round(result.bpm, 2),
                    "confidence": round(result.confidence, 4),
                })
            except Exception as e:
                _send({
                    "file": filepath,
                    "engine": engine_name,
                    "error": str(e),
                })

        else:
            _send({"error": f"unknown command: {cmd!r}"})

    jsonl_out.close()
    logger.info("serve: shutting down")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _setup_logging(
        level_name=getattr(args, "log_level", "info"),
        verbose=args.verbose,
    )

    if args.command == "analyze":
        _cmd_analyze(args)
    elif args.command == "serve":
        _cmd_serve(args)


