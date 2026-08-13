"""Serve an OpenPI policy for asynchronous RTC inference.

This is intentionally a separate entry point from ``scripts/serve_policy.py``.
The original server keeps its observation-only wire protocol, while this server
uses versioned request/response envelopes that can carry RTC guidance inputs.
"""

import asyncio
import dataclasses
import http
import logging
import time
import traceback
from typing import Any

from openpi_client import image_transport as _image_transport
from openpi_client import msgpack_numpy
import tyro
import websockets.asyncio.server as _server
import websockets.frames

from openpi.policies import diagnostics as _diagnostics
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

if __package__:
    from scripts import serve_policy as _serve_policy
else:
    import serve_policy as _serve_policy

PROTOCOL_NAME = "openpi-async-rtc"
PROTOCOL_VERSION = 2

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args(_serve_policy.Args):
    """Arguments for the asynchronous RTC policy server.

    All model, checkpoint, environment, diagnostics, and port arguments are
    inherited from ``scripts/serve_policy.py``.
    """

    asset_id: str | None = None
    """Override the norm-stats directory name under ``<checkpoint>/assets``."""


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dictionary, got {type(value).__name__}.")
    return value


def _policy_attribute(policy: Any, name: str, default: Any = None) -> Any:
    """Read a capability through transparent policy wrappers."""
    current = policy
    while current is not None:
        if hasattr(current, name):
            return getattr(current, name)
        current = getattr(current, "_policy", None)
    return default


def run_inference_request(policy: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Validate one protocol request and invoke the policy.

    Kept separate from the websocket handler so protocol behavior can be unit
    tested without a GPU or network socket.
    """
    if request.get("protocol") != PROTOCOL_NAME or request.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported protocol. Expected {PROTOCOL_NAME!r} version {PROTOCOL_VERSION}, "
            f"got {request.get('protocol')!r} version {request.get('protocol_version')!r}."
        )

    request_id = request.get("request_id")
    if not isinstance(request_id, int) or request_id < 0:
        raise ValueError(f"request_id must be a non-negative integer, got {request_id!r}.")

    wire_observation = _require_dict(request.get("observation"), "observation")
    transport = _require_dict(request.get("image_transport"), "image_transport")
    codec = transport.get("codec")
    if codec not in ("raw", "jpeg"):
        raise ValueError(f"Unsupported image transport codec: {codec!r}.")
    observation, image_transport_stats = _image_transport.decode_observation_images(
        wire_observation,
        codec=codec,
    )
    infer_kwargs: dict[str, Any] = {}
    rtc = request.get("rtc")
    if rtc is not None:
        rtc = _require_dict(rtc, "rtc")
        required_keys = {
            "prev_chunk_left_over",
            "inference_delay",
            "execution_horizon",
            "prefix_attention_schedule",
            "max_guidance_weight",
        }
        missing_keys = sorted(required_keys - rtc.keys())
        if missing_keys:
            raise KeyError(f"RTC request is missing keys: {missing_keys}")
        infer_kwargs = {
            "prev_chunk_left_over": rtc["prev_chunk_left_over"],
            "inference_delay": rtc["inference_delay"],
            "execution_horizon": rtc["execution_horizon"],
            "rtc_prefix_attention_schedule": rtc["prefix_attention_schedule"],
            "rtc_max_guidance_weight": rtc["max_guidance_weight"],
        }

    start_time = time.monotonic()
    result = policy.infer(observation, **infer_kwargs)
    infer_ms = (time.monotonic() - start_time) * 1000
    result["server_timing"] = {
        "image_decode_ms": image_transport_stats["decode_ms"],
        "infer_ms": infer_ms,
    }
    result["image_transport"] = image_transport_stats
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "result": result,
    }


class AsyncRTCPolicyServer:
    def __init__(self, policy: Any, host: str, port: int, metadata: dict[str, Any] | None = None) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = dict(metadata or {})
        self._metadata["async_rtc"] = {
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "model_action_horizon": _policy_attribute(policy, "action_horizon"),
            "model_action_dim": _policy_attribute(policy, "action_dim"),
            "rtc_supported": not _policy_attribute(policy, "_is_pytorch_model", default=False),
            "image_transport_codecs": ["raw", "jpeg"],
        }
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        while True:
            request_id: int | None = None
            try:
                request = _require_dict(msgpack_numpy.unpackb(await websocket.recv()), "request")
                raw_request_id = request.get("request_id")
                request_id = raw_request_id if isinstance(raw_request_id, int) else None
                response = run_inference_request(self._policy, request)
                await websocket.send(packer.pack(response))
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                error = traceback.format_exc()
                logger.exception("Async RTC inference request failed")
                await websocket.send(
                    packer.pack(
                        {
                            "protocol": PROTOCOL_NAME,
                            "protocol_version": PROTOCOL_VERSION,
                            "request_id": request_id,
                            "error": error,
                        }
                    )
                )
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="RTC inference failed. Traceback included in the previous frame.",
                )
                break


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main(args: Args) -> None:
    policy = _create_policy(args)
    metadata = policy.metadata

    if args.debug_policy_diagnostics:
        policy = _diagnostics.PolicyDiagnosticsWrapper(
            policy,
            interval=args.debug_policy_interval,
            max_steps=args.debug_policy_max_steps,
        )

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    server = AsyncRTCPolicyServer(policy, host="0.0.0.0", port=args.port, metadata=metadata)
    logger.info("Creating async RTC server on 0.0.0.0:%d", args.port)
    server.serve_forever()


def _create_policy(args: Args) -> Any:
    if args.asset_id is None:
        return _serve_policy.create_policy(args)

    if isinstance(args.policy, _serve_policy.Checkpoint):
        checkpoint = args.policy
    else:
        checkpoint = _serve_policy.DEFAULT_CHECKPOINT.get(args.env)
        if checkpoint is None:
            raise ValueError(f"Unsupported environment mode: {args.env}")

    train_config = _config.get_config(checkpoint.config)
    assets = dataclasses.replace(train_config.data.assets, asset_id=args.asset_id)
    train_config = dataclasses.replace(
        train_config,
        data=dataclasses.replace(train_config.data, assets=assets),
    )
    return _policy_config.create_trained_policy(
        train_config,
        checkpoint.dir,
        default_prompt=args.default_prompt,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
