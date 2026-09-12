import csv
import json

import numpy as np

from examples.xtrainer_real import inference_action_recorder


def test_recorder_saves_received_selected_and_plot_data(tmp_path):
    recorder = inference_action_recorder.InferenceActionRecorder(
        tmp_path,
        mode="async_rtc",
        plot_start_index=7,
        config={"rtc_enabled": True, "action_horizon": 2},
    )
    first_chunk = np.arange(42, dtype=np.float32).reshape(3, 14)
    second_chunk = first_chunk + 100
    recorder.record_received_chunk(
        first_chunk,
        phase="baseline_warmup",
        request_id=0,
        episode_id=0,
        request_step=0,
        arrival_step=0,
        actual_delay_steps=0,
        rtc_enabled=False,
        installed=True,
        round_trip_ms=20.0,
        server_infer_ms=15.0,
    )
    recorder.record_received_chunk(
        second_chunk,
        phase="online",
        request_id=1,
        episode_id=0,
        request_step=2,
        arrival_step=3,
        actual_delay_steps=1,
        rtc_enabled=True,
        installed=True,
        round_trip_ms=25.0,
        server_infer_ms=18.0,
    )
    recorder.record_selected_action(first_chunk[0], episode_id=0, step=0, request_id=0, chunk_index=0)
    recorder.record_selected_action(first_chunk[1], episode_id=0, step=1, request_id=0, chunk_index=1)
    recorder.record_selected_action(second_chunk[1], episode_id=0, step=2, request_id=1, chunk_index=1)

    output_dir = recorder.save()

    assert recorder.save() == output_dir
    assert (output_dir / "actions.png").is_file()
    with np.load(output_dir / "actions.npz") as data:
        np.testing.assert_array_equal(data["received_actions"], np.stack([first_chunk, second_chunk]))
        np.testing.assert_array_equal(
            data["selected_actions"],
            np.stack([first_chunk[0], first_chunk[1], second_chunk[1]]),
        )
        np.testing.assert_array_equal(data["received_actual_delay_steps"], [0, 1])

    with (output_dir / "received_chunks.csv").open(encoding="utf-8") as file_obj:
        received_rows = list(csv.DictReader(file_obj))
    assert len(received_rows) == 6
    assert received_rows[-1]["rtc_enabled"] == "True"

    with (output_dir / "summary.json").open(encoding="utf-8") as file_obj:
        summary = json.load(file_obj)
    assert summary["received_chunk_count"] == 2
    assert summary["selected_action_count"] == 3
    assert summary["switch_count"] == 1
    assert summary["boundary_action_jump_max"] > 0


def test_recorder_keeps_discarded_warmup_chunk_but_does_not_count_it_as_installed(tmp_path):
    recorder = inference_action_recorder.InferenceActionRecorder(
        tmp_path,
        mode="async_rtc",
        plot_start_index=0,
        config={},
    )
    recorder.record_received_chunk(
        np.zeros((2, 2), dtype=np.float32),
        phase="rtc_warmup",
        request_id=1,
        episode_id=0,
        request_step=0,
        arrival_step=0,
        actual_delay_steps=0,
        rtc_enabled=True,
        installed=False,
        round_trip_ms=10.0,
        server_infer_ms=8.0,
    )

    output_dir = recorder.save()

    with (output_dir / "summary.json").open(encoding="utf-8") as file_obj:
        summary = json.load(file_obj)
    assert summary["received_chunk_count"] == 1
    assert summary["installed_chunk_count"] == 0
    assert summary["selected_action_count"] == 0
    assert not (output_dir / "actions.png").exists()
