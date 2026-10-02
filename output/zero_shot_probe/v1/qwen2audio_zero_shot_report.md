# Qwen2-Audio Zero-shot Probe Report

## Setup

- Model: `Qwen/Qwen2-Audio-7B-Instruct`
- Dataset: `output/zero_shot_probe/v1`
- Evaluation items: `222`
- Conditions: `direct/bellhop × clean/real_noise_light/real_noise_heavy`
- Result files: `qwen2audio_predictions.jsonl`, `qwen2audio_summary.json`

## Accuracy

- `overall:overall`: 71/222 = 32.0%
- `task:signal_type_choice`: 31/126 = 24.6%
- `task:main_frequency_choice`: 9/36 = 25.0%
- `task:lfm_direction_choice`: 18/36 = 50.0%
- `task:fsk_order_choice`: 13/24 = 54.2%
- `noise:clean`: 27/74 = 36.5%
- `noise:real_noise_light`: 23/74 = 31.1%
- `noise:real_noise_heavy`: 21/74 = 28.4%
- `channel:direct`: 37/111 = 33.3%
- `channel:bellhop`: 34/111 = 30.6%

## Task × Noise

- `task_noise:fsk_order_choice|clean`: 6/8 = 75.0%
- `task_noise:fsk_order_choice|real_noise_heavy`: 4/8 = 50.0%
- `task_noise:fsk_order_choice|real_noise_light`: 3/8 = 37.5%
- `task_noise:lfm_direction_choice|clean`: 6/12 = 50.0%
- `task_noise:lfm_direction_choice|real_noise_heavy`: 6/12 = 50.0%
- `task_noise:lfm_direction_choice|real_noise_light`: 6/12 = 50.0%
- `task_noise:main_frequency_choice|clean`: 4/12 = 33.3%
- `task_noise:main_frequency_choice|real_noise_heavy`: 1/12 = 8.3%
- `task_noise:main_frequency_choice|real_noise_light`: 4/12 = 33.3%
- `task_noise:signal_type_choice|clean`: 11/42 = 26.2%
- `task_noise:signal_type_choice|real_noise_heavy`: 10/42 = 23.8%
- `task_noise:signal_type_choice|real_noise_light`: 10/42 = 23.8%

## Task × Channel

- `task_channel:fsk_order_choice|bellhop`: 5/12 = 41.7%
- `task_channel:fsk_order_choice|direct`: 8/12 = 66.7%
- `task_channel:lfm_direction_choice|bellhop`: 9/18 = 50.0%
- `task_channel:lfm_direction_choice|direct`: 9/18 = 50.0%
- `task_channel:main_frequency_choice|bellhop`: 5/18 = 27.8%
- `task_channel:main_frequency_choice|direct`: 4/18 = 22.2%
- `task_channel:signal_type_choice|bellhop`: 15/63 = 23.8%
- `task_channel:signal_type_choice|direct`: 16/63 = 25.4%

## Response Patterns

### `fsk_order_choice`
- Accuracy: 13/24 = 54.2%
- Expected distribution: `2FSK`=12, `4FSK`=12
- Top responses: `2FSK`=17, `4FSK`=7
- Example wrong cases:
  - `audio/probe_000080.wav` `direct/real_noise_light` expected `2FSK`, got `4FSK`
  - `audio/probe_000081.wav` `direct/real_noise_heavy` expected `2FSK`, got `4FSK`
  - `audio/probe_000084.wav` `bellhop/real_noise_heavy` expected `2FSK`, got `4FSK`
  - `audio/probe_000086.wav` `direct/real_noise_light` expected `4FSK`, got `2FSK`
  - `audio/probe_000088.wav` `bellhop/clean` expected `4FSK`, got `2FSK`

### `lfm_direction_choice`
- Accuracy: 18/36 = 50.0%
- Expected distribution: `升频`=18, `降频`=18
- Top responses: `升频`=36
- Example wrong cases:
  - `audio/probe_000043.wav` `direct/clean` expected `降频`, got `升频`
  - `audio/probe_000044.wav` `direct/real_noise_light` expected `降频`, got `升频`
  - `audio/probe_000045.wav` `direct/real_noise_heavy` expected `降频`, got `升频`
  - `audio/probe_000046.wav` `bellhop/clean` expected `降频`, got `升频`
  - `audio/probe_000047.wav` `bellhop/real_noise_light` expected `降频`, got `升频`

### `main_frequency_choice`
- Accuracy: 9/36 = 25.0%
- Expected distribution: `250 Hz`=6, `500 Hz`=6, `1000 Hz`=6, `2000 Hz`=6, `4000 Hz`=6, `6000 Hz`=6
- Top responses: `500 Hz`=17, `6000 Hz`=15, `1000 Hz`=3, `2000 Hz`=1
- Example wrong cases:
  - `audio/probe_000001.wav` `direct/clean` expected `250 Hz`, got `500 Hz`
  - `audio/probe_000002.wav` `direct/real_noise_light` expected `250 Hz`, got `500 Hz`
  - `audio/probe_000003.wav` `direct/real_noise_heavy` expected `250 Hz`, got `500 Hz`
  - `audio/probe_000004.wav` `bellhop/clean` expected `250 Hz`, got `500 Hz`
  - `audio/probe_000005.wav` `bellhop/real_noise_light` expected `250 Hz`, got `1000 Hz`

### `signal_type_choice`
- Accuracy: 31/126 = 24.6%
- Expected distribution: `CW`=36, `LFM`=36, `BPSK`=30, `FSK`=24
- Top responses: `BPSK`=125, `CW`=1
- Example wrong cases:
  - `audio/probe_000001.wav` `direct/clean` expected `CW`, got `BPSK`
  - `audio/probe_000002.wav` `direct/real_noise_light` expected `CW`, got `BPSK`
  - `audio/probe_000003.wav` `direct/real_noise_heavy` expected `CW`, got `BPSK`
  - `audio/probe_000004.wav` `bellhop/clean` expected `CW`, got `BPSK`
  - `audio/probe_000005.wav` `bellhop/real_noise_light` expected `CW`, got `BPSK`

