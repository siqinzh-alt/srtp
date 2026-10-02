# VLM Fine-tuning Dataset Report

## Config

- samples_per_class: `1000`
- qa_per_task: `1000`
- sample_rate_hz: `16000`
- duration_s: `2.0`
- image_kind: `diagnostic_triplet`

## Sample Distribution

- total samples: `4000`
- by signal type: `{'BPSK': 1000, 'CW': 1000, 'FSK': 1000, 'LFM': 1000}`
- by split: `{'test': 400, 'train': 3198, 'val': 402}`
- by condition: `{'clean/bellhop': 400, 'clean/direct': 2800, 'real_noise_heavy/direct': 400, 'real_noise_light/direct': 400}`

## QA Distribution

- total QA: `4996`
- by task: `{'bpsk_symbol_rate_choice': 1000, 'fsk_order_choice': 1000, 'lfm_direction_choice': 1000, 'main_frequency_choice': 996, 'signal_type_choice': 1000}`
- by split: `{'test': 496, 'train': 3998, 'val': 502}`

## Answer Distribution By Task

- signal_type_choice: `{'BPSK': 250, 'CW': 250, 'FSK': 250, 'LFM': 250}`
- main_frequency_choice: `{'1000 Hz': 166, '2000 Hz': 166, '250 Hz': 166, '4000 Hz': 166, '500 Hz': 166, '6000 Hz': 166}`
- lfm_direction_choice: `{'升频': 500, '降频': 500}`
- fsk_order_choice: `{'2FSK': 500, '4FSK': 500}`
- bpsk_symbol_rate_choice: `{'20 symbol/s': 250, '25 symbol/s': 250, '40 symbol/s': 250, '50 symbol/s': 250}`

## Validation

- OK: all validation checks passed.
