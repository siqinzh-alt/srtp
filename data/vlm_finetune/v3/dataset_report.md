# VLM Fine-tuning Dataset Report

## Config

- base_dataset: `data\vlm_finetune\v1` (filtered to non-Animal samples before rewrite)
- animal_samples_requested: `1000`
- animal_qa_items_requested: `1250`
- sample_rate_hz: `16000`
- duration_s: `2.0`
- max_freq_hz: `8000.0`
- source_dataset: `Watkins Marine Mammal Sound Database`
- source_data_page: `https://marine-mammal.soundwave.cl/data.html`
- image_kind: `diagnostic_triplet`
- field_count_weights: `{1: 0.65, 2: 0.25, 3: 0.1}`
- single_format_weights: `{'choice': 0.35, 'open_short': 0.65}`
- animal_field_weights: `{'bio_species_or_group': 0.77, 'bio_species_code': 0.23}`

## Sample Distribution

- total samples: `5000`
- base synthetic samples: `4000`
- animal samples: `1000`
- by signal type: `{'Animal': 1000, 'BPSK': 1000, 'CW': 1000, 'FSK': 1000, 'LFM': 1000}`
- by image kind: `{'diagnostic_triplet': 5000}`
- by split: `{'test': 487, 'train': 4002, 'val': 511}`
- animal by species: `{'Balaena mysticetus': 48, 'Balaenoptera physalus': 55, 'Delphinus delphis': 53, 'Eubalaena glacialis': 45, 'Globicephala macrorhynchus': 44, 'Globicephala melaena': 47, 'Grampus griseus': 53, 'Lagenodelphis hosei': 55, 'Lagenorhynchus acutus': 47, 'Lagenorhynchus albirostris': 52, 'Megaptera novaeangliae': 52, 'Odobenus rosmarus': 50, 'Orcinus orca': 45, 'Physeter catodon': 48, 'Pseudorca crassidens': 48, 'Stenella attenuata': 47, 'Stenella clymene': 50, 'Stenella coeruleoalba': 51, 'Stenella frontalis': 54, 'Stenella longirostris': 56}`

## QA Distribution

- total QA: `5937`
- animal QA: `937`
- by task: `{'bio_species_code_choice': 65, 'bio_species_code_open_short': 106, 'bio_species_or_group_choice': 220, 'bio_species_or_group_open_short': 359, 'bpsk_symbol_rate_choice': 119, 'bpsk_symbol_rate_open_short': 221, 'channel_condition_choice': 101, 'channel_condition_open_short': 188, 'fsk_order_choice': 117, 'fsk_order_open_short': 218, 'lfm_direction_choice': 118, 'lfm_direction_open_short': 218, 'main_frequency_choice': 253, 'main_frequency_open_short': 469, 'multi_field_open_json': 1937, 'noise_level_choice': 152, 'noise_level_open_short': 282, 'signal_type_choice': 278, 'signal_type_open_short': 516}`
- by answer format: `{'choice': 1423, 'open_json': 1937, 'open_short': 2577}`
- by field count: `{1: 4000, 2: 1437, 3: 500}`
- by field occurrence: `{'bio_species_code': 358, 'bio_species_or_group': 766, 'bpsk_symbol_rate': 588, 'channel_condition': 1009, 'fsk_order': 554, 'lfm_direction': 566, 'main_frequency_hz': 908, 'noise_level': 1451, 'signal_type': 2174}`
- by QA reuse per image: `{1: 3613, 2: 1132, 3: 20}`
- by split: `{'test': 571, 'train': 4758, 'val': 608}`

## Validation

- OK: all validation checks passed.
