# Hardware partner packs

One pack per customer/chip selection — not a universal backend matrix.
Start from `_template/`, replace the vendor command, and deliver:

- `postprocess.json` (from the exported artifact);
- target model + SDK/runtime version;
- target-side parity JSONL (≥20 images);
- latency/memory/power readings where available;
- vendor SDK license note.

See [docs/hardware_packs.md](../../docs/hardware_packs.md) for the contract
and boundaries.
