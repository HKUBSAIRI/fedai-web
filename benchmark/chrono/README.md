# ChronoGPT pilot

Model weights: HuggingFace manelalab/chrono-gpt-instruct-v1-<YYYY>1231 (MIT). The official ChronoGPT_instruct.py casts activations to bfloat16;
on CPUs without bf16 support use a float32 copy (sed "s/\.bfloat16()/.float()/g" ChronoGPT_instruct.py > ChronoGPT_fp32.py), about 8x faster here.
Run: python chrono_pilot.py (model year 2015, tests the 2016 SEPs).
