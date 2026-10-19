# Autovision training image (CUDA runtime base).
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime AS runtime

# Non-root runtime user (torch 2.5.1 / torchvision 0.20.1 from the base
# image already satisfy the pyproject floors, so `pip install .` keeps
# them instead of pulling a newer torch over the CUDA stack).
RUN useradd --create-home --shell /bin/bash autovision

ENV AUTOVISION_CACHE_DIR=/home/autovision/.cache/autovision \
    PIP_NO_CACHE_DIR=1
WORKDIR /workspace

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . && python -m compileall -q src/autovision

COPY --chmod=0755 docker/smoke_test.sh ./docker/smoke_test.sh
RUN chown -R autovision:autovision /workspace /home/autovision
USER autovision

# Smoke test: synthetic images -> pretrain -> artifacts asserted.
CMD ["bash", "docker/smoke_test.sh"]
