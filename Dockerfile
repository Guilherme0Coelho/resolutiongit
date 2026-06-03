# =============================================================================
# Stage 1: Build the Rust binary
# =============================================================================
FROM rust:1.82-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src/ ./src/

# x86-64-v3 enables AVX2 SIMD for auto-vectorization of the distance loop
ENV RUSTFLAGS="-C opt-level=3"
RUN cargo build --release

# =============================================================================
# Stage 2: Download dataset & preprocess to compact binary
# =============================================================================
FROM builder AS preprocessor

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data

# Download the 3M references dataset (~16MB gzipped)
RUN curl -L -o /data/references.json.gz \
    "https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources/references.json.gz"

# Convert JSON.gz → compact binary (3M × 14 × f32 + labels)
# This runs during build so containers start instantly
RUN /app/target/release/fraud-detector --preprocess \
    --input /data/references.json.gz \
    --output /data/references.bin && \
    rm /data/references.json.gz

# =============================================================================
# Stage 3: Minimal runtime image
# =============================================================================
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/fraud-detector /usr/local/bin/fraud-detector
COPY --from=preprocessor /data/references.bin /data/references.bin
COPY resources/normalization.json /data/normalization.json
COPY resources/mcc_risk.json /data/mcc_risk.json

ENV DATA_DIR=/data
ENV PORT=8080

EXPOSE 8080

CMD ["fraud-detector"]
