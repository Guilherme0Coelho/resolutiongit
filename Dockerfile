# Stage 1: Build
FROM rust:1.85-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src/ ./src/
ENV RUSTFLAGS="-C target-cpu=x86-64-v3 -C opt-level=3"
RUN cargo build --release

# Stage 2: Preprocess dataset
FROM builder AS preprocessor
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /data
RUN curl -L -o /data/references.json.gz \
    "https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources/references.json.gz"
RUN /app/target/release/fraud-detector --preprocess \
    --input /data/references.json.gz \
    --output /data/references.bin && \
    rm /data/references.json.gz

# Stage 3: Minimal runtime
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/fraud-detector /usr/local/bin/fraud-detector
COPY --from=preprocessor /data/references.bin /data/references.bin
COPY resources/normalization.json /data/normalization.json
COPY resources/mcc_risk.json /data/mcc_risk.json
ENV DATA_DIR=/data
ENV PORT=9999
EXPOSE 9999
CMD ["fraud-detector"]
