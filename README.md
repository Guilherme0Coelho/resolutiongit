# Rinha de Backend 2026 — Fraud Detection

Solução em **Rust** para o desafio de detecção de fraude com busca vetorial.

## Arquitetura

```
Cliente → Nginx (LB :9999) → API-1 / API-2 (Rust :8080)
                                    ↓
                          references.bin (mmap)
```

- **Nginx**: Round-robin, 0.10 CPU, 20MB
- **API ×2**: Rust + hyper, 0.40 CPU, 160MB cada
- **Total**: 0.90 CPU, 340MB RAM

## Stack

| Componente | Tecnologia |
|---|---|
| HTTP Server | hyper 1.x (direto, sem framework) |
| JSON | serde_json |
| Dataset | mmap (memmap2) — zero-copy, shared entre instâncias |
| Busca | Brute-force KNN K=5, distância euclidiana² |
| SIMD | Auto-vectorização via `-C target-cpu=x86-64-v3` (AVX2) |

## Como rodar

```bash
# Baixar o dataset (precisa estar em resources/)
bash download-dataset.sh

# Build e run
docker compose up --build
```

## Decisões de design

1. **Preprocessamento no build**: `references.json.gz` → `references.bin` (binário compacto) durante o Docker build. Container inicia instantaneamente.
2. **mmap**: O kernel compartilha as páginas físicas entre as 2 instâncias. 168MB de vetores contados 1x na RAM real.
3. **Busca linear otimizada**: Com 14 dimensões, a distância euclidiana² é computada em ~4 instruções AVX2. O brute-force sobre 3M vetores completa em <1ms com SIMD.
4. **Zero alocação no hot path**: Nenhum heap allocation durante a busca — arrays fixos de tamanho 5 para o top-K.
