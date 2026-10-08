# ONKOS Mimarisi

5 katmanlı yığın: Frontend (React/Next.js) → API (FastAPI) → AI model katmanı (PyTorch, MONAI,
HuggingFace) → Veri (PostgreSQL, MongoDB, DVC) → Altyapı (bulut GPU, Docker).

## Veri akışı

```mermaid
flowchart LR
    WSI[H&E WSI] --> HISTO[histo: ViT + ABMIL]
    OMICS_IN[TCGA RNA-seq / omics] --> OMICS[omics: Encoder + Cross-Attention]
    HISTO -- patch features --> OMICS
    OMICS -- mutation scores --> GNN[gnn: ilaç önerisi]
    OMICS -- mutation scores --> RAG[rag_llm: rapor]
    GNN -- drug ranking --> RAG
    GNN -- drug embedding --> DIFF[diffusion: kavramsal simülasyon]
    HISTO -- tumor patches --> DIFF
    HISTO -- attention --> XAI[xai: GradCAM / heatmap]
    OMICS -- fusion attention --> XAI
    RAG --> API[api: FastAPI]
    DIFF --> API
    XAI --> API
    GNN --> API
    OMICS --> API
    API --> WEB[apps/web]
```

## Modül bağımlılık grafiği

Modüller birbirini import etmez; yalnızca ortak paketlere bağlanır (ADR-0001).

```mermaid
flowchart TD
    histo --> contracts
    omics --> contracts
    gnn --> contracts
    rag_llm --> contracts
    diffusion --> contracts
    xai --> contracts
    api --> contracts
    histo --> common
    omics --> common
    gnn --> common
    rag_llm --> common
    diffusion --> common
    xai --> common
    api --> common
```

Çalışma zamanında modüller `contracts` içindeki şemalarla (HDF5 özellik dosyaları, JSON/Pydantic
nesneleri) konuşur; `api` bunları HTTP'ye çevirir. Sözleşme ayrıntıları mini sprint 1.7'de.
