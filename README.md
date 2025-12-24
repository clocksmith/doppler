# DOPPLER

**Distributed On-device Pipeline Processing Large Embedded Reploid**

Browser-native LLM inference engine powered by WebGPU.

**[Try it live at replo.id/d](https://replo.id/d)** | **[GitHub](https://github.com/clocksmith/doppler)**

Project source is in `reploid/doppler/`. Root-level `AGENTS.md`, `CLAUDE.md`, and `EMOJI.md` are symlinked there.

See the main [README](reploid/doppler/README.md) for full documentation.

## Quick Start

```bash
cd reploid/doppler
npm install
npm start         # Dev server at http://localhost:8080
npm run bench     # Run benchmarks
```

## Related

- [REPLOID](https://github.com/clocksmith/reploid) - Browser-native AI agent ([replo.id/r](https://replo.id/r))

## Inspiration

- [WebLLM](https://github.com/mlc-ai/web-llm) - High-performance in-browser LLM inference
- [PyTorch](https://pytorch.org/) - Machine learning framework
- [WebGPU](https://www.w3.org/TR/webgpu/) - W3C GPU API specification
- [Mistral 7B](https://arxiv.org/abs/2310.06825) - Sliding window attention, grouped-query attention
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) - Sparse Mixture of Experts architecture
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) - Expert specialization in MoE
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) - Multi-head Latent Attention, 671B MoE
- [Kimi K2](https://arxiv.org/abs/2507.20534) - 1T parameter MoE, agentic intelligence
- [Dr. Doppler](https://megaman.fandom.com/wiki/Dr._Doppler) - Mega Man X3
