<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/header.svg">
  <img alt="S4F3 3TCH — Open-source ComfyUI custom nodes by Itasha Corp" src=".github/assets/header.svg" width="100%">
</picture>

*Image generation nodes for the network. Every pixel etched with precision.*

[![Python](https://img.shields.io/badge/python-3.10+-00FFFF.svg?style=flat-square)](https://www.python.org/downloads/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-nodes-FF00FF.svg?style=flat-square)](https://github.com/comfyanonymous/ComfyUI)
[![License](https://img.shields.io/badge/license-Apache_2.0-00FFFF.svg?style=flat-square)](LICENSE)
[![Workflows](https://img.shields.io/badge/workflows-18_templates-00FFFF.svg?style=flat-square)](#-workflows)
[![Open Source](https://img.shields.io/badge/open_source-01fe36.svg?style=flat-square)](#-contributing)

---

[**MODULES**](#-modules) · [**INSTALL**](#-installation) · [**WORKFLOWS**](#-workflows) · [**ARCHITECTURE**](#-architecture) · [**CONTRIBUTING**](#-contributing)

</div>

---

## > Overview

**S4F3 3TCH** is a collection of open-source tools for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — node authoring, workflow validation, quality optimization, publishing pipelines, and production-ready workflow templates. Built by **Itasha Corp**.

```
  idea                    3TCH pipeline                    output
  ┌──────────┐    ┌─────────────────────────────┐    ┌──────────┐
  │ "blur     │    │  authoring → validation     │    │ blur.py  │
  │  node,    │───►│  quality   → publishing     │───►│ (ComfyUI │
  │  0-100"   │    │  schema    → registry       │    │  node)   │
  └──────────┘    └─────────────────────────────┘    └──────────┘
```

---

## > Modules

### Authoring

Generate ComfyUI custom nodes from natural language specifications.

```python
from authoring.types import NodeSpec
from authoring.generator import generate_node

# Describe your node in plain English
spec = NodeSpec.from_natural_language("takes an image and a blur strength (0-100), outputs blurred image")
code = generate_node(spec)  # Produces valid ComfyUI node Python
```

| Feature | Description |
|---------|-------------|
| Natural language parser | Describe nodes in English, get Pydantic specs |
| Code generator | Jinja2 templates for V1 (legacy) and V3 (stateless) formats |
| Composite nodes | Combine multiple operations into single nodes |
| Layout optimizer | UI layout for node input/output positioning |
| Test harness | Validate generated nodes with dynamic import testing |
| Security scanner | AST-based blocking of dangerous patterns (eval, os.system) |

### Publishing

Package, validate, and publish custom nodes to the ComfyUI ecosystem.

| Feature | Description |
|---------|-------------|
| Scaffolder | Generate project structure for new node packages |
| Registry | ComfyUI Manager registry entry generation and validation |
| Versioning | Semantic versioning with changelog generation |
| Security | Security validation for published nodes |

Supports both legacy ComfyUI Manager format and the new Comfy Registry.

### Control

High-level API for ComfyUI server and model management.

| Feature | Description |
|---------|-------------|
| Model manager | Listing, architecture detection, merging, checksums |
| Batch processing | Parameter sweeps, seed sweeping, concurrent queue |
| Server lifecycle | Start/stop/restart/health monitoring |
| Performance | Optimization utilities and benchmarking |
| Quantization | Model quantization for VRAM optimization |
| Templates | Workflow template CRUD with variable rendering |

### Quality

Image quality optimization and prompt enhancement.

| Feature | Description |
|---------|-------------|
| Prompt enhancer | Optimize prompts for better generation results |
| Upscaler | Image upscaling with model-aware settings |
| Presets | Quality presets for different output targets |
| Workflow builder | High-level workflow construction API |

### Schema

ComfyUI workflow JSON validation and format conversion.

| Feature | Description |
|---------|-------------|
| Validator | Schema validation for V0 and V1 workflow formats |
| Converters | Format conversion between workflow versions |
| CLI | Command-line validation tool |

```bash
python -m schema.cli validate workflow.json
```

---

## > Workflows

Production-tested workflow templates ready for use:

| Workflow | Description |
|----------|-------------|
| `txt2img.json` | Text-to-image base template |
| `flux-txt2img.json` | Flux model text-to-image |
| `flux2-txt2img.json` | Flux 2 text-to-image |
| `flux-controlnet-union.json` | Flux + ControlNet union |
| `flux-pulid.json` | Flux + PuLID face transfer |
| `ip-adapter-style.json` | IP-Adapter style transfer |
| `4x-upscale.json` | 4x image upscaling |
| `character-consistency.json` | Consistent character generation |
| `brand-character-base.json` | Brand character templates |
| `brand-badge-generator.json` | Badge and asset generation |
| `controlnet-pose.json` | ControlNet pose guidance |
| `manga-page-batch.json` | Manga page batch generation |
| `manga-panel-single.json` | Single manga panel |
| `sticker-batch.json` | Batch sticker production |
| `marketing-hero.json` | Marketing hero images |
| `sdxl-optimized-txt2img.json` | Optimized SDXL pipeline |

---

## > Installation

```bash
git clone https://github.com/46b-ETYKiAL/Itasha.Corp_S4F3-3TCH.git
cd Itasha.Corp_S4F3-3TCH
pip install -e .
```

### Dependencies

```bash
pip install pydantic jinja2
```

Each module can also be used standalone — just copy the directory you need.

---

## > Architecture

```
s4f3-etch/
├── authoring/              # Node generation from specs
│   ├── types.py            # NodeSpec, InputSpec, WidgetType
│   ├── generator.py        # Jinja2 code generation
│   ├── composite.py        # Multi-op node composition
│   └── test_harness.py     # Dynamic import validation
├── publishing/             # Package and publish nodes
│   ├── scaffolder.py       # Project structure generation
│   ├── registry.py         # ComfyUI Manager integration
│   └── versioning.py       # Semver + changelogs
├── control/                # Server and model management
│   ├── model_manager.py    # Model ops (list, merge, quantize)
│   ├── batch.py            # Concurrent batch generation
│   └── server_lifecycle.py # ComfyUI server control
├── quality/                # Output optimization
│   ├── prompt_enhancer.py  # Prompt refinement
│   └── upscaler.py         # Image upscaling
├── schema/                 # Workflow validation
│   ├── validator.py        # V0/V1 schema validation
│   ├── converters.py       # Format conversion
│   └── cli.py              # Command-line interface
├── workflows/              # Production JSON templates
│   ├── txt2img.json
│   ├── flux-txt2img.json
│   └── ...                 # 18 templates total
├── pyproject.toml
└── LICENSE                 # Apache 2.0
```

---

## > Contributing

Contributions welcome. Please:

1. Fork the repository
2. Create a feature branch (`feat/your-feature`)
3. Write tests for new functionality
4. Ensure all existing tests pass
5. Submit a PR with a clear description

### Development

```bash
git clone https://github.com/46b-ETYKiAL/Itasha.Corp_S4F3-3TCH.git
cd Itasha.Corp_S4F3-3TCH
pip install -e ".[dev]"
```

---

## > Related

| Repo | Description |
|------|-------------|
| [S4F3 R0UT3 4RB1T3R](https://github.com/46b-ETYKiAL/Itasha.Corp_S4F3-R0UT3-4RB1T3R) | Multi-agent orchestration system |
| [S4F3 R3L4Y](https://github.com/46b-ETYKiAL/Itasha.Corp_S4F3-R3L4Y) | Open-source MCP servers |
| [S4F3 SH3LL](https://github.com/46b-ETYKiAL/Itasha.Corp_S4F3-SH3LL) | AI coding CLI |

---

## > License

[Apache License 2.0](LICENSE) — Itasha Corp, 2026.

<div align="center">

```
  ┌──────────────────────────────────────────┐
  │                                          │
  │   every pixel etched with precision.     │
  │                                          │
  │   ░░░ operator23a is watching ░░░        │
  │                                          │
  └──────────────────────────────────────────┘
```

</div>
