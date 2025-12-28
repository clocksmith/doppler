# [REPLOID](https://github.com/clocksmith/reploid)

**R**ecursive **E**volution **P**rotocol **L**oop **O**rchestrating **I**nference **D**oppler

This directory is a namespace placeholder linking to the sibling project.

```
┌─────────────────────────────────┐
│           REPLOID               │  Browser-native AI agent
│  github.com/clocksmith/reploid  │  with recursive self-improvement
└─────────────────────────────────┘
                ↓ uses
┌─────────────────────────────────┐
│           DOPPLER               │  WebGPU inference engine
│  ./doppler/                     │  ← Actual code is here
└─────────────────────────────────┘
```

## Structure

```
doppler/                          ← You are here (DOPPLER repo)
├── README.md                     ← Root pointer
├── reploid/                      ← Namespace (this directory)
│   ├── README.md                 ← This file
│   └── doppler/                  ← Actual DOPPLER package
│       ├── README.md             ← Full documentation
│       ├── package.json
│       └── ...
```

## Why This Structure?

DOPPLER and REPLOID reference each other in their names:

- **DOPPLER** = Distributed On-device Pipeline Processing Large Embedded **Reploid**
- **REPLOID** = Recursive Evolution Protocol Loop Orchestrating Inference **Doppler**

The mirrored directory structure reflects this relationship. Each repo contains a namespace for the other, creating symmetric navigation patterns for developers working on both projects.

## Links

- [DOPPLER package](./doppler/) - The actual code in this repo
- [REPLOID repo](https://github.com/clocksmith/reploid) - Browser-native AI agent
- [replo.id/r](https://replo.id/r) - REPLOID live demo
- [replo.id/d](https://replo.id/d) - DOPPLER live demo
