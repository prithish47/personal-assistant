# ARIA

ARIA is a local-first autonomous desktop agent for Windows. It is intentionally built around typed capabilities, explicit permissions, auditable tool execution, provider-neutral model interfaces, and inspectable task plans.

## Architecture

`Transport -> application services -> planning -> tools/plugins/providers/memory adapters`

The model never receives shell access. Plugins register typed tools; the executor validates inputs, checks permissions, records an audit result, and only then invokes a platform API.

## Development

Use Python 3.11+ and create a fresh virtual environment. Copy `.env.example` to `.env`, set a unique API key, then install `pip install -e .[dev]`. Start with `aria`.

The local Ollama service and configured models must be available before ARIA reports provider readiness.

## Current milestone

The backend foundation and shell-free tool boundary are being rebuilt first. The legacy C++ bridge, implicit global agents, fake telemetry, and arbitrary command execution have been removed. The next milestone adds durable memory repositories, provider adapters, task execution, and the production frontend against this API contract.
