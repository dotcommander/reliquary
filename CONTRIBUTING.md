# Contributing to reliquary

reliquary is a library-only, boundary-first toolkit of provider-neutral primitives
and thin opt-in adapters. Before adding or moving a package, check it against the
architecture in [ARCHITECTURE.md](ARCHITECTURE.md): a package belongs in
reliquary only if it is (1) library-only, (2) works in a caller-owned semantic space,
(3) a neutral primitive or thin adapter, and (4) domain-free. Anything that fails
the rule belongs in an application or another module.

## Boundaries

- Root packages must not initialize model runtimes, databases, provider clients,
  HTTP servers, CLIs, migrations, prompts, or application policy.
- Adapters live under `adapter/` in this module. They may depend on their upstream
  SDK or driver, but core packages must not depend on adapters (see ARCHITECTURE.md).
- Run `./scripts/check-standard.sh` before submitting.
