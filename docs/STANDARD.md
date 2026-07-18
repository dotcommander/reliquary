# Repository standard

Reliquary is retrieval-only. Core packages remain provider- and driver-neutral;
provider and database dependencies belong under `adapter/`.

Public constructors use typed configuration and injected upstream types.
Database adapter constructors perform validation only, and schema creation is
an explicit `Migrate(ctx)` operation. Candidate retrieval must remain bounded
for positive limits and deterministically order equal scores by result ID.

Every Index implementation must run `index/indextest`. Changes must pass the
root build, test, vet, race, boundary, module, and whitespace checks documented
in the root README.
