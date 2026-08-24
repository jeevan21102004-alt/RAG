# Testing Guidelines

## Test Pyramid
The testing strategy follows the classic pyramid: many fast unit tests, a moderate layer of integration tests, and a small set of end-to-end journeys. Teams own their pyramid balance; the platform team audits ratios quarterly.

## Unit Tests
Unit tests are mandatory for all new production code. They run in-memory, complete in under 100 milliseconds each, and isolate external systems behind fakes. Table-driven tests cover boundary values including empty, single-element, and overflow cases.

## Integration Tests
Integration tests exercise real databases, queues, and downstream stubs. The suite runs nightly against ephemeral environments provisioned per pipeline run. Contract tests verify provider-consumer agreements for every cross-service API dependency.

## End-to-End Tests
End-to-end journeys cover the top five revenue-critical flows: signup, checkout, subscription renewal, invoice download, and refund. These run on every release candidate and block promotion on failure.

## Performance Testing
Performance tests execute before every major release and compare p95 latency and throughput against the previous baseline. Regressions greater than 10 percent require either optimization or explicit acceptance from the product owner.

## Flaky Test Quarantine
Tests failing intermittently enter quarantine automatically after two flaky detections in seven days. Quarantined tests are visible on a dashboard with owners and must be fixed or deleted within 14 days. Quarantine size above 20 tests freezes non-critical merges for the owning team.

## Coverage Gates
Line coverage minimum is 80 percent on changed files. Mutation-testing spot checks run monthly on payment and auth modules to validate assertion strength.

## Test Data
Fixtures are synthetic and committed with tests. Production data copies require anonymization approval and are prohibited in lower environments by default. Secrets never appear in test code; test credentials come from the ephemeral environment injector.

## Review Standards
Reviewers reject tests that assert implementation details, sleep-based synchronization, or order dependence. Deterministic tests are a merge requirement; randomness seeds itself visibly.

## Local Experience
A single command boots dependencies and runs the fast suite in under five minutes. Slow suites are marked and excluded from pre-push hooks.
