# Coding Standards

## Language Baselines
Backend services are written in Python 3.11+ or Go 1.21+. Python code follows PEP 8 enforced by automated linting. Go code follows the output of gofmt and golangci-lint. Frontend code follows the shared ESLint configuration maintained by the platform team.

## Pull Request Size
Pull requests should stay below 400 changed lines excluding tests and generated files. Larger changes must be split into stacked pull requests with independent value. The reviewer may decline oversized PRs with guidance on splitting.

## Reviews
Every merge to main requires at least one peer review from outside the author. Changes to authentication, payments, or data-migration code require two reviewers including a domain owner. Review turnaround expectation is one business day.

## Test Coverage
Minimum code coverage is 80 percent line coverage on changed files. Coverage regressions fail CI. Critical paths such as billing calculations require branch coverage assertions, not just line coverage.

## Naming and Structure
Names describe behavior, not implementation. Functions do one thing. Files exceed 500 lines only with justification comments. Shared utilities live in designated common packages and require platform-team review.

## Error Handling
Errors are handled explicitly; silent except-pass blocks are banned. Errors crossing service boundaries carry correlation identifiers. User-facing messages avoid internal jargon and never include raw exception text.

## Dependencies
Adding a dependency requires justification in the pull request description, license compatibility check, and no known critical CVEs. Dependency updates flow through automated weekly upgrade pull requests.

## Comments and Documentation
Code explains what and why; commit messages explain context. Public functions carry docstrings with parameter and return descriptions. Architectural decisions are recorded in the decision-log repository using the lightweight ADR format.

## Formatting
Formatting is automated and non-negotiable: formatters run as pre-commit hooks and in CI. Style debates end at the formatter configuration, which changes only via platform-team proposal.

## Security Practices
Secrets never enter source control; they load from the secret manager at runtime. Input validation happens at trust boundaries. New endpoints get threat-modeled when they handle personal data.
