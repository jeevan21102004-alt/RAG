# API Design Standards

## Versioning
All public REST APIs are versioned in the URL path using the /v1, /v2 convention. A new major version is created only for breaking changes. Minor additions extend the existing version without a new path. Deprecated versions remain available for at least 180 days after announcement.

## Latency Targets
Interactive endpoints target a p95 latency below 300 milliseconds measured at the gateway. Batch endpoints may run asynchronously and return a job identifier. Endpoints exceeding targets for three consecutive days trigger a performance investigation ticket owned by the service team.

## Error Format
Errors follow RFC 7807 problem-details JSON with fields: type, title, status, detail, and instance. Machine-readable error codes are stable strings such as VALIDATION_FAILED or QUOTA_EXCEEDED. Human-readable messages never leak stack traces or internal hostnames.

## Authentication and Authorization
APIs authenticate with OAuth 2.0 bearer tokens. Service-to-service calls use mutual TLS inside the mesh. Scopes follow resource-action naming, for example invoices:read. Tokens expire after 60 minutes; refresh tokens rotate on every use.

## Pagination and Filtering
List endpoints paginate with cursor-based tokens and a default page size of 50, maximum 200. Filters use explicit query parameters rather than a generic query language. Sorting uses a whitelist of sortable fields.

## Idempotency
POST endpoints that create money-moving or side-effecting resources accept an Idempotency-Key header. Keys are honored for 24 hours. Duplicate submissions return the original response with an idempotent-replay flag.

## Rate Limiting
Default quota is 100 requests per minute per client, returning 429 with Retry-After headers. Higher tiers are negotiated with the platform team. Rate-limit headers expose remaining quota on every response.

## Documentation
Every endpoint ships an OpenAPI specification generated from code annotations. The spec is the contract; drift fails CI. Changelog entries accompany every merged specification change.

## Deprecation
Breaking changes require a deprecation notice at least 90 days before removal. Notices appear in changelogs, response headers (Deprecation and Sunset), and direct email to registered integrators.
