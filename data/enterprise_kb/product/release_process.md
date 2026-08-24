# Release Process

## Cadence
Product releases ship every 2 weeks on Wednesday. The release train does not wait for features; incomplete work ships behind flags or waits for the next train. Hotfixes may bypass the schedule with CTO approval.

## Timeline
Code freeze occurs 3 days before release day. Release candidate builds cut the same evening and run the full regression suite overnight. Release day begins with a go/no-go meeting at 10:00 attended by engineering leads, QA, support, and product.

## Go/No-Go Criteria
Promotion requires: zero open SEV1/SEV2 defects, regression suite green, performance benchmarks within 10 percent of baseline, rollback plan rehearsed, and support briefed on known issues. Any red item defers the release to the next train unless CTO overrides in writing.

## Staged Rollout
Releases roll out in stages: internal dogfood for 24 hours, then 10 percent of customers for 48 hours, then full availability. Feature flags decouple deployment from exposure so stages can pause independently.

## Release Notes
Release notes publish within 2 hours of full rollout, listing user-visible changes, known issues, and upgrade instructions. Enterprise customers receive advance notes 5 business days early under NDA.

## Hotfix Path
Hotfixes address production regressions only. They require a linked incident, minimal diff, expedited review by two engineers, and same-day deployment outside normal windows. Every hotfix replays through the next regular train.

## Versioning
Products follow semantic versioning. Breaking API changes bump the major version and follow the deprecation policy with 90-day notice. Mobile apps follow store-mandated review timelines factored into the train.

## Post-Release Review
Within 3 days of each release, the release captain files a short retrospective: what shipped, incidents triggered, and process friction. Trends feed quarterly process improvements.

## Freeze Exceptions
Freeze weeks align with financial close and major holidays, published in the release calendar at least one month ahead.
