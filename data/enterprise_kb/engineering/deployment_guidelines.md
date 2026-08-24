# Deployment Guidelines

## Deployment Windows
Production deployments are allowed Tuesday through Thursday between 10:00 and 16:00. Friday deployments are prohibited except for critical security patches. Deployments are frozen during the last week of each quarter due to financial close activities.

## Approvals
Every production deployment requires two approvals before execution: the owning team's tech lead and the on-call SRE engineer. Deployments touching payment flows additionally require sign-off from the payments guild lead. Emergency security fixes may deploy with a single SRE approval followed by retrospective review within 24 hours.

## Pre-Deployment Checklist
Before deploying, the release captain confirms: all CI checks green, database migrations reviewed by a second engineer, feature flags default-off for new functionality, rollback plan documented in the deploy ticket, and monitoring dashboards open.

## Rollout Strategy
Services deploy using canary rollout: 5 percent of traffic for 15 minutes, then 25 percent for 15 minutes, then full rollout if error budgets hold. Automatic rollback triggers when the error rate exceeds twice the baseline for 5 minutes or when p99 latency doubles.

## Rollback Requirements
Every deployment must be reversible within 30 minutes. Rollback capability is verified in staging before production promotion. Database migrations must be backward compatible for at least one release so that application rollbacks remain safe.

## Change Records
Each deployment creates an immutable change record containing commit range, approvers, timestamps, and linked tickets. Change records feed the weekly operations review and quarterly audit samples.

## Post-Deployment Verification
The release captain monitors key dashboards for 30 minutes after full rollout. Synthetic probes must pass in all regions. Customer-facing incidents caused by a deployment trigger the incident management process immediately.

## Environments
Changes flow through dev, staging, and production. Direct production changes are forbidden. Hotfixes may skip staging only with CTO approval and must be replayed to staging afterward.

## Capacity and Freezes
Deployments pause automatically when infrastructure costs exceed the weekly guardrail or when a SEV1 incident is open. The release calendar shows planned freezes at least one month ahead.
