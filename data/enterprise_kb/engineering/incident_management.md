# Incident Management Process

## Severity Classification
Incidents are classified at declaration: SEV1 means customer-facing outage or data-integrity risk affecting many customers; SEV2 means degraded service with a workaround or single-region impact; SEV3 means minor issues with no material customer impact. Severity may be upgraded or downgraded by the incident commander as facts emerge.

## Roles
Every SEV1 and SEV2 assigns three roles: incident commander who coordinates response and communication, operations lead who executes technical mitigation, and communications lead who drafts customer and executive updates. The on-call engineer who acknowledges the page becomes the initial commander until handover.

## Response Expectations
SEV1 pages the on-call immediately and requires acknowledgment within 5 minutes. SEV2 requires response within 15 minutes during business hours and 30 minutes otherwise. If unacknowledged, paging escalates to the secondary and then the engineering manager.

## Status Updates
During a SEV1, the communications lead posts status updates every 30 minutes to the incident channel and the status page, even when the update is "investigating". SEV2 updates go out hourly. Updates state impact, current hypothesis, next checkpoint, and help needed.

## Mitigation First
Stabilization precedes root-causing. Preferred mitigations in order: roll back the recent change, fail over to healthy region, shed non-critical load, apply feature-flag kill switches. Destructive remediation requires incident-commander approval.

## Communication Channels
Each incident gets a dedicated channel created from the template with pinned timeline, dashboard links, and customer-impact summary. Executives receive a bridge invite for SEV1 within 15 minutes of declaration.

## Postmortems
SEV1 postmortems are due within 5 business days of resolution; SEV2 within 10 days. Postmortems are blameless, include a full timeline, contributing factors, and action items with owners and due dates. Action items are tracked to completion in the reliability board; overdue items escalate monthly.

## Drills
Game-day exercises run quarterly, rotating failure scenarios such as region loss, certificate expiry, and dependency brownouts. Drill findings feed the action-item backlog.

## Metrics
We track MTTA, MTTR, change-failure rate, and customer-impact minutes per severity, reviewed in the monthly ops review.
