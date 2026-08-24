"""Temporary generator: HR + Engineering enterprise KB documents (synthetic)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "enterprise_kb"

DOCS = {
"hr/leave_policy.md": """# Annual and Sick Leave Policy

## Purpose
This policy defines how annual leave and sick leave are accrued, requested, and approved for all full-time employees of the company. It applies to every department and is administered by the People Operations team.

## Annual Leave Entitlement
Every full-time employee receives 24 days of paid annual leave per calendar year. Leave accrues at a rate of 2 days per completed month of service. New joiners who start mid-year receive leave proportional to their remaining months.

A maximum of 5 unused annual leave days may be carried over into the next calendar year. Any balance above 5 days lapses on 31 December and cannot be claimed or exchanged for cash.

## Requesting Leave
Leave must be requested through the HR portal at least 7 working days in advance. Requests for 5 or more consecutive days require approval from both the direct manager and the HR business partner. Requests shorter than 5 days require only manager approval.

Leave during the last two weeks of December is limited to 40 percent of any team at the same time, to guarantee support coverage during the holiday period.

## Sick Leave
Employees may take up to 12 days of paid sick leave per year. A doctor's note is required for any sick absence of 3 or more consecutive working days. The note must be uploaded to the HR portal within 2 working days of returning to work.

Sick leave does not carry over between years and is not paid out at exit.

## Public Holidays
The company observes 12 public holidays per year. Holidays that fall on a weekend are not substituted. Regional holiday calendars apply for offices outside the headquarters city.

## Exit and Payout
Unused annual leave up to a maximum of 15 days is paid out at the base daily rate when an employee leaves the company. Payout above 15 days requires written approval from the Head of People Operations.

## Contact
Questions about this policy should be directed to People Operations at the internal HR helpdesk. This policy is reviewed once every year in January.
""",
"hr/remote_work_policy.md": """# Remote Work Policy

## Overview
This policy governs remote and hybrid work arrangements for all employees. The company operates a hybrid-first model: teams are expected on site for collaboration days, while individual focus work may be done remotely.

## Weekly Remote Allowance
Employees may work remotely up to 3 days per week. The remaining days are expected on site unless a specific arrangement has been approved. Team leads may set additional on-site collaboration days for their teams with notice of at least one week.

Core collaboration hours are 10:00 to 16:00 local time. During core hours, remote employees must be reachable on the approved chat and video tools and respond to meeting invitations.

## International Remote Work
Working from another country is permitted for up to 20 days per calendar year. Any international remote period longer than 10 consecutive working days requires written approval from the responsible Vice President and a review by Payroll, because tax and social-security implications differ by country.

Employees must not work from countries on the restricted list published by Legal. The restricted list is reviewed quarterly.

## Equipment and Environment
Remote employees must use the company-issued laptop with full-disk encryption enabled. A stable internet connection of at least 25 Mbps is recommended for video calls. The company provides a one-time home-office setup budget; see the employee benefits document for the current amount.

## Availability and Overtime
Remote work does not change contractual working hours. Overtime rules and time tracking apply identically to on-site and remote days. Managers must record team-level remote usage monthly and share a summary with People Operations.

## Approval Workflow
Standard weekly remote days need no extra approval beyond the team calendar. Exceptions, such as fully remote weeks, require manager approval submitted at least 5 working days in advance. Arrangements lasting more than one month require HR sign-off.

## Review
This policy is reviewed every six months by People Operations together with Facilities and IT.
""",
"hr/employee_benefits.md": """# Employee Benefits Guide

## Health Coverage
All full-time employees are enrolled in the group health insurance plan on their first day. The plan covers the employee, spouse, and up to two dependent children. Preventive check-ups are covered once per year without co-payment. Dental and vision coverage can be added during the annual enrollment window each November.

## Wellness Stipend
Each employee receives an annual wellness stipend of 15,000 rupees. Eligible expenses include gym memberships, fitness classes, sports equipment, and mental-health counselling sessions. Receipts must be uploaded through the benefits portal within 60 days of purchase. Unused stipend amounts do not roll over into the next year.

## Fitness Reimbursement
In addition to the wellness stipend, employees may claim a monthly gym reimbursement of up to 1,200 rupees against a receipt from a registered fitness facility. This claim is separate from the annual stipend and resets every month.

## Home Office Setup
New employees receive a one-time home-office setup allowance of 20,000 rupees for chair, desk, monitor, and peripherals. Claims require itemized receipts and approval from the direct manager. Existing employees may refresh equipment once every three years.

## Learning Budget
Every employee has an annual learning budget of 30,000 rupees for courses, certifications, books, and conference tickets. Conference attendance also requires manager approval for travel days. Certifications related to the employee's role are prioritized.

## Meal and Commute
Subsidized lunches are provided at all office locations. Employees commuting by public transport receive a monthly commute card funded at 80 percent of the standard fare.

## Referral Bonus
Successful referrals for open positions pay a bonus of 25,000 rupees after the referred employee completes three months of service. Referrals for senior engineering roles pay 50,000 rupees.

## Insurance Extras
Group accident insurance and term-life cover equal to twice the annual salary are provided at no cost. Nominee details must be kept current in the HR portal.

## Administration
Benefits questions go to the People Operations helpdesk. Stipends and reimbursements are processed with the monthly payroll cycle. This guide is updated each January and July.
""",
"hr/parental_leave.md": """# Parental and Family Leave Policy

## Scope
This policy covers birth, adoption, and surrogacy leave for all permanent employees with at least three months of service. Contractors follow the statutory minimum only.

## Primary Caregiver Leave
The primary caregiver receives 26 weeks of fully paid parental leave. Leave must begin within 6 months of the birth or placement date and may be taken continuously or in up to three blocks within the first 12 months. At least 4 weeks must be taken immediately following the birth or placement.

An additional 8 weeks of unpaid extension may be requested and requires approval from both the manager and the Head of People Operations at least 4 weeks before the intended start.

## Secondary Caregiver Leave
The secondary caregiver receives 4 weeks of fully paid leave. Up to 2 additional weeks may be taken as unpaid leave within the first year. Secondary caregiver leave should normally be completed within 6 months of the birth or placement.

## Adoption and Surrogacy
Adoptive parents taking primary caregiving responsibility receive 16 weeks of fully paid adoption leave, starting from the date of placement. Surrogacy arrangements follow the same structure as adoption leave for the intended primary caregiver.

## Job Protection
Parental leave does not affect seniority, performance review timing, or promotion eligibility. Returning employees have the right to request flexible scheduling for up to 6 months after return, subject to operational feasibility.

## Keeping in Touch
Employees on leave may volunteer for up to 10 keeping-in-touch days per leave year. These days are paid at the normal daily rate and must be logged with the manager's consent. Attendance at mandatory compliance training counts separately and is compensated.

## Benefits During Leave
Health insurance continues unchanged during paid and unpaid parental leave. Annual leave continues to accrue during paid leave but pauses during unpaid extensions.

## Process
Notification of expected dates goes to the manager and People Operations at least 8 weeks before the expected start, where possible. Return-to-work meetings are scheduled two weeks before the return date.
""",
"hr/attendance_policy.md": """# Attendance and Time Tracking Policy

## Working Hours
Standard office hours are 09:00 to 18:00 local time, Monday through Friday, with a one-hour lunch break. Teams may agree shifted schedules with their manager, provided overlap with core collaboration hours is maintained.

## Check-In System
All employees use the biometric check-in system at office entrances. Badge tap records entry and exit times automatically. Employees working remotely log attendance in the time-tracking tool with a daily check-in before 10:00.

## Grace Period
A grace period of 15 minutes applies to morning arrival. Arrivals later than the grace period are recorded as late. Three late arrivals within a calendar month trigger an automated warning to the employee and the direct manager. Six late arrivals in a quarter trigger a formal performance conversation.

## Absence Reporting
Unplanned absences must be reported to the direct manager before 10:00 on the day of absence through the attendance tool or a phone call. Failure to report an unplanned absence for two consecutive days is treated as job abandonment under disciplinary procedures.

## Time Tracking for Projects
Project-based teams record billable hours weekly. Timesheets are due every Friday at 17:00. Missing timesheets for two consecutive weeks escalate to the department head.

## Overtime Recording
Approved overtime is logged in half-hour increments and compensated according to the employment contract or compensated with equivalent time off within 90 days. Unapproved overtime is not compensated; managers are responsible for approving overtime in advance wherever possible.

## Monitoring and Privacy
Attendance data is used for payroll, capacity planning, and compliance only. Access to individual attendance records is restricted to the employee, the direct manager, and People Operations. Data is retained for 36 months.

## Exemptions
Field staff, executives, and employees with documented medical accommodations may be exempt from biometric check-in. Exemptions are granted by the Head of People Operations and reviewed annually.

## Disciplinary Ladder
Attendance issues follow a progressive ladder: automated warning, manager conversation, written warning, and finally disciplinary review. Employees may appeal any step to People Operations within 10 working days.
""",
"engineering/deployment_guidelines.md": """# Deployment Guidelines

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
""",
"engineering/api_standards.md": """# API Design Standards

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
""",
"engineering/coding_standards.md": """# Coding Standards

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
""",
"engineering/incident_management.md": """# Incident Management Process

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
""",
"engineering/testing_guidelines.md": """# Testing Guidelines

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
""",
}

def main() -> None:
    for rel, content in DOCS.items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.strip() + chr(10), encoding="utf-8")
        words = len(content.split())
        print(f"wrote {rel} ({words} words)")

if __name__ == "__main__":
    main()