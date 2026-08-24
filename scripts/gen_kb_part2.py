"""Temporary generator: Finance + Product + Security enterprise KB documents (synthetic)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "enterprise_kb"

DOCS = {
"finance/travel_policy.md": """# Business Travel Policy

## Booking
All business travel is booked through the approved corporate travel portal. Bookings made outside the portal require pre-approval from the department head and may not be fully reimbursable. Tickets should be booked at least 14 days before departure whenever possible to capture lower fares.

## Flights
Domestic flights are booked in economy class. International flights up to 8 hours are economy; flights longer than 8 hours may be booked in premium economy for directors and above. Frequent-flyer points earned on business travel belong to the employee, but upgrades paid with company funds are not permitted.

## Hotels
Domestic hotel reimbursement is capped at 4,000 rupees per night for standard employees and 6,000 rupees per night for senior managers. International hotel reimbursement is capped at 9,000 rupees per night for all grades. Caps include taxes and service fees. Stays exceeding the cap require written pre-approval from the finance controller.

## Per Diem
A domestic per diem of 2,500 rupees per day covers meals and incidentals while travelling. The international per diem is 5,000 rupees per day. Per diem applies only for days with at least 12 hours away from the home city. No meal receipts are required when claiming per diem.

## Ground Transport
Airport transfers, taxis, and ride-hailing are reimbursable against receipts. Car rentals require manager approval and must include insurance. Personal vehicle use is reimbursed at 12 rupees per kilometre up to 300 kilometres per trip.

## Duration Limits
Single business trips may not exceed 21 days without HR and finance approval. Extended assignments convert to relocation under separate policy.

## Compliance
Travel advances are limited to 60 percent of estimated trip cost and must be settled within 15 days of return. Violations of booking or cap rules may result in personal liability for excess amounts.
""",
"finance/expense_policy.md": """# Expense Reporting Policy

## Submission Deadline
Expense reports must be submitted within 30 days of the expense date. Reports older than 30 days require a written justification and director approval before processing. Expenses older than 90 days are not reimbursable except for documented medical emergencies.

## Receipts
Receipts are mandatory for any single expense above 500 rupees. Digital photographs of receipts are accepted if all fields are legible: merchant name, date, amount, tax breakdown, and payment method. Missing receipts above the threshold result in automatic rejection.

## Eligible Expenses
Eligible categories include client meals, team events capped at 1,500 rupees per attendee, software subscriptions approved by IT, conference fees, and local transport. Client meals require attendee names and business purpose in the report notes.

## Ineligible Expenses
Alcohol is never reimbursable. Traffic fines, parking violations, personal entertainment, and spouse travel are excluded. First-class upgrades and minibar charges are rejected automatically by the audit workflow.

## Approval Limits
Monthly cumulative expenses up to 50,000 rupees are approved by the direct manager. Amounts between 50,000 and 200,000 rupees require director approval. Anything above 200,000 rupees requires CFO sign-off. Splitting one purchase into multiple reports to stay under a threshold is treated as a policy violation.

## Currency and Tax
Foreign-currency expenses convert at the corporate card rate on the transaction date. Input-tax-credit claims require tax invoices showing the company GSTIN. Finance ops reviews GST compliance monthly.

## Corporate Cards
Corporate cards are issued to managers and frequent travellers. Card statements reconcile monthly; unresolved items older than 45 days suspend the card automatically. Lost cards are reported within 24 hours to both the bank and finance ops.

## Audit
Finance audits a random 10 percent sample of reports every month plus 100 percent of reports above 100,000 rupees. Findings feed quarterly compliance training.
""",
"finance/reimbursement_policy.md": """# Reimbursement Processing Policy

## Processing Timeline
Approved reimbursements are processed within 10 business days of final approval. Payments run in two batches per week, on Tuesday and Friday. Delays caused by incomplete banking details reset the clock from the date the details are corrected.

## Payment Channels
Reimbursements pay out through NEFT to the salary account on file. Employees must keep bank details current in the payroll portal; changes require OTP verification and take effect after the next verification cycle for security.

## Recurring Allowances
The monthly internet allowance is 1,500 rupees for eligible remote-working roles. The mobile allowance is 800 rupees per month for roles on the approved list. Allowances appear as separate line items in payslips and do not require receipts.

## Advance Settlement
Travel advances must be settled within 15 days of trip completion. Unsettled advances older than 30 days are deducted from the next salary cycle with prior notice. Partial settlement is allowed with an explanation attached.

## Disputes
Disputed reimbursements must be raised within 20 working days of payment. Finance ops responds within 5 working days. Escalations go to the finance controller whose decision is final.

## Documentation Retention
Reimbursement records, including receipts and approvals, are retained for 7 years for audit purposes. Records are stored in the finance document system with role-based access.

## Fraud Handling
Suspected duplicate or fabricated claims are referred to internal audit immediately. Confirmed fraud results in recovery of amounts and disciplinary action up to termination. Anonymous reporting is available through the ethics hotline.

## Status Tracking
Employees track claim status in the finance portal: submitted, manager approval, finance review, scheduled, paid. Notifications fire at each stage transition.
""",
"finance/procurement_policy.md": """# Procurement Policy

## Purpose and Scope
This policy governs purchasing of goods and services above 10,000 rupees. It applies to all departments. Software subscriptions follow the same thresholds but add an IT security review before signature.

## Quotation Requirements
Purchases between 50,000 and 100,000 rupees require three comparative quotes from independent vendors. Purchases below 50,000 rupees require one quote attached to the request. Sole-source purchases need a written justification approved by the department head.

## Approval Thresholds
Department heads approve purchases up to 100,000 rupees. Purchases above 100,000 rupees require CFO approval. Contracts above 500,000 rupees or longer than 24 months additionally require legal review and board-level notification.

## Preferred Vendors
Finance operations maintains the preferred vendor list, refreshed twice a year through competitive review. Preferred vendors offer negotiated rates and standard terms. Using non-preferred vendors for available categories requires justification.

## Purchase Orders
No vendor commitment is valid without a purchase order number. POs are raised in the procurement system after approvals complete. Emergency purchases may proceed verbally during declared incidents but require a retroactive PO within 5 working days.

## Vendor Onboarding
New vendors submit tax registration, bank details verified via penny-drop, and signed code-of-conduct. Vendor risk tiering determines due-diligence depth: high-risk categories such as staffing and data processing get annual reassessment.

## Contract Management
Contracts are stored in the contract repository with renewal alerts at 90, 60, and 30 days. Auto-renewal clauses above 100,000 rupees annually must be flagged and consciously accepted.

## Conflict of Interest
Employees declare conflicts before participating in vendor selection. Gifts above 2,000 rupees from vendors must be declined or surrendered to finance ops.

## Sustainability
Procurement prefers vendors with published environmental policies for categories where alternatives exist. This preference is advisory, not blocking.
""",
"finance/budget_policy.md": """# Budget Planning and Control Policy

## Annual Cycle
Departments submit draft budgets 45 days before each quarter start. Finance consolidates drafts into the company plan, which the executive committee approves two weeks before quarter start. Mid-year re-planning happens once, in July, for material shifts only.

## Variance Management
Monthly variance above 10 percent of line-item budget requires a written explanation from the budget owner. Persistent variance for three consecutive months triggers a corrective-action plan agreed with finance business partners. Underspend above 15 percent also requires explanation to prevent sandbagging.

## Budget Ownership
Each cost centre has exactly one named owner accountable for forecast accuracy. Owners update rolling forecasts monthly by the 5th working day. Delegation of ownership requires finance notification.

## Capital Expenditure
Capex requests follow the procurement thresholds and additionally require a payback analysis. Projects with payback beyond 36 months need CFO presentation. Depreciation schedules follow the accounting policy maintained by the controller.

## Headcount Planning
Headcount budgets split into backfilled and net-new roles. Net-new roles above band P5 require executive approval even when headcount budget exists. Open requisitions lapse at quarter end unless re-requested.

## Contingency Reserve
A central contingency reserve of 3 percent of operating budget is held by finance. Accessing contingency requires CFO approval and a documented risk event. Unused reserve releases to the bonus pool at year end.

## Reporting
Budget-versus-actual dashboards refresh weekly. Department heads receive variance alerts by email. Quarterly business reviews include budget scorecards covering accuracy, timeliness, and compliance.

## Tooling
All planning happens in the FP&A platform; spreadsheet-based submissions are not accepted for official numbers. Access follows least privilege and joins joiner-mover-leaver processes.

## Exceptions
Exceptions to any part of this policy require written CFO approval with expiry dates not exceeding one fiscal year.
""",
"product/product_roadmap.md": """# Product Roadmap Overview

## Planning Horizon
The roadmap covers four quarters in detail and two further quarters at theme level. It is reviewed monthly by the product council and republished internally after each review. Dates represent committed windows, not guarantees.

## Current Year Themes
Quarter one focuses on the mobile application redesign, targeting a modernized navigation shell and offline reading. Quarter two delivers the AI assistant beta for enterprise tenants, gated behind an opt-in flag. The second half expands the marketplace with third-party integrations and launches advanced analytics.

## Committed Windows
Mobile redesign beta ships in week 6 of Q1 with general availability in week 11. AI assistant beta opens to design partners in week 4 of Q2. Marketplace public launch targets week 9 of Q3. Advanced analytics general availability lands in week 7 of Q4.

## Product Council
The product council meets on the first Monday of each month. Membership includes VP product, engineering directors, design lead, data lead, and rotating customer-success representation. Council decisions are recorded in decision logs with dissent noted.

## Intake Alignment
Roadmap candidates come from the feature request process, sales escalations, support themes, and strategic bets. Each candidate carries an impact and effort score reviewed biweekly by the prioritization working group.

## Dependencies
Cross-team dependencies are tracked in the delivery board with named owning teams. A dependency slipping more than two weeks triggers roadmap re-review for affected items.

## Communication
Customer-facing announcements follow the release process calendar. Sales enablement receives roadmap briefings six weeks ahead of any generally available launch. Under-NDA previews require legal-approved decks.

## Deprecation Notices
Sunsetting features appear on the roadmap one full quarter before removal, with migration guides linked from the changelog.

## Feedback Loop
Post-launch reviews happen 30 days after each major delivery, comparing adoption against targets and feeding lessons into the next planning cycle.
""",
"product/release_process.md": """# Release Process

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
""",
"product/pricing_policy.md": """# Pricing Policy

## Plans
The starter plan costs 999 rupees per month billed monthly, including up to 5 users and core workflows. The professional plan costs 2,499 rupees per month for up to 25 users with advanced analytics and priority support. The enterprise plan uses custom pricing negotiated per account with dedicated infrastructure options.

## Billing Terms
Monthly plans bill on the subscription anniversary date. Annual contracts receive a 15 percent discount versus monthly pricing and invoice upfront or in two halves. Currency is INR for India-billed accounts and USD elsewhere, with prices locked for the contract term.

## Discounts
Volume discounts start at 40 seats: 5 percent at 40 seats, 10 percent at 100 seats, and 15 percent at 250 seats. Volume discounts stack with the annual discount to a combined ceiling of 25 percent. Nonprofit and education customers receive a flat 30 percent discount with eligibility verification.

## Trials
A 14-day trial includes professional-tier features without a credit card. Trial extension by 7 days is self-service once. Conversions from trial inherit the trial configuration.

## Upgrades and Downgrades
Upgrades apply immediately with prorated charging. Downgrades apply at the next renewal to prevent mid-cycle disruption. Seat reductions below contracted minimums require account-executive approval.

## Add-ons
Additional storage packs cost 299 rupees per 100 GB per month. Premium support adds 10 percent of subscription value with a minimum of 5,000 rupees monthly. Sandbox environments add 1,999 rupees per environment per month.

## Refunds
Monthly plans refund unused full months within 14 days of charge. Annual contracts refund pro rata minus 10 percent processing fee within 30 days of cancellation request. Chargebacks follow card-network rules.

## Price Changes
Published price changes give existing subscribers 90 days notice and apply at renewal. Grandfathered legacy plans migrate only with customer consent and equivalent discount protection.

## Tax
Displayed prices exclude GST. Invoices show tax separately with the company GSTIN for input credit.
""",
"product/feature_request_process.md": """# Feature Request Process

## Intake Channels
Feature requests arrive through the public voting board, in-app feedback widget, sales escalations, support tickets tagged enhancement, and partner requests. All channels funnel into the single intake queue owned by product operations.

## Triage SLAs
New requests receive first triage within 5 business days. Triage assigns category, deduplicates against existing entries, merges duplicates preserving vote counts, and tags the affected persona and plan tier.

## Scoring Model
Every triaged request scores on impact (reach times severity) and effort (engineering estimate in t-shirt sizes). Impact weights paying-enterprise reach highest, then paying-SMB, then trial, then free. Requests touching compliance or security carry mandatory priority multipliers.

## Review Cadence
The prioritization working group reviews the top-scored queue biweekly. Reviews select items for discovery, specification, or rejection with rationale posted publicly on the board entry.

## Public Voting
Customers vote on the public board; votes influence but do not determine priority. Vote counts display transparently. Requesters receive automated status emails at every state change: received, triaged, planned, in progress, shipped, declined.

## Specification
Planned items move to discovery with a product manager owner. Specifications define problem statement, success metrics, scope boundaries, and rollout plan. Large items split into shippable increments aligned to the release train.

## Declined Requests
Declined entries state reasons plainly: overlap with existing capability, misalignment with strategy, insufficient demand relative to effort, or technical infeasibility. Requesters may reopen discussion once per year with new evidence.

## Internal Escalations
Sales may escalate revenue-blocking requests through the deal-desk fast lane, which guarantees a prioritization review within one week. Escalation abuse is monitored and coached.

## Shipped Communication
Shipped requests trigger changelog entries, board updates crediting top voters, and targeted release-note emails to original requesters.

## Metrics
Process health tracks time-to-triage, time-to-decision, percentage shipped within two quarters of planning, and requester satisfaction sampled quarterly.
""",
"product/product_metrics.md": """# Product Metrics Handbook

## North Star Metric
The North Star metric is weekly active teams: distinct teams with at least three members performing a core action in a given week. Core actions are creating a project, completing a workflow run, or exporting a deliverable. The North Star reviews monthly at the executive level.

## Activation
Activation measures the share of new workspaces reaching defined value milestones within 14 days of signup. The activation target is 60 percent. Milestones: invite a teammate, connect a data source, and complete a first workflow. Activation funnels break down by acquisition channel and plan tier.

## Retention and Churn
Logo churn target stays below 2 percent monthly for SMB and below 0.5 percent quarterly for enterprise. Net revenue retention target is 110 percent annually. Cohort curves review monthly with attention to week-one and week-four drop-offs.

## Engagement
Daily active users over monthly active users targets 0.35 for collaboration products. Session depth tracks median actions per session. Feature adoption dashboards measure 28-day adoption for each major release, with 40 percent as the healthy bar.

## Reliability Perception
In-product reliability sentiment derives from post-interaction micro-surveys. A satisfaction dip correlating with incidents triggers joint review with engineering.

## Satisfaction
Net promoter score target is at least 40 measured quarterly across a stratified sample. Customer-effort score accompanies support interactions with a target below 2.5. Review-site ratings aggregate monthly.

## Experiment Standards
Experiments run with pre-registered hypotheses, minimum detectable effects, and guardrail metrics including latency, error rate, and support volume. Ship decisions require statistically significant primary-movement without guardrail breaches.

## Data Quality
Metric definitions live in the shared semantic layer; ad-hoc definitions are prohibited in executive reporting. Pipeline freshness SLA is 4 hours for behavioral events and 24 hours for billing-derived figures.

## Review Rituals
Weekly growth standup reviews funnel and experiment readouts. Monthly business review covers North Star, retention cohorts, and NPS trend. Quarterly deep dives examine segment economics and metric definition drift.
""",
"security/password_policy.md": """# Password and Authentication Policy

## Password Requirements
Passwords must contain at least 12 characters mixing upper case, lower case, digits, and symbols. Sequential patterns, dictionary words, and previously breached passwords are blocked by the identity provider's screening list. Password hints are disabled everywhere.

## Expiration
Passwords expire every 90 days for standard accounts. Privileged accounts expire every 60 days. The identity provider warns users starting 14 days before expiry. New passwords cannot reuse any of the previous 10 passwords.

## Multi-Factor Authentication
MFA is mandatory for all employees without exception. Approved factors are authenticator apps with TOTP, hardware security keys, and push-based approval with number matching. SMS codes are permitted only as a fallback for accounts awaiting hardware key issuance, and never for administrator accounts.

## Administrator Accounts
Administrator accounts use phishing-resistant MFA exclusively, meaning FIDO2 security keys. Admin sessions idle-timeout after 15 minutes. Administrative actions from unmanaged devices are blocked outright.

## Password Managers
The company provides an enterprise password manager to every employee. Storing work credentials in browsers or plain text files violates this policy. Shared team credentials live only in designated vault collections with audited access.

## Service Accounts
Service accounts use rotated secrets of at least 32 random characters or certificate-based authentication. Secrets rotate automatically every 90 days through the secret manager; manual rotation exceptions expire after one quarter.

## Breach Response
Compromised credentials revoke immediately upon detection. Forced password resets accompany confirmed phishing clicks. Credential-stuffing attempts against customer-facing endpoints trigger rate-limit escalation and security review.

## Onboarding and Offboarding
New hires receive password-manager enrollment during day-one setup. Offboarding revokes credentials within 24 hours of exit, coordinated between People Operations and IT security.

## User Education
Security awareness training covers credential hygiene quarterly. Simulated phishing exercises run monthly with coaching assigned to clickers rather than punishment.

## Exceptions
Any exception to this policy requires written approval from the CISO with a maximum validity of 90 days and compensating controls documented.
""",
"security/access_control.md": """# Access Control Policy

## Principle
Access follows least privilege: every identity receives the minimum permissions necessary for current responsibilities. Access is granted through role-based templates rather than individual grants wherever possible.

## Joiner-Mover-Leaver
Joiners receive template entitlements on day one based on role. Movers trigger recertification of old-role access within 5 working days of transfer. Leavers lose all access within 24 hours of exit, including VPN, SaaS accounts, cloud consoles, and physical badges.

## Production Access
Production infrastructure access requires three prerequisites: completed security training within the last 12 months, manager approval recorded in the access system, and a justified business reason tied to an on-call rotation or project. Standing production access is discouraged; just-in-time elevation with automatic expiry after 4 hours is the default pattern.

## Privileged Accounts
Break-glass administrator accounts exist per critical system, sealed with dual-control envelopes. Use of break-glass triggers automatic post-use review within one business day. Shared administrative passwords are prohibited.

## Access Reviews
Quarterly access reviews cover all production systems, financial systems, and customer-data stores. Line managers certify their reports' entitlements; unreviewed access auto-revokes after two reminder cycles. Review evidence retains for audit.

## Segregation of Duties
Conflicting duties separate: requesters cannot approve their own access, developers cannot approve their own production deployments, and payment initiators cannot approve payments. Automated checks flag SoD conflicts weekly.

## Third-Party Access
Vendor and contractor access is time-boxed to contract duration, sponsored by an internal owner, and restricted to named systems. Third-party privileged access routes through the monitored bastion with session recording.

## Monitoring
Authentication logs stream to the security information event management platform. Anomalies such as impossible-travel logins, off-hours privilege escalation, and dormant-account revival alert the security operations center for triage within 15 minutes.

## Physical Access
Office badge tiers mirror digital roles. Visitor badges expire daily and require escort in restricted zones. Server rooms enforce two-person integrity for hardware changes.

## Exceptions
Time-bound exceptions require CISO approval, documented compensating controls, and calendar reminders for expiry.
""",
"security/security_incidents.md": """# Security Incident Classification and Response

## Classification Levels
Security incidents classify into three priorities. P1 denotes confirmed data breach involving customer or regulated data, ransomware detonation, or compromise of production credentials at scale. P2 denotes malware outbreak contained to endpoints, successful phishing with credential capture, or unauthorized access caught mid-exploitation. P3 denotes attempted attacks blocked by controls, isolated phishing emails reported before compromise, or policy violations without data exposure.

## Reporting Channels
The 24/7 security hotline and the security-alert email alias accept reports around the clock. Employees must report suspected incidents within 1 hour of discovery. Good-faith reporting never results in blame, even when the reporter clicked a malicious link.

## Regulatory Notification
P1 incidents involving personal data require notification to CERT-In within 6 hours of confirmation, following statutory cyber-incident timelines. Affected customers receive notification without undue delay and no later than 72 hours, coordinated with Legal. Regulated financial data breaches add sector-regulator notifications per contract obligations.

## Response Roles
The security operations center declares incidents and assigns an incident commander from the security on-call rota. For P1, a cross-functional bridge includes engineering leadership, legal counsel, communications, and the DPO. External forensics engage under retainer when evidence preservation demands it.

## Containment Playbooks
Standard containment order: isolate affected hosts from the network, rotate exposed credentials and keys, block attacker infrastructure at edge controls, preserve forensic images before remediation wipes state, then eradicate and validate recovery.

## Evidence Handling
Chain-of-custody records accompany all forensic artifacts. Timestamps use UTC with source synchronization noted. Evidence access restricts to named investigators.

## Communication Discipline
Only the communications lead speaks externally about P1/P2 incidents. Internal updates post hourly for P1. Speculation about attribution stays out of written channels until forensics conclude.

## Post-Incident Review
P1 reviews complete within 5 business days, P2 within 10. Reviews identify detection gaps, response friction, and control improvements with owners and deadlines tracked to closure.

## Exercises
Tabletop exercises run twice yearly covering breach, ransomware, and insider-threat scenarios with executive participation.
""",
"security/data_classification.md": """# Data Classification Policy

## Classification Levels
Company data classifies into four levels. Public means approved for external release. Internal means ordinary business information for employees. Confidential means sensitive business information whose disclosure harms the company competitively or legally. Restricted means regulated personal data, payment data, health data, credentials, and encryption keys.

## Default Handling
Documents default to Internal unless marked otherwise. Creators bear responsibility for correct classification at creation and review it when sharing externally. When in doubt, classify higher and consult the data steward.

## Restricted Data Examples
Restricted covers customer personally identifiable information, payment card data, employee records with government identifiers, authentication secrets, private keys, and production database dumps. Aggregated or anonymized derivatives may downgrade only after formal anonymization review.

## Encryption Requirements
Confidential and Restricted data encrypt in transit with TLS 1.2 or higher and at rest with AES-256 or equivalent. Restricted data adds field-level or tokenized protection for identifiers such as card numbers and national IDs. Keys live in the managed KMS with rotation every 12 months.

## Storage Locations
Restricted data stores only in approved systems listed in the data inventory: the production databases, the encrypted data warehouse zone, and the ticketing vault. Spreadsheets containing Restricted data on laptops violate policy; extracts use the de-identified analytics layer instead.

## Sharing Rules
Internal data shares freely inside the company. Confidential sharing requires business justification and recipient need-to-know. Restricted sharing outside approved systems requires DPO approval with a data-processing agreement in place.

## Retention and Deletion
Retention schedules: Restricted personal data purges 7 years after account closure unless legal hold applies; Confidential business records retain 10 years; Internal data reviews for deletion after 3 years of inactivity. Deletion jobs produce certificates that audit samples quarterly.

## Transfers
Cross-border transfers of Restricted data follow the transfer impact assessment process. Vendor processing of Restricted data requires signed DPAs and sub-processor disclosure.

## Labeling Tools
Email and document platforms apply sensitivity labels that persist through forwarding and enforce encryption-plus-watermark for Restricted attachments.

## Training
Role-based data-handling training assigns at onboarding and refreshes annually; handlers of Restricted data complete an additional module with assessment.
""",
"security/device_security.md": """# Device Security Policy

## Scope
This policy covers all devices accessing company systems: company-owned laptops, desktops, mobile phones, tablets, and personally owned devices used for work under BYOD.

## Company Laptops
Company laptops enforce full-disk encryption at provisioning, verified by the endpoint management agent. Screen lock activates after 5 minutes of inactivity with a password of at least 8 characters. Automatic patching installs OS and browser updates within 14 days of release; critical CVEs patch within 72 hours.

## Endpoint Protection
Every device runs the managed endpoint detection and response agent. Tamper protection prevents disabling. Devices failing check-in for 30 days lose VPN certificate trust until re-enrollment.

## Mobile Devices
Company mobile devices enroll in mobile device management with enforced passcodes, remote-wipe capability, and app allowlists for work profiles. Camera use is disabled inside secure manufacturing and lab areas.

## Bring Your Own Device
Personal devices used for email or chat must enroll in MDM with a work container separating corporate data. BYOD grants the company rights to wipe the work container only, never personal photos or messages. Rooted or jailbroken devices are denied access entirely.

## Lost or Stolen Devices
Lost devices are reported within 1 hour of discovery to the service desk and security hotline. Remote wipe executes immediately for lost company devices. Users should enable find-my-device features; physical recovery attempts belong to security, never individuals.

## Removable Media
USB storage is blocked by default. Exceptions for air-gapped transfer scenarios issue encrypted, serial-numbered drives from IT with usage logging. Personal USB drives connecting to company machines trigger an EDR alert.

## Home Networks and Travel
Remote workers should use WPA2/WPA3 home Wi-Fi with changed default router passwords. Public Wi-Fi requires the corporate VPN before any work traffic. International travelers consult the restricted-country list before carrying devices abroad; high-risk destinations use loaner clean devices.

## Disposal
Decommissioned devices return to IT for certified data destruction with serialized certificates before resale or recycling. Storage media leaving custody physically is destroyed by shredding or degaussing.

## Compliance Checks
Automated posture checks gate VPN access: encryption on, agent running, patches current, screen lock configured. Failed checks route to guided remediation with a 72-hour grace window.
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