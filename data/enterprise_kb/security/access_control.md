# Access Control Policy

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
