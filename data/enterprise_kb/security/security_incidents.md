# Security Incident Classification and Response

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
