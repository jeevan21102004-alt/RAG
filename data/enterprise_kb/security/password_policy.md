# Password and Authentication Policy

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
