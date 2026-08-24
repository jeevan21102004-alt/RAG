# Device Security Policy

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
