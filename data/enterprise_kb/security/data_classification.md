# Data Classification Policy

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
