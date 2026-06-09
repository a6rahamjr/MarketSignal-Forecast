# Security Policy

## Supported Version

Security fixes are applied to the latest release on the default branch.

## Reporting

Do not open a public issue for a suspected vulnerability. Use GitHub private vulnerability
reporting for the repository and include:

- Affected version or commit
- Reproduction steps
- Expected and observed behavior
- Impact assessment
- Any suggested mitigation

## Deployment Notes

- Set `MARKETSIGNAL_ENV=production`.
- Set a strong `MARKETSIGNAL_API_KEY`.
- Restrict `MARKETSIGNAL_ALLOWED_HOSTS`.
- Put the API behind TLS, rate limiting, and a hard request-size limit.
- Mount model artifacts read-only.
- Do not expose training commands through the API.

The model loader accepts only the documented JSON/NPZ format. Pickle and joblib artifacts are not
supported.
