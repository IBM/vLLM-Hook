# Security

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues, pull requests or
discussions.**

Report it privately on GitHub: on this repository's **Security** tab, choose **Report a
vulnerability** ([how it works](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability#privately-reporting-a-security-vulnerability)).
You can also report it to the IBM Product Security Incident Response Team (PSIRT), by e-mail or
anonymous form, following the
[IBM Security Vulnerability Management](https://www.ibm.com/support/pages/ibm-security-vulnerability-management)
pages.

Please include:

- the MIA commit and vLLM version;
- whether the server runs offline (`MiaLLM`) or served (`vllm serve`), with CUDA graphs or
  `enforce_eager=True`;
- the steps to reproduce, ideally a short script;
- what an attacker needs (network access, a file on the server, local access) and what they gain.

## Supported versions

Fixes land on `main`. Use the latest `main` or the latest release.
