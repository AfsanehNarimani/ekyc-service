# eKYC Service Staged Rollout Plan

## Document information

- Service: eKYC Service
- API version: 0.1.0
- Primary endpoint: `POST /api/verify-video`
- Authentication: Keycloak JWT
- Deployment model: Kubernetes behind Istio
- Passive liveness mode: Report-only

## Release objective

Deploy the video verification service gradually while controlling:

- technical failures;
- CPU and memory saturation;
- verification latency;
- capacity rejection rate;
- unexpected decision changes;
- model and configuration errors.

## Required deployment configuration

The following values must be supplied by the deployment environment:

```text
KEYCLOAK_AUTH_ENABLED=true
KEYCLOAK_ISSUER=<realm issuer>
KEYCLOAK_JWKS_URL=<realm JWKS URL>
KEYCLOAK_ALLOWED_CLIENT_ID=<backend client ID>
KEYCLOAK_CLIENT_ID_CLAIM=azp
KEYCLOAK_ALGORITHM=RS256

PASSIVE_LIVENESS_BACKEND=onnx
PASSIVE_LIVENESS_DECISION_MODE=report_only

MAX_CONCURRENT_VERIFICATIONS=2
CAPACITY_RETRY_AFTER_SECONDS=3
