# API access and webhooks
The REST API allows 600 requests per minute per workspace; above that you receive HTTP 429 with a Retry-After header. Webhooks are sent with an HMAC-SHA256 signature in the X-HelpHub-Signature header.

If your endpoint returns a non-2xx status we retry up to 8 times with exponential backoff over 24 hours, then disable the webhook and email the workspace owner.
