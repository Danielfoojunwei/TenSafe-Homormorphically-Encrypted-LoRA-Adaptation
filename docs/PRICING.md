# TenSafe Pricing

**Effective Date:** 2026-02-03

---

## Pricing Tiers

### Free Tier

**$0/month** - Perfect for getting started

| Feature | Limit |
|---------|-------|
| API Requests | 10,000/month |
| Tokens Processed | 100,000/month |
| Training Steps | 100/day |
| Model Storage | 1 GB |
| Concurrent Jobs | 1 |
| Support | Community |

**Included:**
- Full API access
- Differential privacy (DP-SGD)
- Basic HE-LoRA inference
- Community Discord access
- Documentation

**Not Included:**
- SLA guarantee
- Priority support
- SSO/SAML
- Advanced analytics

---

### Pro Tier

**$99/month** or **$79/month** (annual billing)

| Feature | Limit |
|---------|-------|
| API Requests | 500,000/month |
| Tokens Processed | 10,000,000/month |
| Training Steps | 10,000/day |
| Model Storage | 50 GB |
| Concurrent Jobs | 5 |
| Support | Email (24h response) |

**Everything in Free, plus:**
- Higher rate limits
- Webhook integrations
- Usage analytics dashboard
- Priority queue for inference
- 99.5% uptime SLA

**Usage-Based Overage:**
- Additional tokens: $0.02 per 1,000 tokens
- Additional training steps: $0.10 per 100 steps
- Additional storage: $0.05 per GB/month

---

### Business Tier

**$499/month** or **$399/month** (annual billing)

| Feature | Limit |
|---------|-------|
| API Requests | 5,000,000/month |
| Tokens Processed | 100,000,000/month |
| Training Steps | Unlimited |
| Model Storage | 500 GB |
| Concurrent Jobs | 20 |
| Support | Priority (4h response) |

**Everything in Pro, plus:**
- SSO/SAML integration
- RBAC (Role-Based Access Control)
- Audit log export (SIEM integration)
- Dedicated inference endpoints
- Custom rate limits
- 99.9% uptime SLA
- Slack Connect support

**Usage-Based Overage:**
- Additional tokens: $0.015 per 1,000 tokens
- Additional storage: $0.03 per GB/month

---

### Enterprise Tier

**Custom Pricing** - Contact sales@tensafe.io

| Feature | Limit |
|---------|-------|
| API Requests | Custom |
| Tokens Processed | Custom |
| Training Steps | Unlimited |
| Model Storage | Custom |
| Concurrent Jobs | Custom |
| Support | Dedicated (1h response) |

**Everything in Business, plus:**
- Dedicated infrastructure
- On-premise deployment option
- Custom SLA (up to 99.99%)
- 24/7 phone support
- Dedicated success manager
- Custom integrations
- Security review & compliance support
- Training & onboarding sessions
- Volume discounts

---

## Feature Comparison

| Feature | Free | Pro | Business | Enterprise |
|---------|------|-----|----------|------------|
| **Privacy Features** |
| Differential Privacy | Yes | Yes | Yes | Yes |
| HE-LoRA Inference | Basic | Full | Full | Full |
| Post-Quantum Crypto | - | Yes | Yes | Yes |
| TGSP Packaging | - | Yes | Yes | Yes |
| **Infrastructure** |
| API Access | Yes | Yes | Yes | Yes |
| vLLM Backend | - | Yes | Yes | Yes |
| Distributed Training | - | - | Yes | Yes |
| Dedicated Endpoints | - | - | Yes | Yes |
| On-Premise | - | - | - | Yes |
| **Security & Compliance** |
| SSO/SAML | - | - | Yes | Yes |
| RBAC | Basic | Basic | Full | Full |
| Audit Logs | 7 days | 30 days | 90 days | Custom |
| SIEM Export | - | - | Yes | Yes |
| SOC 2 Report | - | - | Yes | Yes |
| BAA (HIPAA) | - | - | - | Yes |
| **Support** |
| Documentation | Yes | Yes | Yes | Yes |
| Community Discord | Yes | Yes | Yes | Yes |
| Email Support | - | Yes | Yes | Yes |
| Priority Support | - | - | Yes | Yes |
| Phone Support | - | - | - | Yes |
| Dedicated Manager | - | - | - | Yes |
| **SLA** |
| Uptime Guarantee | - | 99.5% | 99.9% | 99.99% |
| Response Time | - | 24h | 4h | 1h |

---

## Usage-Based Pricing Details

### Token Pricing

Tokens are counted for both input and output:

| Operation | Token Count |
|-----------|-------------|
| Training (input) | 1x |
| Inference (input) | 1x |
| Inference (output) | 1x |
| HE-LoRA (encrypted) | 1.5x |

### GPU Hours

For training operations:

| GPU Type | Price/Hour |
|----------|------------|
| A100 40GB | $2.50 |
| A100 80GB | $3.50 |
| H100 80GB | $4.50 |

*GPU pricing applies to Business and Enterprise tiers only.*

### Storage

| Storage Type | Price/GB/Month |
|--------------|----------------|
| Model Checkpoints | $0.05 |
| Encrypted Artifacts | $0.08 |
| Audit Logs | $0.02 |

---

## Billing

### Payment Methods

- Credit/Debit Cards (Visa, Mastercard, Amex)
- ACH Bank Transfer (US only)
- Wire Transfer (Enterprise)
- Annual invoicing (Enterprise)

### Billing Cycle

- **Monthly:** Billed on the 1st of each month
- **Annual:** Billed upfront, 20% discount

### Overage Handling

1. **Soft Limit (80%):** Warning notification
2. **Hard Limit (100%):**
   - Free: Requests blocked until reset
   - Paid: Overage charges applied

### Refund Policy

- Pro-rated refunds for annual plans within 30 days
- No refunds for monthly plans
- Usage-based charges are non-refundable

---

## Frequently Asked Questions

### Can I change tiers mid-cycle?

**Upgrades:** Immediate, pro-rated for the remainder of the cycle.
**Downgrades:** Effective at the next billing cycle.

### What happens if I exceed my limits?

- **Free Tier:** Requests are rate-limited until the next reset
- **Paid Tiers:** Overage charges apply at the rates listed above

### Do you offer discounts?

- **Annual billing:** 20% discount
- **Startups:** Contact us for startup program
- **Non-profits:** 50% discount (verification required)
- **Education:** Free Pro tier for academic research

### Is there a free trial?

Yes! All new accounts start with a 14-day Pro tier trial, no credit card required.

### How do I estimate my costs?

Use our [Pricing Calculator](https://tensafe.io/pricing/calculator) to estimate costs based on your expected usage.

---

## Contact Sales

For Enterprise pricing or custom requirements:

- **Email:** sales@tensafe.io
- **Phone:** +1 (888) TENSAFE
- **Schedule a demo:** https://tensafe.io/demo

---

*Prices are in USD and subject to change with 30 days notice for existing customers.*
