# Email DNS Records

Configure these records with the transactional email provider before launch.

## SPF

```text
Type: TXT
Name: @
Value: v=spf1 include:_spf.your-provider.example -all
```

## DKIM

```text
Type: CNAME
Name: provider1._domainkey
Value: provider1._domainkey.your-provider.example
```

## DMARC

```text
Type: TXT
Name: _dmarc
Value: v=DMARC1; p=quarantine; rua=mailto:dmarc@slop.dev; ruf=mailto:dmarc@slop.dev; adkim=s; aspf=s
```

Validate with the provider dashboard and send test password reset, welcome, and billing emails.
