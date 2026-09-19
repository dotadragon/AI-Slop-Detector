export const meta = {
  name: 'slop-launch-readiness',
  description: 'Ruflo-style launch checklist orchestration for SLOP. Codex performs edits; Ruflo tracks phases and evidence.',
  phases: [
    { title: 'Must-have audit' },
    { title: 'Implementation verification' },
    { title: 'Launch evidence report' }
  ]
}

phase('Must-have audit')
log('Check analytics, SEO, legal, security, email, monitoring, billing, and launch-day gates.')

phase('Implementation verification')
const verification = await parallel([
  () => agent('Run backend tests for auth, security, and detection. Return pass/fail and notable warnings.', { schema: { type: 'object', properties: { passed: { type: 'boolean' }, summary: { type: 'string' } }, required: ['passed', 'summary'] } }),
  () => agent('Run frontend TypeScript/Vite build. Return pass/fail and notable warnings.', { schema: { type: 'object', properties: { passed: { type: 'boolean' }, summary: { type: 'string' } }, required: ['passed', 'summary'] } }),
  () => agent('Review launch-readiness docs for external-account placeholders. Return unresolved external dependencies.', { schema: { type: 'object', properties: { unresolved: { type: 'array', items: { type: 'string' } } }, required: ['unresolved'] } })
])

phase('Launch evidence report')
return { workflow: 'slop-launch-readiness', verification }
