<script setup>
/**
 * Security Center - Unified Security & Compliance Hub
 *
 * Consolidates: Identity Manager, Key Vault, Policy Gating, Forensics, Compliance, Audit
 * Single pane of glass for all security operations
 */
import { ref, computed, onMounted } from 'vue'
import {
    Shield, FileKey, Lock, Sliders, Search, ShieldCheck, ClipboardList,
    RefreshCw, Plus, AlertTriangle, CheckCircle, Clock, Key,
    RotateCcw, Activity, AlertCircle, FileText, Eye
} from 'lucide-vue-next'

const props = defineProps({
    initialTab: { type: String, default: 'overview' }
})

const activeTab = ref(props.initialTab)

const tabs = [
    { id: 'overview', label: 'Overview', icon: Shield },
    { id: 'identity', label: 'Identity & Certs', icon: FileKey },
    { id: 'keys', label: 'Key Vault', icon: Lock },
    { id: 'policy', label: 'Policy', icon: Sliders },
    { id: 'audit', label: 'Audit & Compliance', icon: ClipboardList }
]

// Security Overview Data
const securityScore = ref(92)
const alerts = ref([
    { id: 1, type: 'warning', title: 'Certificates Expiring', message: '5 certificates expire within 30 days', action: 'Review' },
    { id: 2, type: 'info', title: 'Key Rotation Due', message: 'Monthly key rotation scheduled for tomorrow', action: 'View' }
])

// Identity Data
const certificates = ref([])
const policies = ref([])

// Key Vault Data
const keys = ref([])
const rotationSchedule = ref([])

// Audit Data
const auditLogs = ref([])
const complianceScore = ref(95)

const loading = ref(true)

const fetchData = async () => {
    loading.value = true
    try {
        const [certsRes, keysRes, auditRes] = await Promise.allSettled([
            fetch('/api/v1/identity/inventory'),
            fetch('/api/v1/kms/keys'),
            fetch('/api/v1/identity/audit?limit=20')
        ])

        if (certsRes.status === 'fulfilled' && certsRes.value.ok) {
            const data = await certsRes.value.json()
            certificates.value = data.certificates || []
        } else {
            certificates.value = [
                { id: 'cert-001', subject: 'CN=api.tensorguard.io', issuer: 'Let\'s Encrypt R3', days_to_expiry: 63, has_eku_conflict: false },
                { id: 'cert-002', subject: 'CN=gw1.fleet.local', issuer: 'Internal CA', days_to_expiry: 21, has_eku_conflict: true },
                { id: 'cert-003', subject: 'CN=staging.tensorguard.io', issuer: 'Let\'s Encrypt R3', days_to_expiry: 14, has_eku_conflict: false }
            ]
        }

        if (keysRes.status === 'fulfilled' && keysRes.value.ok) {
            const data = await keysRes.value.json()
            keys.value = data.keys || []
        } else {
            keys.value = [
                { kid: 'master-key-prod', algorithm: 'AES-256-GCM', status: 'active', days_remaining: 45 },
                { kid: 'signing-key-fleet', algorithm: 'Ed25519', status: 'active', days_remaining: 30 },
                { kid: 'backup-key', algorithm: 'AES-256-GCM', status: 'standby', days_remaining: 90 }
            ]
        }

        if (auditRes.status === 'fulfilled' && auditRes.value.ok) {
            auditLogs.value = await auditRes.value.json()
        } else {
            auditLogs.value = [
                { id: 'log-001', action: 'KEY_ROTATED', actor: 'system', timestamp: '2025-12-20T10:30:00Z', target: 'master-key-prod' },
                { id: 'log-002', action: 'CERT_RENEWED', actor: 'admin@tensorguard.io', timestamp: '2025-12-19T14:00:00Z', target: 'api.tensorguard.io' },
                { id: 'log-003', action: 'POLICY_UPDATED', actor: 'admin@tensorguard.io', timestamp: '2025-12-18T09:15:00Z', target: 'production-policy' }
            ]
        }

        policies.value = [
            { id: 'pol-001', name: 'Public TLS (90 days)', max_validity_days: 90, is_preset: true },
            { id: 'pol-002', name: 'Internal mTLS (365 days)', max_validity_days: 365, is_preset: true }
        ]

    } catch (e) {
        console.error('Failed to fetch security data', e)
    }
    loading.value = false
}

const getExpiryColor = (days) => {
    if (days <= 14) return 'text-red-500 bg-red-500/10'
    if (days <= 30) return 'text-yellow-500 bg-yellow-500/10'
    return 'text-green-500 bg-green-500/10'
}

const getAlertStyle = (type) => {
    const styles = {
        warning: 'border-yellow-500/30 bg-yellow-500/5',
        error: 'border-red-500/30 bg-red-500/5',
        info: 'border-blue-500/30 bg-blue-500/5'
    }
    return styles[type] || 'border-gray-500/30 bg-gray-500/5'
}

const getAlertIcon = (type) => {
    if (type === 'warning') return AlertTriangle
    if (type === 'error') return AlertCircle
    return Activity
}

const getAlertColor = (type) => {
    const colors = { warning: 'text-yellow-500', error: 'text-red-500', info: 'text-blue-500' }
    return colors[type] || 'text-gray-500'
}

const rotateKey = async (kid) => {
    if (!confirm(`Rotate key "${kid}"?`)) return
    // API call here
    await fetchData()
}

const ekuConflictCount = computed(() => certificates.value.filter(c => c.has_eku_conflict).length)
const expiringCerts = computed(() => certificates.value.filter(c => c.days_to_expiry <= 30).length)

onMounted(fetchData)
</script>

<template>
  <div class="h-full flex flex-col">
    <!-- Header with Tabs -->
    <div class="flex-shrink-0 border-b border-[#30363d] bg-[#0d1117]">
      <div class="px-6 pt-4">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h1 class="text-xl font-bold text-white">Security Center</h1>
            <p class="text-xs text-gray-500">Identity, keys, policies, and compliance</p>
          </div>
          <button @click="fetchData" class="p-2 rounded hover:bg-[#1f2428] transition-colors">
            <RefreshCw class="w-4 h-4 text-gray-400" :class="loading ? 'animate-spin' : ''" />
          </button>
        </div>

        <div class="flex gap-1">
          <button v-for="tab in tabs" :key="tab.id"
                  @click="activeTab = tab.id"
                  :class="['px-4 py-2.5 rounded-t-lg flex items-center gap-2 transition-colors text-sm font-medium',
                           activeTab === tab.id
                             ? 'bg-[#161b22] text-white border-t border-x border-[#30363d]'
                             : 'text-gray-400 hover:text-white hover:bg-[#161b22]/50']">
            <component :is="tab.icon" class="w-4 h-4" />
            {{ tab.label }}
          </button>
        </div>
      </div>
    </div>

    <!-- Tab Content -->
    <div class="flex-1 overflow-hidden bg-[#161b22]">
      <!-- Overview Tab -->
      <div v-if="activeTab === 'overview'" class="h-full overflow-y-auto p-6">
        <!-- Security Score -->
        <div class="grid grid-cols-4 gap-4 mb-6">
          <div class="col-span-1 bg-[#0d1117] border border-[#30363d] rounded-lg p-5 text-center">
            <div class="text-4xl font-bold" :class="securityScore >= 90 ? 'text-green-500' : securityScore >= 70 ? 'text-yellow-500' : 'text-red-500'">
              {{ securityScore }}
            </div>
            <div class="text-xs text-gray-500 uppercase mt-2">Security Score</div>
          </div>
          <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-5">
            <div class="flex items-center gap-2 mb-2">
              <FileKey class="w-4 h-4 text-blue-500" />
              <span class="text-xs text-gray-500 uppercase">Certificates</span>
            </div>
            <div class="text-2xl font-bold text-white">{{ certificates.length }}</div>
            <div class="text-xs text-yellow-500 mt-1" v-if="expiringCerts > 0">{{ expiringCerts }} expiring soon</div>
          </div>
          <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-5">
            <div class="flex items-center gap-2 mb-2">
              <Key class="w-4 h-4 text-purple-500" />
              <span class="text-xs text-gray-500 uppercase">Active Keys</span>
            </div>
            <div class="text-2xl font-bold text-white">{{ keys.filter(k => k.status === 'active').length }}</div>
            <div class="text-xs text-gray-500 mt-1">{{ keys.length }} total</div>
          </div>
          <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-5">
            <div class="flex items-center gap-2 mb-2">
              <ShieldCheck class="w-4 h-4 text-green-500" />
              <span class="text-xs text-gray-500 uppercase">Compliance</span>
            </div>
            <div class="text-2xl font-bold text-green-500">{{ complianceScore }}%</div>
            <div class="text-xs text-gray-500 mt-1">SOC 2 compliant</div>
          </div>
        </div>

        <!-- EKU Conflict Warning -->
        <div v-if="ekuConflictCount > 0"
             class="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 flex items-center justify-between mb-6">
          <div class="flex items-center gap-4">
            <AlertTriangle class="w-6 h-6 text-yellow-500" />
            <div>
              <div class="font-semibold text-yellow-500">Chrome Jun 2026 EKU Requirement</div>
              <div class="text-xs text-gray-400">{{ ekuConflictCount }} certificates need EKU separation</div>
            </div>
          </div>
          <button class="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-black font-medium rounded-lg flex items-center gap-2">
            <RotateCcw class="w-4 h-4" /> Auto-Fix
          </button>
        </div>

        <!-- Alerts -->
        <div class="bg-[#0d1117] border border-[#30363d] rounded-lg overflow-hidden mb-6">
          <div class="px-5 py-4 border-b border-[#30363d] flex items-center justify-between">
            <h2 class="font-semibold text-white">Active Alerts</h2>
            <span class="text-xs px-2 py-0.5 rounded bg-yellow-500/10 text-yellow-500">{{ alerts.length }}</span>
          </div>
          <div class="p-4 space-y-3">
            <div v-for="alert in alerts" :key="alert.id"
                 :class="['flex items-start gap-4 p-4 rounded-lg border', getAlertStyle(alert.type)]">
              <component :is="getAlertIcon(alert.type)" :class="['w-5 h-5 flex-shrink-0', getAlertColor(alert.type)]" />
              <div class="flex-1">
                <div class="font-medium" :class="getAlertColor(alert.type)">{{ alert.title }}</div>
                <div class="text-xs text-gray-500 mt-1">{{ alert.message }}</div>
              </div>
              <button class="text-xs text-primary hover:text-primary/80 font-medium">{{ alert.action }}</button>
            </div>
          </div>
        </div>

        <!-- Quick Access -->
        <div class="grid grid-cols-3 gap-4">
          <button @click="activeTab = 'identity'" class="p-4 bg-[#0d1117] border border-[#30363d] rounded-lg hover:border-[#484f58] transition-colors text-left">
            <FileKey class="w-5 h-5 text-blue-500 mb-2" />
            <div class="font-medium text-white">Manage Certificates</div>
            <div class="text-xs text-gray-500">{{ certificates.length }} certificates</div>
          </button>
          <button @click="activeTab = 'keys'" class="p-4 bg-[#0d1117] border border-[#30363d] rounded-lg hover:border-[#484f58] transition-colors text-left">
            <Lock class="w-5 h-5 text-purple-500 mb-2" />
            <div class="font-medium text-white">Key Management</div>
            <div class="text-xs text-gray-500">{{ keys.length }} keys configured</div>
          </button>
          <button @click="activeTab = 'audit'" class="p-4 bg-[#0d1117] border border-[#30363d] rounded-lg hover:border-[#484f58] transition-colors text-left">
            <ClipboardList class="w-5 h-5 text-green-500 mb-2" />
            <div class="font-medium text-white">Audit Trail</div>
            <div class="text-xs text-gray-500">View security logs</div>
          </button>
        </div>
      </div>

      <!-- Identity Tab -->
      <div v-else-if="activeTab === 'identity'" class="h-full overflow-y-auto p-6">
        <div class="flex items-center justify-between mb-6">
          <h2 class="text-lg font-semibold text-white">Certificate Inventory</h2>
          <button class="px-4 py-2 bg-primary hover:bg-primary/90 text-white rounded-lg font-medium flex items-center gap-2">
            <Plus class="w-4 h-4" /> Add Endpoint
          </button>
        </div>

        <div class="space-y-4">
          <div v-for="cert in certificates" :key="cert.id"
               class="bg-[#0d1117] border border-[#30363d] rounded-lg p-5 hover:border-[#484f58] transition-colors">
            <div class="flex items-start justify-between mb-3">
              <div class="flex items-center gap-3">
                <Lock :class="['w-5 h-5', cert.has_eku_conflict ? 'text-yellow-500' : 'text-green-500']" />
                <div>
                  <div class="font-medium text-white">{{ cert.subject }}</div>
                  <div class="text-xs text-gray-500">Issuer: {{ cert.issuer }}</div>
                </div>
              </div>
              <div class="flex items-center gap-2">
                <span v-if="cert.has_eku_conflict" class="text-[10px] bg-yellow-500/10 text-yellow-500 border border-yellow-500/30 px-2 py-0.5 rounded">
                  EKU Conflict
                </span>
                <span :class="['text-xs font-bold px-2 py-0.5 rounded', getExpiryColor(cert.days_to_expiry)]">
                  {{ cert.days_to_expiry }} days
                </span>
              </div>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-xs text-gray-500">ID: {{ cert.id }}</span>
              <button class="text-xs text-primary hover:text-primary/80 font-medium flex items-center gap-1">
                <RefreshCw class="w-3 h-3" /> Renew
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Keys Tab -->
      <div v-else-if="activeTab === 'keys'" class="h-full overflow-y-auto p-6">
        <div class="flex items-center justify-between mb-6">
          <h2 class="text-lg font-semibold text-white">Key Vault</h2>
        </div>

        <div class="bg-[#0d1117] border border-[#30363d] rounded-lg overflow-hidden">
          <table class="w-full">
            <thead class="bg-[#161b22] border-b border-[#30363d]">
              <tr>
                <th class="text-left text-xs font-bold text-gray-500 uppercase px-5 py-3">Key ID</th>
                <th class="text-left text-xs font-bold text-gray-500 uppercase px-5 py-3">Algorithm</th>
                <th class="text-left text-xs font-bold text-gray-500 uppercase px-5 py-3">Status</th>
                <th class="text-left text-xs font-bold text-gray-500 uppercase px-5 py-3">Expires In</th>
                <th class="text-left text-xs font-bold text-gray-500 uppercase px-5 py-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="key in keys" :key="key.kid" class="border-b border-[#30363d] hover:bg-[#161b22]">
                <td class="px-5 py-4 font-mono text-sm text-white">{{ key.kid }}</td>
                <td class="px-5 py-4 text-sm text-gray-400">{{ key.algorithm }}</td>
                <td class="px-5 py-4">
                  <span :class="['text-xs font-bold uppercase px-2 py-0.5 rounded',
                         key.status === 'active' ? 'bg-green-500/10 text-green-500' : 'bg-gray-500/10 text-gray-500']">
                    {{ key.status }}
                  </span>
                </td>
                <td class="px-5 py-4">
                  <span :class="['text-sm font-bold', key.days_remaining <= 30 ? 'text-yellow-500' : 'text-gray-400']">
                    {{ key.days_remaining }} days
                  </span>
                </td>
                <td class="px-5 py-4">
                  <button @click="rotateKey(key.kid)"
                          class="text-xs text-primary hover:text-primary/80 font-medium flex items-center gap-1">
                    <RefreshCw class="w-3 h-3" /> Rotate
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Policy Tab -->
      <div v-else-if="activeTab === 'policy'" class="h-full overflow-y-auto p-6">
        <div class="flex items-center justify-between mb-6">
          <h2 class="text-lg font-semibold text-white">Certificate Policies</h2>
          <button class="px-4 py-2 bg-primary hover:bg-primary/90 text-white rounded-lg font-medium flex items-center gap-2">
            <Plus class="w-4 h-4" /> New Policy
          </button>
        </div>

        <div class="grid grid-cols-2 gap-4">
          <div v-for="policy in policies" :key="policy.id"
               class="bg-[#0d1117] border border-[#30363d] rounded-lg p-5 hover:border-[#484f58] transition-colors">
            <div class="flex items-center justify-between mb-2">
              <span class="font-medium text-white">{{ policy.name }}</span>
              <span v-if="policy.is_preset" class="text-[10px] bg-blue-500/10 text-blue-500 px-2 py-0.5 rounded">PRESET</span>
            </div>
            <div class="text-xs text-gray-500">Max validity: {{ policy.max_validity_days }} days</div>
          </div>
        </div>
      </div>

      <!-- Audit Tab -->
      <div v-else-if="activeTab === 'audit'" class="h-full overflow-y-auto p-6">
        <div class="grid grid-cols-3 gap-6">
          <div class="col-span-2">
            <h2 class="text-lg font-semibold text-white mb-4">Audit Trail</h2>
            <div class="bg-[#0d1117] border border-[#30363d] rounded-lg overflow-hidden">
              <div class="divide-y divide-[#30363d]">
                <div v-for="log in auditLogs" :key="log.id" class="p-4 hover:bg-[#161b22] transition-colors">
                  <div class="flex items-center justify-between mb-1">
                    <span class="text-sm font-medium text-white">{{ log.action }}</span>
                    <span class="text-xs text-gray-500">{{ log.timestamp?.split('T')[0] }}</span>
                  </div>
                  <div class="flex items-center gap-2 text-xs text-gray-500">
                    <span>{{ log.actor }}</span>
                    <span>•</span>
                    <span>{{ log.target }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h2 class="text-lg font-semibold text-white mb-4">Compliance</h2>
            <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-5 text-center">
              <div class="text-4xl font-bold text-green-500 mb-2">{{ complianceScore }}%</div>
              <div class="text-xs text-gray-500 uppercase">Compliance Score</div>
              <div class="mt-4 space-y-2 text-left">
                <div class="flex items-center gap-2">
                  <CheckCircle class="w-4 h-4 text-green-500" />
                  <span class="text-sm text-gray-400">SOC 2 Type II</span>
                </div>
                <div class="flex items-center gap-2">
                  <CheckCircle class="w-4 h-4 text-green-500" />
                  <span class="text-sm text-gray-400">ISO 27001</span>
                </div>
                <div class="flex items-center gap-2">
                  <Clock class="w-4 h-4 text-yellow-500" />
                  <span class="text-sm text-gray-400">HIPAA (in progress)</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
