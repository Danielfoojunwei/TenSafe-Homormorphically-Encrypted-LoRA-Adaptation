<script setup>
import { ref, onMounted, computed } from 'vue'
import {
    ShieldCheck, Key, Server, RefreshCw, AlertTriangle, CheckCircle,
    Clock, Plus, FileKey, Lock, Unlock, Shield, RotateCcw,
    AlertCircle, Settings, Scan, Activity
} from 'lucide-vue-next'

const inventory = ref({ endpoints: [], certificates: [], expiry_summary: {} })
const policies = ref([])
const renewals = ref([])
const riskAnalysis = ref(null)
const loading = ref(true)
const scanning = ref(false)
const migratingEku = ref(false)
const selectedEndpoint = ref(null)
const showPolicyModal = ref(false)
const errorMessage = ref('')

// New policy form
const newPolicy = ref({
    name: '',
    description: '',
    max_validity_days: 90,
    renewal_window_days: 30,
    allow_server_auth: true,
    allow_client_auth: false,
    require_eku_separation: true,
    require_public_trust: true,
    acme_challenge_type: 'http-01',
    preset_name: ''
})

const presetPolicies = [
    { id: 'public', name: 'Public TLS (90 days)' },
    { id: 'mtls', name: 'Private mTLS (365 days)' },
    { id: 'short-lived', name: 'Short-lived (30 days)' }
]

const fetchInventory = async () => {
    loading.value = true
    errorMessage.value = ''
    try {
        const [invRes, polRes, renRes] = await Promise.all([
            fetch('/api/v1/identity/inventory'),
            fetch('/api/v1/identity/policies'),
            fetch('/api/v1/identity/renewals')
        ])

        if (!invRes.ok || !polRes.ok || !renRes.ok) {
            throw new Error('Backend unavailable')
        }

        inventory.value = await invRes.json()
        policies.value = await polRes.json()
        renewals.value = await renRes.json()
    } catch (e) {
        console.error("Failed to fetch identity data", e)
        inventory.value = { endpoints: [], certificates: [], expiry_summary: {} }
        policies.value = []
        renewals.value = []
        errorMessage.value = 'Unable to load identity data. Check backend connectivity and authentication.'
    }
    loading.value = false
}

const fetchRiskAnalysis = async () => {
    try {
        const res = await fetch('/api/v1/identity/risk')
        if (res.ok) riskAnalysis.value = await res.json()
    } catch (e) {
        console.error("Failed to fetch risk analysis", e)
    }
}

const requestScan = async (fleetId) => {
    scanning.value = true
    try {
        await fetch(`/api/v1/identity/scan/request?fleet_id=${fleetId}`, { method: 'POST' })
        alert("Scan requested! Agents will report certificates shortly.")
    } catch (e) {
        console.error("Failed to request scan", e)
    }
    scanning.value = false
}

const executeEkuMigration = async () => {
    if (!confirm("Execute EKU split migration? This will schedule renewals for all certificates with EKU conflicts.")) return

    migratingEku.value = true
    try {
        const res = await fetch('/api/v1/identity/migrations/eku-split', { method: 'POST' })
        if (res.ok) {
            const data = await res.json()
            alert(`Migration started! ${data.jobs_created} renewal jobs created.`)
            await fetchInventory()
        }
    } catch (e) {
        console.error("Failed to execute migration", e)
    }
    migratingEku.value = false
}

const createPolicy = async () => {
    try {
        const payload = newPolicy.value.preset_name
            ? { preset_name: newPolicy.value.preset_name, name: newPolicy.value.name }
            : newPolicy.value

        const res = await fetch('/api/v1/identity/policies', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        if (res.ok) {
            showPolicyModal.value = false
            await fetchInventory()
        }
    } catch (e) {
        console.error("Failed to create policy", e)
    }
}

const scheduleRenewal = async (endpointId, policyId) => {
    try {
        const res = await fetch('/api/v1/identity/renewals/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ endpoint_ids: [endpointId], policy_id: policyId })
        })
        if (res.ok) {
            await fetchInventory()
            alert("Renewal scheduled!")
        }
    } catch (e) {
        console.error("Failed to schedule renewal", e)
    }
}

const getExpiryColor = (days) => {
    if (days <= 7) return 'text-red-500 bg-red-500/10'
    if (days <= 30) return 'text-yellow-500 bg-yellow-500/10'
    return 'text-green-500 bg-green-500/10'
}

const getCriticalityColor = (crit) => {
    const colors = { high: 'text-red-500', medium: 'text-yellow-500', low: 'text-green-500' }
    return colors[crit] || 'text-gray-500'
}

const ekuConflictCount = computed(() => inventory.value.certificates.filter(c => c.has_eku_conflict).length)

onMounted(() => {
    fetchInventory()
    fetchRiskAnalysis()
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between border-b border-[#333] pb-6">
       <div>
         <h2 class="text-2xl font-bold flex items-center gap-3">
             <ShieldCheck class="w-7 h-7 text-green-500" />
             Machine Identity Guard
         </h2>
         <span class="text-xs text-gray-500">Certificate Lifecycle, mTLS & ACME Automation</span>
       </div>
       <div class="flex gap-3">
           <button @click="requestScan('default')" :disabled="scanning" class="btn btn-secondary">
               <Scan class="w-4 h-4 mr-2" :class="scanning ? 'animate-pulse' : ''" />
               {{ scanning ? 'Scanning...' : 'Scan Certificates' }}
           </button>
           <button @click="showPolicyModal = true" class="btn btn-primary">
               <Plus class="w-4 h-4 mr-2" /> New Policy
           </button>
       </div>
    </div>

    <!-- Summary Cards -->
    <div class="grid grid-cols-4 gap-4">
        <div class="bg-[#111] border border-[#333] rounded-lg p-4">
            <div class="flex items-center gap-3 mb-2">
                <Server class="w-5 h-5 text-blue-500" />
                <span class="text-xs text-gray-500 uppercase">Endpoints</span>
            </div>
            <div class="text-2xl font-bold text-white">{{ inventory.endpoints.length }}</div>
        </div>
        <div class="bg-[#111] border border-[#333] rounded-lg p-4">
            <div class="flex items-center gap-3 mb-2">
                <AlertCircle class="w-5 h-5 text-red-500" />
                <span class="text-xs text-gray-500 uppercase">Critical Expiry</span>
            </div>
            <div class="text-2xl font-bold text-red-500">{{ inventory.expiry_summary.critical || 0 }}</div>
        </div>
        <div class="bg-[#111] border border-[#333] rounded-lg p-4">
            <div class="flex items-center gap-3 mb-2">
                <AlertTriangle class="w-5 h-5 text-yellow-500" />
                <span class="text-xs text-gray-500 uppercase">EKU Conflicts</span>
            </div>
            <div class="text-2xl font-bold text-yellow-500">{{ ekuConflictCount }}</div>
        </div>
        <div class="bg-[#111] border border-[#333] rounded-lg p-4">
            <div class="flex items-center gap-3 mb-2">
                <Activity class="w-5 h-5 text-green-500" />
                <span class="text-xs text-gray-500 uppercase">Active Renewals</span>
            </div>
            <div class="text-2xl font-bold text-green-500">{{ renewals.filter(r => r.status === 'pending' || r.status === 'in_progress').length }}</div>
        </div>
    </div>

    <div v-if="errorMessage" class="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-sm text-red-200">
        {{ errorMessage }}
    </div>

    <!-- Chrome Jun 2026 Warning -->
    <div v-if="ekuConflictCount > 0" class="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 flex items-center justify-between">
        <div class="flex items-center gap-4">
            <AlertTriangle class="w-6 h-6 text-yellow-500" />
            <div>
                <div class="font-bold text-yellow-500">Chrome Jun 2026 EKU Requirement</div>
                <div class="text-xs text-gray-400">{{ ekuConflictCount }} certificates have both serverAuth + clientAuth EKU. Chrome will reject these after June 2026.</div>
            </div>
        </div>
        <button @click="executeEkuMigration" :disabled="migratingEku" class="btn bg-yellow-600 hover:bg-yellow-700 text-black font-bold">
            <RotateCcw class="w-4 h-4 mr-2" :class="migratingEku ? 'animate-spin' : ''" />
            {{ migratingEku ? 'Migrating...' : 'Auto-Split EKU' }}
        </button>
    </div>

    <div class="grid grid-cols-3 gap-6">
        <!-- Endpoints & Certificates -->
        <div class="col-span-2 space-y-4">
            <h3 class="text-xs font-bold text-gray-500 uppercase">Certificate Inventory</h3>
            <div v-if="loading" class="flex justify-center py-12">
                <div class="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
            </div>
            <div v-else class="space-y-3">
                <div v-for="cert in inventory.certificates" :key="cert.id"
                     class="bg-[#0d1117] border border-[#30363d] rounded-lg p-4 hover:border-primary/50 transition-colors">
                    <div class="flex items-start justify-between mb-3">
                        <div class="flex items-center gap-3">
                            <div :class="['w-10 h-10 rounded-lg flex items-center justify-center', cert.has_eku_conflict ? 'bg-yellow-500/10' : 'bg-green-500/10']">
                                <Lock :class="['w-5 h-5', cert.has_eku_conflict ? 'text-yellow-500' : 'text-green-500']" />
                            </div>
                            <div>
                                <div class="font-bold text-white text-sm">{{ cert.subject }}</div>
                                <div class="text-[10px] text-gray-500">Issuer: {{ cert.issuer }}</div>
                            </div>
                        </div>
                        <div class="flex items-center gap-2">
                            <span v-if="cert.has_eku_conflict" class="text-[10px] bg-yellow-500/10 text-yellow-500 border border-yellow-500/30 px-2 py-0.5 rounded">
                                EKU Conflict
                            </span>
                            <span :class="['text-[10px] font-bold px-2 py-0.5 rounded', getExpiryColor(cert.days_to_expiry)]">
                                {{ cert.days_to_expiry }} days
                            </span>
                        </div>
                    </div>
                    <div class="flex items-center justify-between">
                        <div class="text-xs text-gray-500">
                            Expires: {{ cert.not_after.split('T')[0] }}
                        </div>
                        <button @click="scheduleRenewal(cert.endpoint_id, policies[0]?.id)" class="btn btn-sm btn-secondary">
                            <RefreshCw class="w-3 h-3 mr-1" /> Renew
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Policies & Renewals -->
        <div class="space-y-6">
            <!-- Policies -->
            <div>
                <h3 class="text-xs font-bold text-gray-500 uppercase mb-4">Certificate Policies</h3>
                <div class="space-y-2">
                    <div v-for="policy in policies" :key="policy.id"
                         class="bg-[#111] border border-[#333] rounded-lg p-3 hover:border-primary/30 transition-colors">
                        <div class="flex items-center justify-between mb-1">
                            <span class="font-bold text-sm text-white">{{ policy.name }}</span>
                            <span v-if="policy.is_preset" class="text-[8px] bg-blue-500/10 text-blue-500 px-1.5 py-0.5 rounded">PRESET</span>
                        </div>
                        <div class="text-[10px] text-gray-500">
                            Max: {{ policy.max_validity_days }}d / Renew: {{ policy.renewal_window_days }}d before expiry
                        </div>
                    </div>
                </div>
            </div>

            <!-- Active Renewals -->
            <div>
                <h3 class="text-xs font-bold text-gray-500 uppercase mb-4">Active Renewal Jobs</h3>
                <div v-if="renewals.length === 0" class="text-center py-8 text-gray-600">
                    <Clock class="w-8 h-8 mx-auto mb-2 opacity-30" />
                    <div class="text-xs">No active renewals</div>
                </div>
                <div v-else class="space-y-2">
                    <div v-for="job in renewals" :key="job.id"
                         class="bg-[#111] border border-[#333] rounded-lg p-3">
                        <div class="flex items-center justify-between mb-1">
                            <span class="font-mono text-xs text-gray-400">{{ job.endpoint_id }}</span>
                            <span :class="['text-[10px] font-bold uppercase px-2 py-0.5 rounded',
                                job.status === 'completed' ? 'bg-green-500/10 text-green-500' :
                                job.status === 'failed' ? 'bg-red-500/10 text-red-500' :
                                'bg-yellow-500/10 text-yellow-500']">
                                {{ job.status }}
                            </span>
                        </div>
                        <div class="text-[10px] text-gray-500">{{ job.status_message }}</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Create Policy Modal -->
    <div v-if="showPolicyModal" class="fixed inset-0 bg-black/90 flex items-center justify-center z-50 backdrop-blur-sm p-4">
        <div class="bg-[#0f0f0f] border border-primary/30 w-full max-w-lg rounded-xl shadow-2xl overflow-hidden">
            <div class="p-6 border-b border-[#222]">
                <h3 class="text-xl font-bold text-white">Create Certificate Policy</h3>
                <p class="text-[10px] text-gray-500 uppercase mt-1">Define lifecycle rules for certificates</p>
            </div>
            <div class="p-6 space-y-4">
                <div>
                    <label class="text-[10px] text-gray-500 font-bold uppercase mb-2 block">Use Preset</label>
                    <select v-model="newPolicy.preset_name" class="w-full bg-[#111] border border-[#333] rounded px-4 py-3 text-sm focus:border-primary outline-none cursor-pointer">
                        <option value="">-- Custom Policy --</option>
                        <option v-for="preset in presetPolicies" :key="preset.id" :value="preset.id">{{ preset.name }}</option>
                    </select>
                </div>
                <div v-if="!newPolicy.preset_name">
                    <label class="text-[10px] text-gray-500 font-bold uppercase mb-2 block">Policy Name</label>
                    <input v-model="newPolicy.name" type="text" class="w-full bg-[#111] border border-[#333] rounded px-4 py-3 text-sm focus:border-primary outline-none" placeholder="e.g. Production API TLS">
                </div>
                <div v-if="!newPolicy.preset_name" class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="text-[10px] text-gray-500 font-bold uppercase mb-2 block">Max Validity (days)</label>
                        <input v-model.number="newPolicy.max_validity_days" type="number" class="w-full bg-[#111] border border-[#333] rounded px-4 py-3 text-sm focus:border-primary outline-none" min="1" max="365">
                    </div>
                    <div>
                        <label class="text-[10px] text-gray-500 font-bold uppercase mb-2 block">Renewal Window (days)</label>
                        <input v-model.number="newPolicy.renewal_window_days" type="number" class="w-full bg-[#111] border border-[#333] rounded px-4 py-3 text-sm focus:border-primary outline-none" min="1" max="90">
                    </div>
                </div>
                <div v-if="!newPolicy.preset_name" class="space-y-2">
                    <label class="flex items-center gap-2 cursor-pointer">
                        <input type="checkbox" v-model="newPolicy.require_eku_separation" class="accent-primary">
                        <span class="text-sm text-gray-400">Require EKU Separation (serverAuth vs clientAuth)</span>
                    </label>
                    <label class="flex items-center gap-2 cursor-pointer">
                        <input type="checkbox" v-model="newPolicy.require_public_trust" class="accent-primary">
                        <span class="text-sm text-gray-400">Require Public Trust (Let's Encrypt, etc.)</span>
                    </label>
                </div>
            </div>
            <div class="p-6 bg-[#141414] flex justify-end gap-3 border-t border-[#222]">
                <button @click="showPolicyModal = false" class="text-xs font-bold text-gray-500 uppercase px-4 py-2 hover:text-white transition-colors">Cancel</button>
                <button @click="createPolicy" class="btn btn-primary">
                    <Plus class="w-4 h-4 mr-2" /> Create Policy
                </button>
            </div>
        </div>
    </div>
  </div>
</template>

<style scoped>
.btn {
  @apply px-4 py-2 rounded font-medium transition-colors duration-200 flex items-center;
}
.btn-primary {
  @apply bg-orange-600 text-white hover:bg-orange-700;
}
.btn-secondary {
  @apply border border-[#30363d] text-gray-300 hover:text-white hover:bg-[#161b22];
}
.btn-sm {
  @apply px-3 py-1.5 text-xs;
}
</style>
