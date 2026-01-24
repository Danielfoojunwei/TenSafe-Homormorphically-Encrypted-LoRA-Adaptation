<script setup>
import { Shield, RefreshCw, Lock, Key as KeyIcon, Clock, AlertTriangle, CheckCircle } from 'lucide-vue-next'
import { ref, onMounted, computed } from 'vue'

const keys = ref([])
const schedule = ref([])
const attestationPolicies = ref({ current_level: 4, levels: [] })
const loading = ref(true)
const rotating = ref(null)

const fetchData = async () => {
    loading.value = true
    try {
        const [keysRes, scheduleRes, policiesRes] = await Promise.all([
            fetch('/api/v1/kms/keys'),
            fetch('/api/v1/kms/rotation-schedule'),
            fetch('/api/v1/kms/attestation-policies')
        ])
        const keysData = await keysRes.json()
        const scheduleData = await scheduleRes.json()
        const policiesData = await policiesRes.json()
        
        keys.value = keysData.keys || []
        schedule.value = scheduleData.schedule || []
        attestationPolicies.value = policiesData
    } catch (e) {
        console.error("Failed to load KMS data", e)
    }
    loading.value = false
}

const rotateKey = async (kid) => {
    if (!confirm(`Rotate key "${kid}"? This will create an immutable audit log.`)) return
    rotating.value = kid
    try {
        const res = await fetch('/api/v1/kms/rotate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ kid, reason: 'manual_rotation' })
        })
        if (res.ok) {
            await fetchData()
        }
    } catch (e) {
        console.error("Failed to rotate key", e)
    }
    rotating.value = null
}

const getStatusColor = (days) => {
    if (days <= 7) return 'text-red-500'
    if (days <= 30) return 'text-yellow-500'
    return 'text-green-500'
}

const currentLevel = computed(() => {
    return attestationPolicies.value.levels.find(l => l.level === attestationPolicies.value.current_level)
})

onMounted(fetchData)
</script>

<template>
  <div class="space-y-8">
    <!-- Header Controls -->
    <div class="flex items-center justify-between border-b border-[#333] pb-6">
       <div>
         <h2 class="text-2xl font-bold">Enterprise KMS Manager</h2>
         <span class="text-xs text-gray-500">Key Lifecycle, Attestation & Fleet Policy</span>
       </div>
       <div class="flex gap-4">
           <button @click="fetchData" class="btn btn-secondary text-sm font-bold uppercase tracking-wider flex items-center gap-2">
              <RefreshCw class="w-4 h-4" /> Refresh
           </button>
       </div>
    </div>

    <!-- Key Rotation Schedule -->
    <div>
        <h3 class="text-xs font-bold text-gray-500 uppercase mb-4">Key Rotation Schedule</h3>
        <div class="bg-[#111] border border-[#333] rounded-lg overflow-hidden">
            <table class="w-full">
                <thead class="bg-[#0a0a0a] border-b border-[#333]">
                    <tr>
                        <th class="text-left text-xs font-bold text-gray-500 uppercase px-6 py-3">Key ID</th>
                        <th class="text-left text-xs font-bold text-gray-500 uppercase px-6 py-3">Algorithm</th>
                        <th class="text-left text-xs font-bold text-gray-500 uppercase px-6 py-3">Next Rotation</th>
                        <th class="text-left text-xs font-bold text-gray-500 uppercase px-6 py-3">Days Remaining</th>
                        <th class="text-left text-xs font-bold text-gray-500 uppercase px-6 py-3">Actions</th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="key in schedule" :key="key.kid" class="border-b border-[#222] hover:bg-[#161616]">
                        <td class="px-6 py-4 font-mono text-sm text-white">{{ key.kid }}</td>
                        <td class="px-6 py-4 text-sm text-gray-400">{{ key.algorithm }}</td>
                        <td class="px-6 py-4 text-sm text-gray-400">{{ key.next_rotation?.split('T')[0] }}</td>
                        <td class="px-6 py-4">
                            <span :class="['font-bold text-sm', getStatusColor(key.days_remaining)]">
                                {{ key.days_remaining }} days
                            </span>
                        </td>
                        <td class="px-6 py-4">
                            <button 
                                @click="rotateKey(key.kid)" 
                                :disabled="rotating === key.kid"
                                class="btn btn-sm btn-primary flex items-center gap-2"
                            >
                                <RefreshCw v-if="rotating !== key.kid" class="w-3 h-3" />
                                <div v-else class="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                Rotate Now
                            </button>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <!-- Attestation Policy -->
    <div>
        <h3 class="text-xs font-bold text-gray-500 uppercase mb-4">TEE Attestation Policy</h3>
        <div class="grid grid-cols-4 gap-4">
            <div 
                v-for="level in attestationPolicies.levels" 
                :key="level.level"
                :class="[
                    'bg-[#111] border rounded-lg p-6 transition-all cursor-pointer',
                    level.level === attestationPolicies.current_level 
                        ? 'border-primary bg-primary/5' 
                        : 'border-[#333] hover:border-[#444]'
                ]"
            >
                <div class="flex items-center gap-2 mb-2">
                    <div :class="[
                        'w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold',
                        level.level === attestationPolicies.current_level 
                            ? 'bg-primary text-white' 
                            : 'bg-[#222] text-gray-400'
                    ]">
                        {{ level.level }}
                    </div>
                    <span class="font-bold text-white text-sm">{{ level.name }}</span>
                </div>
                <p class="text-[10px] text-gray-500">{{ level.description }}</p>
                <div v-if="level.level === attestationPolicies.current_level" class="mt-3">
                    <span class="text-[10px] text-primary font-bold uppercase bg-primary/10 px-2 py-1 rounded">Active</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Key Inventory -->
    <div>
        <h3 class="text-xs font-bold text-gray-500 uppercase mb-4">Key Inventory</h3>
        <div class="grid grid-cols-3 gap-6">
            <div v-for="key in keys" :key="key.kid" class="bg-[#111] border border-[#333] rounded-lg p-6 hover:border-primary/50 transition-colors">
                <div class="flex items-center justify-between mb-4">
                    <div class="flex items-center gap-2">
                        <KeyIcon class="w-5 h-5 text-primary" />
                        <span class="font-mono font-bold text-white">{{ key.kid }}</span>
                    </div>
                    <span :class="[
                        'text-[10px] font-bold uppercase px-2 py-1 rounded',
                        key.status === 'active' ? 'bg-green-500/10 text-green-500' : 'bg-yellow-500/10 text-yellow-500'
                    ]">
                        {{ key.status }}
                    </span>
                </div>
                <div class="space-y-2 text-sm">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Region</span>
                        <span class="text-white">{{ key.region }}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Algorithm</span>
                        <span class="text-white font-mono text-xs">{{ key.algorithm }}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Created</span>
                        <span class="text-white">{{ key.created_at?.split('T')[0] }}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Expires</span>
                        <span :class="getStatusColor(key.days_remaining)">{{ key.days_remaining }} days</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
  </div>
</template>

<style scoped>
.btn {
  @apply px-4 py-2 rounded font-medium transition-colors duration-200 flex items-center inline-flex;
}
.btn-primary {
  @apply bg-orange-600 text-white hover:bg-orange-700;
}
.btn-secondary {
  @apply border border-[#30363d] text-gray-300 hover:text-white hover:bg-[#161b22];
}
.btn-sm {
  @apply px-3 py-1 text-xs;
}
</style>
