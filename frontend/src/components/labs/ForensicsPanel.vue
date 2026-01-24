<script setup>
import { ref, onMounted } from 'vue'
import { AlertCircle, Search, ShieldAlert, Activity, FileText } from 'lucide-vue-next'

const incidents = ref([])
const compliance = ref(null)
const loading = ref(false)
const verifying = ref(false)

const fetchIncidents = async () => {
    loading.value = true
    try {
        const res = await fetch('/api/v1/forensics/incidents')
        incidents.value = await res.json()
    } catch (e) {
        console.error("Failed to load incidents", e)
    }
    loading.value = false
}

const runCompliance = async () => {
    verifying.value = true
    try {
        const res = await fetch('/api/v1/forensics/verify-compliance', { method: 'POST' })
        compliance.value = await res.json()
    } catch (e) {
        console.error("Validation failed", e)
    }
    verifying.value = false
}

const analysisResult = ref(null)
const analyzing = ref(false)

const analyzeIncident = async (id) => {
    analyzing.value = true
    analysisResult.value = null
    try {
        const res = await fetch('/api/v1/forensics/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ incident_id: id, time_window_hours: 24 })
        })
        if (res.ok) {
            analysisResult.value = await res.json()
        } else {
            alert(`Failed to analyze incident ${id}`)
        }
    } catch (e) {
        console.error("Analysis failed", e)
        alert(`Analysis error for incident ${id}`)
    }
    analyzing.value = false
}

onMounted(fetchIncidents)
</script>

<template>
  <div class="grid grid-cols-3 gap-6 h-full">
    <!-- Forensics & RCA -->
    <div class="col-span-2 space-y-6">
        <div class="flex items-center justify-between">
            <div>
                 <h2 class="text-lg font-bold flex items-center gap-2">
                     <Search class="w-5 h-5 text-primary" />
                     Forensics & Root Cause Analysis
                 </h2>
                 <span class="text-xs text-gray-500">Tier 4: Post-Incident Investigation</span>
            </div>
            <button @click="fetchIncidents" class="btn btn-secondary btn-sm">Refresh Streams</button>
        </div>

        <div v-if="loading" class="text-center py-10 text-gray-500">Loading incidents...</div>
        
        <div v-else class="space-y-4">
            <div v-for="inc in incidents" :key="inc.id" class="bg-[#111] border border-[#333] rounded-lg p-4 hover:border-primary/50 transition-colors cursor-pointer" @click="analyzeIncident(inc.id)">
                <div class="flex justify-between items-start mb-2">
                    <div class="flex items-center gap-2">
                        <AlertCircle v-if="inc.severity === 'HIGH'" class="w-4 h-4 text-red-500" />
                        <ShieldAlert v-else class="w-4 h-4 text-yellow-500" />
                        <span class="font-bold text-white text-sm">{{ inc.type }}</span>
                    </div>
                    <span class="text-xs font-mono text-gray-500">{{ inc.id }}</span>
                </div>
                <p class="text-sm text-gray-300 mb-3">{{ inc.description }}</p>
                <div class="flex items-center justify-between text-xs">
                    <span class="text-gray-500">{{ inc.timestamp }}</span>
                    <span :class="['px-2 py-0.5 rounded uppercase font-bold', inc.status === 'OPEN' ? 'bg-red-500/20 text-red-500' : 'bg-green-500/20 text-green-500']">
                        {{ inc.status }}
                    </span>
                </div>
            </div>
        </div>
    </div>

    <!-- On-Demand Compliance -->
    <div class="bg-[#0d1117] border-l border-[#333] pl-6 -my-6 py-6 flex flex-col">
        <h2 class="text-lg font-bold mb-1 flex items-center gap-2">
            <FileText class="w-5 h-5 text-green-500" />
            CISO Compliance
        </h2>
        <span class="text-xs text-gray-500 mb-6">On-Demand Verification</span>

        <div class="flex-1 overflow-y-auto">
            <div v-if="!compliance" class="flex flex-col items-center justify-center h-48 text-center text-gray-500">
                <ShieldAlert class="w-12 h-12 mb-3 opacity-20" />
                <p class="text-sm">No verification run in this session.</p>
                <button @click="runCompliance" :disabled="verifying" class="btn btn-primary mt-4 w-full justify-center">
                    {{ verifying ? 'Verifying...' : 'Run Full QA Check' }}
                </button>
            </div>

            <div v-else class="space-y-6 animate-fade-in">
                <div class="text-center p-6 bg-[#161b22] rounded-lg border border-[#333]">
                    <div class="text-4xl font-bold text-white mb-1">{{ compliance.compliance_score.toFixed(0) }}%</div>
                    <div :class="['text-xs font-bold uppercase tracking-wider', compliance.status === 'COMPLIANT' ? 'text-green-500' : 'text-red-500']">
                        {{ compliance.status }}
                    </div>
                    <span class="text-[10px] text-gray-500 mt-2 block">Auditor: {{ compliance.auditor }}</span>
                </div>

                <div class="space-y-2">
                    <div v-for="check in compliance.checks" :key="check.control" class="flex items-start gap-3 text-sm p-2 rounded hover:bg-[#161b22]">
                        <div :class="['w-2 h-2 mt-1.5 rounded-full flex-shrink-0', check.status === 'PASS' ? 'bg-green-500' : 'bg-yellow-500']"></div>
                        <div>
                            <div class="font-bold text-gray-300">{{ check.control }} {{ check.name }}</div>
                            <div class="text-xs text-gray-500">{{ check.details }}</div>
                        </div>
                    </div>
                </div>

                <button @click="runCompliance" :disabled="verifying" class="btn btn-secondary w-full justify-center text-xs">
                    {{ verifying ? 'Verifying...' : 'Re-Run Verification' }}
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
  @apply text-xs py-1;
}
.animate-fade-in {
  animation: fadeIn 0.3s ease-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
