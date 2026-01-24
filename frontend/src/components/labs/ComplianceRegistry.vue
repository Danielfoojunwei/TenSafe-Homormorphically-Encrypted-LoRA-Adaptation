<script setup>
import { ShieldCheck, FileCheck, CheckCircle, AlertTriangle, Play } from 'lucide-vue-next'
import { ref } from 'vue'

const profiles = [
  { id: 'tgsp-1', name: 'finance-v2-hardened', version: '2.1.0', status: 'verified', checksum: 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855' },
  { id: 'tgsp-2', name: 'health-hipaa-compliant', version: '1.0.5', status: 'verified', checksum: 'sha256:8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4' },
  { id: 'tgsp-3', name: 'dev-experimental', version: '0.9.0-beta', status: 'warning', checksum: 'sha256:ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb' },
]

const verifying = ref(false)
const qaResult = ref(null)

const runFullQA = async () => {
    verifying.value = true
    try {
        const res = await fetch('/api/v1/forensics/verify-compliance', { method: 'POST' })
        qaResult.value = await res.json()
    } catch (e) {
        console.error("QA failed", e)
    }
    verifying.value = false
}
</script>

<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
       <div>
         <h2 class="text-2xl font-bold">Compliance Registry</h2>
         <span class="text-xs text-gray-500">TGSP Profiles & Provenance</span>
       </div>
       <button @click="runFullQA" :disabled="verifying" class="btn btn-primary flex items-center gap-2">
           <Play v-if="!verifying" class="w-4 h-4" />
           <div v-else class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
           {{ verifying ? 'Running System QA...' : 'Run Full QA Check' }}
       </button>
    </div>

    <!-- QA Result Banner -->
    <div v-if="qaResult" :class="['p-4 rounded border flex items-center justify-between animate-fade-in', qaResult.status === 'COMPLIANT' ? 'bg-green-900/20 border-green-900' : 'bg-red-900/20 border-red-900']">
        <div>
            <div :class="['font-bold text-lg', qaResult.status === 'COMPLIANT' ? 'text-green-500' : 'text-red-500']">
                System Status: {{ qaResult.status }}
            </div>
            <div class="text-xs text-gray-400">Score: {{ qaResult.compliance_score.toFixed(1) }}% | Auditor: {{ qaResult.auditor }}</div>
        </div>
        <div class="text-right text-xs text-gray-500">
            {{ qaResult.timestamp }}
        </div>
    </div>

    <!-- Stats -->
    <div class="grid grid-cols-3 gap-4">
       <div class="bg-[#161b22] border border-[#30363d] p-4 rounded flex items-center justify-between">
          <div>
            <div class="text-xs text-gray-400">Total Profiles</div>
            <div class="text-2xl font-bold">12</div>
          </div>
          <ShieldCheck class="w-8 h-8 text-green-500 opacity-20" />
       </div>
       <div class="bg-[#161b22] border border-[#30363d] p-4 rounded flex items-center justify-between">
          <div>
            <div class="text-xs text-gray-400">Verified</div>
            <div class="text-2xl font-bold text-green-500">11</div>
          </div>
          <CheckCircle class="w-8 h-8 text-green-500 opacity-20" />
       </div>
       <div class="bg-[#161b22] border border-[#30363d] p-4 rounded flex items-center justify-between">
          <div>
            <div class="text-xs text-gray-400">Warnings</div>
            <div class="text-2xl font-bold text-yellow-500">1</div>
          </div>
          <AlertTriangle class="w-8 h-8 text-yellow-500 opacity-20" />
       </div>
    </div>

    <!-- Table -->
    <div class="bg-[#0d1117] border border-[#30363d] rounded-lg overflow-hidden">
       <table class="w-full text-left text-sm">
          <thead class="bg-[#161b22] text-gray-400 border-b border-[#30363d]">
             <tr>
               <th class="p-4">Profile Name</th>
               <th class="p-4">Version</th>
               <th class="p-4">Checksum (SHA256)</th>
               <th class="p-4">Status</th>
             </tr>
          </thead>
          <tbody>
             <tr v-for="p in profiles" :key="p.id" class="border-b border-[#30363d]/50 hover:bg-[#161b22/50]">
                <td class="p-4 font-medium flex items-center gap-2">
                   <FileCheck class="w-4 h-4 text-gray-500" />
                   {{ p.name }}
                </td>
                <td class="p-4 font-mono text-gray-400">{{ p.version }}</td>
                <td class="p-4 font-mono text-xs text-gray-500">{{ p.checksum.substring(0, 12) }}...</td>
                <td class="p-4">
                   <span class="px-2 py-0.5 rounded textxs font-bold border" 
                         :class="p.status === 'verified' ? 'bg-green-900/20 text-green-400 border-green-900' : 'bg-yellow-900/20 text-yellow-400 border-yellow-900'">
                      {{ p.status.toUpperCase() }}
                   </span>
                </td>
             </tr>
          </tbody>
       </table>
    </div>
  </div>
</template>

<style scoped>
.btn {
  @apply px-4 py-2 rounded font-medium transition-colors duration-200 inline-flex;
}
.btn-primary {
  @apply bg-orange-600 text-white hover:bg-orange-700;
}
.animate-fade-in {
  animation: fadeIn 0.3s ease-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
