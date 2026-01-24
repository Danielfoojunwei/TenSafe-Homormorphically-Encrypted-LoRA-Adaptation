<script setup>
import { ClipboardList, Hash, RefreshCw } from 'lucide-vue-next'
import { ref, onMounted } from 'vue'

const auditLogs = ref([])
const loading = ref(false)
const errorMessage = ref('')

const fetchLogs = async () => {
    loading.value = true
    errorMessage.value = ''
    try {
        const res = await fetch('/api/v1/audit/logs?limit=50')
        if (res.ok) {
            const data = await res.json()
            auditLogs.value = data
        } else {
            throw new Error('Backend unavailable')
        }
    } catch (e) {
        console.warn("Failed to fetch audit logs", e)
        auditLogs.value = []
        errorMessage.value = 'Unable to load audit logs. Ensure the backend is reachable and you are authenticated.'
    }
    loading.value = false
}

const syncRecords = () => fetchLogs()

onMounted(fetchLogs)
</script>

<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
       <div>
         <h2 class="text-2xl font-bold">Immutable Audit Trail</h2>
         <span class="text-xs text-gray-500">Cryptographic Traceability</span>
       </div>
       <button @click="syncRecords" class="btn btn-secondary">
          <RefreshCw class="w-4 h-4 mr-2" :class="loading ? 'animate-spin' : ''" /> Sync Records
       </button>
    </div>

    <div class="bg-[#0d1117] border border-[#30363d] rounded-lg overflow-hidden">
       <div v-if="auditLogs.length === 0" class="flex flex-col items-center justify-center p-20 text-gray-500">
           <ClipboardList class="w-16 h-16 mb-4 opacity-20" />
           <p v-if="errorMessage">{{ errorMessage }}</p>
           <p v-else>No security events found</p>
       </div>
       
       <div v-else class="divide-y divide-[#30363d]">
          <div v-for="log in auditLogs" :key="log.id" class="p-4 hover:bg-[#161b22] transition-colors flex items-center justify-between group">
             <div class="flex items-center gap-4">
                <div class="p-2 bg-[#1f2428] rounded border border-[#30363d] text-gray-400">
                    <Hash class="w-5 h-5" />
                </div>
                <div>
                   <div class="flex items-center gap-2">
                      <span class="font-bold text-sm">{{ log.action }}</span>
                      <span class="bg-gray-800 text-gray-400 text-[10px] px-1.5 rounded font-mono">{{ log.time }}</span>
                   </div>
                   <div class="text-xs text-gray-500 font-mono mt-0.5 max-w-md truncate">HASH: {{ log.hash }}</div>
                </div>
             </div>
             
             <div class="text-right text-sm">
                <div class="text-gray-300">By <span class="font-bold text-white">{{ log.actor }}</span></div>
                <div class="text-xs text-gray-500">Target: {{ log.target }}</div>
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
.btn-secondary {
  @apply border border-[#30363d] text-gray-300 hover:text-white hover:bg-[#161b22];
}
</style>
