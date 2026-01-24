<script setup>
import { ref, onMounted, computed } from 'vue'
import { GitBranch, Clock, RotateCcw, CheckCircle, Archive, AlertTriangle } from 'lucide-vue-next'

const library = ref([])
const loading = ref(true)
const rollingBack = ref(null)

const fetchData = async () => {
    loading.value = true
    try {
        const res = await fetch('/api/v1/skills/library')
        library.value = await res.json()
    } catch (e) {
        console.error("Failed to load skills library", e)
    }
    loading.value = false
}

const getStatusColor = (status) => {
    const colors = {
        'deployed': 'text-green-500 bg-green-500/10',
        'validated': 'text-blue-500 bg-blue-500/10',
        'adapting': 'text-yellow-500 bg-yellow-500/10',
        'archived': 'text-gray-500 bg-gray-500/10'
    }
    return colors[status] || 'text-gray-500'
}

const rollback = async (expertId, version, name) => {
    if (!confirm(`Rollback skill "${name}" to version ${version}? This effectively undeploys the current version.`)) return
    
    rollingBack.value = expertId
    try {
        await fetch('/api/v1/skills/rollback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                expert_id: expertId, 
                target_version: version, 
                reason: 'Frontend manual rollback' 
            })
        })
        await fetchData()
    } catch (e) {
        console.error("Rollback failed", e)
    }
    rollingBack.value = null
}

onMounted(fetchData)
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
       <div>
         <h2 class="text-lg font-bold">Skills Library & Version Control</h2>
         <span class="text-xs text-gray-500">Tier 2: FedMoE Expert Aggregation (EDA)</span>
       </div>
       <button @click="fetchData" class="btn btn-secondary btn-sm">Refresh</button>
    </div>

    <!-- Loading -->
    <div v-if="loading" class="flex justify-center py-12">
        <div class="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
    </div>

    <!-- Skills List -->
    <div v-else class="space-y-6">
        <div v-for="skill in library" :key="skill.name" class="bg-[#111] border border-[#333] rounded-lg overflow-hidden">
            <!-- Skill Header -->
            <div class="px-6 py-4 border-b border-[#333] flex items-center justify-between bg-[#161b22]">
                <div class="flex items-center gap-3">
                    <GitBranch class="w-5 h-5 text-purple-400" />
                    <div>
                        <div class="font-bold text-white">{{ skill.name }}</div>
                        <div class="text-xs text-gray-500 font-mono">{{ skill.base_model }}</div>
                    </div>
                </div>
                <div class="flex items-center gap-2">
                    <span class="text-xs text-gray-500">Active Version:</span>
                    <span class="text-xs font-mono font-bold text-primary">{{ skill.active_version || 'NONE' }}</span>
                </div>
            </div>

            <!-- Version History -->
            <div class="p-4">
                <table class="w-full">
                    <thead>
                        <tr class="text-left text-xs text-gray-500 uppercase">
                            <th class="pb-2">Version</th>
                            <th class="pb-2">Status</th>
                            <th class="pb-2">Created</th>
                            <th class="pb-2">Accuracy</th>
                            <th class="pb-2">Evidence</th>
                            <th class="pb-2 text-right">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="ver in skill.versions" :key="ver.id" class="border-b border-[#222] last:border-0 hover:bg-[#1a1f24] transition-colors">
                            <td class="py-3 font-mono text-sm text-white">v{{ ver.version }}</td>
                            <td class="py-3">
                                <span :class="['text-[10px] font-bold px-2 py-1 rounded-full uppercase', getStatusColor(ver.status)]">
                                    {{ ver.status }}
                                </span>
                            </td>
                            <td class="py-3 text-xs text-gray-400">{{ ver.created_at.split('T')[0] }}</td>
                            <td class="py-3 text-xs text-white">{{ (ver.accuracy * 100).toFixed(1) }}%</td>
                            <td class="py-3 text-xs text-gray-400">{{ ver.evidence_count }} proofs</td>
                            <td class="py-3 text-right">
                                <button 
                                    v-if="ver.status !== 'deployed'"
                                    @click="rollback(ver.id, ver.version, skill.name)"
                                    :disabled="rollingBack === ver.id"
                                    class="text-xs text-gray-400 hover:text-white border border-[#333] rounded px-2 py-1 flex items-center gap-1 ml-auto transition-colors"
                                >
                                    <RotateCcw class="w-3 h-3" />
                                    {{ rollingBack === ver.id ? '...' : 'Rollback' }}
                                </button>
                                <span v-else class="text-xs text-green-500 font-bold flex items-center justify-end gap-1">
                                    <CheckCircle class="w-3 h-3" /> Active
                                </span>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
  </div>
</template>

<style scoped>
.btn {
  @apply px-3 py-1.5 rounded font-medium transition-colors duration-200;
}
.btn-secondary {
  @apply border border-[#30363d] text-gray-300 hover:text-white hover:bg-[#161b22];
}
.btn-sm {
  @apply text-xs;
}
</style>
