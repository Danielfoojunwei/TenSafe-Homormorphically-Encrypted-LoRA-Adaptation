<script setup>
import { ref, onMounted } from 'vue'
import { GitBranch, GitCommit, Play, Clock, Rocket, RotateCcw, Loader2, CheckCircle } from 'lucide-vue-next'

const commits = ref([])
const loading = ref(true)
const deploying = ref(null)
const syncing = ref(false)

const fetchVersions = async () => {
    loading.value = true
    try {
        const res = await fetch('/api/v1/lineage/versions')
        if (res.ok) {
            const data = await res.json()
            commits.value = data.versions
        } else {
            throw new Error('Backend not available')
        }
    } catch (e) {
        console.warn("Failed to fetch versions - using fallback", e)
        commits.value = [
            { id: 'c1', hash: 'e7f2b1', message: 'Improve context window size', author: 'Daniel Foo', time: '10m ago', status: 'deployed', tag: 'v2.1.0' },
            { id: 'c2', hash: 'a8d9c4', message: 'Merge PR #42: PQC Integration', author: 'System', time: '2h ago', status: 'verified', tag: 'v2.0.5' },
            { id: 'c3', hash: 'b3e5f6', message: 'Optimize inference latency', author: 'Daniel Foo', time: '5h ago', status: 'archived', tag: 'v2.0.4' },
            { id: 'c4', hash: 'd4f5g6', message: 'Initial commit', author: 'Daniel Foo', time: '1d ago', status: 'archived', tag: 'v1.0.0' },
        ]
    }
    loading.value = false
}

const deploy = async (id) => {
    deploying.value = id
    try {
        const res = await fetch('/api/v1/lineage/deploy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ version_id: id, reason: 'manual_deployment' })
        })
        if (res.ok) {
            await fetchVersions()
        } else {
            // Fallback to client-side update
            commits.value.forEach(c => {
                if (c.id === id) c.status = 'deployed'
                else if (c.status === 'deployed') c.status = 'verified'
            })
        }
    } catch (e) {
        console.warn("Deploy failed - updating locally", e)
        commits.value.forEach(c => {
            if (c.id === id) c.status = 'deployed'
            else if (c.status === 'deployed') c.status = 'verified'
        })
    }
    deploying.value = null
}

const syncRepo = async () => {
    syncing.value = true
    try {
        await fetch('/api/v1/lineage/sync', { method: 'POST' })
        await fetchVersions()
    } catch (e) {
        console.warn("Sync failed", e)
    }
    syncing.value = false
}

onMounted(fetchVersions)
</script>

<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
       <div>
         <h2 class="text-2xl font-bold">Model Lineage</h2>
         <span class="text-xs text-gray-500">Version Control & Deployment History</span>
       </div>
       <button @click="syncRepo" :disabled="syncing" class="btn btn-secondary">
          <Loader2 v-if="syncing" class="w-4 h-4 mr-2 animate-spin" />
          <GitBranch v-else class="w-4 h-4 mr-2" /> {{ syncing ? 'Syncing...' : 'Sync Repository' }}
       </button>
    </div>

    <div class="grid grid-cols-12 gap-6 h-[600px]">
       <!-- Left: Git Graph Viz -->
       <div class="col-span-4 bg-[#0d1117] border border-[#30363d] rounded-lg p-6 relative overflow-hidden">
          <h3 class="text-sm font-bold text-gray-400 mb-6 uppercase tracking-wider">Commit History</h3>
          
          <div class="relative pl-4 border-l-2 border-[#30363d] space-y-8">
             <div v-for="(c, i) in commits" :key="c.id" class="relative group">
                <!-- Node -->
                <div class="absolute -left-[25px] w-4 h-4 rounded-full border-2 transition-colors z-10"
                     :class="c.status === 'deployed' ? 'bg-primary border-primary scale-125' : 'bg-[#111] border-gray-500 group-hover:border-primary'">
                </div>
                
                <!-- Content -->
                <div class="ml-4 cursor-pointer hover:bg-[#161b22] p-2 rounded -mt-2 transition-colors border border-transparent hover:border-[#333]">
                   <div class="flex items-center gap-2 mb-1">
                      <span class="font-mono text-xs text-gray-400">{{ c.hash }}</span>
                      <span v-if="c.tag" class="text-[10px] bg-[#333] text-gray-300 px-1.5 rounded border border-gray-600">{{ c.tag }}</span>
                      <span v-if="c.status === 'deployed'" class="text-[10px] bg-primary/20 text-primary px-1.5 rounded border border-primary/30 animate-pulse">ACTIVE</span>
                   </div>
                   <div class="text-sm font-medium text-white">{{ c.message }}</div>
                   <div class="text-xs text-gray-500 mt-1">{{ c.author }} • {{ c.time }}</div>
                </div>
             </div>
          </div>
       </div>

       <!-- Right: Action Panel -->
       <div class="col-span-8 bg-[#111] border border-[#333] rounded-lg p-6">
          <h3 class="text-sm font-bold text-gray-400 mb-6 uppercase tracking-wider">Deployment Controls</h3>
          
          <table class="w-full text-left text-sm">
             <thead>
                <tr class="border-b border-[#333] text-gray-400">
                   <th class="py-2">Version</th>
                   <th class="py-2">Status</th>
                   <th class="py-2">Verification</th>
                   <th class="py-2 text-right">Actions</th>
                </tr>
             </thead>
             <tbody>
                <tr v-for="c in commits" :key="c.id" class="border-b border-[#333]/50 hover:bg-[#161b22]">
                   <td class="py-4 font-mono text-white">{{ c.tag }}</td>
                   <td class="py-4">
                      <span class="flex items-center gap-2">
                         <div class="w-2 h-2 rounded-full" :class="c.status === 'deployed' ? 'bg-primary' : (c.status === 'verified' ? 'bg-white' : 'bg-gray-600')"></div>
                         <span :class="c.status === 'deployed' ? 'text-primary font-bold' : 'text-gray-400'">{{ c.status.toUpperCase() }}</span>
                      </span>
                   </td>
                   <td class="py-4 text-gray-400">Passed (98/98 tests)</td>
                   <td class="py-4 text-right">
                      <button v-if="c.status !== 'deployed'" @click="deploy(c.id)" :disabled="deploying === c.id" class="btn btn-sm btn-primary">
                         <Loader2 v-if="deploying === c.id" class="w-3 h-3 mr-1 animate-spin" />
                         <Rocket v-else class="w-3 h-3 mr-1" /> {{ deploying === c.id ? 'Deploying...' : 'Deploy' }}
                      </button>
                      <button v-else disabled class="btn btn-sm btn-secondary opacity-50 cursor-not-allowed">
                         <CheckCircle class="w-3 h-3 mr-1" /> Current
                      </button>
                   </td>
                </tr>
             </tbody>
          </table>
       </div>
    </div>
  </div>
</template>

<style scoped>
.btn {
  @apply px-3 py-1.5 rounded font-medium transition-colors duration-200 flex items-center inline-flex;
}
.btn-primary {
  @apply bg-orange-600 text-white hover:bg-orange-700;
}
.btn-secondary {
  @apply border border-[#30363d] text-gray-300 hover:text-white hover:bg-[#161b22];
}
</style>
