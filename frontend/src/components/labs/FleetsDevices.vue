<script setup>
import { Server, Plus, RefreshCw, Loader2 } from 'lucide-vue-next'
import { ref, onMounted } from 'vue'

const fleets = ref([])
const loading = ref(true)
const enrolling = ref(false)

const fetchFleets = async () => {
    loading.value = true
    try {
        const res = await fetch('/api/v1/fleets/extended')
        if (res.ok) {
            fleets.value = await res.json()
        } else {
            throw new Error('Backend not available')
        }
    } catch (e) {
        console.warn("Failed to fetch fleets", e)
        fleets.value = []
    }
    loading.value = false
}

const enrollDevice = async () => {
    enrolling.value = true
    const fleetName = prompt("Enter fleet name:")
    if (fleetName) {
        try {
            const res = await fetch(`/api/v1/fleets?name=${encodeURIComponent(fleetName)}`, {
                method: 'POST'
            })
            if (res.ok) {
                const data = await res.json()
                alert(`Fleet created! API Key (save this): ${data.api_key}`)
                await fetchFleets()
            }
        } catch (e) {
            console.warn("Failed to enroll", e)
            alert("Failed to create fleet")
        }
    }
    enrolling.value = false
}

const getDeviceCount = (fleetId) => {
    const fleet = fleets.value.find(f => f.id === fleetId)
    return fleet ? fleet.devices_total : 0
}

const getOnlineCount = (fleetId) => {
    const fleet = fleets.value.find(f => f.id === fleetId)
    return fleet ? fleet.devices_online : 0
}

onMounted(fetchFleets)
</script>

<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
       <div>
         <h2 class="text-2xl font-bold">Fleets & Devices</h2>
         <span class="text-xs text-gray-500">Edge Node Orchestration</span>
       </div>
       <div class="flex gap-2">
          <button @click="fetchFleets" :disabled="loading" class="btn btn-secondary">
             <RefreshCw class="w-4 h-4" :class="loading ? 'animate-spin' : ''" />
          </button>
          <button @click="enrollDevice" :disabled="enrolling" class="btn btn-primary">
             <Loader2 v-if="enrolling" class="w-4 h-4 mr-2 animate-spin" />
             <Plus v-else class="w-4 h-4 mr-2" /> {{ enrolling ? 'Creating...' : 'Enroll New Fleet' }}
          </button>
       </div>
    </div>

    <div class="grid grid-cols-1 gap-4">
       <div v-for="fleet in fleets" :key="fleet.id" class="bg-[#0d1117] border border-[#30363d] rounded-lg p-6 relative overflow-hidden group hover:border-[#58a6ff] transition-colors">
          <div class="flex items-start justify-between mb-6">
             <div class="flex items-center gap-4">
                <div class="w-12 h-12 rounded bg-[#1f2428] border border-[#30363d] flex items-center justify-center">
                    <Server class="w-6 h-6 text-gray-400" />
                </div>
                <div>
                   <h3 class="font-bold text-lg">{{ fleet.name }}</h3>
                   <div class="flex items-center gap-2 text-xs text-gray-500">
                      <span>{{ fleet.region }}</span>
                      <span>•</span>
                      <span :class="fleet.status === 'Healthy' ? 'text-green-500' : 'text-yellow-500'">{{ fleet.status }}</span>
                   </div>
                </div>
             </div>
             
             <div class="text-right">
                <div class="text-2xl font-bold">{{ fleet.trust }}%</div>
                <div class="text-xs text-gray-500">Trust Score</div>
             </div>
          </div>
          
          <div class="grid grid-cols-2 gap-4">
             <div class="bg-[#161b22] p-3 rounded border border-[#30363d] flex items-center justify-between">
                 <span class="text-sm text-gray-400">Total Devices</span>
                 <span class="font-mono font-bold">{{ getDeviceCount(fleet.id) }}</span>
             </div>
             <div class="bg-[#161b22] p-3 rounded border border-[#30363d] flex items-center justify-between">
                 <span class="text-sm text-gray-400">Online</span>
                 <span class="font-mono font-bold text-green-500">{{ getOnlineCount(fleet.id) }}</span>
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
</style>
