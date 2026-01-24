<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import {
    Activity, Play, Pause, Square, RefreshCw,
    TrendingUp, TrendingDown, Cpu, Clock, Zap,
    Users, Shield, Database, AlertTriangle, CheckCircle
} from 'lucide-vue-next'

const isRunning = ref(false)
const currentRound = ref(0)
const totalRounds = ref(100)
const activeClients = ref(0)
const aggregatedUpdates = ref(0)
const privacyBudget = ref({ epsilon: 0, delta: 1e-5 })
const metrics = ref({
    loss: [],
    accuracy: [],
    bandwidth: [],
    latency: []
})
const realtimeEvents = ref([])
const expertWeights = ref({})
const healthStatus = ref('healthy')

let pollingInterval = null
let eventSource = null

const progress = computed(() => {
    return totalRounds.value > 0 ? (currentRound.value / totalRounds.value) * 100 : 0
})

const avgLoss = computed(() => {
    if (metrics.value.loss.length === 0) return 0
    return (metrics.value.loss.reduce((a, b) => a + b, 0) / metrics.value.loss.length).toFixed(4)
})

const avgAccuracy = computed(() => {
    if (metrics.value.accuracy.length === 0) return 0
    return (metrics.value.accuracy.reduce((a, b) => a + b, 0) / metrics.value.accuracy.length * 100).toFixed(1)
})

const startTraining = async () => {
    isRunning.value = true
    realtimeEvents.value.unshift({
        time: new Date().toLocaleTimeString(),
        type: 'info',
        message: 'Training session started'
    })

    // Start polling for metrics
    pollingInterval = setInterval(async () => {
        await fetchMetrics()
    }, 2000)
}

const stopTraining = () => {
    isRunning.value = false
    if (pollingInterval) {
        clearInterval(pollingInterval)
        pollingInterval = null
    }
    realtimeEvents.value.unshift({
        time: new Date().toLocaleTimeString(),
        type: 'warning',
        message: 'Training session stopped'
    })
}

const fetchMetrics = async () => {
    try {
        // Fetch aggregation status
        const statusRes = await fetch('/api/v1/status')
        if (statusRes.ok) {
            const data = await statusRes.json()
            currentRound.value = data.round || currentRound.value + 1
            activeClients.value = data.active_clients || Math.floor(Math.random() * 50) + 10
        }

        // Simulate metrics update (in production, this would come from backend)
        const newLoss = Math.max(0.01, (metrics.value.loss[metrics.value.loss.length - 1] || 0.5) - Math.random() * 0.02)
        const newAcc = Math.min(0.99, (metrics.value.accuracy[metrics.value.accuracy.length - 1] || 0.5) + Math.random() * 0.01)
        const newBw = Math.random() * 0.1 + 0.05
        const newLatency = Math.random() * 20 + 40

        metrics.value.loss.push(newLoss)
        metrics.value.accuracy.push(newAcc)
        metrics.value.bandwidth.push(newBw)
        metrics.value.latency.push(newLatency)

        // Keep only last 50 points
        if (metrics.value.loss.length > 50) {
            metrics.value.loss.shift()
            metrics.value.accuracy.shift()
            metrics.value.bandwidth.shift()
            metrics.value.latency.shift()
        }

        // Update privacy budget
        privacyBudget.value.epsilon = Math.min(10, privacyBudget.value.epsilon + 0.015)

        // Update expert weights
        expertWeights.value = {
            'visual_primary': 0.35 + Math.random() * 0.05,
            'language_semantic': 0.25 + Math.random() * 0.03,
            'manipulation_grasp': 0.20 + Math.random() * 0.04,
            'navigation_base': 0.20 + Math.random() * 0.03
        }

        aggregatedUpdates.value++

        // Add event periodically
        if (currentRound.value % 5 === 0) {
            realtimeEvents.value.unshift({
                time: new Date().toLocaleTimeString(),
                type: 'success',
                message: `Round ${currentRound.value} completed. ${activeClients.value} clients aggregated.`
            })
            if (realtimeEvents.value.length > 20) {
                realtimeEvents.value.pop()
            }
        }

    } catch (e) {
        console.error("Failed to fetch metrics", e)
    }
}

const getHealthColor = (status) => {
    if (status === 'healthy') return 'text-green-500'
    if (status === 'degraded') return 'text-yellow-500'
    return 'text-red-500'
}

const sparklinePoints = (data, height = 40) => {
    if (data.length < 2) return ''
    const max = Math.max(...data)
    const min = Math.min(...data)
    const range = max - min || 1
    const width = 200
    const step = width / (data.length - 1)

    return data.map((v, i) => {
        const x = i * step
        const y = height - ((v - min) / range) * height
        return `${x},${y}`
    }).join(' ')
}

onMounted(() => {
    // Initialize with some data points
    for (let i = 0; i < 10; i++) {
        metrics.value.loss.push(0.5 - i * 0.02 + Math.random() * 0.05)
        metrics.value.accuracy.push(0.5 + i * 0.03 + Math.random() * 0.02)
        metrics.value.bandwidth.push(Math.random() * 0.1 + 0.05)
        metrics.value.latency.push(Math.random() * 20 + 40)
    }
})

onUnmounted(() => {
    if (pollingInterval) clearInterval(pollingInterval)
    if (eventSource) eventSource.close()
})
</script>

<template>
  <div class="space-y-6 h-full flex flex-col">
    <!-- Header -->
    <div class="flex items-center justify-between border-b border-[#333] pb-6">
       <div>
         <h2 class="text-2xl font-bold flex items-center gap-3">
             <Activity class="w-7 h-7 text-green-500" :class="isRunning ? 'animate-pulse' : ''" />
             Real-time Training Monitor
         </h2>
         <span class="text-xs text-gray-500">Federated Learning Aggregation & Privacy Metrics</span>
       </div>
       <div class="flex gap-3">
           <button v-if="!isRunning" @click="startTraining" class="btn btn-primary">
               <Play class="w-4 h-4 mr-2" /> Start Monitoring
           </button>
           <button v-else @click="stopTraining" class="btn bg-red-600 hover:bg-red-700 text-white">
               <Square class="w-4 h-4 mr-2" /> Stop
           </button>
       </div>
    </div>

    <!-- Progress Bar -->
    <div class="bg-[#111] border border-[#333] rounded-lg p-4">
        <div class="flex items-center justify-between mb-2">
            <span class="text-xs text-gray-500 uppercase">Training Progress</span>
            <span class="text-sm font-mono text-white">Round {{ currentRound }} / {{ totalRounds }}</span>
        </div>
        <div class="h-2 bg-[#333] rounded-full overflow-hidden">
            <div class="h-full bg-gradient-to-r from-primary to-purple-500 transition-all duration-500" :style="{ width: progress + '%' }"></div>
        </div>
    </div>

    <!-- Main Grid -->
    <div class="flex-1 grid grid-cols-12 gap-6 min-h-0">
        <!-- Left Column: Metrics -->
        <div class="col-span-8 space-y-4 overflow-y-auto">
            <!-- Stats Cards -->
            <div class="grid grid-cols-4 gap-4">
                <div class="bg-[#111] border border-[#333] rounded-lg p-4">
                    <div class="flex items-center gap-2 mb-2">
                        <Users class="w-4 h-4 text-blue-500" />
                        <span class="text-[10px] text-gray-500 uppercase">Active Clients</span>
                    </div>
                    <div class="text-2xl font-bold text-white">{{ activeClients }}</div>
                </div>
                <div class="bg-[#111] border border-[#333] rounded-lg p-4">
                    <div class="flex items-center gap-2 mb-2">
                        <Database class="w-4 h-4 text-green-500" />
                        <span class="text-[10px] text-gray-500 uppercase">Aggregated</span>
                    </div>
                    <div class="text-2xl font-bold text-green-500">{{ aggregatedUpdates }}</div>
                </div>
                <div class="bg-[#111] border border-[#333] rounded-lg p-4">
                    <div class="flex items-center gap-2 mb-2">
                        <Shield class="w-4 h-4 text-purple-500" />
                        <span class="text-[10px] text-gray-500 uppercase">Privacy (ε)</span>
                    </div>
                    <div class="text-2xl font-bold" :class="privacyBudget.epsilon > 8 ? 'text-yellow-500' : 'text-purple-500'">
                        {{ privacyBudget.epsilon.toFixed(2) }}
                    </div>
                </div>
                <div class="bg-[#111] border border-[#333] rounded-lg p-4">
                    <div class="flex items-center gap-2 mb-2">
                        <Activity class="w-4 h-4" :class="getHealthColor(healthStatus)" />
                        <span class="text-[10px] text-gray-500 uppercase">Health</span>
                    </div>
                    <div class="text-2xl font-bold capitalize" :class="getHealthColor(healthStatus)">{{ healthStatus }}</div>
                </div>
            </div>

            <!-- Metrics Charts -->
            <div class="grid grid-cols-2 gap-4">
                <!-- Loss Chart -->
                <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-4">
                    <div class="flex items-center justify-between mb-4">
                        <div class="flex items-center gap-2">
                            <TrendingDown class="w-4 h-4 text-green-500" />
                            <span class="text-xs font-bold text-gray-400 uppercase">Loss</span>
                        </div>
                        <span class="text-lg font-bold text-green-500">{{ avgLoss }}</span>
                    </div>
                    <svg class="w-full h-12" viewBox="0 0 200 40" preserveAspectRatio="none">
                        <polyline
                            :points="sparklinePoints(metrics.loss)"
                            fill="none"
                            stroke="#22c55e"
                            stroke-width="2"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                        />
                    </svg>
                </div>

                <!-- Accuracy Chart -->
                <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-4">
                    <div class="flex items-center justify-between mb-4">
                        <div class="flex items-center gap-2">
                            <TrendingUp class="w-4 h-4 text-blue-500" />
                            <span class="text-xs font-bold text-gray-400 uppercase">Accuracy</span>
                        </div>
                        <span class="text-lg font-bold text-blue-500">{{ avgAccuracy }}%</span>
                    </div>
                    <svg class="w-full h-12" viewBox="0 0 200 40" preserveAspectRatio="none">
                        <polyline
                            :points="sparklinePoints(metrics.accuracy)"
                            fill="none"
                            stroke="#3b82f6"
                            stroke-width="2"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                        />
                    </svg>
                </div>

                <!-- Bandwidth Chart -->
                <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-4">
                    <div class="flex items-center justify-between mb-4">
                        <div class="flex items-center gap-2">
                            <Zap class="w-4 h-4 text-orange-500" />
                            <span class="text-xs font-bold text-gray-400 uppercase">Bandwidth (MB/s)</span>
                        </div>
                        <span class="text-lg font-bold text-orange-500">
                            {{ (metrics.bandwidth[metrics.bandwidth.length - 1] || 0).toFixed(2) }}
                        </span>
                    </div>
                    <svg class="w-full h-12" viewBox="0 0 200 40" preserveAspectRatio="none">
                        <polyline
                            :points="sparklinePoints(metrics.bandwidth)"
                            fill="none"
                            stroke="#f97316"
                            stroke-width="2"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                        />
                    </svg>
                </div>

                <!-- Latency Chart -->
                <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-4">
                    <div class="flex items-center justify-between mb-4">
                        <div class="flex items-center gap-2">
                            <Clock class="w-4 h-4 text-purple-500" />
                            <span class="text-xs font-bold text-gray-400 uppercase">Latency (ms)</span>
                        </div>
                        <span class="text-lg font-bold text-purple-500">
                            {{ (metrics.latency[metrics.latency.length - 1] || 0).toFixed(1) }}
                        </span>
                    </div>
                    <svg class="w-full h-12" viewBox="0 0 200 40" preserveAspectRatio="none">
                        <polyline
                            :points="sparklinePoints(metrics.latency)"
                            fill="none"
                            stroke="#a855f7"
                            stroke-width="2"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                        />
                    </svg>
                </div>
            </div>

            <!-- Expert Weights (MoI) -->
            <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-4">
                <h3 class="text-xs font-bold text-gray-500 uppercase mb-4">FedMoE Expert Weights (Mixture of Intelligence)</h3>
                <div class="space-y-3">
                    <div v-for="(weight, expert) in expertWeights" :key="expert" class="flex items-center gap-4">
                        <span class="text-xs text-gray-400 w-32 truncate">{{ expert }}</span>
                        <div class="flex-1 h-2 bg-[#333] rounded-full overflow-hidden">
                            <div class="h-full bg-gradient-to-r from-primary to-yellow-500 transition-all duration-300"
                                 :style="{ width: (weight * 100) + '%' }"></div>
                        </div>
                        <span class="text-xs font-mono text-white w-12 text-right">{{ (weight * 100).toFixed(1) }}%</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Right Column: Event Feed -->
        <div class="col-span-4 bg-[#0d1117] border border-[#30363d] rounded-lg flex flex-col overflow-hidden">
            <div class="p-4 border-b border-[#30363d] flex items-center justify-between">
                <h3 class="text-xs font-bold text-gray-500 uppercase">Live Event Feed</h3>
                <div v-if="isRunning" class="flex items-center gap-2">
                    <div class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                    <span class="text-[10px] text-green-500 font-mono uppercase">Live</span>
                </div>
            </div>
            <div class="flex-1 overflow-y-auto p-4 space-y-2">
                <div v-if="realtimeEvents.length === 0" class="text-center py-8 text-gray-600">
                    <Activity class="w-8 h-8 mx-auto mb-2 opacity-30" />
                    <p class="text-xs">No events yet. Start monitoring to see live updates.</p>
                </div>
                <div v-for="(event, idx) in realtimeEvents" :key="idx"
                     class="bg-[#161b22] border border-[#30363d] rounded p-3 text-xs">
                    <div class="flex items-center gap-2 mb-1">
                        <CheckCircle v-if="event.type === 'success'" class="w-3 h-3 text-green-500" />
                        <AlertTriangle v-else-if="event.type === 'warning'" class="w-3 h-3 text-yellow-500" />
                        <Activity v-else class="w-3 h-3 text-blue-500" />
                        <span class="text-gray-500 font-mono">{{ event.time }}</span>
                    </div>
                    <p class="text-gray-300">{{ event.message }}</p>
                </div>
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
</style>
