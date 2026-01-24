<script setup>
/**
 * Command Center - Unified Dashboard for TensorGuardFlow
 *
 * Engineering tool design principles:
 * - Single pane of glass for system status
 * - Action-oriented quick access panels
 * - Real-time metrics with drill-down capability
 */
import { ref, onMounted, onUnmounted, computed } from 'vue'
import {
    Activity, Server, Shield, Zap, AlertTriangle, CheckCircle,
    TrendingUp, Clock, Users, Database, Lock, Package,
    Play, ArrowRight, RefreshCw, Radio, Bot, FileKey
} from 'lucide-vue-next'

const emit = defineEmits(['navigate'])

// System Status
const systemHealth = ref({
    overall: 'healthy',
    services: {
        aggregator: { status: 'healthy', latency: 12 },
        identity: { status: 'healthy', latency: 8 },
        kms: { status: 'healthy', latency: 5 },
        storage: { status: 'degraded', latency: 45 }
    }
})

// Real-time Metrics
const metrics = ref({
    activeFleets: 12,
    connectedDevices: 847,
    activeTrainingRuns: 3,
    pendingDeployments: 2,
    privacyBudget: 4.2,
    certificatesExpiring: 5,
    modelsDeployed: 8,
    successRate: 96.8
})

// Recent Activity
const recentActivity = ref([])
const loading = ref(true)

// Polling
let pollInterval = null

const fetchDashboardData = async () => {
    try {
        // Fetch from multiple endpoints in parallel
        const [statusRes, fleetsRes] = await Promise.allSettled([
            fetch('/api/v1/status'),
            fetch('/api/v1/fleets/extended')
        ])

        if (statusRes.status === 'fulfilled' && statusRes.value.ok) {
            const data = await statusRes.value.json()
            metrics.value.activeTrainingRuns = data.active_runs || metrics.value.activeTrainingRuns
        }

        if (fleetsRes.status === 'fulfilled' && fleetsRes.value.ok) {
            const fleets = await fleetsRes.value.json()
            metrics.value.activeFleets = fleets.length
            metrics.value.connectedDevices = fleets.reduce((sum, f) => sum + (f.devices_online || 0), 0)
        }
    } catch (e) {
        console.warn('Dashboard fetch failed, using cached data')
    }
    loading.value = false
}

const getHealthColor = (status) => {
    const colors = { healthy: 'text-green-500', degraded: 'text-yellow-500', critical: 'text-red-500' }
    return colors[status] || 'text-gray-500'
}

const getHealthBg = (status) => {
    const colors = { healthy: 'bg-green-500', degraded: 'bg-yellow-500', critical: 'bg-red-500' }
    return colors[status] || 'bg-gray-500'
}

// Quick Actions
const quickActions = [
    { id: 'new-training', label: 'Start Training Run', icon: Play, color: 'bg-green-600 hover:bg-green-700', navigate: { page: 'models', tab: 'training' } },
    { id: 'deploy-model', label: 'Deploy Model', icon: Zap, color: 'bg-blue-600 hover:bg-blue-700', navigate: { page: 'models', tab: 'registry' } },
    { id: 'view-fleets', label: 'Fleet Status', icon: Server, color: 'bg-purple-600 hover:bg-purple-700', navigate: { page: 'operations', tab: 'fleets' } },
    { id: 'security', label: 'Security Center', icon: Shield, color: 'bg-orange-600 hover:bg-orange-700', navigate: { page: 'security', tab: 'overview' } }
]

const handleQuickAction = (action) => {
    emit('navigate', action.navigate)
}

onMounted(() => {
    fetchDashboardData()
    pollInterval = setInterval(fetchDashboardData, 30000)
})

onUnmounted(() => {
    if (pollInterval) clearInterval(pollInterval)
})
</script>

<template>
  <div class="h-full overflow-y-auto">
    <div class="max-w-7xl mx-auto p-6 space-y-6">
      <!-- Header -->
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-2xl font-bold text-white">Command Center</h1>
          <p class="text-sm text-gray-500">TensorGuardFlow System Overview</p>
        </div>
        <div class="flex items-center gap-4">
          <div class="flex items-center gap-2 text-sm">
            <div :class="['w-2 h-2 rounded-full animate-pulse', getHealthBg(systemHealth.overall)]"></div>
            <span class="text-gray-400">System {{ systemHealth.overall }}</span>
          </div>
          <button @click="fetchDashboardData" class="p-2 rounded hover:bg-[#1f2428] transition-colors">
            <RefreshCw class="w-4 h-4 text-gray-400" :class="loading ? 'animate-spin' : ''" />
          </button>
        </div>
      </div>

      <!-- Quick Actions -->
      <div class="grid grid-cols-4 gap-4">
        <button v-for="action in quickActions" :key="action.id"
                @click="handleQuickAction(action)"
                :class="['p-4 rounded-lg flex items-center gap-3 transition-all hover:scale-[1.02]', action.color]">
          <component :is="action.icon" class="w-5 h-5 text-white" />
          <span class="font-medium text-white">{{ action.label }}</span>
          <ArrowRight class="w-4 h-4 text-white/70 ml-auto" />
        </button>
      </div>

      <!-- Primary Metrics -->
      <div class="grid grid-cols-4 gap-4">
        <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-5 hover:border-[#484f58] transition-colors cursor-pointer"
             @click="emit('navigate', { page: 'operations', tab: 'fleets' })">
          <div class="flex items-center justify-between mb-3">
            <Server class="w-5 h-5 text-blue-500" />
            <span class="text-xs text-gray-500">FLEETS</span>
          </div>
          <div class="text-3xl font-bold text-white mb-1">{{ metrics.activeFleets }}</div>
          <div class="text-xs text-gray-500">{{ metrics.connectedDevices }} devices online</div>
        </div>

        <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-5 hover:border-[#484f58] transition-colors cursor-pointer"
             @click="emit('navigate', { page: 'operations', tab: 'monitor' })">
          <div class="flex items-center justify-between mb-3">
            <Radio class="w-5 h-5 text-green-500" />
            <span class="text-xs text-gray-500">TRAINING</span>
          </div>
          <div class="text-3xl font-bold text-white mb-1">{{ metrics.activeTrainingRuns }}</div>
          <div class="text-xs text-gray-500">active runs</div>
        </div>

        <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-5 hover:border-[#484f58] transition-colors cursor-pointer"
             @click="emit('navigate', { page: 'models', tab: 'registry' })">
          <div class="flex items-center justify-between mb-3">
            <Bot class="w-5 h-5 text-purple-500" />
            <span class="text-xs text-gray-500">MODELS</span>
          </div>
          <div class="text-3xl font-bold text-white mb-1">{{ metrics.modelsDeployed }}</div>
          <div class="text-xs text-gray-500">{{ metrics.successRate }}% success rate</div>
        </div>

        <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-5 hover:border-[#484f58] transition-colors cursor-pointer"
             @click="emit('navigate', { page: 'security', tab: 'identity' })">
          <div class="flex items-center justify-between mb-3">
            <Shield class="w-5 h-5 text-orange-500" />
            <span class="text-xs text-gray-500">SECURITY</span>
          </div>
          <div class="text-3xl font-bold text-white mb-1">{{ metrics.privacyBudget.toFixed(1) }}</div>
          <div class="text-xs text-gray-500">privacy budget (ε) used</div>
        </div>
      </div>

      <!-- Two Column Layout -->
      <div class="grid grid-cols-3 gap-6">
        <!-- Service Health -->
        <div class="col-span-2 bg-[#0d1117] border border-[#30363d] rounded-lg overflow-hidden">
          <div class="px-5 py-4 border-b border-[#30363d] flex items-center justify-between">
            <h2 class="font-semibold text-white">Service Health</h2>
            <span class="text-xs text-gray-500">Real-time status</span>
          </div>
          <div class="p-5">
            <div class="grid grid-cols-2 gap-4">
              <div v-for="(service, name) in systemHealth.services" :key="name"
                   class="flex items-center justify-between p-3 bg-[#161b22] rounded-lg border border-[#30363d]">
                <div class="flex items-center gap-3">
                  <div :class="['w-2 h-2 rounded-full', getHealthBg(service.status)]"></div>
                  <span class="text-sm font-medium text-white capitalize">{{ name }}</span>
                </div>
                <div class="text-right">
                  <span :class="['text-xs font-medium capitalize', getHealthColor(service.status)]">{{ service.status }}</span>
                  <div class="text-[10px] text-gray-500">{{ service.latency }}ms</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Alerts & Warnings -->
        <div class="bg-[#0d1117] border border-[#30363d] rounded-lg overflow-hidden">
          <div class="px-5 py-4 border-b border-[#30363d] flex items-center justify-between">
            <h2 class="font-semibold text-white">Alerts</h2>
            <span class="text-xs px-2 py-0.5 rounded bg-yellow-500/10 text-yellow-500">2 active</span>
          </div>
          <div class="p-3 space-y-2">
            <div class="flex items-start gap-3 p-3 bg-yellow-500/5 border border-yellow-500/20 rounded-lg">
              <AlertTriangle class="w-4 h-4 text-yellow-500 flex-shrink-0 mt-0.5" />
              <div>
                <div class="text-sm font-medium text-yellow-500">Certificates Expiring</div>
                <div class="text-xs text-gray-500">{{ metrics.certificatesExpiring }} certificates expire within 30 days</div>
              </div>
            </div>
            <div class="flex items-start gap-3 p-3 bg-blue-500/5 border border-blue-500/20 rounded-lg">
              <Clock class="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />
              <div>
                <div class="text-sm font-medium text-blue-500">Pending Deployments</div>
                <div class="text-xs text-gray-500">{{ metrics.pendingDeployments }} models awaiting deployment</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Secondary Metrics Row -->
      <div class="grid grid-cols-6 gap-4">
        <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-4 text-center">
          <div class="text-2xl font-bold text-green-500">99.9%</div>
          <div class="text-[10px] text-gray-500 uppercase mt-1">Uptime</div>
        </div>
        <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-4 text-center">
          <div class="text-2xl font-bold text-blue-500">48.2ms</div>
          <div class="text-[10px] text-gray-500 uppercase mt-1">Avg Latency</div>
        </div>
        <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-4 text-center">
          <div class="text-2xl font-bold text-purple-500">7,844x</div>
          <div class="text-[10px] text-gray-500 uppercase mt-1">BW Reduction</div>
        </div>
        <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-4 text-center">
          <div class="text-2xl font-bold text-orange-500">24</div>
          <div class="text-[10px] text-gray-500 uppercase mt-1">Key Rotations</div>
        </div>
        <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-4 text-center">
          <div class="text-2xl font-bold text-cyan-500">4.21%</div>
          <div class="text-[10px] text-gray-500 uppercase mt-1">NBT Score</div>
        </div>
        <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-4 text-center">
          <div class="text-2xl font-bold text-pink-500">Level 4</div>
          <div class="text-[10px] text-gray-500 uppercase mt-1">Compliance</div>
        </div>
      </div>
    </div>
  </div>
</template>
