<script setup>
/**
 * Operations Center - Unified Fleet & Training Operations
 *
 * Consolidates: Fleets & Devices, Training Monitor, TGSP Marketplace, Integrations
 * Real-time operations monitoring and fleet management
 */
import { ref, computed, onMounted, onUnmounted } from 'vue'
import {
    Server, Radio, Package, Link, RefreshCw, Plus,
    Activity, Users, Shield, Zap, TrendingUp, TrendingDown,
    CheckCircle, AlertTriangle, Clock, Play, Square,
    Upload, Download, Cpu, Cloud, Box
} from 'lucide-vue-next'

const props = defineProps({
    initialTab: { type: String, default: 'fleets' }
})

const activeTab = ref(props.initialTab)

const tabs = [
    { id: 'fleets', label: 'Fleet Management', icon: Server },
    { id: 'monitor', label: 'Training Monitor', icon: Radio },
    { id: 'packages', label: 'TGSP Packages', icon: Package },
    { id: 'integrations', label: 'Integrations', icon: Link }
]

// Fleet data
const fleets = ref([])
const loading = ref(true)

// Training monitor state
const isMonitoring = ref(false)
const currentRound = ref(0)
const metrics = ref({ loss: [], accuracy: [] })
const expertWeights = ref({})
let monitorInterval = null

// Packages
const packages = ref([])

// Integrations
const integrations = ref([
    { id: 'isaac_lab', name: 'NVIDIA Isaac Lab', status: 'connected', icon: Cpu, color: 'text-green-500' },
    { id: 'ros2', name: 'ROS2 Bridge', status: 'active', icon: Radio, color: 'text-blue-500' },
    { id: 'formant', name: 'Formant.io', status: 'disconnected', icon: Cloud, color: 'text-gray-500' },
    { id: 'huggingface', name: 'Hugging Face', status: 'connected', icon: Box, color: 'text-yellow-500' }
])

const fetchFleets = async () => {
    loading.value = true
    try {
        const res = await fetch('/api/v1/fleets/extended')
        if (res.ok) {
            fleets.value = await res.json()
        }
    } catch (e) {
        fleets.value = [
            { id: 'f1', name: 'US-East Production', region: 'us-east-1', status: 'Healthy', devices_total: 450, devices_online: 442, trust: 99.2 },
            { id: 'f2', name: 'EU Gigafactory', region: 'eu-central-1', status: 'Degraded', devices_total: 120, devices_online: 89, trust: 84.5 },
            { id: 'f3', name: 'APAC Logistics', region: 'ap-southeast-1', status: 'Healthy', devices_total: 85, devices_online: 85, trust: 100 }
        ]
    }
    loading.value = false
}

const fetchPackages = async () => {
    try {
        const res = await fetch('/api/v1/tgsp/packages')
        if (res.ok) {
            packages.value = await res.json()
        }
    } catch (e) {
        packages.value = [
            { id: 'tgsp-001', filename: 'factory-assembly-v2.1.tgsp', status: 'verified', producer_id: 'tensorguard-official' },
            { id: 'tgsp-002', filename: 'logistics-picker-v1.0.tgsp', status: 'uploaded', producer_id: 'community/robotics-lab' }
        ]
    }
}

// Training monitor functions
const startMonitoring = () => {
    isMonitoring.value = true
    monitorInterval = setInterval(() => {
        currentRound.value++

        // Simulate metrics
        const newLoss = Math.max(0.01, (metrics.value.loss[metrics.value.loss.length - 1] || 0.5) - Math.random() * 0.02)
        const newAcc = Math.min(0.99, (metrics.value.accuracy[metrics.value.accuracy.length - 1] || 0.5) + Math.random() * 0.01)
        metrics.value.loss.push(newLoss)
        metrics.value.accuracy.push(newAcc)

        if (metrics.value.loss.length > 30) {
            metrics.value.loss.shift()
            metrics.value.accuracy.shift()
        }

        expertWeights.value = {
            'visual_primary': 0.35 + Math.random() * 0.05,
            'language_semantic': 0.25 + Math.random() * 0.03,
            'manipulation_grasp': 0.20 + Math.random() * 0.04,
            'navigation_base': 0.20 + Math.random() * 0.03
        }
    }, 2000)
}

const stopMonitoring = () => {
    isMonitoring.value = false
    if (monitorInterval) {
        clearInterval(monitorInterval)
        monitorInterval = null
    }
}

const sparklinePoints = (data, height = 40) => {
    if (data.length < 2) return ''
    const max = Math.max(...data)
    const min = Math.min(...data)
    const range = max - min || 1
    const width = 200
    const step = width / (data.length - 1)
    return data.map((v, i) => `${i * step},${height - ((v - min) / range) * height}`).join(' ')
}

const getStatusColor = (status) => {
    const colors = { Healthy: 'text-green-500', Degraded: 'text-yellow-500', Critical: 'text-red-500' }
    return colors[status] || 'text-gray-500'
}

const getPackageStatus = (status) => {
    const styles = {
        verified: 'bg-green-500/10 text-green-500 border-green-500/30',
        uploaded: 'bg-yellow-500/10 text-yellow-500 border-yellow-500/30',
        rejected: 'bg-red-500/10 text-red-500 border-red-500/30'
    }
    return styles[status] || 'bg-gray-500/10 text-gray-500 border-gray-500/30'
}

onMounted(() => {
    fetchFleets()
    fetchPackages()
    // Initialize metrics
    for (let i = 0; i < 10; i++) {
        metrics.value.loss.push(0.5 - i * 0.02 + Math.random() * 0.05)
        metrics.value.accuracy.push(0.5 + i * 0.03 + Math.random() * 0.02)
    }
})

onUnmounted(() => {
    if (monitorInterval) clearInterval(monitorInterval)
})
</script>

<template>
  <div class="h-full flex flex-col">
    <!-- Header with Tabs -->
    <div class="flex-shrink-0 border-b border-[#30363d] bg-[#0d1117]">
      <div class="px-6 pt-4">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h1 class="text-xl font-bold text-white">Operations</h1>
            <p class="text-xs text-gray-500">Fleet management, training, and deployments</p>
          </div>
          <button @click="fetchFleets" class="p-2 rounded hover:bg-[#1f2428] transition-colors">
            <RefreshCw class="w-4 h-4 text-gray-400" :class="loading ? 'animate-spin' : ''" />
          </button>
        </div>

        <div class="flex gap-1">
          <button v-for="tab in tabs" :key="tab.id"
                  @click="activeTab = tab.id"
                  :class="['px-4 py-2.5 rounded-t-lg flex items-center gap-2 transition-colors text-sm font-medium',
                           activeTab === tab.id
                             ? 'bg-[#161b22] text-white border-t border-x border-[#30363d]'
                             : 'text-gray-400 hover:text-white hover:bg-[#161b22]/50']">
            <component :is="tab.icon" class="w-4 h-4" />
            {{ tab.label }}
          </button>
        </div>
      </div>
    </div>

    <!-- Tab Content -->
    <div class="flex-1 overflow-hidden bg-[#161b22]">
      <!-- Fleets Tab -->
      <div v-if="activeTab === 'fleets'" class="h-full overflow-y-auto p-6">
        <div class="flex items-center justify-between mb-6">
          <div class="flex items-center gap-4">
            <div class="bg-[#0d1117] border border-[#30363d] rounded-lg px-4 py-2">
              <span class="text-2xl font-bold text-white">{{ fleets.length }}</span>
              <span class="text-xs text-gray-500 ml-2">fleets</span>
            </div>
            <div class="bg-[#0d1117] border border-[#30363d] rounded-lg px-4 py-2">
              <span class="text-2xl font-bold text-green-500">{{ fleets.reduce((sum, f) => sum + f.devices_online, 0) }}</span>
              <span class="text-xs text-gray-500 ml-2">devices online</span>
            </div>
          </div>
          <button class="px-4 py-2 bg-primary hover:bg-primary/90 text-white rounded-lg font-medium flex items-center gap-2">
            <Plus class="w-4 h-4" /> Add Fleet
          </button>
        </div>

        <div class="space-y-4">
          <div v-for="fleet in fleets" :key="fleet.id"
               class="bg-[#0d1117] border border-[#30363d] rounded-lg p-5 hover:border-[#484f58] transition-colors">
            <div class="flex items-start justify-between mb-4">
              <div class="flex items-center gap-4">
                <div class="w-12 h-12 rounded-lg bg-[#1f2428] border border-[#30363d] flex items-center justify-center">
                  <Server class="w-6 h-6 text-gray-400" />
                </div>
                <div>
                  <h3 class="font-semibold text-white">{{ fleet.name }}</h3>
                  <div class="flex items-center gap-2 text-xs text-gray-500">
                    <span>{{ fleet.region }}</span>
                    <span>•</span>
                    <span :class="getStatusColor(fleet.status)">{{ fleet.status }}</span>
                  </div>
                </div>
              </div>
              <div class="text-right">
                <div class="text-2xl font-bold text-white">{{ fleet.trust }}%</div>
                <div class="text-xs text-gray-500">Trust Score</div>
              </div>
            </div>

            <div class="grid grid-cols-3 gap-4">
              <div class="bg-[#161b22] p-3 rounded border border-[#30363d]">
                <div class="text-xs text-gray-500">Total Devices</div>
                <div class="text-lg font-bold text-white">{{ fleet.devices_total }}</div>
              </div>
              <div class="bg-[#161b22] p-3 rounded border border-[#30363d]">
                <div class="text-xs text-gray-500">Online</div>
                <div class="text-lg font-bold text-green-500">{{ fleet.devices_online }}</div>
              </div>
              <div class="bg-[#161b22] p-3 rounded border border-[#30363d]">
                <div class="text-xs text-gray-500">Utilization</div>
                <div class="text-lg font-bold text-blue-500">{{ ((fleet.devices_online / fleet.devices_total) * 100).toFixed(0) }}%</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Training Monitor Tab -->
      <div v-else-if="activeTab === 'monitor'" class="h-full overflow-y-auto p-6">
        <div class="flex items-center justify-between mb-6">
          <div class="flex items-center gap-4">
            <div v-if="isMonitoring" class="flex items-center gap-2">
              <div class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
              <span class="text-sm text-green-500 font-medium">Live</span>
            </div>
            <span class="text-sm text-gray-500">Round {{ currentRound }}</span>
          </div>
          <button v-if="!isMonitoring" @click="startMonitoring"
                  class="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium flex items-center gap-2">
            <Play class="w-4 h-4" /> Start Monitoring
          </button>
          <button v-else @click="stopMonitoring"
                  class="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium flex items-center gap-2">
            <Square class="w-4 h-4" /> Stop
          </button>
        </div>

        <div class="grid grid-cols-2 gap-6 mb-6">
          <!-- Loss Chart -->
          <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-5">
            <div class="flex items-center justify-between mb-4">
              <div class="flex items-center gap-2">
                <TrendingDown class="w-4 h-4 text-green-500" />
                <span class="text-sm font-medium text-gray-400">Loss</span>
              </div>
              <span class="text-lg font-bold text-green-500">
                {{ (metrics.loss[metrics.loss.length - 1] || 0).toFixed(4) }}
              </span>
            </div>
            <svg class="w-full h-16" viewBox="0 0 200 40" preserveAspectRatio="none">
              <polyline :points="sparklinePoints(metrics.loss)" fill="none" stroke="#22c55e" stroke-width="2" />
            </svg>
          </div>

          <!-- Accuracy Chart -->
          <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-5">
            <div class="flex items-center justify-between mb-4">
              <div class="flex items-center gap-2">
                <TrendingUp class="w-4 h-4 text-blue-500" />
                <span class="text-sm font-medium text-gray-400">Accuracy</span>
              </div>
              <span class="text-lg font-bold text-blue-500">
                {{ ((metrics.accuracy[metrics.accuracy.length - 1] || 0) * 100).toFixed(1) }}%
              </span>
            </div>
            <svg class="w-full h-16" viewBox="0 0 200 40" preserveAspectRatio="none">
              <polyline :points="sparklinePoints(metrics.accuracy)" fill="none" stroke="#3b82f6" stroke-width="2" />
            </svg>
          </div>
        </div>

        <!-- Expert Weights -->
        <div class="bg-[#0d1117] border border-[#30363d] rounded-lg p-5">
          <h3 class="text-sm font-medium text-gray-400 mb-4">FedMoE Expert Weights</h3>
          <div class="space-y-3">
            <div v-for="(weight, name) in expertWeights" :key="name" class="flex items-center gap-4">
              <span class="text-xs text-gray-400 w-32 truncate">{{ name }}</span>
              <div class="flex-1 h-2 bg-[#30363d] rounded-full overflow-hidden">
                <div class="h-full bg-gradient-to-r from-primary to-yellow-500 transition-all"
                     :style="{ width: (weight * 100) + '%' }"></div>
              </div>
              <span class="text-xs font-mono text-white w-12 text-right">{{ (weight * 100).toFixed(1) }}%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Packages Tab -->
      <div v-else-if="activeTab === 'packages'" class="h-full overflow-y-auto p-6">
        <div class="flex items-center justify-between mb-6">
          <h2 class="text-lg font-semibold text-white">TGSP Packages</h2>
          <button class="px-4 py-2 bg-primary hover:bg-primary/90 text-white rounded-lg font-medium flex items-center gap-2">
            <Upload class="w-4 h-4" /> Upload Package
          </button>
        </div>

        <div class="space-y-4">
          <div v-for="pkg in packages" :key="pkg.id"
               class="bg-[#0d1117] border border-[#30363d] rounded-lg p-5 hover:border-[#484f58] transition-colors">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-4">
                <Package class="w-6 h-6 text-purple-400" />
                <div>
                  <div class="font-semibold text-white">{{ pkg.filename }}</div>
                  <div class="text-xs text-gray-500">{{ pkg.producer_id }}</div>
                </div>
              </div>
              <div class="flex items-center gap-3">
                <span :class="['text-xs font-bold uppercase px-2 py-1 rounded border', getPackageStatus(pkg.status)]">
                  {{ pkg.status }}
                </span>
                <button class="p-2 hover:bg-[#1f2428] rounded transition-colors">
                  <Download class="w-4 h-4 text-gray-400" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Integrations Tab -->
      <div v-else-if="activeTab === 'integrations'" class="h-full overflow-y-auto p-6">
        <div class="grid grid-cols-2 gap-6">
          <div v-for="int in integrations" :key="int.id"
               class="bg-[#0d1117] border border-[#30363d] rounded-lg p-5 hover:border-[#484f58] transition-colors">
            <div class="flex items-center justify-between mb-4">
              <div class="flex items-center gap-3">
                <component :is="int.icon" class="w-6 h-6" :class="int.color" />
                <span class="font-semibold text-white">{{ int.name }}</span>
              </div>
              <span :class="['text-xs font-medium capitalize',
                     int.status === 'connected' || int.status === 'active' ? 'text-green-500' : 'text-gray-500']">
                {{ int.status }}
              </span>
            </div>
            <button class="w-full px-4 py-2 border border-[#30363d] rounded-lg text-sm font-medium hover:bg-[#1f2428] transition-colors"
                    :class="int.status === 'connected' || int.status === 'active' ? 'text-gray-400' : 'text-primary'">
              {{ int.status === 'connected' || int.status === 'active' ? 'Reconfigure' : 'Connect' }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
