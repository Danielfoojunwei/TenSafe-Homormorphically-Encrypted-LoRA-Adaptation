<script setup>
import { ref, onMounted } from 'vue'
import {
    Link, CheckCircle, XCircle, Clock, RefreshCw,
    Cpu, Radio, Cloud, Box, Zap, Settings,
    Play, AlertTriangle, Activity
} from 'lucide-vue-next'

const integrations = ref([])
const loading = ref(true)
const connecting = ref(null)
const showConfigModal = ref(false)
const selectedIntegration = ref(null)

const integrationConfigs = ref({
    isaac_lab: { omniverse_url: '', api_key: '' },
    ros2_bridge: { domain_id: '42', topics_filter: '' },
    formant: { device_id: '', api_key: '' },
    huggingface: { model_id: '', token: '' }
})

const availableIntegrations = [
    {
        id: 'isaac_lab',
        name: 'NVIDIA Isaac Lab',
        description: 'High-fidelity robotics simulation with physics engine',
        icon: Cpu,
        color: 'text-green-500',
        bgColor: 'bg-green-500/10',
        fields: [
            { key: 'omniverse_url', label: 'Omniverse Nucleus URL', placeholder: 'omniverse://nucleus.local' },
            { key: 'api_key', label: 'API Key', placeholder: 'nvidia_api_key_xxx', type: 'password' }
        ]
    },
    {
        id: 'ros2_bridge',
        name: 'ROS2 Bridge',
        description: 'Robot Operating System 2 middleware integration',
        icon: Radio,
        color: 'text-blue-500',
        bgColor: 'bg-blue-500/10',
        fields: [
            { key: 'domain_id', label: 'Domain ID', placeholder: '42' },
            { key: 'topics_filter', label: 'Topics Filter (optional)', placeholder: '/robot/*, /sensor/*' }
        ]
    },
    {
        id: 'formant',
        name: 'Formant.io',
        description: 'Fleet management and telemetry platform',
        icon: Cloud,
        color: 'text-purple-500',
        bgColor: 'bg-purple-500/10',
        fields: [
            { key: 'device_id', label: 'Device ID', placeholder: 'formant-device-xxx' },
            { key: 'api_key', label: 'API Key', placeholder: 'fmnt_xxx', type: 'password' }
        ]
    },
    {
        id: 'huggingface',
        name: 'Hugging Face',
        description: 'Model hub for VLA and foundation models',
        icon: Box,
        color: 'text-yellow-500',
        bgColor: 'bg-yellow-500/10',
        fields: [
            { key: 'model_id', label: 'Model ID', placeholder: 'openvla/openvla-7b' },
            { key: 'token', label: 'HF Token', placeholder: 'hf_xxx', type: 'password' }
        ]
    }
]

const fetchStatus = async () => {
    loading.value = true
    try {
        const res = await fetch('/api/v1/integrations/status')
        if (res.ok) {
            const data = await res.json()
            // Merge status with available integrations
            integrations.value = availableIntegrations.map(int => ({
                ...int,
                status: data[int.id]?.status || 'disconnected',
                details: data[int.id] || {}
            }))
        }
    } catch (e) {
        console.error("Failed to fetch integration status", e)
        // Use default disconnected state
        integrations.value = availableIntegrations.map(int => ({
            ...int,
            status: 'disconnected',
            details: {}
        }))
    }
    loading.value = false
}

const connectIntegration = async (integration) => {
    connecting.value = integration.id
    const config = integrationConfigs.value[integration.id]

    try {
        const res = await fetch('/api/v1/integrations/connect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                service: integration.id,
                config: config
            })
        })

        if (res.ok) {
            const data = await res.json()
            // Update integration status
            const idx = integrations.value.findIndex(i => i.id === integration.id)
            if (idx >= 0) {
                integrations.value[idx].status = data.status
                integrations.value[idx].details = { latency_ms: data.latency_ms, message: data.message }
            }
            showConfigModal.value = false
        } else {
            const err = await res.json()
            alert(`Connection failed: ${err.detail || err.message}`)
        }
    } catch (e) {
        console.error("Failed to connect", e)
        alert("Connection failed. Check console for details.")
    }
    connecting.value = null
}

const openConfigModal = (integration) => {
    selectedIntegration.value = integration
    showConfigModal.value = true
}

const getStatusIcon = (status) => {
    if (status === 'connected' || status === 'active' || status === 'validated') return CheckCircle
    if (status === 'error') return XCircle
    return Clock
}

const getStatusColor = (status) => {
    if (status === 'connected' || status === 'active' || status === 'validated') return 'text-green-500'
    if (status === 'error') return 'text-red-500'
    return 'text-gray-500'
}

onMounted(fetchStatus)
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between border-b border-[#333] pb-6">
       <div>
         <h2 class="text-2xl font-bold flex items-center gap-3">
             <Link class="w-7 h-7 text-blue-500" />
             External Integrations Hub
         </h2>
         <span class="text-xs text-gray-500">Connect to Isaac Lab, ROS2, Formant, and Hugging Face</span>
       </div>
       <button @click="fetchStatus" :disabled="loading" class="btn btn-secondary">
           <RefreshCw class="w-4 h-4" :class="loading ? 'animate-spin' : ''" />
       </button>
    </div>

    <!-- Integration Cards Grid -->
    <div v-if="loading" class="flex justify-center py-12">
        <div class="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
    </div>

    <div v-else class="grid grid-cols-2 gap-6">
        <div v-for="integration in integrations" :key="integration.id"
             class="bg-[#0d1117] border border-[#30363d] rounded-xl overflow-hidden hover:border-primary/50 transition-all group">
            <!-- Header -->
            <div class="p-6 border-b border-[#30363d]">
                <div class="flex items-start justify-between">
                    <div class="flex items-center gap-4">
                        <div :class="['w-14 h-14 rounded-xl flex items-center justify-center', integration.bgColor]">
                            <component :is="integration.icon" class="w-7 h-7" :class="integration.color" />
                        </div>
                        <div>
                            <h3 class="font-bold text-lg text-white">{{ integration.name }}</h3>
                            <p class="text-xs text-gray-500 mt-1">{{ integration.description }}</p>
                        </div>
                    </div>
                    <div class="flex items-center gap-2">
                        <component :is="getStatusIcon(integration.status)"
                                   class="w-5 h-5"
                                   :class="getStatusColor(integration.status)" />
                        <span :class="['text-xs font-bold uppercase', getStatusColor(integration.status)]">
                            {{ integration.status }}
                        </span>
                    </div>
                </div>
            </div>

            <!-- Details -->
            <div class="p-6 bg-[#161b22]">
                <div v-if="integration.status === 'connected' || integration.status === 'active'" class="space-y-3">
                    <div class="flex items-center justify-between text-sm">
                        <span class="text-gray-500">Latency</span>
                        <span class="font-mono text-green-500">{{ integration.details.latency_ms || 'N/A' }}ms</span>
                    </div>
                    <div v-if="integration.details.uptime" class="flex items-center justify-between text-sm">
                        <span class="text-gray-500">Uptime</span>
                        <span class="font-mono text-white">{{ integration.details.uptime }}</span>
                    </div>
                    <div v-if="integration.details.topics" class="flex items-center justify-between text-sm">
                        <span class="text-gray-500">Topics</span>
                        <span class="font-mono text-white">{{ integration.details.topics }}</span>
                    </div>
                    <div v-if="integration.details.message" class="text-xs text-gray-400 mt-2 p-2 bg-[#0d1117] rounded">
                        {{ integration.details.message }}
                    </div>
                </div>

                <div v-else class="text-center py-4">
                    <Activity class="w-8 h-8 text-gray-700 mx-auto mb-2" />
                    <p class="text-xs text-gray-500">Not connected</p>
                </div>

                <!-- Action Buttons -->
                <div class="mt-4 flex gap-2">
                    <button @click="openConfigModal(integration)"
                            :disabled="connecting === integration.id"
                            class="flex-1 btn"
                            :class="integration.status === 'connected' || integration.status === 'active'
                                ? 'btn-secondary' : 'btn-primary'">
                        <Zap class="w-4 h-4 mr-2" :class="connecting === integration.id ? 'animate-pulse' : ''" />
                        {{ integration.status === 'connected' || integration.status === 'active' ? 'Reconfigure' : 'Connect' }}
                    </button>
                    <button v-if="integration.status === 'connected' || integration.status === 'active'"
                            class="btn btn-secondary"
                            title="View Logs">
                        <Settings class="w-4 h-4" />
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Integration Health Summary -->
    <div class="bg-[#111] border border-[#333] rounded-lg p-6">
        <h3 class="text-xs font-bold text-gray-500 uppercase mb-4">Integration Health Summary</h3>
        <div class="grid grid-cols-4 gap-4">
            <div class="text-center">
                <div class="text-3xl font-bold text-green-500">
                    {{ integrations.filter(i => i.status === 'connected' || i.status === 'active').length }}
                </div>
                <div class="text-xs text-gray-500 mt-1">Connected</div>
            </div>
            <div class="text-center">
                <div class="text-3xl font-bold text-gray-500">
                    {{ integrations.filter(i => i.status === 'disconnected').length }}
                </div>
                <div class="text-xs text-gray-500 mt-1">Disconnected</div>
            </div>
            <div class="text-center">
                <div class="text-3xl font-bold text-red-500">
                    {{ integrations.filter(i => i.status === 'error').length }}
                </div>
                <div class="text-xs text-gray-500 mt-1">Errors</div>
            </div>
            <div class="text-center">
                <div class="text-3xl font-bold text-blue-500">
                    {{ integrations.length }}
                </div>
                <div class="text-xs text-gray-500 mt-1">Total</div>
            </div>
        </div>
    </div>

    <!-- Configuration Modal -->
    <div v-if="showConfigModal && selectedIntegration" class="fixed inset-0 bg-black/90 flex items-center justify-center z-50 backdrop-blur-sm p-4">
        <div class="bg-[#0f0f0f] border border-primary/30 w-full max-w-lg rounded-xl shadow-2xl overflow-hidden">
            <div class="p-6 border-b border-[#222] flex items-center gap-4">
                <div :class="['w-12 h-12 rounded-xl flex items-center justify-center', selectedIntegration.bgColor]">
                    <component :is="selectedIntegration.icon" class="w-6 h-6" :class="selectedIntegration.color" />
                </div>
                <div>
                    <h3 class="text-xl font-bold text-white">Configure {{ selectedIntegration.name }}</h3>
                    <p class="text-[10px] text-gray-500 uppercase mt-1">{{ selectedIntegration.description }}</p>
                </div>
            </div>
            <div class="p-6 space-y-4">
                <div v-for="field in selectedIntegration.fields" :key="field.key">
                    <label class="text-[10px] text-gray-500 font-bold uppercase mb-2 block">{{ field.label }}</label>
                    <input
                        v-model="integrationConfigs[selectedIntegration.id][field.key]"
                        :type="field.type || 'text'"
                        :placeholder="field.placeholder"
                        class="w-full bg-[#111] border border-[#333] rounded px-4 py-3 text-sm focus:border-primary outline-none transition-colors"
                    />
                </div>

                <!-- Warning for sensitive data -->
                <div class="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3 flex items-start gap-2">
                    <AlertTriangle class="w-4 h-4 text-yellow-500 flex-shrink-0 mt-0.5" />
                    <div class="text-xs text-yellow-500/80">
                        API keys and tokens are stored securely and encrypted at rest. Never share credentials in logs or chat.
                    </div>
                </div>
            </div>
            <div class="p-6 bg-[#141414] flex justify-end gap-3 border-t border-[#222]">
                <button @click="showConfigModal = false" class="text-xs font-bold text-gray-500 uppercase px-4 py-2 hover:text-white transition-colors">Cancel</button>
                <button
                    @click="connectIntegration(selectedIntegration)"
                    :disabled="connecting === selectedIntegration.id"
                    class="btn btn-primary"
                >
                    <Play class="w-4 h-4 mr-2" :class="connecting === selectedIntegration.id ? 'animate-pulse' : ''" />
                    {{ connecting === selectedIntegration.id ? 'Connecting...' : 'Connect' }}
                </button>
            </div>
        </div>
    </div>
  </div>
</template>

<style scoped>
.btn {
  @apply px-4 py-2 rounded font-medium transition-colors duration-200 flex items-center justify-center;
}
.btn-primary {
  @apply bg-orange-600 text-white hover:bg-orange-700;
}
.btn-secondary {
  @apply border border-[#30363d] text-gray-300 hover:text-white hover:bg-[#161b22];
}
</style>
