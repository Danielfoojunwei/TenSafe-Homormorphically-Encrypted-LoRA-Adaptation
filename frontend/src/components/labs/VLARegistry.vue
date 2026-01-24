<script setup>
import { ref, onMounted, computed } from 'vue'
import {
    Bot, Shield, Zap, CheckCircle, AlertTriangle, Clock,
    Plus, Play, RefreshCw, Eye, Rocket, Activity,
    Target, Settings2, FileCheck, TrendingUp
} from 'lucide-vue-next'

const models = ref([])
const loading = ref(true)
const selectedModel = ref(null)
const showCreateModal = ref(false)
const showSafetyModal = ref(false)
const deploying = ref(null)
const validating = ref(null)
const errorMessage = ref('')

// Create model form
const newModel = ref({
    name: '',
    version: '1.0.0',
    description: '',
    vision_encoder: 'ViT-L/14',
    language_model: 'Llama-3-8B',
    action_head: 'Diffusion-Policy',
    task_types: [],
    action_dim: 7,
    proprioception_dim: 14,
    action_horizon: 16
})

const taskTypeOptions = [
    'pick_and_place', 'navigation', 'manipulation',
    'inspection', 'assembly', 'pouring', 'wiping', 'screwing'
]

const fetchModels = async () => {
    loading.value = true
    errorMessage.value = ''
    try {
        const res = await fetch('/api/v1/vla/models')
        if (res.ok) {
            const data = await res.json()
            models.value = data.models || []
        } else {
            throw new Error('Backend unavailable')
        }
    } catch (e) {
        console.error("Failed to fetch VLA models", e)
        models.value = []
        errorMessage.value = 'Unable to load VLA models. Verify API connectivity.'
    }
    loading.value = false
}

const fetchModelDetails = async (modelId) => {
    try {
        const res = await fetch(`/api/v1/vla/models/${modelId}`)
        if (res.ok) {
            const data = await res.json()
            selectedModel.value = data
        }
    } catch (e) {
        console.error("Failed to fetch model details", e)
    }
}

const createModel = async () => {
    try {
        const res = await fetch('/api/v1/vla/models', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newModel.value)
        })
        if (res.ok) {
            showCreateModal.value = false
            await fetchModels()
            newModel.value = { name: '', version: '1.0.0', description: '', vision_encoder: 'ViT-L/14', language_model: 'Llama-3-8B', action_head: 'Diffusion-Policy', task_types: [], action_dim: 7, proprioception_dim: 14, action_horizon: 16 }
        }
    } catch (e) {
        console.error("Failed to create model", e)
        alert("Failed to create VLA model")
    }
}

const startSafetyValidation = async (modelId) => {
    validating.value = modelId
    try {
        const res = await fetch('/api/v1/vla/safety/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_id: modelId,
                test_environment: 'production',
                test_scenarios: 100
            })
        })
        if (res.ok) {
            await fetchModels()
        }
    } catch (e) {
        console.error("Failed to start validation", e)
    }
    validating.value = null
}

const deployModel = async (modelId) => {
    const fleetId = prompt("Enter Fleet ID to deploy to:")
    if (!fleetId) return

    deploying.value = modelId
    try {
        const res = await fetch('/api/v1/vla/deploy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_id: modelId,
                fleet_id: fleetId,
                rollout_percentage: 100,
                reason: 'Manual deployment from dashboard'
            })
        })
        if (res.ok) {
            await fetchModels()
            alert("Model deployed successfully!")
        } else {
            const err = await res.json()
            alert(`Deployment failed: ${err.detail}`)
        }
    } catch (e) {
        console.error("Failed to deploy", e)
    }
    deploying.value = null
}

const getStatusColor = (status) => {
    const colors = {
        'deployed': 'text-green-500 bg-green-500/10 border-green-500/30',
        'staged': 'text-blue-500 bg-blue-500/10 border-blue-500/30',
        'validating': 'text-yellow-500 bg-yellow-500/10 border-yellow-500/30',
        'failed': 'text-red-500 bg-red-500/10 border-red-500/30'
    }
    return colors[status] || 'text-gray-500 bg-gray-500/10 border-gray-500/30'
}

const getSafetyColor = (score) => {
    if (score === null) return 'text-gray-500'
    if (score >= 0.95) return 'text-green-500'
    if (score >= 0.85) return 'text-yellow-500'
    return 'text-red-500'
}

onMounted(fetchModels)
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between border-b border-[#333] pb-6">
       <div>
         <h2 class="text-2xl font-bold flex items-center gap-3">
             <Bot class="w-7 h-7 text-primary" />
             VLA Model Registry
         </h2>
         <span class="text-xs text-gray-500">Vision-Language-Action Models for Robotics Humanoids</span>
       </div>
       <div class="flex gap-3">
           <button @click="fetchModels" :disabled="loading" class="btn btn-secondary">
               <RefreshCw class="w-4 h-4" :class="loading ? 'animate-spin' : ''" />
           </button>
           <button @click="showCreateModal = true" class="btn btn-primary">
               <Plus class="w-4 h-4 mr-2" /> Register Model
           </button>
       </div>
    </div>

    <!-- Stats Cards -->
    <div v-if="errorMessage" class="text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded p-3">
        {{ errorMessage }}
    </div>
    <div class="grid grid-cols-4 gap-4">
        <div class="bg-[#111] border border-[#333] rounded-lg p-4">
            <div class="flex items-center gap-3 mb-2">
                <Bot class="w-5 h-5 text-blue-500" />
                <span class="text-xs text-gray-500 uppercase">Total Models</span>
            </div>
            <div class="text-2xl font-bold text-white">{{ models.length }}</div>
        </div>
        <div class="bg-[#111] border border-[#333] rounded-lg p-4">
            <div class="flex items-center gap-3 mb-2">
                <Rocket class="w-5 h-5 text-green-500" />
                <span class="text-xs text-gray-500 uppercase">Deployed</span>
            </div>
            <div class="text-2xl font-bold text-green-500">{{ models.filter(m => m.status === 'deployed').length }}</div>
        </div>
        <div class="bg-[#111] border border-[#333] rounded-lg p-4">
            <div class="flex items-center gap-3 mb-2">
                <Shield class="w-5 h-5 text-purple-500" />
                <span class="text-xs text-gray-500 uppercase">Avg Safety Score</span>
            </div>
            <div class="text-2xl font-bold text-purple-500">
                {{ (models.filter(m => m.safety_score).reduce((a, m) => a + m.safety_score, 0) / Math.max(models.filter(m => m.safety_score).length, 1) * 100).toFixed(1) }}%
            </div>
        </div>
        <div class="bg-[#111] border border-[#333] rounded-lg p-4">
            <div class="flex items-center gap-3 mb-2">
                <Zap class="w-5 h-5 text-orange-500" />
                <span class="text-xs text-gray-500 uppercase">Avg Latency</span>
            </div>
            <div class="text-2xl font-bold text-orange-500">
                {{ (models.reduce((a, m) => a + (m.avg_latency_ms || 0), 0) / Math.max(models.length, 1)).toFixed(1) }}ms
            </div>
        </div>
    </div>

    <!-- Models Grid -->
    <div v-if="loading" class="flex justify-center py-12">
        <div class="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
    </div>

    <div v-else class="grid grid-cols-1 gap-4">
        <div v-for="model in models" :key="model.id"
             class="bg-[#0d1117] border border-[#30363d] rounded-lg p-6 hover:border-primary/50 transition-colors">
            <div class="flex items-start justify-between mb-4">
                <div class="flex items-center gap-4">
                    <div class="w-12 h-12 rounded-lg bg-gradient-to-br from-primary/20 to-purple-500/20 border border-primary/30 flex items-center justify-center">
                        <Bot class="w-6 h-6 text-primary" />
                    </div>
                    <div>
                        <div class="flex items-center gap-2">
                            <h3 class="font-bold text-lg text-white">{{ model.name }}</h3>
                            <span class="text-xs font-mono text-gray-500">v{{ model.version }}</span>
                        </div>
                        <div class="flex items-center gap-2 mt-1">
                            <span :class="['text-[10px] font-bold uppercase px-2 py-0.5 rounded border', getStatusColor(model.status)]">
                                {{ model.status }}
                            </span>
                            <span class="text-xs text-gray-500">{{ model.id }}</span>
                        </div>
                    </div>
                </div>

                <div class="flex items-center gap-2">
                    <button @click="fetchModelDetails(model.id)" class="btn btn-sm btn-secondary" title="View Details">
                        <Eye class="w-4 h-4" />
                    </button>
                    <button
                        v-if="model.status === 'staged'"
                        @click="startSafetyValidation(model.id)"
                        :disabled="validating === model.id"
                        class="btn btn-sm bg-purple-600 hover:bg-purple-700 text-white"
                        title="Run Safety Validation"
                    >
                        <Shield class="w-4 h-4" :class="validating === model.id ? 'animate-pulse' : ''" />
                    </button>
                    <button
                        v-if="model.status === 'staged' && model.safety_score >= 0.8"
                        @click="deployModel(model.id)"
                        :disabled="deploying === model.id"
                        class="btn btn-sm btn-primary"
                        title="Deploy to Fleet"
                    >
                        <Rocket class="w-4 h-4" :class="deploying === model.id ? 'animate-bounce' : ''" />
                    </button>
                </div>
            </div>

            <!-- Task Types -->
            <div class="flex flex-wrap gap-2 mb-4">
                <span v-for="task in model.task_types" :key="task"
                      class="text-[10px] font-mono px-2 py-1 rounded bg-[#1f2428] text-gray-300 border border-[#333]">
                    {{ task }}
                </span>
            </div>

            <!-- Metrics -->
            <div class="grid grid-cols-3 gap-4">
                <div class="bg-[#161b22] p-3 rounded border border-[#30363d]">
                    <div class="flex items-center gap-2 mb-1">
                        <Target class="w-3 h-3 text-green-500" />
                        <span class="text-[10px] text-gray-500 uppercase">Success Rate</span>
                    </div>
                    <div class="text-lg font-bold text-green-500">{{ (model.success_rate * 100).toFixed(1) }}%</div>
                </div>
                <div class="bg-[#161b22] p-3 rounded border border-[#30363d]">
                    <div class="flex items-center gap-2 mb-1">
                        <Shield class="w-3 h-3" :class="getSafetyColor(model.safety_score)" />
                        <span class="text-[10px] text-gray-500 uppercase">Safety Score</span>
                    </div>
                    <div class="text-lg font-bold" :class="getSafetyColor(model.safety_score)">
                        {{ model.safety_score ? (model.safety_score * 100).toFixed(1) + '%' : 'N/A' }}
                    </div>
                </div>
                <div class="bg-[#161b22] p-3 rounded border border-[#30363d]">
                    <div class="flex items-center gap-2 mb-1">
                        <Zap class="w-3 h-3 text-orange-500" />
                        <span class="text-[10px] text-gray-500 uppercase">Latency</span>
                    </div>
                    <div class="text-lg font-bold text-orange-500">{{ model.avg_latency_ms?.toFixed(1) || 'N/A' }}ms</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Create Model Modal -->
    <div v-if="showCreateModal" class="fixed inset-0 bg-black/90 flex items-center justify-center z-50 backdrop-blur-sm p-4">
        <div class="bg-[#0f0f0f] border border-primary/30 w-full max-w-2xl rounded-xl shadow-2xl overflow-hidden">
            <div class="p-6 border-b border-[#222] flex items-center gap-3">
                <Bot class="w-6 h-6 text-primary" />
                <div>
                    <h3 class="text-xl font-bold text-white">Register VLA Model</h3>
                    <p class="text-[10px] text-gray-500 uppercase">Add a Vision-Language-Action model to the registry</p>
                </div>
            </div>
            <div class="p-6 space-y-4 max-h-[60vh] overflow-y-auto">
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="text-[10px] text-gray-500 font-bold uppercase mb-2 block">Model Name</label>
                        <input v-model="newModel.name" type="text" class="w-full bg-[#111] border border-[#333] rounded px-4 py-3 text-sm focus:border-primary outline-none" placeholder="e.g. Pi0-Factory-v2">
                    </div>
                    <div>
                        <label class="text-[10px] text-gray-500 font-bold uppercase mb-2 block">Version</label>
                        <input v-model="newModel.version" type="text" class="w-full bg-[#111] border border-[#333] rounded px-4 py-3 text-sm focus:border-primary outline-none" placeholder="1.0.0">
                    </div>
                </div>
                <div>
                    <label class="text-[10px] text-gray-500 font-bold uppercase mb-2 block">Description</label>
                    <textarea v-model="newModel.description" class="w-full bg-[#111] border border-[#333] rounded px-4 py-3 text-sm focus:border-primary outline-none h-20 resize-none" placeholder="Model purpose and capabilities..."></textarea>
                </div>
                <div class="grid grid-cols-3 gap-4">
                    <div>
                        <label class="text-[10px] text-gray-500 font-bold uppercase mb-2 block">Vision Encoder</label>
                        <select v-model="newModel.vision_encoder" class="w-full bg-[#111] border border-[#333] rounded px-4 py-3 text-sm focus:border-primary outline-none cursor-pointer">
                            <option value="ViT-L/14">ViT-L/14</option>
                            <option value="ViT-H/14">ViT-H/14</option>
                            <option value="SigLIP-400M">SigLIP-400M</option>
                            <option value="DINOv2">DINOv2</option>
                        </select>
                    </div>
                    <div>
                        <label class="text-[10px] text-gray-500 font-bold uppercase mb-2 block">Language Model</label>
                        <select v-model="newModel.language_model" class="w-full bg-[#111] border border-[#333] rounded px-4 py-3 text-sm focus:border-primary outline-none cursor-pointer">
                            <option value="Llama-3-8B">Llama-3-8B</option>
                            <option value="Llama-3-70B">Llama-3-70B</option>
                            <option value="Qwen-2.5-7B">Qwen-2.5-7B</option>
                            <option value="PaLM-2">PaLM-2</option>
                        </select>
                    </div>
                    <div>
                        <label class="text-[10px] text-gray-500 font-bold uppercase mb-2 block">Action Head</label>
                        <select v-model="newModel.action_head" class="w-full bg-[#111] border border-[#333] rounded px-4 py-3 text-sm focus:border-primary outline-none cursor-pointer">
                            <option value="Diffusion-Policy">Diffusion-Policy</option>
                            <option value="MLP-Policy">MLP-Policy</option>
                            <option value="ACT">ACT (Action Chunking)</option>
                        </select>
                    </div>
                </div>
                <div>
                    <label class="text-[10px] text-gray-500 font-bold uppercase mb-2 block">Task Types</label>
                    <div class="flex flex-wrap gap-2">
                        <button v-for="task in taskTypeOptions" :key="task"
                                @click="newModel.task_types.includes(task) ? newModel.task_types = newModel.task_types.filter(t => t !== task) : newModel.task_types.push(task)"
                                :class="['text-xs px-3 py-1.5 rounded border transition-colors', newModel.task_types.includes(task) ? 'bg-primary text-black border-primary' : 'bg-[#111] text-gray-400 border-[#333] hover:border-primary/50']">
                            {{ task }}
                        </button>
                    </div>
                </div>
                <div class="grid grid-cols-3 gap-4">
                    <div>
                        <label class="text-[10px] text-gray-500 font-bold uppercase mb-2 block">Action Dimension</label>
                        <input v-model.number="newModel.action_dim" type="number" class="w-full bg-[#111] border border-[#333] rounded px-4 py-3 text-sm focus:border-primary outline-none" min="1" max="30">
                    </div>
                    <div>
                        <label class="text-[10px] text-gray-500 font-bold uppercase mb-2 block">Proprioception Dim</label>
                        <input v-model.number="newModel.proprioception_dim" type="number" class="w-full bg-[#111] border border-[#333] rounded px-4 py-3 text-sm focus:border-primary outline-none" min="1" max="50">
                    </div>
                    <div>
                        <label class="text-[10px] text-gray-500 font-bold uppercase mb-2 block">Action Horizon</label>
                        <input v-model.number="newModel.action_horizon" type="number" class="w-full bg-[#111] border border-[#333] rounded px-4 py-3 text-sm focus:border-primary outline-none" min="1" max="64">
                    </div>
                </div>
            </div>
            <div class="p-6 bg-[#141414] flex justify-end gap-3 border-t border-[#222]">
                <button @click="showCreateModal = false" class="text-xs font-bold text-gray-500 uppercase px-4 py-2 hover:text-white transition-colors">Cancel</button>
                <button @click="createModel" class="btn btn-primary" :disabled="!newModel.name || newModel.task_types.length === 0">
                    <Plus class="w-4 h-4 mr-2" /> Register Model
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
  @apply px-3 py-1.5 text-xs;
}
</style>
