<script setup>
import { ref, onMounted } from 'vue'
import { Settings, Sliders, Shield, Key, Lock, ChevronRight, Save, RotateCcw } from 'lucide-vue-next'

const loading = ref(true)
const saving = ref(false)
const stages = ref([])
const config = ref({})

const fetchConfig = async () => {
    loading.value = true
    try {
        const res = await fetch('/api/v1/pipeline/config')
        const data = await res.json()
        config.value = data.config
        stages.value = data.stages
    } catch (e) {
        console.error("Failed to load pipeline config", e)
    }
    loading.value = false
}

const updateParam = async (key, value) => {
    saving.value = true
    try {
        await fetch('/api/v1/pipeline/config', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key, value: String(value) })
        })
        config.value[key] = value
    } catch (e) {
        console.error("Failed to update config", e)
    }
    saving.value = false
}

const resetConfig = async () => {
    if (!confirm("Reset all pipeline configuration to defaults?")) return
    try {
        await fetch('/api/v1/pipeline/config/reset', { method: 'POST' })
        await fetchConfig()
    } catch (e) {
        console.error("Failed to reset config", e)
    }
}

const getStageIcon = (id) => {
    const icons = { gate: Sliders, privacy: Shield, shield: Lock, kms: Key }
    return icons[id] || Settings
}

const getStageColor = (id) => {
    const colors = { gate: 'text-blue-500', privacy: 'text-green-500', shield: 'text-purple-500', kms: 'text-orange-500' }
    return colors[id] || 'text-gray-500'
}

onMounted(fetchConfig)
</script>

<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="flex items-center justify-between border-b border-[#333] pb-6">
       <div>
         <h2 class="text-2xl font-bold">Privacy Pipeline Configuration</h2>
         <span class="text-xs text-gray-500">Engineer control over all processing stages</span>
       </div>
       <div class="flex gap-3">
           <button @click="resetConfig" class="btn btn-secondary text-sm font-bold uppercase tracking-wider flex items-center gap-2">
              <RotateCcw class="w-4 h-4" /> Reset Defaults
           </button>
           <div v-if="saving" class="flex items-center gap-2 text-primary text-sm">
               <div class="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
               Saving...
           </div>
       </div>
    </div>

    <!-- Pipeline Flow Diagram -->
    <div class="bg-[#111] border border-[#333] rounded-lg p-6">
        <div class="text-xs font-bold text-gray-500 uppercase mb-4">Unified Privacy Pipeline</div>
        <div class="flex items-center justify-between overflow-x-auto">
            <template v-for="(stage, idx) in stages" :key="stage.id">
                <div class="flex flex-col items-center min-w-[140px]">
                    <div :class="['w-12 h-12 rounded-lg flex items-center justify-center', getStageColor(stage.id), 'bg-current/10']">
                        <component :is="getStageIcon(stage.id)" class="w-6 h-6" :class="getStageColor(stage.id)" />
                    </div>
                    <div class="text-sm font-bold text-white mt-2">{{ stage.name.split('(')[0].trim() }}</div>
                    <div class="text-[10px] text-gray-500 text-center">{{ stage.description }}</div>
                </div>
                <ChevronRight v-if="idx < stages.length - 1" class="w-6 h-6 text-gray-600 flex-shrink-0 mx-2" />
            </template>
        </div>
    </div>

    <!-- Stage Configuration Cards -->
    <div v-if="!loading" class="grid grid-cols-2 gap-6">
        <div v-for="stage in stages" :key="stage.id" class="bg-[#111] border border-[#333] rounded-lg p-6 hover:border-primary/50 transition-colors">
            <div class="flex items-center gap-3 mb-4">
                <div :class="['w-10 h-10 rounded-lg flex items-center justify-center', getStageColor(stage.id), 'bg-current/10']">
                    <component :is="getStageIcon(stage.id)" class="w-5 h-5" :class="getStageColor(stage.id)" />
                </div>
                <div>
                    <div class="font-bold text-white">{{ stage.name }}</div>
                    <div class="text-[10px] text-gray-500 uppercase">{{ stage.id.toUpperCase() }} STAGE</div>
                </div>
            </div>
            
            <div class="space-y-4">
                <div v-for="param in stage.parameters" :key="param.key" class="space-y-2">
                    <div class="flex items-center justify-between">
                        <label class="text-xs font-bold text-gray-400">{{ param.label }}</label>
                        <span class="text-xs text-primary font-mono">{{ config[param.key] }}</span>
                    </div>
                    
                    <!-- Slider Control -->
                    <div v-if="param.type === 'slider'" class="flex items-center gap-3">
                        <input 
                            type="range" 
                            :min="param.min" 
                            :max="param.max" 
                            :step="param.step" 
                            :value="config[param.key]"
                            @change="e => updateParam(param.key, e.target.value)"
                            class="flex-1 h-2 bg-[#333] rounded-lg appearance-none cursor-pointer accent-primary"
                        />
                    </div>
                    
                    <!-- Number Input -->
                    <div v-else-if="param.type === 'number'" class="flex items-center gap-2">
                        <input 
                            type="number" 
                            :min="param.min" 
                            :max="param.max" 
                            :value="config[param.key]"
                            @change="e => updateParam(param.key, e.target.value)"
                            class="flex-1 bg-[#0a0a0a] border border-[#333] rounded px-3 py-2 text-white text-sm font-mono focus:border-primary focus:outline-none"
                        />
                    </div>
                    
                    <!-- Select Dropdown -->
                    <div v-else-if="param.type === 'select'">
                        <select 
                            :value="config[param.key]"
                            @change="e => updateParam(param.key, e.target.value)"
                            class="w-full bg-[#0a0a0a] border border-[#333] rounded px-3 py-2 text-white text-sm font-mono focus:border-primary focus:outline-none"
                        >
                            <option v-for="opt in param.options" :key="opt" :value="opt">{{ opt }}</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Loading State -->
    <div v-else class="flex items-center justify-center py-20">
        <div class="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
    </div>
  </div>
</template>

<style scoped>
.btn {
  @apply px-4 py-2 rounded font-medium transition-colors duration-200 flex items-center;
}
.btn-secondary {
  @apply border border-[#30363d] text-gray-300 hover:text-white hover:bg-[#161b22];
}
input[type="range"]::-webkit-slider-thumb {
  @apply w-4 h-4 bg-primary rounded-full cursor-pointer appearance-none;
}
</style>
