<script setup>
import { ref, onMounted } from 'vue'
import { usePeftStore } from '../stores/peft'
import { Zap, Play, RotateCcw, CheckCircle, FileJson, Server, Activity, Database, Box } from 'lucide-vue-next'

const store = usePeftStore()
const steps = [
    'Compute Backend', 
    'VLA Model', 
    'Teleop Dataset', 
    'LoRA Config', 
    'Integrations', 
    'Governance', 
    'Launch'
]

const integrations = ref({})
const integrationsError = ref('')

const vlaInput = ref({
    modelId: 'openvla/openvla-7b',
    datasetPath: 's3://tg-teleop-data/pick-place-v2'
})

const fetchIntegrations = async () => {
    integrationsError.value = ''
    try {
        const res = await fetch('/api/v1/integrations/status')
        if (res.ok) {
            integrations.value = await res.json()
        } else {
            throw new Error('Backend unavailable')
        }
    } catch (e) {
        console.warn('Failed to fetch integrations', e)
        integrations.value = {}
        integrationsError.value = 'Unable to load integration status.'
    }
}

const getIntegration = (service) => integrations.value[service] || { status: 'unavailable' }

onMounted(fetchIntegrations)
</script>

<template>
  <div class="h-full flex flex-col">
    <!-- Header -->
    <div class="flex items-center justify-between mb-6">
      <div class="flex items-center gap-2">
        <h2 class="text-2xl font-bold">VLA Adaptation Studio</h2>
        <span class="bg-primary/20 text-primary text-xs px-2 py-0.5 rounded border border-primary/40 font-bold">ROBOTICS ED.</span>
      </div>
      <div class="flex gap-2">
          <button @click="store.applyProfile('local-hf')" class="btn btn-secondary">
            <FileJson class="w-4 h-4 mr-2" />
            Load Profile
          </button>
      </div>
    </div>

    <!-- 2-Column Layout -->
    <div class="flex-1 grid grid-cols-12 gap-6 min-h-0">
      
      <!-- Left: Stepper -->
      <div class="col-span-3 bg-[#0d1117] border border-[#30363d] rounded-lg p-4 overflow-y-auto">
        <div class="space-y-1">
          <div v-for="(label, idx) in steps" :key="idx"
               @click="store.step = idx + 1"
               class="flex items-center gap-3 p-2 rounded cursor-pointer transition-colors"
               :class="store.step === idx + 1 ? 'bg-[#1f2428] text-white' : 'text-gray-500 hover:text-gray-300'"
          >
            <div class="w-6 h-6 rounded flex items-center justify-center text-xs border"
                 :class="store.step > idx + 1 ? 'bg-green-500 border-green-500 text-white' : (store.step === idx + 1 ? 'bg-white text-black border-white' : 'border-gray-600')">
               <span v-if="store.step > idx + 1"><CheckCircle class="w-3 h-3"/></span>
               <span v-else>{{ idx + 1 }}</span>
            </div>
            <span class="font-mono text-sm">{{ label }}</span>
          </div>
        </div>
      </div>

      <!-- Right: Content -->
      <div class="col-span-9 bg-[#0d1117] border border-[#30363d] rounded-lg p-6 overflow-y-auto relative">
        
        <!-- Step 1: Backend -->
        <div v-if="store.step === 1" class="space-y-4">
          <h3 class="text-lg font-bold border-b border-[#333] pb-2">Choose Compute Backend</h3>
          <div class="grid grid-cols-2 gap-4">
            <div class="p-4 border rounded cursor-pointer transition-all relative overflow-hidden group"
                 :class="store.config.backend === 'local-gpu' ? 'border-primary bg-primary/10' : 'border-[#333] hover:border-gray-500'"
                 @click="store.config.backend = 'local-gpu'">
               <div class="flex justify-between items-start mb-2">
                   <div class="font-bold flex items-center gap-2"><Server class="w-4 h-4"/> Local GPU</div>
                   <span class="text-[10px] bg-[#333] px-2 py-0.5 rounded font-mono text-gray-300">AVAILABLE</span>
               </div>
               <div class="text-xs text-gray-400">Run training jobs on locally attached accelerators.</div>
            </div>
            <div class="p-4 border rounded cursor-pointer transition-all relative overflow-hidden group"
                 :class="store.config.backend === 'kubernetes' ? 'border-primary bg-primary/10' : 'border-[#333] hover:border-gray-500'"
                 @click="store.config.backend = 'kubernetes'">
               <div class="flex justify-between items-start mb-2">
                   <div class="font-bold flex items-center gap-2"><Box class="w-4 h-4"/> Kubernetes Cluster</div>
                   <span class="text-[10px] bg-[#333] px-2 py-0.5 rounded font-mono text-gray-300">REQUIRES CONFIG</span>
               </div>
               <div class="text-xs text-gray-400">Submit runs to a secured cluster with GPU scheduling.</div>
            </div>
          </div>
        </div>

        <!-- Step 2: VLA Model (New) -->
        <div v-else-if="store.step === 2" class="space-y-6">
            <h3 class="text-lg font-bold border-b border-[#333] pb-2">VLA Model Source</h3>
            <div class="space-y-4">
                <div>
                    <label class="block text-xs text-gray-500 mb-1">Hugging Face Model ID</label>
                    <div class="flex gap-2">
                        <input type="text" v-model="vlaInput.modelId" class="flex-1 bg-black border border-[#333] p-2 rounded text-sm font-mono focus:border-primary outline-none text-white">
                    </div>
                    <div class="mt-2 text-xs text-gray-500 flex items-center gap-1">
                        <CheckCircle class="w-3 h-3" /> Model validation occurs during run submission.
                    </div>
                </div>
            </div>
        </div>

        <!-- Step 3: Teleop Dataset (New) -->
        <div v-else-if="store.step === 3" class="space-y-6">
            <h3 class="text-lg font-bold border-b border-[#333] pb-2">Teleoperation Data</h3>
            <div class="space-y-4">
                <div>
                    <label class="block text-xs text-gray-500 mb-1">Dataset Path / URL</label>
                    <div class="flex gap-2">
                         <input type="text" v-model="vlaInput.datasetPath" class="flex-1 bg-black border border-[#333] p-2 rounded text-sm font-mono focus:border-primary outline-none text-white">
                         <button class="btn btn-secondary text-xs"><Database class="w-3 h-3"/></button>
                    </div>
                </div>
                <div class="text-xs text-gray-500">
                    Dataset metadata is available after indexing completes.
                </div>
            </div>
        </div>

        <!-- Step 4: Hyperparams (LoRA) -->
        <div v-else-if="store.step === 4" class="space-y-6">
            <h3 class="text-lg font-bold border-b border-[#333] pb-2">Hyperparameters (LoRA)</h3>
            <div class="grid grid-cols-2 gap-6">
                <div>
                    <label class="block text-xs text-gray-500 mb-1">Learning Rate</label>
                    <input type="text" value="0.00005" class="w-full bg-black border border-[#333] p-2 rounded text-sm font-mono focus:border-primary outline-none text-white">
                </div>
                <div>
                    <label class="block text-xs text-gray-500 mb-1">Batch Size</label>
                    <input type="number" value="4" class="w-full bg-black border border-[#333] p-2 rounded text-sm font-mono focus:border-primary outline-none text-white">
                </div>
                <!-- ... other params ... -->
            </div>
        </div>

        <!-- Step 5: Integrations (New) -->
        <div v-else-if="store.step === 5" class="space-y-6">
            <h3 class="text-lg font-bold border-b border-[#333] pb-2">External Integrations</h3>
            <div class="space-y-4">
                <div v-if="integrationsError" class="text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded p-3">
                    {{ integrationsError }}
                </div>
                
                <!-- Isaac Lab -->
                <div class="flex items-center justify-between p-4 border border-[#333] rounded bg-[#111]">
                    <div class="flex items-center gap-4">
                        <div class="w-10 h-10 bg-black rounded flex items-center justify-center border border-[#333]"><Server class="text-green-500" /></div>
                        <div>
                            <div class="font-bold">NVIDIA Isaac Lab</div>
                            <div class="text-xs text-gray-500">Omniverse Nucleus Connection</div>
                        </div>
                    </div>
                    <div>
                         <span class="text-xs font-mono text-gray-300 bg-[#222] px-2 py-1 rounded border border-[#333]">
                             {{ getIntegration('isaac_lab').status?.toUpperCase() || 'UNKNOWN' }}
                         </span>
                    </div>
                </div>

                <!-- ROS2 -->
                <div class="flex items-center justify-between p-4 border border-[#333] rounded bg-[#111] opacity-60">
                    <div class="flex items-center gap-4">
                         <div class="w-10 h-10 bg-black rounded flex items-center justify-center border border-[#333]"><Activity class="text-white" /></div>
                        <div>
                            <div class="font-bold">ROS2 / RobOps</div>
                            <div class="text-xs text-gray-500">Fleet DDS Bridge</div>
                        </div>
                    </div>
                    <div>
                         <span class="text-xs font-mono text-gray-300 bg-[#222] px-2 py-1 rounded border border-[#333]">
                             {{ getIntegration('ros2_bridge').status?.toUpperCase() || 'UNKNOWN' }}
                         </span>
                    </div>
                </div>

                <!-- Formant -->
                <div class="flex items-center justify-between p-4 border border-[#333] rounded bg-[#111]">
                    <div class="flex items-center gap-4">
                         <div class="w-10 h-10 bg-black rounded flex items-center justify-center border border-[#333]"><Activity class="text-blue-500" /></div>
                        <div>
                            <div class="font-bold">Formant.io</div>
                            <div class="text-xs text-gray-500">Telemetry Stream</div>
                        </div>
                    </div>
                    <div>
                         <span class="text-xs font-mono text-gray-300 bg-[#222] px-2 py-1 rounded border border-[#333]">
                             {{ getIntegration('formant').status?.toUpperCase() || 'UNKNOWN' }}
                         </span>
                    </div>
                </div>

            </div>
        </div>

        <!-- Step 6: Review -->
        <div v-else-if="store.step === 6" class="space-y-6">
             <h3 class="text-lg font-bold border-b border-[#333] pb-2">Governance</h3>
             <div class="p-4 border border-[#333] rounded bg-[#111] flex items-center gap-4">
                 <div class="w-5 h-5 border border-primary bg-primary flex items-center justify-center">
                     <CheckCircle class="w-4 h-4 text-black" />
                 </div>
                 <div>
                     <div class="font-bold text-white">Differential Privacy</div>
                     <div class="text-xs text-gray-500">Add DP-SGD noise to gradients (Epsilon=3.0)</div>
                 </div>
             </div>
        </div>

        <!-- Step 7: Launch -->
        <div v-else class="space-y-6">
           <h3 class="text-lg font-bold border-b border-[#333] pb-2">Launch Training Run</h3>
           
           <div class="bg-[#111] p-4 rounded font-mono text-sm text-gray-400 border border-[#333]">
              <div class="flex justify-between border-b border-[#333] pb-1 mb-1">
                 <span>Backend:</span> <span class="text-primary">{{ store.config.backend }}</span>
              </div>
              <div class="flex justify-between border-b border-[#333] pb-1 mb-1">
                 <span>Base Model:</span> <span class="text-white">{{ vlaInput.modelId }}</span>
              </div>
               <div class="flex justify-between border-b border-[#333] pb-1 mb-1">
                 <span>Dataset:</span> <span class="text-white">.../pick-place-v2</span>
              </div>
              <div class="flex justify-between">
                 <span>Privacy:</span> <span class="text-green-500">DP-SGD Enabled</span>
              </div>
           </div>

           <div v-if="store.run.status === 'idle'">
              <button @click="store.startRun" class="btn btn-primary w-full py-3 flex items-center justify-center gap-2">
                 <Play class="w-4 h-4 text-black font-bold" /> <span class="font-bold text-black">START RUN</span>
              </button>
           </div>
           
           <div v-else class="space-y-4">
              <div class="w-full bg-[#111] rounded-full h-1 border border-[#333]">
                <div class="bg-primary h-full rounded-full transition-all duration-200" :style="{ width: store.run.progress + '%' }"></div>
              </div>
              <div class="bg-black p-4 rounded h-48 overflow-y-auto font-mono text-xs text-primary border border-[#333]">
                 <div v-for="(log, i) in store.run.logs" :key="i">{{ log }}</div>
              </div>
              <button v-if="store.run.status === 'completed'" @click="store.step = 1; store.run.status='idle'" class="btn btn-secondary w-full">
                 <RotateCcw class="w-4 h-4 mr-2 inline" /> Start New Run
              </button>
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
