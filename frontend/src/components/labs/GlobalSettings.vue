<script setup>
import { Settings, Shield, Lock, Globe, Check, Loader2 } from 'lucide-vue-next'
import { ref, onMounted } from 'vue'

const config = ref({
    kms: { provider: 'local', resource_id: '' },
    rtpl: { mode: 'front', profile: 'collaborative' }
})

const saving = ref(false)
const saved = ref(false)
const loading = ref(true)

// Load settings from backend on mount
onMounted(async () => {
    try {
        const res = await fetch('/api/v1/settings')
        const data = await res.json()
        if (data.kms_provider) config.value.kms.provider = data.kms_provider
        if (data.rtpl_mode) config.value.rtpl.mode = data.rtpl_mode
    } catch (e) {
        console.warn("Failed to load settings", e)
    }
    loading.value = false
})

// Save settings to backend
const saveSettings = async () => {
    saving.value = true
    saved.value = false
    try {
        await fetch('/api/v1/settings/bulk', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify([
                { key: 'kms_provider', value: config.value.kms.provider },
                { key: 'rtpl_mode', value: config.value.rtpl.mode }
            ])
        })
        saved.value = true
        setTimeout(() => { saved.value = false }, 2000)
    } catch (e) {
        console.error("Failed to save settings", e)
        alert("Failed to save settings")
    }
    saving.value = false
}
</script>

<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
       <div>
         <h2 class="text-2xl font-bold">Global Settings</h2>
         <span class="text-xs text-gray-500">System Configuration & Policy</span>
       </div>
       <button @click="saveSettings" :disabled="saving" class="btn btn-primary flex items-center gap-2">
          <Loader2 v-if="saving" class="w-4 h-4 animate-spin" />
          <Check v-else-if="saved" class="w-4 h-4" />
          {{ saving ? 'Saving...' : (saved ? 'Saved!' : 'Save Changes') }}
       </button>
    </div>

    <div class="grid grid-cols-2 gap-6">
        <!-- KMS Provider -->
        <div class="bg-[#0d1117] border border-[#30363d] p-6 rounded-lg">
           <div class="flex items-center gap-2 mb-4">
              <Lock class="w-5 h-5 text-gray-400" />
              <h3 class="font-bold">KMS Provider Strategy</h3>
           </div>
           
           <div class="space-y-4">
              <label class="flex items-center gap-3 p-3 border border-[#30363d] rounded cursor-pointer hover:bg-[#161b22]"
                     :class="config.kms.provider === 'local' ? 'border-orange-500 bg-orange-500/5' : ''">
                 <input type="radio" v-model="config.kms.provider" value="local" class="accent-orange-500">
                 <div>
                    <div class="font-bold text-sm">Local Hardware Security Module</div>
                    <div class="text-xs text-gray-500">Use on-prem HSM for master key wrapping</div>
                 </div>
              </label>

              <label class="flex items-center gap-3 p-3 border border-[#30363d] rounded cursor-pointer hover:bg-[#161b22]"
                     :class="config.kms.provider === 'aws' ? 'border-orange-500 bg-orange-500/5' : ''">
                 <input type="radio" v-model="config.kms.provider" value="aws" class="accent-orange-500">
                 <div>
                    <div class="font-bold text-sm">AWS KMS (Cloud Bridge)</div>
                    <div class="text-xs text-gray-500">Delegate to AWS KMS via TEE Enclave</div>
                 </div>
              </label>
           </div>
        </div>

        <!-- RTPL Privacy -->
        <div class="bg-[#0d1117] border border-[#30363d] p-6 rounded-lg">
           <div class="flex items-center gap-2 mb-4">
              <Shield class="w-5 h-5 text-gray-400" />
              <h3 class="font-bold">RTPL Privacy Mode</h3>
           </div>
           
           <select v-model="config.rtpl.mode" class="w-full bg-[#161b22] border border-[#30363d] rounded p-2 text-sm text-white mb-4">
               <option value="front">Front-Running Obfuscation (Fast)</option>
               <option value="onion">Onion Routing (Secure)</option>
               <option value="chaff">Chaff Injection (Extreme)</option>
           </select>

           <div class="p-3 bg-[#1f2428] rounded text-xs text-gray-400 border border-[#30363d]">
               Current mode adds <span class="text-white font-bold">~12ms</span> latency per inference to guarantee <span class="text-white font-bold">k-anonymity=5</span>.
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
</style>
