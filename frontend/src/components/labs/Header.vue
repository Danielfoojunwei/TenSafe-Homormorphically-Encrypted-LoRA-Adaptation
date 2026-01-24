<script setup>
import { Shield, Printer } from 'lucide-vue-next'
import { ref, onMounted } from 'vue'

const security = ref({
  pqc: true,
  dp: false
})

const currentTime = ref(new Date().toISOString().split('T')[0] + ' ' + new Date().toLocaleTimeString())

// Load settings from backend on mount
onMounted(async () => {
    try {
        const res = await fetch('/api/v1/settings')
        const data = await res.json()
        if (data.pqc_enabled !== undefined) security.value.pqc = data.pqc_enabled === 'true'
        if (data.dp_enabled !== undefined) security.value.dp = data.dp_enabled === 'true'
    } catch (e) { console.warn("Failed to load settings", e) }
})

// Persist toggle state to backend
const persistSetting = async (key, value) => {
    try {
        await fetch('/api/v1/settings', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key, value: String(value) })
        })
    } catch (e) { console.warn("Failed to persist setting", e) }
}

const togglePQC = () => { 
    security.value.pqc = !security.value.pqc
    persistSetting('pqc_enabled', security.value.pqc)
}
const toggleDP = () => { 
    security.value.dp = !security.value.dp
    persistSetting('dp_enabled', security.value.dp)
}
const generateReport = () => { window.print() }
</script>

<template>
  <header class="h-16 border-b border-[#333] bg-[#0d1117] flex items-center justify-between px-6 shrink-0 z-10">
    <!-- Left: Status Indicator -->
    <div class="flex items-center gap-4">
      <div class="flex items-center gap-2">
        <div class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></div>
        <span class="text-xs font-mono text-primary font-bold tracking-wider">ROBUST_MODE_ACTIVE</span>
      </div>
      <span class="text-xs text-gray-500 font-mono opacity-50">{{ currentTime }} UTC</span>
    </div>

    <!-- Right: Security Controls -->
    <div class="flex items-center gap-6 actions">
      <div class="flex items-center gap-3 bg-[#161b22] px-3 py-1.5 rounded-full border border-[#333]">
        <div class="flex items-center gap-2">
            <Shield class="w-4 h-4 text-primary" />
            <span class="text-xs font-bold text-primary">SHIELD</span>
        </div>
        <div class="h-4 w-[1px] bg-[#333]"></div>
        
        <!-- PQC Toggle -->
        <div class="flex items-center gap-2 cursor-pointer" @click="togglePQC" title="Quantum-Resistant Encryption">
            <div class="w-8 h-4 rounded-full relative transition-colors" :class="security.pqc ? 'bg-primary' : 'bg-[#333]'">
                <div class="w-3 h-3 bg-white rounded-full absolute top-0.5 transition-all" :style="security.pqc ? 'left: 18px' : 'left: 2px'"></div>
            </div>
            <span class="text-[10px] font-mono" :class="security.pqc ? 'text-white' : 'text-gray-500'">PQC</span>
        </div>

        <!-- DP Toggle -->
        <div class="flex items-center gap-2 cursor-pointer" @click="toggleDP" title="Differential Privacy">
            <div class="w-8 h-4 rounded-full relative transition-colors" :class="security.dp ? 'bg-primary' : 'bg-[#333]'">
                <div class="w-3 h-3 bg-white rounded-full absolute top-0.5 transition-all" :style="security.dp ? 'left: 18px' : 'left: 2px'"></div>
            </div>
            <span class="text-[10px] font-mono" :class="security.dp ? 'text-white' : 'text-gray-500'">DP</span>
        </div>
      </div>

      <button @click="generateReport" class="flex items-center gap-2 px-3 py-1.5 text-xs font-bold border border-primary text-primary rounded-md hover:bg-primary hover:text-white transition-colors">
        <Printer class="w-4 h-4" />
        REPORT
      </button>

      <div class="flex items-center gap-2">
          <div class="w-8 h-8 rounded-full bg-orange-600 flex items-center justify-center text-white text-xs font-bold">DF</div>
      </div>
    </div>
  </header>
</template>

<style scoped>
.bg-header {
  background-color: #0d1117;
}
</style>
