<script setup>
import { ref, computed, watch } from 'vue'
import { useContinuousStore } from '../../stores/continuous' // Adjusted path
import {
  CheckCircle, XCircle, Clock, Zap, Database, Brain, Package, 
  FileCheck, ArrowUpCircle, Eye, Shield, AlertTriangle, Download
} from 'lucide-vue-next'

const props = defineProps({
  routeKey: { type: String, required: true }
})

const store = useContinuousStore()
const timeline = computed(() => store.timeline[props.routeKey] || [])
const exportModalOpen = ref(false)
const exportSpec = ref(null)
const selectedBackend = ref('k8s')

const stageIcons = {
  // ... (same as before)
  ingest: Database,
  NOVELTY_LOW: Eye,
  UPDATE_PROPOSED: Zap,
  TRAIN_STARTED: Brain,
  TRAIN_DONE: Brain,
  EVAL_DONE: FileCheck,
  PACKAGED: Package,
  REGISTERED: FileCheck,
  PROMOTED: ArrowUpCircle,
  CONSOLIDATED: Package,
  ROLLED_BACK: AlertTriangle,
  FAILED: XCircle,
}

// Simple map for strict event types if needed, or fallback
const getIcon = (type) => stageIcons[type] || Clock

const getColor = (verdict) => {
    if (verdict === 'success') return 'text-green-400 bg-green-500/10 border-green-500/30'
    if (verdict === 'failed') return 'text-red-400 bg-red-500/10 border-red-500/30'
    return 'text-blue-400 bg-blue-500/10 border-blue-500/30'
}

watch(() => props.routeKey, (newKey) => {
  if (newKey) store.fetchTimeline(newKey)
}, { immediate: true })

const openExport = async () => {
    const res = await store.exportRoute(props.routeKey, selectedBackend.value)
    if (res && res.run_spec_json) {
        exportSpec.value = JSON.stringify(res.run_spec_json, null, 2)
        exportModalOpen.value = true
    }
}
</script>

<template>
  <div class="space-y-4">
    <!-- Header with Export -->
    <div class="flex justify-between items-center bg-[#111] p-4 rounded-lg border border-[#222]">
        <h3 class="font-bold text-gray-300">Loop History</h3>
        <button @click="openExport" class="flex items-center gap-2 text-sm text-blue-400 hover:text-blue-300">
            <Download class="w-4 h-4" /> Export Spec
        </button>
    </div>

    <!-- Loop Executions -->
    <div v-for="loop in timeline" :key="loop.loop_id" 
         class="bg-[#111] border border-[#222] rounded-lg overflow-hidden">
      
      <!-- Loop Header -->
      <div class="p-4 border-b border-[#222] flex justify-between items-center">
        <div class="flex items-center gap-3">
          <span :class="loop.verdict === 'success' ? 'text-green-400' : 'text-red-400'">
            <CheckCircle v-if="loop.verdict === 'success'" class="w-5 h-5" />
            <XCircle v-else class="w-5 h-5" />
          </span>
          <div>
            <div class="font-mono text-white">{{ loop.loop_id.slice(0, 8) }}...</div>
            <div class="text-xs text-gray-500">{{ loop.trigger }} • {{ new Date(loop.started_at).toLocaleString() }}</div>
          </div>
        </div>
        <div class="text-right">
          <div class="text-sm text-gray-300">{{ loop.summary || 'No summary' }}</div>
          <div v-if="loop.adapter_produced" class="text-xs text-gray-500">
            Adapter: {{ loop.adapter_produced.slice(0, 8) }}...
          </div>
        </div>
      </div>
      
      <!-- Timeline Events -->
      <div class="p-4 space-y-3">
        <div v-for="(event, idx) in loop.events" :key="idx"
             class="flex items-start gap-4">
          
          <!-- Timeline Connector -->
          <div class="flex flex-col items-center">
            <div class="w-8 h-8 rounded-full flex items-center justify-center border"
                 :class="getColor(event.verdict)">
              <component :is="getIcon(event.stage)" class="w-4 h-4" />
            </div>
            <div v-if="idx < loop.events.length - 1" 
                 class="w-0.5 h-8 bg-[#333] mt-1"></div>
          </div>
          
          <!-- Event Content -->
          <div class="flex-1 pb-2">
            <div class="flex items-center gap-2">
              <span class="font-bold text-white">{{ event.headline }}</span>
              <span v-if="event.privacy_encrypted" 
                    class="bg-purple-500/20 text-purple-400 text-xs px-1.5 py-0.5 rounded">
                <Shield class="w-3 h-3 inline" /> Encrypted
              </span>
              <span v-if="event.duration_ms" class="text-xs text-gray-500">
                {{ event.duration_ms }}ms
              </span>
            </div>
            <div class="text-sm text-gray-400 mt-1">{{ event.explanation }}</div>
             <!-- Payload link (optional) -->
          </div>
        </div>
      </div>
    </div>
    
    <!-- Empty State -->
    <div v-if="timeline.length === 0" 
         class="bg-[#111] border border-[#222] rounded-lg p-8 text-center text-gray-500">
      No loop executions yet. Click "Run Now" to execute the continuous learning loop.
    </div>

    <!-- Export Modal -->
    <div v-if="exportModalOpen" class="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
        <div class="bg-[#111] border border-[#333] rounded-lg p-6 max-w-2xl w-full">
            <h3 class="text-lg font-bold text-white mb-4">Export Continuous Loop Spec</h3>
            <p class="text-sm text-gray-400 mb-4">
                Use this specification to run the continuous loop on external infrastructure.
            </p>
            <div class="bg-[#000] p-4 rounded border border-[#222] font-mono text-xs text-green-400 overflow-auto max-h-96">
                <pre>{{ exportSpec }}</pre>
            </div>
            <div class="mt-6 flex justify-end gap-3">
                <button @click="exportModalOpen = false" class="px-4 py-2 text-gray-400 hover:text-white">Close</button>
                <button class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded"
                        onclick="navigator.clipboard.writeText(document.querySelector('pre').innerText); alert('Copied!')">
                    Copy to Clipboard
                </button>
            </div>
        </div>
    </div>
  </div>
</template>
