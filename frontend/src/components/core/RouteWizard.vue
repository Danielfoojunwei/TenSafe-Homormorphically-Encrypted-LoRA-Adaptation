<script setup>
import { ref, computed } from 'vue'
import { useContinuousStore } from '../../stores/continuous'; // Corrected path
import { X, ChevronRight, Route, Database, Sliders, Shield, Check } from 'lucide-vue-next'

const emit = defineEmits(['close'])
const store = useContinuousStore()

const step = ref(1)
const steps = ['Route', 'Feed', 'Policy', 'Privacy']

const form = ref({
  route_key: '',
  base_model_ref: 'microsoft/phi-2',
  description: '',
  feed_type: 'hf_dataset',
  feed_uri: '',
  privacy_mode: 'off',
  novelty_threshold: 0.3,
  forgetting_budget: 0.1,
  regression_budget: 0.05,
  update_cadence: 'daily',
  auto_promote_to_canary: false,
})

const feedTypes = [
  { id: 's3', label: 'AWS S3', example: 's3://bucket/path/' },
  { id: 'gcs', label: 'Google Cloud Storage', example: 'gs://bucket/path/' },
  { id: 'azure_blob', label: 'Azure Blob', example: 'https://*.blob.core.windows.net/' },
  { id: 'hf_dataset', label: 'Hugging Face Dataset', example: 'tatsu-lab/alpaca' },
  { id: 'local', label: 'Local Path', example: '/data/my-dataset/' },
]

const nextStep = () => {
  if (step.value < 4) step.value++
}

const prevStep = () => {
  if (step.value > 1) step.value--
}

const createRoute = async () => {
  const success = await store.createRoute(form.value)
  if (success) {
    emit('close')
    store.fetchRoutes()
  }
}

const selectedFeedType = computed(() => feedTypes.find(f => f.id === form.value.feed_type))
</script>

<template>
  <div class="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
    <div class="bg-[#111] border border-[#333] rounded-lg w-full max-w-2xl overflow-hidden">
      
      <!-- Header -->
      <div class="flex items-center justify-between p-4 border-b border-[#333]">
        <div class="flex items-center gap-2">
          <Route class="w-5 h-5 text-orange-500" />
          <h2 class="font-bold text-lg">Create Route</h2>
        </div>
        <button @click="emit('close')" class="text-gray-400 hover:text-white">
          <X class="w-5 h-5" />
        </button>
      </div>
      
      <!-- Step Indicator -->
      <div class="flex border-b border-[#333]">
        <div v-for="(label, idx) in steps" :key="idx"
             class="flex-1 py-3 text-center text-sm font-medium"
             :class="step === idx + 1 ? 'text-orange-400 border-b-2 border-orange-400' : 'text-gray-500'">
          {{ idx + 1 }}. {{ label }}
        </div>
      </div>
      
      <!-- Content -->
      <div class="p-6 min-h-[300px]">
        
        <!-- Step 1: Route -->
        <div v-if="step === 1" class="space-y-4">
          <p class="text-gray-400 text-sm mb-4">
            A Route is the unit of continuous learning. One route = one evolving adapter family.
          </p>
          <div>
            <label class="block text-xs text-gray-500 mb-1">Route Key *</label>
            <input v-model="form.route_key" type="text" 
                   placeholder="e.g., customer-support, finance-qa"
                   class="w-full bg-black border border-[#333] p-3 rounded font-mono focus:border-orange-500 outline-none">
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1">Base Model *</label>
            <input v-model="form.base_model_ref" type="text"
                   placeholder="e.g., microsoft/phi-2"
                   class="w-full bg-black border border-[#333] p-3 rounded font-mono focus:border-orange-500 outline-none">
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1">Description</label>
            <input v-model="form.description" type="text"
                   placeholder="Optional description"
                   class="w-full bg-black border border-[#333] p-3 rounded focus:border-orange-500 outline-none">
          </div>
        </div>
        
        <!-- Step 2: Feed -->
        <div v-if="step === 2" class="space-y-4">
          <p class="text-gray-400 text-sm mb-4">
            Connect a data feed. This is a reference pointer - we don't store your data.
          </p>
          <div>
            <label class="block text-xs text-gray-500 mb-1">Feed Type</label>
            <div class="grid grid-cols-3 gap-2">
              <button v-for="ft in feedTypes" :key="ft.id"
                      @click="form.feed_type = ft.id"
                      class="p-3 border rounded text-left text-sm"
                      :class="form.feed_type === ft.id ? 'border-orange-500 bg-orange-500/10' : 'border-[#333] hover:border-gray-500'">
                {{ ft.label }}
              </button>
            </div>
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1">Feed URI *</label>
            <input v-model="form.feed_uri" type="text"
                   :placeholder="selectedFeedType?.example"
                   class="w-full bg-black border border-[#333] p-3 rounded font-mono focus:border-orange-500 outline-none">
          </div>
        </div>
        
        <!-- Step 3: Policy -->
        <div v-if="step === 3" class="space-y-4">
          <p class="text-gray-400 text-sm mb-4">
            Set stability vs plasticity thresholds. Start with defaults and tune later.
          </p>
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-xs text-gray-500 mb-1">Novelty Threshold</label>
              <input v-model.number="form.novelty_threshold" type="number" step="0.1" min="0" max="1"
                     class="w-full bg-black border border-[#333] p-3 rounded font-mono focus:border-orange-500 outline-none">
              <div class="text-xs text-gray-600 mt-1">Below this = skip training</div>
            </div>
            <div>
              <label class="block text-xs text-gray-500 mb-1">Forgetting Budget</label>
              <input v-model.number="form.forgetting_budget" type="number" step="0.01" min="0" max="1"
                     class="w-full bg-black border border-[#333] p-3 rounded font-mono focus:border-orange-500 outline-none">
              <div class="text-xs text-gray-600 mt-1">Max allowed forgetting</div>
            </div>
            <div>
              <label class="block text-xs text-gray-500 mb-1">Regression Budget</label>
              <input v-model.number="form.regression_budget" type="number" step="0.01" min="0" max="1"
                     class="w-full bg-black border border-[#333] p-3 rounded font-mono focus:border-orange-500 outline-none">
              <div class="text-xs text-gray-600 mt-1">Max regression on held-out</div>
            </div>
            <div>
              <label class="block text-xs text-gray-500 mb-1">Update Cadence</label>
              <select v-model="form.update_cadence"
                      class="w-full bg-black border border-[#333] p-3 rounded focus:border-orange-500 outline-none">
                <option value="manual">Manual</option>
                <option value="hourly">Hourly</option>
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
              </select>
            </div>
          </div>
          <div class="flex items-center gap-2 mt-4">
            <input type="checkbox" v-model="form.auto_promote_to_canary" id="autoCanary"
                   class="w-4 h-4 rounded border-[#333] bg-black">
            <label for="autoCanary" class="text-sm text-gray-300">Auto-promote passing candidates to CANARY</label>
          </div>
        </div>
        
        <!-- Step 4: Privacy -->
        <div v-if="step === 4" class="space-y-4">
          <p class="text-gray-400 text-sm mb-4">
            Privacy Mode controls how routing decisions are computed.
          </p>
          <div class="space-y-3">
            <div @click="form.privacy_mode = 'off'"
                 class="p-4 border rounded cursor-pointer"
                 :class="form.privacy_mode === 'off' ? 'border-orange-500 bg-orange-500/10' : 'border-[#333]'">
              <div class="font-bold flex items-center gap-2">
                <div class="w-4 h-4 rounded-full border-2" 
                     :class="form.privacy_mode === 'off' ? 'bg-orange-500 border-orange-500' : 'border-gray-500'"></div>
                Privacy Off
              </div>
              <div class="text-sm text-gray-400 mt-1">Standard operation. Routing computed in plaintext.</div>
            </div>
            <div @click="form.privacy_mode = 'n2he'"
                 class="p-4 border rounded cursor-pointer"
                 :class="form.privacy_mode === 'n2he' ? 'border-purple-500 bg-purple-500/10' : 'border-[#333]'">
              <div class="font-bold flex items-center gap-2">
                <Shield class="w-4 h-4 text-purple-400" />
                N2HE (Encrypted Routing)
              </div>
              <div class="text-sm text-gray-400 mt-1">
                Adapter selection computed without exposing sensitive request content.
              </div>
              <div class="text-xs text-purple-400 mt-2">Recommended for PII/sensitive data</div>
            </div>
          </div>
        </div>
        
      </div>
      
      <!-- Footer -->
      <div class="flex justify-between p-4 border-t border-[#333]">
        <button v-if="step > 1" @click="prevStep" class="btn btn-ghost">
          Back
        </button>
        <div v-else></div>
        
        <button v-if="step < 4" @click="nextStep" 
                class="btn btn-primary"
                :disabled="step === 1 && !form.route_key">
          Next <ChevronRight class="w-4 h-4 ml-1" />
        </button>
        <button v-else @click="createRoute" class="btn btn-primary">
          <Check class="w-4 h-4 mr-1" /> Create Route
        </button>
      </div>
      
    </div>
  </div>
</template>

<style scoped>
.btn {
  @apply px-4 py-2 rounded font-medium transition-colors flex items-center;
}
.btn-primary {
  @apply bg-orange-600 text-white hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed;
}
.btn-ghost {
  @apply text-gray-400 hover:text-white hover:bg-[#222];
}
</style>
