<template>
  <div class="evidence-diff p-6 bg-white rounded-lg shadow-sm border border-gray-100">
    <h3 class="text-lg font-bold mb-4 flex items-center gap-2">
      <span class="material-icons text-blue-500">difference</span>
      What Changed?
    </h3>

    <div v-if="loading" class="animate-pulse space-y-3">
      <div class="h-4 bg-gray-200 rounded w-3/4"></div>
      <div class="h-4 bg-gray-200 rounded w-1/2"></div>
    </div>

    <div v-else-if="diffData.diff_available">
      <div class="flex items-center gap-4 text-sm text-gray-500 mb-6">
        <span class="bg-gray-100 px-2 py-1 rounded font-mono">{{ diffData.from }}</span>
        <span class="material-icons text-xs">arrow_forward</span>
        <span class="bg-gray-100 px-2 py-1 rounded font-mono">{{ diffData.to }}</span>
      </div>

      <div class="space-y-4">
        <div v-for="(change, idx) in diffData.changes" :key="idx" 
          class="flex items-start gap-3 p-3 bg-gray-50 rounded-md border border-gray-100 hover:border-gray-300 transition-colors">
          
          <div class="mt-1">
             <span class="material-icons text-blue-500 text-sm" v-if="change.field === 'Primary Metric'">trending_up</span>
             <span class="material-icons text-purple-500 text-sm" v-else-if="change.field === 'Forgetting Score'">memory</span>
             <span class="material-icons text-gray-500 text-sm" v-else>info</span>
          </div>

          <div class="flex-1">
            <div class="flex justify-between">
              <span class="font-medium text-gray-900">{{ change.field }}</span>
              <span class="text-xs font-bold" 
                :class="getDeltaClass(change.delta)">
                {{ change.delta }}
              </span>
            </div>
            <div class="text-sm text-gray-600 mt-1">
              {{ change.summary }}
            </div>
            <div class="flex gap-4 mt-2 text-xs text-gray-400 font-mono">
              <span>Old: {{ change.from }}</span>
              <span>New: {{ change.to }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div v-else class="text-center py-8 text-gray-500">
      <span class="material-icons text-4xl mb-2 block text-gray-300">search_off</span>
      No differences available to display.
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';
import { useContinuousStore } from '../../stores/continuous';

const props = defineProps({
  routeKey: String,
  fromAdapter: String,
  toAdapter: String
});

const diffData = ref({});
const loading = ref(false);

const loadDiff = async () => {
  if (!props.routeKey) return;
  loading.value = true;
  try {
    // Direct fetch or use store
    const res = await fetch(`http://localhost:8000/api/v1/tgflow/routes/${props.routeKey}/diff`);
    diffData.value = await res.json();
  } catch (e) {
    console.error("Failed to load diff", e);
  } finally {
    loading.value = false;
  }
};

const getDeltaClass = (delta) => {
    if (!delta) return 'text-gray-500';
    if (delta.includes('+') && !delta.includes('Forgetting')) return 'text-green-600';
    if (delta.includes('-') && delta.includes('Forgetting')) return 'text-green-600'; // Less forgetting is good? depends on sign convention. Assuming + means more forgetting (bad)
    return 'text-blue-600';
}

onMounted(loadDiff);
watch(() => props.toAdapter, loadDiff);
</script>
